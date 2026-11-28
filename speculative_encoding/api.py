"""High-level API: load a source -> target pair, encode documents into the target's KV
cache, compress it per question, save, and answer.

    from speculative_encoding.api import SpeculativeEncoder

    enc = SpeculativeEncoder.from_pretrained(
        "ckpt/translator", source="qwen3-14b", target="llama31-8b",
        compressors={128: "ckpt/compressor-128x/compressor-step-002400.pt"})
    q = enc.question("Who signed the agreement?")
    doc = enc.encode(open("contract.txt").read(), questions=[q], ratios=[128])
    answers = enc.answer_hf(doc.full, [q])  # Hugging Face, same process
    doc.save("out/", "contract")            # or answer the folder with answer_vllm

``encode`` makes the evaluated pipeline's calls (sample building, source prefill,
translator decode, cache write, per-question compression) in the same order with the
same arguments, so its prefixes are bit-identical to the paper's harness on the same Torch. Saved items use the
same queue format (``engine.json``, ``<name>.pt``, ``<name>.json``).

On one GPU the source model is parked on the CPU outside its prefill; the translator,
the target and the compressors stay on the GPU. ``release()`` frees all of it before
vLLM takes the GPU; ``answer_vllm`` runs vLLM in a subprocess."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from speculative_encoding import settings
from speculative_encoding import encode as E
from speculative_encoding import models as se_models
from speculative_encoding import prompts as se_prompts
from speculative_encoding.checkpoints import load_checkpoint
from speculative_encoding.encode import EncodedMemory
from speculative_encoding.models import CATALOG, HostRows, generation_config, source_rows_host
from speculative_encoding.translator import compressor as C

SELECTION_SCORE = ("kvzip.score: per layer max over K/V heads, normalized to sum 1 over "
                   "the document, mean over layers")


@dataclass
class Encoded:
    """One served prefix: a document's full translated cache, or one question's
    compressed cache."""

    target: str
    arm: str  # "full", or "c16" / "c64" / "c128"
    # CPU tensor [layers, 2, tokens, kv_heads, head_dim], or a dict for hybrid targets
    prefix: Any
    # header + document ids (full), or header + kept slot ids (compressed)
    prefix_ids: list[int]
    header_tokens: int
    doc_tokens: int
    engine: dict  # the target's engine.json: model, revision, eos ids, vLLM options
    questions: list[dict] = field(default_factory=list)  # compressed: exactly one
    meta: dict = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return E.prefix_bytes(self.prefix)

    @classmethod
    def load(cls, folder, name: str, *, map_location="cpu") -> "Encoded":
        """Read back an item written by :meth:`save` (``EncodedDocument.save``'s full
        cache is ``<name>-full``, read for ``name`` when there is no ``<name>``)."""

        folder = Path(folder)
        if not (folder / f"{name}.json").exists() and (folder / f"{name}-full.json").exists():
            name = f"{name}-full"
        meta = json.loads((folder / f"{name}.json").read_text())
        engine = json.loads((folder / "engine.json").read_text())
        prefix = torch.load(folder / f"{name}.pt", map_location=map_location, weights_only=False)
        target = meta.get("target") or next(
            (key for key, spec in CATALOG.items()
             if (spec.hf_id, spec.revision) == (engine["model"], engine["revision"])),
            engine["model"])
        keys = ("name", "arm", "questions", "prefix_ids", "header_tokens", "doc_tokens",
                "prefix_bytes", "max_tokens", "no_native", "target")
        return cls(
            target=target, arm=meta["arm"], prefix=prefix, prefix_ids=meta["prefix_ids"],
            header_tokens=meta["header_tokens"], doc_tokens=meta["doc_tokens"], engine=engine,
            questions=meta.get("questions", []),
            meta={key: value for key, value in meta.items() if key not in keys})

    @classmethod
    def load_folder(cls, folder, *, map_location="cpu") -> dict[str, "Encoded"]:
        """Every item saved in ``folder``: ``{name: Encoded}``."""

        folder = Path(folder)
        names = sorted(path.stem for path in folder.glob("*.json")
                       if path.name != "engine.json" and path.with_suffix(".pt").exists())
        return {name: cls.load(folder, name, map_location=map_location) for name in names}

    def save(self, folder, name: str) -> Path:
        """Write ``<folder>/<name>.pt``, ``<name>.json`` and ``engine.json``."""

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "engine.json").write_text(json.dumps(self.engine))
        # A clone owns its storage: the prefix may be a view of a reused pinned buffer.
        prefix = self.prefix.clone() if torch.is_tensor(self.prefix) else self.prefix
        torch.save(prefix, folder / f"{name}.pt.tmp")
        os.replace(folder / f"{name}.pt.tmp", folder / f"{name}.pt")
        # Every question carries its own max_tokens (free-form ones may want another
        # length); the item-level max_tokens, the engines' fallback, is the longest.
        questions = []
        for question in self.questions:
            if question.get("metric") != "nll" and "max_tokens" not in question:
                limit = se_prompts.generation_tokens(question.get("metric"))
                question = {**question, "max_tokens": limit}
            questions.append(question)
        limits = [int(q["max_tokens"]) for q in questions if q.get("max_tokens") is not None]
        meta = {**self.meta, "name": name, "target": self.target, "arm": self.arm,
                "questions": questions, "prefix_ids": self.prefix_ids,
                "header_tokens": self.header_tokens, "doc_tokens": self.doc_tokens,
                "prefix_bytes": self.bytes,
                "max_tokens": max(limits) if limits else se_prompts.GENERATION_TOKENS["mcq"]}
        if self.arm != "full" and not self.meta.get("native_prefix_ids"):
            meta["no_native"] = True
        (folder / f"{name}.json").write_text(json.dumps(meta))
        (folder / f"{name}.ready").touch()
        return folder / f"{name}.pt"


@dataclass
class EncodedDocument:
    full: Encoded
    compressed: dict = field(default_factory=dict)  # {(question row_id, ratio): Encoded}

    def save(self, folder, name: str) -> list[Path]:
        paths = [self.full.save(folder, f"{name}-full")]
        keys = sorted(self.compressed, key=lambda key: (str(key[0]), key[1]))
        for number, (row_id, ratio) in enumerate(keys):
            item = self.compressed[(row_id, ratio)]
            paths.append(item.save(folder, f"{name}-q{number:03d}-c{ratio:g}"))
        return paths


def _load_model(key: str, device, models: dict):
    """``(interface, tokenizer)``: ``models[key]`` moved to ``device`` (as the catalog
    loader places a model), else the catalog model."""

    if key in models:
        interface, tokenizer = _injected(models, key)
        interface.model.to(device)
        return interface, tokenizer
    _, interface, tokenizer = se_models.load_catalog(key, device)
    return interface, tokenizer


def _injected(models: dict, key: str):
    pair = models[key]
    if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
        raise ValueError(f"models[{key!r}]: pass a (ModelInterface, tokenizer) pair, "
                         f"not {type(pair).__name__}")
    return pair


def _load_shared(encoder, compressors, load_source: bool, models: dict) -> None:
    """The parts of ``encoder`` every target shares: compressors, executor, the parked
    source model and the serving config (``translator`` is already on its device)."""

    device, translator = encoder.device, encoder.translator
    encoder.compressors = {}
    for ratio, path in (compressors or {}).items():
        config = Path(path).with_name("config.json")
        blocks = int(json.loads(config.read_text()).get("blocks", 2)) if config.exists() else 2
        module = C.SlotCompressor(translator.decoder.blocks, blocks=blocks).to(device)
        module.load_state_dict(torch.load(path, map_location=device))
        encoder.compressors[float(ratio)] = module.eval().requires_grad_(False)
    # After the compressors copied the decoder blocks, which keep the eager path.
    # Compiled attention (torch.compile) on CUDA only.
    E.set_executor(translator, encoder.level, device.type == "cuda")
    encoder.source = encoder.source_tokenizer = None
    # Only a source this encoder loaded is parked on the CPU between prefills: an injected
    # one may be another encoder's resident target (a chain A -> B -> C shares B).
    encoder.park_source = encoder.source_key not in models
    if load_source:
        encoder.source, encoder.source_tokenizer = _load_model(encoder.source_key, device, models)
        if encoder.park_source:
            encoder.source.model.to("cpu")
            settings.empty_cache(device)
    elif encoder.source_key in models:  # its tokenizer still renders documents and ids
        encoder.source_tokenizer = _injected(models, encoder.source_key)[1]
    encoder.config = {**se_prompts.arm_config(se_prompts.SERVING_ARM), "max_doc_tokens": 10**7}


def _first_question(questions) -> str:
    # The question only shapes after_ids, which the document's prefix does not use.
    return questions[0].get("text", "") if questions else ""


class SpeculativeEncoder:
    """One source -> target direction of the translator, with optional compressors.

    ``device``: ``"cuda"`` runs the paper path (bf16 autocast, fast-inference ``level``
    6 by default, compiled attention); ``"cpu"`` and ``"mps"`` run the eager reference
    path in the translator's own dtype (``level`` 0, the only level off CUDA).
    ``models``: ``{model key: (ModelInterface, tokenizer)}`` to use instead of catalog
    checkpoints (e.g. tiny random models, ``examples/quickstart_cpu.py``).
    ``load_source=False``: never load the source model; ``encode`` then needs
    ``source_rows`` or ``memory``."""

    def __init__(self, translator, source: str, target: str, *, compressors=None,
                 device="cuda", level: int | None = None, chunk: int = 32768,
                 write_chunk: int = 16384, load_source: bool = True, models: dict | None = None):
        self.device, self.source_key, self.target_key = torch.device(device), source, target
        self.level = settings.resolve_level(level, self.device)
        self.chunk, self.write_chunk = int(chunk), int(write_chunk)
        self.translator = translator.to(self.device).eval().requires_grad_(False)
        models = dict(models or {})
        self.target, self.target_tokenizer = _load_model(target, self.device, models)
        self.reader, _ = E.lean_target(self.target, self.device)
        _load_shared(self, compressors, load_source, models)
        self._bind_target()

    @classmethod
    def _shared(cls, owner, target: str, *, load_target: bool = True) -> "SpeculativeEncoder":
        """A view for ``target`` over ``owner``'s translator, compressors and source (no
        copy). ``load_target=False``: the target's tokenizer only, enough for
        ``source_rows`` and ``encode_memory``, which never touch the target model."""

        self = cls.__new__(cls)
        for name in ("device", "source_key", "level", "chunk", "write_chunk", "translator",
                     "compressors", "source", "source_tokenizer", "park_source", "config"):
            setattr(self, name, getattr(owner, name))
        self.target_key = target
        if load_target:
            self.target, self.target_tokenizer = _load_model(target, self.device, owner.models)
            self.reader, _ = E.lean_target(self.target, self.device)
        elif target in owner.models:
            self.target, self.target_tokenizer = None, _injected(owner.models, target)[1]
        else:
            from speculative_encoding.models import load_tokenizer

            self.target = None
            self.target_tokenizer = load_tokenizer(target, token=os.environ.get("HF_TOKEN"))
        self._bind_target()
        return self

    def _bind_target(self) -> None:
        """Writer, engine.json and translator direction for ``self.target``."""

        key, target = self.target_key, self.target
        if target is None:
            self.reader = self.writer = self.engine = self._fast_writer = None
        else:
            self.hybrid = not target.all_full_attention
            self.writer = E.HybridWriter(target, self.device) if self.hybrid else None
            self._fast_writer = None
            spec = CATALOG.get(key)  # None: an injected, non-catalog model
            options = (
                E.engine_options(target, target, key, key).get("target", {})
                if spec else {})
            self.engine = {
                "model": spec.hf_id if spec else getattr(target.model, "name_or_path", None) or key,
                "revision": spec.revision if spec else None,
                "eos_token_ids": generation_config(target.model, 1)["eos_token_ids"],
                "options": options, "hybrid": self.hybrid,
                "linear": any(kind == "linear_attention" for kind in target.layer_types),
                "source": None}
        self.direction = E.Direction(
            self.translator, self.source_key, key, self.level, False, 0)
        if self.device.type != "cuda":
            # No bf16 autocast: source rows enter the translator in its own dtype.
            self.direction.rows_dtype = next(self.translator.parameters()).dtype

    @classmethod
    def from_pretrained(cls, checkpoint, source: str, target: str, *, compressors=None,
                        dtype=None, **kwargs):
        """``checkpoint``: any translator checkpoint folder (``checkpoints.load_checkpoint``
        detects the format); only this pair's weights are loaded. To download a pair from
        the paper's Hugging Face repo, see ``checkpoints.resolve``."""

        # The compressor reads each question through the target's encoder-side input
        # maps (compressor.question_rows), so compression needs those weights too.
        sources = [source] + ([target] if compressors and target != source else [])
        translator = load_checkpoint(checkpoint, sources=sources, targets=[target], dtype=dtype)
        return cls(translator, source, target, compressors=compressors, **kwargs)

    @classmethod
    def from_pretrained_pairs(cls, checkpoint, pairs, *, compressors=None, dtype=None,
                              device="cuda", models: dict | None = None,
                              **kwargs) -> dict[tuple[str, str], "SpeculativeEncoder"]:
        """Several directions over one translator and one model pool, e.g. a relay
        A -> B -> C: ``{(source, target): SpeculativeEncoder}`` for ``pairs``. The
        checkpoint is read once (every pair's sources and targets) and each model is loaded
        once and stays on ``device`` (none is parked: in a chain a source is another pair's
        target). ``models``: injected models, as in ``SpeculativeEncoder``."""

        pairs = [tuple(pair) for pair in pairs]
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("pairs: a list of (source, target)")
        targets = sorted({target for _, target in pairs})
        sources = sorted({source for source, _ in pairs} | (set(targets) if compressors else set()))
        translator = load_checkpoint(checkpoint, sources=sources, targets=targets, dtype=dtype)
        pool = dict(models or {})
        for key in sorted({key for pair in pairs for key in pair} - set(pool)):
            _, interface, tokenizer = se_models.load_catalog(key, torch.device(device))
            pool[key] = (interface, tokenizer)
        return {(source, target): cls(translator, source, target, compressors=compressors,
                                      device=device, models=pool, **kwargs)
                for source, target in pairs}

    def question(self, text: str, *, metric: str = "f1", row_id: str | None = None,
                 max_tokens: int | None = None, sampling=None, **row) -> se_prompts.Question:
        """``prompts.render_question`` for this target: a question in the form the serving
        engines take (its ``after_ids`` follow the document's prefix)."""

        return se_prompts.render_question(
            self.target_tokenizer, text, metric=metric, row_id=row_id, max_tokens=max_tokens,
            sampling=sampling, model=self.target_key, **row)

    def _source_tokenizer(self):
        if self.source_tokenizer is None:  # load_source=False: the tokenizer alone is enough
            from speculative_encoding.models import load_tokenizer

            self.source_tokenizer = load_tokenizer(self.source_key,
                                                   token=os.environ.get("HF_TOKEN"))
        return self.source_tokenizer

    def _sample(self, document: str, questions=()):
        row = se_prompts.chat_row("api", "document", document, question=_first_question(questions))
        sample = se_prompts.make_sample(row, self._source_tokenizer(), self.target_tokenizer,
                                        kind="task", config=self.config)
        if sample is None:
            raise ValueError("empty document after tokenization")
        sample.source_key = self.source_key
        return sample

    def _host_rows(self, sample, source_rows) -> HostRows:
        """``source_rows`` (from ``source_rows()``) as ``HostRows``; ``None``: run the
        source prefill now."""

        if source_rows is None:
            if self.source is None:
                raise ValueError("this encoder was built with load_source=False: "
                                 "pass source_rows or memory")
            self.source.model.to(self.device)
            rows = source_rows_host(self.source, sample.source.header_ids, sample.source.doc_ids,
                                    self.device)
            if self.park_source:
                self.source.model.to("cpu")
                settings.empty_cache(self.device)
            return rows
        tokens = len(sample.source.doc_ids)
        if source_rows.shape[0] != tokens:
            raise ValueError(f"source_rows has {source_rows.shape[0]} tokens; "
                             f"the document has {tokens} source tokens")
        # Back to the stream-major host layout that the translator streams in chunks.
        return HostRows(source_rows.to("cpu").transpose(0, 1).contiguous(), 0, self.device)

    @torch.no_grad()
    def source_rows(self, document: str) -> torch.Tensor:
        """The source model's hidden states of ``document`` (``[doc tokens, streams,
        width]``, on the CPU): compute once, ``torch.save``, and pass to
        ``encode(..., source_rows=...)`` later, so encoding needs no source model."""

        # Own storage: source_rows_host returns a view of a reused pinned buffer.
        return self._host_rows(self._sample(document), None).to("cpu").clone()

    @torch.no_grad()
    def encode_memory(self, document: str, *, source_rows: torch.Tensor | None = None
                      ) -> EncodedMemory:
        """The translator encoder's output for ``document`` (on the CPU). It depends only
        on the source, so one memory serves every target: save it
        (``.save(path, dtype=torch.bfloat16)``) and pass it to ``encode(memory=...)`` of
        any encoder with this source, which then skips the source prefill and encoder."""

        sample = self._sample(document)
        # The text make_sample rendered.
        text = se_prompts.truncate_to_tokens(self._source_tokenizer(), sample.row["document"],
                                             self.config["max_doc_tokens"])
        rows = self._host_rows(sample, source_rows)
        memory = E.build(self.direction, sample, text, rows, self.chunk, self.device)
        del rows
        settings.empty_cache(self.device)
        return memory

    @torch.no_grad()
    def encode(self, document: str | None = None, *, questions=(), ratios=(),
               source_rows: torch.Tensor | None = None, memory: EncodedMemory | None = None,
               return_scores: bool | int = False) -> EncodedDocument:
        """The full translated cache of ``document``, and one compressed cache per
        (question, ratio).

        ``source_rows``: precomputed ``source_rows(document)``; no source model needed.
        ``memory``: a stored ``encode_memory(document)`` of this source, from any encoder
        with this source: no source prefill and no encoder; ``document`` may be omitted.
        ``return_scores``: each compressed item's meta also gets the scores its slots were
        selected by, one per target document token: ``True`` the full vector
        (``selection_scores``), an int ``k`` the top ``k`` (``selection_top``). Slots and
        prefixes are the same either way."""

        if return_scores is not False and not (
                return_scores is True or (isinstance(return_scores, int) and return_scores > 0)):
            raise ValueError("return_scores: False, True or a positive int (top k)")
        questions = [se_prompts.question_dict(question) for question in questions]
        self._check_questions(questions, ratios)
        if memory is None:
            sample = self._sample(document, questions)
            rows = self._host_rows(sample, source_rows)
        else:
            if source_rows is not None:
                raise ValueError("pass source_rows or memory, not both")
            memory.check(self.translator, self.source_key)
            if document is not None and not str(document).strip().startswith(memory.document):
                raise ValueError("document differs from the memory's document")
            sample = memory.sample(self.target_tokenizer, self.config,
                                   question=_first_question(questions))
            rows = None
        offset = 0 if document is None else len(document) - len(str(document).lstrip())
        return self._encode_sample(sample, rows, questions, ratios, memory, return_scores,
                                   char_offset=offset)

    @torch.no_grad()
    def encode_ids(self, ids, *, text: str | None = None, spans=None, questions=(), ratios=(),
                   source_rows: torch.Tensor | None = None,
                   return_scores: bool | int = False) -> EncodedDocument:
        """``encode`` for a document given as SOURCE token ids (e.g. the source model's own
        generation) instead of text. Special ids (stop tokens, chat markers) are dropped
        together with their ``spans`` and ``source_rows``: they have no text to align.

        The target tokenizes the text the ids decode to, aligned by character spans.
        ``text``/``spans`` default to ``handoff.token_spans(source tokenizer, ids)``; when
        ``ids`` are exactly the tokenization of a stripped ``text``, the result equals
        ``encode(text)``'s.

        ``source_rows``: the rows of ``ids`` (``handoff.capture_rows``); default: the source
        prefills them as ``encode`` does. Rows captured after other context (a translated
        cache, a question) are not the document-prefill rows the translator was trained
        on: the plumbing is exact, their translation quality is unmeasured."""

        from speculative_encoding import handoff

        questions = [se_prompts.question_dict(question) for question in questions]
        self._check_questions(questions, ratios)
        tokenizer = self._source_tokenizer()
        ids, given = [int(token) for token in ids], len(ids)
        special = set(getattr(tokenizer, "all_special_ids", None) or [])
        keep = [index for index, token in enumerate(ids) if token not in special]
        if len(keep) < given:
            ids = [ids[index] for index in keep]
            spans = None if spans is None else [list(spans)[index] for index in keep]
            if source_rows is not None and source_rows.shape[0] == given:
                # Each kept row is still the row captured for its token, special ids before
                # it included; dropping rows never changes the others.
                source_rows = source_rows[torch.tensor(keep, device=source_rows.device)]
        sample = handoff.sample_from_ids(tokenizer, self.target_tokenizer, ids, self.config,
                                         text=text, spans=spans,
                                         question=_first_question(questions))
        sample.source_key = self.source_key
        rows = self._host_rows(sample, source_rows)
        return self._encode_sample(sample, rows, questions, ratios, None, return_scores)

    def _check_questions(self, questions, ratios=()) -> None:
        """Questions carry ``after_ids`` in one target's tokens, and compressed items are
        keyed by ``row_id``: a mismatch would be silently wrong."""

        questions = [question for question in questions if isinstance(question, dict)]
        for question in questions:
            if question.get("target", self.target_key) != self.target_key:
                raise ValueError(f"question {question.get('row_id')!r} was rendered for "
                                 f"{question['target']}, not {self.target_key}: render it with "
                                 "this target's question()")
        row_ids = [question["row_id"] for question in questions]
        if ratios and len(set(row_ids)) != len(row_ids):
            raise ValueError(f"compressed items are keyed by row_id and {row_ids} repeat one: "
                             "pass a distinct row_id to question()")

    def _encode_sample(self, sample, source_rows, questions, ratios, memory,
                       return_scores=False, char_offset=0) -> EncodedDocument:
        header, device, target = list(sample.target.header_ids), self.device, self.target
        doc = se_prompts.document_tensors(sample, source_rows, self.reader, device)
        del source_rows
        try:
            states = E.decode_states(self.direction, doc, self.chunk,
                                     memory=None if memory is None else memory.memory)
        except torch.OutOfMemoryError as error:
            raise torch.OutOfMemoryError(
                f"translating {len(sample.target.doc_ids)} target tokens in chunks of "
                f"{self.chunk} ran out of GPU memory; a smaller chunk (e.g. chunk="
                f"{max(1024, self.chunk // 2)}) needs less memory and gives identical output"
            ) from error
        header_cache = target.native_prefix(torch.tensor(header, device=device))
        doc_ids = torch.tensor(sample.target.doc_ids, device=device)
        head, cache = self.translator.head(self.target_key), header_cache
        total = len(header) + states.shape[0]

        def chunks():
            for first in range(0, states.shape[0], self.write_chunk):
                last = min(states.shape[0], first + self.write_chunk)
                with settings.autocast(device):
                    hidden = head(states[first:last])
                yield first, last, doc.target_positions[first:last].to(device), hidden

        if not ratios and settings.current().fast_writer and E.fast_writer_supported(target):
            # Only the served prefix is needed (compressors read the HF cache): K/V and
            # gated-delta states written directly, bitwise equal to write_prefix's.
            if self._fast_writer is None:
                self._fast_writer = self.writer or E.HybridWriter(target, device)
            full = E.write_prefix_fast(
                self._fast_writer, header_cache,
                ((len(header) + first, len(header) + last, positions, hidden.to(target.dtype))
                 for first, last, positions, hidden in chunks()),
                total)
            cache = None
        else:
            for first, last, positions, hidden in chunks():
                cache, _ = target.write_prefix(doc_ids[first:last], hidden.to(target.dtype),
                                               cache, positions)
                del hidden
            full = E.prefix_from_cache(target, cache, total)
        doc_states = full if self.hybrid else None
        if self.hybrid:
            full_prefix = full
        else:
            full_prefix = torch.stack([full["kv"][layer] for layer in sorted(full["kv"])])
        common = dict(target=self.target_key, header_tokens=len(header),
                      doc_tokens=len(sample.target.doc_ids), engine=self.engine)
        meta = {"source": self.source_key, "source_doc_tokens": len(sample.source.doc_ids)}
        result = EncodedDocument(Encoded(
            arm="full", prefix=E.prefix_to(full_prefix, "cpu"),
            prefix_ids=header + list(sample.target.doc_ids), questions=list(questions),
            meta=meta, **common))
        # Character spans of the target document tokens, in the caller's string (the
        # document is encoded stripped).
        spans = [[int(a) + char_offset, int(b) + char_offset] for a, b in sample.target.doc_spans]
        native_ids = header + list(sample.target.doc_ids)
        for question in questions:
            for ratio in map(float, ratios):
                scores = {} if return_scores is not False else None
                prefix, slots, timings = E.compressed_prefix(
                    target, self.translator, self.compressors[ratio],
                    SimpleNamespace(target=self.target_key, compressor_ratio=ratio), states, doc,
                    sample, question["after_ids"], header_cache, cache, doc_states, self.writer,
                    scores_out=scores)
                kept = slots.tolist()
                # What the compressor kept: target document token indices and their
                # character spans in the text; the whole document's ids for the native control.
                item_meta = {**meta, "slots": int(slots.numel()), "compressor_ratio": ratio,
                             "timings": timings,
                             "kept_positions": [int(i) for i in kept],
                             "kept_char_spans": [spans[i] for i in kept],
                             "native_prefix_ids": native_ids}
                if scores is not None:
                    pooled = scores["pooled"]
                    item_meta["selection_score"] = SELECTION_SCORE
                    if return_scores is True:
                        item_meta["selection_scores"] = pooled.tolist()
                        item_meta["doc_char_spans"] = spans
                    else:
                        top = pooled.topk(min(int(return_scores), pooled.numel()))
                        item_meta["selection_top"] = {"positions": top.indices.tolist(),
                                                      "scores": top.values.tolist()}
                result.compressed[(question["row_id"], ratio)] = Encoded(
                    arm=f"c{ratio:g}", prefix=E.prefix_to(prefix, "cpu"), questions=[question],
                    prefix_ids=header + [int(sample.target.doc_ids[i]) for i in kept],
                    meta=item_meta, **common)
                del prefix
        del states, cache, header_cache, doc, full, doc_states, full_prefix
        settings.empty_cache(device)
        return result

    def answer_hf(self, encoded: Encoded, questions=None, *, native: bool = False,
                  **kwargs) -> list[dict]:
        """Answers from the Hugging Face engine in this process (no vLLM). ``native=True``:
        the control, the target reading the document itself (a compressed item's whole
        document) instead of the translated cache."""

        from speculative_encoding.serving import hf_engine

        if encoded.target != self.target_key:
            raise ValueError(f"this item is {encoded.target}'s cache; this encoder serves "
                             f"{self.target_key}")
        questions = [se_prompts.question_dict(q) for q in (questions or encoded.questions)]
        self._check_questions(questions)
        if native:
            ids = encoded.meta.get("native_prefix_ids") or encoded.prefix_ids
            return hf_engine.answer_batch(self.target, self.target_tokenizer, None, ids,
                                          questions, **kwargs)
        prefix = E.prefix_to(encoded.prefix, self.device)
        return hf_engine.answer_batch(self.target, self.target_tokenizer, prefix,
                                      encoded.prefix_ids, questions, **kwargs)

    def release(self) -> None:
        """Free the GPU (translator, target, source, compressors), e.g. for vLLM."""

        for name in ("translator", "target", "source", "reader", "writer", "direction",
                     "compressors"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        settings.empty_cache(self.device)


class MultiTargetEncoder:
    """Fan-out: one source read, many targets. The translator (the source's weights and
    every target's head), the compressors and the source model are loaded once; each
    document's source prefill (``share="rows"``) or source prefill + translator encoder
    (``share="memory"``) runs once, and every target decodes from it.

        enc = MultiTargetEncoder.from_pretrained(
            "ckpt/translator", source="qwen3-14b",
            targets=["llama31-8b", "gemma3-12b", "ministral3-8b"],
            compressors={128: ".../compressor-128x/compressor-step-002400.pt"},
            offload_targets=True)
        docs = enc.encode(text, questions=["Who signed?"], ratios=[128])  # {target: doc}

    Per target, the encode is ``SpeculativeEncoder.encode`` on a view sharing this
    object's translator, compressors and source. ``share="rows"`` gives exactly what
    separate ``SpeculativeEncoder(source, target).encode(document)`` calls give;
    ``share="memory"`` also skips the encoder per target, bit-exact to it for documents
    within one ``chunk``.

    Targets stay resident on the GPU by default; ``offload_targets=True`` loads one at a
    time (load, encode every pending document, free). The pending documents' shared
    rows then wait on the CPU (``[source tokens, layers + 1, source width]`` each,
    bf16); ``share="memory"`` keeps the encoder output instead, 5-7x smaller.

    ``models``: injected ``(ModelInterface, tokenizer)`` pairs, as for
    ``SpeculativeEncoder``.

    Questions are given as text (or ``{"text": ..., "row_id": ..., "metric": ...}``)
    and rendered per target, since each tokenizer gives its own ``after_ids``; give
    ``row_id`` to key compressed items the same way across targets (default ``q000``,
    ``q001``, ... in order)."""

    def __init__(self, translator, source: str, targets, *, compressors=None, device="cuda",
                 level: int | None = None, chunk: int = 32768, write_chunk: int = 16384,
                 load_source: bool = True, offload_targets: bool = False,
                 models: dict | None = None):
        targets = [targets] if isinstance(targets, str) else list(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError(f"targets: one or more distinct model keys, got {targets}")
        self.device, self.source_key, self.targets = torch.device(device), source, targets
        self.models = dict(models or {})
        self.level = settings.resolve_level(level, self.device)
        self.chunk, self.write_chunk = int(chunk), int(write_chunk)
        self.offload_targets = bool(offload_targets)
        self.translator = translator.to(self.device).eval().requires_grad_(False)
        _load_shared(self, compressors, load_source, self.models)
        # The source side (prefill, encoder memory) never touches a target model.
        self._source_view = SpeculativeEncoder._shared(self, targets[0], load_target=False)
        self.encoders: dict[str, SpeculativeEncoder] = {}
        if not self.offload_targets:
            for target in targets:
                self.encoders[target] = SpeculativeEncoder._shared(self, target)

    @classmethod
    def from_pretrained(cls, checkpoint, source: str, targets, *, compressors=None, dtype=None,
                        **kwargs):
        """``checkpoint``: any translator checkpoint folder; only the source's and these
        targets' weights are loaded (with compressors, also the targets' encoder-side
        weights, which ``compressor.question_rows`` reads each question through)."""

        targets = [targets] if isinstance(targets, str) else list(targets)
        sources = [source] + ([t for t in targets if t != source] if compressors else [])
        translator = load_checkpoint(checkpoint, sources=sources, targets=targets, dtype=dtype)
        return cls(translator, source, targets, compressors=compressors, **kwargs)

    def encoder(self, target: str) -> SpeculativeEncoder:
        """The ``SpeculativeEncoder`` view of one target, for ``question``, ``answer_hf``
        and the rest of the single-pair API (loads it if offloaded; ``free`` drops it)."""

        if target not in self.encoders:
            self.encoders[target] = SpeculativeEncoder._shared(self, target)
        return self.encoders[target]

    def free(self, target: str) -> None:
        view = self.encoders.pop(target, None)
        if view is None:
            return
        for name in ("target", "reader", "writer", "direction", "target_tokenizer"):
            setattr(view, name, None)
        del view
        gc.collect()
        settings.empty_cache(self.device)

    @torch.no_grad()
    def shared_input(self, document: str, share: str = "rows"):
        """What every target decodes ``document`` from, on the CPU: ``("rows",
        source_rows)`` (one source prefill) or ``("memory", EncodedMemory)`` (one source
        prefill + encoder)."""

        if share not in ("rows", "memory"):
            raise ValueError(f"share: rows or memory, got {share!r}")
        if share == "rows":
            return "rows", self._source_view.source_rows(document)
        return "memory", self._source_view.encode_memory(document)

    def _encode_one(self, target: str, document, shared, questions, ratios,
                    return_scores) -> EncodedDocument:
        view = self.encoder(target)
        rendered = []
        for number, question in enumerate(questions):
            spec = ({"text": question} if isinstance(question, str)
                    else dict(se_prompts.question_dict(question)))
            if "after_ids" in spec:  # another target's ids would be silently wrong here
                raise ValueError("questions are rendered per target here: pass the text "
                                 "(and row_id/metric), not a rendered Question")
            spec.setdefault("row_id", f"q{number:03d}")
            rendered.append(view.question(spec.pop("text"), **spec))
        kind, value = shared
        given = {"source_rows": value} if kind == "rows" else {"memory": value}
        return view.encode(document, questions=rendered, ratios=ratios,
                           return_scores=return_scores, **given)

    @torch.no_grad()
    def encode(self, document: str | None = None, *, questions=(), ratios=(), share: str = "rows",
               source_rows: torch.Tensor | None = None, memory: EncodedMemory | None = None,
               return_scores: bool | int = False) -> dict[str, EncodedDocument]:
        """``{target: EncodedDocument}`` for one document (see ``encode_many``).
        ``source_rows`` / ``memory``: a precomputed shared input (no source prefill)."""

        return self.encode_many(
            [document], questions=[questions], ratios=ratios, share=share,
            return_scores=return_scores,
            source_rows=None if source_rows is None else [source_rows],
            memories=None if memory is None else [memory])[0]

    @torch.no_grad()
    def encode_many(self, documents, *, questions=None, ratios=(), share: str = "rows",
                    source_rows=None, memories=None,
                    return_scores: bool | int = False) -> list[dict[str, EncodedDocument]]:
        """One ``{target: EncodedDocument}`` per document. ``questions``: one list per
        document (or ``None``). Each document's shared input is computed once (unless
        given as ``source_rows`` / ``memories``, one per document); resident targets then
        encode document by document, offloaded targets target by target."""

        documents = list(documents)
        questions = [()] * len(documents) if questions is None else [list(q) for q in questions]
        if len(questions) != len(documents):
            raise ValueError("questions: one list per document")
        if source_rows is not None and memories is not None:
            raise ValueError("pass source_rows or memories, not both")
        given = source_rows if source_rows is not None else memories
        if given is not None and len(given) != len(documents):
            raise ValueError("source_rows / memories: one per document")
        kind = "rows" if source_rows is not None else "memory"
        results = [{} for _ in documents]
        if not self.offload_targets:
            for index, document in enumerate(documents):
                if given is None:
                    value = self.shared_input(document, share)
                else:
                    value = kind, given[index]
                for target in self.targets:
                    results[index][target] = self._encode_one(
                        target, document, value, questions[index], ratios, return_scores)
                del value
            return results
        # The source side once, before any target loads.
        if given is None:
            values = [self.shared_input(document, share) for document in documents]
        else:
            values = [(kind, value) for value in given]
        for target in self.targets:
            try:
                for index, document in enumerate(documents):
                    results[index][target] = self._encode_one(
                        target, document, values[index], questions[index], ratios, return_scores)
            finally:
                self.free(target)
        return results

    def release(self) -> None:
        """Free the GPU (every target, translator, source, compressors)."""

        for target in list(self.encoders):
            self.free(target)
        for name in ("translator", "source", "compressors", "_source_view"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        settings.empty_cache(self.device)


def answer_vllm(folder, *, python: str | None = None, out=None, max_model_len: int, native: bool = False,
                gpu: str | None = None, gpu_memory_utilization: float = 0.85) -> list[dict]:
    """Answer a saved folder (``Encoded.save`` / ``EncodedDocument.save``) with vLLM in a
    subprocess (``python``: its interpreter, default this one). Call
    ``SpeculativeEncoder.release()`` first when both share one GPU. Returns the answer
    records, one per question."""

    import subprocess
    import sys

    folder = Path(folder)
    out = Path(out) if out else folder / "answers.jsonl"
    out.unlink(missing_ok=True)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
           "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [python or sys.executable, "-m", "speculative_encoding.serving.vllm_engine", "--folder", str(folder),
               "--out", str(out), "--max-model-len", str(int(max_model_len)),
               "--gpu-memory-utilization", str(gpu_memory_utilization)]
    if native:
        command.append("--native")
    subprocess.run(command, env=env, check=True, stdout=sys.stdout, stderr=sys.stderr)
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
