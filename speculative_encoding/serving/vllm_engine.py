"""Answer questions from a served prefix with vLLM.

* ``VLLMEngine(engine_spec, max_model_len=...)``: a ``vllm.LLM`` for a queue's ``engine.json``
  (bf16, seed 0, long-context options, prefix caching except for gated-delta targets, and the
  ``vllm_connector`` KV connector).
* ``engine.answer(prefix, prefix_ids, questions, native=...)``: the prefix (a tensor
  ``[layers, 2, tokens, kv_heads, head_dim]`` or a ``HybridWriter`` dict) goes into
  ``vllm_connector.STORE[name]``; every question is ``prefix_ids + after_ids`` with cache salt
  ``translated:<name>`` and ``kv_transfer_params={"inject": name, "tokens": len(prefix_ids)}``,
  greedy, stopping at the engine's EOS ids. The first question runs alone (it injects the
  document once; the rest reuse its blocks through the prefix cache), the others together.
  ``native=True`` first sends the same ids without injection (salt ``native:<name>``).
* ``answer_folder(folder, out_path, ...)``: every ``<name>.pt`` / ``<name>.json`` item of a queue
  folder, one JSON line per question.

    python -m speculative_encoding.serving.vllm_engine --folder QUEUE --out answers.jsonl \\
        [--max-model-len N] [--native]

The connector reads ``STORE`` in this process, so vLLM runs with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (set here unless already set). With
``tensor_parallel_size > 1`` the workers load ``<settings.vllm_export_dir>/<name>.pt`` instead
(full-attention tensors only), which this module does not write.

``sampling=`` (or a question dict's ``"sampling"``; ``sampling.Sampling`` or a dict) adds
free-form sampling to ``SamplingParams``; such records take ``text`` from vLLM's detokenized
output and carry ``samples`` when ``n > 1``. No streaming: ``LLM.generate`` returns finished
outputs, and ``AsyncLLM`` runs the engine core in another process, where ``STORE`` is not
visible; use ``hf_engine.stream``.
"""

from __future__ import annotations


import argparse
import hashlib
import gc
import itertools
import json
import os
import sys
from pathlib import Path

import torch

from speculative_encoding.models import long_context_options
from speculative_encoding.encode import prefix_tokens
from speculative_encoding.prompts import question_dict
from speculative_encoding.serving.hf_engine import as_sampling, truncate_at_stop

__all__ = ["VLLMEngine", "answer_folder", "llm_kwargs", "nll_from_logprobs"]

CONNECTOR_MODULE = "speculative_encoding.serving.vllm_connector"
_names = itertools.count()


def nll_from_logprobs(output, start, reference_ids):
    """Mean NLL of ``reference_ids`` at prompt positions ``start``... of a vLLM output with
    ``prompt_logprobs`` (cross entropy over the reference tokens given everything before). None
    when vLLM returned no log-prob for a position (e.g. one inside an injected or cached
    prefix)."""

    logprobs = output.prompt_logprobs or []
    # With an injected (or cached) prefix vLLM returns log-probs only for the positions it
    # computed, so the list is aligned to the prompt's end.
    shift = len(getattr(output, "prompt_token_ids", None) or logprobs) - len(logprobs)
    values = []
    for offset, token in enumerate(reference_ids):
        position = start + offset - shift
        entry = logprobs[position] if 0 <= position < len(logprobs) else None
        if not entry or int(token) not in entry:
            return None
        values.append(-float(entry[int(token)].logprob))
    return sum(values) / len(values) if values else None


def llm_kwargs(
    engine_spec, *, max_model_len, gpu_memory_utilization=0.85, tensor_parallel_size=1,
    **llm_overrides):
    """The ``vllm.LLM`` keyword arguments for ``engine_spec`` (``engine.json``), then
    ``llm_overrides``. ``tensor_parallel_size`` is passed only when not 1."""

    from vllm.config import KVTransferConfig

    model, revision = engine_spec["model"], engine_spec["revision"]
    options = long_context_options(model, revision, engine_spec["options"], max_model_len)
    # Hybrid targets keep vLLM's hybrid KV cache manager on (HybridKVConnector); gated-delta
    # targets run without prefix caching (one state block per request).
    hybrid, linear = engine_spec.get("hybrid", False), engine_spec.get("linear", False)
    kwargs = dict(
        model=model, revision=revision, dtype="bfloat16", seed=0, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=not linear,
        generation_config="vllm", **options,
        kv_transfer_config=KVTransferConfig(
            kv_connector="HybridKVConnector" if hybrid else "TranslatedKVConnector",
            kv_connector_module_path=CONNECTOR_MODULE, kv_role="kv_both"))
    if tensor_parallel_size != 1:
        kwargs["tensor_parallel_size"] = tensor_parallel_size
    kwargs.update(llm_overrides)
    return kwargs


class VLLMEngine:
    """One ``vllm.LLM`` for one target, answering questions from injected prefixes (see the module
    docstring)."""

    def __init__(
        self, engine_spec: dict, *, max_model_len: int, gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1, **llm_overrides):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        if os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0":
            print(
                "vllm_engine: VLLM_ENABLE_V1_MULTIPROCESSING is not 0: the connector will not "
                "see STORE", file=sys.stderr, flush=True)
        from vllm import LLM

        from speculative_encoding.serving import vllm_connector

        self.connector = vllm_connector
        self.spec = dict(engine_spec)
        self.max_model_len = int(max_model_len)
        self.stop_token_ids = list(engine_spec["eos_token_ids"])
        self.linear = bool(engine_spec.get("linear", False))
        self.llm = LLM(**llm_kwargs(
            engine_spec, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size, **llm_overrides))
        self.tokenizer = self.llm.get_tokenizer()
        # Without prefix caching (gated-delta targets) every question holds its whole document in
        # the KV cache. More long requests at once than the cache holds made vLLM preempt one, and
        # a resumed request never ran its injection (Qwen3.5 at 128k on H200). Such targets
        # answer at most ``capacity // prompt`` questions at a time.
        self.capacity = None
        try:
            cache = self.llm.llm_engine.vllm_config.cache_config
            if cache.num_gpu_blocks:
                self.capacity = int(cache.num_gpu_blocks) * int(cache.block_size)
        except Exception:
            pass

    def _arm(self, arm, name, base_ids, tokens, questions, salt, max_tokens, sampling=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        extra = {"kv_transfer_params": {"inject": name, "tokens": tokens}}
        extra = extra if arm == "translated" else None
        prompts, params, samplings = [], [], []
        for question in questions:
            nll = question.get("metric") == "nll"
            own = None if nll else as_sampling(question.get("sampling", sampling))
            samplings.append(own)
            if nll:  # one forward over prompt + reference, log-probs of the reference
                ids = base_ids + list(question["continuation_ids"])
                prompts.append(TokensPrompt(prompt_token_ids=ids, cache_salt=salt))
                params.append(SamplingParams(
                    temperature=0.0, max_tokens=1, prompt_logprobs=0, extra_args=extra))
                continue
            ids = base_ids + list(question["after_ids"])
            prompts.append(TokensPrompt(prompt_token_ids=ids, cache_salt=salt))
            limit = question.get("max_tokens", max_tokens)
            if limit is None:
                raise ValueError(
                    "max_tokens is required (per question or as answer(..., max_tokens=...): "
                    "prompts.GENERATION_TOKENS[metric])")
            if own is None:  # the paper's request
                params.append(SamplingParams(
                    temperature=0.0, max_tokens=limit, stop_token_ids=self.stop_token_ids,
                    extra_args=extra))
                continue
            stops = self.stop_token_ids
            stops = stops + [token for token in own.stop_token_ids if token not in stops]
            params.append(SamplingParams(
                max_tokens=limit, stop_token_ids=stops, extra_args=extra, **own.vllm_kwargs()))
        # The first request computes or injects the document once; the rest hit the prefix cache.
        outputs = self.llm.generate(prompts[:1], params[:1], use_tqdm=False)
        rest, rest_params = prompts[1:], params[1:]
        per = len(rest) or 1
        if self.linear and len(rest) > 1:
            longest = max(len(prompt["prompt_token_ids"]) for prompt in rest)
            if self.capacity:
                per = max(1, int(0.8 * self.capacity) // longest)
            elif longest > 16384:
                per = 1
        for first in range(0, len(rest), per):
            outputs += self.llm.generate(
                rest[first:first + per], rest_params[first:first + per], use_tqdm=False)
        records = []
        for output, question, own in zip(outputs, questions, samplings):
            completion = output.outputs[0]
            record = {
                "text": self.tokenizer.decode(completion.token_ids, skip_special_tokens=True),
                "ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids or []),
                "cached_tokens": output.num_cached_tokens,
            }
            if own is not None:
                samples = []
                for choice in output.outputs:
                    text = getattr(choice, "text", None)
                    if not isinstance(text, str):
                        text = self.tokenizer.decode(choice.token_ids, skip_special_tokens=True)
                        text = truncate_at_stop(text, own.stop)[0]
                    samples.append({
                        "text": text, "ids": list(choice.token_ids),
                        "finish_reason": choice.finish_reason})
                record.update(samples[0])
                if len(samples) > 1:
                    record["samples"] = samples
            if question.get("metric") == "nll":
                after = len(question["after_ids"])
                reference = question["continuation_ids"][after:]
                record["nll"] = nll_from_logprobs(output, tokens + after, reference)
                record["prompt_logprobs"] = [
                    len(output.prompt_logprobs or []), len(output.prompt_token_ids or [])]
            records.append(record)
        return records

    def answer(
        self, prefix, prefix_ids, questions, *, native=False, salt=None, name=None,
        native_ids=None, native_salt=None, max_tokens=None, sampling=None) -> list[dict]:
        """Greedy answers to ``questions`` (dicts: ``after_ids``, ``max_tokens``, optional
        ``metric`` / ``continuation_ids``) after the injected ``prefix``, whose length must equal
        ``len(prefix_ids)``.

        One dict per question: ``text``, ``ids``, ``finish_reason``, ``prompt_tokens``,
        ``cached_tokens`` (``nll`` and ``prompt_logprobs`` for ``metric == "nll"``); with
        ``native=True`` also ``native``: the same for ``native_ids`` (default ``prefix_ids``) sent
        without injection. ``name``: the ``STORE`` key (default: a fresh one), removed afterwards;
        cache salts ``salt`` / ``native_salt`` default to ``translated:<name>`` /
        ``native:<name>``. ``sampling``: free-form sampling for every question without its own
        ``"sampling"``; ``None`` sends the paper's greedy requests."""

        questions = [question_dict(question) for question in questions]
        if not questions:
            return []
        prefix_ids = [int(token) for token in prefix_ids]
        tokens = len(prefix_ids)
        held = prefix_tokens(prefix)
        if held != tokens:
            raise ValueError(f"prefix holds {held} tokens but {tokens} prefix ids were given")
        name = name or f"vllm-engine-{os.getpid()}-{next(_names)}"
        if name in self.connector.STORE:
            raise KeyError(f"STORE already holds {name!r}")
        native_ids = prefix_ids if native_ids is None else [int(token) for token in native_ids]
        self.connector.STORE[name] = prefix  # stays where it is (CPU: copied per layer)
        try:
            if native:
                natives = self._arm(
                    "native", name, native_ids, tokens, questions,
                    native_salt or f"native:{name}", max_tokens, sampling)
            out = self._arm(
                "translated", name, prefix_ids, tokens, questions,
                salt or f"translated:{name}", max_tokens, sampling)
        finally:
            self.connector.STORE.pop(name, None)
        if native:
            for record, native_record in zip(out, natives):
                record["native"] = native_record
        return out

    def close(self):
        """Shut the engine down and free its GPU memory (the engine is unusable afterwards)."""

        llm, self.llm = getattr(self, "llm", None), None
        if llm is None:
            return
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def answer_folder(
    folder, out_path, *, max_model_len, native=False, engine=None, gpu_memory_utilization=0.85,
    tensor_parallel_size=1, skip_too_long=False, **llm_overrides):
    """Answer every item of a queue folder (``engine.json`` + ``<name>.pt`` / ``<name>.json``)
    and append one JSON line per question to ``out_path``. ``native``: also the native arm, for
    items without ``no_native``. Items whose longest prompt plus answer exceeds ``max_model_len``
    raise an error, or with ``skip_too_long`` are skipped and listed in the return value."""

    folder, out_path = Path(folder), Path(out_path)
    spec = json.loads((folder / "engine.json").read_text())
    owned = engine is None
    if owned:
        engine = VLLMEngine(
            spec, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size, **llm_overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(
        path.stem for path in folder.glob("*.json")
        if path.name != "engine.json" and path.with_suffix(".pt").exists())
    written, skipped = 0, []
    try:
        for name in names:
            meta = json.loads((folder / f"{name}.json").read_text())
            if not meta["questions"]:  # e.g. a full cache saved for later questions
                continue
            # Compressed cartridges carry the full document for the native arm.
            native_ids = meta.get("native_prefix_ids") or meta["prefix_ids"]
            longest = max(len(meta["prefix_ids"]), len(native_ids)) + max(
                len(question.get("continuation_ids") or question["after_ids"])
                + question.get("max_tokens", meta["max_tokens"])
                for question in meta["questions"])
            if longest > max_model_len:  # the target's tokenizer made the document too long
                if not skip_too_long:
                    raise ValueError(
                        f"{name}: {longest} tokens with the longest question and answer > "
                        f"max_model_len {max_model_len}; raise --max-model-len (the model "
                        "supports more: check its config) or pass --skip-too-long")
                print(
                    f"vllm_engine: skip {name}: {longest} tokens with the longest question and "
                    f"answer > max model length {max_model_len}", file=sys.stderr, flush=True)
                skipped.append({"name": name, "key": meta.get("key"), "tokens": longest})
                continue
            native_salt = None
            if meta.get("native_prefix_ids"):  # one salt per document: its native prefix is cached once
                document = meta.get("key") or hashlib.sha1(str(native_ids).encode()).hexdigest()[:16]
                native_salt = f"native:{document}"
            answers = engine.answer(
                torch.load(folder / f"{name}.pt"), meta["prefix_ids"], meta["questions"],
                native=native and not meta.get("no_native"), name=name, native_ids=native_ids,
                native_salt=native_salt, max_tokens=meta.get("max_tokens"))
            with out_path.open("a") as handle:
                for question, answer in zip(meta["questions"], answers):
                    record = {
                        "name": name, **{key: meta.get(key) for key in ("key", "length", "domain")},
                        "row_id": question.get("row_id"), "answer": question.get("answer"),
                        "metric": question.get("metric"), "doc_tokens": meta.get("doc_tokens"),
                        "arm": meta.get("arm"), "source": meta.get("source"),
                        "compressor_ratio": meta.get("compressor_ratio")}
                    native_answer = answer.pop("native", None)
                    record["translated"] = answer
                    if native_answer is not None:
                        record["native"] = native_answer
                    handle.write(json.dumps(record) + "\n")
                    written += 1
    finally:
        if owned:
            engine.close()
    return {"questions": written, "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--folder", type=Path, required=True,
        help="a queue folder (engine.json, <name>.pt, <name>.json)")
    parser.add_argument("--out", type=Path, required=True, help="answers JSONL (appended)")
    parser.add_argument("--max-model-len", type=int, default=140000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--native", action="store_true",
        help="also answer the native arm (items without no_native)")
    parser.add_argument(
        "--skip-too-long", action="store_true",
        help="skip items longer than --max-model-len (default: error)")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    result = answer_folder(
        args.folder, args.out, max_model_len=args.max_model_len, native=args.native,
        skip_too_long=args.skip_too_long, gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size)
    print(json.dumps({"out": str(args.out), **result}), flush=True)


if __name__ == "__main__":
    main()
