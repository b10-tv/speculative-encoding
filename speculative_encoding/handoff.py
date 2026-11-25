"""Token-level handoff: any token ids (a model's own generation, not only a text document) as
another model's memory.

Two pieces:

* :func:`capture_rows`: the source-side hidden-state rows of token ids
  (``[tokens, streams, width]``: the embedding output, then every decoder layer's output, the
  layout ``models.source_rows_host`` returns), optionally continuing an existing cache, so a
  model's generation becomes translator input without re-prefilling text.
* :func:`token_spans` / :func:`sample_from_ids`: the translator aligns source and target tokens
  by character spans (``prompts.align``), so source ids need the text they decode to and each
  id's span in it. When the ids are exactly the source tokenizer's tokenization of that text,
  the spans are the tokenizer's own offsets and the sample equals ``prompts.make_sample``'s for
  that text (so ``SpeculativeEncoder.encode_ids`` gives ``encode(text)``'s prefix).

A relay (A reads a document, B answers from A's translated cache, B's answer becomes C's memory):

    # enc_ab, enc_bc = SpeculativeEncoder.from_pretrained_pairs(ckpt, [(A, B), (B, C)]).values()
    # doc = enc_ab.encode(text); q = enc_ab.question(...);
    # eos = hf_engine.stop_ids(B)
    cache, position = hf_engine.build_cache(enc_ab.target, E.prefix_to(doc.full.prefix, device),
        doc.full.prefix_ids)
    before = copy_cache(cache)  # translator.hf.copy_cache: generation consumes the cache
    answer = enc_ab.target.greedy(cache, torch.tensor(q.after_ids), position,
        max_new_tokens=200, eos_token_ids=eos)
    rows = capture_rows(enc_ab.target, q.after_ids + answer, cache=before,
        position=position)[len(q.after_ids):]
    # enc_bc: source = B's model, target = C. Special ids (EOS, chat markers) and their rows
    # are dropped by encode_ids.
    doc_c = enc_bc.encode_ids(answer, source_rows=rows)

Caveat (not a quality claim): the translator was trained on rows of a model *prefilling a
document* from a fresh cache. Rows captured after a translated cache, after a question, or over
a model's own continuation are a different distribution; this module makes the plumbing exact,
not the translation good. Measure before relying on it.
"""

from __future__ import annotations

import torch

from speculative_encoding import prompts as se_prompts
from speculative_encoding import settings
from speculative_encoding.models import source_rows_host


@torch.no_grad()
def capture_rows(
    interface, ids, *, cache=None, position: int | None = None, chunk: int | None = None
) -> torch.Tensor:
    """Source rows of ``ids`` for ``interface`` (a ``translator.hf.ModelInterface``):
    ``[tokens, layers + 1, width]`` on the CPU in the interface's dtype, the embedding output
    then every decoder layer's output (before the final norm).

    ``cache=None``: ``ids`` are read from position 0 with a fresh cache; this is
    ``models.source_rows_host(interface, [], ids)`` (so
    ``capture_rows(src, header + doc)[len(header):]`` equals
    ``source_rows_host(src, header, doc)``).

    ``cache``: an HF cache of the same model (``native_prefix``, ``hf_engine.build_cache``, or
    one a generation left); ``ids`` continue it from ``position`` (required: the number of
    tokens the cache holds, e.g. ``build_cache``'s second result). The cache is extended in
    place with ``ids`` (copy it first, ``translator.hf.copy_cache``, to keep it). ``chunk``:
    tokens per forward (default ``settings.source_chunk``); rows match a single forward up to
    kernel reordering, not bit for bit."""

    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1).to(interface.device)
    if cache is None:
        if position not in (None, 0):
            raise ValueError("position needs a cache: without one, ids are read from position 0")
        return source_rows_host(interface, [], ids.tolist(), interface.device).to("cpu")
    if position is None:
        raise ValueError("pass position= (the number of tokens the cache holds) with a cache")
    tokens, streams, width = int(ids.numel()), len(interface.layers) + 1, interface.width
    out = torch.empty((tokens, streams, width), dtype=interface.dtype, device="cpu")
    if not tokens:
        return out
    out[:, 0] = interface.embed(ids).reshape(tokens, width).to(interface.dtype).cpu()
    span = {"start": 0}

    def capture(index):
        def after(module, args, output):
            value = (output[0] if isinstance(output, tuple) else output)[0]
            out[span["start"]:span["start"] + value.shape[0], index + 1] = value.to(
                "cpu", interface.dtype)
        return after

    chunk = int(chunk or settings.current().source_chunk)
    hooks = [
        layer.register_forward_hook(capture(index))
        for index, layer in enumerate(interface.layers)
    ]
    try:
        for start in range(0, tokens, chunk):
            span["start"] = start
            piece = ids[start:start + chunk]
            positions = torch.arange(
                position + start, position + start + piece.numel(), device=interface.device)
            interface.decoder(
                input_ids=piece.view(1, -1), position_ids=positions.view(1, -1),
                past_key_values=cache, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()
    return out


def token_spans(tokenizer, ids, text: str | None = None) -> tuple[str, list[tuple[int, int]]]:
    """``(text, spans)``: the text ``ids`` decode to and each id's character span in it.

    When ``ids`` are exactly the tokenizer's tokenization of ``text``
    (``add_special_tokens=False``), the spans are the tokenizer's offsets (what
    ``prompts.render`` uses for a document). Otherwise (e.g. a generation that is not the
    canonical tokenization of its own decoded text) each id spans the characters its decode
    adds to the decode of the ids before it; an id that completes no character (a partial UTF-8
    byte) gets an empty span at the current end. ``text`` defaults to
    ``tokenizer.decode(ids)``; in the second case it must equal that decode."""

    ids = [int(token) for token in ids]
    if not ids:
        raise ValueError("no token ids")
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    if any(token in special for token in ids):
        raise ValueError("ids contain special tokens (e.g. EOS); pass only the text tokens")
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decoded if text is None else text
    canonical, offsets = se_prompts._encode_offsets(tokenizer, text)
    if canonical == ids:
        return text, offsets
    if text != decoded:
        raise ValueError("ids are not the tokenization of text, and text differs from their decode")
    spans, end = [], 0
    for count in range(1, len(ids) + 1):
        prefix = tokenizer.decode(
            ids[:count], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(prefix) >= end and text.startswith(prefix):
            spans.append((end, len(prefix)))
            end = len(prefix)
        else:
            # mid-character (byte fallback) or a decode that rewrites earlier text:
            # no characters yet
            spans.append((end, end))
    if end != len(text):
        raise ValueError(f"spans cover {end} of {len(text)} characters of the decoded text")
    return text, spans


def sample_from_ids(source_tokenizer, target_tokenizer, ids, config, *, text: str | None = None,
                    spans=None, question: str = "", role: str = "system") -> se_prompts.Sample:
    """The ``prompts.Sample`` of a document given as source token ids: the source side is
    ``ids`` with their spans (and the source chat header), the target side renders ``text``
    with the target tokenizer, aligned by character spans.

    ``text``/``spans`` default to :func:`token_spans`. The document text is used as given
    (``chat_row`` strips a text document; ids decode to whatever they decode to), and it is not
    truncated. With ``ids`` = the source tokenization of ``text.strip()``, this equals
    ``prompts.make_sample(chat_row(..., text), ...)`` (``tests/test_handoff.py``)."""

    ids = [int(token) for token in ids]
    if spans is None:
        text, spans = token_spans(source_tokenizer, ids, text)
    elif text is None:
        raise ValueError("spans need the text they index")
    spans = [tuple(map(int, span)) for span in spans]
    if len(spans) != len(ids):
        raise ValueError(f"{len(spans)} spans for {len(ids)} ids")
    if any(end > len(text) for _, end in spans):
        raise ValueError("a span ends past the text")
    row = se_prompts.chat_row("api", "document", text, question=question, role=role)
    empty = se_prompts.render(source_tokenizer, row, include_document=False, target_text="")
    source = se_prompts.Rendered(list(empty.header_ids), ids, spans, list(empty.after_ids))
    target = se_prompts.render(target_tokenizer, row, document=text, target_text=None)
    target.target_ids = target.target_ids[: config["max_target_tokens"]]
    if not target.doc_ids or not target.continuation_ids:
        raise ValueError("empty document after target tokenization")
    return se_prompts.Sample(
        row, source, target, se_prompts.align(source.doc_spans, target.doc_spans), "task")
