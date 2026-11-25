"""Prefix algebra: concatenate separately encoded documents into one served prefix, extend a
served prefix with native tokens, truncate it.

A served prefix is what ``api.Encoded.prefix`` holds: a tensor
``[layers, 2, tokens, kv_heads, head_dim]`` (every layer plain full attention) or a
``HybridWriter`` dict (``tokens``; per layer ``kv`` / ``kv_first``; gated-delta ``conv`` /
``ssm``). Keys are post-RoPE and the server numbers positions by cache index, so moving an
entry to a new cache index means rotating its key by the same number of positions
(``rotate_keys``; partial rotary rotates only the first ``rotary_dim`` channels). RoPE depends
only on query-key distance, so the rotation is exact up to the float rounding of one extra
rotation (bit-level differences in bf16, ~1e-7 relative in fp32).

``concatenate`` -- what is exact and what is not
-------------------------------------------------
Each part is ``header + doc_i`` encoded on its own (translated or native). The result is
``header + doc_1 + doc_2 + ...``: the first part as is, every later part with its header
dropped and its keys rotated by the number of document tokens before it. K and V are per-token
functions of a layer's input and the token's position, so the result is *exactly* the cache the
target would hold if each document's layer inputs, as computed when that document was read
alone after the header, were written after the previous documents at their new positions (ds1's
``write_prefix`` at offset positions; checked in ``tests/test_prefix_ops.py``).

It is *not* a native read of the concatenation, and no positional trick makes it one:

* doc_k's tokens never attended to doc_1 .. doc_k-1 (no cross-document attention): their layer
  inputs past the embedding, hence their K/V, are the ones of a read of ``header + doc_k``
  alone;
* doc_k's content was computed with doc_k right after the header (its header-to-document
  distances are those of the standalone read), only its keys moved.

Tokens served after the prefix (questions, answers, ``extend``) attend to every document
normally. The first part's entries are exactly a native read of ``header + doc_1``. Headers
must have equal ids (``prefix_ids[:header_tokens]``); their K/V are taken from the first part.

Hybrid targets: sliding-window layers keep the last ``window`` entries of the concatenation
(mostly the last document; earlier documents' tails when the last one is shorter than the
window) -- exact against the same reference, as long as each part holds its own last ``window``
entries (``HybridWriter`` and ``prefix_from_cache`` both do). Gated-delta
(``linear_attention``) states cannot be concatenated: the state after doc_2 depends on the
state it started from through a product of per-token transitions that the stored prefix does
not keep, so ``concatenate`` raises for any part with ``conv`` / ``ssm`` states. Compressed
cartridges (``arm`` ``c16`` ...) are laid out like full prefixes (slots at ``header + i``) and
concatenate mechanically, but the compressor was trained on one document per cartridge:
untested.

RoPE variants whose frequencies depend on the sequence length (``dynamic``, ``longrope``)
cannot be shifted by a fixed rotation: refused.
"""

from __future__ import annotations

import dataclasses

import torch

from speculative_encoding.encode import prefix_tokens

__all__ = [
    "inv_freqs", "windows", "shift", "concatenate", "concatenate_encoded", "extend", "truncate"]

_UNSHIFTABLE = ("dynamic", "longrope")


def _interface(model):
    from speculative_encoding.serving.hf_engine import as_interface

    return as_interface(model)


def inv_freqs(interface) -> dict[int, torch.Tensor]:
    """``{layer index: inv_freq}`` for every attention layer of the target (each layer type's
    own RoPE frequencies; ``rotary_dim / 2`` entries for partial rotary)."""

    interface = _interface(interface)
    rotary = interface.decoder.rotary_emb
    rope_type = getattr(rotary, "rope_type", "default")
    types = rope_type.values() if isinstance(rope_type, dict) else [rope_type]
    if any(kind in _UNSHIFTABLE for kind in types):
        raise NotImplementedError(
            f"RoPE type {rope_type}: frequencies depend on the sequence length, "
            "keys cannot be shifted exactly")
    out = {}
    for index, kind in enumerate(interface.layer_types):
        if kind == "linear_attention":
            continue
        freq = getattr(rotary, f"{kind}_inv_freq", None)
        out[index] = (freq if freq is not None else rotary.inv_freq).detach().float().cpu()
    return out


def windows(interface) -> dict[int, int]:
    """``{layer index: sliding window}`` (0: full attention) for every attention layer."""

    interface = _interface(interface)
    window = int(getattr(interface.config, "sliding_window", 0) or 0)
    return {
        index: (window if kind == "sliding_attention" else 0)
        for index, kind in enumerate(interface.layer_types)
        if kind != "linear_attention"
    }


def rotate_keys(keys: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Post-RoPE keys (``[..., head_dim]``, neox/rotate-half layout as in HF Qwen3 / Llama and
    vLLM) moved by ``delta`` positions.

    ``inv_freq`` has ``rotary_dim / 2`` entries. Partial-rotary models (e.g. Qwen 3.5) rotate
    only the first ``rotary_dim`` channels of each head and leave the rest as they are; with
    ``rotary_dim == head_dim`` (full rotary) this is the original computation, unchanged."""

    if delta == 0:
        return keys
    angle = float(delta) * inv_freq.to(device=keys.device, dtype=torch.float64)
    cos, sin = torch.cos(angle).float(), torch.sin(angle).float()
    cos, sin = torch.cat((cos, cos)), torch.cat((sin, sin))
    x = keys.float()
    rotary = cos.shape[-1]
    if rotary < x.shape[-1]:  # partial rotary: rotate the first ``rotary`` channels only
        return torch.cat(
            (rotate_keys(keys[..., :rotary], delta, inv_freq), keys[..., rotary:]), dim=-1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + rotated * sin).to(keys.dtype)


def _freq(freqs, index):
    return freqs if torch.is_tensor(freqs) else freqs[index]


def _rotate_kv(kv: torch.Tensor, delta: int, inv_freq) -> torch.Tensor:
    """``[2, tokens, kv_heads, head_dim]`` with its keys moved by ``delta`` positions (new
    tensor)."""

    return torch.stack((rotate_keys(kv[0], delta, inv_freq), kv[1]))


def shift(prefix, delta: int, freqs):
    """Every key of ``prefix`` moved by ``delta`` positions (``freqs``: one ``inv_freq`` or
    ``inv_freqs(interface)``)."""

    if isinstance(prefix, dict):
        return {**prefix, "kv": {
            index: _rotate_kv(kv, delta, _freq(freqs, index))
            for index, kv in prefix["kv"].items()
        }}
    if torch.is_tensor(freqs):
        return torch.stack((rotate_keys(prefix[:, 0], delta, freqs), prefix[:, 1]), dim=1)
    return torch.stack(
        [_rotate_kv(prefix[layer], delta, freqs[layer]) for layer in range(prefix.shape[0])])


@torch.no_grad()
def concatenate(parts, *, header_tokens: int, interface=None, freqs=None):
    """``header + doc_1 + doc_2 + ...`` from ``parts`` = ``[(prefix, prefix_ids), ...]``, each
    ``header + doc_i`` encoded on its own. Returns ``(prefix, prefix_ids)`` in the parts' form.
    ``freqs``: an ``inv_freq`` (tensor prefixes) or ``inv_freqs(interface)``; else taken from
    ``interface`` (the target). Dict prefixes need ``interface`` (sliding windows). See the
    module docstring for what is exact."""

    parts = [(prefix, list(ids)) for prefix, ids in parts]
    h = int(header_tokens)
    if not parts:
        raise ValueError("concatenate needs at least one part")
    header, hybrid = parts[0][1][:h], isinstance(parts[0][0], dict)
    for number, (prefix, ids) in enumerate(parts):
        if prefix_tokens(prefix) != len(ids):
            raise ValueError(f"part {number}: prefix holds {prefix_tokens(prefix)} tokens but has "
                             f"{len(ids)} prefix ids")
        if len(ids) < h or ids[:h] != header:
            raise ValueError(f"part {number}: header ids differ from part 0's "
                             "(concatenation keeps one header)")
        if isinstance(prefix, dict) != hybrid:
            raise ValueError("parts mix tensor and dict prefixes")
        if hybrid and (prefix.get("conv") or prefix.get("ssm")):
            raise NotImplementedError(
                "gated-delta (linear attention) states cannot be concatenated: the state after "
                "a document depends on the state it started from, which a separately encoded "
                "part does not keep")
    if freqs is None:
        if interface is None:
            raise ValueError("pass interface (the target) or freqs")
        freqs = inv_freqs(interface)
    ids = parts[0][1] + [token for _, part_ids in parts[1:] for token in part_ids[h:]]
    offsets, start = [], 0  # document tokens before each part
    for prefix, _ in parts:
        offsets.append(start)
        start += prefix_tokens(prefix) - h
    total = h + start

    if not hybrid:
        pieces = [parts[0][0]] + [shift(prefix[:, :, h:], offset, freqs)
                                  for (prefix, _), offset in zip(parts[1:], offsets[1:])]
        return torch.cat(pieces, dim=2), ids

    if interface is None:
        raise ValueError("dict prefixes need interface (the target's sliding windows)")
    window_of = windows(interface)
    out = {"tokens": total, "kv": {}, "kv_first": {}, "conv": {}, "ssm": {}}
    for index in parts[0][0]["kv"]:
        window = window_of.get(index, 0)
        pieces = []
        # last part first: sliding layers stop once the window is covered
        for number in range(len(parts) - 1, -1, -1):
            prefix, offset = parts[number][0], offsets[number]
            kv, first = prefix["kv"][index], int(prefix["kv_first"][index])
            if number:
                # drop this part's header entries (position < h), move the rest after the
                # previous documents
                kv, first = kv[:, max(0, h - first):], max(first, h)
                kv = _rotate_kv(kv, offset, _freq(freqs, index))
            elif not window and first:
                raise ValueError(
                    f"layer {index}: full-attention layer starts at entry {first}, not 0")
            pieces.insert(0, kv)
            if window and (
                first > (h if number else 0)
                or sum(piece.shape[1] for piece in pieces) >= window
            ):
                break  # earlier entries are unavailable (gap) or unneeded
        kv = torch.cat(pieces, dim=1)
        if window:
            kv = kv[:, -window:]
            # entries the next query reads (HF keeps window - 1, vLLM window)
            needed = min(total, window - 1)
            if kv.shape[1] < needed:
                raise ValueError(
                    f"layer {index}: only {kv.shape[1]} contiguous sliding-window entries, "
                    f"the next query reads {needed}")
        out["kv"][index], out["kv_first"][index] = kv, total - kv.shape[1]
    return out, ids


def concatenate_encoded(items, interface, *, freqs=None):
    """``concatenate`` over ``api.Encoded`` items of one target (same header): a new
    ``Encoded`` whose ``meta`` records each document's span (``documents``: ``start`` /
    ``tokens`` in prefix positions) and that the parts were encoded separately
    (``concatenated``)."""

    items = list(items)
    if not items:
        raise ValueError("concatenate_encoded needs at least one item")
    first = items[0]
    for item in items[1:]:
        if item.target != first.target or item.engine != first.engine:
            raise ValueError(
                "cannot concatenate prefixes of different targets "
                f"({first.target} vs {item.target})")
        if item.header_tokens != first.header_tokens:
            raise ValueError("cannot concatenate prefixes with different header lengths")
    prefix, ids = concatenate(
        [(item.prefix, item.prefix_ids) for item in items], header_tokens=first.header_tokens,
        interface=interface, freqs=freqs)
    documents, start = [], first.header_tokens
    for item in items:
        tokens = len(item.prefix_ids) - item.header_tokens
        documents.append({
            "start": start, "tokens": tokens, "arm": item.arm, "doc_tokens": item.doc_tokens,
            **({"meta": item.meta} if item.meta else {})})
        start += tokens
    arms = {item.arm for item in items}
    return dataclasses.replace(
        first, arm=arms.pop() if len(arms) == 1 else "mixed", prefix=prefix, prefix_ids=ids,
        doc_tokens=sum(item.doc_tokens for item in items),
        questions=[question for item in items for question in item.questions],
        meta={"concatenated": True, "documents": documents})


@torch.no_grad()
def extend(prefix, prefix_ids, new_ids, interface):
    """``prefix`` (served for ``prefix_ids``; ``None``: a native prefill of them) followed by
    native ``new_ids``: the target reads ``new_ids`` at positions ``len(prefix_ids) + j`` with
    ``prefix`` as its cache (as a served question or turn would). Returns
    ``(prefix, prefix_ids + new_ids)`` in ``prefix``'s form (``None``: a tensor for a
    full-attention target, else a dict), on ``prefix``'s device. With a native prefix this is a
    native prefill of ``prefix_ids + new_ids`` (up to matmul-shape rounding); with a translated
    or concatenated one, the new tokens attend to it exactly as served answers do."""

    from speculative_encoding.encode import prefix_to
    from speculative_encoding.serving.hf_engine import build_cache
    from speculative_encoding.encode import prefix_from_cache
    from speculative_encoding.translator.hf import _is_kv

    interface = _interface(interface)
    new_ids = [int(token) for token in new_ids]
    prefix_ids = [int(token) for token in prefix_ids]
    device = (
        None if prefix is None
        else next(iter(prefix["kv"].values())).device if isinstance(prefix, dict)
        else prefix.device)
    cache, tokens = build_cache(interface, prefix, prefix_ids)
    if new_ids:
        interface.forward(
            torch.tensor(new_ids, device=interface.device),
            torch.arange(tokens, tokens + len(new_ids), device=interface.device), cache)
    total = tokens + len(new_ids)
    ids = prefix_ids + new_ids
    full = prefix_from_cache(interface, cache, total)
    if prefix is not None and not isinstance(prefix, dict):
        dtype = prefix.dtype
        return torch.stack([full["kv"][index] for index in sorted(full["kv"])]).to(
            device=device, dtype=dtype), ids
    if prefix is None:
        result = (
            torch.stack([full["kv"][index] for index in sorted(full["kv"])])
            if interface.all_full_attention else full)
        return prefix_to(result, "cpu"), ids
    # Sliding-window layers: HF keeps window - 1 entries; keep the prefix's own tail plus the
    # new entries (up to the window, as HybridWriter does) when HF still holds every new entry.
    window_of = windows(interface)
    for index, kv in prefix["kv"].items():
        window = window_of.get(index, 0)
        layer = cache.layers[index]
        if not window or not _is_kv(layer) or index not in full["kv"]:
            continue
        held = full["kv"][index]
        if len(new_ids) <= held.shape[1]:
            merged = torch.cat(
                (kv.to(held), held[:, held.shape[1] - len(new_ids):]), dim=1)[:, -window:]
            full["kv"][index], full["kv_first"][index] = merged, total - merged.shape[1]
    for part in ("kv", "conv", "ssm"):
        reference = prefix[part]
        full[part] = {
            index: value.to(dtype=reference[index].dtype if index in reference else value.dtype)
            for index, value in full[part].items()
        }
    return prefix_to(full, device), ids


def truncate(prefix, prefix_ids, tokens: int):
    """The first ``tokens`` entries of ``prefix`` (exact: causal attention, no entry depends on
    a later token) and ``prefix_ids[:tokens]``. Dict prefixes: refused when a gated-delta state
    is present (it cannot be rewound) or a sliding-window layer would lose entries its window
    needs (only its last window was kept)."""

    total, tokens = prefix_tokens(prefix), int(tokens)
    if not 0 <= tokens <= total:
        raise ValueError(f"cannot truncate a {total}-token prefix to {tokens}")
    if len(prefix_ids) != total:
        raise ValueError(f"prefix holds {total} tokens but has {len(prefix_ids)} prefix ids")
    ids = list(prefix_ids[:tokens])
    if not isinstance(prefix, dict):
        return prefix[:, :, :tokens], ids
    if tokens == total:
        return prefix, ids
    if prefix.get("conv") or prefix.get("ssm"):
        raise NotImplementedError(
            "gated-delta (linear attention) states cannot be rewound to an earlier token")
    out = {"tokens": tokens, "kv": {}, "kv_first": {}, "conv": {}, "ssm": {}}
    for index, kv in prefix["kv"].items():
        first = int(prefix["kv_first"][index])
        if first:
            raise ValueError(
                f"layer {index}: only the last {kv.shape[1]} entries were kept (sliding window); "
                "cannot truncate")
        out["kv"][index], out["kv_first"][index] = kv[:, :tokens], 0
    return out, ids
