"""The encode stage: a document's source states -> the translator -> a target KV cache, plus
per-question compressed caches.

Sections: KVzip selection scores; the translator direction and cache writers; the encode
stage itself; the stored encoder output (encode once, decode for any target)."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
import torch
from speculative_encoding.translator.hf import ModelInterface, _base_attention, copy_cache
import copy
import inspect
import sys
import time
from types import SimpleNamespace
from speculative_encoding import settings as D
import hashlib
import json
from pathlib import Path
from speculative_encoding import prompts as se_prompts


# ---- KVzip selection ---------------------------------------------------------------------------
# KVzip-style cache selection through each model's own Hugging Face attention.
#
# KVzip (Kim et al., 2025) keeps a cache entry when the model attends to it while
# it reads the context back from its own cache. ``score`` feeds probe tokens
# after a native prefix cache and records, for every compressible cache layer and
# KV head, the largest attention probability any probe query gives each document
# entry:
#
#     s_l,g(t) = max over probe queries q and query heads h in KV group g of
#                softmax(q K_l,g^T)_t
#
# A question as the probe gives a question-aware score. ``select`` keeps the top
# ``budget`` entries per KV head. Only full-attention K/V layers are scored:
# sliding-window and recurrent layers are already bounded and stay native.

_SCORE = "unified_translator_kvzip_score"
_SCORE_REGISTERED = False
_SCORE_STATE: contextvars.ContextVar["_ScoreState | None"] = (
    contextvars.ContextVar("unified_kvzip_score", default=None))


@dataclass
class _ScoreState:
    base: str
    start: int
    stop: int
    skip: int
    owners: dict
    query_block: int = 256
    scores: dict = field(default_factory=dict)


def _score_attention(module, query, key, value, attention_mask, **kwargs):
    """The model's own attention, plus the largest attention probability each
    document key gets from the scored queries, per KV head."""

    state = _SCORE_STATE.get()
    base = _base_attention(state.base, module)
    output = base(module, query, key, value, attention_mask, **kwargs)
    layer = state.owners.get(id(module))
    if layer is None:
        return output
    assert key.shape[0] == 1, "KVzip scoring runs one sequence at a time"
    heads, queries, dim = query.shape[1:]
    kv_heads, keys = key.shape[1], key.shape[2]
    assert heads % kv_heads == 0
    scaling = kwargs.get("scaling", getattr(module, "scaling", None))
    softcap = kwargs.get("softcap")
    start, stop, block = state.start, state.stop, state.query_block
    with torch.no_grad():
        grouped = query[0].float().reshape(kv_heads, heads // kv_heads,
            queries, dim)
        cached = key[0].float()
        scale = dim**-0.5 if scaling is None else float(scaling)
        offset = keys - queries
        best = cached.new_zeros(kv_heads, stop - start)
        columns = torch.arange(keys, device=key.device)
        for first in range(state.skip, queries, block):
            last = min(queries, first + block)
            logits = torch.einsum("gnqd,gsd->gnqs",
                grouped[:, :, first:last], cached) * scale
            if softcap:
                logits = softcap * torch.tanh(logits / softcap)
            rows = offset + torch.arange(first, last, device=key.device)
            future = columns[None, :] > rows[:, None]
            logits = logits.masked_fill(future, float("-inf"))
            probs = torch.softmax(logits, dim=-1)[..., start:stop]
            best = torch.maximum(best, probs.amax(dim=(1, 2)))
    previous = state.scores.get(layer)
    state.scores[layer] = (best if previous is None
        else torch.maximum(previous, best))
    return output


def _score_mask(*args, **kwargs):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    return ALL_MASK_ATTENTION_FUNCTIONS[_SCORE_STATE.get().base](*args, **kwargs)


@torch.no_grad()
def score(interface: ModelInterface, cache, start, stop, probe_ids, position,
        *, skip=0, query_block=256, scores=None):
    """Per compressible layer, ``[kv_heads, stop - start]`` max attention from
    the probe to the document.

    ``cache`` holds the native prefix (``stop`` entries per full layer,
    document at ``[start, stop)``) and is not modified. ``probe_ids`` are read
    from ``position`` on; the first ``skip`` of them (an instruction) are not
    scored. Pass ``scores`` to take the running maximum with an earlier probe.
    """

    from transformers.cache_utils import DynamicLayer

    global _SCORE_REGISTERED
    if not interface.continues_cache:
        raise ValueError(f"{interface.name}: KVzip scoring needs a cache that "
            "continues over several tokens")
    # Full-attention K/V cache layers that hold every document entry.
    layers = tuple(index for index in interface.kv_layers
        if index < len(cache.layers)
        and type(cache.layers[index]) is DynamicLayer
        and not interface.is_sliding_kv(index))
    if not layers:
        raise ValueError(
            f"{interface.name}: no full-attention K/V layer to compress")
    probe_ids = torch.as_tensor(probe_ids, device=interface.device).reshape(-1)
    assert probe_ids.numel() > skip, "the probe has no scored tokens"
    # Attention module -> the cache layer it reads. A shared-K/V layer
    # (Gemma-style) reads the last earlier writer of its layer type.
    owners = {}
    for index, decoder_layer in enumerate(interface.layers):
        attention = getattr(decoder_layer, "self_attn", None)
        if attention is None:
            continue
        owner = index
        if getattr(attention, "is_kv_shared_layer", False):
            kind = getattr(attention, "layer_type", None)
            owner = next((candidate
                for candidate in reversed(interface.kv_layers)
                if candidate < index and getattr(getattr(
                    interface.layers[candidate], "self_attn", None),
                    "layer_type", None) == kind), None)
        if owner in layers:
            owners[id(attention)] = owner
    base = getattr(interface.config, "_attn_implementation", None) or "sdpa"
    state = _ScoreState(base=base, start=int(start), stop=int(stop),
        skip=int(skip), owners=owners, query_block=int(query_block))
    positions = torch.arange(int(position), int(position) + probe_ids.numel(),
        device=interface.device)
    if not _SCORE_REGISTERED:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface

        AttentionInterface.register(_SCORE, _score_attention)
        AttentionMaskInterface.register(_SCORE, _score_mask)
        _SCORE_REGISTERED = True
    previous = getattr(interface.config, "_attn_implementation_internal", None)
    interface.config._attn_implementation_internal = _SCORE
    token = _SCORE_STATE.set(state)
    try:
        interface.forward(probe_ids, positions, copy_cache(cache))
    finally:
        _SCORE_STATE.reset(token)
        interface.config._attn_implementation_internal = previous
    assert set(state.scores) == set(layers), (
        f"{interface.name}: no attention read compressible layers "
        f"{sorted(set(layers) - set(state.scores))}")
    if scores is not None:
        return {layer: torch.maximum(scores[layer], value)
            for layer, value in state.scores.items()}
    return state.scores


# ---- Translator direction and cache writes -----------------------------------------------------
# Writing a target's KV cache from the translator, and the translator's direction.
#
# * ``HybridWriter`` (sliding-window / gated-delta targets): the translator's per-layer
#   target hidden states -> the target's own input norm + K/V projection (+ K norm) + RoPE;
#   ``prefix_from_cache`` / ``cache_from_prefix`` convert between served prefixes and HF
#   caches.
# * ``Direction``: one source -> target direction of a translator checkpoint, read in
#   chunks with their halo (``shard_source_range``); fast-inference levels
#   (``set_executor``).
# * ``engine_options``: the vLLM engine arguments per target.

class HybridWriter:
    """Translated prefix for a target whose layers are not all plain full attention
    (Gemma 3 / 4: sliding-window and full-attention layers with their own RoPE, K/V heads
    and head size; Qwen 3.5: gated-delta layers with a recurrent state). The served prefix
    is a dict (``vllm_translated_kv`` injects it into vLLM's hybrid cache):

    * ``tokens``: prefix length P (header + document);
    * ``kv`` {layer: [2, n, kv_heads, head_dim]} post-RoPE K/V of the last n prefix tokens
      (``kv_first`` = P - n): every token for full attention, the last ``window`` tokens
      for sliding-window layers (no later token reads further back);
    * ``conv`` / ``ssm`` {layer: tensor}: a gated-delta layer's state after the prefix, as
      HF keeps it (conv: the last ``kernel`` raw inputs [conv_dim, kernel]; ssm:
      [v_heads, k_dim, v_dim], float32).

    K/V use each layer's own frozen input norm, K/V projections, K (and V) norms and its
    layer type's RoPE (HF's own ``apply_rotary_pos_emb``); ``encode.hybrid_cartridge``
    writes a cartridge with them."""

    hybrid = True

    def __init__(self, interface, device):
        self.interface, self.device = interface, device
        self.window = int(getattr(interface.config, "sliding_window", 0) or 0)
        on_device = interface.device == device
        self.kinds, self.norms, self.modules = {}, {}, {}
        for index, layer in enumerate(interface.layers):
            kind = interface.layer_types[index]
            attention = getattr(layer, "self_attn", None)
            if kind == "linear_attention":
                module, kind = layer.linear_attn, "linear"
            elif attention is not None and not getattr(attention, "is_kv_shared_layer", False):
                module, kind = attention, ("sliding" if kind == "sliding_attention" else "full")
            else:
                continue  # K/V shared with an earlier layer: vLLM keeps no cache for it
            norm = layer.input_layernorm
            if not on_device:
                # A target kept on the CPU: device copies of what the writer reads.
                module = copy.copy(module)
                module._modules = {name: child for name, child in module._modules.items()
                                   if name not in ("q_proj", "o_proj", "q_norm")}
                module = copy.deepcopy(module).to(device)
                norm = copy.deepcopy(norm).to(device)
            self.kinds[index], self.modules[index], self.norms[index] = kind, module, norm
        rotary = interface.decoder.rotary_emb
        self.rotary = rotary if on_device else copy.deepcopy(rotary).to(device)
        self._rotary_typed = "layer_type" in inspect.signature(self.rotary.forward).parameters
        attention = next(self.modules[index] for index, kind in self.kinds.items()
                         if kind != "linear")
        self._apply = sys.modules[type(attention).__module__].apply_rotary_pos_emb
        self._apply_pair = "k" in inspect.signature(self._apply).parameters

    def _layer_kv(self, index, x, positions):
        """Post-RoPE K and V ``[tokens, kv_heads, head_dim]`` of one attention layer from
        its normed inputs ``x`` [1, T, width]."""

        attention, kind = self.modules[index], self.kinds[index]
        dim = attention.head_dim
        keys = attention.k_proj(x).view(1, x.shape[1], -1, dim)
        values = keys  # Gemma 4 layers without v_proj: K = V
        if getattr(attention, "v_proj", None) is not None:
            values = attention.v_proj(x).view(1, x.shape[1], -1, dim)
        keys = attention.k_norm(keys) if hasattr(attention, "k_norm") else keys
        values = attention.v_norm(values) if hasattr(attention, "v_norm") else values
        if self._rotary_typed:
            layer_type = "sliding_attention" if kind == "sliding" else "full_attention"
            cos, sin = self.rotary(x, positions.view(1, -1), layer_type)
        else:
            cos, sin = self.rotary(x, positions.view(1, -1))
        if self._apply_pair:  # [B, heads, T, D] layout (Gemma 3, Qwen 3.5)
            keys = keys.transpose(1, 2)
            _, keys = self._apply(keys, keys, cos, sin)
            keys = keys.transpose(1, 2)
        else:  # Gemma 4: one tensor, [B, T, heads, D]
            keys = self._apply(keys, cos, sin, unsqueeze_dim=2)
        return keys[0], values[0]


def prefix_from_cache(interface, cache, total):
    """A ``HybridWriter`` dict from an HF cache of the target holding a
    ``total``-token prefix (HF's own sliding-window layers keep only their last window):
    the parity reference."""

    from speculative_encoding.translator.hf import _is_kv, _state_tensors

    out = {"tokens": int(total), "kv": {}, "kv_first": {}, "conv": {}, "ssm": {}}
    for index, layer in enumerate(cache.layers):
        if interface.layer_types[index] == "linear_attention":
            out["conv"][index] = _state_tensors(layer, "conv_states")[0][0]
            out["ssm"][index] = _state_tensors(layer, "recurrent_states")[0][0]
        elif _is_kv(layer) and getattr(layer, "keys", None) is not None and layer.keys.numel():
            kv = torch.stack((layer.keys[0].transpose(0, 1), layer.values[0].transpose(0, 1)))
            out["kv"][index], out["kv_first"][index] = kv, int(total) - kv.shape[1]
    return out


def cache_from_prefix(interface, prefix):
    """An HF cache of the target holding ``prefix`` ``[layers, 2, tokens, kv_heads,
    head_dim]`` (what vLLM gets)."""

    cache = interface.new_cache()
    for layer in range(prefix.shape[0]):
        keys, values = prefix[layer, 0].transpose(0, 1), prefix[layer, 1].transpose(0, 1)
        cache.update(keys[None], values[None], layer)
    return cache


# Chunked translation: every attention in the translator is a local window, so a target
# token depends on at most decoder_blocks * (window - 1) earlier target tokens and the
# source tokens those read. Each chunk runs with that halo (computed and dropped): the
# same math as one call over the whole document.
_TRANSLATE_HALO = 64  # target tokens; >= 8 decoder blocks x 7 (milestone: 8 blocks, window 8)
_SOURCE_HALO = 32  # source tokens before the first read; >= cross window 8 + 2 encoder blocks x 7


def shard_source_range(bounds, limits, index):
    """First target token (with halo) and the source row range chunk ``index`` reads
    (``encode.decode_states``)."""

    start, end = bounds[index], bounds[index + 1]
    first = max(0, start - _TRANSLATE_HALO)
    source_start = max(0, int(limits[first]) - _SOURCE_HALO)
    return first, source_start, max(int(limits[end - 1]), source_start + 1)


class Direction:
    """The translator for one (source, target) direction: what ``decode_states`` and
    ``encoder_memory`` read. ``level``: the fast-inference level; ``graphs`` and
    ``max_graph_tokens``: CUDA-graph settings (off in the library)."""

    def __init__(self, translator, source, target, level, graphs, max_graph_tokens):
        self.translator, self.source, self.target = translator, source, target
        self.level, self.graphs, self.max_graph_tokens = level, graphs, max_graph_tokens
        self.rows_dtype = torch.bfloat16  # source rows under autocast (tests use FP32 on CPU)


def engine_options(source, target, source_key, target_key):
    """Extra ``vllm.LLM`` arguments per role: text-only serving for vision-language
    checkpoints, Ministral 3's untied embeddings and tekken tokenizer, and Qwen3.5's
    Triton gated-delta prefill."""

    from speculative_encoding.models import CATALOG

    # language_model_only: every modality limit 0, so vLLM runs text-only and skips
    # multimodal profiling (image 0 alone left Gemma 4 / Qwen3.5 profiling a dummy video,
    # which crashed in their HF processors in q4ned53).
    options = {key: {"language_model_only": True}
               for key, model in (("target", target), ("source", source))
               if type(model.model).__name__.endswith("ForConditionalGeneration")}
    for key, name in (("target", target_key), ("source", source_key)):
        if CATALOG[name].family == "ministral3":
            # Ministral 3 checkpoints ship separate lm_head and embed_tokens weights, but the
            # 14B config leaves tie_word_embeddings unset (vLLM 0.24 then ties them: garbage
            # native text in qv7yjxq); and its tokenizer.json needs the regex fix HF applies
            # (catalog: fix_mistral_regex), so vLLM uses the tekken tokenizer instead.
            untied = {"tie_word_embeddings": False}
            options.setdefault(key, {}).update(
                hf_overrides={**untied, "text_config": dict(untied)}, tokenizer_mode="mistral")
        if CATALOG[name].family == "qwen3.5":
            # vLLM 0.24 on Hopper picks FlashInfer's JIT-compiled GDN prefill; its first-run
            # compile hung the engine for 7.5 h after torch.compile in wlgyg2q. Triton/FLA is
            # the same chunked kernel the HF side runs.
            options.setdefault(key, {}).update(
                additional_config={"gdn_prefill_backend": "triton"})
    return options


def set_executor(translator, level, compiled):
    from speculative_encoding.translator import fast_inference

    translator.set_compiled_attention(compiled)
    fast_inference.install(level)


# ---- Encode stage ------------------------------------------------------------------------------
# The encode stage: translate a document into a target's KV cache and build per-question
# compressed caches.

def chunk_plan(doc, chunk):
    """``decode_states``'s chunks of at most ``chunk`` target tokens: one device per chunk
    (all the document's own) and the target-token bounds."""

    tokens = int(doc.target_positions.numel())
    count = max(1, -(-tokens // chunk))
    bounds = [round(tokens * index / count) for index in range(count + 1)]
    return [doc.target_embeds.device] * count, bounds


def prefix_tokens(prefix) -> int:
    """Tokens a served prefix holds (a tensor ``[layers, 2, tokens, ...]`` or a hybrid dict)."""

    return int(prefix["tokens"]) if isinstance(prefix, dict) else int(prefix.shape[2])


def prefix_to(prefix, device, non_blocking=False):
    """A served prefix (tensor, or ``HybridWriter``'s dict of per-layer tensors) on
    ``device``."""

    if torch.is_tensor(prefix):
        return prefix.to(device, non_blocking=non_blocking)
    def move(value):
        return value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value

    return {key: {index: move(value) for index, value in part.items()}
            if isinstance(part, dict) else part for key, part in prefix.items()}


def prefix_bytes(prefix):
    if torch.is_tensor(prefix):
        return prefix.numel() * prefix.element_size()
    return sum(value.numel() * value.element_size() for part in prefix.values()
               if isinstance(part, dict) for value in part.values() if torch.is_tensor(value))


def lean_target(interface, device):
    """What the translator worker needs from a target kept on the CPU: device copies of its
    input embeddings (for ``prompts.document_tensors``) and RoPE. The CPU model itself is
    left untouched, so its own forward passes (header K/V, self check) still run on the
    CPU."""

    embedding = copy.deepcopy(interface.model.get_input_embeddings()).to(device)
    rotary = copy.deepcopy(interface.decoder.rotary_emb).to(device)
    return SimpleNamespace(embed=lambda ids: embedding(ids)), rotary


def decode_states(direction, doc, chunk, timings=None, memory=None):
    """The translator's decoder states ``[T, streams, width]`` (after the target's tail
    blocks, before its output head), in chunks with their halo (``shard_source_range``): what
    the query compressor reads.

    ``memory``: the whole document's encoder output ``[source tokens, streams, width]``
    (``encoder_memory``, e.g. a stored ``EncodedMemory``): each chunk reads its
    source range of it and the encoder does not run (``doc.source_rows`` is then unused).
    Default ``None``: the encoder runs on each chunk's rows."""

    from speculative_encoding.translator.model import _key

    devices, bounds = chunk_plan(doc, chunk)
    limits, t = doc.limits.cpu(), direction.translator
    encoder, decoder, head = t.encoder, t.decoder, t.head(direction.target)
    state, device, out = encoder.states[_key(direction.source)], doc.target_embeds.device, []
    with torch.no_grad(), D.autocast(device):
        for index in range(len(devices)):
            start, end = bounds[index], bounds[index + 1]
            first, source_start, source_end = shard_source_range(bounds, limits, index)
            # Encoder vs decoder split (the paper's "encoder output cached" time is the
            # decoder's).
            if timings is not None:
                D.synchronize(device)
                started = time.perf_counter()
            if memory is None:
                rows = doc.source_rows[source_start:source_end].to(device, direction.rows_dtype)
                part, _ = encoder(direction.source, rows, state)
            else:
                part = memory[source_start:source_end].to(device)
            if timings is not None:
                D.synchronize(device)
                timings["encoder_s"] = timings.get("encoder_s", 0.0) + time.perf_counter() - started
                started = time.perf_counter()
            states, _ = decoder(
                head.start(doc.target_embeds[first:end]), state=head.state, memory=part,
                source_state=state,
                source_positions=doc.source_positions[source_start:source_end].to(device),
                target_positions=doc.target_positions[first:end].to(device),
                limits=(doc.limits[first:end] - source_start).clamp_min(0).to(device),
                extra_blocks=head.tail)
            if timings is not None:
                D.synchronize(device)
                timings["decoder_s"] = timings.get("decoder_s", 0.0) + time.perf_counter() - started
            out.append(states[start - first:])
            del part, states
    return torch.cat(out)


def encoder_memory(direction, rows, chunk, device):
    """The translator encoder's output over the whole source document, ``[source tokens,
    streams, width]`` on the CPU (what ``decode_states(..., memory=)`` reads). It depends
    only on the source model and the document, never on the target. Up to ``chunk`` source
    tokens run in one encoder call (then ``decode_states`` of a document within one chunk
    gets bit-identical memory); longer documents in chunks of ``chunk`` with
    ``_SOURCE_HALO`` tokens of left context computed and dropped: the encoder is
    causal with local windows (it reads ``encoder_blocks * (window - 1)`` earlier tokens),
    so this is the same math. ``rows``: ``[tokens, streams, width]`` tensor or
    ``HostRows``."""

    from speculative_encoding.translator.model import _key

    encoder = direction.translator.encoder
    config = encoder.config
    halo = max(_SOURCE_HALO, config.encoder_blocks * (config.window - 1))
    state, tokens, out = encoder.states[_key(direction.source)], int(rows.shape[0]), []
    with torch.no_grad(), D.autocast(device):
        for start in range(0, tokens, chunk):
            first, end = max(0, start - halo), min(tokens, start + chunk)
            part, _ = encoder(
                direction.source, rows[first:end].to(device, direction.rows_dtype), state)
            out.append(part[start - first:].to("cpu"))
            del part
    return torch.cat(out)


def hybrid_cartridge(writer, header_cache, slot_hidden, last_hidden, start, doc_states):
    """A hybrid target's served cartridge (a ``HybridWriter`` dict) of ``start + M``
    positions: full-attention layers hold the header and the ``M`` slots (positions
    ``start..start+M``); sliding-window layers the header and the document's real last
    tokens (``last_hidden``, at the positions right before the question), cut to the window;
    gated-delta layers the full document's states (``doc_states``). K/V come from
    ``HybridWriter``'s per-layer writer (the header's from ``header_cache``): no gated-delta
    pass over the slots, whose states the cartridge would not keep."""

    device, dtype = slot_hidden.device, writer.interface.dtype
    count, keep = int(slot_hidden.shape[0]), int(last_hidden.shape[0])
    total = start + count
    prefix = {"tokens": total, "kv": {}, "kv_first": {},
              "conv": dict(doc_states["conv"]), "ssm": dict(doc_states["ssm"])}
    full_positions = start + torch.arange(count, device=device)
    sliding_positions = start + count - keep + torch.arange(keep, device=device)
    for index, kind in writer.kinds.items():
        if kind == "linear":
            continue
        rows, positions = slot_hidden, full_positions
        if kind != "full":
            rows, positions = last_hidden, sliding_positions
        x = writer.norms[index](rows[:, index].to(dtype).unsqueeze(0))
        keys, values = writer._layer_kv(index, x, positions)
        layer = header_cache.layers[index]
        header = torch.stack((layer.keys[0].transpose(0, 1), layer.values[0].transpose(0, 1)))
        kv = torch.cat((header.to(keys.dtype), torch.stack((keys, values))), dim=1)
        if kind == "sliding" and writer.window:
            kv = kv[:, -writer.window:]
        prefix["kv"][index], prefix["kv_first"][index] = kv, total - kv.shape[1]
    return prefix


def selection_scores(scores):
    """The per-document-token score ``compressed_prefix`` ranks by, from ``score``'s
    ``{layer: [kv_heads, T]}``: each layer's max over its K/V heads, normalized to sum 1 over
    the document, then averaged over layers (``[T]``)."""

    return torch.stack([value.amax(0) / value.amax(0).sum().clamp_min(1e-12)
                        for value in scores.values()]).mean(0)


def compressed_prefix(target, translator, compressor, args, states, doc, sample, after_ids,
                      header_cache, translated_cache, doc_states, writer=None, scores_out=None):
    """One question's served cartridge (the query compressor's) and its timings.

    Slots: the ``ceil(T / ratio)`` document rows the target's question attention reads most
    in the full translated cache (KVzip scoring pooled by ``selection_scores``). The
    compressor reads the translated decoder states and the question, the target head maps
    the slots to layer inputs, and the target's own writer turns them into K/V at
    consecutive positions after the header (a paged server's prefix; the question then
    follows at header + M). Hybrid targets: sliding-window layers hold the document's real
    last ``min(window, M)`` tokens at the positions just before the question, and
    gated-delta layers the full document's state (``doc_states``), as the question reads
    them in the compressor's training.

    ``scores_out``: an optional dict that receives ``"pooled"``, the ``[T]`` selection
    scores the slots are the top-k of (float32, CPU; read-only side output, the selection
    and the prefix are unchanged)."""

    from speculative_encoding.translator import compressor as C

    device = states.device
    header = list(sample.target.header_ids)
    start, tokens = len(header), int(states.shape[0])
    end = start + tokens
    doc_ids = torch.tensor(sample.target.doc_ids, device=device)
    head = translator.head(args.target)
    D.synchronize(device)
    started = time.perf_counter()
    scores = score(target, translated_cache, start, end, list(after_ids), end)
    pooled = selection_scores(scores)
    count = int(C.slot_tokens(tokens, ratio=args.compressor_ratio).numel())
    slots = pooled.topk(min(count, pooled.numel())).indices.sort().values
    D.synchronize(device)
    snap_s = time.perf_counter() - started
    if scores_out is not None:  # after the timed selection, so snap_s is unchanged
        scores_out["pooled"] = pooled.detach().float().cpu()
    started = time.perf_counter()
    question = target.layer_inputs(torch.tensor(header + list(after_ids), device=device))[start:]
    with torch.no_grad(), D.autocast(device):
        rows = compressor.question_streams(
            args.target, C.question_rows(translator.encoder, args.target, question.to(states.dtype)),
            streams=states.shape[1])
        slot_states = compressor(
            states, rows, slots, state=head.state, slot_positions=start + slots,
            question_positions=torch.arange(end, end + rows.shape[0], device=device),
            memory_positions=doc.target_positions.to(device))
        slot_hidden = head(slot_states)
    D.synchronize(device)
    compress_s = time.perf_counter() - started
    started = time.perf_counter()
    count = int(slots.numel())
    with torch.no_grad():
        if writer is None:  # all full attention: the target's own writer on the slots, one tensor
            cache, _ = target.write_prefix(
                doc_ids.index_select(0, slots), slot_hidden.to(target.dtype), header_cache,
                start + torch.arange(count, device=device))
            prefix = prefix_from_cache(target, cache, start + count)
            del cache
            prefix = torch.stack([prefix["kv"][index] for index in sorted(prefix["kv"])])
        else:
            keep = min(writer.window, count) if writer.window else count
            with D.autocast(device):
                last = head(states[tokens - keep:])
            prefix = hybrid_cartridge(writer, header_cache, slot_hidden, last, start, doc_states)
    D.synchronize(device)
    write_s = time.perf_counter() - started
    return prefix, slots, {"snap_s": snap_s, "compress_s": compress_s, "cartridge_write_s": write_s}


# ---- Stored encoder output ---------------------------------------------------------------------
# Encode once, decode for any target: store the translator encoder's output instead of the
# source hidden states.
#
# The translator's encoder reads only the source model's hidden states of the document (and that
# source's learned state table), so its output ``memory``
# ``[source tokens, source streams, width]`` is the same for every target. Its width is the
# translator's (768 for the paper milestone) instead of the source's (3840-5120), so it is 5-7x
# smaller than the hidden states it replaces, and decoding from it skips the source prefill and
# the encoder.
#
# What decoding for a target needs besides ``memory`` is the source side of the document's
# sample: the source token ids and their character spans (the alignment to any target's tokens is
# computed from the character spans of both tokenizations, ``prompts.align``), and the document
# text itself (the target tokenizes it at decode time). An :class:`EncodedMemory` holds exactly
# that, so one stored memory serves every target the translator has a head for:
#
#     mem = enc_a.encode_memory(text)  # source prefill + encoder, once (enc_a: source -> any target)
#     mem.save("doc.memory.safetensors", dtype=torch.bfloat16)
#     mem = EncodedMemory.load("doc.memory.safetensors")
#     doc_b = enc_b.encode(memory=mem)  # enc_b: same source -> target B; no source model needed
#     doc_c = enc_c.encode(memory=mem, questions=[q], ratios=[16])
#
# ``SpeculativeEncoder(..., load_source=False)`` builds a target-side encoder that never loads
# the source LLM.

FORMAT = "speculative-encoding-memory/1"


def translator_fingerprint(translator, source: str) -> str:
    """A short id of the source-side translator weights the memory was computed with (the
    source's learned state table and input gains, in bf16 so an fp32 and a bf16 copy of one
    milestone agree)."""

    from speculative_encoding.translator.model import _key

    encoder, digest = translator.encoder, hashlib.sha256()
    for table in (encoder.states[_key(source)], encoder.gains[_key(source)]):
        digest.update(
            table.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16)
            .numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass
class EncodedMemory:
    """One document's translator encoder output for one source model, and what any target needs
    to decode it."""

    source: str  # source model key (catalog)
    memory: torch.Tensor  # [source doc tokens, source streams, translator width], CPU
    # the document text the source read (after ``chat_row``'s strip and the source-token
    # truncation)
    document: str
    source_header_ids: list[int]
    source_doc_ids: list[int]
    source_doc_spans: list[tuple[int, int]]  # character spans of the source tokens in ``document``
    role: str = "system"  # the chat role the document was rendered in (``prompts.chat_row``)
    fingerprint: str = ""  # translator_fingerprint at encode time
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return int(self.memory.shape[0])

    @property
    def bytes(self) -> int:
        return self.memory.numel() * self.memory.element_size()

    def check(self, translator, source: str) -> None:
        if source != self.source:
            raise ValueError(
                f"memory was encoded from source {self.source}; this encoder's source is {source}")
        if self.memory.shape[0] != len(self.source_doc_ids):
            raise ValueError(
                f"memory has {self.memory.shape[0]} tokens; "
                f"the source document has {len(self.source_doc_ids)}")
        geometry, width = translator.sources[source], translator.config.width
        if tuple(self.memory.shape[1:]) != (geometry.streams, width):
            raise ValueError(
                f"memory shape {tuple(self.memory.shape)}; "
                f"this translator expects [tokens, {geometry.streams}, {width}]")
        if self.fingerprint and self.fingerprint != translator_fingerprint(translator, source):
            raise ValueError(
                "memory was encoded with different translator weights (fingerprint mismatch)")

    def sample(self, target_tokenizer, config, *, question: str = "") -> se_prompts.Sample:
        """The document's sample for a target tokenizer: ``prompts.make_sample(kind="task")``'s
        result, with the stored source side in place of re-tokenizing with the source tokenizer
        (the two are equal)."""

        row = se_prompts.chat_row(
            "api", "document", self.document, question=question, role=self.role)
        source = se_prompts.Rendered(
            list(self.source_header_ids), list(self.source_doc_ids),
            [tuple(span) for span in self.source_doc_spans], [])
        target = se_prompts.render(target_tokenizer, row, document=self.document, target_text=None)
        target.target_ids = target.target_ids[: config["max_target_tokens"]]
        if not target.doc_ids or not target.continuation_ids:
            raise ValueError("empty document after target tokenization")
        sample = se_prompts.Sample(
            row, source, target, se_prompts.align(source.doc_spans, target.doc_spans), "task")
        sample.source_key = self.source
        return sample

    def save(self, path, *, dtype=None) -> Path:
        """One safetensors file: the ``memory`` tensor and a JSON header with everything else.
        ``dtype`` casts the memory (``torch.bfloat16`` halves an fp32 memory; a GPU encoder
        under bf16 autocast already returns bf16)."""

        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        memory = self.memory if dtype is None else self.memory.to(dtype)
        info = {
            "source": self.source, "document": self.document,
            "source_header_ids": list(self.source_header_ids),
            "source_doc_ids": list(self.source_doc_ids),
            "source_doc_spans": [list(span) for span in self.source_doc_spans],
            "role": self.role, "fingerprint": self.fingerprint, "meta": self.meta}
        save_file(
            {"memory": memory.contiguous()}, str(path),
            metadata={"format": FORMAT, "info": json.dumps(info)})
        return path

    @classmethod
    def load(cls, path) -> "EncodedMemory":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            header = handle.metadata()
            if header.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            memory = handle.get_tensor("memory")
        info = json.loads(header["info"])
        info["source_doc_spans"] = [tuple(span) for span in info["source_doc_spans"]]
        return cls(memory=memory, **info)


def build(direction, sample, document, rows, chunk, device) -> EncodedMemory:
    """``EncodedMemory`` of ``sample``'s document from the source rows
    (``[tokens, streams, width]`` or ``HostRows``). ``document``: the text ``make_sample``
    rendered (the row's document after its source-token truncation)."""


    if rows.shape[0] != len(sample.source.doc_ids):
        raise ValueError(
            f"source rows have {rows.shape[0]} tokens; "
            f"the document has {len(sample.source.doc_ids)} source tokens")
    memory = encoder_memory(direction, rows, chunk, device)
    return EncodedMemory(
        source=direction.source, memory=memory, document=document,
        source_header_ids=list(sample.source.header_ids),
        source_doc_ids=list(sample.source.doc_ids),
        source_doc_spans=[tuple(span) for span in sample.source.doc_spans],
        role=sample.row["translated_role"],
        fingerprint=translator_fingerprint(direction.translator, direction.source),
        meta={"encoder_chunk": int(chunk), "dtype": str(memory.dtype).removeprefix("torch.")})

