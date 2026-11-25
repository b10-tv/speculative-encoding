"""The model catalog (pinned Hugging Face checkpoints), model loading, source hidden states,
generation settings and long-context engine options."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from speculative_encoding.translator.hf import ModelInterface
import functools
import json
import os
from speculative_encoding import settings


# ---- Model catalog -----------------------------------------------------------------------------
# Pinned Hugging Face checkpoints the unified translator supports.
#
# The catalog records only checkpoint identities. Loading uses the model's own
# Hugging Face architecture and then runs ``ModelInterface.self_check``.

@dataclass(frozen=True)
class CatalogModel:
    hf_id: str
    revision: str
    family: str
    # bf16 output tolerance for the self-check. Raise only after an fp32 check
    # passes at ~1e-5, which shows the cache path is exact and the bf16 gap is
    # recomputation noise.
    bf16_logit_tolerance: float = 5e-2


CATALOG = {
    "qwen3-4b": CatalogModel(
        "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "qwen3"),
    "qwen3-8b": CatalogModel(
        "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen3"),
    "qwen3-14b": CatalogModel(
        "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "qwen3"),
    "qwen35-4b": CatalogModel(
        "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "qwen3.5"),
    "qwen35-9b": CatalogModel(
        "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "qwen3.5"),
    "gemma3-4b": CatalogModel(
        "google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767",
        "gemma3"),
    "gemma3-12b": CatalogModel(
        "google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "gemma3"),
    "llama31-8b": CatalogModel(
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659", "llama3"),
    # fp32 self-check: cache 1.9e-6, logits 3.6e-6, continuation 5.7e-6,
    # spliced 6.1e-6 (2026-11-16); bf16 spliced recomputation differs by
    # 7.1e-2, so its bf16 tolerance is 1e-1.
    "gemma4-12b": CatalogModel(
        "google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d",
        "gemma4", bf16_logit_tolerance=1e-1),
    "gemma4-e2b": CatalogModel(
        "google/gemma-4-E2B-it", "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "gemma4"),
    "gemma4-e4b": CatalogModel(
        "google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
        "gemma4"),
    # Phi-4 14B uses the Phi3 decoder: 40 full-attention layers, fused
    # qkv/gate_up projections, 16k context.
    "phi4-14b": CatalogModel(
        "microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25", "phi4"),
    "ministral3-8b": CatalogModel(
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "f6fae9795746f63c9be8344932f01275f3c63734", "ministral3"),
    "ministral3-14b": CatalogModel(
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "3cea74c1ebaf5ce5f5a2553de470e2ceab825142", "ministral3"),
    "granite33-2b": CatalogModel(
        "ibm-granite/granite-3.3-2b-instruct",
        "707f574c62054322f6b5b04b6d075f0a8f05e0f0", "granite3"),
    "granite33-8b": CatalogModel(
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", "granite3"),
    # Granite 4.1/4.2 are dense Granite decoders. The Granite 4.0 "-h" Mamba
    # hybrids are left out: their HF Mamba code cannot continue a cache over
    # several tokens (see ModelInterface.continues_cache).
    "granite42-3b": CatalogModel(
        "ibm-granite/granite-4.2-3b",
        "4e9c275e232f10211ec88644d106a1f6a1b60e06", "granite4"),
    "granite42-8b": CatalogModel(
        "ibm-granite/granite-4.2-8b",
        "a7042144125ce5ae64f89610e8289647fc000b9a", "granite4"),
}


def load_model(key: str, *, device="cuda", token: str | None = None,
        dtype=torch.bfloat16, check: bool = True):
    """Load a catalog checkpoint and return ``(ModelInterface, tokenizer)``."""

    from transformers import (AutoConfig, AutoModelForCausalLM,
        AutoModelForImageTextToText)

    spec = CATALOG[key]
    pinned = dict(revision=spec.revision, token=token, trust_remote_code=False)
    config = AutoConfig.from_pretrained(spec.hf_id, **pinned)
    architectures = tuple(getattr(config, "architectures", None) or ())
    multimodal = any(name.endswith("ForConditionalGeneration")
        for name in architectures)
    model_class = (AutoModelForImageTextToText if multimodal
        else AutoModelForCausalLM)
    try:
        model = model_class.from_pretrained(
            spec.hf_id, dtype=dtype, attn_implementation="sdpa", **pinned)
    except ValueError as error:
        if "does not support an attention implementation" not in str(error):
            raise
        # Multimodal wrappers can contain non-text towers without SDPA support.
        # The translator reads and writes only the text decoder, which keeps
        # SDPA; unused towers use eager attention.
        towers = getattr(config, "sub_configs", {})
        implementation = {"text_config": "sdpa", **{
            name: "eager" for name in towers if name != "text_config"}}
        model = model_class.from_pretrained(
            spec.hf_id, dtype=dtype, attn_implementation=implementation,
            **pinned)
    model = model.to(device).requires_grad_(False).eval()
    tokenizer = load_tokenizer(key, token=token)
    interface = ModelInterface(model, key, check=check,
        bf16_logit_tolerance=spec.bf16_logit_tolerance)
    return interface, tokenizer


def load_tokenizer(key: str, *, token: str | None = None):
    """Load the pinned tokenizer for ``key`` with checkpoint-specific tokenizer
    repairs."""

    from transformers import AutoTokenizer

    spec = CATALOG[key]
    options = {"fix_mistral_regex": True} if spec.family == "ministral3" else {}
    return AutoTokenizer.from_pretrained(
        spec.hf_id, revision=spec.revision, token=token,
        trust_remote_code=False, **options)


__all__ = ["CATALOG", "CatalogModel", "load_model", "load_tokenizer"]


# ---- Loading models and source rows ------------------------------------------------------------
# Model loading, source hidden states, generation settings and long-context engine
# options.

@functools.lru_cache(maxsize=None)
def catalog_geometry(key):
    """``(source streams, width)`` of a catalog checkpoint, read from its pinned config."""

    from transformers import AutoConfig

    spec = CATALOG[key]
    config = AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision,
                                        token=os.environ.get("HF_TOKEN"), trust_remote_code=False)
    text = config.get_text_config(decoder=True)
    return int(text.num_hidden_layers) + 1, int(text.hidden_size)


def load_catalog(key, device):
    """``(checkpoint name, ModelInterface, tokenizer)``; loading runs the interface self-check."""


    interface, tokenizer = load_model(key, device=device, token=os.environ.get("HF_TOKEN"))
    geometry = interface.source_geometry()
    if (geometry.streams, geometry.width) != catalog_geometry(key):
        raise ValueError(f"{key}: loaded decoder has {geometry.streams} streams of width "
                         f"{geometry.width}; its config says {catalog_geometry(key)}")
    report = {"model": key, "layers": len(interface.layers), "kv_layers": len(interface.kv_layers),
              "cache_layers": interface.cache_layers,
              "all_full_attention": interface.all_full_attention, **(interface.check_report or {})}
    print("model interface: " + json.dumps(report, sort_keys=True, default=str), flush=True)
    return CATALOG[key].hf_id, interface, tokenizer


def generation_config(model, maximum=512):
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (tuple, list)) else [eos]
    return {
        "eos_token_ids": [int(value) for value in eos if value is not None],
        "pad_token_id": model.generation_config.pad_token_id,
        "max_new_tokens": maximum,
        "do_sample": False,
    }


_PINNED = {}


def pinned(key, count, dtype, pin=True):
    """A reused pinned host buffer of at least ``count`` elements. Grown with 25% headroom:
    allocating 30+ GB of pinned memory takes about a minute, so it happens rarely and, in
    ``produce``, before any timer starts."""

    buffer = _PINNED.get(key)
    if buffer is None or buffer.numel() < count or buffer.dtype != dtype:
        _PINNED.pop(key, None)
        buffer = _PINNED[key] = torch.empty(int(count * 1.25), dtype=dtype, pin_memory=pin)
    return buffer


class HostRows:
    """Source rows ``[tokens, layers + 1, width]`` kept on the host stream-major (each
    layer's rows contiguous, so the device-to-host copies during the prefill and the
    host-to-device copies of a chunk are plain DMA copies). Indexing a token range returns
    those rows on ``device`` in the usual token-major layout; ``to(device)`` moves them
    all."""

    def __init__(self, host, start, device):
        # host: [streams, tokens, width]; start: the first document token
        self.host, self.start, self.device = host, start, device
        self.shape = torch.Size((host.shape[1] - start, host.shape[0], host.shape[2]))
        self.dtype = host.dtype

    def numel(self):
        return self.shape.numel()

    def element_size(self):
        return self.host.element_size()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.step in (None, 1), "contiguous token ranges only"
        first, last, _ = index.indices(self.shape[0])
        out = torch.empty((max(0, last - first), self.shape[1], self.shape[2]), dtype=self.dtype,
                          device=self.device)
        for stream in range(self.shape[1]):
            rows = self.host[stream, self.start + first:self.start + last]
            out[:, stream].copy_(rows, non_blocking=True)
        return out

    def to(self, device=None, dtype=None, non_blocking=False):
        out = self[0:self.shape[0]]
        return out.to(device or out.device, dtype or out.dtype)


def source_rows_host(interface, header_ids, doc_ids, device):
    """The source's hidden states (the raw embedding output, then every layer's output, header
    dropped) captured into a reused pinned host buffer one layer at a time during the
    source's prefill: no stack of every layer on the GPU (a Gemma 3 source at 128k needs
    47 GB for it, twice during the stack). Returns ``HostRows``."""

    device = torch.device(device)
    ids = torch.tensor(list(header_ids) + list(doc_ids), device=device)
    tokens, streams, width = int(ids.numel()), len(interface.layers) + 1, interface.width
    count = tokens * streams * width
    host = pinned("rows", count, interface.dtype, device.type == "cuda")[:count]
    host = host.view(streams, tokens, width)
    embedded = interface.embed(ids).reshape(tokens, width).to(interface.dtype)
    host[0].copy_(embedded, non_blocking=True)
    span = {"start": 0}

    def capture(index):
        def after(module, args, output):
            value = (output[0] if isinstance(output, tuple) else output)[0]
            rows = host[index + 1, span["start"]:span["start"] + value.shape[0]]
            rows.copy_(value, non_blocking=True)
        return after

    # Long prompts prefill in chunks against the source's own cache (as vLLM does): a
    # sliding-window model's HF mask is otherwise [tokens, tokens] (47 GB for Gemma 3 at
    # 128k). Shorter prompts keep the single forward.
    chunk = settings.current().source_chunk
    if interface.all_full_attention:
        # No sliding or recurrent layers: one causal forward needs no mask, so SDPA takes its
        # flash kernel; chunks against a cache carry an explicit mask and fall back to a
        # slower kernel (4-7x slower than vLLM's prefill at 128k).
        chunk = max(chunk, tokens)
    hooks = [layer.register_forward_hook(capture(index))
             for index, layer in enumerate(interface.layers)]
    try:
        if tokens <= chunk:
            interface.decoder(input_ids=ids.reshape(1, -1), use_cache=False)
        else:
            cache = interface.new_cache()
            for start in range(0, tokens, chunk):
                span["start"] = start
                interface.decoder(input_ids=ids[start:start + chunk].reshape(1, -1),
                                  past_key_values=cache, use_cache=True)
            del cache
    finally:
        for hook in hooks:
            hook.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return HostRows(host, len(header_ids), device)


def long_context_options(model, revision, options, max_model_len):
    """``vllm.LLM`` options for documents past the model's ``max_position_embeddings``: vLLM
    sizes its RoPE table from it (Llama 3.1's llama3 scaling does not depend on it), so it is
    raised to the engine length; YaRN models get their length from the factor instead."""

    from transformers import AutoConfig

    options = dict(options)
    config = AutoConfig.from_pretrained(model, revision=revision)
    text = getattr(config, "text_config", None) or config
    scaling = getattr(text, "rope_scaling", None) or getattr(text, "rope_parameters", None) or {}
    if scaling.get("rope_type") != "yarn" and int(text.max_position_embeddings) < max_model_len:
        overrides = dict(options.get("hf_overrides") or {})
        overrides["max_position_embeddings"] = max_model_len
        if text is not config:
            overrides["text_config"] = {**overrides.get("text_config", {}),
                                        "max_position_embeddings": max_model_len}
        options["hf_overrides"] = overrides
    return options

