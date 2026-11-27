"""vLLM KV connectors: inject a translated or compressed prefix into a request's KV blocks
(``TranslatedKVConnector``, ``HybridKVConnector``), and reuse one document across models
(``FirstHitMultiConnector``, a prototype)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata)
from speculative_encoding import settings
from speculative_encoding.encode import prefix_tokens
import hashlib
import threading
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


# ---- Translated-prefix connector ---------------------------------------------------------------
# vLLM KV connectors that inject a translated (or compressed) prefix into a request's paged KV
# blocks.
#
# ``TranslatedKVConnector`` (full-attention targets) and ``HybridKVConnector`` (sliding-window and
# gated-delta targets, with vLLM's hybrid KV cache manager) implement vLLM 0.24's v1 KV-connector
# interface like the reference ``ExampleConnector``, except that the prefix is held in ``STORE``
# (keyed by name) instead of a file per layer, and any prefix length works. vLLM loads this module
# by name (``kv_connector_module_path``), so producers must put prefixes into this module's
# ``STORE``, not a copy. Run vLLM with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the connector's
# worker side shares the process; with tensor parallelism > 1 each worker loads the prefix from
# ``settings.vllm_export_dir`` and keeps its own KV heads.

# Injected prefixes by name: [layers, 2, tokens, kv_heads, head_dim] or a HybridWriter dict.
STORE: dict[str, torch.Tensor] = {}


@dataclass
class InjectMetadata(KVConnectorMetadata):
    # (STORE name, first token to load, prefix tokens, block ids per KV cache group)
    loads: list = field(default_factory=list)
    # (STORE name, tokens, block ids per group): parity reads vLLM's own cache back
    captures: list = field(default_factory=list)


def _layer(name):
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


class TranslatedKVConnector(KVConnectorBase_V1):
    """A request with ``kv_transfer_params={"inject": name, "tokens": n}`` takes its first n
    tokens' K/V from ``STORE[name]``.

    ``STORE[name]`` is either one tensor ``[layers, 2, tokens, kv_heads, head_dim]`` (every layer
    plain full attention) or a ``translate.HybridWriter`` dict (per layer K/V of the last n
    tokens, gated-delta states), written into each layer's own KV cache group
    (``HybridKVConnector``: vLLM's hybrid allocator). A request with ``{"capture": name}`` gets
    its prompt's cache copied back into ``STORE[name]`` (same dict) after its forward pass.
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._pending, self._captures, self._caches = {}, {}, {}
        # Layer -> KV cache group and each group's block size (one group unless hybrid).
        self._group, self._group_block_size = {}, {}
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
        for group_index, group in enumerate(groups):
            self._group_block_size[group_index] = int(group.kv_cache_spec.block_size)
            for name in group.layer_names:
                self._group[_layer(name)] = group_index

    # Scheduler side.
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if "inject" not in params:
            return 0, False
        return max(0, int(params["tokens"]) - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params or {}
        if num_external_tokens > 0:
            # Tokens before ``start`` came from vLLM's own prefix cache: not reloaded.
            start = int(params["tokens"]) - num_external_tokens
            self._pending[request.request_id] = {**params, "start": start}
        if "capture" in params:
            self._captures[request.request_id] = params["capture"]

    def build_connector_meta(self, scheduler_output):
        meta = InjectMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            block_ids = tuple(list(ids) for ids in request.block_ids)
            params = self._pending.pop(request.req_id, None)
            if params is not None:
                start = int(params.get("start", 0))
                meta.loads.append((params["inject"], start, int(params["tokens"]), block_ids))
            name = self._captures.pop(request.req_id, None)
            if name is not None:
                tokens = len(request.prompt_token_ids)
                scheduled = scheduler_output.num_scheduled_tokens[request.req_id]
                assert scheduled == tokens, "capture needs a one-step prefill"
                meta.captures.append((name, tokens, block_ids))
        self._resumed_loads(scheduler_output, meta)
        assert not self._pending, f"injected requests not scheduled: {list(self._pending)}"
        return meta

    def _resumed_loads(self, scheduler_output, meta):
        """Loads for preempted requests resumed this step. vLLM <= 0.24 (V1 model runner)
        lists them in ``scheduled_cached_reqs`` with their whole block table in
        ``new_block_ids``; their blocks were freed, so the prefix is injected again. Newer vLLM
        re-sends them as new requests (handled above)."""

        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", None) or ()
        for index, req_id in enumerate(cached.req_ids):
            if req_id not in resumed or req_id not in self._pending:
                continue
            params = self._pending.pop(req_id)
            block_ids = tuple(list(ids) for ids in cached.new_block_ids[index])
            start = int(params.get("start", 0))
            meta.loads.append((params["inject"], start, int(params["tokens"]), block_ids))

    # Worker side.
    def register_kv_caches(self, kv_caches):
        for name, cache in kv_caches.items():
            if isinstance(cache, (list, tuple)):  # gated-delta: [conv state, recurrent state]
                self._caches[_layer(name)] = list(cache)
                continue
            # FlashAttention / FlashInfer / Triton layout: [blocks, 2 (K, V), block_size, kv_heads,
            # head_dim]. The hybrid allocator may split each of its blocks into kernel blocks.
            group = self._group.get(_layer(name), 0)
            block_size = self._group_block_size.get(group, self._block_size)
            shape = (name, tuple(cache.shape), block_size)
            assert cache.ndim == 5 and cache.shape[1] == 2, shape
            assert block_size % cache.shape[2] == 0, shape
            assert self._group or cache.shape[2] == block_size, shape
            self._caches[_layer(name)] = cache

    def _slots(self, layer, block_ids, first, stop, device):
        """Token positions ``[first, stop)`` held in real blocks of ``layer``'s group (a
        sliding-window group's blocks before the window are vLLM's null block 0) and their
        (kernel block, offset) in its cache."""

        group = self._group.get(layer, 0)
        size = self._group_block_size.get(group, self._block_size)
        kernel = self._caches[layer].shape[2]
        positions = torch.arange(first, stop, device=device)
        blocks = torch.tensor(block_ids[group], device=device)[positions // size]
        real = blocks != 0
        positions, blocks = positions[real], blocks[real]
        kernel_blocks = blocks * (size // kernel) + (positions % size) // kernel
        return positions, kernel_blocks, positions % kernel

    def start_load_kv(self, forward_context, **kwargs):
        tensors = [cache for cache in self._caches.values() if torch.is_tensor(cache)]
        for name, start, tokens, block_ids in self._get_connector_metadata().loads:
            device = tensors[0].device
            if name not in STORE:  # a tensor-parallel worker: load its own heads of the export
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

                rank = get_tensor_model_parallel_rank()
                size = get_tensor_model_parallel_world_size()
                export_dir = settings.current().vllm_export_dir
                if export_dir is None:
                    raise RuntimeError(
                        f"{name!r} is not in this worker's STORE: with tensor_parallel_size > 1 "
                        "each worker loads <settings.vllm_export_dir>/<name>.pt, and that "
                        "setting (environment variable PMT_VLLM_EXPORT_DIR) is unset")
                prefix = torch.load(Path(export_dir) / f"{name}.pt")
                heads = prefix.shape[3] // size
                assert heads == tensors[0].shape[3], (tuple(prefix.shape), size, tensors[0].shape)
                STORE.clear()  # one row at a time per worker
                own = prefix[:, :, :, rank * heads : (rank + 1) * heads]
                STORE[name] = own.contiguous().to(device)
            prefix = STORE[name]  # on the GPU, or on the CPU (copied one layer at a time)
            if torch.is_tensor(prefix):
                positions = torch.arange(start, tokens)
                ids = torch.tensor(block_ids[0])
                slots = ids[positions // self._block_size] * self._block_size
                slots = (slots + positions % self._block_size).to(device)
                blocks, offsets = slots // self._block_size, slots % self._block_size
                for layer, cache in self._caches.items():
                    kv = prefix[layer, :, start:tokens].to(device, non_blocking=True)
                    cache[blocks, :, offsets] = kv.transpose(0, 1)
                continue
            assert prefix["tokens"] == tokens, (name, prefix["tokens"], tokens)
            for layer, cache in self._caches.items():
                if layer in prefix["kv"]:
                    first = prefix["kv_first"][layer]
                    positions, blocks, offsets = self._slots(
                        layer, block_ids, max(start, first), tokens, device)
                    kv = prefix["kv"][layer].to(device, non_blocking=True)
                    kv = kv[:, positions - first].transpose(0, 1)
                    cache[blocks, :, offsets] = kv.to(cache.dtype)
                elif layer in prefix["ssm"]:
                    # One state block per request (no prefix caching for gated-delta targets).
                    # vLLM keeps the last kernel - 1 conv inputs as (dim, state) or (state, dim)
                    # and the recurrent state as [v_heads, v_dim, k_dim].
                    conv, ssm = cache
                    block = block_ids[self._group[layer]][0]
                    state = prefix["conv"][layer].to(device)  # HF: [conv_dim, kernel]
                    if conv.shape[1] == state.shape[0]:
                        conv[block] = state[:, state.shape[1] - conv.shape[2]:].to(conv.dtype)
                    else:
                        state = state[:, state.shape[1] - conv.shape[1]:].transpose(0, 1)
                        conv[block] = state.to(conv.dtype)
                    ssm[block] = prefix["ssm"][layer].to(device).transpose(-1, -2).to(ssm.dtype)
                else:
                    raise KeyError(f"{name}: no K/V or state for layer {layer}")

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        # Parity: a request's own cache after its prefill, in the HybridWriter dict layout.
        for name, tokens, block_ids in self._get_connector_metadata().captures:
            out = {"tokens": tokens, "kv": {}, "kv_first": {}, "conv": {}, "ssm": {}}
            for layer, cache in self._caches.items():
                if torch.is_tensor(cache):
                    positions, blocks, offsets = self._slots(
                        layer, block_ids, 0, tokens, cache.device)
                    out["kv"][layer] = cache[blocks, :, offsets].transpose(0, 1).clone()
                    out["kv_first"][layer] = int(positions[0])
                    continue
                conv, ssm = cache
                block = block_ids[self._group[layer]][0]
                state = conv[block]  # HF orientation: [conv_dim, kernel - 1]
                state = state if state.shape[0] > state.shape[1] else state.transpose(0, 1)
                out["conv"][layer] = state.clone()
                out["ssm"][layer] = ssm[block].transpose(-1, -2).clone()
            STORE[name] = out


try:  # vLLM's hybrid allocator (separate block tables per attention type)
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

    class HybridKVConnector(TranslatedKVConnector, SupportsHMA):
        """``TranslatedKVConnector`` with vLLM's hybrid KV cache manager left on (Gemma 3 / 4,
        Qwen 3.5 targets)."""

        def request_finished_all_groups(self, request, block_ids):
            return False, None
except ImportError:  # older vLLM
    pass


# ---- Cross-model connector ---------------------------------------------------------------------
# Cross-model KV reuse in vLLM: a request for target B reuses a document any source model read.
#
# Prototype (CPU-tested with a fake vLLM; not yet run on a GPU). ``REGISTRY`` holds documents by text key (``document_key``:
# tokenizers differ, text does not): per source an encoder memory, per target the header +
# document ids a request starts with and the ``STORE`` name of a translated prefix. Like
# ``vllm_connector.STORE`` it is module level, since vLLM loads this module by name.
#
# ``CrossModelKVConnector`` matches each new request's prompt against the registry on the
# scheduler side: translated KV for this target -> its tokens, injected by the parent
# ``TranslatedKVConnector``; only an encoder memory -> start the translator decode on a background
# thread and return ``None`` (vLLM asks again next step, holding no blocks for the request);
# nothing -> 0 tokens (normal prefill). Put a persistent tier (LMCache, vLLM's
# ``OffloadingConnector``) in front of it with ``FirstHitMultiConnector`` (``engine_overrides``).
# Everything runs in vLLM's process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``).

# Document tokens (after the header) hashed into the prefix index; registered prefixes shorter
# than header + ANCHOR are indexed at their own length.
ANCHOR = 32
# Seconds the scheduler sleeps each time it asks for a prefix still being decoded.
WAIT_SECONDS = 0.005


def document_key(text: str) -> str:
    """Model-independent document identity: sha256 of the NFC-normalised, stripped text
    (``prompts.chat_row`` strips)."""

    text = unicodedata.normalize("NFC", str(text).strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TargetPrefix:
    ids: tuple[int, ...]  # header + document ids in the target's token space
    header_tokens: int


@dataclass
class DocumentEntry:
    key: str
    # source model -> encoder memory (the first added is used)
    memories: dict[str, Any] = field(default_factory=dict)
    prefixes: dict[str, TargetPrefix] = field(default_factory=dict)  # target -> prefix ids
    translated: dict[str, str] = field(default_factory=dict)  # target -> STORE name


@dataclass(frozen=True)
class Match:
    key: str
    tokens: int  # prompt tokens the registry can supply (header + document, or a prefix of it)
    full: bool  # the whole registered prefix (False: a partial-document match)
    state: str  # "translated", "memory" or "none" (ids known, nothing to inject)


# translate(entry, source, memory, target) -> prefix for STORE: [layers, 2, tokens, kv_heads,
# head_dim] or a HybridWriter dict, holding len(entry.prefixes[target].ids) tokens. Runs on a
# worker thread in vLLM's process.
Translate = Callable[[DocumentEntry, str, Any, str], Any]


class CrossModelRegistry:
    """Documents by text key; per target, an anchor-hash index from prompt prefixes to
    documents."""

    def __init__(self, translate: Translate | None = None, *, workers: int = 1):
        self.translate = translate
        self.documents: dict[str, DocumentEntry] = {}
        # target -> (anchor length, hash) -> keys
        self._index: dict[str, dict[tuple[int, int], list[str]]] = {}
        self._lock = threading.RLock()
        self._jobs: dict[tuple[str, str], Future] = {}
        self.errors: dict[tuple[str, str], BaseException] = {}
        self._workers = int(workers)
        self._executor: ThreadPoolExecutor | None = None

    def add_document(self, text: str | None = None, *, key: str | None = None) -> str:
        assert text is not None or key is not None, "add_document needs the text or its key"
        key = key if key is not None else document_key(text)
        with self._lock:
            self.documents.setdefault(key, DocumentEntry(key))
        return key

    def add_target_ids(self, key: str, target: str, prefix_ids, header_tokens: int) -> None:
        """The prompt prefix a request for ``target`` starts with: header ids + document ids."""

        ids = tuple(int(token) for token in prefix_ids)
        assert 0 <= int(header_tokens) < len(ids), (header_tokens, len(ids))
        with self._lock:
            entry = self.documents.setdefault(key, DocumentEntry(key))
            index = self._index.setdefault(target, {})
            old = entry.prefixes.get(target)
            if old is not None:
                anchor = self._anchor(old)
                bucket = index.get((anchor, hash(old.ids[:anchor])), [])
                if key in bucket:
                    bucket.remove(key)
            entry.prefixes[target] = TargetPrefix(ids, int(header_tokens))
            anchor = self._anchor(entry.prefixes[target])
            bucket = index.setdefault((anchor, hash(ids[:anchor])), [])
            if key not in bucket:
                bucket.append(key)

    def add_memory(self, key: str, source: str, memory: Any) -> None:
        with self._lock:
            self.documents.setdefault(key, DocumentEntry(key)).memories.setdefault(source, memory)

    def add_translated(
        self, key: str, target: str, prefix: Any, *, name: str | None = None) -> str:
        """Put an already-translated prefix into ``STORE``; returns its name."""

        name = name or f"xmodel:{key[:16]}:{target}"
        with self._lock:
            entry = self.documents.setdefault(key, DocumentEntry(key))
            if target in entry.prefixes:
                held = prefix_tokens(prefix)
                expected = len(entry.prefixes[target].ids)
                if held != expected:
                    raise ValueError(
                        f"{name}: prefix holds {held} tokens, the target ids {expected}")
            STORE[name] = prefix
            entry.translated[target] = name
            self.errors.pop((key, target), None)
        return name

    def evict(self, key: str, target: str) -> None:
        """Drop the resident translated prefix (the caller makes sure no scheduled request still
        loads it)."""

        with self._lock:
            name = self.documents[key].translated.pop(target, None)
            if name is not None:
                STORE.pop(name, None)

    @staticmethod
    def _anchor(prefix: TargetPrefix) -> int:
        return min(len(prefix.ids), prefix.header_tokens + ANCHOR)

    def match(
        self, target: str, prompt_ids, *, key: str | None = None, partial: bool = False,
    ) -> Match | None:
        """The registered prefix of ``target`` that ``prompt_ids`` starts with (the longest, when
        several do). ``partial``: also accept a prompt that shares only header + ``ANCHOR``+
        document tokens with a registered prefix; only sound when the translation of a document
        prefix is the prefix of the document's translation (causal translator, full-attention
        target)."""

        prompt = tuple(int(token) for token in prompt_ids)
        with self._lock:
            if key is not None:
                keys = [key]
            else:
                index = self._index.get(target, {})
                anchors = sorted({length for length, _ in index if length <= len(prompt)})
                keys = []
                for anchor in anchors:
                    keys += index.get((anchor, hash(prompt[:anchor])), [])
            best = None
            for candidate in keys:
                entry = self.documents.get(candidate)
                prefix = entry.prefixes.get(target) if entry is not None else None
                if prefix is None:
                    continue
                tokens, full = len(prefix.ids), True
                if prompt[:tokens] != prefix.ids:
                    if not partial:
                        continue
                    size, tokens, full = min(len(prompt), tokens), 0, False
                    while tokens < size and prompt[tokens] == prefix.ids[tokens]:
                        tokens += 1
                    if tokens < self._anchor(prefix):
                        continue
                if best is None or tokens > best.tokens:
                    if entry.translated.get(target) in STORE:
                        state = "translated"
                    else:
                        state = "memory" if entry.memories else "none"
                    best = Match(candidate, tokens, full, state)
            return best

    def ensure_translated(self, key: str, target: str) -> str | None:
        """The ``STORE`` name of ``key``'s translated prefix for ``target``, or None while a
        decode runs (started here when needed; single flight per document and target). A failed
        decode raises its error, now and on every later call, until ``add_translated``."""

        with self._lock:
            entry = self.documents[key]
            name = entry.translated.get(target)
            if name in STORE:
                return name
            job_key = (key, target)
            job = self._jobs.get(job_key)
            if job is None:
                if job_key in self.errors:
                    raise self.errors[job_key]
                if not entry.memories or self.translate is None:
                    raise LookupError(
                        f"{key[:16]}: no translated prefix for {target} and no memory or "
                        "translate function")
                source, memory = next(iter(entry.memories.items()))
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        self._workers, thread_name_prefix="xmodel-translate")

                def run():
                    self.add_translated(key, target, self.translate(entry, source, memory, target))

                self._jobs[job_key] = self._executor.submit(run)
                return None
            if not job.done():
                return None
            del self._jobs[job_key]
            error = job.exception()
            if error is not None:
                self.errors[job_key] = error
                raise error
            return entry.translated[target]

    def failed(self, key: str, target: str) -> bool:
        with self._lock:
            return (key, target) in self.errors

    def wait(self) -> None:
        """Block until every running decode finished (tests, shutdown)."""

        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.exception()


REGISTRY = CrossModelRegistry()


def add_document_for_targets(
    text: str, tokenizers: dict, *, registry: CrossModelRegistry | None = None,
    role: str = "system") -> str:
    """Register ``text`` and, for each ``{target: tokenizer}`` (Hugging Face tokenizers,
    ``models.load_tokenizer``; not ``LLM.get_tokenizer()``, which is vLLM's own for Mistral
    models), the header + document ids a request for that target starts with: ``prompts.render`` of the same chat row
    ``SpeculativeEncoder`` encodes (the question only shapes ``after_ids``). Returns the key."""

    from speculative_encoding import prompts

    registry = registry or REGISTRY
    key = registry.add_document(text)
    row = prompts.chat_row("xmodel", key[:16], text, question="", role=role)
    for target, tokenizer in tokenizers.items():
        rendered = prompts.render(tokenizer, row)
        ids = list(rendered.header_ids) + list(rendered.doc_ids)
        registry.add_target_ids(key, target, ids, len(rendered.header_ids))
    return key


class CrossModelKVConnector(TranslatedKVConnector):
    """``TranslatedKVConnector`` that finds the prefix itself (``REGISTRY``) instead of reading
    ``"inject"`` from the request; an explicit ``kv_transfer_params={"inject": ...}`` still takes
    the parent's path.

    ``kv_connector_extra_config``: ``target`` (catalog key; default: the served model name),
    ``partial`` (accept document-prefix matches; full-attention targets only)."""

    allow_partial_matches = True

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        extra = dict(getattr(transfer, "kv_connector_extra_config", None) or {})
        self.target = extra.get("target") or vllm_config.model_config.model
        self.partial = bool(extra.get("partial", False)) and self.allow_partial_matches
        self.registry = REGISTRY
        self._matched: dict[str, tuple[str, int]] = {}  # request id -> (STORE name, tokens)
        # Requests waiting for a background decode: their match, so the scheduler's repeated
        # asks do not re-hash a long prompt each step.
        self._waiting: dict[str, Match] = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if "inject" in params:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        self._matched.pop(request.request_id, None)
        prompt = request.prompt_token_ids or []
        match = self._waiting.pop(request.request_id, None) or self.registry.match(
            self.target, prompt, key=params.get("xdoc"), partial=self.partial)
        if match is None or match.state == "none":
            return 0, False
        tokens = match.tokens
        if tokens >= len(prompt):
            # vLLM computes at least the last prompt token itself (its logits). A full-attention
            # prefix can be loaded short of its end; a hybrid one (sliding-window tail, gated-delta
            # state) only whole.
            if not self.allow_partial_matches:
                return 0, False
            tokens = len(prompt) - 1
        if tokens <= num_computed_tokens:  # vLLM's own prefix cache (same cache_salt) holds it
            return 0, False
        try:
            name = self.registry.ensure_translated(match.key, self.target)
        except Exception:  # the decode failed (kept in registry.errors): prefill natively
            return 0, False
        if name is None:
            # Decoding: vLLM asks again next step. The scheduler loop is tight Python; yield
            # the GIL so the decode thread can run (without this it starved: hours at 40k tokens).
            self._waiting[request.request_id] = match
            time.sleep(WAIT_SECONDS)
            return None, False
        self._matched[request.request_id] = (name, tokens)
        return tokens - num_computed_tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        matched = self._matched.pop(request.request_id, None)
        if matched is None or num_external_tokens <= 0:
            super().update_state_after_alloc(request, blocks, num_external_tokens)
            return
        name, tokens = matched
        # The parent's pending load: build_connector_meta turns it into a load of [start, tokens).
        start = tokens - num_external_tokens
        self._pending[request.request_id] = {"inject": name, "tokens": tokens, "start": start}


try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

    class HybridCrossModelKVConnector(CrossModelKVConnector, SupportsHMA):
        """Sliding-window / gated-delta targets with vLLM's hybrid KV cache manager on.
        Full-prefix matches only: a recurrent state exists for the whole document, not for a
        prefix of it."""

        allow_partial_matches = False

        def request_finished_all_groups(self, request, block_ids):
            return False, None
except ImportError:  # older vLLM
    pass


try:
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

    class FirstHitMultiConnector(MultiConnector):
        """vLLM's ``MultiConnector`` (save to every child, load from the first with a hit) that
        stops asking at the first child reporting a hit or ``None``. Stock ``MultiConnector``
        queries every child each time, so a ``CrossModelKVConnector`` behind LMCache would start
        a decode for a prefix LMCache already holds, and its ``None`` would hide LMCache's hit."""

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            for index, child in enumerate(self._connectors):
                tokens, load_async = child.get_num_new_matched_tokens(
                    request, num_computed_tokens)
                if tokens is None:
                    return None, False
                if tokens > 0:
                    self._requests_to_connector[request.request_id] = index
                    return tokens, load_async
            return 0, False
except ImportError:  # vLLM without MultiConnector
    pass


def engine_overrides(
    target: str, *, hybrid: bool, tier: str | None = None, partial: bool = False,
    tier_config: dict | None = None) -> dict:
    """``vllm.LLM`` keyword overrides (``vllm_engine.llm_kwargs(..., **overrides)``) serving
    ``target`` with cross-model reuse: ``kv_transfer_config`` and, when needed, the hybrid-manager
    switch.

    ``tier``: None (registry only), ``"lmcache"`` (LMCache's in-process connector; full-attention
    targets only: it keeps one KV cache group and is not ``SupportsHMA``) or ``"offload"`` (vLLM's
    CPU ``OffloadingConnector``, which works with the hybrid allocator). The tier comes first, so
    its hits win and this connector only translates what the tier does not hold."""

    from vllm.config import KVTransferConfig

    own = {
        "kv_connector": "HybridCrossModelKVConnector" if hybrid else "CrossModelKVConnector",
        "kv_connector_module_path": __name__,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"target": target, "partial": bool(partial) and not hybrid},
    }
    if tier is None:
        return {"kv_transfer_config": KVTransferConfig(**own)}
    overrides = {}
    if tier == "lmcache":
        if hybrid:
            raise ValueError("LMCache keeps one KV cache group: use tier='offload' for hybrid")
        child = {
            "kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
            "kv_connector_extra_config": dict(tier_config or {})}
        # vLLM's factory special-cases the name "MultiConnector" when deciding HMA support; this
        # subclass reports SupportsHMA, so a non-HMA child needs the hybrid manager off.
        overrides["disable_hybrid_kv_cache_manager"] = True
    elif tier == "offload":
        child = {
            "kv_connector": "OffloadingConnector", "kv_role": "kv_both",
            "kv_connector_extra_config": dict(tier_config or {"cpu_bytes_to_use": 64 << 30})}
    else:
        raise ValueError(f"tier={tier!r}")
    overrides["kv_transfer_config"] = KVTransferConfig(
        kv_connector="FirstHitMultiConnector", kv_connector_module_path=__name__,
        kv_role="kv_both", kv_connector_extra_config={"connectors": [child, own]})
    return overrides

