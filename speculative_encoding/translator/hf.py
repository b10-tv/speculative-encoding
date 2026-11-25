"""Generic Hugging Face decoder/cache boundary for the unified translator."""

from __future__ import annotations

import contextlib
import contextvars
import copy
from dataclasses import dataclass, field

import torch

from speculative_encoding.translator.model import SourceGeometry, TargetGeometry
from speculative_encoding import settings

_CACHE_ONLY = "unified_translator_cache_only"
_REGISTERED = False
_CAPTURE: contextvars.ContextVar["_Capture | None"] = contextvars.ContextVar(
    "unified_hf_capture", default=None)
_FF_NAMES = ("mlp", "feed_forward", "block_sparse_moe", "moe")


class SelfCheckError(RuntimeError):
    """The model cannot be reproduced through its own Hugging Face cache API."""


@dataclass
class _Capture:
    tokens: int
    calls: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict)


def _cache_only_attention(module, query, key, value, attention_mask, **kwargs):
    capture = _CAPTURE.get()
    if capture is not None and not getattr(module, "is_kv_shared_layer", False):
        index = getattr(module, "layer_idx", None)
        if index is None:
            raise SelfCheckError(f"{type(module).__name__} has no layer_idx")
        capture.calls[int(index)] = (key[..., -capture.tokens:, :],
            value[..., -capture.tokens:, :])
    shape = (query.shape[0], query.shape[2], query.shape[1], value.shape[-1])
    return query.new_zeros(*shape), None


def _base_attention(name, module):
    import sys

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    default = getattr(sys.modules[type(module).__module__],
        "eager_attention_forward", None)
    return ALL_ATTENTION_FUNCTIONS.get_interface(name, default)


def _is_kv(layer):
    from transformers.cache_utils import CacheLayerMixin

    return isinstance(layer, CacheLayerMixin)


def _is_state(layer):
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    return isinstance(layer, LinearAttentionCacheLayerMixin)


def _state_tensors(layer, name):
    """Initialized recurrent/conv state tensors of one cache layer.

    transformers <=5.9 stores one tensor per layer; newer releases store a dict
    keyed by state index together with a matching ``is_*_initialized`` dict.
    """

    value = getattr(layer, name, None)
    initialized = getattr(layer, f"is_{name}_initialized", None)
    if isinstance(value, dict):
        flags = initialized if isinstance(initialized, dict) else {}
        return {index: tensor for index, tensor in value.items()
            if torch.is_tensor(tensor) and flags.get(index, True)}
    if torch.is_tensor(value) and (initialized is None or bool(initialized)):
        return {0: value}
    return {}


def _functional_cache(cache):
    """Make recurrent-state updates copy-on-write, so a copied cache never
    writes into the states it shares with the original."""

    for layer in cache.layers:
        if not _is_state(layer) or getattr(layer, "_unified_functional", False):
            continue
        for method_name, state_name, initialized_name in (
            ("update_conv_state", "conv_states", "is_conv_states_initialized"),
            ("update_recurrent_state", "recurrent_states",
                "is_recurrent_states_initialized"),
        ):
            original = getattr(layer, method_name)

            def update(value, *args, _original=original, _layer=layer,
                    _state=state_name, _initialized=initialized_name,
                    **kwargs):
                states = getattr(_layer, _state)
                initialized = getattr(_layer, _initialized, False)
                if isinstance(states, dict):
                    index = kwargs.get("state_idx", args[0] if args else 0)
                    if (isinstance(initialized, dict) and initialized.get(index)
                            and torch.is_tensor(states.get(index))):
                        states = dict(states)
                        states[index] = states[index].clone()
                        setattr(_layer, _state, states)
                elif initialized:
                    setattr(_layer, _state, states.clone())
                return _original(value, *args, **kwargs)

            setattr(layer, method_name, update)
        layer._unified_functional = True
    return cache


def copy_cache(cache):
    def clone(item):
        return item.clone() if isinstance(item, torch.Tensor) else item

    result = copy.copy(cache)
    result.layers = []
    for layer in cache.layers:
        copied = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                setattr(copied, name, value.clone())
            elif isinstance(value, dict):
                setattr(copied, name,
                    {key: clone(item) for key, item in value.items()})
            elif isinstance(value, list):
                setattr(copied, name, [clone(item) for item in value])
        for name in ("_unified_functional", "update_conv_state",
                "update_recurrent_state"):
            copied.__dict__.pop(name, None)
        result.layers.append(copied)
    return _functional_cache(result)


class ModelInterface:
    """Use one model's own decoder loop, cache classes and attention
    functions."""

    def __init__(self, model: torch.nn.Module, name: str | None = None, *,
            skip_feedforward: bool = True, check: bool = True,
            bf16_logit_tolerance: float = 5e-2):
        self.model = model
        self.bf16_logit_tolerance = float(bf16_logit_tolerance)
        self.name = (name or getattr(model.config, "_name_or_path", None)
            or type(model).__name__)
        decoder = (model.get_decoder() if hasattr(model, "get_decoder")
            else getattr(model, "model", model))
        if not hasattr(decoder, "layers"):
            decoder = getattr(decoder, "language_model", decoder)
        if not hasattr(decoder, "layers"):
            raise SelfCheckError(f"{self.name}: cannot find decoder layers")
        self.decoder, self.layers = decoder, decoder.layers
        text_config = model.config
        if hasattr(model.config, "get_text_config"):
            text_config = model.config.get_text_config(decoder=True)
        self.config = getattr(decoder, "config", text_config)
        types = getattr(self.config, "layer_types", None)
        if types is None or len(types) != len(self.layers):
            window = (getattr(self.config, "sliding_window", None)
                or getattr(self.config, "attention_chunk_size", None))
            kind = "sliding_attention" if window is not None else "full_attention"
            types = [kind] * len(self.layers)
        self.layer_types = tuple(types)
        self.skip_feedforward = skip_feedforward
        self._probe_cache = self.native_prefix(self.probe_ids(4))
        probe_layers = self._probe_cache.layers
        self.cache_layers = len(probe_layers)
        self.kv_layers = tuple(
            index for index, layer in enumerate(self.layers)
            if index < self.cache_layers and _is_kv(probe_layers[index])
            and (getattr(layer, "block_type", None) == "attention"
                or hasattr(getattr(layer, "self_attn", None), "layer_idx"))
            and not getattr(getattr(layer, "self_attn", None),
                "is_kv_shared_layer", False)
            and getattr(probe_layers[index], "is_initialized", True))
        if not self.kv_layers:
            self.kv_layers = tuple(index
                for index, layer in enumerate(probe_layers)
                if _is_kv(layer) and getattr(layer, "is_initialized", False))
        self.all_full_attention = (
            len(self.layers) == self.cache_layers
            and all(kind == "full_attention" for kind in self.layer_types)
            and all(_is_kv(layer) for layer in probe_layers))
        # Set by ``self_check``: whether HF continues a written cache over many
        # tokens.
        self.continues_cache = True
        if check:
            self.self_check()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype

    @property
    def width(self):
        return int(self.config.hidden_size)

    def source_geometry(self):
        return SourceGeometry(len(self.layers) + 1, self.width)

    def target_geometry(self):
        return TargetGeometry(len(self.layers), self.width, self.layer_types)

    def probe_ids(self, count):
        vocab = int(self.model.get_input_embeddings().num_embeddings)
        generator = torch.Generator(device=self.device).manual_seed(1000 + count)
        return torch.randint(vocab // 4, max(vocab // 4 + 1, vocab // 2),
            (count,), generator=generator, device=self.device)

    def embed(self, ids):
        return self.model.get_input_embeddings()(ids)

    def logits(self, hidden):
        logits = self.model.get_output_embeddings()(hidden)
        scale = getattr(self.config, "logits_scaling", None)
        if scale:
            logits = logits / scale
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def new_cache(self):
        from transformers import DynamicCache

        return _functional_cache(DynamicCache(config=self.config))

    def _read_hidden(self, ids, inputs_only):
        inputs, outputs = [], []

        def before(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            inputs.append(hidden.detach())

        def after(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            outputs.append(hidden.detach())

        hooked = self.layers if inputs_only else self.layers[:1]
        hooks = [layer.register_forward_pre_hook(before, with_kwargs=True)
            for layer in hooked]
        if not inputs_only:
            hooks += [layer.register_forward_hook(after) for layer in self.layers]
        try:
            self.decoder(input_ids=ids.reshape(1, -1).to(self.device),
                use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        missed = SelfCheckError(
            f"{self.name}: hidden-state hooks did not capture every layer")
        if inputs_only:
            if len(inputs) != len(self.layers):
                raise missed
            return torch.stack([state[0] for state in inputs], dim=1)
        if len(inputs) != 1 or len(outputs) != len(self.layers):
            raise missed
        # Source streams are the raw embedding output followed by the layer
        # outputs; target layer inputs come from ``layer_inputs``.
        embedding = self.embed(ids).reshape(1, -1, self.width)[0].detach()
        return torch.stack([embedding, *(value[0] for value in outputs)], dim=1)

    @torch.no_grad()
    def read_hidden_states(self, ids):
        """Source streams ``[tokens, layers + 1, width]``: the layer-0 input,
        then every layer output."""

        return self._read_hidden(ids, False)

    @torch.no_grad()
    def layer_inputs(self, ids):
        return self._read_hidden(ids, True)

    @torch.no_grad()
    def native_prefix(self, ids, positions=None):
        cache = self.new_cache()
        if ids.numel():
            kwargs = {}
            if positions is not None:
                kwargs = {"position_ids": positions.reshape(1, -1)}
            self.decoder(input_ids=ids.reshape(1, -1).to(self.device),
                past_key_values=cache, use_cache=True, **kwargs)
        return cache

    @contextlib.contextmanager
    def _substitute(self, hidden, capture):
        """Run the decoder as a cache writer: every layer input is replaced by
        ``hidden``, attention only stores K/V (it returns zeros) and the
        feed-forward blocks are skipped."""

        global _REGISTERED
        if not _REGISTERED:
            from transformers import AttentionInterface
            from transformers.masking_utils import (
                ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface)

            AttentionInterface.register(_CACHE_ONLY, _cache_only_attention)
            AttentionMaskInterface.register(_CACHE_ONLY,
                ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
            _REGISTERED = True
        hooks, skipped = [], []
        for index, layer in enumerate(self.layers):
            def replace(module, args, kwargs, _index=index):
                replacement = hidden[:, _index].unsqueeze(0).to(self.dtype)
                if args:
                    return (replacement, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": replacement}

            hooks.append(layer.register_forward_pre_hook(replace,
                with_kwargs=True))
        for layer in self.layers:
            if getattr(layer, "block_type", None) in ("mlp", "moe"):
                original = layer.forward
                layer.forward = lambda hidden_states, *a, **k: hidden_states
                skipped.append((layer, original))
        if self.skip_feedforward:
            for layer in self.layers:
                for name in _FF_NAMES:
                    module = getattr(layer, name, None)
                    if isinstance(module, torch.nn.Module):
                        original = module.forward
                        module.forward = (lambda hidden_states, *a, **k:
                            torch.zeros_like(hidden_states))
                        skipped.append((module, original))
        previous = getattr(self.config, "_attn_implementation_internal", None)
        self.config._attn_implementation_internal = _CACHE_ONLY
        token = _CAPTURE.set(capture)
        try:
            yield
        finally:
            _CAPTURE.reset(token)
            self.config._attn_implementation_internal = previous
            for module, original in reversed(skipped):
                module.forward = original
            for hook in hooks:
                hook.remove()

    def write_prefix(self, ids, hidden_doc, header_cache, positions, *,
            capture=False):
        """Extend ``header_cache`` with a document whose layer inputs are
        ``hidden_doc``.

        ``ids`` are the document's real target token ids. Every layer input is
        replaced, but some models also derive token-dependent per-layer inputs
        from the ids (Gemma 4's per-layer embeddings), so the ids must be real.
        """

        if (hidden_doc.ndim != 3 or hidden_doc.shape[1] != len(self.layers)
                or hidden_doc.shape[2] != self.width):
            raise ValueError(f"{self.name}: hidden_doc must be "
                f"[tokens, {len(self.layers)}, {self.width}]")
        ids = ids.reshape(-1).to(self.device)
        if ids.numel() != hidden_doc.shape[0]:
            raise ValueError(f"{self.name}: {ids.numel()} document ids for "
                f"{hidden_doc.shape[0]} predicted positions")
        cache = copy_cache(header_cache)
        record = _Capture(int(hidden_doc.shape[0])) if capture else None
        with self._substitute(hidden_doc, record):
            self.decoder(input_ids=ids.view(1, -1),
                position_ids=positions.reshape(1, -1).to(self.device),
                past_key_values=cache, use_cache=True)
        if record is None:
            return cache, None
        if sorted(record.calls) != sorted(self.kv_layers):
            raise SelfCheckError(f"{self.name}: captured K/V layers "
                f"{sorted(record.calls)} != expected {sorted(self.kv_layers)}")
        return cache, record.calls

    def forward(self, ids, positions, cache, attention_mask=None):
        output = self.decoder(input_ids=ids.reshape(1, -1),
            position_ids=positions.reshape(1, -1), past_key_values=cache,
            attention_mask=attention_mask, use_cache=True)
        return output.last_hidden_state[0]

    @torch.no_grad()
    def hidden(self, ids):
        """Final hidden states of one native forward over ``ids`` (no cache)."""

        output = self.decoder(input_ids=ids.reshape(1, -1).to(self.device),
            use_cache=False)
        return output.last_hidden_state[0]

    def spliced_hidden(self, ids, hidden_doc, start, *, use_cache=False):
        """One native forward over ``ids`` whose layer inputs at the document
        positions are ``hidden_doc``.

        Used for models whose HF code cannot continue a cache over several
        tokens. Returns ``(hidden, cache)``.
        """

        end = start + int(hidden_doc.shape[0])
        hooks = []
        for index, layer in enumerate(self.layers):
            def replace(module, args, kwargs, _index=index):
                current = args[0] if args else kwargs["hidden_states"]
                document = hidden_doc[:, _index].unsqueeze(0).to(current.dtype)
                spliced = torch.cat((current[:, :start], document,
                    current[:, end:]), dim=1)
                if args:
                    return (spliced, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": spliced}

            hooks.append(layer.register_forward_pre_hook(replace,
                with_kwargs=True))
        cache = self.new_cache() if use_cache else None
        try:
            output = self.decoder(input_ids=ids.reshape(1, -1).to(self.device),
                past_key_values=cache, use_cache=use_cache)
        finally:
            for hook in hooks:
                hook.remove()
        return output.last_hidden_state[0], cache

    @torch.no_grad()
    def generate(self, ids, hidden_doc=None, start=0, *, max_new_tokens,
            eos_token_ids=()):
        """Greedy decoding of ``ids`` (header + optional translated document +
        prompt), batch 1."""

        ids = ids.reshape(-1).to(self.device)
        if hidden_doc is None:
            return self.greedy(self.new_cache(), ids, 0,
                max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids)
        if self.continues_cache:
            end = start + int(hidden_doc.shape[0])
            cache, _ = self.write_prefix(ids[start:end], hidden_doc,
                self.native_prefix(ids[:start]),
                torch.arange(start, end, device=self.device))
            return self.greedy(cache, ids[end:], end,
                max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids)
        hidden, cache = self.spliced_hidden(ids, hidden_doc, start,
            use_cache=True)
        return self._decode(cache, hidden[-1], int(ids.numel()),
            max_new_tokens, eos_token_ids)

    @torch.no_grad()
    def greedy(self, cache, ids, position, *, max_new_tokens, eos_token_ids=()):
        """Greedy decoding after ``cache`` (consumed): read ``ids`` from
        ``position`` on, then generate."""

        ids = ids.reshape(-1).to(self.device)
        if not ids.numel():
            raise ValueError(f"{self.name}: greedy decoding needs at least one "
                "prompt token after the cache")
        positions = torch.arange(position, position + ids.numel(),
            device=self.device)
        hidden = self.forward(ids, positions, cache)[-1]
        return self._decode(cache, hidden, position + int(ids.numel()),
            max_new_tokens, eos_token_ids)

    def _decode(self, cache, hidden, position, max_new_tokens, eos_token_ids):
        stop, tokens = {int(token) for token in eos_token_ids}, []
        while len(tokens) < max_new_tokens:
            token = int(self.logits(hidden).argmax(-1))
            tokens.append(token)
            if token in stop or len(tokens) >= max_new_tokens:
                break
            hidden = self.forward(torch.tensor([token], device=self.device),
                torch.tensor([position], device=self.device), cache)[-1]
            position += 1
        return tokens

    def is_sliding_kv(self, index):
        """True when layer ``index`` keeps a sliding-window K/V cache."""

        layers = self._probe_cache.layers
        layer = layers[index] if index < len(layers) else None
        return bool(int(getattr(layer, "sliding_window", 0) or 0))

    @torch.no_grad()
    def self_check(self, tokens=4, header=2):
        """Compare the written cache, logits and continuation with the model's
        own forward.

        Errors are relative (norm of the difference over the norm of the native
        value) because bf16 recomputation in differently shaped chunks is not
        bit-identical. The measured errors are kept in ``check_report``.
        """

        wide = self.dtype in (torch.float32, torch.float64)
        # bf16: UT_BF16_CACHE_TOLERANCE overrides 2e-2 (Gemma 4 on B200
        # measured 2.01e-2 on the 4-token probe; 2e-2 passes on H200).
        cache_tolerance, logit_tolerance = 1e-4, 1e-4
        if not wide:
            cache_tolerance = float(settings.current().cache_tolerance)
            logit_tolerance = self.bf16_logit_tolerance
        report = {"dtype": str(self.dtype).replace("torch.", ""), "cache": 0.0}

        def relative(actual, expected):
            expected = expected.detach().float()
            difference = (actual.detach().float() - expected).norm()
            return float(difference / expected.norm().clamp_min(1e-6))

        ids = self.probe_ids(tokens + header)
        native = self.decoder(input_ids=ids.reshape(1, -1), use_cache=True)
        hidden = self.layer_inputs(ids)[header:]
        written, captured = self.write_prefix(ids[header:], hidden,
            self.native_prefix(ids[:header]),
            torch.arange(header, header + tokens, device=self.device),
            capture=True)
        layers = zip(native.past_key_values.layers, written.layers, strict=True)
        for index, (want, got) in enumerate(layers):
            pairs = []
            for name in ("keys", "values"):
                expected = getattr(want, name, None)
                actual = getattr(got, name, None)
                if torch.is_tensor(expected) and torch.is_tensor(actual):
                    pairs.append((name, expected, actual))
            for name in ("conv_states", "recurrent_states"):
                expected_states = _state_tensors(want, name)
                actual_states = _state_tensors(got, name)
                for state in sorted(set(expected_states) & set(actual_states)):
                    pairs.append((name, expected_states[state],
                        actual_states[state]))
            for name, expected, actual in pairs:
                if actual.shape != expected.shape:
                    raise SelfCheckError(f"{self.name}: cache layer {index} "
                        f"{name} has shape {tuple(actual.shape)}, expected "
                        f"{tuple(expected.shape)}")
                error = relative(actual, expected)
                report["cache"] = max(report["cache"], error)
                layer = self.layers[index]
                layer_kind = getattr(layer, "block_type",
                    getattr(layer, "layer_type", None))
                tolerance = cache_tolerance
                if (name in ("conv_states", "recurrent_states")
                        and layer_kind == "mamba"):
                    tolerance = 5e-2
                if error > tolerance:
                    raise SelfCheckError(f"{self.name}: cache layer {index} "
                        f"{name} differs (relative error {error:.2e})")
        if sorted(captured or {}) != sorted(self.kv_layers):
            raise SelfCheckError(f"{self.name}: captured K/V count is wrong")
        native_logits = self.model(input_ids=ids.reshape(1, -1),
            use_cache=False).logits[0]
        hidden_logits = self.logits(self.decoder(input_ids=ids.reshape(1, -1),
            use_cache=False).last_hidden_state[0])
        report["logits"] = relative(hidden_logits, native_logits)
        if report["logits"] > logit_tolerance:
            raise SelfCheckError(f"{self.name}: output projection does not "
                f"match native logits (relative error {report['logits']:.2e})")

        # Multi-token continuation after a written cache, else the spliced
        # path.
        continuation = self.probe_ids(tokens + header + 3)[-3:]
        full = torch.cat((ids, continuation))
        expected = self.model(input_ids=full.reshape(1, -1),
            use_cache=False).logits[0, ids.numel():]
        try:
            positions = torch.arange(ids.numel(), full.numel(),
                device=self.device)
            got = self.logits(self.forward(continuation, positions,
                copy_cache(written)))
            report["continuation"] = relative(got, expected)
            self.continues_cache = report["continuation"] <= logit_tolerance
        except Exception:  # noqa: BLE001 - such models use the spliced path
            report["continuation"] = None
            self.continues_cache = False
        spliced = self.spliced_hidden(full, hidden, header)[0][ids.numel():]
        report["spliced"] = relative(self.logits(spliced), expected)
        if report["spliced"] > logit_tolerance:
            raise SelfCheckError(f"{self.name}: spliced continuation differs "
                f"(relative error {report['spliced']:.2e})")
        report["continues_cache"] = self.continues_cache
        self.check_report = report
        return report


__all__ = ["ModelInterface", "SelfCheckError", "copy_cache"]
