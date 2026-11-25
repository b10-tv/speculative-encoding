"""Answer from a served prefix with Hugging Face alone (no vLLM), and the sampling options
both engines share."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import torch
from speculative_encoding.encode import cache_from_prefix, prefix_tokens
from speculative_encoding.prompts import question_dict
from speculative_encoding.translator.hf import ModelInterface, _is_kv, _is_state, copy_cache


# ---- Sampling ----------------------------------------------------------------------------------
# Sampling options shared by the Hugging Face and vLLM engines (free-form generation).
#
# ``None`` (the default everywhere) is the paper's request: greedy, the engine's EOS stop ids, one
# answer. A ``Sampling`` (or a dict of its fields) changes only what it sets:
#
# * ``temperature`` (0 = greedy argmax), ``top_p`` (1 = off), ``top_k`` (0 or -1 = off), applied in
#   vLLM's order (temperature, then top-k, then top-p) before drawing from the softmax;
# * ``seed``: a per-request seed (vLLM's ``SamplingParams.seed``; HF: a seeded
#   ``torch.Generator``). Same seed, model and device: same tokens. HF and vLLM draw differently,
#   so the same seed samples differently on each;
# * ``stop``: stop strings; generation ends once the decoded text contains one, which is cut from
#   ``text`` (vLLM's default ``include_stop_str_in_output=False``); ``ids`` keep the token that
#   completed it;
# * ``stop_token_ids``: extra stop ids, added to the engine's EOS ids (never replacing them);
# * ``n``: answers per question; the record's ``text`` / ``ids`` / ``finish_reason`` are the first
#   one's and ``samples`` holds all ``n``.



@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    n: int = 1

    def __post_init__(self):
        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        object.__setattr__(self, "stop", tuple(stop))
        stop_ids = tuple(int(token) for token in self.stop_token_ids)
        object.__setattr__(self, "stop_token_ids", stop_ids)
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.n < 1:
            raise ValueError(f"invalid sampling options: {self}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    def to_dict(self) -> dict:
        stop_ids = list(self.stop_token_ids)
        return {**asdict(self), "stop": list(self.stop), "stop_token_ids": stop_ids}

    def vllm_kwargs(self) -> dict:
        """``SamplingParams`` keyword arguments (besides ``max_tokens``, ``stop_token_ids`` and
        ``extra_args``)."""

        kwargs = {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k) if self.top_k > 0 else -1,
            "n": int(self.n),
        }
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if self.stop:
            kwargs["stop"] = list(self.stop)
        return kwargs


def as_sampling(value) -> Sampling | None:
    if value is None or isinstance(value, Sampling):
        return value
    if isinstance(value, dict):
        return Sampling(**value)
    raise TypeError(f"sampling must be None, a Sampling or a dict, not {type(value).__name__}")


def pick(logits, sampling: Sampling | None, generator=None) -> int:
    """The next token from one step's logits ``[vocab]``. Greedy: ``argmax`` on the logits as
    given, exactly as ``ModelInterface._decode``."""

    if sampling is None or sampling.greedy:
        return int(logits.argmax(-1))
    scores = logits.float() / float(sampling.temperature)
    if 0 < sampling.top_k < scores.numel():
        threshold = scores.topk(int(sampling.top_k)).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if sampling.top_p < 1:
        ordered, order = scores.sort(descending=True)
        probs = ordered.softmax(-1)
        # Keep the smallest head whose mass reaches top_p (at least one token).
        drop = probs.cumsum(-1) - probs > sampling.top_p
        scores = scores.scatter(-1, order, ordered.masked_fill(drop, float("-inf")))
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


def truncate_at_stop(text: str, stops) -> tuple[str, bool]:
    """``text`` cut before the first stop string it contains, and whether one was found."""

    found = [index for index in (text.find(stop) for stop in stops if stop) if index >= 0]
    return (text[:min(found)], True) if found else (text, False)


# ---- Engine ------------------------------------------------------------------------------------
# Answer questions from a served prefix with Hugging Face alone (no vLLM): laptops, Colab, CPU.
#
# The same request ``vllm_engine`` sends to vLLM, run on the target's own HF decoder:
#
# * prompt = ``prefix_ids + question_ids`` (``after_ids``), the first ``len(prefix_ids)`` tokens'
#   K/V (and gated-delta states) taken from ``prefix`` instead of computed: a tensor
#   ``[layers, 2, tokens, kv_heads, head_dim]`` (every layer plain full attention;
#   ``translate.cache_from_prefix``) or a ``HybridWriter.finish`` dict (``tokens``, per layer
#   ``kv`` / ``kv_first``, ``conv`` / ``ssm``): translated, native
#   (``translate.prefix_from_cache``) or a compressed cartridge;
# * greedy, at most ``max_tokens`` new tokens (``prompts.GENERATION_TOKENS[metric]``), stopping at
#   the first token in ``stop_token_ids`` (default: the model's generation-config EOS ids plus the
#   tokenizer's ``eos_token_id``, which vLLM always stops on); the stop token is kept in ``ids``
#   and dropped from ``text`` (``decode(..., skip_special_tokens=True)``);
# * positions: the question sits at cache index ``len(prefix_ids)`` on, as vLLM numbers it.
#   SnapKV-compressed prefixes and compressor cartridges are already laid out for that numbering,
#   so they are served as ordinary prefixes; ``prefix_ids`` must have the prefix's length, its
#   values are never read (vLLM reads them only for prefix-cache hashing). No BOS is added:
#   ``prefix_ids`` already carry the chat template's;
# * ``metric == "nll"`` questions: ``reference_nll``, the mean NLL of
#   ``continuation_ids[len(after_ids):]`` given the prefix and everything before.
#
# Checked in fp32 against HF ``generate`` on the full prompt with tiny random Llama / Qwen3 /
# Gemma 3 / Qwen 3.5 models: identical token ids, first-token log-probs within 1e-6. Against vLLM,
# numerics differ (paged / flash attention kernels, bf16 accumulation order, fused RoPE and norms),
# so near-tie argmaxes, and hence long answers, can diverge. ``top_logprobs`` is the top-k of the
# raw log-softmax (vLLM's default ``logprobs_mode``). HF keeps ``window - 1`` sliding-window entries
# and vLLM ``window``; both attend the same keys. No prefix caching and no batching: each question
# runs batch 1 on its own copy of the prefix cache. Gemma 4's shared-K/V layers and multimodal
# targets are untested here.
#
# Free-form generation (the defaults above are unchanged): ``sampling=`` (``sampling.Sampling`` or a
# dict) on ``answer`` / ``answer_batch`` (or a question dict's ``"sampling"``); ``stream`` yields
# tokens and text as they are produced; ``continue_turn`` extends a cache with a turn and its
# answer, so a conversation continues without re-reading the prefix.

__all__ = [
    "as_interface", "stop_ids", "build_cache", "answer", "answer_batch", "reference_nll",
    "stream", "continue_turn", "Sampling", "as_sampling", "truncate_at_stop",
]


def as_interface(model) -> ModelInterface:
    """A ``ModelInterface`` (passed through) or an HF causal LM wrapped without the self check."""

    if isinstance(model, ModelInterface):
        return model
    model.eval()
    return ModelInterface(model, check=False)


def stop_ids(interface, tokenizer=None):
    """The model's generation-config EOS ids plus the tokenizer EOS vLLM stops on."""

    from speculative_encoding.models import generation_config

    ids = list(generation_config(interface.model, 1)["eos_token_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and int(eos) not in ids:
        ids.append(int(eos))
    return ids


@torch.no_grad()
def _kv_geometry(interface, index):
    """``(kv_heads, head_dim)`` of layer ``index``'s attention, or None when it has none."""

    attention = getattr(interface.layers[index], "self_attn", None)
    head_dim = getattr(attention, "head_dim", None)
    k_proj = getattr(attention, "k_proj", None)
    if head_dim is None or k_proj is None:
        return None
    return k_proj.out_features // int(head_dim), int(head_dim)


def _check_geometry(interface, index, kv_heads, head_dim):
    expected = _kv_geometry(interface, index)
    if expected is not None and (kv_heads, head_dim) != expected:
        raise ValueError(
            f"layer {index}: the prefix has {kv_heads} K/V heads of size {head_dim}, this model "
            f"{expected[0]} of size {expected[1]}: a cache made for another model")


def build_cache(interface, prefix, prefix_ids=None):
    """An HF cache of the target holding ``prefix``, and its length.

    ``prefix`` None: HF prefills ``prefix_ids`` natively (the native arm). Tensor:
    ``translate.cache_from_prefix``. Dict: each layer as vLLM's hybrid connector writes it (K/V of
    the last ``n`` positions; sliding-window layers count the whole prefix as seen; gated-delta
    conv and recurrent states, conv left-padded to the kernel when given as vLLM's ``kernel - 1``
    columns)."""

    device, dtype = interface.device, interface.dtype
    if prefix is None:
        ids = torch.as_tensor(list(prefix_ids or []), dtype=torch.long, device=device)
        return interface.native_prefix(ids), int(ids.numel())
    tokens = prefix_tokens(prefix)
    if prefix_ids is not None and len(prefix_ids) != tokens:
        raise ValueError(
            f"prefix holds {tokens} tokens but {len(prefix_ids)} prefix ids were "
            "given (vLLM asserts they match)")
    if not isinstance(prefix, dict):
        layers = len(interface.layers)
        if prefix.dim() != 5 or prefix.shape[0] != layers or prefix.shape[1] != 2:
            raise ValueError(f"prefix {tuple(prefix.shape)}: this model needs [{layers}, 2, "
                             "tokens, kv_heads, head_dim] (a cache made for another model?)")
        for index in range(layers):
            _check_geometry(interface, index, prefix.shape[3], prefix.shape[4])
        return cache_from_prefix(interface, prefix.to(device=device, dtype=dtype)), tokens
    cache = interface.new_cache()
    for index, layer in enumerate(cache.layers):
        block = interface.layers[index]
        attention = getattr(block, "self_attn", None)
        if index in prefix["kv"]:
            kv = prefix["kv"][index].to(device=device, dtype=dtype)
            _check_geometry(interface, index, kv.shape[2], kv.shape[3])
            first = int(prefix["kv_first"][index])
            window = int(getattr(layer, "sliding_window", 0) or 0)
            needed = min(tokens, window - 1) if window else tokens  # HF keeps window - 1
            if kv.shape[1] < needed or first + kv.shape[1] != tokens:
                raise ValueError(
                    f"layer {index}: {kv.shape[1]} K/V entries from {first} for a "
                    f"{tokens}-token prefix (needs the last {needed})")
            cache.update(kv[0].transpose(0, 1)[None], kv[1].transpose(0, 1)[None], index)
            if window:  # the mask and the next update count every prefix token, not only the cached
                layer.cumulative_length = tokens
        elif index in prefix["ssm"]:
            assert _is_state(layer), (index, type(layer).__name__)
            conv = prefix["conv"][index].to(device=device, dtype=dtype)[None]
            kernel = int(
                getattr(getattr(block, "linear_attn", None), "conv_kernel_size", None)
                or getattr(interface.config, "linear_conv_kernel_dim", None)
                or conv.shape[-1])
            if conv.shape[-1] < kernel:
                conv = torch.nn.functional.pad(conv, (kernel - conv.shape[-1], 0))
            layer.update_conv_state(conv[..., -kernel:], conv_kernel_size=kernel)
            layer.update_recurrent_state(prefix["ssm"][index].to(device)[None])
        elif not (_is_kv(layer) and getattr(attention, "is_kv_shared_layer", False)):
            # Gemma 4's shared-K/V layers read an earlier layer's K/V: no cache here or in vLLM.
            raise KeyError(
                f"layer {index} ({interface.layer_types[index]}): "
                "no K/V or state in the prefix")
    return cache, tokens


def _read_question(interface, cache, tokens, question_ids):
    """Reads the question into ``cache``: its last hidden state, the first step's full
    log-softmax (fp32, CPU) and the next position."""

    ids = torch.as_tensor(list(question_ids), dtype=torch.long, device=interface.device)
    if not ids.numel():
        raise ValueError("a question needs at least one token after the prefix")
    position = tokens + int(ids.numel())
    positions = torch.arange(tokens, position, device=interface.device)
    hidden = interface.forward(ids, positions, cache)[-1]
    return hidden, torch.log_softmax(interface.logits(hidden).float(), -1).cpu(), position


def _top(logprobs, k):
    if not k:
        return {}
    top = logprobs.topk(min(int(k), logprobs.numel()))
    return {int(i): float(v) for v, i in zip(top.values.tolist(), top.indices.tolist())}


def _rng(sampling, device):
    if sampling is None or sampling.greedy or sampling.seed is None:
        return None
    return torch.Generator(device=device).manual_seed(int(sampling.seed))


@torch.no_grad()
def _generate(
    interface, tokenizer, cache, tokens, question_ids, max_tokens, stops, sampling,
    top_logprobs, rng, *, keep_cache=False):
    """Generator over one answer: yields ``{"id", "text", "finish_reason"}`` per new token
    (``text``: the newly decodable text; ``finish_reason`` set on the last) and returns the answer
    record. ``cache`` is consumed; ``keep_cache``: it then also holds the last new token, and the
    record carries ``cache`` and ``tokens``."""

    device = interface.device
    strings = sampling.stop if sampling is not None else ()
    if strings and tokenizer is None:
        raise ValueError("stop strings need a tokenizer")
    stop = {int(token) for token in stops}
    stop |= set(sampling.stop_token_ids if sampling is not None else ())
    hidden, logprobs, position = _read_question(interface, cache, tokens, question_ids)
    prompt_tokens, new, emitted, finish = position, [], "", "length"
    text = "" if tokenizer is not None else None
    # Streamed text holds back what could still be the start of a stop string.
    hold = max((len(string) for string in strings), default=1) - 1
    while len(new) < int(max_tokens):
        token = pick(interface.logits(hidden), sampling, rng)
        new.append(token)
        done = token in stop
        if tokenizer is not None:
            text = tokenizer.decode(new, skip_special_tokens=True)
            if strings:
                text, hit = truncate_at_stop(text, strings)
                done = done or hit
        finish = "stop" if done else "length"
        last = done or len(new) >= int(max_tokens)
        delta = ""
        if text is not None:
            safe = text
            if not last:  # an incomplete character decodes to U+FFFD until its last byte arrives
                safe = text.rstrip("�")
                safe = safe[: len(safe) - hold] if hold else safe
            if len(safe) > len(emitted) and safe.startswith(emitted):
                delta, emitted = safe[len(emitted):], safe
        yield {"id": token, "text": delta, "finish_reason": finish if last else None}
        if last:
            break
        step = torch.tensor([token], device=device), torch.tensor([position], device=device)
        hidden = interface.forward(*step, cache)[-1]
        position += 1
    record = {
        "text": text,
        "ids": list(new),
        "first_logprobs": logprobs,
        "top_logprobs": _top(logprobs, top_logprobs),
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
    }
    if keep_cache:
        if new:  # the last new token was sampled but not read: read it so the cache holds the turn
            step = torch.tensor([new[-1]], device=device), torch.tensor([position], device=device)
            interface.forward(*step, cache)
            position += 1
        record["cache"], record["tokens"] = cache, position
    return record


def _sampled(
    interface, tokenizer, cache, tokens, question_ids, max_tokens, stops, sampling,
    top_logprobs):
    """``sampling.n`` answers, each on its own copy of ``cache`` (one seeded generator for all)."""

    rng = _rng(sampling, interface.device)
    records = []
    for index in range(sampling.n):
        own = cache if index == sampling.n - 1 else copy_cache(cache)
        events = _generate(
            interface, tokenizer, own, tokens, question_ids, max_tokens, stops,
            sampling, top_logprobs, rng)
        while True:
            try:
                next(events)
            except StopIteration as stop:
                records.append(stop.value)
                break
    out = records[0]
    if sampling.n > 1:
        keys = ("text", "ids", "finish_reason")
        out["samples"] = [{key: record[key] for key in keys} for record in records]
    return out


@torch.no_grad()
def _answer_from_cache(
    interface, tokenizer, cache, tokens, question_ids, max_tokens, stops,
    top_logprobs):
    hidden, logprobs, position = _read_question(interface, cache, tokens, question_ids)
    new = interface._decode(cache, hidden, position, int(max_tokens), stops) if max_tokens else []
    return {
        "text": tokenizer.decode(new, skip_special_tokens=True) if tokenizer is not None else None,
        "ids": list(new),
        "first_logprobs": logprobs,
        "top_logprobs": _top(logprobs, top_logprobs),
        "finish_reason": "stop" if new and new[-1] in set(stops) else "length",
        "prompt_tokens": position,
    }


@torch.no_grad()
def answer(
    model, tokenizer, prefix, prefix_ids, question_ids, *, max_tokens, stop_token_ids=None,
    top_logprobs=20, sampling=None):
    """Greedy answer to one question after ``prefix``: ``{text, ids, first_logprobs,
    top_logprobs, finish_reason, prompt_tokens}``. ``first_logprobs`` is the full log-softmax at
    the last question token (fp32, CPU); ``top_logprobs`` its top ``top_logprobs`` as
    ``{token: logprob}`` (vLLM's ``logprobs=k`` at step 0). ``sampling`` (``Sampling`` or dict):
    sampled instead (plus ``samples`` when ``n > 1``)."""

    interface = as_interface(model)
    stops = list(stop_token_ids) if stop_token_ids is not None else stop_ids(interface, tokenizer)
    cache, tokens = build_cache(interface, prefix, prefix_ids)
    sampling = as_sampling(sampling)
    if sampling is not None:
        return _sampled(
            interface, tokenizer, cache, tokens, question_ids, max_tokens, stops,
            sampling, top_logprobs)
    return _answer_from_cache(
        interface, tokenizer, cache, tokens, question_ids, max_tokens, stops,
        top_logprobs)


def stream(
    model, tokenizer, prefix, prefix_ids, question_ids, *, max_tokens, stop_token_ids=None,
    top_logprobs=20, sampling=None, keep_cache=False):
    """``answer`` token by token: yields ``{"id", "text", "finish_reason"}`` as each token is
    produced (``text``: the newly decodable text, held back while it could still be an incomplete
    character or the start of a stop string; the deltas join to the final text). The last event
    has ``finish_reason`` and ``result``: ``answer``'s record. Same tokens as ``answer`` with the
    same options (``n`` must be 1). ``keep_cache``: the record also carries ``cache`` (the prefix,
    question and answer read) and ``tokens``, e.g. to capture rows from it
    (``handoff.capture_rows``) or to ``continue_turn``."""

    interface = as_interface(model)
    stops = list(stop_token_ids) if stop_token_ids is not None else stop_ids(interface, tokenizer)
    cache, tokens = build_cache(interface, prefix, prefix_ids)
    yield from _events(
        interface, tokenizer, cache, tokens, question_ids, max_tokens, stops,
        as_sampling(sampling), top_logprobs, keep_cache)


def _events(
    interface, tokenizer, cache, tokens, question_ids, max_tokens, stops, sampling,
    top_logprobs, keep_cache):
    if sampling is not None and sampling.n != 1:
        raise ValueError("streaming and conversation turns produce one answer: n must be 1")
    if int(max_tokens) < 1:
        raise ValueError("max_tokens must be at least 1")
    events = _generate(
        interface, tokenizer, cache, tokens, question_ids, max_tokens, stops,
        sampling, top_logprobs, _rng(sampling, interface.device),
        keep_cache=keep_cache)
    # Each token as soon as it is picked; the last one (finish_reason set) also carries the
    # record, the generator's return value.
    for event in events:
        if event["finish_reason"] is None:
            yield event
            continue
        try:
            next(events)
        except StopIteration as stop:
            yield {**event, "result": stop.value}
        return


def continue_turn(
    model, tokenizer, cache, tokens, turn_ids, *, max_tokens, stop_token_ids=None,
    top_logprobs=20, sampling=None, stream=False):
    """One conversation turn on an HF cache already holding ``tokens`` tokens (a document prefix,
    then earlier turns): reads ``turn_ids`` at positions ``tokens...``, answers (greedy, or
    ``sampling``), and reads the answer's last token too, so the cache then holds prefix + turn +
    answer. Returns ``answer``'s record plus ``cache`` (the same object, extended in place: pass
    ``copy_cache(cache)`` to keep the original) and ``tokens`` (its new length). Start from
    ``build_cache(as_interface(model), prefix, prefix_ids)``; the next turn's ids:
    ``prompts.followup_ids(tokenizer, text, after=record["ids"])``. ``stream=True``: a generator
    of ``stream``'s events, the last one's ``result`` holding the record."""

    interface = as_interface(model)
    stops = list(stop_token_ids) if stop_token_ids is not None else stop_ids(interface, tokenizer)
    events = _events(
        interface, tokenizer, cache, int(tokens), turn_ids, max_tokens, stops,
        as_sampling(sampling), top_logprobs, True)
    if stream:
        return events
    for event in events:
        if "result" in event:
            return event["result"]


@torch.no_grad()
def reference_nll(model, prefix, prefix_ids, continuation_ids, start, *, cache=None):
    """Mean NLL of ``continuation_ids[start:]`` after ``prefix`` and ``continuation_ids[:start]``
    (``start`` = ``len(after_ids)``). ``cache`` (with ``prefix``'s length) is consumed if given."""

    interface = as_interface(model)
    if cache is None:
        cache, tokens = build_cache(interface, prefix, prefix_ids)
    else:
        tokens = prefix_tokens(prefix) if prefix is not None else len(prefix_ids)
    ids = torch.as_tensor(list(continuation_ids), dtype=torch.long, device=interface.device)
    if not 1 <= start < ids.numel():
        return None
    positions = torch.arange(tokens, tokens + ids.numel(), device=interface.device)
    hidden = interface.forward(ids, positions, cache)
    logprobs = torch.log_softmax(interface.logits(hidden[start - 1 : -1]).float(), -1)
    return float(-logprobs.gather(-1, ids[start:, None]).mean())


@torch.no_grad()
def answer_batch(
    model, tokenizer, prefix, prefix_ids, questions, *, max_tokens=None,
    stop_token_ids=None, top_logprobs=20, sampling=None):
    """``answer`` for several questions sharing one prefix: the cache is built once and copied per
    question. ``questions``: id lists (each answered with ``max_tokens``) or question dicts
    or ``prompts.Question``s (``after_ids``, ``max_tokens``, ``metric``; ``nll`` ones return ``{"nll": ...}`` from
    ``continuation_ids``). ``sampling``: for every question, unless a question dict carries its
    own ``"sampling"``."""

    interface = as_interface(model)
    stops = list(stop_token_ids) if stop_token_ids is not None else stop_ids(interface, tokenizer)
    base, tokens = build_cache(interface, prefix, prefix_ids)
    out = []
    for question in questions:
        question = question_dict(question)
        if not isinstance(question, dict):
            question = {"after_ids": question}
        after = list(question["after_ids"])
        cache = copy_cache(base)
        if question.get("metric") == "nll":
            nll = reference_nll(
                interface, prefix, prefix_ids, question.get("continuation_ids"),
                len(after), cache=cache)
            out.append({"nll": nll})
            continue
        limit = question.get("max_tokens", max_tokens)
        if limit is None:
            raise ValueError("max_tokens is required (prompts.GENERATION_TOKENS[metric])")
        own = as_sampling(question.get("sampling", sampling))
        args = interface, tokenizer, cache, tokens, after, limit, stops
        if own is None:
            out.append(_answer_from_cache(*args, top_logprobs))
        else:
            out.append(_sampled(*args, own, top_logprobs))
    return out

