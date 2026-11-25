"""The unified translator: configuration and geometries, layers, local-window attention and
the model (shared encoder, shared decoder, per-target heads)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
import copy
import dataclasses
from typing import Mapping


# ---- Configuration -----------------------------------------------------------------------------
# Model geometries and the hyperparameters of the unified translator.

@dataclass(frozen=True)
class SourceGeometry:
    """Hidden states read from a source model.

    ``streams`` counts hidden-state streams including the embedding output, so a
    36-layer model has 37.
    """

    streams: int
    width: int


@dataclass(frozen=True)
class TargetGeometry:
    """Layer-input hidden states written for a target decoder.

    ``layer_types`` is optional for backwards compatibility. ``None`` means an
    ordinary full-attention decoder; Hugging Face adapters may populate it with
    the target's heterogeneous cache layer kinds.
    """

    layers: int
    width: int
    layer_types: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        assert self.layers >= 1 and self.width >= 1
        if self.layer_types is not None:
            assert len(self.layer_types) == self.layers, "one type per layer"
            object.__setattr__(self, "layer_types", tuple(self.layer_types))


REFERENCE_SOURCES = {
    "qwen3-8b": SourceGeometry(streams=37, width=4096),
    "llama31-8b": SourceGeometry(streams=33, width=4096),
    "granite33-8b": SourceGeometry(streams=41, width=4096),
}

REFERENCE_TARGETS = {
    "qwen3-8b": TargetGeometry(layers=36, width=4096),
    "llama31-8b": TargetGeometry(layers=32, width=4096),
    "granite33-8b": TargetGeometry(layers=40, width=4096),
}


@dataclass(frozen=True)
class UnifiedConfig:
    """Shape of the shared translator. The defaults are the locked reference.

    ``decoder_streams`` sets how many depth streams the decoder runs per target
    token. ``None`` (the reference) gives one stream per target layer. A smaller
    number runs the decoder over that many streams and maps them back to every
    target layer with a learned per-target depth readout that starts as linear
    interpolation (the 16-stream decoder: ``decoder_streams=16``).
    """

    width: int = 512
    heads: int = 8
    ffn: int = 2048
    experts: int = 8
    top_k: int = 2
    encoder_blocks: int = 2
    decoder_blocks: int = 8
    window: int = 8
    decoder_streams: int | None = None
    # Bias on each per-layer output map P^out (the trained reference has one;
    # the paper's eq. 1 has none).
    output_bias: bool = True
    # Target-owned decoder blocks after the shared decoder (0 = none). They use
    # one always-on FFN plus one expert and start as an identity, so they add
    # per-target capacity without changing a shared block.
    target_blocks: int = 0

    def __post_init__(self) -> None:
        assert self.width % self.heads == 0, "width must split into heads"
        assert (self.width // self.heads) % 2 == 0, "head size must be even"
        assert 1 <= self.top_k <= self.experts
        assert self.encoder_blocks >= 0 and self.decoder_blocks >= 1
        assert self.window >= 1 and self.target_blocks >= 0
        assert self.decoder_streams is None or self.decoder_streams >= 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "UnifiedConfig":
        return cls(**values)


# ---- Layers ------------------------------------------------------------------------------------
# Shared building blocks: normalization, row helpers and the routed
# feed-forward layer.

def repeat_layers(per_stream: torch.Tensor, tokens: int) -> torch.Tensor:
    """``[streams, width]`` -> token-major rows ``[tokens * streams,
    width]``."""

    streams, width = per_stream.shape
    rows = per_stream.unsqueeze(0).expand(tokens, -1, -1)
    return rows.reshape(tokens * streams, width)


def unit_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.float().square().mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_square + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight.to(x.dtype)


class RoutedFeedForward(nn.Module):
    """Always-on GELU FFN plus the top-k of ``experts`` full-width GELU experts.

    For normalized content ``c`` and a per-stream shift ``s`` (a block's map of
    the model-layer state), both paths read ``z = c + s``. The router scores the
    content and, through a separate zero-initialized map, the normalized shift:

        logits = W_r c + W_s rms(s)

    Expert choice adds a non-learned balancing bias (``selection_bias``, set
    in training); gates are the chosen probabilities renormalized. No token is dropped.
    """

    def __init__(self, width: int, ffn: int, experts: int, top_k: int):
        super().__init__()
        assert 1 <= top_k <= experts
        self.experts, self.top_k = experts, top_k
        self.up = nn.Linear(width, ffn, bias=False)
        self.down = nn.Linear(ffn, width, bias=False)
        nn.init.zeros_(self.down.weight)
        self.router = nn.Linear(width, experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.state_router = nn.Linear(width, experts, bias=False)
        nn.init.zeros_(self.state_router.weight)
        bound = 1.0 / math.sqrt(width)
        self.expert_up = nn.Parameter(
            torch.empty(experts, width, ffn).uniform_(-bound, bound))
        self.expert_down = nn.Parameter(torch.zeros(experts, ffn, width))
        self.register_buffer("selection_bias", torch.zeros(experts))
        self.register_buffer("load_ema", torch.full((experts,), 1.0 / experts))
        self.register_buffer("load_counts", torch.zeros(experts), persistent=False)
        self.last_balance: torch.Tensor | None = None

    def forward(self, content: torch.Tensor, shift: torch.Tensor,
            tokens: int) -> torch.Tensor:
        """``content`` is ``[tokens * streams, width]``; ``shift`` is
        ``[streams, width]``.

        The Switch-style balance of this pass is stored in ``last_balance`` for
        logging; it is not a training loss.
        """

        z = content + repeat_layers(shift.to(content.dtype), tokens)
        shift_rms = unit_rms(shift).to(self.state_router.weight.dtype)
        shift_logits = self.state_router(shift_rms).float()
        logits = self.router(content).float() + repeat_layers(
            shift_logits, tokens)
        probabilities = logits.softmax(dim=-1)
        biased = logits.detach() + self.selection_bias.float()
        chosen = biased.topk(self.top_k, dim=-1).indices
        picked = probabilities.gather(-1, chosen)
        weights = picked
        if self.top_k > 1:
            weights = picked / picked.sum(dim=-1, keepdim=True)
        # Rows sorted by expert: each expert then reads one contiguous slice.
        flat = chosen.reshape(-1)
        order = torch.argsort(flat, stable=True)
        expert_counts = torch.bincount(flat, minlength=self.experts)
        # Start the expert counts' copy to the host; the always-on FFN below
        # keeps the GPU busy while they arrive.
        host_counts, ready = expert_counts, None
        if expert_counts.is_cuda:
            host_counts = torch.empty(expert_counts.shape,
                dtype=expert_counts.dtype, pin_memory=True)
            host_counts.copy_(expert_counts, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        counts = expert_counts.float()
        # The graph task id is -1 except while autograd recomputes a
        # checkpointed forward, whose counts were already added.
        probe = getattr(torch._C, "_current_graph_task_id", None)
        in_backward = probe is not None and probe() != -1
        if self.training and not in_backward:
            self.load_counts.add_(counts.to(self.load_counts.device))
        load = counts / counts.sum().clamp_min(1.0)
        balance = (load * probabilities.mean(dim=0)).sum()
        self.last_balance = (self.experts * balance).detach()
        dense = self.down(F.gelu(self.up(z)))
        if ready is not None:
            ready.synchronize()
        rows = order // self.top_k
        inputs = z.index_select(0, rows).split(host_counts.tolist())
        ups = self.expert_up.to(z.dtype).unbind(0)
        downs = self.expert_down.to(z.dtype).unbind(0)
        outputs = []
        for part, up, down in zip(inputs, ups, downs):
            # On CUDA, pad the rows to a repeating size (a multiple of 256 and
            # of 1/8 of the power of two at or below the count: under 12.5%
            # padding): cuBLAS picks an algorithm on the host per new shape,
            # which is slow. Padded rows give gelu(0) @ down = 0 and are
            # dropped.
            count = part.shape[0]
            pad = 0
            if part.is_cuda and count:
                step = max(256, 1 << max(0, count.bit_length() - 4))
                pad = -(-count // step) * step - count
            if pad:
                part = F.pad(part, (0, 0, 0, pad))
                outputs.append((F.gelu(part @ up) @ down)[:count])
            else:
                outputs.append(F.gelu(part @ up) @ down)
        outputs = torch.cat(outputs)
        order_weights = weights.reshape(-1).index_select(0, order)
        scaled = outputs * order_weights[:, None].to(outputs.dtype)
        return dense + torch.zeros_like(dense).index_add(0, rows, scaled)


# ---- Attention ---------------------------------------------------------------------------------
# Local-window STILL attention.
#
# Rows are token-major ``(token, stream)`` pairs. A query row of token ``t`` reads
# every stream of the key tokens ``last - window < s <= last``, where ``last`` is
# ``t`` itself for self-attention and the last readable source token for the
# memory read. Q and K are unit-normalized and scaled by a learned per-head
# sharpness (STILL), so scores use scale 1. A row with no readable key token
# returns zeros.
#
# Attention runs per chunk of ``chunk`` query tokens. A chunk reads one
# contiguous range of key tokens (its window), so every chunk goes through a
# single batched SDPA call with a boolean mask, and keys are copied about
# ``(chunk + window - 1) / chunk`` times. Everything that depends only on the
# alignment (chunk ranges, masks, gather indices) is built once per forward by
# ``WindowPlan`` and shared by every block.

ROPE_THETA = 10_000.0
ATTENTION_CHUNK = 16


def rope_tables(positions: torch.Tensor, dim: int, theta: float = ROPE_THETA
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``(cos, sin)`` tables of shape ``[tokens, dim / 2]``, one row per
    token."""

    half = dim // 2
    exponents = torch.arange(half, device=positions.device, dtype=torch.float32)
    frequencies = theta ** (-exponents / half)
    angles = positions.float()[:, None] * frequencies[None]
    return angles.cos(), angles.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary positions on interleaved pairs. ``x`` is ``[tokens, streams,
    heads, dim]``; tables are per token."""

    cos, sin = cos.to(x.dtype)[:, None, None], sin.to(x.dtype)[:, None, None]
    # unflatten + unbind rather than strided slices: the gradient is a stack,
    # not a slice scatter, which keeps the function compilable under dynamic
    # shapes.
    even, odd = x.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = (even * cos - odd * sin, even * sin + odd * cos)
    return torch.stack(rotated, dim=-1).flatten(-2)


class WindowPlan:
    """Chunk layout of one local-window attention, built on the host once per
    forward.

    ``last`` is the last readable key token of every query token (``-1``:
    none). Query tokens are grouped in chunks of ``chunk``; chunk ``c`` reads
    key tokens ``start[c] .. start[c] + span - 1`` (clamped), and a boolean
    mask keeps each row's own window. Self-attention (``strided``: ``last[t] = t`` over the same
    tokens) reads its range as a strided window of the padded keys; other reads
    gather it.

    The layout is the same for every document shape (chunk size, key length a
    multiple of 8, a ``keep`` mask even when no row is empty), so the compiled
    attention never branches on the document.
    """

    def __init__(self, last, *, key_tokens: int, window: int,
            query_streams: int, key_streams: int, chunk: int = ATTENTION_CHUNK,
            strided: bool = False, device=None):
        last = torch.as_tensor(last, dtype=torch.long).cpu()
        self.tokens, self.key_tokens = int(last.numel()), int(key_tokens)
        self.window = int(window)
        self.query_streams = int(query_streams)
        self.key_streams = int(key_streams)
        T, W = self.tokens, self.window
        assert not T or int(last.max()) < self.key_tokens
        self.strided = bool(strided)
        # A strided plan is self-attention: last[t] = t over the same tokens.
        assert not self.strided or (
            T == self.key_tokens and torch.equal(last, torch.arange(T)))
        self.chunk = B = max(1, int(chunk))
        # At least one padding row and at least two chunks: the padded length
        # then never equals the token count and the chunk dimension is never 1,
        # which a compiled caller would otherwise treat as special shapes.
        self.chunks = C = max(2, T // B + 1) if T else 0
        padded = torch.empty(C * B, dtype=torch.long)
        padded[:T] = last
        # Padding rows are discarded; they must not widen the last chunk.
        padded[T:] = last[-1] if T else 0
        by_chunk = padded.view(C, B)
        start = by_chunk.min(dim=1).values - (W - 1)
        if self.strided:
            span = B + W - 1
        elif C:
            span = int((by_chunk.max(dim=1).values - start).max()) + 1
        else:
            span = 0
        # Round the key tokens up to a multiple of 8, so the key length is
        # aligned for SDPA for any stream count and the self-attention span is
        # the same for every geometry; the extra key tokens are masked out.
        self.span = S = -(-span // 8) * 8
        key_token = start[:, None] + torch.arange(S)[None, :]
        keys, rows = key_token[:, None, :], by_chunk[:, :, None]
        visible = (keys <= rows) & (keys > rows - W)
        visible &= (keys >= 0) & (keys < self.key_tokens)
        empty = ~visible.any(dim=-1)
        # Empty rows read key column 0 so the softmax stays finite; their
        # output is zeroed.
        visible[:, :, 0] |= empty
        Sq, Sk = self.query_streams, self.key_streams
        # Copy the token-level [C, B, S] mask and expand it to rows on the
        # device (the row mask is Sq * Sk times larger).
        visible = visible.to(device)
        self.mask = visible[:, :, None, :, None].expand(C, B, Sq, S, Sk)
        self.mask = self.mask.reshape(C, 1, B * Sq, S * Sk)
        self.index = None
        if not self.strided:
            clamped = key_token.clamp(0, max(self.key_tokens - 1, 0))
            self.index = clamped.reshape(-1).to(device)
        self.keep = (~empty.reshape(-1)[:T]).repeat_interleave(Sq).to(device)


def window_attention(query: torch.Tensor, keys: torch.Tensor,
        values: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """``query`` ``[tokens, query_streams, heads, dim]`` (already scaled) reads
    ``keys``/``values`` ``[key_tokens, key_streams, heads, dim]`` through
    ``plan``; returns ``[tokens * query_streams, heads * dim]``."""

    T, Sq, H, D = query.shape
    if T == 0:
        return query.reshape(0, H * D)
    C, B, S, W = plan.chunks, plan.chunk, plan.span, plan.window
    q = F.pad(query, (0, 0, 0, 0, 0, 0, 0, C * B - T))
    q = q.view(C, B * Sq, H, D).transpose(1, 2)

    def windows(t):
        if plan.strided:
            # Each chunk's tokens plus the W - 1 before it (and alignment
            # padding after), as one strided copy: the backward is a sum, with
            # no atomics.
            after = C * B - T + S - (B + W - 1)
            t = F.pad(t, (0, 0, 0, 0, 0, 0, W - 1, after))
            t = t.unfold(0, S, B).permute(0, 4, 1, 2, 3)
        else:
            t = t.index_select(0, plan.index)
        return t.reshape(C, -1, H, D).transpose(1, 2)

    read = F.scaled_dot_product_attention(q, windows(keys), windows(values),
        attn_mask=plan.mask, scale=1.0)
    read = read.transpose(1, 2).reshape(C * B * Sq, H * D)
    # A split rather than a slice: its gradient is a cat, not a slice scatter,
    # and a compiled caller has no shape branch (the remainder may be empty).
    read = read.split((T * Sq, read.shape[0] - T * Sq))[0]
    return torch.where(plan.keep[:, None], read, torch.zeros_like(read))


class SelfAttention(nn.Module):
    """Local attention among one sequence's own rows."""

    def __init__(self, width: int, heads: int, window: int):
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.logit_scale = nn.Parameter(
            torch.full((heads,), math.log(width // heads)))

    def forward(self, x, *, tokens: int, plan: WindowPlan, rope_table):
        """``x`` is ``[tokens * streams, width]``; ``rope_table`` is
        ``(cos, sin)`` per token."""

        head = x.shape[-1] // self.heads
        qkv = self.qkv(x).view(tokens, -1, 3, self.heads, head)
        query, key, value = qkv.unbind(2)
        scale = self.logit_scale.exp().to(x.dtype)[:, None]
        query = rope(F.normalize(query, dim=-1), *rope_table) * scale
        key = rope(F.normalize(key, dim=-1), *rope_table)
        return self.out(window_attention(query, key, value, plan))


class CrossAttention(nn.Module):
    """Target rows read the source memory.

    Queries come only from the target rows and keys/values only from the memory,
    so there is no target key/value projection to compute and discard.
    """

    def __init__(self, width: int, heads: int, window: int):
        super().__init__()
        self.heads, self.window = heads, window
        self.query = nn.Linear(width, width, bias=False)
        self.memory_norm = RMSNorm(width)
        self.memory_kv = nn.Linear(width, 2 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.logit_scale = nn.Parameter(
            torch.full((heads,), math.log(width // heads)))

    def forward(self, x, memory, *, tokens: int, plan: WindowPlan, rope_table,
            memory_rope_table):
        """``x`` is ``[tokens * streams, width]``; ``memory`` is
        ``[source_tokens, source_streams, width]``."""

        head = x.shape[-1] // self.heads
        query = self.query(x).view(tokens, -1, self.heads, head)
        query = rope(F.normalize(query, dim=-1), *rope_table)
        query = query * self.logit_scale.exp().to(x.dtype)[:, None]
        kv = self.memory_kv(self.memory_norm(memory))
        kv = kv.view(*memory.shape[:2], 2, self.heads, head)
        key, value = kv.unbind(2)
        key = rope(F.normalize(key, dim=-1), *memory_rope_table)
        return self.out(window_attention(query, key, value, plan))


# ---- Translator --------------------------------------------------------------------------------
# Additive unified-state K/V translator: a shared encoder over source hidden
# states, a shared decoder per target token and per-target output heads.

def _key(name: str) -> str:
    return name.replace(".", "_")


def _interp(layers: int, streams: int) -> torch.Tensor:
    position = torch.linspace(0, streams - 1, layers)
    low, high = position.floor().long(), position.ceil().long()
    fraction = position - low.float()
    matrix = torch.zeros(layers, streams)
    matrix.scatter_(1, low[:, None], (1 - fraction)[:, None])
    matrix.scatter_add_(1, high[:, None], fraction[:, None])
    return matrix


class Layout:
    """Everything one encoder or decoder forward shares across its blocks:
    attention plans and RoPE tables.

    Built once per forward (one host read of the alignment), then reused by
    every block.
    """

    def __init__(self, tokens: int, streams: int, positions: torch.Tensor,
            config: UnifiedConfig, *, memory_tokens: int = 0,
            memory_streams: int = 0, memory_positions=None, limits=None,
            chunk: int = ATTENTION_CHUNK):
        device, head = positions.device, config.width // config.heads
        self.tokens, self.streams = int(tokens), int(streams)
        self.self_plan = WindowPlan(torch.arange(tokens), key_tokens=tokens,
            window=config.window, query_streams=streams, key_streams=streams,
            chunk=chunk, strided=True, device=device)
        self.rope = rope_tables(positions, head)
        self.cross_plan = self.memory_rope = None
        if limits is not None:
            last = torch.as_tensor(limits).cpu().long() - 1
            self.cross_plan = WindowPlan(last, key_tokens=memory_tokens,
                window=config.window, query_streams=streams,
                key_streams=memory_streams, chunk=chunk, device=device)
            self.memory_rope = rope_tables(memory_positions.to(device), head)


def _self_half(block, x, self_shift, layout: Layout):
    """Self attention and its residual add, plus the FFN input norm: no host
    sync, so it compiles whole."""

    tokens = layout.tokens
    shift = repeat_layers(self_shift, tokens).to(x.dtype)
    x = x + block.self_attn(block.norm_self(x) + shift, tokens=tokens,
        plan=layout.self_plan, rope_table=layout.rope)
    return x, block.norm_ffn(x)


def _attention_half(block, x, self_shift, cross_shift, source, layout: Layout):
    """Self attention, the source-memory read and both residual adds, plus the
    FFN input norm."""

    tokens = layout.tokens
    shift = repeat_layers(self_shift, tokens).to(x.dtype)
    x = x + block.self_attn(block.norm_self(x) + shift, tokens=tokens,
        plan=layout.self_plan, rope_table=layout.rope)
    query = block.norm_cross(x) + repeat_layers(cross_shift, tokens).to(x.dtype)
    x = x + block.cross_attn(query, source, tokens=tokens,
        plan=layout.cross_plan, rope_table=layout.rope,
        memory_rope_table=layout.memory_rope)
    return x, block.norm_ffn(x)


_COMPILED: dict = {}
COMPILE_BACKEND = "inductor"


def _compiled(function):
    """``function`` under ``torch.compile`` with dynamic shapes, compiled once
    per process.

    Weights that change shape across models (state tables) are mapped outside,
    so the compiled graphs only see activations and the shared block weights.
    """

    if function not in _COMPILED:
        import torch._dynamo
        import torch.fx.experimental._config as fx_config

        dynamo = torch._dynamo.config
        dynamo.cache_size_limit = max(dynamo.cache_size_limit, 64)
        # Unrelated sizes that happen to match on the first call (e.g. source
        # and target token counts) must not be assumed equal afterwards.
        fx_config.use_duck_shape = False
        _COMPILED[function] = torch.compile(function, dynamic=True,
            backend=COMPILE_BACKEND)
    return _COMPILED[function]


class UnifiedBlock(nn.Module):
    """Self attention, optional source-memory read, and routed FFN."""

    def __init__(self, config: UnifiedConfig, *, memory: bool = False):
        super().__init__()
        width = config.width
        self.memory = memory
        self.norm_self = RMSNorm(width)
        self.self_attn = SelfAttention(width, config.heads, config.window)
        self.self_state_map = nn.Linear(width, width, bias=False)
        if memory:
            self.norm_cross = RMSNorm(width)
            self.cross_attn = CrossAttention(width, config.heads, config.window)
            self.cross_state_map = nn.Linear(width, width, bias=False)
            self.memory_state_map = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.memory_state_map.weight)
        self.norm_ffn = RMSNorm(width)
        self.ffn = RoutedFeedForward(width, config.ffn, config.experts,
            config.top_k)
        self.ffn_state_map = nn.Linear(width, width, bias=False)
        self.compiled = False

    def forward(self, x, *, state, layout: Layout, memory=None,
            memory_state=None):
        """``x`` is ``[tokens * streams, width]``; ``memory`` is
        ``[source_tokens, source_streams, width]``."""

        if memory is None:
            attend = _compiled(_self_half) if self.compiled else _self_half
            x, content = attend(self, x, self.self_state_map(state), layout)
        else:
            assert self.memory and layout.cross_plan is not None
            assert memory_state is not None
            source = memory + self.memory_state_map(memory_state)[None]
            attend = _attention_half
            if self.compiled:
                attend = _compiled(_attention_half)
            x, content = attend(self, x, self.self_state_map(state),
                self.cross_state_map(state), source, layout)
        update = self.ffn(content, self.ffn_state_map(state), layout.tokens)
        balance = x.new_zeros((), dtype=torch.float32)
        if self.ffn.last_balance is not None:
            balance = balance + self.ffn.last_balance
        return x + update, balance


class UnifiedEncoder(nn.Module):
    def __init__(self, sources: Mapping[str, SourceGeometry],
            config: UnifiedConfig):
        super().__init__()
        self.sources, self.config = dict(sources), config
        self.gains, self.maps = nn.ParameterDict(), nn.ParameterDict()
        self.states = nn.ParameterDict()
        for model, geometry in self.sources.items():
            bound = 1.0 / math.sqrt(geometry.width)
            shape = (geometry.streams, geometry.width)
            self.gains[_key(model)] = nn.Parameter(torch.ones(shape))
            self.maps[_key(model)] = nn.Parameter(
                torch.empty(*shape, config.width).uniform_(-bound, bound))
            self.states[_key(model)] = nn.Parameter(
                torch.randn(geometry.streams, config.width) * 0.02)
        self.blocks = nn.ModuleList(
            UnifiedBlock(config) for _ in range(config.encoder_blocks))
        # Query tokens per attention chunk (speed only).
        self.attention_chunk = ATTENTION_CHUNK

    def forward(self, model, rows, state=None):
        geometry, name = self.sources[model], _key(model)
        state = self.states[name] if state is None else state
        assert tuple(rows.shape[1:]) == (geometry.streams, geometry.width), (
            f"{model} rows have shape {tuple(rows.shape)}")
        mean_square = rows.float().square().mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + 1e-6).to(rows.dtype)
        gains = self.gains[name].to(rows.dtype).unsqueeze(0)
        projected = torch.bmm((rows * scale * gains).transpose(0, 1),
            self.maps[name].to(rows.dtype)).transpose(0, 1)
        tokens, streams, width = projected.shape
        x = projected.reshape(tokens * streams, width)
        layout = Layout(tokens, streams, torch.arange(tokens, device=rows.device),
            self.config, chunk=self.attention_chunk)
        balance = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            x, part = block(x, state=state, layout=layout)
            balance = balance + part
        return x.view(tokens, streams, width), balance / max(1, len(self.blocks))


class UnifiedDecoder(nn.Module):
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(UnifiedBlock(config, memory=True)
            for _ in range(config.decoder_blocks))
        # Query tokens per attention chunk (speed only).
        self.attention_chunk = ATTENTION_CHUNK

    def forward(self, start, *, state, memory, source_state, source_positions,
            target_positions, limits, extra_blocks=None):
        tokens, streams, width = start.shape
        x = start.reshape(tokens * streams, width)
        layout = Layout(tokens, streams, target_positions.to(x.device),
            self.config, memory_tokens=memory.shape[0],
            memory_streams=memory.shape[1], memory_positions=source_positions,
            limits=limits, chunk=self.attention_chunk)
        inputs = dict(state=state, layout=layout, memory=memory,
            memory_state=source_state)
        balance = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            x, part = block(x, **inputs)
            balance = balance + part
        for block in extra_blocks or ():
            # Target-owned blocks have one expert, so their balance is
            # constant and is not logged.
            x, _ = block(x, **inputs)
        return x.view(tokens, streams, width), balance / max(1, len(self.blocks))


class UnifiedHead(nn.Module):
    def __init__(self, geometry: TargetGeometry, config: UnifiedConfig):
        super().__init__()
        self.geometry = geometry
        self.streams = config.decoder_streams or geometry.layers
        self.state = nn.Parameter(torch.randn(self.streams, config.width) * 0.02)
        self.embed_norm = RMSNorm(geometry.width)
        self.embed_proj = nn.Linear(geometry.width, config.width, bias=False)
        self.norms = nn.ModuleList(
            RMSNorm(config.width) for _ in range(geometry.layers))
        first = nn.Linear(config.width, geometry.width, bias=config.output_bias)
        self.maps = nn.ModuleList(
            [first] + [copy.deepcopy(first) for _ in range(geometry.layers - 1)])
        self.depth_readout = None
        if config.decoder_streams is not None:
            self.depth_readout = nn.Parameter(
                _interp(geometry.layers, self.streams))
        self.tail = None
        if config.target_blocks:
            tail = dataclasses.replace(config, experts=1, top_k=1,
                target_blocks=0)
            self.tail = nn.ModuleList(UnifiedBlock(tail, memory=True)
                for _ in range(config.target_blocks))
            for block in self.tail:
                nn.init.zeros_(block.self_attn.out.weight)
                nn.init.zeros_(block.cross_attn.out.weight)

    def start(self, embeds):
        start = self.embed_proj(self.embed_norm(embeds))
        return start[:, None, :].expand(-1, self.streams, -1)

    def forward(self, states):
        """``[tokens, streams, width]`` decoder states -> ``[tokens, layers,
        target_width]`` target hidden states.

        Every layer's norm and output map run as one batched matmul (the same
        math as applying each layer's ``norms[l]`` then ``maps[l]``), which
        saves two kernel launches per layer in each direction."""

        if self.depth_readout is not None:
            states = torch.einsum("tsd,ls->tld", states,
                self.depth_readout.to(states.dtype))
        mean_square = states.float().square().mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_square + self.norms[0].eps)
        gains = torch.stack([norm.weight for norm in self.norms])
        normed = (states.float() * scale).to(states.dtype)
        normed = normed * gains.to(states.dtype)
        weights = torch.stack([mapping.weight for mapping in self.maps])
        if self.maps[0].bias is None:
            out = torch.bmm(normed.transpose(0, 1), weights.transpose(1, 2))
            return out.transpose(0, 1)
        biases = torch.stack([mapping.bias for mapping in self.maps])
        out = torch.baddbmm(biases[:, None, :], normed.transpose(0, 1),
            weights.transpose(1, 2))
        return out.transpose(0, 1)


class UnifiedTranslator(nn.Module):
    def __init__(self, target_geometries, *, source_geometries=None,
            config=None):
        super().__init__()
        self.config = config or UnifiedConfig()
        sources = source_geometries or REFERENCE_SOURCES
        self.sources = {
            model: value if isinstance(value, SourceGeometry)
            else SourceGeometry(**value) for model, value in sources.items()}
        self.targets = {
            model: value if isinstance(value, TargetGeometry)
            else TargetGeometry(**value)
            for model, value in target_geometries.items()}
        self.encoder = UnifiedEncoder(self.sources, self.config)
        self.decoder = UnifiedDecoder(self.config)
        self.heads = nn.ModuleDict({
            _key(model): UnifiedHead(geometry, self.config)
            for model, geometry in self.targets.items()})

    def head(self, target):
        return self.heads[_key(target)]

    def set_compiled_attention(self, enabled: bool = True) -> None:
        """Run each block's attention half under ``torch.compile`` (same math;
        CUDA training and inference)."""

        for block in (*self.encoder.blocks, *self.decoder.blocks):
            block.compiled = bool(enabled)

    def decode(self, target, source, target_embeds, source_rows,
            source_positions, target_positions, limits):
        source_state = self.encoder.states[_key(source)]
        memory, encoder_balance = self.encoder(source, source_rows,
            source_state)
        head = self.head(target)
        states, decoder_balance = self.decoder(head.start(target_embeds),
            state=head.state, memory=memory, source_state=source_state,
            source_positions=source_positions,
            target_positions=target_positions, limits=limits,
            extra_blocks=head.tail)
        return head(states), encoder_balance + decoder_balance

