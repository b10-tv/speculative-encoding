"""No-grad inference executor: local-window attention without gathered key windows.

Based on Jeff Zheng's native-window executor. The eager path (``model.window_attention``)
copies each chunk's key window and runs one dense masked SDPA per chunk. Here PyTorch's CUTLASS
memory-efficient kernel runs in its packed variable-length form instead: every query token's
rows form one sequence, and its key interval (``seqstart_k`` + ``seqlen_k``, intervals may
overlap) is its own window, read in place.

Exactness: each interval starts on the same 64-key block grid as the eager chunk's window, so
the kernel visits the same key blocks in the same order as the eager call, minus blocks that
are fully masked for the row (which add exact zeros there). Keys are front-padded by ``window``
tokens so no interval starts before row 0. The kernel ABI is private: Torch 2.7.0
(the paper's) and 2.11.0 declare it the same way, and ``install`` checks it.

``install(level)`` patches the package process-wide (training is unaffected: every patched path
falls back to eager while gradients are enabled). It resets torch.compile caches so compiled
attention halves retrace with the active implementation. Level 1 is this attention (bitwise);
levels 2-4 add near-exact pieces (see ``LEVELS``). Level 4 compiles each whole block (state
maps, both attention halves, routing, dense FFN, residuals) with the window attention and the
grouped experts as opaque custom ops, so Inductor fuses the element-wise glue around the fast
kernels.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.nn import functional as F

from speculative_encoding import settings
from speculative_encoding.translator import model as M
from speculative_encoding.translator.model import repeat_layers, unit_rms


# ---- Tiled window attention kernel -------------------------------------------------------
# Tiled local-window attention for the no-grad executor (``PMT_FAST_TILED_ATTENTION=1`` on top of level 5+).
#
# Rows are token-major (``row = token * streams + stream``) and every query row of token ``t`` reads the contiguous key
# rows ``[first[t] * Sk, (last[t] + 1) * Sk)``. The FlashAttention-2 varlen path runs one sequence per query token
# (``Sq`` rows: 40 of a 128-row tile on a 40-stream target) and re-reads each key row for every token whose window
# covers it (``W`` times). Here a program takes ``BLOCK_M`` consecutive query rows (several tokens), walks the union
# of their key ranges once in ``BLOCK_N`` blocks and masks each row to its own range, so tiles are full and a key block
# is loaded once per query tile.
#
# Same keys, same online softmax (exp2, FP32 accumulators, BF16 ``P @ V``) as FlashAttention-2; only the key block
# boundaries and order differ, so outputs agree to FP32 accumulation rounding before the BF16 output cast. Rows with no
# key (``last < 0``) return zeros.

try:  # CPU-only environments (tests) have no Triton; the op's CPU path is ``tiled_reference``.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

LOG2E = 1.4426950408889634


def head_split(dim: int):
    """``dim`` as a power of two plus a power-of-two tail (96 = 64 + 32), or ``None`` if not expressible (>= 16 each)."""

    head = 1 << (dim.bit_length() - 1)
    tail = dim - head
    if head < 16 or (tail and (tail < 16 or tail & (tail - 1))):
        return None
    return head, tail


if triton is not None:
    _CONFIGS = [triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
                for m, n, w, s in ((128, 64, 8, 3), (128, 64, 4, 3), (64, 64, 4, 3), (128, 128, 8, 2), (64, 128, 4, 2), (128, 32, 4, 3),
                                   (128, 64, 8, 4), (128, 32, 8, 4), (256, 64, 8, 3), (256, 32, 8, 3), (64, 32, 4, 4))]

    @triton.autotune(configs=_CONFIGS, key=["Sq", "Sk", "D0", "D1"])
    @triton.jit
    def _tiled_window_attention(Q, K, V, O, First, Last, rows, Sq, Sk,
                                sqr, sqh, skr, skh, svr, svh, sor, soh,
                                D0: tl.constexpr, D1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid, h = tl.program_id(0), tl.program_id(1)
        m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = m < rows
        token = tl.where(row_ok, m // Sq, 0)
        first = tl.load(First + token, mask=row_ok, other=0)
        last = tl.load(Last + token, mask=row_ok, other=-1)
        lo = first * Sk
        hi = tl.where(last >= first, (last + 1) * Sk, lo)  # empty rows (last < 0): no key
        kv_lo = tl.min(tl.where(hi > lo, lo, 2147483647), 0)
        kv_hi = tl.max(hi, 0)
        d0 = tl.arange(0, D0)
        q0 = tl.load(Q + m[:, None].to(tl.int64) * sqr + h * sqh + d0[None, :], mask=row_ok[:, None], other=0.0)
        acc0 = tl.zeros((BLOCK_M, D0), dtype=tl.float32)
        if D1 > 0:
            d1 = D0 + tl.arange(0, D1)
            q1 = tl.load(Q + m[:, None].to(tl.int64) * sqr + h * sqh + d1[None, :], mask=row_ok[:, None], other=0.0)
            acc1 = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(kv_lo, kv_hi, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            n_ok = n < kv_hi
            n64 = n.to(tl.int64)
            k0 = tl.load(K + n64[None, :] * skr + h * skh + d0[:, None], mask=n_ok[None, :], other=0.0)
            s = tl.dot(q0, k0)
            if D1 > 0:
                k1 = tl.load(K + n64[None, :] * skr + h * skh + d1[:, None], mask=n_ok[None, :], other=0.0)
                s += tl.dot(q1, k1)
            allowed = (n[None, :] >= lo[:, None]) & (n[None, :] < hi[:, None])
            s = tl.where(allowed, s * 1.4426950408889634, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(s - m_use[:, None])
            alpha = tl.math.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            p = p.to(V.dtype.element_ty)
            v0 = tl.load(V + n64[:, None] * svr + h * svh + d0[None, :], mask=n_ok[:, None], other=0.0)
            acc0 = acc0 * alpha[:, None] + tl.dot(p, v0)
            if D1 > 0:
                v1 = tl.load(V + n64[:, None] * svr + h * svh + d1[None, :], mask=n_ok[:, None], other=0.0)
                acc1 = acc1 * alpha[:, None] + tl.dot(p, v1)
            m_i = m_new
        inv = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        out = O + m[:, None].to(tl.int64) * sor + h * soh
        tl.store(out + d0[None, :], (acc0 * inv[:, None]).to(O.dtype.element_ty), mask=row_ok[:, None])
        if D1 > 0:
            tl.store(out + d1[None, :], (acc1 * inv[:, None]).to(O.dtype.element_ty), mask=row_ok[:, None])


def tiled_window_attention(query, keys, values, first, last, query_streams: int, key_streams: int):
    """CUDA: ``query`` ``[rows, heads, dim]`` reads ``keys``/``values`` ``[key_rows, heads, dim]``; row ``r`` of token
    ``t = r // query_streams`` reads key rows ``[first[t] * key_streams, (last[t] + 1) * key_streams)``."""

    rows, heads, dim = query.shape
    D0, D1 = head_split(dim)
    query, keys, values = (t if t.stride(-1) == 1 else t.contiguous() for t in (query, keys, values))
    out = torch.empty(rows, heads, dim, device=query.device, dtype=query.dtype)
    if rows == 0:
        return out
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), heads)  # noqa: E731
    with torch.cuda.device(query.device):
        _tiled_window_attention[grid](query, keys, values, out, first, last, rows, int(query_streams), int(key_streams),
                                      query.stride(0), query.stride(1), keys.stride(0), keys.stride(1), values.stride(0), values.stride(1),
                                      out.stride(0), out.stride(1), D0=D0, D1=D1)
    return out


def tiled_reference(query, keys, values, first, last, query_streams: int, key_streams: int, block_m: int = 128, block_n: int = 64):
    """The kernel's algorithm in PyTorch (tile key unions, block-wise online softmax, empty-row guards): CPU and tests."""

    rows, heads, dim = query.shape
    first, last = first.long().cpu(), last.long().cpu()
    out = query.new_zeros(rows, heads, values.shape[-1])
    for begin in range(0, rows, block_m):
        m = torch.arange(begin, min(begin + block_m, rows))
        token = m // query_streams
        lo = first[token] * key_streams
        hi = torch.where(last[token] >= first[token], (last[token] + 1) * key_streams, lo)
        live = hi > lo
        if not live.any():
            continue
        kv_lo, kv_hi = int(lo[live].min()), int(hi.max())
        q = query[m].transpose(0, 1)  # [heads, block, dim]
        m_i = q.new_full((heads, len(m)), float("-inf"))
        l_i = q.new_zeros(heads, len(m))
        acc = q.new_zeros(heads, len(m), values.shape[-1])
        for start in range(kv_lo, kv_hi, block_n):
            n = torch.arange(start, min(start + block_n, kv_hi))
            s = (q @ keys[n].permute(1, 2, 0)) * LOG2E
            allowed = (n[None, :] >= lo[:, None]) & (n[None, :] < hi[:, None])
            s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m_i, s.amax(-1))
            m_use = torch.where(m_new == float("-inf"), torch.zeros_like(m_new), m_new)
            p = torch.exp2(s - m_use[..., None])
            alpha = torch.exp2(m_i - m_use)
            l_i = l_i * alpha + p.sum(-1)
            acc = acc * alpha[..., None] + p.to(values.dtype) @ values[n].transpose(0, 1)
            m_i = m_new
        inv = torch.where(l_i > 0, 1.0 / l_i, torch.zeros_like(l_i))
        out[m] = (acc * inv[..., None]).transpose(0, 1).to(out.dtype)
    return out


def _dense_ffn(module, z):
    """``RoutedFeedForward``'s always-on dense branch (``self.down(F.gelu(self.up(z)))`` in its
    forward)."""

    return module.down(F.gelu(module.up(z)))


KEY_BLOCK = 64  # CUTLASS memory-efficient forward, head dim <= 64: 64 keys per block
MAX_SEQUENCES = 65535  # grid z limit: one sequence per query token
_EAGER_ATTENTION = M.window_attention
_EAGER_PLAN_INIT = M.WindowPlan.__init__
_EAGER_MOE = M.RoutedFeedForward.forward
_EAGER_HEAD = M.UnifiedHead.forward
_EAGER_BLOCK = M.UnifiedBlock.forward
_EAGER_ENCODER = M.UnifiedEncoder.forward
_COMPILED: dict = {}
_LEVEL = 0
# Opt-in (levels >= 5): the tiled window attention below instead of FlashAttention-2
# with one sequence per token.
_TILED = settings.current().tiled_attention


@torch.library.custom_op("unified_fast::window_attention", mutates_args=())
def _window_op(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, query_starts: torch.Tensor,
    key_starts: torch.Tensor, key_lengths: torch.Tensor, bias: torch.Tensor,
) -> torch.Tensor:
    """``query`` ``[rows, heads, dim]``; ``keys``/``values`` ``[key_rows, heads, dim]``; one
    sequence per query token; ``bias`` ``[>= sequences, mask_width]`` holds each sequence's
    additive mask from its key start."""

    sequences, width = key_lengths.numel(), bias.shape[1]
    rows, heads, _ = query.shape
    if query.is_cuda:
        # Zero head/row strides: one mask vector per sequence (the kernel offsets the bias by
        # sequence, not key start).
        view = bias.as_strided((1, heads, rows, keys.shape[0]), (width, 0, 0, 1))
        out = torch.ops.aten._efficient_attention_forward.default(
            query[None], keys[None], values[None], view, query_starts, key_starts,
            rows // sequences, width, 0.0, 0, False,
            scale=1.0, seqlen_k=key_lengths, window_size=None)[0]
        return out[0]
    # CPU: the same intervals and masks, gathered (tests).
    columns = torch.arange(width)
    index = (key_starts[:-1, None].long() + columns).clamp(max=keys.shape[0] - 1)
    scores_bias = bias[:sequences].to(query.dtype).masked_fill(
        columns >= key_lengths[:, None], float("-inf"))
    q = query.view(sequences, rows // sequences, heads, -1).transpose(1, 2)
    k, v = keys[index].transpose(1, 2), values[index].transpose(1, 2)
    read = torch.softmax(q @ k.transpose(-1, -2) + scores_bias[:, None, None], dim=-1) @ v
    return read.transpose(1, 2).reshape(rows, heads, -1)


@_window_op.register_fake
def _(query, keys, values, query_starts, key_starts, key_lengths, bias):
    return query.new_empty(*query.shape[:2], values.shape[-1])


@torch.library.custom_op("unified_fast::window_flash", mutates_args=())
def _flash_op(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, query_starts: torch.Tensor,
    key_starts: torch.Tensor, key_lengths: torch.Tensor, width: int,
) -> torch.Tensor:
    """Level 5: FlashAttention-2 varlen, one sequence per query token reading ``key_lengths``
    key rows from ``key_starts`` (``seqused_k``; intervals overlap). No mask: each interval is
    exactly the token's window. Empty sequences are undefined here and zeroed by the caller."""

    sequences = key_lengths.numel()
    rows, heads, _ = query.shape
    if query.is_cuda:
        return torch.ops.aten._flash_attention_forward.default(
            query, keys, values, query_starts, key_starts, rows // sequences, width, 0.0, False,
            False, scale=1.0, seqused_k=key_lengths)[0]
    columns = torch.arange(width)
    index = (key_starts[:-1, None].long() + columns).clamp(max=keys.shape[0] - 1)
    bias = torch.zeros(sequences, width, dtype=query.dtype).masked_fill(
        columns >= key_lengths[:, None], float("-inf"))
    q = query.view(sequences, rows // sequences, heads, -1).transpose(1, 2)
    k, v = keys[index].transpose(1, 2), values[index].transpose(1, 2)
    read = torch.softmax(q @ k.transpose(-1, -2) + bias[:, None, None], dim=-1).nan_to_num() @ v
    return read.transpose(1, 2).reshape(rows, heads, -1)


@_flash_op.register_fake
def _(query, keys, values, query_starts, key_starts, key_lengths, width):
    return query.new_empty(*query.shape[:2], values.shape[-1])


@torch.library.custom_op("unified_fast::window_tiled", mutates_args=())
def _tiled_op(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, first: torch.Tensor,
    last: torch.Tensor, query_streams: int, key_streams: int,
) -> torch.Tensor:
    """``PMT_FAST_TILED_ATTENTION``: query row ``r`` (token ``r // query_streams``) reads key
    rows ``[first, last + 1)`` tokens times ``key_streams``; tiles of consecutive query rows
    walk the union of their windows (the tiled kernel below)."""

    if query.is_cuda:
        return tiled_window_attention(
            query, keys, values, first, last, query_streams, key_streams)
    return tiled_reference(
        query, keys, values, first, last, query_streams, key_streams)


@_tiled_op.register_fake
def _(query, keys, values, first, last, query_streams, key_streams):
    return query.new_empty(*query.shape[:2], values.shape[-1])


def set_tiled_attention(enabled: bool) -> None:
    """Switch the tiled window attention on or off for levels >= 5 (resets compiled code and
    graphs)."""

    global _TILED
    if bool(enabled) != _TILED:
        _TILED = bool(enabled)
        _reset_compiled()


def native_metadata(
    last: torch.Tensor, *, key_tokens: int, window: int, query_streams: int, key_streams: int,
    chunk: int, device,
):
    """Packed-sequence metadata for ``WindowPlan(last, ...)``, aligned to the eager plan's
    key-block grid."""

    tokens, W, Sk = int(last.numel()), int(window), int(key_streams)
    if not tokens or tokens > MAX_SEQUENCES:
        return None
    last = torch.as_tensor(last, dtype=torch.long).to(device)
    chunks = max(2, tokens // chunk + 1)
    # Eager chunk c reads key tokens from min(last over the chunk, padding rows = last[-1]) -
    # (W - 1); in the front-padded key rows (W tokens) that start is >= 0.
    padded = torch.cat([last, last[-1:].expand(chunks * chunk - tokens)])
    anchor = (
        (padded.view(chunks, chunk).min(dim=1).values + 1).repeat_interleave(chunk)[:tokens] * Sk)
    key_rows = (int(key_tokens) + W) * Sk
    # An empty row (last = -1) reads one zero-padded token; its output is zeroed by plan.keep.
    lower = ((last - W + 1).clamp(min=0) + W) * Sk
    lower = torch.where(last < 0, (W - 1) * Sk, lower)
    upper = (last + 1 + W) * Sk
    begin = anchor + torch.div(lower - anchor, KEY_BLOCK, rounding_mode="floor") * KEY_BLOCK
    used = upper - begin
    width = -(-(W * Sk + KEY_BLOCK - 1) // KEY_BLOCK) * KEY_BLOCK
    # The bias tensor also backs a [key_rows]-wide view, so it holds at least key_rows elements.
    columns = torch.arange(width, device=device)
    allowed = (columns >= (lower - begin)[:, None]) & (columns < used[:, None])
    bias = torch.zeros(max(tokens, -(-key_rows // width)), width, device=device)
    bias[:tokens].masked_fill_(~allowed, float("-inf"))
    int32 = dict(device=device, dtype=torch.int32)
    # Level 5 (flash): each query token's exact key rows in the unpadded keys, no mask (empty
    # rows: length 0).
    first = (last - W + 1).clamp(min=0)
    return SimpleNamespace(
        token_first=first.to(torch.int32), token_last=last.to(torch.int32),
        query_starts=torch.arange(tokens + 1, **int32) * int(query_streams),
        key_starts=torch.cat([begin, begin.new_full((1,), key_rows)]).to(torch.int32),
        key_lengths=used.to(torch.int32), bias=bias, pad=W,
        flash_starts=torch.cat(
            [first * Sk, first.new_full((1,), int(key_tokens) * Sk)]).to(torch.int32),
        flash_lengths=((last + 1 - first).clamp(min=0) * Sk).to(torch.int32), flash_width=W * Sk)


def window_attention(query, keys, values, plan):
    """Drop-in for ``attention.window_attention``: the native packed path when ``plan`` has
    metadata, else eager."""

    native = getattr(plan, "native", None)
    if native is None or torch.is_grad_enabled():
        return _EAGER_ATTENTION(query, keys, values, plan)
    read = _native_read(query, keys, values, plan)
    if read is None:
        return _EAGER_ATTENTION(query, keys, values, plan)
    # Self attention (strided plans: last[t] = t) has no empty rows.
    return read if plan.strided else torch.where(
        plan.keep[:, None], read, torch.zeros_like(read))


def _native_read(query, keys, values, plan):
    """The native kernel's ``[rows, heads * dim]`` read, before empty rows are zeroed
    (``None``: not applicable)."""

    native = plan.native
    if query.is_cuda:
        # The eager SDPA casts its inputs to the autocast dtype (the normalized query/keys are
        # FP32 there): same cast.
        dtype = (
            torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda")
            else query.dtype)
        if dtype != torch.bfloat16:
            return None
        query, keys, values = query.to(dtype), keys.to(dtype), values.to(dtype)
    T, Sq, H, D = query.shape
    if _LEVEL >= 5 and _TILED and (not query.is_cuda or _tiled_supported(D)):
        read = _tiled_op(
            query.reshape(T * Sq, H, D), keys.flatten(0, 1), values.flatten(0, 1),
            native.token_first, native.token_last, Sq, keys.shape[1]).flatten(1)
        return read if plan.strided else torch.where(
            plan.keep[:, None], read, torch.zeros_like(read))
    if _LEVEL >= 5:
        return _flash_op(
            query.reshape(T * Sq, H, D), keys.flatten(0, 1), values.flatten(0, 1),
            native.query_starts, native.flash_starts, native.flash_lengths,
            native.flash_width).flatten(1)
    front = (0, 0, 0, 0, 0, 0, native.pad, 0)
    return _window_op(
        query.reshape(T * Sq, H, D), F.pad(keys, front).flatten(0, 1),
        F.pad(values, front).flatten(0, 1), native.query_starts, native.key_starts,
        native.key_lengths, native.bias.to(query.dtype)).flatten(1)


def _tiled_supported(dim: int) -> bool:
    return triton is not None and head_split(dim) is not None


def _plan_init(
    self, last, *, key_tokens, window, query_streams, key_streams, chunk=M.ATTENTION_CHUNK,
    strided=False, device=None,
):
    _EAGER_PLAN_INIT(
        self, last, key_tokens=key_tokens, window=window, query_streams=query_streams,
        key_streams=key_streams, chunk=chunk, strided=strided, device=device)
    self.native = None
    if not torch.is_grad_enabled():
        self.native = native_metadata(
            torch.as_tensor(last).cpu(), key_tokens=key_tokens, window=window,
            query_streams=query_streams, key_streams=key_streams, chunk=self.chunk, device=device)


def _top2(scores):
    """``scores.topk(2).indices`` as two arg-maxes, which Inductor fuses with the softmax
    (``topk`` is a separate sort kernel). Same indices in the same order (ties aside, which
    ``topk`` leaves unspecified)."""

    first = scores.argmax(dim=-1, keepdim=True)
    others = scores.masked_fill(
        torch.arange(scores.shape[-1], device=scores.device) == first, float("-inf"))
    return torch.cat([first, others.argmax(dim=-1, keepdim=True)], dim=-1)


def _routes(ffn, content, shift, tokens, fused=False):
    """``z`` and the top-2 routes of ``RoutedFeedForward.forward`` (same math), without its
    training statistics."""

    z = content + repeat_layers(shift.to(content.dtype), tokens)
    shift_logits = ffn.state_router(unit_rms(shift).to(ffn.state_router.weight.dtype)).float()
    logits = ffn.router(content).float() + repeat_layers(shift_logits, tokens)
    probabilities = logits.softmax(dim=-1)
    scores = logits + ffn.selection_bias.float()
    chosen = _top2(scores) if fused else scores.topk(2, dim=-1).indices
    picked = probabilities.gather(-1, chosen)
    return z, chosen, picked / picked.sum(dim=-1, keepdim=True)


def _expert_weights(ffn, dtype):
    """``ffn``'s expert weights in ``dtype``, contiguous, cast once (static at inference).
    Keyed by the parameters' identity and version, so a token-parallel replica (a deep copy)
    never reuses the original's device copies."""

    versions = (
        id(ffn.expert_up), ffn.expert_up._version, id(ffn.expert_down), ffn.expert_down._version,
        dtype)
    if ffn.__dict__.get("_fast_weights_version") != versions:
        ffn.__dict__["_fast_weights"] = (
            ffn.expert_up.detach().to(dtype).contiguous(),
            ffn.expert_down.detach().to(dtype).contiguous())
        ffn.__dict__["_fast_weights_version"] = versions
    return ffn.__dict__["_fast_weights"]


@torch.library.custom_op("unified_fast::top2_experts", mutates_args=())
def _experts_op(
    z: torch.Tensor, up: torch.Tensor, down: torch.Tensor, chosen: torch.Tensor,
    weights: torch.Tensor, dense: torch.Tensor, tuned: bool = False,
) -> torch.Tensor:
    """``dense + sum_k w_k * gelu(z @ up[e_k]) @ down[e_k]``: the grouped Triton experts on
    CUDA (BF16; ``tuned``: the level-6 launch, same values), the eager ``RoutedFeedForward``
    dispatch on CPU (tests)."""

    if z.is_cuda:
        from speculative_encoding.translator.kernels import top2_experts

        return top2_experts(z, up, down, chosen, weights, dense.contiguous(), tuned)
    flat = chosen.reshape(-1)
    order = torch.argsort(flat, stable=True)
    rows = order // chosen.shape[-1]
    parts = z.index_select(0, rows).split(torch.bincount(flat, minlength=up.shape[0]).tolist())
    outputs = torch.cat(
        [F.gelu(part @ u) @ d for part, u, d in zip(parts, up.unbind(0), down.unbind(0))])
    scaled = outputs * weights.reshape(-1).index_select(0, order)[:, None].to(outputs.dtype)
    return dense + torch.zeros_like(dense).index_add(0, rows, scaled)


@_experts_op.register_fake
def _(z, up, down, chosen, weights, dense, tuned=False):
    return dense.new_empty(dense.shape)


def moe_forward(self, content, shift, tokens):
    """No-grad ``RoutedFeedForward.forward`` without the host sync: same routing math, then the
    grouped Triton experts (device-side counted dispatch, GELU by exact BF16 lookup,
    eager-order BF16 combine). The expert GEMMs accumulate in a different order than cuBLAS, so
    outputs are near-exact, not bitwise. The routing balance (``last_balance``) is a training
    diagnostic and is not computed."""

    if (
        torch.is_grad_enabled() or not content.is_cuda or self.top_k != 2
        or content.dtype != torch.bfloat16
    ):
        return _EAGER_MOE(self, content, shift, tokens)
    from speculative_encoding.translator.kernels import top2_experts

    z, chosen, weights = _routes(self, content, shift, tokens)
    self.last_balance = None
    up, down = _expert_weights(self, z.dtype)
    return top2_experts(z, up, down, chosen, weights, _dense_ffn(self, z), _LEVEL >= 6)


def _rope(x, cos, sin):
    """``attention.rope`` (the same products and sums per element) without its unbind + stack,
    whose Inductor lowering loads every element four times under masks: each element reads
    itself and its pair partner (a flip within the pair), and the parity picks the even
    (``e * cos - o * sin``) or odd (``e * sin + o * cos``) formula."""

    cos, sin = cos.to(x.dtype)[:, None, None, :, None], sin.to(x.dtype)[:, None, None, :, None]
    pairs = x.unflatten(-1, (-1, 2))
    partner = pairs.flip(-1)
    first = torch.arange(2, device=x.device) == 0
    return torch.where(first, pairs * cos - partner * sin, partner * sin + pairs * cos).flatten(-2)


def _self_attention(attention, x, *, tokens, plan, rope_table):
    """``SelfAttention.forward`` with ``_rope``."""

    query, key, value = attention.qkv(x).view(
        tokens, -1, 3, attention.heads, x.shape[-1] // attention.heads).unbind(2)
    query = (
        _rope(F.normalize(query, dim=-1), *rope_table)
        * attention.logit_scale.exp().to(x.dtype)[:, None])
    key = _rope(F.normalize(key, dim=-1), *rope_table)
    return attention.out(M.window_attention(query, key, value, plan))


def _glue_self_half(block, x, self_shift, layout):
    """``model._self_half`` with ``_self_attention``."""

    tokens = layout.tokens
    x = x + _self_attention(
        block.self_attn, block.norm_self(x) + repeat_layers(self_shift, tokens).to(x.dtype),
        tokens=tokens, plan=layout.self_plan, rope_table=layout.rope)
    return x, block.norm_ffn(x)


def _cross_attention(attention, x, memory, *, tokens, plan, rope_table, memory_rope_table):
    """``CrossAttention.forward`` (same math) with the empty rows zeroed after the output map
    instead of before it (``fused_glue``): ``out`` maps rows independently and a zero row to
    zero, so the values are the same, and the zeroing fuses into the residual add instead of
    costing its own pass over the read."""

    head = x.shape[-1] // attention.heads
    query = _rope(
        F.normalize(attention.query(x).view(tokens, -1, attention.heads, head), dim=-1),
        *rope_table)
    query = query * attention.logit_scale.exp().to(x.dtype)[:, None]
    key, value = attention.memory_kv(attention.memory_norm(memory)).view(
        *memory.shape[:2], 2, attention.heads, head).unbind(2)
    key = _rope(F.normalize(key, dim=-1), *memory_rope_table)
    read = None if plan.native is None else _native_read(query, key, value, plan)
    if read is None:
        return attention.out(_EAGER_ATTENTION(query, key, value, plan))
    out = attention.out(read)
    return out if plan.strided else torch.where(plan.keep[:, None], out, torch.zeros_like(out))


def _glue_attention_half(block, x, self_shift, cross_shift, source, layout):
    """``model._attention_half`` with ``_self_attention`` and ``_cross_attention``."""

    tokens = layout.tokens
    x = x + _self_attention(
        block.self_attn, block.norm_self(x) + repeat_layers(self_shift, tokens).to(x.dtype),
        tokens=tokens, plan=layout.self_plan, rope_table=layout.rope)
    query = block.norm_cross(x) + repeat_layers(cross_shift, tokens).to(x.dtype)
    x = x + _cross_attention(
        block.cross_attn, query, source, tokens=tokens, plan=layout.cross_plan,
        rope_table=layout.rope, memory_rope_table=layout.memory_rope)
    return x, block.norm_ffn(x)


def _block_body(block, x, state, layout, memory, memory_state, up, down):
    """One whole ``UnifiedBlock.forward`` (no grad, top-2): compiled at level 4 with the window
    attention and the grouped experts as opaque custom ops, so every norm, state map, residual
    and routing step around them fuses."""

    if memory is None:
        x, content = (_glue_self_half if _GLUE else M._self_half)(
            block, x, block.self_state_map(state), layout)
    else:
        source = memory + block.memory_state_map(memory_state)[None]
        self_shift = None if block.self_state_map is None else block.self_state_map(state)
        half = _glue_attention_half if _GLUE else M._attention_half
        x, content = half(block, x, self_shift, block.cross_state_map(state), source, layout)
    z, chosen, weights = _routes(
        block.ffn, content, block.ffn_state_map(state), layout.tokens, fused=True)
    return x + _experts_op(z, up, down, chosen, weights, _dense_ffn(block.ffn, z), _LEVEL >= 6)


def block_forward(self, x, *, state, layout, memory=None, memory_state=None):
    """Level 4 ``UnifiedBlock.forward``: ``_block_body``, compiled when the block is
    (``block.compiled``). Gradients, top-k other than 2, or non-BF16 CUDA rows (the FP32
    reference) run the eager block."""

    if (
        torch.is_grad_enabled() or self.ffn.top_k != 2
        or (x.is_cuda and x.dtype != torch.bfloat16)
    ):
        return _EAGER_BLOCK(
            self, x, state=state, layout=layout, memory=memory, memory_state=memory_state)
    if memory is not None and (
        not self.memory or layout.cross_plan is None or memory_state is None
    ):
        raise ValueError("decoder memory requires positions, limits and source state")
    self.ffn.last_balance = None
    up, down = _expert_weights(self.ffn, x.dtype)
    body = M._compiled(_block_body) if self.compiled else _block_body
    return (
        body(self, x, state, layout, memory, memory_state, up, down),
        x.new_zeros((), dtype=torch.float32))


def _encoder_input(rows, gains, maps):
    """``UnifiedEncoder.forward``'s input norm and per-stream projection (same explicit casts)."""

    scale = torch.rsqrt(rows.float().square().mean(-1, keepdim=True) + 1e-6).to(rows.dtype)
    return torch.bmm(
        (rows * scale * gains.to(rows.dtype).unsqueeze(0)).transpose(0, 1), maps.to(rows.dtype)
    ).transpose(0, 1)


def encoder_forward(self, model, rows, state=None):
    """Level 4 ``UnifiedEncoder.forward``: the source-row norm
    (``[tokens, streams, source width]`` in FP32, the largest element-wise pass of the encoder)
    runs compiled; the blocks are ``block_forward``."""

    if torch.is_grad_enabled() or not rows.is_cuda:
        return _EAGER_ENCODER(self, model, rows, state)
    geometry, name = self.sources[model], M._key(model)
    state = self.states[name] if state is None else state
    if tuple(rows.shape[1:]) != (geometry.streams, geometry.width):
        raise ValueError(
            f"{model} rows have shape {tuple(rows.shape)}; "
            f"expected [tokens, {geometry.streams}, {geometry.width}]")
    projected = M._compiled(_encoder_input)(rows, self.gains[name], self.maps[name])
    tokens, streams, width = projected.shape
    x = projected.reshape(tokens * streams, width)
    layout = M.Layout(
        tokens, streams, torch.arange(tokens, device=rows.device), self.config,
        chunk=self.attention_chunk)
    for block in self.blocks:
        x, _ = block(x, state=state, layout=layout)
    return x.view(tokens, streams, width), x.new_zeros((), dtype=torch.float32)


def _compiled(function):
    if function not in _COMPILED:
        _COMPILED[function] = torch.compile(function, dynamic=True, backend=M.COMPILE_BACKEND)
    return _COMPILED[function]


def head_forward(self, states):
    """``UnifiedHead.forward`` under torch.compile (no-grad CUDA): the FP32 norm is fused
    instead of materialized. Same explicit casts; the mean reduction order may differ
    (near-exact)."""

    if torch.is_grad_enabled() or not states.is_cuda:
        return _EAGER_HEAD(self, states)
    return _compiled(_EAGER_HEAD)(self, states)


def _reset_compiled():
    import torch._dynamo

    M._COMPILED.clear()
    _COMPILED.clear()
    torch._dynamo.reset()


# Fused glue (``PMT_FAST_FUSED_GLUE=1``, levels >= 4; off by default): fewer passes over the
# ``[rows, width]`` activations around the opaque kernels, same values (see each function).
# ``tune`` (``PMT_FAST_GLUE_TUNE=1``) also lets Inductor tune its own kernels' launch
# configurations (coordinate descent; reduction block sizes may change, so FP32 sums may round
# differently before the BF16 casts).

_GLUE = settings.current().fused_glue
_GLUE_TUNE = settings.current().glue_tune
_INDUCTOR_DEFAULTS: dict = {}


def _apply_tune(tune: bool) -> None:
    import torch._inductor.config as inductor

    settings = {"coordinate_descent_tuning": True}
    if not _INDUCTOR_DEFAULTS:
        _INDUCTOR_DEFAULTS.update({name: getattr(inductor, name) for name in settings})
    for name, value in settings.items():
        setattr(inductor, name, value if tune else _INDUCTOR_DEFAULTS[name])


def set_fused_glue(enabled: bool, tune: bool = False) -> None:
    """Turn the fused glue (and its Inductor tuning) on or off for this process; compiled
    regions retrace."""

    global _GLUE, _GLUE_TUNE
    if (bool(enabled), bool(tune)) == (_GLUE, _GLUE_TUNE):
        return
    _GLUE, _GLUE_TUNE = bool(enabled), bool(tune)
    _apply_tune(_GLUE_TUNE)
    _reset_compiled()


if _GLUE_TUNE:
    _apply_tune(True)


# Levels are cumulative: 1 native-window attention (bitwise), 2 + grouped experts without host
# sync, 3 + compiled output head (and, in the runtime, the compiled frozen K/V projection), 4 +
# whole blocks and the encoder input norm compiled around the fast kernels (custom ops), 5 +
# FlashAttention-2 varlen windows (exact key intervals, no mask) instead of the masked CUTLASS
# kernel, 6 + the tuned grouped-expert launch (``kernels``: column tiles fastest, unmasked
# weight loads, separate up/down tiles; bitwise identical to level 5). Opt-in on levels >= 5:
# ``PMT_FAST_TILED_ATTENTION=1`` (or ``set_tiled_attention``) swaps the FlashAttention-2
# windows for the tiled kernel (same keys; FP32-accumulation rounding
# differences only).
LEVELS = (0, 1, 2, 3, 4, 5, 6)
# The private ops levels >= 1 call, as Torch 2.7.0 declares them (unchanged in 2.11.0's
# native_functions.yaml).
_PINNED_SCHEMAS = {
    "_efficient_attention_forward": (
        "aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, "
        "Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt? max_seqlen_q, "
        "SymInt? max_seqlen_k, float dropout_p, int custom_mask_type, "
        "bool compute_log_sumexp=False, *, float? scale=None, Tensor? seqlen_k=None, "
        "int? window_size=None) -> (Tensor output, Tensor logsumexp, Tensor philox_seed, "
        "Tensor philox_offset, SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)"),
    "_flash_attention_forward": (
        "aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, "
        "Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, "
        "bool is_causal, bool return_debug_mask, *, float? scale=None, "
        "SymInt? window_size_left=None, SymInt? window_size_right=None, Tensor? seqused_k=None, "
        "Tensor? alibi_slopes=None) -> (Tensor output, Tensor softmax_logsumexp, "
        "Tensor rng_state, Tensor unused, Tensor debug_attn_mask)"),
}


def _compatible(name: str) -> bool:
    """Whether Torch's current schema of ``name`` still accepts the pinned call: the pinned
    arguments and returns unchanged, any new arguments optional (Torch 2.14 appends
    ``block_table`` / ``num_splits`` to ``_flash_attention_forward``)."""

    # torch._C.parse_schema is private, but it is how Torch itself turns schema strings into
    # FunctionSchema objects, the type the live op exposes as ``_schema``.
    pinned = torch._C.parse_schema(_PINNED_SCHEMAS[name])
    current = getattr(torch.ops.aten, name).default._schema
    old, new = pinned.arguments, current.arguments
    same = [str(a.type) == str(b.type) and a.name == b.name for a, b in zip(old, new)]
    extra_optional = all(arg.has_default_value() for arg in new[len(old):])
    returns = [str(r.type) for r in pinned.returns] == [str(r.type) for r in current.returns]
    return len(new) >= len(old) and all(same) and extra_optional and returns


def install(enabled: int = 1) -> None:
    """Set the no-grad executor level for every translator forward in this process (0: eager)."""

    global _LEVEL
    target = int(enabled)
    if target not in LEVELS:
        raise ValueError(f"fast inference level must be one of {LEVELS}")
    if target == _LEVEL:
        return
    if target and torch.cuda.is_available() and (
        torch.__version__.split("+")[0] != "2.7.0"
    ):
        # The paper ran Torch 2.7.0; another Torch works if the private attention ops it
        # calls still accept 2.7's calls (2.11 keeps the schemas; 2.14 adds optional ones).
        changed = [name for name in _PINNED_SCHEMAS if not _compatible(name)]
        if changed:
            raise RuntimeError(
                f"Torch {torch.__version__}: private attention op schemas differ from "
                f"Torch 2.7.0: {changed}")
    if target >= 2 and torch.cuda.is_available():
        # Initialize Triton's CUDA driver here: its lazy first-use init is not thread-safe
        # (token-parallel threads).
        import triton

        triton.runtime.driver.active.get_current_device()
    M.window_attention = window_attention if target >= 1 else _EAGER_ATTENTION
    M.WindowPlan.__init__ = _plan_init if target >= 1 else _EAGER_PLAN_INIT
    M.RoutedFeedForward.forward = moe_forward if target >= 2 else _EAGER_MOE
    M.UnifiedHead.forward = head_forward if target >= 3 else _EAGER_HEAD
    M.UnifiedBlock.forward = block_forward if target >= 4 else _EAGER_BLOCK
    M.UnifiedEncoder.forward = encoder_forward if target >= 4 else _EAGER_ENCODER
    _LEVEL = target
    _reset_compiled()
