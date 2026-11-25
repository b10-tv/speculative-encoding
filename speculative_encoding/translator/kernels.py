"""Triton kernels for no-grad top-2 expert dispatch, based on Jeff Zheng's executors. The row-count arguments
(``N``, ``BLOCKS``) are runtime values, not ``tl.constexpr``: every document length would otherwise compile new
kernels.

All dispatch sizes stay on the device (grids are bounded by the row count), so no expert count reaches the host.
"""

import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_GELU_TABLES = {}


def gelu_table(device):
    """``F.gelu`` of every BF16 bit pattern: a lookup reproduces eager BF16 GELU exactly."""

    key = str(device)
    if key not in _GELU_TABLES:
        bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
        _GELU_TABLES[key] = F.gelu(bits.view(torch.bfloat16))
    return _GELU_TABLES[key]


# Dispatch metadata.
@triton.jit
def _histogram(Keys,Counts,N,BLOCKS,EXPERTS:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0);i=block*BLOCK+tl.arange(0,BLOCK)
    keys=tl.load(Keys+i,mask=i<N,other=0)
    expert=tl.arange(0,EXPERTS)
    hits=(keys[None,:]==expert[:,None])&(i[None,:]<N)
    counts=tl.sum(hits.to(tl.int32),1)
    tl.store(Counts+expert*BLOCKS+block,counts)


@triton.jit
def _prefix(Inclusive,Cu,Counts,TileEnds,BLOCKS,EXPERTS:tl.constexpr,BM:tl.constexpr):
    e=tl.arange(0,EXPERTS)
    count=tl.load(Inclusive+e*BLOCKS+BLOCKS-1)
    ends=tl.cumsum(count,0)
    tiles=tl.cumsum((count+BM-1)//BM,0)
    tl.store(Cu+e+1,ends);tl.store(Cu,0)
    tl.store(Counts+e,count);tl.store(TileEnds+e,tiles)


@triton.jit
def _scatter(Keys,Inclusive,Cu,Order,N,BLOCKS,EXPERTS:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0);i=block*BLOCK+tl.arange(0,BLOCK)
    keys=tl.load(Keys+i,mask=i<N,other=0)
    e=tl.arange(0,EXPERTS)
    hits=(keys[None,:]==e[:,None])&(i[None,:]<N)
    rank=tl.cumsum(hits.to(tl.int32),1)-1
    previous=tl.load(Inclusive+e*BLOCKS+block-1,mask=block>0,other=0)
    start=tl.load(Cu+e)+previous
    destination=start[:,None]+rank
    tl.store(Order+destination,i[None,:],mask=hits)


def dispatch_metadata(keys, experts, bm):
    """Stable expert grouping of the flat route ids (same order as ``argsort(stable=True)``), counts, starts and
    per-expert tile ends, without a host sync."""

    block = 256
    blocks = triton.cdiv(keys.numel(), block)
    by_block = torch.empty(experts, blocks, device=keys.device, dtype=torch.int32)
    _histogram[(blocks,)](keys, by_block, keys.numel(), blocks, experts, block, num_warps=4)
    inclusive = by_block.cumsum(1, dtype=torch.int32)
    cu = torch.empty(experts + 1, device=keys.device, dtype=torch.int32)
    counts = torch.empty(experts, device=keys.device, dtype=torch.int32)
    tile_ends = torch.empty_like(counts)
    _prefix[(1,)](inclusive, cu, counts, tile_ends, blocks, experts, bm, num_warps=4)
    order = torch.empty(keys.numel(), device=keys.device, dtype=torch.int64)
    _scatter[(blocks,)](keys, inclusive, cu, order, keys.numel(), blocks, experts, block, num_warps=4)
    return counts, cu[:-1], order, tile_ends


# Grouped experts and combine.
@triton.jit
def _gelu(x):
    return (0.5 * x) * (1.0 + libdevice.erf(x * 0.7071067811865476))


@triton.jit
def _grouped(A, W, Order, Starts, Counts, TileEnds, Out, GeluTable,
             ar: tl.constexpr, ak: tl.constexpr,
             we: tl.constexpr, wk: tl.constexpr, wn: tl.constexpr,
             INNER: tl.constexpr, OUTER: tl.constexpr, EXPERTS: tl.constexpr,
             TOP: tl.constexpr, UP: tl.constexpr, GELU_MODE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SWIZZLE: tl.constexpr = False, EVEN: tl.constexpr = False):
    if SWIZZLE:  # 1-D grid, column tiles fastest: the CTAs of one row tile run together and share its A rows in L2
        tile, coltile = tl.program_id(0) // ((OUTER + BN - 1) // BN), tl.program_id(0) % ((OUTER + BN - 1) // BN)
    else:
        tile, coltile = tl.program_id(0), tl.program_id(1)
    ends = tl.load(TileEnds + tl.arange(0, triton.next_power_of_2(EXPERTS)),
                   mask=tl.arange(0, triton.next_power_of_2(EXPERTS)) < EXPERTS, other=2147483647)
    expert = tl.sum((tile >= ends).to(tl.int32))
    if expert < EXPERTS:
        tile_start = tl.load(TileEnds + expert - 1, mask=expert > 0, other=0)
        begin = tl.load(Starts + expert)
        count = tl.load(Counts + expert)
        local = (tile - tile_start) * BM + tl.arange(0, BM)
        position = begin.to(tl.int64) + local
        order = tl.load(Order + position, mask=local < count, other=0)
        row = order // TOP if UP else position
        cols = coltile * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        accum = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(INNER, BK)):
            inner = block * BK + ks
            if EVEN:  # INNER % BK == 0 and OUTER % BN == 0: only the row mask (same values, unmasked weight loads)
                a = tl.load(A + row[:, None] * ar + inner[None, :] * ak, mask=local[:, None] < count, other=0)
                w = tl.load(W + expert * we + inner[:, None] * wk + cols[None, :] * wn)
            else:
                a = tl.load(A + row[:, None] * ar + inner[None, :] * ak,
                            mask=(local[:, None] < count) & (inner[None, :] < INNER), other=0)
                w = tl.load(W + expert * we + inner[:, None] * wk + cols[None, :] * wn,
                            mask=(inner[:, None] < INNER) & (cols[None, :] < OUTER), other=0)
            accum = tl.dot(a, w, accum)
        destination = position if UP else order
        if GELU_MODE == 2:
            bits=accum.to(tl.bfloat16).to(tl.int16,bitcast=True).to(tl.int32)&65535
            accum=tl.load(GeluTable+bits)
        elif GELU_MODE == 1:
            accum = _gelu(accum.to(tl.bfloat16).to(tl.float32))
        tl.store(Out + destination[:, None] * OUTER + cols[None, :], accum,
                 mask=(local[:, None] < count) & (cols[None, :] < OUTER))


@triton.jit
def _combine(Output, Weights, Chosen, Update, Result, N, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    row,col=i//WIDTH,i%WIDTH
    first=tl.load(Output+row*2*WIDTH+col,mask=i<N,other=0).to(tl.float32)
    second=tl.load(Output+(row*2+1)*WIDTH+col,mask=i<N,other=0).to(tl.float32)
    w0=tl.load(Weights+row*2,mask=i<N,other=0)
    w1=tl.load(Weights+row*2+1,mask=i<N,other=0)
    a=(first*w0.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    b=(second*w1.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    a=tl.where(w0>0,a,0.0);b=tl.where(w1>0,b,0.0)
    e0=tl.load(Chosen+row*2,mask=i<N,other=0)
    e1=tl.load(Chosen+row*2+1,mask=i<N,other=0)
    # The reference accumulates in expert-index order, starting from BF16 zero.
    low=tl.where(e0<e1,a,b);high=tl.where(e0<e1,b,a)
    low=(0.0+low).to(tl.bfloat16).to(tl.float32)
    routed=(low+high).to(tl.bfloat16).to(tl.float32)
    dense=tl.load(Update+i,mask=i<N,other=0).to(tl.float32)
    tl.store(Result+i,dense+routed,mask=i<N)


# BM, BN, BK, warps, stages. The original default was (128, 128, 64, 8, 3); BN 256 is 12% faster on H200 for an 8k-token
# document (14.1 -> 12.5 ms per call) and bitwise identical (same BK).
TILES = (128, 256, 64, 8, 3)


def top2_experts(z, up, down, chosen, weights, dense, tuned=False):
    """``dense + sum_k w_k * gelu(z @ up[e_k]) @ down[e_k]`` for top-2 routes, in BF16 with eager rounding points.

    ``z`` ``[rows, width]`` BF16; ``up`` ``[experts, width, ffn]`` and ``down`` ``[experts, ffn, width]`` BF16;
    ``chosen`` ``[rows, 2]``; ``weights`` ``[rows, 2]`` FP32; ``dense`` ``[rows, width]`` BF16."""

    with torch.cuda.device(z.device):  # Triton launches on the current device; translation threads run on other GPUs
        return _top2_experts(z, up, down, chosen, weights, dense, tuned)


# Level 6 (``tuned=True``): the same kernel with the column tiles fastest in a 1-D grid (A-row reuse in L2), unmasked
# weight loads, and separate tiles for the up (K 512, N 2048) and down (K 2048, N 512) projections. Per-element
# accumulation order is unchanged (sequential K blocks into one FP32 accumulator), so outputs are bitwise identical.
# H200, 8,192-token document: 12.4 -> 11.3 ms per call
# (up + GELU 7.6 -> 6.7 ms, down 4.0 -> 3.3-3.5 ms). torch._grouped_mm (Torch 2.9.1) was bitwise too but
# slower: 13.4 ms vs 11.7 (shipped) / 10.8 ms (tuned) on that build.
TUNED_UP = (128, 128, 64, 4, 3)
TUNED_DOWN = (128, 256, 32, 8, 5)
# GELU mode 1 (erf in FP32 in the kernel) is 0.6 ms faster per call but differs from eager F.gelu on 11
# of the 65,536 BF16 inputs; an erf + bitset-fix variant was exact but 4x slower (dropped).
TUNED_SWIZZLE, TUNED_EVEN, TUNED_GELU = True, True, 2


def _top2_experts(z, up, down, chosen, weights, dense, tuned=False):
    up_tiles, down_tiles = (TUNED_UP, TUNED_DOWN) if tuned else (TILES, TILES)
    experts, width, ffn = up.shape
    flat = chosen.reshape(-1).contiguous()
    counts, starts, order, tile_ends = dispatch_metadata(flat, experts, up_tiles[0])
    table = gelu_table(z.device)
    hidden = torch.empty(flat.numel(), ffn, device=z.device, dtype=z.dtype)
    _grouped_launch(z, up, order, starts, counts, tile_ends, hidden, table, width, ffn, experts, True, TUNED_GELU if tuned else 2, up_tiles, tuned,
                    enable_fp_fusion=False)
    if down_tiles[0] != up_tiles[0]:
        tile_ends = ((counts + down_tiles[0] - 1) // down_tiles[0]).cumsum(0, dtype=torch.int32)
    output = torch.empty(flat.numel(), width, device=z.device, dtype=z.dtype)
    _grouped_launch(hidden, down, order, starts, counts, tile_ends, output, table, ffn, width, experts, False, 0, down_tiles, tuned)
    result = torch.empty_like(dense)
    _combine[(triton.cdiv(dense.numel(), 256),)](output, weights.contiguous(), chosen.contiguous(), dense, result, dense.numel(), width, 256,
                                                 enable_fp_fusion=False)
    return result


def _grouped_launch(a, w, order, starts, counts, tile_ends, out, table, inner, outer, experts, up, gelu_mode, tiles, tuned, **options):
    bm, bn, bk, warps, stages = tiles
    max_tiles = triton.cdiv(order.numel(), bm) + experts
    swizzle = tuned and TUNED_SWIZZLE
    extra = dict(SWIZZLE=swizzle, EVEN=TUNED_EVEN and inner % bk == 0 and outer % bn == 0) if tuned else {}
    grid = (max_tiles * triton.cdiv(outer, bn),) if swizzle else (max_tiles, triton.cdiv(outer, bn))
    _grouped[grid](a, w, order, starts, counts, tile_ends, out, table, *a.stride(), *w.stride(), inner, outer, experts, 2, up, gelu_mode,
                   bm, bn, bk, num_warps=warps, num_stages=stages, **extra, **options)
