"""Query-conditioned slot compressor appended after a frozen unified translator.

The translator's decoder gives one row per target document token and target
stream, ``translated [T, S, d]``, before the target's output interface. The
compressor reads those rows and the target's question, and keeps ``M`` slot
rows:

* slot ``i`` starts from the translated row of document token ``tau_i``
  (evenly spaced, ``floor((i + 1/2) T / M)``) plus a learned slot vector;
* question rows are the target's own hidden states for its question alone,
  mapped through the target model's *input* interface (its encoder-side gain
  and projection per stream), plus a learned question vector;
* ``R`` blocks copied from the last decoder blocks, with global self-attention
  over every slot and question row in every stream, a memory read of all ``T``
  translated rows of the same stream, and the routed FFN. Every block reads the
  target model-layer state through its own maps, as in the decoder; the memory
  rows are target rows, so their offset also reads the target state.

The blocks start as the identity (zero output maps, zero slot and question
vectors, zero memory-state map), so at step 0 the slots are exactly the
translated rows at ``tau``. Only slot rows leave, through the target's output
interface, and the target's own cache writer turns them into the cartridge.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from speculative_encoding.translator.model import rope, rope_tables
from speculative_encoding.translator.model import repeat_layers
from speculative_encoding.translator.model import _interp, _key

# Streams per memory-read chunk in ``_block`` (memory only; the result does not
# depend on it).
MEMORY_STREAM_CHUNK = 8


def slot_tokens(tokens: int, *, ratio: float | None = None,
        budget: int | None = None, device=None) -> torch.Tensor:
    """Document indices ``tau`` of the slots: ``M = ceil(T / ratio)`` (or
    ``budget``) evenly spaced tokens."""

    assert (ratio is None) != (budget is None), "give one of ratio and budget"
    if budget is not None:
        count = int(budget)
    else:
        count = max(1, math.ceil(int(tokens) / float(ratio)))
    count = min(count, int(tokens))
    index = torch.arange(count, device=device, dtype=torch.float64)
    return ((index + 0.5) * int(tokens) / count).floor().long()


def question_rows(encoder, model: str, hidden: torch.Tensor,
        streams: int | None = None) -> torch.Tensor:
    """The target's question hidden states ``[J, L, D]`` (layer inputs) through
    its input interface -> ``[J, L, d]``.

    Stream ``l`` (the input to layer ``l``) uses the interface the encoder
    applies to that model's stream ``l`` when the model is a source. A decoder
    with ``streams < L`` depth streams gets ``[J, streams, d]``: stream ``s`` is
    the average of the layers it feeds under the head's initial depth
    interpolation (``model._interp``).
    """

    name = _key(model)
    gains, maps = encoder.gains[name], encoder.maps[name]
    layers = hidden.shape[1]
    assert layers <= gains.shape[0] and hidden.shape[2] == gains.shape[1], (
        f"{model}: question hidden {tuple(hidden.shape)} does not fit its "
        f"input interface {tuple(gains.shape)}")
    mean_square = hidden.float().square().mean(-1, keepdim=True)
    scale = torch.rsqrt(mean_square + 1e-6).to(hidden.dtype)
    normed = hidden * scale * gains[:layers].to(hidden.dtype).unsqueeze(0)
    rows = torch.bmm(normed.transpose(0, 1),
        maps[:layers].to(hidden.dtype)).transpose(0, 1)
    if streams is None or int(streams) == layers:
        return rows
    return pool_question_rows(rows, int(streams))


def pool_question_rows(rows: torch.Tensor, streams: int) -> torch.Tensor:
    """Per-layer rows ``[J, L, d]`` -> ``[J, S, d]`` with the fixed mix: stream ``s`` is
    the average of the layers it feeds under the head's initial depth interpolation. What
    compressors without a learned ``question_mix`` were trained with."""

    layers = rows.shape[1]
    assert 1 <= int(streams) <= layers
    weights = _interp(layers, int(streams)).to(device=rows.device, dtype=torch.float32)
    weights = weights / weights.sum(0, keepdim=True)
    return torch.einsum("jld,ls->jsd", rows.float(), weights).to(rows.dtype)


def _block(block, x, translated, state, tokens, rope_rows, memory_rope):
    """One decoder block with compressor masks: global self-attention,
    same-stream memory read, routed FFN."""

    streams, width = translated.shape[1], translated.shape[2]
    heads = block.self_attn.heads
    head = width // heads
    rows = tokens * streams
    # Global self-attention over every slot and question row of every stream.
    attention = block.self_attn
    shift = repeat_layers(block.self_state_map(state), tokens).to(x.dtype)
    qkv = attention.qkv(block.norm_self(x) + shift)
    query, key, value = qkv.view(tokens, streams, 3, heads, head).unbind(2)
    scale = attention.logit_scale.exp().to(x.dtype)[:, None]
    query = rope(F.normalize(query, dim=-1), *rope_rows) * scale
    key = rope(F.normalize(key, dim=-1), *rope_rows)

    def flat(t):
        return t.reshape(1, rows, heads, head).transpose(1, 2)

    read = F.scaled_dot_product_attention(flat(query), flat(key), flat(value),
        scale=1.0)
    x = x + attention.out(read.transpose(1, 2).reshape(rows, width))
    # Memory read: each stream reads all translated rows of the same stream.
    cross = block.cross_attn
    shift = repeat_layers(block.cross_state_map(state), tokens).to(x.dtype)
    query = cross.query(block.norm_cross(x) + shift)
    query = query.view(tokens, streams, heads, head)
    query = rope(F.normalize(query, dim=-1), *rope_rows)
    query = query * cross.logit_scale.exp().to(x.dtype)[:, None]
    # Streams read only their own memory rows, so the read runs a few streams
    # at a time (same math): keys and values for every translated row at once
    # do not fit beside a 128k document (22 GB for Gemma at 172k tokens).
    memory_state = block.memory_state_map(state).to(translated.dtype)

    def per_stream(t):
        return t.permute(1, 2, 0, 3)

    reads = []
    for first in range(0, streams, MEMORY_STREAM_CHUNK):
        part = slice(first, min(streams, first + MEMORY_STREAM_CHUNK))
        memory = translated[:, part] + memory_state[part][None]
        kv = cross.memory_kv(cross.memory_norm(memory))
        key, value = kv.view(*memory.shape[:2], 2, heads, head).unbind(2)
        key = rope(F.normalize(key, dim=-1), *memory_rope)
        reads.append(F.scaled_dot_product_attention(per_stream(query[:, part]),
            per_stream(key), per_stream(value), scale=1.0))
        del memory, kv, key, value
    read = torch.cat(reads, dim=0)
    x = x + cross.out(read.permute(2, 0, 1, 3).reshape(rows, width))
    update = block.ffn(block.norm_ffn(x), block.ffn_state_map(state), tokens)
    return x + update


class SlotCompressor(nn.Module):
    """``R`` compressor blocks plus the learned slot and question vectors and, in newer
    checkpoints, a learned question mix per target (``question_mix``)."""

    def __init__(self, decoder_blocks, blocks: int = 2):
        super().__init__()
        if not 1 <= int(blocks) <= len(decoder_blocks):
            raise ValueError(
                f"need 1 <= blocks <= {len(decoder_blocks)} decoder blocks")
        self.blocks = nn.ModuleList(copy.deepcopy(block)
            for block in list(decoder_blocks)[-int(blocks):])
        for block in self.blocks:
            assert block.memory, "copy memory-reading decoder blocks"
            block.compiled = False
            nn.init.zeros_(block.self_attn.out.weight)
            nn.init.zeros_(block.cross_attn.out.weight)
            nn.init.zeros_(block.ffn.down.weight)
            nn.init.zeros_(block.ffn.expert_down)
            nn.init.zeros_(block.memory_state_map.weight)
        width = self.blocks[0].self_attn.qkv.in_features
        self.slot = nn.Parameter(torch.zeros(width))
        self.question = nn.Parameter(torch.zeros(width))
        # Per target, ``[L target layers, S decoder streams]``: how question layer rows are
        # mixed into the decoder's streams. Filled from a checkpoint that has it; empty (the
        # fixed mix) for compressors trained before it.
        self.question_mix = nn.ParameterDict()
        self._register_load_state_dict_pre_hook(self._load_question_mix)

    def _load_question_mix(self, state_dict, prefix, *args):
        head = prefix + "question_mix."
        for key in [key for key in state_dict if key.startswith(head)]:
            name = key[len(head):]
            if name not in self.question_mix:
                self.question_mix[name] = nn.Parameter(
                    torch.empty_like(state_dict[key], device=self.slot.device),
                    requires_grad=self.slot.requires_grad)

    def question_streams(self, model: str, rows: torch.Tensor, streams: int) -> torch.Tensor:
        """Per-layer question rows ``[J, L, d]`` (``question_rows`` without ``streams``) ->
        ``[J, streams, d]``: this compressor's learned mix for ``model``, else the fixed one."""

        mix = self.question_mix[_key(model)] if _key(model) in self.question_mix else None
        if mix is None:
            return rows if int(streams) == rows.shape[1] else pool_question_rows(rows, streams)
        if tuple(mix.shape) != (rows.shape[1], int(streams)):
            raise ValueError(f"{model}: question mix {tuple(mix.shape)} does not map "
                             f"{rows.shape[1]} layer rows to {streams} streams")
        return torch.einsum("jld,ls->jsd", rows.float(), mix.float()).to(rows.dtype)

    def forward(self, translated, questions, slots, *, state, slot_positions,
            question_positions, memory_positions):
        """Slot states ``[M, S, d]``.

        ``translated`` ``[T, S, d]`` are the frozen decoder rows, ``questions``
        ``[J, S, d]`` the question rows from ``question_rows``, ``slots`` the
        document indices ``tau``, ``state`` the target state ``[S, d]``;
        positions are target sequence positions.
        """

        count, streams = int(slots.numel()), translated.shape[1]
        assert questions.shape[1] == streams, (
            f"question rows have {questions.shape[1]} streams, translated rows "
            f"{streams}")
        dtype = translated.dtype
        x = torch.cat((translated.index_select(0, slots) + self.slot.to(dtype),
            questions.to(dtype) + self.question.to(dtype)), dim=0)
        positions = torch.cat((slot_positions, question_positions))
        tokens = x.shape[0]
        head = translated.shape[2] // self.blocks[0].self_attn.heads
        device = translated.device
        rope_rows = rope_tables(positions.to(device), head)
        memory_rope = rope_tables(memory_positions.to(device), head)
        x = x.reshape(tokens * streams, -1)
        for block in self.blocks:
            x = _block(block, x, translated, state, tokens, rope_rows,
                memory_rope)
        return x.view(tokens, streams, -1)[:count]
