# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared inputs and existing NPU gather/quantization for CANN KeyPool."""

from dataclasses import dataclass

import torch
import torch_npu


def build_keypool_metadata(common, buffers, tokens, source, logical, ratio):
    if common.positions.device.type == "npu":
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import (  # type: ignore[import-untyped]
            build_keypool_metadata as fused_metadata,
        )

        return fused_metadata(common, buffers, tokens, source, logical, ratio)
    n = common.num_reqs
    positions, slots = buffers.positions[:tokens], buffers.slot_mapping[:tokens]
    seq, raw = buffers.seq_lens[:n], buffers.raw_seq_lens[:n]
    query, ends = buffers.query_start_loc[: n + 1], buffers.cum_query_lens[:n]
    starts, tails = buffers.start_pos[:n], buffers.pool_tail[:n]
    positions[:source].copy_(common.positions[:source])
    positions[source:].zero_()
    slot = common.slot_mapping[:source]
    slots.fill_(-1)
    slots[:source].copy_(
        torch.where(
            (slot >= 0) & ((positions[:source] + 1) % ratio == 0),
            slot // logical * (logical // ratio) + slot % logical // ratio,
            -1,
        )
    )
    raw.copy_(common.seq_lens[:n])
    seq.copy_(raw // ratio)
    query.copy_(common.query_start_loc[: n + 1])
    query.masked_fill_(query < 0, common.num_actual_tokens).clamp_max_(common.num_actual_tokens)
    ends.copy_(query[1:])
    starts.copy_(raw - query.diff())
    tails.copy_(raw % ratio)
    buffers.start_offsets[:n].copy_(starts % ratio)
    completed = ((positions + 1) % ratio == 0) & (buffers.token_indices[:tokens] < query[-1])
    pooled = completed.to(torch.int32).cumsum(0, dtype=torch.int32).sub_(1).clamp_min_(0).long()
    # Incomplete tokens write to a discarded sentinel; completed rows are unique.
    compact = buffers.pooled_slot_mapping[: tokens + 1]
    compact.fill_(-1)
    if tokens:
        compact.scatter_(0, torch.where(completed, pooled + 1, 0), slots)
    return positions, slots, seq, raw, query, ends, starts, tails


@dataclass
class KeypoolTailInputs:
    history_indices: torch.Tensor
    write_indices: torch.Tensor
    state_indices: torch.Tensor
    retained_indices: torch.Tensor | None
    page_rows: int


def build_tail_inputs(common, pool, capacity, page_rows, retained_width, buffers):
    if common.seq_lens.device.type == "npu":
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import (  # type: ignore[import-untyped]
            build_tail_inputs as fused_inputs,
        )

        return fused_inputs(common, pool, capacity, page_rows, retained_width, buffers)
    n = common.num_reqs
    query = common.query_start_loc[: n + 1].clamp(0, common.num_actual_tokens)
    query = torch.where(common.query_start_loc[: n + 1] < 0, common.num_actual_tokens, query)
    lengths = query.diff()
    starts = common.seq_lens[:n] - lengths
    pages = common.block_table_tensor[:n, 0]
    columns = buffers["columns"]
    history = buffers["history"][:n]
    history.copy_(
        pages.clamp_min(0)[:, None] * page_rows + ((starts - starts % pool)[:, None] + columns[:pool]) % capacity
    )
    width = retained_width if retained_width else pool
    writes, state = buffers["writes"][:n, :width], buffers["state"][:n, :width]
    retained = None
    if retained_width:
        raw = query[1:, None] - width + columns[:width]
        # Shape the prefix as contiguous rows to avoid per-layer flatten copies.
        retained = buffers["retained"].view(-1)[: n * width].view(n, width)
        retained.copy_(raw.clamp_min(0))
        slots = torch.index_select(common.slot_mapping, 0, retained.flatten()).view(n, width)
        writes.copy_(slots // capacity * page_rows + slots % capacity)
        writes.masked_fill_(raw < query[:-1, None], -1)
    else:
        end = starts % pool + lengths
        state.copy_((buffers["requests"][:n] + 1 + (end >= pool) * n)[:, None] * pool + columns[:pool])
        positions = (starts + lengths - end % pool)[:, None] + columns[:pool]
        writes.copy_(pages[:, None] * page_rows + positions % capacity)
        writes.masked_fill_(
            (lengths[:, None] == 0) | (columns[:pool] >= (end % pool)[:, None]) | (pages[:, None] < 0), -1
        )
    return KeypoolTailInputs(history, writes, state, retained, page_rows)


def _cache_rows(cache):
    width = cache.shape[-1]
    if cache.stride(0) % width:
        raise ValueError("KeyPool cache pages must align to complete rows.")
    rows = (cache.shape[0] - 1) * (cache.stride(0) // width) + cache.shape[1]
    return cache.as_strided((rows, width), (width, 1))


def prepare_keypool_state(tail, starts, tail_metadata, pool):
    n, width = starts.numel(), tail.shape[-1]
    inputs = tail_metadata.cann_inputs
    if inputs.page_rows * width != tail.stride(0):
        raise ValueError("KeyPool tail metadata must preserve the physical cache page stride.")
    state = tail.new_empty((1 + 2 * n, pool, width))
    torch.index_select(
        _cache_rows(tail), 0, inputs.history_indices.flatten(), out=state[1 : n + 1].view(n * pool, width)
    )
    return state


def gather_retained_hidden(hidden, indices):
    return torch.index_select(hidden, 0, indices.flatten())


def quantize_keypool_query(query):
    result, scale = torch_npu.npu_dynamic_quant(query, dst_type=torch.float8_e4m3fn)
    scale.masked_fill_(scale == 0, 1.0)
    return result, scale
