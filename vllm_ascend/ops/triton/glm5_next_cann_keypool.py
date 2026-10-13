# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three fused adapters for CANN pooled indexing: inputs, cache writes and indices."""

from types import SimpleNamespace

from vllm.triton_utils import tl, triton

FP8_E4M3_MAX: tl.constexpr = 448.0


@triton.jit(do_not_specialize=["tail_rows"])
def _write_keypool_caches_kernel(
    tail,
    cache,
    scale_cache,
    key,
    gate,
    pooled,
    writes,
    state_rows,
    state,
    slots,
    tail_rows,
    head_dim: tl.constexpr,
    page_rows: tl.constexpr,
    tail_blocks: tl.constexpr,
    capacity: tl.constexpr,
    cache_blocks: tl.constexpr,
    block_size: tl.constexpr,
    cache_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    pooled_stride: tl.constexpr,
    input_stride: tl.constexpr,
    input_width: tl.constexpr,
    speculative: tl.constexpr,
    quantized: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    if row < tail_rows:
        input_row = row // input_width * input_stride + row % input_width
        dest = tl.load(writes + input_row).to(tl.int64)
        valid = (dest >= 0) & (dest // page_rows < tail_blocks) & (dest % page_rows < capacity)
        if speculative:
            k = tl.load(key + row * head_dim + cols, valid & (cols < head_dim), other=0)
            g = tl.load(
                gate + row * head_dim + cols - head_dim, valid & (cols >= head_dim) & (cols < 2 * head_dim), other=0
            )
            tail_value = tl.where(cols < head_dim, k, g)
        else:
            source = tl.load(state_rows + input_row, valid, other=0).to(tl.int64)
            tail_value = tl.load(state + source * 2 * head_dim + cols, valid & (cols < 2 * head_dim), other=0)
        tl.store(tail + dest * 2 * head_dim + cols, tail_value, valid & (cols < 2 * head_dim))
    else:
        pool_row = row - tail_rows
        slot = tl.load(slots + pool_row).to(tl.int64)
        valid = (slot >= 0) & (slot < cache_blocks * block_size)
        pool_value = tl.load(pooled + pool_row * pooled_stride + cols, valid & (cols < head_dim), other=0)
        if quantized:
            pool_value = pool_value.to(tl.float32)
            maximum = tl.max(tl.abs(pool_value), 0)
            scale = tl.where(maximum > 0, maximum / FP8_E4M3_MAX, 1.0)
            pool_value = pool_value / scale
            tl.store(scale_cache + slot // block_size * scale_stride + slot % block_size, scale, valid)
        address = cache + slot // block_size * cache_stride + slot % block_size * head_dim + cols
        tl.store(address, pool_value, valid & (cols < head_dim))


def write_keypool_caches(tail, cache, key, gate, pooled, metadata, tail_metadata, state, scale_cache=None):
    inputs = tail_metadata.cann_inputs
    writes = inputs.write_indices
    _write_keypool_caches_kernel[(writes.numel() + pooled.shape[0],)](
        tail,
        cache,
        scale_cache,
        key,
        gate,
        pooled,
        writes,
        inputs.state_indices,
        state,
        metadata.pooled_slot_mapping,
        writes.numel(),
        cache.shape[-1],
        inputs.page_rows,
        tail.shape[0],
        tail.shape[1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        scale_cache.stride(0) if scale_cache is not None else 0,
        pooled.stride(0),
        writes.stride(0),
        writes.shape[1],
        key is not None,
        scale_cache is not None,
        triton.next_power_of_2(2 * cache.shape[-1]),
    )


@triton.jit(do_not_specialize=["num_reqs"])
def _pack_indices_kernel(
    indices,
    positions,
    cumulative,
    output,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    topk: tl.constexpr,
    pool_size: tl.constexpr,
    num_reqs,
    output_cols: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    valid_row = row < tl.load(cumulative + num_reqs - 1)
    position = tl.load(positions + row).to(tl.int64)
    tail_start = (position + 1) // pool_size * pool_size
    history = tl.minimum(tail_start, topk).to(tl.int32)
    value = tl.load(indices + row * input_stride + cols, valid_row & (cols < history), other=-1)
    tail_columns = cols - history
    tail_count = (position + 1 - tail_start).to(tl.int32)
    # Keep wide index vectors in int32: Ascend scalarizes int64 vector
    # conversion to the int32 SFA output. Only position arithmetic is int64.
    tail_value = tail_start.to(tl.int32) + tail_columns
    value = tl.where(
        valid_row & (tail_columns >= 0) & (tail_columns < tail_count) & (cols < topk + pool_size - 1),
        tail_value,
        value,
    )
    tl.store(output + row * output_stride + cols, value, cols < output_cols)


def pack_keypool_indices(indices, metadata, topk, pool_size, output_buffer=None):
    positions, cumulative = metadata.positions, metadata.cum_query_lens
    output = indices if output_buffer is None else output_buffer[: indices.shape[0]]
    _pack_indices_kernel[(indices.shape[0],)](
        indices,
        positions,
        cumulative,
        output,
        indices.stride(0),
        output.stride(0),
        topk,
        pool_size,
        cumulative.numel(),
        output.shape[1],
        triton.next_power_of_2(output.shape[1]),
    )
    return output[:, : topk + pool_size - 1].unsqueeze(1)


METADATA_BLOCK = 64
# Bound the request inversion tile to fit Ascend unified buffer storage.
METADATA_REQUEST_BLOCK = 128


@triton.jit(do_not_specialize=["n", "tokens", "source", "actual", "pooled_rows"])
def _build_keypool_metadata_kernel(
    raw_query,
    raw_lengths,
    raw_positions,
    raw_slots,
    pages,
    positions,
    slots,
    seq,
    raw,
    query,
    ends,
    starts,
    tails,
    offsets,
    compact,
    history,
    writes,
    state,
    retained,
    n,
    tokens,
    source,
    actual,
    pooled_rows,
    query_stride: tl.constexpr,
    seq_stride: tl.constexpr,
    pos_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    page_stride: tl.constexpr,
    logical: tl.constexpr,
    pool: tl.constexpr,
    capacity: tl.constexpr,
    page_rows: tl.constexpr,
    width: tl.constexpr,
    tail_mode: tl.constexpr,
    speculative: tl.constexpr,
    BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q0 = tl.load(raw_query + row * query_stride, row <= n, other=0)
    q0 = tl.minimum(tl.where(q0 < 0, actual, q0), actual)
    q1 = tl.load(raw_query + (row + 1) * query_stride, row < n, other=0)
    q1 = tl.minimum(tl.where(q1 < 0, actual, q1), actual)
    length = q1 - q0
    raw_len = tl.load(raw_lengths + row * seq_stride, row < n, other=0)
    start = raw_len - length
    if tail_mode:
        cols = tl.arange(0, COLUMNS)
        page = tl.load(pages + row * page_stride, row < n, other=0).to(tl.int64)
        hist = tl.maximum(page, 0)[:, None] * page_rows + ((start - start % pool)[:, None] + cols[None, :]) % capacity
        tl.store(history + row[:, None] * pool + cols[None, :], hist, (row[:, None] < n) & (cols[None, :] < pool))
        if speculative:
            source_row = q1[:, None] - width + cols[None, :]
            safe = tl.maximum(source_row, 0)
            tl.store(
                retained + row[:, None] * width + cols[None, :],
                safe.to(tl.int64),
                (row[:, None] < n) & (cols[None, :] < width),
            )
            slot = tl.load(
                raw_slots + safe * slot_stride, (row[:, None] < n) & (cols[None, :] < width) & (safe < actual), other=-1
            ).to(tl.int64)
            address = slot // capacity * page_rows + slot % capacity
            address = tl.where(source_row < q0[:, None], -1, address)
        else:
            end = start % pool + length
            state_row = (row + 1 + tl.where(end >= pool, n, 0))[:, None] * pool + cols[None, :]
            tl.store(
                state + row[:, None] * width + cols[None, :],
                state_row.to(tl.int64),
                (row[:, None] < n) & (cols[None, :] < width),
            )
            pos = (start + length - end % pool)[:, None] + cols[None, :]
            address = page[:, None] * page_rows + pos % capacity
            address = tl.where(
                (length[:, None] > 0) & (cols[None, :] < (end % pool)[:, None]) & (page[:, None] >= 0), address, -1
            )
        tl.store(writes + row[:, None] * width + cols[None, :], address, (row[:, None] < n) & (cols[None, :] < width))
    else:
        pos = tl.load(raw_positions + row * pos_stride, row < source, other=0).to(tl.int64)
        slot = tl.load(raw_slots + row * slot_stride, row < source, other=-1).to(tl.int64)
        compressed = slot // logical * (logical // pool) + slot % logical // pool
        tl.store(positions + row, pos, row < tokens)
        tl.store(slots + row, tl.where((slot >= 0) & ((pos + 1) % pool == 0), compressed, -1), row < tokens)
        tl.store(query + row, q0, row <= n)
        tl.store(seq + row, raw_len // pool, row < n)
        tl.store(raw + row, raw_len, row < n)
        tl.store(ends + row, q1, row < n)
        tl.store(starts + row, start, row < n)
        tl.store(tails + row, raw_len % pool, row < n)
        tl.store(offsets + row, start % pool, row < n)
        # Invert request pool counts to locate each compact output's boundary
        # token directly; evicted slots still occupy their CANN output row.
        prefix = tl.full((), 0, tl.int32)
        compact_address = tl.full((BLOCK,), -1, tl.int64)
        for request_start in range(0, n, REQUESTS):
            req = request_start + tl.arange(0, REQUESTS)
            begin = tl.load(raw_query + req * query_stride, req < n, other=0)
            finish = tl.load(raw_query + (req + 1) * query_stride, req < n, other=0)
            begin = tl.minimum(tl.where(begin < 0, actual, begin), actual)
            finish = tl.minimum(tl.where(finish < 0, actual, finish), actual)
            total = tl.load(raw_lengths + req * seq_stride, req < n, other=0)
            first = total - (finish - begin)
            count = total // pool - first // pool
            inclusive = prefix + tl.cumsum(count, 0)
            prior = inclusive - count
            match = (req[None, :] < n) & (row[:, None] >= prior[None, :]) & (row[:, None] < inclusive[None, :])
            boundary = begin[None, :] + pool - first[None, :] % pool - 1 + (row[:, None] - prior[None, :]) * pool
            token = tl.sum(tl.where(match, boundary, 0), 1)
            valid = tl.sum(match.to(tl.int32), 1) > 0
            pool_slot = tl.load(
                raw_slots + token * slot_stride, valid & (token < source) & (row < pooled_rows), other=-1
            ).to(tl.int64)
            address = pool_slot // logical * (logical // pool) + pool_slot % logical // pool
            compact_address = tl.where(valid & (pool_slot >= 0), address, compact_address)
            prefix += tl.sum(count, 0)
        tl.store(compact + row + 1, compact_address, row < pooled_rows)
        if tl.program_id(0) == 0:
            tl.store(compact, -1)


def build_keypool_metadata(common, buffers, tokens, source, logical, ratio):
    n = common.num_reqs
    pooled = min(tokens, tokens // ratio + n)
    # Fully padded buckets can have empty source tensors. Keep masked input
    # pointers backed by allocated storage without adding a device operation.
    source_positions = common.positions if source else buffers.positions
    source_slots = common.slot_mapping if source else buffers.slot_mapping
    _build_keypool_metadata_kernel[(triton.cdiv(max(tokens, n + 1), METADATA_BLOCK),)](
        common.query_start_loc,
        common.seq_lens,
        source_positions,
        source_slots,
        None,
        buffers.positions,
        buffers.slot_mapping,
        buffers.seq_lens,
        buffers.raw_seq_lens,
        buffers.query_start_loc,
        buffers.cum_query_lens,
        buffers.start_pos,
        buffers.pool_tail,
        buffers.start_offsets,
        buffers.pooled_slot_mapping,
        None,
        None,
        None,
        None,
        n,
        tokens,
        source,
        common.num_actual_tokens,
        pooled,
        common.query_start_loc.stride(0),
        common.seq_lens.stride(0),
        source_positions.stride(0),
        source_slots.stride(0),
        0,
        logical,
        ratio,
        0,
        0,
        0,
        False,
        False,
        METADATA_BLOCK,
        triton.next_power_of_2(max(1, min(n, METADATA_REQUEST_BLOCK))),
        1,
    )
    return (
        buffers.positions[:tokens],
        buffers.slot_mapping[:tokens],
        buffers.seq_lens[:n],
        buffers.raw_seq_lens[:n],
        buffers.query_start_loc[: n + 1],
        buffers.cum_query_lens[:n],
        buffers.start_pos[:n],
        buffers.pool_tail[:n],
    )


def build_tail_inputs(common, pool, capacity, page_rows, retained_width, buffers):
    n = common.num_reqs
    width = retained_width or pool
    history = buffers["history"][:n]
    writes = buffers["writes"].view(-1)[: n * width].view(n, width)
    state = buffers["state"].view(-1)[: n * width].view(n, width)
    retained = buffers["retained"].view(-1)[: n * width].view(n, width) if retained_width else None
    _build_keypool_metadata_kernel[(triton.cdiv(n + 1, METADATA_BLOCK),)](
        common.query_start_loc,
        common.seq_lens,
        None,
        common.slot_mapping,
        common.block_table_tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        history,
        writes,
        state,
        retained,
        n,
        0,
        0,
        common.num_actual_tokens,
        0,
        common.query_start_loc.stride(0),
        common.seq_lens.stride(0),
        0,
        common.slot_mapping.stride(0),
        common.block_table_tensor.stride(0),
        0,
        pool,
        capacity,
        page_rows,
        width,
        True,
        retained_width > 0,
        METADATA_BLOCK,
        1,
        triton.next_power_of_2(max(pool, width)),
    )
    return SimpleNamespace(
        history_indices=history,
        write_indices=writes,
        state_indices=state,
        retained_indices=retained,
        page_rows=page_rows,
    )
