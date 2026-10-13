# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_cann_keypool import (
    build_keypool_metadata,
    build_tail_inputs,
    pack_keypool_indices,
    write_keypool_caches,
)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_cann_cache_adapters_preserve_storage_and_refresh_graph_inputs(speculative, use_graph):
    dim, pool, tokens = 128, 4, 3
    capacity = 7 if speculative else pool
    tail_stride, cache_stride = 40 * 2 * dim, 48 * dim
    tail_storage = torch.full((2 * tail_stride + dim,), 17.0, device="npu")
    tail = tail_storage.as_strided((2, capacity, 2 * dim), (tail_stride, 2 * dim, 1), dim)
    cache_storage = torch.full((3 * cache_stride + dim,), 17, dtype=torch.bfloat16, device="npu")
    cache = cache_storage.as_strided((3, 32, 1, dim), (cache_stride, dim, dim, 1), dim)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, 3], device="npu", dtype=torch.int32),
        seq_lens=torch.tensor([5], device="npu", dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2], device="npu", dtype=torch.int64),
        block_table_tensor=torch.zeros(1, 1, device="npu", dtype=torch.int32),
    )
    buffers = {
        name: torch.empty(1, capacity, device="npu", dtype=torch.int64) for name in ["writes", "state", "retained"]
    }
    buffers["history"] = torch.empty(1, pool, device="npu", dtype=torch.int64)
    key = torch.randn(pool, dim, device="npu")
    gate = torch.randn_like(key)
    key[0].fill_(float("nan"))
    gate[0].fill_(float("nan"))
    pooled = torch.randn(3, dim, device="npu", dtype=torch.bfloat16)
    state = torch.randn(3, pool, 2 * dim, device="npu")
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor([0, 96, -1], device="npu"))
    packed = SimpleNamespace(
        positions=torch.tensor([0, 1, 4, 7, 18], device="npu"),
        cum_query_lens=torch.tensor([2, 4], device="npu", dtype=torch.int64),
    )
    indices = torch.full((5, 11), -1, dtype=torch.int32, device="npu")
    output = torch.full((6, 16), 19, dtype=torch.int32, device="npu")

    def run():
        inputs = build_tail_inputs(
            common, pool, capacity, tail_stride // (2 * dim), pool if speculative else 0, buffers
        )
        write_keypool_caches(
            tail,
            cache,
            key if speculative else None,
            gate if speculative else None,
            pooled,
            metadata,
            SimpleNamespace(cann_inputs=inputs),
            state,
        )
        return pack_keypool_indices(indices, packed, 8, pool, output)

    for _ in range(3):
        result = run()
    graph = torch.npu.NPUGraph() if use_graph else None
    if graph is not None:
        with torch.npu.graph(graph):
            result = run()
    for start, length, slots in [(2, 3, [0, 1, 2]), (0, 1, [0, capacity - 1, 2 * capacity])]:
        tail_storage.fill_(17)
        cache_storage.fill_(17)
        output.fill_(19)
        common.query_start_loc[1].fill_(length)
        common.seq_lens.fill_(start + length)
        common.slot_mapping.copy_(torch.tensor(slots, device="npu", dtype=torch.int64))
        key.add_(1)
        gate.add_(2)
        pooled.add_(1)
        state.add_(1)
        expected_tail, expected_cache = tail_storage.cpu(), cache_storage.cpu()
        if speculative:
            for row in range(pool):
                source = length - pool + row
                if 0 <= source < length and 0 <= slots[source] < 2 * capacity:
                    slot = slots[source]
                    offset = dim + slot // capacity * tail_stride + slot % capacity * 2 * dim
                    expected_tail[offset : offset + dim] = key[row].cpu()
                    expected_tail[offset + dim : offset + 2 * dim] = gate[row].cpu()
        else:
            end = start % pool + length
            for col in range(end % pool):
                position = start + length - end % pool + col
                offset = dim + position % capacity * 2 * dim
                expected_tail[offset : offset + 2 * dim] = state[1 + int(end >= pool), col].cpu()
        expected_cache[dim : 2 * dim] = pooled[0].cpu()
        raw = torch.full(indices.shape, -1, dtype=torch.int32)
        expected_output = torch.full(output.shape, 19, dtype=torch.int32)
        expected_output[:5].fill_(-1)
        for row, position in enumerate(packed.positions.cpu().tolist()):
            tail_start = (position + 1) // pool * pool
            history = list(range(min(tail_start, 8) - 1, -1, -1))
            raw[row, : len(history)] = torch.tensor(history, dtype=torch.int32)
            if row < 4:
                valid = history + list(range(tail_start, position + 1))
                expected_output[row, : len(valid)] = torch.tensor(valid, dtype=torch.int32)
        indices.copy_(raw)
        if graph is None:
            result = run()
        else:
            graph.replay()
        torch.testing.assert_close(tail_storage.cpu(), expected_tail, rtol=0, atol=0)
        torch.testing.assert_close(cache_storage.cpu(), expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
        assert result.shape == (5, 1, 11)


@pytest.mark.parametrize("n,actual", [(2, 8), (257, 8), (257, 0)])
def test_compact_metadata_preserves_eviction_and_padding_in_graph(n, actual):
    pool, tokens = 4, 10
    common = SimpleNamespace(
        num_reqs=n,
        num_actual_tokens=actual,
        query_start_loc=torch.tensor(
            ([0, 6, 8] if actual else [0, 0, 0]) + [-1] * (n - 2), device="npu", dtype=torch.int32
        ),
        seq_lens=torch.tensor(([65544, 65545] if actual else [0, 0]) + [0] * (n - 2), device="npu", dtype=torch.int32),
        positions=torch.tensor(
            [65538, 65539, 65540, 65541, 65542, 65543, 65543, 65544][:actual], device="npu", dtype=torch.int64
        ),
        slot_mapping=torch.tensor([2, -1, 4, 5, 6, 7, 135, 136][:actual], device="npu", dtype=torch.int64),
    )
    buffers = SimpleNamespace(
        **{
            name: torch.empty(size, device="npu", dtype=dtype)
            for name, size, dtype in [
                ("positions", tokens, torch.int64),
                ("slot_mapping", tokens, torch.int64),
                ("seq_lens", n, torch.int64),
                ("raw_seq_lens", n, torch.int32),
                ("query_start_loc", n + 1, torch.int32),
                ("cum_query_lens", n, torch.int64),
                ("start_pos", n, torch.int32),
                ("pool_tail", n, torch.int64),
                ("start_offsets", n, torch.int32),
                ("pooled_slot_mapping", tokens + 1, torch.int64),
            ]
        }
    )
    for _ in range(3):
        build_keypool_metadata(common, buffers, tokens, actual, 128, pool)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        build_keypool_metadata(common, buffers, tokens, actual, 128, pool)
    for _ in range(2):
        graph.replay()
        pos, slots = common.positions.cpu(), common.slot_mapping.cpu()
        boundary = (pos + 1) % pool == 0
        compressed = torch.where((slots >= 0) & boundary, slots // 128 * 32 + slots % 128 // pool, -1)
        expected = torch.full((4,), -1, dtype=torch.int64)
        valid = compressed[boundary]
        expected[: valid.numel()] = valid
        torch.testing.assert_close(buffers.pooled_slot_mapping[1:5].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(buffers.slot_mapping[:actual].cpu(), compressed, rtol=0, atol=0)
        assert buffers.slot_mapping[actual:].cpu().tolist() == [-1] * (tokens - actual)
        assert buffers.start_pos[:2].cpu().tolist() == [
            int(common.seq_lens[0].cpu()) - (6 if actual else 0),
            int(common.seq_lens[1].cpu()) - (2 if actual else 0),
        ]
        common.positions.add_(1)
        assert buffers.start_pos[2:].cpu().tolist() == [0] * (n - 2)
        common.seq_lens[:2].add_(int(actual > 0))
