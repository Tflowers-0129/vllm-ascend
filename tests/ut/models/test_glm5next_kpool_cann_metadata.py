# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.utils import vllm_version_is

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder, AscendIndexerKPoolTailMetadataBuilder
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer


@pytest.fixture(autouse=True)
def disable_cp(monkeypatch):
    monkeypatch.setattr("vllm_ascend.attention.indexer_kpool.enable_dsa_cp", lambda: False)


def _config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=8192),
    )


def _builder(config):
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    spec = AscendMLAAttentionSpec(
        block_size=config.cache_config.block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version="glm5_next",
        **ratio,
    )
    return AscendIndexerKPoolMetadataBuilder(spec, ["layer.indexer.k_cache"], config, torch.device("cpu"))


class _Common(SimpleNamespace):
    @property
    def _seq_lens_cpu(self):
        raise AssertionError("CANN metadata must not read host lengths")


def _common():
    return _Common(
        num_reqs=2,
        num_input_tokens=10,
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 6, 8], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([2, -1, 4, 5, 6, 7, 135, 136, -1, -1]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7, 7, 8, 3, 7]),
    )


@pytest.mark.parametrize("unpadded", [None, "positions", "slot_mapping"])
def test_compact_pool_slots_skip_eviction_and_padding_and_keep_capture_storage(unpadded):
    builder, common = _builder(_config()), _common()
    if unpadded:
        setattr(common, unpadded, getattr(common, unpadded)[: common.num_actual_tokens])
    first = builder.build(0, common)
    assert first.start_pos.tolist() == [2, 7]
    assert first.query_start_loc.tolist() == [0, 6, 8]
    assert first.seq_lens.tolist() == [2, 2]
    assert first.pooled_slot_mapping.tolist() == [-1, 1, 33, -1]
    assert first.keypool_state_table.tolist() == [[1, 3, 3, 3], [2, 4, 4, 4]]
    fields = ("start_pos", "query_start_loc", "pool_tail", "pooled_slot_mapping", "keypool_state_table")
    pointers = [getattr(first, f).data_ptr() for f in fields]
    common.seq_lens.add_(1)
    common.positions.add_(1)
    second = builder.build(0, common)
    assert pointers == [getattr(second, f).data_ptr() for f in fields]
    assert first.start_pos.tolist() == [3, 8]
    assert first.keypool_start_offsets.tolist() == [3, 0]


@pytest.mark.parametrize("width", [1, 4])
def test_tail_inputs_retain_draft_rows_page_stride_and_step_isolation(width):
    config, common = _config(), _common()
    spec = AscendIndexerKPoolTailSpec(
        block_size=7,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
        page_size_padded=81920,
    )
    builder = AscendIndexerKPoolTailMetadataBuilder(spec, ["layer.indexer.tail_cache"], config, torch.device("cpu"))
    common.max_query_len = width
    common.num_actual_tokens = common.num_input_tokens = 2 * width
    common.query_start_loc = torch.tensor([0, width, 2 * width], dtype=torch.int32)
    common.seq_lens = torch.tensor([16 + width, 27 + width], dtype=torch.int32)
    common.slot_mapping = torch.arange(2 * width)
    first = builder.build(0, common)
    assert first.cann_inputs.retained_indices.tolist() == [list(range(width)), list(range(width, 2 * width))]
    assert first.cann_inputs.page_rows == 80
    other = _Common(**vars(common))
    other.slot_mapping = common.slot_mapping.clone()
    isolated = builder.build_for_drafting(other, 1)
    assert isolated.cann_inputs.write_indices.data_ptr() != first.cann_inputs.write_indices.data_ptr()
    common.query_start_loc[-1] = -1
    common.num_actual_tokens = width
    builder.build_for_graph_capture(common)
    assert first.cann_inputs.write_indices[1].tolist() == [-1] * width
    assert isolated.cann_inputs.write_indices[1].tolist() != [-1] * width


@pytest.mark.parametrize("end", [-1, 2])
def test_empty_padded_request_preserves_common_metadata(end):
    common = _common()
    common.num_input_tokens, common.num_actual_tokens = 2, 1
    common.query_start_loc = torch.tensor([0, 1, end], dtype=torch.int32)
    common.seq_lens = torch.tensor([9, 0], dtype=torch.int32)
    common.positions = torch.tensor([8, 3])
    common.slot_mapping[:2] = torch.tensor([8, -1])
    metadata = _builder(_config()).build(0, common)
    assert metadata.query_start_loc.tolist() == [0, 1, 1]
    assert metadata.start_pos.tolist() == [8, 0]
    assert metadata.pooled_slot_mapping.tolist() == [-1, -1]
    assert common.query_start_loc.tolist() == [0, 1, end]


@pytest.mark.parametrize("logical,primary", [(128, True), (256, False), (1152, False)])
def test_mtp_preserves_expanded_kernel_pages(logical, primary):
    config, common = _config(), _common()
    config.cache_config.block_size = logical
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    builder = _builder(config)
    split, width = logical // 128, (8192 + logical - 1) // logical
    physical = torch.arange(1, 2 * width + 1, dtype=torch.int32).view(2, width)
    table = (physical[:, :, None] * split + torch.arange(split)).reshape(2, -1)
    common.block_table_tensor = table
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache, proposer.kv_cache_gid, proposer.max_model_len = True, 0, 8192
    proposer.vllm_config = config
    gid = 0 if primary else 1
    proposer._multi_group_slot_mapping_buffers = {(gid, 0): common.slot_mapping.clone()}
    proposer._multi_group_block_table_buffers = {gid: torch.empty_like(table)}
    cache = SimpleNamespace(
        is_circular_group=False,
        compute_slot_mapping=lambda *_: None,
        get_device_tensor=lambda: table,
        slot_mapping=SimpleNamespace(gpu=common.slot_mapping),
    )
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[cache, cache]))
    group = SimpleNamespace(kv_cache_group_id=gid, get_metadata_builder=lambda: builder)
    draft = proposer._common_attn_metadata_for_draft_group(common, group, common.num_input_tokens)
    torch.testing.assert_close(builder.build(0, draft).block_table, table)
