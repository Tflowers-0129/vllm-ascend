#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from collections.abc import Iterable
# mypy: ignore-errors


import os
from pathlib import Path

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule_pytorch
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.utils import enable_sp, parse_layer_idx


_DUMP_ROOT = Path(__file__).resolve().parents[3] / "causal_conv1d_dump"
_DUMP_STEP = 0


def _detach_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    return x.detach().cpu().contiguous()


def _canonical_conv_weight(weight: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if weight.shape[0] == feature_dim:
        return weight
    return weight.transpose(0, 1)


def _select_conv_state(conv_state: torch.Tensor, cache_indices: torch.Tensor, feature_dim: int) -> torch.Tensor:
    state = conv_state.index_select(0, cache_indices.to(torch.long))
    if state.shape[-2] == feature_dim:
        return state
    return state.transpose(-1, -2)


def _should_dump_causal_conv1d(
    forward_context,
    prefix: str,
    attn_metadata: GDNAttentionMetadata,
) -> bool:
    return (
        parse_layer_idx(prefix) == 0
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
        and not getattr(forward_context, "in_profile_run", False)
        and not getattr(forward_context, "capturing", False)
    )


def _dump_causal_conv1d(
    prefix: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state_before: torch.Tensor,
    y: torch.Tensor,
    conv_state_after: torch.Tensor,
    cache_indices: torch.Tensor,
) -> None:
    global _DUMP_STEP

    dump_dir = _DUMP_ROOT / f"pid_{os.getpid()}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"step_{_DUMP_STEP:06d}.pt"
    torch.save(
        {
            "step": _DUMP_STEP,
            "prefix": prefix,
            "input": _detach_cpu(x),
            "weight": _detach_cpu(weight),
            "bias": _detach_cpu(bias),
            "cache_indices": _detach_cpu(cache_indices.to(torch.int64)),
            "conv_state_before": _detach_cpu(conv_state_before),
            "output": _detach_cpu(y),
            "conv_state_after": _detach_cpu(conv_state_after),
        },
        dump_path,
    )
    _DUMP_STEP += 1


class Ascend310Qwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # Core attention computation (called by custom op).

        # NOTE: The processing logic of Qwen3_5GatedDeltaNet is the same as Qwen3NextGatedDeltaNet.
        # However, because the ops `torch_npu.npu_recurrent_gated_delta_rule`
        # currently does not support `ssm_state` inputs in float32 format,
        # we temporarily retain the current _forward_core implementation.
        # Once the ops supports float32 `ssm_state`, this patch should be removed.

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[0]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                mixed_qkv_non_spec = causal_conv1d_fn(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=self_kv_cache[0],
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            non_spec_cache_indices = non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens]
            should_dump = _should_dump_causal_conv1d(forward_context, self.prefix, attn_metadata)
            if should_dump:
                dump_input = mixed_qkv_non_spec.detach().clone()
                feature_dim = mixed_qkv_non_spec.shape[-1]
                conv_weight_dump = _canonical_conv_weight(conv_weights, feature_dim)
                conv_state_before = _select_conv_state(conv_state, non_spec_cache_indices, feature_dim)
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_cache_indices,
            )
            if should_dump:
                conv_state_after = _select_conv_state(conv_state, non_spec_cache_indices, feature_dim)
                _dump_causal_conv1d(
                    prefix=self.prefix,
                    x=dump_input,
                    weight=conv_weight_dump,
                    bias=self.conv1d.bias,
                    conv_state_before=conv_state_before,
                    y=mixed_qkv_non_spec,
                    conv_state_after=conv_state_after,
                    cache_indices=non_spec_cache_indices,
                )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        g, beta = fused_gdn_gating_pytorch(self.A_log, a, b, self.dt_bias)
        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2. Recurrent attention

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule_pytorch(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_gated_delta_rule_pytorch(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )

                # Init cache
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule_pytorch(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, _ = fused_recurrent_gated_delta_rule_pytorch(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
        maybe_save_kv_layer_to_connector("", [])


Qwen3_5GatedDeltaNet._forward_core = Ascend310Qwen3_5GatedDeltaNet._forward_core
