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
#

import einops
import torch
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention


def _patched_forward(
    self: Qwen2_5_VisionAttention,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(self, "_head_size_orig"):
        self._head_size_orig = self.hidden_size_per_attention_head

    def _sync_head_dim(head_dim: int) -> None:
        if head_dim == self.hidden_size_per_attention_head:
            return

        self.hidden_size_per_attention_head = head_dim
        if hasattr(self.attn, "head_size"):
            if not hasattr(self.attn, "head_size_orig"):
                self.attn.head_size_orig = getattr(
                    self, "_head_size_orig", self.attn.head_size
                )
            self.attn.head_size = head_dim

        # Keep original scaling based on real (non-padded) head dim.
        if hasattr(self.attn, "scale"):
            base = getattr(
                self.attn,
                "head_size_orig",
                getattr(self, "_head_size_orig", head_dim),
            )
            self.attn.scale = float(base) ** -0.5

        # 310P rotary patch uses this attr to rotate only real dims.
        if (
            hasattr(self, "apply_rotary_emb")
            and not hasattr(self.apply_rotary_emb, "head_size_orig")
        ):
            self.apply_rotary_emb.head_size_orig = getattr(
                self, "_head_size_orig", head_dim
            )

    # [s, b, c] --> [s, b, head * 3 * head_dim]
    x, _ = self.qkv(x)
    seq_len, batch_size, _ = x.shape

    qkv = einops.rearrange(
        x,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=self.num_attention_heads_per_partition,
    )

    if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
        qk, v = qkv[:, :, :2], qkv[:, :, 2]

        qk_reshaped = einops.rearrange(
            qk, "b s two head head_dim -> (two b) s head head_dim", two=2
        )
        qk_reshaped = qk_reshaped.contiguous()
        qk_rotated = self.apply_rotary_emb(
            qk_reshaped,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )
        head_dim = qk_rotated.shape[-1]
        _sync_head_dim(int(head_dim))
        qk_rotated = qk_rotated.view(
            2,
            batch_size,
            seq_len,
            self.num_attention_heads_per_partition,
            int(head_dim),
        )
        q, k = qk_rotated.unbind(dim=0)
    else:
        q, k, v = qkv.unbind(dim=2)
        _sync_head_dim(int(q.shape[-1]))

    context_layer = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )

    context_layer = einops.rearrange(
        context_layer, "b s h d -> s b (h d)", b=batch_size
    ).contiguous()

    output, _ = self.proj(context_layer)
    return output


if not getattr(Qwen2_5_VisionAttention, "_ascend_310p_qwen2_5_vl_patch", False):
    Qwen2_5_VisionAttention.forward = _patched_forward
    Qwen2_5_VisionAttention._ascend_310p_qwen2_5_vl_patch = True
