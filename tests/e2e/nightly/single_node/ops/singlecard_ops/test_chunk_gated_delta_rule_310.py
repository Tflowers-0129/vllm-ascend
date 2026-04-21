from typing import Optional

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.set_compile_mode(jit_compile=False)

def npu_chunk_gated_delta_rule_310(q, k, v, g, beta):
    out, state = torch.ops._C_ascend.npu_chunk_gated_delta_rule_310(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta
    )
    return out, state


def chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        key = torch.nn.functional.normalize(key, p=2, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def golden_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
):
    batch_size = cu_seqlens.size(0) - 1
    num_heads = q.shape[-2]
    k_head_dim = k.shape[-1]
    v_head_dim = v.shape[-1]
    core_attn_out = []
    last_recurrent_state = torch.empty((batch_size, num_heads, k_head_dim, v_head_dim), dtype=torch.float32)
    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]
    return final_cor_attn_out, last_recurrent_state


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("seqlen", [8, 31, 129, 512, 763, 1024, 1373, 2048])
@pytest.mark.parametrize("headnum", [(8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
def test_fused_chunk_gated_delta_rule_310(batch_size, seqlen, headnum, headdim_k, headdim_v):
    enable_custom_op()
    dtype = torch.float16
    headnum_k, headnum_v = headnum
    T = batch_size * seqlen
    q = torch.rand((T, headnum_k, headdim_k), dtype=dtype).npu()
    k = torch.rand((T, headnum_k, headdim_k), dtype=dtype).npu()
    v = torch.rand((T, headnum_v, headdim_v), dtype=dtype).npu()
    g = torch.rand((T, headnum_v), dtype=torch.float32).npu() * -1.0
    beta = torch.rand((T, headnum_v), dtype=dtype).npu()

    if headnum_v // headnum_k > 1:
        q = q.repeat_interleave(headnum_v // headnum_k, dim=-2)
        k = k.repeat_interleave(headnum_v // headnum_k, dim=-2)
    initial_state = None
    actual_seq_lengths = torch.tensor([seqlen] * batch_size, dtype=torch.int32)
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    out_golden, state_golden = golden_chunk_gated_delta_rule(
        q.clone().unsqueeze(0),
        k.clone().unsqueeze(0),
        v.clone().unsqueeze(0),
        g.clone().unsqueeze(0),
        beta.clone().unsqueeze(0),
        cu_seqlens)
    out_golden = out_golden.view(batch_size, -1, headnum_v, headdim_v)
    out, state = npu_chunk_gated_delta_rule_310(
        q.view(batch_size, -1, headnum_v, headdim_k),
        k.view(batch_size, -1, headnum_v, headdim_k),
        v.view(batch_size, -1, headnum_v, headdim_v),
        g.view(batch_size, -1, headnum_v),
        beta.view(batch_size, -1, headnum_v))

    torch.testing.assert_close(
        out.to(torch.float32).cpu(),
        out_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state.to(torch.float32).cpu(),
        state_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
