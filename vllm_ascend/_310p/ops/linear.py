from __future__ import annotations

import gc
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_npu
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs


_ALIGN = 16
_NZ_FORMAT = 29

_QKV_PAD_REGISTRY: Dict[str, Dict[str, int]] = {}


# ----------------------------
# Minimal helpers
# ----------------------------

def _align_up(x: int, a: int = _ALIGN) -> int:
    return ((x + a - 1) // a) * a if a > 0 else x


def _strip_suffix(prefix: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for s in suffixes:
        if prefix.endswith(s):
            return prefix[: -len(s)]
    return None


def _qkv_base_prefix(prefix: str) -> Optional[str]:
    return _strip_suffix(prefix, (".qkv", ".qkv_proj", ".q_proj"))


def _proj_base_prefix(prefix: str) -> Optional[str]:
    return _strip_suffix(prefix, (".proj", ".o_proj", ".out_proj", ".output_proj"))


def _register_qkv_pad_meta(prefix: str, *, num_heads: int, head_size: int, head_size_pad: int) -> None:
    base = _qkv_base_prefix(prefix)
    if base is None:
        return
    if head_size_pad <= head_size:
        return
    _QKV_PAD_REGISTRY[base] = {
        "proj_in_pad": int(num_heads * head_size_pad),
        "head_size": int(head_size),
        "head_size_pad": int(head_size_pad),
    }


def _get_force_in_pad_for_proj(prefix: str) -> Optional[int]:
    base = _proj_base_prefix(prefix)
    if base is None:
        return None
    meta = _QKV_PAD_REGISTRY.get(base)
    if meta is None:
        return None
    return int(meta["proj_in_pad"])


def _patch_attn_meta_from_qkv(layer: "AscendQKVParallelLinear310") -> None:
    if getattr(layer, "_pad_meta_patched", False):
        return
    head_size = int(getattr(layer, "_head_size", 0))
    head_size_pad = int(getattr(layer, "_head_size_pad", head_size))
    if head_size_pad <= head_size:
        layer._pad_meta_patched = True
        return

    def _patch_module(m: nn.Module) -> None:
        if not hasattr(m, "head_size_orig"):
            setattr(m, "head_size_orig", head_size)
        if hasattr(m, "head_size"):
            m.head_size = head_size_pad
        if hasattr(m, "head_dim"):
            m.head_dim = head_size_pad
        if hasattr(m, "hidden_size_per_attention_head"):
            m.hidden_size_per_attention_head = head_size_pad
        if hasattr(m, "projection_size"):
            if hasattr(m, "num_attention_heads_per_partition"):
                m.projection_size = int(m.num_attention_heads_per_partition) * head_size_pad
            elif hasattr(m, "num_heads"):
                m.projection_size = int(m.num_heads) * head_size_pad
        if hasattr(m, "scale"):
            m.scale = 1.0 / math.sqrt(head_size)

        for attr in ("rotary_emb", "rotary_pos_emb", "rotary_embedding"):
            rope = getattr(m, attr, None)
            if rope is None:
                continue
            if hasattr(rope, "head_size"):
                if not hasattr(rope, "head_size_orig"):
                    setattr(rope, "head_size_orig", head_size)
                rope.head_size = head_size_pad

    for ref in gc.get_referrers(layer):
        if not isinstance(ref, nn.Module):
            continue
        for v in ref.__dict__.values():
            if v is layer:
                _patch_module(ref)
                inner = getattr(ref, "attn", None)
                if isinstance(inner, nn.Module):
                    _patch_module(inner)
                inner = getattr(ref, "self_attn", None)
                if isinstance(inner, nn.Module):
                    _patch_module(inner)
                break

    layer._pad_meta_patched = True


def _pad_last_dim(x: torch.Tensor, new_k: int) -> torch.Tensor:
    k = int(x.shape[-1])
    if k == new_k:
        return x
    if k > new_k:
        raise RuntimeError(f"pad expects k<=new_k, got {k}>{new_k}")
    pad = torch.zeros(*x.shape[:-1], new_k - k, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _copy_2d(dst: torch.Tensor, src: torch.Tensor, *, out_off: int, out_real: int, in_real: int) -> None:
    # dst: [out_pad, in_pad], src: [>=out_real, >=in_real]
    if src.ndim != 2:
        raise RuntimeError(f"expected 2D weight, got {src.ndim}D shape={tuple(src.shape)}")
    if src.shape[0] < out_real or src.shape[1] < in_real:
        raise RuntimeError(f"src too small: src={tuple(src.shape)} need=({out_real},{in_real})")
    dst[out_off: out_off + out_real, :in_real].copy_(src[:out_real, :in_real])


def _maybe_tp_slice(w: torch.Tensor, dim: int, *, local: int, rank: int, tp: int) -> torch.Tensor:
    """If w is global on `dim`, slice to local for this rank; else keep as-is."""
    if w.size(dim) == local:
        return w
    if w.size(dim) == local * tp:
        return w.narrow(dim, rank * local, local)
    return w  # caller will validate


def _load_1d_or_scalar_param(
    *, param: Parameter, loaded_weight: torch.Tensor, tp_rank: Optional[int] = None, tp_size: Optional[int] = None
) -> bool:
    """Handle scalar/1D params (bias, scales). Return True if handled."""
    w = loaded_weight.reshape(1) if loaded_weight.ndim == 0 else loaded_weight
    p = param.data

    # scalar
    if p.ndim == 0:
        if w.numel() != 1:
            raise RuntimeError(f"scalar expects 1 elem, got {tuple(w.shape)}")
        p.copy_(w.reshape(()))
        return True

    # 1D
    if p.ndim != 1:
        return False
    if w.ndim != 1:
        return False

    if w.numel() == p.numel():
        p.copy_(w)
        return True

    # allow loading smaller 1D params into padded storage
    if w.numel() < p.numel():
        p.zero_()
        p[: w.numel()].copy_(w)
        return True

    # sometimes bias stored full -> slice per TP
    if tp_rank is not None and tp_size is not None and w.numel() == p.numel() * int(tp_size):
        start = int(tp_rank) * p.numel()
        p.copy_(w.narrow(0, start, p.numel()))
        return True

    # global real -> local padded (slice then pad)
    if tp_rank is not None and tp_size is not None and w.numel() % int(tp_size) == 0:
        local = w.numel() // int(tp_size)
        if local <= p.numel():
            start = int(tp_rank) * int(local)
            p.zero_()
            p[:local].copy_(w.narrow(0, start, int(local)))
            return True

    raise RuntimeError(f"1D param mismatch: param={tuple(p.shape)} loaded={tuple(w.shape)}")


# ----------------------------
# Unquantized method (alloc + optional NZ)
# ----------------------------

class AscendUnquantizedLinearMethod310(UnquantizedLinearMethod):
    """
    Flags set by layer:
      - layer._pad_in  : pad K (weight input dim) to 16
      - layer._pad_out : pad N (weight output dim) to 16 (per-part for merged)
      - layer._qkv_per_head_pad : QKV special per-head pad on OUT
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.pop("weight_loader")

        in_real = int(input_size_per_partition)
        in_pad = _align_up(in_real) if getattr(layer, "_pad_in", False) else in_real
        force_in_pad = getattr(layer, "_force_in_pad", None)
        if force_in_pad is not None:
            in_pad = max(in_pad, int(force_in_pad))

        # ---- QKV special allocation ----
        if getattr(layer, "_qkv_per_head_pad", False):
            parts_real = list(map(int, layer._qkv_parts_real))  # [q,k,v] local real rows
            parts_pad  = list(map(int, layer._qkv_parts_pad))   # [q,k,v] local padded rows

            out_real = sum(parts_real)
            out_pad  = sum(parts_pad)

            weight = Parameter(torch.zeros((out_pad, in_pad), dtype=params_dtype), requires_grad=False)
            layer.register_parameter("weight", weight)

            set_weight_attrs(weight, dict(
                input_dim=1, output_dim=0,
                weight_loader=weight_loader,
                in_real=in_real, in_pad=in_pad,
                out_real=out_real, out_pad=out_pad,
                parts_real=parts_real, parts_pad=parts_pad,
                # QKV meta needed by loader:
                q_heads=int(layer._q_heads),
                kv_heads=int(layer._kv_heads),
                head_size=int(layer._head_size),
                head_size_pad=int(layer._head_size_pad),
                v_head_size=int(layer._v_head_size),
                v_head_size_pad=int(layer._v_head_size_pad),
            ))
            if extra_weight_attrs:
                set_weight_attrs(weight, extra_weight_attrs)
            return

        # ---- generic allocation ----
        parts_real = list(map(int, output_partition_sizes))  # local parts
        if getattr(layer, "_pad_out", False):
            if getattr(layer, "_force_part_align", None) is not None:
                aligns = list(layer._force_part_align)
                parts_pad = [_align_up(s, int(a)) for s, a in zip(parts_real, aligns)]
            else:
                parts_pad = [_align_up(s) for s in parts_real]
        else:
            parts_pad = parts_real

        out_real = sum(parts_real)
        out_pad  = sum(parts_pad)

        weight = Parameter(torch.zeros((out_pad, in_pad), dtype=params_dtype), requires_grad=False)
        layer.register_parameter("weight", weight)

        set_weight_attrs(weight, dict(
            input_dim=1, output_dim=0,
            weight_loader=weight_loader,
            in_real=in_real, in_pad=in_pad,
            out_real=out_real, out_pad=out_pad,
            parts_real=parts_real, parts_pad=parts_pad,
        ))
        if extra_weight_attrs:
            set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if getattr(layer, "_enable_nz", False) and "conv1d" not in getattr(layer, "prefix", ""):
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, _NZ_FORMAT)


# ----------------------------
# LinearBase
# ----------------------------

class AscendLinearBase310(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix
        self.return_bias = return_bias
        self.disable_tp = disable_tp

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = AscendUnquantizedLinearMethod310()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)


# ----------------------------
# ColumnParallel: pad OUT only in weight
# ----------------------------

class AscendColumnParallelLinear310(ColumnParallelLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, gather_output: bool = False,
                 skip_bias_add: bool = False, params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
                 *, return_bias: bool = True, disable_tp: bool = False):
        self.custom_op = None
        self.tp_rank = 0 if disable_tp else get_tensor_model_parallel_rank()
        self.tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()

        self.input_size_per_partition = int(input_size)
        self.output_size_per_partition = divide(int(output_size), int(self.tp_size))

        # merged-column uses self.output_sizes; normal column => one part
        self.output_partition_sizes = [int(self.output_size_per_partition)]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [divide(int(s), int(self.tp_size)) for s in self.output_sizes]

        self._pad_in = False
        self._pad_out = True
        if not hasattr(self, "_keep_out_pad"):
            self._keep_out_pad = False

        AscendLinearBase310.__init__(self, input_size, output_size,
                                     skip_bias_add=skip_bias_add, params_dtype=params_dtype,
                                     quant_config=quant_config, prefix=prefix,
                                     return_bias=return_bias, disable_tp=disable_tp)
        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            out_pad = int(getattr(self.weight, "out_pad", self.output_size_per_partition))
            self.bias = Parameter(torch.empty(out_pad, dtype=self.params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, *_, **__):
        if _load_1d_or_scalar_param(param=param, loaded_weight=loaded_weight,
                                    tp_rank=self.tp_rank, tp_size=self.tp_size):
            return

        w = loaded_weight
        if getattr(param, "is_gguf_weight", False) and isinstance(param, UninitializedParameter):
            param.materialize(tuple(param.data.shape), dtype=w.dtype)

        out_dim = int(getattr(param, "output_dim", 0))
        in_real = int(getattr(param, "in_real"))
        out_real = int(getattr(param, "out_real"))

        if w.ndim != 2:
            raise RuntimeError("Column weight_loader expects 2D")

        w = _maybe_tp_slice(w, out_dim, local=out_real, rank=int(self.tp_rank), tp=int(self.tp_size))
        if w.size(out_dim) != out_real:
            raise RuntimeError(f"[{self.prefix}] Column out mismatch: got={w.size(out_dim)} expect={out_real}")

        _copy_2d(param.data, w, out_off=0, out_real=out_real, in_real=in_real)

    def forward(self, input_: torch.Tensor):
        _patch_attn_meta_from_qkv(self)
        bias = None if self.skip_bias_add else self.bias
        out = self.quant_method.apply(self, input_, bias)
        out_bias = self.bias if self.skip_bias_add else None

        out_real = int(getattr(self.weight, "out_real", out.shape[-1]))
        if (not getattr(self, "_keep_out_pad", False)) and out.shape[-1] != out_real:
            out = out[..., :out_real].contiguous()
            if out_bias is not None and out_bias.numel() != out_real:
                out_bias = out_bias[:out_real].contiguous()

        if not self.return_bias:
            return out
        return out, out_bias


# ----------------------------
# RowParallel: pad IN only in weight, forward pads input
# ----------------------------

class AscendRowParallelLinear310(RowParallelLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, input_is_parallel: bool = True,
                 skip_bias_add: bool = False, params_dtype: Optional[torch.dtype] = None, reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
                 *, return_bias: bool = True, disable_tp: bool = False):
        self.custom_op = None
        self.tp_rank = 0 if disable_tp else get_tensor_model_parallel_rank()
        self.tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()

        self.input_size_per_partition = divide(int(input_size), int(self.tp_size))
        self.output_size_per_partition = int(output_size)
        self.output_partition_sizes = [int(output_size)]

        self._force_in_pad = _get_force_in_pad_for_proj(prefix)
        self._pad_in = True
        self._pad_out = False

        AscendLinearBase310.__init__(self, input_size, output_size,
                                     skip_bias_add=skip_bias_add, params_dtype=params_dtype,
                                     quant_config=quant_config, prefix=prefix,
                                     return_bias=return_bias, disable_tp=disable_tp)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        in_real = int(getattr(self.weight, "in_real", self.input_size_per_partition))
        in_pad = int(getattr(self.weight, "in_pad", in_real))
        if in_pad > in_real:
            # keep comm shapes aligned with padded K
            self.input_size_per_partition = in_pad

        if bias:
            self.bias = Parameter(torch.empty(int(self.output_size), dtype=self.params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, *_, **__):
        if _load_1d_or_scalar_param(param=param, loaded_weight=loaded_weight):
            return

        w = loaded_weight
        if w.ndim != 2:
            raise RuntimeError("Row weight_loader expects 2D")

        in_dim = int(getattr(param, "input_dim", 1))
        in_real = int(getattr(param, "in_real"))
        out_real = int(getattr(param, "out_real"))

        w = _maybe_tp_slice(w, in_dim, local=in_real, rank=int(self.tp_rank), tp=int(self.tp_size))
        if w.size(in_dim) != in_real:
            raise RuntimeError(f"[{self.prefix}] Row in mismatch: got={w.size(in_dim)} expect={in_real}")

        _copy_2d(param.data, w, out_off=0, out_real=out_real, in_real=in_real)

    def forward(self, input_: torch.Tensor, **_):
        x = input_
        if not self.input_is_parallel:
            x = torch.chunk(x, int(self.tp_size), dim=-1)[int(self.tp_rank)].contiguous()

        in_real = int(getattr(self.weight, "in_real", x.shape[-1]))
        in_pad = int(getattr(self.weight, "in_pad", in_real))
        if x.shape[-1] == in_real and in_pad > in_real:
            x = _pad_last_dim(x, in_pad)

        bias = None if (int(self.tp_rank) > 0 or self.skip_bias_add) else self.bias
        out = self.quant_method.apply(self, x, bias)

        if self.reduce_results and int(self.tp_size) > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            out = tensor_model_parallel_all_reduce(out)

        if not self.return_bias:
            return out
        return out, (self.bias if self.skip_bias_add else None)


# ----------------------------
# MergedColumnParallel (gate_up): per-part OUT padding, only loader differs
# ----------------------------

class AscendMergedColumnParallelLinear310(MergedColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: List[int], **kwargs):
        self.output_sizes = list(map(int, output_sizes))
        self._force_part_align = [_ALIGN] * len(self.output_sizes)
        self._keep_out_pad = os.environ.get("VLLM_ASCEND_KEEP_QKV_PAD", "1") != "0"
        AscendColumnParallelLinear310.__init__(
            self,
            input_size=input_size,
            output_size=sum(self.output_sizes),
            **kwargs,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[int] = None, *_, **__):
        if _load_1d_or_scalar_param(param=param, loaded_weight=loaded_weight,
                                    tp_rank=self.tp_rank, tp_size=self.tp_size):
            return

        w = loaded_weight
        if w.ndim != 2:
            raise RuntimeError("MergedColumn weight_loader expects 2D")

        out_dim = int(getattr(param, "output_dim", 0))
        in_real = int(getattr(param, "in_real"))
        parts_pad = list(map(int, getattr(param, "parts_pad")))

        # If caller passes full concatenated [gate;up], split then recurse.
        if loaded_shard_id is None:
            cur = 0
            for i, part_global in enumerate(self.output_sizes):
                seg = w.narrow(out_dim, cur, int(part_global))
                cur += int(part_global)
                self.weight_loader(param, seg, i)
            return

        sid = int(loaded_shard_id)
        part_global = int(self.output_sizes[sid])
        part_local = part_global // int(self.tp_size)

        seg = _maybe_tp_slice(w, out_dim, local=int(part_local), rank=int(self.tp_rank), tp=int(self.tp_size))
        if seg.size(out_dim) != int(part_local):
            raise RuntimeError(f"[{self.prefix}] gate_up part{sid} mismatch")

        out_off = sum(parts_pad[:sid])
        _copy_2d(param.data, seg, out_off=int(out_off), out_real=int(part_local), in_real=in_real)


# ----------------------------
# QKVParallelLinear: per-head OUT pad + scatter copy
# ----------------------------

class AscendQKVParallelLinear310(QKVParallelLinear):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
                 total_num_kv_heads: Optional[int] = None, bias: bool = True,
                 skip_bias_add: bool = False, params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
                 *, return_bias: bool = True, disable_tp: bool = False,
                 v_head_size: Optional[int] = None):
        # mimic vllm QKV init but route to 310P column impl
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        # per-head output padding (QKV)
        self._pad_in = False
        self._pad_out = True
        self._keep_out_pad = True
        self._qkv_per_head_pad = True
        self._q_heads = self.num_heads
        self._kv_heads = self.num_kv_heads
        self._head_size = self.head_size
        self._v_head_size = self.v_head_size
        self._head_size_pad = _align_up(self.head_size)
        self._v_head_size_pad = _align_up(self.v_head_size)
        self._qkv_parts_real = [
            self.num_heads * self.head_size,
            self.num_kv_heads * self.head_size,
            self.num_kv_heads * self.v_head_size,
        ]
        self._qkv_parts_pad = [
            self.num_heads * self._head_size_pad,
            self.num_kv_heads * self._head_size_pad,
            self.num_kv_heads * self._v_head_size_pad,
        ]
        _register_qkv_pad_meta(
            prefix,
            num_heads=self.num_heads,
            head_size=self.head_size,
            head_size_pad=self._head_size_pad,
        )
        self._pad_meta_patched = False

        AscendColumnParallelLinear310.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _kv_shard_rank(self) -> int:
        replicas = int(getattr(self, "num_kv_head_replicas", 1))
        if replicas <= 1:
            return int(self.tp_rank)
        return int(self.tp_rank) // replicas

    def _scatter_per_head(self, *, dst: torch.Tensor, dst_off: int, heads: int,
                          head_real: int, head_pad: int, src: torch.Tensor, in_real: int) -> None:
        part = dst.narrow(0, dst_off, heads * head_pad).view(heads, head_pad, -1)
        src = src[:, :in_real].contiguous().view(heads, head_real, in_real)
        part[:, :head_real, :in_real].copy_(src)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[str] = None, *_, **__):
        if _load_1d_or_scalar_param(param=param, loaded_weight=loaded_weight,
                                    tp_rank=self.tp_rank, tp_size=self.tp_size):
            return

        w = loaded_weight
        if w.ndim != 2:
            raise RuntimeError("QKV weight_loader expects 2D")

        out_dim = int(getattr(param, "output_dim", 0))
        in_real = int(getattr(param, "in_real"))

        q_pad, k_pad, v_pad = map(int, getattr(param, "parts_pad"))
        q_off, k_off, v_off = 0, q_pad, q_pad + k_pad

        q_heads = int(getattr(param, "q_heads"))
        kv_heads = int(getattr(param, "kv_heads"))
        head_real = int(getattr(param, "head_size"))
        head_pad  = int(getattr(param, "head_size_pad"))
        v_head_real = int(getattr(param, "v_head_size"))
        v_head_pad  = int(getattr(param, "v_head_size_pad"))

        def narrow_tp(seg: torch.Tensor, shard: str) -> torch.Tensor:
            if shard == "q":
                local = q_heads * head_real
                global_ = int(self.total_num_heads * self.head_size)
                rank = int(self.tp_rank)
            elif shard == "k":
                local = kv_heads * head_real
                global_ = int(self.total_num_kv_heads * self.head_size)
                rank = self._kv_shard_rank()
            else:  # "v"
                local = kv_heads * v_head_real
                global_ = int(self.total_num_kv_heads * self.v_head_size)
                rank = self._kv_shard_rank()

            if seg.size(out_dim) == local:
                return seg
            if seg.size(out_dim) == global_:
                return seg.narrow(out_dim, int(rank) * int(local), int(local))
            raise RuntimeError(f"[{self.prefix}] QKV shard {shard} mismatch")

        # fused on disk: [Q_global; K_global; V_global]
        if loaded_shard_id is None:
            qg, kg, vg = self.output_sizes
            if w.size(out_dim) != qg + kg + vg:
                raise RuntimeError(f"[{self.prefix}] fused QKV rows mismatch")

            q_full = w.narrow(out_dim, 0, qg)
            k_full = w.narrow(out_dim, qg, kg)
            v_full = w.narrow(out_dim, qg + kg, vg)

            q_seg = narrow_tp(q_full, "q")
            k_seg = narrow_tp(k_full, "k")
            v_seg = narrow_tp(v_full, "v")

            self._scatter_per_head(dst=param.data, dst_off=q_off, heads=q_heads,
                                   head_real=head_real, head_pad=head_pad, src=q_seg, in_real=in_real)
            self._scatter_per_head(dst=param.data, dst_off=k_off, heads=kv_heads,
                                   head_real=head_real, head_pad=head_pad, src=k_seg, in_real=in_real)
            self._scatter_per_head(dst=param.data, dst_off=v_off, heads=kv_heads,
                                   head_real=v_head_real, head_pad=v_head_pad, src=v_seg, in_real=in_real)
            return

        sid = str(loaded_shard_id)
        seg = narrow_tp(w, sid)
        if sid == "q":
            self._scatter_per_head(dst=param.data, dst_off=q_off, heads=q_heads,
                                   head_real=head_real, head_pad=head_pad, src=seg, in_real=in_real)
        elif sid == "k":
            self._scatter_per_head(dst=param.data, dst_off=k_off, heads=kv_heads,
                                   head_real=head_real, head_pad=head_pad, src=seg, in_real=in_real)
        elif sid == "v":
            self._scatter_per_head(dst=param.data, dst_off=v_off, heads=kv_heads,
                                   head_real=v_head_real, head_pad=v_head_pad, src=seg, in_real=in_real)
        else:
            raise RuntimeError("loaded_shard_id must be q/k/v")

    def forward(self, input_: torch.Tensor):
        _patch_attn_meta_from_qkv(self)
        bias = None if self.skip_bias_add else self.bias
        out = self.quant_method.apply(self, input_, bias)
        out_bias = self.bias if self.skip_bias_add else None

        out_real = int(getattr(self.weight, "out_real", out.shape[-1]))
        if (not getattr(self, "_keep_out_pad", False)) and out.shape[-1] != out_real:
            out = out[..., :out_real].contiguous()
            if out_bias is not None and out_bias.numel() != out_real:
                out_bias = out_bias[:out_real].contiguous()

        if not self.return_bias:
            return out
        return out, out_bias


# ----------------------------
# Replicated
# ----------------------------

class AscendReplicatedLinear310(ReplicatedLinear):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        self.custom_op = None

        self.output_partition_sizes = [int(output_size)]
        self._pad_in = False
        self._pad_out = False

        AscendLinearBase310.__init__(self, input_size, output_size, **kwargs)

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=int(input_size),
            output_partition_sizes=self.output_partition_sizes,
            input_size=int(input_size),
            output_size=int(output_size),
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if kwargs.get("bias", True):
            self.bias = Parameter(torch.empty(int(output_size), dtype=self.params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, *_, **__):
        if _load_1d_or_scalar_param(param=param, loaded_weight=loaded_weight):
            return
        if loaded_weight.ndim != 2:
            raise RuntimeError("Replicated weight_loader expects 2D")
        in_real = int(getattr(param, "in_real"))
        out_real = int(getattr(param, "out_real"))
        _copy_2d(param.data, loaded_weight, out_off=0, out_real=out_real, in_real=in_real)

    def forward(self, input_: torch.Tensor):
        in_real = int(getattr(self.weight, "in_real", input_.shape[-1]))
        in_pad  = int(getattr(self.weight, "in_pad", in_real))
        x = _pad_last_dim(input_, in_pad) if (in_pad > in_real and input_.shape[-1] == in_real) else input_

        bias = None if self.skip_bias_add else self.bias
        out = self.quant_method.apply(self, x, bias)

        out_real = int(getattr(self.weight, "out_real", out.shape[-1]))
        if out.shape[-1] != out_real:
            out = out[..., :out_real].contiguous()

        if not self.return_bias:
            return out
        return out, (self.bias if self.skip_bias_add else None)


__all__ = [
    "AscendLinearBase310",
    "AscendUnquantizedLinearMethod310",
    "AscendColumnParallelLinear310",
    "AscendRowParallelLinear310",
    "AscendMergedColumnParallelLinear310",
    "AscendQKVParallelLinear310",
    "AscendReplicatedLinear310",
]
