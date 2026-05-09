from __future__ import annotations

from typing import Any

import torch


def normalize_target(target: str) -> str:
    if target.startswith("aten.mul_"):
        return "multiply_inplace"
    if target.startswith("aten._assert_tensor_metadata"):
        return "identity"
    if target.startswith("aten.lift_fresh_copy"):
        return "identity"
    if target.startswith("aten.clone"):
        return "identity"
    if target.startswith("aten.alias"):
        return "identity"
    if target.startswith("aten.detach"):
        return "identity"
    if target.startswith("aten.arange.start"):
        return "arange"
    if target.startswith("aten.arange"):
        return "arange"
    if target.startswith("aten.add"):
        return "add"
    if target.startswith("aten.sub"):
        return "subtract"
    if target.startswith("aten.mul"):
        return "multiply"
    if target.startswith("aten.div"):
        return "divide"
    if target.startswith("aten.to.dtype"):
        return "precision_cast"
    if target.startswith("aten.type_as"):
        return "type_as"
    if target.startswith("aten.neg"):
        return "negate"
    if target.startswith("aten.ne.") or target == "aten.ne":
        return "not_equal"
    if target.startswith("aten.abs"):
        return "abs"
    if target.startswith("aten.expand"):
        return "expand"
    if target.startswith("aten.exp"):
        return "scalar_exp"
    if target.startswith("aten.sqrt"):
        return "scalar_sqrt"
    if target.startswith("aten.log"):
        return "scalar_log"
    if target.startswith("aten.rsqrt"):
        return "rsqrt"
    if target.startswith("aten.cos"):
        return "cos"
    if target.startswith("aten.sin"):
        return "sin"
    if target in ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"):
        return "reshape"
    if target == "aten.flatten.using_ints":
        return "flatten"
    if target.startswith("aten.unsqueeze"):
        return "unsqueeze"
    if target == "aten.t.default" or target.startswith("aten.transpose"):
        return "transpose"
    if target.startswith("aten.permute"):
        return "permute"
    if target.startswith("aten.contiguous"):
        return "contiguous"
    if target.startswith("aten.mm") or target.startswith("aten.matmul") or target.startswith("aten.bmm"):
        return "matmul"
    if target.startswith("aten.linear"):
        return "linear"
    if target.startswith("aten.addmm"):
        return "addmm"
    if target.startswith("aten.relu"):
        return "relu"
    if target.startswith("aten.silu"):
        return "silu"
    if target == "aten.gelu.erf":
        return "gelu_erf"
    if target.startswith("aten.gelu"):
        return "gelu"
    if target.startswith("aten.sigmoid"):
        return "sigmoid"
    if target.startswith("aten.softplus"):
        return "softplus"
    if target.startswith("aten.tanh"):
        return "tanh"
    if target.startswith("aten.softmax"):
        return "softmax"
    if target.startswith("aten.scaled_dot_product_attention"):
        return "scaled_dot_product_attention"
    if target.startswith("aten.sum"):
        return "sum"
    if target.startswith("aten.mean"):
        return "mean"
    if target.startswith("aten.var"):
        return "variance"
    if target.startswith("aten.min") or target.startswith("aten.amin"):
        return "min"
    if target.startswith("aten.max") or target.startswith("aten.amax"):
        return "max"
    if target.startswith("aten.cat"):
        return "cat"
    if target.startswith("aten.split_with_sizes"):
        return "split_with_sizes"
    if target.startswith("aten.chunk"):
        return "chunk"
    if target.startswith("aten.ones"):
        return "ones"
    if target.startswith("aten.slice"):
        return "slice"
    if target.startswith("aten.select"):
        return "index"
    if target.startswith("aten.gather"):
        return "gather"
    if target.startswith("aten.pad"):
        return "pad"
    if target.startswith("aten.embedding"):
        return "embedding"
    if target.startswith("aten.conv1d"):
        return "conv1d"
    if target.startswith("aten.pow"):
        return "pow"
    if target.startswith("aten.layer_norm"):
        return "layer_norm"
    if target.startswith("aten.native_layer_norm"):
        return "layer_norm"
    if target.startswith("aten.group_norm"):
        return "group_norm"
    if target.startswith("aten.batch_norm") or target.startswith("aten._native_batch_norm_legit_no_training"):
        return "batch_norm"
    if target.startswith("aten.rms_norm"):
        return "rms_norm"
    if target == "<built-in function getitem>":
        return "getitem"
    return target


def dtype_to_ir(dtype: Any | None) -> str | None:
    if dtype is None:
        return None

    dtype_map = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float64: "fp64",
        torch.bfloat16: "bf16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    return dtype_map.get(dtype, str(dtype))
