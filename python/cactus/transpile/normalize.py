from __future__ import annotations

from typing import Any

import torch

from cactus.transpile.aten_ops import canonical_ir_op


def normalize_target(target: Any) -> str:
    return canonical_ir_op(target)


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
