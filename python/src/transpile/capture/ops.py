from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSchema:
    name: str
    num_inputs: int | None = None
    attrs: tuple[str, ...] = ()
    backend_op: str | None = None
    is_layout: bool = False
    is_reduction: bool = False
    is_fusible: bool = False


# Small canonical op registry for the transpiler.
# This is metadata for canonical IR ops, not another IR layer.

OPS: tuple[OpSchema, ...] = (
    OpSchema("add", num_inputs=2, backend_op="add", is_fusible=True),
    OpSchema("add_clipped", num_inputs=2, backend_op="add_clipped", is_fusible=True),
    OpSchema("subtract", num_inputs=2, backend_op="subtract", is_fusible=True),
    OpSchema("multiply", num_inputs=2, backend_op="multiply", is_fusible=True),
    OpSchema("divide", num_inputs=2, backend_op="divide", is_fusible=True),
    OpSchema("not_equal", num_inputs=2, backend_op="not_equal"),
    OpSchema("abs", num_inputs=1, backend_op="abs", is_fusible=True),
    OpSchema("pow", num_inputs=1, attrs=("exponent",), backend_op="pow"),
    OpSchema("negate", num_inputs=1, is_fusible=True),
    OpSchema("scalar_add", num_inputs=1, attrs=("value",), backend_op="scalar_add", is_fusible=True),
    OpSchema("scalar_subtract", num_inputs=1, attrs=("value",), backend_op="scalar_subtract", is_fusible=True),
    OpSchema("scalar_subtract_reverse", num_inputs=1, attrs=("value",)),
    OpSchema("scalar_multiply", num_inputs=1, attrs=("value",), backend_op="scalar_multiply", is_fusible=True),
    OpSchema("scalar_divide", num_inputs=1, attrs=("value",), backend_op="scalar_divide", is_fusible=True),
    OpSchema("scalar_divide_reverse", num_inputs=1, attrs=("value",)),
    OpSchema("scalar_not_equal", num_inputs=1, attrs=("value",), backend_op="scalar_not_equal"),
    OpSchema("scalar_exp", num_inputs=1, backend_op="scalar_exp", is_fusible=True),
    OpSchema("scalar_sqrt", num_inputs=1, backend_op="scalar_sqrt", is_fusible=True),
    OpSchema("scalar_log", num_inputs=1, backend_op="scalar_log", is_fusible=True),
    OpSchema("scalar_cos", num_inputs=1, backend_op="scalar_cos"),
    OpSchema("scalar_sin", num_inputs=1, backend_op="scalar_sin"),
    OpSchema("precision_cast", num_inputs=1, attrs=("dtype",), backend_op="precision_cast"),
    OpSchema("view", num_inputs=1, attrs=("shape",), backend_op="reshape", is_layout=True),
    OpSchema("permute", num_inputs=1, attrs=("permutation",), backend_op="permute", is_layout=True),
    OpSchema("slice", num_inputs=1, attrs=("axis", "start", "end", "step"), backend_op="slice"),
    OpSchema("split_with_sizes", num_inputs=1, attrs=("sizes", "axis")),
    OpSchema("chunk", num_inputs=1, attrs=("chunks", "axis")),
    OpSchema("ones", num_inputs=0, attrs=("shape", "dtype")),
    OpSchema("index", num_inputs=1, attrs=("axis", "index_value"), backend_op="index"),
    OpSchema("cat", num_inputs=None, attrs=("axis",), backend_op="cat"),
    OpSchema("matmul", num_inputs=2, backend_op="matmul"),
    OpSchema("linear", num_inputs=None, attrs=("has_bias",), backend_op="matmul"),
    OpSchema("addmm", num_inputs=3),
    OpSchema("gather", num_inputs=2, backend_op="gather"),
    OpSchema("embedding", num_inputs=2, backend_op="embedding_from_tensor"),
    OpSchema("sum", num_inputs=1, attrs=("axis", "keepdim"), backend_op="sum", is_reduction=True),
    OpSchema("mean", num_inputs=1, attrs=("axis", "keepdim"), backend_op="mean", is_reduction=True),
    OpSchema("variance", num_inputs=1, attrs=("axis", "keepdim"), backend_op="variance", is_reduction=True),
    OpSchema("min", num_inputs=1, attrs=("axis", "keepdim"), backend_op="min", is_reduction=True),
    OpSchema("max", num_inputs=1, attrs=("axis", "keepdim"), backend_op="max", is_reduction=True),
    OpSchema("relu", num_inputs=1, backend_op="relu", is_fusible=True),
    OpSchema("silu", num_inputs=1, backend_op="silu", is_fusible=True),
    OpSchema("gelu", num_inputs=1, backend_op="gelu", is_fusible=True),
    OpSchema("gelu_erf", num_inputs=1, backend_op="gelu_erf", is_fusible=True),
    OpSchema("sigmoid", num_inputs=1, backend_op="sigmoid", is_fusible=True),
    OpSchema("tanh", num_inputs=1, backend_op="tanh", is_fusible=True),
    OpSchema("softplus", num_inputs=1),
    OpSchema("softmax", num_inputs=1, attrs=("axis",), backend_op="softmax"),
    OpSchema("layer_norm", num_inputs=None, attrs=("eps",), backend_op="layer_norm"),
    OpSchema("group_norm", num_inputs=3, attrs=("num_groups", "eps"), backend_op="group_norm"),
    OpSchema("batch_norm", num_inputs=5, attrs=("axis", "eps"), backend_op="batch_norm"),
    OpSchema("conv1d", num_inputs=2, attrs=("stride", "padding", "dilation", "groups"), backend_op="conv1d"),
    OpSchema("pad", num_inputs=1, attrs=("pads", "value", "mode")),

    # CUSTOM / FUSED OPS
    OpSchema("rms_norm", num_inputs=2, attrs=("eps",), backend_op="rms_norm"),
    OpSchema("rope", num_inputs=1, attrs=("theta", "position_offset"), backend_op="rope"),
    OpSchema(
        "attention",
        num_inputs=3,
        attrs=("scale", "is_causal", "window_size", "mask", "additive_mask"),
        backend_op="attention",
    ),
    OpSchema(
        "attention_block",
        num_inputs=None,
        attrs=("scale", "is_causal", "window_size", "has_gate", "has_bias", "attention_output_shape"),
        is_fusible=True,
    ),
    OpSchema(
        "gated_deltanet_prefill",
        num_inputs=None,
        attrs=(
            "num_k_heads",
            "num_v_heads",
            "key_dim",
            "value_dim",
            "eps",
            "chunk_size",
            "has_z",
            "has_dt_bias",
            "has_a_log",
            "has_conv",
        ),
        is_fusible=True,
    ),
    OpSchema(
        "gated_deltanet_decode",
        num_inputs=None,
        attrs=(
            "num_k_heads",
            "num_v_heads",
            "key_dim",
            "value_dim",
            "eps",
            "has_z",
            "has_dt_bias",
            "has_a_log",
            "has_conv",
        ),
        is_fusible=True,
    ),
)


ALL_OPS: dict[str, OpSchema] = {op.name: op for op in OPS}


# Import/capture spellings reduced into the canonical registry above.
ALIASES: dict[str, str] = {
    "reshape": "view",
    "flatten": "view",
    "unsqueeze": "view",
    "expand": "view",
    "contiguous": "view",
    "identity": "view",
    "transpose": "permute",
    "concat": "cat",
    "type_as": "precision_cast",
    "scaled_dot_product_attention": "attention",
    "cos": "scalar_cos",
    "sin": "scalar_sin",
}


PRECISION_OP = "precision_cast"
SUPPORTED_PRECISIONS: tuple[str, ...] = ("int4", "int8", "fp16", "fp32")


def canonicalize_op(name: str) -> str:
    return ALIASES.get(name, name)


def get_op(name: str) -> OpSchema:
    return ALL_OPS[canonicalize_op(name)]


def has_op(name: str) -> bool:
    return canonicalize_op(name) in ALL_OPS
