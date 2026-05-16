from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AtenOpRule:
    torch_prefix: str
    ir_op: str


def canonical_torch_op(target: Any) -> str:
    """Return a stable torch/export op spelling independent of Python repr noise."""

    if isinstance(target, str):
        value = target
    else:
        rendered = str(target).strip()
        if rendered.startswith("aten.") or rendered.startswith("<built-in function "):
            value = rendered
        else:
            name = getattr(target, "__name__", None)
            overload = getattr(target, "overloadpacket", None)
            overload_name = getattr(target, "_overloadname", None)
            packet_name = getattr(overload, "__name__", None)
            if packet_name and overload_name:
                value = f"{packet_name}.{overload_name}"
            elif packet_name:
                value = str(packet_name)
            elif name:
                value = str(name)
            else:
                value = rendered

    value = value.strip()
    if value.startswith("torch.ops."):
        value = value[len("torch.ops.") :]
    return value


_EXACT_RULES: dict[str, str] = {
    "<built-in function getitem>": "getitem",
    "aten.view.default": "reshape",
    "aten.reshape.default": "reshape",
    "aten._unsafe_view.default": "reshape",
    "aten.flatten.using_ints": "flatten",
    "aten.t.default": "transpose",
    "aten.gelu.erf": "gelu_erf",
    # Keep the legacy importer key until the importer itself is renamed.
    "aten.diff.default": "aten.diff.default",
}


_PREFIX_RULES: tuple[AtenOpRule, ...] = (
    AtenOpRule("aten.mul_", "multiply_inplace"),
    AtenOpRule("aten._assert_tensor_metadata", "identity"),
    AtenOpRule("aten.lift_fresh_copy", "identity"),
    AtenOpRule("aten.clone", "identity"),
    AtenOpRule("aten.alias", "identity"),
    AtenOpRule("aten.detach", "identity"),
    AtenOpRule("aten.dropout", "identity"),
    AtenOpRule("aten.native_dropout", "identity"),
    AtenOpRule("aten.arange.start", "arange"),
    AtenOpRule("aten.arange", "arange"),
    AtenOpRule("aten.add", "add"),
    AtenOpRule("aten.sub", "subtract"),
    AtenOpRule("aten.mul", "multiply"),
    AtenOpRule("aten.floor_divide", "floor_divide"),
    AtenOpRule("aten.div", "divide"),
    AtenOpRule("aten.to.dtype", "precision_cast"),
    AtenOpRule("aten.to.device", "identity"),
    AtenOpRule("aten.type_as", "type_as"),
    AtenOpRule("aten.neg", "negate"),
    AtenOpRule("aten.ne.", "not_equal"),
    AtenOpRule("aten.eq.", "equal"),
    AtenOpRule("aten.gt.", "greater"),
    AtenOpRule("aten.ge.", "greater_equal"),
    AtenOpRule("aten.lt.", "less"),
    AtenOpRule("aten.le.", "less_equal"),
    AtenOpRule("aten.logical_or", "logical_or"),
    AtenOpRule("aten.__or__", "logical_or"),
    AtenOpRule("aten.logical_and", "logical_and"),
    AtenOpRule("aten.__and__", "logical_and"),
    AtenOpRule("aten.logical_not", "logical_not"),
    AtenOpRule("aten.bitwise_not", "logical_not"),
    AtenOpRule("aten.abs", "abs"),
    AtenOpRule("aten.clamp", "clamp"),
    AtenOpRule("aten.where", "where"),
    AtenOpRule("aten.masked_scatter", "masked_scatter"),
    AtenOpRule("aten.masked_fill", "masked_fill"),
    AtenOpRule("aten.expand", "expand"),
    AtenOpRule("aten.repeat", "repeat"),
    AtenOpRule("aten.exp", "scalar_exp"),
    AtenOpRule("aten.sqrt", "scalar_sqrt"),
    AtenOpRule("aten.reciprocal", "reciprocal"),
    AtenOpRule("aten.log", "scalar_log"),
    AtenOpRule("aten.rsqrt", "rsqrt"),
    AtenOpRule("aten.cos", "cos"),
    AtenOpRule("aten.sin", "sin"),
    AtenOpRule("aten.squeeze", "squeeze"),
    AtenOpRule("aten.unsqueeze", "unsqueeze"),
    AtenOpRule("aten.transpose", "transpose"),
    AtenOpRule("aten.permute", "permute"),
    AtenOpRule("aten.numpy_T", "numpy_T"),
    AtenOpRule("aten.movedim", "movedim"),
    AtenOpRule("aten.contiguous", "contiguous"),
    AtenOpRule("aten.mm", "matmul"),
    AtenOpRule("aten.matmul", "matmul"),
    AtenOpRule("aten.bmm", "matmul"),
    AtenOpRule("aten.linear", "linear"),
    AtenOpRule("aten.addmm", "addmm"),
    AtenOpRule("aten.relu", "relu"),
    AtenOpRule("aten.silu", "silu"),
    AtenOpRule("aten.gelu", "gelu"),
    AtenOpRule("aten.sigmoid", "sigmoid"),
    AtenOpRule("aten.glu", "glu"),
    AtenOpRule("aten.softplus", "softplus"),
    AtenOpRule("aten.tanh", "tanh"),
    AtenOpRule("aten.softmax", "softmax"),
    AtenOpRule("aten.scaled_dot_product_attention", "scaled_dot_product_attention"),
    AtenOpRule("aten.sum", "sum"),
    AtenOpRule("aten.mean", "mean"),
    AtenOpRule("aten.all", "min"),
    AtenOpRule("aten.var", "variance"),
    AtenOpRule("aten.min", "min"),
    AtenOpRule("aten.amin", "min"),
    AtenOpRule("aten.max", "max"),
    AtenOpRule("aten.amax", "max"),
    AtenOpRule("aten.cat", "cat"),
    AtenOpRule("aten.split_with_sizes", "split_with_sizes"),
    AtenOpRule("aten.chunk", "chunk"),
    AtenOpRule("aten.ones", "ones"),
    AtenOpRule("aten.slice", "slice"),
    AtenOpRule("aten.select", "index"),
    AtenOpRule("aten.gather", "gather"),
    AtenOpRule("aten.pad", "pad"),
    AtenOpRule("aten.one_hot", "one_hot"),
    AtenOpRule("aten.tril", "tril"),
    AtenOpRule("aten.unfold", "unfold"),
    AtenOpRule("aten.embedding", "embedding"),
    AtenOpRule("aten.conv1d", "conv1d"),
    AtenOpRule("aten.conv2d", "conv2d"),
    AtenOpRule("aten.pow", "pow"),
    AtenOpRule("aten.layer_norm", "layer_norm"),
    AtenOpRule("aten.native_layer_norm", "layer_norm"),
    AtenOpRule("aten.group_norm", "group_norm"),
    AtenOpRule("aten.batch_norm", "batch_norm"),
    AtenOpRule("aten._native_batch_norm_legit_no_training", "batch_norm"),
    AtenOpRule("aten.rms_norm", "rms_norm"),
)


def canonical_ir_op(target: Any) -> str:
    torch_op = canonical_torch_op(target)
    exact = _EXACT_RULES.get(torch_op)
    if exact is not None:
        return exact
    if torch_op in {"aten.ne", "aten.eq", "aten.gt", "aten.ge", "aten.lt", "aten.le"}:
        return {
            "aten.ne": "not_equal",
            "aten.eq": "equal",
            "aten.gt": "greater",
            "aten.ge": "greater_equal",
            "aten.lt": "less",
            "aten.le": "less_equal",
        }[torch_op]
    for rule in _PREFIX_RULES:
        if torch_op.startswith(rule.torch_prefix):
            return rule.ir_op
    return torch_op


def is_aten_op(target: Any) -> bool:
    return canonical_torch_op(target).startswith("aten.")
