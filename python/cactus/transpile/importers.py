from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace
from typing import Any

import torch

from cactus.transpile.aten_ops import canonical_torch_op
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.normalize import dtype_to_ir
from cactus.transpile.normalize import normalize_target


class UnsupportedImportError(NotImplementedError):
    pass


@dataclass
class ImportContext:
    strict: bool = True
    alias_values: dict[str, str] = field(default_factory=dict)
    tuple_aliases: dict[str, list[str]] = field(default_factory=dict)
    local_alias_stack: list[dict[str, str]] = field(default_factory=list)
    node_stack: list[str] = field(default_factory=list)
    transpile_metadata: dict[str, object] = field(default_factory=dict)

    def push_local_aliases(self, aliases: dict[str, str]) -> None:
        self.local_alias_stack.append(aliases)

    def pop_local_aliases(self) -> None:
        if not self.local_alias_stack:
            raise RuntimeError("local alias stack underflow")
        self.local_alias_stack.pop()

    def resolve_value_id(self, node: Any) -> str:
        if is_fx_node(node):
            name = getattr(node, "name", "")
            for aliases in reversed(self.local_alias_stack):
                if name in aliases:
                    return aliases[name]
            resolved = self.alias_values.get(name)
            if resolved is not None:
                return resolved
            return f"v_{name}"
        raise TypeError(f"cannot resolve value id for non-FX node: {type(node).__name__}")

    def fail(self, message: str) -> None:
        if self.node_stack:
            message = f"{message} [context: {' > '.join(self.node_stack)}]"
        if self.strict:
            raise UnsupportedImportError(message)
        raise UnsupportedImportError(f"{message} [non-strict import has no generic fallback]")

    def push_node(self, description: str) -> None:
        self.node_stack.append(description)

    def pop_node(self) -> None:
        if self.node_stack:
            self.node_stack.pop()


def value_id(node: Any, ctx: ImportContext) -> str:
    return ctx.resolve_value_id(node)


def node_id(node: Any) -> str:
    return f"n_{node.name}"


def is_fx_node(value: Any) -> bool:
    return hasattr(value, "name") and hasattr(value, "op")


def extract_input_ids(arg: Any, ctx: ImportContext) -> list[str]:
    if is_fx_node(arg):
        return [value_id(arg, ctx)]
    if isinstance(arg, (tuple, list)):
        ids: list[str] = []
        for item in arg:
            ids.extend(extract_input_ids(item, ctx))
        return ids
    if isinstance(arg, dict):
        ids: list[str] = []
        for item in arg.values():
            ids.extend(extract_input_ids(item, ctx))
        return ids
    return []


def extract_literals(value: Any) -> Any:
    if is_fx_node(value):
        meta = getattr(value, "meta", {}) or {}
        meta_value = meta.get("val")
        if meta_value is not None and not isinstance(meta_value, torch.Tensor):
            return extract_literals(meta_value)
        return None
    if isinstance(value, torch.Size):
        return tuple(extract_literals(v) for v in value)
    if isinstance(value, (tuple, list)):
        return type(value)(extract_literals(v) for v in value)
    if isinstance(value, dict):
        return {k: extract_literals(v) for k, v in value.items()}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def make_prefixed_node(node: Any, prefix: str) -> Any:
    return SimpleNamespace(
        name=f"{prefix}{node.name}",
        op=node.op,
        target=node.target,
        args=node.args,
        kwargs=getattr(node, "kwargs", {}),
        meta=getattr(node, "meta", {}),
    )


def add_value_if_missing(ir: IRGraph, value: IRValue) -> None:
    if value.id not in ir.values:
        ir.add_value(value)


def register_node(
    ir: IRGraph,
    ir_node: IRNode,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> None:
    ir.add_node(ir_node)
    ir.order.append(ir_node.id)
    for output_id in ir_node.outputs:
        ir.values[output_id].shape = shape
        ir.values[output_id].dtype = dtype
    for input_id in ir_node.inputs:
        if input_id in ir.values:
            ir.values[input_id].users.append(ir_node.id)


def _extract_numeric_literal(value: Any) -> float | int | None:
    literal = extract_literals(value)
    if isinstance(literal, bool):
        return int(literal)
    if isinstance(literal, (int, float)):
        return literal
    return None


def _extract_static_int(value: Any) -> int | None:
    if is_fx_node(value):
        meta = getattr(value, "meta", {}) or {}
        meta_value = meta.get("val")
        if meta_value is not None and not isinstance(meta_value, torch.Tensor):
            return _extract_static_int(meta_value)
        return None

    literal = extract_literals(value)
    if isinstance(literal, bool):
        return int(literal)
    if isinstance(literal, int):
        return int(literal)
    if isinstance(literal, float) and float(literal).is_integer():
        return int(literal)

    try:
        return int(value)
    except Exception:
        return None


def _extract_int_sequence_literal(value: Any) -> tuple[int, ...] | None:
    if is_fx_node(value):
        meta = getattr(value, "meta", {}) or {}
        meta_value = meta.get("val")
        if meta_value is not None and not isinstance(meta_value, torch.Tensor):
            return _extract_int_sequence_literal(meta_value)
        return None

    if isinstance(value, torch.Size):
        value = tuple(value)

    if isinstance(value, (tuple, list)):
        result: list[int] = []
        for item in value:
            item_value = _extract_static_int(item)
            if item_value is None:
                return None
            result.append(item_value)
        return tuple(result)

    scalar_value = _extract_static_int(value)
    if scalar_value is not None:
        return (scalar_value,)
    return None


def _try_materialize_shape(shape: tuple[Any, ...] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    result: list[int] = []
    for dim in shape:
        dim_value = _extract_static_int(dim)
        if dim_value is None:
            return None
        result.append(dim_value)
    return tuple(result)


def _extract_fx_tensor_shape(value: Any) -> tuple[Any, ...] | None:
    if not is_fx_node(value):
        return None
    meta = getattr(value, "meta", {}) or {}
    meta_value = meta.get("val")
    if isinstance(meta_value, torch.Tensor):
        return tuple(meta_value.shape)
    tensor_meta = meta.get("tensor_meta")
    tensor_shape = getattr(tensor_meta, "shape", None)
    if tensor_shape is not None:
        return tuple(tensor_shape)
    return None


def _base_meta(shape: tuple[int, ...] | None, dtype: str | None, torch_op: str, node: Any) -> dict[str, object]:
    aten_op = canonical_torch_op(torch_op)
    return {
        "shape": shape,
        "dtype": dtype,
        "torch_op": aten_op,
        "aten_op": aten_op,
        "torch_name": node.name,
    }


def _extract_fx_torch_dtype(node: Any) -> torch.dtype | None:
    if not is_fx_node(node):
        return None
    meta = getattr(node, "meta", {}) or {}
    value = meta.get("val")
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    tensor_meta = meta.get("tensor_meta")
    dtype = getattr(tensor_meta, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return None


def _import_scalar_binary(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
    op_name: str,
) -> None:
    lhs, rhs = node.args[0], node.args[1]
    lhs_is_node = is_fx_node(lhs)
    rhs_is_node = is_fx_node(rhs)
    lhs_literal = _extract_numeric_literal(lhs)
    rhs_literal = _extract_numeric_literal(rhs)

    if lhs_is_node and rhs_is_node:
        ir_node = IRNode(
            id=node_id(node),
            op=op_name,
            inputs=[value_id(lhs, ctx), value_id(rhs, ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
        return

    if op_name == "add":
        if lhs_is_node and rhs_literal is not None:
            scalar_op = "scalar_add"
            inputs = [value_id(lhs, ctx)]
            attrs = {"value": rhs_literal}
        elif rhs_is_node and lhs_literal is not None:
            scalar_op = "scalar_add"
            inputs = [value_id(rhs, ctx)]
            attrs = {"value": lhs_literal}
        else:
            scalar_op = None
    elif op_name == "multiply":
        if lhs_is_node and rhs_literal is not None:
            scalar_op = "scalar_multiply"
            inputs = [value_id(lhs, ctx)]
            attrs = {"value": rhs_literal}
        elif rhs_is_node and lhs_literal is not None:
            scalar_op = "scalar_multiply"
            inputs = [value_id(rhs, ctx)]
            attrs = {"value": lhs_literal}
        else:
            scalar_op = None
    elif op_name == "subtract":
        if lhs_is_node and rhs_literal is not None:
            scalar_op = "scalar_subtract"
            inputs = [value_id(lhs, ctx)]
            attrs = {"value": rhs_literal}
        elif rhs_is_node and lhs_literal is not None:
            scalar_op = "scalar_subtract_reverse"
            inputs = [value_id(rhs, ctx)]
            attrs = {"value": lhs_literal}
        else:
            scalar_op = None
    elif op_name == "divide":
        if lhs_is_node and rhs_literal is not None:
            scalar_op = "scalar_divide"
            inputs = [value_id(lhs, ctx)]
            attrs = {"value": rhs_literal}
        elif rhs_is_node and lhs_literal is not None:
            scalar_op = "scalar_divide_reverse"
            inputs = [value_id(rhs, ctx)]
            attrs = {"value": lhs_literal}
        else:
            scalar_op = None
    else:
        scalar_op = None

    if scalar_op is None:
        ctx.fail(f"unsupported scalar form for {torch_op}: {node.args!r}")

    ir_node = IRNode(
        id=node_id(node),
        op=scalar_op,
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_placeholder(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> None:
    add_value_if_missing(ir, IRValue(id=value_id(node, ctx), shape=shape, dtype=dtype, producer=None))
    ir.inputs.append(value_id(node, ctx))


def import_get_attr(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    value: Any,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    source_name: str | None = None,
) -> None:
    add_value_if_missing(ir, IRValue(id=value_id(node, ctx), shape=shape, dtype=dtype, producer=None))
    ir.constants[value_id(node, ctx)] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    ir.values[value_id(node, ctx)].meta["source_name"] = source_name
    if source_name is not None:
        bindings = ir.meta.setdefault("weight_bindings", {})
        if isinstance(bindings, dict):
            bindings.setdefault(value_id(node, ctx), {})["source_name"] = source_name


def import_output(ir: IRGraph, node: Any, ctx: ImportContext) -> None:
    ir.outputs.extend(extract_input_ids(node.args[0], ctx))


def import_call_function(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    op = normalize_target(torch_op)
    importer = OP_IMPORTERS.get(op)
    if importer is None:
        import_opaque_call_function(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name=op)
    else:
        importer(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op)


def import_opaque_call_function(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
    op_name: str,
) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op=op_name,
        inputs=extract_input_ids(getattr(node, "args", ()), ctx) + extract_input_ids(getattr(node, "kwargs", {}), ctx),
        outputs=[value_id(node, ctx)],
        attrs={
            "opaque": True,
            "torch_op": torch_op,
            "args": extract_literals(getattr(node, "args", ())),
            "kwargs": extract_literals(getattr(node, "kwargs", {})),
        },
        meta=_base_meta(shape, dtype, torch_op, node),
        kind="opaque",
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_add(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="add")


def import_subtract(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="subtract")


def import_multiply(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="multiply")


def import_multiply_inplace(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="multiply")


def import_divide(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="divide")


def import_not_equal(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    lhs, rhs = node.args[0], node.args[1]
    lhs_is_node = is_fx_node(lhs)
    rhs_is_node = is_fx_node(rhs)
    lhs_literal = _extract_numeric_literal(lhs)
    rhs_literal = _extract_numeric_literal(rhs)

    if lhs_is_node and rhs_is_node:
        ir_node = IRNode(
            id=node_id(node),
            op="not_equal",
            inputs=[value_id(lhs, ctx), value_id(rhs, ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
        return

    if lhs_is_node and rhs_literal is not None:
        inputs = [value_id(lhs, ctx)]
        attrs = {"value": rhs_literal}
    elif rhs_is_node and lhs_literal is not None:
        inputs = [value_id(rhs, ctx)]
        attrs = {"value": lhs_literal}
    else:
        ctx.fail(f"unsupported scalar form for {torch_op}: {node.args!r}")

    ir_node = IRNode(
        id=node_id(node),
        op="scalar_not_equal",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_equal(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    lhs, rhs = node.args[0], node.args[1]
    lhs_is_node = is_fx_node(lhs)
    rhs_is_node = is_fx_node(rhs)
    lhs_literal = _extract_numeric_literal(lhs)
    rhs_literal = _extract_numeric_literal(rhs)

    if lhs_is_node and rhs_is_node:
        ir_node = IRNode(
            id=node_id(node),
            op="equal",
            inputs=[value_id(lhs, ctx), value_id(rhs, ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
        return

    if lhs_is_node and rhs_literal is not None:
        inputs = [value_id(lhs, ctx)]
        attrs = {"value": rhs_literal}
    elif rhs_is_node and lhs_literal is not None:
        inputs = [value_id(rhs, ctx)]
        attrs = {"value": lhs_literal}
    else:
        ctx.fail(f"unsupported scalar form for {torch_op}: {node.args!r}")

    ir_node = IRNode(
        id=node_id(node),
        op="scalar_equal",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_compare(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str, op_name: str) -> None:
    lhs, rhs = node.args[0], node.args[1]
    lhs_is_node = is_fx_node(lhs)
    rhs_is_node = is_fx_node(rhs)
    lhs_literal = _extract_numeric_literal(lhs)
    rhs_literal = _extract_numeric_literal(rhs)

    if lhs_is_node and rhs_is_node:
        ir_node = IRNode(
            id=node_id(node),
            op=op_name,
            inputs=[value_id(lhs, ctx), value_id(rhs, ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
        return

    if lhs_is_node and rhs_literal is not None:
        inputs = [value_id(lhs, ctx)]
        scalar_op = f"scalar_{op_name}"
        scalar_value = rhs_literal
    elif rhs_is_node and lhs_literal is not None:
        reverse_ops = {
            "greater": "less",
            "greater_equal": "less_equal",
            "less": "greater",
            "less_equal": "greater_equal",
        }
        inputs = [value_id(rhs, ctx)]
        scalar_op = f"scalar_{reverse_ops[op_name]}"
        scalar_value = lhs_literal
    else:
        ctx.fail(f"unsupported scalar form for {torch_op}: {node.args!r}")

    ir_node = IRNode(
        id=node_id(node),
        op=scalar_op,
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs={"value": scalar_value},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_binary(op_name: str):
    def _importer(
        ir: IRGraph,
        node: Any,
        ctx: ImportContext,
        *,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        torch_op: str,
    ) -> None:
        ir_node = IRNode(
            id=node_id(node),
            op=op_name,
            inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)

    return _importer


def import_where(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    condition = node.args[0]
    true_value = node.args[1]
    false_value = node.args[2]

    inputs = [value_id(condition, ctx)]
    attrs: dict[str, object] = {}

    if is_fx_node(true_value):
        inputs.append(value_id(true_value, ctx))
        attrs["true_is_scalar"] = False
    else:
        literal = _extract_numeric_literal(true_value)
        if literal is None:
            ctx.fail(f"unsupported true branch for {torch_op}: {node.args!r}")
        attrs["true_is_scalar"] = True
        attrs["true_value"] = float(literal)

    if is_fx_node(false_value):
        inputs.append(value_id(false_value, ctx))
        attrs["false_is_scalar"] = False
    else:
        literal = _extract_numeric_literal(false_value)
        if literal is None:
            ctx.fail(f"unsupported false branch for {torch_op}: {node.args!r}")
        attrs["false_is_scalar"] = True
        attrs["false_value"] = float(literal)

    ir_node = IRNode(
        id=node_id(node),
        op="where",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_negate(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="scalar_multiply",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"value": -1.0},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_precision_cast(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    target_dtype = extract_literals(node.args[1]) if len(node.args) > 1 else None
    ir_node = IRNode(
        id=node_id(node),
        op="precision_cast",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"dtype": target_dtype},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_arange(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    attrs: dict[str, object] = {}
    if len(node.args) == 1:
        attrs["start"] = 0
        attrs["end"] = int(node.args[0])
    elif len(node.args) >= 2:
        attrs["start"] = int(node.args[0])
        attrs["end"] = int(node.args[1])
    else:
        ctx.fail(f"unsupported arange signature for {torch_op}: {node.args!r}")

    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        attrs["step"] = int(node.args[2])

    target_dtype = None
    if "dtype" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["dtype"]) is not None:
        target_dtype = extract_literals(node.kwargs["dtype"])
    elif dtype is not None:
        target_dtype = dtype
    if target_dtype is not None:
        attrs["dtype"] = target_dtype

    ir_node = IRNode(
        id=node_id(node),
        op="arange",
        inputs=[],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_type_as(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="type_as",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_identity(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="identity",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_reshape(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    target_shape = _extract_int_sequence_literal(node.args[1])
    if target_shape is None:
        ctx.fail(f"unsupported reshape shape for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="reshape",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"shape": target_shape},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_flatten(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="flatten",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"start_dim": int(node.args[1]), "end_dim": int(node.args[2])},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_unsqueeze(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    dim = int(node.args[1])
    attrs: dict[str, object] = {"dim": dim}
    materialized_shape = _try_materialize_shape(shape)
    if materialized_shape is not None:
        attrs["shape"] = materialized_shape
    ir_node = IRNode(
        id=node_id(node),
        op="unsqueeze",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_squeeze(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    dim = int(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else 0
    attrs: dict[str, object] = {"dim": dim}
    materialized_shape = _try_materialize_shape(shape)
    if materialized_shape is not None:
        attrs["shape"] = materialized_shape
    ir_node = IRNode(
        id=node_id(node),
        op="squeeze",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_masked_scatter(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="masked_scatter",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx), value_id(node.args[2], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_masked_fill(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    value = _extract_numeric_literal(node.args[2])
    if value is None:
        ctx.fail(f"unsupported masked_fill value for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="masked_fill",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"value": float(value)},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_expand(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    target_shape = _extract_int_sequence_literal(node.args[1])
    if target_shape is None and shape is not None:
        target_shape = _extract_int_sequence_literal(shape)
    if target_shape is None:
        ctx.fail(f"unsupported expand shape for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="expand",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"shape": target_shape},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_repeat(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    repeats = _extract_int_sequence_literal(node.args[1]) if len(node.args) > 1 else None
    if repeats is None:
        ctx.fail(f"unsupported repeat factors for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="repeat",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"repeats": repeats},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_one_hot(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    num_classes = _extract_static_int(node.args[1]) if len(node.args) > 1 else None
    if (num_classes is None or num_classes <= 0) and shape is not None and shape:
        num_classes = int(shape[-1])
    if num_classes is None or num_classes <= 0:
        ctx.fail(f"unsupported one_hot num_classes for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="one_hot",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"num_classes": int(num_classes)},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_tril(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    diagonal = int(extract_literals(node.args[1])) if len(node.args) > 1 and extract_literals(node.args[1]) is not None else 0
    ir_node = IRNode(
        id=node_id(node),
        op="tril",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"diagonal": diagonal},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_unfold(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 4:
        ctx.fail(f"unsupported unfold args for {torch_op}: {node.args!r}")
    dimension = extract_literals(node.args[1])
    size = extract_literals(node.args[2])
    step = extract_literals(node.args[3])
    if not isinstance(dimension, int) or not isinstance(size, int) or not isinstance(step, int):
        ctx.fail(f"unsupported unfold args for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="unfold",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"dimension": dimension, "size": size, "step": step},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_transpose(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    attrs = {"dim0": 0, "dim1": 1} if torch_op == "aten.t.default" else {"dim0": int(node.args[1]), "dim1": int(node.args[2])}
    ir_node = IRNode(
        id=node_id(node),
        op="transpose",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_permute(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    permutation = _extract_int_sequence_literal(node.args[1])
    if permutation is None:
        ctx.fail(f"unsupported permute dims for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="permute",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"permutation": permutation},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_numpy_t(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    input_shape = _extract_fx_tensor_shape(node.args[0])
    rank_source = input_shape if input_shape is not None else shape
    if rank_source is None:
        ctx.fail(f"unsupported numpy_T rank inference for {torch_op}: {node.args!r}")
    permutation = tuple(range(len(rank_source) - 1, -1, -1))
    ir_node = IRNode(
        id=node_id(node),
        op="permute",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"permutation": permutation},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_movedim(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    source = _extract_int_sequence_literal(node.args[1])
    destination = _extract_int_sequence_literal(node.args[2])
    if source is None or destination is None:
        ctx.fail(f"unsupported movedim dims for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="movedim",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"source": source, "destination": destination},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_matmul(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="matmul",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_linear(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = [value_id(node.args[0], ctx), value_id(node.args[1], ctx)]
    attrs = {"has_bias": False}
    if len(node.args) > 2 and node.args[2] is not None:
        inputs.append(value_id(node.args[2], ctx))
        attrs["has_bias"] = True
    ir_node = IRNode(
        id=node_id(node),
        op="linear",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_addmm(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="addmm",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[1], ctx), value_id(node.args[2], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_unary(op_name: str):
    def _importer(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
        ir_node = IRNode(
            id=node_id(node),
            op=op_name,
            inputs=[value_id(node.args[0], ctx)],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
    return _importer


def import_clamp(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    min_value = extract_literals(node.args[1]) if len(node.args) > 1 else None
    max_value = extract_literals(node.args[2]) if len(node.args) > 2 else None
    attrs: dict[str, object] = {}
    if isinstance(min_value, (int, float)):
        attrs["min"] = float(min_value)
    if isinstance(max_value, (int, float)):
        attrs["max"] = float(max_value)
    ir_node = IRNode(
        id=node_id(node),
        op="clamp",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_pow_rsqrt(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="pow",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"exponent": -0.5},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_reciprocal(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="pow",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"exponent": -1.0},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_floor_divide(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported floor_divide signature for {torch_op}: {node.args!r}")
    divisor = _extract_numeric_literal(node.args[1])
    if divisor is None:
        ctx.fail(f"unsupported floor_divide divisor for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="scalar_floor_divide",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"value": float(divisor)},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_softmax(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="softmax",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"axis": int(node.args[1])},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_glu(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    axis = -1
    if len(node.args) > 1 and extract_literals(node.args[1]) is not None:
        axis = int(node.args[1])
    ir_node = IRNode(
        id=node_id(node),
        op="glu",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"axis": axis},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_scaled_dot_product_attention(
    ir: IRGraph,
    node: Any,
    ctx: ImportContext,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    torch_op: str,
) -> None:
    attrs: dict[str, object] = {}
    inputs = [value_id(node.args[0], ctx), value_id(node.args[1], ctx), value_id(node.args[2], ctx)]
    if len(node.args) > 3:
        mask_literal = extract_literals(node.args[3])
        if mask_literal is not None:
            attrs["mask"] = mask_literal
        elif is_fx_node(node.args[3]):
            inputs.append(value_id(node.args[3], ctx))
            mask_dtype = _extract_fx_torch_dtype(node.args[3])
            if mask_dtype is not None and mask_dtype != torch.bool:
                attrs["additive_mask"] = True
    if len(node.args) > 4 and extract_literals(node.args[4]) is not None:
        attrs["dropout_p"] = float(node.args[4])
    if len(node.args) > 5 and extract_literals(node.args[5]) is not None:
        attrs["is_causal"] = bool(node.args[5])
    if "scale" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["scale"]) is not None:
        attrs["scale"] = float(node.kwargs["scale"])
    if "enable_gqa" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["enable_gqa"]) is not None:
        attrs["enable_gqa"] = bool(node.kwargs["enable_gqa"])
    ir_node = IRNode(
        id=node_id(node),
        op="scaled_dot_product_attention",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_reduce(op_name: str):
    def _importer(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
        attrs: dict[str, object] = {}
        if len(node.args) > 1 and extract_literals(node.args[1]) is not None:
            attrs["axis"] = extract_literals(node.args[1])
        keepdim = None
        if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
            keepdim = bool(extract_literals(node.args[2]))
        elif "keepdim" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["keepdim"]) is not None:
            keepdim = bool(extract_literals(node.kwargs["keepdim"]))
        if keepdim is not None:
            attrs["keepdim"] = keepdim
        ir_node = IRNode(
            id=node_id(node),
            op=op_name,
            inputs=[value_id(node.args[0], ctx)],
            outputs=[value_id(node, ctx)],
            attrs=attrs,
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, ir_node, shape=shape, dtype=dtype)
    return _importer


def import_cat(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    tensors_arg = node.args[0]
    axis = int(node.args[1]) if len(node.args) > 1 else 0
    ir_node = IRNode(
        id=node_id(node),
        op="cat",
        inputs=extract_input_ids(tensors_arg, ctx),
        outputs=[value_id(node, ctx)],
        attrs={"axis": axis},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_split_with_sizes(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported split_with_sizes signature for {torch_op}: {node.args!r}")
    sizes = _extract_int_sequence_literal(node.args[1])
    if sizes is None:
        ctx.fail(f"unsupported split sizes for {torch_op}: {node.args!r}")
    axis = -1
    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        axis = int(extract_literals(node.args[2]))
    ir_node = IRNode(
        id=node_id(node),
        op="split_with_sizes",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"sizes": sizes, "axis": axis},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_chunk(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported chunk signature for {torch_op}: {node.args!r}")
    chunks = extract_literals(node.args[1])
    if not isinstance(chunks, int):
        ctx.fail(f"unsupported chunk count for {torch_op}: {node.args!r}")
    axis = 0
    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        axis = int(extract_literals(node.args[2]))
    ir_node = IRNode(
        id=node_id(node),
        op="chunk",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"chunks": int(chunks), "axis": axis},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_pow(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    base_literal = _extract_numeric_literal(node.args[0]) if len(node.args) > 0 else None
    attrs: dict[str, object] = {}
    exponent_literal = extract_literals(node.args[1]) if len(node.args) > 1 else None

    if base_literal is not None and isinstance(exponent_literal, (int, float)):
        add_value_if_missing(ir, IRValue(id=value_id(node, ctx), shape=shape, dtype=dtype, producer=None))
        ir.constants[value_id(node, ctx)] = base_literal**exponent_literal
        return

    if base_literal is not None:
        if not (len(node.args) > 1 and is_fx_node(node.args[1])):
            ctx.fail(f"unsupported pow signature for {torch_op}: scalar base with non-literal exponent {node.args!r}")
        if float(base_literal) <= 0.0:
            ctx.fail(f"unsupported pow signature for {torch_op}: non-positive scalar base {base_literal!r}")

        exponent_value_id = value_id(node.args[1], ctx)
        mul_output_id = f"{value_id(node, ctx)}__pow_scalar_base_log_mul"
        mul_node = IRNode(
            id=f"{node_id(node)}__pow_scalar_base_log_mul",
            op="scalar_multiply",
            inputs=[exponent_value_id],
            outputs=[mul_output_id],
            attrs={"value": float(math.log(float(base_literal)))},
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, mul_node, shape=shape, dtype=dtype)

        exp_node = IRNode(
            id=node_id(node),
            op="scalar_exp",
            inputs=[mul_output_id],
            outputs=[value_id(node, ctx)],
            meta=_base_meta(shape, dtype, torch_op, node),
        )
        register_node(ir, exp_node, shape=shape, dtype=dtype)
        return

    inputs = [value_id(node.args[0], ctx)]
    if exponent_literal is None and len(node.args) > 1 and is_fx_node(node.args[1]):
        inputs.append(value_id(node.args[1], ctx))
    else:
        attrs["exponent"] = exponent_literal
    ir_node = IRNode(
        id=node_id(node),
        op="pow",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_slice(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    attrs = {"axis": int(node.args[1]), "start": int(node.args[2]), "end": int(node.args[3])}
    if len(node.args) > 4 and extract_literals(node.args[4]) is not None:
        attrs["step"] = int(node.args[4])
    ir_node = IRNode(
        id=node_id(node),
        op="slice",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_diff(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = list(getattr(node, "args", ()))
    if not inputs:
        ctx.fail(f"unsupported diff signature for {torch_op}: missing input tensor")

    source = inputs[0]
    if not is_fx_node(source):
        ctx.fail(f"unsupported diff signature for {torch_op}: source is not FX node")

    n_value = 1
    if len(inputs) > 1 and extract_literals(inputs[1]) is not None:
        n_value = int(extract_literals(inputs[1]))
    elif "n" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["n"]) is not None:
        n_value = int(extract_literals(node.kwargs["n"]))
    if n_value != 1:
        ctx.fail(f"unsupported diff order for {torch_op}: n={n_value}")

    dim_value = None
    if len(inputs) > 2 and extract_literals(inputs[2]) is not None:
        dim_value = int(extract_literals(inputs[2]))
    elif "dim" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["dim"]) is not None:
        dim_value = int(extract_literals(node.kwargs["dim"]))
    if dim_value is None:
        ctx.fail(f"unsupported diff signature for {torch_op}: missing integer dim")

    prepend_value = None
    append_value = None
    prepend_node = None
    append_node = None
    if len(inputs) > 3:
        if is_fx_node(inputs[3]):
            prepend_node = inputs[3]
        else:
            prepend_value = extract_literals(inputs[3])
    if len(inputs) > 4:
        if is_fx_node(inputs[4]):
            append_node = inputs[4]
        else:
            append_value = extract_literals(inputs[4])
    if "prepend" in getattr(node, "kwargs", {}):
        prepend_arg = node.kwargs["prepend"]
        if is_fx_node(prepend_arg):
            prepend_node = prepend_arg
            prepend_value = None
        else:
            prepend_value = extract_literals(prepend_arg)
    if "append" in getattr(node, "kwargs", {}):
        append_arg = node.kwargs["append"]
        if is_fx_node(append_arg):
            append_node = append_arg
            append_value = None
        else:
            append_value = extract_literals(append_arg)
    if prepend_value is not None or append_value is not None:
        ctx.fail(f"unsupported diff signature for {torch_op}: literal prepend/append are not supported")

    source_value_id = value_id(source, ctx)
    source_shape = None
    if source_value_id in ir.values:
        source_shape = ir.values[source_value_id].shape
    if source_shape is None:
        source_shape = shape
    if source_shape is None:
        ctx.fail(f"unsupported diff import for {torch_op}: missing source shape")

    rank = len(source_shape)
    normalized_dim = dim_value if dim_value >= 0 else dim_value + rank
    if normalized_dim < 0 or normalized_dim >= rank:
        ctx.fail(f"unsupported diff dimension for {torch_op}: dim={dim_value} rank={rank}")

    dim_extent = int(source_shape[normalized_dim])
    if dim_extent < 1:
        ctx.fail(f"unsupported diff input extent for {torch_op}: dim size {dim_extent}")

    concat_inputs = [source_value_id]
    concat_shapes = [tuple(int(v) for v in source_shape)]

    def _append_concat_input(arg: Any, label: str) -> None:
        nonlocal concat_inputs, concat_shapes
        if arg is None:
            return
        arg_value_id = value_id(arg, ctx)
        arg_value = ir.values.get(arg_value_id)
        arg_shape = arg_value.shape if arg_value is not None else None
        if arg_shape is None:
            ctx.fail(f"unsupported diff import for {torch_op}: missing {label} shape")
        arg_shape = tuple(int(v) for v in arg_shape)
        if len(arg_shape) != rank:
            ctx.fail(
                f"unsupported diff import for {torch_op}: {label} rank {len(arg_shape)} does not match source rank {rank}"
            )
        for axis, (arg_dim, src_dim) in enumerate(zip(arg_shape, source_shape, strict=True)):
            if axis == normalized_dim:
                continue
            if int(arg_dim) != int(src_dim):
                ctx.fail(
                    f"unsupported diff import for {torch_op}: {label} shape {arg_shape} "
                    f"is incompatible with source shape {source_shape}"
                )
        concat_inputs.append(arg_value_id)
        concat_shapes.append(arg_shape)

    if prepend_node is not None:
        concat_inputs = []
        concat_shapes = []
        _append_concat_input(prepend_node, "prepend")
        concat_inputs.append(source_value_id)
        concat_shapes.append(tuple(int(v) for v in source_shape))
        if append_node is not None:
            _append_concat_input(append_node, "append")
    elif append_node is not None:
        _append_concat_input(append_node, "append")

    augmented_shape = list(source_shape)
    augmented_shape[normalized_dim] = sum(int(shape_item[normalized_dim]) for shape_item in concat_shapes)
    augmented_shape_tuple = tuple(int(v) for v in augmented_shape)
    diff_shape = list(augmented_shape)
    diff_shape[normalized_dim] = max(0, int(augmented_shape[normalized_dim]) - 1)
    diff_shape_tuple = tuple(int(v) for v in diff_shape)

    diff_source_value_id = source_value_id
    if len(concat_inputs) > 1:
        concat_output_id = f"{value_id(node, ctx)}__diff_source"
        concat_node = IRNode(
            id=f"{node_id(node)}__diff_source",
            op="cat",
            inputs=concat_inputs,
            outputs=[concat_output_id],
            attrs={"axis": normalized_dim},
            meta=_base_meta(augmented_shape_tuple, dtype, torch_op, node),
        )
        register_node(ir, concat_node, shape=augmented_shape_tuple, dtype=dtype)
        diff_source_value_id = concat_output_id

    left_shape = list(diff_shape_tuple)
    left_shape_tuple = tuple(int(v) for v in left_shape)

    left_node = IRNode(
        id=f"{node_id(node)}__diff_left",
        op="slice",
        inputs=[diff_source_value_id],
        outputs=[f"{value_id(node, ctx)}__diff_left"],
        attrs={"axis": normalized_dim, "start": 0, "end": int(augmented_shape[normalized_dim]) - 1, "step": 1},
        meta=_base_meta(left_shape_tuple, dtype, torch_op, node),
    )
    register_node(ir, left_node, shape=left_shape_tuple, dtype=dtype)

    right_node = IRNode(
        id=f"{node_id(node)}__diff_right",
        op="slice",
        inputs=[diff_source_value_id],
        outputs=[f"{value_id(node, ctx)}__diff_right"],
        attrs={"axis": normalized_dim, "start": 1, "end": int(augmented_shape[normalized_dim]), "step": 1},
        meta=_base_meta(left_shape_tuple, dtype, torch_op, node),
    )
    register_node(ir, right_node, shape=left_shape_tuple, dtype=dtype)

    diff_node = IRNode(
        id=node_id(node),
        op="subtract",
        inputs=[right_node.outputs[0], left_node.outputs[0]],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(diff_shape_tuple, dtype, torch_op, node),
    )
    register_node(ir, diff_node, shape=diff_shape_tuple, dtype=dtype)


def import_index(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if torch_op.startswith("aten.select"):
        axis = int(node.args[1])
        index_value = int(node.args[2])
    else:
        index_value = int(node.args[1])
        axis = int(node.args[2]) if len(node.args) > 2 else 0
    ir_node = IRNode(
        id=node_id(node),
        op="index",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"index_value": index_value, "axis": axis},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_gather(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="gather",
        inputs=[value_id(node.args[0], ctx), value_id(node.args[2], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"axis": int(node.args[1])},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_embedding(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = [value_id(node.args[0], ctx), value_id(node.args[1], ctx)]
    attrs: dict[str, object] = {}
    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        attrs["padding_idx"] = int(node.args[2])
    if len(node.args) > 3 and extract_literals(node.args[3]) is not None:
        attrs["scale_grad_by_freq"] = bool(node.args[3])
    if len(node.args) > 4 and extract_literals(node.args[4]) is not None:
        attrs["sparse"] = bool(node.args[4])
    ir_node = IRNode(
        id=node_id(node),
        op="embedding",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def _extract_int_list_or_scalar(value: Any, *, default: int) -> int:
    literal = extract_literals(value)
    if literal is None:
        return default
    if isinstance(literal, int):
        return int(literal)
    if isinstance(literal, (tuple, list)) and literal:
        first = literal[0]
        if isinstance(first, int):
            return int(first)
    raise UnsupportedImportError(f"unsupported integer/list literal: {value!r}")


def import_conv1d(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported conv1d signature for {torch_op}: {node.args!r}")

    inputs = [value_id(node.args[0], ctx), value_id(node.args[1], ctx)]
    if len(node.args) > 2 and is_fx_node(node.args[2]):
        inputs.append(value_id(node.args[2], ctx))

    stride = _extract_int_list_or_scalar(node.args[3], default=1) if len(node.args) > 3 else 1
    padding = _extract_int_list_or_scalar(node.args[4], default=0) if len(node.args) > 4 else 0
    dilation = _extract_int_list_or_scalar(node.args[5], default=1) if len(node.args) > 5 else 1
    groups = _extract_int_list_or_scalar(node.args[6], default=1) if len(node.args) > 6 else 1

    ir_node = IRNode(
        id=node_id(node),
        op="conv1d",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs={"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_conv2d(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported conv2d signature for {torch_op}: {node.args!r}")

    inputs = [value_id(node.args[0], ctx), value_id(node.args[1], ctx)]
    if len(node.args) > 2 and is_fx_node(node.args[2]):
        inputs.append(value_id(node.args[2], ctx))

    stride = _extract_int_list_or_scalar(node.args[3], default=1) if len(node.args) > 3 else 1
    padding = _extract_int_list_or_scalar(node.args[4], default=0) if len(node.args) > 4 else 0
    dilation = _extract_int_list_or_scalar(node.args[5], default=1) if len(node.args) > 5 else 1
    groups = _extract_int_list_or_scalar(node.args[6], default=1) if len(node.args) > 6 else 1

    ir_node = IRNode(
        id=node_id(node),
        op="conv2d",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs={"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_pad(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if len(node.args) < 2:
        ctx.fail(f"unsupported pad signature for {torch_op}: {node.args!r}")

    pads = extract_literals(node.args[1])
    if not isinstance(pads, (tuple, list)) or not all(isinstance(v, int) for v in pads):
        ctx.fail(f"unsupported pads for {torch_op}: {node.args!r}")

    mode = "constant"
    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        mode = str(extract_literals(node.args[2]))
    if "mode" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["mode"]) is not None:
        mode = str(extract_literals(node.kwargs["mode"]))

    value = 0.0
    if len(node.args) > 3 and extract_literals(node.args[3]) is not None:
        value = float(extract_literals(node.args[3]))
    if "value" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["value"]) is not None:
        value = float(extract_literals(node.kwargs["value"]))

    ir_node = IRNode(
        id=node_id(node),
        op="pad",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"pads": tuple(int(v) for v in pads), "mode": mode, "value": value},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_ones(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if not node.args:
        ctx.fail(f"unsupported ones signature for {torch_op}: {node.args!r}")

    output_shape = _extract_int_sequence_literal(node.args[0])
    if output_shape is None:
        ctx.fail(f"unsupported ones shape for {torch_op}: {node.args!r}")

    node_dtype = dtype
    if "dtype" in getattr(node, "kwargs", {}) and extract_literals(node.kwargs["dtype"]) is not None:
        node_dtype = dtype_to_ir(extract_literals(node.kwargs["dtype"]))
    elif len(node.args) > 1 and extract_literals(node.args[1]) is not None:
        node_dtype = dtype_to_ir(extract_literals(node.args[1]))
    if node_dtype is None:
        node_dtype = "fp32"

    ir_node = IRNode(
        id=node_id(node),
        op="ones",
        inputs=[],
        outputs=[value_id(node, ctx)],
        attrs={"shape": output_shape, "dtype": node_dtype},
        meta=_base_meta(shape or output_shape, node_dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape or output_shape, dtype=node_dtype)


def import_layer_norm(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    if not node.args:
        ctx.fail(f"unsupported layer_norm signature for {torch_op}: {node.args!r}")

    normalized_shape_value = None
    if len(node.args) > 1:
        normalized_shape_value = _extract_int_sequence_literal(node.args[1])
    if normalized_shape_value is None:
        normalized_shape_value = _extract_int_sequence_literal(getattr(node, "kwargs", {}).get("normalized_shape"))
    if normalized_shape_value is None:
        ctx.fail(f"unsupported normalized_shape for {torch_op}: {node.args!r} {getattr(node, 'kwargs', {})!r}")

    eps_value = None
    if len(node.args) > 4:
        eps_value = _extract_numeric_literal(node.args[4])
    if eps_value is None:
        eps_value = _extract_numeric_literal(getattr(node, "kwargs", {}).get("eps"))
    if eps_value is None:
        eps_value = 1e-5

    inputs = [value_id(node.args[0], ctx)]
    weight = node.args[2] if len(node.args) > 2 else getattr(node, "kwargs", {}).get("weight")
    bias = node.args[3] if len(node.args) > 3 else getattr(node, "kwargs", {}).get("bias")
    if not is_fx_node(weight):
        ctx.fail(f"unsupported layer_norm weight form for {torch_op}: {node.args!r} {getattr(node, 'kwargs', {})!r}")
    inputs.append(value_id(weight, ctx))
    if bias is None:
        pass
    elif is_fx_node(bias):
        inputs.append(value_id(bias, ctx))
    else:
        ctx.fail(f"unsupported layer_norm bias form for {torch_op}: {node.args!r} {getattr(node, 'kwargs', {})!r}")

    attrs = {"normalized_shape": tuple(int(v) for v in normalized_shape_value), "eps": float(eps_value)}
    ir_node = IRNode(
        id=node_id(node),
        op="layer_norm",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_rms_norm(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = [value_id(node.args[0], ctx), value_id(node.args[2], ctx)]
    attrs = {"normalized_shape": tuple(int(v) for v in node.args[1]), "eps": float(node.args[3])}
    ir_node = IRNode(
        id=node_id(node),
        op="rms_norm",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_group_norm(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = [value_id(node.args[0], ctx), value_id(node.args[2], ctx), value_id(node.args[3], ctx)]
    attrs = {"num_groups": int(node.args[1]), "eps": float(node.args[4])}
    ir_node = IRNode(
        id=node_id(node),
        op="group_norm",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_batch_norm(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    inputs = [
        value_id(node.args[0], ctx),
        value_id(node.args[1], ctx),
        value_id(node.args[2], ctx),
        value_id(node.args[3], ctx),
        value_id(node.args[4], ctx),
    ]
    attrs = {"training": bool(node.args[5]), "momentum": float(node.args[6]), "eps": float(node.args[7])}
    ir_node = IRNode(
        id=node_id(node),
        op="batch_norm",
        inputs=inputs,
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_contiguous(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    ir_node = IRNode(
        id=node_id(node),
        op="contiguous",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_getitem(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    index_value = extract_literals(node.args[1])
    if not isinstance(index_value, int):
        ctx.fail(f"unsupported getitem index for {torch_op}: {node.args!r}")
    ir_node = IRNode(
        id=node_id(node),
        op="getitem",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"index": index_value},
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


OP_IMPORTERS = {
    "arange": import_arange,
    "identity": import_identity,
    "add": import_add,
    "subtract": import_subtract,
    "multiply": import_multiply,
    "multiply_inplace": import_multiply_inplace,
    "divide": import_divide,
    "precision_cast": import_precision_cast,
    "type_as": import_type_as,
    "negate": import_negate,
    "abs": import_unary("abs"),
    "clamp": import_clamp,
    "not_equal": import_not_equal,
    "equal": import_equal,
    "greater": lambda ir, node, ctx, *, shape, dtype, torch_op: import_compare(
        ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="greater"
    ),
    "greater_equal": lambda ir, node, ctx, *, shape, dtype, torch_op: import_compare(
        ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="greater_equal"
    ),
    "less": lambda ir, node, ctx, *, shape, dtype, torch_op: import_compare(
        ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="less"
    ),
    "less_equal": lambda ir, node, ctx, *, shape, dtype, torch_op: import_compare(
        ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="less_equal"
    ),
    "logical_and": import_binary("logical_and"),
    "logical_or": import_binary("logical_or"),
    "logical_not": import_unary("logical_not"),
    "where": import_where,
    "masked_scatter": import_masked_scatter,
    "masked_fill": import_masked_fill,
    "cos": import_unary("cos"),
    "sin": import_unary("sin"),
    "floor_divide": import_floor_divide,
    "scalar_exp": import_unary("scalar_exp"),
    "scalar_sqrt": import_unary("scalar_sqrt"),
    "reciprocal": import_reciprocal,
    "scalar_log": import_unary("scalar_log"),
    "rsqrt": import_pow_rsqrt,
    "reshape": import_reshape,
    "flatten": import_flatten,
    "unsqueeze": import_unsqueeze,
    "squeeze": import_squeeze,
    "expand": import_expand,
    "repeat": import_repeat,
    "one_hot": import_one_hot,
    "tril": import_tril,
    "unfold": import_unfold,
    "numpy_T": import_numpy_t,
    "transpose": import_transpose,
    "permute": import_permute,
    "movedim": import_movedim,
    "matmul": import_matmul,
    "linear": import_linear,
    "addmm": import_addmm,
    "relu": import_unary("relu"),
    "silu": import_unary("silu"),
    "gelu": import_unary("gelu"),
    "gelu_erf": import_unary("gelu_erf"),
    "sigmoid": import_unary("sigmoid"),
    "glu": import_glu,
    "softplus": import_unary("softplus"),
    "tanh": import_unary("tanh"),
    "softmax": import_softmax,
    "scaled_dot_product_attention": import_scaled_dot_product_attention,
    "sum": import_reduce("sum"),
    "mean": import_reduce("mean"),
    "variance": import_reduce("variance"),
    "min": import_reduce("min"),
    "max": import_reduce("max"),
    "cat": import_cat,
    "split_with_sizes": import_split_with_sizes,
    "chunk": import_chunk,
    "ones": import_ones,
    "pad": import_pad,
    "pow": import_pow,
    "aten.diff.default": import_diff,
    "slice": import_slice,
    "index": import_index,
    "gather": import_gather,
    "embedding": import_embedding,
    "conv1d": import_conv1d,
    "conv2d": import_conv2d,
    "layer_norm": import_layer_norm,
    "rms_norm": import_rms_norm,
    "group_norm": import_group_norm,
    "batch_norm": import_batch_norm,
    "contiguous": import_contiguous,
    "getitem": import_getitem,
}
