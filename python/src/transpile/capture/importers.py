from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace
from typing import Any

import torch

from src.transpile.capture.graph_ir import IRGraph
from src.transpile.capture.graph_ir import IRNode
from src.transpile.capture.graph_ir import IRValue
from src.transpile.capture.normalize import dtype_to_ir
from src.transpile.capture.normalize import normalize_target


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

    def _node_module_paths(self, node: Any) -> tuple[str, ...]:
        meta = getattr(node, "meta", {}) or {}
        raw_stack = meta.get("nn_module_stack")
        if not isinstance(raw_stack, dict):
            return ()
        paths: list[str] = []
        for entry in raw_stack.values():
            if not isinstance(entry, (tuple, list)) or not entry:
                continue
            module_path = entry[0]
            if isinstance(module_path, str) and module_path:
                paths.append(module_path)
        # Prefer the deepest module path first.
        paths.reverse()
        deduped: list[str] = []
        for path in paths:
            if path not in deduped:
                deduped.append(path)
        return tuple(deduped)

    def lookup_import_hint(self, node: Any, *, torch_op: str, op_name: str) -> dict[str, object]:
        hints = self.transpile_metadata.get("import_hints", ())
        if not isinstance(hints, list):
            return {}

        module_paths = self._node_module_paths(node)
        matched_attrs: dict[str, object] = {}
        matched_meta: dict[str, object] = {}
        matched_provider: list[str] = []

        for hint in hints:
            if not isinstance(hint, dict):
                continue
            hint_op = hint.get("op")
            if hint_op is not None and hint_op != op_name:
                continue
            hint_torch_op = hint.get("torch_op")
            if hint_torch_op is not None and hint_torch_op != torch_op:
                continue

            module_path = hint.get("module_path")
            if module_path is not None and module_path not in module_paths:
                continue
            module_path_suffix = hint.get("module_path_suffix")
            if module_path_suffix is not None and not any(
                path == module_path_suffix or path.endswith(f".{module_path_suffix}") for path in module_paths
            ):
                continue

            attrs = hint.get("attrs", {})
            if isinstance(attrs, dict):
                matched_attrs.update(attrs)
            meta = hint.get("meta", {})
            if isinstance(meta, dict):
                matched_meta.update(meta)
            provider = hint.get("provider")
            if isinstance(provider, str) and provider not in matched_provider:
                matched_provider.append(provider)

        if not matched_attrs and not matched_meta and not matched_provider:
            return {}

        if matched_provider:
            matched_meta.setdefault("import_hint_providers", tuple(matched_provider))
            matched_meta.setdefault("import_hint_source", "capture_metadata")
        return {"attrs": matched_attrs, "meta": matched_meta}


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
        return None
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


def _base_meta(shape: tuple[int, ...] | None, dtype: str | None, torch_op: str, node: Any) -> dict[str, object]:
    return {
        "shape": shape,
        "dtype": dtype,
        "torch_op": torch_op,
        "torch_name": node.name,
    }


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
    import_hint = ctx.lookup_import_hint(node, torch_op=torch_op, op_name=op)
    importer = OP_IMPORTERS.get(op)
    if importer is None:
        import_opaque_call_function(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name=op)
    else:
        importer(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op)
    _apply_import_hint(ir, node, import_hint)


def _apply_import_hint(ir: IRGraph, node: Any, hint: dict[str, object]) -> None:
    if not hint:
        return
    ir_node = ir.nodes.get(node_id(node))
    if ir_node is None:
        return
    attrs = hint.get("attrs", {})
    if isinstance(attrs, dict):
        ir_node.attrs.update(attrs)
    meta = hint.get("meta", {})
    if isinstance(meta, dict):
        ir_node.meta.update(meta)


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
    _import_scalar_binary(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op, op_name="multiply_inplace")


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
    target_shape = tuple(int(v) for v in node.args[1])
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
    if shape is not None:
        attrs["shape"] = tuple(int(v) for v in shape)
    ir_node = IRNode(
        id=node_id(node),
        op="unsqueeze",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs=attrs,
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, ir_node, shape=shape, dtype=dtype)


def import_expand(ir: IRGraph, node: Any, ctx: ImportContext, *, shape: tuple[int, ...] | None, dtype: str | None, torch_op: str) -> None:
    target_shape = tuple(int(v) for v in node.args[1])
    ir_node = IRNode(
        id=node_id(node),
        op="expand",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"shape": target_shape},
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
    permutation = tuple(int(v) for v in node.args[1])
    ir_node = IRNode(
        id=node_id(node),
        op="permute",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"permutation": permutation},
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
    sizes = extract_literals(node.args[1])
    if not isinstance(sizes, (tuple, list)) or not all(isinstance(v, int) for v in sizes):
        ctx.fail(f"unsupported split sizes for {torch_op}: {node.args!r}")
    axis = -1
    if len(node.args) > 2 and extract_literals(node.args[2]) is not None:
        axis = int(extract_literals(node.args[2]))
    ir_node = IRNode(
        id=node_id(node),
        op="split_with_sizes",
        inputs=[value_id(node.args[0], ctx)],
        outputs=[value_id(node, ctx)],
        attrs={"sizes": tuple(int(v) for v in sizes), "axis": axis},
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
    inputs = [value_id(node.args[0], ctx)]
    attrs: dict[str, object] = {}
    exponent_literal = extract_literals(node.args[1]) if len(node.args) > 1 else None
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
    if len(inputs) > 3:
        prepend_value = extract_literals(inputs[3])
    if len(inputs) > 4:
        append_value = extract_literals(inputs[4])
    if "prepend" in getattr(node, "kwargs", {}):
        prepend_value = extract_literals(node.kwargs["prepend"])
    if "append" in getattr(node, "kwargs", {}):
        append_value = extract_literals(node.kwargs["append"])
    if prepend_value is not None or append_value is not None:
        ctx.fail(f"unsupported diff signature for {torch_op}: prepend/append are not supported")

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

    left_shape = list(source_shape)
    left_shape[normalized_dim] = max(0, dim_extent - 1)
    left_shape_tuple = tuple(int(v) for v in left_shape)

    left_node = IRNode(
        id=f"{node_id(node)}__diff_left",
        op="slice",
        inputs=[source_value_id],
        outputs=[f"{value_id(node, ctx)}__diff_left"],
        attrs={"axis": normalized_dim, "start": 0, "end": dim_extent - 1, "step": 1},
        meta=_base_meta(left_shape_tuple, dtype, torch_op, node),
    )
    register_node(ir, left_node, shape=left_shape_tuple, dtype=dtype)

    right_node = IRNode(
        id=f"{node_id(node)}__diff_right",
        op="slice",
        inputs=[source_value_id],
        outputs=[f"{value_id(node, ctx)}__diff_right"],
        attrs={"axis": normalized_dim, "start": 1, "end": dim_extent, "step": 1},
        meta=_base_meta(left_shape_tuple, dtype, torch_op, node),
    )
    register_node(ir, right_node, shape=left_shape_tuple, dtype=dtype)

    diff_node = IRNode(
        id=node_id(node),
        op="subtract",
        inputs=[right_node.outputs[0], left_node.outputs[0]],
        outputs=[value_id(node, ctx)],
        meta=_base_meta(shape, dtype, torch_op, node),
    )
    register_node(ir, diff_node, shape=shape, dtype=dtype)


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

    shape_literal = extract_literals(node.args[0])
    if isinstance(shape_literal, int):
        output_shape = (int(shape_literal),)
    elif isinstance(shape_literal, (tuple, list)) and all(isinstance(v, int) for v in shape_literal):
        output_shape = tuple(int(v) for v in shape_literal)
    else:
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
    inputs = [value_id(node.args[0], ctx), value_id(node.args[2], ctx), value_id(node.args[3], ctx)]
    attrs = {"normalized_shape": tuple(int(v) for v in node.args[1]), "eps": float(node.args[4])}
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
    "not_equal": import_not_equal,
    "cos": import_unary("cos"),
    "sin": import_unary("sin"),
    "scalar_exp": import_unary("scalar_exp"),
    "scalar_sqrt": import_unary("scalar_sqrt"),
    "scalar_log": import_unary("scalar_log"),
    "rsqrt": import_pow_rsqrt,
    "reshape": import_reshape,
    "flatten": import_flatten,
    "unsqueeze": import_unsqueeze,
    "expand": import_expand,
    "transpose": import_transpose,
    "permute": import_permute,
    "matmul": import_matmul,
    "linear": import_linear,
    "addmm": import_addmm,
    "relu": import_unary("relu"),
    "silu": import_unary("silu"),
    "gelu": import_unary("gelu"),
    "gelu_erf": import_unary("gelu_erf"),
    "sigmoid": import_unary("sigmoid"),
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
    "layer_norm": import_layer_norm,
    "rms_norm": import_rms_norm,
    "group_norm": import_group_norm,
    "batch_norm": import_batch_norm,
    "contiguous": import_contiguous,
    "getitem": import_getitem,
}
