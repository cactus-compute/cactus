from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.jax_semantic_rewrites import apply_jax_semantic_rewrites
from cactus.transpile.lower import TranspiledGraph
from cactus.transpile.lower import transpile_ir
from cactus.transpile.weight_binding import resolve_weight_binding


_BINARY_OPS = {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
}

_BINARY_EVALUATORS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
}

_UNARY_OPS = {
    "logistic": "sigmoid",
    "tanh": "tanh",
    "sqrt": "scalar_sqrt",
    "exp": "scalar_exp",
    "log": "scalar_log",
    "cos": "scalar_cos",
    "sin": "scalar_sin",
    "neg": "negate",
    "abs": "abs",
}

_UNARY_EVALUATORS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "cos": np.cos,
    "sin": np.sin,
    "neg": np.negative,
    "logistic": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "tanh": np.tanh,
}

_COMPARE_OPS = {
    "eq": "equal",
    "ne": "not_equal",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
}

_RHS_SCALAR_COMPARE_OPS = {
    "eq": "scalar_equal",
    "ne": "scalar_not_equal",
    "lt": "scalar_less",
    "le": "scalar_less_equal",
    "gt": "scalar_greater",
    "ge": "scalar_greater_equal",
}

_LHS_SCALAR_COMPARE_OPS = {
    "eq": "scalar_equal",
    "ne": "scalar_not_equal",
    "lt": "scalar_greater",
    "le": "scalar_greater_equal",
    "gt": "scalar_less",
    "ge": "scalar_less_equal",
}


def _dtype_to_ir(dtype: Any) -> str:
    dtype = np.dtype(dtype)
    if dtype.name == "bfloat16":
        return "fp32"
    if dtype == np.dtype(np.float16):
        return "fp16"
    if dtype == np.dtype(np.float32):
        return "fp32"
    if dtype == np.dtype(np.float64):
        return "fp32"
    if dtype == np.dtype(np.int8):
        return "int8"
    if dtype in {
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.bool_),
    }:
        return "fp32"
    raise NotImplementedError(f"unsupported JAX dtype: {dtype}")


def _shape_dtype_from_aval(aval: Any) -> tuple[tuple[int, ...] | None, str | None]:
    shape = getattr(aval, "shape", None)
    dtype = getattr(aval, "dtype", None)
    ir_shape = None if shape is None else tuple(int(dim) for dim in shape)
    if ir_shape == ():
        ir_shape = (1,)
    ir_dtype = None if dtype is None else _dtype_to_ir(dtype)
    return ir_shape, ir_dtype


@dataclass(frozen=True)
class _SyntheticAval:
    shape: tuple[int, ...]
    dtype: Any


def _constant_to_torch(value: Any) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype.name == "bfloat16":
        array = array.astype(np.float32)
    if array.dtype == np.dtype(np.float64):
        array = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.floating):
        compare_array = array.astype(np.float32, copy=False)
        array = np.where(compare_array < -1.0e30, -65504.0, array)
        array = np.where(compare_array > 1.0e30, 65504.0, array)
    elif array.dtype in {
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.bool_),
    }:
        array = array.astype(np.float32)
    return torch.from_numpy(np.array(array))


def _tree_path_name(path: Sequence[Any]) -> str:
    parts: list[str] = []
    for entry in path:
        if hasattr(entry, "key"):
            part = str(entry.key)
        elif hasattr(entry, "name"):
            part = str(entry.name)
        elif hasattr(entry, "idx"):
            part = str(entry.idx)
        else:
            part = str(entry)
        parts.append(part)
    return ".".join(parts) or "param"


def _flatten_named_leaves(tree_util: Any, params: Any) -> list[tuple[str, np.ndarray]]:
    leaves: list[tuple[str, np.ndarray]] = []
    for path, value in tree_util.tree_flatten_with_path(params)[0]:
        leaves.append((_tree_path_name(path), np.asarray(value)))
    return leaves


def _weight_binding_fields(meta: dict[str, object]) -> dict[str, object] | None:
    path = meta.get("path")
    kind = meta.get("kind")
    source_name = meta.get("source_name")
    if isinstance(path, str) and isinstance(kind, str) and isinstance(source_name, str):
        return {"path": path, "kind": kind, "source_name": source_name}
    return None


def _propagate_weight_binding_meta(graph: IRGraph) -> None:
    changed = True
    while changed:
        changed = False
        for value_id in list(graph.constants):
            value = graph.values[value_id]
            if _weight_binding_fields(value.meta) is not None:
                continue
            if value.meta.get("derived_by_op") != "convert_element_type":
                continue
            derived_from = value.meta.get("derived_from_value_ids")
            if not isinstance(derived_from, (tuple, list)):
                continue
            for source_id in derived_from:
                source_value = graph.values.get(str(source_id))
                if source_value is None:
                    continue
                binding_meta = _weight_binding_fields(source_value.meta)
                if binding_meta is None:
                    continue
                value.meta.update(binding_meta)
                value.meta["materialized_from_value_id"] = str(source_id)
                graph.meta.setdefault("weight_bindings", {})[value_id] = dict(value.meta)
                changed = True
                break


def _binding_meta(
    *,
    name: str,
    weights_dir: str | None,
    explicit: dict[str, dict[str, str]],
) -> dict[str, str]:
    if name in explicit:
        return dict(explicit[name])
    binding = resolve_weight_binding(weights_dir=weights_dir, source_name=name)
    if binding is None:
        return {}
    return {
        "path": binding.path,
        "kind": binding.kind,
        "source_name": binding.source_name,
    }


def _freeze_leading_param_inputs(
    graph: IRGraph,
    named_leaves: Sequence[tuple[str, np.ndarray]],
    *,
    weights_dir: str | None,
    explicit: dict[str, dict[str, str]],
) -> None:
    if not named_leaves:
        return
    if len(graph.inputs) < len(named_leaves):
        raise ValueError(
            f"JAX graph has {len(graph.inputs)} inputs, cannot freeze {len(named_leaves)} parameter leaves"
        )
    param_input_ids = list(graph.inputs[: len(named_leaves)])
    graph.inputs = graph.inputs[len(named_leaves) :]
    graph.meta["weight_bindings"] = {}
    for value_id, (name, leaf_array) in zip(param_input_ids, named_leaves, strict=True):
        value = graph.values[value_id]
        tensor = _constant_to_torch(leaf_array)
        value.shape = tuple(int(dim) for dim in tensor.shape)
        value.dtype = _dtype_to_ir(tensor.numpy().dtype)
        value.meta.update(
            {
                "source_name": name,
                "jax_param_input": True,
                **_binding_meta(name=name, weights_dir=weights_dir, explicit=explicit),
            }
        )
        graph.constants[value_id] = tensor
        if "path" in value.meta:
            graph.meta["weight_bindings"][value_id] = dict(value.meta)


def _literal_value(literal: Any) -> Any:
    value = getattr(literal, "val", literal)
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


class _JaxImportContext:
    def __init__(self) -> None:
        self.var_ids: dict[int, str] = {}
        self.gather_index_aliases: dict[str, str] = {}
        self.next_value_id = 0
        self.next_node_id = 0

    def value_id(self, var: Any) -> str:
        key = id(var)
        existing = self.var_ids.get(key)
        if existing is not None:
            return existing
        name = f"v{self.next_value_id}"
        self.next_value_id += 1
        self.var_ids[key] = name
        return name

    def bind_value(self, var: Any, value_id: str) -> None:
        self.var_ids[id(var)] = value_id

    def node_id(self, op: str) -> str:
        name = f"n{self.next_node_id}_{op}"
        self.next_node_id += 1
        return name


@dataclass(frozen=True)
class JaxGraphSpec:
    name: str
    fn: Callable[..., Any]
    example_args: Sequence[Any]
    role: str = "generic"
    input_names: Sequence[str] | None = None
    output_names: Sequence[str] | None = None
    graph_meta: dict[str, object] | None = None


@dataclass
class CapturedJaxGraph:
    spec: JaxGraphSpec
    raw_ir_graph: IRGraph
    ir_graph: IRGraph
    graph: TranspiledGraph

    def execute(self, *args: Any) -> list[Any]:
        if len(args) != len(self.spec.example_args):
            raise ValueError(
                f"graph {self.spec.name!r} expected {len(self.spec.example_args)} inputs, got {len(args)}"
            )
        self.graph.set_inputs(
            [
                _coerce_runtime_input(arg, example)
                for arg, example in zip(args, self.spec.example_args, strict=True)
            ]
        )
        return self.graph.execute()


@dataclass
class CapturedJaxGraphBundle:
    graphs: dict[str, CapturedJaxGraph]
    params: Any
    weights_dir: str | None = None

    def execute(self, graph_name: str, *args: Any) -> list[Any]:
        if graph_name not in self.graphs:
            available = ", ".join(sorted(self.graphs)) or "<none>"
            raise ValueError(f"unknown JAX graph {graph_name!r}; available graphs: {available}")
        return self.graphs[graph_name].execute(*args)


def _add_value(graph: IRGraph, value_id: str, aval: Any, *, producer: str | None = None) -> None:
    shape, dtype = _shape_dtype_from_aval(aval)
    graph.add_value(IRValue(id=value_id, shape=shape, dtype=dtype, producer=producer))


def _coerce_runtime_input(value: Any, example: Any) -> np.ndarray:
    array = np.asarray(value)
    example_array = np.asarray(example)
    target_dtype = example_array.dtype
    if target_dtype.name == "bfloat16":
        target_dtype = np.dtype(np.float32)
    if array.dtype != target_dtype:
        array = array.astype(target_dtype, copy=False)
    if array.shape == ():
        array = array.reshape((1,))
    return array


def _add_constant(
    graph: IRGraph,
    ctx: _JaxImportContext,
    var: Any,
    value: Any,
    *,
    source_name: str,
    meta: dict[str, object] | None = None,
) -> str:
    value_id = ctx.value_id(var)
    if value_id in graph.values:
        return value_id
    tensor = _constant_to_torch(value)
    graph.add_value(
        IRValue(
            id=value_id,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=_dtype_to_ir(tensor.numpy().dtype),
            producer=None,
            meta={"source_name": source_name, **dict(meta or {})},
        )
    )
    graph.constants[value_id] = tensor
    return value_id


def _register_node(
    graph: IRGraph,
    node: IRNode,
    *,
    out_avals: Sequence[Any],
) -> None:
    graph.add_node(node)
    graph.order.append(node.id)
    for output_id, aval in zip(node.outputs, out_avals, strict=True):
        shape, dtype = _shape_dtype_from_aval(aval)
        graph.values[output_id].shape = shape
        graph.values[output_id].dtype = dtype


def _primitive_name(eqn: Any) -> str:
    primitive = getattr(eqn, "primitive", None)
    return str(getattr(primitive, "name", primitive))


def _is_literal(var: Any) -> bool:
    return type(var).__name__ == "Literal"


def _ensure_literal_constant(graph: IRGraph, ctx: _JaxImportContext, literal: Any) -> str:
    value_id = ctx.value_id(literal)
    if value_id in graph.values:
        return value_id
    value = _literal_value(literal)
    return _add_constant(graph, ctx, literal, value, source_name=f"literal:{value!r}")


def _derived_meta(input_ids: Sequence[str], *, op: str) -> dict[str, object]:
    if not input_ids:
        return {"derived_by_op": op}
    return {"derived_from_value_ids": tuple(input_ids), "derived_by_op": op}


def _input_ids(graph: IRGraph, ctx: _JaxImportContext, invars: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for var in invars:
        if _is_literal(var):
            result.append(_ensure_literal_constant(graph, ctx, var))
        else:
            result.append(ctx.value_id(var))
    return result


def _out_avals(eqn: Any) -> tuple[Any, ...]:
    return tuple(getattr(var, "aval", None) for var in eqn.outvars)


def _literal_number(var: Any) -> float | None:
    if not _is_literal(var):
        return None
    value = _literal_value(var)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _constant_scalar(graph: IRGraph, value_id: str) -> float | None:
    value = graph.constants.get(value_id)
    if value is None:
        return None
    if graph.values[value_id].meta.get("jax_closed_constant"):
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    return float(array.item())


def _constant_singleton_scalar(graph: IRGraph, value_id: str) -> float | None:
    value = graph.constants.get(value_id)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    return float(array.reshape(()).item())


def _constant_scalar_or_singleton(graph: IRGraph, value_id: str) -> float | None:
    scalar = _constant_scalar(graph, value_id)
    if scalar is not None:
        return scalar
    singleton = _constant_singleton_scalar(graph, value_id)
    if singleton is not None:
        return singleton
    value = graph.constants.get(value_id)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    first = float(array.reshape(-1)[0].item())
    if np.allclose(array, first, rtol=0.0, atol=0.0):
        return first
    return None


def _replace_scalar_constant_with_broadcast(
    graph: IRGraph,
    value_id: str,
    output_var: Any,
    *,
    source_op: str,
) -> str:
    scalar = _constant_singleton_scalar(graph, value_id)
    if scalar is None:
        return value_id
    shape = tuple(int(dim) for dim in getattr(output_var.aval, "shape", ()))
    if not shape or _product(shape) <= 1:
        return value_id
    dtype = getattr(output_var.aval, "dtype", np.float32)
    replacement_id = f"{value_id}__{source_op}_broadcast"
    if replacement_id in graph.values:
        return replacement_id
    tensor = _constant_to_torch(np.full(shape, scalar, dtype=np.dtype(dtype)))
    graph.add_value(
        IRValue(
            id=replacement_id,
            shape=shape,
            dtype=_dtype_to_ir(tensor.numpy().dtype),
            producer=None,
            meta={
                "source_name": f"broadcasted_scalar:{source_op}",
                **_derived_meta((value_id,), op=f"{source_op}:scalar_broadcast"),
            },
        )
    )
    graph.constants[replacement_id] = tensor
    return replacement_id


def _where_scalar_value(value: float) -> float:
    if value < -1.0e30:
        return -65504.0
    if value > 1.0e30:
        return 65504.0
    return value


def _constant_array(graph: IRGraph, value_id: str) -> np.ndarray | None:
    value = graph.constants.get(value_id)
    if value is None:
        return None
    if graph.values[value_id].meta.get("jax_closed_constant"):
        return None
    return np.asarray(value)


def _try_fold_constant_unary(
    graph: IRGraph,
    ctx: _JaxImportContext,
    *,
    prim: str,
    input_id: str,
    outvar: Any,
) -> bool:
    evaluator = _UNARY_EVALUATORS.get(prim)
    array = _constant_array(graph, input_id)
    if evaluator is None or array is None:
        return False
    _add_constant(
        graph,
        ctx,
        outvar,
        evaluator(array),
        source_name=f"folded:{prim}",
        meta=_derived_meta((input_id,), op=prim),
    )
    return True


def _try_fold_constant_binary(
    graph: IRGraph,
    ctx: _JaxImportContext,
    *,
    prim: str,
    input_ids: Sequence[str],
    outvar: Any,
) -> bool:
    evaluator = _BINARY_EVALUATORS.get(prim)
    if evaluator is None:
        return False
    lhs_array = _constant_array(graph, input_ids[0])
    rhs_array = _constant_array(graph, input_ids[1])
    if lhs_array is None or rhs_array is None:
        return False
    _add_constant(
        graph,
        ctx,
        outvar,
        evaluator(lhs_array, rhs_array),
        source_name=f"folded:{prim}",
        meta=_derived_meta(input_ids, op=prim),
    )
    return True


def _alias_output(ctx: _JaxImportContext, outvar: Any, source_id: str) -> None:
    ctx.bind_value(outvar, source_id)


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _conv_static_int(values: Sequence[int], *, name: str) -> int:
    ints = tuple(int(value) for value in values)
    if not ints:
        return 1
    if any(value != ints[0] for value in ints):
        raise NotImplementedError(f"JAX conv_general_dilated requires uniform {name}, got {ints}")
    return ints[0]


def _conv_padding_int(padding: Sequence[Sequence[int]]) -> int:
    pairs = tuple((int(left), int(right)) for left, right in padding)
    if not pairs:
        return 0
    if any(left != right for left, right in pairs):
        raise NotImplementedError(f"JAX conv_general_dilated requires symmetric padding, got {pairs}")
    return _conv_static_int(tuple(left for left, _ in pairs), name="padding")


def _conv_output_permutation(out_spec: Sequence[int]) -> tuple[int, ...]:
    spec = tuple(int(axis) for axis in out_spec)
    return tuple(spec.index(final_axis) for final_axis in range(len(spec)))


def _jaxpr_invars(jaxpr_like: Any) -> Sequence[Any]:
    return tuple(getattr(getattr(jaxpr_like, "jaxpr", jaxpr_like), "invars", ()))


def _jaxpr_outvars(jaxpr_like: Any) -> Sequence[Any]:
    return tuple(getattr(getattr(jaxpr_like, "jaxpr", jaxpr_like), "outvars", ()))


def _jaxpr_eqns(jaxpr_like: Any) -> Sequence[Any]:
    return tuple(getattr(getattr(jaxpr_like, "jaxpr", jaxpr_like), "eqns", ()))


def _jaxpr_constvars(jaxpr_like: Any) -> Sequence[Any]:
    return tuple(getattr(getattr(jaxpr_like, "jaxpr", jaxpr_like), "constvars", ()))


def _jaxpr_consts(jaxpr_like: Any) -> Sequence[Any]:
    return tuple(getattr(jaxpr_like, "consts", ()) or ())


def _inline_jaxpr(
    graph: IRGraph,
    ctx: _JaxImportContext,
    jaxpr_like: Any,
    input_ids: Sequence[str],
    outvars: Sequence[Any],
) -> None:
    inner_output_ids = _inline_jaxpr_outputs(graph, ctx, jaxpr_like, input_ids)
    if len(outvars) != len(inner_output_ids):
        raise NotImplementedError("nested JAXPR output arity mismatch")
    for outer_var, inner_output_id in zip(outvars, inner_output_ids, strict=True):
        _alias_output(ctx, outer_var, inner_output_id)


def _inline_jaxpr_outputs(
    graph: IRGraph,
    ctx: _JaxImportContext,
    jaxpr_like: Any,
    input_ids: Sequence[str],
) -> list[str]:
    invars = _jaxpr_invars(jaxpr_like)
    if len(input_ids) != len(invars):
        raise NotImplementedError("nested JAXPR input arity mismatch")
    invar_keys = {id(var) for var in invars}
    for inner_eqn in _jaxpr_eqns(jaxpr_like):
        for outvar in inner_eqn.outvars:
            if id(outvar) not in invar_keys:
                ctx.var_ids.pop(id(outvar), None)
    for inner_var, input_id in zip(invars, input_ids, strict=True):
        ctx.bind_value(inner_var, input_id)
    for index, (constvar, const) in enumerate(zip(_jaxpr_constvars(jaxpr_like), _jaxpr_consts(jaxpr_like), strict=True)):
        _add_constant(graph, ctx, constvar, const, source_name=f"nested_const_{index}")
    for inner_eqn in _jaxpr_eqns(jaxpr_like):
        _import_eqn(graph, ctx, inner_eqn)
    return [ctx.value_id(inner_var) for inner_var in _jaxpr_outvars(jaxpr_like)]


def _generated_value(
    graph: IRGraph,
    ctx: _JaxImportContext,
    *,
    stem: str,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    producer: str,
    meta: dict[str, object] | None = None,
) -> str:
    value_id = f"v{ctx.next_value_id}_{stem}"
    ctx.next_value_id += 1
    if value_id in graph.values:
        raise ValueError(f"duplicate generated IR value id: {value_id}")
    graph.values[value_id] = IRValue(
        id=value_id,
        shape=shape,
        dtype=dtype,
        producer=producer,
        meta=dict(meta or {}),
    )
    return value_id


def _register_generated_node(
    graph: IRGraph,
    node: IRNode,
    *,
    output_shapes: Sequence[tuple[int, ...] | None],
    output_dtypes: Sequence[str | None],
) -> None:
    if node.id in graph.nodes:
        raise ValueError(f"duplicate IR node id: {node.id}")
    graph.nodes[node.id] = node
    graph.order.append(node.id)
    for output_id, shape, dtype in zip(node.outputs, output_shapes, output_dtypes, strict=True):
        graph.values[output_id].shape = shape
        graph.values[output_id].dtype = dtype


def _scan_slice_input(
    graph: IRGraph,
    ctx: _JaxImportContext,
    value_id: str,
    *,
    iteration: int,
    input_index: int,
) -> str:
    source_value = graph.values[value_id]
    if source_value.shape is None or not source_value.shape:
        return value_id
    slice_node_id = ctx.node_id("scan_slice")
    sliced_shape = (1, *tuple(int(dim) for dim in source_value.shape[1:]))
    sliced_id = _generated_value(
        graph,
        ctx,
        stem=f"scan{iteration}_{input_index}",
        shape=sliced_shape,
        dtype=source_value.dtype,
        producer=slice_node_id,
        meta={
            "derived_from_value_ids": (value_id,),
            "derived_by_op": "scan_slice",
        },
    )
    slice_node = IRNode(
        slice_node_id,
        "slice",
        [value_id],
        [sliced_id],
        attrs={"axis": 0, "start": iteration, "end": iteration + 1, "step": 1},
        meta={"jax_generated": "scan_unroll"},
    )
    _register_generated_node(graph, slice_node, output_shapes=(sliced_shape,), output_dtypes=(source_value.dtype,))
    view_node_id = ctx.node_id("scan_slice_view")
    output_shape = tuple(int(dim) for dim in source_value.shape[1:])
    if output_shape == () and sliced_shape == (1,):
        return sliced_id
    output_id = _generated_value(
        graph,
        ctx,
        stem=f"scan{iteration}_{input_index}_view",
        shape=output_shape,
        dtype=source_value.dtype,
        producer=view_node_id,
        meta={
            "derived_from_value_ids": (sliced_id,),
            "derived_by_op": "scan_slice_view",
        },
    )
    view_node = IRNode(
        view_node_id,
        "view",
        [sliced_id],
        [output_id],
        attrs={"shape": output_shape},
        meta={"jax_generated": "scan_unroll"},
    )
    _register_generated_node(graph, view_node, output_shapes=(output_shape,), output_dtypes=(source_value.dtype,))
    return output_id


def _add_iota_constant(graph: IRGraph, ctx: _JaxImportContext, eqn: Any) -> None:
    params = dict(getattr(eqn, "params", {}) or {})
    shape = tuple(int(dim) for dim in params["shape"])
    dimension = int(params["dimension"])
    dtype = params.get("dtype", getattr(eqn.outvars[0].aval, "dtype", np.float32))
    values = np.arange(shape[dimension], dtype=np.dtype(dtype))
    reshape = [1] * len(shape)
    reshape[dimension] = shape[dimension]
    values = np.broadcast_to(values.reshape(reshape), shape)
    _add_constant(graph, ctx, eqn.outvars[0], values, source_name=f"iota:{shape}:{dimension}")


def _import_eqn(graph: IRGraph, ctx: _JaxImportContext, eqn: Any) -> None:
    prim = _primitive_name(eqn)
    params = dict(getattr(eqn, "params", {}) or {})

    if prim in {"jit", "remat2"}:
        _inline_jaxpr(graph, ctx, params["jaxpr"], _input_ids(graph, ctx, eqn.invars), eqn.outvars)
        return

    if prim in {"custom_jvp_call", "custom_jvp_call_jaxpr"}:
        call_jaxpr = params.get("call_jaxpr") or params.get("fun_jaxpr") or params.get("jaxpr")
        if call_jaxpr is None:
            raise NotImplementedError("JAX custom_jvp_call import requires a primal call_jaxpr")
        _inline_jaxpr(graph, ctx, call_jaxpr, _input_ids(graph, ctx, eqn.invars), eqn.outvars)
        return

    if prim == "scan":
        length = int(params.get("length", 1))
        num_consts = int(params.get("num_consts", 0))
        num_carry = int(params.get("num_carry", len(eqn.outvars)))
        input_ids = _input_ids(graph, ctx, eqn.invars)
        const_ids = input_ids[:num_consts]
        carry_ids = input_ids[num_consts : num_consts + num_carry]
        xs_ids = input_ids[num_consts + num_carry :]
        if len(eqn.outvars) != num_carry:
            raise NotImplementedError("JAX scan import currently supports carry-only scan outputs")
        for iteration in range(length):
            sliced_xs = [
                _scan_slice_input(graph, ctx, value_id, iteration=iteration, input_index=index)
                for index, value_id in enumerate(xs_ids)
            ]
            body_outputs = _inline_jaxpr_outputs(graph, ctx, params["jaxpr"], [*const_ids, *carry_ids, *sliced_xs])
            if len(body_outputs) != num_carry:
                raise NotImplementedError("JAX scan body output arity mismatch")
            carry_ids = body_outputs
        for outvar, carry_id in zip(eqn.outvars, carry_ids, strict=True):
            _alias_output(ctx, outvar, carry_id)
        return

    if prim == "stop_gradient":
        _alias_output(ctx, eqn.outvars[0], _input_ids(graph, ctx, eqn.invars)[0])
        return

    if prim == "iota":
        _add_iota_constant(graph, ctx, eqn)
        return

    node = _node_for_eqn(graph, ctx, eqn)
    if node is None:
        return
    _register_node(graph, node, out_avals=_out_avals(eqn))


def _node_for_eqn(graph: IRGraph, ctx: _JaxImportContext, eqn: Any) -> IRNode | None:
    prim = _primitive_name(eqn)
    inputs = _input_ids(graph, ctx, eqn.invars)
    outputs = [ctx.value_id(var) for var in eqn.outvars]
    node_id = ctx.node_id(prim)
    params = dict(getattr(eqn, "params", {}) or {})
    meta = {"jax_primitive": prim}

    if prim in _BINARY_OPS:
        if _try_fold_constant_binary(graph, ctx, prim=prim, input_ids=inputs, outvar=eqn.outvars[0]):
            return None
    if prim == "div":
        lhs_literal = _literal_number(eqn.invars[0])
        if lhs_literal is not None:
            reciprocal_id = f"{outputs[0]}__reciprocal"
            reciprocal_node = IRNode(
                node_id,
                "pow",
                [inputs[1]],
                [reciprocal_id],
                attrs={"exponent": -1.0},
                meta=meta,
            )
            _register_node(graph, reciprocal_node, out_avals=(eqn.outvars[0].aval,))
            if lhs_literal == 1.0:
                _alias_output(ctx, eqn.outvars[0], reciprocal_id)
                return None
            return IRNode(
                ctx.node_id("scalar_div_mul"),
                "scalar_multiply",
                [reciprocal_id],
                outputs,
                attrs={"value": lhs_literal},
                meta=meta,
            )
    if prim in _BINARY_OPS:
        if len(inputs) == 2 and len(outputs) == 1:
            inputs = [
                _replace_scalar_constant_with_broadcast(graph, input_id, eqn.outvars[0], source_op=prim)
                for input_id in inputs
            ]
        return IRNode(node_id, _BINARY_OPS[prim], inputs, outputs, meta=meta)
    if prim in _UNARY_OPS:
        if _try_fold_constant_unary(graph, ctx, prim=prim, input_id=inputs[0], outvar=eqn.outvars[0]):
            return None
        return IRNode(node_id, _UNARY_OPS[prim], inputs, outputs, meta=meta)
    if prim == "log1p":
        add_id = f"{outputs[0]}__log1p_add"
        _register_node(
            graph,
            IRNode(
                node_id,
                "scalar_add",
                inputs,
                [add_id],
                attrs={"value": 1.0},
                meta=meta,
            ),
            out_avals=(eqn.outvars[0].aval,),
        )
        return IRNode(ctx.node_id("log1p_log"), "scalar_log", [add_id], outputs, meta=meta)
    if prim == "expm1":
        exp_id = f"{outputs[0]}__expm1_exp"
        _register_node(
            graph,
            IRNode(node_id, "scalar_exp", inputs, [exp_id], meta=meta),
            out_avals=(eqn.outvars[0].aval,),
        )
        return IRNode(ctx.node_id("expm1_sub"), "scalar_add", [exp_id], outputs, attrs={"value": -1.0}, meta=meta)
    if prim in _COMPARE_OPS:
        lhs_scalar = _constant_scalar(graph, inputs[0])
        rhs_scalar = _constant_scalar(graph, inputs[1])
        if rhs_scalar is not None:
            return IRNode(node_id, _RHS_SCALAR_COMPARE_OPS[prim], [inputs[0]], outputs, attrs={"value": rhs_scalar}, meta=meta)
        if lhs_scalar is not None:
            return IRNode(node_id, _LHS_SCALAR_COMPARE_OPS[prim], [inputs[1]], outputs, attrs={"value": lhs_scalar}, meta=meta)
        return IRNode(node_id, _COMPARE_OPS[prim], inputs, outputs, meta=meta)
    if prim == "and":
        return IRNode(node_id, "logical_and", inputs, outputs, meta=meta)
    if prim == "or":
        return IRNode(node_id, "logical_or", inputs, outputs, meta=meta)
    if prim == "square":
        return IRNode(node_id, "multiply", [inputs[0], inputs[0]], outputs, meta=meta)
    if prim == "rsqrt":
        return IRNode(node_id, "pow", inputs, outputs, attrs={"exponent": -0.5}, meta=meta)
    if prim == "integer_pow":
        exponent = int(params["y"])
        if exponent == 2:
            return IRNode(node_id, "multiply", [inputs[0], inputs[0]], outputs, meta=meta)
        return IRNode(node_id, "pow", inputs, outputs, attrs={"exponent": float(exponent)}, meta=meta)
    if prim == "pow":
        rhs_literal = _literal_number(eqn.invars[1])
        if rhs_literal is not None:
            return IRNode(node_id, "pow", [inputs[0]], outputs, attrs={"exponent": float(rhs_literal)}, meta=meta)
        lhs_literal = _literal_number(eqn.invars[0])
        if lhs_literal is not None and lhs_literal > 0.0:
            intermediate = f"{outputs[0]}__logmul"
            scale_node = IRNode(
                node_id,
                "scalar_multiply",
                [inputs[1]],
                [intermediate],
                attrs={"value": math.log(lhs_literal)},
                meta=meta,
            )
            _register_node(graph, scale_node, out_avals=(eqn.outvars[0].aval,))
            return IRNode(ctx.node_id("pow_exp"), "scalar_exp", [intermediate], outputs, meta=meta)
        raise NotImplementedError("JAX pow import only supports positive scalar base")
    if prim == "dot_general":
        dimension_numbers = params.get("dimension_numbers")
        if dimension_numbers is not None:
            ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
            lhs_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
            rhs_shape = tuple(getattr(eqn.invars[1].aval, "shape", ()))
            lhs_contract = tuple(int(dim) for dim in lhs_contract)
            rhs_contract = tuple(int(dim) for dim in rhs_contract)
            lhs_batch = tuple(int(dim) for dim in lhs_batch)
            rhs_batch = tuple(int(dim) for dim in rhs_batch)
            if lhs_batch or rhs_batch:
                if tuple(lhs_shape[dim] for dim in lhs_batch) != tuple(rhs_shape[dim] for dim in rhs_batch):
                    raise NotImplementedError("JAX dot_general batch dimensions must have matching sizes")
            expected_rhs_contract = (len(rhs_batch),)
            if (
                tuple(lhs_contract) != (len(lhs_shape) - 1,)
                or tuple(rhs_contract) != expected_rhs_contract
                or len(rhs_shape) > 2
            ):
                batch_shape = tuple(lhs_shape[dim] for dim in lhs_batch)
                lhs_contract_shape = tuple(lhs_shape[dim] for dim in lhs_contract)
                rhs_contract_shape = tuple(rhs_shape[dim] for dim in rhs_contract)
                if lhs_contract_shape != rhs_contract_shape:
                    raise NotImplementedError("JAX dot_general contraction dimensions must have matching sizes")
                lhs_non_contract = tuple(
                    dim for dim in range(len(lhs_shape)) if dim not in set(lhs_batch) | set(lhs_contract)
                )
                rhs_non_contract = tuple(
                    dim for dim in range(len(rhs_shape)) if dim not in set(rhs_batch) | set(rhs_contract)
                )
                lhs_order = lhs_batch + lhs_non_contract + lhs_contract
                rhs_order = rhs_batch + rhs_contract + rhs_non_contract
                lhs_non_shape = tuple(lhs_shape[dim] for dim in lhs_non_contract)
                rhs_non_shape = tuple(rhs_shape[dim] for dim in rhs_non_contract)
                lhs_matrix_shape = batch_shape + (_product(lhs_non_shape), _product(lhs_contract_shape))
                rhs_matrix_shape = batch_shape + (_product(rhs_contract_shape), _product(rhs_non_shape))
                output_shape = batch_shape + lhs_non_shape + rhs_non_shape
                dtype = getattr(eqn.outvars[0].aval, "dtype", getattr(eqn.invars[0].aval, "dtype", np.float32))

                lhs_id = inputs[0]
                if lhs_order != tuple(range(len(lhs_shape))):
                    lhs_id = f"{outputs[0]}__lhs_transpose"
                    _register_node(
                        graph,
                        IRNode(
                            ctx.node_id("dot_general_lhs_transpose"),
                            "permute",
                            [inputs[0]],
                            [lhs_id],
                            attrs={"permutation": lhs_order},
                            meta=meta,
                        ),
                        out_avals=(_SyntheticAval(tuple(lhs_shape[dim] for dim in lhs_order), dtype),),
                    )
                lhs_2d = f"{outputs[0]}__lhs_reshape"
                _register_node(
                    graph,
                    IRNode(
                        ctx.node_id("dot_general_lhs_reshape"),
                        "reshape",
                        [lhs_id],
                        [lhs_2d],
                        attrs={"shape": lhs_matrix_shape},
                        meta=meta,
                    ),
                    out_avals=(_SyntheticAval(lhs_matrix_shape, dtype),),
                )

                rhs_id = inputs[1]
                if rhs_order != tuple(range(len(rhs_shape))):
                    rhs_id = f"{outputs[0]}__rhs_transpose"
                    _register_node(
                        graph,
                        IRNode(
                            ctx.node_id("dot_general_rhs_transpose"),
                            "permute",
                            [inputs[1]],
                            [rhs_id],
                            attrs={"permutation": rhs_order},
                            meta=meta,
                        ),
                        out_avals=(_SyntheticAval(tuple(rhs_shape[dim] for dim in rhs_order), dtype),),
                    )
                rhs_2d = f"{outputs[0]}__rhs_reshape"
                _register_node(
                    graph,
                    IRNode(
                        ctx.node_id("dot_general_rhs_reshape"),
                        "reshape",
                        [rhs_id],
                        [rhs_2d],
                        attrs={"shape": rhs_matrix_shape},
                        meta=meta,
                    ),
                    out_avals=(_SyntheticAval(rhs_matrix_shape, dtype),),
                )

                matmul_id = f"{outputs[0]}__matmul"
                _register_node(
                    graph,
                    IRNode(ctx.node_id("dot_general_matmul"), "matmul", [lhs_2d, rhs_2d], [matmul_id], meta=meta),
                    out_avals=(_SyntheticAval(batch_shape + (lhs_matrix_shape[-2], rhs_matrix_shape[-1]), dtype),),
                )
                return IRNode(
                    ctx.node_id("dot_general_output_reshape"),
                    "reshape",
                    [matmul_id],
                    outputs,
                    attrs={"shape": output_shape},
                    meta=meta,
                )
        return IRNode(node_id, "matmul", inputs, outputs, meta=meta)
    if prim == "conv_general_dilated":
        if len(inputs) != 2 or len(outputs) != 1:
            raise NotImplementedError("JAX conv_general_dilated import supports input and kernel only")
        if int(params.get("batch_group_count", 1)) != 1:
            raise NotImplementedError("JAX conv_general_dilated batch_group_count is unsupported")
        lhs_dilation = tuple(int(value) for value in params.get("lhs_dilation", ()))
        if any(value != 1 for value in lhs_dilation):
            raise NotImplementedError(f"JAX conv_general_dilated lhs_dilation is unsupported: {lhs_dilation}")
        dimension_numbers = params.get("dimension_numbers")
        if dimension_numbers is None:
            raise NotImplementedError("JAX conv_general_dilated requires static dimension_numbers")

        lhs_spec = tuple(int(axis) for axis in dimension_numbers.lhs_spec)
        rhs_spec = tuple(int(axis) for axis in dimension_numbers.rhs_spec)
        out_spec = tuple(int(axis) for axis in dimension_numbers.out_spec)
        lhs_shape = tuple(int(dim) for dim in getattr(eqn.invars[0].aval, "shape", ()))
        rhs_shape = tuple(int(dim) for dim in getattr(eqn.invars[1].aval, "shape", ()))
        out_shape = tuple(int(dim) for dim in getattr(eqn.outvars[0].aval, "shape", ()))
        rank = len(lhs_shape)
        if rank not in {3, 4}:
            raise NotImplementedError(f"JAX conv_general_dilated supports 1D/2D convs, got rank {rank}")
        if tuple(sorted(lhs_spec)) != tuple(range(rank)) or tuple(sorted(rhs_spec)) != tuple(range(rank)):
            raise NotImplementedError("JAX conv_general_dilated requires complete dimension specs")
        if tuple(sorted(out_spec)) != tuple(range(rank)):
            raise NotImplementedError("JAX conv_general_dilated requires complete output dimension spec")

        stride = _conv_static_int(params.get("window_strides", (1,) * (rank - 2)), name="stride")
        dilation = _conv_static_int(params.get("rhs_dilation", (1,) * (rank - 2)), name="rhs_dilation")
        padding = _conv_padding_int(params.get("padding", ((0, 0),) * (rank - 2)))
        groups = int(params.get("feature_group_count", 1))
        dtype = getattr(eqn.outvars[0].aval, "dtype", getattr(eqn.invars[0].aval, "dtype", np.float32))

        x_id = inputs[0]
        x_shape = tuple(lhs_shape[axis] for axis in lhs_spec)
        if lhs_spec != tuple(range(rank)):
            x_id = f"{outputs[0]}__conv_lhs"
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("conv_lhs_transpose"),
                    "permute",
                    [inputs[0]],
                    [x_id],
                    attrs={"permutation": lhs_spec},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(x_shape, dtype),),
            )

        weight_id = inputs[1]
        weight_shape = tuple(rhs_shape[axis] for axis in rhs_spec)
        if rhs_spec != tuple(range(rank)):
            weight_id = f"{outputs[0]}__conv_rhs"
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("conv_rhs_transpose"),
                    "permute",
                    [inputs[1]],
                    [weight_id],
                    attrs={"permutation": rhs_spec},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(weight_shape, dtype),),
            )

        conv_id = outputs[0]
        output_permutation = _conv_output_permutation(out_spec)
        conv_shape = tuple(out_shape[axis] for axis in out_spec)
        if output_permutation != tuple(range(rank)):
            conv_id = f"{outputs[0]}__conv"
        _register_node(
            graph,
            IRNode(
                node_id,
                "conv1d" if rank == 3 else "conv2d",
                [x_id, weight_id],
                [conv_id],
                attrs={"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
                meta=meta,
            ),
            out_avals=(_SyntheticAval(conv_shape, dtype),),
        )
        if conv_id == outputs[0]:
            return None
        return IRNode(
            ctx.node_id("conv_output_transpose"),
            "permute",
            [conv_id],
            outputs,
            attrs={"permutation": output_permutation},
            meta=meta,
        )
    if prim == "reshape":
        if inputs[0] in ctx.gather_index_aliases:
            ctx.gather_index_aliases[outputs[0]] = ctx.gather_index_aliases[inputs[0]]
        new_shape = tuple(int(dim) for dim in params["new_sizes"])
        input_shape = tuple(int(dim) for dim in getattr(eqn.invars[0].aval, "shape", ()))
        source_value = graph.values.get(inputs[0])
        graph_input_shape = tuple(int(dim) for dim in (source_value.shape or ())) if source_value is not None else input_shape
        if new_shape == () and graph_input_shape == (1,):
            _alias_output(ctx, eqn.outvars[0], inputs[0])
            return None
        return IRNode(
            node_id,
            "reshape",
            inputs,
            outputs,
            attrs={"shape": new_shape},
            meta=meta,
        )
    if prim == "squeeze":
        shape = tuple(int(dim) for dim in getattr(eqn.outvars[0].aval, "shape", ()))
        return IRNode(node_id, "reshape", inputs, outputs, attrs={"shape": shape}, meta=meta)
    if prim == "transpose":
        permutation = params.get("permutation")
        if permutation is None:
            permutation = params.get("permutation_or_none")
        if permutation is None:
            raise NotImplementedError("JAX transpose missing permutation")
        return IRNode(
            node_id,
            "permute",
            inputs,
            outputs,
            attrs={"permutation": tuple(int(dim) for dim in permutation)},
            meta=meta,
        )
    if prim == "convert_element_type":
        array = _constant_array(graph, inputs[0])
        if array is not None:
            _add_constant(
                graph,
                ctx,
                eqn.outvars[0],
                array.astype(np.dtype(params["new_dtype"])),
                source_name="folded:convert_element_type",
                meta=_derived_meta(inputs, op=prim),
            )
            return None
        return IRNode(
            node_id,
            "precision_cast",
            inputs,
            outputs,
            attrs={"dtype": _dtype_to_ir(params["new_dtype"])},
            meta=meta,
        )
    if prim == "broadcast_in_dim":
        shape = tuple(int(dim) for dim in params["shape"])
        scalar = _constant_singleton_scalar(graph, inputs[0])
        if scalar is not None:
            _add_constant(
                graph,
                ctx,
                eqn.outvars[0],
                np.full(shape, scalar, dtype=np.float32),
                source_name="folded:broadcast_in_dim",
                meta=_derived_meta(inputs, op=prim),
            )
            return None
        if inputs[0] in ctx.gather_index_aliases:
            ctx.gather_index_aliases[outputs[0]] = ctx.gather_index_aliases[inputs[0]]
        broadcast_dimensions = tuple(int(dim) for dim in params.get("broadcast_dimensions", ()))
        input_shape = tuple(graph.values[inputs[0]].shape or ())
        if broadcast_dimensions and broadcast_dimensions != tuple(range(len(input_shape))):
            reshape_shape = [1] * len(shape)
            for input_axis, output_axis in enumerate(broadcast_dimensions):
                reshape_shape[output_axis] = input_shape[input_axis]
            reshaped_shape = tuple(reshape_shape)
            if reshaped_shape == shape:
                return IRNode(node_id, "view", inputs, outputs, attrs={"shape": reshaped_shape}, meta=meta)
            reshaped_id = _generated_value(
                graph,
                ctx,
                stem="broadcast_base",
                shape=reshaped_shape,
                dtype=graph.values[inputs[0]].dtype,
                producer=node_id,
                meta=_derived_meta(inputs, op=f"{prim}:reshape"),
            )
            reshape_node = IRNode(node_id, "view", inputs, [reshaped_id], attrs={"shape": reshaped_shape}, meta=meta)
            _register_generated_node(
                graph,
                reshape_node,
                output_shapes=(reshaped_shape,),
                output_dtypes=(graph.values[inputs[0]].dtype,),
            )
            node_id = ctx.node_id(prim)
            return IRNode(node_id, "expand", [reshaped_id], outputs, attrs={"shape": shape}, meta=meta)
        return IRNode(node_id, "expand", inputs, outputs, attrs={"shape": shape}, meta=meta)
    if prim == "concatenate":
        return IRNode(node_id, "cat", inputs, outputs, attrs={"axis": int(params["dimension"])}, meta=meta)
    if prim == "stack":
        axis = int(params.get("axis", 0))
        input_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
        rank = len(input_shape) + 1
        if axis < 0:
            axis += rank
        if axis < 0 or axis > len(input_shape):
            raise NotImplementedError(f"JAX stack axis out of range: {axis}")
        dtype = getattr(eqn.outvars[0].aval, "dtype", getattr(eqn.invars[0].aval, "dtype", np.float32))
        expanded_inputs: list[str] = []
        for input_index, input_id in enumerate(inputs):
            expanded_id = f"{outputs[0]}__stack_{input_index}"
            expanded_shape = input_shape[:axis] + (1,) + input_shape[axis:]
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("stack_view"),
                    "view",
                    [input_id],
                    [expanded_id],
                    attrs={"shape": expanded_shape},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(expanded_shape, dtype),),
            )
            expanded_inputs.append(expanded_id)
        return IRNode(node_id, "cat", expanded_inputs, outputs, attrs={"axis": axis}, meta=meta)
    if prim == "tile":
        reps = tuple(int(value) for value in params.get("reps", ()))
        if any(rep <= 0 for rep in reps):
            raise NotImplementedError(f"JAX tile import requires positive reps, got {reps}")
        input_shape = tuple(int(dim) for dim in getattr(eqn.invars[0].aval, "shape", ()))
        if len(reps) < len(input_shape):
            reps = (1,) * (len(input_shape) - len(reps)) + reps
        current_id = inputs[0]
        current_shape = input_shape
        dtype = getattr(eqn.invars[0].aval, "dtype", getattr(eqn.outvars[0].aval, "dtype", np.float32))
        if len(reps) > len(current_shape):
            current_shape = (1,) * (len(reps) - len(current_shape)) + current_shape
            viewed_id = f"{outputs[0]}__tile_view"
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("tile_view"),
                    "view",
                    [current_id],
                    [viewed_id],
                    attrs={"shape": current_shape},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(current_shape, dtype),),
            )
            current_id = viewed_id
        for axis, rep in enumerate(reps):
            if rep == 1:
                continue
            next_id = outputs[0] if axis == len(reps) - 1 and all(factor == 1 for factor in reps[axis + 1 :]) else f"{outputs[0]}__tile_axis_{axis}"
            next_shape = current_shape[:axis] + (current_shape[axis] * rep,) + current_shape[axis + 1 :]
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("tile_cat"),
                    "cat",
                    [current_id] * rep,
                    [next_id],
                    attrs={"axis": axis},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(next_shape, dtype),),
            )
            current_id = next_id
            current_shape = next_shape
        if current_id != outputs[0]:
            _alias_output(ctx, eqn.outvars[0], current_id)
        return None
    if prim == "split":
        axis = int(params.get("axis", 0))
        start = 0
        for output_id, output_var, size in zip(outputs, eqn.outvars, params.get("sizes", ()), strict=True):
            end = start + int(size)
            _register_node(
                graph,
                IRNode(
                    ctx.node_id("split_slice"),
                    "slice",
                    inputs,
                    [output_id],
                    attrs={"axis": axis, "start": start, "end": end, "step": 1},
                    meta=meta,
                ),
                out_avals=(output_var.aval,),
            )
            start = end
        return None
    if prim == "slice":
        starts = tuple(int(value) for value in params["start_indices"])
        limits = tuple(int(value) for value in params["limit_indices"])
        raw_strides = params.get("strides")
        strides = (1,) * len(starts) if raw_strides is None else tuple(int(value) for value in raw_strides)
        input_shape = getattr(eqn.invars[0].aval, "shape", ())
        changed_axes = [axis for axis, (start, limit, stride) in enumerate(zip(starts, limits, strides, strict=True)) if start != 0 or limit != input_shape[axis] or stride != 1]
        if not changed_axes:
            _alias_output(ctx, eqn.outvars[0], inputs[0])
            return None
        if len(changed_axes) != 1:
            raise NotImplementedError("JAX slice import supports one non-trivial axis")
        axis = changed_axes[0]
        return IRNode(
            node_id,
            "slice",
            inputs,
            outputs,
            attrs={"axis": axis, "start": starts[axis], "end": limits[axis], "step": strides[axis]},
            meta=meta,
        )
    if prim == "select_n":
        if len(inputs) != 3:
            raise NotImplementedError("JAX select_n import supports ternary select")
        ctx.gather_index_aliases[outputs[0]] = inputs[1]
        false_scalar = _constant_scalar(graph, inputs[1])
        true_scalar = _constant_scalar(graph, inputs[2])
        where_inputs = [inputs[0]]
        attrs: dict[str, object] = {}
        if true_scalar is None:
            where_inputs.append(inputs[2])
        else:
            attrs["true_is_scalar"] = True
            attrs["true_value"] = _where_scalar_value(true_scalar)
        if false_scalar is None:
            where_inputs.append(inputs[1])
        else:
            attrs["false_is_scalar"] = True
            attrs["false_value"] = _where_scalar_value(false_scalar)
        return IRNode(node_id, "where", where_inputs, outputs, attrs=attrs, meta=meta)
    if prim == "gather":
        if len(inputs) != 2:
            raise NotImplementedError("JAX gather import supports embedding-style gather")
        indices_id = ctx.gather_index_aliases.get(inputs[1], inputs[1])
        index_shape = tuple(getattr(eqn.invars[1].aval, "shape", ()))
        if index_shape and index_shape[-1] == 1:
            squeezed_id = f"{indices_id}__squeezed"
            squeeze_shape = tuple(int(dim) for dim in index_shape[:-1])
            if squeezed_id not in graph.values:
                squeeze_node = IRNode(
                    node_id,
                    "reshape",
                    [indices_id],
                    [squeezed_id],
                    attrs={"shape": squeeze_shape},
                    meta=meta,
                )
                _register_node(graph, squeeze_node, out_avals=(eqn.invars[1].aval,))
                graph.values[squeezed_id].shape = squeeze_shape
                if indices_id in ctx.gather_index_aliases:
                    ctx.gather_index_aliases[squeezed_id] = ctx.gather_index_aliases[indices_id]
            indices_id = squeezed_id
            node_id = ctx.node_id("gather_embedding")
        indices_id = ctx.gather_index_aliases.get(indices_id, indices_id)
        return IRNode(node_id, "embedding", [inputs[0], indices_id], outputs, meta=meta)
    if prim in {"max", "min"}:
        lhs_literal = _literal_number(eqn.invars[0])
        rhs_literal = _literal_number(eqn.invars[1])
        identity = -math.inf if prim == "max" else math.inf
        if lhs_literal == identity:
            _alias_output(ctx, eqn.outvars[0], inputs[1])
            return None
        if rhs_literal == identity:
            _alias_output(ctx, eqn.outvars[0], inputs[0])
            return None

        lhs_scalar = _constant_scalar(graph, inputs[0])
        rhs_scalar = _constant_scalar(graph, inputs[1])
        condition_id = f"{outputs[0]}__{prim}_condition"
        condition_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
        condition_dtype = np.bool_
        if rhs_scalar is not None:
            compare_op = "scalar_greater" if prim == "max" else "scalar_less"
            _register_node(
                graph,
                IRNode(
                    ctx.node_id(f"{prim}_condition"),
                    compare_op,
                    [inputs[0]],
                    [condition_id],
                    attrs={"value": rhs_scalar},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(condition_shape, condition_dtype),),
            )
            attrs = {"false_is_scalar": True, "false_value": _where_scalar_value(rhs_scalar)}
            return IRNode(ctx.node_id(f"{prim}_where"), "where", [condition_id, inputs[0]], outputs, attrs=attrs, meta=meta)
        if lhs_scalar is not None:
            compare_op = "scalar_less" if prim == "max" else "scalar_greater"
            _register_node(
                graph,
                IRNode(
                    ctx.node_id(f"{prim}_condition"),
                    compare_op,
                    [inputs[1]],
                    [condition_id],
                    attrs={"value": lhs_scalar},
                    meta=meta,
                ),
                out_avals=(_SyntheticAval(condition_shape, condition_dtype),),
            )
            attrs = {"true_is_scalar": True, "true_value": _where_scalar_value(lhs_scalar)}
            return IRNode(ctx.node_id(f"{prim}_where"), "where", [condition_id, inputs[1]], outputs, attrs=attrs, meta=meta)

        compare_op = "greater" if prim == "max" else "less"
        _register_node(
            graph,
            IRNode(ctx.node_id(f"{prim}_condition"), compare_op, inputs, [condition_id], meta=meta),
            out_avals=(_SyntheticAval(condition_shape, condition_dtype),),
        )
        return IRNode(ctx.node_id(f"{prim}_where"), "where", [condition_id, inputs[0], inputs[1]], outputs, meta=meta)
    if prim in {"reduce_sum", "reduce_max", "reduce_min"}:
        reduce_ops = {
            "reduce_sum": "sum",
            "reduce_max": "max",
            "reduce_min": "min",
        }
        return IRNode(
            node_id,
            reduce_ops[prim],
            inputs,
            outputs,
            attrs={"axis": tuple(int(axis) for axis in params.get("axes", ()))},
            meta=meta,
        )

    raise NotImplementedError(f"unsupported JAX primitive: {prim}")


def capture_jax_function(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    constant_names: Sequence[str] | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    weights_dir: str | None = None,
    graph_meta: dict[str, object] | None = None,
) -> IRGraph:
    try:
        import jax
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("capture_jax_function requires the optional jax package") from exc

    closed = jax.make_jaxpr(fn)(*example_args)
    jaxpr = closed.jaxpr
    constants = tuple(getattr(closed, "consts", ()) or ())
    names = tuple(constant_names or ())
    bindings = weight_bindings or {}
    ctx = _JaxImportContext()
    graph = IRGraph(
        values={},
        nodes={},
        order=[],
        inputs=[],
        outputs=[],
        constants={},
        meta={
            "frontend": "jax",
            "adapter_family": "generic",
            **({"weights_dir": weights_dir} if weights_dir else {}),
            **dict(graph_meta or {}),
        },
    )

    for var in jaxpr.invars:
        value_id = ctx.value_id(var)
        _add_value(graph, value_id, var.aval)
        graph.inputs.append(value_id)

    for index, (var, value) in enumerate(zip(jaxpr.constvars, constants, strict=True)):
        value_id = ctx.value_id(var)
        name = names[index] if index < len(names) else f"const_{index}"
        tensor = _constant_to_torch(value)
        graph.add_value(
            IRValue(
                id=value_id,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_dtype_to_ir(tensor.numpy().dtype),
                meta={
                    "source_name": name,
                    "jax_closed_constant": True,
                    **_binding_meta(name=name, weights_dir=weights_dir, explicit=bindings),
                },
            )
        )
        graph.constants[value_id] = tensor
        if "path" in graph.values[value_id].meta:
            graph.meta.setdefault("weight_bindings", {})[value_id] = dict(graph.values[value_id].meta)

    for eqn in jaxpr.eqns:
        _import_eqn(graph, ctx, eqn)

    graph.outputs = [ctx.value_id(var) for var in jaxpr.outvars]
    apply_jax_semantic_rewrites(graph)
    verify_ir(graph)
    return graph


def capture_jax_function_with_params(
    fn: Callable[..., Any],
    params: Any,
    example_args: Sequence[Any],
    *,
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
) -> IRGraph:
    try:
        import jax
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("capture_jax_function_with_params requires the optional jax package") from exc

    named_leaves = _flatten_named_leaves(jax.tree_util, params)
    flat_leaves, treedef = jax.tree_util.tree_flatten(params)
    if len(flat_leaves) != len(named_leaves):
        raise ValueError("JAX param flattening produced mismatched leaf counts")
    leaf_count = len(flat_leaves)

    def bound_fn(*flat_params_and_args: Any) -> Any:
        flat_param_values = flat_params_and_args[:leaf_count]
        runtime_args = flat_params_and_args[leaf_count:]
        rebuilt_params = jax.tree_util.tree_unflatten(treedef, flat_param_values)
        return fn(rebuilt_params, *runtime_args)

    graph = capture_jax_function(
        bound_fn,
        (*flat_leaves, *example_args),
        weight_bindings=weight_bindings,
        weights_dir=weights_dir,
        graph_meta=graph_meta,
    )
    explicit = weight_bindings or {}
    _freeze_leading_param_inputs(
        graph,
        named_leaves,
        weights_dir=weights_dir,
        explicit=explicit,
    )
    _propagate_weight_binding_meta(graph)
    verify_ir(graph)
    return graph


def capture_jax_graphs(
    params: Any,
    specs: Sequence[JaxGraphSpec],
    *,
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
) -> CapturedJaxGraphBundle:
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"duplicate JAX graph spec name: {spec.name!r}")
        seen_names.add(spec.name)
        if spec.input_names is not None and len(spec.input_names) != len(spec.example_args):
            raise ValueError(
                f"graph {spec.name!r} has {len(spec.input_names)} input names for "
                f"{len(spec.example_args)} example inputs"
            )

    captured: dict[str, CapturedJaxGraph] = {}
    for spec in specs:
        ir = capture_jax_function_with_params(
            spec.fn,
            params,
            tuple(spec.example_args),
            weights_dir=weights_dir,
            weight_bindings=weight_bindings,
            graph_meta={
                **dict(graph_meta or {}),
                **dict(spec.graph_meta or {}),
                "jax_graph_name": spec.name,
                "jax_graph_role": spec.role,
                **({"input_names": tuple(spec.input_names)} if spec.input_names is not None else {}),
                **({"output_names": tuple(spec.output_names)} if spec.output_names is not None else {}),
            },
        )
        raw_ir = copy.deepcopy(ir)
        transpiled_graph = transpile_ir(ir)
        captured[spec.name] = CapturedJaxGraph(
            spec=spec,
            raw_ir_graph=raw_ir,
            ir_graph=ir,
            graph=transpiled_graph,
        )
    return CapturedJaxGraphBundle(graphs=captured, params=params, weights_dir=weights_dir)


__all__ = [
    "CapturedJaxGraph",
    "CapturedJaxGraphBundle",
    "JaxGraphSpec",
    "capture_jax_function",
    "capture_jax_function_with_params",
    "capture_jax_graphs",
]
