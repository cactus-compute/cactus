from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.import_semantics import apply_import_semantics
from cactus.transpile.lower import TranspiledGraph
from cactus.transpile.lower import transpile_ir
from cactus.transpile.weight_binding import resolve_weight_binding


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


def _match_param_names_to_constants(
    named_leaves: Sequence[tuple[str, np.ndarray]],
    constants: dict[str, object],
) -> dict[str, str]:
    matches: dict[str, str] = {}
    used_leaf_indexes: set[int] = set()
    for value_id, const in constants.items():
        const_array = np.asarray(const)
        for leaf_index, (name, leaf_array) in enumerate(named_leaves):
            if leaf_index in used_leaf_indexes:
                continue
            if const_array.shape != leaf_array.shape:
                continue
            comparable_leaf = leaf_array
            comparable_const = const_array
            if leaf_array.dtype.name == "bfloat16" and const_array.dtype == np.dtype(np.float32):
                comparable_leaf = leaf_array.astype(np.float32)
            elif const_array.dtype.name == "bfloat16" and leaf_array.dtype == np.dtype(np.float32):
                comparable_const = const_array.astype(np.float32)
            elif const_array.dtype != leaf_array.dtype:
                continue
            if not np.array_equal(comparable_const, comparable_leaf):
                continue
            matches[value_id] = name
            used_leaf_indexes.add(leaf_index)
            break
    return matches


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


@dataclass
class PreparedJaxSequenceInputs:
    source_tokens: Any
    target_tokens: Any
    source_mask: Any
    target_mask: Any

    @property
    def args(self) -> tuple[Any, Any, Any, Any]:
        return (self.source_tokens, self.target_tokens, self.source_mask, self.target_mask)

    def numpy_args(self) -> list[np.ndarray]:
        return [np.asarray(value) for value in self.args]


@dataclass
class CapturedJaxSequenceModel:
    ir_graph: IRGraph
    graph: TranspiledGraph
    params: Any
    apply_fn: Callable[..., Any]
    pad_token_id: int
    mask_style: str

    def prepare_inputs(self, source_tokens: Any, target_tokens: Any) -> PreparedJaxSequenceInputs:
        return prepare_jax_sequence_inputs(
            source_tokens,
            target_tokens,
            pad_token_id=self.pad_token_id,
            mask_style=self.mask_style,
        )

    def execute(self, source_tokens: Any, target_tokens: Any) -> list[Any]:
        prepared = self.prepare_inputs(source_tokens, target_tokens)
        self.graph.set_inputs(prepared.numpy_args())
        return self.graph.execute()

    def jax_reference(self, source_tokens: Any, target_tokens: Any) -> Any:
        prepared = self.prepare_inputs(source_tokens, target_tokens)
        return self.apply_fn(self.params, *prepared.args)


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
    ir_graph: IRGraph
    graph: TranspiledGraph

    def execute(self, *args: Any) -> list[Any]:
        self.graph.set_inputs([np.asarray(arg) for arg in args])
        return self.graph.execute()


@dataclass
class CapturedJaxGraphBundle:
    graphs: dict[str, CapturedJaxGraph]
    params: Any
    weights_dir: str | None = None

    def execute(self, graph_name: str, *args: Any) -> list[Any]:
        return self.graphs[graph_name].execute(*args)


def capture_jax_generation_graphs(
    params: Any,
    *,
    encoder: JaxGraphSpec | None = None,
    decoder_prefill: JaxGraphSpec | None = None,
    decoder_step: JaxGraphSpec | None = None,
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
    max_cache_seq_len: int | None = None,
    cache_sink_size: int = 4,
    enable_attention_fusion: bool = True,
) -> CapturedJaxGraphBundle:
    """Capture standard generation entrypoints as role-tagged JAX graphs.

    This is the generic sequence/generation layer above ``capture_jax_graphs``:
    callers provide explicit tensor entrypoints, and the helper adds the graph
    roles and cache metadata that the Cactus lowerer already understands.
    """
    specs: list[JaxGraphSpec] = []

    def _with_meta(
        spec: JaxGraphSpec | None,
        *,
        name: str,
        role: str,
        component: str,
        use_cache: bool,
    ) -> None:
        if spec is None:
            return
        cache_meta: dict[str, object] = {}
        if use_cache:
            cache_meta = {
                "use_internal_kv_cache": True,
                "cache_sink_size": int(cache_sink_size),
                **({"max_cache_seq_len": int(max_cache_seq_len)} if max_cache_seq_len is not None else {}),
            }
        specs.append(
            JaxGraphSpec(
                name=spec.name or name,
                role=spec.role if spec.role != "generic" else role,
                fn=spec.fn,
                example_args=spec.example_args,
                input_names=spec.input_names,
                output_names=spec.output_names,
                graph_meta={
                    "component": component,
                    "enable_jax_attention_fusion": bool(enable_attention_fusion),
                    **cache_meta,
                    **dict(spec.graph_meta or {}),
                },
            )
        )

    _with_meta(encoder, name="encoder", role="encoder", component="encoder", use_cache=False)
    _with_meta(
        decoder_prefill,
        name="decoder_prefill",
        role="decoder_prefill",
        component="decoder_prefill_chunk",
        use_cache=True,
    )
    _with_meta(
        decoder_step,
        name="decoder_step",
        role="decoder_step",
        component="decoder_step",
        use_cache=True,
    )
    if not specs:
        raise ValueError("capture_jax_generation_graphs requires at least one graph spec")
    return capture_jax_graphs(
        params,
        specs,
        weights_dir=weights_dir,
        weight_bindings=weight_bindings,
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "generation",
            **dict(graph_meta or {}),
        },
    )


def _add_value(graph: IRGraph, value_id: str, aval: Any, *, producer: str | None = None) -> None:
    shape, dtype = _shape_dtype_from_aval(aval)
    graph.add_value(IRValue(id=value_id, shape=shape, dtype=dtype, producer=producer))


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


def _rebuild_users(graph: IRGraph) -> None:
    for value in graph.values.values():
        value.users.clear()
    for node_id in graph.order:
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            graph.values[input_id].users.append(node_id)


def _rewrite_sum_div_to_mean(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "divide" or len(node.inputs) != 2:
            continue
        divisor = _constant_scalar(graph, node.inputs[1])
        if divisor is None:
            continue
        lhs_id = node.inputs[0]
        lhs_producer_id = graph.values[lhs_id].producer
        if lhs_producer_id is None:
            continue
        lhs_producer = graph.nodes[lhs_producer_id]
        sum_node = lhs_producer
        if lhs_producer.op == "expand":
            sum_input_id = lhs_producer.inputs[0]
            sum_producer_id = graph.values[sum_input_id].producer
            if sum_producer_id is None:
                continue
            sum_node = graph.nodes[sum_producer_id]
        if sum_node.op != "sum" or len(sum_node.inputs) != 1:
            continue
        input_shape = graph.values[sum_node.inputs[0]].shape
        axes = tuple(int(axis) for axis in sum_node.attrs.get("axis", ()))
        if input_shape is None or not axes:
            continue
        reduced_size = 1
        for axis in axes:
            reduced_size *= int(input_shape[axis])
        if not math.isclose(float(divisor), float(reduced_size), rel_tol=0.0, abs_tol=1e-6):
            continue
        sum_node.op = "mean"
        node.op = "view"
        node.inputs = [lhs_id]
        node.attrs = {"shape": graph.values[node.outputs[0]].shape}
        node.meta = {**node.meta, "rewritten_from": "sum_div_to_mean"}


def _producer(graph: IRGraph, value_id: str) -> IRNode | None:
    value = graph.values.get(value_id)
    if value is None or value.producer is None:
        return None
    return graph.nodes.get(value.producer)


def _strip_simple_wrappers(graph: IRGraph, value_id: str) -> str:
    current = value_id
    for _ in range(8):
        producer = _producer(graph, current)
        if producer is None or producer.op not in {"precision_cast", "view", "expand"} or not producer.inputs:
            return current
        current = producer.inputs[0]
    return current


def _trace_rms_denominator(graph: IRGraph, value_id: str) -> str | None:
    sqrt_node = _producer(graph, value_id)
    if sqrt_node is None or sqrt_node.op != "scalar_sqrt" or not sqrt_node.inputs:
        return None
    add_node = _producer(graph, sqrt_node.inputs[0])
    if add_node is None or add_node.op != "add":
        return None
    for add_input in add_node.inputs:
        mean_id = _strip_simple_wrappers(graph, add_input)
        mean_node = _producer(graph, mean_id)
        if mean_node is None or mean_node.op not in {"mean", "sum"} or not mean_node.inputs:
            continue
        square_node = _producer(graph, mean_node.inputs[0])
        if square_node is None or square_node.op != "multiply" or len(square_node.inputs) != 2:
            continue
        lhs = _strip_simple_wrappers(graph, square_node.inputs[0])
        rhs = _strip_simple_wrappers(graph, square_node.inputs[1])
        if lhs == rhs:
            return lhs
    return None


def _trace_rms_numerator(graph: IRGraph, value_id: str, source_id: str) -> tuple[str, str] | None:
    current = value_id
    producer = _producer(graph, current)
    if producer is not None and producer.op == "precision_cast" and producer.inputs:
        current = producer.inputs[0]
        producer = _producer(graph, current)
    if producer is None or producer.op != "multiply" or len(producer.inputs) != 2:
        return None
    lhs, rhs = producer.inputs
    if _strip_simple_wrappers(graph, lhs) == source_id:
        return lhs, rhs
    if _strip_simple_wrappers(graph, rhs) == source_id:
        return rhs, lhs
    return None


def _base_weight_value_id(graph: IRGraph, value_id: str) -> str:
    current = value_id
    for _ in range(8):
        producer = _producer(graph, current)
        if producer is None or producer.op not in {"view", "expand"} or not producer.inputs:
            return current
        current = producer.inputs[0]
    return current


def _rewrite_jax_rms_norms(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op == "multiply" and len(node.inputs) == 2:
            rewritten = _try_rewrite_jax_multiply_rms_norm(graph, node)
            if rewritten:
                continue
        if node.op != "divide" or len(node.inputs) != 2:
            continue
        source_id = _trace_rms_denominator(graph, node.inputs[1])
        if source_id is None:
            continue
        numerator = _trace_rms_numerator(graph, node.inputs[0], source_id)
        if numerator is None:
            continue
        x_id, weight_id = numerator
        weight_id = _base_weight_value_id(graph, weight_id)
        weight_value = graph.values.get(weight_id)
        x_value = graph.values.get(x_id)
        if weight_value is None or x_value is None:
            continue
        if weight_value.shape is None or len(weight_value.shape) != 1:
            continue
        if x_value.shape is None or int(weight_value.shape[0]) != int(x_value.shape[-1]):
            continue
        node.op = "rms_norm"
        node.inputs = [x_id, weight_id]
        node.attrs = {"eps": 1.0e-6}
        node.kind = "semantic"
        node.meta = {**node.meta, "rewritten_from": "jax_rms_norm"}


def _trace_rms_inv_std(graph: IRGraph, value_id: str) -> tuple[str, float] | None:
    current = _strip_simple_wrappers(graph, value_id)
    producer = _producer(graph, current)
    if producer is None:
        return None
    if producer.op == "pow" and float(producer.attrs.get("exponent", 0.0)) == -0.5 and producer.inputs:
        add_id = producer.inputs[0]
    elif producer.op == "divide" and len(producer.inputs) == 2:
        add_id = producer.inputs[1]
    else:
        return None

    add_node = _producer(graph, _strip_simple_wrappers(graph, add_id))
    if add_node is None or add_node.op != "add" or len(add_node.inputs) != 2:
        return None

    eps = 1.0e-6
    source_id: str | None = None
    for add_input in add_node.inputs:
        scalar = _constant_scalar(graph, add_input)
        if scalar is not None:
            eps = float(scalar)
            continue
        mean_node = _producer(graph, _strip_simple_wrappers(graph, add_input))
        if mean_node is None or mean_node.op not in {"mean", "sum"} or not mean_node.inputs:
            continue
        square_node = _producer(graph, _strip_simple_wrappers(graph, mean_node.inputs[0]))
        if square_node is None or square_node.op != "multiply" or len(square_node.inputs) != 2:
            continue
        lhs = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, square_node.inputs[0]))
        rhs = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, square_node.inputs[1]))
        if _strip_simple_wrappers(graph, lhs) == _strip_simple_wrappers(graph, rhs):
            source_id = _strip_simple_wrappers(graph, lhs)
    if source_id is None:
        return None
    return source_id, eps


def _try_trace_rms_norm_multiply_branch(graph: IRGraph, value_id: str) -> tuple[str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for source_candidate, inv_candidate in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        inv = _trace_rms_inv_std(graph, inv_candidate)
        if inv is None:
            continue
        source_id, eps = inv
        candidate_id = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, source_candidate))
        if _strip_simple_wrappers(graph, candidate_id) == _strip_simple_wrappers(graph, source_id):
            return source_id, eps
    return None


def _try_rewrite_jax_multiply_rms_norm(graph: IRGraph, node: IRNode) -> bool:
    for norm_id, weight_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        weight_id = _base_weight_value_id(graph, weight_id)
        weight_value = graph.values.get(weight_id)
        if weight_value is None or weight_value.shape is None or len(weight_value.shape) != 1:
            continue
        traced = _try_trace_rms_norm_multiply_branch(graph, norm_id)
        if traced is None:
            continue
        source_id, eps = traced
        source_value = graph.values.get(source_id)
        if source_value is None or source_value.shape is None:
            continue
        if int(weight_value.shape[0]) != int(source_value.shape[-1]):
            continue
        node.op = "rms_norm"
        node.inputs = [source_id, weight_id]
        node.attrs = {"eps": eps}
        node.kind = "semantic"
        node.meta = {**node.meta, "rewritten_from": "jax_rms_norm_multiply"}
        return True
    return False


def _rewrite_jax_silus(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "multiply" or len(node.inputs) != 2:
            continue
        for source_id, sigmoid_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            sigmoid_node = _producer(graph, _strip_simple_wrappers(graph, sigmoid_id))
            if sigmoid_node is None or sigmoid_node.op not in {"sigmoid", "logistic"} or len(sigmoid_node.inputs) != 1:
                continue
            if _strip_simple_wrappers(graph, source_id) != _strip_simple_wrappers(graph, sigmoid_node.inputs[0]):
                continue
            node.op = "silu"
            node.inputs = [source_id]
            node.attrs = {}
            node.kind = "semantic"
            node.meta = {**node.meta, "rewritten_from": "jax_silu"}
            break


def _strip_jax_pattern_wrappers(graph: IRGraph, value_id: str) -> str:
    current = value_id
    for _ in range(16):
        producer = _producer(graph, current)
        if producer is None or producer.op not in {"precision_cast", "reshape", "view", "expand"} or not producer.inputs:
            return current
        current = producer.inputs[0]
    return current


def _trace_jax_rotate_half_source(graph: IRGraph, value_id: str) -> str | None:
    cat_node = _producer(graph, _strip_jax_pattern_wrappers(graph, value_id))
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2:
        return None
    for neg_id, passthrough_id in ((cat_node.inputs[0], cat_node.inputs[1]), (cat_node.inputs[1], cat_node.inputs[0])):
        neg_node = _producer(graph, _strip_jax_pattern_wrappers(graph, neg_id))
        passthrough_node = _producer(graph, _strip_jax_pattern_wrappers(graph, passthrough_id))
        if neg_node is None or passthrough_node is None or passthrough_node.op != "slice":
            continue
        if neg_node.op == "negate" and neg_node.inputs:
            neg_source_id = neg_node.inputs[0]
        elif (
            neg_node.op == "scalar_multiply"
            and neg_node.inputs
            and float(neg_node.attrs.get("value", 0.0)) == -1.0
        ):
            neg_source_id = neg_node.inputs[0]
        else:
            continue
        neg_source_node = _producer(graph, _strip_jax_pattern_wrappers(graph, neg_source_id))
        if neg_source_node is None or neg_source_node.op != "slice" or not neg_source_node.inputs:
            continue
        if _strip_jax_pattern_wrappers(graph, neg_source_node.inputs[0]) != _strip_jax_pattern_wrappers(
            graph, passthrough_node.inputs[0]
        ):
            continue
        return neg_source_node.inputs[0]
    return None


def _trace_jax_rope_trig(graph: IRGraph, value_id: str, *, expected: str) -> str | None:
    current = _strip_jax_pattern_wrappers(graph, value_id)
    trig_node = _producer(graph, current)
    if trig_node is None or trig_node.op != expected or not trig_node.inputs:
        return None
    cat_node = _producer(graph, _strip_jax_pattern_wrappers(graph, trig_node.inputs[0]))
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2:
        return None
    if _strip_jax_pattern_wrappers(graph, cat_node.inputs[0]) != _strip_jax_pattern_wrappers(graph, cat_node.inputs[1]):
        return None
    angle_node = _producer(graph, _strip_jax_pattern_wrappers(graph, cat_node.inputs[0]))
    if angle_node is None or angle_node.op != "multiply" or len(angle_node.inputs) != 2:
        return None
    if any(_strip_jax_pattern_wrappers(graph, input_id) in graph.inputs for input_id in angle_node.inputs):
        return "dynamic_position_ids"
    if any(_constant_array(graph, _strip_jax_pattern_wrappers(graph, input_id)) is not None for input_id in angle_node.inputs):
        return "static_or_closed"
    return "unknown"


def _detect_jax_rope_patterns(graph: IRGraph) -> None:
    counts: dict[str, int] = {}
    for node in graph.nodes.values():
        if node.op != "add" or len(node.inputs) != 2:
            continue
        for direct_id, rotated_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            direct_node = _producer(graph, _strip_jax_pattern_wrappers(graph, direct_id))
            rotated_node = _producer(graph, _strip_jax_pattern_wrappers(graph, rotated_id))
            if direct_node is None or rotated_node is None or direct_node.op != "multiply" or rotated_node.op != "multiply":
                continue
            for direct_source_id, cos_id in ((direct_node.inputs[0], direct_node.inputs[1]), (direct_node.inputs[1], direct_node.inputs[0])):
                cos_kind = _trace_jax_rope_trig(graph, cos_id, expected="scalar_cos")
                if cos_kind is None:
                    continue
                for rotated_source_id, sin_id in (
                    (rotated_node.inputs[0], rotated_node.inputs[1]),
                    (rotated_node.inputs[1], rotated_node.inputs[0]),
                ):
                    sin_kind = _trace_jax_rope_trig(graph, sin_id, expected="scalar_sin")
                    if sin_kind is None:
                        continue
                    rotated_input_id = _trace_jax_rotate_half_source(graph, rotated_source_id)
                    if rotated_input_id is None:
                        continue
                    if _strip_jax_pattern_wrappers(graph, direct_source_id) != _strip_jax_pattern_wrappers(graph, rotated_input_id):
                        continue
                    kind = "dynamic_position_ids" if "dynamic_position_ids" in {cos_kind, sin_kind} else cos_kind
                    counts[kind] = counts.get(kind, 0) + 1
                    node.meta = {**node.meta, "jax_rope_pattern": kind}
                    break
                if "jax_rope_pattern" in node.meta:
                    break
            if "jax_rope_pattern" in node.meta:
                break
    if counts:
        graph.meta["jax_rope_patterns"] = counts


def _record_jax_semantic_pattern_counts(graph: IRGraph) -> None:
    counts: dict[str, int] = {}
    for node in graph.nodes.values():
        rewritten_from = node.meta.get("rewritten_from")
        if isinstance(rewritten_from, str) and rewritten_from.startswith("jax_"):
            name = rewritten_from.removeprefix("jax_")
            counts[name] = counts.get(name, 0) + 1
        rope_pattern = node.meta.get("jax_rope_pattern")
        if isinstance(rope_pattern, str):
            key = f"rope_{rope_pattern}"
            counts[key] = counts.get(key, 0) + 1
    if counts:
        graph.meta["jax_semantic_patterns"] = counts


def _trace_precision_cast_source(graph: IRGraph, value_id: str) -> str:
    producer = _producer(graph, value_id)
    if producer is not None and producer.op == "precision_cast" and producer.inputs:
        return producer.inputs[0]
    return value_id


def _trace_layer_norm_centered(graph: IRGraph, value_id: str) -> tuple[str, str] | None:
    producer = _producer(graph, value_id)
    if producer is None or producer.op != "subtract" or len(producer.inputs) != 2:
        return None
    lhs, rhs = producer.inputs
    mean_node = _producer(graph, _strip_simple_wrappers(graph, rhs))
    if mean_node is None or mean_node.op not in {"mean", "sum"} or not mean_node.inputs:
        return None
    source_id = mean_node.inputs[0]
    lhs_source_id = _trace_precision_cast_source(graph, lhs)
    raw_source_id = _trace_precision_cast_source(graph, source_id)
    if _strip_simple_wrappers(graph, lhs_source_id) != _strip_simple_wrappers(graph, raw_source_id):
        return None
    return value_id, raw_source_id


def _trace_layer_norm_inv_std(graph: IRGraph, value_id: str, source_id: str) -> float | None:
    producer = _producer(graph, value_id)
    if producer is None:
        return None
    if producer.op == "pow" and float(producer.attrs.get("exponent", 0.0)) == -0.5 and producer.inputs:
        add_id = producer.inputs[0]
    elif producer.op == "divide" and len(producer.inputs) == 2:
        add_id = producer.inputs[1]
    else:
        return None

    add_node = _producer(graph, add_id)
    if add_node is None or add_node.op != "add" or len(add_node.inputs) != 2:
        return None
    eps = 1.0e-5
    found_variance = False
    for add_input in add_node.inputs:
        scalar = _constant_scalar(graph, add_input)
        if scalar is not None:
            eps = float(scalar)
            continue
        mean_node = _producer(graph, _strip_simple_wrappers(graph, add_input))
        if mean_node is None or mean_node.op not in {"mean", "sum"} or not mean_node.inputs:
            continue
        square_node = _producer(graph, mean_node.inputs[0])
        if square_node is None or square_node.op != "multiply" or len(square_node.inputs) != 2:
            continue
        lhs = _trace_layer_norm_centered(graph, _strip_simple_wrappers(graph, square_node.inputs[0]))
        rhs = _trace_layer_norm_centered(graph, _strip_simple_wrappers(graph, square_node.inputs[1]))
        if lhs is not None and rhs is not None and lhs[1] == source_id and rhs[1] == source_id:
            found_variance = True
    return eps if found_variance else None


def _trace_layer_norm_normalized(graph: IRGraph, value_id: str) -> tuple[str, str, float] | None:
    producer = _producer(graph, value_id)
    if producer is None or producer.op != "multiply" or len(producer.inputs) != 2:
        return None
    for centered_candidate, inv_std_candidate in (
        (producer.inputs[0], producer.inputs[1]),
        (producer.inputs[1], producer.inputs[0]),
    ):
        centered = _trace_layer_norm_centered(graph, _strip_simple_wrappers(graph, centered_candidate))
        if centered is None:
            continue
        centered_id, source_id = centered
        eps = _trace_layer_norm_inv_std(graph, _strip_simple_wrappers(graph, inv_std_candidate), source_id)
        if eps is not None:
            return source_id, centered_id, eps
    return None


def _rewrite_jax_layer_norms(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "add" or len(node.inputs) != 2:
            continue
        for affine_id, bias_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            bias_id = _base_weight_value_id(graph, bias_id)
            bias_value = graph.values.get(bias_id)
            if bias_value is None or bias_value.shape is None or len(bias_value.shape) != 1:
                continue
            affine_node = _producer(graph, _strip_simple_wrappers(graph, affine_id))
            if affine_node is None or affine_node.op != "multiply" or len(affine_node.inputs) != 2:
                continue
            for normalized_id, weight_id in (
                (affine_node.inputs[0], affine_node.inputs[1]),
                (affine_node.inputs[1], affine_node.inputs[0]),
            ):
                weight_id = _base_weight_value_id(graph, weight_id)
                weight_value = graph.values.get(weight_id)
                if weight_value is None or weight_value.shape is None or len(weight_value.shape) != 1:
                    continue
                traced = _trace_layer_norm_normalized(graph, _strip_simple_wrappers(graph, normalized_id))
                if traced is None:
                    continue
                source_id, _centered_id, eps = traced
                source_value = graph.values.get(source_id)
                if source_value is None or source_value.shape is None:
                    continue
                if int(source_value.shape[-1]) != int(weight_value.shape[0]):
                    continue
                if int(weight_value.shape[0]) != int(bias_value.shape[0]):
                    continue
                node.op = "layer_norm"
                node.inputs = [source_id, weight_id, bias_id]
                node.attrs = {"eps": eps}
                node.kind = "semantic"
                node.meta = {**node.meta, "rewritten_from": "jax_layer_norm"}
                break
            if node.op == "layer_norm":
                break


def _branch_uses_rms_norm_of(graph: IRGraph, branch_value_id: str, residual_value_id: str) -> bool:
    residual_base = _strip_simple_wrappers(graph, residual_value_id)
    stack = [branch_value_id]
    visited: set[str] = set()
    while stack and len(visited) < 512:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        node = _producer(graph, _strip_simple_wrappers(graph, current))
        if node is None:
            continue
        if node.op == "rms_norm" and node.inputs:
            if _strip_simple_wrappers(graph, node.inputs[0]) == residual_base:
                return True
        stack.extend(node.inputs)
    return False


def _rewrite_prenorm_residual_adds_to_clipped(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "add" or len(node.inputs) != 2:
            continue
        lhs, rhs = node.inputs
        if _branch_uses_rms_norm_of(graph, rhs, lhs) or _branch_uses_rms_norm_of(graph, lhs, rhs):
            node.op = "add_clipped"
            node.kind = "semantic"
            node.meta = {**node.meta, "rewritten_from": "prenorm_residual_add"}


def _trace_passthrough_to_input(graph: IRGraph, value_id: str, *, ops: set[str]) -> str:
    current = value_id
    for _ in range(8):
        value = graph.values.get(current)
        if value is not None and value.shape is not None and len(value.shape) == 4:
            return current
        producer = _producer(graph, current)
        if producer is None or producer.op not in ops or not producer.inputs:
            return current
        current = producer.inputs[0]
    return current


def _trace_jax_attention_scores(graph: IRGraph, scores_id: str) -> tuple[str, str, str | None, float] | None:
    current = scores_id
    scale = 0.0
    producer = _producer(graph, current)
    if producer is not None and producer.op == "precision_cast" and producer.inputs:
        current = producer.inputs[0]
        producer = _producer(graph, current)
    if producer is not None and producer.op == "divide" and len(producer.inputs) == 2:
        divisor = _constant_scalar(graph, producer.inputs[1])
        if divisor is not None and divisor != 0.0:
            scale = 1.0 / float(divisor)
        current = producer.inputs[0]
        producer = _producer(graph, current)
    if producer is not None and producer.op in {"view", "expand", "precision_cast"} and producer.inputs:
        current = producer.inputs[0]
        producer = _producer(graph, current)
    if producer is None or producer.op != "matmul" or len(producer.inputs) != 2:
        return None

    query_id = _trace_passthrough_to_input(graph, producer.inputs[0], ops={"reshape", "view"})
    key_transposed_id = _trace_passthrough_to_input(graph, producer.inputs[1], ops={"reshape", "view"})
    key_transposed = _producer(graph, key_transposed_id)
    if key_transposed is None or key_transposed.op != "permute" or not key_transposed.inputs:
        return None
    key_id = key_transposed.inputs[0]
    if graph.values.get(query_id) is None or graph.values.get(key_id) is None:
        return None
    if graph.values[query_id].shape is None or len(graph.values[query_id].shape) != 4:
        return None
    if graph.values[key_id].shape is None or len(graph.values[key_id].shape) != 4:
        return None
    return query_id, key_id, None, scale


def _trace_jax_attention_probs(graph: IRGraph, probs_id: str) -> tuple[str, str, str | None, float] | None:
    current = _trace_passthrough_to_input(graph, probs_id, ops={"reshape", "view"})
    div_node = _producer(graph, current)
    if div_node is None or div_node.op != "divide" or len(div_node.inputs) != 2:
        return None
    exp_node = _producer(graph, div_node.inputs[0])
    if exp_node is None or exp_node.op != "scalar_exp" or not exp_node.inputs:
        return None
    subtract_node = _producer(graph, exp_node.inputs[0])
    if subtract_node is None or subtract_node.op != "subtract" or not subtract_node.inputs:
        return None
    masked_scores_id = subtract_node.inputs[0]
    masked_scores = _producer(graph, masked_scores_id)
    if masked_scores is not None and masked_scores.op == "where" and len(masked_scores.inputs) == 3:
        traced = _trace_jax_attention_scores(graph, masked_scores.inputs[1])
        if traced is None:
            return None
        query_id, key_id, _, scale = traced
        return query_id, key_id, masked_scores.inputs[0], scale
    return _trace_jax_attention_scores(graph, masked_scores_id)


def _rewrite_jax_attentions(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "matmul" or len(node.inputs) != 2:
            continue
        traced = _trace_jax_attention_probs(graph, node.inputs[0])
        if traced is None:
            continue
        query_id, key_id, mask_id, scale = traced
        value_id = _trace_passthrough_to_input(graph, node.inputs[1], ops={"reshape", "view"})
        value_shape = graph.values.get(value_id).shape if value_id in graph.values else None
        if value_shape is None or len(value_shape) != 4:
            continue
        node.op = "attention"
        node.inputs = [query_id, key_id, value_id] + ([] if mask_id is None else [mask_id])
        node.attrs = {
            "qkv_layout": "bhsd",
            "output_layout": "bhsd",
            "scale": float(scale),
            "is_causal": False,
            "additive_mask": False,
        }
        node.kind = "semantic"
        node.meta = {**node.meta, "rewritten_from": "jax_attention"}


def _prune_dead_nodes(graph: IRGraph) -> None:
    live_values = set(graph.outputs)
    live_nodes: set[str] = set()
    stack = list(graph.outputs)
    while stack:
        value_id = stack.pop()
        value = graph.values.get(value_id)
        if value is None or value.producer is None:
            continue
        node = graph.nodes.get(value.producer)
        if node is None or node.id in live_nodes:
            continue
        live_nodes.add(node.id)
        for input_id in node.inputs:
            if input_id not in live_values:
                live_values.add(input_id)
                stack.append(input_id)

    removed_nodes = set(graph.nodes) - live_nodes
    if not removed_nodes:
        return
    for node_id in removed_nodes:
        node = graph.nodes.pop(node_id)
        for output_id in node.outputs:
            graph.values.pop(output_id, None)
            graph.constants.pop(output_id, None)
    graph.order = [node_id for node_id in graph.order if node_id in live_nodes]


def _rewrite_mean_square_to_scaled_mean(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "mean" or len(node.inputs) != 1:
            continue
        square_id = node.inputs[0]
        square_value = graph.values.get(square_id)
        square_producer = graph.nodes.get(square_value.producer) if square_value is not None else None
        if square_producer is None or square_producer.op != "multiply" or len(square_producer.inputs) != 2:
            continue
        if square_producer.inputs[0] != square_producer.inputs[1]:
            continue
        source_id = square_producer.inputs[0]
        source_value = graph.values[source_id]
        if source_value.dtype != "fp32":
            continue
        axes = tuple(int(axis) for axis in node.attrs.get("axis", ()))
        source_shape = source_value.shape
        reduced_size = 1
        if source_shape is not None:
            for axis in axes:
                reduced_size *= int(source_shape[axis])
        if reduced_size < 64:
            continue

        scale_down_id = f"v{graph.meta.get('_next_rewrite_value_id', 0)}_rms_scaled"
        graph.meta["_next_rewrite_value_id"] = int(graph.meta.get("_next_rewrite_value_id", 0)) + 1
        while scale_down_id in graph.values:
            scale_down_id = f"v{graph.meta['_next_rewrite_value_id']}_rms_scaled"
            graph.meta["_next_rewrite_value_id"] = int(graph.meta["_next_rewrite_value_id"]) + 1
        scale_down_node_id = f"{square_producer.id}_rms_scale_down"
        while scale_down_node_id in graph.nodes:
            scale_down_node_id = f"{scale_down_node_id}_x"
        graph.add_node(
            IRNode(
                scale_down_node_id,
                "scalar_multiply",
                [source_id],
                [scale_down_id],
                attrs={"value": 1.0 / 64.0},
                meta={"rewritten_for": "mean_square_overflow"},
            )
        )
        graph.values[scale_down_id].shape = source_value.shape
        graph.values[scale_down_id].dtype = source_value.dtype
        square_producer.inputs = [scale_down_id, scale_down_id]

        original_mean_output = node.outputs[0]
        original_mean_value = graph.values[original_mean_output]
        scaled_mean_id = f"{original_mean_output}_scaled"
        while scaled_mean_id in graph.values:
            scaled_mean_id = f"{scaled_mean_id}_x"
        node.outputs = [scaled_mean_id]
        graph.values.pop(original_mean_output, None)
        graph.values[scaled_mean_id] = IRValue(
            id=scaled_mean_id,
            shape=original_mean_value.shape,
            dtype="fp32",
            producer=node.id,
            meta={"rewritten_for": "mean_square_overflow"},
        )
        graph.add_node(
            IRNode(
                f"{node.id}_rms_scaled_alias",
                "view",
                [scaled_mean_id],
                [original_mean_output],
                attrs={"shape": original_mean_value.shape},
                meta={
                    "rewritten_for": "mean_square_overflow",
                    "rms_source_id": source_id,
                    "rms_scale": 64.0,
                },
            )
        )
        graph.values[original_mean_output].shape = graph.values[scaled_mean_id].shape
        graph.values[original_mean_output].dtype = "fp32"
        graph.values[original_mean_output].meta.update({"rms_source_id": source_id, "rms_scale": 64.0})
        square_index = graph.order.index(square_producer.id)
        graph.order.insert(square_index, scale_down_node_id)
        mean_index = graph.order.index(node.id)
        graph.order.insert(mean_index + 1, f"{node.id}_rms_scaled_alias")


def _trace_rms_mean_source(graph: IRGraph, value_id: str) -> tuple[str, float] | None:
    current_id = value_id
    for _ in range(4):
        value = graph.values.get(current_id)
        if value is None:
            return None
        source_id = value.meta.get("rms_source_id")
        scale = value.meta.get("rms_scale")
        if isinstance(source_id, str) and isinstance(scale, (int, float)):
            return source_id, float(scale)
        producer_id = value.producer
        if producer_id is None:
            return None
        producer = graph.nodes[producer_id]
        if producer.op not in {"view", "expand"} or not producer.inputs:
            return None
        current_id = producer.inputs[0]
    return None


def _rewrite_rms_divide_to_scaled_numerator(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "divide" or len(node.inputs) != 2:
            continue
        denominator = graph.values.get(node.inputs[1])
        if denominator is None or denominator.producer is None:
            continue
        sqrt_node = graph.nodes[denominator.producer]
        if sqrt_node.op != "scalar_sqrt" or not sqrt_node.inputs:
            continue
        add_value = graph.values.get(sqrt_node.inputs[0])
        if add_value is None or add_value.producer is None:
            continue
        add_node = graph.nodes[add_value.producer]
        if add_node.op != "add":
            continue
        traced = None
        for input_id in add_node.inputs:
            traced = _trace_rms_mean_source(graph, input_id)
            if traced is not None:
                break
        if traced is None:
            continue
        source_id, scale = traced
        numerator_id = node.inputs[0]
        numerator_value = graph.values.get(numerator_id)
        if numerator_value is None:
            continue
        if numerator_id != source_id:
            producer_id = numerator_value.producer
            if producer_id is None:
                continue
            producer = graph.nodes[producer_id]
            source_value = graph.values.get(source_id)
            source_producer = graph.nodes.get(source_value.producer) if source_value is not None and source_value.producer else None
            equivalent_sources = {source_id}
            if source_producer is not None and source_producer.op == "precision_cast" and source_producer.inputs:
                equivalent_sources.add(source_producer.inputs[0])
            if producer.op != "precision_cast" or producer.inputs[0] not in equivalent_sources:
                continue
        scaled_id = f"{numerator_id}_rms_div_scaled"
        while scaled_id in graph.values:
            scaled_id = f"{scaled_id}_x"
        scale_node_id = f"{node.id}_rms_div_scale"
        while scale_node_id in graph.nodes:
            scale_node_id = f"{scale_node_id}_x"
        graph.add_node(
            IRNode(
                scale_node_id,
                "scalar_multiply",
                [numerator_id],
                [scaled_id],
                attrs={"value": 1.0 / scale},
                meta={"rewritten_for": "rms_divide_overflow"},
            )
        )
        graph.values[scaled_id].shape = numerator_value.shape
        graph.values[scaled_id].dtype = numerator_value.dtype
        divide_index = graph.order.index(node.id)
        graph.order.insert(divide_index, scale_node_id)
        node.inputs[0] = scaled_id


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


def _alias_output(ctx: _JaxImportContext, outvar: Any, source_id: str) -> None:
    ctx.bind_value(outvar, source_id)


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


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

    binary_ops = {
        "add": "add",
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
    }
    unary_ops = {
        "logistic": "sigmoid",
        "tanh": "tanh",
        "sqrt": "scalar_sqrt",
        "exp": "scalar_exp",
        "log": "scalar_log",
        "cos": "scalar_cos",
        "sin": "scalar_sin",
        "neg": "negate",
    }
    compare_ops = {
        "eq": "equal",
        "ne": "not_equal",
        "lt": "less",
        "le": "less_equal",
        "gt": "greater",
        "ge": "greater_equal",
    }
    if prim in binary_ops:
        lhs_array = _constant_array(graph, inputs[0])
        rhs_array = _constant_array(graph, inputs[1])
        if lhs_array is not None and rhs_array is not None:
            evaluators = {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "div": np.divide,
            }
            _add_constant(
                graph,
                ctx,
                eqn.outvars[0],
                evaluators[prim](lhs_array, rhs_array),
                source_name=f"folded:{prim}",
                meta=_derived_meta(inputs, op=prim),
            )
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
    if prim in binary_ops:
        return IRNode(node_id, binary_ops[prim], inputs, outputs, meta=meta)
    if prim in unary_ops:
        array = _constant_array(graph, inputs[0])
        if array is not None:
            evaluators = {
                "sqrt": np.sqrt,
                "exp": np.exp,
                "log": np.log,
                "cos": np.cos,
                "sin": np.sin,
                "neg": np.negative,
                "logistic": lambda value: 1.0 / (1.0 + np.exp(-value)),
                "tanh": np.tanh,
            }
            _add_constant(
                graph,
                ctx,
                eqn.outvars[0],
                evaluators[prim](array),
                source_name=f"folded:{prim}",
                meta=_derived_meta(inputs, op=prim),
            )
            return None
        return IRNode(node_id, unary_ops[prim], inputs, outputs, meta=meta)
    if prim in compare_ops:
        lhs_scalar = _constant_scalar(graph, inputs[0])
        rhs_scalar = _constant_scalar(graph, inputs[1])
        if rhs_scalar is not None:
            scalar_ops = {
                "eq": "scalar_equal",
                "ne": "scalar_not_equal",
                "lt": "scalar_less",
                "le": "scalar_less_equal",
                "gt": "scalar_greater",
                "ge": "scalar_greater_equal",
            }
            return IRNode(node_id, scalar_ops[prim], [inputs[0]], outputs, attrs={"value": rhs_scalar}, meta=meta)
        if lhs_scalar is not None:
            scalar_ops = {
                "eq": "scalar_equal",
                "ne": "scalar_not_equal",
                "lt": "scalar_greater",
                "le": "scalar_greater_equal",
                "gt": "scalar_less",
                "ge": "scalar_less_equal",
            }
            return IRNode(node_id, scalar_ops[prim], [inputs[1]], outputs, attrs={"value": lhs_scalar}, meta=meta)
        return IRNode(node_id, compare_ops[prim], inputs, outputs, meta=meta)
    if prim == "and":
        return IRNode(node_id, "logical_and", inputs, outputs, meta=meta)
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
                expected_lhs_batch = tuple(range(len(lhs_batch)))
                expected_rhs_batch = tuple(range(len(rhs_batch)))
                if lhs_batch != expected_lhs_batch or rhs_batch != expected_rhs_batch:
                    raise NotImplementedError("JAX dot_general only supports leading batch dims")
                if tuple(lhs_shape[dim] for dim in lhs_batch) != tuple(rhs_shape[dim] for dim in rhs_batch):
                    raise NotImplementedError("JAX dot_general batch dimensions must have matching sizes")
            expected_rhs_contract = (len(rhs_batch),)
            if tuple(lhs_contract) != (len(lhs_shape) - 1,) or tuple(rhs_contract) != expected_rhs_contract:
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
    if prim == "reshape":
        if inputs[0] in ctx.gather_index_aliases:
            ctx.gather_index_aliases[outputs[0]] = ctx.gather_index_aliases[inputs[0]]
        return IRNode(
            node_id,
            "reshape",
            inputs,
            outputs,
            attrs={"shape": tuple(int(dim) for dim in params["new_sizes"])},
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
        scalar = _constant_scalar(graph, inputs[0])
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
            while True:
                reshaped_id = f"v{ctx.next_value_id}_broadcast_base"
                ctx.next_value_id += 1
                if reshaped_id not in graph.values:
                    break
            reshape_node = IRNode(node_id, "view", inputs, [reshaped_id], attrs={"shape": reshaped_shape}, meta=meta)
            graph.add_node(reshape_node)
            graph.order.append(reshape_node.id)
            graph.values[reshaped_id].shape = reshaped_shape
            graph.values[reshaped_id].dtype = graph.values[inputs[0]].dtype
            graph.values[reshaped_id].meta.update(_derived_meta(inputs, op=f"{prim}:reshape"))
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
        changed_axes = [axis for axis, (start, limit) in enumerate(zip(starts, limits, strict=True)) if start != 0 or limit != getattr(eqn.invars[0].aval, "shape", ())[axis]]
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
    if prim == "max":
        lhs_literal = _literal_number(eqn.invars[0])
        rhs_literal = _literal_number(eqn.invars[1])
        if lhs_literal == -math.inf:
            _alias_output(ctx, eqn.outvars[0], inputs[1])
            return None
        if rhs_literal == -math.inf:
            _alias_output(ctx, eqn.outvars[0], inputs[0])
            return None
        raise NotImplementedError("JAX elementwise max import only supports -inf identity")
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
    """Capture a small pure JAX function into Cactus IR.

    This is a phase-1 frontend for simple inference graphs. It imports JAXPR
    primitives directly and then relies on the existing Cactus IR optimizer and
    lowering stack.
    """
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
    _rewrite_sum_div_to_mean(graph)
    _rewrite_jax_rms_norms(graph)
    _rewrite_jax_layer_norms(graph)
    _rewrite_jax_silus(graph)
    _detect_jax_rope_patterns(graph)
    if bool(graph.meta.get("enable_jax_attention_fusion", False)):
        _rewrite_jax_attentions(graph)
    _rewrite_prenorm_residual_adds_to_clipped(graph)
    apply_import_semantics(graph)
    _record_jax_semantic_pattern_counts(graph)
    _prune_dead_nodes(graph)
    _rebuild_users(graph)
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
    """Capture ``fn(params, *args)`` with pytree params frozen as named constants."""
    try:
        import jax
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("capture_jax_function_with_params requires the optional jax package") from exc

    named_leaves = _flatten_named_leaves(jax.tree_util, params)

    def bound_fn(*args: Any) -> Any:
        return fn(params, *args)

    graph = capture_jax_function(
        bound_fn,
        example_args,
        weight_bindings=weight_bindings,
        weights_dir=weights_dir,
        graph_meta=graph_meta,
    )
    explicit = weight_bindings or {}
    graph.meta["weight_bindings"] = {}
    for value_id, name in _match_param_names_to_constants(named_leaves, graph.constants).items():
        meta = {
            "source_name": name,
            **_binding_meta(name=name, weights_dir=weights_dir, explicit=explicit),
        }
        graph.values[value_id].meta.update(meta)
        if "path" in meta:
            graph.meta["weight_bindings"][value_id] = dict(meta)
    _propagate_weight_binding_meta(graph)
    return graph


def capture_jax_graphs(
    params: Any,
    specs: Sequence[JaxGraphSpec],
    *,
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
) -> CapturedJaxGraphBundle:
    """Capture named JAX entrypoints as separate Cactus graphs.

    Each spec function must have the tensor boundary ``fn(params, *args)``.
    The bundle is intentionally only orchestration; every graph still uses the
    generic JAX importer and the same mmap weight binding path as
    ``capture_jax_function_with_params``.
    """
    captured: dict[str, CapturedJaxGraph] = {}
    for spec in specs:
        if spec.name in captured:
            raise ValueError(f"duplicate JAX graph spec name: {spec.name!r}")
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
        captured[spec.name] = CapturedJaxGraph(
            spec=spec,
            ir_graph=ir,
            graph=transpile_ir(ir),
        )
    return CapturedJaxGraphBundle(graphs=captured, params=params, weights_dir=weights_dir)


def _as_jax_token_matrix(jnp: Any, tokens: Any) -> Any:
    value = jnp.asarray(tokens)
    if len(getattr(value, "shape", ())) == 1:
        return value[None, :]
    if len(getattr(value, "shape", ())) != 2:
        raise ValueError(f"token inputs must be rank 1 or rank 2, got shape={getattr(value, 'shape', None)}")
    return value


def prepare_jax_sequence_inputs(
    source_tokens: Any,
    target_tokens: Any,
    *,
    pad_token_id: int = 0,
    mask_style: str = "compact",
) -> PreparedJaxSequenceInputs:
    """Prepare standard seq2seq token inputs and masks for generic JAX capture.

    ``compact`` masks are intentionally broadcast-friendly:
    source mask has shape ``(batch, source_seq)`` and target mask has shape
    ``(batch, target_seq, target_seq)``. This keeps the captured graph less
    model-specific while matching common JAX/Flax attention broadcasting.
    ``attention_4d`` emits ``(batch, 1, 1, source_seq)`` and
    ``(batch, 1, target_seq, target_seq)`` for models that require explicit
    attention dimensions.
    """
    try:
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("prepare_jax_sequence_inputs requires the optional jax package") from exc

    src = _as_jax_token_matrix(jnp, source_tokens)
    tgt = _as_jax_token_matrix(jnp, target_tokens)
    source_padding = src != pad_token_id
    target_padding = tgt != pad_token_id
    target_seq = int(tgt.shape[1])
    causal = jnp.tril(jnp.ones((target_seq, target_seq), dtype=jnp.bool_))

    if mask_style == "compact":
        source_mask = source_padding
        target_mask = causal[None, :, :] & target_padding[:, None, :]
    elif mask_style == "attention_4d":
        source_mask = source_padding[:, None, None, :]
        target_mask = causal[None, None, :, :] & target_padding[:, None, None, :]
    else:
        raise ValueError(f"unsupported JAX sequence mask_style: {mask_style!r}")

    return PreparedJaxSequenceInputs(
        source_tokens=src,
        target_tokens=tgt,
        source_mask=source_mask,
        target_mask=target_mask,
    )


def capture_jax_sequence_model(
    apply_fn: Callable[..., Any],
    params: Any,
    example_source_tokens: Any,
    example_target_tokens: Any,
    *,
    pad_token_id: int = 0,
    mask_style: str = "compact",
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
) -> CapturedJaxSequenceModel:
    """Capture a generic JAX seq2seq-style model with auto-prepared masks.

    ``apply_fn`` must have the tensor-only boundary
    ``apply_fn(params, source_tokens, target_tokens, source_mask, target_mask)``.
    The helper intentionally handles only generic token/mask preparation; model
    internals still go through ``capture_jax_function_with_params``.
    """
    prepared = prepare_jax_sequence_inputs(
        example_source_tokens,
        example_target_tokens,
        pad_token_id=pad_token_id,
        mask_style=mask_style,
    )
    ir = capture_jax_function_with_params(
        apply_fn,
        params,
        prepared.args,
        weights_dir=weights_dir,
        weight_bindings=weight_bindings,
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "input_preparation": "sequence_masks",
            "mask_style": mask_style,
            "pad_token_id": int(pad_token_id),
            **dict(graph_meta or {}),
        },
    )
    graph = transpile_ir(ir)
    return CapturedJaxSequenceModel(
        ir_graph=ir,
        graph=graph,
        params=params,
        apply_fn=apply_fn,
        pad_token_id=int(pad_token_id),
        mask_style=mask_style,
    )


__all__ = [
    "CapturedJaxGraph",
    "CapturedJaxGraphBundle",
    "CapturedJaxSequenceModel",
    "JaxGraphSpec",
    "PreparedJaxSequenceInputs",
    "capture_jax_function",
    "capture_jax_function_with_params",
    "capture_jax_generation_graphs",
    "capture_jax_graphs",
    "capture_jax_sequence_model",
    "prepare_jax_sequence_inputs",
]
