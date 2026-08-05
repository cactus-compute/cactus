from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.import_semantics import apply_import_semantics


@dataclass(frozen=True)
class JaxPattern:
    name: str
    apply: Callable[[IRGraph], None]


def _constant_scalar_or_singleton(graph: IRGraph, value_id: str) -> float | None:
    value = graph.constants.get(value_id)
    if value is None:
        return None
    array = np.asarray(value)
    if graph.values[value_id].meta.get("jax_closed_constant") and array.size != 1:
        return None
    if array.size == 0:
        return None
    first = float(array.reshape(-1)[0].item())
    if np.allclose(array, first, rtol=0.0, atol=0.0):
        return first
    return None


def _rebuild_users(graph: IRGraph) -> None:
    for value in graph.values.values():
        value.users.clear()
    for node_id in graph.order:
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            graph.values[input_id].users.append(node_id)


def _producer(graph: IRGraph, value_id: str) -> IRNode | None:
    value = graph.values.get(value_id)
    if value is None or value.producer is None:
        return None
    return graph.nodes.get(value.producer)


def _strip_simple_wrappers(graph: IRGraph, value_id: str) -> str:
    current = value_id
    for _ in range(8):
        producer = _producer(graph, current)
        if producer is None or producer.op not in {"precision_cast", "reshape", "view", "expand"} or not producer.inputs:
            return current
        current = producer.inputs[0]
    return current


def _trace_reduction_source(graph: IRGraph, value_id: str) -> str | None:
    current = _strip_simple_wrappers(graph, value_id)
    node = _producer(graph, current)
    if node is not None and node.op == "divide" and len(node.inputs) == 2:
        divisor = _constant_scalar_or_singleton(graph, node.inputs[1])
        if divisor is None:
            return None
        current = _strip_simple_wrappers(graph, node.inputs[0])
        node = _producer(graph, current)
    if node is None or node.op not in {"mean", "sum"} or not node.inputs:
        return None
    return _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, node.inputs[0]))


def _trace_square_reduction_source(graph: IRGraph, value_id: str, term_trace) -> str | None:
    current = _strip_simple_wrappers(graph, value_id)
    node = _producer(graph, current)
    if node is not None and node.op == "divide" and len(node.inputs) == 2:
        divisor = _constant_scalar_or_singleton(graph, node.inputs[1])
        if divisor is None:
            return None
        current = _strip_simple_wrappers(graph, node.inputs[0])
        node = _producer(graph, current)
    if node is None or node.op not in {"mean", "sum"} or not node.inputs:
        return None
    square_node = _producer(graph, _strip_simple_wrappers(graph, node.inputs[0]))
    if square_node is None or square_node.op != "multiply" or len(square_node.inputs) != 2:
        return None
    lhs = term_trace(graph, square_node.inputs[0])
    rhs = term_trace(graph, square_node.inputs[1])
    if lhs is None or rhs is None:
        return None
    if _strip_simple_wrappers(graph, lhs) != _strip_simple_wrappers(graph, rhs):
        return None
    return _strip_simple_wrappers(graph, lhs)


def _trace_raw_square_source(graph: IRGraph, value_id: str) -> str | None:
    return _strip_simple_wrappers(graph, _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, value_id)))


def _trace_mean_square_source(graph: IRGraph, value_id: str) -> str | None:
    return _trace_square_reduction_source(graph, value_id, _trace_raw_square_source)


def _trace_layer_norm_variance_source(graph: IRGraph, value_id: str) -> str | None:
    current = _strip_simple_wrappers(graph, value_id)
    node = _producer(graph, current)
    if node is not None and node.op == "where" and node.inputs:
        current = _strip_simple_wrappers(graph, node.inputs[-1])
        node = _producer(graph, current)
    if node is None or node.op != "subtract" or len(node.inputs) != 2:
        return None
    mean_square_source = _trace_mean_square_source(graph, node.inputs[0])
    square_mean_node = _producer(graph, _strip_simple_wrappers(graph, node.inputs[1]))
    if mean_square_source is None or square_mean_node is None or square_mean_node.op != "multiply":
        return None
    if len(square_mean_node.inputs) != 2:
        return None
    lhs_mean_source = _trace_reduction_source(graph, square_mean_node.inputs[0])
    rhs_mean_source = _trace_reduction_source(graph, square_mean_node.inputs[1])
    if lhs_mean_source is None or rhs_mean_source is None:
        return None
    if _strip_simple_wrappers(graph, lhs_mean_source) != _strip_simple_wrappers(graph, rhs_mean_source):
        return None
    if _strip_simple_wrappers(graph, mean_square_source) != _strip_simple_wrappers(graph, lhs_mean_source):
        return None
    return mean_square_source


def _trace_layer_norm_centered_source(graph: IRGraph, value_id: str) -> str | None:
    centered = _trace_layer_norm_centered(graph, value_id)
    return None if centered is None else centered[0]


def _trace_layer_norm_centered(graph: IRGraph, value_id: str) -> tuple[str, str] | None:
    centered_id = _strip_simple_wrappers(graph, value_id)
    node = _producer(graph, centered_id)
    if node is None or node.op != "subtract" or len(node.inputs) != 2:
        return None
    source_id = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, node.inputs[0]))
    mean_source_id = _trace_reduction_source(graph, node.inputs[1])
    if mean_source_id is None:
        return None
    if _strip_simple_wrappers(graph, source_id) != _strip_simple_wrappers(graph, mean_source_id):
        return None
    return _strip_simple_wrappers(graph, source_id), centered_id


def _trace_inv_std_addend(graph: IRGraph, value_id: str) -> str | None:
    current = _strip_simple_wrappers(graph, value_id)
    node = _producer(graph, current)
    if node is None:
        return None
    if node.op == "pow" and node.inputs:
        exponent = float(node.attrs.get("exponent", 0.0))
        if exponent == -0.5:
            return node.inputs[0]
        if exponent == -1.0:
            return _trace_inv_std_addend(graph, node.inputs[0])
    if node.op == "scalar_sqrt" and node.inputs:
        return node.inputs[0]
    if node.op == "divide" and len(node.inputs) == 2:
        numerator = _constant_scalar_or_singleton(graph, node.inputs[0])
        if numerator == 1.0:
            return _trace_inv_std_addend(graph, node.inputs[1]) or node.inputs[1]
        return _trace_inv_std_addend(graph, node.inputs[1])
    return None


def _trace_norm_inv_std(
    graph: IRGraph,
    value_id: str,
    default_eps: float,
    variance_trace,
) -> tuple[str, float] | None:
    add_id = _trace_inv_std_addend(graph, value_id)
    add_node = _producer(graph, _strip_simple_wrappers(graph, add_id)) if add_id is not None else None
    if add_node is None or add_node.op != "add" or len(add_node.inputs) != 2:
        return None

    eps = default_eps
    source_id: str | None = None
    for add_input in add_node.inputs:
        scalar = _constant_scalar_or_singleton(graph, add_input)
        if scalar is not None:
            eps = float(scalar)
            continue
        variance_source = variance_trace(graph, add_input)
        if variance_source is not None:
            source_id = variance_source
    if source_id is None:
        return None
    return source_id, eps


def _trace_rms_inv_std(graph: IRGraph, value_id: str) -> tuple[str, float] | None:
    return _trace_norm_inv_std(graph, value_id, 1.0e-6, _trace_mean_square_source)


def _trace_layer_norm_inv_std(graph: IRGraph, value_id: str) -> tuple[str, float] | None:
    def _variance_source(g: IRGraph, candidate_id: str) -> str | None:
        return _trace_layer_norm_variance_source(g, candidate_id) or _trace_square_reduction_source(
            g,
            candidate_id,
            _trace_layer_norm_centered_source,
        )

    return _trace_norm_inv_std(graph, value_id, 1.0e-5, _variance_source)


def _valid_last_dim_weight(graph: IRGraph, weight_id: str, source_id: str) -> str | None:
    weight_id = _strip_simple_wrappers(graph, weight_id)
    weight_value = graph.values.get(weight_id)
    source_value = graph.values.get(source_id)
    if weight_value is None or source_value is None:
        return None
    if weight_value.shape is None or source_value.shape is None:
        return None
    if len(weight_value.shape) != 1 or int(weight_value.shape[0]) != int(source_value.shape[-1]):
        return None
    return weight_id


def _rewrite_norm_node(node: IRNode, op: str, inputs: list[str], eps: float, rewritten_from: str) -> None:
    node.op = op
    node.inputs = inputs
    node.attrs = {"eps": eps}
    node.kind = "semantic"
    node.meta = {**node.meta, "rewritten_from": rewritten_from}


def _trace_norm_branch(graph: IRGraph, value_id: str, source_trace, inv_trace) -> tuple[str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or len(node.inputs) != 2:
        return None
    pairs = ((node.inputs[0], node.inputs[1]),) if node.op == "divide" else ()
    if node.op == "multiply":
        pairs = ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0]))
    for source_candidate, inv_candidate in pairs:
        source_id = source_trace(graph, source_candidate)
        inv = inv_trace(graph, inv_candidate)
        if source_id is None or inv is None:
            continue
        inv_source_id, eps = inv
        if _strip_simple_wrappers(graph, source_id) == _strip_simple_wrappers(graph, inv_source_id):
            return source_id, eps
    return None


def _trace_weighted_norm(graph: IRGraph, value_id: str, branch_trace) -> tuple[str, str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for norm_id, weight_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        traced = branch_trace(graph, norm_id)
        if traced is None:
            continue
        source_id, eps = traced
        weight_id = _valid_last_dim_weight(graph, weight_id, source_id)
        if weight_id is None:
            continue
        return source_id, weight_id, eps
    return None


def _trace_factored_weighted_norm(
    graph: IRGraph,
    value_id: str,
    source_trace,
    inv_trace,
) -> tuple[str, str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for source_candidate, scaled_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        source_id = source_trace(graph, source_candidate)
        scaled_node = _producer(graph, _strip_simple_wrappers(graph, scaled_id))
        if source_id is None or scaled_node is None or scaled_node.op != "multiply" or len(scaled_node.inputs) != 2:
            continue
        for inv_id, weight_id in ((scaled_node.inputs[0], scaled_node.inputs[1]), (scaled_node.inputs[1], scaled_node.inputs[0])):
            inv = inv_trace(graph, inv_id)
            if inv is None:
                continue
            inv_source_id, eps = inv
            if _strip_simple_wrappers(graph, source_id) != _strip_simple_wrappers(graph, inv_source_id):
                continue
            weight_id = _valid_last_dim_weight(graph, weight_id, source_id)
            if weight_id is not None:
                return source_id, weight_id, eps
    return None


def _trace_weighted_rms_norm(graph: IRGraph, value_id: str) -> tuple[str, str, float] | None:
    traced = _trace_weighted_norm(
        graph,
        value_id,
        lambda g, candidate: _trace_norm_branch(g, candidate, _trace_raw_square_source, _trace_rms_inv_std),
    )
    if traced is not None:
        source_id, _, _ = traced
        source_producer = _producer(graph, source_id)
        if source_producer is None or source_producer.op != "subtract":
            return traced
    return _trace_factored_weighted_norm(graph, value_id, _trace_raw_square_source, _trace_rms_inv_std)


def _trace_weighted_layer_norm(graph: IRGraph, value_id: str) -> tuple[str, str, float] | None:
    traced = _trace_weighted_norm(
        graph,
        value_id,
        lambda g, candidate: _trace_norm_branch(g, candidate, _trace_layer_norm_centered_source, _trace_layer_norm_inv_std),
    )
    if traced is not None:
        return traced
    return _trace_factored_weighted_norm(
        graph,
        value_id,
        _trace_layer_norm_centered_source,
        _trace_layer_norm_inv_std,
    )


def _rewrite_jax_rms_norms(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op == "multiply" and len(node.inputs) == 2:
            traced = _trace_weighted_rms_norm(graph, node.outputs[0] if node.outputs else "")
            if traced is not None:
                source_id, weight_id, eps = traced
                _rewrite_norm_node(node, "rms_norm", [source_id, weight_id], eps, "jax_rms_norm_multiply")
                continue
        if node.op == "divide" and len(node.inputs) == 2:
            denominator = _trace_rms_inv_std(graph, node.inputs[1])
            if denominator is None:
                continue
            source_id, eps = denominator
            numerator = _producer(graph, _strip_simple_wrappers(graph, node.inputs[0]))
            if numerator is None or numerator.op != "multiply" or len(numerator.inputs) != 2:
                continue
            for x_id, weight_id in ((numerator.inputs[0], numerator.inputs[1]), (numerator.inputs[1], numerator.inputs[0])):
                if _strip_simple_wrappers(graph, x_id) != _strip_simple_wrappers(graph, source_id):
                    continue
                weight_id = _valid_last_dim_weight(graph, weight_id, x_id)
                if weight_id is not None:
                    _rewrite_norm_node(node, "rms_norm", [x_id, weight_id], eps, "jax_rms_norm")
                    break


def _rewrite_jax_layer_norms(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op == "add" and len(node.inputs) == 2:
            for affine_id, bias_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
                bias_id = _strip_simple_wrappers(graph, bias_id)
                bias_value = graph.values.get(bias_id)
                if bias_value is None or bias_value.shape is None or len(bias_value.shape) != 1:
                    continue
                traced_affine = _trace_weighted_layer_norm(graph, affine_id)
                if traced_affine is None:
                    continue
                source_id, weight_id, eps = traced_affine
                source_value = graph.values.get(source_id)
                if source_value is None or source_value.shape is None:
                    continue
                if int(bias_value.shape[0]) != int(source_value.shape[-1]):
                    continue
                _rewrite_norm_node(node, "layer_norm", [source_id, weight_id, bias_id], eps, "jax_layer_norm")
                break
            continue
        if node.op == "multiply" and len(node.inputs) == 2:
            traced = _trace_weighted_layer_norm(graph, node.outputs[0] if node.outputs else "")
            if traced is not None:
                source_id, weight_id, eps = traced
                _rewrite_norm_node(node, "layer_norm", [source_id, weight_id], eps, "jax_layer_norm_multiply")


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


def _trace_batch_norm_centered(graph: IRGraph, value_id: str) -> tuple[str, str] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "subtract" or len(node.inputs) != 2:
        return None
    source_id = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, node.inputs[0]))
    mean_id = _strip_simple_wrappers(graph, node.inputs[1])
    source_value = graph.values.get(source_id)
    mean_value = graph.values.get(mean_id)
    if source_value is None or source_value.shape is None:
        return None
    if mean_value is None or mean_value.shape is None or len(mean_value.shape) != 1:
        return None
    if int(source_value.shape[-1]) != int(mean_value.shape[0]):
        return None
    return source_id, mean_id


def _trace_batch_norm_inv_scale(graph: IRGraph, value_id: str) -> tuple[str, str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for inv_id, scale_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        scale_id = _strip_simple_wrappers(graph, scale_id)
        scale_value = graph.values.get(scale_id)
        if scale_value is None or scale_value.shape is None or len(scale_value.shape) != 1:
            continue
        inv_node = _producer(graph, _strip_simple_wrappers(graph, inv_id))
        if inv_node is None or inv_node.op not in {"pow", "divide"} or not inv_node.inputs:
            continue
        if inv_node.op == "pow" and float(inv_node.attrs.get("exponent", 0.0)) != -0.5:
            continue
        add_id = inv_node.inputs[0] if inv_node.op == "pow" else inv_node.inputs[1]
        add_node = _producer(graph, _strip_simple_wrappers(graph, add_id))
        if add_node is None or add_node.op != "add" or len(add_node.inputs) != 2:
            continue
        eps = 1.0e-5
        var_id: str | None = None
        for add_input in add_node.inputs:
            scalar = _constant_scalar_or_singleton(graph, add_input)
            if scalar is not None:
                eps = float(scalar)
                continue
            candidate = _strip_simple_wrappers(graph, add_input)
            candidate_value = graph.values.get(candidate)
            if candidate_value is not None and candidate_value.shape is not None and len(candidate_value.shape) == 1:
                var_id = candidate
        if var_id is None:
            continue
        var_value = graph.values.get(var_id)
        if var_value is None or var_value.shape is None or int(var_value.shape[0]) != int(scale_value.shape[0]):
            continue
        return var_id, scale_id, eps
    return None


def _trace_batch_norm_affine(graph: IRGraph, value_id: str) -> tuple[str, str, str, str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for centered_id, inv_scale_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        centered = _trace_batch_norm_centered(graph, centered_id)
        inv_scale = _trace_batch_norm_inv_scale(graph, inv_scale_id)
        if centered is None or inv_scale is None:
            continue
        source_id, mean_id = centered
        var_id, scale_id, eps = inv_scale
        channel_count = int(graph.values[scale_id].shape[0])  # type: ignore[index]
        if int(graph.values[mean_id].shape[0]) != channel_count:  # type: ignore[index]
            continue
        if int(graph.values[var_id].shape[0]) != channel_count:  # type: ignore[index]
            continue
        return source_id, scale_id, mean_id, var_id, eps
    return None


def _rewrite_jax_batch_norms(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op != "add" or len(node.inputs) != 2:
            continue
        for affine_id, bias_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            bias_id = _strip_simple_wrappers(graph, bias_id)
            bias_value = graph.values.get(bias_id)
            if bias_value is None or bias_value.shape is None or len(bias_value.shape) != 1:
                continue
            traced = _trace_batch_norm_affine(graph, affine_id)
            if traced is None:
                continue
            source_id, scale_id, mean_id, var_id, eps = traced
            source_value = graph.values.get(source_id)
            if source_value is None or source_value.shape is None or not source_value.shape:
                continue
            channel_count = int(bias_value.shape[0])
            if int(source_value.shape[-1]) != channel_count:
                continue
            node.op = "batch_norm"
            node.inputs = [source_id, scale_id, bias_id, mean_id, var_id]
            node.attrs = {"axis": len(source_value.shape) - 1, "eps": eps}
            node.kind = "semantic"
            node.meta = {**node.meta, "rewritten_from": "jax_batch_norm"}
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


def _shape(graph: IRGraph, value_id: str) -> tuple[int, ...] | None:
    value = graph.values.get(value_id)
    if value is None or value.shape is None:
        return None
    return tuple(int(dim) for dim in value.shape)


def _trace_view_input(graph: IRGraph, value_id: str) -> str:
    current = value_id
    for _ in range(8):
        node = _producer(graph, current)
        if node is None or node.op not in {"view", "reshape", "precision_cast"} or not node.inputs:
            return current
        current = node.inputs[0]
    return current


def _trace_gqa_repeat_base(graph: IRGraph, value_id: str) -> str:
    current = _trace_view_input(graph, value_id)
    shape = _shape(graph, current)
    if shape is None or len(shape) != 5:
        return current
    expand = _producer(graph, current)
    if expand is None or expand.op != "expand" or not expand.inputs:
        return current
    base = _trace_view_input(graph, expand.inputs[0])
    base_shape = _shape(graph, base)
    if base_shape is None or len(base_shape) != 4:
        return current
    if base_shape[0] == shape[0] and base_shape[1] == shape[1] and base_shape[2] == shape[3] and base_shape[3] == shape[4]:
        return base
    return current


def _trace_jax_attention_logits(graph: IRGraph, value_id: str) -> tuple[str, str, float, str | None, bool] | None:
    current = _trace_view_input(graph, value_id)
    node = _producer(graph, current)
    mask_id: str | None = None
    additive_mask = False
    if node is not None and node.op == "where" and len(node.inputs) >= 2:
        mask_id = node.inputs[0]
        current = _trace_view_input(graph, node.inputs[1])
        node = _producer(graph, current)
    if node is not None and node.op == "add" and len(node.inputs) == 2:
        for score_id, additive_mask_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            score_node = _producer(graph, _trace_view_input(graph, score_id))
            if score_node is not None and score_node.op == "divide":
                mask_id = additive_mask_id
                additive_mask = True
                current = _trace_view_input(graph, score_id)
                node = score_node
                break
    if node is None or node.op != "divide" or len(node.inputs) != 2:
        return None
    divisor = _constant_scalar_or_singleton(graph, node.inputs[1])
    if divisor is None or divisor == 0.0:
        return None
    matmul = _producer(graph, _trace_view_input(graph, node.inputs[0]))
    if matmul is None or matmul.op != "matmul" or len(matmul.inputs) != 2:
        return None
    query_id = _trace_view_input(graph, matmul.inputs[0])
    key_t_id = _trace_view_input(graph, matmul.inputs[1])
    key_t_node = _producer(graph, key_t_id)
    if key_t_node is None or key_t_node.op != "permute" or len(key_t_node.inputs) != 1:
        return None
    if tuple(int(dim) for dim in key_t_node.attrs.get("permutation", ())) != (0, 1, 3, 2):
        return None
    key_id = _trace_gqa_repeat_base(graph, key_t_node.inputs[0])
    return query_id, key_id, 1.0 / float(divisor), mask_id, additive_mask


def _trace_jax_attention_probs(graph: IRGraph, value_id: str) -> tuple[str, str, float, str | None, bool] | None:
    current = _trace_view_input(graph, value_id)
    cast = _producer(graph, current)
    if cast is not None and cast.op == "precision_cast" and cast.inputs:
        current = _trace_view_input(graph, cast.inputs[0])
    divide = _producer(graph, current)
    if divide is None or divide.op != "divide" or len(divide.inputs) != 2:
        return None
    exp_node = _producer(graph, _trace_view_input(graph, divide.inputs[0]))
    if exp_node is None or exp_node.op != "scalar_exp" or not exp_node.inputs:
        return None
    subtract = _producer(graph, _trace_view_input(graph, exp_node.inputs[0]))
    if subtract is None or subtract.op != "subtract" or not subtract.inputs:
        return None
    return _trace_jax_attention_logits(graph, subtract.inputs[0])


def _rewrite_attention_node(
    node: IRNode,
    query_id: str,
    key_id: str,
    value_id: str,
    mask_id: str | None,
    *,
    additive_mask: bool,
    scale: float,
    output_layout: str,
    is_causal: bool,
    rewritten_from: str,
) -> None:
    node.op = "attention"
    node.inputs = [query_id, key_id, value_id] + ([] if mask_id is None else [mask_id])
    node.attrs = {
        "scale": float(scale),
        "is_causal": bool(is_causal),
        "window_size": 0,
        "q_layout": "bhsd",
        "k_layout": "bhsd",
        "v_layout": "bhsd",
        "output_layout": output_layout,
        **({"additive_mask": True} if additive_mask else {}),
    }
    node.kind = "semantic"
    node.meta = {**node.meta, "rewritten_from": rewritten_from}


def _trace_attention_output(graph: IRGraph, node: IRNode) -> tuple[str, str, str, float, str | None, bool] | None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return None
    matmul = _producer(graph, _trace_view_input(graph, node.inputs[0]))
    if matmul is None or matmul.op != "matmul" or len(matmul.inputs) != 2:
        return None
    traced = _trace_jax_attention_probs(graph, matmul.inputs[0])
    if traced is None:
        return None
    query_id, key_id, scale, mask_id, additive_mask = traced
    return query_id, key_id, _trace_gqa_repeat_base(graph, matmul.inputs[1]), scale, mask_id, additive_mask


def _valid_attention_qkv_shapes(
    query_shape: tuple[int, ...] | None,
    key_shape: tuple[int, ...] | None,
    value_shape: tuple[int, ...] | None,
) -> bool:
    if query_shape is None or key_shape is None or value_shape is None:
        return False
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False
    if query_shape[0] != key_shape[0] or key_shape[0] != value_shape[0]:
        return False
    return key_shape[1:] == value_shape[1:]


def _valid_attention_mask_shape(
    mask_shape: tuple[int, ...] | None,
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
) -> bool:
    return mask_shape == (query_shape[0], query_shape[1], query_shape[2], key_shape[2])


def _rewrite_jax_prefill_attentions(graph: IRGraph) -> None:
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op not in {"permute", "view", "reshape"}:
            continue
        component = str(graph.meta.get("component", "") or "").strip().lower()
        is_decoder_step = component == "decoder_step"
        is_causal_prefill = component in {"decoder_prefill_chunk", "decoder_media_step"}
        if node.op == "permute" and tuple(int(dim) for dim in node.attrs.get("permutation", ())) != (0, 2, 1, 3):
            continue
        if node.op in {"view", "reshape"} and not is_decoder_step:
            continue
        output_shape = _shape(graph, node.outputs[0])
        if output_shape is None or len(output_shape) != 4:
            continue
        traced = _trace_attention_output(graph, node)
        if traced is None:
            continue
        query_id, key_id, value_id, scale, mask_id, additive_mask = traced
        query_shape = _shape(graph, query_id)
        key_shape = _shape(graph, key_id)
        value_shape = _shape(graph, value_id)
        if not _valid_attention_qkv_shapes(query_shape, key_shape, value_shape):
            continue
        assert query_shape is not None and key_shape is not None
        if mask_id is not None and not _valid_attention_mask_shape(_shape(graph, mask_id), query_shape, key_shape):
            continue
        is_self_attention = query_shape[2] == key_shape[2]
        if is_decoder_step:
            if query_shape[2] != 1 or (not is_self_attention and mask_id is None):
                continue
            _rewrite_attention_node(
                node,
                query_id,
                key_id,
                value_id,
                mask_id,
                additive_mask=additive_mask,
                scale=scale,
                output_layout="bhsd",
                is_causal=is_self_attention,
                rewritten_from="jax_decoder_step_self_attention" if is_self_attention else "jax_decoder_step_cross_attention",
            )
            continue
        if output_shape != (query_shape[0], query_shape[2], query_shape[1], query_shape[3]):
            continue
        _rewrite_attention_node(
            node,
            query_id,
            key_id,
            value_id,
            mask_id,
            additive_mask=additive_mask,
            scale=scale,
            output_layout="bthd",
            is_causal=is_causal_prefill and mask_id is None and is_self_attention,
            rewritten_from="jax_prefill_attention",
        )


def _trace_precision_cast_source(graph: IRGraph, value_id: str) -> str:
    producer = _producer(graph, value_id)
    if producer is not None and producer.op == "precision_cast" and producer.inputs:
        return producer.inputs[0]
    return value_id


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


def _unique_graph_id(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _legalize_remaining_jax_reductions(graph: IRGraph) -> None:
    new_order: list[str] = []
    for node_id in list(graph.order):
        node = graph.nodes[node_id]
        if node.op not in {"sum", "mean"} or len(node.inputs) != 1 or len(node.outputs) != 1:
            new_order.append(node_id)
            continue
        input_value = graph.values.get(node.inputs[0])
        output_value = graph.values.get(node.outputs[0])
        if input_value is None or output_value is None or input_value.dtype == "fp16":
            new_order.append(node_id)
            continue

        cast_input_id = _unique_graph_id(set(graph.values), f"{node.outputs[0]}__reduce_fp16_input")
        cast_input_node_id = _unique_graph_id(set(graph.nodes), f"{node.id}_reduce_input_cast")
        graph.add_node(
            IRNode(
                cast_input_node_id,
                "precision_cast",
                [node.inputs[0]],
                [cast_input_id],
                attrs={"dtype": "fp16"},
                meta={
                    "jax_reduce_legalization": "input_fp16",
                    "source_dtype": input_value.dtype,
                },
            )
        )
        graph.values[cast_input_id].shape = input_value.shape
        graph.values[cast_input_id].dtype = "fp16"
        node.inputs = [cast_input_id]
        new_order.append(cast_input_node_id)
        new_order.append(node_id)

        if output_value.dtype == "fp16":
            continue

        original_output_id = node.outputs[0]
        reduce_output_id = _unique_graph_id(set(graph.values), f"{original_output_id}__reduce_fp16")
        node.outputs = [reduce_output_id]
        graph.values[reduce_output_id] = IRValue(
            id=reduce_output_id,
            shape=output_value.shape,
            dtype="fp16",
            producer=node.id,
            meta={
                "jax_reduce_legalization": "fp16_output",
                "source_output": original_output_id,
            },
        )
        cast_output_node_id = _unique_graph_id(set(graph.nodes), f"{node.id}_reduce_output_cast")
        graph.nodes[cast_output_node_id] = IRNode(
            cast_output_node_id,
            "precision_cast",
            [reduce_output_id],
            [original_output_id],
            attrs={"dtype": output_value.dtype},
            meta={
                "jax_reduce_legalization": "restore_output_dtype",
                "source_dtype": "fp16",
            },
        )
        output_value.producer = cast_output_node_id
        new_order.append(cast_output_node_id)
    graph.order = new_order


JAX_PATTERNS = (
    JaxPattern("batch_norm", _rewrite_jax_batch_norms),
    JaxPattern("layer_norm", _rewrite_jax_layer_norms),
    JaxPattern("rms_norm", _rewrite_jax_rms_norms),
    JaxPattern("silu", _rewrite_jax_silus),
    JaxPattern("attention", _rewrite_jax_prefill_attentions),
    JaxPattern("prenorm_add_clipped", _rewrite_prenorm_residual_adds_to_clipped),
)


def _apply_pattern(graph: IRGraph, pattern: JaxPattern) -> None:
    pattern.apply(graph)


def apply_jax_semantic_rewrites(graph: IRGraph) -> None:
    for pattern in JAX_PATTERNS:
        _apply_pattern(graph, pattern)
    apply_import_semantics(graph)
    _prune_dead_nodes(graph)
    _legalize_remaining_jax_reductions(graph)
    _rebuild_users(graph)
