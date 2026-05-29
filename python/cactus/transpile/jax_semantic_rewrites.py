from __future__ import annotations

import numpy as np

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.import_semantics import apply_import_semantics


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


def _constant_array(graph: IRGraph, value_id: str) -> np.ndarray | None:
    value = graph.constants.get(value_id)
    if value is None:
        return None
    if graph.values[value_id].meta.get("jax_closed_constant"):
        return None
    return np.asarray(value)


def _where_scalar_value(value: float) -> float:
    if value < -1.0e30:
        return -65504.0
    if value > 1.0e30:
        return 65504.0
    return value


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
        if mean_node is not None and mean_node.op == "divide" and len(mean_node.inputs) == 2:
            divisor = _constant_scalar_or_singleton(graph, mean_node.inputs[1])
            if divisor is None:
                continue
            mean_id = _strip_simple_wrappers(graph, mean_node.inputs[0])
            mean_node = _producer(graph, mean_id)
        if mean_node is None or mean_node.op not in {"mean", "sum"} or not mean_node.inputs:
            continue
        square_node = _producer(graph, _strip_simple_wrappers(graph, mean_node.inputs[0]))
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
        if producer is None or producer.op not in {"precision_cast", "reshape", "view", "expand"} or not producer.inputs:
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
    if node is None or len(node.inputs) != 2:
        return None
    if node.op == "divide":
        source_id = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, node.inputs[0]))
        traced_source_id = _trace_rms_denominator(graph, node.inputs[1])
        if traced_source_id is None:
            return None
        if _strip_simple_wrappers(graph, source_id) == _strip_simple_wrappers(graph, traced_source_id):
            eps = _trace_rms_denominator_eps(graph, node.inputs[1])
            return _strip_simple_wrappers(graph, traced_source_id), eps
        return None
    if node.op != "multiply":
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


def _trace_rms_denominator_eps(graph: IRGraph, value_id: str) -> float:
    sqrt_node = _producer(graph, value_id)
    if sqrt_node is None or sqrt_node.op != "scalar_sqrt" or not sqrt_node.inputs:
        return 1.0e-6
    add_node = _producer(graph, sqrt_node.inputs[0])
    if add_node is None or add_node.op != "add":
        return 1.0e-6
    for add_input in add_node.inputs:
        eps = _constant_scalar_or_singleton(graph, add_input)
        if eps is not None:
            return float(eps)
    return 1.0e-6


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
        source_producer = _producer(graph, source_id)
        if source_producer is not None and source_producer.op == "subtract":
            continue
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


def _record_jax_semantic_pattern_counts(graph: IRGraph) -> None:
    counts: dict[str, int] = {}
    for node in graph.nodes.values():
        rewritten_from = node.meta.get("rewritten_from")
        if isinstance(rewritten_from, str) and rewritten_from.startswith("jax_"):
            name = rewritten_from.removeprefix("jax_")
            counts[name] = counts.get(name, 0) + 1
    if counts:
        graph.meta["jax_semantic_patterns"] = counts


def _trace_precision_cast_source(graph: IRGraph, value_id: str) -> str:
    producer = _producer(graph, value_id)
    if producer is not None and producer.op == "precision_cast" and producer.inputs:
        return producer.inputs[0]
    return value_id


def _trace_batch_norm_centered(graph: IRGraph, value_id: str) -> tuple[str, str] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "subtract" or len(node.inputs) != 2:
        return None
    lhs = _trace_precision_cast_source(graph, _strip_simple_wrappers(graph, node.inputs[0]))
    mean_id = _base_weight_value_id(graph, node.inputs[1])
    source_value = graph.values.get(lhs)
    mean_value = graph.values.get(mean_id)
    if source_value is None or source_value.shape is None:
        return None
    if mean_value is None or mean_value.shape is None or len(mean_value.shape) != 1:
        return None
    if int(source_value.shape[-1]) != int(mean_value.shape[0]):
        return None
    return lhs, mean_id


def _trace_batch_norm_inv_scale(graph: IRGraph, value_id: str) -> tuple[str, str, float] | None:
    node = _producer(graph, _strip_simple_wrappers(graph, value_id))
    if node is None or node.op != "multiply" or len(node.inputs) != 2:
        return None
    for inv_id, scale_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        scale_id = _base_weight_value_id(graph, scale_id)
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
            candidate = _base_weight_value_id(graph, add_input)
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
            bias_id = _base_weight_value_id(graph, bias_id)
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
        if node.op not in {"sum", "mean", "min", "max"} or len(node.inputs) != 1 or len(node.outputs) != 1:
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


def apply_jax_semantic_rewrites(graph: IRGraph) -> None:
    _rewrite_jax_batch_norms(graph)
    _rewrite_jax_rms_norms(graph)
    _rewrite_jax_silus(graph)
    apply_import_semantics(graph)
    _rewrite_prenorm_residual_adds_to_clipped(graph)
    _record_jax_semantic_pattern_counts(graph)
    _prune_dead_nodes(graph)
    _legalize_remaining_jax_reductions(graph)
    _rebuild_users(graph)
