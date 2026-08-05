from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.fusion.linear import match_linear
from cactus.transpile.fusion.rms_norm import match_rms_norm
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode


@dataclass(frozen=True)
class GatedDeltaNetMatch:
    output_value_id: str
    normalized_input_value_id: str
    qkv_weight_value_id: str
    a_weight_value_id: str
    b_weight_value_id: str
    norm_weight_value_id: str
    z_weight_value_id: str | None
    dt_bias_value_id: str | None
    a_log_value_id: str | None
    conv_weight_value_id: str | None
    num_k_heads: int
    num_v_heads: int
    key_dim: int
    value_dim: int
    eps: float
    chunk_size: int
    mode: str
    node_ids: tuple[str, ...]


def match_gated_deltanet(graph: IRGraph, node: IRNode) -> GatedDeltaNetMatch | None:
    if node.op not in {"view", "reshape"} or len(node.outputs) != 1:
        return None
    output_value = graph.values.get(node.outputs[0])
    if output_value is None or output_value.shape is None or len(output_value.shape) != 3:
        return None

    body_value_id = strip_passthrough(graph, node.inputs[0])
    body_node = producer(graph, body_value_id)
    if body_node is None:
        return None

    z_weight_value_id: str | None = None
    rms_match = None
    normalized_input_value_id: str | None = None
    node_ids: set[str] = {node.id}

    if body_node.op == "multiply" and len(body_node.inputs) == 2:
        for rms_value_id, z_value_id in ((body_node.inputs[0], body_node.inputs[1]), (body_node.inputs[1], body_node.inputs[0])):
            rms_match = _match_rms_from_value(graph, rms_value_id)
            if rms_match is None:
                continue
            z_weight_value_id, normalized_input_value_id, z_node_ids = _extract_z_branch(graph, z_value_id)
            if normalized_input_value_id is None:
                continue
            node_ids.update(z_node_ids)
            node_ids.add(body_node.id)
            break
        if rms_match is None or normalized_input_value_id is None:
            return None
    else:
        rms_match = _match_rms_from_value(graph, body_value_id)
        if rms_match is None:
            return None

    y2d_value_id = rms_match.input_value_id
    if rms_match.weight_value_id is None:
        return None
    norm_weight_value_id = rms_match.weight_value_id
    eps = float(rms_match.eps)
    node_ids.update(rms_match.node_ids)

    if normalized_input_value_id is None:
        normalized_input_value_id = _infer_normalized_input_from_y_branch(graph, y2d_value_id)
        if normalized_input_value_id is None:
            return None

    qkv_info = _find_qkv_linear(graph, normalized_input_value_id)
    if qkv_info is None:
        return None
    qkv_weight_value_id, conv_weight_value_id, num_k_heads, key_dim, num_v_heads, value_dim, qkv_node_ids = qkv_info
    node_ids.update(qkv_node_ids)

    a_info = _find_a_linear(graph, normalized_input_value_id)
    b_info = _find_b_linear(graph, normalized_input_value_id)
    if a_info is None or b_info is None:
        return None
    a_weight_value_id, dt_bias_value_id, a_log_value_id, a_node_ids = a_info
    b_weight_value_id, b_node_ids = b_info
    node_ids.update(a_node_ids)
    node_ids.update(b_node_ids)

    mode, chunk_size = _classify_mode(graph, y2d_value_id, match_node_ids=node_ids)

    return GatedDeltaNetMatch(
        output_value_id=node.outputs[0],
        normalized_input_value_id=normalized_input_value_id,
        qkv_weight_value_id=qkv_weight_value_id,
        a_weight_value_id=a_weight_value_id,
        b_weight_value_id=b_weight_value_id,
        norm_weight_value_id=norm_weight_value_id,
        z_weight_value_id=z_weight_value_id,
        dt_bias_value_id=dt_bias_value_id,
        a_log_value_id=a_log_value_id,
        conv_weight_value_id=conv_weight_value_id,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        eps=eps,
        chunk_size=chunk_size,
        mode=mode,
        node_ids=tuple(sorted(node_ids)),
    )


def _match_rms_from_value(graph: IRGraph, value_id: str):
    node = producer(graph, strip_passthrough(graph, value_id))
    if node is None:
        return None
    return match_rms_norm(graph, node)


def _extract_z_branch(graph: IRGraph, value_id: str) -> tuple[str | None, str | None, tuple[str, ...]]:
    current = strip_passthrough(graph, value_id)
    silu_node = producer(graph, current)
    if silu_node is None or silu_node.op != "silu" or len(silu_node.inputs) != 1:
        return None, None, ()
    linear_value_id = strip_passthrough(graph, silu_node.inputs[0])
    linear_node = producer(graph, linear_value_id)
    while linear_node is not None and linear_node.op in {"view", "reshape"} and len(linear_node.inputs) == 1:
        linear_value_id = strip_passthrough(graph, linear_node.inputs[0])
        linear_node = producer(graph, linear_value_id)
    if linear_node is None:
        return None, None, ()
    linear_match = match_linear(graph, linear_node)
    if linear_match is None:
        return None, None, ()
    return (
        linear_match.weight_value_id,
        linear_match.input_value_id,
        tuple(sorted({silu_node.id, *linear_match.node_ids})),
    )


def _infer_normalized_input_from_y_branch(graph: IRGraph, value_id: str) -> str | None:
    current = strip_passthrough(graph, value_id)
    node = producer(graph, current)
    visited: set[str] = set()
    while node is not None and node.id not in visited:
        visited.add(node.id)
        for input_value_id in node.inputs:
            candidate = producer(graph, strip_passthrough(graph, input_value_id))
            if candidate is None:
                continue
            linear_match = match_linear(graph, candidate)
            if linear_match is not None:
                return linear_match.input_value_id
        if len(node.inputs) != 1:
            return None
        node = producer(graph, strip_passthrough(graph, node.inputs[0]))
    return None


def _find_qkv_linear(
    graph: IRGraph,
    normalized_input_value_id: str,
) -> tuple[str, str | None, int, int, int, int, tuple[str, ...]] | None:
    for node_id in graph.order:
        node = graph.nodes[node_id]
        linear_match = match_linear(graph, node)
        if linear_match is None or linear_match.input_value_id != normalized_input_value_id:
            continue
        split_node = _find_descendant_op(graph, linear_match.output_value_id, {"split_with_sizes"}, max_depth=8)
        if split_node is None:
            continue
        split_sizes = tuple(int(v) for v in split_node.attrs.get("sizes", ()))
        if len(split_sizes) != 3:
            continue

        conv_node = _find_descendant_op(graph, linear_match.output_value_id, {"conv1d"}, max_depth=6)
        conv_weight_value_id = conv_node.inputs[1] if conv_node is not None and len(conv_node.inputs) >= 2 else None

        q_shape = None
        v_shape = None
        node_ids = {node.id, split_node.id}
        if conv_node is not None:
            node_ids.add(conv_node.id)

        split_value = graph.values.get(split_node.outputs[0])
        if split_value is None:
            continue
        for getitem_node_id in split_value.users:
            getitem_node = graph.nodes.get(getitem_node_id)
            if getitem_node is None or getitem_node.op != "getitem" or not getitem_node.outputs:
                continue
            shape = _find_rank4_shape(graph, getitem_node.outputs[0], max_depth=5)
            if shape is None:
                continue
            branch_has_sum = _find_descendant_op(graph, getitem_node.outputs[0], {"sum"}, max_depth=8) is not None
            node_ids.add(getitem_node.id)
            if branch_has_sum and q_shape is None:
                q_shape = shape
            elif not branch_has_sum and v_shape is None:
                v_shape = shape

        if q_shape is None or v_shape is None:
            continue

        num_k_heads = int(q_shape[2])
        key_dim = int(q_shape[3])
        num_v_heads = int(v_shape[2])
        value_dim = int(v_shape[3])
        if num_k_heads <= 0 or key_dim <= 0 or num_v_heads <= 0 or value_dim <= 0:
            continue

        return (
            linear_match.weight_value_id,
            conv_weight_value_id,
            num_k_heads,
            key_dim,
            num_v_heads,
            value_dim,
            tuple(sorted(node_ids | set(linear_match.node_ids))),
        )
    return None


def _find_a_linear(
    graph: IRGraph,
    normalized_input_value_id: str,
) -> tuple[str, str | None, str | None, tuple[str, ...]] | None:
    for node_id in graph.order:
        node = graph.nodes[node_id]
        linear_match = match_linear(graph, node)
        if linear_match is None or linear_match.input_value_id != normalized_input_value_id:
            continue
        softplus_node = _find_descendant_op(graph, linear_match.output_value_id, {"softplus"}, max_depth=6)
        if softplus_node is None:
            continue
        dt_bias_value_id = None
        softplus_input_node = producer(graph, strip_passthrough(graph, softplus_node.inputs[0]))
        if softplus_input_node is not None and softplus_input_node.op == "add":
            for input_value_id in softplus_input_node.inputs:
                value = graph.values.get(input_value_id)
                if value is not None and value.producer is None and input_value_id in graph.constants:
                    dt_bias_value_id = input_value_id
                    break
        a_log_value_id = None
        multiply_node = _find_descendant_op(graph, softplus_node.outputs[0], {"multiply"}, max_depth=6)
        if multiply_node is not None:
            for input_value_id in multiply_node.inputs:
                a_log_value_id = _extract_exp_constant_input(graph, input_value_id)
                if a_log_value_id is not None:
                    break
        node_ids = {softplus_node.id, *linear_match.node_ids}
        if softplus_input_node is not None:
            node_ids.add(softplus_input_node.id)
        if multiply_node is not None:
            node_ids.add(multiply_node.id)
        return linear_match.weight_value_id, dt_bias_value_id, a_log_value_id, tuple(sorted(node_ids))
    return None


def _find_b_linear(graph: IRGraph, normalized_input_value_id: str) -> tuple[str, tuple[str, ...]] | None:
    for node_id in graph.order:
        node = graph.nodes[node_id]
        linear_match = match_linear(graph, node)
        if linear_match is None or linear_match.input_value_id != normalized_input_value_id:
            continue
        sigmoid_node = _find_descendant_op(graph, linear_match.output_value_id, {"sigmoid"}, max_depth=4)
        if sigmoid_node is None:
            continue
        return linear_match.weight_value_id, tuple(sorted({sigmoid_node.id, *linear_match.node_ids}))
    return None


def _extract_exp_constant_input(graph: IRGraph, value_id: str) -> str | None:
    node = producer(graph, strip_passthrough(graph, value_id))
    if node is None:
        value = graph.values.get(value_id)
        if value is not None and value.producer is None and value_id in graph.constants:
            return value_id
        return None
    if node.op == "scalar_multiply":
        return _extract_exp_constant_input(graph, node.inputs[0])
    if node.op == "scalar_exp":
        source = strip_passthrough(graph, node.inputs[0])
        value = graph.values.get(source)
        if value is not None and value.producer is None and source in graph.constants:
            return source
    return None


def _find_rank4_shape(graph: IRGraph, value_id: str, *, max_depth: int) -> tuple[int, int, int, int] | None:
    frontier = [(value_id, 0)]
    seen: set[str] = set()
    while frontier:
        current, depth = frontier.pop(0)
        if current in seen or depth > max_depth:
            continue
        seen.add(current)
        value = graph.values.get(current)
        if value is not None and value.shape is not None and len(value.shape) == 4:
            shape = tuple(int(dim) for dim in value.shape)
            return shape  # type: ignore[return-value]
        for user_id in graph.values.get(current, None).users if graph.values.get(current, None) is not None else ():
            user = graph.nodes.get(user_id)
            if user is None or not user.outputs:
                continue
            frontier.extend((output_id, depth + 1) for output_id in user.outputs)
    return None


def _find_descendant_op(graph: IRGraph, value_id: str, targets: set[str], *, max_depth: int) -> IRNode | None:
    frontier = [(value_id, 0)]
    seen: set[str] = set()
    while frontier:
        current, depth = frontier.pop(0)
        if current in seen or depth > max_depth:
            continue
        seen.add(current)
        value = graph.values.get(current)
        if value is None:
            continue
        for user_id in value.users:
            user = graph.nodes.get(user_id)
            if user is None or not user.outputs:
                continue
            if user.op in targets:
                return user
            frontier.extend((output_id, depth + 1) for output_id in user.outputs)
    return None


def _classify_mode(graph: IRGraph, y2d_value_id: str, *, match_node_ids: set[str]) -> tuple[str, int]:
    del y2d_value_id
    chunk_size = _infer_prefill_chunk_size(graph, match_node_ids)
    # Current exported graphs are uncached/full-sequence captures, so fuse the
    # linear-attention block to the prefill kernel by default. Decode should be
    # emitted only once the capture path models cache/state explicitly.
    return "prefill", chunk_size


def _infer_prefill_chunk_size(graph: IRGraph, match_node_ids: set[str]) -> int:
    for node_id in match_node_ids:
        node = graph.nodes.get(node_id)
        if node is None or node.op != "aten.triu.default" or not node.inputs:
            continue
        ones_value = graph.values.get(node.inputs[0])
        if ones_value is not None and ones_value.shape and len(ones_value.shape) >= 1:
            return int(ones_value.shape[-1])
    return 64
