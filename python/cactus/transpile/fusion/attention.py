from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.fusion.common import collect_node_ids
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.fusion.linear import match_linear
from cactus.transpile.fusion.rms_norm import match_rms_norm
from cactus.transpile.fusion.rope import match_rope


@dataclass(frozen=True)
class AttentionMatch:
    query_value_id: str
    key_value_id: str
    value_value_id: str
    source_input_value_ids: tuple[str, str, str]
    weight_value_ids: tuple[str | None, str | None, str | None]
    has_rope: bool
    has_qk_norm: bool
    has_gqa_repeat: bool
    is_causal: bool
    scale: float
    window_size: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttentionBlockMatch:
    attention_node_id: str
    query_value_id: str
    key_value_id: str
    value_value_id: str
    mask_value_id: str | None
    gate_value_id: str | None
    output_projection_weight_value_id: str
    output_projection_bias_value_id: str | None
    attention_output_shape: tuple[int, ...]
    is_causal: bool
    additive_mask: bool
    scale: float
    window_size: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class SelfAttentionBlockMatch:
    attention_node_id: str
    hidden_value_id: str
    query_weight_value_id: str
    query_projection_bias_value_id: str | None
    query_add_value_id: str | None
    rel_query_add_value_id: str | None
    key_weight_value_id: str
    key_projection_bias_value_id: str | None
    value_weight_value_id: str
    value_projection_bias_value_id: str | None
    mask_value_id: str | None
    relative_key_input_value_id: str | None
    relative_key_weight_value_id: str | None
    relative_key_projection_bias_value_id: str | None
    gate_value_id: str | None
    output_projection_weight_value_id: str
    output_projection_bias_value_id: str | None
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    relative_key_shape: tuple[int, ...] | None
    attention_output_shape: tuple[int, ...]
    is_causal: bool
    additive_mask: bool
    scale: float
    rel_pos_scale: float | None
    window_size: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ProjectedAttentionInputMatch:
    hidden_value_id: str
    weight_value_id: str
    projection_bias_value_id: str | None
    projected_shape: tuple[int, ...]
    add_value_id: str | None
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RelativeKeyProjectionMatch:
    input_value_id: str
    weight_value_id: str
    projection_bias_value_id: str | None
    projected_shape: tuple[int, ...]
    node_ids: tuple[str, ...]


def match_attention(graph: IRGraph, node: IRNode) -> AttentionMatch | None:
    if node.op not in {"attention", "scaled_dot_product_attention"} or len(node.inputs) < 3:
        return None

    q_info = _extract_attention_input(graph, node.inputs[0], role="q")
    k_info = _extract_attention_input(graph, node.inputs[1], role="k")
    v_info = _extract_attention_input(graph, node.inputs[2], role="v")
    if q_info is None or k_info is None or v_info is None:
        return None

    has_rope = bool(q_info["has_rope"] and k_info["has_rope"])
    has_qk_norm = bool(q_info["has_rms_norm"] and k_info["has_rms_norm"])
    has_gqa = bool(k_info["has_gqa_repeat"] or v_info["has_gqa_repeat"] or node.attrs.get("enable_gqa", False))
    node_ids = {node.id, *q_info["node_ids"], *k_info["node_ids"], *v_info["node_ids"]}

    return AttentionMatch(
        query_value_id=node.inputs[0],
        key_value_id=node.inputs[1],
        value_value_id=node.inputs[2],
        source_input_value_ids=(q_info["source_input"], k_info["source_input"], v_info["source_input"]),
        weight_value_ids=(q_info.get("weight_value_id"), k_info.get("weight_value_id"), v_info.get("weight_value_id")),
        has_rope=has_rope,
        has_qk_norm=has_qk_norm,
        has_gqa_repeat=has_gqa,
        is_causal=bool(node.attrs.get("is_causal", False)),
        scale=float(node.attrs.get("scale", 0.0)),
        window_size=int(node.attrs.get("window_size", 0)),
        node_ids=tuple(sorted(node_ids)),
    )


def match_self_attention_block(graph: IRGraph, node: IRNode) -> SelfAttentionBlockMatch | None:
    projection = match_linear(graph, node)
    if projection is None:
        return None

    output_path = _extract_attention_output_path(graph, projection.input_value_id)
    if output_path is None:
        return None
    attention_node = graph.nodes.get(output_path["attention_node_id"])
    if attention_node is None or attention_node.op not in {"attention", "scaled_dot_product_attention"} or len(attention_node.inputs) < 3:
        return None

    query_match = _extract_projected_attention_input(graph, attention_node.inputs[0], role="q")
    key_match = _extract_projected_attention_input(graph, attention_node.inputs[1], role="k")
    value_match = _extract_projected_attention_input(graph, attention_node.inputs[2], role="v")
    if query_match is None or key_match is None or value_match is None:
        return None

    if not (
        query_match.hidden_value_id == key_match.hidden_value_id == value_match.hidden_value_id
    ):
        return None

    mask_value_id = attention_node.inputs[3] if len(attention_node.inputs) > 3 else None
    rel_query_add_value_id: str | None = None
    relative_key_input_value_id: str | None = None
    relative_key_weight_value_id: str | None = None
    relative_key_projection_bias_value_id: str | None = None
    relative_key_shape: tuple[int, ...] | None = None
    rel_pos_scale: float | None = None
    rel_pos_node_ids: tuple[str, ...] = ()

    if mask_value_id is not None:
        rel_pos_node = producer(graph, strip_passthrough(graph, mask_value_id))
        if rel_pos_node is not None and rel_pos_node.op == "rel_pos_bias" and len(rel_pos_node.inputs) == 2:
            rel_query_match = _extract_projected_attention_input(graph, rel_pos_node.inputs[0], role="q")
            relative_key_match = _extract_relative_key_projection(graph, rel_pos_node.inputs[1])
            if rel_query_match is None or relative_key_match is None:
                return None
            if not (
                rel_query_match.hidden_value_id == query_match.hidden_value_id
                and rel_query_match.weight_value_id == query_match.weight_value_id
                and rel_query_match.projection_bias_value_id == query_match.projection_bias_value_id
                and rel_query_match.projected_shape == query_match.projected_shape
            ):
                return None
            rel_query_add_value_id = rel_query_match.add_value_id
            relative_key_input_value_id = relative_key_match.input_value_id
            relative_key_weight_value_id = relative_key_match.weight_value_id
            relative_key_projection_bias_value_id = relative_key_match.projection_bias_value_id
            relative_key_shape = relative_key_match.projected_shape
            rel_pos_scale = float(rel_pos_node.attrs.get("scale", 1.0))
            rel_pos_node_ids = collect_node_ids(
                rel_pos_node.id,
                rel_pos_node.meta.get("rel_pos_bias_nodes", ()),
                rel_query_match.node_ids,
                relative_key_match.node_ids,
            )
            mask_value_id = None

    return SelfAttentionBlockMatch(
        attention_node_id=attention_node.id,
        hidden_value_id=query_match.hidden_value_id,
        query_weight_value_id=query_match.weight_value_id,
        query_projection_bias_value_id=query_match.projection_bias_value_id,
        query_add_value_id=query_match.add_value_id,
        rel_query_add_value_id=rel_query_add_value_id,
        key_weight_value_id=key_match.weight_value_id,
        key_projection_bias_value_id=key_match.projection_bias_value_id,
        value_weight_value_id=value_match.weight_value_id,
        value_projection_bias_value_id=value_match.projection_bias_value_id,
        mask_value_id=mask_value_id,
        relative_key_input_value_id=relative_key_input_value_id,
        relative_key_weight_value_id=relative_key_weight_value_id,
        relative_key_projection_bias_value_id=relative_key_projection_bias_value_id,
        gate_value_id=output_path["gate_value_id"],
        output_projection_weight_value_id=projection.weight_value_id,
        output_projection_bias_value_id=projection.bias_value_id,
        query_shape=query_match.projected_shape,
        key_shape=key_match.projected_shape,
        value_shape=value_match.projected_shape,
        relative_key_shape=relative_key_shape,
        attention_output_shape=output_path["attention_output_shape"],
        is_causal=bool(attention_node.attrs.get("is_causal", True)),
        additive_mask=bool(attention_node.attrs.get("additive_mask", False)),
        scale=float(attention_node.attrs.get("scale", 0.0)),
        rel_pos_scale=rel_pos_scale,
        window_size=int(attention_node.attrs.get("window_size", 0)),
        node_ids=collect_node_ids(
            attention_node.id,
            projection.node_ids,
            output_path["node_ids"],
            query_match.node_ids,
            key_match.node_ids,
            value_match.node_ids,
            rel_pos_node_ids,
        ),
    )


def match_attention_block(graph: IRGraph, node: IRNode) -> AttentionBlockMatch | None:
    projection = match_linear(graph, node)
    if projection is None:
        return None

    output_path = _extract_attention_output_path(graph, projection.input_value_id)
    if output_path is None:
        return None
    attention_node = graph.nodes.get(output_path["attention_node_id"])
    if attention_node is None or attention_node.op not in {"attention", "scaled_dot_product_attention"} or len(attention_node.inputs) < 3:
        return None

    return AttentionBlockMatch(
        attention_node_id=attention_node.id,
        query_value_id=attention_node.inputs[0],
        key_value_id=attention_node.inputs[1],
        value_value_id=attention_node.inputs[2],
        mask_value_id=attention_node.inputs[3] if len(attention_node.inputs) > 3 else None,
        gate_value_id=output_path["gate_value_id"],
        output_projection_weight_value_id=projection.weight_value_id,
        output_projection_bias_value_id=projection.bias_value_id,
        attention_output_shape=output_path["attention_output_shape"],
        is_causal=bool(attention_node.attrs.get("is_causal", True)),
        additive_mask=bool(attention_node.attrs.get("additive_mask", False)),
        scale=float(attention_node.attrs.get("scale", 0.0)),
        window_size=int(attention_node.attrs.get("window_size", 0)),
        node_ids=tuple(sorted({attention_node.id, *projection.node_ids, *output_path["node_ids"]})),
    )


def _extract_attention_output_path(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    current = value_id
    gate_value_id: str | None = None

    while True:
        node = producer(graph, current)
        if node is None:
            return None
        if node.op in {"precision_cast", "contiguous"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        if node.op == "type_as" and len(node.inputs) >= 1:
            current = node.inputs[0]
            continue
        break

    node = producer(graph, current)
    if node is not None and node.op == "multiply" and len(node.inputs) == 2:
        for attn_candidate, gate_candidate in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            attn_match = _extract_attention_output_path(graph, attn_candidate)
            if attn_match is None:
                continue
            attn_match["gate_value_id"] = gate_candidate
            attn_match["node_ids"] = tuple(sorted({node.id, *attn_match["node_ids"]}))
            return attn_match

    reshape_node = producer(graph, current)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"} or len(reshape_node.inputs) != 1:
        return None

    attention_output_shape = tuple(int(v) for v in reshape_node.attrs.get("shape", ()))
    current = reshape_node.inputs[0]
    transpose_node = producer(graph, current)
    if transpose_node is None:
        return None

    if transpose_node.op == "transpose":
        if int(transpose_node.attrs.get("dim0", -1)) != 1 or int(transpose_node.attrs.get("dim1", -1)) != 2:
            return None
        current = transpose_node.inputs[0]
    elif transpose_node.op == "permute":
        permutation = tuple(int(v) for v in transpose_node.attrs.get("permutation", ()))
        if permutation != (0, 2, 1, 3):
            return None
        current = transpose_node.inputs[0]
    else:
        return None

    attention_node = producer(graph, current)
    if attention_node is None or attention_node.op not in {"attention", "scaled_dot_product_attention"}:
        return None

    return {
        "attention_node_id": attention_node.id,
        "gate_value_id": gate_value_id,
        "attention_output_shape": attention_output_shape,
        "node_ids": tuple(sorted({reshape_node.id, transpose_node.id})),
    }


def _extract_projected_attention_input(graph: IRGraph, value_id: str, *, role: str) -> _ProjectedAttentionInputMatch | None:
    current = strip_passthrough(graph, value_id)
    add_node = producer(graph, current)
    if add_node is not None and add_node.op == "add" and len(add_node.inputs) == 2:
        for projected_candidate, add_candidate in ((add_node.inputs[0], add_node.inputs[1]), (add_node.inputs[1], add_node.inputs[0])):
            projected = _extract_transposed_projection(graph, projected_candidate)
            if projected is None:
                continue
            if not _looks_like_attention_addend(graph, add_candidate, projected.projected_shape):
                continue
            return _ProjectedAttentionInputMatch(
                hidden_value_id=projected.hidden_value_id,
                weight_value_id=projected.weight_value_id,
                projection_bias_value_id=projected.projection_bias_value_id,
                projected_shape=projected.projected_shape,
                add_value_id=strip_passthrough(graph, add_candidate),
                node_ids=collect_node_ids(add_node.id, projected.node_ids),
            )

        if role == "q":
            return None

    return _extract_transposed_projection(graph, current)


def _extract_transposed_projection(graph: IRGraph, value_id: str) -> _ProjectedAttentionInputMatch | None:
    current = strip_passthrough(graph, value_id)
    layout_node = producer(graph, current)
    if layout_node is None or len(layout_node.inputs) != 1:
        return None

    if layout_node.op == "permute":
        permutation = tuple(int(v) for v in layout_node.attrs.get("permutation", ()))
        if permutation != (0, 2, 1, 3):
            return None
    elif layout_node.op == "transpose":
        if int(layout_node.attrs.get("dim0", -1)) != 1 or int(layout_node.attrs.get("dim1", -1)) != 2:
            return None
    else:
        return None

    projected_value_id = strip_passthrough(graph, layout_node.inputs[0])
    projected_value = graph.values.get(projected_value_id)
    if projected_value is None or projected_value.shape is None or len(projected_value.shape) != 4:
        return None

    reshape_node = producer(graph, projected_value_id)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"} or len(reshape_node.inputs) != 1:
        return None

    linear_node = producer(graph, reshape_node.inputs[0])
    if linear_node is None:
        return None
    linear = match_linear(graph, linear_node)
    if linear is None:
        return None

    return _ProjectedAttentionInputMatch(
        hidden_value_id=linear.input_value_id,
        weight_value_id=linear.weight_value_id,
        projection_bias_value_id=linear.bias_value_id,
        projected_shape=tuple(int(v) for v in projected_value.shape),
        add_value_id=None,
        node_ids=collect_node_ids(layout_node.id, reshape_node.id, linear.node_ids),
    )


def _extract_relative_key_projection(graph: IRGraph, value_id: str) -> _RelativeKeyProjectionMatch | None:
    current = strip_passthrough(graph, value_id)
    projected_value = graph.values.get(current)
    if projected_value is None or projected_value.shape is None or len(projected_value.shape) != 4:
        return None

    reshape_node = producer(graph, current)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"} or len(reshape_node.inputs) != 1:
        return None

    linear_node = producer(graph, reshape_node.inputs[0])
    if linear_node is None:
        return None
    linear = match_linear(graph, linear_node)
    if linear is None:
        return None

    return _RelativeKeyProjectionMatch(
        input_value_id=linear.input_value_id,
        weight_value_id=linear.weight_value_id,
        projection_bias_value_id=linear.bias_value_id,
        projected_shape=tuple(int(v) for v in projected_value.shape),
        node_ids=collect_node_ids(reshape_node.id, linear.node_ids),
    )


def _looks_like_attention_addend(graph: IRGraph, value_id: str, target_shape: tuple[int, ...]) -> bool:
    value = graph.values.get(strip_passthrough(graph, value_id))
    if value is None or value.shape is None:
        return False

    shape = tuple(int(v) for v in value.shape)
    if len(shape) != 4 or len(target_shape) != 4:
        return False

    batch, seq_len, heads, head_dim = target_shape
    if shape[0] not in (1, batch) or shape[3] != head_dim:
        return False

    # Allow either BHSD-style tensors like [B, H, 1, D] or BSHD-style tensors like [B, 1, H, D].
    if shape[1] == heads and shape[2] in (1, seq_len):
        return True
    if shape[1] in (1, seq_len) and shape[2] == heads:
        return True
    return False


def _extract_attention_input(graph: IRGraph, value_id: str, *, role: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    current = strip_passthrough(graph, value_id)
    has_gqa_repeat = False

    while True:
        node = producer(graph, current)
        if node is None:
            return {
                "source_input": current,
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        node_ids.add(node.id)

        if node.op in {"reshape", "view", "transpose", "permute"}:
            current = node.inputs[0]
            continue

        rope = match_rope(graph, current)
        if rope is not None:
            rope_info = _extract_attention_input(graph, rope.input_value_id, role=role)
            if rope_info is None:
                return None
            rope_info["has_rope"] = True
            rope_info["node_ids"] = tuple(sorted(set(rope_info["node_ids"]) | node_ids | set(rope.node_ids)))
            rope_info["has_gqa_repeat"] = bool(rope_info["has_gqa_repeat"] or has_gqa_repeat)
            return rope_info

        rms = match_rms_norm(graph, node)
        if rms is not None:
            if role == "v":
                return None
            rms_info = _extract_attention_input(graph, rms.input_value_id, role=role)
            if rms_info is None:
                return None
            rms_info["has_rms_norm"] = True
            rms_info["node_ids"] = tuple(sorted(set(rms_info["node_ids"]) | node_ids | set(rms.node_ids)))
            rms_info["has_gqa_repeat"] = bool(rms_info["has_gqa_repeat"] or has_gqa_repeat)
            return rms_info

        if _looks_like_gqa_repeat(graph, current):
            has_gqa_repeat = True
            current = _unwrap_gqa_repeat(graph, current)
            if current is None:
                return None
            continue

        linear_node = producer(graph, current)
        if linear_node is None:
            return {
                "source_input": strip_passthrough(graph, current),
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        linear = match_linear(graph, linear_node)
        if linear is None:
            return {
                "source_input": strip_passthrough(graph, current),
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        return {
            "source_input": strip_passthrough(graph, linear.input_value_id),
            "weight_value_id": linear.weight_value_id,
            "has_rope": False,
            "has_rms_norm": False,
            "has_gqa_repeat": has_gqa_repeat,
            "node_ids": tuple(sorted(set(node_ids) | set(linear.node_ids))),
        }


def _looks_like_gqa_repeat(graph: IRGraph, value_id: str) -> bool:
    return _unwrap_gqa_repeat(graph, value_id) is not None


def _unwrap_gqa_repeat(graph: IRGraph, value_id: str) -> str | None:
    current = strip_passthrough(graph, value_id)
    reshape_node = producer(graph, current)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"}:
        return None
    current = reshape_node.inputs[0]
    expand_node = producer(graph, current)
    if expand_node is None or expand_node.op != "expand":
        return None
    current = expand_node.inputs[0]
    while True:
        node = producer(graph, current)
        if node is None:
            return None
        if node.op == "slice":
            current = node.inputs[0]
            continue
        if node.op != "unsqueeze":
            return None
        if int(node.attrs.get("dim", -999)) != 2:
            return None
        break
    base_value_id = strip_passthrough(graph, node.inputs[0])
    base_shape = graph.values.get(base_value_id).shape if graph.values.get(base_value_id) is not None else None
    target_shape = tuple(int(v) for v in expand_node.attrs.get("shape", ()))
    if base_shape is None or len(base_shape) != 4 or len(target_shape) != 5:
        return None
    expected = (int(base_shape[0]), int(base_shape[1]), int(target_shape[2]), int(base_shape[2]), int(base_shape[3]))
    if expected != target_shape:
        return None
    return base_value_id
