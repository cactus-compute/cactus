from __future__ import annotations

from dataclasses import dataclass

import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough


@dataclass(frozen=True)
class RMSNormMatch:
    input_value_id: str
    weight_value_id: str | None
    weight_offset: float
    eps: float
    node_ids: tuple[str, ...]


def match_rms_norm(graph: IRGraph, node: IRNode) -> RMSNormMatch | None:
    anchor_value_id = node.inputs[0] if node.op == "type_as" and node.inputs else node.outputs[0]
    anchor_value_id = strip_passthrough(graph, anchor_value_id)
    anchor_node = producer(graph, anchor_value_id)
    if anchor_node is None or anchor_node.op != "multiply" or len(anchor_node.inputs) != 2:
        return None

    for normed_value_id, scale_value_id in (
        (anchor_node.inputs[0], anchor_node.inputs[1]),
        (anchor_node.inputs[1], anchor_node.inputs[0]),
    ):
        scale = _extract_scale(graph, scale_value_id)
        norm = _extract_normed_value(graph, normed_value_id)
        if scale is None or norm is None:
            continue
        if strip_passthrough(graph, norm["input_value_id"]) != strip_passthrough(graph, norm["pow_input_value_id"]):
            continue
        return RMSNormMatch(
            input_value_id=norm["input_value_id"],
            weight_value_id=scale["weight_value_id"],
            weight_offset=float(scale["offset"]),
            eps=float(norm["eps"]),
            node_ids=tuple(sorted({anchor_node.id, *norm["node_ids"], *scale["node_ids"]})),
        )

    norm = _extract_normed_value(graph, anchor_node.outputs[0])
    if norm is None:
        return None
    if strip_passthrough(graph, norm["input_value_id"]) != strip_passthrough(graph, norm["pow_input_value_id"]):
        return None
    return RMSNormMatch(
        input_value_id=norm["input_value_id"],
        weight_value_id=None,
        weight_offset=0.0,
        eps=float(norm["eps"]),
        node_ids=tuple(sorted({anchor_node.id, *norm["node_ids"]})),
    )


def _extract_scale(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    current = value_id
    offset = 0.0
    node_ids: set[str] = set()
    producer_node = producer(graph, current)
    if producer_node is not None and producer_node.op == "scalar_add":
        add_value = float(producer_node.attrs.get("value", 0.0))
        if add_value in (0.0, 1.0):
            offset = add_value
            node_ids.add(producer_node.id)
            current = producer_node.inputs[0]
    current = strip_passthrough(graph, current)
    if current not in graph.constants or not isinstance(graph.constants[current], torch.Tensor):
        return None
    return {"weight_value_id": current, "offset": offset, "node_ids": tuple(sorted(node_ids))}


def _extract_normed_value(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    mul = producer(graph, strip_passthrough(graph, value_id))
    if mul is None or mul.op != "multiply" or len(mul.inputs) != 2:
        return None
    for x_candidate, rsqrt_candidate in ((mul.inputs[0], mul.inputs[1]), (mul.inputs[1], mul.inputs[0])):
        rsqrt = _extract_rsqrt_chain(graph, rsqrt_candidate)
        if rsqrt is None:
            continue
        if strip_passthrough(graph, x_candidate) != strip_passthrough(graph, rsqrt["pow_input_value_id"]):
            continue
        return {
            "input_value_id": strip_passthrough(graph, x_candidate),
            "pow_input_value_id": rsqrt["pow_input_value_id"],
            "eps": rsqrt["eps"],
            "node_ids": tuple(sorted({mul.id, *rsqrt["node_ids"]})),
        }
    return None


def _extract_rsqrt_chain(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    pow_node = producer(graph, strip_passthrough(graph, value_id))
    if pow_node is None or pow_node.op != "pow" or float(pow_node.attrs.get("exponent", 0.0)) != -0.5:
        return None
    add_node = producer(graph, pow_node.inputs[0])
    if add_node is None or add_node.op != "scalar_add":
        return None
    mean_node = producer(graph, add_node.inputs[0])
    if mean_node is None or mean_node.op != "mean" or not bool(mean_node.attrs.get("keepdim", False)):
        return None
    pow2_node = producer(graph, mean_node.inputs[0])
    if pow2_node is None or pow2_node.op != "pow" or float(pow2_node.attrs.get("exponent", 0.0)) != 2.0:
        return None
    return {
        "pow_input_value_id": pow2_node.inputs[0],
        "eps": float(add_node.attrs.get("value", 0.0)),
        "node_ids": (pow_node.id, add_node.id, mean_node.id, pow2_node.id),
    }
