from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_layout_passthrough
from cactus.transpile.fusion.common import strip_passthrough


@dataclass(frozen=True)
class RoPEMatch:
    input_value_id: str
    theta: float
    position_offset: int
    partial: bool
    node_ids: tuple[str, ...]


def match_rope(graph: IRGraph, value_id: str) -> RoPEMatch | None:
    desc = _extract_rope_descriptor(graph, value_id)
    if desc is None:
        return None
    return RoPEMatch(
        input_value_id=strip_passthrough(graph, desc["input_value_id"]),
        theta=float(desc["theta"]),
        position_offset=int(desc["position_offset"]),
        partial=bool(desc.get("partial_rope", False)),
        node_ids=tuple(sorted(set(desc.get("node_ids", ())))),
    )


def _extract_rope_descriptor(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    current = strip_layout_passthrough(graph, value_id)
    node = producer(graph, current)
    if node is None:
        return None
    if node.op == "rope":
        return {
            "input_value_id": node.inputs[0],
            "theta": float(node.attrs.get("theta", 0.0)),
            "position_offset": int(node.attrs.get("position_offset", 0)),
            "partial_rope": False,
            "node_ids": (node.id,),
        }
    classic = _extract_classic_rope_descriptor(graph, current)
    if classic is not None:
        classic["partial_rope"] = False
        return classic
    if node.op == "cat" and len(node.inputs) == 2:
        for rope_branch_value_id, passthrough_value_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            rope_branch = _extract_rope_descriptor(graph, rope_branch_value_id)
            if rope_branch is None:
                continue
            passthrough_node = producer(graph, strip_layout_passthrough(graph, passthrough_value_id))
            if passthrough_node is None or passthrough_node.op != "slice":
                continue
            if strip_passthrough(graph, passthrough_node.inputs[0]) != strip_passthrough(graph, rope_branch["input_value_id"]):
                continue
            rope_branch["partial_rope"] = True
            rope_branch["node_ids"] = tuple(sorted({node.id, passthrough_node.id, *rope_branch["node_ids"]}))
            return rope_branch
    return None


def _extract_classic_rope_descriptor(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    node = producer(graph, strip_layout_passthrough(graph, value_id))
    if node is None or node.op != "add" or len(node.inputs) != 2:
        return None
    for direct_mult_id, rotated_mult_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        direct_mult = producer(graph, direct_mult_id)
        rotated_mult = producer(graph, rotated_mult_id)
        if direct_mult is None or direct_mult.op != "multiply" or rotated_mult is None or rotated_mult.op != "multiply":
            continue
        direct_match = _extract_direct_rope_branch(graph, direct_mult)
        rotated_match = _extract_rotated_rope_branch(graph, rotated_mult)
        if direct_match is None or rotated_match is None:
            continue
        input_value_id = strip_passthrough(graph, direct_match["input_value_id"])
        if input_value_id != strip_passthrough(graph, rotated_match["input_value_id"]):
            continue
        cos_info = _extract_rope_trig(graph, direct_match["trig_value_id"], expected="scalar_cos")
        sin_info = _extract_rope_trig(graph, rotated_match["trig_value_id"], expected="scalar_sin")
        if cos_info is None or sin_info is None:
            continue
        if abs(float(cos_info["theta"]) - float(sin_info["theta"])) > 1e-2:
            continue
        if int(cos_info["position_offset"]) != int(sin_info["position_offset"]):
            continue
        return {
            "input_value_id": input_value_id,
            "theta": cos_info["theta"],
            "position_offset": cos_info["position_offset"],
            "node_ids": tuple(sorted({node.id, *direct_match["node_ids"], *rotated_match["node_ids"], *cos_info["node_ids"], *sin_info["node_ids"]})),
        }
    return None


def _extract_direct_rope_branch(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    for input_value_id, trig_value_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        if _unwrap_trig_value(graph, trig_value_id, expected="scalar_cos") is not None:
            return {"input_value_id": input_value_id, "trig_value_id": trig_value_id, "node_ids": (node.id,)}
    return None


def _extract_rotated_rope_branch(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    for rotated_value_id, trig_value_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        input_value_id = _extract_rotate_half_source(graph, rotated_value_id)
        if input_value_id is None:
            continue
        if _unwrap_trig_value(graph, trig_value_id, expected="scalar_sin") is None:
            continue
        return {"input_value_id": input_value_id, "trig_value_id": trig_value_id, "node_ids": (node.id,)}
    return None


def _extract_rotate_half_source(graph: IRGraph, value_id: str) -> str | None:
    cat_node = producer(graph, strip_passthrough(graph, value_id))
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2:
        return None
    neg_node = producer(graph, cat_node.inputs[0])
    right_slice = producer(graph, cat_node.inputs[1])
    if neg_node is None or neg_node.op != "scalar_multiply" or float(neg_node.attrs.get("value", 0.0)) != -1.0:
        return None
    left_slice = producer(graph, neg_node.inputs[0])
    if left_slice is None or right_slice is None or left_slice.op != "slice" or right_slice.op != "slice":
        return None
    if strip_passthrough(graph, left_slice.inputs[0]) != strip_passthrough(graph, right_slice.inputs[0]):
        return None
    return left_slice.inputs[0]


def _extract_rope_trig(graph: IRGraph, value_id: str, *, expected: str) -> dict[str, object] | None:
    trig_info = _unwrap_trig_value(graph, value_id, expected=expected)
    if trig_info is None:
        return None
    trig_node = trig_info["trig_node"]
    cat_node = producer(graph, trig_node.inputs[0])
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2 or cat_node.inputs[0] != cat_node.inputs[1]:
        return None
    angle_info = _extract_rope_angle_source(graph, cat_node.inputs[0])
    if angle_info is None:
        return None
    matmul_node = angle_info["matmul_node"]
    inv_freq_const_id = None
    arange_node = None
    for input_id in matmul_node.inputs:
        if inv_freq_const_id is None:
            inv_freq_const_id = _find_constant_ancestor(graph, input_id)
        if arange_node is None:
            arange_node = _find_arange_ancestor(graph, input_id)
    if inv_freq_const_id is None or arange_node is None:
        return None
    theta = _infer_rope_theta(graph.constants[inv_freq_const_id])
    if theta is None:
        return None
    return {
        "theta": theta,
        "position_offset": int(arange_node.attrs.get("start", 0)),
        "node_ids": tuple(sorted({trig_node.id, cat_node.id, matmul_node.id})),
    }


def _unwrap_trig_value(graph: IRGraph, value_id: str, *, expected: str) -> dict[str, object] | None:
    current = value_id
    while True:
        node = producer(graph, current)
        if node is None:
            return None
        if node.op in {"unsqueeze", "precision_cast", "reshape", "view", "expand"}:
            current = node.inputs[0]
            continue
        if node.op == "type_as":
            current = node.inputs[0]
            continue
        if node.op == "scalar_multiply":
            current = node.inputs[0]
            continue
        if node.op != expected:
            return None
        return {"trig_node": node}


def _extract_rope_angle_source(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    current = value_id
    while True:
        current = strip_layout_passthrough(graph, current)
        node = producer(graph, current)
        if node is None:
            return None
        if node.op == "permute":
            matmul_node = producer(graph, node.inputs[0])
            if matmul_node is None or matmul_node.op != "matmul":
                return None
            return {"matmul_node": matmul_node}
        if node.op == "matmul":
            return {"matmul_node": node}
        if node.op in {"index", "slice"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        return None


def _find_constant_ancestor(graph: IRGraph, value_id: str) -> str | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = strip_layout_passthrough(graph, current)
        if current in graph.constants and isinstance(graph.constants[current], torch.Tensor):
            return current
        node = producer(graph, current)
        if node is None or len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _find_arange_ancestor(graph: IRGraph, value_id: str) -> IRNode | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = strip_layout_passthrough(graph, current)
        node = producer(graph, current)
        if node is None:
            return None
        if node.op == "arange":
            return node
        if current in graph.constants and isinstance(graph.constants[current], torch.Tensor):
            return None
        if len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _infer_rope_theta(value: Any) -> float | None:
    if not isinstance(value, torch.Tensor):
        return None
    flat = value.detach().cpu().float().reshape(-1)
    if flat.numel() < 2:
        return None
    second = float(flat[1].item())
    if second <= 0.0:
        return None
    inv_count = flat.numel()
    return float((1.0 / second) ** inv_count)
