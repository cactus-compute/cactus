from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode


@dataclass(frozen=True)
class FusionMatch:
    kind: str
    anchor_node_id: str
    node_ids: tuple[str, ...]
    value_ids: tuple[str, ...]
    attrs: dict[str, object]


def producer(graph: IRGraph, value_id: str) -> IRNode | None:
    value = graph.values.get(value_id)
    if value is None or value.producer is None:
        return None
    return graph.nodes.get(value.producer)


def strip_passthrough(graph: IRGraph, value_id: str) -> str:
    current = value_id
    while True:
        node = producer(graph, current)
        if node is None:
            return current
        if node.op in {"precision_cast", "contiguous"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        if node.op == "type_as" and len(node.inputs) >= 1:
            current = node.inputs[0]
            continue
        return current


def strip_layout_passthrough(graph: IRGraph, value_id: str) -> str:
    current = value_id
    while True:
        node = producer(graph, current)
        if node is None:
            return current
        if node.op in {"precision_cast", "contiguous", "reshape", "view", "unsqueeze", "expand"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        if node.op == "type_as" and len(node.inputs) >= 1:
            current = node.inputs[0]
            continue
        return current


def collect_node_ids(*parts: Any) -> tuple[str, ...]:
    node_ids: set[str] = set()
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            node_ids.add(part)
            continue
        if isinstance(part, IRNode):
            node_ids.add(part.id)
            continue
        if isinstance(part, (list, tuple, set)):
            for item in part:
                if isinstance(item, str):
                    node_ids.add(item)
                elif isinstance(item, IRNode):
                    node_ids.add(item.id)
    return tuple(sorted(node_ids))
