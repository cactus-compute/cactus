from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough


@dataclass(frozen=True)
class LinearMatch:
    input_value_id: str
    weight_value_id: str
    bias_value_id: str | None
    output_value_id: str
    node_ids: tuple[str, ...]
    kind: str


def match_linear(graph: IRGraph, node: IRNode) -> LinearMatch | None:
    if node.op == "linear" and len(node.inputs) >= 2 and len(node.outputs) == 1:
        return LinearMatch(
            input_value_id=strip_passthrough(graph, node.inputs[0]),
            weight_value_id=node.inputs[1],
            bias_value_id=node.inputs[2] if bool(node.attrs.get("has_bias", False)) and len(node.inputs) > 2 else None,
            output_value_id=node.outputs[0],
            node_ids=(node.id,),
            kind="linear",
        )

    if node.op == "addmm" and len(node.inputs) == 3 and len(node.outputs) == 1:
        return LinearMatch(
            input_value_id=strip_passthrough(graph, node.inputs[1]),
            weight_value_id=node.inputs[2],
            bias_value_id=node.inputs[0],
            output_value_id=node.outputs[0],
            node_ids=(node.id,),
            kind="addmm",
        )

    if node.op != "add" or len(node.inputs) != 2 or len(node.outputs) != 1:
        return None

    for matmul_value_id, bias_value_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
        matmul_node = producer(graph, strip_passthrough(graph, matmul_value_id))
        if matmul_node is None or matmul_node.op != "matmul" or len(matmul_node.inputs) != 2:
            continue
        return LinearMatch(
            input_value_id=strip_passthrough(graph, matmul_node.inputs[0]),
            weight_value_id=matmul_node.inputs[1],
            bias_value_id=bias_value_id,
            output_value_id=node.outputs[0],
            node_ids=tuple(sorted({node.id, matmul_node.id})),
            kind="matmul_add",
        )

    return None
