from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.fusion.linear import match_linear


@dataclass(frozen=True)
class GatedMLPMatch:
    input_value_id: str
    gate_weight_value_id: str
    up_weight_value_id: str
    down_weight_value_id: str
    activation: str
    bias_value_ids: tuple[str | None, str | None, str | None]
    node_ids: tuple[str, ...]


def match_gated_mlp(graph: IRGraph, node: IRNode) -> GatedMLPMatch | None:
    down = match_linear(graph, node)
    if down is None:
        return None

    mul_node = graph.nodes.get(graph.values[down.input_value_id].producer) if graph.values.get(down.input_value_id) is not None else None
    if mul_node is None or mul_node.op != "multiply" or len(mul_node.inputs) != 2:
        return None

    for activated_value_id, up_value_id in ((mul_node.inputs[0], mul_node.inputs[1]), (mul_node.inputs[1], mul_node.inputs[0])):
        activated_node = graph.nodes.get(graph.values[strip_passthrough(graph, activated_value_id)].producer) if graph.values.get(strip_passthrough(graph, activated_value_id)) is not None else None
        if activated_node is None or activated_node.op not in {"gelu", "silu"}:
            continue

        gate = match_linear(graph, graph.nodes[graph.values[strip_passthrough(graph, activated_node.inputs[0])].producer]) if graph.values.get(strip_passthrough(graph, activated_node.inputs[0])) is not None and graph.values[strip_passthrough(graph, activated_node.inputs[0])].producer in graph.nodes else None
        up = match_linear(graph, graph.nodes[graph.values[strip_passthrough(graph, up_value_id)].producer]) if graph.values.get(strip_passthrough(graph, up_value_id)) is not None and graph.values[strip_passthrough(graph, up_value_id)].producer in graph.nodes else None
        if gate is None or up is None:
            continue
        if strip_passthrough(graph, gate.input_value_id) != strip_passthrough(graph, up.input_value_id):
            continue

        return GatedMLPMatch(
            input_value_id=strip_passthrough(graph, gate.input_value_id),
            gate_weight_value_id=gate.weight_value_id,
            up_weight_value_id=up.weight_value_id,
            down_weight_value_id=down.weight_value_id,
            activation=activated_node.op,
            bias_value_ids=(gate.bias_value_id, up.bias_value_id, down.bias_value_id),
            node_ids=tuple(sorted({*gate.node_ids, activated_node.id, *up.node_ids, mul_node.id, *down.node_ids})),
        )

    return None
