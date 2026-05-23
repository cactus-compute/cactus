from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.fusion.common import collect_node_ids
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode


@dataclass(frozen=True)
class LSTMCellMatch:
    anchor_node_id: str
    node_ids: tuple[str, ...]
    x_value_id: str
    h_prev_value_id: str
    c_prev_value_id: str
    weight_ih_value_id: str
    weight_hh_value_id: str
    bias_ih_value_id: str
    bias_hh_value_id: str | None
    h_output_value_id: str
    c_output_value_id: str


def match_lstm_cell(graph: IRGraph, node: IRNode) -> LSTMCellMatch | None:
    if node.op != "multiply" or len(node.inputs) != 2 or len(node.outputs) != 1:
        return None

    for output_gate_value_id, c_tanh_value_id in (
        (node.inputs[0], node.inputs[1]),
        (node.inputs[1], node.inputs[0]),
    ):
        output_gate = _extract_gate_activation(graph, output_gate_value_id, expected_activation="sigmoid")
        if output_gate is None or output_gate["index"] != 3:
            continue

        c_tanh_node = producer(graph, strip_passthrough(graph, c_tanh_value_id))
        if c_tanh_node is None or c_tanh_node.op != "tanh" or len(c_tanh_node.inputs) != 1:
            continue

        c_next_value_id = strip_passthrough(graph, c_tanh_node.inputs[0])
        c_next_node = producer(graph, c_next_value_id)
        if c_next_node is None or c_next_node.op != "add" or len(c_next_node.inputs) != 2 or len(c_next_node.outputs) != 1:
            continue

        for forget_term_value_id, input_term_value_id in (
            (c_next_node.inputs[0], c_next_node.inputs[1]),
            (c_next_node.inputs[1], c_next_node.inputs[0]),
        ):
            forget_term = _extract_forget_term(graph, forget_term_value_id)
            input_term = _extract_input_term(graph, input_term_value_id)
            if forget_term is None or input_term is None:
                continue

            input_gate = input_term["input_gate"]
            cell_gate = input_term["cell_gate"]
            forget_gate = forget_term["forget_gate"]
            chunk_node_id = output_gate["chunk_node_id"]
            if (
                input_gate["chunk_node_id"] != chunk_node_id
                or forget_gate["chunk_node_id"] != chunk_node_id
                or cell_gate["chunk_node_id"] != chunk_node_id
            ):
                continue
            if input_gate["index"] != 0 or forget_gate["index"] != 1 or cell_gate["index"] != 2:
                continue

            chunk_node = graph.nodes.get(chunk_node_id)
            if chunk_node is None or len(chunk_node.inputs) != 1:
                continue

            gates = _extract_lstm_gate_sum(graph, chunk_node.inputs[0])
            if gates is None:
                continue

            return LSTMCellMatch(
                anchor_node_id=node.id,
                node_ids=collect_node_ids(
                    node,
                    c_tanh_node,
                    c_next_node,
                    output_gate["node_ids"],
                    forget_gate["node_ids"],
                    input_gate["node_ids"],
                    cell_gate["node_ids"],
                    forget_term["node_ids"],
                    input_term["node_ids"],
                    chunk_node,
                    gates["node_ids"],
                ),
                x_value_id=gates["x_value_id"],
                h_prev_value_id=gates["h_prev_value_id"],
                c_prev_value_id=forget_term["c_prev_value_id"],
                weight_ih_value_id=gates["weight_ih_value_id"],
                weight_hh_value_id=gates["weight_hh_value_id"],
                bias_ih_value_id=gates["bias_value_ids"][0],
                bias_hh_value_id=gates["bias_value_ids"][1] if len(gates["bias_value_ids"]) > 1 else None,
                h_output_value_id=node.outputs[0],
                c_output_value_id=c_next_node.outputs[0],
            )

    return None


def _extract_gate_activation(
    graph: IRGraph,
    value_id: str,
    *,
    expected_activation: str,
) -> dict[str, object] | None:
    activation_node = producer(graph, strip_passthrough(graph, value_id))
    if activation_node is None or activation_node.op != expected_activation or len(activation_node.inputs) != 1:
        return None

    getitem_node = producer(graph, strip_passthrough(graph, activation_node.inputs[0]))
    if getitem_node is None or getitem_node.op != "getitem" or len(getitem_node.inputs) != 1:
        return None

    chunk_node = producer(graph, strip_passthrough(graph, getitem_node.inputs[0]))
    if chunk_node is None or chunk_node.op != "chunk":
        return None
    if int(chunk_node.attrs.get("chunks", 0)) != 4:
        return None

    return {
        "chunk_node_id": chunk_node.id,
        "index": int(getitem_node.attrs.get("index", -1)),
        "preactivation_value_id": getitem_node.inputs[0],
        "node_ids": (activation_node.id, getitem_node.id),
    }


def _extract_forget_term(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    mul_node = producer(graph, strip_passthrough(graph, value_id))
    if mul_node is None or mul_node.op != "multiply" or len(mul_node.inputs) != 2:
        return None

    for gate_value_id, c_prev_value_id in (
        (mul_node.inputs[0], mul_node.inputs[1]),
        (mul_node.inputs[1], mul_node.inputs[0]),
    ):
        forget_gate = _extract_gate_activation(graph, gate_value_id, expected_activation="sigmoid")
        if forget_gate is None:
            continue
        return {
            "forget_gate": forget_gate,
            "c_prev_value_id": strip_passthrough(graph, c_prev_value_id),
            "node_ids": (mul_node.id,),
        }
    return None


def _extract_input_term(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    mul_node = producer(graph, strip_passthrough(graph, value_id))
    if mul_node is None or mul_node.op != "multiply" or len(mul_node.inputs) != 2:
        return None

    for input_gate_value_id, cell_gate_value_id in (
        (mul_node.inputs[0], mul_node.inputs[1]),
        (mul_node.inputs[1], mul_node.inputs[0]),
    ):
        input_gate = _extract_gate_activation(graph, input_gate_value_id, expected_activation="sigmoid")
        if input_gate is None:
            continue
        cell_tanh_node = producer(graph, strip_passthrough(graph, cell_gate_value_id))
        if cell_tanh_node is None or cell_tanh_node.op != "tanh" or len(cell_tanh_node.inputs) != 1:
            continue
        cell_gate = _extract_gate_activation(graph, cell_gate_value_id, expected_activation="tanh")
        if cell_gate is None:
            continue
        return {
            "input_gate": input_gate,
            "cell_gate": cell_gate,
            "node_ids": (mul_node.id, cell_tanh_node.id),
        }
    return None


def _extract_lstm_gate_sum(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    terms, node_ids = _flatten_add_terms(graph, strip_passthrough(graph, value_id))
    linear_nodes: list[IRNode] = []
    bias_value_ids: list[str] = []
    for term_value_id in terms:
        term_value_id = strip_passthrough(graph, term_value_id)
        term_node = producer(graph, term_value_id)
        if term_node is not None and term_node.op == "linear" and len(term_node.inputs) >= 2:
            linear_nodes.append(term_node)
            continue
        if term_value_id in graph.constants:
            bias_value_ids.append(term_value_id)
            continue
        return None

    if len(linear_nodes) != 2 or len(bias_value_ids) not in {1, 2}:
        return None

    first_linear, second_linear = linear_nodes
    return {
        "x_value_id": first_linear.inputs[0],
        "weight_ih_value_id": first_linear.inputs[1],
        "h_prev_value_id": second_linear.inputs[0],
        "weight_hh_value_id": second_linear.inputs[1],
        "bias_value_ids": tuple(bias_value_ids),
        "node_ids": (*node_ids, first_linear.id, second_linear.id),
    }


def _flatten_add_terms(graph: IRGraph, value_id: str) -> tuple[list[str], tuple[str, ...]]:
    node = producer(graph, strip_passthrough(graph, value_id))
    if node is None or node.op != "add" or len(node.inputs) != 2:
        return [value_id], ()
    left_terms, left_nodes = _flatten_add_terms(graph, node.inputs[0])
    right_terms, right_nodes = _flatten_add_terms(graph, node.inputs[1])
    return [*left_terms, *right_terms], (*left_nodes, *right_nodes, node.id)
