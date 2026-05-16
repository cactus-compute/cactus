from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode


@dataclass(frozen=True)
class RelPosBiasMatch:
    query_value_id: str
    relative_key_value_id: str
    scale: float
    node_ids: tuple[str, ...]


def match_rel_pos_bias(graph: IRGraph, node: IRNode) -> RelPosBiasMatch | None:
    if node.op != "scalar_multiply" or len(node.inputs) != 1:
        return None

    gather_match = _match_gather_rel_pos_bias(graph, node)
    if gather_match is not None:
        return gather_match

    final_slice = producer(graph, strip_passthrough(graph, node.inputs[0]))
    if final_slice is None or final_slice.op != "slice":
        return None
    if int(final_slice.attrs.get("axis", -1)) != 3 or int(final_slice.attrs.get("start", -1)) != 0:
        return None

    reshape_after_shift = producer(graph, strip_passthrough(graph, final_slice.inputs[0]))
    if reshape_after_shift is None or reshape_after_shift.op not in {"reshape", "view"} or len(reshape_after_shift.inputs) != 1:
        return None

    shift_slice = producer(graph, strip_passthrough(graph, reshape_after_shift.inputs[0]))
    if shift_slice is None or shift_slice.op != "slice":
        return None
    if int(shift_slice.attrs.get("axis", -1)) != 2 or int(shift_slice.attrs.get("start", -1)) != 1:
        return None

    reshape_for_shift = producer(graph, strip_passthrough(graph, shift_slice.inputs[0]))
    if reshape_for_shift is None or reshape_for_shift.op not in {"reshape", "view"} or len(reshape_for_shift.inputs) != 1:
        return None

    pad = producer(graph, strip_passthrough(graph, reshape_for_shift.inputs[0]))
    if pad is None or pad.op != "pad" or len(pad.inputs) != 1:
        return None
    if tuple(int(v) for v in pad.attrs.get("pads", ())) != (1, 0):
        return None
    if str(pad.attrs.get("mode", "constant")) != "constant":
        return None
    if float(pad.attrs.get("value", 0.0)) != 0.0:
        return None

    matmul = producer(graph, strip_passthrough(graph, pad.inputs[0]))
    if matmul is None or matmul.op != "matmul" or len(matmul.inputs) != 2:
        return None

    rhs_permute = producer(graph, strip_passthrough(graph, matmul.inputs[1]))
    if rhs_permute is None or rhs_permute.op != "permute" or len(rhs_permute.inputs) != 1:
        return None
    if tuple(int(v) for v in rhs_permute.attrs.get("permutation", ())) != (0, 2, 3, 1):
        return None

    query_value_id = strip_passthrough(graph, matmul.inputs[0])
    relative_key_value_id = strip_passthrough(graph, rhs_permute.inputs[0])

    query_shape = graph.values.get(query_value_id).shape if query_value_id in graph.values else None
    relative_key_shape = graph.values.get(relative_key_value_id).shape if relative_key_value_id in graph.values else None
    output_shape = graph.values.get(node.outputs[0]).shape if node.outputs and node.outputs[0] in graph.values else None
    if query_shape is None or relative_key_shape is None or output_shape is None:
        return None
    if len(query_shape) != 4 or len(relative_key_shape) != 4 or len(output_shape) != 4:
        return None

    batch, heads, seq_len, head_dim = (int(v) for v in query_shape)
    rel_batch, rel_len, rel_heads, rel_dim = (int(v) for v in relative_key_shape)
    out_batch, out_heads, out_t, out_s = (int(v) for v in output_shape)
    if rel_batch not in {1, batch}:
        return None
    if rel_heads != heads or rel_dim != head_dim:
        return None
    if rel_len < 2 * seq_len - 1:
        return None
    if (out_batch, out_heads, out_t, out_s) != (batch, heads, seq_len, seq_len):
        return None

    return RelPosBiasMatch(
        query_value_id=query_value_id,
        relative_key_value_id=relative_key_value_id,
        scale=float(node.attrs.get("value", 1.0)),
        node_ids=tuple(
            sorted(
                {
                    node.id,
                    final_slice.id,
                    reshape_after_shift.id,
                    shift_slice.id,
                    reshape_for_shift.id,
                    pad.id,
                    matmul.id,
                    rhs_permute.id,
                }
            )
        ),
    )


def _match_gather_rel_pos_bias(graph: IRGraph, node: IRNode) -> RelPosBiasMatch | None:
    gather = producer(graph, strip_passthrough(graph, node.inputs[0]))
    if gather is None or gather.op != "gather" or len(gather.inputs) < 2:
        return None
    if int(gather.attrs.get("axis", -1)) not in {-1, 3}:
        return None

    matmul = producer(graph, strip_passthrough(graph, gather.inputs[0]))
    if matmul is None or matmul.op != "matmul" or len(matmul.inputs) != 2:
        return None

    query_heads = producer(graph, strip_passthrough(graph, matmul.inputs[0]))
    rhs_transpose = producer(graph, strip_passthrough(graph, matmul.inputs[1]))
    if query_heads is None or query_heads.op != "permute" or len(query_heads.inputs) != 1:
        return None
    if rhs_transpose is None or rhs_transpose.op != "permute" or len(rhs_transpose.inputs) != 1:
        return None
    if tuple(int(v) for v in query_heads.attrs.get("permutation", ())) != (0, 2, 1, 3):
        return None
    if tuple(int(v) for v in rhs_transpose.attrs.get("permutation", ())) != (0, 1, 3, 2):
        return None

    relative_key_heads = producer(graph, strip_passthrough(graph, rhs_transpose.inputs[0]))
    if relative_key_heads is None or relative_key_heads.op != "permute" or len(relative_key_heads.inputs) != 1:
        return None
    if tuple(int(v) for v in relative_key_heads.attrs.get("permutation", ())) != (0, 2, 1, 3):
        return None

    query_value_id = strip_passthrough(graph, query_heads.inputs[0])
    relative_key_value_id = strip_passthrough(graph, relative_key_heads.inputs[0])
    output_value_id = node.outputs[0] if node.outputs else ""

    query_shape = graph.values.get(query_value_id).shape if query_value_id in graph.values else None
    relative_key_shape = graph.values.get(relative_key_value_id).shape if relative_key_value_id in graph.values else None
    output_shape = graph.values.get(output_value_id).shape if output_value_id in graph.values else None
    if query_shape is None or relative_key_shape is None or output_shape is None:
        return None
    if len(query_shape) != 4 or len(relative_key_shape) != 4 or len(output_shape) != 4:
        return None

    batch, seq_len, heads, head_dim = (int(v) for v in query_shape)
    rel_batch, rel_len, rel_heads, rel_dim = (int(v) for v in relative_key_shape)
    out_batch, out_heads, out_t, out_s = (int(v) for v in output_shape)
    if rel_batch not in {1, batch}:
        return None
    if rel_heads != heads or rel_dim != head_dim:
        return None
    if rel_len < 2 * seq_len - 1:
        return None
    if (out_batch, out_heads, out_t, out_s) != (batch, heads, seq_len, seq_len):
        return None

    return RelPosBiasMatch(
        query_value_id=query_value_id,
        relative_key_value_id=relative_key_value_id,
        scale=float(node.attrs.get("value", 1.0)),
        node_ids=tuple(
            sorted(
                {
                    node.id,
                    gather.id,
                    matmul.id,
                    query_heads.id,
                    rhs_transpose.id,
                    relative_key_heads.id,
                }
            )
        ),
    )
