from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.canonicalize.utils import normalize_dtype_name
from cactus.transpile.canonicalize.utils import remove_node
from cactus.transpile.canonicalize.utils import rebuild_graph
from cactus.transpile.fusion import match_attention
from cactus.transpile.fusion import match_attention_block
from cactus.transpile.fusion import match_conv_module
from cactus.transpile.fusion import match_gated_deltanet
from cactus.transpile.fusion import match_gated_mlp
from cactus.transpile.fusion import match_lstm_cell
from cactus.transpile.fusion import match_rel_pos_bias
from cactus.transpile.fusion import match_rms_norm
from cactus.transpile.fusion import match_rope
from cactus.transpile.fusion import match_self_attention_block
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.fusion.linear import match_linear
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.model_patterns import GOLD_PATTERNS
from cactus.transpile.normalize import dtype_to_ir


@dataclass(frozen=True)
class DetectedPattern:
    name: str
    anchor_node_id: str
    node_ids: tuple[str, ...]
    value_ids: tuple[str, ...]
    details: dict[str, object]


@dataclass(frozen=True)
class FusionConfig:
    enable_gated_deltanet: bool = True
    enable_lstm_cell: bool = True
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_rel_pos_bias: bool = True
    enable_attention: bool = True
    enable_self_attention_block: bool = True
    enable_attention_block: bool = True
    enable_conv_module: bool = True
    enable_add_clipped: bool = True
    enable_dense_mlp_tq_fused: bool = True


def optimize_graph(graph: IRGraph, *, max_passes: int = 8, config: FusionConfig | None = None) -> IRGraph:
    config = config or FusionConfig()
    verify_ir(graph)
    canonicalize_exported_graph(graph)
    enable_attention_block_fusions = not (
        _is_gemma4_graph(graph) or _is_whisper_seq2seq_decoder_graph(graph)
    )

    for _ in range(max_passes):
        changed = False
        if config.enable_gated_deltanet and fuse_gated_deltanet(graph):
            changed = True
        if config.enable_lstm_cell and fuse_lstm_cells(graph):
            changed = True
        if config.enable_rms_norm:
            if fuse_rms_norm(graph):
                changed = True
            if fuse_rms_norm_scale_multiply(graph):
                changed = True
        if config.enable_rope and fuse_rope(graph):
            changed = True
        if config.enable_rel_pos_bias and fuse_rel_pos_bias(graph):
            changed = True
        if config.enable_attention and fuse_attention(graph):
            changed = True
        if normalize_attention_layouts(graph):
            changed = True
        if (
            enable_attention_block_fusions
            and config.enable_self_attention_block
            and fuse_self_attention_blocks(graph)
        ):
            changed = True
        if (
            enable_attention_block_fusions
            and config.enable_attention_block
            and fuse_attention_blocks(graph)
        ):
            changed = True
        if normalize_gemma4_decoder_attention_semantics(graph):
            changed = True
        if config.enable_dense_mlp_tq_fused and fuse_dense_mlp_tq(graph):
            changed = True
        if config.enable_conv_module and fuse_conv_modules(graph):
            changed = True
        if config.enable_add_clipped and fuse_add_clipped(graph):
            changed = True
        if not changed:
            break
        canonicalize_exported_graph(graph)

    annotate_gold_patterns(graph)
    _prune_unused_inputs(graph)
    verify_ir(graph)
    return graph


def fuse_gated_deltanet(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_gated_deltanet(graph, node)
        if match is None:
            continue

        inputs = [
            match.normalized_input_value_id,
            match.qkv_weight_value_id,
            match.a_weight_value_id,
            match.b_weight_value_id,
            match.norm_weight_value_id,
        ]
        if match.z_weight_value_id is not None:
            inputs.append(match.z_weight_value_id)
        if match.dt_bias_value_id is not None:
            inputs.append(match.dt_bias_value_id)
        if match.a_log_value_id is not None:
            inputs.append(match.a_log_value_id)
        if match.conv_weight_value_id is not None:
            inputs.append(match.conv_weight_value_id)

        node.op = f"gated_deltanet_{match.mode}"
        node.inputs = inputs
        node.attrs = {
            "num_k_heads": int(match.num_k_heads),
            "num_v_heads": int(match.num_v_heads),
            "key_dim": int(match.key_dim),
            "value_dim": int(match.value_dim),
            "eps": float(match.eps),
            "chunk_size": int(match.chunk_size),
            "has_z": bool(match.z_weight_value_id is not None),
            "has_dt_bias": bool(match.dt_bias_value_id is not None),
            "has_a_log": bool(match.a_log_value_id is not None),
            "has_conv": bool(match.conv_weight_value_id is not None),
        }
        node.kind = "semantic"
        node.meta["gated_deltanet_mode"] = match.mode
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_lstm_cells(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op != "multiply":
            continue

        match = match_lstm_cell(graph, node)
        if match is None:
            continue

        bias_hh_value_id = match.bias_hh_value_id
        if bias_hh_value_id is None:
            bias_hh_value_id = _materialize_zero_like_constant(
                graph,
                match.bias_ih_value_id,
                suffix="lstm_bias_hh_zero",
            )

        node.op = "lstm_cell"
        node.inputs = [
            match.x_value_id,
            match.h_prev_value_id,
            match.c_prev_value_id,
            match.weight_ih_value_id,
            match.weight_hh_value_id,
            match.bias_ih_value_id,
            bias_hh_value_id,
        ]
        node.outputs = [match.h_output_value_id, match.c_output_value_id]
        node.attrs = {}
        node.kind = "semantic"
        node.meta["lstm_cell_nodes"] = match.node_ids
        node.meta["lstm_cell_bias_hh_zero"] = bool(match.bias_hh_value_id is None)

        for other_node_id in match.node_ids:
            if other_node_id == node.id:
                continue
            graph.nodes.pop(other_node_id, None)
            if other_node_id in graph.order:
                graph.order.remove(other_node_id)
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def annotate_gold_patterns(graph: IRGraph) -> list[DetectedPattern]:
    _clear_gold_pattern_annotations(graph)
    patterns = [
        *_detect_gated_mlps(graph),
        *_detect_decoder_attentions(graph),
        *_detect_transformer_blocks(graph),
    ]

    for pattern in patterns:
        for node_id in pattern.node_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            node.meta.setdefault("gold_patterns", [])
            node.meta["gold_patterns"].append(pattern.name)

        anchor = graph.nodes.get(pattern.anchor_node_id)
        if anchor is not None:
            anchor.meta["gold_pattern_anchor"] = True
            anchor.meta["gold_pattern_details"] = pattern.details

    graph.meta["gold_patterns_catalog"] = tuple(pattern.name for pattern in GOLD_PATTERNS)
    graph.meta["detected_gold_patterns"] = patterns
    return patterns


def summarize_detected_gold_patterns(graph: IRGraph) -> dict[str, int]:
    summary: dict[str, int] = {}
    for pattern in annotate_gold_patterns(graph):
        summary[pattern.name] = summary.get(pattern.name, 0) + 1
    return summary


def fuse_rms_norm(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"multiply", "type_as", "precision_cast"}:
            continue

        match = match_rms_norm(graph, node)
        if match is None:
            continue
        input_value = graph.values.get(match.input_value_id)
        if input_value is not None and normalize_dtype_name(input_value.dtype) == "fp32":
            # Keep FP32 normalization paths unfused so they can execute with FP32
            # reductions and elementwise math instead of the FP16-only RMS kernel.
            continue

        weight_value_id = match.weight_value_id
        if weight_value_id is None:
            if input_value is None or input_value.shape is None or not input_value.shape:
                continue
            hidden_dim = input_value.shape[-1]
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                continue
            weight_value_id = _materialize_ones_constant(
                graph,
                hidden_dim,
                dtype=input_value.dtype,
                suffix="rms_norm_ones",
            )
        if float(match.weight_offset) != 0.0:
            weight_value_id = _materialize_shifted_constant(
                graph,
                weight_value_id,
                float(match.weight_offset),
                suffix="rms_norm_scale",
            )

        node.op = "rms_norm"
        node.inputs = [match.input_value_id, weight_value_id]
        node.attrs = {"eps": float(match.eps)}
        node.kind = "semantic"
        node.meta["rms_weight_offset"] = float(match.weight_offset)
        node.meta["rms_input_value_id"] = match.input_value_id
        node.meta["rms_weight_value_id"] = weight_value_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_rms_norm_scale_multiply(graph: IRGraph) -> bool:
    if not _is_gemma4_graph(graph):
        return False

    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op != "multiply" or len(node.inputs) != 2 or len(node.outputs) != 1:
            continue

        rms_node: IRNode | None = None
        rms_output_id: str | None = None
        scale_value_id: str | None = None
        for input_id, other_id in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            candidate = producer(graph, input_id)
            if candidate is None or candidate.op != "rms_norm" or len(candidate.inputs) != 2:
                continue
            if not _is_materialized_ones_constant(graph, candidate.inputs[1]):
                continue
            rms_node = candidate
            rms_output_id = input_id
            scale_value_id = other_id
            break

        if rms_node is None or rms_output_id is None or scale_value_id is None:
            continue
        if rms_node.outputs != [rms_output_id]:
            continue
        if _has_other_users(graph, rms_output_id, excluding=node.id) or rms_output_id in graph.outputs:
            continue

        node.op = "rms_norm"
        node.inputs = [rms_node.inputs[0], scale_value_id]
        node.attrs = dict(rms_node.attrs)
        node.kind = "semantic"
        node.meta["rms_folded_scale_multiply"] = True
        node.meta["rms_input_value_id"] = rms_node.inputs[0]
        node.meta["rms_weight_value_id"] = scale_value_id
        remove_node(graph, rms_node.id)
        graph.values.pop(rms_output_id, None)
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_rope(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or not node.outputs:
            continue

        match = match_rope(graph, node.outputs[0])
        if match is None or match.partial:
            continue

        node.op = "rope"
        node.inputs = [match.input_value_id]
        node.attrs = {
            "theta": float(match.theta),
            "position_offset": int(match.position_offset),
        }
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_attention(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"scaled_dot_product_attention", "attention"}:
            continue

        match = match_attention(graph, node)
        if match is None:
            continue

        window_size = int(node.attrs.get("window_size", 0))
        if len(node.inputs) > 3:
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                node.meta["mask_window_size_hint"] = int(mask_info["window_size"])
                if window_size == 0:
                    window_size = int(mask_info["window_size"])
                    node.meta["window_size_source"] = "mask_pattern"

        if window_size != 0 and "window_size_source" not in node.meta:
            node.meta["window_size_source"] = "import_attr"

        semantic_attrs = {
            key: value
            for key, value in node.attrs.items()
            if key not in {"mask", "dropout_p", "scale", "is_causal"}
        }
        semantic_attrs.update(
            {
                "scale": float(node.attrs.get("scale", 0.0)),
                "is_causal": bool(node.attrs.get("is_causal", True)),
                "window_size": window_size,
            }
        )

        if (
            node.op == "attention"
            and node.inputs[:3] == [match.query_value_id, match.key_value_id, match.value_value_id]
            and node.attrs == semantic_attrs
        ):
            continue

        node.op = "attention"
        node.inputs = [match.query_value_id, match.key_value_id, match.value_value_id]
        node.attrs = semantic_attrs
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_rel_pos_bias(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_rel_pos_bias(graph, node)
        if match is None:
            continue

        node.op = "rel_pos_bias"
        node.inputs = [match.query_value_id, match.relative_key_value_id]
        node.attrs = {"scale": float(match.scale)}
        node.kind = "semantic"
        node.meta["rel_pos_bias_nodes"] = match.node_ids
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_attention_blocks(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_attention_block(graph, node)
        if match is None:
            continue

        qkv_inputs = [match.query_value_id, match.key_value_id, match.value_value_id]
        qkv_layout = "bhsd"
        native_qkv_inputs = [
            _strip_bhsd_to_bthd_permute_input(graph, value_id)
            for value_id in qkv_inputs
        ]
        if all(value_id is not None for value_id in native_qkv_inputs):
            qkv_inputs = [str(value_id) for value_id in native_qkv_inputs]
            qkv_layout = "bthd"

        inputs = qkv_inputs
        if match.mask_value_id is not None:
            inputs.append(match.mask_value_id)
        if match.gate_value_id is not None:
            inputs.append(match.gate_value_id)
        inputs.append(match.output_projection_weight_value_id)
        if match.output_projection_bias_value_id is not None:
            inputs.append(match.output_projection_bias_value_id)

        node.op = "attention_block"
        node.inputs = inputs
        node.attrs = {
            "scale": float(match.scale),
            "is_causal": bool(match.is_causal),
            "window_size": int(match.window_size),
            "has_mask": bool(match.mask_value_id is not None),
            "additive_mask": bool(match.additive_mask),
            "has_gate": bool(match.gate_value_id is not None),
            "has_bias": bool(match.output_projection_bias_value_id is not None),
            "attention_output_shape": tuple(int(dim) for dim in match.attention_output_shape),
            "qkv_layout": qkv_layout,
        }
        node.kind = "semantic"
        node.meta["attention_block_source"] = match.attention_node_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _strip_bhsd_to_bthd_permute_input(graph: IRGraph, value_id: str) -> str | None:
    value = graph.values.get(value_id)
    if value is None or value.producer is None:
        return None
    node = graph.nodes.get(value.producer)
    if node is None or node.op != "permute" or len(node.inputs) != 1:
        return None
    permutation = tuple(int(dim) for dim in node.attrs.get("permutation", ()))
    if permutation != (0, 2, 1, 3):
        return None
    source_id = strip_passthrough(graph, node.inputs[0])
    source_value = graph.values.get(source_id)
    if source_value is None or source_value.shape is None or value.shape is None:
        return None
    source_shape = tuple(int(dim) for dim in source_value.shape)
    output_shape = tuple(int(dim) for dim in value.shape)
    if len(source_shape) != 4 or len(output_shape) != 4:
        return None
    if (
        output_shape[0] == source_shape[0]
        and output_shape[1] == source_shape[2]
        and output_shape[2] == source_shape[1]
        and output_shape[3] == source_shape[3]
    ):
        return source_id
    return None


def normalize_attention_layouts(graph: IRGraph) -> bool:
    """Erase PyTorch BHSD<->BTHD layout round-trips around attention ops."""

    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"attention", "scaled_dot_product_attention"}:
            continue
        if len(node.inputs) < 3 or len(node.outputs) != 1:
            continue

        node_changed = False
        layouts = ["bhsd", "bhsd", "bhsd"]
        for index, value_id in enumerate(node.inputs[:3]):
            native_value_id = _strip_bhsd_to_bthd_permute_input(graph, value_id)
            if native_value_id is None:
                continue
            node.inputs[index] = str(native_value_id)
            layouts[index] = "bthd"
            node_changed = True
        if node_changed:
            node.attrs["q_layout"] = layouts[0]
            node.attrs["k_layout"] = layouts[1]
            node.attrs["v_layout"] = layouts[2]
            changed = True

        output_id = node.outputs[0]
        output_value = graph.values.get(output_id)
        output_users = list(output_value.users) if output_value is not None else []
        if len(output_users) != 1:
            continue
        output_user = graph.nodes.get(output_users[0])
        if output_user is None or output_user.op != "permute" or len(output_user.inputs) != 1 or len(output_user.outputs) != 1:
            continue
        permutation = tuple(int(dim) for dim in output_user.attrs.get("permutation", ()))
        if permutation != (0, 2, 1, 3):
            continue

        replacement_id = output_user.outputs[0]
        replacement_value = graph.values.get(replacement_id)
        if replacement_value is not None:
            old_value = graph.values.get(output_id)
            if old_value is not None:
                old_value.shape = replacement_value.shape
                old_value.dtype = replacement_value.dtype
                old_value.meta.update(replacement_value.meta)
        node.outputs = [replacement_id]
        node.attrs["output_layout"] = "bthd"
        remove_node(graph, output_user.id)
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def normalize_gemma4_decoder_attention_semantics(graph: IRGraph) -> bool:
    if not _is_gemma4_graph(graph):
        return False

    changed = False
    if _assign_gemma4_decoder_attention_hints_from_graph_meta(graph):
        changed = True
    sliding_window = _graph_sliding_window(graph)
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"attention", "scaled_dot_product_attention", "attention_block"}:
            continue
        layer_type = str(node.meta.get("attention_layer_type") or "").strip().lower()
        if layer_type == "full_attention":
            mask_input_index: int | None = None
            if node.op == "attention_block":
                if bool(node.attrs.get("has_mask", False)) and len(node.inputs) > 3:
                    mask_input_index = 3
            elif len(node.inputs) > 3:
                mask_input_index = 3

            if mask_input_index is not None:
                mask_value_id = node.inputs[mask_input_index]
                if _gemma4_can_elide_attention_mask(graph, mask_value_id):
                    del node.inputs[mask_input_index]
                    node.meta["gemma4_full_mask_elided"] = True
                    changed = True

            if node.op == "attention_block" and bool(node.attrs.get("has_mask", False)):
                node.attrs["has_mask"] = False
                changed = True
            if not bool(node.attrs.get("is_causal", False)):
                node.attrs["is_causal"] = True
                changed = True
            if "additive_mask" in node.attrs and not bool(node.attrs.get("additive_mask", False)):
                node.attrs.pop("additive_mask", None)
                changed = True
            if int(node.attrs.get("window_size", 0)) == 0:
                seq_len = 0
                attention_output_shape = node.attrs.get("attention_output_shape")
                if isinstance(attention_output_shape, (list, tuple)) and len(attention_output_shape) >= 2:
                    try:
                        seq_len = int(attention_output_shape[1])
                    except (TypeError, ValueError):
                        seq_len = 0
                if seq_len <= 0 and node.inputs:
                    query_value = graph.values.get(node.inputs[0])
                    if query_value is not None and query_value.shape is not None and len(query_value.shape) >= 3:
                        seq_len = int(query_value.shape[2])
                if seq_len > 0:
                    # Gemma4 full-attention layers are semantically unwindowed, but using a
                    # window_size >= seq_len avoids the Apple Accelerate full-window fast path
                    # that has been unstable for transpiled grouped-query Gemma4 graphs.
                    node.attrs["window_size"] = seq_len
                    node.meta["gemma4_full_attention_window_compat"] = True
                    changed = True
            continue
        if layer_type != "sliding_attention":
            continue
        if bool(node.attrs.get("additive_mask", False)):
            continue

        mask_input_index: int | None = None
        if node.op == "attention_block":
            if bool(node.attrs.get("has_mask", False)) and len(node.inputs) > 3:
                mask_input_index = 3
        elif len(node.inputs) > 3:
            mask_input_index = 3

        if mask_input_index is not None:
            mask_value_id = node.inputs[mask_input_index]
            if _gemma4_can_elide_attention_mask(graph, mask_value_id):
                mask_info = _extract_sliding_window_mask(graph, mask_value_id)
                mask_input_name = _graph_input_name(graph, mask_value_id)
                del node.inputs[mask_input_index]
                node.meta["gemma4_sliding_mask_elided"] = True
                if mask_info is not None:
                    node.meta["mask_window_size_hint"] = int(mask_info["window_size"])
                    if int(node.attrs.get("window_size", 0)) == 0 and int(mask_info["window_size"]) > 0:
                        node.attrs["window_size"] = int(mask_info["window_size"])
                        node.meta.setdefault("window_size_source", "mask_pattern")
                elif mask_input_name is not None:
                    node.meta["gemma4_elided_mask_input_name"] = mask_input_name
                changed = True

        if node.op == "attention_block" and bool(node.attrs.get("has_mask", False)):
            node.attrs["has_mask"] = False
            changed = True
        if not bool(node.attrs.get("is_causal", False)):
            node.attrs["is_causal"] = True
            changed = True
        if (
            int(node.attrs.get("window_size", 0)) == 0
            and sliding_window is not None
            and sliding_window > 0
        ):
            node.attrs["window_size"] = int(sliding_window)
            node.meta.setdefault("window_size_source", "layer_type_config")
            changed = True
        if "additive_mask" in node.attrs and not bool(node.attrs.get("additive_mask", False)):
            node.attrs.pop("additive_mask", None)
            changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _assign_gemma4_decoder_attention_hints_from_graph_meta(graph: IRGraph) -> bool:
    component = str(graph.meta.get("component", "") or "").strip().lower()
    if component != "decoder":
        return False

    layer_types = _graph_layer_types(graph)
    if not layer_types:
        return False

    attention_nodes: list[IRNode] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"attention", "scaled_dot_product_attention", "attention_block"}:
            continue
        attention_nodes.append(node)

    if len(attention_nodes) != len(layer_types):
        return False

    changed = False
    for layer_index, (node, layer_type) in enumerate(zip(attention_nodes, layer_types, strict=True)):
        if "attention_layer_type" not in node.meta:
            node.meta["attention_layer_type"] = str(layer_type)
            changed = True
        if "attention_layer_index" not in node.meta:
            node.meta["attention_layer_index"] = int(layer_index)
            changed = True
    return changed


def _gemma4_can_elide_attention_mask(
    graph: IRGraph,
    value_id: str,
    *,
    _visited: set[str] | None = None,
) -> bool:
    mask_value = graph.values.get(value_id)
    if mask_value is None:
        return False
    if mask_value.dtype in (None, "bool"):
        return True

    input_name = _graph_input_name(graph, value_id)
    if isinstance(input_name, str) and "attention_mask" in input_name:
        return True

    if _visited is None:
        _visited = set()
    if value_id in _visited:
        return False
    _visited.add(value_id)

    node = producer(graph, strip_passthrough(graph, value_id))
    if node is None:
        return False
    if node.op in {
        "logical_and",
        "logical_or",
        "scalar_equal",
        "scalar_not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "aten.__and__.Tensor",
    }:
        return True
    if node.op in {"precision_cast", "view", "expand", "permute", "slice", "index", "where"}:
        return any(_gemma4_can_elide_attention_mask(graph, input_id, _visited=_visited) for input_id in node.inputs)
    return False


def fuse_self_attention_blocks(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_self_attention_block(graph, node)
        if match is None:
            continue

        inputs = [match.hidden_value_id, match.query_weight_value_id]
        if match.query_projection_bias_value_id is not None:
            inputs.append(match.query_projection_bias_value_id)
        if match.query_add_value_id is not None:
            inputs.append(match.query_add_value_id)
        if match.rel_query_add_value_id is not None:
            inputs.append(match.rel_query_add_value_id)
        inputs.append(match.key_weight_value_id)
        if match.key_projection_bias_value_id is not None:
            inputs.append(match.key_projection_bias_value_id)
        inputs.append(match.value_weight_value_id)
        if match.value_projection_bias_value_id is not None:
            inputs.append(match.value_projection_bias_value_id)
        if match.mask_value_id is not None:
            inputs.append(match.mask_value_id)
        if match.relative_key_input_value_id is not None and match.relative_key_weight_value_id is not None:
            inputs.append(match.relative_key_input_value_id)
            inputs.append(match.relative_key_weight_value_id)
            if match.relative_key_projection_bias_value_id is not None:
                inputs.append(match.relative_key_projection_bias_value_id)
        if match.gate_value_id is not None:
            inputs.append(match.gate_value_id)
        inputs.append(match.output_projection_weight_value_id)
        if match.output_projection_bias_value_id is not None:
            inputs.append(match.output_projection_bias_value_id)

        node.op = "self_attention_block"
        node.inputs = inputs
        node.attrs = {
            "scale": float(match.scale),
            "is_causal": bool(match.is_causal),
            "window_size": int(match.window_size),
            "has_mask": bool(match.mask_value_id is not None),
            "additive_mask": bool(match.additive_mask),
            "has_gate": bool(match.gate_value_id is not None),
            "has_bias": bool(match.output_projection_bias_value_id is not None),
            "has_query_projection_bias": bool(match.query_projection_bias_value_id is not None),
            "has_query_add": bool(match.query_add_value_id is not None),
            "has_rel_query_add": bool(match.rel_query_add_value_id is not None),
            "has_key_projection_bias": bool(match.key_projection_bias_value_id is not None),
            "has_value_projection_bias": bool(match.value_projection_bias_value_id is not None),
            "has_rel_pos_bias": bool(match.relative_key_input_value_id is not None and match.relative_key_weight_value_id is not None),
            "has_relative_key_projection_bias": bool(match.relative_key_projection_bias_value_id is not None),
            "query_shape": tuple(int(dim) for dim in match.query_shape),
            "key_shape": tuple(int(dim) for dim in match.key_shape),
            "value_shape": tuple(int(dim) for dim in match.value_shape),
            "relative_key_shape": tuple(int(dim) for dim in match.relative_key_shape) if match.relative_key_shape is not None else (),
            "rel_pos_scale": float(match.rel_pos_scale) if match.rel_pos_scale is not None else 1.0,
            "attention_output_shape": tuple(int(dim) for dim in match.attention_output_shape),
        }
        node.kind = "semantic"
        node.meta["self_attention_block_source"] = match.attention_node_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_add_clipped(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op != "add" or len(node.inputs) != 2:
            continue

        lhs = strip_passthrough(graph, node.inputs[0])
        rhs = strip_passthrough(graph, node.inputs[1])
        if not (_looks_like_gemma_residual_add(graph, lhs, rhs) or _looks_like_gemma_residual_add(graph, rhs, lhs)):
            continue

        node.op = "add_clipped"
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_dense_mlp_tq(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = _match_dense_mlp_tq(graph, node)
        if match is None:
            continue

        matched_node_ids = set(match["node_ids"])
        if not _matched_nodes_are_private(graph, matched_node_ids, keep_node_id=node.id):
            continue

        node.op = "dense_mlp_tq_fused"
        node.inputs = [
            str(match["input_value_id"]),
            str(match["gate_weight_value_id"]),
            str(match["up_weight_value_id"]),
            str(match["down_weight_value_id"]),
        ]
        node.attrs = {"product_scale": float(match.get("product_scale") or 1.0)}
        node.kind = "semantic"
        node.meta["dense_mlp_tq_fused"] = True
        node.meta["dense_mlp_tq_nodes"] = tuple(sorted(matched_node_ids))
        if match.get("product_scale") is not None:
            node.meta["product_scale_from_export"] = float(match["product_scale"])

        for other_node_id in matched_node_ids - {node.id}:
            graph.nodes.pop(other_node_id, None)
            if other_node_id in graph.order:
                graph.order.remove(other_node_id)
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _match_dense_mlp_tq(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    down = match_linear(graph, node)
    if down is None or down.bias_value_id is not None:
        return None
    if not _is_cq_weight_value(graph, down.weight_value_id):
        return None

    mul_node = producer(graph, down.input_value_id)
    if mul_node is None or mul_node.op != "multiply" or len(mul_node.inputs) != 2:
        return None

    for activated_value_id, up_value_id in ((mul_node.inputs[0], mul_node.inputs[1]), (mul_node.inputs[1], mul_node.inputs[0])):
        activation_node, product_scale, scale_node_id = _unwrap_gemma4_scaled_activation(graph, activated_value_id)
        if activation_node is None or activation_node.op != "gelu" or len(activation_node.inputs) != 1:
            continue

        gate_input_producer = producer(graph, strip_passthrough(graph, activation_node.inputs[0]))
        up_producer = producer(graph, strip_passthrough(graph, up_value_id))
        if gate_input_producer is None or up_producer is None:
            continue

        gate = match_linear(graph, gate_input_producer)
        up = match_linear(graph, up_producer)
        if gate is None or up is None:
            continue
        if gate.bias_value_id is not None or up.bias_value_id is not None:
            continue
        if strip_passthrough(graph, gate.input_value_id) != strip_passthrough(graph, up.input_value_id):
            continue
        if not _is_cq_weight_value(graph, gate.weight_value_id):
            continue
        if not _is_cq_weight_value(graph, up.weight_value_id):
            continue
        if not _dense_mlp_weight_shapes_match(graph, gate.weight_value_id, up.weight_value_id, down.weight_value_id):
            continue

        node_ids = {
            *gate.node_ids,
            activation_node.id,
            *up.node_ids,
            mul_node.id,
            *down.node_ids,
        }
        if scale_node_id is not None:
            node_ids.add(scale_node_id)

        return {
            "input_value_id": strip_passthrough(graph, gate.input_value_id),
            "gate_weight_value_id": gate.weight_value_id,
            "up_weight_value_id": up.weight_value_id,
            "down_weight_value_id": down.weight_value_id,
            "product_scale": product_scale,
            "node_ids": tuple(sorted(node_ids)),
        }

    return None


def _unwrap_gemma4_scaled_activation(graph: IRGraph, value_id: str) -> tuple[IRNode | None, float | None, str | None]:
    node = producer(graph, strip_passthrough(graph, value_id))
    if node is None:
        return None, None, None
    if node.op == "scalar_multiply" and len(node.inputs) == 1:
        scale = float(node.attrs.get("value", 1.0))
        inner = producer(graph, strip_passthrough(graph, node.inputs[0]))
        return inner, scale, node.id
    return node, None, None


def _is_cq_weight_value(graph: IRGraph, value_id: str) -> bool:
    value = graph.values.get(value_id)
    if value is None:
        return False
    path = value.meta.get("path") if isinstance(value.meta, dict) else None
    if isinstance(path, str) and ".cq" in path:
        return True
    return False


def _dense_mlp_weight_shapes_match(
    graph: IRGraph,
    gate_weight_value_id: str,
    up_weight_value_id: str,
    down_weight_value_id: str,
) -> bool:
    gate = graph.values.get(gate_weight_value_id)
    up = graph.values.get(up_weight_value_id)
    down = graph.values.get(down_weight_value_id)
    if gate is None or up is None or down is None:
        return False
    if gate.shape is None or up.shape is None or down.shape is None:
        return False
    if len(gate.shape) != 2 or len(up.shape) != 2 or len(down.shape) != 2:
        return False
    return (
        int(gate.shape[0]) == int(up.shape[0])
        and int(gate.shape[1]) == int(up.shape[1])
        and int(down.shape[1]) == int(gate.shape[0])
    )


def _matched_nodes_are_private(graph: IRGraph, matched_node_ids: set[str], *, keep_node_id: str) -> bool:
    for node_id in matched_node_ids:
        if node_id == keep_node_id:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        for output_id in node.outputs:
            value = graph.values.get(output_id)
            if value is None:
                continue
            for user_id in value.users:
                if user_id not in matched_node_ids:
                    return False
    return True


def fuse_conv_modules(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_conv_module(graph, node)
        if match is None:
            continue

        inputs = [
            match.input_value_id,
            match.pointwise1_weight_value_id,
        ]
        if match.pointwise1_bias_value_id is not None:
            inputs.append(match.pointwise1_bias_value_id)
        inputs.append(match.depthwise_weight_value_id)
        if match.depthwise_bias_value_id is not None:
            inputs.append(match.depthwise_bias_value_id)
        inputs.extend(
            [
                match.batch_norm_weight_value_id,
                match.batch_norm_bias_value_id,
                match.batch_norm_running_mean_value_id,
                match.batch_norm_running_var_value_id,
                match.pointwise2_weight_value_id,
            ]
        )
        if match.pointwise2_bias_value_id is not None:
            inputs.append(match.pointwise2_bias_value_id)

        node.op = "conv_module"
        node.inputs = inputs
        node.attrs = {
            "eps": float(match.eps),
            "has_pointwise1_bias": bool(match.pointwise1_bias_value_id is not None),
            "has_depthwise_bias": bool(match.depthwise_bias_value_id is not None),
            "has_pointwise2_bias": bool(match.pointwise2_bias_value_id is not None),
            "depthwise_kernel_size": int(match.depthwise_kernel_size),
            "depthwise_padding": int(match.depthwise_padding),
        }
        node.kind = "semantic"
        node.meta["conv_module_nodes"] = match.node_ids
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _materialize_shifted_constant(graph: IRGraph, value_id: str, offset: float, *, suffix: str) -> str:
    base = graph.constants[value_id]
    if not isinstance(base, torch.Tensor):
        raise NotImplementedError(f"expected tensor constant for {value_id}, got {type(base).__name__}")

    new_value_id = f"{value_id}_{suffix}"
    if new_value_id in graph.constants:
        return new_value_id

    shifted = base.detach().cpu() + offset
    graph.constants[new_value_id] = shifted
    meta: dict[str, object] = {}
    source_value = graph.values.get(value_id)
    if _is_gemma4_graph(graph) and float(offset) == 1.0 and source_value is not None:
        path = source_value.meta.get("path")
        kind = source_value.meta.get("kind")
        source_name = source_value.meta.get("source_name")
        if isinstance(path, str) and isinstance(kind, str) and isinstance(source_name, str):
            meta = dict(source_value.meta)
            meta["materialized_from_value_id"] = value_id
            meta["materialized_by_op"] = "scalar_add"
    graph.values[new_value_id] = IRValue(
        id=new_value_id,
        shape=tuple(shifted.shape),
        dtype=dtype_to_ir(shifted.dtype),
        producer=None,
        users=[],
        meta=meta,
    )
    return new_value_id


def _materialize_zero_like_constant(graph: IRGraph, value_id: str, *, suffix: str) -> str:
    base = graph.constants[value_id]
    new_value_id = f"{value_id}_{suffix}"
    if new_value_id in graph.constants:
        return new_value_id

    if isinstance(base, torch.Tensor):
        zero = torch.zeros_like(base.detach().cpu())
    else:
        value = graph.values.get(value_id)
        if value is None or value.shape is None or value.dtype is None:
            raise NotImplementedError(
                f"expected tensor constant metadata for {value_id}, got {type(base).__name__}"
            )
        dtype_map = {
            "bool": torch.bool,
            "u8": torch.uint8,
            "i8": torch.int8,
            "i16": torch.int16,
            "i32": torch.int32,
            "i64": torch.int64,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp64": torch.float64,
        }
        torch_dtype = dtype_map.get(str(value.dtype))
        if torch_dtype is None:
            raise NotImplementedError(f"unsupported IR dtype for zero-like constant: {value.dtype}")
        zero = torch.zeros(tuple(int(dim) for dim in value.shape), dtype=torch_dtype)

    graph.constants[new_value_id] = zero
    graph.values[new_value_id] = IRValue(
        id=new_value_id,
        shape=tuple(zero.shape),
        dtype=dtype_to_ir(zero.dtype),
        producer=None,
        users=[],
    )
    return new_value_id


def _materialize_ones_constant(graph: IRGraph, size: int, *, dtype: str | None, suffix: str) -> str:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    new_value_id = f"c_{suffix}_{size}_{dtype or 'fp32'}"
    if new_value_id in graph.constants:
        return new_value_id

    value = torch.ones((size,), dtype=torch_dtype)
    graph.constants[new_value_id] = value
    graph.values[new_value_id] = IRValue(
        id=new_value_id,
        shape=(size,),
        dtype=dtype_to_ir(value.dtype),
        producer=None,
        users=[],
    )
    return new_value_id


def _is_gemma4_graph(graph: IRGraph) -> bool:
    return str(graph.meta.get("adapter_family") or graph.meta.get("family") or "").lower() == "gemma4"


def _is_whisper_seq2seq_decoder_graph(graph: IRGraph) -> bool:
    family = str(graph.meta.get("adapter_family") or graph.meta.get("family") or "").lower()
    task = str(graph.meta.get("task") or "").lower()
    component = str(graph.meta.get("component") or "").lower()
    return family == "whisper" and task == "seq2seq_transcription" and component == "decoder"


def _prune_unused_inputs(graph: IRGraph) -> bool:
    kept_inputs: list[str] = []
    changed = False
    for value_id in graph.inputs:
        value = graph.values.get(value_id)
        if value_id in graph.outputs:
            kept_inputs.append(value_id)
            continue
        if value is not None and value.users:
            kept_inputs.append(value_id)
            continue
        changed = True

    if not changed:
        return False

    graph.inputs = kept_inputs
    rebuild_graph(graph)
    return True


def _graph_input_name(graph: IRGraph, value_id: str) -> str | None:
    input_names = _coerce_string_tuple(graph.meta.get("input_names"))
    match = re.match(r"^v_args_(\d+)$", value_id)
    if match is not None:
        input_index = int(match.group(1))
        if input_index < len(input_names):
            return input_names[input_index]
    try:
        input_index = list(graph.inputs).index(value_id)
    except ValueError:
        return None
    if input_index < len(input_names):
        return input_names[input_index]
    return None


def _is_materialized_ones_constant(graph: IRGraph, value_id: str) -> bool:
    value = graph.constants.get(value_id)
    if not isinstance(value, torch.Tensor):
        return False
    if value.numel() == 0:
        return False
    return bool(torch.all(value.detach().cpu() == 1).item())


def _has_other_users(graph: IRGraph, value_id: str, *, excluding: str) -> bool:
    for other_id, other in graph.nodes.items():
        if other_id == excluding:
            continue
        if value_id in other.inputs:
            return True
    return False


def _clear_gold_pattern_annotations(graph: IRGraph) -> None:
    for node in graph.nodes.values():
        node.meta.pop("gold_patterns", None)
        node.meta.pop("gold_pattern_anchor", None)
        node.meta.pop("gold_pattern_details", None)


def _detect_gated_mlps(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_gated_mlp(graph, node)
        if match is None:
            continue

        pattern_name = "gated_mlp_gelu" if match.activation == "gelu" else "gated_mlp_silu"
        patterns.append(
            DetectedPattern(
                name=pattern_name,
                anchor_node_id=node.id,
                node_ids=match.node_ids,
                value_ids=(match.input_value_id,),
                details={
                    "activation": match.activation,
                    "input_value_id": match.input_value_id,
                    "gate_weight_value_id": match.gate_weight_value_id,
                    "up_weight_value_id": match.up_weight_value_id,
                    "down_weight_value_id": match.down_weight_value_id,
                },
            )
        )
    return patterns


def _detect_decoder_attentions(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"scaled_dot_product_attention", "attention"}:
            continue

        match = match_attention(graph, node)
        if match is None:
            continue

        q_input, k_input, v_input = match.source_input_value_ids
        pattern_name = "decoder_attention_gqa" if match.has_gqa_repeat else "decoder_attention"

        node_ids = set(match.node_ids)
        details = {
            "input_value_ids": (q_input, k_input, v_input),
            "q_linear_weight_value_id": match.weight_value_ids[0],
            "k_linear_weight_value_id": match.weight_value_ids[1],
            "v_linear_weight_value_id": match.weight_value_ids[2],
            "has_rope": bool(match.has_rope),
            "has_qk_rms_norm": bool(match.has_qk_norm),
            "has_gqa_repeat": bool(match.has_gqa_repeat),
            "is_causal": bool(match.is_causal),
            "scale": float(match.scale),
            "window_size_hint": int(match.window_size),
        }

        if len(node.inputs) > 3:
            details["mask_value_id"] = node.inputs[3]
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                details["mask_pattern"] = "sliding_window_attention_mask"
                details["window_size_hint"] = int(mask_info["window_size"])
                node_ids.update(mask_info["node_ids"])

        patterns.append(
            DetectedPattern(
                name=pattern_name,
                anchor_node_id=node.id,
                node_ids=tuple(sorted(node_ids)),
                value_ids=(q_input, k_input, v_input),
                details=details,
            )
        )
    return patterns


def _detect_transformer_blocks(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"add", "add_clipped"} or len(node.inputs) != 2:
            continue

        rhs_pattern = _find_anchor_pattern(graph, node.inputs[1], {"gated_mlp_gelu", "gated_mlp_silu"})
        lhs_pattern = _find_anchor_pattern(graph, node.inputs[0], {"gated_mlp_gelu", "gated_mlp_silu"})
        mlp_pattern = rhs_pattern or lhs_pattern
        if mlp_pattern is None:
            continue

        residual_value_id = node.inputs[0] if rhs_pattern is not None else node.inputs[1]
        attn_pattern = _find_anchor_pattern(
            graph,
            residual_value_id,
            {"decoder_attention_gqa", "decoder_attention", "gemma4_partial_rope_attention"},
        )
        if attn_pattern is None:
            continue

        node_ids = {node.id}
        for candidate in graph.nodes.values():
            gold_patterns = candidate.meta.get("gold_patterns", [])
            if attn_pattern in gold_patterns or mlp_pattern in gold_patterns:
                node_ids.add(candidate.id)

        patterns.append(
            DetectedPattern(
                name="decoder_block_post_attn_norm" if node.op == "add_clipped" else "decoder_block_simple_residual",
                anchor_node_id=node.id,
                node_ids=tuple(sorted(node_ids)),
                value_ids=(residual_value_id,),
                details={
                    "residual_value_id": residual_value_id,
                    "attention_pattern": attn_pattern,
                    "mlp_pattern": mlp_pattern,
                    "residual_op": node.op,
                },
            )
        )
    return patterns


def _looks_like_gemma_residual_add(graph: IRGraph, residual_value_id: str, branch_value_id: str) -> bool:
    branch_node = producer(graph, strip_passthrough(graph, branch_value_id))
    if branch_node is None or branch_node.op != "rms_norm":
        return False
    return strip_passthrough(graph, residual_value_id) != strip_passthrough(graph, branch_node.inputs[0])


def _find_anchor_pattern(graph: IRGraph, value_id: str, names: set[str]) -> str | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = strip_passthrough(graph, current)
        node = producer(graph, current)
        if node is None:
            return None

        for name in node.meta.get("gold_patterns", []):
            if name in names and node.meta.get("gold_pattern_anchor"):
                return name

        if len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _extract_sliding_window_mask(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    top_and = producer(graph, strip_passthrough(graph, value_id))
    if top_and is None or top_and.op not in {"aten.__and__.Tensor", "logical_and"}:
        return None
    node_ids.add(top_and.id)

    stack = list(top_and.inputs)
    saw_diff = False
    saw_cumsum = False
    window_candidates: list[int] = []

    while stack:
        current = strip_passthrough(graph, stack.pop())
        node = producer(graph, current)
        if node is None or node.id in node_ids:
            continue
        node_ids.add(node.id)

        if node.op in {"aten.diff.default", "diff"}:
            saw_diff = True
        elif node.op in {"aten.cumsum.default", "cumsum"}:
            saw_cumsum = True
        elif node.op == "scalar_subtract":
            maybe_window = int(node.attrs.get("value", 0))
            if maybe_window > 0:
                window_candidates.append(maybe_window)

        stack.extend(node.inputs)

    if not saw_diff or not saw_cumsum:
        return None

    return {
        "window_size": max((candidate for candidate in window_candidates if candidate > 1), default=0),
        "node_ids": tuple(sorted(node_ids)),
    }


def _graph_layer_types(graph: IRGraph) -> tuple[str, ...]:
    layer_types = _coerce_string_tuple(graph.meta.get("layer_types"))
    if layer_types:
        return layer_types

    providers = graph.meta.get("transpile_metadata_providers")
    if isinstance(providers, dict):
        for provider_meta in providers.values():
            if not isinstance(provider_meta, dict):
                continue
            layer_types = _coerce_string_tuple(provider_meta.get("layer_types"))
            if layer_types:
                return layer_types
    return ()


def _graph_sliding_window(graph: IRGraph) -> int | None:
    sliding_window = _coerce_optional_int(graph.meta.get("sliding_window"))
    if sliding_window is not None:
        return sliding_window

    providers = graph.meta.get("transpile_metadata_providers")
    if isinstance(providers, dict):
        for provider_meta in providers.values():
            if not isinstance(provider_meta, dict):
                continue
            sliding_window = _coerce_optional_int(provider_meta.get("sliding_window"))
            if sliding_window is not None:
                return sliding_window
    return None


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return ()
        result.append(item)
    return tuple(result)


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None
