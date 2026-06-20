from __future__ import annotations

from dataclasses import dataclass

from cactus.transpile.fusion.common import collect_node_ids
from cactus.transpile.fusion.common import producer
from cactus.transpile.fusion.common import strip_passthrough
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode


@dataclass(frozen=True)
class ConvModuleMatch:
    input_value_id: str
    pointwise1_weight_value_id: str
    pointwise1_bias_value_id: str | None
    depthwise_weight_value_id: str
    depthwise_bias_value_id: str | None
    batch_norm_weight_value_id: str
    batch_norm_bias_value_id: str
    batch_norm_running_mean_value_id: str
    batch_norm_running_var_value_id: str
    pointwise2_weight_value_id: str
    pointwise2_bias_value_id: str | None
    eps: float
    depthwise_kernel_size: int
    depthwise_padding: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class CausalLayerNormConvModuleMatch:
    input_value_id: str
    pointwise1_weight_value_id: str
    pointwise1_bias_value_id: str | None
    depthwise_weight_value_id: str
    depthwise_bias_value_id: str | None
    layer_norm_weight_value_id: str
    layer_norm_bias_value_id: str
    pointwise2_weight_value_id: str
    pointwise2_bias_value_id: str | None
    eps: float
    depthwise_kernel_size: int
    node_ids: tuple[str, ...]


def match_conv_module(graph: IRGraph, node: IRNode) -> ConvModuleMatch | None:
    if node.op != "permute" or len(node.inputs) != 1 or len(node.outputs) != 1:
        return None
    if tuple(node.attrs.get("permutation", ())) != (0, 2, 1):
        return None

    pointwise2 = producer(graph, strip_passthrough(graph, node.inputs[0]))
    if not _is_pointwise_conv1d(pointwise2):
        return None

    silu = producer(graph, strip_passthrough(graph, pointwise2.inputs[0]))
    if silu is None or silu.op != "silu" or len(silu.inputs) != 1:
        return None

    batch_norm = producer(graph, strip_passthrough(graph, silu.inputs[0]))
    if batch_norm is None or batch_norm.op != "batch_norm" or len(batch_norm.inputs) != 5:
        return None

    depthwise = producer(graph, strip_passthrough(graph, batch_norm.inputs[0]))
    if not _is_same_depthwise_conv1d(depthwise):
        return None

    glu = producer(graph, strip_passthrough(graph, depthwise.inputs[0]))
    if glu is None or glu.op != "glu" or len(glu.inputs) != 1:
        return None
    if int(glu.attrs.get("axis", -1)) not in {1, -2}:
        return None

    pointwise1 = producer(graph, strip_passthrough(graph, glu.inputs[0]))
    if not _is_pointwise_conv1d(pointwise1):
        return None

    input_permute = producer(graph, strip_passthrough(graph, pointwise1.inputs[0]))
    if input_permute is None or input_permute.op != "permute" or len(input_permute.inputs) != 1:
        return None
    if tuple(input_permute.attrs.get("permutation", ())) != (0, 2, 1):
        return None

    depthwise_weight_shape = graph.values.get(depthwise.inputs[1]).shape if depthwise.inputs[1] in graph.values else None
    if depthwise_weight_shape is None or len(depthwise_weight_shape) != 3:
        return None
    if (
        int(depthwise.attrs.get("groups", 1)) != int(depthwise_weight_shape[0])
        or int(depthwise_weight_shape[1]) != 1
    ):
        return None
    depthwise_kernel_size = int(depthwise_weight_shape[2])

    return ConvModuleMatch(
        input_value_id=strip_passthrough(graph, input_permute.inputs[0]),
        pointwise1_weight_value_id=pointwise1.inputs[1],
        pointwise1_bias_value_id=pointwise1.inputs[2] if len(pointwise1.inputs) > 2 else None,
        depthwise_weight_value_id=depthwise.inputs[1],
        depthwise_bias_value_id=depthwise.inputs[2] if len(depthwise.inputs) > 2 else None,
        batch_norm_weight_value_id=batch_norm.inputs[1],
        batch_norm_bias_value_id=batch_norm.inputs[2],
        batch_norm_running_mean_value_id=batch_norm.inputs[3],
        batch_norm_running_var_value_id=batch_norm.inputs[4],
        pointwise2_weight_value_id=pointwise2.inputs[1],
        pointwise2_bias_value_id=pointwise2.inputs[2] if len(pointwise2.inputs) > 2 else None,
        eps=float(batch_norm.attrs.get("eps", 1e-5)),
        depthwise_kernel_size=depthwise_kernel_size,
        depthwise_padding=int(depthwise.attrs.get("padding", 0)),
        node_ids=collect_node_ids(
            node,
            pointwise2,
            silu,
            batch_norm,
            depthwise,
            glu,
            pointwise1,
            input_permute,
        ),
    )


def match_causal_layer_norm_conv_module(graph: IRGraph, node: IRNode) -> CausalLayerNormConvModuleMatch | None:
    if node.op != "permute" or len(node.inputs) != 1 or len(node.outputs) != 1:
        return None
    if tuple(node.attrs.get("permutation", ())) != (0, 2, 1):
        return None

    pointwise2 = producer(graph, strip_passthrough(graph, node.inputs[0]))
    if not _is_pointwise_conv1d(pointwise2):
        return None

    silu = producer(graph, strip_passthrough(graph, pointwise2.inputs[0]))
    if silu is None or silu.op != "silu" or len(silu.inputs) != 1:
        return None

    layer_norm_input_permute = producer(graph, strip_passthrough(graph, silu.inputs[0]))
    if (
        layer_norm_input_permute is None
        or layer_norm_input_permute.op != "permute"
        or tuple(layer_norm_input_permute.attrs.get("permutation", ())) != (0, 2, 1)
        or len(layer_norm_input_permute.inputs) != 1
    ):
        return None

    layer_norm = producer(graph, strip_passthrough(graph, layer_norm_input_permute.inputs[0]))
    if layer_norm is None or layer_norm.op != "layer_norm" or len(layer_norm.inputs) != 3:
        return None

    depthwise_output_permute = producer(graph, strip_passthrough(graph, layer_norm.inputs[0]))
    if (
        depthwise_output_permute is None
        or depthwise_output_permute.op != "permute"
        or tuple(depthwise_output_permute.attrs.get("permutation", ())) != (0, 2, 1)
        or len(depthwise_output_permute.inputs) != 1
    ):
        return None

    depthwise = producer(graph, strip_passthrough(graph, depthwise_output_permute.inputs[0]))
    if not _is_causal_depthwise_conv1d(depthwise):
        return None

    pad = producer(graph, strip_passthrough(graph, depthwise.inputs[0]))
    if pad is None or pad.op != "pad" or len(pad.inputs) != 1:
        return None
    if pad.attrs.get("mode") != "constant" or float(pad.attrs.get("value", 0.0)) != 0.0:
        return None

    depthwise_weight_shape = graph.values.get(depthwise.inputs[1]).shape if depthwise.inputs[1] in graph.values else None
    if depthwise_weight_shape is None or len(depthwise_weight_shape) != 3:
        return None
    if (
        int(depthwise.attrs.get("groups", 1)) != int(depthwise_weight_shape[0])
        or int(depthwise_weight_shape[1]) != 1
    ):
        return None
    depthwise_kernel_size = int(depthwise_weight_shape[2])
    if tuple(pad.attrs.get("pads", ())) != (depthwise_kernel_size - 1, 0):
        return None

    glu = producer(graph, strip_passthrough(graph, pad.inputs[0]))
    if glu is None or glu.op != "glu" or len(glu.inputs) != 1:
        return None
    if int(glu.attrs.get("axis", -1)) not in {1, -2}:
        return None

    pointwise1 = producer(graph, strip_passthrough(graph, glu.inputs[0]))
    if not _is_pointwise_conv1d(pointwise1):
        return None

    input_permute = producer(graph, strip_passthrough(graph, pointwise1.inputs[0]))
    if input_permute is None or input_permute.op != "permute" or len(input_permute.inputs) != 1:
        return None
    if tuple(input_permute.attrs.get("permutation", ())) != (0, 2, 1):
        return None

    return CausalLayerNormConvModuleMatch(
        input_value_id=strip_passthrough(graph, input_permute.inputs[0]),
        pointwise1_weight_value_id=pointwise1.inputs[1],
        pointwise1_bias_value_id=pointwise1.inputs[2] if len(pointwise1.inputs) > 2 else None,
        depthwise_weight_value_id=depthwise.inputs[1],
        depthwise_bias_value_id=depthwise.inputs[2] if len(depthwise.inputs) > 2 else None,
        layer_norm_weight_value_id=layer_norm.inputs[1],
        layer_norm_bias_value_id=layer_norm.inputs[2],
        pointwise2_weight_value_id=pointwise2.inputs[1],
        pointwise2_bias_value_id=pointwise2.inputs[2] if len(pointwise2.inputs) > 2 else None,
        eps=float(layer_norm.attrs.get("eps", 1e-5)),
        depthwise_kernel_size=depthwise_kernel_size,
        node_ids=collect_node_ids(
            node,
            pointwise2,
            silu,
            layer_norm_input_permute,
            layer_norm,
            depthwise_output_permute,
            depthwise,
            pad,
            glu,
            pointwise1,
            input_permute,
        ),
    )


def _is_pointwise_conv1d(node: IRNode | None) -> bool:
    return (
        node is not None
        and node.op == "conv1d"
        and len(node.inputs) >= 2
        and int(node.attrs.get("stride", 1)) == 1
        and int(node.attrs.get("padding", 0)) == 0
        and int(node.attrs.get("dilation", 1)) == 1
        and int(node.attrs.get("groups", 1)) == 1
    )


def _is_same_depthwise_conv1d(node: IRNode | None) -> bool:
    if node is None or node.op != "conv1d" or len(node.inputs) < 2:
        return False
    groups = int(node.attrs.get("groups", 1))
    padding = int(node.attrs.get("padding", 0))
    stride = int(node.attrs.get("stride", 1))
    dilation = int(node.attrs.get("dilation", 1))
    if stride != 1 or dilation != 1:
        return False
    return groups > 1 and padding >= 0


def _is_causal_depthwise_conv1d(node: IRNode | None) -> bool:
    if not _is_same_depthwise_conv1d(node):
        return False
    return int(node.attrs.get("padding", 0)) == 0
