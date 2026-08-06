from __future__ import annotations

from typing import Any

import numpy as np

from . import constants
from . import models
from .errors import UnsupportedLoweringError
from .lowering_utils import *
from ..IR import models as IRModels

def lower_norm(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.rms_norm":
        if len(inputs) not in {1, 2}:
            raise unsupported_arity(node, len(inputs), "activation plus optional weight")

        weight = inputs[1] if len(inputs) == 2 else None
        return lower_rms_norm(context, node, inputs[0], weight)

    if target in {"cactus.layernorm", "cactus.layer_norm", "aten.native_layer_norm.default", "aten.layer_norm.default"}:
        require_len(node, inputs, 2)
        bias = inputs[2] if len(inputs) > 2 else None
        return context.graph.layernorm(inputs[0], inputs[1], bias=bias, eps=epsilon_attr(node))

    if target in {"cactus.groupnorm", "cactus.group_norm", "aten.native_group_norm.default", "aten.group_norm.default"}:
        require_len(node, inputs, 3)
        return context.graph.groupnorm(inputs[0], inputs[1], inputs[2], node.attrs.get("num_groups", 1), eps=epsilon_attr(node))

    if target in {"cactus.batchnorm", "cactus.batch_norm", "aten.native_batch_norm.default", "aten.batch_norm.default"}:
        require_len(node, inputs, 5)
        return context.graph.batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], axis=int(node.attrs.get("axis", 1)), eps=epsilon_attr(node))

    raise UnsupportedLoweringError(f"{node.name}: unsupported norm target {target}")

def lower_rms_norm(context: models.GenerationContext, node: IRModels.Node, x: Any, weight: Any | None = None) -> Any:
    x = cast_to_precision(context, x, context.graph.FP16)
    declared_input_shape = meta_shape(node.parents[0]) if node.parents else meta_shape(node)
    input_shape = concrete_shape(declared_input_shape)

    if input_shape is None:
        actual_shape = tuple(getattr(x, "shape", ()))

        if actual_shape and concrete_dim(declared_input_shape[-1] if declared_input_shape else None) is not None:
            input_shape = actual_shape

    if input_shape is None:
        raise UnsupportedLoweringError(f"{node.name}: cactus.rms_norm requires concrete input shape")

    if weight is None:
        weight = rms_norm_unit_weight(context, int(input_shape[-1]))
    else:
        weight = cast_to_precision(context, weight, context.graph.FP16)

    if len(input_shape) == 2:
        output = context.graph.rms_norm(x, weight, eps=epsilon_attr(node))
        return apply_inverse_weight_scale_for_parent(context, node, output, 1)

    if len(input_shape) == 1:
        output = context.graph.reshape(
            context.graph.rms_norm(context.graph.reshape(x, (1, input_shape[0])), weight, eps=epsilon_attr(node)),
            input_shape,
        )
        return apply_inverse_weight_scale_for_parent(context, node, output, 1)

    rows = element_count(input_shape[:-1])

    if rows is None:
        raise UnsupportedLoweringError(f"{node.name}: cactus.rms_norm requires concrete leading dimensions")

    normalized = context.graph.rms_norm(context.graph.reshape(x, (rows, input_shape[-1])), weight, eps=epsilon_attr(node))
    output = context.graph.reshape(normalized, input_shape)
    return apply_inverse_weight_scale_for_parent(context, node, output, 1)

def rms_norm_unit_weight(context: models.GenerationContext, hidden_dim: int) -> Any:
    key = f"__embedded_rms_norm_unit_weight_{hidden_dim}"
    existing = context.lookup(key)

    if existing is not None:
        return existing

    weight = context.graph.input((hidden_dim,), dtype=context.graph.FP16)
    context.graph.set_input(weight, np.ones((hidden_dim,), dtype=np.float16))
    context.graph.mark_embedded_input(weight)
    context.values[key] = weight
    return weight

def lower_conv(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target.startswith(constants.CACTUS_TARGET_PREFIX):
        method = target.removeprefix(constants.CACTUS_TARGET_PREFIX)
        return lower_named_conv(context, node, method, inputs)

    require_len(node, inputs, 2)
    bias = inputs[2] if len(inputs) > 2 else None
    rank = output_rank(node)

    if target == "aten.conv1d.default" or rank == 3:
        return lower_generic_conv1d(context, node, inputs, bias)

    if target == "aten.conv2d.default" or rank == 4:
        specialized = lower_specialized_conv2d(context, node, inputs, bias)

        if specialized is not None:
            return specialized

        return context.graph.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=first_int(node.attrs.get("stride"), 1),
            padding=first_int(node.attrs.get("padding"), 0),
            dilation=first_int(node.attrs.get("dilation"), 1),
            groups=int(node.attrs.get("groups", 1)),
        )

    raise UnsupportedLoweringError(f"{node.name}: cannot infer conv rank for {target}")

def lower_named_conv(context: models.GenerationContext, node: IRModels.Node, method: str, inputs: tuple[Any, ...]) -> Any:
    require_len(node, inputs, 2)
    bias = inputs[2] if len(inputs) > 2 else None

    if method == "conv1d":
        return lower_generic_conv1d(context, node, inputs, bias)

    if method == "conv1d_causal":
        x = inputs[0]

        if node.attrs.get("layout") == "batch_hidden_sequence":
            return context.graph.conv1d_causal_channel_first(
                x,
                inputs[1],
                kernel_size=int(node.attrs.get("kernel_size", 0)),
                dilation=first_int(node.attrs.get("dilation"), 1),
            )

        output = context.graph.conv1d_causal(x, inputs[1], kernel_size=int(node.attrs.get("kernel_size", 0)), dilation=first_int(node.attrs.get("dilation"), 1))
        return output

    if method == "conv1d_k3":
        require_plain_conv1d_attrs(node, method)
        return context.graph.conv1d_k3(inputs[0], inputs[1], stride=first_int(node.attrs.get("stride"), 1))

    if method == "conv1d_k7s3":
        require_len(node, inputs, 3)
        return context.graph.conv1d_k7s3(inputs[0], inputs[1], inputs[2])

    if method in {"conv1d_same_depthwise_k9", "conv1d_pointwise", "conv2d_k3s2p1", "conv2d_depthwise_k3s2p1", "conv2d_pointwise_1x1", "conv2d_k3s1p1"}:
        return getattr(context.graph, method)(inputs[0], inputs[1], bias=bias)

    if method == "conv2d":
        specialized = lower_specialized_conv2d(context, node, inputs, bias)

        if specialized is not None:
            return specialized

        return context.graph.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=first_int(node.attrs.get("stride"), 1),
            padding=first_int(node.attrs.get("padding"), 0),
            dilation=first_int(node.attrs.get("dilation"), 1),
            groups=int(node.attrs.get("groups", 1)),
        )

    raise UnsupportedLoweringError(f"{node.name}: unsupported Cactus conv method {method}")

def lower_generic_conv1d(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...], bias: Any | None) -> Any:
    stride = first_int(node.attrs.get("stride"), 1)
    padding = first_int(node.attrs.get("padding"), 0)
    dilation = first_int(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)
    input_shape = meta_shape(node.parents[0]) if node.parents else ()
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()

    if (
        padding == 0
        and dilation == 1
        and len(input_shape) == 3
        and len(weight_shape) == 3
        and groups == input_shape[1]
        and weight_shape[0] == input_shape[1]
            and weight_shape[1] == 1
    ):
        return context.graph.conv1d_depthwise(inputs[0], inputs[1], bias=bias, stride=stride)

    if dilation != 1 or groups != 1:
        require_plain_conv1d_attrs(node, "conv1d")

    x = inputs[0]

    if padding > 0:
        x = context.graph.pad(x, (padding, padding), 0.0)

    return context.graph.conv1d(x, inputs[1], bias=bias, stride=stride)

def lower_specialized_conv2d(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...], bias: Any | None) -> Any | None:
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()
    stride = tuple_int_values(node.attrs.get("stride"), 1)
    padding = tuple_int_values(node.attrs.get("padding"), 0)
    dilation = tuple_int_values(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)

    if len(weight_shape) != 4 or dilation != (1, 1):
        return None

    kernel = tuple(int(dim) for dim in weight_shape[2:])

    if kernel == (3, 3) and stride == (2, 2) and padding == (1, 1):
        if groups != 1 and weight_shape[1] == 1:
            return context.graph.conv2d_depthwise_k3s2p1(inputs[0], inputs[1], bias=bias)

        return context.graph.conv2d_k3s2p1(inputs[0], inputs[1], bias=bias)

    if kernel == (3, 3) and stride == (1, 1) and padding == (1, 1) and groups == 1:
        return context.graph.conv2d_k3s1p1(inputs[0], inputs[1], bias=bias)

    if kernel == (1, 1) and stride == (1, 1) and padding == (0, 0) and groups == 1:
        return context.graph.conv2d_pointwise_1x1(inputs[0], inputs[1], bias=bias)

    return None

def require_plain_conv1d_attrs(node: IRModels.Node, method: str) -> None:
    padding = first_int(node.attrs.get("padding"), 0)
    dilation = first_int(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)

    if padding != 0 or dilation != 1 or groups != 1:
        raise UnsupportedLoweringError(
            f"{node.name}: {method} lowering only supports padding=0, dilation=1, groups=1; "
            f"got padding={padding}, dilation={dilation}, groups={groups}"
        )
