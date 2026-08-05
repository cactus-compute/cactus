from __future__ import annotations

from typing import Any

from . import models
from .errors import UnsupportedLoweringError
from .lowering_utils import *
from ..IR import models as IRModels


def lower_constant_pad_nd(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    pads = node.attrs.get("pad", node.attrs.get("arg_1"))
    value = scalar_attr(node, "value")

    if value is None:
        value = scalar_attr(node, "arg_2")

    if value is None:
        value = 0.0

    if not isinstance(pads, (list, tuple)) or len(pads) % 2 != 0:
        raise UnsupportedLoweringError(f"{node.name}: constant_pad_nd lowering requires an even pad list")

    result = inputs[0]
    parent_shape = meta_shape(node.parents[0])
    current_shape = list(parent_shape)
    rank = len(parent_shape)
    positive_pads = [0 for _ in pads]

    for pair_index in range(0, len(pads), 2):
        left_pad = int(pads[pair_index])
        right_pad = int(pads[pair_index + 1])
        positive_pads[pair_index] = max(left_pad, 0)
        positive_pads[pair_index + 1] = max(right_pad, 0)

        if left_pad >= 0 and right_pad >= 0:
            continue

        axis = rank - 1 - (pair_index // 2)

        if axis < 0:
            raise UnsupportedLoweringError(f"{node.name}: pad rank exceeds input rank")

        axis_size = current_shape[axis]

        if not isinstance(axis_size, int):
            raise UnsupportedLoweringError(f"{node.name}: cannot crop symbolic pad dimension {axis_size}")

        start = max(-left_pad, 0)
        end_crop = max(-right_pad, 0)
        length = axis_size - start - end_crop

        if length < 0:
            raise UnsupportedLoweringError(f"{node.name}: pad crop removes more than the full axis")

        result = context.graph.slice(result, axis, start, length=length)
        current_shape[axis] = length

    if any(positive_pads):
        return context.graph.pad(result, positive_pads, float(value))

    return result


def lower_constant_input(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if node.target == "aten.full_like.default":
        return lower_full_like(context, node)

    constant = deterministic_constant_values(node)

    if constant is not None:
        values, shape = constant
        dtype = cactus_precision(context.graph, tensor_dtype(node))
        tensor = context.graph.input(shape, dtype=dtype)
        path = write_constant_tensor(context, node, values, shape, dtype)
        node_id = models.tensor_node_id(tensor)

        if node_id is None:
            raise UnsupportedLoweringError(f"{node.name}: generated constant does not expose a Cactus node id")

        context.component.add_weight_binding(
            models.WeightBinding(
                placeholder=node.name,
                source_target=node.target,
                node_id=node_id,
                path=path,
                output_name=path,
                source_name=node.target,
                value_id=node.name,
                precision=precision_name(context.graph, dtype),
                component=context.component.name,
                binding_kind="generated_constant",
            )
        )
        return tensor

    context.component.warn(f"{node.name}: constant-producing op {node.target} lowered as graph input")
    shape, dynamic_dims = graph_input_shape(node)
    dtype = cactus_precision(context.graph, tensor_dtype(node))
    tensor = context.graph.input(shape, dtype=dtype, dynamic_dims=dynamic_dims if any(dynamic_dims) else None)
    context.component.add_runtime_input(tensor, node.name)
    return tensor


def lower_full_like(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    fill_value = scalar_attr(node, "fill_value")

    if fill_value is None:
        fill_value = scalar_attr(node, "arg_1")

    if fill_value is None:
        fill_value = 0.0

    output = context.graph.scalar_multiply(inputs[0], 0.0)
    fill_float = float(fill_value)

    if fill_float != 0.0:
        output = context.graph.scalar_add(output, fill_float)

    return cast_to_precision(context, output, cactus_precision(context.graph, tensor_dtype(node)))


def deterministic_constant_values(node: IRModels.Node) -> tuple[list[float], tuple[int, ...]] | None:
    shape = concrete_shape(meta_shape(node))

    if node.target.startswith("aten.arange"):
        start = numeric_attr(node, "start", "arg_0", default=0)
        end = numeric_attr(node, "end", "arg_1")
        step = numeric_attr(node, "step", "arg_2", default=1)

        if end is None or step in {None, 0}:
            return None

        values = arange_values(float(start), float(end), float(step))
        return values, (len(values),)

    if node.target == "aten.scalar_tensor.default":
        value = numeric_attr(node, "value", "arg_0", default=0)
        return [float(value or 0)], (1,)

    if node.target in {"aten.full.default", "aten.full_like.default"}:
        if shape is None:
            raw_size = node.attrs.get("size", node.attrs.get("arg_0"))
            shape = tuple(int(dim) for dim in raw_size) if isinstance(raw_size, (list, tuple)) else None

        if shape is None:
            return None

        fill_value = numeric_attr(node, "fill_value", "arg_1", default=0)
        count = math.prod(shape) if shape else 1
        return [float(fill_value or 0)] * int(count), tuple(shape or (1,))

    return None


def arange_values(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    current = start

    if step > 0:
        while current < end:
            values.append(current)
            current += step
    else:
        while current > end:
            values.append(current)
            current += step

    return values


def lower_neg(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.scalar_multiply(inputs[0], -1.0)


def lower_cumsum(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    axis = axis_attr(node)

    if axis is None:
        raise UnsupportedLoweringError(f"{node.name}: cumsum lowering missing axis/dim attr")

    return context.graph.cumsum(inputs[0], normalize_dim(axis, len(meta_shape(node.parents[0]))))


def lower_softmax(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    axis = axis_attr(node, default=-1)

    if axis is None:
        raise UnsupportedLoweringError(f"{node.name}: softmax lowering missing axis/dim attr")

    return context.graph.softmax(fp16_tensor(context, inputs[0]), axis=normalize_dim(axis, len(meta_shape(node.parents[0]))))


def lower_topk(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    k = node.attrs.get("k")

    if k is None:
        raise UnsupportedLoweringError(f"{node.name}: topk lowering missing k attr")

    return context.graph.topk(inputs[0], int(k))


def lower_gather(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)
    axis = axis_attr(node, default=0)
    return context.graph.gather(inputs[0], inputs[1], axis=axis if axis is not None else 0)


def lower_embedding(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)
    output = context.graph.embedding_from_tensor(inputs[0], inputs[1])
    return apply_inverse_weight_scale_for_parent(context, node, output, 0)


def lower_clamp(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    lo = scalar_weight_bound_value(context, node.parents[1]) if len(node.parents) > 1 else None
    hi = scalar_weight_bound_value(context, node.parents[2]) if len(node.parents) > 2 else None

    if lo is None:
        lo = inputs[1] if len(inputs) > 1 else node.attrs.get("lo", node.attrs.get("min", node.attrs.get("arg_1")))

    if hi is None:
        hi = inputs[2] if len(inputs) > 2 else node.attrs.get("hi", node.attrs.get("max", node.attrs.get("arg_2")))

    if lo is None and hi is None:
        return inputs[0]

    result = inputs[0]

    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return context.graph.clamp(result, float(lo), float(hi))

    if lo is not None:
        if isinstance(lo, (int, float)):
            result = context.graph.clamp(result, float(lo), math.inf)
        else:
            lo = cast_to_precision(context, lo, getattr(result, "dtype", context.graph.FP16))
            result = context.graph.where(context.graph.greater_equal(result, lo), result, lo)

    if hi is not None:
        if isinstance(hi, (int, float)):
            result = context.graph.clamp(result, -math.inf, float(hi))
        else:
            hi = cast_to_precision(context, hi, getattr(result, "dtype", context.graph.FP16))
            result = context.graph.where(context.graph.less_equal(result, hi), result, hi)

    return result
