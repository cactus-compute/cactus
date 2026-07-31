from __future__ import annotations

from typing import Any

from . import constants
from . import models
from .errors import UnsupportedLoweringError
from .lowering_cache import lower_decode_cache_cat, lower_prefill_cache_cat
from .lowering_utils import *
from ..IR import models as IRModels


def lower_binary(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    method = constants.BINARY_TARGETS[node.target]

    if len(inputs) == 2:
        left, right = align_binary_inputs(context, node, inputs)

        if method == "add" and context.component.name in constants.GEMMA_ADD_CLIPPED_COMPONENTS and looks_like_gemma_residual_add(node):
            return context.graph.add_clipped(left, right)

        if method == "multiply" and looks_like_gemma4_mlp_product(node):
            if is_gemma4_mlp_gate_activation(node.parents[0]):
                left = context.graph.scalar_multiply(left, constants.GEMMA4_MLP_PRODUCT_SCALE)
            else:
                right = context.graph.scalar_multiply(right, constants.GEMMA4_MLP_PRODUCT_SCALE)

        return getattr(context.graph, method)(left, right)

    scalar = scalar_attr(node, "other")

    if len(inputs) == 1 and scalar is not None:
        scalar_method = binary_method_to_scalar_method(method)
        return getattr(context.graph, scalar_method)(inputs[0], scalar)

    raise unsupported_arity(node, len(inputs), "two tensor inputs or one tensor plus scalar attr")


def looks_like_gemma_residual_add(node: IRModels.Node) -> bool:
    if len(node.parents) != 2:
        return False

    shape = meta_shape(node)

    if not shape or len(shape) > 4:
        return False

    left, right = node.parents
    return (
        is_gemma_residual_branch(left, right)
        or is_gemma_residual_branch(right, left)
    )


def is_gemma_residual_branch(residual: IRModels.Node, branch: IRModels.Node) -> bool:
    branch_source = strip_precision_passthrough(branch)

    if branch_source.target != "cactus.rms_norm" or not branch_source.parents:
        return False

    return strip_precision_passthrough(residual) is not strip_precision_passthrough(branch_source.parents[0])


def looks_like_gemma4_mlp_product(node: IRModels.Node) -> bool:
    if len(node.parents) != 2:
        return False

    text = node_context_text(node)

    if "gemma" not in text.lower() or "language_model" not in text or ".mlp" not in text:
        return False

    left, right = node.parents
    return (
        is_gemma4_mlp_gate_activation(left) and has_ancestor_text(right, "mlp_up_proj")
    ) or (
        is_gemma4_mlp_gate_activation(right) and has_ancestor_text(left, "mlp_up_proj")
    )


def is_gemma4_mlp_gate_activation(node: IRModels.Node) -> bool:
    source = strip_precision_passthrough(node)

    if source.target not in {"cactus.gelu", "aten.gelu.default"}:
        return False

    return has_ancestor_text(source, "mlp_gate_proj")


def has_ancestor_text(node: IRModels.Node, pattern: str, max_depth: int = 8) -> bool:
    pattern = pattern.lower()
    worklist: list[tuple[IRModels.Node, int]] = [(node, 0)]
    seen: set[str] = set()

    while worklist:
        current, depth = worklist.pop(0)

        if current.name in seen:
            continue

        seen.add(current.name)

        if pattern in node_context_text(current).lower():
            return True

        if depth >= max_depth:
            continue

        for parent in current.parents:
            worklist.append((parent, depth + 1))

    return False


def node_context_text(node: IRModels.Node) -> str:
    return f"{node.name} {node.target} {node.module_stack!r}"


def strip_precision_passthrough(node: IRModels.Node) -> IRModels.Node:
    current = node
    passthrough_targets = constants.PASS_THROUGH_TARGETS | constants.TO_COPY_TARGETS

    while current.target in passthrough_targets and len(current.parents) == 1:
        current = current.parents[0]

    return current


def lower_scalar(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method, attr_name = constants.SCALAR_TARGETS[node.target]
    value = scalar_attr(node, attr_name)

    if value is None:
        value = scalar_attr(node, "arg_1")

    if value is None:
        if method == "scalar_equal":
            return context.graph.scalar_multiply(inputs[0], 0.0)

        if method == "scalar_not_equal":
            return context.graph.scalar_add(context.graph.scalar_multiply(inputs[0], 0.0), 1.0)

        raise UnsupportedLoweringError(f"{node.name}: scalar lowering {node.target} missing attr {attr_name}")

    if method == "scalar_multiply" and float(value) == 1.0:
        return inputs[0]

    return getattr(context.graph, method)(inputs[0], value)


def align_binary_inputs(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    target_shape = output_shape(node)
    left = align_input_to_shape(context, inputs[0], node.parents[0], target_shape)
    right = align_input_to_shape(context, inputs[1], node.parents[1], target_shape)
    return align_binary_precision(context, left, right)


def align_binary_precision(context: models.GenerationContext, left: Any, right: Any) -> tuple[Any, Any]:
    left_precision = getattr(left, "dtype", None)
    right_precision = getattr(right, "dtype", None)

    if left_precision is None or right_precision is None or left_precision == right_precision:
        return left, right

    return left, cast_to_precision(context, right, left_precision)


def align_input_to_shape(context: models.GenerationContext, value: Any, source_node: IRModels.Node, target_shape: tuple[Any, ...]) -> Any:
    value = align_value_to_declared_shape(context, value, source_node)
    source_shape = meta_shape(source_node)

    if shape_matches_tensor(source_node, target_shape):
        return value

    if element_count(source_shape) == element_count(target_shape):
        return context.graph.reshape(value, target_shape)

    return value


def align_value_to_declared_shape(context: models.GenerationContext, value: Any, source_node: IRModels.Node) -> Any:
    actual_shape = tuple(getattr(value, "shape", ()))
    declared_shape = meta_shape(source_node)

    if not actual_shape or not declared_shape or len(declared_shape) <= len(actual_shape):
        return value

    resolved_shape = resolve_declared_shape_from_actual(declared_shape, actual_shape)

    if resolved_shape is None or tuple(resolved_shape) == actual_shape:
        return value

    return context.graph.reshape(value, resolved_shape)


def resolve_declared_shape_from_actual(declared_shape: tuple[Any, ...], actual_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    actual_index = len(actual_shape) - 1
    resolved = [1 for _ in declared_shape]

    for declared_index in range(len(declared_shape) - 1, -1, -1):
        declared_dim = concrete_dim(declared_shape[declared_index])

        if actual_index >= 0 and (declared_dim is None or declared_dim == actual_shape[actual_index]):
            resolved[declared_index] = actual_shape[actual_index]
            actual_index -= 1
            continue

        if declared_dim == 1:
            resolved[declared_index] = 1
            continue

        return None

    if actual_index >= 0:
        return None

    return tuple(resolved)


def lower_unary(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.UNARY_TARGETS[node.target]
    value = fp16_tensor(context, inputs[0]) if method in constants.FP16_UNARY_METHODS else inputs[0]
    return getattr(context.graph, method)(value)


def lower_log1p(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)

    if node.parents and node.parents[0].target in {"aten.exp.default", "cactus.scalar_exp"} and node.parents[0].parents:
        x = context.require(node.parents[0].parents[0].name)
        return lower_stable_softplus(context, x)

    return context.graph.scalar_log(context.graph.scalar_add(inputs[0], 1.0))


def lower_stable_softplus(context: models.GenerationContext, x: Any) -> Any:
    x = cast_to_precision(context, x, context.graph.FP16)
    neg_abs = context.graph.scalar_multiply(context.graph.abs(x), -1.0)
    log_term = context.graph.scalar_log(context.graph.scalar_add(context.graph.scalar_exp(neg_abs), 1.0))
    positive_term = context.graph.clamp(x, 0.0, math.inf)
    return context.graph.add(log_term, positive_term)


def lower_pow(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    exponent = scalar_attr(node, "exponent")

    if exponent is None:
        raise UnsupportedLoweringError(f"{node.name}: pow lowering missing exponent attr")

    return context.graph.pow(inputs[0], exponent)


def lower_reduce(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.REDUCE_TARGETS[node.target]
    axis = axis_attr(node)

    if axis is None:
        raise UnsupportedLoweringError(f"{node.name}: reduce lowering missing axis/dim attr")

    normalized_axis = normalize_dim(axis, len(meta_shape(node.parents[0])))
    reduced = getattr(context.graph, method)(inputs[0], normalized_axis)
    target_shape = output_shape(node)
    dropped_shape = reduction_dropped_shape(meta_shape(node.parents[0]), normalized_axis)

    if bool(node.attrs.get("keepdim", False)) or (
        tuple(dropped_shape) != tuple(target_shape)
        and element_count(dropped_shape) == element_count(target_shape)
    ):
        return context.graph.reshape(reduced, target_shape)

    return reduced


def lower_shape(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.SHAPE_TARGETS[node.target]
    shape = decoder_prefill_logits_shape(context, node, shape_attr(node))

    return getattr(context.graph, method)(inputs[0], shape)


def lower_expand(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    target_shape = shape_attr(node)

    if shape_matches_tensor(node.parents[0], target_shape):
        return inputs[0]

    if element_count(meta_shape(node.parents[0])) == element_count(target_shape):
        return context.graph.reshape(inputs[0], target_shape)

    try:
        return context.graph.expand(inputs[0], target_shape)
    except AttributeError as e:
        raise UnsupportedLoweringError(
            f"{node.name}: native Cactus expand is unavailable for broadcast shape "
            f"{meta_shape(node.parents[0])} -> {target_shape}"
        ) from e


def lower_unsqueeze(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.reshape(inputs[0], output_shape(node))


def lower_flatten(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.flatten(
        inputs[0],
        start_dim=int(node.attrs.get("start_dim", 0)),
        end_dim=int(node.attrs.get("end_dim", -1)),
    )


def lower_repeat(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    repeats = node.attrs.get("repeats", node.attrs.get("arg_1"))

    if isinstance(repeats, (list, tuple)) and all(int(value) == 1 for value in repeats):
        return inputs[0]

    raise UnsupportedLoweringError(f"{node.name}: repeat lowering only supports no-op repeat factors, got {repeats}")


def lower_transpose(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)

    if can_alias_quantized_weight_transpose(context, node, inputs[0]):
        return inputs[0]

    permutation = node.attrs.get("permutation")

    if permutation is not None:
        return context.graph.permute(inputs[0], tuple_ints(permutation))

    if node.target == "aten.t.default":
        return context.graph.transpose(inputs[0])

    dim0 = node.attrs.get("dim0")
    dim1 = node.attrs.get("dim1")

    if dim0 is None or dim1 is None:
        raise UnsupportedLoweringError(f"{node.name}: transpose lowering missing permutation or dim attrs")

    return context.graph.permute(inputs[0], swap_permutation(parent_rank(node), int(dim0), int(dim1)))


def can_alias_quantized_weight_transpose(context: models.GenerationContext, node: IRModels.Node, value: Any) -> bool:
    return (
        is_weight_transpose_node(node)
        and is_cq_tensor(context, value)
        and bool(node.children)
        and all(child.target in constants.MATMUL_TARGETS or child.target in constants.ADDM_CONST_TARGETS for child in node.children)
    )


def is_weight_transpose_node(node: IRModels.Node) -> bool:
    if not node.parents:
        return False

    parent = node.parents[0]
    if not parent.is_placeholder or parent.value_kind not in constants.WEIGHT_VALUE_KINDS:
        return False

    if node.target == "aten.t.default":
        return True

    rank = len(meta_shape(parent))
    if rank < 2:
        return False

    permutation = node.attrs.get("permutation")
    if permutation is not None:
        return tuple_ints(permutation) == swap_permutation(rank, rank - 2, rank - 1)

    dim0 = node.attrs.get("dim0")
    dim1 = node.attrs.get("dim1")
    if dim0 is None or dim1 is None:
        return False

    return {
        normalize_dim(int(dim0), rank),
        normalize_dim(int(dim1), rank),
    } == {rank - 2, rank - 1}


def is_cq_tensor(context: models.GenerationContext, value: Any) -> bool:
    return getattr(value, "dtype", None) in {
        context.graph.CQ1,
        context.graph.CQ2,
        context.graph.CQ3,
        context.graph.CQ4,
    }


def lower_slice(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    axis_value = node.attrs.get("axis")

    if axis_value is None:
        axis_value = node.attrs.get("dim")

    if axis_value is None:
        if shape_matches_tensor(node.parents[0], output_shape(node)):
            return inputs[0]

        raise UnsupportedLoweringError(f"{node.name}: slice lowering missing axis/dim attr")

    axis = normalize_dim(int(axis_value), len(meta_shape(node.parents[0])))

    if node.target == "aten.select.int" or "index_value" in node.attrs:
        return context.graph.index(inputs[0], int(node.attrs.get("index_value", node.attrs.get("index", 0))), axis=axis)

    start = int(node.attrs.get("start", 0) or 0)
    step = node.attrs.get("step", 1)

    if is_full_axis_slice(node, axis, start, step):
        return inputs[0]

    if step is not None and int(step) != 1:
        target_shape = output_shape(node)
        length = target_shape[axis] if axis < len(target_shape) else slice_length(node, start)
        return context.graph.strided_slice(inputs[0], axis, start, length, int(step))

    length = slice_length(node, start)
    return context.graph.slice(inputs[0], axis, start, length=length)


def is_full_axis_slice(node: IRModels.Node, axis: int, start: int, step: Any) -> bool:
    if start != 0:
        return False

    if step is not None and int(step) != 1:
        return False

    if not node.parents:
        return False

    return shape_matches_tensor(node.parents[0], output_shape(node))


def lower_index(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if node.target == "aten.index.Tensor":
        return lower_index_tensor(context, node)

    inputs = require_input_count(context, node, 1)
    index_value = node.attrs.get("index_value", node.attrs.get("index"))

    if index_value is None:
        raise UnsupportedLoweringError(f"{node.name}: index lowering missing index_value/index attr")

    return context.graph.index(inputs[0], int(index_value), axis=int(node.attrs.get("axis", node.attrs.get("dim", 0))))


def lower_index_tensor(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if len(inputs) < 2:
        raise unsupported_arity(node, len(inputs), "source tensor plus at least one index tensor")

    source = inputs[0]
    source_shape = meta_shape(node.parents[0])
    target_shape = meta_shape(node)

    if len(inputs) >= 2 and len(source_shape) == 2 and len(target_shape) == 3 and target_shape[-1] == source_shape[-1]:
        return context.graph.embedding_from_tensor(source, inputs[1])

    if len(source_shape) == 3 and len(target_shape) == 2 and concrete_dim(source_shape[0]) == 1:
        feature_count = concrete_dim(source_shape[1])
        feature_dim = concrete_dim(source_shape[2])

        if feature_count is not None and feature_dim is not None:
            return context.graph.reshape(source, (feature_count, feature_dim))

    if len(source_shape) == 2 and len(target_shape) == 4 and concrete_dim(source_shape[0]) == 1:
        source_length = concrete_dim(source_shape[1])
        target_length = concrete_dim(target_shape[-1])

        if source_length is not None and target_length is not None:
            reshaped = context.graph.reshape(source, (1, 1, 1, source_length))

            if target_length < source_length:
                return context.graph.slice(reshaped, 3, 0, length=target_length)

            if target_length == source_length:
                return reshaped

    concrete_target_shape = concrete_shape(target_shape)

    if concrete_target_shape is not None and element_count(source_shape) == element_count(concrete_target_shape):
        return context.graph.reshape(source, concrete_target_shape)

    raise UnsupportedLoweringError(f"{node.name}: unsupported aten.index.Tensor pattern")


def lower_where(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 3)
    output_precision = cactus_precision(context.graph, tensor_dtype(node))
    true_value = cast_to_precision(context, inputs[1], output_precision)
    false_value = cast_to_precision(context, inputs[2], output_precision)
    return context.graph.where(inputs[0], true_value, false_value)


def lower_unfold(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    dimension = int(node.attrs.get("dimension", node.attrs.get("dim", node.attrs.get("arg_1", 0))))
    size = int(node.attrs.get("size", node.attrs.get("arg_2", 1)))
    step = int(node.attrs.get("step", node.attrs.get("arg_3", 1)))
    return context.graph.unfold(inputs[0], dimension, size, step)


def lower_cat(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if node.name in context.prefill_cache_cat_annotations:
        prefill_cache = lower_prefill_cache_cat(context, node)

        if prefill_cache is not None:
            return prefill_cache

    decode_cache = lower_decode_cache_cat(context, node)

    if decode_cache is not None:
        return decode_cache

    inputs = context.inputs_for(node)

    if not inputs:
        raise unsupported_arity(node, 0, "at least one input")

    passthrough = empty_cat_passthrough(node, inputs)

    if passthrough is not None:
        return passthrough

    axis = normalize_dim(int(node.attrs.get("axis", node.attrs.get("dim", 0))), cat_rank(node))
    return context.graph.cat(fp16_cat_inputs(context, inputs), axis=axis)


def fp16_cat_inputs(context: models.GenerationContext, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    fp16 = int(context.graph.FP16)
    return tuple(value if getattr(value, "dtype", None) == fp16 else context.graph.precision_cast(value, fp16) for value in inputs)


def lower_matmul(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)

    if node.target == "aten.bmm.default":
        return lower_bmm(context, node, inputs)

    lhs = matmul_activation_operand(context, inputs[0])
    rhs = inputs[1]
    lhs = slice_decoder_prefill_logits_lhs(context, node, lhs)

    if len(node.parents) >= 2 and is_weight_transpose_node(node.parents[1]):
        original_weight_node = node.parents[1].parents[0]
        original_weight = context.lookup(original_weight_node.name)

        if original_weight is not None and is_cq_tensor(context, original_weight):
            return context.graph.matmul(lhs, original_weight, pretransposed_rhs=True)

    if not is_cq_tensor(context, rhs):
        rhs = matmul_activation_operand(context, rhs)

    return context.graph.matmul(
        lhs,
        rhs,
        pretransposed_rhs=bool(node.attrs.get("pretransposed_rhs", False)),
    )


def matmul_activation_operand(context: models.GenerationContext, value: Any) -> Any:
    if getattr(value, "dtype", None) == context.graph.FP32:
        return cast_to_precision(context, value, context.graph.FP16)

    return value


def slice_decoder_prefill_logits_lhs(context: models.GenerationContext, node: IRModels.Node, lhs: Any) -> Any:
    if not is_decoder_prefill_logits_matmul(context, node):
        return lhs

    lhs_shape = meta_shape(node.parents[0]) if node.parents else ()

    if len(lhs_shape) != 2 or not isinstance(lhs_shape[0], int) or lhs_shape[0] <= 1:
        return lhs

    return context.graph.slice(lhs, 0, int(lhs_shape[0]) - 1, length=1)


def is_decoder_prefill_logits_matmul(context: models.GenerationContext, node: IRModels.Node) -> bool:
    shape = meta_shape(node)

    return (
        context.component.name == "decoder_prefill_chunk"
        and node.target in constants.MATMUL_TARGETS
        and len(shape) == 2
        and len(node.parents) >= 2
        and isinstance(shape[0], int)
        and shape[0] > 1
        and isinstance(shape[-1], int)
        and shape[-1] >= 100000
    )


def decoder_prefill_logits_shape(context: models.GenerationContext, node: IRModels.Node, shape: tuple[int, ...]) -> tuple[int, ...]:
    if (
        context.component.name == "decoder_prefill_chunk"
        and len(shape) == 3
        and shape[0] == 1
        and isinstance(shape[1], int)
        and shape[1] > 1
        and shape[2] >= 100000
    ):
        return (1, 1, shape[2])

    return shape


def empty_cat_passthrough(node: IRModels.Node, inputs: tuple[Any, ...]) -> Any | None:
    target_shape = meta_shape(node)
    non_empty_inputs = []

    for input_value, parent in zip(inputs, node.parents):
        parent_shape = meta_shape(parent)
        parent_elements = element_count(parent_shape)

        if parent_elements == 0:
            continue

        non_empty_inputs.append((input_value, parent_shape))

    if len(non_empty_inputs) != 1:
        return None

    input_value, input_shape = non_empty_inputs[0]

    if tuple(input_shape) == tuple(target_shape):
        return input_value

    return None


def cat_rank(node: IRModels.Node) -> int:
    target_shape = meta_shape(node)

    if target_shape:
        return len(target_shape)

    return len(meta_shape(node.parents[0])) if node.parents else 0


def lower_bmm(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> Any:
    left_shape = meta_shape(node.parents[0])
    right_shape = meta_shape(node.parents[1])

    if len(left_shape) != 3 or len(right_shape) != 3:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires rank-3 inputs")

    if left_shape[0] != right_shape[0]:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires matching batch dimensions")

    batch_size = concrete_dim(left_shape[0])

    if batch_size is None:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires a concrete batch size")

    if batch_size == 1:
        left_2d = matmul_activation_operand(context, context.graph.reshape(inputs[0], (left_shape[1], left_shape[2])))
        right_2d = matmul_activation_operand(context, context.graph.reshape(inputs[1], (right_shape[1], right_shape[2])))
        product = context.graph.matmul(left_2d, right_2d)
        return context.graph.reshape(product, output_shape(node))

    batch_outputs = []

    for batch_index in range(batch_size):
        left_2d = matmul_activation_operand(context, context.graph.index(inputs[0], batch_index, axis=0))
        right_2d = matmul_activation_operand(context, context.graph.index(inputs[1], batch_index, axis=0))
        product = context.graph.matmul(left_2d, right_2d)
        batch_outputs.append(context.graph.reshape(product, (1, left_shape[1], right_shape[2])))

    return context.graph.cat(batch_outputs, axis=0)


def lower_addmm(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 3)
    pretransposed_rhs = False
    rhs = inputs[2]

    if len(node.parents) >= 3 and is_weight_transpose_node(node.parents[2]):
        original_weight_node = node.parents[2].parents[0]
        original_weight = context.lookup(original_weight_node.name)

        if original_weight is not None and is_cq_tensor(context, original_weight):
            rhs = original_weight
            pretransposed_rhs = True

    product = context.graph.matmul(
        matmul_activation_operand(context, inputs[1]),
        rhs if is_cq_tensor(context, rhs) else matmul_activation_operand(context, rhs),
        pretransposed_rhs=pretransposed_rhs,
    )
    return context.graph.add(inputs[0], product)


def lower_split(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    split_sizes = node.attrs.get("split_sizes")

    if not isinstance(split_sizes, list):
        raise UnsupportedLoweringError(f"{node.name}: split lowering requires split_sizes list")

    axis = normalize_dim(int(node.attrs.get("axis", node.attrs.get("dim", 0))), len(meta_shape(node.parents[0])))
    offset = 0
    outputs = []

    for size in split_sizes:
        size_int = int(size)
        outputs.append(context.graph.slice(inputs[0], axis, offset, length=size_int))
        offset += size_int

    return tuple(outputs)


def lower_getitem(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    index = int(node.attrs.get("index", 0))
    source = inputs[0]

    if isinstance(source, (tuple, list)):
        return source[index]

    if index == 0:
        return source

    raise UnsupportedLoweringError(f"{node.name}: getitem index {index} requires tuple-producing parent")


def lower_to_copy(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    dtype = node.attrs.get("dtype")

    if dtype is None or dtype == "torch.bool":
        return inputs[0]

    if str(dtype) == "torch.float32" and can_skip_float32_copy(node):
        return inputs[0]

    precision_name = constants.DTYPE_TO_PRECISION.get(str(dtype))

    if precision_name is None:
        return inputs[0]

    return cast_to_precision(context, inputs[0], cactus_precision(context.graph, str(dtype)))


def can_skip_float32_copy(node: IRModels.Node) -> bool:
    if not node.children or not node.parents:
        return False

    source_dtype = tensor_dtype(node.parents[0])

    if constants.DTYPE_TO_PRECISION.get(str(source_dtype)) != "FP16":
        return False

    safe_consumers = {
        "aten.bmm.default",
        "cactus.attention",
        "cactus.attention_cached",
        "cactus.add",
        "cactus.dense_mlp_tq_fused",
        "cactus.multiply",
        "cactus.pow",
        "cactus.rope",
        "cactus.rms_norm",
        "cactus.scalar_multiply",
        "cactus.silu",
        "cactus.slice",
        "cactus.transpose",
        "cactus.view",
    }
    return all(child.target in safe_consumers for child in node.children)


def lower_pass_through(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_at_least_one_input(context, node)
    return inputs[0]


def lower_copy(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_at_least_one_input(context, node)
    return inputs[-1]
