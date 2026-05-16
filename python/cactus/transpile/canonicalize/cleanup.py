from __future__ import annotations

from collections import Counter

import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.ops import canonicalize_op
from cactus.transpile.ops import get_op
from cactus.transpile.ops import has_op
from cactus.transpile.canonicalize.utils import bypass_unary_node
from cactus.transpile.canonicalize.utils import dce
from cactus.transpile.canonicalize.utils import ir_dtype_to_torch
from cactus.transpile.canonicalize.utils import materialize_constant_output
from cactus.transpile.canonicalize.utils import normalize_dim
from cactus.transpile.canonicalize.utils import normalize_dtype_name
from cactus.transpile.canonicalize.utils import numel
from cactus.transpile.canonicalize.utils import rebuild_graph


COMPILER_SUPPORTED_OPS = {
    "arange",
    "ones",
}

CANONICAL_UNSUPPORTED_RENAMES = {
    "aten.__and__.Tensor": "logical_and",
    "aten.__or__.Tensor": "logical_or",
    "aten.cumsum.default": "cumsum",
    "aten.eq.Scalar": "equal",
    "aten.eq.Tensor": "equal",
    "aten.ge.Scalar": "greater_equal",
    "aten.ge.Tensor": "greater_equal",
    "aten.gt.Scalar": "greater",
    "aten.gt.Tensor": "greater",
    "aten.index.Tensor": "advanced_index",
    "aten.le.Scalar": "less_equal",
    "aten.le.Tensor": "less_equal",
    "aten.lt.Scalar": "less",
    "aten.lt.Tensor": "less",
}

FP32_SUPPORTED_ALL_INPUT_OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "precision_cast",
    "view",
    "permute",
    "transpose",
    "slice",
    "index",
    "advanced_index",
    "gather",
    "matmul",
    "linear",
    "addmm",
    "layer_norm",
    "batch_norm",
    "glu",
    "sum",
    "mean",
    "variance",
    "min",
    "max",
    "scalar_add",
    "scalar_subtract",
    "scalar_subtract_reverse",
    "scalar_multiply",
    "scalar_divide",
    "scalar_floor_divide",
    "scalar_not_equal",
    "scalar_equal",
    "scalar_greater",
    "scalar_greater_equal",
    "scalar_less",
    "scalar_less_equal",
    "abs",
    "negate",
    "pow",
    "scalar_exp",
    "scalar_sqrt",
    "scalar_log",
    "scalar_cos",
    "scalar_sin",
}

FP32_SUPPORTED_INPUT_INDICES = {
    "embedding": {1},
}

FP16_ONLY_OUTPUT_OPS = {
    "scalar_divide_reverse",
    "not_equal",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_not",
    "one_hot",
    "scalar_not_equal",
    "scalar_equal",
    "scalar_greater",
    "scalar_greater_equal",
    "scalar_less",
    "scalar_less_equal",
    "cat",
    "softmax",
    "rms_norm",
    "embedding",
    "group_norm",
    "lstm_cell",
    "relu",
    "silu",
    "gelu",
    "gelu_erf",
    "sigmoid",
    "tanh",
    "attention",
    "attention_block",
    "conv_module",
    "dense_mlp_tq_fused",
    "rope",
}


def canonicalize_exported_graph(graph: IRGraph, *, max_passes: int = 8) -> IRGraph:
    verify_ir(graph)

    for _ in range(max_passes):
        changed = False

        for node_id in list(graph.order):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if _canonicalize_node(graph, node):
                changed = True

        rebuild_graph(graph)

        for node_id in list(graph.order):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if _simplify_node(graph, node):
                changed = True

        rebuild_graph(graph)

        if _legalize_precisions(graph):
            changed = True
            rebuild_graph(graph)

        dce(graph)

        if not changed:
            break

    unsupported_counts = summarize_unsupported_ops(graph)
    graph.meta = dict(graph.meta)
    graph.meta["canonical_unsupported_ops"] = tuple(sorted(unsupported_counts))
    graph.meta["canonical_unsupported_op_counts"] = unsupported_counts
    verify_ir(graph)
    return graph


def summarize_unsupported_ops(graph: IRGraph) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node_id in graph.order:
        node = graph.nodes[node_id]
        if not _is_supported_without_backend_kernel(node.op):
            counts[node.op] += 1
    return dict(sorted(counts.items()))


def _canonicalize_node(graph: IRGraph, node: IRNode) -> bool:
    original_op = node.op

    if original_op in {"aten.new_ones.default", "aten.ones.default"}:
        return _materialize_new_ones_constant(graph, node)

    if original_op in {"aten.new_zeros.default", "aten.zeros.default", "new_zeros"}:
        return _materialize_new_zeros_constant(graph, node)

    if original_op == "arange":
        return _materialize_arange_constant(graph, node)

    if original_op in {"identity", "contiguous"}:
        return bypass_unary_node(graph, node)

    if original_op in CANONICAL_UNSUPPORTED_RENAMES:
        node.op = CANONICAL_UNSUPPORTED_RENAMES[original_op]
        return True

    if original_op == "multiply_inplace":
        node.op = "multiply"
        return True

    if original_op == "view" and "shape" not in node.attrs:
        return _rewrite_view_from_output_shape(graph, node)

    if original_op in {"reshape", "flatten", "unsqueeze", "squeeze"}:
        return _rewrite_view_from_output_shape(graph, node)

    if original_op == "expand":
        return _rewrite_expand_if_view_equivalent(graph, node)

    if original_op == "transpose":
        return _rewrite_transpose_as_permute(graph, node)

    if original_op == "movedim":
        return _rewrite_movedim_as_permute(graph, node)

    if original_op == "type_as":
        return _rewrite_type_as(graph, node)

    if original_op == "concat":
        node.op = "cat"
        if "dim" in node.attrs and "axis" not in node.attrs:
            node.attrs = {**node.attrs, "axis": int(node.attrs["dim"])}
        return True

    if original_op == "scaled_dot_product_attention":
        node.op = "attention"
        node.attrs = {
            "scale": float(node.attrs.get("scale", 0.0)),
            "is_causal": bool(node.attrs.get("is_causal", False)),
            "window_size": int(node.attrs.get("window_size", 0)),
            **({"mask": node.attrs["mask"]} if "mask" in node.attrs else {}),
            **({"additive_mask": bool(node.attrs.get("additive_mask", False))} if "additive_mask" in node.attrs else {}),
        }
        return True

    canonical_op = canonicalize_op(original_op)
    if canonical_op != original_op:
        node.op = canonical_op
        _normalize_attr_spellings(node)
        if has_op(node.op):
            _filter_attrs_to_schema(node)
        return True

    if has_op(node.op):
        _normalize_attr_spellings(node)
        _filter_attrs_to_schema(node)
        return True

    return False


def _simplify_node(graph: IRGraph, node: IRNode) -> bool:
    if _fold_constant_node(graph, node):
        return True

    if node.op == "precision_cast":
        return _remove_noop_precision_cast(graph, node)

    if node.op == "view":
        if _remove_noop_view(graph, node):
            return True
        return _collapse_view_chain(graph, node)

    if node.op == "slice":
        return _remove_noop_slice(graph, node)

    if node.op == "scalar_add":
        return _remove_noop_scalar(graph, node, neutral=0.0)

    if node.op == "scalar_subtract":
        return _remove_noop_scalar(graph, node, neutral=0.0)

    if node.op == "scalar_multiply":
        return _remove_noop_scalar(graph, node, neutral=1.0)

    if node.op == "scalar_divide":
        return _remove_noop_scalar(graph, node, neutral=1.0)

    return False


def _fold_constant_node(graph: IRGraph, node: IRNode) -> bool:
    if len(node.outputs) != 1:
        return False

    output_value = graph.values.get(node.outputs[0])
    if output_value is None:
        return False

    if node.op in FP16_ONLY_OUTPUT_OPS:
        output_value.dtype = "fp16"
        node.meta["dtype"] = "fp16"

    output_torch_dtype = ir_dtype_to_torch(output_value.dtype) if output_value.dtype is not None else None

    constant_inputs = [graph.constants.get(value_id) for value_id in node.inputs]
    if node.op == "view" and len(constant_inputs) == 1 and isinstance(constant_inputs[0], torch.Tensor):
        if output_value.shape is None:
            return False
        return materialize_constant_output(
            graph,
            node,
            constant_inputs[0].reshape(tuple(int(dim) for dim in output_value.shape)).to(dtype=output_torch_dtype),
        )

    if node.op == "expand" and len(constant_inputs) == 1 and isinstance(constant_inputs[0], torch.Tensor):
        if output_value.shape is None:
            return False
        return materialize_constant_output(
            graph,
            node,
            constant_inputs[0].expand(tuple(int(dim) for dim in output_value.shape)).clone().to(dtype=output_torch_dtype),
        )

    if node.op == "precision_cast" and len(constant_inputs) == 1 and isinstance(constant_inputs[0], torch.Tensor):
        dtype = normalize_dtype_name(node.attrs.get("dtype"))
        if dtype is None:
            return False
        return materialize_constant_output(graph, node, constant_inputs[0].to(dtype=ir_dtype_to_torch(dtype)))

    if node.op in {"cos", "scalar_cos"} and len(constant_inputs) == 1 and isinstance(constant_inputs[0], torch.Tensor):
        return materialize_constant_output(graph, node, torch.cos(constant_inputs[0]))

    if node.op in {"sin", "scalar_sin"} and len(constant_inputs) == 1 and isinstance(constant_inputs[0], torch.Tensor):
        return materialize_constant_output(graph, node, torch.sin(constant_inputs[0]))

    if node.op in {"scalar_add", "scalar_subtract", "scalar_multiply", "scalar_divide"}:
        if len(constant_inputs) != 1 or not isinstance(constant_inputs[0], torch.Tensor):
            return False
        scalar = float(node.attrs["value"])
        if node.op == "scalar_add":
            result = constant_inputs[0] + scalar
        elif node.op == "scalar_subtract":
            result = constant_inputs[0] - scalar
        elif node.op == "scalar_multiply":
            result = constant_inputs[0] * scalar
        else:
            result = constant_inputs[0] / scalar
        return materialize_constant_output(graph, node, result.to(dtype=output_torch_dtype))

    if node.op in {"scalar_not_equal", "scalar_equal", "scalar_greater", "scalar_greater_equal", "scalar_less", "scalar_less_equal"}:
        if len(constant_inputs) != 1 or not isinstance(constant_inputs[0], torch.Tensor):
            return False
        scalar = float(node.attrs["value"])
        if node.op == "scalar_not_equal":
            result = constant_inputs[0] != scalar
        elif node.op == "scalar_equal":
            result = constant_inputs[0] == scalar
        elif node.op == "scalar_greater":
            result = constant_inputs[0] > scalar
        elif node.op == "scalar_greater_equal":
            result = constant_inputs[0] >= scalar
        elif node.op == "scalar_less":
            result = constant_inputs[0] < scalar
        else:
            result = constant_inputs[0] <= scalar
        return materialize_constant_output(graph, node, result.to(dtype=output_torch_dtype))

    if node.op in {"not_equal", "equal", "greater", "greater_equal", "less", "less_equal", "logical_and", "logical_or"}:
        if len(constant_inputs) != 2 or not all(isinstance(value, torch.Tensor) for value in constant_inputs):
            return False
        lhs, rhs = constant_inputs
        if node.op == "not_equal":
            result = lhs != rhs
        elif node.op == "equal":
            result = lhs == rhs
        elif node.op == "greater":
            result = lhs > rhs
        elif node.op == "greater_equal":
            result = lhs >= rhs
        elif node.op == "less":
            result = lhs < rhs
        elif node.op == "less_equal":
            result = lhs <= rhs
        elif node.op == "logical_and":
            result = torch.logical_and(lhs != 0, rhs != 0)
        else:
            result = torch.logical_or(lhs != 0, rhs != 0)
        return materialize_constant_output(graph, node, result.to(dtype=output_torch_dtype))

    if node.op == "logical_not":
        if len(constant_inputs) != 1 or not isinstance(constant_inputs[0], torch.Tensor):
            return False
        result = torch.logical_not(constant_inputs[0] != 0)
        return materialize_constant_output(graph, node, result.to(dtype=output_torch_dtype))

    return False


def _legalize_precisions(graph: IRGraph) -> bool:
    changed = False
    counter = 0

    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if node.op in FP32_SUPPORTED_ALL_INPUT_OPS:
            continue

        inserted: list[IRNode] = []
        allowed_indices = FP32_SUPPORTED_INPUT_INDICES.get(node.op, set())
        for input_index, value_id in enumerate(list(node.inputs)):
            if input_index in allowed_indices:
                continue
            value = graph.values.get(value_id)
            if value is None or normalize_dtype_name(value.dtype) != "fp32":
                continue

            counter += 1
            cast_node_id = _unique_ir_id(
                f"{node.id}_legalize_fp16_{counter}",
                graph.nodes.keys(),
                graph.order,
            )
            cast_value_id = _unique_ir_id(
                f"{value_id}_legalize_fp16_{counter}",
                graph.values.keys(),
                graph.constants.keys(),
            )
            cast_node = IRNode(
                id=cast_node_id,
                op="precision_cast",
                inputs=[value_id],
                outputs=[cast_value_id],
                attrs={"dtype": "fp16"},
                meta={"inserted_by": "canonicalize_exported_graph"},
            )
            graph.values[cast_value_id] = IRValue(
                id=cast_value_id,
                shape=value.shape,
                dtype="fp16",
                producer=cast_node_id,
                users=[],
            )
            node.inputs[input_index] = cast_value_id
            inserted.append(cast_node)
            changed = True

        if inserted:
            insert_at = graph.order.index(node_id)
            for offset, cast_node in enumerate(inserted):
                graph.nodes[cast_node.id] = cast_node
                graph.order.insert(insert_at + offset, cast_node.id)

        if node.op in FP16_ONLY_OUTPUT_OPS:
            for output_id in node.outputs:
                output_value = graph.values.get(output_id)
                if output_value is not None:
                    output_value.dtype = "fp16"
            node.meta["dtype"] = "fp16"

    return changed


def _unique_ir_id(base: str, *existing_groups) -> str:
    existing: set[str] = set()
    for group in existing_groups:
        existing.update(str(item) for item in group)
    if base not in existing:
        return base
    suffix = 1
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in existing:
            return candidate
        suffix += 1


def _rewrite_view_from_output_shape(graph: IRGraph, node: IRNode) -> bool:
    if not node.outputs:
        return False
    output_value = graph.values.get(node.outputs[0])
    if output_value is None or output_value.shape is None:
        return False
    node.op = "view"
    node.attrs = {"shape": tuple(int(dim) for dim in output_value.shape)}
    return True


def _rewrite_expand_if_view_equivalent(graph: IRGraph, node: IRNode) -> bool:
    if not node.inputs or not node.outputs:
        return False
    input_value = graph.values.get(node.inputs[0])
    output_value = graph.values.get(node.outputs[0])
    if input_value is None or output_value is None:
        return False
    if input_value.shape is None or output_value.shape is None:
        return False
    if numel(input_value.shape) != numel(output_value.shape):
        node.meta["canonicalization_blocked"] = "expand_requires_broadcast"
        return False
    node.op = "view"
    node.attrs = {"shape": tuple(int(dim) for dim in output_value.shape)}
    return True


def _rewrite_transpose_as_permute(graph: IRGraph, node: IRNode) -> bool:
    if not node.inputs:
        return False
    input_value = graph.values.get(node.inputs[0])
    if input_value is None or input_value.shape is None:
        return False
    rank = len(input_value.shape)
    dim0 = normalize_dim(int(node.attrs["dim0"]), rank)
    dim1 = normalize_dim(int(node.attrs["dim1"]), rank)
    permutation = list(range(rank))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    node.op = "permute"
    node.attrs = {"permutation": tuple(permutation)}
    return True


def _rewrite_movedim_as_permute(graph: IRGraph, node: IRNode) -> bool:
    if not node.inputs:
        return False
    input_value = graph.values.get(node.inputs[0])
    if input_value is None or input_value.shape is None:
        return False
    rank = len(input_value.shape)
    source = [normalize_dim(int(dim), rank) for dim in node.attrs["source"]]
    destination = [normalize_dim(int(dim), rank) for dim in node.attrs["destination"]]
    if len(source) != len(destination):
        return False

    remaining = [dim for dim in range(rank) if dim not in source]
    permutation = list(remaining)
    for dst, src in sorted(zip(destination, source)):
        permutation.insert(dst, src)

    node.op = "permute"
    node.attrs = {"permutation": tuple(permutation)}
    return True


def _rewrite_type_as(graph: IRGraph, node: IRNode) -> bool:
    if not node.inputs:
        return False
    target_dtype = None
    if node.outputs:
        output_value = graph.values.get(node.outputs[0])
        if output_value is not None:
            target_dtype = output_value.dtype
    if target_dtype is None and len(node.inputs) > 1:
        target_value = graph.values.get(node.inputs[1])
        if target_value is not None:
            target_dtype = target_value.dtype
    if target_dtype is None:
        return False
    node.op = "precision_cast"
    node.inputs = [node.inputs[0]]
    node.attrs = {"dtype": normalize_dtype_name(target_dtype)}
    return True


def _normalize_attr_spellings(node: IRNode) -> None:
    attrs = dict(node.attrs)
    if "dim" in attrs and "axis" not in attrs and node.op in {"softmax", "sum", "mean", "variance", "min", "max", "cat", "slice", "index"}:
        attrs["axis"] = attrs.pop("dim")
    if node.op == "permute" and "dims" in attrs and "permutation" not in attrs:
        attrs["permutation"] = tuple(int(dim) for dim in attrs.pop("dims"))
    if node.op == "slice":
        attrs.setdefault("step", 1)
    if node.op == "precision_cast" and "dtype" in attrs:
        attrs["dtype"] = normalize_dtype_name(attrs["dtype"])
    node.attrs = attrs


def _filter_attrs_to_schema(node: IRNode) -> None:
    schema = get_op(node.op)
    if not schema.attrs:
        node.attrs = {}
        return
    node.attrs = {name: node.attrs[name] for name in schema.attrs if name in node.attrs}


def _remove_noop_precision_cast(graph: IRGraph, node: IRNode) -> bool:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    input_value = graph.values.get(node.inputs[0])
    output_value = graph.values.get(node.outputs[0])
    target_dtype = normalize_dtype_name(node.attrs.get("dtype"))
    input_dtype = None if input_value is None else normalize_dtype_name(input_value.dtype)
    output_dtype = None if output_value is None else normalize_dtype_name(output_value.dtype)
    if target_dtype is None or target_dtype == input_dtype or (input_dtype is not None and output_dtype == input_dtype):
        return bypass_unary_node(graph, node)
    return False


def _remove_noop_view(graph: IRGraph, node: IRNode) -> bool:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    input_value = graph.values.get(node.inputs[0])
    output_value = graph.values.get(node.outputs[0])
    if input_value is None or output_value is None:
        return False
    if input_value.shape is None or output_value.shape is None:
        return False
    if tuple(input_value.shape) == tuple(output_value.shape):
        return bypass_unary_node(graph, node)
    return False


def _collapse_view_chain(graph: IRGraph, node: IRNode) -> bool:
    if len(node.inputs) != 1:
        return False
    input_value = graph.values.get(node.inputs[0])
    if input_value is None or input_value.producer is None:
        return False
    producer = graph.nodes.get(input_value.producer)
    if producer is None or producer.op != "view" or len(producer.inputs) != 1:
        return False
    node.inputs = [producer.inputs[0]]
    return True


def _remove_noop_slice(graph: IRGraph, node: IRNode) -> bool:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    input_value = graph.values.get(node.inputs[0])
    output_value = graph.values.get(node.outputs[0])
    if input_value is None or output_value is None:
        return False
    if input_value.shape is None or output_value.shape is None:
        return False
    if tuple(input_value.shape) == tuple(output_value.shape):
        return bypass_unary_node(graph, node)
    return False


def _remove_noop_scalar(graph: IRGraph, node: IRNode, *, neutral: float) -> bool:
    value = node.attrs.get("value")
    if value is None:
        return False
    if float(value) != float(neutral):
        return False
    return bypass_unary_node(graph, node)


def _materialize_new_ones_constant(graph: IRGraph, node: IRNode) -> bool:
    if len(node.outputs) != 1:
        return False
    output_id = node.outputs[0]
    output_value = graph.values.get(output_id)
    if output_value is None or output_value.shape is None or output_value.dtype is None:
        return False
    torch_dtype = ir_dtype_to_torch(output_value.dtype)
    const = torch.ones(tuple(int(dim) for dim in output_value.shape), dtype=torch_dtype)
    return materialize_constant_output(graph, node, const)


def _materialize_new_zeros_constant(graph: IRGraph, node: IRNode) -> bool:
    if len(node.outputs) != 1:
        return False
    output_id = node.outputs[0]
    output_value = graph.values.get(output_id)
    if output_value is None or output_value.shape is None or output_value.dtype is None:
        return False
    torch_dtype = ir_dtype_to_torch(output_value.dtype)
    const = torch.zeros(tuple(int(dim) for dim in output_value.shape), dtype=torch_dtype)
    return materialize_constant_output(graph, node, const)


def _materialize_arange_constant(graph: IRGraph, node: IRNode) -> bool:
    if len(node.outputs) != 1:
        return False
    start = int(node.attrs.get("start", 0))
    end = node.attrs.get("end")
    if end is None:
        return False
    step = int(node.attrs.get("step", 1))
    dtype = normalize_dtype_name(node.attrs.get("dtype", "fp32"))
    torch_dtype = ir_dtype_to_torch(dtype)
    const = torch.arange(start, int(end), step=step, dtype=torch_dtype)
    return materialize_constant_output(graph, node, const)


def _is_supported_without_backend_kernel(op_name: str) -> bool:
    return has_op(op_name) or op_name in COMPILER_SUPPORTED_OPS
