from __future__ import annotations

from typing import Any

from . import models, match_utils
from .extra_matcher_common import *
from ..Fusions import models as FModels

def match_cache_roll_append_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    cache_node = match_utils.get_first_input_by_role(fusion, bindings, spec["cache_role"])
    new_data_node = match_utils.get_first_input_by_role(fusion, bindings, spec["new_data_role"])

    if cache_node is None or new_data_node is None:
        return False

    cache_shape = match_utils.get_tensor_shape(cache_node)
    new_data_shape = match_utils.get_tensor_shape(new_data_node)
    output_shape = match_utils.get_tensor_shape(source)

    if not cache_shape or not new_data_shape or not output_shape:
        return False

    if not match_utils.values_equal(cache_shape, output_shape):
        return False

    if len(cache_shape) != len(new_data_shape):
        return False

    window_size = get_known_int(cache_shape[-1])

    if window_size is None or window_size <= 0:
        return False

    if not cache_append_shapes_match(cache_shape, new_data_shape):
        return False

    arange_node = bindings.get(spec["arange_node"])
    add_node = bindings.get(spec["add_node"])
    mod_node = bindings.get(spec["mod_node"])

    if arange_node is None or add_node is None or mod_node is None:
        return False

    if not match_utils.values_equal(get_attr(arange_node, "start", "arg_0"), 0):
        return False

    if not match_utils.values_equal(get_attr(arange_node, "end", "arg_1"), window_size):
        return False

    if not match_utils.values_equal(get_attr(add_node, "other", "value", "arg_1"), 1):
        return False

    if not match_utils.values_equal(get_attr(mod_node, "other", "value", "arg_1"), window_size):
        return False

    return (
        slice_node_matches(bindings.get(spec["scatter_slice_dim0_node"]), axis=0, start=0)
        and slice_node_matches(bindings.get(spec["scatter_slice_dim1_node"]), axis=1, start=0)
        and slice_node_matches(bindings.get(spec["value_slice_dim0_node"]), axis=0, start=0)
        and slice_node_matches(bindings.get(spec["value_slice_dim1_node"]), axis=1, start=0)
        and slice_node_matches(bindings.get(spec["value_slice_last_node"]), axis=2, start=-1)
        and scatter_node_matches(bindings.get(spec["scatter_dim2_node"]), axis=2, start=-1)
        and scatter_node_matches(bindings.get(spec["scatter_dim1_node"]), axis=1, start=0)
        and scatter_node_matches(bindings.get(spec["scatter_dim0_node"]), axis=0, start=0)
    )

def match_empty_cache_initializer(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    empty_node = match_utils.get_first_input_by_role(fusion, bindings, spec["empty_role"])
    new_value_node = match_utils.get_first_input_by_role(fusion, bindings, spec["new_value_role"])
    output_node = bindings.get(spec["output_node"])

    if empty_node is None or new_value_node is None or output_node is None:
        return False

    empty_shape = match_utils.get_tensor_shape(empty_node)
    new_shape = match_utils.get_tensor_shape(new_value_node)
    output_shape = match_utils.get_tensor_shape(output_node)

    if not empty_shape or not new_shape or not output_shape:
        return False

    if element_count(empty_shape) != 0:
        return False

    return match_utils.values_equal(new_shape, output_shape)

def match_conv_cache_initialize_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    rows_node = match_utils.get_first_input_by_role(fusion, bindings, spec["rows_role"])
    empty_node = match_utils.get_first_input_by_role(fusion, bindings, spec["empty_role"])
    output_node = bindings.get(spec["output_node"])

    if rows_node is None or empty_node is None or output_node is None:
        return False

    rows_shape = match_utils.get_tensor_shape(rows_node)
    empty_shape = match_utils.get_tensor_shape(empty_node)
    output_shape = match_utils.get_tensor_shape(output_node)

    if len(rows_shape) != 3 or len(empty_shape) != 3 or len(output_shape) != 3:
        return False

    if not match_utils.values_equal(rows_shape, output_shape):
        return False

    if not match_utils.values_equal(empty_shape, output_shape):
        return False

    if empty_node.target != "aten.full_like.default":
        return False

    if not match_utils.values_equal(get_attr(empty_node, "fill_value", "arg_1"), 0):
        return False

    return get_known_int(rows_shape[1]) is not None and get_known_int(rows_shape[2]) is not None

def match_short_conv_prefill_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    conv_node = bindings.get(spec["conv_node"])
    slice_node = bindings.get(spec["slice_node"])
    x_node = match_utils.get_first_input_by_role(fusion, bindings, spec["x_role"])
    weight_node = match_utils.get_first_input_by_role(fusion, bindings, spec["weight_role"])

    if conv_node is None or slice_node is None or x_node is None or weight_node is None:
        return False

    x_shape = match_utils.get_tensor_shape(x_node)
    weight_shape = match_utils.get_tensor_shape(weight_node)
    conv_shape = match_utils.get_tensor_shape(conv_node)
    output_shape = match_utils.get_tensor_shape(slice_node)

    if len(x_shape) != 3 or len(weight_shape) != 3 or len(conv_shape) != 3 or len(output_shape) != 3:
        return False

    hidden_dim = get_known_int(x_shape[1])
    kernel_size = get_known_int(weight_shape[2])
    input_length = get_known_int(x_shape[2])

    if hidden_dim is None or kernel_size is None or input_length is None:
        return False

    if not match_utils.values_equal(weight_shape[0], hidden_dim):
        return False

    if not match_utils.values_equal(weight_shape[1], 1):
        return False

    if first_int_attr(conv_node, "stride", 1) != 1:
        return False

    if first_int_attr(conv_node, "dilation", 1) != 1:
        return False

    if first_int_attr(conv_node, "padding", 0) != kernel_size - 1:
        return False

    if first_int_attr(conv_node, "groups", 1) != hidden_dim:
        return False

    slice_axis = first_int_attr(slice_node, "axis", first_int_attr(slice_node, "dim", -1))
    if normalize_dim(slice_axis, 3) != 2:
        return False

    if first_int_attr(slice_node, "start", 0) != 0:
        return False

    return match_utils.values_equal(output_shape, x_shape) and get_shape_dim(conv_node, 2) == input_length + kernel_size - 1

def match_short_conv_decode_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    cache_window_node = match_utils.get_first_input_by_role(fusion, bindings, spec["cache_window_role"])
    multiply_node = bindings.get(spec["multiply_node"])
    sum_node = bindings.get(spec["sum_node"])
    weight_node = match_utils.get_first_input_by_role(fusion, bindings, spec["weight_role"])

    if cache_window_node is None or multiply_node is None or sum_node is None or weight_node is None:
        return False

    if cache_window_node.target != "cactus.conv_cache_append" and not (
        cache_window_node.cache is not None and cache_window_node.cache.kind == FModels.CacheKind.CONV
    ):
        return False

    cache_window_shape = match_utils.get_tensor_shape(cache_window_node)
    weight_shape = match_utils.get_tensor_shape(weight_node)
    multiply_shape = match_utils.get_tensor_shape(multiply_node)
    output_shape = match_utils.get_tensor_shape(sum_node)

    if len(cache_window_shape) != 3 or len(multiply_shape) != 3 or len(output_shape) != 2:
        return False

    if not match_utils.values_equal(cache_window_shape, multiply_shape):
        return False

    batch_size = get_known_int(cache_window_shape[0])
    hidden_dim = get_known_int(cache_window_shape[1])
    window_size = get_known_int(cache_window_shape[2])

    if batch_size is None or hidden_dim is None or window_size is None:
        return False

    if not match_utils.values_equal(output_shape, [batch_size, hidden_dim]):
        return False

    if not short_conv_decode_weight_shape_matches(weight_shape, hidden_dim, window_size):
        return False

    sum_axis = first_int_attr(sum_node, "axis", first_int_attr(sum_node, "dim", -1))
    if normalize_dim(sum_axis, 3) != 2:
        return False

    keepdim = get_attr(sum_node, "keepdim")
    return keepdim is match_utils.MISSING or keepdim is False

def match_cache_output_consumers(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    node = bindings.get(spec["node"])

    if node is None:
        return False

    allow_output_child = bool(spec.get("allow_output_child", True))

    for child in node.children:
        if child.target in match_utils.METADATA_ONLY_TARGETS:
            continue

        if allow_output_child and child.is_output:
            continue

        return False

    return True
