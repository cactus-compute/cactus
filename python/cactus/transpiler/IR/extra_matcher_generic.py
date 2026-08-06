from __future__ import annotations

from typing import Any

from . import models, match_utils
from .extra_matcher_common import *
from ..Fusions import models as FModels

def match_node_attr_equals(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks that one matched node has a specific normalized attr value."""
    node = bindings.get(spec["node"])
    if node is None:
        return False
    actual = node.attrs.get(spec["attr"], match_utils.MISSING)
    if actual is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))
    return match_utils.values_equal(actual, spec["value"])

def match_node_attrs_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks that attrs on two matched nodes are equal to each other."""
    left_node = bindings.get(spec["left_node"])
    right_node = bindings.get(spec["right_node"])
    if left_node is None or right_node is None:
        return False
    left_value = left_node.attrs.get(spec["left_attr"], match_utils.MISSING)
    right_value = right_node.attrs.get(spec["right_attr"], match_utils.MISSING)
    if left_value is match_utils.MISSING or right_value is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))
    return match_utils.values_equal(left_value, right_value)

def match_input_value_kind(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks the semantic kind of an external input to a fusion."""
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])
    if input_node is None:
        return bool(spec.get("allow_missing", False))
    return input_node.value_kind in spec["allowed_value_kinds"]

def match_parent_tensor_dim_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Compares one dimension from two parent tensors of matched nodes."""
    left_parent = get_bound_parent(bindings, spec["left_node"], spec["left_parent_index"])
    right_parent = get_bound_parent(bindings, spec["right_node"], spec["right_parent_index"])
    if left_parent is None or right_parent is None:
        return bool(spec.get("allow_missing", False))
    left_dim = get_node_dim(left_parent, spec["left_dim"])
    right_dim = get_node_dim(right_parent, spec["right_dim"])
    if left_dim is match_utils.MISSING or right_dim is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))
    return match_utils.values_equal(left_dim, right_dim)

def match_input_broadcastable_to_node(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks that a declared fusion input can broadcast to a target node shape."""
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])
    target_node = bindings.get(spec["node"])
    if input_node is None or target_node is None:
        return bool(spec.get("allow_missing", False))
    input_shape = match_utils.get_tensor_shape(input_node)
    target_shape = match_utils.get_tensor_shape(target_node)
    if not input_shape or not target_shape:
        return bool(spec.get("allow_missing", False))
    return shapes_are_broadcastable(input_shape, target_shape)

def match_same_layer(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks that multiple matched nodes appear to belong to the same layer."""
    layer_keys = []
    for node_name in spec["nodes"]:
        node = bindings.get(node_name)
        if node is None:
            return False
        layer_key = get_layer_key(node)
        if layer_key is None:
            return bool(spec.get("allow_missing", False))
        layer_keys.append(layer_key)
    return len(set(layer_keys)) == 1

def match_slice_halves(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Checks that two slice nodes split the same tensor into adjacent equal halves."""
    left = bindings.get(spec["left_node"])
    right = bindings.get(spec["right_node"])
    if left is None or right is None:
        return False
    left_source = match_utils.get_parent(left, 0)
    right_source = match_utils.get_parent(right, 0)
    if left_source is None or left_source is not right_source:
        return False
    source_shape = match_utils.get_tensor_shape(left_source)
    left_shape = match_utils.get_tensor_shape(left)
    right_shape = match_utils.get_tensor_shape(right)
    if not source_shape or not left_shape or not right_shape:
        return bool(spec.get("allow_missing", False))
    left_axis = normalized_slice_axis(left, len(source_shape))
    right_axis = normalized_slice_axis(right, len(source_shape))
    if left_axis is match_utils.MISSING or right_axis is match_utils.MISSING or left_axis != right_axis:
        return False
    if not slice_step_is_one(left) or not slice_step_is_one(right):
        return False
    left_size = match_utils.get_dim(left_shape, left_axis)
    right_size = match_utils.get_dim(right_shape, right_axis)
    source_size = match_utils.get_dim(source_shape, left_axis)
    if left_size is None or right_size is None:
        return bool(spec.get("allow_missing", False))
    if not match_utils.values_equal(left_size, right_size):
        return False
    if not attr_matches(left, "start", 0, default=0):
        return False
    if not attr_matches(right, "start", left_size):
        return False
    if not optional_attr_matches(left, "end", left_size):
        return False
    if not optional_attr_matches(right, "end", add_dims(left_size, right_size)):
        return False
    if source_size is not None and not match_utils.values_equal(source_size, add_dims(left_size, right_size)):
        return False
    return True
