from typing import Any, Callable

from . import models, match_utils
from ..Fusions import models as FModels


ExtraMatcher = Callable[[models.Node, models.Graph, FModels.FusionGraph, dict[str, models.Node], dict[str, Any]], bool]


############################################ Extra Matcher Utils!!!!! ############################################
def get_bound_parent(bindings: dict[str, models.Node], node_name: str, parent_index: int) -> models.Node | None:
    node = bindings.get(node_name)

    if node is None:
        return None

    return match_utils.get_parent(node, parent_index)


def get_node_dim(node: models.Node, dim_index: int) -> Any:
    shape = match_utils.get_tensor_shape(node)

    if not shape:
        return match_utils.MISSING

    if dim_index < 0:
        dim_index += len(shape)

    if dim_index < 0 or dim_index >= len(shape):
        return match_utils.MISSING

    return shape[dim_index]


def shapes_are_broadcastable(left_shape: list[Any], right_shape: list[Any]) -> bool:
    for left_dim, right_dim in zip(reversed(left_shape), reversed(right_shape)):
        if match_utils.values_equal(left_dim, 1) or match_utils.values_equal(right_dim, 1):
            continue

        if not match_utils.values_equal(left_dim, right_dim):
            return False

    return True


def get_layer_key(node: models.Node) -> str | None:
    for value in (module_stack_text(node), node.name, node.target):
        layer_key = first_layer_key(value)

        if layer_key is not None:
            return layer_key

    return None


def module_stack_text(node: models.Node) -> str:
    return "" if node.module_stack is None else repr(node.module_stack)


def first_layer_key(value: str) -> str | None:
    parts = value.replace("/", ".").replace("_", ".").split(".")

    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "layer", "h", "blocks", "block"} and parts[index + 1].isdigit():
            return f"{part}.{parts[index + 1]}"

    return None



############################################ Extra Matchers!!!!! ############################################
def match_extra_constraints(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for constraint_name, raw_spec in fusion.constraints.items():
        matcher = EXTRA_MATCHERS.get(constraint_name)

        if matcher is None:
            return False

        specs = normalize_constraint_specs(raw_spec)

        if specs is None:
            return False

        for spec in specs:
            if not matcher(source, graph, fusion, bindings, spec):
                return False

    return True


def normalize_constraint_specs(raw_spec: Any) -> tuple[dict[str, Any], ...] | None:
    if isinstance(raw_spec, dict):
        return (raw_spec,)

    if isinstance(raw_spec, (list, tuple)) and all(isinstance(spec, dict) for spec in raw_spec):
        return tuple(raw_spec)

    return None


def match_note(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    return True


def match_node_attr_equals(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    node = bindings.get(spec["node"])

    if node is None:
        return False

    actual = node.attrs.get(spec["attr"], match_utils.MISSING)

    if actual is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(actual, spec["value"])


def match_node_attrs_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
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
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])

    if input_node is None:
        return bool(spec.get("allow_missing", False))

    return input_node.value_kind in spec["allowed_value_kinds"]


def match_parent_tensor_dim_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
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




EXTRA_MATCHERS: dict[str, ExtraMatcher] = {
    "note": match_note,
    "node_attr_equals": match_node_attr_equals,
    "node_attrs_equal": match_node_attrs_equal,
    "input_value_kind": match_input_value_kind,
    "parent_tensor_dim_equal": match_parent_tensor_dim_equal,
    "input_broadcastable_to_node": match_input_broadcastable_to_node,
    "same_layer": match_same_layer,
}
