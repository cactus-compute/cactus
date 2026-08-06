from __future__ import annotations

from typing import Any, Callable

from . import models, match_utils
from ..Fusions import models as FModels

ExtraMatcher = Callable[[models.Node, models.Graph, FModels.FusionGraph, dict[str, models.Node], dict[str, Any]], bool]

WEIGHT_VALUE_KINDS = (
    FModels.ValueKind.PARAMETER,
    FModels.ValueKind.BUFFER,
    FModels.ValueKind.LIFTED_CONSTANT,
)

COPY_TARGETS = {
    "aten.clone.default",
    "aten._to_copy.default",
    "aten.contiguous.default",
}

TRANSPOSE_TARGETS = {
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
}

COS_TARGETS = {"aten.cos.default"}
SIN_TARGETS = {"aten.sin.default"}
SPLIT_TARGETS = {
    "aten.chunk.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
}

def get_bound_parent(bindings: dict[str, models.Node], node_name: str, parent_index: int) -> models.Node | None:
    """Returns a real parent node from a bound synthetic node/input position."""
    node = bindings.get(node_name)
    if node is None:
        return None
    return match_utils.get_parent(node, parent_index)

def get_node_dim(node: models.Node, dim_index: int) -> Any:
    """Returns one tensor dimension from a real IR node's output shape."""
    shape = match_utils.get_tensor_shape(node)
    if not shape:
        return match_utils.MISSING
    if dim_index < 0:
        dim_index += len(shape)
    if dim_index < 0 or dim_index >= len(shape):
        return match_utils.MISSING
    return shape[dim_index]

def walk_children(node: models.Node, max_depth: int) -> tuple[models.Node, ...]:
    """Returns all reachable child nodes within a bounded forward traversal."""
    found: list[models.Node] = []
    seen: set[int] = set()
    def visit(current: models.Node, depth: int) -> None:
        """Recursively collects children while respecting the maximum depth."""
        if depth > max_depth:
            return
        for child in current.children:
            if id(child) in seen:
                continue
            seen.add(id(child))
            found.append(child)
            visit(child, depth + 1)
    visit(node, 0)
    return tuple(found)

def get_node_dtype(node: models.Node) -> Any:
    """Reads a node's output dtype from tensor metadata."""
    if not isinstance(node.tensor_output_meta, dict):
        return match_utils.MISSING
    return node.tensor_output_meta.get("dtype", match_utils.MISSING)

def normalize_dim(dim: Any, rank: int) -> Any:
    """Converts a possibly negative dimension index into a non-negative one."""
    if not isinstance(dim, int):
        return dim
    return dim + rank if dim < 0 else dim

def get_attr(node: models.Node, *names: str) -> Any:
    """Returns the first available attr from a node."""
    for name in names:
        value = node.attrs.get(name, match_utils.MISSING)
        if value is not match_utils.MISSING:
            return value
    return match_utils.MISSING

def get_known_int(value: Any) -> int | None:
    """Converts plain integer-like values into int, otherwise returns None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None

def get_shape_dim(node: models.Node | None, dim_index: int) -> int | None:
    """Returns a concrete int dimension from a node shape when available."""
    if node is None:
        return None
    return get_known_int(get_node_dim(node, dim_index))

def shapes_are_broadcastable(left_shape: list[Any], right_shape: list[Any]) -> bool:
    """Implements standard trailing-dimension broadcasting for two shapes."""
    for left_dim, right_dim in zip(reversed(left_shape), reversed(right_shape)):
        if match_utils.values_equal(left_dim, 1) or match_utils.values_equal(right_dim, 1):
            continue
        if not match_utils.values_equal(left_dim, right_dim):
            return False
    return True

def get_layer_key(node: models.Node) -> str | None:
    """Extracts a coarse layer identifier from module stack, node name, or target."""
    for value in (module_stack_text(node), node.name, node.target):
        layer_key = first_layer_key(value)
        if layer_key is not None:
            return layer_key
    return None

def module_stack_text(node: models.Node) -> str:
    """Converts module_stack metadata into searchable text."""
    return "" if node.module_stack is None else repr(node.module_stack)

def first_layer_key(value: str) -> str | None:
    """Finds the first layer/block path fragment in a string."""
    parts = value.replace("/", ".").replace("_", ".").split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "layer", "h", "blocks", "block"} and parts[index + 1].isdigit():
            return f"{part}.{parts[index + 1]}"
    return None

def element_count(shape: list[Any]) -> int | None:
    count = 1
    for dim in shape:
        known_dim = get_known_int(dim)
        if known_dim is None:
            return None
        count *= known_dim
    return count

def first_int_attr(node: models.Node, name: str, default: int) -> int:
    value = get_attr(node, name)
    if value is match_utils.MISSING:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])
    return int(value)

def cache_append_shapes_match(cache_shape: list[Any], new_data_shape: list[Any]) -> bool:
    for cache_dim, new_dim in zip(cache_shape[:-1], new_data_shape[:-1]):
        if not match_utils.values_equal(cache_dim, new_dim):
            return False
    return match_utils.values_equal(new_data_shape[-1], 1)

def short_conv_decode_weight_shape_matches(weight_shape: list[Any], hidden_dim: int, window_size: int) -> bool:
    if len(weight_shape) == 2:
        return match_utils.values_equal(weight_shape[0], hidden_dim) and match_utils.values_equal(weight_shape[1], window_size)
    if len(weight_shape) != 3:
        return False
    return (
        match_utils.values_equal(weight_shape, [1, hidden_dim, window_size])
        or match_utils.values_equal(weight_shape, [hidden_dim, 1, window_size])
    )

def slice_node_matches(node: models.Node | None, *, axis: int, start: int) -> bool:
    if node is None:
        return False
    return match_utils.values_equal(get_attr(node, "axis", "dim", "arg_1"), axis) and match_utils.values_equal(get_attr(node, "start", "arg_2"), start)

def scatter_node_matches(node: models.Node | None, *, axis: int, start: int) -> bool:
    if node is None:
        return False
    return match_utils.values_equal(get_attr(node, "dim", "axis", "arg_2"), axis) and match_utils.values_equal(get_attr(node, "start", "arg_3"), start)

def normalized_slice_axis(node: models.Node, rank: int) -> Any:
    axis = node.attrs.get("axis", node.attrs.get("dim", match_utils.MISSING))
    if axis is match_utils.MISSING:
        return match_utils.MISSING
    return normalize_dim(axis, rank)

def slice_step_is_one(node: models.Node) -> bool:
    step = node.attrs.get("step", 1)
    return step is None or match_utils.values_equal(step, 1)

def attr_matches(node: models.Node, attr: str, expected: Any, *, default: Any = match_utils.MISSING) -> bool:
    actual = node.attrs.get(attr, default)
    if actual is match_utils.MISSING:
        return False
    return match_utils.values_equal(actual, expected)

def optional_attr_matches(node: models.Node, attr: str, expected: Any) -> bool:
    actual = node.attrs.get(attr, match_utils.MISSING)
    if actual is match_utils.MISSING or actual is None:
        return True
    return match_utils.values_equal(actual, expected)

def add_dims(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return left
