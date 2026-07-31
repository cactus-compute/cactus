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
    """
    Returns a real parent node from a bound synthetic node/input position.

    `bindings` maps synthetic fusion-node names to real IR nodes. This helper
    first finds the real node bound to `node_name`, then asks match_utils for
    the parent at `parent_index`. It returns None when the synthetic node was
    not bound or the requested parent index does not exist.
    """
    node = bindings.get(node_name)

    if node is None:
        return None

    return match_utils.get_parent(node, parent_index)


def get_node_dim(node: models.Node, dim_index: int) -> Any:
    """
    Returns one tensor dimension from a real IR node's output shape.

    This reads `tensor_output_meta["shape"]`, supports negative indices in the
    Python style, and returns `match_utils.MISSING` instead of raising when the
    shape is unavailable or the dimension index is invalid. Matchers use that
    missing sentinel so they can either fail closed or honor `allow_missing`.
    """
    shape = match_utils.get_tensor_shape(node)

    if not shape:
        return match_utils.MISSING

    if dim_index < 0:
        dim_index += len(shape)

    if dim_index < 0 or dim_index >= len(shape):
        return match_utils.MISSING

    return shape[dim_index]


def walk_children(node: models.Node, max_depth: int) -> tuple[models.Node, ...]:
    """
    Returns all reachable child nodes within a bounded forward traversal.

    Extra matchers use this when they need local forward evidence, such as
    proving an LSTM gate tensor is split later. The traversal tracks object ids
    to avoid revisiting shared children in a DAG and to remain safe if graph
    rewrites produce unexpected cycles.
    """
    found: list[models.Node] = []
    seen: set[int] = set()

    def visit(current: models.Node, depth: int) -> None:
        """
        Recursively collects children while respecting the maximum depth.

        The outer `walk_children` owns the result list and visited set; this
        nested helper only handles traversal mechanics.
        """
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
    """
    Reads a node's output dtype from tensor metadata.

    The exported JSON stores tensor metadata as dictionaries. If metadata is
    missing or has some non-dict form, the function returns the shared MISSING
    sentinel so callers can decide whether to fail or allow missing metadata.
    """
    if not isinstance(node.tensor_output_meta, dict):
        return match_utils.MISSING

    return node.tensor_output_meta.get("dtype", match_utils.MISSING)


def normalize_dim(dim: Any, rank: int) -> Any:
    """
    Converts a possibly negative dimension index into a non-negative one.

    This mirrors Python indexing for known integer dimensions. Non-integer
    values are returned unchanged so symbolic or malformed values do not crash
    the matcher; they simply fail later if an exact comparison is required.
    """
    if not isinstance(dim, int):
        return dim

    return dim + rank if dim < 0 else dim


def get_attr(node: models.Node, *names: str) -> Any:
    """
    Returns the first available attr from a node.

    Some attrs are normalized with friendly names such as `split_sizes`, while
    older/generated JSON may only expose fallback names such as `arg_1`. This
    lets extra matchers accept either form without duplicating logic.
    """
    for name in names:
        value = node.attrs.get(name, match_utils.MISSING)

        if value is not match_utils.MISSING:
            return value

    return match_utils.MISSING


def get_known_int(value: Any) -> int | None:
    """
    Converts plain integer-like values into int, otherwise returns None.

    Shape metadata can contain symbolic values, so matchers use this helper
    when a semantic check is only safe for known concrete dimensions.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def get_shape_dim(node: models.Node | None, dim_index: int) -> int | None:
    """
    Returns a concrete int dimension from a node shape when available.

    The helper supports negative indexing and intentionally returns None for
    missing/symbolic dimensions so callers can decide whether to fail or skip.
    """
    if node is None:
        return None

    return get_known_int(get_node_dim(node, dim_index))


def shapes_are_broadcastable(left_shape: list[Any], right_shape: list[Any]) -> bool:
    """
    Implements standard trailing-dimension broadcasting for two shapes.

    It compares dimensions from right to left. Each pair is compatible if the
    dims are equal or either side is 1. Extra leading dimensions on the longer
    shape are allowed, matching normal tensor broadcasting rules.
    """
    for left_dim, right_dim in zip(reversed(left_shape), reversed(right_shape)):
        if match_utils.values_equal(left_dim, 1) or match_utils.values_equal(right_dim, 1):
            continue

        if not match_utils.values_equal(left_dim, right_dim):
            return False

    return True


def get_layer_key(node: models.Node) -> str | None:
    """
    Extracts a coarse layer identifier from module stack, node name, or target.

    Cached attention checks need to know whether different matched nodes belong
    to the same decoder layer. This tries the richer module stack first, then
    falls back to name/target strings, and returns keys like `layers.0` when it
    can identify a layer index.
    """
    for value in (module_stack_text(node), node.name, node.target):
        layer_key = first_layer_key(value)

        if layer_key is not None:
            return layer_key

    return None


def module_stack_text(node: models.Node) -> str:
    """
    Converts module_stack metadata into searchable text.

    Exported module stacks may be None or structured objects/lists. The matcher
    only needs a best-effort string to search for layer-like path fragments, so
    None becomes an empty string and everything else uses repr.
    """
    return "" if node.module_stack is None else repr(node.module_stack)


def first_layer_key(value: str) -> str | None:
    """
    Finds the first layer/block path fragment in a string.

    The helper normalizes `/` and `_` into `.` so names like
    `model_layers_12_attn` can be treated similarly to
    `model.layers.12.attn`. It then looks for common containers such as
    `layers`, `layer`, `blocks`, or `h` followed by a numeric index.
    """
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
