from typing import Any
from . import models
from ..Fusions import models as FModels

MISSING = object()
WILDCARD_DIM_VALUES = (None, "*", "?", -1)
METADATA_ATTR_ALIASES = {
    "approximation": ("approximation", "approximate"),
}

def get_tensor_meta_value(tensor_meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(tensor_meta, dict):
        return tensor_meta.get(key, default)

    return getattr(tensor_meta, key, default)

def resolve_bound_node(
    node_name: str,
    graph: models.Graph,
    bindings: dict[str, models.Node],
) -> models.Node | None:
    if node_name in bindings:
        return bindings[node_name]

    return graph.nodes_map.get(node_name)



def get_metadata_constraint_value(node: models.Node, key: str) -> Any:
    for attr_name in METADATA_ATTR_ALIASES.get(key, (key,)):
        if attr_name in node.attrs:
            return node.attrs[attr_name]

    if key == "approximation" and node.target == "aten.gelu.default":
        return "erf"

    return MISSING


def match_shape(actual_shape: Any, expected_shape: tuple[Any, ...]) -> bool:
    if not isinstance(actual_shape, (list, tuple)):
        return False

    if len(actual_shape) != len(expected_shape):
        return False

    return all(compare_dimensions(actual, expected) for actual, expected in zip(actual_shape, expected_shape))


def get_dim(shape: Any, index: int) -> Any:
    if not isinstance(shape, (list, tuple)):
        return MISSING

    normalized_index = index if index >= 0 else len(shape) + index

    if normalized_index < 0 or normalized_index >= len(shape):
        return MISSING

    return shape[normalized_index]


def compare_dimensions(actual: Any, expected: Any) -> bool:
    if expected in WILDCARD_DIM_VALUES:
        return True

    return compare_values(actual, expected)


def compare_dtypes(actual: Any, expected: Any) -> bool:
    return normalize_dtype(actual) == normalize_dtype(expected)


def normalize_dtype(value: Any) -> str:
    return str(value).lower().replace("torch.", "").strip()


def compare_values(actual: Any, expected: Any, comparator: str = "eq") -> bool:
    comparator = comparator.lower()

    if comparator in {"eq", "=="}:
        return values_equal(actual, expected)

    if comparator in {"ne", "!="}:
        return not values_equal(actual, expected)

    if comparator == "in":
        return value_in(actual, expected)

    if comparator == "not_in":
        return not value_in(actual, expected)

    if comparator in {"gt", ">"}:
        return actual > expected

    if comparator in {"gte", "ge", ">="}:
        return actual >= expected

    if comparator in {"lt", "<"}:
        return actual < expected

    if comparator in {"lte", "le", "<="}:
        return actual <= expected

    if comparator == "contains":
        return expected in actual

    if comparator == "truthy":
        return bool(actual)

    if comparator == "falsy":
        return not bool(actual)

    if comparator == "is_none":
        return actual is None

    if comparator == "not_none":
        return actual is not None

    raise ValueError(f"Unknown attr comparator: {comparator}")


def value_in(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(values_equal(actual, item) for item in expected)

    return values_equal(actual, expected)


def values_equal(actual: Any, expected: Any) -> bool:
    actual = unwrap_singleton(actual, expected)
    expected = unwrap_singleton(expected, actual)

    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False

        return all(values_equal(actual[key], expected[key]) for key in actual)

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False

        return all(values_equal(left, right) for left, right in zip(actual, expected))

    if expected == "erf" and actual == "none":
        return True

    return actual == expected


def unwrap_singleton(value: Any, other: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1 and not isinstance(other, (list, tuple)):
        return value[0]

    return value

def match_attr_constraint(
    constraint: FModels.AttrConstraint,
    graph: models.Graph,
    node: models.Node,
    bindings: dict[str, models.Node],
) -> bool:
    actual = node.attrs.get(constraint.name, MISSING)

    if actual is MISSING:
        return not constraint.required

    expected = constraint.value

    if constraint.source_node is not None and constraint.source_attr is not None:
        source_node = resolve_bound_node(constraint.source_node, graph, bindings)

        if source_node is None:
            return True

        expected = source_node.attrs.get(constraint.source_attr, MISSING)

        if expected is MISSING:
            return not constraint.required

    return compare_values(actual, expected, constraint.comparator)

def match_tensor_constraint(
    constraint: FModels.TensorConstraint,
    graph: models.Graph,
    node: models.Node,
    bindings: dict[str, models.Node],
) -> bool:
    tensor_meta = node.tensor_output_meta

    if tensor_meta is None:
        return False

    shape = get_tensor_meta_value(tensor_meta, "shape", ())
    dtype = get_tensor_meta_value(tensor_meta, "dtype", None)
    rank = len(shape) if isinstance(shape, (list, tuple)) else None

    if constraint.rank is not None and rank != constraint.rank:
        return False

    if constraint.min_rank is not None and (rank is None or rank < constraint.min_rank):
        return False

    if constraint.max_rank is not None and (rank is None or rank > constraint.max_rank):
        return False

    if constraint.dtype is not None and not compare_dtypes(dtype, constraint.dtype):
        return False

    if constraint.shape and not match_shape(shape, constraint.shape):
        return False

    for dim_index, expected in constraint.dim_equals:
        actual = get_dim(shape, dim_index)

        if actual is MISSING or not compare_dimensions(actual, expected):
            return False

    for dim_index, other_node_name, other_dim_index in constraint.same_dim_as:
        actual = get_dim(shape, dim_index)
        other_node = resolve_bound_node(other_node_name, graph, bindings)

        if other_node is None:
            continue

        other_shape = get_tensor_meta_value(other_node.tensor_output_meta, "shape", ())
        other_dim = get_dim(other_shape, other_dim_index)

        if actual is MISSING or other_dim is MISSING or not compare_dimensions(actual, other_dim):
            return False

    for key, expected in constraint.metadata.items():
        actual = get_tensor_meta_value(tensor_meta, key, MISSING)

        if actual is MISSING or not compare_values(actual, expected):
            return False

    return True
