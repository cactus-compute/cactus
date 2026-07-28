from typing import Any

from . import models
from ..Fusions import models as FModels


# MISSING = object()

def shape_matches(actual_shape: list[Any], expected_shape: tuple[Any, ...]) -> bool:
    if len(actual_shape) != len(expected_shape):
        return False

    return all(expected is None or values_equal(actual, expected) for actual, expected in zip(actual_shape, expected_shape))


def get_dim(shape: list[Any], index: int) -> Any:
    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        return None

    return shape[index]


def normalize_dtype(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def normalize_gelu_approximation(value: Any) -> Any:
    if value == "none":
        return "erf"

    return value


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, tuple):
        actual = list(actual)

    if isinstance(expected, tuple):
        expected = list(expected)

    if isinstance(actual, list) and len(actual) == 1 and not isinstance(expected, (list, tuple)):
        actual = actual[0]

    if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, (list, tuple)):
        expected = expected[0]

    return actual == expected



def get_metadata_constraint_value(node: models.Node, key: str) -> Any:
    if key == "approximation" and node.target == "aten.gelu.default":
        return normalize_gelu_approximation(node.attrs.get("approximate", "erf"))

    return node.attrs.get(key, None)


def compare_values(actual: Any, expected: Any, comparator: str = "eq") -> bool:
    if comparator == "eq":
        return values_equal(actual, expected)

    if comparator == "ne":
        return not values_equal(actual, expected)

    if comparator == "in":
        return any(values_equal(actual, candidate) for candidate in expected)

    raise ValueError(f"Unknown attr comparator: {comparator}")


def match_attr_constraint(constraint: FModels.AttrConstraint, node: models.Node, bindings: dict[str, models.Node]) -> bool:
    actual = node.attrs.get(constraint.name, None)
    if actual is None:
        return not constraint.required

    expected = constraint.value

    if constraint.source_node is not None and constraint.source_attr is not None:
        source_node = bindings.get(constraint.source_node)

        if source_node is None:
            return False

        expected = source_node.attrs.get(constraint.source_attr, None)

        if expected is None:
            return not constraint.required

    return compare_values(actual, expected, constraint.comparator)


def match_tensor_constraint(constraint: FModels.TensorConstraint, node: models.Node, bindings: dict[str, models.Node]) -> bool:
    if node.tensor_output_meta is None:
        return False

    shape = node.tensor_output_meta.get("shape", ())
    dtype = node.tensor_output_meta.get("dtype")
    rank = len(shape)

    if constraint.rank is not None and rank != constraint.rank:
        return False

    if constraint.min_rank is not None and rank < constraint.min_rank:
        return False

    if constraint.max_rank is not None and rank > constraint.max_rank:
        return False

    if constraint.dtype is not None and normalize_dtype(dtype) != normalize_dtype(constraint.dtype):
        return False

    if constraint.shape and not shape_matches(shape, constraint.shape):
        return False

    for dim_index, expected in constraint.dim_equals:
        if not values_equal(get_dim(shape, dim_index), expected):
            return False

    for dim_index, other_node_name, other_dim_index in constraint.same_dim_as:
        other_node = bindings.get(other_node_name)

        if other_node is None or other_node.tensor_output_meta is None:
            return False

        if not values_equal(get_dim(shape, dim_index), get_dim(other_node.tensor_output_meta.get("shape", ()), other_dim_index)):
            return False

    for key, expected in constraint.metadata.items():
        if not values_equal(node.tensor_output_meta.get(key, None), expected):
            return False

    return True



