from typing import Any

from . import models
from ..Fusions import models as FModels


MISSING = object()

####################################### Node Matching Utils!!!!! #######################################

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


####################################### Other matching utils!!!!! #######################################

def bind_fusion_graph(
    source: models.Node,
    fusion: FModels.FusionGraph,
    node_matcher: Any,
) -> dict[str, models.Node] | None:
    if fusion.root not in fusion.nodes:
        return None

    bindings: dict[str, models.Node] = {}

    if not bind_node(fusion.root, source, fusion, bindings, node_matcher):
        return None

    changed = True

    while changed:
        changed = False

        for edge in fusion.edges:
            if edge.dest not in bindings:
                continue

            parent = get_edge_parent(edge, bindings)

            if parent is None:
                return None

            if edge.source not in bindings:
                if not bind_node(edge.source, parent, fusion, bindings, node_matcher):
                    return None

                changed = True
                continue

            if bindings[edge.source] is not parent:
                return None

    for synth_node_name in fusion.nodes:
        if synth_node_name not in bindings:
            return None

    return bindings


def bind_node(
    synth_node_name: str,
    node: models.Node,
    fusion: FModels.FusionGraph,
    bindings: dict[str, models.Node],
    node_matcher: Any,
) -> bool:
    synth_node = fusion.nodes.get(synth_node_name)

    if synth_node is None:
        return False

    if synth_node_name in bindings:
        return bindings[synth_node_name] is node

    if any(bound_node is node for bound_node in bindings.values()):
        return False

    if not node_matcher(node, synth_node, bindings):
        return False

    bindings[synth_node_name] = node
    return True


def edge_matches(edge: FModels.FusionEdge, bindings: dict[str, models.Node]) -> bool:
    source = bindings.get(edge.source)
    dest = bindings.get(edge.dest)

    if source is None or dest is None:
        return False

    parent = get_edge_parent(edge, bindings)
    return parent is source


def get_edge_parent(edge: FModels.FusionEdge, bindings: dict[str, models.Node]) -> models.Node | None:
    dest = bindings.get(edge.dest)

    if dest is None:
        return None

    if edge.dest_input_index is None:
        if edge.source not in bindings:
            return None

        source = bindings[edge.source]
        return source if any(parent is source for parent in dest.parents) else None

    return get_parent(dest, edge.dest_input_index)


def get_node_ref_parent(ref: FModels.NodeRef, bindings: dict[str, models.Node]) -> models.Node | None:
    node = bindings.get(ref.node)

    if node is None or ref.parent_index is None:
        return None

    return get_parent(node, ref.parent_index)


def get_parent(node: models.Node, parent_index: int) -> models.Node | None:
    if parent_index < 0 or parent_index >= len(node.parents):
        return None

    return node.parents[parent_index]


def get_fusion_input_nodes(input_spec: FModels.FusionInput, bindings: dict[str, models.Node]) -> tuple[models.Node, ...]:
    source_node = bindings.get(input_spec.source.node)

    if source_node is None or input_spec.source.parent_index is None:
        return ()

    if not input_spec.variadic:
        parent = get_parent(source_node, input_spec.source.parent_index)
        return (parent,) if parent is not None else ()

    end_parent_index = input_spec.end_parent_index

    if end_parent_index is None:
        end_parent_index = len(source_node.parents)

    parents = source_node.parents[input_spec.source.parent_index:end_parent_index]

    if len(parents) < input_spec.min_count:
        return ()

    if input_spec.max_count is not None and len(parents) > input_spec.max_count:
        return ()

    return parents


def match_boundary_value(
    node: models.Node,
    allowed_value_kinds: tuple[str, ...],
    tensor_constraints: tuple[FModels.TensorConstraint, ...],
) -> bool:
    if allowed_value_kinds and node.value_kind not in allowed_value_kinds:
        return False

    return all(match_tensor_constraint(constraint, node, {}) for constraint in tensor_constraints)


def all_external_parents_declared(bindings: dict[str, models.Node], declared_input_ids: set[int]) -> bool:
    internal_ids = {id(node) for node in bindings.values()}

    for node in bindings.values():
        for parent in node.parents:
            if id(parent) in internal_ids:
                continue

            if id(parent) not in declared_input_ids:
                return False

    return True


def external_children_are_valid(bindings: dict[str, models.Node], fusion: FModels.FusionGraph) -> bool:
    internal_ids = {id(node) for node in bindings.values()}
    exposed_output_names = {output.node for output in fusion.outputs}

    if not fusion.allow_root_external_children:
        exposed_output_names.discard(fusion.root)

    for synth_node_name, node in bindings.items():
        for child in node.children:
            if id(child) in internal_ids:
                continue

            if synth_node_name not in exposed_output_names:
                return False

    return True
