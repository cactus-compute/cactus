from typing import Any
from . import models, match_utils
from ..Fusions import models as FModels






def match_node_op(synth_node: FModels.FusionNode, node: models.Node) -> bool:
    if not synth_node.ops:
        return True

    if node.node_type in synth_node.ops:
        return True

    if node.is_placeholder:
        return "placeholder" in synth_node.ops

    if node.is_output:
        return "output" in synth_node.ops

    return node.is_operation and node.target in synth_node.ops


def match_value_kind(synth_node: FModels.FusionNode, node: models.Node) -> bool:
    if not synth_node.allowed_value_kinds:
        return True

    if node.value_kind in synth_node.allowed_value_kinds:
        return True

    return (
        FModels.ValueKind.CACHE_STATE in synth_node.allowed_value_kinds
        and node.value_kind in {FModels.ValueKind.CACHE_INPUT, FModels.ValueKind.CACHE_OUTPUT}
    )


def match_attr_constraints(synth_node: FModels.FusionNode, graph: models.Graph, node: models.Node, bindings: dict[str, models.Node]) -> bool:
    for constraint in synth_node.attrs:
        if not match_utils.match_attr_constraint(constraint, graph, node, bindings):
            return False

    return True



def match_tensor_constraints(
    synth_node: FModels.FusionNode,
    graph: models.Graph,
    node: models.Node,
    bindings: dict[str, models.Node],
) -> bool:
    for constraint in synth_node.tensor_constraints:
        if not match_utils.match_tensor_constraint(constraint, graph, node, bindings):
            return False

    return True





def match_metadata_constraints(synth_node: FModels.FusionNode, node: models.Node) -> bool:
    for key, expected in synth_node.metadata.items():
        actual = match_utils.get_metadata_constraint_value(node, key)

        if actual is match_utils.MISSING or not match_utils.compare_values(actual, expected):
            return False

    return True




