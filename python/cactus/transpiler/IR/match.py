from . import models, match_utils
from ..Fusions import models as FModels


####################################### Node Matching Logic!!!!! #######################################
def match_node_op(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    if not synth_node.ops:
        return True

    if node.node_type in synth_node.ops:
        return True

    if node.is_placeholder:
        return "placeholder" in synth_node.ops

    if node.is_output:
        return "output" in synth_node.ops

    return node.is_operation and node.target in synth_node.ops


def match_value_kind(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    if not synth_node.allowed_value_kinds:
        return True

    return node.value_kind in synth_node.allowed_value_kinds


def match_attr_constraints(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    for constraint in synth_node.attrs:
        if not match_utils.match_attr_constraint(constraint, node, bindings or {}):
            return False

    return True


def match_tensor_constraints(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    for constraint in synth_node.tensor_constraints:
        if not match_utils.match_tensor_constraint(constraint, node, bindings or {}):
            return False

    return True


def match_metadata_constraints(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    for key, expected in synth_node.metadata.items():
        actual = match_utils.get_metadata_constraint_value(node, key)

        if actual is match_utils.MISSING or not match_utils.compare_values(actual, expected):
            return False

    return True


NODE_MATCHERS = [
    match_node_op,
    match_value_kind,
    match_attr_constraints,
    match_tensor_constraints,
    match_metadata_constraints,
]


def match_nodes(node: models.Node, synth_node: FModels.FusionNode, bindings: dict[str, models.Node] | None = None) -> bool:
    return all(matcher(node, synth_node, bindings or {}) for matcher in NODE_MATCHERS)


####################################### Other Matching Logic!!!!! #######################################
def match_subgraph(node: models.Node, graph: models.Graph, subgraph: FModels.RepeatedSubgraph) -> bool:
    match_count = 0

    for candidate in graph.nodes:
        if match_fusion(candidate, graph, subgraph.graph):
            match_count += 1

    if match_count < subgraph.min_count or (match_count is not None and match_count > subgraph.max_count):
        return False

    return True


def match_repeated_subgraphs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for subgraph in fusion.repeated_subgraphs:
        if subgraph.anchor_node is not None and subgraph.anchor_node not in bindings:
            return False

        if not match_subgraph(node, graph, subgraph):
            return False

    return True


def match_source_node(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    return match_utils.bind_fusion_graph(node, fusion, match_nodes) is not None


def match_edges(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for edge in fusion.edges:
        if not match_utils.edge_matches(edge, bindings):
            return False

    return True


def match_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    declared_inputs = set()

    for input_spec in fusion.inputs:
        input_nodes = match_utils.get_fusion_input_nodes(input_spec, bindings)

        if not input_nodes and not input_spec.optional:
            return False

        for input_node in input_nodes:
            if not match_utils.match_boundary_value(input_node, input_spec.allowed_value_kinds, input_spec.tensor_constraints):
                return False

            declared_inputs.add(id(input_node))

    for cache_input in fusion.cache_inputs:
        cache_node = match_utils.get_node_ref_parent(cache_input.source, bindings)

        if cache_node is not None:
            declared_inputs.add(id(cache_node))

    return match_utils.all_external_parents_declared(bindings, declared_inputs)


def match_shared_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for left_ref, right_ref in fusion.shared_inputs:
        left_parent = match_utils.get_node_ref_parent(left_ref, bindings)
        right_parent = match_utils.get_node_ref_parent(right_ref, bindings)

        if left_parent is None or right_parent is None:
            return False

        if left_parent is not right_parent:
            return False

    return True


def match_constrains(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    return True


def match_cache_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for cache_input in fusion.cache_inputs:
        cache_node = match_utils.get_node_ref_parent(cache_input.source, bindings)

        if cache_node is None:
            if cache_input.optional:
                continue

            return False

        if not match_utils.match_boundary_value(cache_node, (), cache_input.tensor_constraints):
            return False

    return True


def match_cache_mutations(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    input_roles = {cache_input.role for cache_input in fusion.cache_inputs}
    output_roles = {cache_output.role for cache_output in fusion.cache_outputs}

    for mutation in fusion.cache_mutations:
        if mutation.required and not set(mutation.read_roles).issubset(input_roles):
            return False

        if mutation.required and not set(mutation.write_roles).issubset(output_roles):
            return False

    return True


def match_cache_outputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for cache_output in fusion.cache_outputs:
        output_node = bindings.get(cache_output.node)

        if output_node is None:
            if cache_output.optional:
                continue

            return False

        for constraint in cache_output.tensor_constraints:
            if not match_utils.match_tensor_constraint(constraint, output_node, bindings):
                return False

    return True


def match_metadata(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    return True


def match_output(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(node, fusion, match_nodes)

    if bindings is None:
        return False

    for output in fusion.outputs:
        output_node = bindings.get(output.node)

        if output_node is None:
            return False

        for constraint in output.tensor_constraints:
            if not match_utils.match_tensor_constraint(constraint, output_node, bindings):
                return False

    return match_utils.external_children_are_valid(bindings, fusion)


FUSION_MATCHERS = [
    match_source_node,
    match_edges,
    match_inputs,
    match_shared_inputs,
    match_repeated_subgraphs,
    match_constrains,
    match_cache_inputs,
    match_cache_mutations,
    match_cache_outputs,
    match_metadata,
    match_output,
]


def match_fusion(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    return all(matcher(source, graph, fusion) for matcher in FUSION_MATCHERS)


####################################### Top. Sort + Graph updating!!!!! #######################################
