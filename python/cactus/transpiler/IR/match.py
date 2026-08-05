from . import extra_matchers, models, match_utils
from ..Fusions import models as FModels

ANY = ()

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

def match_subgraph(node: models.Node, graph: models.Graph, subgraph: FModels.RepeatedSubgraph) -> bool:
    match_count = 0

    for candidate in graph.nodes:
        if match_fusion(candidate, graph, subgraph.graph):
            match_count += 1

            if subgraph.max_count is not None and match_count > subgraph.max_count:
                return False

    if match_count < subgraph.min_count:
        return False

    return True

def match_repeated_subgraphs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for subgraph in fusion.repeated_subgraphs:
        if subgraph.anchor_node is not None and subgraph.anchor_node not in bindings:
            return False

        if not match_subgraph(node, graph, subgraph):
            return False

    return True

def match_edges(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for edge in fusion.edges:
        if not match_utils.edge_matches(edge, bindings):
            return False

    return True

def match_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
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

def match_shared_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for left_ref, right_ref in fusion.shared_inputs:
        left_parent = match_utils.get_node_ref_parent(left_ref, bindings)
        right_parent = match_utils.get_node_ref_parent(right_ref, bindings)

        if left_parent is None or right_parent is None:
            return False

        if left_parent is not right_parent:
            return False

    return True

def match_constraints(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    return extra_matchers.match_extra_constraints(node, graph, fusion, bindings)

def match_cache_inputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for cache_input in fusion.cache_inputs:
        cache_node = match_utils.get_node_ref_parent(cache_input.source, bindings)

        if cache_node is None:
            if cache_input.optional:
                continue

            return False

        if not match_utils.match_cache_boundary_value(cache_node, cache_input):
            return False

    return True

def match_cache_mutations(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    input_roles = {cache_input.role for cache_input in fusion.cache_inputs}
    output_roles = {cache_output.role for cache_output in fusion.cache_outputs}

    for mutation in fusion.cache_mutations:
        if mutation.required and not set(mutation.read_roles).issubset(input_roles):
            return False

        if mutation.required and not set(mutation.write_roles).issubset(output_roles):
            return False

    return True

def match_cache_outputs(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
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

def match_metadata(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    return True

def match_output(node: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    for output in fusion.outputs:
        output_node = bindings.get(output.node)

        if output_node is None:
            return False

        for constraint in output.tensor_constraints:
            if not match_utils.match_tensor_constraint(constraint, output_node, bindings):
                return False

    return match_utils.external_children_are_valid(bindings, fusion)

FUSION_MATCHERS = [
    match_edges,
    match_inputs,
    match_shared_inputs,
    match_repeated_subgraphs,
    match_constraints,
    match_cache_inputs,
    match_cache_mutations,
    match_cache_outputs,
    match_metadata,
    match_output,
]

def match_fusion(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph) -> bool:
    bindings = match_utils.bind_fusion_graph(source, fusion, match_nodes)

    if bindings is None:
        return False

    return match_fusion_bindings(source, graph, fusion, bindings)

def match_fusion_bindings(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    return all(matcher(source, graph, fusion, bindings) for matcher in FUSION_MATCHERS)

def match_definition_graph(source: models.Node, graph: models.Graph, fusion: FModels.FusionDefinition, bindings: dict[str, models.Node], inference_mode: str | None = None, input_modalities: tuple[str, ...] = ANY, fusion_fields: tuple[str, ...] = ANY) -> bool:
    return match_fusion_bindings(source, graph, fusion.graph, bindings)

def match_definition_metadata(source: models.Node, graph: models.Graph, fusion: FModels.FusionDefinition, bindings: dict[str, models.Node], inference_mode: str | None = None, input_modalities: tuple[str, ...] = ANY, fusion_fields: tuple[str, ...] = ANY) -> bool:
    return match_utils.match_definition_metadata(fusion, bindings)

def match_definition_inference_mode(source: models.Node, graph: models.Graph, fusion: FModels.FusionDefinition, bindings: dict[str, models.Node], inference_mode: str | None = None, input_modalities: tuple[str, ...] = ANY, fusion_fields: tuple[str, ...] = ANY) -> bool:
    if not fusion.supported_inference_modes or inference_mode is None:
        return True

    return inference_mode in fusion.supported_inference_modes

def match_definition_modalities(source: models.Node, graph: models.Graph, fusion: FModels.FusionDefinition, bindings: dict[str, models.Node], inference_mode: str | None = None, input_modalities: tuple[str, ...] = ANY, fusion_fields: tuple[str, ...] = ANY) -> bool:
    if not fusion.supported_modalities or not input_modalities:
        return True

    return set(fusion.supported_modalities).issubset(input_modalities)

def match_definition_fusion_fields(source: models.Node, graph: models.Graph, fusion: FModels.FusionDefinition, bindings: dict[str, models.Node], inference_mode: str | None = None, input_modalities: tuple[str, ...] = ANY, fusion_fields: tuple[str, ...] = ANY) -> bool:
    if not fusion.fusion_fields or not fusion_fields:
        return True

    return not set(fusion.fusion_fields).isdisjoint(fusion_fields)

FUSION_DEFINITION_MATCHERS = (
    match_definition_graph,
    match_definition_metadata,
    match_definition_inference_mode,
    match_definition_modalities,
    match_definition_fusion_fields,
)
