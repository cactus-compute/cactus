import models, nodes, fusions





"""###################################### FUSION UTILS!!!!!!! ######################################"""

def node_match(node:models.Node, synth_node:models.NodeSpec) -> bool:
    if node.underlying_op != synth_node.underlying_op:
        return False

    for key, expected_value in synth_node.required_attrs.items():
        if key not in node.normalized_attrs:
            return False

        if node.normalized_attrs[key] != expected_value:
            return False

    return True

def get_bound_parent(bindings: dict[str, models.Node], node_name: str, parent_index: int) -> models.Node | None:
    if node_name not in bindings:
        return None

    node = bindings[node_name]
    if parent_index < 0 or parent_index >= len(node.parents):
        return None

    return node.parents[parent_index]

def get_input_parent(bindings: dict[str, models.Node], input_spec: models.InputSpec) -> models.Node | None:
    return get_bound_parent(bindings, input_spec.node, input_spec.parent_index)

#Match edges
def edges_match(node: models.Node, synth_graph: models.FusionGraph) -> dict[str, models.Node] | None:
    bindings:dict[str,models.Node] = {}

    if not node_match(node, nodes.NODE_MAP[synth_graph.root]):
        return None

    bindings[synth_graph.root] = node
    
    for edge in synth_graph.edges:
        parent = get_bound_parent(bindings, edge.from_node, edge.parent_index)
        if parent is None:
            return None

        if edge.to_node not in nodes.NODE_MAP:
            return None

        if not node_match(parent, nodes.NODE_MAP[edge.to_node]):
            return None

        if edge.to_node in bindings and bindings[edge.to_node] is not parent:
            return None

        bindings[edge.to_node] = parent

    return bindings


def match_inputs_exist(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
    for input_spec in synth_graph.inputs:
        if get_input_parent(bindings, input_spec) is None:
            return False

    return True

def match_external_inputs_dec(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
    internal_node_ids = {id(node) for node in bindings.values()}
    declared_input_ids = set()

    for input_spec in synth_graph.inputs:
        parent = get_input_parent(bindings, input_spec)
        if parent is None:
            return False

        declared_input_ids.add(id(parent))

    for node in bindings.values():
        for parent in node.parents:
            if id(parent) not in internal_node_ids and id(parent) not in declared_input_ids:
                return False

    return True

def match_shared_inputs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
    for left_input, right_input in synth_graph.shared_inputs:
        left_parent = get_input_parent(bindings, left_input)
        right_parent = get_input_parent(bindings, right_input)
        if left_parent is None or right_parent is None:
            return False

        if left_parent is not right_parent:
            return False

    return True

def match_no_external_internal_children(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
    if synth_graph.root not in bindings:
        return False

    root_node = bindings[synth_graph.root]
    internal_node_ids = {id(node) for node in bindings.values()}

    for node in bindings.values():
        if node is root_node:
            continue

        for child in node.children:
            if id(child) not in internal_node_ids:
                return False

    return True

def match_output_attrs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
    for attr_value in synth_graph.output_attrs.values():
        if not isinstance(attr_value, models.AttrRef):
            continue

        if attr_value.node not in bindings:
            return False

        node = bindings[attr_value.node]
        if attr_value.attr not in node.normalized_attrs:
            return False

    return True


#Update this with any additional matching functions if any are added
MATCHERS = (
    match_inputs_exist,
    match_external_inputs_dec,
    match_shared_inputs,
    match_no_external_internal_children,
    match_output_attrs,
)

def fusion_match(node: models.Node) -> models.FusionGraph | None:

    for fusion in fusions.ROOT_TARGET_MAP.get(node.underlying_op, ()):
        bindings = edges_match(node, fusion)
        if bindings != None and all(matcher(bindings, fusion) for matcher in MATCHERS):
            #Will change this to return FusionResult object which will have a list of all nodes being fused
            return fusion
