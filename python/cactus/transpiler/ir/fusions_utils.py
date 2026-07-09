





































































































# import models, nodes, fusions





# """###################################### FUSION UTILS!!!!!!! ######################################"""

# def attr_match(expected_value, actual_value, node: models.Node) -> bool:
#     if hasattr(expected_value, "matches"):
#         return expected_value.matches(actual_value, node)

#     return expected_value == actual_value

# def node_match(node:models.Node, synth_node:models.NodeSpec, bindings: dict[str, models.Node] | None = None) -> bool:
#     underlying_op = synth_node.underlying_op

#     if isinstance(underlying_op, tuple):
#         if node.underlying_op not in underlying_op:
#             return False
#     elif node.underlying_op != underlying_op:
#         return False

#     allowed_value_kinds = getattr(synth_node, "allowed_value_kinds", ())
#     if allowed_value_kinds and getattr(node, "value_kind", None) not in allowed_value_kinds:
#         return False

#     for key, expected_value in synth_node.required_attrs.items():
#         if key not in node.normalized_attrs:
#             return False

#         if not attr_match(expected_value, node.normalized_attrs[key], node):
#             return False

#     for constraint in getattr(synth_node, "tensor_constraints", ()):
#         if not constraint.matches(node, bindings):
#             return False

#     return True

# #Returns the desired parent if it is in the parents list
# def get_bound_parent(bindings: dict[str, models.Node], node_name: str, parent_index: int) -> models.Node | None:
#     if node_name not in bindings:
#         return None

#     node = bindings[node_name]
#     if parent_index < 0 or parent_index >= len(node.parents):
#         return None

#     return node.parents[parent_index]

# #Returns parent base on InputSpec object
# def get_input_parent(bindings: dict[str, models.Node], input_spec: models.InputSpec) -> models.Node | None:
#     return get_bound_parent(bindings, input_spec.node, input_spec.parent_index)

# def get_variadic_input_parents(bindings: dict[str, models.Node], variadic_input) -> list[models.Node] | None:
#     if variadic_input.node not in bindings:
#         return None

#     node = bindings[variadic_input.node]
#     start = variadic_input.start_parent_index
#     end = variadic_input.end_parent_index if variadic_input.end_parent_index is not None else len(node.parents)

#     if start < 0 or start > len(node.parents):
#         return None

#     if end < start or end > len(node.parents):
#         return None

#     parents = node.parents[start:end]

#     if len(parents) < variadic_input.min_count:
#         return None

#     if variadic_input.max_count is not None and len(parents) > variadic_input.max_count:
#         return None

#     return parents

# def match_input_constraints(parent: models.Node, input_spec, bindings: dict[str, models.Node]) -> bool:
#     allowed_value_kinds = getattr(input_spec, "allowed_value_kinds", ())
#     if allowed_value_kinds and getattr(parent, "value_kind", None) not in allowed_value_kinds:
#         return False

#     for constraint in getattr(input_spec, "tensor_constraints", ()):
#         if not constraint.matches(parent, bindings):
#             return False

#     return True

# #Match edges and returns binding
# def edges_match(node: models.Node, synth_graph: models.FusionGraph) -> dict[str, models.Node] | None:
#     bindings:dict[str,models.Node] = {}

#     if not node_match(node, nodes.NODE_MAP[synth_graph.root]):
#         return None

#     bindings[synth_graph.root] = node
    
#     for edge in synth_graph.edges:
#         parent = get_bound_parent(bindings, edge.from_node, edge.parent_index)
#         if parent is None:
#             if getattr(edge, "required", True):
#                 return None
#             continue

#         if edge.to_node not in nodes.NODE_MAP:
#             return None

#         if not node_match(parent, nodes.NODE_MAP[edge.to_node], bindings):
#             return None

#         to_output_index = getattr(edge, "to_output_index", None)
#         if to_output_index is not None:
#             tuple_source = getattr(parent, "tuple_source", None)
#             if tuple_source is None or tuple_source.output_index != to_output_index:
#                 return None

#         if edge.to_node in bindings and bindings[edge.to_node] is not parent:
#             return None

#         bindings[edge.to_node] = parent

#     return bindings

# #Ensures inputs exist within the fusion
# def match_inputs_exist(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     for input_spec in synth_graph.inputs:
#         parent = get_input_parent(bindings, input_spec)
#         if parent is None:
#             return False

#         if not match_input_constraints(parent, input_spec, bindings):
#             return False

#     for variadic_input in getattr(synth_graph, "variadic_inputs", ()):
#         parents = get_variadic_input_parents(bindings, variadic_input)
#         if parents is None:
#             return False

#         for parent in parents:
#             if not match_input_constraints(parent, variadic_input, bindings):
#                 return False

#     return True

# #Enusres all of a nodes inputs are included in the fusion
# def match_external_inputs_dec(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     internal_node_ids = {id(node) for node in bindings.values()}
#     declared_input_ids = set()

#     for input_spec in synth_graph.inputs:
#         parent = get_input_parent(bindings, input_spec)
#         if parent is None:
#             return False

#         declared_input_ids.add(id(parent))

#     for variadic_input in getattr(synth_graph, "variadic_inputs", ()):
#         parents = get_variadic_input_parents(bindings, variadic_input)
#         if parents is None:
#             return False

#         for parent in parents:
#             declared_input_ids.add(id(parent))

#     for node in bindings.values():
#         for parent in node.parents:
#             if id(parent) not in internal_node_ids and id(parent) not in declared_input_ids:
#                 return False

#     return True

# #Ensures named input roles point to the expected kind of value/tensor
# def match_input_roles(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     role_map = {}

#     for input_spec in synth_graph.inputs:
#         role = getattr(input_spec, "role", None)
#         if role is None:
#             continue

#         parent = get_input_parent(bindings, input_spec)
#         if parent is None:
#             return False

#         role_map.setdefault(role, [])
#         role_map[role].append(parent)

#     for variadic_input in getattr(synth_graph, "variadic_inputs", ()):
#         role = getattr(variadic_input, "role", None)
#         if role is None:
#             continue

#         parents = get_variadic_input_parents(bindings, variadic_input)
#         if parents is None:
#             return False

#         role_map.setdefault(role, [])
#         role_map[role].extend(parents)

#     for role_spec in getattr(synth_graph, "input_roles", ()):
#         if role_spec.input_name not in role_map:
#             return False

#         for parent in role_map[role_spec.input_name]:
#             if not match_input_constraints(parent, role_spec, bindings):
#                 return False

#     return True

# #Ensures ops with same inputs are actually pointing to same input node
# def match_shared_inputs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     for left_input, right_input in synth_graph.shared_inputs:
#         left_parent = get_input_parent(bindings, left_input)
#         right_parent = get_input_parent(bindings, right_input)
#         if left_parent is None or right_parent is None:
#             return False

#         if left_parent is not right_parent:
#             return False

#     return True

# #Ensures that a node within fusion is not called by a differnt node not in the fusion
# def match_no_external_internal_children(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     if synth_graph.root not in bindings:
#         return False

#     root_node = bindings[synth_graph.root]
#     internal_node_ids = {id(node) for node in bindings.values()}
#     allow_root_external_children = getattr(synth_graph, "allow_root_external_children", True)

#     for node in bindings.values():
#         if node is root_node and allow_root_external_children:
#             continue

#         for child in node.children:
#             if id(child) not in internal_node_ids:
#                 return False

#     return True

# def is_attr_ref(value) -> bool:
#     return hasattr(value, "node") and hasattr(value, "attr")

# def match_output_attrs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     for attr_value in synth_graph.output_attrs.values():
#         if not is_attr_ref(attr_value):
#             continue

#         if attr_value.node not in bindings:
#             if getattr(attr_value, "required", True):
#                 return False
#             continue

#         node = bindings[attr_value.node]
#         if attr_value.attr not in node.normalized_attrs:
#             if getattr(attr_value, "required", True):
#                 return False
#             continue

#     return True

# def match_optional_nodes(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     for optional_node in getattr(synth_graph, "optional_nodes", ()):
#         if optional_node.node not in bindings:
#             continue

#         node = bindings[optional_node.node]
#         if optional_node.bypass_parent_index < 0 or optional_node.bypass_parent_index >= len(node.parents):
#             return False

#     return True

# def match_tuple_outputs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     for tuple_output in getattr(synth_graph, "tuple_outputs", ()):
#         if tuple_output.node not in bindings or tuple_output.source_node not in bindings:
#             return False

#         node = bindings[tuple_output.node]
#         source_node = bindings[tuple_output.source_node]
#         tuple_source = getattr(node, "tuple_source", None)

#         if tuple_source is None:
#             return False

#         if tuple_source.source_node_name != source_node.name:
#             return False

#         if tuple_source.output_index != tuple_output.output_index:
#             return False

#     return True

# def match_repeated_subgraphs(bindings: dict[str, models.Node], synth_graph: models.FusionGraph) -> bool:
#     repeated_bindings = getattr(synth_graph, "metadata", {}).get("repeated_bindings", {})

#     for repeated in getattr(synth_graph, "repeated_subgraphs", ()):
#         if repeated.anchor_node is not None and repeated.anchor_node not in bindings:
#             return False

#         matches = repeated_bindings.get(repeated.name, [])

#         if len(matches) < repeated.min_count:
#             return False

#         if repeated.max_count is not None and len(matches) > repeated.max_count:
#             return False

#     return True


# #Update this with any additional matching functions if any are added
# MATCHERS = (
#     match_inputs_exist,
#     match_external_inputs_dec,
#     match_input_roles,
#     match_shared_inputs,
#     match_no_external_internal_children,
#     match_output_attrs,
#     match_optional_nodes,
#     match_tuple_outputs,
#     match_repeated_subgraphs,
# )

# def fusion_match(node: models.Node) -> models.FusionResult | None:

#     for fusion in fusions.ROOT_TARGET_MAP.get(node.underlying_op, ()):
#         bindings = edges_match(node, fusion)
#         if bindings != None and all(matcher(bindings, fusion) for matcher in MATCHERS):
#             return models.FusionResult.from_fusion(fusion, bindings)

#     return None
