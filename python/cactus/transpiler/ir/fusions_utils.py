import models, constants
#TODO: Update with str paths and their respective fusion object
OPS_MAP: dict[str, list[models.FusionPattern]] = {}



"""###################################### FUSION UTILS!!!!!!! ######################################"""



# def len_match(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     return len(fusion.ops) == len(nodes)

# def match_ops(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     for i in range(len(fusion.ops)):
#         if(fusion.ops[i] != nodes[i].layer.target):
#             return False
#     return True


# def match_path(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     if len(fusion.path) != len(nodes) - 1:
#         return False

#     for i, parent_index in enumerate(fusion.path):
#         if parent_index < 0 or parent_index >= len(nodes[i].parents):
#             return False

#         if nodes[i].parents[parent_index].layer.name != nodes[i + 1].layer.name:
#             return False

#     return True

# def match_attrs(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     for node_index, required_attrs in fusion.required_attrs.items():
#         if node_index < 0 or node_index >= len(nodes):
#             return False

#         node_attrs = nodes[node_index].normalized_attrs
#         for attr_name, expected_value in required_attrs.items():
#             if attr_name not in node_attrs:
#                 return False

#             if node_attrs[attr_name] != expected_value:
#                 return False

#     return True


# def match_input_refs(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     for node_index, parent_index in fusion.input_refs:
#         if node_index < 0 or node_index >= len(nodes):
#             return False

#         if parent_index < 0 or parent_index >= len(nodes[node_index].parents):
#             return False

#     return True


# def match_shared_input_refs(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     for left_ref, right_ref in fusion.shared_input_refs:
#         left_node_index, left_parent_index = left_ref
#         right_node_index, right_parent_index = right_ref

#         if left_node_index < 0 or left_node_index >= len(nodes):
#             return False

#         if right_node_index < 0 or right_node_index >= len(nodes):
#             return False

#         if left_parent_index < 0 or left_parent_index >= len(nodes[left_node_index].parents):
#             return False

#         if right_parent_index < 0 or right_parent_index >= len(nodes[right_node_index].parents):
#             return False

#         left_parent = nodes[left_node_index].parents[left_parent_index]
#         right_parent = nodes[right_node_index].parents[right_parent_index]

#         if left_parent.layer.name != right_parent.layer.name:
#             return False

#     return True


# def match_no_external_internal_children(fusion: models.FusionPattern, nodes: list[models.Node]) -> bool:
#     fused_node_names = {node.layer.name for node in nodes}

#     for node in nodes[1:]:
#         for child in node.children:
#             if child.layer.name not in fused_node_names:
#                 return False

#     return True

# def match_external_inputs(fusion: models.FusionPattern, nodes: list[models.Node]):
#     proposed_fusion_node_names = {node.layer.name for node in nodes}

#     declared_inputs = {nodes[node_index].parents[parent_index].layer.name for node_index, parent_index in fusion.input_refs}

#     for node in nodes:
#         for parent in node.parents:
#             if parent.layer.name not in proposed_fusion_node_names and parent.layer.name not in declared_inputs:
#                 return False
            
#     return True



#Update this with any additional matching functions if any are added
MATCHERS = (
    len_match,
    match_attrs,
    match_input_refs,
    match_ops,
    match_path,
    match_shared_input_refs,
    match_no_external_internal_children,
    match_external_inputs,
)

def fusion_match(ops:str, nodes: list[models.Node]) -> models.FusionPattern | None:
    if ops in OPS_MAP:
        for fusion in OPS_MAP[ops]:
            if all(matcher(fusion, nodes) for matcher in MATCHERS):
                return fusion
        
        return None
    

    
"""###################################### FUSION GRAPH TRAVERSAL UTILS!!!!!!! ######################################"""
def dfs_path_gen(path:list[models.Node], graph:models.Graph, max_depth: int = constants.DFS_DEPTH):
    if len(path) >= max_depth or not path[-1].parents:
        yield path
        return
    
    added_to_path = False

    for parent in path[-1].parents:
        if parent.layer.name in graph.consumed_ids:
            continue
            
        if any(existing.layer.name == parent.layer.name for existing in path):
            continue

        added_to_path = True
        yield from dfs_path_gen([*path, parent], graph, max_depth)

    if not added_to_path:
        yield path
