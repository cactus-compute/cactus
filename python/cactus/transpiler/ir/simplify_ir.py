"""
The code in this directory is responsible for taking in the raw components lists from the converter folder and transform
it into a valid computation DAG (consisting of both operation nodes and value nodes). Topological sort will then be performed
on the DAG to generate an in-order execution list.
"""

import models, fusions_utils, fusions
import converter.models as CVModels


def try_match_from_node(node: models.Node, graph: models.Graph) -> models.FusionResult | None:
    if node.layer.name in graph.consumed_ids:
        return None
    
    best_match = None
    best_match_len = 0
    checked_candidates = set()

    for path in fusions_utils.dfs_path_gen([node], graph):
        for curr_len in range(len(path), 0, -1): 
            candidate_path = path[:curr_len]
            candidate_ids = tuple(candidate_node.layer.name for candidate_node in candidate_path)

            if candidate_ids in checked_candidates:
                continue
            checked_candidates.add(candidate_ids)

            ops_key = "->".join(candidate_node.layer.target for candidate_node in candidate_path)
            if ops_key not in fusions_utils.OPS_MAP:
                continue

            fusion = fusions_utils.fusion_match(ops_key, candidate_path)
            if fusion is None:
                continue

            if curr_len > best_match_len:
                best_match = models.FusionResult.from_match(fusion, candidate_path)
                best_match_len = curr_len

            if best_match_len == len(path):
                return best_match
    
    return best_match


def rev_top_sort(graph:models.Graph) -> models.SimplifiedGraph:
    worklist = list(graph.source_nodes)
    visited = set()
    fused_results = []

    while worklist:
        node = worklist.pop()
        node_id = node.layer.name

        if node_id in visited:
            continue
        visited.add(node_id)

        if node_id in graph.consumed_ids:
            continue

        match = try_match_from_node(node, graph)

        if match is not None:
            fused_results.append(match)
            graph.consumed_ids.update(match.matched_node_ids)
            worklist.extend(input_node for input_node in match.external_input_nodes if input_node.layer.name not in graph.consumed_ids)
            continue

        worklist.extend(parent for parent in node.parents if parent.layer.name not in graph.consumed_ids)

    simplified_graph = models.SimplifiedGraph.from_graph(graph)
    simplified_graph.fused_results = fused_results

    for match in fused_results:
        simplified_graph.name_map[match.node.layer.name] = match.node
        for node_id in match.matched_node_ids:
            simplified_graph.replacement_map[node_id] = match.node.layer.name

    for node_id, node in graph.name_map.items():
        if node_id not in graph.consumed_ids:
            simplified_graph.name_map[node_id] = node

    simplified_graph.source_nodes = [
        simplified_graph.name_map[simplified_graph.replacement_map.get(node.layer.name, node.layer.name)]
        for node in graph.source_nodes
        if simplified_graph.replacement_map.get(node.layer.name, node.layer.name) in simplified_graph.name_map
    ]

    return simplified_graph


# 


def simplify(messy_ir:CVModels.LayerMap) -> CVModels.LayerMap:
    graph: models.Graph = models.Graph.from_layer_map(messy_ir)
    simplified_graph = rev_top_sort(graph)
    return layer_map_from_simplified_graph(simplified_graph, messy_ir)




# def resolve_name(name: str, replacement_map: dict[str, str]) -> str:
#     seen = set()

#     while name in replacement_map and name not in seen:
#         seen.add(name)
#         name = replacement_map[name]

#     return name


# def rewrite_node_refs(value, replacement_map: dict[str, str]):
#     if isinstance(value, dict):
#         rewritten = {key: rewrite_node_refs(item, replacement_map) for key, item in value.items()}
#         if "node" in rewritten:
#             rewritten["node"] = resolve_name(rewritten["node"], replacement_map)
#         return rewritten

#     if isinstance(value, list):
#         return [rewrite_node_refs(item, replacement_map) for item in value]

#     return value


# def extract_node_refs(value) -> list[str]:
#     if isinstance(value, dict):
#         refs = []
#         if "node" in value:
#             refs.append(value["node"])

#         for item in value.values():
#             refs.extend(extract_node_refs(item))

#         return refs

#     if isinstance(value, list):
#         refs = []
#         for item in value:
#             refs.extend(extract_node_refs(item))
#         return refs

#     return []


# def topo_sort(graph: models.SimplifiedGraph, args_map: dict[str, object], kwargs_map: dict[str, object]) -> list[str]:
#     deps = {}
#     users = {node_id: set() for node_id in graph.name_map}

#     for node_id in graph.name_map:
#         node_refs = extract_node_refs(args_map[node_id]) + extract_node_refs(kwargs_map[node_id])
#         deps[node_id] = {
#             ref
#             for ref in node_refs
#             if ref in graph.name_map and ref != node_id
#         }

#         for dep in deps[node_id]:
#             users[dep].add(node_id)

#     ready = sorted(
#         [node_id for node_id, node_deps in deps.items() if not node_deps],
#         key=lambda node_id: graph.name_map[node_id].layer.index,
#     )
#     ordered = []

#     while ready:
#         node_id = ready.pop(0)
#         ordered.append(node_id)

#         for user_id in sorted(users[node_id], key=lambda item: graph.name_map[item].layer.index):
#             deps[user_id].remove(node_id)
#             if not deps[user_id]:
#                 ready.append(user_id)

#         ready.sort(key=lambda item: graph.name_map[item].layer.index)

#     if len(ordered) != len(graph.name_map):
#         raise ValueError("Simplified graph contains a cycle or unresolved dependency")

#     return ordered


# def layer_map_from_simplified_graph(graph: models.SimplifiedGraph, original: CVModels.LayerMap) -> CVModels.LayerMap:
#     fused_node_ids = {result.node.layer.name for result in graph.fused_results}
#     args_map = {}
#     kwargs_map = {}

#     for node_id, node in graph.name_map.items():
#         args_map[node_id] = rewrite_node_refs(node.layer.args, graph.replacement_map)

#         if node_id in fused_node_ids:
#             kwargs_map[node_id] = rewrite_node_refs(node.normalized_attrs, graph.replacement_map)
#         else:
#             kwargs_map[node_id] = rewrite_node_refs(node.layer.kwargs, graph.replacement_map)

#     ordered_node_ids = topo_sort(graph, args_map, kwargs_map)
#     users_map = {node_id: [] for node_id in graph.name_map}

#     for node_id in ordered_node_ids:
#         node_refs = extract_node_refs(args_map[node_id]) + extract_node_refs(kwargs_map[node_id])
#         for ref in node_refs:
#             if ref in users_map and node_id not in users_map[ref]:
#                 users_map[ref].append(node_id)

#     nodes = []
#     for index, node_id in enumerate(ordered_node_ids):
#         node = graph.name_map[node_id]
#         nodes.append(
#             CVModels.LayerRecord(
#                 index=index,
#                 name=node.layer.name,
#                 node_type=node.layer.node_type,
#                 target=node.layer.target,
#                 args=args_map[node_id],
#                 kwargs=kwargs_map[node_id],
#                 users=users_map[node_id],
#                 tensor_output_meta=node.layer.tensor_output_meta,
#                 module_stack=node.layer.module_stack,
#             )
#         )


#     return CVModels.LayerMap(
#         model_name=original.model_name,
#         task=original.task,
#         graph_signature=original.graph_signature,
#         range_constants=original.range_constants,
#         nodes=nodes,
#     )

    

    
