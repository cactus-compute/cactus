"""
The code in this directory is responsible for taking in the raw components lists from the converter folder and transform
it into a valid computation DAG (consisting of both operation nodes and value nodes). Topological sort will then be performed
on the DAG to generate an in-order execution list.
"""

import models, fusions_utils
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




def simplify(messy_ir:CVModels.LayerMap):
    graph: models.Graph = models.Graph.from_layer_map(messy_ir)
    



    

    
