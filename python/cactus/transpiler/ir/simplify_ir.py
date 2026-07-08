"""
The code in this directory is responsible for taking in the raw components lists from the converter folder and transform
it into a valid computation DAG (consisting of both operation nodes and value nodes). Topological sort will then be performed
on the DAG to generate an in-order execution list.
"""

from dataclasses import dataclass
import models, python.cactus.transpiler.ir.fusions_utils as fusions_utils
import converter.models as CVModels


@dataclass(slots=True)
class FusionResult:
    fusion: fusions_utils.FusionPattern
    matched_nodes: list[models.Node]
    external_input_nodes: list[models.Node]
    fused_node: models.Node

    @property
    def matched_node_ids(self) -> list[str]:
        return [node.layer.name for node in self.matched_nodes]

    @classmethod
    def from_match(cls, fusion: fusions_utils.FusionPattern, nodes: list[models.Node]) -> "FusionResult":
        external_input_nodes = [nodes[node_index].parents[parent_index] for node_index, parent_index in fusion.input_refs]

        fused_name = f"{fusion.target}_{nodes[0].layer.name}".replace(".", "_")
        fused_layer = CVModels.LayerRecord(
            index=nodes[0].layer.index,
            name=fused_name,
            node_type="call_function",
            target=fusion.target,
            args=[{"node": node.layer.name} for node in external_input_nodes],
            kwargs={},
            users=nodes[0].layer.users,
            tensor_output_meta=nodes[0].layer.tensor_output_meta,
            module_stack=nodes[0].layer.module_stack,
        )

        return cls(
            fusion=fusion,
            matched_nodes=nodes,
            external_input_nodes=external_input_nodes,
            fused_node=models.Node(
                layer=fused_layer,
                normalized_attrs={},
                parents=external_input_nodes,
                children=nodes[0].children,
            ),
        )


def try_match_from_node(node: models.Node, graph: models.Graph, max_depth: int = 10) -> FusionResult | None:
    if node.layer.name in graph.consumed_ids:
        return None

    candidate_paths: list[list[models.Node]] = []

    def collect_paths(path: list[models.Node]) -> None:
        candidate_paths.append(path)

        if len(path) >= max_depth:
            return

        for parent in path[-1].parents:
            if parent.layer.name in graph.consumed_ids:
                continue

            if any(existing.layer.name == parent.layer.name for existing in path):
                continue

            collect_paths([*path, parent])

    collect_paths([node])

    for candidate in sorted(candidate_paths, key=len, reverse=True):
        ops = "->".join(candidate_node.layer.target for candidate_node in candidate)
        if ops not in fusions_utils.OPS_MAP:
            continue

        fusion = fusions_utils.fusion_match(ops, candidate)
        if fusion is not None:
            return FusionResult.from_match(fusion, candidate)

    return None


def rev_top_sort(graph:models.Graph) -> models.Graph:
    """Option 1:
    Take the source node
    Do DFS style traversal of that source node its parents
    Max depth of dfs = 10
    Check if path potential fusion (does string exist as keys in dict)
    If fused, add the og layers to set of fused nodes
    Layers in set of fused nodes will not be proposed in fusions again
    """
    worklist = list(graph.source_nodes)
    visited = set()
    fused = []

    while worklist:
        node = worklist.pop()

        if node.layer.name in visited:
            continue
        visited.add(node.layer.name)

        if node.layer.name in graph.consumed_ids:
            continue

        match = try_match_from_node(node, graph)

        if match:
            fused.append(match)
            graph.consumed_ids.update(match.matched_node_ids)
            worklist.extend(match.external_input_nodes)
        else:
            worklist.extend(node.parents)

    return graph




def simplify(messy_ir:CVModels.LayerMap):
    graph: models.Graph = models.Graph.from_layer_map(messy_ir)
    



    

    
