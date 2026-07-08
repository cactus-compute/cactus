from dataclasses import dataclass
from typing import Any
from collections import deque
import converter.models as CVModels
import constants

@dataclass(slots=True)
class Node:
    layer: CVModels.LayerRecord
    normalized_attrs: dict[str, Any]
    parents: list["Node"]
    children: list["Node"]


@dataclass(slots=True)
class Graph:
    name_map:dict[str, Node]
    source_nodes: deque[Node]
    consumed_ids: set[str]

    @classmethod
    def from_layer_map(cls, map:CVModels.LayerMap):
        return generate_graph(map)
    
@dataclass(slots=True)
class FusionPattern:
    #Input requirements
    target:str
    ops:tuple[str, ...]
    path:tuple[int, ...]
    required_attrs:dict[int, dict[str, Any]]
    input_refs:tuple[tuple[int, int],...]
    shared_input_refs:tuple[tuple[tuple[int, int], tuple[int, int]], ...]

    #Output requirements
    output_attrs: dict[str, Any]

    
@dataclass(slots=True)
class FusionResult:
    fusion: FusionPattern
    matched_nodes: list[Node]
    external_input_nodes: list[Node]
    node: Node

    @property
    def matched_node_ids(self) -> list[str]:
        return [node.layer.name for node in self.matched_nodes]

    @classmethod
    def from_match(cls, fusion: FusionPattern, nodes: list[Node]) -> "FusionResult":
        return generate_fusion(fusion, nodes)
        
    

"""###################################### MODEL UTILS!!!!!!! ######################################"""

def generate_fusion(fusion: FusionPattern, nodes: list[Node]) -> FusionResult:
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

    fused_node = Node(
        layer=fused_layer,
        normalized_attrs= fusion.output_attrs,
        parents=external_input_nodes, 
        children=nodes[0].children
    )


    return FusionResult(
        fusion=fusion,
        matched_nodes=nodes,
        external_input_nodes=external_input_nodes,
        node=fused_node
    )
    

def extract_attrs(layer_: CVModels.LayerRecord) -> dict[str, Any]:
    
    def contains_node_ref(value: Any) -> bool:
        if isinstance(value, dict):
            if "node" in value:
                return True
            return any(contains_node_ref(item) for item in value.values())

        if isinstance(value, list):
            return any(contains_node_ref(item) for item in value)

        return False

    attrs = {}

    if isinstance(layer_.kwargs, dict):
        attrs.update(layer_.kwargs)

    attr_names = constants.LAYER_ATTRS_MAP.get(layer_.target, [])
    positional_attrs = [item for item in layer_.args if not contains_node_ref(item)]

    for i, attr_name in enumerate(attr_names):
        if i < len(positional_attrs) and attr_name not in attrs:
            attrs[attr_name] = positional_attrs[i]

    return attrs

def generate_graph(map: CVModels.LayerMap):
    temp_map:dict[str, Node] = {}
    temp_source = deque()

    for layer_ in map.nodes:
        temp = Node(layer=layer_, normalized_attrs=extract_attrs(layer_), parents=[], children=[])
        temp_map[layer_.name] = temp
        temp_source.append(temp) if layer_.node_type == "output" else None

    for layer_ in map.nodes:
        for name in layer_.users:
            temp_map[layer_.name].children.append(temp_map[name])

        for item in layer_.args:
            if isinstance(item, dict) and "node" in item:
                temp_map[layer_.name].parents.append(temp_map[item["node"]])

    return Graph(name_map=temp_map, source_nodes=temp_source, consumed_ids={})



