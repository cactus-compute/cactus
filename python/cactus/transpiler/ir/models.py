from dataclasses import dataclass
from typing import Any
from collections import deque
import converter.models as CVModels
import constants

@dataclass(slots=True)
class Node:
    layer: CVModels.LayerRecord
    parents: list["Node"]
    children: list["Node"]


@dataclass(slots=True)
class Graph:
    name_map:dict[str, CVModels.LayerRecord]
    source_nodes: deque[Node]
    consumed_ids: set[str]

    @classmethod
    def from_layer_map(cls, map:CVModels.LayerMap):
        return generate_graph(map)
    
    

"""###################################### MODEL UTILS!!!!!!! ######################################"""

def generate_graph(map: CVModels.LayerMap):
    temp_map:dict[str, Node] = {}
    temp_source = deque()

    for layer_ in map.nodes:
        temp = Node(layer=layer_, parents=[], children=[])
        temp_map[layer_.name] = temp
        temp_source.append(temp) if layer_.node_type == "output" else None

    for layer_ in map.nodes:
        for name in layer_.users:
            temp_map[layer_.name].children.append(temp_map[name])

        for item in layer_.args:
            if isinstance(item, dict) and "node" in item:
                temp_map[layer_.name].parents.append(temp_map[item["node"]])

    return Graph(name_map=temp_map, source_nodes=temp_source, consumed_ids={})




