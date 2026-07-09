from dataclasses import dataclass
from typing import Any, Optional
import converter.models as CVModels
import constants

@dataclass(slots=True)
class Node:
    layer: Optional[CVModels.LayerRecord]
    underlying_op: str
    normalized_attrs: dict[str, Any]
    parents: list["Node"]
    children: list["Node"]


@dataclass(slots=True)
class Graph:
    name_map:dict[str, Node]
    source_nodes: list[Node]
    consumed_ids: set[str]

    @classmethod
    def from_layer_map(cls, map:CVModels.LayerMap) -> "Graph":
        return generate_graph(map)

@dataclass(slots = True)
class NodeSpec:
    underlying_op:str
    required_attrs: dict[str, Any]

@dataclass(slots = True)
class Edge:
    from_node:str
    parent_index:int
    to_node:str

@dataclass(slots=True)
class FusionGraph:
    target:str
    root:str
    edges:tuple[Edge]
    


"""###################################### MODEL UTILS!!!!!!! ######################################"""

def extract_node_names(value: Any) -> list[str]:
    
    if isinstance(value, dict):
        refs = []
        if "node" in value:
            refs.append(value["node"])

        for item in value.values():
            refs.extend(extract_node_names(item))

        return refs

    if isinstance(value, list):
        refs = []
        for item in value:
            refs.extend(extract_node_names(item))
        return refs

    return []

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
    temp_source = []

    for layer_ in map.nodes:
        temp = Node(layer=layer_, normalized_attrs=extract_attrs(layer_), underlying_op= layer_.target, parents=[], children=[])
        temp_map[layer_.name] = temp
        temp_source.append(temp) if layer_.node_type == "output" else None

    for layer_ in map.nodes:
        for name in layer_.users:
            temp_map[layer_.name].children.append(temp_map[name])

        for name in extract_node_names(layer_.args):
            temp_map[layer_.name].parents.append(temp_map[name])

    return Graph(name_map=temp_map, source_nodes=temp_source, consumed_ids=set())











