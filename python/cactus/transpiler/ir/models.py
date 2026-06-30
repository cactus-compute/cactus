from pydantic import BaseModel
from typing import Any
from collections import deque
import converter.models as CVModels
import constants


class ValueNode(BaseModel):
    id:str
    type:str
    # shape:list[int] | None TODO: May need shape (value or output) later on, but may not be necessary since graph is known to be complete/correct

    @classmethod
    def from_placeholder(cls, x:CVModels.LayerRecord) -> "ValueNode":
        return cls(id = x.name, type = x.node_type)


class OperationNode(BaseModel):
    id:str
    op:str
    inputs_id:list[str]
    users:list[str]
    attrs:dict[Any, Any]

    #Constructor
    @classmethod
    def from_op(cls, x:CVModels.LayerRecord) -> "OperationNode":
        attrs_list:list[str] = constants.OP_ATTRS_SCHEMAS[x.target]
        attrs_temp:dict[str,Any] = {}
        inputs:list[str] = []
        count = 0

        for item in x.args:
            if isinstance(item, dict) and "node" in item:
                inputs.append(item["node"])
            else:
                if count < len(attrs_list):
                    attrs_temp[attrs_list[count]] = item
                    count += 1
                else:
                    raise ValueError(f"{x.name}: Layer has more attributes than expected!")


        return cls(id = x.name, op = x.target, inputs_id = inputs, users = x.users, attrs = attrs_temp)
    
class OutputNode(BaseModel):
    id:str
    inputs_id:list[str]

    #Constructor
    @classmethod
    def from_output(cls, x: CVModels.LayerRecord) -> "OutputNode":
        inputs:list[str] = []
        count = 0

        for item in x.args:
            if isinstance(item, dict) and "node" in item:
                inputs.append(item["node"])

        return cls(id = x.name, inputs_id = inputs)



class CompGraph(BaseModel):
    nodes:dict[str, Any] # Turn this into dict of (name: node)
    source_nodes:deque[OperationNode, ValueNode]

    #Constructor
    @classmethod
    def from_graph(cls, graph:CVModels.LayerMap):
        return generate_comp_graph_(graph)
    

    def perform_fusions():
        pass




"""###################################### MODEL UTILS!!!!!!! ######################################"""

function_map = {
    "placeholder" : ValueNode.from_placeholder,
    "call_function" : OperationNode.from_op,
    "output" : OutputNode.from_output,
}

def generate_comp_graph_(graph: CVModels.LayerMap):
    nodes_dict = {}
    output_nodes = deque()

    for node in graph.nodes:
        new_node = function[node.node_type](node)
        nodes_dict[node.name] = new_node

        if isinstance(new_node, OutputNode):
            output_nodes.append(new_node)

    return CompGraph(nodes = nodes_dict, source_nodes = output_nodes)
            

# def is_source_node(value: list) -> bool:
#     #TODO: Horrifically ugly logical statement here, try to shorten it down. Just checking if input to this node is source node
#     if len(value) > 0 and isinstance(value, dict) and "node" in value[0] and value[0]["node"] == "input_ids":
#         return True
#     else:
#         False

# def add_node_(self:CompGraph, x: CVModels.LayerRecord) -> None:
#     node = function_map[x.node_type](x)
#     self.nodes.append(node)

#     if(is_source_node(x.args)):
#         self.source_nodes.append()

    

