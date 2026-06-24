from pydantic import BaseModel
from typing import Any
import model_utils
import torch


class InputtedModel(BaseModel):
    name: str
    input_modalities: list[str]
#    output_modalities: list[str] -> Possible need for this, but not sure yet

class LayerRecord(BaseModel):
    index: str
    name: str
    node_type: str
    target: str
    schema: str
    args: Any
    kwargs: Any
    users: list[str]
    tensor_output_meta: Any | None
    module_stack: Any | None

    @classmethod
    def from_node(cls, i:int, node: torch.Node) -> "LayerRecord":
        return cls(
            index = i,
            name = str(node.name),
            node_type = str(node.op),
            target = str(node.target),
            args = model_utils.jsonable(node.args),
            kwargs = model_utils.jsonable(node.kwargs),
            users = [user.name for user in node.users],
            tensor_output_meta = model_utils.extract_tensor_meta(node),
            module_stack = model_utils.extract_module_stack(node)
        )

class TensorInstance(BaseModel):
    shape: list
    dtype: str
    device: str
    requires_grad: bool

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "TensorInstance":
        return cls(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device), requires_grad = bool(x.requires_grad))
    
class Slice(BaseModel):
    start: Any
    stop: Any
    step: Any

    @classmethod
    def from_slice(cls, x: slice) -> "Slice":
        return cls(start = model_utils.jsonable(x.start), stop = model_utils.jsonable(x.stop), step = model_utils.jsonable(x.step))
    

class JSONResult(BaseModel):

class LayerMap(BaseModel)