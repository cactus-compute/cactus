from pydantic import BaseModel
from typing import Any, Optional
import torch
import converter.models as CVModels

class Value(BaseModel):
    id:str
    type:str
    shape:list[int] | None

    @classmethod
    def from_placeholder(x:CVModels.LayerRecord) -> "Value":
        pass


class Operation(BaseModel):
    id:str
    op:str
    inputs:list["Operation"]
    attrs:dict[Any, Any]
    module_path:str

    @classmethod
    def from_op(x:CVModels.LayerRecord) -> "Operation":
        pass

class Output(BaseModel):
    id:str
    inputs:list[Operation]
    attrs:dict[Any, Any]
    module_path:str

    @classmethod
    def from_output(x:CVModels.LayerRecord):
        pass


class SimplifiedModel(BaseModel):
    operations:list[Operation]
    values:list[Value]

    @classmethod
    def from_(x:CVModels) -> "SimplifiedModel":
        pass







"""###################################### MODEL UTILS!!!!!!! ######################################"""

