from pydantic import BaseModel
from typing import Any, Optional
from dataclasses import dataclass
import torch

model_presets: dict[str, str] = {
    "PREFILL_NO_CACHE": "prefill_no_cache",
    "PREFILL_WITH_CACHE": "prefill_with_cache",
    "DECODING_WITH_CACHE": "decoding_with_cache",
}

@dataclass(slots=True)
class ExportModel:
    name: str
    
    input_modalities: list[str]
    output_modalities: list[str]

class InputtedModel(BaseModel):
    name: str
    input_modalities: list[str]
    output_modalities: list[str]

class LayerRecord(BaseModel):
    index: int
    name: str
    node_type: str
    target: str
    args: Any
    kwargs: Any
    users: list[str]
    tensor_output_meta: Any | None
    module_stack: Any | None

    @classmethod
    def from_node(cls, num:int, x: torch.Node) -> "LayerRecord":
        return cls(
            index = num,
            name = str(x.name),
            node_type = str(x.op),
            target = str(x.target),
            args = jsonable(x.args),
            kwargs = jsonable(x.kwargs),
            users = [user.name for user in x.users],
            tensor_output_meta = extract_tensor_meta(x),
            module_stack = extract_module_stack(x)
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
        return cls(start = jsonable(x.start), stop = jsonable(x.stop), step = jsonable(x.step))
    

class LayerMap(BaseModel):
    model_name: str
    task: str
    graph_signature: str
    range_constants: str
    nodes: list[LayerRecord]

    @classmethod
    def from_data(cls, x:torch.export.ExportedProgram, name:str, model_task:str, nodes_list:list[LayerRecord]) -> "LayerMap":
        return cls(model_name = name, task = model_task, graph_signature = repr(x.graph_signature), range_constants = repr(x.range_constraints), nodes = nodes_list)



"""###################################### MODEL UTILS!!!!!!! ######################################"""

def jsonable(x: Any) -> Any:
    """
    Convert graph/export objects into JSON-safe values.
    """
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        return TensorInstance.from_tensor(x)

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return Slice.from_slice(x)

    if isinstance(x, range):
        return list(x)

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return repr(x)


def aten_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)

    if schema is not None:
        if "::" in schema.name:
            namespace, op = schema.name.split("::", 1)
            overload = schema.overload_name if schema.overload_name else "default"
            return f"{namespace}.{op}.{overload}"

    if hasattr(target, "name"):
        try:
            return target.name()
        except Exception:
            pass

    return str(target)


def extract_tensor_meta(node: torch.fx.Node) -> Optional[Any]:
    """
    torch.export usually stores fake tensor output metadata in node.meta["val"].
    """
    if "val" not in node.meta:
        return None

    return jsonable(node.meta["val"])


#This function needs to be further optimized and cleaned up
#CLEANUP: Create object for output dictionary (potentially one outter object as well); Condense down for loop
def extract_module_stack(node: torch.fx.Node) -> Optional[Any]:
    """
    Sometimes torch.export nodes contain original module context here.
    This can help later when mapping ops back to model.layers.X.self_attn, mlp, etc.
    """
    stack = node.meta.get("nn_module_stack", None)

    if stack is None:
        return None

    out = []

    for key, value in stack.items():
        if isinstance(value, tuple) and len(value) >= 2:
            module_path = value[0]
            module_type = value[1]
            out.append(
                {
                    "key": str(key),
                    "module_path": str(module_path),
                    "module_type": getattr(module_type, "__name__", str(module_type)),
                }
            )
        else:
            out.append(
                {
                    "key": str(key),
                    "value": repr(value),
                }
            )

    return out