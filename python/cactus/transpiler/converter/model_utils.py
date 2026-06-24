from typing import Any, Optional
import torch
from models import TensorInstance, Slice

def jsonable(x: Any) -> Any:
    """
    Convert graph/export objects into JSON-safe values.
    """
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        return TensorInstance(x)

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return Slice(x)

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
    """
    Convert default naming to simpler name.
    For non-ATen targets, falls back to str(target).
    """
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
