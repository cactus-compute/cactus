import os
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
import torch
from transformers import AutoModel

from . import input_utils as IU
from . import cache_utils as CU
from . import export_patches as EP
from ..ModelProfiles import models as MP_Models
from . import constants

token = os.environ.get("HF_TOKEN")


@dataclass(slots=True)
class Input:
    args: tuple
    kwargs: dict
    modalities: tuple[str, ...]
    inference_mode: str

@dataclass(slots=True)
class Model:
    name: str
    model_profile: MP_Models.ModelProfile
    input: Input
    model: torch.nn.Module

    def export(self, input: Input) -> "LayerMap":
        return export_(model=self, input=input)


#Serializable record for one exported FX graph node.
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

    #Builds a LayerRecord from a torch.fx.Node.
    @classmethod
    def from_node(cls, num: int, x: torch.fx.Node) -> "LayerRecord":
        return cls(index=num, name=str(x.name), node_type=str(x.op), target=str(x.target), args=jsonable(x.args), kwargs=jsonable(x.kwargs), users=[user.name for user in x.users], tensor_output_meta=extract_tensor_meta(x), module_stack=extract_module_stack(x))

#Serializable top-level export IR container.
class LayerMap(BaseModel):
    model_name: str
    task: str
    graph_signature: str
    range_constants: str
    nodes: list[LayerRecord]

    #Builds a LayerMap from a torch.export ExportedProgram and serialized nodes.
    @classmethod
    def from_data(cls, x: torch.export.ExportedProgram, name: str, model_task: str, nodes_list: list[LayerRecord]) -> "LayerMap":
        return cls(model_name=name, task=model_task, graph_signature=repr(x.graph_signature), range_constants=repr(x.range_constraints), nodes=nodes_list)

###################################################### Model utility helpers!!!!! ########################################################################.

#Recursively converts FX/export metadata into JSON-safe Python values.
def jsonable(x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        return {"shape": x if isinstance(x, int) else str(x), "dtype": x.dtype}

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return {"start":jsonable(x.start), "stop":jsonable(x.stop), "step":jsonable(x.step)}

    if isinstance(x, range):
        return list(x)

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return repr(x)


#Extracts tensor metadata from an exported FX node.
def extract_tensor_meta(node: torch.fx.Node) -> Any | None:
    return jsonable(node.meta["val"]) if "val" not in node.meta else None


#Extracts module stack metadata from an exported FX node.
def extract_module_stack(node: torch.fx.Node) -> Any | None:
    stack = node.meta.get("nn_module_stack", None)
    out = []

    if stack is None:
        return None

    for key, value in stack.items():
        if isinstance(value, tuple) and len(value) >= 2:
            module_path, module_type = value[0], value[1]
            out.append({"key": str(key), "module_path": str(module_path), "module_type": getattr(module_type, "__name__", str(module_type))})
        else:
            out.append({"key": str(key), "value": repr(value)})

    return out


#Loads the best HF model class for the requested model profile.
def load_model(model_id: str, mp: MP_Models.ModelProfile | None = None) -> torch.nn.Module:
    if mp is None:
        candidate_classes = (AutoModel,)
        export_patches = ()
    else:
        candidate_classes = constants.LOAD_STRATEGIES.get(mp.load_strategy, (AutoModel,))
        export_patches = mp.export_patches

    seen_classes: set[Any] = set()

    for model_class in candidate_classes:
        if model_class in seen_classes:
            continue

        seen_classes.add(model_class)
        try:
            model = model_class.from_pretrained(model_id)
            model.eval()
        except:
            pass

        for patch in export_patches:
            patch_fn = EXPORT_PATCHES[patch]
            patch_fn()

        return model

    raise RuntimeError(f"Unable to load model {model_id}")



EXPORT_PATCHES = {
    "gemma4_audio_mask": EP.patch_gemma4_audio_mask_for_export,
}


#Builds a loaded model bundle from profile, modalities, model id, and inference mode.
def create_model(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], model_id: str, inference_mode: str = "prefill_no_cache") -> Model:
    
    input_ = IU.build_input(mp, input_modalities, Input, model_id, inference_mode)
    if input_ is None:
        raise ValueError(f"Could not build input for modalities {input_modalities}")

    loaded_model = load_model(model_id, mp)
    
    if input_.inference_mode == IU.DECODE_WITH_CACHE_MODE and constants.DYNAMIC_CACHE_POLICY in mp.cache_policy:
        input_ = IU.build_decode_with_cache_input(model=loaded_model, input_=input_, input_cls=Input, cache_spec_cls=CU.CacheSpec, model_dtype_fn=CU.model_dtype, drop_multimodal=constants.DROP_MULTIMODAL_ON_DECODE_POLICY in mp.cache_policy)
    
    #TODO: Add for models that do support multimodal decoders
    # elif input_.inference_mode == IU.DECODE_WITH_CACHE_MODE:
    #     input_ = IU.build_decode_with_cache_input()



    return Model(name=model_id, model_profile=mp, input=input_, model=loaded_model)


#Exports the prepared model and serializes its FX graph into a LayerMap. X
def export_(model: Model, input: Input) -> LayerMap:
    should_use_cache = input.inference_mode in constants.CACHE_INFERENCE_MODES
    CU.configure_model_for_export(model.model, should_use_cache=should_use_cache)
    export_model = model.model

    if should_use_cache and constants.DYNAMIC_CACHE_POLICY in model.model_profile.cache_policy:
        export_model = CU.CacheExportWrapper(
            model=model.model,
            cache_spec=CU.CacheSpec.from_model(model=model.model, batch_size=IU.infer_batch_size(input.kwargs), past_sequence_length=IU.infer_past_sequence_length(input)),
        )

    export_kwargs = dict(input.kwargs)
    export_kwargs["use_cache"] = should_use_cache
    export_kwargs = CU.filter_forward_kwargs(export_model, export_kwargs)

    with torch.no_grad():
        exported = torch.export.export(export_model, args=input.args, kwargs=export_kwargs, strict=False).run_decompositions()

    graph = exported.graph_module.graph if hasattr(exported, "graph_module") else exported.graph
    records = [LayerRecord.from_node(i, node) for i, node in enumerate(graph.nodes)]
    return LayerMap.from_data(x=exported, name=model.name, model_task=input.inference_mode, nodes_list=records)
