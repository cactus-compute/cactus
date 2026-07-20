import json
import os
from dataclasses import dataclass, asdict
from typing import Any
from pydantic import BaseModel
import torch
from . import input_utils as IU
from ..ModelProfiles import models as MP_Models
from ..ModelProfiles import profiles as MP_Profiles

default_model_ids = {
    "gemma4_e2b": "google/gemma-4-E2B",
    "whisper": "openai/whisper-tiny",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "lfm_vlm": "LiquidAI/LFM2-VL-3B",
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
}


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

    @classmethod
    def from_profile(mp: MP_Models.ModelProfile, input_modalities: tuple[str,...], model_id: str) -> "Model":
        return create_model(mp, input_modalities, model_id)


    def export(self, input: Input) -> "LayerMap":
        export_(model=self, input=input)


@dataclass(slots=True)
class TensorInstance:
    shape: list[int]
    dtype: str

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "TensorInstance":
        return cls(shape=list(x.shape), dtype=str(x.dtype))

@dataclass(slots=True)
class Slice:
    start: Any
    stop: Any
    step: Any

    @classmethod
    def from_slice(cls, x: slice) -> "Slice":
        return cls(start=jsonable(x.start), stop=jsonable(x.stop), step=jsonable(x.step))


@dataclass(slots=True)
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
    def from_node(cls, num: int, x: torch.fx.Node) -> "LayerRecord":
        return cls(
            index=num,
            name=str(x.name),
            node_type=str(x.op),
            target=str(x.target),
            args=jsonable(x.args),
            kwargs=jsonable(x.kwargs),
            users=[user.name for user in x.users],
            tensor_output_meta=extract_tensor_meta(x),
            module_stack=extract_module_stack(x),
        )

    

@dataclass(slots=True)
class LayerMap(BaseModel):
    model_name: str
    task: str
    graph_signature: str
    range_constants: str
    nodes: list[LayerRecord]

    @classmethod
    def from_data(cls, x: torch.export.ExportedProgram, name: str, model_task: str, nodes_list: list[LayerRecord]) -> "LayerMap":
        return cls(model_name=name, task=model_task, graph_signature=repr(x.graph_signature), range_constants=repr(x.range_constraints), nodes=nodes_list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)

"""#####################################Model Utils#####################################"""

def jsonable(x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        return asdict(TensorInstance.from_tensor(x))

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return asdict(Slice.from_slice(x))

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

    if schema is not None and "::" in schema.name:
        namespace, op = schema.name.split("::", 1)
        overload = schema.overload_name if schema.overload_name else "default"
        return f"{namespace}.{op}.{overload}"

    if hasattr(target, "name"):
       return target.name

    return str(target)


def extract_tensor_meta(node: torch.fx.Node) -> Any | None:
    if "val" not in node.meta:
        return None

    return jsonable(node.meta["val"])


def extract_module_stack(node: torch.fx.Node) -> Any | None:
    stack = node.meta.get("nn_module_stack", None)
    if stack is None:
        return None

    out = []

    for key, value in stack.items():
        if isinstance(value, tuple) and len(value) >= 2:
            module_path = value[0]
            module_type = value[1]
            out.append({"key": str(key), "module_path": str(module_path), "module_type": getattr(module_type, "__name__", str(module_type))})
        else:
            out.append({"key": str(key), "value": repr(value)})

    return out


#TODO: Clean up this function
def build_input(
    mp: MP_Models.ModelProfile,
    input_modalties: tuple[str, ...],
    model_id: str | None = None,
    inference_mode: str = "prefill_no_cache",
) -> Input | None:
    if not all(modality in mp.supported_modalties for modality in input_modalties):
        print("Requesting unsupported modalities")
        return None

    loaded_files = IU.load_files(mp, model_id)

    configs = IU._load_json_files(loaded_files or {})
    modalities = tuple(input_modalties)
    batch_size = 1
    sequence_length = 8
    past_sequence_length = 8
    kwargs: dict[str, Any] = {}

    if "text" in modalities:
        if inference_mode == "decode_with_cache":
            sequence_length = 1

        kwargs["input_ids"] = IU._build_input_ids(batch_size, sequence_length)
        kwargs["attention_mask"] = IU._build_attention_mask(batch_size, sequence_length)

    if "vision" in modalities:
        kwargs["pixel_values"] = IU._build_pixel_values(batch_size, configs)

    if "audio" in modalities:
        kwargs["input_features"] = IU._build_input_features(batch_size, configs)

    if "audio" in modalities and "text" in modalities and "vision" not in modalities and "speech" in inference_mode:
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs["decoder_input_ids"] = torch.full(
            (batch_size, 1),
            IU._decoder_start_token_id(configs),
            dtype=torch.long,
        )

    if "audio" in modalities and "text" not in modalities:
        input_features = kwargs["input_features"]
        kwargs["attention_mask"] = IU._build_attention_mask(batch_size, input_features.shape[-1])

    if inference_mode in ("prefill_with_cache", "decode_with_cache"):
        total_sequence_length = past_sequence_length + sequence_length
        kwargs["attention_mask"] = IU._build_attention_mask(batch_size, total_sequence_length)
        kwargs["past_key_values"] = IU._build_past_key_values(configs, batch_size, past_sequence_length)
        kwargs["cache_position"] = torch.arange(
            past_sequence_length,
            past_sequence_length + sequence_length,
            dtype=torch.long,
        )
        kwargs["use_cache"] = True

    return Input(
        args=(),
        kwargs=kwargs,
        modalities=modalities,
        inference_mode=inference_mode,
    )




def create_model(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], model_id: str) -> Model:
    name_ = model_id
    input_ = build_input(mp, input_modalities)



def export_(model: Model, input: Input) -> LayerMap:
    exported = torch.export.export(model.model, args=input.args, kwargs=input.kwargs, strict=False,).run_decompositions()
    records = [LayerRecord.from_node(i, node) for i, node in enumerate(exported.graph.nodes)]
    return LayerMap.from_exported_model(model=model, input=input, exported_model=exported, nodes=records)
