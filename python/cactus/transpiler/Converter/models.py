import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import torch
from ..ModelProfiles import models as MP_Models

modality_input_path = {
    "vision": "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test_monkey.png",
    "audio" : "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test.wav"
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



def load_files(mp: MP_Models.ModelProfile, model_id: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    from ..ModelProfiles import profiles as MP_Profiles

    default_model_ids = {
        "gemma4_e2b": "google/gemma-4-E2B",
        "whisper": "openai/whisper-tiny",
        "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
        "lfm_vlm": "LiquidAI/LFM2-VL-3B",
        "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
    }

    repo_id = (
        model_id
        or MP_Profiles.MODEL_ID_MAP.get(mp.model_profiles)
        or default_model_ids.get(mp.model_profiles)
        or mp.model_profiles
    )
    output_dir = Path(__file__).resolve().parent / "jsons" / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    downloaded_files: dict[str, str] = {}

    for file in mp.files:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=output_dir,
                token=token,
            )
        except EntryNotFoundError:
            if file == "config.json":
                raise

            print(f"Skipping missing optional file for {repo_id}: {file}")
            continue
        except RepositoryNotFoundError:
            raise ValueError(f"Could not find HuggingFace repository: {repo_id}") from None

        downloaded_files[file] = str(downloaded_path)

    return downloaded_files


def build_input(mp: MP_Models.ModelProfile, input_modalties: tuple[str, ...]) -> Input | None:
    if not all(modality in mp.supported_modalties for modality in input_modalties):
        print("Requesting unsupported modalities")
        return None
    

    




def create_model(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], model_id: str) -> Model:
    name_ = model_id
    b



def export_(model: Model, input: Input) -> LayerMap:
    exported = torch.export.export(model.model, args=input.args, kwargs=input.kwargs, strict=False,).run_decompositions()
    records = [LayerRecord.from_node(i, node) for i, node in enumerate(exported.graph.nodes)]
    return LayerMap.from_exported_model(model=model, input=input, exported_model=exported, nodes=records)
