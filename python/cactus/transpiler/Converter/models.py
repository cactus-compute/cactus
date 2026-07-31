import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download
import numpy as np
from pydantic import BaseModel, Field
import torch
from transformers import AutoModel

from . import cache_utils as CU
from . import input_processor as IP
from . import overrides as OV
from ..ModelProfiles import models as MP_Models
from . import constants

token = constants.token
EXPORT_PATCHES = {
    "gemma4_audio_mask": OV.patch_gemma4_audio_mask_for_export,
    "transformers_moe_grouped_mm_fallback": OV.patch_transformers_moe_grouped_mm_for_export,
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
    ir_metadata: dict[str, Any] = Field(default_factory=dict)

    #Builds a LayerRecord from a torch.fx.Node.
    @classmethod
    def from_node(cls, num: int, x: torch.fx.Node) -> "LayerRecord":
        return cls(index=num, name=str(x.name), node_type=str(x.op), target=str(x.target), args=jsonable(x.args), kwargs=jsonable(x.kwargs), users=[user.name for user in x.users], tensor_output_meta=extract_tensor_meta(x), module_stack=extract_module_stack(x))


class GraphSpecRecord(BaseModel):
    kind: str
    arg_name: str | None = None
    target: str | None = None
    persistent: bool | None = None


#Serializable top-level export IR container.
class LayerMap(BaseModel):
    model_name: str
    task: str
    graph_signature: str
    range_constants: str
    input_specs: list[GraphSpecRecord] = Field(default_factory=list)
    output_specs: list[GraphSpecRecord] = Field(default_factory=list)
    nodes: list[LayerRecord]

    #Builds a LayerMap from a torch.export ExportedProgram and serialized nodes.
    @classmethod
    def from_data(cls, x: torch.export.ExportedProgram, name: str, model_task: str, nodes_list: list[LayerRecord]) -> "LayerMap":
        return cls(
            model_name=name,
            task=model_task,
            graph_signature=repr(x.graph_signature),
            range_constants=repr(x.range_constraints),
            input_specs=extract_graph_signature_specs(x.graph_signature, "input_specs"),
            output_specs=extract_graph_signature_specs(x.graph_signature, "output_specs"),
            nodes=nodes_list,
        )

###################################################### Model utility helpers!!!!! ########################################################################.

#Recursively converts FX/export metadata into JSON-safe Python values.
def jsonable(x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        return {
            "shape": [jsonable(dim) for dim in x.shape],
            "dtype": str(x.dtype),
            "device": str(x.device),
            "requires_grad": bool(x.requires_grad),
            "stride": [jsonable(dim) for dim in x.stride()],
        }

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


def extract_tensor_meta(node: torch.fx.Node) -> Any | None:
    return jsonable(node.meta["val"]) if "val" in node.meta else None


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


def extract_graph_signature_specs(graph_signature: Any, field_name: str) -> list[GraphSpecRecord]:
    specs = getattr(graph_signature, field_name, ()) or ()
    return [graph_spec_record(spec) for spec in specs]


def graph_spec_record(spec: Any) -> GraphSpecRecord:
    return GraphSpecRecord(
        kind=spec_kind(spec),
        arg_name=spec_arg_name(spec),
        target=none_or_str(getattr(spec, "target", None)),
        persistent=getattr(spec, "persistent", None),
    )


def spec_kind(spec: Any) -> str:
    kind = getattr(spec, "kind", None)

    if kind is None:
        return "unknown"

    name = getattr(kind, "name", None)

    if name is not None:
        return str(name).lower()

    return str(kind)


def spec_arg_name(spec: Any) -> str | None:
    arg = getattr(spec, "arg", None)

    if arg is None:
        return None

    name = getattr(arg, "name", None)

    if name is not None:
        return str(name)

    return None


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def load_configs(mp: MP_Models.ModelProfile, model_id: str | None) -> dict[str, dict[str, Any]]:
    output_dir = constants.CONVERTER_JSON_DIR / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict[str, Any]] = {}

    for filename in mp.files:
        if not filename.endswith(".json"):
            continue

        local_path = output_dir / filename

        if not local_path.exists():
            try:
                local_path = Path(hf_hub_download(repo_id=model_id, filename=filename, local_dir=output_dir, token=constants.token))
            except Exception as e:
                print(f"Error downloading {filename} from {model_id}: {e}")
                continue

        with local_path.open("r", encoding="utf-8") as f:
            configs[filename] = json.load(f)

    return configs


def build_input(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], input_cls: Any, model_id: str | None = None, inference_mode: str = "prefill_no_cache") -> Any | None:
    if not all(modality in mp.supported_modalties for modality in input_modalities):
        print("Requesting unsupported modalities")
        return None

    configs = load_configs(mp, model_id)

    if mp.input_strategy == constants.SYNTHETIC_INPUT_STRATEGY:
        raise ValueError("Synthetic input creation is currently disabled")

    return input_cls(
        args=(),
        kwargs=build_processor_kwargs(model_id, input_modalities, configs, mp.model_profiles),
        modalities=input_modalities,
        inference_mode=inference_mode,
    )


#How: slices full prompt tokens to one decode token, adds cache_position, attention_mask, and flat cache tensors.
#Why: create_model uses this for decode_with_cache so the exported graph represents one-token generation.
def build_decode_with_cache_input(model: torch.nn.Module, input_: Any, input_cls: Any, cache_spec_cls: Any, model_dtype_fn: Any, drop_multimodal: bool) -> Any:
    kwargs = dict(input_.kwargs)
    token_key = "input_ids" if "input_ids" in kwargs else "decoder_input_ids"

    if token_key not in kwargs:
        raise ValueError(f"{constants.DECODE_WITH_CACHE_MODE} requires input_ids or decoder_input_ids")

    token_ids = kwargs[token_key]
    batch_size = int(token_ids.shape[0])
    past_sequence_length = int(token_ids.shape[1])
    cache_spec = cache_spec_cls.from_model(model=model, batch_size=batch_size, past_sequence_length=past_sequence_length)

    kwargs[token_key] = token_ids[:, -1:].clone()
    kwargs["attention_mask"] = torch.ones(
        (batch_size, past_sequence_length + 1),
        dtype=torch.long,
        device=token_ids.device,
    )
    kwargs["cache_position"] = torch.arange(
        past_sequence_length,
        past_sequence_length + 1,
        dtype=torch.long,
        device=token_ids.device,
    )
    kwargs["past_key_values"] = cache_spec.empty_tensors(
        dtype=model_dtype_fn(model),
        device=token_ids.device,
    )

    if drop_multimodal:
        for multimodal_key in constants.MULTIMODAL_KEYS:
            kwargs.pop(multimodal_key, None)

    return input_cls(
        args=input_.args,
        kwargs=kwargs,
        modalities=input_.modalities,
        inference_mode=input_.inference_mode,
    )



def infer_batch_size(kwargs: dict[str, Any]) -> int:
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])

    return 1


#How: prefers cache_position for decode, otherwise checks token or attention-mask sequence length.
#Why: export_ uses this to tell CacheSpec how much past context the exported cache represents.
def infer_past_sequence_length(input_: Any) -> int:
    if input_.inference_mode == constants.DECODE_WITH_CACHE_MODE and "cache_position" in input_.kwargs:
        return int(input_.kwargs["cache_position"][0].item())

    if "input_ids" in input_.kwargs:
        return int(input_.kwargs["input_ids"].shape[1])

    if "decoder_input_ids" in input_.kwargs:
        return int(input_.kwargs["decoder_input_ids"].shape[1])

    if "attention_mask" in input_.kwargs:
        return int(input_.kwargs["attention_mask"].shape[1])

    return 0


def _load_image_asset():
    from PIL import Image
    return Image.open(constants.MODALITY_INPUT_PATH["vision"]).convert("RGB")


def _load_audio_asset() -> np.ndarray:
    from scipy.io import wavfile

    _sample_rate, audio = wavfile.read(constants.MODALITY_INPUT_PATH["audio"])

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    return audio


def build_processor_kwargs(model_id: str, input_modalities: tuple[str, ...], configs: dict[str, dict[str, Any]], model_profile: str) -> dict[str, Any]:
    processor_builder = IP.PROCESSOR_MAP.get(model_id, IP.default_processor)
    processor = processor_builder(model_id=model_id, configs=configs, model_profile=model_profile)

    prompt = []
    call_kwargs: dict[str, Any] = {"return_tensors": "pt"}

    if "vision" in input_modalities:
        image_token = getattr(processor, "image_token", None)
        if image_token is not None:
            prompt.append(image_token)
        call_kwargs["images"] = _load_image_asset()

    if "audio" in input_modalities:
        audio_token = getattr(processor, "audio_token", None)
        if audio_token is not None:
            prompt.append(audio_token)
        call_kwargs["audio"] = _load_audio_asset()
        call_kwargs["sampling_rate"] = 16000

    if "text" in input_modalities:
        prompt.append("Describe this input.")

    call_kwargs["text"] = " ".join(prompt) if prompt else "Hello"
    processed = processor(**call_kwargs)
    pad_audio_export_inputs(processed, configs)
    kwargs = {key: value for key, value in dict(processed).items() if isinstance(value, torch.Tensor)}
    normalize_processor_kwargs_for_export(kwargs, model_profile)
    return kwargs


def normalize_processor_kwargs_for_export(kwargs: dict[str, torch.Tensor], model_profile: str) -> None:
    if model_profile == "whisper" and "labels" in kwargs and "decoder_input_ids" not in kwargs:
        kwargs["decoder_input_ids"] = kwargs.pop("labels")


def pad_audio_export_inputs(processed: Any, configs: dict[str, dict[str, Any]]) -> None:
    target_frames = gemma4_audio_export_frame_count(configs)
    if target_frames is None:
        return

    if "input_features" in processed:
        processed["input_features"] = pad_tensor_dim(processed["input_features"], dim=1, target=target_frames, value=0)

    if "input_features_mask" in processed:
        processed["input_features_mask"] = pad_tensor_dim(processed["input_features_mask"], dim=1, target=target_frames, value=0)


def gemma4_audio_export_frame_count(configs: dict[str, dict[str, Any]]) -> int | None:
    processor_config = configs.get("processor_config.json", {})
    config = configs.get("config.json", {})

    if processor_config.get("processor_class") != "Gemma4Processor":
        return None

    audio_seq_length = processor_config.get("audio_seq_length")
    audio_config = config.get("audio_config", {})
    attention_chunk_size = audio_config.get("attention_chunk_size")

    if audio_seq_length is None or attention_chunk_size is None:
        return None

    return int(audio_seq_length) * 4 + int(attention_chunk_size)


def pad_tensor_dim(tensor: torch.Tensor, dim: int, target: int, value: int | float | bool = 0) -> torch.Tensor:
    if dim >= tensor.ndim or tensor.shape[dim] >= target:
        return tensor

    shape = list(tensor.shape)
    shape[dim] = target - int(tensor.shape[dim])
    padding = torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=dim)


#Loads the best HF model class for the requested model profile.
def load_model(model_id: str, mp: MP_Models.ModelProfile | None = None) -> torch.nn.Module:
    if mp is None:
        candidate_classes = (AutoModel,)
        export_patches = ()
    else:
        candidate_classes = constants.LOAD_STRATEGIES.get(mp.load_strategy, (AutoModel,))
        export_patches = mp.export_patches

    seen_classes: set[Any] = set()
    base_kwargs: dict[str, Any] = {"trust_remote_code": True}
    last_error: Exception | None = None

    if constants.token is not None:
        base_kwargs["token"] = constants.token

    load_attempts = (
        {**base_kwargs, "dtype": "auto", "low_cpu_mem_usage": True, "local_files_only": True},
        {**base_kwargs, "torch_dtype": "auto", "low_cpu_mem_usage": True, "local_files_only": True},
        {**base_kwargs, "local_files_only": True},
        {**base_kwargs, "dtype": "auto", "low_cpu_mem_usage": True},
        {**base_kwargs, "torch_dtype": "auto", "low_cpu_mem_usage": True},
        base_kwargs,
    )

    for model_class in candidate_classes:
        if model_class in seen_classes:
            continue

        seen_classes.add(model_class)

        for load_kwargs in load_attempts:
            try:
                model = model_class.from_pretrained(model_id, **load_kwargs)
                model.eval()

                for patch in export_patches:
                    patch_fn = EXPORT_PATCHES[patch]
                    patch_fn()

                return model
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Unable to load model {model_id}") from last_error


#Builds a loaded model bundle from profile, modalities, model id, and inference mode.
def create_model(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], model_id: str, inference_mode: str = "prefill_no_cache") -> Model:
    
    input_ = build_input(mp, input_modalities, Input, model_id, inference_mode)
    if input_ is None:
        raise ValueError(f"Could not build input for modalities {input_modalities}")

    loaded_model = load_model(model_id, mp)
    
    if input_.inference_mode == constants.DECODE_WITH_CACHE_MODE and constants.DYNAMIC_CACHE_POLICY in mp.cache_policy:
        input_ = build_decode_with_cache_input(model=loaded_model, input_=input_, input_cls=Input, cache_spec_cls=CU.CacheSpec, model_dtype_fn=CU.model_dtype, drop_multimodal=constants.DROP_MULTIMODAL_ON_DECODE_POLICY in mp.cache_policy)
    
    #TODO: Add for models that do support multimodal decoders
    # elif input_.inference_mode == constants.DECODE_WITH_CACHE_MODE:
    #     input_ = build_decode_with_cache_input()



    return Model(name=model_id, model_profile=mp, input=input_, model=loaded_model)


def export_(model: Model, input: Input) -> LayerMap:
    should_use_cache = input.inference_mode in constants.CACHE_INFERENCE_MODES
    CU.configure_model_for_export(model.model, should_use_cache=should_use_cache)
    export_model = model.model

    if should_use_cache and constants.DYNAMIC_CACHE_POLICY in model.model_profile.cache_policy:
        export_model = CU.CacheExportWrapper(
            model=model.model,
            cache_spec=CU.CacheSpec.from_model(model=model.model, batch_size=infer_batch_size(input.kwargs), past_sequence_length=infer_past_sequence_length(input)),
        )

    export_kwargs = dict(input.kwargs)
    export_kwargs["use_cache"] = should_use_cache
    export_kwargs = CU.filter_forward_kwargs(export_model, export_kwargs)

    with torch.no_grad():
        exported = torch.export.export(export_model, args=input.args, kwargs=export_kwargs, strict=False).run_decompositions()

    graph = exported.graph_module.graph if hasattr(exported, "graph_module") else exported.graph
    records = [LayerRecord.from_node(i, node) for i, node in enumerate(graph.nodes)]
    return LayerMap.from_data(x=exported, name=model.name, model_task=input.inference_mode, nodes_list=records)
