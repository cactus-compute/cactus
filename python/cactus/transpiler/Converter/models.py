import json
import os
import inspect
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
import torch
from . import input_utils as IU
from ..ModelProfiles import models as MP_Models
from ..ModelProfiles import profiles as MP_Profiles

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForCTC, AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq

default_model_ids: dict[str, MP_Models.ModelProfile] = {
    "google/gemma-4-E2B": MP_Profiles.GEMMA4_E2B_PROFILE,
    "openai/whisper-tiny": MP_Profiles.WHISPER_PROFILE,
    "nvidia/parakeet-tdt-0.6b-v3": MP_Profiles.PARAKEET_PROFILE,
    "LiquidAI/LFM2-VL-3B": MP_Profiles.LFM_VLM_PROFILE,
    "Qwen/Qwen2.5-0.5B": MP_Profiles.QWEN2_5_0_5B_PROFILE,
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
    model: Any

    @classmethod
    def from_profile(mp: MP_Models.ModelProfile, input_modalities: tuple[str,...], model_id: str) -> "Model":
        return create_model(mp, input_modalities, model_id)

    def export(self, input: Input) -> "LayerMap":
        return export_(model=self, input=input)

@dataclass(slots=True)
class TensorInstance:
    shape: list[Any]
    dtype: str

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "TensorInstance":
        return cls(shape=[jsonable_shape_dim(dim) for dim in x.shape], dtype=str(x.dtype))

@dataclass(slots=True)
class Slice:
    start: Any
    stop: Any
    step: Any

    @classmethod
    def from_slice(cls, x: slice) -> "Slice":
        return cls(start=jsonable(x.start), stop=jsonable(x.stop), step=jsonable(x.step))

@dataclass(slots=True)
class CacheLayerSpec:
    index: int
    layer_type: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    sliding_window: int | None = None

@dataclass(slots=True)
class CacheSpec:
    layers: tuple[CacheLayerSpec, ...]
    config: Any
    past_sequence_length: int
    dtype: torch.dtype
    device: torch.device

    #Infers cache tensor shapes from the loaded model config and attention modules.
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        batch_size: int,
        past_sequence_length: int,
    ) -> "CacheSpec":
        config = getattr(model, "config", None)
        text_config = get_text_config(config)
        dtype, device = model_dtype_and_device(model)
        num_key_value_heads = int(getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads", 1)))
        num_attention_heads = int(getattr(text_config, "num_attention_heads", num_key_value_heads))
        hidden_size = int(getattr(text_config, "hidden_size", num_attention_heads))
        head_dim = int(getattr(text_config, "head_dim", max(1, hidden_size // max(1, num_attention_heads))))
        sliding_window = getattr(text_config, "sliding_window", None) or getattr(text_config, "attention_chunk_size", None)
        decoder_layers = model_decoder_layers(model)
        layers: list[CacheLayerSpec] = []

        for index, layer_type in enumerate(cache_layer_types(text_config)):
            cache_sequence_length = past_sequence_length
            layer_sliding_window = None
            layer_num_key_value_heads = num_key_value_heads
            layer_head_dim = head_dim

            if index < len(decoder_layers):
                attention = attention_module(decoder_layers[index])
                layer_head_dim = int(getattr(attention, "head_dim", layer_head_dim))
                key_out_features = linear_out_features(getattr(attention, "k_proj", None))

                if key_out_features is not None:
                    layer_num_key_value_heads = max(1, key_out_features // max(1, layer_head_dim))

            if layer_type in ("sliding_attention", "chunked_attention") and sliding_window is not None:
                layer_sliding_window = int(sliding_window)
                cache_sequence_length = min(past_sequence_length, max(1, layer_sliding_window - 1))

            cache_shape = (batch_size, layer_num_key_value_heads, cache_sequence_length, layer_head_dim)
            layers.append(
                CacheLayerSpec(
                    index=index,
                    layer_type=str(layer_type),
                    key_shape=cache_shape,
                    value_shape=cache_shape,
                    sliding_window=layer_sliding_window,
                )
            )

        return cls(
            layers=tuple(layers),
            config=config,
            past_sequence_length=past_sequence_length,
            dtype=dtype,
            device=device,
        )

    #Creates zero-filled flat KV tensors that match this cache spec.
    def empty_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []

        for layer in self.layers:
            tensors.append(torch.zeros(layer.key_shape, dtype=self.dtype, device=self.device))
            tensors.append(torch.zeros(layer.value_shape, dtype=self.dtype, device=self.device))

        return tuple(tensors)

    #Rebuilds an HF DynamicCache from flat exported tensor inputs.
    def to_dynamic_cache(self, flat_tensors: tuple[torch.Tensor, ...]):
        from transformers.cache_utils import DynamicCache

        if len(flat_tensors) != len(self.layers) * 2:
            raise ValueError(f"Expected {len(self.layers) * 2} cache tensors, got {len(flat_tensors)}")

        cache = DynamicCache(config=self.config)

        for layer_spec, flat_index in zip(self.layers, range(0, len(flat_tensors), 2)):
            layer = cache.layers[layer_spec.index]
            key = flat_tensors[flat_index]
            value = flat_tensors[flat_index + 1]
            layer.keys = key
            layer.values = value
            layer.dtype = key.dtype
            layer.device = key.device
            layer.is_initialized = True

            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = self.past_sequence_length

        return cache


#Wraps HF cache-mode forwards so torch.export only sees tensor inputs/outputs.
class CacheExportWrapper(torch.nn.Module):
    #Stores the wrapped model and cache spec used to tensorize cache state.
    def __init__(self, model: torch.nn.Module, cache_spec: CacheSpec, mode: str):
        super().__init__()
        self.model = model
        self.cache_spec = cache_spec
        self.mode = mode

    #Runs the HF model with DynamicCache internally and returns logits plus flat cache tensors.
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        position_ids=None,
        image_position_ids=None,
        video_position_ids=None,
        cache_position=None,
        past_key_values=None,
        mm_token_type_ids=None,
        inputs_embeds=None,
    ):
        cache = self.cache_spec.to_dynamic_cache(past_key_values) if past_key_values is not None else None
        model_kwargs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "input_features": input_features,
            "attention_mask": attention_mask,
            "input_features_mask": input_features_mask,
            "position_ids": position_ids,
            "image_position_ids": image_position_ids,
            "video_position_ids": video_position_ids,
            "cache_position": cache_position,
            "past_key_values": cache,
            "mm_token_type_ids": mm_token_type_ids,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
        outputs = self.model(**filter_forward_kwargs(self.model, model_kwargs))

        return (primary_model_output(outputs), *flatten_dynamic_cache(outputs.past_key_values))


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

"""#####################################Model Utils#####################################"""

#Converts symbolic shape dims to JSON-safe values without forcing torch guards. x
def jsonable_shape_dim(x: Any) -> Any:
    if type(x) is int:
        return x

    return str(x)


#Recursively converts FX/export metadata into JSON-safe Python values. x
def jsonable(x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        return {"node": x.name}

    if isinstance(x, torch.Tensor):
        tensor = TensorInstance.from_tensor(x)
        return {"shape": tensor.shape, "dtype": tensor.dtype}

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        slice_ = Slice.from_slice(x)
        return {"start": slice_.start, "stop": slice_.stop, "step": slice_.step}

    if isinstance(x, range):
        return list(x)

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return repr(x)


#Returns the text/decoder config for models that wrap language configs inside multimodal configs.
def get_text_config(config: Any) -> Any:
    if config is not None and hasattr(config, "get_text_config"):
        try:
            return config.get_text_config(decoder=True)
        except TypeError:
            return config.get_text_config()

    return config


#Finds the dtype and device of the loaded model parameters.
def model_dtype_and_device(model: torch.nn.Module) -> tuple[torch.dtype, torch.device]:
    try:
        param = next(model.parameters())
        return param.dtype, param.device
    except StopIteration:
        return torch.float32, torch.device("cpu")


#Finds decoder layer modules across common HF model wrapper layouts.
def model_decoder_layers(model: torch.nn.Module) -> tuple[Any, ...]:
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("model", "layers"),
        ("layers",),
    )

    for path in candidates:
        value: Any = model

        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break

        if value is not None:
            return tuple(value)

    return ()


#Finds the attention module inside one decoder layer across common naming conventions.
def attention_module(layer: Any) -> Any:
    for attr in ("self_attn", "attention", "attn"):
        value = getattr(layer, attr, None)
        if value is not None:
            return value

    return layer


#Returns a linear-like module's output feature count when available.
def linear_out_features(module: Any) -> int | None:
    if module is None:
        return None

    if hasattr(module, "out_features"):
        return int(module.out_features)

    weight = getattr(module, "weight", None)
    if weight is None and hasattr(module, "linear"):
        weight = getattr(module.linear, "weight", None)

    if weight is None:
        return None

    return int(weight.shape[0])


#Infers which decoder layers actually own KV cache entries.
def cache_layer_types(text_config: Any) -> tuple[str, ...]:
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 1))
    layer_types = getattr(text_config, "layer_types", None)

    if layer_types is None:
        sliding_window = getattr(text_config, "sliding_window", None) or getattr(text_config, "attention_chunk_size", None)
        default_layer_type = "sliding_attention" if sliding_window is not None else "full_attention"
        layer_types = tuple(default_layer_type for _ in range(num_hidden_layers))
    else:
        layer_types = tuple(str(layer_type) for layer_type in layer_types)

    num_kv_shared_layers = int(getattr(text_config, "num_kv_shared_layers", 0) or 0)
    if num_kv_shared_layers > 0:
        layer_types = layer_types[: -num_kv_shared_layers]

    return layer_types


#Selects the main tensor output from a model output object.
def primary_model_output(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]

    raise ValueError("Could not find a tensor output to export")


#Flattens an HF DynamicCache into key/value tensor outputs.
def flatten_dynamic_cache(cache: Any) -> tuple[torch.Tensor, ...]:
    if cache is None:
        return ()

    flat: list[torch.Tensor] = []

    for layer in cache.layers:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)

        if key is None or value is None:
            continue

        flat.append(key)
        flat.append(value)

    return tuple(flat)


#Drops kwargs unsupported by a module's forward signature.
def filter_forward_kwargs(model: torch.nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


#Normalizes torch operator targets into stable ATen-style names.
def aten_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)

    if schema is not None and "::" in schema.name:
        namespace, op = schema.name.split("::", 1)
        overload = schema.overload_name if schema.overload_name else "default"
        return f"{namespace}.{op}.{overload}"

    if hasattr(target, "name"):
       return target.name

    return str(target)


#Extracts tensor metadata from an exported FX node.
def extract_tensor_meta(node: torch.fx.Node) -> Any | None:
    if "val" not in node.meta:
        return None

    return jsonable(node.meta["val"])


#Extracts module stack metadata from an exported FX node.
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


#Builds representative model inputs for the requested modalities and inference mode.
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

    #Gemma-specific: use the real Gemma4 processor to create multimodal placeholder ids and position tensors.
    if model_id is not None and "gemma" in model_id.lower() and any(m in modalities for m in ("vision", "audio")):
        try:
            kwargs = IU.build_processor_kwargs(model_id, modalities, configs)
            return Input(
                args=(),
                kwargs=kwargs,
                modalities=modalities,
                inference_mode=inference_mode,
            )
        except Exception as e:
            print(f"Processor input build failed for {model_id}, falling back to synthetic inputs: {e}")

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


#Applies mode-specific input rewrites after the real model/config is loaded.
def prepare_input_for_export(model: torch.nn.Module, input: Input) -> Input:
    if input.inference_mode == "decode_with_cache":
        return build_decode_with_cache_input(model, input)

    return input


#Builds one-token decode inputs and flat KV tensors for cache-mode exports.
def build_decode_with_cache_input(model: torch.nn.Module, input: Input) -> Input:
    kwargs = dict(input.kwargs)
    token_key = "input_ids" if "input_ids" in kwargs else "decoder_input_ids"

    if token_key not in kwargs:
        raise ValueError("decode_with_cache requires input_ids or decoder_input_ids")

    token_ids = kwargs[token_key]
    batch_size = int(token_ids.shape[0])
    past_sequence_length = int(token_ids.shape[1])
    cache_spec = CacheSpec.from_model(
        model=model,
        batch_size=batch_size,
        past_sequence_length=past_sequence_length,
    )

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
    kwargs["past_key_values"] = cache_spec.empty_tensors()

    for multimodal_key in (
        "pixel_values",
        "pixel_values_videos",
        "input_features",
        "input_features_mask",
        "image_position_ids",
        "video_position_ids",
        "mm_token_type_ids",
    ):
        kwargs.pop(multimodal_key, None)

    return Input(
        args=input.args,
        kwargs=kwargs,
        modalities=input.modalities,
        inference_mode=input.inference_mode,
    )


#Loads the best HF model class for the requested model profile.
def load_model(model_id: str, mp: MP_Models.ModelProfile | None = None) -> torch.nn.Module:
    

    token = os.environ.get("HF_TOKEN")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    AutoConfig.from_pretrained(model_id, **load_kwargs)

    candidate_classes: list[Any] = []
    modalities = set(mp.supported_modalties if mp is not None else ())
    load_strategy = mp.load_strategy if mp is not None else ""
    is_gemma4_profile = mp is not None and mp.model_profiles == "gemma4_e2b"

    if load_strategy == "image_text_to_text" or ("vision" in modalities and "text" in modalities):
        candidate_classes.extend((AutoModelForImageTextToText, AutoModelForCausalLM))
        #Gemma-specific: avoid falling back to plain AutoModel, which loads Gemma4 weights with wrong prefixes.
        if not is_gemma4_profile:
            candidate_classes.append(AutoModel)
    elif load_strategy == "speech_seq2seq" or ("audio" in modalities and "text" in modalities):
        candidate_classes.extend((AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel))
    elif load_strategy == "ctc" or "audio" in modalities:
        candidate_classes.extend((AutoModelForCTC, AutoModel))
    elif load_strategy == "causal_lm" or "text" in modalities:
        candidate_classes.extend((AutoModelForCausalLM, AutoModel))
    else:
        candidate_classes.append(AutoModel)

    last_error: Exception | None = None
    seen_classes: set[Any] = set()

    for model_class in candidate_classes:
        if model_class in seen_classes:
            continue
        seen_classes.add(model_class)

        for attempt_kwargs in _model_load_kwargs(load_kwargs):
            try:
                model = model_class.from_pretrained(model_id, **attempt_kwargs)
                model.eval()

                configure_model_for_export(model, should_use_cache=False)

                #Gemma-specific: patch Gemma4 audio masking so torch.export can trace it.
                if mp is not None and mp.model_profiles == "gemma4_e2b" and "audio" in mp.supported_modalties:
                    patch_gemma4_audio_mask_for_export()

                return model
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Unable to load model {model_id}") from last_error


#Returns normal load kwargs plus a local-files-only retry for cached HF models.
def _model_load_kwargs(load_kwargs: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if load_kwargs.get("local_files_only") is True:
        return (load_kwargs,)

    return (
        load_kwargs,
        {**load_kwargs, "local_files_only": True},
    )


#Sets an attribute on a config object and common nested config objects.
def _set_config_attr_recursive(config: Any, attr: str, value: Any, seen: set[int] | None = None) -> None:
    if config is None:
        return

    seen = seen or set()
    config_id = id(config)
    if config_id in seen:
        return

    seen.add(config_id)

    if hasattr(config, attr):
        setattr(config, attr, value)

    for child_name in (
        "text_config",
        "vision_config",
        "audio_config",
        "encoder",
        "decoder",
        "model_config",
    ):
        _set_config_attr_recursive(getattr(config, child_name, None), attr, value, seen)


#Configures cache-related model settings before torch.export.
def configure_model_for_export(model: torch.nn.Module, should_use_cache: bool) -> None:
    for config_name in ("config", "generation_config"):
        config = getattr(model, config_name, None)
        _set_config_attr_recursive(config, "use_cache", should_use_cache)


#Checks whether a module forward can accept a specific keyword.
def _forward_accepts_kwarg(model: torch.nn.Module, key: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False

    if key in signature.parameters:
        return True

    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


#Builds the kwargs actually passed into torch.export.
def _export_kwargs(model: torch.nn.Module, input: Input) -> dict[str, Any]:
    kwargs = filter_forward_kwargs(model, dict(input.kwargs))
    should_use_cache = input.inference_mode in ("prefill_with_cache", "decode_with_cache")

    if _forward_accepts_kwarg(model, "use_cache"):
        kwargs["use_cache"] = should_use_cache

    return kwargs


#Gemma-specific: replaces Gemma4's audio bidirectional mask helper with an exportable tensor implementation.
def patch_gemma4_audio_mask_for_export() -> None:
    import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

    #Gemma-specific: computes Gemma4 audio tower local bidirectional attention masks without vmap/proxy issues.
    def exportable_bidirectional_mask(config, inputs_embeds, attention_mask=None, and_mask_function=None, **kwargs):
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
        kv_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)

        left_window_size = getattr(config, "attention_context_left", seq_len) - 1
        right_window_size = getattr(config, "attention_context_right", 0)
        distance = q_idx - kv_idx
        left_mask = (distance >= 0) & (distance < left_window_size)
        right_mask = (distance < 0) & (-distance < right_window_size)
        mask = left_mask | right_mask

        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        return mask.expand(batch_size, 1, seq_len, seq_len)

    gemma4_modeling.create_bidirectional_mask = exportable_bidirectional_mask


#Builds a loaded model bundle from profile, modalities, model id, and inference mode.
def create_model(
    mp: MP_Models.ModelProfile,
    input_modalities: tuple[str, ...],
    model_id: str,
    inference_mode: str = "prefill_no_cache",
) -> Model:
    name_ = model_id
    input_ = build_input(mp, input_modalities, model_id=model_id, inference_mode=inference_mode)

    if input_ is None:
        raise ValueError(f"Could not build input for modalities {input_modalities}")

    loaded_model = load_model(model_id, mp)
    input_ = prepare_input_for_export(loaded_model, input_)
    return Model(name=name_, model_profile=mp, input=input_, model=loaded_model)


#Selects the raw model or cache tensorization wrapper for export.
def build_export_model(model: torch.nn.Module, input: Input) -> torch.nn.Module:
    if input.inference_mode not in ("prefill_with_cache", "decode_with_cache"):
        return model

    batch_size = infer_batch_size(input.kwargs)
    past_sequence_length = infer_past_sequence_length(input)
    cache_spec = CacheSpec.from_model(
        model=model,
        batch_size=batch_size,
        past_sequence_length=past_sequence_length,
    )
    return CacheExportWrapper(model=model, cache_spec=cache_spec, mode=input.inference_mode)


#Infers batch size from the first tensor-shaped input.
def infer_batch_size(kwargs: dict[str, Any]) -> int:
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])

    return 1


#Infers the past/prompt sequence length represented by the export input.
def infer_past_sequence_length(input: Input) -> int:
    if input.inference_mode == "decode_with_cache" and "cache_position" in input.kwargs:
        return int(input.kwargs["cache_position"][0].item())

    if "input_ids" in input.kwargs:
        return int(input.kwargs["input_ids"].shape[1])

    if "decoder_input_ids" in input.kwargs:
        return int(input.kwargs["decoder_input_ids"].shape[1])

    if "attention_mask" in input.kwargs:
        return int(input.kwargs["attention_mask"].shape[1])

    return 0



#Exports the prepared model and serializes its FX graph into a LayerMap.
def export_(model: Model, input: Input) -> LayerMap:
    should_use_cache = input.inference_mode in ("prefill_with_cache", "decode_with_cache")
    configure_model_for_export(model.model, should_use_cache=should_use_cache)
    export_model = build_export_model(model.model, input)
    export_kwargs = _export_kwargs(export_model, input)

    with torch.no_grad():
        exported = torch.export.export(
            export_model,
            args=input.args,
            kwargs=export_kwargs,
            strict=False,
        ).run_decompositions()

    graph = exported.graph_module.graph if hasattr(exported, "graph_module") else exported.graph
    records = [LayerRecord.from_node(i, node) for i, node in enumerate(graph.nodes)]
    return LayerMap.from_data(
        x=exported,
        name=model.name,
        model_task=input.inference_mode,
        nodes_list=records,
    )
