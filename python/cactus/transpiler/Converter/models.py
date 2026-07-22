import os
import inspect
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
import torch
from transformers import AutoModel

from . import input_utils as IU
from ..ModelProfiles import models as MP_Models
from . import constants


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

    def export(self, input: Input) -> "LayerMap":
        return export_(model=self, input=input)


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

    #Delegates cache tensor shape inference to the utility helper below. X
    @classmethod
    def from_model(cls, model: torch.nn.Module, batch_size: int, past_sequence_length: int) -> "CacheSpec":
        return cache_spec_from_model(cls, model, batch_size, past_sequence_length)

    #Delegates creation of zero-filled flat KV tensors to the utility helper below. X
    def empty_tensors(self, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> tuple[torch.Tensor, ...]:
        return cache_spec_empty_tensors(self, dtype=dtype, device=device)

    #Delegates rebuilding an HF DynamicCache from flat exported tensor inputs. X
    def to_dynamic_cache(self, flat_tensors: tuple[torch.Tensor, ...]):
        return cache_spec_to_dynamic_cache(self, flat_tensors)


#Wraps HF cache-mode forwards so torch.export only sees tensor inputs/outputs. X
class CacheExportWrapper(torch.nn.Module):
    #Stores the wrapped model and cache spec used to tensorize cache state. X
    def __init__(self, model: torch.nn.Module, cache_spec: CacheSpec):
        super().__init__()
        self.model = model
        self.cache_spec = cache_spec

    #Runs the HF model with DynamicCache internally and returns logits plus flat cache tensors. X
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

        return (primary_model_output(outputs), *flatten_dynamic_cache(getattr(outputs, "past_key_values", None)))


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


#Returns the text/decoder config for models that wrap language configs inside multimodal configs. X
def get_text_config(config: Any) -> Any:
    if config is not None and hasattr(config, "get_text_config"):
        try:
            return config.get_text_config(decoder=True)
        except TypeError:
            return config.get_text_config()

    return config


#Finds the dtype of the loaded model parameters. X
def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        param = next(model.parameters())
        return param.dtype
    except StopIteration:
        return torch.float32


#Finds decoder layer modules across common HF model wrapper layouts. X
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


#Finds the attention module inside one decoder layer across common naming conventions. X
def attention_module(layer: Any) -> Any:
    for attr in ("self_attn", "attention", "attn"):
        value = getattr(layer, attr, None)
        if value is not None:
            return value

    return layer


#Returns a linear-like module's output feature count when available. X
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


#Infers which decoder layers actually own KV cache entries. X
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


#Infers cache tensor shapes from the loaded model config and attention modules. X
def cache_spec_from_model(cls: type[CacheSpec], model: torch.nn.Module, batch_size: int, past_sequence_length: int) -> CacheSpec:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
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
    )


#Creates zero-filled flat KV tensors that match this cache spec. X
def cache_spec_empty_tensors(
    cache_spec: CacheSpec,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    device = device or torch.device("cpu")

    for layer in cache_spec.layers:
        tensors.append(torch.zeros(layer.key_shape, dtype=dtype, device=device))
        tensors.append(torch.zeros(layer.value_shape, dtype=dtype, device=device))

    return tuple(tensors)


#Rebuilds an HF DynamicCache from flat exported tensor inputs. X
def cache_spec_to_dynamic_cache(cache_spec: CacheSpec, flat_tensors: tuple[torch.Tensor, ...]):
    from transformers.cache_utils import DynamicCache

    if len(flat_tensors) != len(cache_spec.layers) * 2:
        raise ValueError(f"Expected {len(cache_spec.layers) * 2} cache tensors, got {len(flat_tensors)}")

    cache = DynamicCache(config=cache_spec.config)

    for index, layer_spec in enumerate(cache_spec.layers):
        flat_index = index * 2
        layer = cache.layers[layer_spec.index]
        key = flat_tensors[flat_index]
        value = flat_tensors[flat_index + 1]
        layer.keys = key
        layer.values = value
        layer.dtype = key.dtype
        layer.device = key.device
        layer.is_initialized = True

        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = cache_spec.past_sequence_length

    return cache


#Selects the main tensor output from a model output object. X
def primary_model_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        for key in ("logits", "last_hidden_state"):
            output = outputs.get(key)
            if output is not None:
                return output

    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]

    raise ValueError("Could not find a tensor output to export")


#Flattens an HF DynamicCache into key/value tensor outputs. X
def flatten_dynamic_cache(cache: Any) -> tuple[torch.Tensor, ...]:
    if cache is None:
        return ()

    flat: list[torch.Tensor] = []
    layers = getattr(cache, "layers", None)

    if layers is None and isinstance(cache, (tuple, list)):
        for entry in cache:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                key, value = entry[:2]
                if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                    flat.extend((key, value))
        return tuple(flat)

    if layers is None:
        return ()

    for layer in layers:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)

        if key is None or value is None:
            continue

        flat.append(key)
        flat.append(value)

    return tuple(flat)


#Drops kwargs unsupported by a module's forward signature. X
def filter_forward_kwargs(model: torch.nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


#Extracts tensor metadata from an exported FX node. X
def extract_tensor_meta(node: torch.fx.Node) -> Any | None:
    if "val" not in node.meta:
        return None

    return jsonable(node.meta["val"])


#Extracts module stack metadata from an exported FX node. X
def extract_module_stack(node: torch.fx.Node) -> Any | None:
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
            out.append({"key": str(key), "value": repr(value)})

    return out


#Loads the best HF model class for the requested model profile. X
def load_model(model_id: str, mp: MP_Models.ModelProfile | None = None) -> torch.nn.Module:
    token = os.environ.get("HF_TOKEN")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    if mp is None:
        candidate_classes = (AutoModel,)
        export_patches = ()
    else:
        candidate_classes = constants.LOAD_STRATEGIES.get(mp.load_strategy, (AutoModel,))
        export_patches = mp.export_patches

    last_error: Exception | None = None
    seen_classes: set[Any] = set()

    for model_class in candidate_classes:
        if model_class in seen_classes:
            continue
        seen_classes.add(model_class)

        for attempt_kwargs in IU.hf_load_kwargs(load_kwargs):
            try:
                model = model_class.from_pretrained(model_id, **attempt_kwargs)
                model.eval()

                for patch in export_patches:
                    patch_fn = EXPORT_PATCHES.get(patch)
                    if patch_fn is None:
                        raise ValueError(f"Unknown export patch: {patch}")
                    patch_fn()

                return model
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Unable to load model {model_id}") from last_error


#Sets an attribute on a config object and common nested config objects. X
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


#Configures cache-related model settings before torch.export. X
def configure_model_for_export(model: torch.nn.Module, should_use_cache: bool) -> None:
    for config_name in ("config", "generation_config"):
        config = getattr(model, config_name, None)
        _set_config_attr_recursive(config, "use_cache", should_use_cache)


#Gemma-specific: replaces Gemma4's audio bidirectional mask helper with an exportable tensor implementation. X
def patch_gemma4_audio_mask_for_export() -> None:
    import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

    #Gemma-specific: computes Gemma4 audio tower local bidirectional attention masks without vmap/proxy issues. X
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


EXPORT_PATCHES = {
    "gemma4_audio_mask": patch_gemma4_audio_mask_for_export,
}


#Builds a loaded model bundle from profile, modalities, model id, and inference mode. X
def create_model(
    mp: MP_Models.ModelProfile,
    input_modalities: tuple[str, ...],
    model_id: str,
    inference_mode: str = "prefill_no_cache",
) -> Model:
    input_ = IU.build_input(
        mp=mp,
        input_modalities=input_modalities,
        input_cls=Input,
        model_id=model_id,
        inference_mode=inference_mode,
    )

    if input_ is None:
        raise ValueError(f"Could not build input for modalities {input_modalities}")

    loaded_model = load_model(model_id, mp)
    if input_.inference_mode == IU.DECODE_WITH_CACHE_MODE and constants.DYNAMIC_CACHE_POLICY in mp.cache_policy:
        input_ = IU.build_decode_with_cache_input(
            model=loaded_model,
            input_=input_,
            input_cls=Input,
            cache_spec_cls=CacheSpec,
            model_dtype_fn=model_dtype,
            drop_multimodal=constants.DROP_MULTIMODAL_ON_DECODE_POLICY in mp.cache_policy,
        )

    return Model(name=model_id, model_profile=mp, input=input_, model=loaded_model)


#Exports the prepared model and serializes its FX graph into a LayerMap. X
def export_(model: Model, input: Input) -> LayerMap:
    should_use_cache = input.inference_mode in constants.CACHE_INFERENCE_MODES
    configure_model_for_export(model.model, should_use_cache=should_use_cache)
    export_model = model.model

    if should_use_cache and constants.DYNAMIC_CACHE_POLICY in model.model_profile.cache_policy:
        export_model = CacheExportWrapper(
            model=model.model,
            cache_spec=CacheSpec.from_model(
                model=model.model,
                batch_size=IU.infer_batch_size(input.kwargs),
                past_sequence_length=IU.infer_past_sequence_length(input),
            ),
        )

    export_kwargs = dict(input.kwargs)
    export_kwargs["use_cache"] = should_use_cache
    export_kwargs = filter_forward_kwargs(export_model, export_kwargs)

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
