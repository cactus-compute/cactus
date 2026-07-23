import inspect
from dataclasses import dataclass
from typing import Any

import torch


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
