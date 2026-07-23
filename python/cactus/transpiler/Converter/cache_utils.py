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

    #How: delegates model/cache inspection to cache_spec_from_model.
    #Why: used by models.export_ and input_utils.build_decode_with_cache_input to create cache shapes for export.
    @classmethod
    def from_model(cls, model: torch.nn.Module, batch_size: int, past_sequence_length: int) -> "CacheSpec":
        return cache_spec_from_model(cls, model, batch_size, past_sequence_length)

    #How: delegates zero tensor creation to cache_spec_empty_tensors.
    #Why: used when building decode inputs so torch.export receives real tensor placeholders for past_key_values.
    def empty_tensors(self, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> tuple[torch.Tensor, ...]:
        return cache_spec_empty_tensors(self, dtype=dtype, device=device)

    #How: delegates flat tensor to DynamicCache reconstruction to cache_spec_to_dynamic_cache.
    #Why: used inside CacheExportWrapper.forward because HF models expect DynamicCache, not a flat tuple.
    def to_dynamic_cache(self, flat_tensors: tuple[torch.Tensor, ...]):
        return cache_spec_to_dynamic_cache(self, flat_tensors)


#Wraps HF cache-mode forwards so torch.export only sees tensor inputs/outputs.
class CacheExportWrapper(torch.nn.Module):
    #How: stores the original HF model plus the CacheSpec that describes flattened cache tensors.
    #Why: constructed by models.export_ whenever a cache-mode export needs tensor-only inputs and outputs.
    def __init__(self, model: torch.nn.Module, cache_spec: CacheSpec):
        super().__init__()
        self.model = model
        self.cache_spec = cache_spec

    #How: rebuilds DynamicCache from flat tensors, calls the model, then flattens the returned cache.
    #Why: torch.export cannot directly expose HF cache objects, so this is the bridge for decode/prefill cache graphs.
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


#How: asks HF configs for their decoder/text sub-config when available, otherwise returns the original config.
#Why: cache_spec_from_model needs decoder fields even when the top-level config is multimodal, like Gemma.
def get_text_config(config: Any) -> Any:
    if config is not None and hasattr(config, "get_text_config"):
        try:
            return config.get_text_config(decoder=True)
        except TypeError:
            return config.get_text_config()

    return config


#How: reads the dtype from the first model parameter and falls back to float32 for parameterless modules.
#Why: input_utils.build_decode_with_cache_input uses this so synthetic cache tensors match the model dtype.
def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        param = next(model.parameters())
        return param.dtype
    except StopIteration:
        return torch.float32


#How: tries common attribute paths until it finds a layers collection.
#Why: cache_spec_from_model uses real layers to validate or refine cache head dimensions beyond config defaults.
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


#How: checks common attention attribute names and falls back to the layer itself.
#Why: cache_spec_from_model needs the attention module to inspect fields like head_dim and k_proj.
def attention_module(layer: Any) -> Any:
    for attr in ("self_attn", "attention", "attn"):
        value = getattr(layer, attr, None)
        if value is not None:
            return value

    return layer


#How: reads out_features or the first dimension of a weight tensor from linear-like modules.
#Why: cache_spec_from_model uses k_proj output size to infer per-layer KV head count.
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


#How: reads layer_types from config, otherwise derives full/sliding attention from window settings.
#Why: cache_spec_from_model uses this to know how many cache tensor pairs to create and which are sliding-window.
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


#How: combines config defaults with actual attention module inspection to create one CacheLayerSpec per cache layer.
#Why: CacheSpec.from_model calls this before decode/cache export so past_key_values tensors have correct shapes.
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


#How: allocates key and value zero tensors for every CacheLayerSpec, in flat key0/value0/key1/value1 order.
#Why: CacheSpec.empty_tensors uses this to create placeholder cache inputs for torch.export decode traces.
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


#How: creates DynamicCache and assigns each flat key/value tensor into the matching cache layer.
#Why: CacheSpec.to_dynamic_cache uses this so CacheExportWrapper.forward can call the original HF model normally.
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


#How: checks common HF output containers for logits, last_hidden_state, or first tuple item.
#Why: CacheExportWrapper.forward needs a stable first exported output before appending flat cache tensors.
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


#How: walks either legacy tuple caches or DynamicCache layers and emits key/value tensors in flat order.
#Why: CacheExportWrapper.forward returns this flat tuple so the layermap contains explicit cache output tensors.
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


#How: inspects forward(...) and drops kwargs unless the module accepts **kwargs.
#Why: used by CacheExportWrapper.forward and models.export_ to avoid passing unsupported modality/cache args.
def filter_forward_kwargs(model: torch.nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


#How: recursively sets one config attr on top-level and nested text/vision/audio/encoder/decoder configs.
#Why: configure_model_for_export uses it to consistently enable or disable use_cache before tracing.
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


#How: sets use_cache on model.config and model.generation_config, including nested configs.
#Why: models.export_ calls this so prefill/decode cache exports include cache paths and no-cache exports do not.
def configure_model_for_export(model: torch.nn.Module, should_use_cache: bool) -> None:
    for config_name in ("config", "generation_config"):
        config = getattr(model, config_name, None)
        _set_config_attr_recursive(config, "use_cache", should_use_cache)
