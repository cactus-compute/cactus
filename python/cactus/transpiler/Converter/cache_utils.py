import inspect
from dataclasses import dataclass
from typing import Any
import torch

@dataclass(slots=True)
class CacheLayerSpec:
    index: int
    layer_type: str
    key_shape: tuple[int, ...] | None = None
    value_shape: tuple[int, ...] | None = None
    cross_key_shape: tuple[int, ...] | None = None
    cross_value_shape: tuple[int, ...] | None = None
    conv_state_shape: tuple[int, ...] | None = None
    recurrent_state_shape: tuple[int, ...] | None = None
    sliding_window: int | None = None


@dataclass(slots=True)
class CacheSpec:
    layers: tuple[CacheLayerSpec, ...]
    config: Any
    past_sequence_length: int

    #How: delegates model/cache inspection to cache_spec_from_model.
    #Why: used by models.export_ and models.build_decode_with_cache_input to create cache shapes for export.
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
        pixel_attention_mask=None,
        spatial_shapes=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
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
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
            "pixel_values_videos": pixel_values_videos,
            "input_features": input_features,
            "attention_mask": attention_mask,
            "input_features_mask": input_features_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "decoder_position_ids": decoder_position_ids,
            "position_ids": position_ids,
            "image_position_ids": image_position_ids,
            "video_position_ids": video_position_ids,
            "cache_position": cache_position,
            "past_key_values": cache,
            "mm_token_type_ids": mm_token_type_ids,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
            "return_dict": True,
        }
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
        outputs = self.model(**filter_forward_kwargs(self.model, model_kwargs))

        return (primary_model_output(outputs), *flatten_dynamic_cache(getattr(outputs, "past_key_values", None), self.cache_spec))


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
#Why: models.build_decode_with_cache_input uses this so synthetic cache tensors match the model dtype.
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
    num_hidden_layers = cache_layer_count(text_config)
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


def cache_layer_count(text_config: Any) -> int:
    for attr in ("num_hidden_layers", "decoder_layers", "num_layers", "n_layer"):
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    return 1


def cache_num_attention_heads(text_config: Any) -> int:
    for attr in ("num_attention_heads", "decoder_attention_heads", "n_head"):
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    return 1


def cache_hidden_size(text_config: Any, num_attention_heads: int) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    return num_attention_heads


def encoder_decoder_source_length(text_config: Any) -> int | None:
    for attr in ("max_source_positions", "encoder_seq_length", "max_encoder_position_embeddings"):
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    return None


def encoder_decoder_target_length(text_config: Any) -> int | None:
    for attr in ("max_target_positions", "decoder_seq_length", "max_position_embeddings", "n_positions"):
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    return None


#How: combines config defaults with actual attention module inspection to create one CacheLayerSpec per cache layer.
#Why: CacheSpec.from_model calls this before decode/cache export so past_key_values tensors have correct shapes.
def cache_spec_from_model(cls: type[CacheSpec], model: torch.nn.Module, batch_size: int, past_sequence_length: int) -> CacheSpec:
    config = getattr(model, "config", None)
    text_config = get_text_config(config)
    num_attention_heads = cache_num_attention_heads(text_config)
    num_key_value_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
    hidden_size = cache_hidden_size(text_config, num_attention_heads)
    head_dim = int(getattr(text_config, "head_dim", max(1, hidden_size // max(1, num_attention_heads))))
    sliding_window = getattr(text_config, "sliding_window", None) or getattr(text_config, "attention_chunk_size", None)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False) or getattr(text_config, "is_encoder_decoder", False))
    cross_sequence_length = encoder_decoder_source_length(text_config) if is_encoder_decoder else None
    target_sequence_length = encoder_decoder_target_length(text_config) if is_encoder_decoder else None
    decoder_layers = model_decoder_layers(model)
    layers: list[CacheLayerSpec] = []

    for index, layer_type in enumerate(cache_layer_types(text_config)):
        cache_sequence_length = int(target_sequence_length or past_sequence_length)
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

        if layer_type == "conv":
            layers.append(
                CacheLayerSpec(
                    index=index,
                    layer_type=str(layer_type),
                    conv_state_shape=(batch_size, hidden_size, int(getattr(text_config, "conv_L_cache", 1))),
                )
            )
            continue

        if layer_type in ("mamba", "linear_attention", "moe"):
            layers.append(CacheLayerSpec(index=index, layer_type=str(layer_type)))
            continue

        cache_shape = (batch_size, layer_num_key_value_heads, cache_sequence_length, layer_head_dim)
        cross_cache_shape = (
            (batch_size, layer_num_key_value_heads, int(cross_sequence_length), layer_head_dim)
            if cross_sequence_length is not None
            else None
        )
        layers.append(
            CacheLayerSpec(
                index=index,
                layer_type=str(layer_type),
                key_shape=cache_shape,
                value_shape=cache_shape,
                cross_key_shape=cross_cache_shape,
                cross_value_shape=cross_cache_shape,
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
        if layer.key_shape is not None:
            tensors.append(torch.zeros(layer.key_shape, dtype=dtype, device=device))
        if layer.value_shape is not None:
            tensors.append(torch.zeros(layer.value_shape, dtype=dtype, device=device))
        if layer.cross_key_shape is not None:
            tensors.append(torch.zeros(layer.cross_key_shape, dtype=dtype, device=device))
        if layer.cross_value_shape is not None:
            tensors.append(torch.zeros(layer.cross_value_shape, dtype=dtype, device=device))
        if layer.conv_state_shape is not None:
            tensors.append(torch.zeros(layer.conv_state_shape, dtype=dtype, device=device))
        if layer.recurrent_state_shape is not None:
            tensors.append(torch.zeros(layer.recurrent_state_shape, dtype=dtype, device=device))

    return tuple(tensors)


#How: creates DynamicCache and assigns each flat key/value tensor into the matching cache layer.
#Why: CacheSpec.to_dynamic_cache uses this so CacheExportWrapper.forward can call the original HF model normally.
def cache_spec_to_dynamic_cache(cache_spec: CacheSpec, flat_tensors: tuple[torch.Tensor, ...]):
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    expected_tensor_count = cache_spec_tensor_count(cache_spec)

    if len(flat_tensors) != expected_tensor_count:
        raise ValueError(f"Expected {expected_tensor_count} cache tensors, got {len(flat_tensors)}")

    has_cross_cache = any(layer.cross_key_shape is not None or layer.cross_value_shape is not None for layer in cache_spec.layers)
    cache = EncoderDecoderCache(DynamicCache(config=cache_spec.config), DynamicCache(config=cache_spec.config)) if has_cross_cache else DynamicCache(config=cache_spec.config)
    self_cache = cache.self_attention_cache if has_cross_cache else cache
    cross_cache = cache.cross_attention_cache if has_cross_cache else None
    flat_index = 0

    for layer_spec in cache_spec.layers:
        layer = self_cache.layers[layer_spec.index]

        if layer_spec.key_shape is not None and layer_spec.value_shape is not None:
            key = flat_tensors[flat_index]
            value = flat_tensors[flat_index + 1]
            flat_index += 2
            layer.keys = key
            layer.values = value
            layer.dtype = key.dtype
            layer.device = key.device
            layer.is_initialized = True

            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = cache_spec.past_sequence_length

        if cross_cache is not None and layer_spec.cross_key_shape is not None and layer_spec.cross_value_shape is not None:
            cross_key = flat_tensors[flat_index]
            cross_value = flat_tensors[flat_index + 1]
            flat_index += 2
            cross_layer = cross_cache.layers[layer_spec.index]
            cross_layer.keys = cross_key
            cross_layer.values = cross_value
            cross_layer.dtype = cross_key.dtype
            cross_layer.device = cross_key.device
            cross_layer.is_initialized = True
            cache.is_updated[layer_spec.index] = True

        if layer_spec.conv_state_shape is not None:
            conv_state = flat_tensors[flat_index].clone()
            flat_index += 1
            layer.conv_states = conv_state
            layer.dtype = conv_state.dtype
            layer.device = conv_state.device
            layer.max_batch_size = conv_state.shape[0]
            layer.conv_kernel_size = conv_state.shape[-1]
            layer.is_conv_states_initialized = True
            layer.has_previous_state = cache_spec.past_sequence_length > 0

        if layer_spec.recurrent_state_shape is not None:
            recurrent_state = flat_tensors[flat_index].clone()
            flat_index += 1
            layer.recurrent_states = recurrent_state
            layer.dtype = recurrent_state.dtype
            layer.device = recurrent_state.device
            layer.is_recurrent_states_initialized = True
            layer.has_previous_state = cache_spec.past_sequence_length > 0

    return cache


def cache_spec_tensor_count(cache_spec: CacheSpec) -> int:
    count = 0

    for layer in cache_spec.layers:
        count += int(layer.key_shape is not None)
        count += int(layer.value_shape is not None)
        count += int(layer.cross_key_shape is not None)
        count += int(layer.cross_value_shape is not None)
        count += int(layer.conv_state_shape is not None)
        count += int(layer.recurrent_state_shape is not None)

    return count


#How: checks common HF output containers for logits, last_hidden_state, or first tuple item.
#Why: CacheExportWrapper.forward needs a stable first exported output before appending flat cache tensors.
def primary_model_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        for key in ("logits", "last_hidden_state"):
            output = outputs.get(key)
            if isinstance(output, torch.Tensor):
                return output

    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits

    if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
        return outputs.last_hidden_state

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
        return outputs[0]

    raise ValueError("Could not find a tensor output to export")


#How: walks either legacy tuple caches or DynamicCache layers and emits key/value tensors in flat order.
#Why: CacheExportWrapper.forward returns this flat tuple so the layermap contains explicit cache output tensors.
def flatten_dynamic_cache(cache: Any, cache_spec: CacheSpec | None = None) -> tuple[torch.Tensor, ...]:
    if cache is None:
        return ()

    flat: list[torch.Tensor] = []
    self_cache = getattr(cache, "self_attention_cache", None)
    cross_cache = getattr(cache, "cross_attention_cache", None)

    if self_cache is not None and cross_cache is not None and cache_spec is not None:
        for layer_spec in cache_spec.layers:
            self_layer = self_cache.layers[layer_spec.index]
            cross_layer = cross_cache.layers[layer_spec.index]

            if layer_spec.key_shape is not None:
                key = getattr(self_layer, "keys", None)
                if key is not None:
                    flat.append(key)

            if layer_spec.value_shape is not None:
                value = getattr(self_layer, "values", None)
                if value is not None:
                    flat.append(value)

            if layer_spec.cross_key_shape is not None:
                cross_key = getattr(cross_layer, "keys", None)
                if cross_key is not None:
                    flat.append(cross_key)

            if layer_spec.cross_value_shape is not None:
                cross_value = getattr(cross_layer, "values", None)
                if cross_value is not None:
                    flat.append(cross_value)

        return tuple(flat)

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

    if cache_spec is not None:
        for layer_spec in cache_spec.layers:
            layer = layers[layer_spec.index]

            if layer_spec.key_shape is not None:
                key = getattr(layer, "keys", None)
                if key is not None:
                    flat.append(key)

            if layer_spec.value_shape is not None:
                value = getattr(layer, "values", None)
                if value is not None:
                    flat.append(value)

            if layer_spec.conv_state_shape is not None:
                conv_state = getattr(layer, "conv_states", None)
                if conv_state is not None:
                    flat.append(conv_state)

            if layer_spec.recurrent_state_shape is not None:
                recurrent_state = getattr(layer, "recurrent_states", None)
                if recurrent_state is not None:
                    flat.append(recurrent_state)

        return tuple(flat)

    for layer in layers:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)

        if key is None or value is None:
            conv_state = getattr(layer, "conv_states", None)
            recurrent_state = getattr(layer, "recurrent_states", None)

            if conv_state is not None:
                flat.append(conv_state)
            if recurrent_state is not None:
                flat.append(recurrent_state)
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
