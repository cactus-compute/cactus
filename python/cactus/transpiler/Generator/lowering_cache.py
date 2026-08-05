from __future__ import annotations

import math
from typing import Any

from . import models
from .errors import UnsupportedLoweringError
from .lowering_utils import *
from ..Fusions import models as FModels
from ..IR import models as IRModels
from ..RuntimePlan import models as RPModels


def cache_attention_generation_plan(graph: IRModels.Graph, model_profile: Any | None = None) -> tuple[frozenset[str], frozenset[str], dict[str, IRModels.CacheAnnotation]]:
    skip_names: set[str] = set()
    cache_state_names: set[str] = set()
    prefill_annotations = prefill_cache_cat_annotations(graph, model_profile)

    for node in graph.nodes:
        if node.target != "cactus.attention" or len(node.parents) < 3:
            continue

        key_match = find_cache_concat_ancestor(node.parents[1], FModels.CacheTensorRole.KEY)
        value_match = find_cache_concat_ancestor(node.parents[2], FModels.CacheTensorRole.VALUE)

        if key_match is None or value_match is None:
            continue

        key_cat, value_cat = key_match.concat, value_match.concat
        key_cache, value_cache = key_match.state, value_match.state

        if key_cache.cache is None or value_cache.cache is None:
            continue

        if key_cache.cache.layer_index != value_cache.cache.layer_index:
            continue

        skip_names.update(cache_wrapper_path_names(node.parents[1], key_cat))
        skip_names.update(cache_wrapper_path_names(node.parents[2], value_cat))
        skip_names.update(wrapper.name for wrapper in key_match.state_wrappers)
        skip_names.update(wrapper.name for wrapper in value_match.state_wrappers)
        cache_state_names.update((key_cache.name, value_cache.name))

    # Cache-side layout wrappers describe the exported tensor view, not native
    # cache storage.  Never materialize them as graph ops, including when a
    # structural attention fusion already bypassed the concatenation.
    for node in graph.nodes:
        for role in (FModels.CacheTensorRole.KEY, FModels.CacheTensorRole.VALUE):
            match = IRModels.find_cache_concat_ancestor(node, role, max_depth=1)
            if match is None or match.concat is not node:
                continue
            skip_names.update(wrapper.name for wrapper in match.state_wrappers)
            cache_state_names.add(match.state.name)

    for cat_name in prefill_annotations:
        cat_node = graph.nodes_map.get(cat_name)

        if cat_node is not None and cat_node.parents:
            skip_names.add(cat_node.parents[0].name)

    return frozenset(skip_names), frozenset(cache_state_names), prefill_annotations


def prefill_cache_cat_annotations(graph: IRModels.Graph, model_profile: Any | None = None) -> dict[str, IRModels.CacheAnnotation]:
    annotations: dict[str, IRModels.CacheAnnotation] = {}
    candidates = [node for node in graph.nodes if is_empty_cache_cat(node)]

    for index, node in enumerate(candidates):
        shape = concrete_shape(meta_shape(node))

        if shape is None or len(shape) != 4:
            continue

        _, num_kv_heads, sequence_length, head_dim = shape
        layer_index = IRModels.cache_layer_index_from_node(node, index // 2)
        window_size = prefill_cache_window_size(graph, layer_index, sequence_length, model_profile)
        requested_capacity = int(node.ir_metadata.get("prefill_cache_capacity", sequence_length))
        cache_capacity = prefill_cache_capacity(graph, requested_capacity, window_size)
        annotations[node.name] = IRModels.CacheAnnotation(
            kind=FModels.CacheKind.KV,
            role=FModels.CacheTensorRole.KEY if index % 2 == 0 else FModels.CacheTensorRole.VALUE,
            tensor_index=index,
            layer_index=layer_index,
            shape=tuple(shape),
            layout="batch_heads_sequence_head_dim",
            sequence_length=cache_capacity,
            window_size=window_size,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            source="prefill_empty_cache_cat",
        )

    return annotations


def prefill_cache_window_size(graph: IRModels.Graph, layer_index: int | None, sequence_length: int, model_profile: Any | None = None) -> int | None:
    model_name = str(getattr(graph, "model_name", "")).lower()

    if "gemma" not in model_name:
        return None

    if layer_index in full_retention_kv_layers(model_profile):
        return 0

    if IRModels.is_gemma4_model_name(model_name) and layer_index == 13:
        return 0

    if layer_index is None:
        return None

    return 0 if layer_index % 5 == 4 else 512


def full_retention_kv_layers(model_profile: Any | None) -> frozenset[int]:
    cache_contract = getattr(model_profile, "cache_contract", None)
    layers = getattr(cache_contract, "full_retention_kv_layers", ()) if cache_contract is not None else ()
    return frozenset(int(layer_index) for layer_index in layers)


def prefill_cache_capacity(graph: IRModels.Graph, sequence_length: int, window_size: int | None) -> int:
    model_name = str(getattr(graph, "model_name", "")).lower()
    capacity = int(sequence_length)

    if "gemma" in model_name:
        if window_size:
            return int(window_size)

        return capacity + 1

    return capacity


def is_empty_cache_cat(node: IRModels.Node) -> bool:
    if node.target not in {"cactus.cat", "aten.cat.default"} or len(node.parents) < 2:
        return False

    first_parent = node.parents[0]
    return first_parent.value_kind == FModels.ValueKind.LIFTED_CONSTANT and element_count(meta_shape(first_parent)) == 0


def find_cache_cat_ancestor(node: IRModels.Node, role: str, max_depth: int = 16) -> IRModels.Node | None:
    match = IRModels.find_cache_concat_ancestor(node, role, max_depth=max_depth)
    return match.concat if match is not None else None


def find_cache_concat_ancestor(node: IRModels.Node, role: str, max_depth: int = 16) -> IRModels.CacheConcatMatch | None:
    return IRModels.find_cache_concat_ancestor(node, role, max_depth=max_depth)


def cache_wrapper_path_names(start: IRModels.Node, stop: IRModels.Node) -> set[str]:
    names: set[str] = set()
    current = start

    while True:
        names.add(current.name)

        if current is stop:
            return names

        if len(current.parents) != 1:
            return set()

        current = current.parents[0]


def should_lower_cache_placeholder_as_state(context: models.GenerationContext, node: IRModels.Node) -> bool:
    if node.cache is None:
        return False

    if node.name in context.cache_state_placeholder_names:
        return True

    if node.cache.kind == FModels.CacheKind.CONV:
        return any(child.target in {"cactus.conv_cache_append", "cactus.conv_cache_initialize"} for child in node.children)

    if node.cache.kind == FModels.CacheKind.KV:
        if any(child.target in {"cactus.cat", "aten.cat.default"} for child in node.children):
            return True

        return bool(node.children) and all(child.target in {"cactus.kv_cache_append", "cactus.attention_cached"} for child in node.children)

    return False


def lower_cache_placeholder(context: models.GenerationContext, node: IRModels.Node) -> Any:
    annotation = require_cache_annotation(node)
    return create_bound_cache_state(context, node, annotation)


def lower_decode_cache_cat(context: models.GenerationContext, node: IRModels.Node) -> Any | None:
    if len(node.parents) < 2:
        return None

    for role in (FModels.CacheTensorRole.KEY, FModels.CacheTensorRole.VALUE):
        match = IRModels.find_cache_concat_ancestor(node, role, max_depth=1)
        if match is not None and match.concat is node:
            return context.require(match.state.name)
    return None


def lower_prefill_cache_cat(context: models.GenerationContext, node: IRModels.Node) -> Any | None:
    annotation = context.prefill_cache_cat_annotations.get(node.name)

    if annotation is None or len(node.parents) < 2:
        return None

    new_kv_node = node.parents[1]
    original_new_kv = context.require(new_kv_node.name)
    new_kv = cache_new_kv_for_native(context, new_kv_node, original_new_kv, annotation.role or "")
    new_kv = cast_to_precision(context, new_kv, context.graph.FP16)
    cache_state = context.graph.kv_cache_state(
        kv_cache_capacity(context, annotation),
        required_cache_int(annotation.num_kv_heads, node, "num_kv_heads"),
        required_cache_int(annotation.head_dim, node, "head_dim"),
        window_size=int(annotation.window_size or 0),
        sink_size=0,
        num_slots=1,
    )
    record_cache_state_binding(context, node, cache_state, annotation)
    context.graph.kv_cache_append(new_kv, cache_state, window_size=int(annotation.window_size or 0), sink_size=0)
    context.prefill_cache_cat_states[node.name] = cache_state
    context.prefill_cache_cat_new_values[node.name] = new_kv
    return original_new_kv


def lower_attention(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if node.target == "cactus.attention" or node.target == "aten.scaled_dot_product_attention.default":
        cached = lower_cache_backed_attention(context, node)

        if cached is not None:
            return cached

        inputs = context.inputs_for(node)
        require_len(node, inputs, 3)
        query, key, value, mask = attention_inputs_for_layout(context, node, inputs)
        query = cast_to_precision(context, query, context.graph.FP16)
        key = cast_to_precision(context, key, context.graph.FP16)
        value = cast_to_precision(context, value, context.graph.FP16)
        mask = cast_to_precision(context, mask, context.graph.FP16) if mask is not None else None
        output = context.graph.attention(
            query,
            key,
            value,
            scale=attention_scale(node),
            is_causal=bool(node.attrs.get("is_causal", True)),
            position_offset=attention_position_offset(node),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask,
            additive_mask=bool(node.attrs.get("additive_mask", False)),
        )
        return attention_output_for_layout(context, node, output)

    if node.target == "cactus.attention_cached":
        inputs = context.inputs_for(node)
        require_len(node, inputs, 5)
        query, key_new, value_new = cached_attention_inputs_for_layout(context, node, inputs)
        mask = cached_attention_mask(context, node, inputs)
        output = context.graph.attention_cached(
            query,
            key_new,
            value_new,
            inputs[3],
            inputs[4],
            scale=attention_scale(node),
            position_offset=cached_attention_position_offset(context, node),
            window_size=int(node.attrs.get("window_size", 0)),
            v_head_dim=int(node.attrs.get("v_head_dim", 0)),
            is_causal=bool(node.attrs.get("is_causal", True)),
            mask=mask,
            additive_mask=bool(node.attrs.get("additive_mask", False)),
        )
        return attention_output_for_layout(context, node, output)

    raise UnsupportedLoweringError(f"{node.name}: unsupported attention target {node.target}")


def cached_attention_inputs_for_layout(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> tuple[Any, Any, Any]:
    if node.attrs.get("input_layout") == "bhqd_bhsd_bhsd":
        return (
            context.graph.permute(inputs[0], (0, 2, 1, 3)),
            context.graph.permute(inputs[1], (0, 2, 1, 3)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
        )

    return inputs[0], inputs[1], inputs[2]


def lower_cache_backed_attention(context: models.GenerationContext, node: IRModels.Node) -> Any | None:
    if node.target != "cactus.attention" or len(node.parents) < 3:
        return None

    key_match = find_attention_cache_concat_ancestor(context, node.parents[1], FModels.CacheTensorRole.KEY)
    value_match = find_attention_cache_concat_ancestor(context, node.parents[2], FModels.CacheTensorRole.VALUE)

    if key_match is None or value_match is None:
        return None

    key_cat, value_cat = key_match.concat, value_match.concat
    key_cache, value_cache = key_match.state, value_match.state
    key_new_node, value_new_node = key_match.new_value, value_match.new_value

    query = context.require(node.parents[0].name)
    input_layout = str(node.attrs.get("input_layout", ""))
    key_new = context.prefill_cache_cat_new_values.get(key_cat.name)
    if key_new is None:
        key_new = cache_new_kv_for_native(context, key_new_node, context.require(key_new_node.name), FModels.CacheTensorRole.KEY, input_layout)

    value_new = context.prefill_cache_cat_new_values.get(value_cat.name)
    if value_new is None:
        value_new = cache_new_kv_for_native(context, value_new_node, context.require(value_new_node.name), FModels.CacheTensorRole.VALUE, input_layout)
    key_cache_state = prefill_cache_state_for_cat(context, key_cat)
    value_cache_state = prefill_cache_state_for_cat(context, value_cat)

    if key_cache_state is None:
        key_cache_state = context.require(key_cache.name)

    if value_cache_state is None:
        value_cache_state = context.require(value_cache.name)

    if node.attrs.get("input_layout") in {"bhqd_bhds_bhsd", "bhqd_bhsd_bhsd"}:
        query = context.graph.permute(query, (0, 2, 1, 3))

    query = cast_to_precision(context, query, context.graph.FP16)
    key_new = cast_to_precision(context, key_new, context.graph.FP16)
    value_new = cast_to_precision(context, value_new, context.graph.FP16)
    mask = cache_backed_attention_mask(context, node)

    cache_pair = (key_cat.name, value_cat.name)
    prefill_pair_already_appended = key_cat.name in context.prefill_cache_cat_states and value_cat.name in context.prefill_cache_cat_states

    if cache_pair not in context.appended_cache_pairs and not prefill_pair_already_appended:
        context.graph.kv_cache_append(key_new, key_cache_state, window_size=cache_retention_window_size(node), sink_size=0)
        context.graph.kv_cache_append(value_new, value_cache_state, window_size=cache_retention_window_size(node), sink_size=0)
        context.appended_cache_pairs.add(cache_pair)

    context.values.setdefault(key_cat.name, key_cache_state)
    context.values.setdefault(value_cat.name, value_cache_state)

    output = context.graph.attention_cached(
        query,
        key_new,
        value_new,
        key_cache_state,
        value_cache_state,
        scale=attention_scale(node),
        position_offset=cached_attention_position_offset(context, node),
        window_size=int(node.attrs.get("window_size", 0)),
        v_head_dim=last_dim(value_new_node),
        is_causal=bool(node.attrs.get("is_causal", True)),
        mask=mask,
        additive_mask=bool(node.attrs.get("additive_mask", False)),
    )
    return attention_output_for_layout(context, node, output)


def cached_attention_mask(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> Any | None:
    if len(inputs) <= 5:
        return None

    return cast_to_precision(context, inputs[5], context.graph.FP16)


def cache_backed_attention_mask(context: models.GenerationContext, node: IRModels.Node) -> Any | None:
    if len(node.parents) <= 3:
        return None

    return cast_to_precision(context, context.require(node.parents[3].name), context.graph.FP16)


def find_attention_cache_cat_ancestor(context: models.GenerationContext, node: IRModels.Node, role: str) -> IRModels.Node | None:
    match = find_attention_cache_concat_ancestor(context, node, role)
    return match.concat if match is not None else None


def find_attention_cache_concat_ancestor(context: models.GenerationContext, node: IRModels.Node, role: str) -> IRModels.CacheConcatMatch | None:
    cache_match = find_cache_concat_ancestor(node, role)

    if cache_match is not None:
        return cache_match

    cache_cat = find_prefill_cache_cat_ancestor(context, node)
    if cache_cat is None or len(cache_cat.parents) < 2:
        return None
    # Empty prefill cache tensors are initializers rather than persistent
    # states; retain the established placeholder here for shared lowering.
    return IRModels.CacheConcatMatch(cache_cat, cache_cat.parents[0], cache_cat.parents[1])


def find_prefill_cache_cat_ancestor(context: models.GenerationContext, node: IRModels.Node, max_depth: int = 16) -> IRModels.Node | None:
    current = node

    for _ in range(max_depth):
        if current.name in context.prefill_cache_cat_annotations:
            return current

        if len(current.parents) != 1:
            return None

        current = current.parents[0]

    return None


def prefill_cache_state_for_cat(context: models.GenerationContext, node: IRModels.Node) -> Any | None:
    return context.prefill_cache_cat_states.get(node.name)


def cached_attention_position_offset(context: models.GenerationContext, node: IRModels.Node) -> int:
    if context.component.name == "decoder_step":
        return SIZE_T_MAX

    return int(node.attrs.get("position_offset", 0))


def cache_new_kv_for_native(context: models.GenerationContext, node: IRModels.Node, value: Any, role: str, input_layout: str = "") -> Any:
    if len(meta_shape(node)) == 4:
        return context.graph.permute(value, (0, 2, 1, 3))

    return value


def last_dim(node: IRModels.Node) -> int:
    shape = concrete_shape(meta_shape(node))

    if not shape:
        return 0

    return int(shape[-1])


def attention_scale(node: IRModels.Node) -> float:
    value = node.attrs.get("scale")

    if value is not None:
        return float(value)

    if not node.parents:
        return 1.0

    shape = meta_shape(node.parents[0])

    if not shape:
        return 1.0

    head_dim = concrete_dim(shape[-1])

    if head_dim is None or head_dim <= 0:
        return 1.0

    return 1.0 / math.sqrt(float(head_dim))


def attention_position_offset(node: IRModels.Node) -> int:
    value = int(node.attrs.get("position_offset", 0) or 0)
    if value >= (2**64 - 2):
        return 0

    return value


def attention_inputs_for_layout(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> tuple[Any, Any, Any, Any | None]:
    mask = inputs[3] if len(inputs) > 3 else None

    if node.attrs.get("input_layout") == "bhqd_bhds_bhsd":
        return (
            context.graph.permute(inputs[0], (0, 2, 1, 3)),
            context.graph.permute(inputs[1], (0, 3, 1, 2)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
            mask,
        )

    if node.attrs.get("input_layout") == "bqhd_bhds_bhsd":
        return (
            inputs[0],
            context.graph.permute(inputs[1], (0, 3, 1, 2)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
            mask,
        )

    if node.attrs.get("input_layout") == "bqhd_bhsd_bhsd":
        return (
            inputs[0],
            context.graph.permute(inputs[1], (0, 2, 1, 3)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
            mask,
        )

    if node.attrs.get("input_layout") == "bhqd_bhsd_bhsd":
        return (
            context.graph.permute(inputs[0], (0, 2, 1, 3)),
            context.graph.permute(inputs[1], (0, 2, 1, 3)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
            mask,
        )

    return inputs[0], inputs[1], inputs[2], mask


def attention_output_for_layout(context: models.GenerationContext, node: IRModels.Node, output: Any) -> Any:
    output_layout = node.attrs.get("output_layout")

    if output_layout == "bthd_flat":
        return context.graph.reshape(output, output_shape(node))

    if output_layout == "bhqd":
        return context.graph.permute(output, (0, 2, 1, 3))

    return output


def lower_cache(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.kv_cache_append":
        new_kv = cache_append_value_for_native(context, node, inputs[0]) if inputs else None

        if len(inputs) == 1:
            cache_state = create_bound_cache_state(context, node, require_cache_annotation(node))
            context.graph.kv_cache_append(new_kv, cache_state, window_size=cache_retention_window_size(node), sink_size=int(node.attrs.get("sink_size", 0)))
            return cache_state

        require_len(node, inputs, 2)
        require_cache_state_parent(node, 1, "cactus.kv_cache_state")
        context.graph.kv_cache_append(new_kv, inputs[1], window_size=cache_retention_window_size(node), sink_size=int(node.attrs.get("sink_size", 0)))
        return inputs[1]

    if target == "cactus.conv_cache_append":
        require_len(node, inputs, 2)
        require_cache_state_parent(node, 1, "cactus.conv_cache_state")
        window = context.graph.conv_cache_append(inputs[0], inputs[1])
        return conv_cache_window_to_ir_layout(context, node, window)

    if target == "cactus.conv_cache_initialize":
        if len(inputs) == 1:
            cache_state = create_bound_cache_state(context, node, require_cache_annotation(node))
            rows = conv_cache_rows_for_native(context, node, inputs[0])
            context.graph.conv_cache_initialize(rows, cache_state)
            return cache_state

        require_len(node, inputs, 2)
        rows = conv_cache_rows_for_native(context, node, inputs[0])
        context.graph.conv_cache_initialize(rows, inputs[1])
        return inputs[1]

    if target == "cactus.recurrent_cache_write":
        require_len(node, inputs, 2)
        return context.graph.recurrent_cache_write(inputs[0], inputs[1])

    if target == "cactus.kv_cache_state":
        return context.graph.kv_cache_state(
            required_int_attr(node, "max_seq_len"),
            required_int_attr(node, "num_kv_heads"),
            required_int_attr(node, "head_dim"),
            window_size=int(node.attrs.get("window_size", 0)),
            sink_size=int(node.attrs.get("sink_size", 0)),
            num_slots=int(node.attrs.get("num_slots", 1)),
        )

    if target == "cactus.conv_cache_state":
        return context.graph.conv_cache_state(required_int_attr(node, "window_size"), required_int_attr(node, "hidden_dim"))

    if target == "cactus.recurrent_cache_state":
        return context.graph.recurrent_cache_state(shape_attr(node), dtype=cactus_precision(context.graph, tensor_dtype(node)))

    raise UnsupportedLoweringError(f"{node.name}: unsupported cache target {target}")


def cache_append_value_for_native(context: models.GenerationContext, node: IRModels.Node, value: Any) -> Any:
    annotation = node.cache

    if annotation is None and len(node.parents) > 1:
        annotation = node.parents[1].cache

    if annotation is None or annotation.layout != "batch_heads_sequence_head_dim":
        return value

    if not node.parents or len(meta_shape(node.parents[0])) != 4:
        return value

    return context.graph.permute(value, (0, 2, 1, 3))


def cache_retention_window_size(node: IRModels.Node) -> int:
    return int(node.attrs.get("cache_window_size", node.attrs.get("window_size", 0)) or 0)


def require_cache_state_parent(node: IRModels.Node, parent_index: int, expected_target: str) -> None:
    if parent_index >= len(node.parents):
        raise UnsupportedLoweringError(f"{node.name}: {node.target} missing cache parent {parent_index}")

    parent = node.parents[parent_index]

    if parent.target == expected_target:
        return

    if expected_target == "cactus.conv_cache_state" and parent.cache is not None and parent.cache.kind == FModels.CacheKind.CONV:
        return

    if expected_target == "cactus.kv_cache_state" and parent.cache is not None and parent.cache.kind == FModels.CacheKind.KV:
        return

    raise UnsupportedLoweringError(
        f"{node.name}: {node.target} requires parent {parent_index} to be {expected_target}; got {parent.target}. "
        "Raw HF cache tensors need a cache-state bridge before native cache append lowering."
    )


def require_cache_annotation(node: IRModels.Node) -> IRModels.CacheAnnotation:
    if node.cache is None:
        raise UnsupportedLoweringError(f"{node.name}: cache lowering requires cache annotation metadata")

    return node.cache


def required_cache_int(value: int | None, node: IRModels.Node, name: str) -> int:
    if value is None:
        raise UnsupportedLoweringError(f"{node.name}: cache lowering missing {name}")

    return int(value)


def kv_cache_capacity(context: models.GenerationContext, annotation: IRModels.CacheAnnotation) -> int:
    capacity = int(annotation.sequence_length or 1)
    profile_capacity = profile_cache_sequence_length(context)
    decode_extra = 1 if context.component.ir_graph.task == "decode_with_cache" else 0

    if profile_capacity > 0:
        return max(profile_capacity, capacity + decode_extra, 1)

    return max(capacity + decode_extra, 1)


def profile_cache_sequence_length(context: models.GenerationContext) -> int:
    cache_contract = getattr(context.model_profile, "cache_contract", None)
    value = getattr(cache_contract, "max_cache_sequence_length", 0) if cache_contract is not None else 0

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def create_bound_cache_state(
    context: models.GenerationContext,
    node: IRModels.Node,
    annotation: IRModels.CacheAnnotation,
) -> Any:
    if annotation.kind == FModels.CacheKind.KV:
        cache_state = context.graph.kv_cache_state(
            kv_cache_capacity(context, annotation),
            required_cache_int(annotation.num_kv_heads, node, "num_kv_heads"),
            required_cache_int(annotation.head_dim, node, "head_dim"),
            window_size=int(annotation.window_size or 0),
            sink_size=0,
            num_slots=1,
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    if annotation.kind == FModels.CacheKind.CONV:
        cache_state = context.graph.conv_cache_state(
            required_cache_int(annotation.window_size, node, "window_size"),
            required_cache_int(annotation.hidden_dim, node, "hidden_dim"),
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    raise UnsupportedLoweringError(f"{node.name}: unsupported cache kind {annotation.kind}")


def record_cache_state_binding(context: models.GenerationContext, node: IRModels.Node, cache_state: Any, annotation: IRModels.CacheAnnotation) -> None:
    node_id = models.tensor_node_id(cache_state)

    if node_id is None:
        raise UnsupportedLoweringError(f"{node.name}: cache state tensor does not expose a Cactus node id")

    context.component.add_cache_state_binding(RPModels.cache_state_binding_from_annotation(annotation, node_id))


def conv_cache_rows_for_native(context: models.GenerationContext, node: IRModels.Node, rows: Any) -> Any:
    rows_shape = meta_shape(node.parents[0]) if node.parents else output_shape(node)

    if len(rows_shape) == 3:
        batch, hidden_dim, window_size = rows_shape

        if batch != 1:
            raise UnsupportedLoweringError(f"{node.name}: conv cache initialize currently supports batch size 1")

        return context.graph.reshape(context.graph.permute(rows, (0, 2, 1)), (int(window_size), int(hidden_dim)))

    if len(rows_shape) == 2:
        return rows

    raise UnsupportedLoweringError(f"{node.name}: conv cache initialize rows must be rank 2 or 3, got shape {rows_shape}")


def conv_cache_window_to_ir_layout(context: models.GenerationContext, node: IRModels.Node, window: Any) -> Any:
    target_shape = output_shape(node)

    if len(target_shape) != 3:
        return window

    batch, hidden_dim, window_size = target_shape

    if batch != 1:
        raise UnsupportedLoweringError(f"{node.name}: conv cache append layout bridge currently supports batch size 1")

    return context.graph.reshape(context.graph.transpose(window), (int(batch), int(hidden_dim), int(window_size)))
