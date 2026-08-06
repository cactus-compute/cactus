from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .component_split_boundaries import *
from .component_split_builder import extract_component_graph, retarget_whisper_decoder_cross_kv_layout
from .component_split_types import *
from ..IR import models as IRModels

def split_component_graphs(
    graphs: Mapping[str, IRModels.Graph],
    model_profile: Any | None = None,
) -> dict[str, IRModels.Graph] | None:
    prefill = find_graph_by_task(graphs, PREFILL_WITH_CACHE_TASK)
    decode = find_graph_by_task(graphs, DECODE_WITH_CACHE_TASK)

    if prefill is None or decode is None:
        return None

    if is_gemma4_profile(model_profile):
        return split_gemma4_components(prefill, decode)

    if is_whisper_profile(model_profile):
        return split_whisper_components(prefill, decode)

    if is_lfm_vlm_profile(model_profile):
        return split_lfm_vlm_components(prefill, decode)

    if is_causal_lm_profile(model_profile):
        return split_causal_lm_components(prefill, decode, model_profile)

    return None

def split_gemma4_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    boundaries = find_gemma4_boundaries(prefill, decode)
    position_step = PlaceholderSpec("position_ids", "position_ids", source_node=boundaries.decode_position_ids)
    position_chunk = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids")
    inputs_embeds_step = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.decode_text_inputs_embeds)
    inputs_embeds_chunk = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.merged_inputs_embeds)
    per_layer_step = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.decode_per_layer_inputs)
    per_layer_chunk = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.prefill_per_layer_inputs)
    prefill_rope_position_aliases = gemma4_rope_position_aliases(prefill, position_chunk.name)
    decode_rope_position_aliases = gemma4_rope_position_aliases(decode, position_step.name)

    step_specs = (
        ComponentSplitSpec(
            name="vision_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.vision_features, "image_features", row_limit=256),),
            input_aliases={"image_position_ids": "pixel_position_ids"},
        ),
        ComponentSplitSpec(
            name="audio_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.audio_features, "audio_features"),),
        ),
        ComponentSplitSpec(
            name="lm_encoder_step",
            graph=decode,
            outputs=(
                OutputSpec(boundaries.decode_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(boundaries.decode_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_step.name, "position_ids"),
            ),
            placeholders=(position_step,),
            ref_aliases={boundaries.decode_token_ids: "input_ids"},
        ),
        ComponentSplitSpec(
            name="lm_encoder_text_prefill_chunk",
            graph=prefill,
            outputs=(
                OutputSpec(boundaries.prefill_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(boundaries.prefill_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_chunk.name, "position_ids"),
            ),
            placeholders=(position_chunk,),
            ref_aliases={
                boundaries.merged_inputs_embeds: boundaries.prefill_text_inputs_embeds,
                boundaries.prefill_token_ids: "input_ids",
            },
            chunk_tokens=GEMMA4_TEXT_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="lm_encoder_text_chunk",
            graph=prefill,
            outputs=(
                OutputSpec(boundaries.prefill_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(boundaries.prefill_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_chunk.name, "position_ids"),
            ),
            placeholders=(position_chunk,),
            ref_aliases={
                boundaries.merged_inputs_embeds: boundaries.prefill_text_inputs_embeds,
                boundaries.prefill_token_ids: "input_ids",
            },
            chunk_tokens=GEMMA4_MEDIA_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="lm_encoder_media_chunk",
            graph=prefill,
            outputs=(
                OutputSpec(inputs_embeds_chunk.name, "inputs_embeds"),
                OutputSpec(boundaries.prefill_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_chunk.name, "position_ids"),
            ),
            placeholders=(inputs_embeds_chunk, position_chunk),
            ref_aliases={
                boundaries.merged_inputs_embeds: inputs_embeds_chunk.name,
                boundaries.prefill_token_ids: "input_ids",
            },
            chunk_tokens=GEMMA4_MEDIA_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="lm_encoder_media_step",
            graph=decode,
            outputs=(
                OutputSpec(inputs_embeds_step.name, "inputs_embeds"),
                OutputSpec(boundaries.decode_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_step.name, "position_ids"),
            ),
            placeholders=(inputs_embeds_step, position_step),
            ref_aliases={boundaries.decode_token_ids: "input_ids"},
        ),
        ComponentSplitSpec(
            name="decoder_prefill_text_chunk",
            graph=prefill,
            outputs=(OutputSpec(boundaries.prefill_logits, "logits"),),
            side_effects=cache_side_effect_node_names(prefill),
            placeholders=(inputs_embeds_chunk, per_layer_chunk, position_chunk),
            ref_aliases=prefill_rope_position_aliases,
            chunk_tokens=GEMMA4_TEXT_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=(OutputSpec(boundaries.prefill_logits, "logits"),),
            side_effects=cache_side_effect_node_names(prefill),
            placeholders=(inputs_embeds_chunk, per_layer_chunk, position_chunk),
            ref_aliases=prefill_rope_position_aliases,
            chunk_tokens=GEMMA4_MEDIA_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=(OutputSpec(boundaries.decode_logits, "logits"),),
            side_effects=cache_side_effect_node_names(decode),
            placeholders=(inputs_embeds_step, per_layer_step, position_step),
            ref_aliases=decode_rope_position_aliases,
        ),
    )

    return {spec.name: extract_component_graph(spec) for spec in step_specs}

def gemma4_rope_position_aliases(graph: IRModels.Graph, placeholder_name: str) -> dict[str, str]:
    aliases: dict[str, str] = {}

    for node in graph.nodes:
        if node.target != "cactus.gemma4_rope_table_lookup":
            continue

        if not node.parents:
            continue

        aliases[node.parents[0].name] = placeholder_name

    return aliases

def is_gemma4_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return "gemma4" in profile_name or "gemma_4" in profile_name

def is_whisper_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return "whisper" in profile_name

def is_lfm_vlm_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return profile_name in {"lfm_vlm", "lfm2_vl", "lfm-vlm"} or "lfm_vlm" in profile_name or "lfm2_vl" in profile_name

def is_causal_lm_profile(model_profile: Any | None) -> bool:
    return str(getattr(model_profile, "load_strategy", "") or "") == "causal_lm"

def split_causal_lm_components(
    prefill: IRModels.Graph,
    decode: IRModels.Graph,
    model_profile: Any | None = None,
) -> dict[str, IRModels.Graph]:
    step_position = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids", force=True)
    chunk_position = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids", force=True)
    chunk_tokens = causal_lm_prefill_chunk_tokens(model_profile)

    step_specs = (
        ComponentSplitSpec(
            name="lm_encoder_step",
            graph=decode,
            outputs=(
                OutputSpec("input_ids", "inputs_embeds"),
                OutputSpec(step_position.name, "position_ids"),
            ),
            placeholders=(step_position,),
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=graph_output_specs(decode, publish_only_logits=True),
            side_effects=cache_side_effect_node_names(decode),
            placeholders=(step_position,),
            input_aliases={"input_ids": "inputs_embeds"},
        ),
    )
    prefill_specs = (
        ComponentSplitSpec(
            name="lm_encoder_text_prefill_chunk",
            graph=prefill,
            outputs=(
                OutputSpec("input_ids", "inputs_embeds"),
                OutputSpec(chunk_position.name, "position_ids"),
            ),
            placeholders=(chunk_position,),
            chunk_tokens=chunk_tokens,
        ),
        ComponentSplitSpec(
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=graph_output_specs(prefill, publish_only_logits=True),
            side_effects=cache_side_effect_node_names(prefill),
            placeholders=(chunk_position,),
            input_aliases={"input_ids": "inputs_embeds"},
            chunk_tokens=chunk_tokens,
        ),
    )

    cache_policy = tuple(getattr(model_profile, "cache_policy", ()) or ())
    specs = step_specs if "scalar_prefill" in cache_policy else (*step_specs, *prefill_specs)

    return {spec.name: extract_component_graph(spec) for spec in specs}

def causal_lm_prefill_chunk_tokens(model_profile: Any | None) -> int:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()

    if "lfm" in profile_name:
        return LFM_MOE_PREFILL_CHUNK_TOKENS

    return GENERIC_CAUSAL_PREFILL_CHUNK_TOKENS

def split_lfm_vlm_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    boundaries = find_lfm_vlm_boundaries(prefill, decode)
    position_step = PlaceholderSpec("position_ids", "position_ids", source_node=boundaries.decode_position_ids)
    position_chunk = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids")
    attention_mask_chunk = PlaceholderSpec("attention_mask", "attention_mask", tensor_node="input_ids", force=True)
    inputs_embeds_step = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.decode_text_inputs_embeds)
    inputs_embeds_chunk = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.merged_inputs_embeds)
    positional_embeddings = PlaceholderSpec(
        "positional_embeddings",
        "positional_embeddings",
        source_node=boundaries.vision_position_embeddings,
    )
    vision_features = PlaceholderSpec(
        "vision_features",
        "vision_features",
        source_node=boundaries.vision_projector_input,
    )

    specs = (
        ComponentSplitSpec(
            name="vision_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.vision_encoder_features, "vision_features"),),
            placeholders=(positional_embeddings,),
            metadata={
                "position_embedding_grid_path": "lfm2_vl_position_embedding_grid.f32",
                "position_embedding_grid_shape": "16,16,1152",
            },
        ),
        ComponentSplitSpec(
            name="vision_projector",
            graph=prefill,
            outputs=(OutputSpec(boundaries.image_features, "image_features"),),
            placeholders=(vision_features,),
        ),
        ComponentSplitSpec(
            name="lm_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.prefill_text_inputs_embeds, "inputs_embeds"),),
            placeholders=(position_chunk,),
            chunk_tokens=LFM_VLM_MEDIA_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="lm_encoder_text_prefill_chunk",
            graph=prefill,
            outputs=(
                OutputSpec(boundaries.prefill_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(position_chunk.name, "position_ids"),
            ),
            placeholders=(position_chunk,),
            chunk_tokens=LFM_VLM_TEXT_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="lm_encoder_step",
            graph=decode,
            outputs=(
                OutputSpec(boundaries.decode_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(position_step.name, "position_ids"),
            ),
            placeholders=(position_step,),
        ),
        ComponentSplitSpec(
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=graph_output_specs(prefill, publish_only_logits=True),
            side_effects=cache_side_effect_node_names(prefill),
            placeholders=(inputs_embeds_chunk, attention_mask_chunk, position_chunk),
            ref_aliases={boundaries.merged_inputs_embeds: inputs_embeds_chunk.name},
            input_aliases={"attention_mask": "attention_mask"},
            chunk_tokens=LFM_VLM_MEDIA_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_prefill_text_chunk",
            graph=prefill,
            outputs=graph_output_specs(prefill, publish_only_logits=True),
            side_effects=cache_side_effect_node_names(prefill),
            placeholders=(inputs_embeds_chunk, attention_mask_chunk, position_chunk),
            ref_aliases={boundaries.merged_inputs_embeds: inputs_embeds_chunk.name},
            input_aliases={"attention_mask": "attention_mask"},
            chunk_tokens=LFM_VLM_TEXT_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=graph_output_specs(decode, publish_only_logits=True),
            side_effects=cache_side_effect_node_names(decode),
            placeholders=(inputs_embeds_step, position_step),
            ref_aliases={boundaries.decode_text_inputs_embeds: inputs_embeds_step.name},
        ),
    )

    return {spec.name: extract_component_graph(spec) for spec in specs}

def split_whisper_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    boundaries = find_whisper_boundaries(prefill, decode)
    position_index = find_whisper_decode_position_index(decode)
    hidden = PlaceholderSpec(
        "encoder_hidden_states",
        "encoder_hidden_states",
        source_node=boundaries.encoder_hidden_states,
    )
    position = PlaceholderSpec(
        "position_ids",
        "position_ids",
        tensor_node=position_index,
    )
    route_metadata = {"runtime_route": "encoder_cross_kv_decoder_step"}
    decoder_input_aliases = whisper_decoder_cross_input_aliases(decode, len(boundaries.cross_key_values))

    specs = (
        ComponentSplitSpec(
            name="audio_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.encoder_hidden_states, "encoder_hidden_states"),),
            metadata={
                **route_metadata,
                "runtime_role": "source_encoder",
                "source_kind": "audio_features",
            },
        ),
        ComponentSplitSpec(
            name="decoder_cross_kv",
            graph=prefill,
            outputs=tuple(
                output
                for layer_index, (key_node, value_node) in enumerate(boundaries.cross_key_values)
                for output in (
                    whisper_cross_kv_output_spec(prefill, key_node, f"cross_k_{layer_index}"),
                    whisper_cross_kv_output_spec(prefill, value_node, f"cross_v_{layer_index}"),
                )
            ),
            placeholders=(hidden,),
            metadata={**route_metadata, "runtime_role": "decoder_cross_kv"},
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=(OutputSpec(boundaries.decode_logits, "logits"),),
            side_effects=cache_side_effect_node_names(decode),
            placeholders=(position,),
            ref_aliases={position_index: "position_ids"},
            input_aliases=decoder_input_aliases,
            metadata={**route_metadata, "runtime_role": "decoder_step"},
        ),
    )

    components = {spec.name: extract_component_graph(spec) for spec in specs}
    components["decoder_step"] = retarget_whisper_decoder_cross_kv_layout(components["decoder_step"])
    return components

def whisper_cross_kv_output_spec(graph: IRModels.Graph, node_name: str, logical_name: str) -> OutputSpec:
    node = graph.nodes_map[node_name]
    permutation = (0, 2, 1, 3) if tensor_shape(node) == [1, 6, 1500, 64] else None
    return OutputSpec(node_name, logical_name, permutation=permutation)

def cache_side_effect_node_names(graph: IRModels.Graph) -> tuple[str, ...]:
    return tuple(
        node.name
        for node in graph.nodes
        if node.target in {
            "cactus.kv_cache_append",
            "cactus.conv_cache_append",
            "cactus.conv_cache_initialize",
            "cactus.recurrent_cache_write",
        }
    )
