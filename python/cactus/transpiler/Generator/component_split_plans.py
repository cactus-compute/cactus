from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .component_split_boundaries import *
from .component_split_builder import extract_component_graph
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

    if is_causal_lm_profile(model_profile):
        return split_causal_lm_components(prefill, decode)

    return None


def split_gemma4_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    boundaries = find_gemma4_boundaries(prefill, decode)
    position_step = PlaceholderSpec("position_ids", "position_ids", source_node=boundaries.decode_position_ids)
    position_chunk = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids")
    inputs_embeds_step = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.decode_text_inputs_embeds)
    inputs_embeds_chunk = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.merged_inputs_embeds)
    per_layer_step = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.decode_per_layer_inputs)
    per_layer_chunk = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.prefill_per_layer_inputs)

    specs = (
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
            chunk_tokens=GEMMA4_PREFILL_CHUNK_TOKENS,
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
            chunk_tokens=GEMMA4_PREFILL_CHUNK_TOKENS,
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
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=(OutputSpec(boundaries.prefill_logits, "logits"),),
            placeholders=(inputs_embeds_chunk, per_layer_chunk, position_chunk),
            chunk_tokens=GEMMA4_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=(OutputSpec(boundaries.decode_logits, "logits"),),
            placeholders=(inputs_embeds_step, per_layer_step, position_step),
        ),
    )

    return {spec.name: extract_component_graph(spec) for spec in specs}


def is_gemma4_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return "gemma4" in profile_name or "gemma_4" in profile_name


def is_whisper_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return "whisper" in profile_name


def is_causal_lm_profile(model_profile: Any | None) -> bool:
    return str(getattr(model_profile, "load_strategy", "") or "") == "causal_lm"


def split_causal_lm_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    step_position = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids", force=True)
    chunk_position = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids", force=True)

    specs = (
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
            name="lm_encoder_text_chunk",
            graph=prefill,
            outputs=(
                OutputSpec("input_ids", "inputs_embeds"),
                OutputSpec(chunk_position.name, "position_ids"),
            ),
            placeholders=(chunk_position,),
            chunk_tokens=GENERIC_CAUSAL_PREFILL_CHUNK_TOKENS,
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=graph_output_specs(decode),
            placeholders=(step_position,),
            input_aliases={"input_ids": "inputs_embeds"},
        ),
        ComponentSplitSpec(
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=graph_output_specs(prefill),
            placeholders=(chunk_position,),
            input_aliases={"input_ids": "inputs_embeds"},
            chunk_tokens=GENERIC_CAUSAL_PREFILL_CHUNK_TOKENS,
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
                    OutputSpec(key_node, f"cross_k_{layer_index}"),
                    OutputSpec(value_node, f"cross_v_{layer_index}"),
                )
            ),
            placeholders=(hidden,),
            metadata={**route_metadata, "runtime_role": "decoder_cross_kv"},
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=(OutputSpec(boundaries.decode_logits, "logits"),),
            placeholders=(position,),
            ref_aliases={position_index: "position_ids"},
            input_aliases=decoder_input_aliases,
            metadata={**route_metadata, "runtime_role": "decoder_step"},
        ),
    )

    return {spec.name: extract_component_graph(spec) for spec in specs}
