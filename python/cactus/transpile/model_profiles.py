from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import torch


@dataclass(frozen=True)
class ModelProfile:
    family: str
    model_types: tuple[str, ...] = ()
    transformer_module: str | None = None
    multimodal_context_tokens: int | None = None
    input_combinations: tuple[tuple[str, ...], ...] = ()
    model_id_aliases: tuple[tuple[str, str], ...] = ()
    model_id_markers: tuple[str, ...] = ()
    family_aliases: tuple[str, ...] = ()
    stop_tokens: tuple[str, ...] = ()
    avoid_native_loader: bool = False
    cached_step_components: tuple[str, ...] = ()
    cached_step_skip_components: tuple[str, ...] = ()
    dynamic_media_step_component: str | None = None
    fp16_kv_cache_components: tuple[str, ...] = ()
    ir_replay_components: tuple[str, ...] = ()
    prompt_style: str = "auto"
    multimodal_preprocessor: str = "auto"
    text_only_component: str | None = None
    default_task: str | None = None
    default_components: tuple[str, ...] = ()
    text_component_plans: tuple[tuple[str, tuple[str, ...]], ...] = ()
    default_max_new_tokens: int | None = None
    needs_image: bool = False
    needs_audio: bool = False
    force_component_pipeline: bool = False
    aliases: tuple[tuple[str, str], ...] = ()
    regex_aliases: tuple[tuple[str, str], ...] = ()


GEMMA4_PROFILE = ModelProfile(
    family="gemma4",
    model_types=("gemma4",),
    transformer_module="transformers.models.gemma4.modeling_gemma4",
    multimodal_context_tokens=2048,
    input_combinations=((), ("image",), ("audio",), ("image", "audio")),
    model_id_aliases=(
        ("gemma4", "google/gemma-4-E2B-it"),
        ("gemma4-e2b", "google/gemma-4-E2B-it"),
    ),
    model_id_markers=("gemma-4", "gemma4"),
    family_aliases=("gemma",),
    stop_tokens=("<turn|>", "<eos>"),
    avoid_native_loader=True,
    cached_step_components=("lm_encoder_step", "decoder_step"),
    cached_step_skip_components=("decoder",),
    dynamic_media_step_component="lm_encoder_media_step",
    fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
    ir_replay_components=(
        "vision_encoder",
        "audio_encoder",
        "decoder_prefill_chunk",
        "decoder_step",
    ),
    prompt_style="gemma4",
    multimodal_preprocessor="gemma4",
    default_task="multimodal_causal_lm_logits",
    default_components=("vision_encoder", "audio_encoder", "lm_encoder", "decoder"),
    needs_image=True,
    needs_audio=True,
    force_component_pipeline=True,
)


LFM2_PROFILE = ModelProfile(
    family="lfm2",
    model_types=("lfm2_vl", "lfm2", "lfm2_moe"),
    multimodal_context_tokens=256,
    input_combinations=((), ("image",)),
    model_id_aliases=(("lfm", "LiquidAI/LFM2-VL-450M"),),
    model_id_markers=("lfm2-vl", "lfm-vl"),
    family_aliases=("lfm2_vl", "lfm"),
    stop_tokens=("<|im_end|>",),
    avoid_native_loader=True,
    cached_step_components=("lm_encoder_step", "decoder_step"),
    cached_step_skip_components=("decoder",),
    fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
    ir_replay_components=("decoder_prefill_chunk", "decoder_step"),
    prompt_style="lfm_chat",
    multimodal_preprocessor="lfm2_vl",
    text_only_component="text_lm_encoder",
    default_task="multimodal_causal_lm_logits",
    default_components=("vision_encoder", "lm_encoder", "decoder"),
    text_component_plans=(
        ("lfm2forcausallm", ("decoder_step", "lm_encoder_step", "lm_encoder_text_chunk",
                             "decoder_prefill_chunk")),
        ("lfm2moeforcausallm", ("decoder_step", "lm_encoder_step", "lm_encoder_text_chunk",
                                "decoder_prefill_chunk")),
    ),
    needs_image=True,
    force_component_pipeline=True,
)


PARAKEET_TDT_PROFILE = ModelProfile(
    family="parakeet_tdt",
    model_types=("parakeet_tdt",),
    model_id_aliases=(
        ("parakeet", "nvidia/parakeet-tdt-0.6b-v3"),
        ("parakeet-tdt", "nvidia/parakeet-tdt-0.6b-v3"),
    ),
    model_id_markers=("parakeet-tdt",),
    avoid_native_loader=True,
    default_task="tdt_transcription",
    default_components=("audio_encoder", "decoder"),
    needs_audio=True,
    force_component_pipeline=True,
    aliases=(
        ("decoder.prediction.embed.weight", "decoder.embedding.weight"),
        ("joint.enc.weight", "encoder_projector.weight"),
        ("joint.enc.bias", "encoder_projector.bias"),
        ("joint.pred.weight", "decoder.decoder_projector.weight"),
        ("joint.pred.bias", "decoder.decoder_projector.bias"),
        ("joint.joint_net.2.weight", "joint.head.weight"),
        ("joint.joint_net.2.bias", "joint.head.bias"),
        ("encoder.pre_encode.out.weight", "encoder.subsampling.linear.weight"),
        ("encoder.pre_encode.out.bias", "encoder.subsampling.linear.bias"),
    ),
    regex_aliases=(
        (
            r"encoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight",
            r"encoder.layers.\1.self_attn.linear_\2.weight",
        ),
        (
            r"encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight",
            r"encoder.layers.\1.self_attn.linear_out.weight",
        ),
        (
            r"encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.weight",
            r"encoder.layers.\1.self_attn.linear_pos.weight",
        ),
        (
            r"encoder\.layers\.(\d+)\.self_attn\.bias_u",
            r"encoder.layers.\1.self_attn.pos_bias_u",
        ),
        (
            r"encoder\.layers\.(\d+)\.self_attn\.bias_v",
            r"encoder.layers.\1.self_attn.pos_bias_v",
        ),
        (
            r"encoder\.layers\.(\d+)\.conv\.norm\.(.+)",
            r"encoder.layers.\1.conv.batch_norm.\2",
        ),
    ),
)


WHISPER_PROFILE = ModelProfile(
    family="whisper",
    model_types=("whisper",),
    model_id_aliases=(("whisper", "openai/whisper-small"),),
    model_id_markers=("whisper",),
    family_aliases=("whisperforconditionalgeneration",),
    avoid_native_loader=True,
    cached_step_components=("decoder_step",),
    cached_step_skip_components=("decoder",),
    fp16_kv_cache_components=("decoder_step",),
    default_task="seq2seq_transcription",
    default_components=("audio_encoder", "decoder"),
    needs_audio=True,
    force_component_pipeline=True,
)


QWEN_PROFILE = ModelProfile(
    family="qwen",
    model_types=("qwen", "qwen2", "qwen2_moe", "qwen3", "qwen3_5", "qwen3.5", "qwen3_vl"),
    multimodal_context_tokens=512,
    input_combinations=((), ("image",)),
    model_id_aliases=(
        ("qwen", "Qwen/Qwen3.5-0.8B"),
        ("qwen3.5", "Qwen/Qwen3.5-0.8B"),
        ("qwen35", "Qwen/Qwen3.5-0.8B"),
        ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"),
        ("qwen3", "Qwen/Qwen3-1.7B"),
    ),
    model_id_markers=("qwen",),
    family_aliases=("qwen2", "qwen2_moe", "qwen3", "qwen3_5", "qwen3.5", "qwen3_vl"),
    stop_tokens=("<|im_end|>",),
    avoid_native_loader=True,
    cached_step_components=("decoder_media_step",),
    cached_step_skip_components=("decoder",),
    fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_media_step", "decoder_step"),
    prompt_style="qwen_chat",
    multimodal_preprocessor="qwen3_5",
    default_task="multimodal_causal_lm_logits",
    default_components=(
        "vision_encoder",
        "lm_encoder",
        "decoder",
        "lm_encoder_text_chunk",
        "decoder_prefill_chunk",
        "lm_encoder_step",
        "decoder_media_step",
        "decoder_step",
    ),
    text_component_plans=(
        ("qwen3forcausallm", ("decoder_step", "lm_encoder_step", "lm_encoder_text_chunk",
                              "decoder_prefill_chunk", "decoder_embed_chunk", "decoder_media_step")),
    ),
    needs_image=True,
    force_component_pipeline=True,
)


NEEDLE_PROFILE = ModelProfile(
    family="needle",
    model_types=("needle",),
    model_id_markers=("needle",),
    avoid_native_loader=True,
    default_task="causal_lm_logits",
    default_components=("source_encoder", "decoder_cross_kv", "decoder_step"),
    default_max_new_tokens=1022,
    force_component_pipeline=True,
)


PROFILES: tuple[ModelProfile, ...] = (
    GEMMA4_PROFILE,
    LFM2_PROFILE,
    PARAKEET_TDT_PROFILE,
    WHISPER_PROFILE,
    QWEN_PROFILE,
    NEEDLE_PROFILE,
)


def profile_for_model_type(model_type: str) -> ModelProfile | None:
    normalized = str(model_type or "").strip().lower()
    for profile in PROFILES:
        if normalized in profile.model_types:
            return profile
    return None


def profile_for_family(family: str) -> ModelProfile | None:
    normalized = str(family or "").strip().lower()
    for profile in PROFILES:
        if normalized == profile.family or normalized in profile.family_aliases:
            return profile
    return None


def profile_for_model_id(model_id: str) -> ModelProfile | None:
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return None
    alias_map = model_id_alias_map()
    normalized = alias_map.get(normalized, normalized).lower()
    for profile in PROFILES:
        for _, profile_model_id in profile.model_id_aliases:
            if normalized == profile_model_id.lower():
                return profile
        if any(marker and marker in normalized for marker in profile.model_id_markers):
            return profile
    return None


def model_id_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for profile in PROFILES:
        for alias, model_id in profile.model_id_aliases:
            aliases[str(alias).strip().lower()] = str(model_id).strip()
    return aliases


def multimodal_context_tokens_for_model_type(model_type: str, default: int) -> int:
    profile = profile_for_model_type(model_type)
    if profile is not None and profile.multimodal_context_tokens is not None:
        return max(0, int(profile.multimodal_context_tokens))
    return max(0, int(default))


def add_tensor_aliases(
    state_dict: dict[str, torch.Tensor],
    profile: ModelProfile,
    *,
    derived_aliases: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> dict[str, torch.Tensor]:
    def alias(target: str, source: str) -> None:
        if target not in state_dict and source in state_dict:
            state_dict[target] = state_dict[source]

    for target, source in profile.aliases:
        alias(target, source)

    for source_key in list(state_dict):
        for source_pattern, target_template in profile.regex_aliases:
            match = re.fullmatch(source_pattern, source_key)
            if match is None:
                continue
            alias(match.expand(target_template), source_key)

    if derived_aliases is not None:
        derived_aliases(state_dict)

    return state_dict
