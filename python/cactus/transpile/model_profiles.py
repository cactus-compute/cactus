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
    aliases: tuple[tuple[str, str], ...] = ()
    regex_aliases: tuple[tuple[str, str], ...] = ()


GEMMA4_PROFILE = ModelProfile(
    family="gemma4",
    model_types=("gemma4",),
    transformer_module="transformers.models.gemma4.modeling_gemma4",
    multimodal_context_tokens=2048,
)


LFM2_VL_PROFILE = ModelProfile(
    family="lfm2_vl",
    model_types=("lfm2_vl",),
    multimodal_context_tokens=512,
)


PARAKEET_TDT_PROFILE = ModelProfile(
    family="parakeet_tdt",
    model_types=("parakeet_tdt",),
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


PROFILES: tuple[ModelProfile, ...] = (
    GEMMA4_PROFILE,
    LFM2_VL_PROFILE,
    PARAKEET_TDT_PROFILE,
)


def profile_for_model_type(model_type: str) -> ModelProfile | None:
    normalized = str(model_type or "").strip().lower()
    for profile in PROFILES:
        if normalized in profile.model_types:
            return profile
    return None


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
