from __future__ import annotations

from .models import Components


TEXT_EMBED = Components(
    name="text_embed",
    patterns=("embed_tokens", "embedding"),
)

VISION_ENCODER = Components(
    name="vision_encoder",
    patterns=("vision", "image"),
)

AUDIO_ENCODER = Components(
    name="audio_encoder",
    patterns=("audio",),
)

TOKEN_MERGE = Components(
    name="token_merge",
    patterns=("multi_modal", "multimodal", "projector", "merge"),
)

TEXT_DECODER = Components(
    name="text_decoder",
    patterns=("language_model", "text", "decoder", "model.layers"),
)

LM_HEAD = Components(
    name="lm_head",
    patterns=("lm_head", "logits"),
)


TEXT_COMPONENTS = {
    TEXT_EMBED.name: TEXT_EMBED,
    TEXT_DECODER.name: TEXT_DECODER,
    LM_HEAD.name: LM_HEAD,
}

MULTIMODAL_COMPONENTS = {
    **TEXT_COMPONENTS,
    VISION_ENCODER.name: VISION_ENCODER,
    AUDIO_ENCODER.name: AUDIO_ENCODER,
    TOKEN_MERGE.name: TOKEN_MERGE,
}
