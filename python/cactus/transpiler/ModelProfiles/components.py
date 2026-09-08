from __future__ import annotations

from .models import Components

TEXT_EMBED = Components(name="text_embed", patterns=("embed_tokens", "embedding"))
VISION_ENCODER = Components(name="vision_encoder", patterns=("vision", "image"))
AUDIO_ENCODER = Components(name="audio_encoder", patterns=("audio",))
ASR_DECODER = Components(name="asr_decoder", patterns=("decoder", "tdt_decoder", "rnnt_decoder", "prediction"))
ASR_HEAD = Components(name="asr_head", patterns=("joint", "tdt_head", "rnnt_head", "ctc", "log_probs"))
TOKEN_MERGE = Components(name="token_merge", patterns=("multi_modal", "multimodal", "projector", "merge"))
TEXT_DECODER = Components(name="text_decoder", patterns=("language_model", "text", "decoder", "model.layers"))
LM_HEAD = Components(name="lm_head", patterns=("lm_head", "logits"),)
TEXT_ENCODER = Components(name="text_encoder", patterns=("text_encoder", "text_model"))
UNET = Components(name="unet", patterns=("unet",))
VAE_DECODER = Components(name="vae_decoder", patterns=("vae", "decoder"))

TEXT_COMPONENTS = {
    TEXT_EMBED.name: TEXT_EMBED,
    TEXT_DECODER.name: TEXT_DECODER,
    LM_HEAD.name: LM_HEAD,
}

SPEECH_SEQ2SEQ_COMPONENTS = {
    TEXT_EMBED.name: TEXT_EMBED,
    AUDIO_ENCODER.name: AUDIO_ENCODER,
    TEXT_DECODER.name: TEXT_DECODER,
    LM_HEAD.name: LM_HEAD,
}

SPEECH_TRANSCRIBER_COMPONENTS = {
    AUDIO_ENCODER.name: AUDIO_ENCODER,
    ASR_DECODER.name: ASR_DECODER,
    ASR_HEAD.name: ASR_HEAD,
}

VISION_LANGUAGE_COMPONENTS = {
    **TEXT_COMPONENTS,
    VISION_ENCODER.name: VISION_ENCODER,
    TOKEN_MERGE.name: TOKEN_MERGE,
}

MULTIMODAL_COMPONENTS = {
    **TEXT_COMPONENTS,
    VISION_ENCODER.name: VISION_ENCODER,
    AUDIO_ENCODER.name: AUDIO_ENCODER,
    TOKEN_MERGE.name: TOKEN_MERGE,
}

T2I_COMPONENTS = {
    TEXT_ENCODER.name: TEXT_ENCODER,
    UNET.name: UNET,
    VAE_DECODER.name: VAE_DECODER,
}
