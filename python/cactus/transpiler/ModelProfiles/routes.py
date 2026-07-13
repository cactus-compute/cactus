from __future__ import annotations

from .models import Combinations, InferencePattern, Route


TEXT_PREFILL_ROUTE = Route(
    route=(
        Combinations(
            input=("text_embed",),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

TEXT_VISION_PREFILL_ROUTE = Route(
    route=(
        Combinations(
            input=("text_embed", "vision_encoder"),
            output="token_merge",
        ),
        Combinations(
            input=("token_merge",),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

TEXT_AUDIO_PREFILL_ROUTE = Route(
    route=(
        Combinations(
            input=("text_embed", "audio_encoder"),
            output="token_merge",
        ),
        Combinations(
            input=("token_merge",),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

TEXT_VISION_AUDIO_PREFILL_ROUTE = Route(
    route=(
        Combinations(
            input=("text_embed", "vision_encoder", "audio_encoder"),
            output="token_merge",
        ),
        Combinations(
            input=("token_merge",),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

TEXT_DECODE_ROUTE = Route(
    route=(
        Combinations(
            input=("text_embed",),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

SPEECH_SEQ2SEQ_PREFILL_ROUTE = Route(
    route=(
        Combinations(
            input=("audio_encoder", "text_embed"),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

SPEECH_SEQ2SEQ_DECODE_ROUTE = Route(
    route=(
        Combinations(
            input=("audio_encoder", "text_embed"),
            output="text_decoder",
        ),
        Combinations(
            input=("text_decoder",),
            output="lm_head",
        ),
    )
)

SPEECH_TRANSCRIBE_ROUTE = Route(
    route=(
        Combinations(
            input=("audio_encoder",),
            output="asr_decoder",
        ),
        Combinations(
            input=("audio_encoder", "asr_decoder"),
            output="asr_head",
        ),
    )
)


TEXT_PREFILL = InferencePattern(
    name="text_prefill",
    route=TEXT_PREFILL_ROUTE,
)

TEXT_VISION_PREFILL = InferencePattern(
    name="text_vision_prefill",
    route=TEXT_VISION_PREFILL_ROUTE,
)

TEXT_AUDIO_PREFILL = InferencePattern(
    name="text_audio_prefill",
    route=TEXT_AUDIO_PREFILL_ROUTE,
)

TEXT_VISION_AUDIO_PREFILL = InferencePattern(
    name="text_vision_audio_prefill",
    route=TEXT_VISION_AUDIO_PREFILL_ROUTE,
)

TEXT_DECODE = InferencePattern(
    name="text_decode",
    route=TEXT_DECODE_ROUTE,
)

SPEECH_SEQ2SEQ_PREFILL = InferencePattern(
    name="speech_seq2seq_prefill",
    route=SPEECH_SEQ2SEQ_PREFILL_ROUTE,
)

SPEECH_SEQ2SEQ_DECODE = InferencePattern(
    name="speech_seq2seq_decode",
    route=SPEECH_SEQ2SEQ_DECODE_ROUTE,
)

SPEECH_TRANSCRIBE = InferencePattern(
    name="speech_transcribe",
    route=SPEECH_TRANSCRIBE_ROUTE,
)


GEMMA4_INFERENCE_PATTERNS = {
    TEXT_PREFILL.name: TEXT_PREFILL,
    TEXT_VISION_PREFILL.name: TEXT_VISION_PREFILL,
    TEXT_AUDIO_PREFILL.name: TEXT_AUDIO_PREFILL,
    TEXT_VISION_AUDIO_PREFILL.name: TEXT_VISION_AUDIO_PREFILL,
    TEXT_DECODE.name: TEXT_DECODE,
}

WHISPER_INFERENCE_PATTERNS = {
    SPEECH_SEQ2SEQ_PREFILL.name: SPEECH_SEQ2SEQ_PREFILL,
    SPEECH_SEQ2SEQ_DECODE.name: SPEECH_SEQ2SEQ_DECODE,
}

PARAKEET_INFERENCE_PATTERNS = {
    SPEECH_TRANSCRIBE.name: SPEECH_TRANSCRIBE,
}

LFM_VLM_INFERENCE_PATTERNS = {
    TEXT_PREFILL.name: TEXT_PREFILL,
    TEXT_VISION_PREFILL.name: TEXT_VISION_PREFILL,
    TEXT_DECODE.name: TEXT_DECODE,
}
