from .components import (
    MULTIMODAL_COMPONENTS,
    SPEECH_SEQ2SEQ_COMPONENTS,
    SPEECH_TRANSCRIBER_COMPONENTS,
    TEXT_COMPONENTS,
    VISION_LANGUAGE_COMPONENTS,
)
from .models import ModelProfile
from .routes import (
    GEMMA4_INFERENCE_PATTERNS,
    LFM_VLM_INFERENCE_PATTERNS,
    PARAKEET_INFERENCE_PATTERNS,
    QWEN2_5_INFERENCE_PATTERNS,
    WHISPER_INFERENCE_PATTERNS,
)


GEMMA4_E2B_PROFILE = ModelProfile(
    model_profiles="gemma4_e2b",
    components=MULTIMODAL_COMPONENTS,
    inference_type=GEMMA4_INFERENCE_PATTERNS,
    cache_type=("attention_kv", "sliding_window_attention_kv"),
    files=("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "embedding",
        "vision",
        "audio",
        "gemma4_token_merge",
        "gemma4_rmsnorm",
        "gemma4_rope",
        "gemma4_attention",
        "gemma4_mlp",
        "linear",
    ),
    features=(
        "text",
        "vision",
        "audio",
        "attention",
        "sliding_window_attention",
        "rope",
        "rms_norm",
        "conv",
        "kv_cache",
        "multimodal_token_merge",
    ),
    supported_modalties=("audio", "vision", "text")
)

WHISPER_PROFILE = ModelProfile(
    model_profiles="whisper",
    components=SPEECH_SEQ2SEQ_COMPONENTS,
    inference_type=WHISPER_INFERENCE_PATTERNS,
    cache_type=("cross_attention_kv", "decoder_attention_kv"),
    files=("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "audio",
        "whisper_audio_encoder",
        "whisper_attention",
        "whisper_mlp",
        "linear",
    ),
    features=(
        "audio",
        "speech_to_text",
        "encoder_decoder",
        "attention",
        "cross_attention",
        "conv",
        "kv_cache",
    ),
    supported_modalties=("audio", "text"),
)

PARAKEET_PROFILE = ModelProfile(
    model_profiles="parakeet",
    components=SPEECH_TRANSCRIBER_COMPONENTS,
    inference_type=PARAKEET_INFERENCE_PATTERNS,
    cache_type=(),
    files=("config.json", "preprocessor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "audio",
        "parakeet_fastconformer",
        "parakeet_tdt",
        "conv",
        "linear",
    ),
    features=(
        "audio",
        "speech_to_text",
        "conformer",
        "tdt",
        "conv",
        "attention",
    ),
    supported_modalties=("audio",),
)

LFM_VLM_PROFILE = ModelProfile(
    model_profiles="lfm_vlm",
    components=VISION_LANGUAGE_COMPONENTS,
    inference_type=LFM_VLM_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    files=("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "embedding",
        "vision",
        "lfm_vlm_token_merge",
        "lfm_attention",
        "lfm_mlp",
        "linear",
    ),
    features=(
        "text",
        "vision",
        "vision_language",
        "attention",
        "rope",
        "rms_norm",
        "kv_cache",
        "multimodal_token_merge",
    ),
    supported_modalties=("vision", "text"),
)

QWEN2_5_0_5B_PROFILE = ModelProfile(
    model_profiles="qwen2_5_0_5b",
    components=TEXT_COMPONENTS,
    inference_type=QWEN2_5_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    files=("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
    fusion_fields=(
        "generic",
        "embedding",
        "qwen2_5_rmsnorm",
        "qwen2_5_rope",
        "qwen2_5_attention",
        "qwen2_5_mlp",
        "linear",
    ),
    features=(
        "text",
        "causal_lm",
        "attention",
        "rope",
        "rms_norm",
        "mlp",
        "kv_cache",
    ),
    supported_modalties=("text",),
)

ALL_PROFILES = (
    GEMMA4_E2B_PROFILE,
    WHISPER_PROFILE,
    PARAKEET_PROFILE,
    LFM_VLM_PROFILE,
    QWEN2_5_0_5B_PROFILE,
)

MODEL_ID_MAP = {}
