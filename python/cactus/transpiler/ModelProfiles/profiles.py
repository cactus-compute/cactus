from __future__ import annotations

from .components import MULTIMODAL_COMPONENTS
from .models import Cache, Files, ModelProfile
from .routes import GEMMA4_INFERENCE_PATTERNS


GEMMA4_E2B_PROFILE = ModelProfile(
    model_profiles="gemma4_e2b",
    components=MULTIMODAL_COMPONENTS,
    inference_type=GEMMA4_INFERENCE_PATTERNS,
    cache_type=Cache(
        types=("attention_kv", "sliding_window_attention_kv"),
    ),
    files=Files(
        required=("config.json",),
        optional=(
            "generation_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ),
    ),
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
)


ALL_PROFILES = (GEMMA4_E2B_PROFILE,)
