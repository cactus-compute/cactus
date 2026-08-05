from dataclasses import replace

from .components import (
    MULTIMODAL_COMPONENTS,
    SPEECH_SEQ2SEQ_COMPONENTS,
    SPEECH_TRANSCRIBER_COMPONENTS,
    TEXT_COMPONENTS,
    VISION_LANGUAGE_COMPONENTS,
)
from .models import (
    AliasContract,
    CacheContract,
    MediaContract,
    ModelProfile,
    PromptContract,
    RuntimeContract,
    StateContract,
)
from .routes import (
    GEMMA4_INFERENCE_PATTERNS,
    LFM_VLM_INFERENCE_PATTERNS,
    PARAKEET_INFERENCE_PATTERNS,
    QWEN2_5_INFERENCE_PATTERNS,
    WHISPER_INFERENCE_PATTERNS,
)


TEXT_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="generic_text",
    states=(
        StateContract(
            name="decoder_kv_cache",
            kind="kv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
    ),
    aliases=(
        AliasContract(
            source_component="text_embedding",
            source_output="inputs_embeds",
            target_component="decoder_prefill_chunk",
            target_input="inputs_embeds",
            required=False,
        ),
    ),
)

GEMMA4_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="gemma4_multimodal",
    execution_strategy="component_graph_chunked_prefill",
    state_owner="session",
    cache_persistence="component_move",
    output_alias_policy="alias_if_compatible",
    cache_transfer_policy="move",
    states=(
        StateContract(
            name="decoder_kv_cache",
            kind="kv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
        StateContract(
            name="media_features",
            kind="activation",
            producer="vision_encoder,audio_encoder",
            consumers=("lm_encoder_media_chunk", "lm_encoder_media_step"),
            lifetime="request",
            transfer="copy_or_alias",
        ),
    ),
    aliases=(
        AliasContract(
            source_component="lm_encoder_text_chunk",
            source_output="inputs_embeds",
            target_component="decoder_prefill_chunk",
            target_input="inputs_embeds",
        ),
        AliasContract(
            source_component="lm_encoder_media_chunk",
            source_output="inputs_embeds",
            target_component="decoder_prefill_chunk",
            target_input="inputs_embeds",
        ),
    ),
)

WHISPER_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="whisper_encoder_decoder",
    execution_strategy="encoder_decoder_component_graph",
    states=(
        StateContract(
            name="encoder_hidden_states",
            kind="activation",
            producer="audio_encoder",
            consumers=("decoder_step", "decoder_prefill_chunk"),
            lifetime="request",
            transfer="copy_or_alias",
        ),
        StateContract(
            name="decoder_kv_cache",
            kind="kv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
        StateContract(
            name="cross_attention_kv",
            kind="kv",
            producer="decoder_cross_kv",
            consumers=("decoder_step",),
            lifetime="request",
            transfer="copy_or_alias",
        ),
    ),
    aliases=(
        AliasContract(
            source_component="audio_encoder",
            source_output="encoder_hidden_states",
            target_component="decoder_step",
            target_input="encoder_hidden_states",
        ),
    ),
)

PARAKEET_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="parakeet_tdt",
    execution_strategy="speech_transcriber_component_graph",
    states=(
        StateContract(
            name="encoder_hidden_states",
            kind="activation",
            producer="audio_encoder",
            consumers=("asr_decoder",),
            lifetime="request",
            transfer="copy_or_alias",
        ),
        StateContract(
            name="tdt_decoder_state",
            kind="recurrent",
            producer="asr_decoder",
            consumers=("asr_decoder", "asr_head"),
            lifetime="sequence",
            transfer="alias_if_compatible",
        ),
    ),
)

LFM_VLM_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="lfm_vlm",
    execution_strategy="component_graph_chunked_prefill",
    states=(
        StateContract(
            name="vision_features",
            kind="activation",
            producer="vision_encoder",
            consumers=("token_merge", "lm_encoder_media_chunk"),
            lifetime="request",
            transfer="copy_or_alias",
        ),
        StateContract(
            name="decoder_kv_cache",
            kind="kv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
    ),
)

LFM_MOE_RUNTIME_CONTRACT = RuntimeContract(
    plan_name="lfm_moe",
    execution_strategy="component_graph_chunked_prefill",
    states=(
        StateContract(
            name="decoder_kv_cache",
            kind="kv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
        StateContract(
            name="conv_state",
            kind="conv",
            producer="decoder_prefill_chunk",
            consumers=("decoder_step",),
            lifetime="sequence",
            transfer="move",
        ),
    ),
)

WHISPER_SUPPRESS_TOKEN_IDS = (
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
    90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
    1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
    6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
    13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
    21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
    36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361,
    50362,
)


GEMMA4_E2B_PROFILE = ModelProfile(
    model_profiles="gemma4_e2b",
    components=MULTIMODAL_COMPONENTS,
    inference_type=GEMMA4_INFERENCE_PATTERNS,
    cache_type=("attention_kv", "sliding_window_attention_kv"),
    cache_policy=("dynamic_cache", "drop_multimodal_on_decode", "shared_kv_layers"),
    files=("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json", "merges.txt"),
    fusion_fields=(
        "generic",
        "embedding",
        "vision",
        "audio",
        "attention",
        "normalization",
        "gemma4_token_merge",
        "gemma4_rmsnorm",
        "gemma4_attention",
        "gemma4_rope",
        "gemma4_mlp",
        "linear",
        "cache",
    ),
    supported_modalties=("audio", "vision", "text"),
    input_strategy="manual_gemma4_processor",
    export_patches=("gemma4_audio_mask",),
    load_strategy="image_text_to_text_strict",
    prompt_contract=PromptContract(
        style="gemma4",
        template_source="tokenizer_config",
        text_style="default_chat",
        media_style="gemma4_turns",
        turn_start_token="<|turn>",
        turn_end_token="<turn|>",
        suppress_generation_token_ids=(255999, 256000, 258880, 258881, 258882, 258883),
    ),
    media_contract=MediaContract(
        image_preprocess_strategy="gemma4",
        audio_preprocess_strategy="gemma4",
        injection_strategy="token_row_replacement",
        media_order=("image", "audio"),
        focus_policy="",
        image_focus_keywords=("image", "picture", "pictured", "photo", "visual", "shown", "see", "look", "animal"),
        audio_focus_keywords=("audio", "sound", "speech", "spoken", "voice", "hear", "heard", "listen", "transcribe"),
        chunk_prefill_modalities=("vision", "image", "audio"),
        prefill_fallback="error",
        min_new_tokens=1,
        chunk_output_sources=(),
        mask_polarity="true_means_visible",
        span_strategy="one_placeholder_token_per_feature_row",
        placeholder_token_id=0,
        image_token_id=258880,
        audio_token_id=258881,
        image_token="<|image|>",
        audio_token="<|audio|>",
        image_begin_token="<|image>",
        image_end_token="<image|>",
        audio_begin_token="<|audio>",
        audio_end_token="<audio|>",
        image_prompt_position="before_text",
        audio_prompt_position="after_text",
        audio_rows_per_frames="ceil_div:4",
        image_feature_names=("image_features", "image_embeddings", "vision_features", "inputs_embeds"),
        audio_feature_names=("audio_features", "audio_embeddings", "encoder_hidden_states", "inputs_embeds"),
    ),
    cache_contract=CacheContract(
        prefill_decode_compatibility="shared_kv_layers",
        state_transfer="prefill_to_decode",
        decode_uses_media_components=False,
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
        max_cache_sequence_length=131072,
        full_retention_kv_layers=(13, 14),
    ),
    runtime_contract=GEMMA4_RUNTIME_CONTRACT,
)

GEMMA4_E2B_IT_PROFILE = replace(
    GEMMA4_E2B_PROFILE,
    model_profiles="gemma4_e2b_it",
    prompt_contract=PromptContract(
        style="gemma4",
        template_source="tokenizer_config",
        text_style="gemma4_turns",
        media_style="gemma4_turns",
        turn_start_token="<|turn>",
        turn_end_token="<turn|>",
        suppress_generation_token_ids=(255999, 256000, 258880, 258881, 258882, 258883),
    ),
)

WHISPER_PROFILE = ModelProfile(
    model_profiles="whisper",
    components=SPEECH_SEQ2SEQ_COMPONENTS,
    inference_type=WHISPER_INFERENCE_PATTERNS,
    cache_type=("cross_attention_kv", "decoder_attention_kv"),
    cache_policy=("dynamic_cache", "encoder_decoder_cache"),
    files=("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "audio",
        "attention",
        "normalization",
        "mlp",
        "cache",
        "whisper_audio_encoder",
        "whisper_attention",
        "whisper_mlp",
        "linear",
    ),
    supported_modalties=("audio", "text"),
    input_strategy="processor",
    export_patches=(),
    load_strategy="speech_seq2seq",
    prompt_contract=PromptContract(
        style="speech_seq2seq",
        template_source="processor",
        suppress_generation_token_ids=WHISPER_SUPPRESS_TOKEN_IDS,
    ),
    cache_contract=CacheContract(
        prefill_decode_compatibility="encoder_decoder_cache",
        state_transfer="encoder_cross_kv_to_decode",
        fp16_kv_cache_components=("decoder_step",),
        max_cache_sequence_length=448,
    ),
    runtime_contract=WHISPER_RUNTIME_CONTRACT,
)

PARAKEET_PROFILE = ModelProfile(
    model_profiles="parakeet",
    components=SPEECH_TRANSCRIBER_COMPONENTS,
    inference_type=PARAKEET_INFERENCE_PATTERNS,
    cache_type=(),
    cache_policy=(),
    files=("config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "audio",
        "attention",
        "normalization",
        "mlp",
        "parakeet_fastconformer",
        "parakeet_tdt",
        "conv",
        "linear",
    ),
    supported_modalties=("audio",),
    input_strategy="processor",
    export_patches=(),
    load_strategy="parakeet_tdt",
    prompt_contract=PromptContract(style="speech_transcriber", template_source="processor"),
    runtime_contract=PARAKEET_RUNTIME_CONTRACT,
)

LFM_VLM_PROFILE = ModelProfile(
    model_profiles="lfm_vlm",
    components=VISION_LANGUAGE_COMPONENTS,
    inference_type=LFM_VLM_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    cache_policy=("dynamic_cache", "drop_multimodal_on_decode"),
    files=("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "embedding",
        "vision",
        "attention",
        "normalization",
        "mlp",
        "cache",
        "lfm_rmsnorm",
        "lfm_vlm_token_merge",
        "lfm_attention",
        "lfm_mlp",
        "linear",
    ),
    supported_modalties=("vision", "text"),
    input_strategy="processor",
    export_patches=("lfm2_vl_image_features",),
    load_strategy="image_text_to_text",
    prompt_contract=PromptContract(
        style="lfm2_vl",
        template_source="tokenizer_config",
        suppress_generation_token_ids=(396, 497, 498, 499, 500),
    ),
    media_contract=MediaContract(
        image_preprocess_strategy="lfm2_vl",
        injection_strategy="feature_input_then_token_row_replacement",
        mask_polarity="true_means_replace",
        span_strategy="processor_image_token_span",
        image_token_id=396,
        image_token="<|image|>",
        image_feature_names=("image_features", "image_embeddings", "vision_features", "inputs_embeds"),
    ),
    cache_contract=CacheContract(
        prefill_decode_compatibility="dynamic_cache",
        state_transfer="prefill_to_decode",
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
        decode_uses_media_components=False,
        max_cache_sequence_length=128000,
    ),
    runtime_contract=LFM_VLM_RUNTIME_CONTRACT,
)

QWEN2_5_0_5B_PROFILE = ModelProfile(
    model_profiles="qwen2_5_0_5b",
    components=TEXT_COMPONENTS,
    inference_type=QWEN2_5_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    cache_policy=("dynamic_cache",),
    files=("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
    fusion_fields=(
        "generic",
        "embedding",
        "attention",
        "normalization",
        "mlp",
        "cache",
        "qwen2_5_rmsnorm",
        "qwen2_5_rope",
        "qwen2_5_attention",
        "qwen2_5_mlp",
        "linear",
    ),
    supported_modalties=("text",),
    input_strategy="synthetic",
    export_patches=(),
    load_strategy="causal_lm",
    prompt_contract=PromptContract(style="chat", template_source="tokenizer_config"),
    cache_contract=CacheContract(
        prefill_decode_compatibility="dynamic_cache",
        state_transfer="prefill_to_decode",
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
        max_cache_sequence_length=32768,
    ),
    runtime_contract=TEXT_RUNTIME_CONTRACT,
)

LFM_MOE_PROFILE = ModelProfile(
    model_profiles="lfm_moe",
    components=TEXT_COMPONENTS,
    inference_type=QWEN2_5_INFERENCE_PATTERNS,
    cache_type=("attention_kv", "conv_state"),
    cache_policy=("dynamic_cache",),
    files=("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"),
    fusion_fields=(
        "generic",
        "embedding",
        "attention",
        "normalization",
        "mlp",
        "cache",
        "lfm_attention",
        "lfm_mlp",
        "lfm_moe",
        "moe",
        "conv",
        "linear",
    ),
    supported_modalties=("text",),
    input_strategy="processor",
    export_patches=("transformers_moe_grouped_mm_fallback",),
    load_strategy="causal_lm",
    prompt_contract=PromptContract(style="chat", template_source="tokenizer_config"),
    cache_contract=CacheContract(
        prefill_decode_compatibility="attention_kv_and_conv_state",
        state_transfer="prefill_to_decode",
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
        max_cache_sequence_length=128000,
    ),
    runtime_contract=LFM_MOE_RUNTIME_CONTRACT,
)

GENERIC_TEXT_PROFILE = ModelProfile(
    model_profiles="generic_text",
    components=TEXT_COMPONENTS,
    inference_type=QWEN2_5_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    cache_policy=("dynamic_cache",),
    files=("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"),
    fusion_fields=("generic", "embedding", "attention", "normalization", "mlp", "cache", "linear"),
    supported_modalties=("text",),
    input_strategy="synthetic",
    export_patches=(),
    load_strategy="causal_lm",
    prompt_contract=PromptContract(style="chat", template_source="tokenizer_config"),
    cache_contract=CacheContract(
        prefill_decode_compatibility="dynamic_cache",
        state_transfer="prefill_to_decode",
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
    ),
    runtime_contract=TEXT_RUNTIME_CONTRACT,
)

GENERIC_VISION_LANGUAGE_PROFILE = ModelProfile(
    model_profiles="generic_vlm",
    components=VISION_LANGUAGE_COMPONENTS,
    inference_type=LFM_VLM_INFERENCE_PATTERNS,
    cache_type=("attention_kv",),
    cache_policy=("dynamic_cache", "drop_multimodal_on_decode"),
    files=("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json"),
    fusion_fields=("generic", "embedding", "vision", "attention", "normalization", "mlp", "cache", "linear"),
    supported_modalties=("vision", "text"),
    input_strategy="processor",
    export_patches=(),
    load_strategy="image_text_to_text",
    prompt_contract=PromptContract(style="chat", template_source="tokenizer_config"),
    media_contract=MediaContract(
        image_preprocess_strategy="processor",
        injection_strategy="processor_feature_input",
        mask_polarity="processor_default",
        span_strategy="processor_image_token_span",
        image_feature_names=("image_features", "image_embeddings", "vision_features", "inputs_embeds"),
    ),
    cache_contract=CacheContract(
        prefill_decode_compatibility="dynamic_cache",
        state_transfer="prefill_to_decode",
        fp16_kv_cache_components=("decoder_prefill_chunk", "decoder_step"),
    ),
    runtime_contract=LFM_VLM_RUNTIME_CONTRACT,
)

GENERIC_SPEECH_SEQ2SEQ_PROFILE = ModelProfile(
    model_profiles="generic_speech_seq2seq",
    components=SPEECH_SEQ2SEQ_COMPONENTS,
    inference_type=WHISPER_INFERENCE_PATTERNS,
    cache_type=("cross_attention_kv", "decoder_attention_kv"),
    cache_policy=("dynamic_cache", "encoder_decoder_cache"),
    files=("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"),
    fusion_fields=("generic", "audio", "attention", "normalization", "mlp", "cache", "linear"),
    supported_modalties=("audio", "text"),
    input_strategy="processor",
    export_patches=(),
    load_strategy="speech_seq2seq",
    prompt_contract=PromptContract(style="speech_seq2seq", template_source="processor"),
    cache_contract=CacheContract(
        prefill_decode_compatibility="encoder_decoder_cache",
        state_transfer="encoder_cross_kv_to_decode",
        fp16_kv_cache_components=("decoder_step",),
    ),
    runtime_contract=WHISPER_RUNTIME_CONTRACT,
)

MODEL_ID_MAP = {
    "google/gemma-4-E2B": GEMMA4_E2B_PROFILE,
    "google/gemma-4-E2B-it": GEMMA4_E2B_IT_PROFILE,
    "openai/whisper-tiny": WHISPER_PROFILE,
    "nvidia/parakeet-tdt-0.6b-v3": PARAKEET_PROFILE,
    "LiquidAI/LFM2-VL-3B": LFM_VLM_PROFILE,
    "Qwen/Qwen2.5-0.5B": QWEN2_5_0_5B_PROFILE,
    "LiquidAI/LFM2.5-8B-A1B": LFM_MOE_PROFILE,
}


def profile_for_model_id(model_id: str) -> ModelProfile:
    if model_id in MODEL_ID_MAP:
        return MODEL_ID_MAP[model_id]

    normalized = model_id.lower()
    for candidate_id, profile in MODEL_ID_MAP.items():
        if candidate_id.lower() == normalized:
            return profile

    return generic_profile_for_model_id(model_id)


def generic_profile_for_model_id(model_id: str) -> ModelProfile:
    normalized = model_id.lower()

    if any(marker in normalized for marker in ("vl", "vision", "llava", "image")):
        return GENERIC_VISION_LANGUAGE_PROFILE

    if any(marker in normalized for marker in ("whisper", "speech", "audio", "asr")):
        return GENERIC_SPEECH_SEQ2SEQ_PROFILE

    return GENERIC_TEXT_PROFILE
