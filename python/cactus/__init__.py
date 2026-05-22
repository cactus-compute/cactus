"""Cactus — on-device AI inference."""

__version__ = "1.14.0"

from .cli.download import ensure_model, get_weights_dir, get_model_dir_name

from .cli import main


_FFI_NAMES = frozenset({
    "cactus_init", "cactus_destroy", "cactus_reset", "cactus_stop",
    "cactus_complete", "cactus_prefill",
    "cactus_embed", "cactus_image_embed", "cactus_audio_embed",
    "cactus_transcribe", "cactus_detect_language",
    "cactus_vad", "cactus_diarize", "cactus_embed_speaker",
    "cactus_tokenize", "cactus_decode_tokens", "cactus_score_window",
    "cactus_rag_query",
    "cactus_stream_transcribe_start", "cactus_stream_transcribe_process",
    "cactus_stream_transcribe_stop",
    "cactus_index_init", "cactus_index_add", "cactus_index_delete",
    "cactus_index_query", "cactus_index_get", "cactus_index_compact",
    "cactus_index_destroy",
    "cactus_set_app_id", "cactus_set_telemetry_environment",
    "cactus_telemetry_flush", "cactus_telemetry_shutdown",
    "cactus_log_set_level", "cactus_log_set_callback",
    "cactus_get_last_error", "cactus_preprocess_audio_features",
})

def __getattr__(name):
    if name in ("Graph", "Tensor"):
        from .bindings import graph
        return getattr(graph, name)

    if name in _FFI_NAMES:
        from .bindings import cactus
        return getattr(cactus, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
