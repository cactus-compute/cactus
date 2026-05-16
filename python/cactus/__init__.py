"""Cactus — on-device AI inference."""

__version__ = "0.1.0"

# Download (always available)
from .cli.download import ensure_model, get_weights_dir, get_model_dir_name

# CLI entry point
from .cli import main


def __getattr__(name):
    """Lazy-load bindings that require the native library."""
    if name in ("Graph", "Tensor"):
        from .bindings import graph
        return getattr(graph, name)

    _ffi_names = {
        "cactus_init", "cactus_destroy", "cactus_reset", "cactus_stop",
        "cactus_complete", "cactus_prefill",
        "cactus_embed", "cactus_image_embed", "cactus_audio_embed",
        "cactus_transcribe", "cactus_detect_language",
        "cactus_vad", "cactus_diarize", "cactus_embed_speaker",
    }
    if name in _ffi_names:
        from .bindings import cactus
        return getattr(cactus, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
