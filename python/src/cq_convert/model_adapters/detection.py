from __future__ import annotations

from ..cactus_adapters.config_utils import cfg_get, detect_model_type


SUPPORTED_FAMILIES = {"auto", "gemma4", "qwen", "lfm2", "whisper", "parakeet", "parakeet_tdt", "moonshine", "nomic", "generic"}


def detect_family(config, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    text_config = cfg_get(config, "text_config", None)
    base = text_config if text_config is not None else config
    detected = detect_model_type(base, config)
    if detected in {"gemma4", "qwen", "lfm2", "whisper", "moonshine", "parakeet", "parakeet_tdt", "nomic"}:
        return detected
    return "generic"
