from __future__ import annotations

from ..cactus_adapters.config_utils import cfg_get, detect_model_type


SUPPORTED_FAMILIES = {"auto", "gemma3", "gemma4", "qwen", "lfm2", "whisper", "parakeet", "parakeet_tdt", "moonshine", "nomic", "needle", "generic"}


def detect_family(config, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    text_config = cfg_get(config, "text_config", None)
    base = text_config if text_config is not None else config
    detected = detect_model_type(base, config)
    if detected == "gemma":
        raw = str(cfg_get(base, "model_type", cfg_get(config, "model_type", "")) or "").lower()
        if "gemma3" in raw:
            return "gemma3"
        return "generic"
    if detected in {"gemma4", "qwen", "lfm2", "whisper", "moonshine", "parakeet", "parakeet_tdt", "nomic", "needle"}:
        return detected
    return "generic"
