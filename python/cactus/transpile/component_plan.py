from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ComponentPlan:
    task: str
    components: tuple[str, ...] = ()
    needs_image: bool = False
    needs_audio: bool = False
    force_component_pipeline: bool = False


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _load_config_txt(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return config
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def _architecture_names(config: Mapping[str, object], config_txt: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw_architectures = config.get("architectures", [])
    names = tuple(str(value) for value in raw_architectures if isinstance(value, str))
    if names or config_txt is None:
        return names
    raw_txt = config_txt.get("architectures")
    if not raw_txt:
        return ()
    return tuple(value.strip() for value in raw_txt.split(",") if value.strip())


def _has_dict_config(config: Mapping[str, object], *keys: str) -> bool:
    return any(isinstance(config.get(key), dict) for key in keys)


def _has_positive_txt_int(config_txt: Mapping[str, str] | None, *keys: str) -> bool:
    if config_txt is None:
        return False
    for key in keys:
        try:
            if int(config_txt.get(key, "0") or 0) > 0:
                return True
        except ValueError:
            continue
    return False


def _model_type(config: Mapping[str, object], config_txt: Mapping[str, str] | None = None) -> str:
    value = config.get("model_type")
    if value is None and config_txt is not None:
        value = config_txt.get("model_type")
    return str(value or "").strip().lower()


def _lowered_architectures(config: Mapping[str, object], config_txt: Mapping[str, str] | None = None) -> tuple[str, ...]:
    return tuple(value.lower() for value in _architecture_names(config, config_txt))


def _is_tdt_config(config: Mapping[str, object], model_type: str, lowered_id: str) -> bool:
    decoding_cfg = config.get("decoding")
    if isinstance(decoding_cfg, Mapping) and str(decoding_cfg.get("model_type", "") or "").lower() == "tdt":
        return True
    loss_cfg = config.get("loss")
    if isinstance(loss_cfg, Mapping) and str(loss_cfg.get("loss_name", "") or "").lower() == "tdt":
        return True
    return model_type == "parakeet_tdt" or "parakeet-tdt" in lowered_id


def _looks_like_vision_language_model(
    *,
    model_type: str,
    architectures: tuple[str, ...],
    lowered_id: str,
    has_vision: bool,
) -> bool:
    if not has_vision:
        return False
    if any(token in model_type for token in ("vl", "vision", "image", "multimodal")):
        return True
    if any(
        token in architecture
        for architecture in architectures
        for token in ("vision", "image", "vl", "imagetexttotext", "conditionalgeneration")
    ):
        return True
    return any(token in lowered_id for token in ("-vl", "_vl", "vision", "image"))


def infer_component_plan_from_config(
    config: Mapping[str, object],
    *,
    model_id: str = "",
    config_txt: Mapping[str, str] | None = None,
) -> ComponentPlan | None:
    model_type = _model_type(config, config_txt)
    lowered_id = model_id.lower()
    architectures = _lowered_architectures(config, config_txt)

    if _is_tdt_config(config, model_type, lowered_id):
        return ComponentPlan(
            task="tdt_transcription",
            components=("audio_encoder", "decoder"),
            needs_audio=True,
            force_component_pipeline=True,
        )

    if model_type == "whisper" or "whisper" in lowered_id or any("whisper" in value for value in architectures):
        return ComponentPlan(
            task="seq2seq_transcription",
            components=("audio_encoder", "decoder"),
            needs_audio=True,
            force_component_pipeline=True,
        )

    if any("ctc" in value for value in architectures) or "ctc" in lowered_id:
        return ComponentPlan(
            task="ctc_logits",
            components=("audio_encoder",),
            needs_audio=True,
        )

    has_vision = (
        _has_dict_config(config, "vision_config", "visual_config", "image_config")
        or _has_positive_txt_int(config_txt, "vision_num_layers")
    )
    has_audio = (
        _has_dict_config(config, "audio_config", "speech_config", "acoustic_config")
        or (
            _has_dict_config(config, "encoder_config")
            and (
                "audio" in model_type
                or "speech" in model_type
                or "audio_token_index" in config
                or any(token in lowered_id for token in ("speech", "audio"))
                or any(token in value for value in architectures for token in ("speech", "audio"))
            )
        )
        or _has_positive_txt_int(config_txt, "audio_num_layers")
    )
    if has_vision or has_audio:
        if has_audio or _looks_like_vision_language_model(
            model_type=model_type,
            architectures=architectures,
            lowered_id=lowered_id,
            has_vision=has_vision,
        ):
            components: list[str] = []
            if has_vision:
                components.append("vision_encoder")
            if has_audio:
                components.append("audio_encoder")
            components.extend(("lm_encoder", "decoder"))
            return ComponentPlan(
                task="multimodal_causal_lm_logits",
                components=tuple(components),
                needs_image=has_vision,
                needs_audio=has_audio,
                force_component_pipeline=True,
            )

    if any("causallm" in value for value in architectures):
        return ComponentPlan(task="causal_lm_logits", components=("decoder",))
    if any(token in lowered_id or token in model_type for token in ("qwen", "gemma", "llama", "mistral", "lfm")):
        return ComponentPlan(task="causal_lm_logits", components=("decoder",))
    return None


def infer_component_plan_from_output(output_dir: str | Path, *, model_id: str = "") -> ComponentPlan | None:
    root = Path(output_dir)
    hf_config = _load_json(root / "hf_config.json")
    config_txt = _load_config_txt(root / "config.txt")
    return infer_component_plan_from_config(hf_config, model_id=model_id, config_txt=config_txt)
