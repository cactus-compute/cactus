from ..ModelProfiles import models as MP_Models
import torch
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import json
from typing import Any

#Locations to inputs for
modality_input_path = {
    "vision": "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test_monkey.png",
    "audio" : "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test.wav"
}


#Downloads the file from hf
def load_files(mp: MP_Models.ModelProfile, model_id: str) -> dict[str, dict[str, Any]]:
    output_dir = Path(__file__).resolve().parent / "jsons" / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    downloaded_files: dict[str, dict[str, Any]] = {}

    for file in mp.files:
        try:
            downloaded_path = hf_hub_download(repo_id=model_id, filename=file, local_dir=output_dir, token=token)
        except Exception as e:
            print(f"Error downloading {file} from {model_id}: {e}")
            continue
        
        if file.endswith(".json"):
            with open(downloaded_path, "r", encoding="utf-8") as f:
                downloaded_files[file] = json.load(f)

        downloaded_files[file] = str(downloaded_path)
    return downloaded_files


#TODO: Clean up all functions below this point
#Searches through configs to find desired keys, and their values
def _find_config_value(configs: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> Any | None:
    search_sections = (
        (),
        ("text_config",),
        ("vision_config",),
        ("audio_config",),
        ("encoder",),
        ("decoder",),
        ("model_config",),
    )

    for config in configs.values():
        for section_path in search_sections:
            section = config

            for section_name in section_path:
                if not isinstance(section, dict):
                    section = None
                    break
                section = section.get(section_name)

            if not isinstance(section, dict):
                continue

            for key in keys:
                if key in section and section[key] is not None:
                    return section[key]

    return None


def _first_int(configs: dict[str, dict[str, Any]], keys: tuple[str, ...], default: int) -> int:
    value = _find_config_value(configs, keys)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _build_input_ids(batch_size: int, sequence_length: int) -> torch.Tensor:
    tokens = torch.arange(sequence_length, dtype=torch.long) + 1
    return tokens.unsqueeze(0).repeat(batch_size, 1)


def _build_attention_mask(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.ones((batch_size, sequence_length), dtype=torch.long)


def _build_pixel_values(batch_size: int, configs: dict[str, dict[str, Any]]) -> torch.Tensor:
    size = _find_config_value(configs, ("size", "image_size"))
    channels = _first_int(configs, ("num_channels", "image_num_channels"), 3)
    height = 224
    width = 224

    if isinstance(size, int):
        height = size
        width = size
    elif isinstance(size, dict):
        height = int(size.get("height") or size.get("shortest_edge") or height)
        width = int(size.get("width") or size.get("shortest_edge") or width)

    return torch.zeros((batch_size, channels, height, width), dtype=torch.float32)


def _build_input_features(batch_size: int, configs: dict[str, dict[str, Any]]) -> torch.Tensor:
    feature_bins = _first_int(configs, ("feature_size", "num_mel_bins", "n_mels"), 80)
    max_source_positions = _first_int(configs, ("max_source_positions",), 1500)
    feature_frames = _first_int(configs, ("nb_max_frames", "max_length", "feature_frames"), max_source_positions * 2)
    return torch.zeros((batch_size, feature_bins, feature_frames), dtype=torch.float32)


def _build_past_key_values(configs: dict[str, dict[str, Any]], batch_size: int, past_sequence_length: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    num_layers = _first_int(configs, ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers"), 1)
    num_attention_heads = _first_int(configs, ("num_attention_heads", "n_head", "decoder_attention_heads"), 1)
    num_key_value_heads = _first_int(configs, ("num_key_value_heads", "num_kv_heads"), num_attention_heads)
    hidden_size = _first_int(configs, ("hidden_size", "n_embd", "d_model"), 64)
    head_dim = _first_int(configs, ("head_dim", "kv_channels"), max(1, hidden_size // max(1, num_attention_heads)))
    cache_shape = (batch_size, num_key_value_heads, past_sequence_length, head_dim)

    return tuple(
        (
            torch.zeros(cache_shape, dtype=torch.float32),
            torch.zeros(cache_shape, dtype=torch.float32),
        )
        for _ in range(num_layers)
    )


def _decoder_start_token_id(configs: dict[str, dict[str, Any]]) -> int:
    return _first_int(configs, ("decoder_start_token_id", "bos_token_id", "pad_token_id"), 1)