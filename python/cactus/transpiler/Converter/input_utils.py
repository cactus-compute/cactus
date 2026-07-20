from ..ModelProfiles import models as MP_Models
import torch
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import json
from typing import Any
import numpy as np

#Locations to inputs for
modality_input_path = {
    "vision": "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test_monkey.png",
    "audio" : "/Users/sandhup/Documents/personal/cactus/python/cactus/assets/test.wav"
}


#Downloads profile-declared files from Hugging Face into the local converter jsons folder.
def load_files(mp: MP_Models.ModelProfile, model_id: str) -> dict[str, str]:
    output_dir = Path(__file__).resolve().parent / "jsons" / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    downloaded_files: dict[str, str] = {}

    for file in mp.files:
        try:
            downloaded_path = hf_hub_download(repo_id=model_id, filename=file, local_dir=output_dir, token=token)
        except Exception as e:
            print(f"Error downloading {file} from {model_id}: {e}")
            continue
        
        downloaded_files[file] = str(downloaded_path)
    return downloaded_files


#Loads downloaded JSON files into dictionaries keyed by filename.
def _load_json_files(files: dict[str, str]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}

    for file_name, file_path in files.items():
        if not file_name.endswith(".json"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            configs[file_name] = json.load(f)

    return configs


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


#Returns the first config value that can safely be interpreted as an int.
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


#Builds simple token ids for generic text-only synthetic exports.
def _build_input_ids(batch_size: int, sequence_length: int) -> torch.Tensor:
    tokens = torch.arange(sequence_length, dtype=torch.long) + 1
    return tokens.unsqueeze(0).repeat(batch_size, 1)


#Builds an all-ones attention mask for synthetic exports.
def _build_attention_mask(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.ones((batch_size, sequence_length), dtype=torch.long)


#Builds synthetic image pixel tensors from config-derived image dimensions.
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


#Builds synthetic audio feature tensors from config-derived audio dimensions.
def _build_input_features(batch_size: int, configs: dict[str, dict[str, Any]]) -> torch.Tensor:
    feature_bins = _first_int(configs, ("feature_size", "num_mel_bins", "n_mels"), 80)
    max_source_positions = _first_int(configs, ("max_source_positions",), 1500)
    feature_frames = _first_int(configs, ("nb_max_frames", "max_length", "feature_frames"), max_source_positions * 2)
    return torch.zeros((batch_size, feature_bins, feature_frames), dtype=torch.float32)


#Builds generic tuple-style KV cache tensors for models that still accept legacy cache inputs.
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


#Finds a decoder start token id for seq2seq-style synthetic inputs.
def _decoder_start_token_id(configs: dict[str, dict[str, Any]]) -> int:
    return _first_int(configs, ("decoder_start_token_id", "bos_token_id", "pad_token_id"), 1)


#Loads the local image asset used for processor-backed multimodal exports.
def _load_image_asset():
    from PIL import Image

    return Image.open(modality_input_path["vision"]).convert("RGB")


#Loads and normalizes the local audio asset used for processor-backed multimodal exports.
def _load_audio_asset() -> np.ndarray:
    from scipy.io import wavfile

    _sample_rate, audio = wavfile.read(modality_input_path["audio"])

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    return audio


#Removes Hugging Face processor type metadata before constructing processor components directly.
def _without_type_key(config: dict[str, Any], type_key: str) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != type_key}


#Finds a previously cached Hugging Face file without requiring network access.
def _find_cached_hf_file(model_id: str, filename: str) -> str | None:
    cache_model_id = model_id.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{cache_model_id}"

    if not cache_root.exists():
        return None

    for candidate in cache_root.glob(f"snapshots/*/{filename}"):
        if candidate.is_file():
            return str(candidate)

    return None


#Gemma-specific: manually builds Gemma4Processor when AutoProcessor cannot be loaded cleanly offline.
def _build_gemma4_processor(model_id: str, configs: dict[str, dict[str, Any]]):
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gemma4 import (
        Gemma4AudioFeatureExtractor,
        Gemma4ImageProcessor,
        Gemma4Processor,
        Gemma4VideoProcessor,
    )

    processor_config = configs.get("processor_config.json")
    if processor_config is None or processor_config.get("processor_class") != "Gemma4Processor":
        return None

    token = os.environ.get("HF_TOKEN")
    tokenizer_path = _find_cached_hf_file(model_id, "tokenizer.json")
    if tokenizer_path is None:
        try:
            tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json", token=token, local_files_only=True)
        except Exception:
            tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json", token=token)

    tokenizer_config = configs.get("tokenizer_config.json", {})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token=tokenizer_config.get("bos_token", "<bos>"),
        eos_token=tokenizer_config.get("eos_token", "<eos>"),
        unk_token=tokenizer_config.get("unk_token", "<unk>"),
        pad_token=tokenizer_config.get("pad_token", "<pad>"),
    )

    for token_name in (
        "image_token",
        "audio_token",
        "boi_token",
        "eoi_token",
        "boa_token",
        "eoa_token",
    ):
        token_value = tokenizer_config.get(token_name)
        if token_value is not None:
            setattr(tokenizer, token_name, token_value)

    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    tokenizer.audio_token_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_token)
    image_processor = Gemma4ImageProcessor(
        **_without_type_key(processor_config.get("image_processor", {}), "image_processor_type")
    )
    feature_extractor = Gemma4AudioFeatureExtractor(
        **_without_type_key(processor_config.get("feature_extractor", {}), "feature_extractor_type")
    )
    video_processor = Gemma4VideoProcessor(
        **_without_type_key(processor_config.get("video_processor", {}), "video_processor_type")
    )

    return Gemma4Processor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_seq_length=processor_config.get("image_seq_length", 280),
        audio_seq_length=processor_config.get("audio_seq_length", 750),
        audio_ms_per_token=processor_config.get("audio_ms_per_token", 40),
    )


#Builds real processor-backed tensor kwargs for multimodal models.
def build_processor_kwargs(
    model_id: str,
    input_modalities: tuple[str, ...],
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from transformers import AutoProcessor

    configs = configs or {}
    token = os.environ.get("HF_TOKEN")
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": True}
    if token:
        processor_kwargs["token"] = token

    processor = _build_gemma4_processor(model_id, configs)
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    prompt_parts = []
    call_kwargs: dict[str, Any] = {"return_tensors": "pt"}

    if "vision" in input_modalities:
        image_token = getattr(processor, "image_token", None)
        if image_token is not None:
            prompt_parts.append(image_token)
        call_kwargs["images"] = _load_image_asset()

    if "audio" in input_modalities:
        audio_token = getattr(processor, "audio_token", None)
        if audio_token is not None:
            prompt_parts.append(audio_token)
        call_kwargs["audio"] = _load_audio_asset()

    if "text" in input_modalities:
        prompt_parts.append("Describe this input.")

    call_kwargs["text"] = " ".join(prompt_parts) if prompt_parts else "Hello"
    processed = processor(**call_kwargs)
    return {key: value for key, value in dict(processed).items() if isinstance(value, torch.Tensor)}
