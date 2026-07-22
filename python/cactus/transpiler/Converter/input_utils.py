import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from ..ModelProfiles import models as MP_Models


DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_PAST_SEQUENCE_LENGTH = 8
SYNTHETIC_INPUT_STRATEGY = "synthetic"
PROCESSOR_INPUT_STRATEGY = "processor"
PREFILL_WITH_CACHE_MODE = "prefill_with_cache"
DECODE_WITH_CACHE_MODE = "decode_with_cache"
DYNAMIC_CACHE_POLICY = "dynamic_cache"
CONVERTER_JSON_DIR = Path(__file__).resolve().parent / "jsons"
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"

CONFIG_SEARCH_SECTIONS = (
    (),
    ("text_config",),
    ("vision_config",),
    ("audio_config",),
    ("encoder",),
    ("decoder",),
    ("model_config",),
)

MULTIMODAL_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "video_position_ids",
    "mm_token_type_ids",
)

#Locations to local sample inputs used by processor-backed multimodal exports.
MODALITY_INPUT_PATH = {
    "vision": ASSET_DIR / "test_monkey.png",
    "audio": ASSET_DIR / "test.wav",
}


#Downloads profile-declared JSON files from Hugging Face and loads them.
def load_configs(mp: MP_Models.ModelProfile, model_id: str | None) -> dict[str, dict[str, Any]]:
    if model_id is None:
        return {}

    output_dir = CONVERTER_JSON_DIR / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    configs: dict[str, dict[str, Any]] = {}

    for filename in mp.files:
        if not filename.endswith(".json"):
            continue

        local_path = output_dir / filename

        if not local_path.exists():
            try:
                local_path = Path(hf_hub_download(repo_id=model_id, filename=filename, local_dir=output_dir, token=token))
            except Exception as e:
                print(f"Error downloading {filename} from {model_id}: {e}")
                continue

        with local_path.open("r", encoding="utf-8") as f:
            configs[filename] = json.load(f)

    return configs


#Searches through configs to find desired keys, and their values
def _find_config_value(configs: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> Any | None:
    for config in configs.values():
        for section_path in CONFIG_SEARCH_SECTIONS:
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


#Builds config-shaped synthetic kwargs when a profile does not use a processor.
def build_synthetic_kwargs(
    modalities: tuple[str, ...],
    configs: dict[str, dict[str, Any]],
    inference_mode: str,
    uses_dynamic_cache: bool,
) -> dict[str, Any]:
    batch_size = DEFAULT_BATCH_SIZE
    sequence_length = DEFAULT_SEQUENCE_LENGTH
    past_sequence_length = DEFAULT_PAST_SEQUENCE_LENGTH
    kwargs: dict[str, Any] = {}

    if "text" in modalities:
        if inference_mode == DECODE_WITH_CACHE_MODE and not uses_dynamic_cache:
            sequence_length = 1

        kwargs["input_ids"] = _build_input_ids(batch_size, sequence_length)
        kwargs["attention_mask"] = _build_attention_mask(batch_size, sequence_length)

    if "vision" in modalities:
        kwargs["pixel_values"] = _build_pixel_values(batch_size, configs)

    if "audio" in modalities:
        kwargs["input_features"] = _build_input_features(batch_size, configs)

    if "audio" in modalities and "text" in modalities and "vision" not in modalities and "speech" in inference_mode:
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs["decoder_input_ids"] = torch.full(
            (batch_size, 1),
            _decoder_start_token_id(configs),
            dtype=torch.long,
        )

    if "audio" in modalities and "text" not in modalities:
        input_features = kwargs["input_features"]
        kwargs["attention_mask"] = _build_attention_mask(batch_size, input_features.shape[-1])

    if inference_mode == DECODE_WITH_CACHE_MODE and not uses_dynamic_cache:
        total_sequence_length = past_sequence_length + sequence_length
        kwargs["attention_mask"] = _build_attention_mask(batch_size, total_sequence_length)
        kwargs["past_key_values"] = _build_past_key_values(configs, batch_size, past_sequence_length)
        kwargs["cache_position"] = torch.arange(
            past_sequence_length,
            past_sequence_length + sequence_length,
            dtype=torch.long,
        )
        kwargs["use_cache"] = True
    elif inference_mode == PREFILL_WITH_CACHE_MODE:
        kwargs["use_cache"] = True

    return kwargs


#Builds representative model inputs for the requested modalities and inference mode. X
def build_input(
    mp: MP_Models.ModelProfile,
    input_modalities: tuple[str, ...],
    input_cls: Any,
    model_id: str | None = None,
    inference_mode: str = "prefill_no_cache",
) -> Any | None:
    if not all(modality in mp.supported_modalties for modality in input_modalities):
        print("Requesting unsupported modalities")
        return None

    configs = load_configs(mp, model_id)
    modalities = tuple(input_modalities)
    uses_dynamic_cache = DYNAMIC_CACHE_POLICY in mp.cache_policy

    if model_id is not None and mp.input_strategy != SYNTHETIC_INPUT_STRATEGY:
        try:
            return input_cls(
                args=(),
                kwargs=build_processor_kwargs(
                    model_id=model_id,
                    input_modalities=modalities,
                    configs=configs,
                    input_strategy=mp.input_strategy,
                ),
                modalities=modalities,
                inference_mode=inference_mode,
            )
        except Exception as e:
            print(f"{mp.input_strategy} input build failed for {model_id}, falling back to synthetic inputs: {e}")

    return input_cls(
        args=(),
        kwargs=build_synthetic_kwargs(
            modalities=modalities,
            configs=configs,
            inference_mode=inference_mode,
            uses_dynamic_cache=uses_dynamic_cache,
        ),
        modalities=modalities,
        inference_mode=inference_mode,
    )


#Builds one-token decode inputs and flat KV tensors for cache-mode exports. X
def build_decode_with_cache_input(
    model: torch.nn.Module,
    input_: Any,
    input_cls: Any,
    cache_spec_cls: Any,
    model_dtype_fn: Any,
    drop_multimodal: bool,
) -> Any:
    kwargs = dict(input_.kwargs)
    token_key = "input_ids" if "input_ids" in kwargs else "decoder_input_ids"

    if token_key not in kwargs:
        raise ValueError(f"{DECODE_WITH_CACHE_MODE} requires input_ids or decoder_input_ids")

    token_ids = kwargs[token_key]
    batch_size = int(token_ids.shape[0])
    past_sequence_length = int(token_ids.shape[1])
    cache_spec = cache_spec_cls.from_model(
        model=model,
        batch_size=batch_size,
        past_sequence_length=past_sequence_length,
    )

    kwargs[token_key] = token_ids[:, -1:].clone()
    kwargs["attention_mask"] = torch.ones(
        (batch_size, past_sequence_length + 1),
        dtype=torch.long,
        device=token_ids.device,
    )
    kwargs["cache_position"] = torch.arange(
        past_sequence_length,
        past_sequence_length + 1,
        dtype=torch.long,
        device=token_ids.device,
    )
    kwargs["past_key_values"] = cache_spec.empty_tensors(
        dtype=model_dtype_fn(model),
        device=token_ids.device,
    )

    if drop_multimodal:
        for multimodal_key in MULTIMODAL_KEYS:
            kwargs.pop(multimodal_key, None)

    return input_cls(
        args=input_.args,
        kwargs=kwargs,
        modalities=input_.modalities,
        inference_mode=input_.inference_mode,
    )


#Infers batch size from the first tensor-shaped input. X
def infer_batch_size(kwargs: dict[str, Any]) -> int:
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])

    return 1


#Infers the past/prompt sequence length represented by the export input. X
def infer_past_sequence_length(input_: Any) -> int:
    if input_.inference_mode == DECODE_WITH_CACHE_MODE and "cache_position" in input_.kwargs:
        return int(input_.kwargs["cache_position"][0].item())

    if "input_ids" in input_.kwargs:
        return int(input_.kwargs["input_ids"].shape[1])

    if "decoder_input_ids" in input_.kwargs:
        return int(input_.kwargs["decoder_input_ids"].shape[1])

    if "attention_mask" in input_.kwargs:
        return int(input_.kwargs["attention_mask"].shape[1])

    return 0


#Loads the local image asset used for processor-backed multimodal exports.
def _load_image_asset():
    from PIL import Image

    return Image.open(MODALITY_INPUT_PATH["vision"]).convert("RGB")


#Loads and normalizes the local audio asset used for processor-backed multimodal exports.
def _load_audio_asset() -> np.ndarray:
    from scipy.io import wavfile

    _sample_rate, audio = wavfile.read(MODALITY_INPUT_PATH["audio"])

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


#Loads AutoProcessor from local HF cache first, then falls back to normal HF loading.
def _load_auto_processor(model_id: str, processor_kwargs: dict[str, Any]):
    from transformers import AutoProcessor

    last_error: Exception | None = None

    for attempt_kwargs in hf_load_kwargs(processor_kwargs):
        try:
            return AutoProcessor.from_pretrained(model_id, **attempt_kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Unable to load processor for {model_id}") from last_error


#Returns local-cache-first HF loading kwargs plus a normal network-capable fallback.
def hf_load_kwargs(load_kwargs: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if load_kwargs.get("local_files_only") is True:
        return (load_kwargs,)

    return (
        {**load_kwargs, "local_files_only": True},
        load_kwargs,
    )


#Builds real processor-backed tensor kwargs for multimodal models.
def build_processor_kwargs(
    model_id: str,
    input_modalities: tuple[str, ...],
    configs: dict[str, dict[str, Any]] | None = None,
    input_strategy: str = PROCESSOR_INPUT_STRATEGY,
) -> dict[str, Any]:
    configs = configs or {}
    token = os.environ.get("HF_TOKEN")
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if token:
        processor_kwargs["token"] = token

    if input_strategy == PROCESSOR_INPUT_STRATEGY:
        processor = _load_auto_processor(model_id, processor_kwargs)
    elif input_strategy == "manual_gemma4_processor":
        processor = _build_gemma4_processor(model_id, configs) or _load_auto_processor(model_id, processor_kwargs)
    else:
        raise ValueError(f"Unknown input strategy: {input_strategy}")

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
