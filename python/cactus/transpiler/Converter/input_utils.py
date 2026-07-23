import json
from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download
import numpy as np
import torch

from . import constants
from . import input_processor as IP
from ..ModelProfiles import models as MP_Models


def load_configs(mp: MP_Models.ModelProfile, model_id: str | None) -> dict[str, dict[str, Any]]:
    output_dir = constants.CONVERTER_JSON_DIR / mp.model_profiles
    output_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict[str, Any]] = {}

    for filename in mp.files:
        if not filename.endswith(".json"):
            continue

        local_path = output_dir / filename

        if not local_path.exists():
            try:
                local_path = Path(hf_hub_download(repo_id=model_id, filename=filename, local_dir=output_dir, token=constants.token))
            except Exception as e:
                print(f"Error downloading {filename} from {model_id}: {e}")
                continue

        with local_path.open("r", encoding="utf-8") as f:
            configs[filename] = json.load(f)

    return configs


def build_input(mp: MP_Models.ModelProfile, input_modalities: tuple[str, ...], input_cls: Any, model_id: str | None = None, inference_mode: str = "prefill_no_cache") -> Any | None:
    if not all(modality in mp.supported_modalties for modality in input_modalities):
        print("Requesting unsupported modalities")
        return None

    configs = load_configs(mp, model_id)

    if mp.input_strategy == constants.SYNTHETIC_INPUT_STRATEGY:
        raise ValueError("Synthetic input creation is currently disabled")

    return input_cls(
        args=(),
        kwargs=build_processor_kwargs(model_id, input_modalities),
        modalities=input_modalities,
        inference_mode=inference_mode,
    )


#How: slices full prompt tokens to one decode token, adds cache_position, attention_mask, and flat cache tensors.
#Why: models.create_model uses this for decode_with_cache so the exported graph represents one-token generation.
def build_decode_with_cache_input(model: torch.nn.Module, input_: Any, input_cls: Any, cache_spec_cls: Any, model_dtype_fn: Any, drop_multimodal: bool) -> Any:
    kwargs = dict(input_.kwargs)
    token_key = "input_ids" if "input_ids" in kwargs else "decoder_input_ids"

    if token_key not in kwargs:
        raise ValueError(f"{constants.DECODE_WITH_CACHE_MODE} requires input_ids or decoder_input_ids")

    token_ids = kwargs[token_key]
    batch_size = int(token_ids.shape[0])
    past_sequence_length = int(token_ids.shape[1])
    cache_spec = cache_spec_cls.from_model(model=model, batch_size=batch_size, past_sequence_length=past_sequence_length)

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
        for multimodal_key in constants.MULTIMODAL_KEYS:
            kwargs.pop(multimodal_key, None)

    return input_cls(
        args=input_.args,
        kwargs=kwargs,
        modalities=input_.modalities,
        inference_mode=input_.inference_mode,
    )


#How: scans kwargs for the first tensor with a batch dimension and returns shape[0].
#Why: models.export_ uses this to size CacheSpec when wrapping cache-mode exports.
def infer_batch_size(kwargs: dict[str, Any]) -> int:
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])

    return 1


#How: prefers cache_position for decode, otherwise checks token or attention-mask sequence length.
#Why: models.export_ uses this to tell CacheSpec how much past context the exported cache represents.
def infer_past_sequence_length(input_: Any) -> int:
    if input_.inference_mode == constants.DECODE_WITH_CACHE_MODE and "cache_position" in input_.kwargs:
        return int(input_.kwargs["cache_position"][0].item())

    if "input_ids" in input_.kwargs:
        return int(input_.kwargs["input_ids"].shape[1])

    if "decoder_input_ids" in input_.kwargs:
        return int(input_.kwargs["decoder_input_ids"].shape[1])

    if "attention_mask" in input_.kwargs:
        return int(input_.kwargs["attention_mask"].shape[1])

    return 0


#How: opens the configured image path and converts it to RGB.
#Why: build_processor_kwargs needs a real image object when using HF processors for vision models.
def _load_image_asset():
    from PIL import Image
    return Image.open(constants.MODALITY_INPUT_PATH["vision"]).convert("RGB")


#How: reads the configured WAV file, mixes stereo to mono, and normalizes integer samples to float32.
#Why: build_processor_kwargs needs real waveform data when using HF processors for audio models.
def _load_audio_asset() -> np.ndarray:
    from scipy.io import wavfile

    _sample_rate, audio = wavfile.read(constants.MODALITY_INPUT_PATH["audio"])

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    return audio




#How: loads the requested processor, builds a prompt with modality tokens/assets, then keeps only tensor outputs.
#Why: build_input uses this for profile-driven processor exports so the graph sees realistic multimodal inputs.
def build_processor_kwargs(model_id: str, input_modalities: tuple[str, ...]) -> dict[str, Any]:
    processor = IP.PROCESSOR_MAP[model_id]

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
