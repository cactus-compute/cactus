from typing import Any
from huggingface_hub import hf_hub_download
from . import constants
from pathlib import Path


def _model_json_dir(model_profile: str) -> Path:
    output_dir = constants.CONVERTER_JSON_DIR / model_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _local_tokenizer_path(model_id: str, model_profile: str) -> str:
    output_dir = _model_json_dir(model_profile)
    tokenizer_path = output_dir / "tokenizer.json"

    if tokenizer_path.exists():
        return str(tokenizer_path)

    return str(Path(hf_hub_download(repo_id=model_id, filename="tokenizer.json", local_dir=output_dir, token=constants.token)))


#How: returns a config copy without the HF *_type key that component constructors do not expect.
#Why: _build_gemma4_processor uses this when manually constructing Gemma processor pieces from processor_config.json.
def _without_type_key(config: dict[str, Any], type_key: str) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != type_key}


def gemma4_processor(model_id: str, configs: dict[str, dict[str, Any]], model_profile: str):
    
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gemma4 import Gemma4AudioFeatureExtractor, Gemma4ImageProcessor, Gemma4Processor, Gemma4VideoProcessor

    processor_config = configs.get("processor_config.json")
    if processor_config is None or processor_config.get("processor_class") != "Gemma4Processor":
        return default_processor(model_id=model_id, configs=configs, model_profile=model_profile)

    tokenizer_path = _local_tokenizer_path(model_id=model_id, model_profile=model_profile)

    tokenizer_config = configs.get("tokenizer_config.json", {})
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token=tokenizer_config.get("bos_token", "<bos>"), eos_token=tokenizer_config.get("eos_token", "<eos>"), unk_token=tokenizer_config.get("unk_token", "<unk>"), pad_token=tokenizer_config.get("pad_token", "<pad>"))

    for token_name in ("image_token", "audio_token", "boi_token", "eoi_token", "boa_token", "eoa_token"):
        token_value = tokenizer_config.get(token_name)
        if token_value is not None:
            setattr(tokenizer, token_name, token_value)

    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    tokenizer.audio_token_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_token)
    image_processor = Gemma4ImageProcessor(**_without_type_key(processor_config.get("image_processor", {}), "image_processor_type"))
    feature_extractor = Gemma4AudioFeatureExtractor(**_without_type_key(processor_config.get("feature_extractor", {}), "feature_extractor_type"))
    video_processor = Gemma4VideoProcessor(**_without_type_key(processor_config.get("video_processor", {}), "video_processor_type"))

    return Gemma4Processor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_seq_length=processor_config.get("image_seq_length", 280),
        audio_seq_length=processor_config.get("audio_seq_length", 750),
        audio_ms_per_token=processor_config.get("audio_ms_per_token", 40),
    )


#How: loops over hf_load_kwargs attempts and returns the first AutoProcessor that loads.
#Why: build_processor_kwargs needs an HF processor to convert real sample assets into model-ready tensors.
def default_processor(model_id: str, configs: dict[str, dict[str, Any]], model_profile: str):
    from transformers import AutoProcessor

    processor_kwargs: dict[str, Any] = {"trust_remote_code": True, "token": constants.token}

    try:
        return AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    except Exception as e:
        raise RuntimeError(f"Unable to load processor for {model_id}")


def text_tokenizer_processor(model_id: str, configs: dict[str, dict[str, Any]], model_profile: str):
    from transformers import PreTrainedTokenizerFast

    tokenizer_config = configs.get("tokenizer_config.json", {})
    return PreTrainedTokenizerFast(
        tokenizer_file=_local_tokenizer_path(model_id=model_id, model_profile=model_profile),
        bos_token=tokenizer_config.get("bos_token"),
        eos_token=tokenizer_config.get("eos_token"),
        pad_token=tokenizer_config.get("pad_token"),
    )


PROCESSOR_MAP = {
    "google/gemma-4-E2B": gemma4_processor,
    "openai/whisper-tiny": default_processor,
    "nvidia/parakeet-tdt-0.6b-v3": default_processor,
    "LiquidAI/LFM2-VL-3B": default_processor,
    "Qwen/Qwen2.5-0.5B": default_processor,
    "LiquidAI/LFM2.5-8B-A1B": text_tokenizer_processor,
}
