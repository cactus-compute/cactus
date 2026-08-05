from transformers import AutoModel, AutoModelForCausalLM, AutoModelForCTC, AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
import os
from pathlib import Path

token = os.environ.get("HF_TOKEN")

LOAD_STRATEGIES = {
    "image_text_to_text": (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel),
    "image_text_to_text_strict": (AutoModelForImageTextToText, AutoModelForCausalLM),
    "speech_seq2seq": (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel),
    "ctc": (AutoModelForCTC, AutoModel),
    "parakeet_tdt": (AutoModel, AutoModelForCTC),
    "causal_lm": (AutoModelForCausalLM, AutoModel),
}

CACHE_INFERENCE_MODES = {"prefill_with_cache", "decode_with_cache"}
DYNAMIC_CACHE_POLICY = "dynamic_cache"
DROP_MULTIMODAL_ON_DECODE_POLICY = "drop_multimodal_on_decode"

SYNTHETIC_INPUT_STRATEGY = "synthetic"
DECODE_WITH_CACHE_MODE = "decode_with_cache"

CONVERTER_JSON_DIR = Path(__file__).resolve().parent / "jsons"
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"

MULTIMODAL_KEYS = (
    "pixel_values",
    "pixel_attention_mask",
    "spatial_shapes",
    "pixel_values_videos",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "video_position_ids",
    "token_type_ids",
    "mm_token_type_ids",
)

MODALITY_INPUT_PATH = {
    "vision": ASSET_DIR / "test_monkey.png",
    "audio": ASSET_DIR / "test.wav",
}
