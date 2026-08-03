from transformers import AutoModel, AutoModelForCausalLM, AutoModelForCTC, AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
import os
from pathlib import Path

token = os.environ.get("HF_TOKEN")

#Names of component configs in config.txt (put in constants to allow future expansion to tuple to support different names)
TEXT_CONFIG_NAME = "text_config"
VISION_CONFIG_NAME = "vision_config"
AUDIO_CONFIG_NAME = "audio_config"

LOAD_STRATEGIES = {
    "image_text_to_text": (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel),
    "image_text_to_text_strict": (AutoModelForImageTextToText, AutoModelForCausalLM),
    "speech_seq2seq": (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel),
    "ctc": (AutoModelForCTC, AutoModel),
    "causal_lm": (AutoModelForCausalLM, AutoModel),
}

CACHE_INFERENCE_MODES = {"prefill_with_cache", "decode_with_cache"}
DYNAMIC_CACHE_POLICY = "dynamic_cache"
DROP_MULTIMODAL_ON_DECODE_POLICY = "drop_multimodal_on_decode"

CONFIG_SEARCH_SECTIONS = (
    (),
    ("text_config",),
    ("vision_config",),
    ("audio_config",),
    ("encoder",),
    ("decoder",),
    ("model_config",),
)

DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_PAST_SEQUENCE_LENGTH = 8
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_BIN_NUM = 80
DEFAULT_SOURCE_POSITION = 1500

SYNTHETIC_INPUT_STRATEGY = "synthetic"
PROCESSOR_INPUT_STRATEGY = "processor"

PREFILL_WITH_CACHE_MODE = "prefill_with_cache"
DECODE_WITH_CACHE_MODE = "decode_with_cache"
DYNAMIC_CACHE_POLICY = "dynamic_cache"

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
    "mm_token_type_ids",
)

#Locations to local sample inputs used by processor-backed multimodal exports.
MODALITY_INPUT_PATH = {
    "vision": ASSET_DIR / "test_monkey.png",
    "audio": ASSET_DIR / "test.wav",
}
