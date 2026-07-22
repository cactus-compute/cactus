from transformers import AutoModel, AutoModelForCausalLM, AutoModelForCTC, AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
from . import input_utils as IU

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

CACHE_INFERENCE_MODES = {IU.PREFILL_WITH_CACHE_MODE, IU.DECODE_WITH_CACHE_MODE}
DYNAMIC_CACHE_POLICY = IU.DYNAMIC_CACHE_POLICY
DROP_MULTIMODAL_ON_DECODE_POLICY = "drop_multimodal_on_decode"
