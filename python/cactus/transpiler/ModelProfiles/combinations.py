from .models import Combinations

# The input and output strings are component names from components.py.
TEXT_EMBED_TO_DECODER = Combinations(input=("text_embed",), output="text_decoder")
DECODER_TO_LM_HEAD = Combinations(input=("text_decoder",), output="lm_head")
TEXT_VISION_TO_TOKEN_MERGE = Combinations(input=("text_embed", "vision_encoder"), output="token_merge")
TEXT_AUDIO_TO_TOKEN_MERGE = Combinations(input=("text_embed", "audio_encoder"), output="token_merge")
TEXT_VISION_AUDIO_TO_TOKEN_MERGE = Combinations(input=("text_embed", "vision_encoder", "audio_encoder"), output="token_merge")
TOKEN_MERGE_TO_DECODER = Combinations(input=("token_merge",), output="text_decoder")
AUDIO_TEXT_TO_DECODER = Combinations(input=("audio_encoder", "text_embed"), output="text_decoder")
AUDIO_ENCODER_TO_ASR_DECODER = Combinations(input=("audio_encoder",), output="asr_decoder")
AUDIO_ASR_DECODER_TO_ASR_HEAD = Combinations(input=("audio_encoder", "asr_decoder"), output="asr_head")
