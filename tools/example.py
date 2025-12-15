#!/usr/bin/env python3
"""
Cactus Python FFI Example

Usage:
  1. cactus build
  2. cactus download LiquidAI/LFM2-VL-450M
  3. cactus download openai/whisper-small
  4. cd tools && python example.py
"""

import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from cactus_ffi import (
    cactus_init,
    cactus_complete,
    cactus_transcribe,
    cactus_embed,
    cactus_image_embed,
    cactus_audio_embed,
    cactus_reset,
    cactus_destroy
)

WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"

# Load model
print("Loading LFM2-VL-450M...")
vlm = cactus_init(str(WEIGHTS_DIR / "lfm2-vl-450m"), context_size=2048)

# Text completion
messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
response = cactus_complete(vlm, messages)
print("\nCompletion:")
print(json.dumps(json.loads(response), indent=2))

# Text embedding
embedding = cactus_embed(vlm, "Hello world")
print(f"\nText embedding dim: {len(embedding)}")

# Image embedding
embedding = cactus_image_embed(vlm, str(ASSETS_DIR / "test_monkey.png"))
print(f"\nImage embedding dim: {len(embedding)}")

# VLM - describe image
messages = json.dumps([{"role": "user", "content": "Describe this image", "images": [str(ASSETS_DIR / "test_monkey.png")]}])
response = cactus_complete(vlm, messages)
print("\nVLM Image Description:")
print(json.dumps(json.loads(response), indent=2))

cactus_reset(vlm)
cactus_destroy(vlm)

# Transcription
print("\nLoading whisper-small...")
whisper = cactus_init(str(WEIGHTS_DIR / "whisper-small"), context_size=448)
whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, str(ASSETS_DIR / "test.wav"), prompt=whisper_prompt)
print("Transcription:")
print(json.dumps(json.loads(response), indent=2))

# Audio embedding
embedding = cactus_audio_embed(whisper, str(ASSETS_DIR / "test.wav"))
print(f"\nAudio embedding dim: {len(embedding)}")

cactus_destroy(whisper)

print("\nDone!")
