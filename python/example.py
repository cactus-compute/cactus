#!/usr/bin/env python3
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from cactus import CactusModel

WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"

with CactusModel(str(WEIGHTS_DIR / "lfm2-vl-450m")) as vlm:
    print("Loading LFM2-VL-450M...")

    messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
    response = vlm.complete(messages)
    print("\nCompletion:")
    print(json.dumps(json.loads(response), indent=2))

    embedding = vlm.embed("Hello world")
    print(f"\nText embedding dim: {len(embedding)}")

    embedding = vlm.image_embed(str(ASSETS_DIR / "test_monkey.png"))
    print(f"\nImage embedding dim: {len(embedding)}")

    messages = json.dumps([{"role": "user", "content": "Describe this image", "images": [str(ASSETS_DIR / "test_monkey.png")]}])
    response = vlm.complete(messages)
    print("\nVLM Image Description:")
    print(json.dumps(json.loads(response), indent=2))

    vlm.reset()

with CactusModel(str(WEIGHTS_DIR / "whisper-small")) as whisper:
    print("\nLoading whisper-small...")

    whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    response = whisper.transcribe(str(ASSETS_DIR / "test.wav"), prompt=whisper_prompt)
    print("Transcription:")
    print(json.dumps(json.loads(response), indent=2))

    embedding = whisper.audio_embed(str(ASSETS_DIR / "test.wav"))
    print(f"\nAudio embedding dim: {len(embedding)}")

print("\nDone!")
