#!/usr/bin/env python3
"""Reproducer for GitHub issue #584.

Gemma 4 E4B audio input via cactus_complete crashes with:
    unordered_map::at: key not found

The NPU audio encoder loads at init (skipping CPU weight loading),
then fails at runtime on the actual input shape. The CPU fallback
references weight node IDs that were never populated (default 0),
and node 0 does not exist in the graph.

Usage:
    python tests/repro_584.py [weights_dir]
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))
from cactus import cactus_init, cactus_complete, cactus_destroy

WEIGHTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights/gemma-4-e4b-it"
WAV_PATH = "/tmp/repro_584.wav"


def generate_test_wav():
    subprocess.run(
        ["say", "-v", "Samantha", "-o", "/tmp/repro_584.aiff", "What is the capital of France?"],
        check=True,
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         "/tmp/repro_584.aiff", WAV_PATH],
        check=True,
    )


def main():
    generate_test_wav()

    print(f"Loading model from {WEIGHTS_DIR} ...")
    model = cactus_init(WEIGHTS_DIR, None, False)
    print("Model loaded.\n")

    opts = json.dumps({"max_tokens": 30})

    # 1) Text-only control -- should succeed
    msgs_text = json.dumps([{"role": "user", "content": "What is the capital of France?"}])
    raw = cactus_complete(model, msgs_text, opts, None, None)
    text_resp = json.loads(raw).get("response", "")
    print(f"[PASS] Text-only: {text_resp}")

    # 2) Audio via pcm_data -- expected to crash before fix
    with wave.open(WAV_PATH, "rb") as w:
        pcm = w.readframes(w.getnframes())

    msgs_audio = json.dumps([
        {"role": "user", "content": "Answer the question I just spoke in one short sentence."}
    ])

    try:
        raw = cactus_complete(model, msgs_audio, opts, None, None, pcm)
        resp = json.loads(raw).get("response", "")
        print(f"[PASS] Audio pcm_data path: {resp}")
    except Exception as e:
        print(f"[FAIL] Audio pcm_data path: {e}")

    # 3) Audio via message "audio" field -- expected to crash before fix
    msgs_field = json.dumps([{
        "role": "user",
        "content": "Answer the question I just spoke in one short sentence.",
        "audio": [WAV_PATH],
    }])

    try:
        raw = cactus_complete(model, msgs_field, opts, None, None)
        resp = json.loads(raw).get("response", "")
        print(f"[PASS] Audio field path: {resp}")
    except Exception as e:
        print(f"[FAIL] Audio field path: {e}")

    cactus_destroy(model)


if __name__ == "__main__":
    main()
