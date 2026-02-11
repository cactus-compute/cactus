#!/usr/bin/env python3
"""
TrOCR OCR test: cactus_complete with a message that includes an image.

Prerequisites: source setup; build Cactus (cactus/build); run from repo root.
Downloads microsoft/trocr-small-printed if needed, then runs completion and prints result.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights" / "microsoft--trocr-small-printed"
TROCR_MODEL_ID = "microsoft/trocr-small-printed"
# Use an image with text for OCR; fallback to test image if no text image
TEST_IMAGE = REPO_ROOT / "tests" / "assets" / "test_monkey.png"


def ensure_trocr_model():
    """Download and convert TrOCR model if weights dir or config.txt missing."""
    if WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "config.txt").exists():
        print(f"TrOCR weights found at {WEIGHTS_DIR}")
        return str(WEIGHTS_DIR)
    print(f"Downloading and converting {TROCR_MODEL_ID}...")
    # Invoke CLI without loading cactus bindings (which require the built library)
    sys.path.insert(0, str(REPO_ROOT / "python" / "src"))
    try:
        from cli import main as cli_main
        old_argv = sys.argv
        sys.argv = ["cactus", "download", TROCR_MODEL_ID, "--precision", "FP16"]
        try:
            cli_main()
        except SystemExit as e:
            code = e.code if e.code is not None else 0
            if code != 0:
                print("Download failed with exit code", code)
                return None
        finally:
            sys.argv = old_argv
    except Exception as e:
        print("Download failed:", e)
        return None
    print("Download and conversion done.")
    return str(WEIGHTS_DIR)


def run_ocr_test(model_path: str, image_path: str) -> bool:
    """Run cactus_complete with one user message containing the image. Returns True on success."""
    sys.path.insert(0, str(REPO_ROOT / "python" / "src"))
    try:
        from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_get_last_error
    except Exception as e:
        print("Could not import cactus (library not built or not on path):", e)
        print("Build the library first: cd cactus && mkdir -p build && cd build && cmake .. && make")
        return False

    image_path = str(Path(image_path).resolve())
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return False

    model = cactus_init(model_path.encode(), None, False)
    if not model:
        err = cactus_get_last_error()
        print("cactus_init failed:", err.decode() if err else "unknown")
        return False

    messages = [
        {"role": "user", "content": "", "images": [image_path]}
    ]
    try:
        response_json = cactus_complete(
            model,
            messages,
            max_tokens=64,
            temperature=0.0,
        )
        cactus_destroy(model)
    except Exception as e:
        print("cactus_complete failed:", e)
        cactus_destroy(model)
        return False

    data = json.loads(response_json)
    if data.get("error"):
        print("Error:", data["error"])
        return False
    print("OCR response:", data.get("response", "(empty)"))
    print("Confidence:", data.get("confidence"), "| Tokens:", data.get("decode_tokens"))
    return data.get("success", False)


def main():
    model_path = ensure_trocr_model()
    if not model_path:
        return 1

    if not TEST_IMAGE.exists():
        print(f"Test image not found: {TEST_IMAGE}")
        return 1

    ok = run_ocr_test(model_path, TEST_IMAGE)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
