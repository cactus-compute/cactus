"""Run gemma-4-e2b-it over the manifest; log response + confidence per row.

The current (baseline) handoff strategy is the rolling-entropy confidence in
cactus_complete. We disable actual cloud dispatch (auto_handoff=false) so every
row returns a LOCAL output, and we log the confidence that the built-in
strategy would have used to decide a handoff.

Usage:
    python run_baseline.py [--weights /path/to/weights] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HARNESS))

from src.cactus import cactus_init, cactus_destroy, cactus_complete, cactus_reset  # noqa: E402
from io_utils import pcm_bytes_from_wav  # noqa: E402

DEFAULT_WEIGHTS = str(ROOT / "weights" / "gemma-4-e2b-it")
MANIFEST = HARNESS / "manifest.jsonl"
OUT_DIR = HARNESS / "outputs"

SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. The user will show you an image "
    "and speak a short instruction. Follow the spoken instruction with respect "
    "to the image, concisely."
)

OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": 256,
    "auto_handoff": False,           # do NOT actually call cloud
    "confidence_threshold": 0.7,     # logged only, not acted on
    "enable_thinking_if_supported": False,
    "telemetry_enabled": False,
}


def build_messages(image_path: str) -> str:
    return json.dumps([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "", "images": [image_path]},
    ])


def load_manifest() -> list[dict]:
    return [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0, help="0 = run all")
    ap.add_argument("--out", default=str(OUT_DIR / "baseline.jsonl"))
    args = ap.parse_args()

    rows = load_manifest()
    if args.limit:
        rows = rows[: args.limit]

    print(f"loading model {args.weights}")
    model = cactus_init(args.weights, None, False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            cactus_reset(model)
            pcm = pcm_bytes_from_wav(Path(row["audio"]))
            messages = build_messages(row["image"])
            t0 = time.time()
            try:
                resp_json = cactus_complete(
                    model, messages, json.dumps(OPTIONS), None, None, pcm_data=pcm
                )
                resp = json.loads(resp_json)
                out = {
                    **row,
                    "response": resp.get("response", ""),
                    "confidence": resp.get("confidence"),
                    "cloud_handoff": resp.get("cloud_handoff"),
                    "decode_tps": resp.get("decode_tps"),
                    "elapsed_s": round(time.time() - t0, 2),
                    "error": None,
                }
                n_ok += 1
            except Exception as e:
                out = {**row, "response": "", "confidence": None, "error": str(e),
                       "elapsed_s": round(time.time() - t0, 2)}
            f.write(json.dumps(out) + "\n")
            f.flush()
            conf = out.get("confidence")
            print(f"[{i+1}/{len(rows)}] {row['audio_tag']}/{row['noise_tag']}/snr={row['snr_db']} "
                  f"conf={conf} -> {out['response'][:60]!r}")

    cactus_destroy(model)
    print(f"\ndone: {n_ok}/{len(rows)} ok  -> {args.out}")


if __name__ == "__main__":
    main()
