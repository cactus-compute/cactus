"""Strategy 1: prefill probe.

Prompt the model with the image + audio, then prefill an unfinished assistant
turn ending in "This task is ". Read the next-token distribution and compare
P(easy) vs P(hard). Whichever is higher is the model's self-assessed
difficulty; low-confidence inputs (P(hard) high) would be candidates for
cloud handoff.

Usage:
    python run_strategy1_prefill.py [--weights ...] [--limit N]
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

from src.cactus import (                 # noqa: E402
    cactus_init, cactus_destroy, cactus_reset,
    cactus_tokenize, cactus_score_candidates,
)
from io_utils import pcm_bytes_from_wav  # noqa: E402

DEFAULT_WEIGHTS = str(ROOT / "weights" / "gemma-4-e2b-it")
MANIFEST = HARNESS / "manifest.jsonl"
DEFAULT_OUT = HARNESS / "outputs" / "strategy1_prefill.jsonl"

USER_PROMPT = (
    "Look at the image and listen to the spoken instruction. Assess whether "
    "you can complete the instruction reliably given these inputs. Fill in "
    "the blank with a single plain word, no markdown, no bold, no "
    "punctuation, no quotes: \"This task is ___\". Use \"easy\" if you can "
    "confidently carry out the instruction, or \"hard\" if the inputs are "
    "unclear, the instruction is ambiguous, or you are unsure of the answer."
)

PREFILL_SUFFIX = "This task is"

OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_tokens": 1,
    "auto_handoff": False,
    "enable_thinking_if_supported": False,
    "telemetry_enabled": False,
}


def build_messages(image_path: str) -> str:
    return json.dumps([
        {"role": "user", "content": USER_PROMPT, "images": [image_path]},
    ])


def resolve_candidate_tokens(model) -> dict[str, list[int]]:
    """Tokenize a few surface forms of 'easy'/'hard' and use all variants as
    candidates. We aggregate probability mass across variants per class."""
    classes: dict[str, list[int]] = {"easy": [], "hard": []}
    for cls in classes:
        seen = set()
        # The prefill ends with "This task is " (trailing space), so the
        # next token is usually the bare word, but models vary. Probe a few.
        for form in [cls, " " + cls, cls.capitalize(), " " + cls.capitalize()]:
            toks = cactus_tokenize(model, form)
            if toks:
                tid = toks[0]
                if tid not in seen:
                    seen.add(tid)
                    classes[cls].append(tid)
    return classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"loading model {args.weights}")
    model = cactus_init(args.weights, None, False)

    classes = resolve_candidate_tokens(model)
    easy_ids = classes["easy"]
    hard_ids = classes["hard"]
    all_ids = easy_ids + hard_ids
    print(f"easy token ids: {easy_ids}")
    print(f"hard token ids: {hard_ids}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            cactus_reset(model)
            pcm = pcm_bytes_from_wav(Path(row["audio"]))
            messages = build_messages(row["image"])
            t0 = time.time()
            try:
                probs = cactus_score_candidates(
                    model, messages, json.dumps(OPTIONS), None,
                    PREFILL_SUFFIX, all_ids, pcm_data=pcm,
                )
                p_easy = sum(probs[: len(easy_ids)])
                p_hard = sum(probs[len(easy_ids):])
                verdict = "easy" if p_easy >= p_hard else "hard"
                out = {
                    **row,
                    "p_easy": p_easy,
                    "p_hard": p_hard,
                    "verdict": verdict,
                    "prob_margin": p_easy - p_hard,
                    "per_token_probs": probs,
                    "elapsed_s": round(time.time() - t0, 2),
                    "error": None,
                }
                n_ok += 1
            except Exception as e:
                out = {**row, "p_easy": None, "p_hard": None, "verdict": None,
                       "error": str(e), "elapsed_s": round(time.time() - t0, 2)}
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"[{i+1}/{len(rows)}] {row['audio_tag']}/{row['noise_tag']}/snr={row['snr_db']} "
                  f"P(easy)={out.get('p_easy')}, P(hard)={out.get('p_hard')} -> {out.get('verdict')}")

    cactus_destroy(model)
    print(f"\ndone: {n_ok}/{len(rows)} ok -> {args.out}")


if __name__ == "__main__":
    main()
