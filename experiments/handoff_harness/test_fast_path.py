"""Verify use_multimodal_cache (encoder output cache) preserves probe output
and measure the speed delta vs the default path.

For each row we run 2 configurations:
  none  : default (no encoder cache)
  cache : use_multimodal_cache=True

The cache config should produce bit-identical outputs and save the vision +
audio encoder runtime on repeat calls for the same image/audio.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HARNESS))

from src.cactus import (                              # noqa: E402
    cactus_init, cactus_destroy, cactus_reset,
    cactus_tokenize, cactus_score_candidates,
)
from io_utils import pcm_bytes_from_wav              # noqa: E402
from run_strategy1_prefill import (                  # noqa: E402
    USER_PROMPT, PREFILL_SUFFIX, build_messages,
)


WEIGHTS = str(ROOT / "weights" / "gemma-4-e2b-it")
MANIFEST = HARNESS / "manifest.jsonl"

CONFIGS = [
    ("none",  False),
    ("cache", True),
]


def opts(use_mm_cache: bool) -> str:
    return json.dumps({
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": 1,
        "auto_handoff": False,
        "enable_thinking_if_supported": False,
        "telemetry_enabled": False,
        "use_multimodal_cache": use_mm_cache,
    })


def resolve_candidates(model) -> list[int]:
    ids: list[int] = []
    seen = set()
    for form in ["easy", " easy", "Easy", " Easy",
                 "hard", " hard", "Hard", " Hard"]:
        toks = cactus_tokenize(model, form)
        if toks and toks[0] not in seen:
            seen.add(toks[0])
            ids.append(toks[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=12,
                    help="how many manifest rows to exercise (default 12)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    # pick a small mix: clean + noisy + different audio tags
    subset = rows[::max(1, len(rows) // args.rows)][:args.rows]

    print(f"loading model {WEIGHTS}")
    model = cactus_init(WEIGHTS, None, False)

    all_ids = resolve_candidates(model)
    split = len(all_ids) // 2

    results: dict[str, list[dict]] = {name: [] for name, _ in CONFIGS}
    for i, row in enumerate(subset):
        pcm = pcm_bytes_from_wav(Path(row["audio"]))
        messages = build_messages(row["image"])
        for name, cache in CONFIGS:
            cactus_reset(model)
            # For the cache config, warm up with one call so the cache fills,
            # then time the repeat call (which is the realistic reuse scenario).
            if cache:
                _ = cactus_score_candidates(
                    model, messages, opts(True), None,
                    PREFILL_SUFFIX, all_ids, pcm_data=pcm,
                )
                cactus_reset(model)
            t0 = time.time()
            probs = cactus_score_candidates(
                model, messages, opts(cache), None,
                PREFILL_SUFFIX, all_ids, pcm_data=pcm,
            )
            dt = time.time() - t0
            p_easy = sum(probs[:split])
            p_hard = sum(probs[split:])
            results[name].append({
                "row_idx": i,
                "audio_tag": row["audio_tag"],
                "noise_tag": row["noise_tag"],
                "snr_db": row["snr_db"],
                "p_easy": p_easy,
                "p_hard": p_hard,
                "elapsed_s": dt,
            })
        print(f"[{i+1}/{len(subset)}] {row['audio_tag']}/{row['noise_tag']}/{row['snr_db']}  "
              f"none={results['none'][-1]['elapsed_s']:.2f}s  "
              f"cache={results['cache'][-1]['elapsed_s']:.2f}s")

    cactus_destroy(model)

    # Summaries
    print("\n=== timing (mean / median seconds per row) ===")
    for name, _ in CONFIGS:
        times = [r["elapsed_s"] for r in results[name]]
        print(f"  {name:<6}  mean={statistics.mean(times):.3f}s  "
              f"median={statistics.median(times):.3f}s")

    baseline = {r["row_idx"]: r for r in results["none"]}
    deltas = [abs(r["p_easy"] - baseline[r["row_idx"]]["p_easy"])
              for r in results["cache"]]
    print(f"\n=== correctness: |Δ p_easy| cache vs none ===")
    print(f"  mean={statistics.mean(deltas):.4f}  max={max(deltas):.4f}")
    if max(deltas) > 0:
        for idx in range(len(subset)):
            r = next(x for x in results["cache"] if x["row_idx"] == idx)
            b = baseline[idx]
            if abs(r["p_easy"] - b["p_easy"]) > 0:
                print(f"  row {idx} ({b['audio_tag']}/{b['noise_tag']}/{b['snr_db']}): "
                      f"p_easy {b['p_easy']:.6f} -> {r['p_easy']:.6f}")


if __name__ == "__main__":
    main()
