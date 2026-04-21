"""Print ~20 representative (image, audio, probabilities) rows from strategy 1.

Stratified to preserve maximum signal:
  - one clean row per audio tag (baseline behavior)
  - the worst ambiguous rows (|margin| smallest)
  - rows that flipped from hard->easy with more noise (same audio/image/noise,
    different SNR) to surface the high-noise fallback anomaly
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

S1 = Path(__file__).resolve().parent / "outputs" / "strategy1_prefill.jsonl"


def short(p: str) -> str:
    return Path(p).name


def fmt_row(r: dict) -> str:
    snr = "clean" if r["snr_db"] is None else f"{int(r['snr_db']):>3}"
    p_e = f"{r['p_easy']:.3f}"
    p_h = f"{r['p_hard']:.3f}"
    return (f"  {r['audio_tag']:<18}  {r['noise_tag']:<18}  snr={snr:>5}  "
            f"img={short(r['image']):<18}  P(easy)={p_e}  P(hard)={p_h}  -> {r['verdict']}")


def main() -> None:
    rows = [json.loads(l) for l in S1.read_text().splitlines() if l.strip()]

    # Section 1: one clean row per audio tag
    print("=== clean-audio baseline per audio tag ===")
    seen_clean = set()
    clean_rows = []
    for r in rows:
        if r["noise_tag"] != "clean": continue
        if r["audio_tag"] in seen_clean: continue
        seen_clean.add(r["audio_tag"])
        clean_rows.append(r)
    for r in sorted(clean_rows, key=lambda x: -x["p_easy"]):
        print(fmt_row(r))

    # Section 2: the 6 most ambiguous rows (|margin| smallest)
    print("\n=== most ambiguous (smallest |p_easy - p_hard|) ===")
    rows_sorted = sorted(rows, key=lambda r: abs(r["p_easy"] - r["p_hard"]))
    for r in rows_sorted[:6]:
        print(fmt_row(r))

    # Section 3: noise-flip examples — same (audio_tag, image, noise_tag) across SNRs
    # where verdict flips from hard -> easy at 0 dB
    print("\n=== hard -> easy flips as SNR drops to 0 dB ===")
    by_key = defaultdict(dict)
    for r in rows:
        if r["noise_tag"] == "clean": continue
        key = (r["audio_tag"], r["image"], r["noise_tag"])
        by_key[key][int(r["snr_db"])] = r

    flips = []
    for key, snr_map in by_key.items():
        if 0 in snr_map and 20 in snr_map:
            if snr_map[20]["verdict"] == "hard" and snr_map[0]["verdict"] == "easy":
                flips.append((key, snr_map))
    # sort by how dramatic the flip is
    flips.sort(key=lambda x: -(x[1][0]["p_easy"] - x[1][20]["p_easy"]))
    for key, snr_map in flips[:5]:
        for snr in sorted(snr_map):
            print(fmt_row(snr_map[snr]))
        print()


if __name__ == "__main__":
    main()
