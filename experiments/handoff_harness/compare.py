"""Join strategy1_prefill.jsonl with baseline.jsonl on (image, audio) and print
per-row comparison: probe verdict + probabilities vs. the actual completion
and its rolling-entropy confidence.

Usage:
    python compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
OUT = HARNESS / "outputs"
S1_PATH = OUT / "strategy1_prefill.jsonl"
BASE_PATH = OUT / "baseline.jsonl"
JOIN_PATH = OUT / "comparison.jsonl"


def _load(p: Path) -> dict:
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {(r["image"], r["audio"]): r for r in rows}


def main() -> None:
    s1 = _load(S1_PATH)
    base = _load(BASE_PATH)

    keys = sorted(set(s1) & set(base))
    print(f"{len(keys)} rows joined (strategy1={len(s1)}, baseline={len(base)})")

    merged = []
    n_easy = n_hard = 0
    for k in keys:
        s1r, br = s1[k], base[k]
        row = {
            "image": k[0],
            "audio": k[1],
            "audio_tag":  s1r.get("audio_tag"),
            "noise_tag":  s1r.get("noise_tag"),
            "snr_db":     s1r.get("snr_db"),
            "is_frame":   s1r.get("is_frame"),
            # strategy 1 (prefill probe)
            "p_easy":     s1r.get("p_easy"),
            "p_hard":     s1r.get("p_hard"),
            "p_margin":   s1r.get("prob_margin"),
            "probe_verdict": s1r.get("verdict"),
            # baseline
            "response":       br.get("response", ""),
            "rolling_confidence": br.get("confidence"),
            "decode_tps":     br.get("decode_tps"),
        }
        if row["probe_verdict"] == "easy": n_easy += 1
        elif row["probe_verdict"] == "hard": n_hard += 1
        merged.append(row)

    JOIN_PATH.write_text("\n".join(json.dumps(r) for r in merged) + "\n")

    # Per-category summary
    print(f"\nProbe verdicts: easy={n_easy}  hard={n_hard}")
    print(f"\n{'audio':>11} {'noise':>18} {'snr':>5} {'verdict':>7} "
          f"{'p(easy)':>8} {'p(hard)':>8} {'rolling_conf':>13}  response")
    print("-" * 130)
    for r in merged:
        snr = "clean" if r["snr_db"] is None else f"{int(r['snr_db']):>3}"
        p_e = f"{r['p_easy']:.3f}" if r['p_easy'] is not None else "   -  "
        p_h = f"{r['p_hard']:.3f}" if r['p_hard'] is not None else "   -  "
        rc = f"{r['rolling_confidence']:.4f}" if r['rolling_confidence'] is not None else "   -  "
        resp = (r["response"] or "").replace("\n", " ")[:60]
        print(f"{r['audio_tag']:>11} {r['noise_tag']:>18} {snr:>5} "
              f"{r['probe_verdict'] or '-':>7} {p_e:>8} {p_h:>8} {rc:>13}  {resp}")

    print(f"\njoined -> {JOIN_PATH}")


if __name__ == "__main__":
    main()
