"""Compare all strategy1_<variant>.jsonl against baseline.jsonl.

For each variant, compute:
  - overall verdict split
  - agreement rate with original (task_easy_hard) variant
  - P(attempt | probe=easy)   -- precision as positive signal
  - P(dodge   | probe=hard)   -- precision as negative / handoff signal
  - row count at each margin bin (how decisive is the probe)

Prints a summary table so variants can be ranked.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
OUT = HARNESS / "outputs"
BASE = OUT / "baseline.jsonl"

DODGE_RE = re.compile(
    r"^\s*("
    r"the image (shows|depicts|is|appears|contains|features)|"
    r"this is (a |an )?(photograph|picture|image|photo)|"
    r"a (man|woman|group|person|crowd)|"
    r"an audience|"
    r"the (photograph|picture|photo)|"
    r"a photograph of|"
    r"an image of"
    r")", re.IGNORECASE,
)


def classify(resp: str) -> str:
    r = (resp or "").strip()
    if not r: return "empty"
    return "dodge" if DODGE_RE.match(r) else "attempt"


def load(p: Path) -> dict:
    return {(r["image"], r["audio"]): r
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def main() -> None:
    baseline = load(BASE)
    for k, r in baseline.items():
        r["behavior"] = classify(r.get("response", ""))

    variant_files = sorted(OUT.glob("strategy1_*.jsonl"))
    # drop the original strategy1_prefill.jsonl if present (we have task_easy_hard now)
    variant_files = [p for p in variant_files if p.name != "strategy1_prefill.jsonl"]

    rows_per_variant: dict[str, dict] = {}
    for vf in variant_files:
        name = vf.stem.replace("strategy1_", "")
        rows_per_variant[name] = load(vf)

    if not rows_per_variant:
        print("no variant files found yet")
        return

    print(f"{'variant':<24} {'n':>5} {'easy%':>7} "
          f"{'P(att|easy)':>12} {'P(dodge|hard)':>14} "
          f"{'margin μ':>10} {'decisive%':>10}")
    print("-" * 95)
    for name, v_rows in rows_per_variant.items():
        n_total = len(v_rows)
        n_easy = sum(1 for r in v_rows.values() if r.get("verdict") == "easy")
        n_hard = sum(1 for r in v_rows.values() if r.get("verdict") == "hard")

        # Join with baseline
        joined = [(v_rows[k], baseline[k]) for k in v_rows if k in baseline]
        att_easy = sum(1 for v, b in joined if v["verdict"] == "easy" and b["behavior"] == "attempt")
        dge_hard = sum(1 for v, b in joined if v["verdict"] == "hard" and b["behavior"] == "dodge")
        j_easy = sum(1 for v, b in joined if v["verdict"] == "easy")
        j_hard = sum(1 for v, b in joined if v["verdict"] == "hard")

        margins = [abs(r.get("prob_margin", 0)) for r in v_rows.values()
                   if r.get("prob_margin") is not None]
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        decisive = sum(1 for m in margins if m > 0.9) / len(margins) if margins else 0.0

        p_att_e = att_easy / j_easy if j_easy else 0.0
        p_dge_h = dge_hard / j_hard if j_hard else 0.0

        print(f"{name:<24} {n_total:>5} {100*n_easy/n_total:>6.1f}% "
              f"{p_att_e:>12.3f} {p_dge_h:>14.3f} "
              f"{mean_margin:>10.3f} {100*decisive:>9.1f}%")

    # Agreement matrix between variants (Jaccard-like on "hard" set)
    print("\n=== pairwise agreement on verdict (% of joined rows where both agree) ===")
    names = list(rows_per_variant)
    common_keys = set.intersection(*(set(v) for v in rows_per_variant.values()))
    print(f"(on {len(common_keys)} common rows)\n")
    hdr = " " * 24 + " ".join(f"{n[:10]:>11}" for n in names)
    print(hdr)
    for n1 in names:
        r1 = rows_per_variant[n1]
        line = f"{n1:<24}"
        for n2 in names:
            r2 = rows_per_variant[n2]
            agree = sum(1 for k in common_keys if r1[k]["verdict"] == r2[k]["verdict"])
            line += f"{100*agree/len(common_keys):>10.1f}%"
        print(line)


if __name__ == "__main__":
    main()
