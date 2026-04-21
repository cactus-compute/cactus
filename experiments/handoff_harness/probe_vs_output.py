"""Cross-tab the strategy 1 probe verdict with a cheap classifier of the actual
model response: did it attempt an answer to the spoken instruction, or did it
dodge into a generic image description?

Heuristic "dodge" pattern: response starts with one of a handful of phrases
that clearly indicate image-describe-only output.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

COMP = Path(__file__).resolve().parent / "outputs" / "comparison.jsonl"

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
    if not r:
        return "empty"
    if DODGE_RE.match(r):
        return "dodge"
    return "attempt"


def main() -> None:
    rows = [json.loads(l) for l in COMP.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} rows\n")

    # Label each row
    for r in rows:
        r["behavior"] = classify(r["response"])

    # Overall confusion matrix: probe verdict vs behavior
    cm = defaultdict(int)
    for r in rows:
        cm[(r["probe_verdict"], r["behavior"])] += 1
    print("=== Confusion: probe_verdict × actual_behavior ===")
    print(f"{'verdict':<10} {'attempt':>8} {'dodge':>8} {'empty':>8}")
    for v in ["easy", "hard", None]:
        label = v if v else "null"
        row = [cm.get((v, b), 0) for b in ["attempt", "dodge", "empty"]]
        print(f"{label:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    print()

    # When probe says 'easy', what fraction are actual attempts?
    n_easy = sum(1 for r in rows if r["probe_verdict"] == "easy")
    n_easy_attempt = sum(1 for r in rows if r["probe_verdict"] == "easy" and r["behavior"] == "attempt")
    n_hard = sum(1 for r in rows if r["probe_verdict"] == "hard")
    n_hard_dodge = sum(1 for r in rows if r["probe_verdict"] == "hard" and r["behavior"] == "dodge")
    print(f"P(attempt | probe=easy) = {n_easy_attempt}/{n_easy} = {n_easy_attempt/n_easy:.3f}")
    print(f"P(dodge   | probe=hard) = {n_hard_dodge}/{n_hard} = {n_hard_dodge/n_hard:.3f}\n")

    # Compare with baseline's rolling_confidence as a handoff signal.
    # Convention: existing strategy hands off when rolling_confidence < 0.7.
    threshold = 0.7
    # For each row, count how often rolling_confidence would trigger vs probe.
    rc_trigger = [r for r in rows if (r.get("rolling_confidence") or 0) < threshold]
    print(f"Rows where rolling_confidence<{threshold}: {len(rc_trigger)}/{len(rows)} "
          f"({100*len(rc_trigger)/len(rows):.1f}%)")
    # Of those, how many were dodges?
    rc_dodge = [r for r in rc_trigger if r["behavior"] == "dodge"]
    if rc_trigger:
        print(f"  of which dodge: {len(rc_dodge)} ({100*len(rc_dodge)/len(rc_trigger):.1f}%)")
    print()

    probe_trigger = [r for r in rows if r["probe_verdict"] == "hard"]
    probe_dodge = [r for r in probe_trigger if r["behavior"] == "dodge"]
    print(f"Rows where probe=hard: {len(probe_trigger)}/{len(rows)} "
          f"({100*len(probe_trigger)/len(rows):.1f}%)")
    print(f"  of which dodge: {len(probe_dodge)} ({100*len(probe_dodge)/len(probe_trigger):.1f}%)")
    print()

    # By audio_tag × behavior
    print("=== behavior breakdown per audio_tag ===")
    print(f"{'audio_tag':<20} {'n':>4} {'attempt':>8} {'dodge':>8} {'empty':>6}  probe-easy  probe-hard")
    per_tag = defaultdict(list)
    for r in rows: per_tag[r["audio_tag"]].append(r)
    for tag in sorted(per_tag):
        sub = per_tag[tag]
        c = Counter(r["behavior"] for r in sub)
        e = sum(1 for r in sub if r["probe_verdict"] == "easy")
        h = sum(1 for r in sub if r["probe_verdict"] == "hard")
        print(f"{tag:<20} {len(sub):>4} {c.get('attempt',0):>8} "
              f"{c.get('dodge',0):>8} {c.get('empty',0):>6} "
              f"     {e:>4}        {h:>4}")


if __name__ == "__main__":
    main()
