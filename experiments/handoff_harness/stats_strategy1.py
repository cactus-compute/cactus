"""Quick summary stats of the strategy 1 prefill probe."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

S1 = Path(__file__).resolve().parent / "outputs" / "strategy1_prefill.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in S1.read_text().splitlines() if l.strip()]
    print(f"total rows: {len(rows)}\n")

    # Overall verdict split
    verdicts = defaultdict(int)
    for r in rows:
        verdicts[r.get("verdict") or "error"] += 1
    print("Overall probe verdicts:")
    for k, v in sorted(verdicts.items()):
        print(f"  {k:>6}: {v:>4}  ({100*v/len(rows):.1f}%)")
    print()

    # Confidence distribution
    margins = [abs(r["p_easy"] - r["p_hard"]) for r in rows if r.get("p_easy") is not None]
    print(f"|p_easy - p_hard| margin — "
          f"min={min(margins):.3f}  mean={mean(margins):.3f}  median={median(margins):.3f}  max={max(margins):.3f}")

    decisive = sum(1 for m in margins if m > 0.9)
    close = sum(1 for m in margins if m < 0.5)
    print(f"  decisive (|margin|>0.9): {decisive}/{len(margins)} ({100*decisive/len(margins):.1f}%)")
    print(f"  close    (|margin|<0.5): {close}/{len(margins)} ({100*close/len(margins):.1f}%)\n")

    # By audio tag
    print(f"{'audio_tag':<20} {'n':>4} {'easy':>5} {'hard':>5} {'mean p_easy':>12} {'mean margin':>12}")
    print("-" * 65)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in rows: by_tag[r["audio_tag"]].append(r)
    for tag in sorted(by_tag):
        sub = by_tag[tag]
        e = sum(1 for r in sub if r["verdict"] == "easy")
        h = sum(1 for r in sub if r["verdict"] == "hard")
        mpe = mean(r["p_easy"] for r in sub)
        mm = mean(r["p_easy"] - r["p_hard"] for r in sub)
        print(f"{tag:<20} {len(sub):>4} {e:>5} {h:>5} {mpe:>12.3f} {mm:>+12.3f}")
    print()

    # By noise/SNR
    print(f"{'noise_tag':<20} {'snr':>6} {'n':>4} {'easy':>5} {'hard':>5} {'mean p_easy':>12}")
    print("-" * 60)
    by_ns: dict[tuple[str, float|None], list[dict]] = defaultdict(list)
    for r in rows: by_ns[(r["noise_tag"], r["snr_db"])].append(r)
    for key in sorted(by_ns, key=lambda k: (k[0], k[1] if k[1] is not None else 1e9)):
        sub = by_ns[key]
        snr_s = "clean" if key[1] is None else f"{int(key[1])}"
        e = sum(1 for r in sub if r["verdict"] == "easy")
        h = sum(1 for r in sub if r["verdict"] == "hard")
        mpe = mean(r["p_easy"] for r in sub)
        print(f"{key[0]:<20} {snr_s:>6} {len(sub):>4} {e:>5} {h:>5} {mpe:>12.3f}")
    print()

    # Frame vs image
    print("By input type:")
    for subset_name, sub in [
        ("frames (audience)",    [r for r in rows if r.get("is_frame")]),
        ("static images",        [r for r in rows if not r.get("is_frame")]),
    ]:
        if not sub: continue
        e = sum(1 for r in sub if r["verdict"] == "easy")
        h = sum(1 for r in sub if r["verdict"] == "hard")
        mpe = mean(r["p_easy"] for r in sub)
        print(f"  {subset_name:<22} n={len(sub):>4} easy={e:>4} hard={h:>4} mean p_easy={mpe:.3f}")


if __name__ == "__main__":
    main()
