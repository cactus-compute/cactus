"""Milestone 0 spike: prove the probe splits turns BEFORE building the TUI.

Loads the probe bundle (weights/gemma-4), runs the email sandbox over a query,
and prints per-turn confidence / cloud_handoff / reason. Run at threshold 0.0 to
measure the natural confidence distribution (probe runs but never escalates);
run at 0.5 / 0.815 to see where handoff actually fires.

    ./venv/bin/python demos/hybrid_handoff/spike.py [--threshold 0.0] [--query "..."] [--sandbox email]
"""
from __future__ import annotations

import argparse
import os

from cactus_env import DEFAULT_BUNDLE, have_cloud_key, load_env, setup_cactus_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--query", default=None)
    ap.add_argument("--sandbox", default="email")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    load_env()
    setup_cactus_path()

    import agent
    import sandboxes
    from cactus import cactus_destroy, cactus_init

    sb = sandboxes.get(args.sandbox)
    query = args.query or sb.default_query

    print(f"sandbox={args.sandbox}  threshold={args.threshold}  cloud_key={'yes' if have_cloud_key() else 'NO'}")
    print(f"bundle={DEFAULT_BUNDLE}")
    print(f"query={query!r}\n")
    if not os.path.isdir(DEFAULT_BUNDLE):
        raise SystemExit(f"bundle not found: {DEFAULT_BUNDLE}")

    print("loading model… (gemma-4 E2B probe bundle)")
    model = cactus_init(DEFAULT_BUNDLE)
    try:
        def on_turn(phase: str, info: dict) -> None:
            if phase != "done":
                return
            loc = "CLOUD" if info.get("cloud_handoff") else "local"
            conf = info.get("confidence")
            conf_s = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
            tool = info.get("_tool_name") or "(final)"
            toks = f"pf={info.get('prefill_tokens')} dec={info.get('decode_tokens')}"
            ct = info.get("cloud_prompt_tokens")
            if ct is not None:
                toks += f" cloud_pf={ct} cloud_dec={info.get('cloud_completion_tokens')}"
            print(f"  turn {info['_index']:>2}  {loc:<5}  conf={conf_s}  thr={info.get('confidence_threshold')}  "
                  f"{info.get('total_time_ms', 0):>6.0f}ms  {toks}")
            print(f"         reason={info.get('cloud_handoff_reason')!r}  tool={tool}")
            resp = (info.get("response") or "").strip().replace("\n", " ")
            print(f"         say={resp[:110]!r}")

        turns = agent.run(model, sb, query, args.threshold, on_turn=on_turn,
                          max_tokens=args.max_tokens)

        n_cloud = sum(1 for t in turns if t.get("cloud_handoff"))
        confs = [t.get("confidence") for t in turns if isinstance(t.get("confidence"), (int, float))]
        print(f"\nsummary: {len(turns)} turns, {n_cloud} cloud, {len(turns)-n_cloud} local")
        if confs:
            print(f"confidence range: min={min(confs):.3f} max={max(confs):.3f} "
                  f"values={[round(c,3) for c in confs]}")
    finally:
        cactus_destroy(model)


if __name__ == "__main__":
    main()
