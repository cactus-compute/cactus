"""Cactus hybrid-handoff live demo.

Runs a multi-turn agent inside a mock sandbox and visualizes, per turn, whether
the Cactus engine answered ON-DEVICE or escalated to the CLOUD (probe-based).
The --threshold flag IS the mode:  0.0 = pure on-device, ~0.35 = probe hybrid,
1.0 = forced cloud. Run three terminals side-by-side for a head-to-head.

    ./venv/bin/python demos/hybrid_handoff/run.py --sandbox email --threshold 0.35 \
        --query "Reply to Jake's email about our meeting; check my calendar and pull context."
"""
from __future__ import annotations

import argparse
import sys

from cactus_env import (
    DEFAULT_BUNDLE,
    have_cloud_key,
    load_env,
    quiet_engine_logs,
    setup_cactus_path,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cactus hybrid-handoff live demo")
    ap.add_argument("--sandbox", default="email", help="mock environment to run in")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="handoff threshold: 0=on-device, ~0.35=hybrid, 1=cloud")
    ap.add_argument("--query", default=None, help="open-ended task (defaults to the sandbox's example)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-tool-steps", type=int, default=5)
    ap.add_argument("--bundle", default=None,
                    help="runnable bundle dir to load (overrides CACTUS_DEMO_BUNDLE / weights/gemma-4)")
    ap.add_argument("--list-sandboxes", action="store_true")
    args = ap.parse_args()

    bundle = args.bundle or DEFAULT_BUNDLE

    load_env()
    setup_cactus_path()

    import agent
    import records
    import sandboxes
    import tui
    from cactus import cactus_destroy, cactus_init

    if args.list_sandboxes:
        print("sandboxes:", ", ".join(sandboxes.available()))
        return

    try:
        sb = sandboxes.get(args.sandbox)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    if args.query is not None:
        query = args.query
    else:
        from rich.prompt import Prompt
        query = Prompt.ask("Query", default=sb.default_query)

    if args.threshold > 0.0 and not have_cloud_key():
        print("WARNING: no CACTUS_CLOUD_KEY set -- cloud turns will fail and fall back to "
              "local. Add it to demos/hybrid_handoff/.env.", file=sys.stderr)

    print(f"loading {bundle} …", file=sys.stderr)
    state = records.RunState(sandbox=args.sandbox, threshold=args.threshold, query=query)
    with quiet_engine_logs():
        model = cactus_init(bundle)
        try:
            with tui.HandoffTUI(state) as ui:
                turns = agent.run(model, sb, query, args.threshold, on_turn=ui.on_turn,
                                  max_tokens=args.max_tokens, max_tool_steps=args.max_tool_steps)
        finally:
            cactus_destroy(model)

    final = next((t for t in reversed(turns) if (t.get("response") or "").strip()), None)
    if final:
        print("\n" + "─" * 70)
        print("FINAL ANSWER:\n")
        print(final["response"].strip())


if __name__ == "__main__":
    main()
