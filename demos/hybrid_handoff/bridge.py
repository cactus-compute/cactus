"""Resident NDJSON driver for the OpenTUI front end.

Loads the gemma-4 bundle once and stays alive, reading commands on stdin and
emitting events on stdout (one JSON object per line). The probe confidence and
cloud token usage we visualize are only exposed through the Cactus FFI, not the
OpenAI-compatible HTTP server, so the brain must live here in Python; the TUI is
a pure presentation client over this stream.

Commands (stdin, one JSON per line):
    {"type": "query", "text": "...", "threshold": 0.35}
    {"type": "quit"}

Events (stdout, one JSON per line):
    {"type": "loading"}
    {"type": "ready", "sandbox", "default_query", "tools", "inbox"}
    {"type": "turn_start", "index"}
    {"type": "turn", ...StepRecord fields..., "tokens_cloud", "tool_result_snippet"}
    {"type": "final", "text"}
    {"type": "run_complete", ...RunState totals...}
    {"type": "error", "message"}

Run standalone for a smoke test:
    echo '{"type":"query","text":"...","threshold":0.35}' | \
        ./venv/bin/python demos/hybrid_handoff/bridge.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import cactus_env


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _snippet(text: str, n: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def _inbox_preview(sandbox) -> list[dict]:
    rows = getattr(sandbox, "inbox", []) or []
    return [
        {"id": m["id"], "from": m["from"], "subject": m["subject"],
         "date": m["date"], "unread": bool(m.get("unread"))}
        for m in rows
    ]


def handle_query(model, sandbox, agent, records, text: str, threshold: float,
                 max_tokens: int, max_tool_steps: int) -> None:
    state = records.RunState(sandbox=sandbox.name, threshold=threshold, query=text)

    def on_turn(phase: str, info: dict) -> None:
        if phase == "inflight":
            emit({"type": "turn_start", "index": int(info.get("index", 0))})
            return
        rec = records.build_record(info)
        state.add(rec)
        payload = dataclasses.asdict(rec)
        payload["type"] = "turn"
        payload["tokens_cloud"] = rec.tokens_cloud
        result = info.get("_tool_result")
        if isinstance(result, dict):
            payload["tool_result_snippet"] = _snippet(json.dumps(result))
        emit(payload)

    turns = agent.run(model, sandbox, text, threshold, on_turn=on_turn,
                      max_tokens=max_tokens, max_tool_steps=max_tool_steps)
    final = next((t for t in reversed(turns) if (t.get("response") or "").strip()), None)
    emit({"type": "final", "text": final["response"].strip() if final else ""})
    emit({
        "type": "run_complete",
        "turns_local": state.turns_local,
        "turns_cloud": state.turns_cloud,
        "tokens_local_total": state.tokens_local_total,
        "tokens_cloud_total": state.tokens_cloud_total,
        "cost_usd_total": state.cost_usd_total,
        "latency_ms_total": state.latency_ms_total,
        "pct_local": state.pct_local,
    })


def ready_event(sandbox) -> dict:
    return {
        "type": "ready",
        "sandbox": sandbox.name,
        "default_query": sandbox.default_query,
        "tools": [t.name for t in sandbox.tools],
        "inbox": _inbox_preview(sandbox),
    }


def serve(sandbox_name: str, max_tokens: int, max_tool_steps: int) -> None:
    cactus_env.load_env()
    cactus_env.setup_cactus_path()

    import agent
    import records
    import sandboxes
    from cactus import cactus_destroy, cactus_init

    sandbox = sandboxes.get(sandbox_name)
    emit({"type": "loading"})

    with cactus_env.quiet_engine_logs():
        model = cactus_init(cactus_env.DEFAULT_BUNDLE)
        emit(ready_event(sandbox))
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as exc:
                    emit({"type": "error", "message": f"bad command: {exc}"})
                    continue
                kind = cmd.get("type")
                if kind == "quit":
                    break
                if kind == "sandbox":
                    name = (cmd.get("name") or "").strip()
                    try:
                        sandbox = sandboxes.get(name)
                    except KeyError as exc:
                        emit({"type": "error", "message": str(exc)})
                        continue
                    emit(ready_event(sandbox))
                    continue
                if kind != "query":
                    emit({"type": "error", "message": f"unknown command '{kind}'"})
                    continue
                text = (cmd.get("text") or "").strip() or sandbox.default_query
                threshold = float(cmd.get("threshold", 0.35))
                try:
                    handle_query(model, sandbox, agent, records, text, threshold,
                                 max_tokens, max_tool_steps)
                except Exception as exc:
                    emit({"type": "error", "message": f"run failed: {exc}"})
        finally:
            cactus_destroy(model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resident NDJSON bridge for the OpenTUI demo")
    ap.add_argument("--sandbox", default="email")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-tool-steps", type=int, default=5)
    args = ap.parse_args()
    serve(args.sandbox, args.max_tokens, args.max_tool_steps)


if __name__ == "__main__":
    main()
