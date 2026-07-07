"""Optional bridge: run a demo sandbox through benchmarks/handoff_multistep.

The benchmark harness (`harness.run_mode_b`) feeds tool results from a STATIC
`task["tool_stubs"]` dict (one fixed result per tool name). These sandboxes are
dynamic (tools query a data pool by args). `sandbox_as_task` wraps a sandbox as a
benchmark-style task that additionally carries a `dispatch` callable.

To actually use it, apply this one-line change in `harness.run_mode_b` where it
resolves a tool result:

    # before:
    result = task.get("tool_stubs", {}).get(c["name"], {"status": "ok"})
    # after (prefer dynamic dispatch when present):
    if callable(task.get("dispatch")):
        result = task["dispatch"](c["name"], c.get("args") or {})
    else:
        result = task.get("tool_stubs", {}).get(c["name"], {"status": "ok"})

Then: TASKS.append(sandbox_as_task(get("email"))) and run as usual.
"""
from __future__ import annotations

from typing import Any

from sandboxes import Sandbox, get


def sandbox_as_task(sandbox: Sandbox, query: str | None = None) -> dict[str, Any]:
    return {
        "id": f"sandbox_{sandbox.name}",
        "system_prompt": sandbox.system_prompt,
        "user_goal": query or sandbox.default_query,
        "tools": sandbox.tool_schemas(),
        "tool_stubs": {},
        "dispatch": sandbox.dispatch,
        "steps": [],
    }


def all_sandbox_tasks() -> list[dict[str, Any]]:
    from sandboxes import available
    return [sandbox_as_task(get(name)) for name in available()]
