"""Multi-turn agent loop driving a Sandbox via the Cactus FFI.

One cactus_complete() call == one agent turn == one independent local/cloud
handoff decision. Tool calls are dispatched to the sandbox (which returns REAL
queried data), fed back, and the loop continues until the model stops calling
tools (the FINAL answer) or max_steps is hit.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from sandboxes import Sandbox


def build_options(threshold: float, max_tokens: int = 1024,
                  cloud_timeout_ms: int = 20000) -> dict[str, Any]:
    """Map --threshold to engine InferenceOptions.

    0.0  -> probe runs but `confidence < 0` never fires -> pure on-device
    0.5+ -> probe hybrid (hands off when confidence < threshold)
    >=1.0 -> forced pre-generation cloud every turn

    auto_handoff stays True even at 0.0 so the probe still reports confidence
    (keeps the decision gauge populated in pure-local mode). max_tokens MUST be
    set: the engine default is only 100 and would truncate replies.
    """
    return {
        "temperature": 0,
        "max_tokens": max_tokens,
        "auto_handoff": True,
        "confidence_threshold": float(threshold),
        "cloud_timeout_ms": cloud_timeout_ms,
        "tool_rag_top_k": 0,
    }


def _coerce_args(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


_TEXT_CALL_RE = re.compile(r"^(\w+)\s*\(\s*(\{.*\})\s*\)\s*$", re.S)


def parse_text_tool_call(text: str, tool_names: set[str] | None) -> dict[str, Any] | None:
    """Recognize a tool call emitted as plain TEXT instead of a structured
    function_call (cloud/local models sometimes do this, e.g. `read_email({"id":
    "m011"})`). Handles  name({json}) ,  {"name":..,"arguments":..} ,  [ {..} ] ,
    and <tool_call>{..}</tool_call> . Gated by tool_names to avoid matching prose."""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"<tool_call>\s*(.+?)\s*</tool_call>", text, re.S)
    if m:
        text = m.group(1).strip()
    m = _TEXT_CALL_RE.match(text)
    if m and (tool_names is None or m.group(1) in tool_names):
        return {"name": m.group(1), "args": _coerce_args(m.group(2))}
    if text[:1] in "{[":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, list):
            obj = obj[0] if obj and isinstance(obj[0], dict) else {}
        if isinstance(obj, dict):
            name = obj.get("name") or (obj.get("function") or {}).get("name")
            args = obj.get("arguments")
            if args is None:
                args = obj.get("args") or obj.get("parameters") or (obj.get("function") or {}).get("arguments")
            if name and (tool_names is None or name in tool_names):
                return {"name": name, "args": _coerce_args(args)}
    return None


def extract_calls(resp: dict[str, Any], tool_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Normalize tool calls to [{name, args(dict)}]. Prefers structured
    function_calls; falls back to a tool call written as plain text in response."""
    out: list[dict[str, Any]] = []
    for call in resp.get("function_calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or (call.get("function") or {}).get("name")
        args = call.get("arguments")
        if args is None:
            args = (call.get("function") or {}).get("arguments")
        if name:
            out.append({"name": name, "args": _coerce_args(args)})
    if out:
        return out
    parsed = parse_text_tool_call(resp.get("response") or "", tool_names)
    return [parsed] if parsed else []


def complete_with_retry(model, messages, options, tools, retries: int = 3) -> dict[str, Any]:
    """cactus_complete with exponential backoff on transient failure."""
    from cactus import cactus_complete

    last_err = ""
    for attempt in range(retries):
        try:
            resp = cactus_complete(model, messages, options, tools, None)
        except Exception as exc:
            last_err = str(exc)
            resp = None
        if isinstance(resp, dict) and resp.get("success", True) and resp.get("error") in (None, ""):
            return resp
        last_err = (resp or {}).get("error", last_err) if isinstance(resp, dict) else last_err
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return {"success": False, "error": last_err or "completion failed", "response": "",
            "function_calls": [], "cloud_handoff": False}


FINALIZE_DIRECTIVE = (
    "You now have enough information from the tools. Do NOT call any more tools. "
    "Write the complete final answer for the user now."
)


def run(model, sandbox: Sandbox, query: str, threshold: float,
        on_turn: Callable[[str, dict[str, Any]], None] | None = None,
        max_tokens: int = 1024, max_tool_steps: int = 5) -> list[dict[str, Any]]:
    """Free-running agent loop. Returns the list of per-turn native result dicts
    (each annotated with _tool_name/_tool_args/_tool_result/_kind for rendering).

    The loop gathers via tools, then ALWAYS ends with a tool-free FINAL synthesis
    turn -- triggered when the model stops calling tools, repeats a call, or hits
    max_tool_steps. Small local models tend to loop on tool calls, so this forced
    finalize guarantees the synthesis turn (the hero handoff candidate).

    on_turn(phase, info): phase is "inflight" (about to call the model) or "done".
    """
    from cactus import cactus_reset

    cactus_reset(model)
    opts = build_options(threshold, max_tokens=max_tokens)
    tools = sandbox.tool_schemas()
    tool_names = {t.name for t in sandbox.tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sandbox.system_prompt},
        {"role": "user", "content": query},
    ]
    turns: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    index = 0

    while True:
        finalize = index >= max_tool_steps
        active_tools = None if finalize else tools
        if on_turn:
            on_turn("inflight", {"index": index})
        resp = complete_with_retry(model, messages, opts, active_tools)
        resp["_index"] = index
        calls = [] if finalize else extract_calls(resp, tool_names)

        repeated = False
        if calls:
            sig = f"{calls[0]['name']}({json.dumps(calls[0]['args'], sort_keys=True)})"
            repeated = sig in seen_calls
            seen_calls.add(sig)

        if calls and not repeated:
            c = calls[0]
            result = sandbox.dispatch(c["name"], c["args"])
            resp.update(_tool_name=c["name"], _tool_args=c["args"], _tool_result=result, _kind=c["name"])
            messages.append({
                "role": "assistant",
                "content": resp.get("response", ""),
                "tool_calls": [{"type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}],
            })
            messages.append({"role": "tool", "name": c["name"], "content": json.dumps(result)})
            turns.append(resp)
            if on_turn:
                on_turn("done", resp)
            index += 1
            continue

        final_text = (resp.get("response") or "").strip()
        if (repeated or finalize) and (not final_text or parse_text_tool_call(final_text, tool_names)):
            messages.append({"role": "user", "content": FINALIZE_DIRECTIVE})
            resp = complete_with_retry(model, messages, opts, None)
            resp["_index"] = index
        resp["_kind"] = "FINAL"
        turns.append(resp)
        if on_turn:
            on_turn("done", resp)
        break

    return turns
