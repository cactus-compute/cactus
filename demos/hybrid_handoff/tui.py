"""Live Rich TUI for the hybrid-handoff demo.

Renders a per-turn timeline: each agent turn is a card that lands as ON-DEVICE
(green) or CLOUD (blue), with a confidence-vs-threshold gauge (green = kept
local, red = escalated) and a cumulative HUD. Rendering is per-turn (the probe
path defers token streaming), driven by the agent loop's on_turn callback.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import records
from records import RunState, StepRecord

LOCAL_STYLE = "green"
CLOUD_STYLE = "blue"

_MODE = {
    0.0: ("PURE ON-DEVICE", "green"),
    1.0: ("FORCED CLOUD", "blue"),
}


def mode_label(threshold: float) -> tuple[str, str]:
    if threshold <= 0.0:
        return _MODE[0.0]
    if threshold >= 1.0:
        return _MODE[1.0]
    return (f"PROBE HYBRID (thr {threshold:g})", "cyan")


def render_gauge(confidence: float | None, threshold: float, width: int = 18) -> Text:
    """Bar filled to `confidence`, with a marker at `threshold`. Green when
    confidence >= threshold (kept local), red when below (escalated)."""
    if confidence is None:
        txt = "forced cloud" if threshold >= 1.0 else "…"
        return Text(f"[{txt}]", style="dim")
    filled = int(round(max(0.0, min(1.0, confidence)) * width))
    marker = int(round(max(0.0, min(1.0, threshold)) * width))
    below = confidence < threshold
    bar = Text()
    for i in range(width + 1):
        if i == marker:
            bar.append("┃", style="bold yellow")
        elif i < filled:
            bar.append("█", style="bold red" if below else "bold green")
        else:
            bar.append("─", style="grey37")
    bar.append(f"  conf {confidence:.2f}", style="bold red" if below else "bold green")
    bar.append(f"  thr {threshold:g}", style="yellow")
    return bar


def _badge(rec: StepRecord) -> Text:
    if rec.location == "cloud":
        return Text("☁ CLOUD", style=f"bold {CLOUD_STYLE}")
    return Text("◉ ON-DEVICE", style=f"bold {LOCAL_STYLE}")


def _step_panel(rec: StepRecord) -> Panel:
    escalated = rec.location == "cloud"
    head = Table.grid(expand=True)
    head.add_column(ratio=1)
    head.add_column(justify="right")
    left = Text()
    left.append(f"#{rec.index}  ", style="bold white")
    left.append(f"{rec.kind:<14}", style="bold")
    left.append_text(_badge(rec))
    meta = Text()
    meta.append(f"{rec.latency_ms:>6.0f}ms", style="dim")
    if escalated:
        meta.append(f"  ↑{rec.tokens_cloud} tok", style=CLOUD_STYLE)
        meta.append(f"  ${rec.cost_usd:.4f}", style="yellow")
    else:
        meta.append(f"  {rec.tokens_local} tok", style="dim green")
        meta.append("  $0.0000", style="dim")
    head.add_row(left, meta)

    body = Group(head, render_gauge(rec.confidence, rec.threshold))
    rows = [body]
    if rec.tool_name:
        args = ", ".join(f"{k}={v!r}" for k, v in (rec.tool_args or {}).items())
        rows.append(Text(f"  → {rec.tool_name}({args})", style="dim cyan"))
    elif rec.content_snippet:
        rows.append(Text(f"  “{rec.content_snippet}”", style="italic dim"))
    if rec.reason and "fail" in rec.reason.lower():
        rows.append(Text(f"  ⚠ {rec.reason}", style="yellow"))

    border = CLOUD_STYLE if escalated else LOCAL_STYLE
    title = "☁ ESCALATED TO CLOUD" if (escalated and "probe" in rec.reason) else None
    return Panel(Group(*rows), border_style=border, title=title, title_align="left", padding=(0, 1))


def _header(state: RunState) -> Panel:
    label, color = mode_label(state.threshold)
    t = Table.grid(expand=True)
    t.add_column(ratio=1)
    t.add_column(justify="right")
    left = Text()
    left.append("Cactus Hybrid Handoff", style="bold white")
    left.append(f"   sandbox={state.sandbox}", style="dim")
    t.add_row(left, Text(label, style=f"bold {color}"))
    t.add_row(Text(f"query: {records._snippet(state.query, 90)}", style="italic dim"), Text(""))
    return Panel(t, border_style=color, padding=(0, 1))


def _hud(state: RunState) -> Panel:
    t = Table.grid(expand=True, padding=(0, 2))
    for _ in range(4):
        t.add_column()
    t.add_row(
        Text.assemble(("turns  ", "dim"), (f"{state.turns_local}", f"bold {LOCAL_STYLE}"),
                      (" local / ", "dim"), (f"{state.turns_cloud}", f"bold {CLOUD_STYLE}"), (" cloud", "dim")),
        Text.assemble(("on-device  ", "dim"), (f"{state.pct_local:.0f}%", f"bold {LOCAL_STYLE}")),
        Text.assemble(("tokens  ", "dim"), (f"{state.tokens_local_total}", LOCAL_STYLE),
                      (" local / ", "dim"), (f"{state.tokens_cloud_total}", CLOUD_STYLE), (" cloud", "dim")),
        Text.assemble(("cost  ", "dim"), (f"${state.cost_usd_total:.4f}", "bold yellow")),
    )
    t.add_row(
        Text.assemble(("latency  ", "dim"), (f"{state.latency_ms_total/1000:.1f}s", "white")),
        Text(""), Text(""), Text(""),
    )
    return Panel(t, border_style="grey50", title="totals", title_align="left", padding=(0, 1))


class HandoffTUI:
    """Sequential, artifact-free renderer: header once, each step card as it
    completes (cards stream in real time as the agent works), then the totals HUD
    at the end. No Rich Live region -- that was duplicating the header/HUD into
    scrollback once output exceeded the viewport -- and no in-flight 'working' lines."""

    def __init__(self, state: RunState) -> None:
        self.state = state
        self.console = Console()

    def on_turn(self, phase: str, info: dict[str, Any]) -> None:
        if phase != "done":
            return
        rec = records.build_record(info)
        self.state.add(rec)
        self.console.print(_step_panel(rec))

    def __enter__(self) -> "HandoffTUI":
        self.console.print(_header(self.state))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.console.print(_hud(self.state))
