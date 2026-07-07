"""Per-turn record + run aggregation derived from the native completion dict."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cost


def _snippet(text: str, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


@dataclass
class StepRecord:
    index: int
    status: str = "done"
    kind: str = "REASON"
    location: str = "local"
    confidence: float | None = None
    advantage: float | None = None
    threshold: float = 0.0
    reason: str = ""
    latency_ms: float = 0.0
    tokens_local: int = 0
    tokens_cloud_prompt: int = 0
    tokens_cloud_out: int = 0
    cost_usd: float = 0.0
    content_snippet: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    error: str | None = None

    @property
    def tokens_cloud(self) -> int:
        return self.tokens_cloud_prompt + self.tokens_cloud_out


def inflight(index: int, threshold: float) -> StepRecord:
    return StepRecord(index=index, status="inflight", threshold=threshold, kind="…")


def build_record(resp: dict[str, Any]) -> StepRecord:
    cloud = bool(resp.get("cloud_handoff"))
    conf = resp.get("confidence")
    adv = resp.get("cloud_advantage")
    kind = resp.get("_kind") or (resp.get("_tool_name") or "FINAL")
    rec = StepRecord(
        index=int(resp.get("_index", 0)),
        status="done",
        kind=kind,
        location="cloud" if cloud else "local",
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        advantage=float(adv) if isinstance(adv, (int, float)) else None,
        threshold=float(resp.get("confidence_threshold") or 0.0),
        reason=resp.get("cloud_handoff_reason") or "",
        latency_ms=float(resp.get("total_time_ms") or 0.0),
        tool_name=resp.get("_tool_name"),
        tool_args=resp.get("_tool_args"),
        error=resp.get("error") or None,
        content_snippet=_snippet(resp.get("response") or resp.get("thinking") or ""),
    )
    if cloud:
        pt, ct = cost.resolve_cloud_tokens(resp)
        rec.tokens_cloud_prompt, rec.tokens_cloud_out = pt, ct
        rec.cost_usd = cost.cost_usd(pt, ct)
    else:
        rec.tokens_local = int(resp.get("total_tokens") or 0)
    return rec


@dataclass
class RunState:
    sandbox: str
    threshold: float
    query: str
    steps: list[StepRecord] = field(default_factory=list)
    turns_local: int = 0
    turns_cloud: int = 0
    tokens_local_total: int = 0
    tokens_cloud_total: int = 0
    cost_usd_total: float = 0.0
    latency_ms_total: float = 0.0

    def add(self, rec: StepRecord) -> None:
        self.steps.append(rec)
        if rec.location == "cloud":
            self.turns_cloud += 1
            self.tokens_cloud_total += rec.tokens_cloud
        else:
            self.turns_local += 1
            self.tokens_local_total += rec.tokens_local
        self.cost_usd_total += rec.cost_usd
        self.latency_ms_total += rec.latency_ms

    @property
    def turns_total(self) -> int:
        return self.turns_local + self.turns_cloud

    @property
    def pct_local(self) -> float:
        return 100.0 * self.turns_local / self.turns_total if self.turns_total else 0.0
