"""Incident sandbox: triage an alert from logs + runbook.

Tools query alerts, logs, and runbooks; the severity/action judgment (correlate
log signals against runbook thresholds) is the hard turn. Data in ./data/."""
from __future__ import annotations

import json
from typing import Any

from ..base import Sandbox, ToolSpec

SYSTEM_PROMPT = (
    "You are an on-device incident assistant. You can list/inspect alerts, query "
    "service logs, and read runbooks. Investigate the alert, correlate the log "
    "signals against the runbook thresholds, then state the severity (SEV1/2/3) and "
    "a concrete recommended action. One tool per step; don't repeat calls."
)


class IncidentSandbox(Sandbox):
    name = "incident"
    system_prompt = SYSTEM_PROMPT
    default_query = (
        "An alert fired on the checkout service. Pull the logs and the runbook, decide "
        "the severity, and recommend a concrete next action."
    )

    def __init__(self) -> None:
        super().__init__()
        self.alerts: list[dict[str, Any]] = self.load_json("alerts.json")
        self.runbooks: list[dict[str, Any]] = self.load_json("runbooks.json")
        self.logs: list[dict[str, Any]] = self._load_logs("logs.jsonl")

        self.register(ToolSpec(
            name="list_alerts",
            description="List alerts, optionally only active ones.",
            parameters={"type": "object", "properties": {
                "active_only": {"type": "boolean"}}, "required": []},
            fn=self.list_alerts,
        ))
        self.register(ToolSpec(
            name="fetch_alert",
            description="Get one alert by id.",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            fn=self.fetch_alert,
        ))
        self.register(ToolSpec(
            name="query_logs",
            description="Query log lines for a service, optionally filtered by level (INFO/WARN/ERROR).",
            parameters={"type": "object", "properties": {
                "service": {"type": "string"},
                "level": {"type": "string"},
                "limit": {"type": "integer"}},
                "required": ["service"]},
            fn=self.query_logs,
        ))
        self.register(ToolSpec(
            name="fetch_runbook",
            description="Get the runbook (thresholds + procedure) for a service.",
            parameters={"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            fn=self.fetch_runbook,
        ))

    def _load_logs(self, filename: str) -> list[dict[str, Any]]:
        out = []
        for line in self.load_text(filename).splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def list_alerts(self, active_only: bool = False) -> dict[str, Any]:
        al = [a for a in self.alerts if a["active"]] if active_only else self.alerts
        return {"count": len(al), "alerts": al}

    def fetch_alert(self, id: str) -> dict[str, Any]:
        for a in self.alerts:
            if a["id"] == id:
                return dict(a)
        return {"error": f"no alert '{id}'"}

    def query_logs(self, service: str, level: str = "", limit: int = 20) -> dict[str, Any]:
        svc = (service or "").lower()
        lvl = (level or "").upper()
        hits = [l for l in self.logs if l["service"].lower() == svc and (not lvl or l["level"] == lvl)]
        hits = hits[: max(1, int(limit or 20))]
        n_err = sum(1 for l in self.logs if l["service"].lower() == svc and l["level"] == "ERROR")
        return {"service": service, "count": len(hits), "error_count": n_err, "logs": hits}

    def fetch_runbook(self, service: str) -> dict[str, Any]:
        for r in self.runbooks:
            if r["service"].lower() == (service or "").lower():
                return dict(r)
        return {"error": f"no runbook for '{service}'"}
