"""Expense sandbox: decide if a receipt is policy-compliant and submit it.

Tools query the receipt pool and the policy doc; the compliance judgment (per-
person cap math, approval thresholds) is the hard turn. Data in ./data/."""
from __future__ import annotations

import re
from typing import Any

from ..base import Sandbox, ToolSpec

SYSTEM_PROMPT = (
    "You are an on-device expense assistant. You can search/read receipts, look up "
    "the expense policy by category, and submit an expense. Determine whether a "
    "receipt complies with policy (mind per-person caps, approval thresholds, and "
    "required notes), then submit it if allowed or explain what's needed. One tool "
    "per step; don't repeat calls."
)


class ExpenseSandbox(Sandbox):
    name = "expense"
    system_prompt = SYSTEM_PROMPT
    default_query = (
        "Can I expense the team dinner receipt? Check it against policy and submit it "
        "if it's compliant; otherwise tell me what's needed."
    )

    def __init__(self) -> None:
        super().__init__()
        self.receipts: list[dict[str, Any]] = self.load_json("receipts.json")
        self.policy_text: str = self.load_text("policy.md")
        self.policy = self._parse_policy(self.policy_text)
        self.submitted: list[str] = []

        self.register(ToolSpec(
            name="search_receipts",
            description="Search receipts by free-text query (vendor/description) and/or category.",
            parameters={"type": "object", "properties": {
                "query": {"type": "string"},
                "category": {"type": "string", "description": "meals|travel|lodging|equipment"}},
                "required": []},
            fn=self.search_receipts,
        ))
        self.register(ToolSpec(
            name="read_receipt",
            description="Read a full receipt by id.",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            fn=self.read_receipt,
        ))
        self.register(ToolSpec(
            name="check_policy",
            description="Get the expense policy text for a category (or 'general').",
            parameters={"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
            fn=self.check_policy,
        ))
        self.register(ToolSpec(
            name="submit_expense",
            description="Submit a receipt for reimbursement by id.",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            fn=self.submit_expense,
        ))

    @staticmethod
    def _parse_policy(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        cur = None
        buf: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^##\s+(.+)$", line)
            if m:
                if cur:
                    out[cur] = "\n".join(buf).strip()
                cur, buf = m.group(1).strip().lower(), []
            elif cur:
                buf.append(line)
        if cur:
            out[cur] = "\n".join(buf).strip()
        return out

    def search_receipts(self, query: str = "", category: str = "") -> dict[str, Any]:
        q = (query or "").lower()
        cat = (category or "").lower()
        hits = []
        for r in self.receipts:
            if cat and r["category"] != cat:
                continue
            if q and not all(t in f"{r['vendor']} {r['description']}".lower() for t in q.split()):
                continue
            hits.append({"id": r["id"], "vendor": r["vendor"], "amount": r["amount"],
                         "category": r["category"], "date": r["date"]})
        return {"count": len(hits), "receipts": hits}

    def read_receipt(self, id: str) -> dict[str, Any]:
        for r in self.receipts:
            if r["id"] == id:
                return dict(r)
        return {"error": f"no receipt '{id}'"}

    def check_policy(self, category: str) -> dict[str, Any]:
        cat = (category or "").lower()
        if cat in self.policy:
            return {"category": cat, "policy": self.policy[cat], "general": self.policy.get("general", "")}
        return {"error": f"no policy for '{category}'", "categories": list(self.policy)}

    def submit_expense(self, id: str) -> dict[str, Any]:
        if not any(r["id"] == id for r in self.receipts):
            return {"error": f"no receipt '{id}'"}
        self.submitted.append(id)
        return {"status": "submitted", "id": id}
