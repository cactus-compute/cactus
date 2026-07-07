"""Email sandbox: a queryable mock inbox + calendar + docs.

Tools actually search/filter the data pool by the model's arguments, so any
open-ended --query works (e.g. "reply to Jake", "summarize unread offsite mail",
"who emailed about the Q3 budget"). Data lives in ./data/*.json -- edit those to
add more examples.
"""
from __future__ import annotations

from typing import Any

from ..base import Sandbox, ToolSpec

SYSTEM_PROMPT = (
    "You are an on-device email assistant with tools to search and read the user's "
    "inbox, check their calendar, and look up reference docs.\n"
    "Work in a single efficient pass: each step, call AT MOST ONE tool, and never "
    "repeat a search you have already done. Once you have searched the inbox, read "
    "the relevant email, checked the calendar, and pulled any needed doc, STOP "
    "calling tools and write the complete final answer. When asked to draft a reply, "
    "output the full email text (greeting, body, sign-off)."
)


def _snippet(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


class EmailSandbox(Sandbox):
    name = "email"
    system_prompt = SYSTEM_PROMPT
    default_query = (
        "Reply to Jake's latest email about our meeting. Check my calendar for any "
        "conflict with the time he proposed, and pull any context doc he asked for."
    )

    def __init__(self) -> None:
        super().__init__()
        self.inbox: list[dict[str, Any]] = self.load_json("inbox.json")
        self.calendar: list[dict[str, Any]] = self.load_json("calendar.json")
        self.docs: list[dict[str, Any]] = self.load_json("docs.json")

        self.register(ToolSpec(
            name="search_inbox",
            description="Search the inbox. Filter by free-text query (matches subject/body/sender), "
                        "sender substring, and/or unread status. Returns matching email headers.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text keywords to match in subject, body, or sender."},
                    "sender": {"type": "string", "description": "Substring of the sender name/address."},
                    "unread_only": {"type": "boolean", "description": "If true, only unread emails."},
                    "limit": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": [],
            },
            fn=self.search_inbox,
        ))
        self.register(ToolSpec(
            name="read_email",
            description="Read the full body of one email by its id (from search_inbox results).",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The email id, e.g. 'm001'."}},
                "required": ["id"],
            },
            fn=self.read_email,
        ))
        self.register(ToolSpec(
            name="get_calendar",
            description="List the user's calendar events, optionally filtered to a single date (YYYY-MM-DD).",
            parameters={
                "type": "object",
                "properties": {"date": {"type": "string", "description": "Optional date YYYY-MM-DD to filter to."}},
                "required": [],
            },
            fn=self.get_calendar,
        ))
        self.register(ToolSpec(
            name="search_docs",
            description="Search reference/context docs by keyword (matches title, tags, body). "
                        "Returns matching docs with their full body.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords to match."}},
                "required": ["query"],
            },
            fn=self.search_docs,
        ))

    def search_inbox(self, query: str = "", sender: str = "", unread_only: bool = False,
                     limit: int = 10) -> dict[str, Any]:
        q = (query or "").lower()
        snd = (sender or "").lower()
        hits = []
        for m in self.inbox:
            if unread_only and not m.get("unread"):
                continue
            if snd and snd not in m["from"].lower():
                continue
            if q:
                hay = f"{m['subject']} {m['body']} {m['from']}".lower()
                if not all(term in hay for term in q.split()):
                    continue
            hits.append(m)
        hits.sort(key=lambda m: m["date"], reverse=True)
        hits = hits[: max(1, int(limit or 10))]
        return {
            "count": len(hits),
            "emails": [
                {"id": m["id"], "from": m["from"], "subject": m["subject"],
                 "date": m["date"], "unread": m["unread"], "snippet": _snippet(m["body"])}
                for m in hits
            ],
        }

    def read_email(self, id: str) -> dict[str, Any]:
        for m in self.inbox:
            if m["id"] == id:
                return {k: m[k] for k in ("id", "from", "to", "date", "subject", "body", "unread", "thread_id")}
        return {"error": f"no email with id '{id}'"}

    def get_calendar(self, date: str = "") -> dict[str, Any]:
        events = self.calendar if not date else [e for e in self.calendar if e["date"] == date]
        events = sorted(events, key=lambda e: (e["date"], e["start"]))
        return {"count": len(events), "events": events}

    def search_docs(self, query: str = "") -> dict[str, Any]:
        q = (query or "").lower()
        hits = []
        for d in self.docs:
            hay = f"{d['title']} {' '.join(d.get('tags', []))} {d['body']}".lower()
            if not q or any(term in hay for term in q.split()):
                hits.append(d)
        return {"count": len(hits), "docs": hits}
