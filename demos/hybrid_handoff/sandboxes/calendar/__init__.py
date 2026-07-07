"""Calendar sandbox: schedule a meeting under real availability constraints.

Tools query per-person busy blocks; the model must reason about a conflict-free
slot (the hard turn) and can book it. Data in ./data/*.json."""
from __future__ import annotations

from typing import Any

from ..base import Sandbox, ToolSpec

SYSTEM_PROMPT = (
    "You are an on-device scheduling assistant. You can list people, read each "
    "person's calendar for a date, and create an event. Gather everyone's busy "
    "blocks, then reason about a time when ALL required attendees are free (assume "
    "working hours 09:00-17:00). Call at most one tool per step and don't repeat "
    "calls. When you've found a valid slot, state it clearly (and create the event "
    "if asked)."
)


class CalendarSandbox(Sandbox):
    name = "calendar"
    system_prompt = SYSTEM_PROMPT
    default_query = (
        "Schedule a 1-hour design sync on 2026-06-25 with the whole design team, "
        "at a time when everyone is free. Create the event once you find a slot."
    )

    def __init__(self) -> None:
        super().__init__()
        self.people: list[dict[str, Any]] = self.load_json("people.json")
        self.events: list[dict[str, Any]] = self.load_json("calendars.json")
        self.created: list[dict[str, Any]] = []

        self.register(ToolSpec(
            name="list_people",
            description="List people, optionally filtered by team (e.g. 'design').",
            parameters={"type": "object", "properties": {
                "team": {"type": "string", "description": "Optional team filter."}}, "required": []},
            fn=self.list_people,
        ))
        self.register(ToolSpec(
            name="get_calendar",
            description="Get a person's busy blocks for a date (YYYY-MM-DD).",
            parameters={"type": "object", "properties": {
                "person": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["person", "date"]},
            fn=self.get_calendar,
        ))
        self.register(ToolSpec(
            name="create_event",
            description="Create a calendar event once a free slot is found.",
            parameters={"type": "object", "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "start": {"type": "string", "description": "HH:MM"},
                "end": {"type": "string", "description": "HH:MM"},
                "attendees": {"type": "array", "items": {"type": "string"}}},
                "required": ["title", "date", "start", "end"]},
            fn=self.create_event,
        ))

    def list_people(self, team: str = "") -> dict[str, Any]:
        ppl = self.people if not team else [p for p in self.people if p["team"].lower() == team.lower()]
        return {"count": len(ppl), "people": ppl}

    def get_calendar(self, person: str, date: str) -> dict[str, Any]:
        busy = [e for e in (self.events + self.created)
                if e["person"].lower() == person.lower() and e["date"] == date]
        busy.sort(key=lambda e: e["start"])
        return {"person": person, "date": date, "busy": [
            {"start": e["start"], "end": e["end"], "title": e["title"]} for e in busy]}

    def create_event(self, title: str, date: str, start: str, end: str,
                     attendees: list[str] | None = None) -> dict[str, Any]:
        attendees = attendees or []
        for a in attendees:
            self.created.append({"person": a, "date": date, "start": start, "end": end, "title": title})
        return {"status": "created", "title": title, "date": date, "start": start, "end": end,
                "attendees": attendees}
