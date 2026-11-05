"""Personal organizer actions for a phone: timers, reminders, events, tasks, notes.

Date and time phrases are passed verbatim for the host app to resolve; the
model never rewrites them. Reminders require a time on purpose - an undated
"remind me" is treated as incomplete rather than guessed at.
"""

import sys
from typing import Annotated, Literal, Optional

import cactus
from cactus.environments import _harness

Phrase = Annotated[str, cactus.Field(min_length=1, max_length=60)]


@cactus.tool
def set_timer(time_human: Phrase):
    """Set a timer for the specified duration or end time.

    Args:
        time_human: The duration or target end time in human readable format e.g. '45 minutes', 'at 13:30'.
    """
    return {"ok": True, "time_human": time_human}


@cactus.tool
def create_reminder(message: Annotated[str, cactus.Field(min_length=1, max_length=120)], date_time_human: Phrase):
    """Create a reminder that fires at a stated time. A reminder needs both a message and a time phrase; use add_task for undated to-dos.

    Args:
        message: What to be reminded about, copied word for word.
        date_time_human: The user's date or time phrase copied word for word.
    """
    return {"ok": True, "message": message, "date_time_human": date_time_human}


@cactus.tool
def create_calendar_event(
    title: Annotated[str, cactus.Field(min_length=1, max_length=120)],
    start_time_human: Phrase,
    location: Optional[Annotated[str, cactus.Field(min_length=1, max_length=80)]] = None,
):
    """Create a calendar event with a title and start time. Copy both word for word; never rephrase or resolve them.

    Args:
        title: The event title copied word for word.
        start_time_human: The start date or time phrase copied word for word.
        location: The event location; include only when stated.
    """
    return {"ok": True, "title": title, "start_time_human": start_time_human, "location": location}


@cactus.tool
def add_task(
    title: Annotated[str, cactus.Field(min_length=1, max_length=120)],
    priority: Optional[Literal["low", "medium", "high"]] = None,
):
    """Add a task to the task manager. Tasks are undated; dated or timed requests belong to create_reminder or create_calendar_event.

    Args:
        title: The task copied word for word.
        priority: low, medium, or high; include only when the user says a priority word.
    """
    return {"ok": True, "title": title, "priority": priority}


@cactus.tool
def create_note(
    text: Annotated[str, cactus.Field(min_length=1, max_length=200)],
    title: Optional[Phrase] = None,
):
    """Save a free-form note. Notes store information; they never fire alerts.

    Args:
        text: The note content copied word for word.
        title: A short note title; include only when the user names one.
    """
    return {"ok": True, "text": text, "title": title}


TOOLS = [set_timer, create_reminder, create_calendar_event, add_task, create_note]
SYSTEM = "Copy titles, messages, and date or time phrases verbatim from the user; never rephrase or resolve them. Map each explicit supported request to exactly one declared call. Do not guess missing values. Unsupported, invalid, ambiguous, and negated requests return no call."


TEST_CASES = [
    {'query': 'start a timer for 25 minutes', 'calls': [{'name': 'set_timer', 'arguments': {'time_human': '25 minutes'}}], 'category': 'positive'},
    {'query': 'run a timer for 90 seconds', 'calls': [{'name': 'set_timer', 'arguments': {'time_human': '90 seconds'}}], 'category': 'positive'},
    {'query': 'set a timer that goes off at 9:15pm', 'calls': [{'name': 'set_timer', 'arguments': {'time_human': '9:15pm'}}], 'category': 'positive'},
    {'query': 'remind me at 6pm to water the plants', 'calls': [{'name': 'create_reminder', 'arguments': {'message': 'water the plants', 'date_time_human': '6pm'}}], 'category': 'positive'},
    {'query': 'set a reminder for Sunday morning to call Grandma', 'calls': [{'name': 'create_reminder', 'arguments': {'message': 'call Grandma', 'date_time_human': 'Sunday morning'}}], 'category': 'positive'},
    {'query': 'remind me tomorrow at noon to renew the car insurance', 'calls': [{'name': 'create_reminder', 'arguments': {'message': 'renew the car insurance', 'date_time_human': 'tomorrow at noon'}}], 'category': 'positive'},
    {'query': 'could you remind me at 8:15am to take my vitamins', 'calls': [{'name': 'create_reminder', 'arguments': {'message': 'take my vitamins', 'date_time_human': '8:15am'}}], 'category': 'positive'},
    {'query': 'add dentist appointment to my calendar tomorrow at 3pm', 'calls': [{'name': 'create_calendar_event', 'arguments': {'title': 'dentist appointment', 'start_time_human': 'tomorrow at 3pm'}}], 'category': 'positive'},
    {'query': 'schedule team standup for Monday at 9am', 'calls': [{'name': 'create_calendar_event', 'arguments': {'title': 'team standup', 'start_time_human': 'Monday at 9am'}}], 'category': 'positive'},
    {'query': 'put book club on my calendar Wednesday at 7pm at Riverside Library', 'calls': [{'name': 'create_calendar_event', 'arguments': {'title': 'book club', 'start_time_human': 'Wednesday at 7pm', 'location': 'Riverside Library'}}], 'category': 'positive'},
    {'query': 'schedule lunch with Sam for Friday at 1pm at Cafe Roma', 'calls': [{'name': 'create_calendar_event', 'arguments': {'title': 'lunch with Sam', 'start_time_human': 'Friday at 1pm', 'location': 'Cafe Roma'}}], 'category': 'positive'},
    {'query': 'add renew my passport to my to-do list', 'calls': [{'name': 'add_task', 'arguments': {'title': 'renew my passport'}}], 'category': 'positive'},
    {'query': 'add descale the kettle to my tasks', 'calls': [{'name': 'add_task', 'arguments': {'title': 'descale the kettle'}}], 'category': 'positive'},
    {'query': 'add file the insurance claim as a high priority task', 'calls': [{'name': 'add_task', 'arguments': {'title': 'file the insurance claim', 'priority': 'high'}}], 'category': 'positive'},
    {'query': 'add a low priority task to organize the garage', 'calls': [{'name': 'add_task', 'arguments': {'title': 'organize the garage', 'priority': 'low'}}], 'category': 'positive'},
    {'query': 'make a note that sunflower42 is the wifi password', 'calls': [{'name': 'create_note', 'arguments': {'text': 'sunflower42 is the wifi password'}}], 'category': 'positive'},
    {'query': 'save a note titled Packing List saying bring the hiking boots', 'calls': [{'name': 'create_note', 'arguments': {'text': 'bring the hiking boots', 'title': 'Packing List'}}], 'category': 'positive'},
    {'query': 'write a note titled Garden that says plant the tulip bulbs in October', 'calls': [{'name': 'create_note', 'arguments': {'text': 'plant the tulip bulbs in October', 'title': 'Garden'}}], 'category': 'positive'},
    {'query': 'remind me to take out the recycling', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'set a timer for the pasta', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'put dinner with Alex on my calendar', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'set a reminder about picking up the dry cleaning', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'email the shopping list to my roommate', 'calls': [], 'category': 'irrelevant'},
    {'query': 'check my calendar for this weekend', 'calls': [], 'category': 'irrelevant'},
    {'query': 'delete the reminder about the oil change', 'calls': [], 'category': 'irrelevant'},
    {'query': "don't set a timer for 20 minutes", 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'do not put yoga class on my calendar for Saturday at 8am', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'never remind me about the water bill again', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'set a timer for negative 5 minutes', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'remind me yesterday at 4pm to submit the timesheet', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'set a timer for 10 minutes and remind me at 5pm to hydrate', 'calls': [{'name': 'set_timer', 'arguments': {'time_human': '10 minutes'}}, {'name': 'create_reminder', 'arguments': {'message': 'hydrate', 'date_time_human': '5pm'}}], 'category': 'parallel'},
    {'query': 'add buy candles to my task list and note that rent is due Friday', 'calls': [{'name': 'add_task', 'arguments': {'title': 'buy candles'}}, {'name': 'create_note', 'arguments': {'text': 'rent is due Friday'}}], 'category': 'parallel'},
]


def run_tests(min_confidence=0.0, verbose=True):
    return _harness.run_tests(sys.modules[__name__], min_confidence, verbose)


def __getattr__(name):
    if name == "agent":
        return _harness.agent_for(sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
