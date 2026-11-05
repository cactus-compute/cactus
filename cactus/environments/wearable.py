"""Smartwatch companion actions: notifications, workouts, finding the phone.

Replace the sender list with the contacts your integration exposes; the model
only ever replies to senders it can name. Workout types use the gerund forms
the model was trained on.
"""

import sys
from typing import Annotated, Literal

import cactus
from cactus.environments import _harness

Sender = Literal["Maya", "Leo", "Dr. Patel", "Cactus Team"]


@cactus.tool
def reply_to_notification(notification_match: Sender, text: Annotated[str, cactus.Field(min_length=1, max_length=240)]):
    """Reply to a message notification from a known sender. Copy the reply text word for word, preserving capitalization. Use dismiss_notification to clear one without replying.

    Args:
        notification_match: Who the notification is from.
        text: The reply message copied word for word.
    """
    return {"ok": True, "notification_match": notification_match, "text": text}


@cactus.tool
def dismiss_notification(notification_match: Sender):
    """Dismiss a sender's notification without replying.

    Args:
        notification_match: Whose notification to dismiss.
    """
    return {"ok": True, "notification_match": notification_match}


@cactus.tool
def start_workout(workout_type: Literal["running", "walking", "cycling", "swimming", "strength", "yoga"]):
    """Start tracking a workout of an explicitly named type: running, walking, cycling, swimming, strength, or yoga. Use end_workout to finish one already running.

    Args:
        workout_type: The type of workout the user named.
    """
    return {"ok": True, "workout_type": workout_type}


@cactus.tool
def end_workout():
    """End the current workout session. Starting a new one is start_workout. Takes no arguments."""
    return {"ok": True, "state": "ended"}


@cactus.tool
def find_my_phone():
    """Trigger the phone to ring so you can find it. Takes no arguments."""
    return {"ok": True, "ringing": True}


TOOLS = [reply_to_notification, dismiss_notification, start_workout, end_workout, find_my_phone]
SYSTEM = "Copy reply text verbatim, preserving capitalization. Map each explicit supported watch action to exactly one declared call. Do not guess missing values. Unsupported, invalid, ambiguous, and negated requests return no call."


TEST_CASES = [
    {'query': 'reply to Maya saying be there in 10 minutes', 'calls': [{'name': 'reply_to_notification', 'arguments': {'notification_match': 'Maya', 'text': 'be there in 10 minutes'}}], 'category': 'positive'},
    {'query': 'reply to Dr. Patel with See you Thursday', 'calls': [{'name': 'reply_to_notification', 'arguments': {'notification_match': 'Dr. Patel', 'text': 'See you Thursday'}}], 'category': 'positive'},
    {'query': 'answer the Cactus Team message saying the build is green', 'calls': [{'name': 'reply_to_notification', 'arguments': {'notification_match': 'Cactus Team', 'text': 'the build is green'}}], 'category': 'positive'},
    {'query': 'send Leo a reply that says lunch works for me', 'calls': [{'name': 'reply_to_notification', 'arguments': {'notification_match': 'Leo', 'text': 'lunch works for me'}}], 'category': 'positive'},
    {'query': 'dismiss the notification from Leo', 'calls': [{'name': 'dismiss_notification', 'arguments': {'notification_match': 'Leo'}}], 'category': 'positive'},
    {'query': 'swipe away the alert from Maya', 'calls': [{'name': 'dismiss_notification', 'arguments': {'notification_match': 'Maya'}}], 'category': 'positive'},
    {'query': 'clear the Dr. Patel notification', 'calls': [{'name': 'dismiss_notification', 'arguments': {'notification_match': 'Dr. Patel'}}], 'category': 'positive'},
    {'query': 'get rid of the Cactus Team notification', 'calls': [{'name': 'dismiss_notification', 'arguments': {'notification_match': 'Cactus Team'}}], 'category': 'positive'},
    {'query': 'start a running workout', 'calls': [{'name': 'start_workout', 'arguments': {'workout_type': 'running'}}], 'category': 'positive'},
    {'query': 'begin a swimming session on my watch', 'calls': [{'name': 'start_workout', 'arguments': {'workout_type': 'swimming'}}], 'category': 'positive'},
    {'query': 'kick off some strength training', 'calls': [{'name': 'start_workout', 'arguments': {'workout_type': 'strength'}}], 'category': 'positive'},
    {'query': 'track a cycling workout', 'calls': [{'name': 'start_workout', 'arguments': {'workout_type': 'cycling'}}], 'category': 'positive'},
    {'query': 'go ahead and start a walking workout', 'calls': [{'name': 'start_workout', 'arguments': {'workout_type': 'walking'}}], 'category': 'positive'},
    {'query': 'end my workout', 'calls': [{'name': 'end_workout', 'arguments': {}}], 'category': 'positive'},
    {'query': 'finish the current workout session', 'calls': [{'name': 'end_workout', 'arguments': {}}], 'category': 'positive'},
    {'query': 'wrap up my workout now', 'calls': [{'name': 'end_workout', 'arguments': {}}], 'category': 'positive'},
    {'query': 'find my phone', 'calls': [{'name': 'find_my_phone', 'arguments': {}}], 'category': 'positive'},
    {'query': 'make my phone ring so I can locate it', 'calls': [{'name': 'find_my_phone', 'arguments': {}}], 'category': 'positive'},
    {'query': 'send a reply to Maya', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'dismiss the latest notification', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'start a workout', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'reply saying I am stuck in a meeting', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'check my heart rate on the watch', 'calls': [], 'category': 'irrelevant'},
    {'query': 'call Maya from my watch', 'calls': [], 'category': 'irrelevant'},
    {'query': 'show my step count for today', 'calls': [], 'category': 'irrelevant'},
    {'query': "don't reply to the message from Leo", 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'do not start a running workout yet', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'never dismiss notifications from Dr. Patel', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'reply to Priya saying happy birthday', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'start a pilates workout', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'dismiss the notification from Leo and start a strength workout', 'calls': [{'name': 'dismiss_notification', 'arguments': {'notification_match': 'Leo'}}, {'name': 'start_workout', 'arguments': {'workout_type': 'strength'}}], 'category': 'parallel'},
    {'query': 'find my phone and reply to Maya saying almost home', 'calls': [{'name': 'reply_to_notification', 'arguments': {'notification_match': 'Maya', 'text': 'almost home'}}, {'name': 'find_my_phone', 'arguments': {}}], 'category': 'parallel'},
]


def run_tests(min_confidence=0.0, verbose=True):
    return _harness.run_tests(sys.modules[__name__], min_confidence, verbose)


def __getattr__(name):
    if name == "agent":
        return _harness.agent_for(sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
