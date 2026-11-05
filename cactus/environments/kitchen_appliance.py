"""Countertop appliance hub: oven, coffee maker, dishwasher.

The shape that matters on a microcontroller: hard numeric bounds make unsafe
requests unrepresentable, and the one read-only tool keeps checks separate
from actions. Optional settings the model tends to guess (oven modes, cup
sizes, default cycles) are deliberately absent.
"""

import sys
from typing import Annotated, Literal, Optional

import cactus
from cactus.environments import _harness


@cactus.tool
def set_oven(temperature: Annotated[int, cactus.Field(ge=50, le=250)]):
    """Preheat or set the oven to a temperature in degrees Celsius. Use get_appliance_status to check on it.

    Args:
        temperature: Target temperature in degrees Celsius.
    """
    return {"ok": True, "temperature": temperature}


@cactus.tool
def control_coffee_maker(action: Literal["brew", "stop", "warm"]):
    """Control the coffee maker. Never use this for the oven or the dishwasher.

    Args:
        action: brew, stop, or warm.
    """
    return {"ok": True, "action": action}


@cactus.tool
def start_dishwasher(cycle: Optional[Literal["eco", "heavy", "quick"]] = None):
    """Start the dishwasher, optionally on a stated wash cycle.

    Args:
        cycle: The wash cycle; include only when stated.
    """
    return {"ok": True, "cycle": cycle}


@cactus.tool
def set_cooking_timer(
    label: Annotated[str, cactus.Field(min_length=1, max_length=40)],
    minutes: Annotated[int, cactus.Field(ge=1, le=360)],
):
    """Set a named cooking timer.

    Args:
        label: Label for the timer, copied from the user.
        minutes: Timer duration in minutes.
    """
    return {"ok": True, "label": label, "minutes": minutes}


@cactus.tool
def get_appliance_status(appliance: Literal["oven", "coffee_maker", "dishwasher"]):
    """Read an appliance's current status. This changes nothing; any request to change settings uses the other tools.

    Args:
        appliance: Which appliance to check.
    """
    return {"ok": True, "appliance": appliance, "status": "idle"}


TOOLS = [set_oven, control_coffee_maker, start_dishwasher, set_cooking_timer, get_appliance_status]
SYSTEM = "Map each explicit supported appliance action to exactly one declared call; never duplicate an action. Do not guess missing values. Unsupported, invalid, ambiguous, and negated requests return no call."


TEST_CASES = [
    {'query': 'preheat the oven to 180 degrees', 'calls': [{'name': 'set_oven', 'arguments': {'temperature': 180}}], 'category': 'positive'},
    {'query': 'set the oven to 220 degrees', 'calls': [{'name': 'set_oven', 'arguments': {'temperature': 220}}], 'category': 'positive'},
    {'query': 'bake at 200 degrees', 'calls': [{'name': 'set_oven', 'arguments': {'temperature': 200}}], 'category': 'positive'},
    {'query': 'bring the oven up to 160 degrees', 'calls': [{'name': 'set_oven', 'arguments': {'temperature': 160}}], 'category': 'positive'},
    {'query': 'brew a fresh pot of coffee', 'calls': [{'name': 'control_coffee_maker', 'arguments': {'action': 'brew'}}], 'category': 'positive'},
    {'query': 'start brewing another batch of coffee', 'calls': [{'name': 'control_coffee_maker', 'arguments': {'action': 'brew'}}], 'category': 'positive'},
    {'query': 'stop the coffee maker', 'calls': [{'name': 'control_coffee_maker', 'arguments': {'action': 'stop'}}], 'category': 'positive'},
    {'query': 'keep my coffee warm', 'calls': [{'name': 'control_coffee_maker', 'arguments': {'action': 'warm'}}], 'category': 'positive'},
    {'query': 'run the dishwasher on the eco cycle', 'calls': [{'name': 'start_dishwasher', 'arguments': {'cycle': 'eco'}}], 'category': 'positive'},
    {'query': 'start a heavy dishwasher cycle', 'calls': [{'name': 'start_dishwasher', 'arguments': {'cycle': 'heavy'}}], 'category': 'positive'},
    {'query': 'run a quick cycle on the dishwasher', 'calls': [{'name': 'start_dishwasher', 'arguments': {'cycle': 'quick'}}], 'category': 'positive'},
    {'query': 'set a pasta timer for 10 minutes', 'calls': [{'name': 'set_cooking_timer', 'arguments': {'label': 'pasta', 'minutes': 10}}], 'category': 'positive'},
    {'query': 'start a 90 minute timer for the brisket', 'calls': [{'name': 'set_cooking_timer', 'arguments': {'label': 'brisket', 'minutes': 90}}], 'category': 'positive'},
    {'query': 'give me a 5 minute timer for the eggs', 'calls': [{'name': 'set_cooking_timer', 'arguments': {'label': 'eggs', 'minutes': 5}}], 'category': 'positive'},
    {'query': 'the cookies need a 12 minute timer', 'calls': [{'name': 'set_cooking_timer', 'arguments': {'label': 'cookies', 'minutes': 12}}], 'category': 'positive'},
    {'query': 'check the oven', 'calls': [{'name': 'get_appliance_status', 'arguments': {'appliance': 'oven'}}], 'category': 'positive'},
    {'query': 'check on the dishwasher', 'calls': [{'name': 'get_appliance_status', 'arguments': {'appliance': 'dishwasher'}}], 'category': 'positive'},
    {'query': 'check the coffee maker for me', 'calls': [{'name': 'get_appliance_status', 'arguments': {'appliance': 'coffee_maker'}}], 'category': 'positive'},
    {'query': 'heat up the oven for the garlic bread', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'set a cooking timer for the rice', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'start a timer for 25 minutes', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'change the coffee maker setting', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'defrost some chicken in the microwave', 'calls': [], 'category': 'irrelevant'},
    {'query': 'boil the kettle for tea', 'calls': [], 'category': 'irrelevant'},
    {'query': 'turn the fridge down to 3 degrees', 'calls': [], 'category': 'irrelevant'},
    {'query': "don't preheat the oven to 220 degrees tonight", 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'do not run the dishwasher on heavy today', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'never brew coffee this late at night', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'get the oven up to 300 degrees', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'set a stew timer for 500 minutes', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'set the oven to 180 degrees and start a pizza timer for 15 minutes', 'calls': [{'name': 'set_oven', 'arguments': {'temperature': 180}}, {'name': 'set_cooking_timer', 'arguments': {'label': 'pizza', 'minutes': 15}}], 'category': 'parallel'},
    {'query': 'brew coffee and run the dishwasher on quick', 'calls': [{'name': 'control_coffee_maker', 'arguments': {'action': 'brew'}}, {'name': 'start_dishwasher', 'arguments': {'cycle': 'quick'}}], 'category': 'parallel'},
]


def run_tests(min_confidence=0.0, verbose=True):
    return _harness.run_tests(sys.modules[__name__], min_confidence, verbose)


def __getattr__(name):
    if name == "agent":
        return _harness.agent_for(sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
