"""Smart home control for a bounded set of rooms and devices.

Replace the Literal values with the rooms and devices your product exposes.
Keep the shapes: every closed set is an enum, every number has bounds, and
no correct call ever needs a value the user did not say. One learned rule:
avoid enum values that hide inside likely query words (a room named office
poisons an off action, so this home has a study).
"""

import sys
from typing import Annotated, Literal, Optional

import cactus
from cactus.environments import _harness

Room = Literal["kitchen", "living_room", "bedroom", "study"]


@cactus.tool
def control_lights(
    room: Room,
    action: Literal["on", "off", "dim"],
    brightness_percent: Annotated[Optional[int], cactus.Field(ge=0, le=100)] = None,
    color: Optional[Literal["warm white", "cool white", "red", "green", "blue"]] = None,
):
    """Turn lights on or off in a room, dim them to a brightness percentage, or set their color. A color request means action on. Never use this for blinds, fans, or any other device.

    Args:
        room: The room to control.
        action: on, off, or dim.
        brightness_percent: Brightness from 0 to 100. A dim request with a number must carry it in this same call.
        color: The light color; include only when the user names one.
    """
    return {"ok": True, "room": room, "action": action, "brightness_percent": brightness_percent, "color": color}


@cactus.tool
def set_thermostat(temperature: Annotated[int, cactus.Field(ge=10, le=30)]):
    """Set the home thermostat to a target temperature in degrees Celsius. This never controls fans, lights, or any other device.

    Args:
        temperature: Target temperature in degrees Celsius.
    """
    return {"ok": True, "temperature": temperature}


@cactus.tool
def control_fan(
    room: Literal["living_room", "bedroom", "study"],
    action: Literal["on", "off"],
    speed: Optional[Literal["low", "medium", "high"]] = None,
):
    """Turn a room fan on or off, optionally at a stated speed. This never changes the thermostat.

    Args:
        room: The room whose fan to control.
        action: on or off.
        speed: Fan speed; include only when stated.
    """
    return {"ok": True, "room": room, "action": action, "speed": speed}


@cactus.tool
def control_blinds(room: Room, action: Literal["open", "close"]):
    """Open or close the window blinds in one stated room. Never pick the room yourself. Never use this for lights or the vacuum.

    Args:
        room: The room whose blinds to move.
        action: open or close.
    """
    return {"ok": True, "room": room, "action": action}


@cactus.tool
def start_robot_vacuum(
    action: Literal["start", "stop", "dock"],
    room: Optional[Literal["kitchen", "living_room", "bedroom"]] = None,
):
    """Start or stop the robot vacuum, or send it back to its dock to charge.

    Args:
        action: start, stop, or dock.
        room: The room to clean; include only when starting a run in a stated room.
    """
    return {"ok": True, "action": action, "room": room}


TOOLS = [control_lights, set_thermostat, control_fan, control_blinds, start_robot_vacuum]
SYSTEM = "Map each explicit supported home action to exactly one declared call; never duplicate an action. Do not guess missing targets or values. Unsupported, invalid, ambiguous, and negated requests return no call."


TEST_CASES = [
    {'query': 'turn on the kitchen lights', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'kitchen', 'action': 'on'}}], 'category': 'positive'},
    {'query': 'switch off the lights in the bedroom', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'bedroom', 'action': 'off'}}], 'category': 'positive'},
    {'query': 'dim the living room lights to 35 percent', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'living_room', 'action': 'dim', 'brightness_percent': 35}}], 'category': 'positive'},
    {'query': 'turn on the study lights in warm white', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'study', 'action': 'on', 'color': 'warm white'}}], 'category': 'positive'},
    {'query': 'put the bedroom lights on in blue', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'bedroom', 'action': 'on', 'color': 'blue'}}], 'category': 'positive'},
    {'query': 'set the thermostat to 22 degrees', 'calls': [{'name': 'set_thermostat', 'arguments': {'temperature': 22}}], 'category': 'positive'},
    {'query': 'warm the house to 24 degrees', 'calls': [{'name': 'set_thermostat', 'arguments': {'temperature': 24}}], 'category': 'positive'},
    {'query': 'cool the whole home down to 19 degrees', 'calls': [{'name': 'set_thermostat', 'arguments': {'temperature': 19}}], 'category': 'positive'},
    {'query': 'turn on the bedroom fan', 'calls': [{'name': 'control_fan', 'arguments': {'room': 'bedroom', 'action': 'on'}}], 'category': 'positive'},
    {'query': 'switch the study fan off', 'calls': [{'name': 'control_fan', 'arguments': {'room': 'study', 'action': 'off'}}], 'category': 'positive'},
    {'query': 'turn on the living room fan at high speed', 'calls': [{'name': 'control_fan', 'arguments': {'room': 'living_room', 'action': 'on', 'speed': 'high'}}], 'category': 'positive'},
    {'query': 'run the study fan on low', 'calls': [{'name': 'control_fan', 'arguments': {'room': 'study', 'action': 'on', 'speed': 'low'}}], 'category': 'positive'},
    {'query': 'open the kitchen blinds', 'calls': [{'name': 'control_blinds', 'arguments': {'room': 'kitchen', 'action': 'open'}}], 'category': 'positive'},
    {'query': 'close the blinds in the living room', 'calls': [{'name': 'control_blinds', 'arguments': {'room': 'living_room', 'action': 'close'}}], 'category': 'positive'},
    {'query': 'open up the bedroom blinds', 'calls': [{'name': 'control_blinds', 'arguments': {'room': 'bedroom', 'action': 'open'}}], 'category': 'positive'},
    {'query': 'start the robot vacuum', 'calls': [{'name': 'start_robot_vacuum', 'arguments': {'action': 'start'}}], 'category': 'positive'},
    {'query': 'start vacuuming the bedroom', 'calls': [{'name': 'start_robot_vacuum', 'arguments': {'action': 'start', 'room': 'bedroom'}}], 'category': 'positive'},
    {'query': 'stop the robot vacuum', 'calls': [{'name': 'start_robot_vacuum', 'arguments': {'action': 'stop'}}], 'category': 'positive'},
    {'query': 'turn on the lights', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'set the thermostat to something comfortable', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'open the blinds', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'turn the fan on', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'lock the back door', 'calls': [], 'category': 'irrelevant'},
    {'query': 'check whether the robot vacuum is charging', 'calls': [], 'category': 'irrelevant'},
    {'query': 'play some jazz in the living room', 'calls': [], 'category': 'irrelevant'},
    {'query': "don't turn on the study lights", 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'do not close the living room blinds', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'never run the vacuum while I am on a call', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'dim the bedroom lights to 150 percent', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'set the thermostat to 40 degrees', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'turn off the bedroom lights and set the thermostat to 18 degrees', 'calls': [{'name': 'control_lights', 'arguments': {'room': 'bedroom', 'action': 'off'}}, {'name': 'set_thermostat', 'arguments': {'temperature': 18}}], 'category': 'parallel'},
    {'query': 'start the vacuum in the kitchen and open the living room blinds', 'calls': [{'name': 'start_robot_vacuum', 'arguments': {'action': 'start', 'room': 'kitchen'}}, {'name': 'control_blinds', 'arguments': {'room': 'living_room', 'action': 'open'}}], 'category': 'parallel'},
]


def run_tests(min_confidence=0.0, verbose=True):
    return _harness.run_tests(sys.modules[__name__], min_confidence, verbose)


def __getattr__(name):
    if name == "agent":
        return _harness.agent_for(sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
