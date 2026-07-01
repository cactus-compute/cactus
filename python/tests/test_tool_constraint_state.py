from __future__ import annotations

import json

import pytest

from cactus import cactus_complete, cactus_destroy, cactus_init

from .bundles import WEIGHTS, _iter_bundle_candidates, _valid_bundle

BUNDLE = next(
    (c for c in _iter_bundle_candidates("gemma-4-e2b-it") if _valid_bundle(c)),
    WEIGHTS / "gemma-4-e2b-it",
)

def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


GET_WEATHER = _tool("get_weather", "Get the current weather for a location",
                    {"location": {"type": "string", "description": "City name"}}, ["location"])
SET_ALARM = _tool("set_alarm", "Set an alarm for a time",
                  {"time": {"type": "string", "description": "HH:MM"}}, ["time"])
SEND = _tool("send", "Send a quick ping (no body) to a user id",
             {"user": {"type": "string"}}, ["user"])
SEND_MESSAGE = _tool("send_message", "Send a text message with a body to a contact",
                     {"recipient": {"type": "string"}, "body": {"type": "string"}}, ["recipient", "body"])
SEARCH = _tool("search", "Search local notes", {"query": {"type": "string"}}, ["query"])
SEARCH_WEB = _tool("search_web", "Search the public web", {"query": {"type": "string"}}, ["query"])
GET = _tool("get", "Get a raw value by key", {"key": {"type": "string"}}, ["key"])
GET_USER = _tool("get_user", "Fetch a user profile by id", {"id": {"type": "string"}}, ["id"])
PING_SERVER = _tool("ping_server", "Check server health, takes no arguments", {}, [])
WEATHER_UNIT = _tool("get_weather", "Get weather for a city in a unit",
                     {"location": {"type": "string", "description": "city"},
                      "unit": {"type": "string", "description": "temperature unit",
                               "enum": ["celsius", "fahrenheit"]}},
                     ["location", "unit"])
CREATE_EVENT = _tool("create_event", "Create a calendar event",
                     {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                      "location": {"type": "string"}},
                     ["title", "date"])


def _forced_calls(model, tools, user):
    result = cactus_complete(
        model,
        [
            {"role": "system", "content": "You are a helpful assistant that can use tools."},
            {"role": "user", "content": user},
        ],
        {"force_tools": True, "max_tokens": 128, "temperature": 0},
        tools,
    )
    assert result.get("success"), result
    calls = result.get("function_calls") or []
    return [c if isinstance(c, dict) else json.loads(c) for c in calls]


@pytest.fixture(scope="module")
def model():
    if not _valid_bundle(BUNDLE):
        pytest.skip(f"Live-test model not found: {BUNDLE}")
    handle = cactus_init(str(BUNDLE))
    yield handle
    cactus_destroy(handle)


def test_turn_after_force_tools_is_not_degenerate(model):
    tools = [GET_WEATHER]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in Tokyo right now?"},
    ]
    forced = cactus_complete(
        model, messages, {"force_tools": True, "max_tokens": 200, "temperature": 0}, tools
    )
    assert forced.get("success"), forced
    assert forced.get("function_calls"), forced

    messages.append({
        "role": "assistant",
        "content": forced.get("response", ""),
        "tool_calls": [{"type": "function", "function": {
            "name": "get_weather", "arguments": {"location": "Tokyo"},
        }}],
    })
    messages.append({
        "role": "tool",
        "name": "get_weather",
        "content": json.dumps({"temperature_c": 18}),
    })
    followup = cactus_complete(model, messages, {"max_tokens": 150, "temperature": 0}, tools)
    assert followup.get("success"), followup
    response = followup.get("response", "")
    assert response.strip()
    assert "<|tool_call>" not in response


def test_prefix_name_is_not_shadowed(model):
    calls = _forced_calls(model, [SEND, SEND_MESSAGE], "Message my contact Alice: running late, start without me.")
    assert calls
    assert calls[0]["name"] == "send_message"


@pytest.mark.parametrize("tools, user, want", [
    ([SEARCH, SEARCH_WEB], "Search the public web for the best ramen in Osaka.", "search_web"),
    ([GET, GET_USER], "Fetch the user profile for account 42.", "get_user"),
])
def test_prefix_pairs_reach_longer_name(model, tools, user, want):
    calls = _forced_calls(model, tools, user)
    assert calls
    assert calls[0]["name"] == want


def test_shorter_prefix_name_still_reachable(model):
    calls = _forced_calls(model, [SEND, SEND_MESSAGE], "Ping user u_42.")
    assert calls
    assert calls[0]["name"] == "send"


def test_forced_name_is_always_a_declared_tool(model):
    tools = [GET_WEATHER, SET_ALARM]
    declared = {t["function"]["name"] for t in tools}
    calls = _forced_calls(model, tools, "What's the weather in Tokyo right now?")
    assert calls
    assert calls[0]["name"] in declared


def test_no_argument_function_emits_empty_object(model):
    calls = _forced_calls(model, [PING_SERVER], "Check whether the server is healthy.")
    assert calls
    assert calls[0]["name"] == "ping_server"
    assert calls[0]["arguments"] == {}


def test_argument_keys_are_declared_parameters(model):
    calls = _forced_calls(model, [GET_WEATHER], "What's the weather in Berlin?")
    assert calls
    assert calls[0]["name"] == "get_weather"
    assert set(calls[0]["arguments"]).issubset({"location"})


@pytest.mark.parametrize("user", [
    "What's the weather in Tokyo in kelvin?",
    "Weather in Paris, in Kelvin please.",
    "Weather in Berlin.",
])
def test_enum_argument_restricted_to_allowed_values(model, user):
    calls = _forced_calls(model, [WEATHER_UNIT], user)
    assert calls
    assert calls[0]["arguments"].get("unit") in ("celsius", "fahrenheit")


@pytest.mark.parametrize("user", [
    "Make a calendar event.",
    "Add an event called Standup.",
    "Schedule a dentist appointment.",
])
def test_required_arguments_are_always_present(model, user):
    calls = _forced_calls(model, [CREATE_EVENT], user)
    assert calls
    args = calls[0]["arguments"]
    assert "title" in args and "date" in args


def test_enum_with_shared_prefixes_can_reach_longest_value(model):
    tool = _tool("pick_plan", "pick a plan",
                 {"plan": {"type": "string", "enum": ["pro", "pro_plus", "pro_plus_max"]}}, ["plan"])
    calls = _forced_calls(model, [tool], "Pick the pro plus max plan.")
    assert calls
    assert calls[0]["arguments"].get("plan") in ("pro", "pro_plus", "pro_plus_max")


def test_enum_value_containing_spaces(model):
    tool = _tool("move_card", "move a kanban card",
                 {"card": {"type": "string"},
                  "column": {"type": "string", "enum": ["to do", "in progress", "done"]}},
                 ["card", "column"])
    calls = _forced_calls(model, [tool], "Move the login card to in progress.")
    assert calls
    assert calls[0]["arguments"].get("column") in ("to do", "in progress", "done")


def test_single_value_enum_is_forced(model):
    tool = _tool("finalize", "finalize", {"status": {"type": "string", "enum": ["confirmed"]}}, ["status"])
    calls = _forced_calls(model, [tool], "Finalize it however you want.")
    assert calls
    assert calls[0]["arguments"].get("status") == "confirmed"


def test_prefix_parameter_names_both_required_present(model):
    tool = _tool("make_range", "range", {"to": {"type": "string"}, "total": {"type": "string"}}, ["to", "total"])
    calls = _forced_calls(model, [tool], "Make a range up to 10 with a total of 100.")
    assert calls
    assert {"to", "total"} <= set(calls[0]["arguments"])


def test_many_required_arguments_all_present(model):
    tool = _tool("register", "register a device",
                 {k: {"type": "string"} for k in ("name", "os", "owner", "ip", "room")},
                 ["name", "os", "owner", "ip", "room"])
    calls = _forced_calls(model, [tool], "Register a device.")
    assert calls
    assert {"name", "os", "owner", "ip", "room"} <= set(calls[0]["arguments"])


def test_prefix_tool_with_required_enum_on_longer_name(model):
    send = _tool("send", "ping a user", {"user": {"type": "string"}}, ["user"])
    send_message = _tool("send_message", "send a message",
                         {"to": {"type": "string"}, "body": {"type": "string"},
                          "urgency": {"type": "string", "enum": ["normal", "urgent"]}},
                         ["to", "body", "urgency"])
    calls = _forced_calls(model, [send, send_message], "Send an urgent message to Alice saying the server is down.")
    assert calls
    assert calls[0]["name"] == "send_message"
    assert calls[0]["arguments"].get("urgency") in ("normal", "urgent")
    assert {"to", "body", "urgency"} <= set(calls[0]["arguments"])
