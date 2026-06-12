from __future__ import annotations

import json
from pathlib import Path

import pytest

from cactus import cactus_complete, cactus_destroy, cactus_init

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT_ROOT / "weights" / "gemma-4-e2b-it"

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}


def _valid_bundle(path: Path) -> bool:
    return (path / "config.txt").exists() and (path / "components" / "manifest.json").exists()


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
