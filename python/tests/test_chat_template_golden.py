from __future__ import annotations

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from .bundles import PROJECT_ROOT, WEIGHTS, _iter_bundle_candidates, _read_model_type, _valid_bundle

MONKEY_IMAGE = PROJECT_ROOT / "cactus-engine" / "tests" / "assets" / "test_monkey.png"

FAMILIES = [
    ("gemma-4-e2b-it", "google/gemma-4-E2B-it"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("lfm2-350m", "LiquidAI/LFM2-350M"),
    ("functiongemma-270m-it", "google/functiongemma-270m-it"),
]
FAMILY_IDS = ["gemma4", "qwen3", "lfm2", "functiongemma"]
THINKING_TYPES = {"gemma4", "qwen"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name."},
                },
                "required": ["timezone"],
            },
        },
    },
]

SYSTEM_CHAT = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Germany?"},
]

TOOLS_USER = [
    {"role": "user", "content": "What is the weather in Paris?"},
]

TOOL_CALL_RESPONSE = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 22C"},
]

PARALLEL_TOOL_CALLS = [
    {"role": "user", "content": "Weather in Paris and time in UTC?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
            {"id": "call_2", "type": "function",
             "function": {"name": "get_time", "arguments": {"timezone": "UTC"}}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 22C"},
    {"role": "tool", "tool_call_id": "call_2", "name": "get_time", "content": "12:00"},
]

THINKING_CHAT = [
    {"role": "user", "content": "Briefly explain entropy."},
]

CASES = [
    ("system_chat", SYSTEM_CHAT, None, False),
    ("tools_user", TOOLS_USER, TOOLS, False),
    ("tool_call_response", TOOL_CALL_RESPONSE, TOOLS, False),
    ("parallel_tool_calls", PARALLEL_TOOL_CALLS, TOOLS, False),
    ("thinking_enabled", THINKING_CHAT, None, True),
    ("thinking_disabled", THINKING_CHAT, None, False),
]


def _pythonic_call(call: dict) -> str:
    fn = call["function"]
    args = ", ".join(f"{key}={json.dumps(value)}" for key, value in fn["arguments"].items())
    return f"{fn['name']}({args})"


def _embed_lfm2_tool_calls(message: dict) -> dict:
    calls = ", ".join(_pythonic_call(call) for call in message["tool_calls"])
    embedded = dict(message)
    embedded["content"] = message["content"] + f"<|tool_call_start|>[{calls}]<|tool_call_end|>"
    del embedded["tool_calls"]
    return embedded


def _to_hf_messages(messages: list[dict], model_type: str) -> list[dict]:
    out = []
    for message in messages:
        if message.get("images"):
            message = dict(message)
            message["content"] = [{"type": "image"} for _ in message.pop("images")] + [
                {"type": "text", "text": message["content"]}
            ]
        if model_type == "lfm2" and message.get("tool_calls"):
            message = _embed_lfm2_tool_calls(message)
        out.append(message)
    return out


_LOADED: dict[str, tuple] = {}


@pytest.fixture(scope="session")
def load_family():
    from cactus import cactus_destroy, cactus_init

    def _load(bundle_name: str, hf_id: str):
        if bundle_name not in _LOADED:
            bundle = next(
                (c for c in _iter_bundle_candidates(bundle_name) if _valid_bundle(c)),
                WEIGHTS / bundle_name,
            )
            if not _valid_bundle(bundle):
                pytest.skip(f"bundle not prepared: {bundle}")
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
            except OSError as exc:
                pytest.skip(f"HF tokenizer not cached for {hf_id}: {exc}")
            _LOADED[bundle_name] = (cactus_init(str(bundle)), tokenizer, _read_model_type(bundle))
        return _LOADED[bundle_name]

    yield _load
    while _LOADED:
        model, _, _ = _LOADED.popitem()[1]
        cactus_destroy(model)


def _assert_render_matches(model, tokenizer, engine_render: str, hf_render: str, label: str):
    from cactus import cactus_tokenize

    assert engine_render == hf_render, (
        f"{label}: engine chat-template render diverges from the HF reference template"
    )
    engine_ids = cactus_tokenize(model, engine_render)
    hf_ids = tokenizer(engine_render, add_special_tokens=False).input_ids
    assert engine_ids == hf_ids, (
        f"{label}: engine tokenization of the render diverges from the HF tokenizer"
    )


@pytest.mark.parametrize(("case_name", "messages", "tools", "thinking"), CASES,
                         ids=[case[0] for case in CASES])
@pytest.mark.parametrize(("bundle_name", "hf_id"), FAMILIES, ids=FAMILY_IDS)
def test_render_matches_hf_template(load_family, bundle_name, hf_id, case_name, messages,
                                    tools, thinking):
    from cactus import cactus_render_prompt

    model, tokenizer, model_type = load_family(bundle_name, hf_id)
    if case_name.startswith("thinking") and not any(t in model_type for t in THINKING_TYPES):
        pytest.skip(f"{model_type} has no thinking template support")
    engine_render = cactus_render_prompt(
        model, messages, options={"enable_thinking_if_supported": thinking}, tools=tools
    )
    hf_render = tokenizer.apply_chat_template(
        _to_hf_messages(messages, model_type),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    _assert_render_matches(model, tokenizer, engine_render, hf_render,
                           f"{bundle_name}/{case_name}")


def _monkey_soft_tokens(hf_id: str) -> int:
    from huggingface_hub.constants import HF_HUB_CACHE
    from PIL import Image
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

    repo_dir = Path(HF_HUB_CACHE) / f"models--{hf_id.replace('/', '--')}"
    configs = sorted(repo_dir.glob("snapshots/*/processor_config.json"))
    if not configs:
        pytest.skip(f"no cached processor_config.json for {hf_id}")
    image_processor = Gemma4ImageProcessor(
        **json.loads(configs[0].read_text(encoding="utf-8"))["image_processor"]
    )
    with Image.open(MONKEY_IMAGE) as image:
        return image_processor(image)["num_soft_tokens_per_image"][0]


def test_gemma4_image_render_matches_hf_template(load_family):
    from cactus import cactus_render_prompt

    bundle_name, hf_id = FAMILIES[0]
    model, tokenizer, model_type = load_family(bundle_name, hf_id)
    messages = [
        {"role": "user", "content": "What is in this image?", "images": [str(MONKEY_IMAGE)]},
    ]
    engine_render = cactus_render_prompt(
        model, messages, options={"enable_thinking_if_supported": False}
    )
    hf_render = tokenizer.apply_chat_template(
        _to_hf_messages(messages, model_type),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    soft_tokens = _monkey_soft_tokens(hf_id)
    expanded = hf_render.replace(
        tokenizer.image_token,
        tokenizer.boi_token + tokenizer.image_token * soft_tokens + tokenizer.eoi_token,
    )
    _assert_render_matches(model, tokenizer, engine_render, expanded, f"{bundle_name}/image")
