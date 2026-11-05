import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tok():
    from cactus.model.tokenizer import get_tokenizer
    return get_tokenizer(8192)


def test_render_example_markers():
    from cactus.model.finetune import render_example
    from cactus.model.tokenizer import IM_START, TOOLS_START, TOOL_CALL_START, THINK_START

    example = {"tools": [{"name": "f", "parameters": {"type": "object", "properties": {}}}],
               "query": "do the thing", "reasoning": "because reasons",
               "answers": [{"name": "f", "arguments": {}}]}
    prompt, target = render_example(example)
    assert IM_START in prompt
    assert TOOLS_START in prompt
    assert "do the thing" in prompt
    assert THINK_START in target
    assert "because reasons" in target
    assert TOOL_CALL_START in target
    assert '"name":"f"' in target


def test_render_example_matches_training_template():
    from cactus.model.finetune import render_example
    from cactus.model.tokenizer import (IM_END, THINK_START, THINK_END,
                                        TOOL_CALL_START, TOOL_CALL_END)

    example = {"tools": [], "query": "q", "reasoning": "'q' -> f",
               "answers": [{"name": "f", "arguments": {}}]}
    _, target = render_example(example)
    assert target == (THINK_START + "\n'q' -> f\n" + THINK_END + "\n"
                      + TOOL_CALL_START + '[{"name":"f","arguments":{}}]'
                      + TOOL_CALL_END + IM_END)


def test_render_example_empty_reasoning_omits_think():
    from cactus.model.finetune import render_example
    from cactus.model.tokenizer import THINK_START, TOOL_CALL_START

    for example in ({"tools": [], "query": "q", "answers": []},
                    {"tools": [], "query": "q", "reasoning": "", "answers": []},
                    {"tools": [], "query": "q", "reasoning": None, "answers": []},
                    {"tools": [], "query": "q", "reasoning": "  ", "answers": []}):
        _, target = render_example(example)
        assert THINK_START not in target
        assert target.startswith(TOOL_CALL_START)


def test_render_example_keeps_unicode():
    from cactus.model.finetune import render_example

    example = {"tools": [{"name": "f", "parameters": {"type": "object", "properties": {
                   "name": {"type": "string", "description": "café"}}}}],
               "query": "add Tomás",
               "answers": [{"name": "f", "arguments": {"name": "Tomás"}}]}
    prompt, target = render_example(example)
    assert "café" in prompt
    assert '"Tomás"' in target


def test_render_example_accepts_function_calls_alias():
    from cactus.model.finetune import render_example
    example = {"tools": [], "query": "hi",
               "function_calls": [{"name": "g", "arguments": {"x": 1}}]}
    _, target = render_example(example)
    assert '"name":"g"' in target


def test_encode_loss_mask_targets_only(tok):
    from cactus.model.finetune import _encode, render_example
    from cactus.model.tokenizer import BOS_ID

    example = {"tools": [{"name": "f", "parameters": {"type": "object", "properties": {}}}],
               "query": "hello", "answers": [{"name": "f", "arguments": {}}]}
    ids, mask = _encode(tok, example, max_len=128)

    assert len(ids) == 128 and len(mask) == 128
    assert ids[0] == BOS_ID
    assert mask[0] == 0.0
    assert any(m == 1.0 for m in mask)

    _, target = render_example(example)
    assert sum(1 for m in mask if m == 1.0) == len(tok.encode(target)) + 1


def test_fit_max_len_buckets(tok, tmp_path):
    from cactus.model.finetune import fit_max_len

    path = tmp_path / "data.jsonl"
    with open(path, "w") as handle:
        handle.write(json.dumps({"tools": [], "query": "short", "answers": []}) + "\n")

    assert fit_max_len(str(path), tok, 1024) == 128
    assert fit_max_len(str(path), tok, 64) == 64


def test_load_jsonl_shapes_and_skips_invalid(tok, tmp_path):
    from cactus.model.finetune import load_jsonl

    path = tmp_path / "data.jsonl"
    rows = [
        {"tools": [], "query": "a", "answers": []},
        {"tools": [], "query": "b", "answers": []},
        {"tools": [], "reasoning": "no query here"},
    ]
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.write("\n")

    seqs, masks = load_jsonl(str(path), tok, max_len=32)
    assert seqs.shape == (2, 32)
    assert masks.shape == (2, 32)
    assert seqs.dtype == np.int32
    assert masks.dtype == np.float32
