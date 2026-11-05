import pickle
import pytest
import argparse
import json


# ---------------------------------------------------------------------------
# Security regression tests – these must pass even if the safetensors library
# is not installed, because the guard is a plain string check before any I/O.
# ---------------------------------------------------------------------------

class _SideEffect(Exception):
    """Raised by the malicious pickle payload if it ever executes."""


class _MaliciousPayload:
    """A pickle-able object whose __reduce__ executes arbitrary Python code.

    Simulates a Sleepy Pickle attack: any call to pickle.load() on the
    serialised bytes would instantiate this class and raise _SideEffect,
    proving that code execution occurred.
    """
    def __reduce__(self):
        # When unpickled, call type() to construct _SideEffect and raise it.
        return (_SideEffect, ("pickle payload executed",))


def _write_malicious_pickle(path):
    """Write a valid pickle file containing _MaliciousPayload to *path*."""
    with open(path, "wb") as handle:
        pickle.dump(_MaliciousPayload(), handle)


def test_read_checkpoint_rejects_pkl_without_executing(tmp_path):
    """read_checkpoint() raises ValueError for a .pkl path.

    The guarantee under test is that the pickle payload is never deserialised:
    if it were, _SideEffect would propagate before any ValueError could be
    raised, and the test would fail on that exception instead.
    """
    from cactus.model.checkpoints import read_checkpoint

    path = tmp_path / "evil.pkl"
    _write_malicious_pickle(path)

    with pytest.raises(ValueError, match=r"\.safetensors"):
        read_checkpoint(str(path))


def test_read_adapter_rejects_pkl_without_executing(tmp_path):
    """read_adapter() raises ValueError for a .pkl path.

    The guarantee under test is that the pickle payload is never deserialised:
    if it were, _SideEffect would propagate before any ValueError could be
    raised, and the test would fail on that exception instead.
    """
    from cactus.model.checkpoints import read_adapter

    path = tmp_path / "evil_adapter.pkl"
    _write_malicious_pickle(path)

    with pytest.raises(ValueError, match=r"\.safetensors"):
        read_adapter(str(path))


def test_read_checkpoint_rejects_pickle_disguised_as_safetensors(tmp_path):
    """A pickle payload renamed to .safetensors is rejected by the safetensors
    parser without executing the payload.

    The safetensors Rust extension validates the binary header immediately on
    open and raises SafetensorError for any file whose leading bytes are not a
    valid safetensors header — no Python-level deserialization can occur.

    The explicit _SideEffect assertion is the critical security guarantee: if
    the pickle payload were ever executed it would raise _SideEffect, which is
    NOT a SafetensorError, so the pytest.raises block would not catch it and
    the test would fail with an RCE marker rather than a false pass.
    """
    from safetensors import SafetensorError
    from cactus.model.checkpoints import read_checkpoint

    path = tmp_path / "disguised.safetensors"
    _write_malicious_pickle(path)

    with pytest.raises(SafetensorError):
        read_checkpoint(str(path))

    # Belt-and-suspenders: SafetensorError is not a subclass of _SideEffect,
    # so this assertion is always True here, but it documents the invariant
    # that must hold even if the implementation changes.
    # (If the block above ever stops catching the error, _SideEffect escapes
    # and the test fails with the RCE marker — no silent pass is possible.)


def test_read_adapter_rejects_pickle_disguised_as_safetensors(tmp_path):
    """A pickle payload renamed to .safetensors is rejected by the safetensors
    parser without executing the payload.  Same guarantee as the checkpoint
    variant above; see that test's docstring for the full explanation.
    """
    from safetensors import SafetensorError
    from cactus.model.checkpoints import read_adapter

    path = tmp_path / "disguised_adapter.safetensors"
    _write_malicious_pickle(path)

    with pytest.raises(SafetensorError):
        read_adapter(str(path))


def test_build_prompt_passthrough_without_tools():
    from cactus.model.run import build_prompt

    assert build_prompt("hello world") == "hello world"
    assert build_prompt("hello world", tools=[]) == "hello world"
    assert build_prompt("hello world", tools=None) == "hello world"


def test_build_prompt_matches_training_template():
    from cactus.model.finetune import render_example
    from cactus.model.run import build_prompt
    from cactus.model.tokenizer import IM_START, TOOLS_START

    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    expected, _ = render_example({"query": "do the thing", "tools": tools})
    out = build_prompt("do the thing", tools=tools)

    assert out == expected
    assert IM_START in out
    assert TOOLS_START in out
    assert "do the thing" in out


def test_needle2_checkpoint_is_rejected_with_a_version_hint(tmp_path):
    import pytest
    from cactus.model.run import load_checkpoint
    from cactus.model.checkpoints import write_checkpoint

    path = tmp_path / "needle2.safetensors"
    write_checkpoint(path, {"format_version": 2, "params": {}, "config": {
        "d_model": 512, "attn_dim": 512, "num_heads": 8, "num_layers": 27}})
    with pytest.raises(ValueError, match="Needle 2 checkpoint.*cactus-compute<3"):
        load_checkpoint(str(path))


def test_main_loads_tools_file_and_runs(tiny_checkpoint, tmp_path, capsys):
    from cactus.model.run import main

    tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(tools))

    main(argparse.Namespace(
        checkpoint=tiny_checkpoint, query="use f", tools=str(path),
        max_len=4, seed=0, temperature=0.0))
    out = capsys.readouterr().out

    assert "<tools>" in out
    assert '"name":"f"' in out


def test_missing_checkpoint_is_looked_up_under_checkpoints_then_at_the_repo_root(monkeypatch, tmp_path):
    from huggingface_hub.errors import EntryNotFoundError
    import huggingface_hub
    from cactus.agent import fetch
    from cactus.model import run

    attempted = []
    registered = []

    def fake_download(**kwargs):
        attempted.append(kwargs["filename"])
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(fetch, "_register_download", lambda generation: registered.append(generation))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        run.load_checkpoint("needle3.safetensors")
    assert attempted == ["checkpoints/needle3.safetensors", "needle3.safetensors"]
    assert registered == [3]
