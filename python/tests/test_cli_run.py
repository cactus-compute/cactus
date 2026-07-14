import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cactus.cli.common as common_mod
import cactus.cli.model as model_mod
import cactus.cli.run as run_mod


def test_cmd_run_forwards_chunked_bundle_flags(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "components").mkdir(parents=True)
    (bundle_dir / "components" / "manifest.json").write_text("{}", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_run = fake_bin / "run"
    fake_bin.mkdir(parents=True)
    fake_run.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(common_mod, "BIN_DIR", fake_bin)

    image_file = tmp_path / "image.png"
    audio_file = tmp_path / "audio.wav"
    result_json = tmp_path / "result.json"
    input_ids_file = tmp_path / "tokens.txt"
    image_file.write_bytes(b"image")
    audio_file.write_bytes(b"audio")
    input_ids_file.write_text("1 2 3", encoding="utf-8")

    calls = []

    def fake_subprocess_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common_mod.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(model_mod, "ensure_runnable_bundle", lambda *a, **k: bundle_dir)

    args = Namespace(
        no_cloud_tele=False,
        model_id="org/model",
        bits=4,
        token=None,
        reconvert=False,
        system=None,
        prompt="hi",
        tools=None,
        image=str(image_file),
        audio=str(audio_file),
        input_ids="1,2,3",
        input_ids_file=str(input_ids_file),
        max_new_tokens=4,
        result_json=str(result_json),
        confidence_threshold=None,
        cloud_timeout_ms=None,
        backend="auto",
        thinking=False,
        no_cloud_handoff=False,
    )

    assert run_mod.cmd_run(args) == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:2] == [str(fake_run), str(bundle_dir)]
    assert cmd[cmd.index("--prompt") + 1] == "hi"
    assert cmd[cmd.index("--image") + 1] == str(image_file)
    assert cmd[cmd.index("--audio") + 1] == str(audio_file)
    assert cmd[cmd.index("--input-ids") + 1] == "1,2,3"
    assert cmd[cmd.index("--input-ids-file") + 1] == str(input_ids_file)
    assert cmd[cmd.index("--max-new-tokens") + 1] == "4"
    assert cmd[cmd.index("--result-json") + 1] == str(result_json)
    assert "--tools" not in cmd


def test_resolve_tools_accepts_openai_format(tmp_path: Path) -> None:
    openai_style = ('[{"type": "function", "function": {"name": "get_weather", "parameters": '
                    '{"type": "object", "properties": {"location": {"type": "string"}}, '
                    '"required": ["location"]}}}]')
    resolved = run_mod.resolve_tools(openai_style)
    assert json.loads(resolved) == json.loads(openai_style)

    tools_file = tmp_path / "tools.json"
    tools_file.write_text(openai_style, encoding="utf-8")
    assert run_mod.resolve_tools(str(tools_file)) == resolved


def test_resolve_tools_rejects_improper_schema() -> None:
    for bad in (
        '{"type": "function"}',
        '[{"name": "get_weather", "parameters": {}}]',
        '[{"type": "function", "function": {"parameters": {}}}]',
        '[{"type": "function", "function": {"name": "set_timer", "parameters": '
        '{"time_human": {"type": "string", "required": true}}}}]',
    ):
        with pytest.raises(ValueError):
            run_mod.resolve_tools(bad)


def test_cmd_run_defaults_needle_bundle_to_demo_tools(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "components").mkdir(parents=True)
    (bundle_dir / "components" / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "config.txt").write_text("model_type=needle\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "run").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(common_mod, "BIN_DIR", fake_bin)

    calls = []
    monkeypatch.setattr(common_mod.subprocess, "run",
                        lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(model_mod, "ensure_runnable_bundle", lambda *a, **k: bundle_dir)

    args = Namespace(
        no_cloud_tele=False,
        model_id="Cactus-Compute/needle",
        bits=4,
        token=None,
        reconvert=False,
        system=None,
        prompt=None,
        tools=None,
        image=None,
        audio=None,
        input_ids=None,
        input_ids_file=None,
        max_new_tokens=None,
        result_json=None,
        confidence_threshold=None,
        cloud_timeout_ms=None,
        backend=None,
        thinking=False,
        no_cloud_handoff=False,
    )

    assert run_mod.cmd_run(args) == 0
    cmd = calls[0]
    tools = cmd[cmd.index("--tools") + 1]
    assert '"name":"set_timer"' in tools and '"type":"function"' in tools
