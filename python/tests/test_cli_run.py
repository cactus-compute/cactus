from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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
