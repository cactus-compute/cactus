from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import cactus.cli.run as run_mod


def test_cmd_run_forwards_chunked_bundle_flags(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "components").mkdir(parents=True)
    (bundle_dir / "components" / "manifest.json").write_text("{}", encoding="utf-8")

    fake_pkg = tmp_path / "pkg"
    fake_chat = fake_pkg / "bin" / "chat"
    fake_chat.parent.mkdir(parents=True)
    fake_chat.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(run_mod, "__file__", str(fake_pkg / "cli" / "run.py"))

    image_file = tmp_path / "image.png"
    audio_file = tmp_path / "audio.wav"
    result_json = tmp_path / "result.json"
    image_file.write_bytes(b"image")
    audio_file.write_bytes(b"audio")

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)

    args = Namespace(
        no_cloud_tele=False,
        model_id=str(bundle_dir),
        token=None,
        cache_dir=None,
        reconvert=False,
        system=None,
        prompt="hi",
        image=None,
        image_file=[str(image_file)],
        audio=None,
        audio_file=str(audio_file),
        input_ids="1,2,3",
        max_new_tokens=4,
        result_json=str(result_json),
        thinking=False,
    )

    assert run_mod.cmd_run(args) == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:2] == [str(fake_chat), str(bundle_dir)]
    assert cmd[cmd.index("--prompt") + 1] == "hi"
    assert cmd[cmd.index("--image") + 1] == str(image_file.resolve())
    assert cmd[cmd.index("--audio") + 1] == str(audio_file.resolve())
    assert cmd[cmd.index("--input-ids") + 1] == "1,2,3"
    assert cmd[cmd.index("--max-new-tokens") + 1] == "4"
    assert cmd[cmd.index("--result-json") + 1] == str(result_json)
