"""Tests for `cactus run <bundle_dir>` and related transpile-time helpers.

The Python LM-runtime path (`cmd_run_transpiled` and the
`component_bundle_runtime._run_*_bundle` helpers) was removed in favour of the
C++ libcactus pipeline; the old tests that exercised those internals are gone
along with the code. The tests below keep coverage for the still-live pieces:

* `cmd_run` resolves a transpiled bundle path and shells out to the C++ `chat`
  binary with the expected arguments.
* Transpile-time helpers (`_add_multimodal_generation_headroom`,
  `_write_cactus_constant_tensor`, `load_audio_waveform`,
  `multimodal_runtime._load_image_inputs`) still behave as documented.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from cactus import cli
from cactus.cli import run as run_cli
from cactus.transpile import audio_preprocess
from cactus.transpile import hf_model
from cactus.transpile import multimodal_runtime


def test_cactus_run_resolves_bundle_and_invokes_chat_binary(monkeypatch, tmp_path: Path, capsys) -> None:
    bundle_dir = tmp_path / "bundle"
    components_dir = bundle_dir / "components"
    components_dir.mkdir(parents=True)
    (components_dir / "manifest.json").write_text(
        '{"model_id":"example/model","family":"generic","task":"causal_lm_logits","components":[]}',
        encoding="utf-8",
    )
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")
    image_file = tmp_path / "input.png"
    image_file.write_bytes(b"PNGish")

    fake_chat = tmp_path / "chat"
    fake_chat.write_text("#!/bin/sh\nexit 0\n")
    fake_chat.chmod(0o755)
    monkeypatch.setattr(run_cli, "_ensure_chat_binary", lambda: fake_chat)

    invocations: list[list[str]] = []

    class _Completed:
        returncode = 0

    def fake_run(cmd, **_kwargs):
        invocations.append(list(cmd))
        return _Completed()

    monkeypatch.setattr(run_cli.subprocess, "run", fake_run)

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "run",
            str(bundle_dir),
            "--audio",
            str(audio_file),
            "--image",
            str(image_file),
            "--prompt",
            "Hello",
            "--system",
            "Be terse.",
            "--thinking",
        ]
    )

    rc = run_cli.cmd_run(args)
    assert rc == 0
    assert len(invocations) == 1
    cmd = invocations[0]
    assert cmd[0] == str(fake_chat)
    assert cmd[1] == str(bundle_dir)
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert flags.get("--prompt") == "Hello"
    assert flags.get("--system") == "Be terse."
    assert flags.get("--image") == str(image_file.resolve())
    assert flags.get("--audio") == str(audio_file.resolve())
    assert "--thinking" in cmd
    captured = capsys.readouterr().out
    assert "Starting Cactus Chat with model:" in captured


def test_cactus_run_rejects_non_bundle_path(tmp_path: Path, capsys) -> None:
    not_a_bundle = tmp_path / "weights"
    not_a_bundle.mkdir()
    (not_a_bundle / "config.txt").write_text("placeholder", encoding="utf-8")

    parser = cli.create_parser()
    args = parser.parse_args(["run", str(not_a_bundle)])

    rc = run_cli.cmd_run(args)
    assert rc == 1
    err = capsys.readouterr().out + capsys.readouterr().err
    assert "transpiled bundle" in err.lower() or "not a transpiled bundle" in err.lower()


def test_gemma4_multimodal_headroom_uses_context_floor() -> None:
    prepared = hf_model.PreparedInputs(
        names=("input_ids", "attention_mask", "token_type_ids"),
        tensors=(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
        ),
        metadata={"input_shapes": {"input_ids": [1, 3]}},
    )

    padded = hf_model._add_multimodal_generation_headroom(
        prepared,
        tokenizer=None,
        max_new_tokens=1,
        min_context_tokens=8,
    )

    assert padded.tensors[0].shape == (1, 8)
    assert padded.metadata["target_token_count"] == 8


def test_audio_waveform_loader_caps_duration(monkeypatch, tmp_path: Path) -> None:
    sample_rate = 16000
    audio_path = tmp_path / "long.wav"
    wavfile.write(audio_path, sample_rate, np.ones(sample_rate * 2, dtype=np.float32))

    monkeypatch.setenv("CACTUS_TRANSPILER_MAX_AUDIO_SECONDS", "0.5")
    waveform = audio_preprocess.load_audio_waveform(
        audio_path,
        target_sample_rate=sample_rate,
    )

    assert waveform.shape == (sample_rate // 2,)


def test_materialized_transpile_constants_round_trip_header(tmp_path: Path) -> None:
    """Verify `_write_cactus_constant_tensor` emits the documented binary layout.

    The Python-side reader (`_open_cactus_tensor_file`) was removed along with
    the rest of the Python LM-runtime; the C++ side now mmaps these files via
    `cactus_graph_bind_mmap_weights`. This test still pins the on-disk format
    so the C++ loader doesn't silently break.
    """
    from cactus.convert.cactus_adapters.tensor_io import CACTUS_MAGIC

    tensor_path = tmp_path / "constant.weights"
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)

    hf_model._write_cactus_constant_tensor(
        output_path=tensor_path,
        value=expected,
        precision=int(hf_model.Graph.FP16),
    )

    blob = tensor_path.read_bytes()
    assert blob.startswith(CACTUS_MAGIC)
    cursor = len(CACTUS_MAGIC)
    (flags,) = struct.unpack_from("<I", blob, cursor); cursor += 4
    (_alignment,) = struct.unpack_from("<I", blob, cursor); cursor += 4
    (rank,) = struct.unpack_from("<I", blob, cursor); cursor += 4
    shape = list(struct.unpack_from("<QQQQ", blob, cursor))
    cursor += 32
    (precision,) = struct.unpack_from("<I", blob, cursor); cursor += 4
    (data_bytes,) = struct.unpack_from("<Q", blob, cursor); cursor += 8

    assert flags == 0
    assert rank == 2
    assert shape[:2] == [2, 3]
    assert precision == int(hf_model.Graph.FP16)
    assert data_bytes == expected.nbytes
    payload = np.frombuffer(blob[-expected.nbytes:], dtype=np.float16).reshape(2, 3)
    np.testing.assert_array_equal(payload, expected)


def test_runtime_image_inputs_resize_to_static_square(tmp_path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    image_path = tmp_path / "tall.png"
    Image.new("RGB", (20, 40), color=(255, 0, 0)).save(image_path)

    images = multimodal_runtime._load_image_inputs((str(image_path),))
    assert images[0].size == (256, 256)
