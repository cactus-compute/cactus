import sys
import types
from pathlib import Path

import pytest

from cactus import cli
from cactus.cli import convert as convert_cli
from cactus.cli import model as model_cli


def test_convert_parser_accepts_calibration_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "calibration.json"

    args = cli.create_parser().parse_args([
        "convert",
        "org/model",
        "--calibration-manifest",
        str(manifest),
    ])

    assert args.calibration_manifest == str(manifest)


def test_convert_command_forwards_calibration_manifest(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "calibration.json"
    output = tmp_path / "weights"
    args = cli.create_parser().parse_args([
        "convert",
        "org/model",
        str(output),
        "--weights-only",
        "--calibration-manifest",
        str(manifest),
    ])

    weight_calls: list[dict] = []

    def fake_ensure_weights(model_id, **kwargs):
        weight_calls.append({"model_id": model_id, **kwargs})
        return output

    fake_transpiler = types.ModuleType("cactus.cli.transpiler")
    fake_transpiler.build_transpiled_bundle = lambda *args, **kwargs: output
    fake_transpiler.parse_modalities = lambda value: value
    fake_transpiler.resolve_transpile_config = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "cactus.cli.transpiler", fake_transpiler)
    monkeypatch.setattr(model_cli, "ensure_weights", fake_ensure_weights)
    monkeypatch.setattr(model_cli, "package_handoff_probe", lambda *args: None)

    assert convert_cli.cmd_convert(args) == 0
    assert weight_calls == [{
        "model_id": "org/model",
        "bits": 4,
        "token": None,
        "reconvert": False,
        "output_dir": str(output),
        "skip_model_load": False,
        "calibration_manifest": str(manifest),
    }]


def test_convert_from_source_forwards_calibration_manifest(monkeypatch, tmp_path: Path) -> None:
    from cactus.convert import cli as cq_cli

    calls: list[list[str]] = []
    monkeypatch.setattr("cactus.cli.common.convert_toolchain_error", lambda: None)
    monkeypatch.setattr(cq_cli, "main", lambda args: calls.append(args))

    manifest = tmp_path / "calibration.json"
    output = tmp_path / "weights"
    model_cli._convert_from_source(
        "org/model",
        bits=4,
        token=None,
        weights_dir=output,
        calibration_manifest=manifest,
    )

    assert calls == [[
        "convert",
        "--model",
        "org/model",
        "--out",
        str(output),
        "--bits",
        "4",
        "--calibration-manifest",
        str(manifest),
    ]]


def test_cached_bundle_rejects_ignored_calibration_manifest(tmp_path: Path) -> None:
    output = tmp_path / "weights"
    output.mkdir()
    (output / "config.txt").write_text("model_type=test\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="--reconvert"):
        model_cli.ensure_weights(
            "org/model",
            output_dir=output,
            calibration_manifest=tmp_path / "calibration.json",
        )


def test_cached_bundle_is_reused_without_calibration_manifest(tmp_path: Path) -> None:
    output = tmp_path / "weights"
    output.mkdir()
    (output / "config.txt").write_text("model_type=test\n", encoding="utf-8")

    assert model_cli.ensure_weights("org/model", output_dir=output) == output
