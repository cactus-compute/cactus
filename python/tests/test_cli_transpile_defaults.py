from __future__ import annotations

from pathlib import Path

from cactus import cli
from cactus.cli import convert as convert_cli


def test_cmd_convert_does_not_transpile(monkeypatch, tmp_path: Path) -> None:
    """`cactus convert` does CQ quantization only; the runtime graph is built
    by `cactus transpile` per docs/cactus_transpiler.md."""
    parser = cli.create_parser()
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(tmp_path / "out"), "--reconvert"])

    cq_calls: list[list[str]] = []

    def _fake_cq_main(command):
        cq_calls.append(list(command))
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.txt").write_text("model_type=gemma4\n", encoding="utf-8")
        return 0

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

    # cmd_convert should call ensure_weights, never ensure_bundle (the latter
    # transpiles as well). Patch ensure_bundle to fail loudly if invoked.
    import cactus.cli.model as model_mod

    def _fail_if_called(*a, **kw):
        raise AssertionError("cmd_convert must not invoke ensure_bundle (would transpile)")

    monkeypatch.setattr(model_mod, "ensure_bundle", _fail_if_called)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert len(cq_calls) == 1
    assert "--out" in cq_calls[0]


def test_cli_registers_transpile_command() -> None:
    """`cactus transpile` is a first-class subcommand per docs/cactus_transpiler.md."""
    parser = cli.create_parser()
    args = parser.parse_args(["transpile", "google/gemma-4-E2B-it",
                              "--weights-dir", "/tmp/x", "--task", "causal_lm_logits"])
    assert args.command == "transpile"
    assert args.model_id == "google/gemma-4-E2B-it"
    assert args.weights_dir == "/tmp/x"
    assert args.task == "causal_lm_logits"


def test_run_accepts_local_bundle_path() -> None:
    """`cactus run` accepts a HF id (org/model) OR a local path. Bare names
    like 'whisper-base' (no slash) are rejected."""
    parser = cli.create_parser()
    args = parser.parse_args(["run", "/tmp/bundle",
                              "--prompt", "hi", "--system", "be brief", "--thinking"])
    assert args.command == "run"
    assert args.model_id == "/tmp/bundle"
    assert args.prompt == "hi"
    assert args.system == "be brief"
    assert args.thinking is True
