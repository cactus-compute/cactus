from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from cactus import cli
from cactus.cli import convert as convert_cli
from cactus.transpile.model_adapters import _cache_context_length


class _ConfigWithText:
    def get_text_config(self):
        return SimpleNamespace(max_position_embeddings=128000)


def test_cache_context_length_uses_explicit_value() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))

    assert _cache_context_length(
        model,
        input_seq_len=2048,
        cache_context_length="32768",
        fallback_extra_tokens=512,
    ) == 32768


def test_cache_context_length_reads_top_level_config() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))

    assert _cache_context_length(
        model,
        input_seq_len=2048,
        cache_context_length=None,
        fallback_extra_tokens=512,
    ) == 40960


def test_cache_context_length_reads_text_config() -> None:
    model = SimpleNamespace(config=_ConfigWithText())

    assert _cache_context_length(
        model,
        input_seq_len=2048,
        cache_context_length="auto",
        fallback_extra_tokens=512,
    ) == 128000


def test_cache_context_length_falls_back_to_capture_size() -> None:
    model = SimpleNamespace(config=SimpleNamespace())

    assert _cache_context_length(
        model,
        input_seq_len=2048,
        cache_context_length=None,
        fallback_extra_tokens=512,
    ) == 2560


def test_cache_context_length_rejects_non_positive_explicit_value() -> None:
    with pytest.raises(ValueError):
        _cache_context_length(
            SimpleNamespace(config=SimpleNamespace()),
            input_seq_len=2048,
            cache_context_length="0",
            fallback_extra_tokens=512,
        )


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
