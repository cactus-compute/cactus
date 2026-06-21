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


def test_cmd_convert_builds_graph_after_weights(monkeypatch, tmp_path: Path) -> None:
    """`cactus convert` quantizes weights and then builds the runtime graph in
    one step; the graph build is pointed at the weights just written."""
    parser = cli.create_parser()
    out = tmp_path / "out"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(out), "--reconvert"])

    weight_calls: list[dict] = []

    def _fake_ensure_weights(model_id, **kw):
        weight_calls.append(kw)
        Path(kw["output_dir"]).mkdir(parents=True, exist_ok=True)
        return kw["output_dir"]

    transpile_calls: list[dict] = []

    def _fake_run_transpile(model_id, **kw):
        transpile_calls.append(kw)
        return 0

    import cactus.cli.model as model_mod
    import cactus.cli.transpile as transpile_mod
    monkeypatch.setattr(model_mod, "ensure_weights", _fake_ensure_weights)
    monkeypatch.setattr(model_mod, "_default_multimodal_assets", lambda: ([], None))
    monkeypatch.setattr(transpile_mod, "run_transpile", _fake_run_transpile)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert len(weight_calls) == 1
    assert len(transpile_calls) == 1
    extra_args = transpile_calls[0]["extra_args"]
    assert "--weights-dir" in extra_args
    assert str(out) in extra_args


def test_cmd_convert_weights_only_skips_graph(monkeypatch, tmp_path: Path) -> None:
    """`--weights-only` stops after CQ quantization, before the runtime graph."""
    parser = cli.create_parser()
    out = tmp_path / "out"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(out), "--weights-only"])

    def _fake_ensure_weights(model_id, **kw):
        Path(kw["output_dir"]).mkdir(parents=True, exist_ok=True)
        return kw["output_dir"]

    def _fail_if_called(*a, **kw):
        raise AssertionError("--weights-only must skip the runtime graph build")

    import cactus.cli.model as model_mod
    import cactus.cli.transpile as transpile_mod
    monkeypatch.setattr(model_mod, "ensure_weights", _fake_ensure_weights)
    monkeypatch.setattr(transpile_mod, "run_transpile", _fail_if_called)

    rc = convert_cli.cmd_convert(args)
    assert rc == 0


def test_cli_convert_absorbs_graph_flags() -> None:
    """The former `transpile` options now live on `cactus convert`; the separate
    `transpile` subcommand no longer exists."""
    parser = cli.create_parser()
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it",
                              "--weights-dir", "/tmp/x", "--task", "causal_lm_logits",
                              "--dynamic-batch", "--max-slots", "8"])
    assert args.command == "convert"
    assert args.model_id == "google/gemma-4-E2B-it"
    assert args.weights_dir == "/tmp/x"
    assert args.task == "causal_lm_logits"
    assert args.dynamic_batch is True
    assert args.max_slots == 8

    with pytest.raises(SystemExit):
        parser.parse_args(["transpile", "google/gemma-4-E2B-it"])


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
