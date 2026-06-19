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


def test_cmd_convert_builds_full_bundle(monkeypatch, tmp_path: Path) -> None:
    """`cactus convert` builds a runnable bundle locally (convert + transpile)
    and skips the prebuilt-bundle fetch."""
    parser = cli.create_parser()
    out_dir = tmp_path / "out"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(out_dir)])

    import cactus.cli.model as model_mod
    import cactus.cli.download as download_mod

    built: dict[str, object] = {}

    def _fake_ensure_bundle(model_id, **kwargs):
        built["model_id"] = model_id
        built["output_dir"] = kwargs.get("output_dir")
        return kwargs.get("output_dir")

    monkeypatch.setattr(model_mod, "ensure_bundle", _fake_ensure_bundle)

    def _fail_download(*a, **kw):
        raise AssertionError("cactus convert must not fetch a prebuilt bundle")

    monkeypatch.setattr(download_mod, "download_bundle", _fail_download)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert built["model_id"] == "google/gemma-4-E2B-it"
    assert Path(str(built["output_dir"])).resolve() == out_dir.resolve()


def test_ensure_bundle_resolves_relative_output_dir(monkeypatch, tmp_path: Path) -> None:
    """A relative output_dir must reach the transpile subprocess (which runs with
    cwd=PROJECT_ROOT) as an absolute path resolved against the caller's cwd —
    otherwise the in-process CQ-weights step and the transpiler disagree on where
    the weights live, and the build fails with 'weights_dir does not exist'."""
    import cactus.cli.model as model_mod
    from cactus.cli import transpile as transpile_mod
    import cactus.transpile.component_plan as cp

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    def fake_ensure_weights(model_id, **kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        captured["weights_out"] = kwargs["output_dir"]
        return out

    def fake_run_transpile(model_id, *, extra_args, **kwargs):
        captured["extra"] = list(extra_args)
        return 0

    plan = SimpleNamespace(task="causal_lm_logits", components=(), default_max_new_tokens=8,
                           needs_image=False, needs_audio=False, force_component_pipeline=False)

    monkeypatch.setattr(model_mod, "ensure_weights", fake_ensure_weights)
    monkeypatch.setattr(model_mod, "_has_transpiled_bundle", lambda p: False)
    monkeypatch.setattr(transpile_mod, "run_transpile", fake_run_transpile)
    monkeypatch.setattr(cp, "infer_component_plan_from_output", lambda *a, **k: plan)

    model_mod.ensure_bundle("some/model", output_dir="rel-out")

    weights_out = Path(str(captured["weights_out"]))
    assert weights_out.is_absolute()
    assert weights_out == (tmp_path / "rel-out").resolve()
    wd = captured["extra"][captured["extra"].index("--weights-dir") + 1]
    assert Path(wd).is_absolute()


def test_cli_registers_transpile_command() -> None:
    """`cactus transpile` stays a registered subcommand (hidden from the CLI menu)."""
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
