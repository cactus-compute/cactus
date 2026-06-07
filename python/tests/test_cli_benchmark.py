import json
import sys
from argparse import Namespace

import cactus.cli.benchmark as benchmark


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def cactus_complete(self, model, messages, options, tools, callback):
        self.calls.append((model, messages, options, tools, callback))
        idx = len(self.calls)
        return {
            "success": True,
            "response": "ok",
            "time_to_first_token_ms": 10.0 + idx,
            "total_time_ms": 20.0 + idx,
            "prefill_tps": 100.0,
            "decode_tps": 40.0 + idx,
            "prefill_tokens": 8,
            "decode_tokens": 4,
            "total_tokens": 12,
        }

    def cactus_reset(self, model):
        self.resets += 1


def test_load_profiles_filters_and_applies_base_options(tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "profiles": [
            {"name": "short", "messages": [{"role": "user", "content": "hi"}]},
            {"name": "long", "messages": [{"role": "user", "content": "hello"}], "options": {"temperature": 0.2}},
        ]
    }), encoding="utf-8")

    selected = benchmark.load_profiles(str(profiles), {"long"}, {"max_tokens": 32, "temperature": 0.0})

    assert [p.name for p in selected] == ["long"]
    assert selected[0].options == {"max_tokens": 32, "temperature": 0.2}


def test_run_benchmark_records_json_ready_rows_and_summary():
    runtime = FakeRuntime()
    profiles = [benchmark.BenchmarkProfile(
        name="short",
        messages=[{"role": "user", "content": "hi"}],
        options={"max_tokens": 8},
    )]

    rows, summary = benchmark.run_benchmark(
        runtime,
        "model",
        profiles,
        warmup=1,
        iterations=3,
        reset_between=True,
        metadata={"model_id": "fake"},
    )

    assert runtime.resets == 4
    assert [row["phase"] for row in rows] == ["warmup", "measure", "measure", "measure"]
    assert rows[1]["decode_tps"] == 42.0
    assert summary["profiles"][0]["profile"] == "short"
    assert summary["environment"]["model_id"] == "fake"
    assert summary["profiles"][0]["runs"] == 3
    assert summary["profiles"][0]["metrics"]["decode_tps"]["p50"] == 43.0


def test_build_sweep_profiles_creates_named_context_profiles():
    profiles = benchmark.build_sweep_profiles([8, 16], {"max_tokens": 4})

    assert [profile.name for profile in profiles] == ["context_sweep_8", "context_sweep_16"]
    assert profiles[0].options == {"max_tokens": 4}
    assert "Return one short sentence" in profiles[0].messages[0]["content"]


def test_compare_summaries_flags_latency_and_throughput_regressions():
    baseline = {
        "profiles": [{
            "profile": "short",
            "runs": 2,
            "metrics": {
                "time_to_first_token_ms": {"p50": 100.0},
                "decode_tps": {"p50": 50.0},
            },
        }]
    }
    candidate = {
        "profiles": [{
            "profile": "short",
            "runs": 2,
            "metrics": {
                "time_to_first_token_ms": {"p50": 112.0},
                "decode_tps": {"p50": 44.0},
            },
        }]
    }

    comparison = benchmark.compare_summaries(
        baseline,
        candidate,
        metrics=["time_to_first_token_ms", "decode_tps"],
        stat="p50",
        threshold_pct=5.0,
    )

    assert len(comparison["regressions"]) == 2
    assert {row["metric"] for row in comparison["regressions"]} == {"time_to_first_token_ms", "decode_tps"}


def test_render_markdown_includes_environment_and_comparison():
    summary = {
        "environment": {"model_id": "m", "python": "3.11"},
        "profiles": [{
            "profile": "short",
            "runs": 1,
            "metrics": {
                "time_to_first_token_ms": {"p50": 1.0},
                "decode_tps": {"p50": 2.0},
                "prefill_tps": {"p50": 3.0},
                "ram_usage_mb": {"p50": 4.0},
            },
        }],
    }
    comparison = {
        "comparisons": [{
            "profile": "short",
            "metric": "decode_tps",
            "stat": "p50",
            "baseline": 3.0,
            "candidate": 2.0,
            "change_pct": -33.3,
            "regression": True,
        }]
    }

    report = benchmark.render_markdown(summary, comparison)

    assert "# Cactus benchmark report" in report
    assert "| model_id | m |" in report
    assert "regression" in report


def test_cmd_benchmark_uses_bundle_and_writes_outputs(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = tmp_path / "bench.jsonl"
    summary = tmp_path / "summary.json"
    fake_runtime = FakeRuntime()
    fake_runtime.cactus_init = lambda path: "model"
    fake_runtime.cactus_destroy = lambda model: None

    monkeypatch.setattr(benchmark, "print_color", lambda *args, **kwargs: None)
    monkeypatch.setattr("cactus.cli.model.resolve_bundle_dir", lambda model_id: bundle)
    monkeypatch.setattr("cactus.cli.model.ensure_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setitem(sys.modules, "cactus.bindings.cactus", fake_runtime)

    args = Namespace(
        model_id="bundle",
        profiles_file=None,
        profile=["short_chat"],
        max_tokens=16,
        temperature=0.0,
        token=None,
        reconvert=False,
        warmup=0,
        iterations=2,
        keep_cache=False,
        output=str(output),
        summary_json=str(summary),
        markdown_report=None,
        sweep_token_counts=None,
        compare=None,
        compare_metric=None,
        compare_stat="p50",
        regression_threshold=5.0,
        fail_on_regression=False,
    )

    assert benchmark.cmd_benchmark(args) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["profiles"][0]["runs"] == 2


def test_parser_accepts_benchmark_command():
    from cactus.cli import create_parser

    args = create_parser().parse_args([
        "benchmark",
        "LiquidAI/LFM2-1.2B",
        "--profile",
        "short_chat",
        "--iterations",
        "5",
        "--output",
        "bench.jsonl",
        "--sweep-token-counts",
        "512,2048",
        "--markdown-report",
        "report.md",
    ])

    assert args.command == "benchmark"
    assert args.profile == ["short_chat"]
    assert args.iterations == 5
    assert args.sweep_token_counts == "512,2048"
    assert args.markdown_report == "report.md"


def test_parser_accepts_benchmark_compare_mode():
    from cactus.cli import create_parser

    args = create_parser().parse_args([
        "benchmark",
        "--compare",
        "base.json",
        "candidate.json",
        "--compare-metric",
        "decode_tps",
        "--fail-on-regression",
    ])

    assert args.command == "benchmark"
    assert args.compare == ["base.json", "candidate.json"]
    assert args.compare_metric == ["decode_tps"]
    assert args.fail_on_regression is True
