from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import GREEN, RED, print_color


DEFAULT_PROFILES = [
    {
        "name": "short_chat",
        "messages": [
            {
                "role": "user",
                "content": "Summarize on-device inference in one sentence.",
            }
        ],
    },
    {
        "name": "long_prefill",
        "messages": [
            {
                "role": "user",
                "content": " ".join(
                    [
                        "Mobile inference benchmarks need stable prompt shapes, warmup runs,",
                        "and separate reporting for prefill latency and decode throughput.",
                    ]
                    * 24
                ),
            }
        ],
    },
    {
        "name": "tool_json",
        "messages": [
            {
                "role": "user",
                "content": "Return JSON with device, prompt_tokens, decode_tokens, and one bottleneck.",
            }
        ],
        "options": {"response_format": {"type": "json_object"}},
    },
]

METRIC_FIELDS = (
    "time_to_first_token_ms",
    "total_time_ms",
    "prefill_tps",
    "decode_tps",
    "ram_usage_mb",
    "prefill_tokens",
    "decode_tokens",
    "total_tokens",
)

LOWER_IS_BETTER = {"time_to_first_token_ms", "total_time_ms", "ram_usage_mb"}
DEFAULT_COMPARE_METRICS = (
    "time_to_first_token_ms",
    "total_time_ms",
    "prefill_tps",
    "decode_tps",
)
DEFAULT_COMPARE_STAT = "p50"


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    messages: list[dict[str, Any]]
    options: dict[str, Any]
    tools: list[dict[str, Any]] | None = None


def _load_profile_data(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_PROFILES)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("profiles", [])
    if not isinstance(data, list):
        raise ValueError(
            "Benchmark profile file must contain a list or a {'profiles': [...]} object"
        )
    return data


def load_profiles(
    path: str | None, names: set[str] | None, base_options: dict[str, Any]
) -> list[BenchmarkProfile]:
    profiles = []
    for item in _load_profile_data(path):
        if not isinstance(item, dict):
            raise ValueError("Each benchmark profile must be an object")
        name = str(item.get("name") or "").strip()
        messages = item.get("messages")
        if not name or not isinstance(messages, list):
            raise ValueError("Each benchmark profile needs a name and messages list")
        if names and name not in names:
            continue
        options = dict(base_options)
        options.update(item.get("options") or {})
        profiles.append(
            BenchmarkProfile(
                name=name,
                messages=messages,
                options=options,
                tools=item.get("tools"),
            )
        )
    if not profiles:
        selected = ", ".join(sorted(names or [])) or "all"
        raise ValueError(f"No benchmark profiles selected: {selected}")
    return profiles


def _parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("Sweep token counts must be positive")
        values.append(value)
    return values


def build_sweep_profiles(
    token_counts: list[int], base_options: dict[str, Any]
) -> list[BenchmarkProfile]:
    profiles = []
    seed = "Cactus local inference regression profile. "
    for count in token_counts:
        words = (seed.split() * ((count // len(seed.split())) + 1))[:count]
        profiles.append(
            BenchmarkProfile(
                name=f"context_sweep_{count}",
                messages=[
                    {
                        "role": "user",
                        "content": " ".join(words)
                        + "\n\nReturn one short sentence about the bottleneck.",
                    }
                ],
                options=dict(base_options),
            )
        )
    return profiles


def collect_environment(model_id: str, bundle_dir: Path | str | None) -> dict[str, Any]:
    try:
        from cactus import __version__ as cactus_version
    except Exception:
        cactus_version = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "cactus_version": cactus_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_results(
    results: Iterable[dict[str, Any]], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("phase") == "measure" and row.get("success"):
            grouped.setdefault(row["profile"], []).append(row)

    summaries = []
    for profile, rows in sorted(grouped.items()):
        metrics: dict[str, dict[str, float]] = {}
        for field in METRIC_FIELDS:
            values = [
                float(row[field])
                for row in rows
                if isinstance(row.get(field), (int, float))
            ]
            if values:
                metrics[field] = {
                    "mean": statistics.fmean(values),
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "min": min(values),
                    "max": max(values),
                }
        summaries.append({"profile": profile, "runs": len(rows), "metrics": metrics})
    summary = {"profiles": summaries}
    if metadata is not None:
        summary["environment"] = metadata
    return summary


def _run_once(
    runtime, model, profile: BenchmarkProfile, *, phase: str, iteration: int
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = runtime.cactus_complete(
            model, profile.messages, profile.options, profile.tools, None
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        row = {
            "profile": profile.name,
            "phase": phase,
            "iteration": iteration,
            "success": bool(result.get("success", True)),
            "wall_time_ms": wall_ms,
            "response_chars": len(str(result.get("response", ""))),
        }
        for field in METRIC_FIELDS:
            if field in result:
                row[field] = result[field]
        if result.get("error"):
            row["error"] = result["error"]
        return row
    except Exception as exc:
        return {
            "profile": profile.name,
            "phase": phase,
            "iteration": iteration,
            "success": False,
            "wall_time_ms": (time.perf_counter() - started) * 1000.0,
            "error": str(exc),
        }


def run_benchmark(
    runtime,
    model,
    profiles: list[BenchmarkProfile],
    *,
    warmup: int,
    iterations: int,
    reset_between: bool,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for profile in profiles:
        for i in range(warmup):
            if reset_between:
                runtime.cactus_reset(model)
            rows.append(_run_once(runtime, model, profile, phase="warmup", iteration=i))
        for i in range(iterations):
            if reset_between:
                runtime.cactus_reset(model)
            rows.append(
                _run_once(runtime, model, profile, phase="measure", iteration=i)
            )
    return rows, summarize_results(rows, metadata=metadata)


def _profile_metric(
    summary: dict[str, Any], profile: str, metric: str, stat: str
) -> float | None:
    for item in summary.get("profiles", []):
        if item.get("profile") != profile:
            continue
        value = item.get("metrics", {}).get(metric, {}).get(stat)
        return float(value) if isinstance(value, (int, float)) else None
    return None


def load_regression_budget(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Regression budget must be a JSON object")
    return data


def _budget_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _metric_budget(entry: Any, metric: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    value = entry.get(metric)
    return value if isinstance(value, dict) else {}


def _comparison_budget(
    budget: dict[str, Any],
    profile: str,
    metric: str,
    default_stat: str,
    default_threshold_pct: float,
) -> tuple[str, float]:
    stat = str(budget.get("stat") or default_stat)
    threshold_pct = _budget_number(budget.get("threshold_pct")) or default_threshold_pct

    metric_budget = _metric_budget(budget.get("metrics"), metric)
    stat = str(metric_budget.get("stat") or stat)
    threshold_pct = _budget_number(metric_budget.get("threshold_pct")) or threshold_pct

    profiles_budget = budget.get("profiles")
    if isinstance(profiles_budget, dict):
        profile_budget = profiles_budget.get(profile)
        if isinstance(profile_budget, dict):
            profile_metric_budget = _metric_budget(profile_budget, metric)
            stat = str(
                profile_metric_budget.get("stat") or profile_budget.get("stat") or stat
            )
            threshold_pct = (
                _budget_number(profile_metric_budget.get("threshold_pct"))
                or _budget_number(profile_budget.get("threshold_pct"))
                or threshold_pct
            )

    return stat, threshold_pct


def _regression_severity(change_pct: float, threshold_pct: float) -> str:
    over_budget = abs(change_pct) - threshold_pct
    if over_budget >= 15:
        return "critical"
    if over_budget >= 5:
        return "major"
    return "minor"


def compare_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metrics: list[str],
    stat: str,
    threshold_pct: float,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget = budget or {}
    profiles = sorted(
        {
            item.get("profile")
            for source in (baseline, candidate)
            for item in source.get("profiles", [])
            if item.get("profile")
        }
    )
    rows = []
    for profile in profiles:
        for metric in metrics:
            metric_stat, metric_threshold_pct = _comparison_budget(
                budget,
                profile,
                metric,
                stat,
                threshold_pct,
            )
            base_value = _profile_metric(baseline, profile, metric, metric_stat)
            cand_value = _profile_metric(candidate, profile, metric, metric_stat)
            if base_value is None or cand_value is None or base_value == 0:
                continue
            change_pct = ((cand_value - base_value) / abs(base_value)) * 100.0
            lower_is_better = metric in LOWER_IS_BETTER
            regression = (
                change_pct > metric_threshold_pct
                if lower_is_better
                else change_pct < -metric_threshold_pct
            )
            rows.append(
                {
                    "profile": profile,
                    "metric": metric,
                    "stat": metric_stat,
                    "baseline": base_value,
                    "candidate": cand_value,
                    "change_pct": change_pct,
                    "threshold_pct": metric_threshold_pct,
                    "lower_is_better": lower_is_better,
                    "regression": regression,
                    "severity": _regression_severity(change_pct, metric_threshold_pct)
                    if regression
                    else "ok",
                }
            )
    return {
        "threshold_pct": threshold_pct,
        "budget": budget,
        "regressions": [row for row in rows if row["regression"]],
        "comparisons": rows,
    }


def render_markdown(
    summary: dict[str, Any], comparison: dict[str, Any] | None = None
) -> str:
    lines = ["# Cactus benchmark report", ""]
    env = summary.get("environment") or {}
    if env:
        lines.extend(
            [
                "## Environment",
                "",
                "| Field | Value |",
                "| --- | --- |",
            ]
        )
        for key in (
            "created_at",
            "model_id",
            "bundle_dir",
            "cactus_version",
            "python",
            "platform",
            "machine",
            "processor",
        ):
            if env.get(key) is not None:
                lines.append(f"| {key} | {env[key]} |")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "| Profile | Runs | TTFT p50 ms | Decode p50 tok/s | Prefill p50 tok/s | RAM p50 MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary.get("profiles", []):
        metrics = item.get("metrics", {})

        def p50(name: str) -> float:
            return float(metrics.get(name, {}).get("p50", 0.0))

        lines.append(
            f"| {item.get('profile')} | {item.get('runs', 0)} | "
            f"{p50('time_to_first_token_ms'):.2f} | {p50('decode_tps'):.2f} | "
            f"{p50('prefill_tps'):.2f} | {p50('ram_usage_mb'):.2f} |"
        )
    lines.append("")

    if comparison is not None:
        lines.extend(
            [
                "## Regression comparison",
                "",
                "| Profile | Metric | Baseline | Candidate | Change | Allowed | Result |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparison.get("comparisons", []):
            result = row["severity"] if row["regression"] else "ok"
            lines.append(
                f"| {row['profile']} | {row['metric']} {row['stat']} | "
                f"{row['baseline']:.3f} | {row['candidate']:.3f} | "
                f"{row['change_pct']:.2f}% | {row['threshold_pct']:.2f}% | {result} |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: str | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_summary(path: str | None, summary: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_summary(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_benchmark(args):
    if args.compare:
        baseline = _load_summary(args.compare[0])
        candidate = _load_summary(args.compare[1])
        metrics = args.compare_metric or list(DEFAULT_COMPARE_METRICS)
        try:
            comparison = compare_summaries(
                baseline,
                candidate,
                metrics=metrics,
                stat=args.compare_stat,
                threshold_pct=args.regression_threshold,
                budget=load_regression_budget(args.budget_json),
            )
        except ValueError as exc:
            print_color(RED, str(exc))
            return 1
        report = render_markdown(candidate, comparison)
        if args.markdown_report:
            Path(args.markdown_report).write_text(report, encoding="utf-8")
        print(report)
        return 1 if args.fail_on_regression and comparison["regressions"] else 0

    from .model import TranspileOptions, ensure_bundle, resolve_bundle_dir
    from ..bindings import cactus as runtime

    base_options = {}
    if args.max_tokens is not None:
        base_options["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        base_options["temperature"] = args.temperature

    try:
        profiles = load_profiles(
            args.profiles_file, set(args.profile or []), base_options
        )
        profiles.extend(
            build_sweep_profiles(_parse_int_list(args.sweep_token_counts), base_options)
        )
    except ValueError as exc:
        print_color(RED, str(exc))
        return 1

    bundle_dir = resolve_bundle_dir(args.model_id)
    if bundle_dir is None:
        try:
            bundle_dir = ensure_bundle(
                args.model_id,
                token=args.token,
                reconvert=args.reconvert,
                transpile=TranspileOptions(max_new_tokens=args.max_tokens),
            )
        except RuntimeError as exc:
            print_color(RED, f"Model setup failed: {exc}")
            return 1

    model = None
    try:
        model = runtime.cactus_init(str(bundle_dir))
        rows, summary = run_benchmark(
            runtime,
            model,
            profiles,
            warmup=args.warmup,
            iterations=args.iterations,
            reset_between=not args.keep_cache,
            metadata=collect_environment(args.model_id, bundle_dir),
        )
    finally:
        if model is not None:
            runtime.cactus_destroy(model)

    _write_jsonl(args.output, rows)
    _write_summary(args.summary_json, summary)
    if args.markdown_report:
        Path(args.markdown_report).write_text(
            render_markdown(summary), encoding="utf-8"
        )

    print_color(
        GREEN,
        f"Benchmarked {len(profiles)} profile(s), {args.iterations} measured run(s) each",
    )
    for item in summary["profiles"]:
        decode = item["metrics"].get("decode_tps", {})
        ttft = item["metrics"].get("time_to_first_token_ms", {})
        print(
            f"{item['profile']}: ttft_p50={ttft.get('p50', 0.0):.2f}ms decode_p50={decode.get('p50', 0.0):.2f} tok/s"
        )
    return 0
