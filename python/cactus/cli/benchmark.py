from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import GREEN, RED, print_color


DEFAULT_PROFILES = [
    {
        "name": "short_chat",
        "messages": [{"role": "user", "content": "Summarize on-device inference in one sentence."}],
    },
    {
        "name": "long_prefill",
        "messages": [
            {
                "role": "user",
                "content": " ".join([
                    "Mobile inference benchmarks need stable prompt shapes, warmup runs,",
                    "and separate reporting for prefill latency and decode throughput.",
                ] * 24),
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
        raise ValueError("Benchmark profile file must contain a list or a {'profiles': [...]} object")
    return data


def load_profiles(path: str | None, names: set[str] | None, base_options: dict[str, Any]) -> list[BenchmarkProfile]:
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
        profiles.append(BenchmarkProfile(
            name=name,
            messages=messages,
            options=options,
            tools=item.get("tools"),
        ))
    if not profiles:
        selected = ", ".join(sorted(names or [])) or "all"
        raise ValueError(f"No benchmark profiles selected: {selected}")
    return profiles


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


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("phase") == "measure" and row.get("success"):
            grouped.setdefault(row["profile"], []).append(row)

    summaries = []
    for profile, rows in sorted(grouped.items()):
        metrics: dict[str, dict[str, float]] = {}
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
            if values:
                metrics[field] = {
                    "mean": statistics.fmean(values),
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "min": min(values),
                    "max": max(values),
                }
        summaries.append({"profile": profile, "runs": len(rows), "metrics": metrics})
    return {"profiles": summaries}


def _run_once(runtime, model, profile: BenchmarkProfile, *, phase: str, iteration: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = runtime.cactus_complete(model, profile.messages, profile.options, profile.tools, None)
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


def run_benchmark(runtime, model, profiles: list[BenchmarkProfile], *, warmup: int, iterations: int,
                  reset_between: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for profile in profiles:
        for i in range(warmup):
            if reset_between:
                runtime.cactus_reset(model)
            rows.append(_run_once(runtime, model, profile, phase="warmup", iteration=i))
        for i in range(iterations):
            if reset_between:
                runtime.cactus_reset(model)
            rows.append(_run_once(runtime, model, profile, phase="measure", iteration=i))
    return rows, summarize_results(rows)


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
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_benchmark(args):
    from .model import TranspileOptions, ensure_bundle, resolve_bundle_dir
    from ..bindings import cactus as runtime

    base_options = {}
    if args.max_tokens is not None:
        base_options["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        base_options["temperature"] = args.temperature

    try:
        profiles = load_profiles(args.profiles_file, set(args.profile or []), base_options)
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
        )
    finally:
        if model is not None:
            runtime.cactus_destroy(model)

    _write_jsonl(args.output, rows)
    _write_summary(args.summary_json, summary)

    print_color(GREEN, f"Benchmarked {len(profiles)} profile(s), {args.iterations} measured run(s) each")
    for item in summary["profiles"]:
        decode = item["metrics"].get("decode_tps", {})
        ttft = item["metrics"].get("time_to_first_token_ms", {})
        print(f"{item['profile']}: ttft_p50={ttft.get('p50', 0.0):.2f}ms decode_p50={decode.get('p50', 0.0):.2f} tok/s")
    return 0
