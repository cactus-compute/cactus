#!/usr/bin/env python3

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_REF = "c2f8b67cfb92bf5196497be7c8c6edc60c11c651"
DEFAULT_BASELINE_MODEL = REPO_ROOT / "weights" / "tester1.7b"
DEFAULT_CURRENT_MODEL = REPO_ROOT / "weights" / "tester1.7b-int4sign-v1"

STABLE_SCENARIOS = [
    {"name": "prefill_only_64", "kind": "prefill", "prompt_repeats": 8},
    {"name": "prefill_only_2048", "kind": "prefill", "prompt_repeats": 256},
    {"name": "small_prefill_small_decode", "kind": "complete", "prompt_repeats": 8, "max_tokens": 16},
    {"name": "large_prefill_small_decode", "kind": "complete", "prompt_repeats": 256, "max_tokens": 16},
]

LONG_DECODE_SCENARIOS = [
    {"name": "small_prefill_large_decode", "kind": "complete", "prompt_repeats": 8, "max_tokens": 256},
    {"name": "large_prefill_large_decode", "kind": "complete", "prompt_repeats": 256, "max_tokens": 256},
]

PROMPT_CHUNK = (
    "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
    "rho sigma tau upsilon phi chi psi omega. "
)

GENERATION_INSTRUCTION = (
    "Output a sequence of integers starting from 1 separated by single spaces, with no prose and no punctuation."
)


def run(cmd, *, cwd=None, capture_output=False, env=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        check=True,
        capture_output=capture_output,
    )


def ensure_worktree(repo_root: Path, ref: str) -> Path:
    worktree_root = repo_root / "build" / "rev_compare" / "worktrees"
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_dir = worktree_root / ref[:12]

    if worktree_dir.exists():
        head = run(["git", "-C", str(worktree_dir), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
        if head.startswith(ref[:12]) or head == ref:
            return worktree_dir
        run(["git", "worktree", "remove", "--force", str(worktree_dir)])

    run(["git", "worktree", "add", "--detach", "--force", str(worktree_dir), ref], cwd=repo_root)
    return worktree_dir


def build_cactus_lib(source_root: Path, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", str(source_root / "cactus"), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(build_dir), "-j8", "--target", "cactus_ffi"])
    lib_path = build_dir / "libcactus.dylib"
    if not lib_path.exists():
        raise FileNotFoundError(f"Expected library at {lib_path}")
    return lib_path


def mean_summary(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def pct_delta(current: float, baseline: float) -> float:
    return ((current / baseline) - 1.0) * 100.0


def build_messages(prompt_repeats: int) -> str:
    large_prompt = (PROMPT_CHUNK * prompt_repeats).strip()
    messages = [
        {"role": "system", "content": "You are a terse assistant. Follow the user's output instructions exactly."},
        {"role": "user", "content": large_prompt + "\n\n" + GENERATION_INSTRUCTION},
    ]
    return json.dumps(messages, separators=(",", ":"))


def options_json(max_tokens: int) -> str:
    return json.dumps(
        {
            "max_tokens": max_tokens,
            "stop_sequences": [],
            "confidence_threshold": 0.0,
            "telemetry_enabled": False,
        },
        separators=(",", ":"),
    )


def load_ffi(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
    lib.cactus_init.restype = ctypes.c_void_p
    lib.cactus_destroy.argtypes = [ctypes.c_void_p]
    lib.cactus_destroy.restype = None
    lib.cactus_reset.argtypes = [ctypes.c_void_p]
    lib.cactus_reset.restype = None
    lib.cactus_complete.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cactus_complete.restype = ctypes.c_int
    lib.cactus_prefill.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.cactus_prefill.restype = ctypes.c_int
    return lib


def run_single_revision(lib_path: Path, model_path: Path, warmup_runs: int, measured_runs: int, include_large_decode: bool):
    lib = load_ffi(lib_path)
    model = lib.cactus_init(str(model_path).encode(), None, False)
    if not model:
        raise RuntimeError(f"Failed to initialize model from {model_path}")

    scenarios = list(STABLE_SCENARIOS)
    if include_large_decode:
        scenarios.extend(LONG_DECODE_SCENARIOS)

    response_buf = ctypes.create_string_buffer(256 * 1024)
    results = []

    try:
        for scenario in scenarios:
            messages = build_messages(scenario["prompt_repeats"]).encode()
            options = options_json(scenario["max_tokens"]).encode() if scenario["kind"] == "complete" else None

            for _ in range(warmup_runs):
                lib.cactus_reset(model)
                response_buf.value = b""
                if scenario["kind"] == "prefill":
                    rc = lib.cactus_prefill(model, messages, response_buf, len(response_buf), None, None)
                else:
                    rc = lib.cactus_complete(model, messages, response_buf, len(response_buf), options, None, None, None)
                if rc <= 0:
                    raise RuntimeError(f"Warmup failed for scenario {scenario['name']}: {response_buf.value.decode(errors='ignore')}")

            run_metrics = []
            for _ in range(measured_runs):
                lib.cactus_reset(model)
                response_buf.value = b""
                if scenario["kind"] == "prefill":
                    rc = lib.cactus_prefill(model, messages, response_buf, len(response_buf), None, None)
                else:
                    rc = lib.cactus_complete(model, messages, response_buf, len(response_buf), options, None, None, None)
                raw = response_buf.value.decode()
                data = json.loads(raw)
                data["_rc"] = rc
                run_metrics.append(data)

            summary = {}
            if scenario["kind"] == "prefill":
                for key in ["prefill_tokens", "prefill_tps", "total_time_ms"]:
                    summary[key] = mean_summary([float(run[key]) for run in run_metrics])
            else:
                for key in ["prefill_tokens", "decode_tokens", "prefill_tps", "decode_tps", "time_to_first_token_ms", "total_time_ms"]:
                    summary[key] = mean_summary([float(run[key]) for run in run_metrics])

            results.append(
                {
                    "name": scenario["name"],
                    "kind": scenario["kind"],
                    "prompt_repeats": scenario["prompt_repeats"],
                    "max_tokens": scenario.get("max_tokens"),
                    "runs": run_metrics,
                    "summary": summary,
                }
            )
    finally:
        lib.cactus_destroy(model)

    return {"model_path": str(model_path), "scenarios": results}


def write_markdown(report: dict, md_path: Path):
    current = report["current"]
    baseline = report["baseline"]

    lines = [
        "## Tester1.7b Revision Comparison",
        "",
        f"Current rev: `{report['current_rev']}`",
        f"Baseline rev: `{report['baseline_rev']}`",
        f"Current model: `{report['current_model']}`",
        f"Baseline model: `{report['baseline_model']}`",
        "",
    ]

    baseline_by_name = {s["name"]: s for s in baseline["scenarios"]}
    for scenario in current["scenarios"]:
        base = baseline_by_name[scenario["name"]]
        lines.append(f"### {scenario['name']}")
        if scenario["kind"] == "prefill":
            cur_tps = scenario["summary"]["prefill_tps"]["mean"]
            old_tps = base["summary"]["prefill_tps"]["mean"]
            cur_ms = scenario["summary"]["total_time_ms"]["mean"]
            old_ms = base["summary"]["total_time_ms"]["mean"]
            lines.append(f"- prefill TPS: `{cur_tps:.2f}` vs `{old_tps:.2f}` ({pct_delta(cur_tps, old_tps):+.2f}%)")
            lines.append(f"- total time: `{cur_ms:.2f} ms` vs `{old_ms:.2f} ms` ({pct_delta(cur_ms, old_ms):+.2f}%)")
        else:
            for key, label in [
                ("prefill_tps", "prefill TPS"),
                ("decode_tps", "decode TPS"),
                ("time_to_first_token_ms", "TTFT"),
                ("total_time_ms", "total time"),
            ]:
                cur = scenario["summary"][key]["mean"]
                old = base["summary"][key]["mean"]
                suffix = " ms" if key.endswith("_ms") else ""
                lines.append(f"- {label}: `{cur:.2f}{suffix}` vs `{old:.2f}{suffix}` ({pct_delta(cur, old):+.2f}%)")
        lines.append("")

    md_path.write_text("\n".join(lines))


def run_child(args):
    result = run_single_revision(
        Path(args.lib),
        Path(args.model),
        args.warmup_runs,
        args.measured_runs,
        args.include_large_decode,
    )
    json.dump(result, sys.stdout)


def run_parent(args):
    current_rev = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True).stdout.strip()
    baseline_worktree = ensure_worktree(REPO_ROOT, args.baseline_ref)

    current_build = REPO_ROOT / "build" / "rev_compare" / "current_cactus"
    baseline_build = REPO_ROOT / "build" / "rev_compare" / f"baseline_{args.baseline_ref[:12]}_cactus"

    current_lib = build_cactus_lib(REPO_ROOT, current_build)
    baseline_lib = build_cactus_lib(baseline_worktree, baseline_build)

    child_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bench-one",
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
    ]
    if args.include_large_decode:
        child_base.append("--include-large-decode")

    current_out = run(
        child_base + ["--lib", str(current_lib), "--model", str(args.current_model)],
        cwd=REPO_ROOT,
        capture_output=True,
        env={**os.environ, "CACTUS_NO_CLOUD_TELE": "1"},
    ).stdout
    baseline_out = run(
        child_base + ["--lib", str(baseline_lib), "--model", str(args.baseline_model)],
        cwd=REPO_ROOT,
        capture_output=True,
        env={**os.environ, "CACTUS_NO_CLOUD_TELE": "1"},
    ).stdout

    report = {
        "current_rev": current_rev,
        "baseline_rev": args.baseline_ref,
        "current_model": str(args.current_model),
        "baseline_model": str(args.baseline_model),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "current": json.loads(current_out),
        "baseline": json.loads(baseline_out),
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    if args.md_out:
        write_markdown(report, Path(args.md_out))

    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tester1.7b current revision vs baseline revision.")
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--current-model", type=Path, default=DEFAULT_CURRENT_MODEL)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--include-large-decode", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)

    parser.add_argument("--bench-one", action="store_true")
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bench_one:
        if not args.lib or not args.model:
            raise SystemExit("--bench-one requires --lib and --model")
        run_child(args)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
