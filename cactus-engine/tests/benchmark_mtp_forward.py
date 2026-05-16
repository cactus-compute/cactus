#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target-forward-only MTP timing benchmark from chat JSON output."
    )
    parser.add_argument("model", help="Transpiled MTP model bundle path")
    parser.add_argument("--chat", default="cactus-engine/tests/build/chat")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--profile-check",
        action="store_true",
        help="Run one profiled MTP sample per M and report whether the target verifier graph uses that small M.",
    )
    parser.add_argument(
        "--require-small-m",
        action="store_true",
        help="Fail if --profile-check does not verify small-M target matmul rows for every M.",
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def cmake_build_type(chat_path: str) -> str:
    cache = Path(chat_path).resolve().parent / "CMakeCache.txt"
    if not cache.exists():
        return "unknown"
    for line in cache.read_text().splitlines():
        if line.startswith("CMAKE_BUILD_TYPE:STRING="):
            return line.split("=", 1)[1] or "unknown"
    return "unknown"


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def thread_environment() -> str:
    entries = [
        f"{name}={value}"
        for name, value in sorted(os.environ.items())
        if (
            name != "CODEX_THREAD_ID"
            and ("THREAD" in name or "AFFINITY" in name or "OMP_" in name)
        )
    ]
    return ", ".join(entries) if entries else "none"


def run_chat(args: argparse.Namespace, *, draft_tokens: int | None) -> dict:
    return run_chat_command(args, draft_tokens=draft_tokens, max_tokens=args.max_tokens)


def run_chat_command(
    args: argparse.Namespace,
    *,
    draft_tokens: int | None,
    max_tokens: int,
    profile_path: Path | None = None,
) -> dict:
    cmd = [
        args.chat,
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--confidence-threshold",
        "-1.0",
        "--json",
    ]
    if draft_tokens is not None:
        cmd += ["--mtp", "--mtp-draft-tokens", str(draft_tokens)]

    env = None
    if profile_path is not None:
        env = dict(os.environ)
        env["CACTUS_PROFILE"] = str(profile_path)

    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "chat failed with exit code "
            f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    for line in reversed(proc.stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            row = json.loads(text)
            if row.get("success") is False:
                raise RuntimeError(f"chat JSON returned error: {row.get('error')}")
            return row
    raise RuntimeError(f"chat output did not include JSON\nstdout:\n{proc.stdout}")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _profile_sections(profile_text: str) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in profile_text.splitlines():
        if line.startswith("=== Graph Execution Profile ==="):
            if current:
                sections.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current:
        sections.append(current)
    return sections


def _matmul_row_dims(section: list[str]) -> list[int]:
    rows: list[int] = []
    for line in section:
        if not line.startswith("MATMUL"):
            continue
        start = line.find("[")
        end = line.find("]", start + 1)
        if start < 0 or end < 0:
            continue
        dims = [part.strip() for part in line[start + 1 : end].split(",")]
        if dims:
            try:
                rows.append(int(dims[0]))
            except ValueError:
                pass
    return rows


def profile_small_m(args: argparse.Namespace, draft_tokens: int) -> dict[str, object]:
    expected_m = draft_tokens + 1
    with tempfile.NamedTemporaryFile(prefix="cactus_mtp_profile_", suffix=".txt", delete=False) as handle:
        profile_path = Path(handle.name)
    try:
        row = run_chat_command(
            args,
            draft_tokens=draft_tokens,
            max_tokens=expected_m,
            profile_path=profile_path,
        )
        sections = _profile_sections(profile_path.read_text())
    finally:
        try:
            profile_path.unlink()
        except FileNotFoundError:
            pass

    verifier_rows: list[int] = []
    for section in sections:
        verifier_rows.extend(_matmul_row_dims(section))
    return {
        "M": expected_m,
        "reported_verifier_width": int(row.get("mtp_verifier_width", 0)),
        "graph_executions": len(sections),
        "verifier_matmul_rows": sorted(set(verifier_rows)),
        "small_m_verified": expected_m in verifier_rows,
    }


def run_mode(args: argparse.Namespace, label: str, draft_tokens: int | None) -> dict:
    for _ in range(args.warmup):
        run_chat(args, draft_tokens=draft_tokens)

    rows = [run_chat(args, draft_tokens=draft_tokens) for _ in range(args.repeats)]
    target = [float(row.get("avg_target_forward_ms_per_token", math.nan)) for row in rows]
    if any(math.isnan(value) for value in target):
        raise RuntimeError("chat JSON is missing avg_target_forward_ms_per_token")
    target_context = [
        float(row.get("avg_target_context_forward_ms_per_token", 0.0))
        for row in rows
    ]
    assistant = [float(row.get("avg_assistant_forward_ms_per_token", 0.0)) for row in rows]
    misc = [float(row.get("avg_misc_completion_ms_per_token", math.nan)) for row in rows]
    ram = [float(row.get("ram_usage_mb", math.nan)) for row in rows]
    prompt_tokens = [float(row.get("prefill_tokens", math.nan)) for row in rows]
    accepted = [float(row.get("mtp_accepted_tokens", 0.0)) for row in rows]
    rejected = [float(row.get("mtp_rejected_tokens", 0.0)) for row in rows]
    drafted = [float(row.get("mtp_drafted_tokens", 0.0)) for row in rows]
    verifier_width = [float(row.get("mtp_verifier_width", 1.0)) for row in rows]

    generated = [float(row.get("decode_tokens", 0.0)) for row in rows]
    acceptance = []
    for a, r in zip(accepted, rejected):
        total = a + r
        acceptance.append((a / total) if total > 0 else 0.0)

    return {
        "label": label,
        "draft_tokens": draft_tokens,
        "rows": rows,
        "target": summarize(target),
        "target_context": summarize(target_context),
        "assistant": summarize(assistant),
        "misc": summarize(misc),
        "ram_peak": max(ram),
        "prompt_tokens_median": statistics.median(prompt_tokens),
        "accepted_median": statistics.median(accepted),
        "rejected_median": statistics.median(rejected),
        "drafted_median": statistics.median(drafted),
        "acceptance_median": statistics.median(acceptance),
        "generated_median": statistics.median(generated),
        "verifier_width_median": statistics.median(verifier_width),
    }


def print_summary(args: argparse.Namespace, results: list[dict]) -> None:
    baseline = results[0]["target"]["median"]
    print("benchmark_metadata")
    print(f"  model={args.model}")
    print(f"  chat={args.chat}")
    print(f"  commit={git_commit()}")
    print(f"  build_type={cmake_build_type(args.chat)}")
    print(f"  platform={platform.platform()}")
    print(f"  machine={platform.machine()}")
    print(f"  processor={cpu_name()}")
    print(f"  cpu_count={os.cpu_count()}")
    print(f"  thread_override_environment={thread_environment()}")
    print(f"  command_thread_flags=none")
    print(f"  prompt={args.prompt!r}")
    print(f"  max_tokens={args.max_tokens}")
    print(f"  repeats={args.repeats}")
    print(f"  warmup={args.warmup}")
    print(f"  temperature={args.temperature}")
    print(f"  top_k={args.top_k}")
    print("  confidence_threshold=-1.0")
    print()

    print(
        "| mode | M | prompt tokens | generated | baseline target ms/token | "
        "candidate target ms/token | slowdown | accepted | rejected | acceptance | "
        "target context ms/token | assistant ms/token | misc ms/token | peak RAM |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        m = 1 if result["draft_tokens"] is None else result["draft_tokens"] + 1
        candidate = result["target"]["median"]
        valid_slowdown = (
            baseline > 0.0
            and candidate > 0.0
            and result["generated_median"] > 0.0
        )
        slowdown = 100.0 * (candidate / baseline - 1.0) if valid_slowdown else math.nan
        slowdown_text = f"{slowdown:.1f}%" if math.isfinite(slowdown) else "n/a"
        print(
            f"| {result['label']} | {m} | {result['prompt_tokens_median']:.0f} | "
            f"{result['generated_median']:.0f} | {baseline:.6f} | {candidate:.6f} | "
            f"{slowdown_text} | {result['accepted_median']:.0f} | "
            f"{result['rejected_median']:.0f} | {100.0 * result['acceptance_median']:.1f}% | "
            f"{result['target_context']['median']:.6f} | "
            f"{result['assistant']['median']:.6f} | "
            f"{result['misc']['median']:.6f} | {result['ram_peak']:.2f} MB |"
        )

    print()
    print("raw_runs")
    for result in results:
        m = 1 if result["draft_tokens"] is None else result["draft_tokens"] + 1
        print(
            f"  M={m} verifier_width={result['verifier_width_median']:.0f} "
            f"target={result['target']} target_context={result['target_context']} "
            f"assistant={result['assistant']} "
            f"misc={result['misc']}"
        )


def print_profile_checks(args: argparse.Namespace) -> bool:
    print()
    print("profile_checks")
    all_verified = True
    for draft_tokens in (1, 2, 3):
        check = profile_small_m(args, draft_tokens)
        all_verified = all_verified and bool(check["small_m_verified"])
        print(
            f"  M={check['M']} reported_verifier_width={check['reported_verifier_width']} "
            f"graph_executions={check['graph_executions']} "
            f"verifier_matmul_rows={check['verifier_matmul_rows']} "
            f"small_m_verified={str(check['small_m_verified']).lower()}"
        )
    return all_verified


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not Path(args.chat).exists():
        raise FileNotFoundError(args.chat)

    results = [
        run_mode(args, "baseline", None),
        run_mode(args, "mtp", 1),
        run_mode(args, "mtp", 2),
        run_mode(args, "mtp", 3),
    ]
    print_summary(args, results)
    if args.profile_check:
        all_verified = print_profile_checks(args)
        if args.require_small_m and not all_verified:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
