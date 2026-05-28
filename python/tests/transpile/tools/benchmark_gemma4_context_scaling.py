#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
SWEEPS = (
    ("warmup", (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)),
    ("measured_1", (2048, 4096, 1024, 8192, 512, 16384, 256, 32768, 128, 64)),
    ("measured_2", (64, 128, 32768, 256, 16384, 512, 8192, 1024, 4096, 2048)),
    ("measured_3", (4096, 1024, 8192, 512, 16384, 256, 32768, 128, 64, 2048)),
)
JSON_FIELDS = (
    "success",
    "time_to_first_token_ms",
    "total_time_ms",
    "prefill_tps",
    "prefill_compute_tps",
    "prefill_prepare_tps",
    "ttft_prompt_tps",
    "cache_prime_ms",
    "cache_prime_compute_ms",
    "cache_state_copy_ms",
    "cache_prime_tokens",
    "first_decode_ms",
    "first_token_from_prefill",
    "prefill_padding_tokens",
    "prefill_scalar_tail_tokens",
    "decode_tps",
    "prompt_tokens",
    "completion_tokens",
    "peak_ram_usage_mb",
    "ram_usage_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma4 E4B context-scaling decode benchmarks through cactus run.",
    )
    parser.add_argument(
        "--model",
        default="weights/gemma-4-e4b-it",
        help="Prepared Cactus bundle to benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for token files, raw results, logs, JSONL, CSV, and summary.",
    )
    parser.add_argument(
        "--cactus-bin",
        default="cactus",
        help="Cactus CLI executable to invoke.",
    )
    parser.add_argument(
        "--token-id",
        type=int,
        default=106,
        help="Token id repeated to build exact-length prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate after each prefill.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the required cactus build step when the caller already built this checkout.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed runs and continue instead of stopping at the first failure.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate PNG plots after the benchmark completes.",
    )
    return parser.parse_args()


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("tmp") / "gemma4-context-scaling" / timestamp


def write_token_file(path: Path, *, token_id: int, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 4096
    remaining = count
    first = True
    with path.open("w", encoding="utf-8") as handle:
        while remaining > 0:
            current = min(chunk_size, remaining)
            text = ",".join([str(token_id)] * current)
            if not first:
                handle.write(",")
            handle.write(text)
            first = False
            remaining -= current


def ensure_token_files(out_dir: Path, *, token_id: int) -> dict[int, Path]:
    token_dir = out_dir / "tokens"
    files: dict[int, Path] = {}
    for length in LENGTHS:
        path = token_dir / f"tokens_{length}.txt"
        write_token_file(path, token_id=token_id, count=length)
        files[length] = path
    return files


def run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    return int(proc.returncode)


def read_result_json(result_path: Path, stdout_path: Path) -> dict[str, Any]:
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise RuntimeError(f"No result JSON found in {result_path} or {stdout_path}")


def decode_ms_per_token(payload: dict[str, Any]) -> float | None:
    decode_tps = float(payload.get("decode_tps") or 0.0)
    if decode_tps <= 0.0:
        return None
    return 1000.0 / decode_tps


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "sweep_index",
        "sweep_position",
        "context_tokens",
        "max_new_tokens",
        "returncode",
        "decode_ms_per_token",
        *JSON_FIELDS,
        "result_json",
        "stdout_log",
        "stderr_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    measured = [record for record in records if str(record.get("phase", "")).startswith("measured")]
    fieldnames = [
        "context_tokens",
        "runs",
        "decode_ms_per_token_mean",
        "decode_ms_per_token_stdev",
        "decode_tps_mean",
        "decode_tps_stdev",
        "total_time_ms_mean",
        "total_time_ms_stdev",
        "peak_ram_usage_mb_mean",
        "peak_ram_usage_mb_stdev",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for length in LENGTHS:
            rows = [record for record in measured if int(record.get("context_tokens", -1)) == length]
            ms_values = [value for row in rows if (value := _optional_float(row.get("decode_ms_per_token"))) is not None]
            tps_values = [value for row in rows if (value := _optional_float(row.get("decode_tps"))) is not None]
            total_values = [value for row in rows if (value := _optional_float(row.get("total_time_ms"))) is not None]
            ram_values = [value for row in rows if (value := _optional_float(row.get("peak_ram_usage_mb"))) is not None]
            writer.writerow({
                "context_tokens": length,
                "runs": len(rows),
                "decode_ms_per_token_mean": statistics.mean(ms_values) if ms_values else None,
                "decode_ms_per_token_stdev": statistics.stdev(ms_values) if len(ms_values) > 1 else 0.0 if ms_values else None,
                "decode_tps_mean": statistics.mean(tps_values) if tps_values else None,
                "decode_tps_stdev": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0 if tps_values else None,
                "total_time_ms_mean": statistics.mean(total_values) if total_values else None,
                "total_time_ms_stdev": statistics.stdev(total_values) if len(total_values) > 1 else 0.0 if total_values else None,
                "peak_ram_usage_mb_mean": statistics.mean(ram_values) if ram_values else None,
                "peak_ram_usage_mb_stdev": statistics.stdev(ram_values) if len(ram_values) > 1 else 0.0 if ram_values else None,
            })


def _float_column(rows: list[dict[str, str]], key: str, *, default: float = 0.0) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        values.append(float(raw) if raw not in {None, ""} else default)
    return values


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def write_plots(summary_path: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    contexts = [int(row["context_tokens"]) for row in rows]
    ms_mean = _float_column(rows, "decode_ms_per_token_mean")
    ms_std = _float_column(rows, "decode_ms_per_token_stdev")
    tps_mean = _float_column(rows, "decode_tps_mean")
    tps_std = _float_column(rows, "decode_tps_stdev")
    ram_mean = _float_column(rows, "peak_ram_usage_mb_mean")
    ram_std = _float_column(rows, "peak_ram_usage_mb_stdev")
    total_mean = [value / 1000.0 for value in _float_column(rows, "total_time_ms_mean")]
    total_std = [value / 1000.0 for value in _float_column(rows, "total_time_ms_stdev")]

    plt.style.use("seaborn-v0_8-whitegrid")
    error_style = {
        "ecolor": "#1f1f1f",
        "elinewidth": 1.8,
        "capthick": 1.8,
        "capsize": 6,
        "markeredgecolor": "#1f1f1f",
        "markeredgewidth": 0.9,
    }

    def configure_x_axis(ax) -> None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels([str(context) for context in contexts], rotation=35, ha="right")
        ax.grid(True, which="both", axis="y", alpha=0.35)
        ax.grid(True, which="major", axis="x", alpha=0.18)

    def finish(fig, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(contexts, ms_mean, yerr=ms_std, marker="o", linewidth=2, color="#2f6f9f", **error_style)
    configure_x_axis(ax)
    ax.set_xlabel("Prefill context tokens")
    ax.set_ylabel("Decode latency (ms/token)")
    ax.set_title("Gemma 4 E4B IT decode latency vs context length")
    paths.append(finish(fig, out_dir / "decode_ms_per_token.png"))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(contexts, tps_mean, yerr=tps_std, marker="o", linewidth=2, color="#3d7d54", **error_style)
    configure_x_axis(ax)
    ax.set_xlabel("Prefill context tokens")
    ax.set_ylabel("Decode throughput (tokens/sec)")
    ax.set_title("Gemma 4 E4B IT decode throughput vs context length")
    paths.append(finish(fig, out_dir / "decode_tps.png"))

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    line1 = ax1.errorbar(
        contexts,
        ms_mean,
        yerr=ms_std,
        marker="o",
        linewidth=2,
        color="#2f6f9f",
        label="Decode ms/token",
        **error_style,
    )
    configure_x_axis(ax1)
    ax1.set_xlabel("Prefill context tokens")
    ax1.set_ylabel("Decode latency (ms/token)", color="#2f6f9f")
    ax1.tick_params(axis="y", labelcolor="#2f6f9f")
    ax2 = ax1.twinx()
    line2 = ax2.errorbar(
        contexts,
        ram_mean,
        yerr=ram_std,
        marker="s",
        linewidth=2,
        color="#8a5a2b",
        label="Peak RAM MB",
        **error_style,
    )
    ax2.set_ylabel("Peak RAM (MB)", color="#8a5a2b")
    ax2.tick_params(axis="y", labelcolor="#8a5a2b")
    ax1.set_title("Gemma 4 E4B IT decode latency and RAM vs context length")
    ax1.legend([line1, line2], ["Decode ms/token", "Peak RAM MB"], loc="upper left")
    paths.append(finish(fig, out_dir / "decode_latency_and_ram.png"))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(contexts, total_mean, yerr=total_std, marker="o", linewidth=2, color="#7b4f9f", **error_style)
    configure_x_axis(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Prefill context tokens")
    ax.set_ylabel("Mean total run time (seconds, log scale)")
    ax.set_title("Gemma 4 E4B IT total run time vs context length")
    paths.append(finish(fig, out_dir / "total_time_seconds.png"))

    baseline = ms_mean[0]
    baseline_std = ms_std[0]
    slowdown = [value / baseline for value in ms_mean]
    slowdown_std = []
    for value, std in zip(ms_mean, ms_std):
        if value <= 0.0 or baseline <= 0.0:
            slowdown_std.append(0.0)
        else:
            relative_std = ((std / value) ** 2 + (baseline_std / baseline) ** 2) ** 0.5
            slowdown_std.append((value / baseline) * relative_std)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(contexts, slowdown, yerr=slowdown_std, marker="o", linewidth=2, color="#b24c4c", **error_style)
    ax.axhline(1.0, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
    configure_x_axis(ax)
    ax.set_xlabel("Prefill context tokens")
    ax.set_ylabel("Decode slowdown vs 64-token baseline")
    ax.set_title("Gemma 4 E4B IT decode slowdown vs initial speed")
    for context, value in zip(contexts, slowdown):
        ax.annotate(f"{value:.2f}x", (context, value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    paths.append(finish(fig, out_dir / "decode_slowdown_vs_64.png"))

    return paths


def validate_environment(cactus_bin: str) -> None:
    if shutil.which(cactus_bin) is None:
        raise RuntimeError(f"Could not find Cactus CLI executable: {cactus_bin}")


def main() -> int:
    args = parse_args()
    validate_environment(args.cactus_bin)

    out_dir = args.out_dir or default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    token_files = ensure_token_files(out_dir, token_id=args.token_id)

    metadata = {
        "model": args.model,
        "token_id": args.token_id,
        "max_new_tokens": args.max_new_tokens,
        "sweeps": [{"phase": phase, "lengths": list(lengths)} for phase, lengths in SWEEPS],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.skip_build:
        build_rc = run_command(
            [args.cactus_bin, "build"],
            stdout_path=out_dir / "build.stdout.log",
            stderr_path=out_dir / "build.stderr.log",
        )
        if build_rc != 0:
            raise RuntimeError(f"cactus build failed; see {out_dir / 'build.stderr.log'}")

    records: list[dict[str, Any]] = []
    jsonl_path = out_dir / "runs.jsonl"
    csv_path = out_dir / "runs.csv"
    summary_path = out_dir / "summary.csv"

    for sweep_index, (phase, lengths) in enumerate(SWEEPS):
        for sweep_position, length in enumerate(lengths):
            prefix = f"{sweep_index:02d}_{phase}_{sweep_position:02d}_{length}"
            result_path = out_dir / "results" / f"{prefix}.json"
            stdout_path = out_dir / "logs" / f"{prefix}.stdout.log"
            stderr_path = out_dir / "logs" / f"{prefix}.stderr.log"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)

            command = [
                args.cactus_bin,
                "run",
                args.model,
                "--input-ids-file",
                str(token_files[length]),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--result-json",
                str(result_path),
            ]
            print(f"[{phase} {sweep_position + 1}/{len(lengths)}] context={length}", flush=True)
            returncode = run_command(command, stdout_path=stdout_path, stderr_path=stderr_path)

            if returncode == 0:
                payload = read_result_json(result_path, stdout_path)
            else:
                payload = {"success": False}

            record = {
                "phase": phase,
                "sweep_index": sweep_index,
                "sweep_position": sweep_position,
                "context_tokens": length,
                "max_new_tokens": args.max_new_tokens,
                "returncode": returncode,
                "decode_ms_per_token": decode_ms_per_token(payload),
                "result_json": str(result_path),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
            for field in JSON_FIELDS:
                record[field] = payload.get(field)
            records.append(record)
            append_jsonl(jsonl_path, record)
            write_csv(csv_path, records)
            write_summary(summary_path, records)

            if returncode != 0 and not args.continue_on_error:
                raise RuntimeError(f"Benchmark run failed for context={length}; see {stderr_path}")

    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    if not args.no_plots:
        for path in write_plots(summary_path, out_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
