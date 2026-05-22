#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from collect_memory import run_with_peak_memory, sample_android_pss_mb, sample_process_tree_memory_mb
import runtime_executorch
import runtime_litert_lm
import runtime_onnxruntime
from sizes import artifact_sizes_mb


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_FIELDS = [
    "device",
    "runtime",
    "model",
    "operation",
    "seqlen",
    "decode_tokens",
    "throughput_tok_per_s",
    "peak_ram_mb",
    "disk_size_mb",
    "zipped_size_mb",
    "status",
    "notes",
]
STATUSES = {"ok", "unsupported", "oom", "error"}
_ANDROID_DEVICE_CACHE: list[tuple[str, str]] | None = None
_ARTIFACT_SIZE_CACHE: dict[Path, tuple[float, float]] = {}
_SIDEBAR_SIZE_CACHE: dict[tuple[str, str], tuple[Path, tuple[float, float]]] | None = None
UNSUPPORTED_GENERATED_BEGIN = "<!-- BEGIN GENERATED UNSUPPORTED ROWS -->"
UNSUPPORTED_GENERATED_END = "<!-- END GENERATED UNSUPPORTED ROWS -->"
STRICT_BENCHMARK_MODE = "strict"
FULL_CORE_PREFILL_BENCHMARK_MODE = "full_core_prefill"
ASR_TIMING_SOURCE = "wrapper_elapsed_seconds"


class MatrixRunError(RuntimeError):
    def __init__(self, message: str, *, peak_process_memory_mb: float = 0.0) -> None:
        super().__init__(message)
        self.peak_process_memory_mb = peak_process_memory_mb


def asr_rtfs_from_elapsed(measurements: list[dict[str, Any]], audio_seconds: float) -> list[float]:
    return [float(result["elapsed_seconds"]) / audio_seconds for result in measurements]


def is_cactus_lfm_context_capacity_error(model: str, message: str) -> bool:
    return model == "lfm_2_5_vl_1_6b" and "context exceeds transpiled text_lm_encoder capacity" in message


def cactus_lfm_context_capacity_reason(operation: dict[str, Any]) -> str:
    return (
        "Cactus LFM raw C++ artifact cannot run this prompt context; "
        f"requested seqlen={operation['seqlen']} exceeds transpiled text_lm_encoder capacity"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cactus experiment matrix.")
    parser.add_argument("--config", required=True, help="Path to matrix.yaml")
    parser.add_argument("--device", action="append", default=None, help="Device id to run. Repeatable.")
    parser.add_argument("--runtime", action="append", default=None, help="Runtime id to run. Repeatable.")
    parser.add_argument("--model", action="append", default=None, help="Model id to run. Repeatable.")
    parser.add_argument("--operation", action="append", default=None, help="Operation id to run. Repeatable.")
    parser.add_argument("--seqlen", action="append", type=int, default=None, help="LLM seqlen/context to run. Repeatable.")
    parser.add_argument("--warmup-runs", type=int, default=None, help="Override configured warmup runs.")
    parser.add_argument("--measurement-runs", type=int, default=None, help="Override configured measurement runs.")
    parser.add_argument(
        "--benchmark-mode",
        choices=[STRICT_BENCHMARK_MODE, FULL_CORE_PREFILL_BENCHMARK_MODE],
        default=STRICT_BENCHMARK_MODE,
        help="Benchmark mode. full_core_prefill runs LLM prefill rows only without strict Android taskset affinity.",
    )
    parser.add_argument("--full-core-threads", type=int, default=8, help="Thread target to record and export in full_core_prefill mode.")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not load as a mapping")
    return data


def apply_run_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config["_benchmark_mode"] = args.benchmark_mode
    config["_full_core_threads"] = args.full_core_threads
    for operation_group in ("llm", "parakeet"):
        operation_config = config.get("operations", {}).get(operation_group)
        if not isinstance(operation_config, dict):
            continue
        if args.warmup_runs is not None:
            operation_config["warmup_runs"] = args.warmup_runs
        if args.measurement_runs is not None:
            operation_config["measurement_runs"] = args.measurement_runs


def selected_ids(config: dict[str, Any], section: str, filters: list[str] | None) -> list[str]:
    values = config.get(section)
    if not isinstance(values, dict):
        raise ValueError(f"config section {section!r} must be a mapping")
    if filters is None:
        return list(values)
    missing = [value for value in filters if value not in values]
    if missing:
        raise ValueError(f"unknown {section.rstrip('s')} filter(s): {', '.join(missing)}")
    return filters


def operations_for_model(config: dict[str, Any], model_id: str) -> list[dict[str, Any]]:
    model = config["models"][model_id]
    if model.get("type") == "asr":
        parakeet = config["operations"]["parakeet"]
        return [
            {
                "operation": parakeet["operation"],
                "seqlen": "",
                "decode_tokens": "",
                "input_path": parakeet["wav_path"],
                "reference_path": parakeet["reference_transcript_path"],
            }
        ]

    llm = config["operations"]["llm"]
    token_inputs = llm.get("token_inputs", {})
    rows: list[dict[str, Any]] = []
    for seqlen in llm["prefill"]["seqlens"]:
        rows.append(
            {
                "operation": "prefill",
                "seqlen": int(seqlen),
                "decode_tokens": int(llm["prefill"]["decode_tokens"]),
                "input_path": token_input_path(token_inputs, int(seqlen)),
            }
        )
    for context in llm["decode"]["contexts"]:
        rows.append(
            {
                "operation": "decode",
                "seqlen": int(context),
                "decode_tokens": int(llm["decode"]["decode_tokens"]),
                "input_path": token_input_path(token_inputs, int(context)),
            }
        )
    return rows


def filtered_operations(
    operations: list[dict[str, Any]],
    operation_filters: list[str] | None,
    seqlen_filters: list[int] | None,
) -> list[dict[str, Any]]:
    rows = operations
    if operation_filters is not None:
        allowed_operations = set(operation_filters)
        rows = [operation for operation in rows if str(operation["operation"]) in allowed_operations]
    if seqlen_filters is not None:
        allowed_seqlens = set(seqlen_filters)
        rows = [
            operation
            for operation in rows
            if operation["seqlen"] == "" or int(operation["seqlen"]) in allowed_seqlens
        ]
    return rows


def benchmark_mode(config: dict[str, Any]) -> str:
    return str(config.get("_benchmark_mode") or STRICT_BENCHMARK_MODE)


def full_core_prefill_mode(config: dict[str, Any]) -> bool:
    return benchmark_mode(config) == FULL_CORE_PREFILL_BENCHMARK_MODE


def full_core_threads(config: dict[str, Any]) -> int:
    return max(1, int(config.get("_full_core_threads") or 8))


def full_core_env(config: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    threads = str(full_core_threads(config))
    env.update(
        {
            "OMP_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "XNNPACK_NUM_THREADS": threads,
            "CACTUS_MATRIX_BENCH_THREADS": threads,
            "CACTUS_THREADPOOL_PIN_MAX_PERF_ONLY": "1",
            "CACTUS_BENCH_PIN_MAIN_MAX_PERF": "1",
        }
    )
    return env


def paired_llm_operations(
    operations: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    decode_by_context = {
        int(operation["seqlen"]): operation
        for operation in operations
        if operation["operation"] == "decode"
    }
    prefill_contexts = {
        int(operation["seqlen"])
        for operation in operations
        if operation["operation"] == "prefill"
    }
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for operation in operations:
        seqlen = int(operation["seqlen"])
        if operation["operation"] == "prefill":
            pairs.append((operation, decode_by_context.get(seqlen)))
        elif operation["operation"] == "decode" and seqlen not in prefill_contexts:
            pairs.append((None, operation))
    return pairs


def token_input_path(token_inputs: dict[Any, Any], seqlen: int) -> str:
    value = token_inputs.get(seqlen, token_inputs.get(str(seqlen)))
    return str(value) if value else ""


def base_row(device: str, runtime: str, model: str, operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": device,
        "runtime": runtime,
        "model": model,
        "operation": operation["operation"],
        "seqlen": operation.get("seqlen", ""),
        "decode_tokens": operation.get("decode_tokens", ""),
        "throughput_tok_per_s": "",
        "peak_ram_mb": "",
        "disk_size_mb": "",
        "zipped_size_mb": "",
        "status": "",
        "notes": "",
    }


def fill_artifact_sizes(row: dict[str, Any], sizes: tuple[float, float] | None) -> dict[str, Any]:
    if sizes is not None:
        row["disk_size_mb"] = f"{sizes[0]:.3f}"
        row["zipped_size_mb"] = f"{sizes[1]:.3f}"
    return row


def fill_peak_ram(row: dict[str, Any], peak_ram_mb: float | None) -> dict[str, Any]:
    if peak_ram_mb and peak_ram_mb > 0:
        row["peak_ram_mb"] = f"{peak_ram_mb:.3f}"
    return row


def cactus_benchmark_notes(config: dict[str, Any], device: str, runtime: str, model: str) -> list[str]:
    artifact = config["models"][model]["artifacts"][runtime]
    artifact_path = resolve_artifact_path(str(artifact["path"]))
    chunk_size = artifact.get("chunk_size")
    if not chunk_size:
        match = re.search(r"(?:^|[-_])c(\d+)(?:$|[-_])", artifact_path.name)
        chunk_size = f"c{match.group(1)}" if match else "c128"
    notes = [
        f"benchmark_mode={benchmark_mode(config)}",
        f"cactus_runtime_version={config['runtimes'][runtime]['version']}",
        f"artifact_path={artifact_path}",
        "provider=cpu",
        "gpu=disabled",
        f"cactus_chunk_size={chunk_size}",
    ]
    if full_core_prefill_mode(config):
        notes.extend(
            [
                "prefill_only=true",
                f"threads={full_core_threads(config)}",
                f"thread_count={full_core_threads(config)}",
                "thread_mode=full_core_prefill",
                "affinity=not_set",
                "taskset_mask=none",
                "android_threadpool_pin_max_perf_only=true",
                "android_bench_pin_main_max_perf=true",
            ]
        )
    elif config["devices"][device].get("kind") == "mac":
        notes.extend(["threads=default", "affinity=not_set", "taskset_mask=none"])
    return notes


def unsupported_row(
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    reason: str,
    sizes: tuple[float, float] | None = None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    row = base_row(device, runtime, model, operation)
    row["status"] = "unsupported"
    notes = [reason]
    if extra_notes:
        notes.extend(extra_notes)
    row["notes"] = "; ".join(notes)
    return fill_artifact_sizes(row, sizes)


def error_row(
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    note: str,
    sizes: tuple[float, float] | None = None,
    peak_ram_mb: float | None = None,
) -> dict[str, Any]:
    row = base_row(device, runtime, model, operation)
    row["status"] = "error"
    row["notes"] = note
    fill_artifact_sizes(row, sizes)
    fill_peak_ram(row, peak_ram_mb)
    return row


def unsupported_reason(config: dict[str, Any], device: str, runtime: str, model: str) -> str | None:
    device_config = config["devices"][device]
    model_config = config["models"][model]

    if runtime == "onnxruntime":
        return runtime_onnxruntime.unsupported_reason(config, device, model)
    if runtime == "executorch":
        return runtime_executorch.unsupported_reason(config, device, model)
    if runtime == "litert_lm":
        return runtime_litert_lm.litert_lm_unsupported_reason(config, device, model, runtime)

    artifact = model_config.get("artifacts", {}).get(runtime)

    if device_config.get("kind") == "android":
        device_reason = android_device_unavailable_reason(config, device)
        if device_reason is not None:
            return device_reason

    if not isinstance(artifact, dict) or not artifact.get("path"):
        if isinstance(artifact, dict) and artifact.get("unsupported_reason"):
            return str(artifact["unsupported_reason"])
        return f"{runtime} artifact for {model} is not configured"
    artifact_path = resolve_artifact_path(str(artifact["path"]))
    if not artifact_path.exists():
        return f"artifact path does not exist: {artifact['path']}"
    if device_config.get("kind") == "android":
        if runtime == "cactus" and device == "pixel_10a" and model == "parakeet_tdt_v3":
            return None
        if runtime == "cactus":
            if model_config.get("type") == "llm":
                return android_cactus_llm_unsupported_reason(artifact_path)
            return None
        if runtime == "llama_cpp":
            if model_config.get("type") == "asr":
                return "llama.cpp does not have a configured Parakeet TDT ASR runner or artifact"
            source_root = android_llama_cpp_source_root()
            if not (source_root / "include" / "llama.h").exists():
                return f"Android llama.cpp source checkout is not configured: {source_root}"
            return None
        return f"{device} {runtime}/{model} requires an Android runner binary and deployment recipe"
    if device_config.get("kind") != "mac" or device != "mac_m4pro":
        return f"{device} host runner selection is not configured for {runtime}/{model}"
    if runtime == "llama_cpp":
        if model_config.get("type") == "asr":
            return "llama.cpp has no Parakeet TDT ASR runner for this matrix"
        if not llama_cpp_runner_available():
            return "llama.cpp native runner cannot be built because pkg-config llama is unavailable"
        return None
    if runtime != "cactus":
        return f"{runtime}/{model} has an artifact path but no Mac runner binary or Python package is available"
    if model not in {"gemma_4_e2b", "lfm_2_5_vl_1_6b", "parakeet_tdt_v3", "qwen3_vl_2b"}:
        return f"{runtime}/{model} has no configured Cactus matrix entry point"
    return None


def resolve_artifact_path(path: str) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return REPO_ROOT / artifact_path


def android_cactus_llm_unsupported_reason(artifact_path: Path) -> str | None:
    manifest_path = artifact_path / "components" / "manifest.json"
    if not manifest_path.exists():
        return f"Android Cactus token runner requires a transpiled bundle manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_names = {
        str(component.get("component"))
        for component in manifest.get("components", [])
        if isinstance(component, dict) and component.get("component")
    }
    if {"lm_encoder_step", "decoder_step"}.issubset(component_names):
        return None
    if {"text_lm_encoder", "decoder"}.issubset(component_names):
        return None
    missing = sorted({"lm_encoder_step", "decoder_step"} - component_names)
    if missing:
        available = ",".join(sorted(component_names)) or "none"
        return (
            "Android Cactus token runner requires lm_encoder_step+decoder_step or text_lm_encoder+decoder "
            f"components; missing {','.join(missing)} (available: {available})"
        )
    return None


def android_cactus_llm_operation_unsupported_reason(artifact_path: Path, operation: dict[str, Any]) -> str | None:
    manifest_path = artifact_path / "components" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_names = {
        str(component.get("component"))
        for component in manifest.get("components", [])
        if isinstance(component, dict) and component.get("component")
    }
    if {"lm_encoder_step", "decoder_step"}.issubset(component_names):
        return None
    if not {"text_lm_encoder", "decoder"}.issubset(component_names):
        return None

    input_shapes = manifest.get("inputs", {}).get("input_shapes", {})
    input_ids_shape = input_shapes.get("input_ids")
    if not (isinstance(input_ids_shape, list) and input_ids_shape):
        return "Android Cactus text_lm_encoder route requires inputs.input_shapes.input_ids"
    context_capacity = int(input_ids_shape[-1])
    required_tokens = int(operation["seqlen"])
    if int(operation.get("decode_tokens") or 0) > 0:
        required_tokens += int(operation["decode_tokens"]) - 1
    if required_tokens > context_capacity:
        return (
            "Android Cactus text_lm_encoder route exceeds transpiled context "
            f"{context_capacity}: requires {required_tokens} tokens"
        )
    return None


def collect_artifact_sizes(config: dict[str, Any], runtime: str, model: str) -> tuple[float, float] | None:
    artifact = config["models"][model].get("artifacts", {}).get(runtime)
    if not isinstance(artifact, dict) or not artifact.get("path"):
        return None
    artifact_path = resolve_artifact_path(str(artifact["path"]))
    if model == "parakeet_tdt_v3":
        transpiled_path = parakeet_transpiled_artifact_path(artifact_path)
        if transpiled_path is not None:
            artifact_path = transpiled_path
    artifact_path = artifact_path.resolve()
    sidebar_sizes = cached_sidebar_artifact_sizes(runtime, model, artifact_path)
    if sidebar_sizes is not None:
        return sidebar_sizes
    if artifact_path not in _ARTIFACT_SIZE_CACHE:
        _ARTIFACT_SIZE_CACHE[artifact_path] = artifact_sizes_mb(artifact_path)
    return _ARTIFACT_SIZE_CACHE[artifact_path]


def cached_sidebar_artifact_sizes(runtime: str, model: str, artifact_path: Path) -> tuple[float, float] | None:
    cache = sidebar_size_cache()
    cached = cache.get((runtime, model))
    if cached is None:
        return None
    cached_path, sizes = cached
    if cached_path == artifact_path:
        return sizes
    return None


def sidebar_size_cache() -> dict[tuple[str, str], tuple[Path, tuple[float, float]]]:
    global _SIDEBAR_SIZE_CACHE
    if _SIDEBAR_SIZE_CACHE is not None:
        return _SIDEBAR_SIZE_CACHE

    cache: dict[tuple[str, str], tuple[Path, tuple[float, float]]] = {}
    sidebar_path = REPO_ROOT / "experiments" / "matrix" / "results" / "disk_size_sidebar.csv"
    if not sidebar_path.exists():
        _SIDEBAR_SIZE_CACHE = cache
        return cache
    with sidebar_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            notes = str(row.get("notes") or "")
            if not notes.startswith("artifact_path="):
                continue
            try:
                disk_mb = float(str(row.get("disk_size_mb") or ""))
                zipped_mb = float(str(row.get("zipped_size_mb") or ""))
            except ValueError:
                continue
            path = Path(notes.split("=", 1)[1]).expanduser().resolve()
            cache[(str(row.get("runtime")), str(row.get("model")))] = (path, (disk_mb, zipped_mb))
    _SIDEBAR_SIZE_CACHE = cache
    return cache


def append_unsupported(unsupported_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = []
    seen = set()
    for row in rows:
        if row["status"] == "unsupported":
            line = (
                f"{row['device']},{row['runtime']},{row['model']},{row['operation']},"
                f"seqlen={row['seqlen']},decode_tokens={row['decode_tokens']}: {row['notes']}"
            )
            if line not in seen:
                lines.append(line)
                seen.add(line)
    generated_lines = [
        UNSUPPORTED_GENERATED_BEGIN,
        "# Generated Unsupported Rows",
        "",
        *lines,
        UNSUPPORTED_GENERATED_END,
        "",
    ]
    unsupported_path.parent.mkdir(parents=True, exist_ok=True)
    generated_text = "\n".join(generated_lines)
    if unsupported_path.exists():
        existing = unsupported_path.read_text(encoding="utf-8")
        if UNSUPPORTED_GENERATED_BEGIN in existing and UNSUPPORTED_GENERATED_END in existing:
            prefix, rest = existing.split(UNSUPPORTED_GENERATED_BEGIN, 1)
            _, suffix = rest.split(UNSUPPORTED_GENERATED_END, 1)
            unsupported_path.write_text(prefix.rstrip() + "\n\n" + generated_text + suffix.lstrip(), encoding="utf-8")
            return
        unsupported_path.write_text(existing.rstrip() + "\n\n" + generated_text, encoding="utf-8")
        return
    unsupported_path.write_text(generated_text, encoding="utf-8")


def should_update_unsupported(args: argparse.Namespace) -> bool:
    return args.runtime is None and args.model is None and args.operation is None and args.seqlen is None


def input_ids_for_operation(operation: dict[str, Any]) -> str:
    if not operation.get("input_path"):
        return ",".join(["2"] * int(operation["seqlen"]))
    input_path = Path(str(operation["input_path"]))
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    parts = [part.strip() for part in input_path.read_text(encoding="utf-8").split(",") if part.strip()]
    seqlen = int(operation["seqlen"])
    if len(parts) != seqlen:
        raise ValueError(f"{input_path} has {len(parts)} token ids, expected {seqlen}")
    return ",".join(parts)


def cactus_command(config: dict[str, Any], model: str, artifact_path: str, operation: dict[str, Any], result_json: Path) -> list[str]:
    if operation["operation"] == "prefill":
        max_new_tokens = 0 if full_core_prefill_mode(config) else 1
    else:
        max_new_tokens = int(operation["decode_tokens"])
    command = [
        "cactus",
        "run",
        artifact_path,
        "--max-new-tokens",
        str(max_new_tokens),
        "--result-json",
        str(result_json),
    ]
    command.extend(["--input-ids", input_ids_for_operation(operation)])
    return command


def run_cactus_once(config: dict[str, Any], model: str, artifact_path: str, operation: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="cactus_matrix_",
        suffix=".json",
        dir=result_dir,
        delete=False,
    ) as handle:
        result_path = Path(handle.name)

    command = cactus_command(config, model, artifact_path, operation, result_path)
    completed = run_with_peak_memory(
        command,
        cwd=REPO_ROOT,
        env=full_core_env(config) if full_core_prefill_mode(config) else None,
    )
    if completed.returncode != 0:
        message = ""
        if result_path.exists() and result_path.stat().st_size > 0:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                message = str(payload.get("error") or "")
        if not message:
            try:
                payload = parse_last_json_object(completed.stdout)
                message = str(payload.get("error") or "")
            except (json.JSONDecodeError, ValueError):
                message = ""
        if not message:
            detail = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
            if detail:
                message = f"{detail[-1]} (command exited {completed.returncode})"
            else:
                message = f"command exited {completed.returncode}"
        raise MatrixRunError(message, peak_process_memory_mb=completed.peak_process_memory_mb)
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["_peak_process_memory_mb"] = completed.peak_process_memory_mb
    return result


def run_cactus_gemma(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    output_path: Path,
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    artifact_path = str(config["models"][model]["artifacts"][runtime]["path"])
    runs = config["operations"]["llm"]
    result_dir = output_path.parent / ".run_json"

    try:
        for _ in range(int(runs["warmup_runs"])):
            run_cactus_once(config, model, artifact_path, operation, result_dir)

        measurements = [
            run_cactus_once(config, model, artifact_path, operation, result_dir)
            for _ in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        if is_cactus_lfm_context_capacity_error(model, str(exc)):
            return unsupported_row(
                device,
                runtime,
                model,
                operation,
                cactus_lfm_context_capacity_reason(operation),
                sizes=sizes,
            )
        return error_row(
            device,
            runtime,
            model,
            operation,
            str(exc),
            sizes=sizes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes)

    throughputs = [
        measured_throughput(result, operation=operation)
        for result in measurements
    ]
    ram_values = [
        float(result.get("peak_ram_usage_mb") or result.get("ram_usage_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}",
        *cactus_benchmark_notes(config, device, runtime, model),
    ]
    decode_modes = sorted(
        {
            str(result.get("decode_mode"))
            for result in measurements
            if result.get("decode_mode")
        }
    )
    if decode_modes:
        notes.append(f"decode_mode={','.join(decode_modes)}")
    if model == "lfm_2_5_vl_1_6b":
        notes.append("uses fixed image fixture")
    if operation["operation"] == "prefill" and full_core_prefill_mode(config):
        notes.append("prefill-only measured with zero decode tokens")
    elif operation["operation"] == "prefill":
        notes.append("prefill measured with one generated token for TTFT")
    token_note = actual_token_note(measurements, operation)
    if token_note:
        notes.append(token_note)
    if any(result.get("peak_ram_usage_mb") for result in measurements):
        notes.append("ram_source=cactus_run_native_peak")
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def cactus_llm_measurements(
    config: dict[str, Any],
    model: str,
    runtime: str,
    operation: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    artifact_path = str(config["models"][model]["artifacts"][runtime]["path"])
    runs = config["operations"]["llm"]
    result_dir = output_path.parent / ".run_json"

    for _ in range(int(runs["warmup_runs"])):
        run_cactus_once(config, model, artifact_path, operation, result_dir)

    return [
        run_cactus_once(config, model, artifact_path, operation, result_dir)
        for _ in range(int(runs["measurement_runs"]))
    ]


def cactus_llm_row_from_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    measurements: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    runs = config["operations"]["llm"]
    throughputs = [
        measured_throughput(result, operation=operation)
        for result in measurements
    ]
    ram_values = [
        float(result.get("peak_ram_usage_mb") or result.get("ram_usage_mb") or result.get("_peak_process_memory_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}",
        *cactus_benchmark_notes(config, device, runtime, model),
    ]
    if extra_notes:
        notes.extend(extra_notes)
    decode_modes = sorted(
        {
            str(result.get("decode_mode"))
            for result in measurements
            if result.get("decode_mode")
        }
    )
    if decode_modes:
        notes.append(f"decode_mode={','.join(decode_modes)}")
    if model == "lfm_2_5_vl_1_6b":
        notes.append("uses fixed image fixture")
    if operation["operation"] == "prefill" and full_core_prefill_mode(config):
        notes.append("prefill-only measured with zero decode tokens")
    elif operation["operation"] == "prefill" and "paired_prefill_decode=true" not in (extra_notes or []):
        notes.append("prefill measured with one generated token for TTFT")
    token_note = actual_token_note(measurements, operation)
    if token_note:
        notes.append(token_note)
    if any(result.get("peak_ram_usage_mb") for result in measurements):
        notes.append("ram_source=cactus_run_native_peak")
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def run_cactus_llm_pair(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    prefill_operation: dict[str, Any],
    decode_operation: dict[str, Any],
    output_path: Path,
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    try:
        measurements = cactus_llm_measurements(config, model, runtime, decode_operation, output_path)
    except MatrixRunError as exc:
        if is_cactus_lfm_context_capacity_error(model, str(exc)):
            reason = cactus_lfm_context_capacity_reason(decode_operation)
            return [
                unsupported_row(device, runtime, model, prefill_operation, reason, sizes=sizes),
                unsupported_row(device, runtime, model, decode_operation, reason, sizes=sizes),
            ]
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
        ]
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes),
        ]

    seqlen = int(prefill_operation["seqlen"])
    return [
        cactus_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            prefill_operation,
            measurements,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                f"shared_decode_tokens={decode_operation['decode_tokens']}",
            ],
        ),
        cactus_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            decode_operation,
            measurements,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                "shares measured invocation with prefill row",
            ],
        ),
    ]


def run_cactus_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    output_path: Path,
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    artifact_root = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    if full_core_prefill_mode(config):
        operations = [operation for operation in operations if operation["operation"] == "prefill"]
    rows: list[dict[str, Any]] = []
    for prefill_operation, decode_operation in paired_llm_operations(operations):
        if prefill_operation is not None and decode_operation is not None:
            rows.extend(
                run_cactus_llm_pair(
                    config,
                    device,
                    runtime,
                    model,
                    prefill_operation,
                    decode_operation,
                    output_path,
                    sizes,
                )
            )
        else:
            operation = prefill_operation or decode_operation
            if operation is None:
                continue
            rows.append(run_cactus_gemma(config, device, runtime, model, operation, output_path, sizes))
    return rows


def llama_cpp_runner_available() -> bool:
    completed = subprocess.run(
        ["pkg-config", "--exists", "llama"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def native_llama_cpp_runner() -> Path:
    source = REPO_ROOT / "experiments" / "matrix" / "native_llama_cpp_json.cpp"
    build_dir = REPO_ROOT / "experiments" / "matrix" / ".build"
    binary = build_dir / "native_llama_cpp_json"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary

    cflags = pkg_config_args("--cflags", "llama", "ggml")
    libs = pkg_config_args("--libs", "llama", "ggml")
    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "clang++" if platform.system() == "Darwin" else "g++")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        *cflags,
        str(source),
        "-o",
        str(binary),
        *libs,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "native llama.cpp runner build failed")
    return binary


def pkg_config_args(*args: str) -> list[str]:
    completed = subprocess.run(
        ["pkg-config", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"pkg-config {' '.join(args)} failed")
    return shlex.split(completed.stdout)


def run_llama_cpp_once(config: dict[str, Any], artifact_path: Path, operation: dict[str, Any]) -> dict[str, Any]:
    runner = native_llama_cpp_runner()
    threads = full_core_threads(config) if full_core_prefill_mode(config) else 1
    command = [
        str(runner),
        str(artifact_path),
        str(operation["seqlen"]),
        str(operation["decode_tokens"]),
        str(threads),
    ]
    completed = run_with_peak_memory(command, cwd=REPO_ROOT, env=full_core_env(config) if full_core_prefill_mode(config) else None)
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        message = detail[-1] if detail else f"command exited {completed.returncode}"
        raise MatrixRunError(message, peak_process_memory_mb=completed.peak_process_memory_mb)
    payload = parse_last_json_object(completed.stdout)
    payload["_peak_process_memory_mb"] = completed.peak_process_memory_mb
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    return payload


def llama_cpp_measurements(
    config: dict[str, Any],
    model: str,
    runtime: str,
    operation: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact_path = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    runs = config["operations"]["llm"]
    for _ in range(int(runs["warmup_runs"])):
        run_llama_cpp_once(config, artifact_path, operation)
    return [
        run_llama_cpp_once(config, artifact_path, operation)
        for _ in range(int(runs["measurement_runs"]))
    ]


def llama_cpp_llm_row_from_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    measurements: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    runs = config["operations"]["llm"]
    throughputs = [measured_throughput(result, operation=operation) for result in measurements]
    ram_values = [float(result.get("_peak_process_memory_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}",
        "runner=native_llama_cpp_json",
        f"benchmark_mode={benchmark_mode(config)}",
        f"threads={full_core_threads(config) if full_core_prefill_mode(config) else 1}",
        f"thread_count={full_core_threads(config) if full_core_prefill_mode(config) else 1}",
        "mmap=on",
        "n_gpu_layers=0",
        "synthetic exact-length token ids",
    ]
    quantizations = sorted(
        {
            str(result.get("gguf_file_type_name"))
            for result in measurements
            if result.get("gguf_file_type_name")
        }
    )
    if quantizations:
        notes.append(f"quantization={','.join(quantizations)}")
    if extra_notes:
        notes.extend(extra_notes)
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def run_llama_cpp_llm_pair(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    prefill_operation: dict[str, Any],
    decode_operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    try:
        measurements = llama_cpp_measurements(config, model, runtime, decode_operation)
    except MatrixRunError as exc:
        if is_llama_cpp_unsupported_error(str(exc)):
            reason = f"configured llama.cpp GGUF is unloadable by the installed runner: {exc}"
            return [
                unsupported_row(device, runtime, model, prefill_operation, reason, sizes=sizes),
                unsupported_row(device, runtime, model, decode_operation, reason, sizes=sizes),
            ]
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
        ]
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes),
        ]

    seqlen = int(prefill_operation["seqlen"])
    return [
        llama_cpp_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            prefill_operation,
            measurements,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                f"shared_decode_tokens={decode_operation['decode_tokens']}",
            ],
        ),
        llama_cpp_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            decode_operation,
            measurements,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                "shares measured invocation with prefill row",
            ],
        ),
    ]


def run_llama_cpp_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    artifact_root = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    rows: list[dict[str, Any]] = []
    for prefill_operation, decode_operation in paired_llm_operations(operations):
        if prefill_operation is not None and decode_operation is not None:
            rows.extend(
                run_llama_cpp_llm_pair(
                    config,
                    device,
                    runtime,
                    model,
                    prefill_operation,
                    decode_operation,
                    sizes,
                )
            )
        else:
            operation = prefill_operation or decode_operation
            if operation is None:
                continue
            try:
                measurements = llama_cpp_measurements(config, model, runtime, operation)
                rows.append(
                    llama_cpp_llm_row_from_measurements(
                        config,
                        device,
                        runtime,
                        model,
                        operation,
                        measurements,
                        sizes,
                    )
                )
            except MatrixRunError as exc:
                if is_llama_cpp_unsupported_error(str(exc)):
                    rows.append(
                        unsupported_row(
                            device,
                            runtime,
                            model,
                            operation,
                            f"configured llama.cpp GGUF is unloadable by the installed runner: {exc}",
                            sizes=sizes,
                        )
                    )
                else:
                    rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb))
            except (OSError, RuntimeError, ValueError) as exc:
                rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes))
    return rows


def is_llama_cpp_unsupported_error(message: str) -> bool:
    return (
        "failed to load llama.cpp model" in message
        or "llama.cpp matrix requires Q4" in message
        or "failed to read GGUF metadata" in message
        or "GGUF metadata is missing" in message
    )


def run_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    output_path: Path,
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    if config["devices"][device].get("kind") == "android" and runtime == "cactus":
        return run_android_cactus_llm_operations(config, device, runtime, model, operations, sizes)
    if config["devices"][device].get("kind") == "android" and runtime == "llama_cpp":
        return run_android_llama_cpp_llm_operations(config, device, runtime, model, operations, sizes)
    if runtime == "cactus":
        return run_cactus_llm_operations(config, device, runtime, model, operations, output_path, sizes)
    if runtime == "llama_cpp":
        return run_llama_cpp_llm_operations(config, device, runtime, model, operations, sizes)
    if runtime == "onnxruntime":
        return runtime_onnxruntime.run_llm_operations(config, device, runtime, model, operations, sizes)
    if runtime == "executorch":
        rows = runtime_executorch.rows_for_operations(config, device, model, operations)
        for row in rows:
            fill_artifact_sizes(row, sizes)
        return rows
    if runtime == "litert_lm":
        return runtime_litert_lm.run_litert_lm_llm_operations(config, device, runtime, model, operations, sizes)
    return [
        unsupported_row(
            device,
            runtime,
            model,
            operation,
            f"{runtime}/{model} has no LLM runner wired for this host",
            sizes=sizes,
        )
        for operation in operations
    ]


def run_cactus_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    artifact_path = str(config["models"][model]["artifacts"][runtime]["path"])
    artifact_root = resolve_artifact_path(artifact_path)
    runs = config["operations"]["parakeet"]
    audio_path = resolve_repo_path(str(operation["input_path"]))
    reference_path = resolve_repo_path(str(operation["reference_path"]))
    audio_seconds = float(runs["audio_seconds"])

    if config["devices"][device].get("kind") == "android":
        return run_android_native_cactus_parakeet(
            config,
            device,
            runtime,
            model,
            operation,
            artifact_root,
            audio_path,
            reference_path,
            audio_seconds,
            sizes,
        )

    if not audio_path.exists():
        return error_row(device, runtime, model, operation, f"audio fixture does not exist: {audio_path}", sizes=sizes)
    if not reference_path.exists():
        return error_row(
            device,
            runtime,
            model,
            operation,
            f"reference transcript does not exist: {reference_path}",
            sizes=sizes,
        )

    command = [
        str(native_transcribe_runner()),
        str(artifact_root),
        str(audio_path),
        '{"max_tokens":500,"telemetry_enabled":false,"auto_handoff":false}',
    ]
    run_once = lambda: run_transcribe_once(command)

    try:
        for _ in range(int(runs["warmup_runs"])):
            run_once()
        measurements = [
            run_once()
            for _ in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        return error_row(
            device,
            runtime,
            model,
            operation,
            str(exc),
            sizes=sizes,
        )

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [
        float(result.get("peak_ram_usage_mb") or result.get("ram_usage_mb") or result.get("peak_process_memory_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("response") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [f"WER={wer:.6f}", "runner=native_transcribe_json", f"timing_source={ASR_TIMING_SOURCE}"]
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    notes.append(f"transcript={transcript}")
    row["notes"] = "; ".join(notes)
    return row


def run_android_native_cactus_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    artifact_root: Path,
    audio_path: Path,
    reference_path: Path,
    audio_seconds: float,
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    if not audio_path.exists():
        return error_row(device, runtime, model, operation, f"audio fixture does not exist: {audio_path}", sizes=sizes)
    if not reference_path.exists():
        return error_row(device, runtime, model, operation, f"reference transcript does not exist: {reference_path}", sizes=sizes)

    try:
        serial = select_android_serial_for_device(config, device)
        device_info = android_device_info(serial)
        runner = android_native_transcribe_runner()
        prepared = prepare_android_native_transcribe(
            serial=serial,
            artifact_root=artifact_root,
            runner=runner,
            audio_path=audio_path,
        )
        runs = config["operations"]["parakeet"]
        _, affinity_mask = android_core_affinity(config, device)
        for run_index in range(int(runs["warmup_runs"])):
            run_android_native_transcribe_once(serial, prepared, f"warmup_native_transcribe_{run_index}", affinity_mask)
        measurements = [
            run_android_native_transcribe_once(serial, prepared, f"measure_native_transcribe_{run_index}", affinity_mask)
            for run_index in range(int(runs["measurement_runs"]))
        ]
    except (MatrixRunError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=getattr(exc, "peak_process_memory_mb", 0.0))

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [
        float(result.get("peak_process_memory_mb") or result.get("peak_pss_mb") or result.get("ram_usage_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("response") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    row["notes"] = "; ".join(
        [
            f"WER={wer:.6f}",
            "runner=android_native_transcribe_json",
            f"timing_source={ASR_TIMING_SOURCE}",
            *android_core_affinity_notes(config, device),
            f"serial={serial}",
            f"android={device_info['android_release']}",
            f"thermal_status={device_info['thermal_status']}",
            f"transcript={transcript}",
        ]
    )
    return row


def run_executorch_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    artifact = config["models"][model]["artifacts"][runtime]
    runner_path = resolve_repo_path(str(artifact["runner_path"]))
    model_path = resolve_repo_path(str(artifact["path"]))
    tokenizer_path = resolve_repo_path(str(artifact["tokenizer_path"]))
    audio_path = resolve_repo_path(str(operation["input_path"]))
    reference_path = resolve_repo_path(str(operation["reference_path"]))
    runs = config["operations"]["parakeet"]
    audio_seconds = float(runs["audio_seconds"])

    if not audio_path.exists():
        return error_row(device, runtime, model, operation, f"audio fixture does not exist: {audio_path}", sizes=sizes)
    if not reference_path.exists():
        return error_row(device, runtime, model, operation, f"reference transcript does not exist: {reference_path}", sizes=sizes)

    if config["devices"][device].get("kind") == "android":
        return run_android_executorch_parakeet(
            config,
            device,
            runtime,
            model,
            operation,
            artifact,
            model_path,
            tokenizer_path,
            audio_path,
            reference_path,
            audio_seconds,
            sizes,
        )

    command = [
        str(runner_path),
        f"-model_path={model_path}",
        f"-tokenizer_path={tokenizer_path}",
        f"-audio_path={audio_path}",
        "-timestamps=none",
    ]

    try:
        for _ in range(int(runs["warmup_runs"])):
            run_executorch_parakeet_once(command)
        measurements = [
            run_executorch_parakeet_once(command)
            for _ in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb)

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [float(result.get("peak_process_memory_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("transcript") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"WER={wer:.6f}",
        "runner=executorch_parakeet",
        f"timing_source={ASR_TIMING_SOURCE}",
        f"format={runtime_executorch.quantization_display_name(str(artifact.get('quantization') or 'q4_per_channel'))}",
    ]
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    notes.append(f"transcript={transcript}")
    row["notes"] = "; ".join(notes)
    return row


def run_android_executorch_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    artifact: dict[str, Any],
    model_path: Path,
    tokenizer_path: Path,
    audio_path: Path,
    reference_path: Path,
    audio_seconds: float,
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    runs = config["operations"]["parakeet"]
    try:
        serial = select_android_serial_for_device(config, device)
        device_info = android_device_info(serial)
        runner_path = runtime_executorch.android_asr_runner_path(artifact)
        if not runner_path.exists():
            raise MatrixRunError(f"missing arm64-v8a ASR runner binary: {runtime_executorch.display_path(runner_path)}")
        prepared = prepare_android_executorch_parakeet(
            serial=serial,
            runner=runner_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            audio_path=audio_path,
        )
        _, affinity_mask = android_core_affinity(config, device)
        for run_index in range(int(runs["warmup_runs"])):
            run_android_executorch_parakeet_once(serial, prepared, f"warmup_executorch_parakeet_{run_index}", affinity_mask)
        measurements = [
            run_android_executorch_parakeet_once(serial, prepared, f"measure_executorch_parakeet_{run_index}", affinity_mask)
            for run_index in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes)

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [
        float(result.get("peak_process_memory_mb") or result.get("peak_pss_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("transcript") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    row["notes"] = "; ".join(
        [
            f"WER={wer:.6f}",
            "runner=android_executorch_parakeet",
            f"timing_source={ASR_TIMING_SOURCE}",
            f"format={runtime_executorch.quantization_display_name(str(artifact.get('quantization') or 'q4_per_channel'))}",
            *android_core_affinity_notes(config, device),
            f"serial={serial}",
            f"android={device_info['android_release']}",
            f"thermal_status={device_info['thermal_status']}",
            f"transcript={transcript}",
        ]
    )
    return row


def run_executorch_parakeet_once(command: list[str]) -> dict[str, Any]:
    completed = run_with_peak_memory(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        message = detail[-1] if detail else f"command exited {completed.returncode}"
        raise MatrixRunError(message, peak_process_memory_mb=completed.peak_process_memory_mb)
    transcript = parse_executorch_parakeet_transcript(completed.stdout + "\n" + completed.stderr)
    return {
        "transcript": transcript,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "peak_process_memory_mb": completed.peak_process_memory_mb,
        "elapsed_seconds": completed.elapsed_seconds,
    }


def parse_executorch_parakeet_transcript(output: str) -> str:
    match = re.search(r"Transcribed text:\s*(.+)", output)
    if not match:
        raise MatrixRunError("ExecuTorch Parakeet output did not contain a transcript")
    return match.group(1).strip()


def run_litert_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    artifact = config["models"][model]["artifacts"][runtime]
    runs = config["operations"]["parakeet"]
    audio_path = resolve_repo_path(str(operation["input_path"]))
    reference_path = resolve_repo_path(str(operation["reference_path"]))
    audio_seconds = float(runs["audio_seconds"])

    if not audio_path.exists():
        return error_row(device, runtime, model, operation, f"audio fixture does not exist: {audio_path}", sizes=sizes)
    if not reference_path.exists():
        return error_row(device, runtime, model, operation, f"reference transcript does not exist: {reference_path}", sizes=sizes)

    if config["devices"][device].get("kind") == "android":
        return run_android_litert_parakeet(
            config,
            device,
            runtime,
            model,
            operation,
            artifact,
            audio_path,
            reference_path,
            audio_seconds,
            sizes,
        )

    try:
        for _ in range(int(runs["warmup_runs"])):
            run_litert_parakeet_once(artifact, audio_path)
        measurements = [
            run_litert_parakeet_once(artifact, audio_path)
            for _ in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes)

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [float(result.get("peak_process_memory_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("transcript") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"WER={wer:.6f}",
        "runner=normal_litert_parakeet",
        f"timing_source={ASR_TIMING_SOURCE}",
        f"chunks={measurements[-1].get('chunks', '')}",
        f"encoder_frames={measurements[-1].get('encoder_frames', '')}",
        f"decoder_steps={measurements[-1].get('decoder_steps', '')}",
        f"transcript={transcript}",
    ]
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def run_android_litert_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    artifact: dict[str, Any],
    audio_path: Path,
    reference_path: Path,
    audio_seconds: float,
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    runs = config["operations"]["parakeet"]
    try:
        serial = select_android_serial_for_device(config, device)
        device_info = android_device_info(serial)
        runner_path = runtime_litert_lm.android_litert_parakeet_runner_path(artifact)
        if not runner_path.exists():
            raise MatrixRunError(f"missing Android LiteRT Parakeet ASR runner binary: {runner_path}")
        host_inputs = prepare_litert_parakeet_feature_inputs(artifact, audio_path)
        prepared = prepare_android_litert_parakeet(
            serial=serial,
            artifact=artifact,
            runner=runner_path,
            host_inputs=host_inputs,
        )
        _, affinity_mask = android_core_affinity(config, device)
        for run_index in range(int(runs["warmup_runs"])):
            run_android_litert_parakeet_once(serial, prepared, f"warmup_litert_parakeet_{run_index}", affinity_mask)
        measurements = [
            run_android_litert_parakeet_once(serial, prepared, f"measure_litert_parakeet_{run_index}", affinity_mask)
            for run_index in range(int(runs["measurement_runs"]))
        ]
    except MatrixRunError as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb)
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes)

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    ram_values = [
        float(result.get("peak_process_memory_mb") or result.get("peak_pss_mb") or 0.0)
        for result in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("transcript") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    row["notes"] = "; ".join(
        [
            f"WER={wer:.6f}",
            "runner=android_litert_parakeet_asr",
            f"timing_source={ASR_TIMING_SOURCE}",
            f"chunks={measurements[-1].get('chunks', '')}",
            f"encoder_frames={measurements[-1].get('encoder_frames', '')}",
            f"decoder_steps={measurements[-1].get('decoder_steps', '')}",
            *android_core_affinity_notes(config, device),
            f"serial={serial}",
            f"android={device_info['android_release']}",
            f"thermal_status={device_info['thermal_status']}",
            f"transcript={transcript}",
        ]
    )
    return row


def run_litert_parakeet_once(artifact: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from ai_edge_litert.interpreter import Interpreter
    from scipy.io import wavfile
    from cactus.transpile.audio_preprocess import load_audio_waveform, prepare_native_parakeet_audio_features

    started = time.perf_counter()
    peak_memory_mb = sample_process_tree_memory_mb(os.getpid())
    root = resolve_artifact_path(str(artifact["path"]))
    encoder_path = resolve_repo_path(str(artifact.get("encoder_path") or root / "parakeet-encoder.tflite"))
    decoder_joint_value = artifact.get("decoder_joint_path")
    decoder_step_value = artifact.get("decoder_step_path")
    joint_step_value = artifact.get("joint_step_path")
    split_decoder = decoder_step_value is not None or joint_step_value is not None
    if split_decoder and (decoder_step_value is None or joint_step_value is None):
        raise MatrixRunError("split LiteRT Parakeet ASR requires both decoder_step_path and joint_step_path")
    decoder_path = (
        resolve_repo_path(str(decoder_joint_value))
        if decoder_joint_value is not None
        else resolve_repo_path(str(root / "parakeet-decoder-joint.tflite"))
    )
    decoder_step_path = resolve_repo_path(str(decoder_step_value)) if decoder_step_value is not None else None
    joint_step_path = resolve_repo_path(str(joint_step_value)) if joint_step_value is not None else None
    vocab_path = resolve_repo_path(str(artifact.get("vocab_path") or root / "vocab.json"))
    config_path = resolve_repo_path(str(artifact.get("config_path") or root / "config.json"))
    litert_config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder_config = litert_config.get("decoder", {})
    vocab = {int(key): str(value) for key, value in json.loads(vocab_path.read_text(encoding="utf-8")).items()}
    vocab_size = len(vocab)
    blank_id = vocab_size
    hidden_size = int(decoder_config.get("hidden_size") or 640)
    num_layers = int(decoder_config.get("num_layers") or 2)
    durations = [int(value) for value in artifact.get("durations", litert_config.get("durations", [0, 1, 2, 3, 4]))]
    max_symbols_per_step = int(artifact.get("max_symbols_per_step") or 10)
    max_chunk_seconds = float(artifact.get("max_chunk_seconds") or 5.0)
    expected_frames = int(artifact.get("expected_mel_frames") or artifact.get("t_mel") or round(max_chunk_seconds * 100.0))
    expected_mels = int(artifact.get("expected_mels") or litert_config.get("encoder", {}).get("input_mel_bins") or 128)

    waveform = load_audio_waveform(audio_path, target_sample_rate=16000, max_seconds=None)
    chunk_samples = max(1, int(round(16000.0 * max_chunk_seconds)))
    encoder = Interpreter(model_path=str(encoder_path), num_threads=1).get_signature_runner("serving_default")
    if split_decoder:
        decoder = None
        decoder_step = Interpreter(model_path=str(decoder_step_path), num_threads=1).get_signature_runner("serving_default")
        joint_step = Interpreter(model_path=str(joint_step_path), num_threads=1).get_signature_runner("serving_default")
    else:
        decoder = Interpreter(model_path=str(decoder_path), num_threads=1).get_signature_runner("serving_default")
        decoder_step = None
        joint_step = None
    peak_memory_mb = max(peak_memory_mb, sample_process_tree_memory_mb(os.getpid()))
    pieces: list[str] = []
    encoder_frames: list[int] = []
    decoder_steps = 0

    with tempfile.TemporaryDirectory(prefix="cactus_litert_parakeet_") as tmpdir:
        tmp_root = Path(tmpdir)
        for chunk_index, start in enumerate(range(0, len(waveform), chunk_samples)):
            chunk = waveform[start : start + chunk_samples]
            if chunk.size == 0:
                continue
            chunk_path = tmp_root / f"chunk_{chunk_index}.wav"
            wavfile.write(str(chunk_path), 16000, (np.clip(chunk, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16))
            features, active_frames = prepare_native_parakeet_audio_features(
                chunk_path,
                expected_frames=expected_frames,
                expected_mels=expected_mels,
                torch_dtype=torch.float32,
            )
            mel = np.ascontiguousarray(features.cpu().numpy().transpose(0, 2, 1).astype(np.float32))
            encoded = encoder(args_0=mel, args_1=np.array([min(int(active_frames), expected_frames)], dtype=np.int64))
            peak_memory_mb = max(peak_memory_mb, sample_process_tree_memory_mb(os.getpid()))
            encoder_out = next(value for value in encoded.values() if value.dtype == np.float32 and value.ndim == 3)
            encoder_length_value = next(value for value in encoded.values() if value.dtype == np.int64)
            encoder_length = int(encoder_length_value.reshape(-1)[0])
            encoder_frames.append(encoder_length)
            if split_decoder:
                token_ids, chunk_decoder_steps = greedy_litert_split_tdt_decode(
                    decoder_step,
                    joint_step,
                    encoder_out,
                    encoder_length,
                    blank_id=blank_id,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    durations=durations,
                    max_symbols_per_step=max_symbols_per_step,
                )
            else:
                token_ids, chunk_decoder_steps = greedy_litert_tdt_decode(
                    decoder,
                    encoder_out,
                    encoder_length,
                    blank_id=blank_id,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
            decoder_steps += chunk_decoder_steps
            peak_memory_mb = max(peak_memory_mb, sample_process_tree_memory_mb(os.getpid()))
            pieces.extend(vocab.get(token_id, "") for token_id in token_ids)

    transcript = detokenize_sentencepiece_pieces(pieces)
    return {
        "transcript": transcript,
        "chunks": len(encoder_frames),
        "encoder_frames": ",".join(str(value) for value in encoder_frames),
        "decoder_steps": decoder_steps,
        "peak_process_memory_mb": peak_memory_mb,
        "elapsed_seconds": time.perf_counter() - started,
    }


def greedy_litert_tdt_decode(
    decoder: Any,
    encoder_out: Any,
    encoder_length: int,
    *,
    blank_id: int,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
) -> tuple[list[int], int]:
    import numpy as np

    h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    target = np.array([[blank_id]], dtype=np.int64)
    token_ids: list[int] = []
    frame = 0
    steps = 0
    max_steps = max(1, int(encoder_length) * 8)
    while frame < int(encoder_length) and steps < max_steps:
        current = np.ascontiguousarray(encoder_out[:, :, frame].reshape(1, 1, -1).astype(np.float32))
        outputs = decoder(args_0=current, args_1=target, args_2=h, args_3=c)
        logits = next(value.reshape(-1) for value in outputs.values() if value.shape[-1] > hidden_size)
        token_id = int(np.argmax(logits[: blank_id + 1]))
        duration_logits = logits[vocab_size + 1 :]
        duration = int(np.argmax(duration_logits)) if duration_logits.size else 1
        if token_id != blank_id and token_id < vocab_size:
            token_ids.append(token_id)
            target = np.array([[token_id]], dtype=np.int64)
            if "output_1" in outputs:
                h = outputs["output_1"]
            if "output_2" in outputs:
                c = outputs["output_2"]
        else:
            target = np.array([[blank_id]], dtype=np.int64)
        frame += max(1, duration)
        steps += 1
    return token_ids, steps


def greedy_litert_split_tdt_decode(
    decoder_step: Any,
    joint_step: Any,
    encoder_out: Any,
    encoder_length: int,
    *,
    blank_id: int,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    durations: list[int],
    max_symbols_per_step: int,
) -> tuple[list[int], int]:
    import numpy as np

    h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    token = np.array([[blank_id]], dtype=np.int64)
    pred_outputs = decoder_step(args_0=token, args_1=h, args_2=c)
    g_proj = np.ascontiguousarray(pred_outputs["output_0"].transpose(0, 2, 1).astype(np.float32))
    h = pred_outputs["output_1"]
    c = pred_outputs["output_2"]

    token_ids: list[int] = []
    frame = 0
    steps = 0
    symbols_on_frame = 0
    max_steps = max(1, int(encoder_length) * max(1, max_symbols_per_step + 2))
    while frame < int(encoder_length) and steps < max_steps:
        enc_frame = np.ascontiguousarray(encoder_out[:, :, frame : frame + 1].astype(np.float32))
        joint_outputs = joint_step(args_0=enc_frame, args_1=g_proj)
        logits = next(value.reshape(-1) for value in joint_outputs.values() if value.shape[-1] > hidden_size)
        token_id = int(np.argmax(logits[: blank_id + 1]))
        duration_logits = logits[vocab_size + 1 :]
        duration_index = int(np.argmax(duration_logits)) if duration_logits.size else 1
        duration = durations[duration_index] if 0 <= duration_index < len(durations) else max(1, duration_index)

        if token_id == blank_id or token_id >= vocab_size:
            frame += max(duration, 1)
            symbols_on_frame = 0
        else:
            token_ids.append(token_id)
            token = np.array([[token_id]], dtype=np.int64)
            pred_outputs = decoder_step(args_0=token, args_1=h, args_2=c)
            g_proj = np.ascontiguousarray(pred_outputs["output_0"].transpose(0, 2, 1).astype(np.float32))
            h = pred_outputs["output_1"]
            c = pred_outputs["output_2"]
            frame += duration
            if duration == 0:
                symbols_on_frame += 1
                if symbols_on_frame >= max_symbols_per_step:
                    frame += 1
                    symbols_on_frame = 0
            else:
                symbols_on_frame = 0
        steps += 1
    return token_ids, steps


def detokenize_sentencepiece_pieces(pieces: list[str]) -> str:
    ignored = {
        "<unk>",
        "<pad>",
        "<|nospeech|>",
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|startofcontext|>",
        "<|itn|>",
        "<|noitn|>",
        "<|timestamp|>",
        "<|notimestamp|>",
        "<|diarize|>",
        "<|nodiarize|>",
        "<|spkchange|>",
        "<|audioseparator|>",
    }
    text = "".join(piece for piece in pieces if piece and piece not in ignored)
    return re.sub(r"\s+", " ", text.replace("▁", " ")).strip()


def run_android_cactus_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    artifact_root = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    if full_core_prefill_mode(config):
        operations = [operation for operation in operations if operation["operation"] == "prefill"]
    rows: list[dict[str, Any]] = []
    for prefill_operation, decode_operation in paired_llm_operations(operations):
        if prefill_operation is not None and decode_operation is not None:
            prefill_reason = android_cactus_llm_operation_unsupported_reason(artifact_root, prefill_operation)
            decode_reason = android_cactus_llm_operation_unsupported_reason(artifact_root, decode_operation)
            if prefill_reason is not None or decode_reason is not None:
                reason = prefill_reason or decode_reason or ""
                rows.append(unsupported_row(device, runtime, model, prefill_operation, reason, sizes=sizes))
                rows.append(unsupported_row(device, runtime, model, decode_operation, reason, sizes=sizes))
                continue
            rows.extend(
                run_android_cactus_llm_pair(
                    config,
                    device,
                    runtime,
                    model,
                    prefill_operation,
                    decode_operation,
                    sizes,
                )
            )
        else:
            operation = prefill_operation or decode_operation
            if operation is None:
                continue
            reason = android_cactus_llm_operation_unsupported_reason(artifact_root, operation)
            if reason is not None:
                rows.append(unsupported_row(device, runtime, model, operation, reason, sizes=sizes))
                continue
            try:
                measurements, metadata = android_cactus_llm_measurements(config, device, runtime, model, operation)
                rows.append(
                    android_cactus_llm_row_from_measurements(
                        config,
                        device,
                        runtime,
                        model,
                        operation,
                        measurements,
                        metadata,
                        sizes,
                    )
                )
            except MatrixRunError as exc:
                rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb))
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes))
    return rows


def run_android_cactus_llm_pair(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    prefill_operation: dict[str, Any],
    decode_operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    try:
        measurements, metadata = android_cactus_llm_measurements(config, device, runtime, model, decode_operation)
    except MatrixRunError as exc:
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
        ]
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes),
        ]

    seqlen = int(prefill_operation["seqlen"])
    return [
        android_cactus_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            prefill_operation,
            measurements,
            metadata,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                f"shared_decode_tokens={decode_operation['decode_tokens']}",
            ],
        ),
        android_cactus_llm_row_from_measurements(
            config,
            device,
            runtime,
            model,
            decode_operation,
            measurements,
            metadata,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                "shares measured invocation with prefill row",
            ],
        ),
    ]


def android_cactus_llm_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    serial = select_android_serial_for_device(config, device)
    device_info = android_device_info(serial)
    runner = android_cactus_llm_runner()
    artifact_root = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    prepared = prepare_android_cactus_llm(serial=serial, artifact_root=artifact_root, runner=runner, operation=operation)
    if full_core_prefill_mode(config):
        prepared["full_core_threads"] = str(full_core_threads(config))
    runs = config["operations"]["llm"]
    affinity_mask = None if full_core_prefill_mode(config) else android_core_affinity(config, device)[1]
    for run_index in range(int(runs["warmup_runs"])):
        run_android_cactus_llm_once(serial, prepared, operation, f"warmup_{model}_{operation['operation']}_{operation['seqlen']}_{run_index}", affinity_mask)
    measurements = [
        run_android_cactus_llm_once(serial, prepared, operation, f"measure_{model}_{operation['operation']}_{operation['seqlen']}_{run_index}", affinity_mask)
        for run_index in range(int(runs["measurement_runs"]))
    ]
    return measurements, {"serial": serial, **device_info}


def android_cactus_llm_row_from_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    measurements: list[dict[str, Any]],
    metadata: dict[str, str],
    sizes: tuple[float, float] | None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    runs = config["operations"]["llm"]
    throughputs = [measured_throughput(result, operation=operation) for result in measurements]
    ram_values = [float(result.get("peak_pss_mb") or result.get("ram_usage_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    affinity_notes = ["taskset_mask=none", "affinity=not_set"] if full_core_prefill_mode(config) else android_core_affinity_notes(config, device)
    threads = full_core_threads(config) if full_core_prefill_mode(config) else 1
    notes = [
        f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}",
        "runner=android_cactus_llm_bench",
        f"threads={threads}",
        f"thread_count={threads}",
        *affinity_notes,
        *cactus_benchmark_notes(config, device, runtime, model),
        f"serial={metadata['serial']}",
        f"android={metadata['android_release']}",
        f"thermal_status={metadata['thermal_status']}",
        "fixed token-id fixture",
    ]
    if extra_notes:
        notes.extend(extra_notes)
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def run_android_llama_cpp_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefill_operation, decode_operation in paired_llm_operations(operations):
        if prefill_operation is not None and decode_operation is not None:
            rows.extend(
                run_android_llama_cpp_llm_pair(
                    config,
                    device,
                    runtime,
                    model,
                    prefill_operation,
                    decode_operation,
                    sizes,
                )
            )
        else:
            operation = prefill_operation or decode_operation
            if operation is None:
                continue
            try:
                measurements, metadata = android_llama_cpp_measurements(config, device, runtime, model, operation)
                rows.append(
                    android_llama_cpp_row_from_measurements(
                        config,
                        device,
                        runtime,
                        model,
                        operation,
                        measurements,
                        metadata,
                        sizes,
                    )
                )
            except MatrixRunError as exc:
                if is_llama_cpp_unsupported_error(str(exc)):
                    rows.append(
                        unsupported_row(
                            device,
                            runtime,
                            model,
                            operation,
                            f"configured llama.cpp GGUF is unloadable by the Android runner: {exc}",
                            sizes=sizes,
                        )
                    )
                else:
                    rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb))
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                rows.append(error_row(device, runtime, model, operation, str(exc), sizes=sizes))
    return rows


def run_android_llama_cpp_llm_pair(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    prefill_operation: dict[str, Any],
    decode_operation: dict[str, Any],
    sizes: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    try:
        measurements, metadata = android_llama_cpp_measurements(config, device, runtime, model, decode_operation)
    except MatrixRunError as exc:
        if is_llama_cpp_unsupported_error(str(exc)):
            reason = f"configured llama.cpp GGUF is unloadable by the Android runner: {exc}"
            return [
                unsupported_row(device, runtime, model, prefill_operation, reason, sizes=sizes),
                unsupported_row(device, runtime, model, decode_operation, reason, sizes=sizes),
            ]
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes, peak_ram_mb=exc.peak_process_memory_mb),
        ]
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return [
            error_row(device, runtime, model, prefill_operation, str(exc), sizes=sizes),
            error_row(device, runtime, model, decode_operation, str(exc), sizes=sizes),
        ]

    seqlen = int(prefill_operation["seqlen"])
    return [
        android_llama_cpp_row_from_measurements(
            config,
            device,
            runtime,
            model,
            prefill_operation,
            measurements,
            metadata,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                f"shared_decode_tokens={decode_operation['decode_tokens']}",
            ],
        ),
        android_llama_cpp_row_from_measurements(
            config,
            device,
            runtime,
            model,
            decode_operation,
            measurements,
            metadata,
            sizes,
            extra_notes=[
                "paired_prefill_decode=true",
                f"shared_context={seqlen}",
                "shares measured invocation with prefill row",
            ],
        ),
    ]


def android_llama_cpp_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    serial = select_android_serial_for_device(config, device)
    device_info = android_device_info(serial)
    runner = android_llama_cpp_runner()
    artifact_path = resolve_artifact_path(str(config["models"][model]["artifacts"][runtime]["path"]))
    prepared = prepare_android_llama_cpp(serial=serial, artifact_path=artifact_path, runner=runner)
    runs = config["operations"]["llm"]
    affinity_mask = None if full_core_prefill_mode(config) else android_core_affinity(config, device)[1]
    threads = full_core_threads(config) if full_core_prefill_mode(config) else 1
    for run_index in range(int(runs["warmup_runs"])):
        run_android_llama_cpp_once(serial, prepared, operation, f"warmup_{model}_{operation['operation']}_{operation['seqlen']}_{run_index}", affinity_mask, threads)
    measurements = [
        run_android_llama_cpp_once(serial, prepared, operation, f"measure_{model}_{operation['operation']}_{operation['seqlen']}_{run_index}", affinity_mask, threads)
        for run_index in range(int(runs["measurement_runs"]))
    ]
    return measurements, {"serial": serial, **device_info}


def android_llama_cpp_row_from_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    measurements: list[dict[str, Any]],
    metadata: dict[str, str],
    sizes: tuple[float, float] | None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    runs = config["operations"]["llm"]
    throughputs = [measured_throughput(result, operation=operation) for result in measurements]
    ram_values = [float(result.get("peak_pss_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]

    row = base_row(device, runtime, model, operation)
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    notes = [
        f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}",
        "runner=android_llama_cpp_bench",
        f"benchmark_mode={benchmark_mode(config)}",
        f"llama_cpp_version={config['runtimes'][runtime]['version']}",
        f"threads={full_core_threads(config) if full_core_prefill_mode(config) else 1}",
        f"thread_count={full_core_threads(config) if full_core_prefill_mode(config) else 1}",
        *(["taskset_mask=none", "affinity=not_set"] if full_core_prefill_mode(config) else android_core_affinity_notes(config, device)),
        "mmap=on",
        "n_gpu_layers=0",
        f"serial={metadata['serial']}",
        f"android={metadata['android_release']}",
        f"thermal_status={metadata['thermal_status']}",
        "synthetic exact-length token ids",
    ]
    quantizations = sorted(
        {
            str(result.get("gguf_file_type_name"))
            for result in measurements
            if result.get("gguf_file_type_name")
        }
    )
    if quantizations:
        notes.append(f"quantization={','.join(quantizations)}")
    if extra_notes:
        notes.extend(extra_notes)
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def run_android_transpiled_parakeet(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    artifact_root: Path,
    audio_path: Path,
    reference_path: Path,
    audio_seconds: float,
    sizes: tuple[float, float] | None,
) -> dict[str, Any]:
    if not audio_path.exists():
        return error_row(device, runtime, model, operation, f"audio fixture does not exist: {audio_path}", sizes=sizes)
    if not reference_path.exists():
        return error_row(device, runtime, model, operation, f"reference transcript does not exist: {reference_path}", sizes=sizes)

    try:
        serial = select_android_serial_for_device(config, device)
        device_info = android_device_info(serial)
        runner = android_transpiled_tdt_runner()
        host_inputs = prepare_android_tdt_inputs(artifact_root, audio_path)
        prepared = prepare_android_transpiled_tdt(
            serial=serial,
            artifact_root=artifact_root,
            runner=runner,
            host_inputs=host_inputs,
        )
        runs = config["operations"]["parakeet"]
        _, affinity_mask = android_core_affinity(config, device)
        for run_index in range(int(runs["warmup_runs"])):
            run_android_transpiled_tdt_once(serial, prepared, f"warmup_{run_index}", affinity_mask)
        measurements = [
            run_android_transpiled_tdt_once(serial, prepared, f"measure_{run_index}", affinity_mask)
            for run_index in range(int(runs["measurement_runs"]))
        ]
    except (MatrixRunError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return error_row(device, runtime, model, operation, str(exc), sizes=sizes)

    rtfs = asr_rtfs_from_elapsed(measurements, audio_seconds)
    rtfs = [value for value in rtfs if value > 0.0]
    ram_values = [float(result.get("peak_pss_mb") or 0.0) for result in measurements]
    ram_values = [value for value in ram_values if value > 0.0]
    transcript = normalize_text(str(measurements[-1].get("transcript") or ""))
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)

    row = base_row(device, runtime, model, operation)
    if rtfs:
        row["throughput_tok_per_s"] = f"{statistics.median(rtfs):.6f}"
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    fill_artifact_sizes(row, sizes)
    row["status"] = "ok"
    row["notes"] = "; ".join(
        [
            f"WER={wer:.6f}",
            "runner=android_transpiled_tdt",
            f"timing_source={ASR_TIMING_SOURCE}",
            *android_core_affinity_notes(config, device),
            f"serial={serial}",
            f"android={device_info['android_release']}",
            f"thermal_status={device_info['thermal_status']}",
            f"decoder_steps={measurements[-1].get('decoder_steps', '')}",
            f"transcript={transcript}",
        ]
    )
    return row


def android_transpiled_tdt_runner() -> Path:
    binary = REPO_ROOT / "tests" / "android" / "build_transpiled" / "transpiled_tdt"
    source = REPO_ROOT / "tests" / "android" / "transpiled_tdt.cpp"
    cmake_file = REPO_ROOT / "tests" / "android" / "CMakeLists.txt"
    inputs = [source, cmake_file]
    for root in (REPO_ROOT / "cactus-graph", REPO_ROOT / "cactus-kernels"):
        inputs.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and (path.suffix in {".cpp", ".h", ".hpp", ".cc", ".cxx", ".cmake"} or path.name == "CMakeLists.txt")
        )
    if binary.exists() and all(binary.stat().st_mtime >= path.stat().st_mtime for path in inputs):
        return binary

    ndk_home = os.environ.get("ANDROID_NDK_HOME")
    if not ndk_home:
        android_home = os.environ.get("ANDROID_HOME") or str(Path.home() / "Library" / "Android" / "sdk")
        ndk_root = Path(android_home) / "ndk"
        if ndk_root.exists():
            versions = sorted(path for path in ndk_root.iterdir() if path.is_dir())
            if versions:
                ndk_home = str(versions[-1])
    if not ndk_home and Path("/opt/homebrew/share/android-ndk").exists():
        ndk_home = "/opt/homebrew/share/android-ndk"
    if not ndk_home:
        brew_root = Path("/opt/homebrew/Caskroom/android-ndk")
        if brew_root.exists():
            versions = sorted(brew_root.glob("*/AndroidNDK*.app/Contents/NDK"))
            if versions:
                ndk_home = str(versions[-1])
    if not ndk_home:
        raise RuntimeError("Android NDK not found; set ANDROID_NDK_HOME")
    toolchain = Path(ndk_home) / "build" / "cmake" / "android.toolchain.cmake"
    if not toolchain.exists():
        raise RuntimeError(f"Android CMake toolchain not found: {toolchain}")

    build_dir = binary.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "tests" / "android"),
            "-B",
            str(build_dir),
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            "-DANDROID_ABI=arm64-v8a",
            f"-DANDROID_PLATFORM={os.environ.get('ANDROID_PLATFORM', 'android-21')}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "transpiled_tdt", "-j", str(os.cpu_count() or 4)],
        cwd=REPO_ROOT,
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"Android transpiled TDT runner was not built: {binary}")
    return binary


def android_native_transcribe_runner() -> Path:
    binary = REPO_ROOT / "android" / "build" / "native_transcribe_json"
    inputs = [
        REPO_ROOT / "android" / "CMakeLists.txt",
        REPO_ROOT / "experiments" / "matrix" / "native_transcribe_json.cpp",
        REPO_ROOT / "cactus-engine" / "cactus_engine.h",
        REPO_ROOT / "cactus-engine" / "src" / "transcribe.cpp",
        REPO_ROOT / "cactus-engine" / "src" / "model.cpp",
        REPO_ROOT / "cactus-engine" / "src" / "engine.h",
    ]
    if binary.exists() and all(path.exists() and binary.stat().st_mtime >= path.stat().st_mtime for path in inputs):
        return binary
    subprocess.run(
        ["cmake", "--build", str(binary.parent), "--target", "native_transcribe_json", "-j", str(os.cpu_count() or 4)],
        cwd=REPO_ROOT,
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"Android native transcribe runner was not built: {binary}")
    return binary


def android_cactus_llm_runner() -> Path:
    binary = REPO_ROOT / "android" / "build" / "cactus_llm_bench"
    inputs = [
        REPO_ROOT / "android" / "CMakeLists.txt",
        REPO_ROOT / "tests" / "android" / "cactus_llm_bench.cpp",
        REPO_ROOT / "cactus-engine" / "cactus_engine.h",
        REPO_ROOT / "cactus-engine" / "src" / "complete.cpp",
        REPO_ROOT / "cactus-engine" / "src" / "model.cpp",
        REPO_ROOT / "cactus-engine" / "src" / "engine.h",
    ]
    if binary.exists() and all(binary.stat().st_mtime >= path.stat().st_mtime for path in inputs):
        return binary
    subprocess.run(
        ["cmake", "--build", str(binary.parent), "--target", "cactus_llm_bench", "-j", str(os.cpu_count() or 4)],
        cwd=REPO_ROOT,
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"Android Cactus LLM runner was not built: {binary}")
    return binary


def android_llama_cpp_source_root() -> Path:
    return Path(os.environ.get("LLAMA_CPP_ROOT", "/Users/noahcylich/Documents/Desert/third_party/llama.cpp")).expanduser()


def find_android_ndk_home() -> Path:
    candidates: list[Path] = []
    if os.environ.get("ANDROID_NDK_HOME"):
        candidates.append(Path(str(os.environ["ANDROID_NDK_HOME"])))
    android_home = os.environ.get("ANDROID_HOME") or str(Path.home() / "Library" / "Android" / "sdk")
    ndk_root = Path(android_home) / "ndk"
    if ndk_root.exists():
        candidates.extend(sorted(path for path in ndk_root.iterdir() if path.is_dir()))
    brew_root = Path("/opt/homebrew/Caskroom/android-ndk")
    if brew_root.exists():
        candidates.extend(sorted(brew_root.glob("*/AndroidNDK*.app/Contents/NDK")))
    if Path("/opt/homebrew/share/android-ndk").exists():
        candidates.append(Path("/opt/homebrew/share/android-ndk"))
    for candidate in reversed(candidates):
        if (candidate / "build" / "cmake" / "android.toolchain.cmake").exists():
            return candidate
    raise RuntimeError("Android NDK not found; set ANDROID_NDK_HOME")


def android_llama_cpp_runner() -> Path:
    binary = REPO_ROOT / "tests" / "android" / "build_llama_cpp" / "llama_cpp_bench"
    source_root = android_llama_cpp_source_root()
    inputs = [
        REPO_ROOT / "tests" / "android" / "llama_cpp" / "CMakeLists.txt",
        REPO_ROOT / "experiments" / "matrix" / "native_llama_cpp_json.cpp",
        source_root / "include" / "llama.h",
        source_root / "src" / "llama.cpp",
    ]
    if binary.exists() and all(path.exists() and binary.stat().st_mtime >= path.stat().st_mtime for path in inputs):
        return binary

    ndk_home = find_android_ndk_home()
    toolchain = ndk_home / "build" / "cmake" / "android.toolchain.cmake"
    build_dir = binary.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "tests" / "android" / "llama_cpp"),
            "-B",
            str(build_dir),
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            "-DANDROID_ABI=arm64-v8a",
            f"-DANDROID_PLATFORM={os.environ.get('ANDROID_PLATFORM', 'android-23')}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLLAMA_CPP_ROOT={source_root}",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "llama_cpp_bench", "-j", str(os.cpu_count() or 4)],
        cwd=REPO_ROOT,
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"Android llama.cpp runner was not built: {binary}")
    return binary


def prepare_android_tdt_inputs(artifact_root: Path, audio_path: Path) -> dict[str, Path]:
    import numpy as np
    import torch
    from cactus.transpile.audio_preprocess import prepare_native_parakeet_audio_features

    out_dir = REPO_ROOT / "experiments" / "matrix" / "results" / "android_tdt_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "components" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_shapes = manifest.get("inputs", {}).get("input_shapes", {})
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise RuntimeError("Parakeet TDT manifest is missing inputs.input_shapes.input_features")

    features_path = out_dir / "input_features.weights"
    bindings_path = out_dir / "bindings.tsv"
    if not features_path.exists() or features_path.stat().st_mtime < audio_path.stat().st_mtime:
        features, _ = prepare_native_parakeet_audio_features(
            audio_path,
            expected_frames=int(expected_shape[1]),
            expected_mels=int(expected_shape[2]),
            torch_dtype=torch.float16,
        )
        array = np.ascontiguousarray(features.cpu().numpy().astype(np.float16, copy=False))
        write_cactus_tensor(features_path, array, precision=1)

    if not bindings_path.exists() or bindings_path.stat().st_mtime < manifest_path.stat().st_mtime:
        lines: list[str] = []
        for component in manifest.get("components", []):
            component_name = str(component.get("component") or "")
            for binding in component.get("bound_constant_bindings") or []:
                raw_path = Path(str(binding["path"]))
                relative = raw_path.expanduser().resolve().relative_to(artifact_root) if raw_path.is_absolute() else raw_path
                target = artifact_root / relative
                if not target.exists():
                    raise FileNotFoundError(target)
                lines.append(f"{component_name}\t{int(binding['node_id'])}\t{relative.as_posix()}")
        bindings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"features": features_path, "bindings": bindings_path}


def write_cactus_tensor(path: Path, array: Any, *, precision: int) -> None:
    shape = [int(dim) for dim in array.shape]
    rank = len(shape)
    flags = 1 << 4 if rank > 4 else 0
    alignment = 32
    header_size = 84 + (32 if flags else 0)
    with path.open("wb") as handle:
        handle.write(b"CACT")
        handle.write(struct.pack("<I", flags))
        handle.write(struct.pack("<I", alignment))
        handle.write(struct.pack("<I", rank))
        for index in range(4):
            handle.write(struct.pack("<Q", shape[index] if index < rank else 0))
        handle.write(struct.pack("<I", precision))
        handle.write(struct.pack("<Q", int(array.nbytes)))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<I", 0))
        handle.write(struct.pack("<I", 0))
        handle.write(struct.pack("<Q", shape[0] if shape else 0))
        if flags:
            for index in range(4, 8):
                handle.write(struct.pack("<Q", shape[index] if index < rank else 0))
        handle.write(b"\0" * ((alignment - (header_size % alignment)) % alignment))
        handle.write(array.tobytes())


def attached_android_devices() -> list[tuple[str, str]]:
    global _ANDROID_DEVICE_CACHE
    if _ANDROID_DEVICE_CACHE is not None:
        return _ANDROID_DEVICE_CACHE
    completed = subprocess.run(
        ["adb", "devices", "-l"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "adb devices failed")
    candidates: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            candidates.append((parts[0], line))
    _ANDROID_DEVICE_CACHE = candidates
    return candidates


def android_device_unavailable_reason(config: dict[str, Any], device: str) -> str | None:
    device_config = config["devices"][device]
    try:
        candidates = attached_android_devices()
    except RuntimeError as exc:
        detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else "adb devices failed"
        return f"adb device readiness check failed: {detail}"
    if not candidates:
        return f"{device} is not attached or authorized over adb"

    configured_serial = device_config.get("serial")
    if configured_serial:
        for serial, _ in candidates:
            if serial == configured_serial:
                return None
        return f"{device} configured serial {configured_serial} is not attached or authorized over adb"

    exact_model = str(device_config.get("exact_model") or "")
    if exact_model:
        model_token = "model:" + exact_model.replace(" ", "_")
        for _, line in candidates:
            if model_token in line:
                return None
        return f"{device} exact model {exact_model} is not attached or authorized over adb"

    return f"{device} has no configured Android serial or exact model for deterministic adb selection"


def select_android_serial_for_device(config: dict[str, Any], device: str) -> str:
    reason = android_device_unavailable_reason(config, device)
    if reason is not None:
        raise RuntimeError(reason)

    device_config = config["devices"][device]
    candidates = attached_android_devices()
    configured_serial = device_config.get("serial")
    if configured_serial:
        return str(configured_serial)

    exact_model = str(device_config.get("exact_model") or "")
    model_token = "model:" + exact_model.replace(" ", "_")
    for serial, line in candidates:
        if model_token in line:
            return serial
    raise RuntimeError(f"could not select Android serial for {device}")


def android_core_affinity(config: dict[str, Any], device: str) -> tuple[int, str]:
    device_config = config["devices"][device]
    cpu = int(device_config.get("core_affinity_cpu") or 7)
    mask = str(device_config.get("core_affinity_mask") or format(1 << cpu, "x"))
    return cpu, mask


def android_core_affinity_notes(config: dict[str, Any], device: str) -> list[str]:
    cpu, mask = android_core_affinity(config, device)
    return [f"core_affinity=cpu{cpu}", f"taskset_mask={mask}"]


def android_taskset_invocation(invocation: str, affinity_mask: str) -> str:
    command = f"taskset -p {shlex.quote(affinity_mask)} $$ >/dev/null && exec {invocation}"
    return f"sh -c {shlex.quote(command)}"


def android_device_info(serial: str) -> dict[str, str]:
    def shell_output(command: str) -> str:
        completed = adb_checked(serial, "shell", command)
        return completed.stdout.strip()

    thermal = shell_output("dumpsys thermalservice")
    thermal_status = ""
    for line in thermal.splitlines():
        stripped = line.strip()
        if stripped.startswith("Thermal Status:"):
            thermal_status = stripped.split(":", 1)[1].strip()
            break
    return {
        "model": shell_output("getprop ro.product.model"),
        "android_release": shell_output("getprop ro.build.version.release"),
        "fingerprint": shell_output("getprop ro.build.fingerprint"),
        "thermal_status": thermal_status or "unknown",
    }


def prepare_android_transpiled_tdt(*, serial: str, artifact_root: Path, runner: Path, host_inputs: dict[str, Path]) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix"
    device_bin = f"{device_root}/bin/transpiled_tdt"
    device_model_root = f"{device_root}/models"
    device_model = f"{device_model_root}/{artifact_root.name}"
    device_inputs = f"{device_root}/tdt_inputs"
    device_logs = f"{device_root}/logs"
    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_root)}/bin {shlex.quote(device_model_root)} {shlex.quote(device_inputs)} {shlex.quote(device_logs)}")
    adb_checked(serial, "push", str(runner), device_bin)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_bin)}")
    if adb_checked(serial, "shell", f"test -f {shlex.quote(device_model)}/.cactus_matrix_deployed", check=False).returncode != 0:
        adb_push_directory_dereferenced(serial, artifact_root, device_model_root)
    adb_checked(serial, "push", str(host_inputs["features"]), f"{device_inputs}/input_features.weights")
    adb_checked(serial, "push", str(host_inputs["bindings"]), f"{device_inputs}/bindings.tsv")
    return {
        "runner": device_bin,
        "model": device_model,
        "features": f"{device_inputs}/input_features.weights",
        "bindings": f"{device_inputs}/bindings.tsv",
        "logs": device_logs,
    }


def prepare_android_native_transcribe(*, serial: str, artifact_root: Path, runner: Path, audio_path: Path) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix"
    device_bin = f"{device_root}/bin/native_transcribe_json"
    device_model_root = f"{device_root}/models"
    device_model = f"{device_model_root}/{artifact_root.name}"
    device_inputs = f"{device_root}/asr_inputs"
    device_logs = f"{device_root}/logs"
    device_audio = f"{device_inputs}/{audio_path.name}"
    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_root)}/bin {shlex.quote(device_model_root)} {shlex.quote(device_inputs)} {shlex.quote(device_logs)}")
    adb_checked(serial, "push", str(runner), device_bin)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_bin)}")
    if adb_checked(serial, "shell", f"test -f {shlex.quote(device_model)}/.cactus_matrix_deployed", check=False).returncode != 0:
        adb_push_directory_dereferenced(serial, artifact_root, device_model_root)
    adb_push_if_needed(serial, audio_path, device_audio)
    return {
        "runner": device_bin,
        "model": device_model,
        "audio": device_audio,
        "logs": device_logs,
    }


def prepare_android_cactus_llm(*, serial: str, artifact_root: Path, runner: Path, operation: dict[str, Any]) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix"
    device_bin = f"{device_root}/bin/cactus_llm_bench"
    device_model_root = f"{device_root}/models"
    device_model = f"{device_model_root}/{artifact_root.name}"
    device_inputs = f"{device_root}/inputs"
    device_logs = f"{device_root}/logs"
    input_path = None
    if operation.get("input_path"):
        input_path = resolve_repo_path(str(operation["input_path"]))
        device_input = f"{device_inputs}/{input_path.name}"
    else:
        device_input = f"{device_inputs}/seqlen_{operation['seqlen']}_default_token2.csv"
    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_root)}/bin {shlex.quote(device_model_root)} {shlex.quote(device_inputs)} {shlex.quote(device_logs)}")
    adb_checked(serial, "push", str(runner), device_bin)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_bin)}")
    if adb_checked(serial, "shell", f"test -f {shlex.quote(device_model)}/.cactus_matrix_deployed", check=False).returncode != 0:
        adb_push_directory_dereferenced(serial, artifact_root, device_model_root)
    if input_path is not None:
        adb_checked(serial, "push", str(input_path), device_input)
    else:
        adb_checked(serial, "shell", f"printf '%s' {shlex.quote(input_ids_for_operation(operation))} > {shlex.quote(device_input)}")
    return {
        "runner": device_bin,
        "model": device_model,
        "input": device_input,
        "logs": device_logs,
    }


def prepare_android_llama_cpp(*, serial: str, artifact_path: Path, runner: Path) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix"
    device_bin = f"{device_root}/bin/llama_cpp_bench"
    device_model_root = f"{device_root}/models"
    device_model = f"{device_model_root}/{artifact_path.name}"
    device_logs = f"{device_root}/logs"
    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_root)}/bin {shlex.quote(device_model_root)} {shlex.quote(device_logs)}")
    adb_checked(serial, "push", str(runner), device_bin)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_bin)}")
    if adb_checked(serial, "shell", f"test -f {shlex.quote(device_model)}", check=False).returncode != 0:
        adb_checked(serial, "push", str(artifact_path), device_model)
    return {
        "runner": device_bin,
        "model": device_model,
        "logs": device_logs,
    }


def prepare_android_executorch_parakeet(
    *,
    serial: str,
    runner: Path,
    model_path: Path,
    tokenizer_path: Path,
    audio_path: Path,
) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix/executorch_parakeet"
    device_bin_dir = f"{device_root}/bin"
    device_model_dir = f"{device_root}/models/parakeet_tdt_v3"
    device_inputs = f"{device_root}/inputs"
    device_logs = f"{device_root}/logs"
    device_runner = f"{device_bin_dir}/{runner.name}"
    device_model = f"{device_model_dir}/{model_path.name}"
    device_tokenizer = f"{device_model_dir}/{tokenizer_path.name}"
    device_audio = f"{device_inputs}/{audio_path.name}"

    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_bin_dir)} {shlex.quote(device_model_dir)} {shlex.quote(device_inputs)} {shlex.quote(device_logs)}")
    adb_push_if_needed(serial, runner, device_runner)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_runner)}")
    adb_push_if_needed(serial, model_path, device_model)
    adb_push_if_needed(serial, tokenizer_path, device_tokenizer)
    adb_push_if_needed(serial, audio_path, device_audio)
    return {
        "runner": device_runner,
        "model": device_model,
        "tokenizer": device_tokenizer,
        "audio": device_audio,
        "logs": device_logs,
    }


def prepare_litert_parakeet_feature_inputs(artifact: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from scipy.io import wavfile
    from cactus.transpile.audio_preprocess import load_audio_waveform, prepare_native_parakeet_audio_features

    out_dir = REPO_ROOT / "experiments" / "matrix" / "results" / "android_litert_parakeet_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_chunk_seconds = float(artifact.get("max_chunk_seconds") or 5.0)
    waveform = load_audio_waveform(audio_path, target_sample_rate=16000, max_seconds=None)
    chunk_samples = max(1, int(round(16000.0 * max_chunk_seconds)))
    chunks: list[dict[str, Path]] = []
    for chunk_index, start in enumerate(range(0, len(waveform), chunk_samples)):
        chunk = waveform[start : start + chunk_samples]
        if chunk.size == 0:
            continue
        stem = f"{audio_path.stem}_chunk_{chunk_index}"
        chunk_path = out_dir / f"{stem}.wav"
        features_path = out_dir / f"{stem}.f32"
        active_frames_path = out_dir / f"{stem}.active_i64"
        wavfile.write(str(chunk_path), 16000, (np.clip(chunk, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16))
        features, active_frames = prepare_native_parakeet_audio_features(
            chunk_path,
            expected_frames=500,
            expected_mels=128,
            torch_dtype=torch.float32,
        )
        mel = np.ascontiguousarray(features.cpu().numpy().transpose(0, 2, 1).astype(np.float32))
        features_path.write_bytes(mel.tobytes())
        active_frames_path.write_bytes(np.array([min(int(active_frames), 500)], dtype=np.int64).tobytes())
        chunks.append({"features": features_path, "active_frames": active_frames_path})
    if not chunks:
        raise MatrixRunError(f"no LiteRT Parakeet chunks produced for {audio_path}")

    vocab_path = resolve_repo_path(str(artifact["vocab_path"]))
    vocab_tsv = out_dir / "vocab.tsv"
    vocab = {int(key): str(value) for key, value in json.loads(vocab_path.read_text(encoding="utf-8")).items()}
    vocab_tsv.write_text("".join(f"{key}\t{value}\n" for key, value in sorted(vocab.items())), encoding="utf-8")
    return {"chunks": chunks, "vocab_tsv": vocab_tsv}


def prepare_android_litert_parakeet(
    *,
    serial: str,
    artifact: dict[str, Any],
    runner: Path,
    host_inputs: dict[str, Any],
) -> dict[str, str]:
    device_root = "/data/local/tmp/cactus_matrix/litert_parakeet"
    device_bin_dir = f"{device_root}/bin"
    device_model_dir = f"{device_root}/models"
    device_inputs = f"{device_root}/inputs"
    device_logs = f"{device_root}/logs"
    device_runner = f"{device_bin_dir}/{runner.name}"
    encoder_path = resolve_repo_path(str(artifact["encoder_path"]))
    decoder_path = resolve_repo_path(str(artifact["decoder_joint_path"]))
    device_encoder = f"{device_model_dir}/{encoder_path.name}"
    device_decoder = f"{device_model_dir}/{decoder_path.name}"
    device_vocab = f"{device_inputs}/vocab.tsv"
    device_manifest = f"{device_inputs}/inputs.tsv"

    adb_checked(serial, "shell", f"mkdir -p {shlex.quote(device_bin_dir)} {shlex.quote(device_model_dir)} {shlex.quote(device_inputs)} {shlex.quote(device_logs)}")
    adb_push_if_needed(serial, runner, device_runner)
    adb_checked(serial, "shell", f"chmod +x {shlex.quote(device_runner)}")
    adb_push_if_needed(serial, encoder_path, device_encoder)
    adb_push_if_needed(serial, decoder_path, device_decoder)
    adb_push_if_needed(serial, Path(str(host_inputs["vocab_tsv"])), device_vocab)

    manifest_lines: list[str] = []
    for chunk_index, chunk in enumerate(host_inputs["chunks"]):
        feature_path = Path(str(chunk["features"]))
        active_frames_path = Path(str(chunk["active_frames"]))
        device_features = f"{device_inputs}/{feature_path.name}"
        device_active_frames = f"{device_inputs}/{active_frames_path.name}"
        adb_push_if_needed(serial, feature_path, device_features)
        adb_push_if_needed(serial, active_frames_path, device_active_frames)
        manifest_lines.append(f"{device_features}\t{device_active_frames}")
    manifest_host = REPO_ROOT / "experiments" / "matrix" / "results" / "android_litert_parakeet_inputs" / "device_inputs.tsv"
    manifest_host.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    adb_push_if_needed(serial, manifest_host, device_manifest)
    return {
        "runner": device_runner,
        "encoder": device_encoder,
        "decoder": device_decoder,
        "inputs_tsv": device_manifest,
        "vocab_tsv": device_vocab,
        "logs": device_logs,
    }


def adb_checked(serial: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["adb", "-s", serial, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"adb {' '.join(args)} failed")
    return completed


def rewrite_android_cactus_weight_paths(root: Path, device_model: str) -> list[Path]:
    weights_root = REPO_ROOT / "weights"
    required_dirs: set[Path] = set()
    replacements: dict[str, str] = {}
    for metadata_path in root.rglob("*"):
        if metadata_path.suffix not in {".json", ".txt"} or not metadata_path.is_file():
            continue
        text = metadata_path.read_text()
        for match in re.findall(rf"{re.escape(str(weights_root))}/([^/\"'\\s]+)", text):
            source_dir = weights_root / match
            if source_dir.is_dir():
                required_dirs.add(source_dir)
                replacements[str(source_dir)] = f"{device_model}/weights/{source_dir.name}"
        for local_path, device_path in replacements.items():
            text = text.replace(local_path, device_path)
        metadata_path.write_text(text)
    return sorted(required_dirs)


def adb_push_directory_dereferenced(serial: str, source: Path, destination_parent: str) -> None:
    destination = f"{destination_parent}/{source.name}"
    with tempfile.TemporaryDirectory(prefix="cactus_matrix_push_") as temp_dir:
        dereferenced = Path(temp_dir) / source.name
        shutil.copytree(source, dereferenced, symlinks=False)
        required_weight_dirs = rewrite_android_cactus_weight_paths(dereferenced, destination)
        adb_checked(serial, "shell", f"rm -rf {shlex.quote(destination)}")
        adb_checked(serial, "push", str(dereferenced), destination_parent)
        if required_weight_dirs:
            adb_checked(serial, "shell", f"mkdir -p {shlex.quote(destination)}/weights")
            for weight_dir in required_weight_dirs:
                adb_checked(serial, "push", str(weight_dir), f"{destination}/weights")
        adb_checked(serial, "shell", f"touch {shlex.quote(destination)}/.cactus_matrix_deployed")


def adb_push_if_needed(serial: str, source: Path, destination: str) -> None:
    stat = adb_checked(serial, "shell", f"stat -c %s {shlex.quote(destination)} 2>/dev/null || true")
    if stat.stdout.strip() == str(source.stat().st_size):
        return
    adb_checked(serial, "push", str(source), destination)


def run_android_transpiled_tdt_once(serial: str, prepared: dict[str, str], run_name: str, affinity_mask: str) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    invocation = f"{shlex.quote(prepared['runner'])} {shlex.quote(prepared['model'])} {shlex.quote(prepared['features'])} {shlex.quote(prepared['bindings'])}"
    invocation = android_taskset_invocation(invocation, affinity_mask)
    return run_android_logged_json_once(serial, invocation, log_path)


def run_android_native_transcribe_once(serial: str, prepared: dict[str, str], run_name: str, affinity_mask: str) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    options = '{"max_tokens":500,"telemetry_enabled":false,"auto_handoff":false}'
    invocation = (
        f"{shlex.quote(prepared['runner'])} {shlex.quote(prepared['model'])} "
        f"{shlex.quote(prepared['audio'])} {shlex.quote(options)}"
    )
    invocation = android_taskset_invocation(invocation, affinity_mask)
    return run_android_logged_json_once(serial, invocation, log_path)


def run_android_cactus_llm_once(
    serial: str,
    prepared: dict[str, str],
    operation: dict[str, Any],
    run_name: str,
    affinity_mask: str | None,
) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    invocation = (
        f"{shlex.quote(prepared['runner'])} {shlex.quote(prepared['model'])} "
        f"{shlex.quote(prepared['input'])} {int(operation['decode_tokens'])}"
    )
    if full_core_env_value := prepared.get("full_core_threads"):
        invocation = " ".join(
            [
                *(f"{name}={shlex.quote(full_core_env_value)}" for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "XNNPACK_NUM_THREADS", "CACTUS_MATRIX_BENCH_THREADS")),
                "CACTUS_THREADPOOL_PIN_MAX_PERF_ONLY=1",
                "CACTUS_BENCH_PIN_MAIN_MAX_PERF=1",
            ]
        ) + f" {invocation}"
    if affinity_mask is not None:
        invocation = android_taskset_invocation(invocation, affinity_mask)
    return run_android_logged_json_once(serial, invocation, log_path)


def run_android_llama_cpp_once(
    serial: str,
    prepared: dict[str, str],
    operation: dict[str, Any],
    run_name: str,
    affinity_mask: str | None,
    threads: int,
) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    invocation = (
        f"{shlex.quote(prepared['runner'])} {shlex.quote(prepared['model'])} "
        f"{int(operation['seqlen'])} {int(operation['decode_tokens'])} {int(threads)}"
    )
    if affinity_mask is not None:
        invocation = android_taskset_invocation(invocation, affinity_mask)
    return run_android_logged_json_once(serial, invocation, log_path)


def run_android_executorch_parakeet_once(serial: str, prepared: dict[str, str], run_name: str, affinity_mask: str) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    invocation = (
        f"{shlex.quote(prepared['runner'])} "
        f"-model_path={shlex.quote(prepared['model'])} "
        f"-tokenizer_path={shlex.quote(prepared['tokenizer'])} "
        f"-audio_path={shlex.quote(prepared['audio'])} "
        "-timestamps=none"
    )
    invocation = android_taskset_invocation(invocation, affinity_mask)
    result = run_android_logged_text_once(serial, invocation, log_path)
    result["transcript"] = parse_executorch_parakeet_transcript(result["stdout"] + "\n" + result["stderr"])
    return result


def run_android_litert_parakeet_once(serial: str, prepared: dict[str, str], run_name: str, affinity_mask: str) -> dict[str, Any]:
    log_path = f"{prepared['logs']}/{run_name}.log"
    invocation = (
        f"{shlex.quote(prepared['runner'])} "
        f"--encoder={shlex.quote(prepared['encoder'])} "
        f"--decoder={shlex.quote(prepared['decoder'])} "
        f"--inputs_tsv={shlex.quote(prepared['inputs_tsv'])} "
        f"--vocab_tsv={shlex.quote(prepared['vocab_tsv'])}"
    )
    invocation = android_taskset_invocation(invocation, affinity_mask)
    return run_android_logged_json_once(serial, invocation, log_path)


def run_android_logged_text_once(serial: str, invocation: str, log_path: str) -> dict[str, Any]:
    command = (
        f"rm -f {shlex.quote(log_path)}; "
        f"{invocation} > {shlex.quote(log_path)} 2>&1 & "
        "pid=$!; "
        "echo __CACTUS_MATRIX_PID__:$pid; "
        "wait $pid; "
        "rc=$?; "
        "echo __CACTUS_MATRIX_RC__:$rc; "
        f"cat {shlex.quote(log_path)}; "
        "exit $rc"
    )
    process = subprocess.Popen(
        ["adb", "-s", serial, "shell", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise MatrixRunError("failed to capture adb stdout")

    started = time.perf_counter()
    stdout_lines: list[str] = []
    device_pid = 0
    deadline = started + 10.0
    while time.perf_counter() < deadline:
        line = process.stdout.readline()
        if line:
            stdout_lines.append(line)
            match = re.search(r"__CACTUS_MATRIX_PID__:(\d+)", line)
            if match:
                device_pid = int(match.group(1))
                break
        elif process.poll() is not None:
            break
    if device_pid <= 0:
        stdout, stderr = process.communicate()
        raise MatrixRunError((stderr.strip() or stdout.strip() or "Android runner did not report a PID").splitlines()[-1])

    peak_pss_mb = 0.0
    while process.poll() is None:
        try:
            peak_pss_mb = max(peak_pss_mb, sample_android_pss_mb(adb="adb", serial=serial, pid=device_pid))
        except (RuntimeError, ValueError):
            pass
        time.sleep(0.1)
    try:
        peak_pss_mb = max(peak_pss_mb, sample_android_pss_mb(adb="adb", serial=serial, pid=device_pid))
    except (RuntimeError, ValueError):
        pass

    stdout_tail, stderr = process.communicate()
    stdout = "".join(stdout_lines) + stdout_tail
    elapsed_seconds = time.perf_counter() - started
    if process.returncode != 0:
        detail = (stderr.strip() or stdout.strip()).splitlines()
        raise MatrixRunError(detail[-1] if detail else f"Android runner exited {process.returncode}", peak_process_memory_mb=peak_pss_mb)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_seconds": elapsed_seconds,
        "peak_pss_mb": peak_pss_mb,
        "peak_process_memory_mb": peak_pss_mb,
    }


def run_android_logged_json_once(serial: str, invocation: str, log_path: str) -> dict[str, Any]:
    result = run_android_logged_text_once(serial, invocation, log_path)
    payload = parse_last_json_object(result["stdout"])
    return {**payload, **result}


def parakeet_transpiled_artifact_path(artifact_root: Path) -> Path | None:
    candidates = [
        artifact_root,
        artifact_root.with_name(f"{artifact_root.name}-transpiled"),
    ]
    for candidate in candidates:
        if (candidate / "components" / "manifest.json").exists():
            return candidate
    return None


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_transcribe_once(command: list[str]) -> dict[str, Any]:
    completed = run_with_peak_memory(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        message = detail[-1] if detail else f"command exited {completed.returncode}"
        raise MatrixRunError(message, peak_process_memory_mb=completed.peak_process_memory_mb)
    payload = parse_last_json_object(completed.stdout)
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        **payload,
        "peak_process_memory_mb": completed.peak_process_memory_mb,
        "elapsed_seconds": completed.elapsed_seconds,
    }


def parse_last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError("native transcribe JSON was not an object")
            return loaded
    raise ValueError("native transcribe output did not contain a JSON object")


def native_transcribe_runner() -> Path:
    source = REPO_ROOT / "experiments" / "matrix" / "native_transcribe_json.cpp"
    build_dir = REPO_ROOT / "experiments" / "matrix" / ".build"
    binary = build_dir / "native_transcribe_json"
    lib_path = REPO_ROOT / "cactus" / "build" / "libcactus.a"
    if (
        binary.exists()
        and binary.stat().st_mtime >= source.stat().st_mtime
        and binary.stat().st_mtime >= lib_path.stat().st_mtime
    ):
        return binary

    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "clang++" if platform.system() == "Darwin" else "g++")
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        *(["-DACCELERATE_NEW_LAPACK"] if platform.system() == "Darwin" else []),
        f"-I{REPO_ROOT}",
        f"-I{REPO_ROOT / 'cactus-engine'}",
        f"-I{REPO_ROOT / 'cactus-graph'}",
        f"-I{REPO_ROOT / 'cactus-kernels'}",
        str(source),
        str(lib_path),
        "-o",
        str(binary),
    ]
    vendored_curl = REPO_ROOT / "cactus-engine" / "libs" / "curl" / "lib" / "libcurl.a"
    if platform.system() == "Darwin":
        command.extend(
            [
                str(vendored_curl) if vendored_curl.exists() else "-lcurl",
                "-framework",
                "Accelerate",
                "-framework",
                "CoreML",
                "-framework",
                "Foundation",
                "-framework",
                "Security",
                "-framework",
                "SystemConfiguration",
                "-framework",
                "CFNetwork",
            ]
        )
    else:
        command.extend(["-lcurl", "-pthread"])

    completed = subprocess.run(
        command,
        cwd=str(build_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "native transcribe runner build failed")
    return binary


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1] / len(ref_words)


def measured_throughput(result: dict[str, Any], *, operation: dict[str, Any]) -> float:
    if operation["operation"] == "decode":
        return float(result.get("decode_tps") or 0.0)
    reported_compute = result.get("prefill_compute_tps")
    if reported_compute:
        return float(reported_compute)
    cache_prime_tokens = result.get("cache_prime_tokens")
    cache_prime_compute_ms = result.get("cache_prime_compute_ms")
    if cache_prime_tokens and cache_prime_compute_ms:
        return (float(cache_prime_tokens) * 1000.0) / float(cache_prime_compute_ms)
    cache_prime_ms = result.get("cache_prime_ms")
    if cache_prime_tokens and cache_prime_ms:
        return (float(cache_prime_tokens) * 1000.0) / float(cache_prime_ms)
    reported = result.get("prefill_tps")
    if reported:
        return float(reported)
    ttft_ms = result.get("time_to_first_token_ms")
    if ttft_ms:
        return (float(operation["seqlen"]) * 1000.0) / float(ttft_ms)
    return 0.0


def actual_token_note(measurements: list[dict[str, Any]], operation: dict[str, Any]) -> str:
    actual_values = {
        int(value)
        for result in measurements
        for value in (result.get("cache_prime_tokens"),)
        if value
    }
    if not actual_values:
        return ""
    expected = int(operation["seqlen"])
    if actual_values == {expected}:
        return ""
    if actual_values == {expected - 1}:
        return (
            f"paired invocation cache-primed {expected - 1} tokens; "
            "final prompt token is included in first-token timing"
        )
    actual = ",".join(str(value) for value in sorted(actual_values))
    return (
        f"requested token fixture seqlen={expected}; bundle used cache_prime_tokens={actual}; "
        "exact token-id input is unsupported for this multimodal bundle"
    )


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [field for field in CSV_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} missing fields: {', '.join(missing)}")
        if row["status"] not in STATUSES:
            raise ValueError(f"row {index} has invalid status {row['status']!r}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.out)
    config = load_config(config_path)
    apply_run_overrides(config, args)

    devices = selected_ids(config, "devices", args.device)
    runtimes = selected_ids(config, "runtimes", args.runtime)
    models = selected_ids(config, "models", args.model)

    rows: list[dict[str, Any]] = []
    for device in devices:
        for runtime in runtimes:
            for model in models:
                operations = filtered_operations(
                    operations_for_model(config, model),
                    args.operation,
                    args.seqlen,
                )
                if full_core_prefill_mode(config) and config["models"][model].get("type") == "llm":
                    operations = [operation for operation in operations if operation["operation"] == "prefill"]
                elif full_core_prefill_mode(config):
                    operations = []
                if not operations:
                    continue
                reason = unsupported_reason(config, device, runtime, model)
                sizes = None
                if reason is None:
                    try:
                        sizes = collect_artifact_sizes(config, runtime, model)
                    except FileNotFoundError as exc:
                        reason = f"artifact path does not exist: {exc}"
                    except (OSError, RuntimeError) as exc:
                        rows.extend(
                            error_row(
                                device,
                                runtime,
                                model,
                                operation,
                                f"size collection failed: {exc}",
                            )
                            for operation in operations
                        )
                        continue
                if reason is not None:
                    extra_notes = []
                    if full_core_prefill_mode(config):
                        extra_notes = [
                            f"benchmark_mode={benchmark_mode(config)}",
                            "provider=cpu",
                            "gpu=disabled",
                            f"threads={full_core_threads(config)}",
                            f"thread_count={full_core_threads(config)}",
                            "affinity=not_set",
                            "taskset_mask=none",
                        ]
                    rows.extend(
                        unsupported_row(device, runtime, model, operation, reason, sizes=sizes, extra_notes=extra_notes)
                        for operation in operations
                    )
                    continue
                if model == "parakeet_tdt_v3":
                    for operation in operations:
                        if runtime == "executorch":
                            rows.append(
                                run_executorch_parakeet(
                                    config,
                                    device,
                                    runtime,
                                    model,
                                    operation,
                                    sizes,
                                )
                            )
                        elif runtime == "litert_lm":
                            rows.append(
                                run_litert_parakeet(
                                    config,
                                    device,
                                    runtime,
                                    model,
                                    operation,
                                    sizes,
                                )
                            )
                        else:
                            rows.append(
                                run_cactus_parakeet(
                                    config,
                                    device,
                                    runtime,
                                    model,
                                    operation,
                                    sizes,
                                )
                            )
                else:
                    rows.extend(
                        run_llm_operations(
                            config,
                            device,
                            runtime,
                            model,
                            operations,
                            output_path,
                            sizes,
                        )
                    )

    validate_rows(rows)
    write_csv(output_path, rows)
    if should_update_unsupported(args):
        append_unsupported(config_path.parent / "unsupported.md", rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
