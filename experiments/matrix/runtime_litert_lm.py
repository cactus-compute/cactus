#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
EXPECTED_QUANTIZATION = "int4_per_output_channel"
MANIFEST_NAMES = (
    "litert_lm_manifest.json",
    "litertlm_manifest.json",
    "manifest.json",
    "metadata.json",
)
KERNEL_GRAPH_SUFFIXES = (".tflite", ".bin", ".json")
LITERTLM_MAGIC = b"LITERTLM"
LITERTLM_HEADER_BEGIN_BYTE_OFFSET = 32
LITERTLM_HEADER_END_LOCATION_BYTE_OFFSET = 24
THIRD_PARTY_LITERT_LM = REPO_ROOT.parent / "third_party" / "litert-lm"
DEFAULT_NATIVE_RUNNER = THIRD_PARTY_LITERT_LM / "bazel-bin" / "runtime" / "engine" / "litert_lm_advanced_main"
DEFAULT_ANDROID_RUNNER = THIRD_PARTY_LITERT_LM / "bazel-bin" / "runtime" / "engine" / "litert_lm_advanced_main"
DEFAULT_ANDROID_ASR_RUNNER = THIRD_PARTY_LITERT_LM / "bazel-bin" / "runtime" / "asr" / "litert_parakeet_asr"
DEFAULT_ANDROID_GPU_LIB_DIR = THIRD_PARTY_LITERT_LM / "prebuilt" / "android_arm64"
ANDROID_GPU_LIBS = (
    "libGemmaModelConstraintProvider.so",
    "libLiteRtGpuAccelerator.so",
    "libLiteRtOpenClAccelerator.so",
    "libLiteRtTopKOpenClSampler.so",
    "libLiteRtTopKWebGpuSampler.so",
    "libLiteRtWebGpuAccelerator.so",
)


class LiteRTLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiteRTLMOperation:
    operation: str
    seqlen: int
    decode_tokens: int
    input_path: str

    @classmethod
    def from_matrix_operation(cls, operation: dict[str, Any]) -> "LiteRTLMOperation":
        return cls(
            operation=str(operation["operation"]),
            seqlen=int(operation["seqlen"]),
            decode_tokens=int(operation["decode_tokens"]),
            input_path=str(operation.get("input_path") or ""),
        )


@dataclass(frozen=True)
class LiteRTLMPairedRequest:
    prefill: LiteRTLMOperation | None
    decode: LiteRTLMOperation | None

    @property
    def context_tokens(self) -> int:
        return self.matrix_operation.seqlen

    @property
    def generated_tokens(self) -> int:
        if self.decode is None:
            return self.matrix_operation.decode_tokens
        return self.decode.decode_tokens

    @property
    def matrix_operation(self) -> LiteRTLMOperation:
        operation = self.decode or self.prefill
        if operation is None:
            raise ValueError("LiteRTLMPairedRequest has no operation")
        return operation

    @property
    def operations(self) -> list[LiteRTLMOperation]:
        return [operation for operation in (self.prefill, self.decode) if operation is not None]


@dataclass(frozen=True)
class LiteRTLMArtifactStatus:
    supported: bool
    reason: str
    manifest_path: Path | None = None
    artifact_kind: str = ""
    quantization: str = ""


@dataclass(frozen=True)
class LiteRTLMSection:
    data_type: str
    model_type: str = ""


def paired_litert_lm_requests(operations: list[dict[str, Any]]) -> list[LiteRTLMPairedRequest]:
    prefill = [
        LiteRTLMOperation.from_matrix_operation(operation)
        for operation in operations
        if operation["operation"] == "prefill"
    ]
    decode_by_context = {
        int(operation["seqlen"]): LiteRTLMOperation.from_matrix_operation(operation)
        for operation in operations
        if operation["operation"] == "decode"
    }
    prefill_contexts = {operation.seqlen for operation in prefill}
    requests: list[LiteRTLMPairedRequest] = []
    for operation in operations:
        seqlen = int(operation["seqlen"])
        if operation["operation"] == "prefill":
            prefill_operation = LiteRTLMOperation.from_matrix_operation(operation)
            requests.append(LiteRTLMPairedRequest(prefill=prefill_operation, decode=decode_by_context.get(seqlen)))
        elif operation["operation"] == "decode" and seqlen not in prefill_contexts:
            requests.append(LiteRTLMPairedRequest(prefill=None, decode=LiteRTLMOperation.from_matrix_operation(operation)))
    return requests


def resolve_artifact_path(path: str, repo_root: Path = REPO_ROOT) -> Path:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    return artifact_path


def litert_lm_artifact_status(
    artifact_path: Path,
    artifact_config: dict[str, Any] | None = None,
) -> LiteRTLMArtifactStatus:
    if not artifact_path.exists():
        return LiteRTLMArtifactStatus(False, f"LiteRT-LM artifact path does not exist: {artifact_path}")

    manifest_path = _find_manifest(artifact_path)
    if manifest_path is None:
        if artifact_path.is_file() and artifact_path.suffix.lower() == ".litertlm":
            return _litertlm_container_status(artifact_path, artifact_config)
        return _unsupported_without_manifest(artifact_path)

    manifest = _load_json(manifest_path)
    artifact_kind = str(
        manifest.get("artifact_type")
        or manifest.get("type")
        or (artifact_config or {}).get("artifact_format")
        or ""
    )
    normalized_kind = _normalize_token(artifact_kind)
    if normalized_kind not in {"litert_lm_e2e", "litertlm_e2e", "litert_lm", "litertlm"}:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM artifact manifest is not marked as an end-to-end LiteRT-LM artifact "
            f"(artifact_type={artifact_kind!r})",
            manifest_path=manifest_path,
            artifact_kind=artifact_kind,
        )

    quantization = _manifest_quantization(manifest, artifact_config)
    if _normalize_token(quantization) != EXPECTED_QUANTIZATION:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM matrix expects int4 per-output-channel artifacts; "
            f"manifest/config reports {quantization or 'missing quantization metadata'}",
            manifest_path=manifest_path,
            artifact_kind=artifact_kind,
            quantization=quantization,
        )

    missing = _missing_entry_points(manifest)
    if missing:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM artifact manifest must expose paired prefill+decode entry points; "
            f"missing {', '.join(missing)}",
            manifest_path=manifest_path,
            artifact_kind=artifact_kind,
            quantization=quantization,
        )

    contract = _normalize_token(str(manifest.get("runner_contract") or manifest.get("contract") or ""))
    if contract and contract not in {"paired_prefill_decode", "prefill_decode", "paired"}:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM artifact manifest has an unsupported runner contract "
            f"{manifest.get('runner_contract') or manifest.get('contract')!r}; expected paired prefill+decode",
            manifest_path=manifest_path,
            artifact_kind=artifact_kind,
            quantization=quantization,
        )

    return LiteRTLMArtifactStatus(
        True,
        "ok",
        manifest_path=manifest_path,
        artifact_kind=artifact_kind,
        quantization=quantization,
    )


def _litertlm_container_status(
    artifact_path: Path,
    artifact_config: dict[str, Any] | None,
) -> LiteRTLMArtifactStatus:
    quantization = _manifest_quantization({}, artifact_config)
    if _normalize_token(quantization) != EXPECTED_QUANTIZATION:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM .litertlm artifact requires matrix quantization metadata "
            f"{EXPECTED_QUANTIZATION}; config reports {quantization or 'missing quantization metadata'}",
            artifact_kind="litertlm",
            quantization=quantization,
        )

    try:
        sections = _read_litertlm_sections(artifact_path)
    except (OSError, ImportError, ValueError, struct.error) as exc:
        return LiteRTLMArtifactStatus(
            False,
            f"LiteRT-LM .litertlm artifact header could not be inspected: {exc}",
            artifact_kind="litertlm",
            quantization=quantization,
        )

    missing = _missing_litertlm_sections(sections)
    if missing:
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM .litertlm artifact is missing required end-to-end sections: "
            f"{', '.join(missing)}",
            artifact_kind="litertlm",
            quantization=quantization,
        )

    return LiteRTLMArtifactStatus(
        True,
        "ok",
        artifact_kind="litertlm",
        quantization=quantization,
    )


def local_inventory(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    candidates = []
    roots = [
        repo_root / "weights",
        repo_root / "weights" / ".cache",
        repo_root / "int4-benchmark",
        repo_root / "v2-bench",
        repo_root.parent / "int4-benchmark",
        repo_root.parent / "v2-bench",
        THIRD_PARTY_LITERT_LM / "runtime" / "testdata",
        THIRD_PARTY_LITERT_LM / "schema" / "testdata",
    ]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.endswith((".litertlm", ".litertlm.metadata", ".tflite", ".task")) or "litert" in lower:
                candidates.append(str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path))

    return {
        "packages": package_inventory(),
        "tools": tool_inventory(),
        "android": android_inventory(),
        "artifact_candidates": candidates,
        "searched_roots": [str(root) for root in roots],
    }


def package_inventory() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in (
        "ai_edge_litert",
        "ai_edge_torch",
        "litert_lm",
        "litert",
        "tensorflow",
        "tflite",
        "tflite_runtime",
    ):
        spec = importlib.util.find_spec(name)
        packages[name] = spec.origin if spec else "not installed"
    return packages


def tool_inventory() -> dict[str, str]:
    tools: dict[str, str] = {}
    for name in ("litert-lm", "litert_lm", "litert-torch", "tflite_convert", "flatc"):
        tools[name] = _which(name) or "not found"
    return tools


def android_inventory() -> dict[str, Any]:
    runner = Path(os.environ.get("ANDROID_LITERT_LM_RUNNER") or DEFAULT_ANDROID_RUNNER)
    gpu_lib_dir = Path(os.environ.get("ANDROID_LITERT_LM_GPU_LIB_DIR") or DEFAULT_ANDROID_GPU_LIB_DIR)
    return {
        "runner": str(runner),
        "runner_exists": runner.exists(),
        "runner_file_type": _file_type(runner) if runner.exists() else "",
        "gpu_lib_dir": str(gpu_lib_dir),
        "missing_gpu_libs": _missing_android_gpu_libs(gpu_lib_dir),
    }


def android_litert_parakeet_runner_path(artifact_config: dict[str, Any]) -> Path:
    configured = artifact_config.get("android_asr_runner_path") or artifact_config.get("android_runner_path")
    if configured:
        return resolve_artifact_path(str(configured))
    return DEFAULT_ANDROID_ASR_RUNNER


def litert_lm_unsupported_reason(
    config: dict[str, Any],
    device: str,
    model: str,
    runtime: str = "litert_lm",
    repo_root: Path = REPO_ROOT,
) -> str | None:
    device_config = config["devices"][device]
    model_config = config["models"][model]
    if device_config.get("kind") != "mac":
        return android_litert_lm_unsupported_reason(config, model, runtime, repo_root)

    artifact_config = model_config.get("artifacts", {}).get(runtime)
    if isinstance(artifact_config, dict) and not artifact_config.get("path") and artifact_config.get("unsupported_reason"):
        return str(artifact_config["unsupported_reason"])

    if model_config.get("type") != "llm":
        if (
            model == "parakeet_tdt_v3"
            and isinstance(artifact_config, dict)
            and _normalize_token(str(artifact_config.get("artifact_format") or "")) == "litert_tflite_asr"
        ):
            artifact_path = resolve_artifact_path(str(artifact_config.get("path") or ""), repo_root)
            if not artifact_path.exists():
                return f"normal LiteRT Parakeet artifact path does not exist: {artifact_config.get('path')}"
            for key in ("encoder_path", "decoder_joint_path", "vocab_path", "config_path"):
                value = artifact_config.get(key)
                if not value:
                    return f"normal LiteRT Parakeet artifact is missing {key}"
                if not resolve_artifact_path(str(value), repo_root).exists():
                    return f"normal LiteRT Parakeet {key} does not exist: {value}"
            if not importlib.util.find_spec("ai_edge_litert.interpreter"):
                return "normal LiteRT Parakeet requires ai_edge_litert.interpreter"
            return None
        return f"LiteRT-LM backend only supports LLM rows, got model type {model_config.get('type')!r}"

    if not isinstance(artifact_config, dict) or not artifact_config.get("path"):
        if isinstance(artifact_config, dict) and artifact_config.get("unsupported_reason"):
            return str(artifact_config["unsupported_reason"])
        return f"LiteRT-LM artifact for {model} is not configured"

    status = litert_lm_artifact_status(
        resolve_artifact_path(str(artifact_config["path"]), repo_root),
        artifact_config,
    )
    if not status.supported:
        return status.reason

    runner = litert_lm_runner_command(artifact_config)
    if runner is None:
        return (
            "LiteRT-LM runner is unavailable: no litert_lm/litert-lm command found and "
            "LITERT_LM_RUNNER is not set"
        )
    return None


def android_litert_lm_unsupported_reason(
    config: dict[str, Any],
    model: str,
    runtime: str = "litert_lm",
    repo_root: Path = REPO_ROOT,
) -> str | None:
    model_config = config["models"][model]
    artifact_config = model_config.get("artifacts", {}).get(runtime)
    if model_config.get("type") != "llm":
        if isinstance(artifact_config, dict) and _normalize_token(str(artifact_config.get("artifact_format") or "")) == "litert_tflite_asr":
            artifact_path = resolve_artifact_path(str(artifact_config.get("path") or ""), repo_root)
            if not artifact_path.exists():
                return f"Android LiteRT Parakeet artifact path does not exist: {artifact_config.get('path')}"
            for key in ("encoder_path", "decoder_joint_path", "vocab_path", "config_path"):
                value = artifact_config.get(key)
                if not value:
                    return f"Android LiteRT Parakeet artifact is missing {key}"
                if not resolve_artifact_path(str(value), repo_root).exists():
                    return f"Android LiteRT Parakeet {key} does not exist: {value}"
            runner = android_litert_parakeet_runner_path(artifact_config)
            if not runner.exists():
                return f"Android LiteRT Parakeet ASR runner does not exist: {runner}"
            return None
        if isinstance(artifact_config, dict) and artifact_config.get("unsupported_reason"):
            return str(artifact_config["unsupported_reason"])
        return "LiteRT-LM does not have a configured Android non-LLM runner for this model"

    if not isinstance(artifact_config, dict):
        return f"Android LiteRT-LM artifact is not configured for {model}; expected an end-to-end .litertlm file"
    if not artifact_config.get("path"):
        expected = artifact_config.get("expected_android_artifact") or artifact_config.get("expected_artifact")
        if expected:
            return f"Android LiteRT-LM artifact does not exist: {expected}"
        if artifact_config.get("unsupported_reason"):
            return str(artifact_config["unsupported_reason"])
        return f"Android LiteRT-LM artifact is not configured for {model}; expected an end-to-end .litertlm file"

    artifact_path = resolve_artifact_path(str(artifact_config["path"]), repo_root)
    if not artifact_path.exists():
        return f"Android LiteRT-LM artifact does not exist: {artifact_config['path']}"
    if artifact_path.is_file() and artifact_path.suffix.lower() != ".litertlm":
        return f"Android LiteRT-LM requires an end-to-end .litertlm artifact, got {artifact_config['path']}"

    status = litert_lm_artifact_status(artifact_path, artifact_config)
    if not status.supported:
        return status.reason

    backend = str(artifact_config.get("android_backend") or "cpu")
    normalized_backend = _normalize_token(backend)
    if normalized_backend in {"npu", "google_tensor_npu", "google_tensor"}:
        return "Android LiteRT-LM Google Tensor NPU path requires LiteRT AOT AI Pack app/runtime deployment, not a direct adb shell binary"
    if normalized_backend == "gpu":
        gpu_lib_dir = Path(
            str(artifact_config.get("android_gpu_lib_dir") or os.environ.get("ANDROID_LITERT_LM_GPU_LIB_DIR") or DEFAULT_ANDROID_GPU_LIB_DIR)
        ).expanduser()
        missing = _missing_android_gpu_libs(gpu_lib_dir)
        if missing:
            return (
                "Android LiteRT-LM GPU backend requires prebuilt/android_arm64 shared libraries on LD_LIBRARY_PATH; "
                f"missing {', '.join(missing)} in {gpu_lib_dir}"
            )

    runner = _android_runner_path(artifact_config)
    if runner is None:
        return (
            "Android LiteRT-LM runner binary is not configured; build //runtime/engine:litert_lm_advanced_main "
            f"with --config=android_arm64 (expected {DEFAULT_ANDROID_RUNNER})"
        )
    if not runner.exists():
        return f"Android LiteRT-LM runner binary does not exist: {runner}"
    file_type = _file_type(runner)
    if file_type and "Mach-O" in file_type:
        return (
            f"Android LiteRT-LM runner binary is not an Android executable: {runner} "
            f"({file_type}); rebuild //runtime/engine:litert_lm_advanced_main with --config=android_arm64"
        )
    if file_type and "ELF" not in file_type:
        return f"Android LiteRT-LM runner binary has unexpected file type: {runner} ({file_type})"

    return None


def _android_runner_path(artifact_config: dict[str, Any]) -> Path | None:
    value = artifact_config.get("android_runner") or os.environ.get("ANDROID_LITERT_LM_RUNNER")
    if not value:
        return DEFAULT_ANDROID_RUNNER
    return Path(str(value)).expanduser()


def _missing_android_gpu_libs(gpu_lib_dir: Path) -> list[str]:
    return [name for name in ANDROID_GPU_LIBS if not (gpu_lib_dir / name).exists()]


def _file_type(path: Path) -> str:
    completed = subprocess.run(
        ["file", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def run_litert_lm_llm_operations(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operations: list[dict[str, Any]],
    sizes: tuple[float, float] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    reason = litert_lm_unsupported_reason(config, device, model, runtime, repo_root)
    if reason is not None:
        return [
            unsupported_row(device, runtime, model, operation, reason, sizes=sizes)
            for operation in operations
        ]

    artifact_config = config["models"][model]["artifacts"][runtime]
    artifact_path = resolve_artifact_path(str(artifact_config["path"]), repo_root)
    rows: list[dict[str, Any]] = []
    for request in paired_litert_lm_requests(operations):
        rows.extend(
            _rows_for_request(
                config,
                device,
                runtime,
                model,
                artifact_path,
                artifact_config,
                request,
                sizes,
                repo_root,
            )
        )
    return rows


def unsupported_row(
    device: str,
    runtime: str,
    model: str,
    operation: dict[str, Any],
    reason: str,
    sizes: tuple[float, float] | None = None,
) -> dict[str, Any]:
    row = _base_row(device, runtime, model, operation)
    row["status"] = "unsupported"
    row["notes"] = reason
    _fill_sizes(row, sizes)
    return row


def _rows_for_request(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    artifact_path: Path,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
    sizes: tuple[float, float] | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    try:
        measurements = _measure_request(config, device, artifact_path, artifact_config, request, repo_root)
    except (OSError, RuntimeError, ValueError, LiteRTLMError) as exc:
        return [
            _error_row(device, runtime, model, operation, str(exc), sizes=sizes)
            for operation in request.operations
        ]

    rows = []
    if request.prefill is not None:
        rows.append(
            _row_from_measurements(
                config,
                device,
                runtime,
                model,
                request.prefill,
                measurements,
                sizes,
                _request_notes(config, device, artifact_config, request, for_decode=False),
            )
        )
    if request.decode is not None:
        rows.append(
            _row_from_measurements(
                config,
                device,
                runtime,
                model,
                request.decode,
                measurements,
                sizes,
                _request_notes(config, device, artifact_config, request, for_decode=True),
            )
        )
    return rows


def _request_notes(config: dict[str, Any], device: str, artifact_config: dict[str, Any], request: LiteRTLMPairedRequest, *, for_decode: bool) -> list[str]:
    notes = [
        f"benchmark_mode={config.get('_benchmark_mode') or 'strict'}",
        "runner=litert_lm",
        "provider=cpu",
        "gpu=disabled",
        litert_lm_thread_note(config, device, artifact_config),
        f"thread_count={matrix_thread_count(config)}",
        "quantization=int4_per_output_channel",
    ]
    command_max_num_tokens = _command_max_num_tokens(config, device, artifact_config, request)
    benchmark_prefill_tokens = _benchmark_prefill_tokens(command_max_num_tokens, request)
    if benchmark_prefill_tokens != request.context_tokens:
        notes.append(f"requested_context={request.context_tokens}")
        notes.append(f"litert_lm_prefill_tokens={benchmark_prefill_tokens}")
        notes.append(f"max_num_tokens={command_max_num_tokens}")
    notes.extend(android_core_affinity_notes(config, device))
    if request.prefill is not None and request.decode is not None:
        notes.insert(1, "paired_prefill_decode=true")
        notes.insert(2, f"shared_context={request.context_tokens}")
        if for_decode:
            notes.insert(3, "shares measured invocation with prefill row")
        else:
            notes.insert(3, f"shared_decode_tokens={request.generated_tokens}")
    return notes


def litert_lm_thread_note(config: dict[str, Any], device: str, artifact_config: dict[str, Any]) -> str:
    if config["devices"][device].get("kind") == "android":
        return f"threads={matrix_thread_count(config)}"
    runner = litert_lm_runner_command(artifact_config)
    if runner is not None and not _uses_litert_lm_cli(runner):
        return f"threads={matrix_thread_count(config)}"
    return "threads=unverified_litert_lm_cli_no_thread_flag"


def full_core_prefill_mode(config: dict[str, Any]) -> bool:
    return str(config.get("_benchmark_mode") or "strict") == "full_core_prefill"


def matrix_thread_count(config: dict[str, Any]) -> int:
    if full_core_prefill_mode(config):
        return max(1, int(config.get("_full_core_threads") or 8))
    return 1


def android_core_affinity(config: dict[str, Any], device: str) -> tuple[int, str]:
    device_config = config["devices"][device]
    cpu = int(device_config.get("core_affinity_cpu") or 7)
    mask = str(device_config.get("core_affinity_mask") or format(1 << cpu, "x"))
    return cpu, mask


def android_core_affinity_notes(config: dict[str, Any], device: str) -> list[str]:
    if full_core_prefill_mode(config):
        return ["affinity=not_set", "taskset_mask=none"]
    if config["devices"][device].get("kind") != "android":
        return []
    cpu, mask = android_core_affinity(config, device)
    return [f"core_affinity=cpu{cpu}", f"taskset_mask={mask}"]


def _measure_request(
    config: dict[str, Any],
    device: str,
    artifact_path: Path,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
    repo_root: Path,
) -> list[dict[str, Any]]:
    runs = config["operations"]["llm"]
    if config["devices"][device].get("kind") == "android":
        for _ in range(int(runs["warmup_runs"])):
            _run_android_litert_lm_once(config, device, artifact_path, artifact_config, request, repo_root)
        return [
            _run_android_litert_lm_once(config, device, artifact_path, artifact_config, request, repo_root)
            for _ in range(int(runs["measurement_runs"]))
        ]
    for _ in range(int(runs["warmup_runs"])):
        _run_litert_lm_once(config, artifact_path, artifact_config, request, repo_root)
    return [
        _run_litert_lm_once(config, artifact_path, artifact_config, request, repo_root)
        for _ in range(int(runs["measurement_runs"]))
    ]


def _run_litert_lm_once(
    config: dict[str, Any],
    artifact_path: Path,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
    repo_root: Path,
) -> dict[str, Any]:
    runner = litert_lm_runner_command(artifact_config)
    if runner is None:
        raise LiteRTLMError("LiteRT-LM runner command is unavailable")

    command = _litert_lm_benchmark_command(runner, artifact_path, artifact_config, request, matrix_thread_count(config))
    env = os.environ.copy()
    threads = str(matrix_thread_count(config))
    env["OMP_NUM_THREADS"] = threads
    env["VECLIB_MAXIMUM_THREADS"] = threads
    env["XNNPACK_NUM_THREADS"] = threads
    env["LITERT_NUM_THREADS"] = threads
    completed = _run_with_peak_memory(command, cwd=repo_root, env=env)
    output = _combined_output(completed)
    if completed.returncode != 0:
        message = _litert_lm_failure_message(output, completed.returncode)
        raise LiteRTLMError(message)
    payload = _parse_litert_lm_metrics(output)
    payload["_peak_process_memory_mb"] = completed.peak_process_memory_mb
    return payload


def _run_android_litert_lm_once(
    config: dict[str, Any],
    device: str,
    artifact_path: Path,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
    repo_root: Path,
) -> dict[str, Any]:
    serial = str(config["devices"][device].get("serial") or "")
    if not serial:
        raise LiteRTLMError(f"Android device {device} is missing a serial")
    runner = _android_runner_path(artifact_config)
    if runner is None or not runner.exists():
        raise LiteRTLMError(f"Android LiteRT-LM runner binary does not exist: {runner}")

    base_dir = "/data/local/tmp/cactus_matrix/litert_lm"
    bin_dir = f"{base_dir}/bin"
    model_dir = f"{base_dir}/models"
    lib_dir = f"{base_dir}/lib"
    runner_device = f"{bin_dir}/{runner.name}"
    model_device = f"{model_dir}/{artifact_path.name}"

    _adb(serial, "shell", "mkdir", "-p", bin_dir, model_dir, lib_dir)
    _adb_push_if_needed(serial, runner, runner_device)
    _adb(serial, "shell", "chmod", "755", runner_device)
    _adb_push_if_needed(serial, artifact_path, model_device)
    gpu_lib_dir = Path(
        str(artifact_config.get("android_gpu_lib_dir") or os.environ.get("ANDROID_LITERT_LM_GPU_LIB_DIR") or DEFAULT_ANDROID_GPU_LIB_DIR)
    ).expanduser()
    if gpu_lib_dir.exists():
        for shared_lib in sorted(gpu_lib_dir.glob("*.so")):
            _adb_push_if_needed(serial, shared_lib, f"{lib_dir}/{shared_lib.name}")

    backend = str(artifact_config.get("android_backend") or artifact_config.get("backend") or "cpu")
    max_num_tokens = _command_max_num_tokens(config, device, artifact_config, request)
    benchmark_prefill_tokens = _benchmark_prefill_tokens(max_num_tokens, request)
    prompt = str(artifact_config.get("android_input_prompt") or artifact_config.get("input_prompt") or "Hello")
    threads = matrix_thread_count(config)
    affinity_command = ""
    if not full_core_prefill_mode(config):
        affinity_command = f"taskset -p {shlex.quote(android_core_affinity(config, device)[1])} $$ >/dev/null && "
    command = (
        f"cd {shlex.quote(base_dir)} && "
        "echo CACTUS_MATRIX_PID=$$ && "
        f"export LD_LIBRARY_PATH={shlex.quote(lib_dir)} && "
        f"{affinity_command}"
        f"exec {shlex.quote(runner_device)} "
        f"--backend={shlex.quote(backend)} "
        f"--model_path={shlex.quote(model_device)} "
        f"--input_prompt={shlex.quote(prompt)} "
        "--benchmark=true "
        f"--benchmark_prefill_tokens={benchmark_prefill_tokens} "
        f"--benchmark_decode_tokens={request.generated_tokens} "
        f"--max_num_tokens={max_num_tokens} "
        f"--prefill_batch_sizes={benchmark_prefill_tokens} "
        "--async=false "
        f"--num_cpu_threads={threads} "
        "--report_peak_memory_footprint=true"
    )
    completed = _adb(serial, "shell", command, cwd=repo_root)
    text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise LiteRTLMError(_litert_lm_failure_message(text, completed.returncode))
    return _parse_litert_lm_metrics(text)


def _adb(serial: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = ["adb", "-s", serial, *args]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 and (len(args) < 1 or args[0] != "shell"):
        raise LiteRTLMError(completed.stderr.strip() or completed.stdout.strip() or f"adb exited {completed.returncode}")
    return completed


def _adb_push_if_needed(serial: str, source: Path, destination: str) -> None:
    size = source.stat().st_size
    stat = _adb(serial, "shell", f"stat -c %s {shlex.quote(destination)} 2>/dev/null || true")
    if stat.stdout.strip() == str(size):
        return
    pushed = _adb(serial, "push", str(source), destination)
    if pushed.returncode != 0:
        raise LiteRTLMError(pushed.stderr.strip() or pushed.stdout.strip() or "adb push failed")


def _litert_lm_benchmark_command(
    runner: list[str],
    artifact_path: Path,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
    threads: int = 1,
) -> list[str]:
    backend = str(
        artifact_config.get("backend")
        or artifact_config.get("litert_lm_backend")
        or "cpu"
    )
    max_num_tokens = _native_max_num_tokens(artifact_config, request)
    benchmark_prefill_tokens = _benchmark_prefill_tokens(max_num_tokens, request)
    if _uses_litert_lm_cli(runner):
        command = [
            *runner,
            "benchmark",
            str(artifact_path),
            "--prefill-tokens",
            str(benchmark_prefill_tokens),
            "--decode-tokens",
            str(request.generated_tokens),
            "--backend",
            backend,
            "--enable-speculative-decoding",
            str(artifact_config.get("enable_speculative_decoding") or "auto"),
        ]
        if artifact_config.get("max_num_tokens") is not None:
            command.extend(
                [
                    "--max-num-tokens",
                    str(max_num_tokens),
                ]
            )
        if artifact_config.get("verbose"):
            command.append("--verbose")
        return command

    command = [
        *runner,
        f"--model_path={artifact_path}",
        f"--backend={backend}",
        "--benchmark=true",
        f"--benchmark_prefill_tokens={benchmark_prefill_tokens}",
        f"--benchmark_decode_tokens={request.generated_tokens}",
        f"--max_num_tokens={max_num_tokens}",
        f"--prefill_batch_sizes={benchmark_prefill_tokens}",
        "--async=false",
        "--num_iterations=1",
    ]
    if request.generated_tokens > 0:
        command.append(f"--max_output_tokens={request.generated_tokens}")
    if _normalize_token(backend) == "cpu":
        command.append(f"--num_cpu_threads={threads}")
    extra_args = artifact_config.get("runner_args")
    if isinstance(extra_args, str):
        command.extend(shlex.split(extra_args))
    elif isinstance(extra_args, list):
        command.extend(str(value) for value in extra_args)
    return command


def _native_max_num_tokens(artifact_config: dict[str, Any], request: LiteRTLMPairedRequest) -> int:
    configured = artifact_config.get("max_num_tokens")
    minimum = _minimum_max_num_tokens(request)
    if configured is None:
        return minimum
    return max(int(configured), minimum)


def _minimum_max_num_tokens(request: LiteRTLMPairedRequest) -> int:
    return request.context_tokens + request.generated_tokens


def _command_max_num_tokens(
    config: dict[str, Any],
    device: str,
    artifact_config: dict[str, Any],
    request: LiteRTLMPairedRequest,
) -> int:
    if config["devices"][device].get("kind") == "android":
        configured = int(artifact_config.get("android_max_num_tokens") or artifact_config.get("max_num_tokens") or 4096)
        return max(configured, _minimum_max_num_tokens(request))
    return _native_max_num_tokens(artifact_config, request)


def _benchmark_prefill_tokens(max_num_tokens: int, request: LiteRTLMPairedRequest) -> int:
    if request.generated_tokens == 0 and request.context_tokens >= max_num_tokens:
        return max(1, max_num_tokens - 1)
    return request.context_tokens


def _combined_output(completed: Any) -> str:
    return "\n".join(part for part in (completed.stdout, completed.stderr) if part)


def _litert_lm_failure_message(text: str, returncode: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    error_lines = [
        line
        for line in lines
        if "ERROR:" in line or "Check failed:" in line or "failed to prepare" in line
    ]
    if error_lines:
        return "; ".join(error_lines[-3:])
    return lines[-1] if lines else f"LiteRT-LM runner exited {returncode}"


def _uses_litert_lm_cli(runner: list[str]) -> bool:
    if not runner:
        return False
    executable = Path(runner[0]).name
    return executable == "litert-lm"


def litert_lm_runner_command(artifact_config: dict[str, Any] | None = None) -> list[str] | None:
    configured = None if artifact_config is None else artifact_config.get("runner")
    if configured:
        if isinstance(configured, str):
            return shlex.split(configured)
        if isinstance(configured, list):
            return [str(value) for value in configured]
    env_runner = os.environ.get("LITERT_LM_RUNNER")
    if env_runner:
        return shlex.split(env_runner)
    if is_host_executable(DEFAULT_NATIVE_RUNNER):
        return [str(DEFAULT_NATIVE_RUNNER)]
    for name in ("litert-lm", "litert_lm"):
        path = _which(name)
        if path:
            return [path]
    return None


def is_host_executable(path: Path) -> bool:
    if not path.exists():
        return False
    file_type = _file_type(path)
    if sys.platform == "darwin":
        return "Mach-O" in file_type
    if sys.platform.startswith("linux"):
        return "ELF" in file_type and "Android" not in file_type and "aarch64" not in file_type
    return os.access(path, os.X_OK)


def _unsupported_without_manifest(artifact_path: Path) -> LiteRTLMArtifactStatus:
    if artifact_path.is_file() and artifact_path.suffix.lower() == ".metadata" and artifact_path.name.endswith(".litertlm.metadata"):
        expected = artifact_path.with_suffix("")
        return LiteRTLMArtifactStatus(
            False,
            "LiteRT-LM download metadata is present but the model artifact is missing: "
            f"found {artifact_path}, expected {expected}",
        )

    if artifact_path.is_dir():
        litertlm_files = sorted(path.name for path in artifact_path.glob("*.litertlm"))
        if litertlm_files:
            preview = ", ".join(litertlm_files[:4])
            more = "" if len(litertlm_files) <= 4 else f", +{len(litertlm_files) - 4} more"
            return LiteRTLMArtifactStatus(
                False,
                "LiteRT-LM artifact path must point to the end-to-end .litertlm file, "
                f"not its parent directory; found {preview}{more} under {artifact_path}",
            )

    if artifact_path.is_file() and artifact_path.suffix.lower() == ".tflite":
        detail = _tflite_detail(artifact_path)
        return LiteRTLMArtifactStatus(
            False,
            "kernel-only LiteRT graph is not a LiteRT-LM end-to-end artifact: "
            "missing LiteRT-LM tokenizer/session metadata and paired prefill+decode entry points"
            f"{detail}",
        )

    files = _artifact_files(artifact_path)
    kernel_files = [
        path.name
        for path in files
        if path.suffix.lower() in KERNEL_GRAPH_SUFFIXES
        and ("prefill" in path.name.lower() or "decode" in path.name.lower() or path.suffix.lower() == ".tflite")
    ]
    if kernel_files:
        preview = ", ".join(kernel_files[:8])
        more = "" if len(kernel_files) <= 8 else f", +{len(kernel_files) - 8} more"
        return LiteRTLMArtifactStatus(
            False,
            "kernel-only LiteRT graph inventory is unsupported for LiteRT-LM matrix rows: "
            "found graph files but no end-to-end LiteRT-LM manifest with tokenizer/session and paired "
            f"prefill+decode contract ({preview}{more})",
        )

    return LiteRTLMArtifactStatus(
        False,
        "LiteRT-LM artifact must be an end-to-end LiteRT-LM bundle with manifest metadata; "
        f"no {', '.join(MANIFEST_NAMES)} found under {artifact_path}",
    )


def _find_manifest(artifact_path: Path) -> Path | None:
    if artifact_path.is_file():
        if artifact_path.suffix.lower() == ".json" and artifact_path.name in MANIFEST_NAMES:
            return artifact_path
        sibling = artifact_path.with_suffix(artifact_path.suffix + ".json")
        if sibling.exists():
            return sibling
        return None

    for name in MANIFEST_NAMES:
        path = artifact_path / name
        if path.exists():
            return path
    return None


def _read_litertlm_sections(path: Path) -> list[LiteRTLMSection]:
    from ai_edge_litert.internal import litertlm_header_schema_py_generated as schema

    with path.open("rb") as handle:
        if handle.read(len(LITERTLM_MAGIC)) != LITERTLM_MAGIC:
            raise ValueError("invalid LiteRT-LM magic number")
        handle.seek(LITERTLM_HEADER_END_LOCATION_BYTE_OFFSET)
        header_end_offset = struct.unpack("<Q", handle.read(8))[0]
        if header_end_offset <= LITERTLM_HEADER_BEGIN_BYTE_OFFSET:
            raise ValueError(f"invalid LiteRT-LM header end offset {header_end_offset}")
        handle.seek(LITERTLM_HEADER_BEGIN_BYTE_OFFSET)
        header = handle.read(header_end_offset - LITERTLM_HEADER_BEGIN_BYTE_OFFSET)

    metadata = schema.LiteRTLMMetaData.GetRootAs(header, 0)
    section_metadata = metadata.SectionMetadata()
    if section_metadata is None:
        return []

    type_names = {
        value: key
        for key, value in schema.AnySectionDataType.__dict__.items()
        if isinstance(value, int)
    }
    sections = []
    for index in range(section_metadata.ObjectsLength()):
        section = section_metadata.Objects(index)
        if section is None:
            continue
        sections.append(
            LiteRTLMSection(
                data_type=type_names.get(section.DataType(), str(section.DataType())),
                model_type=_section_model_type(section, schema),
            )
        )
    return sections


def _section_model_type(section: Any, schema: Any) -> str:
    for index in range(section.ItemsLength()):
        item = section.Items(index)
        if item is None:
            continue
        key = item.Key()
        if key is None or key.decode("utf-8", errors="replace") != "model_type":
            continue
        if item.ValueType() != schema.VData.StringValue:
            continue
        value = item.Value()
        if value is None:
            continue
        string_value = schema.StringValue()
        string_value.Init(value.Bytes, value.Pos)
        raw = string_value.Value()
        return raw.decode("utf-8", errors="replace") if raw else ""
    return ""


def _missing_litertlm_sections(sections: list[LiteRTLMSection]) -> list[str]:
    data_types = {_normalize_token(section.data_type) for section in sections}
    model_types = {_normalize_token(section.model_type) for section in sections if section.model_type}
    missing = []
    if "llmmetadataproto" not in data_types and "llm_metadata_proto" not in data_types:
        missing.append("LlmMetadataProto")
    if "sp_tokenizer" not in data_types and "hf_tokenizer_zlib" not in data_types:
        missing.append("tokenizer")
    has_prefill_decode = bool(
        model_types
        & {
            "tf_lite_prefill_decode",
            "tflite_prefill_decode",
            "prefill_decode",
            "decoder_prefill_decode",
        }
    )
    has_split_prefill_decode = (
        bool(model_types & {"prefill", "decoder_prefill", "tf_lite_prefill", "tflite_prefill"})
        and bool(model_types & {"decode", "decoder_step", "tf_lite_decode", "tflite_decode"})
    )
    if "tflitemodel" not in data_types and "tflite_model" not in data_types:
        missing.append("TFLiteModel")
    elif not has_prefill_decode and not has_split_prefill_decode:
        missing.append("prefill+decode TFLite model_type")
    return missing


def _missing_entry_points(manifest: dict[str, Any]) -> list[str]:
    entry_points = manifest.get("entry_points") or manifest.get("components") or manifest.get("graphs")
    names: set[str] = set()
    if isinstance(entry_points, dict):
        names = {_normalize_token(str(key)) for key in entry_points}
    elif isinstance(entry_points, list):
        for item in entry_points:
            if isinstance(item, str):
                names.add(_normalize_token(item))
            elif isinstance(item, dict):
                names.add(_normalize_token(str(item.get("name") or item.get("type") or item.get("operation") or "")))

    missing = []
    if "prefill" not in names and "decoder_prefill" not in names:
        missing.append("prefill")
    if "decode" not in names and "decoder_step" not in names:
        missing.append("decode")
    return missing


def _manifest_quantization(manifest: dict[str, Any], artifact_config: dict[str, Any] | None) -> str:
    for source in (manifest, manifest.get("model") if isinstance(manifest.get("model"), dict) else None, artifact_config):
        if not isinstance(source, dict):
            continue
        value = source.get("quantization") or source.get("format")
        if value:
            return str(value)
    return ""


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not load as a JSON object")
    return data


def _artifact_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]


def _tflite_detail(path: Path) -> str:
    spec = importlib.util.find_spec("tflite")
    if spec is None:
        return "; tflite schema package is unavailable for FlatBuffer inspection"
    try:
        import tflite

        data = path.read_bytes()
        model = tflite.Model.GetRootAsModel(data, 0)
        subgraphs = []
        for index in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(index)
            name = subgraph.Name()
            subgraphs.append(name.decode("utf-8", errors="replace") if name else f"subgraph_{index}")
        metadata = []
        for index in range(model.MetadataLength()):
            item = model.Metadata(index)
            name = item.Name()
            metadata.append(name.decode("utf-8", errors="replace") if name else f"metadata_{index}")
        return f"; subgraphs={subgraphs or 'none'}, metadata={metadata or 'none'}"
    except Exception as exc:
        return f"; FlatBuffer inspection failed: {exc}"


def _row_from_measurements(
    config: dict[str, Any],
    device: str,
    runtime: str,
    model: str,
    operation: LiteRTLMOperation,
    measurements: list[dict[str, Any]],
    sizes: tuple[float, float] | None,
    extra_notes: list[str],
) -> dict[str, Any]:
    runs = config["operations"]["llm"]
    row = _base_row(
        device,
        runtime,
        model,
        {
            "operation": operation.operation,
            "seqlen": operation.seqlen,
            "decode_tokens": operation.decode_tokens,
        },
    )
    throughputs = [_measured_throughput(measurement, operation) for measurement in measurements]
    ram_values = [
        float(measurement.get("peak_ram_usage_mb") or measurement.get("ram_usage_mb") or measurement.get("_peak_process_memory_mb") or 0.0)
        for measurement in measurements
    ]
    ram_values = [value for value in ram_values if value > 0.0]
    if ram_values:
        row["peak_ram_mb"] = f"{max(ram_values):.3f}"
    _fill_sizes(row, sizes)
    notes = [f"warmup_runs={runs['warmup_runs']},measurement_runs={runs['measurement_runs']}", *extra_notes]
    if not throughputs or any(value <= 0.0 for value in throughputs):
        row["status"] = "error"
        notes.append(f"LiteRT-LM runner did not report positive {operation.operation} throughput")
        notes.append(f"throughputs={','.join(f'{value:.6f}' for value in throughputs) or 'none'}")
        row["notes"] = "; ".join(notes)
        return row
    row["throughput_tok_per_s"] = f"{statistics.median(throughputs):.6f}"
    row["status"] = "ok"
    if not ram_values:
        notes.append("peak RAM unavailable until memory collector is wired")
    row["notes"] = "; ".join(notes)
    return row


def _measured_throughput(result: dict[str, Any], operation: LiteRTLMOperation) -> float:
    if operation.operation == "decode":
        return float(result.get("decode_tps") or 0.0)
    reported = result.get("prefill_tps")
    if reported:
        return float(reported)
    prefill_ms = result.get("prefill_ms") or result.get("time_to_first_token_ms")
    if prefill_ms:
        return (float(operation.seqlen) * 1000.0) / float(prefill_ms)
    return 0.0


def _base_row(device: str, runtime: str, model: str, operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": device,
        "runtime": runtime,
        "model": model,
        "operation": operation["operation"],
        "seqlen": operation["seqlen"],
        "decode_tokens": operation["decode_tokens"],
        "throughput_tok_per_s": "",
        "peak_ram_mb": "",
        "disk_size_mb": "",
        "zipped_size_mb": "",
        "status": "",
        "notes": "",
    }


def _error_row(
    device: str,
    runtime: str,
    model: str,
    operation: LiteRTLMOperation,
    note: str,
    sizes: tuple[float, float] | None = None,
) -> dict[str, Any]:
    row = _base_row(
        device,
        runtime,
        model,
        {
            "operation": operation.operation,
            "seqlen": operation.seqlen,
            "decode_tokens": operation.decode_tokens,
        },
    )
    row["status"] = "error"
    row["notes"] = note
    _fill_sizes(row, sizes)
    return row


def _fill_sizes(row: dict[str, Any], sizes: tuple[float, float] | None) -> None:
    if sizes is not None:
        row["disk_size_mb"] = f"{sizes[0]:.3f}"
        row["zipped_size_mb"] = f"{sizes[1]:.3f}"


def _parse_litert_lm_metrics(text: str) -> dict[str, Any]:
    try:
        return _parse_last_json_object(text)
    except ValueError:
        return _parse_benchmark_info(text)


def _parse_last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
    raise ValueError("LiteRT-LM runner did not emit a JSON object")


def _parse_benchmark_info(text: str) -> dict[str, Any]:
    prefill_speeds = [
        float(match.group(1))
        for match in re.finditer(r"Prefill [Ss]peed:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s(?:ec)?", text)
    ]
    decode_speeds = [
        float(match.group(1))
        for match in re.finditer(r"Decode [Ss]peed:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s(?:ec)?", text)
    ]
    time_to_first_token = re.search(r"Time to first token:\s*([0-9]+(?:\.[0-9]+)?)\s*s", text)
    init_time = re.search(r"Init time:\s*([0-9]+(?:\.[0-9]+)?)\s*s", text)
    peak_private = re.search(r"Peak private footprint:\s*([0-9]+(?:\.[0-9]+)?)MB", text)
    peak_system = re.search(r"Peak system ram usage:\s*([0-9]+(?:\.[0-9]+)?)MB", text)
    if not prefill_speeds and not decode_speeds and time_to_first_token is None:
        raise ValueError("LiteRT-LM runner did not emit JSON or BenchmarkInfo metrics")
    payload: dict[str, Any] = {}
    if prefill_speeds:
        payload["prefill_tps"] = statistics.median(prefill_speeds)
    if decode_speeds:
        payload["decode_tps"] = statistics.median(decode_speeds)
    if time_to_first_token is not None:
        payload["time_to_first_token_ms"] = float(time_to_first_token.group(1)) * 1000.0
    if init_time is not None:
        payload["init_time_ms"] = float(init_time.group(1)) * 1000.0
    if peak_private is not None:
        payload["peak_ram_usage_mb"] = float(peak_private.group(1))
    elif peak_system is not None:
        payload["peak_ram_usage_mb"] = float(peak_system.group(1))
    return payload


def _run_with_peak_memory(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> Any:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from collect_memory import run_with_peak_memory

        return run_with_peak_memory(command, cwd=cwd, env=env)
    finally:
        if sys.path and sys.path[0] == str(Path(__file__).resolve().parent):
            sys.path.pop(0)


def _which(name: str) -> str | None:
    completed = subprocess.run(
        ["which", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return None


def _write_inventory(path: Path, repo_root: Path) -> None:
    payload = local_inventory(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LiteRT-LM matrix runtime availability.")
    parser.add_argument("--inventory", action="store_true", help="Print package/tool/artifact inventory as JSON")
    parser.add_argument("--out", help="Optional JSON output path for --inventory")
    args = parser.parse_args()

    if args.inventory:
        payload = local_inventory(REPO_ROOT)
        if args.out:
            _write_inventory(Path(args.out), REPO_ROOT)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
