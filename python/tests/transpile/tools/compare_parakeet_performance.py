from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile

TOOLS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TOOLS_DIR.parents[2]
PROJECT_ROOT = PYTHON_ROOT.parent

sys.path.insert(0, str(PYTHON_ROOT))

from cactus.bindings.cactus import cactus_destroy
from cactus.bindings.cactus import cactus_init
from cactus.bindings.cactus import cactus_transcribe
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.model_adapters import canonicalize_model_interface
from cactus.transpile.optimize_graph import FusionConfig
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.hf_model import TranspileWrapper
from cactus.transpile.hf_model import _ctc_greedy_decode_token_ids
from cactus.transpile.hf_model import _decode_token_ids
from cactus.transpile.hf_model import _infer_task_from_config
from cactus.transpile.hf_model import _load_optional_json
from cactus.transpile.hf_model import _load_optional_tokenizer
from cactus.transpile.hf_model import _load_transformers_bundle
from cactus.transpile.hf_model import _lower_preoptimized_ir
from cactus.transpile.hf_model import _parse_dtype
from cactus.transpile.hf_model import _prepare_audio_inputs
from cactus.transpile.hf_model import _validate_weights_dir


@dataclass
class NumericSummary:
    mean: float
    min: float
    max: float
    median: float


@dataclass
class ProfileEntry:
    op: str
    time_ms: float
    shape: str
    raw_line: str


@dataclass
class ProfileSection:
    entries: list[ProfileEntry]
    total_ms: float


@dataclass
class HandwrittenRun:
    wall_ms: float
    total_time_ms: float
    time_to_first_token_ms: float
    prefill_tps: float
    decode_tps: float
    decode_tokens: int
    prefill_tokens: int
    total_tokens: int
    confidence: float
    transcript: str
    response_json: dict[str, Any]


@dataclass
class TranspiledRun:
    preprocess_ms: float
    execute_ms: float
    decode_ms: float
    total_ms: float
    graph_only_ms: float
    output_frames: int
    output_vocab: int
    decode_tokens: int
    transcript: str | None
    input_shapes: dict[str, list[int]]


def _numeric_summary(values: list[float]) -> NumericSummary:
    arr = np.asarray(values, dtype=np.float64)
    return NumericSummary(
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        median=float(np.median(arr)),
    )


def _resolve_default_weights_dir(model_id: str) -> Path | None:
    candidate = PROJECT_ROOT / "weights" / model_id.split("/")[-1].lower()
    if (candidate / "config.txt").exists():
        return candidate
    return None


def _audio_duration_seconds(audio_file: str) -> float:
    sample_rate, samples = wavfile.read(audio_file)
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate in {audio_file}: {sample_rate}")
    return float(len(samples)) / float(sample_rate)


def _profile_env(profile_file: Path | None):
    class _ProfileContext:
        def __enter__(self_inner):
            self_inner.old_profile = os.environ.get("CACTUS_PROFILE")
            self_inner.old_profile_file = os.environ.get("CACTUS_PROFILE_FILE")
            if profile_file is None:
                os.environ.pop("CACTUS_PROFILE", None)
                os.environ.pop("CACTUS_PROFILE_FILE", None)
                return self_inner
            profile_file.parent.mkdir(parents=True, exist_ok=True)
            if profile_file.exists():
                profile_file.unlink()
            os.environ.pop("CACTUS_PROFILE", None)
            os.environ["CACTUS_PROFILE_FILE"] = str(profile_file)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            if self_inner.old_profile is None:
                os.environ.pop("CACTUS_PROFILE", None)
            else:
                os.environ["CACTUS_PROFILE"] = self_inner.old_profile

            if self_inner.old_profile_file is None:
                os.environ.pop("CACTUS_PROFILE_FILE", None)
            else:
                os.environ["CACTUS_PROFILE_FILE"] = self_inner.old_profile_file
            return False

    return _ProfileContext()


def _parse_profile_file(path: Path) -> list[ProfileSection]:
    if not path.exists():
        return []

    lines = path.read_text().splitlines()
    sections: list[ProfileSection] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "=== Graph Execution Profile ===":
            idx += 1
            continue

        entries: list[ProfileEntry] = []
        total_ms = 0.0
        idx += 1
        while idx < len(lines):
            line = lines[idx].rstrip()
            stripped = line.strip()
            if stripped.startswith("Total execution time:"):
                try:
                    total_ms = float(stripped.split(":", 1)[1].strip().split()[0])
                except Exception:
                    total_ms = 0.0
            elif stripped == "================================":
                break
            elif (
                stripped
                and not stripped.startswith("Operation")
                and not set(stripped).issubset({"-"})
                and not stripped.startswith("Total execution time:")
            ):
                parts = stripped.split()
                if len(parts) >= 3:
                    op = parts[0]
                    try:
                        time_ms = float(parts[1])
                    except ValueError:
                        time_ms = 0.0
                    shape = parts[2] if parts[2].startswith("[") else ""
                    entries.append(ProfileEntry(op=op, time_ms=time_ms, shape=shape, raw_line=stripped))
            idx += 1

        sections.append(ProfileSection(entries=entries, total_ms=total_ms))
        idx += 1

    return sections


def _summarize_profile_sections(sections: list[ProfileSection]) -> dict[str, Any]:
    op_total_ms: dict[str, float] = defaultdict(float)
    op_count: dict[str, int] = defaultdict(int)
    all_entries: list[ProfileEntry] = []
    total_ms_values: list[float] = []

    for section in sections:
        total_ms_values.append(section.total_ms)
        all_entries.extend(section.entries)
        for entry in section.entries:
            op_total_ms[entry.op] += entry.time_ms
            op_count[entry.op] += 1

    by_op = []
    for op, total_ms in sorted(op_total_ms.items(), key=lambda item: item[1], reverse=True):
        count = op_count[op]
        by_op.append(
            {
                "op": op,
                "count": count,
                "total_ms": round(total_ms, 6),
                "mean_ms": round(total_ms / max(count, 1), 6),
            }
        )

    slowest_nodes = [
        {
            "op": entry.op,
            "time_ms": round(entry.time_ms, 6),
            "shape": entry.shape,
            "raw_line": entry.raw_line,
        }
        for entry in sorted(all_entries, key=lambda item: item.time_ms, reverse=True)[:50]
    ]

    return {
        "runs_profiled": len(sections),
        "graph_total_ms": asdict(_numeric_summary(total_ms_values)) if total_ms_values else None,
        "entries_per_run": [len(section.entries) for section in sections],
        "by_op": by_op,
        "slowest_nodes": slowest_nodes,
    }


def _run_handwritten_once(
    *,
    model_handle: Any,
    audio_file: str,
) -> HandwrittenRun:
    options_json = json.dumps(
        {
            "use_vad": False,
            "max_tokens": 4096,
        }
    )
    start = time.perf_counter()
    response = cactus_transcribe(
        model_handle,
        audio_file,
        "",
        options_json,
        None,
        None,
    )
    end = time.perf_counter()

    payload = json.loads(response)
    return HandwrittenRun(
        wall_ms=(end - start) * 1000.0,
        total_time_ms=float(payload.get("total_time_ms", 0.0)),
        time_to_first_token_ms=float(payload.get("time_to_first_token_ms", 0.0)),
        prefill_tps=float(payload.get("prefill_tps", 0.0)),
        decode_tps=float(payload.get("decode_tps", 0.0)),
        decode_tokens=int(payload.get("decode_tokens", 0)),
        prefill_tokens=int(payload.get("prefill_tokens", 0)),
        total_tokens=int(payload.get("total_tokens", 0)),
        confidence=float(payload.get("confidence", 0.0)),
        transcript=str(payload.get("response", "")),
        response_json=payload,
    )


def _run_transpiled_once(
    *,
    tg: Any,
    processor: object | None,
    input_names: tuple[str, ...],
    model_config: dict[str, object],
    preprocessor_config: dict[str, object],
    model: torch.nn.Module,
    torch_dtype: torch.dtype,
    audio_file: str,
    tokenizer: object | None,
    blank_token_id: int | None,
) -> TranspiledRun:
    start_pre = time.perf_counter()
    prepared = _prepare_audio_inputs(
        processor,
        input_names=input_names,
        config=model_config,
        preprocessor_config=preprocessor_config,
        model=model,
        task="ctc_logits",
        audio_file=audio_file,
        torch_dtype=torch_dtype,
    )
    end_pre = time.perf_counter()

    input_arrays = [tensor.detach().cpu().numpy() for tensor in prepared.tensors]

    start_exec = time.perf_counter()
    tg.set_inputs(input_arrays)
    outputs = tg.execute()
    end_exec = time.perf_counter()

    start_decode = time.perf_counter()
    logits = outputs[0].numpy().astype(np.float32)
    token_ids = _ctc_greedy_decode_token_ids(logits, blank_token_id=blank_token_id)
    transcript = _decode_token_ids(tokenizer, token_ids) if tokenizer is not None else None
    end_decode = time.perf_counter()

    return TranspiledRun(
        preprocess_ms=(end_pre - start_pre) * 1000.0,
        execute_ms=(end_exec - start_exec) * 1000.0,
        decode_ms=(end_decode - start_decode) * 1000.0,
        total_ms=(end_decode - start_pre) * 1000.0,
        graph_only_ms=(end_exec - start_exec) * 1000.0,
        output_frames=int(logits.shape[1]),
        output_vocab=int(logits.shape[2]),
        decode_tokens=len(token_ids),
        transcript=transcript,
        input_shapes={
            name: list(tensor.shape)
            for name, tensor in zip(prepared.names, prepared.tensors)
        },
    )


def _run_transpiled_graph_only_once(
    *,
    tg: Any,
    cached_inputs: list[np.ndarray],
) -> float:
    start = time.perf_counter()
    tg.set_inputs(cached_inputs)
    tg.execute()
    end = time.perf_counter()
    return (end - start) * 1000.0


def _warmup(fn, count: int) -> None:
    for _ in range(max(count, 0)):
        fn()


def _roundtrip_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _roundtrip_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_roundtrip_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_roundtrip_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, NumericSummary):
        return asdict(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def _build_transpiled_graph(
    *,
    model_id: str,
    audio_file: str,
    weights_dir: str | None,
    torch_dtype: torch.dtype,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
    fusion_config: FusionConfig,
):
    task = _infer_task_from_config(model_id)
    if task != "ctc_logits":
        raise RuntimeError(f"expected a CTC model, got task={task}")

    model_source, processor, model, model_config = _load_transformers_bundle(
        model_id=model_id,
        task=task,
        torch_dtype=torch_dtype,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    preprocessor_config = _load_optional_json(model_source, "preprocessor_config.json")
    if not preprocessor_config:
        preprocessor_config = _load_optional_json(model_id, "preprocessor_config.json")

    tokenizer = _load_optional_tokenizer(
        model_id=model_id,
        model_source=model_source,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    canonical = canonicalize_model_interface(model, task=task)
    prepared = _prepare_audio_inputs(
        processor,
        input_names=canonical.input_names,
        config=model_config,
        preprocessor_config=preprocessor_config,
        model=model,
        task=task,
        audio_file=audio_file,
        torch_dtype=torch_dtype,
    )
    canonical = canonicalize_model_interface(model, task=task, input_names=prepared.names)

    wrapper = TranspileWrapper(canonical.module, weights_dir=weights_dir).eval()

    capture_started = time.perf_counter()
    captured = capture_model(wrapper, prepared.tensors)
    canonicalize_exported_graph(captured.ir_graph)
    optimize_graph(captured.ir_graph, config=fusion_config)
    tg = _lower_preoptimized_ir(captured.ir_graph)
    capture_finished = time.perf_counter()

    cached_inputs = [tensor.detach().cpu().numpy() for tensor in prepared.tensors]
    blank_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    return {
        "tg": tg,
        "model": model,
        "processor": processor,
        "model_config": model_config,
        "preprocessor_config": preprocessor_config,
        "tokenizer": tokenizer,
        "blank_token_id": int(blank_token_id) if blank_token_id is not None else None,
        "compile_time_ms": (capture_finished - capture_started) * 1000.0,
        "cached_inputs": cached_inputs,
        "input_names": tuple(prepared.names),
        "prepared_input_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(prepared.names, prepared.tensors)
        },
        "model_source": model_source,
        "weight_bindings": sum(
            1
            for value in captured.ir_graph.values.values()
            if isinstance(value.meta, dict) and isinstance(value.meta.get("path"), str)
        ),
    }


def _benchmark_handwritten(
    *,
    weights_dir: Path,
    audio_file: str,
    warmup: int,
    repeats: int,
    profile_repeats: int,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    model_handle = cactus_init(str(weights_dir), None, False)
    try:
        _warmup(lambda: _run_handwritten_once(model_handle=model_handle, audio_file=audio_file), warmup)

        runs: list[HandwrittenRun] = []
        for _ in range(max(repeats, 1)):
            runs.append(_run_handwritten_once(model_handle=model_handle, audio_file=audio_file))

        profile_path = (
            artifact_dir / "handwritten_profile.txt"
            if artifact_dir is not None and profile_repeats > 0
            else None
        )
        with _profile_env(profile_path):
            for _ in range(max(profile_repeats, 0)):
                _run_handwritten_once(model_handle=model_handle, audio_file=audio_file)

        profile_sections = _parse_profile_file(profile_path) if profile_path is not None else []
        profile_summary = _summarize_profile_sections(profile_sections)

        return {
            "weights_dir": str(weights_dir),
            "runs": [_roundtrip_jsonable(run) for run in runs],
            "wall_ms": asdict(_numeric_summary([run.wall_ms for run in runs])),
            "internal_total_time_ms": asdict(_numeric_summary([run.total_time_ms for run in runs])),
            "time_to_first_token_ms": asdict(_numeric_summary([run.time_to_first_token_ms for run in runs])),
            "decode_tps": asdict(_numeric_summary([run.decode_tps for run in runs])),
            "prefill_tps": asdict(_numeric_summary([run.prefill_tps for run in runs])),
            "decode_tokens": asdict(_numeric_summary([float(run.decode_tokens) for run in runs])),
            "confidence": asdict(_numeric_summary([run.confidence for run in runs])),
            "latest_transcript": runs[-1].transcript if runs else "",
            "profile_file": str(profile_path) if profile_path is not None else None,
            "profile_summary": profile_summary,
        }
    finally:
        cactus_destroy(model_handle)


def _benchmark_transpiled(
    *,
    model_id: str,
    audio_file: str,
    weights_dir: str | None,
    warmup: int,
    repeats: int,
    profile_repeats: int,
    torch_dtype: torch.dtype,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
    artifact_dir: Path | None,
    fusion_config: FusionConfig,
) -> dict[str, Any]:
    built = _build_transpiled_graph(
        model_id=model_id,
        audio_file=audio_file,
        weights_dir=weights_dir,
        torch_dtype=torch_dtype,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        fusion_config=fusion_config,
    )

    tg = built["tg"]
    model = built["model"]
    processor = built["processor"]
    input_names = built["input_names"]
    model_config = built["model_config"]
    preprocessor_config = built["preprocessor_config"]
    tokenizer = built["tokenizer"]
    blank_token_id = built["blank_token_id"]
    cached_inputs = built["cached_inputs"]

    _warmup(
        lambda: _run_transpiled_once(
            tg=tg,
            processor=processor,
            input_names=input_names,
            model_config=model_config,
            preprocessor_config=preprocessor_config,
            model=model,
            torch_dtype=torch_dtype,
            audio_file=audio_file,
            tokenizer=tokenizer,
            blank_token_id=blank_token_id,
        ),
        warmup,
    )
    _warmup(lambda: _run_transpiled_graph_only_once(tg=tg, cached_inputs=cached_inputs), warmup)

    runs: list[TranspiledRun] = []
    graph_only_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        runs.append(
            _run_transpiled_once(
                tg=tg,
                processor=processor,
                input_names=input_names,
                model_config=model_config,
                preprocessor_config=preprocessor_config,
                model=model,
                torch_dtype=torch_dtype,
                audio_file=audio_file,
                tokenizer=tokenizer,
                blank_token_id=blank_token_id,
            )
        )
        graph_only_ms.append(_run_transpiled_graph_only_once(tg=tg, cached_inputs=cached_inputs))

    profile_path = (
        artifact_dir / "transpiled_profile.txt"
        if artifact_dir is not None and profile_repeats > 0
        else None
    )
    with _profile_env(profile_path):
        for _ in range(max(profile_repeats, 0)):
            _run_transpiled_graph_only_once(tg=tg, cached_inputs=cached_inputs)

    profile_sections = _parse_profile_file(profile_path) if profile_path is not None else []
    profile_summary = _summarize_profile_sections(profile_sections)

    return {
        "model_source": built["model_source"],
        "compile_time_ms": built["compile_time_ms"],
        "weight_bindings": built["weight_bindings"],
        "prepared_input_shapes": built["prepared_input_shapes"],
        "runs": [_roundtrip_jsonable(run) for run in runs],
        "preprocess_ms": asdict(_numeric_summary([run.preprocess_ms for run in runs])),
        "execute_ms": asdict(_numeric_summary([run.execute_ms for run in runs])),
        "decode_ms": asdict(_numeric_summary([run.decode_ms for run in runs])),
        "total_ms": asdict(_numeric_summary([run.total_ms for run in runs])),
        "graph_only_ms": asdict(_numeric_summary(graph_only_ms)),
        "decode_tokens": asdict(_numeric_summary([float(run.decode_tokens) for run in runs])),
        "output_frames": asdict(_numeric_summary([float(run.output_frames) for run in runs])),
        "latest_transcript": runs[-1].transcript if runs else "",
        "profile_file": str(profile_path) if profile_path is not None else None,
        "profile_summary": profile_summary,
    }


def _print_summary(
    *,
    audio_duration_sec: float,
    handwritten: dict[str, Any] | None,
    transpiled: dict[str, Any] | None,
) -> None:
    print()
    print(f"audio_duration_sec={audio_duration_sec:.3f}")

    if handwritten is not None:
        hw_total = handwritten["internal_total_time_ms"]["mean"]
        hw_graph = None
        profile_total = handwritten.get("profile_summary", {}).get("graph_total_ms")
        if isinstance(profile_total, dict):
            hw_graph = profile_total.get("mean")
        hw_tok = handwritten["decode_tokens"]["mean"]
        hw_rtf = hw_total / max(audio_duration_sec * 1000.0, 1e-9)
        hw_xrt = (audio_duration_sec * 1000.0) / max(hw_total, 1e-9)
        hw_tok_s = hw_tok * 1000.0 / max(hw_total, 1e-9)
        print("handwritten:")
        print(f"  total_ms mean={hw_total:.3f}")
        print(f"  graph_profile_ms mean={hw_graph:.3f}" if hw_graph is not None else "  graph_profile_ms unavailable")
        print(f"  decode_tokens mean={hw_tok:.3f}")
        print(f"  output_tokens_per_sec={hw_tok_s:.3f}")
        print(f"  real_time_factor={hw_rtf:.3f}")
        print(f"  audio_x_realtime={hw_xrt:.3f}")

    if transpiled is not None:
        tr_total = transpiled["total_ms"]["mean"]
        tr_graph = transpiled["graph_only_ms"]["mean"]
        tr_tok = transpiled["decode_tokens"]["mean"]
        tr_rtf = tr_total / max(audio_duration_sec * 1000.0, 1e-9)
        tr_xrt = (audio_duration_sec * 1000.0) / max(tr_total, 1e-9)
        tr_tok_s = tr_tok * 1000.0 / max(tr_total, 1e-9)
        tr_graph_tok_s = tr_tok * 1000.0 / max(tr_graph, 1e-9)
        print("transpiled:")
        print(f"  total_ms mean={tr_total:.3f}")
        print(f"  graph_only_ms mean={tr_graph:.3f}")
        print(f"  decode_tokens mean={tr_tok:.3f}")
        print(f"  output_tokens_per_sec={tr_tok_s:.3f}")
        print(f"  graph_only_tokens_per_sec={tr_graph_tok_s:.3f}")
        print(f"  real_time_factor={tr_rtf:.3f}")
        print(f"  audio_x_realtime={tr_xrt:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare handwritten Cactus Parakeet performance against the transpiled "
            "Parakeet graph on the same WAV input, including per-op graph profiles."
        )
    )
    parser.add_argument("--model-id", default="nvidia/parakeet-ctc-1.1b")
    parser.add_argument("--audio-file", required=True)
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Converted Cactus weights directory. Reused for handwritten runtime and transpiled mmap bindings.",
    )
    parser.add_argument("--artifact-dir", default="", help="Optional directory for JSON summaries and raw profiles.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-handwritten", action="store_true")
    parser.add_argument("--skip-transpiled", action="store_true")
    parser.add_argument("--no-fuse-gated-deltanet", action="store_true")
    parser.add_argument("--no-fuse-rms-norm", action="store_true")
    parser.add_argument("--no-fuse-rope", action="store_true")
    parser.add_argument("--no-fuse-attention", action="store_true")
    parser.add_argument("--no-fuse-attention-block", action="store_true")
    parser.add_argument("--no-fuse-add-clipped", action="store_true")
    args = parser.parse_args()

    audio_file = str(Path(args.audio_file).resolve())
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"audio file does not exist: {audio_file}")

    torch_dtype = _parse_dtype(args.torch_dtype)
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    weights_dir_path = None
    if args.weights_dir.strip():
        weights_dir_path = _validate_weights_dir(args.weights_dir.strip(), model_id=args.model_id)
    else:
        weights_dir_path = _resolve_default_weights_dir(args.model_id)

    if not args.skip_handwritten and weights_dir_path is None:
        raise RuntimeError(
            "handwritten comparison requires a converted weights directory.\n"
            "\n"
            f"Create one with:\n"
            f"  cactus convert {args.model_id} {PROJECT_ROOT / 'weights' / args.model_id.split('/')[-1].lower()}\n"
            "\n"
            "Or rerun with --skip-handwritten."
        )

    fusion_config = FusionConfig(
        enable_gated_deltanet=not args.no_fuse_gated_deltanet,
        enable_rms_norm=not args.no_fuse_rms_norm,
        enable_rope=not args.no_fuse_rope,
        enable_attention=not args.no_fuse_attention,
        enable_attention_block=not args.no_fuse_attention_block,
        enable_add_clipped=not args.no_fuse_add_clipped,
    )

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "audio_file": audio_file,
        "audio_duration_sec": _audio_duration_seconds(audio_file),
        "weights_dir": str(weights_dir_path) if weights_dir_path is not None else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }

    handwritten_summary = None
    if not args.skip_handwritten and weights_dir_path is not None:
        handwritten_summary = _benchmark_handwritten(
            weights_dir=weights_dir_path,
            audio_file=audio_file,
            warmup=args.warmup,
            repeats=args.repeats,
            profile_repeats=args.profile_repeats,
            artifact_dir=artifact_dir,
        )
        summary["handwritten"] = handwritten_summary

    transpiled_summary = None
    if not args.skip_transpiled:
        transpiled_summary = _benchmark_transpiled(
            model_id=args.model_id,
            audio_file=audio_file,
            weights_dir=str(weights_dir_path) if weights_dir_path is not None else None,
            warmup=args.warmup,
            repeats=args.repeats,
            profile_repeats=args.profile_repeats,
            torch_dtype=torch_dtype,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            artifact_dir=artifact_dir,
            fusion_config=fusion_config,
        )
        summary["transpiled"] = transpiled_summary

    if handwritten_summary is not None and transpiled_summary is not None:
        hw_total = handwritten_summary["internal_total_time_ms"]["mean"]
        tr_total = transpiled_summary["total_ms"]["mean"]
        hw_graph = handwritten_summary.get("profile_summary", {}).get("graph_total_ms")
        tr_graph = transpiled_summary["graph_only_ms"]["mean"]
        summary["comparison"] = {
            "end_to_end_speedup_vs_handwritten": hw_total / max(tr_total, 1e-9),
            "graph_only_speedup_vs_handwritten": (
                float(hw_graph["mean"]) / max(tr_graph, 1e-9)
                if isinstance(hw_graph, dict) and "mean" in hw_graph
                else None
            ),
        }

    if artifact_dir is not None:
        summary_path = artifact_dir / "summary.json"
        summary_path.write_text(json.dumps(_roundtrip_jsonable(summary), indent=2, sort_keys=True) + "\n")
        print(f"saved_summary={summary_path}")
        if handwritten_summary is not None and handwritten_summary.get("profile_file"):
            print(f"saved_handwritten_profile={handwritten_summary['profile_file']}")
        if transpiled_summary is not None and transpiled_summary.get("profile_file"):
            print(f"saved_transpiled_profile={transpiled_summary['profile_file']}")

    _print_summary(
        audio_duration_sec=float(summary["audio_duration_sec"]),
        handwritten=handwritten_summary,
        transpiled=transpiled_summary,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
