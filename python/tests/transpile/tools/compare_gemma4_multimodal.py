from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TOOLS_DIR.parents[2]
PROJECT_ROOT = PYTHON_ROOT.parent

sys.path.insert(0, str(PYTHON_ROOT))

from cactus.bindings.cactus import cactus_complete
from cactus.bindings.cactus import cactus_destroy
from cactus.bindings.cactus import cactus_init
from cactus.bindings.cactus import cactus_reset
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.model_adapters import canonicalize_model_interface
from cactus.transpile.optimize_graph import FusionConfig
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.hf_model import TranspileWrapper
from cactus.transpile.hf_model import _graph_to_dict
from cactus.transpile.hf_model import _load_transformers_bundle
from cactus.transpile.hf_model import _lower_preoptimized_ir
from cactus.transpile.hf_model import _parse_dtype
from cactus.transpile.hf_model import _prepare_gemma4_multimodal_inputs
from cactus.transpile.hf_model import _serialize_json_compatible
from cactus.transpile.hf_model import _validate_weights_dir
from cactus.transpile.hf_model import _write_json


_DEFAULT_STOP_SEQUENCES = ("<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>")


@dataclass
class NumericSummary:
    mean: float
    min: float
    max: float
    median: float


@dataclass
class HandwrittenRun:
    wall_ms: float
    total_time_ms: float
    time_to_first_token_ms: float
    prefill_tps: float
    decode_tps: float
    decode_tokens: int
    total_tokens: int
    response: str
    response_json: dict[str, Any]


@dataclass
class TranspiledRun:
    preprocess_ms: float
    execute_ms: float
    total_ms: float
    time_to_first_token_ms: float
    prefill_tps: float
    decode_tps: float
    decode_tokens: int
    total_tokens: int
    response: str
    stop_reason: str
    generated_token_ids: list[int]
    input_shapes: dict[str, list[int]]


def _numeric_summary(values: list[float]) -> NumericSummary:
    arr = np.asarray(values, dtype=np.float64)
    return NumericSummary(
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        median=float(np.median(arr)),
    )


def _roundtrip_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(_serialize_json_compatible(value)))


def _resolve_default_weights_dir(model_id: str) -> Path | None:
    env_path = os.environ.get("CACTUS_TEST_GEMMA4_MODEL")
    if env_path:
        candidate = Path(env_path).resolve()
        if (candidate / "config.txt").exists():
            return candidate

    candidates = (
        PROJECT_ROOT / "weights" / "gemma4_int4",
        PROJECT_ROOT / "weights" / model_id.split("/")[-1].lower(),
        PROJECT_ROOT / "weights" / model_id.split("/")[-1].lower().replace("_", "-"),
    )
    for candidate in candidates:
        if (candidate / "config.txt").exists():
            return candidate
    return None


def _require_processor_tokenizer(processor_or_tokenizer: object) -> object:
    tokenizer_like = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if not hasattr(tokenizer_like, "decode") or not hasattr(tokenizer_like, "encode"):
        raise RuntimeError(
            "Gemma4 compare script requires a tokenizer-like object with encode/decode support."
        )
    return tokenizer_like


def _build_messages_payload(
    *,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    system_prompt: str = "",
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    normalized_system = system_prompt.strip()
    if normalized_system:
        messages.append({"role": "system", "content": normalized_system})

    user_message: dict[str, object] = {
        "role": "user",
        "content": prompt,
    }
    if image_files:
        user_message["images"] = [str(Path(path).resolve()) for path in image_files]
    if audio_file:
        user_message["audio"] = [str(Path(audio_file).resolve())]
    messages.append(user_message)
    return messages


def _encode_stop_sequences(tokenizer: object, stop_sequences: tuple[str, ...]) -> list[list[int]]:
    encoded: list[list[int]] = []
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return encoded
    for stop_sequence in stop_sequences:
        try:
            token_ids = list(encode(stop_sequence, add_special_tokens=False))
        except TypeError:
            token_ids = list(encode(stop_sequence))
        if token_ids:
            encoded.append([int(token_id) for token_id in token_ids])
    return encoded


def _has_suffix(token_ids: list[int], suffix: list[int]) -> bool:
    if not suffix or len(token_ids) < len(suffix):
        return False
    return token_ids[-len(suffix) :] == suffix


def _trim_stop_suffix(token_ids: list[int], stop_sequences: list[list[int]]) -> bool:
    for stop_sequence in stop_sequences:
        if _has_suffix(token_ids, stop_sequence):
            del token_ids[-len(stop_sequence) :]
            return True
    return False


def _decode_generated_text(tokenizer: object, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError(f"tokenizer does not expose decode(): {type(tokenizer).__name__}")
    try:
        return str(decode(token_ids, skip_special_tokens=skip_special_tokens))
    except TypeError:
        return str(decode(token_ids))


def _summarize_handwritten_runs(
    runs: list[HandwrittenRun],
    *,
    weights_dir: Path,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("expected at least one handwritten run")
    return {
        "weights_dir": str(weights_dir),
        "latest_response": runs[-1].response,
        "total_time_ms": asdict(_numeric_summary([run.total_time_ms for run in runs])),
        "time_to_first_token_ms": asdict(_numeric_summary([run.time_to_first_token_ms for run in runs])),
        "prefill_tps": asdict(_numeric_summary([run.prefill_tps for run in runs])),
        "decode_tps": asdict(_numeric_summary([run.decode_tps for run in runs])),
        "decode_tokens": asdict(_numeric_summary([float(run.decode_tokens) for run in runs])),
        "total_tokens": asdict(_numeric_summary([float(run.total_tokens) for run in runs])),
        "wall_ms": asdict(_numeric_summary([run.wall_ms for run in runs])),
        "runs": [_roundtrip_jsonable(asdict(run)) for run in runs],
    }


def _summarize_transpiled_runs(
    runs: list[TranspiledRun],
    *,
    compile_time_ms: float,
    model_source: str,
    weight_bindings: int,
    prepared_input_shapes: dict[str, list[int]],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("expected at least one transpiled run")
    return {
        "compile_time_ms": compile_time_ms,
        "model_source": model_source,
        "weight_bindings": weight_bindings,
        "prepared_input_shapes": prepared_input_shapes,
        "latest_response": runs[-1].response,
        "preprocess_ms": asdict(_numeric_summary([run.preprocess_ms for run in runs])),
        "execute_ms": asdict(_numeric_summary([run.execute_ms for run in runs])),
        "total_ms": asdict(_numeric_summary([run.total_ms for run in runs])),
        "time_to_first_token_ms": asdict(_numeric_summary([run.time_to_first_token_ms for run in runs])),
        "prefill_tps": asdict(_numeric_summary([run.prefill_tps for run in runs])),
        "decode_tps": asdict(_numeric_summary([run.decode_tps for run in runs])),
        "decode_tokens": asdict(_numeric_summary([float(run.decode_tokens) for run in runs])),
        "total_tokens": asdict(_numeric_summary([float(run.total_tokens) for run in runs])),
        "runs": [_roundtrip_jsonable(asdict(run)) for run in runs],
    }


def _prepare_repeat_inputs(
    processor: object,
    *,
    tokenizer: object,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    torch_dtype: torch.dtype,
    system_prompt: str,
    enable_thinking_if_supported: bool,
    max_new_tokens: int,
) -> tuple[object, int, float]:
    start = time.perf_counter()
    prepared = _prepare_gemma4_multimodal_inputs(
        processor,
        prompt=prompt,
        image_files=image_files,
        audio_file=audio_file,
        torch_dtype=torch_dtype,
        system_prompt=system_prompt,
        enable_thinking_if_supported=enable_thinking_if_supported,
        use_gemma4_chat_template=True,
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    prompt_tokens = 0
    for name, tensor in zip(prepared.names, prepared.tensors):
        if name == "input_ids":
            prompt_tokens = int(tensor.shape[1])
            break
    if prompt_tokens <= 0:
        raise RuntimeError("Gemma4 multimodal prepared inputs did not include a non-empty input_ids tensor")
    target_tokens = prompt_tokens + int(max_new_tokens)

    input_tensors: list[torch.Tensor] = []
    for name, tensor in zip(prepared.names, prepared.tensors):
        if name in {"input_ids", "attention_mask", "token_type_ids"}:
            padded_shape = list(tensor.shape)
            padded_shape[1] = target_tokens
            if name == "input_ids":
                padded = torch.full(
                    padded_shape,
                    int(pad_token_id),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            else:
                padded = torch.zeros(
                    padded_shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            padded[:, : tensor.shape[1]] = tensor
            input_tensors.append(padded)
        else:
            input_tensors.append(tensor)
    metadata = dict(prepared.metadata)
    metadata["prompt_token_count"] = prompt_tokens
    metadata["target_token_count"] = int(target_tokens)
    metadata["input_shapes"] = {
        name: list(tensor.shape)
        for name, tensor in zip(prepared.names, input_tensors)
    }
    prepared = type(prepared)(
        names=prepared.names,
        tensors=tuple(input_tensors),
        metadata=metadata,
    )
    preprocess_ms = (time.perf_counter() - start) * 1000.0
    return prepared, prompt_tokens, preprocess_ms


def _run_handwritten_once(
    *,
    model_handle: Any,
    messages_json: str,
    options_json: str,
) -> HandwrittenRun:
    cactus_reset(model_handle)
    streamed_token_ids: list[int] = []

    def _collect(_text: str, token_id: int) -> None:
        streamed_token_ids.append(int(token_id))

    start = time.perf_counter()
    response_json_text = cactus_complete(
        model_handle,
        messages_json,
        options_json,
        "[]",
        _collect,
        pcm_data=None,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    response_payload = json.loads(response_json_text)
    return HandwrittenRun(
        wall_ms=wall_ms,
        total_time_ms=float(response_payload.get("total_time_ms", wall_ms)),
        time_to_first_token_ms=float(response_payload.get("time_to_first_token_ms", 0.0)),
        prefill_tps=float(response_payload.get("prefill_tps", 0.0)),
        decode_tps=float(response_payload.get("decode_tps", 0.0)),
        decode_tokens=len(streamed_token_ids),
        total_tokens=int(response_payload.get("total_tokens", len(streamed_token_ids))),
        response=str(response_payload.get("response", "")),
        response_json=response_payload,
    )


def _run_transpiled_generation_once(
    *,
    tg,
    processor: object,
    tokenizer: object,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    torch_dtype: torch.dtype,
    system_prompt: str,
    enable_thinking_if_supported: bool,
    max_new_tokens: int,
    stop_sequences: tuple[str, ...],
    expected_input_shapes: dict[str, list[int]] | None = None,
    prepared_override: object | None = None,
    progress_label: str = "transpiled",
    progress_every_tokens: int = 8,
) -> TranspiledRun:
    if prepared_override is not None:
        prepared = type(prepared_override)(
            names=prepared_override.names,
            tensors=tuple(tensor.clone() for tensor in prepared_override.tensors),
            metadata=dict(prepared_override.metadata),
        )
        prompt_tokens = int(prepared.metadata.get("prompt_token_count", 0))
        preprocess_ms = 0.0
    else:
        prepared, prompt_tokens, preprocess_ms = _prepare_repeat_inputs(
            processor,
            tokenizer=tokenizer,
            prompt=prompt,
            image_files=image_files,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt,
            enable_thinking_if_supported=enable_thinking_if_supported,
            max_new_tokens=max_new_tokens,
        )
    inputs_by_name = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(prepared.names, prepared.tensors)
    }
    if expected_input_shapes is not None:
        for name, tensor in zip(prepared.names, prepared.tensors):
            expected_shape = expected_input_shapes.get(name)
            if expected_shape is None:
                continue
            actual_shape = list(tensor.shape)
            if actual_shape != list(expected_shape):
                raise RuntimeError(
                    f"runtime prepared input shape mismatch for {name}: "
                    f"compiled with {list(expected_shape)}, got {actual_shape}"
                )
    if "input_ids" not in inputs_by_name:
        raise RuntimeError("transpiled Gemma4 generation requires input_ids")

    target_tokens = int(prepared.metadata["target_token_count"])

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    padded_inputs: dict[str, np.ndarray] = dict(inputs_by_name)
    generated_ids: list[int] = []
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    current_length = prompt_tokens
    first_token_ms = 0.0
    execute_start = time.perf_counter()
    stop_reason = "max_new_tokens"
    progress_interval = max(1, int(progress_every_tokens))

    for step_index in range(max_new_tokens):
        runtime_inputs = [padded_inputs[name] for name in prepared.names]
        if step_index == 0:
            print("transpiled_step0_set_inputs_begin=true", flush=True)
        tg.set_inputs(runtime_inputs)
        if step_index == 0:
            print("transpiled_step0_set_inputs_done=true", flush=True)
            print("transpiled_step0_execute_begin=true", flush=True)
        logits = tg.execute()[0].numpy()
        if step_index == 0:
            print("transpiled_step0_execute_done=true", flush=True)
        if logits.ndim == 3:
            token_position = current_length - 1
            if logits.shape[1] == 1:
                token_position = 0
            next_token_id = int(np.argmax(logits[0, token_position]))
        elif logits.ndim == 2:
            next_token_id = int(np.argmax(logits[0]))
        else:
            raise RuntimeError(f"unexpected transpiled logits rank: {logits.ndim}")
        generated_ids.append(next_token_id)

        if step_index == 0:
            first_token_ms = preprocess_ms + (time.perf_counter() - execute_start) * 1000.0

        if eos_token_id is not None and next_token_id == int(eos_token_id):
            stop_reason = "eos_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break

        if current_length >= target_tokens:
            stop_reason = "context_limit"
            break

        padded_inputs["input_ids"][0, current_length] = next_token_id
        padded_inputs["attention_mask"][0, current_length] = 1
        padded_inputs["token_type_ids"][0, current_length] = 0
        if step_index == 0:
            print("transpiled_step0_state_update_done=true", flush=True)
        current_length += 1
        generated_count = len(generated_ids)
        if generated_count % progress_interval == 0 or generated_count == max_new_tokens:
            elapsed_ms = (time.perf_counter() - execute_start) * 1000.0
            print(
                f"{progress_label}_decode_progress={generated_count}/{max_new_tokens} "
                f"elapsed_ms={elapsed_ms:.3f}",
                flush=True,
            )

    execute_ms = (time.perf_counter() - execute_start) * 1000.0
    response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False).strip()
    decode_time_ms = max(0.0, execute_ms + preprocess_ms - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )
    prefill_tps = (prompt_tokens * 1000.0) / first_token_ms if first_token_ms > 0.0 else 0.0
    print(
        f"{progress_label}_decode_done=true tokens={len(generated_ids)} "
        f"stop_reason={stop_reason} execute_ms={execute_ms:.3f}",
        flush=True,
    )

    return TranspiledRun(
        preprocess_ms=preprocess_ms,
        execute_ms=execute_ms,
        total_ms=preprocess_ms + execute_ms,
        time_to_first_token_ms=first_token_ms,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        decode_tokens=len(generated_ids),
        total_tokens=prompt_tokens + len(generated_ids),
        response=response,
        stop_reason=stop_reason,
        generated_token_ids=generated_ids,
        input_shapes={
            name: list(tensor.shape)
            for name, tensor in zip(prepared.names, prepared.tensors)
        },
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"model_id={summary['model_id']}")
    print(f"prompt={summary['prompt']!r}")
    if summary.get("image_files"):
        print(f"image_files={summary['image_files']}")
    if summary.get("audio_file"):
        print(f"audio_file={summary['audio_file']}")

    handwritten = summary.get("handwritten")
    if handwritten is not None:
        hw_total = handwritten["total_time_ms"]["mean"]
        hw_ttft = handwritten["time_to_first_token_ms"]["mean"]
        hw_tokens = handwritten["decode_tokens"]["mean"]
        hw_tps = hw_tokens * 1000.0 / max(hw_total, 1e-9)
        print("handwritten:")
        print(f"  total_ms mean={hw_total:.3f}")
        print(f"  time_to_first_token_ms mean={hw_ttft:.3f}")
        print(f"  decode_tokens mean={hw_tokens:.3f}")
        print(f"  output_tokens_per_sec={hw_tps:.3f}")
        print(f"  latest_response={handwritten['latest_response']!r}")

    transpiled = summary.get("transpiled")
    if transpiled is not None:
        tr_total = transpiled["total_ms"]["mean"]
        tr_ttft = transpiled["time_to_first_token_ms"]["mean"]
        tr_tokens = transpiled["decode_tokens"]["mean"]
        tr_tps = tr_tokens * 1000.0 / max(tr_total, 1e-9)
        print("transpiled:")
        print(f"  compile_time_ms={transpiled['compile_time_ms']:.3f}")
        print(f"  total_ms mean={tr_total:.3f}")
        print(f"  time_to_first_token_ms mean={tr_ttft:.3f}")
        print(f"  decode_tokens mean={tr_tokens:.3f}")
        print(f"  output_tokens_per_sec={tr_tps:.3f}")
        print(f"  weight_bindings={transpiled['weight_bindings']}")
        print(f"  latest_response={transpiled['latest_response']!r}")

    comparison = summary.get("comparison")
    if comparison is not None:
        print("comparison:")
        print(
            "  end_to_end_speedup_vs_handwritten="
            f"{comparison['end_to_end_speedup_vs_handwritten']:.6f}"
        )
        print(f"  responses_match={comparison['responses_match']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare handwritten Cactus Gemma4 multimodal generation against the "
            "transpiled Gemma4 multimodal graph on the same prompt, image, and audio."
        )
    )
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--image-file", action="append", default=[])
    parser.add_argument("--audio-file", default="")
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Converted Cactus weights directory. Required for handwritten comparison and reused for transpiled mmap bindings.",
    )
    parser.add_argument("--artifact-dir", default="", help="Optional directory for summary and transpile artifacts.")
    parser.add_argument("--graph-filename", default="graph.cactus")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--transpiled-progress-every-tokens",
        type=int,
        default=1,
        help=(
            "Emit transpiled decode progress every N generated tokens. "
            "Use 1 for per-token visibility on long CPU-bound Gemma4 runs."
        ),
    )
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-handwritten", action="store_true")
    parser.add_argument("--skip-transpiled", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--stop-sequence", action="append", default=[])
    parser.add_argument("--no-fuse-gated-deltanet", action="store_true")
    parser.add_argument("--no-fuse-rms-norm", action="store_true")
    parser.add_argument("--no-fuse-rope", action="store_true")
    parser.add_argument("--no-fuse-attention", action="store_true")
    parser.add_argument("--no-fuse-attention-block", action="store_true")
    parser.add_argument("--no-fuse-add-clipped", action="store_true")
    args = parser.parse_args()

    if args.skip_handwritten and args.skip_transpiled:
        raise RuntimeError("cannot skip both handwritten and transpiled runs")

    image_files = tuple(str(Path(path).resolve()) for path in args.image_file if str(path).strip())
    audio_file = str(Path(args.audio_file).resolve()) if args.audio_file.strip() else None
    if audio_file and not Path(audio_file).exists():
        raise FileNotFoundError(f"audio file does not exist: {audio_file}")
    for image_file in image_files:
        if not Path(image_file).exists():
            raise FileNotFoundError(f"image file does not exist: {image_file}")

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
            "handwritten Gemma4 comparison requires a converted weights directory.\n"
            "\n"
            "Create one with something like:\n"
            f"  cactus convert {args.model_id} /path/to/gemma4_weights\n"
            "\n"
            "Then rerun with --weights-dir /path/to/gemma4_weights, or rerun with --skip-handwritten."
        )

    stop_sequences = tuple(args.stop_sequence) if args.stop_sequence else _DEFAULT_STOP_SEQUENCES
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
        "prompt": args.prompt,
        "system_prompt": args.system_prompt,
        "image_files": list(image_files),
        "audio_file": audio_file,
        "weights_dir": str(weights_dir_path) if weights_dir_path is not None else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "transpiled_progress_every_tokens": int(args.transpiled_progress_every_tokens),
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "enable_thinking": bool(args.enable_thinking),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "stop_sequences": list(stop_sequences),
    }

    messages_payload = _build_messages_payload(
        prompt=args.prompt,
        image_files=image_files,
        audio_file=audio_file,
        system_prompt=args.system_prompt,
    )
    messages_json = json.dumps(messages_payload)
    options_json = json.dumps(
        {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_tokens": int(args.max_new_tokens),
            "stop_sequences": list(stop_sequences),
            "enable_thinking_if_supported": bool(args.enable_thinking),
            "auto_handoff": False,
            "telemetry_enabled": False,
        }
    )

    transpiled_runs: list[TranspiledRun] = []
    if not args.skip_transpiled:
        model_source, processor_or_tokenizer, model, _ = _load_transformers_bundle(
            model_id=args.model_id,
            task="multimodal_causal_lm_logits",
            torch_dtype=torch_dtype,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        processor = processor_or_tokenizer
        tokenizer = _require_processor_tokenizer(processor_or_tokenizer)
        initial_prepared, _, initial_prepare_ms = _prepare_repeat_inputs(
            processor,
            tokenizer=tokenizer,
            prompt=args.prompt,
            image_files=image_files,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
            system_prompt=args.system_prompt,
            enable_thinking_if_supported=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
        canonical = canonicalize_model_interface(
            model,
            task="multimodal_causal_lm_logits",
            input_names=initial_prepared.names,
            weights_dir=str(weights_dir_path) if weights_dir_path is not None else None,
        )
        if hasattr(canonical.module, "last_token_logits_only"):
            canonical.module.last_token_logits_only = True
        prime_static_features = getattr(canonical.module, "prime_static_multimodal_features", None)
        if callable(prime_static_features):
            prime_static_features(*initial_prepared.tensors)
        wrapper = TranspileWrapper(
            canonical.module,
            weights_dir=str(weights_dir_path) if weights_dir_path is not None else None,
        ).eval()

        compile_start = time.perf_counter()
        print("transpile_capture_begin=true", flush=True)
        prepare_cpu_float32_capture = getattr(canonical.module, "prepare_cpu_float32_capture", None)
        restore_cpu_float32_capture = getattr(canonical.module, "restore_cpu_float32_capture", None)
        if callable(prepare_cpu_float32_capture):
            prepare_cpu_float32_capture()
        try:
            captured = capture_model(wrapper, initial_prepared.tensors)
        finally:
            if callable(restore_cpu_float32_capture):
                restore_cpu_float32_capture()
        print("transpile_capture_done=true", flush=True)
        raw_ir_graph = copy.deepcopy(captured.ir_graph)
        print("transpile_canonicalize_begin=true", flush=True)
        canonicalize_exported_graph(captured.ir_graph)
        print("transpile_canonicalize_done=true", flush=True)
        print("transpile_optimize_begin=true", flush=True)
        optimize_graph(captured.ir_graph, config=fusion_config)
        print("transpile_optimize_done=true", flush=True)
        optimized_ir_graph = copy.deepcopy(captured.ir_graph)
        print("transpile_lower_begin=true", flush=True)
        tg = _lower_preoptimized_ir(captured.ir_graph)
        print("transpile_lower_done=true", flush=True)
        compile_time_ms = (time.perf_counter() - compile_start) * 1000.0

        op_counts: dict[str, int] = {}
        for node_id in optimized_ir_graph.order:
            op = optimized_ir_graph.nodes[node_id].op
            op_counts[op] = op_counts.get(op, 0) + 1
        weight_bindings = sum(
            1
            for value in optimized_ir_graph.values.values()
            if isinstance(value.meta, dict) and isinstance(value.meta.get("path"), str)
        )

        summary["transpile_compile"] = {
            "model_source": model_source,
            "raw_ir_nodes": len(raw_ir_graph.order),
            "optimized_ir_nodes": len(optimized_ir_graph.order),
            "weight_bindings": weight_bindings,
            "compile_time_ms": compile_time_ms,
            "initial_prepare_ms": initial_prepare_ms,
            "ops": op_counts,
        }
        prompt_token_count = int(initial_prepared.metadata.get("prompt_token_count", 0))
        target_token_count = int(initial_prepared.metadata.get("target_token_count", 0))
        total_transpiled_runs = int(args.warmup) + int(args.repeats)
        print(
            "transpiled_decode_plan="
            f"prompt_tokens:{prompt_token_count} "
            f"target_tokens:{target_token_count} "
            f"max_new_tokens:{int(args.max_new_tokens)} "
            f"runs:{total_transpiled_runs} "
            f"progress_every_tokens:{int(args.transpiled_progress_every_tokens)}",
            flush=True,
        )

        if weights_dir_path is not None and weight_bindings == 0:
            raise RuntimeError(
                f"No weight bindings were resolved from {weights_dir_path}\n"
                "\n"
                "The converted weights folder exists, but none of the captured constants matched weights_manifest.json.\n"
                "Re-convert Gemma4 with the current converter before benchmarking."
            )

        if artifact_dir is not None:
            _write_json(
                artifact_dir / "raw_ir.json",
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": "multimodal_causal_lm_logits",
                    "inputs": _serialize_json_compatible(initial_prepared.metadata),
                    "graph": _graph_to_dict(raw_ir_graph),
                },
            )
            _write_json(
                artifact_dir / "optimized_ir.json",
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": "multimodal_causal_lm_logits",
                    "inputs": _serialize_json_compatible(initial_prepared.metadata),
                    "graph": _graph_to_dict(optimized_ir_graph),
                },
            )
            print(f"saved_raw_ir={artifact_dir / 'raw_ir.json'}")
            print(f"saved_optimized_ir={artifact_dir / 'optimized_ir.json'}")
            if weight_bindings > 0:
                skip_reason = (
                    "skipped saved_graph because graph serialization does not yet preserve "
                    "mmap-bound weights/embeddings required by this transpiled Gemma4 graph"
                )
                summary["transpile_compile"]["saved_graph_skipped_reason"] = skip_reason
                print(f"note={skip_reason}")
            else:
                graph_path = artifact_dir / args.graph_filename
                tg.graph.save(graph_path)
                print(f"saved_graph={graph_path}")

        for _ in range(args.warmup):
            warmup_index = _ + 1
            print(f"transpiled_warmup_begin={warmup_index}/{args.warmup}", flush=True)
            _run_transpiled_generation_once(
                tg=tg,
                processor=processor,
                tokenizer=tokenizer,
                prompt=args.prompt,
                image_files=image_files,
                audio_file=audio_file,
                torch_dtype=torch_dtype,
                system_prompt=args.system_prompt,
                enable_thinking_if_supported=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                stop_sequences=stop_sequences,
                expected_input_shapes=dict(initial_prepared.metadata.get("input_shapes", {})),
                prepared_override=initial_prepared,
                progress_label=f"transpiled_warmup{warmup_index}",
                progress_every_tokens=args.transpiled_progress_every_tokens,
            )
        for _ in range(args.repeats):
            repeat_index = _ + 1
            print(f"transpiled_repeat_begin={repeat_index}/{args.repeats}", flush=True)
            transpiled_runs.append(
                _run_transpiled_generation_once(
                    tg=tg,
                    processor=processor,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    image_files=image_files,
                    audio_file=audio_file,
                    torch_dtype=torch_dtype,
                    system_prompt=args.system_prompt,
                    enable_thinking_if_supported=args.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    stop_sequences=stop_sequences,
                    expected_input_shapes=dict(initial_prepared.metadata.get("input_shapes", {})),
                    prepared_override=initial_prepared,
                    progress_label=f"transpiled_repeat{repeat_index}",
                    progress_every_tokens=args.transpiled_progress_every_tokens,
                )
            )

        summary["transpiled"] = _summarize_transpiled_runs(
            transpiled_runs,
            compile_time_ms=compile_time_ms,
            model_source=model_source,
            weight_bindings=weight_bindings,
            prepared_input_shapes=dict(initial_prepared.metadata.get("input_shapes", {})),
        )

        # Keep the HF/export capture stack alive until the transpiled graph has
        # completed execution. Gemma4 teardown before first execute has been
        # unstable in practice, even though the lowered graph is already built.
        del model
        del wrapper
        del canonical
        del captured
        gc.collect()

    handwritten_runs: list[HandwrittenRun] = []
    if not args.skip_handwritten and weights_dir_path is not None:
        model_handle = cactus_init(str(weights_dir_path), None, False)
        try:
            for _ in range(args.warmup):
                warmup_index = _ + 1
                print(f"handwritten_warmup_begin={warmup_index}/{args.warmup}", flush=True)
                _run_handwritten_once(
                    model_handle=model_handle,
                    messages_json=messages_json,
                    options_json=options_json,
                )
            for _ in range(args.repeats):
                repeat_index = _ + 1
                print(f"handwritten_repeat_begin={repeat_index}/{args.repeats}", flush=True)
                handwritten_runs.append(
                    _run_handwritten_once(
                        model_handle=model_handle,
                        messages_json=messages_json,
                        options_json=options_json,
                    )
                )
        finally:
            cactus_destroy(model_handle)
        summary["handwritten"] = _summarize_handwritten_runs(
            handwritten_runs,
            weights_dir=weights_dir_path,
        )

    if handwritten_runs and transpiled_runs:
        handwritten_latest = handwritten_runs[-1].response.strip()
        transpiled_latest = transpiled_runs[-1].response.strip()
        summary["comparison"] = {
            "end_to_end_speedup_vs_handwritten": (
                summary["handwritten"]["total_time_ms"]["mean"]
                / max(summary["transpiled"]["total_ms"]["mean"], 1e-9)
            ),
            "responses_match": handwritten_latest == transpiled_latest,
            "handwritten_latest_response": handwritten_latest,
            "transpiled_latest_response": transpiled_latest,
        }

    if artifact_dir is not None:
        summary_path = artifact_dir / "summary.json"
        summary_path.write_text(json.dumps(_roundtrip_jsonable(summary), indent=2, sort_keys=True) + "\n")
        print(f"saved_summary={summary_path}")

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
