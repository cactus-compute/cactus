"""Pre-refactor microbenchmark for the Python runtime orchestration code.

The Python pipeline currently being ported to C++ is the *orchestration*
layer that wraps the C++ graph kernels — image preprocessing (PIL +
NumPy), the TDT per-step decode loop, and the cached-step causal-LM
decode loop. Graph execution itself is already C++, so this bench
isolates the Python overhead that the port will replace.

We do not need real model weights for this: the C++ graph calls inside
each loop are mocked with a no-op or a tiny NumPy op that returns the
same shape the real graph would. That isolates Python wall time per
iteration, which is exactly what the port saves.

Captures: total wall, per-iteration mean, p50/p95 (where iteration count
is meaningful), peak RSS. Output: ``results.json`` in this folder.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "cactus-engine" / "tests" / "assets"
DEFAULT_OUT = Path(__file__).resolve().parent / "results.json"


@dataclass
class BenchResult:
    name: str
    description: str
    ok: bool
    error: str | None
    iterations: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    peak_rss_mb: float
    notes: dict[str, Any] = field(default_factory=dict)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _timed(fn: Callable[[], None], *, iterations: int) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summarize(name: str, description: str, samples: list[float], notes: dict[str, Any]) -> BenchResult:
    return BenchResult(
        name=name,
        description=description,
        ok=True,
        error=None,
        iterations=len(samples),
        total_ms=sum(samples),
        mean_ms=statistics.fmean(samples),
        p50_ms=statistics.median(samples),
        p95_ms=samples[max(0, int(round(0.95 * (len(samples) - 1))))] if samples else 0.0,
        peak_rss_mb=_peak_rss_mb(),
        notes=notes,
    )


def _failed(name: str, description: str, exc: BaseException) -> BenchResult:
    return BenchResult(
        name=name,
        description=description,
        ok=False,
        error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
        iterations=0,
        total_ms=0.0,
        mean_ms=0.0,
        p50_ms=0.0,
        p95_ms=0.0,
        peak_rss_mb=_peak_rss_mb(),
        notes={},
    )


def bench_audio_preprocess(iterations: int) -> BenchResult:
    """Calls the public Cactus audio frontend on a real wav. Most of the
    heavy lifting is already C++ via the bindings; this measures the
    end-to-end wrapper cost (sample-rate detect, padding, reshape)."""
    name = "audio_preprocess.prepare_cactus_audio_features"
    desc = "End-to-end audio feature wrapper (C++ frontend + Python reshape/padding)"
    try:
        import torch
        from cactus.transpile.audio_preprocess import prepare_cactus_audio_features
    except Exception as exc:
        return _failed(name, desc, exc)

    wav = ASSETS / "test.wav"
    if not wav.exists():
        return _failed(name, desc, FileNotFoundError(str(wav)))

    def call() -> None:
        prepare_cactus_audio_features(
            str(wav),
            model_type="parakeet_tdt",
            expected_frames=None,
            expected_mels=128,
            torch_dtype=torch.float16,
            layout="frames_mels",
        )

    try:
        call()  # warmup
        samples = _timed(call, iterations=iterations)
    except Exception as exc:
        return _failed(name, desc, exc)
    return _summarize(name, desc, samples, {"audio_file": str(wav)})


def bench_image_preprocess_gemma4(iterations: int) -> BenchResult:
    """Calls _prepare_gemma4_native_image_tensors with a synthetic processor.
    All of this work is pure Python/NumPy/PIL today; it is exactly what
    moves to C++ in the port."""
    name = "multimodal_runtime._prepare_gemma4_native_image_tensors"
    desc = "Pure-Python Gemma4 image preprocessing (PIL load+resize, NumPy patch extract)"
    try:
        from cactus.transpile.multimodal_runtime import _prepare_gemma4_native_image_tensors
    except Exception as exc:
        return _failed(name, desc, exc)

    img = ASSETS / "test_monkey.png"
    if not img.exists():
        return _failed(name, desc, FileNotFoundError(str(img)))

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=16,
            pooling_kernel_size=3,
            max_soft_tokens=280,
            do_rescale=True,
            do_normalize=True,
            rescale_factor=1.0 / 255.0,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
    )
    image_files = (str(img),)

    def call() -> None:
        out = _prepare_gemma4_native_image_tensors(processor, image_files)
        assert out is not None

    try:
        call()
        samples = _timed(call, iterations=iterations)
    except Exception as exc:
        return _failed(name, desc, exc)
    return _summarize(name, desc, samples, {"image_file": str(img)})


def bench_image_load_basic(iterations: int) -> BenchResult:
    name = "multimodal_runtime._load_image_inputs"
    desc = "PIL Image.open + convert('RGB') + resize_static_image"
    try:
        from cactus.transpile.multimodal_runtime import _load_image_inputs
    except Exception as exc:
        return _failed(name, desc, exc)
    img = ASSETS / "test_monkey.png"
    if not img.exists():
        return _failed(name, desc, FileNotFoundError(str(img)))

    image_files = (str(img),)

    def call() -> None:
        _load_image_inputs(image_files)

    try:
        call()
        samples = _timed(call, iterations=iterations)
    except Exception as exc:
        return _failed(name, desc, exc)
    return _summarize(name, desc, samples, {"image_file": str(img)})


def bench_tdt_inner_loop(iterations: int) -> BenchResult:
    """Replicates the per-step Python overhead from tdt_runtime.py:151-196
    with a mock step() that returns precomputed logits. Each iteration
    represents one inner-loop pass of the TDT decoder."""
    import numpy as np

    name = "tdt_runtime.inner_loop_step"
    desc = "Per-step TDT Python wrapper overhead (argmax + state copy + emit logic), graph step mocked"
    token_class_count = 1024
    num_durations = 5
    total_classes = token_class_count + num_durations
    hidden_dim = 1024
    state_dim = 320

    fake_logits = np.random.randn(1, total_classes).astype(np.float32)
    fake_states = (
        np.zeros((1, 1, state_dim), dtype=np.float32),
        np.zeros((1, 1, state_dim), dtype=np.float32),
    )

    def mock_step(frame, last_token, states):
        return fake_logits, fake_states

    emitted: list[int] = []
    durations = list(range(num_durations))
    blank_id = token_class_count - 1
    states = fake_states

    def call() -> None:
        last_token = 0
        symbols_added = 0
        frame = np.zeros((1, hidden_dim), dtype=np.float32)
        local_states = states
        while symbols_added < 1:
            logits, next_states = mock_step(frame, last_token, local_states)
            logits_array = np.asarray(logits, dtype=np.float32)
            tc = total_classes - num_durations
            token_scores = logits_array[:, :tc]
            duration_scores = logits_array[:, tc:]
            next_token = int(np.argmax(token_scores[0]))
            duration_index = int(np.argmax(duration_scores[0]))
            skip = int(durations[min(duration_index, len(durations) - 1)])
            if next_token != blank_id:
                emitted.append(next_token)
                last_token = next_token
                local_states = tuple(
                    np.ascontiguousarray(np.asarray(state).copy()) for state in next_states
                )
            symbols_added += 1
            if skip > 0:
                break

    try:
        call()
        samples = _timed(call, iterations=iterations)
    except Exception as exc:
        return _failed(name, desc, exc)
    return _summarize(name, desc, samples, {"token_class_count": token_class_count, "hidden_dim": hidden_dim})


def bench_cached_step_inner_loop(iterations: int) -> BenchResult:
    """Replicates the per-token Python overhead from the Gemma4 cached-step
    decode loop in component_bundle_runtime.py:2207-2234.

    Mocks the underlying graph.execute() with a no-op; measures the
    per-token Python wrapper cost (argmax, stop checks, buffer fill,
    numpy conversions, encoder->decoder copy)."""
    import numpy as np

    name = "component_bundle_runtime.cached_step_inner_loop"
    desc = "Per-token cached-step decode Python overhead (argmax+stop+buffer fill+numpy copy), graph mocked"
    vocab = 256128
    hidden = 2304
    per_layer = 32
    fake_logits = np.random.randn(1, 1, vocab).astype(np.float32)
    fake_inputs_embeds = np.zeros((1, 1, hidden), dtype=np.float32)
    fake_per_layer = np.zeros((1, 1, per_layer), dtype=np.float32)
    fake_position = np.zeros((1, 1), dtype=np.int64)

    lm_encoder_input_ids = np.zeros((1, 1), dtype=np.int64)
    lm_encoder_position = np.zeros((1, 1), dtype=np.int64)
    decoder_inputs_embeds = np.zeros((1, 1, hidden), dtype=np.float32)
    decoder_per_layer = np.zeros((1, 1, per_layer), dtype=np.float32)
    decoder_position = np.zeros((1, 1), dtype=np.int64)

    stop_token_ids: set[int] = set()
    encoded_stop_sequences: tuple = ()

    def _trim_stop_suffix(_ids, _seqs):
        return False

    def _run_step_token(token_id: int, position_id: int) -> np.ndarray:
        lm_encoder_input_ids.fill(int(token_id))
        lm_encoder_position.fill(int(position_id))
        # graph.execute() — mocked as no-op
        decoder_inputs_embeds[:] = fake_inputs_embeds
        decoder_per_layer[:] = fake_per_layer
        decoder_position[:] = fake_position
        # decoder graph.execute() — mocked as no-op
        return np.asarray(fake_logits)

    requested_tokens = 32

    def call() -> None:
        generated_ids: list[int] = []
        next_position_id = 0
        logits = _run_step_token(0, next_position_id)
        for step_index in range(requested_tokens):
            next_token_id = int(np.argmax(logits[0, -1]))
            generated_ids.append(next_token_id)
            if next_token_id in stop_token_ids:
                break
            if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
                break
            if step_index + 1 >= requested_tokens:
                break
            logits = _run_step_token(next_token_id, next_position_id)
            next_position_id += 1

    try:
        call()
        samples = _timed(call, iterations=iterations)
    except Exception as exc:
        return _failed(name, desc, exc)
    return _summarize(
        name,
        desc,
        samples,
        {"vocab_size": vocab, "tokens_per_iter": requested_tokens, "hidden_dim": hidden, "per_layer_dim": per_layer},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations-light", type=int, default=10)
    parser.add_argument("--iterations-loop", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    benches = [
        ("audio_preprocess", bench_audio_preprocess, args.iterations_light),
        ("image_preprocess_gemma4", bench_image_preprocess_gemma4, args.iterations_light),
        ("image_load_basic", bench_image_load_basic, args.iterations_light),
        ("tdt_inner_loop", bench_tdt_inner_loop, args.iterations_loop),
        ("cached_step_inner_loop", bench_cached_step_inner_loop, args.iterations_loop),
    ]

    results: list[BenchResult] = []
    for label, fn, iters in benches:
        print(f"[{label}] running ({iters} iterations)...", flush=True)
        r = fn(iters)
        status = "OK " if r.ok else "FAIL"
        print(
            f"[{label}] {status} mean={r.mean_ms:.4f}ms p50={r.p50_ms:.4f}ms "
            f"p95={r.p95_ms:.4f}ms total={r.total_ms:.1f}ms peak_rss={r.peak_rss_mb:.1f}MB",
            flush=True,
        )
        if not r.ok:
            print(r.error, file=sys.stderr)
        results.append(r)

    payload = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "commit": os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip(),
        "note": (
            "Pre-refactor Python orchestration microbench. Graph kernels are "
            "mocked: numbers reflect ONLY the per-call/per-iteration Python "
            "wrapper overhead that the C++ port replaces."
        ),
        "benches": [
            {
                "name": r.name,
                "description": r.description,
                "ok": r.ok,
                "error": r.error,
                "iterations": r.iterations,
                "total_ms": round(r.total_ms, 4),
                "mean_ms": round(r.mean_ms, 4),
                "p50_ms": round(r.p50_ms, 4),
                "p95_ms": round(r.p95_ms, 4),
                "peak_rss_mb": round(r.peak_rss_mb, 2),
                "notes": r.notes,
            }
            for r in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
