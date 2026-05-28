"""Top-level GPU transpile pipeline.

Takes the captured HF model + the component specs the graph transpiler
already built, and produces the per-model GPU artifact bundle.

Layout of the emitted bundle (under ``<artifact_dir>/components/gpu/``):

    weights.bin       — packed int4/fp16 weights, GPU-ready layout
    scales.bin        — per-group scales (fp16) for int4 weights
    embedding.bin     — fp16 embedding table (and tied LM head, if shared)
    gpu_plan.json     — dispatch plan (per-layer, per-op kernel calls)

The runtime side reads ``gpu_plan.json``, builds Metal pipelines per the
plan, and dispatches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .plan import build_gpu_plan
from .weight_pack import pack_decoder_weights


def run_gpu_pipeline(
    hf_model: torch.nn.Module,
    artifact_dir: Path,
    *,
    enabled: bool = True,
    quantize_bits: int | None = None,
) -> dict[str, str]:
    """Build a GPU bundle for the captured model.

    Returns a dict suitable for merging into ``manifest.json`` — keys
    are ``gpu_plan``, ``gpu_weights``, ``gpu_scales``, ``gpu_embedding``
    (each pointing to the relative file path inside the bundle).
    """
    if not enabled:
        return {}

    artifact_dir = Path(artifact_dir)
    gpu_dir = artifact_dir / "components" / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)

    print(f"gpu.pipeline: emitting GPU bundle to {gpu_dir}")

    # Step 1: build the dispatch plan from the HF model's architecture.
    plan = build_gpu_plan(hf_model, quantize_bits=quantize_bits)
    plan_path = gpu_dir / "gpu_plan.json"

    # Step 2: pack the weights. This MUTATES the plan ops to record byte
    # offsets per weight; we serialize the plan AFTER packing so the runtime
    # sees the offsets.
    weights_path = gpu_dir / "weights.bin"
    scales_path  = gpu_dir / "scales.bin"
    embed_path   = gpu_dir / "embedding.bin"
    pack_decoder_weights(
        hf_model,
        plan,
        weights_path=weights_path,
        scales_path=scales_path,
        embedding_path=embed_path,
        quantize_bits=quantize_bits,
    )

    # Step 3: serialize the populated plan.
    plan.write(plan_path)
    print(f"gpu.pipeline: wrote plan -> {plan_path}")
    print(
        "gpu.pipeline: wrote weights ({:.1f} MB) + scales ({:.1f} MB) + "
        "embedding ({:.1f} MB)".format(
            weights_path.stat().st_size / 1e6,
            scales_path.stat().st_size / 1e6 if scales_path.exists() else 0.0,
            embed_path.stat().st_size / 1e6 if embed_path.exists() else 0.0,
        )
    )

    return {
        "gpu_plan":      "components/gpu/gpu_plan.json",
        "gpu_weights":   "components/gpu/weights.bin",
        "gpu_scales":    "components/gpu/scales.bin" if scales_path.exists() else "",
        "gpu_embedding": "components/gpu/embedding.bin" if embed_path.exists() else "",
    }
