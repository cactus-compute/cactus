"""NPU transpiler orchestrator.

Mirrors the role of `component_pipeline.execute_component_pipeline` but
for CoreML. Invoked from `hf_model._run_component_pipeline_transpile` and
the legacy single-graph path when `--npu` is passed on the CLI.

Two entry points:
- `run_prefill_pipeline`: emits the text-decoder prefill `.mlpackage`.
- `run_encoder_pipeline`: emits audio / vision encoder `.mlpackage`s by
  reusing the same component adapter modules the graph transpiler captures.

Historically (`origin/main`) NPU covered text prefill (Gemma/Qwen),
audio encoders (Whisper/Moonshine/Parakeet/Gemma4) and vision encoders
(SigLIP2/Gemma4). This pipeline regenerates those `.mlpackage`s through
the v2 transpiler instead of shipping them pre-built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .audio import emit_audio_encoder_mlpackage
from .prefill import emit_prefill_mlpackage
from .vision import emit_vision_encoder_mlpackage


DEFAULT_CHUNK_SIZE = 256


def _extract_dims(hf_model: torch.nn.Module) -> tuple[int, int]:
    """Walk through possible config locations to find text decoder dims.

    Multimodal HF models (Gemma 4, LFM-VL, etc.) nest text dims under
    `config.text_config`. Plain causal LMs expose them at the top level.
    """
    candidates = []
    cfg = getattr(hf_model, "config", None)
    if cfg is not None:
        candidates.append(cfg)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            candidates.insert(0, text_cfg)
    inner_cfg = getattr(getattr(hf_model, "model", None), "config", None)
    if inner_cfg is not None and inner_cfg not in candidates:
        candidates.append(inner_cfg)

    for c in candidates:
        hidden_dim = int(getattr(c, "hidden_size", 0) or 0)
        num_layers = int(getattr(c, "num_hidden_layers", 0) or 0)
        if hidden_dim > 0 and num_layers > 0:
            return hidden_dim, num_layers
    return 0, 0


def run_prefill_pipeline(
    hf_model: torch.nn.Module,
    artifact_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    enabled: bool = True,
    quantize_bits: int | None = None,
) -> str | None:
    """Run the NPU prefill transpilation. Returns the manifest-relative
    filename to embed in `manifest.json["npu_prefill"]`, or None.
    """
    if not enabled:
        return None

    hidden_dim, num_layers = _extract_dims(hf_model)
    if hidden_dim <= 0 or num_layers <= 0:
        print("npu.pipeline: model config missing hidden_size/num_hidden_layers; skipping")
        return None

    bundle_root = artifact_dir / "components"
    bundle_root.mkdir(parents=True, exist_ok=True)

    qbits: int | None = quantize_bits
    if qbits == 0:
        qbits = None

    filename = emit_prefill_mlpackage(
        hf_model,
        bundle_root,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        chunk_size=chunk_size,
        quantize_bits=qbits,
    )
    return f"components/{filename}" if filename else None


# Component name -> (emit function, output mlpackage filename).
_ENCODER_COMPONENTS = {
    "audio_encoder": (emit_audio_encoder_mlpackage, "audio_encoder.mlpackage"),
    "vision_encoder": (emit_vision_encoder_mlpackage, "vision_encoder.mlpackage"),
}


def run_encoder_pipeline(
    component_specs,
    artifact_dir: Path,
    *,
    enabled: bool = True,
    quantize_bits: int | None = None,
) -> dict[str, str]:
    """Emit audio / vision encoder `.mlpackage`s from captured component specs.

    Reuses the exact adapter modules + example inputs the graph transpiler
    captured (``ComponentModuleSpec``), so the NPU encoder stays in lockstep
    with the CPU component graph. Auxiliary adapter inputs (masks, position
    ids) are baked into the exported model as constants.

    Returns a dict mapping ``manifest.json`` key (``npu_audio_encoder`` /
    ``npu_vision_encoder``) to the manifest-relative `.mlpackage` path.
    """
    results: dict[str, str] = {}
    if not enabled or not component_specs:
        return results

    bundle_root = artifact_dir / "components"
    bundle_root.mkdir(parents=True, exist_ok=True)

    qbits: int | None = quantize_bits
    if qbits == 0:
        qbits = None

    for spec in component_specs:
        component = getattr(spec, "component", None)
        if component not in _ENCODER_COMPONENTS:
            continue
        emit_fn, filename = _ENCODER_COMPONENTS[component]
        example_inputs = tuple(getattr(spec, "example_inputs", ()) or ())
        if not example_inputs:
            print(f"npu.pipeline: {component} spec has no example inputs; skipping")
            continue
        primary = example_inputs[0]
        baked = example_inputs[1:]
        emitted = emit_fn(
            spec.module,
            bundle_root,
            example_input=primary,
            baked_inputs=baked,
            filename=filename,
            quantize_bits=qbits,
        )
        if emitted:
            results[f"npu_{component}"] = f"components/{emitted}"
    return results
