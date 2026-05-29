"""NPU transpiler orchestrator.

Emits CoreML `.mlpackage`s for **audio + vision encoders** from the same
``ComponentModuleSpec`` adapter modules the graph transpiler captures.
Invoked from ``hf_model._run_component_pipeline_transpile`` when ``--npu``
is passed on the CLI.

Text-decoder prefill is intentionally not on NPU — CPU prefill is the
supported path. (Historically ``origin/main`` shipped pre-built NPU prefill
mlpackages for Gemma/Qwen; v2 does not regenerate them.)
"""
from __future__ import annotations

from pathlib import Path

from .audio import emit_audio_encoder_mlpackage
from .vision import emit_vision_encoder_mlpackage


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
    quantize_bits: int | None = None,          # legacy override: forces both
    audio_quantize_bits: int | None = None,    # per-component, defaults to int8
    vision_quantize_bits: int | None = None,   # per-component, defaults to fp16
) -> dict[str, str]:
    """Emit audio / vision encoder `.mlpackage`s from captured component specs.

    Per-component quantization defaults — set independently because the two
    encoder families have very different sensitivity to weight quant:

    - **Audio** (Conformer-style: Parakeet, Gemma 4 audio): wide hidden
      dims, conv subsampling that averages noise across many input frames,
      long sequences. int8 holds up cleanly (verified cos_sim 0.9963 vs HF
      on Gemma 4); even int4 generally works.
    - **Vision** (ViT-style: LFM-VL SigLIP2, Gemma 4 vision): narrow
      attention heads, one patch-embedding conv is a single point of
      failure, short sequence (256-1024 tokens). int4 visibly degrades
      outputs; int8 is fine but fp16 is the rock-solid default since the
      vision tower is small (one-shot per request, bandwidth not critical).

    Quantization knob is per-component (``audio_quantize_bits`` /
    ``vision_quantize_bits``). The legacy ``quantize_bits`` arg, when set,
    overrides BOTH per-component defaults — preserves prior behavior where
    one ``--npu-quantize 4`` flag controlled the whole conversion.

    Defaults: audio → int8, vision → fp16. ``0`` means fp16 (no quant).

    Returns a dict mapping ``manifest.json`` key (``npu_audio_encoder`` /
    ``npu_vision_encoder``) to the manifest-relative `.mlpackage` path.
    """
    results: dict[str, str] = {}
    if not enabled or not component_specs:
        return results

    bundle_root = artifact_dir / "components"
    bundle_root.mkdir(parents=True, exist_ok=True)

    # Resolve final quant per component. Per-component arg wins if given;
    # else the legacy `quantize_bits` override; else per-component defaults.
    def _resolve(per_component: int | None, default: int | None) -> int | None:
        chosen = per_component if per_component is not None else (
            quantize_bits if quantize_bits is not None else default
        )
        return None if chosen == 0 else chosen

    component_quants = {
        "audio_encoder":  _resolve(audio_quantize_bits,  default=8),
        "vision_encoder": _resolve(vision_quantize_bits, default=None),  # default fp16
    }

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
        qbits = component_quants[component]
        qdesc = f"int{qbits}" if qbits else "fp16"
        print(f"npu.pipeline: emitting {component} quant={qdesc}")
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
