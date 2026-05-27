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
