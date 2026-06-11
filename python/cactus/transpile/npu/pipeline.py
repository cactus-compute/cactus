from __future__ import annotations

from pathlib import Path

from .audio import emit_audio_encoder_mlpackage
from .source import emit_source_encoder_mlpackage
from .vision import emit_vision_encoder_mlpackage


_ENCODER_COMPONENTS = {
    "audio_encoder": (emit_audio_encoder_mlpackage, "audio_encoder.mlpackage"),
    "source_encoder": (emit_source_encoder_mlpackage, "source_encoder.mlpackage"),
    "vision_encoder": (emit_vision_encoder_mlpackage, "vision_encoder.mlpackage"),
}


def run_encoder_pipeline(
    component_specs,
    artifact_dir: Path,
    *,
    enabled: bool = True,
    quantize_bits: int | None = None,
    audio_quantize_bits: int | None = None,
    vision_quantize_bits: int | None = None,
) -> dict[str, str]:
    results: dict[str, str] = {}
    if not enabled or not component_specs:
        return results

    bundle_root = artifact_dir / "components"
    bundle_root.mkdir(parents=True, exist_ok=True)

    def _resolve(per_component: int | None, default: int | None) -> int | None:
        chosen = per_component if per_component is not None else (
            quantize_bits if quantize_bits is not None else default
        )
        return None if chosen == 0 else chosen

    component_quants = {
        "audio_encoder":  _resolve(audio_quantize_bits,  default=8),
        "source_encoder": _resolve(None, default=None),
        "vision_encoder": _resolve(vision_quantize_bits, default=None),
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
        module = spec.npu_module or spec.module
        n_runtime = min(max(1, spec.npu_runtime_input_count), len(example_inputs))
        qbits = component_quants[component]
        qdesc = f"int{qbits}" if qbits else "fp16"
        print(f"npu.pipeline: emitting {component} quant={qdesc}")
        if component == "source_encoder":
            emitted = emit_fn(
                module,
                bundle_root,
                example_inputs=example_inputs,
                input_names=tuple(getattr(spec, "input_keys", ()) or ()),
                filename=filename,
                quantize_bits=qbits,
            )
        elif component == "vision_encoder":
            names = ("x",) + tuple(spec.input_keys[1:n_runtime])
            emitted = emit_fn(
                module,
                bundle_root,
                runtime_inputs=tuple(zip(names, example_inputs[:n_runtime])),
                baked_inputs=example_inputs[n_runtime:],
                filename=filename,
                quantize_bits=qbits,
            )
        else:
            emitted = emit_fn(
                module,
                bundle_root,
                example_input=example_inputs[0],
                baked_inputs=example_inputs[1:],
                filename=filename,
                quantize_bits=qbits,
            )
        if emitted:
            results[f"npu_{component}"] = f"components/{emitted}"
    return results
