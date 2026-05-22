"""CoreML vision encoder emit.

Wraps a vision encoder PyTorch module (SigLIP2 / Gemma 4 vision tower)
into an ANE-targeted `.mlpackage`. The runtime side
(`cactus-engine/src/npu_ane.mm`) loads this with a single named input "x"
and a single named output, matching the `NPUEncoder::encode` interface.

Mirrors `audio.py`. Historically (`origin/main`) the NPU vision targets
were SigLIP2 and the Gemma 4 vision tower; LFM-VL was never NPU-accelerated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class VisionEncoderWrapper(torch.nn.Module):
    """Single-input single-output wrapper around a vision encoder module.

    Matches the runtime interface in `ANEEncoder::encode(input, output, ...)`:
    one fp16 input tensor, one fp16 output tensor.

    Auxiliary inputs that a vision tower needs (position ids, spatial
    shapes, attention masks) are supplied via ``baked_inputs`` and frozen
    as constant buffers so the exported model keeps a static single-input
    signature, as the ANE requires.
    """

    def __init__(
        self,
        vision_module: torch.nn.Module,
        baked_inputs: tuple[torch.Tensor, ...] = (),
    ):
        super().__init__()
        self.vision = vision_module
        self._n_baked = len(baked_inputs)
        for idx, tensor in enumerate(baked_inputs):
            self.register_buffer(f"_baked_{idx}", tensor, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        extra = tuple(getattr(self, f"_baked_{i}") for i in range(self._n_baked))
        return self.vision(pixel_values, *extra)


def _import_coremltools() -> Any:
    try:
        import coremltools as ct
        from .coremltools_patches import apply_all_coremltools_patches
        apply_all_coremltools_patches()
        return ct
    except Exception:
        return None


def _apply_weight_quantization(mlmodel: Any, bits: int) -> Any:
    try:
        from coremltools.optimize.coreml import (
            linear_quantize_weights,
            OpLinearQuantizerConfig,
            OptimizationConfig,
        )
        op_config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=f"int{bits}",
            granularity="per_channel",
        )
        config = OptimizationConfig(global_config=op_config)
        return linear_quantize_weights(mlmodel, config)
    except Exception as exc:
        print(f"npu.vision: weight quantization to int{bits} failed ({type(exc).__name__}: {exc}); keeping FP16 weights")
        return mlmodel


def emit_vision_encoder_mlpackage(
    vision_module: torch.nn.Module,
    bundle_dir: Path,
    *,
    example_input: torch.Tensor,
    baked_inputs: tuple[torch.Tensor, ...] = (),
    filename: str = "vision_encoder.mlpackage",
    input_name: str = "x",
    output_name: str = "encoded",
    minimum_deployment_target: str = "iOS18",
    quantize_bits: int | None = None,
) -> str | None:
    """Trace + convert + save. Returns the filename or None on failure.

    `example_input` should be the shape the runtime will use at inference
    time (e.g. ``[1, num_patches, patch_dim]``). The ANE requires a
    fully-specified shape; dynamic dims are not supported here.
    `baked_inputs` carries any auxiliary encoder inputs (position ids,
    masks) that get frozen into the model as constants.
    """
    ct = _import_coremltools()
    if ct is None:
        print("npu.vision: coremltools not installed; skipping mlpackage emit")
        return None

    wrapper = VisionEncoderWrapper(vision_module, baked_inputs)
    wrapper.eval()

    try:
        with torch.no_grad():
            exported = torch.export.export(wrapper, (example_input,))
            exported = exported.run_decompositions({})
    except Exception as exc:
        print(f"npu.vision: torch.export failed ({type(exc).__name__}: {exc}); skipping mlpackage emit")
        return None

    # Free the live module — ExportedProgram now owns what it needs.
    del wrapper
    import gc as _gc
    _gc.collect()

    target_attr = getattr(ct.target, minimum_deployment_target, None) or ct.target.iOS17

    from .coremltools_patches import build_cactus_pass_pipeline
    try:
        mlmodel = ct.convert(
            exported,
            inputs=[ct.TensorType(name=input_name, shape=tuple(example_input.shape))],
            outputs=[ct.TensorType(name=output_name)],
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=target_attr,
            pass_pipeline=build_cactus_pass_pipeline(),
        )
    except Exception as exc:
        print(f"npu.vision: coremltools.convert failed ({type(exc).__name__}: {exc})")
        return None

    if quantize_bits is not None:
        before_id = id(mlmodel)
        mlmodel = _apply_weight_quantization(mlmodel, quantize_bits)
        if id(mlmodel) != before_id:
            print(f"npu.vision: applied int{quantize_bits} weight quantization")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    out_path = bundle_dir / filename
    try:
        mlmodel.save(str(out_path))
    except Exception as exc:
        print(f"npu.vision: mlpackage save failed ({type(exc).__name__}: {exc})")
        return None

    print(f"npu.vision: wrote {out_path} (input_shape={tuple(example_input.shape)})")
    return filename
