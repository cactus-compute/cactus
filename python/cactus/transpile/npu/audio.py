from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class AudioEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        audio_module: torch.nn.Module,
        baked_inputs: tuple[torch.Tensor, ...] = (),
    ):
        super().__init__()
        self.audio = audio_module
        self._n_baked = len(baked_inputs)
        for idx, tensor in enumerate(baked_inputs):
            self.register_buffer(f"_baked_{idx}", tensor, persistent=False)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        extra = tuple(getattr(self, f"_baked_{i}") for i in range(self._n_baked))
        return self.audio(input_features, *extra)


def _import_coremltools() -> Any:
    import coremltools as ct
    from .coremltools_patches import apply_all_coremltools_patches
    apply_all_coremltools_patches()
    return ct


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
        print(f"npu.audio: weight quantization to int{bits} failed ({type(exc).__name__}: {exc}); keeping FP16 weights")
        return mlmodel


def emit_audio_encoder_mlpackage(
    audio_module: torch.nn.Module,
    bundle_dir: Path,
    *,
    example_input: torch.Tensor,
    baked_inputs: tuple[torch.Tensor, ...] = (),
    filename: str = "audio_encoder.mlpackage",
    input_name: str = "x",
    output_name: str = "encoded",
    minimum_deployment_target: str = "iOS17",
    quantize_bits: int | None = None,
) -> str | None:
    ct = _import_coremltools()

    wrapper = AudioEncoderWrapper(audio_module, baked_inputs)
    wrapper.eval()

    try:
        with torch.no_grad():
            exported = torch.export.export(wrapper, (example_input,))
            exported = exported.run_decompositions({})
    except Exception as exc:
        print(f"npu.audio: torch.export failed ({type(exc).__name__}: {exc}); skipping mlpackage emit")
        return None

    del wrapper
    import gc as _gc
    _gc.collect()

    if quantize_bits == 4:
        minimum_deployment_target = "iOS18"
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
        print(f"npu.audio: coremltools.convert failed ({type(exc).__name__}: {exc})")
        return None

    if quantize_bits is not None:
        before_id = id(mlmodel)
        mlmodel = _apply_weight_quantization(mlmodel, quantize_bits)
        if id(mlmodel) != before_id:
            print(f"npu.audio: applied int{quantize_bits} weight quantization")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    out_path = bundle_dir / filename
    try:
        mlmodel.save(str(out_path))
    except Exception as exc:
        print(f"npu.audio: mlpackage save failed ({type(exc).__name__}: {exc})")
        return None

    print(f"npu.audio: wrote {out_path} (input_shape={tuple(example_input.shape)})")
    return filename
