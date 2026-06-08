from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audio import _apply_weight_quantization, _import_coremltools


class SourceEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        source_module: torch.nn.Module,
        input_dtypes: tuple[torch.dtype, ...],
    ):
        super().__init__()
        self.source = source_module
        self.input_dtypes = input_dtypes

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        typed_args = []
        for arg, dtype in zip(args, self.input_dtypes, strict=True):
            typed_args.append(arg.to(dtype=dtype) if arg.dtype != dtype else arg)
        out = self.source(*typed_args)
        if isinstance(out, tuple):
            return out[0]
        return out


def emit_source_encoder_mlpackage(
    source_module: torch.nn.Module,
    bundle_dir: Path,
    *,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    filename: str = "source_encoder.mlpackage",
    output_name: str = "encoder_hidden_states",
    minimum_deployment_target: str = "iOS18",
    quantize_bits: int | None = None,
) -> str | None:
    ct = _import_coremltools()
    if not example_inputs:
        print("npu.source: source_encoder spec has no example inputs; skipping")
        return None

    input_dtypes = tuple(t.dtype for t in example_inputs)
    exported_inputs = tuple(
        t.to(dtype=torch.int32) if t.dtype in {torch.int64, torch.long} else t
        for t in example_inputs
    )
    wrapper = SourceEncoderWrapper(source_module, input_dtypes)
    wrapper.eval()

    original_sdpa = F.scaled_dot_product_attention

    def _sdpa_without_gqa(query, key, value, *args, **kwargs):
        enable_gqa = bool(kwargs.pop("enable_gqa", False))
        if enable_gqa and query.dim() >= 3 and key.dim() >= 3 and query.shape[-3] != key.shape[-3]:
            repeats = int(query.shape[-3]) // max(int(key.shape[-3]), 1)
            key = key.repeat_interleave(repeats, dim=-3)
            value = value.repeat_interleave(repeats, dim=-3)
        return original_sdpa(query, key, value, *args, **kwargs)

    try:
        F.scaled_dot_product_attention = _sdpa_without_gqa
        with torch.no_grad():
            exported = torch.export.export(wrapper, exported_inputs)
            exported = exported.run_decompositions({})
    except Exception as exc:
        print(f"npu.source: torch.export failed ({type(exc).__name__}: {exc}); skipping mlpackage emit")
        return None
    finally:
        F.scaled_dot_product_attention = original_sdpa

    del wrapper
    import gc as _gc
    _gc.collect()

    target_attr = getattr(ct.target, minimum_deployment_target, None) or ct.target.iOS17
    input_types = []
    for name, tensor in zip(input_names, exported_inputs, strict=True):
        dtype = np.int32 if tensor.dtype in {torch.int32, torch.int64, torch.long} else None
        input_types.append(ct.TensorType(name=name, shape=tuple(tensor.shape), dtype=dtype))

    from .coremltools_patches import build_cactus_pass_pipeline
    try:
        mlmodel = ct.convert(
            exported,
            inputs=input_types,
            outputs=[ct.TensorType(name=output_name)],
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=target_attr,
            pass_pipeline=build_cactus_pass_pipeline(),
        )
    except Exception as exc:
        print(f"npu.source: coremltools.convert failed ({type(exc).__name__}: {exc})")
        return None

    if quantize_bits is not None:
        before_id = id(mlmodel)
        mlmodel = _apply_weight_quantization(mlmodel, quantize_bits)
        if id(mlmodel) != before_id:
            print(f"npu.source: applied int{quantize_bits} weight quantization")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    out_path = bundle_dir / filename
    try:
        mlmodel.save(str(out_path))
    except Exception as exc:
        print(f"npu.source: mlpackage save failed ({type(exc).__name__}: {exc})")
        return None

    shapes = ",".join(f"{name}={tuple(t.shape)}" for name, t in zip(input_names, exported_inputs, strict=True))
    print(f"npu.source: wrote {out_path} ({shapes})")
    return filename
