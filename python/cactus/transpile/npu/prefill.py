"""CoreML prefill emit.

Wraps an HF causal LM so its forward pass exposes per-layer K/V outputs,
then runs the wrapper through `coremltools.convert` to produce a
`.mlpackage`. Output naming matches what `cactus-engine/src/npu_ane.mm`
reads at load time: `hidden`, `k_0`, `v_0`, ... (the runtime infers
`num_layers` by scanning for outputs prefixed with `k_`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _find_text_decoder(hf_model: torch.nn.Module) -> torch.nn.Module:
    """Locate the inner text decoder.

    Handles three layouts:
    - Plain `AutoModelForCausalLM`: returns `model.model` (e.g., LlamaModel).
    - Multimodal HF model: `model.model.language_model` (e.g., Gemma4Model.language_model).
    - Already-text model: returns the model itself.
    """
    inner = getattr(hf_model, "model", None) or hf_model
    nested = getattr(inner, "language_model", None)
    if nested is not None and isinstance(nested, torch.nn.Module):
        return nested
    return inner


class PrefillWrapper(torch.nn.Module):
    """Wraps an HF causal LM into a (input_ids, position_ids) -> (hidden, k_0, v_0, ...) module.

    Input is token IDs (not embeddings) because some architectures (Gemma 4)
    require the original input_ids to compute auxiliary per-layer embeddings
    that get mixed into each transformer layer. Passing input_ids lets the
    model handle embedding internally and is also more efficient.
    """

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.inner = _find_text_decoder(hf_model)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.inner(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        flat: list[torch.Tensor] = [hidden]
        for layer_kv in outputs.past_key_values:
            flat.append(layer_kv[0])
            flat.append(layer_kv[1])
        return tuple(flat)


def _estimate_save_bytes(hidden_dim: int, num_layers: int, chunk_size: int, quantize_bits: int | None) -> int:
    """Rough size of the on-disk .mlpackage weight blob.

    Weight bytes dominate; ignore proto/activation overhead. Per parameter:
    FP16 = 2 bytes, int8 = 1 byte, int4 = 0.5 bytes (palettized).
    """
    bytes_per_param = 2.0 if quantize_bits is None else (quantize_bits / 8.0)
    approx_params = num_layers * (12 * hidden_dim * hidden_dim) + 256_000 * hidden_dim
    return int(approx_params * bytes_per_param)


def output_names(num_layers: int) -> list[str]:
    names = ["hidden"]
    for i in range(num_layers):
        names.append(f"k_{i}")
        names.append(f"v_{i}")
    return names


def _import_coremltools() -> Any:
    try:
        import coremltools as ct
        from .coremltools_patches import apply_all_coremltools_patches
        apply_all_coremltools_patches()
        return ct
    except Exception:
        return None


def _apply_weight_quantization(mlmodel: Any, ct: Any, bits: int) -> Any:
    """Compress weights to bits-bit linear quantization.

    Big bandwidth win on ANE — smaller weights = faster matmuls. Per-channel
    int4/int8 mappings come straight from coremltools' optimize pipeline.
    No-op if anything in the chain fails (returns the original model).
    """
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
        print(f"npu.prefill: weight quantization to int{bits} failed ({type(exc).__name__}: {exc}); keeping FP16 weights")
        return mlmodel


def emit_prefill_mlpackage(
    hf_model: torch.nn.Module,
    bundle_dir: Path,
    *,
    hidden_dim: int,
    num_layers: int,
    chunk_size: int,
    filename: str = "model.mlpackage",
    minimum_deployment_target: str = "iOS17",
    quantize_bits: int | None = None,
) -> str | None:
    """Trace + convert + save. Returns the manifest-relative filename or None on failure."""
    ct = _import_coremltools()
    if ct is None:
        print("npu.prefill: coremltools not installed; skipping mlpackage emit")
        return None

    wrapper = PrefillWrapper(hf_model)
    wrapper.eval()

    example_ids = torch.zeros(1, chunk_size, dtype=torch.long)
    example_positions = torch.arange(0, chunk_size, dtype=torch.long).unsqueeze(0)

    # Run a dry forward to count actual outputs (Gemma 4 has shared-KV layers,
    # so #past_key_values < num_hidden_layers).
    try:
        with torch.no_grad():
            dry_out = wrapper(example_ids, example_positions)
        actual_outputs = len(dry_out)
        actual_kv_layers = max(0, (actual_outputs - 1) // 2)
        if actual_kv_layers != num_layers:
            print(f"npu.prefill: model reports {actual_kv_layers} K/V layers (config says {num_layers}); using actual.")
        emit_layers = actual_kv_layers
    except Exception as exc:
        print(f"npu.prefill: dry-forward failed ({type(exc).__name__}: {exc}); falling back to config layer count")
        emit_layers = num_layers

    try:
        with torch.no_grad():
            exported = torch.export.export(wrapper, (example_ids, example_positions))
            exported = exported.run_decompositions({})
    except Exception as exc:
        print(f"npu.prefill: torch.export failed ({type(exc).__name__}: {exc}); skipping mlpackage emit")
        return None

    # Drop references to the live HF model — torch.export.ExportedProgram now
    # owns the weights it needs. Frees ~4 GB on 2B param FP16 models so the
    # coremltools MIL pipeline has headroom on 16 GB machines.
    del wrapper
    import gc as _gc
    _gc.collect()

    target_attr = getattr(ct.target, minimum_deployment_target, None) or ct.target.iOS17

    from .coremltools_patches import build_cactus_pass_pipeline
    try:
        import numpy as np
        mlmodel = ct.convert(
            exported,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, chunk_size), dtype=np.int32),
                ct.TensorType(name="position_ids", shape=(1, chunk_size), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name=name) for name in output_names(emit_layers)],
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=target_attr,
            pass_pipeline=build_cactus_pass_pipeline(),
        )
    except Exception as exc:
        print(f"npu.prefill: coremltools.convert failed ({type(exc).__name__}: {exc})")
        print("npu.prefill: see python/cactus/transpile/npu/README.md for known coremltools op-coverage gaps")
        return None

    if quantize_bits is not None:
        mlmodel = _apply_weight_quantization(mlmodel, ct, quantize_bits)
        print(f"npu.prefill: applied int{quantize_bits} weight quantization")

    out_path = bundle_dir / filename
    try:
        import shutil
        free_bytes = shutil.disk_usage(str(bundle_dir)).free
        approx_save_bytes = _estimate_save_bytes(hidden_dim, num_layers, chunk_size, quantize_bits)
        if free_bytes < approx_save_bytes * 2:
            print(
                f"npu.prefill: WARNING free disk {free_bytes / 1e9:.1f} GB may be tight "
                f"for save (~{approx_save_bytes / 1e9:.1f} GB needed, x2 during copy)."
            )
    except Exception:
        pass
    try:
        mlmodel.save(str(out_path))
    except Exception as exc:
        print(f"npu.prefill: mlpackage save failed ({type(exc).__name__}: {exc})")
        return None

    print(f"npu.prefill: wrote {out_path} (chunk={chunk_size}, hidden={hidden_dim}, layers={num_layers})")
    return filename
