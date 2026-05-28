"""Pack HF weights into the GPU-ready binary layout described by the GPU plan.

This is the bridge between PyTorch's ``state_dict`` and the byte
layout the Metal kernels expect:

- For INT4 (CQ4 group=128) layers: pack 4 nibbles per uint16 into the
  ``qs`` buffer, and write per-group fp16 scales to the ``scales`` buffer.
  Layout: ``qs[K/4 × N]`` (row-major, k4 = k/4 is the slow dim, n is fast)
  and ``scales[K/128 × N]`` likewise.
- For FP16 layers (norms, embeddings, LM head): write raw fp16 bytes
  in row-major order.

The plan's per-op ``weight_offset_bytes`` / ``scale_offset_bytes`` are
populated *here* (mutated in place) as a side effect of the byte-level
layout decisions.

This M1 implementation is a SKELETON: it walks the plan and writes
zero-filled placeholders for every weight slot, but does not yet hook
into the cactus CQ4 quantization machinery to actually quantize HF
weights. That hookup is M2 — we need to:

1. Run the rotation+quantization (``cactus.convert.quantization.cq``)
   per linear layer.
2. Write the resulting nibbles + scales out in the layout above.

For M1, this skeleton lets us validate the runtime side independently of
the quantization side: a coworker can build the C++ engine, load a bundle
with zero weights, and see "outputs are zero (or NaN)" — proving the
dispatch path works end-to-end. M2 fills in real weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .plan import GPUPlan


_WEIGHT_OPS = {
    "rms_norm",                # gain weights, fp16, dim = axis_size
    "mul_mv_int4_fp16",        # int4 weights + fp16 scales
    "mul_mv_fp16",             # fp16 weights
    "mul_mm_int4_fp16",
    "mul_mm_fp16",
}


def _bytes_per_int4_weight_pack(K: int, N: int) -> int:
    """K/4 uint16s per N column = K/4 × N × 2 bytes."""
    assert K % 4 == 0, f"K={K} must be divisible by 4 for CQ4 packing"
    return (K // 4) * N * 2


def _bytes_per_int4_scales(K: int, N: int) -> int:
    """One fp16 scale per group of 128 weight values, per column."""
    assert K % 128 == 0, f"K={K} must be divisible by 128 for CQ4 group=128"
    return (K // 128) * N * 2


def _bytes_per_fp16_weight(K: int, N: int) -> int:
    return K * N * 2


def pack_decoder_weights(
    hf_model: torch.nn.Module,
    plan: GPUPlan,
    *,
    weights_path: Path,
    scales_path: Path,
    embedding_path: Path,
    quantize_bits: int | None,
) -> None:
    """Lay out the GPU weights bundle on disk.

    For M1: writes zero-filled placeholders of the correct size so the
    runtime side can size its MTLBuffers and exercise the dispatch path.
    The plan dict is mutated in place to add ``weight_offset_bytes`` and
    ``scale_offset_bytes`` per linear-layer op.

    For M2: this routine will also run cactus CQ4 quantization on each
    Linear weight and write real nibbles+scales.
    """
    weight_off = 0
    scale_off  = 0

    for layer in plan.layers:
        for op in layer.ops:
            if op.kind not in _WEIGHT_OPS:
                continue
            args = op.args
            if op.kind == "rms_norm":
                D = int(args["axis_size"])
                args["weight_offset_bytes"] = weight_off
                args["weight_nbytes"] = _bytes_per_fp16_weight(1, D)
                weight_off += args["weight_nbytes"]
            elif op.kind in ("mul_mv_int4_fp16", "mul_mm_int4_fp16"):
                K = int(args["K"]); N = int(args["N"])
                args["weight_offset_bytes"] = weight_off
                args["weight_nbytes"] = _bytes_per_int4_weight_pack(K, N)
                weight_off += args["weight_nbytes"]
                args["scale_offset_bytes"] = scale_off
                args["scale_nbytes"]  = _bytes_per_int4_scales(K, N)
                scale_off += args["scale_nbytes"]
            elif op.kind in ("mul_mv_fp16", "mul_mm_fp16"):
                K = int(args["K"]); N = int(args["N"])
                args["weight_offset_bytes"] = weight_off
                args["weight_nbytes"] = _bytes_per_fp16_weight(K, N)
                weight_off += args["weight_nbytes"]

    # final_norm
    if plan.final_norm:
        D = int(plan.final_norm["axis_size"])
        plan.final_norm["weight_offset_bytes"] = weight_off
        plan.final_norm["weight_nbytes"] = D * 2
        weight_off += D * 2

    # lm_head — fp16 (heavy; vocab_size × hidden_dim)
    if plan.lm_head:
        K = int(plan.lm_head["K"]); N = int(plan.lm_head["N"])
        plan.lm_head["weight_offset_bytes"] = weight_off
        plan.lm_head["weight_nbytes"] = _bytes_per_fp16_weight(K, N)
        weight_off += plan.lm_head["weight_nbytes"]

    # Allocate the two buffers as zero-filled placeholders.
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "wb") as f:
        f.truncate(weight_off)
    if scale_off > 0:
        with open(scales_path, "wb") as f:
            f.truncate(scale_off)

    # Embedding table — fp16, [vocab × hidden]. M1: zero placeholder.
    embedding_nbytes = plan.vocab_size * plan.hidden_dim * 2
    if embedding_nbytes > 0:
        with open(embedding_path, "wb") as f:
            f.truncate(embedding_nbytes)

    # NOTE: M2 will replace the truncate() calls with real bytes:
    #   - For each Linear: run cactus CQ4 quantization, write nibbles + scales.
    #   - For each RMSNorm: cast .weight to fp16, write contiguous.
    #   - For embedding: cast model.get_input_embeddings().weight to fp16.
    # The plan offsets above are already correct; only the file contents need
    # to be populated.
