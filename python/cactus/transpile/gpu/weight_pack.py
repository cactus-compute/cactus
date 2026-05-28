"""Pack HF weights into the GPU-ready binary layout the Metal kernels consume.

Layouts:

- **INT4 (q4_0-style symmetric, group=128)**:
    qs:     uint16[K/4 × N]   (4 nibbles per uint16, row-major over [k4, n])
    scales: fp16  [K/128 × N] (per-group scale, row-major over [group, n])
  Dequant: ``weight[k, n] = scale[k/128, n] * (nibble[k, n] - 8)``
  Per-group scale: ``scale = max(|w_in_group|) / 7`` (so the largest
  magnitude weight in the group lands at nibble 15 → (15-8) = 7).

  NB: this is simpler than cactus's CPU CQ4 (learned codebook + hadamard
  rotation). We picked symmetric q4_0 for the GPU bundle because (1) the
  matmul kernel is ~50 lines, (2) it's the same format llama.cpp ships,
  and (3) accuracy is acceptable for decode at int4. CPU keeps CQ4.

- **FP16** (norms, LM head, embedding): raw little-endian fp16 bytes,
  row-major.

Mutates the GPUPlan's ops in place to add ``weight_offset_bytes``,
``weight_nbytes``, ``scale_offset_bytes``, ``scale_nbytes``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .plan import GPUPlan


_WEIGHT_OPS_INT4 = {"mul_mv_int4_fp16", "mul_mm_int4_fp16"}
_WEIGHT_OPS_FP16 = {"mul_mv_fp16", "mul_mm_fp16"}


# ----------------------------------------------------------------------------
# q4_0 symmetric quantizer
# ----------------------------------------------------------------------------

def _quantize_q4_0_one_column(w_col: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a single output-column slice (length K) into q4_0.

    Returns:
        nibbles: uint8[K], values in [0, 15].
        scales:  fp16[K/group_size].
    """
    K = w_col.shape[0]
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    n_groups = K // group_size
    w_grouped = w_col.reshape(n_groups, group_size).astype(np.float32, copy=False)
    abs_max = np.abs(w_grouped).max(axis=1)
    scales = np.where(abs_max > 0, abs_max / 7.0, 1.0).astype(np.float32)
    nibbles = np.clip(
        np.round(w_grouped / scales[:, None]).astype(np.int32) + 8,
        0, 15,
    ).astype(np.uint8)
    return nibbles.reshape(K), scales.astype(np.float16)


def _pack_q4_0_layer(w_NK: np.ndarray, group_size: int = 128) -> tuple[bytes, bytes]:
    """Pack a [N, K] HF Linear weight into the kernel's qs + scales bytes.

    HF stores Linear.weight as [out_features, in_features] = [N, K].
    Kernel layout:
        qs:     uint16[K/4 × N]  (k4-major, then N cols)
        scales: fp16  [K/128 × N]
    """
    N, K = w_NK.shape
    assert K % group_size == 0, f"K={K} not divisible by {group_size}"
    assert K % 4 == 0, f"K={K} not divisible by 4"
    n_groups = K // group_size
    k4 = K // 4

    qs     = np.zeros((k4, N), dtype=np.uint16)
    scales = np.zeros((n_groups, N), dtype=np.float16)

    for n in range(N):
        nibbles, sc = _quantize_q4_0_one_column(w_NK[n], group_size=group_size)
        scales[:, n] = sc
        n4 = nibbles.reshape(k4, 4).astype(np.uint16)
        qs[:, n] = (n4[:, 0] & 0xF) | ((n4[:, 1] & 0xF) << 4) | ((n4[:, 2] & 0xF) << 8) | ((n4[:, 3] & 0xF) << 12)

    return qs.tobytes(), scales.tobytes()


# ----------------------------------------------------------------------------
# Module → weight tensor extraction
# ----------------------------------------------------------------------------

_LAYER_LINEAR_TAGS = {
    "q_proj":   "self_attn.q_proj",
    "k_proj":   "self_attn.k_proj",
    "v_proj":   "self_attn.v_proj",
    "out_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj":   "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}

_DECODER_LAYER_LIST_CANDIDATES = (
    "model.language_model.layers",
    "model.layers",
    "language_model.layers",
    "model.model.layers",
    "transformer.h",
)
_ATTN_NORM_CANDIDATES = ("input_layernorm", "pre_attn_norm", "attn_norm")
_MLP_NORM_CANDIDATES  = ("post_attention_layernorm", "post_attn_norm", "pre_mlp_norm", "mlp_norm")
_FINAL_NORM_CANDIDATES = (
    "model.language_model.norm", "model.norm",
    "language_model.norm", "model.model.norm",
    "transformer.ln_f",
)
_LM_HEAD_CANDIDATES = ("lm_head", "language_model.lm_head", "model.lm_head")
_EMBED_CANDIDATES = (
    "model.language_model.embed_tokens", "model.embed_tokens",
    "language_model.embed_tokens", "model.model.embed_tokens",
    "transformer.wte",
)


def _resolve_attr(root: Any, dotted: str) -> Optional[Any]:
    obj = root
    for part in dotted.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _first_resolvable(root: Any, candidates: tuple[str, ...]) -> Any:
    for c in candidates:
        v = _resolve_attr(root, c)
        if v is not None:
            return v
    return None


def _layers_list(model: torch.nn.Module):
    layers = _first_resolvable(model, _DECODER_LAYER_LIST_CANDIDATES)
    if layers is None:
        raise RuntimeError(
            f"gpu.weight_pack: could not locate decoder layers list (tried "
            f"{_DECODER_LAYER_LIST_CANDIDATES}) on {type(model).__name__}"
        )
    return layers


def _layer_linear(layer_module: torch.nn.Module, tag: str) -> torch.nn.Module:
    path = _LAYER_LINEAR_TAGS.get(tag)
    if path is None:
        raise KeyError(f"gpu.weight_pack: no Linear path for tag {tag}")
    mod = _resolve_attr(layer_module, path)
    if mod is None:
        raise RuntimeError(f"gpu.weight_pack: {type(layer_module).__name__} has no {path}")
    return mod


def _layer_norm(layer_module: torch.nn.Module, tag: str) -> torch.nn.Module:
    candidates = _ATTN_NORM_CANDIDATES if tag == "attn_norm" else _MLP_NORM_CANDIDATES
    norm = _first_resolvable(layer_module, candidates)
    if norm is None:
        raise RuntimeError(
            f"gpu.weight_pack: {type(layer_module).__name__} missing {tag} "
            f"(tried {candidates})"
        )
    return norm


# ----------------------------------------------------------------------------
# Main entry: pack the decoder weights
# ----------------------------------------------------------------------------

def pack_decoder_weights(
    hf_model: torch.nn.Module,
    plan: GPUPlan,
    *,
    weights_path: Path,
    scales_path: Path,
    embedding_path: Path,
    quantize_bits: int | None,
) -> None:
    """Lay out the GPU weights bundle on disk + populate plan offsets.

    For every weight-bearing op in the plan: resolve the matching HF tensor,
    quantize (q4_0) or cast (fp16), append to the right buffer, record the
    byte offset back in the plan op.
    """
    use_int4 = quantize_bits == 4
    layers = _layers_list(hf_model)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_file = open(weights_path, "wb")
    scales_file  = open(scales_path,  "wb")

    weight_off = 0
    scale_off  = 0

    def _emit_fp16_tensor(t: torch.Tensor) -> tuple[int, int]:
        nonlocal weight_off
        arr = t.detach().to(torch.float16).contiguous().cpu().numpy()
        bytes_data = arr.tobytes()
        start = weight_off
        weights_file.write(bytes_data)
        weight_off += len(bytes_data)
        return start, len(bytes_data)

    def _emit_int4_linear(linear: torch.nn.Module) -> tuple[int, int, int, int]:
        nonlocal weight_off, scale_off
        w_NK = linear.weight.detach().to(torch.float32).contiguous().cpu().numpy()
        qs_bytes, scales_bytes = _pack_q4_0_layer(w_NK)
        w_start, s_start = weight_off, scale_off
        weights_file.write(qs_bytes);   weight_off += len(qs_bytes)
        scales_file.write(scales_bytes); scale_off  += len(scales_bytes)
        return w_start, len(qs_bytes), s_start, len(scales_bytes)

    def _emit_fp16_linear(linear: torch.nn.Module) -> tuple[int, int]:
        return _emit_fp16_tensor(linear.weight)

    missing_tags: dict[str, int] = {}

    def _try_layer_linear(layer_module, tag):
        """Resolve a Linear, returning None if absent (e.g. Gemma 4's
        sliding_attention layers omit k_proj / v_proj — they share KV
        from a prior full_attention layer. Per-architecture KV-sharing
        is a M3+ enhancement; for now we skip and the plan stays valid
        but execution will need the runtime to also know to skip)."""
        path = _LAYER_LINEAR_TAGS.get(tag)
        if path is None:
            return None
        mod = _resolve_attr(layer_module, path)
        return mod

    for layer_idx, layer in enumerate(plan.layers):
        layer_module = layers[layer_idx]
        for op in layer.ops:
            kind = op.kind
            args = op.args
            if kind == "rms_norm":
                norm = _layer_norm(layer_module, args["tag"])
                off, sz = _emit_fp16_tensor(norm.weight)
                args["weight_offset_bytes"] = off
                args["weight_nbytes"]       = sz
            elif kind in _WEIGHT_OPS_INT4 or kind in _WEIGHT_OPS_FP16:
                linear = _try_layer_linear(layer_module, args["tag"])
                if linear is None:
                    # Tag not present on this layer; record None offsets
                    # so the runtime can detect + skip / share KV.
                    args["weight_offset_bytes"] = None
                    args["weight_nbytes"]       = 0
                    if kind in _WEIGHT_OPS_INT4:
                        args["scale_offset_bytes"] = None
                        args["scale_nbytes"]       = 0
                    missing_tags[args["tag"]] = missing_tags.get(args["tag"], 0) + 1
                    continue
                if kind in _WEIGHT_OPS_INT4:
                    w_off, w_sz, s_off_, s_sz = _emit_int4_linear(linear)
                    args["weight_offset_bytes"] = w_off
                    args["weight_nbytes"]       = w_sz
                    args["scale_offset_bytes"]  = s_off_
                    args["scale_nbytes"]        = s_sz
                else:
                    off, sz = _emit_fp16_linear(linear)
                    args["weight_offset_bytes"] = off
                    args["weight_nbytes"]       = sz
            # other ops (rope, swiglu, residual, kv_cache_append, flash_attn)
            # have no weights — skip.

    if missing_tags:
        print("gpu.weight_pack: skipped tags not present on every layer:", missing_tags,
              "(typically KV-shared sliding-attention layers — runtime must handle)")

    # Final norm
    if plan.final_norm:
        final_norm = _first_resolvable(hf_model, _FINAL_NORM_CANDIDATES)
        if final_norm is not None:
            off, sz = _emit_fp16_tensor(final_norm.weight)
            plan.final_norm["weight_offset_bytes"] = off
            plan.final_norm["weight_nbytes"]       = sz

    # LM head (fp16 — see plan.lm_head.kind)
    if plan.lm_head:
        lm_head = _first_resolvable(hf_model, _LM_HEAD_CANDIDATES)
        if lm_head is not None and hasattr(lm_head, "weight"):
            off, sz = _emit_fp16_tensor(lm_head.weight)
        else:
            embed = _first_resolvable(hf_model, _EMBED_CANDIDATES)
            if embed is None:
                raise RuntimeError("gpu.weight_pack: no lm_head and no embed_tokens — can't size LM head")
            off, sz = _emit_fp16_tensor(embed.weight)
        plan.lm_head["weight_offset_bytes"] = off
        plan.lm_head["weight_nbytes"]       = sz

    weights_file.close()
    scales_file.close()
    if scale_off == 0:
        # No int4 ops emitted anything — drop the empty scales file.
        try:
            scales_path.unlink()
        except FileNotFoundError:
            pass

    # Embedding lookup table — separate file so the embedding kernel can bind
    # it directly without offsetting into the big weights blob.
    embed = _first_resolvable(hf_model, _EMBED_CANDIDATES)
    if embed is not None and hasattr(embed, "weight"):
        embed_arr = embed.weight.detach().to(torch.float16).contiguous().cpu().numpy()
        with open(embedding_path, "wb") as f:
            f.write(embed_arr.tobytes())
    else:
        try:
            embedding_path.unlink()
        except FileNotFoundError:
            pass
