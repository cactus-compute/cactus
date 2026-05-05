"""Dequantize cactus v3 TQH weights → fp16 safetensors for HuggingFace.

Usage:
    python qdq_weights.py \
        --weights /path/to/gemma-4-e2b-it-tqh-prod-v3 \
        --out /tmp/qdq_weights \
        [--layers 0,1,2]   # optional: only these layer indices

Output: /tmp/qdq_weights/model.safetensors  (fp16, HF tensor names)

Only the quantized (CQ*) language-model weight matrices are written.
Non-CQ tensors (norms, scalars, etc.) are skipped.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

CACTUS_MAGIC = 0x54434143
HEADER_SIZE = 84
FLAG_ORTHOGONAL_ROTATION = 1 << 1

PRECISION_BITS = {3: 1, 4: 2, 5: 3, 6: 4}

# Cactus bakes these scale factors into the norms. To recover HF-scale weights,
# apply the inverse: 1/scale for MULT tensors, scale for DIV tensors.
_CACTUS_MULT_BASENAMES = {"ffn_gate", "ffn_up", "per_layer_gate", "moe_gate_proj", "moe_up_proj"}
_CACTUS_DIV_BASENAMES = {"token_embeddings", "output_weight", "embed_vision_proj", "embed_vision_embedding"}
_CACTUS_WEIGHT_SCALE = 16.0


def cactus_to_hf_scale(stem: str) -> float:
    """Inverse of gemma4_scale_factor: undo the scale baked into cactus norms."""
    base = stem
    parts = base.split("_", 2)
    if len(parts) == 3 and parts[0] == "layer" and parts[1].isdigit():
        base = parts[2]
    if base in _CACTUS_MULT_BASENAMES:
        return 1.0 / _CACTUS_WEIGHT_SCALE   # was ×16 in cactus → ÷16 for HF
    if base in _CACTUS_DIV_BASENAMES:
        return _CACTUS_WEIGHT_SCALE          # was ÷16 in cactus → ×16 for HF
    return 1.0

# cactus basename → HuggingFace suffix under model.language_model.layers.N.*
_LAYER_NAME_MAP = {
    "attn_q":       "self_attn.q_proj",
    "attn_k":       "self_attn.k_proj",
    "attn_v":       "self_attn.v_proj",
    "attn_output":  "self_attn.o_proj",
    "ffn_gate":     "mlp.gate_proj",
    "ffn_up":       "mlp.up_proj",
    "ffn_down":     "mlp.down_proj",
    "per_layer_gate": "per_layer_input_gate",
    "per_layer_proj": "per_layer_projection",
}

# Non-layer CQ files
_TOPLEVEL_NAME_MAP = {
    "token_embeddings":    "model.language_model.embed_tokens",
    "per_layer_model_proj": "model.language_model.per_layer_model_projection",
    "embed_tokens_per_layer": "model.language_model.embed_tokens_per_layer",
}


def hf_name(cactus_stem: str) -> Optional[str]:
    """Map cactus weight stem (no .weights) to HuggingFace tensor name."""
    if cactus_stem in _TOPLEVEL_NAME_MAP:
        return _TOPLEVEL_NAME_MAP[cactus_stem]
    # layer_N_basename
    parts = cactus_stem.split("_", 2)
    if len(parts) == 3 and parts[0] == "layer" and parts[1].isdigit():
        n, base = parts[1], parts[2]
        hf_suffix = _LAYER_NAME_MAP.get(base)
        if hf_suffix:
            return f"model.language_model.layers.{n}.{hf_suffix}"
    return None


def align_up(offset: int, alignment: int) -> int:
    r = offset % alignment
    return offset if r == 0 else offset + (alignment - r)


def read_header(data: bytes) -> dict:
    magic, flags, alignment, ndim = struct.unpack_from("<IIII", data, 0)
    if magic != CACTUS_MAGIC:
        raise ValueError(f"Bad magic {magic:#010x}")
    dims = struct.unpack_from("<4Q", data, 16)
    prec, data_bytes, scales_bytes, gs, ng, orig_n = struct.unpack_from("<IQQIIq", data, 48)
    shape = tuple(int(d) for d in dims[:ndim] if d > 0)
    aligned_hdr = align_up(HEADER_SIZE, alignment)
    scales_off = aligned_hdr if scales_bytes > 0 else 0
    data_off = align_up(scales_off + scales_bytes, alignment) if scales_bytes > 0 else aligned_hdr
    return {
        "flags": flags,
        "alignment": alignment,
        "shape": shape,
        "prec": prec,
        "data_bytes": data_bytes,
        "scales_bytes": scales_bytes,
        "gs": gs,
        "ng": ng,
        "orig_n": orig_n,
        "scales_off": scales_off,
        "data_off": data_off,
        "orthogonal": bool(flags & FLAG_ORTHOGONAL_ROTATION),
    }


def unpack_indices_batch(data: np.ndarray, bits: int, N: int, K: int) -> np.ndarray:
    """Unpack LSB-packed indices for N rows × K elements. Returns uint8 [N, K]."""
    if bits == 4:
        # two 4-bit indices per byte
        flat = data.reshape(N, -1)  # [N, K//2]
        lo = flat & 0x0F
        hi = (flat >> 4) & 0x0F
        return np.stack([lo, hi], axis=2).reshape(N, K)
    elif bits == 2:
        flat = data.reshape(N, -1)  # [N, K//4]
        b0 = flat & 0x03
        b1 = (flat >> 2) & 0x03
        b2 = (flat >> 4) & 0x03
        b3 = (flat >> 6) & 0x03
        return np.stack([b0, b1, b2, b3], axis=2).reshape(N, K)
    elif bits == 1:
        return np.unpackbits(data.reshape(N, -1), axis=1, bitorder='little')[:, :K]
    else:  # bits == 3 — slower, use per-element extraction
        out = np.zeros((N, K), dtype=np.uint8)
        row_bytes = K * 3 // 8
        for k in range(K):
            bit_pos = k * 3
            byte_idx = bit_pos // 8
            bit_shift = bit_pos % 8
            col_lo = data[:, byte_idx]
            col_hi = data[:, byte_idx + 1]
            word = col_lo.astype(np.uint16) | (col_hi.astype(np.uint16) << 8)
            out[:, k] = (word >> bit_shift) & 0x7
        return out


def fwht_batch(x: np.ndarray) -> np.ndarray:
    """Batched FWHT over last dimension. x shape: [..., n]. Normalized. In-place."""
    n = x.shape[-1]
    h = 1
    while h < n:
        x_view = x.reshape(-1, n // (h * 2), h * 2)
        a = x_view[..., :h].copy()
        b = x_view[..., h:].copy()
        x_view[..., :h] = a + b
        x_view[..., h:] = a - b
        h *= 2
    x *= 1.0 / math.sqrt(n)
    return x


def dequant_file(path: Path, hf_scale: float = 1.0) -> Optional[np.ndarray]:
    """Dequantize a single CQ .weights file. Returns fp16 [N, K] or None if not CQ."""
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        return None
    h = read_header(raw)
    if h["prec"] not in PRECISION_BITS:
        return None  # not CQ

    bits = PRECISION_BITS[h["prec"]]
    shape = h["shape"]
    if len(shape) != 2:
        return None
    N, K = shape
    gs = h["gs"]
    ng = h["ng"]
    cb_size = 1 << bits

    scales = raw[h["scales_off"]: h["scales_off"] + h["scales_bytes"]]
    scales_arr = np.frombuffer(scales, dtype=np.uint8)

    off = 0
    codebook = np.frombuffer(scales_arr[off: off + cb_size * 2].tobytes(), dtype=np.float16).astype(np.float32)
    off += cb_size * 2
    off += K * 2  # skip input_scale
    isr = np.frombuffer(scales_arr[off: off + K * 2].tobytes(), dtype=np.float16).astype(np.float32)
    off += K * 2

    norms = np.frombuffer(scales_arr[off: off + N * ng * 2].tobytes(), dtype=np.float16).astype(np.float32).reshape(N, ng)
    off += N * ng * 2

    data = np.frombuffer(raw[h["data_off"]: h["data_off"] + h["data_bytes"]], dtype=np.uint8).reshape(N, K * bits // 8)
    result = np.zeros((N, K), dtype=np.float32)

    if h["orthogonal"]:
        R = np.frombuffer(scales_arr[off: off + K * K * 2].tobytes(), dtype=np.float16).astype(np.float32).reshape(K, K)
        # Vectorized: unpack all rows, gather codebook, matmul
        # Process in chunks to avoid OOM on large embedding tables
        chunk = max(1, min(N, 4096 * 1024 * 1024 // (K * K * 4)))  # ~4 GB working set
        chunk = min(chunk, 2048)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idxs = unpack_indices_batch(data[start:end], bits, end - start, K)  # [c, K]
            dq = codebook[idxs]  # [c, K]
            # recon[c, j] = sum_i dq[c, i] * R[j, i]  → dq @ R.T
            recon = dq @ R.T  # [c, K]
            result[start:end] = recon * norms[start:end, 0:1] * isr[np.newaxis, :]
    else:
        left = scales_arr[off: off + gs].view(np.int8).astype(np.float32)
        off += gs
        right = scales_arr[off: off + gs].view(np.int8).astype(np.float32)
        off += gs
        perm = np.frombuffer(scales_arr[off: off + gs * 4].tobytes(), dtype=np.uint32)

        # Pre-build inv_perm for scatter: rotated[perm[k]] = ..., equiv rotated[:, perm] = cb * right
        inv_right_at_perm = right[perm]  # right[perm[k]] for each k, shape (gs,)

        # Reshape data as [N, ng, bytes_per_group]
        bpg = gs * bits // 8
        data_g = data.reshape(N, ng, bpg)

        for g in range(ng):
            idxs = unpack_indices_batch(data_g[:, g, :], bits, N, gs)  # [N, gs]
            # rotated[perm[k]] = codebook[idxs[k]] * right[perm[k]]
            # Equivalent: rotated_permuted[k] = codebook[idxs[k]] * right[perm[k]]
            # then scatter: for each row, result[perm[k]] = val[k]
            val = codebook[idxs] * inv_right_at_perm[np.newaxis, :]  # [N, gs]
            rotated = np.zeros((N, gs), dtype=np.float32)
            rotated[:, perm] = val  # scatter: rotated[:, perm[k]] = val[:, k]

            fwht_batch(rotated)  # [N, gs]

            # out[:, g*gs:(g+1)*gs] = rotated * left * norm[:, g:g+1] * isr[g*gs:(g+1)*gs]
            result[:, g * gs:(g + 1) * gs] = (
                rotated * left[np.newaxis, :] * norms[:, g:g + 1] * isr[np.newaxis, g * gs:(g + 1) * gs]
            )

    if hf_scale != 1.0:
        result *= hf_scale
    return result.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="v3 weights directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory for safetensors")
    parser.add_argument("--layers", type=str, default=None,
                        help="comma-separated layer indices to include (default: all)")
    parser.add_argument("--no_hf_scale", action="store_true",
                        help="skip cactus→HF inverse scale correction (default: apply it)")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
        import torch
    except ImportError:
        raise SystemExit("pip install safetensors torch")

    layer_filter: Optional[set[int]] = None
    if args.layers:
        layer_filter = {int(x.strip()) for x in args.layers.split(",")}

    args.out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, "torch.Tensor"] = {}
    skipped = 0
    written = 0

    for wf in sorted(args.weights.glob("*.weights")):
        stem = wf.stem
        hf = hf_name(stem)
        if hf is None:
            skipped += 1
            continue

        # apply layer filter
        if layer_filter is not None:
            parts = stem.split("_", 2)
            if parts[0] == "layer":
                if int(parts[1]) not in layer_filter:
                    skipped += 1
                    continue

        hf_scale = 1.0 if args.no_hf_scale else cactus_to_hf_scale(stem)
        print(f"  dequant {wf.name} → {hf} ...", end=" ", flush=True)
        arr = dequant_file(wf, hf_scale=hf_scale)
        if arr is None:
            print("skip (not CQ)")
            skipped += 1
            continue

        tensors[hf] = torch.from_numpy(arr)
        print(f"shape={arr.shape}")
        written += 1

    out_path = args.out / "model.safetensors"
    save_file(tensors, str(out_path))
    print(f"\nWrote {written} tensors to {out_path}  (skipped {skipped})")


if __name__ == "__main__":
    main()
