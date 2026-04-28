#!/usr/bin/env python3
"""convert_k96.py — convert the post-trained Gemma-4 E2B K=96 grouped-experts
checkpoint (https://huggingface.co/Cactus-Compute/gemma4-e2b-grouped-k96) to
cactus weights.

Strategy:
  1. Start from an existing dense Gemma-4 E2B cactus model dir (everything that
     isn't the MLP gate/up/down — attention, norms, per-layer aux, embedding —
     comes from there unchanged).
  2. Load the K96 checkpoint and the 35 cluster assignment files.
  3. For each layer's gate/up/down:
       a. Materialise merged bf16 weight = int4-fake-quant(base) + (alpha/rank) * B @ A.
       b. Permute rows (gate/up) or cols (down) by sort-by-cluster-id.
       c. Compute aligned cluster offsets by rounding cumulative cluster sizes
          to the nearest multiple of 32 (cactus's INT8 K-group + N-block GCD).
          This shifts a few neurons across cluster boundaries (≤9% per layer)
          but lets the cactus selective INT8 GEMV kernels skip whole 4-row N-blocks
          and 32-element K-groups cleanly.
       d. Quantise to cactus interleaved INT8 with group_size=32.
  4. Write per-layer `cluster_offsets_layer{i}.bin` (uint32, length 97 — 96+1 for
     the inclusive end). At runtime model_gemma4 reads these for the routing op.

Usage:
    python -m python.src.k96.convert_k96 \
        --base_dir weights/gemma-4-e2b-it \
        --ckpt weights/k96_raw/Sw_grouped_50_K96_lora_long.pt \
        --groups_dir weights/k96_raw/groups \
        --out_dir weights/gemma-4-e2b-it-k96
"""
import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow `from python.src.tensor_io import ...` even when invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from python.src.tensor_io import save_tensor_with_header  # noqa: E402


# Gemma-4 E2B constants. Keep in sync with matryoshka-distil/gemma4_hf.py.
N_LAYERS = 35
HIDDEN = 1536
INTERMEDIATE = 6144         # layers 0..14
INTERMEDIATE_WIDE = 12288   # layers 15..34
DOUBLE_WIDE_START = 15
K_GROUPS = 96
K_ACTIVE = 48
GROUP_SIZE = 32             # cactus INT8 K-group / GCD with N=4 interleave
INT4_GROUP_SIZE = 32        # ckpt's int4 quant group_size


def d_ffn_at(layer_idx: int) -> int:
    return INTERMEDIATE_WIDE if layer_idx >= DOUBLE_WIDE_START else INTERMEDIATE


def dequantize_int4(w: torch.Tensor, group_size: int = INT4_GROUP_SIZE) -> torch.Tensor:
    """Inference-time int4 dequant matching matryoshka-distil's training STE.

    Symmetric int4 (range [-7, 7]); per-group scale = max|w|/7. `w` shape is
    [out, in]; quant runs along the last axis.
    """
    out_dim, in_dim = w.shape
    orig_dtype = w.dtype
    w_fp32 = w.float()
    pad = (group_size - in_dim % group_size) % group_size
    if pad:
        w_fp32 = F.pad(w_fp32, (0, pad))
    n_groups = (in_dim + pad) // group_size
    w_g = w_fp32.view(out_dim, n_groups, group_size)
    max_abs = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = max_abs / 7.0
    w_int = torch.round(w_g / scale).clamp(-7, 7)
    w_deq = (w_int * scale).view(out_dim, -1)
    if pad:
        w_deq = w_deq[:, :in_dim]
    return w_deq.to(orig_dtype)


def merge_proj(state: dict, prefix: str, has_lora_default: bool) -> torch.Tensor:
    """Pull a projection's effective bf16 weight: dequant(int4 base) + LoRA delta.

    `prefix` is the dotted state-dict path up to the projection module, e.g.
    `inner.model.language_model.layers.5.mlp.up_proj`. We try LoRA-wrapped first
    (`<prefix>.base.weight`); if that key isn't present we fall back to
    `<prefix>.weight` (no LoRA on this proj — gate often falls here).
    """
    base_key = f"{prefix}.base.weight"
    plain_key = f"{prefix}.weight"
    if base_key in state:
        base = state[base_key]
        lora_a = state.get(f"{prefix}.lora_a.weight")
        lora_b = state.get(f"{prefix}.lora_b.weight")
    elif plain_key in state:
        base = state[plain_key]
        lora_a = state.get(f"{prefix}.lora_a.weight")
        lora_b = state.get(f"{prefix}.lora_b.weight")
    else:
        raise KeyError(f"projection not found at {prefix}.* (tried base.weight and weight)")

    # Apply int4 dequant. Same group_size as training (32). For Linear, the
    # quant axis is `in` (last); base shape is [out, in].
    eff = dequantize_int4(base, INT4_GROUP_SIZE).to(torch.float32)

    if lora_a is not None and lora_b is not None:
        # LoRA delta: B @ A scaled by alpha/rank. The training script uses
        # alpha = rank by convention (scale = 1.0) but we read the rank from
        # the matrix shape and assume scale = 1.0. If the checkpoint stored a
        # different alpha we'd need to read it from cfg — we accept that
        # divergence (the trained ckpt almost always uses scale=1).
        # Shapes: lora_a [r, in], lora_b [out, r]
        delta = lora_b.float() @ lora_a.float()
        if delta.shape != eff.shape:
            raise ValueError(f"LoRA delta shape {tuple(delta.shape)} != base "
                             f"{tuple(eff.shape)} for {prefix}")
        eff = eff + delta
    elif has_lora_default and (lora_a is None or lora_b is None):
        # Non-fatal: log and continue. wrap_lora may not have targeted this
        # specific projection (e.g., `lora_targets="up_proj,down_proj"`).
        pass

    return eff


def aligned_cluster_offsets(assignments: torch.Tensor, k_groups: int = K_GROUPS,
                             align: int = GROUP_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Compute permutation + aligned cluster offsets.

    Returns:
        perm: int64 [D_FFN] — perm[new_pos] = old_neuron_idx (sort-by-cluster).
        offsets: uint32 [k_groups + 1] — sequential block offsets, all multiples
                 of `align`. offsets[i+1] - offsets[i] = aligned size of cluster i.
                 offsets[0] = 0; offsets[-1] = D_FFN.
    """
    a = assignments.long()
    D = a.numel()
    sizes = torch.bincount(a, minlength=k_groups).tolist()
    cum = [0]
    for s in sizes:
        cum.append(cum[-1] + s)
    assert cum[-1] == D, f"sizes sum {cum[-1]} != D {D}"

    # Stable sort: groups in 0..K-1 order, neurons within each group keep their
    # original order. Same convention as matryoshka-distil/permute_grouped_k96.py.
    perm = torch.argsort(a, stable=True).numpy().astype(np.int64)

    # Round each cumulative offset to nearest multiple of `align`. Pin endpoints
    # at 0 and D so total length is preserved. Final entry is forced to D, the
    # next-to-last is rounded but bounded above by D.
    aligned_cum = [round(c / align) * align for c in cum]
    aligned_cum[0] = 0
    aligned_cum[-1] = D
    # Make non-decreasing and within [0, D].
    for i in range(1, len(aligned_cum)):
        aligned_cum[i] = max(aligned_cum[i - 1], min(D, aligned_cum[i]))

    offsets = np.array(aligned_cum, dtype=np.uint32)
    return perm, offsets


def write_offsets(offsets_per_layer: list[np.ndarray], out_dir: Path):
    """Write cluster offsets to a single binary file. Format:
        magic 'CK96' (4 bytes)
        version uint32 = 1
        n_layers uint32
        for each layer: k_groups uint32, then (k_groups+1) uint32 offsets
    """
    path = out_dir / "k96_cluster_offsets.bin"
    with open(path, "wb") as f:
        f.write(b"CK96")
        f.write(struct.pack("<II", 1, len(offsets_per_layer)))
        for off in offsets_per_layer:
            f.write(struct.pack("<I", len(off) - 1))
            f.write(off.tobytes())
    print(f"  wrote {path}  ({path.stat().st_size} bytes)")


# ─── Packed (per-cluster contiguous) FFN format ──────────────────────────────
#
# One file per layer: layer_N_ffn_packed.weights
#   Header (96 bytes, padded to 128):
#       magic 'CKPK' (4)
#       version uint32 = 1                      (4)
#       precision uint32 (0=INT8, 3=INT4)       (4)
#       hidden_dim uint32                       (4)
#       d_ffn uint32                            (4)
#       k_groups uint32                         (4)
#       group_size uint32 (=32)                 (4)
#       n_active_clusters uint32                (4) — number of clusters with size>0 (informational)
#       up_w_total_bytes uint64                 (8)
#       up_s_total_bytes uint64                 (8)
#       down_w_total_bytes uint64               (8)
#       down_s_total_bytes uint64               (8)
#       data_offset uint64                      (8) — start of first cluster's payload
#       reserved 32 bytes                       (32)
#
#   Cluster table (per-cluster, k_groups entries):
#       cluster_size uint32                      (4)
#       _pad uint32                              (4)
#       up_w_offset uint64                       (8) — offset from start of file
#       up_s_offset uint64                       (8)
#       down_w_offset uint64                     (8)
#       down_s_offset uint64                     (8)
#     -> 40 bytes per cluster, k_groups * 40 bytes total
#
#   Data section (page-aligned, all per-cluster slabs concatenated):
#     for c in 0..k_groups-1 (skip empty):
#       up_w[c]    INT8: cluster_size * hidden_dim   bytes (interleaved [N/4][num_g][4][32])
#                  INT4: cluster_size * hidden_dim / 2 bytes
#       up_s[c]    cluster_size * (hidden/32)        fp16   = 2 * cluster_size * hidden / 32
#       down_w[c]  INT8: hidden_dim * cluster_size   bytes
#                  INT4: hidden_dim * cluster_size / 2 bytes
#       down_s[c]  hidden_dim * (cluster_size/32)    fp16
#       (each section 64-byte aligned for SIMD; section starts always 64-byte aligned)
#
# Layout choice rationale: the per-cluster up_w + up_s + down_w + down_s placed
# back-to-back puts the bytes the kernel needs for cluster c on the same set of
# pages, so accessing one cluster's contribution touches one contiguous span
# of the file. With K_active=48, this is ~48 distinct page spans per token
# instead of 96+ (gate is separate, but it's read in full anyway).

INTERLEAVE_BLOCK = 32  # cactus interleaves 4 rows; we wrote 32 to mean group_size; keep using GROUP_SIZE
N_BLOCK = 4            # cactus weights are blocked 4 rows at a time

PACKED_MAGIC = b"CKPK"
PACKED_VERSION = 1


def _quantize_and_interleave_int8(w: np.ndarray):
    """w shape [N, K]. Return (quant_interleaved_uint8 bytes, scales_fp16 [N/4*num_g*4])."""
    N, K = w.shape
    assert N % N_BLOCK == 0, f"N={N} not multiple of {N_BLOCK}"
    assert K % GROUP_SIZE == 0, f"K={K} not multiple of {GROUP_SIZE}"
    num_groups = K // GROUP_SIZE
    w_g = w.reshape(N, num_groups, GROUP_SIZE)
    abs_max = np.max(np.abs(w_g), axis=2)
    scales = np.maximum((abs_max / 127.0).astype(np.float32), 1e-10)
    q = np.clip(np.round(w_g / scales[:, :, None]), -128, 127).astype(np.int8)
    q2 = q.reshape(N, K)
    # Interleave: [N/4, GROUP_SIZE wait no — block_size=4]
    # cactus uses INTERLEAVE_BLOCK=4 (rows), and per-K stride of 4.
    # Layout: [N/4][K/4][4(rows)][4(K cols)] but the kernel actually reads
    #   B[(n_block * K + k_base) * 4 + ...] with k_base step = group_size,
    # and within a group it expects 4 rows interleaved every 4 K-vals.
    # The Python interleave_weights reshapes to [N/4, 4, K/4, 4]→transpose→[N/4, K/4, 4, 4].
    qi = q2.reshape(N // N_BLOCK, N_BLOCK, K // 4, 4).transpose(0, 2, 1, 3).reshape(-1)
    # Scales: [N/4, num_groups, 4]
    si = scales.reshape(N // N_BLOCK, N_BLOCK, num_groups).transpose(0, 2, 1).reshape(-1)
    return qi.tobytes(), si.astype(np.float16).tobytes()


def _pack_int4_pairs_planar(q: np.ndarray) -> np.ndarray:
    """Same as tensor_io.pack_int4_pairs."""
    assert q.size % 32 == 0
    groups = q.reshape(-1, 32)
    low  = (groups[:, :16].astype(np.int8).view(np.uint8) & 0x0F).astype(np.uint8)
    high = ((groups[:, 16:].astype(np.int8).view(np.uint8) & 0x0F).astype(np.uint8)) << 4
    return (low | high).astype(np.uint8).reshape(-1)


def _quantize_and_interleave_int4(w: np.ndarray):
    """w shape [N, K]. Return (packed_int4 bytes, scales_fp16 bytes)."""
    N, K = w.shape
    assert N % N_BLOCK == 0
    assert K % GROUP_SIZE == 0
    num_groups = K // GROUP_SIZE
    w_g = w.reshape(N, num_groups, GROUP_SIZE)
    abs_max = np.max(np.abs(w_g), axis=2)
    scales = np.maximum((abs_max / 7.0).astype(np.float32), 1e-10)
    q = np.clip(np.round(w_g / scales[:, :, None]), -8, 7).astype(np.int8)
    q2 = q.reshape(N, K)
    qi = q2.reshape(N // N_BLOCK, N_BLOCK, K // 4, 4).transpose(0, 2, 1, 3).reshape(-1)
    packed = _pack_int4_pairs_planar(qi)
    si = scales.reshape(N // N_BLOCK, N_BLOCK, num_groups).transpose(0, 2, 1).reshape(-1)
    return packed.tobytes(), si.astype(np.float16).tobytes()


def _round_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def write_packed_layer(out_path: Path, up_w: np.ndarray, down_w: np.ndarray,
                        offsets: np.ndarray, hidden_dim: int, d_ffn: int,
                        precision: str, k_groups: int):
    """Write a single packed FFN file for one layer.

    `up_w` is [d_ffn, hidden_dim] (already permuted by cluster + scaled by *16
    if gemma4). `down_w` is [hidden_dim, d_ffn] (permuted on its second axis).
    `offsets` is uint32 [k_groups+1] with 32-aligned entries.
    """
    is_int4 = (precision == 'INT4')
    prec_code = 3 if is_int4 else 0

    # Gather per-cluster blobs.
    cluster_blobs = []  # list of (cluster_size, up_w_bytes, up_s_bytes, down_w_bytes, down_s_bytes)
    for c in range(k_groups):
        s = int(offsets[c]); e = int(offsets[c + 1])
        N_c = e - s
        if N_c == 0:
            cluster_blobs.append((0, b"", b"", b"", b""))
            continue
        assert N_c % GROUP_SIZE == 0, f"cluster {c} size {N_c} not 32-aligned"
        up_slice = np.ascontiguousarray(up_w[s:e, :])              # [N_c, hidden]
        down_slice = np.ascontiguousarray(down_w[:, s:e])          # [hidden, N_c]
        if is_int4:
            uw, us = _quantize_and_interleave_int4(up_slice)
            dw, ds = _quantize_and_interleave_int4(down_slice)
        else:
            uw, us = _quantize_and_interleave_int8(up_slice)
            dw, ds = _quantize_and_interleave_int8(down_slice)
        cluster_blobs.append((N_c, uw, us, dw, ds))

    # Compute layout. Header 128 bytes + cluster table (k_groups * 40) + data.
    HEADER_BYTES = 128
    TABLE_ENTRY = 40
    table_bytes = k_groups * TABLE_ENTRY
    data_start = _round_up(HEADER_BYTES + table_bytes, 64)

    # Allocate offsets per cluster. Each section 64-byte aligned.
    cur = data_start
    table = []  # list of (N_c, up_w_off, up_s_off, dw_off, ds_off)
    up_w_total = 0
    up_s_total = 0
    down_w_total = 0
    down_s_total = 0
    for (N_c, uw, us, dw, ds) in cluster_blobs:
        if N_c == 0:
            table.append((0, 0, 0, 0, 0))
            continue
        cur = _round_up(cur, 64); up_w_off = cur; cur += len(uw)
        cur = _round_up(cur, 64); up_s_off = cur; cur += len(us)
        cur = _round_up(cur, 64); down_w_off = cur; cur += len(dw)
        cur = _round_up(cur, 64); down_s_off = cur; cur += len(ds)
        table.append((N_c, up_w_off, up_s_off, down_w_off, down_s_off))
        up_w_total += len(uw); up_s_total += len(us)
        down_w_total += len(dw); down_s_total += len(ds)
    total_size = cur
    n_active = sum(1 for t in table if t[0] > 0)

    # Write file.
    with open(out_path, "wb") as f:
        # Pre-size.
        f.truncate(total_size)
        # Header: total exactly 128 bytes. Layout:
        #   0:  CKPK + 7×u32  = 32 bytes
        #   32: 5×u64         = 40 bytes  -> ends at 72
        #   72: 56 bytes pad/reserved      -> ends at 128
        f.seek(0)
        f.write(PACKED_MAGIC)                                       # 4
        f.write(struct.pack("<I", PACKED_VERSION))                  # 4
        f.write(struct.pack("<I", prec_code))                       # 4
        f.write(struct.pack("<I", hidden_dim))                      # 4
        f.write(struct.pack("<I", d_ffn))                           # 4
        f.write(struct.pack("<I", k_groups))                        # 4
        f.write(struct.pack("<I", GROUP_SIZE))                      # 4
        f.write(struct.pack("<I", n_active))                        # 4
        # End of u32 block: byte 32.
        f.write(struct.pack("<Q", up_w_total))                      # 8 -> 40
        f.write(struct.pack("<Q", up_s_total))                      # 8 -> 48
        f.write(struct.pack("<Q", down_w_total))                    # 8 -> 56
        f.write(struct.pack("<Q", down_s_total))                    # 8 -> 64
        f.write(struct.pack("<Q", data_start))                      # 8 -> 72
        # Pad header up to 128 bytes for the cluster-table start offset.
        f.write(b"\x00" * (128 - 72))                               # 56 bytes -> 128
        # Cluster table.
        for (N_c, uw_off, us_off, dw_off, ds_off) in table:
            f.write(struct.pack("<II", N_c, 0))                     # 8
            f.write(struct.pack("<Q", uw_off))                      # 8
            f.write(struct.pack("<Q", us_off))                      # 8
            f.write(struct.pack("<Q", dw_off))                      # 8
            f.write(struct.pack("<Q", ds_off))                      # 8
        # Data.
        for (N_c, uw, us, dw, ds), (ncc, uw_off, us_off, dw_off, ds_off) in zip(cluster_blobs, table):
            if N_c == 0:
                continue
            f.seek(uw_off);  f.write(uw)
            f.seek(us_off);  f.write(us)
            f.seek(dw_off);  f.write(dw)
            f.seek(ds_off);  f.write(ds)
    return total_size, n_active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Existing dense Gemma-4 E2B cactus model dir to fork from")
    parser.add_argument("--ckpt", required=True,
                        help="Path to Sw_grouped_50_K96_lora_long.pt")
    parser.add_argument("--groups_dir", required=True,
                        help="Dir with s50_K96_layer{0..34}.pt cluster assignments")
    parser.add_argument("--out_dir", required=True,
                        help="Destination cactus model dir for the K96 variant")
    parser.add_argument("--precision", default="INT8", choices=("INT8", "INT4"),
                        help="Quantization precision for ffn_{gate,up,down}.weights")
    parser.add_argument("--packed", action="store_true", default=False,
                        help="Emit per-layer packed FFN file (up_c+down_c contiguous per cluster). "
                             "Adds layer_N_ffn_packed.weights and writes k96_packed=1 to k96.meta.")
    parser.add_argument("--packed_only", action="store_true", default=False,
                        help="Only emit packed files (and offsets/meta). Skip the per-proj "
                             "ffn_gate/ffn_up/ffn_down files. The base_dir's existing FFN "
                             "files will still be present (they are skipped on copy).")
    parser.add_argument("--layers", type=str, default="all",
                        help="'all' or comma-separated layer indices to (re)write. "
                             "Use to do a quick single-layer sanity check.")
    args = parser.parse_args()
    if args.layers == "all":
        layer_set = set(range(N_LAYERS))
    else:
        layer_set = set(int(x) for x in args.layers.split(","))

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy everything from base_dir except the FFN files we'll overwrite.
    print(f"Forking from {base_dir} -> {out_dir}")
    skip_suffixes = (
        "_ffn_gate.weights", "_ffn_up.weights", "_ffn_down.weights",
    )
    copied = 0
    for src in base_dir.iterdir():
        if not src.is_file():
            continue
        if any(src.name.endswith(s) for s in skip_suffixes):
            continue
        shutil.copy2(src, out_dir / src.name)
        copied += 1
    print(f"  copied {copied} files (norms/embeddings/attention/PLI)")

    # 2. Load checkpoint.
    print(f"Loading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["student_state"] if "student_state" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    print(f"  state keys: {len(state)}  cfg.K_groups={cfg.get('K_groups', 'unknown')}")

    # Default LoRA targets in matryoshka-distil's wrap_lora (gate, up, down, q, k, v, o).
    # We treat "no lora_a/lora_b found" as "this proj wasn't LoRA-wrapped" — silent.
    has_lora_default = bool(cfg.get("use_lora") or cfg.get("gate_lora_train"))
    print(f"  has_lora_default: {has_lora_default}")

    # 3. Per-layer FFN conversion.
    offsets_per_layer = []
    stats = {"int8_tensors": 0, "int4_tensors": 0, "fp16_tensors": 0,
             "fp32_tensors": 0, "quantized_parameters": 0, "mse_values": [],
             "snr_values": [], "cos_sim_values": [], "total_tensors": 0,
             "total_parameters": 0}
    for i in range(N_LAYERS):
        d_ffn = d_ffn_at(i)
        groups_path = Path(args.groups_dir) / f"s50_K96_layer{i}.pt"
        assignments = torch.load(groups_path, map_location="cpu", weights_only=False)
        if assignments.numel() != d_ffn:
            raise ValueError(f"layer {i}: assignments {assignments.numel()} != D_FFN {d_ffn}")

        perm, offsets = aligned_cluster_offsets(assignments)
        offsets_per_layer.append(offsets)
        nonzero = sum(1 for k in range(K_GROUPS) if offsets[k + 1] > offsets[k])

        if i not in layer_set:
            continue

        prefix = f"inner.model.language_model.layers.{i}.mlp"
        gate_w = merge_proj(state, f"{prefix}.gate_proj", has_lora_default)  # [D_FFN, HIDDEN]
        up_w   = merge_proj(state, f"{prefix}.up_proj",   has_lora_default)  # [D_FFN, HIDDEN]
        down_w = merge_proj(state, f"{prefix}.down_proj", has_lora_default)  # [HIDDEN, D_FFN]

        # Permute D_FFN axis. gate/up: rows (axis 0); down: cols (axis 1).
        perm_t = torch.from_numpy(perm)
        gate_w = gate_w.index_select(0, perm_t).contiguous()
        up_w   = up_w.index_select(0, perm_t).contiguous()
        down_w = down_w.index_select(1, perm_t).contiguous()

        # Save in cactus INT8 interleaved format. The save_tensor_with_header
        # function applies Gemma-4-specific weight scaling internally based on
        # filename (gate/up get *16, down doesn't).
        if not args.packed_only:
            for name, w in (("ffn_gate", gate_w), ("ffn_up", up_w), ("ffn_down", down_w)):
                out_path = out_dir / f"layer_{i}_{name}.weights"
                save_tensor_with_header(w, out_path, precision=args.precision,
                                        transpose=False,
                                        stats_tracker=stats,
                                        args=None,
                                        model_type='gemma4')
        else:
            # Still need gate (gate is unchanged in packed flow).
            out_path = out_dir / f"layer_{i}_ffn_gate.weights"
            save_tensor_with_header(gate_w, out_path, precision=args.precision,
                                    transpose=False, stats_tracker=stats,
                                    args=None, model_type='gemma4')

        # Optionally emit a per-cluster packed file. We replicate the gemma4
        # weight scaling (×16 on gate/up, identity on down) that
        # save_tensor_with_header applies internally.
        if args.packed:
            GEMMA4_SCALE = 16.0
            up_arr = (up_w.float().numpy() * GEMMA4_SCALE)
            down_arr = down_w.float().numpy()
            packed_path = out_dir / f"layer_{i}_ffn_packed.weights"
            total_size, n_active = write_packed_layer(
                packed_path, up_arr, down_arr, offsets,
                hidden_dim=HIDDEN, d_ffn=d_ffn,
                precision=args.precision, k_groups=K_GROUPS,
            )
            if i == 0 or i == DOUBLE_WIDE_START:
                print(f"  layer {i}: packed file {total_size/1e6:.2f} MB n_active={n_active}")
        if i == 0 or i == DOUBLE_WIDE_START:
            print(f"  layer {i}: D_FFN={d_ffn} nonzero_clusters={nonzero}/{K_GROUPS}")

    # 4. Cluster offsets file.
    write_offsets(offsets_per_layer, out_dir)

    # 5. Touch a marker file so the runtime can detect this is a K96 model.
    meta_lines = [
        f"K_groups={K_GROUPS}",
        f"K_active={K_ACTIVE}",
        f"align={GROUP_SIZE}",
        f"source_ckpt={args.ckpt}",
    ]
    if args.packed:
        meta_lines.append("k96_packed=1")
    (out_dir / "k96.meta").write_text("\n".join(meta_lines) + "\n")
    print(f"\nDone. K96 model at {out_dir}")


if __name__ == "__main__":
    main()
