"""Pack TurboQuant-Hadamard (TQH) shipping artifacts into cactus weight files.

Three input artifacts under packed_shipping/:
    bf16_baseline/     full-precision tensors (norms, vision/audio towers, etc.)
    pli_embed/         PLI 2-bit (hadamard) + token embed 3-bit (orthogonal full-width)
    p3/                275 transformer linears 3-bit (hadamard, gs=128)

Output: a cactus-format weights directory with config.txt, vocab.txt, and one
.weights file per parameter. Quantized linears are emitted in TQ2 / TQ3 format
(cactus precisions 10 / 11). Everything else is FP16 via the existing converter.

Encoding alignment with cactus_tq2_load (cactus/kernel/kernel_tq2.cpp):
  - Indices LSB-first within each byte: byte = idx0 | (idx1<<b) | ...
    (TQH ships them MSB-first via numpy.packbits — we unpack and re-pack.)
  - Rotation stored as left_signs[gs] int8 || right_signs[gs] int8 || perm[gs] u32,
    derived from `seed + 17*group_dim` so the chain
        out[k] = tmp[inv_perm[k]]; out *= right; FWHT(out); out *= left
    reproduces  dq @ R^T  with R = (left.unsq * H_norm * right.unsq)[:, perm].

GEMMA4_WEIGHT_SCALE folding: tensor_io.py multiplies/divides certain Gemma4
tensors by 16.0 at save time; we fold the same factor into per-row scales so
the runtime sees pre-scaled values exactly like the FP16 path does.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .tensor_io import (
    CACTUS_MAGIC, CACTUS_ALIGNMENT,
    align_offset, compute_padding, format_config_value,
    save_tensor_with_header,
)

# Must match cactus/graph/graph.h Precision enum.
PRECISION_TQ1 = 9
PRECISION_TQ2 = 10
PRECISION_TQ3 = 11
PRECISION_TQ4 = 12

_BITS_TO_PRECISION = {1: PRECISION_TQ1, 2: PRECISION_TQ2, 3: PRECISION_TQ3, 4: PRECISION_TQ4}

GEMMA4_WEIGHT_SCALE = 16.0

# Tensors whose final value cactus expects to be (raw * GEMMA4_WEIGHT_SCALE).
_GEMMA4_MULT_BASENAMES = {'ffn_gate', 'ffn_up', 'per_layer_gate', 'moe_gate_proj', 'moe_up_proj'}
# Tensors whose final value cactus expects to be (raw / GEMMA4_WEIGHT_SCALE).
_GEMMA4_DIV_BASENAMES = {'token_embeddings', 'output_weight',
                         'embed_vision_proj', 'embed_vision_embedding'}


def _gemma4_scale_factor(out_filename: str) -> float:
    """Match the per-tensor scaling rules in tensor_io.save_tensor_with_header for gemma4."""
    base = out_filename[:-len('.weights')] if out_filename.endswith('.weights') else out_filename
    # Layer-prefixed tensors (layer_{i}_ffn_gate -> ffn_gate)
    parts = base.split('_', 2)
    if len(parts) == 3 and parts[0] == 'layer' and parts[1].isdigit():
        base = parts[2]
    if base in _GEMMA4_MULT_BASENAMES:
        return GEMMA4_WEIGHT_SCALE
    if base in _GEMMA4_DIV_BASENAMES:
        return 1.0 / GEMMA4_WEIGHT_SCALE
    return 1.0


# ── Codebook & rotation generation (matches tqh_runtime.py exactly) ──────────

def make_codebook(group_dim: int, bits: int,
                  grid_size: int = 200_001, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Lloyd-Max on the Beta(group_dim) coordinate distribution. Deterministic."""
    from scipy.special import gammaln
    n_centroids = 1 << bits
    if n_centroids <= 1:
        return np.array([0.0], dtype=np.float32)
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    log_c = gammaln(group_dim / 2.0) - 0.5 * math.log(math.pi) - gammaln((group_dim - 1.0) / 2.0)
    weights = math.exp(log_c) * np.power(np.clip(1.0 - grid * grid, 0.0, None),
                                          (group_dim - 3.0) / 2.0)
    quantiles = np.linspace(0.0, 1.0, n_centroids + 2, dtype=np.float64)[1:-1]
    cum = np.cumsum(weights); cum /= cum[-1]
    centroids = np.sort(np.clip(np.interp(quantiles, cum, grid), -1.0, 1.0))
    for _ in range(max_iter):
        boundaries = np.concatenate(([-1.0], (centroids[:-1] + centroids[1:]) / 2.0, [1.0]))
        updated = centroids.copy()
        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = ((grid >= lo) & (grid <= hi)) if i == n_centroids - 1 else ((grid >= lo) & (grid < hi))
            ws = weights[mask]
            if ws.sum() > 0:
                updated[i] = float((grid[mask] * ws).sum() / ws.sum())
        updated = np.sort(np.clip(updated, -1.0, 1.0))
        if np.max(np.abs(updated - centroids)) < tol:
            return updated.astype(np.float32)
        centroids = updated
    return centroids.astype(np.float32)


def make_hadamard_components(group_dim: int, bank_seed: int, variant: int = 0):
    """Returns (left, right, perm) — same RNG sequence as tqh_runtime.make_hadamard_rotation.

    cactus_tq2_load reads these as int8[gs] || int8[gs] || u32[gs] from the
    rotation blob, then derives inv_permutation at load time.
    """
    if torch is None:
        raise RuntimeError("torch is required to derive deterministic rotation components")
    if not (group_dim > 0 and (group_dim & (group_dim - 1)) == 0):
        raise ValueError(f"group_dim must be power of two, got {group_dim}")
    seed = bank_seed + 17 * group_dim + 7919 * int(variant)
    g = torch.Generator(device='cpu').manual_seed(seed)
    left = (2 * torch.randint(0, 2, (group_dim,), generator=g, dtype=torch.int64) - 1).to(torch.int8).numpy()
    right = (2 * torch.randint(0, 2, (group_dim,), generator=g, dtype=torch.int64) - 1).to(torch.int8).numpy()
    perm = torch.randperm(group_dim, generator=g).to(torch.int64).numpy().astype(np.uint32)
    return left, right, perm


def make_orthogonal_rotation(group_dim: int, bank_seed: int, variant: int = 0) -> np.ndarray:
    """Random orthogonal R[group_dim, group_dim] — matches tqh_runtime.make_orthogonal_rotation."""
    if torch is None:
        raise RuntimeError("torch is required to derive orthogonal rotation")
    seed = bank_seed + 17 * group_dim + 7919 * int(variant)
    g = torch.Generator(device='cpu').manual_seed(seed)
    a = torch.randn(group_dim, group_dim, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(a, mode='reduced')
    diag = torch.sign(torch.diagonal(r))
    diag[diag == 0] = 1
    return (q * diag).contiguous().numpy()


# ── Bit packing ──────────────────────────────────────────────────────────────

def unpack_bits_msb(packed: np.ndarray, bits: int, total_count: int) -> np.ndarray:
    """Inverse of tqh_runtime.pack_bits — read MSB-first packed uint8 stream."""
    if bits == 8:
        return packed[:total_count].astype(np.uint8)
    flat = np.unpackbits(packed.astype(np.uint8))[:total_count * bits]
    bit_array = flat.reshape(total_count, bits)
    out = np.zeros(total_count, dtype=np.uint8)
    for i in range(bits):
        out |= (bit_array[:, i] << (bits - 1 - i))
    return out


def pack_indices_lsb(indices_2d: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    """Pack indices LSB-first within each byte, group-by-group.

    Layout: per_group_bytes = group_size * bits / 8 bytes per group. Indices are
    laid out so that byte k contains the bits of indices [floor(8k/bits) ..],
    starting at the LSB. Equivalent to a little-endian bit stream concat.

    This matches the read pattern in cactus_tq2_load's dequant_group:
        byte = packed[k>>2]; idx[k]   = (byte) & 0x3
                              idx[k+1] = (byte >> 2) & 0x3
                              ...
    extended uniformly to bits in {1,2,3,4,8}.
    """
    N, K = indices_2d.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    G = K // group_size
    if bits == 8:
        return indices_2d.astype(np.uint8).reshape(N, K)
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"unsupported bits={bits}")

    # Build a per-group bit stream of (group_size * bits) bits, LSB-first.
    # Then pack 8 bits → byte, LSB-first within each byte.
    # Vectorized: emit bit b of every index, stride=bits.
    grp = indices_2d.reshape(N, G, group_size).astype(np.uint64)
    # per_group_bits = group_size * bits;  per_group_bytes = group_size * bits / 8
    per_group_bytes = group_size * bits // 8
    # Pack 8 indices at a time into a uint32 (or uint64) word: word = sum_i idx[i] << (i*bits)
    # We do it in fixed-size chunks of 8 indices to avoid overflow even at bits=4 (32 bits used).
    chunk = 8 if bits <= 4 else 1
    assert group_size % chunk == 0
    chunks_per_group = group_size // chunk
    bytes_per_chunk = chunk * bits // 8
    grp = grp.reshape(N, G, chunks_per_group, chunk)
    word = np.zeros((N, G, chunks_per_group), dtype=np.uint64)
    for i in range(chunk):
        word |= (grp[..., i] << (i * bits))
    # word now holds chunk*bits low bits; emit bytes_per_chunk LE bytes.
    out = np.zeros((N, G, chunks_per_group, bytes_per_chunk), dtype=np.uint8)
    for b in range(bytes_per_chunk):
        out[..., b] = ((word >> (8 * b)) & 0xFF).astype(np.uint8)
    return out.reshape(N, G * per_group_bytes)


# ── Cactus TQ2/TQ3 .weights writer ───────────────────────────────────────────

def write_tq_weights(
    out_path: Path,
    *,
    indices: np.ndarray,           # uint8 [N, K] (raw indices in [0, 2^bits))
    norms: np.ndarray,             # fp16 or fp32 [N, num_groups]  (per-row L2 norms)
    input_scale: Optional[np.ndarray],  # fp16 or fp32 [K] or None
    codebook: np.ndarray,          # fp32 [2^bits]
    group_size: int,
    bits: int,
    rotation_kind: str,            # "hadamard" or "orthogonal"
    rotation_seed: int = 1234,
    full_orth_R: Optional[np.ndarray] = None,  # required if rotation_kind == 'orthogonal'
    extra_scale: float = 1.0,      # folded into norms (e.g. GEMMA4_WEIGHT_SCALE)
):
    """Emit a TQ2/TQ3 .weights file matching cactus_tq{2,3}_load layout.

    Disk layout (TQ2 uses precision=10, TQ3 precision=11; both share the same
    header schema):

        offset 0    'CACT'
        offset 4    flags = 0
        offset 8    alignment = 32
        offset 12   ndim = 2
        offset 16   dim0 (rows N)
        offset 24   dim1 (cols K)
        offset 32   dim2,dim3 = 0
        offset 48   precision (10 or 11)
        offset 52   indices_bytes
        offset 60   scales_bytes
        offset 68   group_size
        offset 72   num_groups
        offset 76   bits_per_index
        offset 80   off_cb       fp32[2^bits]
        offset 88   off_is       fp16[K] (or zero if absent)
        offset 96   off_rot      hadamard: int8[gs]||int8[gs]||u32[gs]
                                 orth:    fp16[K*K]
        offset 104  off_sc       fp16[N * num_groups]
        offset 112  off_ix       packed indices (LSB-first within byte)
        offset 120  total file size
        offset 128  rotation_family (0 hadamard, 1 orth full-width)
        offset 132  has_input_scale (0/1)
        ↓ padded to 32 ↓
        codebook
        ↓ padded to 32 ↓
        input_scale (if present)
        ↓ padded to 32 ↓
        rotation blob
        ↓ padded to 32 ↓
        scales (norms × extra_scale)
        ↓ padded to 32 ↓
        packed indices
    """
    N, K = indices.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")
    num_groups = K // group_size
    if (1 << bits) != codebook.shape[0]:
        raise ValueError(f"codebook size {codebook.shape[0]} != 2^bits={1 << bits}")
    if bits not in _BITS_TO_PRECISION:
        raise ValueError(f"unsupported bits={bits}; expected one of {sorted(_BITS_TO_PRECISION)}")

    if rotation_kind == 'hadamard':
        rotation_family = 0
        left, right, perm = make_hadamard_components(group_size, rotation_seed)
        rot_blob = (left.astype(np.int8).tobytes()
                    + right.astype(np.int8).tobytes()
                    + perm.astype(np.uint32).tobytes())
    elif rotation_kind == 'orthogonal':
        rotation_family = 1
        if full_orth_R is None:
            full_orth_R = make_orthogonal_rotation(group_size, rotation_seed)
        if full_orth_R.shape != (group_size, group_size):
            raise ValueError(f"orth R shape mismatch: {full_orth_R.shape} vs {(group_size, group_size)}")
        rot_blob = full_orth_R.astype(np.float16).tobytes()
    else:
        raise ValueError(f"unknown rotation_kind={rotation_kind}")

    cb_bytes = codebook.astype(np.float32).tobytes()
    is_present = input_scale is not None
    is_bytes = input_scale.astype(np.float16).tobytes() if is_present else b''

    # Fold extra_scale into per-row norms (saves a runtime multiply).
    norms_eff = (norms.astype(np.float32) * float(extra_scale)).astype(np.float16)
    sc_bytes = norms_eff.tobytes()

    packed = pack_indices_lsb(indices.astype(np.uint8), group_size, bits)
    ix_bytes = packed.tobytes()

    HEADER_SIZE = 136  # extended (with rotation_family + has_input_scale)
    off_after_header = align_offset(HEADER_SIZE, CACTUS_ALIGNMENT)

    off_cb = off_after_header
    off_is = align_offset(off_cb + len(cb_bytes), CACTUS_ALIGNMENT)
    off_rot = align_offset((off_is + len(is_bytes)) if is_present else off_is, CACTUS_ALIGNMENT)
    off_sc = align_offset(off_rot + len(rot_blob), CACTUS_ALIGNMENT)
    off_ix = align_offset(off_sc + len(sc_bytes), CACTUS_ALIGNMENT)
    total = off_ix + len(ix_bytes)

    precision = _BITS_TO_PRECISION[bits]

    with open(out_path, 'wb') as f:
        f.write(CACTUS_MAGIC)                                    # 0
        f.write(struct.pack('<I', 0))                             # 4   flags
        f.write(struct.pack('<I', CACTUS_ALIGNMENT))              # 8   alignment
        f.write(struct.pack('<I', 2))                             # 12  ndim
        f.write(struct.pack('<Q', N))                             # 16  dim0
        f.write(struct.pack('<Q', K))                             # 24  dim1
        f.write(struct.pack('<Q', 0))                             # 32  dim2
        f.write(struct.pack('<Q', 0))                             # 40  dim3
        f.write(struct.pack('<I', precision))                     # 48
        f.write(struct.pack('<Q', len(ix_bytes)))                 # 52  indices_bytes
        f.write(struct.pack('<Q', len(sc_bytes)))                 # 60  scales_bytes
        f.write(struct.pack('<I', group_size))                    # 68
        f.write(struct.pack('<I', num_groups))                    # 72
        f.write(struct.pack('<I', bits))                          # 76
        f.write(struct.pack('<Q', off_cb))                        # 80
        f.write(struct.pack('<Q', off_is if is_present else 0))   # 88
        f.write(struct.pack('<Q', off_rot))                       # 96
        f.write(struct.pack('<Q', off_sc))                        # 104
        f.write(struct.pack('<Q', off_ix))                        # 112
        f.write(struct.pack('<Q', total))                         # 120
        f.write(struct.pack('<I', rotation_family))               # 128
        f.write(struct.pack('<I', 1 if is_present else 0))        # 132
        # header end at 136
        f.write(compute_padding(HEADER_SIZE, CACTUS_ALIGNMENT))

        f.write(cb_bytes)
        f.write(compute_padding(off_cb + len(cb_bytes), CACTUS_ALIGNMENT))

        if is_present:
            f.write(is_bytes)
            f.write(compute_padding(off_is + len(is_bytes), CACTUS_ALIGNMENT))

        f.write(rot_blob)
        f.write(compute_padding(off_rot + len(rot_blob), CACTUS_ALIGNMENT))

        f.write(sc_bytes)
        f.write(compute_padding(off_sc + len(sc_bytes), CACTUS_ALIGNMENT))

        f.write(ix_bytes)


# ── Reading TQH packed_shipping safetensors → cactus tensor ──────────────────

def _load_tqh_layer(packed_dir: Path, layer_name: str, layer_meta: dict, top_meta: dict):
    """Read one TQH-packed layer's raw tensors and return them in unpacked form."""
    from safetensors import safe_open
    N, K = layer_meta['shape']
    gs = layer_meta.get('group_size_runtime', top_meta['group_size'])
    bits_per_group = layer_meta['bits_per_group']
    if len(set(bits_per_group)) != 1:
        raise NotImplementedError(f"layer {layer_name}: mixed bits/group not supported here")
    bits = int(bits_per_group[0])
    # Indices: stored bit-packed (MSB-first via numpy.packbits) in TQH format.
    with safe_open(str(packed_dir / 'packed.safetensors'), framework='pt') as f:
        idx_raw = f.get_tensor(f"{layer_name}.indices").numpy()
        norms = f.get_tensor(f"{layer_name}.norms").numpy()  # fp16 [N, G]
        input_scale = (f.get_tensor(f"{layer_name}.input_scale").numpy()
                       if layer_meta.get('has_input_scale') else None)
    if layer_meta.get('index_format') == 'bitpacked':
        indices = unpack_bits_msb(idx_raw.flatten(), bits, N * K).reshape(N, K)
    else:
        indices = idx_raw.astype(np.uint8).reshape(N, K)
    return {
        'indices': indices, 'norms': norms, 'input_scale': input_scale,
        'group_size': gs, 'bits': bits,
        'rotation_family': layer_meta.get('rotation_family', top_meta.get('rotation_family', 'hadamard')),
        'rotation_group_dim': layer_meta.get('rotation_group_dim', gs),
    }


# ── HF → cactus name mapping for Gemma4 ──────────────────────────────────────

_GEMMA4_LINEAR_SUFFIX_MAP = {
    'self_attn.q_proj':       'attn_q',
    'self_attn.k_proj':       'attn_k',
    'self_attn.v_proj':       'attn_v',
    'self_attn.o_proj':       'attn_output',
    'mlp.gate_proj':          'ffn_gate',
    'mlp.up_proj':             'ffn_up',
    'mlp.down_proj':           'ffn_down',
    'per_layer_input_gate':    'per_layer_gate',
    'per_layer_projection':    'per_layer_proj',
}


def _hf_layer_name_to_cactus(hf_name: str) -> Optional[str]:
    """Map a HF Gemma4 transformer linear name to its cactus output filename.

    e.g. model.language_model.layers.0.self_attn.q_proj → layer_0_attn_q.weights
    """
    prefix = 'model.language_model.layers.'
    if not hf_name.startswith(prefix):
        return None
    rest = hf_name[len(prefix):]
    parts = rest.split('.', 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    layer_idx = int(parts[0])
    suffix = parts[1]
    cactus_suffix = _GEMMA4_LINEAR_SUFFIX_MAP.get(suffix)
    if cactus_suffix is None:
        return None
    return f"layer_{layer_idx}_{cactus_suffix}.weights"


# ── Main assembler ───────────────────────────────────────────────────────────

def assemble(baseline_dir: Path, pli_embed_dir: Path, transformer_dir: Path,
             output_dir: Path, *, bank_seed: Optional[int] = None) -> None:
    """Build a complete cactus weights/ dir from the three packed_shipping pieces."""
    baseline_dir = Path(baseline_dir)
    pli_embed_dir = Path(pli_embed_dir)
    transformer_dir = Path(transformer_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pli_meta = json.loads((pli_embed_dir / 'metadata.json').read_text())
    trans_meta = json.loads((transformer_dir / 'metadata.json').read_text())
    seed = bank_seed if bank_seed is not None else int(pli_meta.get('seed', 1234))
    if int(trans_meta.get('seed', seed)) != seed:
        raise ValueError("seed mismatch between pli_embed and transformer packs")

    # 1. Quantized transformer linears (per-layer bit-width: TQ1/TQ2/TQ3/TQ4
    # depending on metadata; uniform within each layer, mixed across layers).
    from collections import Counter
    bit_counts = Counter()
    print(f"[1/3] Packing {len(trans_meta['layers'])} transformer linears "
          f"(variant kind={trans_meta.get('kind')}) ...")
    for hf_name, layer_meta in trans_meta['layers'].items():
        cactus_name = _hf_layer_name_to_cactus(hf_name)
        if cactus_name is None:
            print(f"  skip (unmapped): {hf_name}")
            continue
        loaded = _load_tqh_layer(transformer_dir, hf_name, layer_meta, trans_meta)
        gs = loaded['group_size']
        cb = make_codebook(gs, loaded['bits'])
        scale = _gemma4_scale_factor(cactus_name)
        write_tq_weights(
            output_dir / cactus_name,
            indices=loaded['indices'], norms=loaded['norms'],
            input_scale=loaded['input_scale'], codebook=cb,
            group_size=gs, bits=loaded['bits'],
            rotation_kind='hadamard', rotation_seed=seed,
            extra_scale=scale,
        )
        bit_counts[loaded['bits']] += 1
    print(f"  bit-width distribution: {dict(sorted(bit_counts.items()))}")

    # 2. PLI embed (TQ2, hadamard, gs=128, has_input_scale) — embed_tokens_per_layer.
    # 3. Token embed (TQ3, orthogonal full-width K=1536) — token_embeddings.
    print(f"[2/3] Packing PLI + token_embeddings ...")
    for hf_name, layer_meta in pli_meta['layers'].items():
        loaded = _load_tqh_layer(pli_embed_dir, hf_name, layer_meta, pli_meta)
        gs = loaded['group_size']
        cb = make_codebook(gs, loaded['bits'])

        if hf_name == 'model.language_model.embed_tokens_per_layer':
            cactus_name = 'embed_tokens_per_layer.weights'
            scale = _gemma4_scale_factor(cactus_name)
            write_tq_weights(
                output_dir / cactus_name,
                indices=loaded['indices'], norms=loaded['norms'],
                input_scale=loaded['input_scale'], codebook=cb,
                group_size=gs, bits=loaded['bits'],
                rotation_kind='hadamard', rotation_seed=seed,
                extra_scale=scale,
            )
        elif hf_name == 'model.language_model.embed_tokens':
            cactus_name = 'token_embeddings.weights'
            rot_kind = loaded['rotation_family']
            scale = _gemma4_scale_factor(cactus_name)
            full_R = (make_orthogonal_rotation(gs, seed)
                      if rot_kind == 'orthogonal' else None)
            write_tq_weights(
                output_dir / cactus_name,
                indices=loaded['indices'], norms=loaded['norms'],
                input_scale=loaded['input_scale'], codebook=cb,
                group_size=gs, bits=loaded['bits'],
                rotation_kind=rot_kind, rotation_seed=seed,
                full_orth_R=full_R,
                extra_scale=scale,
            )
        else:
            print(f"  skip pli_embed unknown: {hf_name}")

    # 3. Everything else from bf16_baseline → FP16 via existing converter.
    # We synthesize a transformers-like state_dict-driven traversal manually here
    # because the existing convert_hf_model_weights() loads the model architecture
    # via from_pretrained and applies QDQ scaling. We want only the FP16 path.
    print(f"[3/3] Streaming bf16_baseline → FP16 .weights ...")
    _emit_baseline_fp16(baseline_dir, output_dir)

    # 4. Tokenizer files (vocab.txt + tokenizer_config.txt + chat_template.jinja2)
    print(f"[4/4] Emitting tokenizer ...")
    _emit_tokenizer(baseline_dir, output_dir)

    # 5. config.txt
    _emit_config_txt(baseline_dir, output_dir)
    print(f"Done. Output: {output_dir}")


def _emit_tokenizer(baseline_dir: Path, output_dir: Path) -> None:
    """Run the existing HF→cactus tokenizer converter against bf16_baseline."""
    from transformers import AutoTokenizer
    from .tokenizer import convert_hf_tokenizer
    tok = AutoTokenizer.from_pretrained(str(baseline_dir), local_files_only=True)
    convert_hf_tokenizer(tok, output_dir, model_type='gemma4')


def _emit_baseline_fp16(baseline_dir: Path, output_dir: Path) -> None:
    """Walk bf16_baseline/model.safetensors and emit FP16 .weights with the
    Gemma4 name map and the per-tensor scaling rules from tensor_io.py."""
    from safetensors import safe_open
    from .converter import _gemma_tower_output_name, _remap_gemma4_audio_keys
    from .weight_patterns import (
        GEMMA4_GLOBAL_WEIGHTS, GEMMA4_VISION_TOWER_PREFIX, GEMMA4_AUDIO_TOWER_PREFIX,
    )

    sf_path = baseline_dir / 'model.safetensors'
    with safe_open(str(sf_path), framework='pt') as f:
        keys = list(f.keys())
        sd = {k: f.get_tensor(k) for k in keys}

    sd = _remap_gemma4_audio_keys(sd)

    saved = set()

    # Per-layer FP16 tensors (norms, q_norm, layer_scalar, etc.).
    layer_prefix = 'model.language_model.layers.'
    layer_suffix_map = {
        'input_layernorm.weight':              'input_norm.weights',
        'post_attention_layernorm.weight':     'post_attn_norm.weights',
        'pre_feedforward_layernorm.weight':    'pre_ffn_norm.weights',
        'post_feedforward_layernorm.weight':   'post_ffn_norm.weights',
        'post_per_layer_input_norm.weight':    'post_per_layer_norm.weights',
        'self_attn.q_norm.weight':             'attn_q_norm.weights',
        'self_attn.k_norm.weight':             'attn_k_norm.weights',
        'layer_scalar':                        'layer_scalar.weights',
    }
    for k in list(sd.keys()):
        if not k.startswith(layer_prefix):
            continue
        rest = k[len(layer_prefix):]
        head, _, suf = rest.partition('.')
        if not head.isdigit():
            continue
        if suf not in layer_suffix_map:
            continue
        out_name = f"layer_{int(head)}_{layer_suffix_map[suf]}"
        save_tensor_with_header(sd[k], output_dir / out_name, 'FP16',
                                 model_type='gemma4')
        saved.add(k)

    # output_norm
    if 'model.language_model.norm.weight' in sd:
        save_tensor_with_header(sd['model.language_model.norm.weight'],
                                 output_dir / 'output_norm.weights', 'FP16',
                                 model_type='gemma4')
        saved.add('model.language_model.norm.weight')

    # Gemma4 global weights (PLI projector etc.) — skipping ones already TQ-packed.
    skip_names = {
        'model.language_model.embed_tokens_per_layer.weight',  # TQ2
        'model.language_model.embed_tokens.weight',            # TQ3
    }
    for hf_key, save_name in GEMMA4_GLOBAL_WEIGHTS:
        if hf_key in skip_names or hf_key not in sd:
            continue
        save_tensor_with_header(sd[hf_key], output_dir / save_name, 'FP16',
                                 model_type='gemma4')
        saved.add(hf_key)

    # Synthesize embed_vision_post_proj_norm (matches converter.py:480).
    text_hidden = sd.get('model.language_model.norm.weight')
    if text_hidden is not None:
        proj_norm = np.ones(int(text_hidden.shape[0]), dtype=np.float32)
        save_tensor_with_header(proj_norm, output_dir / 'embed_vision_post_proj_norm.weights',
                                 'FP16', model_type='gemma4')

    # Vision and audio towers.
    for hf_key in sorted(sd.keys()):
        if hf_key in saved:
            continue
        if hf_key.startswith(GEMMA4_VISION_TOWER_PREFIX):
            out_name = _gemma_tower_output_name(hf_key, GEMMA4_VISION_TOWER_PREFIX, 'vision_')
            save_tensor_with_header(sd[hf_key], output_dir / out_name, 'FP16',
                                     model_type='gemma4')
            saved.add(hf_key)
        elif hf_key.startswith(GEMMA4_AUDIO_TOWER_PREFIX):
            out_name = _gemma_tower_output_name(hf_key, GEMMA4_AUDIO_TOWER_PREFIX, 'audio_')
            save_tensor_with_header(sd[hf_key], output_dir / out_name, 'FP16',
                                     model_type='gemma4')
            saved.add(hf_key)

    # Report unsaved keys (debug aid).
    unsaved = sorted(set(sd.keys()) - saved)
    if unsaved:
        print(f"  note: {len(unsaved)} bf16_baseline keys went unsaved (first 10):")
        for k in unsaved[:10]:
            print(f"    {k}")


def _emit_config_txt(baseline_dir: Path, output_dir: Path) -> None:
    """Mirror what converter.py writes for a gemma4 model."""
    from .config_utils import (
        cfg_get, extract_base_config, extract_complex_gemma_config,
        extract_audio_config, extract_vision_config,
    )
    cfg_path = baseline_dir / 'config.json'
    raw = json.loads(cfg_path.read_text())

    class _Cfg(dict):
        def __getattr__(self, k):
            v = self.get(k)
            if isinstance(v, dict):
                return _Cfg(v)
            return v

    root = _Cfg(raw)
    text_cfg = _Cfg(raw.get('text_config', raw))
    audio_cfg = _Cfg(raw.get('audio_config', {})) if 'audio_config' in raw else None
    vision_cfg = _Cfg(raw.get('vision_config', {})) if 'vision_config' in raw else None

    cfg = extract_base_config(text_cfg, root)
    cfg['model_type'] = 'gemma4'
    cfg['tie_word_embeddings'] = bool(cfg_get(text_cfg, 'tie_word_embeddings',
                                              cfg_get(root, 'tie_word_embeddings', True)))
    cfg.update(extract_complex_gemma_config(text_cfg, root))
    if vision_cfg is not None:
        cfg.update(extract_vision_config(root, vision_cfg))
    if audio_cfg is not None:
        cfg.update(extract_audio_config(root, audio_cfg))
    cfg['precision'] = 'FP16'
    cfg['quantization'] = 'TQ3'
    cfg['model_variant'] = 'tqh_p3'

    with open(output_dir / 'config.txt', 'w') as f:
        for k, v in cfg.items():
            f.write(f"{k}={format_config_value(v)}\n")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Assemble TQH packed_shipping → cactus weights/")
    ap.add_argument('--baseline-dir',    required=True)
    ap.add_argument('--pli-embed-dir',   required=True)
    ap.add_argument('--transformer-dir', required=True)
    ap.add_argument('--output-dir',      required=True)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    assemble(Path(args.baseline_dir), Path(args.pli_embed_dir),
             Path(args.transformer_dir), Path(args.output_dir), bank_seed=args.seed)
