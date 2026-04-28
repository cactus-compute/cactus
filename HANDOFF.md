# TQH (TurboQuant-Hadamard) bring-up in cactus — handoff

Branch: `karen/tq` — built on top of commit `269b6eb0` ("cleaned tq implementation").

## Goal

Pack `packed_shipping/` artifacts (TQH-quantized gemma-4-e2b-it) into cactus
weights format with `tq1`/`tq2`/`tq3`/`tq4` precision labels and run them in
the cactus engine. Format math:

```
W[n, gs*g+k] = (FWHT(P^T (codebook[indices] ⊙ right_signs)))[k] * left_signs[k]
              * row_norm[n, g] / input_scale[gs*g + k]
```

Codebook is deterministic from `(group_dim, bits)` via Lloyd-Max on the Beta
coordinate distribution. Rotation is randomized hadamard (`gs=128`) for
transformer linears + PLI, full-width orthogonal (`K=1536`) for the LM token
embedding.

`packed_shipping/` variants (analyzed):

| Dir | Bits used | Notes |
|---|---|---|
| `bf16_baseline/` | — | norms, vision/audio towers, embed projectors (FP16) |
| `pli_embed/` | 2 (PLI), 3 (orth-embed) | both |
| `p3/` | 3, 4 | per-layer mix (207 @ 3-bit, 68 @ 4-bit) |
| `p6/` | 1, 2, 3, 4 | per-layer mix |
| `tqh_u3/` | 3 | uniform |
| `tqh_u4/` | 4 | uniform |

Each layer is uniform internally; bit-widths vary across layers.

## What's been built

### Disk format (TQ3/TQ4 share, extends TQ2)

`.weights` header is 136 bytes (TQ2 legacy is 128 — kernel_tq2 still ignores
the extension fields and works unchanged). Layout:

```
0    'CACT'
4    flags
8    alignment = 32
12   ndim = 2
16   dim0 (rows N)
24   dim1 (cols K)
32   dim2 = dim3 = 0
48   precision   (9=TQ1, 10=TQ2, 11=TQ3, 12=TQ4)
52   indices_bytes
60   scales_bytes
68   group_size
72   num_groups
76   bits_per_index
80   off_cb        → fp32[2^bits]
88   off_is        → fp16[K]   (or 0)
96   off_rot       → hadamard:  int8[gs] left || int8[gs] right || u32[gs] perm
                  → orth full: fp16[K*K] R
104  off_sc        → fp16[N * num_groups]   row L2 norms
112  off_ix        → packed indices, LSB-first within byte
120  total file size
128  rotation_family (0=hadamard, 1=orth full-width)
132  has_input_scale
```

Index packing is LSB-first within each byte:
- `bits=1`: `idx[i] = (byte >> i) & 1`
- `bits=2`: 4 idx/byte (matches existing TQ2)
- `bits=3`: 8 idx per 24-bit LE word (3 bytes)
- `bits=4`: low nibble = `idx[2k]`, high nibble = `idx[2k+1]`

### Python (`python/src/`)

- **`tqh_pack.py`** — packer.
  - `make_codebook(gs, bits)`, `make_hadamard_components(gs, seed)`,
    `make_orthogonal_rotation(K, seed)` mirror `tqh_runtime.py` RNG sequence
    bit-exactly so the cactus kernel's chain (`inv_perm → right → FWHT → left`)
    reproduces the reference `dq @ R^T`.
  - `pack_indices_lsb` packs uint8 indices into the LSB-first cactus layout
    for any `bits ∈ {1,2,3,4}`.
  - `write_tq_weights` dispatches `bits → precision` via `_BITS_TO_PRECISION`.
  - `assemble(baseline_dir, pli_embed_dir, transformer_dir, output_dir)` walks
    the three sources, dispatches each layer to TQ1/2/3/4 based on its
    `index_bits` (per-layer uniform), and emits FP16 for everything else via
    `tensor_io.save_tensor_with_header(... model_type='gemma4')` so the
    `GEMMA4_WEIGHT_SCALE = 16` per-tensor scaling is applied correctly.
    Tokenizer is emitted via the existing `convert_hf_tokenizer`.
  - HF→cactus name map covers q/k/v/o, mlp gate/up/down, **and** the
    `per_layer_input_gate` / `per_layer_projection` suffixes (those were
    initially missed; now fixed). The `_gemma4_scale_factor` helper folds the
    runtime ×16 / ÷16 expectation into per-row `norms` at pack time.
- **`tqh_pack_verify.py`** — round-trip checker. Packs one layer with
  `write_tq_weights`, parses the `.weights` file back from raw bytes, runs the
  reference dehydration, and bit-checks the rotation components against
  `tqh_runtime.make_hadamard_rotation`. **All four bit-widths pass** (max abs
  err under 5e-3 vs reference, rotation components bit-exact).

### C++ (`cactus/`)

- **`graph/graph.h`** — `Precision::TQ1=9 / TQ3=11 / TQ4=12` added; PrecisionTraits
  switches updated; `BufferDesc::tqn` field; `MappedFile::tqn()` accessor;
  `MappedFile::dequant_to_fp16()` declared; `tqn_` and `dequanted_fp16_`
  private members.
- **`kernel/kernel.h`** — `CactusTQN` generic descriptor (carries `bits`,
  `rotation_family`, both hadamard sign/perm pointers and the orth `R*`).
  Functions `cactus_tqn_load`, `cactus_tqn_dequant_row`,
  `cactus_tqn_dequant_layer`, the bring-up companion `cactus_tq2_dequant_layer`,
  and the (currently unused) `cactus_gemv_tq4_hadamard_f16`.
- **`kernel/kernel_tqn.cpp`** (new) — single TU handles TQ1/TQ3/TQ4. The
  hadamard chain is shared; the unpack step switches on `bits` (1-bit
  byte-shift, 3-bit 8-per-3-bytes LE word, 4-bit nibble pair). Orth full-width
  uses an fp16 NEON dot accumulating in fp32. `kernel_tq2.cpp` is left
  untouched — the legacy 128-byte TQ2 header still loads via `cactus_tq2_load`
  (the 8 trailing bytes my packer adds are ignored by it).
- **`graph/graph_io.cpp`** — `parse_header` dispatches TQ3/TQ4 (and TQ1) into
  `cactus_tqn_load`. `mmap_weights` lifts the prior TQ2-embedding-only
  restriction: any TQ tensor used as a weight pre-dequants to fp16 via
  `dequant_to_fp16()`, after which the file presents an FP16 face and the
  standard fp16 matmul path takes over. `MappedFile::data()` returns the
  dequanted buffer when present. `BufferDesc.tqn` is populated even after
  pre-dequant so a future fused gemv path can still see the packed bytes.
- **`tests/test_gemma4_suite.cpp`** — `main(int argc, char** argv)` now
  filters by name, e.g. `test_gemma4_suite 1k_context` runs only that.

## What works (validated)

End-to-end inference on `tqh_u4` after assembly:

```
CACTUS_TEST_GEMMA4_MODEL=/Users/karen/cactus/weights/gemma-4-e2b-it-tqh-u4 \
  /Users/karen/cactus/tests/build/test_gemma4_suite 1k_context
```

Numbers (1618-token prefill, 74-token decode):

```
prefill_tps:  572.68
decode_tps:    6.65
ram_usage_mb: ~5400
ttft:          2.83 s
```

Audio transcription works without `embed_audio.embedding` (graceful zero-init).

Round-trip numerical verification at the Python layer for **every bit-width**
(`python -m src.tqh_pack_verify --packed-dir packed_shipping/<dir> --layer
<name>`).

## What's broken / what we tried

### Fused TQ4 hadamard gemv → was 14× *slower* than baseline

I added `cactus_gemv_tq4_hadamard_f16` (in `kernel_tqn.cpp`) and dispatched
to it from `compute_matmul_node` (`graph_ops_nn.cpp`) when `M==1 && tqn->bits==4
&& rotation_family==0`. Decode dropped from 6.65 → 0.45 tps.

**The dispatch is currently reverted** but the kernel and `BufferDesc.tqn`
plumbing are still in tree, so re-enabling is a single hunk in
`compute_matmul_node` (the block right above the `if
(PrecisionTraits::is_integer(...) && group_size > 0)` check).

### Why it was slow (diagnosis, not yet acted on)

1. **`std::vector<__fp16> x_scaled(K)` allocates per call.** With 245 matmuls
   per decode token × 74 decode tokens = ~18k mallocs/frees. Should be a
   `thread_local` reusable buffer.
2. **`vdivq_f16` for the input_scale fold.** No native fp16 divide on Apple
   Silicon — emits a Newton-Raphson sequence per lane. Cheap fix: precompute
   `recip_input_scale = 1/input_scale` once at first use and `vmulq_f16`.
3. **Single-row loop, no row-block tiling.** INT4 gemv processes 8 outputs in
   parallel (`n_block` × 4 wide × 2-unroll). The fused TQ4 version processes
   one output row at a time, redoing all the per-row dequant work serially
   even though x_scaled is shared. Should run 4 rows in parallel and share
   the activation slice loaded into registers.
4. **Scalar codebook lookup.** The 4-bit nibble→fp16 gather is a hot scalar
   loop. Should use `vqtbl1q_u8` / `vqtbl2q_u8` against a 32-byte fp16
   codebook, or exploit codebook symmetry (`cb[i] = -cb[15-i]`) to halve the
   LUT and use the high index bit as a sign.
5. **Inv-permutation gather is also scalar.** `y[k] = tmp[inv_perm[k]]`. NEON
   `vqtbl1q_u8` with a precomputed permutation in byte form would do 16
   gathers per op.

For decode (memory-bound, M=1), TQ4 *should* be at least as fast as INT4
because the per-element memory traffic is comparable (`bits=4 + tiny scales`
vs INT4 `+ groupwise scales`). The compute overhead of FWHT + signs + perm
is real but small per element (~12 fp16 ops/element extra) and decode isn't
compute-bound. So the headroom is there — the current kernel just leaves a
lot on the floor.

### Load-time slowness (the user reported "actual wait was much longer")

The reported `total_time_ms` is inference only. Pre-dequanting all TQ tensors
to FP16 at `mmap_weights` time is what eats the wall clock before the
benchmark starts. The dominant single contributor is **`dequant_row_orth`
for the orth-embed token table**: 262144 rows × (K=1536 fp16 matvec + tmp
materialization). Order ~600B fp16 ops, single-threaded.

Two issues in `kernel_tqn.cpp::dequant_row_orth`:
- `std::vector<float> tmp(K)` per row → 262144 mallocs.
- The fp32→fp16 conversion of `tmp[j]` lives *inside* the matvec inner loop
  (re-converted K times per output element).

`cactus_tqn_dequant_layer` and `cactus_tq2_dequant_layer` also iterate rows
serially.

## What to do next

### Priority 1 — Make load fast

In `kernel_tqn.cpp`:

1. Replace the per-row `std::vector<float> tmp(K)` in `dequant_row_orth` with
   a `thread_local std::vector<float>` (or `thread_local __fp16` buffer if we
   first cast tmp to fp16 once per row).
2. Convert `tmp` from fp32 → fp16 *once per row* before the matvec, not once
   per output column.
3. Parallelize the row loops in `cactus_tqn_dequant_layer` and
   `cactus_tq2_dequant_layer` across cores. Apple Silicon has 4 P-cores;
   simple `std::thread`-pool fan-out should give ~6–8× wall-clock speedup
   for the orth-embed pass.

Even better — Priority 1.5 — **don't materialize the orth-embed at all**.
The token embedding is used in two places: (a) embedding lookup at input
(per-token row dequant — already cheap), and (b) tied LM head matmul (full
gemm against the vocab table at every decode step). Skipping pre-dequant
requires a fused TQ3-orth gemv for the LM head, which is the same shape of
work as the TQ4-hadamard gemv discussed above — both can come out of a
single kernel campaign.

### Priority 2 — Make decode fast (fused gemv done right)

Re-enable the dispatch block in `compute_matmul_node` (it's reverted but
trivial to re-add — see the `if (M == 1 && rhs_buffer.tqn != nullptr ...)`
hunk in this turn's diff history). Then rewrite `cactus_gemv_tq4_hadamard_f16`
addressing all five issues listed above. Concrete shape:

- `thread_local` scratch buffer for `x_scaled` and per-row `y_grp`.
- Precompute `recip_input_scale` once on first call, cached on a side
  `std::unique_ptr<__fp16[]>` hung off the BufferDesc (or off a per-CactusTQN
  cache structure).
- Process **4 output rows at a time**: dequant all 4 row groups for the same
  group index `g`, accumulate against the same loaded `x_scaled[g]` slice.
  Each row's hadamard chain is independent — same FWHT routine, just 4×.
- Replace the codebook loop with two `vqtbl1q_u8` ops over the 32-byte fp16
  codebook (low-byte LUT + high-byte LUT, then interleave). Or exploit
  codebook symmetry: 8-magnitude fp16 LUT (16 bytes, fits a single vqtbl1) +
  sign from index bit 3 via `vbicq_u16` / `veorq_u16`.
- Replace `inv_permutation[k]` gather with `vqtbl1q_u8` over a 128-byte
  inverse-permutation table (precomputed at load).

Match or beat INT4 decode speed. The packed weight bytes per element are the
same; the only extra work is FWHT (~7 ops/elt) and the perm + signs (~3
ops/elt) — well within memory-bound headroom.

### Priority 3 — Memory

Once the fused gemv covers both decode-gemv and prefill-gemm, the
`dequant_to_fp16` materialization can be deleted entirely. Expected RAM
drop: 5.4 GB → ~1.5 GB for `tqh_u4`.

### Priority 4 — Other variants

- `p3` (mixed 3-bit + 4-bit): per-layer dispatch already works in the Python
  packer. The fused TQ3-hadamard gemv hasn't been written; if we want fast
  decode for `p3`, we need it. Same code shape as TQ4 but with the 3-bit
  unpack (already present in `codebook_lookup_group`).
- `p6` (mixed 1/2/3/4): adds TQ1 and TQ2 to the fused-gemv list. TQ1's win
  is biggest because the codebook collapses to two values (`±a`); the gemv
  becomes a sign-flip + accumulate, no LUT needed.
- `tqh_u3`: same as `tqh_u4` but with the 3-bit fused gemv.

### Priority 5 — Hygiene

- The `BufferDesc.tqn` plumbing is in place but no other code uses it yet. If
  the fused-gemv plan is dropped, that field can come out (and the load-time
  capture in `mmap_weights`).
- `kernel_tq2.cpp`'s 128-byte header is silently compatible with my packer's
  136-byte header because the kernel only reads up to offset 120 + uses
  `total != blob_size` to validate. Worth adding a comment in `kernel_tq2.cpp`
  noting that the 8 trailing bytes are an extension used by TQ3/TQ4.
- The orth-embed dequant (`dequant_row_orth`) lives in `kernel_tqn.cpp` and
  is currently only invoked via `cactus_tqn_dequant_row` / `_layer`. If
  Priority 1.5 happens, it'll grow a fused-gemv variant and the materializing
  variant can keep its current shape for verification only.

## Files touched / added

```
M cactus/graph/graph.h                # Precision::TQ1/3/4, CactusTQN, BufferDesc::tqn, MappedFile::tqn()/dequant_to_fp16()
M cactus/graph/graph_io.cpp           # parse_header dispatch, mmap_weights pre-dequant, BufferDesc.tqn capture
M cactus/kernel/kernel.h              # CactusTQN, signatures
A cactus/kernel/kernel_tqn.cpp        # TQ1/3/4 load + dequant + orth dequant + cactus_tq2_dequant_layer + (unused) fused gemv
M tests/test_gemma4_suite.cpp         # argv name filter

A python/src/tqh_pack.py              # packer
A python/src/tqh_pack_verify.py       # round-trip verifier
```

## Reproduction

Assemble:
```
python -m src.tqh_pack \
  --baseline-dir packed_shipping/bf16_baseline \
  --pli-embed-dir packed_shipping/pli_embed \
  --transformer-dir packed_shipping/tqh_u4 \
  --output-dir weights/gemma-4-e2b-it-tqh-u4
```

Verify a single layer numerically:
```
python -m src.tqh_pack_verify --packed-dir packed_shipping/p3 \
  --layer model.language_model.layers.2.mlp.gate_proj
```

Run 1K-context test only:
```
CACTUS_TEST_GEMMA4_MODEL=/Users/karen/cactus/weights/gemma-4-e2b-it-tqh-u4 \
  /Users/karen/cactus/tests/build/test_gemma4_suite 1k_context
```

Build:
```
cmake --build /Users/karen/cactus/cactus/build -j 8
cmake --build /Users/karen/cactus/tests/build --target test_gemma4_suite -j 8
```
