# ACLE SME / SME2 Intrinsics — Authoritative Signature Reference

> Goal: EXACT intrinsic names/signatures + attribute semantics for writing an SME2 quantized
> matmul in C++ with **Apple clang 17** (`-march=armv9-a+sme2`, `#include <arm_sme.h>`).
> Every signature below is verified VERBATIM against LLVM's own clang CodeGen test suite
> (the ground truth the compiler must accept) and the Arm "Multiplying matrices with SME2"
> learning path. Where a source summary disagreed with the LLVM test, the **LLVM test wins**.

Date-checked: 2026-06-09. Primary verification source: `llvm-project/clang/test/CodeGen/AArch64/`.

---

## 0. TL;DR — the load-bearing facts that break builds

1. **Tile index is the FIRST argument** of `svmopa_za32_*_m` and must be a compile-time immediate
   `0..3` for za32 (i32). Signature: `svmopa_za32_*_m(uint64_t tile, svbool_t pn, svbool_t pm, zn, zm)`.
   Verified: `acle_sme_mopa-za32.c` → `SME_ACLE_FUNC(svmopa_za32, _s8, _m)(0, pn, pm, zn, zm)`.
2. **There is NO `svmopa_za32_s16_m` / `svmopa_za32_u16_m`.** 16-bit-integer MOPA widens to **za64**
   (`svmopa_za64_s16_m`, needs `FEAT_SME_I16I64`). za32 integer MOPA only comes from **8-bit**
   (s8/u8, 4-way widening). Do not write `..._za32_s16_m` — it will not compile.
3. **ZA store readout: `svst1_hor_za32(tile, slice, pg, ptr)`** — tile first, slice second, predicate
   third, pointer fourth. Verified: `acle_sme_st1.c` → `svst1_hor_za32(0, slice_base, pg, ptr)`.
   (Note: some web summaries put `pg` before tile — that is WRONG for the C ACLE intrinsic.)
4. `svzero_za()` takes **no args**; `svzero_mask_za(uint64_t mask)` takes an 8-bit tile mask.
5. Inside `__arm_streaming`, vector length = **streaming VL (SVL)**. Use `svcntsw()` for "FP32 lanes
   in streaming mode". NEON/AdvSIMD intrinsics are **invalid** in streaming mode.
6. The f16→za32 and f32 forms share the `mopa.wide`/`mopa` LLVM intrinsic and pass a **mode immediate
   `i32 1`** internally (sub=0 for the accumulate `_m` form via the second immediate); you still just
   call `svmopa_za32_f32_m(tile, pn, pm, zn, zm)`.

---

## 1. Outer-product accumulate intrinsics (MOPA → ZA accumulators)

Header: `<arm_sme.h>`. All require `__arm_streaming` (or streaming-compatible context) and
`__arm_inout("za")` / `__arm_out("za")` on the enclosing function. Tile arg is a compile-time
immediate. `pn` selects active **rows** (n / M-dim) and `pm` selects active **columns** (m / N-dim).

### za32 accumulator forms (4 tiles: za0.s .. za3.s, tile ∈ {0,1,2,3})

| Intrinsic | Full signature | Instruction | LLVM intrinsic | Notes |
|---|---|---|---|---|
| `svmopa_za32_f32_m` | `void svmopa_za32_f32_m(uint64_t tile, svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm)` | **FMOPA** (non-widening) | `llvm.aarch64.sme.mopa.nxv4f32` | 1×1, FP32→FP32 |
| `svmopa_za32_f16_m` | `void svmopa_za32_f16_m(uint64_t tile, svbool_t pn, svbool_t pm, svfloat16_t zn, svfloat16_t zm)` | **FMOPA (widening)** | `llvm.aarch64.sme.mopa.wide.nxv8f16` | 2-way widen FP16→FP32. Base `FEAT_SME`. |
| `svmopa_za32_bf16_m` | `void svmopa_za32_bf16_m(uint64_t tile, svbool_t pn, svbool_t pm, svbfloat16_t zn, svbfloat16_t zm)` | **BFMOPA (widening)** | `llvm.aarch64.sme.mopa.wide.nxv8bf16` | 2-way widen BF16→FP32 |
| `svmopa_za32_s8_m` | `void svmopa_za32_s8_m(uint64_t tile, svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm)` | **SMOPA (widening)** | `llvm.aarch64.sme.smopa.wide.nxv16i8` | **4-way** widen, S8×S8→S32. THE int8 path. |
| `svmopa_za32_u8_m` | `void svmopa_za32_u8_m(uint64_t tile, svbool_t pn, svbool_t pm, svuint8_t zn, svuint8_t zm)` | **UMOPA (widening)** | `llvm.aarch64.sme.umopa.wide.nxv16i8` | 4-way widen, U8×U8→U32 |

Mixed-sign 8-bit (SME2, `<arm_sme.h>`, `FEAT_SME2`): `svsumopa_za32_s8_m` (S8×U8 → SUMOPA,
`llvm.aarch64.sme.sumopa.wide.nxv16i8`) and `svusmopa_za32_u8_m` (U8×S8 → USMOPA). Same arg shape.

### za64 accumulator forms (16-bit integers; needs `FEAT_SME_I16I64`, tile ∈ {0..7})

> Use these if you ever want **S16×S16→S32-in-za64** instead of S8. They are NOT za32.

| Intrinsic | Full signature | Instruction | LLVM intrinsic |
|---|---|---|---|
| `svmopa_za64_s16_m` | `void svmopa_za64_s16_m(uint64_t tile, svbool_t pn, svbool_t pm, svint16_t zn, svint16_t zm)` | **SMOPA (widening)** | `llvm.aarch64.sme.smopa.wide.nxv8i16` |
| `svmopa_za64_u16_m` | `void svmopa_za64_u16_m(uint64_t tile, svbool_t pn, svbool_t pm, svuint16_t zn, svuint16_t zm)` | **UMOPA (widening)** | `llvm.aarch64.sme.umopa.wide.nxv8i16` |
| `svsumopa_za64_s16_m` | `void svsumopa_za64_s16_m(uint64_t tile, svbool_t pn, svbool_t pm, svint16_t zn, svuint16_t zm)` | **SUMOPA** | `llvm.aarch64.sme.sumopa.wide.nxv8i16` |
| `svusmopa_za64_u16_m` | `void svusmopa_za64_u16_m(uint64_t tile, svbool_t pn, svbool_t pm, svuint16_t zn, svint16_t zm)` | **USMOPA** | `llvm.aarch64.sme.usmopa.wide.nxv8i16` |

Verbatim evidence (LLVM clang test, `acle_sme_mopa-za32.c` / `-za64.c`):
```c
void test_svmopa_za32_f32(svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm)
    __arm_streaming __arm_inout("za") {
  svmopa_za32_f32_m(1, pn, pm, zn, zm);   // -> llvm.aarch64.sme.mopa.nxv4f32(i32 1, ...)
}
void test_svmopa_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm)
    __arm_streaming __arm_inout("za") {
  svmopa_za32_s8_m(0, pn, pm, zn, zm);    // -> llvm.aarch64.sme.smopa.wide.nxv16i8(i32 0, ...)
}
void test_svmopa_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svint16_t zm)
    __arm_streaming __arm_inout("za") {
  svmopa_za64_s16_m(7, pn, pm, zn, zm);   // -> llvm.aarch64.sme.smopa.wide.nxv8i16(i32 7, ...)
}
```

**Overloaded short form** (preferred in the Arm tutorial): `svmopa_za32_m(tile, pn, pm, zn, zm)` —
clang resolves the type suffix from the operand types. The Arm learning path uses
`svmopa_za32_m(0, pMDim, pNDim, zL, zR)`. Both the suffixed and overloaded forms exist; the suffixed
form is safest when the operand type is ambiguous.

### Semantics
- za32 MOPA computes, for each cell `(i,j)` of the SVL/4 × SVL/4 (for 8-bit: lanes/4) tile:
  `ZA32[i][j] += sum over the widening group of  zn[i-lane group] * zm[j-lane group]`.
- For **f32 (non-widening)**: `ZA[i][j] += zn[i] * zm[j]`, a pure rank-1 update; tile is SVL_words ×
  SVL_words (16×16 on a 512-bit SVL / M4).
- For **s8/u8 (4-way widening)**: each 32-bit ZA cell accumulates **4** S8×S8 products from 4
  contiguous K elements packed in each 32-bit lane group. K is consumed in steps of 4 per MOPA.
- `pn`/`pm` are **byte/element predicates matching the operand element width** (b8 for s8/u8 ops,
  b16 for f16/bf16, b32 for f32). Inactive lanes contribute 0 to the accumulate.

---

## 2. ZA zeroing

| Intrinsic | Signature | Instruction | Notes |
|---|---|---|---|
| `svzero_za` | `void svzero_za(void)` | `ZERO {ZA}` | Zeroes the entire ZA array. No args. |
| `svzero_mask_za` | `void svzero_mask_za(uint64_t mask)` | `ZERO { <mask> }` | 8-bit tile-slice mask; `0xFF`(255) == all, `0`==none. |

Both lower to `llvm.aarch64.sme.zero(i32 mask)`; `svzero_za()` is `zero(i32 255)`. Caller must be
`__arm_out("za")` (it defines ZA) or `__arm_inout("za")`.
Verbatim: `acle_sme_zero.c` → `void test_svzero_za(void) __arm_out("za") { svzero_za(); }` and
`svzero_mask_za(0); svzero_mask_za(176); svzero_mask_za(255);`.

---

## 3. ZA read-out to memory (the actual store of accumulated results)

> Two families: **(A)** `svst1_{hor,ver}_za*` store one tile slice straight to memory (what the Arm
> matmul tutorial uses); **(B)** `svread_{hor,ver}_za*_m` extract a slice into a Z register first,
> then you `svst1` it as a normal SVE vector; **(C)** `svstr_za` stores the whole ZA (or a vnum slice)
> opaquely for spill/restore — NOT a per-tile readout.

### (A) Direct slice store — USE THIS for matmul output

```c
// VERBATIM signature (acle_sme_st1.c):
void svst1_hor_za32(uint64_t tile, uint32_t slice, svbool_t pg, void *ptr);
void svst1_ver_za32(uint64_t tile, uint32_t slice, svbool_t pg, void *ptr);
void svst1_hor_za128(uint64_t tile, uint32_t slice, svbool_t pg, void *ptr);
void svst1_ver_za128(uint64_t tile, uint32_t slice, svbool_t pg, void *ptr);
// (also za8 / za16 / za64 variants with identical arg shape)
```
- **Argument order: `(tile, slice, pg, ptr)`** — tile is a compile-time immediate, `slice` (a.k.a.
  `slice_base`) is a runtime `uint32_t` row/column index within the tile.
- `_hor` = **horizontal slice** = one **row** of the tile written to contiguous memory.
  `_ver` = **vertical slice** = one **column** of the tile written to contiguous memory.
- The `slice` index selects which row (hor) / column (ver) of the tile, modulo the number of slices
  (= SVL/elem-size). `za32` has SVL/4 horizontal and SVL/4 vertical slices.
- Lowers to `llvm.aarch64.sme.st1w.horiz.p0(<vscale x 4 x i1> pg, ptr, i32 tile, i32 slice)`
  (note: at IR level the order is pg, ptr, tile, slice — but the **C intrinsic order is tile, slice,
  pg, ptr**, which is what you write).
- Enclosing function must be `__arm_in("za")` (reads ZA) or `__arm_inout("za")`.

Verbatim: `acle_sme_st1.c` →
```c
void test_svst1_hor_za32(uint32_t slice_base, svbool_t pg, void *ptr)
    __arm_streaming __arm_in("za") {
  svst1_hor_za32(0, slice_base, pg, ptr);
  svst1_hor_za32(3, slice_base + 3, pg, ptr);   // -> st1w.horiz.p0(pg, ptr, i32 3, i32 slice+3)
}
```
And the Arm tutorial's readout loop (FP32, 4 rows at a time):
```c
svst1_hor_za32(0, trow + 0, p0, &matResult[result_UL + (trow + 0) * N]);
svst1_hor_za32(0, trow + 1, p1, &matResult[result_UL + (trow + 1) * N]);
// ... trow+2, trow+3
```

### (B) Slice → Z register, then store (more flexible / lets you post-scale before store)
```c
// VERBATIM (acle_sme_read.c), overloaded as svread_hor_za32_m / suffixed _s32/_f32:
svint32_t   svread_hor_za32_s32_m(svint32_t inactive,   svbool_t pg, uint64_t tile, uint32_t slice);
svfloat32_t svread_hor_za32_f32_m(svfloat32_t inactive, svbool_t pg, uint64_t tile, uint32_t slice);
//  + _u32, + svread_ver_za32_* (vertical). 
//  arg order: (inactive, pg, tile, slice)
```
Lowers to `llvm.aarch64.sme.read.horiz.nxv4i32(inactive, pg, i32 tile, i32 slice)`. After reading,
post-process in Z regs (e.g. requant scale for INT8 matmul) and store with normal `svst1`.

### (C) Whole-ZA spill (NOT a per-tile readout)
```c
void svstr_za(uint32_t slice, void *ptr);   // -> llvm.aarch64.sme.str.p0(i32 slice, ptr, i32 0)
void svldr_za(uint32_t slice, const void *ptr);
```
For saving/restoring ZA state opaquely; do not use for matmul result extraction.

**Recommendation for INT8 matmul:** use **(A) `svst1_hor_za32`** if you store raw int32 accumulators
and rescale afterward in non-streaming code, OR **(B) `svread_hor_za32_s32_m`** if you want to convert
to float / apply per-output-channel scale inside the streaming kernel before storing.

---

## 4. Tile/slice helpers, counting, predicates

### Counting (element counts of the CURRENT vector length)
| Intrinsic | Returns | Notes |
|---|---|---|
| `uint64_t svcntb(void)` | bytes in a vector | In streaming fn = SVL bytes. |
| `uint64_t svcnth(void)` | 16-bit elems (halfwords) | |
| `uint64_t svcntw(void)` | 32-bit elems (words) | FP32 lanes. |
| `uint64_t svcntd(void)` | 64-bit elems (doublewords) | |

### Streaming-VL counts (SME-specific; return the STREAMING VL regardless of current PSTATE.SM)
| Intrinsic | Returns | Notes |
|---|---|---|
| `uint64_t svcntsb(void)` | streaming VL in bytes | |
| `uint64_t svcntsh(void)` | streaming VL in halfwords | |
| `uint64_t svcntsw(void)` | streaming VL in words | **FP32 lanes / tile side for f32.** Arm tutorial uses this. |
| `uint64_t svcntsd(void)` | streaming VL in doublewords | |

> Inside `__arm_streaming`, `svcntw() == svcntsw()`. Use `svcntsw()` when you need the SVL from a
> NON-streaming caller (e.g. to compute tiling before entering the kernel). On Apple M4 SVL = 512 bit
> ⇒ `svcntsw() == 16` (16 FP32 lanes, 16×16 f32 tile; for s8: 64-byte vec, SVL/4 = 16 → 16×16 s32 tile
> accumulating 4 K per step).

### Governing predicates (all-true)
```c
svbool_t svptrue_b8(void);    // ptrue p.b, ALL   — for s8/u8 ops
svbool_t svptrue_b16(void);   // ptrue p.h, ALL   — for f16/bf16/s16 ops
svbool_t svptrue_b32(void);   // ptrue p.s, ALL   — for f32 ops
svbool_t svptrue_b64(void);   // ptrue p.d, ALL
```
`svptrue_b8()` == `svptrue_pat_b8(SV_ALL)`.

### Loop/tail predicates
```c
svbool_t svwhilelt_b8(uint64_t op1,  uint64_t op2);   // WHILELT p.b
svbool_t svwhilelt_b16(uint64_t op1, uint64_t op2);   // WHILELT p.h
svbool_t svwhilelt_b32(uint64_t op1, uint64_t op2);   // WHILELT p.s  — used for M/N tail in tutorial
svbool_t svwhilelt_b64(uint64_t op1, uint64_t op2);   // WHILELT p.d
// element n active iff (op1 + n) < op2.  Also _u32/_u64/_s32/_s64 typed and svwhilele_* variants.
```
(`svwhilelt_b32_u64`, `svwhilelt_b32_s64`, etc. are the fully-typed names; the bare `svwhilelt_b32`
is the overload.)

---

## 5. Streaming-mode loads/stores (usable inside `__arm_streaming`)

Inside streaming mode you use **SVE** contiguous load/store intrinsics (NOT NEON). They operate on
SVL-wide Z registers and take a governing predicate.

```c
// Loads (overloaded form resolves type from the typed pointer):
svfloat32_t svld1_f32(svbool_t pg, const float32_t *ptr);
svint8_t    svld1_s8 (svbool_t pg, const int8_t   *ptr);
svuint8_t   svld1_u8 (svbool_t pg, const uint8_t  *ptr);
svfloat16_t svld1_f16(svbool_t pg, const float16_t *ptr);
svbfloat16_t svld1_bf16(svbool_t pg, const bfloat16_t *ptr);
// Overloaded: svld1(pg, ptr)  -> picks suffix from ptr type. (Arm tutorial uses svld1(pMDim, &..).)

// Stores:
void svst1_f32(svbool_t pg, float32_t *ptr, svfloat32_t data);
void svst1_s8 (svbool_t pg, int8_t   *ptr, svint8_t data);
// Overloaded: svst1(pg, ptr, data).

// Other useful in-streaming SVE ops: svdup_f32/_s8 (broadcast), svsel, svadd, svmla,
// svcvt_f32_* (widen/convert), svreinterpret_*.
```
- These are normal SVE intrinsics from `<arm_sve.h>` (pulled in by `<arm_sme.h>`); they are legal in
  both streaming and non-streaming mode (they are streaming-compatible) — that is why you stage data
  into Z registers with `svld1` and then feed MOPA.
- The "Hello SME!" benchmark note: stage memory → Z via `svld1` then MOPA; direct memory→ZA loads
  (`svld1_hor_za*` / `ldr za`) are ~2.6× slower on M4.

ZA-direct loads (slice loads, the inverse of §3A) also exist if you must:
```c
void svld1_hor_za32(uint64_t tile, uint32_t slice, svbool_t pg, const void *ptr);
void svld1_ver_za32(uint64_t tile, uint32_t slice, svbool_t pg, const void *ptr);
// (tile, slice, pg, ptr) — same shape as svst1_hor_za32; enclosing fn __arm_out("za").
```

---

## 6. Function attributes (Apple clang 17 keyword spelling)

Use the **keyword form** (`__arm_streaming`, etc.) on clang 17 — the older
`__attribute__((arm_streaming))` GNU form is deprecated/inconsistent. Keywords go in the function
**type** position (after the parameter list for member/free functions), e.g.
`void f(...) __arm_streaming __arm_inout("za") { ... }`.

| Attribute | Spelling | LLVM IR attr | Effect |
|---|---|---|---|
| Streaming body | `__arm_streaming` | `aarch64_pstate_sm_enabled` | Function runs with **PSTATE.SM=1**. The **caller** emits `smstart sm` before the call and `smstop sm` after. Part of the function's type/ABI — callers must know. Inside, SVL applies; NEON illegal. |
| Streaming-agnostic | `__arm_streaming_compatible` | `aarch64_pstate_sm_compatible` | Runs correctly with SM=0 OR 1; does not change SM. Conditionally toggles based on caller. Restrict to instructions legal in both modes. |
| Locally streaming | `__arm_locally_streaming` | `aarch64_pstate_sm_body` | **Definition-only** (not ABI). Compiler inserts `smstart sm` in prologue / `smstop sm` in epilogue; callers see a normal (non-streaming) function. Handy for a self-contained kernel. |
| Owns ZA | `__arm_new("za")` (also `__arm_new_za`) | `aarch64_new_za` | Function gets **fresh, zeroed ZA**; compiler emits `smstart za` on entry / `smstop za` on exit and handles lazy-save. Use for the top-level kernel that owns the accumulators. |
| Reads ZA in | `__arm_in("za")` | `aarch64_in_za` | ZA is a live input; preserved unless function also writes. Use on a slice-store helper that consumes accumulators. |
| Writes ZA out | `__arm_out("za")` | `aarch64_out_za` | ZA defined on return; incoming value undefined. Use on a helper that zeroes/loads ZA. |
| Read+write ZA | `__arm_inout("za")` | `aarch64_inout_za` | ZA is live in and out. Use on the MOPA-accumulate inner kernel. |
| Preserves ZA | `__arm_preserves("za")` | `aarch64_preserves_za` | Promises ZA unchanged across the call (lets compiler avoid save/restore). |

### Critical codegen consequences (LLVM AArch64SME doc, "Changing PSTATE.SM")
> "When changing PSTATE.SM the execution of FP/vector operations may be transferred to another
> processing element. This has three important implications:
> 1. The runtime SVE vector length may change.
> 2. The contents of FP/AdvSIMD/SVE registers are zeroed.
> 3. The set of allowable instructions changes."

Practical rules:
- **Crossing a streaming boundary zeroes Z/P registers** and is not free. Do all the MOPA work for a
  whole N-panel inside ONE streaming region; never enter/exit per column.
- **NEON/AdvSIMD intrinsics are invalid in streaming mode.** Keep Hadamard transform, INT8 quant,
  codebook handling, and final rescale in the NON-streaming NEON caller. The streaming kernel does
  ONLY the MOPA accumulate (and optionally the readout/rescale via `svread` + SVE ops).
- `smstart za` / `smstop za` (from `__arm_new("za")`) is distinct from `smstart sm`/`smstop sm` (from
  `__arm_streaming`). A kernel that both owns ZA and runs streaming gets both:
  `__arm_new("za") __arm_locally_streaming` (or `__arm_streaming` if you want it in the type/ABI).
- Do **not** use SME in signal handlers (corrupts the interrupted thread's ZA). macOS xnu
  auto-saves/restores ZA across context switches (no entitlement needed).

---

## 7. Minimal code snippets

### 7.1 Correct FP32 FMOPA tile loop (mirrors the Arm learning path)
```cpp
#include <arm_sme.h>

// One SVL×SVL FP32 tile of C = A_panel * B_panel, accumulated over K.
// A_mod is pre-transposed so each k gives a contiguous SVL-long column of A (lanes = M rows).
// B is row-major (each k gives a contiguous SVL-long row of B, lanes = N cols).
__arm_new("za") __arm_locally_streaming
void matmul_f32_tile(uint64_t M, uint64_t K, uint64_t N,
                     const float *restrict A_mod,   // [M*K], column-staged per tutorial
                     const float *restrict B,       // [K*N], row-major
                     float *restrict C)             // [M*N], row-major
{
    const uint64_t SVL = svcntsw();                 // FP32 lanes in streaming mode (16 on M4)
    for (uint64_t row = 0; row < M; row += SVL) {
        svbool_t pM = svwhilelt_b32(row, M);        // active M rows in this tile
        for (uint64_t col = 0; col < N; col += SVL) {
            svbool_t pN = svwhilelt_b32(col, N);    // active N cols in this tile
            svzero_za();                            // clear all ZA tiles
            const uint64_t aPos = row * K;
            for (uint64_t k = 0; k < K; ++k) {
                svfloat32_t zA = svld1_f32(pM, &A_mod[aPos + k * SVL]);
                svfloat32_t zB = svld1_f32(pN, &B[col + k * N]);
                svmopa_za32_f32_m(0, pM, pN, zA, zB);   // FMOPA: ZA0.s += zA (outer) zB
            }
            // Read out tile 0 row-by-row (horizontal slices) to C.
            const uint64_t cUL = row * N + col;
            for (uint64_t r = 0; r < SVL && row + r < M; ++r) {
                svst1_hor_za32(0, (uint32_t)r, pN, &C[cUL + r * N]);
            }
        }
    }
}
```

### 7.2 INT8 SMOPA tile loop (S8×S8 → S32, 4-way widening)
```cpp
#include <arm_sme.h>

// One SVL/4 (=16) × SVL/4 tile of int32 accumulators = A8 * B8, K consumed 4-at-a-time.
// Layouts must pack 4 contiguous K into each 32-bit lane group of zA/zB (the SMOPA widening group).
__arm_new("za") __arm_locally_streaming
void matmul_s8_tile(uint64_t M, uint64_t K, uint64_t N,
                    const int8_t *restrict A8,   // pre-tiled: SVL bytes per k-step (4 K interleaved)
                    const int8_t *restrict B8,   // pre-tiled likewise
                    int32_t *restrict C32)       // [M*N] raw int32 accumulators
{
    const uint64_t SVLb = svcntsb();             // SVL in bytes (64 on M4)
    const uint64_t TILE = SVLb / 4;              // 16: int32 cells per tile side
    const svbool_t pAll8 = svptrue_b8();         // byte predicate for s8 operands
    for (uint64_t row = 0; row < M; row += TILE) {
        svbool_t pM = svwhilelt_b32(row, M);     // 32-bit accumulator-cell predicate (rows)
        for (uint64_t col = 0; col < N; col += TILE) {
            svbool_t pN = svwhilelt_b32(col, N); // (cols)
            svzero_za();
            // K is iterated in steps of 4 (each MOPA consumes 4 K via 4-way widening).
            for (uint64_t kg = 0; kg < K; kg += 4) {
                // Each load brings SVL bytes = TILE rows/cols × 4 contiguous K.
                svint8_t zA = svld1_s8(pAll8, &A8[(row * K) + kg * TILE]);
                svint8_t zB = svld1_s8(pAll8, &B8[(col * K) + kg * TILE]);
                svmopa_za32_s8_m(0, pM, pN, zA, zB);   // SMOPA: ZA0.s += sum_{c<4} A*B
            }
            // Readout: extract int32 slice to Z, (optionally) requant-scale, then store.
            const uint64_t cUL = row * N + col;
            for (uint64_t r = 0; r < TILE && row + r < M; ++r) {
                // Option A: store raw int32 accumulators, rescale in non-streaming caller.
                svst1_hor_za32(0, (uint32_t)r, pN, &C32[cUL + r * N]);
                // Option B (rescale in-kernel):
                //   svint32_t acc = svread_hor_za32_s32_m(svdup_s32(0), pN, 0, (uint32_t)r);
                //   svfloat32_t f = svmul_f32_x(pN, svcvt_f32_s32_x(pN, acc), zScale);
                //   svst1_f32(pN, &Cf[cUL + r*N], f);
            }
        }
    }
}
```
> NOTE: the exact byte-packing of zA/zB (which K maps to which lane group) must match the SMOPA
> widening contract — verify against KleidiAI `..._sme2_mopa` and scalable-analyses/sme INT8 kernel
> before trusting the index math above. The intrinsic names/signatures are correct; the *tiling
> arithmetic* in the snippet is illustrative and unverified on hardware.

### 7.3 ZA-owning wrapper that splits streaming vs non-streaming work
```cpp
// Top-level entry: non-streaming, does NEON pre/post; calls a __arm_streaming inner that touches ZA.
void cactus_quant_matmul_sme2(/* CactusQuantMatrix*, fp16* A, M, fp16* C */) {
    // ... NEON: Hadamard transform, INT8 quantize activations + codebook (NON-streaming) ...
    matmul_s8_tile(/* ... */);   // streaming + __arm_new("za") kernel: MOPA only
    // ... NEON: rescale int32 -> fp16 by norm_f32, write C (NON-streaming) ...
}
```

---

## 8. Build / toolchain notes (Apple clang 17)

- Header: `#include <arm_sme.h>` (transitively includes `<arm_sve.h>`).
  Guard real code with `#if defined(__ARM_FEATURE_SME2)`.
- Compile the SME TU with `-march=armv9-a+sme2` (probe-confirmed to compile on Apple clang 17 and
  emit `smstart`/`*mopa`). Keep this OFF the library-wide flag; confine SME to its own translation
  unit (`src/matmul_sme2.cpp`) so the optimizer never emits streaming code into NEON paths.
- Feature-test macros you can rely on inside the TU: `__ARM_FEATURE_SME`, `__ARM_FEATURE_SME2`,
  `__ARM_FEATURE_SME_F16F16` (for za16 f16 MOPA — note this is `0` on M4, so use the za32 widening
  f16 form `svmopa_za32_f16_m`), `__ARM_FEATURE_SME_I16I64` (for za64 s16 MOPA).
- Runtime detect on Apple: sysctl `hw.optional.arm.FEAT_SME` / `hw.optional.arm.FEAT_SME2`.

---

## 9. Sources (verified 2026-06-09)

1. **Arm ACLE spec** — https://arm-software.github.io/acle/main/acle.html — §"BFMOPA, FMOPA
   (widening), SMOPA, UMOPA" and §"SME language extensions and intrinsics". Defines the intrinsic
   family and attribute semantics. (Full prototype tables are large; cross-checked against LLVM.)
2. **LLVM clang CodeGen tests** (ground truth the compiler accepts) — verbatim signatures:
   - `clang/test/CodeGen/AArch64/sme-intrinsics/acle_sme_mopa-za32.c` — f32/f16/bf16/s8/u8 MOPA.
   - `.../acle_sme_mopa-za64.c` — s16/u16/f64 + su/us MOPA (za64).
   - `.../acle_sme_zero.c` — svzero_za / svzero_mask_za.
   - `.../acle_sme_st1.c` — svst1_{hor,ver}_za32/za128 arg order (tile, slice, pg, ptr).
   - `.../acle_sme_read.c` — svread_{hor,ver}_za32_{s32,f32}_m (inactive, pg, tile, slice).
   - `.../acle_sme_str.c` — svstr_za(slice, ptr).
   - `.../acle_sme_ld1.c` — svld1_{hor,ver}_za32 (tile, slice, pg, ptr).
3. **LLVM AArch64SME doc** — https://llvm.org/docs/AArch64SME.html — attribute→IR mapping
   (`aarch64_pstate_sm_enabled` etc.), PSTATE.SM change consequences, streaming restrictions.
4. **Arm Learning Path "Multiplying matrices with SME2"** —
   https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/7-sme2-matmul-intr/
   — canonical FP32 kernel: `svcntsw`, `svwhilelt_b32`, `svzero_za`, `svld1`, `svmopa_za32_m`,
   `svst1_hor_za32(0, slice, pg, ptr)`, attributes `__arm_new("za")` + `__arm_locally_streaming` (top)
   and `__arm_streaming __arm_inout("za")` (inner).
5. **"Hello SME!"** — https://arxiv.org/abs/2409.18779 — M4 microbenchmarks; Z-reg→ZA staging via
   svld1 + MOPA is ~2.6× faster than direct ZA loads.
6. KleidiAI (`..._sme2_mopa` micro-kernels) and scalable-analyses/sme — for the EXACT int8 byte-pack
   tiling contract (snippet 7.2 arithmetic must be validated against these).
