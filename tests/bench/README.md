# Cactus v2 Benchmark Suite

Compares `cactus` against `llama.cpp` (ggml), LiteRT (both Ruy and the TFLite
NEON kernels), ONNX Runtime, and ExecuTorch (XNNPACK) on the four kernels we
currently care about on this branch:

- **GEMV** — CQ4 / INT4 Cactus Quant and backend-specific quantized variants
- **GEMM** — CQ4 / INT4 Cactus Quant and backend-specific quantized variants
- **Attention prefill** — FP16 fused attention
- **Hybrid attention decode** — INT8 KV-cache, FP16 query

The driver sweeps the five graphs the team needs:

| Graph | Sweep over | Fixed |
|-------|------------|-------|
| 1. `gemv_d`            | d ∈ {128…4096}, K=N=d, M=1 | – |
| 2. `gemm_d`            | d ∈ {128…4096}, K=d        | M=N=512 |
| 3. `gemm_mn`           | M=N ∈ {128…4096}           | K=512 |
| 4. `attn_prefill_s`    | S ∈ {128…4096}             | model_dim=1024, h=8 |
| 5. `attn_decode_cache` | cache ∈ {128…4096}         | model_dim=1024, h=8 |

## Quick start (cactus only — no third-party deps)

```bash
# 1. Build the cactus library once.
cd cactus-v2/cactus && ./build.sh

# 2. Configure + build the bench.
cd ../tests
mkdir -p build && cd build
cmake ..
make -j matmul_bench attn_bench

# 3. Run.
./matmul_bench --csv matmul.csv
./attn_bench   --csv attn.csv
```

Or use the wrapper script (it builds libcactus + the bench, then runs both):

```bash
./tests/run_benchmark.sh
```

## Enabling third-party backends

Each backend is a CMake option. Drop the source/binary into `../third_party/`
and toggle the flag at configure time.

```bash
# llama.cpp / ggml
git clone https://github.com/ggml-org/ggml.git ../../third_party/ggml
cmake .. -DWITH_GGML=ON

# LiteRT (Ruy + neon)
git clone https://github.com/google-ai-edge/LiteRT.git ../../third_party/litert
cmake .. -DWITH_LITERT=ON

# ExecuTorch matmul (fetches XNNPACK at configure time)
cmake .. -DWITH_EXECUTORCH=ON

# ExecuTorch attention (custom_sdpa) — separate flag, requires a built ET tree.
git clone https://github.com/pytorch/executorch ../../third_party/executorch
cd ../../third_party/executorch
./install_executorch.sh                          # installs build deps
cmake -Bcmake-out -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
                  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
                  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
cmake --build cmake-out -j
cd ../../cactus-v2/tests/build
cmake .. -DWITH_EXECUTORCH_LLM=ON \
         -DEXECUTORCH_DIR=$(realpath ../../../third_party/executorch)

# ONNX Runtime (prebuilt; download from the GitHub releases page and extract)
#   ../../third_party/onnxruntime/{include,lib}/...
cmake .. -DWITH_ONNXRT=ON
```

`./tests/run_benchmark.sh --external-frameworks` auto-enables any tree it
finds under `../third_party/`.

## CLI options

```
./matmul_bench
  --warmup N          Warmup iterations (default: 50)
  --iterations N      Timed iterations  (default: 200)
  --graphs ...        Comma list of: gemv_d, gemm_d, gemm_mn
  --dims 128,256,...  Sweep dimensions (default: 128…4096)
  --backends LIST     Comma-separated framework names (cactus,ggml,litert,...)
  --threads N|max     Override thread count
  --csv path          Append CSV results to path
```

The number of distinct weight matrices the bench cycles through (`NM`) is
chosen automatically per config to make the weight pool exceed 64 MB —
larger than any plausible Apple Silicon SLC. This forces every iteration's
weight load to miss to RAM, matching real inference where every layer has
unique weights. NM is reported next to each shape in the output.

The plotter renders a monotonic latency envelope per backend (`cummax` over
the raw latency points). The CSVs remain raw per-call timings; the PNGs use
the envelope so the visible scaling shape does not dip when a larger shape
lands on a more efficient backend kernel regime.

### Threading

Default is to honor `--threads`. We recommend `--threads 4` for these
plots because:
- Most mobile target devices have 2–6 high-performance cores (iPhone
  A-series: 2P+4E, mid-range Snapdragon: 1+3+4, high-end Snapdragon: 1+5+2).
  4 P-core threads is a reasonable mobile-realistic budget.
- Each backend in this bench has its own independent threadpool. With
  `--threads max`, multiple backends contend for the same cores during
  the round-robin loop, producing measurement noise. `--threads 4` keeps
  total worker threads bounded (~24 across 6 backends) and matches a
  realistic mobile workload.

### What's still asymmetric (read before plotting)

We've fixed warm-cache reads (NM scales to exceed SLC) and ORT's per-call
session-creation overhead (sessions cached per (M,K,N)). Three known
asymmetries remain:

1. **Mixed precision plotted on the same axis.** Cactus and ggml matmul use
   4-bit weight paths; LiteRT, ExecuTorch, and ONNX Runtime matmul remain
   INT8 because those benchmarked APIs are INT8-only here. ExecuTorch SDPA prefill
   is FP32-only (no FP16 path exists in `op_sdpa.cpp`); ORT GQA decode is
   FP16-KV (no INT8-KV op exists in ORT). Both flagged with † / ‡ in plot
   legends. ExecuTorch is doing 2× the floating-point work cactus is;
   ORT decode is doing higher-precision KV reads. Y-axis is wall-clock,
   not "useful work per second."

2. **Per-channel vs per-group(32) INT8.** LiteRT and ExecuTorch use
   per-channel scales (1 scale per row); cactus / ggml / ORT use per-group
   scales (4 scales per 128-d row). Per-channel is faster (fewer scale
   lookups) but coarser. Flagged with * in plot legends.

3. **Graph-runtime per-call overhead.** ggml is a graph runtime — each
   `ggml_graph_compute` has a per-call dispatch cost (~50–100μs) that
   wouldn't exist if you were running a full LLM forward pass through
   ggml. We've cut what we can (honor `--threads`, share K/V graphs), but
   ggml's per-call floor is intrinsic to its design. Look at the small-S
   end of graph 4/5 to see this floor as a flat segment in the ggml line.

```
./attn_bench
  --warmup N          (default: 50)
  --iterations N      (default: 200)
  --graphs ...        Comma list of: attn_prefill_s, attn_decode_cache
  --dims 128,...      Sweep S / cache_len values
  --model_dim N       Default 1024
  --heads N           Default 8 (q_heads = kv_heads)
  --backends LIST     ...
  --threads N|max     ...
  --csv path
```

## CSV format

`matmul_bench.csv` columns:
```
graph, sweep_dim, M, K, N, backend, framework, time_us, gops, nrmse, max_err
```

`attn_bench.csv` columns:
```
graph, sweep_dim, seq_len, cache_len, head_dim, q_heads, kv_heads,
backend, framework, time_us, gflops, nrmse, max_err
```

The five user-facing graphs map onto these CSVs as:

| Graph (slide deck) | CSV file              | `graph` filter        | x-axis is `sweep_dim` |
|--------------------|------------------------|-----------------------|------------------------|
| 1 GEMV             | matmul_bench.csv       | `gemv_d`              | d                      |
| 2 GEMM (d sweep)   | matmul_bench.csv       | `gemm_d`              | d                      |
| 3 GEMM (M=N sweep) | matmul_bench.csv       | `gemm_mn`             | M=N                    |
| 4 Attn prefill     | attn_bench.csv         | `attn_prefill_s`      | S                      |
| 5 Hybrid decode    | attn_bench.csv         | `attn_decode_cache`   | cache_len              |

## Status of third-party backends on this branch

| Backend | Matmul (graphs 1–3) | Attention prefill (graph 4) | Hybrid decode (graph 5) | Kernel kind |
|---------|---------------------|------------------------------|--------------------------|-------------|
| **cactus**     | `cactus_cq4` (CQ4 / INT4 weights) | `cactus_prefill` (FP16) | `cactus_decode` (FP16 Q + INT8 KV group=32) | fused |
| **ggml**       | `ggml_q4_0` (Q4_0 weights) | `ggml_fa_q8_prefill` (Q8_0 KV) / `ggml_mm_q8_prefill` (composed) | `ggml_fa_q8_decode` (Q8_0 KV) | fa_*: fused / mm_*: graph |
| **LiteRT**     | `litert_neon`, `litert_ruy` (INT8 per-channel) | `litert_ruy_prefill` (composed) | `litert_ruy_decode`, `litert_neon_decode` (composed) | matmul-composed (no fused op) |
| **ONNX RT**    | `onnxrt_int8` (INT8 group=32 via `MatMulNBits`) | `onnxrt_gqa_prefill` (**FP16**, GQA) | `onnxrt_gqa_decode_fp16kv` (**FP16 KV**) | fused (MlasFlashAttention) |
| **ExecuTorch** | `executorch_int8` (INT8 per-channel via XNNPACK `qc8w`) | `executorch_sdpa_prefill_fp32` (**FP32**) | `executorch_qsdpa_decode_int8pc` (**INT8 per-channel**) | fused |

### INT4 support in this bench

| Backend | INT4 / CQ4 path |
|---------|-----------------|
| **cactus** | Yes — `cactus_cq4` uses `cactus_quant_matmul` with CQ4 / INT4 weights. |
| **ggml** | Yes for matmul — `ggml_q4_0` uses ggml Q4_0 weights with Q8_0 activations. Attention remains Q8_0 KV in this bench. |
| **LiteRT** | No INT4 variant registered here; Ruy/NEON paths are INT8. |
| **ONNX RT** | No INT4 variant registered here; `MatMulNBits` is configured for 8-bit. |
| **ExecuTorch** | No INT4 variant registered here; XNNPACK path is INT8 `qc8w`. |

### How attention is implemented per backend (read this before plotting)

Not every backend ships a fused attention kernel. Three categories exist in
this bench, and they are NOT all apples-to-apples — call the difference out
in any plot caption.

| Backend                      | What "attention" means here                                                                                  | Composed in this bench? |
|------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------|
| `cactus_prefill`             | `cactus_attention_f16` — single fused kernel call                                                            | **No**, fused           |
| `cactus_decode`              | `cactus_attention_hybrid_int8_fp16` — single fused kernel call                                               | **No**, fused           |
| `ggml_fa_q8_prefill/decode`  | one `ggml_flash_attn_ext` node                                                                               | **No**, fused           |
| `ggml_mm_q8_prefill`         | three ggml ops (`mul_mat → soft_max_ext → mul_mat`) chained, executed by ggml's graph scheduler              | Yes — via **ggml's graph API** |
| `litert_ruy_prefill/decode`  | I call `ruy::Mul` directly twice with my own NEON dequant + causal softmax in between                        | **Yes — by me, in C++** |
| `litert_neon_decode`         | I call `tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate` directly, with my own softmax             | **Yes — by me, in C++** |
| `executorch_sdpa_prefill_fp32` | `torch::executor::native::custom_sdpa_out` — single fused kernel                                           | **No**, fused (real FlashAttention impl in `op_sdpa_impl.h`) |
| `executorch_qsdpa_decode_int8pc` | `custom_quantized_sdpa_out` — single fused kernel                                                        | **No**, fused           |
| `onnxrt_gqa_prefill/decode`  | one-node ONNX model (`com.microsoft.GroupQueryAttention`); ORT dispatches its own CPU kernel (`MlasFlashAttention` when conditions match) | **No**, fused |

**The only hand-composed-in-C++ attention is the LiteRT triplet.** ggml's
`mm_q8` is also composed, but at the framework's graph-API level — one
abstraction layer cleaner than what I do for LiteRT. The other backends
each call a real fused attention kernel exactly once per attention.

What this means for the LiteRT numbers specifically: a real LiteRT
deployment would convert a model to `.tflite` and run via `tflite::Interpreter`,
which executes a MatMul → Softmax → MatMul graph using the same Ruy + NEON
primitives I'm calling directly. My direct-Ruy approach skips the
interpreter dispatch overhead, so the LiteRT timings here are slightly
**optimistic** vs what a real TFLite `.tflite` run would clock.

### Attention precision/scheme map

When plotting graphs 4 and 5, label the precision/scheme to be honest about
what's being compared:

| Backend / variant | Prefill precision | Decode KV scheme |
|-------------------|-------------------|------------------|
| cactus            | FP16              | INT8 per-group(32) |
| ggml `fa_q8`      | Q8_0 KV (group=32) | Q8_0 KV (group=32) |
| ggml `mm_q8`      | matmul-composed (Q8_0 KV)  | (decode dropped — wrong output) |
| litert `ruy_*`    | matmul-composed INT8 per-row | matmul-composed INT8 per-row |
| litert `neon_decode` | (decode-only path) | matmul-composed INT8 per-row |
| onnxrt            | FP16              | FP16 KV (no INT8-KV op exists in ORT) |
| executorch fp32   | FP32              | INT8 per-channel (FP32 accum) |

### Notes

- **LiteRT** has no fused attention op — it's a runtime that executes
  graphs, not a kernel library. The "attention" entries are hand-composed
  (Q·Kᵀ → softmax → @V) using LiteRT's INT8 matmul kernels, mirroring
  ggml's `mm_q8_*` matmul-composed paths.
- **ONNX Runtime** CPU EP only supports FP16 attention through
  `com.microsoft.GroupQueryAttention`. `MultiHeadAttention` and
  `DecoderMaskedMultiHeadAttention` exist on CPU but are FP32-only. There is
  **no INT8-KV-cache attention op** anywhere in ORT.
- **ExecuTorch** SDPA lives in `extension/llm/custom_ops/op_sdpa.cpp`
  (verified to be the only fused CPU attention path in the ExecuTorch /
  XNNPACK stack). Wired under `WITH_EXECUTORCH_LLM` since it requires the
  full ExecuTorch runtime (not just XNNPACK).
- **ggml `mm_q8_decode`** (matmul-composed decode) is registered in source
  but commented out — it produces wrong output (nrmse=1.24). The flash-
  attention path (`fa_q8_decode`) is the canonical comparator.

All 11 attention backends and 6 matmul backends have been smoke-tested at
S=cache=512 with passing accuracy. Plot ready.
