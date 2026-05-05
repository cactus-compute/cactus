# Cactus v2 + TQH4 — top 5 hypotheses for residual decode collapse

For symptoms see `CACTUS_TOOL_CALL_DIAGNOSIS.md`. Branch: `karen/tq-v2-debugging`, HEAD `171d27db`.

Rebuilt the prior analysis after a deeper read. The "logits exit at 1/16 magnitude" claim from round 2 was **wrong** — the compensation is present: `tensor_io.py:172-173` writes `output_norm.weights` × 16, then `model_gemma4.cpp:451` does `rms_norm(final_hidden, output_norm_weight)` before the LM-head matmul. RMSNorm normalises out the scale of `final_hidden`; the ×16 in `output_norm` is what survives, and that pairs cleanly with the ÷16 baked into `token_embeddings` (used as the tied LM head). So that hypothesis is dead.

The actual top-5 candidates, ranked by symptom fit + ease of triggering. Each entry has: where, why-it-fits, fix, **ARM-runnable test**.

ARM-test convention: tests must be C++ in `cactus-kernels/tests/` (built with `cactus-kernels/build.sh`) or pure-Python that simulates the kernel and diffs against a torch/numpy reference. No GPU paths.

---

## H1. `cactus_quant_matmul` int8 activation quantisation collapses decode-time MLP-down rows

### Where

`cactus-kernels/src/matmul.cpp:1175-1181` — inside `cactus_quant_matmul`, after the Hadamard activation transform (line 1173), before the GEMM (line 1227+):

```cpp
for (uint32_t m = 0; m < M; m++) {
    for (uint32_t g = 0; g < num_groups; g++) {
        act_scales_ptr[m * num_groups + g] = tq_quantize_group_i8(
            code_basis_ptr + m * W->K + g * gs,
            act_i8_ptr + m * W->K + g * gs, gs);   //  per-group int8 quant of activations
    }
}
```

`tq_quantize_group_i8` (lines 1054-1080) is `scale = max(|act|)/127; act_i8 = round(act/scale)`.

### Why-it-fits

- `gate * up` in the Gemma-4 MLP frequently emits a sharp positive spike from the GeLU on one or two channels of a 128-wide group (this is documented in the SmoothQuant / GPTQ literature for SwiGLU/GeLU MLPs).
- A single channel ≥10× the median collapses every other channel in the group to `int8 ∈ {-1, 0, +1}` (≈ 0.8 % of full scale).
- For `M=1` decode there is no averaging across query positions; any one-token activation profile that has a hot channel determines the entire group's scale.
- `down_proj` has K=8192 → 64 groups. At least one of those is virtually guaranteed to have a 10× spike in a token where the gate fires.
- After `down_proj`, residual + post-norm → next layer's input. Repeat 30 layers; the noise is correlated with the activation pattern.
- Symptom shape: identical-token loops "Obsessive: Obsessive: Obsessive:" and "I am doing well." × 6 are exactly what you get from a ~few-mV per-step logit drift that lands on the same argmax repeatedly.
- **Specifically supports decode-only failure**: prefill at `M=N` rotates through different one-hot spike channels per-position, so the per-group scale is closer to a population max; the int8 noise floor averages out across rows. Decode `M=1` is the worst case.

### Fix

Two options:

1. **Promote activations to int16 instead of int8** at this step. Doubles transient memory, halves quantisation noise per group from ~0.8 % to ~0.003 %. Add a flag-gated path:

   ```cpp
   #if defined(CACTUS_DECODE_ACT_INT16)
   // act_i8_ptr → act_i16_ptr, GEMM int16×int8 path
   #else
   // existing int8 path
   #endif
   ```

   The GEMM then becomes `int16(act) × int8(weight) → int32` — there's no SDOT for that, you'd do `vmlal_s8`/`vmlal_s16` chains. Likely 2–3× slower on M=1.

2. **Subgroup the int8 scale** — quantise per 16-element chunk instead of per 128. `gs/16 = 8` sub-scales per group, max(|act|) per 16 instead of per 128. Cuts the worst-case noise floor by sqrt(8) at the cost of 8× scale storage. The GEMM unrolls already process 16-element chunks (kPanelKChunk=16) — sub-scale aligns with the inner unroll. Cleanest approximate fix.

The second option is the one to land first.

### ARM-runnable test

Add to `cactus-kernels/tests/test_quant.cpp`. Mirrors the kernel's exact buffer layout, uses one decode activation captured from a real run, and checks that the int8-then-dequant round-trip stays within 5 % per element on at least 95 % of channels:

```cpp
// test_int8_act_quant_decode_floor
TEST(QuantMatmul, ActIntQuantNoiseFloor) {
    constexpr uint32_t gs = 128;
    // Simulate a decode-time post-Hadamard activation row: 4 hot
    // channels at ~10x the rest.
    std::vector<__fp16> act(gs);
    std::mt19937 rng(42);
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (uint32_t k = 0; k < gs; k++) act[k] = (__fp16)N(rng);
    act[7] = (__fp16)15.0f; act[33] = (__fp16)-12.0f;
    act[88] = (__fp16)11.0f; act[121] = (__fp16)-9.0f;

    std::vector<int8_t> i8(gs);
    float scale = tq_quantize_group_i8(act.data(), i8.data(), gs);

    int n_collapsed = 0, n_total = 0;
    for (uint32_t k = 0; k < gs; k++) {
        float orig = (float)act[k];
        if (std::abs(orig) < 0.5f) continue;          // skip near-zero channels
        n_total++;
        float deq = (float)i8[k] * scale;
        float rel = std::abs(deq - orig) / std::abs(orig);
        if (rel > 0.05f) n_collapsed++;
    }
    // Today this fails: ≥30 % of channels lose >5 % relative precision.
    EXPECT_LT(n_collapsed, n_total / 20);
}
```

If the test as written FAILS on the current code, that confirms the noise floor is real. After the sub-group fix it should PASS. Run via `cactus-kernels/build.sh && ./build/test_quant`.

---

## H2. KV-cache int8 quantisation per 32-element group accumulates over context length

### Where

- Append: `cactus-kernels/src/quants.cpp:267-296` — `cactus_quantize_kv_fp16_to_int8`.
- Append plumbing: `cactus-graph/src/ops_cache.cpp:94-155` — every appended K, V row is int8-quantised per group of 32.
- Read: `cactus-graph/src/ops_cache.cpp:157-207` — `cactus_attention_hybrid_int8_fp16` consumes the int8 cache + per-(seq_pos, head, group) `float` scales.
- Group constant: `cactus-kernels/cactus_kernels.h:34` — `KV_QUANT_GROUP_SIZE = 32`.

### Why-it-fits

- Each cached K and V token row is quantised to int8 with one `float` scale per 32-channel group of head_dim. That's another ~0.4 % per-element noise floor on top of H1.
- Errors accumulate over context: at decode token N, attention dot-products sum N int8-dequantised products. Variance grows with N.
- Symptom shape: "Turn 2: Hello, I am doing well." echoes Turn 1 — a textbook KV-cache leakage signature. If KV scales drift slightly across turns, attention scores tilt toward earlier-stored tokens, and greedy decode collapses onto a near-copy of the previous turn.
- KV `group_size=32` is **smaller** than activations' 128 (so per-element noise is lower) but every cached token passes through it; 1024-token context = 32 K cached vectors all individually noisy.

### Fix

Two options, lighter then heavier:

1. **fp16 KV cache for short contexts** — gate on `current_seq_len < CACTUS_KV_INT8_THRESHOLD` (e.g. 512). For decode of typical chat lengths (≤ 1 K) fp16 KV doubles cache memory but eliminates the noise. `BufferDesc::cq_codebook` already has the precision-switch hooks; just route `compute_kv_cache_state_node` to allocate fp16 when threshold isn't met, and have the attention kernel detect and dispatch. This is the cleanest test-trigger.

2. **Per-token (not per-group) scale** — use one fp16 scale per row of K/V, not per 32 channels. Cuts cache scale storage by 4× and removes the per-group noise stratification. Attention dequant becomes a single multiply per `dot(Q, K)` row instead of per group. Slightly less precise per-element but uniform across the head_dim — empirically this is the more robust choice in published quantised KV work (KIVI, KVQuant). Likely no measurable speed change.

### ARM-runnable test

Two-step: (a) verify the noise floor; (b) verify the multi-turn-echo failure mode by feeding the same prefilled state through hybrid-int8 attention and a fp16-reference attention and diffing.

```cpp
// test_kv_int8_attention_drift
TEST(KVCache, Int8AttnHybridAgainstFp16Reference) {
    constexpr size_t H = 4;
    constexpr size_t hdim = 256;
    constexpr size_t Tprev = 512;
    constexpr size_t Tnew = 1;

    std::vector<__fp16> Q(H * hdim), K_all(Tprev * H * hdim), V_all(Tprev * H * hdim);
    std::mt19937 rng(7);
    std::normal_distribution<float> N(0, 0.1f);
    for (auto& x : Q) x = (__fp16)N(rng);
    for (auto& x : K_all) x = (__fp16)N(rng);
    for (auto& x : V_all) x = (__fp16)N(rng);

    // Reference: pure fp16 attention.
    std::vector<__fp16> out_ref(H * hdim);
    cactus_attention_fp16_reference(Q.data(), K_all.data(), V_all.data(),
                                    out_ref.data(), Tprev, H, hdim, /*scale=*/0.0625f);

    // Cactus: int8-quantise each K, V row, then call hybrid attention.
    std::vector<int8_t> K_i8(Tprev * H * hdim);
    std::vector<float> K_scales(Tprev * H * (hdim / 32));
    cactus_quantize_kv_fp16_to_int8(K_all.data(), K_i8.data(), K_scales.data(),
                                    Tprev, H, hdim, 32);
    /* same for V */
    std::vector<__fp16> out_test(H * hdim);
    cactus_attention_hybrid_int8_fp16(Q.data(), K_i8.data(), V_i8.data(),
                                      K_scales.data(), V_scales.data(),
                                      /*K_new=*/nullptr, /*V_new=*/nullptr,
                                      out_test.data(),
                                      /*B=*/1, /*Tnew=*/0, /*Thist=*/Tprev,
                                      /*Tq=*/1, H, H, hdim,
                                      0.0625f, 0, true, Tprev + 1, 32, hdim);

    float max_rel = 0.0f, mean_rel = 0.0f; size_t n = 0;
    for (size_t i = 0; i < H * hdim; i++) {
        float r = (float)out_ref[i], t = (float)out_test[i];
        if (std::abs(r) < 1e-3f) continue;
        float rel = std::abs(t - r) / std::abs(r);
        max_rel = std::max(max_rel, rel);
        mean_rel += rel; n++;
    }
    mean_rel /= n;

    // Today: max_rel often exceeds 10 % at Tprev=512.
    EXPECT_LT(max_rel, 0.05f);
    EXPECT_LT(mean_rel, 0.005f);
}
```

If the test fails as expected, swap the cache to fp16 (option 1 above) and re-run; pass should be tight (mean_rel < 1e-4).

---

## H3. `tool_constrainer_.get_bias()` returns a non-empty bias map even when no tools are active

### Where

- `cactus-engine/src/model.cpp:256` — every decode call does `auto combined_bias = tool_constrainer_.get_bias();`, then adds vocab_bias_ and the rep-penalty.
- `cactus-engine/src/constraints.cpp:298` — `tool_constrainer_.init(config_.model_type, {}, tokenizer_.get())` is called with empty tools list at start of every completion. If `init` doesn't fully reset internal `current_bias_` to `{}`, the cached bias persists.
- `cactus-engine/src/engine.h:503` — `get_bias()` returns `current_bias_` directly.

### Why-it-fits

- The user reports the *non-tool* test cases ("1k_context", "streaming", "prefill cold/warm") also producing repetition. Repetition under greedy is consistent with mild but consistent positive bias on a small set of tokens (e.g. the model name token, the `<|tool_call>` opener token, "the", etc.).
- If the constrainer carries over biases from a prior tool-active conversation into a subsequent non-tool one (or even on the *same* conversation after a partial init), every sample step has the wrong logit landscape. This is independent of the activation noise above.
- **Specifically supports cross-test contamination**: the test suite runs cases sequentially; an earlier `tool_calls` test that does FSM-style tracking of bias updates may leave residual state when the next test (`prefill cold`) starts.

### Fix

In `constraints.cpp:298`, audit `tool_constrainer_.init(empty_tools, ...)` and confirm that:

1. `current_bias_.clear()` is called unconditionally on init (even when tools is empty).
2. The FSM `state_` is reset to `Idle` (or equivalent).
3. Any per-token-history accumulators are wiped.

If `init` already does these, add an unconditional `tool_constrainer_.reset()` at the head of each `do_prefill` call (`complete.cpp:515`). Cost: O(vocab_size) memset; negligible.

The minimal pre-fix patch is:

```cpp
// constraints.cpp, in compose_per_completion_state or similar:
- tool_constrainer_.init(config_.model_type, tools, tokenizer_.get());
+ tool_constrainer_.reset();
+ tool_constrainer_.init(config_.model_type, tools, tokenizer_.get());
```

### ARM-runnable test

Pure C++, no GPU, no model. Wire up a mock tokenizer and exercise the constrainer:

```cpp
// test_constrainer_reset
TEST(ToolConstrainer, ResetClearsBiasAcrossCompletions) {
    MockTokenizer tok;
    ToolCallConstrainer ctr;

    // Completion 1: tool-active.
    std::vector<ToolFunction> tools = { make_dummy_tool("get_weather") };
    ctr.init("gemma4", tools, &tok);
    // simulate a few token updates that put the FSM into "after-tool-name" state
    for (uint32_t tok_id : {105 /*<|turn>*/, /*model*/, 48 /*<|tool_call>*/}) {
        ctr.update(tok_id, tok.decode(tok_id));
    }
    ASSERT_FALSE(ctr.get_bias().empty());

    // Completion 2: no tools.
    ctr.init("gemma4", {}, &tok);
    EXPECT_TRUE(ctr.get_bias().empty()) << "Bias not cleared after non-tool re-init";

    // After current state-change, bias must remain empty for tool-free completions.
    ctr.update(106 /*<turn|>*/, "<turn|>");
    EXPECT_TRUE(ctr.get_bias().empty()) << "Bias contaminated mid-completion when no tools";
}
```

---

## H4. `compute_embedding_node` and `cactus_quant_orthogonal_matmul` use the **same** packed `token_embeddings` weight via two **different** code paths

### Where

- Input embedding (token-id → hidden): `cactus-graph/src/ops_tensor.cpp:241-310` — `dequantize_orthogonal_embedding_row` reconstructs one row at a time as fp16, then writes it into the output buffer.
- LM head (hidden → logits): `cactus-graph/src/ops_nn.cpp:117-149` dispatches to `cactus_quant_orthogonal_matmul` (`matmul.cpp:1428-1496`), which contracts the **whole** `[V × K]` matrix against the hidden state in fp32.
- Both consume the same `BufferDesc` (the mmapped `token_embeddings.weights`). But the math is implemented twice.

### Why-it-fits

- The previous diagnosis verified `cactus_quant_orthogonal_matmul` against the reference dehydrator at 2.5e-7 max abs error. Tight.
- It did **not** verify `dequantize_orthogonal_embedding_row` against the same reference for the same set of rows. If the row-reconstruction differs from the matmul reconstruction by even fp16-noise per element, the input embedding fed into layer 0 is **not** what training/QDQ saw.
- Cumulative effect across 30 layers of an off-distribution layer-0 input is exactly the symptom shape — model can't anchor on any specific output token, drifts into a fixed-point repetition.
- This is a uniquely cactus issue (other inference engines treat embedding as a separate gather op and don't have a "matmul vs row-decode" duality on the same buffer).

### Fix

Either:

1. **Replace `dequantize_orthogonal_embedding_row` with a one-row call to `cactus_quant_orthogonal_matmul`** with `M=1, A = onehot(token_id)`. Wasteful — full matrix multiply for 1 token — but trivially correct.
2. **Better**: rewrite `dequantize_orthogonal_embedding_row` so its math is *line-for-line* the same algebra as `cactus_quant_orthogonal_matmul`'s inner loop, with the same dtype precision sequence (fp32 accumulate, single fp16 cast at end).

Likely (2) just needs a careful audit of where each multiply happens; (1) is a one-liner band-aid for testing.

### ARM-runnable test

Wholly Python-side, since this is about cross-checking two C++ functions that operate on bytes from disk:

```python
# test_embed_lookup_vs_matmul.py — runs on ARM CPU, no GPU
import numpy as np
import torch
from pathlib import Path
import struct

# Read converted v2 token_embeddings.weights, reconstruct row r=12345
# via: (a) the equivalent of dequantize_orthogonal_embedding_row,
#      (b) call to cactus_quant_orthogonal_matmul with onehot([12345]).
# Both should match within fp16 round-trip (max abs ~ 2.5e-3 due to rotation × cast).

ROW = 12345

# (a) Path A — mirror dequantize_orthogonal_embedding_row exactly:
def reconstruct_row_via_embed(weights_path, row):
    # parse header, extract codebook, norms, input_scale_recip, rotation R[K,K], indices[V,K]
    ... (see verify_tqh_v2.py for the parse)
    indices_row = indices[row]                      # [K] uint8
    cb_row = codebook[indices_row]                  # [K] fp16
    rotated = (cb_row.astype(np.float32) @ R.astype(np.float32))   # [K] fp32
    out = rotated * float(norms[row]) * input_scale_recip.astype(np.float32)
    return out.astype(np.float16)

# (b) Path B — mirror cactus_quant_orthogonal_matmul on M=1, A = onehot(row):
def reconstruct_row_via_matmul(weights_path, row):
    ... (parse the same way)
    A = np.zeros(V, dtype=np.float32); A[row] = 1.0
    # but the matmul is C[m,n] = norm[n] * sum_k cb[idx[n,k]] * A_rot[m,k]
    # for a one-hot A at row r, A_rot = (A @ isr @ R) → only contribution
    # from the r-th row of (isr @ R), so this is essentially the same algebra
    # but expressed via the matmul shape.
    A_rot = (A * input_scale_recip.astype(np.float32)) @ R.astype(np.float32)
    cb_for_row = codebook[indices[row]]
    out = float(norms[row]) * (cb_for_row.astype(np.float32) @ A_rot.astype(np.float32))
    # but this doesn't fully exercise the matmul's contraction — different shape.
    # Instead, do a 1×K input and contract against the whole V×K weight,
    # then read the row r entry of logits.
    ...

a = reconstruct_row_via_embed("token_embeddings.weights", ROW)
b = reconstruct_row_via_matmul("token_embeddings.weights", ROW)
print(f"max |a-b| = {np.abs(a.astype(np.float32) - b.astype(np.float32)).max()}")
assert np.abs(a.astype(np.float32) - b.astype(np.float32)).max() < 5e-3
```

Run this for 100 random rows; if any diverge by > 5e-3 max abs, the two paths are not consistent.

A C++ version is also possible by linking the two functions and feeding the same buffer through both, but Python is the lower-friction path for static analysis.

---

## H5. Hadamard activation transform's `permute` step uses a fixed-size stack buffer that may misalign on `gs ≠ 128`

### Where

- `cactus-kernels/src/matmul.cpp:265` — inside `cactus_quant_transform_hadamard_activations`:

  ```cpp
  __fp16 tmp[256];          // fixed-size stack
  ```

- This buffer is used to permute a single group of `gs` activation channels post-Hadamard.

### Why-it-fits

- For Gemma-4-E2B `gs=128` always; safe with margin.
- BUT: this is fragile — any model/config that bumps `gs > 256` (say, a 2-bit model with `gs=512` for storage savings) silently overruns the stack. If Karen's pipeline has any layer that ended up with `gs=256` (e.g. embed orthogonal path reuses the same kernel — verify), the permute step writes one element past the end → corrupts the next stack slot, which on ARM is typically the `act_scales[]` accumulator or the codebook pointer.
- **In symptom space**: a corrupted `act_scales` for one group of one row produces *exactly* the "row of zeros" outcome which translates to a logit landscape with one or two outliers boosted enormously. Greedy then locks onto those outliers.
- This is more of a latent bug than a current one for this specific config — but it's been triggered before in similar codebases when someone bumps a config knob.

### Fix

Replace fixed buffer with:

```cpp
- __fp16 tmp[256];
+ __fp16 tmp[gs];   // or std::vector<__fp16> tmp(gs); if gs is non-constexpr
```

Or static_assert that `gs ≤ 256` in the kernel:

```cpp
assert(gs <= 256 && "permute stack buffer too small");
```

### ARM-runnable test

```cpp
// test_hadamard_activation_transform_oversized_gs
TEST(HadamardTransform, RespectsGroupSizeBound) {
    // Construct a synthetic CactusQuantMatrix with gs=256 and run the
    // activation transform; if the existing kernel reads/writes past
    // tmp[256] this test will trigger an ASAN failure when built with
    // -fsanitize=address.
    CactusQuantMatrix W = make_dummy_qmatrix(/*gs=*/256, /*bits=*/4, /*N=*/16, /*K=*/256);
    std::vector<__fp16> A(256), Out(16 * 256);
    for (auto& x : A) x = 1.0f;
    cactus_quant_transform_hadamard_activations(W, A.data(), 1, Out.data());
    // No EXPECT — the test passes if ASAN doesn't trip.
}
```

Build with `cactus-kernels/build.sh ASAN=1`. If ASAN is off, this won't catch the overflow (it's silent) — Karen should run the test suite under ASAN at least once.

---

## How to triage in 30 minutes

Run the H1 and H2 tests first. They both fail today (predicted), and the failure magnitudes will tell you which is the dominant cause:

- If H1 fails with "many channels collapse" (count > 5 % of channels for the synthetic decode activation), the int8 act-quant noise floor is the dominant cause. Fix with the sub-grouping approach.
- If H1 passes (or fails by a small margin) but H2's `max_rel` is large (> 5 %), KV-cache int8 quantisation is the dominant cause. Fix with fp16 KV at short context lengths.
- H3 is the cheapest fix — patch and run the existing `tool_calls` test; if the symptom of "non-tool tests showing repetition" disappears, that was the bug.
- H4 and H5 are the long shots; check after the top 3 if symptoms persist.

## Files referenced

- `cactus-kernels/src/matmul.cpp:265, 1054-1080, 1126-1326, 1428-1496`
- `cactus-kernels/src/quants.cpp:267-296`
- `cactus-kernels/cactus_kernels.h:34, 420, 715`
- `cactus-graph/src/ops_cache.cpp:94-220`
- `cactus-graph/src/ops_nn.cpp:93-150`
- `cactus-graph/src/ops_tensor.cpp:241-310, 313-360`
- `cactus-engine/src/model.cpp:218-272`
- `cactus-engine/src/constraints.cpp:280-310`
- `cactus-engine/src/complete.cpp:515-573, 623-820`
- `cactus-engine/models/gemma4/model_gemma4.cpp:97-111, 215-411, 440-553`
- `python/src/tensor_io.py:150-185`
- `python/src/tqh_prod_convert.py:35-58`
- `cactus-kernels/tests/test_quant.cpp` — extend
- `cactus-kernels/tests/test_attention.cpp` — extend (add hybrid-int8-vs-fp16-ref test)
- `cactus-engine/tests/` — extend with constrainer reset test
