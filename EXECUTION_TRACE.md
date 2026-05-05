# Cactus v2 + TQH4 Gemma-4 — execution trace

Branch: `karen/tq-v2-debugging`, HEAD `171d27db v2 latest`. All file:line refs are
against this commit. This document was produced by static read-through of the
C++; nothing here was actually executed end-to-end, but I did run the converter
and a Python equivalence test for the on-disk format (see "Conversion
verification" below).

## Conversion verification (Task A)

`tqh_prod_convert.py` ran cleanly:

```
$ python -m src.tqh_prod_convert --bundle /tmp/tqh_bundle \
    --base /workspace/cactus/test_weights/gemma-4-E2B-it-fp16-base \
    --out  /workspace/cactus/test_weights/gemma-4-E2B-it-tqh-prod-v2
wrote 527 CQ weights to /workspace/cactus/test_weights/gemma-4-E2B-it-tqh-prod-v2
```

(The converter expects `<bundle>/packed/...` so I symlinked
`/workspace/turboquant/artifacts/packed_av/PROD_v2_a2_L4_pli2_emb4/{packed.safetensors,metadata.json,...}`
into `/tmp/tqh_bundle/packed/`.)

Output dir: 1.9 GB, 1964 files (1437 inherited fp16 from the FP16 base, 527
freshly written CQ weights). `tqh_conversion_report.json` shows
`written_count=527, skipped=0`.

A direct re-implementation of the on-disk parsing in
`/workspace/cactus/test_weights/verify_tqh_v2.py` reads the cactus header
+ scales blob, unpacks LSB indices, rebuilds the weight matrix in fp32,
and compares against `tqh_runtime.TQHPackedLoader.dehydrate(layer_name)`.
Eight representative layers (one orthogonal embed + seven Hadamard
projections in layer 0):

```
layer                           ok?       max|d|    mean|d|
token_embeddings.weights        OK     1.009e-05  5.892e-07
layer_0_attn_q.weights          OK     2.126e-04  9.093e-06
layer_0_attn_k.weights          OK     9.079e-05  9.640e-06
layer_0_attn_v.weights          OK     1.168e-04  9.145e-06
layer_0_attn_output.weights     OK     2.058e-04  7.831e-06
layer_0_ffn_gate.weights        OK     2.114e-03  1.341e-04
layer_0_ffn_up.weights          OK     2.224e-03  1.481e-04
layer_0_ffn_down.weights        OK     2.462e-04  7.248e-06

8 passed, 0 failed
```

The 2e-3 max-abs on `ffn_gate`/`ffn_up` is the largest, but consistent with
fp16 round-tripping of norms × `max(|cb|)≈1` × group-of-128 dot products.
All errors are dominated by fp16 storage of `norms`, `codebook`, and
`input_scale_recip`. Conversion is correct.

(Note: `gemma4_scale_factor` pre-bakes a ×1/16 into `token_embeddings`,
`output_weight`, `embed_vision_*` and ×16 into `ffn_gate`, `ffn_up`,
`per_layer_gate`, `moe_gate_proj`, `moe_up_proj`. The verifier multiplies
the reference by the same factor before diffing — this matches the
`tensor_io.py:154-172` convention.)

---

## Task B — Static execution trace

### 1. Loader path (init)

`cactus_init` in `cactus-engine/src/cactus.cpp` constructs a `Gemma4Model`,
which is the only path for Gemma-4 v2. Its constructor calls
`load_weights`, which invokes:

- `cactus-graph/src/io.cpp:329 CactusGraph::mmap_weights(filename)` for every
  per-tensor `.weights` file. After parsing the header (`MappedFile::parse_header`,
  `io.cpp:777-829`), it inspects `precision`:
  - **Float / FP16**: `io.cpp:378-413` falls through to the integer-scales branch
    or no-scales branch; the buffer is a plain mmap with `precision=FP16`.
  - **CQ4 (precision id 6) and `group_size > 0`**: `io.cpp:343-377` is the
    interesting branch. It points `buffer.cq_codebook`,
    `buffer.cq_input_scale`, `buffer.cq_input_scale_recip`, `buffer.cq_norms`
    into the scales blob. If `FLAG_ORTHOGONAL_ROTATION` is set (id 1<<1), it
    sets `buffer.cq_rotation = ...` and `buffer.cq_flags = CACTUS_QUANT_FLAG_ORTHOGONAL`
    (`io.cpp:367-368`); otherwise it sets the Hadamard sign tables
    `cq_left_signs / cq_right_signs / cq_permutation` (`io.cpp:370-374`).
- `mmap_embeddings` is the same logic with the parallel branch in
  `io.cpp:279-311`. It is called for `token_embeddings.weights` and
  `embed_tokens_per_layer.weights` from `model_gemma4.cpp:97,109`. For
  Gemma-4 with tied embeddings, `output_weight_node_id_` aliases
  `embedding_node_id_` (`model_gemma4.cpp:99-101`).

The previously-precomputed `cq_expanded` / `cq_norm_f32` block (the per-layer
expansion of CQ4 indices into i8 bytes plus a single global `cb_scale` per
layer) **is not present in v2**. `BufferDesc::cq_expanded` and
`cq_norm_f32` default-init to `nullptr`. This means the matmul kernel always
takes the fallback path (`matmul.cpp:1188-1224`) which rebuilds those
structures **on every matmul call** and uses a **per-call** `cb_scale` from
`tq_quantize_codebook_i8(W->codebook, cb_i8, ...)` at `matmul.cpp:1139`.

Op attachment: in `cactus-graph/src/builder.cpp` `matmul()` (line 127)
emits an `OpType::MATMUL` node; `embedding()` emits `OpType::EMBEDDING`.
Karen's branch added the EMBEDDING case to the precision-override list
at `builder.cpp:1488-1493`, which is the H6 fix from the round-2 diagnosis
(without it, an embedding consuming a CQ4 weight would inherit `precision=CQ4`
for its output buffer and then `compute_embedding_node` would write fp16
into a half-byte buffer — heap corruption).

### 2. Forward pass path (per layer of Gemma-4 LM)

Entry: `cactus_complete` (`complete.cpp:623`) →

1. `format_chat_prompt` (`tokenizer.cpp:355` and friends; the
   `<escape>` bug fixed in patch 1, see `CACTUS_TOOL_CALL_DIAGNOSIS.md`).
2. `do_prefill` (`complete.cpp:515-573`) → `Model::prefill`
   (`model.cpp:154`) → `forward(tokens, true)` chunked.
3. `generate_first_token` → `decode` (`model.cpp:205-251`) → `forward(...)`
   then `gb->matmul(last_hidden, output_weight_node_id_, true, backend)`
   (line 230) — this is the LM head matmul.
4. The decode loop (`complete.cpp:740-820`) calls `Model::decode` per token.

Inside `forward` (Gemma-4):

```
build_preamble_and_embed                  model_gemma4.cpp:389-411
   token_input  = gb->input(...)
   hidden       = scalar_multiply(embedding(token_embeddings, token_input),
                                  sqrt(hidden_dim))                   line 394
   pli_embed    = scalar_multiply(embedding(embed_tokens_per_layer, …),
                                  sqrt(pli_dim))                      line 398
build_pli_combined                        model_gemma4.cpp:376-387
   pli = matmul(hidden, per_layer_model_proj) / sqrt(hidden_dim);     line 381
   pli = rms_norm(pli, per_layer_proj_norm)                           line 384
   pli = (pli + pli_embed) / sqrt(2)                                  line 386
forward_from_embeddings (per layer):
   build_transformer_block                 model_gemma4.cpp:336-364
       pre_attn_norm = rms_norm(hidden, input_layernorm)              line 340
       attn_raw      = build_attention(...)                            line 341
       attn          = rms_norm(attn_raw, post_attention_layernorm)    line 342
       residual      = hidden + attn                                   line 343
       pre_mlp_norm  = rms_norm(residual, pre_feedforward_layernorm)   line 359
       mlp_raw       = build_mlp(pre_mlp_norm)                         line 360
       mlp           = rms_norm(mlp_raw, post_feedforward_layernorm)   line 361
       out           = residual + mlp                                  line 363
   build_per_layer_input (PLI gate path)   model_gemma4.cpp:215-225
       gate = gelu(matmul(hidden, per_layer_gate))                    line 221
       pli_proj = matmul(gate * pli, per_layer_proj)                  line 224
   apply layer_scalar (multiplicative)                                 line 372
```

`build_attention` (`model_gemma4.cpp:254-310`):

```
q = matmul(input, attn_q_weight)                line 269
q = rms_norm(q, attn_q_norm_weight)             line 271
q = apply_partial_rope(q, ...)                  line 274
k = matmul(input, attn_k_weight)                line 281
k = rms_norm(k, attn_k_norm_weight)             line 283
k = apply_partial_rope(k, ...)                  line 286
v_proj = matmul(input, attn_v_weight)           line 288
v = rms_norm(v_proj, v_ones)                    line 290
kv_cache_append(k4, cache_k_nodes_[src])        line 300
kv_cache_append(v4, cache_v_nodes_[src])        line 301
attn = attention_cached(q4, k4, v4, ...)        line 302
out = matmul(attn, attn_output_weight)          line 309
```

`build_mlp` (`model_gemma4.cpp:312-318`):

```
gate = gelu(matmul(input, ffn_gate_weight))     line 315
up   = matmul(input, ffn_up_weight)             line 316
out  = matmul(gate * up, ffn_down_weight)       line 317
```

The fused `dense_mlp_int4_fused` op from commit `a53f0ce4` is **not** here.
The unfused chain is what runs.

`Model::decode` finishes with:

```
last_hidden = gb->index(final_hidden, last_idx)              model.cpp:225
logits      = gb->matmul(last_hidden, output_weight_node)    model.cpp:230
logits      = softcap * tanh(logits / softcap)               model.cpp:232-237
sample      = sample_token(gb, logits, T, top_p, top_k, ...)  model.cpp:238
gb->execute(profile_file)                                     model.cpp:240
```

### 3. Per-kernel mapping for the matmul nodes

Every `gb->matmul(input, W, true, backend)` node ends up at
`compute_matmul_node` (`ops_nn.cpp:93`). The dispatch is at line 117-149:

- `is_cq(W.precision) && W.group_size > 0`:
  - `W.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL` → **`cactus_quant_orthogonal_matmul`**
    (matmul.cpp:1428-1496). This runs **only** for `embed_tokens / output_weight`
    in Gemma-4 (i.e. the LM head matmul on every decode step).
  - else → **`cactus_quant_matmul`** (matmul.cpp:1126-…). This runs for
    every transformer linear: q/k/v/o, ffn_gate/ffn_up/ffn_down, per_layer_*,
    per_layer_model_proj, embed_vision_proj, audio tower linears, etc.
- non-CQ FP16 → `cactus_matmul_f16`.

Embedding lookup is **not** done via matmul — it's
`compute_embedding_node` (`ops_tensor.cpp:313`). For a CQ4 orthogonal
buffer (i.e. `token_embeddings`), every requested row goes through
`dequantize_orthogonal_embedding_row` (`ops_tensor.cpp:241-267`) which
expands one row's K indices, materializes K floats from the codebook,
multiplies by `R[K,K]` row-by-row, then `* norm[row] * input_scale_recip[k]`.
This **does** exercise the same algebra as the LM head matmul — same
weight tensor, but row-decoded into fp16 instead of contracted with an
activation.

So: **input embedding lookup and LM head touch the same memory but go through
two different code paths.** Input uses `dequantize_orthogonal_embedding_row`
(per-row reconstruction in fp16). LM head uses
`cactus_quant_orthogonal_matmul` (whole-matrix contraction, returns fp16
logits).

### 4. Decode-step specifics

There is **no separate decode kernel** in v2. Decode is just
`forward(tokens=[1])` with `use_cache=true` so that `kv_cache_append` is
called and `attention_cached` reads accumulated K/V. The same matmul
kernel `cactus_quant_matmul` runs with `M=1` instead of `M=N` for prefill.

The kernel does have an `M==1` fast path at `matmul.cpp:1157-1163`
(thread-local scratch buffers) and `1227-1326` (parallel-by-N-block
inner loop with per-block running_sum). For `M>1` it goes to a tiled
i8×i8 GEMM at `matmul.cpp:1330+`. So the M=1 vs M>1 distinction is
hot — but it is the same kernel and the same `tq_quantize_codebook_i8`
call.

The "inner==1 fast path" (commit `d917981f`) was in
`cactus/kernel/kernel_reduce.cpp` — that file does not exist any more
(deleted in the `b8527fe0 Aggressive Refactor!` rewrite that produced
the cactus-graph / cactus-kernels / cactus-engine layout). The
equivalent file `cactus-kernels/src/reduce.cpp` has no `inner==1` fast
path; it goes through `axis_reduce_*_impl` for everything. Not on this
codepath.

The `cq_expanded` / `cq_norm_f32` precompute is **not built at load
time** in this branch (`io.cpp:343-377` populates only the codebook /
input_scale / norms / sign-tables and stops). The matmul kernel detects
this via `if (!w_il)` at `matmul.cpp:1190` and rebuilds the expanded
i8 weight block + the f32 norm-table-times-cb_scale **per call**, every
matmul, every layer, every step. That fallback path uses a per-call
`cb_scale = max(|cb_fp16|)/127` (`tq_quantize_codebook_i8`). The codebook
table is fp16, so `cb_scale` is deterministic per layer (each layer has
its own codebook, but the codebook depends only on `(group_size, bits)`,
which for Hadamard transformers is always `(128, 4)` → the codebook is
the **same** fp16 table for every Hadamard 4-bit layer, so `cb_scale`
is in fact a single per-(group_size,bits) constant). I checked: the
converter (`tqh_prod_convert.py:319-320`) caches the codebook by
`(group_size, bits)` and uses the cached fp16 array. So the worry that
`cb_scale` "drifts between calls" was overstated — it is the same fp16
codebook every call, so `cb_scale` is bit-identical. Crisis averted on
that specific concern, but the **performance** cost of rebuilding
`w_il_buf` and `n_f32_buf` on every matmul is real.

### 5. Hadamard `cactus_quant_matmul` (the dominant kernel)

Inputs (`matmul.cpp:1126-1136`):
- `W`: `CactusQuantMatrix` packing
  - `codebook` fp16[1<<bits], `norms` fp16[N×G],
  - `input_scale_recip` fp16[K], `left_signs` int8[gs], `right_signs` int8[gs],
    `permutation` uint32[gs],
  - `packed_indices` uint8[N×G×pgb] LSB-packed,
  - and the optional `expanded`/`norm_f32` precomputes (always null in v2).
- `A`: fp16[M×K] activations,
- `M`, `C`: fp16[M×N] output.

Steps:
1. **Codebook → int8** at line 1139: `cb_scale = max(|cb_fp16|)/127`,
   `cb_i8 = round(cb_fp16 / cb_scale)`. One per matmul call; deterministic
   per (group_size, bits, codebook).
2. **Hadamard activation transform** at line 1173: writes into
   `code_basis_ptr[M×K]` via `cactus_quant_transform_hadamard_activations`.
   Per-group it does `x · isr · left_signs → FWHT → · right_signs → permute`.
   The permute step uses a 256-fp16 stack buffer (`matmul.cpp:265
   __fp16 tmp[256]`); fine for `gs=128` always in Gemma-4.
3. **Quantise activations** to int8 with per-group scale at line 1175-1181:
   `act_scales[m,g] = max(|code_basis[m,g,:]|)/127`,
   `act_i8[m,g,k] = round(code_basis[m,g,k] / act_scales[m,g])`. **This is the
   activation noise floor.** For decode (M=1), one per-group scale per
   call per group. With Gemma's GeLU producing occasional outliers in
   `gate*up` activations entering down_proj, a single channel can
   dominate `max(|act|)`, and the rest of the group quantises to ±0.
4. **(re)build expanded weight** at 1188-1224: 4-row blocks of i8 weights
   in interleaved layout, plus `n_f32 = norms * cb_scale`. Done once
   per call.
5. **GEMM**: at 1227+ for M=1, 1330+ for M>1. Inner loop is dot4
   (`SDOT` / `CACTUS_DOTQ_LANE`) of 128 i8 elements per group, accumulating
   into int32, then scaled by `n_f32 * act_scale` and cast to fp16.

vs `cactus_quant_orthogonal_matmul` (matmul.cpp:1428-1496):
- Acts in **f32** throughout (no int8 quantise of activations).
- Activation transform is one full `K×K` fp16 matmul per row of the input:
  `A_rot = A · isr · R` (the `R[K,K]` rotation). That's `O(M·K²)` per call.
- Output: `C[m,n] = norm[n] · sum_i cb[idx[n,i]] · A_rot[m,i]`. f32 accumulate,
  cast to fp16 once at the end.
- No int8 codebook, no per-call cb_scale.

**Numerical consequences**:
- `cactus_quant_orthogonal_matmul` is essentially exact (verified in the
  prior diagnosis at 2.5e-7 max rel error).
- `cactus_quant_matmul`'s per-group int8 act quantise is a real,
  per-call quantisation noise of ~1/127 ≈ 0.8% per group. After 30+
  layers of GeLU + multiply that compounds, but Karen's training-time
  TQH is supposed to have made the model robust to this. If the model
  is collapsing on **decode but not on prefill**, the int8 act-quantise
  is a suspect (single-token activations have less averaging across
  the channel dimension than batched activations).

### 6. Sampling and stop-detection

`sample_token` (`model.cpp:253-272`):
- Builds `combined_bias` from `tool_constrainer_.get_bias()` + `vocab_bias_` +
  optional `extra_bias`.
- If `repetition_penalty > 1.0`, **subtracts** `log(repetition_penalty)` from
  `combined_bias[tok]` for every `tok` in `token_history_` (additive logit
  penalty, **not** multiplicative on probabilities). Default
  `repetition_penalty = 1.1` → `log(1.1) ≈ 0.095` per repeated token.
- Calls `gb->sample_with_options(logits, T, top_p, min_p, freq_penalty=1.0,
  top_k, combined_bias)`.

`compute_sample_with_options_node` (`ops_sample.cpp`) → `cactus_sample_f16_ex`
(`nn.cpp:549`). The kernel applies `bias[token]` additively to logits before
softmax. The `(void)repetition_penalty;` at `nn.cpp:585` is intentional —
the penalty was already folded into `combined_bias` upstream.

Defaults (`model.cpp:695-697` in `Config::default_*`):
`temperature=1.0, top_p=0.95, top_k=64`. `min_p=0.15` and
`repetition_penalty=1.1` come from the function-default arguments at the
Gemma-4 `Model::decode` virtuals (`model_gemma4.h:319,332,339,348`). These
are the values the engine uses unless the caller overrides.

Stop sequences (`complete.cpp:230-258`):
- always: `eos_token` (single token), `<turn|>` (id 106) — both single-token.
- when `tools.size() > 0`: also `<tool_call|>` (49), `<|tool_response>` (50).
- match is **token-id sequence**. `tokenizer->encode("<turn|>")` returns
  `[106]` and `find_in_history` does substring on the token ID stream.

`force_tools` (`complete.cpp:93`) is the structural constraint that biases
logits toward the `<|tool_call> ... <tool_call|>` envelope. **Default is
false.** So in practice nothing constrains the model's output.

---

## Ranked candidate sites for the residual decode bug

Listed by likelihood given the static-analysis evidence and the
"Obsessive: Obsessive: ..." / "I am doing well." 6× exact-token loops Karen
is seeing:

1. **`cactus_quant_matmul` activation int8 quantisation noise on M=1**
   (matmul.cpp:1175-1181). The noise floor is 1/127 per group, and a
   single GeLU spike in `gate*up` will collapse the rest of an MLP-down
   group to ±0. Compounded over 30 layers and a low-confidence
   distribution this is plausibly a few-millivolt logit drift, exactly
   what produces top-2 ties broken the wrong way and tight loops.
   *Smallest test that would prove or disprove*: dump
   `act_scales_ptr` and `act_i8_ptr` for `layer_29.ffn_down` on a single
   decode step, compare to fp32 reference activation for the same layer.
   If many groups have `max(|act|) >> 5×median(|act|)` and the int8 reps
   fall to ≤2 distinct values, this is a real bug.

2. **KV-cache int8 quantisation** (`ops_cache.cpp:94-155`). K and V are
   quantised to int8 with per-group scale **on append**, then dequantised
   inside `attention_cached`. That's another ±0.4% per-token noise that
   compounds over context length. Look at
   `cactus_quantize_kv_fp16_to_int8` and the dequantise inside
   `compute_attention_cached_node`. Symptom shape: bias toward repeating
   the previous turn's tokens (because their cached K's are slightly
   off so attention scores tilt) is **exactly** the
   "Turn 2: Hello, I am doing well." failure pattern.
   *Smallest test*: run the same prefilled-state into both
   `attention_cached` (via the cache) and a one-shot `attention` that
   re-runs over the full context with fresh fp16 K/V; compare attention
   outputs.

3. **`cb_scale` rounding in `cactus_quant_matmul`** (matmul.cpp:1139).
   The earlier diagnosis flagged this as drifting between calls; on
   re-read the codebook is the same fp16 table for every (gs=128, bits=4)
   layer, so `cb_scale` is bit-identical across calls. **This is NOT a
   bug** — but the absence of a precomputed `cq_norm_f32` means we pay
   the fp16-norms × cb_scale cast on every matmul. The fp16 round of
   `norms*cb_scale` is the same value every call, so still not a bug.
   Lower priority than 1, 2.

4. **`final_logit_softcapping` interaction with the 1/16 LM-head scale.**
   `model.cpp:232-237` does `softcap * tanh(logits / softcap)`. `embed_tokens`
   norms are pre-divided by 16 (`tqh_prod_convert.py:54-57`), and the input
   embed lookup compensates with `* sqrt(hidden_dim) ≈ 48` at
   `model_gemma4.cpp:394-395`, giving a net ×3. The LM head matmul has **no
   compensating multiplier** — logits exit the matmul at 1/16 of their
   "natural" magnitude before softcap. With softcap=30, `logits/softcap ≈
   logits/(16*30) = logits/480`, which is so small that `tanh` is in the
   linear regime and basically a no-op — the softcap does nothing. The
   relative ordering is preserved, BUT temperature/top-p sampling on
   logits 1/16 their natural size is equivalent to running the model at
   `T=1/16` (much sharper than intended), which **will** cause repetition.
   *Smallest test*: dump the raw `logits` tensor pre-softcap on one decode
   step. If `max(logits) - min(logits) ≈ 1.5` (instead of the typical 25),
   this is the bug. The fix is either to NOT divide `embed_tokens`-as-LM-head
   by 16, or to multiply LM-head output by 16 explicitly in
   `model.cpp:230` before softcap. **This is the most architecturally
   suspicious finding and I recommend Karen check it first.**

5. **Permutation-table buffer in Hadamard transform** (matmul.cpp:265
   `__fp16 tmp[256]`). gs=128 in Gemma-4, fits with margin. Not a bug
   here, but a latent landmine for any model with gs>256.

6. **EMBEDDING op precision propagation** (builder.cpp:1488-1493).
   Already fixed in v2; left as a "checked" item.

The previous diagnosis's H6 (Hadamard precompute removed → kernel falls
back to per-call rebuild) is a **performance** issue, not a numerical
bug. I no longer think it's the cause.

---

## Files referenced

- `/workspace/cactus/cactus-engine/src/complete.cpp:230-258, 515-573, 623-820`
- `/workspace/cactus/cactus-engine/src/model.cpp:154-272, 695-697`
- `/workspace/cactus/cactus-engine/models/gemma4/model_gemma4.cpp:97-111, 215-411`
- `/workspace/cactus/cactus-engine/models/gemma4/model_gemma4.h:319-348`
- `/workspace/cactus/cactus-graph/src/io.cpp:279-413, 777-829`
- `/workspace/cactus/cactus-graph/src/builder.cpp:127-160, 1488-1493`
- `/workspace/cactus/cactus-graph/src/ops_nn.cpp:93-150`
- `/workspace/cactus/cactus-graph/src/ops_tensor.cpp:241-310, 313-360`
- `/workspace/cactus/cactus-graph/src/ops_cache.cpp:70-220`
- `/workspace/cactus/cactus-graph/src/ops_sample.cpp` (sample dispatch)
- `/workspace/cactus/cactus-kernels/src/matmul.cpp:259-305 (Hadamard transform), 1126-1326 (Hadamard matmul), 1428-1496 (Orthogonal matmul)`
- `/workspace/cactus/cactus-kernels/src/nn.cpp:549-739` (sampler kernel)
- `/workspace/cactus/python/src/tqh_prod_convert.py:49-58, 154, 326`
- `/workspace/turboquant/research/tqh_runtime.py:202-348`

Conversion artifact: `/workspace/cactus/test_weights/gemma-4-E2B-it-tqh-prod-v2/` (1.9 GB, 1964 files, 527 CQ + 1437 inherited fp16).

Verifier: `/workspace/cactus/test_weights/verify_tqh_v2.py`.
