# Cactus tool-calling garbled output — root-cause diagnosis

## Executive summary

Cactus's hand-rewritten Gemma-4 prompt formatter (`gemma_tools.h::escape()` and `format_gemma4_style`) emits the literal string `<escape>...<escape>` to delimit string literals inside tool declarations. **The Gemma-4-E2B-it tokenizer has no `<escape>` token** — the actual escape token is `<|"|>` (id 52). At inference time `<escape>` is BPE-fragmented into three garbage tokens (`<` 236820 / `escape` 44732 / `>` 236813), so every quoted field of every tool declaration the model sees is wrapped in 6 wrong, off-distribution tokens. Roughly 20+ such fragments per tool turn the entire tool-declaration block into noise relative to what the model was trained on, and the assistant response degenerates accordingly. This is a pure C++ formatter bug; the TQH4 weights and BPE tokenizer are fine. Three smaller drift bugs (system-content trim, missing `\n` after `<|think|>`, and the `force_tools=false` default that disables the grammar safety net) compound the problem. The cleanest long-term fix is to delete the C++ rewrite entirely and apply the shipped `chat_template.jinja` via a tiny Jinja runtime; the minimal short-term fix is a 1-line change in `gemma_tools.h` plus a 1-line change in `tokenizer.cpp`.

## Hypothesis-by-hypothesis findings

### H1 — Prompt drift between `format_gemma4_style` + `gemma::format_tools` and `chat_template.jinja`. **CONFIRMED, severe.**

Built `/workspace/cactus/diagnose_tool_call.py` that mirrors the C++ logic in Python and diffs against `tokenizer.apply_chat_template(...)` ground truth.

For one trivial weather tool the official prompt is **104 tokens**; Cactus's is **129 tokens** (+24%). The structural template (`<bos>`, `<|turn>system\n`, `<|tool>...<tool|>`, `<turn|>\n`, ...) lines up byte-for-byte. The only divergence is inside `format_function_declaration`, where every field that is supposed to be wrapped by the escape-quote token `<|"|>` (id 52) is instead wrapped by the literal string `<escape>`.

Concrete repr of the wrapped portion:

- Official: `description:<|"|>Get current weather for a city.<|"|>`
- Cactus:   `description:<escape>Get current weather for a city.<escape>`

In token ids the official sees `[..., 7777 'description', 236787 ':', 52 '<|"|>', 3407 'Get', ...]`. Cactus produces `[..., 7777 'description', 61916 ':<', 44732 'escape', 236813 '>', 3407 'Get', ...]`. There is a 3-token noise burst on every open and every close. With ~10 escape pairs in a one-parameter tool, that is ~40 garbage tokens per tool. With multiple tools it scales linearly.

The C++ code is in two places:

- `/workspace/cactus/cactus-engine/src/gemma_tools.h:18-20` — `inline std::string escape(const std::string& s) { return "<escape>" + s + "<escape>"; }`
- Used 11 times inside `format_parameters` and `format_function_declaration` in the same file (lines 221, 232, 259, 263, 300, 307, 318, 331, 363, 374).

The Jinja ground truth uses `<|"|>` directly (lines 11, 17 (via format_argument), 37, 75, 82, 87, 97, 103, 110, 113, 120, 130 of `chat_template.jinja`).

A second drift exists in the same macro: when there is **no `properties` block but the params object is empty/missing**, Jinja still emits `<|tool>declaration:NAME{description:<|"|>...<|"|>}<tool|>` (no `parameters:` key at all). Cactus's `format_function_declaration` does the same thing because it gates on `params_json.empty()`. Behaviour matches in this edge case. The dominant bug is the escape token.

A third drift: the Jinja macro uses `tool_data['function']['parameters']` and emits the **trailing `}`** for the `parameters:` block only inside the `type:` clause (chat_template.jinja:103). The C++ unconditionally appends `}` after the inner block (`gemma_tools.h:377`). Currently both produce one `}` because the C++ does not emit `type:` separately and the Jinja does — this happens to balance out in every input we tried, but it is fragile to a tool with `parameters` but no `type` field (Jinja would unbalance). Low priority.

### H2 — Special-token tokenization. Confirmed: `<escape>` is NOT a special token; everything else is.

Direct check via `tokenizer.encode(s, add_special_tokens=False)`:

```
'<bos>'              -> [2]       OK
'<|tool>'            -> [46]      OK
'<tool|>'            -> [47]      OK
'<|tool_call>'       -> [48]      OK
'<tool_call|>'       -> [49]      OK
'<|tool_response>'   -> [50]      OK
'<tool_response|>'   -> [51]      OK
'<|turn>'            -> [105]     OK
'<turn|>'            -> [106]     OK
'<|think|>'          -> [98]      OK
'<|channel>'         -> [100]     OK
'<channel|>'         -> [101]     OK
'<|"|>'              -> [52]      OK    <-- ID is 52, not 113 as the brief speculated
'<escape>'           -> [236820, 44732, 236813]   *** MULTI-TOKEN, 3 tokens of garbage ***
```

`<bos>` does tokenize correctly (it is in `added_tokens` with `special:true` and the HF tokenizer round-trips it as a single token), so the C++ `result = "<bos>"` literal at `tokenizer.cpp:378` is fine in the runtime — provided the Cactus C++ tokenizer's encode path also recognises it as a special token. Worth a one-line guard, but not urgent. The `<escape>` case is a guaranteed regression on every prompt with tools.

### H3 — `force_tools` and the grammar safety net. Confirmed dormant by default.

`/workspace/cactus/cactus-engine/src/complete.cpp:93-95`:

```cpp
void setup_tool_constraints(CactusModelHandle* handle, const std::vector<ToolFunction>& tools,
                           bool force_tools, float& temperature) {
    if (!force_tools || tools.empty()) return;
    ...
```

The bias-mask grammar that constrains output to `<|tool_call>...<tool_call|>` only runs when `force_tools=true`. Default is false. So even if the prompt were correct, an unconstrained sample on a borderline-confident model with broken declarations is hopeless. The `tests/test_llm.cpp::test_tool_call` test passes `force_tools: true` in its options — that is why the test does not hit this in CI. Real users almost certainly do not.

### H4 — Tokenizer-converter omission. Mostly fine.

`python/src/tokenizer.py:250-258` reads every entry of `tokenizer_json_data["added_tokens"]` and registers them as special tokens. All 13 pipe-tag tokens (ids 2, 46–52, 98, 100, 101, 105, 106) are in that list with `special:True`, so they make it into Cactus's special-token table. No fix needed there.

The hardcoded `tool_related` list at `python/src/tokenizer.py:345-353` is an irrelevant secondary mechanism — it only adds tokens that exist in `tokenizer_config.json::added_tokens_decoder`, and again the pipe-tag ones are already covered by line 250's loop. Notably this list still includes `<escape>` even though that string does not exist in the Gemma-4 vocab; this is dead code for Gemma-4 and harmless, but should be removed when fixing H1 to avoid future confusion.

### H5 — Stop sequences. Looks correct.

`complete.cpp:251` adds `<turn|>` (id 106), and `complete.cpp:253-254` adds `<tool_call|>` (49) and `<|tool_response>` (50) when tools are present. All three are single-token. Provided the model emits them, stopping works. Not a bug.

### Other drift in `format_gemma4_style` (cosmetic but real)

- `tokenizer.cpp:383, 392`: `sys_content` is taken straight from `messages[0].content`; Jinja does `messages[0]['content'] | trim`. Will produce off-by-N tokens whenever the system message has leading/trailing whitespace.
- `tokenizer.cpp:390`: emits `<|think|>`; Jinja emits `<|think|>\n`. Tokenizes differently.
- `tokenizer.cpp:458`: user-message content is appended raw. Jinja does `message['content'] | trim`.
- `tokenizer.cpp:442`: model-message content goes through `strip_channel`. Jinja goes through `strip_thinking`, which does `trim()` at the end. Off-by-whitespace at every assistant turn boundary.
- `tokenizer.cpp:466`: emits `<turn|>\n` after every turn, including assistant turns whose only content was a tool_call. Jinja conditionally emits `<|tool_response>` instead of `<turn|>\n` when the turn ended with a tool_call (`chat_template.jinja:332-336`). This is a **bigger** discrepancy on multi-turn agent traces; it is not exercised by the simple weather-tool test but matters for any actual agent loop.

## Recommended patches

### Patch 1 — fix the escape token (1 line, the actual root cause)

`/workspace/cactus/cactus-engine/src/gemma_tools.h:18-20`:

```diff
 inline std::string escape(const std::string& s) {
-    return "<escape>" + s + "<escape>";
+    return "<|\"|>" + s + "<|\"|>";
 }
```

Note the C++ tokenizer must recognise `<|"|>` as token id 52. Verified above: it is in `added_tokens` with `special:true`, so the converter (`python/src/tokenizer.py:250`) will register it.

This single change removes ~25% of the prompt-token bloat and aligns every tool declaration byte-for-byte with the chat template for the common case.

The corresponding output-side parser at `gemma_tools.h:407-419` already accepts both `<escape>` and `<|"|>`, so legacy parses still work.

### Patch 2 — make `<bos>` defensible (1 line)

`/workspace/cactus/cactus-engine/src/tokenizer.cpp:378`:

The literal `"<bos>"` is fine *if and only if* the Cactus tokenizer treats it as a special token during `encode`. If the C++ encoder ever stops doing that, this silently fragments. Defensive option: emit the BOS token id directly during `encode` rather than as a string. Lower priority than Patch 1.

### Patch 3 — match Jinja whitespace (a few lines)

`/workspace/cactus/cactus-engine/src/tokenizer.cpp`:

```diff
-    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
-        sys_content = messages[0].content;
-        first_msg = 1;
+    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
+        sys_content = trim(messages[0].content);     // helper or inline trim
+        first_msg = 1;
     }
@@
-        result += "<|think|>";
+        result += "<|think|>\n";
@@
-            result += msg.content;
+            result += trim(msg.content);             // user / tool turns
@@
-            result += strip_channel(msg.content);
+            result += trim(strip_channel(msg.content));   // model turns
```

(`trim` should match Python `str.strip()` — strip leading and trailing ASCII whitespace including `\n`, `\r`, `\t`, space.)

### Patch 4 — clean up Python converter dead code

`/workspace/cactus/python/src/tokenizer.py:345-353` — remove `<escape>` from `tool_related` and remove the `<start_function_*>` / `<end_function_*>` fallback lookup at lines 261-303 that targets Gemma-3-style tokens (Gemma-4 uses pipe tags). Optional cleanup; not load-bearing.

### Patch 5 — document `force_tools`

The README / `complete.cpp` doc comment for `force_tools` should call out: **without `force_tools=true`, no grammar constraint is applied; the model is free to emit prose instead of `<|tool_call>...`**. This is the user-facing footgun. If Cactus wants to make tool-calling robust by default, change the default at `complete.cpp:445` to true whenever `tools.size() > 0`, or always run the structural FSM constraint that limits the model to `<|tool_call>` / prose-then-`<turn|>` once tools are present.

### Patch 6 (recommended, larger) — replace the C++ rewrite with the actual `chat_template.jinja`

Even after Patches 1+3 there are still drift surfaces:

- multi-tool with continuations,
- assistant turns with `tool_calls` followed by `tool` messages,
- assistant turns where the model emitted `<|channel>thought\n...<channel|>` reasoning,
- the `<|tool_response>` vs `<turn|>` choice at `chat_template.jinja:332-336`,
- the `format_argument` / `format_parameters` handling of arrays-of-objects,
- the `nullable` / response-block fields.

Maintaining a parallel C++ rewrite of a 343-line Jinja file across model upgrades is not sustainable. Recommend pulling in [`inja`](https://github.com/pantor/inja) or [`minja`](https://github.com/google/minja) (the latter is what llama.cpp uses for chat templates) and loading `chat_template.jinja2` from the converted model dir. The Python converter already writes that file (`python/src/tokenizer.py:308-311`). Then `format_chat_prompt` becomes ~10 lines that call into the Jinja engine. This is the durable fix and would have prevented this regression entirely.

## Test recipe Karen can run to verify the fix

1. Apply Patch 1 (the 1-line `escape()` change).
2. Rebuild the cactus engine.
3. Re-run the diagnostic harness and confirm token count and content match:

   ```bash
   cd /workspace/cactus
   python3 diagnose_tool_call.py | grep -E "Official: |Cactus: |MULTI-TOKEN"
   # Expect: identical token counts (104 == 104 with the patch translated to Python).
   ```

4. Add a unit test under `cactus-engine/tests` that compares the C++ output of `format_gemma4_style` + `format_tools` against the official rendered prompt for the same `(messages, tools)`. Sketch:

   ```cpp
   TEST(GemmaToolFormat, MatchesOfficialChatTemplate) {
       std::string expected = read_file("tests/data/gemma4_weather_tool_prompt.txt"); // dump from HF tokenizer
       auto formatted_tools = gemma::format_tools(tools, true);
       auto cactus_prompt = tokenizer.format_chat_prompt(messages, true, formatted_tools, false);
       ASSERT_EQ(cactus_prompt, expected);
   }
   ```

   The fixture file is one `tokenizer.apply_chat_template(...)` call in Python — already produced by the diagnostic harness as the "OFFICIAL prompt" line. Lock it in to prevent regressions.

5. End-to-end: run BFCL with native tool tokens (not the JSON-in-user-message scaffold from `eval_bfcl.py`). Pass `force_tools: true` and a single tool from the `simple` BFCL split. The model should emit `<|tool_call>call:NAME{...}<tool_call|>` cleanly. Without the patch this currently produces garbled prose because of the `<escape>` token contamination of the declaration block.

## Files referenced

- `/workspace/cactus/cactus-engine/src/gemma_tools.h` — Patch 1 site, line 18-20.
- `/workspace/cactus/cactus-engine/src/tokenizer.cpp` — Patch 3 sites, lines 378, 383, 390, 442, 458.
- `/workspace/cactus/cactus-engine/src/complete.cpp` — `force_tools` gating, lines 93-95, 444-445.
- `/workspace/cactus/python/src/tokenizer.py` — Patch 4 cleanup, lines 261-303 and 345-353.
- `/workspace/model/.../chat_template.jinja` — ground truth.
- `/workspace/cactus/diagnose_tool_call.py` — regression harness (created by this investigation).

## Decode-time collapse — diagnosis (round 2)

### Recap

After Patch 1 (the `<escape>` -> `<|"|>` fix) shipped, Karen's TQH4-quantised
`google/gemma-4-E2B-it` running on Cactus branch `karen/tq-v2-debugging`
(HEAD `171d27db v2 latest`) still produces collapsed / repetitive output
on **non-tool prompts** as well. Symptoms (from Karen's harness):

- streaming: `Turn 1: I am doing well. I am doing well. I am doing well. ...` (6× exact phrase repetition, then partial recovery on Turn 2)
- prefill (warm): `It refers to a state of content that describes.` (truncated but coherent)
- prefill (cold): `Obsessive: Obsessive: Obsessive: ... online online online fixation.` (single-token loop that escapes once)
- tool_calls (output side): `<|tool_call>call:<|tool_call>call:getWeather:getWeather:...`

Default sampling on Gemma-4 mm in this branch is `T=1.0, top_p=0.95, top_k=64,
min_p=0.15, repetition_penalty=1.1` (`cactus-engine/src/model.cpp:695-697`,
`cactus-engine/models/gemma4/model_gemma4.h:319`). These defaults are healthy
enough that 6× exact-token loops do not happen on a numerically intact
forward pass — repetition_penalty=1.1 alone subtracts `log(1.1)≈0.095` per
already-seen token, which is enough to break a tie unless the top logit
is dominating by a lot. So: **the symptom is consistent with the logit
distribution being genuinely degenerate — single-token mass — not with a
sampler bug.**

### Things that turned out NOT to be the bug

#### F1. The fused INT4 dense-MLP commit (`a53f0ce4`). NOT ACTIVE.

Karen's `171d27db v2 latest` already reverted the fused-MLP call site:
`cactus-engine/models/gemma4/model_gemma4.cpp:312-319` is now the unfused
`gate = gelu(matmul); up = matmul; multiply; matmul(down)` chain. The
`compute_dense_mlp_int4_fused_node` op definition was also removed from
`cactus-graph/src/ops_nn.cpp` and the `dense_mlp_int4_fused` builder API
was removed from `cactus-graph/src/builder.cpp`. Searching for any remaining
reference to `dense_mlp_fused`, `cactus_dense_mlp_fused`, or
`DENSE_MLP_INT4_FUSED` returns nothing. Dead code, not on the decode path.

#### F2. The "inner 1 fast path" commit (`d917981f`). Not exercised on this codepath.

That commit added `cactus/kernel/kernel_reduce.cpp` in the now-deleted layout
(the cactus-graph / cactus-kernels / cactus-engine refactor in `b8527fe0
Aggressive Refactor!` blew away the `cactus/kernel` directory). On
`171d27db`, `cactus-kernels/src/reduce.cpp` has no `inner == 1` fast path;
all axis_reduce paths go through `axis_reduce_*_impl`. So the inner-1
optimisation is not in this branch.

#### F3. The new `cactus_quant_orthogonal_matmul`. CORRECT.

This is the kernel that runs on the LM head every decode step (Gemma-4
has tied embeddings — `cactus-engine/models/gemma4/model_gemma4.cpp:99-101`
maps `output_weight` to `embedding_node_id_`, and `embed_tokens` is the
sole orthogonal-rotation layer in TQH4 — `metadata.json` shows
`{'hadamard': 526, 'orthogonal': 1}` and the orthogonal one is
`embed_tokens` with `rotation_group_dim=1536, rotation_family=orthogonal`).

I verified the kernel against the trusted QDQ reference
(`/workspace/turboquant/research/tqh_runtime.py`) using the actual packed
weights at `/workspace/turboquant/artifacts/packed_av/PROD_v2_a2_L4_pli2_emb4`.
For `M=4, N_slice=1024, K=1536` random fp16 activations:

```
=== orthogonal matmul vs reference (fp32) ===
max abs diff: 2.533e-07     mean abs diff: 5.514e-08
--- with fp16-cast intermediates (mimicking the kernel) ---
max abs diff: 2.414e-04     mean abs diff: 3.833e-05
```

The 2.4e-4 max abs error is well inside the 3.9e-3 packed-vs-QDQ tolerance
that Karen verified globally. The math at
`cactus-kernels/src/matmul.cpp:1428-1496` is correct: it computes
`A_rot[m,i] = sum_k (A[m,k] * isr[k]) * R[k,i]` then
`C[m,n] = norm[n] * sum_i cb[idx[n,i]] * A_rot[m,i]`, which expands
algebraically to `y = A @ W^T` with `W[n,k] = norm[n] * (cb[idx[n,:]] @ R^T)[k] * isr[k]`
exactly matching `tqh_runtime.dehydrate_layer`'s
`(dq_g @ R.T) * norms ; recon /= input_scale`.

Index unpacking (LSB, 4-bit nibble: low for even k, high for odd k) matches
the convert's `pack_indices_lsb` packing.

#### F4. `dequantize_orthogonal_embedding_row`. CORRECT.

Same algebra as F3, used for the input embedding lookup. Matches reference.
`ops_tensor.cpp:241-267`.

#### F5. The `gemma4_scale_factor` baking 1/16 into `embed_tokens` norms. INTENTIONAL.

`tqh_prod_convert.py:49-58` applies `1/16` to `token_embeddings`,
`output_weight`, `embed_vision_proj`, `embed_vision_embedding` and `*16`
to `ffn_gate`, `ffn_up`, `per_layer_gate`, `moe_gate_proj`, `moe_up_proj`.
This matches the existing Cactus convention in
`python/src/tensor_io.py:154-172`, and Cactus's Gemma-4 forward graph is
designed against pre-divided embeddings (the embed lookup is multiplied
by `sqrt(hidden_dim)` at `model_gemma4.cpp:394-395`, a layout choice that
assumes `embed_tokens` is stored already divided by 16). Not a regression.

#### F6. Sampler ignoring `repetition_penalty` parameter. RED HERRING.

`cactus-kernels/src/nn.cpp:585` does `(void)repetition_penalty;` inside
`cactus_sample_f32_ex` / `cactus_sample_f16_ex`. This looks alarming but
is intentional: `Model::sample_token` at `cactus-engine/src/model.cpp:265-269`
applies repetition penalty as an additive logit bias against the token
history *before* invoking `gb->sample_with_options`, which feeds the bias
into the kernel via the `bias_indices/bias_values` arrays at
`ops_sample.cpp:18-20` and `nn.cpp:564-571`. The penalty IS applied.

#### F7. `OpType::EMBEDDING` precision-cast bug (FIXED in this branch).

Karen's `v2 latest` adds `OpType::EMBEDDING` to the list of ops whose
output precision is taken from `params.output_precision` rather than
inherited from `inputs[0]`
(`cactus-graph/src/builder.cpp:1488-1493`). Before this fix, an embedding
op consuming a CQ4 weight would have allocated its output buffer at CQ4
precision (i.e. 4-bit packed = ½ byte per element) but
`compute_embedding_node` writes fp16 (2 bytes per element), guaranteeing
heap corruption in the output node. With the fix, the output is allocated
correctly as FP16. Confirmed correct in current HEAD.

### Top remaining hypotheses (in priority order)

I cannot run the engine end-to-end here, but the symptoms point at a
small number of remaining sites. Listed by likelihood given the evidence.

#### H6 (most likely). The Hadamard `cactus_quant_matmul` activation transform on M=1 decode.

Every transformer matmul on the decode step (q, k, v, o, gate, up, down,
per_layer_*) goes through `cactus_quant_matmul` at
`cactus-kernels/src/matmul.cpp:1126-…`. Two things I observed without
being able to execute:

1. The "M=1 thread-local scratch" branch at `1157-1163` reuses
   `tl_code_basis`/`tl_act_i8`/`tl_act_scales` across calls. The same
   thread-local buffers are used by the sparse-activation
   matmul kernels at lines 785/800/922/1036 and by 64+ other call sites
   that all also call `cactus_quant_transform_hadamard_activations`. If
   any caller during a single `gb->execute()` resets `M` between the
   M=1 fast path and a later M>1 batched path *on the same thread*,
   the thread-local size check is `if (size < act_size) resize`, so
   it's a one-way grow — that's fine. But all these paths are taken
   per-layer per-step. A subtle race between the per-layer thread pool
   and the M=1 thread-local scratch is plausible if the M=1 path is
   ever entered re-entrantly from worker threads (it normally is not,
   but the new `parallel_for` in the orthogonal matmul does enqueue
   work to the same pool from the master thread).

2. Karen's `v2 latest` explicitly removed the precomputed `cq_expanded`
   and `cq_norm_f32` block from `cactus-graph/src/io.cpp` (the
   `tq_expand_i8_16` precompute previously in `mmap_weights`). The
   matmul kernel has a fallback at lines 1188-1224 that re-builds those
   structures inside the kernel call, so correctness is preserved, but
   it is dramatically slower and runs on every matmul on every step.
   That fallback path is now hot. Worth verifying that `w_il_buf` /
   `n_f32_buf` are sized correctly for non-aligned `W->N` (the
   `(W->N + 3) / 4` rounding at line 1183 plus the `valid_n` masking
   at 1195/1199 should be correct, but this code is now exercised on
   every layer instead of being precompute-and-cached, so any bug here
   hits every forward).

   **Concrete risk**: line 1220 stores
   `nd[ni] = (n_start+ni < W->N) ? float(W->norms[(n_start+ni)*num_groups+g]) * cb_scale : 0.f;`
   but the deleted precompute at the equivalent site previously
   applied this exact transform too. If `cb_scale` here is computed
   from `tq_quantize_codebook_i8(W->codebook, cb_i8, 1u << bits)` per
   call (line 1139) and the codebook in fp16 is read from mmap, fp16
   rounding of the codebook can perturb `cb_scale`, and the thread that
   loses the race could see a slightly different `n_f32` than the one
   the dot product was scaled against. **Recommend**: rebuild the
   precompute block — it was both faster and provided a single,
   stable cb_scale per layer.

#### H5. `cactus_quant_matmul` M=1 fast path with permutation when `cq_permutation` is null vs non-null.

`cactus_quant_transform_hadamard_group` at `matmul.cpp:259-305` writes
into `tmp[256]` (a 256-element stack array) when permutation is enabled.
For `gs > 256` (which TQH never uses but the audio tower has matmuls of
varying sizes), `tmp` would overflow. TQH4 weights all have gs=128 so
this is presumably safe in practice, but the assertion is missing.

#### H4. KV cache append on cached decode.

`gb->kv_cache_append` is called twice per layer at decode
(`model_gemma4.cpp:300-301`). I cannot inspect the kernel here without
chasing further, but the pathological symptom of `Turn 2: Hello, I am
doing well.` repeating from Turn 1 is consistent with **cache contents
leaking across turns** (specifically, a scenario where the cache from
turn 1 is not reset and turn-2 attention also reads those positions,
biasing the model toward repeating the previous turn's tokens).
Worth a quick audit of `compact_kv_cache()` at `model_gemma4.cpp:66`
and the cache-reset path in the user-facing API.

#### H3. Re-quantising activations to int8 at every group with codebook range collapse.

`tq_quantize_group_i8` at the matmul (line 1177) computes per-group
scale `max(abs)/127`. For the **decode step** with M=1, activations
arriving at the down-projection input are produced by `gate*up` after
gelu, and then quantised group-by-group. If gelu produces a single
huge spike (one group dominates the abs-max), the rest of the group
gets clipped to ±0 in int8 → information collapse. With the deleted
fused-MLP path that did per-tile rescaling, this would have been less
catastrophic. Now every layer's down-proj input goes through global
group-wise int8 quantisation. Not sure this is the bug, but it is
at minimum a quantisation-noise floor that gets exercised hard.

#### H2. The new orthogonal-rotation layout in `mmap_embeddings` overwriting `cq_left_signs/right_signs/permutation` with `cq_rotation`.

`io.cpp:301-311` sets `buffer.cq_rotation` and `buffer.cq_flags = ORTHOGONAL`
in the orthogonal branch, but does not zero out
`cq_left_signs/cq_right_signs/cq_permutation`. They default-init to
`nullptr` in `BufferDesc` so this is fine for a fresh buffer. Defensive
zeroing is recommended for safety against future refactors but not the
bug.

### Smoking-gun candidate

I could not run the engine, so I cannot point at a single line and say
"this is wrong." The strongest signal is **H6**: the deletion of the
`cq_expanded` / `cq_norm_f32` precompute from `io.cpp` puts the kernel
on the slow per-call rebuild path that previously was only a fallback,
and that path applies a per-call `cb_scale` rounding which can drift
between calls. The unit tests for `cactus_quant_matmul` in
`cactus-kernels/tests/test_matmul.cpp` mostly run with the precompute
populated by hand from the test, so they would not catch this regression.

The repetition-then-recovery patterns ("Obsessive: Obsessive: ...
online online online fixation") are characteristic of greedy-ish decoding
on a forward pass that is producing logits within a few tens of millivolts
of each other but slightly off the true distribution — exactly what you
get from a kernel that is computing the right thing in expectation but
drifting by ~1e-3 per matmul, compounding over 30+ transformer layers.

### Recommended fixes & test recipes

#### R1. Restore the precomputed `cq_expanded` / `cq_norm_f32` in `io.cpp::mmap_weights`.

The block deleted in `171d27db` (the `~120` lines starting "for (size_t nb = 0;
nb < N_blocks; ++nb)" inside the `is_cq && group_size > 0` branch of the
old `mmap_weights`) needs to come back. It populated
`buffer.cq_expanded` and `buffer.cq_norm_f32` once per layer at load time
using a single global `cb_scale = max(|cb|)/127`. Reinstate it, run the
existing test, and confirm `W->expanded != nullptr` at every matmul call
site by adding a one-line assertion.

If perf was the reason it was removed, do the work in a worker pool:

```cpp
// in mmap_weights, after parsing scales blob:
if (PrecisionTraits::is_cq(precision) && group_size > 0 && !is_orthogonal) {
    populate_cq_expanded_and_norm_f32(buffer);   // the old precompute, lifted into a helper
}
```

#### R2. Add a pure-Python kernel-equivalence test for the Hadamard `cactus_quant_matmul`.

We already have one for the orthogonal kernel (this conversation produced
it; the script is in the prior shell history). Mirror it for Hadamard
weights using `tqh_runtime.dehydrate_layer` with `rotation_family=hadamard`,
group_size=128, sign tables matching `make_hadamard_components`. Do it for
*every* layer shape that the model uses (q_proj, k_proj, ffn_gate, etc.)
so that any drift between the C++ kernel and the reference math is caught
at unit-test time, not at end-to-end token time.

#### R3. Replace decode-time greedy with `T=0.0` exact greedy and a logit dump.

To unambiguously locate the decode bug:

```python
# In the test harness, run:
#   1. cactus.complete(prompt, T=0.0)  -> dump first 16 logits-argmax tokens
#   2. transformers.AutoModelForCausalLM.from_pretrained(dehydrated_dir).generate(..., do_sample=False, max_new_tokens=16)
# Compare token ids step by step. The first divergence is where the bug lives.
```

Karen's QDQ reference produces "The capital of France is **Paris**." which
matches HF transformers. So the dehydrated bf16 path is bit-equivalent at
the token level. Cactus must match the same trajectory at T=0; any
divergence at step N localises the bug to the layer/op active at that step.

#### R4. Sanity-check the LM-head logits scale and softcap interaction.

`final_logit_softcapping` at `model_gemma4_mm.cpp:308-313` does
`logits = softcap * tanh(logits / softcap)`. If `embed_tokens` norms are
divided by 16 (per F5) and the LM-head matmul does NOT undo that 1/16,
logits exit the matmul at 1/16 of their natural magnitude. With the
softcap value the config likely uses (30), the post-softcap range is
identical (since tanh saturates at ±1 regardless), but the *relative*
ordering of logits inside the linear region of tanh is preserved. So
this is **NOT** a functional bug — but it IS odd that the same physical
tensor `embed_tokens` is used both as an input embedding (where the
1/16 is undone by `*sqrt(hidden_dim)≈48` giving a ×3 net) and as the
LM head (where there is no compensating multiplier).

Recommend: dump cactus's logits for a fixed prompt, dump HF's logits for
the same prompt against the dehydrated checkpoint, and compare. If the
ratio is 1/16 across the board, you have your missing factor; if it is
random per-token, the bug is upstream of the LM head.

#### R5. KV-cache state inspection between turns.

For the `streaming` failure pattern (Turn 1 6× repeats, Turn 2 succeeds
with a residual hello), instrument `compact_kv_cache()` and the cache
reset between turns. Confirm `cache_total_seq_len_` is reset at turn
boundary and that `graph_cache_k_nodes_` are zeroed (or that the KV
slots beyond `cache_total_seq_len_` are masked out by attention).

### Files referenced (round 2)

- `/workspace/cactus/cactus-engine/models/gemma4/model_gemma4.cpp:312-319` — fused-MLP call site already reverted
- `/workspace/cactus/cactus-graph/src/io.cpp:279-380` — new mmap_weights / mmap_embeddings layout (precompute deleted, see R1)
- `/workspace/cactus/cactus-graph/src/ops_nn.cpp:117-150,180-196` — orthogonal-vs-Hadamard matmul dispatch
- `/workspace/cactus/cactus-graph/src/ops_tensor.cpp:241-309` — orthogonal & Hadamard embedding row dequant (verified correct)
- `/workspace/cactus/cactus-graph/src/builder.cpp:1488-1493` — EMBEDDING op output-precision fix (already in v2 latest)
- `/workspace/cactus/cactus-kernels/src/matmul.cpp:1126-1300` — Hadamard `cactus_quant_matmul` (slow path now hot, see H6 / R1)
- `/workspace/cactus/cactus-kernels/src/matmul.cpp:1428-1496` — `cactus_quant_orthogonal_matmul` (verified correct)
- `/workspace/cactus/cactus-kernels/src/nn.cpp:549-739` — sampler kernels (verified bias-based rep_pen path)
- `/workspace/cactus/python/src/tqh_prod_convert.py` — TQH packed -> cactus weights converter (verified scale_factor convention matches `tensor_io.py`)
- `/workspace/turboquant/research/tqh_runtime.py:200-253` — reference dehydrator
- `/workspace/turboquant/research/verify_packed_av.py` — Karen's bit-equality verification (passes globally)
