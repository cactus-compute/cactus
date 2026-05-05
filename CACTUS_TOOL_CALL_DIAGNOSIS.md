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
