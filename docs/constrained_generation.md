# Constrained Generation: Current Logic, Flaws, and Improvements

## Overview

The constrained generation system enforces structured output during token sampling for tool/function calls. It operates via a **logit bias state machine**: after every generated token, the system computes a bias map that is additively applied to logits before sampling, steering the model toward (or away from) specific tokens based on where it is in the generation structure.

The core implementation lives in:

- `cactus/engine/engine_constraints.cpp` — state machine logic and bias computation (~1,026 lines)
- `cactus/engine/engine.h` — `ToolCallConstrainer` class definition and `ToolConstraintSpec` struct
- `cactus/ffi/cactus_complete.cpp` — schema extraction, `build_tool_constraint_specs`, `setup_tool_constraints`
- `cactus/ffi/cactus_utils.h` — `ToolFunction` struct definition

---

## How It Works

### Entry Point

When `force_tools=true`, `setup_tool_constraints()` is called:

1. Each `ToolFunction`'s JSON schema is parsed by `extract_schema_property_types()` to pull out property names (ignoring types, required fields, enums, etc.)
2. A `ToolConstraintSpec` is created per tool: `{ name, parameter_names[] }`
3. These specs are passed to `Model::set_tool_constraints()` → `ToolCallConstrainer::init()`

### State Machine

The constrainer maintains a `State` enum and transitions between states as tokens are decoded. There are four model-specific state machine variants:

**QWEN (default)**
```
QWEN_START → QWEN_EXPECT_OPEN_BRACE → QWEN_EXPECT_NAME_KEY → QWEN_EXPECT_NAME_COLON
→ QWEN_EXPECT_NAME_VALUE → QWEN_EXPECT_COMMA → QWEN_EXPECT_ARGS_KEY
→ QWEN_EXPECT_ARGS_COLON → QWEN_IN_ARGUMENTS → QWEN_EXPECT_END → DONE
```
Format enforced: `<tool_call>{"name": "<func>", "arguments": {...}}</tool_call>`

**Needle**
```
NEEDLE_START → DONE (then character-level trie validation begins)
```
After `<tool_call>` is seen, switches to a character-feeding parser (`feed_needle_char`) that tracks JSON state and enforces function names and parameter key names via prefix tries.

**Gemma**
```
GEMMA_START → GEMMA_EXPECT_CALL → GEMMA_IN_FUNC_NAME → GEMMA_EXPECT_BRACE
→ GEMMA_IN_ARGUMENTS → GEMMA_EXPECT_END → DONE
```
Format enforced: `<start_function_call>call: <func>{...}<end_function_call>`

**LFM2**
```
LFM_START → LFM_EXPECT_BRACKET → LFM_IN_FUNC_NAME → LFM_EXPECT_PAREN
→ LFM_IN_ARGUMENTS → LFM_EXPECT_BRACKET_CLOSE → LFM_EXPECT_END → DONE
```
Format enforced: `<|tool_call_start|>[<func>(...)]<|tool_call_end|>`

### Logit Bias Mechanism

Two constants define the extremes:
```cpp
constexpr float FORCE_BIAS = 500.0f;   // effectively forces next token
constexpr float BLOCK_BIAS = -500.0f;  // effectively prevents token
```

Soft biases (2.0–15.0) are used in "in arguments" states to encourage JSON structure tokens without hard-blocking free generation. Hard biases are used for structural tokens (opening tags, braces, closing tags) where the format is unambiguous.

After each sampled token, `update()` is called with the decoded text, which advances the state and recomputes the bias map via `compute_bias()`.

### Needle Trie Validation

For the Needle model, parameter key names and function names are stored in prefix tries (`NeedleTrieNode`). During `IN_ARG_KEY` and `IN_NAME` states, the bias computation blocks all tokens that don't form valid continuations in the trie, and allows only tokens that extend a valid prefix or terminate it with `"`.

---

## Current Flaws

### 1. No enforcement of required parameters

**This is the critical missing feature.** The JSON Schema `required` array is never read anywhere in the codebase. `extract_schema_property_types()` only extracts property names from the `properties` object. As a result:

- A model can generate `{"name": "call_api", "arguments": {}}` when `url` is a required argument — this passes all constraints
- The Needle trie prevents invalid key names but cannot require that specific keys appear
- QWEN/Gemma/LFM2 do not constrain argument keys at all — the model freely writes whatever it wants inside the arguments object

`ToolConstraintSpec` needs a `required_parameter_names` field alongside `parameter_names`, and the constraint engine must track which required parameters have been seen and block closing the arguments object until all are present.

### 2. `extract_schema_property_types` is fragile and lossy

The function is a hand-rolled string parser that:
- Only extracts property names and their types — drops `required`, `enum`, `description`, `default`, nested schemas, etc.
- Uses `find('"', ...)` sequentially, which breaks on escaped quotes in property names
- Returns `"string"` as a default type fallback even when no type is specified
- The extracted type is currently unused — it's computed but thrown away by `build_tool_constraint_specs`

The required array is a top-level sibling of `properties`, not nested inside it, and it is never extracted.

### 3. Argument content is unconstrained for most models

For QWEN, Gemma, and LFM2, once the state machine enters the "in arguments" state, only soft biases apply. The model can freely:
- Write parameter keys that don't exist in the schema (hallucinated params)
- Write values of the wrong type
- Skip required parameters entirely
- Close the arguments object prematurely

Only Needle applies trie-based key name enforcement.

### 4. State machine transitions rely on substring search in accumulated text

State transitions like:
```cpp
if (generated_text_.find("name\"") != std::string::npos)
```
accumulate decoded text in `generated_text_` and substring-search it. This is fragile:
- `generated_text_` is never bounded — for long tool calls it grows unboundedly
- Substring matches can trigger on argument values (e.g., an argument value containing `"arguments"` could advance state prematurely)
- No proper escaping awareness means strings containing JSON keywords can confuse the state machine

### 5. Temperature silently overridden

```cpp
if (temperature == 0.0f) {
    temperature = 0.01f;
}
```

When the user requests deterministic generation (`temperature=0`), it is silently changed to `0.01`. This is undocumented and can produce non-deterministic output the caller didn't expect.

### 6. `clear_tool_constraints` reinitializes with empty tool list rather than truly clearing

```cpp
void Model::clear_tool_constraints() {
    tool_constrainer_.reset();
    tool_constrainer_.init(config_.model_type, {}, tokenizer_.get());  // re-init with empty tools
}
```

This sets `active_ = false` (since tools are empty), but it re-tokenizes grammar elements unnecessarily. A simpler deactivation path would be cleaner.

### 7. Grammar element tokenization is re-run on every `init` call

`tokenize_grammar_elements()` encodes all grammar strings on every `init()`, including the full vocabulary scan for Needle (`needle_token_strings_`). For models with large vocabularies, this is expensive and happens even when the tool list changes but the model/tokenizer does not. The vocabulary decode cache (`needle_token_strings_`) does check for size change but the grammar token sets are always rebuilt.

### 8. `QWEN_EXPECT_CLOSE_BRACE` state is dead code

The state `QWEN_EXPECT_CLOSE_BRACE` has a compute_bias handler and a reset path, but examining the update logic for QWEN shows the state machine never transitions into it — there's no case that sets `state_ = QWEN_EXPECT_CLOSE_BRACE`. It was likely a relic of a previous version.

---

## Proposed Improvements

### 1. Extract and propagate `required` from JSON schema

**`ToolConstraintSpec` should carry required parameter information:**

```cpp
struct ToolConstraintSpec {
    std::string name;
    std::vector<std::string> parameter_names;      // all valid param names
    std::vector<std::string> required_parameter_names;  // must appear in output
};
```

**`extract_schema_property_types` (or a replacement) should parse the `required` array:**

The `required` array is a JSON array at the top level of the schema object:
```json
{
  "properties": { "url": {...}, "method": {...} },
  "required": ["url"]
}
```

A proper extraction function should return both the property names and the required list. Replace the hand-rolled parser with a minimal but correct JSON traversal (at minimum, one that handles the `required` array correctly).

### 2. Add required-parameter tracking to the constraint engine

The constrainer needs to track which required parameters have been seen for the current function call. In states where the argument object can be closed (brace depth returns to 0), the closing brace token should be blocked unless all required parameters have been emitted.

For Needle (which already tracks argument keys via `needle_json_state_ == IN_ARG_KEY`), this can be added by recording confirmed argument keys and checking the set against `required_parameter_names` before allowing `}` at the top argument depth.

For QWEN/Gemma/LFM2, a similar character-level tracking pass (like the existing Needle approach) needs to be introduced — or the "in arguments" state needs to track detected key names via buffer scanning.

### 3. Extend Needle-style key enforcement to QWEN

The Needle model benefits from trie-based parameter key validation that QWEN/Gemma/LFM2 lack entirely. A unified approach would apply this to all models: when inside the arguments object at the top level, detect when a key string is starting (`{"` or `,"`), switch to a constrained key-name mode, and block tokens that don't continue a valid parameter name.

### 4. Fix substring-based state transitions

Replace accumulating `generated_text_` + `find()` with a cleaner streaming parser approach:
- Use a fixed-size ring buffer or just track a small suffix of recent decoded text (enough to match the longest expected token sequence)
- Add explicit bounds on `generated_text_` length to prevent unbounded growth
- Use a dedicated parsing state for each structural element rather than relying on substring search across the full accumulated text

### 5. Refactor `compute_bias` into per-model strategy objects

The current `compute_bias()` is a 400-line monolithic function with nested `if/else if` for model type and `switch` for state. This should be split into per-model constrainer implementations sharing a common interface, reducing complexity and making it easier to add or modify individual model behavior without touching unrelated code.

Sketch:
```cpp
class ModelConstraintStrategy {
public:
    virtual void update(const std::string& decoded_text) = 0;
    virtual void compute_bias(BiasMap& bias) = 0;
    virtual void reset() = 0;
    virtual ~ModelConstraintStrategy() = default;
};
```

With concrete subclasses `QwenConstraintStrategy`, `NeedleConstraintStrategy`, `GemmaConstraintStrategy`, `LFM2ConstraintStrategy`.

### 6. Document and make bias constants configurable or at least named

The magic numbers `500.0f`, `-500.0f`, `15.0f`, `10.0f`, `8.0f`, `5.0f`, `3.0f`, `2.0f` should be named constants with comments explaining their purpose. The soft bias values in particular are undocumented and their effect on generation quality is not explained.

### 7. Remove dead `QWEN_EXPECT_CLOSE_BRACE` state

The state is never entered and its bias handler will never execute. Remove it to reduce confusion.

### 8. Don't silently mutate temperature; surface it as a warning

Rather than silently changing `temperature = 0.0f` to `0.01f`, either document this behavior clearly or surface it as a logged warning. Callers using `temperature=0` for determinism deserve to know their setting was overridden.

---

## Priority Order

1. **Extract `required` from schema and propagate to `ToolConstraintSpec`** — most impactful correctness fix; tools can silently produce invalid calls without this
2. **Block arguments close until required params are satisfied** — the enforcement half of item 1, needed for the guarantee to be meaningful
3. **Extend key-name enforcement to QWEN/Gemma/LFM2** — prevents hallucinated parameter names for all models
4. **Fix unbounded `generated_text_` accumulation** — correctness and memory issue
5. **Refactor `compute_bias` into per-model strategies** — maintainability, enables items 2 and 3 cleanly
6. **Remove dead state, document temperature override** — cleanup
