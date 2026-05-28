# KV Cache Implementation Plan

This is an execution plan for implementing the KV cache policy described in `CACHE.md`. It is written for an agent working in the Cactus repo. Keep the implementation small, explicit, and aligned with existing cache paths.

## Goal

Implement bounded KV cache behavior with llama.cpp-style user controls:

```text
-c, --ctx-size N
--context-shift / --no-context-shift
--keep K
--keep-prompt / --no-keep-prompt
```

Defaults:

```text
N = 16384
M = N / 2
K = M / 2 = N / 4
context_shift = true
keep_prompt = true
```

Compaction rule:

```text
if keep_prompt:
  prefix_keep = min(prompt_len, K)
else:
  prefix_keep = min(cache_len, K)

tail_keep = M - prefix_keep
new_cache = kept_prefix + most_recent(tail_keep)
```

Fixed-window rule:

```text
if context_shift is false and cache_len + incoming > N:
  fail clearly
```

## Existing Code Map

### CLI

Files:

- `python/cactus/cli/__init__.py`
- `python/cactus/cli/run.py`
- `python/cactus/cli/convert.py`
- `python/cactus/cli/model.py`

Current state:

- `convert` exposes `--cache-context-length`.
- `run` exposes `--cache-context-length` for auto-conversion.
- `TranspileOptions` in `python/cactus/cli/model.py` carries `cache_context_length`.
- `ensure_bundle()` passes `--cache-context-length` into `run_transpile()`.

Needed:

- Add `-c, --ctx-size`.
- Keep `--cache-context-length` as a compatibility alias.
- Add `--context-shift` / `--no-context-shift`.
- Add `--keep`.
- Add `--keep-prompt` / `--no-keep-prompt`.
- Normalize and validate the policy once before passing it onward.

### Transpile Policy and Metadata

Files:

- `python/cactus/transpile/model_adapters.py`
- `python/cactus/transpile/hf_model.py`
- `python/cactus/transpile/optimize_graph.py`
- `python/cactus/transpile/lower.py`

Current state:

- `_parse_cache_context_length()` parses the existing cache-length input.
- `_cache_context_length()` defaults to model config when explicit length is absent.
- `_max_cache_seq_len()` currently returns at least `1024`.
- Component specs write metadata:

```text
max_cache_seq_len
cache_sink_size
prefill_chunk_size
```

- `lower.py::_lower_attention_with_internal_kv_cache()` reads:

```text
max_cache_seq_len
cache_sink_size
window_size
```

and emits:

```text
g.kv_cache_state(max_cache_seq_len, ..., window_size, sink_size)
g.kv_cache_append(..., window_size, sink_size)
```

Needed:

- Replace the loose cache-length parsing path with a small normalized cache policy.
- Default to `ctx_size=16384`, not model config, unless the user explicitly requests `auto`.
- Carry policy fields through component graph metadata.
- Keep model-specific sliding-window attention metadata separate from cache compaction policy.

### Graph Builder and Serialized Params

Files:

- `cactus-graph/cactus_graph.h`
- `cactus-graph/src/builder.cpp`
- `cactus-graph/src/param_io.cpp`
- `cactus-engine/cactus_engine.h`
- `cactus-engine/src/graph_ffi.cpp`
- `python/cactus/bindings/graph.py`
- `python/cactus/bindings/cactus.py`

Current state:

- `OpParams` has:

```text
window_size
max_cache_seq_len
cache_sink_size
```

- `KV_CACHE_STATE` serializes:

```text
MaxCacheSeqLen
NumKvHeads
HeadDim
WindowSize
CacheSinkSize
```

- `KV_CACHE_APPEND` serializes:

```text
WindowSize
CacheSinkSize
```

Needed:

- Add explicit parameter names for new behavior instead of overloading `window_size` if possible.
- Suggested new fields:

```text
cache_compact_to
cache_keep
cache_prompt_len
cache_flags
```

Where `cache_flags` can encode:

```text
context_shift enabled
keep_prompt enabled
```

Alternative: separate booleans are more readable, but `ParamField` serialization already has simple scalar fields. Prefer readability unless this creates a large diff.

Important: graph file format changes require updating `ParamField`, write/read switch statements, op schemas, FFI signatures, and Python binding signatures together.

### Runtime Cache Ops

File:

- `cactus-graph/src/ops_cache.cpp`

Current state:

- `CacheMetadata` layout is:

```text
current_seq_len
max_seq_len
num_kv_heads
head_dim
sink_size
reserved[3]
```

- `compute_kv_cache_state_node()` initializes the metadata.
- `compute_kv_cache_append_node()` appends FP16 or quantized INT8 KV rows.
- Existing eviction behavior:

```text
window = node.params.window_size
if window == 0: window = max_len

if current_len + new_seq_len > window:
  keep_sink = min(sink, current_len, window)
  tail_capacity = window - keep_sink
  keep first keep_sink rows
  keep recent tail rows
```

This already resembles:

```text
prefix_keep + recent_tail
```

but it uses `window` as the retained target, not as the hard allocation limit.

Needed:

- Distinguish hard limit `N = max_seq_len` from compact target `M`.
- On overflow, compact to `M`, then append incoming rows.
- If context shift is disabled, fail clearly instead of evicting.
- Preserve the same behavior for FP16 and INT8 branches.
- Avoid duplicating FP16 and INT8 compaction logic more than necessary.

### Engine Runtime State

Files:

- `cactus-engine/src/model.cpp`
- `cactus-engine/src/engine.h`

Current state:

- `Model::run_step()` passes `position_ids` from caller-provided position.
- `cache_total_seq_len_` is used as the logical position counter in `decode()`, `prefill()`, and media paths.
- `context_tokens_` is used for full-context text path.
- `reset_cache()` clears cache state and token history.
- `remove_thinking_tokens()` already mutates cache rows directly after removing ranges.
- `copy_cache_states()` copies prefill cache into step cache and already performs a sink plus recent-tail copy if source has more rows than destination.

Needed:

- Track prompt length explicitly for prompt-aware keep:

```text
prompt_len
```

- Preserve a distinction between:

```text
logical_position / total tokens processed
retained_cache_rows / current KV rows
```

- Decide whether compaction changes future `position_ids`.

Recommended first policy:

- Keep `cache_total_seq_len_` as the logical absolute position counter.
- Let cache compaction change retained KV rows only.
- Do not renumber positions in v1.

Reason: step inputs already use absolute position IDs. Renumbering positions after compaction is model-sensitive and should be handled as a separate feature.

Constraint: after compaction, attention position offsets must continue to match retained cache rows. Verify this with tests because `ATTENTION_CACHED` computes dynamic offsets from cache length.

## Recommended Software Design

### One Policy Object

Create one normalized policy representation rather than passing loose integers.

Suggested Python dataclass:

```text
CachePolicy:
  ctx_size: int
  context_shift: bool
  compact_to: int
  keep: int
  keep_prompt: bool
```

Suggested C++ equivalent:

```text
struct CachePolicy {
  size_t ctx_size;
  size_t compact_to;
  size_t keep;
  size_t prompt_len;
  bool context_shift;
  bool keep_prompt;
};
```

Use a helper constructor/normalizer:

```text
normalize_cache_policy(ctx_size, context_shift, keep, keep_prompt)
```

Validation:

```text
ctx_size > 0
0 < compact_to <= ctx_size
0 <= keep < compact_to
```

Design pattern: **Single Source of Truth**. Defaults and validation live in one place.

### Keep Policy Separate From Attention Window

Do not make `window_size` carry the meaning of both model sliding-window attention and cache compaction target.

Existing `window_size` is used by attention kernels and by cache append eviction. That coupling is already risky. The new implementation should separate:

```text
attention window size
cache hard capacity
cache compaction target
cache prefix keep
```

Design pattern: **Separate Domain Concepts**. Avoid overloading a generic field because the names are similar.

### Small Runtime Primitive

In `ops_cache.cpp`, implement one conceptual primitive:

```text
append_with_policy(cache, new_rows, policy)
```

Even if FP16 and INT8 branches remain separate, they should follow the same formula and ideally share the same computed plan:

```text
should_compact
prefix_keep
tail_keep_from_existing
incoming_tail_keep
append_offset
new_current_len
```

Design pattern: **Plan Then Apply**. Compute row ranges once, then apply those ranges to FP16 or INT8 storage.

### Explicit Failure Mode

Do not silently evict when `context_shift=false`.

Design pattern: **Fail Fast**. The fixed-window behavior is a correctness feature.

### Backward Compatibility

Existing graphs may not have new params. Defaults when reading missing fields should preserve old behavior where possible, but newly generated graphs should use the new explicit fields.

Design pattern: **Compatible Extension**. Add fields without changing unrelated op semantics.

## Implementation Steps

### Step 1: Add Python Cache Policy Normalization

Target files:

- `python/cactus/transpile/model_adapters.py` or a new small module such as `python/cactus/transpile/cache_policy.py`
- `python/tests/test_cli_transpile_defaults.py` or new `python/tests/test_cache_policy.py`

Add:

```text
DEFAULT_CTX_SIZE = 16384
normalize_cache_policy(...)
```

Behavior:

- `ctx_size=None` means default `16384`.
- `ctx_size="auto"` means model config path remains available.
- `cache_context_length` maps to `ctx_size`.
- `keep=None` means `ctx_size / 4`.
- `compact_to=ctx_size / 2` in v1.

Tests:

- default resolves to `16384/8192/4096`.
- `auto` does not become `16384`.
- invalid `keep >= compact_to` fails.
- `keep=0` is valid.

### Step 2: Add CLI Flags

Target files:

- `python/cactus/cli/__init__.py`
- `python/cactus/cli/run.py`
- `python/cactus/cli/convert.py`
- `python/cactus/cli/model.py`

Add flags to `run` and `convert`:

```text
-c, --ctx-size
--context-shift / --no-context-shift
--keep
--keep-prompt / --no-keep-prompt
```

Update `TranspileOptions`:

```text
ctx_size
context_shift
keep
keep_prompt
cache_context_length
```

Conflict rule:

```text
if --ctx-size and --cache-context-length are both set and differ:
  error
```

Tests:

- help includes new flags.
- alias still works.
- conflicts fail.
- options are passed to `run_transpile()`.

### Step 3: Thread Policy Through HF Transpile

Target files:

- `python/cactus/transpile/hf_model.py`
- `python/cactus/transpile/model_adapters.py`

Add transpile CLI args matching the outer CLI.

Component graph metadata should include:

```text
max_cache_seq_len = ctx_size
cache_compact_to = compact_to
cache_keep = keep
cache_keep_prompt = keep_prompt
cache_context_shift = context_shift
```

For v1 prompt length:

```text
cache_prompt_len = input_ids.shape[1]
```

This is conservative for transpiled static prompt examples, but engine runtime prompt length may differ. If runtime prompt length can differ, add engine-level runtime override later. Do not pretend static metadata is enough for all chat sessions.

Tests:

- component specs include expected metadata.
- default cache length is 16k, not model config.
- explicit `--ctx-size` overrides.
- explicit `auto` still reads model config.

### Step 4: Lower Metadata Into Graph Params

Target file:

- `python/cactus/transpile/lower.py`

Current method:

```text
_lower_attention_with_internal_kv_cache()
```

Update it to read:

```text
cache_compact_to
cache_keep
cache_keep_prompt
cache_context_shift
cache_prompt_len
```

and pass them into:

```text
g.kv_cache_state(...)
g.kv_cache_append(...)
```

Keep `window_size` for attention behavior.

Tests:

- generated graph params contain compact/keep policy.
- sliding-window attention still sets attention `window_size`.

### Step 5: Extend Graph Params and FFI

Target files:

- `cactus-graph/cactus_graph.h`
- `cactus-graph/src/builder.cpp`
- `cactus-graph/src/param_io.cpp`
- `cactus-engine/cactus_engine.h`
- `cactus-engine/src/graph_ffi.cpp`
- `python/cactus/bindings/graph.py`
- `python/cactus/bindings/cactus.py`

Add params:

```text
cache_compact_to
cache_keep
cache_prompt_len
cache_context_shift
cache_keep_prompt
```

If minimizing serialized bool fields, use:

```text
cache_flags
```

but prefer readable fields unless the diff becomes large.

Update:

- `CactusGraph::kv_cache_state`
- `CactusGraph::kv_cache_append`
- FFI declarations
- Python bindings
- param serialization schemas
- read/write switch cases

Tests:

- `cactus-graph/tests/test_io.cpp` should still pass.
- add graph IO coverage for new fields if existing IO tests do not catch them.

### Step 6: Implement Cache Append Compaction

Target file:

- `cactus-graph/src/ops_cache.cpp`

Change:

```text
window = node.params.window_size or max_len
```

to:

```text
hard_limit = meta->max_seq_len
compact_to = node.params.cache_compact_to or hard_limit
```

On append:

```text
new_total = current_len + new_seq_len
if new_total <= hard_limit:
  append normally
else if !context_shift:
  throw runtime_error
else:
  compact existing + incoming to compact_to/incoming-aware target
```

Range planning must handle large incoming chunks.

Desired final retained length after append:

```text
target_after_append = min(hard_limit, compact_to + new_seq_len)
```

But if `new_seq_len >= target_after_append - prefix_keep`, incoming rows may consume the entire tail. In that case:

```text
keep latest incoming tail rows after prefix
```

The existing code already has a similar branch:

```text
if new_seq_len >= tail_capacity
```

Reuse that shape.

Tests:

- FP16 basic compaction.
- INT8 basic compaction.
- incoming larger than tail capacity.
- `keep=0`.
- `context_shift=false` failure.
- repeated compaction remains bounded.

### Step 7: Prompt-Length Handling

Target files:

- `cactus-engine/src/model.cpp`
- `cactus-engine/src/engine.h`
- possibly manifest loading in `Model::load_manifest()`

The graph-level cache op needs `prompt_len` for `keep_prompt`.

Best v1 strategy:

1. Store prompt length in `Model` at initial prefill.
2. Push prompt length into cache policy before decode starts if graph params are mutable at runtime.
3. If graph params are not conveniently mutable, keep `keep_prompt` implemented as compile-time/transpile prompt length for v1 and document that dynamic runtime prompt-aware keep needs a follow-up.

Preferred acceptable implementation:

```text
prompt_len_ = tokens.size() during initial prefill/prefill_and_sample_first_token
```

Then ensure cache append nodes see:

```text
cache_prompt_len = prompt_len_
```

Constraint:

- Do not infer prompt length from current cache length.
- Reset must clear `prompt_len_`.
- Media paths should default to no context shift or no prompt-aware shift until multimodal semantics are explicit.

### Step 8: Fixed-Window Failure Path

Target files:

- `cactus-graph/src/ops_cache.cpp`
- optionally `cactus-engine/src/model.cpp` for cleaner user-facing error/logs

Behavior:

```text
if !context_shift and current_len + incoming > max_len:
  throw std::runtime_error("KV cache context limit exceeded ...")
```

Tests:

- synthetic graph append fails.
- engine-level decode reports a clear error.

### Step 9: Logs

Target files:

- `python/cactus/cli/model.py`
- `python/cactus/transpile/hf_model.py`
- `cactus-graph/src/ops_cache.cpp` only if graph logging patterns are acceptable
- `cactus-engine/src/model.cpp`

Add concise logs:

```text
KV cache policy: ctx_size=16384 compact_to=8192 keep=4096 keep_prompt=true context_shift=true
KV cache compacted: old_len=... incoming=... prefix_keep=... tail_keep=... new_len=...
context limit exceeded: cache_len=... incoming=... ctx_size=... context_shift=false
```

Do not log on every token.

## Key Constraints

### Keep Diffs Small

Reuse existing cache append shape. Do not rewrite attention kernels or model execution paths unless necessary.

### Preserve Existing Sliding-Window Attention

Model attention `window_size` is not the same as cache compaction target. Do not break Gemma/Qwen sliding-window hints.

### Preserve FP16 and INT8 Cache Behavior

Both branches in `ops_cache.cpp` must implement the same policy. Tests need to cover both if possible.

### Preserve Prefill/Step Cache Compatibility

`copy_cache_states()` assumes cache headers and row layouts match. Any metadata layout extension must remain 64 bytes or be updated consistently everywhere.

### Avoid Silent Prompt Truncation

If:

```text
P > N
```

fail clearly in v1. Do not silently prefill a truncated prompt.

### Keep Logical Position Separate From Retained Rows

`cache_total_seq_len_` currently feeds `position_ids`. Do not conflate it with retained cache row count after compaction.

### Multimodal Is Future Work

Do not assume media token retention follows text retention. For v1, either disable context shift for multimodal paths or keep behavior conservative and documented.

### No Broad Refactors

Do not introduce a new runtime architecture. This should be a policy extension to the existing graph cache operation and transpile metadata path.

## Acceptance Criteria

### CLI and Policy

- `cactus run --help` and `cactus convert --help` show:

```text
--ctx-size
--context-shift
--keep
--keep-prompt
```

- `--cache-context-length` remains supported.
- Conflicting `--ctx-size` and `--cache-context-length` values fail.
- Defaults resolve to:

```text
ctx_size=16384
compact_to=8192
keep=4096
keep_prompt=true
context_shift=true
```

### Transpile Metadata

- Cached decoder component metadata receives the normalized policy.
- Default cache allocation is 16k, not full model metadata.
- Explicit `--ctx-size` changes `max_cache_seq_len`.
- Explicit `auto` continues to use model metadata.

### Graph Cache Behavior

- Cache appends normally before `N`.
- With context shift enabled, overflow compacts to the planned retained shape.
- With context shift disabled, overflow fails clearly.
- `--keep 0` produces recent-only retention.
- `--no-keep-prompt --keep K` produces raw prefix plus tail retention.
- Repeated overflow keeps:

```text
cache_length <= N
```

### Runtime Behavior

- Reset clears cache state and prompt-length state.
- Initial prompt longer than `N` fails clearly.
- Long generation continues across compactions when context shift is enabled.
- Position handling remains stable enough for real model smoke tests.

### Tests

Minimum test suite before considering this done:

- Python policy normalization tests.
- Python CLI parse/default/conflict tests.
- C++ graph cache shape tests in `cactus-graph/tests/test_cache.cpp`.
- Graph IO serialization test for new params.
- Fixed-window failure test.
- Tiny real-model forced-compaction smoke test, if a suitable local model/bundle is available.

Long-term suite:

- real LLM instruction-retention scenario
- recent-tail retention scenario
- middle-discard scenario
- prompt-cap scenario
- overlong-prompt failure
- device/performance runs on Mac, Pixel, and Samsung

## Suggested Verification Commands

Follow repo instructions before any Cactus command:

```bash
source ./venv/bin/activate
cactus build
```

Then run the smallest relevant checks:

```bash
python python/test.py -k cache
cactus-graph/test.sh
cactus run --help | rg -- "--ctx-size|--context-shift|--keep|--keep-prompt"
cactus convert --help | rg -- "--ctx-size|--context-shift|--keep|--keep-prompt"
```

For C++ graph tests, prefer the existing graph test runner rather than adding one-off binaries.

## Open Questions To Resolve During Implementation

### How should graph params receive runtime `prompt_len`?

Prompt-aware keep is only fully correct if the cache op knows the actual initial prompt length. Static transpile-time prompt length is not enough for arbitrary user prompts.

Preferred answer:

- store `prompt_len_` in `Model`
- propagate it into cache append params before execution

Fallback:

- implement raw prefix retention first
- land `--keep-prompt` only once runtime prompt length can be passed correctly

### Should fixed-window failure happen before graph execution?

Graph-level failure is sufficient for correctness, but model-level preflight can produce better user errors. Prefer graph-level correctness first, model-level ergonomics second.

### What should multimodal do in v1?

Conservative answer:

- disable context shift for multimodal paths or treat it as future work
- document that media retention semantics are not finalized

Do not silently partially retain media-derived context until the semantics are explicit.
