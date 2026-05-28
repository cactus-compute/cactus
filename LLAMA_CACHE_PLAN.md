# llama.cpp-Style KV Cache Shift Plan

This plan describes the next cache implementation pass: making Cactus context shift follow llama.cpp's RoPE-aware semantics instead of treating compaction as a plain KV row copy.

The important distinction is that llama.cpp does not only retain and move KV rows. It updates the retained tokens' logical positions and then applies a K-shift operation to cached keys when RoPE is active. Values are moved but not rotated. New decode tokens are then evaluated at the shifted logical positions.

## Goal

Implement context shifting with positional fidelity:

1. Detect that the next prefill/decode would exceed the KV context limit.
2. Compact the cache before evaluating the next token or chunk.
3. Compute new logical positions for retained tokens.
4. Apply a RoPE shift to retained K-cache rows so they match their new logical positions.
5. Move V-cache rows without rotation.
6. Continue generation using the shifted logical cache length as the position base.

The acceptable result is not merely "the cache stays under the limit." The shifted-cache path must match, within normal numeric tolerance, a full recompute over the same compacted logical context.

## llama.cpp Behavior To Mirror

llama.cpp models context shift as a sequence operation:

- Discard a range of old cache cells.
- Keep selected prefix and tail cells.
- Shift retained cells' positions by a delta through sequence metadata.
- If the model uses RoPE, run a K-shift graph over cached key tensors.
- Reset the pending shift metadata after the key cache has been corrected.

The corresponding upstream concepts are `seq_rm`, `seq_add`, `get_has_shift`, `set_input_k_shift`, `build_graph_shift`, and `build_rope_shift` in `llama-kv-cache.cpp`.

Cactus should copy these semantics, adapted to the simpler current dense-cache representation.

## Current Gap

The current first-pass Cactus implementation compacts inside `KV_CACHE_APPEND`. That keeps memory bounded, but it is not fully llama.cpp-correct for RoPE because:

- Compaction happens after the incoming token's Q/K RoPE has already been computed.
- Dense cache rows do not currently carry explicit logical positions.
- Moved K rows are not re-rotated to match shifted logical positions.
- Attention position offsets are derived from cache length, not from a RoPE-shift-aware logical position model.

This is acceptable only for a narrower absolute-position interpretation, and even then it depends on attention masks not using compacted row index as logical position. For llama.cpp-style context shift, this must be fixed.

## Testing Plan First

Implement tests before changing runtime behavior. Keep them small, deterministic, and mostly unit-level.

### 1. RoPE Shift Unit Tests

Add graph/kernel tests that prove K-shift math directly:

- Given an unrotated FP16 tensor `x`, compute `k_old = rope(x, old_pos)`.
- Apply `rope_shift(k_old, delta)`.
- Assert the result matches `rope(x, old_pos + delta)`.

Cases:

- `delta < 0`, the normal context-shift case.
- `delta > 0`, for completeness and symmetry.
- Multi-token rows with different deltas per row.
- Multiple heads.
- Partial rotary dimensions if supported by the model/kernel path.

Expected invariant:

```text
rope_shift(rope(x, p_old), p_new - p_old) ~= rope(x, p_new)
```

### 2. Compaction Plan Position Tests

Split compaction planning from byte movement so it can be tested in isolation.

For a cache with logical positions:

```text
old_positions = [0, 1, 2, 3, 4, 5, 6, 7]
N = 8
M = 4
```

Recent-only compaction should retain the tail and renumber it:

```text
retained_old = [4, 5, 6, 7]
retained_new = [0, 1, 2, 3]
shift        = [-4, -4, -4, -4]
```

Prompt-keep compaction with `K = 2` should keep prefix plus tail:

```text
retained_old = [0, 1, 6, 7]
retained_new = [0, 1, 2, 3]
shift        = [0, 0, -4, -4]
```

These tests should assert row source, row destination, old position, new position, and shift.

### 3. K/V Cache Movement Tests

Use FP16 cache mode for readable assertions:

- K rows are moved and RoPE-shifted.
- V rows are moved only.
- Current cache length becomes the compacted logical length.
- Appending after compaction writes at the compacted tail.

Then add equivalent coverage for the default int8 KV path once int8 K-shift exists.

### 4. Cached Attention Equivalence Tests

Build a tiny cached-attention graph and compare two paths:

- Full recompute over the compacted logical context.
- Original cache, context shift, then one decode step.

The output should match within tolerance.

This is the key correctness test because it verifies row movement, K-shift, position offsets, and attention masking together.

### 5. Engine Position Tests

Add tests around runtime position state:

- Before compaction, decode position advances normally.
- After compaction from `N` to `M`, the next decode position is `M`, not the absolute number of tokens ever seen.
- Prompt-keep cases preserve prefix positions and shift tail positions into the compacted range.

This catches the bug where cache memory is compacted but new Q/K RoPE continues using stale absolute positions.

### 6. Sliding-Window Tests

Sliding-window layers should not preserve prompt sink tokens by default.

Test that layers with `window_size > 0` use recent-only retention:

```text
K_effective = 0
M_effective = min(window_size, cache_compact_to)
```

The result should be a contiguous recent tail so local attention semantics remain coherent.

## Implementation Plan

### Phase 1: Make Compaction Planning Explicit

Extract a small cache planning helper from `ops_cache.cpp`.

The helper should return structured rows:

```text
src_row
dst_row
old_pos
new_pos
shift = new_pos - old_pos
```

Keep it independent of FP16/int8 storage. This isolates policy math from data movement and makes tests precise.

Design constraints:

- No model-specific branching in the planner.
- No direct tensor mutation in the planner.
- Fail clearly when `keep >= compact_to`.
- For sliding-window layers, force recent-only retention unless there is a later explicit design for prompt sink plus local attention.

### Phase 2: Move Context Shift Before Decode Execution

Add a pre-execution cache maintenance step in the engine.

Before running decoder or decoder-prefill:

```text
if cache_len + incoming_len > ctx_limit:
    compact caches
    update logical cache length
```

Do this before the graph computes RoPE for incoming Q/K. That matches llama.cpp's order and avoids rotating an incoming token that was created at the wrong position.

Relevant areas:

- `cactus-engine/src/model.cpp`
- `cactus-engine/src/engine.h`
- graph cache nodes and runtime cache metadata

### Phase 3: Track Logical Positions

Extend KV cache metadata so dense rows can represent shifted logical positions.

At minimum, store enough state to derive:

```text
logical_position(row)
current_logical_len
```

For prompt-keep compaction, a single base offset is not enough because the retained rows may be non-contiguous before renumbering. The compaction step should write the final compacted row order and make positions contiguous after the shift:

```text
row i => logical position i
```

The engine's next decode position should be:

```text
position = current_logical_len
```

not:

```text
position = total_tokens_seen
```

### Phase 4: Add K-Shift Operation

Add a graph/runtime operation that applies RoPE delta to cached K rows.

The operation needs:

- K cache tensor.
- Per-row shift values.
- RoPE theta.
- Rotary dimension or model-specific rotary metadata.
- Cache length.
- Head count and head dimension.

Expected behavior:

```text
K[row] = rope_shift(K[row], shift[row])
```

For non-RoPE models, the shift operation is a no-op.

For unsupported RoPE variants, fail clearly when context shift is enabled.

### Phase 5: Distinguish K Cache From V Cache

Add a small cache-kind field to cache state or append nodes:

```text
cache_kind = key | value
```

On compaction:

- `key`: move rows, then apply K-shift to moved/retained rows with non-zero shift.
- `value`: move rows only.

This avoids duplicating separate K/V planning logic and keeps the policy shared.

### Phase 6: Support Default int8 KV Cache

Cactus defaults to int8 KV cache, so llama.cpp-style K-shift must eventually support it.

Implementation options:

1. Dequantize K row to FP16/FP32, apply RoPE shift, requantize.
2. Add a quantized rotation kernel.

The surgical first implementation should use dequantize/rotate/requantize because it is simpler and easier to validate. Optimize later only if benchmarks show K-shift is expensive enough to matter.

Do not silently skip K-shift for int8. If int8 support is not implemented in the first patch, context shift must fail clearly for int8 cache.

### Phase 7: Update Lowering Metadata

Thread RoPE metadata from transpile/lowering into cache nodes.

Relevant areas:

- `python/cactus/transpile/lower.py`
- `python/cactus/transpile/model_adapters.py`
- graph params, serialization, FFI, and Python bindings

The metadata should be graph-level when uniform, and node-level when layers differ.

For models with local and global RoPE settings, preserve the per-layer values rather than assuming one global theta.

## Software Design Constraints

Keep the design small and explicit:

- Put cache policy math in one helper.
- Put RoPE shift math in one operation/kernel path.
- Keep K/V differences represented by data, not duplicated branches.
- Make unsupported cases fail at conversion or graph execution with a specific message.
- Avoid silent fallback to absolute-position behavior when `--context-shift` is enabled.
- Preserve existing graph serialization compatibility where possible by using defaulted params.
- Do not mix this with unrelated cache performance work.

## Acceptance Criteria

The implementation is acceptable when:

- Static context limit still defaults to `16k`.
- `--context-shift` compacts before decode execution.
- `--no-context-shift` fails clearly when the limit is exceeded.
- Prompt-keep and recent-only policies produce the expected retained logical context.
- K rows are RoPE-shifted and V rows are not.
- The next decode position after compaction equals the compacted logical length.
- Sliding-window layers use recent-only compaction.
- FP16 cache tests pass.
- Default int8 cache either passes equivalent tests or fails clearly with context shift enabled.
- Cached attention after context shift matches full recompute over the compacted logical context.
- Long-running LLM E2E tests show stable generation across at least one forced compaction.

## Long-Term E2E Validation

After unit tests pass, add real model validation:

- Force a small context limit such as `128` or `256`.
- Use deterministic decoding.
- Generate past the limit with recent-only compaction.
- Generate past the limit with prompt-keep compaction.
- Compare against a full-recompute reference over the compacted logical context where practical.
- Run at least one model with global attention and one model with sliding-window layers.
- Keep multimodal context shift as a future validation area because image/audio prompt tokens complicate prompt-keep semantics.

## Risk Assessment

This is a moderate-to-hard change.

The hard parts are:

- Correct ordering: compaction must happen before incoming RoPE.
- K-shift for default int8 cache.
- Per-layer RoPE metadata.
- Sliding-window interactions.
- Avoiding accidental absolute-position behavior hidden behind passing memory tests.

The lowest-risk path is to implement the tests first, then add FP16 K-shift, then int8 K-shift, then run real generation tests.
