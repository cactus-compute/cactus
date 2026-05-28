# KV Cache Design

This document describes the intended Cactus KV cache policy for context limits and long-running generation. It explains the user-facing behavior and the math behind the design. It does not describe implementation details.

## Goals

Cactus should have a bounded, predictable KV cache by default. A model may advertise a very large context length, but that should not force Cactus to allocate that full length on mobile hardware. The default should be practical, explicit, and easy to override.

The cache policy should support three common needs:

- fixed memory use
- continued generation after the cache fills
- preservation of the original task or system prompt when compacting

The design is intentionally close to llama.cpp's vocabulary where the concepts match:

- `--ctx-size` controls the context/KV cache limit
- `--context-shift` enables continued generation after the limit
- `--keep` controls how much prefix context survives shifting

Cactus adds `--keep-prompt` because prompt-aware retention is the default behavior we want for chat and instruction use.

## CLI

### `-c, --ctx-size N`

Sets the maximum KV cache context size in tokens.

Default:

```text
N = 16384
```

This is the static cache limit. Cactus should not silently allocate the full model context unless the user asks for it.

Existing `--cache-context-length` should remain a compatibility alias for `--ctx-size`.

### `--context-shift` / `--no-context-shift`

Controls what happens when adding more tokens would exceed `--ctx-size`.

Default:

```text
--context-shift
```

With context shift enabled, Cactus compacts the retained KV history and continues generation. With context shift disabled, Cactus uses fixed-window behavior: the request should fail or stop clearly when the context fills.

### `--keep K`

Sets how many prefix tokens to preserve during compaction.

Default:

```text
K = N / 4
```

With the default `N = 16384`:

```text
K = 4096
```

`--keep 0` means recent-only sliding behavior.

### `--keep-prompt` / `--no-keep-prompt`

Controls what `--keep` means.

Default:

```text
--keep-prompt
```

With `--keep-prompt`, Cactus keeps up to `K` tokens from the original prompt, then fills the remaining retained cache with recent tokens.

With `--no-keep-prompt`, Cactus keeps the first `K` cache tokens, regardless of whether they came from the prompt, generated text, media placeholders, or another source. This is closer to llama.cpp's raw `--keep` behavior.

## Core Math

Let:

```text
N = maximum cache context size
M = compacted cache target size
K = configured prefix keep amount
P = original prompt length in tokens
C = current cache length before compaction
```

Defaults:

```text
N = 16384
M = N / 2
K = M / 2 = N / 4
```

With defaults:

```text
M = 8192
K = 4096
```

When context shift is enabled and the cache would exceed `N`, Cactus compacts the cache to `M` retained tokens.

The key equation is:

```text
retained_tokens = prefix_keep + tail_keep = M
```

For prompt-aware retention:

```text
prefix_keep = min(P, K)
tail_keep = M - prefix_keep
```

For raw prefix retention:

```text
prefix_keep = min(C, K)
tail_keep = M - prefix_keep
```

The new retained cache is:

```text
new_cache = kept_prefix + most_recent(tail_keep)
```

## Invariants

The policy should obey:

```text
N > 0
0 < M <= N
0 <= K < M
```

For the default policy:

```text
M = N / 2
K = M / 2
```

This means:

```text
K = N / 4
tail_keep = N / 4
```

The strict `K < M` requirement matters. If `K >= M`, compaction can preserve only the prefix and no recent tokens. That defeats the purpose of continued generation because the model loses the most recent local context.

## Why Compact to Half?

Compacting from `N` to `N / 2` creates headroom.

If Cactus only dropped enough tokens to fit the next token, it would need to compact almost every token after reaching the limit. That would be inefficient and difficult to reason about. A half compaction gives a simple amortized policy:

```text
headroom_after_compaction = N - M
```

With `M = N / 2`:

```text
headroom_after_compaction = N / 2
```

For the default:

```text
headroom_after_compaction = 8192 tokens
```

So after compaction, generation can continue for thousands of tokens before the next compaction.

## Why Keep Prompt Plus Recent Tail?

A long-running chat or instruction task usually has two important regions:

- the beginning, where system instructions and task framing live
- the end, where the most recent conversation or generated text lives

The middle is usually the least valuable region under pressure.

The default policy keeps:

```text
up to N / 4 prompt tokens
plus
N / 4 recent tokens
```

With `N = 16384`, this is:

```text
up to 4096 prompt tokens
plus
4096 recent tokens
```

This balances durable task framing with immediate recency. It is more suitable for chat than recent-only sliding, which can lose system instructions, and more suitable than keeping too much prefix, which can lose the current conversation state.

## Recent-Only Sliding

Recent-only sliding is expressed as:

```text
--keep 0
```

Then:

```text
prefix_keep = 0
tail_keep = M
new_cache = most_recent(M)
```

With defaults:

```text
new_cache = most_recent(8192)
```

This is useful for free-form continuation where the initial prompt is not important after generation has moved on. It is less safe for instruction-following tasks because the model may forget the original task constraints.

## Raw Prefix Keeping

Raw prefix keeping is expressed as:

```text
--no-keep-prompt --keep K
```

Then:

```text
prefix_keep = min(C, K)
tail_keep = M - prefix_keep
```

This is useful when the user wants llama.cpp-like prefix behavior. It preserves the first tokens in cache, not necessarily the prompt boundary.

## Prompt-Aware Keeping

Prompt-aware keeping is expressed as:

```text
--keep-prompt --keep K
```

Then:

```text
prefix_keep = min(P, K)
tail_keep = M - prefix_keep
```

This makes `--keep` a cap, not a promise to preserve the whole prompt. If the prompt is shorter than `K`, Cactus keeps all prompt tokens. If the prompt is longer than `K`, Cactus keeps only the first `K` prompt tokens and uses the rest of `M` for recent tail context.

This behavior is deliberate. The cache should not preserve an arbitrarily long prompt at the expense of all recency.

## Long Prompt Policy

Long prompts are the hardest case because preserving all prompt tokens can consume the entire compacted cache.

The default policy should be:

```text
prefix_keep = min(P, K)
```

not:

```text
prefix_keep = P
```

Because:

```text
tail_keep = M - prefix_keep
```

If `prefix_keep` grows too large, `tail_keep` becomes too small. If `prefix_keep >= M`, there is no room for recent context.

For example, with defaults:

```text
N = 16384
M = 8192
K = 4096
```

If:

```text
P = 12000
```

the retained context is still:

```text
prefix_keep = min(12000, 4096) = 4096
tail_keep = 8192 - 4096 = 4096
```

Cactus keeps the first 4096 prompt tokens and the most recent 4096 tokens.

## Overlong Initial Prompts

If the original prompt itself exceeds `N`, the safest initial policy is to fail clearly rather than silently truncate.

Reason:

```text
P > N
```

means the prompt cannot fit even before generation begins. Silent truncation would make the model answer a different prompt than the user provided.

Later, Cactus can add explicit prompt truncation modes. Those should be opt-in because they change request semantics.

## Choosing Different Values

The defaults are:

```text
N = 16384
M = N / 2
K = N / 4
```

Smaller mobile setting:

```text
--ctx-size 8192
```

gives:

```text
N = 8192
M = 4096
K = 2048
```

More aggressive prompt preservation:

```text
--ctx-size 16384 --keep 6144
```

gives:

```text
N = 16384
M = 8192
K = 6144
tail_keep = 2048
```

This is allowed because:

```text
K < M
```

but it is less balanced. It preserves more initial instruction context and less recent context.

Recent-heavy behavior:

```text
--ctx-size 16384 --keep 1024
```

gives:

```text
N = 16384
M = 8192
K = 1024
tail_keep = 7168
```

This favors recency while still preserving a small instruction prefix.

## Fixed-Window Mode

Fixed-window mode is:

```text
--no-context-shift
```

In this mode, when the request would exceed:

```text
C > N
```

Cactus should fail or stop with a clear context-limit message.

This mode is useful for tests, reproducibility, and users who prefer explicit failure over automatic history compaction.

## Recommended Defaults

The recommended default configuration is:

```text
--ctx-size 16384
--context-shift
--keep 4096
--keep-prompt
```

Equivalent equations:

```text
N = 16384
M = 8192
K = 4096
prefix_keep = min(P, K)
tail_keep = M - prefix_keep
```

This gives a bounded cache, avoids model-metadata over-allocation, preserves the task prefix, and maintains a substantial recent tail.

## Non-Goals

This design does not claim that compacted context is equivalent to a true larger context window. Tokens that are discarded cannot influence future attention.

This design also does not define model-specific long-context extrapolation, RoPE scaling, or prompt summarization. Those are separate features. The goal here is bounded KV cache behavior with explicit, understandable retention rules.

## Testing Strategy

The tests should prove the policy at several levels. Unit tests should validate exact math and retention behavior. End-to-end tests with real models should validate that the policy produces the intended user-visible behavior over long generations.

The key principle is that deterministic tests should cover deterministic rules. Real LLM tests should check behavioral invariants, not exact output strings.

### Policy Unit Tests

Start with pure policy tests that do not require a model.

The default configuration is:

```text
N = 16384
M = 8192
K = 4096
```

Test prompt-aware retention:

```text
P = 1000
prefix_keep = min(1000, 4096) = 1000
tail_keep = 8192 - 1000 = 7192
```

Test long-prompt retention:

```text
P = 12000
prefix_keep = min(12000, 4096) = 4096
tail_keep = 8192 - 4096 = 4096
```

Test recent-only retention:

```text
K = 0
prefix_keep = 0
tail_keep = M
```

Test raw prefix retention:

```text
--no-keep-prompt --keep 2048
prefix_keep = 2048
tail_keep = 8192 - 2048 = 6144
```

Test invalid configurations:

```text
N <= 0
M <= 0
K >= M
```

Each invalid configuration should fail clearly. In particular, `K >= M` must not be accepted because it leaves no guaranteed room for recent context.

### Cache Retention Shape Tests

Use token IDs as abstract cache entries. These tests should verify ordering and exact retained spans without using an LLM.

For:

```text
cache = [0, 1, 2, ..., 16383]
N = 16384
M = 8192
K = 4096
```

Default prompt-aware retention with a prompt at least `K` tokens long should retain:

```text
[0..4095] + [12288..16383]
```

Recent-only retention should retain:

```text
[8192..16383]
```

Raw prefix retention with `K = 1024` should retain:

```text
[0..1023] + [9216..16383]
```

These tests catch off-by-one errors, duplicate-token retention, missing-token retention, and reversed ordering.

### Overflow Decision Tests

The cache should not compact before it has to:

```text
C + incoming <= N
```

The cache should compact when appending would exceed the limit:

```text
C + incoming > N
```

After compaction, the retained history should be:

```text
retained_tokens = M
```

or, after accounting for the new incoming tokens:

```text
retained_tokens_after_append <= M + incoming
```

Repeated overflow should never allow cache history to grow without bound:

```text
cache_length <= N
```

### CLI Behavior Tests

The CLI should expose the final policy in a predictable way.

The main cases are:

```text
cactus run model --ctx-size 8192
cactus run model --ctx-size 8192 --keep 1024
cactus run model --ctx-size 8192 --keep 0
cactus run model --ctx-size 8192 --no-keep-prompt
cactus run model --no-context-shift
```

The assertions are:

- `--ctx-size` sets `N`
- `--cache-context-length` remains a compatibility alias
- default `--ctx-size` is `16384`
- default `M` is `N / 2`
- default `--keep` is `N / 4`
- default prompt behavior is `--keep-prompt`
- `--keep 0` selects recent-only retention
- `--keep >= M` fails clearly
- `--no-context-shift` selects fixed-window behavior

If both `--ctx-size` and `--cache-context-length` are provided, Cactus should either reject the ambiguity or resolve it with an explicitly documented precedence rule.

### Fixed-Window Tests

Fixed-window mode should fail or stop clearly when context is exhausted.

With:

```text
--no-context-shift
```

and:

```text
C + incoming > N
```

Cactus should not silently discard history. This mode is important for reproducibility and for users who prefer explicit failure over automatic compaction.

### Absolute-Position Policy Tests

If Cactus uses absolute positional embeddings after compaction, the central invariant is:

```text
cache row index may change
absolute token position must not change
```

Under this policy, compaction does not renumber retained tokens and does not apply a RoPE K-shift. A token originally encoded at absolute position `p` remains position `p` even if it is moved to a different cache row.

Policy tests should verify that cache capacity and model position capacity are separate concepts:

```text
N = KV cache capacity
L = true model position limit
```

Generation may compact while:

```text
absolute_position < L
```

but must fail clearly before evaluating tokens where:

```text
absolute_position >= L
```

This should be true even if compaction would leave free KV cache rows.

Compaction plan tests should preserve absolute positions exactly. Recent-only retention:

```text
old rows:      [0, 1, 2, 3, 4, 5, 6, 7]
kept rows:     [4, 5, 6, 7]
new rows:      [0, 1, 2, 3]
abs positions: [4, 5, 6, 7]
```

Prompt-aware retention:

```text
kept rows:     [0, 1, 6, 7]
new rows:      [0, 1, 2, 3]
abs positions: [0, 1, 6, 7]
```

Cache movement tests should verify:

- K rows move with their absolute-position metadata
- V rows move with their absolute-position metadata
- no K row is RoPE-shifted during compaction
- appending after compaction writes the new absolute position, not a renumbered compacted position

Attention tests should use absolute positions for masks. Full causal attention should allow:

```text
key_abs_pos <= query_abs_pos
```

Sliding-window attention should allow:

```text
query_abs_pos - key_abs_pos < window_size
```

These tests must not derive attention eligibility from dense row distance after compaction. A compacted cache such as:

```text
abs positions = [0, 1, 100, 101]
query_abs_pos = 102
```

should allow all retained keys for full causal attention, but with:

```text
window_size = 16
```

the sliding-window layer should attend only to:

```text
[100, 101]
```

Engine tests should verify that the decode position remains absolute:

```text
before compaction: next_position = 16384
after compaction to 8192 rows: next_position = 16384
```

The next position must not become:

```text
8192
```

Hard-limit tests should use a small artificial model limit. With:

```text
L = 32
N = 8
```

generation can compact repeatedly while positions are below `32`, but position `32` should fail if valid positions are `0..31`.

Chunked prefill tests should cover both boundaries:

- a chunk that exceeds `N` but not `L` should compact and continue
- a chunk that exceeds `L` should fail before partial mutation

For sliding-window models, tests should either verify that prompt rows outside the local window are masked by absolute distance, or verify that sliding-window layers use recent-only retention. The safer default is:

```text
K_effective = 0
```

for layers with:

```text
window_size > 0
```

### Real LLM Smoke Tests

Use a small model and a deliberately tiny context limit so compaction happens quickly.

Example configuration:

```text
--ctx-size 128 --keep 32 --keep-prompt
```

Use a prompt with a durable instruction, such as:

```text
Always begin your answer with BLUE.
```

Then add enough filler or generated turns to force compaction and ask a question that requires the instruction to be preserved.

Expected invariant:

```text
with --keep-prompt, the answer should still follow the retained instruction
```

The same test with:

```text
--keep 0
```

is allowed to forget the instruction more often because recent-only mode intentionally discards the prompt prefix.

These tests should avoid exact full-string matching. They should check narrow invariants such as whether a required prefix appears.

### Long-Running End-to-End Tests

Real LLM end-to-end tests should validate behavior across multiple compactions.

Instruction-retention scenario:

- put an important rule in the prompt prefix
- force several compactions
- ask whether the model still follows the rule

Recent-tail scenario:

- place a unique fact near the recent tail
- force compaction
- ask about that fact immediately afterward

Middle-discard scenario:

- place a unique fact in the middle region expected to be discarded
- after compaction, verify the model is less likely to recover it than retained prefix or retained tail facts

Prompt-cap scenario:

- use a prompt longer than `K` but smaller than `N`
- put one instruction inside the first `K` tokens
- put another instruction after the first `K` tokens
- after compaction, the first instruction should be retained more reliably

Overlong-prompt scenario:

```text
P > N
```

Expected behavior:

```text
clear failure, no silent truncation
```

### Device and Performance Tests

KV cache limits are partly a memory feature, so tests should include realistic device runs.

Measure:

- peak memory use
- tokens per second before compaction
- tokens per second after compaction
- latency of the compaction event
- stability across repeated compactions

Run on:

- Mac local
- Pixel
- Samsung

Device-specific benchmark and inference work should respect the repo's device concurrency rules: only one active benchmark or inference command per target device, while different devices can be tested concurrently.

### Long-Term Regression Suite

The long-term suite should include:

- a small deterministic policy suite
- a fake-token retention shape suite
- a tiny real-model forced-compaction smoke test
- a fixed-window failure test
- a long generation stress test that runs for many multiples of `N`

The stress test should assert:

```text
cache_length <= N
generation continues without crashing
compaction can happen repeatedly
```

It should not require exact generated text.

### Future Area: Multimodal

Multimodal context shifting needs separate validation. Image, audio, and other media inputs may expand into multiple internal tokens or embeddings, and those units may not map cleanly to simple text-token retention.

Future multimodal tests should answer questions such as:

- can media-derived context be safely shifted?
- should media tokens count as prompt tokens for `--keep-prompt`?
- should media prefixes be retained all-or-nothing rather than partially?
- should context shift be disabled for some multimodal paths until retention semantics are explicit?

Until those questions are answered, multimodal context shifting should be treated as a separate compatibility area rather than assumed to follow text-only behavior automatically.
