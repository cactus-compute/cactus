# llama.cpp Context Window and KV Cache Approaches

This note summarizes how current `llama.cpp` handles context-window limits, KV-cache sizing, cache eviction, and prompt reuse. Sources checked:

- `tools/cli/README.md`: <https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md>
- `common/arg.cpp`: <https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp>
- `src/llama-context.cpp`: <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp>
- `src/llama-kv-cache.cpp`: <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp>
- `tools/server/server-context.cpp`: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp>

## Mental model

`llama.cpp` treats context length primarily as an allocation and scheduling limit, not only as a model-quality setting. The configured context decides how many token positions the KV memory can hold. More context means more KV memory, and KV memory grows linearly with:

- context tokens
- number of layers
- number of KV heads
- K head dimension and V head dimension
- K/V storage type
- number of independent sequences or slots, unless unified KV is used

The model's training context is separate. If `--ctx-size 0` is used, `llama.cpp` loads the context size from model metadata. If the configured per-sequence context is smaller than training context, it logs that the model's full capacity is not being used. If it is larger, it warns about possible training-context overflow.

## Fixed context allocation

The main knob is:

```text
-c, --ctx-size N
```

In `llama-context.cpp`, `params.n_ctx == 0` maps to the model's training context. The context is padded internally, then split into `n_ctx_seq` depending on whether KV is unified:

- unified KV: each sequence can address the full configured `n_ctx`
- non-unified KV: `n_ctx_seq = n_ctx / n_seq_max`, padded and rounded so the total context is divisible by the max sequence count

This means `--ctx-size` is a total capacity knob in multi-sequence modes. The effective context available to one request can be lower than the raw number if the cache is partitioned across slots.

## Batch and microbatch limits

The related knobs are:

```text
-b, --batch-size N
-ub, --ubatch-size N
```

For causal attention, `llama.cpp` caps logical batch size by context size. Microbatch is then capped by logical batch. These do not increase the context window; they control how much prompt/eval work is processed in one logical and physical step. They matter because large context prompts can be processed in chunks without changing the KV allocation.

## Multi-slot serving

Server mode has:

```text
-np, --parallel N
-kvu, --kv-unified
-no-kvu, --no-kv-unified
```

Recent server defaults use auto parallelism as `n_parallel = 4` with unified KV enabled. Unified KV is important because it makes the KV buffer a shared pool instead of statically dividing context by slot. Without unified KV, increasing the number of parallel slots reduces the per-slot context budget.

Server code still assigns an `n_ctx` to each slot. It caps slot context to the model training context when the effective slot context exceeds the model's training context, because using more positions than the model was trained for is not automatically safe.

## Rejecting overlong requests

When context shifting is disabled, server mode rejects prompts that do not fit:

- if a prompt is larger than slot context, it returns an exceed-context error
- if a prompt reaches or exceeds available context for a normal text request, it asks the caller to increase context
- during generation, if the prompt plus one next token reaches context capacity and shifting is disabled, generation stops/errors instead of silently corrupting cache state

This is the simplest and safest policy: fixed context, explicit failure at the boundary.

## Context shift

The main knobs are:

```text
--context-shift
--no-context-shift
--keep N
```

When context shift is enabled and a slot fills, `llama.cpp` keeps an initial prefix and discards part of the middle/tail history. The server code computes:

- `n_keep`: tokens to preserve from the initial prompt; `--keep -1` means keep the original task prompt
- `n_left`: tokens after the kept prefix
- `n_discard`: explicit discard count when provided, otherwise about half of `n_left`

It then removes the discarded KV positions and shifts later positions backward with `seq_rm` and `seq_add`. This preserves a stable system/prefix region while making room for more generated tokens.

Important constraints:

- context shifting requires a memory implementation that supports shifting
- it is disabled for multimodal contexts in server mode
- it is disabled when shared-prompt parent/child slots are involved
- it changes positions, so models or memory types with special recurrent/sliding behavior may have limited support

This is a throughput optimization and long-generation strategy, not the same as giving the model unlimited real context.

## Sliding-window attention and SWA cache

For models with sliding-window attention, only a local window of past positions matters for affected layers. `llama.cpp` has:

```text
--swa-full
```

By default, SWA-aware cache behavior can avoid storing full history for layers where older tokens are masked out by the model's attention pattern. `--swa-full` forces a full-size SWA cache. That costs more memory but can be useful for compatibility and checkpoint/prompt-cache behavior.

Internally, the KV-cache allocator can reuse cache cells whose positions are outside the active SWA mask. This is a structural memory limit distinct from context shifting: old entries become irrelevant because the attention mask excludes them.

## KV cache dtype and memory pressure

The main knobs are:

```text
-ctk, --cache-type-k TYPE
-ctv, --cache-type-v TYPE
-kvo, --kv-offload
-nkvo, --no-kv-offload
```

Allowed K/V cache types include `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1`. The default is `f16`.

KV cache dtype is a direct memory lever. Quantized K/V cache allows larger context on the same memory budget, but there are backend constraints. The context initialization code validates quantized cache types and has special requirements around Flash Attention, especially for quantized V cache.

KV offload controls whether KV cache can live on accelerator memory. Disabling it can reduce device memory pressure but moves work and storage to host memory, usually with a speed tradeoff.

## Fit-to-memory behavior

The main knobs are:

```text
-fit, --fit on|off
-fitt, --fit-target MiB0,MiB1,...
-fitc, --fit-ctx N
```

`llama.cpp` can adjust unset arguments to fit available device memory. `--fit-ctx` sets the minimum context size that this automatic fitting is allowed to choose. If the user explicitly sets `--ctx-size 0`, the argument parser disables context reduction for fitting because the user asked for the model's full context.

This is a practical mobile/desktop pattern: choose the largest context that fits instead of treating model metadata as a mandatory allocation.

## Prompt cache and context checkpoints

The relevant knobs include:

```text
-cram, --cache-ram N
-ctxcp, --ctx-checkpoints N
--checkpoint-min-step N
--cache-idle-slots
--cache-reuse N
```

These are not the same as the live KV cache allocation. They are reuse mechanisms to avoid reprocessing prompt tokens.

Server mode selects reusable slots by longest common prefix and LRU. It can save prompt state into a RAM cache, restore matching prompts, and create context checkpoints. Checkpoints let the server resume from a saved KV state near the requested prompt when full live cache data is unavailable.

`--cache-reuse N` is more aggressive. When supported, it searches for matching chunks of at least `N` tokens in cached prompt state and shifts their KV range into the new prompt position. This depends on KV shifting support and is disabled for multimodal contexts.

## Defragmentation

`llama.cpp` still exposes:

```text
-dt, --defrag-thold N
```

but the option is marked deprecated in current argument handling. Older designs exposed explicit cache defragmentation thresholds because sparse KV cell allocation could fragment. Current code considers the option unnecessary for normal use.

## Practical policy patterns

`llama.cpp` effectively offers several context policies:

1. Fixed-window fail-fast: set `--ctx-size`, keep `--no-context-shift`, reject overlong requests.
2. Fixed-window rolling generation: set `--ctx-size`, enable `--context-shift`, choose `--keep`.
3. Model-full context: set `--ctx-size 0`, allocate from model metadata, accept high memory use.
4. Memory-fitting context: leave context adjustable and use `--fit`/`--fit-ctx`.
5. Sliding-window model policy: rely on SWA cache behavior and model layer masks.
6. Smaller KV dtype: use quantized K/V cache types to fit larger windows.
7. Server reuse: keep live slots, prompt cache, checkpoints, and cache reuse to avoid repeated prefill.
8. Unified multi-slot pool: use unified KV so slots share capacity instead of statically dividing it.

## Lessons for Cactus

Cactus should separate three ideas that are easy to conflate:

- maximum compiled KV allocation
- attention window used by specific layers
- history retention policy when generation exceeds allocation

For current Cactus bundles, `--cache-context-length` is the closest equivalent to `llama.cpp --ctx-size`: it controls the compiled KV cache length for cached decode graphs. Since this is baked into graph state allocation, changing it requires retranspiling or reconverting the bundle.

The safest Cactus policy is:

- expose the cache length wherever a bundle can be auto-created
- default to model metadata only when the user explicitly asks for `auto`
- reject or clearly fail when a request cannot fit a fixed cache
- use model-provided sliding-window metadata to reduce per-layer cache allocation
- add rolling context shift only after position-shift behavior is explicit for each supported model family

For mobile defaults, the llama.cpp lesson is that model metadata should not be treated as a mandatory allocation. A 128K model context can be correct architecturally and still be the wrong default for a phone. A practical default should be bounded, user-overridable, and logged.
