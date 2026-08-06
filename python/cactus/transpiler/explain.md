# Cactus transpiler and runtime optimization guide

This document explains the architecture and the optimization work on the
`custom-transpiler` branch through 2026-08-05. It is intended to answer four
questions for every important change:

1. What problem does it solve?
2. How does the algorithm work?
3. Where is it implemented?
4. Why does it improve speed or memory?

The central design rule is that a new Hugging Face model should first be able
to run through a conservative generic path. Model-specific profiles, component
splits, cache contracts, and fusion groups are then added as optimizations; they
are not prerequisites for proving basic compatibility.

## 1. End-to-end mental model

The optimized pipeline is:

```text
Hugging Face model and processor
        |
        | torch.export with representative text/image/audio and cache state
        v
Converter LayerMap (lossless JSON IR)
        |
        | structural graph matching and repeated simplification
        v
Simplified IR with Cactus semantic/fused operations
        |
        | model-profile-directed component partitioning
        v
Generator: Cactus graphs + bound weight metadata
        |
        | runtime-plan generation
        v
components/*.cactus + components/manifest.json + runtime_plan.json
        |
        | lazy component loading, graph-independent state ownership,
        | output aliasing, last-consumer release, pooled graph buffers
        v
C++ engine and Cactus kernels
```

There are three different kinds of metadata, and they deliberately have
different responsibilities:

- A **model profile** describes what is known at transpilation time: supported
  modalities, graph components, legal fusion groups, prompt/media behavior,
  cache compatibility, and the desired runtime contract.
- A **component manifest** describes a generated graph mechanically: its input
  and output node IDs, logical names, weight bindings, cache-state nodes, and
  graph-specific metadata.
- A **runtime plan** describes relationships between components: routes,
  state producers and consumers, lifetimes, aliases, transfer policies, and
  when intermediate state can be released.

This separation lets graph operations remain ordinary graph operations while
the storage backing cache and cross-component activations belongs to the model
session rather than to whichever component happens to be loaded.

**Example.** A Gemma image request first executes `vision_encoder`. Its graph
publishes `image_features`, then unloads. `lm_encoder_media_chunk` later binds
that retained storage, produces `inputs_embeds`, and unloads. Decoder prefill
binds those embeddings and publishes its KV storage into the session arena.
Finally, `decoder_step` binds the same KV allocation and generates tokens. Four
component graphs participate, but live state is not permanently owned by any
one of them.

## 2. Generic models versus registered optimized models

**Example.** `meta-llama/Llama-3.2-1B-Instruct` was tested through the unknown
model contract and produced one conservative full-context component. Gemma 4,
which is registered, produces modality encoders, prefill components, a decoder
step, cache bindings, and runtime aliases. Both use the same pipeline; the
difference is how much verified architectural metadata is supplied.

### 2.1 Profile resolution

The CLI resolves a model ID in `python/cactus/cli/transpiler.py`.

If the model ID is present in `ModelProfiles/profiles.py`, Cactus always uses
that registered profile. Generic-only flags are rejected for a registered
model, preventing a user from accidentally disabling the carefully tested
Gemma, LFM, Whisper, or Parakeet plan.

If the model ID is unknown, the user can declare its minimum execution
contract:

```bash
cactus convert MODEL_ID OUTPUT_DIR \
  --modalities text \
  --task causal-lm \
  --cache-style none
```

Available contract fields are represented by `GenericTranspileContract` in
`ModelProfiles/models.py`:

- `--modalities`: `text`, `vision`, and/or `audio`.
- `--task`: `causal-lm` or `speech-seq2seq`.
- `--cache-style`: `none`, `dynamic-kv`, or `encoder-decoder-kv`.
- `--fusion-groups`: an optional comma-separated allow-list for extra fusion
  families.

The default generic causal-LM contract is deliberately `cache-style none`.
It exports `decoder_full_context` and recomputes the full prefix for each new
token. This is slower than cached decoding, but it makes no assumptions about
the model's cache container, layer ordering, or attention layout. It is the
safe “does this arbitrary architecture run?” baseline.

`dynamic-kv` is the next optimization step. It enables generic structurally
matched cached attention and emits prefill/decode components, but should be
selected only when the model exposes a compatible dynamic key/value cache.

**Example.** These commands select different levels of assumption:

```bash
# Safest first attempt: one full-context graph, no cache assumptions.
cactus convert acme/NewCausalLM weights/new-lm \
  --modalities text --task causal-lm --cache-style none

# Follow-up after inspecting the exported cache structure.
cactus convert acme/NewCausalLM weights/new-lm-cached \
  --modalities text --task causal-lm --cache-style dynamic-kv
```

Passing generic flags to `google/gemma-4-E2B-it` instead fails early because
Gemma already has a registered optimized profile. Omitting the flags selects
that profile automatically.

### 2.2 Representative input construction

Generic input creation is in:

- `Converter/input_processor.py`
- `Converter/models.py:build_processor_kwargs`
- `Converter/constants.py:MODALITY_INPUT_PATH`

The converter first tries `AutoProcessor`, then `AutoTokenizer`, using locally
cached files before allowing a download. It always supplies default text. It
adds `python/cactus/assets/test_monkey.png` only when vision was declared and
adds `python/cactus/assets/test.wav` only when audio was declared. The WAV is
normalized to mono FP32 at 16 kHz. Processor-returned tensor arguments are
passed directly to `torch.export`.

This is better than fabricating arbitrary tensor shapes because the model's
own processor decides token IDs, media dimensions, masks, and modality-specific
inputs. Synthetic input remains available for profiles that explicitly need a
fixed export-only representation, such as Parakeet.

**Example.** For `--modalities text,vision`, the logical call is equivalent to:

```python
processor(
    text="<image-token> Describe this input.",
    images=Image.open("test_monkey.png").convert("RGB"),
    return_tensors="pt",
)
```

No WAV is opened. For `--modalities text,audio`, the processor receives the
normalized `test.wav`, `sampling_rate=16000`, and default text; the monkey image
is never loaded.

### 2.3 Tested generic baseline

The following bundles were regenerated using existing CQ weights and the
generic text/no-cache contract:

| Model | Transpilation wall time | Unsupported nodes | Runtime smoke test |
|---|---:|---:|---|
| SmolLM2-135M-Instruct | 12.73 s | 0 | Passed |
| Llama-3.2-1B-Instruct | 9.00 s | 0 | Passed |
| Qwen2.5-0.5B-Instruct | 10.52 s | 0 | Passed |

These times include Hugging Face model loading, export, simplification,
lowering, and bundle writing, but not weight conversion. Their slow decode
rates are expected: the tested contract intentionally performs full-context
recomputation.

**Example.** The Smol check produced this essential manifest contract:

```json
{
  "family": "generic_text",
  "metadata": {
    "profile_source": "generic",
    "cache_style": "none",
    "runtime_execution_strategy": "full_context_recompute"
  },
  "components": [
    {"component": "decoder_full_context", "unsupported_nodes": []}
  ]
}
```

The bundle generated tokens successfully. This check proves compatibility,
not competitive decode speed.

## 3. Converter and lossless IR capture

The converter in `Converter/models.py` turns a `torch.export.ExportedProgram`
into a `LayerMap`. It records:

- stable node index and name;
- node kind and fully qualified operator target;
- positional and keyword arguments with node references preserved;
- users, tensor metadata, and module stack;
- graph input/output signature and persistent-buffer information;
- inference task (`prefill_with_cache`, `decode_with_cache`, or
  `prefill_no_cache`).

Non-finite floating-point values are serialized as `"Infinity"`,
`"-Infinity"`, or `"NaN"`, not JSON null. This matters for attention masks:
turning `-inf` into zero silently changes causal masking and model output.

Cache examples are produced by `Converter/cache_utils.py`. Rather than treating
past keys and values as ordinary anonymous tensor inputs, conversion assigns
semantic cache annotations: cache kind, layer index, tensor index, key/value
role, sequence dimension, window size, and related shape information. These
annotations survive IR rewriting and eventually become runtime cache bindings.

Model loading is profile-controlled. `LOAD_STRATEGIES` tries the most specific
Hugging Face auto class first and falls back carefully. Export patches are
scoped context managers for model-library behavior that cannot be exported
directly, such as grouped MoE operations or specific multimodal feature paths.

**Example.** A simplified captured linear node looks like this:

```json
{
  "index": 42,
  "name": "q_proj",
  "node_type": "call_function",
  "target": "aten.linear.default",
  "args": [{"node": "hidden_states"}, {"node": "q_proj_weight"}],
  "users": ["view_q"],
  "tensor_output_meta": {"shape": [1, 32, 2048], "dtype": "torch.float16"},
  "module_stack": [{"module_path": "model.layers.0.self_attn.q_proj"}]
}
```

The node references are structural. A mask constant of negative infinity is
stored as `"-Infinity"` and restored during lowering.

## 4. Fusion matching and repeated IR simplification

**Example.** A raw exported decoder block can contain hundreds of nodes for
normalization, projections, cache concatenation, masks, softmax, MLP, and layout
wrappers. Repeated simplification may reduce these to semantic nodes such as
`RMS_NORM`, `QKV_TQ_FUSED`, `ATTENTION_CACHED`, and `DENSE_MLP_TQ_FUSED`, while
leaving any unmatched operation intact for ordinary lowering.

### 4.1 Why simplification is iterative

A fusion replaces multiple nodes with one semantic node. That replacement can
expose another pattern that did not exist in the original graph. For example,
fusing a linear projection or removing a clone can make an adjacent attention
or MLP pattern contiguous.

`IR/simplify_ir.py:simplify_repeated` therefore executes at least two complete
rounds and continues until the serialized graph stabilizes, up to eight rounds.
Within each round, `simplify` has two phases and up to three reverse-topological
fusion passes per phase:

1. Match the exported graph exactly as captured. Some LFM patterns include
   clones or contiguous nodes and would be destroyed by cleaning too early.
2. Remove structural no-ops.
3. Match again so cleanup-exposed patterns can fuse.
4. Run final no-op removal, logits fusion, and transpose composition.

This is a fixed-point algorithm. Termination occurs when a complete round makes
no JSON-visible change. The upper bounds prevent a malformed rewrite from
looping forever.

**Example.** An exported chain may be:

```text
linear -> clone -> view -> divide -> tanh -> multiply
```

Round one fuses or normalizes the linear and removes the clone. That exposes a
single-consumer LM-head softcap chain. Round two replaces the remaining chain
with `cactus.logits_tq_softcap`. A later stability check produces identical JSON
and stops. A single simplification pass would miss the larger rewrite.

### 4.2 Reverse-topological selection

`IR/simplify_ir.py:rev_top_sort` walks nodes from outputs toward inputs. For
each root it obtains candidates indexed by root operator, sorts more-specific
patterns ahead of smaller/generic patterns, and accepts the first complete
match. A consumed-node set prevents overlapping replacements.

Candidate priority considers special matchers, graph node count, edge count,
constraints, required attributes, and whether the fusion is merely a direct
one-node lowering. This lets a full MLP or attention fusion win before a small
linear fusion consumes part of it.

Ordinary fusion graphs use declarative node, edge, attribute, and shape
constraints from `Fusions/`. More complex layouts use structural matchers in
`IR/special_fusions.py`. A match produces `FusionResult`, including matched
nodes, external inputs, extracted attributes, and cache annotations.
`IR/models.py:apply_fusions_to_graph` validates non-overlap, creates semantic
Cactus nodes, rewrites references, rebuilds parent/child edges, and prunes dead
nodes.

**Example.** For sibling `q_proj`, `k_proj`, and `v_proj` nodes, a three-way QKV
match competes with three independent linear matches. The larger constrained
candidate wins. Its projection nodes are marked consumed, preventing smaller
overlapping matches, and one `cactus.qkv_tq_fused` replacement publishes the
three results.

### 4.3 Fusion groups and safety controls

Every fusion has one or more fields such as `generic`, `attention`, `mlp`,
`gemma4_attention`, `lfm_moe`, or `whisper_attention`. A model profile selects
fields through `fusion_fields` and can disable a family or an individual fusion
with `disabled_fusion_fields` and `disabled_fusions`.

This makes optimization opt-in by architectural knowledge. Generic fusions are
available broadly; layout-sensitive fusions run only for profiles that declare
them. A fusion can also restrict supported inference modes and modalities.

**Example.** A conservative generic text profile can select:

```python
fusion_fields=("generic", "linear", "normalization", "attention", "mlp")
```

Gemma additionally selects `gemma4_attention`, `gemma4_rope`, and
`gemma4_mlp`. A Whisper graph therefore cannot accidentally receive a Gemma
RoPE-table rewrite merely because a small local subgraph resembles it.

### 4.4 Structural no-op elimination

`IR/models.py:remove_noop_nodes_from_graph` removes identity-like view, clone,
contiguous, and equivalent passthrough nodes when their semantics permit it. It
builds a transitive replacement map, rewrites all references, rebuilds edges,
and then performs dead-code elimination.

Fewer nodes mean fewer kernel dispatches, smaller serialized graphs, and fewer
temporary buffers. More importantly, cleanup exposes larger semantic patterns
to later fusion rounds.

**Example.** When semantics and shapes prove that every wrapper is inert,

```text
x -> contiguous -> clone -> view(same shape) -> rmsnorm
```

becomes `x -> rmsnorm`. Downstream references point directly to `x`, and none
of the wrapper outputs needs a runtime buffer.

### 4.5 Transpose-chain composition

`IR/models.py:collapse_transpose_chains_from_graph` handles adjacent full-rank
permutations algebraically. If permutation `p` is followed by `q`, the composed
permutation is:

```text
r[i] = p[q[i]]
```

If `r` is the identity, the entire chain is replaced by its source. Otherwise
the chain becomes one transpose with `r`. Dead intermediates are pruned.

This was particularly valuable for LFM-VLM, where component extraction exposed
layout chains that were obscured in the full graph. It reduced redundant
transpose work and temporary memory without relying on model names. The same
cleanup is run again after component splitting in
`Generator/component_split_builder.py`, because partition boundaries can expose
new chains.

**Example.** If a rank-four tensor receives permutations
`p=(0,2,1,3)` and then `q=(0,2,1,3)`, their composition is
`r[i]=p[q[i]]=(0,1,2,3)`. Both transposes disappear. A nonidentity composition
would remain as one transpose instead of two.

### 4.6 Generic cached-attention fusion

`generic_cached_attention` in `Fusions/fusions.py` and its matcher in
`IR/special_fusions.py` recognize decomposed causal attention structurally:

- new K/V projections;
- concatenation with past K/V;
- query/key score matmul and scaling;
- causal/mask application;
- softmax;
- probability/value matmul;
- compatible layout wrappers.

The matcher identifies the cache concatenations and key/value roles rather
than checking for “Llama” or another family name. The replacement is
`cactus.attention_cached`, with native `kv_cache_state` and `kv_cache_append`
operations. This prevents cache storage from flowing into ordinary reshape or
concatenation logic, which previously caused shape failures when the physical
cache capacity exceeded the logical sequence length.

The structural approach is the important generalization: future causal models
with the same semantics can use `--cache-style dynamic-kv` without adding a
model-specific profile. A profile is needed only if layout or cache semantics
are genuinely different.

**Example.** A decomposed decoder can export:

```text
past_k + new_k -> cat -> K
past_v + new_v -> cat -> V
Q @ transpose(K) -> scale -> causal mask -> softmax -> probabilities
probabilities @ V -> context
```

The fusion turns the two concatenations into native cache appends and the
attention subgraph into `attention_cached(Q, key_state, value_state)`. A cache
with physical capacity 2048 may then have logical length 17 without being
reshaped as an ordinary 17-position tensor.

### 4.7 Model-specific attention matchers

`IR/special_fusions.py` contains the layout-aware cases:

- **Gemma 4 attention:** recognizes prefill, decode, and vision layouts,
  including wrapper views/transposes, grouped-query head shapes, masks, logit
  caps, and sliding-window behavior.
- **Whisper attention:** recognizes encoder attention, decoder self-attention,
  and cached decoder attention while preserving cross-attention semantics.
- **LFM BMM masked attention:** recognizes the decomposed batched-matmul layout
  used by LFM vision and language blocks.
- **Gemma RoPE table lookup:** replaces repeated decomposed cosine/sine table
  indexing and layout operations with the native RoPE path.

These matchers validate semantic structure and attributes; model/profile checks
only guard known-safe activation. They do not blindly replace nodes by name.

**Example.** Gemma can wrap a query as `view -> transpose -> contiguous` before
score matmul. The matcher walks through allowed layout wrappers, reconstructs
the effective `[batch,time,heads,head_dim]` layout, and validates K and the mask
against it. An unexpected data-changing operator causes the match to fail and
leaves the graph decomposed.

### 4.8 MLP, QKV, and sibling-projection fusions

The fusion registry includes:

- decomposed RMSNorm and LayerNorm variants;
- decomposed SiLU and GLU;
- SwiGLU and Gemma GEGLU MLPs;
- direct and decomposed attention;
- linear variants (`linear`, transposed, `addmm`, and bias forms);
- convolution, recurrent, DSP, MoE, sampling, and cache patterns.

After general simplification, `fuse_decode_qkv_projections` finds sibling Q, K,
and V linears that consume the same hidden tensor and belong to the same
self-attention module. It replaces them with `cactus.qkv_tq_fused`.
`fuse_decode_projection_pairs` does the equivalent for compatible pairs.

The runtime kernels share activation transformation/quantization across the
sibling CQ projections. They still evaluate distinct weight matrices, so the
mathematics is unchanged, but they avoid repeating work that depends only on
the common activation.

**Example.** Independent decode projections do this three times:

```text
transform(hidden); CQ_matmul(Wq)
transform(hidden); CQ_matmul(Wk)
transform(hidden); CQ_matmul(Wv)
```

`QKV_TQ_FUSED` transforms `hidden` once and evaluates `Wq`, `Wk`, and `Wv` from
that representation. Gate/up projection pairing applies the same algorithm to
two matrices.

### 4.9 LM-head projection plus softcap

Gemma logits use:

```text
logits = cap * tanh((hidden @ W^T) / cap)
```

`IR/models.py:fuse_logits_softcap_from_graph` matches the LM-head linear or
matmul, optional layout wrappers, scalar divide, tanh, and scalar multiply. It
requires a single-consumer chain and equal positive divide/multiply caps, then
emits `cactus.logits_tq_softcap`.

`Generator/lowering_basic_ops.py:lower_logits_tq_softcap` lowers CQ weights to
the fused graph operator and preserves a decomposed fallback for non-CQ
weights. It also slices decoder-prefill hidden state to the final row before
the vocabulary projection, avoiding logits for prefix rows that will never be
sampled.

CQ conversion may store `W` with a scale factor. The lowering passes the
inverse scale to the kernel before softcap. This ordering matters because
softcap is nonlinear: scaling after tanh would not be equivalent.

**Example.** If the stored projection is `2.0` but the conversion scale makes
the model projection `2.0*16=32.0`, with cap 30 the correct result is
`30*tanh(32/30)`, approximately `23.65`. Scaling after tanh would compute
`16*(30*tanh(2/30))`, approximately `31.95`, which violates the intended cap.

## 5. Component partitioning and chunked prefill

Component splitting lives in `Generator/component_split_*.py`. A split spec
declares:

- which subgraph nodes belong to the component;
- placeholders and logical input names;
- published outputs and logical output names;
- side-effect/cache nodes that must remain even when not dataflow outputs;
- reference aliases across the cut;
- fixed chunk size and component metadata.

The generator rebuilds a valid standalone graph, adds explicit output nodes,
retargets fixed sequence dimensions, re-runs local structural cleanup, lowers
it, and writes a graph-specific manifest.

**Example.** A decoder split can publish logits while retaining cache mutation
as a side effect:

```python
ComponentSplitSpec(
    name="decoder_prefill_chunk",
    graph=prefill_graph,
    outputs=(OutputSpec("last_logits", "logits"),),
    side_effects=("layer_0_kv_append", "layer_1_kv_append"),
    placeholders=(inputs_embeds, attention_mask, position_ids),
    chunk_tokens=128,
)
```

The append nodes remain live even though they are not dataflow outputs.

### 5.1 Prefill/decode split

Optimized causal models use larger fixed-shape prefill components and a
single-token decoder component. Prefill amortizes graph dispatch and lets
matmul kernels operate on multiple rows; decode keeps only the minimal graph
and weights resident.

The engine's chunked-prefill algorithm in `cactus-engine/src/model.cpp`:

1. Selects the largest compatible text or shared-media prefill component whose
   declared `prefill_chunk_tokens` fits the prompt.
2. Executes all complete chunks.
3. For a sufficiently large safe tail, pads to a full chunk, snapshots the
   cache region affected by padding, executes once, and rolls back padded cache
   entries.
4. Executes a small or unsafe tail token-by-token.
5. Moves/aliases the resulting cache storage into `decoder_step`.

Sliding-window and recurrent cache constraints disable unsafe padded-tail
cases. The engine records logical, executed, padded, and scalar-tail token
counts so this behavior is measurable.

**Example.** For a 300-token prompt and a 128-token component, two full chunks
cover 256 tokens. A safe 44-token tail executes using the fixed-size graph with
padding; the cache snapshot/rollback retains the real tail positions and drops
the padded entries. The final scalar step supplies sampled logits. The model
still observes exactly 300 logical positions.

### 5.2 Separate LFM-VLM text prefill

LFM-VLM now has distinct media-compatible and text-only prefill components:

- `lm_encoder_text_prefill_chunk`
- `decoder_prefill_text_chunk`
- the existing shared media chunk path

Their chunk sizes can differ. For text-only prompts the engine chooses the
larger/faster text path; media prompts retain the layout required by the media
merge. The selected component pointers are returned in `ChunkedPrefillResult`
so the exact loaded graphs—not stale default pointers—are unloaded afterward.

This improves prefill without changing the overall component architecture and
prevents a graph-selection optimization from accidentally retaining the wrong
weight set.

**Example.** A 512-token text-only request selects
`lm_encoder_text_prefill_chunk -> decoder_prefill_text_chunk`. An image request
selects the media-compatible encoder and `decoder_prefill_chunk`. Both routes
publish identical semantic cache layer keys, so either can hand state to the
same `decoder_step`.

### 5.3 Prefix extension correctness

Media chunk graphs can initialize convolution or other state and are therefore
only valid at sequence position zero. On a warm prefix extension, the engine
keeps decoder-owned state and executes only the delta through `decoder_step`,
with positions offset by the existing cache length. This avoids corrupting
state by rerunning media initialization while preserving warm-prefix reuse.

**Example.** If the first request prefills positions 0–199 with an image and a
follow-up adds 30 text tokens, the engine retains the 200-token decoder cache,
assigns positions 200–229, and executes only those new tokens. It does not
rerun the image tower or reinitialize convolution state at position zero.

## 6. Model profiles as optimization policy

`ModelProfiles/models.py:ModelProfile` is the policy surface. Important fields
are:

- `components` and `inference_type`: partition and route selection;
- `supported_modalties`: which representative inputs to create;
- `input_strategy`, `load_strategy`, and `export_patches`: capture behavior;
- `fusion_fields`, disabled fields, and disabled fusion names;
- `PromptContract`: chat formatting, generation suppression, repetition scope;
- `MediaContract`: preprocessing, prompt placement, feature names, merge span,
  mask polarity, chunk-prefill eligibility, and fallback behavior;
- `CacheContract`: prefill/decode compatibility, state transfer, FP16 cache
  components, maximum sequence length, and full-retention layer exceptions;
- `RuntimeContract`: component execution strategy, state ownership, cache
  persistence, output alias policy, transfer policy, states, and aliases.

The profile is metadata, not hard-coded engine control flow. Generation copies
the relevant values into the runtime plan, and the engine reads those values.
This is the mechanism by which future model work should describe exact cache
layers and outputs to retain.

**Example.** A profile can declare a vision activation and its exact lifetime:

```python
StateContract(
    name="vision_features",
    kind="activation",
    producer="vision_encoder",
    consumers=("vision_projector",),
    lifetime="request",
    transfer="copy_or_alias",
    release_after_consumers=("vision_projector",),
    metadata=(("outputs", "vision_features"),),
)
```

This tells the engine both what must survive producer unload and when it dies.

Current specialized profiles cover:

- Gemma 4 multimodal attention, audio, vision, GEGLU, cache, and chunk routes;
- Whisper encoder/decoder and cross-attention state;
- Parakeet TDT encoder, recurrent decoder, and head;
- LFM-VLM vision/text routes and dynamic KV cache;
- LFM-MoE attention, short convolution state, grouped MoE, and dynamic KV;
- registered text/Qwen-compatible profiles where applicable.

## 7. Runtime plan and graph-independent ownership

**Example.** During one request, the runtime plan can keep decoder KV state for
the entire sequence, keep image features only until media embedding, alias
`inputs_embeds` into prefill, and unload all three producer graphs. The plan
describes those different lifetimes even though every value is represented by
the same underlying `TensorStorage` abstraction.

### 7.1 Runtime plan schema

The typed schema is in `RuntimePlan/models.py`:

- `RuntimeComponent`: graph, logical I/O, exact node IDs, constant bindings,
  cache-state bindings, warnings, and metadata.
- `CacheStateBinding`: stable layer key, key node, value node, cache kind, and
  tensor indices.
- `RuntimeRoute`: named component edges for an inference mode.
- `RuntimeState`: producer, consumers, kind, lifetime, transfer policy,
  persistence flag, required flag, explicit release consumers, and metadata.
- `RuntimeAlias`: source component/output, target component/input, policy,
  lifetime, copy fallback, resolved node IDs, and storage-stability proof.

Generation writes both `runtime_plan.json` (diagnostic and complete) and
`components/manifest.json` (engine-facing). Keeping them derived from the same
typed objects prevents plan/manifest drift.

**Example.** A resolved cross-component edge can look like:

```json
{
  "source_component": "audio_encoder",
  "source_output": "encoder_hidden_states",
  "target_component": "decoder_cross_kv",
  "target_input": "encoder_hidden_states",
  "policy": "alias_if_compatible",
  "fallback": "copy",
  "source_node_id": 418,
  "target_node_id": 7
}
```

Logical names explain the contract; concrete IDs make binding unambiguous.

### 7.2 Why cache cannot belong only to a component graph

Prefill and decode are different components with different weight residency.
If KV cache storage belongs exclusively to the prefill graph, unloading that
graph either destroys the cache or forces a full copy into the decoder graph.
The same issue applies to vision/audio features and encoder hidden states.

The new ownership model is:

```text
Model session
  StateArena
    kv:layer:0:key   -> shared TensorStorage
    kv:layer:0:value -> shared TensorStorage
    conv:layer:3     -> shared TensorStorage

  Runtime state store
    image_features         -> TensorStorage + shape/type/lifetime
    encoder_hidden_states  -> TensorStorage + shape/type/lifetime

  Loaded component graph
    cache-state/output node BufferDesc -> references the shared storage
```

Graph operations access state normally through their input/cache-state node.
The difference is only storage ownership: the node's `BufferDesc` is bound to
a session-owned `TensorStorage` before execution.

**Example.** Prefill node 900 and decode node 120 can both mean “layer 6 key
cache.” Their numeric IDs are unrelated. Their manifests assign the semantic
layer key `kv:6`, so the engine publishes under `kv:6:key` and binds that same
storage to decode node 120.

### 7.3 StateArena cache transfer

`StateArena` is in `cactus-engine/src/engine.h`; publishing and binding are in
`cactus-engine/src/model.cpp`.

Each cache annotation becomes a stable key such as `kv:12:key` or a conv-state
key. When a producer component finishes, `export_tensor_storage` returns a
`shared_ptr<TensorStorage>` and the engine publishes it in the arena. Loading a
consumer calls `bind_tensor_storage` on the consumer's corresponding cache
node.

Because the producer graph and StateArena share ownership, resetting the graph
does not free the cache. Because the consumer binds the same storage, there is
normally no cache memcpy. The old copy path remains available as a safety
fallback and can be forced for comparison with `CACTUS_DISABLE_SHARED_STATE`.

KV, convolution, and recurrent state share this abstraction but retain
different native graph operators:

- `KV_CACHE_STATE`, `KV_CACHE_APPEND`, `ATTENTION_CACHED`;
- `CONV_CACHE_STATE`, `CONV_CACHE_APPEND`, `CONV_CACHE_INITIALIZE`;
- `RECURRENT_CACHE_STATE`, `RECURRENT_CACHE_WRITE`.

This is the generic cache object in practice: common storage ownership and
binding, with type-specific update semantics.

**Example.** One cache allocation follows this ownership sequence:

```text
prefill allocates storage S -> KV_CACHE_APPEND writes S
StateArena publishes shared_ptr(S)
prefill unloads; StateArena still owns S
decoder binds shared_ptr(S) -> ATTENTION_CACHED reads/appends S
```

There is no `memcpy(S_prefill, S_decode)`.

### 7.4 Persistent component outputs

`runtime_state_store_` retains only outputs declared by a runtime alias or a
matching `StateContract`. `persist_component_outputs` exports durable storage
when possible; otherwise it copies bytes as a compatibility fallback. Stored
tensors include shape, precision, producer, kind, and lifetime.

When the consumer loads, `bind_runtime_state_inputs` first attempts a storage
alias. If shapes/storage are incompatible, it performs a typed copy into the
consumer input. This makes zero-copy an optimization rather than a correctness
requirement.

An explicit runtime alias is resolved to concrete source/output and
target/input node IDs during plan generation. Zero-copy based solely on that
alias is enabled only when metadata proves `storage_stable`; resolving node IDs
proves the logical edge, not the lifetime of the producer's allocation. The
engine also recognizes a small set of safe floating-point feature outputs such
as `inputs_embeds`, `encoder_hidden_states`, and image/vision features. It
specifically rejects integer input-backed passthrough tensors from that
implicit fast path.

**Example.** A stable FP16 vision tensor shaped `[1,256,1152]` and an identical
projector input can share storage, incrementing `media_aliased_bytes`. If the
consumer expects FP32 or a different shape, the fallback converts/copies into
its input and increments `media_copied_bytes`.

### 7.5 Component unload behavior

`Model::unload_component_graph` now follows this order:

1. Persist only declared live outputs.
2. Release graph-local runtime buffers.
3. Clear input buffers and destroy the graph/weight bindings.
4. Apply runtime last-consumer releases.
5. Update resident-weight accounting.

The cache and live output storage survive through `shared_ptr` ownership in the
session stores. Unrelated intermediates and component weights are released.
This is how Cactus keeps RAM low while avoiding repeated cache/output copies.

**Example.** Unloading a 1 GB vision component may leave only a declared 72 MB
feature tensor. Weights and temporary attention buffers disappear immediately.
After the projector consumes the feature, its last-consumer rule releases the
remaining 72 MB.

## 8. Last-consumer release and buffer pooling

There are two levels of lifetime analysis.

**Example.** The graph executor can release an attention score tile immediately
after its local consumer, while the model runtime retains the final
`image_features` graph output until a later component executes. Local liveness
handles nodes; runtime-plan liveness handles semantic values between graphs.

### 8.1 Inside one graph

`cactus-graph/src/execute.cpp` computes each node's use count and final consumer
index before execution. It builds `release_after[last_use]` lists. Immediately
after that consumer executes, the producer buffer is returned to the graph's
buffer pool.

The release algorithm excludes:

- graph inputs;
- KV, convolution, and recurrent state nodes;
- explicit persistent nodes;
- published/retained graph outputs.

Slice and index can alias their input. The analysis extends the base tensor's
lifetime to the view's last consumer, or keeps it through cleanup for an
unconsumed alias, preventing use-after-free.

This changes peak temporary memory from “sum of every node output” toward “sum
of simultaneously live values.” Reusing pooled buffers also reduces allocator
traffic and page churn.

**Example.** In `A -> B -> C`, A is released after B and B after C. Their pool
blocks can be reused by a later independent branch `D -> E`. If C is a retained
output, only C survives cleanup. In `A -> slice(A) -> C`, A instead remains live
through C because the slice may point into A's storage.

### 8.2 Across components

`StateContract.release_after_consumers` is serialized into the runtime plan.
After a named consumer component unloads,
`release_runtime_states_after_consumer` deletes the associated outputs from the
runtime store and media arena. Examples include:

- Gemma image/audio features after the media encoder/merge components;
- LFM vision features after `vision_projector`;
- Whisper encoder hidden state after cross-KV preparation;
- Parakeet encoder hidden state after `asr_decoder`.

This is explicit semantic last-consumer metadata. The runtime does not guess
that a multimodal feature is dead based on allocation order.

**Example.** Whisper releases `encoder_hidden_states` after
`decoder_cross_kv`. Hundreds of decoder steps can still use the smaller derived
cross-K/V state without retaining the raw encoder activation.

## 9. Attention memory and speed optimizations

**Example.** For vision attention with `Q=K=2520`, a full FP32 score matrix is
about 25.4 MB per head before counting probabilities or concurrent heads.
Streaming a 64-score block needs only 256 bytes of FP32 scores per active query,
plus its output accumulator, while producing the same normalized result.

### 9.1 Streaming masked attention

The attention kernels in `cactus-kernels/src/attention.cpp` process keys and
values in fixed-size blocks. They do not materialize the full
`[query_length, key_length]` score and probability matrices.

For each query, the kernel maintains an online softmax state:

```text
m = running maximum score
l = running sum of exp(score - m)
o = running weighted value accumulator
```

For a new block with maximum `m_b`, the combined maximum is
`m_new = max(m, m_b)`. The previous accumulator and sum are rescaled by
`exp(m - m_new)`; block values are scaled by `exp(score - m_new)` and added.
After all blocks, `o` is divided by `l`.

This is mathematically equivalent to stable softmax but requires only a small
score block plus the output accumulator:

- old temporary memory: `O(Q * K)` scores and often another `O(Q * K)`
  probability tensor;
- streaming temporary memory: `O(block_size + value_dim)` per active query.

Causal, sliding-window, explicit mask, and logit-cap tests are applied while
each score block is produced. Masked entries become negative infinity and
never enter the weighted accumulation. NEON performs FP16 loads and FP32 dot
accumulation where appropriate.

This is especially important for Gemma vision attention at sequence length
2520, where full score/probability matrices consumed substantial time and RAM.

**Example.** Process scores `[1,2]` and `[3]` as two blocks. After the first
block, `m=2` and `l=exp(-1)+1`. The second block raises the maximum to 3, so the
old sum and weighted accumulator are multiplied by `exp(2-3)`, then the new
weight `exp(3-3)=1` is added. The final denominator equals
`exp(1-3)+exp(2-3)+exp(3-3)`, exactly the stable full softmax denominator, but
the three scores never coexist in a full matrix.

### 9.2 Cached attention

`ATTENTION_CACHED` reads the logical cache length and absolute offset from the
native KV state. It attends directly over valid cache storage plus the current
query, applies causal/window constraints using absolute positions, and updates
the cache through native append operations. No ordinary concat tensor for the
entire history is required per token.

This improves decode from repeated `O(sequence_length)` cache copies to an
append plus streaming read, and it makes cache capacity distinct from logical
shape.

**Example.** At decode position 513, a cache may have capacity 4096 and logical
length 513. The new K/V row is written at logical position 513, attention reads
only positions 0–513, and the logical length becomes 514. The engine neither
concatenates a new 514-row tensor nor touches unused capacity 514–4095.

## 10. Quantized projection and MLP kernels

**Example.** One decoder layer can use a triple projection for Q/K/V, cached
streaming attention, a paired gate/up projection, a fused activation/product,
and a CQ down projection. The transpiler creates the semantic grouping; the
kernel layer decides whether the concrete CQ layout supports its fastest shared
transform path or needs the compatible fallback.

### 10.1 Shared activation transform for projection pairs/triples

`cactus_quant_matmul_pair` and `cactus_quant_matmul_triple` are declared in
`cactus-kernels/cactus_kernels.h` and implemented with the CQ matmul code.

CQ4 interleaved matrices require processing/transforming the activation into a
form consumed by the quantized dot-product kernel. Q, K, and V—or MLP gate and
up—share that activation. The pair/triple kernels perform the activation-side
work once and evaluate multiple independent weight matrices from it.

Fallback to independent matmuls is automatic for incompatible layouts or
orthogonal CQ flags. This preserves support while optimizing the common layout.

The graph operators are:

- `QKV_TQ_FUSED`;
- `PROJECTION_PAIR_TQ_FUSED`;
- `DENSE_MLP_TQ_FUSED`.

**Example.** If transforming a 1536-element decode activation costs `T` and
each CQ weight evaluation costs `M`, three independent projections cost roughly
`3T+3M`. The shared triple path costs `T+3M`. It does not make the weight dot
products disappear; it removes two redundant activation transforms.

### 10.2 Dense MLP fusion

`DENSE_MLP_TQ_FUSED` combines:

```text
gate = hidden @ W_gate^T
up   = hidden @ W_up^T
gate = activation(gate / gate_input_scale)
gate = gate * product_scale
mixed = gate * up
output = mixed @ W_down^T
```

Gate/up use the paired CQ projection when compatible. The down projection
cannot share the same activation transform because its input is the nonlinear
product.

Weight scale factors are folded into `gate_input_scale` and `product_scale` by
`Generator/lowering_special_ops.py`. This preserves converted-model scaling
without adding independent graph nodes.

**Example.** For `hidden=[1,1,1536]`, gate/up weights producing 8960 values, and
a down weight returning 1536 values, the fused operator owns two 8960-value
work buffers and one 1536-value result. It does not publish the gate, up,
activation, or product as graph nodes, so those intermediates cannot each
acquire separate graph-lifetime allocations.

### 10.3 Fused scaled GELU and multiply

Gemma GEGLU previously made four passes over the intermediate vector:

1. gate scale;
2. GELU;
3. product scale;
4. gate/up Hadamard product.

`cactus_gelu_scaled_multiply_f16` in `cactus-kernels/src/nn.cpp` performs the
same staged FP16 operations in one parallel NEON loop. Eight FP16 values are
loaded at a time, converted to FP32 for the tanh-based GELU approximation, then
rounded back to FP16 at the same semantic boundaries before scaling and
multiplication.

The exact staged rounding is intentional: moving all arithmetic to FP32 until
the end would be faster-looking but could alter token selection. The unit test
compares every FP16 output bit against the original four-kernel sequence.

`cactus-graph/src/ops_nn.cpp:compute_dense_mlp_tq_fused_node` uses the fast path
unless safety diagnostics or dense-MLP tracing are enabled. The diagnostic path
keeps the decomposed operations so intermediate non-finite and range checks
remain available.

On Gemma text decode this final fusion improved median decode from 33.06 to
36.09 tok/s in a controlled same-graph/same-weight five-run comparison (+9.2%)
while leaving active RAM flat.

**Example.** For one element with `gate=4`, `up=3`, `gate_scale=1/16`, and
`product_scale=1/16`, the fused kernel computes the same staged sequence as:

```text
g0 = fp16(4 * 1/16)
g1 = fp16(GELU(float(g0)))
g2 = fp16(g1 * 1/16)
out = fp16(g2 * 3)
```

The bit-exact test compares this reference with the one-pass NEON result over
8961 elements, including a scalar tail.

### 10.4 Fused CQ logits softcap

`LOGITS_TQ_SOFTCAP` performs the CQ vocabulary projection directly into the
output buffer and then applies:

```text
output[i] = cap * tanh((projection[i] * projection_scale) / cap)
```

`cactus_softcap_f16` is a parallel NEON kernel using the existing fast tanh
approximation and exact FP16 stage boundaries. It removes separate scale,
divide, tanh, and multiply graph dispatches and their temporary buffers.

The larger prefill win comes from projecting only the final hidden row. The
decode win is smaller because decode already has one row, but it still removes
elementwise passes and allocation traffic.

**Example.** A 128-token prefill with vocabulary size 256,000 previously
computed roughly 32.8 million logits even though sampling reads only row 127.
Slicing the hidden state first computes one 256,000-value row. The fused kernel
then applies the conversion scale and cap in place on that row.

## 11. Convolution layout optimization

LFM's decomposed causal depthwise convolution often operates in channel-first
`[N, C, L]` layout. The old path transposed to the generic channel-last layout,
ran convolution, then transposed back.

`CONV1D_CAUSAL_CHANNEL_FIRST` is wired through:

- fusion/lowering detection in `Generator/lowering_nn_ops.py`;
- Python graph binding in `python/cactus/bindings/cactus.py`;
- graph builder, FFI, serialization, dispatch, and `ops_conv.cpp`;
- kernel `cactus_conv1d_causal_depthwise_channel_first_f16` in
  `cactus-kernels/src/conv.cpp`.

The kernel indexes the source directly as `[n, c, l]`, computes each causal
window with dilation, and writes the same layout. This deletes two full tensor
transposes, reduces temporary memory, and improves locality for each channel's
time series.

The generic convolution lowering selects this operator from structural layout
and grouping properties, not a hard-coded LFM model ID.

**Example.** For input `[N=1,C=2,L=4]`, kernel width 3, and dilation 1, output
`y[0,1,2]` reads channel 1 positions 0, 1, and 2 and multiplies them by that
channel's three weights. Earlier positions use causal zero padding. Direct
`[N,C,L]` indexing avoids materializing `[N,L,C]` before and after the kernel.

## 12. Model-specific execution consequences

**Example.** Gemma and LFM-VLM both accept images, but their component routes
and attention layouts differ; Whisper and Parakeet both accept audio, but one
uses encoder-decoder attention while the other uses recurrent TDT state. Shared
runtime concepts are reused, while profiles describe these semantic
differences instead of forcing one hard-coded execution route.

### 12.1 Gemma 4

Gemma combines text, vision, and audio encoders with a shared language decoder.
The optimized profile provides separate text/media encoders, chunk-prefill
graphs, one-token media/text steps, dynamic/sliding KV state, streaming vision
attention, GEGLU fusion, QKV projection fusion, and LM-head softcap fusion.

Media features are retained only until their final LM-encoder consumers. KV
storage moves by ownership from prefill to decode, and decoder execution does
not retain the heavy media components.

**Example.** An image prompt follows this approximate route:

```text
image -> vision_encoder -> image_features
tokens + image_features -> lm_encoder_media_chunk -> inputs_embeds
inputs_embeds -> decoder_prefill_chunk -> logits + KV state
sampled token + KV state -> decoder_step -> next logits
```

After the second line, image features can be released; after prefill, media and
prefill weights unload while KV state remains.

### 12.2 LFM-VLM

LFM-VLM benefits most from streaming masked vision attention, transpose-chain
composition, a separate text-prefill route, and early release of vision tower
features. Text decode remains close to main while active RAM is dramatically
lower; image prefill is more than twice main's throughput in the current
benchmark.

**Example.** A text request bypasses `vision_encoder` and `vision_projector`
entirely and selects the larger text-only prefill component. An image request
executes the vision route, aliases `vision_features` into the projector when
compatible, releases them after projection, and uses streaming attention for
the long visual sequence.

### 12.3 LFM-MoE

The IR recognizes grouped expert layouts, router variants, optional token
clones, and SiLU variants. These become a native gated MoE operator rather than
many per-expert graph nodes. The profile also carries convolution and KV state
across prefill/decode. Decode is now faster than main, while prefill is roughly
equal.

**Example.** If routing selects experts 2 and 7 for one token, the native MoE
operation gathers those experts' packed gate/up/down weights, evaluates only
the selected experts, applies normalized routing weights, and accumulates their
outputs. The graph does not dispatch separate generic nodes for every unused
expert.

### 12.4 Whisper

Whisper is split into audio encoder, cross-KV preparation, decoder prefill, and
decoder step. Encoder hidden state is retained until cross-KV preparation,
then released. Cross-attention keys/values remain request state for decoder
steps. Whisper-specific attention matching handles encoder, cached self-, and
cross-attention layouts.

**Example.** `audio_encoder` produces 1500 hidden rows. `decoder_cross_kv`
projects those rows once into per-layer cross K/V, then releases the 1500-row
encoder state. Each subsequent decoder token reuses cross K/V while appending
only its self-attention K/V.

### 12.5 Parakeet TDT

Parakeet uses an audio encoder, recurrent ASR decoder, and prediction head.
Encoder state has an explicit final consumer and the TDT recurrent state has
sequence lifetime. Its runtime path is already dominated by model computation;
the current branch is approximately at parity with main while using slightly
less RAM.

**Example.** The audio encoder runs once and publishes encoder states. The TDT
decoder repeatedly updates its recurrent state and sends the current prediction
state to `asr_head`. Recurrent state survives for the sequence; encoder output
is released after the decoder has consumed it.

## 13. Serialization and ABI changes

New graph operators are appended to `OpType`, never inserted in the middle.
Appending preserves numeric IDs for existing serialized graphs. Each operator
must be updated consistently in:

1. `cactus-graph/cactus_graph.h` enum and public builder/FFI declarations;
2. `cactus-graph/src/builder.cpp` shape/parameter construction;
3. `cactus-graph/src/graph_ffi.cpp` C ABI wrapper;
4. `cactus-graph/src/execute.cpp` dispatch and name table;
5. `cactus-graph/src/io.cpp` and `param_io.cpp` when parameters require it;
6. Python ctypes declarations and `Graph` method;
7. transpiler lowering rule;
8. kernel implementation and tests.

The recently appended operators are:

- `QKV_TQ_FUSED`;
- `PROJECTION_PAIR_TQ_FUSED`;
- `CONV1D_CAUSAL_CHANNEL_FIRST`;
- `LOGITS_TQ_SOFTCAP`.

Runtime-plan cache bindings use stable semantic `layer_key` values, so cache
ownership does not depend on graph node IDs matching across components.

**Example.** If `PROJECTION_PAIR_TQ_FUSED` currently has numeric ID 82, adding
`CONV1D_CAUSAL_CHANNEL_FIRST` appends ID 83. Inserting the new value near the
old convolution enums would shift 82 and cause an existing serialized graph to
deserialize its projection-pair node as the wrong operation. Appending avoids
that corruption.

## 14. Instrumentation and correctness controls

Important instrumentation includes:

- `CACTUS_PROFILE_FILE`: per-component, per-operation timing and output size.
- `CACTUS_RUNTIME_PROFILE`: copied/aliased bytes, media bytes, component load
  and unload counts, resident weights, state/store peaks, and released bytes.
- `CACTUS_DISABLE_SHARED_STATE`: force the legacy cache-transfer behavior.
- `CACTUS_DISABLE_OUTPUT_ALIAS`: force copies for component outputs.
- dense-MLP tracing/safety controls used to inspect intermediate ranges and
  non-finite values.

Correctness coverage includes:

- exact kernel tests for softcap and fused scaled-GELU/product;
- channel-first causal-convolution comparison with the old layout path;
- graph serialization and operation tests;
- fusion reports and unsupported-node reports for every exported mode;
- cache reuse, prefix extension, invalidation, chunk-tail padding, streaming,
  tool-call, and multimodal engine tests;
- controlled old/new executable comparisons on identical graphs and weights.

The simplifier writes a `*.fusion_report.json` next to each simplified IR. It
contains before/after operator counts, applied fusions, and reasons candidates
did not match. This is the primary tool for deciding what fusion to add next.

**Example.** A profiling run can be launched as:

```bash
CACTUS_PROFILE_FILE=/tmp/ops.txt \
CACTUS_RUNTIME_PROFILE=/tmp/runtime.jsonl \
cactus run weights/gemma4-e2b-it \
  --prompt "Explain online softmax" --no-cloud-handoff
```

To measure the ownership optimization, repeat once with
`CACTUS_DISABLE_SHARED_STATE=1` and compare `copied_bytes`, `aliased_bytes`,
peak state bytes, and latency. This isolates cache transfer without changing
the transpiled graphs.

## 15. Full current-versus-main benchmark

Method: one warmup and three measured isolated processes per branch, alternating
branch order, 64 maximum generated tokens. RAM is native process peak RSS;
“active RAM” is the engine-reported active measurement. Speech models use
effective tokens divided by end-to-end transcription time because their native
decode-loop timer is near zero and produces meaningless million-token/s values.

| Scenario | Current prefill | Main prefill | Current decode/effective | Main decode/effective | Current total | Main total | Current active RAM | Main active RAM | Current peak RSS | Main peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma text | 82.49 | 63.54 | 35.88 | 30.39 | 2083 ms | 2494 ms | 240 MB | 854 MB | 1511 MB | 3372 MB |
| Gemma image | 60.09 | 42.89 | 30.52 | 29.79 | 6838 ms | 8826 ms | 1174 MB | 1171 MB | 2498 MB | 2603 MB |
| Gemma audio | 100.35 | 90.13 | 27.75 | 27.86 | 7137 ms | 8462 ms | 163 MB | 440 MB | 1547 MB | 1981 MB |
| Gemma image+audio | 82.04 | 67.82 | 26.39 | 27.70 | 10686 ms | 14823 ms | 162 MB | 241 MB | 1962 MB | 2167 MB |
| LFM-VLM text | 29.01 | 30.31 | 30.10 | 31.16 | 3020 ms | 2911 ms | 45 MB | 1140 MB | 1514 MB | 2542 MB |
| LFM-VLM image | 47.31 | 20.22 | 29.61 | 30.94 | 4149 ms | 6834 ms | 243 MB | 1360 MB | 1703 MB | 2757 MB |
| LFM-MoE text | 28.53 | 29.00 | 35.05 | 32.76 | 2707 ms | 2809 ms | 144 MB | 144 MB | 4107 MB | 4063 MB |
| Whisper audio | 13.08 | 10.71 | 192.88 effective | 160.61 effective | 306 ms | 374 ms | 124 MB | 201 MB | 162 MB | 239 MB |
| Parakeet audio | n/a | n/a | 88.54 effective | 86.55 effective | 1016 ms | 1040 ms | 240 MB | 242 MB | 616 MB | 620 MB |

Interpretation:

- Gemma text is 29.8% faster in prefill and 18.1% faster in decode, with 71.9%
  lower active RAM.
- Gemma image is 40.1% faster in prefill and 22.5% faster end-to-end.
- LFM-VLM image prefill is 134% faster and peak RSS is 38% lower.
- Whisper effective throughput is 20.1% higher and peak RSS is 32% lower.
- LFM-MoE decode is 7.0% faster; its large mapped weight footprint makes peak
  RSS approximately equal to main.
- LFM-VLM text remains 3–4% slower than main, but active RAM is 96% lower.
- Gemma image+audio decode is not directly comparable because current stopped
  at 30 output tokens while main generated 64; total latency and normalized
  prefill remain materially better.

The raw report is `/private/tmp/cactus_full_optimized_models_vs_main_2026-08-05.json`.

**Example.** Gemma text prefill speedup is calculated from medians as
`82.49/63.54 - 1 = 29.8%`. Active-RAM reduction is
`1 - 240/854 = 71.9%`. Medians are computed within each branch before deriving
the comparison; individual best runs are never compared against one another.

## 16. How to optimize a newly working generic model

Use this progression rather than adding architecture checks to generic code:

1. **Prove capture and lowering.** Use text/no-cache and inspect unsupported
   nodes and output plausibility.
2. **Enable cache structurally.** Try `dynamic-kv`; inspect cache annotations
   and the generic cached-attention fusion report. Add a new cache style only
   when semantics differ, not merely because the model has a new name.
3. **Profile operations.** Use `CACTUS_PROFILE_FILE` to locate actual time and
   allocation hotspots.
4. **Add declarative fusions.** Prefer a fusion graph with semantic constraints.
   Use a special matcher only for genuinely variable layout wrappers.
5. **Add a profile.** Declare modalities, safe fusion groups, component cuts,
   chunk sizes, cache compatibility, and runtime state/aliases.
6. **Add a kernel only when fusion is insufficient.** A kernel is justified
   when it eliminates materialization, shares expensive transforms, improves
   locality, or maps a semantic operation to a substantially better algorithm.
7. **Describe lifetimes.** Add exact producer outputs and
   `release_after_consumers`; do not retain every producer output.
8. **Validate equivalence.** Compare unfused/fused graphs on identical weights,
   run cache and prefix tests, then benchmark isolated processes against main.

**Example.** For a hypothetical `acme/FalconLike-350M`:

```text
1. Export with text/no-cache; it runs but decode is slow.
2. Export dynamic-kv; fusion report shows 18/24 attention layers fused.
3. Inspect the six misses; they contain an extra harmless layout wrapper.
4. Generalize the cached-attention wrapper matcher and add a structural test.
5. Profile; MLP now dominates, with a recognizable SwiGLU chain.
6. Enable the existing mlp fusion field rather than writing a model-name case.
7. Add a registered profile only when selecting chunk size and cache lifetime.
8. Compare generic, optimized, and main/reference outputs and benchmarks.
```

The model becomes specialized through metadata and reusable structural rules,
not through scattered `if "falcon" in model_name` checks.

## 17. Primary code map

| Concern | Primary files |
|---|---|
| CLI/profile resolution | `python/cactus/cli/transpiler.py`, `python/cactus/cli/__init__.py` |
| Hugging Face capture | `Converter/models.py`, `Converter/input_processor.py`, `Converter/cache_utils.py` |
| Declarative fusions | `Fusions/fusions.py`, `fusion_*.py`, `nodes.py`, `edges.py` |
| Matching and simplification | `IR/simplify_ir.py`, `IR/match*.py`, `IR/special_fusions.py` |
| Graph rewriting | `IR/models.py` |
| Component partitioning | `Generator/component_split_*.py` |
| Lowering | `Generator/lowering_*.py` |
| Runtime plan | `RuntimePlan/models.py` |
| Model policy | `ModelProfiles/models.py`, `ModelProfiles/profiles.py` |
| Session state/runtime execution | `cactus-engine/src/engine.h`, `cactus-engine/src/model.cpp` |
| Graph lifetime/pooling | `cactus-graph/src/execute.cpp`, `cactus-graph/cactus_graph.h` |
| Attention algorithms | `cactus-kernels/src/attention.cpp`, `attention_hybrid.cpp` |
| CQ projection kernels | `cactus-kernels/src/matmul.cpp` and related quant sources |
| New MLP/softcap kernels | `cactus-kernels/src/nn.cpp` |
| Channel-first convolution | `cactus-kernels/src/conv.cpp` |

**Example.** To trace one optimization end to end, start with
`IR/models.py:fuse_logits_softcap_from_graph`, follow its semantic target into
`Generator/lowering_basic_ops.py`, then the Python `Graph` binding, graph
builder/FFI/dispatch, `ops_nn.cpp`, and finally `cactus_softcap_f16` in
`cactus-kernels/src/nn.cpp`. The row in this table identifies the entry point
for each layer of that investigation.

## 18. Remaining performance boundary

After the latest MLP fusion, a representative Gemma decode step spends roughly:

- 13.7 ms in dense MLP;
- 5.4 ms in vocabulary projection and softcap;
- 4.4 ms in remaining matmuls;
- less than 1 ms each in QKV and cached attention.

There are no further obvious low-risk graph or transpiler fusions in that hot
path. Larger gains now require backend work: a redesigned CQ microkernel,
hardware-specific matrix acceleration, or an exact fused vocabulary
projection/top-k sampling algorithm that avoids producing all logits. Those are
larger projects with different correctness and portability tradeoffs; they are
not missing profile metadata or another simplification pass.

**Example.** Even eliminating every operation other than the 13.7 ms dense MLP
would cap a 34.5 ms decoder step at roughly `34.5/13.7 = 2.5x`, an impossible
upper bound because attention and sampling cannot actually vanish. A 20% faster
MLP kernel would save about 2.7 ms and improve the whole step by roughly 8%.
That scale of gain requires CQ backend work; another zero-cost view fusion
cannot produce it.

## 19. Final transpiler cleanup and canonicalization

The final cleanup removes representation code only where it does not affect
semantics. The audit initially identified `Fusions/fusion_direct.py` as a
candidate because it converts individual ATen nodes such as
`aten.view.default` into `cactus.view`. Regeneration and output-parity testing
showed that this canonicalization is significant: downstream matchers depend on
its normalized attributes and node identity. It was therefore retained. The
few raw aliases that lowering lacked, such as `aten.max.dim` and floor-mode
division, are also accepted in `Generator/constants.py` for generic graphs that
reach lowering without canonicalization.

**Example.** A raw graph containing:

```text
aten.view.default(x, [1, 128, 2048])
```

becomes `cactus.view` during simplification and then a Cactus view during
lowering. Although raw `aten.view.default` has a lowering rule, retaining the
canonical form ensures that shape captures and composite matcher boundaries are
identical for registered optimized models.

Fusion reporting also stops rerunning every unsuccessful matcher merely to
construct diagnostic `missed_candidates`. Reports retain input/output node
counts, operation counts, and applied fusion counts—the stable information used
to validate optimization coverage—without repeating much of simplification's
work after the result is already known.

**Example.** If 35 Gemma attention blocks fuse, the report still records
`gemma4_bmm_masked_attention: 35`. It no longer rematches every remaining node
against every rejected attention variant to guess why unrelated nodes did not
fuse.

The fusion-schema audit deliberately retained fields consumed by matching and
rewriting, including variadic input bounds, repeated-subgraph bounds, output
constraints, and cache constraints. Removing any of those changed which nodes
canonicalized and failed byte-level graph parity. Only genuinely redundant
paths outside those matching contracts were removed or consolidated.

**Example.** The direct `cat` fusion keeps `min_count=2`. Without that bound,
an exported one-input `aten.cat.default` was incorrectly canonicalized and the
LFM prefill graph changed. The parity test caught the difference, so the bound
and its matcher logic remain part of the architecture.

Runtime-plan metadata construction in `RuntimePlan/models.py` now uses one
dataclass-field mapper for prompt, media, cache, and runtime contracts. Explicit
aliases handle the handful of on-disk names that differ from Python field
names. This replaces four near-identical field-by-field serializers without
changing emitted values.

**Example.** `PromptContract.turn_start_token`,
`MediaContract.image_token_id`, and `CacheContract.max_cache_sequence_length`
all pass through the shared mapper. Tuple values are encoded consistently, and
an alias maps a field only when the runtime-plan schema uses a different key.

Generated component manifests now write only the canonical
`bound_constant_bindings` field and `graph` path. The duplicate legacy
`weight_bindings` and `graph_path` copies were removed, and the runtime-plan
builder consumes the canonical fields. Regenerated bundles therefore carry one
source of truth for each value.

**Example.** A quantized projection weight formerly appeared twice in the same
manifest under `bound_constant_bindings` and `weight_bindings`. It now appears
once, with the same node id, file path, precision, and binding kind used by the
runtime.

Finally, model boundaries that were structurally shared but encoded with exact
model dimensions were generalized. LFM vision boundaries use rank, module
ancestry, and fanout; Whisper cross-attention uses compatible rank/layout and
encoder sequence length. Model-specific behavior remains in profiles and
runtime contracts rather than duplicated size checks.

**Example.** The LFM vision position-grid metadata is now derived from the
captured tensor width. A 3B graph emits `16,16,1152`, while the 450M graph emits
`16,16,768`, using the same component plan and without a model-name branch.

### LFM-MoE chunked-prefill projection classification

`Generator/lowering_basic_ops.py:is_logits_projection` identifies the final
vocabulary projection so chunked prefill can compute only its last row. The
search is intentionally limited to the projection node and its weight/bias
operands. Searching backward through the activation input is incorrect for
models with tied input embeddings and output weights: that path eventually
reaches an `lm_head`-named weight even when the current operation is an
unrelated convolution or MLP projection.

**Example.** LFM-MoE's first convolution projection has shape
`[128, 2048] @ [6144, 2048]^T -> [128, 6144]`. Its activation originates at an
embedding tied to `p_model_lm_head_weight`. The old ancestor search therefore
sliced the projection input to `[1, 2048]`, after which the exported restore
view incorrectly attempted `[1, 6144] -> [1, 128, 6144]`. The narrowed check
leaves all 128 convolution rows intact and still slices the real
`model.lm_head` vocabulary projection to its final row.

## 20. Contiguous last-axis padding

`cactus-graph/src/ops_tensor.cpp` detects padding confined to the final,
contiguous axis. It fills the destination once and copies each complete input
row with one `memcpy`. The general coordinate-mapping implementation remains
available for padding on arbitrary axes.

**Example.** Whisper pads `[1, 384, 3000]` to `[1, 384, 3002]`. The general
loop previously computed a three-dimensional coordinate and issued a
two-byte copy 1,152,000 times. The specialized path performs 384 row copies;
the two profiled Whisper padding nodes fell from about 8.10 ms to 0.34 ms.

### Measured encoder effect

The specialized padding path removes coordinate bookkeeping from Whisper's
audio encoder. The remaining encoder cost is primarily matrix projection and
fused attention, not padding or graph bookkeeping. End-to-end transcription
benefits less because the decoder runs once per emitted token.

**Example.** On `test.wav`, the optimized encoder profile attributes roughly
33 ms to 24 matmuls, 18 ms to four attention operations, 4–6 ms to the two
convolutions and 0.34–0.46 ms to padding. This breakdown makes the next useful
work a matrix/attention kernel change, not another padding specialization.
