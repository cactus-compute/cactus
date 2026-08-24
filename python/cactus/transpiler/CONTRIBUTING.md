# Contributing to Transpiler (The First Triumvirate)

The Cactus transpiler turns a PyTorch model into a runnable Cactus bundle. Its consists of 3 key susbsystems:

1. **Crassus — Converter + Model Profiles** Captures the source model and describes its model-level contract. Generates the intermediary representation of the model through LayerMap object, which is representable through a JSON.
2. **Pompey — Fusions + IR** Defines fusions of basic .
3. **Caesar — Generator + Runtime Plan** lowers the simplified computation into Cactus graphs and describes how their components execute together.

The CLI announces these phases as it runs:

```text
Running converter (Crassus vX.Y.Z)
Running IR simplifier (Pompey vX.Y.Z)
Running Generator (Caesar vX.Y.Z)
```

Their versions live in `Converter/version.py`, `IR/version.py`, and
`Generator/version.py` respectively.

Keep model-specific policy in `ModelProfiles/` or in a narrowly scoped
structural matcher. Shared conversion, IR, fusion, lowering, and runtime-plan
code must not select behavior from Hugging Face repository-name substrings.

## Crassus: Converter and Model Profiles

Crassus answers two questions:

1. What model and inputs should PyTorch export?
2. What capabilities and model-specific contracts should the remaining phases
   apply?

### Converter responsibilities

`Converter/` loads the source model, prepares representative inputs, exports its
PyTorch graph, and serializes that graph as a `LayerMap`. The raw JSON form is
the inspectable boundary between source-model execution and Cactus graph
optimization.

Important files:

- `Converter/convert.py` orchestrates graph export and writes raw `LayerMap`
  JSON.
- `Converter/models.py` defines the exported graph representation and model
  loading paths.
- `Converter/input_processor.py` creates text, image, and audio example inputs.
- `Converter/overrides.py` contains narrowly scoped export-time compatibility
  patches.
- `Converter/cache_utils.py` interprets and annotates exported cache objects.
- `Converter/constants.py` holds shared converter configuration.

Converter changes should preserve source-model semantics. Do not add a rewrite
here merely because it makes lowering easier; semantic graph simplification
belongs to Pompey.

### Model Profile responsibilities

`ModelProfiles/` is the declarative model-policy layer. Profiles describe:

- Supported modalities and sample-input strategies.
- Model loading and export patches.
- Prefill and decode inference modes.
- Cache style and cache precision contracts.
- Enabled and disabled fusion groups.
- Component definitions and execution routes.
- Prompt, media, persistent-state, and output-alias contracts.

Important files:

- `ModelProfiles/profiles.py` defines optimized profiles and `MODEL_ID_MAP`.
- `ModelProfiles/models.py` defines profile and contract dataclasses.
- `ModelProfiles/components.py` defines reusable component declarations.
- `ModelProfiles/routes.py` defines execution patterns and routes.
- `ModelProfiles/combinations.py` defines valid component combinations.

`MODEL_ID_MAP` is the boundary between optimized and generic conversion. An
exact, case-insensitive registered model ID uses its complete profile. Generic
CLI flags must not override that profile.

An unregistered model uses a generic contract built from `--task`,
`--modalities`, `--cache-style`, and optional `--fusion-groups`. Never infer an
optimized architecture from words in a repository name. To optimize a new
model, create or reuse a profile and explicitly register every supported model
ID.

When changing Crassus:

- Test every modality and inference mode declared by the profile.
- Compare exported raw IR before and after the change.
- Keep generic conversion functional for unregistered model IDs.
- Confirm export changes do not alter PyTorch reference outputs unexpectedly.

## Pompey: Fusions and IR

Pompey turns the verbose exported graph into a smaller semantic graph that maps
cleanly to Cactus operations.

### IR responsibilities

`IR/` owns graph structure, matching, normalization, simplification, dead-code
removal, and repeated fusion passes.

Important files:

- `IR/models.py` defines graph, node, edge, value, and annotation structures.
- `IR/simplify_ir.py` runs simplification and fusion passes to a stable result.
- `IR/match.py` performs structural fusion-graph matching.
- `IR/match_utils.py` provides reusable graph and tensor matching helpers.
- `IR/extra_matchers/` implements semantic constraints that cannot be expressed
  by graph shape alone.
- `IR/special_fusions.py` handles complex structural replacements requiring
  dedicated logic.

Simplification must run repeatedly because one cleanup or fusion can expose a
pattern consumed by an earlier pass. Run at least two complete rounds and
continue until the serialized graph is stable or the configured pass limit is
reached.

### Fusion responsibilities

`Fusions/` describes reusable source subgraphs and their replacement Cactus
operations.

Important files:

- `Fusions/fusions.py` is the fusion registry and assigns fusion groups.
- `Fusions/fusion_defs/` contains operation-family graph definitions.
- `Fusions/fusion_cache_conv.py` contains cache and convolution definitions.
- `Fusions/fusion_builders.py` provides helpers for constructing definitions.
- `Fusions/models.py`, `nodes.py`, and `edges.py` define reusable fusion data
  and graph fragments.

When adding or modifying a fusion:

- Match graph structure and tensor semantics, not generated node names alone.
- Validate ranks, dimensions, axes, constants, layouts, cache modes, inference
  modes, and source-module context.
- Preserve every output consumed outside the replaced subgraph.
- Keep generic fusion groups conservative; model-specific groups may be more
  aggressive when a profile proves their preconditions.
- Add a focused matcher regression test.
- Regenerate both prefill and decode graphs for every affected profile.
- Compare operation censuses before and after simplification. Unexpected `bmm`,
  `softmax`, cache copies, reshapes, or transposes often indicate a partial
  match.
- Verify that another complete simplification pass makes no unintended change.

Pompey decides **what computation a graph means**. It must not decide component
residency, execution order, or cross-component state lifetime; those belong to
Caesar.

## Caesar: Generator and Runtime Plan

Caesar converts the simplified semantic graph into native Cactus components and
defines how those components cooperate at runtime.

### Generator responsibilities

`Generator/` splits IR graphs into components, lowers nodes to Cactus graph
operations, binds converted weights, and writes component graph artifacts.

Important files:

- `Generator/generate.py` orchestrates component generation and bundle output.
- `Generator/component_splits/` identifies component boundaries and extracts
  component graphs.
- `Generator/lowerings/` maps simplified IR operations to Cactus graph-builder
  calls.
- `Generator/models.py` defines generation results, component graphs, manifests,
  and lowering rules.
- `Generator/constants.py` and `errors.py` hold shared generator configuration
  and lowering failures.

Lowering should be mechanical. If a lowering function needs to rediscover a
large decomposed pattern, add or improve a Pompey fusion instead. Unsupported
semantic operations should remain visible as unsupported rather than silently
lowering to an incorrect approximation.

### Runtime Plan responsibilities

`RuntimePlan/` converts generated component metadata and model-profile contracts
into `runtime_plan.json` and `components/manifest.json`.

`RuntimePlan/models.py` defines and serializes:

- Components, graph paths, logical inputs, and logical outputs.
- Weight bindings and cache-state node bindings.
- Execution routes and component roles.
- Runtime-owned persistent states.
- Producer-to-consumer tensor aliases and copy fallbacks.
- State lifetimes and last-consumer release metadata.
- Cache movement, persistence, and ownership policies.

When changing Caesar:

- Declare component routes and persistent outputs explicitly.
- Retain only outputs and cache entries consumed by a later component.
- Keep reusable cache and activation state in runtime-owned storage.
- Prefer compatible `TensorStorage` aliasing over inter-component copies.
- Always retain a correct copy fallback when storage cannot be aliased.
- Ensure unloading a component cannot invalidate state required by a later one.
- Confirm final component manifests reference valid graph nodes and weight files.
- Test bundle loading, prefill, decode, reset, and every supported modality.

Caesar decides **where computation lives and how it executes**. It should not
redefine source-model semantics or compensate for missing IR fusions.

## End-to-end contribution workflow

Follow the data through all three members before deciding where a change
belongs:

```text
PyTorch model
    -> Crassus: raw LayerMap + model contract
    -> Pompey: stable simplified/fused IR
    -> Caesar: component graphs + runtime plan
    -> Cactus engine execution
```

For every meaningful change:

1. Inspect raw and simplified IR.
2. Confirm the selected optimized or generic model profile.
3. Compare operation and component censuses.
4. Regenerate the complete model bundle from scratch.
5. Load it through `cactus run` and test every supported modality.
6. Verify output quality against the source model or a known-good bundle.
7. Measure prefill throughput, decode throughput, peak and active RAM, component
   load time, copied bytes, and aliased bytes.
8. Compare performance with `main` using equivalent weights and backend flags.

At minimum, compile all transpiler modules and run the converter, IR/fusion,
generator/component-split, runtime-plan, graph, and engine tests relevant to the
change. A successful transpilation is not sufficient: the generated bundle must
load and execute correctly.
