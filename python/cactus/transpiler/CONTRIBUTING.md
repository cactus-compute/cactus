# Transpiler development

The transpiler turns an exported PyTorch graph into a Cactus runtime bundle in four stages:

1. `Converter/` exports a model and serializes the graph as a `LayerMap`, this can be visualized through JSON and serves as the intermediary representation of the model.
2. `IR/` normalizes the exported graph and applies safe graph fusions for cactus-native operations.
3. `Generator/` splits the graph into ModelProfile specified components, and lowers each fused operation to Cactus calls.
4. `RuntimePlan/` and `ModelProfiles/` describe component routes, persistent state, cache bindings, and metadata files.

Keep model-specific behavior in `ModelProfiles/` or a narrowly scoped matcher. Shared lowering and graph code must remain independent of Hugging Face model names!!!

## Updating Converter Logic (Crassus)

`ModelProfiles/profiles.py::MODEL_ID_MAP` is the boundary between optimized and
generic conversion. An exact, case-insensitive registered model ID always uses
its complete profile: input strategy, modalities, export patches, cache
contract, fusion fields, component split, and runtime plan. Generic CLI options
must not override a registered profile.

An unregistered model returns no profile from `profile_for_model_id`. The CLI
then constructs a generic contract from `--task`, `--modalities`,
`--cache-style`, and optional `--fusion-groups`. Never select a generic
architecture from substrings in a repository name. To add an optimized model,
create or reuse a profile and explicitly add every supported model ID to
`MODEL_ID_MAP`.

## Adding or changing a fusion

- Match structure and tensor semantics, not generated node names alone.
- Validate ranks, axes, constants, cache mode, and source-module context before replacing a subgraph.
- Preserve every externally consumed output of the matched region.
- Add a focused matcher regression test and regenerate both prefill and decode graphs for every affected profile.
- Run at least two complete simplify/fusion rounds over raw IR, then continue until the serialized graph is stable. Later fusion families and cleanup can expose patterns owned by an earlier family.
- Compare the simplified operation census before and after the change. An unexpected remaining `bmm`, `softmax`, cache copy, or transpose usually means a pattern only partially matched.

## Changing a model profile or runtime plan

- Declare component routes and persistent outputs explicitly.
- Only retain outputs or cache entries that a later component consumes.
- Keep reusable state in runtime-owned storage; component graphs should bind to it rather than own or copy it.
- Regenerate the complete bundle and test every modality combination supported by the profile.

## Validation

Run the transpiler and runtime-plan tests, compile all transpiler modules, regenerate all available model bundles, and execute the graph/runtime tests. Performance changes should be benchmarked to determine prefill throughput, decode thorughput, peak RAM usage, and model output quality.
