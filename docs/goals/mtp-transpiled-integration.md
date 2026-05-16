# Goal Doc: Transpiled MTP Speculative Decoding Integration

## Objective

Create and complete a new branch named `mtp-scaffolding-transpiled`, based on `transpiler-complete`, that ports and adapts the generic MTP/speculative decoding architecture from `mtp-scaffolding-split` so MTP works with transpiled Cactus models.

Do not edit or continue implementation work directly on `mtp-scaffolding-split`.

The finished branch should support:

- Normal C++ engine completion for transpiled causal LM bundles.
- Assistant-only MTP/speculative decoding through transpiled graph components.
- Generic MTP primitives under `cactus-engine/src`.
- No Gemma-specific speculative decode files under `cactus-engine/src`.
- Future extension to non-Gemma assistants, exact speculative decoding, dynamic draft policy, and incremental target execution without redesigning completion flow.

## Background

The `mtp-scaffolding-split` branch contains a generic MTP refactor with reusable decode primitives under `cactus-engine/src`, including sampling helpers, speculative accept/reject logic, and completion-level MTP integration. That branch intentionally removed earlier `gemma4_mtp_*` files from `src` and moved Gemma-specific native assistant behavior out of generic code.

The `transpiler-complete` branch is structurally different. Native model subclasses have been removed from the C++ engine build, and `create_model()` currently does not instantiate a usable C++ model for transpiled bundles. Python transpiler infrastructure exists under `python/cactus/transpile`, and saved component bundles are described by `components/manifest.json`. Python can execute saved component bundles, but C++ engine integration needs to be added.

Do not wholesale cherry-pick `mtp-scaffolding-split`. A full cherry-pick or merge would reintroduce native model files that `transpiler-complete` removed. Port only the generic MTP concepts and adapt them to the transpiled runtime.

The first implementation should be correctness-first. Use static full-context verification/state extraction for the target model instead of implementing incremental transpiled KV cache immediately.

## Branch Safety

Required branch setup:

```bash
git switch transpiler-complete
git switch -c mtp-scaffolding-transpiled
```

Rules:

- Do not modify `mtp-scaffolding-split`.
- Do not merge or cherry-pick the whole MTP branch.
- Selectively port only generic MTP concepts and files.
- Do not re-add deleted native model subclasses from `mtp-scaffolding-split`.
- Do not use destructive git commands.
- Preserve unrelated user changes if present.

## Engineering Direction

Use these design patterns explicitly:

- Capability pattern: target models expose speculative decode support only when the transpiled bundle provides the required roles.
- Adapter pattern: Python model-specific transpiler adapters map HF/model-specific tensors to generic manifest roles.
- Strategy pattern: speculative decode method and draft policy are pluggable.
- Factory/registry pattern: draft runtimes and spec methods are constructed from manifest metadata.
- Composition over inheritance: C++ transpiled runtimes compose graph/component loaders instead of creating model-specific subclasses.
- Manifest/schema contract: tensor meaning must come from explicit logical roles, not positional order.

Avoid:

- Gemma-specific names or branches in generic C++ completion code.
- New `gemma4_mtp_*` files under `cactus-engine/src`.
- Recreating native Gemma implementation paths.
- Large style-only rewrites.
- Catching unexpected errors silently.
- Treating numeric issues as "quantization error" without proof.

## Implementation Plan

### 1. Establish C++ transpiled main model support

Implement baseline C++ loading and execution for transpiled causal LM bundles.

Expected behavior:

- `create_model(model_folder)` detects a transpiled causal LM bundle.
- The C++ engine can run normal non-MTP completion through the existing completion/chat path.
- Existing Python `cactus run-transpiled` behavior remains unchanged.

This must be completed before MTP wiring.

### 2. Add generic spec decode manifest contract

Extend transpiler bundle metadata with a versioned `spec_decode` section.

Minimum required fields:

- `version`
- `method`, initially `assistant_chain`
- target component role mappings:
  - verifier logits
  - target hidden state
  - assistant-shared state tensors
- assistant component role mappings:
  - current token or token embedding input
  - previous target hidden input
  - target shared state inputs
  - position input
  - assistant logits output
  - next assistant hidden output

The exact JSON shape may follow the existing manifest style, but the contract must be explicit, versioned, and validated.

### 3. Add transpiler adapters

Add Python transpiler adapters for speculative decoding.

Target adapter:

- Runs the transpiled main model over the full prompt/candidate context.
- Emits verifier logits for candidate verification.
- Emits target hidden state for assistant input.
- Emits shared state tensors needed by the assistant.

Assistant adapter:

- Consumes the generic role inputs from the manifest.
- Produces assistant logits and next assistant hidden.
- May contain Gemma-specific HF wiring internally, but exports only generic role names.

Keep model-specific implementation details in Python adapter/transpiler code, not generic C++ completion code.

### 4. Port generic MTP primitives

Selectively port the generic ideas from `mtp-scaffolding-split`:

- sampling helpers
- speculative accept/reject logic
- completion-level metrics and options
- completion integration points

Keep these under `cactus-engine/src`.

Do not port:

- native Gemma model files
- native Gemma assistant implementation
- any `gemma4_mtp_*` files under `src`
- deleted native-model build structure from the old branch

### 5. Add C++ runtime interfaces

Add generic runtime interfaces for speculative decoding:

- target speculative capability
- draft assistant runtime
- draft input/output structs
- verifier input/output structs
- opaque target state representation

Target state must be opaque and role-keyed. It must not expose Gemma-specific structs, layer names, or native KV-cache node assumptions.

### 6. Wire completion flow

Update completion logic so MTP depends only on generic capabilities:

- If the model lacks spec decode capability, reject or ignore MTP settings with an explicit error/log.
- If manifest roles are incomplete, fail loudly with a clear message.
- If assistant runtime is unavailable, fail loudly.
- Non-MTP completion must remain unaffected.

Initial draft policy:

- Static/fixed draft length.
- Keep dynamic draft policy as a future strategy object, but do not implement complex policy logic now.

### 7. Keep future paths open

The implementation must make these future changes straightforward:

- Non-Gemma assistant MTP by adding a new Python adapter and manifest role mapping.
- Exact speculative decoding by adding another method implementation.
- Dynamic draft policy by replacing the fixed draft strategy.
- Incremental/stateful transpiled target cache by adding another target capability implementation.
- Tree speculation remains out of scope, but the method registry should not block it architecturally.

## Testing Requirements

Follow test-driven development:

1. Add or update tests for new interfaces before implementing runtime behavior.
2. Implement one subsystem at a time.
3. Run the relevant test subset after each subsystem.
4. Run full validation at the end.

Required test categories:

- Manifest validation tests:
  - valid spec decode manifest loads
  - missing required roles fails clearly
  - unsupported method fails clearly
  - unsupported manifest version fails clearly

- Generic MTP logic tests:
  - assistant proposal accepted when verifier agrees
  - assistant proposal rejected when verifier disagrees
  - sampler behavior remains deterministic under greedy settings
  - non-MTP path is unchanged

- C++ transpiled model tests:
  - transpiled causal LM bundle loads through `create_model`
  - normal non-MTP completion works before MTP is enabled
  - incomplete transpiled bundle fails with explicit error

- Transpiler adapter tests:
  - target adapter emits expected role names and shapes
  - assistant adapter emits expected role names and shapes
  - role mapping is manifest-driven, not positional-order driven

- End-to-end tests:
  - non-MTP transpiled chat CLI generation
  - MTP transpiled chat CLI generation
  - real speculative decode trajectory with accept/reject metrics visible

## Required Command Discipline

Before running any Cactus command:

```bash
source ./venv/bin/activate
cactus build
```

Rebuild after C++/FFI/model changes. If running several commands in the same activated terminal after one build, rebuilding is not needed unless C++/FFI/model code changed again.

Suggested final validation sequence:

```bash
source ./venv/bin/activate
cactus build
```

Then run:

- targeted manifest/unit tests
- targeted MTP tests
- transpiled non-MTP chat CLI generation
- transpiled MTP chat CLI generation
- full relevant test suite
- `git diff --check`

## Strict Acceptance Criteria

The work is complete only when all of the following are true:

- Current branch is `mtp-scaffolding-transpiled`.
- Branch base is `transpiler-complete`, not `mtp-scaffolding-split`.
- `mtp-scaffolding-split` remains unmodified.
- No native model subclasses removed by `transpiler-complete` are reintroduced.
- No `gemma4_mtp_*` files exist under `cactus-engine/src`.
- Generic MTP code lives under `cactus-engine/src`.
- Gemma-specific speculative decode behavior, if any, lives only in model-specific transpiler adapter code or model-specific metadata generation.
- C++ non-MTP completion works for transpiled causal LM bundles.
- C++ MTP completion works through transpiled target and assistant components.
- Completion code does not cast to Gemma-specific classes or check Gemma-specific model names.
- Manifest validation is explicit and fails loudly for missing roles.
- Static full-context verifier/state extraction is used for v1 target execution.
- Interfaces leave room for exact spec decode, non-Gemma assistants, dynamic draft policy, and future stateful target execution.
- All required tests pass.
- Real chat CLI trajectories have been run for both non-MTP and MTP transpiled paths.
- Final diff is focused and reviewable.
- `git diff --check` passes.

## Final Deliverables

- New branch: `mtp-scaffolding-transpiled`.
- Focused implementation commits.
- Updated tests covering manifest contracts, generic MTP logic, transpiled runtime loading, and end-to-end CLI behavior.
- A short final implementation note summarizing:
  - what was ported
  - what was intentionally not ported
  - how to run validation
  - known limitations of the v1 static verifier approach
