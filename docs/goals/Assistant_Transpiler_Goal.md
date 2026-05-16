# Goal Doc: Assistant Transpiler Integration

## Goal

Integrate Gemma4 assistant MTP support into Cactus transpile as a clean, manifest-driven graph-component feature.

The finished system should convert, package, load, and execute a Gemma4 target model plus its Gemma4 assistant without reintroducing native Gemma assistant subclasses. The assistant must use the real Hugging Face contract:

- target token embedding for the last seen token
- previous target hidden state
- target shared KV states
- constant assistant position
- assistant logits plus next hidden state output

The assistant must not be treated as a standalone causal LM that accepts only token ids.

## Objectives

- Define a versioned spec-decode manifest contract for single-position MTP assistant graphs.
- Add transpiler adapters that expose target and assistant tensors through explicit logical roles.
- Package target decoder, target embedding, and assistant graphs into one Cactus bundle.
- Update the C++ transpiled runtime to execute speculative decoding from manifest roles, not positional assumptions.
- Preserve normal transpiled causal LM completion when MTP is disabled.
- Keep conversion memory bounded by loading/transpiling target and assistant in separate phases, using BF16 by default, and avoiding duplicated weight materialization.

## Design Principles

- Use explicit manifest roles as the API between Python transpile and C++ runtime.
- Keep Gemma4-specific behavior inside Python model adapters and generated bundle metadata.
- Keep C++ completion generic: it should know about speculative roles, not Gemma4 layer names.
- Prefer composition over native model inheritance.
- Keep the first implementation correctness-first; optimize incremental target/cache behavior later.
- Do not silently fall back when required assistant tensors are missing.

## Hard Constraints

- Do not modify `cactus-kernels` or low-level kernel implementations for this goal unless a focused failing test proves the assistant integration cannot work without a kernel bug fix.
- Do not rewrite the transpiler, converter, runtime loader, or completion flow wholesale. Make small, targeted changes that extend existing abstractions.
- Do not reintroduce native Gemma assistant model subclasses or `gemma4_mtp_*` C++ model files.
- Do not add Gemma4-specific branches to generic C++ completion logic. Gemma4 details belong in Python adapters and manifest metadata.
- Do not treat the assistant as a normal causal LM or make it accept only token ids.
- Do not duplicate target embedding weights into the assistant bundle.
- Do not load the target HF model and assistant HF model in the same long-lived process during full conversion.
- Do not change defaults away from BF16 loading or assistant bits inheriting main model bits.
- Do not skip, delete, weaken, or rewrite tests to make failures pass.
- Do not change expected test behavior without first proving the existing expectation is wrong and documenting why in the commit or PR notes.
- Do not hide missing-role, missing-component, precision, or binding errors behind broad exception handling or fallback behavior.
- Do not explain runtime failures as quantization issues without a specific experiment that localizes the failure to quantization.
- Do not do style-only refactors, broad renames, or formatting churn while implementing this goal.
- Do not use destructive git commands or revert unrelated local changes.

## Test-First Task Order

Tests are the implementation driver for this goal. Each task starts with the smallest meaningful failing test, then the minimum implementation needed to pass it, then the relevant test command. A task is not complete until its tests pass without skipping or weakening coverage.

### 1. Spec-Decode Schema Tests

Add tests before implementation for the corrected assistant contract.

Required coverage:

- A valid `single_position_mtp` manifest loads and validates.
- Missing target roles fail clearly.
- Missing assistant roles fail clearly.
- Unsupported manifest versions fail clearly.
- Unsupported methods fail clearly.
- A token-only assistant manifest is rejected for Gemma4 MTP.
- Shared KV roles are named as `full_attention` and `sliding_attention`, not anonymous positional arrays.

Acceptance criteria:

- Python and C++ schema validators agree on required roles.
- Error messages name the missing role.

### 2. Transpiler Component IO Tests

Add tests for generated component specs before changing conversion behavior.

Required coverage:

- Gemma4 target spec-decode mode emits logical outputs for verifier logits, last hidden state, full-attention K/V, and sliding-attention K/V.
- Target embedding component exposes token id input and embedding output.
- Gemma4 assistant component exposes inputs for current token embedding, previous hidden, position, full-attention K/V, and sliding-attention K/V.
- Assistant component emits logits and next hidden state.
- Component IO is resolved by logical names, not output-node order.

Acceptance criteria:

- A toy or mocked Gemma4 component spec produces the expected manifest roles without loading the full model.

### 3. Memory and Packaging Tests

Add tests around conversion defaults and artifact binding.

Required coverage:

- `--torch-dtype` defaults to `bfloat16`.
- Assistant bits default to main model bits unless explicitly overridden.
- Assistant conversion/transpile can use a target tokenizer source when assistant tokenizer files are absent.
- Target embedding component references the target embedding artifact instead of writing a second copied embedding.
- Bound constants support `weight`, `embedding`, and `saved_constant` entries.

Acceptance criteria:

- Manifest bindings are enough for C++ to mmap or bind artifacts without duplicating large tensors in memory.

### 4. C++ Runtime Role Tests

Add C++ tests with small synthetic graphs before wiring the real assistant path.

Required coverage:

- The transpiled model loader resolves target decoder, target embedding, and assistant components from logical roles.
- The assistant draft path receives embedding, hidden, shared K/V, and position inputs.
- The runtime does not pass token context as the assistant’s only input.
- Missing assistant graph or missing role fails with an explicit MTP-unavailable reason.
- Non-MTP completion still loads and runs the decoder component unchanged.

Acceptance criteria:

- Synthetic graphs can prove data flows through the correct component inputs and outputs.

### 5. MTP Algorithm Tests

Add or extend tests for the generic speculative decode loop.

Required coverage:

- Assistant proposal is accepted when verifier agrees.
- Assistant proposal is rejected when verifier disagrees.
- Greedy draft selection remains deterministic.
- Accepted token count selects the correct next hidden state row.
- Draft loop carries assistant `next_hidden_state` forward inside a draft round.
- Between rounds, verified target hidden replaces assistant hidden.

Acceptance criteria:

- The generic MTP logic is independent of Gemma4-specific graph details.

### 6. Base Transpiled Runtime Fix Tests

Before end-to-end assistant validation, add a focused regression test for the current transpiled Gemma execution failure.

Required coverage:

- The base transpiled Gemma causal decoder executes past the first residual `ADD_CLIPPED`.
- Precision contracts are explicit: binary ops receive supported precisions or the graph lowering inserts the required cast.

Acceptance criteria:

- The main target bundle can run short non-MTP generation before assistant MTP is evaluated.

### 7. End-to-End Smoke Tests

Run only after the smaller tests pass.

Required coverage:

- Full target conversion with BF16 default.
- Full assistant conversion in a separate subprocess.
- Bundle manifest contains decoder, target embedding, assistant, and `single_position_mtp` roles.
- One-token non-MTP generation works.
- Short MTP generation works and reports draft/accept/reject metrics.

Acceptance criteria:

- Full conversion stays within the available 24 GB machine budget.
- Runtime failure modes are explicit and actionable.

## Implementation Plan

### 1. Replace the Assistant Contract

Update the spec-decode manifest from the current token-only assistant shape to a single-position MTP shape.

Target roles:

- `verifier_logits`
- `target_hidden_state`
- `target_token_embedding`
- `shared_kv.full_attention.key`
- `shared_kv.full_attention.value`
- `shared_kv.sliding_attention.key`
- `shared_kv.sliding_attention.value`

Assistant roles:

- `current_token_embedding`
- `previous_target_hidden`
- `position`
- `shared_kv.full_attention.key`
- `shared_kv.full_attention.value`
- `shared_kv.sliding_attention.key`
- `shared_kv.sliding_attention.value`
- `logits_output`
- `next_hidden_output`

Use a new manifest method name such as `single_position_mtp` or a new schema version so old token-only scaffolding cannot be mistaken for valid Gemma4 assistant support.

### 2. Add Gemma4 Target Spec-Decode Components

Extend the Gemma4 causal transpiler path so spec-decode mode emits:

- a decoder/verifier component
- a target embedding component
- logical outputs for hidden state and shared KV states

The target component should mirror Hugging Face’s requirement that the candidate generator has access to target hidden states and `shared_kv_states`.

### 3. Add Gemma4 Assistant Adapter

Add a dedicated assistant adapter instead of routing `Gemma4AssistantForCausalLM` through the normal causal LM adapter.

The adapter should:

- accept current target token embedding and previous target hidden as separate graph inputs
- concatenate them before the assistant `pre_projection`
- pass shared KV states as `full_attention` and `sliding_attention`
- pass a constant `position_ids` value for the current draft round
- emit logits and projected hidden state

This should match Hugging Face’s `SinglePositionMultiTokenCandidateGenerator` behavior.

### 4. Package Assistant Bundles

Update `cactus convert --assistant-model ...` packaging so the final bundle includes:

- target decoder component
- target embedding component
- assistant component
- tokenizer assets from the target model when needed
- shared artifact bindings instead of duplicated target embedding weights
- a validated spec-decode manifest

Target and assistant conversion/transpile should remain separate memory phases.

### 5. Update C++ Runtime Execution

Update the transpiled C++ model runtime to execute MTP from manifest roles.

The runtime should:

- load components by manifest role
- bind constants and embeddings robustly
- run target verification over the current context plus candidate tokens
- run target embedding for the last seen token
- run assistant drafts with embedding, hidden, position, and shared KV inputs
- use existing generic MTP accept/reject helpers
- fail loudly when any required role or component is unavailable

Do not add Gemma4-specific C++ model subclasses.

### 6. Fix Base Decoder Runtime Issues

Resolve the current base transpiled Gemma runtime blockers before final MTP validation.

Required fixes:

- make bound constant loading robust for all manifest artifact kinds
- make manifest component parsing order-independent
- fix the precision mismatch that causes `ADD_CLIPPED` to receive unsupported precision

### 7. Validate Full Conversion

Run the full conversion only after schema, component, packaging, C++, MTP, and base runtime tests pass.

Expected command shape:

```bash
source ./venv/bin/activate
cactus build
cactus convert google/gemma-4-E2B-it /private/tmp/cactus_gemma4_e2b_it_mtp_full_bf16 \
  --bits 4 \
  --device cpu \
  --task causal_lm_logits \
  --torch-dtype bfloat16 \
  --assistant-model google/gemma-4-E2B-it-assistant \
  --local-files-only
```

Use explicit memory logging during this run.

## Acceptance Criteria

- `cactus convert` can produce a target-plus-assistant transpiled bundle for Gemma4 assistant MTP.
- The generated manifest has explicit target, target embedding, assistant, and shared KV roles.
- C++ runtime can load the bundle and run normal non-MTP completion.
- C++ runtime can run short MTP completion using assistant graph drafts and target verification.
- Assistant graph inputs match HF behavior: current token embedding plus previous hidden, not token ids alone.
- Target embedding weights are not duplicated into the assistant bundle.
- Conversion uses BF16 by default and does not load target and assistant HF models simultaneously.
- All missing-role, missing-component, and unsupported-method failures are explicit.
- No native Gemma assistant model subclass is reintroduced into C++ engine code.

## Out of Scope

- Tree speculation.
- Dynamic draft scheduling beyond existing fixed draft-token options.
- Fully incremental transpiled target KV-cache execution.
- Generic support for non-Gemma assistant architectures.
- Native Gemma model subclass restoration.

## Required Command Discipline

Before running any Cactus command:

```bash
source ./venv/bin/activate
cactus build
```

Rebuild after C++/FFI/model changes. If running multiple Cactus commands in the same activated terminal session without code changes, one build is enough.
