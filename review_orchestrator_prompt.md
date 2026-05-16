# Orchestrator Prompt: Comprehensive Review of MTP Transpiler Diffs

You are the orchestrator for a comprehensive code review of the current branch against the base branch `origin/transpiler-complete`.

## Objective

Review every diff introduced by the current branch on top of `origin/transpiler-complete`. The goal is to determine whether the implementation is clean, logically correct, compatible with existing transpilation and compilation pipelines, and ready for future engine/transpiler evolution.

Treat this as a serious pre-merge review. Prioritize correctness, maintainability, and regression risk over style preference.

## Review Scope

Use this diff as the source of truth:

```bash
git fetch origin
git diff --stat origin/transpiler-complete...HEAD
git diff origin/transpiler-complete...HEAD
```

Also inspect the commit stack:

```bash
git log --oneline --decorate --graph origin/transpiler-complete..HEAD
```

Do not review unrelated local files outside this diff unless they are needed to understand an integration boundary.

## Required Standards

Evaluate the implementation against these standards:

1. Clean code
   - Changes should be minimal, targeted, and readable.
   - Comments should be minimal and only explain genuinely non-obvious logic.
   - Existing naming, structure, and patterns in `cactus`, `cactus-engine`, `cactus-graph`, and `python/cactus` should be preserved.
   - Avoid duplicate logic where an existing helper or pattern should be reused.

2. Logical correctness
   - Every new code path should make sense end to end.
   - Manifest fields, graph node bindings, assistant/target model wiring, saved constants, MTP behavior, and dtype/precision handling should be internally consistent.
   - Error messages should be explicit when a model, manifest, graph, or binding is unsupported.
   - Do not accept vague explanations such as “quantization error” without evidence.

3. Compatibility with existing pipelines
   - Confirm the changes do not break existing conversion, transpilation, compilation, runtime loading, or generation flows for other supported model families.
   - Pay special attention to Gemma4, Qwen, Whisper, Parakeet, LFM, VLM/audio paths, graph serialization, mmap weight binding, saved constants, and Python CLI defaults.
   - Verify that new assumptions about manifests, weight formats, tensor precision, graph inputs/outputs, and runtime binding are either backward-compatible or fail clearly.

4. Engineering patterns and future readiness
   - The design should be easy to extend for new engines, new assistant models, new saved-constant formats, and future transpiler/runtime component boundaries.
   - Shared behavior should be centralized in the appropriate layer.
   - CLI behavior should remain coherent and predictable.
   - Runtime/model code should not become tightly coupled to one specific converted bundle unless that coupling is explicit and well-contained.

## Delegation Plan

Launch multiple specialist agents. Each agent should inspect the relevant diffs, run targeted commands where useful, and return findings with file/line references. Findings should be ordered by severity and should distinguish confirmed bugs from risks or open questions.

### Agent 1: Runtime and Engine Integration

Review C++ runtime/model loading changes in:

- `cactus-engine/src/model.cpp`
- `cactus-engine/tests/*`
- Any touched `cactus-graph` or FFI files

Focus on:

- Manifest parsing and binding correctness
- MTP runtime behavior
- Saved constant handling, including `npy` and `tensor_io`
- mmap weight binding shape/precision safety
- Failure modes and error messages
- Whether this design can support future engines/components

Run or inspect:

```bash
source ./venv/bin/activate
cactus build
cmake --build cactus-engine/tests/build --target test_model_loading test_spec_decode_manifest test_mtp_decode
cactus-engine/tests/build/test_model_loading
cactus-engine/tests/build/test_spec_decode_manifest
cactus-engine/tests/build/test_mtp_decode
```

### Agent 2: Python Converter and CLI

Review Python conversion/transpilation changes in:

- `python/cactus/cli/*`
- `python/cactus/convert/*`
- `python/cactus/transpile/*`
- `python/tests/*`

Focus on:

- CLI defaults and option propagation
- Component manifest generation
- Assistant bundle integration
- Weight format and precision policy correctness
- BF16/FP16 dtype handling
- Local-files-only behavior
- Whether conversion logic remains centralized and avoids one-off paths

Run or inspect:

```bash
source ./venv/bin/activate
cactus build
python -c "from pathlib import Path; from cactus.cli.transpile import _link_python_runtime_library; _link_python_runtime_library(static_library_path=Path('cactus/build/libcactus.a').resolve(), library_path=Path('cactus/build/libcactus.dylib').resolve())"
PYTHONPATH=python pytest python/tests python/cactus/convert/tests --ignore=python/tests/test_model.py
```

### Agent 3: Cross-Model Regression Risk

Review the diff specifically for unintended impact on existing model families and tasks.

Focus on:

- Qwen, Gemma4, Whisper, Parakeet, LFM, VLM, audio, embedding, and transcription flows
- Existing model detection, naming, weight policy, and component routing
- Any assumptions that only hold for the new MTP bundle
- Backward compatibility for existing manifests and weights
- Whether new defaults alter behavior for non-MTP users

Use tests and static inspection. Propose additional tests if coverage is insufficient.

### Agent 4: Architecture and Maintainability

Review the full diff at the design level.

Focus on:

- Whether responsibilities are in the right modules
- Whether abstractions are too specific or too broad
- Whether the implementation is ready for future assistant models, future graph formats, and future engine backends
- Whether naming, manifest schema usage, and control flow are understandable
- Whether comments are minimal and useful
- Whether any code should be extracted, centralized, or simplified before merge

Do not suggest broad refactors unless they reduce real risk or remove meaningful duplication.

### Agent 5: Test Coverage and Verification

Review all added/modified tests and compare them to the behavioral risk.

Focus on:

- Whether tests prove the intended MTP and saved-constant behavior
- Whether failure modes are covered
- Whether cross-model regression tests are missing
- Whether runtime smoke tests are sufficient
- Whether any tests are brittle, too coupled to local artifacts, or silently skip important behavior

Also summarize which tests were run and which important tests were not run due to external model/download requirements.

## Required Output From Each Agent

Each agent must return:

1. Findings first, ordered by severity.
2. File and line references for each finding.
3. Clear distinction between:
   - confirmed bug
   - likely regression risk
   - maintainability concern
   - missing test coverage
   - open question
4. A short list of tests or commands run.
5. A short statement if no issues were found in that area.

Do not bury findings behind summaries. If there are no findings, say so explicitly and state the residual risk.

## Orchestrator Final Report

After all agents report back, produce a consolidated review with:

1. Blockers
2. High-priority non-blockers
3. Medium/low-priority issues
4. Open questions
5. Suggested follow-up tests
6. Overall merge recommendation

The final recommendation must be one of:

- `merge as-is`
- `merge after small fixes`
- `needs another review after fixes`
- `do not merge yet`

Ground the recommendation in concrete findings, not general impressions.

## Important Context

The branch has been rebased onto `origin/transpiler-complete` and includes work for manifest-driven Gemma4 assistant MTP, updated weight formats, and runtime support for transpiled saved constants.

Known current validation from the implementer:

- Fresh conversion completed for Gemma4 target plus assistant.
- Base generation passed on the freshly converted bundle.
- 2-token MTP generation passed on the freshly converted bundle.
- 32-token MTP generation passed on the freshly converted bundle.
- Python unit/converter suite passed excluding `python/tests/test_model.py`, which depends on external model integration/download behavior.
- Engine, graph, and kernels shell suites passed, with native-model engine binaries skipped by the repo script where native subclasses are unavailable.

Do not rely only on this validation. Verify the parts relevant to your assigned review area.
