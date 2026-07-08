# Cactus Transpiler

This folder contains the Python side of the Cactus transpiler: the code that takes
a Hugging Face or PyTorch model, captures it with `torch.export`, converts that
export into Cactus' intermediate representation (`IRGraph`), canonicalizes and
optimizes the IR, lowers it into a runtime `Graph`, and optionally saves or runs
the result.

The transpiler has two main output shapes:

- A single lowered graph for simple tasks such as text logits or encoder-only audio.
- A component bundle for models that are easier to split into staged graphs, most
  notably Gemma4 multimodal and Parakeet TDT.

This README is a code-first walkthrough of the actual flow in the repo today.

## Related Entry Points

These files are outside `python/cactus/transpile`, but they are the top-level
entry points for the transpiler (driven by `cactus convert` for end users; the
`cactus transpile` command is hidden from the CLI menu and used internally):

- `python/cactus/cli/convert.py` (`cmd_convert` → `cmd_transpile`)
- `python/cactus/cli/transpile.py` (`run_transpile`)
- `python/cactus/transpile/hf_model.py`

There is no separate `cactus transpile` subcommand. `cactus convert <model>`
quantizes the weights to CQ and then builds the runtime graph in one step; the graph
build is `run_transpile`, a thin wrapper around `python/cactus/transpile/hf_model.py`.
The packaged module is the real top-level transpile program.

## CLI Entry Flow

### `cactus convert`

`cmd_convert` (`python/cactus/cli/convert.py`) quantizes the weights via `ensure_weights`,
then (unless `--weights-only`) calls `cmd_transpile`, which forwards options to
`run_transpile`:

```python
def run_transpile(model_id, *, extra_args, execute_after_transpile=False, ...):
    transpile_lib = _ensure_python_runtime_library()
    model_id = resolve_model_id_alias(model_id)
    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if "--weights-dir" not in extra_args:
        default_weights_dir = get_weights_dir(model_id)
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
        else:
            return 1
    if not execute_after_transpile and "--skip-execute" not in extra_args:
        command.append("--skip-execute")
    command.extend(extra_args)
```

Two important details:

- The graph build saves artifacts and skips execution by default
  (pass `--execute-after-transpile` to run a reference execution).
- Building the runtime graph requires converted Cactus CQ weights. `cactus convert`
  produces them before this stage; or pass `--weights-dir` to an existing converted folder.

### `cactus run`

`cactus run <model-id-or-bundle-path>` is the single user-facing entry. Resolution
order (`python/cactus/cli/model.py:ensure_runnable_bundle`):

1. `<arg>` is an existing path with `components/manifest.json` → run as-is.
2. Cached download at `weights/<model>-cq<bits>[-<platform>]/` → run.
3. Fresh download from `huggingface.co/Cactus-Compute` via `download_bundle`.
4. Fallback: local `ensure_bundle` (convert + transpile for custom models not on HF).

Bundle handoff is to the native `run` binary (`python/cactus/bin/run`), which
loads the bundle via `cactus_init()`.

## End-To-End Timeline

## 1. Parse Args, Infer Task, Load Model Bundle

The real transpile program lives in `python/cactus/transpile/hf_model.py`.
Its `main()` function decides what kind of model/task it is dealing with, validates
optional converted weights, and loads the HF model plus tokenizer/processor:

```python
if args.task == "auto":
    inferred_task = _infer_task_from_config(args.model_id)
    ...
    if image_files or (args.audio_file and has_multimodal_config and inferred_task == "causal_lm_logits"):
        task = "multimodal_causal_lm_logits"
    else:
        task = inferred_task
else:
    task = args.task

validated_weights_dir = _validate_weights_dir(args.weights_dir.strip() or None, model_id=args.model_id)
weights_dir = str(validated_weights_dir) if validated_weights_dir is not None else None

model_source, processor_or_tokenizer, model, model_config = _load_transformers_bundle(
    model_id=args.model_id,
    task=task,
    torch_dtype=torch_dtype,
    token=args.token,
    trust_remote_code=args.trust_remote_code,
    local_files_only=args.local_files_only,
)
```

### What decides the task?

`_infer_task_from_config()` looks at:

- `config.json` architectures
- model type
- known special cases like Parakeet TDT and Whisper
- model id substrings as a last fallback

Today the main task shapes are:

- `causal_lm_logits`
- `multimodal_causal_lm_logits`
- `ctc_logits`
- `encoder_hidden_states`
- `tdt_transcription`

## 2. Prepare Example Inputs

Before capture, the transpiler builds one concrete example input batch. That
example batch drives both `torch.export` capture and later optional execution.

The high-level branches are:

- `_prepare_text_inputs()` for text-only causal LM capture
- `_prepare_audio_inputs()` for speech/audio tasks
- `_prepare_gemma4_multimodal_inputs()` for Gemma4 multimodal

Example:

```python
if task == "causal_lm_logits":
    prepared = _prepare_text_inputs(...)
elif task == "multimodal_causal_lm_logits":
    prepared = _prepare_gemma4_multimodal_inputs(...)
elif task == "tdt_transcription":
    prepared = _prepare_audio_inputs(...)
else:
    prepared = _prepare_audio_inputs(...)
```

The `PreparedInputs` object carries:

- `names`: logical input names like `input_ids` or `input_features`
- `tensors`: the actual example tensors
- `metadata`: prompt/audio/image/sample-rate info that later gets written into
  manifests

## 3. Canonicalize The Model Interface

The transpiler does not try to capture every model's native HF forward signature
as-is. It first wraps the model in an adapter that exposes a smaller, more stable
task-specific interface.

That happens in `model_adapters.py`:

```python
def canonicalize_model_interface(model, task="causal_lm_logits", *, input_names=None, weights_dir=None):
    family = _family_key(model)
    ...
    if task == "causal_lm_logits":
        ...
    elif task == "multimodal_causal_lm_logits":
        adapter_factory = lambda inner_model: Gemma4MultimodalCausalLMLogitsAdapter(
            inner_model,
            input_names=resolved_input_names,
            weights_dir=weights_dir,
        )
    elif task == "ctc_logits":
        adapter_factory = lambda inner_model: CTCLogitsAdapter(...)
    elif task == "encoder_hidden_states":
        adapter_factory = lambda inner_model: EncoderHiddenStatesAdapter(...)
```

This step is where family-specific knowledge lives:

- Gemma/Gemma3/Gemma4/Qwen causal LM wrappers
- Gemma4 multimodal wrappers
- generic CTC and encoder wrappers

The adapter is also where graph metadata and import hints get attached.

## 3b. Import PyTorch Ops, Not Model Layer Names

The compiler boundary is the `torch.export` graph. After capture, Cactus imports
`call_function` nodes by their canonical ATen operation name, not by Hugging Face
module names. For example, `torch.ops.aten.linear.default`, `aten.linear.default`,
and equivalent exported spellings normalize through `aten_ops.py` to the same
canonical IR op:

```python
aten.linear.default -> linear
aten.silu.default   -> silu
aten.conv1d.default -> conv1d
aten.rms_norm.default -> rms_norm
```

This is the main generality boundary: model wrappers may still prepare inputs or
split components, but actual compute lowering is ATen op / pattern based. Layer
names are allowed as structural hints for component boundaries and exact
converted-weight manifest aliases; they should not be required to decide how a
PyTorch op maps to a Cactus op.

## 4. Decide Between Monolithic Capture And Component Capture

After input prep, the script asks `build_component_module_specs()` whether the
current model supports a split component pipeline:

```python
component_specs = build_component_module_specs(
    model,
    task=task,
    named_tensors=_named_tensor_store(prepared),
    weights_dir=weights_dir,
)
```

Today that returns component specs for:

- Gemma4 multimodal: `vision_encoder`, `audio_encoder`, `lm_encoder`, `decoder`
- Parakeet TDT: `audio_encoder`, `decoder`

If component specs exist and `--component-pipeline` is `auto` or `on`, the script
takes the component path. Otherwise it captures a single wrapped model graph.

## 5A. Component Pipeline Path

The component path is driven by `_run_component_pipeline_transpile()` in
`python/cactus/transpile/hf_model.py`:

```python
for spec in component_specs:
    captured_components[spec.component] = capture_component_spec(spec, fusion_config=fusion_config)
```

Each component spec is captured by `component_pipeline.capture_component_spec()`:

```python
captured = capture_model(wrapped, spec.example_inputs)
raw_ir_graph = copy.deepcopy(captured.ir_graph)
optimized_ir_graph = copy.deepcopy(captured.ir_graph)
canonicalize_exported_graph(optimized_ir_graph)
optimize_graph(optimized_ir_graph, config=fusion_config)
transpiled_graph = transpile_preoptimized_ir(copy.deepcopy(optimized_ir_graph))
```

So each component goes through the same capture/import/canonicalize/optimize/lower
pipeline independently.

For Gemma4 multimodal, `model_adapters.py` builds the specs by first computing
example image/audio features, then generating example inputs for the downstream
`lm_encoder` and `decoder` stages.

## 5B. Monolithic Capture Path

The non-component path wraps the canonical adapter in `TranspileWrapper` and runs
the full graph through capture:

```python
wrapper = TranspileWrapper(canonical.module, weights_dir=weights_dir).eval()

captured = capture_model(wrapper, prepared.tensors)
raw_ir_graph = copy.deepcopy(captured.ir_graph)

canonicalize_exported_graph(captured.ir_graph)
optimize_graph(captured.ir_graph, config=fusion_config)
tg = _lower_preoptimized_ir(captured.ir_graph)
```

`TranspileWrapper` mainly exists to pass `weights_dir` into the graph metadata:

```python
def get_transpile_metadata(self) -> dict[str, object]:
    ...
    if self.weights_dir:
        graph_meta["weights_dir"] = self.weights_dir
```

## 6. Capture With `torch.export`

`capture_pytorch.py` owns the `torch.export` boundary:

```python
normalized_kwargs = _inject_export_safe_kwargs(model, _normalize_kwargs(kwargs))
...
transpile_metadata = _collect_transpile_metadata(model, example_args, example_kwargs)

ep = export(model, args=example_args, kwargs=example_kwargs, strict=strict)
...
ir_graph = import_captured_to_ir(raw_captured, strict=strict)
```

Important behavior here:

- HF-style `use_cache=False` and `return_dict=False` are injected automatically
  when the model signature supports them.
- Metadata providers are collected from any module that defines
  `get_transpile_metadata()` or `transpile_metadata()`.
- The captured export is immediately imported into the repo's own IR.

The capture output is a `CapturedModel` dataclass holding:

- the exported program
- the FX graph
- the lifted state dict
- the imported `IRGraph`
- the example inputs
- the collected metadata

## 7. Import Exported FX Into `IRGraph`

`import_ir.py` converts the exported FX/Export graph into the repo's intermediate
representation:

```python
ir = IRGraph(values={}, nodes={}, order=[], inputs=[], outputs=[], constants={}, meta=dict(graph_meta))
ctx = ImportContext(strict=strict, transpile_metadata=transpile_metadata)
weights_dir = resolve_transpile_weights_dir(ir.meta)

for node in captured.graph.nodes:
    if node.op == "placeholder":
        ...
    if node.op == "get_attr":
        ...
    if node.op == "call_function":
        import_call_function(...)
    if node.op == "output":
        import_output(ir, node, ctx)

apply_import_semantics(ir)
annotate_ir_components(ir)
verify_ir(ir)
```

### What gets imported?

- Real graph inputs become `IRGraph.inputs`
- lifted parameters/buffers/constants become `IRGraph.constants`
- FX `call_function` nodes become `IRNode`s via `importers.py`

### How weight binding metadata enters the IR

When a captured constant corresponds to a known converted weight file,
`import_captured_to_ir()` resolves that mapping early:

```python
binding = resolve_weight_binding(weights_dir=weights_dir, source_name=target)
...
ir.values[value_id_str].meta.update(
    {
        "path": binding.path,
        "kind": binding.kind,
        "source_name": binding.source_name,
    }
)
```

That metadata is what later lets lowering use `g.mmap_weights()` instead of
embedding the constant payload into the graph. Embedding lookups still lower to
`embedding_from_tensor()`, but the backing table is represented as a normal mmap
weight tensor so tied output heads can reuse the same constant safely.
By default this resolver only trusts explicit `weights_manifest.json` entries
written by `cactus convert` plus wrapper-local aliases derived from those entries.
It does not guess old model-specific filenames from layer names.

## 8. Importers: FX Target To Canonical IR Node

`importers.py` is the large target-dispatch layer. It normalizes FX/PyTorch targets
to canonical op names and creates `IRNode`s with stable attributes.

The dispatch point is:

```python
def import_call_function(ir, node, ctx, *, shape, dtype, torch_op):
    op = normalize_target(torch_op)
    importer = OP_IMPORTERS.get(op)
    if importer is None:
        import_opaque_call_function(...)
    else:
        importer(...)
```

Import is intentionally ATen/op based. It does not apply module-path hints such
as `model.layers.N.self_attn`; model-specific behavior should live in explicit
preprocessing/component boundaries or in topology-based fusion passes.

Some notable importer behaviors:

- `normalize.py` reduces many `aten.*` spellings into one canonical name.
- Unknown ops are imported as `kind="opaque"` IR nodes rather than being silently
  dropped.
- Import hints can inject attrs/meta into matching nodes after import.
- Some nested exported subgraphs such as `wrap_with_set_grad_enabled` are inlined
  explicitly in `import_ir.py`.

## 9. Import-Time Semantic Rewrites

Immediately after raw import, `import_semantics.py` performs a first semantic pass:

```python
def apply_import_semantics(graph: IRGraph) -> IRGraph:
    _rewrite_explicit_attention(graph)
    _tag_explicit_linear(graph)
    _rewrite_early_rope(graph)
    rebuild_graph(graph)
    verify_ir(graph)
    return graph
```

This step is intentionally early. It rewrites export patterns into higher-level
meaning before the main optimizer sees them.

Examples:

- `scaled_dot_product_attention` becomes semantic `attention`
- explicit linears get tagged
- RoPE patterns can become semantic `rope` before later fusion passes

## 10. Canonical Cleanup

`canonicalize/cleanup.py` turns imported IR into a stricter canonical form:

```python
def canonicalize_exported_graph(graph: IRGraph, *, max_passes: int = 8) -> IRGraph:
    for _ in range(max_passes):
        for node_id in list(graph.order):
            _canonicalize_node(graph, node)
        rebuild_graph(graph)

        for node_id in list(graph.order):
            _simplify_node(graph, node)
        rebuild_graph(graph)

        if _legalize_precisions(graph):
            rebuild_graph(graph)

        dce(graph)
```

What this pass is responsible for:

- renaming ops into canonical spellings
- turning shape-only ops into a stable `view`
- rewriting `transpose` and `movedim` into `permute`
- constant-folding trivial expressions
- materializing `ones`, `zeros`, and `arange` constants
- removing no-op casts/views/slices
- inserting FP16 legalizing casts where the runtime requires them
- dead-code elimination

It also records unsupported canonical ops in `graph.meta["canonical_unsupported_op_counts"]`.

## 11. Fusion And Graph Optimization

`optimize_graph.py` runs the semantic fusion passes:

```python
def optimize_graph(graph: IRGraph, *, max_passes: int = 8, config: FusionConfig | None = None) -> IRGraph:
    canonicalize_exported_graph(graph)
    ...
    if config.enable_rms_norm:
        fuse_rms_norm(graph)
        fuse_rms_norm_scale_multiply(graph)
    if config.enable_rope:
        fuse_rope(graph)
    if config.enable_attention:
        fuse_attention(graph)
    ...
    normalize_gemma4_decoder_attention_semantics(graph)
    fuse_dense_mlp_tq(graph)
    ...
    annotate_gold_patterns(graph)
    _prune_unused_inputs(graph)
```

This is where recognizable subgraphs collapse into semantic or fused nodes such as:

- `rms_norm`
- `rope`
- `attention`
- `attention_block`
- `self_attention_block`
- `conv_module`
- `lstm_cell`
- `gated_deltanet_prefill`
- `gated_deltanet_decode`

### Gemma4-specific optimization logic

Gemma4 gets extra handling in this pass:

- import hints and graph metadata carry layer types and sliding window info
- exported masks may be elided and rewritten into `window_size`
- some decoder attention behavior is normalized specifically for the saved runtime

## 12. Lower IR Into A Cactus `Graph`

`lower.py` is the last compiler stage. It maps `IRValue`s and `IRNode`s into the
runtime `Graph` object from `cactus.bindings.cactus`.

The high-level structure is:

```python
def transpile_preoptimized_ir(ir: IRGraph) -> TranspiledGraph:
    g = Graph()
    env = {}

    for value_id in ir.inputs:
        tensor = _lower_input_value(g, ir.values[value_id])
        env[value_id] = tensor

    for value_id, const in ir.constants.items():
        binding = _lookup_weight_binding(value)
        if binding is not None:
            binding = ensure_binding_compatible(binding, source_tensor=const)
        lowered_const = _lower_constant_value(g, value, const, binding=binding)
        env[value_id] = lowered_const

    for node_id in ir.order:
        outputs = _lower_ir_node(g, ir.nodes[node_id], env, ir)
```

### Constant lowering

This is the critical constant binding behavior:

```python
def _lower_constant_value(g, value, const, *, binding=None):
    if binding is not None:
        return g.mmap_weights(binding.path)

    if tensor_value.numel() == 1:
        return tensor_value.item()

    return _materialize_constant_tensor(g, tensor_value)
```

So constants can become:

- `g.mmap_weights(path)`
- an inline scalar
- a materialized tensor node baked into the graph

### Where weight compatibility is repaired

`weight_compat.py` sits between IR metadata and actual lowering. It can rewrite a
binding to a more executable companion file, for example:

- convert INT8 embeddings to a cached CQ embedding format
- materialize CQ4 companion files for legacy packed INT4 tensors

### What lowering implements

`_lower_ir_node()` is the actual op dispatcher. It covers:

- elementwise math and comparisons
- views, reshape, expand, permute, transpose
- matmul and linear
- embedding and gather/index paths
- reductions and softmax
- conv1d/conv2d
- norms
- fused attention-family nodes
- fused LSTM and Conformer-style blocks

If an IR op survives all prior passes and still has no lowering, `lower.py` raises:

```python
raise NotImplementedError(f"unsupported IR op in lowering: {op}")
```

### Gemma4 decoder attention fallback

One especially important special case is that Gemma4 decoder attention can bypass
the runtime attention kernel and be lowered manually as matmul + mask + softmax:

```python
def _should_lower_gemma4_decoder_attention_without_kernel(ir, node):
    family = str(ir.meta.get("adapter_family") or ir.meta.get("family") or "").strip().lower()
    component = str(ir.meta.get("component", "") or "").strip().lower()
    if family != "gemma4" or component != "decoder":
        return False
    return node.op in {"attention", "scaled_dot_product_attention"}
```

That path exists because the transpiled Gemma4 decoder has some runtime/kernel
compatibility constraints that are easier to control in lowered graph code.

## 13. Save Artifacts

After lowering, `hf_model.py` writes the IR and bundle artifacts.

For the monolithic path it saves:

- `raw_ir.json`
- `optimized_ir.json`
- `graph.cactus`
- `graph_bindings.json`
- component subgraphs under `components/`

For the component path it saves:

- `raw_ir.json`
- `optimized_ir.json`
- per-component IRs
- `components/manifest.json`
- one lowered graph per component

The component bundle writer is `_write_component_bundle()`:

```python
manifest_path = bundle_dir / "manifest.json"
_write_json(
    manifest_path,
    {
        "model_id": model_id,
        "model_source": model_source,
        "task": task,
        "family": family,
        "component_order": component_order,
        "inputs": _serialize_json_compatible(inputs_metadata),
        "components": manifest_components,
    },
)
```

Why both a `.cactus` graph and a binding manifest?

Because `Graph.save()` stores the graph structure, but the file-backed weight
bindings need to be reattached later. The manifests carry:

- logical input/output names
- runtime node ids
- paths to bound weights or saved Cactus tensor constants

## 14. Optional Execution And Reference Compare

If execution is enabled, the transpiler module will run the lowered graph immediately.

By task:

- text LM: run graph, report next token
- Gemma4 multimodal: run pipeline, report next token
- Parakeet TDT: run component encoder + custom decode loop, report transcript
- generic audio: run encoder graph and report output shapes

Many paths also compare the transpiled output against a PyTorch reference forward
unless `--skip-reference-compare` is set.

## 15. Run A Saved Bundle Later

Saved bundles are executed by the native C++ `run` binary, not by Python. The
Python CLI just resolves the bundle path and exec's the binary:

```
cactus run <bundle-path>
  → python/cactus/cli/run.py:cmd_run
  → subprocess.run([run_bin, bundle_dir, ...])
  → run invokes cactus_init(bundle_dir) from libcactus_engine
```

The runtime reads `components/manifest.json` from the bundle directory, mmaps
the `.weights` files, and loads the serialized `.cactus` graphs.

## Major Situations And How They Are Handled

### Hugging Face models that normally return caches or `ModelOutput`

`capture_pytorch.py` forces export-safe kwargs:

- `use_cache=False`
- `return_dict=False`

This keeps the capture boundary tensor-only.

### Weights directory exists but has no manifest

`hf_model.py` fails fast:

```python
if not manifest_path.exists():
    raise RuntimeError("weights_dir is missing weights_manifest.json")
```

This is intentional. Converted CQ/Cactus weights are the source of truth for
transpiled bundles; raw filename guessing is not part of the runtime contract.

### Weights directory resolves, but no bindings match

The transpile script fails hard and suggests re-running conversion:

- component path: if component graphs resolve `0` bindings
- monolithic path: if full optimized IR resolves `0` bindings

### Local snapshot exists but is incomplete

`_load_model_source()` will reject incomplete local snapshots when
`--local-files-only` is set.

### Gemma4 transformers support is missing locally

`runtime_support.py` and `hf_model.py` search alternate site-packages and
patch import compatibility for:

- missing `transformers.models.gemma4`
- missing `_lzma`
- torchvision import probes
- missing `torchvision::nms`
- missing `flex_attention.AuxRequest`

### Gemma4 local checkpoint key mismatches

`_repair_gemma4_checkpoint_weights()` remaps local checkpoint keys before loading
the model.

### Gemma4 multimodal prompt drift

The transpiler normalizes prompt and tokenization details to match native Cactus:

- inserts missing image/audio placeholder tokens
- builds a Cactus-style chat prompt when requested
- expands HF newline-merge tokens `108` and `109` back into repeated newline token
  `107`
- can replace processor-generated image/audio tensors with native-like versions

### Parakeet TDT does not run as a single generic forward

`tdt_runtime.py` provides:

- a local PyTorch reconstruction of the model
- audio preprocessing
- a greedy TDT decode loop
- component specs for an encoder graph and a recurrent decoder-step graph

### Some runtime symbols are not present in the current shared library

`runtime_compat.py` monkey-patches the Python `Graph` wrapper so the transpiler can
still build graphs against the current runtime. It provides compatibility behavior
for items like:

- generic `matmul`
- `gather(axis=0)`
- limited `conv2d`
- `not_equal`
- scalar math legalization

### Opaque import is allowed, but execution may still fail later

If `importers.py` does not know a target, it creates an opaque IR node instead of
dropping it. That preserves information, but lowering still needs a real lowering
case. Opaque nodes are therefore a sign that later lowering may stop with
`unsupported IR op`.

## Artifact Layout

A typical transpile artifact directory looks like this:

```text
transpiled/<model>/
  raw_ir.json
  optimized_ir.json
  graph.cactus
  graph_bindings.json
  result.json
  components/
    manifest.json
    vision_encoder/
      graph.cactus
      bound_constants/
        ...
    audio_encoder/
      graph.cactus
    lm_encoder/
      graph.cactus
    decoder/
      graph.cactus
```

Notes:

- `graph.cactus` is the monolithic lowered graph, when one exists.
- `components/manifest.json` is the portable bundle manifest.
- `graph_bindings.json` and per-component `bound_constant_bindings` tell the loader
  how to reattach mmap-backed weights and saved Cactus tensor constants.

## File-By-File Map

### Core transpiler files

| File | Responsibility |
| --- | --- |
| `audio_preprocess.py` | WAV loading, resampling, generic log-mel extraction, Cactus audio frontend wrappers, Parakeet-native features, Gemma4-native audio features. |
| `capture_pytorch.py` | `torch.export` capture wrapper, export-safe kwargs injection, metadata collection, shape propagation, and `CapturedModel` creation. |
| `component_partition.py` | Heuristic component labeling of IR nodes/values and subgraph extraction into `audio_encoder` / `vision_encoder` / `lm_encoder` / `decoder`. |
| `component_pipeline.py` | Capture/optimize/lower loop for one component spec, plus runtime execution of a staged pipeline of components. |
| `graph_ir.py` | Definitions for `IRValue`, `IRNode`, `IRGraph`, deep-copy behavior, and IR consistency verification. |
| `import_ir.py` | Top-level importer from exported FX graph into `IRGraph`; resolves lifted constants, weight binding metadata, and inlined subgraphs. |
| `import_semantics.py` | Early semantic rewrites done immediately after import, especially attention and RoPE recognition. |
| `importers.py` | Large target-to-IR importer registry; converts normalized PyTorch ops into canonical `IRNode`s. |
| `lower.py` | Final lowering from optimized IR into runtime `Graph`, including op lowering, constant binding, fused op handling, and execution wrapper. |
| `model_adapters.py` | Family- and task-specific forward adapters, Gemma4 multimodal pipeline adapters, task canonicalization, and component spec builders. |
| `model_patterns.py` | Topology/op catalog used to annotate recognizable transformer structures after optimization. |
| `model_profiles.py` | Compact model-family profile constants, including compatibility modules, graph context defaults, and weight-name aliases used by fallback runtimes. |
| `multimodal_runtime.py` | Shared multimodal runtime helpers used by bundle execution: prompt normalization, image/audio limiting, and native-like multimodal preprocessing. |
| `normalize.py` | Target-name normalization (`aten.*` to canonical names) and dtype-name normalization into IR spellings. |
| `ops.py` | Canonical op registry and alias table for IR ops, including backend op names and attr schemas. |
| `optimize_graph.py` | Main fusion/optimization pass driver, pattern fusion, Gemma4 attention normalization, and gold-pattern annotation. |
| `runtime_support.py` | Shared import/runtime compatibility helpers for optional Transformers modules, torchvision probes, and flex-attention shims. |
| `tdt_runtime.py` | TDT fallback runtime: local recurrent decoder loop, config loader, and encoder/decoder component-spec generation. |
| `runtime_compat.py` | Runtime wrapper patches that make the current Python `Graph` API usable for transpiled graphs even when some C symbols are absent. |
| `weight_binding.py` | Resolves IR constants to converted `.weights` files through `weights_manifest.json`; also resolves default weights dirs. |
| `weight_compat.py` | Makes bound weights executable with the current runtime by materializing compatibility companion files when needed. |

### Canonicalization subdirectory

| File | Responsibility |
| --- | --- |
| `canonicalize/cleanup.py` | Multi-pass canonical cleanup, constant folding, no-op removal, attr normalization, and precision legalization. |
| `canonicalize/utils.py` | Shared graph rebuild, DCE, node removal, dtype helpers, and constant materialization utilities. |

### Fusion subdirectory

| File | Responsibility |
| --- | --- |
| `fusion/__init__.py` | Re-exports the fusion matchers used by `optimize_graph.py`. |
| `fusion/common.py` | Shared fusion helper types and graph traversal helpers. |
| `fusion/linear.py` | Matcher for linear projection patterns. |
| `fusion/rms_norm.py` | Matcher for RMSNorm computational subgraphs. |
| `fusion/rope.py` | Matcher for rotary embedding patterns. |
| `fusion/attention.py` | Matchers for attention, attention blocks, self-attention blocks, and GQA-style projection patterns. |
| `fusion/conv.py` | Matcher for Conformer-like convolution modules. |
| `fusion/lstm.py` | Matcher for LSTM cell subgraphs. |
| `fusion/mlp.py` | Matcher for gated MLP branches such as GELU/SiLU gate-up-down patterns. |
| `fusion/rel_pos_bias.py` | Matcher for relative-position-bias subgraphs. |
| `fusion/deltanet.py` | Matcher for fused gated DeltaNet patterns. |

## CLI Usage

## 1. Basic Convert

The simplest invocation (internal command, not in the CLI menu) is:

```bash
cactus convert <hf-model-id>
```

This quantizes the weights to CQ and then builds the runtime graph. The graph stage:

- requires the converted Cactus CQ weights (produced by the same command, or passed
  with `--weights-dir`; default lookup is `weights/<model>`)
- resolves the runtime library
- launches `python/cactus/transpile/hf_model.py`
- saves artifacts under `weights/<model>/` (alongside the converted CQ weights; override with `--artifact-dir`)
- adds `--skip-execute`, so it does not run the graph unless you ask

To stop after the CQ weights (skip the graph), pass `--weights-only`. To also execute
a reference run right after building the graph, pass `--execute-after-transpile`.

## 2. Graph Options

The graph-build options are parsed by `python/cactus/transpile/hf_model.py` and exposed
on `cactus convert`.

Common ones:

- `--task`
- `--prompt`
- `--input-ids`
- `--audio-file`
- `--image-file`
- `--weights-dir`
- `--artifact-dir`
- `--component-pipeline`
- `--skip-reference-compare`
- `--graph-filename`
- `--no-fuse-rms-norm`
- `--no-fuse-rope`
- `--no-fuse-attention`
- `--allow-unconverted-weights` for compiler-only debugging

Examples:

### Text causal LM

```bash
cactus convert <text-model-id> /path/to/converted_weights --bits 4 \
  --prompt "The capital of France is" \
  --artifact-dir ./transpiled/text_demo
```

### Gemma4 multimodal

```bash
cactus convert <gemma4-model-id> \
  --task multimodal_causal_lm_logits \
  --image-file /path/to/image.jpg \
  --audio-file /path/to/audio.wav \
  --prompt "Describe what is happening in the image and audio."
```

### Whisper-style encoder capture

```bash
cactus convert <whisper-model-id> /path/to/converted_weights --bits 4 \
  --audio-file /path/to/sample.wav \
  --task encoder_hidden_states
```

### Parakeet TDT

```bash
cactus convert <parakeet-tdt-model-id> \
  --audio-file /path/to/sample.wav \
  --task tdt_transcription
```

If you want the full direct script help:

```bash
python -m cactus.transpile.hf_model --help
```

## 3. Run A Saved Transpiled Bundle

Use the bundle root directory:

```bash
cactus run /path/to/transpiled/model
```

### Text causal LM bundle

```bash
cactus run /path/to/transpiled/model \
  --prompt "The capital of France is"
```

Or pass token ids directly:

```bash
cactus run /path/to/transpiled/model \
  --input-ids 2,13,42
```

### Multimodal Gemma4 bundle

```bash
cactus run /path/to/transpiled/model \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "What do you observe?" \
  --system "Answer concisely."
```

### Audio encoder / transcription bundle

```bash
cactus run /path/to/transpiled/model \
  --audio /path/to/sample.wav
```

### Save result payload

```bash
cactus run /path/to/transpiled/model \
  --prompt "Hello" \
  --result-json ./result.json
```

Internally, `cli/model.py:ensure_runnable_bundle` checks whether the positional
argument is a local path with `components/manifest.json` and dispatches the
native `run` binary against it.

## Current Runtime Limitations

As of the current code:

- Saved bundle execution is implemented for `causal_lm_logits`,
  `multimodal_causal_lm_logits`, `encoder_hidden_states`, and
  `tdt_transcription`.
- `ctc_logits` can be transpiled, but the runtime does not currently implement
  a saved-bundle executor for that task.
- The component pipeline is model-family-specific; it is not a universal split-IR
  path.

## Useful Environment Variables

- `CACTUS_TRANSPILER_WEIGHTS_DIR`:
  fallback weights directory used by `weight_binding.py`
- `CACTUS_TRANSPILER_WEIGHTS_DIR_<FAMILY>`:
  family-specific weights directory override
- `CACTUS_TRANSPILER_DEBUG_MMAP=1`:
  log constant binding behavior during lowering
- `CACTUS_GEMMA4_CAPTURE_FP32=1`:
  temporarily promote some Gemma4 text modules to FP32 during capture on CPU

## Practical Summary

If you only want the high-level lifecycle, it is:

1. `cactus convert` quantizes the weights, then launches `python/cactus/transpile/hf_model.py`.
2. The script infers a task, loads the HF model, and creates one example input batch.
3. `model_adapters.py` wraps the model into a smaller canonical task interface.
4. `capture_pytorch.py` runs `torch.export`.
5. `import_ir.py` + `importers.py` convert the export to `IRGraph`.
6. `import_semantics.py` adds early semantic rewrites.
7. `canonicalize/cleanup.py` normalizes and simplifies the IR.
8. `optimize_graph.py` fuses patterns into higher-level semantic ops.
9. `lower.py` builds a runtime `Graph`, mmaps bound weights when possible, and
   materializes everything else.
10. The script writes `raw_ir.json`, `optimized_ir.json`, `graph.cactus`, and
    `components/manifest.json`.
11. `cactus run` later resolves the bundle path and invokes the native `run`
    binary, which mmaps the weights, loads the serialized graph, and runs it
    via `cactus_init()` + `cactus_complete()` (see `cactus-engine`).
