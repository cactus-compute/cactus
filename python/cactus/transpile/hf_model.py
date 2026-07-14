from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cactus.transpile.runtime_compat import Graph
from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.convert.model_adapters.nemo import ensure_parakeet_tdt_nemo_source
from cactus.transpile.audio_preprocess import generic_log_mel_features as _generic_log_mel_features
from cactus.transpile.audio_preprocess import load_audio_waveform as _load_audio_waveform
from cactus.transpile.audio_preprocess import prepare_cactus_audio_features
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.component_partition import extract_component_subgraphs
from cactus.transpile.component_partition import summarize_ir_components
from cactus.transpile.component_plan import infer_component_plan_from_config
from cactus.transpile.component_pipeline import capture_component_spec
from cactus.transpile.component_pipeline import execute_component_pipeline
from cactus.transpile.multimodal_runtime import prepare_gemma4_multimodal_inputs as _shared_prepare_gemma4_multimodal_inputs
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.lower import TranspiledGraph
from cactus.transpile.lower import _lower_constant_value
from cactus.transpile.lower import _lower_input_value
from cactus.transpile.lower import _lower_ir_node
from cactus.transpile.lower import _lookup_weight_binding
from cactus.transpile.media_limits import resize_static_image
from cactus.transpile.model_adapters import UnsupportedComponentsError
from cactus.transpile.model_adapters import build_component_module_specs
from cactus.transpile.model_adapters import canonicalize_model_interface
from cactus.transpile.model_profiles import multimodal_context_tokens_for_model_type
from cactus.transpile.optimize_graph import FusionConfig
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.runtime_support import ensure_transformers_supports_model_type as _ensure_transformers_supports_profiled_model_type
from cactus.transpile.runtime_support import patch_torch_flex_attention_compat as _patch_torch_flex_attention_compat
from cactus.transpile.runtime_support import patch_transformers_torchvision_probe as _patch_transformers_torchvision_probe
from cactus.transpile.tdt_runtime import greedy_decode_parakeet_tdt_token_ids
from cactus.transpile.tdt_runtime import load_tdt_local_model
from cactus.transpile.tdt_runtime import prepare_parakeet_tdt_audio_features
from cactus.transpile.weight_compat import ensure_binding_compatible

_DEFAULT_CAUSAL_PROMPT = "The capital of France is"
_DEFAULT_MULTIMODAL_CONTEXT_TOKENS = 2048


@dataclass
class PreparedInputs:
    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    metadata: dict[str, object]


def _ensure_transformers_supports_model_type(model_type: str) -> str | None:
    return _ensure_transformers_supports_profiled_model_type(model_type)


def _resolve_local_snapshot(model_id_or_path: str) -> str | None:
    explicit = Path(model_id_or_path)
    if explicit.exists():
        nemo_export = ensure_parakeet_tdt_nemo_source(model_id_or_path)
        if nemo_export is not None:
            return nemo_export
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id_or_path.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    snapshot = str(snapshots[-1])
    nemo_export = ensure_parakeet_tdt_nemo_source(snapshot)
    if nemo_export is not None:
        return nemo_export
    return snapshot


def _snapshot_has_model_weights(path: str | Path) -> bool:
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return False
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((root / name).exists() for name in candidates)


def _validate_weights_dir(weights_dir: str | None, *, model_id: str) -> Path | None:
    if not weights_dir:
        return None

    root = Path(weights_dir).resolve()
    if not root.exists():
        raise RuntimeError(
            f"weights_dir does not exist: {root}\n"
            "\n"
            f"Create the folder first with:\n"
            f"  cactus convert {model_id} {root}\n"
        )

    manifest_path = root / "weights_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"weights_dir is missing weights_manifest.json: {manifest_path}\n"
            "\n"
            "The transpiler binds weights only through the converted CQ/Cactus "
            "manifest; it no longer guesses filenames from model-specific layer names.\n"
            "\n"
            f"Re-convert with the current converter:\n"
            f"  cactus convert {model_id} {root}\n"
        )

    return root


def _serialize_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json_compatible(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_compatible(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        payload = {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if value.numel() == 1 and not value.is_meta:
            payload["data"] = value.detach().cpu().reshape(-1)[0].item()
        return payload
    if isinstance(value, np.ndarray):
        payload = {
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if value.size == 1:
            payload["data"] = np.asarray(value).reshape(-1)[0].item()
        return payload
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _serialize_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__module__}.{type(value).__name__}>"


def _slugify_model_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def _default_artifact_dir_for_model(model_id: str) -> Path:
    return (Path.cwd() / "transpiled" / _slugify_model_artifact_name(model_id)).resolve()


def _graph_to_dict(graph) -> dict[str, object]:
    return {
        "meta": _serialize_json_compatible(graph.meta),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "constants": {
            value_id: _serialize_json_compatible(constant)
            for value_id, constant in graph.constants.items()
        },
        "values": {
            value_id: _serialize_json_compatible(value)
            for value_id, value in graph.values.items()
        },
        "nodes": [
            _serialize_json_compatible(graph.nodes[node_id])
            for node_id in graph.order
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _component_artifact_name(prefix: str, component: str) -> str:
    return f"{prefix}_{component}.json"


def _component_graphs_to_payload(component_graphs: dict[str, IRGraph]) -> dict[str, object]:
    return {
        component: _graph_to_dict(graph)
        for component, graph in component_graphs.items()
    }


def _binding_entries_by_node_id(
    bindings: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for binding in bindings:
        try:
            result[int(binding["node_id"])] = binding
        except Exception:
            continue
    return result


def _externalize_shared_rope_tables(
    transpiled_graph: TranspiledGraph,
    *,
    weights_root: Path,
    registry: dict[str, str],
) -> int:
    value_ids = getattr(transpiled_graph, "bound_constant_value_ids", {}) or {}
    already_bound = {
        int(binding["node_id"])
        for binding in transpiled_graph.bound_constant_bindings
        if isinstance(binding, dict) and "node_id" in binding
    }
    externalized = 0
    for constant in transpiled_graph.bound_constants:
        node_id = int(constant.id)
        if node_id in already_bound:
            continue
        value_id = str(value_ids.get(node_id, ""))
        if not value_id.startswith("c_rope_table_"):
            continue
        data = np.ascontiguousarray(constant.numpy())
        digest = hashlib.sha1(data.tobytes()).hexdigest()[:16]
        filename = registry.get(digest)
        if filename is None:
            filename = f"rope_table_{digest}.weights"
            save_tensor_with_header(data, weights_root / filename, precision="FP16")
            registry[digest] = filename
        transpiled_graph.bound_constant_bindings.append(
            {
                "node_id": node_id,
                "value_id": value_id,
                "path": filename,
                "kind": "weight",
                "source_name": value_id,
            }
        )
        externalized += 1
    return externalized


def _embed_materialized_bound_constants(transpiled_graph: TranspiledGraph) -> int:
    """Serialize small graph-local constants into the .cactus graph itself.

    CQ model weights remain external mmap files. Only materialized constants
    without an existing weight binding are embedded.
    """
    existing_bindings = list(_serialize_json_compatible(transpiled_graph.bound_constant_bindings))
    external_node_ids = set(_binding_entries_by_node_id(existing_bindings))
    mark_embedded_input = getattr(transpiled_graph.graph, "mark_embedded_input", None)
    if not callable(mark_embedded_input):
        raise RuntimeError(
            "Cactus runtime does not support embedded graph constants; rebuild with cactus build --python"
        )

    embedded_count = 0
    for constant_tensor in transpiled_graph.bound_constants:
        node_id = int(constant_tensor.id)
        if node_id in external_node_ids:
            continue
        mark_embedded_input(constant_tensor)
        embedded_count += 1
    return embedded_count


def _write_graph_binding_manifest(
    *,
    artifact_dir: Path,
    filename: str,
    model_id: str,
    model_source: str,
    task: str,
    family: str,
    inputs_metadata: dict[str, object],
    transpiled_graph: TranspiledGraph,
) -> Path:
    manifest_path = artifact_dir / filename
    _write_json(
        manifest_path,
        {
            "model_id": model_id,
            "model_source": model_source,
            "task": task,
            "family": family,
            "inputs": _serialize_json_compatible(inputs_metadata),
            "runtime_input_node_ids": [int(tensor.id) for tensor in transpiled_graph.runtime_inputs],
            "output_node_ids": [int(tensor.id) for tensor in transpiled_graph.outputs],
            "bound_constant_bindings": _serialize_json_compatible(transpiled_graph.bound_constant_bindings),
        },
    )
    return manifest_path


def _export_lfm2_vl_position_embedding_grid(
    *,
    model,
    artifact_dir: Path,
    component_metadata: dict[str, dict[str, object]],
) -> None:
    root = getattr(model, "model", None)
    vision_tower = getattr(root, "vision_tower", None)
    vision_model = getattr(vision_tower, "vision_model", None)
    embeddings = getattr(vision_model, "embeddings", None)
    if embeddings is None:
        return
    pos_size = int(embeddings.position_embedding_size)
    embed_dim = int(embeddings.embed_dim)
    grid = (
        embeddings.position_embedding.weight.detach().cpu().to(torch.float32)
        .reshape(pos_size, pos_size, embed_dim)
        .contiguous()
        .numpy()
    )
    rel_path = Path("components") / "vision_encoder" / "position_embedding_grid.f32"
    out_path = artifact_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.tofile(out_path)
    meta = component_metadata.setdefault("vision_encoder", {})
    meta["position_embedding_grid_path"] = str(rel_path)
    meta["position_embedding_grid_shape"] = f"{pos_size},{pos_size},{embed_dim}"
    print(f"saved_vision_position_embedding_grid={out_path}")


def _write_component_bundle(
    *,
    artifact_dir: Path,
    model_id: str,
    model_source: str,
    task: str,
    family: str,
    inputs_metadata: dict[str, object],
    raw_component_graphs: dict[str, IRGraph],
    optimized_component_graphs: dict[str, IRGraph],
    transpiled_component_graphs: dict[str, TranspiledGraph] | None = None,
    component_io_signatures: dict[str, dict[str, tuple[str, ...]]] | None = None,
    component_metadata: dict[str, dict[str, object]] | None = None,
    graph_filename: str = "graph.cactus",
) -> Path:
    bundle_dir = artifact_dir / "components"
    component_order = [
        component
        for component in (
            "source_encoder",
            "audio_encoder",
            "vision_encoder",
            "vision_projector",
            "text_embedding",
            "lm_encoder",
            "lm_encoder_text_chunk",
            "decoder_prefill_chunk",
            "decoder_embed_chunk",
            "decoder",
            "decoder_cross_kv",
            "lm_encoder_step",
            "lm_encoder_media_step",
            "decoder_media_step",
            "decoder_step",
            "unspecified",
        )
        if component in raw_component_graphs or component in optimized_component_graphs
    ]
    extra_components = sorted(
        component
        for component in set(raw_component_graphs) | set(optimized_component_graphs)
        if component not in component_order
    )
    component_order.extend(extra_components)

    manifest_components: list[dict[str, object]] = []
    
    rope_table_registry: dict[str, str] = {}
    for component in component_order:
        raw_graph = raw_component_graphs.get(component)
        optimized_graph = optimized_component_graphs.get(component)
        transpiled_graph = None if transpiled_component_graphs is None else transpiled_component_graphs.get(component)
        component_dir = bundle_dir / component
        raw_relpath = None
        optimized_relpath = None
        graph_relpath = None
        if raw_graph is not None:
            raw_relpath = Path(component) / "raw_ir.json"
            _write_json(
                bundle_dir / raw_relpath,
                {
                    "model_id": model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": family,
                    "component": component,
                    "inputs": _serialize_json_compatible(inputs_metadata),
                    "graph": _graph_to_dict(raw_graph),
                },
            )
        if optimized_graph is not None:
            optimized_relpath = Path(component) / "optimized_ir.json"
            _write_json(
                bundle_dir / optimized_relpath,
                {
                    "model_id": model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": family,
                    "component": component,
                    "inputs": _serialize_json_compatible(inputs_metadata),
                    "graph": _graph_to_dict(optimized_graph),
                },
            )
        if transpiled_graph is not None:
            graph_relpath = Path(component) / graph_filename
            component_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(component_dir / "bound_constants", ignore_errors=True)
            _externalize_shared_rope_tables(
                transpiled_graph,
                weights_root=artifact_dir,
                registry=rope_table_registry,
            )
            _embed_materialized_bound_constants(transpiled_graph)
            transpiled_graph.graph.save(bundle_dir / graph_relpath)

        graph_for_signature = optimized_graph or raw_graph
        if graph_for_signature is None:
            continue
        manifest_components.append(
            {
                "component": component,
                "directory": str(component_dir.relative_to(artifact_dir)),
                "raw_ir": None if raw_relpath is None else str((bundle_dir / raw_relpath).relative_to(artifact_dir)),
                "optimized_ir": None if optimized_relpath is None else str((bundle_dir / optimized_relpath).relative_to(artifact_dir)),
                "graph": None if graph_relpath is None else str((bundle_dir / graph_relpath).relative_to(artifact_dir)),
                "inputs": list(graph_for_signature.inputs),
                "outputs": list(graph_for_signature.outputs),
                "logical_inputs": list((component_io_signatures or {}).get(component, {}).get("input_keys", ())),
                "logical_outputs": list((component_io_signatures or {}).get(component, {}).get("output_keys", ())),
                "metadata": _serialize_json_compatible((component_metadata or {}).get(component, {})),
                "node_count": len(graph_for_signature.order),
                "weight_binding_count": _count_weight_bindings(graph_for_signature),
                "runtime_input_node_ids": [] if transpiled_graph is None else [int(tensor.id) for tensor in transpiled_graph.runtime_inputs],
                "output_node_ids": [] if transpiled_graph is None else [int(tensor.id) for tensor in transpiled_graph.outputs],
                "cache_state_node_ids": [] if transpiled_graph is None else [
                    {
                        "layer_key": str(layer_key),
                        "key": int(key_tensor.id),
                        "value": int(value_tensor.id),
                    }
                    for layer_key, key_tensor, value_tensor in getattr(transpiled_graph, "cache_state_tensors", [])
                ],
                "bound_constant_bindings": [] if transpiled_graph is None else (
                    list(_serialize_json_compatible(transpiled_graph.bound_constant_bindings))
                ),
            }
        )

    manifest_path = bundle_dir / "manifest.json"
    manifest_payload: dict[str, object] = {
        "model_id": model_id,
        "model_source": model_source,
        "task": task,
        "family": family,
        "component_order": component_order,
        "inputs": _serialize_json_compatible(inputs_metadata),
        "components": manifest_components,
    }
    _write_json(manifest_path, manifest_payload)
    return manifest_path


def _named_tensor_store(prepared: PreparedInputs) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in zip(prepared.names, prepared.tensors, strict=True)
    }


def _aggregate_component_counts(component_graphs: dict[str, IRGraph]) -> dict[str, int]:
    return {
        component: len(graph.order)
        for component, graph in sorted(component_graphs.items())
    }


def _aggregate_component_op_counts(component_graphs: dict[str, IRGraph]) -> Counter:
    counter: Counter[str] = Counter()
    for graph in component_graphs.values():
        counter.update(graph.nodes[node_id].op for node_id in graph.order)
    return counter


def _count_component_weight_bindings(component_graphs: dict[str, IRGraph]) -> int:
    return sum(_count_weight_bindings(graph) for graph in component_graphs.values())


def _execute_gemma4_component_pipeline(
    *,
    component_graphs,
    prepared: PreparedInputs,
) -> np.ndarray:
    store, _ = execute_component_pipeline(
        [component_graphs[name] for name in ("vision_encoder", "audio_encoder", "lm_encoder", "decoder")],
        initial_store=_named_tensor_store(prepared),
    )
    logits = store["logits"]
    return np.asarray(logits, dtype=np.float32)


def _run_parakeet_tdt_component_decode(
    *,
    component_graphs,
    model,
    prepared: PreparedInputs,
) -> dict[str, object]:
    store, _ = execute_component_pipeline(
        [component_graphs["audio_encoder"]],
        initial_store=_named_tensor_store(prepared),
    )
    encoder_hidden_states = np.asarray(store["encoder_hidden_states"])
    batch_size = int(encoder_hidden_states.shape[0])
    if batch_size != 1:
        raise ValueError("Parakeet TDT component decode currently expects batch size 1")

    hidden_dtype = prepared.tensors[0].dtype
    initial_states = model.initial_decoder_state(
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=hidden_dtype,
    )
    initial_state_arrays = tuple(state.detach().cpu().numpy() for state in initial_states)
    decoder_component = component_graphs["decoder"]

    def _step(
        frame: np.ndarray,
        token_id: int,
        state_values: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        input_store: dict[str, object] = {
            "encoder_frame": np.ascontiguousarray(frame),
            "token_ids": np.full((batch_size,), token_id, dtype=np.int64),
        }
        for index in range(model.config.predictor_num_layers):
            input_store[f"state_h_{index}"] = np.ascontiguousarray(state_values[index * 2])
            input_store[f"state_c_{index}"] = np.ascontiguousarray(state_values[index * 2 + 1])

        runtime_inputs = [input_store[key] for key in decoder_component.input_keys]
        decoder_component.transpiled_graph.set_inputs(runtime_inputs)
        outputs = decoder_component.transpiled_graph.execute()
        logits = outputs[0].numpy().astype(np.float32, copy=False)
        next_states = tuple(output.numpy() for output in outputs[1:])
        return logits, next_states

    emitted = greedy_decode_parakeet_tdt_token_ids(
        config=model.config,
        encoder_hidden_states=encoder_hidden_states,
        initial_states=initial_state_arrays,
        step=_step,
    )

    return {
        "token_ids": emitted,
        "transcript": model.decode_token_ids(emitted),
        "encoder_hidden_shape": list(encoder_hidden_states.shape),
    }


def _run_component_pipeline_transpile(
    *,
    args,
    task: str,
    family: str,
    model_source: str,
    model,
    prepared: PreparedInputs,
    component_specs,
    fusion_config: FusionConfig,
    weights_dir: str | None,
    artifact_dir: Path | None,
    processor_or_tokenizer,
    canonical,
) -> int:
    print(f"model_id={args.model_id}")
    print(f"model_source={model_source}")
    print(f"task={task}")
    print(f"adapter_family={family}")
    print("adapter_module=component_pipeline")
    print(f"input_names={','.join(prepared.names)}")
    for name, tensor in zip(prepared.names, prepared.tensors, strict=True):
        print(f"input_{name}_shape={list(tensor.shape)}")
    if weights_dir:
        print(f"weights_dir={weights_dir}")

    print("capture_begin=true", flush=True)
    captured_components = {}
    for spec in component_specs:
        print(f"capture_component_begin={spec.component}", flush=True)
        captured_components[spec.component] = capture_component_spec(spec, fusion_config=fusion_config)
        print(f"capture_component_done={spec.component}", flush=True)
    print("capture_done=true", flush=True)

    raw_component_graphs = {
        name: captured.raw_ir_graph
        for name, captured in captured_components.items()
    }
    optimized_component_graphs = {
        name: captured.optimized_ir_graph
        for name, captured in captured_components.items()
    }
    transpiled_component_graphs = {
        name: captured.transpiled_graph
        for name, captured in captured_components.items()
    }
    component_io_signatures = {
        name: {
            "input_keys": tuple(captured.input_keys),
            "output_keys": tuple(captured.output_keys),
        }
        for name, captured in captured_components.items()
    }
    component_metadata = {
        name: dict(captured.metadata)
        for name, captured in captured_components.items()
    }

    raw_ir_nodes = sum(len(graph.order) for graph in raw_component_graphs.values())
    optimized_ir_nodes = sum(len(graph.order) for graph in optimized_component_graphs.values())
    binding_count = _count_component_weight_bindings(optimized_component_graphs)
    raw_component_counts = _aggregate_component_counts(raw_component_graphs)
    optimized_component_counts = _aggregate_component_counts(optimized_component_graphs)
    op_counts = _aggregate_component_op_counts(optimized_component_graphs)

    print(f"raw_ir_nodes={raw_ir_nodes}")
    print(f"optimized_ir_nodes={optimized_ir_nodes}")
    print(f"weight_bindings={binding_count}")
    if raw_component_counts:
        print(
            "raw_components="
            + ",".join(f"{name}:{count}" for name, count in sorted(raw_component_counts.items()))
        )
    if optimized_component_counts:
        print(
            "optimized_components="
            + ",".join(f"{name}:{count}" for name, count in sorted(optimized_component_counts.items()))
        )
    print(
        "ops="
        f"attention:{op_counts.get('attention', 0)} "
        f"conv1d:{op_counts.get('conv1d', 0)} "
        f"conv2d:{op_counts.get('conv2d', 0)} "
        f"batch_norm:{op_counts.get('batch_norm', 0)} "
        f"layer_norm:{op_counts.get('layer_norm', 0)} "
        f"rms_norm:{op_counts.get('rms_norm', 0)} "
        f"rope:{op_counts.get('rope', 0)} "
        f"linear:{op_counts.get('linear', 0)}"
    )

    if weights_dir and binding_count == 0:
        raise RuntimeError(
            f"No weight bindings were resolved from {weights_dir}\n"
            "\n"
            "The weights folder exists, but none of the component graphs matched entries in weights_manifest.json.\n"
            "\n"
            f"Recommended fix:\n"
            f"  cactus convert {args.model_id} {weights_dir}\n"
        )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_dir / "raw_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "component_order": [spec.component for spec in component_specs],
                "components": _component_graphs_to_payload(raw_component_graphs),
            },
        )
        _write_json(
            artifact_dir / "optimized_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "component_order": [spec.component for spec in component_specs],
                "components": _component_graphs_to_payload(optimized_component_graphs),
            },
        )
        for component, component_graph in raw_component_graphs.items():
            _write_json(
                artifact_dir / _component_artifact_name("raw_ir", component),
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": family,
                    "component": component,
                    "inputs": _serialize_json_compatible(prepared.metadata),
                    "graph": _graph_to_dict(component_graph),
                },
            )
        for component, component_graph in optimized_component_graphs.items():
            _write_json(
                artifact_dir / _component_artifact_name("optimized_ir", component),
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": family,
                    "component": component,
                    "inputs": _serialize_json_compatible(prepared.metadata),
                    "graph": _graph_to_dict(component_graph),
                },
            )
        print(f"saved_raw_ir={artifact_dir / 'raw_ir.json'}")
        print(f"saved_optimized_ir={artifact_dir / 'optimized_ir.json'}")
        for component in raw_component_graphs:
            print(f"saved_raw_component_ir_{component}={artifact_dir / _component_artifact_name('raw_ir', component)}")
        for component in optimized_component_graphs:
            print(
                f"saved_optimized_component_ir_{component}="
                f"{artifact_dir / _component_artifact_name('optimized_ir', component)}"
            )

        if family == "lfm2_vl" and "vision_encoder" in component_metadata:
            _export_lfm2_vl_position_embedding_grid(
                model=model,
                artifact_dir=artifact_dir,
                component_metadata=component_metadata,
            )

        component_manifest_path = _write_component_bundle(
            artifact_dir=artifact_dir,
            model_id=args.model_id,
            model_source=model_source,
            task=task,
            family=family,
            inputs_metadata=prepared.metadata,
            raw_component_graphs=raw_component_graphs,
            optimized_component_graphs=optimized_component_graphs,
            transpiled_component_graphs=transpiled_component_graphs,
            component_io_signatures=component_io_signatures,
            component_metadata=component_metadata,
            graph_filename=args.graph_filename,
        )
        print(f"saved_component_bundle_manifest={component_manifest_path}")
        for component in transpiled_component_graphs:
            print(
                f"saved_component_graph_{component}="
                f"{artifact_dir / 'components' / component / args.graph_filename}"
            )
        print("note=saved split component graphs; use the component bundle manifest to rebind mmap weights/embeddings when loading")

    if args.skip_execute:
        return 0

    if task == "multimodal_causal_lm_logits":
        print("execute_begin=true")
        transpiled_output = _execute_gemma4_component_pipeline(
            component_graphs=captured_components,
            prepared=prepared,
        )
        print("execute_done=true")
        tokenizer_like = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        if args.skip_reference_compare:
            result_payload = {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "output_shape": list(transpiled_output.shape),
                "raw_ir_nodes": raw_ir_nodes,
                "optimized_ir_nodes": optimized_ir_nodes,
                "weight_bindings": binding_count,
                "reference_compare_skipped": True,
            }
            print(f"output_shape={list(transpiled_output.shape)}")
            transpiled_next = int(np.argmax(transpiled_output[0, -1]))
            print(f"transpiled_next_token_id={transpiled_next}")
            result_payload["transpiled_next_token_id"] = transpiled_next
            if hasattr(tokenizer_like, "decode"):
                transpiled_token = tokenizer_like.decode([transpiled_next])
                print(f"transpiled_next_token={transpiled_token!r}")
                result_payload["transpiled_next_token"] = transpiled_token
            if artifact_dir is not None:
                _write_json(artifact_dir / "result.json", result_payload)
                print(f"saved_result={artifact_dir / 'result.json'}")
            return 0

        print("reference_begin=true")
        with torch.no_grad():
            reference_output = canonical.module(*prepared.tensors).detach().float().cpu().numpy()
        print("reference_done=true")
        max_abs_diff = float(np.max(np.abs(reference_output - transpiled_output)))
        mean_abs_diff = float(np.mean(np.abs(reference_output - transpiled_output)))
        hf_next = int(np.argmax(reference_output[0, -1]))
        transpiled_next = int(np.argmax(transpiled_output[0, -1]))
        print(f"hf_next_token_id={hf_next}")
        print(f"transpiled_next_token_id={transpiled_next}")
        print(f"logits_max_abs_diff={max_abs_diff:.6f}")
        print(f"logits_mean_abs_diff={mean_abs_diff:.6f}")
        if hasattr(tokenizer_like, "decode"):
            print(f"hf_next_token={tokenizer_like.decode([hf_next])!r}")
            print(f"transpiled_next_token={tokenizer_like.decode([transpiled_next])!r}")
        result_payload = {
            "model_id": args.model_id,
            "model_source": model_source,
            "task": task,
            "family": family,
            "inputs": _serialize_json_compatible(prepared.metadata),
            "output_shape": list(reference_output.shape),
            "raw_ir_nodes": raw_ir_nodes,
            "optimized_ir_nodes": optimized_ir_nodes,
            "weight_bindings": binding_count,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "hf_next_token_id": hf_next,
            "transpiled_next_token_id": transpiled_next,
        }
        if hasattr(tokenizer_like, "decode"):
            result_payload["hf_next_token"] = tokenizer_like.decode([hf_next])
            result_payload["transpiled_next_token"] = tokenizer_like.decode([transpiled_next])
        if artifact_dir is not None:
            _write_json(artifact_dir / "result.json", result_payload)
            print(f"saved_result={artifact_dir / 'result.json'}")
        return 0

    if task == "causal_lm_logits":
        tokenizer_like = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        result_payload = {
            "model_id": args.model_id,
            "model_source": model_source,
            "task": task,
            "family": family,
            "inputs": _serialize_json_compatible(prepared.metadata),
            "raw_ir_nodes": raw_ir_nodes,
            "optimized_ir_nodes": optimized_ir_nodes,
            "weight_bindings": binding_count,
        }
        if "decoder" in captured_components:
            print("execute_begin=true")
            initial_store = {
                name: tensor
                for name, tensor in zip(prepared.names, prepared.tensors, strict=True)
            }
            store, _ = execute_component_pipeline(
                [captured_components["decoder"]],
                initial_store=initial_store,
            )
            print("execute_done=true")
            transpiled_output = np.asarray(store["logits"])
            transpiled_next = int(np.argmax(transpiled_output[0, -1]))
            print(f"output_shape={list(transpiled_output.shape)}")
            print(f"transpiled_next_token_id={transpiled_next}")
            result_payload["output_shape"] = list(transpiled_output.shape)
            result_payload["transpiled_next_token_id"] = transpiled_next
            if hasattr(tokenizer_like, "decode"):
                transpiled_token = tokenizer_like.decode([transpiled_next])
                print(f"transpiled_next_token={transpiled_token!r}")
                result_payload["transpiled_next_token"] = transpiled_token
            if not args.skip_reference_compare and canonical is not None:
                print("reference_begin=true")
                with torch.no_grad():
                    reference_output = canonical.module(*prepared.tensors).detach().float().cpu().numpy()
                print("reference_done=true")
                max_abs_diff = float(np.max(np.abs(reference_output - transpiled_output)))
                mean_abs_diff = float(np.mean(np.abs(reference_output - transpiled_output)))
                hf_next = int(np.argmax(reference_output[0, -1]))
                print(f"hf_next_token_id={hf_next}")
                print(f"logits_max_abs_diff={max_abs_diff:.6f}")
                print(f"logits_mean_abs_diff={mean_abs_diff:.6f}")
                result_payload["hf_next_token_id"] = hf_next
                result_payload["max_abs_diff"] = max_abs_diff
                result_payload["mean_abs_diff"] = mean_abs_diff
                if hasattr(tokenizer_like, "decode"):
                    result_payload["hf_next_token"] = tokenizer_like.decode([hf_next])
        else:
            result_payload["reference_compare_skipped"] = True
            result_payload["note"] = "causal LM component bundle contains only cached decode step components"
        if artifact_dir is not None:
            _write_json(artifact_dir / "result.json", result_payload)
            print(f"saved_result={artifact_dir / 'result.json'}")
        return 0

    if task == "tdt_transcription":
        print("execute_begin=true")
        transpiled_decode = _run_parakeet_tdt_component_decode(
            component_graphs=captured_components,
            model=model,
            prepared=prepared,
        )
        print("execute_done=true")
        print(f"transpiled_transcript={transpiled_decode['transcript']!r}")
        result_payload = {
            "model_id": args.model_id,
            "model_source": model_source,
            "task": task,
            "family": family,
            "inputs": _serialize_json_compatible(prepared.metadata),
            "raw_ir_nodes": raw_ir_nodes,
            "optimized_ir_nodes": optimized_ir_nodes,
            "weight_bindings": binding_count,
            "transpiled_token_ids": transpiled_decode["token_ids"],
            "transpiled_transcript": transpiled_decode["transcript"],
            "encoder_hidden_shape": transpiled_decode["encoder_hidden_shape"],
        }
        if args.skip_reference_compare:
            result_payload["reference_compare_skipped"] = True
            if artifact_dir is not None:
                _write_json(artifact_dir / "result.json", result_payload)
                print(f"saved_result={artifact_dir / 'result.json'}")
            return 0

        print("reference_begin=true")
        reference_token_ids = model.greedy_decode_token_ids(prepared.tensors[0])
        reference_transcript = model.decode_token_ids(reference_token_ids)
        print("reference_done=true")
        print(f"reference_transcript={reference_transcript!r}")
        print(f"transcript_match={reference_transcript == transpiled_decode['transcript']}")
        result_payload["reference_token_ids"] = reference_token_ids
        result_payload["reference_transcript"] = reference_transcript
        result_payload["transcript_match"] = bool(reference_transcript == transpiled_decode["transcript"])
        result_payload["token_id_match"] = bool(reference_token_ids == transpiled_decode["token_ids"])
        if artifact_dir is not None:
            _write_json(artifact_dir / "result.json", result_payload)
            print(f"saved_result={artifact_dir / 'result.json'}")
        return 0

    raise NotImplementedError(f"component pipeline execution is not implemented for task={task}")


def _parse_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[normalized]


def _to_meta_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.is_meta:
        return value
    return torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        dtype=value.dtype,
        device="meta",
        requires_grad=value.requires_grad,
    )


def _prepared_inputs_for_meta_capture(prepared: PreparedInputs) -> PreparedInputs:
    return PreparedInputs(
        names=prepared.names,
        tensors=tuple(_to_meta_tensor(tensor) for tensor in prepared.tensors),
        metadata=prepared.metadata,
    )


def _materialize_meta_rope_buffers(model: torch.nn.Module, *, dtype: torch.dtype) -> None:
    def _compute_rope(
        compute_default,
        config: object,
        *,
        layer_type: str | None = None,
        rope_init_fn=None,
        rope_type: str | None = None,
    ) -> torch.Tensor:
        fn = rope_init_fn or compute_default
        kwargs: dict[str, object] = {"device": torch.device("cpu")}
        if layer_type is not None:
            kwargs["layer_type"] = layer_type
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
        try:
            materialized, _ = fn(config, **kwargs)
        except TypeError:
            materialized, _ = fn(config=config, **kwargs)
        return materialized.to(dtype=dtype)

    for module in model.modules():
        compute_default = getattr(module, "compute_default_rope_parameters", None)
        config = getattr(module, "config", None)
        if not callable(compute_default) or config is None:
            continue
        buffers = getattr(module, "_buffers", None)
        if not isinstance(buffers, dict):
            continue
        inv_freq = buffers.get("inv_freq")
        if isinstance(inv_freq, torch.Tensor) and inv_freq.is_meta:
            materialized = _compute_rope(compute_default, config)
            buffers["inv_freq"] = materialized
            original = buffers.get("original_inv_freq")
            if isinstance(original, torch.Tensor) and original.is_meta:
                buffers["original_inv_freq"] = materialized.clone()

        rope_init_fns = getattr(module, "rope_init_fns", None)
        rope_types = getattr(module, "rope_type", None)
        layer_types = getattr(module, "layer_types", ())
        if not isinstance(layer_types, (set, tuple, list)):
            continue
        for layer_type in layer_types:
            if not isinstance(layer_type, str):
                continue
            key = f"{layer_type}_inv_freq"
            inv_freq = buffers.get(key)
            if not isinstance(inv_freq, torch.Tensor) or not inv_freq.is_meta:
                continue
            rope_init_fn = rope_init_fns.get(layer_type) if isinstance(rope_init_fns, dict) else None
            rope_type = rope_types.get(layer_type) if isinstance(rope_types, dict) else None
            materialized = _compute_rope(
                compute_default,
                config,
                layer_type=layer_type,
                rope_init_fn=rope_init_fn,
                rope_type=rope_type if isinstance(rope_type, str) else None,
            )
            buffers[key] = materialized
            original = buffers.get(f"{layer_type}_original_inv_freq")
            if isinstance(original, torch.Tensor) and original.is_meta:
                buffers[f"{layer_type}_original_inv_freq"] = materialized.clone()


def _materialize_meta_scalar_buffers(model: torch.nn.Module, *, dtype: torch.dtype) -> None:
    for module in model.modules():
        buffers = getattr(module, "_buffers", None)
        if not isinstance(buffers, dict):
            continue
        embed_scale = buffers.get("embed_scale")
        scalar_embed_scale = getattr(module, "scalar_embed_scale", None)
        if isinstance(embed_scale, torch.Tensor) and embed_scale.is_meta and scalar_embed_scale is not None:
            buffers["embed_scale"] = torch.tensor(
                float(scalar_embed_scale),
                dtype=embed_scale.dtype if embed_scale.dtype.is_floating_point else dtype,
                device="cpu",
            )


class TranspileWrapper(torch.nn.Module):
    def __init__(self, adapter_module: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__()
        self.adapter = adapter_module
        self.weights_dir = weights_dir

    def forward(self, *bound_inputs: torch.Tensor) -> torch.Tensor:
        return self.adapter(*bound_inputs)

    def get_transpile_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        provider = getattr(self.adapter, "get_transpile_metadata", None)
        if callable(provider):
            provided = provider()
            if isinstance(provided, dict):
                metadata.update(provided)
        graph_meta: dict[str, object] = {}
        base_graph = metadata.get("graph", {})
        if isinstance(base_graph, dict):
            graph_meta.update(base_graph)
        if self.weights_dir:
            graph_meta["weights_dir"] = self.weights_dir
        metadata["graph"] = graph_meta
        return metadata


def _tie_lfm2_vl_lm_head_if_needed(model: torch.nn.Module) -> str | None:
    if str(getattr(getattr(model, "config", None), "model_type", "") or "").lower() != "lfm2_vl":
        return None
    lm_head = getattr(model, "lm_head", None)
    model_root = getattr(model, "model", None)
    language_model = getattr(model_root, "language_model", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if not isinstance(lm_head, torch.nn.Linear) or not isinstance(embed_tokens, torch.nn.Embedding):
        return None
    if tuple(lm_head.weight.shape) != tuple(embed_tokens.weight.shape):
        return None
    if lm_head.weight.is_meta or embed_tokens.weight.is_meta:
        return None
    if lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr():
        return None
    lm_head.weight = embed_tokens.weight
    return "tied LFM2-VL lm_head.weight to language_model.embed_tokens.weight"


def _load_local_torch_state_dict(model_source: str) -> dict[str, torch.Tensor] | None:
    root = Path(model_source)
    if not root.exists() or not root.is_dir():
        return None

    safetensors_path = root / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file  # type: ignore

            return load_file(str(safetensors_path))
        except Exception:
            return None

    pytorch_path = root / "pytorch_model.bin"
    if pytorch_path.exists():
        try:
            loaded = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(str(pytorch_path), map_location="cpu")
        return loaded if isinstance(loaded, dict) else None

    return None


def _remap_gemma4_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if "audio_tower" in new_key:
            new_key = re.sub(r"subsample_conv_projection\.layer(\d+)\.", r"subsample_conv_projection.conv_\1.", new_key)
            new_key = new_key.replace("audio_tower.layers.", "audio_tower.conformer.")
            new_key = new_key.replace(".feed_forward1.", ".ffw_layer_start.")
            new_key = new_key.replace(".feed_forward2.", ".ffw_layer_end.")
            new_key = re.sub(r"\.self_attn\.(q_proj|k_proj|v_proj)\.", r".attention.attn.\1.", new_key)
            new_key = new_key.replace(".self_attn.per_dim_scale", ".attention.attn.per_dim_scale")
            new_key = new_key.replace(".self_attn.relative_k_proj.", ".attention.attn.relative_position_embedding.pos_proj.")
            new_key = new_key.replace(".self_attn.post.", ".attention.post.")
            new_key = new_key.replace(".norm_pre_attn.", ".attention.pre_attn_norm.")
            new_key = new_key.replace(".norm_post_attn.", ".attention.post_norm.")
            new_key = new_key.replace(".norm_out.", ".norm.")
        new_key = new_key.replace(".linear.weight", ".weight")
        remapped[new_key] = value
        if new_key.endswith(".attention.attn.per_dim_scale"):
            remapped[new_key.replace(".per_dim_scale", ".per_dim_key_scale")] = value
    if "lm_head.weight" not in remapped:
        tied_embedding = remapped.get("model.language_model.embed_tokens.weight")
        if tied_embedding is not None:
            remapped["lm_head.weight"] = tied_embedding
    return remapped


def _repair_gemma4_checkpoint_weights(model: torch.nn.Module, model_source: str) -> dict[str, object]:
    multimodal_backbone = getattr(model, "model", model)
    audio_tower = getattr(multimodal_backbone, "audio_tower", None)
    # Older/local Gemma4 variants used Cactus-style module names such as
    # audio_tower.conformer and bare clippable-linear weights. Current HF
    # Gemma4 classes use audio_tower.layers and *.linear.weight, and
    # from_pretrained already loads them correctly. Do not reload a legacy
    # remap into the current layout, because that leaves real weights missing.
    if hasattr(audio_tower, "layers"):
        return {"applied": False, "reason": "current HF Gemma4 layout uses audio_tower.layers"}
    if not hasattr(audio_tower, "conformer"):
        return {"applied": False, "reason": "current HF Gemma4 layout does not need legacy key remap"}

    raw_state_dict = _load_local_torch_state_dict(model_source)
    if raw_state_dict is None:
        return {"applied": False, "reason": "no local checkpoint state_dict"}

    remapped_state_dict = _remap_gemma4_checkpoint_state_dict(raw_state_dict)
    load_result = model.load_state_dict(remapped_state_dict, strict=False)
    return {
        "applied": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _load_config_json(model_id_or_path: str) -> dict[str, object]:
    local_snapshot = _resolve_local_snapshot(model_id_or_path)
    config_source = Path(local_snapshot) / "config.json" if local_snapshot else None
    if config_source is not None and config_source.exists():
        return json.loads(config_source.read_text())
    explicit = Path(model_id_or_path) / "config.json"
    if explicit.exists():
        return json.loads(explicit.read_text())
    return {}


def _load_optional_json(model_id_or_path: str, filename: str) -> dict[str, object]:
    local_snapshot = _resolve_local_snapshot(model_id_or_path)
    candidate = Path(local_snapshot) / filename if local_snapshot else None
    if candidate is not None and candidate.exists():
        return json.loads(candidate.read_text())
    explicit = Path(model_id_or_path) / filename
    if explicit.exists():
        return json.loads(explicit.read_text())
    return {}


def _load_gemma4_tokenizer_fallback(source_candidates: list[str]) -> object | None:
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
    except Exception:
        return None

    for source in source_candidates:
        root = Path(source)
        if not root.exists() or not root.is_dir():
            continue
        tokenizer_json = root / "tokenizer.json"
        tokenizer_config_path = root / "tokenizer_config.json"
        if not tokenizer_json.exists() or not tokenizer_config_path.exists():
            continue

        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
            mask_token=tokenizer_config.get("mask_token"),
            padding_side=str(tokenizer_config.get("padding_side", "right")),
            additional_special_tokens=list(tokenizer_config.get("extra_special_tokens", []) or []),
        )
        model_max_length = tokenizer_config.get("model_max_length")
        if isinstance(model_max_length, int):
            tokenizer.model_max_length = model_max_length

        chat_template_path = root / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

        for token_attr in (
            "image_token",
            "audio_token",
            "boi_token",
            "eoi_token",
            "boa_token",
            "eoa_token",
        ):
            token_value = tokenizer_config.get(token_attr)
            if isinstance(token_value, str):
                setattr(tokenizer, token_attr, token_value)
                setattr(tokenizer, f"{token_attr}_id", tokenizer.convert_tokens_to_ids(token_value))
        return tokenizer

    return None


def _load_gemma4_processor_fallback(
    *,
    source_candidates: list[str],
    common_kwargs: dict[str, object],
) -> object | None:
    try:
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # type: ignore
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # type: ignore
        from transformers.models.gemma4.processing_gemma4 import Gemma4Processor  # type: ignore
    except Exception:
        return None

    tokenizer = None
    processor_config: dict[str, object] = {}
    for source in source_candidates:
        try:
            try:
                from transformers import AutoTokenizer  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
            except Exception:
                tokenizer = _load_gemma4_tokenizer_fallback([source])
            processor_config = _load_optional_json(source, "processor_config.json")
            if tokenizer is not None and processor_config:
                break
        except Exception:
            continue
    if tokenizer is None or not processor_config:
        return None

    feature_config = dict(processor_config.get("feature_extractor", {}) or {})
    image_config = dict(processor_config.get("image_processor", {}) or {})
    feature_config.pop("feature_extractor_type", None)
    image_config.pop("image_processor_type", None)

    feature_extractor = Gemma4AudioFeatureExtractor(**feature_config)
    image_processor = Gemma4ImageProcessor(**image_config)

    processor_kwargs = {}
    for key in ("image_seq_length", "audio_seq_length", "audio_ms_per_token"):
        if key in processor_config:
            processor_kwargs[key] = processor_config[key]

    return Gemma4Processor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        **processor_kwargs,
    )


def _infer_task_from_config(model_id_or_path: str) -> str:
    config = _load_config_json(model_id_or_path)
    plan = infer_component_plan_from_config(config, model_id=model_id_or_path)
    if plan is not None:
        return plan.task
    architectures = [str(value) for value in config.get("architectures", []) if isinstance(value, str)]
    model_type = str(config.get("model_type", "") or "").lower()
    decoding_cfg = config.get("decoding")
    if isinstance(decoding_cfg, dict) and str(decoding_cfg.get("model_type", "") or "").lower() == "tdt":
        return "tdt_transcription"
    loss_cfg = config.get("loss")
    if isinstance(loss_cfg, dict) and str(loss_cfg.get("loss_name", "") or "").lower() == "tdt":
        return "tdt_transcription"

    if "nomic" in model_type or any("NomicBert" in value for value in architectures):
        return "text_embedding"
    if any("CausalLM" in value for value in architectures):
        return "causal_lm_logits"
    if any("CTC" in value for value in architectures):
        return "ctc_logits"
    if model_type == "whisper":
        return "seq2seq_transcription"
    if any("ConditionalGeneration" in value and "Whisper" in value for value in architectures):
        return "seq2seq_transcription"

    lowered_id = model_id_or_path.lower()
    if "nomic-embed" in lowered_id:
        return "text_embedding"
    if "parakeet-tdt" in lowered_id:
        return "tdt_transcription"
    if "whisper" in lowered_id:
        return "seq2seq_transcription"
    if "ctc" in lowered_id:
        return "ctc_logits"
    if any(token in lowered_id for token in ("qwen", "gemma", "llama", "mistral", "lfm")):
        return "causal_lm_logits"

    raise RuntimeError(
        f"Could not infer transpile task for {model_id_or_path}.\n"
        "\n"
        "Pass one explicitly with --task, for example:\n"
        "  --task causal_lm_logits\n"
        "  --task tdt_transcription\n"
        "  --task ctc_logits\n"
        "  --task encoder_hidden_states\n"
        "  --task seq2seq_transcription\n"
    )


def _resolve_audio_sample_rate(processor: object) -> int:
    for attr_name in ("feature_extractor", "tokenizer"):
        child = getattr(processor, attr_name, None)
        sample_rate = getattr(child, "sampling_rate", None)
        if isinstance(sample_rate, int) and sample_rate > 0:
            return sample_rate
    sample_rate = getattr(processor, "sampling_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return 16000


def _infer_fallback_audio_input_names(config: dict[str, object], task: str) -> tuple[str, ...]:
    model_type = str(config.get("model_type", "") or "").lower()
    if model_type == "whisper":
        return ("input_features",)
    if task == "seq2seq_transcription":
        return ("input_features",)
    if task == "encoder_hidden_states":
        return ("input_features",)
    if task == "tdt_transcription":
        return ("input_features",)
    audio_cfg = config.get("audio_config")
    if isinstance(audio_cfg, dict) and any(key in audio_cfg for key in ("features", "input_feat_size", "num_mel_bins")):
        return ("input_features", "attention_mask") if task == "ctc_logits" else ("input_features",)
    encoder_cfg = config.get("encoder")
    if isinstance(encoder_cfg, dict) and any(key in encoder_cfg for key in ("feat_in", "num_mel_bins")):
        return ("input_features", "attention_mask") if task == "ctc_logits" else ("input_features",)
    return ("input_values", "attention_mask") if task == "ctc_logits" else ("input_values",)


def _resolve_encoder_module(model: torch.nn.Module) -> torch.nn.Module | None:
    get_encoder = getattr(model, "get_encoder", None)
    if callable(get_encoder):
        encoder = get_encoder()
        if isinstance(encoder, torch.nn.Module):
            return encoder
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, torch.nn.Module):
        return encoder
    model_attr = getattr(model, "model", None)
    if model_attr is not None:
        encoder = getattr(model_attr, "encoder", None)
        if isinstance(encoder, torch.nn.Module):
            return encoder
    return None


def _infer_expected_input_feature_frames(model: torch.nn.Module) -> int | None:
    config = getattr(model, "config", None)
    max_source_positions = getattr(config, "max_source_positions", None)
    if not isinstance(max_source_positions, int) or max_source_positions <= 0:
        return None

    encoder = _resolve_encoder_module(model)
    if encoder is None:
        return None

    stride_product = 1
    found_conv = False
    for child in encoder.children():
        if isinstance(child, torch.nn.Conv1d):
            stride = child.stride[0] if isinstance(child.stride, tuple) else child.stride
            stride_product *= int(stride)
            found_conv = True

    if not found_conv:
        return None
    return int(max_source_positions) * stride_product


def _prepare_fallback_audio_inputs(
    *,
    input_names: tuple[str, ...],
    config: dict[str, object],
    preprocessor_config: dict[str, object],
    model: torch.nn.Module,
    task: str,
    audio_file: str,
    torch_dtype: torch.dtype,
) -> PreparedInputs:
    if not input_names:
        input_names = _infer_fallback_audio_input_names(config, task)
    preprocessor_root = config.get("preprocessor")
    preprocessor_root = preprocessor_root if isinstance(preprocessor_root, dict) else {}
    encoder_root = config.get("encoder")
    encoder_root = encoder_root if isinstance(encoder_root, dict) else {}

    sample_rate = int(
        preprocessor_config.get(
            "sampling_rate",
            preprocessor_root.get("sample_rate", config.get("sampling_rate", 16000)),
        )
        or 16000
    )
    num_mels = int(
        preprocessor_config.get(
            "feature_size",
            preprocessor_root.get(
                "features",
                encoder_root.get("feat_in", config.get("num_mel_bins", config.get("feature_size", 80))),
            ),
        )
        or 80
    )
    encoder_cfg = config.get("encoder_config")
    if isinstance(encoder_cfg, dict):
        num_mels = int(encoder_cfg.get("num_mel_bins", encoder_cfg.get("feat_in", num_mels)) or num_mels)
    raw_hop_length = preprocessor_config.get(
        "hop_length",
        preprocessor_root.get("window_stride", config.get("hop_length", 160)),
    )
    if raw_hop_length is None:
        raw_hop_length = 160
    if isinstance(raw_hop_length, float) and raw_hop_length > 0.0 and raw_hop_length < 1.0:
        hop_length = max(1, int(round(float(sample_rate) * raw_hop_length)))
    else:
        hop_length = int(raw_hop_length)

    n_fft = int(preprocessor_config.get("n_fft", config.get("n_fft", preprocessor_root.get("n_fft", 400))) or 400)
    raw_frame_length = preprocessor_config.get(
        "win_length",
        preprocessor_config.get(
            "frame_length",
            preprocessor_root.get("window_size", preprocessor_root.get("n_window_size", config.get("frame_length", n_fft))),
        ),
    )
    if raw_frame_length is None:
        raw_frame_length = n_fft
    if isinstance(raw_frame_length, float) and raw_frame_length > 0.0 and raw_frame_length < 1.0:
        frame_length = max(1, int(round(float(sample_rate) * raw_frame_length)))
    else:
        frame_length = int(raw_frame_length)
    preemphasis = preprocessor_config.get("preemphasis", preprocessor_root.get("preemph"))
    if preemphasis is not None:
        preemphasis = float(preemphasis)
    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)

    tensors: list[torch.Tensor] = []
    if input_names and input_names[0] == "input_features":
        model_type = str(config.get("model_type", "") or "").lower()
        is_parakeet_like = (
            task == "tdt_transcription"
            or "parakeet" in model_type
            or "parakeet" in type(model).__module__.lower()
        )
        if is_parakeet_like:
            expected_frames = _infer_expected_input_feature_frames(model)
            parakeet_features, _ = prepare_parakeet_tdt_audio_features(
                audio_file,
                expected_frames=expected_frames,
                expected_mels=num_mels,
                torch_dtype=torch_dtype,
            )
            tensors.append(parakeet_features)
            return PreparedInputs(
                names=input_names[: len(tensors)],
                tensors=tuple(tensors),
                metadata={
                    "audio_file": str(Path(audio_file).resolve()),
                    "sample_rate": sample_rate,
                    "fallback_audio_preprocessor": True,
                    "native_parakeet_audio_preprocessor": True,
                    "input_shapes": {
                        name: list(tensor.shape)
                        for name, tensor in zip(input_names, tensors)
                    },
                },
            )
        is_whisper_like = (
            str(config.get("model_type", "") or "").lower() == "whisper"
            or "transformers.models.whisper." in type(model).__module__
        )
        if is_whisper_like:
            expected_frames = _infer_expected_input_feature_frames(model) or 3000
            try:
                whisper_features, active_frames = prepare_cactus_audio_features(
                    audio_file,
                    model_type="whisper",
                    expected_frames=expected_frames,
                    expected_mels=num_mels,
                    torch_dtype=torch_dtype,
                    layout="mels_frames",
                )
                tensors.append(whisper_features)
                return PreparedInputs(
                    names=input_names[: len(tensors)],
                    tensors=tuple(tensors),
                    metadata={
                        "audio_file": str(Path(audio_file).resolve()),
                        "sample_rate": sample_rate,
                        "fallback_audio_preprocessor": True,
                        "native_cactus_audio_preprocessor": True,
                        "active_feature_frames": active_frames,
                        "input_shapes": {
                            name: list(tensor.shape)
                            for name, tensor in zip(input_names, tensors)
                        },
                    },
                )
            except Exception:
                pass
        features, feature_length = _generic_log_mel_features(
            waveform,
            sample_rate=sample_rate,
            num_mels=num_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            frame_length=frame_length,
            preemphasis=preemphasis,
        )
        expected_frames = _infer_expected_input_feature_frames(model)
        attention_mask: np.ndarray | None = None
        target_frames = feature_length
        if isinstance(expected_frames, int) and expected_frames > 0:
            target_frames = expected_frames

        active_frames = min(feature_length, target_frames)
        features = features[:active_frames, :]
        if target_frames > active_frames:
            pad_width = target_frames - active_frames
            features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
            attention_mask = np.zeros((target_frames,), dtype=np.bool_)
            attention_mask[:active_frames] = True

        if is_whisper_like:
            features = np.ascontiguousarray(features.T)

        tensors.append(torch.from_numpy(features).unsqueeze(0).to(dtype=torch_dtype))
        if attention_mask is not None and len(input_names) > 1 and input_names[1] == "attention_mask":
            tensors.append(torch.from_numpy(attention_mask).unsqueeze(0))
    else:
        input_values = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch_dtype)
        tensors.append(input_values)
        if len(input_names) > 1 and input_names[1] == "attention_mask":
            tensors.append(torch.ones_like(input_values, dtype=torch.float32))

    return PreparedInputs(
        names=input_names[: len(tensors)],
        tensors=tuple(tensors),
        metadata={
            "audio_file": str(Path(audio_file).resolve()),
            "sample_rate": sample_rate,
            "fallback_audio_preprocessor": True,
            "input_shapes": {
                name: list(tensor.shape)
                for name, tensor in zip(input_names, tensors)
            },
        },
    )


def _prepare_audio_inputs(
    processor: object | None,
    *,
    input_names: tuple[str, ...],
    config: dict[str, object],
    preprocessor_config: dict[str, object],
    model: torch.nn.Module,
    task: str,
    audio_file: str,
    torch_dtype: torch.dtype,
) -> PreparedInputs:
    if processor is None:
        return _prepare_fallback_audio_inputs(
            input_names=input_names,
            config=config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )

    model_type = str(config.get("model_type", "") or "").lower()
    if model_type == "whisper" or "transformers.models.whisper." in type(model).__module__:
        return _prepare_fallback_audio_inputs(
            input_names=input_names,
            config=config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )

    sample_rate = _resolve_audio_sample_rate(processor)
    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)
    batch = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    preferred_keys = tuple(input_names) + tuple(
        key for key in ("input_features", "input_values", "attention_mask") if key not in input_names
    )
    tensor_keys = [key for key, value in batch.items() if isinstance(value, torch.Tensor)]
    ordered_keys = [key for key in preferred_keys if key in tensor_keys]
    ordered_keys.extend(key for key in tensor_keys if key not in ordered_keys)
    if not ordered_keys:
        raise RuntimeError(f"processor did not return tensor inputs for audio file: {audio_file}")

    tensors: list[torch.Tensor] = []
    for key in ordered_keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        tensors.append(value)

    return PreparedInputs(
        names=tuple(ordered_keys[: len(tensors)]),
        tensors=tuple(tensors),
        metadata={
            "audio_file": str(Path(audio_file).resolve()),
            "sample_rate": sample_rate,
            "fallback_audio_preprocessor": False,
            "input_shapes": {
                name: list(tensor.shape)
                for name, tensor in zip(ordered_keys, tensors)
            },
        },
    )


def _contains_token_subsequence(token_ids: list[int], subsequence: list[int]) -> bool:
    if not subsequence:
        return False
    width = len(subsequence)
    return any(token_ids[index : index + width] == subsequence for index in range(len(token_ids) - width + 1))


def _resolve_whisper_decoder_prompt_token_ids(
    tokenizer_or_processor: object,
    *,
    prompt: str | None,
    decoder_start_token_id: int | None,
    forced_decoder_ids: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
) -> list[int]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        encode = None

    normalized_prompt = (prompt or "").strip()
    if normalized_prompt == _DEFAULT_CAUSAL_PROMPT:
        normalized_prompt = ""

    prompt_token_ids: list[int] = []
    if decoder_start_token_id is not None:
        prompt_token_ids.append(int(decoder_start_token_id))

    normalized_forced_ids: list[tuple[int, int]] = []
    if forced_decoder_ids is not None:
        for item in forced_decoder_ids:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], int)
                and isinstance(item[1], int)
            ):
                normalized_forced_ids.append((int(item[0]), int(item[1])))
        normalized_forced_ids.sort(key=lambda pair: pair[0])

    next_position = 1
    for position, token_id in normalized_forced_ids:
        if position != next_position:
            break
        prompt_token_ids.append(int(token_id))
        next_position += 1

    if normalized_prompt:
        if encode is None:
            raise RuntimeError("Whisper transpile requires a tokenizer to encode a non-empty prompt")
        try:
            encoded_prompt = encode(normalized_prompt, add_special_tokens=False)
        except TypeError:
            encoded_prompt = encode(normalized_prompt)
        prompt_token_ids.extend(int(value) for value in encoded_prompt)

    if not prompt_token_ids:
        if encode is None:
            raise RuntimeError("Whisper transpile requires a tokenizer or decoder_start_token_id")
        try:
            prompt_token_ids = [int(value) for value in encode("<|startoftranscript|>", add_special_tokens=False)]
        except TypeError:
            prompt_token_ids = [int(value) for value in encode("<|startoftranscript|>")]
        if not prompt_token_ids and decoder_start_token_id is not None:
            prompt_token_ids = [int(decoder_start_token_id)]

    return prompt_token_ids


def _augment_whisper_seq2seq_metadata(
    prepared: PreparedInputs,
    *,
    tokenizer_or_processor: object,
    model: torch.nn.Module,
    prompt: str | None,
    max_new_tokens: int,
) -> PreparedInputs:
    config = getattr(model, "config", None)
    decoder_start_token_id = getattr(config, "decoder_start_token_id", None)
    eos_token_id = getattr(config, "eos_token_id", None)
    pad_token_id = getattr(config, "pad_token_id", None)
    max_target_positions = int(getattr(config, "max_target_positions", 0) or 0)
    suppress_tokens = [int(value) for value in (getattr(config, "suppress_tokens", None) or [])]
    begin_suppress_tokens = [int(value) for value in (getattr(config, "begin_suppress_tokens", None) or [])]

    decoder_input_ids = _resolve_whisper_decoder_prompt_token_ids(
        tokenizer_or_processor,
        prompt=prompt,
        decoder_start_token_id=int(decoder_start_token_id) if isinstance(decoder_start_token_id, int) else None,
        forced_decoder_ids=getattr(config, "forced_decoder_ids", None),
    )
    if not decoder_input_ids:
        raise RuntimeError("Whisper transpile could not resolve decoder prompt token ids")

    target_token_count = len(decoder_input_ids) + max(1, int(max_new_tokens))
    if max_target_positions > 0:
        target_token_count = min(target_token_count, max_target_positions)
    target_token_count = max(target_token_count, len(decoder_input_ids))

    metadata = dict(prepared.metadata)
    metadata.update(
        {
            "decoder_input_ids": [int(value) for value in decoder_input_ids],
            "decoder_start_token_id": None if decoder_start_token_id is None else int(decoder_start_token_id),
            "eos_token_id": None if eos_token_id is None else int(eos_token_id),
            "pad_token_id": None if pad_token_id is None else int(pad_token_id),
            "target_token_count": int(target_token_count),
            "max_target_positions": int(max_target_positions),
            "suppress_tokens": suppress_tokens,
            "begin_suppress_tokens": begin_suppress_tokens,
        }
    )
    return PreparedInputs(
        names=prepared.names,
        tensors=prepared.tensors,
        metadata=metadata,
    )


def _tokenize_text_prompt(
    tokenizer: object,
    prompt: str,
    *,
    enable_thinking_if_supported: bool = False,
) -> torch.Tensor:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            encoded = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=bool(enable_thinking_if_supported),
            )
            if isinstance(encoded, torch.Tensor) and encoded.ndim == 2:
                return encoded.to(dtype=torch.long)
            if isinstance(encoded, Mapping):
                input_ids = encoded.get("input_ids")
                if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                    return input_ids.to(dtype=torch.long)
        except Exception:
            pass
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(dtype=torch.long)


def _resolve_text_padding_token_id(tokenizer: object | None) -> int:
    for attr_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr_name, None) if tokenizer is not None else None
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)
    return 0


def _resolve_graph_safe_text_padding_token_id(
    tokenizer: object | None,
    prompt_input_ids: torch.Tensor,
) -> int:
    # Keep the tokenizer pad ID in the captured graph metadata. The lowerer keeps
    # large token-id comparisons in FP32, so Qwen-style high pad IDs are safe and
    # runtime padding stays consistent with the traced mask.
    return _resolve_text_padding_token_id(tokenizer)


def _prepare_text_inputs(
    tokenizer: object,
    *,
    prompt: str,
    input_ids_text: str | None,
    max_new_tokens: int,
    enable_thinking_if_supported: bool = False,
) -> PreparedInputs:
    if input_ids_text:
        token_ids = [int(part.strip()) for part in input_ids_text.split(",") if part.strip()]
        if not token_ids:
            raise ValueError("--input-ids was provided but no ids were parsed")
        prompt_input_ids = torch.tensor([token_ids], dtype=torch.long)
    else:
        prompt_input_ids = _tokenize_text_prompt(
            tokenizer,
            prompt,
            enable_thinking_if_supported=enable_thinking_if_supported,
        )
    if prompt_input_ids.ndim != 2 or int(prompt_input_ids.shape[0]) != 1:
        raise ValueError(
            "causal_lm_logits transpile currently expects prompt input ids with shape [1, T], "
            f"got {tuple(int(dim) for dim in prompt_input_ids.shape)}"
        )
    if int(max_new_tokens) < 0:
        raise ValueError("--max-new-tokens must be non-negative")

    prompt_token_count = int(prompt_input_ids.shape[1])
    target_token_count = prompt_token_count + int(max_new_tokens)
    padding_token_id = _resolve_graph_safe_text_padding_token_id(tokenizer, prompt_input_ids)
    if target_token_count > prompt_token_count:
        input_ids = torch.full((1, target_token_count), padding_token_id, dtype=torch.long)
        input_ids[:, :prompt_token_count] = prompt_input_ids
    else:
        input_ids = prompt_input_ids
    return PreparedInputs(
        names=("input_ids",),
        tensors=(input_ids,),
        metadata={
            "prompt": prompt,
            "prompt_input_ids": prompt_input_ids.tolist(),
            "input_ids": input_ids.tolist(),
            "prompt_token_count": prompt_token_count,
            "target_token_count": target_token_count,
            "max_new_tokens": int(max_new_tokens),
            "padding_token_id": int(padding_token_id),
            "enable_thinking": bool(enable_thinking_if_supported),
        },
    )


_NOMIC_EMBED_SEQ_LEN = 256


def _prepare_text_embedding_inputs(
    tokenizer: object,
    *,
    prompt: str,
    input_ids_text: str | None,
    seq_len: int = _NOMIC_EMBED_SEQ_LEN,
) -> PreparedInputs:
    base = _prepare_text_inputs(
        tokenizer,
        prompt=prompt,
        input_ids_text=input_ids_text,
        max_new_tokens=0,
    )
    prompt_input_ids = base.tensors[0]
    prompt_token_count = int(prompt_input_ids.shape[1])
    padding_token_id = int(base.metadata.get("padding_token_id", 0))
    target = max(1, int(seq_len))
    input_ids = torch.full((1, target), padding_token_id, dtype=torch.long)
    attention_mask = torch.zeros((1, target), dtype=torch.long)
    keep = min(prompt_token_count, target)
    input_ids[:, :keep] = prompt_input_ids[:, :keep]
    attention_mask[:, :keep] = 1
    return PreparedInputs(
        names=("input_ids", "attention_mask"),
        tensors=(input_ids, attention_mask),
        metadata={
            "prompt": prompt,
            "prompt_token_count": prompt_token_count,
            "target_token_count": target,
            "padding_token_id": padding_token_id,
        },
    )


def _add_multimodal_generation_headroom(
    prepared: PreparedInputs,
    *,
    tokenizer: object | None,
    max_new_tokens: int,
    min_context_tokens: int | None = None,
) -> PreparedInputs:
    requested = max(0, int(max_new_tokens))
    if "input_ids" not in prepared.names:
        return prepared

    tensor_by_name = dict(zip(prepared.names, prepared.tensors, strict=True))
    input_ids = tensor_by_name.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return prepared

    prompt_token_count = int(input_ids.shape[1])
    min_context = max(0, int(min_context_tokens or 0))
    target_token_count = max(prompt_token_count + requested, min_context)
    if target_token_count <= prompt_token_count:
        return prepared
    padding_token_id = _resolve_graph_safe_text_padding_token_id(tokenizer, input_ids)

    padded_tensors: list[torch.Tensor] = []
    for name, tensor in zip(prepared.names, prepared.tensors, strict=True):
        if (
            name == "position_ids"
            and tensor.ndim == 3
            and int(tensor.shape[1]) == 1
            and int(tensor.shape[2]) == prompt_token_count
        ):
            padded = torch.zeros(
                (int(tensor.shape[0]), 1, target_token_count),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded[:, :, :prompt_token_count] = tensor
            padded_tensors.append(padded)
            continue
        if (
            name not in {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
            or tensor.ndim != 2
            or int(tensor.shape[0]) != 1
            or int(tensor.shape[1]) != prompt_token_count
        ):
            padded_tensors.append(tensor)
            continue
        pad_value = int(padding_token_id) if name == "input_ids" else 0
        padded = torch.full(
            (1, target_token_count),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, :prompt_token_count] = tensor
        padded_tensors.append(padded)

    metadata = dict(prepared.metadata)
    metadata["prompt_token_count"] = prompt_token_count
    metadata["target_token_count"] = target_token_count
    metadata["max_new_tokens"] = max(requested, target_token_count - prompt_token_count)
    metadata["padding_token_id"] = int(padding_token_id)
    input_shapes = dict(metadata.get("input_shapes") or {})
    for name, tensor in zip(prepared.names, padded_tensors, strict=True):
        input_shapes[name] = [int(dim) for dim in tensor.shape]
    metadata["input_shapes"] = input_shapes
    return PreparedInputs(
        names=prepared.names,
        tensors=tuple(padded_tensors),
        metadata=metadata,
    )


def _multimodal_context_token_floor(model_type: str = "") -> int:
    default_context = multimodal_context_tokens_for_model_type(
        model_type,
        _DEFAULT_MULTIMODAL_CONTEXT_TOKENS,
    )
    raw = os.environ.get(
        "CACTUS_TRANSPILER_MULTIMODAL_CONTEXT_TOKENS",
        str(default_context),
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default_context


_LFM2_VL_MULTIMODAL_INPUT_ORDER = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "spatial_shapes",
    "pixel_attention_mask",
)

_QWEN3_5_MULTIMODAL_INPUT_ORDER = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "image_grid_thw",
)


def _load_image_inputs(image_files: tuple[str, ...]) -> list[object]:
    if not image_files:
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for --image-file: {exc}") from exc

    images: list[object] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        if not path.exists():
            raise RuntimeError(f"image_file does not exist: {path}")
        with Image.open(path) as image:
            images.append(resize_static_image(image.convert("RGB")).copy())
    return images


def _prepare_gemma4_multimodal_inputs(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
    use_gemma4_chat_template: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("multimodal Gemma4 transpile requires an AutoProcessor with image and audio support")
    shared = _shared_prepare_gemma4_multimodal_inputs(
        processor,
        prompt=prompt,
        image_files=image_files,
        audio_file=audio_file,
        torch_dtype=torch_dtype,
        system_prompt=system_prompt,
        enable_thinking_if_supported=enable_thinking_if_supported,
        use_gemma4_chat_template=use_gemma4_chat_template,
    )
    return PreparedInputs(
        names=shared.names,
        tensors=shared.tensors,
        metadata=shared.metadata,
    )


def _prepare_lfm2_vl_multimodal_inputs(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("LFM2-VL multimodal transpile requires an AutoProcessor with image support")
    images = _load_image_inputs(image_files)
    if not images:
        raise RuntimeError("LFM2-VL multimodal transpile requires at least one --image-file")

    user_content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": prompt.strip()})

    messages: list[dict[str, object]] = []
    normalized_system = system_prompt.strip()
    if normalized_system:
        messages.append({"role": "system", "content": normalized_system})
    messages.append({"role": "user", "content": user_content})

    batch: Mapping[str, object]
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    thinking_prefix = "<think>\n" if enable_thinking_if_supported else "<think>\n\n</think>\n\n"
    image_placeholders = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in images)
    fallback_text = (
        f"<|im_start|>user\n{image_placeholders}{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{thinking_prefix}"
    )
    if callable(apply_chat_template):
        try:
            batch = apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise
            batch = processor(text=fallback_text, images=images, return_tensors="pt")
    else:
        batch = processor(text=fallback_text, images=images, return_tensors="pt")

    tensors: list[torch.Tensor] = []
    names: list[str] = []
    input_shapes: dict[str, list[int]] = {}
    for key in _LFM2_VL_MULTIMODAL_INPUT_ORDER:
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"LFM2-VL processor did not return required tensor input: {key}")
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        elif key == "pixel_attention_mask":
            value = value.to(dtype=torch.int64)
        else:
            value = value.to(dtype=torch.long)
        names.append(key)
        tensors.append(value)
        input_shapes[key] = [int(dim) for dim in value.shape]

    return PreparedInputs(
        names=tuple(names),
        tensors=tuple(tensors),
        metadata={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "image_files": [str(Path(path).resolve()) for path in image_files],
            "input_shapes": input_shapes,
            "enable_thinking": bool(enable_thinking_if_supported),
        },
    )


def _compute_qwen3_5_position_ids(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    position_ids = torch.zeros(
        3,
        int(input_ids.shape[0]),
        int(input_ids.shape[1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_grid_iter = iter(image_grid_thw)
    for batch_idx in range(int(input_ids.shape[0])):
        token_types = mm_token_type_ids[batch_idx]
        valid_mask = attention_mask[batch_idx].bool() if attention_mask is not None else None
        if valid_mask is not None:
            token_types = token_types[valid_mask]

        current_pos = 0
        position_parts: list[torch.Tensor] = []
        for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda item: int(item[1])):
            group_items = list(group)
            token_count = group_items[-1][0] - group_items[0][0] + 1
            if int(modality_type) == 0:
                part = torch.arange(token_count, dtype=input_ids.dtype, device=input_ids.device)
                position_parts.append(part.view(1, -1).expand(3, -1) + current_pos)
                current_pos += token_count
                continue

            grid_thw = next(image_grid_iter)
            grid_t = int(grid_thw[0].item())
            grid_h = int(grid_thw[1].item())
            grid_w = int(grid_thw[2].item())
            llm_grid_t = grid_t
            llm_grid_h = grid_h // int(spatial_merge_size)
            llm_grid_w = grid_w // int(spatial_merge_size)
            image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w
            position_width = torch.arange(
                current_pos,
                current_pos + llm_grid_w,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(llm_grid_h * llm_grid_t)
            position_height = torch.arange(
                current_pos,
                current_pos + llm_grid_h,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat_interleave(llm_grid_w * llm_grid_t)
            position_temporal = torch.full(
                (image_seq_length,),
                current_pos,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            position_parts.append(torch.stack([position_temporal, position_height, position_width], dim=0))
            current_pos += max(grid_h, grid_w) // int(spatial_merge_size)

        if not position_parts:
            continue
        positions = torch.cat(position_parts, dim=1)
        if valid_mask is not None:
            position_ids[:, batch_idx, valid_mask] = positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx, : positions.shape[1]] = positions.to(position_ids.device)
    return position_ids


def _prepare_qwen3_5_multimodal_inputs(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("Qwen3.5 multimodal transpile requires an AutoProcessor with image support")
    images = _load_image_inputs(image_files)
    if not images:
        raise RuntimeError("Qwen3.5 multimodal transpile requires at least one --image-file")

    user_content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": prompt.strip()})

    messages: list[dict[str, object]] = []
    normalized_system = system_prompt.strip()
    if normalized_system:
        messages.append({"role": "system", "content": normalized_system})
    messages.append({"role": "user", "content": user_content})

    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_chat_template):
        batch = apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        batch = processor(text=prompt, images=images, return_tensors="pt")

    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    attention_mask = batch.get("attention_mask") if hasattr(batch, "get") else None
    mm_token_type_ids = batch.get("mm_token_type_ids") if hasattr(batch, "get") else None
    image_grid_thw = batch.get("image_grid_thw") if hasattr(batch, "get") else None
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, mm_token_type_ids, image_grid_thw)):
        raise RuntimeError("Qwen3.5 processor did not return required multimodal text/grid tensors")
    spatial_merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2)
    position_ids = _compute_qwen3_5_position_ids(
        input_ids.to(dtype=torch.long),
        mm_token_type_ids.to(dtype=torch.long),
        image_grid_thw.to(dtype=torch.long),
        attention_mask.to(dtype=torch.long),
        spatial_merge_size=spatial_merge_size,
    )
    batch["position_ids"] = position_ids

    tensors: list[torch.Tensor] = []
    names: list[str] = []
    input_shapes: dict[str, list[int]] = {}
    for key in _QWEN3_5_MULTIMODAL_INPUT_ORDER:
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Qwen3.5 processor did not return required tensor input: {key}")
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        else:
            value = value.to(dtype=torch.long)
        names.append(key)
        tensors.append(value)
        input_shapes[key] = [int(dim) for dim in value.shape]

    return PreparedInputs(
        names=tuple(names),
        tensors=tuple(tensors),
        metadata={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "image_files": [str(Path(path).resolve()) for path in image_files],
            "input_shapes": input_shapes,
            "enable_thinking": bool(enable_thinking_if_supported),
            "spatial_merge_size": spatial_merge_size,
        },
    )


def _load_model_source(model_id: str, *, local_files_only: bool, require_weights: bool = True) -> str:
    local_snapshot = _resolve_local_snapshot(model_id)
    if local_snapshot and _snapshot_has_model_weights(local_snapshot):
        return local_snapshot
    if local_snapshot and not require_weights:
        return local_snapshot
    if local_snapshot and local_files_only:
        raise RuntimeError(
            f"Found local snapshot for {model_id}, but it is incomplete and has no model weights:\n"
            f"  {local_snapshot}\n"
            "\n"
            "Re-run without --local-files-only to let transformers download the missing weights."
        )
    return model_id


def _load_transformers_bundle(
    *,
    model_id: str,
    task: str,
    torch_dtype: torch.dtype,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
    low_memory_load: bool,
):
    config = _load_config_json(model_id)
    config_model_type = str(config.get("model_type", "") or "").lower()
    if not config_model_type:
        normalized_model_id = model_id.lower().replace("_", "-")
        if "qwen3-vl" in normalized_model_id:
            config_model_type = "qwen3_vl"
        elif "lfm2.5-vl" in normalized_model_id or "lfm2-vl" in normalized_model_id:
            config_model_type = "lfm2_vl"
        if config_model_type:
            config = {"model_type": config_model_type}
    external_transformers_site_packages = _ensure_transformers_supports_model_type(config_model_type)
    patch_note = _patch_transformers_torchvision_probe()
    if patch_note:
        print(f"note={patch_note}")
    flex_patch_note = _patch_torch_flex_attention_compat()
    if flex_patch_note:
        print(f"note={flex_patch_note}")
    if external_transformers_site_packages:
        print(f"note=using external transformers install for {config_model_type}: {external_transformers_site_packages}")

    try:
        from transformers import AutoFeatureExtractor  # type: ignore
        from transformers import AutoConfig  # type: ignore
        from transformers import AutoModel  # type: ignore
        from transformers import AutoModelForCTC  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoModelForImageTextToText  # type: ignore
        from transformers import AutoModelForSeq2SeqLM  # type: ignore
        from transformers import AutoModelForSpeechSeq2Seq  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers is not available: {exc}") from exc

    model_source = _load_model_source(
        model_id,
        local_files_only=local_files_only,
        require_weights=not low_memory_load,
    )
    source_candidates = []
    for candidate in (model_source, model_id):
        if candidate not in source_candidates:
            source_candidates.append(candidate)
    common_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if token:
        common_kwargs["token"] = token

    hf_config = None
    if low_memory_load:
        if task == "tdt_transcription":
            raise RuntimeError("--low-memory-load is not supported for TDT transcription models")
        try:
            hf_config = AutoConfig.from_pretrained(model_source, **common_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"--low-memory-load requires loading the HF config without checkpoint weights: {exc}"
            ) from exc

    def _from_config(loader, *, trust_remote_code_override: bool | None = None):
        if hf_config is None:
            raise RuntimeError("internal error: low-memory config was not loaded")
        with torch.device("meta"):
            try:
                model = loader.from_config(
                    hf_config,
                    trust_remote_code=(
                        trust_remote_code
                        if trust_remote_code_override is None
                        else bool(trust_remote_code_override)
                    ),
                )
            except TypeError:
                model = loader.from_config(hf_config)
        model.to(dtype=torch_dtype)
        _materialize_meta_rope_buffers(model, dtype=torch_dtype)
        _materialize_meta_scalar_buffers(model, dtype=torch_dtype)
        return model.eval()

    if task == "tdt_transcription":
        model = load_tdt_local_model(model_source, torch_dtype=torch_dtype).eval()
        return model_source, None, model, config

    if task == "text_embedding":
        # nomic-bert weights are produced from the model's remote modeling code
        # (fused Wqkv / experts.mlp.w1,w2); the transformers-native nomic_bert has a
        # different architecture, so the remote code is required to match the bundle.
        remote_kwargs = {**common_kwargs, "trust_remote_code": True}
        tokenizer = None
        tokenizer_errors: list[str] = []
        for source in source_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, **remote_kwargs)
                break
            except Exception as exc:
                tokenizer_errors.append(f"{source}: {exc}")
        if tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for {model_id}:\n"
                + "\n".join(tokenizer_errors)
            )
        if low_memory_load:
            model = _from_config(AutoModel, trust_remote_code_override=True)
        else:
            model = AutoModel.from_pretrained(
                model_source,
                dtype=torch_dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                **remote_kwargs,
            ).eval()
        return model_source, tokenizer, model, config

    if task == "causal_lm_logits":
        tokenizer = None
        tokenizer_errors: list[str] = []
        for source in source_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                tokenizer_errors.append(f"{source}: {exc}")
        if tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for {model_id}:\n"
                + "\n".join(tokenizer_errors)
            )
        if config_model_type in {"lfm2_vl", "qwen3_5", "qwen3_vl"}:
            if low_memory_load:
                model = _from_config(AutoModelForImageTextToText)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
            tie_note = _tie_lfm2_vl_lm_head_if_needed(model)
            if tie_note:
                print(f"note={tie_note}")
        elif config_model_type in {"qwen3_5", "qwen3_vl"} and isinstance(config.get("text_config"), dict):
            if low_memory_load:
                model = _from_config(AutoModelForImageTextToText)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
        else:
            if low_memory_load:
                model = _from_config(AutoModelForCausalLM)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
        return model_source, tokenizer, model, config
    if task == "multimodal_causal_lm_logits":
        processor = None
        processor_errors: list[str] = []
        for source in source_candidates:
            try:
                processor = AutoProcessor.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                processor_errors.append(f"{source}: {exc}")
        if processor is None and config_model_type == "gemma4":
            processor = _load_gemma4_processor_fallback(
                source_candidates=source_candidates,
                common_kwargs=common_kwargs,
            )
            if processor is not None:
                print("note=using manual gemma4 processor fallback")
        if processor is None:
            processor_config_hint = ""
            if config_model_type == "gemma4":
                processor_config_hint = (
                    "\n"
                    "Gemma4 multimodal transpile needs a processor bundle, not just tokenizer/model weights.\n"
                    "Your local snapshot may be missing files such as `processor_config.json` or modality-specific\n"
                    "preprocessor configs. Use an official Gemma4 snapshot that includes the processor, or let\n"
                    "transformers download one by re-running without `--local-files-only`.\n"
                )
            raise RuntimeError(
                f"Could not load processor for {model_id}:\n"
                + "\n".join(processor_errors)
                + processor_config_hint
            )

        if config_model_type in {"lfm2_vl", "qwen3_5", "qwen3_vl"}:
            if low_memory_load:
                model = _from_config(AutoModelForImageTextToText)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
            tie_note = _tie_lfm2_vl_lm_head_if_needed(model)
            if tie_note:
                print(f"note={tie_note}")
        else:
            if low_memory_load:
                model = _from_config(AutoModelForCausalLM)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
        if config_model_type == "gemma4" and not low_memory_load:
            repair_result = _repair_gemma4_checkpoint_weights(model, model_source)
            if repair_result.get("applied"):
                missing = repair_result.get("missing_keys", [])
                unexpected = repair_result.get("unexpected_keys", [])
                print(
                    "note=applied gemma4 checkpoint key remap"
                    f" missing_after={len(missing)} unexpected_after={len(unexpected)}"
                )
        return model_source, processor, model, config

    processor = None
    processor_errors: list[str] = []
    missing_optional_audio_dep: str | None = None
    for source in source_candidates:
        for loader in (AutoProcessor, AutoFeatureExtractor):
            try:
                processor = loader.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                processor_errors.append(f"{loader.__name__}@{source}: {exc}")
                if isinstance(exc, ImportError) and "requires the librosa library" in str(exc):
                    missing_optional_audio_dep = "librosa"
                    break
        if processor is not None:
            break
        if missing_optional_audio_dep is not None:
            break
    if processor is None:
        if missing_optional_audio_dep == "librosa":
            print("note=falling back to built-in audio preprocessing because the HF feature extractor requires librosa")
        else:
            print("note=falling back to built-in audio preprocessing because no HF processor/feature extractor was available")

    if task == "ctc_logits":
        model_loaders = (AutoModelForCTC, AutoModel)
    elif task in {"encoder_hidden_states", "seq2seq_transcription"}:
        model_loaders = (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel)
    else:
        raise NotImplementedError(f"unsupported generic HF task: {task}")

    load_errors: list[str] = []
    for loader in model_loaders:
        try:
            if low_memory_load:
                model = _from_config(loader)
            else:
                model = loader.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                ).eval()
            return model_source, processor, model, config
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")

    raise RuntimeError(
        f"Could not load model for task={task} from {model_source}.\n"
        "\n".join(load_errors)
    )


def _load_optional_tokenizer(
    *,
    model_id: str,
    model_source: str,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None

    source_candidates = []
    for candidate in (model_source, model_id):
        if candidate not in source_candidates:
            source_candidates.append(candidate)

    common_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if token:
        common_kwargs["token"] = token

    for source in source_candidates:
        try:
            return AutoTokenizer.from_pretrained(source, **common_kwargs)
        except Exception:
            continue
    return None


def _ctc_greedy_decode_token_ids(logits: np.ndarray, *, blank_token_id: int | None) -> list[int]:
    if logits.ndim != 3 or logits.shape[0] < 1:
        raise ValueError(f"expected CTC logits with shape [batch, time, vocab], got {list(logits.shape)}")

    raw_ids = np.argmax(logits[0], axis=-1).tolist()
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in raw_ids:
        if token_id != previous:
            collapsed.append(int(token_id))
        previous = int(token_id)

    if blank_token_id is None:
        return collapsed
    return [token_id for token_id in collapsed if int(token_id) != int(blank_token_id)]


def _decode_token_ids(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError(f"tokenizer does not expose decode(): {type(tokenizer).__name__}")
    try:
        return str(decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(decode(token_ids))


def _count_weight_bindings(ir_graph: IRGraph) -> int:
    count = 0
    for value in ir_graph.values.values():
        if isinstance(value.meta, dict) and isinstance(value.meta.get("path"), str):
            count += 1
    return count


def _lower_preoptimized_ir(ir: IRGraph) -> TranspiledGraph:
    verify_ir(ir)
    graph = Graph()
    graph._transpile_materialized_constants = []  # type: ignore[attr-defined]
    env: dict[str, Any] = {}
    runtime_inputs = []
    bound_constants = []
    bound_constant_bindings = []
    bound_constant_value_ids: dict[int, str] = {}

    for value_id in ir.inputs:
        value = ir.values[value_id]
        tensor = _lower_input_value(graph, value)
        env[value_id] = tensor
        runtime_inputs.append(tensor)

    for value_id, const in ir.constants.items():
        value = ir.values[value_id]
        binding = _lookup_weight_binding(value)
        if binding is not None:
            binding = ensure_binding_compatible(binding, source_tensor=const)
        lowered_const = _lower_constant_value(graph, value, const, binding=binding)
        env[value_id] = lowered_const
        if hasattr(lowered_const, "g") and hasattr(lowered_const, "id"):
            bound_constants.append(lowered_const)
            bound_constant_value_ids[int(lowered_const.id)] = str(value_id)
            if binding is not None:
                bound_constant_bindings.append(
                    {
                        "node_id": int(lowered_const.id),
                        "value_id": str(value_id),
                        "path": binding.path,
                        "kind": binding.kind,
                        "source_name": binding.source_name,
                    }
                )

    for node_id in ir.order:
        node = ir.nodes[node_id]
        outputs = _lower_ir_node(graph, node, env, ir)
        if len(outputs) != len(node.outputs):
            raise ValueError(
                f"node {node.id} produced {len(outputs)} outputs, expected {len(node.outputs)}"
            )
        for output_id, tensor in zip(node.outputs, outputs):
            env[output_id] = tensor

    outputs = [env[value_id] for value_id in ir.outputs]
    seen_bound_constant_ids = {int(tensor.id) for tensor in bound_constants}
    for tensor in getattr(graph, "_transpile_materialized_constants", []):
        tensor_id = int(tensor.id)
        if tensor_id in seen_bound_constant_ids:
            continue
        bound_constants.append(tensor)
        seen_bound_constant_ids.add(tensor_id)
    return TranspiledGraph(
        graph=graph,
        runtime_inputs=runtime_inputs,
        bound_constants=bound_constants,
        bound_constant_bindings=bound_constant_bindings,
        bound_constant_value_ids=bound_constant_value_ids,
        outputs=outputs,
    )


def main() -> int:
    from cactus.models.needle import register_with_transformers

    register_with_transformers()
    parser = argparse.ArgumentParser(
        description=(
            "Load a Hugging Face model, canonicalize it into a generic transpile task, "
            "capture it with the Cactus transpiler, lower it to a Cactus Graph, and "
            "optionally save artifacts or run the lowered graph."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id or local snapshot path.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=(
            "auto",
            "causal_lm_logits",
            "multimodal_causal_lm_logits",
            "ctc_logits",
            "encoder_hidden_states",
            "seq2seq_transcription",
            "tdt_transcription",
            "text_embedding",
        ),
        help="Transpile task. Use auto to infer from config/model id.",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_CAUSAL_PROMPT,
        help="Prompt used for causal_lm_logits or multimodal_causal_lm_logits when --input-ids is not set.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt used for Gemma4 multimodal chat-style prompt construction.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model-specific thinking markers when the multimodal prompt format supports them.",
    )
    parser.add_argument(
        "--input-ids",
        default="",
        help="Optional comma-separated token ids for causal_lm_logits.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help=(
            "For causal_lm_logits bundles, preallocate prompt-plus-generation context "
            "so the saved bundle can do greedy autoregressive decoding."
        ),
    )
    parser.add_argument(
        "--cache-context-length",
        default="auto",
        help="KV cache context length for cached decode graphs. Use auto to read the model config.",
    )
    parser.add_argument(
        "--audio-file",
        default="",
        help="Path to a WAV file for audio or multimodal tasks.",
    )
    parser.add_argument(
        "--image-file",
        action="append",
        default=[],
        help="Path to an image file for multimodal tasks. Repeat to pass multiple images.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype for model loading: float16, float32, or bfloat16.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers loaders.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the model/processor to already exist locally.",
    )
    parser.add_argument(
        "--low-memory-load",
        action="store_true",
        help=(
            "Instantiate the HF model on the meta device for graph capture instead of "
            "loading checkpoint tensors into RAM. Requires converted CQ weights and "
            "only supports bundle generation, not execution."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Converted Cactus CQ weights directory for mmap weight binding.",
    )
    parser.add_argument(
        "--allow-unconverted-weights",
        action="store_true",
        help="Debug only: allow transpiling without converted Cactus CQ weights.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Directory where raw_ir.json, optimized_ir.json, graph.cactus, component graphs, and result.json are saved. Defaults to ./transpiled/<model-id>/.",
    )
    parser.add_argument(
        "--graph-filename",
        default="graph.cactus",
        help="Filename to use for Graph.save() inside --artifact-dir.",
    )
    parser.add_argument(
        "--component-pipeline",
        default="auto",
        choices=("auto", "on", "off"),
        help="Use a split component pipeline when the model adapter supports it.",
    )
    parser.add_argument(
        "--components",
        default=None,
        help=(
            "Optional comma-separated component subset for component-pipeline models "
            "(for example: vision_encoder,audio_encoder,lm_encoder,decoder)."
        ),
    )
    parser.add_argument("--no-fuse-gated-deltanet", action="store_true")
    parser.add_argument("--no-fuse-rms-norm", action="store_true")
    parser.add_argument("--no-fuse-rope", action="store_true")
    parser.add_argument("--no-fuse-attention", action="store_true")
    parser.add_argument("--no-fuse-attention-block", action="store_true")
    parser.add_argument("--no-fuse-add-clipped", action="store_true")
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        help="Stop after lowering instead of running the transpiled graph.",
    )
    parser.add_argument(
        "--skip-reference-compare",
        action="store_true",
        help="Run the transpiled graph but skip the follow-up PyTorch reference pass.",
    )
    args = parser.parse_args()

    validated_weights_dir = _validate_weights_dir(args.weights_dir.strip() or None, model_id=args.model_id)
    if validated_weights_dir is None and not args.allow_unconverted_weights:
        raise RuntimeError(
            "Building the runtime graph requires converted Cactus CQ weights.\n"
            "\n"
            "Run:\n"
            f"  cactus convert {args.model_id} --bits 4\n"
            "\n"
            "For compiler-only debugging, pass --allow-unconverted-weights."
        )
    if args.low_memory_load:
        if validated_weights_dir is None:
            raise RuntimeError("--low-memory-load requires --weights-dir with converted Cactus CQ weights")
        if not args.skip_execute:
            print("note=low_memory_load_forces_skip_execute=true")
            args.skip_execute = True
        args.skip_reference_compare = True

    image_files = tuple(str(path) for path in args.image_file if str(path).strip())
    if args.task == "auto":
        inferred_task = _infer_task_from_config(args.model_id)
        config_for_auto = _load_config_json(args.model_id)
        plan_for_auto = infer_component_plan_from_config(config_for_auto, model_id=args.model_id)
        has_multimodal_config = (
            plan_for_auto is not None
            and plan_for_auto.task == "multimodal_causal_lm_logits"
        )
        if image_files or (args.audio_file and has_multimodal_config and inferred_task == "causal_lm_logits"):
            task = plan_for_auto.task if plan_for_auto is not None else "multimodal_causal_lm_logits"
        else:
            task = inferred_task
    else:
        task = args.task
    torch_dtype = _parse_dtype(args.torch_dtype)
    weights_dir = str(validated_weights_dir) if validated_weights_dir is not None else None
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else _default_artifact_dir_for_model(args.model_id)

    model_source, processor_or_tokenizer, model, model_config = _load_transformers_bundle(
        model_id=args.model_id,
        task=task,
        torch_dtype=torch_dtype,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_memory_load=bool(args.low_memory_load),
    )
    preprocessor_config = _load_optional_json(model_source, "preprocessor_config.json")
    if not preprocessor_config:
        preprocessor_config = _load_optional_json(args.model_id, "preprocessor_config.json")
    auxiliary_tokenizer = None
    if task == "ctc_logits":
        auxiliary_tokenizer = _load_optional_tokenizer(
            model_id=args.model_id,
            model_source=model_source,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    canonical = None
    if task == "causal_lm_logits":
        prepared = _prepare_text_inputs(
            processor_or_tokenizer,
            prompt=args.prompt,
            input_ids_text=args.input_ids.strip() or None,
            max_new_tokens=int(args.max_new_tokens),
            enable_thinking_if_supported=args.enable_thinking,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
            weights_dir=weights_dir,
            inputs_metadata=prepared.metadata,
        )
    elif task == "text_embedding":
        prepared = _prepare_text_embedding_inputs(
            processor_or_tokenizer,
            prompt=args.prompt,
            input_ids_text=args.input_ids.strip() or None,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
            weights_dir=weights_dir,
            inputs_metadata=prepared.metadata,
        )
    elif task == "multimodal_causal_lm_logits":
        config_model_type = str(model_config.get("model_type", "") or "").lower()
        if config_model_type == "lfm2_vl":
            prepared = _prepare_lfm2_vl_multimodal_inputs(
                processor_or_tokenizer,
                prompt=args.prompt,
                image_files=image_files,
                torch_dtype=torch_dtype,
                system_prompt=args.system_prompt,
                enable_thinking_if_supported=args.enable_thinking,
            )
        elif config_model_type in {"qwen3_5", "qwen3_vl"}:
            prepared = _prepare_qwen3_5_multimodal_inputs(
                processor_or_tokenizer,
                prompt=args.prompt,
                image_files=image_files,
                torch_dtype=torch_dtype,
                system_prompt=args.system_prompt,
                enable_thinking_if_supported=args.enable_thinking,
            )
        else:
            prepared = _prepare_gemma4_multimodal_inputs(
                processor_or_tokenizer,
                prompt=args.prompt,
                image_files=image_files,
                audio_file=args.audio_file.strip() or None,
                torch_dtype=torch_dtype,
                system_prompt=args.system_prompt,
                enable_thinking_if_supported=args.enable_thinking,
                use_gemma4_chat_template=True,
            )
        prepared = _add_multimodal_generation_headroom(
            prepared,
            tokenizer=getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer),
            max_new_tokens=int(args.max_new_tokens),
            min_context_tokens=_multimodal_context_token_floor(config_model_type),
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
            weights_dir=weights_dir,
        )
        prime_static_features = getattr(canonical.module, "prime_static_multimodal_features", None)
        if callable(prime_static_features) and not args.low_memory_load:
            prime_static_features(*prepared.tensors)
    elif task == "tdt_transcription":
        if not args.audio_file:
            raise RuntimeError(f"--audio-file is required for task={task}")
        prepared = _prepare_audio_inputs(
            processor_or_tokenizer,
            input_names=("input_features",),
            config=model_config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=args.audio_file,
            torch_dtype=torch_dtype,
        )
    elif task == "seq2seq_transcription":
        if not args.audio_file:
            raise RuntimeError(f"--audio-file is required for task={task}")
        prepared = _prepare_audio_inputs(
            processor_or_tokenizer,
            input_names=("input_features",),
            config=model_config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=args.audio_file,
            torch_dtype=torch_dtype,
        )
        if getattr(model_config, "get", None) is not None or getattr(model, "config", None) is not None:
            prepared = _augment_whisper_seq2seq_metadata(
                prepared,
                tokenizer_or_processor=processor_or_tokenizer,
                model=model,
                prompt=args.prompt,
                max_new_tokens=int(args.max_new_tokens),
            )
    else:
        if not args.audio_file:
            raise RuntimeError(f"--audio-file is required for task={task}")
        canonical = canonicalize_model_interface(model, task=task)
        prepared = _prepare_audio_inputs(
            processor_or_tokenizer,
            input_names=canonical.input_names,
            config=model_config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=args.audio_file,
            torch_dtype=torch_dtype,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
        )
    defer_low_memory_meta_inputs = (
        bool(args.low_memory_load)
        and task == "multimodal_causal_lm_logits"
        and config_model_type == "gemma4"
    )
    if args.low_memory_load and not defer_low_memory_meta_inputs:
        prepared = _prepared_inputs_for_meta_capture(prepared)
    requested_components = None
    if args.components:
        requested_components = tuple(
            component.strip()
            for component in str(args.components).split(",")
            if component.strip()
        )
    plan_derived_components = False
    if requested_components is None and task == "causal_lm_logits":
        inferred_plan = infer_component_plan_from_config(
            _load_config_json(args.model_id), model_id=args.model_id
        )
        if (
            inferred_plan is not None
            and inferred_plan.task == task
            and inferred_plan.force_component_pipeline
            and inferred_plan.components
        ):
            requested_components = tuple(inferred_plan.components)
            plan_derived_components = True
    spec_kwargs = dict(
        task=task,
        named_tensors=_named_tensor_store(prepared),
        weights_dir=weights_dir,
        inputs_metadata=prepared.metadata,
        cache_context_length=args.cache_context_length,
    )
    try:
        component_specs = build_component_module_specs(
            model, components=requested_components, **spec_kwargs
        )
    except UnsupportedComponentsError as exc:
        if not plan_derived_components:
            raise
        print(f"warning: dropping inferred component plan ({exc}); using builder defaults", file=sys.stderr)
        component_specs = build_component_module_specs(model, components=None, **spec_kwargs)
    use_component_pipeline = False
    if args.component_pipeline == "on":
        if component_specs is None:
            raise RuntimeError(
                f"--component-pipeline=on was requested, but {type(model).__name__} does not expose component specs for task={task}"
            )
        use_component_pipeline = True
    elif args.component_pipeline == "auto" and component_specs is not None:
        use_component_pipeline = True

    if use_component_pipeline:
        family = canonical.family if canonical is not None else getattr(model, "family", type(model).__name__)
        return _run_component_pipeline_transpile(
            args=args,
            task=task,
            family=str(family),
            model_source=model_source,
            model=model,
            prepared=prepared,
            component_specs=component_specs,
            fusion_config=FusionConfig(
                enable_gated_deltanet=not args.no_fuse_gated_deltanet,
                enable_rms_norm=not args.no_fuse_rms_norm,
                enable_rope=not args.no_fuse_rope,
                enable_attention=not args.no_fuse_attention,
                enable_attention_block=not args.no_fuse_attention_block,
                enable_add_clipped=not args.no_fuse_add_clipped,
            ),
            weights_dir=weights_dir,
            artifact_dir=artifact_dir,
            processor_or_tokenizer=processor_or_tokenizer,
            canonical=canonical,
        )

    if canonical is None:
        raise RuntimeError(f"task={task} requires a component pipeline but none was selected")
    if args.low_memory_load and defer_low_memory_meta_inputs:
        prepared = _prepared_inputs_for_meta_capture(prepared)
    wrapper = TranspileWrapper(canonical.module, weights_dir=weights_dir).eval()

    print(f"model_id={args.model_id}")
    print(f"model_source={model_source}")
    print(f"task={task}")
    print(f"adapter_family={canonical.family}")
    print(f"adapter_module={type(canonical.module).__name__}")
    print(f"input_names={','.join(prepared.names)}")
    for name, tensor in zip(prepared.names, prepared.tensors):
        print(f"input_{name}_shape={list(tensor.shape)}")
    if weights_dir:
        print(f"weights_dir={weights_dir}")

    print("capture_begin=true", flush=True)
    prepare_cpu_float32_capture = getattr(canonical.module, "prepare_cpu_float32_capture", None)
    restore_cpu_float32_capture = getattr(canonical.module, "restore_cpu_float32_capture", None)
    if callable(prepare_cpu_float32_capture):
        prepare_cpu_float32_capture()
    try:
        captured = capture_model(wrapper, prepared.tensors)
    finally:
        if callable(restore_cpu_float32_capture):
            restore_cpu_float32_capture()
    print("capture_done=true", flush=True)
    captured.ir_graph.meta.setdefault("task", task)
    captured.ir_graph.meta.setdefault("adapter_family", canonical.family)
    raw_ir_graph = copy.deepcopy(captured.ir_graph)

    fusion_config = FusionConfig(
        enable_gated_deltanet=not args.no_fuse_gated_deltanet,
        enable_rms_norm=not args.no_fuse_rms_norm,
        enable_rope=not args.no_fuse_rope,
        enable_attention=not args.no_fuse_attention,
        enable_attention_block=not args.no_fuse_attention_block,
        enable_add_clipped=not args.no_fuse_add_clipped,
    )

    print("canonicalize_begin=true", flush=True)
    canonicalize_exported_graph(captured.ir_graph)
    print("canonicalize_done=true", flush=True)
    print("optimize_begin=true", flush=True)
    optimize_graph(captured.ir_graph, config=fusion_config)
    print("optimize_done=true", flush=True)
    print("lower_begin=true", flush=True)
    tg = _lower_preoptimized_ir(captured.ir_graph)
    print("lower_done=true", flush=True)

    optimized_ir_graph = copy.deepcopy(captured.ir_graph)
    binding_count = _count_weight_bindings(optimized_ir_graph)
    op_counts = Counter(optimized_ir_graph.nodes[node_id].op for node_id in optimized_ir_graph.order)
    raw_component_counts = summarize_ir_components(raw_ir_graph)
    optimized_component_counts = summarize_ir_components(optimized_ir_graph)
    raw_component_graphs = extract_component_subgraphs(raw_ir_graph)
    optimized_component_graphs = extract_component_subgraphs(optimized_ir_graph)
    transpiled_component_graphs: dict[str, TranspiledGraph] | None = None

    print(f"raw_ir_nodes={len(raw_ir_graph.order)}")
    print(f"optimized_ir_nodes={len(optimized_ir_graph.order)}")
    print(f"weight_bindings={binding_count}")
    if raw_component_counts:
        print(
            "raw_components="
            + ",".join(f"{name}:{count}" for name, count in sorted(raw_component_counts.items()))
        )
    if optimized_component_counts:
        print(
            "optimized_components="
            + ",".join(f"{name}:{count}" for name, count in sorted(optimized_component_counts.items()))
        )
    print(
        "ops="
        f"attention:{op_counts.get('attention', 0)} "
        f"conv1d:{op_counts.get('conv1d', 0)} "
        f"conv2d:{op_counts.get('conv2d', 0)} "
        f"batch_norm:{op_counts.get('batch_norm', 0)} "
        f"layer_norm:{op_counts.get('layer_norm', 0)} "
        f"rms_norm:{op_counts.get('rms_norm', 0)} "
        f"rope:{op_counts.get('rope', 0)} "
        f"linear:{op_counts.get('linear', 0)}"
    )

    if weights_dir and binding_count == 0:
        raise RuntimeError(
            f"No weight bindings were resolved from {weights_dir}\n"
            "\n"
            "The weights folder exists, but none of the captured constants matched entries in weights_manifest.json.\n"
            "\n"
            f"Recommended fix:\n"
            f"  cactus convert {args.model_id} {weights_dir}\n"
        )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        transpiled_component_graphs = {
            component: _lower_preoptimized_ir(copy.deepcopy(component_graph))
            for component, component_graph in optimized_component_graphs.items()
        }
        _write_json(
            artifact_dir / "raw_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": canonical.family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "graph": _graph_to_dict(raw_ir_graph),
            },
        )
        _write_json(
            artifact_dir / "optimized_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": canonical.family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "graph": _graph_to_dict(optimized_ir_graph),
            },
        )
        for component, component_graph in raw_component_graphs.items():
            _write_json(
                artifact_dir / _component_artifact_name("raw_ir", component),
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": canonical.family,
                    "component": component,
                    "inputs": _serialize_json_compatible(prepared.metadata),
                    "graph": _graph_to_dict(component_graph),
                },
            )
        for component, component_graph in optimized_component_graphs.items():
            _write_json(
                artifact_dir / _component_artifact_name("optimized_ir", component),
                {
                    "model_id": args.model_id,
                    "model_source": model_source,
                    "task": task,
                    "family": canonical.family,
                    "component": component,
                    "inputs": _serialize_json_compatible(prepared.metadata),
                    "graph": _graph_to_dict(component_graph),
                },
            )
        print(f"saved_raw_ir={artifact_dir / 'raw_ir.json'}")
        print(f"saved_optimized_ir={artifact_dir / 'optimized_ir.json'}")
        for component in raw_component_graphs:
            print(f"saved_raw_component_ir_{component}={artifact_dir / _component_artifact_name('raw_ir', component)}")
        for component in optimized_component_graphs:
            print(
                f"saved_optimized_component_ir_{component}="
                f"{artifact_dir / _component_artifact_name('optimized_ir', component)}"
            )

        component_manifest_path = _write_component_bundle(
            artifact_dir=artifact_dir,
            model_id=args.model_id,
            model_source=model_source,
            task=task,
            family=canonical.family,
            inputs_metadata=prepared.metadata,
            raw_component_graphs=raw_component_graphs,
            optimized_component_graphs=optimized_component_graphs,
            transpiled_component_graphs=transpiled_component_graphs,
            graph_filename=args.graph_filename,
        )
        print(f"saved_component_bundle_manifest={component_manifest_path}")
        for component in optimized_component_graphs:
            print(
                f"saved_component_graph_{component}="
                f"{artifact_dir / 'components' / component / args.graph_filename}"
            )
        graph_path = artifact_dir / args.graph_filename
        _embed_materialized_bound_constants(tg)
        tg.graph.save(graph_path)
        print(f"saved_graph={graph_path}")
        graph_binding_manifest_path = _write_graph_binding_manifest(
            artifact_dir=artifact_dir,
            filename="graph_bindings.json",
            model_id=args.model_id,
            model_source=model_source,
            task=task,
            family=canonical.family,
            inputs_metadata=prepared.metadata,
            transpiled_graph=tg,
        )
        print(f"saved_graph_bindings={graph_binding_manifest_path}")
        if binding_count > 0:
            print(
                "note=saved graph structure without embedded mmap bindings; use the saved component/full-graph manifests "
                "to rebind weights/embeddings from the weights folder when loading"
            )

    if args.skip_execute:
        return 0

    tg.set_inputs([tensor.cpu().numpy() for tensor in prepared.tensors])
    print("execute_begin=true")
    transpiled_output = tg.execute()[0].numpy().astype(np.float32)
    print("execute_done=true")

    if args.skip_reference_compare:
        result_payload = {
            "model_id": args.model_id,
            "model_source": model_source,
            "task": task,
            "family": canonical.family,
            "inputs": _serialize_json_compatible(prepared.metadata),
            "output_shape": list(transpiled_output.shape),
            "raw_ir_nodes": len(raw_ir_graph.order),
            "optimized_ir_nodes": len(optimized_ir_graph.order),
            "weight_bindings": binding_count,
            "reference_compare_skipped": True,
        }
        print(f"output_shape={list(transpiled_output.shape)}")
        if task in {"causal_lm_logits", "multimodal_causal_lm_logits"}:
            tokenizer_like = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
            transpiled_next = int(np.argmax(transpiled_output[0, -1]))
            print(f"transpiled_next_token_id={transpiled_next}")
            result_payload["transpiled_next_token_id"] = transpiled_next
            if hasattr(tokenizer_like, "decode"):
                transpiled_token = tokenizer_like.decode([transpiled_next])
                print(f"transpiled_next_token={transpiled_token!r}")
                result_payload["transpiled_next_token"] = transpiled_token
        if artifact_dir is not None:
            _write_json(artifact_dir / "result.json", result_payload)
            print(f"saved_result={artifact_dir / 'result.json'}")
        del model
        del wrapper
        gc.collect()
        return 0

    print("reference_begin=true")
    with torch.no_grad():
        reference_output = wrapper(*prepared.tensors).detach().float().cpu().numpy()
    print("reference_done=true")

    max_abs_diff = float(np.max(np.abs(reference_output - transpiled_output)))
    mean_abs_diff = float(np.mean(np.abs(reference_output - transpiled_output)))
    result_payload: dict[str, object] = {
        "model_id": args.model_id,
        "model_source": model_source,
        "task": task,
        "family": canonical.family,
        "inputs": _serialize_json_compatible(prepared.metadata),
        "output_shape": list(reference_output.shape),
        "raw_ir_nodes": len(raw_ir_graph.order),
        "optimized_ir_nodes": len(optimized_ir_graph.order),
        "weight_bindings": binding_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }

    if task == "causal_lm_logits":
        tokenizer = processor_or_tokenizer
        hf_next = int(np.argmax(reference_output[0, -1]))
        transpiled_next = int(np.argmax(transpiled_output[0, -1]))
        print(f"hf_next_token_id={hf_next}")
        print(f"transpiled_next_token_id={transpiled_next}")
        print(f"logits_max_abs_diff={max_abs_diff:.6f}")
        print(f"logits_mean_abs_diff={mean_abs_diff:.6f}")
        print(f"hf_next_token={tokenizer.decode([hf_next])!r}")
        print(f"transpiled_next_token={tokenizer.decode([transpiled_next])!r}")
        result_payload.update(
            {
                "hf_next_token_id": hf_next,
                "transpiled_next_token_id": transpiled_next,
                "hf_next_token": tokenizer.decode([hf_next]),
                "transpiled_next_token": tokenizer.decode([transpiled_next]),
            }
        )
    elif task == "ctc_logits":
        blank_token_id = getattr(model.config, "pad_token_id", None)
        if blank_token_id is None and auxiliary_tokenizer is not None:
            blank_token_id = getattr(auxiliary_tokenizer, "pad_token_id", None)
        print(f"output_shape={list(reference_output.shape)}")
        print(f"output_max_abs_diff={max_abs_diff:.6f}")
        print(f"output_mean_abs_diff={mean_abs_diff:.6f}")
        if auxiliary_tokenizer is not None:
            hf_token_ids = _ctc_greedy_decode_token_ids(reference_output, blank_token_id=blank_token_id)
            transpiled_token_ids = _ctc_greedy_decode_token_ids(transpiled_output, blank_token_id=blank_token_id)
            hf_transcript = _decode_token_ids(auxiliary_tokenizer, hf_token_ids)
            transpiled_transcript = _decode_token_ids(auxiliary_tokenizer, transpiled_token_ids)
            print(f"hf_transcript={hf_transcript!r}")
            print(f"transpiled_transcript={transpiled_transcript!r}")
            result_payload.update(
                {
                    "blank_token_id": None if blank_token_id is None else int(blank_token_id),
                    "hf_transcript_token_ids": hf_token_ids,
                    "transpiled_transcript_token_ids": transpiled_token_ids,
                    "hf_transcript": hf_transcript,
                    "transpiled_transcript": transpiled_transcript,
                }
            )
    else:
        print(f"output_shape={list(reference_output.shape)}")
        print(f"output_max_abs_diff={max_abs_diff:.6f}")
        print(f"output_mean_abs_diff={mean_abs_diff:.6f}")

    if artifact_dir is not None:
        _write_json(artifact_dir / "result.json", result_payload)
        print(f"saved_result={artifact_dir / 'result.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
