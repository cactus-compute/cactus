from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import ctypes
import json
import os
from collections.abc import Mapping
from pathlib import Path
import re
import struct
import time
from typing import Any

import numpy as np
import torch

from cactus.convert.cactus_adapters.tensor_io import CACTUS_MAGIC
from cactus.convert.cactus_adapters.tensor_io import FLAG_INTERLEAVED
from cactus.convert.cactus_adapters.tensor_io import align_offset
from cactus.transpile.audio_preprocess import generic_log_mel_features as _generic_log_mel_features
from cactus.transpile.audio_preprocess import load_audio_waveform as _load_audio_waveform
from cactus.transpile.audio_preprocess import prepare_cactus_audio_features
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.multimodal_runtime import prepare_gemma4_multimodal_inputs
from cactus.transpile.multimodal_runtime import _build_gemma4_chat_prompt
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.media_limits import resize_static_image
from cactus.transpile.lower import transpile_preoptimized_ir
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.runtime_support import ensure_transformers_supports_gemma4
from cactus.transpile.runtime_support import patch_torch_flex_attention_compat
from cactus.transpile.runtime_support import patch_transformers_torchvision_probe
from cactus.transpile.runtime_support import PreparedInputs
from cactus.transpile.tdt_runtime import greedy_decode_parakeet_tdt_token_ids
from cactus.transpile.tdt_runtime import load_parakeet_tdt_config
from cactus.transpile.tdt_runtime import prepare_parakeet_tdt_audio_features
from cactus.transpile.runtime_compat import _lib
from cactus.transpile.runtime_compat import cactus_node_t
from cactus.transpile.runtime_compat import Graph
from cactus.transpile.runtime_compat import Tensor


_HEADER_SIZE = 84
_FLAG_EXTENDED_SHAPE = 1 << 4
_PRECISION_TO_DTYPE = {
    Graph.INT8: np.int8,
    Graph.FP16: np.float16,
    Graph.FP32: np.float32,
    Graph.INT4: np.uint8,
    getattr(Graph, "CQ2", 4): np.uint8,
    getattr(Graph, "CQ3", 5): np.uint8,
    getattr(Graph, "CQ4", 6): np.uint8,
}

_STATEFUL_DECODE_COMPONENTS = frozenset({"decoder_prefill_chunk", "decoder_step"})
_COMPONENT_GRAPH_CACHE: dict[tuple[str, str | None, str], tuple[dict[str, "LoadedComponentGraph"], dict[str, object]]] = {}
_MULTIMODAL_ENCODER_FEATURE_CACHE: dict[tuple[str, str, tuple[str, ...], str | None], dict[str, np.ndarray]] = {}
_UNBOUNDED_GENERATION_GUARD_TOKENS = 512


def _has_runtime_symbol(name: str) -> bool:
    symbol = getattr(_lib, name, None)
    return symbol is not None and not getattr(symbol, "_cactus_missing_symbol", False)


@dataclass
class LoadedTensorFile:
    path: Path
    precision: int
    shape: tuple[int, ...]
    data: np.memmap
    scales: np.memmap | None
    group_size: int
    num_groups: int
    is_interleaved: bool
    original_n: int


@dataclass
class LoadedComponentGraph:
    component: str
    graph: Graph
    runtime_inputs: list[Tensor]
    outputs: list[Tensor]
    bound_constant_bindings: list[dict[str, object]]
    bound_tensor_files: list[object]
    cache_state_tensors: list[tuple[str, Tensor, Tensor]] = field(default_factory=list)
    external_input_refs: dict[int, np.ndarray] = field(default_factory=dict)

    def set_input(self, index: int, data: Any, *, dtype: int | None = None) -> None:
        if index < 0 or index >= len(self.runtime_inputs):
            raise IndexError(
                f"runtime input index out of range: {index} (have {len(self.runtime_inputs)})"
            )
        self.graph.set_input(self.runtime_inputs[index], data, dtype=dtype)

    def set_inputs(self, inputs: list[Any] | tuple[Any, ...]) -> None:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"expected {len(self.runtime_inputs)} runtime inputs, got {len(inputs)}"
            )
        for index, value in enumerate(inputs):
            self.set_input(index, value)

    def set_external_input(self, index: int, data: Any, *, dtype: int | None = None) -> np.ndarray:
        if index < 0 or index >= len(self.runtime_inputs):
            raise IndexError(
                f"runtime input index out of range: {index} (have {len(self.runtime_inputs)})"
            )
        tensor = self.runtime_inputs[index]
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        array = self.graph._coerce_input_array(data, target_dtype)
        self.graph.set_external_input(tensor, int(array.ctypes.data), dtype=target_dtype)
        self.external_input_refs[index] = array
        return array

    def set_external_inputs(self, inputs: list[Any] | tuple[Any, ...]) -> list[np.ndarray]:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"expected {len(self.runtime_inputs)} runtime inputs, got {len(inputs)}"
            )
        bound: list[np.ndarray] = []
        for index, value in enumerate(inputs):
            bound.append(self.set_external_input(index, value))
        return bound

    def execute(self) -> list[Tensor]:
        self.graph.execute()
        return self.outputs


def load_component_bundle_manifest(bundle_dir_or_manifest: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(bundle_dir_or_manifest).expanduser().resolve()
    if path.is_dir():
        candidate = path / "manifest.json" if path.name == "components" else path / "components" / "manifest.json"
        if not candidate.exists():
            candidate = path / "manifest.json"
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"component bundle manifest not found: {path}")
    manifest = json.loads(path.read_text())
    bundle_root = path.parent.parent if path.parent.name == "components" else path.parent
    return bundle_root, manifest


def _component_cache_state_tensors(
    graph: Graph,
    component_entry: Mapping[str, object],
) -> list[tuple[str, Tensor, Tensor]]:
    result: list[tuple[str, Tensor, Tensor]] = []
    for entry in component_entry.get("cache_state_node_ids", []) or []:
        if not isinstance(entry, Mapping):
            continue
        layer_key = str(entry.get("layer_key", ""))
        key_id = entry.get("key")
        value_id = entry.get("value")
        if not isinstance(key_id, int) or not isinstance(value_id, int):
            continue
        result.append((layer_key, graph._tensor_from_node(int(key_id)), graph._tensor_from_node(int(value_id))))
    return result


def load_saved_component_graph(
    *,
    bundle_root: str | Path,
    component_entry: dict[str, object],
    weights_dir: str | Path | None = None,
) -> LoadedComponentGraph:
    root = Path(bundle_root).expanduser().resolve()
    graph_relpath = component_entry.get("graph")
    has_graph = isinstance(graph_relpath, str) and bool(graph_relpath)
    graph_path = (root / str(graph_relpath)).resolve() if has_graph else None
    prefer_saved_graph = os.environ.get("CACTUS_TRANSPILER_PREFER_SAVED_GRAPH", "1") != "0"
    if prefer_saved_graph and graph_path is not None and graph_path.exists():
        try:
            graph = Graph.load(graph_path)

            runtime_inputs = [
                graph._tensor_from_node(int(node_id))
                for node_id in component_entry.get("runtime_input_node_ids", [])
            ]
            outputs = [
                graph._tensor_from_node(int(node_id))
                for node_id in component_entry.get("output_node_ids", [])
            ]
            bound_constant_bindings = list(component_entry.get("bound_constant_bindings") or [])
            bound_tensor_files = _rebind_bound_constants(
                graph=graph,
                bundle_root=root,
                bindings=bound_constant_bindings,
                weights_dir=weights_dir,
            )
            return LoadedComponentGraph(
                component=str(component_entry.get("component", "unknown")),
                graph=graph,
                runtime_inputs=runtime_inputs,
                outputs=outputs,
                bound_constant_bindings=bound_constant_bindings,
                bound_tensor_files=bound_tensor_files,
                cache_state_tensors=_component_cache_state_tensors(graph, component_entry),
            )
        except Exception as exc:
            if not (
                isinstance(component_entry.get("raw_ir"), str)
                or isinstance(component_entry.get("optimized_ir"), str)
            ):
                raise
            print(
                f"note=component_graph_load_failed component={component_entry.get('component', 'unknown')} "
                f"path={graph_path} fallback=ir reason={exc}",
                flush=True,
            )

    raw_ir_relpath = component_entry.get("raw_ir")
    optimized_ir_relpath = component_entry.get("optimized_ir")
    has_raw_ir = isinstance(raw_ir_relpath, str) and bool(raw_ir_relpath)
    has_optimized_ir = isinstance(optimized_ir_relpath, str) and bool(optimized_ir_relpath)
    if has_raw_ir or has_optimized_ir:
        return _load_component_graph_from_ir(
            bundle_root=root,
            component_entry=component_entry,
            weights_dir=weights_dir,
        )

    if not isinstance(graph_relpath, str) or not graph_relpath:
        raise ValueError(f"component entry is missing graph path: {component_entry}")

    graph_path = (root / graph_relpath).resolve()
    graph = Graph.load(graph_path)

    runtime_inputs = [
        graph._tensor_from_node(int(node_id))
        for node_id in component_entry.get("runtime_input_node_ids", [])
    ]
    outputs = [
        graph._tensor_from_node(int(node_id))
        for node_id in component_entry.get("output_node_ids", [])
    ]
    bound_constant_bindings = list(component_entry.get("bound_constant_bindings") or [])
    bound_tensor_files = _rebind_bound_constants(
        graph=graph,
        bundle_root=root,
        bindings=bound_constant_bindings,
        weights_dir=weights_dir,
    )
    return LoadedComponentGraph(
        component=str(component_entry.get("component", "unknown")),
        graph=graph,
        runtime_inputs=runtime_inputs,
        outputs=outputs,
        bound_constant_bindings=bound_constant_bindings,
        bound_tensor_files=bound_tensor_files,
        cache_state_tensors=_component_cache_state_tensors(graph, component_entry),
    )


def _load_component_graph_from_ir(
    *,
    bundle_root: Path,
    component_entry: dict[str, object],
    weights_dir: str | Path | None,
) -> LoadedComponentGraph:
    raw_ir_relpath = component_entry.get("raw_ir")
    optimized_ir_relpath = component_entry.get("optimized_ir")
    has_raw_ir = isinstance(raw_ir_relpath, str) and bool(raw_ir_relpath)
    has_optimized_ir = isinstance(optimized_ir_relpath, str) and bool(optimized_ir_relpath)
    use_raw_ir = has_raw_ir and not has_optimized_ir
    ir_relpath = str(raw_ir_relpath if use_raw_ir else optimized_ir_relpath)
    ir_path = (bundle_root / ir_relpath).resolve()
    payload = json.loads(ir_path.read_text())
    graph_payload = payload.get("graph")
    if not isinstance(graph_payload, dict):
        raise ValueError(f"saved IR payload is missing graph data: {ir_path}")

    ir_graph = _deserialize_saved_ir_graph(
        graph_payload=graph_payload,
        component_entry=component_entry,
        bundle_root=bundle_root,
        weights_dir=weights_dir,
    )
    if use_raw_ir:
        canonicalize_exported_graph(ir_graph)
        optimize_graph(ir_graph)
    else:
        family = str(ir_graph.meta.get("adapter_family") or ir_graph.meta.get("family") or "").strip().lower()
        component = str(ir_graph.meta.get("component", "") or component_entry.get("component", "") or "").strip().lower()
        if family == "gemma4" and component in {"decoder", "decoder_step"}:
            optimize_graph(ir_graph)
    transpiled = transpile_preoptimized_ir(ir_graph)
    return LoadedComponentGraph(
        component=str(component_entry.get("component", "unknown")),
        graph=transpiled.graph,
        runtime_inputs=list(transpiled.runtime_inputs),
        outputs=list(transpiled.outputs),
        bound_constant_bindings=list(component_entry.get("bound_constant_bindings") or []),
        bound_tensor_files=[],
        cache_state_tensors=list(getattr(transpiled, "cache_state_tensors", [])),
    )


def load_saved_component_graphs(
    bundle_dir_or_manifest: str | Path,
    *,
    weights_dir: str | Path | None = None,
    include_components: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, LoadedComponentGraph], dict[str, object]]:
    bundle_root, manifest = load_component_bundle_manifest(bundle_dir_or_manifest)
    include_component_names = (
        frozenset(str(name) for name in include_components)
        if include_components is not None
        else None
    )
    cache_key = (
        str(bundle_root),
        None if weights_dir is None else str(Path(weights_dir).expanduser().resolve()),
        os.environ.get("CACTUS_TRANSPILER_PREFER_SAVED_GRAPH", ""),
        None if include_component_names is None else tuple(sorted(include_component_names)),
    )
    has_stateful_decode_graph = any(
        isinstance(component_entry, dict)
        and str(component_entry.get("component", "")).strip() in _STATEFUL_DECODE_COMPONENTS
        and (
            include_component_names is None
            or str(component_entry.get("component", "")).strip() in include_component_names
        )
        for component_entry in manifest.get("components", [])
    )
    cache_components = os.environ.get("CACTUS_TRANSPILER_DISABLE_GRAPH_CACHE") != "1"
    if cache_components:
        cached = _COMPONENT_GRAPH_CACHE.get(cache_key)
        if cached is not None:
            if not has_stateful_decode_graph:
                _attach_component_io_names(manifest, cached[0])
                return cached
            loaded = dict(cached[0])
            for component_entry in manifest.get("components", []):
                if not isinstance(component_entry, dict):
                    continue
                component_name = str(component_entry.get("component", "")).strip()
                if component_name not in _STATEFUL_DECODE_COMPONENTS:
                    continue
                if include_component_names is not None and component_name not in include_component_names:
                    continue
                loaded[component_name] = load_saved_component_graph(
                    bundle_root=bundle_root,
                    component_entry=component_entry,
                    weights_dir=weights_dir,
                )
            _attach_component_io_names(manifest, loaded)
            return loaded, manifest

    loaded: dict[str, LoadedComponentGraph] = {}
    for component_entry in manifest.get("components", []):
        if not isinstance(component_entry, dict):
            continue
        component_name = str(component_entry.get("component", "")).strip()
        if not component_name:
            continue
        if include_component_names is not None and component_name not in include_component_names:
            continue
        loaded[component_name] = load_saved_component_graph(
            bundle_root=bundle_root,
            component_entry=component_entry,
            weights_dir=weights_dir,
        )
    _attach_component_io_names(manifest, loaded)
    result = (loaded, manifest)
    if cache_components:
        if has_stateful_decode_graph:
            _COMPONENT_GRAPH_CACHE[cache_key] = (
                {name: graph for name, graph in loaded.items() if name not in _STATEFUL_DECODE_COMPONENTS},
                manifest,
            )
        else:
            _COMPONENT_GRAPH_CACHE[cache_key] = result
    return result


def run_transpiled_bundle(
    bundle_dir_or_manifest: str | Path,
    *,
    audio_file: str | Path | None = None,
    image_files: tuple[str, ...] = (),
    prompt: str | None = None,
    input_ids: str | list[int] | tuple[int, ...] | None = None,
    weights_dir: str | Path | None = None,
    torch_dtype: torch.dtype = torch.float16,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    max_new_tokens: int | None = None,
    stop_sequences: tuple[str, ...] = (),
) -> dict[str, object]:
    bundle_root, manifest = load_component_bundle_manifest(bundle_dir_or_manifest)
    manifest = dict(manifest)
    manifest["_bundle_root"] = str(bundle_root)
    resolved_weights_dir = _default_weights_dir_for_manifest(manifest, explicit=weights_dir)
    family = str(manifest.get("family", "") or "")
    task = str(manifest.get("task", "") or "")

    include_components = runtime_include_components_for_manifest(
        family=family,
        task=task,
        manifest=manifest,
    )
    if (
        family == "gemma4"
        and task == "multimodal_causal_lm_logits"
        and prompt is not None
        and not image_files
        and audio_file is None
    ):
        manifest_components = {
            str(component_entry.get("component", "")).strip()
            for component_entry in manifest.get("components", [])
            if isinstance(component_entry, dict)
        }
        if {"lm_encoder_step", "decoder_step"}.issubset(manifest_components):
            include_components = {"lm_encoder_step", "decoder_step"}
    component_graphs, manifest = load_saved_component_graphs(
        bundle_dir_or_manifest,
        weights_dir=resolved_weights_dir,
        include_components=include_components,
    )
    manifest = dict(manifest)
    manifest["_bundle_root"] = str(bundle_root)
    if family == "parakeet_tdt" and task == "tdt_transcription":
        if audio_file is None:
            raise ValueError("audio_file is required for Parakeet TDT component bundles")
        return _run_parakeet_tdt_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )
    if task == "multimodal_causal_lm_logits":
        return _run_multimodal_causal_lm_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=prompt,
            image_files=image_files,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
    if task == "causal_lm_logits":
        return _run_causal_lm_logits_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=prompt,
            input_ids=input_ids,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
    if task == "seq2seq_transcription":
        if audio_file is None:
            inputs_meta = manifest.get("inputs")
            if isinstance(inputs_meta, dict):
                stored_audio = inputs_meta.get("audio_file")
                if isinstance(stored_audio, str) and stored_audio:
                    audio_file = stored_audio
        if audio_file is None:
            raise ValueError("audio_file is required for seq2seq_transcription bundles")
        return _run_seq2seq_transcription_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            audio_file=audio_file,
            prompt=prompt,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
    if task == "encoder_hidden_states":
        if audio_file is None:
            inputs_meta = manifest.get("inputs")
            if isinstance(inputs_meta, dict):
                stored_audio = inputs_meta.get("audio_file")
                if isinstance(stored_audio, str) and stored_audio:
                    audio_file = stored_audio
        if audio_file is None:
            raise ValueError("audio_file is required for encoder_hidden_states bundles")
        return _run_encoder_hidden_states_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )
    raise NotImplementedError(
        f"saved transpiled bundle execution is not implemented for family={family!r} task={task!r}"
    )


def runtime_include_components_for_manifest(
    *,
    family: str,
    task: str,
    manifest: Mapping[str, object],
) -> set[str] | None:
    manifest_components = {
        str(component_entry.get("component", "")).strip()
        for component_entry in manifest.get("components", [])
        if isinstance(component_entry, dict)
    }
    if (
        family == "gemma4"
        and task == "multimodal_causal_lm_logits"
        and {"lm_encoder_step", "decoder_step"}.issubset(manifest_components)
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        # Older Gemma bundles may include a full static decoder. Cached decode
        # should not pay to load it when step graphs are available.
        include_components = set(manifest_components)
        include_components.discard("decoder")
        return include_components
    return None


def _default_weights_dir_for_manifest(
    manifest: Mapping[str, object],
    *,
    explicit: str | Path | None,
) -> str | Path | None:
    if explicit is not None:
        return explicit
    model_id = str(manifest.get("model_id", "") or "").strip()
    if not model_id:
        return None
    try:
        from cactus.cli.download import get_weights_dir

        candidate = get_weights_dir(model_id)
    except Exception:
        return None
    return candidate if candidate.exists() else None


def _bundle_root_from_manifest(manifest: Mapping[str, object]) -> Path | None:
    raw_root = manifest.get("_bundle_root")
    if not isinstance(raw_root, str) or not raw_root:
        return None
    root = Path(raw_root).expanduser().resolve()
    return root if root.exists() else None


def _looks_like_tokenizer_source(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "tokenizer.model",
        )
    )


def _looks_like_processor_source(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "processor_config.json",
            "preprocessor_config.json",
            "image_processor_config.json",
            "feature_extractor_config.json",
        )
    )


def _pretrained_source_candidates(
    manifest: Mapping[str, object],
    *,
    processor: bool,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str) or not value:
            return
        if value in seen:
            return
        seen.add(value)
        candidates.append(value)

    bundle_root = _bundle_root_from_manifest(manifest)
    if bundle_root is not None:
        if processor:
            if _looks_like_processor_source(bundle_root):
                add(str(bundle_root))
        elif _looks_like_tokenizer_source(bundle_root):
            add(str(bundle_root))

    model_source = str(manifest.get("model_source", "") or "")
    if model_source:
        source_path = Path(model_source).expanduser()
        if source_path.exists():
            add(str(source_path.resolve()))
        elif not source_path.is_absolute():
            add(model_source)

    add(manifest.get("model_id"))
    return candidates


def _deserialize_saved_ir_graph(
    *,
    graph_payload: Mapping[str, object],
    component_entry: Mapping[str, object],
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> IRGraph:
    values_payload = graph_payload.get("values")
    nodes_payload = graph_payload.get("nodes")
    if not isinstance(values_payload, dict) or not isinstance(nodes_payload, list):
        raise ValueError("saved optimized IR graph is missing values or nodes payload")

    values: dict[str, IRValue] = {}
    for value_id, raw_value in values_payload.items():
        if not isinstance(value_id, str) or not isinstance(raw_value, dict):
            continue
        shape = raw_value.get("shape")
        values[value_id] = IRValue(
            id=str(raw_value.get("id", value_id)),
            shape=None if shape is None else tuple(int(dim) for dim in shape),
            dtype=None if raw_value.get("dtype") is None else str(raw_value.get("dtype")),
            producer=None if raw_value.get("producer") is None else str(raw_value.get("producer")),
            users=[str(item) for item in raw_value.get("users", [])],
            meta=dict(raw_value.get("meta") or {}),
        )

    nodes: dict[str, IRNode] = {}
    order: list[str] = []
    for raw_node in nodes_payload:
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node["id"])
        node = IRNode(
            id=node_id,
            op=str(raw_node["op"]),
            inputs=[str(item) for item in raw_node.get("inputs", [])],
            outputs=[str(item) for item in raw_node.get("outputs", [])],
            attrs=dict(raw_node.get("attrs") or {}),
            meta=dict(raw_node.get("meta") or {}),
            kind=str(raw_node.get("kind", "generic")),
        )
        nodes[node_id] = node
        order.append(node_id)

    constants_payload = graph_payload.get("constants")
    if not isinstance(constants_payload, dict):
        raise ValueError("saved optimized IR graph is missing constants payload")

    constant_bindings_by_value_id = {
        str(binding.get("value_id")): binding
        for binding in component_entry.get("bound_constant_bindings", [])
        if isinstance(binding, dict) and isinstance(binding.get("value_id"), str)
    }

    constants: dict[str, object] = {}
    for value_id, serialized in constants_payload.items():
        if not isinstance(value_id, str):
            continue
        value = values.get(value_id)
        if value is None:
            continue
        constants[value_id] = _deserialize_saved_ir_constant(
            value=value,
            serialized=serialized,
            binding=constant_bindings_by_value_id.get(value_id),
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )

    return IRGraph(
        values=values,
        nodes=nodes,
        order=order,
        inputs=[str(item) for item in graph_payload.get("inputs", [])],
        outputs=[str(item) for item in graph_payload.get("outputs", [])],
        constants=constants,
        meta=dict(graph_payload.get("meta") or {}),
    )


def _deserialize_saved_ir_constant(
    *,
    value: IRValue,
    serialized: object,
    binding: Mapping[str, object] | None,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> object:
    def _torch_dtype_from_name(name: str) -> torch.dtype | None:
        return {
            "torch.bool": torch.bool,
            "torch.uint8": torch.uint8,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }.get(name)

    meta = value.meta if isinstance(value.meta, dict) else {}
    if isinstance(meta.get("path"), str) and isinstance(meta.get("kind"), str):
        # The lowerer ignores the constant payload when a weight binding is present
        # in IRValue metadata and re-attaches the mmap-backed tensor directly.
        return 0

    if isinstance(binding, Mapping):
        binding_format = str(binding.get("format", "tensor_io") or "tensor_io")
        if binding_format != "tensor_io":
            raise RuntimeError(
                f"unsupported bound constant format {binding_format!r}; re-run cactus convert to rebuild the bundle"
            )
        tensor_path = _resolve_bound_tensor_path(
            str(binding["path"]),
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )
        value.meta = {
            **meta,
            "path": str(tensor_path),
            "kind": str(binding.get("kind", "saved_constant") or "saved_constant"),
            "source_name": str(binding.get("source_name", value.id) or value.id),
        }
        return 0

    if isinstance(serialized, (str, int, float, bool)) or serialized is None:
        return serialized

    if isinstance(serialized, list):
        return serialized

    if isinstance(serialized, dict):
        value_type = str(serialized.get("type", ""))
        if value_type in {"torch.Tensor", "numpy.ndarray"}:
            shape = tuple(int(dim) for dim in serialized.get("shape", []) or [])
            if not shape:
                dtype = _torch_dtype_from_name(str(serialized.get("dtype", ""))) or torch.float32
                zero_scalar = False if dtype is torch.bool else 0
                return torch.tensor(zero_scalar, dtype=dtype)
            raise ValueError(
                "saved optimized IR is missing a materialized constant payload for "
                f"{value.id} with shape={shape}; expected a bound_constants entry"
            )
        return dict(serialized)

    return serialized


def execute_loaded_component_pipeline(
    components: list[LoadedComponentGraph],
    *,
    initial_store: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    store: dict[str, np.ndarray] = {}
    for key, value in initial_store.items():
        store[key] = _to_numpy(value)

    outputs_by_component: dict[str, list[np.ndarray]] = {}
    for component in components:
        runtime_inputs = []
        input_names = component_input_names(component)
        for input_name in input_names:
            if input_name not in store:
                raise KeyError(
                    f"component {component.component} is missing pipeline input {input_name!r}"
                )
            runtime_inputs.append(store[input_name])
        for tensor, value, input_name in zip(
            component.runtime_inputs,
            runtime_inputs,
            input_names,
            strict=True,
        ):
            expected_shape = tuple(int(dim) for dim in tensor.shape)
            actual_shape = tuple(int(dim) for dim in np.asarray(value).shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"component {component.component} input {input_name!r} shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}"
                )
        component.set_external_inputs(runtime_inputs)
        raw_outputs = component.execute()
        numpy_outputs = [output.numpy().copy() for output in raw_outputs]
        output_names = component_output_names(component)
        if len(numpy_outputs) != len(output_names):
            raise ValueError(
                f"component {component.component} produced {len(numpy_outputs)} outputs, "
                f"expected {len(output_names)}"
            )
        for output_name, value in zip(output_names, numpy_outputs, strict=True):
            store[output_name] = value
        outputs_by_component[component.component] = numpy_outputs
    return store, outputs_by_component


def execute_loaded_component(
    component: LoadedComponentGraph,
    store: dict[str, np.ndarray],
) -> list[np.ndarray]:
    runtime_inputs = []
    input_names = component_input_names(component)
    for input_name in input_names:
        if input_name not in store:
            raise KeyError(
                f"component {component.component} is missing pipeline input {input_name!r}"
            )
        runtime_inputs.append(store[input_name])
    for tensor, value, input_name in zip(
        component.runtime_inputs,
        runtime_inputs,
        input_names,
        strict=True,
    ):
        expected_shape = tuple(int(dim) for dim in tensor.shape)
        actual_shape = tuple(int(dim) for dim in np.asarray(value).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"component {component.component} input {input_name!r} shape mismatch: "
                f"expected {expected_shape}, got {actual_shape}"
            )
    component.set_external_inputs(runtime_inputs)
    raw_outputs = component.execute()
    numpy_outputs = [output.numpy().copy() for output in raw_outputs]
    output_names = component_output_names(component)
    if len(numpy_outputs) != len(output_names):
        raise ValueError(
            f"component {component.component} produced {len(numpy_outputs)} outputs, "
            f"expected {len(output_names)}"
        )
    for output_name, value in zip(output_names, numpy_outputs, strict=True):
        store[output_name] = value
    return numpy_outputs


def component_input_names(component: LoadedComponentGraph) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(component, "_input_names", ()))


def component_output_names(component: LoadedComponentGraph) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(component, "_output_names", ()))


def _zero_component_outputs(
    component: LoadedComponentGraph,
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for name, tensor in zip(component_output_names(component), component.outputs, strict=True):
        dtype = _PRECISION_TO_DTYPE.get(int(tensor.dtype), np.float16)
        shape = tuple(int(dim) for dim in tensor.shape)
        outputs[name] = np.zeros(shape, dtype=dtype)
    return outputs


def _seed_skipped_component_outputs(
    store: dict[str, np.ndarray],
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    component_names: tuple[str, ...],
) -> None:
    for component_name in component_names:
        component = component_graphs.get(component_name)
        if component is None:
            continue
        for output_name, value in _zero_component_outputs(component).items():
            store.setdefault(output_name, value)


def _bind_zero_input_buffers(
    component: LoadedComponentGraph,
    dtype_by_name: Mapping[str, np.dtype | type],
) -> dict[str, np.ndarray]:
    input_names = component_input_names(component)
    buffers: dict[str, np.ndarray] = {}
    for name, tensor in zip(input_names, component.runtime_inputs, strict=True):
        if name not in dtype_by_name:
            raise RuntimeError(f"missing dtype for {component.component} input {name!r}")
        shape = tuple(int(dim) for dim in tensor.shape)
        buffers[name] = np.zeros(shape, dtype=dtype_by_name[name])
    bound = component.set_external_inputs([buffers[name] for name in input_names])
    return {name: bound[index] for index, name in enumerate(input_names)}


def _tensor_data_ptr(tensor: Tensor) -> int:
    out_ptr = ctypes.c_void_p()
    rc = _lib.cactus_graph_get_output_ptr(tensor.g.h, cactus_node_t(tensor.id), ctypes.byref(out_ptr))
    if rc != 0 or not out_ptr.value:
        raise RuntimeError(f"graph_get_output_ptr failed for node {tensor.id}")
    return int(out_ptr.value)


def _tensor_byte_size(tensor: Tensor) -> int:
    info = tensor.g._get_output_info(tensor.id)
    return int(info["byte_size"])


def _copy_component_cache_states(
    source: LoadedComponentGraph,
    target: LoadedComponentGraph,
) -> None:
    source_states = list(source.cache_state_tensors)
    target_states = list(target.cache_state_tensors)
    if not source_states or not target_states:
        raise RuntimeError(
            f"cannot transfer KV cache from {source.component} to {target.component}: "
            "cache state metadata is missing"
        )
    if len(source_states) != len(target_states):
        raise RuntimeError(
            f"cache state count mismatch: {source.component} has {len(source_states)}, "
            f"{target.component} has {len(target_states)}"
        )
    for (source_layer, source_k, source_v), (target_layer, target_k, target_v) in zip(source_states, target_states, strict=True):
        if source_layer != target_layer:
            raise RuntimeError(f"cache layer mismatch: {source_layer!r} != {target_layer!r}")
        for source_tensor, target_tensor in ((source_k, target_k), (source_v, target_v)):
            byte_size = _tensor_byte_size(source_tensor)
            target_byte_size = _tensor_byte_size(target_tensor)
            if byte_size != target_byte_size:
                raise RuntimeError(
                    f"cache byte-size mismatch for layer {source_layer}: "
                    f"{byte_size} != {target_byte_size}"
                )
            ctypes.memmove(_tensor_data_ptr(target_tensor), _tensor_data_ptr(source_tensor), byte_size)


def _copy_gemma4_decoder_inputs(
    buffers: Mapping[str, np.ndarray],
    *,
    inputs_embeds: np.ndarray,
    per_layer_inputs: np.ndarray,
    position_ids: np.ndarray,
) -> None:
    np.copyto(
        buffers["inputs_embeds"],
        np.asarray(inputs_embeds, dtype=buffers["inputs_embeds"].dtype),
    )
    np.copyto(
        buffers["per_layer_inputs"],
        np.asarray(per_layer_inputs, dtype=buffers["per_layer_inputs"].dtype),
    )
    np.copyto(
        buffers["position_ids"],
        np.asarray(position_ids, dtype=buffers["position_ids"].dtype),
    )


def _run_parakeet_tdt_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    audio_file: str | Path,
    torch_dtype: torch.dtype,
) -> dict[str, object]:
    if "audio_encoder" not in component_graphs or "decoder" not in component_graphs:
        raise ValueError("Parakeet TDT component bundle must include audio_encoder and decoder graphs")

    inputs_meta = manifest.get("inputs") or {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("Parakeet TDT bundle manifest is missing inputs.input_shapes.input_features")

    model_source = str(manifest.get("model_source", "") or "")
    config = load_parakeet_tdt_config(model_source)
    preprocess_start = time.perf_counter()
    input_features, active_frames = prepare_parakeet_tdt_audio_features(
        audio_file=audio_file,
        expected_frames=int(expected_shape[1]),
        expected_mels=int(expected_shape[2]),
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    _attach_component_io_names(manifest, component_graphs)
    encoder_start = time.perf_counter()
    store, _ = execute_loaded_component_pipeline(
        [component_graphs["audio_encoder"]],
        initial_store={"input_features": input_features},
    )
    encoder_end = time.perf_counter()
    encoder_hidden_states = np.asarray(store["encoder_hidden_states"])
    batch_size = int(encoder_hidden_states.shape[0])
    if batch_size != 1:
        raise ValueError("Parakeet TDT saved bundle runtime currently expects batch size 1")

    state_dtype = np.float16 if torch_dtype == torch.float16 else np.float32
    initial_states: list[np.ndarray] = []
    for _ in range(config.predictor_num_layers):
        state_shape = (batch_size, config.predictor_hidden_dim)
        initial_states.append(np.zeros(state_shape, dtype=state_dtype))
        initial_states.append(np.zeros(state_shape, dtype=state_dtype))

    decoder_component = component_graphs["decoder"]
    decoder_steps = 0
    decoder_input_names = component_input_names(decoder_component)
    decoder_input_buffers: dict[str, np.ndarray] = {
        "encoder_frame": np.zeros((batch_size, int(encoder_hidden_states.shape[-1])), dtype=encoder_hidden_states.dtype),
        "token_ids": np.zeros((batch_size,), dtype=np.int64),
    }
    for index in range(config.predictor_num_layers):
        decoder_input_buffers[f"state_h_{index}"] = np.zeros(
            (batch_size, config.predictor_hidden_dim),
            dtype=state_dtype,
        )
        decoder_input_buffers[f"state_c_{index}"] = np.zeros(
            (batch_size, config.predictor_hidden_dim),
            dtype=state_dtype,
        )
    bound_decoder_inputs = decoder_component.set_external_inputs(
        [decoder_input_buffers[name] for name in decoder_input_names]
    )
    decoder_input_buffers = {
        name: bound_decoder_inputs[index] for index, name in enumerate(decoder_input_names)
    }

    def _step(
        frame: np.ndarray,
        token_id: int,
        state_values: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        nonlocal decoder_steps
        decoder_steps += 1
        np.copyto(decoder_input_buffers["encoder_frame"], np.asarray(frame, dtype=encoder_hidden_states.dtype))
        decoder_input_buffers["token_ids"].fill(int(token_id))
        for index in range(config.predictor_num_layers):
            np.copyto(decoder_input_buffers[f"state_h_{index}"], np.asarray(state_values[index * 2], dtype=state_dtype))
            np.copyto(decoder_input_buffers[f"state_c_{index}"], np.asarray(state_values[index * 2 + 1], dtype=state_dtype))
        outputs = decoder_component.execute()
        logits = outputs[0].numpy().astype(np.float32, copy=False)
        next_states = tuple(output.numpy() for output in outputs[1:])
        return logits, next_states

    decoder_start = time.perf_counter()
    emitted = greedy_decode_parakeet_tdt_token_ids(
        config=config,
        encoder_hidden_states=encoder_hidden_states,
        initial_states=tuple(initial_states),
        step=_step,
    )
    decoder_end = time.perf_counter()
    total_end = decoder_end

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "decoder_ms": (decoder_end - decoder_start) * 1000.0,
        "total_ms": (total_end - preprocess_start) * 1000.0,
        "decoder_steps": decoder_steps,
        "active_feature_frames": active_frames,
        "token_ids": emitted,
        "transcript": _decode_parakeet_tdt_token_ids(config.vocabulary, emitted),
        "encoder_hidden_shape": list(encoder_hidden_states.shape),
        "component_order": list(manifest.get("component_order", [])),
    }


def _execute_multimodal_component_pipeline_for_generation(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    required_components: tuple[str, ...],
    initial_store: dict[str, Any],
    prompt_token_count: int,
    image_files: tuple[str, ...],
    audio_file: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    store: dict[str, np.ndarray] = {
        key: _to_numpy(value)
        for key, value in initial_store.items()
    }
    outputs_by_component: dict[str, list[np.ndarray]] = {}
    family = str(manifest.get("family", "") or "").strip().lower()
    cache_key = (
        str(manifest.get("model_id", "") or manifest.get("model_source", "") or ""),
        family,
        tuple(str(path) for path in image_files),
        None if audio_file is None else str(audio_file),
    )
    cached_features = _MULTIMODAL_ENCODER_FEATURE_CACHE.get(cache_key)
    if cached_features is not None:
        for key, value in cached_features.items():
            store[key] = np.ascontiguousarray(value)

    for component_name in required_components:
        component = component_graphs[component_name]
        output_names = component_output_names(component)
        if (
            cached_features is not None
            and component_name in {"vision_encoder", "audio_encoder"}
            and output_names
            and all(output_name in cached_features for output_name in output_names)
        ):
            for output_name in output_names:
                store[output_name] = np.ascontiguousarray(cached_features[output_name])
            continue
        if component_name == "decoder":
            _right_align_decoder_inputs_to_static_tail(
                store,
                component=component,
                prompt_token_count=prompt_token_count,
            )
        outputs = execute_loaded_component(component, store)
        outputs_by_component[component_name] = outputs

        if component_name in {"vision_encoder", "audio_encoder"}:
            if all(name in store for name in output_names):
                feature_payload = _MULTIMODAL_ENCODER_FEATURE_CACHE.setdefault(cache_key, {})
                for output_name in output_names:
                    feature_payload[output_name] = np.asarray(store[output_name]).copy()

    return store, outputs_by_component


def _right_align_decoder_inputs_to_static_tail(
    store: dict[str, np.ndarray],
    *,
    component: LoadedComponentGraph,
    prompt_token_count: int,
) -> None:
    if prompt_token_count <= 0:
        return
    input_names = component_input_names(component)
    if "inputs_embeds" not in input_names:
        return
    embeds = store.get("inputs_embeds")
    if not isinstance(embeds, np.ndarray) or embeds.ndim < 2:
        return
    static_token_count = int(embeds.shape[1])
    valid_tokens = min(int(prompt_token_count), static_token_count)
    if valid_tokens <= 0 or valid_tokens == static_token_count:
        return

    for key in input_names:
        value = store.get(key)
        if not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        if int(value.shape[0]) != 1 or int(value.shape[1]) != static_token_count:
            continue
        shifted = np.zeros_like(value)
        shifted[:, static_token_count - valid_tokens :, ...] = value[:, :valid_tokens, ...]
        store[key] = np.ascontiguousarray(shifted)


def _run_multimodal_causal_lm_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str | None,
    image_files: tuple[str, ...],
    audio_file: str | Path | None,
    torch_dtype: torch.dtype,
    system_prompt: str | None,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    family = str(manifest.get("family", "") or "").strip().lower()
    use_cached_step_decode = (
        family == "gemma4"
        and "lm_encoder_step" in component_graphs
        and "decoder_step" in component_graphs
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    )
    use_chunk_prefill = use_cached_step_decode and "decoder_prefill_chunk" in component_graphs
    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    resolved_prompt = prompt if prompt is not None else str(inputs_meta.get("prompt", "") or "")
    if not resolved_prompt:
        raise ValueError("provide --prompt for multimodal causal-LM bundles")

    allow_stored_media = prompt is None
    resolved_image_files: tuple[str, ...]
    if image_files:
        resolved_image_files = tuple(str(Path(path).expanduser().resolve()) for path in image_files)
    elif allow_stored_media:
        stored_images = inputs_meta.get("image_files")
        if isinstance(stored_images, list):
            resolved_image_files = tuple(str(Path(path).expanduser().resolve()) for path in stored_images if isinstance(path, str) and path)
        else:
            resolved_image_files = ()
    else:
        resolved_image_files = ()

    resolved_audio: str | None = None
    if audio_file is not None:
        resolved_audio = str(Path(audio_file).expanduser().resolve())
    elif allow_stored_media:
        stored_audio = inputs_meta.get("audio_file")
        if isinstance(stored_audio, str) and stored_audio:
            resolved_audio = str(Path(stored_audio).expanduser().resolve())

    has_image = bool(resolved_image_files)
    has_audio = resolved_audio is not None
    if family == "gemma4" and not has_image and not has_audio:
        if not use_cached_step_decode:
            raise ValueError(
                "Gemma4 text-only execution from a multimodal bundle requires "
                "lm_encoder_step and decoder_step components; re-run cactus convert."
            )
        return _run_gemma4_text_only_cached_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=resolved_prompt,
            input_ids=None,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )

    encoder_components: list[str] = []
    skipped_encoder_components: list[str] = []
    if has_image:
        encoder_components.append("vision_encoder")
    elif "vision_encoder" in component_graphs:
        skipped_encoder_components.append("vision_encoder")
    if has_audio:
        encoder_components.append("audio_encoder")
    elif "audio_encoder" in component_graphs:
        skipped_encoder_components.append("audio_encoder")
    if family == "lfm2_vl" and not has_image:
        raise ValueError("provide --image or --image-file for LFM2-VL multimodal bundles")

    if use_cached_step_decode:
        required_components = tuple(encoder_components) + (
            ("lm_encoder", "decoder_prefill_chunk") if use_chunk_prefill else ("lm_encoder",)
        )
    else:
        required_components = tuple(encoder_components) + ("lm_encoder", "decoder")
    missing = [name for name in required_components if name not in component_graphs]
    if missing:
        raise ValueError(
            "multimodal causal LM bundle requires components "
            f"{required_components!r}, missing {missing!r}"
        )

    if family != "lfm2_vl":
        external_transformers_site_packages = ensure_transformers_supports_gemma4()
        if external_transformers_site_packages:
            print(
                "note=using external transformers install for gemma4 runtime: "
                f"{external_transformers_site_packages}"
            )
    patch_transformers_torchvision_probe()
    patch_torch_flex_attention_compat()

    from transformers import AutoProcessor

    processor_sources = _pretrained_source_candidates(manifest, processor=True)
    if not processor_sources:
        raise ValueError("bundle manifest is missing model_source/model_id for multimodal preprocessing")
    processor_errors: list[str] = []
    processor = None
    for source in processor_sources:
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                local_files_only=Path(source).exists(),
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            processor_errors.append(f"{source}: {exc}")
    if processor is None:
        raise RuntimeError(
            "failed to load tokenizer/processor assets for multimodal preprocessing. "
            "Re-run cactus convert so processor files are copied into the weights folder. "
            f"Tried: {'; '.join(processor_errors)}"
        )
    if family == "lfm2_vl":
        prepared = _prepare_lfm2_vl_multimodal_inputs_for_runtime(
            processor,
            prompt=resolved_prompt,
            image_files=resolved_image_files,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt or "",
            enable_thinking_if_supported=enable_thinking,
        )
    else:
        prepared = prepare_gemma4_multimodal_inputs(
            processor,
            prompt=resolved_prompt,
            image_files=resolved_image_files,
            audio_file=resolved_audio,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt or "",
            enable_thinking_if_supported=enable_thinking,
            use_gemma4_chat_template=True,
        )

    _attach_component_io_names(manifest, component_graphs)
    prepared_store = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(prepared.names, prepared.tensors, strict=True)
    }
    unpadded_input_ids = prepared_store.get("input_ids")
    unpadded_token_count = (
        int(unpadded_input_ids.shape[1])
        if isinstance(unpadded_input_ids, np.ndarray) and unpadded_input_ids.ndim >= 2
        else 0
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    _pad_prepared_store_to_static_input_shapes(
        prepared_store,
        inputs_meta=inputs_meta,
        tokenizer=tokenizer,
    )
    unpadded_token_count = _infer_multimodal_token_count(
        prepared_store,
        tokenizer=tokenizer,
        inputs_meta=inputs_meta,
        fallback=unpadded_token_count,
    )
    _seed_skipped_component_outputs(
        prepared_store,
        component_graphs=component_graphs,
        component_names=tuple(skipped_encoder_components),
    )
    initial_components = (
        tuple(encoder_components) + ("lm_encoder",)
        if use_cached_step_decode
        else required_components
    )

    start = time.perf_counter()
    store, _ = _execute_multimodal_component_pipeline_for_generation(
        component_graphs=component_graphs,
        manifest=manifest,
        required_components=initial_components,
        initial_store=prepared_store,
        prompt_token_count=unpadded_token_count,
        image_files=resolved_image_files,
        audio_file=resolved_audio,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"

    current_length = _infer_multimodal_token_count(
        prepared_store,
        tokenizer=tokenizer,
        inputs_meta=inputs_meta,
        fallback=unpadded_token_count,
    )
    if use_cached_step_decode:
        return _run_gemma4_cached_step_multimodal_decode(
            component_graphs=component_graphs,
            manifest=manifest,
            store=store,
            prepared_store=prepared_store,
            tokenizer=tokenizer,
            prompt=resolved_prompt,
            image_files=resolved_image_files,
            audio_file=resolved_audio,
            current_length=current_length,
            start=start,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )

    static_token_count = _static_multimodal_token_count(prepared_store)
    available_headroom = max(0, static_token_count - current_length)
    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )
    token_budget = max(0, min(requested_tokens, max(1, available_headroom + 1)))
    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)

    for step_index in range(token_budget):
        logits = np.asarray(store["logits"], dtype=np.float32)
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        next_token_id = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - start) * 1000.0

        if next_token_id in stop_token_ids:
            stop_reason = "stop_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break
        if step_index + 1 >= token_budget:
            break
        if current_length >= static_token_count:
            stop_reason = "context_limit"
            break

        _append_multimodal_token_in_place(store, current_length=current_length, token_id=next_token_id)
        current_length += 1
        store, _ = _execute_multimodal_component_pipeline_for_generation(
            component_graphs=component_graphs,
            manifest=manifest,
            required_components=("lm_encoder", "decoder"),
            initial_store=store,
            prompt_token_count=current_length,
            image_files=resolved_image_files,
            audio_file=resolved_audio,
        )

    end = time.perf_counter()
    response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False).strip()
    decode_time_ms = max(0.0, (end - start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )
    first_generated_token_id = generated_ids[0] if generated_ids else None
    first_generated_token = (
        _decode_generated_text(tokenizer, [first_generated_token_id], skip_special_tokens=False)
        if first_generated_token_id is not None
        else None
    )

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "prompt": resolved_prompt,
        "image_files": list(resolved_image_files),
        "audio_file": resolved_audio,
        "input_shapes": {
            name: list(np.asarray(value).shape)
            for name, value in prepared_store.items()
        },
        "output_shape": logits_shape or [],
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
        "next_token_id": first_generated_token_id,
        "next_token": first_generated_token,
    }


def _run_gemma4_text_only_cached_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    if "lm_encoder_step" not in component_graphs or "decoder_step" not in component_graphs:
        raise ValueError("Gemma4 text-only multimodal bundle requires lm_encoder_step and decoder_step")

    prompt_token_ids, tokenizer = _resolve_causal_lm_input_ids(
        manifest=manifest,
        prompt=prompt,
        input_ids=input_ids,
        enable_thinking=enable_thinking,
    )
    if not prompt_token_ids:
        raise ValueError("Gemma4 text-only bundle input token ids are empty")
    if tokenizer is None:
        try:
            tokenizer = _load_bundle_tokenizer(manifest)
        except Exception:
            tokenizer = None

    _attach_component_io_names(manifest, component_graphs)
    lm_encoder_step = component_graphs["lm_encoder_step"]
    decoder_step = component_graphs["decoder_step"]
    lm_encoder_input_buffers = _bind_zero_input_buffers(
        lm_encoder_step,
        {"input_ids": np.int64, "position_ids": np.int64},
    )
    decoder_dtypes = {
        name: _PRECISION_TO_DTYPE.get(int(tensor.dtype), np.float16)
        for name, tensor in zip(component_input_names(decoder_step), decoder_step.runtime_inputs, strict=True)
    }
    decoder_input_buffers = _bind_zero_input_buffers(decoder_step, decoder_dtypes)

    def _run_step_token(token_id: int, position_id: int, *, read_logits: bool) -> np.ndarray | None:
        lm_encoder_input_buffers["input_ids"].fill(int(token_id))
        lm_encoder_input_buffers["position_ids"].fill(int(position_id))
        lm_encoder_step.graph.execute()
        _copy_gemma4_decoder_inputs(
            decoder_input_buffers,
            inputs_embeds=lm_encoder_step.outputs[0].numpy(),
            per_layer_inputs=lm_encoder_step.outputs[1].numpy(),
            position_ids=lm_encoder_step.outputs[2].numpy(),
        )
        decoder_step.graph.execute()
        if not read_logits:
            return None
        return np.asarray(decoder_step.outputs[0].numpy())

    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )
    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"
    logits: np.ndarray | None = None

    start = time.perf_counter()
    for position_id, token_id in enumerate(prompt_token_ids):
        logits = _run_step_token(
            int(token_id),
            int(position_id),
            read_logits=position_id + 1 == len(prompt_token_ids),
        )
    if logits is None:
        raise RuntimeError("Gemma4 text-only cached decoder did not produce logits")

    token_position = len(prompt_token_ids)
    for step_index in range(requested_tokens):
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        next_token_id = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - start) * 1000.0

        if next_token_id in stop_token_ids:
            stop_reason = "stop_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break
        if step_index + 1 >= requested_tokens:
            if max_new_tokens is None:
                stop_reason = "generation_guard"
            break

        logits = _run_step_token(
            next_token_id,
            token_position,
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("Gemma4 text-only cached decoder did not produce decode logits")
        token_position += 1

    end = time.perf_counter()
    response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False).strip()
    decode_time_ms = max(0.0, (end - start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )
    prefill_tps = (
        (len(prompt_token_ids) * 1000.0) / first_token_ms
        if first_token_ms > 0.0
        else 0.0
    )
    first_generated_token_id = generated_ids[0] if generated_ids else None
    first_generated_token = (
        _decode_generated_text(tokenizer, [first_generated_token_id], skip_special_tokens=False)
        if first_generated_token_id is not None
        else None
    )
    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "prompt": prompt,
        "input_ids": prompt_token_ids,
        "output_shape": logits_shape or [],
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "total_tokens": len(prompt_token_ids) + len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
        "next_token_id": first_generated_token_id,
        "next_token": first_generated_token,
        "decode_mode": "cached_step_text",
    }


def _run_gemma4_cached_step_multimodal_decode(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    store: dict[str, np.ndarray],
    prepared_store: dict[str, np.ndarray],
    tokenizer: object,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    current_length: int,
    start: float,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    """Run Gemma4 generation through a one-token decoder graph with internal KV state.

    The full-context decoder recomputes the entire text stack for every token.
    The chunk graph primes K/V in fixed windows, then the step graph appends one
    token at a time through the cached attention kernel.
    """

    lm_encoder_step = component_graphs["lm_encoder_step"]
    decoder_step = component_graphs["decoder_step"]
    decoder_prefill_chunk = component_graphs.get("decoder_prefill_chunk")
    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )

    inputs_embeds = np.asarray(store["inputs_embeds"])
    per_layer_inputs = np.asarray(store["per_layer_inputs"])
    position_ids = np.asarray(store["position_ids"])
    prompt_tokens = max(0, min(int(current_length), int(inputs_embeds.shape[1])))
    if prompt_tokens <= 0:
        raise RuntimeError("Gemma4 cached decode requires at least one prompt token")

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"

    decoder_dtypes = {
        "inputs_embeds": inputs_embeds.dtype,
        "per_layer_inputs": per_layer_inputs.dtype,
        "position_ids": position_ids.dtype,
    }
    decoder_input_buffers = _bind_zero_input_buffers(decoder_step, decoder_dtypes)

    def _run_decoder_step(
        *,
        step_inputs_embeds: np.ndarray,
        step_per_layer_inputs: np.ndarray,
        step_position_ids: np.ndarray,
        read_logits: bool,
    ) -> np.ndarray | None:
        _copy_gemma4_decoder_inputs(
            decoder_input_buffers,
            inputs_embeds=step_inputs_embeds,
            per_layer_inputs=step_per_layer_inputs,
            position_ids=step_position_ids,
        )
        decoder_step.graph.execute()
        if not read_logits:
            return None
        return np.asarray(decoder_step.outputs[0].numpy())

    lm_encoder_input_buffers = _bind_zero_input_buffers(
        lm_encoder_step,
        {"input_ids": np.int64, "position_ids": np.int64},
    )

    prefill_input_buffers: dict[str, np.ndarray] | None = None
    prefill_chunk_tokens = 0
    if decoder_prefill_chunk is not None:
        prefill_input_buffers = _bind_zero_input_buffers(decoder_prefill_chunk, decoder_dtypes)
        prefill_chunk_tokens = int(prefill_input_buffers["inputs_embeds"].shape[1])

    logits: np.ndarray | None = None
    prime_start = time.perf_counter()
    token_index = 0
    if (
        decoder_prefill_chunk is not None
        and prefill_input_buffers is not None
        and prefill_chunk_tokens > 1
        and prompt_tokens >= prefill_chunk_tokens
    ):
        chunked_tokens = (prompt_tokens // prefill_chunk_tokens) * prefill_chunk_tokens
        for chunk_start in range(0, chunked_tokens, prefill_chunk_tokens):
            chunk_end = chunk_start + prefill_chunk_tokens
            _copy_gemma4_decoder_inputs(
                prefill_input_buffers,
                inputs_embeds=inputs_embeds[:, chunk_start:chunk_end, :],
                per_layer_inputs=per_layer_inputs[:, chunk_start:chunk_end, ...],
                position_ids=position_ids[:, chunk_start:chunk_end],
            )
            decoder_prefill_chunk.graph.execute()
            if chunk_end == prompt_tokens:
                logits = np.asarray(decoder_prefill_chunk.outputs[0].numpy())
        _copy_component_cache_states(decoder_prefill_chunk, decoder_step)
        token_index = chunked_tokens

    while token_index < prompt_tokens:
        logits = _run_decoder_step(
            step_inputs_embeds=inputs_embeds[:, token_index : token_index + 1, :],
            step_per_layer_inputs=per_layer_inputs[:, token_index : token_index + 1, ...],
            step_position_ids=position_ids[:, token_index : token_index + 1],
            read_logits=token_index + 1 == prompt_tokens,
        )
        token_index += 1
    prime_end = time.perf_counter()

    if logits is None:
        raise RuntimeError("Gemma4 cached decoder did not produce logits")

    token_position = prompt_tokens
    for step_index in range(requested_tokens):
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        next_token_id = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - start) * 1000.0

        if next_token_id in stop_token_ids:
            stop_reason = "stop_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break
        if step_index + 1 >= requested_tokens:
            if max_new_tokens is None:
                stop_reason = "generation_guard"
            break

        lm_encoder_input_buffers["input_ids"].fill(int(next_token_id))
        lm_encoder_input_buffers["position_ids"].fill(int(token_position))
        lm_encoder_step.graph.execute()
        logits = _run_decoder_step(
            step_inputs_embeds=lm_encoder_step.outputs[0].numpy(),
            step_per_layer_inputs=lm_encoder_step.outputs[1].numpy(),
            step_position_ids=lm_encoder_step.outputs[2].numpy(),
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("Gemma4 cached decoder did not produce logits during decode")
        token_position += 1

    end = time.perf_counter()
    response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False).strip()
    decode_time_ms = max(0.0, (end - start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )
    first_generated_token_id = generated_ids[0] if generated_ids else None
    first_generated_token = (
        _decode_generated_text(tokenizer, [first_generated_token_id], skip_special_tokens=False)
        if first_generated_token_id is not None
        else None
    )

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "prompt": prompt,
        "image_files": list(image_files),
        "audio_file": audio_file,
        "input_shapes": {
            name: list(np.asarray(value).shape)
            for name, value in prepared_store.items()
        },
        "output_shape": logits_shape or [],
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "cache_prime_ms": (prime_end - prime_start) * 1000.0,
        "cache_prime_tokens": prompt_tokens,
        "cache_prime_chunk_tokens": prefill_chunk_tokens,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
        "next_token_id": first_generated_token_id,
        "next_token": first_generated_token,
        "decode_mode": "cached_step",
    }


def _static_multimodal_token_count(store: Mapping[str, np.ndarray]) -> int:
    for key in ("input_ids", "attention_mask", "token_type_ids", "inputs_embeds"):
        value = store.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            return int(value.shape[1])
    return 0


def _infer_multimodal_token_count(
    store: Mapping[str, np.ndarray],
    *,
    tokenizer: object | None,
    inputs_meta: Mapping[str, object],
    fallback: int,
) -> int:
    attention_mask = store.get("attention_mask")
    if isinstance(attention_mask, np.ndarray) and attention_mask.ndim >= 2:
        return int(np.count_nonzero(attention_mask[0]))
    input_ids = store.get("input_ids")
    if isinstance(input_ids, np.ndarray) and input_ids.ndim >= 2:
        padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
        return int(np.count_nonzero(input_ids[0] != int(padding_token_id)))
    return max(0, int(fallback))


def _append_multimodal_token_in_place(
    store: dict[str, np.ndarray],
    *,
    current_length: int,
    token_id: int,
) -> None:
    input_ids = store.get("input_ids")
    if isinstance(input_ids, np.ndarray) and input_ids.ndim >= 2 and current_length < input_ids.shape[1]:
        input_ids[0, current_length] = int(token_id)
    attention_mask = store.get("attention_mask")
    if isinstance(attention_mask, np.ndarray) and attention_mask.ndim >= 2 and current_length < attention_mask.shape[1]:
        attention_mask[0, current_length] = 1
    token_type_ids = store.get("token_type_ids")
    if isinstance(token_type_ids, np.ndarray) and token_type_ids.ndim >= 2 and current_length < token_type_ids.shape[1]:
        token_type_ids[0, current_length] = 0


def _load_image_inputs_for_runtime(image_files: tuple[str, ...]) -> list[object]:
    if not image_files:
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for --image: {exc}") from exc

    images: list[object] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        if not path.exists():
            raise RuntimeError(f"image file does not exist: {path}")
        with Image.open(path) as image:
            images.append(resize_static_image(image.convert("RGB")).copy())
    return images


def _prepare_lfm2_vl_multimodal_inputs_for_runtime(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
):
    if processor is None:
        raise RuntimeError("LFM2-VL multimodal bundle execution requires an AutoProcessor with image support")
    images = _load_image_inputs_for_runtime(image_files)
    if not images:
        raise RuntimeError("LFM2-VL multimodal bundle execution requires at least one --image")

    user_content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": prompt.strip()})

    messages: list[dict[str, object]] = []
    normalized_system = system_prompt.strip()
    if normalized_system:
        messages.append({"role": "system", "content": normalized_system})
    messages.append({"role": "user", "content": user_content})

    apply_chat_template = getattr(processor, "apply_chat_template", None)
    image_placeholders = "\n".join("<image>" for _ in images)
    fallback_text = (
        f"<|startoftext|><|im_start|>user\n"
        f"{image_placeholders}\n"
        f"{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
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

    names = ("input_ids", "attention_mask", "pixel_values", "spatial_shapes", "pixel_attention_mask")
    tensors: list[torch.Tensor] = []
    input_shapes: dict[str, list[int]] = {}
    for key in names:
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"LFM2-VL processor did not return required tensor input: {key}")
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        elif key == "pixel_attention_mask":
            value = value.to(dtype=torch.int64)
        else:
            value = value.to(dtype=torch.long)
        tensors.append(value)
        input_shapes[key] = [int(dim) for dim in value.shape]

    return PreparedInputs(
        names=names,
        tensors=tuple(tensors),
        metadata={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "image_files": [str(Path(path).resolve()) for path in image_files],
            "input_shapes": input_shapes,
            "enable_thinking": bool(enable_thinking_if_supported),
        },
    )


def _run_causal_lm_logits_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    if "decoder" not in component_graphs:
        raise ValueError("causal LM component bundle must include a decoder graph")

    prompt_token_ids, tokenizer = _resolve_causal_lm_input_ids(
        manifest=manifest,
        prompt=prompt,
        input_ids=input_ids,
        enable_thinking=enable_thinking,
    )
    if not prompt_token_ids:
        raise ValueError("causal LM bundle input token ids are empty")
    if tokenizer is None:
        try:
            tokenizer = _load_bundle_tokenizer(manifest)
        except Exception:
            tokenizer = None

    _attach_component_io_names(manifest, component_graphs)
    decoder = component_graphs["decoder"]
    runtime_inputs = component_input_names(decoder)
    if runtime_inputs and runtime_inputs != ("input_ids",):
        raise ValueError(
            "causal LM bundle runner currently expects decoder logical input ('input_ids',), "
            f"got {runtime_inputs!r}"
        )

    inputs_meta = manifest.get("inputs")
    if not isinstance(inputs_meta, dict):
        inputs_meta = {}
    stored_input_ids = _parse_nested_manifest_input_ids(inputs_meta.get("input_ids")) or []
    stored_target_token_count = int(inputs_meta.get("target_token_count", 0) or 0)
    target_token_count = max(
        len(prompt_token_ids),
        stored_target_token_count,
        len(stored_input_ids),
    )
    if target_token_count <= 0:
        raise ValueError("causal LM bundle manifest did not provide a valid target token count")
    if len(prompt_token_ids) > target_token_count:
        raise ValueError(
            f"prompt token length {len(prompt_token_ids)} exceeds transpiled bundle context {target_token_count}; "
            "re-transpile with a larger --max-new-tokens budget or use a shorter prompt"
        )

    padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
    input_array = np.full((1, target_token_count), padding_token_id, dtype=np.int64)
    input_array[0, : len(prompt_token_ids)] = np.asarray(prompt_token_ids, dtype=np.int64)
    input_array = decoder.set_external_input(0, input_array)

    available_headroom = max(0, target_token_count - len(prompt_token_ids))
    if max_new_tokens is None:
        token_budget = available_headroom if available_headroom > 0 else 1
    else:
        requested = max(0, int(max_new_tokens))
        if available_headroom > 0:
            token_budget = min(requested, available_headroom)
        else:
            token_budget = 1 if requested > 0 else 0

    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    current_length = len(prompt_token_ids)
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"
    start = time.perf_counter()

    for step_index in range(token_budget):
        outputs = decoder.execute()
        if not outputs:
            raise RuntimeError("causal LM decoder graph produced no outputs")
        logits = outputs[0].numpy()
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        token_position = current_length - 1
        if logits.shape[1] == 1:
            token_position = 0
        next_token_id = int(np.argmax(logits[0, token_position]))
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - start) * 1000.0

        if next_token_id in stop_token_ids:
            stop_reason = "stop_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break

        if current_length >= target_token_count:
            stop_reason = "context_limit"
            break
        if step_index + 1 >= token_budget:
            break

        input_array[0, current_length] = next_token_id
        current_length += 1

    end = time.perf_counter()
    response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not response:
        response = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False).strip()
    decode_time_ms = max(0.0, (end - start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )
    prefill_tps = (
        (len(prompt_token_ids) * 1000.0) / first_token_ms
        if first_token_ms > 0.0
        else 0.0
    )
    first_generated_token_id = generated_ids[0] if generated_ids else None
    first_generated_token = None
    if first_generated_token_id is not None:
        first_generated_token = _decode_generated_text(
            tokenizer,
            [int(first_generated_token_id)],
            skip_special_tokens=False,
        )

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "input_ids": prompt_token_ids,
        "input_shape": list(input_array.shape),
        "output_shape": logits_shape or [],
        "decoder_ms": (end - start) * 1000.0,
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "total_tokens": len(prompt_token_ids) + len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
        "next_token_id": first_generated_token_id,
        "next_token": first_generated_token,
    }


def _run_seq2seq_transcription_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    audio_file: str | Path,
    prompt: str | None,
    torch_dtype: torch.dtype,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    if "audio_encoder" not in component_graphs or "decoder" not in component_graphs:
        raise ValueError("seq2seq_transcription bundle must include audio_encoder and decoder graphs")

    inputs_meta = manifest.get("inputs")
    if not isinstance(inputs_meta, dict):
        inputs_meta = {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("seq2seq_transcription bundle manifest is missing inputs.input_shapes.input_features")

    tokenizer = None
    try:
        tokenizer = _load_bundle_tokenizer(manifest)
    except Exception:
        tokenizer = None

    prompt_token_ids = _resolve_seq2seq_prompt_token_ids(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
    )
    if not prompt_token_ids:
        raise ValueError("seq2seq_transcription bundle input token ids are empty")

    _attach_component_io_names(manifest, component_graphs)
    encoder = component_graphs["audio_encoder"]
    decoder = component_graphs["decoder"]
    encoder_inputs = component_input_names(encoder)
    decoder_inputs = component_input_names(decoder)
    if encoder_inputs and encoder_inputs != ("input_features",):
        raise ValueError(
            "seq2seq_transcription audio_encoder must accept logical input ('input_features',), "
            f"got {encoder_inputs!r}"
        )
    if decoder_inputs and decoder_inputs != ("decoder_input_ids", "encoder_hidden_states"):
        raise ValueError(
            "seq2seq_transcription decoder must accept logical inputs "
            "('decoder_input_ids', 'encoder_hidden_states'), "
            f"got {decoder_inputs!r}"
        )

    preprocess_start = time.perf_counter()
    input_features, active_frames = _prepare_generic_audio_encoder_features(
        audio_file=audio_file,
        manifest=manifest,
        expected_shape=expected_shape,
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    encoder_start = time.perf_counter()
    encoder.set_inputs([input_features])
    encoder_outputs = encoder.execute()
    encoder_end = time.perf_counter()
    if not encoder_outputs:
        raise RuntimeError("seq2seq_transcription encoder graph produced no outputs")
    encoder_hidden_states = np.asarray(encoder_outputs[0].numpy())

    stored_target_token_count = int(inputs_meta.get("target_token_count", 0) or 0)
    target_token_count = max(stored_target_token_count, len(prompt_token_ids))
    if target_token_count <= 0:
        raise ValueError("seq2seq_transcription bundle manifest did not provide a valid target token count")
    if len(prompt_token_ids) > target_token_count:
        raise ValueError(
            f"prompt token length {len(prompt_token_ids)} exceeds transpiled bundle context {target_token_count}; "
            "re-transpile with a larger --max-new-tokens budget or use a shorter prompt"
        )

    padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
    input_array = np.full((1, target_token_count), padding_token_id, dtype=np.int64)
    input_array[0, : len(prompt_token_ids)] = np.asarray(prompt_token_ids, dtype=np.int64)
    if hasattr(decoder, "set_external_inputs"):
        bound_decoder_inputs = decoder.set_external_inputs([input_array, encoder_hidden_states])
        input_array = bound_decoder_inputs[0]
        encoder_hidden_states = bound_decoder_inputs[1]
    else:
        decoder.set_inputs([input_array, encoder_hidden_states])

    available_headroom = max(0, target_token_count - len(prompt_token_ids))
    if max_new_tokens is None:
        token_budget = available_headroom if available_headroom > 0 else 1
    else:
        requested = max(0, int(max_new_tokens))
        if available_headroom > 0:
            token_budget = min(requested, available_headroom)
        else:
            token_budget = 1 if requested > 0 else 0

    default_stop_sequences = ("<|endoftext|>", "<|endoftranscript|>", "</s>", "<pad>")
    resolved_stop_sequences = stop_sequences or default_stop_sequences
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, resolved_stop_sequences)
    eos_token_id = inputs_meta.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
    suppress_tokens = [int(value) for value in inputs_meta.get("suppress_tokens", []) if isinstance(value, int)]
    begin_suppress_tokens = [int(value) for value in inputs_meta.get("begin_suppress_tokens", []) if isinstance(value, int)]
    if tokenizer is not None and (
        "whisper" in str(manifest.get("family", "") or "").lower()
        or "whisper" in str(manifest.get("model_id", "") or "").lower()
    ):
        eos_int = int(eos_token_id) if isinstance(eos_token_id, int) else None
        whisper_special_ids = [
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", []) or []
            if eos_int is None or int(token_id) != eos_int
        ]
        suppress_tokens = sorted(set(suppress_tokens).union(whisper_special_ids))

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    current_length = len(prompt_token_ids)
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"
    decoder_start = time.perf_counter()

    for step_index in range(token_budget):
        outputs = decoder.execute()
        if not outputs:
            raise RuntimeError("seq2seq_transcription decoder graph produced no outputs")
        logits = outputs[0].numpy()
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected decoder logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        token_position = current_length - 1
        if logits.shape[1] == 1:
            token_position = 0
        next_token_id = _select_next_token_with_suppression(
            np.asarray(logits[0, token_position]),
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens if step_index == 0 else (),
        )
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - decoder_start) * 1000.0

        if eos_token_id is not None and next_token_id == int(eos_token_id):
            stop_reason = "eos_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break
        if current_length >= target_token_count:
            stop_reason = "context_limit"
            break
        if step_index + 1 >= token_budget:
            break

        input_array[0, current_length] = next_token_id
        current_length += 1

    decoder_end = time.perf_counter()
    transcript = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not transcript:
        transcript = _strip_whisper_control_tokens(
            _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False)
        ).strip()
    decode_time_ms = max(0.0, (decoder_end - decoder_start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "component_order": list(manifest.get("component_order", [])),
        "active_feature_frames": active_frames,
        "input_shape": list(input_features.shape),
        "encoder_hidden_shape": list(encoder_hidden_states.shape),
        "output_shape": logits_shape or [],
        "input_ids": prompt_token_ids,
        "generated_token_ids": generated_ids,
        "transcript": transcript,
        "response": transcript,
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "decoder_ms": (decoder_end - decoder_start) * 1000.0,
        "total_ms": (decoder_end - preprocess_start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "stop_reason": stop_reason,
    }


def _run_encoder_hidden_states_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    audio_file: str | Path,
    torch_dtype: torch.dtype,
) -> dict[str, object]:
    component_name = "audio_encoder" if "audio_encoder" in component_graphs else "encoder"
    if component_name not in component_graphs:
        raise ValueError("encoder_hidden_states bundle must include an audio_encoder or encoder graph")

    inputs_meta = manifest.get("inputs") or {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("encoder bundle manifest is missing inputs.input_shapes.input_features")

    preprocess_start = time.perf_counter()
    input_features, active_frames = _prepare_generic_audio_encoder_features(
        audio_file=audio_file,
        manifest=manifest,
        expected_shape=expected_shape,
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    _attach_component_io_names(manifest, component_graphs)
    encoder = component_graphs[component_name]
    encoder_start = time.perf_counter()
    encoder.set_inputs([input_features])
    outputs = encoder.execute()
    encoder_end = time.perf_counter()
    if not outputs:
        raise RuntimeError("encoder graph produced no outputs")
    hidden = outputs[0].numpy()

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "component_order": list(manifest.get("component_order", [])),
        "active_feature_frames": active_frames,
        "input_shape": list(input_features.shape),
        "encoder_hidden_shape": list(hidden.shape),
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "total_ms": (encoder_end - preprocess_start) * 1000.0,
    }


def _prepare_generic_audio_encoder_features(
    *,
    audio_file: str | Path,
    manifest: dict[str, object],
    expected_shape: list[object],
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, int]:
    family = str(manifest.get("family", "") or "")
    family_lower = family.strip().lower()
    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    sample_rate = int(inputs_meta.get("sample_rate", 16000) if isinstance(inputs_meta, dict) else 16000)
    batch = int(expected_shape[0])
    if batch != 1:
        raise ValueError("saved audio encoder bundle runtime currently expects batch size 1")

    if "whisper" in family_lower:
        expected_mels = int(expected_shape[1])
        expected_frames = int(expected_shape[2])
        try:
            features, active_frames = prepare_cactus_audio_features(
                audio_file,
                model_type="whisper",
                expected_frames=expected_frames,
                expected_mels=expected_mels,
                torch_dtype=torch_dtype,
                layout="mels_frames",
            )
            return np.ascontiguousarray(features.detach().cpu().numpy()), active_frames
        except Exception:
            pass
    else:
        expected_frames = int(expected_shape[1])
        expected_mels = int(expected_shape[2])

    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)
    features, feature_length = _generic_log_mel_features(
        waveform,
        sample_rate=sample_rate,
        num_mels=expected_mels,
        n_fft=400,
        hop_length=160,
        frame_length=400,
        preemphasis=None,
    )
    active_frames = min(feature_length, expected_frames)
    features = features[:active_frames, :]
    if expected_frames > active_frames:
        features = np.pad(features, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    if "whisper" in family_lower:
        features = np.ascontiguousarray(features.T)
    features = np.ascontiguousarray(features, dtype=np.float16 if torch_dtype == torch.float16 else np.float32)
    return np.expand_dims(features, axis=0), active_frames


def _resolve_causal_lm_input_ids(
    *,
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
    enable_thinking: bool = False,
) -> tuple[list[int], object | None]:
    if input_ids is not None:
        return _parse_input_ids(input_ids), None

    if prompt is None:
        inputs_meta = manifest.get("inputs")
        if isinstance(inputs_meta, dict):
            stored_prompt_ids = inputs_meta.get("prompt_input_ids")
            parsed_prompt_ids = _parse_nested_manifest_input_ids(stored_prompt_ids)
            if parsed_prompt_ids:
                return parsed_prompt_ids, None
            stored_ids = inputs_meta.get("input_ids")
            parsed = _parse_nested_manifest_input_ids(stored_ids)
            if parsed:
                return parsed, None
            stored_prompt = inputs_meta.get("prompt")
            if isinstance(stored_prompt, str) and stored_prompt:
                prompt = stored_prompt
    if not prompt:
        raise ValueError("provide --input-ids or --prompt for causal LM component bundles")

    tokenizer = _load_bundle_tokenizer(manifest)
    token_ids = _tokenize_bundle_prompt_for_manifest(
        manifest,
        tokenizer,
        prompt,
        enable_thinking_if_supported=enable_thinking,
    )
    return token_ids, tokenizer


def _parse_input_ids(input_ids: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(input_ids, str):
        parsed = [int(part.strip()) for part in input_ids.split(",") if part.strip()]
    else:
        parsed = [int(value) for value in input_ids]
    if not parsed:
        raise ValueError("input_ids was provided but no token ids were parsed")
    return parsed


def _parse_nested_manifest_input_ids(value: object) -> list[int] | None:
    if isinstance(value, list) and value:
        if all(isinstance(item, int) for item in value):
            return [int(item) for item in value]
        first = value[0]
        if isinstance(first, list) and all(isinstance(item, int) for item in first):
            return [int(item) for item in first]
    return None


def _patch_missing_lzma_backport() -> str | None:
    try:
        import importlib.util
        import sys

        if importlib.util.find_spec("_lzma") is not None:
            return None
        if importlib.util.find_spec("backports.lzma") is None:
            return None
        import backports.lzma as backports_lzma  # type: ignore

        sys.modules.setdefault("lzma", backports_lzma)
        return "using backports.lzma because this Python build is missing _lzma"
    except Exception:
        return None


def _load_bundle_tokenizer(manifest: dict[str, object]):
    _patch_missing_lzma_backport()
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required to tokenize --prompt: {exc}") from exc

    tokenizer_sources = _pretrained_source_candidates(manifest, processor=False)
    if not tokenizer_sources:
        raise ValueError("bundle manifest is missing model_source/model_id; provide --input-ids instead")
    errors: list[str] = []
    for source in tokenizer_sources:
        try:
            return AutoTokenizer.from_pretrained(
                source,
                local_files_only=Path(source).exists(),
                trust_remote_code=True,
            )
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError(
        "failed to load tokenizer assets for prompt tokenization. "
        "The CQ weights are present, but tokenizer files are also required for text prompts. "
        f"Tried: {'; '.join(errors)}"
    )


def _tokenize_bundle_prompt(
    tokenizer: object,
    prompt: str,
    *,
    enable_thinking_if_supported: bool = False,
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            encoded = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=bool(enable_thinking_if_supported),
            )
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(value) for value in ids]
        except Exception:
            pass

    encoded = tokenizer(prompt, return_tensors=None)  # type: ignore[operator]
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _tokenize_bundle_prompt_for_manifest(
    manifest: Mapping[str, object],
    tokenizer: object,
    prompt: str,
    *,
    enable_thinking_if_supported: bool = False,
) -> list[int]:
    family = str(manifest.get("family", "") or "").strip().lower()
    if family in {"qwen", "qwen3", "qwen3_5", "qwen3.5"}:
        return _encode_prompt_text(
            tokenizer,
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
    if family in {"lfm2", "lfm2_vl", "lfm"}:
        return _encode_prompt_text(
            tokenizer,
            f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
    if family in {"gemma4", "gemma"}:
        return _encode_prompt_text(
            tokenizer,
            _build_gemma4_chat_prompt(
                prompt=prompt,
                image_token=None,
                num_images=0,
                audio_token=None,
                num_audio_segments=0,
                enable_thinking_if_supported=enable_thinking_if_supported,
            ),
        )
    return _tokenize_bundle_prompt(
        tokenizer,
        prompt,
        enable_thinking_if_supported=enable_thinking_if_supported,
    )


def _encode_prompt_text(tokenizer: object, prompt_text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return [int(value) for value in encode(prompt_text, add_special_tokens=False)]
        except TypeError:
            return [int(value) for value in encode(prompt_text)]
    encoded = tokenizer(prompt_text, return_tensors=None, add_special_tokens=False)  # type: ignore[operator]
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _resolve_bundle_padding_token_id(inputs_meta: Mapping[str, object] | None, tokenizer: object | None) -> int:
    if isinstance(inputs_meta, Mapping):
        value = inputs_meta.get("padding_token_id")
        if isinstance(value, int) and value >= 0:
            return int(value)
    for attr_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr_name, None) if tokenizer is not None else None
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)
    return 0


def _static_input_pad_value(name: str, *, padding_token_id: int) -> int | float:
    normalized = name.strip().lower()
    if normalized in {"input_ids", "decoder_input_ids"}:
        return int(padding_token_id)
    if normalized.endswith("position_ids") and "pixel" in normalized:
        return -1
    if normalized.endswith("mask") or normalized in {"attention_mask", "token_type_ids"}:
        return 0
    return 0.0


def _pad_prepared_store_to_static_input_shapes(
    prepared_store: dict[str, np.ndarray],
    *,
    inputs_meta: Mapping[str, object],
    tokenizer: object | None,
) -> None:
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, Mapping) else None
    if not isinstance(input_shapes, Mapping):
        return

    padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
    for name, raw_target_shape in input_shapes.items():
        if not isinstance(name, str):
            continue
        value = prepared_store.get(name)
        if not isinstance(value, np.ndarray):
            continue
        if not isinstance(raw_target_shape, (list, tuple)):
            continue
        target_shape = tuple(int(dim) for dim in raw_target_shape)
        if tuple(int(dim) for dim in value.shape) == target_shape:
            continue
        if value.ndim != len(target_shape):
            raise ValueError(
                f"{name} rank {value.ndim} does not match transpiled bundle input rank {len(target_shape)}; "
                "re-transpile with representative inputs for this model."
            )
        if any(int(current) > int(target) for current, target in zip(value.shape, target_shape, strict=True)):
            if (
                name in {"input_ids", "attention_mask", "token_type_ids", "decoder_input_ids"}
                and value.ndim == 2
                and len(target_shape) == 2
                and int(value.shape[0]) == int(target_shape[0])
                and int(value.shape[1]) > int(target_shape[1])
            ):
                value = np.ascontiguousarray(value[:, -int(target_shape[1]) :])
                prepared_store[name] = value
            elif (
                name in {"input_features", "input_features_mask"}
                and value.ndim == len(target_shape)
                and len(target_shape) >= 2
                and int(value.shape[0]) == int(target_shape[0])
                and int(value.shape[1]) > int(target_shape[1])
                and all(
                    int(current) <= int(target)
                    for current, target in zip(value.shape[2:], target_shape[2:], strict=True)
                )
            ):
                slices = [slice(None)] * value.ndim
                slices[1] = slice(0, int(target_shape[1]))
                value = np.ascontiguousarray(value[tuple(slices)])
                prepared_store[name] = value
            else:
                raise ValueError(
                    f"{name} shape {list(value.shape)} exceeds transpiled bundle input shape {list(target_shape)}; "
                    "re-transpile with a longer representative prompt/media sample."
                )
        if tuple(int(dim) for dim in value.shape) == target_shape:
            continue
        if any(int(current) > int(target) for current, target in zip(value.shape, target_shape, strict=True)):
            raise ValueError(
                f"{name} shape {list(value.shape)} exceeds transpiled bundle input shape {list(target_shape)}; "
                "re-transpile with a longer representative prompt/media sample."
            )

        pad_value = _static_input_pad_value(name, padding_token_id=padding_token_id)
        padded = np.full(target_shape, pad_value, dtype=value.dtype)
        copy_slices = tuple(slice(0, int(dim)) for dim in value.shape)
        padded[copy_slices] = value
        prepared_store[name] = np.ascontiguousarray(padded)


def _encode_stop_sequences(tokenizer: object | None, stop_sequences: tuple[str, ...]) -> list[list[int]]:
    if tokenizer is None or not stop_sequences:
        return []
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return []
    encoded: list[list[int]] = []
    for stop_sequence in stop_sequences:
        try:
            token_ids = list(encode(stop_sequence, add_special_tokens=False))
        except TypeError:
            token_ids = list(encode(stop_sequence))
        if token_ids:
            encoded.append([int(token_id) for token_id in token_ids])
    return encoded


def _bundle_stop_token_ids(
    *,
    manifest: Mapping[str, object],
    tokenizer: object | None,
) -> set[int]:
    token_ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        token_ids.add(int(eos_token_id))

    family = str(manifest.get("family", "") or "").strip().lower()
    stop_tokens: tuple[str, ...] = ()
    if family in {"gemma4", "gemma"}:
        stop_tokens = ("<turn|>", "<eos>")
    elif family in {"qwen", "qwen3", "qwen3_5", "qwen3.5"}:
        stop_tokens = ("<|im_end|>",)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    encode = getattr(tokenizer, "encode", None)
    for token in stop_tokens:
        token_id = None
        if callable(convert):
            try:
                token_id = convert(token)
            except Exception:
                token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.add(int(token_id))
            continue
        if callable(encode):
            try:
                encoded = encode(token, add_special_tokens=False)
            except TypeError:
                encoded = encode(token)
            except Exception:
                encoded = []
            if isinstance(encoded, list) and len(encoded) == 1:
                token_ids.add(int(encoded[0]))
    return token_ids


def _has_token_suffix(token_ids: list[int], suffix: list[int]) -> bool:
    if not suffix or len(token_ids) < len(suffix):
        return False
    return token_ids[-len(suffix) :] == suffix


def _trim_stop_suffix(token_ids: list[int], stop_sequences: list[list[int]]) -> bool:
    for stop_sequence in stop_sequences:
        if _has_token_suffix(token_ids, stop_sequence):
            del token_ids[-len(stop_sequence) :]
            return True
    return False


def _decode_generated_text(tokenizer: object | None, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    if tokenizer is None:
        return ""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        return str(decode(token_ids, skip_special_tokens=skip_special_tokens))
    except TypeError:
        return str(decode(token_ids))


def _resolve_seq2seq_prompt_token_ids(
    *,
    manifest: dict[str, object],
    prompt: str | None,
    tokenizer: object | None,
) -> list[int]:
    if prompt:
        if tokenizer is None:
            raise ValueError("transformers tokenizer is required when providing --prompt for seq2seq bundles")
        return _tokenize_bundle_prompt(tokenizer, prompt, enable_thinking_if_supported=False)

    inputs_meta = manifest.get("inputs")
    if isinstance(inputs_meta, dict):
        stored_ids = _parse_nested_manifest_input_ids(inputs_meta.get("decoder_input_ids"))
        if stored_ids:
            return stored_ids
        decoder_start_token_id = inputs_meta.get("decoder_start_token_id")
        if isinstance(decoder_start_token_id, int):
            return [int(decoder_start_token_id)]
    return []


def _select_next_token_with_suppression(
    logits: np.ndarray,
    *,
    suppress_tokens: list[int] | tuple[int, ...],
    begin_suppress_tokens: list[int] | tuple[int, ...],
) -> int:
    masked = np.asarray(logits, dtype=np.float32).copy()
    vocab_size = masked.shape[-1]
    for token_id in (*suppress_tokens, *begin_suppress_tokens):
        token_index = int(token_id)
        if 0 <= token_index < vocab_size:
            masked[token_index] = -np.inf
    return int(np.argmax(masked))


def _strip_whisper_control_tokens(text: str) -> str:
    cleaned = re.sub(r"<\|\d+(?:\.\d+)?\|>", " ", text)
    cleaned = re.sub(r"<\|[^|>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _attach_component_io_names(
    manifest: dict[str, object],
    component_graphs: dict[str, LoadedComponentGraph],
) -> None:
    for component_entry in manifest.get("components", []):
        if not isinstance(component_entry, dict):
            continue
        name = str(component_entry.get("component", "")).strip()
        if not name or name not in component_graphs:
            continue
        component = component_graphs[name]
        logical_inputs = tuple(str(value) for value in component_entry.get("logical_inputs", []))
        logical_outputs = tuple(str(value) for value in component_entry.get("logical_outputs", []))
        if not logical_inputs or not logical_outputs:
            raise ValueError(
                f"component bundle manifest is missing logical IO names for component={name!r}"
            )
        component._input_names = logical_inputs
        component._output_names = logical_outputs


def _rebind_bound_constants(
    *,
    graph: Graph,
    bundle_root: Path,
    bindings: list[dict[str, object]],
    weights_dir: str | Path | None,
) -> list[object]:
    loaded: list[object] = []
    for binding in bindings:
        node_id = int(binding["node_id"])
        if node_id < 0:
            continue
        raw_path = str(binding["path"])
        tensor_path = _resolve_bound_tensor_path(
            raw_path,
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )
        tensor = graph._tensor_from_node(node_id)
        binding_format = str(binding.get("format", "tensor_io") or "tensor_io")
        if binding_format != "tensor_io":
            raise RuntimeError(
                f"unsupported bound constant format {binding_format!r}; re-run cactus convert to rebuild the bundle"
            )
        try:
            tensor_info = graph._get_output_info(tensor.id)
        except Exception:
            tensor_info = {}
        if int(tensor_info.get("num_elements", 1) or 0) == 0:
            # Empty constants carry only shape information. There is no data
            # payload to bind, and mmap rebinding rejects zero-byte weights.
            continue
        if _has_runtime_symbol("cactus_graph_bind_mmap_weights"):
            rc = _lib.cactus_graph_bind_mmap_weights(
                graph.h,
                cactus_node_t(tensor.id),
                str(tensor_path).encode(),
            )
            if rc != 0:
                raise RuntimeError("graph_bind_mmap_weights failed")
            loaded.append(tensor_path)
            continue

        tensor_file = _open_cactus_tensor_file(tensor_path)
        graph.set_external_input(tensor, int(tensor_file.data.ctypes.data), dtype=tensor_file.precision)
        if tensor_file.scales is not None and tensor_file.group_size > 0:
            rc = _lib.cactus_graph_set_grouped_scales(
                graph.h,
                cactus_node_t(tensor.id),
                int(tensor_file.group_size),
                int(tensor_file.num_groups),
                tensor_file.scales.ctypes.data_as(ctypes.c_void_p),
            )
            if rc != 0:
                raise RuntimeError("graph_set_grouped_scales failed")
        if tensor_file.is_interleaved:
            graph.set_interleaved(tensor, True, tensor_file.original_n)
        loaded.append(tensor_file)
    return loaded


def _resolve_bound_tensor_path(
    raw_path: str,
    *,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    explicit = Path(raw_path).expanduser()
    if explicit.is_absolute():
        candidates.append(explicit)
    else:
        candidates.append((bundle_root / explicit).resolve())
        candidates.append(explicit.resolve())

    if weights_dir is not None:
        weights_root = Path(weights_dir).expanduser().resolve()
        raw_parts = Path(raw_path).parts
        for index in range(len(raw_parts)):
            candidates.append(weights_root.joinpath(*raw_parts[index:]))
        candidates.append(weights_root / Path(raw_path).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not resolve bound tensor file {raw_path!r} from bundle_root={bundle_root}"
        + ("" if weights_dir is None else f" weights_dir={Path(weights_dir).expanduser().resolve()}")
    )


def _open_cactus_tensor_file(path: str | Path) -> LoadedTensorFile:
    tensor_path = Path(path).expanduser().resolve()
    with tensor_path.open("rb") as handle:
        header = handle.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        raise RuntimeError(f"tensor file is too small for a Cactus header: {tensor_path}")

    magic = header[:4]
    if magic != CACTUS_MAGIC:
        raise RuntimeError(f"tensor file is missing the CACT header: {tensor_path}")

    flags = struct.unpack_from("<I", header, 4)[0]
    alignment = max(1, int(struct.unpack_from("<I", header, 8)[0]))
    ndim = int(struct.unpack_from("<I", header, 12)[0])
    dims = list(struct.unpack_from("<QQQQ", header, 16))
    precision = int(struct.unpack_from("<I", header, 48)[0])
    byte_size = int(struct.unpack_from("<Q", header, 52)[0])
    scales_bytes = int(struct.unpack_from("<Q", header, 60)[0])
    group_size = int(struct.unpack_from("<I", header, 68)[0])
    num_groups = int(struct.unpack_from("<I", header, 72)[0])
    original_n = int(struct.unpack_from("<Q", header, 76)[0])
    header_size = _HEADER_SIZE
    if flags & _FLAG_EXTENDED_SHAPE:
        with tensor_path.open("rb") as handle:
            handle.seek(_HEADER_SIZE)
            extended = handle.read(32)
        if len(extended) < 32:
            raise RuntimeError(f"tensor file is too small for extended Cactus shape header: {tensor_path}")
        dims.extend(struct.unpack("<QQQQ", extended))
        header_size += 32
    shape = tuple(int(dim) for dim in dims[:ndim] if int(dim) > 0)

    dtype = _PRECISION_TO_DTYPE.get(precision)
    if dtype is None:
        raise RuntimeError(f"unsupported tensor precision {precision} in {tensor_path}")

    aligned_header = align_offset(header_size, alignment)
    scales_offset = aligned_header if scales_bytes > 0 else 0
    data_offset = (
        align_offset(scales_offset + scales_bytes, alignment)
        if scales_bytes > 0
        else aligned_header
    )

    data_element_count = byte_size // np.dtype(dtype).itemsize
    data = np.memmap(tensor_path, mode="r", dtype=dtype, offset=data_offset, shape=(data_element_count,))
    scales = None
    if scales_bytes > 0:
        scales = np.memmap(
            tensor_path,
            mode="r",
            dtype=np.float16,
            offset=scales_offset,
            shape=(scales_bytes // np.dtype(np.float16).itemsize,),
        )
    return LoadedTensorFile(
        path=tensor_path,
        precision=precision,
        shape=shape,
        data=data,
        scales=scales,
        group_size=group_size,
        num_groups=num_groups,
        is_interleaved=bool(flags & FLAG_INTERLEAVED),
        original_n=original_n,
    )


def _decode_parakeet_tdt_token_ids(vocabulary: tuple[str, ...], token_ids: list[int]) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(vocabulary):
            continue
        piece = vocabulary[token_id]
        if piece.startswith("<|") and piece.endswith("|>"):
            continue
        pieces.append(piece)
    text = "".join(pieces).replace("▁", " ")
    return re.sub(r"\s+", " ", text).strip()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    raise TypeError(f"unsupported runtime value type: {type(value).__name__}")
