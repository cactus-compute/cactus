from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import ctypes
import itertools
import json
import math
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
from cactus.transpile.model_profiles import profile_for_family
from cactus.transpile.lower import transpile_preoptimized_ir
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.runtime_support import ensure_transformers_supports_profile
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
from cactus.transpile.weight_binding import resolve_weight_binding


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

_STATEFUL_DECODE_COMPONENTS = frozenset({"decoder_prefill_chunk", "decoder_media_step", "decoder_step"})
_COMPONENT_GRAPH_CACHE: dict[tuple[str, str | None, str], tuple[dict[str, "LoadedComponentGraph"], dict[str, object]]] = {}
_MULTIMODAL_ENCODER_FEATURE_CACHE: dict[tuple[str, str, tuple[str, ...], str | None], dict[str, np.ndarray]] = {}
_TOKENIZER_CACHE: dict[tuple[str, ...], object] = {}
_PROCESSOR_CACHE: dict[tuple[str, ...], object] = {}
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
    family = str(component_entry.get("family", "") or component_entry.get("adapter_family", "") or "").strip().lower()
    component_name = str(component_entry.get("component", "") or "").strip().lower()
    if not family:
        ir_relpath = component_entry.get("optimized_ir") or component_entry.get("raw_ir")
        if isinstance(ir_relpath, str) and ir_relpath:
            try:
                ir_payload = json.loads((root / ir_relpath).read_text())
                family = str(ir_payload.get("family", "") or "").strip().lower()
            except Exception:
                family = ""
    profile = profile_for_family(family)
    bound_constant_bindings = list(component_entry.get("bound_constant_bindings") or [])
    has_saved_constant_bindings = any(
        isinstance(binding, dict) and str(binding.get("kind", "")) == "saved_constant"
        for binding in bound_constant_bindings
    )
    if profile is not None and component_name in profile.fp16_kv_cache_components:
        os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")
    if profile is not None and component_name in profile.ir_replay_components:
        # Older Gemma4 media bundles can deserialize successfully while returning
        # stale media features from graph.cactus. Replaying the op IR preserves the
        # same Cactus ops and picks up runtime fixes for native media preprocessing
        # and Gemma4's tensor-valued clippable-linear bounds / split-decoder
        # sliding-attention metadata.
        prefer_saved_graph = False
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
        profile = profile_for_family(family)
        if profile is not None and component in profile.fp16_kv_cache_components + ("decoder",):
            optimize_graph(ir_graph)
    transpiled = transpile_preoptimized_ir(ir_graph)
    bound_tensor_files = _rebind_bound_constants(
        graph=transpiled.graph,
        bundle_root=bundle_root,
        bindings=list(transpiled.bound_constant_bindings),
        weights_dir=weights_dir,
    )
    return LoadedComponentGraph(
        component=str(component_entry.get("component", "unknown")),
        graph=transpiled.graph,
        runtime_inputs=list(transpiled.runtime_inputs),
        outputs=list(transpiled.outputs),
        bound_constant_bindings=list(transpiled.bound_constant_bindings),
        bound_tensor_files=bound_tensor_files,
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
    profile = profile_for_family(family)

    include_components = runtime_include_components_for_manifest(
        family=family,
        task=task,
        manifest=manifest,
    )
    manifest_components = {
        str(component_entry.get("component", "")).strip()
        for component_entry in manifest.get("components", [])
        if isinstance(component_entry, dict)
    }
    if (
        profile is not None
        and profile.family == "gemma4"
        and profile.cached_step_components
        and task == "multimodal_causal_lm_logits"
        and prompt is not None
        and set(profile.cached_step_components).issubset(manifest_components)
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        include_components = set(profile.cached_step_components)
        for component_name in (
            "decoder_prefill_chunk",
            "lm_encoder_text_chunk",
            "lm_encoder_media_step",
            "lm_encoder_media_chunk",
        ):
            if component_name in manifest_components:
                include_components.add(component_name)
        if image_files and "vision_encoder" in manifest_components:
            include_components.add("vision_encoder")
        if audio_file is not None and "audio_encoder" in manifest_components:
            include_components.add("audio_encoder")
    if (
        profile is not None
        and profile.family == "qwen"
        and task == "multimodal_causal_lm_logits"
        and image_files
    ):
        if {"vision_encoder", "lm_encoder", "lm_encoder_step", "decoder_media_step"}.issubset(manifest_components):
            include_components = {"vision_encoder", "lm_encoder", "lm_encoder_step", "decoder_media_step"}
        else:
            include_components = {
                name
                for name in ("vision_encoder", "lm_encoder", "decoder")
                if name in manifest_components
            }
    if (
        profile is not None
        and profile.family == "lfm2_vl"
        and task == "multimodal_causal_lm_logits"
        and prompt is not None
        and not image_files
        and audio_file is None
        and {"lm_encoder_step", "decoder_step"}.issubset(manifest_components)
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        include_components = {"lm_encoder_step", "decoder_step"}
    elif (
        profile is not None
        and profile.family == "lfm2_vl"
        and task == "multimodal_causal_lm_logits"
        and prompt is not None
        and not image_files
        and audio_file is None
        and {"text_lm_encoder", "text_decoder"}.issubset(manifest_components)
    ):
        include_components = {"text_lm_encoder", "text_decoder"}
    elif (
        profile is not None
        and profile.cached_step_components
        and task == "causal_lm_logits"
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        cached_components = set(profile.cached_step_components)
        if cached_components.issubset(manifest_components):
            include_components = cached_components
    elif profile is not None and profile.cached_step_components and task == "multimodal_causal_lm_logits" and prompt is not None and not image_files and audio_file is None:
        cached_components = set(profile.cached_step_components)
        if cached_components.issubset(manifest_components):
            include_components = cached_components
    component_graphs, manifest = load_saved_component_graphs(
        bundle_dir_or_manifest,
        weights_dir=resolved_weights_dir,
        include_components=include_components,
    )
    manifest = dict(manifest)
    manifest["_bundle_root"] = str(bundle_root)
    if profile is not None and profile.family == "parakeet_tdt" and task == "tdt_transcription":
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
    profile = profile_for_family(family)
    if (
        profile is not None
        and profile.cached_step_components
        and task in {"multimodal_causal_lm_logits", "causal_lm_logits"}
        and set(profile.cached_step_components).issubset(manifest_components)
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        # Older Gemma bundles may include a full static decoder. Cached decode
        # should not pay to load it when step graphs are available.
        include_components = set(manifest_components)
        for component in profile.cached_step_skip_components:
            include_components.discard(component)
        return include_components
    return None


def _default_weights_dir_for_manifest(
    manifest: Mapping[str, object],
    *,
    explicit: str | Path | None,
) -> str | Path | None:
    if explicit is not None:
        return explicit
    bundle_root = _bundle_root_from_manifest(manifest)
    if bundle_root is not None and (bundle_root / "weights_manifest.json").exists():
        return bundle_root
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
    _repair_saved_ir_graph_structure(
        values=values,
        nodes=nodes,
        order=order,
        constants_payload=constants_payload,
    )
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


def _repair_saved_ir_graph_structure(
    *,
    values: dict[str, IRValue],
    nodes: dict[str, IRNode],
    order: list[str],
    constants_payload: object,
) -> None:
    """Repair deterministic helper values omitted by older IR JSON writers."""

    if not isinstance(constants_payload, dict):
        return
    query_mask = values.get("v_and_1")
    valid_mask = values.get("v_bitwise_not")
    and_node = nodes.get("n_and_2")
    if (
        query_mask is not None
        and valid_mask is not None
        and and_node is not None
        and query_mask.producer is None
        and tuple(query_mask.shape or ())[:2] == (1, 1)
        and tuple(query_mask.shape or ())[-1:] == (1,)
        and "v_and_1" in and_node.inputs
    ):
        repair_node_id = "n_repair_v_and_1_query_mask"
        if repair_node_id not in nodes:
            shape = tuple(int(dim) for dim in (query_mask.shape or ()))
            nodes[repair_node_id] = IRNode(
                id=repair_node_id,
                op="view",
                inputs=["v_bitwise_not"],
                outputs=["v_and_1"],
                attrs={"shape": shape},
                meta={"repair": "vision_query_attention_mask"},
            )
            try:
                insert_at = order.index("n_and_2")
            except ValueError:
                insert_at = len(order)
            order.insert(insert_at, repair_node_id)
        query_mask.producer = repair_node_id
        if repair_node_id not in valid_mask.users:
            valid_mask.users.append(repair_node_id)
        constants_payload.pop("v_and_1", None)


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
    if isinstance(serialized, dict):
        value_type = str(serialized.get("type", ""))
        shape = tuple(int(dim) for dim in serialized.get("shape", []) or [])
        if value_type in {"torch.Tensor", "numpy.ndarray"} and not shape and "data" in serialized:
            dtype = _torch_dtype_from_name(str(serialized.get("dtype", ""))) or torch.float32
            return torch.as_tensor(serialized.get("data"), dtype=dtype).reshape(())

    if isinstance(meta.get("path"), str) and isinstance(meta.get("kind"), str):
        # The lowerer ignores the constant payload when a weight binding is present
        # in IRValue metadata and re-attaches the mmap-backed tensor directly.
        # Scalar bound constants are handled above because some ops, notably
        # Gemma4's clippable-linears, need the min/max values at lower time.
        return 0

    source_name = str(meta.get("source_name", "") or "")
    if source_name and weights_dir is not None:
        repaired_binding = resolve_weight_binding(
            weights_dir=str(weights_dir),
            source_name=source_name,
        )
        if repaired_binding is not None:
            value.meta = {
                **meta,
                "path": repaired_binding.path,
                "kind": repaired_binding.kind,
                "source_name": source_name,
            }
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
            dtype = _torch_dtype_from_name(str(serialized.get("dtype", ""))) or torch.float32
            if "data" in serialized:
                tensor = torch.as_tensor(serialized.get("data"), dtype=dtype)
                return tensor.reshape(shape) if shape else tensor.reshape(())
            zero_scalar = False if dtype is torch.bool else 0
            if not shape:
                repaired = _repair_missing_saved_ir_scalar_constant(
                    value=value,
                    dtype=dtype,
                    bundle_root=bundle_root,
                    weights_dir=weights_dir,
                )
                if repaired is not None:
                    return repaired
                return torch.tensor(zero_scalar, dtype=dtype)
            repaired = _repair_missing_saved_ir_tensor_constant(
                value=value,
                shape=shape,
                dtype=dtype,
                bundle_root=bundle_root,
                weights_dir=weights_dir,
            )
            if repaired is not None:
                return repaired
            # Older bundles can contain tiny folded tensor constants whose payload
            # was intentionally omitted because they came from shape/index helper
            # ops instead of model weights. Only zero-fill anonymous helpers; named
            # model/config constants must be materialized or explicitly repaired.
            source_name = str((value.meta or {}).get("source_name", "") or "")
            if not source_name and int(np.prod(shape, dtype=np.int64)) <= 16:
                return torch.full(shape, zero_scalar, dtype=dtype)
            if str(value.id).startswith("c_rms_norm_ones_"):
                return torch.ones(shape, dtype=dtype)
            raise ValueError(
                "saved optimized IR is missing a materialized constant payload for "
                f"{value.id} with shape={shape}; expected a bound_constants entry"
            )
        return dict(serialized)

    return serialized


def _repair_missing_saved_ir_scalar_constant(
    *,
    value: IRValue,
    dtype: torch.dtype,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> torch.Tensor | None:
    """Repair scalar config constants omitted by older IR JSON writers."""

    source_name = str((value.meta or {}).get("source_name", "") or "")
    if source_name.endswith(".self_attn.softcap"):
        cap = _resolve_gemma4_audio_attention_logit_cap(bundle_root=bundle_root, weights_dir=weights_dir)
        return torch.tensor(cap, dtype=dtype)
    return None


def _repair_missing_saved_ir_tensor_constant(
    *,
    value: IRValue,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> torch.Tensor | None:
    """Repair deterministic non-parameter buffers omitted by older IR JSON writers."""

    source_name = str((value.meta or {}).get("source_name", "") or "")
    value_id = str(value.id)
    if (
        dtype in {torch.float16, torch.float32}
        and (
            source_name.endswith("rotary_emb.inv_freq")
            or "rotary_emb_inv_freq" in value_id
        )
    ):
        repaired_rope = _repair_missing_text_rope_inv_freq(
            value=value,
            shape=shape,
            dtype=dtype,
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )
        if repaired_rope is not None:
            return repaired_rope
    if "audio_tower.rel_pos_enc.inv_timescales" in source_name or "audio_tower_rel_pos_enc_inv_timescales" in value_id:
        num_timescales = int(shape[-1]) if shape else 0
        if num_timescales <= 0:
            return None
        max_timescale = _resolve_gemma4_audio_max_timescale(bundle_root=bundle_root, weights_dir=weights_dir)
        increment = math.log(max_timescale) / max(num_timescales - 1, 1)
        inv = torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -increment)
        return inv.reshape(shape).to(dtype=dtype)
    if dtype is torch.int64 and value_id.startswith("v_unsqueeze") and len(shape) == 2:
        if shape[1] == 1:
            return torch.arange(shape[0], dtype=dtype).reshape(shape)
        if shape[0] == 1:
            return torch.arange(shape[1], dtype=dtype).reshape(shape)
    if dtype is torch.int64 and value_id.startswith("v_unsqueeze") and len(shape) == 4:
        if shape[:3] == (1, 1, 1):
            return torch.arange(shape[3], dtype=dtype).reshape(shape)
    if dtype in {torch.bool, torch.float16, torch.float32} and value_id.startswith("v_and_") and len(shape) == 4:
        if shape[:2] == (1, 1) and shape[2] == shape[3]:
            mask = torch.ones((shape[2], shape[3]), dtype=torch.bool).tril()
            return mask.reshape(shape).to(dtype=dtype)
        if shape[:2] == (1, 1) and (shape[2] == 1 or shape[3] == 1):
            return torch.ones(shape, dtype=dtype)
    if dtype is torch.int64 and value_id.startswith("v_inl_") and len(shape) == 2 and shape[1] == 1:
        return torch.arange(shape[0] - 1, -1, -1, dtype=dtype).reshape(shape)
    if dtype in {torch.float16, torch.float32} and value_id.startswith("v_inl_") and len(shape) == 3:
        repaired_rope = _repair_gemma4_missing_rope_inv_freq(
            value=value,
            shape=shape,
            dtype=dtype,
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )
        if repaired_rope is not None:
            return repaired_rope
    return None


def _repair_missing_text_rope_inv_freq(
    *,
    value: IRValue,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> torch.Tensor | None:
    """Repair deterministic text RoPE buffers omitted from saved IR JSON.

    These buffers are derived from config (`rope_theta`, head size) rather than
    learned weights, so regenerating them keeps IR replay graph-only and avoids
    storing ad hoc NumPy sidecars.
    """

    flat_count = int(np.prod(shape, dtype=np.int64)) if shape else 0
    if flat_count <= 0:
        return None
    config = _load_bundle_config_json(bundle_root=bundle_root, weights_dir=weights_dir)
    if not config:
        return None
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    if not isinstance(text_config, Mapping):
        return None
    head_dim = int(text_config.get("head_dim") or 0)
    if head_dim <= 0:
        hidden_size = int(text_config.get("hidden_size") or text_config.get("hidden_dim") or 0)
        num_heads = int(text_config.get("num_attention_heads") or text_config.get("num_heads") or 0)
        if hidden_size > 0 and num_heads > 0:
            head_dim = hidden_size // num_heads
    if head_dim <= 0 or flat_count != head_dim // 2:
        return None
    rope_parameters = text_config.get("rope_parameters")
    base = float(text_config.get("rope_theta", config.get("rope_theta", 10000.0)) or 10000.0)
    if isinstance(rope_parameters, Mapping):
        default_params = rope_parameters.get("default")
        if isinstance(default_params, Mapping):
            base = float(default_params.get("rope_theta", base) or base)
        else:
            base = float(rope_parameters.get("rope_theta", base) or base)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
    return inv_freq.reshape(shape).to(dtype=dtype)


def _repair_gemma4_missing_rope_inv_freq(
    *,
    value: IRValue,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> torch.Tensor | None:
    if len(shape) != 3 or int(shape[0]) != 1 or int(shape[2]) != 1:
        return None
    component = str((value.meta or {}).get("component", "") or "").strip().lower()
    if component not in {"decoder", "decoder_step", "decoder_prefill_chunk", "vision_encoder"}:
        return None
    config = _load_bundle_config_json(bundle_root=bundle_root, weights_dir=weights_dir)
    if not config:
        return None
    count = int(shape[1])
    if component == "vision_encoder":
        vision_config = config.get("vision_config")
        if not isinstance(vision_config, Mapping):
            return None
        rope_parameters = vision_config.get("rope_parameters")
        if not isinstance(rope_parameters, Mapping):
            return None
        base = float(rope_parameters.get("rope_theta", 100.0) or 100.0)
        spatial_dim = max(2, count * 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / float(spatial_dim)))
        return inv_freq[:count].reshape(shape).to(dtype=dtype)

    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return None
    rope_parameters = text_config.get("rope_parameters")
    head_dim = int(text_config.get("head_dim") or 0)
    if head_dim <= 0:
        hidden_size = int(text_config.get("hidden_size") or text_config.get("hidden_dim") or 0)
        num_heads = int(text_config.get("num_attention_heads") or text_config.get("num_heads") or 0)
        if hidden_size > 0 and num_heads > 0:
            head_dim = hidden_size // num_heads
    global_head_dim = int(text_config.get("global_head_dim") or 0)
    if head_dim > 0 and count == head_dim // 2 and not isinstance(rope_parameters, Mapping):
        base = float(text_config.get("rope_theta", config.get("rope_theta", 10000.0)) or 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
        return inv_freq.reshape(shape).to(dtype=dtype)
    if not isinstance(rope_parameters, Mapping):
        return None
    if head_dim > 0 and count == head_dim // 2 and "sliding_attention" not in rope_parameters:
        base = float(
            rope_parameters.get("rope_theta", text_config.get("rope_theta", config.get("rope_theta", 10000.0)))
            or 10000.0
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
        return inv_freq.reshape(shape).to(dtype=dtype)
    if head_dim > 0 and count == head_dim // 2:
        sliding = rope_parameters.get("sliding_attention")
        if not isinstance(sliding, Mapping):
            return None
        base = float(sliding.get("rope_theta", 10000.0) or 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
        return inv_freq.reshape(shape).to(dtype=dtype)
    if global_head_dim > 0 and count == global_head_dim // 2:
        full = rope_parameters.get("full_attention")
        if not isinstance(full, Mapping):
            return None
        base = float(full.get("rope_theta", 1000000.0) or 1000000.0)
        factor = float(full.get("factor", 1.0) or 1.0)
        rotary_factor = float(full.get("partial_rotary_factor", 1.0) or 1.0)
        rope_angles = int(rotary_factor * global_head_dim // 2)
        rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / float(global_head_dim))
        )
        if rope_angles < count:
            inv_freq = torch.cat((rotated, torch.zeros(count - rope_angles, dtype=torch.float32)), dim=0)
        else:
            inv_freq = rotated[:count]
        return (inv_freq / factor).reshape(shape).to(dtype=dtype)
    return None


def _load_bundle_config_json(
    *,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> dict[str, object]:
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(Path(weights_dir).expanduser().resolve() / "config.json")
    candidates.append(bundle_root / "config.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_gemma4_audio_max_timescale(
    *,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> float:
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(Path(weights_dir).expanduser().resolve() / "config.json")
    candidates.append(bundle_root / "config.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        audio_config = payload.get("audio_config")
        if not isinstance(audio_config, Mapping):
            continue
        raw = audio_config.get("rel_pos_max_timescale")
        if isinstance(raw, (int, float)) and float(raw) > 0:
            return float(raw)
    return 10000.0


def _resolve_gemma4_audio_attention_logit_cap(
    *,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> float:
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(Path(weights_dir).expanduser().resolve() / "config.json")
    candidates.append(bundle_root / "config.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        audio_config = payload.get("audio_config")
        if not isinstance(audio_config, Mapping):
            continue
        for key in ("attention_logit_cap", "conf_attention_logit_cap", "audio_logit_cap"):
            raw = audio_config.get(key)
            if isinstance(raw, (int, float)):
                return float(raw)
    return 50.0


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


def _component_cache_states_compatible(
    source: LoadedComponentGraph,
    target: LoadedComponentGraph,
) -> bool:
    source_states = list(source.cache_state_tensors)
    target_states = list(target.cache_state_tensors)
    if not source_states or not target_states or len(source_states) != len(target_states):
        return False
    for (source_layer, source_k, source_v), (target_layer, target_k, target_v) in zip(source_states, target_states, strict=True):
        if source_layer != target_layer:
            return False
        for source_tensor, target_tensor in ((source_k, target_k), (source_v, target_v)):
            if _tensor_byte_size(source_tensor) != _tensor_byte_size(target_tensor):
                return False
    return True


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


def _copy_lfm2_decoder_inputs(
    buffers: Mapping[str, np.ndarray],
    *,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
) -> None:
    np.copyto(
        buffers["inputs_embeds"],
        np.asarray(inputs_embeds, dtype=buffers["inputs_embeds"].dtype),
    )
    np.copyto(
        buffers["attention_mask"],
        np.asarray(attention_mask, dtype=buffers["attention_mask"].dtype),
    )
    np.copyto(
        buffers["position_ids"],
        np.asarray(position_ids, dtype=buffers["position_ids"].dtype),
    )


def _copy_qwen_decoder_inputs(
    buffers: Mapping[str, np.ndarray],
    *,
    inputs_embeds: np.ndarray,
    position_ids: np.ndarray,
) -> None:
    np.copyto(
        buffers["inputs_embeds"],
        np.asarray(inputs_embeds, dtype=buffers["inputs_embeds"].dtype),
    )
    np.copyto(
        buffers["position_ids"],
        np.asarray(position_ids, dtype=buffers["position_ids"].dtype),
    )


def _bind_gated_deltanet_state_feedback(component: LoadedComponentGraph) -> dict[str, np.ndarray]:
    """Bind hidden Gated DeltaNet state inputs and feed outputs back each step."""

    buffers: dict[str, np.ndarray] = {}
    for layer_key, state_input, _ in component.cache_state_tensors:
        if not str(layer_key).startswith("gdn:"):
            continue
        buffer = np.zeros(tuple(int(dim) for dim in state_input.shape), dtype=np.float16)
        component.graph.set_external_input(
            state_input,
            int(buffer.ctypes.data),
            dtype=state_input.dtype,
        )
        buffers[str(layer_key)] = buffer
    return buffers


def _refresh_gated_deltanet_state_feedback(
    component: LoadedComponentGraph,
    buffers: Mapping[str, np.ndarray],
) -> None:
    if not buffers:
        return
    for layer_key, _, state_output in component.cache_state_tensors:
        buffer = buffers.get(str(layer_key))
        if buffer is None:
            continue
        np.copyto(buffer, np.asarray(state_output.numpy(), dtype=buffer.dtype))


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
        if family not in {"lfm2_vl", "qwen"} and (component_name == "decoder" or component_name.endswith("_decoder")):
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
    profile = profile_for_family(family)
    cached_step_components = tuple(profile.cached_step_components) if profile is not None else ()
    use_cached_step_decode = (
        bool(cached_step_components)
        and set(cached_step_components).issubset(component_graphs)
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    )
    use_chunk_prefill = (
        use_cached_step_decode
        and "decoder_prefill_chunk" in component_graphs
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CHUNK_PREFILL") != "1"
    )
    dynamic_media_step_component = profile.dynamic_media_step_component if profile is not None else None
    use_dynamic_media_encoder = (
        dynamic_media_step_component is not None
        and use_cached_step_decode
        and dynamic_media_step_component in component_graphs
    )
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
    if profile is not None and profile.family == "qwen" and has_image:
        use_cached_step_decode = {"lm_encoder_step", "decoder_media_step"}.issubset(component_graphs)
        use_chunk_prefill = False
        use_dynamic_media_encoder = False
    if profile is not None and profile.family == "qwen" and not has_image and not has_audio:
        return _run_causal_lm_cached_step_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=resolved_prompt,
            input_ids=None,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
    if profile is not None and profile.family == "gemma4" and not has_image and not has_audio:
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
    if profile is not None and profile.family == "lfm2_vl" and not has_image:
        if {"lm_encoder_step", "decoder_step"}.issubset(component_graphs):
            return _run_lfm2_vl_text_cached_bundle(
                component_graphs=component_graphs,
                manifest=manifest,
                prompt=resolved_prompt,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
                stop_sequences=stop_sequences,
            )
        text_component = profile.text_only_component or "text_lm_encoder"
        if text_component not in component_graphs:
            raise ValueError(
                "LFM2-VL text-only execution requires cached step graphs or a text_lm_encoder route; "
                "re-run cactus convert to rebuild this bundle."
            )
        return _run_lfm2_vl_text_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=resolved_prompt,
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
    if use_dynamic_media_encoder:
        required_components = tuple(encoder_components)
    elif use_cached_step_decode:
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

    external_transformers_site_packages = ensure_transformers_supports_profile(profile)
    if external_transformers_site_packages:
        print(
            "note=using external transformers install for "
            f"{profile.family if profile is not None else family} runtime: "
            f"{external_transformers_site_packages}"
        )
    patch_transformers_torchvision_probe()
    patch_torch_flex_attention_compat()

    processor = _load_bundle_processor(manifest)
    preprocessor = profile.multimodal_preprocessor if profile is not None else "auto"
    if preprocessor == "qwen3_5":
        prepared = _prepare_qwen3_5_multimodal_inputs_for_runtime(
            processor,
            prompt=resolved_prompt,
            image_files=resolved_image_files,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt or "",
            enable_thinking_if_supported=enable_thinking,
        )
    elif preprocessor == "lfm2_vl":
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
        tuple(encoder_components)
        if use_dynamic_media_encoder
        else (
            tuple(encoder_components) + ("lm_encoder",)
            if use_cached_step_decode
            else required_components
        )
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
        profile_family = profile.family if profile is not None else ""
        if use_dynamic_media_encoder:
            store = _build_gemma4_dynamic_multimodal_encoder_store(
                component_graphs=component_graphs,
                manifest=manifest,
                store=store,
                tokenizer=tokenizer,
                current_length=current_length,
            )
        if profile_family == "lfm2_vl":
            return _run_lfm2_vl_cached_step_multimodal_decode(
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
        if profile_family == "qwen":
            return _run_qwen3_5_cached_step_multimodal_decode(
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
    repetition_penalty = _lfm2_repetition_penalty() if family == "lfm2_vl" else 1.0
    no_repeat_ngram_size = _lfm2_no_repeat_ngram_size() if family == "lfm2_vl" else 0

    for step_index in range(token_budget):
        logits = np.asarray(store["logits"], dtype=np.float32)
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        token_position = _multimodal_decoder_logits_token_position(
            component_graphs.get("decoder"),
            logits=logits,
            current_length=current_length,
            right_aligned=family != "lfm2_vl",
        )
        next_token_id = _select_next_token_with_repetition_penalty(
            logits[0, token_position],
            token_ids=generated_ids,
            penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
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
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

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

    next_position_id = len(prompt_token_ids)
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
            next_position_id,
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("Gemma4 text-only cached decoder did not produce decode logits")
        next_position_id += 1

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


def _run_lfm2_vl_text_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    decoder_component_name = "text_decoder" if "text_decoder" in component_graphs else "decoder"
    if decoder_component_name not in component_graphs or "text_lm_encoder" not in component_graphs:
        raise ValueError("LFM2-VL text route requires text_lm_encoder and decoder components")

    tokenizer = _load_bundle_tokenizer(manifest)
    prompt_token_ids = _tokenize_bundle_prompt_for_manifest(
        manifest,
        tokenizer,
        prompt,
        enable_thinking_if_supported=enable_thinking,
    )
    if not prompt_token_ids:
        raise ValueError("LFM2-VL text prompt produced no token ids")

    _attach_component_io_names(manifest, component_graphs)
    text_lm_encoder = component_graphs["text_lm_encoder"]
    input_names = component_input_names(text_lm_encoder)
    if input_names != ("input_ids", "attention_mask"):
        raise ValueError(
            "LFM2-VL text_lm_encoder must use logical inputs ('input_ids', 'attention_mask'), "
            f"got {input_names!r}"
        )
    target_token_count = int(text_lm_encoder.runtime_inputs[0].shape[1])
    if len(prompt_token_ids) > target_token_count:
        raise ValueError(
            f"prompt token length {len(prompt_token_ids)} exceeds transpiled text context {target_token_count}; "
            "re-run cactus convert with a larger profile context or use a shorter prompt."
        )

    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
    input_ids = np.full((1, target_token_count), padding_token_id, dtype=np.int64)
    attention_mask = np.zeros((1, target_token_count), dtype=np.int64)
    store: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    active_token_ids = list(prompt_token_ids)

    def _prepare_text_window() -> int:
        input_ids.fill(int(padding_token_id))
        attention_mask.fill(0)
        clipped_ids = active_token_ids[-target_token_count:]
        if not clipped_ids:
            return 0
        input_ids[0, : len(clipped_ids)] = np.asarray(clipped_ids, dtype=np.int64)
        attention_mask[0, : len(clipped_ids)] = 1
        return len(clipped_ids)

    def _run_text_window() -> int:
        active_count = _prepare_text_window()
        execute_loaded_component(text_lm_encoder, store)
        position_ids = np.cumsum(attention_mask, axis=1, dtype=np.int64) - 1
        store["position_ids"] = np.where(attention_mask > 0, position_ids, 0).astype(np.int64, copy=False)
        execute_loaded_component(component_graphs[decoder_component_name], store)
        return active_count

    start = time.perf_counter()
    active_count = _run_text_window()

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )
    token_budget = max(0, requested_tokens)
    stop_reason = "max_new_tokens"
    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    repetition_penalty = _lfm2_repetition_penalty()
    no_repeat_ngram_size = _lfm2_no_repeat_ngram_size()

    for step_index in range(token_budget):
        logits = np.asarray(store["logits"], dtype=np.float32)
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        token_position = -1 if logits.shape[1] == 1 else max(0, min(active_count - 1, logits.shape[1] - 1))
        next_token_id = _select_next_token_with_repetition_penalty(
            logits[0, token_position],
            token_ids=generated_ids,
            penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
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
        active_token_ids.append(next_token_id)
        active_count = _run_text_window()

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
    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "route": "text",
        "decode_mode": f"text_lm_encoder+{decoder_component_name}",
        "prompt": prompt,
        "input_shapes": {
            "input_ids": list(input_ids.shape),
            "attention_mask": list(attention_mask.shape),
        },
        "output_shape": logits_shape or [],
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
    }


def _run_lfm2_vl_text_cached_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    if "lm_encoder_step" not in component_graphs or "decoder_step" not in component_graphs:
        raise ValueError("LFM2-VL text cached route requires lm_encoder_step and decoder_step components")

    tokenizer = _load_bundle_tokenizer(manifest)
    prompt_token_ids = _tokenize_bundle_prompt_for_manifest(
        manifest,
        tokenizer,
        prompt,
        enable_thinking_if_supported=enable_thinking,
    )
    if not prompt_token_ids:
        raise ValueError("LFM2-VL text prompt produced no token ids")

    _attach_component_io_names(manifest, component_graphs)
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

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
        _copy_lfm2_decoder_inputs(
            decoder_input_buffers,
            inputs_embeds=lm_encoder_step.outputs[0].numpy(),
            attention_mask=lm_encoder_step.outputs[1].numpy(),
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
    prime_start = start
    for position_id, token_id in enumerate(prompt_token_ids):
        logits = _run_step_token(
            int(token_id),
            int(position_id),
            read_logits=position_id + 1 == len(prompt_token_ids),
        )
    prime_end = time.perf_counter()
    if logits is None:
        raise RuntimeError("LFM2-VL text cached decoder did not produce prompt logits")

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

        logits = _run_step_token(next_token_id, token_position, read_logits=True)
        if logits is None:
            raise RuntimeError("LFM2-VL text cached decoder did not produce decode logits")
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
    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "route": "text",
        "decode_mode": "cached_step_text",
        "prompt": prompt,
        "output_shape": logits_shape or [],
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "cache_prime_ms": (prime_end - prime_start) * 1000.0,
        "cache_prime_tokens": len(prompt_token_ids),
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
    }


def _as_gemma4_feature_sequence(
    value: np.ndarray,
    *,
    feature_name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise RuntimeError(f"Gemma4 {feature_name} must be rank 2 or 3, got shape {list(array.shape)}")
    if int(array.shape[0]) != 1:
        raise RuntimeError(f"Gemma4 {feature_name} currently expects batch=1, got shape {list(array.shape)}")
    return np.ascontiguousarray(array)


def _build_gemma4_dynamic_multimodal_encoder_store(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    store: dict[str, np.ndarray],
    tokenizer: object | None,
    current_length: int,
) -> dict[str, np.ndarray]:
    """Build Gemma4 prompt embeddings for the media actually present this turn.

    Static `lm_encoder` graphs are tied to the representative prompt used when
    the bundle was created. The step encoder is shape-stable, so using it here
    lets one Gemma4 bundle support text, image+text, audio+text, and
    image+audio+text without compiling separate entrypoints.
    """

    if "lm_encoder_step" not in component_graphs:
        raise ValueError("Gemma4 dynamic multimodal execution requires lm_encoder_step")
    lm_encoder_step = component_graphs["lm_encoder_step"]
    lm_encoder_media_step = component_graphs.get("lm_encoder_media_step")
    lm_encoder_text_chunk = component_graphs.get("lm_encoder_text_chunk")
    lm_encoder_media_chunk = component_graphs.get("lm_encoder_media_chunk")
    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    pad_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)

    input_ids = np.asarray(store.get("input_ids"))
    token_type_ids = np.asarray(store.get("token_type_ids"))
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError(f"Gemma4 dynamic encoder requires input_ids shape [1, N], got {list(input_ids.shape)}")
    if token_type_ids.ndim != 2 or token_type_ids.shape != input_ids.shape:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    token_count = max(0, min(int(current_length), int(input_ids.shape[1])))
    if token_count <= 0:
        raise RuntimeError("Gemma4 dynamic encoder received an empty prompt")

    image_features_raw = store.get("image_features")
    audio_features_raw = store.get("audio_features")
    image_features = (
        _as_gemma4_feature_sequence(image_features_raw, feature_name="image_features")
        if isinstance(image_features_raw, np.ndarray)
        else None
    )
    audio_features = (
        _as_gemma4_feature_sequence(audio_features_raw, feature_name="audio_features")
        if isinstance(audio_features_raw, np.ndarray)
        else None
    )

    image_needed = int(np.count_nonzero(token_type_ids[0, :token_count] == 1))
    audio_needed = int(
        np.count_nonzero(
            (token_type_ids[0, :token_count] == 2)
            | (token_type_ids[0, :token_count] == 3)
        )
    )
    if image_needed and (image_features is None or int(image_features.shape[1]) < image_needed):
        got = 0 if image_features is None else int(image_features.shape[1])
        raise RuntimeError(f"Gemma4 image feature count mismatch: need {image_needed}, got {got}")
    if audio_needed and (audio_features is None or int(audio_features.shape[1]) < audio_needed):
        got = 0 if audio_features is None else int(audio_features.shape[1])
        raise RuntimeError(f"Gemma4 audio feature count mismatch: need {audio_needed}, got {got}")

    input_buffers = _bind_zero_input_buffers(
        lm_encoder_step,
        {"input_ids": np.int64, "position_ids": np.int64},
    )
    text_chunk_buffers: dict[str, np.ndarray] | None = None
    text_chunk_tokens = 0
    if lm_encoder_text_chunk is not None:
        text_chunk_buffers = _bind_zero_input_buffers(
            lm_encoder_text_chunk,
            {"input_ids": np.int64, "position_ids": np.int64},
        )
        text_chunk_tokens = int(text_chunk_buffers["input_ids"].shape[1])

    media_input_buffers: dict[str, np.ndarray] | None = None
    media_chunk_buffers: dict[str, np.ndarray] | None = None
    media_chunk_tokens = 0
    if lm_encoder_media_step is not None:
        media_dtype = np.float16
        for candidate in (image_features, audio_features):
            if isinstance(candidate, np.ndarray):
                media_dtype = np.asarray(candidate).dtype
                break
        media_input_buffers = _bind_zero_input_buffers(
            lm_encoder_media_step,
            {"inputs_embeds": media_dtype, "input_ids": np.int64, "position_ids": np.int64},
        )
        if lm_encoder_media_chunk is not None:
            media_chunk_buffers = _bind_zero_input_buffers(
                lm_encoder_media_chunk,
                {"inputs_embeds": media_dtype, "input_ids": np.int64, "position_ids": np.int64},
            )
            media_chunk_tokens = int(media_chunk_buffers["inputs_embeds"].shape[1])

    input_embed_parts: list[np.ndarray] = []
    per_layer_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    image_index = 0
    audio_index = 0

    def _append_encoder_outputs(raw_outputs: list[Tensor], count: int) -> None:
        step_inputs_embeds = np.asarray(raw_outputs[0].numpy())[:, :count, ...].copy()
        step_per_layer_inputs = np.asarray(raw_outputs[1].numpy())[:, :count, ...].copy()
        step_position_ids = np.asarray(raw_outputs[2].numpy())[:, :count].copy()
        input_embed_parts.append(np.ascontiguousarray(step_inputs_embeds))
        per_layer_parts.append(np.ascontiguousarray(step_per_layer_inputs))
        position_parts.append(np.ascontiguousarray(step_position_ids))

    def _media_kind(token_type: int) -> str | None:
        if token_type == 1 and image_features is not None:
            return "image"
        if token_type in {2, 3} and audio_features is not None:
            return "audio"
        return None

    position = 0
    while position < token_count:
        token_type = int(token_type_ids[0, position])
        kind = _media_kind(token_type)

        if (
            kind is None
            and text_chunk_buffers is not None
            and lm_encoder_text_chunk is not None
            and text_chunk_tokens > 1
        ):
            run = 0
            while (
                position + run < token_count
                and run < text_chunk_tokens
                and _media_kind(int(token_type_ids[0, position + run])) is None
            ):
                run += 1
            if run == text_chunk_tokens:
                text_chunk_buffers["input_ids"][...] = input_ids[:, position : position + run]
                text_chunk_buffers["position_ids"][...] = np.arange(
                    position,
                    position + run,
                    dtype=np.int64,
                )[None, :]
                lm_encoder_text_chunk.graph.execute()
                _append_encoder_outputs(lm_encoder_text_chunk.outputs, run)
                position += run
                continue

        if (
            kind is not None
            and media_chunk_buffers is not None
            and lm_encoder_media_chunk is not None
            and media_chunk_tokens > 1
        ):
            run = 0
            while (
                position + run < token_count
                and run < media_chunk_tokens
                and _media_kind(int(token_type_ids[0, position + run])) == kind
            ):
                run += 1
            if run == media_chunk_tokens:
                features = image_features if kind == "image" else audio_features
                feature_index = image_index if kind == "image" else audio_index
                if features is not None and feature_index + run <= int(features.shape[1]):
                    media_chunk_buffers["inputs_embeds"][...] = features[
                        :,
                        feature_index : feature_index + run,
                        :,
                    ].astype(media_chunk_buffers["inputs_embeds"].dtype, copy=False)
                    media_chunk_buffers["input_ids"].fill(0)
                    media_chunk_buffers["position_ids"][...] = np.arange(
                        position,
                        position + run,
                        dtype=np.int64,
                    )[None, :]
                    lm_encoder_media_chunk.graph.execute()
                    _append_encoder_outputs(lm_encoder_media_chunk.outputs, run)
                    if kind == "image":
                        image_index += run
                    else:
                        audio_index += run
                    position += run
                    continue

        token_id = int(input_ids[0, position])
        media_slice: np.ndarray | None = None
        if token_type == 1 and image_features is not None:
            media_slice = image_features[:, image_index : image_index + 1, :]
            image_index += 1
        elif token_type in {2, 3} and audio_features is not None:
            media_slice = audio_features[:, audio_index : audio_index + 1, :]
            audio_index += 1

        if media_slice is not None and lm_encoder_media_step is not None and media_input_buffers is not None:
            media_input_buffers["inputs_embeds"][...] = media_slice.astype(
                media_input_buffers["inputs_embeds"].dtype,
                copy=False,
            )
            media_input_buffers["input_ids"].fill(0)
            media_input_buffers["position_ids"].fill(position)
            lm_encoder_media_step.graph.execute()
            step_inputs_embeds = np.asarray(lm_encoder_media_step.outputs[0].numpy()).copy()
            step_per_layer_inputs = np.asarray(lm_encoder_media_step.outputs[1].numpy()).copy()
            step_position_ids = np.asarray(lm_encoder_media_step.outputs[2].numpy()).copy()
        else:
            step_token_id = int(pad_token_id) if token_type in {1, 2, 3} else token_id
            input_buffers["input_ids"].fill(step_token_id)
            input_buffers["position_ids"].fill(position)
            lm_encoder_step.graph.execute()
            step_inputs_embeds = np.asarray(lm_encoder_step.outputs[0].numpy()).copy()
            step_per_layer_inputs = np.asarray(lm_encoder_step.outputs[1].numpy()).copy()
            step_position_ids = np.asarray(lm_encoder_step.outputs[2].numpy()).copy()
            if media_slice is not None:
                step_inputs_embeds = media_slice.astype(step_inputs_embeds.dtype, copy=False)

        input_embed_parts.append(np.ascontiguousarray(step_inputs_embeds))
        per_layer_parts.append(np.ascontiguousarray(step_per_layer_inputs))
        position_parts.append(np.ascontiguousarray(step_position_ids))
        position += 1

    dynamic_store = dict(store)
    dynamic_store["inputs_embeds"] = np.ascontiguousarray(np.concatenate(input_embed_parts, axis=1))
    dynamic_store["per_layer_inputs"] = np.ascontiguousarray(np.concatenate(per_layer_parts, axis=1))
    dynamic_store["position_ids"] = np.ascontiguousarray(np.concatenate(position_parts, axis=1))
    return dynamic_store


def _run_lfm2_vl_cached_step_multimodal_decode(
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
    if "lm_encoder_step" not in component_graphs or "decoder_step" not in component_graphs:
        raise ValueError("LFM2-VL cached multimodal route requires lm_encoder_step and decoder_step")
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

    lm_encoder_step = component_graphs["lm_encoder_step"]
    decoder_step = component_graphs["decoder_step"]
    decoder_prefill_chunk = component_graphs.get("decoder_prefill_chunk")
    # LFM2-VL chunk prefill currently captures a larger rolling KV state than
    # the one-token step graph accepts. Prime with decoder_step until chunk and
    # step cache capture are unified.
    decoder_prefill_chunk = None
    if os.environ.get("CACTUS_TRANSPILER_DISABLE_CHUNK_PREFILL") == "1":
        decoder_prefill_chunk = None

    inputs_embeds = np.asarray(store["inputs_embeds"])
    attention_mask = np.asarray(store["attention_mask"])
    position_ids = np.asarray(store["position_ids"])
    prompt_tokens = max(0, min(int(current_length), int(inputs_embeds.shape[1])))
    if prompt_tokens <= 0:
        raise RuntimeError("LFM2-VL cached decode requires at least one prompt token")

    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )

    decoder_dtypes = {
        "inputs_embeds": inputs_embeds.dtype,
        "attention_mask": attention_mask.dtype,
        "position_ids": position_ids.dtype,
    }
    decoder_input_buffers = _bind_zero_input_buffers(decoder_step, decoder_dtypes)

    def _run_decoder_step(
        *,
        step_inputs_embeds: np.ndarray,
        step_attention_mask: np.ndarray,
        step_position_ids: np.ndarray,
        read_logits: bool,
    ) -> np.ndarray | None:
        _copy_lfm2_decoder_inputs(
            decoder_input_buffers,
            inputs_embeds=step_inputs_embeds,
            attention_mask=step_attention_mask,
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
            _copy_lfm2_decoder_inputs(
                prefill_input_buffers,
                inputs_embeds=inputs_embeds[:, chunk_start:chunk_end, :],
                attention_mask=attention_mask[:, chunk_start:chunk_end],
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
            step_attention_mask=attention_mask[:, token_index : token_index + 1],
            step_position_ids=position_ids[:, token_index : token_index + 1],
            read_logits=token_index + 1 == prompt_tokens,
        )
        token_index += 1
    prime_end = time.perf_counter()

    if logits is None:
        raise RuntimeError("LFM2-VL cached decoder did not produce logits")

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"
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
            step_attention_mask=lm_encoder_step.outputs[1].numpy(),
            step_position_ids=lm_encoder_step.outputs[2].numpy(),
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("LFM2-VL cached decoder did not produce decode logits")
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


def _run_qwen3_5_cached_step_multimodal_decode(
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
    if "lm_encoder_step" not in component_graphs or "decoder_media_step" not in component_graphs:
        raise ValueError("Qwen3.5 image decode requires lm_encoder_step and decoder_media_step")
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

    lm_encoder_step = component_graphs["lm_encoder_step"]
    decoder_step = component_graphs["decoder_media_step"]

    inputs_embeds = np.asarray(store["inputs_embeds"])
    position_ids = np.asarray(store["position_ids"])
    if position_ids.ndim == 3:
        position_token_capacity = int(position_ids.shape[2])
    else:
        position_token_capacity = int(position_ids.shape[1])
    prompt_tokens = max(
        0,
        min(
            int(current_length),
            int(inputs_embeds.shape[1]),
            position_token_capacity,
        ),
    )
    if prompt_tokens <= 0:
        raise RuntimeError("Qwen3.5 cached multimodal decode requires at least one prompt token")

    stop_token_ids = _bundle_stop_token_ids(manifest=manifest, tokenizer=tokenizer)
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, stop_sequences)
    requested_tokens = (
        _UNBOUNDED_GENERATION_GUARD_TOKENS
        if max_new_tokens is None
        else max(0, int(max_new_tokens))
    )

    decoder_input_buffers = _bind_zero_input_buffers(
        decoder_step,
        {
            "inputs_embeds": inputs_embeds.dtype,
            "position_ids": position_ids.dtype,
        },
    )
    gdn_state_buffers = _bind_gated_deltanet_state_feedback(decoder_step)

    def _position_slice(token_index: int) -> np.ndarray:
        if position_ids.ndim == 3:
            return position_ids[:, :, token_index : token_index + 1]
        return position_ids[:, token_index : token_index + 1]

    def _run_decoder_step(
        *,
        step_inputs_embeds: np.ndarray,
        step_position_ids: np.ndarray,
        read_logits: bool,
    ) -> np.ndarray | None:
        _copy_qwen_decoder_inputs(
            decoder_input_buffers,
            inputs_embeds=step_inputs_embeds,
            position_ids=step_position_ids,
        )
        decoder_step.graph.execute()
        _refresh_gated_deltanet_state_feedback(decoder_step, gdn_state_buffers)
        if not read_logits:
            return None
        return np.asarray(decoder_step.outputs[0].numpy())

    lm_encoder_input_buffers = _bind_zero_input_buffers(
        lm_encoder_step,
        {"input_ids": np.int64, "position_ids": np.int64},
    )

    logits: np.ndarray | None = None
    prime_start = time.perf_counter()
    for token_index in range(prompt_tokens):
        logits = _run_decoder_step(
            step_inputs_embeds=inputs_embeds[:, token_index : token_index + 1, :],
            step_position_ids=_position_slice(token_index),
            read_logits=token_index + 1 == prompt_tokens,
        )
    prime_end = time.perf_counter()
    if logits is None:
        raise RuntimeError("Qwen3.5 cached multimodal decoder did not produce prompt logits")

    if position_ids.ndim == 3:
        next_position_id = int(np.max(position_ids[:, 0, prompt_tokens - 1])) + 1
    else:
        next_position_id = int(np.max(position_ids[0, prompt_tokens - 1])) + 1
    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"

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
        lm_encoder_input_buffers["position_ids"].fill(int(next_position_id))
        lm_encoder_step.graph.execute()
        logits = _run_decoder_step(
            step_inputs_embeds=lm_encoder_step.outputs[0].numpy(),
            step_position_ids=lm_encoder_step.outputs[1].numpy(),
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("Qwen3.5 cached multimodal decoder did not produce decode logits")
        next_position_id += 1

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
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "response": response,
        "stop_reason": stop_reason,
        "next_token_id": first_generated_token_id,
        "next_token": first_generated_token,
        "decode_mode": "cached_step_multimodal",
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

    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")
    lm_encoder_step = component_graphs["lm_encoder_step"]
    decoder_step = component_graphs["decoder_step"]
    decoder_prefill_chunk = component_graphs.get("decoder_prefill_chunk")
    if os.environ.get("CACTUS_TRANSPILER_DISABLE_CHUNK_PREFILL") == "1":
        decoder_prefill_chunk = None
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
        if not _component_cache_states_compatible(decoder_prefill_chunk, decoder_step):
            decoder_prefill_chunk = None
            prefill_input_buffers = None
            prefill_chunk_tokens = 0

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


def _multimodal_decoder_logits_token_position(
    component: LoadedComponentGraph | None,
    *,
    logits: np.ndarray,
    current_length: int,
    right_aligned: bool,
) -> int:
    if int(logits.shape[1]) <= 1:
        return 0
    if right_aligned and component is not None and "inputs_embeds" in component_input_names(component):
        return int(logits.shape[1]) - 1
    return max(0, min(int(current_length) - 1, int(logits.shape[1]) - 1))


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
    position_ids = store.get("position_ids")
    if (
        isinstance(position_ids, np.ndarray)
        and position_ids.ndim == 3
        and int(position_ids.shape[0]) in {3, 4}
        and int(position_ids.shape[1]) == 1
        and current_length < int(position_ids.shape[2])
    ):
        previous_index = max(0, int(current_length) - 1)
        next_position = int(np.max(position_ids[:, 0, previous_index])) + 1
        position_ids[:, 0, current_length] = next_position


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


def _compute_qwen3_5_position_ids_for_runtime(
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
        parts: list[torch.Tensor] = []
        for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda item: int(item[1])):
            group_items = list(group)
            token_count = group_items[-1][0] - group_items[0][0] + 1
            if int(modality_type) == 0:
                text_positions = torch.arange(token_count, dtype=input_ids.dtype, device=input_ids.device)
                parts.append(text_positions.view(1, -1).expand(3, -1) + current_pos)
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
            parts.append(torch.stack([position_temporal, position_height, position_width], dim=0))
            current_pos += max(grid_h, grid_w) // int(spatial_merge_size)
        if not parts:
            continue
        positions = torch.cat(parts, dim=1)
        if valid_mask is not None:
            position_ids[:, batch_idx, valid_mask] = positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx, : positions.shape[1]] = positions.to(position_ids.device)
    return position_ids


def _prepare_qwen3_5_multimodal_inputs_for_runtime(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("Qwen3.5 multimodal bundle execution requires an AutoProcessor with image support")
    images = _load_image_inputs_for_runtime(image_files)
    if not images:
        raise RuntimeError("Qwen3.5 multimodal bundle execution requires at least one --image")

    user_content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": prompt.strip()})
    messages: list[dict[str, object]] = []
    normalized_system = system_prompt.strip()
    if normalized_system:
        messages.append({"role": "system", "content": normalized_system})
    messages.append({"role": "user", "content": user_content})

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

    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    attention_mask = batch.get("attention_mask") if hasattr(batch, "get") else None
    mm_token_type_ids = batch.get("mm_token_type_ids") if hasattr(batch, "get") else None
    image_grid_thw = batch.get("image_grid_thw") if hasattr(batch, "get") else None
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, mm_token_type_ids, image_grid_thw)):
        raise RuntimeError("Qwen3.5 processor did not return required multimodal text/grid tensors")
    spatial_merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2)
    batch["position_ids"] = _compute_qwen3_5_position_ids_for_runtime(
        input_ids.to(dtype=torch.long),
        mm_token_type_ids.to(dtype=torch.long),
        image_grid_thw.to(dtype=torch.long),
        attention_mask.to(dtype=torch.long),
        spatial_merge_size=spatial_merge_size,
    )

    names = ("input_ids", "attention_mask", "position_ids", "pixel_values", "image_grid_thw")
    tensors: list[torch.Tensor] = []
    input_shapes: dict[str, list[int]] = {}
    for key in names:
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Qwen3.5 processor did not return required tensor input: {key}")
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
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
            "spatial_merge_size": spatial_merge_size,
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
    if (
        "decoder_step" in component_graphs
        and os.environ.get("CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE") != "1"
    ):
        return _run_causal_lm_cached_step_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=prompt,
            input_ids=input_ids,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
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


def _run_causal_lm_cached_step_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
    enable_thinking: bool,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    decoder_step = component_graphs.get("decoder_step")
    if decoder_step is None:
        raise ValueError("causal LM cached decode requires a decoder_step graph")
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

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
    input_names = component_input_names(decoder_step)
    if set(input_names) != {"input_ids", "position_ids"}:
        raise ValueError(
            "causal LM decoder_step must accept logical inputs ('input_ids', 'position_ids'), "
            f"got {input_names!r}"
        )
    input_buffers = _bind_zero_input_buffers(
        decoder_step,
        {"input_ids": np.int64, "position_ids": np.int64},
    )
    gdn_state_entries = [
        (layer_key, state_input, state_output)
        for layer_key, state_input, state_output in decoder_step.cache_state_tensors
        if str(layer_key).startswith("gdn:")
    ]
    gdn_state_buffers: dict[str, np.ndarray] = {}
    for layer_key, state_input, _ in gdn_state_entries:
        buffer = np.zeros(tuple(int(dim) for dim in state_input.shape), dtype=np.float16)
        decoder_step.graph.set_external_input(
            state_input,
            int(buffer.ctypes.data),
            dtype=state_input.dtype,
        )
        gdn_state_buffers[str(layer_key)] = buffer

    def _run_step_token(token_id: int, position_id: int, *, read_logits: bool) -> np.ndarray | None:
        input_buffers["input_ids"].fill(int(token_id))
        input_buffers["position_ids"].fill(int(position_id))
        decoder_step.graph.execute()
        for layer_key, _, state_output in gdn_state_entries:
            np.copyto(
                gdn_state_buffers[str(layer_key)],
                np.asarray(state_output.numpy(), dtype=np.float16),
            )
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
    prime_start = start
    for position_id, token_id in enumerate(prompt_token_ids):
        logits = _run_step_token(
            int(token_id),
            int(position_id),
            read_logits=position_id + 1 == len(prompt_token_ids),
        )
    prime_end = time.perf_counter()
    if logits is None:
        raise RuntimeError("causal LM cached decoder did not produce prompt logits")

    next_position_id = len(prompt_token_ids)
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
            next_position_id,
            read_logits=True,
        )
        if logits is None:
            raise RuntimeError("causal LM cached decoder did not produce decode logits")
        next_position_id += 1

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
        "decode_mode": "cached_step",
        "input_ids": prompt_token_ids,
        "input_shape": [1, len(prompt_token_ids)],
        "output_shape": logits_shape or [],
        "decoder_ms": (end - start) * 1000.0,
        "total_ms": (end - start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "cache_prime_ms": (prime_end - prime_start) * 1000.0,
        "cache_prime_tokens": len(prompt_token_ids),
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


def _run_seq2seq_cached_step_decode(
    *,
    decoder_step: LoadedComponentGraph,
    decoder_cross_kv: LoadedComponentGraph | None,
    manifest: dict[str, object],
    audio_file: str | Path,
    tokenizer: object | None,
    prompt_token_ids: list[int],
    encoder_hidden_states: np.ndarray,
    input_features: np.ndarray,
    active_frames: int,
    preprocess_start: float,
    preprocess_end: float,
    encoder_start: float,
    encoder_end: float,
    target_token_count: int,
    token_budget: int,
    suppress_tokens: list[int],
    begin_suppress_tokens: list[int],
    eos_token_id: object,
    encoded_stop_sequences: tuple[tuple[int, ...], ...],
    stop_reason_default: str,
) -> dict[str, object]:
    os.environ.setdefault("CACTUS_KV_CACHE_FP16", "1")

    input_names = component_input_names(decoder_step)
    cross_kv_store: dict[str, np.ndarray] = {}
    decoder_start = time.perf_counter()
    if decoder_cross_kv is not None:
        cross_store = {"encoder_hidden_states": encoder_hidden_states}
        execute_loaded_component(decoder_cross_kv, cross_store)
        cross_kv_store = {
            name: np.ascontiguousarray(value)
            for name, value in cross_store.items()
            if name.startswith(("cross_k_", "cross_v_")) and isinstance(value, np.ndarray)
        }

    old_step_inputs = input_names == ("decoder_input_ids", "encoder_hidden_states", "position_ids")
    new_step_inputs = (
        len(input_names) >= 2
        and input_names[:2] == ("decoder_input_ids", "position_ids")
        and all(name in cross_kv_store for name in input_names[2:])
    )
    if not old_step_inputs and not new_step_inputs:
        raise ValueError(
            "seq2seq cached decoder_step must accept logical inputs "
            "('decoder_input_ids', 'encoder_hidden_states', 'position_ids') or "
            "('decoder_input_ids', 'position_ids', cross_k/v...), "
            f"got {input_names!r}"
        )

    encoder_hidden_states = np.ascontiguousarray(encoder_hidden_states)
    runtime_inputs: list[np.ndarray] = []
    mutable_inputs: dict[str, np.ndarray] = {}
    for name, tensor in zip(input_names, decoder_step.runtime_inputs, strict=True):
        if name == "decoder_input_ids":
            value = np.zeros(tuple(int(dim) for dim in tensor.shape), dtype=np.int64)
            mutable_inputs[name] = value
        elif name == "position_ids":
            value = np.zeros(tuple(int(dim) for dim in tensor.shape), dtype=np.int64)
            mutable_inputs[name] = value
        elif name == "encoder_hidden_states":
            value = encoder_hidden_states
        elif name in cross_kv_store:
            value = cross_kv_store[name]
        else:
            raise RuntimeError(f"missing decoder_step input {name!r}")
        runtime_inputs.append(value)
    bound_inputs = decoder_step.set_external_inputs(runtime_inputs)
    for index, name in enumerate(input_names):
        if name in mutable_inputs:
            mutable_inputs[name] = bound_inputs[index]
        elif name == "encoder_hidden_states":
            encoder_hidden_states = bound_inputs[index]
        elif name in cross_kv_store:
            cross_kv_store[name] = bound_inputs[index]
    decoder_input_ids = mutable_inputs["decoder_input_ids"]
    position_ids = mutable_inputs["position_ids"]

    def _execute_step(token_id: int, position: int) -> np.ndarray:
        decoder_input_ids.fill(int(token_id))
        position_ids.fill(int(position))
        outputs = decoder_step.execute()
        if not outputs:
            raise RuntimeError("seq2seq decoder_step graph produced no outputs")
        logits = outputs[0].numpy()
        if logits.ndim != 3:
            raise RuntimeError(f"expected decoder_step logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        return np.asarray(logits)

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    first_token_ms = 0.0
    stop_reason = stop_reason_default
    logits: np.ndarray | None = None

    for position, token_id in enumerate(prompt_token_ids):
        logits = _execute_step(token_id, position)

    if logits is None:
        raise RuntimeError("seq2seq cached decoder did not produce prompt logits")

    for step_index in range(token_budget):
        logits_shape = list(logits.shape)
        next_token_id = _select_next_token_with_suppression(
            np.asarray(logits[0, -1]),
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
        if len(prompt_token_ids) + len(generated_ids) >= target_token_count:
            stop_reason = "context_limit"
            break
        if step_index + 1 >= token_budget:
            break

        logits = _execute_step(next_token_id, len(prompt_token_ids) + step_index)

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
        "decode_mode": "cached_step",
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
    if "audio_encoder" not in component_graphs or (
        "decoder" not in component_graphs and "decoder_step" not in component_graphs
    ):
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
    decoder = component_graphs.get("decoder")
    decoder_step = component_graphs.get("decoder_step")
    encoder_inputs = component_input_names(encoder)
    if encoder_inputs and encoder_inputs != ("input_features",):
        raise ValueError(
            "seq2seq_transcription audio_encoder must accept logical input ('input_features',), "
            f"got {encoder_inputs!r}"
        )
    decoder_inputs = component_input_names(decoder) if decoder is not None else ()
    if decoder is not None and decoder_inputs and decoder_inputs != ("decoder_input_ids", "encoder_hidden_states"):
        raise ValueError(
            "seq2seq_transcription decoder must accept logical inputs "
            "('decoder_input_ids', 'encoder_hidden_states'), "
            f"got {decoder_inputs!r}"
        )
    decoder_step_inputs = component_input_names(decoder_step) if decoder_step is not None else ()
    valid_decoder_step_inputs = (
        decoder_step_inputs == ("decoder_input_ids", "encoder_hidden_states", "position_ids")
        or (
            len(decoder_step_inputs) >= 2
            and decoder_step_inputs[:2] == ("decoder_input_ids", "position_ids")
            and all(name.startswith(("cross_k_", "cross_v_")) for name in decoder_step_inputs[2:])
        )
    )
    if decoder_step is not None and decoder_step_inputs and not valid_decoder_step_inputs:
        raise ValueError(
            "seq2seq_transcription decoder_step must accept logical inputs "
            "('decoder_input_ids', 'encoder_hidden_states', 'position_ids') or "
            "('decoder_input_ids', 'position_ids', cross_k/v...), "
            f"got {decoder_step_inputs!r}"
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

    if decoder_step is not None and os.environ.get("CACTUS_TRANSPILER_DISABLE_SEQ2SEQ_CACHED_STEP") != "1":
        return _run_seq2seq_cached_step_decode(
            decoder_step=decoder_step,
            decoder_cross_kv=component_graphs.get("decoder_cross_kv"),
            manifest=manifest,
            audio_file=audio_file,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            encoder_hidden_states=encoder_hidden_states,
            input_features=input_features,
            active_frames=active_frames,
            preprocess_start=preprocess_start,
            preprocess_end=preprocess_end,
            encoder_start=encoder_start,
            encoder_end=encoder_end,
            target_token_count=target_token_count,
            token_budget=token_budget,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            eos_token_id=eos_token_id,
            encoded_stop_sequences=encoded_stop_sequences,
            stop_reason_default="max_new_tokens",
        )

    if decoder is None:
        raise ValueError("seq2seq_transcription bundle must include decoder or decoder_step graph")

    input_array = np.full((1, target_token_count), padding_token_id, dtype=np.int64)
    input_array[0, : len(prompt_token_ids)] = np.asarray(prompt_token_ids, dtype=np.int64)
    if hasattr(decoder, "set_external_inputs"):
        bound_decoder_inputs = decoder.set_external_inputs([input_array, encoder_hidden_states])
        input_array = bound_decoder_inputs[0]
        encoder_hidden_states = bound_decoder_inputs[1]
    else:
        decoder.set_inputs([input_array, encoder_hidden_states])

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


def _load_bundle_processor(manifest: dict[str, object]):
    _patch_missing_lzma_backport()
    processor_sources = tuple(_pretrained_source_candidates(manifest, processor=True))
    if not processor_sources:
        raise ValueError("bundle manifest is missing model_source/model_id for multimodal preprocessing")
    cached = _PROCESSOR_CACHE.get(processor_sources)
    if cached is not None:
        return cached

    try:
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required for multimodal preprocessing: {exc}") from exc

    processor_errors: list[str] = []
    for source in processor_sources:
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                local_files_only=Path(source).exists(),
                trust_remote_code=True,
            )
            _PROCESSOR_CACHE[processor_sources] = processor
            return processor
        except Exception as exc:
            processor_errors.append(f"{source}: {exc}")
    raise RuntimeError(
        "failed to load tokenizer/processor assets for multimodal preprocessing. "
        "Re-run cactus convert so processor files are copied into the weights folder. "
        f"Tried: {'; '.join(processor_errors)}"
    )


def _load_bundle_tokenizer(manifest: dict[str, object]):
    _patch_missing_lzma_backport()
    tokenizer_sources = tuple(_pretrained_source_candidates(manifest, processor=False))
    if not tokenizer_sources:
        raise ValueError("bundle manifest is missing model_source/model_id; provide --input-ids instead")
    cached = _TOKENIZER_CACHE.get(tokenizer_sources)
    if cached is not None:
        return cached

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required to tokenize --prompt: {exc}") from exc

    errors: list[str] = []
    for source in tokenizer_sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=Path(source).exists(),
                trust_remote_code=True,
            )
            _TOKENIZER_CACHE[tokenizer_sources] = tokenizer
            return tokenizer
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
    profile = profile_for_family(family)
    prompt_style = profile.prompt_style if profile is not None else "auto"
    if prompt_style == "chatml":
        return _encode_prompt_text(
            tokenizer,
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
    if prompt_style == "lfm_chat":
        return _encode_prompt_text(
            tokenizer,
            f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
    if prompt_style == "qwen_chat":
        thinking_prefix = "<think>\n" if enable_thinking_if_supported else "<think>\n\n</think>\n\n"
        return _encode_prompt_text(
            tokenizer,
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{thinking_prefix}",
        )
    if prompt_style == "gemma4":
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
    profile = profile_for_family(family)
    stop_tokens = profile.stop_tokens if profile is not None else ()

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


def _select_next_token_with_repetition_penalty(
    logits: np.ndarray,
    *,
    token_ids: list[int] | tuple[int, ...],
    penalty: float,
    no_repeat_ngram_size: int = 0,
) -> int:
    adjusted = np.asarray(logits, dtype=np.float32).copy()
    vocab_size = int(adjusted.shape[-1])
    if penalty > 1.0 and token_ids:
        for token_id in set(int(value) for value in token_ids):
            if 0 <= token_id < vocab_size:
                adjusted[token_id] = adjusted[token_id] / penalty if adjusted[token_id] > 0 else adjusted[token_id] * penalty
    _mask_repeated_ngrams(adjusted, token_ids=token_ids, ngram_size=no_repeat_ngram_size)
    return int(np.argmax(adjusted))


def _lfm2_repetition_penalty() -> float:
    raw_value = os.environ.get("CACTUS_LFM2_REPETITION_PENALTY", "1.25")
    try:
        return max(1.0, float(raw_value or "1.25"))
    except ValueError:
        return 1.25


def _lfm2_no_repeat_ngram_size() -> int:
    raw_value = os.environ.get("CACTUS_LFM2_NO_REPEAT_NGRAM", "4")
    try:
        return max(0, int(raw_value or "4"))
    except ValueError:
        return 4


def _mask_repeated_ngrams(
    logits: np.ndarray,
    *,
    token_ids: list[int] | tuple[int, ...],
    ngram_size: int,
) -> None:
    if ngram_size <= 1 or len(token_ids) + 1 < ngram_size:
        return
    prefix_length = ngram_size - 1
    current_prefix = tuple(int(value) for value in token_ids[-prefix_length:])
    if len(current_prefix) != prefix_length:
        return
    vocab_size = int(logits.shape[-1])
    for index in range(0, len(token_ids) - prefix_length):
        previous_prefix = tuple(int(value) for value in token_ids[index : index + prefix_length])
        if previous_prefix != current_prefix:
            continue
        blocked_token = int(token_ids[index + prefix_length])
        if 0 <= blocked_token < vocab_size:
            logits[blocked_token] = -np.inf


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
