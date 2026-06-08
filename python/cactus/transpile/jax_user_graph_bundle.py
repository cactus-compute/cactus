from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.capture_jax import capture_jax_graphs
from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.lower import TranspiledGraph
from cactus.transpile.runtime_compat import Graph
from cactus.transpile.runtime_compat import Tensor


@dataclass(frozen=True)
class JaxUserGraphBundleResult:
    bundle: Any
    output_dir: Path
    components_manifest_path: Path
    weights_dir: Path


@dataclass
class LoadedJaxUserGraph:
    name: str
    graph: Graph
    runtime_inputs: list[Tensor]
    outputs: list[Tensor]
    logical_inputs: list[str]
    logical_outputs: list[str]

    def set_inputs(self, inputs: Sequence[Any]) -> None:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"graph {self.name!r} expected {len(self.runtime_inputs)} inputs, got {len(inputs)}"
            )
        for tensor, value in zip(self.runtime_inputs, inputs, strict=True):
            if isinstance(value, Tensor):
                if tuple(int(dim) for dim in value.shape) != tuple(int(dim) for dim in tensor.shape):
                    raise ValueError(
                        f"graph {self.name!r} input shape mismatch for node {tensor.id}: "
                        f"expected {tensor.shape}, got {value.shape}"
                    )
                if int(value.dtype) != int(tensor.dtype):
                    raise ValueError(
                        f"graph {self.name!r} input dtype mismatch for node {tensor.id}: "
                        f"expected {tensor.dtype}, got {value.dtype}"
                    )
                self.graph.set_input(tensor, value.numpy())
            else:
                self.graph.set_input(tensor, np.asarray(value))

    def execute(self, *inputs: Any) -> list[Tensor]:
        self.set_inputs(inputs)
        self.graph.execute()
        return self.outputs

    def reset(self) -> None:
        self.graph.hard_reset()


@dataclass
class LoadedJaxUserGraphBundle:
    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    graphs: dict[str, LoadedJaxUserGraph]

    def execute(self, graph_name: str, *inputs: Any) -> list[Tensor]:
        if graph_name not in self.graphs:
            available = ", ".join(sorted(self.graphs)) or "<none>"
            raise ValueError(f"unknown JAX user graph {graph_name!r}; available graphs: {available}")
        return self.graphs[graph_name].execute(*inputs)

    def reset(self, graph_name: str | None = None) -> None:
        if graph_name is not None:
            if graph_name not in self.graphs:
                available = ", ".join(sorted(self.graphs)) or "<none>"
                raise ValueError(f"unknown JAX user graph {graph_name!r}; available graphs: {available}")
            self.graphs[graph_name].reset()
            return
        for graph in self.graphs.values():
            graph.reset()


def flatten_jax_params(params: object) -> dict[str, np.ndarray]:
    import jax

    return {
        _tree_path_name(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
    }


def _tree_path_name(path: Sequence[Any]) -> str:
    parts: list[str] = []
    for entry in path:
        if hasattr(entry, "key"):
            part = str(entry.key)
        elif hasattr(entry, "name"):
            part = str(entry.name)
        elif hasattr(entry, "idx"):
            part = str(entry.idx)
        else:
            part = str(entry)
        parts.append(part)
    return ".".join(parts) or "param"


def write_fp16_weights_manifest(
    weights_dir: str | Path,
    params: dict[str, object],
    *,
    exclude: set[str] | None = None,
) -> Path:
    weights_root = Path(weights_dir)
    weights_root.mkdir(parents=True, exist_ok=True)
    excluded = exclude or set()
    manifest: dict[str, dict[str, str]] = {}
    for name, value in params.items():
        if name in excluded:
            continue
        filename = f"{name}.weights"
        save_tensor_with_header(np.asarray(value), weights_root / filename, precision="FP16")
        manifest[name] = {"filename": filename, "kind": "weight"}
    manifest_path = weights_root / "weights_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def load_jax_user_graph_bundle(bundle_dir_or_manifest: str | Path) -> LoadedJaxUserGraphBundle:
    manifest_path = _resolve_components_manifest(bundle_dir_or_manifest)
    root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    weights_dir = _resolve_manifest_path(root, manifest.get("weights_dir"))
    graphs: dict[str, LoadedJaxUserGraph] = {}
    for component in manifest.get("components", []):
        if not isinstance(component, dict):
            continue
        name = str(component.get("component", "") or "")
        graph_relpath = component.get("graph")
        if not name or not isinstance(graph_relpath, str) or not graph_relpath:
            continue
        graph = Graph.load(_resolve_manifest_path(root, graph_relpath))
        for binding in component.get("bound_constant_bindings", []) or []:
            if not isinstance(binding, dict):
                continue
            node_id = binding.get("node_id")
            path = binding.get("path")
            if not isinstance(node_id, int) or not isinstance(path, str):
                continue
            graph.bind_mmap_weights(
                graph._tensor_from_node(int(node_id)),
                _resolve_weight_path(root, weights_dir, path),
            )
        graphs[name] = LoadedJaxUserGraph(
            name=name,
            graph=graph,
            runtime_inputs=[
                graph._tensor_from_node(int(node_id))
                for node_id in component.get("runtime_input_node_ids", []) or []
            ],
            outputs=[
                graph._tensor_from_node(int(node_id))
                for node_id in component.get("output_node_ids", []) or []
            ],
            logical_inputs=[str(value) for value in component.get("logical_inputs", []) or []],
            logical_outputs=[str(value) for value in component.get("logical_outputs", []) or []],
        )
    return LoadedJaxUserGraphBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        graphs=graphs,
    )


def _resolve_components_manifest(bundle_dir_or_manifest: str | Path) -> Path:
    path = Path(bundle_dir_or_manifest).expanduser()
    if path.is_dir():
        components_manifest = path / "components" / "manifest.json"
        if components_manifest.exists():
            return components_manifest
        if path.name == "components" and (path / "manifest.json").exists():
            return path / "manifest.json"
    if path.exists():
        return path
    raise FileNotFoundError(f"JAX user graph bundle manifest not found: {path}")


def _reject_relative_escape(value: str, *, field: str) -> None:
    """Reject relative manifest entries that try to escape via `..`. Absolute
    paths are a legitimate workflow (shared weights cache) and stay allowed."""
    if not value:
        return
    p = Path(value)
    if p.is_absolute():
        return
    if any(part == ".." for part in p.parts):
        raise RuntimeError(
            f"refusing to follow bundle manifest relative path with '..' segments: "
            f"{field}={value}"
        )


def _resolve_manifest_path(root: Path, value: object) -> str:
    if not isinstance(value, str) or not value:
        return str(root)
    _reject_relative_escape(value, field="manifest path")
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else root / path)


def _resolve_weight_path(root: Path, weights_dir: str, value: str) -> str:
    _reject_relative_escape(value, field="weight path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = Path(weights_dir) / path
    if candidate.exists():
        return str(candidate)
    return str(root / path)


def _embed_materialized_bound_constants(transpiled_graph: TranspiledGraph) -> None:
    external_node_ids = {
        int(binding["node_id"])
        for binding in transpiled_graph.bound_constant_bindings
    }
    for constant in transpiled_graph.bound_constants:
        if int(constant.id) in external_node_ids:
            continue
        transpiled_graph.graph.mark_embedded_input(constant)


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
        payload: dict[str, object] = {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if value.numel() == 1:
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


def _ir_graph_to_dict(graph: IRGraph) -> dict[str, object]:
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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to path via tempfile + os.replace so a crash mid-write
    cannot leave a half-written file the next loader would choke on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_jax_user_graph_bundle(
    *,
    params: Any,
    specs: Sequence[JaxGraphSpec],
    output_dir: str | Path,
    model_id: str,
    task: str = "generic",
    family: str = "jax_user_graph",
    inputs_metadata: dict[str, object] | None = None,
    graph_meta: dict[str, object] | None = None,
    weight_arrays: dict[str, object] | None = None,
    exclude_weights: set[str] | None = None,
    weights_dir: str | Path | None = None,
) -> JaxUserGraphBundleResult:
    arrays = weight_arrays if weight_arrays is not None else flatten_jax_params(params)
    root = Path(output_dir)
    weights_root = Path(weights_dir) if weights_dir is not None else root
    write_fp16_weights_manifest(weights_root, arrays, exclude=exclude_weights)

    bundle = capture_jax_graphs(
        params,
        specs,
        weights_dir=str(weights_root),
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "jax_user_graph_bundle",
            **dict(graph_meta or {}),
        },
    )
    return _write_jax_user_graph_bundle(
        bundle=bundle,
        specs=specs,
        output_dir=root,
        weights_dir=weights_root,
        model_id=model_id,
        task=task,
        family=family,
        inputs_metadata=inputs_metadata,
    )


def build_jax_generation_graph_bundle(
    *,
    params: Any,
    output_dir: str | Path,
    model_id: str,
    encoder: JaxGraphSpec | None = None,
    decoder_prefill: JaxGraphSpec | None = None,
    decoder_step: JaxGraphSpec | None = None,
    task: str = "text-generation",
    family: str = "jax_user_graph",
    inputs_metadata: dict[str, object] | None = None,
    graph_meta: dict[str, object] | None = None,
    weight_arrays: dict[str, object] | None = None,
    exclude_weights: set[str] | None = None,
    weights_dir: str | Path | None = None,
) -> JaxUserGraphBundleResult:
    root = Path(output_dir)
    weights_root = Path(weights_dir) if weights_dir is not None else root
    arrays = weight_arrays if weight_arrays is not None else flatten_jax_params(params)
    write_fp16_weights_manifest(weights_root, arrays, exclude=exclude_weights)

    input_specs = tuple(spec for spec in (encoder, decoder_prefill, decoder_step) if spec is not None)
    bundle = capture_jax_generation_graphs(
        params,
        encoder=encoder,
        decoder_prefill=decoder_prefill,
        decoder_step=decoder_step,
        weights_dir=str(weights_root),
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "jax_user_graph_bundle",
            **dict(graph_meta or {}),
        },
    )
    return _write_jax_user_graph_bundle(
        bundle=bundle,
        specs=input_specs,
        output_dir=root,
        weights_dir=weights_root,
        model_id=model_id,
        task=task,
        family=family,
        inputs_metadata=inputs_metadata,
    )


def capture_jax_generation_graphs(
    params: Any,
    *,
    encoder: JaxGraphSpec | None = None,
    decoder_prefill: JaxGraphSpec | None = None,
    decoder_step: JaxGraphSpec | None = None,
    weights_dir: str | None = None,
    weight_bindings: dict[str, dict[str, str]] | None = None,
    graph_meta: dict[str, object] | None = None,
) -> Any:
    specs: list[JaxGraphSpec] = []

    def _with_meta(
        spec: JaxGraphSpec | None,
        *,
        name: str,
        role: str,
        component: str,
    ) -> None:
        if spec is None:
            return
        specs.append(
            JaxGraphSpec(
                name=spec.name or name,
                role=spec.role if spec.role != "generic" else role,
                fn=spec.fn,
                example_args=spec.example_args,
                input_names=spec.input_names,
                output_names=spec.output_names,
                graph_meta={
                    "component": component,
                    **dict(spec.graph_meta or {}),
                },
            )
        )

    _with_meta(encoder, name="encoder", role="encoder", component="encoder")
    _with_meta(
        decoder_prefill,
        name="decoder_prefill",
        role="decoder_prefill",
        component="decoder_prefill_chunk",
    )
    _with_meta(
        decoder_step,
        name="decoder_step",
        role="decoder_step",
        component="decoder_step",
    )
    if not specs:
        raise ValueError("capture_jax_generation_graphs requires at least one graph spec")
    return capture_jax_graphs(
        params,
        specs,
        weights_dir=weights_dir,
        weight_bindings=weight_bindings,
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "generation",
            **dict(graph_meta or {}),
        },
    )


def _write_jax_user_graph_bundle(
    *,
    bundle: Any,
    specs: Sequence[JaxGraphSpec],
    output_dir: Path,
    weights_dir: Path,
    model_id: str,
    task: str,
    family: str,
    inputs_metadata: dict[str, object] | None,
) -> JaxUserGraphBundleResult:
    root = output_dir
    weights_root = weights_dir
    component_root = root / "components"
    component_root.mkdir(parents=True, exist_ok=True)

    components = []
    component_order = [spec.name for spec in specs]
    base_payload = {
        "model_id": model_id,
        "model_source": "jax_user_graph",
        "task": task,
        "family": family,
        "inputs": _serialize_json_compatible(dict(inputs_metadata or {})),
    }
    for name, captured in bundle.graphs.items():
        component_dir = component_root / name
        component_dir.mkdir(parents=True, exist_ok=True)
        graph_path = component_dir / "graph.cactus"
        raw_ir_path = component_dir / "raw_ir.json"
        optimized_ir_path = component_dir / "optimized_ir.json"
        _write_json(
            raw_ir_path,
            {
                **base_payload,
                "component": name,
                "graph": _ir_graph_to_dict(captured.raw_ir_graph),
            },
        )
        _write_json(
            optimized_ir_path,
            {
                **base_payload,
                "component": name,
                "graph": _ir_graph_to_dict(captured.ir_graph),
            },
        )
        _embed_materialized_bound_constants(captured.graph)
        captured.graph.graph.save(str(graph_path))
        components.append(
            {
                "component": name,
                "directory": str(component_dir.relative_to(root)),
                "raw_ir": str(raw_ir_path.relative_to(root)),
                "optimized_ir": str(optimized_ir_path.relative_to(root)),
                "graph": str(graph_path.relative_to(root)),
                "inputs": list(captured.ir_graph.inputs),
                "outputs": list(captured.ir_graph.outputs),
                "logical_inputs": list(captured.spec.input_names or ()),
                "logical_outputs": list(captured.spec.output_names or ()),
                "node_count": len(captured.ir_graph.order),
                "weight_binding_count": len(captured.graph.bound_constant_bindings),
                "runtime_input_node_ids": [int(tensor.id) for tensor in captured.graph.runtime_inputs],
                "output_node_ids": [int(tensor.id) for tensor in captured.graph.outputs],
                "cache_state_node_ids": [
                    {
                        "layer_key": str(layer_key),
                        "key": int(key_tensor.id),
                        "value": int(value_tensor.id),
                    }
                    for layer_key, key_tensor, value_tensor in getattr(captured.graph, "cache_state_tensors", [])
                ],
                "bound_constant_bindings": captured.graph.bound_constant_bindings,
            }
        )

    manifest = {
        "model_id": model_id,
        "model_source": "jax_user_graph",
        "task": task,
        "family": family,
        "component_order": component_order,
        "inputs": _serialize_json_compatible(dict(inputs_metadata or {})),
        "components": components,
        "weights_dir": str(weights_root),
        "weights_manifest": str(weights_root / "weights_manifest.json"),
    }
    manifest_path = component_root / "manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    return JaxUserGraphBundleResult(
        bundle=bundle,
        output_dir=root,
        components_manifest_path=manifest_path,
        weights_dir=weights_root,
    )
