from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.capture_jax import capture_jax_graphs
from cactus.transpile.capture_jax import CapturedJaxGraphBundle
from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.capture_jax import _flatten_named_leaves


@dataclass(frozen=True)
class JaxUserGraphBundleResult:
    bundle: CapturedJaxGraphBundle
    output_dir: Path
    components_manifest_path: Path
    weights_dir: Path


def flatten_jax_params(params: object) -> dict[str, np.ndarray]:
    import jax

    return {name: np.asarray(value) for name, value in _flatten_named_leaves(jax.tree_util, params)}


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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


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
) -> JaxUserGraphBundleResult:
    """Capture user-supplied JAX entrypoints and write a Cactus component bundle.

    The caller owns model loading, tokenization, masks, and graph boundaries.
    This function only needs params, example tensors, named graph functions, and
    logical input/output names.
    """
    root = Path(output_dir)
    weights_dir = root / "weights"
    component_root = root / "components"
    component_root.mkdir(parents=True, exist_ok=True)

    arrays = weight_arrays if weight_arrays is not None else flatten_jax_params(params)
    write_fp16_weights_manifest(weights_dir, arrays, exclude=exclude_weights)

    bundle = capture_jax_graphs(
        params,
        specs,
        weights_dir=str(weights_dir),
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "jax_user_graph_bundle",
            **dict(graph_meta or {}),
        },
    )
    components = []
    for name, captured in bundle.graphs.items():
        component_dir = component_root / name
        component_dir.mkdir(parents=True, exist_ok=True)
        graph_path = component_dir / "graph.cactus"
        captured.graph.graph.save(str(graph_path))
        components.append(
            {
                "component": name,
                "directory": str(component_dir.relative_to(root)),
                "raw_ir": None,
                "optimized_ir": None,
                "graph": str(graph_path.relative_to(root)),
                "inputs": list(captured.ir_graph.inputs),
                "outputs": list(captured.ir_graph.outputs),
                "logical_inputs": list(captured.spec.input_names or ()),
                "logical_outputs": list(captured.spec.output_names or ()),
                "node_count": len(captured.ir_graph.order),
                "weight_binding_count": len(captured.graph.bound_constant_bindings),
                "runtime_input_node_ids": [int(tensor.id) for tensor in captured.graph.runtime_inputs],
                "output_node_ids": [int(tensor.id) for tensor in captured.graph.outputs],
                "cache_state_node_ids": [],
                "bound_constant_bindings": captured.graph.bound_constant_bindings,
            }
        )

    manifest = {
        "model_id": model_id,
        "model_source": "jax_user_graph",
        "task": task,
        "family": family,
        "component_order": [spec.name for spec in specs],
        "inputs": dict(inputs_metadata or {}),
        "components": components,
        "weights_dir": str(weights_dir),
        "weights_manifest": str(weights_dir / "weights_manifest.json"),
    }
    manifest_path = component_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return JaxUserGraphBundleResult(
        bundle=bundle,
        output_dir=root,
        components_manifest_path=manifest_path,
        weights_dir=weights_dir,
    )
