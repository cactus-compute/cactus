from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
import re
from typing import Any

import numpy as np
import torch

from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.lower import TranspiledGraph
from cactus.transpile.lower import transpile_preoptimized_ir
from cactus.transpile.optimize_graph import FusionConfig
from cactus.transpile.optimize_graph import optimize_graph


@dataclass
class ComponentModuleSpec:
    component: str
    module: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    graph_meta: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class CapturedComponent:
    component: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    raw_ir_graph: IRGraph
    optimized_ir_graph: IRGraph
    transpiled_graph: TranspiledGraph
    metadata: dict[str, object] = field(default_factory=dict)


_GRAPH_ARG_INPUT_RE = re.compile(r"^v_args_(\d+)$")


def _align_input_keys_to_graph_inputs(
    declared_input_keys: tuple[str, ...],
    graph_inputs: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    aligned: list[str] = []
    for value_id in graph_inputs:
        match = _GRAPH_ARG_INPUT_RE.match(str(value_id))
        if match is None:
            return declared_input_keys
        input_index = int(match.group(1))
        if input_index < 0 or input_index >= len(declared_input_keys):
            return declared_input_keys
        aligned.append(declared_input_keys[input_index])
    return tuple(aligned)


def _call_optional_capture_hook(spec: ComponentModuleSpec, attr_name: str) -> None:
    hook = getattr(spec.module, attr_name, None)
    if not callable(hook):
        return
    try:
        hook(
            component=spec.component,
            example_inputs=spec.example_inputs,
            input_keys=spec.input_keys,
            output_keys=spec.output_keys,
        )
    except TypeError:
        hook()


def capture_component_spec(
    spec: ComponentModuleSpec,
    *,
    fusion_config: FusionConfig | None = None,
) -> CapturedComponent:
    wrapped = _ComponentCaptureWrapper(spec.module, graph_meta=spec.graph_meta).eval()
    capture_error: Exception | None = None
    _call_optional_capture_hook(spec, "prepare_for_capture")
    try:
        captured = capture_model(wrapped, spec.example_inputs)
    except Exception as exc:
        capture_error = exc
        raise
    finally:
        try:
            _call_optional_capture_hook(spec, "restore_after_capture")
        except Exception:
            if capture_error is None:
                raise
    raw_ir_graph = copy.deepcopy(captured.ir_graph)
    optimized_ir_graph = copy.deepcopy(captured.ir_graph)
    canonicalize_exported_graph(optimized_ir_graph)
    optimize_graph(optimized_ir_graph, config=fusion_config)
    transpiled_graph = transpile_preoptimized_ir(copy.deepcopy(optimized_ir_graph))
    if len(spec.output_keys) != len(transpiled_graph.outputs):
        raise ValueError(
            f"component {spec.component} declared {len(spec.output_keys)} output keys but lowered "
            f"{len(transpiled_graph.outputs)} graph outputs"
        )
    aligned_input_keys = _align_input_keys_to_graph_inputs(spec.input_keys, optimized_ir_graph.inputs)
    return CapturedComponent(
        component=spec.component,
        input_keys=aligned_input_keys,
        output_keys=spec.output_keys,
        raw_ir_graph=raw_ir_graph,
        optimized_ir_graph=optimized_ir_graph,
        transpiled_graph=transpiled_graph,
        metadata=dict(spec.metadata),
    )


def execute_component_pipeline(
    components: list[CapturedComponent],
    *,
    initial_store: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    store: dict[str, np.ndarray] = {}
    for key, value in initial_store.items():
        store[key] = _to_numpy(value)

    outputs_by_component: dict[str, list[np.ndarray]] = {}
    for component in components:
        runtime_inputs = []
        for input_key in component.input_keys:
            if input_key not in store:
                raise KeyError(
                    f"component {component.component} is missing pipeline input {input_key!r}"
                )
            runtime_inputs.append(store[input_key])
        component.transpiled_graph.set_inputs(runtime_inputs)
        raw_outputs = component.transpiled_graph.execute()
        numpy_outputs = [output.numpy().copy() for output in raw_outputs]
        if len(numpy_outputs) != len(component.output_keys):
            raise ValueError(
                f"component {component.component} produced {len(numpy_outputs)} outputs, "
                f"expected {len(component.output_keys)}"
            )
        for output_key, value in zip(component.output_keys, numpy_outputs, strict=True):
            store[output_key] = value
        outputs_by_component[component.component] = numpy_outputs

    return store, outputs_by_component


def initial_store_from_named_tensors(
    keys: tuple[str, ...] | list[str],
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> dict[str, np.ndarray]:
    if len(keys) != len(tensors):
        raise ValueError(f"expected {len(keys)} tensors, got {len(tensors)}")
    return {
        str(key): _to_numpy(tensor)
        for key, tensor in zip(keys, tensors, strict=True)
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    raise TypeError(f"unsupported pipeline value type: {type(value).__name__}")


class _ComponentCaptureWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, *, graph_meta: dict[str, object]):
        super().__init__()
        self.module = module
        self.graph_meta = dict(graph_meta)

    def forward(self, *args: torch.Tensor) -> Any:
        return self.module(*args)

    def get_transpile_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        provider = getattr(self.module, "get_transpile_metadata", None)
        if callable(provider):
            provided = provider()
            if isinstance(provided, dict):
                metadata.update(provided)
        graph_meta = {}
        existing = metadata.get("graph", {})
        if isinstance(existing, dict):
            graph_meta.update(existing)
        graph_meta.update(self.graph_meta)
        metadata["graph"] = graph_meta
        return metadata
