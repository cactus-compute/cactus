from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class RuntimeTensorBinding:
    node_id: int
    logical_name: str


@dataclass(slots=True, frozen=True)
class ConstantBinding:
    node_id: int
    path: str


@dataclass(slots=True)
class CacheStateBinding:
    layer_key: str
    key_node_id: int = -1
    value_node_id: int = -1
    cache_kind: str = "kv"
    tensor_indices: tuple[int, ...] = ()


@dataclass(slots=True, frozen=True)
class RuntimeRouteEdge:
    inputs: tuple[str, ...]
    output: str


@dataclass(slots=True, frozen=True)
class RuntimeRoute:
    name: str
    edges: tuple[RuntimeRouteEdge, ...] = ()


@dataclass(slots=True, frozen=True)
class RuntimeComponent:
    component: str
    graph: str
    runtime_input_node_ids: tuple[int, ...] = ()
    logical_inputs: tuple[str, ...] = ()
    output_node_ids: tuple[int, ...] = ()
    logical_outputs: tuple[str, ...] = ()
    bound_constant_bindings: tuple[ConstantBinding, ...] = ()
    cache_state_node_ids: tuple[CacheStateBinding, ...] = ()
    metadata: dict[str, str] | None = None
    unsupported_nodes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_engine_dict(self) -> dict[str, Any]:
        return runtime_component_to_engine_dict(self)

    def to_plan_dict(self) -> dict[str, Any]:
        return runtime_component_to_plan_dict(self)


@dataclass(slots=True, frozen=True)
class RuntimePlan:
    family: str = ""
    components: tuple[RuntimeComponent, ...] = ()
    routes: tuple[RuntimeRoute, ...] = ()
    metadata: dict[str, str] | None = None

    def to_engine_manifest(self) -> dict[str, Any]:
        return runtime_plan_to_engine_manifest(self)

    def to_plan_dict(self) -> dict[str, Any]:
        return runtime_plan_to_dict(self)

    def write(self, bundle_dir: str | Path) -> tuple[Path, Path]:
        return write_runtime_plan(self, bundle_dir)


################################################# Runtime Plan Utils!!!!!!! #################################################


def cache_state_binding_from_annotation(annotation: Any, node_id: int) -> CacheStateBinding:
    cache_kind = str(getattr(annotation, "kind", "kv"))
    role = str(getattr(annotation, "role", "state"))
    tensor_index = getattr(annotation, "tensor_index", None)
    tensor_indices = (int(tensor_index),) if tensor_index is not None else ()

    key_node_id = -1
    value_node_id = -1

    if cache_kind == "conv" or role == "state":
        key_node_id = node_id
        value_node_id = node_id
    elif role == "key":
        key_node_id = node_id
    elif role == "value":
        value_node_id = node_id
    else:
        key_node_id = node_id
        value_node_id = node_id

    return CacheStateBinding(
        layer_key=cache_layer_key(annotation),
        key_node_id=key_node_id,
        value_node_id=value_node_id,
        cache_kind=cache_kind,
        tensor_indices=tensor_indices,
    )


def cache_layer_key(annotation: Any) -> str:
    cache_kind = str(getattr(annotation, "kind", "cache"))
    layer_index = getattr(annotation, "layer_index", None)
    tensor_index = getattr(annotation, "tensor_index", None)

    if layer_index is not None:
        return f"{cache_kind}:{int(layer_index)}"

    if tensor_index is not None:
        return f"{cache_kind}:tensor:{int(tensor_index)}"

    return f"{cache_kind}:unknown"


def add_runtime_tensor_binding(bindings: list[RuntimeTensorBinding], binding: RuntimeTensorBinding) -> None:
    for index, existing in enumerate(bindings):
        if existing.node_id != binding.node_id:
            continue

        if existing.logical_name == binding.logical_name:
            return

        bindings[index] = binding
        return

    bindings.append(binding)


def merge_cache_state_binding(bindings: list[CacheStateBinding], binding: CacheStateBinding) -> None:
    for existing in bindings:
        if existing.layer_key != binding.layer_key:
            continue

        if binding.key_node_id >= 0:
            existing.key_node_id = binding.key_node_id

        if binding.value_node_id >= 0:
            existing.value_node_id = binding.value_node_id

        existing.tensor_indices = unique_ints((*existing.tensor_indices, *binding.tensor_indices))
        return

    bindings.append(binding)


def cache_state_binding_from_dict(data: dict[str, Any]) -> CacheStateBinding:
    return CacheStateBinding(
        layer_key=str(data.get("layer_key", "")),
        key_node_id=int(data.get("key", data.get("key_node_id", -1))),
        value_node_id=int(data.get("value", data.get("value_node_id", -1))),
        cache_kind=str(data.get("cache_kind", "kv")),
        tensor_indices=tuple(int(value) for value in data.get("tensor_indices", ())),
    )


def cache_state_binding_to_dict(binding: CacheStateBinding) -> dict[str, Any]:
    return {
        "layer_key": binding.layer_key,
        "key": binding.key_node_id,
        "value": binding.value_node_id,
        "cache_kind": binding.cache_kind,
        "tensor_indices": list(binding.tensor_indices),
    }


def constant_binding_from_dict(data: dict[str, Any]) -> ConstantBinding:
    return ConstantBinding(
        node_id=int(data["node_id"]),
        path=str(data["path"]),
    )


def constant_binding_to_dict(binding: ConstantBinding) -> dict[str, Any]:
    return {
        "node_id": binding.node_id,
        "path": binding.path,
    }


def runtime_component_from_generator_manifest(
    manifest_path: str | Path,
    *,
    bundle_dir: str | Path | None = None,
    component_name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> RuntimeComponent:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    graph_path = str(data.get("graph") or data.get("graph_path") or "")

    return RuntimeComponent(
        component=component_name or str(data["component"]),
        graph=runtime_graph_path(path, graph_path, Path(bundle_dir) if bundle_dir is not None else None),
        runtime_input_node_ids=tuple(int(value) for value in data.get("runtime_input_node_ids", ())),
        logical_inputs=tuple(str(value) for value in data.get("logical_inputs", ())),
        output_node_ids=tuple(int(value) for value in data.get("output_node_ids", ())),
        logical_outputs=tuple(str(value) for value in data.get("logical_outputs", ())),
        bound_constant_bindings=constant_bindings_from_generator_manifest(data),
        cache_state_node_ids=tuple(
            cache_state_binding_from_dict(value)
            for value in data.get("cache_state_node_ids", ())
            if isinstance(value, dict)
        ),
        metadata=string_dict({**string_dict(data.get("metadata")), **string_dict(metadata)}),
        unsupported_nodes=tuple(str(value) for value in data.get("unsupported_nodes", ())),
        warnings=tuple(str(value) for value in data.get("warnings", ())),
    )


def runtime_plan_from_generator_manifests(
    component_manifest_paths: dict[str, str | Path],
    *,
    bundle_dir: str | Path | None = None,
    model_profile: Any | None = None,
    component_name_map: dict[str, str] | None = None,
    component_metadata: dict[str, dict[str, str]] | None = None,
    metadata: dict[str, str] | None = None,
) -> RuntimePlan:
    components = tuple(
        runtime_component_from_generator_manifest(
            manifest_path,
            bundle_dir=bundle_dir,
            component_name=(component_name_map or {}).get(name, name),
            metadata=(component_metadata or {}).get(name),
        )
        for name, manifest_path in component_manifest_paths.items()
    )

    return RuntimePlan(
        family=runtime_family_from_model_profile(model_profile),
        components=components,
        routes=runtime_routes_from_model_profile(model_profile),
        metadata=string_dict(metadata),
    )


def runtime_plan_from_generation_result(
    generation_result: Any,
    *,
    bundle_dir: str | Path | None = None,
    model_profile: Any | None = None,
    component_name_map: dict[str, str] | None = None,
    component_metadata: dict[str, dict[str, str]] | None = None,
    metadata: dict[str, str] | None = None,
) -> RuntimePlan:
    return runtime_plan_from_generator_manifests(
        dict(getattr(generation_result, "component_manifest_paths", {})),
        bundle_dir=bundle_dir,
        model_profile=model_profile,
        component_name_map=component_name_map,
        component_metadata=component_metadata,
        metadata=metadata,
    )


def runtime_routes_from_model_profile(model_profile: Any | None) -> tuple[RuntimeRoute, ...]:
    if model_profile is None:
        return ()

    routes = getattr(model_profile, "inference_type", {}) or {}
    return tuple(runtime_route_from_profile_route(route) for route in routes.values())


def runtime_family_from_model_profile(model_profile: Any | None) -> str:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "")
    normalized_name = profile_name.lower()

    if "gemma4" in normalized_name or "gemma_4" in normalized_name:
        return "gemma4"

    if normalized_name in {"lfm_vlm", "lfm2_vl", "lfm-vlm"}:
        return "lfm2_vl"

    return profile_name


def runtime_route_from_profile_route(route: Any) -> RuntimeRoute:
    return RuntimeRoute(
        name=str(getattr(route, "name", "")),
        edges=tuple(
            RuntimeRouteEdge(
                inputs=tuple(str(value) for value in getattr(edge, "input", ())),
                output=str(getattr(edge, "output", "")),
            )
            for edge in getattr(route, "route", ())
        ),
    )


def runtime_plan_to_engine_manifest(plan: RuntimePlan) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "components": [component.to_engine_dict() for component in plan.components],
    }

    if plan.family:
        manifest["family"] = plan.family

    for key, value in string_dict(plan.metadata).items():
        manifest[key] = value

    return manifest


def runtime_plan_to_dict(plan: RuntimePlan) -> dict[str, Any]:
    return {
        "family": plan.family,
        "components": [component.to_plan_dict() for component in plan.components],
        "routes": [runtime_route_to_dict(route) for route in plan.routes],
        "metadata": string_dict(plan.metadata),
    }


def runtime_component_to_engine_dict(component: RuntimeComponent) -> dict[str, Any]:
    return {
        "component": component.component,
        "graph": component.graph,
        "runtime_input_node_ids": list(component.runtime_input_node_ids),
        "logical_inputs": list(component.logical_inputs),
        "output_node_ids": list(component.output_node_ids),
        "logical_outputs": list(component.logical_outputs),
        "metadata": string_dict(component.metadata),
        "bound_constant_bindings": [
            constant_binding_to_dict(binding)
            for binding in component.bound_constant_bindings
        ],
        "cache_state_node_ids": [
            cache_state_binding_to_dict(binding)
            for binding in component.cache_state_node_ids
        ],
    }


def runtime_component_to_plan_dict(component: RuntimeComponent) -> dict[str, Any]:
    data = runtime_component_to_engine_dict(component)
    data["unsupported_nodes"] = list(component.unsupported_nodes)
    data["warnings"] = list(component.warnings)
    return data


def runtime_route_to_dict(route: RuntimeRoute) -> dict[str, Any]:
    return {
        "name": route.name,
        "edges": [
            {
                "inputs": list(edge.inputs),
                "output": edge.output,
            }
            for edge in route.edges
        ],
    }


def write_runtime_plan(plan: RuntimePlan, bundle_dir: str | Path) -> tuple[Path, Path]:
    bundle_path = Path(bundle_dir)
    components_dir = bundle_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    engine_manifest_path = components_dir / "manifest.json"
    runtime_plan_path = bundle_path / "runtime_plan.json"

    write_json(engine_manifest_path, plan.to_engine_manifest())
    write_json(runtime_plan_path, plan.to_plan_dict())
    return engine_manifest_path, runtime_plan_path


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def runtime_graph_path(manifest_path: Path, graph_path: str, bundle_dir: Path | None) -> str:
    if not graph_path or bundle_dir is None:
        return graph_path

    path = Path(graph_path)

    if not path.is_absolute():
        path = manifest_path.parent / path

    return os.path.relpath(path, bundle_dir)


def constant_bindings_from_generator_manifest(data: dict[str, Any]) -> tuple[ConstantBinding, ...]:
    if "bound_constant_bindings" in data:
        return tuple(
            constant_binding_from_dict(value)
            for value in data.get("bound_constant_bindings", ())
            if isinstance(value, dict)
        )

    return tuple(
        ConstantBinding(
            node_id=int(value["node_id"]),
            path=str(value["path"]),
        )
        for value in data.get("weight_bindings", ())
        if isinstance(value, dict) and "node_id" in value and "path" in value
    )


def string_dict(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        return {}

    return {
        str(key): str(value)
        for key, value in data.items()
        if value is not None
    }


def unique_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    unique: list[int] = []

    for value in values:
        if value in unique:
            continue

        unique.append(value)

    return tuple(unique)
