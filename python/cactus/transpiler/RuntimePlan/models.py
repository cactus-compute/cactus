from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class ConstantBinding:
    node_id: int
    path: str
    kind: str = "weight"
    source_name: str | None = None
    value_id: str | None = None
    scale_factor: float = 1.0
    adapter_family: str | None = None
    transform: str = "none"
    qdq_restore: str = "hf_key"


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
class RuntimeState:
    name: str
    kind: str
    producer: str = ""
    consumers: tuple[str, ...] = ()
    lifetime: str = "request"
    transfer: str = "move"
    persist_after_component_unload: bool = True
    required: bool = True
    release_after_consumers: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


@dataclass(slots=True, frozen=True)
class RuntimeAlias:
    source_component: str
    source_output: str
    target_component: str
    target_input: str
    policy: str = "alias_if_compatible"
    lifetime: str = "until_target_execute"
    fallback: str = "copy"
    required: bool = False
    source_node_id: int = -1
    target_node_id: int = -1
    storage_stable: bool = False
    metadata: dict[str, str] | None = None


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
    states: tuple[RuntimeState, ...] = ()
    aliases: tuple[RuntimeAlias, ...] = ()
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
        kind=str(data.get("kind", "weight")),
        source_name=none_or_str(data.get("source_name")),
        value_id=none_or_str(data.get("value_id")),
        scale_factor=float(data.get("scale_factor", 1.0) or 1.0),
        adapter_family=none_or_str(data.get("adapter_family")),
        transform=str(data.get("transform", "none") or "none"),
        qdq_restore=str(data.get("qdq_restore", "hf_key") or "hf_key"),
    )


def constant_binding_to_dict(binding: ConstantBinding) -> dict[str, Any]:
    return {
        "kind": binding.kind,
        "node_id": binding.node_id,
        "path": binding.path,
        "source_name": binding.source_name,
        "value_id": binding.value_id,
        "scale_factor": binding.scale_factor,
        "adapter_family": binding.adapter_family,
        "transform": binding.transform,
        "qdq_restore": binding.qdq_restore,
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


def runtime_component_from_engine_dict(data: dict[str, Any]) -> RuntimeComponent:
    return RuntimeComponent(
        component=str(data.get("component", "")),
        graph=str(data.get("graph", "")),
        runtime_input_node_ids=tuple(int(value) for value in data.get("runtime_input_node_ids", ())),
        logical_inputs=tuple(str(value) for value in data.get("logical_inputs", ())),
        output_node_ids=tuple(int(value) for value in data.get("output_node_ids", ())),
        logical_outputs=tuple(str(value) for value in data.get("logical_outputs", ())),
        bound_constant_bindings=tuple(
            constant_binding_from_dict(value)
            for value in data.get("bound_constant_bindings", ())
            if isinstance(value, dict)
        ),
        cache_state_node_ids=tuple(
            cache_state_binding_from_dict(value)
            for value in data.get("cache_state_node_ids", ())
            if isinstance(value, dict)
        ),
        metadata=string_dict(data.get("metadata")),
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
    plan_metadata = runtime_plan_metadata_from_model_profile(model_profile)
    plan_metadata.update(runtime_plan_metadata_from_components(components))
    plan_metadata.update(string_dict(metadata))

    return RuntimePlan(
        family=runtime_family_from_model_profile(model_profile),
        components=components,
        routes=runtime_routes_from_model_profile(model_profile),
        states=runtime_states_from_model_profile(model_profile),
        aliases=resolve_runtime_aliases(
            runtime_aliases_from_model_profile(model_profile), components
        ),
        metadata=plan_metadata,
    )


def runtime_plan_metadata_from_components(components: tuple[RuntimeComponent, ...]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    component_by_name = {component.component: component for component in components}
    decoder_prefill = component_by_name.get("decoder_prefill_chunk")

    if decoder_prefill is None:
        return metadata

    component_metadata = string_dict(decoder_prefill.metadata)
    chunk_tokens = component_metadata.get("prefill_chunk_tokens")

    if chunk_tokens is None:
        return metadata

    metadata["prefill_strategy"] = "chunked"
    metadata["prefill_chunk_tokens"] = chunk_tokens

    return metadata


def runtime_plan_metadata_from_model_profile(model_profile: Any | None) -> dict[str, str]:
    if model_profile is None:
        return {}

    metadata: dict[str, str] = {}
    prompt = getattr(model_profile, "prompt_contract", None)
    media = getattr(model_profile, "media_contract", None)
    cache = getattr(model_profile, "cache_contract", None)
    runtime = getattr(model_profile, "runtime_contract", None)

    put_csv(metadata, "disabled_fusion_fields", getattr(model_profile, "disabled_fusion_fields", ()))
    put_csv(metadata, "disabled_fusions", getattr(model_profile, "disabled_fusions", ()))

    if prompt is not None:
        put_if_present(metadata, "prompt_style", getattr(prompt, "style", ""))
        put_if_present(metadata, "prompt_template_source", getattr(prompt, "template_source", ""))
        put_if_present(metadata, "prompt_text_style", getattr(prompt, "text_style", ""))
        put_if_present(metadata, "prompt_media_style", getattr(prompt, "media_style", ""))
        put_if_present(metadata, "prompt_turn_start_token", getattr(prompt, "turn_start_token", ""))
        put_if_present(metadata, "prompt_turn_end_token", getattr(prompt, "turn_end_token", ""))
        put_if_present(metadata, "repetition_penalty_scope", getattr(prompt, "repetition_penalty_scope", ""))
        put_csv(metadata, "suppress_generation_token_ids", getattr(prompt, "suppress_generation_token_ids", ()))

    if media is not None:
        put_if_present(metadata, "media_image_preprocess_strategy", getattr(media, "image_preprocess_strategy", ""))
        put_if_present(metadata, "media_audio_preprocess_strategy", getattr(media, "audio_preprocess_strategy", ""))
        put_if_present(metadata, "media_injection_strategy", getattr(media, "injection_strategy", ""))
        put_csv(metadata, "media_order", getattr(media, "media_order", ()))
        put_if_present(metadata, "media_focus_policy", getattr(media, "focus_policy", ""))
        put_csv(metadata, "media_image_focus_keywords", getattr(media, "image_focus_keywords", ()))
        put_csv(metadata, "media_audio_focus_keywords", getattr(media, "audio_focus_keywords", ()))
        put_csv(metadata, "media_chunk_prefill_modalities", getattr(media, "chunk_prefill_modalities", ()))
        put_if_present(metadata, "media_prefill_fallback", getattr(media, "prefill_fallback", ""))
        put_if_present(metadata, "media_min_new_tokens", getattr(media, "min_new_tokens", 0))
        put_pair_csv(metadata, "media_chunk_output_sources", getattr(media, "chunk_output_sources", ()))
        put_if_present(metadata, "media_mask_polarity", getattr(media, "mask_polarity", ""))
        put_if_present(metadata, "media_span_strategy", getattr(media, "span_strategy", ""))
        put_if_present(metadata, "media_audio_rows_per_frames", getattr(media, "audio_rows_per_frames", ""))
        if getattr(media, "injection_strategy", ""):
            metadata["media_placeholder_token_id"] = str(int(getattr(media, "placeholder_token_id", 0) or 0))
        put_if_present(metadata, "media_image_token_id", getattr(media, "image_token_id", 0))
        put_if_present(metadata, "media_audio_token_id", getattr(media, "audio_token_id", 0))
        put_if_present(metadata, "media_image_token", getattr(media, "image_token", ""))
        put_if_present(metadata, "media_audio_token", getattr(media, "audio_token", ""))
        put_if_present(metadata, "media_image_begin_token", getattr(media, "image_begin_token", ""))
        put_if_present(metadata, "media_image_end_token", getattr(media, "image_end_token", ""))
        put_if_present(metadata, "media_audio_begin_token", getattr(media, "audio_begin_token", ""))
        put_if_present(metadata, "media_audio_end_token", getattr(media, "audio_end_token", ""))
        put_if_present(metadata, "media_image_prompt_position", getattr(media, "image_prompt_position", ""))
        put_if_present(metadata, "media_audio_prompt_position", getattr(media, "audio_prompt_position", ""))
        put_csv(metadata, "media_image_feature_names", getattr(media, "image_feature_names", ()))
        put_csv(metadata, "media_audio_feature_names", getattr(media, "audio_feature_names", ()))

    if cache is not None:
        put_if_present(metadata, "cache_prefill_decode_compatibility", getattr(cache, "prefill_decode_compatibility", ""))
        put_if_present(metadata, "cache_state_transfer", getattr(cache, "state_transfer", ""))
        put_if_present(metadata, "cache_decode_uses_media_components", bool(getattr(cache, "decode_uses_media_components", False)))
        put_if_present(metadata, "cache_max_sequence_length", getattr(cache, "max_cache_sequence_length", 0))
        put_csv(metadata, "fp16_kv_cache_components", getattr(cache, "fp16_kv_cache_components", ()))

    if runtime is not None:
        put_if_present(metadata, "runtime_plan_name", getattr(runtime, "plan_name", ""))
        put_if_present(metadata, "runtime_execution_strategy", getattr(runtime, "execution_strategy", ""))
        put_if_present(metadata, "runtime_state_owner", getattr(runtime, "state_owner", ""))
        put_if_present(metadata, "runtime_cache_persistence", getattr(runtime, "cache_persistence", ""))
        put_if_present(metadata, "runtime_output_alias_policy", getattr(runtime, "output_alias_policy", ""))
        put_if_present(metadata, "runtime_cache_transfer_policy", getattr(runtime, "cache_transfer_policy", ""))

    return metadata


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


def runtime_states_from_model_profile(model_profile: Any | None) -> tuple[RuntimeState, ...]:
    runtime = getattr(model_profile, "runtime_contract", None)
    if runtime is None:
        return ()

    return tuple(runtime_state_from_contract(state) for state in getattr(runtime, "states", ()) or ())


def runtime_aliases_from_model_profile(model_profile: Any | None) -> tuple[RuntimeAlias, ...]:
    runtime = getattr(model_profile, "runtime_contract", None)
    if runtime is None:
        return ()

    return tuple(runtime_alias_from_contract(alias) for alias in getattr(runtime, "aliases", ()) or ())


def resolve_runtime_aliases(
    aliases: tuple[RuntimeAlias, ...],
    components: tuple[RuntimeComponent, ...],
) -> tuple[RuntimeAlias, ...]:
    """Resolve profile-level logical aliases to exact generated graph nodes.

    Resolving node IDs proves that the logical edge exists; it does not prove
    that the source output owns durable storage. Zero-copy therefore remains
    opt-in through alias metadata until generation can establish ownership.
    """
    component_by_name = {component.component: component for component in components}
    resolved: list[RuntimeAlias] = []
    for alias in aliases:
        source = component_by_name.get(alias.source_component)
        target = component_by_name.get(alias.target_component)
        source_node_id = logical_node_id(
            source.logical_outputs if source else (),
            source.output_node_ids if source else (),
            alias.source_output,
        )
        target_node_id = logical_node_id(
            target.logical_inputs if target else (),
            target.runtime_input_node_ids if target else (),
            alias.target_input,
        )
        metadata = alias.metadata or {}
        storage_stable_requested = str(metadata.get("storage_stable", "")).lower() in {
            "1", "true", "yes", "on",
        }
        if storage_stable_requested and source_node_id >= 0 and target_node_id >= 0:
            resolved.append(replace(
                alias,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                storage_stable=True,
            ))
        else:
            resolved.append(alias)
    return tuple(resolved)


def logical_node_id(names: tuple[str, ...], node_ids: tuple[int, ...], name: str) -> int:
    try:
        index = names.index(name)
    except ValueError:
        return -1
    return node_ids[index] if index < len(node_ids) else -1


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


def runtime_state_from_contract(state: Any) -> RuntimeState:
    return RuntimeState(
        name=str(getattr(state, "name", "")),
        kind=str(getattr(state, "kind", "")),
        producer=str(getattr(state, "producer", "")),
        consumers=tuple(str(value) for value in getattr(state, "consumers", ()) or ()),
        lifetime=str(getattr(state, "lifetime", "request")),
        transfer=str(getattr(state, "transfer", "move")),
        persist_after_component_unload=bool(getattr(state, "persist_after_component_unload", True)),
        required=bool(getattr(state, "required", True)),
        release_after_consumers=tuple(
            str(value) for value in getattr(state, "release_after_consumers", ()) or ()
        ),
        metadata=tuple_metadata_dict(getattr(state, "metadata", ())),
    )


def runtime_alias_from_contract(alias: Any) -> RuntimeAlias:
    return RuntimeAlias(
        source_component=str(getattr(alias, "source_component", "")),
        source_output=str(getattr(alias, "source_output", "")),
        target_component=str(getattr(alias, "target_component", "")),
        target_input=str(getattr(alias, "target_input", "")),
        policy=str(getattr(alias, "policy", "alias_if_compatible")),
        lifetime=str(getattr(alias, "lifetime", "until_target_execute")),
        fallback=str(getattr(alias, "fallback", "copy")),
        required=bool(getattr(alias, "required", False)),
        metadata=tuple_metadata_dict(getattr(alias, "metadata", ())),
    )


def runtime_plan_to_engine_manifest(plan: RuntimePlan) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "components": [component.to_engine_dict() for component in plan.components],
    }

    if plan.family:
        manifest["family"] = plan.family

    if plan.states:
        manifest["states"] = [runtime_state_to_dict(state) for state in plan.states]

    if plan.aliases:
        manifest["aliases"] = [runtime_alias_to_dict(alias) for alias in plan.aliases]

    for key, value in string_dict(plan.metadata).items():
        manifest[key] = value

    return manifest


def runtime_plan_to_dict(plan: RuntimePlan) -> dict[str, Any]:
    return {
        "family": plan.family,
        "components": [component.to_plan_dict() for component in plan.components],
        "routes": [runtime_route_to_dict(route) for route in plan.routes],
        "states": [runtime_state_to_dict(state) for state in plan.states],
        "aliases": [runtime_alias_to_dict(alias) for alias in plan.aliases],
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


def runtime_state_to_dict(state: RuntimeState) -> dict[str, Any]:
    return {
        "name": state.name,
        "kind": state.kind,
        "producer": state.producer,
        "consumers": list(state.consumers),
        "release_after_consumers": list(state.release_after_consumers),
        "lifetime": state.lifetime,
        "transfer": state.transfer,
        "persist_after_component_unload": state.persist_after_component_unload,
        "required": state.required,
        "metadata": string_dict(state.metadata),
    }


def runtime_alias_to_dict(alias: RuntimeAlias) -> dict[str, Any]:
    result = {
        "source_component": alias.source_component,
        "source_output": alias.source_output,
        "target_component": alias.target_component,
        "target_input": alias.target_input,
        "policy": alias.policy,
        "lifetime": alias.lifetime,
        "fallback": alias.fallback,
        "required": alias.required,
        "metadata": string_dict(alias.metadata),
    }
    if alias.source_node_id >= 0:
        result["source_node_id"] = alias.source_node_id
    if alias.target_node_id >= 0:
        result["target_node_id"] = alias.target_node_id
    if alias.storage_stable:
        result["storage_stable"] = True
    return result


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
    if path.is_symlink():
        path.unlink()
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
            kind=str(value.get("binding_kind", "weight")),
            source_name=none_or_str(value.get("source_name") or value.get("source_target")),
            value_id=none_or_str(value.get("value_id") or value.get("placeholder")),
            scale_factor=float(value.get("scale_factor", 1.0) or 1.0),
            adapter_family=none_or_str(value.get("adapter_family")),
            transform=str(value.get("transform", "none") or "none"),
            qdq_restore=str(value.get("qdq_restore", "hf_key") or "hf_key"),
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


def put_if_present(metadata: dict[str, str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        metadata[key] = str(value).lower()
        return

    if value is None or value == "" or value == 0:
        return

    metadata[key] = str(value)


def put_csv(metadata: dict[str, str], key: str, values: Any) -> None:
    if not values:
        return

    metadata[key] = ",".join(str(value) for value in values)


def put_pair_csv(metadata: dict[str, str], key: str, values: Any) -> None:
    if not values:
        return

    pairs: list[str] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        left, right = item
        pairs.append(f"{left}:{right}")

    if pairs:
        metadata[key] = ",".join(pairs)


def tuple_metadata_dict(values: Any) -> dict[str, str]:
    if isinstance(values, dict):
        return string_dict(values)

    metadata: dict[str, str] = {}
    for item in values or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        key, value = item
        if value is None:
            continue

        metadata[str(key)] = str(value)

    return metadata


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def unique_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    unique: list[int] = []

    for value in values:
        if value in unique:
            continue

        unique.append(value)

    return tuple(unique)
