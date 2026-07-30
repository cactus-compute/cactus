from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..IR import models as IRModels
from ..RuntimePlan import models as RPModels


GraphTensor = Any
CactusGraph = Any
LoweringFn = Callable[["GenerationContext", IRModels.Node], GraphTensor | tuple[GraphTensor, ...] | None]


@dataclass(slots=True, frozen=True)
class TensorSpec:
    name: str
    shape: tuple[Any, ...] = ()
    dtype: str | None = None
    dynamic_dims: tuple[bool, ...] = ()

    @classmethod
    def from_node(cls, node: IRModels.Node, *, name: str | None = None) -> "TensorSpec":
        return tensor_spec_from_node(cls, node, name=name)


@dataclass(slots=True, frozen=True)
class LoweringRule:
    target: str
    lower: LoweringFn


@dataclass(slots=True, frozen=True)
class GeneratorConfig:
    output_dir: Path
    weights_dir: Path | None = None
    weights_manifest_path: Path | None = None
    graph_suffix: str = ".cactus"
    strict: bool = True
    allow_unsupported_ops: bool = False

    def component_path(self, component_name: str) -> Path:
        return generator_component_path(self, component_name)


@dataclass(slots=True)
class ComponentGraph:
    name: str
    ir_graph: IRModels.Graph
    output_path: Path
    manifest_path: Path
    graph: CactusGraph | None = None
    weight_resolver: "WeightResolver | None" = None
    weight_bindings: list["WeightBinding"] = field(default_factory=list)
    cache_state_bindings: list[RPModels.CacheStateBinding] = field(default_factory=list)
    runtime_input_ids: list[int] = field(default_factory=list)
    logical_inputs: list[str] = field(default_factory=list)
    output_node_ids: list[int] = field(default_factory=list)
    logical_outputs: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    unsupported_nodes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_ir(cls, name: str, ir_graph: IRModels.Graph, config: GeneratorConfig) -> "ComponentGraph":
        return component_graph_from_ir(cls, name, ir_graph, config)

    def save(self) -> Path:
        return save_component_graph(self)

    def save_manifest(self) -> Path:
        return save_component_manifest(self)

    def mark_unsupported(self, node: IRModels.Node) -> None:
        mark_component_node_unsupported(self, node)

    def warn(self, message: str) -> None:
        add_component_warning(self, message)

    def add_weight_binding(self, binding: "WeightBinding") -> None:
        add_component_weight_binding(self, binding)

    def add_cache_state_binding(self, binding: RPModels.CacheStateBinding) -> None:
        add_component_cache_state_binding(self, binding)

    def add_runtime_input(self, tensor: GraphTensor, logical_name: str | None = None) -> None:
        add_component_runtime_input(self, tensor, logical_name)

    def add_output(self, tensor: GraphTensor, logical_name: str | None = None) -> None:
        add_component_output(self, tensor, logical_name)


@dataclass(slots=True)
class GenerationContext:
    component: ComponentGraph
    config: GeneratorConfig
    lowerings: dict[str, LoweringRule] = field(default_factory=dict)
    values: dict[str, GraphTensor] = field(default_factory=dict)
    skip_node_names: frozenset[str] = frozenset()
    cache_state_placeholder_names: frozenset[str] = frozenset()
    prefill_cache_cat_annotations: dict[str, IRModels.CacheAnnotation] = field(default_factory=dict)
    appended_cache_pairs: set[tuple[str, str]] = field(default_factory=set)

    @property
    def graph(self) -> CactusGraph:
        return active_cactus_graph(self)

    def bind(self, node: IRModels.Node, value: GraphTensor) -> GraphTensor:
        return bind_context_value(self, node, value)

    def lookup(self, node_name: str) -> GraphTensor | None:
        return lookup_context_value(self, node_name)

    def require(self, node_name: str) -> GraphTensor:
        return require_context_value(self, node_name)

    def inputs_for(self, node: IRModels.Node) -> tuple[GraphTensor, ...]:
        return context_inputs_for_node(self, node)

    def lowering_for(self, node: IRModels.Node) -> LoweringRule | None:
        return context_lowering_for_node(self, node)

    def mark_unsupported(self, node: IRModels.Node) -> None:
        self.component.mark_unsupported(node)


@dataclass(slots=True, frozen=True)
class WeightRecord:
    source_name: str | None = None
    hf_name: str | None = None
    adapter_name: str | None = None
    output_name: str | None = None
    shape: tuple[Any, ...] = ()
    precision: str | None = None
    status: str | None = None
    component: str | None = None
    scale_factor: float = 1.0
    adapter_family: str | None = None
    source_names: tuple[str, ...] = ()
    transform: str = "none"
    qdq_restore: str = "hf_key"

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, Any]) -> "WeightRecord":
        return weight_record_from_manifest_row(cls, row)

    @property
    def aliases(self) -> tuple[str, ...]:
        return weight_record_aliases(self)


@dataclass(slots=True, frozen=True)
class WeightBinding:
    placeholder: str
    source_target: str
    node_id: int
    path: str
    output_name: str
    source_name: str | None = None
    value_id: str | None = None
    precision: str | None = None
    component: str | None = None
    scale_factor: float = 1.0
    adapter_family: str | None = None
    transform: str = "none"
    qdq_restore: str = "hf_key"
    binding_kind: str = "mmap_weight"


@dataclass(slots=True)
class WeightResolver:
    weights_dir: Path
    records: tuple[WeightRecord, ...] = ()
    records_by_name: dict[str, WeightRecord] = field(default_factory=dict)
    placeholder_targets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph: IRModels.Graph, weights_dir: Path, manifest_path: Path | None = None) -> "WeightResolver":
        return weight_resolver_from_graph(cls, graph, weights_dir, manifest_path)

    def resolve(self, placeholder_name: str) -> WeightRecord | None:
        return resolve_weight_record(self, placeholder_name)

    def source_target_for(self, placeholder_name: str) -> str | None:
        return self.placeholder_targets.get(placeholder_name)


@dataclass(slots=True, frozen=True)
class ComponentGraphManifest:
    component: str
    graph_path: str
    weight_bindings: tuple[WeightBinding, ...] = ()
    cache_state_node_ids: tuple[RPModels.CacheStateBinding, ...] = ()
    runtime_input_node_ids: tuple[int, ...] = ()
    logical_inputs: tuple[str, ...] = ()
    output_node_ids: tuple[int, ...] = ()
    logical_outputs: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None
    unsupported_nodes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_component(cls, component: ComponentGraph) -> "ComponentGraphManifest":
        return component_manifest_from_component(cls, component)

    def to_dict(self) -> dict[str, Any]:
        return component_manifest_to_dict(self)


@dataclass(slots=True, frozen=True)
class GenerationResult:
    component_paths: dict[str, Path] = field(default_factory=dict)
    component_manifest_paths: dict[str, Path] = field(default_factory=dict)
    engine_manifest_path: Path | None = None
    runtime_plan_path: Path | None = None
    unsupported_nodes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_components(cls, components: tuple[ComponentGraph, ...] | list[ComponentGraph]) -> "GenerationResult":
        return generation_result_from_components(cls, components)

    @property
    def ok(self) -> bool:
        return generation_result_ok(self)


################################################# Model Utils!!!!!!! #################################################


def tensor_spec_from_node(cls: type[TensorSpec], node: IRModels.Node, *, name: str | None = None) -> TensorSpec:
    meta = node.tensor_output_meta if isinstance(node.tensor_output_meta, dict) else {}
    shape = meta.get("shape", ())

    return cls(
        name=name or node.name,
        shape=tuple(shape or ()),
        dtype=meta.get("dtype"),
    )


def generator_component_path(config: GeneratorConfig, component_name: str) -> Path:
    safe_name = sanitize_component_name(component_name)
    return config.output_dir / f"{safe_name}{config.graph_suffix}"


def generator_component_manifest_path(config: GeneratorConfig, component_name: str) -> Path:
    safe_name = sanitize_component_name(component_name)
    return config.output_dir / f"{safe_name}.graph_manifest.json"


def component_graph_from_ir(cls: type[ComponentGraph], name: str, ir_graph: IRModels.Graph, config: GeneratorConfig) -> ComponentGraph:
    return cls(
        name=name,
        ir_graph=ir_graph,
        output_path=config.component_path(name),
        manifest_path=generator_component_manifest_path(config, name),
    )


def save_component_graph(component: ComponentGraph) -> Path:
    if component.graph is None:
        raise ValueError(f"Component {component.name} has no Cactus graph to save")

    component.output_path.parent.mkdir(parents=True, exist_ok=True)
    component.graph.save(component.output_path)
    component.save_manifest()
    return component.output_path


def save_component_manifest(component: ComponentGraph) -> Path:
    component.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = ComponentGraphManifest.from_component(component)
    component.manifest_path.write_text(json.dumps(manifest.to_dict(), indent=4), encoding="utf-8")
    return component.manifest_path


def mark_component_node_unsupported(component: ComponentGraph, node: IRModels.Node) -> None:
    if node.name not in component.unsupported_nodes:
        component.unsupported_nodes.append(node.name)


def add_component_warning(component: ComponentGraph, message: str) -> None:
    if message not in component.warnings:
        component.warnings.append(message)


def add_component_weight_binding(component: ComponentGraph, binding: WeightBinding) -> None:
    if all(existing.node_id != binding.node_id for existing in component.weight_bindings):
        component.weight_bindings.append(binding)


def add_component_cache_state_binding(component: ComponentGraph, binding: RPModels.CacheStateBinding) -> None:
    RPModels.merge_cache_state_binding(component.cache_state_bindings, binding)


def add_component_runtime_input(component: ComponentGraph, tensor: GraphTensor, logical_name: str | None = None) -> None:
    node_id = tensor_node_id(tensor)

    if node_id is None:
        return

    name = logical_name or f"input_{len(component.runtime_input_ids)}"

    if node_id in component.runtime_input_ids:
        index = component.runtime_input_ids.index(node_id)
        component.logical_inputs[index] = name
        return

    component.runtime_input_ids.append(node_id)
    component.logical_inputs.append(name)


def add_component_output(component: ComponentGraph, tensor: GraphTensor, logical_name: str | None = None) -> None:
    node_id = tensor_node_id(tensor)

    if node_id is None:
        return

    name = logical_name or f"output_{len(component.output_node_ids)}"

    if node_id in component.output_node_ids:
        index = component.output_node_ids.index(node_id)
        component.logical_outputs[index] = name
        return

    component.output_node_ids.append(node_id)
    component.logical_outputs.append(name)


def tensor_node_id(tensor: GraphTensor) -> int | None:
    node_id = getattr(tensor, "id", None)

    if node_id is None:
        return None

    return int(node_id)


def active_cactus_graph(context: GenerationContext) -> CactusGraph:
    if context.component.graph is None:
        raise ValueError(f"Component {context.component.name} has no active Cactus graph")

    return context.component.graph


def bind_context_value(context: GenerationContext, node: IRModels.Node, value: GraphTensor) -> GraphTensor:
    context.values[node.name] = value
    return value


def lookup_context_value(context: GenerationContext, node_name: str) -> GraphTensor | None:
    return context.values.get(node_name)


def require_context_value(context: GenerationContext, node_name: str) -> GraphTensor:
    value = context.lookup(node_name)

    if value is None:
        raise KeyError(f"Node {node_name} has not been lowered in component {context.component.name}")

    return value


def context_inputs_for_node(context: GenerationContext, node: IRModels.Node) -> tuple[GraphTensor, ...]:
    return tuple(context.require(parent.name) for parent in node.parents)


def context_lowering_for_node(context: GenerationContext, node: IRModels.Node) -> LoweringRule | None:
    return context.lowerings.get(node.target)


def generation_result_from_components(cls: type[GenerationResult], components: tuple[ComponentGraph, ...] | list[ComponentGraph]) -> GenerationResult:
    return cls(
        component_paths={component.name: component.output_path for component in components},
        component_manifest_paths={component.name: component.manifest_path for component in components},
        unsupported_nodes=tuple(
            unsupported
            for component in components
            for unsupported in component.unsupported_nodes
        ),
        warnings=tuple(
            warning
            for component in components
            for warning in component.warnings
        ),
    )


def generation_result_ok(result: GenerationResult) -> bool:
    return not result.unsupported_nodes


def sanitize_component_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)
    return safe.strip("_-") or "component"


def weight_record_from_manifest_row(cls: type[WeightRecord], row: Mapping[str, Any]) -> WeightRecord:
    return cls(
        source_name=none_or_str(row.get("source_name")),
        hf_name=none_or_str(row.get("hf_name")),
        adapter_name=none_or_str(row.get("adapter_name")),
        output_name=none_or_str(row.get("output_name") or row.get("output_file")),
        shape=tuple(row.get("shape") or ()),
        precision=none_or_str(row.get("precision")),
        status=none_or_str(row.get("status")),
        component=none_or_str(row.get("component")),
        scale_factor=float(row.get("scale_factor", 1.0) or 1.0),
        adapter_family=none_or_str(row.get("adapter_family")),
        source_names=tuple(str(name) for name in row.get("source_names") or ()),
        transform=str(row.get("transform", "none") or "none"),
        qdq_restore=str(row.get("qdq_restore", "hf_key") or "hf_key"),
    )


def weight_record_aliases(record: WeightRecord) -> tuple[str, ...]:
    aliases = (
        record.source_name,
        record.hf_name,
        record.adapter_name,
        *record.source_names,
    )
    return tuple(alias for alias in unique_strings(aliases) if alias)


def weight_resolver_from_graph(
    cls: type[WeightResolver],
    graph: IRModels.Graph,
    weights_dir: Path,
    manifest_path: Path | None = None,
) -> WeightResolver:
    records = load_weight_records(manifest_path or weights_dir / "weights_manifest.json")
    records_by_name = index_weight_records(records)
    placeholder_targets = input_spec_placeholder_targets(graph)

    return cls(
        weights_dir=weights_dir,
        records=records,
        records_by_name=records_by_name,
        placeholder_targets=placeholder_targets,
    )


def load_weight_records(manifest_path: Path) -> tuple[WeightRecord, ...]:
    if not manifest_path.exists():
        return ()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("weights", ())
    return tuple(WeightRecord.from_manifest_row(row) for row in rows if isinstance(row, Mapping))


def index_weight_records(records: tuple[WeightRecord, ...]) -> dict[str, WeightRecord]:
    index: dict[str, WeightRecord] = {}

    for record in records:
        if not record.output_name:
            continue

        for alias in record.aliases:
            for normalized in weight_name_variants(alias):
                index.setdefault(normalized, record)

    return index


def input_spec_placeholder_targets(graph: IRModels.Graph) -> dict[str, str]:
    targets: dict[str, str] = {}

    for spec in graph.input_specs:
        arg_name = getattr(spec, "arg_name", None)
        target = getattr(spec, "target", None)

        if arg_name and target:
            targets[str(arg_name)] = str(target)

    if targets:
        return targets

    return parse_graph_signature_placeholder_targets(graph.graph_signature)


def parse_graph_signature_placeholder_targets(graph_signature: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    pattern = re.compile(
        r"InputSpec\(kind=<InputKind\.(?:PARAMETER|BUFFER):[^>]+>, "
        r"arg=TensorArgument\(name='([^']+)'\), target='([^']+)'"
    )

    for match in pattern.finditer(graph_signature):
        targets[match.group(1)] = match.group(2)

    return targets


def resolve_weight_record(resolver: WeightResolver, placeholder_name: str) -> WeightRecord | None:
    source_target = resolver.placeholder_targets.get(placeholder_name)

    if source_target is None:
        return None

    for variant in weight_name_variants(source_target):
        record = resolver.records_by_name.get(variant)

        if record is not None:
            return record

    return None


def weight_name_variants(name: str) -> tuple[str, ...]:
    variants = [name]

    current = name
    while current.startswith("model."):
        current = current[len("model."):]
        variants.append(current)

    if name.endswith("lm_head.weight"):
        variants.extend(
            (
                "model.language_model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            )
        )

    if name.startswith("_orig_mod."):
        variants.append(name[len("_orig_mod."):])

    return tuple(unique_strings(variants))


def component_manifest_from_component(cls: type[ComponentGraphManifest], component: ComponentGraph) -> ComponentGraphManifest:
    return cls(
        component=component.name,
        graph_path=component.output_path.name,
        weight_bindings=tuple(component.weight_bindings),
        cache_state_node_ids=tuple(component.cache_state_bindings),
        runtime_input_node_ids=tuple(component.runtime_input_ids),
        logical_inputs=tuple(component.logical_inputs),
        output_node_ids=tuple(component.output_node_ids),
        logical_outputs=tuple(component.logical_outputs),
        metadata=dict(component.metadata),
        unsupported_nodes=tuple(component.unsupported_nodes),
        warnings=tuple(component.warnings),
    )


def component_manifest_to_dict(manifest: ComponentGraphManifest) -> dict[str, Any]:
    return {
        "component": manifest.component,
        "graph": manifest.graph_path,
        "graph_path": manifest.graph_path,
        "weight_bindings": [
            {
                "placeholder": binding.placeholder,
                "source_target": binding.source_target,
                "node_id": binding.node_id,
                "path": binding.path,
                "output_name": binding.output_name,
                "source_name": binding.source_name,
                "value_id": binding.value_id,
                "precision": binding.precision,
                "component": binding.component,
                "scale_factor": binding.scale_factor,
                "adapter_family": binding.adapter_family,
                "transform": binding.transform,
                "qdq_restore": binding.qdq_restore,
                "binding_kind": binding.binding_kind,
            }
            for binding in manifest.weight_bindings
        ],
        "bound_constant_bindings": [
            {
                "kind": "weight",
                "node_id": binding.node_id,
                "path": binding.path,
                "source_name": binding.source_name or binding.source_target,
                "value_id": binding.value_id or binding.placeholder,
                "scale_factor": binding.scale_factor,
                "adapter_family": binding.adapter_family,
                "transform": binding.transform,
                "qdq_restore": binding.qdq_restore,
            }
            for binding in manifest.weight_bindings
        ],
        "cache_state_node_ids": [
            RPModels.cache_state_binding_to_dict(binding)
            for binding in manifest.cache_state_node_ids
        ],
        "runtime_input_node_ids": list(manifest.runtime_input_node_ids),
        "logical_inputs": list(manifest.logical_inputs),
        "output_node_ids": list(manifest.output_node_ids),
        "logical_outputs": list(manifest.logical_outputs),
        "metadata": dict(manifest.metadata or {}),
        "unsupported_nodes": list(manifest.unsupported_nodes),
        "warnings": list(manifest.warnings),
    }


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def unique_strings(values: tuple[str | None, ...] | list[str | None]) -> tuple[str, ...]:
    unique: list[str] = []

    for value in values:
        if value is None or value in unique:
            continue

        unique.append(value)

    return tuple(unique)
