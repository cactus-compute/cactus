from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..Fusions import models as FModels
from ..IR import models as IRModels


PREFILL_WITH_CACHE_TASK = "prefill_with_cache"
DECODE_WITH_CACHE_TASK = "decode_with_cache"


@dataclass(slots=True, frozen=True)
class PlaceholderSpec:
    name: str
    logical_name: str
    source_node: str | None = None
    tensor_node: str | None = None
    target: str | None = None
    value_kind: str = FModels.ValueKind.USER_INPUT


@dataclass(slots=True, frozen=True)
class OutputSpec:
    node: str
    logical_name: str


@dataclass(slots=True, frozen=True)
class ComponentSplitSpec:
    name: str
    graph: IRModels.Graph
    outputs: tuple[OutputSpec, ...]
    placeholders: tuple[PlaceholderSpec, ...] = ()
    ref_aliases: Mapping[str, str] = field(default_factory=dict)
    input_aliases: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Gemma4Boundaries:
    vision_features: str
    audio_features: str
    merged_inputs_embeds: str
    prefill_text_inputs_embeds: str
    prefill_token_ids: str
    prefill_per_layer_inputs: str
    decode_text_inputs_embeds: str
    decode_token_ids: str
    decode_per_layer_inputs: str
    decode_position_ids: str
    prefill_logits: str
    decode_logits: str


def split_component_graphs(
    graphs: Mapping[str, IRModels.Graph],
    model_profile: Any | None = None,
) -> dict[str, IRModels.Graph] | None:
    if not is_gemma4_profile(model_profile):
        return None

    prefill = find_graph_by_task(graphs, PREFILL_WITH_CACHE_TASK)
    decode = find_graph_by_task(graphs, DECODE_WITH_CACHE_TASK)

    if prefill is None or decode is None:
        return None

    return split_gemma4_components(prefill, decode)


def split_gemma4_components(prefill: IRModels.Graph, decode: IRModels.Graph) -> dict[str, IRModels.Graph]:
    boundaries = find_gemma4_boundaries(prefill, decode)
    position_step = PlaceholderSpec("position_ids", "position_ids", source_node=boundaries.decode_position_ids)
    position_chunk = PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids")
    inputs_embeds_step = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.decode_text_inputs_embeds)
    inputs_embeds_chunk = PlaceholderSpec("inputs_embeds", "inputs_embeds", source_node=boundaries.merged_inputs_embeds)
    per_layer_step = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.decode_per_layer_inputs)
    per_layer_chunk = PlaceholderSpec("per_layer_inputs", "per_layer_inputs", source_node=boundaries.prefill_per_layer_inputs)

    specs = (
        ComponentSplitSpec(
            name="vision_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.vision_features, "image_features"),),
            input_aliases={"image_position_ids": "pixel_position_ids"},
        ),
        ComponentSplitSpec(
            name="audio_encoder",
            graph=prefill,
            outputs=(OutputSpec(boundaries.audio_features, "audio_features"),),
        ),
        ComponentSplitSpec(
            name="lm_encoder_step",
            graph=decode,
            outputs=(
                OutputSpec(boundaries.decode_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(boundaries.decode_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_step.name, "position_ids"),
            ),
            placeholders=(position_step,),
            ref_aliases={boundaries.decode_token_ids: "input_ids"},
        ),
        ComponentSplitSpec(
            name="lm_encoder_text_chunk",
            graph=prefill,
            outputs=(
                OutputSpec(boundaries.prefill_text_inputs_embeds, "inputs_embeds"),
                OutputSpec(boundaries.prefill_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_chunk.name, "position_ids"),
            ),
            placeholders=(position_chunk,),
            ref_aliases={
                boundaries.merged_inputs_embeds: boundaries.prefill_text_inputs_embeds,
                boundaries.prefill_token_ids: "input_ids",
            },
        ),
        ComponentSplitSpec(
            name="lm_encoder_media_step",
            graph=decode,
            outputs=(
                OutputSpec(inputs_embeds_step.name, "inputs_embeds"),
                OutputSpec(boundaries.decode_per_layer_inputs, "per_layer_inputs"),
                OutputSpec(position_step.name, "position_ids"),
            ),
            placeholders=(inputs_embeds_step, position_step),
            ref_aliases={boundaries.decode_token_ids: "input_ids"},
        ),
        ComponentSplitSpec(
            name="decoder_prefill_chunk",
            graph=prefill,
            outputs=(OutputSpec(boundaries.prefill_logits, "logits"),),
            placeholders=(inputs_embeds_chunk, per_layer_chunk, position_chunk),
        ),
        ComponentSplitSpec(
            name="decoder_step",
            graph=decode,
            outputs=(OutputSpec(boundaries.decode_logits, "logits"),),
            placeholders=(inputs_embeds_step, per_layer_step, position_step),
        ),
    )

    return {spec.name: extract_component_graph(spec) for spec in specs}


def is_gemma4_profile(model_profile: Any | None) -> bool:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()
    return "gemma4" in profile_name or "gemma_4" in profile_name


def find_graph_by_task(graphs: Mapping[str, IRModels.Graph], task: str) -> IRModels.Graph | None:
    for graph in graphs.values():
        if graph.task == task:
            return graph

    for name, graph in graphs.items():
        if task in name:
            return graph

    return None


def find_gemma4_boundaries(prefill: IRModels.Graph, decode: IRModels.Graph) -> Gemma4Boundaries:
    vision_merge, audio_merge = find_gemma4_masked_scatter_merges(prefill)

    return Gemma4Boundaries(
        vision_features=feature_parent(vision_merge, "vision"),
        audio_features=feature_parent(audio_merge, "audio"),
        merged_inputs_embeds=audio_merge.name,
        prefill_text_inputs_embeds=find_text_inputs_embeds(prefill),
        prefill_token_ids=find_embedding_token_ids(prefill),
        prefill_per_layer_inputs=find_per_layer_inputs(prefill),
        decode_text_inputs_embeds=find_text_inputs_embeds(decode),
        decode_token_ids=find_embedding_token_ids(decode),
        decode_per_layer_inputs=find_per_layer_inputs(decode),
        decode_position_ids=find_decode_position_ids(decode),
        prefill_logits=find_logits_output(prefill),
        decode_logits=find_logits_output(decode),
    )


def find_gemma4_masked_scatter_merges(graph: IRModels.Graph) -> tuple[IRModels.Node, IRModels.Node]:
    merges = [
        node
        for node in graph.nodes
        if node.target == "aten.masked_scatter.default"
        and tensor_rank(node) == 3
        and tensor_last_dim(node) == 1536
        and len(node.parents) >= 3
    ]

    if len(merges) < 2:
        raise ValueError("Gemma4 component split requires the vision/audio masked_scatter merge nodes")

    return tuple(sorted(merges, key=lambda node: node.index)[:2])


def feature_parent(merge: IRModels.Node, role: str) -> str:
    if len(merge.parents) < 3:
        raise ValueError(f"{merge.name}: Gemma4 {role} merge does not have a feature parent")

    return merge.parents[2].name


def find_text_inputs_embeds(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if node.target not in {"aten.mul.Tensor", "cactus.multiply"} or tensor_rank(node) != 3 or tensor_last_dim(node) != 1536:
            continue

        if any(parent.target in {"aten.embedding.default", "cactus.embedding_from_tensor"} for parent in node.parents):
            return node.name

    raise ValueError("Gemma4 component split could not find text inputs_embeds")


def find_embedding_token_ids(graph: IRModels.Graph) -> str:
    text_inputs_embeds = graph.nodes_map[find_text_inputs_embeds(graph)]

    for parent in text_inputs_embeds.parents:
        if parent.target not in {"aten.embedding.default", "cactus.embedding_from_tensor"}:
            continue

        for embedding_parent in parent.parents:
            if embedding_parent.value_kind != FModels.ValueKind.PARAMETER:
                return embedding_parent.name

    raise ValueError("Gemma4 component split could not find embedding token-id input")


def find_per_layer_inputs(graph: IRModels.Graph) -> str:
    candidates = [
        node
        for node in graph.nodes
        if node.target in {"aten.mul.Tensor", "aten.mul.Scalar", "cactus.multiply", "cactus.scalar_multiply"}
        and tensor_shape(node)[-2:] == [35, 256]
        and len(node.children) >= 10
    ]

    if not candidates:
        raise ValueError("Gemma4 component split could not find per_layer_inputs")

    return max(candidates, key=lambda node: len(node.children)).name


def find_decode_position_ids(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if node.target == "aten.unsqueeze.default" and tensor_shape(node) == [1, 1]:
            return node.name

    for node in graph.nodes:
        if node.name == "cache_position":
            return node.name

    raise ValueError("Gemma4 component split could not find decode position_ids")


def find_logits_output(graph: IRModels.Graph) -> str:
    output = graph.outputs[0] if graph.outputs else None
    refs = IRModels.extract_node_refs((output.args, output.kwargs)) if output is not None else ()

    if refs:
        return refs[0]

    for node in reversed(graph.nodes):
        if tensor_rank(node) == 3 and tensor_last_dim(node) == 262144:
            return node.name

    raise ValueError("Gemma4 component split could not find logits output")


def extract_component_graph(spec: ComponentSplitSpec) -> IRModels.Graph:
    placeholder_specs = {placeholder.name: placeholder for placeholder in spec.placeholders}
    ref_replacements = {
        **dict(spec.ref_aliases),
        **{
            placeholder.source_node: placeholder.name
            for placeholder in spec.placeholders
            if placeholder.source_node is not None
        },
    }
    required_names = collect_required_node_names(spec, placeholder_specs, ref_replacements)
    output_names = tuple(ref_replacements.get(output.node, output.node) for output in spec.outputs)
    output_logical_names = {
        ref_replacements.get(output.node, output.node): output.logical_name
        for output in spec.outputs
    }
    used_names: set[str] = set()
    nodes: list[IRModels.Node] = []

    for placeholder in spec.placeholders:
        if placeholder.name not in required_names and placeholder.name not in output_names:
            continue

        nodes.append(create_placeholder_node(spec, placeholder, output_logical_names, used_names))

    for node in spec.graph.nodes:
        if node.name not in required_names:
            continue

        if node.name in ref_replacements:
            continue

        nodes.append(clone_component_node(spec, node, ref_replacements, output_logical_names, used_names))

    nodes.append(create_output_node(spec.name, output_names, nodes))
    return IRModels.rebuild_graph(tuple(nodes), spec.graph)


def collect_required_node_names(
    spec: ComponentSplitSpec,
    placeholder_specs: Mapping[str, PlaceholderSpec],
    ref_replacements: Mapping[str, str],
) -> frozenset[str]:
    required: set[str] = set()

    for output in spec.outputs:
        collect_node_ancestors(output.node, spec.graph, required, placeholder_specs, ref_replacements)

    for placeholder in spec.placeholders:
        if placeholder.source_node is None:
            required.add(placeholder.name)

    return frozenset(required)


def collect_node_ancestors(
    node_name: str,
    graph: IRModels.Graph,
    required: set[str],
    placeholder_specs: Mapping[str, PlaceholderSpec],
    ref_replacements: Mapping[str, str],
) -> None:
    stack = [node_name]

    while stack:
        current_name = stack.pop()
        effective_name = ref_replacements.get(current_name, current_name)

        if effective_name in required:
            continue

        required.add(effective_name)

        if effective_name in placeholder_specs:
            continue

        node = graph.nodes_map.get(effective_name)

        if node is None:
            raise ValueError(f"Component split references missing node {effective_name}")

        stack.extend(parent.name for parent in node.parents)


def create_placeholder_node(
    spec: ComponentSplitSpec,
    placeholder: PlaceholderSpec,
    output_logical_names: Mapping[str, str],
    used_names: set[str],
) -> IRModels.Node:
    template = source_template_node(spec.graph, placeholder)
    name = unique_component_node_name(placeholder.name, used_names)
    metadata = dict(template.ir_metadata) if template is not None else {}
    metadata["logical_input"] = placeholder.logical_name

    if placeholder.name in output_logical_names:
        metadata["logical_output"] = output_logical_names[placeholder.name]

    return IRModels.Node(
        index=template.index if template is not None else -1,
        name=name,
        node_type="placeholder",
        target=placeholder.target or placeholder.logical_name,
        args=[],
        kwargs={},
        users=(),
        tensor_output_meta=template.tensor_output_meta if template is not None else None,
        module_stack=template.module_stack if template is not None else None,
        value_kind=placeholder.value_kind,
        attrs={},
        ir_metadata=metadata,
        cache=template.cache if template is not None else None,
    )


def source_template_node(graph: IRModels.Graph, placeholder: PlaceholderSpec) -> IRModels.Node | None:
    for name in (placeholder.tensor_node, placeholder.source_node, placeholder.name):
        if name is not None and name in graph.nodes_map:
            return graph.nodes_map[name]

    return None


def clone_component_node(
    spec: ComponentSplitSpec,
    node: IRModels.Node,
    ref_replacements: Mapping[str, str],
    output_logical_names: Mapping[str, str],
    used_names: set[str],
) -> IRModels.Node:
    metadata = dict(node.ir_metadata)

    if node.is_placeholder:
        logical_input = spec.input_aliases.get(node.name) or spec.input_aliases.get(node.target)

        if logical_input is not None:
            metadata["logical_input"] = logical_input

    if node.name in output_logical_names:
        metadata["logical_output"] = output_logical_names[node.name]

    name = unique_component_node_name(node.name, used_names)

    return IRModels.Node(
        index=node.index,
        name=name,
        node_type=node.node_type,
        target=node.target,
        args=IRModels.rewrite_node_refs(node.args, dict(ref_replacements)),
        kwargs=IRModels.rewrite_node_refs(node.kwargs, dict(ref_replacements)),
        users=(),
        tensor_output_meta=node.tensor_output_meta,
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=dict(node.attrs),
        ir_metadata=metadata,
        cache=node.cache,
    )


def create_output_node(
    component_name: str,
    output_names: tuple[str, ...],
    nodes: list[IRModels.Node],
) -> IRModels.Node:
    max_index = max((node.index for node in nodes), default=-1)

    return IRModels.Node(
        index=max_index + 1,
        name=f"{component_name}_output",
        node_type="output",
        target="output",
        args=[[{"node": name} for name in output_names]],
        kwargs={},
        users=(),
        tensor_output_meta=None,
        module_stack=None,
        value_kind=FModels.ValueKind.OUTPUT,
        attrs={},
        ir_metadata={},
    )


def unique_component_node_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    suffix = 1
    candidate = f"{name}_{suffix}"

    while candidate in used_names:
        suffix += 1
        candidate = f"{name}_{suffix}"

    used_names.add(candidate)
    return candidate


def tensor_shape(node: IRModels.Node) -> list[Any]:
    if not isinstance(node.tensor_output_meta, dict):
        return []

    shape = node.tensor_output_meta.get("shape")
    return list(shape) if isinstance(shape, list) else []


def tensor_rank(node: IRModels.Node) -> int:
    return len(tensor_shape(node))


def tensor_last_dim(node: IRModels.Node) -> Any | None:
    shape = tensor_shape(node)
    return shape[-1] if shape else None
