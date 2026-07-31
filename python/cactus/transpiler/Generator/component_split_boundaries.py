from __future__ import annotations

from .component_split_types import Gemma4Boundaries, OutputSpec, WhisperBoundaries
from .component_split_utils import tensor_last_dim, tensor_rank, tensor_shape
from ..IR import models as IRModels


def find_graph_by_task(graphs: Mapping[str, IRModels.Graph], task: str) -> IRModels.Graph | None:
    for graph in graphs.values():
        if graph.task == task:
            return graph

    for name, graph in graphs.items():
        if task in name:
            return graph

    return None


def find_whisper_boundaries(prefill: IRModels.Graph, decode: IRModels.Graph) -> WhisperBoundaries:
    return WhisperBoundaries(
        encoder_hidden_states=find_whisper_encoder_hidden_states(prefill),
        cross_key_values=find_whisper_cross_key_values(prefill),
        decode_logits=find_logits_output(decode),
    )


def find_whisper_encoder_hidden_states(graph: IRModels.Graph) -> str:
    candidates = [
        node
        for node in graph.nodes
        if node.target == "<built-in function getitem>"
        and tensor_shape(node) == [1, 1500, 384]
        and len(node.children) >= 4
    ]

    if not candidates:
        raise ValueError("Whisper component split could not find encoder_hidden_states")

    return max(candidates, key=lambda node: len(node.children)).name


def find_whisper_cross_key_values(graph: IRModels.Graph) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []

    for node in graph.nodes:
        if node.target != "cactus.attention" or len(node.parents) < 3:
            continue

        key_node, value_node = node.parents[1], node.parents[2]

        if (
            key_node.target == "cactus.cat"
            and value_node.target == "cactus.cat"
            and tensor_shape(key_node) == [1, 6, 1500, 64]
            and tensor_shape(value_node) == [1, 6, 1500, 64]
        ):
            pairs.append((key_node.name, value_node.name))

    if not pairs:
        raise ValueError("Whisper component split could not find decoder cross-attention KV outputs")

    return tuple(pairs)


def find_decoder_token_input(graph: IRModels.Graph) -> str:
    for name in ("decoder_input_ids", "input_ids"):
        if name in graph.nodes_map:
            return name

    raise ValueError("Whisper component split could not find decoder token input")


def find_whisper_decode_position_index(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if node.target == "aten.repeat.default" and tensor_shape(node) == [1, 1]:
            if any(child.target == "aten.index.Tensor" for child in node.children):
                return node.name

    raise ValueError("Whisper component split could not find decoder position index")


def whisper_decoder_cross_input_aliases(graph: IRModels.Graph, layer_count: int) -> dict[str, str]:
    aliases: dict[str, str] = {}

    for node in graph.nodes:
        if node.cache is None or node.cache.tensor_index is None:
            continue

        tensor_index = int(node.cache.tensor_index)
        layer_index = tensor_index // 4

        if layer_index >= layer_count:
            continue

        role_index = tensor_index % 4

        if role_index == 2:
            aliases[node.name] = f"cross_k_{layer_index}"
        elif role_index == 3:
            aliases[node.name] = f"cross_v_{layer_index}"

    return aliases


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


def graph_output_specs(graph: IRModels.Graph) -> tuple[OutputSpec, ...]:
    refs = graph_output_refs(graph)

    if not refs:
        return (OutputSpec(find_logits_output(graph), "logits"),)

    logits = refs[0]

    return tuple(
        OutputSpec(ref, "logits" if ref == logits else ref)
        for ref in refs
    )


def graph_output_refs(graph: IRModels.Graph) -> tuple[str, ...]:
    refs: list[str] = []

    for output in graph.outputs:
        refs.extend(IRModels.extract_node_refs((output.args, output.kwargs)))

    return tuple(dict.fromkeys(refs))
