from __future__ import annotations

from collections.abc import Mapping

from .component_split_types import Gemma4Boundaries, LfmVlmBoundaries, OutputSpec, WhisperBoundaries
from .component_split_utils import tensor_last_dim, tensor_rank, tensor_shape
from ..Fusions import models as FModels
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
            pairs.append((
                direct_whisper_cross_kv_source(key_node).name,
                direct_whisper_cross_kv_source(value_node).name,
            ))

    if not pairs:
        raise ValueError("Whisper component split could not find decoder cross-attention KV outputs")

    return tuple(pairs)

def direct_whisper_cross_kv_source(node: IRModels.Node) -> IRModels.Node:
    if node.target != "cactus.cat" or len(node.parents) != 2:
        return node

    left, right = node.parents

    if tensor_shape(left) != [0] or right.target != "cactus.transpose" or not right.parents:
        return node

    source = right.parents[0]

    if tensor_shape(source) == [1, 1500, 6, 64]:
        return source

    return node

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

def find_lfm_vlm_boundaries(prefill: IRModels.Graph, decode: IRModels.Graph) -> LfmVlmBoundaries:
    merge = find_lfm_vlm_masked_scatter_merge(prefill)
    return LfmVlmBoundaries(
        vision_position_embeddings=find_lfm_vlm_vision_position_embeddings(prefill),
        vision_encoder_features=find_lfm_vlm_vision_encoder_features(prefill),
        vision_projector_input=find_lfm_vlm_vision_projector_input(prefill),
        image_features=find_lfm_vlm_image_features(merge),
        prefill_text_inputs_embeds=find_lfm_vlm_text_inputs_embeds(prefill),
        merged_inputs_embeds=merge.name,
        decode_text_inputs_embeds=find_lfm_vlm_text_inputs_embeds(decode),
        decode_position_ids=find_lfm_vlm_decode_position_ids(decode),
    )

def find_lfm_vlm_masked_scatter_merge(graph: IRModels.Graph) -> IRModels.Node:
    candidates = [
        node
        for node in graph.nodes
        if node.target == "aten.masked_scatter.default"
        and tensor_rank(node) == 3
        and tensor_last_dim(node) == 2048
        and len(node.parents) >= 3
    ]

    if not candidates:
        raise ValueError("LFM-VL component split could not find the image/text masked_scatter merge")

    return min(candidates, key=lambda node: node.index)

def find_lfm_vlm_image_features(merge: IRModels.Node) -> str:
    if len(merge.parents) < 3:
        raise ValueError(f"{merge.name}: LFM-VL merge does not have an image feature parent")

    image_features = merge.parents[2]

    if image_features.target in {"aten.cat.default", "cactus.cat"} and len(image_features.parents) == 1:
        return image_features.parents[0].name

    return image_features.name

def find_lfm_vlm_text_inputs_embeds(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if node.target not in {"aten.embedding.default", "cactus.embedding_from_tensor"}:
            continue

        shape = tensor_shape(node)
        if tensor_rank(node) == 3 and tensor_last_dim(node) == 2048 and shape[0] == 1:
            return node.name

    raise ValueError("LFM-VL component split could not find text inputs_embeds")

def find_lfm_vlm_decode_position_ids(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if node.name == "cache_position":
            return node.name

    for node in graph.nodes:
        if node.target in {"aten.unsqueeze.default", "cactus.unsqueeze"} and tensor_shape(node) == [1, 1]:
            return node.name

    raise ValueError("LFM-VL component split could not find decode position_ids")

def find_lfm_vlm_vision_position_embeddings(graph: IRModels.Graph) -> str:
    for node in graph.nodes:
        if tensor_shape(node) != [1, 1024, 1152]:
            continue

        if not any(child.target in {"aten.add.Tensor", "cactus.add"} for child in node.children):
            continue

        text = f"{node.name} {node.target} {node.module_stack!r}".lower()
        if "vision" in text and "embedding" in text and has_position_embedding_ancestor(node):
            return node.name

    raise ValueError("LFM-VL component split could not find vision positional embeddings")

def has_position_embedding_ancestor(node: IRModels.Node, max_depth: int = 24) -> bool:
    stack: list[tuple[IRModels.Node, int]] = [(node, 0)]
    seen: set[str] = set()

    while stack:
        current, depth = stack.pop()
        if current.name in seen or depth > max_depth:
            continue

        seen.add(current.name)
        text = f"{current.name} {current.target} {current.module_stack!r}".lower()

        if "position_embedding" in text or "position_embeddings" in text:
            return True

        stack.extend((parent, depth + 1) for parent in current.parents)

    return False

def find_lfm_vlm_vision_encoder_features(graph: IRModels.Graph) -> str:
    candidates = [
        node
        for node in graph.nodes
        if tensor_shape(node) == [1, 1024, 1152]
        and any("multi_modal_projector" in f"{child.module_stack!r}" for child in node.children)
    ]

    if candidates:
        return max(candidates, key=lambda node: node.index).name

    candidates = [
        node
        for node in graph.nodes
        if tensor_shape(node) == [1, 1024, 1152]
        and "vision_tower" in f"{node.module_stack!r}"
    ]

    if not candidates:
        raise ValueError("LFM-VL component split could not find vision encoder features")

    return max(candidates, key=lambda node: node.index).name

def find_lfm_vlm_vision_projector_input(graph: IRModels.Graph) -> str:
    candidates = [
        node
        for node in graph.nodes
        if tensor_shape(node) == [1, 8, 8, 4608]
        and "multi_modal_projector" in f"{node.module_stack!r}"
        and any(child.target in {"aten.native_layer_norm.default", "aten.layer_norm.default", "cactus.layernorm"} for child in node.children)
    ]

    if not candidates:
        raise ValueError("LFM-VL component split could not find vision projector input")

    return min(candidates, key=lambda node: node.index).name

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
    projected_candidates = [
        node
        for node in graph.nodes
        if node.target in {"aten.mul.Tensor", "aten.mul.Scalar", "cactus.multiply", "cactus.scalar_multiply"}
        and tensor_shape(node)[-2:] == [35, 256]
        and len(node.children) >= 10
    ]

    if projected_candidates:
        return max(projected_candidates, key=lambda node: len(node.children)).name

    raw_candidates = [
        node
        for node in graph.nodes
        if node.target in {"aten.view.default", "aten.reshape.default", "cactus.view", "cactus.reshape"}
        and tensor_shape(node)[-2:] == [35, 256]
        and node.parents
        and "embed_tokens_per_layer" in f"{node.parents[0].module_stack!r}"
    ]

    if raw_candidates:
        return min(raw_candidates, key=lambda node: node.index).name

    raise ValueError("Gemma4 component split could not find per_layer_inputs")

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

def graph_output_specs(graph: IRModels.Graph, publish_only_logits: bool = False) -> tuple[OutputSpec, ...]:
    refs = graph_output_refs(graph)

    if not refs:
        return (OutputSpec(find_logits_output(graph), "logits"),)

    logits = refs[0]

    return tuple(
        OutputSpec(ref, "logits" if ref == logits else ref, publish=not publish_only_logits or ref == logits)
        for ref in refs
    )

def graph_output_refs(graph: IRModels.Graph) -> tuple[str, ...]:
    refs: list[str] = []

    for output in graph.outputs:
        refs.extend(IRModels.extract_node_refs((output.args, output.kwargs)))

    return tuple(dict.fromkeys(refs))
