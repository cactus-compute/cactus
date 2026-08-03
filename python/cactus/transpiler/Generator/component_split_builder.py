from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .component_split_types import ComponentSplitSpec, PlaceholderSpec
from .component_split_utils import element_count, tensor_shape
from ..Fusions import models as FModels
from ..IR import models as IRModels


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
        if output.publish and output.row_limit is None and output.permutation is None
    }
    used_names: set[str] = set()
    nodes: list[IRModels.Node] = []

    for placeholder in spec.placeholders:
        if not placeholder.force and placeholder.name not in required_names and placeholder.name not in output_names:
            continue

        nodes.append(create_placeholder_node(spec, placeholder, output_logical_names, used_names))

    for node in spec.graph.nodes:
        if node.name not in required_names:
            continue

        if node.name in ref_replacements:
            continue

        nodes.append(clone_component_node(spec, node, ref_replacements, output_logical_names, used_names))

    final_output_names = []

    for output, output_name in zip(spec.outputs, output_names):
        final_output_name = output_name

        if not output.publish:
            continue

        if output.permutation is not None:
            final_output_name = add_permuted_output_node(nodes, final_output_name, output.logical_name, output.permutation, used_names)

        if output.row_limit is not None:
            final_output_name = add_row_limited_output_node(nodes, output_name, output.logical_name, output.row_limit, used_names)

        output_logical_names[final_output_name] = output.logical_name
        final_output_names.append(final_output_name)

    nodes.append(create_output_node(spec.name, tuple(final_output_names), nodes))
    graph = IRModels.rebuild_graph(tuple(nodes), spec.graph)
    graph.metadata.update({str(key): str(value) for key, value in spec.metadata.items()})

    if spec.chunk_tokens is not None:
        graph = retarget_chunk_graph_sequence_length(graph, spec.chunk_tokens)

    if spec.scalar_tail_layer_start is not None and spec.chunk_tokens is not None:
        graph = scalarize_gemma4_decoder_tail_graph(graph, spec.scalar_tail_layer_start, spec.chunk_tokens)

    return graph


def add_permuted_output_node(
    nodes: list[IRModels.Node],
    source_name: str,
    logical_name: str,
    permutation: tuple[int, ...],
    used_names: set[str],
) -> str:
    source = next((node for node in nodes if node.name == source_name), None)

    if source is None:
        raise ValueError(f"Component split cannot add permuted output for missing node {source_name}")

    source_shape = tensor_shape(source)

    if len(source_shape) != len(permutation):
        raise ValueError(f"{source_name}: output permutation rank {len(permutation)} does not match shape {source_shape}")

    output_shape = [source_shape[index] for index in permutation]
    name = unique_component_node_name(f"{source_name}_permute", used_names)

    nodes.append(
        IRModels.Node(
            index=max((node.index for node in nodes), default=-1) + 1,
            name=name,
            node_type="call_function",
            target="cactus.transpose",
            args=[{"node": source_name}],
            kwargs={"permutation": list(permutation)},
            users=(),
            tensor_output_meta=tensor_meta_with_shape(source.tensor_output_meta, output_shape),
            module_stack=source.module_stack,
            value_kind=FModels.ValueKind.ACTIVATION,
            attrs={"permutation": list(permutation)},
            ir_metadata={"logical_output": logical_name},
            cache=source.cache,
        )
    )

    return name


def add_row_limited_output_node(
    nodes: list[IRModels.Node],
    source_name: str,
    logical_name: str,
    row_limit: int,
    used_names: set[str],
) -> str:
    source = next((node for node in nodes if node.name == source_name), None)

    if source is None:
        raise ValueError(f"Component split cannot add row-limited output for missing node {source_name}")

    source_shape = tensor_shape(source)

    if not source_shape:
        raise ValueError(f"{source_name}: row-limited output requires tensor shape metadata")

    output_shape = list(source_shape)
    output_shape[0] = min(row_limit, int(output_shape[0])) if isinstance(output_shape[0], int) else row_limit
    name = unique_component_node_name(f"{source_name}_first_{row_limit}", used_names)

    source_meta = source.tensor_output_meta if isinstance(source.tensor_output_meta, dict) else {}
    nodes.append(
        IRModels.Node(
            index=max((node.index for node in nodes), default=-1) + 1,
            name=name,
            node_type="call_function",
            target="cactus.slice",
            args=[{"node": source_name}],
            kwargs={},
            users=(),
            tensor_output_meta={**source_meta, "shape": output_shape},
            module_stack=source.module_stack,
            value_kind=FModels.ValueKind.ACTIVATION,
            attrs={"axis": 0, "start": 0, "length": row_limit, "step": 1},
            ir_metadata={"logical_output": logical_name},
            cache=None,
        )
    )

    return name


def retarget_chunk_graph_sequence_length(graph: IRModels.Graph, chunk_tokens: int) -> IRModels.Graph:
    source_lengths = chunk_source_lengths(graph)

    if not source_lengths:
        return graph

    cache_capacity = max(source_lengths)
    nodes = tuple(retarget_chunk_node(node, source_lengths, chunk_tokens, cache_capacity) for node in graph.nodes)
    return IRModels.rebuild_graph(nodes, graph)


def retarget_whisper_decoder_cross_kv_layout(graph: IRModels.Graph) -> IRModels.Graph:
    nodes = tuple(retarget_whisper_decoder_cross_kv_node(node) for node in graph.nodes)
    return IRModels.rebuild_graph(nodes, graph)


def retarget_whisper_decoder_cross_kv_node(node: IRModels.Node) -> IRModels.Node:
    attrs = dict(node.attrs)
    kwargs = dict(node.kwargs) if isinstance(node.kwargs, dict) else node.kwargs
    tensor_output_meta = node.tensor_output_meta
    cache = node.cache

    if node.is_placeholder and is_whisper_cross_kv_input(node):
        shape = tensor_shape(node)

        if len(shape) == 4 and isinstance(shape[1], int) and isinstance(shape[2], int) and shape[1] < shape[2]:
            new_shape = [shape[0], shape[2], shape[1], shape[3]]
            tensor_output_meta = tensor_meta_with_shape(tensor_output_meta, new_shape)

            if cache is not None:
                cache = replace(cache, shape=tuple(new_shape), layout="batch_sequence_heads_head_dim")

    if node.target == "cactus.attention" and attrs.get("input_layout") == "bqhd_bhsd_bhsd":
        attrs.pop("input_layout", None)

        if isinstance(kwargs, dict):
            kwargs.pop("input_layout", None)

    return IRModels.Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=node.args,
        kwargs=kwargs,
        users=(),
        tensor_output_meta=tensor_output_meta,
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=attrs,
        ir_metadata=dict(node.ir_metadata),
        cache=cache,
    )


def is_whisper_cross_kv_input(node: IRModels.Node) -> bool:
    logical_name = str(node.ir_metadata.get("logical_input") or node.target)
    return logical_name.startswith("cross_k_") or logical_name.startswith("cross_v_")


def chunk_source_lengths(graph: IRModels.Graph) -> frozenset[int]:
    lengths: set[int] = set()

    for node in graph.nodes:
        if not node.is_placeholder:
            continue

        logical_name = str(node.ir_metadata.get("logical_input") or node.target)

        if logical_name not in {"input_ids", "attention_mask", "position_ids", "inputs_embeds", "per_layer_inputs"}:
            continue

        shape = tensor_shape(node)

        if len(shape) >= 2 and shape[0] == 1 and isinstance(shape[1], int) and shape[1] > 1:
            lengths.add(shape[1])

    return frozenset(lengths)


def retarget_chunk_node(
    node: IRModels.Node,
    source_lengths: frozenset[int],
    chunk_tokens: int,
    cache_capacity: int,
) -> IRModels.Node:
    metadata = dict(node.ir_metadata)

    if is_empty_cache_cat_node(node):
        metadata.setdefault("prefill_cache_capacity", cache_capacity)

    if node.is_placeholder and metadata.get("logical_input") in {"input_ids", "attention_mask", "position_ids", "inputs_embeds", "per_layer_inputs"}:
        metadata.setdefault("prefill_chunk_tokens", chunk_tokens)

    return IRModels.Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=retarget_sequence_value(node.args, source_lengths, chunk_tokens),
        kwargs=retarget_sequence_value(node.kwargs, source_lengths, chunk_tokens),
        users=node.users,
        tensor_output_meta=retarget_sequence_value(node.tensor_output_meta, source_lengths, chunk_tokens),
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=retarget_attr_values(node.attrs, source_lengths, chunk_tokens),
        ir_metadata=metadata,
        cache=node.cache,
    )


def retarget_sequence_value(value: Any, source_lengths: frozenset[int], chunk_tokens: int) -> Any:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value < 0:
            for source_length in source_lengths:
                kept = source_length + value

                if 0 <= kept <= chunk_tokens:
                    return kept - chunk_tokens

        return chunk_tokens if value in source_lengths else value

    if isinstance(value, list):
        return [retarget_sequence_value(item, source_lengths, chunk_tokens) for item in value]

    if isinstance(value, tuple):
        return tuple(retarget_sequence_value(item, source_lengths, chunk_tokens) for item in value)

    if isinstance(value, dict):
        return {
            key: item if is_dimension_index_attr(key) else retarget_sequence_value(item, source_lengths, chunk_tokens)
            for key, item in value.items()
        }

    return value


def retarget_attr_values(attrs: dict[str, Any], source_lengths: frozenset[int], chunk_tokens: int) -> dict[str, Any]:
    return {
        key: retarget_sequence_value(value, source_lengths, chunk_tokens) if is_sequence_length_attr(key) else value
        for key, value in attrs.items()
    }


def scalarize_gemma4_decoder_tail_graph(graph: IRModels.Graph, layer_start: int, chunk_tokens: int) -> IRModels.Graph:
    tail_names = gemma4_decoder_tail_node_names(graph, layer_start)

    if not tail_names:
        return graph

    replacement_names, slice_nodes = gemma4_decoder_tail_slice_nodes(graph, tail_names, chunk_tokens)

    if not replacement_names and not any(node_needs_scalar_tail_retarget(node, chunk_tokens) for node in graph.nodes if node.name in tail_names):
        return graph

    rewritten_nodes: list[IRModels.Node] = []

    for node in graph.nodes:
        if node.name in tail_names:
            rewritten_nodes.append(clone_scalar_tail_node(node, replacement_names, chunk_tokens))
        else:
            rewritten_nodes.append(clone_component_graph_node(node))

    rewritten_nodes.extend(slice_nodes)
    return IRModels.prune_dead_nodes(IRModels.rebuild_graph(tuple(rewritten_nodes), graph))


def gemma4_decoder_tail_node_names(graph: IRModels.Graph, layer_start: int) -> frozenset[str]:
    tail_names = {
        node.name
        for node in graph.nodes
        if (layer_index := gemma4_language_layer_index(node)) is not None and layer_index >= layer_start
    }

    stack = list(tail_names)

    while stack:
        node = graph.nodes_map[stack.pop()]

        for child in node.children:
            if child.is_output or child.name in tail_names:
                continue

            tail_names.add(child.name)
            stack.append(child.name)

    return frozenset(tail_names)


def gemma4_language_layer_index(node: IRModels.Node) -> int | None:
    text = f"{node.name} {node.target} {node.module_stack!r}"
    marker = "language_model.layers."
    start = text.find(marker)

    if start < 0:
        return None

    index_start = start + len(marker)
    index_end = index_start

    while index_end < len(text) and text[index_end].isdigit():
        index_end += 1

    if index_end == index_start:
        return None

    return int(text[index_start:index_end])


def gemma4_decoder_tail_slice_nodes(
    graph: IRModels.Graph,
    tail_names: frozenset[str],
    chunk_tokens: int,
) -> tuple[dict[str, str], tuple[IRModels.Node, ...]]:
    replacement_names: dict[str, str] = {}
    slice_nodes: list[IRModels.Node] = []
    used_names = set(graph.nodes_map)
    next_index = max((node.index for node in graph.nodes), default=-1) + 1

    for node in graph.nodes:
        if node.name not in tail_names:
            continue

        for parent in node.parents:
            if parent.name in tail_names or parent.name in replacement_names:
                continue

            axis_and_shape = token_stream_slice_axis_and_shape(parent, chunk_tokens)

            if axis_and_shape is None:
                continue

            axis, shape = axis_and_shape
            slice_name = unique_component_node_name(f"{parent.name}_last_token", used_names)
            slice_nodes.append(create_scalar_tail_slice_node(parent, slice_name, next_index, axis, chunk_tokens, shape))
            replacement_names[parent.name] = slice_name
            next_index += 1

    return replacement_names, tuple(slice_nodes)


def token_stream_slice_axis_and_shape(node: IRModels.Node, chunk_tokens: int) -> tuple[int, list[Any]] | None:
    shape = tensor_shape(node)

    if len(shape) >= 2 and shape[0] == 1 and shape[1] == chunk_tokens:
        output_shape = list(shape)
        output_shape[1] = 1
        return 1, output_shape

    if len(shape) >= 1 and shape[0] == chunk_tokens:
        output_shape = list(shape)
        output_shape[0] = 1
        return 0, output_shape

    return None


def create_scalar_tail_slice_node(
    parent: IRModels.Node,
    name: str,
    index: int,
    axis: int,
    chunk_tokens: int,
    shape: list[Any],
) -> IRModels.Node:
    return IRModels.Node(
        index=index,
        name=name,
        node_type="call_function",
        target="cactus.slice",
        args=[{"node": parent.name}],
        kwargs={"axis": axis, "start": chunk_tokens - 1, "end": None, "step": None},
        users=(),
        tensor_output_meta=tensor_meta_with_shape(parent.tensor_output_meta, shape),
        module_stack=parent.module_stack,
        value_kind=FModels.ValueKind.ACTIVATION,
        attrs={"axis": axis, "start": chunk_tokens - 1, "end": None, "step": None},
        ir_metadata={"inserted_by": "gemma4_decoder_prefill_tail_scalarization"},
        cache=None,
    )


def tensor_meta_with_shape(meta: Any, shape: list[Any]) -> Any:
    if not isinstance(meta, dict):
        return {"shape": shape}

    return {**meta, "shape": shape}


def clone_scalar_tail_node(
    node: IRModels.Node,
    replacement_names: Mapping[str, str],
    chunk_tokens: int,
) -> IRModels.Node:
    return IRModels.Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=scalarize_tail_value(IRModels.rewrite_node_refs(node.args, dict(replacement_names)), chunk_tokens),
        kwargs=scalarize_tail_value(IRModels.rewrite_node_refs(node.kwargs, dict(replacement_names)), chunk_tokens),
        users=(),
        tensor_output_meta=scalarize_tail_value(node.tensor_output_meta, chunk_tokens),
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=scalarize_tail_value(dict(node.attrs), chunk_tokens),
        ir_metadata=dict(node.ir_metadata),
        cache=node.cache,
    )


def clone_component_graph_node(node: IRModels.Node) -> IRModels.Node:
    return IRModels.Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=node.args,
        kwargs=node.kwargs,
        users=(),
        tensor_output_meta=node.tensor_output_meta,
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=dict(node.attrs),
        ir_metadata=dict(node.ir_metadata),
        cache=node.cache,
    )


def scalarize_tail_value(value: Any, chunk_tokens: int) -> Any:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return 1 if value == chunk_tokens else value

    if isinstance(value, list):
        return [scalarize_tail_value(item, chunk_tokens) for item in value]

    if isinstance(value, tuple):
        return tuple(scalarize_tail_value(item, chunk_tokens) for item in value)

    if isinstance(value, dict):
        return {key: scalarize_tail_value(item, chunk_tokens) for key, item in value.items()}

    return value


def node_needs_scalar_tail_retarget(node: IRModels.Node, chunk_tokens: int) -> bool:
    return value_contains_int(node.args, chunk_tokens) or value_contains_int(node.kwargs, chunk_tokens) or value_contains_int(node.tensor_output_meta, chunk_tokens)


def value_contains_int(value: Any, expected: int) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, int):
        return value == expected

    if isinstance(value, (list, tuple)):
        return any(value_contains_int(item, expected) for item in value)

    if isinstance(value, dict):
        return any(value_contains_int(item, expected) for item in value.values())

    return False


def is_dimension_index_attr(key: Any) -> bool:
    return str(key) in {
        "axis",
        "dim",
        "dim0",
        "dim1",
        "dims",
        "end_dim",
        "permutation",
        "start_dim",
    }


def is_sequence_length_attr(key: Any) -> bool:
    return str(key) in {
        "end",
        "length",
        "new_shape",
        "pad",
        "shape",
        "size",
        "sizes",
        "split_sizes",
        "start",
    }


def is_empty_cache_cat_node(node: IRModels.Node) -> bool:
    if node.target not in {"cactus.cat", "aten.cat.default"} or len(node.parents) < 2:
        return False

    first_parent = node.parents[0]
    return first_parent.value_kind == FModels.ValueKind.LIFTED_CONSTANT and element_count(tensor_shape(first_parent)) == 0


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
