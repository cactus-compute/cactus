from collections.abc import Callable

from . import models
from ..Fusions import models as FModels


SpecialFusionMatcher = Callable[
    [
        models.Node,
        models.Graph,
        FModels.FusionDefinition,
        str | None,
        tuple[str, ...],
        tuple[str, ...],
    ],
    models.FusionResult | None,
]


SPECIAL_MATCHER_KEY = "special_matcher"
GEMMA4_ATTENTION_LAYOUT = "gemma4_attention_layout"
HISTORY_POSITION_OFFSET = (1 << 64) - 1


def has_special_matcher(fusion: FModels.FusionDefinition) -> bool:
    return special_matcher_name(fusion) in SPECIAL_MATCHERS


def match_special_fusion(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    *,
    inference_mode: str | None = None,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
) -> models.FusionResult | None:
    matcher = SPECIAL_MATCHERS.get(special_matcher_name(fusion))

    if matcher is None:
        return None

    return matcher(source, graph, fusion, inference_mode, input_modalities, fusion_fields)


def special_matcher_name(fusion: FModels.FusionDefinition) -> str | None:
    value = fusion.metadata.get(SPECIAL_MATCHER_KEY)

    if value is None:
        value = fusion.graph.metadata.get(SPECIAL_MATCHER_KEY)

    return str(value) if value is not None else None


def match_gemma4_attention_layout(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> models.FusionResult | None:
    if inference_mode not in {"prefill_with_cache", "decode_with_cache"}:
        return None

    if fusion_fields and "gemma4_attention" not in fusion_fields and "attention" not in fusion_fields:
        return None

    if inference_mode == "prefill_with_cache":
        prefill_match = gemma4_prefill_attention_match(graph, source, fusion)
        if prefill_match is not None:
            return prefill_match

        return gemma4_vision_attention_match(graph, source, fusion)

    return gemma4_decode_attention_match(graph, source, fusion)


def gemma4_prefill_attention_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    value_bmm, layout_nodes = gemma4_prefill_value_bmm_from_output(source)

    if value_bmm is None or len(value_bmm.parents) < 2:
        return None

    softmax = first_target_ancestor(value_bmm.parents[0], {"cactus.softmax", "aten.softmax.int", "aten._softmax.default"})

    if softmax is None or len(softmax.parents) != 1:
        return None

    qk_bmm = gemma4_prefill_qk_bmm_from_softmax(softmax, value_bmm)

    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    query = first_rank4_single_parent_ancestor(qk_bmm.parents[0])
    key_cat = prefill_cache_cat_ancestor(qk_bmm.parents[1])
    value_cat = prefill_cache_cat_ancestor(value_bmm.parents[1])

    if query is None or key_cat is None or value_cat is None:
        return None

    if not gemma4_prefill_attention_shapes_match(query, key_cat, value_cat, qk_bmm, value_bmm):
        return None

    external_inputs = (query, key_cat, value_cat)

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=layout_nodes,
        external_inputs=external_inputs,
        attrs={
            "scale": 1.0,
            "is_causal": True,
            "input_layout": "bhqd_bhds_bhsd",
            "q_layout": "bhqd",
            "k_layout": "bhsd",
            "v_layout": "bhsd",
            "output_layout": gemma4_prefill_output_layout(source),
            "position_offset": HISTORY_POSITION_OFFSET,
            "window_size": gemma4_layer_window_size(source),
            "dropped_mask_builder": True,
        },
    )


def gemma4_prefill_value_bmm_from_output(source: models.Node, max_depth: int = 10) -> tuple[models.Node | None, tuple[models.Node, ...]]:
    current = source
    matched_nodes: list[models.Node] = [source]

    for _ in range(max_depth):
        if current.target == "aten.bmm.default":
            return current, tuple(matched_nodes)

        if len(current.parents) != 1:
            return None, ()

        parent = current.parents[0]

        if parent.target == "aten.bmm.default":
            matched_nodes.append(parent)
            return parent, tuple(matched_nodes)

        if parent.target not in {
            "cactus.view",
            "aten.view.default",
            "aten.reshape.default",
            "cactus.transpose",
            "aten.permute.default",
            "aten.transpose.int",
            "cactus.precision_cast",
            "aten._to_copy.default",
            "aten.clone.default",
            "aten.contiguous.default",
        }:
            return None, ()

        matched_nodes.append(parent)
        current = parent

    return None, ()


def gemma4_prefill_qk_bmm_from_softmax(softmax: models.Node, value_bmm: models.Node) -> models.Node | None:
    logits = softmax.parents[0]

    if logits.target in {"cactus.add", "aten.add.Tensor"} and logits.parents:
        return first_bmm_ancestor(logits.parents[0], value_bmm)

    return first_bmm_ancestor(logits, value_bmm)


def prefill_cache_cat_ancestor(node: models.Node, max_depth: int = 24) -> models.Node | None:
    current = node

    for _ in range(max_depth):
        if current.target in {"cactus.cat", "aten.cat.default"} and len(current.parents) >= 2:
            first_parent_shape = models.tensor_shape(current.parents[0])

            if current.parents[0].cache is not None:
                return current

            if element_count(first_parent_shape) == 0:
                return current

        if len(current.parents) != 1:
            return None

        current = current.parents[0]

    return None


def gemma4_prefill_attention_shapes_match(
    query: models.Node,
    key: models.Node,
    value: models.Node,
    qk_bmm: models.Node,
    value_bmm: models.Node,
) -> bool:
    query_shape = models.tensor_shape(query)
    key_shape = models.tensor_shape(key)
    value_shape = models.tensor_shape(value)
    qk_shape = models.tensor_shape(qk_bmm)
    value_shape_out = models.tensor_shape(value_bmm)

    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False

    if len(qk_shape) != 3 or len(value_shape_out) != 3:
        return False

    return (
        query_shape[0] == key_shape[0] == value_shape[0]
        and query_shape[1] == qk_shape[0]
        and qk_shape[0] == value_shape_out[0]
        and query_shape[2] == qk_shape[1]
        and key_shape[2] == qk_shape[2]
        and value_shape[2] == qk_shape[2]
        and query_shape[3] == key_shape[3]
    )


def gemma4_prefill_output_layout(source: models.Node) -> str:
    rank = node_rank(source)

    if rank in {2, 3}:
        return "bthd_flat"

    return "bhqd"


def gemma4_layer_window_size(source: models.Node) -> int:
    layer_index = models.layer_index_from_text(f"{source.name} {source.target} {source.module_stack!r}")

    if layer_index is not None and layer_index % 5 == 4:
        return 0

    return 512


def gemma4_vision_attention_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    value_bmm, output_layout_nodes = gemma4_vision_value_bmm_from_output(source)

    if value_bmm is None or len(value_bmm.parents) < 2:
        return None

    softmax = first_target_ancestor(value_bmm.parents[0], {"cactus.softmax", "aten.softmax.int", "aten._softmax.default"})

    if softmax is None:
        return None

    qk_bmm, mask = gemma4_vision_qk_and_mask_from_softmax(softmax, value_bmm)

    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    query = first_rank4_single_parent_ancestor(qk_bmm.parents[0])
    key = first_rank4_single_parent_ancestor(qk_bmm.parents[1])
    value = first_rank4_single_parent_ancestor(value_bmm.parents[1])

    if query is None or key is None or value is None:
        return None

    if not gemma4_vision_attention_shapes_match(source, query, key, value):
        return None

    external_inputs = (query, key, value) if mask is None else (query, key, value, mask)
    matched_nodes = attention_internal_nodes(source, external_inputs)

    if value_bmm.name not in {node.name for node in matched_nodes}:
        matched_nodes = tuple(sorted((*matched_nodes, *output_layout_nodes, value_bmm), key=lambda node: (node.index, node.name)))

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=matched_nodes,
        external_inputs=external_inputs,
        attrs={
            "scale": 1.0,
            "is_causal": False,
            "input_layout": "bhqd_bhds_bhsd",
            "output_layout": "bthd_flat",
            "window_size": 0,
            "additive_mask": mask is not None,
            "dropped_softmax_nan_guard": True,
        },
    )


def gemma4_vision_value_bmm_from_output(source: models.Node, max_depth: int = 8) -> tuple[models.Node | None, tuple[models.Node, ...]]:
    if source.target not in {"cactus.view", "aten.view.default", "aten.reshape.default"} or node_rank(source) != 3:
        return None, ()

    current = source
    matched_nodes: list[models.Node] = [source]

    for _ in range(max_depth):
        if len(current.parents) != 1:
            return None, ()

        parent = current.parents[0]

        if parent.target == "aten.bmm.default":
            matched_nodes.append(parent)
            return parent, tuple(matched_nodes)

        if parent.target not in {
            "cactus.view",
            "aten.view.default",
            "aten.reshape.default",
            "cactus.transpose",
            "aten.permute.default",
            "aten.transpose.int",
            "cactus.precision_cast",
            "aten._to_copy.default",
        }:
            return None, ()

        matched_nodes.append(parent)
        current = parent

    return None, ()


def gemma4_vision_qk_and_mask_from_softmax(
    softmax: models.Node,
    value_bmm: models.Node,
) -> tuple[models.Node | None, models.Node | None]:
    if not softmax.parents:
        return None, None

    logits = softmax.parents[0]

    if logits.target in {"cactus.add", "aten.add.Tensor"} and len(logits.parents) >= 2:
        qk_parent: models.Node | None = None
        mask: models.Node | None = None

        for parent in logits.parents:
            candidate = first_bmm_ancestor(parent, value_bmm)

            if candidate is not None:
                qk_parent = candidate
            else:
                mask = parent

        return qk_parent, mask

    return first_bmm_ancestor(logits, value_bmm), None


def gemma4_vision_attention_shapes_match(
    source: models.Node,
    query: models.Node,
    key: models.Node,
    value: models.Node,
) -> bool:
    source_shape = models.tensor_shape(source)
    query_shape = models.tensor_shape(query)
    key_shape = models.tensor_shape(key)
    value_shape = models.tensor_shape(value)

    if len(source_shape) != 3 or len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False

    batch, heads, tokens, head_dim = query_shape

    if not all(isinstance(dim, int) for dim in (batch, heads, tokens, head_dim)):
        return False

    return (
        key_shape == (batch, heads, head_dim, tokens)
        and value_shape[0] == batch
        and value_shape[1] == heads
        and value_shape[2] == tokens
        and value_shape[3] == head_dim
        and source_shape[0] == batch
        and source_shape[1] == tokens
        and source_shape[2] == heads * head_dim
    )


def attention_internal_nodes(source: models.Node, external_inputs: tuple[models.Node, ...]) -> tuple[models.Node, ...]:
    external_names = {node.name for node in external_inputs}
    matched: dict[str, models.Node] = {}
    stack = [source]

    while stack:
        node = stack.pop()

        if node.name in external_names or node.name in matched:
            continue

        matched[node.name] = node

        for parent in node.parents:
            if parent.name not in external_names:
                stack.append(parent)

    return tuple(sorted(matched.values(), key=lambda node: (node.index, node.name)))


def first_target_ancestor(node: models.Node, targets: set[str], max_depth: int = 12) -> models.Node | None:
    stack: list[tuple[models.Node, int]] = [(node, 0)]
    seen: set[int] = set()

    while stack:
        current, depth = stack.pop()

        if id(current) in seen or depth > max_depth:
            continue

        seen.add(id(current))

        if current.target in targets:
            return current

        stack.extend((parent, depth + 1) for parent in reversed(current.parents))

    return None


def element_count(shape: tuple[object, ...]) -> int | None:
    count = 1

    for dim in shape:
        if not isinstance(dim, int):
            return None

        count *= dim

    return count


def gemma4_decode_attention_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    flat_match = gemma4_decode_attention_flat_output_match(graph, source, fusion)

    if flat_match is not None:
        return flat_match

    return gemma4_decode_attention_core_output_match(graph, source, fusion)


def gemma4_decode_attention_flat_output_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    if source.target not in {"cactus.view", "aten.view.default", "aten.reshape.default"}:
        return None

    if node_rank(source) != 3:
        return None

    existing_attention, existing_layout_nodes = existing_attention_layout_chain(source)

    if existing_attention is not None:
        attrs = dict(existing_attention.attrs)
        attrs["output_layout"] = "bthd_flat"
        external_inputs = tuple(existing_attention.parents)

        if len(existing_attention.parents) >= 3:
            key_cat = cache_cat_ancestor(existing_attention.parents[1], FModels.CacheTensorRole.KEY)
            value_cat = cache_cat_ancestor(existing_attention.parents[2], FModels.CacheTensorRole.VALUE)

            if key_cat is not None and value_cat is not None:
                query, query_layout = gemma4_attention_query_input(existing_attention.parents[0])
                external_inputs = (query, key_cat, value_cat)
                attrs["input_layout"] = query_layout
                attrs["window_size"] = gemma4_attention_window_size(key_cat)
            else:
                query, query_layout = gemma4_attention_query_input(existing_attention.parents[0])
                key = gemma4_attention_passthrough_input(existing_attention.parents[1])
                value = gemma4_attention_passthrough_input(existing_attention.parents[2])
                external_inputs = (query, key, value, *existing_attention.parents[3:])
                attrs["input_layout"] = query_layout

        return models.FusionResult.from_match(
            fusion=fusion,
            source=source,
            matched_nodes=(*existing_layout_nodes, existing_attention),
            external_inputs=external_inputs,
            attrs=attrs,
        )

    inner_view, layout_nodes = post_attention_layout_chain(source)

    if inner_view is None:
        return None

    result = gemma4_decode_attention_core_output_match(graph, inner_view, fusion)

    if result is None:
        return None

    attrs = dict(result.attrs)
    attrs["output_layout"] = "bthd_flat"
    attrs["window_size"] = gemma4_attention_window_size(result.external_inputs[1])

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=(*layout_nodes, *result.matched_nodes),
        external_inputs=result.external_inputs,
        attrs=attrs,
    )


def existing_attention_layout_chain(source: models.Node, max_depth: int = 8) -> tuple[models.Node | None, tuple[models.Node, ...]]:
    current = source
    matched_nodes: list[models.Node] = [source]
    saw_transpose = False

    for _ in range(max_depth):
        if len(current.parents) != 1:
            return None, ()

        parent = current.parents[0]

        if parent.target in {"cactus.transpose", "aten.permute.default", "aten.transpose.int"}:
            saw_transpose = True
            matched_nodes.append(parent)
            current = parent
            continue

        if parent.target == "aten._to_copy.default":
            matched_nodes.append(parent)
            current = parent
            continue

        if parent.target == "cactus.attention" and saw_transpose:
            return parent, tuple(matched_nodes)

        return None, ()

    return None, ()


def post_attention_layout_chain(source: models.Node, max_depth: int = 8) -> tuple[models.Node | None, tuple[models.Node, ...]]:
    current = source
    matched_nodes: list[models.Node] = [source]
    saw_transpose = False

    for _ in range(max_depth):
        if len(current.parents) != 1:
            return None, ()

        parent = current.parents[0]

        if parent.target in {"cactus.transpose", "aten.permute.default", "aten.transpose.int"}:
            saw_transpose = True
            matched_nodes.append(parent)
            current = parent
            continue

        if parent.target == "aten._to_copy.default":
            matched_nodes.append(parent)
            current = parent
            continue

        if parent.target in {"cactus.view", "aten.view.default", "aten.reshape.default"} and node_rank(parent) == 4:
            if saw_transpose and len(parent.parents) == 1 and parent.parents[0].target == "aten.bmm.default":
                matched_nodes.append(parent)
                return parent, tuple(matched_nodes)

        return None, ()

    return None, ()


def gemma4_decode_attention_core_output_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    if source.target != "cactus.view" or len(source.parents) != 1:
        return None

    value_bmm = source.parents[0]

    if value_bmm.target != "aten.bmm.default" or len(value_bmm.parents) < 2:
        return None

    value_cat = cache_cat_ancestor(value_bmm.parents[1], FModels.CacheTensorRole.VALUE)

    if value_cat is None:
        return None

    qk_bmm = first_bmm_ancestor(value_bmm.parents[0], value_bmm)

    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    key_cat = cache_cat_ancestor(qk_bmm.parents[1], FModels.CacheTensorRole.KEY)
    query = first_rank4_single_parent_ancestor(qk_bmm.parents[0])

    if key_cat is None or query is None:
        return None

    query, query_layout = gemma4_attention_query_input(query)

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=(source,),
        external_inputs=(query, key_cat, value_cat),
        attrs={
            "scale": 1.0,
            "input_layout": query_layout,
            "output_layout": "bhqd",
            "window_size": gemma4_attention_window_size(key_cat),
        },
    )


def gemma4_attention_window_size(key_cat: models.Node) -> int:
    for node in ancestor_nodes(key_cat):
        if node.cache is not None and node.cache.window_size is not None:
            return int(node.cache.window_size)

        layer_index = models.layer_index_from_text(f"{node.name} {node.target} {node.module_stack!r}")

        if layer_index is not None:
            return 0 if layer_index % 5 == 4 else 512

        shape = models.tensor_shape(node)
        sequence_length = shape[2] if len(shape) >= 3 else None

        if sequence_length == 511:
            return 512

    return 0


def ancestor_nodes(node: models.Node, max_depth: int = 16) -> tuple[models.Node, ...]:
    found: list[models.Node] = []
    worklist: list[tuple[models.Node, int]] = [(node, 0)]
    seen: set[str] = set()

    while worklist:
        current, depth = worklist.pop(0)

        if current.name in seen:
            continue

        seen.add(current.name)
        found.append(current)

        if depth >= max_depth:
            continue

        for parent in current.parents:
            worklist.append((parent, depth + 1))

    return tuple(found)


def gemma4_attention_query_input(node: models.Node) -> tuple[models.Node, str]:
    current = node
    layout = "bhqd_bhds_bhsd"

    for _ in range(6):
        if len(current.parents) != 1:
            return current, layout

        if current.target == "cactus.scalar_multiply" and float(current.attrs.get("value", 1.0)) == 1.0:
            current = current.parents[0]
            continue

        if current.target == "aten._to_copy.default" and str(current.attrs.get("dtype")) == "torch.float32":
            current = current.parents[0]
            continue

        if is_bqhd_to_bhqd_transpose(current):
            current = current.parents[0]
            layout = "bqhd_bhds_bhsd"
            continue

        return current, layout

    return current, layout


def gemma4_attention_passthrough_input(node: models.Node) -> models.Node:
    current = node

    for _ in range(4):
        if len(current.parents) != 1:
            return current

        if current.target == "cactus.scalar_multiply" and float(current.attrs.get("value", 1.0)) == 1.0:
            current = current.parents[0]
            continue

        if current.target == "aten._to_copy.default" and str(current.attrs.get("dtype")) == "torch.float32":
            current = current.parents[0]
            continue

        return current

    return current


def is_bqhd_to_bhqd_transpose(node: models.Node) -> bool:
    if node.target not in {"cactus.transpose", "aten.permute.default"} or len(node.parents) != 1:
        return False

    permutation = node.attrs.get("permutation")

    if permutation != [0, 2, 1, 3]:
        return False

    parent_shape = models.tensor_shape(node.parents[0])
    node_shape = models.tensor_shape(node)

    if len(parent_shape) != 4 or len(node_shape) != 4:
        return False

    return parent_shape[0] == node_shape[0] and parent_shape[1] == node_shape[2] and parent_shape[2] == node_shape[1] and parent_shape[3] == node_shape[3]


def cache_cat_ancestor(node: models.Node, role: str, max_depth: int = 18) -> models.Node | None:
    current = node

    for _ in range(max_depth):
        if current.target in {"cactus.cat", "aten.cat.default"} and len(current.parents) >= 2:
            cache_parent = current.parents[0]

            if cache_parent.cache is not None and cache_parent.cache.kind == FModels.CacheKind.KV and cache_parent.cache.role == role:
                return current

        if len(current.parents) != 1:
            return None

        current = current.parents[0]

    return None


def first_bmm_ancestor(node: models.Node, excluded: models.Node) -> models.Node | None:
    stack = [node]
    seen: set[str] = set()

    while stack:
        current = stack.pop()

        if current.name in seen:
            continue

        seen.add(current.name)

        if current is not excluded and current.target == "aten.bmm.default":
            return current

        stack.extend(current.parents)

    return None


def first_rank4_single_parent_ancestor(node: models.Node, max_depth: int = 10) -> models.Node | None:
    current = node

    for _ in range(max_depth):
        if node_rank(current) == 4:
            return current

        if len(current.parents) != 1:
            return None

        current = current.parents[0]

    return None


def node_rank(node: models.Node) -> int:
    if not isinstance(node.tensor_output_meta, dict):
        return 0

    shape = node.tensor_output_meta.get("shape")
    return len(shape) if isinstance(shape, list) else 0


SPECIAL_MATCHERS: dict[str, SpecialFusionMatcher] = {
    GEMMA4_ATTENTION_LAYOUT: match_gemma4_attention_layout,
}
