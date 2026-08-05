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
WHISPER_ATTENTION_LAYOUT = "whisper_attention_layout"
LFM_BMM_MASKED_ATTENTION = "lfm_bmm_masked_attention"
GENERIC_CACHED_ATTENTION = "generic_cached_attention"
GEMMA4_ROPE_TABLE_LOOKUP = "gemma4_rope_table_lookup"
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
    if "gemma" not in graph.model_name.lower():
        return None

    if inference_mode not in {"prefill_with_cache", "decode_with_cache"}:
        return None

    if fusion_fields and "gemma4_attention" not in fusion_fields and "attention" not in fusion_fields:
        return None

    if inference_mode == "prefill_with_cache":
        prefill_match = gemma4_prefill_attention_match(graph, source, fusion)
        if prefill_match is not None:
            return prefill_match

        # Gemma vision matches the native masked-attention contract, including
        # its exported scale and boolean additive mask. Audio uses rank-5 local
        # attention with relative-position scores and must remain decomposed.
        return gemma4_vision_attention_match(graph, source, fusion)

    return gemma4_decode_attention_match(graph, source, fusion)


def match_whisper_attention_layout(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> models.FusionResult | None:
    if "whisper_attention" not in fusion_fields and "whisper" not in graph.model_name.lower():
        return None

    return whisper_attention_match(graph, source, fusion)


def match_lfm_bmm_masked_attention(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> models.FusionResult | None:
    if "lfm" not in graph.model_name.lower():
        return None

    if inference_mode not in {"prefill_with_cache", "decode_with_cache"}:
        return None

    if "vision" in f"{source.name} {source.module_stack!r}".lower():
        vision_match = gemma4_vision_attention_match(graph, source, fusion)
        if vision_match is not None:
            return vision_match

    from . import match, match_utils

    bindings = match_utils.bind_fusion_graph(source, fusion.graph, match.match_nodes)

    if bindings is None or not match.match_fusion_bindings(source, graph, fusion.graph, bindings):
        return None

    query_expand = bindings.get("lfm_attn_query_expand")
    key_expand = bindings.get("lfm_attn_key_expand")
    value_expand = bindings.get("lfm_attn_value_expand")

    if query_expand is None or key_expand is None or value_expand is None:
        return None

    if not query_expand.parents or not key_expand.parents or not value_expand.parents:
        return None

    query = lfm_attention_query_input(query_expand.parents[0])
    key_cat = lfm_attention_cache_cat(key_expand.parents[0], FModels.CacheTensorRole.KEY)
    value_cat = lfm_attention_cache_cat(value_expand.parents[0], FModels.CacheTensorRole.VALUE)

    if key_cat is None or value_cat is None:
        return None

    external_inputs = (query, key_cat, value_cat)

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=lfm_attention_matched_nodes(bindings, external_inputs),
        bindings=dict(bindings),
        external_inputs=external_inputs,
        attrs={
            "scale": None,
            "is_causal": True,
            "position_offset": 0,
            "window_size": lfm_attention_window_size(key_cat),
            "input_layout": "bhqd_bhsd_bhsd",
            "output_layout": "bhqd",
            "dropped_mask_builder": True,
            "dropped_gqa_repeat": True,
        },
    )


def match_generic_cached_attention(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> models.FusionResult | None:
    """Fuse exported causal attention by its typed cache/dataflow contract.

    This deliberately ignores model and module names.  Exporters commonly
    decompose grouped-query attention into repeat/expand/view operations and
    add mask and NaN-guard subgraphs around two BMMs.  The stable structure is
    a QK BMM fed by a typed key-cache concatenation, softmax on its path to a
    value BMM fed by the same layer's typed value-cache concatenation.
    """
    if inference_mode != "decode_with_cache":
        return None

    # Anchor on the first rank-4 layout restoration after the value BMM.  This
    # avoids absorbing projection/output-layout operations that follow it.
    if source.target not in {"cactus.view", "aten.view.default", "aten.reshape.default"}:
        return None
    if len(source.parents) != 1 or source.parents[0].target != "aten.bmm.default":
        return None

    value_bmm = source.parents[0]
    if len(value_bmm.parents) < 2 or not has_softmax_ancestor(value_bmm.parents[0]):
        return None

    qk_bmm = first_bmm_ancestor(value_bmm.parents[0], value_bmm)
    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    key_match: models.CacheConcatMatch | None = None
    query_path: models.Node | None = None
    for parent_index, parent in enumerate(qk_bmm.parents[:2]):
        candidate = models.find_cache_concat_ancestor(parent, FModels.CacheTensorRole.KEY, max_depth=40)
        if candidate is None:
            continue
        other_index = 1 - parent_index
        key_match = candidate
        query_path = qk_bmm.parents[other_index]
        break

    value_match = models.find_cache_concat_ancestor(
        value_bmm.parents[1], FModels.CacheTensorRole.VALUE, max_depth=40,
    )
    if key_match is None or value_match is None or query_path is None:
        return None

    key_cache = key_match.state.cache
    value_cache = value_match.state.cache
    if key_cache is None or value_cache is None:
        return None
    if key_cache.layer_index != value_cache.layer_index:
        return None
    supported_layouts = {None, "batch_heads_sequence_head_dim"}
    if key_cache.layout not in supported_layouts or value_cache.layout not in supported_layouts:
        return None

    query = whisper_bqhd_attention_input(query_path, max_depth=24)
    if query is None or not generic_cached_attention_shapes_match(
        query, key_match.new_value, value_match.new_value,
    ):
        return None

    external_inputs = (
        query,
        key_match.new_value,
        value_match.new_value,
        key_match.state,
        value_match.state,
    )
    boundaries = (*external_inputs, key_match.concat, value_match.concat)
    matched_nodes = attention_internal_nodes(source, boundaries)
    matched_names = {node.name for node in matched_nodes}
    if qk_bmm.name not in matched_names or value_bmm.name not in matched_names:
        return None

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=matched_nodes,
        external_inputs=external_inputs,
        attrs={
            "scale": decomposed_attention_scale(qk_bmm),
            "is_causal": True,
            "position_offset": HISTORY_POSITION_OFFSET,
            "window_size": int(key_cache.window_size or 0),
            "input_layout": "bhqd_bhsd_bhsd",
            "output_layout": "bhqd",
            "v_head_dim": last_int_dim(value_match.new_value),
            "dropped_mask_builder": True,
            "dropped_gqa_repeat": True,
            "cache_contract": "typed_kv_concat",
        },
    )


def has_softmax_ancestor(node: models.Node, max_depth: int = 32) -> bool:
    return first_target_ancestor(
        node,
        {"cactus.softmax", "aten.softmax.int", "aten._softmax.default"},
        max_depth=max_depth,
    ) is not None


def generic_cached_attention_shapes_match(
    query: models.Node,
    key_new: models.Node,
    value_new: models.Node,
) -> bool:
    query_shape = models.tensor_shape(query)
    key_shape = models.tensor_shape(key_new)
    value_shape = models.tensor_shape(value_new)
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return False
    if key_shape[:3] != value_shape[:3]:
        return False
    if query_shape[0] != key_shape[0] or query_shape[-1] != key_shape[-1]:
        return False
    if query_shape[2] != key_shape[2]:
        return False
    return all(isinstance(dim, int) and dim > 0 for dim in (*query_shape, *key_shape))


def lfm_attention_matched_nodes(
    bindings: dict[str, models.Node],
    external_inputs: tuple[models.Node, ...],
) -> tuple[models.Node, ...]:
    external_names = {node.name for node in external_inputs}
    matched: dict[str, models.Node] = {
        node.name: node
        for binding_name, node in bindings.items()
        if node.is_operation and node.name not in external_names
    }

    for node_name in ("lfm_attn_query_expand", "lfm_attn_key_expand", "lfm_attn_value_expand"):
        node = bindings.get(node_name)

        if node is None or not node.parents:
            continue

        current = node.parents[0]

        for _ in range(32):
            if current.name in external_names:
                break

            if current.is_operation:
                matched[current.name] = current

            if len(current.parents) != 1:
                break

            current = current.parents[0]

    return tuple(sorted(matched.values(), key=lambda node: (node.index, node.name)))


def match_gemma4_rope_table_lookup(
    source: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> models.FusionResult | None:
    if inference_mode not in {"prefill_with_cache", "decode_with_cache"}:
        return None

    if fusion_fields and "gemma4_attention" not in fusion_fields and "gemma4_rope" not in fusion_fields:
        return None

    table_kind = str(fusion.metadata.get("table_kind") or fusion.graph.metadata.get("table_kind") or "")
    expected_targets = {"cos": "aten.cos.default", "sin": "aten.sin.default"}

    if source.target != expected_targets.get(table_kind):
        return None

    if not is_gemma4_language_rotary_node(source):
        return None

    shape = models.tensor_shape(source)

    if len(shape) != 3 or shape[-1] not in {256, 512}:
        return None

    position_node = gemma4_rope_position_node(source)

    if position_node is None:
        return None

    rotary_dim = int(shape[-1])
    table_name = "full_attention" if rotary_dim == 512 else "sliding_attention"

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=(source,),
        external_inputs=(position_node,),
        attrs={
            "table_kind": table_kind,
            "table_name": table_name,
            "rotary_dim": rotary_dim,
            "max_position_embeddings": 131072,
            "position_source": position_node.name,
        },
    )


def is_gemma4_language_rotary_node(node: models.Node) -> bool:
    text = f"{node.name} {node.target} {node.module_stack!r}".lower()
    return "gemma4" in text and "language_model" in text and "rotary_emb" in text


def gemma4_rope_position_node(source: models.Node) -> models.Node | None:
    if not source.parents:
        return None

    matmul = first_target_ancestor(
        source.parents[0],
        {"aten.bmm.default", "aten.matmul.default", "aten.mm.default", "cactus.matmul"},
        max_depth=12,
    )

    if matmul is None or len(matmul.parents) < 2:
        return None

    for parent in matmul.parents:
        if has_inv_freq_ancestor(parent):
            continue

        candidate = rank2_position_ancestor(parent)

        if candidate is not None:
            return candidate

    return None


def has_inv_freq_ancestor(node: models.Node, max_depth: int = 12) -> bool:
    for candidate in ancestor_nodes(node, max_depth=max_depth):
        text = f"{candidate.name} {candidate.target}".lower()

        if "inv_freq" in text:
            return True

    return False


def rank2_position_ancestor(node: models.Node, max_depth: int = 16) -> models.Node | None:
    fallback: models.Node | None = None

    for candidate in ancestor_nodes(node, max_depth=max_depth):
        shape = models.tensor_shape(candidate)

        if len(shape) == 2 and shape[0] == 1:
            return candidate

        if fallback is None and (
            candidate.target.startswith("aten.arange")
            or candidate.value_kind in {FModels.ValueKind.USER_INPUT, FModels.ValueKind.CACHE_INPUT}
        ):
            fallback = candidate

    return fallback


def whisper_attention_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    is_decoder_self_attention = is_whisper_decoder_self_attention(source)

    if is_decoder_self_attention:
        if graph.task == "decode_with_cache":
            if fusion.target != "cactus.attention_cached":
                return None
            return whisper_self_attention_cached_match(graph, source, fusion)

        if graph.task != "prefill_with_cache" or fusion.target != "cactus.attention":
            return None

    if fusion.target == "cactus.attention_cached":
        return None

    value_bmm, output_layout_nodes = whisper_value_bmm_from_output(source)

    if value_bmm is None or len(value_bmm.parents) < 2:
        return None

    softmax = first_target_ancestor(value_bmm.parents[0], {"cactus.softmax", "aten.softmax.int", "aten._softmax.default"})

    if softmax is None or not softmax.parents:
        return None

    qk_bmm = first_bmm_ancestor(softmax.parents[0], value_bmm)

    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    external_inputs, layout_attrs = whisper_attention_inputs(qk_bmm, value_bmm)

    if external_inputs is None:
        return None

    matched_nodes = attention_internal_nodes(source, external_inputs)

    if value_bmm.name not in {node.name for node in matched_nodes}:
        matched_nodes = tuple(sorted((*matched_nodes, *output_layout_nodes, value_bmm), key=lambda node: (node.index, node.name)))

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=matched_nodes,
        external_inputs=external_inputs,
        attrs={
            # The Whisper external query/key bindings already include the
            # exported scaling nodes. Passing that factor to the native op as
            # well would apply it twice and corrupt encoder/decoder outputs.
            "scale": 1.0,
            "is_causal": is_whisper_decoder_self_attention(source),
            "output_layout": "bthd_flat",
            "window_size": 0,
            "dropped_softmax_nan_guard": True,
            **layout_attrs,
        },
    )


def decomposed_attention_scale(qk_bmm: models.Node) -> float:
    scale = 1.0

    for parent in qk_bmm.parents[:2]:
        current = parent
        for _ in range(10):
            if current.target in {"cactus.scalar_multiply", "aten.mul.Scalar"}:
                value = safe_float(current.attrs.get("value", current.attrs.get("other")))
                if value is not None:
                    scale *= value

            if len(current.parents) != 1:
                break

            if current.target not in {
                "cactus.view", "aten.view.default", "aten.reshape.default",
                "cactus.expand", "aten.expand.default",
                "cactus.transpose", "aten.permute.default",
                "cactus.precision_cast", "aten._to_copy.default",
                "cactus.scalar_multiply", "aten.mul.Scalar",
            }:
                break
            current = current.parents[0]

    return scale


def whisper_self_attention_cached_match(
    graph: models.Graph,
    source: models.Node,
    fusion: FModels.FusionDefinition,
) -> models.FusionResult | None:
    value_bmm, output_layout_nodes = whisper_value_bmm_from_output(source)

    if value_bmm is None or len(value_bmm.parents) < 2:
        return None

    softmax = first_target_ancestor(value_bmm.parents[0], {"cactus.softmax", "aten.softmax.int", "aten._softmax.default"})

    if softmax is None or not softmax.parents:
        return None

    qk_bmm = first_bmm_ancestor(softmax.parents[0], value_bmm)

    if qk_bmm is None or len(qk_bmm.parents) < 2:
        return None

    query = whisper_bqhd_attention_input(qk_bmm.parents[0])
    key_cat = cache_cat_ancestor(qk_bmm.parents[1], FModels.CacheTensorRole.KEY, max_depth=32)
    value_cat = cache_cat_ancestor(value_bmm.parents[1], FModels.CacheTensorRole.VALUE, max_depth=32)

    if query is None or key_cat is None or value_cat is None:
        return None

    if len(key_cat.parents) < 2 or len(value_cat.parents) < 2:
        return None

    key_cache, key_new = key_cat.parents[0], key_cat.parents[1]
    value_cache, value_new = value_cat.parents[0], value_cat.parents[1]

    if key_cache.cache is None or value_cache.cache is None:
        return None

    if key_cache.cache.layer_index != value_cache.cache.layer_index:
        return None

    external_inputs = (query, key_new, value_new, key_cache, value_cache)
    boundary_nodes = (*external_inputs, key_cat, value_cat)
    matched_nodes = attention_internal_nodes(source, boundary_nodes)

    if value_bmm.name not in {node.name for node in matched_nodes}:
        matched_nodes = tuple(sorted((*matched_nodes, *output_layout_nodes, value_bmm), key=lambda node: (node.index, node.name)))

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=matched_nodes,
        external_inputs=external_inputs,
        attrs={
            "scale": 1.0,
            "is_causal": True,
            "input_layout": "bhqd_bhsd_bhsd",
            "output_layout": "bthd_flat",
            "window_size": 0,
            "v_head_dim": last_int_dim(value_new),
        },
    )


def whisper_value_bmm_from_output(source: models.Node, max_depth: int = 10) -> tuple[models.Node | None, tuple[models.Node, ...]]:
    if source.target not in {
        "cactus.view",
        "aten.view.default",
        "aten.reshape.default",
        "cactus.transpose",
        "aten.permute.default",
        "aten.transpose.int",
    }:
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
            "aten.clone.default",
            "aten.contiguous.default",
        }:
            return None, ()

        matched_nodes.append(parent)
        current = parent

    return None, ()


def whisper_attention_inputs(
    qk_bmm: models.Node,
    value_bmm: models.Node,
) -> tuple[tuple[models.Node, models.Node, models.Node] | None, dict[str, object]]:
    query = whisper_bqhd_attention_input(qk_bmm.parents[0])
    key = whisper_bqhd_attention_input(qk_bmm.parents[1])
    value = whisper_bqhd_attention_input(value_bmm.parents[1])

    if query is None or key is None or value is None:
        return None, {}

    layout = whisper_attention_input_layout(query, key, value)

    if layout == "native":
        return (query, key, value), {}

    if layout is not None:
        return (query, key, value), {"input_layout": layout}

    return None, {}


def whisper_bqhd_attention_input(node: models.Node, max_depth: int = 12) -> models.Node | None:
    return whisper_layout_attention_input(
        node,
        lambda shape: len(shape) == 4,
        max_depth=max_depth,
    )


def whisper_layout_attention_input(
    node: models.Node,
    shape_matches: Callable[[tuple[object, ...]], bool],
    max_depth: int = 12,
) -> models.Node | None:
    current = node
    best: models.Node | None = current if node_rank(current) == 4 and shape_matches(models.tensor_shape(current)) else None

    for _ in range(max_depth):
        if len(current.parents) != 1:
            return best

        if current.target in {
            "cactus.view",
            "aten.view.default",
            "aten.reshape.default",
            "cactus.expand",
            "aten.expand.default",
            "cactus.transpose",
            "aten.permute.default",
            "aten.transpose.int",
            "cactus.precision_cast",
            "aten._to_copy.default",
            "aten.clone.default",
            "aten.contiguous.default",
        }:
            current = current.parents[0]
        elif current.target == "cactus.scalar_multiply" and safe_float(current.attrs.get("value")) == 1.0:
            current = current.parents[0]
        else:
            return best

        if node_rank(current) == 4 and shape_matches(models.tensor_shape(current)):
            best = current

    return best


def whisper_attention_input_layout(
    query: models.Node,
    key: models.Node,
    value: models.Node,
) -> str | None:
    query_shape = models.tensor_shape(query)
    key_shape = models.tensor_shape(key)
    value_shape = models.tensor_shape(value)

    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return None

    if key_shape != value_shape:
        return None

    if query_shape[0] != key_shape[0] or query_shape[3] != key_shape[3]:
        return None

    if query_shape[2] == key_shape[2]:
        return "native"

    if query_shape[2] == key_shape[1]:
        return "bqhd_bhsd_bhsd"

    if query_shape[1] == key_shape[1]:
        return "bhqd_bhsd_bhsd"

    return None


def is_whisper_decoder_self_attention(source: models.Node) -> bool:
    text = f"{source.name} {source.target} {source.module_stack!r}".lower()
    return "decoder" in text and "self_attn" in text


def last_int_dim(node: models.Node) -> int:
    shape = models.tensor_shape(node)

    if shape and isinstance(shape[-1], int):
        return int(shape[-1])

    return 0


def safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    query, query_layout = gemma4_attention_query_input(query)
    external_inputs = (query, key_cat, value_cat)

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=layout_nodes,
        external_inputs=external_inputs,
        attrs={
            "scale": 1.0,
            "is_causal": True,
            "input_layout": query_layout,
            "q_layout": "bthd" if query_layout == "bqhd_bhds_bhsd" else "bhqd",
            "k_layout": "bhsd",
            "v_layout": "bhsd",
            "output_layout": gemma4_prefill_output_layout(source),
            "position_offset": HISTORY_POSITION_OFFSET,
            "window_size": gemma4_layer_window_size(source),
            "cache_window_size": gemma4_cache_window_size(source),
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


def gemma4_cache_window_size(source: models.Node) -> int:
    for node in ancestor_nodes(source):
        if node.cache is not None and node.cache.window_size is not None:
            return int(node.cache.window_size)

        layer_index = node.cache.layer_index if node.cache is not None else None

        if layer_index is None:
            layer_index = models.layer_index_from_text(f"{node.name} {node.target} {node.module_stack!r}")

        if layer_index is not None:
            if gemma4_full_retention_cache_layer(layer_index):
                return 0

            return 0 if layer_index % 5 == 4 else 512

        shape = models.tensor_shape(node)
        sequence_length = shape[2] if len(shape) >= 3 else None

        if sequence_length is not None and sequence_length > 511:
            return 0

    return 0


def gemma4_full_retention_cache_layer(layer_index: int) -> bool:
    return layer_index == 13 or layer_index % 5 == 4


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

    # Exported vision attention flattens B/H for bmm, but the native attention
    # op consumes [B, T, H, D]. Walk back through only layout/cast operations
    # with verified shapes so lowering does not recreate three large
    # transposes for every vision layer.
    native_query = gemma4_vision_native_attention_input(query)
    native_key = gemma4_vision_native_attention_input(key)
    native_value = gemma4_vision_native_attention_input(value)
    native_shape = models.tensor_shape(native_query)
    use_native_layout = (
        len(native_shape) == 4
        and native_shape == models.tensor_shape(native_key) == models.tensor_shape(native_value)
        and native_shape[1] == models.tensor_shape(query)[2]
        and native_shape[2] == models.tensor_shape(query)[1]
    )
    if use_native_layout:
        query, key, value = native_query, native_key, native_value

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
            "scale": decomposed_attention_scale(qk_bmm),
            "is_causal": False,
            "input_layout": "bthd" if use_native_layout else "bhqd_bhds_bhsd",
            "output_layout": "bthd_flat",
            "window_size": 0,
            "additive_mask": mask is not None,
            "dropped_softmax_nan_guard": True,
        },
    )


def gemma4_vision_native_attention_input(node: models.Node, max_depth: int = 12) -> models.Node:
    current = node

    for _ in range(max_depth):
        if len(current.parents) != 1:
            break

        parent = current.parents[0]
        current_shape = models.tensor_shape(current)
        parent_shape = models.tensor_shape(parent)

        if current.target in {"cactus.expand", "aten.expand.default", "cactus.view", "aten.view.default", "aten.reshape.default"}:
            if current_shape == parent_shape:
                current = parent
                continue
            break

        if current.target in {"cactus.scalar_multiply", "aten.mul.Scalar"}:
            scale = safe_float(current.attrs.get("value", current.attrs.get("other")))
            if scale is not None:
                current = parent
                continue
            break

        if current.target in {"cactus.precision_cast", "aten._to_copy.default"}:
            current = parent
            continue

        if current.target in {"cactus.transpose", "aten.permute.default"}:
            permutation = tuple(current.attrs.get("permutation", current.attrs.get("dims", ())))
            if permutation in {(0, 2, 1, 3), (0, 1, 3, 2)} and len(parent_shape) == 4:
                expected_shape = tuple(parent_shape[index] for index in permutation)
                if current_shape == expected_shape:
                    current = parent
                    continue

        break

    return current


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

    matched_nodes = remove_external_fanout_nodes(tuple(matched.values()), source)
    return tuple(sorted(matched_nodes, key=lambda node: (node.index, node.name)))


def remove_external_fanout_nodes(nodes: tuple[models.Node, ...], source: models.Node) -> tuple[models.Node, ...]:
    kept = {node.name for node in nodes}
    node_by_name = {node.name: node for node in nodes}
    changed = True

    while changed:
        changed = False

        for name in tuple(kept):
            node = node_by_name[name]

            if node is source:
                continue

            if all(child.name in kept for child in node.children):
                continue

            kept.remove(name)
            changed = True

    return tuple(node for node in nodes if node.name in kept)


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
        attrs["scale"] = 1.0
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
                attrs["cache_window_size"] = gemma4_cache_window_size(key_cat)
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
    attrs["scale"] = 1.0
    attrs["output_layout"] = "bthd_flat"
    attrs["window_size"] = gemma4_attention_window_size(result.external_inputs[1])
    attrs["cache_window_size"] = gemma4_cache_window_size(result.external_inputs[1])

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

        if parent.target in {"aten._to_copy.default", "cactus.precision_cast"}:
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

        if parent.target in {"aten._to_copy.default", "cactus.precision_cast"}:
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
            "cache_window_size": gemma4_cache_window_size(key_cat),
        },
    )


def gemma4_attention_window_size(key_cat: models.Node) -> int:
    for node in ancestor_nodes(key_cat):
        layer_index = node.cache.layer_index if node.cache is not None else None

        if layer_index is None:
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

        parent = current.parents[0]

        if current.target in {"cactus.expand", "aten.expand.default", "cactus.view", "aten.view.default", "aten.reshape.default"}:
            if models.tensor_shape(current) == models.tensor_shape(parent):
                current = parent
                continue

            return current, layout

        if current.target == "cactus.scalar_multiply" and safe_float(current.attrs.get("value")) == 1.0:
            current = parent
            continue

        if current.target == "aten.mul.Scalar" and safe_float(current.attrs.get("other")) == 1.0:
            current = parent
            continue

        if current.target in {"aten._to_copy.default", "cactus.precision_cast"} and str(current.attrs.get("dtype")) == "torch.float32":
            current = parent
            continue

        if is_bqhd_to_bhqd_transpose(current):
            current = parent
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


def lfm_attention_query_input(node: models.Node) -> models.Node:
    current = node

    for _ in range(8):
        if len(current.parents) != 1:
            return current

        parent = current.parents[0]

        if current.target in {"aten.mul.Scalar", "cactus.scalar_multiply"} and lfm_attention_scale_value(current) is not None:
            current = parent
            continue

        if current.target in {"aten._to_copy.default", "cactus.precision_cast"} and str(current.attrs.get("dtype")) == "torch.float32":
            current = parent
            continue

        return current

    return current


def lfm_attention_scale_value(node: models.Node) -> float | None:
    value = safe_float(node.attrs.get("value"))

    if value is None:
        value = safe_float(node.attrs.get("other"))

    return value


def lfm_attention_cache_cat(node: models.Node, role: str) -> models.Node | None:
    cache_cat = cache_cat_ancestor(node, role, max_depth=32)

    if cache_cat is not None:
        return cache_cat

    return prefill_cache_cat_ancestor(node, max_depth=32)


def lfm_attention_window_size(key_cat: models.Node) -> int:
    for node in ancestor_nodes(key_cat):
        if node.cache is not None and node.cache.window_size is not None:
            return int(node.cache.window_size)

    return 0


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
    match = models.find_cache_concat_ancestor(node, role, max_depth=max_depth)
    return match.concat if match is not None else None


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
    WHISPER_ATTENTION_LAYOUT: match_whisper_attention_layout,
    LFM_BMM_MASKED_ATTENTION: match_lfm_bmm_masked_attention,
    GENERIC_CACHED_ATTENTION: match_generic_cached_attention,
    GEMMA4_ROPE_TABLE_LOOKUP: match_gemma4_rope_table_lookup,
}
