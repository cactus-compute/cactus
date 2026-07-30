from collections.abc import Iterable
from pathlib import Path

from . import match, match_utils, models
from ..Converter import models as CModels
from ..Fusions import fusions as Fusions
from ..Fusions import models as FModels


#TODO: Clean up code

def simplify(
    layer_map: CModels.LayerMap,
    *,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    max_fusions: int | None = None,
    max_passes: int = 3,
) -> CModels.LayerMap:
    graph = models.Graph.from_map(layer_map)
    total_fusions = 0

    for _ in range(max_passes):
        remaining_fusions = None if max_fusions is None else max_fusions - total_fusions

        if remaining_fusions is not None and remaining_fusions <= 0:
            break

        simplified_graph = rev_top_sort(
            graph,
            inference_mode=layer_map.task,
            input_modalities=input_modalities,
            fusion_fields=fusion_fields,
            max_fusions=remaining_fusions,
        )

        if not simplified_graph.fusions:
            break

        total_fusions += len(simplified_graph.fusions)
        graph = simplified_graph

    graph = fuse_gemma4_decode_attention(graph, layer_map.task, fusion_fields)
    return graph.remove_noop_nodes().to_layer_map()


def write_simplified_json(
    layer_map: CModels.LayerMap,
    output_path: str | Path,
    *,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    max_fusions: int | None = None,
    max_passes: int = 3,
) -> CModels.LayerMap:
    simplified = simplify(
        layer_map,
        input_modalities=input_modalities,
        fusion_fields=fusion_fields,
        max_fusions=max_fusions,
        max_passes=max_passes,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(simplified.model_dump_json(indent=4), encoding="utf-8")
    return simplified


def rev_top_sort(
    graph: models.Graph,
    *,
    inference_mode: str | None = None,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    max_fusions: int | None = None,
) -> models.Graph:
    consumed_names: set[str] = set()
    fusion_results: list[models.FusionResult] = []

    for node in reverse_topological_nodes(graph):
        if node.name in consumed_names:
            continue

        result = try_match_from_node(
            node,
            graph,
            consumed_names=consumed_names,
            inference_mode=inference_mode,
            input_modalities=input_modalities,
            fusion_fields=fusion_fields,
        )

        if result is None:
            continue

        fusion_results.append(result)
        consumed_names.update(result.consumed_node_names)

        if max_fusions is not None and len(fusion_results) >= max_fusions:
            break

    graph.fusions = fusion_results
    return graph.apply_fusions(fusion_results)


def try_match_from_node(
    node: models.Node,
    graph: models.Graph,
    *,
    consumed_names: set[str] | frozenset[str] = frozenset(),
    inference_mode: str | None = None,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
) -> models.FusionResult | None:
    if not node.is_operation:
        return None

    for fusion in candidate_fusions_for_node(node, fusion_fields):
        bindings = bind_matching_fusion(
            node,
            graph,
            fusion,
            inference_mode=inference_mode,
            input_modalities=input_modalities,
            fusion_fields=fusion_fields,
        )

        if bindings is None:
            continue

        result = fusion_result_from_bindings(fusion, node, bindings)

        if is_noop_fusion(node, result):
            continue

        if result.consumed_node_names.intersection(consumed_names):
            continue

        return result

    return None


def is_noop_fusion(node: models.Node, result: models.FusionResult) -> bool:
    return node.target == result.target and len(result.matched_nodes) == 1


def bind_matching_fusion(
    node: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    *,
    inference_mode: str | None = None,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
) -> dict[str, models.Node] | None:
    bindings = match_utils.bind_fusion_graph(node, fusion.graph, match.match_nodes)

    if bindings is None:
        return None

    if not all(
        matcher(node, graph, fusion, bindings, inference_mode, input_modalities, fusion_fields)
        for matcher in match.FUSION_DEFINITION_MATCHERS
    ):
        return None

    return bindings


def fusion_result_from_bindings(
    fusion: FModels.FusionDefinition,
    source: models.Node,
    bindings: dict[str, models.Node],
) -> models.FusionResult:
    matched_nodes = tuple(sorted(unique_nodes(bindings.values()), key=lambda node: (node.index, node.name)))
    external_inputs = collect_external_inputs(fusion.graph, bindings)
    attrs = match_utils.collect_definition_attrs(fusion.graph, bindings)

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=matched_nodes,
        bindings=dict(bindings),
        external_inputs=external_inputs,
        attrs=attrs,
    )


def collect_external_inputs(
    fusion: FModels.FusionGraph,
    bindings: dict[str, models.Node],
) -> tuple[models.Node, ...]:
    external_inputs: list[models.Node] = []

    for input_spec in fusion.inputs:
        if input_spec.metadata.get("drop_after_fusion"):
            continue

        external_inputs.extend(match_utils.get_fusion_input_nodes(input_spec, bindings))

    for cache_input in fusion.cache_inputs:
        cache_node = match_utils.get_node_ref_parent(cache_input.source, bindings)

        if cache_node is not None and not any(existing is cache_node for existing in external_inputs):
            external_inputs.append(cache_node)

    return tuple(external_inputs)


def candidate_fusions_for_node(
    node: models.Node,
    fusion_fields: tuple[str, ...] = (),
) -> tuple[FModels.FusionDefinition, ...]:
    candidates: list[FModels.FusionDefinition] = []
    seen: set[int] = set()

    for key in (node.target, node.node_type):
        for fusion in Fusions.FUSIONS_BY_ROOT_OP.get(key, ()):
            if id(fusion) in seen:
                continue

            if not fusion_enabled_for_fields(fusion, fusion_fields):
                continue

            seen.add(id(fusion))
            candidates.append(fusion)

    return tuple(sorted(candidates, key=fusion_priority, reverse=True))


def fusion_enabled_for_fields(fusion: FModels.FusionDefinition, fusion_fields: tuple[str, ...]) -> bool:
    if not fusion_fields or not fusion.fusion_fields:
        return True

    selected_fields = set(fusion_fields)
    fusion_specific_fields = set(fusion.fusion_fields) - {"generic"}

    if not fusion_specific_fields:
        return "generic" in selected_fields

    if fusion_specific_fields == {"direct"}:
        return "generic" in selected_fields or "direct" in selected_fields

    return not fusion_specific_fields.isdisjoint(selected_fields - {"generic"})


def fusion_priority(fusion: FModels.FusionDefinition) -> tuple[int, int, int, int, int]:
    required_attrs = fusion.metadata.get("required_attrs", {})

    return (
        len(fusion.graph.nodes),
        len(fusion.graph.edges),
        len(fusion.graph.constraints),
        len(required_attrs),
        int("direct" not in fusion.fusion_fields),
    )


def reverse_topological_nodes(graph: models.Graph) -> tuple[models.Node, ...]:
    return tuple(reversed(models.topological_sort(graph)))


def unique_nodes(nodes: Iterable[models.Node]) -> tuple[models.Node, ...]:
    unique: list[models.Node] = []

    for node in nodes:
        if any(existing is node for existing in unique):
            continue

        unique.append(node)

    return tuple(unique)


def fuse_gemma4_decode_attention(
    graph: models.Graph,
    inference_mode: str | None,
    fusion_fields: tuple[str, ...] = (),
) -> models.Graph:
    if inference_mode != "decode_with_cache":
        return graph

    if fusion_fields and "gemma4_attention" not in fusion_fields and "attention" not in fusion_fields:
        return graph

    fusion = Fusions.FUSIONS["attention_masked"]
    results: list[models.FusionResult] = []

    for source in graph.nodes:
        match_result = gemma4_decode_attention_match(graph, source, fusion)

        if match_result is not None:
            results.append(match_result)

    if not results:
        return graph

    return graph.apply_fusions(results)


def gemma4_decode_attention_match(
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

    return models.FusionResult.from_match(
        fusion=fusion,
        source=source,
        matched_nodes=(source,),
        external_inputs=(query, key_cat, value_cat),
        attrs={
            "scale": 1.0,
            "input_layout": "bhqd_bhds_bhsd",
            "output_layout": "bhqd",
            "window_size": 0,
        },
    )


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
