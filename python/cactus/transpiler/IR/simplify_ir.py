import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from . import match, match_utils, models, special_fusions
from ..Converter import models as CModels
from ..Fusions import fusions as Fusions
from ..Fusions import models as FModels

def simplify(
    layer_map: CModels.LayerMap,
    *,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
    max_fusions: int | None = None,
    max_passes: int = 3,
) -> CModels.LayerMap:
    graph = models.Graph.from_map(layer_map)
    total_fusions = 0

    # First match the exported structure as-is: some model-specific patterns
    # intentionally include clone/contiguous nodes. Then remove structural
    # no-ops and run another convergence phase so cleanup-exposed patterns are
    # fused in the same invocation. Cleaning before the first phase loses the
    # exact LFM attention and short-convolution patterns.
    for phase in range(2):
        for _ in range(max_passes):
            remaining_fusions = None if max_fusions is None else max_fusions - total_fusions

            if remaining_fusions is not None and remaining_fusions <= 0:
                break

            simplified_graph = rev_top_sort(
                graph,
                inference_mode=layer_map.task,
                input_modalities=input_modalities,
                fusion_fields=fusion_fields,
                disabled_fusion_fields=disabled_fusion_fields,
                disabled_fusions=disabled_fusions,
                max_fusions=remaining_fusions,
            )

            if not simplified_graph.fusions:
                break

            total_fusions += len(simplified_graph.fusions)
            graph = simplified_graph

        if phase == 0:
            graph = graph.remove_noop_nodes()

    graph = graph.remove_noop_nodes()
    if layer_map.task == "decode_with_cache" and "decode_qkv" in fusion_fields:
        graph = fuse_decode_qkv_projections(graph)
    if layer_map.task == "decode_with_cache" and "decode_projection_pair" in fusion_fields:
        graph = fuse_decode_projection_pairs(graph)
    return graph.remove_noop_nodes().to_layer_map()

def fuse_decode_qkv_projections(graph: models.Graph) -> models.Graph:
    """Combine sibling decode Q/K/V linears into a shared-transform CQ op."""
    return fuse_decode_projection_group(
        graph, roles=("q", "k", "v"), path_template=".self_attn.{}_proj",
        target="cactus.qkv_tq_fused", fusion_name="decode_qkv", stem_suffix="qkv",
    )

def module_path_for_node(node: models.Node) -> str:
    stack = node.module_stack
    if not isinstance(stack, list):
        return ""
    paths = [str(item.get("module_path", "")) for item in stack if isinstance(item, dict)]
    return paths[-1] if paths else ""

def fuse_decode_projection_pairs(graph: models.Graph) -> models.Graph:
    """Share the CQ input transform for sibling LFM feed-forward W1/W3 projections."""
    return fuse_decode_projection_group(
        graph, roles=("w1", "w3"), path_template=".feed_forward.{}",
        target="cactus.projection_pair_tq_fused",
        fusion_name="decode_projection_pair", stem_suffix="w1_w3",
    )

def fuse_decode_projection_group(
    graph: models.Graph,
    *,
    roles: tuple[str, ...],
    path_template: str,
    target: str,
    fusion_name: str,
    stem_suffix: str,
) -> models.Graph:
    grouped: dict[str, dict[str, models.Node]] = {}
    for node in graph.nodes:
        if node.target != "cactus.linear" or len(node.parents) != 2:
            continue
        path = module_path_for_node(node)
        role = next((role for role in roles if path.endswith(path_template.format(role))), None)
        if role is None:
            continue
        grouped.setdefault(path.rsplit(".", 1)[0], {})[role] = node

    replacements: dict[str, str] = {}
    inserted_before: dict[str, tuple[models.Node, ...]] = {}
    for layer_path, projections in grouped.items():
        if set(projections) != set(roles):
            continue
        ordered = tuple(projections[role] for role in roles)
        activation_views = tuple(projection.parents[0] for projection in ordered)
        if any(len(view.parents) != 1 for view in activation_views):
            continue
        hidden = activation_views[0].parents[0]
        if any(view.parents[0].name != hidden.name for view in activation_views[1:]):
            continue
        sizes = tuple(models.tensor_shape(projection)[-1] for projection in ordered)
        if any(not isinstance(size, int) or size <= 0 for size in sizes):
            continue

        stem = models.sanitize_node_name(f"{layer_path}_{stem_suffix}")
        fused_name = f"{stem}_fused"
        suffix = 0
        while fused_name in graph.nodes_map or fused_name in replacements.values():
            suffix += 1
            fused_name = f"{stem}_fused_{suffix}"
        hidden_shape = list(models.tensor_shape(hidden))
        if not hidden_shape:
            continue
        hidden_shape[-1] = sum(sizes)
        fused_meta = dict(hidden.tensor_output_meta or {})
        fused_meta["shape"] = hidden_shape
        first = ordered[0]
        fused = models.Node(
            index=first.index, name=fused_name, node_type="call_function",
            target=target,
            args=[{"node": hidden.name}, *({"node": projection.parents[1].name} for projection in ordered)],
            kwargs={}, users=(), tensor_output_meta=fused_meta,
            module_stack=first.module_stack, value_kind=FModels.ValueKind.ACTIVATION,
            attrs={}, ir_metadata={"fusion": {
                "name": fusion_name, "target": target,
                "matched_nodes": [projection.name for projection in ordered],
            }}, cache=None,
        )
        slices: list[models.Node] = []
        offset = 0
        for role, projection, size in zip(roles, ordered, sizes, strict=True):
            slice_name = f"{fused_name}_{role}"
            slice_shape = list(hidden_shape)
            slice_shape[-1] = size
            slice_meta = dict(projection.tensor_output_meta or {})
            slice_meta["shape"] = slice_shape
            attrs = {"axis": -1, "start": offset, "length": size, "step": 1}
            slices.append(models.Node(
                index=projection.index, name=slice_name, node_type="call_function",
                target="cactus.slice", args=[{"node": fused_name}], kwargs=dict(attrs), users=(),
                tensor_output_meta=slice_meta, module_stack=projection.module_stack,
                value_kind=FModels.ValueKind.ACTIVATION, attrs=attrs, ir_metadata={}, cache=None,
            ))
            replacements[projection.name] = slice_name
            offset += size
        inserted_before[first.name] = (fused, *slices)

    if not replacements:
        return graph

    rewritten: list[models.Node] = []
    for node in graph.nodes:
        rewritten.extend(inserted_before.get(node.name, ()))
        if node.name in replacements:
            continue
        rewritten.append(models.clone_rewritten_node(node, replacements))
    return models.prune_dead_nodes(models.rebuild_graph(tuple(rewritten), graph, tuple(graph.fusions)))

def simplify_repeated(
    layer_map: CModels.LayerMap,
    *,
    minimum_rounds: int = 2,
    maximum_rounds: int = 8,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
    max_fusions: int | None = None,
    max_passes: int = 3,
) -> CModels.LayerMap:
    """Run complete simplification rounds until the result is stable."""
    if minimum_rounds < 1:
        raise ValueError("minimum_rounds must be at least 1")
    if maximum_rounds < minimum_rounds:
        raise ValueError("maximum_rounds must be >= minimum_rounds")

    current = layer_map
    current_json = current.model_dump_json()
    for round_index in range(maximum_rounds):
        simplified = simplify(
            current,
            input_modalities=input_modalities,
            fusion_fields=fusion_fields,
            disabled_fusion_fields=disabled_fusion_fields,
            disabled_fusions=disabled_fusions,
            max_fusions=max_fusions,
            max_passes=max_passes,
        )
        simplified_json = simplified.model_dump_json()
        changed = simplified_json != current_json
        current = simplified
        current_json = simplified_json
        if round_index + 1 >= minimum_rounds and not changed:
            break
    return current

def write_simplified_json(
    layer_map: CModels.LayerMap,
    output_path: str | Path,
    *,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
    max_fusions: int | None = None,
    max_passes: int = 3,
    fusion_report_path: str | Path | None = None,
) -> CModels.LayerMap:
    simplified = simplify_repeated(
        layer_map,
        input_modalities=input_modalities,
        fusion_fields=fusion_fields,
        disabled_fusion_fields=disabled_fusion_fields,
        disabled_fusions=disabled_fusions,
        max_fusions=max_fusions,
        max_passes=max_passes,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(simplified.model_dump_json(indent=4), encoding="utf-8")
    report_path = Path(fusion_report_path) if fusion_report_path is not None else path.with_suffix(".fusion_report.json")
    report_path.write_text(json.dumps(build_fusion_report(
        layer_map,
        simplified,
        input_modalities=input_modalities,
        fusion_fields=fusion_fields,
        disabled_fusion_fields=disabled_fusion_fields,
        disabled_fusions=disabled_fusions,
    ), indent=4), encoding="utf-8")
    return simplified

def build_fusion_report(
    original: CModels.LayerMap,
    simplified: CModels.LayerMap,
    *,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
) -> dict[str, object]:
    before = models.Graph.from_map(original)
    after = models.Graph.from_map(simplified)

    def op_counts(graph: models.Graph) -> dict[str, int]:
        return dict(sorted(Counter(node.target for node in graph.nodes if node.is_operation).items()))

    applied: list[dict[str, object]] = []
    for node in after.nodes:
        fusion = node.ir_metadata.get("fusion") if isinstance(node.ir_metadata, dict) else None
        if not isinstance(fusion, dict):
            continue
        applied.append({
            "node": node.name,
            "name": str(fusion.get("name", "")),
            "target": node.target,
            "matched_nodes": list(fusion.get("matched_nodes", ())),
        })

    missed: list[dict[str, object]] = []
    for node in after.nodes:
        if not node.is_operation:
            continue
        candidates = candidate_fusions_for_node(
            node,
            fusion_fields,
            disabled_fusion_fields=disabled_fusion_fields,
            disabled_fusions=disabled_fusions,
        )
        candidate_misses: list[dict[str, str]] = []
        for fusion in candidates:
            reason = fusion_miss_reason(
                node, after, fusion,
                inference_mode=after.task,
                input_modalities=input_modalities,
                fusion_fields=fusion_fields,
            )
            if reason:
                candidate_misses.append({"fusion": fusion.name, "reason": reason})
        if candidate_misses:
            missed.append({"node": node.name, "target": node.target, "candidates": candidate_misses})

    return {
        "model_name": before.model_name,
        "task": before.task,
        "before": {"nodes": len(before.nodes), "operations": op_counts(before)},
        "after": {"nodes": len(after.nodes), "operations": op_counts(after)},
        "applied_fusions": applied,
        "applied_fusion_counts": dict(sorted(Counter(item["name"] for item in applied).items())),
        "missed_candidates": missed,
        "disabled_fusion_fields": list(disabled_fusion_fields),
        "disabled_fusions": list(disabled_fusions),
    }

def fusion_miss_reason(
    node: models.Node,
    graph: models.Graph,
    fusion: FModels.FusionDefinition,
    *,
    inference_mode: str | None,
    input_modalities: tuple[str, ...],
    fusion_fields: tuple[str, ...],
) -> str:
    if special_fusions.has_special_matcher(fusion):
        result = special_fusions.match_special_fusion(
            node, graph, fusion,
            inference_mode=inference_mode,
            input_modalities=input_modalities,
            fusion_fields=fusion_fields,
        )
        return "" if result is not None else "special_matcher"

    bindings = match_utils.bind_fusion_graph(node, fusion.graph, match.match_nodes)
    if bindings is None:
        return "structure"
    for matcher in match.FUSION_DEFINITION_MATCHERS:
        if not matcher(node, graph, fusion, bindings, inference_mode, input_modalities, fusion_fields):
            return matcher.__name__
    return "matched_but_not_selected"

def rev_top_sort(
    graph: models.Graph,
    *,
    inference_mode: str | None = None,
    input_modalities: tuple[str, ...] = (),
    fusion_fields: tuple[str, ...] = (),
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
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
            disabled_fusion_fields=disabled_fusion_fields,
            disabled_fusions=disabled_fusions,
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
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
) -> models.FusionResult | None:
    if not node.is_operation:
        return None

    for fusion in candidate_fusions_for_node(
        node,
        fusion_fields,
        disabled_fusion_fields=disabled_fusion_fields,
        disabled_fusions=disabled_fusions,
    ):
        if special_fusions.has_special_matcher(fusion):
            result = special_fusions.match_special_fusion(
                node,
                graph,
                fusion,
                inference_mode=inference_mode,
                input_modalities=input_modalities,
                fusion_fields=fusion_fields,
            )

            if result is None:
                continue
        else:
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
    *,
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
) -> tuple[FModels.FusionDefinition, ...]:
    candidates: list[FModels.FusionDefinition] = []
    seen: set[int] = set()

    for key in (node.target, node.node_type):
        for fusion in Fusions.FUSIONS_BY_ROOT_OP.get(key, ()):
            if id(fusion) in seen:
                continue

            if not fusion_enabled_for_fields(fusion, fusion_fields):
                continue

            if fusion_disabled(fusion, disabled_fusion_fields, disabled_fusions):
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

def fusion_disabled(
    fusion: FModels.FusionDefinition,
    disabled_fusion_fields: tuple[str, ...] = (),
    disabled_fusions: tuple[str, ...] = (),
) -> bool:
    disabled_names = set(disabled_fusions)

    if fusion.name in disabled_names or fusion.cactus_op in disabled_names:
        return True

    if disabled_fusion_fields and not set(fusion.fusion_fields).isdisjoint(disabled_fusion_fields):
        return True

    return False

def fusion_priority(fusion: FModels.FusionDefinition) -> tuple[int, int, int, int, int, int]:
    required_attrs = fusion.metadata.get("required_attrs", {})

    return (
        int(special_fusions.has_special_matcher(fusion)),
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
