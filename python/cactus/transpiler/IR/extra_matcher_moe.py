from __future__ import annotations

from typing import Any

from . import models, match_utils
from .extra_matcher_common import *
from ..Fusions import models as FModels

def get_repeated_subgraph(fusion: FModels.FusionGraph, name: str) -> FModels.RepeatedSubgraph | None:
    """Fetches a named repeated subgraph declaration from a fusion graph."""
    for subgraph in fusion.repeated_subgraphs:
        if subgraph.name == name:
            return subgraph

    return None

def collect_repeated_subgraph_bindings(graph: models.Graph, subgraph: FModels.RepeatedSubgraph) -> tuple[dict[str, models.Node], ...]:
    """Finds every real subgraph binding that matches a repeated pattern."""
    from . import match

    matches: list[dict[str, models.Node]] = []

    for candidate in graph.nodes:
        candidate_bindings = match_utils.bind_fusion_graph(candidate, subgraph.graph, match.match_nodes)

        if candidate_bindings is None:
            continue

        if not match.match_fusion_bindings(candidate, graph, subgraph.graph, candidate_bindings):
            continue

        matches.append(candidate_bindings)

    return tuple(matches)

def relevant_moe_branch_bindings(graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> tuple[dict[str, models.Node], ...]:
    """Filters repeated expert branches down to ones driven by MoE routing."""
    subgraph = get_repeated_subgraph(fusion, spec["repeated_subgraph"])

    if subgraph is None:
        return ()

    topk_node = bindings.get(spec.get("topk_node", "topk"))
    combine_node = bindings.get(spec.get("combine_node", "moe_combine"))
    topk_nodes = (topk_node,) if topk_node is not None else find_reachable_topk_nodes(graph, combine_node, spec)

    if not topk_nodes:
        return ()

    max_depth = int(spec.get("max_depth", 16))
    branch_bindings = collect_repeated_subgraph_bindings(graph, subgraph)
    relevant = []

    for branch_binding in branch_bindings:
        routing_weight = match_utils.get_first_input_by_role(subgraph.graph, branch_binding, spec["routing_weight_role"])

        if routing_weight is None:
            continue

        routing_sources = tuple(source for topk in topk_nodes for source in (topk, first_parent(topk)) if source is not None)

        if not has_any_path(routing_sources, routing_weight, max_depth):
            continue

        relevant.append(branch_binding)

    return tuple(relevant)

def required_moe_branch_count(subgraph: FModels.RepeatedSubgraph, topk_node: models.Node | None, spec: dict[str, Any]) -> int:
    """Determines how many routed expert branches must be proven."""
    if "min_count" in spec:
        return int(spec["min_count"])

    if topk_node is not None and spec.get("min_count_from_topk", True):
        k = topk_node.attrs.get("k", match_utils.MISSING)

        if isinstance(k, int):
            return k

    return subgraph.min_count

def get_num_experts_from_inputs(fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> int | None:
    """Infers the MoE expert count from weight/bias shapes."""
    candidates: list[int] = []

    for role in (spec["gate_up_weight_role"], spec["down_weight_role"], spec.get("router_bias_role")):
        if role is None:
            continue

        input_node = match_utils.get_first_input_by_role(fusion, bindings, role)
        shape = match_utils.get_tensor_shape(input_node)

        if not shape:
            continue

        dim = get_known_int(shape[0])

        if dim is not None:
            candidates.append(dim)

    if not candidates:
        return None

    if len(set(candidates)) != 1:
        return None

    return candidates[0]

def split_sizes_are_valid(split_node: models.Node, gate_up_node: models.Node, down_weight: models.Node | None, spec: dict[str, Any]) -> bool:
    """Checks the grouped gate/up projection split."""
    split_sizes = get_attr(split_node, "split_sizes", "arg_1")

    if not isinstance(split_sizes, (list, tuple)) or len(split_sizes) != 2:
        return bool(spec.get("allow_missing", False))

    left_size = get_known_int(split_sizes[0])
    right_size = get_known_int(split_sizes[1])

    if left_size is None or right_size is None:
        return bool(spec.get("allow_symbolic", True))

    if left_size != right_size:
        return False

    gate_up_output_dim = get_shape_dim(gate_up_node, -1)

    if gate_up_output_dim is not None and gate_up_output_dim != left_size + right_size:
        return False

    down_input_dim = get_shape_dim(down_weight, -1)

    if down_input_dim is not None and down_input_dim != right_size:
        return False

    return True

def grouped_moe_rows_match_topk(hidden_node: models.Node | None, gate_up_node: models.Node, topk_node: models.Node, spec: dict[str, Any]) -> bool:
    """Checks that routed grouped-mm rows equal token count times top-k."""
    token_count = get_shape_dim(hidden_node, 0)
    grouped_rows = get_shape_dim(gate_up_node, 0)
    topk = get_known_int(get_attr(topk_node, "k", "arg_1"))

    if token_count is None or grouped_rows is None or topk is None:
        return bool(spec.get("allow_missing", False))

    return grouped_rows == token_count * topk

def grouped_moe_weight_permute_is_valid(node: models.Node | None) -> bool:
    """Checks the expert weight permutation used before grouped_mm."""
    if node is None:
        return False

    return match_utils.values_equal(get_attr(node, "permutation", "arg_1"), [0, 2, 1])

def find_reachable_topk_nodes(graph: models.Graph, combine_node: models.Node | None, spec: dict[str, Any]) -> tuple[models.Node, ...]:
    """Finds top-k router nodes that can flow into the MoE combine output."""
    if combine_node is None:
        return ()

    max_depth = int(spec.get("max_depth", 16))
    return tuple(node for node in graph.nodes if node.target == "aten.topk.default" and has_path(node, combine_node, max_depth))

def first_parent(node: models.Node) -> models.Node | None:
    """Returns the first parent of a node, if it has one."""
    return node.parents[0] if node.parents else None

def branch_output_reaches_combine(branch_binding: dict[str, models.Node], subgraph: FModels.RepeatedSubgraph, combine_node: models.Node, spec: dict[str, Any]) -> bool:
    """Checks whether one matched expert branch contributes to the combine node."""
    weighted_node_name = spec.get("weighted_node", subgraph.graph.root)
    weighted_node = branch_binding.get(weighted_node_name)

    if weighted_node is None:
        return False

    return has_path(weighted_node, combine_node, int(spec.get("max_depth", 16)))

def has_any_path(sources: tuple[models.Node | None, ...], dest: models.Node, max_depth: int) -> bool:
    """Returns True if any non-None source has a forward path to `dest`."""
    for source in sources:
        if source is not None and has_path(source, dest, max_depth):
            return True

    return False

def has_path(source: models.Node, dest: models.Node, max_depth: int) -> bool:
    """Performs a bounded forward reachability check in the IR DAG."""
    if source is dest:
        return True

    seen: set[int] = set()

    def visit(node: models.Node, depth: int) -> bool:
        if depth > max_depth or id(node) in seen:
            return False

        seen.add(id(node))

        for child in node.children:
            if child is dest:
                return True

            if visit(child, depth + 1):
                return True

        return False

    return visit(source, 0)

def match_moe_expert_branch_routing(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Verifies that enough matched expert branches are driven by router top-k."""
    subgraph = get_repeated_subgraph(fusion, spec["repeated_subgraph"])
    combine_node = bindings.get(spec.get("combine_node", "moe_combine"))
    topk_node = bindings.get(spec.get("topk_node", "topk"))

    if subgraph is None:
        return bool(spec.get("allow_missing", False))

    if topk_node is None:
        topk_nodes = find_reachable_topk_nodes(graph, combine_node, spec)
        topk_node = topk_nodes[0] if topk_nodes else None

    if topk_node is None:
        return bool(spec.get("allow_missing", False))

    branch_bindings = relevant_moe_branch_bindings(graph, fusion, bindings, spec)
    min_count = required_moe_branch_count(subgraph, topk_node, spec)

    return len(branch_bindings) >= min_count

def match_moe_routing_weights_combine(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Verifies that routed expert outputs are actually combined into MoE output."""
    subgraph = get_repeated_subgraph(fusion, spec["repeated_subgraph"])
    topk_node = bindings.get(spec.get("topk_node", "topk"))
    combine_node = bindings.get(spec["combine_node"])

    if subgraph is None or combine_node is None:
        return bool(spec.get("allow_missing", False))

    if topk_node is None:
        topk_nodes = find_reachable_topk_nodes(graph, combine_node, spec)
        topk_node = topk_nodes[0] if topk_nodes else None

    if topk_node is None:
        return bool(spec.get("allow_missing", False))

    branch_bindings = relevant_moe_branch_bindings(graph, fusion, bindings, spec)
    combined_branch_count = sum(
        1 for branch_binding in branch_bindings
        if branch_output_reaches_combine(branch_binding, subgraph, combine_node, spec)
    )

    return combined_branch_count >= required_moe_branch_count(subgraph, topk_node, spec)

def match_grouped_moe_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Verifies the grouped-mm MoE structure exported by LFM-style MoE models."""
    topk_node = bindings.get(spec["topk_node"])
    topk_indices_node = bindings.get(spec["topk_indices_node"])
    gate_up_grouped_node = bindings.get(spec["gate_up_grouped_node"])
    down_grouped_node = bindings.get(spec["down_grouped_node"])
    split_node = bindings.get(spec["split_node"])
    offsets_node = bindings.get(spec["offsets_node"])
    combine_node = bindings.get(spec["combine_node"])
    hidden_node = bindings.get("moe_hidden_view")
    gate_up_weight_transpose = bindings.get("moe_gate_up_weight_transpose")
    down_weight_transpose = bindings.get("moe_down_weight_transpose")
    down_weight = match_utils.get_first_input_by_role(fusion, bindings, spec["down_weight_role"])

    if any(node is None for node in (topk_node, topk_indices_node, gate_up_grouped_node, down_grouped_node, split_node, offsets_node, combine_node)):
        return bool(spec.get("allow_missing", False))

    if get_known_int(get_attr(topk_node, "k", "arg_1")) is None:
        return False

    if not match_utils.values_equal(get_attr(topk_indices_node, "index", "arg_1"), 1):
        return False

    if not grouped_moe_weight_permute_is_valid(gate_up_weight_transpose):
        return False

    if not grouped_moe_weight_permute_is_valid(down_weight_transpose):
        return False

    gate_up_offsets = match_utils.get_parent(gate_up_grouped_node, 2)
    down_offsets = match_utils.get_parent(down_grouped_node, 2)

    if gate_up_offsets is not offsets_node or down_offsets is not offsets_node:
        return False

    if not match_utils.values_equal(get_attr(offsets_node, "dim", "axis", "arg_1"), 0):
        return False

    if not split_sizes_are_valid(split_node, gate_up_grouped_node, down_weight, spec):
        return False

    num_experts = get_num_experts_from_inputs(fusion, bindings, spec)
    histc_node = bindings.get("moe_histc")
    histc_bins = get_known_int(get_attr(histc_node, "bins", "arg_1")) if histc_node is not None else None

    if num_experts is not None and histc_bins is not None and num_experts != histc_bins:
        return False

    if not grouped_moe_rows_match_topk(hidden_node, gate_up_grouped_node, topk_node, spec):
        return False

    if not match_utils.values_equal(get_attr(combine_node, "dim", "axis", "arg_1"), 1):
        return False

    return True
