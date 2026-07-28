from typing import Any, Callable

from . import models, match_utils
from ..Fusions import models as FModels

#TODO: Clean up this file
ExtraMatcher = Callable[[models.Node, models.Graph, FModels.FusionGraph, dict[str, models.Node], dict[str, Any]], bool]


WEIGHT_VALUE_KINDS = (
    FModels.ValueKind.PARAMETER,
    FModels.ValueKind.BUFFER,
    FModels.ValueKind.LIFTED_CONSTANT,
)

COPY_TARGETS = {
    "aten.clone.default",
    "aten._to_copy.default",
    "aten.contiguous.default",
}

TRANSPOSE_TARGETS = {
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
}

COS_TARGETS = {"aten.cos.default"}
SIN_TARGETS = {"aten.sin.default"}
SPLIT_TARGETS = {
    "aten.chunk.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
}


############################################ Extra Matcher Utils!!!!! ############################################
def get_bound_parent(bindings: dict[str, models.Node], node_name: str, parent_index: int) -> models.Node | None:
    """
    Returns a real parent node from a bound synthetic node/input position.

    `bindings` maps synthetic fusion-node names to real IR nodes. This helper
    first finds the real node bound to `node_name`, then asks match_utils for
    the parent at `parent_index`. It returns None when the synthetic node was
    not bound or the requested parent index does not exist.
    """
    node = bindings.get(node_name)

    if node is None:
        return None

    return match_utils.get_parent(node, parent_index)


def get_node_dim(node: models.Node, dim_index: int) -> Any:
    """
    Returns one tensor dimension from a real IR node's output shape.

    This reads `tensor_output_meta["shape"]`, supports negative indices in the
    Python style, and returns `match_utils.MISSING` instead of raising when the
    shape is unavailable or the dimension index is invalid. Matchers use that
    missing sentinel so they can either fail closed or honor `allow_missing`.
    """
    shape = match_utils.get_tensor_shape(node)

    if not shape:
        return match_utils.MISSING

    if dim_index < 0:
        dim_index += len(shape)

    if dim_index < 0 or dim_index >= len(shape):
        return match_utils.MISSING

    return shape[dim_index]


def classify_linear_rhs(node: models.Node) -> tuple[str, models.Node] | None:
    """
    Classifies the right-hand side of a linear/matmul as direct or transposed.

    The node passed here is the actual RHS input to the matmul. The helper
    delegates to the recursive implementation with an empty `seen` set so we
    can safely walk through layout-only wrapper nodes without infinite loops.
    """
    return classify_linear_rhs_(node, set())


def classify_linear_rhs_(node: models.Node, seen: set[int]) -> tuple[str, models.Node] | None:
    """
    Walks backward through simple RHS wrapper ops to find the source weight.

    It accepts either a direct parameter/buffer/lifted constant weight, or a
    valid transpose/permute/t of such a weight. Copy-like wrappers such as
    `_to_copy` and `contiguous` are skipped recursively. The return value is
    `(layout_kind, source_weight)`, where layout_kind is `direct` or
    `transposed`; None means the RHS is not safely recognizable as a weight.
    """
    if id(node) in seen:
        return None

    seen.add(id(node))

    if node.target in COPY_TARGETS and len(node.parents) == 1:
        return classify_linear_rhs_(node.parents[0], seen)

    if node.value_kind in WEIGHT_VALUE_KINDS:
        return ("direct", node)

    if node.target in TRANSPOSE_TARGETS and len(node.parents) == 1 and node.parents[0].value_kind in WEIGHT_VALUE_KINDS:
        if linear_transpose_is_valid(node, node.parents[0]):
            return ("transposed", node.parents[0])

    return None


def linear_transpose_is_valid(node: models.Node, source_weight: models.Node) -> bool:
    """
    Checks whether a transpose-like op really swaps a 2D linear weight.

    Cactus needs to know whether the RHS has already been converted from
    `[out_dim, in_dim]` to `[in_dim, out_dim]`. This accepts `t`, `transpose`
    over dimensions 0/1, and `permute([1, 0])`, but rejects non-2D weights or
    arbitrary permutations that would not be a standard linear transpose.
    """
    source_shape = match_utils.get_tensor_shape(source_weight)

    if len(source_shape) != 2:
        return False

    if node.target == "aten.t.default":
        return True

    if node.target == "aten.transpose.int":
        dim0 = node.attrs.get("dim0", match_utils.MISSING)
        dim1 = node.attrs.get("dim1", match_utils.MISSING)
        return {normalize_dim(dim0, 2), normalize_dim(dim1, 2)} == {0, 1}

    permutation = node.attrs.get("permutation", match_utils.MISSING)
    return match_utils.values_equal(permutation, [1, 0])


def linear_shapes_match(input_node: models.Node, rhs_node: models.Node, output_node: models.Node, source_weight: models.Node, layout_kind: str, spec: dict[str, Any]) -> bool:
    """
    Verifies the dimensions around a linear/matmul are internally consistent.

    The RHS node is the tensor consumed by the real matmul, so its last two dims
    should be `[input_dim, output_dim]`. If that RHS came from a transposed
    original weight, the original source weight should be `[output_dim,
    input_dim]`. Missing shapes are handled through the constraint spec's
    `allow_missing` flag so callers can choose conservative or permissive mode.
    """
    input_shape = match_utils.get_tensor_shape(input_node)
    rhs_shape = match_utils.get_tensor_shape(rhs_node)
    output_shape = match_utils.get_tensor_shape(output_node)

    if len(rhs_shape) < 2:
        return bool(spec.get("allow_missing", False))

    if not input_shape or not output_shape:
        return bool(spec.get("allow_missing", False))

    input_dim = input_shape[-1]
    output_dim = output_shape[-1]

    if not match_utils.values_equal(rhs_shape[-2], input_dim):
        return False

    if not match_utils.values_equal(rhs_shape[-1], output_dim):
        return False

    source_shape = match_utils.get_tensor_shape(source_weight)

    if layout_kind == "transposed" and len(source_shape) >= 2:
        if not match_utils.values_equal(source_shape[-2], output_dim):
            return False

        if not match_utils.values_equal(source_shape[-1], input_dim):
            return False

    return True


def compatible_shapes(left_shape: list[Any], right_shape: list[Any], spec: dict[str, Any]) -> bool:
    """
    Checks whether two tensor shapes are equal or broadcast-compatible.

    This is used for things like RoPE cos/sin tables, where shapes may be
    exactly identical or broadcastable across singleton dimensions. If either
    shape is missing, the result is controlled by `allow_missing` in the spec.
    """
    if not left_shape or not right_shape:
        return bool(spec.get("allow_missing", False))

    return (
        match_utils.values_equal(left_shape, right_shape)
        or shapes_are_broadcastable(left_shape, right_shape)
        or shapes_are_broadcastable(right_shape, left_shape)
    )


def compatible_dtypes(left_node: models.Node, right_node: models.Node, spec: dict[str, Any]) -> bool:
    """
    Checks whether two nodes have matching output dtypes when required.

    Most table-like values in one fusion should have the same dtype. The check
    can be disabled with `require_same_dtype=False`; otherwise missing dtype
    metadata follows `allow_missing`, and available dtype strings are compared
    with the shared `values_equal` normalization behavior.
    """
    if not spec.get("require_same_dtype", True):
        return True

    left_dtype = get_node_dtype(left_node)
    right_dtype = get_node_dtype(right_node)

    if left_dtype is match_utils.MISSING or right_dtype is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(left_dtype, right_dtype)


def compatible_trig_sources(cos_node: models.Node, sin_node: models.Node, spec: dict[str, Any]) -> bool:
    """
    Checks whether RoPE cos and sin tables appear to come from the same angles.

    The helper walks backward from the cos/sin input nodes until it finds
    `aten.cos` and `aten.sin`, then compares the node feeding those trig ops.
    If no trig source is visible, the matcher can either allow that partial
    evidence or fail closed with `require_trig_sources=True`.
    """
    cos_sources = collect_trig_angle_sources(cos_node, COS_TARGETS, int(spec.get("max_trig_depth", 8)))
    sin_sources = collect_trig_angle_sources(sin_node, SIN_TARGETS, int(spec.get("max_trig_depth", 8)))

    if cos_sources or sin_sources:
        return cos_sources == sin_sources

    return not spec.get("require_trig_sources", False)


def collect_trig_angle_sources(node: models.Node, trig_targets: set[str], max_depth: int) -> frozenset[str]:
    """
    Finds the angle-producing parent nodes for visible trig ops upstream.

    Starting from `node`, this walks backward through parents up to `max_depth`.
    When it reaches a target such as `aten.cos.default` or `aten.sin.default`,
    it records that trig node's first parent as the angle source. The result is
    a frozenset so cos and sin paths can be compared independent of ordering.
    """
    sources: set[str] = set()
    seen: set[int] = set()

    def visit(current: models.Node, depth: int) -> None:
        """
        Recursively walks parents while avoiding cycles and depth blowups.

        The exported graph should be a DAG, but the `seen` guard makes this
        helper robust if malformed graph data or future graph rewrites ever
        introduce repeated references.
        """
        if depth > max_depth or id(current) in seen:
            return

        seen.add(id(current))

        if current.target in trig_targets:
            sources.add(current.parents[0].name if current.parents else current.name)
            return

        for parent in current.parents:
            visit(parent, depth + 1)

    visit(node, 0)
    return frozenset(sources)


def rope_tables_match_input(x_node: models.Node, cos_shape: list[Any], sin_shape: list[Any], spec: dict[str, Any]) -> bool:
    """
    Checks whether RoPE cos/sin table shapes can apply to the rotated input.

    RoPE tables often have fewer broadcast dimensions than the activation, and
    the final table dimension can represent either the full hidden/head dim or
    half of it depending on the rotate-half implementation. This delegates the
    actual table-vs-input rule to `rope_table_shape_matches_x` for both tables.
    """
    x_shape = match_utils.get_tensor_shape(x_node)

    if not x_shape:
        return bool(spec.get("allow_missing", False))

    return rope_table_shape_matches_x(cos_shape, x_shape) and rope_table_shape_matches_x(sin_shape, x_shape)


def rope_table_shape_matches_x(table_shape: list[Any], x_shape: list[Any]) -> bool:
    """
    Checks one RoPE table shape against the activation being rotated.

    The table is accepted if it is broadcastable to the activation shape. If
    not, we allow the common rotate-half form where the table's last dimension
    is half of the activation's last dimension. Symbolic or missing dimensions
    are intentionally conservative here.
    """
    if not table_shape:
        return False

    if shapes_are_broadcastable(table_shape, x_shape):
        return True

    table_last = table_shape[-1]
    x_last = x_shape[-1]

    if isinstance(table_last, int) and isinstance(x_last, int) and table_last > 0:
        return x_last in {table_last, table_last * 2}

    return False


def gate_shape_matches(gate_node: models.Node, gate_count: int, spec: dict[str, Any]) -> bool:
    """
    Checks whether an LSTM combined-gate tensor can hold `gate_count` gates.

    LSTM gate projections commonly produce a last dimension equal to
    `4 * hidden_size`. We do not know hidden_size directly here, so this checks
    that the final dimension is divisible by the expected gate count. Symbolic
    dimensions can be allowed through `allow_symbolic`.
    """
    gate_shape = match_utils.get_tensor_shape(gate_node)

    if not gate_shape:
        return bool(spec.get("allow_missing", False))

    hidden_dim = gate_shape[-1]

    if not isinstance(hidden_dim, int):
        return bool(spec.get("allow_symbolic", True))

    return hidden_dim % gate_count == 0


def has_explicit_gate_split(gate_node: models.Node, gate_count: int, max_depth: int) -> bool:
    """
    Looks for a visible split/chunk/unbind that separates combined LSTM gates.

    Shape divisibility alone is not enough to prove LSTM semantics. This walks
    forward from the combined gate node and requires a split-like op whose attr
    or fanout indicates the expected number of gates. If no such split is
    visible, the LSTM fusion fails closed.
    """
    for node in walk_children(gate_node, max_depth):
        if node.target not in SPLIT_TARGETS:
            continue

        if split_node_has_gate_count(node, gate_count):
            return True

    return False


def split_node_has_gate_count(node: models.Node, gate_count: int) -> bool:
    """
    Determines whether a split-like node creates the expected number of gates.

    Different export paths encode splits differently: `unbind` is inferred from
    child count, `chunk` stores the chunk count as positional attr `arg_1`, and
    `split_with_sizes` stores the explicit split sizes. If those attrs are not
    available, child count is used as the fallback evidence.
    """
    if node.target == "aten.unbind.int":
        return len(node.children) == gate_count

    if node.target == "aten.chunk.default":
        return match_utils.values_equal(node.attrs.get("arg_1", match_utils.MISSING), gate_count)

    split_sizes = node.attrs.get("arg_1", match_utils.MISSING)

    if isinstance(split_sizes, (list, tuple)):
        return len(split_sizes) == gate_count

    return len(node.children) == gate_count


def walk_children(node: models.Node, max_depth: int) -> tuple[models.Node, ...]:
    """
    Returns all reachable child nodes within a bounded forward traversal.

    Extra matchers use this when they need local forward evidence, such as
    proving an LSTM gate tensor is split later. The traversal tracks object ids
    to avoid revisiting shared children in a DAG and to remain safe if graph
    rewrites produce unexpected cycles.
    """
    found: list[models.Node] = []
    seen: set[int] = set()

    def visit(current: models.Node, depth: int) -> None:
        """
        Recursively collects children while respecting the maximum depth.

        The outer `walk_children` owns the result list and visited set; this
        nested helper only handles traversal mechanics.
        """
        if depth > max_depth:
            return

        for child in current.children:
            if id(child) in seen:
                continue

            seen.add(id(child))
            found.append(child)
            visit(child, depth + 1)

    visit(node, 0)
    return tuple(found)


def get_node_dtype(node: models.Node) -> Any:
    """
    Reads a node's output dtype from tensor metadata.

    The exported JSON stores tensor metadata as dictionaries. If metadata is
    missing or has some non-dict form, the function returns the shared MISSING
    sentinel so callers can decide whether to fail or allow missing metadata.
    """
    if not isinstance(node.tensor_output_meta, dict):
        return match_utils.MISSING

    return node.tensor_output_meta.get("dtype", match_utils.MISSING)


def normalize_dim(dim: Any, rank: int) -> Any:
    """
    Converts a possibly negative dimension index into a non-negative one.

    This mirrors Python indexing for known integer dimensions. Non-integer
    values are returned unchanged so symbolic or malformed values do not crash
    the matcher; they simply fail later if an exact comparison is required.
    """
    if not isinstance(dim, int):
        return dim

    return dim + rank if dim < 0 else dim


def get_repeated_subgraph(fusion: FModels.FusionGraph, name: str) -> FModels.RepeatedSubgraph | None:
    """
    Fetches a named repeated subgraph declaration from a fusion graph.

    MoE fusions store expert branches as repeated subgraphs. This helper finds
    the declaration by name so the MoE-specific matchers can reuse its pattern
    and constraints when collecting real expert-branch bindings.
    """
    for subgraph in fusion.repeated_subgraphs:
        if subgraph.name == name:
            return subgraph

    return None


def collect_repeated_subgraph_bindings(graph: models.Graph, subgraph: FModels.RepeatedSubgraph) -> tuple[dict[str, models.Node], ...]:
    """
    Finds every real subgraph binding that matches a repeated pattern.

    This scans each node as a possible root for `subgraph.graph`, runs the same
    graph binder used by normal fusion matching, then validates the full set of
    fusion matchers against that candidate. The result is not just a count: it
    returns each binding dictionary so later matchers can inspect which real
    nodes were bound to expert_gate, expert_weighted, routing_weight, etc.
    """
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
    """
    Filters repeated expert branches down to ones driven by MoE routing.

    The first step collects every expert-like branch in the whole graph. Then
    this keeps only branches whose declared `routing_weight` input is reachable
    from a top-k node or that top-k node's probability input. This is the key
    safety check that prevents us from fusing arbitrary MLP-looking branches as
    MoE experts when they are not actually selected by the router.
    """
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
    """
    Determines how many routed expert branches must be proven.

    The spec may override this directly with `min_count`. Otherwise, if a top-k
    node exposes its `k` attr, that is the strongest graph-local value because
    it tells us how many experts each token selects. If neither exists, the
    repeated subgraph's own `min_count` is used.
    """
    if "min_count" in spec:
        return int(spec["min_count"])

    if topk_node is not None and spec.get("min_count_from_topk", True):
        k = topk_node.attrs.get("k", match_utils.MISSING)

        if isinstance(k, int):
            return k

    return subgraph.min_count


def get_attr(node: models.Node, *names: str) -> Any:
    """
    Returns the first available attr from a node.

    Some attrs are normalized with friendly names such as `split_sizes`, while
    older/generated JSON may only expose fallback names such as `arg_1`. This
    lets extra matchers accept either form without duplicating logic.
    """
    for name in names:
        value = node.attrs.get(name, match_utils.MISSING)

        if value is not match_utils.MISSING:
            return value

    return match_utils.MISSING


def get_known_int(value: Any) -> int | None:
    """
    Converts plain integer-like values into int, otherwise returns None.

    Shape metadata can contain symbolic values, so matchers use this helper
    when a semantic check is only safe for known concrete dimensions.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def get_shape_dim(node: models.Node | None, dim_index: int) -> int | None:
    """
    Returns a concrete int dimension from a node shape when available.

    The helper supports negative indexing and intentionally returns None for
    missing/symbolic dimensions so callers can decide whether to fail or skip.
    """
    if node is None:
        return None

    return get_known_int(get_node_dim(node, dim_index))


def get_num_experts_from_inputs(fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> int | None:
    """
    Infers the MoE expert count from weight/bias shapes.

    For LFM grouped MoE, router bias is `[num_experts]`, gate/up expert weight
    is `[num_experts, 2 * intermediate, hidden]`, and down expert weight is
    `[num_experts, hidden, intermediate]`. Matching these first dimensions is
    a strong safety check that the grouped matmuls are really expert weights.
    """
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
    """
    Checks the grouped gate/up projection split.

    LFM projects gate and up together, then splits the final dimension into two
    equal chunks. The second chunk should match the down-projection expert
    input dimension when that shape is available.
    """
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
    """
    Checks that routed grouped-mm rows equal token count times top-k.

    This links the routing decision to the grouped expert matmul size. Missing
    or symbolic dimensions can be allowed by `allow_missing` because shape
    metadata may be incomplete in some exported graphs.
    """
    token_count = get_shape_dim(hidden_node, 0)
    grouped_rows = get_shape_dim(gate_up_node, 0)
    topk = get_known_int(get_attr(topk_node, "k", "arg_1"))

    if token_count is None or grouped_rows is None or topk is None:
        return bool(spec.get("allow_missing", False))

    return grouped_rows == token_count * topk


def grouped_moe_weight_permute_is_valid(node: models.Node | None) -> bool:
    """
    Checks the expert weight permutation used before grouped_mm.

    The LFM export stores expert weights with expert dimension first and then
    permutes the last two dims to feed grouped_mm. That appears as
    `permute([0, 2, 1])`.
    """
    if node is None:
        return False

    return match_utils.values_equal(get_attr(node, "permutation", "arg_1"), [0, 2, 1])


def find_reachable_topk_nodes(graph: models.Graph, combine_node: models.Node | None, spec: dict[str, Any]) -> tuple[models.Node, ...]:
    """
    Finds top-k router nodes that can flow into the MoE combine output.

    Some MoE fusion graphs do not bind top-k as a direct synthetic node because
    the exported routing path can contain indexing/scatter/gather wrappers.
    This helper searches the real graph for `aten.topk.default` nodes that have
    a forward path to the matched combine node within the configured depth.
    """
    if combine_node is None:
        return ()

    max_depth = int(spec.get("max_depth", 16))
    return tuple(node for node in graph.nodes if node.target == "aten.topk.default" and has_path(node, combine_node, max_depth))


def first_parent(node: models.Node) -> models.Node | None:
    """
    Returns the first parent of a node, if it has one.

    For top-k, parent 0 is usually the routing probabilities tensor. The MoE
    routing matcher treats both the top-k node and its first parent as valid
    routing sources because exported graphs may use either values or indices
    through different wrapper chains.
    """
    return node.parents[0] if node.parents else None


def branch_output_reaches_combine(branch_binding: dict[str, models.Node], subgraph: FModels.RepeatedSubgraph, combine_node: models.Node, spec: dict[str, Any]) -> bool:
    """
    Checks whether one matched expert branch contributes to the combine node.

    The branch's exposed weighted output should eventually flow into the outer
    MoE combine/sum/scatter-add node. This proves that a routed expert branch is
    actually part of the candidate MoE output instead of being an unrelated
    branch elsewhere in the graph.
    """
    weighted_node_name = spec.get("weighted_node", subgraph.graph.root)
    weighted_node = branch_binding.get(weighted_node_name)

    if weighted_node is None:
        return False

    return has_path(weighted_node, combine_node, int(spec.get("max_depth", 16)))


def has_any_path(sources: tuple[models.Node | None, ...], dest: models.Node, max_depth: int) -> bool:
    """
    Returns True if any non-None source has a forward path to `dest`.

    This is a small convenience wrapper around `has_path` for cases where a
    matcher has multiple plausible routing sources, such as top-k itself and
    the routing-probability parent of top-k.
    """
    for source in sources:
        if source is not None and has_path(source, dest, max_depth):
            return True

    return False


def has_path(source: models.Node, dest: models.Node, max_depth: int) -> bool:
    """
    Performs a bounded forward reachability check in the IR DAG.

    The function starts at `source`, walks through `children`, and succeeds if
    it reaches `dest` within `max_depth`. The visited set prevents repeated
    work on shared DAG nodes and protects against accidental cycles.
    """
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


def shapes_are_broadcastable(left_shape: list[Any], right_shape: list[Any]) -> bool:
    """
    Implements standard trailing-dimension broadcasting for two shapes.

    It compares dimensions from right to left. Each pair is compatible if the
    dims are equal or either side is 1. Extra leading dimensions on the longer
    shape are allowed, matching normal tensor broadcasting rules.
    """
    for left_dim, right_dim in zip(reversed(left_shape), reversed(right_shape)):
        if match_utils.values_equal(left_dim, 1) or match_utils.values_equal(right_dim, 1):
            continue

        if not match_utils.values_equal(left_dim, right_dim):
            return False

    return True


def get_layer_key(node: models.Node) -> str | None:
    """
    Extracts a coarse layer identifier from module stack, node name, or target.

    Cached attention checks need to know whether different matched nodes belong
    to the same decoder layer. This tries the richer module stack first, then
    falls back to name/target strings, and returns keys like `layers.0` when it
    can identify a layer index.
    """
    for value in (module_stack_text(node), node.name, node.target):
        layer_key = first_layer_key(value)

        if layer_key is not None:
            return layer_key

    return None


def module_stack_text(node: models.Node) -> str:
    """
    Converts module_stack metadata into searchable text.

    Exported module stacks may be None or structured objects/lists. The matcher
    only needs a best-effort string to search for layer-like path fragments, so
    None becomes an empty string and everything else uses repr.
    """
    return "" if node.module_stack is None else repr(node.module_stack)


def first_layer_key(value: str) -> str | None:
    """
    Finds the first layer/block path fragment in a string.

    The helper normalizes `/` and `_` into `.` so names like
    `model_layers_12_attn` can be treated similarly to
    `model.layers.12.attn`. It then looks for common containers such as
    `layers`, `layer`, `blocks`, or `h` followed by a numeric index.
    """
    parts = value.replace("/", ".").replace("_", ".").split(".")

    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "layer", "h", "blocks", "block"} and parts[index + 1].isdigit():
            return f"{part}.{parts[index + 1]}"

    return None



############################################ Extra Matchers!!!!! ############################################
def match_extra_constraints(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    """
    Dispatches every structured FusionGraph constraint to its registered matcher.

    `fusion.constraints` is a dict keyed by constraint name. This function looks
    up each name in `EXTRA_MATCHERS`, normalizes each raw spec into one or more
    spec dictionaries, and requires every matcher invocation to return True.
    Unknown constraint names or malformed specs fail closed.
    """
    for constraint_name, raw_spec in fusion.constraints.items():
        matcher = EXTRA_MATCHERS.get(constraint_name)

        if matcher is None:
            return False

        specs = normalize_constraint_specs(raw_spec)

        if specs is None:
            return False

        for spec in specs:
            if not matcher(source, graph, fusion, bindings, spec):
                return False

    return True


def normalize_constraint_specs(raw_spec: Any) -> tuple[dict[str, Any], ...] | None:
    """
    Normalizes one constraint value into a tuple of spec dictionaries.

    A constraint key can map to a single dict or a list/tuple of dicts. The
    latter lets one matcher type be applied multiple times without duplicating
    registry keys. Non-dict specs return None so the caller can fail safely.
    """
    if isinstance(raw_spec, dict):
        return (raw_spec,)

    if isinstance(raw_spec, (list, tuple)) and all(isinstance(spec, dict) for spec in raw_spec):
        return tuple(raw_spec)

    return None


def match_linear_weight_layout(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Verifies that a linear/matmul RHS is a compatible model weight layout.

    The matcher resolves the fusion's input, weight, and output nodes by role.
    It then classifies the weight path as direct or transposed and checks that
    dimensions line up with the matmul output. This is what prevents a random
    activation matrix from being treated as a Cactus linear weight.
    """
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["input_role"])
    weight_node = match_utils.get_first_input_by_role(fusion, bindings, spec["weight_role"])
    output_node = bindings.get(spec["output_node"])

    if input_node is None or weight_node is None or output_node is None:
        return bool(spec.get("allow_missing", False))

    layout = classify_linear_rhs(weight_node)

    if layout is None:
        return False

    layout_kind, source_weight = layout
    allowed_layouts = spec.get("allowed_layouts", ("direct", "transposed"))

    if layout_kind not in allowed_layouts:
        return False

    return linear_shapes_match(input_node, weight_node, output_node, source_weight, layout_kind, spec)


def match_rope_tables_compatible(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Verifies that RoPE cos/sin inputs look like compatible rotation tables.

    It gets the matched x/cos/sin values by role, checks cos/sin shape and dtype
    compatibility, optionally proves they came from matching angle sources, and
    then checks that both tables can broadcast or half-dim match the RoPE input.
    """
    x_node = match_utils.get_first_input_by_role(fusion, bindings, spec["x_role"])
    cos_node = match_utils.get_first_input_by_role(fusion, bindings, spec["cos_role"])
    sin_node = match_utils.get_first_input_by_role(fusion, bindings, spec["sin_role"])

    if cos_node is None or sin_node is None:
        return bool(spec.get("allow_missing", False))

    cos_shape = match_utils.get_tensor_shape(cos_node)
    sin_shape = match_utils.get_tensor_shape(sin_node)

    if not compatible_shapes(cos_shape, sin_shape, spec):
        return False

    if not compatible_dtypes(cos_node, sin_node, spec):
        return False

    if not compatible_trig_sources(cos_node, sin_node, spec):
        return False

    if x_node is not None and not rope_tables_match_input(x_node, cos_shape, sin_shape, spec):
        return False

    return True


def match_lstm_gate_structure(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Verifies that an LSTM gate node has real four-gate structure evidence.

    The matcher first checks that the combined gate tensor's final dimension is
    divisible by the expected gate count. By default it then requires an
    explicit downstream split/chunk/unbind into that many gates, so LSTM fusion
    fails closed when the graph only vaguely resembles an LSTM.
    """
    gate_node = bindings.get(spec["gate_node"])

    if gate_node is None:
        return bool(spec.get("allow_missing", False))

    gate_count = int(spec.get("gate_count", 4))

    if not gate_shape_matches(gate_node, gate_count, spec):
        return False

    if spec.get("require_explicit_gate_split", True):
        return has_explicit_gate_split(gate_node, gate_count, int(spec.get("max_depth", 4)))

    return True


def match_moe_expert_branch_routing(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Verifies that enough matched expert branches are driven by router top-k.

    This locates the repeated expert-branch pattern, finds or discovers a top-k
    node connected to the MoE combine output, then collects expert branch
    bindings whose routing-weight input is reachable from that routing source.
    The match succeeds only if the number of routed branches meets the expected
    count from `topk.k`, an explicit spec override, or the subgraph minimum.
    """
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
    """
    Verifies that routed expert outputs are actually combined into MoE output.

    This reuses the routed expert-branch filtering, then checks that each
    branch's weighted expert output has a forward path to the matched combine
    node. That proves both sides of MoE semantics: routing weights are used on
    expert outputs, and those weighted outputs feed the final combine op.
    """
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
    """
    Verifies the grouped-mm MoE structure exported by LFM-style MoE models.

    This matcher handles the compact representation where expert branches are
    not expanded into separate per-expert matmuls. The graph edges prove the
    local dataflow; this semantic check verifies top-k routing, expert-count
    consistency, shared grouped-mm offsets, valid gate/up splitting, and final
    top-k combine shape.
    """
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

    if not match_utils.values_equal(get_attr(offsets_node, "dim", "arg_1"), 0):
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

    if not match_utils.values_equal(get_attr(combine_node, "dim", "arg_1"), 1):
        return False

    return True


def match_node_attr_equals(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Checks that one matched node has a specific normalized attr value.

    The spec names a synthetic node, an attr key, and the expected value. The
    function resolves the real node from bindings, reads its normalized attrs,
    and compares with `values_equal`. Missing attrs fail unless
    `allow_missing=True` is present in the spec.
    """
    node = bindings.get(spec["node"])

    if node is None:
        return False

    actual = node.attrs.get(spec["attr"], match_utils.MISSING)

    if actual is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(actual, spec["value"])


def match_node_attrs_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Checks that attrs on two matched nodes are equal to each other.

    This is useful for pattern checks like layernorm, where mean and variance
    should reduce over the same dimension. The spec identifies left/right
    synthetic nodes and attr names; both attrs must exist and compare equal
    unless missing attrs are explicitly allowed.
    """
    left_node = bindings.get(spec["left_node"])
    right_node = bindings.get(spec["right_node"])

    if left_node is None or right_node is None:
        return False

    left_value = left_node.attrs.get(spec["left_attr"], match_utils.MISSING)
    right_value = right_node.attrs.get(spec["right_attr"], match_utils.MISSING)

    if left_value is match_utils.MISSING or right_value is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(left_value, right_value)


def match_input_value_kind(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Checks the semantic kind of an external input to a fusion.

    The fusion graph declares inputs by role, such as `bias` or `weight`. This
    matcher resolves the first real node for that role and verifies its
    `value_kind` is in the allowed set, preventing activations from being
    mistaken for parameters/buffers.
    """
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])

    if input_node is None:
        return bool(spec.get("allow_missing", False))

    return input_node.value_kind in spec["allowed_value_kinds"]


def match_parent_tensor_dim_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Compares one dimension from two parent tensors of matched nodes.

    The spec names the left/right synthetic nodes, which parent index to inspect
    on each real node, and which shape dimension to compare. It is used for
    checks such as attention query/key head-dim compatibility. Missing parents
    or shapes follow the spec's `allow_missing` behavior.
    """
    left_parent = get_bound_parent(bindings, spec["left_node"], spec["left_parent_index"])
    right_parent = get_bound_parent(bindings, spec["right_node"], spec["right_parent_index"])

    if left_parent is None or right_parent is None:
        return bool(spec.get("allow_missing", False))

    left_dim = get_node_dim(left_parent, spec["left_dim"])
    right_dim = get_node_dim(right_parent, spec["right_dim"])

    if left_dim is match_utils.MISSING or right_dim is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(left_dim, right_dim)


def match_input_broadcastable_to_node(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Checks that a declared fusion input can broadcast to a target node shape.

    This is mainly for mask-like inputs. It resolves an input by role, resolves
    the target synthetic node, reads both output shapes, and applies standard
    broadcasting. Missing nodes or shapes are controlled by `allow_missing`.
    """
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])
    target_node = bindings.get(spec["node"])

    if input_node is None or target_node is None:
        return bool(spec.get("allow_missing", False))

    input_shape = match_utils.get_tensor_shape(input_node)
    target_shape = match_utils.get_tensor_shape(target_node)

    if not input_shape or not target_shape:
        return bool(spec.get("allow_missing", False))

    return shapes_are_broadcastable(input_shape, target_shape)


def match_same_layer(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """
    Checks that multiple matched nodes appear to belong to the same layer.

    The spec provides synthetic node names. For each bound real node, we derive
    a layer key from module stack/name/target text. The check passes only when
    every derived key is identical, which helps avoid mixing cache/key/value
    nodes from different decoder layers.
    """
    layer_keys = []

    for node_name in spec["nodes"]:
        node = bindings.get(node_name)

        if node is None:
            return False

        layer_key = get_layer_key(node)

        if layer_key is None:
            return bool(spec.get("allow_missing", False))

        layer_keys.append(layer_key)

    return len(set(layer_keys)) == 1




EXTRA_MATCHERS: dict[str, ExtraMatcher] = {
    "linear_weight_layout": match_linear_weight_layout,
    "rope_tables_compatible": match_rope_tables_compatible,
    "lstm_gate_structure": match_lstm_gate_structure,
    "moe_expert_branch_routing": match_moe_expert_branch_routing,
    "moe_routing_weights_combine": match_moe_routing_weights_combine,
    "grouped_moe_structure": match_grouped_moe_structure,
    "node_attr_equals": match_node_attr_equals,
    "node_attrs_equal": match_node_attrs_equal,
    "input_value_kind": match_input_value_kind,
    "parent_tensor_dim_equal": match_parent_tensor_dim_equal,
    "input_broadcastable_to_node": match_input_broadcastable_to_node,
    "same_layer": match_same_layer,
}
