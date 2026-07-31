from __future__ import annotations

from typing import Any

from . import models, match_utils
from .extra_matcher_common import *
from ..Fusions import models as FModels


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
