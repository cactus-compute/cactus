from __future__ import annotations

from typing import Any

from . import models, match_utils
from .extra_matcher_common import *
from ..Fusions import models as FModels

def classify_linear_rhs(node: models.Node) -> tuple[str, models.Node] | None:
    """Classifies the right-hand side of a linear/matmul as direct or transposed."""
    return classify_linear_rhs_(node, set())

def classify_linear_rhs_(node: models.Node, seen: set[int]) -> tuple[str, models.Node] | None:
    """Walks backward through simple RHS wrapper ops to find the source weight."""
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
    """Checks whether a transpose-like op really swaps a 2D linear weight."""
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
    """Verifies the dimensions around a linear/matmul are internally consistent."""
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
    """Checks whether two tensor shapes are equal or broadcast-compatible."""
    if not left_shape or not right_shape:
        return bool(spec.get("allow_missing", False))
    return (
        match_utils.values_equal(left_shape, right_shape)
        or shapes_are_broadcastable(left_shape, right_shape)
        or shapes_are_broadcastable(right_shape, left_shape)
    )

def compatible_dtypes(left_node: models.Node, right_node: models.Node, spec: dict[str, Any]) -> bool:
    """Checks whether two nodes have matching output dtypes when required."""
    if not spec.get("require_same_dtype", True):
        return True
    left_dtype = get_node_dtype(left_node)
    right_dtype = get_node_dtype(right_node)
    if left_dtype is match_utils.MISSING or right_dtype is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))
    return match_utils.values_equal(left_dtype, right_dtype)

def compatible_trig_sources(cos_node: models.Node, sin_node: models.Node, spec: dict[str, Any]) -> bool:
    """Checks whether RoPE cos and sin tables appear to come from the same angles."""
    cos_sources = collect_trig_angle_sources(cos_node, COS_TARGETS, int(spec.get("max_trig_depth", 8)))
    sin_sources = collect_trig_angle_sources(sin_node, SIN_TARGETS, int(spec.get("max_trig_depth", 8)))
    if cos_sources or sin_sources:
        return cos_sources == sin_sources
    return not spec.get("require_trig_sources", False)

def collect_trig_angle_sources(node: models.Node, trig_targets: set[str], max_depth: int) -> frozenset[str]:
    """Finds the angle-producing parent nodes for visible trig ops upstream."""
    sources: set[str] = set()
    seen: set[int] = set()
    def visit(current: models.Node, depth: int) -> None:
        """Recursively walks parents while avoiding cycles and depth blowups."""
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
    """Checks whether RoPE cos/sin table shapes can apply to the rotated input."""
    x_shape = match_utils.get_tensor_shape(x_node)
    if not x_shape:
        return bool(spec.get("allow_missing", False))
    return rope_table_shape_matches_x(cos_shape, x_shape) and rope_table_shape_matches_x(sin_shape, x_shape)

def rope_table_shape_matches_x(table_shape: list[Any], x_shape: list[Any]) -> bool:
    """Checks one RoPE table shape against the activation being rotated."""
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
    """Checks whether an LSTM combined-gate tensor can hold `gate_count` gates."""
    gate_shape = match_utils.get_tensor_shape(gate_node)
    if not gate_shape:
        return bool(spec.get("allow_missing", False))
    hidden_dim = gate_shape[-1]
    if not isinstance(hidden_dim, int):
        return bool(spec.get("allow_symbolic", True))
    return hidden_dim % gate_count == 0

def has_explicit_gate_split(gate_node: models.Node, gate_count: int, max_depth: int) -> bool:
    """Looks for a visible split/chunk/unbind that separates combined LSTM gates."""
    for node in walk_children(gate_node, max_depth):
        if node.target not in SPLIT_TARGETS:
            continue
        if split_node_has_gate_count(node, gate_count):
            return True
    return False

def split_node_has_gate_count(node: models.Node, gate_count: int) -> bool:
    """Determines whether a split-like node creates the expected number of gates."""
    if node.target == "aten.unbind.int":
        return len(node.children) == gate_count
    if node.target == "aten.chunk.default":
        return match_utils.values_equal(node.attrs.get("arg_1", match_utils.MISSING), gate_count)
    split_sizes = node.attrs.get("arg_1", match_utils.MISSING)
    if isinstance(split_sizes, (list, tuple)):
        return len(split_sizes) == gate_count
    return len(node.children) == gate_count

def match_linear_weight_layout(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    """Verifies that a linear/matmul RHS is a compatible model weight layout."""
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
    """Verifies that RoPE cos/sin inputs look like compatible rotation tables."""
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
    """Verifies that an LSTM gate node has real four-gate structure evidence."""
    gate_node = bindings.get(spec["gate_node"])
    if gate_node is None:
        return bool(spec.get("allow_missing", False))
    gate_count = int(spec.get("gate_count", 4))
    if not gate_shape_matches(gate_node, gate_count, spec):
        return False
    if spec.get("require_explicit_gate_split", True):
        return has_explicit_gate_split(gate_node, gate_count, int(spec.get("max_depth", 4)))
    return True
