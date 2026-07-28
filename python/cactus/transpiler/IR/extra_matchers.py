from typing import Any, Callable

from . import models, match_utils
from ..Fusions import models as FModels


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
    node = bindings.get(node_name)

    if node is None:
        return None

    return match_utils.get_parent(node, parent_index)


def get_node_dim(node: models.Node, dim_index: int) -> Any:
    shape = match_utils.get_tensor_shape(node)

    if not shape:
        return match_utils.MISSING

    if dim_index < 0:
        dim_index += len(shape)

    if dim_index < 0 or dim_index >= len(shape):
        return match_utils.MISSING

    return shape[dim_index]


def classify_linear_rhs(node: models.Node) -> tuple[str, models.Node] | None:
    return classify_linear_rhs_(node, set())


def classify_linear_rhs_(node: models.Node, seen: set[int]) -> tuple[str, models.Node] | None:
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
    if not left_shape or not right_shape:
        return bool(spec.get("allow_missing", False))

    return (
        match_utils.values_equal(left_shape, right_shape)
        or shapes_are_broadcastable(left_shape, right_shape)
        or shapes_are_broadcastable(right_shape, left_shape)
    )


def compatible_dtypes(left_node: models.Node, right_node: models.Node, spec: dict[str, Any]) -> bool:
    if not spec.get("require_same_dtype", True):
        return True

    left_dtype = get_node_dtype(left_node)
    right_dtype = get_node_dtype(right_node)

    if left_dtype is match_utils.MISSING or right_dtype is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(left_dtype, right_dtype)


def compatible_trig_sources(cos_node: models.Node, sin_node: models.Node, spec: dict[str, Any]) -> bool:
    cos_sources = collect_trig_angle_sources(cos_node, COS_TARGETS, int(spec.get("max_trig_depth", 8)))
    sin_sources = collect_trig_angle_sources(sin_node, SIN_TARGETS, int(spec.get("max_trig_depth", 8)))

    if cos_sources or sin_sources:
        return cos_sources == sin_sources

    return not spec.get("require_trig_sources", False)


def collect_trig_angle_sources(node: models.Node, trig_targets: set[str], max_depth: int) -> frozenset[str]:
    sources: set[str] = set()
    seen: set[int] = set()

    def visit(current: models.Node, depth: int) -> None:
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
    x_shape = match_utils.get_tensor_shape(x_node)

    if not x_shape:
        return bool(spec.get("allow_missing", False))

    return rope_table_shape_matches_x(cos_shape, x_shape) and rope_table_shape_matches_x(sin_shape, x_shape)


def rope_table_shape_matches_x(table_shape: list[Any], x_shape: list[Any]) -> bool:
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
    gate_shape = match_utils.get_tensor_shape(gate_node)

    if not gate_shape:
        return bool(spec.get("allow_missing", False))

    hidden_dim = gate_shape[-1]

    if not isinstance(hidden_dim, int):
        return bool(spec.get("allow_symbolic", True))

    return hidden_dim % gate_count == 0


def has_explicit_gate_split(gate_node: models.Node, gate_count: int, max_depth: int) -> bool:
    for node in walk_children(gate_node, max_depth):
        if node.target not in SPLIT_TARGETS:
            continue

        if split_node_has_gate_count(node, gate_count):
            return True

    return False


def split_node_has_gate_count(node: models.Node, gate_count: int) -> bool:
    if node.target == "aten.unbind.int":
        return len(node.children) == gate_count

    if node.target == "aten.chunk.default":
        return match_utils.values_equal(node.attrs.get("arg_1", match_utils.MISSING), gate_count)

    split_sizes = node.attrs.get("arg_1", match_utils.MISSING)

    if isinstance(split_sizes, (list, tuple)):
        return len(split_sizes) == gate_count

    return len(node.children) == gate_count


def walk_children(node: models.Node, max_depth: int) -> tuple[models.Node, ...]:
    found: list[models.Node] = []
    seen: set[int] = set()

    def visit(current: models.Node, depth: int) -> None:
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
    if not isinstance(node.tensor_output_meta, dict):
        return match_utils.MISSING

    return node.tensor_output_meta.get("dtype", match_utils.MISSING)


def normalize_dim(dim: Any, rank: int) -> Any:
    if not isinstance(dim, int):
        return dim

    return dim + rank if dim < 0 else dim


def shapes_are_broadcastable(left_shape: list[Any], right_shape: list[Any]) -> bool:
    for left_dim, right_dim in zip(reversed(left_shape), reversed(right_shape)):
        if match_utils.values_equal(left_dim, 1) or match_utils.values_equal(right_dim, 1):
            continue

        if not match_utils.values_equal(left_dim, right_dim):
            return False

    return True


def get_layer_key(node: models.Node) -> str | None:
    for value in (module_stack_text(node), node.name, node.target):
        layer_key = first_layer_key(value)

        if layer_key is not None:
            return layer_key

    return None


def module_stack_text(node: models.Node) -> str:
    return "" if node.module_stack is None else repr(node.module_stack)


def first_layer_key(value: str) -> str | None:
    parts = value.replace("/", ".").replace("_", ".").split(".")

    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "layer", "h", "blocks", "block"} and parts[index + 1].isdigit():
            return f"{part}.{parts[index + 1]}"

    return None



############################################ Extra Matchers!!!!! ############################################
def match_extra_constraints(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
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
    if isinstance(raw_spec, dict):
        return (raw_spec,)

    if isinstance(raw_spec, (list, tuple)) and all(isinstance(spec, dict) for spec in raw_spec):
        return tuple(raw_spec)

    return None


def match_note(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    return True


def match_linear_weight_layout(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
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
    gate_node = bindings.get(spec["gate_node"])

    if gate_node is None:
        return bool(spec.get("allow_missing", False))

    gate_count = int(spec.get("gate_count", 4))

    if not gate_shape_matches(gate_node, gate_count, spec):
        return False

    if spec.get("require_explicit_gate_split", True):
        return has_explicit_gate_split(gate_node, gate_count, int(spec.get("max_depth", 4)))

    return True


def match_node_attr_equals(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
    node = bindings.get(spec["node"])

    if node is None:
        return False

    actual = node.attrs.get(spec["attr"], match_utils.MISSING)

    if actual is match_utils.MISSING:
        return bool(spec.get("allow_missing", False))

    return match_utils.values_equal(actual, spec["value"])


def match_node_attrs_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
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
    input_node = match_utils.get_first_input_by_role(fusion, bindings, spec["role"])

    if input_node is None:
        return bool(spec.get("allow_missing", False))

    return input_node.value_kind in spec["allowed_value_kinds"]


def match_parent_tensor_dim_equal(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node], spec: dict[str, Any]) -> bool:
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
    "note": match_note,
    "node_attr_equals": match_node_attr_equals,
    "node_attrs_equal": match_node_attrs_equal,
    "input_value_kind": match_input_value_kind,
    "parent_tensor_dim_equal": match_parent_tensor_dim_equal,
    "input_broadcastable_to_node": match_input_broadcastable_to_node,
    "same_layer": match_same_layer,
}
