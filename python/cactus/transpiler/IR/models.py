from dataclasses import dataclass, field
from typing import Any

from . import constants

from ..Converter import models as CModels
from ..Fusions import models as FModels

@dataclass(slots=True)
class CacheAnnotation:
    kind: str
    role: str
    tensor_index: int | None = None
    layer_index: int | None = None
    shape: tuple[Any, ...] = ()
    layout: str | None = None
    sequence_length: int | None = None
    window_size: int | None = None
    hidden_dim: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    source: str = "inferred"

@dataclass(slots=True, frozen=True)
class CacheConcatMatch:
    """A structural boundary between opaque cache state and a new tensor."""

    concat: "Node"
    state: "Node"
    new_value: "Node"
    state_wrappers: tuple["Node", ...] = ()

@dataclass(slots=True)
class Node:
    index: int
    name: str
    node_type: str
    target: str
    args: Any
    kwargs: Any
    users: tuple[str, ...]
    tensor_output_meta: Any | None
    module_stack: Any | None
    value_kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    ir_metadata: dict[str, Any] = field(default_factory=dict)
    cache: CacheAnnotation | None = None
    parents: tuple["Node", ...] = field(default_factory=tuple, repr=False)
    children: tuple["Node", ...] = field(default_factory=tuple, repr=False)

    @classmethod
    def from_layer(cls, record: CModels.LayerRecord) -> "Node":
        return generate_node(record)

    @property
    def is_placeholder(self) -> bool:
        return self.node_type == "placeholder"

    @property
    def is_operation(self) -> bool:
        return self.node_type == "call_function"

    @property
    def is_output(self) -> bool:
        return self.node_type == "output"

@dataclass(slots=True)
class Graph:
    source: Node | None
    sources: tuple[Node, ...]
    outputs: tuple[Node, ...]
    nodes: tuple[Node, ...]
    nodes_map: dict[str, Node]
    model_name: str = ""
    task: str = ""
    graph_signature: str = ""
    range_constants: str = ""
    input_specs: tuple[CModels.GraphSpecRecord, ...] = field(default_factory=tuple)
    output_specs: tuple[CModels.GraphSpecRecord, ...] = field(default_factory=tuple)
    cache_annotations: tuple[CacheAnnotation, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)
    fusions: list["FusionResult"] = field(default_factory=list)

    @classmethod
    def from_map(cls, layer_map: CModels.LayerMap) -> "Graph":
        return generate_graph(cls, layer_map)

    def apply_fusions(self, fusion_results: list["FusionResult"] | tuple["FusionResult", ...]) -> "Graph":
        return apply_fusions_to_graph(self, tuple(fusion_results))

    def remove_noop_nodes(self) -> "Graph":
        return remove_noop_nodes_from_graph(self)

    def collapse_transpose_chains(self) -> "Graph":
        return collapse_transpose_chains_from_graph(self)

    def fuse_logits_softcap(self) -> "Graph":
        return fuse_logits_softcap_from_graph(self)

    def to_layer_map(self) -> CModels.LayerMap:
        return graph_to_layer_map(self)

@dataclass(slots=True)
class FusionResult:
    fusion: FModels.FusionDefinition
    source: Node
    matched_nodes: tuple[Node, ...] = field(default_factory=tuple)
    bindings: dict[str, Node] = field(default_factory=dict)
    external_inputs: tuple[Node, ...] = field(default_factory=tuple)
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_match(
        cls,
        fusion: FModels.FusionDefinition,
        source: Node,
        *,
        matched_nodes: tuple[Node, ...] = (),
        bindings: dict[str, Node] | None = None,
        external_inputs: tuple[Node, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> "FusionResult":
        return cls(
            fusion=fusion,
            source=source,
            matched_nodes=matched_nodes or (source,),
            bindings=bindings or {},
            external_inputs=external_inputs,
            attrs=attrs or {},
        )

    @property
    def fusion_name(self) -> str:
        return self.fusion.name

    @property
    def target(self) -> str:
        return self.fusion.target

    @property
    def cactus_op(self) -> str | None:
        return self.fusion.cactus_op

    @property
    def consumed_node_names(self) -> frozenset[str]:
        return frozenset(node.name for node in self.matched_nodes)

CACHE_LAYOUT_TARGETS = frozenset({
    "cactus.view", "aten.view.default", "aten.reshape.default",
    "cactus.transpose", "aten.transpose.int", "aten.permute.default",
    "cactus.precision_cast", "aten._to_copy.default",
    "aten.clone.default", "aten.contiguous.default",
})

def find_cache_concat_ancestor(
    node: Node,
    role: str,
    *,
    max_depth: int = 18,
    cache_wrapper_depth: int = 8,
) -> CacheConcatMatch | None:
    """Match a typed cache/new-value concatenation through layout wrappers."""
    current = node
    for _ in range(max_depth):
        if current.target in {"cactus.cat", "aten.cat.default"} and len(current.parents) >= 2:
            for parent_index, parent in enumerate(current.parents[:2]):
                state, wrappers = find_cache_state_ancestor(
                    parent, role, max_depth=cache_wrapper_depth,
                )
                if state is not None:
                    return CacheConcatMatch(
                        concat=current,
                        state=state,
                        new_value=current.parents[1 - parent_index],
                        state_wrappers=wrappers,
                    )
        if len(current.parents) != 1:
            return None
        current = current.parents[0]
    return None

def find_cache_state_ancestor(
    node: Node,
    role: str,
    *,
    max_depth: int = 8,
) -> tuple[Node | None, tuple[Node, ...]]:
    current = node
    wrappers: list[Node] = []
    for _ in range(max_depth + 1):
        annotation = current.cache
        if annotation is not None and annotation.kind == FModels.CacheKind.KV and annotation.role == role:
            return current, tuple(wrappers)
        if current.target not in CACHE_LAYOUT_TARGETS or len(current.parents) != 1:
            break
        wrappers.append(current)
        current = current.parents[0]
    return None, ()

def extract_node_refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict) and set(value.keys()) == {"node"} and isinstance(value["node"], str):
        return (value["node"],)

    if isinstance(value, (list, tuple)):
        refs: list[str] = []

        for item in value:
            refs.extend(extract_node_refs(item))

        return tuple(refs)

    if isinstance(value, dict):
        refs: list[str] = []

        for item in value.values():
            refs.extend(extract_node_refs(item))

        return tuple(refs)

    return ()

def infer_value_kind(record: CModels.LayerRecord) -> str:
    if record.node_type == "output":
        return FModels.ValueKind.OUTPUT
    if record.node_type != "placeholder":
        return FModels.ValueKind.ACTIVATION

    target = record.target

    if target.startswith("p_"):
        return FModels.ValueKind.PARAMETER
    if target.startswith("b_"):
        return FModels.ValueKind.BUFFER
    if target.startswith("c_") or "lifted_tensor" in target or "lifted" in target:
        return FModels.ValueKind.LIFTED_CONSTANT
    if "past_key_values" in target or target in {"cache_position"}:
        return FModels.ValueKind.CACHE_INPUT

    return FModels.ValueKind.USER_INPUT

def extract_attrs(record: CModels.LayerRecord) -> dict[str, Any]:
    attrs: dict[str, Any] = {}

    if isinstance(record.kwargs, dict):
        attrs.update(record.kwargs)

    args = record.args if isinstance(record.args, list) else []
    positional_attrs = constants.POSITIONAL_ATTRS_MAP.get(record.target, {})

    for index, attr_name in positional_attrs.items():
        if index >= len(args):
            continue

        value = args[index]
        if extract_node_refs(value):
            continue

        attrs.setdefault(attr_name, value)

    for index, value in enumerate(args):
        if extract_node_refs(value):
            continue

        attrs.setdefault(f"arg_{index}", value)

    return attrs

def generate_node(record: CModels.LayerRecord) -> Node:
    return Node(
        index=record.index,
        name=record.name,
        node_type=record.node_type,
        target=record.target,
        args=record.args,
        kwargs=record.kwargs,
        users=tuple(record.users),
        tensor_output_meta=record.tensor_output_meta,
        module_stack=record.module_stack,
        value_kind=infer_value_kind(record),
        attrs=extract_attrs(record),
        ir_metadata=dict(record.ir_metadata),
        cache=cache_annotation_from_metadata(record.ir_metadata),
    )

def generate_graph(cls: type[Graph], layer_map: CModels.LayerMap) -> Graph:
    nodes = tuple(Node.from_layer(record) for record in layer_map.nodes)
    nodes_map = {node.name: node for node in nodes}
    parent_lists: dict[str, list[Node]] = {node.name: [] for node in nodes}
    child_lists: dict[str, list[Node]] = {node.name: [] for node in nodes}

    for record in layer_map.nodes:
        node = nodes_map[record.name]
        parent_names = extract_node_refs((record.args, record.kwargs))

        for parent_name in parent_names:
            if parent_name not in nodes_map:
                raise ValueError(f"{record.name} references missing parent node {parent_name}")

            parent = nodes_map[parent_name]
            parent_lists[node.name].append(parent)

            if node not in child_lists[parent.name]:
                child_lists[parent.name].append(node)

    for node in nodes:
        node.parents = tuple(parent_lists[node.name])
        node.children = tuple(child_lists[node.name])

    annotate_cache_nodes(nodes, layer_map.model_name)
    sources = tuple(node for node in nodes if not node.parents)
    outputs = tuple(node for node in nodes if node.is_output)

    if not outputs:
        outputs = tuple(node for node in nodes if not node.children)

    return cls(
        source=sources[0] if sources else None,
        sources=sources,
        outputs=outputs,
        nodes=nodes,
        nodes_map=nodes_map,
        model_name=layer_map.model_name,
        task=layer_map.task,
        graph_signature=layer_map.graph_signature,
        range_constants=layer_map.range_constants,
        input_specs=tuple(layer_map.input_specs),
        output_specs=tuple(layer_map.output_specs),
        cache_annotations=cache_annotations_from_nodes(nodes),
        metadata={},
    )

def apply_fusions_to_graph(graph: Graph, fusion_results: tuple[FusionResult, ...]) -> Graph:
    if not fusion_results:
        return graph

    validate_non_overlapping_fusions(fusion_results)

    fused_nodes, replacement_names = build_fused_nodes(graph, fusion_results)
    consumed_names = expanded_consumed_names(graph, replacement_names)
    kept_nodes = tuple(clone_rewritten_node(node, replacement_names) for node in graph.nodes if node.name not in consumed_names)
    rewritten_fused_nodes = tuple(clone_rewritten_node(node, replacement_names) for node in fused_nodes)
    rebuilt_graph = rebuild_graph((*kept_nodes, *rewritten_fused_nodes), graph, fusion_results)
    return prune_dead_nodes(rebuilt_graph)

def validate_non_overlapping_fusions(fusion_results: tuple[FusionResult, ...]) -> None:
    consumed_names: set[str] = set()

    for result in fusion_results:
        overlap = consumed_names.intersection(result.consumed_node_names)

        if overlap:
            overlap_list = ", ".join(sorted(overlap))
            raise ValueError(f"Fusion {result.fusion_name} overlaps an earlier fusion at: {overlap_list}")

        consumed_names.update(result.consumed_node_names)

def build_fused_nodes(graph: Graph, fusion_results: tuple[FusionResult, ...]) -> tuple[tuple[Node, ...], dict[str, str]]:
    fused_nodes: list[Node] = []
    replacement_names: dict[str, str] = {}
    used_names = set(graph.nodes_map)

    for count, result in enumerate(fusion_results):
        fused_name = unique_fused_name(result, count, used_names)
        used_names.add(fused_name)
        fused_node = fused_node_from_result(result, fused_name)
        fused_nodes.append(fused_node)

        for consumed_name in result.consumed_node_names:
            replacement_names[consumed_name] = fused_name

    return tuple(fused_nodes), replacement_names

def unique_fused_name(result: FusionResult, count: int, used_names: set[str]) -> str:
    base = sanitize_node_name(result.cactus_op or result.fusion_name)
    candidate = f"{base}_{count}"
    suffix = count

    while candidate in used_names:
        suffix += 1
        candidate = f"{base}_{suffix}"

    return candidate

def sanitize_node_name(name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in name).strip("_") or "fused"

def fused_node_from_result(result: FusionResult, fused_name: str) -> Node:
    attrs = normalized_fusion_attrs(result)
    cache = cache_annotation_from_fusion_result(result)
    ir_metadata = fusion_result_metadata(result, cache)

    return Node(
        index=result.source.index,
        name=fused_name,
        node_type="call_function",
        target=result.target,
        args=[{"node": node.name} for node in result.external_inputs],
        kwargs=attrs,
        users=(),
        tensor_output_meta=result.source.tensor_output_meta,
        module_stack=result.source.module_stack,
        value_kind=FModels.ValueKind.CACHE_OUTPUT if cache is not None else FModels.ValueKind.ACTIVATION,
        attrs=attrs,
        ir_metadata=ir_metadata,
        cache=cache,
    )

def normalized_fusion_attrs(result: FusionResult) -> dict[str, Any]:
    attrs = dict(result.attrs)

    if result.target in {"cactus.attention", "cactus.attention_cached"} and int(attrs.get("window_size", 0) or 0) == 0:
        layer_index = gemma_language_layer_index(result)

        if layer_index is not None:
            attrs["window_size"] = 0 if layer_index % 5 == 4 else 512

    if result.target in {"cactus.attention", "cactus.attention_cached"} and "cache_window_size" not in attrs:
        cache_input = first_cache_input_annotation(result)

        if cache_input is not None and cache_input.window_size is not None:
            attrs["cache_window_size"] = cache_input.window_size

    return attrs

def gemma_language_layer_index(result: FusionResult) -> int | None:
    for node in (result.source, *result.matched_nodes, *result.external_inputs):
        text = f"{node.name} {node.target} {node.module_stack!r}"

        if "gemma" not in text.lower() or "language_model" not in text:
            continue

        layer_index = layer_index_from_text(text)

        if layer_index is not None:
            return layer_index

    return None

def expanded_consumed_names(graph: Graph, replacement_names: dict[str, str]) -> frozenset[str]:
    consumed_names = set(replacement_names)
    changed = True

    while changed:
        changed = False

        for node in graph.nodes:
            if node.name in consumed_names:
                continue

            if node.target != "aten._assert_tensor_metadata.default":
                continue

            parent_replacements = [replacement_names[parent.name] for parent in node.parents if parent.name in replacement_names]

            if not parent_replacements:
                continue

            replacement_names[node.name] = parent_replacements[0]
            consumed_names.add(node.name)
            changed = True

    return frozenset(consumed_names)

def clone_rewritten_node(node: Node, replacement_names: dict[str, str]) -> Node:
    return Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=rewrite_node_refs(node.args, replacement_names),
        kwargs=rewrite_node_refs(node.kwargs, replacement_names),
        users=(),
        tensor_output_meta=node.tensor_output_meta,
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=dict(node.attrs),
        ir_metadata=dict(node.ir_metadata),
        cache=node.cache,
    )

def rewrite_node_refs(value: Any, replacement_names: dict[str, str]) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"node"} and isinstance(value["node"], str):
        return {"node": replacement_names.get(value["node"], value["node"])}

    if isinstance(value, list):
        return [rewrite_node_refs(item, replacement_names) for item in value]

    if isinstance(value, tuple):
        return tuple(rewrite_node_refs(item, replacement_names) for item in value)

    if isinstance(value, dict):
        return {key: rewrite_node_refs(item, replacement_names) for key, item in value.items()}

    return value

def remove_noop_nodes_from_graph(graph: Graph) -> Graph:
    replacement_names: dict[str, str] = {}

    for node in graph.nodes:
        replacement = noop_replacement_name(node)

        if replacement is not None:
            replacement_names[node.name] = replacement

    if not replacement_names:
        return graph

    replacement_names = resolve_replacement_names(replacement_names)
    kept_nodes = tuple(clone_rewritten_node(node, replacement_names) for node in graph.nodes if node.name not in replacement_names)
    return prune_dead_nodes(rebuild_graph(kept_nodes, graph, tuple(graph.fusions)))

def collapse_transpose_chains_from_graph(graph: Graph) -> Graph:
    """Compose adjacent full-rank permutations and discard dead intermediates."""
    replacements: dict[str, str] = {}
    rewritten: dict[str, Node] = {}

    for node in graph.nodes:
        permutation = transpose_permutation(node)

        if permutation is None or len(node.parents) != 1:
            continue

        source = node.parents[0]
        combined = permutation
        collapsed = False

        while len(source.parents) == 1:
            parent_permutation = transpose_permutation(source)

            if parent_permutation is None or len(parent_permutation) != len(combined):
                break

            combined = tuple(parent_permutation[index] for index in combined)
            source = source.parents[0]
            collapsed = True

        if not collapsed:
            continue

        if combined == tuple(range(len(combined))):
            replacements[node.name] = source.name
            continue

        rewritten[node.name] = clone_composed_transpose(node, source.name, combined)

    if not replacements and not rewritten:
        return graph

    replacements = resolve_replacement_names(replacements)
    nodes = tuple(
        clone_rewritten_node(rewritten.get(node.name, node), replacements)
        for node in graph.nodes
        if node.name not in replacements
    )
    return prune_dead_nodes(rebuild_graph(nodes, graph, tuple(graph.fusions)))

def fuse_logits_softcap_from_graph(graph: Graph) -> Graph:
    """Fuse a CQ logits projection followed by cap*tanh(logits/cap)."""
    consumed: set[str] = set()
    replacements: dict[str, Node] = {}
    layout_targets = {"cactus.view", "cactus.reshape", "aten.view.default", "aten.reshape.default"}

    for node in graph.nodes:
        if node.target not in {"cactus.scalar_multiply", "aten.mul.Scalar"} or len(node.parents) != 1:
            continue
        cap = scalar_node_value(node)
        tanh = node.parents[0]

        if cap is None or cap <= 0.0 or tanh.target not in {"cactus.tanh", "aten.tanh.default"} or len(tanh.parents) != 1:
            continue

        divide = tanh.parents[0]
        if divide.target not in {"cactus.scalar_divide", "aten.div.Scalar"} or len(divide.parents) != 1:
            continue
        divisor = scalar_node_value(divide)
        if divisor is None or abs(divisor - cap) > max(1e-6, abs(cap) * 1e-6):
            continue

        source = divide.parents[0]
        layouts: list[Node] = []
        while source.target in layout_targets and len(source.parents) == 1:
            layouts.append(source)
            source = source.parents[0]

        if source.target not in {"cactus.linear", "aten.linear.default", "cactus.matmul", "aten.matmul.default"}:
            continue
        if len(source.parents) < 2 or not has_ancestor_name(source, "lm_head"):
            continue

        chain = [source, *reversed(layouts), divide, tanh, node]
        if any(tuple(child.name for child in current.children) != (next_node.name,)
               for current, next_node in zip(chain, chain[1:])):
            continue

        attrs = {
            "cap": cap,
            "pretransposed_rhs": bool(source.attrs.get("pretransposed_rhs", source.target == "aten.linear.default")),
        }
        replacements[node.name] = Node(
            index=node.index, name=node.name, node_type="call_function",
            target="cactus.logits_tq_softcap",
            args=[{"node": source.parents[0].name}, {"node": source.parents[1].name}],
            kwargs=dict(attrs), users=(), tensor_output_meta=node.tensor_output_meta,
            module_stack=source.module_stack, value_kind=node.value_kind, attrs=attrs,
            ir_metadata=dict(node.ir_metadata), cache=node.cache,
        )
        consumed.update(current.name for current in chain[:-1])

    if not replacements:
        return graph

    nodes = tuple(
        replacements.get(node.name, clone_rewritten_node(node, {}))
        for node in graph.nodes
        if node.name not in consumed
    )
    return prune_dead_nodes(rebuild_graph(nodes, graph, tuple(graph.fusions)))

def scalar_node_value(node: Node) -> float | None:
    for key in ("value", "other", "arg_1"):
        value = node.attrs.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None

def has_ancestor_name(node: Node, pattern: str, max_depth: int = 4) -> bool:
    pattern = pattern.lower()
    worklist = [(node, 0)]
    seen: set[str] = set()
    while worklist:
        current, depth = worklist.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        if pattern in f"{current.name} {current.target} {current.module_stack!r}".lower():
            return True
        if depth < max_depth:
            worklist.extend((parent, depth + 1) for parent in current.parents)
    return False

def transpose_permutation(node: Node) -> tuple[int, ...] | None:
    if node.target not in {"cactus.transpose", "aten.permute.default"}:
        return None

    value = node.attrs.get("permutation")

    if not isinstance(value, (list, tuple)):
        return None

    permutation = tuple(value)

    if any(not isinstance(index, int) for index in permutation):
        return None
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        return None

    return permutation

def clone_composed_transpose(node: Node, source_name: str, permutation: tuple[int, ...]) -> Node:
    args = rewrite_node_refs(node.args, {node.parents[0].name: source_name})
    kwargs = rewrite_node_refs(node.kwargs, {node.parents[0].name: source_name})

    if isinstance(kwargs, dict) and "permutation" in kwargs:
        kwargs["permutation"] = list(permutation)
    if node.target == "aten.permute.default" and isinstance(args, list) and len(args) > 1:
        args[1] = list(permutation)

    attrs = dict(node.attrs)
    attrs["permutation"] = list(permutation)
    return Node(
        index=node.index,
        name=node.name,
        node_type=node.node_type,
        target=node.target,
        args=args,
        kwargs=kwargs,
        users=(),
        tensor_output_meta=node.tensor_output_meta,
        module_stack=node.module_stack,
        value_kind=node.value_kind,
        attrs=attrs,
        ir_metadata=dict(node.ir_metadata),
        cache=node.cache,
    )

def noop_replacement_name(node: Node) -> str | None:
    if len(node.parents) != 1 or node.is_output:
        return None

    parent = node.parents[0]

    if node.target == "aten._assert_tensor_metadata.default":
        return parent.name

    if node.target in {"aten.clone.default", "aten.contiguous.default", "aten.detach.default"}:
        return parent.name

    if node.target in {"aten._to_copy.default", "aten.view.default", "aten.reshape.default", "aten.expand.default", "cactus.view", "cactus.reshape"} and same_tensor_signature(node, parent):
        return parent.name

    if is_precision_cast_only_for_rms_norm(node, parent):
        return parent.name

    return None

def is_precision_cast_only_for_rms_norm(node: Node, parent: Node) -> bool:
    if node.target not in {"aten._to_copy.default", "cactus.precision_cast"}:
        return False

    if tensor_meta_value(parent, "shape") != tensor_meta_value(node, "shape"):
        return False

    if effective_dtype(tensor_meta_value(parent, "dtype")) != "torch.float16":
        return False

    if str(tensor_meta_value(node, "dtype")) != "torch.float32":
        return False

    return bool(node.children) and all(child.target == "cactus.rms_norm" for child in node.children)

def resolve_replacement_names(replacement_names: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}

    for name in replacement_names:
        current = replacement_names[name]
        seen = {name}

        while current in replacement_names and current not in seen:
            seen.add(current)
            current = replacement_names[current]

        resolved[name] = current

    return resolved

def same_tensor_signature(left: Node, right: Node) -> bool:
    return same_shape(left, right) and same_effective_dtype(left, right)

def same_shape(left: Node, right: Node) -> bool:
    return tensor_meta_value(left, "shape") == tensor_meta_value(right, "shape")

def same_effective_dtype(left: Node, right: Node) -> bool:
    return effective_dtype(tensor_meta_value(left, "dtype")) == effective_dtype(tensor_meta_value(right, "dtype"))

def tensor_meta_value(node: Node, key: str) -> Any:
    if not isinstance(node.tensor_output_meta, dict):
        return None

    return node.tensor_output_meta.get(key)

def effective_dtype(dtype: Any) -> str | None:
    if dtype is None:
        return None

    dtype_name = str(dtype)

    if dtype_name in {"torch.float16", "torch.bfloat16"}:
        return "torch.float16"

    return dtype_name

def rebuild_graph(nodes: tuple[Node, ...], original_graph: Graph, fusion_results: tuple[FusionResult, ...] = ()) -> Graph:
    nodes_map = {node.name: node for node in nodes}
    parent_lists: dict[str, list[Node]] = {node.name: [] for node in nodes}
    child_lists: dict[str, list[Node]] = {node.name: [] for node in nodes}

    for node in nodes:
        parent_names = extract_node_refs((node.args, node.kwargs))

        for parent_name in parent_names:
            if parent_name not in nodes_map:
                raise ValueError(f"{node.name} references missing parent node {parent_name}")

            parent = nodes_map[parent_name]
            parent_lists[node.name].append(parent)

            if node not in child_lists[parent.name]:
                child_lists[parent.name].append(node)

    for node in nodes:
        node.parents = tuple(parent_lists[node.name])
        node.children = tuple(child_lists[node.name])
        node.users = tuple(child.name for child in node.children)

    sources = tuple(node for node in nodes if not node.parents)
    outputs = tuple(node for node in nodes if node.is_output)

    if not outputs:
        outputs = tuple(node for node in nodes if not node.children)

    return Graph(
        source=sources[0] if sources else None,
        sources=sources,
        outputs=outputs,
        nodes=nodes,
        nodes_map=nodes_map,
        model_name=original_graph.model_name,
        task=original_graph.task,
        graph_signature=original_graph.graph_signature,
        range_constants=original_graph.range_constants,
        input_specs=original_graph.input_specs,
        output_specs=original_graph.output_specs,
        cache_annotations=cache_annotations_from_nodes(nodes),
        metadata=dict(original_graph.metadata),
        fusions=list(fusion_results),
    )

def prune_dead_nodes(graph: Graph) -> Graph:
    live_names = output_ancestor_names(graph)

    if len(live_names) == len(graph.nodes):
        return graph

    live_nodes = tuple(node for node in graph.nodes if node.name in live_names)
    return rebuild_graph(live_nodes, graph, tuple(graph.fusions))

def output_ancestor_names(graph: Graph) -> frozenset[str]:
    live_names: set[str] = set()
    stack = list(graph.outputs)

    while stack:
        node = stack.pop()

        if node.name in live_names:
            continue

        live_names.add(node.name)
        stack.extend(node.parents)

    return frozenset(live_names)

def graph_to_layer_map(graph: Graph) -> CModels.LayerMap:
    records = [
        CModels.LayerRecord(
            index=index,
            name=node.name,
            node_type=node.node_type,
            target=node.target,
            args=node.args,
            kwargs=node.kwargs,
            users=[child.name for child in node.children],
            tensor_output_meta=node.tensor_output_meta,
            module_stack=node.module_stack,
            ir_metadata=node_ir_metadata(node),
        )
        for index, node in enumerate(topological_sort(graph))
    ]

    return CModels.LayerMap(
        model_name=graph.model_name,
        task=graph.task,
        graph_signature=graph.graph_signature,
        range_constants=graph.range_constants,
        input_specs=list(graph.input_specs),
        output_specs=list(graph.output_specs),
        nodes=records,
    )

def topological_sort(graph: Graph) -> tuple[Node, ...]:
    indegree = {node.name: len({parent.name for parent in node.parents}) for node in graph.nodes}
    ready = sorted((node for node in graph.nodes if indegree[node.name] == 0), key=lambda node: (node.index, node.name))
    ordered: list[Node] = []

    while ready:
        node = ready.pop(0)
        ordered.append(node)

        for child in sorted(node.children, key=lambda child: (child.index, child.name)):
            indegree[child.name] -= 1

            if indegree[child.name] == 0:
                ready.append(child)
                ready.sort(key=lambda ready_node: (ready_node.index, ready_node.name))

    if len(ordered) != len(graph.nodes):
        raise ValueError("Cannot topologically sort graph with cycles or missing edges")

    return tuple(ordered)

def annotate_cache_nodes(nodes: tuple[Node, ...], model_name: str = "") -> None:
    kv_count = 0
    conv_count = 0

    for node in nodes:
        tensor_index = past_key_value_index(node)

        if node.cache is not None and tensor_index is None:
            continue

        if tensor_index is None:
            if node.target == "cache_position":
                node.cache = CacheAnnotation(kind="position", role=FModels.CacheTensorRole.POSITION, source="placeholder_name")
                node.ir_metadata["cache"] = cache_annotation_to_dict(node.cache)

            continue

        shape = tensor_shape(node)

        if len(shape) == 4:
            layer_index = cache_layer_index_from_node(node, kv_count // 2)
            sequence_length = known_int(shape[2])
            role = FModels.CacheTensorRole.KEY if kv_count % 2 == 0 else FModels.CacheTensorRole.VALUE
            node.cache = CacheAnnotation(
                kind=FModels.CacheKind.KV,
                role=role,
                tensor_index=tensor_index,
                layer_index=layer_index,
                shape=tuple(shape),
                layout="batch_heads_sequence_head_dim",
                num_kv_heads=known_int(shape[1]),
                sequence_length=sequence_length,
                window_size=kv_cache_window_size_for_model(model_name, layer_index, sequence_length),
                head_dim=known_int(shape[3]),
                source="past_key_values_placeholder",
            )
            kv_count += 1
        elif len(shape) == 3:
            node.cache = CacheAnnotation(
                kind=FModels.CacheKind.CONV,
                role=FModels.CacheTensorRole.STATE,
                tensor_index=tensor_index,
                layer_index=cache_layer_index_from_node(node, conv_count),
                shape=tuple(shape),
                layout="batch_hidden_window",
                hidden_dim=known_int(shape[1]),
                window_size=known_int(shape[2]),
                source="past_key_values_placeholder",
            )
            conv_count += 1

        if node.cache is not None:
            node.ir_metadata["cache"] = cache_annotation_to_dict(node.cache)

def kv_cache_window_size_for_model(model_name: str, layer_index: int | None, sequence_length: int | None = None) -> int | None:
    if layer_index is None:
        return None

    if "gemma" not in model_name.lower():
        return None

    if is_gemma4_model_name(model_name) and layer_index == 13:
        return 0

    return 0 if layer_index % 5 == 4 else 512

def is_gemma4_model_name(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", "-")
    return "gemma-4" in normalized or "gemma4" in normalized

def cache_annotation_from_metadata(metadata: dict[str, Any]) -> CacheAnnotation | None:
    raw_cache = metadata.get("cache")

    if not isinstance(raw_cache, dict):
        return None

    return CacheAnnotation(
        kind=str(raw_cache.get("kind")),
        role=str(raw_cache.get("role")),
        tensor_index=optional_int(raw_cache.get("tensor_index")),
        layer_index=optional_int(raw_cache.get("layer_index")),
        shape=tuple(raw_cache.get("shape") or ()),
        layout=optional_str(raw_cache.get("layout")),
        sequence_length=optional_int(raw_cache.get("sequence_length")),
        window_size=optional_int(raw_cache.get("window_size")),
        hidden_dim=optional_int(raw_cache.get("hidden_dim")),
        num_kv_heads=optional_int(raw_cache.get("num_kv_heads")),
        head_dim=optional_int(raw_cache.get("head_dim")),
        source=str(raw_cache.get("source", "metadata")),
    )

def cache_annotation_to_dict(annotation: CacheAnnotation) -> dict[str, Any]:
    return {
        "kind": annotation.kind,
        "role": annotation.role,
        "tensor_index": annotation.tensor_index,
        "layer_index": annotation.layer_index,
        "shape": list(annotation.shape),
        "layout": annotation.layout,
        "sequence_length": annotation.sequence_length,
        "window_size": annotation.window_size,
        "hidden_dim": annotation.hidden_dim,
        "num_kv_heads": annotation.num_kv_heads,
        "head_dim": annotation.head_dim,
        "source": annotation.source,
    }

def cache_annotation_from_fusion_result(result: FusionResult) -> CacheAnnotation | None:
    if not result.fusion.graph.cache_outputs:
        return None

    cache_output = result.fusion.graph.cache_outputs[0]
    base = first_cache_input_annotation(result)
    shape = tensor_shape(result.source)
    kind = cache_output.cache_kind or (base.kind if base is not None else FModels.CacheKind.KV)
    role = cache_output.tensor_role or (base.role if base is not None else FModels.CacheTensorRole.STATE)

    return CacheAnnotation(
        kind=kind,
        role=role,
        tensor_index=base.tensor_index if base is not None else None,
        layer_index=base.layer_index if base is not None else infer_layer_index(result),
        shape=tuple(shape),
        layout=base.layout if base is not None else inferred_cache_layout(kind, shape),
        sequence_length=known_int(shape[2]) if len(shape) == 4 else None,
        window_size=cache_window_size(result, shape, base),
        hidden_dim=cache_hidden_dim(result, shape, base),
        num_kv_heads=known_int(shape[1]) if len(shape) == 4 else None,
        head_dim=known_int(shape[3]) if len(shape) == 4 else None,
        source=f"fusion:{result.fusion_name}",
    )

def first_cache_input_annotation(result: FusionResult) -> CacheAnnotation | None:
    for node in result.external_inputs:
        if node.cache is not None:
            return node.cache

    return None

def infer_layer_index(result: FusionResult) -> int | None:
    for node in result.matched_nodes:
        layer_index = layer_index_from_text(f"{node.name} {node.target} {node.module_stack!r}")

        if layer_index is not None:
            return layer_index

    return None

def cache_layer_index_from_node(node: Node, default: int | None = None) -> int | None:
    candidates: list[tuple[Node, int]] = [(node, 0)]
    seen = {id(node)}

    for candidate, depth in candidates:
        layer_index = layer_index_from_text(f"{candidate.name} {candidate.target} {candidate.module_stack!r}")

        if layer_index is not None:
            return layer_index

        if depth >= 4:
            continue

        for neighbor in (*candidate.children, *candidate.parents):
            if id(neighbor) in seen:
                continue

            seen.add(id(neighbor))
            candidates.append((neighbor, depth + 1))

    return default

def layer_index_from_text(value: str) -> int | None:
    parts = value.replace("/", ".").replace("_", ".").split(".")

    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "layer", "blocks", "block", "h"} and parts[index + 1].isdigit():
            return int(parts[index + 1])

    return None

def cache_window_size(result: FusionResult, shape: tuple[Any, ...], base: CacheAnnotation | None) -> int | None:
    if "cache_window_size" in result.attrs and result.attrs["cache_window_size"] is not None:
        return int(result.attrs["cache_window_size"])

    if result.target in {"cactus.attention", "cactus.attention_cached"} and base is not None and base.window_size is not None:
        return base.window_size

    if "window_size" in result.attrs and result.attrs["window_size"] is not None:
        return int(result.attrs["window_size"])

    if base is not None and base.window_size is not None:
        return base.window_size

    if len(shape) == 3:
        return known_int(shape[2])

    return None

def cache_hidden_dim(result: FusionResult, shape: tuple[Any, ...], base: CacheAnnotation | None) -> int | None:
    if "hidden_dim" in result.attrs and result.attrs["hidden_dim"] is not None:
        return int(result.attrs["hidden_dim"])

    if base is not None and base.hidden_dim is not None:
        return base.hidden_dim

    if len(shape) == 3:
        return known_int(shape[1])

    return None

def inferred_cache_layout(kind: str, shape: tuple[Any, ...]) -> str | None:
    if kind == FModels.CacheKind.KV and len(shape) == 4:
        return "batch_heads_sequence_head_dim"

    if kind == FModels.CacheKind.CONV and len(shape) == 3:
        return "batch_hidden_window"

    return None

def fusion_result_metadata(result: FusionResult, cache: CacheAnnotation | None) -> dict[str, Any]:
    metadata = {
        "fusion": {
            "name": result.fusion_name,
            "cactus_op": result.cactus_op,
            "matched_nodes": [node.name for node in result.matched_nodes],
        }
    }

    if cache is not None:
        metadata["cache"] = cache_annotation_to_dict(cache)

    return metadata

def cache_annotations_from_nodes(nodes: tuple[Node, ...]) -> tuple[CacheAnnotation, ...]:
    return tuple(node.cache for node in nodes if node.cache is not None)

def node_ir_metadata(node: Node) -> dict[str, Any]:
    metadata = dict(node.ir_metadata)

    if node.cache is not None:
        metadata["cache"] = cache_annotation_to_dict(node.cache)

    return metadata

def past_key_value_index(node: Node) -> int | None:
    for value in (node.target, node.name):
        prefix = "past_key_values_"

        if isinstance(value, str) and value.startswith(prefix):
            suffix = value[len(prefix):]

            if suffix.isdigit():
                return int(suffix)

    return None

def tensor_shape(node: Node) -> tuple[Any, ...]:
    if not isinstance(node.tensor_output_meta, dict):
        return ()

    shape = node.tensor_output_meta.get("shape")

    if isinstance(shape, list):
        return tuple(shape)

    if isinstance(shape, tuple):
        return shape

    return ()

def known_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, str) and value.isdigit():
        return int(value)

    return None

def optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return known_int(value)

def optional_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)
