from dataclasses import dataclass, field
from typing import Any

from . import constants

from ..Converter import models as CModels
from ..Fusions import models as FModels


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
    fusions: list["FusionResult"] = field(default_factory=list)

    @classmethod
    def from_map(cls, layer_map: CModels.LayerMap) -> "Graph":
        return generate_graph(cls, layer_map)

    def apply_fusions(self, fusion_results: list["FusionResult"] | tuple["FusionResult", ...]) -> "Graph":
        return apply_fusions_to_graph(self, tuple(fusion_results))

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




################################################# Model Utils!!!!!!! #################################################


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
    )


def apply_fusions_to_graph(graph: Graph, fusion_results: tuple[FusionResult, ...]) -> Graph:
    if not fusion_results:
        return graph

    validate_non_overlapping_fusions(fusion_results)

    fused_nodes, replacement_names = build_fused_nodes(graph, fusion_results)
    consumed_names = expanded_consumed_names(graph, replacement_names)
    kept_nodes = tuple(clone_rewritten_node(node, replacement_names) for node in graph.nodes if node.name not in consumed_names)
    rewritten_fused_nodes = tuple(clone_rewritten_node(node, replacement_names) for node in fused_nodes)
    return rebuild_graph((*kept_nodes, *rewritten_fused_nodes), graph, fusion_results)


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
    return Node(
        index=result.source.index,
        name=fused_name,
        node_type="call_function",
        target=result.target,
        args=[{"node": node.name} for node in result.external_inputs],
        kwargs=dict(result.attrs),
        users=(),
        tensor_output_meta=result.source.tensor_output_meta,
        module_stack=result.source.module_stack,
        value_kind=FModels.ValueKind.ACTIVATION,
        attrs=dict(result.attrs),
    )


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
        fusions=list(fusion_results),
    )


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
        )
        for index, node in enumerate(topological_sort(graph))
    ]

    return CModels.LayerMap(
        model_name=graph.model_name,
        task=graph.task,
        graph_signature=graph.graph_signature,
        range_constants=graph.range_constants,
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
