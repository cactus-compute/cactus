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
    fusions: list["FusionResult"] = field(default_factory=list)

    @classmethod
    def from_map(cls, layer_map: CModels.LayerMap) -> "Graph":
        return generate_graph(cls, layer_map)


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

    return cls(source=sources[0] if sources else None, sources=sources, outputs=outputs, nodes=nodes, nodes_map=nodes_map)
