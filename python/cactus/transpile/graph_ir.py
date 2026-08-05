import copy
from dataclasses import dataclass
from dataclasses import field

import torch


def _clone_ir_payload(value, memo):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        # IR constants may be real tensors or export-time tensor wrappers.
        # They are treated as immutable graph payloads, so preserve them by
        # reference instead of triggering Tensor.__deepcopy__.
        return value
    if isinstance(value, tuple):
        return tuple(_clone_ir_payload(item, memo) for item in value)
    if isinstance(value, list):
        return [_clone_ir_payload(item, memo) for item in value]
    if isinstance(value, dict):
        return {
            copy.deepcopy(key, memo): _clone_ir_payload(inner, memo)
            for key, inner in value.items()
        }
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        return value


@dataclass
class IRValue:
    id: str 
    shape: tuple[int, ...] | None = None 
    dtype: str | None = None 
    producer: str | None = None
    users: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        cloned = IRValue(
            id=self.id,
            shape=_clone_ir_payload(self.shape, memo),
            dtype=self.dtype,
            producer=self.producer,
            users=list(self.users),
            meta=_clone_ir_payload(self.meta, memo),
        )
        memo[id(self)] = cloned
        return cloned

@dataclass 
class IRNode:
    id: str
    op: str 
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    kind: str = "generic"

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        cloned = IRNode(
            id=self.id,
            op=self.op,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            attrs=_clone_ir_payload(self.attrs, memo),
            meta=_clone_ir_payload(self.meta, memo),
            kind=self.kind,
        )
        memo[id(self)] = cloned
        return cloned

@dataclass 
class IRGraph:
    values: dict[str, IRValue] 
    nodes: dict[str, IRNode]
    order: list[str]
    inputs: list[str]
    outputs: list[str]
    constants: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        cloned = IRGraph(
            values={value_id: copy.deepcopy(value, memo) for value_id, value in self.values.items()},
            nodes={node_id: copy.deepcopy(node, memo) for node_id, node in self.nodes.items()},
            order=list(self.order),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            constants={
                value_id: _clone_ir_payload(constant, memo)
                for value_id, constant in self.constants.items()
            },
            meta=_clone_ir_payload(self.meta, memo),
        )
        memo[id(self)] = cloned
        return cloned

    def add_node(self, node: IRNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate IR node id: {node.id}")
        self.nodes[node.id] = node
        for output in node.outputs:
            if output in self.values:
                raise ValueError(f"duplicate IR value id: {output}")
            self.values[output] = IRValue(id=output, producer=node.id)

    def add_value(self, value: IRValue) -> None:
        if value.id in self.values:
            raise ValueError(f"duplicate IR value id: {value.id}")
        self.values[value.id] = value


def verify_ir(graph: IRGraph) -> None:
    order_set = set(graph.order)
    node_keys = set(graph.nodes.keys())
    if len(order_set) != len(graph.order):
        raise ValueError("IR order contains duplicate node ids")
    if order_set != node_keys:
        missing_in_order = node_keys - order_set
        missing_in_nodes = order_set - node_keys
        raise ValueError(
            "IR order/nodes mismatch: "
            f"missing_in_order={sorted(missing_in_order)} "
            f"missing_in_nodes={sorted(missing_in_nodes)}"
        )

    seen_nodes: set[str] = set()
    seen_outputs: set[str] = set()
    expected_users: dict[str, list[str]] = {}

    for value_id in graph.inputs:
        if value_id not in graph.values:
            raise ValueError(f"IR input missing value entry: {value_id}")
        expected_users.setdefault(value_id, [])

    for value_id in graph.constants:
        if value_id not in graph.values:
            raise ValueError(f"IR constant missing value entry: {value_id}")
        expected_users.setdefault(value_id, [])

    for node_id in graph.order:
        if node_id in seen_nodes:
            raise ValueError(f"duplicate node in IR order: {node_id}")
        seen_nodes.add(node_id)

        node = graph.nodes[node_id]
        if not node.outputs:
            raise ValueError(f"IR node has no outputs: {node_id}")

        for input_id in node.inputs:
            if input_id not in graph.values:
                raise ValueError(f"IR node {node_id} references missing input value {input_id}")
            expected_users.setdefault(input_id, []).append(node_id)
            producer = graph.values[input_id].producer
            if producer is not None and producer not in graph.nodes:
                raise ValueError(
                    f"IR value {input_id} for node {node_id} has missing producer {producer}"
                )

        for output_id in node.outputs:
            if output_id in seen_outputs:
                raise ValueError(f"IR value produced more than once: {output_id}")
            seen_outputs.add(output_id)
            if output_id not in graph.values:
                raise ValueError(f"IR node {node_id} missing output value entry {output_id}")
            value = graph.values[output_id]
            if value.producer != node_id:
                raise ValueError(
                    f"IR value {output_id} producer mismatch: expected {node_id}, got {value.producer}"
                )
            expected_users.setdefault(output_id, [])

    for output_id in graph.outputs:
        if output_id not in graph.values:
            raise ValueError(f"IR graph output missing value entry: {output_id}")

    for value_id, value in graph.values.items():
        expected = expected_users.get(value_id, [])
        if value.users != expected:
            raise ValueError(
                f"IR value {value_id} user list mismatch: expected {expected}, got {value.users}"
            )
        if value.producer is None and value_id not in graph.inputs and value_id not in graph.constants:
            if value_id in seen_outputs:
                raise ValueError(f"IR value {value_id} is produced but has no producer")
        if value.producer is not None and value.producer not in graph.nodes:
            raise ValueError(
                f"IR value {value_id} references unknown producer {value.producer}"
            )
