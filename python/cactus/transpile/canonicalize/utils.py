from __future__ import annotations

from typing import Any

import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir


def normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if not isinstance(dtype, str):
        return str(dtype)
    if dtype.startswith("torch."):
        torch_name = dtype.split(".", 1)[1]
        mapping = {
            "bool": "bool",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
            "float64": "fp64",
        }
        return mapping.get(torch_name, dtype)
    return dtype


def ir_dtype_to_torch(dtype: str) -> torch.dtype:
    normalized = normalize_dtype_name(dtype)
    if normalized == "bool":
        return torch.bool
    if normalized == "int8":
        return torch.int8
    if normalized == "int16":
        return torch.int16
    if normalized == "int32":
        return torch.int32
    if normalized == "int64":
        return torch.int64
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    if normalized == "fp64":
        return torch.float64
    raise NotImplementedError(f"unsupported IR dtype for constant materialization: {dtype}")


def normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim


def numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def refresh_value_users(graph: IRGraph) -> None:
    for value in graph.values.values():
        value.users = []
    for node_id in graph.order:
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            value = graph.values.get(input_id)
            if value is not None:
                value.users.append(node_id)


def rebuild_graph(graph: IRGraph) -> None:
    new_values: dict[str, IRValue] = {}

    for value_id in graph.inputs:
        old = graph.values.get(value_id, IRValue(id=value_id))
        new_values[value_id] = IRValue(
            id=value_id,
            shape=old.shape,
            dtype=old.dtype,
            producer=None,
            users=[],
            meta=dict(old.meta),
        )

    for value_id in graph.constants.keys():
        old = graph.values.get(value_id, IRValue(id=value_id))
        new_values[value_id] = IRValue(
            id=value_id,
            shape=old.shape,
            dtype=old.dtype,
            producer=None,
            users=[],
            meta=dict(old.meta),
        )

    for node_id in graph.order:
        node = graph.nodes[node_id]
        for output_id in node.outputs:
            old = graph.values.get(output_id, IRValue(id=output_id))
            new_values[output_id] = IRValue(
                id=output_id,
                shape=old.shape,
                dtype=old.dtype,
                producer=node_id,
                users=[],
                meta=dict(old.meta),
            )

    for node_id in graph.order:
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            if input_id not in new_values:
                old = graph.values.get(input_id, IRValue(id=input_id))
                new_values[input_id] = IRValue(
                    id=input_id,
                    shape=old.shape,
                    dtype=old.dtype,
                    producer=old.producer,
                    users=[],
                    meta=dict(old.meta),
                )
            new_values[input_id].users.append(node_id)

    for output_id in graph.outputs:
        if output_id not in new_values:
            old = graph.values.get(output_id, IRValue(id=output_id))
            new_values[output_id] = IRValue(
                id=output_id,
                shape=old.shape,
                dtype=old.dtype,
                producer=old.producer,
                users=[],
                meta=dict(old.meta),
            )

    graph.values = new_values
    verify_ir(graph)


def dce(graph: IRGraph) -> None:
    live_values = set(graph.outputs)
    live_nodes: set[str] = set()

    changed = True
    while changed:
        changed = False
        for node_id in reversed(graph.order):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if not any(output in live_values for output in node.outputs):
                continue
            if node_id not in live_nodes:
                live_nodes.add(node_id)
                changed = True
            for input_id in node.inputs:
                if input_id not in live_values:
                    live_values.add(input_id)
                    changed = True

    graph.order = [node_id for node_id in graph.order if node_id in live_nodes]
    graph.nodes = {node_id: node for node_id, node in graph.nodes.items() if node_id in live_nodes}

    live_value_ids = set(graph.inputs) | set(graph.outputs)
    for node in graph.nodes.values():
        live_value_ids.update(node.inputs)
        live_value_ids.update(node.outputs)

    graph.values = {value_id: value for value_id, value in graph.values.items() if value_id in live_value_ids}
    graph.constants = {value_id: value for value_id, value in graph.constants.items() if value_id in live_value_ids}
    rebuild_graph(graph)


def remove_node(graph: IRGraph, node_id: str) -> None:
    if node_id in graph.nodes:
        del graph.nodes[node_id]
    graph.order = [existing_id for existing_id in graph.order if existing_id != node_id]


def drop_value_if_unused(graph: IRGraph, value_id: str) -> None:
    if value_id in graph.inputs or value_id in graph.outputs or value_id in graph.constants:
        return
    value = graph.values.get(value_id)
    if value is None:
        return
    if value.producer is None and not value.users:
        del graph.values[value_id]


def bypass_unary_node(graph: IRGraph, node: IRNode) -> bool:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    input_id = node.inputs[0]
    output_id = node.outputs[0]

    for other in graph.nodes.values():
        other.inputs = [input_id if value_id == output_id else value_id for value_id in other.inputs]
    graph.outputs = [input_id if value_id == output_id else value_id for value_id in graph.outputs]

    remove_node(graph, node.id)
    if output_id in graph.values:
        del graph.values[output_id]
    drop_value_if_unused(graph, input_id)
    return True


def _is_gemma4_graph(graph: IRGraph) -> bool:
    return str(graph.meta.get("adapter_family") or graph.meta.get("family") or "").lower() == "gemma4"


def _weight_binding_meta(value: IRValue | None) -> dict[str, object] | None:
    if value is None:
        return None
    path = value.meta.get("path")
    kind = value.meta.get("kind")
    source_name = value.meta.get("source_name")
    if isinstance(path, str) and isinstance(kind, str) and isinstance(source_name, str):
        return dict(value.meta)
    return None


def _gemma4_materialized_binding_meta(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    if not _is_gemma4_graph(graph) or len(node.inputs) != 1:
        return None
    if node.op == "precision_cast":
        pass
    elif node.op == "scalar_add" and float(node.attrs.get("value", 0.0)) in (0.0, 1.0):
        pass
    else:
        return None

    value_id = node.inputs[0]
    for _ in range(4):
        value = graph.values.get(value_id)
        binding_meta = _weight_binding_meta(value)
        if binding_meta is not None:
            binding_meta["materialized_from_value_id"] = value_id
            binding_meta["materialized_by_op"] = node.op
            return binding_meta
        producer_id = value.producer if value is not None else None
        producer_node = graph.nodes.get(producer_id) if producer_id is not None else None
        if producer_node is None or producer_node.op not in {"precision_cast", "type_as"}:
            break
        if len(producer_node.inputs) != 1:
            break
        value_id = producer_node.inputs[0]
    return None


def materialize_constant_output(graph: IRGraph, node: IRNode, value: Any) -> bool:
    if len(node.outputs) != 1:
        return False
    output_id = node.outputs[0]
    output_value = graph.values.get(output_id)
    if output_value is None:
        return False
    binding_meta = _gemma4_materialized_binding_meta(graph, node)
    if binding_meta is not None:
        output_value.meta.update(binding_meta)
    graph.constants[output_id] = value
    output_value.producer = None
    remove_node(graph, node.id)
    return True
