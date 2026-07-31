from __future__ import annotations

from . import edges as E
from . import models as M
from . import nodes as N


def _nodes(*names: str) -> dict[str, M.FusionNode]:
    return {name: N.NODES[name] for name in names}


def _edges(*names: str) -> tuple[M.FusionEdge, ...]:
    return tuple(E.EDGES[name] for name in names)


def _input(role: str, node: str, parent_index: int | None = None, **kwargs) -> M.FusionInput:
    return M.FusionInput(role=role, source=M.NodeRef(node=node, parent_index=parent_index), **kwargs)


def _variadic_input(
    role: str,
    node: str,
    start_parent_index: int,
    *,
    min_count: int = 1,
    max_count: int | None = None,
    end_parent_index: int | None = None,
) -> M.FusionInput:
    return M.FusionInput(
        role=role,
        source=M.NodeRef(node=node, parent_index=start_parent_index),
        variadic=True,
        min_count=min_count,
        max_count=max_count,
        end_parent_index=end_parent_index,
    )


def _shared_input(
    left_node: str,
    left_parent_index: int,
    right_node: str,
    right_parent_index: int,
) -> tuple[M.NodeRef, M.NodeRef]:
    return (
        M.NodeRef(left_node, left_parent_index),
        M.NodeRef(right_node, right_parent_index),
    )


def _output(node: str, role: str = "out", output_index: int | None = None) -> M.FusionOutput:
    return M.FusionOutput(role=role, node=node, output_index=output_index)


def _cache_input(
    role: str,
    node: str,
    parent_index: int,
    *,
    cache_kind: str = M.CacheKind.KV,
    tensor_role: str | None = None,
    optional: bool = False,
) -> M.CacheInput:
    return M.CacheInput(
        role,
        M.NodeRef(node, parent_index),
        cache_kind=cache_kind,
        tensor_role=tensor_role,
        optional=optional,
    )


def _cache_output(
    role: str,
    node: str,
    *,
    cache_kind: str = M.CacheKind.KV,
    tensor_role: str | None = None,
) -> M.CacheOutput:
    return M.CacheOutput(role, node, cache_kind=cache_kind, tensor_role=tensor_role)


def _cache_mutation(
    name: str,
    *,
    cache_kind: str = M.CacheKind.KV,
    read_roles: tuple[str, ...] = (),
    write_roles: tuple[str, ...] = (),
) -> M.CacheMutation:
    return M.CacheMutation(
        name,
        cache_kind=cache_kind,
        read_roles=read_roles,
        write_roles=write_roles,
    )


def _required_attrs(**attrs) -> dict:
    return {"required_attrs": attrs}


def _graph(
    name: str,
    root: str,
    node_names: tuple[str, ...],
    *,
    edge_names: tuple[str, ...] = (),
    inputs: tuple[M.FusionInput, ...] = (),
    shared_inputs: tuple[tuple[M.NodeRef, M.NodeRef], ...] = (),
    outputs: tuple[M.FusionOutput, ...] | None = None,
    attr_captures: tuple[M.AttrCapture, ...] = (),
    repeated_subgraphs: tuple[M.RepeatedSubgraph, ...] = (),
    cache_inputs: tuple[M.CacheInput, ...] = (),
    cache_outputs: tuple[M.CacheOutput, ...] = (),
    cache_mutations: tuple[M.CacheMutation, ...] = (),
    constraints: dict[str, M.ConstraintValue] | None = None,
    variants: tuple[str, ...] = (),
    metadata: dict | None = None,
    allow_root_external_children: bool = True,
) -> M.FusionGraph:
    return M.FusionGraph(
        name=name,
        root=root,
        nodes=_nodes(*node_names),
        edges=_edges(*edge_names),
        inputs=inputs,
        shared_inputs=shared_inputs,
        outputs=outputs or (_output(root),),
        attr_captures=attr_captures,
        repeated_subgraphs=repeated_subgraphs,
        cache_inputs=cache_inputs,
        cache_outputs=cache_outputs,
        cache_mutations=cache_mutations,
        constraints=constraints or {},
        variants=variants,
        metadata=metadata or {},
        allow_root_external_children=allow_root_external_children,
    )


def _single_node_graph(
    name: str,
    root: str,
    input_roles: tuple[str, ...] = (),
    *,
    attr_captures: tuple[M.AttrCapture, ...] = (),
    metadata: dict | None = None,
) -> M.FusionGraph:
    return _graph(
        name,
        root,
        (root,),
        inputs=tuple(_input(role, root, index) for index, role in enumerate(input_roles)),
        attr_captures=attr_captures,
        metadata=metadata,
    )


def _definition(
    name: str,
    cactus_op: str,
    graph: M.FusionGraph,
    *,
    fusion_fields: tuple[str, ...] = ("generic",),
    supported_inference_modes: tuple[str, ...] = (),
    supported_modalities: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> M.FusionDefinition:
    return M.FusionDefinition(
        name=name,
        target=f"cactus.{cactus_op}",
        graph=graph,
        fusion_fields=fusion_fields,
        supported_inference_modes=supported_inference_modes,
        supported_modalities=supported_modalities,
        cactus_op=cactus_op,
        metadata=metadata or {},
    )


def _index_by_target(fusions: dict[str, M.FusionDefinition]) -> dict[str, M.FusionDefinition]:
    return {fusion.target: fusion for fusion in fusions.values()}


def _index_by_cactus_op(
    fusions: dict[str, M.FusionDefinition],
) -> dict[str, tuple[M.FusionDefinition, ...]]:
    index: dict[str, tuple[M.FusionDefinition, ...]] = {}

    for fusion in fusions.values():
        if fusion.cactus_op is None:
            continue

        index[fusion.cactus_op] = (*index.get(fusion.cactus_op, ()), fusion)

    return index


def _index_by_field(
    fusions: dict[str, M.FusionDefinition],
) -> dict[str, tuple[M.FusionDefinition, ...]]:
    index: dict[str, tuple[M.FusionDefinition, ...]] = {}

    for fusion in fusions.values():
        for field in fusion.fusion_fields:
            index[field] = (*index.get(field, ()), fusion)

    return index


def _index_by_root_op(
    fusions: dict[str, M.FusionDefinition],
) -> dict[str, tuple[M.FusionDefinition, ...]]:
    index: dict[str, tuple[M.FusionDefinition, ...]] = {}

    for fusion in fusions.values():
        root_node = fusion.graph.nodes.get(fusion.graph.root)
        if root_node is None:
            continue

        for op in root_node.ops:
            index[op] = (*index.get(op, ()), fusion)

    return index
