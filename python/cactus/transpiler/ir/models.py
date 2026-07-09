



















































































































# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Any, Callable, Optional
# import converter.models as CVModels
# import constants


# class ValueKind:
#     UNKNOWN = "unknown"
#     PARAMETER = "parameter"
#     BUFFER = "buffer"
#     USER_INPUT = "user_input"
#     LIFTED_CONSTANT = "lifted_constant"
#     ACTIVATION = "activation"
#     OUTPUT = "output"


# @dataclass(slots=True)
# class TensorMeta:
#     shape: list[int] = field(default_factory=list)
#     dtype: str | None = None
#     device: str | None = None
#     requires_grad: bool | None = None
#     stride: list[int] | None = None

#     @property
#     def rank(self) -> int:
#         return len(self.shape)

#     def dim(self, index: int) -> int | None:
#         if index < 0:
#             index += len(self.shape)

#         if index < 0 or index >= len(self.shape):
#             return None

#         return self.shape[index]


# @dataclass(slots=True)
# class TupleOutputRef:
#     source_node_name: str
#     output_index: int


# @dataclass(slots=True)
# class Node:
#     name: str
#     underlying_op: str
#     normalized_attrs: dict[str, Any]
#     parents: list["Node"]
#     children: list["Node"]
#     tensor_meta: TensorMeta | None = None
#     tuple_output_metas: list[TensorMeta | None] = field(default_factory=list)
#     value_kind: str = ValueKind.UNKNOWN
#     tuple_source: TupleOutputRef | None = None
#     raw_layer: Optional[CVModels.LayerRecord] = None


# @dataclass(frozen=True, slots=True)
# class AttrPredicate:
#     description: str
#     predicate: Callable[[Any, Node], bool] = field(repr=False, compare=False)

#     def matches(self, value: Any, node: Node) -> bool:
#         return self.predicate(value, node)


# def attr_any() -> AttrPredicate:
#     return AttrPredicate("any value", lambda _value, _node: True)


# def attr_in(values: tuple[Any, ...]) -> AttrPredicate:
#     return AttrPredicate(f"in {values}", lambda value, _node: value in values)


# def attr_rank_dim(dim_values: tuple[int, ...]) -> AttrPredicate:
#     def matches(value: Any, node: Node) -> bool:
#         if value in dim_values:
#             return True

#         if node.tensor_meta is None:
#             return False

#         normalized_values = tuple(dim + node.tensor_meta.rank if dim < 0 else dim for dim in dim_values)
#         normalized_value = value + node.tensor_meta.rank if isinstance(value, int) and value < 0 else value
#         return normalized_value in normalized_values

#     return AttrPredicate(f"rank-aware dim in {dim_values}", matches)


# @dataclass(frozen=True, slots=True)
# class DimRef:
#     node: str
#     dim: int


# @dataclass(frozen=True, slots=True)
# class TensorConstraint:
#     rank: int | None = None
#     min_rank: int | None = None
#     max_rank: int | None = None
#     dtype: str | None = None
#     dim_equals: tuple[tuple[int, int], ...] = ()
#     same_dim_as: tuple[tuple[int, DimRef], ...] = ()

#     def matches(self, node: Node, bindings: dict[str, Node] | None = None) -> bool:
#         if node.tensor_meta is None:
#             return False

#         meta = node.tensor_meta

#         if self.rank is not None and meta.rank != self.rank:
#             return False

#         if self.min_rank is not None and meta.rank < self.min_rank:
#             return False

#         if self.max_rank is not None and meta.rank > self.max_rank:
#             return False

#         if self.dtype is not None and meta.dtype != self.dtype:
#             return False

#         for dim_index, expected in self.dim_equals:
#             if meta.dim(dim_index) != expected:
#                 return False

#         if bindings is None and self.same_dim_as:
#             return False

#         for dim_index, ref in self.same_dim_as:
#             other = bindings.get(ref.node) if bindings is not None else None
#             if other is None or other.tensor_meta is None:
#                 return False

#             if meta.dim(dim_index) != other.tensor_meta.dim(ref.dim):
#                 return False

#         return True


# @dataclass(slots=True)
# class NodeSpec:
#     underlying_op: str | tuple[str, ...]
#     required_attrs: dict[str, Any] = field(default_factory=dict)
#     allowed_value_kinds: tuple[str, ...] = ()
#     tensor_constraints: tuple[TensorConstraint, ...] = ()
#     optional: bool = False


# @dataclass(slots=True)
# class Edge:
#     from_node: str
#     parent_index: int
#     to_node: str
#     required: bool = True
#     to_output_index: int | None = None


# @dataclass(slots=True)
# class InputSpec:
#     node: str
#     parent_index: int
#     role: str | None = None
#     allowed_value_kinds: tuple[str, ...] = ()
#     tensor_constraints: tuple[TensorConstraint, ...] = ()


# @dataclass(slots=True)
# class VariadicInputSpec:
#     node: str
#     start_parent_index: int = 0
#     end_parent_index: int | None = None
#     min_count: int = 0
#     max_count: int | None = None
#     role: str | None = None
#     allowed_value_kinds: tuple[str, ...] = ()
#     tensor_constraints: tuple[TensorConstraint, ...] = ()


# @dataclass(slots=True)
# class InputRoleSpec:
#     input_name: str
#     role: str
#     allowed_value_kinds: tuple[str, ...] = ()
#     tensor_constraints: tuple[TensorConstraint, ...] = ()


# @dataclass(slots=True)
# class OptionalNodeSpec:
#     node: str
#     bypass_parent_index: int = 0


# @dataclass(slots=True)
# class TupleOutputSpec:
#     node: str
#     source_node: str
#     output_index: int


# @dataclass(slots=True)
# class RepeatedSubgraphSpec:
#     name: str
#     pattern: "FusionGraph"
#     min_count: int = 1
#     max_count: int | None = None
#     anchor_node: str | None = None


# @dataclass(slots=True)
# class AttrRef:
#     node: str
#     attr: str
#     default: Any = None
#     required: bool = True


# @dataclass(slots=True)
# class FusionGraph:
#     target: str
#     root: str
#     edges: tuple[Edge, ...]
#     inputs: tuple[InputSpec, ...] = ()
#     shared_inputs: tuple[tuple[InputSpec, InputSpec], ...] = ()
#     output_attrs: dict[str, Any] = field(default_factory=dict)
#     optional_nodes: tuple[OptionalNodeSpec, ...] = ()
#     variadic_inputs: tuple[VariadicInputSpec, ...] = ()
#     input_roles: tuple[InputRoleSpec, ...] = ()
#     tuple_outputs: tuple[TupleOutputSpec, ...] = ()
#     repeated_subgraphs: tuple[RepeatedSubgraphSpec, ...] = ()
#     metadata: dict[str, Any] = field(default_factory=dict)
#     allow_root_external_children: bool = True


# @dataclass(slots=True)
# class FusionResult:
#     node: Node
#     bindings: dict[str, Node]
#     internal_nodes: list[Node]
#     external_input: list[Node]
#     external_consumers: list[Node]
#     edges_consumed: list[str]
#     fusion: FusionGraph | None = None
#     variadic_external_inputs: dict[str, list[Node]] = field(default_factory=dict)
#     input_role_nodes: dict[str, list[Node]] = field(default_factory=dict)

#     @classmethod
#     def from_fusion(cls, fusion: FusionGraph, bindings: dict[str, Node]):
#         return build_fusion_result(fusion, bindings)


# @dataclass(slots=True)
# class Graph:
#     name_map: dict[str, Node]
#     source_nodes: list[Node]
#     consumed_ids: set[str]
#     fusedNodes: list[FusionResult] = field(default_factory=list)

#     @classmethod
#     def from_layer_map(cls, map: CVModels.LayerMap) -> "Graph":
#         return generate_graph(map)

#     def queue_fusion(self, fusion_result: FusionResult) -> None:
#         self.fusedNodes.append(fusion_result)

#     def update_graph_with_fusions(self) -> bool:
#         if not len(self.fusedNodes):
#             return False

#         update_graph(self)
#         return True


# """###################################### MODEL UTILS!!!!!!! ######################################"""

# # TODO: Understand
# def extract_node_names(value: Any) -> list[str]:
#     if isinstance(value, dict):
#         refs = []
#         if "node" in value:
#             refs.append(value["node"])

#         for item in value.values():
#             refs.extend(extract_node_names(item))

#         return refs

#     if isinstance(value, list):
#         refs = []
#         for item in value:
#             refs.extend(extract_node_names(item))
#         return refs

#     return []

# # TODO: Understand
# def extract_attrs(layer_: CVModels.LayerRecord) -> dict[str, Any]:
#     def contains_node_ref(value: Any) -> bool:
#         if isinstance(value, dict):
#             if "node" in value:
#                 return True
#             return any(contains_node_ref(item) for item in value.values())

#         if isinstance(value, list):
#             return any(contains_node_ref(item) for item in value)

#         return False

#     attrs = {}

#     if isinstance(layer_.kwargs, dict):
#         attrs.update(layer_.kwargs)

#     attr_names = constants.LAYER_ATTRS_MAP.get(layer_.target, [])

#     if None in attr_names:
#         for i, attr_name in enumerate(attr_names):
#             if attr_name is None or i >= len(layer_.args):
#                 continue

#             if contains_node_ref(layer_.args[i]) or attr_name in attrs:
#                 continue

#             attrs[attr_name] = layer_.args[i]
#     else:
#         positional_attrs = [item for item in layer_.args if not contains_node_ref(item)]

#         for i, attr_name in enumerate(attr_names):
#             if i < len(positional_attrs) and attr_name not in attrs:
#                 attrs[attr_name] = positional_attrs[i]

#     return attrs

# # TODO: Understand
# def attr_value_matches(expected_value: Any, actual_value: Any, node: Node) -> bool:
#     if isinstance(expected_value, AttrPredicate):
#         return expected_value.matches(actual_value, node)

#     return actual_value == expected_value

# # TODO: Understand
# def node_spec_matches(node: Node, spec: NodeSpec, bindings: dict[str, Node] | None = None) -> bool:
#     allowed_ops = spec.underlying_op if isinstance(spec.underlying_op, tuple) else (spec.underlying_op,)
#     if node.underlying_op not in allowed_ops:
#         return False

#     if spec.allowed_value_kinds and node.value_kind not in spec.allowed_value_kinds:
#         return False

#     for key, expected_value in spec.required_attrs.items():
#         if key not in node.normalized_attrs:
#             return False

#         if not attr_value_matches(expected_value, node.normalized_attrs[key], node):
#             return False

#     for constraint in spec.tensor_constraints:
#         if not constraint.matches(node, bindings):
#             return False

#     return True

# # TODO: Understand
# def _plain(value: Any) -> Any:
#     if hasattr(value, "model_dump"):
#         return value.model_dump()

#     if hasattr(value, "dict"):
#         return value.dict()

#     return value

# # TODO: Understand
# def _parse_one_tensor_meta(value: Any) -> TensorMeta | None:
#     value = _plain(value)

#     if not isinstance(value, dict):
#         return None

#     shape = value.get("shape")
#     if shape is None:
#         return None

#     return TensorMeta(
#         shape=[int(item) for item in shape],
#         dtype=value.get("dtype"),
#         device=value.get("device"),
#         requires_grad=value.get("requires_grad"),
#         stride=value.get("stride"),
#     )

# # TODO: Optimize this function further.
# def extract_tensor_metas(layer_: CVModels.LayerRecord) -> tuple[TensorMeta | None, list[TensorMeta | None]]:
#     meta = _plain(layer_.tensor_output_meta)

#     single_meta = _parse_one_tensor_meta(meta)
#     if single_meta is not None:
#         return single_meta, []

#     if isinstance(meta, list):
#         tuple_metas = [_parse_one_tensor_meta(item) for item in meta]
#         if tuple_metas:
#             return None, tuple_metas

#     return None, []

# # TODO: Optimize this function further.
# def infer_value_kind(layer_: CVModels.LayerRecord) -> str:
#     name = layer_.name
#     target = layer_.target

#     if layer_.node_type == "output":
#         return ValueKind.OUTPUT

#     if layer_.node_type == "placeholder":
#         if name.startswith("p_") or "_weight" in name or name.endswith("weight"):
#             return ValueKind.PARAMETER

#         if name.startswith("b_") or "buffer" in name:
#             return ValueKind.BUFFER

#         return ValueKind.USER_INPUT

#     if layer_.node_type == "get_attr":
#         return ValueKind.PARAMETER

#     if target in {
#         "aten.scalar_tensor.default",
#         "aten.full.default",
#         "aten.arange.start_step",
#     }:
#         return ValueKind.LIFTED_CONSTANT

#     return ValueKind.ACTIVATION

# # TODO: Optimize this function further.
# def extract_tuple_source(layer_: CVModels.LayerRecord) -> TupleOutputRef | None:
#     target = layer_.target
#     if target not in {"operator.getitem", "<built-in function getitem>"}:
#         return None

#     node_refs = extract_node_names(layer_.args)
#     if not node_refs:
#         return None

#     output_index = None
#     if isinstance(layer_.args, list) and len(layer_.args) >= 2 and isinstance(layer_.args[1], int):
#         output_index = layer_.args[1]

#     if output_index is None:
#         return None

#     return TupleOutputRef(source_node_name=node_refs[0], output_index=output_index)


# def append_unique(nodes: list[Node], node: Node) -> None:
#     if all(existing is not node for existing in nodes):
#         nodes.append(node)


# def resolve_attr_ref(attr_name: str, attr_value: Any, bindings: dict[str, Node]) -> Any:
#     if not isinstance(attr_value, AttrRef):
#         return attr_value

#     if attr_value.node not in bindings:
#         if attr_value.required:
#             raise ValueError(f"Output attr {attr_name} references unbound node {attr_value.node}")
#         return attr_value.default

#     bound_node = bindings[attr_value.node]
#     if attr_value.attr not in bound_node.normalized_attrs:
#         if attr_value.required:
#             raise ValueError(f"Output attr {attr_name} references missing attr {attr_value.attr}")
#         return attr_value.default

#     return bound_node.normalized_attrs[attr_value.attr]


# # TODO: Optimize this function further.
# def build_fusion_result(fusion: FusionGraph, bindings: dict[str, Node]) -> FusionResult:
#     root_node = bindings[fusion.root]
#     internal_nodes = []

#     for node in bindings.values():
#         append_unique(internal_nodes, node)

#     internal_node_ids = {id(node) for node in internal_nodes}
#     output_attrs = {}

#     for attr_name, attr_value in fusion.output_attrs.items():
#         output_attrs[attr_name] = resolve_attr_ref(attr_name, attr_value, bindings)

#     external_input = []
#     input_role_nodes = {}

#     for input_spec in fusion.inputs:
#         if input_spec.node not in bindings:
#             raise ValueError(f"Fusion input references unbound node {input_spec.node}")

#         node = bindings[input_spec.node]
#         if input_spec.parent_index < 0 or input_spec.parent_index >= len(node.parents):
#             raise ValueError(f"Fusion input has invalid parent index {input_spec.parent_index}")

#         parent = node.parents[input_spec.parent_index]
#         append_unique(external_input, parent)

#         if input_spec.role is not None:
#             input_role_nodes.setdefault(input_spec.role, [])
#             append_unique(input_role_nodes[input_spec.role], parent)

#     variadic_external_inputs = {}

#     for variadic_input in fusion.variadic_inputs:
#         if variadic_input.node not in bindings:
#             raise ValueError(f"Variadic fusion input references unbound node {variadic_input.node}")

#         node = bindings[variadic_input.node]
#         end = variadic_input.end_parent_index if variadic_input.end_parent_index is not None else len(node.parents)
#         parents = node.parents[variadic_input.start_parent_index:end]

#         if len(parents) < variadic_input.min_count:
#             raise ValueError(f"Variadic fusion input {variadic_input.node} has too few parents")

#         if variadic_input.max_count is not None and len(parents) > variadic_input.max_count:
#             raise ValueError(f"Variadic fusion input {variadic_input.node} has too many parents")

#         role = variadic_input.role or variadic_input.node
#         variadic_external_inputs[role] = []

#         for parent in parents:
#             append_unique(external_input, parent)
#             append_unique(variadic_external_inputs[role], parent)

#             if variadic_input.role is not None:
#                 input_role_nodes.setdefault(variadic_input.role, [])
#                 append_unique(input_role_nodes[variadic_input.role], parent)

#     external_consumers = []

#     for node in internal_nodes:
#         for child in node.children:
#             if id(child) in internal_node_ids:
#                 continue

#             if node is not root_node or not fusion.allow_root_external_children:
#                 raise ValueError("Cannot fuse node with external consumers outside the fusion boundary")

#             append_unique(external_consumers, child)

#     edges_consumed = []

#     for child in internal_nodes:
#         for parent in child.parents:
#             if id(parent) in internal_node_ids:
#                 edges_consumed.append(f"{parent.name}->{child.name}")

#     fused_node = Node(
#         name=f"{fusion.target}_{root_node.name}",
#         underlying_op=fusion.target,
#         normalized_attrs=output_attrs,
#         parents=external_input,
#         children=external_consumers,
#         tensor_meta=root_node.tensor_meta,
#         tuple_output_metas=root_node.tuple_output_metas.copy(),
#         value_kind=ValueKind.ACTIVATION,
#         tuple_source=None,
#         raw_layer=None,
#     )

#     return FusionResult(
#         node=fused_node,
#         bindings=bindings,
#         internal_nodes=internal_nodes,
#         external_input=external_input,
#         external_consumers=external_consumers,
#         edges_consumed=edges_consumed,
#         fusion=fusion,
#         variadic_external_inputs=variadic_external_inputs,
#         input_role_nodes=input_role_nodes,
#     )

# # TODO: Optimize this function further.
# def generate_graph(map: CVModels.LayerMap) -> Graph:
#     temp_map: dict[str, Node] = {}
#     temp_source = []

#     for layer_ in map.nodes:
#         tensor_meta, tuple_output_metas = extract_tensor_metas(layer_)
#         temp = Node(
#             name=layer_.name,
#             normalized_attrs=extract_attrs(layer_),
#             underlying_op=layer_.target,
#             parents=[],
#             children=[],
#             tensor_meta=tensor_meta,
#             tuple_output_metas=tuple_output_metas,
#             value_kind=infer_value_kind(layer_),
#             tuple_source=extract_tuple_source(layer_),
#             raw_layer=layer_,
#         )
#         temp_map[layer_.name] = temp
#         temp_source.append(temp) if layer_.node_type == "output" else None

#     for layer_ in map.nodes:
#         for name in layer_.users:
#             temp_map[layer_.name].children.append(temp_map[name])

#         for name in extract_node_names(layer_.args):
#             temp_map[layer_.name].parents.append(temp_map[name])

#     for node in temp_map.values():
#         if node.tuple_source is None:
#             continue

#         source_node = temp_map.get(node.tuple_source.source_node_name)
#         output_index = node.tuple_source.output_index
#         if source_node is None or output_index < 0 or output_index >= len(source_node.tuple_output_metas):
#             continue

#         node.tensor_meta = source_node.tuple_output_metas[output_index]

#     return Graph(name_map=temp_map, source_nodes=temp_source, consumed_ids=set(), fusedNodes=[])


# # TODO: Optimize this function further.
# def update_graph(graph: Graph) -> None:
#     replacement_by_id: dict[int, Node] = {}
#     fused_nodes = []

#     for fusion_result in graph.fusedNodes:
#         append_unique(fused_nodes, fusion_result.node)

#         for internal_node in fusion_result.internal_nodes:
#             internal_id = id(internal_node)
#             existing_replacement = replacement_by_id.get(internal_id)

#             if existing_replacement is not None and existing_replacement is not fusion_result.node:
#                 raise ValueError(f"Node {internal_node.name} is consumed by multiple fusions")

#             replacement_by_id[internal_id] = fusion_result.node

#     def resolve_node(node: Node) -> Node:
#         seen_ids = set()

#         while id(node) in replacement_by_id and id(node) not in seen_ids:
#             seen_ids.add(id(node))
#             node = replacement_by_id[id(node)]

#         return node

#     final_nodes = []

#     for node in graph.name_map.values():
#         if id(node) not in replacement_by_id:
#             append_unique(final_nodes, node)

#     for node in fused_nodes:
#         append_unique(final_nodes, node)

#     new_name_map = {}

#     for node in final_nodes:
#         if node.name in new_name_map and new_name_map[node.name] is not node:
#             raise ValueError(f"Duplicate node name after fusion: {node.name}")

#         new_name_map[node.name] = node

#     final_node_ids = {id(node) for node in final_nodes}
#     fusion_result_by_node_id = {id(fusion_result.node): fusion_result for fusion_result in graph.fusedNodes}

#     for node in final_nodes:
#         fusion_result = fusion_result_by_node_id.get(id(node))
#         old_parents = fusion_result.external_input if fusion_result is not None else node.parents
#         new_parents = []

#         for parent in old_parents:
#             resolved_parent = resolve_node(parent)

#             if resolved_parent is node:
#                 continue

#             if id(resolved_parent) not in final_node_ids:
#                 continue

#             new_parents.append(resolved_parent)

#         node.parents = new_parents
#         node.children = []

#         if node.tuple_source is not None:
#             source_node = graph.name_map.get(node.tuple_source.source_node_name)
#             if source_node is not None:
#                 resolved_source = resolve_node(source_node)
#                 if resolved_source is not source_node:
#                     node.tuple_source = TupleOutputRef(
#                         source_node_name=resolved_source.name,
#                         output_index=node.tuple_source.output_index,
#                     )

#     for node in final_nodes:
#         for parent in node.parents:
#             append_unique(parent.children, node)

#     new_source_nodes = []

#     for source_node in graph.source_nodes:
#         resolved_source = resolve_node(source_node)

#         if id(resolved_source) in final_node_ids:
#             append_unique(new_source_nodes, resolved_source)

#     graph.name_map = new_name_map
#     graph.source_nodes = new_source_nodes
#     graph.consumed_ids.clear()
#     graph.fusedNodes.clear()