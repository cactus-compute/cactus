from dataclasses import dataclass, field
from typing import Any
import constants
import models


@dataclass(slots=True)
class FusionPattern:
    target:str
    ops:tuple[str, ...]
    path:tuple[int, ...]
    required_attrs:dict[int, dict[str, Any]]
    input_refs:tuple[tuple[int, int],...]
    shared_input_refs:tuple[tuple[tuple[int, int], tuple[int, int]], ...]


OPS_MAP: dict[str, list[FusionPattern]] = {}

def len_match(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    return len(fusion.ops) == len(nodes)

def match_ops(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    for i in range(len(fusion.ops)):
        if(fusion.ops[i] != nodes[i].layer.target):
            return False
    return True


def match_path(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    for i, parent_index in enumerate(fusion.path):
        if parent_index < 0 or parent_index >= len(nodes[i].parents):
            return False

        if nodes[i].parents[parent_index].layer.name != nodes[i + 1].layer.name:
            return False

    return True

def match_attrs(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    for node_index, required_attrs in fusion.required_attrs.items():
        if node_index < 0 or node_index >= len(nodes):
            return False

        node_attrs = nodes[node_index].normalized_attrs
        for attr_name, expected_value in required_attrs.items():
            if attr_name not in node_attrs:
                return False

            if node_attrs[attr_name] != expected_value:
                return False

    return True


def match_input_refs(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    for node_index, parent_index in fusion.input_refs:
        if node_index < 0 or node_index >= len(nodes):
            return False

        if parent_index < 0 or parent_index >= len(nodes[node_index].parents):
            return False

    return True


def match_shared_input_refs(fusion: FusionPattern, nodes: list[models.Node]) -> bool:
    for left_ref, right_ref in fusion.shared_input_refs:
        left_node_index, left_parent_index = left_ref
        right_node_index, right_parent_index = right_ref

        if left_node_index < 0 or left_node_index >= len(nodes):
            return False

        if right_node_index < 0 or right_node_index >= len(nodes):
            return False

        if left_parent_index < 0 or left_parent_index >= len(nodes[left_node_index].parents):
            return False

        if right_parent_index < 0 or right_parent_index >= len(nodes[right_node_index].parents):
            return False

        left_parent = nodes[left_node_index].parents[left_parent_index]
        right_parent = nodes[right_node_index].parents[right_parent_index]

        if left_parent.layer.name != right_parent.layer.name:
            return False

    return True



def fusion_match(ops:str, nodes: list[models.Node]) -> FusionPattern | None:
    if ops in OPS_MAP:
        for fusion in OPS_MAP[ops]:
            if match(fusion, nodes):
                return fusion
        
        return None




        



    




# @dataclass(frozen=True)
# class FusionPattern:
#     """Description of an operation sequence that can become one fused op."""

#     target: str
#     operations: tuple[str, ...]
#     path_inputs: tuple[int, ...]
#     required_attrs: dict[int, dict[str, Any]] = field(default_factory=dict)
#     same_inputs: tuple[tuple[InputRef, InputRef], ...] = ()
#     input_order: tuple[InputRef, ...] = ()
#     output_attrs: dict[str, AttrRef] = field(default_factory=dict)

#     @property
#     def sequence(self) -> str:
#         return "->".join(self.operations)

#     def __post_init__(self) -> None:
#         if not self.operations:
#             raise ValueError("operations cannot be empty")

#         if len(self.path_inputs) != len(self.operations) - 1:
#             raise ValueError("path_inputs must contain one entry between each operation")

#         for node_index in self.required_attrs:
#             if node_index < 0 or node_index >= len(self.operations):
#                 raise ValueError(f"required_attrs contains invalid node index {node_index}")

#         input_refs = [input_ref for input_pair in self.same_inputs for input_ref in input_pair]
#         input_refs.extend(self.input_order)
        
#         for node_index, input_index in input_refs:
#             if node_index < 0 or node_index >= len(self.operations):
#                 raise ValueError(f"input reference contains invalid node index {node_index}")
#             if input_index < 0:
#                 raise ValueError("input reference indexes cannot be negative")

#         for node_index, _ in self.output_attrs.values():
#             if node_index < 0 or node_index >= len(self.operations):
#                 raise ValueError(f"output_attrs contains invalid node index {node_index}")


# @dataclass(frozen=True)
# class FusionMatch:
#     """Concrete graph nodes and boundary values matched by a FusionPattern."""

#     root_id: str
#     target: str
#     matched_ids: tuple[str, ...]
#     external_input_ids: tuple[str, ...]
#     attrs: dict[str, Any] = field(default_factory=dict)


# RMSNORM_FUSION = FusionPattern(
#     target="rmsnorm",
#     operations=(
#         "aten.mul.Tensor",
#         "aten.mul.Tensor",
#         "aten.pow.Tensor_Scalar",
#         "aten.add.Tensor",
#         "aten.mean.dim",
#         "aten.pow.Tensor_Scalar",
#     ),
#     path_inputs=(0, 1, 0, 0, 0),
#     required_attrs={
#         2: {"exponent": -0.5},
#         3: {"scalar": 1e-06},
#         4: {"dim": [-1], "keepdim": True},
#         5: {"exponent": 2},
#     },
#     same_inputs=(((1, 0), (5, 0)),),
#     input_order=((1, 0), (0, 1)),
#     output_attrs={"epsilon": (3, "scalar")},
# )


# LINEAR_RESHAPE_FUSION = FusionPattern(
#     target="linear_reshape",
#     operations=(
#         "aten.view.default",
#         "aten.mm.default",
#     ),
#     path_inputs=(0,),
#     input_order=((1, 0), (1, 1)),
#     output_attrs={"shape": (0, "shape")},
# )
