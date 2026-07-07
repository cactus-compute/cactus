from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class FusionPatter:
    target:str
    ops:tuple[str]
    path:tuple[int]
    




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
