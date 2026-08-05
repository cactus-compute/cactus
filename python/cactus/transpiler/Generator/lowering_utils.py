from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import constants
from . import models
from .errors import UnsupportedLoweringError
from ..IR import models as IRModels


SIZE_T_MAX = (1 << 64) - 1


def write_constant_tensor(
    context: models.GenerationContext,
    node: IRModels.Node,
    values: list[float],
    shape: tuple[int, ...],
    precision: int,
) -> str:
    constants_dir = context.component.output_path.parent / "constants"
    constants_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{models.sanitize_component_name(context.component.name)}__{models.sanitize_component_name(node.name)}.weights"
    path = constants_dir / filename
    data = constant_data_bytes(values, precision)
    write_cactus_tensor_file(path, shape, precision, data)
    return constant_binding_path(context, path)


def constant_data_bytes(values: list[float], precision: int) -> bytes:
    if precision == 1:
        return b"".join(struct.pack("<e", float(value)) for value in values)

    if precision == 2:
        return b"".join(struct.pack("<f", float(value)) for value in values)

    return bytes(int(value) & 0xFF for value in values)


def write_cactus_tensor_file(path: Path, shape: tuple[int, ...], precision: int, data: bytes) -> None:
    alignment = 32
    header_size = 84
    ndim = len(shape)
    original_n = shape[0] if shape else 0

    with path.open("wb") as f:
        f.write(struct.pack("<I", 0x54434143))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", alignment))
        f.write(struct.pack("<I", ndim))

        for index in range(4):
            f.write(struct.pack("<Q", int(shape[index]) if index < ndim else 0))

        f.write(struct.pack("<I", int(precision)))
        f.write(struct.pack("<Q", len(data)))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", int(original_n)))
        f.write(b"\0" * ((alignment - (header_size % alignment)) % alignment))
        f.write(data)


def write_dequantized_int8_weight_as_fp16(
    context: models.GenerationContext,
    node: IRModels.Node,
    record: models.WeightRecord,
) -> str:
    if context.config.weights_dir is None or record.output_name is None:
        raise UnsupportedLoweringError(f"{node.name}: missing converted weight path for FP16 dequant fallback")

    source_path = context.config.weights_dir / record.output_name
    shape, values = read_dequantized_int8_weight_fp16(source_path)
    constants_dir = context.component.output_path.parent / "constants"
    constants_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{models.sanitize_component_name(context.component.name)}__"
        f"{models.sanitize_component_name(node.name)}__dequant_fp16.weights"
    )
    path = constants_dir / filename
    write_cactus_tensor_file(path, shape, int(context.graph.FP16), values.tobytes())
    return constant_binding_path(context, path)


def read_dequantized_int8_weight_fp16(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as f:
        header = f.read(84)

        if len(header) != 84 or header[:4] != b"CACT":
            raise UnsupportedLoweringError(f"{path}: invalid Cactus tensor header")

        fields = struct.unpack("<IIIQQQQIQQIIQ", header[4:84])
        _, alignment, ndim, d0, d1, d2, d3, precision, data_bytes, scales_bytes, group_size, _, original_n = fields

        if int(precision) != 0:
            raise UnsupportedLoweringError(f"{path}: expected INT8 tensor for FP16 dequant fallback")

        shape = tuple(int(dim) for dim in (d0, d1, d2, d3)[: int(ndim)])
        if not shape:
            shape = (1,)

        scales_offset = aligned_offset(84, int(alignment))
        data_offset = aligned_offset(scales_offset + int(scales_bytes), int(alignment))
        f.seek(scales_offset)
        scales = np.frombuffer(f.read(int(scales_bytes)), dtype=np.float16).astype(np.float32)
        f.seek(data_offset)
        quantized = np.frombuffer(f.read(int(data_bytes)), dtype=np.int8).copy()

    count = math.prod(shape)
    if quantized.size < count:
        raise UnsupportedLoweringError(f"{path}: INT8 payload is smaller than tensor shape")

    quantized = quantized[:count].reshape(shape).astype(np.float32)

    if int(scales_bytes) > 0 and int(group_size) > 0 and scales.size > 0:
        if len(shape) == 1:
            group_ids = np.arange(count) // int(group_size)
            dequantized = (quantized.reshape(-1) * scales[group_ids]).reshape(shape)
        else:
            rows = int(shape[0])
            inner = count // rows
            if inner % int(group_size) != 0:
                raise UnsupportedLoweringError(f"{path}: grouped INT8 inner size is not divisible by group size")

            groups = inner // int(group_size)
            if scales.size < rows * groups:
                raise UnsupportedLoweringError(f"{path}: grouped INT8 scale metadata is too small")

            q2 = quantized.reshape(rows, inner)
            scales2 = scales[: rows * groups].reshape(rows, groups)
            cols = np.arange(inner) // int(group_size)
            dequantized = (q2 * scales2[:, cols]).reshape(shape)
    else:
        dequantized = quantized

    if original_n and len(shape) > 0 and int(original_n) < shape[0]:
        dequantized = dequantized[: int(original_n)]
        shape = tuple(int(dim) for dim in dequantized.shape)

    return shape, np.ascontiguousarray(dequantized.astype(np.float16))


def constant_binding_path(context: models.GenerationContext, path: Path) -> str:
    try:
        return str(path.relative_to(context.config.output_dir.parent))
    except ValueError:
        return str(path)


def precision_name(graph: Any, precision: int) -> str:
    for name in ("INT8", "FP16", "FP32", "CQ1", "CQ2", "CQ3", "CQ4"):
        if int(getattr(graph, name)) == int(precision):
            return name

    return str(precision)


def fp16_tensor(context: models.GenerationContext, value: Any) -> Any:
    fp16 = int(context.graph.FP16)
    return value if getattr(value, "dtype", None) == fp16 else context.graph.precision_cast(value, fp16)


def numeric_attr(node: IRModels.Node, *names: str, default: Any | None = None) -> Any | None:
    for name in names:
        value = node.attrs.get(name)

        if value is not None:
            return value

    return default


def require_input_count(context: models.GenerationContext, node: IRModels.Node, min_count: int) -> tuple[Any, ...]:
    inputs = context.inputs_for(node)
    require_len(node, inputs, min_count)
    return inputs


def require_at_least_one_input(context: models.GenerationContext, node: IRModels.Node) -> tuple[Any, ...]:
    return require_input_count(context, node, 1)


def require_len(node: IRModels.Node, inputs: tuple[Any, ...], min_count: int) -> None:
    if len(inputs) < min_count:
        raise unsupported_arity(node, len(inputs), f"at least {min_count} inputs")


def unsupported_arity(node: IRModels.Node, actual: int, expected: str) -> UnsupportedLoweringError:
    return UnsupportedLoweringError(f"{node.name}: {node.target} got {actual} lowered inputs; expected {expected}")


def tensor_dtype(node: IRModels.Node) -> str | None:
    if isinstance(node.tensor_output_meta, dict):
        dtype = node.tensor_output_meta.get("dtype")

        if dtype is not None:
            return str(dtype)

    if isinstance(node.tensor_output_meta, list) and node.tensor_output_meta:
        first_meta = node.tensor_output_meta[0]

        if isinstance(first_meta, dict):
            dtype = first_meta.get("dtype")

            if dtype is not None:
                return str(dtype)

    return None


def cactus_precision(graph: Any, dtype: str | None) -> int:
    precision_name = constants.DTYPE_TO_PRECISION.get(str(dtype), constants.DEFAULT_INPUT_PRECISION)
    return int(getattr(graph, precision_name))


def cast_to_precision(context: models.GenerationContext, value: Any, precision: int) -> Any:
    if getattr(value, "dtype", precision) == precision:
        return value

    return context.graph.precision_cast(value, precision)


def graph_input_shape(node: IRModels.Node) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    shape = meta_shape(node)

    if not shape:
        raw_shape = node.tensor_output_meta.get("shape") if isinstance(node.tensor_output_meta, dict) else None
        if raw_shape == [] or raw_shape == ():
            return (1,), (False,)

        return (), ()

    dims: list[int] = []
    dynamic_dims: list[bool] = []

    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            dims.append(dim)
            dynamic_dims.append(False)
            continue

        if isinstance(dim, str) and dim.isdigit():
            dims.append(int(dim))
            dynamic_dims.append(False)
            continue

        dims.append(1)
        dynamic_dims.append(True)

    return tuple(dims), tuple(dynamic_dims)


def meta_shape(node: IRModels.Node) -> tuple[Any, ...]:
    if isinstance(node.tensor_output_meta, dict):
        shape = node.tensor_output_meta.get("shape")

        if isinstance(shape, list):
            return tuple(shape)

        if isinstance(shape, tuple):
            return shape

    if isinstance(node.tensor_output_meta, list) and node.tensor_output_meta:
        first_meta = node.tensor_output_meta[0]

        if isinstance(first_meta, dict):
            shape = first_meta.get("shape")

            if isinstance(shape, list):
                return tuple(shape)

            if isinstance(shape, tuple):
                return shape

    return ()


def output_shape(node: IRModels.Node) -> tuple[int, ...]:
    shape, _ = graph_input_shape(node)

    if shape:
        return shape

    raise UnsupportedLoweringError(f"{node.name}: missing concrete output shape")


def concrete_dim(dim: Any) -> int | None:
    if isinstance(dim, int) and dim >= 0:
        return dim

    if isinstance(dim, str) and dim.isdigit():
        return int(dim)

    return None


def concrete_shape(shape: tuple[Any, ...]) -> tuple[int, ...] | None:
    dims = tuple(concrete_dim(dim) for dim in shape)

    if any(dim is None for dim in dims):
        return None

    return tuple(int(dim) for dim in dims)


def reduction_dropped_shape(shape: tuple[Any, ...], axis: int) -> tuple[Any, ...]:
    dropped = tuple(dim for index, dim in enumerate(shape) if index != axis)
    return dropped or (1,)


def shape_matches_tensor(node: IRModels.Node, expected_shape: tuple[int, ...]) -> bool:
    actual_shape = meta_shape(node)
    return tuple(actual_shape) == tuple(expected_shape)


def element_count(shape: tuple[Any, ...]) -> int | None:
    if not shape:
        return None

    count = 1

    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            return None

        count *= dim

    return count


def shape_attr(node: IRModels.Node) -> tuple[int, ...]:
    raw_shape = node.attrs.get("shape")

    if raw_shape is None:
        return output_shape(node)

    resolved_output_shape = output_shape(node)
    dims: list[int] = []

    for index, dim in enumerate(raw_shape):
        output_dim = resolved_output_shape[index] if index < len(resolved_output_shape) else None

        if isinstance(dim, int) and dim == 0 and isinstance(output_dim, int) and output_dim > 0:
            dims.append(output_dim)
        elif isinstance(dim, int) and dim >= 0:
            dims.append(dim)
        elif isinstance(dim, str) and dim.isdigit():
            dims.append(int(dim))
        elif output_dim is not None:
            dims.append(output_dim)
        else:
            raise UnsupportedLoweringError(f"{node.name}: cannot resolve shape dim {dim!r}")

    return tuple(dims)


def axis_attr(node: IRModels.Node, default: int | None = None) -> int | None:
    axis = node.attrs.get("axis", node.attrs.get("dim", node.attrs.get("arg_1", default)))

    if isinstance(axis, list):
        if len(axis) != 1:
            raise UnsupportedLoweringError(f"{node.name}: Cactus reduction lowering only supports one axis")
        return int(axis[0])

    if axis is None:
        return None

    return int(axis)


def scalar_attr(node: IRModels.Node, name: str) -> Any | None:
    value = node.attrs.get(name)

    if isinstance(value, dict) and "node" in value:
        return None

    if isinstance(value, list) and len(value) == 1:
        return value[0]

    return value


def scalar_weight_bound_value(context: models.GenerationContext, node: IRModels.Node) -> float | None:
    resolver = context.component.weight_resolver

    if resolver is None:
        return None

    record = resolver.resolve(node.name)

    if record is None or record.output_name is None:
        return None

    if record.shape and math.prod(int(dim) for dim in record.shape) != 1:
        return None

    value = read_scalar_cactus_weight(resolver.weights_dir / record.output_name)

    if value is None:
        return None

    return float(value) * float(record.scale_factor)


def apply_inverse_weight_scale_for_parent(
    context: models.GenerationContext,
    node: IRModels.Node,
    value: Any,
    parent_index: int,
) -> Any:
    scale_factor = weight_scale_factor_for_parent(context, node, parent_index)

    if scale_factor is None or scale_factor == 1.0:
        return value

    return context.graph.scalar_multiply(value, 1.0 / scale_factor)


def weight_scale_factor_for_parent(context: models.GenerationContext, node: IRModels.Node, parent_index: int) -> float | None:
    if parent_index < 0 or parent_index >= len(node.parents):
        return None

    weight_node = source_weight_node(node.parents[parent_index])

    if weight_node is None:
        return None

    resolver = context.component.weight_resolver

    if resolver is None:
        return None

    record = resolver.resolve(weight_node.name)

    if record is None:
        return None

    return float(record.scale_factor or 1.0)


def source_weight_node(node: IRModels.Node) -> IRModels.Node | None:
    current = node

    while current.target in {"aten.t.default", "cactus.transpose"} and len(current.parents) == 1:
        current = current.parents[0]

    if current.value_kind not in constants.WEIGHT_VALUE_KINDS:
        return None

    return current


def read_scalar_cactus_weight(path: Path) -> float | None:
    if not path.exists():
        return None

    with path.open("rb") as f:
        header = f.read(84)

        if len(header) != 84 or header[:4] != b"CACT":
            return None

        fields = struct.unpack("<IIIQQQQIQQIIQ", header[4:84])
        _, alignment, ndim, d0, d1, d2, d3, precision, data_bytes, scales_bytes, _, _, _ = fields
        dims = (d0, d1, d2, d3)
        element_count = math.prod(int(dim) for dim in dims[:ndim]) if ndim else 1

        if element_count != 1:
            return None

        scales_offset = aligned_offset(84, alignment)
        data_offset = aligned_offset(scales_offset + scales_bytes, alignment)
        f.seek(data_offset)

        if precision == 1 and data_bytes >= 2:
            return float(struct.unpack("<e", f.read(2))[0])

        if precision == 2 and data_bytes >= 4:
            return float(struct.unpack("<f", f.read(4))[0])

    return None


def aligned_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        alignment = 32

    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def attr_value(node: IRModels.Node, name: str, default: Any) -> Any:
    value = node.attrs.get(name)
    return default if value is None else value


def epsilon_attr(node: IRModels.Node) -> float:
    return float(node.attrs.get("epsilon", node.attrs.get("eps", 1e-5)))


def required_int_attr(node: IRModels.Node, name: str) -> int:
    value = node.attrs.get(name)

    if value is None:
        raise UnsupportedLoweringError(f"{node.name}: missing required attr {name}")

    return int(value)


def has_attrs(node: IRModels.Node, names: tuple[str, ...]) -> bool:
    return all(node.attrs.get(name) is not None for name in names)


def first_int(value: Any, default: int) -> int:
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])

    return int(value)


def tuple_int_values(value: Any, default: int) -> tuple[int, int]:
    if value is None:
        return (default, default)

    if isinstance(value, (list, tuple)):
        if not value:
            return (default, default)

        if len(value) == 1:
            item = int(value[0])
            return (item, item)

        return (int(value[0]), int(value[1]))

    item = int(value)
    return (item, item)


def tuple_ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected list/tuple of ints, got {type(value).__name__}")

    return tuple(int(item) for item in value)


def parent_rank(node: IRModels.Node) -> int:
    if not node.parents:
        raise UnsupportedLoweringError(f"{node.name}: cannot infer parent rank without parents")

    rank = len(meta_shape(node.parents[0]))

    if rank == 0:
        raise UnsupportedLoweringError(f"{node.name}: cannot infer parent rank without parent shape metadata")

    return rank


def output_rank(node: IRModels.Node) -> int:
    return len(meta_shape(node))


def swap_permutation(rank: int, dim0: int, dim1: int) -> tuple[int, ...]:
    dim0 = normalize_dim(dim0, rank)
    dim1 = normalize_dim(dim1, rank)
    permutation = list(range(rank))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return tuple(permutation)


def normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank

    if dim < 0 or dim >= rank:
        raise ValueError(f"Dimension {dim} is outside rank {rank}")

    return dim


def slice_length(node: IRModels.Node, start: int) -> int:
    if "length" in node.attrs:
        return int(node.attrs["length"])

    end = node.attrs.get("end")

    if end is None:
        return open_slice_length(node, start)

    end = int(end)

    if end >= constants.OPEN_SLICE_END:
        return open_slice_length(node, start)

    return max(end - start, 0)


def open_slice_length(node: IRModels.Node, start: int) -> int:
    if not node.parents:
        return 0

    axis = int(node.attrs.get("axis", node.attrs.get("dim", 0)))
    parent_shape = meta_shape(node.parents[0])

    if axis < 0:
        axis += len(parent_shape)

    if axis < 0 or axis >= len(parent_shape):
        return 0

    axis_size = parent_shape[axis]

    if not isinstance(axis_size, int):
        return 0

    return max(axis_size - start, 0)


def binary_method_to_scalar_method(method: str) -> str:
    return {
        "add": "scalar_add",
        "subtract": "scalar_subtract",
        "multiply": "scalar_multiply",
        "divide": "scalar_divide",
        "not_equal": "scalar_not_equal",
    }[method]


def ensure_tensor_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]
