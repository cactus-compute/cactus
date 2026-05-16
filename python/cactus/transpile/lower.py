from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
import os
from typing import Any

import numpy as np
import torch

from cactus.transpile.runtime_compat import Graph
from cactus.transpile.runtime_compat import Tensor
from cactus.transpile.capture_pytorch import CapturedModel
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.graph_ir import verify_ir
from cactus.transpile.optimize_graph import optimize_graph
from cactus.transpile.weight_compat import ensure_binding_compatible
from cactus.transpile.weight_binding import WeightBinding


_DYNAMIC_KV_POSITION_OFFSET = (1 << 64) - 1


@dataclass
class TranspiledGraph:
    graph: Graph
    runtime_inputs: list[Tensor]
    bound_constants: list[Tensor]
    bound_constant_bindings: list[dict[str, object]]
    bound_constant_value_ids: dict[int, str]
    outputs: list[Tensor]
    cache_state_tensors: list[tuple[str, Tensor, Tensor]] = field(default_factory=list)

    def set_input(self, index: int, data: Any, *, dtype: int | None = None) -> None:
        if index < 0 or index >= len(self.runtime_inputs):
            raise IndexError(
                f"runtime input index out of range: {index} (have {len(self.runtime_inputs)})"
            )
        self.graph.set_input(self.runtime_inputs[index], data, dtype=dtype)

    def set_inputs(self, inputs: list[Any] | tuple[Any, ...]) -> None:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"expected {len(self.runtime_inputs)} runtime inputs, got {len(inputs)}"
            )
        for index, value in enumerate(inputs):
            self.set_input(index, value)

    def execute(self) -> list[Tensor]:
        self.graph.execute()
        return self.outputs


@dataclass
class BroadcastAlias:
    tensor: Tensor
    logical_shape: tuple[int, ...]
    kind: str


def _is_quantized_runtime_dtype(dtype: int) -> bool:
    return int(dtype) in {
        int(Graph.INT8),
        int(getattr(Graph, "CQ1", 3)),
        int(getattr(Graph, "CQ2", 4)),
        int(getattr(Graph, "CQ3", 5)),
        int(getattr(Graph, "CQ4", 6)),
    }


def transpile_captured(captured: CapturedModel) -> TranspiledGraph:
    return transpile_ir(captured.ir_graph)


def transpile_ir(ir: IRGraph) -> TranspiledGraph:
    verify_ir(ir)
    canonicalize_exported_graph(ir)
    optimize_graph(ir)
    return transpile_preoptimized_ir(ir)


def transpile_preoptimized_ir(ir: IRGraph) -> TranspiledGraph:
    verify_ir(ir)
    g = Graph()
    g._transpile_materialized_constants = []  # type: ignore[attr-defined]
    trace_lower = os.environ.get("CACTUS_LOWER_TRACE_NODES", "0") != "0"
    env: dict[str, Any] = {}
    runtime_inputs: list[Tensor] = []
    bound_constants: list[Tensor] = []
    bound_constant_bindings: list[dict[str, object]] = []
    bound_constant_value_ids: dict[int, str] = {}

    for value_id in ir.inputs:
        value = ir.values[value_id]
        tensor = _lower_input_value(g, value)
        env[value_id] = tensor
        runtime_inputs.append(tensor)

    for value_id, const in ir.constants.items():
        value = ir.values[value_id]
        binding = _lookup_weight_binding(value)
        if binding is not None:
            binding = ensure_binding_compatible(binding, source_tensor=const)
        lowered_const = _lower_constant_value(g, value, const, binding=binding)
        env[value_id] = lowered_const
        if isinstance(lowered_const, Tensor):
            bound_constants.append(lowered_const)
            bound_constant_value_ids[int(lowered_const.id)] = str(value_id)
            if binding is not None:
                bound_constant_bindings.append(
                    {
                        "node_id": int(lowered_const.id),
                        "value_id": str(value_id),
                        "path": binding.path,
                        "kind": binding.kind,
                        "source_name": binding.source_name,
                    }
                )

    for node_id in ir.order:
        node = ir.nodes[node_id]
        outputs = _lower_ir_node(g, node, env, ir)
        if len(outputs) != len(node.outputs):
            raise ValueError(
                f"node {node.id} produced {len(outputs)} outputs, expected {len(node.outputs)}"
            )
        for output_id, tensor in zip(node.outputs, outputs):
            env[output_id] = tensor
            if trace_lower and isinstance(tensor, Tensor):
                print(
                    "lower_trace"
                    f" cactus_id={int(tensor.id)}"
                    f" ir_node={node.id}"
                    f" op={node.op}"
                    f" output={output_id}"
                    f" shape={tuple(int(dim) for dim in tensor.shape)}",
                    flush=True,
                )

    outputs = [env[value_id] for value_id in ir.outputs]
    seen_bound_constant_ids = {int(tensor.id) for tensor in bound_constants}
    for tensor in getattr(g, "_transpile_materialized_constants", []):
        tensor_id = int(tensor.id)
        if tensor_id in seen_bound_constant_ids:
            continue
        bound_constants.append(tensor)
        seen_bound_constant_ids.add(tensor_id)
    cache_state_tensors = list(env.get("__internal_kv_cache_state_entries", []))
    return TranspiledGraph(
        graph=g,
        runtime_inputs=runtime_inputs,
        bound_constants=bound_constants,
        bound_constant_bindings=bound_constant_bindings,
        bound_constant_value_ids=bound_constant_value_ids,
        outputs=outputs,
        cache_state_tensors=cache_state_tensors,
    )


def _lower_input_value(g: Graph, value: IRValue) -> Tensor:
    if value.shape is None or value.dtype is None:
        raise ValueError(f"IR input missing shape or dtype: {value.id}")
    return g.input(shape=value.shape, dtype=_map_ir_dtype(value.dtype))


def _should_lower_attention_with_internal_kv_cache(ir: IRGraph, node: IRNode) -> bool:
    if not bool(ir.meta.get("use_internal_kv_cache", False)):
        return False
    if node.op not in {"attention", "scaled_dot_product_attention"}:
        return False
    component = str(ir.meta.get("component", "") or "").strip().lower()
    return component in {"decoder_step", "decoder_prefill_chunk"}


def _lower_attention_with_internal_kv_cache(
    g: Graph,
    ir: IRGraph,
    env: dict[str, Any],
    node: IRNode,
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output_layout: str,
) -> list[Tensor]:
    key_shape = tuple(int(dim) for dim in key.shape)
    value_shape = tuple(int(dim) for dim in value.shape)
    if len(key_shape) != 4 or len(value_shape) != 4:
        raise NotImplementedError(
            f"cached attention expects key/value as [batch, seq, heads, dim], got {key_shape} and {value_shape}"
        )
    if int(key_shape[0]) != 1:
        raise NotImplementedError("cached attention currently expects batch size 1")

    cache_states: dict[str, tuple[Tensor, Tensor]] = env.setdefault("__internal_kv_cache_states", {})  # type: ignore[assignment]
    layer_key = str(
        node.meta.get("attention_layer_index")
        or node.meta.get("layer_index")
        or node.id
    )
    if layer_key not in cache_states:
        default_cache_len = max(512, int(key_shape[1]))
        max_cache_seq_len = int(ir.meta.get("max_cache_seq_len", default_cache_len) or default_cache_len)
        sink_size = int(ir.meta.get("cache_sink_size", 4) or 4)
        window_size = int(node.attrs.get("window_size", 0) or 0)
        k_cache = g.kv_cache_state(
            max_cache_seq_len,
            int(key_shape[2]),
            int(key_shape[3]),
            window_size=window_size,
            sink_size=sink_size,
        )
        v_cache = g.kv_cache_state(
            max_cache_seq_len,
            int(value_shape[2]),
            int(value_shape[3]),
            window_size=window_size,
            sink_size=sink_size,
        )
        cache_states[layer_key] = (k_cache, v_cache)
        cache_entries: list[tuple[str, Tensor, Tensor]] = env.setdefault("__internal_kv_cache_state_entries", [])  # type: ignore[assignment]
        cache_entries.append((layer_key, k_cache, v_cache))

    k_cache, v_cache = cache_states[layer_key]
    window_size = int(node.attrs.get("window_size", 0) or 0)
    sink_size = int(ir.meta.get("cache_sink_size", 4) or 4)
    g.kv_cache_append(key, k_cache, window_size=window_size, sink_size=sink_size)
    g.kv_cache_append(value, v_cache, window_size=window_size, sink_size=sink_size)
    out = g.attention_cached(
        query,
        key,
        value,
        k_cache,
        v_cache,
        scale=_resolve_attention_scale(node, query),
        position_offset=_DYNAMIC_KV_POSITION_OFFSET,
        window_size=window_size,
        v_head_dim=int(value_shape[3]),
    )
    if output_layout == "bthd":
        return [out]
    return [g.permute(out, (0, 2, 1, 3))]

def _lower_constant_value(
    g: Graph,
    value: IRValue,
    const: Any,
    *,
    binding: WeightBinding | None = None,
) -> Any:
    if binding is None:
        binding = _lookup_weight_binding(value)
    if binding is not None:
        _debug_mmap_binding(value.id, binding)
        return g.mmap_weights(binding.path)

    _debug_constant_fallback(value, const)

    if isinstance(const, torch.nn.Parameter):
        const = const.detach()
    if isinstance(const, torch.Tensor):
        tensor_value = const.detach().cpu()
    else:
        raise NotImplementedError(
            f"unsupported IR constant type for {value.id}: {type(const).__name__}"
        )

    if tensor_value.numel() == 1:
        return tensor_value.item()

    return _materialize_constant_tensor(g, tensor_value)


def _lookup_weight_binding(value: IRValue) -> WeightBinding | None:
    meta = getattr(value, "meta", None)
    if isinstance(meta, dict):
        path = meta.get("path")
        kind = meta.get("kind")
        source_name = meta.get("source_name")
        if isinstance(path, str) and isinstance(kind, str) and isinstance(source_name, str):
            return WeightBinding(path=path, kind=kind, source_name=source_name)
    return None


def _debug_mmap_binding(value_id: str, binding: WeightBinding) -> None:
    if os.environ.get("CACTUS_TRANSPILER_DEBUG_MMAP") != "1":
        return
    print(
        "[transpile:mmap] "
        f"value={value_id} "
        f"kind={binding.kind} "
        f"source={binding.source_name} "
        f"path={binding.path}"
    )


def _debug_constant_fallback(value: IRValue, const: Any) -> None:
    if os.environ.get("CACTUS_TRANSPILER_DEBUG_MMAP") != "1":
        return
    const_type = type(const).__name__
    shape = None
    dtype = None
    if isinstance(const, torch.Tensor):
        shape = tuple(const.shape)
        dtype = str(const.dtype)
    source_name = None
    if isinstance(value.meta, dict):
        source_name = value.meta.get("source_name")
    print(
        "[transpile:fallback] "
        f"value={value.id} "
        f"source={source_name} "
        f"const_type={const_type} "
        f"shape={shape} "
        f"dtype={dtype}"
    )


def _debug_embedding_lowering(node: IRNode, embedding_tensor: Tensor, indices_tensor: Tensor) -> None:
    if os.environ.get("CACTUS_TRANSPILER_DEBUG_MMAP") != "1":
        return
    print(
        "[transpile:embedding] "
        f"node={node.id} "
        f"embedding_id={embedding_tensor.id} "
        f"embedding_shape={tuple(embedding_tensor.shape)} "
        f"embedding_dtype={embedding_tensor.dtype} "
        f"indices_id={indices_tensor.id} "
        f"indices_shape={tuple(indices_tensor.shape)} "
        f"indices_dtype={indices_tensor.dtype}"
    )


def _matmul_with_quantized_rhs_legalization(
    g: Graph,
    lhs: Tensor,
    rhs: Tensor,
    *,
    pretransposed_rhs: bool = False,
    output_dtype: int | None = None,
) -> Tensor:
    if _is_quantized_runtime_dtype(rhs.dtype) and lhs.dtype == Graph.FP32:
        lhs = g.precision_cast(lhs, Graph.FP16)
    return g.matmul(lhs, rhs, pretransposed_rhs=pretransposed_rhs, output_dtype=output_dtype)


def _matmul_output_dtype(
    lhs: Tensor,
    rhs: Tensor,
    *,
    output_dtype: int | None,
) -> int | None:
    if output_dtype != Graph.FP16:
        return output_dtype
    if _is_quantized_runtime_dtype(rhs.dtype):
        return output_dtype
    if lhs.dtype == Graph.FP32 or rhs.dtype == Graph.FP32:
        return Graph.FP32
    return output_dtype


def _trim_padded_last_dim(g: Graph, tensor: Tensor, expected_shape: tuple[int, ...] | None) -> Tensor:
    if expected_shape is None:
        return tensor
    actual_shape = tuple(int(dim) for dim in tensor.shape)
    target_shape = tuple(int(dim) for dim in expected_shape)
    if actual_shape == target_shape:
        return tensor
    if (
        len(actual_shape) == len(target_shape)
        and len(actual_shape) >= 1
        and actual_shape[:-1] == target_shape[:-1]
        and actual_shape[-1] > target_shape[-1]
    ):
        return g.slice(tensor, len(actual_shape) - 1, 0, target_shape[-1])
    return tensor


def _legalize_for_transpose(g: Graph, x: Tensor) -> Tensor:
    return x


def _reshape_attention_output_for_linear(
    g: Graph,
    attn_out: Tensor,
    target_shape: tuple[int, ...],
) -> Tensor:
    actual_shape = tuple(int(dim) for dim in attn_out.shape)
    target_shape = tuple(int(dim) for dim in target_shape)
    if len(actual_shape) == 4 and len(target_shape) == 3 and actual_shape[0] == target_shape[0]:
        batch, dim1, dim2, dim3 = actual_shape
        _, target_seq, target_hidden = target_shape
        if dim1 == target_seq and dim2 * dim3 == target_hidden:
            return g.reshape(attn_out, target_shape)
        if dim2 == target_seq and dim1 * dim3 == target_hidden:
            return g.reshape(g.permute(attn_out, (0, 2, 1, 3)), target_shape)
    return g.reshape(attn_out, target_shape)


def _lower_ir_node(g: Graph, node: IRNode, env: dict[str, Any], ir: IRGraph) -> list[Any]:
    op = node.op

    if op == "arange":
        start = int(node.attrs.get("start", 0))
        end = int(node.attrs["end"])
        step = int(node.attrs.get("step", 1))
        dtype = _materialize_constant_torch_dtype(node.attrs.get("dtype"))
        tensor_value = torch.arange(start, end, step=step, dtype=dtype)
        return [_materialize_constant_tensor(g, tensor_value)]

    if op == "add":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "add")]

    if op == "add_clipped":
        lhs, rhs = _legalize_elementwise_binary_inputs(g, _tensor(env, node.inputs[0]), _tensor(env, node.inputs[1]))
        return [g.add_clipped(lhs, rhs)]

    if op == "subtract":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "subtract")]

    if op == "multiply":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "multiply")]

    if op == "multiply_inplace":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "multiply")]

    if op == "divide":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "divide")]

    if op == "not_equal":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "not_equal")]

    if op == "equal":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "equal")]

    if op == "greater":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "greater")]

    if op == "greater_equal":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "greater_equal")]

    if op == "less":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "less")]

    if op == "less_equal":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "less_equal")]

    if op == "logical_and":
        lhs = _lower_compare_op(g, env[node.inputs[0]], 0.0, "not_equal")
        rhs = _lower_compare_op(g, env[node.inputs[1]], 0.0, "not_equal")
        return [_lower_binary_op(g, lhs, rhs, "multiply")]

    if op == "logical_or":
        lhs = _lower_compare_op(g, env[node.inputs[0]], 0.0, "not_equal")
        rhs = _lower_compare_op(g, env[node.inputs[1]], 0.0, "not_equal")
        return [_lower_compare_op(g, _lower_binary_op(g, lhs, rhs, "add"), 0.0, "not_equal")]

    if op == "logical_not":
        return [_lower_compare_op(g, env[node.inputs[0]], 0.0, "equal")]

    if op == "where":
        return [_lower_where_op(g, node, env)]

    if op == "scalar_add":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_add(x, float(node.attrs["value"]))]

    if op == "scalar_subtract":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_subtract(x, float(node.attrs["value"]))]

    if op == "scalar_subtract_reverse":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_add(g.scalar_multiply(x, -1.0), float(node.attrs["value"]))]

    if op == "scalar_multiply":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_multiply(x, float(node.attrs["value"]))]

    if op == "scalar_divide":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_divide(x, float(node.attrs["value"]))]

    if op == "scalar_floor_divide":
        source = _tensor(env, node.inputs[0])
        x = _ensure_scalar_math_tensor(g, source)
        y = g.scalar_floor_divide(x, float(node.attrs["value"]))
        output_dtype = ir.values[node.outputs[0]].dtype
        if output_dtype is None:
            return [y]
        target_dtype = _map_ir_dtype(output_dtype)
        if y.dtype == target_dtype:
            return [y]
        return [g.precision_cast(y, target_dtype)]

    if op == "scalar_not_equal":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [g.scalar_not_equal(x, float(node.attrs["value"]))]

    if op == "scalar_equal":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        not_equal = g.scalar_not_equal(x, float(node.attrs["value"]))
        return [g.scalar_not_equal(not_equal, 1.0)]

    if op == "scalar_greater":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [_lower_compare_op(g, x, float(node.attrs["value"]), "greater")]

    if op == "scalar_greater_equal":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [_lower_compare_op(g, x, float(node.attrs["value"]), "greater_equal")]

    if op == "scalar_less":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [_lower_compare_op(g, x, float(node.attrs["value"]), "less")]

    if op == "scalar_less_equal":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        return [_lower_compare_op(g, x, float(node.attrs["value"]), "less_equal")]

    if op == "scalar_divide_reverse":
        raise NotImplementedError("scalar_divide_reverse is not directly supported by Cactus graph ops")

    if op == "precision_cast":
        target_dtype = node.attrs.get("dtype")
        if target_dtype is None:
            return [_tensor(env, node.inputs[0])]
        return [g.precision_cast(_tensor(env, node.inputs[0]), _map_ir_or_torch_dtype(target_dtype))]

    if op == "type_as":
        source = _tensor(env, node.inputs[0])
        target = _tensor(env, node.inputs[1])
        return [g.precision_cast(source, target.dtype)]

    if op == "abs":
        return [g.abs(_tensor(env, node.inputs[0]))]

    if op == "clamp":
        x = _ensure_fp16_tensor(g, _tensor(env, node.inputs[0]))
        min_value = node.attrs.get("min")
        max_value = node.attrs.get("max")
        lo = float(min_value) if min_value is not None and math.isfinite(float(min_value)) else -65504.0
        hi = float(max_value) if max_value is not None and math.isfinite(float(max_value)) else 65504.0
        return [g.clamp(x, lo, hi)]

    if op == "negate":
        return [g.scalar_multiply(_tensor(env, node.inputs[0]), -1.0)]

    if op == "pow":
        exponent = node.attrs.get("exponent")
        if exponent is None:
            raise NotImplementedError("tensor-tensor pow is not supported by Cactus graph ops")
        return [g.pow(_tensor(env, node.inputs[0]), float(exponent))]

    if op == "scalar_exp":
        return [g.scalar_exp(_tensor(env, node.inputs[0]))]

    if op == "scalar_sqrt":
        return [g.scalar_sqrt(_tensor(env, node.inputs[0]))]

    if op == "scalar_log":
        return [g.scalar_log(_tensor(env, node.inputs[0]))]

    if op == "scalar_cos":
        return [g.scalar_cos(_tensor(env, node.inputs[0]))]

    if op == "scalar_sin":
        return [g.scalar_sin(_tensor(env, node.inputs[0]))]

    if op in {"reshape", "view"}:
        source = env[node.inputs[0]]
        if isinstance(source, BroadcastAlias):
            input_shape = tuple(source.logical_shape)
        else:
            input_shape = tuple(_tensor(env, node.inputs[0]).shape)
        target_shape = _resolve_reshape_shape(input_shape, tuple(node.attrs["shape"]))
        if isinstance(source, BroadcastAlias):
            if source.kind != "gqa_repeat_kv":
                raise NotImplementedError(f"unsupported broadcast alias kind in reshape: {source.kind}")
            base_shape = tuple(source.tensor.shape)
            if len(base_shape) != 4:
                raise NotImplementedError(f"gqa_repeat_kv alias requires 4D base tensor, got {base_shape}")
            if len(target_shape) != 4:
                raise NotImplementedError(f"gqa_repeat_kv reshape requires 4D target shape, got {target_shape}")
            batch, kv_heads, seq_len, head_dim = base_shape
            if target_shape[0] != batch or target_shape[2] != seq_len or target_shape[3] != head_dim:
                raise NotImplementedError(
                    f"gqa_repeat_kv reshape mismatch: base {base_shape}, target {target_shape}"
                )
            if target_shape[1] % max(kv_heads, 1) != 0:
                raise NotImplementedError(
                    f"gqa_repeat_kv target head count must be a multiple of kv heads: {base_shape} -> {target_shape}"
                )
            return [BroadcastAlias(source.tensor, target_shape, source.kind)]
        return [g.reshape(_tensor(env, node.inputs[0]), target_shape)]

    if op == "flatten":
        return [
            g.flatten(
                _tensor(env, node.inputs[0]),
                start_dim=int(node.attrs["start_dim"]),
                end_dim=int(node.attrs["end_dim"]),
            )
        ]

    if op == "unsqueeze":
        x = _tensor(env, node.inputs[0])
        target_shape = node.attrs.get("shape")
        if target_shape is None:
            dim = _normalize_dim(int(node.attrs["dim"]), len(x.shape) + 1)
            shape_list = list(x.shape)
            shape_list.insert(dim, 1)
            target_shape = tuple(shape_list)
        return [g.reshape(x, tuple(target_shape))]

    if op == "expand":
        matched_base = _match_gqa_expand_alias(ir, node)
        if matched_base is not None:
            return [BroadcastAlias(_tensor(env, matched_base), tuple(node.attrs["shape"]), "gqa_repeat_kv")]
        x = _tensor(env, node.inputs[0])
        target_shape = _resolve_expand_shape(tuple(x.shape), tuple(node.attrs["shape"]))
        if target_shape == tuple(x.shape):
            return [x]
        return [g.expand(x, target_shape)]

    if op == "repeat":
        x = _tensor(env, node.inputs[0])
        repeats = tuple(int(value) for value in node.attrs["repeats"])
        return [_lower_repeat(g, x, repeats)]

    if op == "one_hot":
        indices = _tensor(env, node.inputs[0])
        num_classes = int(node.attrs["num_classes"])
        eye = torch.eye(num_classes, dtype=torch.float16)
        embedding = _materialize_constant_tensor(g, eye)
        return [g.embedding_from_tensor(embedding, indices)]

    if op == "transpose":
        x = _legalize_for_transpose(g, _tensor(env, node.inputs[0]))
        dim0 = _normalize_dim(int(node.attrs["dim0"]), len(x.shape))
        dim1 = _normalize_dim(int(node.attrs["dim1"]), len(x.shape))
        rank = len(x.shape)
        permutation = list(range(rank))
        permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
        if rank == 2 and permutation == [1, 0]:
            return [g.transpose(x)]
        return [g.permute(x, permutation)]

    if op == "permute":
        x = _legalize_for_transpose(g, _tensor(env, node.inputs[0]))
        permutation = tuple(_normalize_dim(int(dim), len(x.shape)) for dim in node.attrs["permutation"])
        if len(permutation) == 2 and permutation == (1, 0):
            return [g.transpose(x)]
        return [g.permute(x, permutation)]

    if op == "matmul":
        lhs = _tensor(env, node.inputs[0])
        rhs = _tensor(env, node.inputs[1])
        output_value = ir.values.get(node.outputs[0])
        output_dtype = (
            _map_ir_dtype(output_value.dtype)
            if output_value is not None and output_value.dtype is not None
            else None
        )
        matmul_output_dtype = _matmul_output_dtype(lhs, rhs, output_dtype=output_dtype)
        if len(lhs.shape) == 2 and len(rhs.shape) == 2:
            out = _matmul_with_quantized_rhs_legalization(g, lhs, rhs, output_dtype=matmul_output_dtype)
            if output_dtype is not None and out.dtype != output_dtype:
                out = g.precision_cast(out, output_dtype)
            return [out]
        legalized = _legalize_matmul_inputs(g, lhs, rhs, node)
        if legalized is None:
            static_batched = _lower_static_batched_matmul(g, lhs, rhs, output_dtype=matmul_output_dtype)
            if static_batched is not None:
                if output_dtype is not None and static_batched.dtype != output_dtype:
                    static_batched = g.precision_cast(static_batched, output_dtype)
                return [static_batched]
            out = _matmul_with_quantized_rhs_legalization(g, lhs, rhs, output_dtype=matmul_output_dtype)
            if output_dtype is not None and out.dtype != output_dtype:
                out = g.precision_cast(out, output_dtype)
            return [out]
        lhs_2d, rhs_2d, output_shape = legalized
        out = _matmul_with_quantized_rhs_legalization(g, lhs_2d, rhs_2d, output_dtype=matmul_output_dtype)
        out = g.reshape(out, output_shape)
        if output_dtype is not None and out.dtype != output_dtype:
            out = g.precision_cast(out, output_dtype)
        return [out]

    if op == "linear":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        output_value = ir.values.get(node.outputs[0])
        output_dtype = _map_ir_dtype(output_value.dtype) if output_value is not None and output_value.dtype is not None else x.dtype
        reshape_back: tuple[int, ...] | None = None
        if len(x.shape) > 2:
            x = _flatten_to_2d_for_linear(g, x)
            output_shape = output_value.shape if output_value is not None else None
            if output_shape is None:
                output_shape = node.meta.get("shape")
            if not isinstance(output_shape, tuple):
                raise NotImplementedError(f"linear missing output shape metadata for node {node.id}")
            reshape_back = tuple(int(v) for v in output_shape)

        matmul_output_dtype = _matmul_output_dtype(x, weight, output_dtype=output_dtype)
        if node.attrs.get("has_bias"):
            bias = _tensor(env, node.inputs[2])
            if (
                x.dtype == Graph.FP16
                and weight.dtype == Graph.FP16
                and output_dtype == Graph.FP16
            ):
                matmul_output_dtype = Graph.FP32
            elif bias.dtype == Graph.FP32 and not _is_quantized_runtime_dtype(weight.dtype):
                matmul_output_dtype = Graph.FP32
        out = _matmul_with_quantized_rhs_legalization(
            g,
            x,
            weight,
            pretransposed_rhs=True,
            output_dtype=matmul_output_dtype,
        )
        out = _trim_padded_last_dim(g, out, output_value.shape if output_value is not None else None)
        if node.attrs.get("has_bias"):
            bias = _ensure_tensor_dtype(g, bias, matmul_output_dtype)
            if output_value is not None and output_value.shape is not None:
                expected_last_dim = int(output_value.shape[-1])
                if len(bias.shape) >= 1 and int(bias.shape[-1]) > expected_last_dim:
                    bias = g.slice(bias, len(bias.shape) - 1, 0, expected_last_dim)
            out, bias = _legalize_elementwise_binary_inputs(g, out, bias)
            try:
                out = g.add(out, bias)
            except RuntimeError as exc:
                raise RuntimeError(
                    "linear bias add failed while lowering "
                    f"{node.id}: out_shape={tuple(out.shape)} bias_shape={tuple(bias.shape)} "
                    f"expected_shape={None if output_value is None else output_value.shape}"
                ) from exc
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        if out.dtype != output_dtype:
            out = g.precision_cast(out, output_dtype)
        return [out]

    if op == "dense_mlp_tq_fused":
        hidden = _tensor(env, node.inputs[0])
        gate_weight = _tensor(env, node.inputs[1])
        up_weight = _tensor(env, node.inputs[2])
        down_weight = _tensor(env, node.inputs[3])
        product_scale = (
            node.attrs.get("product_scale")
            if "product_scale" in node.attrs
            else node.meta.get("product_scale_from_export", node.meta.get("product_scale_elided_by_post_norm", 1.0))
        )
        out = g.dense_mlp_tq_fused(
            hidden,
            gate_weight,
            up_weight,
            down_weight,
            product_scale=float(product_scale),
        )
        output_value = ir.values.get(node.outputs[0])
        output_dtype = _map_ir_dtype(output_value.dtype) if output_value is not None and output_value.dtype is not None else out.dtype
        if out.dtype != output_dtype:
            out = g.precision_cast(out, output_dtype)
        return [out]

    if op == "addmm":
        bias = _tensor(env, node.inputs[0])
        lhs = _tensor(env, node.inputs[1])
        rhs = _tensor(env, node.inputs[2])
        output_value = ir.values.get(node.outputs[0])
        output_dtype = _map_ir_dtype(output_value.dtype) if output_value is not None and output_value.dtype is not None else lhs.dtype
        matmul_output_dtype = _matmul_output_dtype(lhs, rhs, output_dtype=output_dtype)
        if lhs.dtype == Graph.FP16 and rhs.dtype == Graph.FP16 and bias.dtype == Graph.FP16 and output_dtype == Graph.FP16:
            matmul_output_dtype = Graph.FP32
        elif bias.dtype == Graph.FP32 and not _is_quantized_runtime_dtype(rhs.dtype):
            matmul_output_dtype = Graph.FP32
        out = _matmul_with_quantized_rhs_legalization(g, lhs, rhs, output_dtype=matmul_output_dtype)
        out = _trim_padded_last_dim(g, out, output_value.shape if output_value is not None else None)
        bias = _ensure_tensor_dtype(g, bias, matmul_output_dtype)
        out = g.add(bias, out)
        if out.dtype != output_dtype:
            out = g.precision_cast(out, output_dtype)
        return [out]

    if op == "relu":
        return [g.relu(_tensor(env, node.inputs[0]))]

    if op == "silu":
        return [g.silu(_tensor(env, node.inputs[0]))]

    if op == "gelu":
        return [g.gelu(_tensor(env, node.inputs[0]))]

    if op == "gelu_erf":
        return [g.gelu_erf(_tensor(env, node.inputs[0]))]

    if op == "sigmoid":
        return [g.sigmoid(_tensor(env, node.inputs[0]))]

    if op == "glu":
        return [g.glu(_tensor(env, node.inputs[0]), axis=int(node.attrs.get("axis", -1)))]

    if op == "softplus":
        x = _tensor(env, node.inputs[0])
        return [_lower_softplus(g, x)]

    if op == "tanh":
        return [g.tanh(_tensor(env, node.inputs[0]))]

    if op == "softmax":
        return [g.softmax(_tensor(env, node.inputs[0]), axis=int(node.attrs.get("axis", -1)))]

    if op == "lstm_cell":
        x = _tensor(env, node.inputs[0])
        h_prev = _tensor(env, node.inputs[1])
        c_prev = _tensor(env, node.inputs[2])
        weight_ih = _tensor(env, node.inputs[3])
        weight_hh = _tensor(env, node.inputs[4])
        bias_ih = _tensor(env, node.inputs[5])
        bias_hh = _tensor(env, node.inputs[6])

        lstm_out = g.lstm_cell(
            _ensure_fp16_tensor(g, x),
            _ensure_fp16_tensor(g, h_prev),
            _ensure_fp16_tensor(g, c_prev),
            _ensure_fp16_tensor(g, weight_ih),
            _ensure_fp16_tensor(g, weight_hh),
            _ensure_fp16_tensor(g, bias_ih),
            _ensure_fp16_tensor(g, bias_hh),
        )
        h_new = g.slice(lstm_out, axis=2, start=0, length=1)
        c_new = g.slice(lstm_out, axis=2, start=1, length=1)
        h_shape = ir.values[node.outputs[0]].shape
        c_shape = ir.values[node.outputs[1]].shape
        if h_shape is not None:
            h_new = g.reshape(h_new, tuple(int(dim) for dim in h_shape))
        if c_shape is not None:
            c_new = g.reshape(c_new, tuple(int(dim) for dim in c_shape))
        return [h_new, c_new]

    if op in {"scaled_dot_product_attention", "attention"}:
        if _should_lower_gemma4_decoder_attention_without_kernel(ir, node):
            return [_lower_gemma4_decoder_attention_without_kernel(g, ir, env, node)]
        mask = node.attrs.get("mask")
        mask_tensor: Tensor | None = None
        additive_mask = bool(node.attrs.get("additive_mask", False))
        if len(node.inputs) > 3:
            mask_tensor = _tensor(env, node.inputs[3])
        elif mask is not None:
            raise NotImplementedError(f"{op} with literal mask is not supported yet")
        if op == "scaled_dot_product_attention":
            dropout_p = float(node.attrs.get("dropout_p", 0.0))
            if dropout_p != 0.0:
                raise NotImplementedError("scaled_dot_product_attention with dropout is not supported yet")

        # PyTorch SDPA exports tensors as [batch, heads, seq, dim], while Cactus attention
        # expects [batch, seq, heads, dim]. The optimizer may strip surrounding layout
        # permutes and mark the node as already-native to avoid large round-trip copies.
        qkv_layout = str(node.attrs.get("qkv_layout", "bhsd") or "bhsd").lower()
        q_layout = str(node.attrs.get("q_layout", qkv_layout) or qkv_layout).lower()
        k_layout = str(node.attrs.get("k_layout", qkv_layout) or qkv_layout).lower()
        v_layout = str(node.attrs.get("v_layout", qkv_layout) or qkv_layout).lower()
        output_layout = str(node.attrs.get("output_layout", "bhsd") or "bhsd").lower()
        query = _attention_tensor(env, node.inputs[0])
        key = _attention_tensor(env, node.inputs[1])
        value = _attention_tensor(env, node.inputs[2])
        if q_layout != "bthd":
            query = g.permute(query, (0, 2, 1, 3))
        if k_layout != "bthd":
            key = g.permute(key, (0, 2, 1, 3))
        if v_layout != "bthd":
            value = g.permute(value, (0, 2, 1, 3))
        if mask_tensor is not None:
            mask_tensor = _ensure_fp16_tensor(g, mask_tensor)
            mask_tensor = _normalize_attention_mask_for_cactus(g, mask_tensor, query)
        if _should_lower_attention_with_internal_kv_cache(ir, node):
            return _lower_attention_with_internal_kv_cache(
                g,
                ir,
                env,
                node,
                query=query,
                key=key,
                value=value,
                output_layout=output_layout,
            )
        out = g.attention(
            query,
            key,
            value,
            scale=_resolve_attention_scale(node, query),
            is_causal=bool(node.attrs.get("is_causal", False)),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask_tensor,
            additive_mask=additive_mask,
        )
        if output_layout == "bthd":
            return [out]
        return [g.permute(out, (0, 2, 1, 3))]

    if op == "attention_block":
        has_mask = bool(node.attrs.get("has_mask", False))
        has_gate = bool(node.attrs.get("has_gate", False))
        has_bias = bool(node.attrs.get("has_bias", False))
        qkv_layout = str(node.attrs.get("qkv_layout", "bhsd") or "bhsd").lower()
        input_index = 0
        query = _attention_tensor(env, node.inputs[input_index])
        if qkv_layout != "bthd":
            query = g.permute(query, (0, 2, 1, 3))
        input_index += 1
        key = _attention_tensor(env, node.inputs[input_index])
        if qkv_layout != "bthd":
            key = g.permute(key, (0, 2, 1, 3))
        input_index += 1
        value = _attention_tensor(env, node.inputs[input_index])
        if qkv_layout != "bthd":
            value = g.permute(value, (0, 2, 1, 3))
        input_index += 1
        mask_tensor: Tensor | None = None
        if has_mask:
            mask_tensor = _tensor(env, node.inputs[input_index])
            input_index += 1
            mask_tensor = _ensure_fp16_tensor(g, mask_tensor)
            mask_tensor = _normalize_attention_mask_for_cactus(g, mask_tensor, query)

        attn_out = g.attention(
            query,
            key,
            value,
            scale=_resolve_attention_scale(node, query),
            is_causal=bool(node.attrs.get("is_causal", False)),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask_tensor,
            additive_mask=bool(node.attrs.get("additive_mask", False)),
        )
        flat_shape_attr = tuple(int(v) for v in node.attrs.get("attention_output_shape", ()))
        flat_shape = _resolve_reshape_shape(tuple(attn_out.shape), flat_shape_attr) if flat_shape_attr else ()
        if flat_shape:
            attn_out = _reshape_attention_output_for_linear(g, attn_out, flat_shape)

        if has_gate:
            gate = _tensor(env, node.inputs[input_index])
            input_index += 1
            attn_out, gate = _legalize_elementwise_binary_inputs(g, attn_out, gate)
            attn_out = g.multiply(attn_out, gate)

        weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        reshape_back: tuple[int, ...] | None = None
        linear_input = attn_out
        if len(linear_input.shape) > 2:
            linear_input = _flatten_to_2d_for_linear(g, linear_input)
            output_value = ir.values.get(node.outputs[0])
            output_shape = output_value.shape if output_value is not None else None
            if output_shape is None:
                output_shape = node.meta.get("shape")
            if not isinstance(output_shape, tuple):
                raise NotImplementedError(f"attention_block missing output shape metadata for node {node.id}")
            reshape_back = tuple(int(v) for v in output_shape)

        out = _matmul_with_quantized_rhs_legalization(g, linear_input, weight, pretransposed_rhs=True)
        if has_bias:
            out = g.add(out, _tensor(env, node.inputs[input_index]))
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        return [out]

    if op == "self_attention_block":
        has_mask = bool(node.attrs.get("has_mask", False))
        has_gate = bool(node.attrs.get("has_gate", False))
        has_bias = bool(node.attrs.get("has_bias", False))
        has_query_projection_bias = bool(node.attrs.get("has_query_projection_bias", False))
        has_query_add = bool(node.attrs.get("has_query_add", False))
        has_rel_query_add = bool(node.attrs.get("has_rel_query_add", False))
        has_key_projection_bias = bool(node.attrs.get("has_key_projection_bias", False))
        has_value_projection_bias = bool(node.attrs.get("has_value_projection_bias", False))
        has_rel_pos_bias = bool(node.attrs.get("has_rel_pos_bias", False))
        has_relative_key_projection_bias = bool(node.attrs.get("has_relative_key_projection_bias", False))

        input_index = 0
        hidden = _tensor(env, node.inputs[input_index])
        input_index += 1

        query_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        query_projection_bias = None
        if has_query_projection_bias:
            query_projection_bias = _tensor(env, node.inputs[input_index])
            input_index += 1
        query_add = None
        if has_query_add:
            query_add = _tensor(env, node.inputs[input_index])
            input_index += 1
        rel_query_add = None
        if has_rel_query_add:
            rel_query_add = _tensor(env, node.inputs[input_index])
            input_index += 1

        key_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        key_projection_bias = None
        if has_key_projection_bias:
            key_projection_bias = _tensor(env, node.inputs[input_index])
            input_index += 1

        value_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        value_projection_bias = None
        if has_value_projection_bias:
            value_projection_bias = _tensor(env, node.inputs[input_index])
            input_index += 1

        mask_tensor: Tensor | None = None
        if has_mask:
            mask_tensor = _tensor(env, node.inputs[input_index])
            input_index += 1

        relative_key_input = None
        relative_key_weight = None
        relative_key_projection_bias = None
        if has_rel_pos_bias:
            relative_key_input = _tensor(env, node.inputs[input_index])
            input_index += 1
            relative_key_weight = _tensor(env, node.inputs[input_index])
            input_index += 1
            if has_relative_key_projection_bias:
                relative_key_projection_bias = _tensor(env, node.inputs[input_index])
                input_index += 1

        gate = None
        if has_gate:
            gate = _tensor(env, node.inputs[input_index])
            input_index += 1

        output_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        output_bias = None
        if has_bias:
            output_bias = _tensor(env, node.inputs[input_index])

        query_shape = tuple(int(v) for v in node.attrs.get("query_shape", ()))
        key_shape = tuple(int(v) for v in node.attrs.get("key_shape", ()))
        value_shape = tuple(int(v) for v in node.attrs.get("value_shape", ()))
        relative_key_shape = tuple(int(v) for v in node.attrs.get("relative_key_shape", ()))

        query_base = _lower_projected_attention_tensor(g, hidden, query_weight, query_projection_bias, query_shape)
        query = query_base
        if query_add is not None:
            query_add = _normalize_attention_add_tensor(g, query_add, query_shape)
            query, query_add = _legalize_elementwise_binary_inputs(g, query, query_add)
            query = g.add(query, query_add)

        key = _lower_projected_attention_tensor(g, hidden, key_weight, key_projection_bias, key_shape)
        value = _lower_projected_attention_tensor(g, hidden, value_weight, value_projection_bias, value_shape)

        if has_rel_pos_bias:
            if relative_key_input is None or relative_key_weight is None or not relative_key_shape:
                raise NotImplementedError(f"self_attention_block missing relative position bias inputs for node {node.id}")
            rel_query = query_base
            if rel_query_add is not None:
                rel_query_add = _normalize_attention_add_tensor(g, rel_query_add, query_shape)
                rel_query, rel_query_add = _legalize_elementwise_binary_inputs(g, rel_query, rel_query_add)
                rel_query = g.add(rel_query, rel_query_add)
            relative_key = _lower_projected_attention_tensor(
                g,
                relative_key_input,
                relative_key_weight,
                relative_key_projection_bias,
                relative_key_shape,
            )
            rel_mask = g.rel_pos_bias(rel_query, relative_key, float(node.attrs.get("rel_pos_scale", 1.0)))
            if mask_tensor is None:
                mask_tensor = rel_mask
            else:
                mask_tensor, rel_mask = _legalize_elementwise_binary_inputs(g, mask_tensor, rel_mask)
                mask_tensor = g.add(mask_tensor, rel_mask)
            mask_tensor = _ensure_fp16_tensor(g, mask_tensor)
        elif mask_tensor is not None:
            mask_tensor = _ensure_fp16_tensor(g, mask_tensor)
        if mask_tensor is not None:
            mask_tensor = _normalize_attention_mask_for_cactus(g, mask_tensor, query)

        attn_out = g.attention(
            query,
            key,
            value,
            scale=_resolve_attention_scale(node, query),
            is_causal=bool(node.attrs.get("is_causal", False)),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask_tensor,
            additive_mask=bool(node.attrs.get("additive_mask", False)),
        )
        flat_shape_attr = tuple(int(v) for v in node.attrs.get("attention_output_shape", ()))
        flat_shape = _resolve_reshape_shape(tuple(attn_out.shape), flat_shape_attr) if flat_shape_attr else ()
        if flat_shape:
            attn_out = _reshape_attention_output_for_linear(g, attn_out, flat_shape)

        if gate is not None:
            attn_out, gate = _legalize_elementwise_binary_inputs(g, attn_out, gate)
            attn_out = g.multiply(attn_out, gate)

        linear_input = attn_out
        reshape_back: tuple[int, ...] | None = None
        if len(linear_input.shape) > 2:
            linear_input = _flatten_to_2d_for_linear(g, linear_input)
            output_value = ir.values.get(node.outputs[0])
            output_shape = output_value.shape if output_value is not None else None
            if output_shape is None:
                output_shape = node.meta.get("shape")
            if not isinstance(output_shape, tuple):
                raise NotImplementedError(f"self_attention_block missing output shape metadata for node {node.id}")
            reshape_back = tuple(int(v) for v in output_shape)

        out = _matmul_with_quantized_rhs_legalization(g, linear_input, output_weight, pretransposed_rhs=True)
        if output_bias is not None:
            out = g.add(out, output_bias)
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        return [out]

    if op in ("sum", "mean", "variance", "min", "max"):
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        axes = _normalize_reduction_axes(node.attrs.get("axis"), len(x.shape))
        keepdim = bool(node.attrs.get("keepdim", False))
        out = _lower_reduction(g, op, x, axes=axes)
        if keepdim:
            new_shape = list(x.shape)
            for reduced_axis in axes:
                new_shape[reduced_axis] = 1
            out = g.reshape(out, tuple(new_shape))
        return [out]

    if op == "cumsum":
        x = _ensure_scalar_math_tensor(g, _tensor(env, node.inputs[0]))
        raw_axis = node.attrs.get("axis")
        if not isinstance(raw_axis, int):
            args = node.attrs.get("args", ())
            if isinstance(args, (list, tuple)) and len(args) > 1 and isinstance(args[1], int):
                raw_axis = int(args[1])
            else:
                raw_axis = -1
        axis = _normalize_dim(int(raw_axis), len(x.shape))
        return [g.cumsum(x, axis)]

    if op == "cat":
        tensors = [_tensor(env, value_id) for value_id in node.inputs]
        output_dtype: int | None = None
        if node.outputs:
            output_value = ir.values.get(node.outputs[0])
            if output_value is not None and output_value.dtype is not None:
                output_dtype = _map_ir_or_torch_dtype(output_value.dtype)
        return [_cat_with_legalized_dtype(g, tensors, axis=int(node.attrs.get("axis", 0)), dtype=output_dtype)]

    if op in {"stack", "aten.stack.default"}:
        tensors = [_tensor(env, value_id) for value_id in node.inputs]
        if not tensors:
            raise NotImplementedError("stack requires at least one input tensor")

        raw_axis = node.attrs.get("axis")
        if not isinstance(raw_axis, int):
            args = node.attrs.get("args", ())
            if isinstance(args, (list, tuple)) and len(args) > 1 and isinstance(args[1], int):
                raw_axis = int(args[1])
            else:
                raw_axis = 0

        base_shape = tuple(int(dim) for dim in tensors[0].shape)
        axis = _normalize_dim(int(raw_axis), len(base_shape) + 1)
        reshaped: list[Tensor] = []
        for tensor in tensors:
            if tuple(int(dim) for dim in tensor.shape) != base_shape:
                raise NotImplementedError(
                    f"stack requires equal input shapes, got {[tuple(int(dim) for dim in t.shape) for t in tensors]}"
                )
            expanded_shape = list(base_shape)
            expanded_shape.insert(axis, 1)
            reshaped.append(g.reshape(tensor, tuple(expanded_shape)))
        return [g.cat(reshaped, axis=axis)]

    if op == "slice":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs["axis"]), len(x.shape))
        start = int(node.attrs["start"])
        end = int(node.attrs["end"])
        step = int(node.attrs.get("step", 1))
        dim_size = x.shape[axis]
        if step == 0:
            raise NotImplementedError("slice with step == 0 is invalid")
        if step != 1:
            indices = list(range(*slice(start, end, step).indices(dim_size)))
            if axis == 0 or x.dtype == Graph.FP16:
                return [_lower_static_strided_slice_via_gather(g, x, axis=axis, indices=indices)]
            if not indices:
                return [g.slice(x, axis=axis, start=0, length=0)]
            expanded_shape = list(int(dim) for dim in x.shape)
            expanded_shape[axis] = 1
            pieces: list[Tensor] = []
            for idx in indices:
                piece = g.index(x, index_value=idx, axis=axis)
                piece = g.reshape(piece, tuple(expanded_shape))
                pieces.append(piece)
            if len(pieces) == 1:
                return [pieces[0]]
            return [g.cat(pieces, axis=axis)]
        start = _normalize_index(start, dim_size)
        end = _normalize_slice_end(end, dim_size)
        length = max(0, end - start)
        return [g.slice(x, axis=axis, start=start, length=length)]

    if op == "split_with_sizes":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs.get("axis", -1)), len(x.shape))
        sizes = tuple(int(v) for v in node.attrs["sizes"])
        start = 0
        outputs: list[Tensor] = []
        for size in sizes:
            outputs.append(g.slice(x, axis=axis, start=start, length=int(size)))
            start += int(size)
        return [outputs]

    if op == "chunk":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs.get("axis", 0)), len(x.shape))
        chunks = int(node.attrs["chunks"])
        if chunks <= 0:
            raise NotImplementedError(f"chunk requires a positive chunk count, got {chunks}")
        dim_size = int(x.shape[axis])
        if dim_size <= 0:
            return [[g.slice(x, axis=axis, start=0, length=0)]]
        chunk_size = (dim_size + chunks - 1) // chunks
        outputs: list[Tensor] = []
        start = 0
        while start < dim_size:
            length = min(chunk_size, dim_size - start)
            outputs.append(g.slice(x, axis=axis, start=start, length=length))
            start += length
        return [outputs]

    if op == "ones":
        shape = tuple(int(v) for v in node.attrs.get("shape", ()))
        if not shape:
            raise NotImplementedError(f"ones requires a static shape, got {shape}")
        torch_dtype = _materialize_constant_torch_dtype(node.attrs.get("dtype"))
        return [_materialize_constant_tensor(g, torch.ones(shape, dtype=torch_dtype))]

    if op == "tril":
        x = _tensor(env, node.inputs[0])
        shape = tuple(int(dim) for dim in x.shape)
        if len(shape) < 2:
            raise NotImplementedError(f"tril expects rank >= 2, got shape {shape}")
        diagonal = int(node.attrs.get("diagonal", 0))
        mask_dtype = _torch_dtype_for_graph_dtype(x.dtype)
        mask = torch.ones(shape, dtype=mask_dtype).tril(diagonal=diagonal)
        mask_tensor = _materialize_constant_tensor(g, mask)
        return [_lower_binary_op(g, x, mask_tensor, "multiply")]

    if op == "unfold":
        x = _tensor(env, node.inputs[0])
        rank = len(x.shape)
        axis = _normalize_dim(int(node.attrs["dimension"]), rank)
        size = int(node.attrs["size"])
        step = int(node.attrs["step"])
        if size <= 0:
            raise NotImplementedError(f"unfold requires a positive size, got {size}")
        if step <= 0:
            raise NotImplementedError(f"unfold requires a positive step, got {step}")
        dim_size = int(x.shape[axis])
        if size > dim_size:
            raise NotImplementedError(
                f"unfold size {size} exceeds dimension size {dim_size} for axis {axis}"
            )

        window_starts = list(range(0, dim_size - size + 1, step))
        if not window_starts:
            raise NotImplementedError(
                f"unfold produced no windows for axis={axis}, size={size}, step={step}, dim_size={dim_size}"
            )

        base_shape = [int(dim) for dim in x.shape]
        trailing_shape = [base_shape[idx] for idx in range(rank) if idx != axis] + [size]
        expanded_shape = tuple(trailing_shape[:axis] + [1] + trailing_shape[axis:])
        permutation = tuple(idx for idx in range(rank) if idx != axis) + (axis,)

        windows: list[Tensor] = []
        for start in window_starts:
            piece = g.slice(x, axis=axis, start=start, length=size)
            if axis != rank - 1:
                piece = g.permute(piece, permutation=permutation)
            piece = g.reshape(piece, expanded_shape)
            windows.append(piece)
        if len(windows) == 1:
            return [windows[0]]
        return [g.cat(windows, axis=axis)]

    if op == "pad":
        x = _tensor(env, node.inputs[0])
        mode = str(node.attrs.get("mode", "constant"))
        if mode != "constant":
            raise NotImplementedError(f"pad mode is unsupported: {mode}")
        pads = tuple(int(v) for v in node.attrs.get("pads", ()))
        if len(pads) % 2 != 0:
            raise NotImplementedError(f"pad expects an even-length pads tuple, got {pads}")
        value = float(node.attrs.get("value", 0.0))

        current = x
        current_shape = list(int(dim) for dim in x.shape)
        pad_dims = len(pads) // 2
        if pad_dims > len(current_shape):
            raise NotImplementedError(
                f"pad rank mismatch: pads={pads} for input shape {tuple(current_shape)}"
            )

        for pad_index in range(pad_dims):
            before = pads[2 * pad_index]
            after = pads[2 * pad_index + 1]
            axis = len(current_shape) - 1 - pad_index
            if before < 0 or after < 0:
                raise NotImplementedError(f"negative pad is unsupported: {pads}")
            pieces = []
            torch_dtype = _torch_dtype_for_graph_dtype(current.dtype)
            if before > 0:
                left_shape = list(current_shape)
                left_shape[axis] = before
                left = _materialize_constant_tensor(
                    g,
                    torch.full(tuple(left_shape), value, dtype=torch_dtype),
                )
                pieces.append(left)
            pieces.append(current)
            if after > 0:
                right_shape = list(current_shape)
                right_shape[axis] = after
                right = _materialize_constant_tensor(
                    g,
                    torch.full(tuple(right_shape), value, dtype=torch_dtype),
                )
                pieces.append(right)
            if len(pieces) > 1:
                current = g.cat(pieces, axis=axis)
                current_shape[axis] += before + after
        return [current]

    if op == "index":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs.get("axis", 0)), len(x.shape))
        index_value = _normalize_index(int(node.attrs["index_value"]), x.shape[axis])
        return [g.index(x, index_value=index_value, axis=axis)]

    if op == "gather":
        return [
            g.gather(
                _tensor(env, node.inputs[0]),
                _tensor(env, node.inputs[1]),
                axis=int(node.attrs.get("axis", 0)),
            )
        ]

    if op == "embedding":
        embedding_tensor = _tensor(env, node.inputs[0])
        indices_tensor = _tensor(env, node.inputs[1])
        _debug_embedding_lowering(node, embedding_tensor, indices_tensor)
        return [g.embedding_from_tensor(embedding_tensor, indices_tensor)]

    if op == "conv_module":
        input_index = 0
        x_nlc = _tensor(env, node.inputs[input_index])
        input_index += 1

        pointwise1_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        pointwise1_bias = None
        if bool(node.attrs.get("has_pointwise1_bias", False)):
            pointwise1_bias = _tensor(env, node.inputs[input_index])
            input_index += 1

        depthwise_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        depthwise_bias = None
        if bool(node.attrs.get("has_depthwise_bias", False)):
            depthwise_bias = _tensor(env, node.inputs[input_index])
            input_index += 1

        batch_norm_weight = _tensor(env, node.inputs[input_index])
        batch_norm_bias = _tensor(env, node.inputs[input_index + 1])
        batch_norm_running_mean = _tensor(env, node.inputs[input_index + 2])
        batch_norm_running_var = _tensor(env, node.inputs[input_index + 3])
        input_index += 4

        pointwise2_weight = _tensor(env, node.inputs[input_index])
        input_index += 1
        pointwise2_bias = None
        if bool(node.attrs.get("has_pointwise2_bias", False)):
            pointwise2_bias = _tensor(env, node.inputs[input_index])

        if len(x_nlc.shape) != 3:
            raise NotImplementedError(f"conv_module expects rank-3 NLC input, got {x_nlc.shape}")
        kernel_size = int(node.attrs.get("depthwise_kernel_size", 0))
        padding = int(node.attrs.get("depthwise_padding", 0))
        if len(depthwise_weight.shape) != 3 or kernel_size != 9 or padding != 4:
            raise NotImplementedError(
                f"conv_module currently supports same depthwise kernel-9 lowering, got "
                f"kernel_size={kernel_size} padding={padding} weight_shape={depthwise_weight.shape}"
            )

        current = g.conv1d_pointwise(x_nlc, pointwise1_weight, bias=pointwise1_bias)
        current = g.glu(current, axis=-1)
        current = g.conv1d_same_depthwise_k9(current, depthwise_weight, bias=depthwise_bias)
        current = g.batch_norm(
            current,
            batch_norm_weight,
            batch_norm_bias,
            batch_norm_running_mean,
            batch_norm_running_var,
            axis=len(current.shape) - 1,
            eps=float(node.attrs.get("eps", 1e-5)),
        )
        current = g.silu(current)
        current = g.conv1d_pointwise(current, pointwise2_weight, bias=pointwise2_bias)
        return [current]

    if op == "conv1d":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        bias = _tensor(env, node.inputs[2]) if len(node.inputs) > 2 else None
        stride = int(node.attrs.get("stride", 1))
        padding = int(node.attrs.get("padding", 0))
        dilation = int(node.attrs.get("dilation", 1))
        groups = int(node.attrs.get("groups", 1))

        if (
            len(x.shape) == 3
            and len(weight.shape) == 3
            and weight.shape[2] == 1
            and stride == 1
            and padding == 0
            and dilation == 1
            and groups == 1
        ):
            x_nlc = g.permute(x, (0, 2, 1))
            out_nlc = g.conv1d_pointwise(x_nlc, weight, bias=bias)
            return [g.permute(out_nlc, (0, 2, 1))]

        if (
            len(x.shape) == 3
            and len(weight.shape) == 3
            and groups == x.shape[1] == weight.shape[0]
            and weight.shape[1] == 1
            and weight.shape[2] == 9
            and stride == 1
            and padding == 4
            and dilation == 1
        ):
            x_nlc = g.permute(x, (0, 2, 1))
            out_nlc = g.conv1d_same_depthwise_k9(x_nlc, weight, bias=bias)
            return [g.permute(out_nlc, (0, 2, 1))]

        if (
            len(x.shape) == 3
            and len(weight.shape) == 3
            and groups == x.shape[1] == weight.shape[0]
            and weight.shape[1] == 1
            and stride == 1
            and padding == dilation * max(weight.shape[2] - 1, 0)
        ):
            x_nlc = g.permute(x, (0, 2, 1))
            out_nlc = g.conv1d_causal(x_nlc, weight, kernel_size=weight.shape[2], dilation=dilation)
            if bias is not None:
                bias_reshaped = g.reshape(bias, (1, 1, int(weight.shape[0])))
                out_nlc, bias_reshaped = _legalize_elementwise_binary_inputs(g, out_nlc, bias_reshaped)
                out_nlc = g.add(out_nlc, bias_reshaped)
            return [g.permute(out_nlc, (0, 2, 1))]

        if (
            len(x.shape) == 3
            and len(weight.shape) == 3
            and groups == 1
            and dilation == 1
            and weight.shape[2] == 3
            and padding == 1
            and stride in {1, 2}
        ):
            out = g.conv1d_k3(x, weight, stride=stride)
            if bias is not None:
                bias_reshaped = g.reshape(bias, (1, int(weight.shape[0]), 1))
                out, bias_reshaped = _legalize_elementwise_binary_inputs(g, out, bias_reshaped)
                out = g.add(out, bias_reshaped)
            return [out]

        if dilation != 1:
            raise NotImplementedError(f"conv1d with dilation != 1 is unsupported by generic lowering: {dilation}")
        if padding != 0:
            if len(x.shape) != 3:
                raise NotImplementedError(f"generic conv1d padding expects rank-3 input, got {x.shape}")
            batch_size, channels, _ = (int(dim) for dim in x.shape)
            pad_dtype = _torch_dtype_for_graph_dtype(x.dtype)
            pieces: list[Tensor] = []
            if padding > 0:
                left = _materialize_constant_tensor(
                    g,
                    torch.zeros((batch_size, channels, padding), dtype=pad_dtype),
                )
                pieces.append(left)
            pieces.append(x)
            if padding > 0:
                right = _materialize_constant_tensor(
                    g,
                    torch.zeros((batch_size, channels, padding), dtype=pad_dtype),
                )
                pieces.append(right)
            x = g.cat(pieces, axis=2)
        if groups != 1:
            if len(x.shape) != 3 or len(weight.shape) != 3:
                raise NotImplementedError(
                    f"grouped conv1d lowering expects rank-3 input and weight, got {x.shape} and {weight.shape}"
                )
            input_channels = int(x.shape[1])
            if input_channels % groups != 0:
                raise NotImplementedError(
                    f"grouped conv1d requires input channels divisible by groups, got C_in={input_channels}, groups={groups}"
                )
            output_channels = int(weight.shape[0])
            if output_channels % groups != 0:
                raise NotImplementedError(
                    f"grouped conv1d requires output channels divisible by groups, got C_out={output_channels}, groups={groups}"
                )
            input_channels_per_group = input_channels // groups
            output_channels_per_group = output_channels // groups
            if int(weight.shape[1]) != input_channels_per_group:
                raise NotImplementedError(
                    "grouped conv1d weight has incompatible per-group input channels: "
                    f"weight C_in/group={int(weight.shape[1])}, expected {input_channels_per_group}"
                )
            outputs: list[Tensor] = []
            for group_index in range(groups):
                x_group = g.slice(
                    x,
                    axis=1,
                    start=group_index * input_channels_per_group,
                    length=input_channels_per_group,
                )
                weight_group = g.slice(
                    weight,
                    axis=0,
                    start=group_index * output_channels_per_group,
                    length=output_channels_per_group,
                )
                bias_group = None
                if bias is not None:
                    bias_group = g.slice(
                        bias,
                        axis=0,
                        start=group_index * output_channels_per_group,
                        length=output_channels_per_group,
                    )
                outputs.append(g.conv1d(x_group, weight_group, bias=bias_group, stride=stride))
            if len(outputs) == 1:
                return [outputs[0]]
            return [g.cat(outputs, axis=1)]
        return [g.conv1d(x, weight, bias=bias, stride=stride)]

    if op == "conv2d":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        bias = _tensor(env, node.inputs[2]) if len(node.inputs) > 2 else None
        stride = int(node.attrs.get("stride", 1))
        padding = int(node.attrs.get("padding", 0))
        dilation = int(node.attrs.get("dilation", 1))
        groups = int(node.attrs.get("groups", 1))

        if len(x.shape) != 4 or len(weight.shape) != 4:
            raise NotImplementedError(f"conv2d lowering expects rank-4 tensors, got {x.shape} and {weight.shape}")
        if dilation != 1:
            raise NotImplementedError(f"conv2d with dilation != 1 is unsupported: {dilation}")

        if stride == 2 and padding == 0 and groups == 1 and weight.shape[2:] == (3, 3):
            batch_size, channels, height, width = (int(dim) for dim in x.shape)
            pad_dtype = _torch_dtype_for_graph_dtype(x.dtype)
            top = _materialize_constant_tensor(
                g,
                torch.zeros((batch_size, channels, 1, width), dtype=pad_dtype),
            )
            bottom = _materialize_constant_tensor(
                g,
                torch.zeros((batch_size, channels, 1, width), dtype=pad_dtype),
            )
            x = g.cat([top, x, bottom], axis=2)
            left = _materialize_constant_tensor(
                g,
                torch.zeros((batch_size, channels, height + 2, 1), dtype=pad_dtype),
            )
            right = _materialize_constant_tensor(
                g,
                torch.zeros((batch_size, channels, height + 2, 1), dtype=pad_dtype),
            )
            x = g.cat([left, x, right], axis=3)
            y = g.conv2d_k3s2p1(x, weight, bias=bias)
            output_height = ((height - 3) // 2) + 1
            output_width = ((width - 3) // 2) + 1
            y = g.slice(y, axis=2, start=1, length=output_height)
            y = g.slice(y, axis=3, start=1, length=output_width)
            return [y]

        if stride == 2 and padding == 1 and groups == 1 and weight.shape[2:] == (3, 3):
            return [g.conv2d_k3s2p1(x, weight, bias=bias)]

        if (
            stride == 2
            and padding == 1
            and groups == x.shape[1] == weight.shape[0]
            and weight.shape[1] == 1
            and weight.shape[2:] == (3, 3)
        ):
            return [g.conv2d_depthwise_k3s2p1(x, weight, bias=bias)]

        if (
            stride == 1
            and padding == 0
            and groups == 1
            and weight.shape[2:] == (1, 1)
        ):
            return [g.conv2d_pointwise_1x1(x, weight, bias=bias)]

        if stride == 1 and padding == 1 and groups == 1 and weight.shape[2:] == (3, 3):
            return [g.conv2d_k3s1p1(x, weight, bias=bias)]

        return [g.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)]

    if op == "layer_norm":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        bias = _tensor(env, node.inputs[2]) if len(node.inputs) > 2 else None
        return [g.layer_norm(x, weight, bias=bias, eps=float(node.attrs["eps"]))]

    if op == "rms_norm":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        reshape_back: tuple[int, ...] | None = None
        if len(x.shape) > 2:
            reshape_back = tuple(int(dim) for dim in x.shape)
            x = _flatten_to_2d_for_linear(g, x)
        out = g.rms_norm(x, weight, eps=float(node.attrs["eps"]))
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        return [out]

    if op == "rope":
        rope_input = _tensor(env, node.inputs[0])
        if _rope_input_is_bhsd(ir, node.inputs[0]):
            rope_input = g.permute(rope_input, (0, 2, 1, 3))
            rope_out = g.rope(
                rope_input,
                float(node.attrs["theta"]),
                position_offset=int(node.attrs.get("position_offset", 0)),
            )
            return [g.permute(rope_out, (0, 2, 1, 3))]
        return [
            g.rope(
                rope_input,
                float(node.attrs["theta"]),
                position_offset=int(node.attrs.get("position_offset", 0)),
            )
        ]

    if op == "rel_pos_bias":
        query = _attention_tensor(env, node.inputs[0])
        if len(query.shape) != 4:
            raise NotImplementedError(f"rel_pos_bias expects rank-4 query input, got {query.shape}")
        relative_key = _tensor(env, node.inputs[1])
        if len(relative_key.shape) != 4:
            raise NotImplementedError(f"rel_pos_bias expects rank-4 relative_key input, got {relative_key.shape}")
        if int(query.shape[2]) == int(relative_key.shape[2]) and int(query.shape[3]) == int(relative_key.shape[3]):
            pass
        elif int(query.shape[1]) == int(relative_key.shape[2]) and int(query.shape[3]) == int(relative_key.shape[3]):
            query = g.permute(query, (0, 2, 1, 3))
        else:
            raise NotImplementedError(
                "rel_pos_bias query/relative_key layout mismatch: "
                f"query={query.shape}, relative_key={relative_key.shape}"
            )
        return [g.rel_pos_bias(query, relative_key, float(node.attrs.get("scale", 1.0)))]

    if op in {"gated_deltanet_prefill", "gated_deltanet_decode"}:
        x = _tensor(env, node.inputs[0])
        qkv_weight = _tensor(env, node.inputs[1])
        a_weight = _tensor(env, node.inputs[2])
        b_weight = _tensor(env, node.inputs[3])
        norm_weight = _tensor(env, node.inputs[4])

        input_index = 5
        z_weight = None
        if bool(node.attrs.get("has_z", False)):
            z_weight = _tensor(env, node.inputs[input_index])
            input_index += 1
        dt_bias = None
        if bool(node.attrs.get("has_dt_bias", False)):
            dt_bias = _tensor(env, node.inputs[input_index])
            input_index += 1
        a_log = None
        if bool(node.attrs.get("has_a_log", False)):
            a_log = _tensor(env, node.inputs[input_index])
            input_index += 1
        conv_weight = None
        if bool(node.attrs.get("has_conv", False)):
            conv_weight = _tensor(env, node.inputs[input_index])

        if len(x.shape) != 3:
            raise NotImplementedError(f"{op} currently expects rank-3 normalized input, got {x.shape}")

        batch_size, seq_len, hidden_dim = (int(dim) for dim in x.shape)
        if batch_size != 1:
            raise NotImplementedError(f"{op} currently supports batch size 1, got {x.shape}")

        num_k_heads = int(node.attrs["num_k_heads"])
        num_v_heads = int(node.attrs["num_v_heads"])
        key_dim = int(node.attrs["key_dim"])
        value_dim = int(node.attrs["value_dim"])
        eps = float(node.attrs.get("eps", 1e-6))
        chunk_size = int(node.attrs.get("chunk_size", 64))

        mixed_qkv = _matmul_with_quantized_rhs_legalization(
            g,
            _flatten_to_2d_for_linear(g, x),
            qkv_weight,
            pretransposed_rhs=True,
        )
        mixed_qkv_dim = int(qkv_weight.shape[0])
        mixed_qkv = g.reshape(mixed_qkv, (batch_size, seq_len, mixed_qkv_dim))
        if conv_weight is not None:
            kernel_size = int(conv_weight.shape[2])
            mixed_qkv = g.conv1d_causal(mixed_qkv, conv_weight, kernel_size=kernel_size, dilation=1)
            mixed_qkv = g.silu(mixed_qkv)

        q_proj_dim = num_k_heads * key_dim
        v_proj_dim = num_v_heads * value_dim
        k_proj_dim = num_k_heads * key_dim
        q_proj = g.slice(mixed_qkv, axis=2, start=0, length=q_proj_dim)
        k_proj = g.slice(mixed_qkv, axis=2, start=q_proj_dim, length=k_proj_dim)
        v_proj = g.slice(mixed_qkv, axis=2, start=q_proj_dim + k_proj_dim, length=v_proj_dim)

        q_4d = g.reshape(q_proj, (batch_size, seq_len, num_k_heads, key_dim))
        k_4d = g.reshape(k_proj, (batch_size, seq_len, num_k_heads, key_dim))
        v_4d = g.reshape(v_proj, (batch_size, seq_len, num_v_heads, value_dim))

        q_norm = g.sum(g.multiply(q_4d, q_4d), axis=3)
        q_norm = g.scalar_sqrt(g.scalar_add(q_norm, eps))
        q_norm = g.reshape(q_norm, (batch_size, seq_len, num_k_heads, 1))
        q_4d = g.divide(q_4d, q_norm)

        k_norm = g.sum(g.multiply(k_4d, k_4d), axis=3)
        k_norm = g.scalar_sqrt(g.scalar_add(k_norm, eps))
        k_norm = g.reshape(k_norm, (batch_size, seq_len, num_k_heads, 1))
        k_4d = g.divide(k_4d, k_norm)

        a_logits = _matmul_with_quantized_rhs_legalization(
            g,
            _flatten_to_2d_for_linear(g, x),
            a_weight,
            pretransposed_rhs=True,
        )
        a_logits = g.reshape(a_logits, (batch_size, seq_len, int(a_weight.shape[0])))
        b_logits = _matmul_with_quantized_rhs_legalization(
            g,
            _flatten_to_2d_for_linear(g, x),
            b_weight,
            pretransposed_rhs=True,
        )
        b_logits = g.reshape(b_logits, (batch_size, seq_len, int(b_weight.shape[0])))

        if dt_bias is not None:
            dt_bias_2d = g.reshape(dt_bias, (1, int(dt_bias.shape[0])))
            a_logits, dt_bias_2d = _legalize_elementwise_binary_inputs(g, a_logits, dt_bias_2d)
            a_logits = g.add(a_logits, dt_bias_2d)
        a_softplus = _lower_softplus(g, a_logits)

        if a_log is not None:
            a_log_2d = g.reshape(a_log, (1, int(a_log.shape[0])))
            neg_exp_a = g.scalar_multiply(g.scalar_exp(a_log_2d), -1.0)
            neg_exp_a, a_softplus = _legalize_elementwise_binary_inputs(g, neg_exp_a, a_softplus)
            gate_log = g.multiply(neg_exp_a, a_softplus)
        else:
            gate_log = g.scalar_multiply(a_softplus, -1.0)
        beta = g.sigmoid(b_logits)

        initial_state = _materialize_constant_tensor(
            g,
            torch.zeros((batch_size, key_dim, num_v_heads, value_dim), dtype=torch.float16),
        )

        if op == "gated_deltanet_decode":
            deltanet_out = g.gated_deltanet_decode(q_4d, k_4d, v_4d, gate_log, beta, initial_state, 0.0)
        else:
            deltanet_out = g.gated_deltanet_prefill(q_4d, k_4d, v_4d, gate_log, beta, initial_state, chunk_size, 0.0)

        y_4d = g.slice(deltanet_out, axis=1, start=0, length=seq_len)
        y_2d = g.reshape(y_4d, (seq_len * num_v_heads, value_dim))

        if z_weight is not None:
            z_proj = _matmul_with_quantized_rhs_legalization(
                g,
                _flatten_to_2d_for_linear(g, x),
                z_weight,
                pretransposed_rhs=True,
            )
            z_proj = g.reshape(z_proj, (seq_len * num_v_heads, value_dim))
            y_2d = g.multiply(g.rms_norm(y_2d, norm_weight, eps=eps), g.silu(z_proj))

        return [g.reshape(y_2d, (batch_size, seq_len, num_v_heads * value_dim))]

    if op == "group_norm":
        return [
            g.group_norm(
                _tensor(env, node.inputs[0]),
                _tensor(env, node.inputs[1]),
                _tensor(env, node.inputs[2]),
                num_groups=int(node.attrs["num_groups"]),
                eps=float(node.attrs["eps"]),
            )
        ]

    if op == "batch_norm":
        axis = int(node.attrs.get("axis", 1))
        return [
            g.batch_norm(
                _tensor(env, node.inputs[0]),
                _tensor(env, node.inputs[1]),
                _tensor(env, node.inputs[2]),
                _tensor(env, node.inputs[3]),
                _tensor(env, node.inputs[4]),
                axis=axis,
                eps=float(node.attrs["eps"]),
            )
        ]

    if op == "identity":
        return [env[node.inputs[0]]]

    if op == "contiguous":
        return [_tensor(env, node.inputs[0])]

    if op == "advanced_index":
        lowered = _try_lower_advanced_index(g, node, env, ir)
        if lowered is not None:
            return [lowered]
        raise NotImplementedError(f"unsupported advanced_index pattern for node {node.id}")

    if op in {"masked_scatter", "aten.masked_scatter.default"}:
        base = _tensor(env, node.inputs[0])
        mask = _tensor(env, node.inputs[1])
        source = _tensor(env, node.inputs[2])
        if source.dtype != base.dtype:
            source = _ensure_tensor_dtype(g, source, base.dtype)
        return [g.masked_scatter(base, mask, source)]

    if op == "masked_fill":
        base = _tensor(env, node.inputs[0])
        mask = _lower_compare_op(g, env[node.inputs[1]], 0.0, "not_equal")
        inverse_mask = g.scalar_add(g.scalar_multiply(mask, -1.0), 1.0)
        fill_value = _clamp_scalar_for_dtype(float(node.attrs["value"]), base.dtype)
        fill_term = g.scalar_multiply(_ensure_tensor_dtype(g, mask, base.dtype), fill_value)
        base_term = _lower_binary_op(g, base, _ensure_tensor_dtype(g, inverse_mask, base.dtype), "multiply")
        return [_lower_binary_op(g, fill_term, base_term, "add")]

    if op == "getitem":
        source = env[node.inputs[0]]
        index = int(node.attrs["index"])
        if isinstance(source, (tuple, list)):
            return [source[index]]
        if isinstance(source, Tensor):
            normalized_index = _normalize_index(index, int(source.shape[0]))
            return [g.index(source, normalized_index, axis=0)]
        raise NotImplementedError(f"getitem source is not tuple/list for node {node.id}")

    raise NotImplementedError(f"unsupported IR op in lowering: {op}")


def _tensor(env: dict[str, Any], value_id: str) -> Tensor:
    try:
        value = env[value_id]
    except KeyError as exc:
        raise NotImplementedError(f"missing IR value during lowering: {value_id}") from exc
    if not isinstance(value, Tensor):
        raise TypeError(f"expected lowered tensor for {value_id}, got {type(value).__name__}")
    return value


def _attention_tensor(env: dict[str, Any], value_id: str) -> Tensor:
    try:
        value = env[value_id]
    except KeyError as exc:
        raise NotImplementedError(f"missing IR value during lowering: {value_id}") from exc
    if isinstance(value, BroadcastAlias):
        if value.kind != "gqa_repeat_kv":
            raise TypeError(f"unsupported broadcast alias for attention input {value_id}: {value.kind}")
        return value.tensor
    if not isinstance(value, Tensor):
        raise TypeError(f"expected lowered tensor for {value_id}, got {type(value).__name__}")
    return value


def _resolve_attention_scale(node: IRNode, query: Tensor) -> float:
    scale = float(node.attrs.get("scale", 0.0) or 0.0)
    if scale > 0.0:
        return scale
    if len(query.shape) >= 1 and int(query.shape[-1]) > 0:
        return float(int(query.shape[-1]) ** -0.5)
    return 1.0


def _normalize_attention_mask_for_cactus(g: Graph, mask: Tensor, query: Tensor) -> Tensor:
    """Convert broadcastable HF masks into the native Cactus attention contract."""

    mask_shape = tuple(int(dim) for dim in mask.shape)
    query_shape = tuple(int(dim) for dim in query.shape)
    if len(query_shape) != 4:
        return mask

    batch_size, seq_len, num_heads, _ = query_shape
    if len(mask_shape) == 2 and batch_size == 1 and mask_shape[0] == seq_len:
        return g.reshape(mask, (1, mask_shape[0], mask_shape[1]))

    if len(mask_shape) == 4:
        if (
            mask_shape[0] == batch_size
            and mask_shape[1] == 1
            and mask_shape[2] == seq_len
        ):
            return g.reshape(mask, (mask_shape[0], mask_shape[2], mask_shape[3]))
        if (
            mask_shape[0] == batch_size
            and mask_shape[1] == num_heads
            and mask_shape[2] == seq_len
        ):
            return mask

    return mask


def _lower_projected_attention_tensor(
    g: Graph,
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target_shape: tuple[int, ...],
) -> Tensor:
    if len(target_shape) != 4:
        raise NotImplementedError(f"projected attention tensor expects rank-4 target shape, got {target_shape}")

    linear_input = _flatten_to_2d_for_linear(g, hidden)
    out = _matmul_with_quantized_rhs_legalization(g, linear_input, weight, pretransposed_rhs=True)
    if bias is not None:
        out = g.add(out, bias)
    return g.reshape(out, target_shape)


def _normalize_attention_add_tensor(g: Graph, tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor

    if len(tensor.shape) != 4 or len(target_shape) != 4:
        raise NotImplementedError(
            f"unsupported attention add tensor shape {tuple(tensor.shape)} for target {target_shape}"
        )

    batch, seq_len, heads, head_dim = target_shape
    shape = tuple(int(v) for v in tensor.shape)

    if shape[0] in (1, batch) and shape[1] == heads and shape[2] in (1, seq_len) and shape[3] == head_dim:
        return g.permute(tensor, (0, 2, 1, 3))
    if shape[0] in (1, batch) and shape[1] in (1, seq_len) and shape[2] == heads and shape[3] == head_dim:
        return tensor

    raise NotImplementedError(
        f"unsupported attention add tensor shape {shape} for target {target_shape}"
    )


def _rope_input_is_bhsd(ir: IRGraph, value_id: str) -> bool:
    value = ir.values.get(value_id)
    if value is None or value.producer is None:
        return False
    node = ir.nodes.get(value.producer)
    if node is None:
        return False
    if node.op == "permute":
        permutation = tuple(int(dim) for dim in node.attrs.get("permutation", ()))
        return permutation == (0, 2, 1, 3)
    if node.op == "transpose":
        return int(node.attrs.get("dim0", -1)) == 1 and int(node.attrs.get("dim1", -1)) == 2
    return False


def _lower_softplus(g: Graph, x: Tensor) -> Tensor:
    # Stable fp16 softplus: relu(x) + log(1 + exp(-abs(x))).
    abs_x = g.abs(x)
    neg_abs_x = g.scalar_multiply(abs_x, -1.0)
    exp_term = g.scalar_exp(neg_abs_x)
    log_term = g.scalar_log(g.scalar_add(exp_term, 1.0))
    return g.add(g.relu(x), log_term)


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim


def _normalize_reduction_axes(axis: Any, rank: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(rank))
    if isinstance(axis, int):
        return (_normalize_dim(axis, rank),)
    if isinstance(axis, (list, tuple)):
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_axis in axis:
            if not isinstance(raw_axis, int):
                raise NotImplementedError("reduction axes must be integers")
            reduced_axis = _normalize_dim(raw_axis, rank)
            if reduced_axis in seen:
                continue
            seen.add(reduced_axis)
            normalized.append(reduced_axis)
        if normalized:
            return tuple(sorted(normalized))
    raise NotImplementedError("reduction axes must be an int, a tuple/list of ints, or None")


def _lower_reduction(g: Graph, op: str, x: Tensor, *, axes: tuple[int, ...]) -> Tensor:
    if not axes:
        return x

    fn = getattr(g, op)
    if len(axes) == 1:
        return fn(x, axes[0])

    if op == "variance":
        flattened = g.flatten(x, start_dim=axes[0], end_dim=axes[-1])
        collapsed_axes = set(axes)
        expected_axes = set(range(axes[0], axes[-1] + 1))
        if collapsed_axes != expected_axes:
            raise NotImplementedError("variance currently requires contiguous multi-axis reductions")
        return fn(flattened, axes[0])

    reduced = x
    for axis in sorted(axes, reverse=True):
        reduced = fn(reduced, axis)
    return reduced


def _normalize_index(index: int, dim_size: int) -> int:
    if index < 0:
        index += dim_size
    return index


def _static_shape(shape: Any) -> tuple[int, ...] | None:
    if not isinstance(shape, tuple):
        return None
    dims: list[int] = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except Exception:
            return None
    return tuple(dims)


def _ir_value_dtype(ir: IRGraph, value_id: str) -> str | None:
    value = ir.values.get(value_id)
    if value is None or value.dtype is None:
        return None
    return str(value.dtype).strip().lower()


def _is_integer_index_dtype(dtype: str | None) -> bool:
    return dtype in {"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"}


def _constant_tensor_from_ir(ir: IRGraph, value_id: str) -> torch.Tensor | None:
    const = ir.constants.get(value_id)
    if isinstance(const, torch.nn.Parameter):
        const = const.detach()
    if isinstance(const, torch.Tensor):
        return const.detach().cpu()
    return None


def _try_lower_identity_broadcast_advanced_index(
    g: Graph,
    node: IRNode,
    env: dict[str, Any],
    ir: IRGraph,
) -> Tensor | None:
    if len(node.inputs) != 3:
        return None

    source = _tensor(env, node.inputs[0])
    source_shape = tuple(int(dim) for dim in source.shape)
    if len(source_shape) != 2 or int(source_shape[0]) != 1:
        return None

    output_shape = _static_shape(ir.values[node.outputs[0]].shape if node.outputs and node.outputs[0] in ir.values else None)
    if output_shape is None:
        return None

    batch_index = _constant_tensor_from_ir(ir, node.inputs[1])
    token_index = _constant_tensor_from_ir(ir, node.inputs[2])
    if batch_index is None or token_index is None:
        return None

    try:
        batch_index = torch.broadcast_to(batch_index.to(torch.int64), output_shape)
        token_index = torch.broadcast_to(token_index.to(torch.int64), output_shape)
    except RuntimeError:
        return None

    if torch.count_nonzero(batch_index).item() != 0:
        return None

    flat_index = token_index.reshape(-1)
    expected = torch.arange(int(source_shape[1]), dtype=torch.int64)
    if flat_index.numel() != expected.numel() or not torch.equal(flat_index, expected):
        return None

    return g.reshape(source, output_shape)


def _try_lower_embedding_advanced_index(
    g: Graph,
    node: IRNode,
    env: dict[str, Any],
    ir: IRGraph,
) -> Tensor | None:
    if len(node.inputs) != 2:
        return None
    if not _is_integer_index_dtype(_ir_value_dtype(ir, node.inputs[1])):
        return None

    source = _tensor(env, node.inputs[0])
    indices = _tensor(env, node.inputs[1])
    source_shape = tuple(int(dim) for dim in source.shape)
    if len(source_shape) != 2:
        return None

    output_shape = _static_shape(ir.values[node.outputs[0]].shape if node.outputs and node.outputs[0] in ir.values else None)
    gathered = g.embedding_from_tensor(source, indices)
    if output_shape is not None and tuple(int(dim) for dim in gathered.shape) != tuple(int(dim) for dim in output_shape):
        gathered = g.reshape(gathered, output_shape)
    return gathered


def _try_lower_prefix_mask_advanced_index(
    g: Graph,
    node: IRNode,
    env: dict[str, Any],
    ir: IRGraph,
) -> Tensor | None:
    if len(node.inputs) != 2:
        return None
    if _is_integer_index_dtype(_ir_value_dtype(ir, node.inputs[1])):
        return None

    source = _tensor(env, node.inputs[0])
    mask = _tensor(env, node.inputs[1])
    source_shape = tuple(int(dim) for dim in source.shape)
    mask_shape = tuple(int(dim) for dim in mask.shape)
    if not mask_shape or len(mask_shape) > len(source_shape):
        return None
    if source_shape[: len(mask_shape)] != mask_shape:
        return None

    output_shape = _static_shape(ir.values[node.outputs[0]].shape if node.outputs and node.outputs[0] in ir.values else None)
    if output_shape:
        prefix_elems = 1
        for dim in mask_shape:
            prefix_elems *= int(dim)
        trailing_shape = source_shape[len(mask_shape) :]
        flattened_shape = (prefix_elems, *trailing_shape)
        flattened = source if source_shape == flattened_shape else g.reshape(source, flattened_shape)
        expected_prefix = int(output_shape[0])
        if 0 <= expected_prefix <= prefix_elems:
            selected = g.slice(flattened, axis=0, start=0, length=expected_prefix)
            if tuple(int(dim) for dim in selected.shape) != tuple(int(dim) for dim in output_shape):
                selected = g.reshape(selected, output_shape)
            return selected

    selected = g.masked_select_prefix(source, mask)
    if output_shape is None or not output_shape:
        return selected
    if len(output_shape) != len(selected.shape):
        return selected
    expected_prefix = int(output_shape[0])
    if expected_prefix < int(selected.shape[0]):
        selected = g.slice(selected, axis=0, start=0, length=expected_prefix)
    return selected


def _try_lower_advanced_index(
    g: Graph,
    node: IRNode,
    env: dict[str, Any],
    ir: IRGraph,
) -> Tensor | None:
    lowered = _try_lower_identity_broadcast_advanced_index(g, node, env, ir)
    if lowered is not None:
        return lowered
    lowered = _try_lower_prefix_mask_advanced_index(g, node, env, ir)
    if lowered is not None:
        return lowered
    return _try_lower_embedding_advanced_index(g, node, env, ir)


def _normalize_slice_end(end: int, dim_size: int) -> int:
    if end < 0:
        end += dim_size
    return max(0, min(end, dim_size))


def _legalize_elementwise_binary_inputs(g: Graph, lhs: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
    if lhs.shape == rhs.shape:
        return lhs, rhs

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)

    if lhs_rank > rhs_rank:
        rhs = _reshape_for_trailing_broadcast(g, rhs, lhs.shape)
    elif rhs_rank > lhs_rank:
        lhs = _reshape_for_trailing_broadcast(g, lhs, rhs.shape)

    if lhs.shape == rhs.shape:
        return lhs, rhs

    return lhs, rhs


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


def _lower_binary_op(g: Graph, lhs_value: Any, rhs_value: Any, op: str) -> Tensor:
    if isinstance(lhs_value, Tensor) and isinstance(rhs_value, Tensor):
        lhs, rhs = _legalize_elementwise_binary_inputs(g, lhs_value, rhs_value)
        target_dtype = Graph.FP16 if lhs.dtype == Graph.FP16 and rhs.dtype == Graph.FP16 else Graph.FP32
        lhs = _ensure_tensor_dtype(g, lhs, target_dtype)
        rhs = _ensure_tensor_dtype(g, rhs, target_dtype)
        if op == "add":
            return g.add(lhs, rhs)
        if op == "subtract":
            return g.subtract(lhs, rhs)
        if op == "multiply":
            return g.multiply(lhs, rhs)
        if op == "divide":
            return g.divide(lhs, rhs)
        raise NotImplementedError(f"unsupported binary op: {op}")

    if isinstance(lhs_value, Tensor) and _is_scalar_like(rhs_value):
        lhs_value = _ensure_scalar_math_tensor(g, lhs_value)
        scalar = float(rhs_value)
        if op == "add":
            return g.scalar_add(lhs_value, scalar)
        if op == "subtract":
            return g.scalar_subtract(lhs_value, scalar)
        if op == "multiply":
            return g.scalar_multiply(lhs_value, scalar)
        if op == "divide":
            return g.scalar_divide(lhs_value, scalar)
        raise NotImplementedError(f"unsupported binary op: {op}")

    if _is_scalar_like(lhs_value) and isinstance(rhs_value, Tensor):
        rhs_value = _ensure_scalar_math_tensor(g, rhs_value)
        scalar = float(lhs_value)
        if op == "add":
            return g.scalar_add(rhs_value, scalar)
        if op == "subtract":
            return g.scalar_add(g.scalar_multiply(rhs_value, -1.0), scalar)
        if op == "multiply":
            return g.scalar_multiply(rhs_value, scalar)
        if op == "divide":
            raise NotImplementedError("scalar/tensor divide is not directly supported by Cactus graph ops")
        raise NotImplementedError(f"unsupported binary op: {op}")

    raise TypeError(
        f"unsupported lowered operand types for {op}: "
        f"{type(lhs_value).__name__}, {type(rhs_value).__name__}"
    )


def _lower_compare_op(g: Graph, lhs_value: Any, rhs_value: Any, op: str) -> Tensor:
    if op == "equal":
        not_equal = _lower_compare_op(g, lhs_value, rhs_value, "not_equal")
        return g.scalar_not_equal(not_equal, 1.0)

    if op == "greater":
        delta = _lower_binary_op(g, lhs_value, rhs_value, "subtract")
        delta = _ensure_fp16_tensor(g, delta)
        return g.scalar_not_equal(g.relu(delta), 0.0)

    if op == "less":
        return _lower_compare_op(g, rhs_value, lhs_value, "greater")

    if op == "greater_equal":
        less = _lower_compare_op(g, lhs_value, rhs_value, "less")
        return g.scalar_not_equal(less, 1.0)

    if op == "less_equal":
        greater = _lower_compare_op(g, lhs_value, rhs_value, "greater")
        return g.scalar_not_equal(greater, 1.0)

    if op != "not_equal":
        raise NotImplementedError(f"unsupported compare op: {op}")

    if isinstance(lhs_value, Tensor) and isinstance(rhs_value, Tensor):
        lhs, rhs = _legalize_elementwise_binary_inputs(g, lhs_value, rhs_value)
        target_dtype = Graph.FP16 if lhs.dtype == Graph.FP16 and rhs.dtype == Graph.FP16 else Graph.FP32
        lhs = _ensure_tensor_dtype(g, lhs, target_dtype)
        rhs = _ensure_tensor_dtype(g, rhs, target_dtype)
        return g.not_equal(lhs, rhs)

    if isinstance(lhs_value, Tensor) and _is_scalar_like(rhs_value):
        lhs_value = _ensure_tensor_dtype(
            g,
            lhs_value,
            Graph.FP16 if lhs_value.dtype == Graph.FP16 else Graph.FP32,
        )
        return g.scalar_not_equal(lhs_value, float(rhs_value))

    if _is_scalar_like(lhs_value) and isinstance(rhs_value, Tensor):
        rhs_value = _ensure_tensor_dtype(
            g,
            rhs_value,
            Graph.FP16 if rhs_value.dtype == Graph.FP16 else Graph.FP32,
        )
        return g.scalar_not_equal(rhs_value, float(lhs_value))

    if _is_scalar_like(lhs_value) and _is_scalar_like(rhs_value):
        result = 1.0 if float(lhs_value) != float(rhs_value) else 0.0
        return _materialize_constant_tensor(g, torch.tensor([result], dtype=torch.float32))

    raise TypeError(
        f"unsupported lowered operand types for {op}: "
        f"{type(lhs_value).__name__}, {type(rhs_value).__name__}"
    )


def _reshape_for_trailing_broadcast(g: Graph, tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    tensor_shape = tuple(tensor.shape)
    target_rank = len(target_shape)
    tensor_rank = len(tensor_shape)

    if tensor_rank > target_rank:
        return tensor

    padded_shape = (1,) * (target_rank - tensor_rank) + tensor_shape

    # Only legalize cases that are valid trailing broadcasts, e.g.:
    # (H,) -> (1, H), (1, 1, H), etc.
    for src_dim, tgt_dim in zip(padded_shape, target_shape):
        if src_dim != 1 and src_dim != tgt_dim:
            return tensor

    if padded_shape == tensor_shape:
        return tensor

    return g.reshape(tensor, padded_shape)


def _ensure_fp16_tensor(g: Graph, tensor: Tensor) -> Tensor:
    if tensor.dtype == Graph.FP16:
        return tensor
    return g.precision_cast(tensor, Graph.FP16)


def _ensure_scalar_math_tensor(g: Graph, tensor: Tensor) -> Tensor:
    if tensor.dtype == Graph.FP16:
        return tensor
    return g.precision_cast(tensor, Graph.FP16)


def _ensure_tensor_dtype(g: Graph, tensor: Tensor, dtype: int) -> Tensor:
    if tensor.dtype == dtype:
        return tensor
    return g.precision_cast(tensor, dtype)


def _cat_with_legalized_dtype(g: Graph, tensors: list[Tensor], *, axis: int, dtype: int | None) -> Tensor:
    if not tensors:
        raise NotImplementedError("cat requires at least one tensor")
    if dtype is None:
        dtypes = {tensor.dtype for tensor in tensors}
        dtype = tensors[0].dtype if len(dtypes) == 1 else Graph.FP16
    legalized = [_ensure_tensor_dtype(g, tensor, dtype) for tensor in tensors]
    return g.cat(legalized, axis=axis)


def _invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for index, source_axis in enumerate(permutation):
        inverse[int(source_axis)] = index
    return tuple(inverse)


def _lower_static_strided_slice_via_gather(
    g: Graph,
    x: Tensor,
    *,
    axis: int,
    indices: list[int],
) -> Tensor:
    if not indices:
        return g.slice(x, axis=axis, start=0, length=0)

    indices_tensor = _materialize_constant_tensor(g, torch.tensor(indices, dtype=torch.float32))
    if axis == 0:
        return g.gather(x, indices_tensor)

    permutation = (axis, *[dim for dim in range(len(x.shape)) if dim != axis])
    transposed = g.permute(x, permutation)
    gathered = g.gather(transposed, indices_tensor)
    inverse_permutation = _invert_permutation(permutation)
    return g.permute(gathered, inverse_permutation)


def _lower_where_branch_term(g: Graph, branch_value: Any, mask: Tensor, *, dtype: int | None) -> Tensor:
    if dtype is not None:
        mask = _ensure_tensor_dtype(g, mask, dtype)
    if isinstance(branch_value, Tensor):
        branch_mask = mask if dtype is None else _ensure_tensor_dtype(g, mask, branch_value.dtype)
        return _lower_binary_op(g, branch_value, branch_mask, "multiply")
    return g.scalar_multiply(mask, _clamp_scalar_for_dtype(float(branch_value), mask.dtype))


def _clamp_scalar_for_dtype(value: float, dtype: int | None) -> float:
    if dtype != Graph.FP16:
        return value
    if math.isnan(value):
        return value
    if value == math.inf:
        return 65504.0
    if value == -math.inf:
        return -65504.0
    return max(-65504.0, min(65504.0, value))


def _lower_where_op(g: Graph, node: IRNode, env: dict[str, Any]) -> Tensor:
    condition = _lower_compare_op(g, env[node.inputs[0]], 0.0, "not_equal")
    false_mask = g.scalar_add(g.scalar_multiply(condition, -1.0), 1.0)

    input_index = 1
    if bool(node.attrs.get("true_is_scalar", False)):
        true_value: Any = float(node.attrs["true_value"])
    else:
        true_value = env[node.inputs[input_index]]
        input_index += 1

    if bool(node.attrs.get("false_is_scalar", False)):
        false_value: Any = float(node.attrs["false_value"])
    else:
        false_value = env[node.inputs[input_index]]

    result_dtype: int | None = None
    if isinstance(true_value, Tensor):
        result_dtype = true_value.dtype
    elif isinstance(false_value, Tensor):
        result_dtype = false_value.dtype

    true_term = _lower_where_branch_term(g, true_value, condition, dtype=result_dtype)
    false_term = _lower_where_branch_term(g, false_value, false_mask, dtype=result_dtype)
    return _lower_binary_op(g, true_term, false_term, "add")


def _flatten_to_2d_for_linear(g: Graph, tensor: Tensor) -> Tensor:
    shape = tuple(tensor.shape)
    if len(shape) <= 2:
        return tensor
    leading = 1
    for dim in shape[:-1]:
        leading *= int(dim)
    return g.view(tensor, (leading, int(shape[-1])))


def _broadcast_batch_shape(lhs_batch: tuple[int, ...], rhs_batch: tuple[int, ...]) -> tuple[int, ...] | None:
    rank = max(len(lhs_batch), len(rhs_batch))
    lhs_padded = (1,) * (rank - len(lhs_batch)) + lhs_batch
    rhs_padded = (1,) * (rank - len(rhs_batch)) + rhs_batch
    result: list[int] = []
    for lhs_dim, rhs_dim in zip(lhs_padded, rhs_padded):
        if lhs_dim == rhs_dim:
            result.append(int(lhs_dim))
        elif lhs_dim == 1:
            result.append(int(rhs_dim))
        elif rhs_dim == 1:
            result.append(int(lhs_dim))
        else:
            return None
    return tuple(result)


def _project_broadcast_coord(coord: tuple[int, ...], operand_batch: tuple[int, ...]) -> tuple[int, ...]:
    if not operand_batch:
        return ()
    offset = len(coord) - len(operand_batch)
    projected: list[int] = []
    for axis, dim in enumerate(operand_batch):
        value = coord[offset + axis]
        projected.append(0 if int(dim) == 1 else int(value))
    return tuple(projected)


def _index_batch_prefix(g: Graph, tensor: Tensor, coord: tuple[int, ...]) -> Tensor:
    for index_value in coord:
        tensor = g.index(tensor, index_value, axis=0)
    return tensor


def _lower_static_batched_matmul(
    g: Graph,
    lhs: Tensor,
    rhs: Tensor,
    *,
    output_dtype: int | None = None,
) -> Tensor | None:
    lhs_shape = tuple(int(dim) for dim in lhs.shape)
    rhs_shape = tuple(int(dim) for dim in rhs.shape)
    if len(lhs_shape) < 3 or len(rhs_shape) < 3:
        return None
    if lhs_shape[-1] != rhs_shape[-2]:
        return None

    lhs_batch = lhs_shape[:-2]
    rhs_batch = rhs_shape[:-2]
    batch_shape = _broadcast_batch_shape(lhs_batch, rhs_batch)
    if batch_shape is None:
        return None

    output_matrix_shape = (int(lhs_shape[-2]), int(rhs_shape[-1]))
    rank = len(batch_shape)

    def _build(dim: int, coord_prefix: tuple[int, ...]) -> Tensor:
        if dim == rank:
            lhs_coord = _project_broadcast_coord(coord_prefix, lhs_batch)
            rhs_coord = _project_broadcast_coord(coord_prefix, rhs_batch)
            lhs_slice = _index_batch_prefix(g, lhs, lhs_coord)
            rhs_slice = _index_batch_prefix(g, rhs, rhs_coord)
            out = _matmul_with_quantized_rhs_legalization(
                g,
                lhs_slice,
                rhs_slice,
                output_dtype=output_dtype,
            )
            return g.reshape(out, (1,) * rank + output_matrix_shape)

        pieces = [_build(dim + 1, coord_prefix + (index_value,)) for index_value in range(int(batch_shape[dim]))]
        if len(pieces) == 1:
            return pieces[0]
        return g.cat(pieces, axis=dim)

    return _build(0, ())


def _should_lower_gemma4_decoder_attention_without_kernel(ir: IRGraph, node: IRNode) -> bool:
    if os.environ.get("CACTUS_GEMMA4_DECODER_MANUAL_ATTENTION", "0") != "1":
        return False
    family = str(ir.meta.get("adapter_family") or ir.meta.get("family") or "").strip().lower()
    component = str(ir.meta.get("component", "") or "").strip().lower()
    if family != "gemma4" or component != "decoder":
        return False
    return node.op in {"attention", "scaled_dot_product_attention"}


def _lower_gemma4_decoder_attention_without_kernel(
    g: Graph,
    ir: IRGraph,
    env: dict[str, Any],
    node: IRNode,
) -> Tensor:
    query = _attention_tensor(env, node.inputs[0])
    key = _attention_tensor(env, node.inputs[1])
    value = _attention_tensor(env, node.inputs[2])

    key_transposed = g.permute(key, (0, 1, 3, 2))
    scores = _lower_static_batched_matmul(g, query, key_transposed, output_dtype=Graph.FP16)
    if scores is None:
        raise NotImplementedError("Gemma4 decoder attention requires static batched matmul support")

    scale = float(node.attrs.get("scale", 1.0))
    if scale != 1.0:
        scores = g.scalar_multiply(_ensure_scalar_math_tensor(g, scores), scale)

    additive_mask: Tensor | None = None
    if len(node.inputs) > 3:
        mask_tensor = _ensure_fp16_tensor(g, _tensor(env, node.inputs[3]))
        if bool(node.attrs.get("additive_mask", False)):
            additive_mask = mask_tensor
        else:
            additive_mask = _lower_binary_op(g, mask_tensor, 1.0, "subtract")
            additive_mask = g.scalar_multiply(_ensure_scalar_math_tensor(g, additive_mask), 1.0e4)
    else:
        query_shape = tuple(int(dim) for dim in query.shape)
        key_shape = tuple(int(dim) for dim in key.shape)
        if len(query_shape) >= 4 and len(key_shape) >= 4:
            query_seq = int(query_shape[-2])
            key_seq = int(key_shape[-2])
            additive_mask = _materialize_gemma4_attention_mask(
                g,
                query_seq=query_seq,
                key_seq=key_seq,
                is_causal=bool(node.attrs.get("is_causal", False)),
                window_size=int(node.attrs.get("window_size", 0)),
                dtype=torch.float16,
            )

    if additive_mask is not None:
        scores = _lower_binary_op(g, scores, additive_mask, "add")

    probs = g.softmax(scores, axis=-1)
    probs = _ensure_tensor_dtype(g, probs, value.dtype)
    out = _lower_static_batched_matmul(g, probs, value, output_dtype=value.dtype)
    if out is None:
        raise NotImplementedError("Gemma4 decoder attention output requires static batched matmul support")

    output_value = ir.values.get(node.outputs[0])
    output_dtype = _map_ir_dtype(output_value.dtype) if output_value is not None and output_value.dtype is not None else out.dtype
    if out.dtype != output_dtype:
        out = g.precision_cast(out, output_dtype)
    return out


def _materialize_gemma4_attention_mask(
    g: Graph,
    *,
    query_seq: int,
    key_seq: int,
    is_causal: bool,
    window_size: int,
    dtype: torch.dtype,
) -> Tensor | None:
    if query_seq <= 0 or key_seq <= 0:
        return None
    if not is_causal and window_size <= 0:
        return None

    q_index = np.arange(query_seq, dtype=np.int32)[:, None]
    k_index = np.arange(key_seq, dtype=np.int32)[None, :]
    allowed = np.ones((query_seq, key_seq), dtype=np.bool_)

    if is_causal:
        allowed &= k_index <= q_index
    if window_size > 0 and window_size < key_seq:
        allowed &= k_index >= (q_index - (window_size - 1))

    mask = np.where(allowed, 0.0, -1.0e4).astype(np.float16)
    tensor = torch.from_numpy(mask.reshape(1, 1, query_seq, key_seq)).to(dtype=dtype)
    return _materialize_constant_tensor(g, tensor)


def _legalize_matmul_inputs(
    g: Graph,
    lhs: Tensor,
    rhs: Tensor,
    node: IRNode,
) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    lhs_shape = tuple(lhs.shape)
    rhs_shape = tuple(rhs.shape)
    if len(lhs_shape) <= 2 and len(rhs_shape) <= 2:
        return None

    if len(lhs_shape) > 2 and len(rhs_shape) == 2 and lhs_shape[-1] == rhs_shape[0]:
        leading = 1
        for dim in lhs_shape[:-1]:
            leading *= int(dim)
        lhs_2d = g.reshape(lhs, (leading, int(lhs_shape[-1])))
        output_shape = lhs_shape[:-1] + (int(rhs_shape[1]),)
        return lhs_2d, rhs, output_shape

    # Cactus matmul is 2D-only. Legalize the narrow case where both operands have
    # only singleton leading dims, e.g. rotary helper matmuls:
    # (1, M, K) @ (1, K, N) -> reshape to (M, K) @ (K, N) -> reshape back.
    if any(dim != 1 for dim in lhs_shape[:-2]):
        return None
    if any(dim != 1 for dim in rhs_shape[:-2]):
        return None

    lhs_2d_shape = lhs_shape[-2:]
    rhs_2d_shape = rhs_shape[-2:]
    if lhs_2d_shape[-1] != rhs_2d_shape[0]:
        return None

    output_shape = node.meta.get("shape")
    if not isinstance(output_shape, tuple):
        output_shape = lhs_shape[:-2] + (lhs_2d_shape[0], rhs_2d_shape[1])
    output_shape = tuple(int(v) for v in output_shape)

    lhs_2d = g.reshape(lhs, lhs_2d_shape)
    rhs_2d = g.reshape(rhs, rhs_2d_shape)
    return lhs_2d, rhs_2d, output_shape


def _resolve_expand_shape(input_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(requested_shape) < len(input_shape):
        raise NotImplementedError(f"expand cannot reduce rank: {input_shape} -> {requested_shape}")

    padded_input = (1,) * (len(requested_shape) - len(input_shape)) + input_shape
    resolved: list[int] = []
    for in_dim, req_dim in zip(padded_input, requested_shape):
        if req_dim == -1:
            resolved.append(in_dim)
            continue
        if req_dim < -1:
            raise NotImplementedError(f"invalid expand dimension: {req_dim}")
        resolved.append(int(req_dim))
    return tuple(resolved)


def _lower_repeat(g: Graph, x: Tensor, repeats: tuple[int, ...]) -> Tensor:
    if not repeats:
        return x

    input_shape = tuple(int(dim) for dim in x.shape)
    if len(repeats) < len(input_shape):
        raise NotImplementedError(f"repeat cannot reduce rank: {input_shape} with repeats={repeats}")

    if len(repeats) > len(input_shape):
        padded_shape = (1,) * (len(repeats) - len(input_shape)) + input_shape
        x = g.reshape(x, padded_shape)

    result = x
    for axis, factor in enumerate(repeats):
        if factor < 0:
            raise NotImplementedError(f"repeat does not support negative factors: {repeats}")
        if factor == 0:
            raise NotImplementedError(f"repeat does not support zero factors: {repeats}")
        if factor == 1:
            continue
        result = g.cat([result] * factor, axis=axis)
    return result


def _resolve_reshape_shape(input_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> tuple[int, ...]:
    resolved = [int(v) for v in requested_shape]
    unknown_indices = [idx for idx, dim in enumerate(resolved) if dim == -1]
    if len(unknown_indices) > 1:
        raise NotImplementedError(f"reshape with multiple inferred dimensions is unsupported: {requested_shape}")
    if not unknown_indices:
        return tuple(resolved)

    input_elements = 1
    for dim in input_shape:
        input_elements *= int(dim)

    known_elements = 1
    for dim in resolved:
        if dim != -1:
            known_elements *= int(dim)

    if known_elements == 0 or input_elements % known_elements != 0:
        raise NotImplementedError(f"cannot infer reshape target {requested_shape} from input shape {input_shape}")

    resolved[unknown_indices[0]] = input_elements // known_elements
    return tuple(resolved)


def _match_gqa_expand_alias(ir: IRGraph, expand_node: IRNode) -> str | None:
    target_shape = tuple(int(v) for v in expand_node.attrs["shape"])
    if len(target_shape) != 5:
        return None

    current_value_id = expand_node.inputs[0]
    base_value_id: str | None = None

    while True:
        producer_id = ir.values[current_value_id].producer
        if producer_id is None:
            return None
        producer = ir.nodes[producer_id]
        if producer.op == "slice":
            axis = int(producer.attrs.get("axis", -1))
            input_shape = ir.values[producer.inputs[0]].shape
            if input_shape is None:
                return None
            normalized_axis = _normalize_dim(axis, len(input_shape))
            start = int(producer.attrs.get("start", 0))
            end = int(producer.attrs.get("end", input_shape[normalized_axis]))
            if start != 0 or end < input_shape[normalized_axis]:
                return None
            current_value_id = producer.inputs[0]
            continue
        if producer.op == "unsqueeze":
            base_value_id = producer.inputs[0]
            base_shape = ir.values[base_value_id].shape
            if base_shape is None:
                return None
            unsqueezed_dim = _normalize_dim(int(producer.attrs.get("dim", 0)), len(base_shape) + 1)
            if unsqueezed_dim != 2:
                return None
            break
        if producer.op in {"reshape", "view"}:
            base_value_id = producer.inputs[0]
            base_shape = ir.values[base_value_id].shape
            view_shape = ir.values[current_value_id].shape
            if base_shape is None or view_shape is None:
                return None
            if len(base_shape) != 4 or len(view_shape) != 5:
                return None
            if tuple(int(v) for v in view_shape) != (
                int(base_shape[0]),
                int(base_shape[1]),
                1,
                int(base_shape[2]),
                int(base_shape[3]),
            ):
                return None
            break
        return None

    if base_value_id is None:
        return None
    base_shape = ir.values[base_value_id].shape
    if base_shape is None or len(base_shape) != 4:
        return None

    expected_target = (
        int(base_shape[0]),
        int(base_shape[1]),
        int(target_shape[2]),
        int(base_shape[2]),
        int(base_shape[3]),
    )
    if target_shape != expected_target:
        return None
    if int(target_shape[2]) <= 1:
        return None

    return base_value_id


def _map_ir_dtype(dtype: str) -> int:
    if dtype == "bf16":
        return Graph.FP32
    if dtype == "fp16":
        return Graph.FP16
    if dtype in ("fp32", "fp64"):
        return Graph.FP32
    if dtype == "int8":
        return Graph.INT8
    if dtype in ("int32", "int64", "bool"):
        return Graph.FP32

    raise NotImplementedError(f"unsupported IR dtype: {dtype}")


def _map_ir_or_torch_dtype(dtype: Any) -> int:
    if dtype is None:
        raise NotImplementedError("missing dtype for precision_cast")
    if isinstance(dtype, str):
        if dtype.startswith("torch."):
            return _map_torch_dtype(getattr(torch, dtype.split(".", 1)[1]))
        return _map_ir_dtype(dtype)
    return _map_torch_dtype(dtype)


def _map_torch_dtype(dtype: Any) -> int:
    if dtype == torch.bfloat16:
        return Graph.FP32
    if dtype == torch.float16:
        return Graph.FP16
    if dtype == torch.float32:
        return Graph.FP32
    if dtype == torch.float64:
        return Graph.FP32
    if dtype == torch.int8:
        return Graph.INT8
    if dtype in (torch.int16, torch.int32, torch.int64, torch.bool):
        return Graph.FP32
    raise NotImplementedError(f"unsupported torch dtype: {dtype}")


def _materialize_constant_tensor(g: Graph, tensor_value: torch.Tensor) -> Tensor:
    graph_dtype = _map_torch_dtype(tensor_value.dtype)
    materialized = tensor_value.detach().cpu()
    if graph_dtype == Graph.FP32 and materialized.dtype not in (torch.float32, torch.float64):
        materialized = materialized.to(torch.float32)
    elif graph_dtype == Graph.FP16 and materialized.dtype != torch.float16:
        materialized = materialized.to(torch.float16)
    elif graph_dtype == Graph.INT8 and materialized.dtype != torch.int8:
        materialized = materialized.to(torch.int8)

    tensor = g.input(shape=tuple(materialized.shape), dtype=graph_dtype)
    g.set_input(tensor, materialized, dtype=graph_dtype)
    materialized_constants = getattr(g, "_transpile_materialized_constants", None)
    if isinstance(materialized_constants, list):
        materialized_constants.append(tensor)
    return tensor


def _materialize_constant_torch_dtype(dtype: Any) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        if dtype.startswith("torch."):
            return getattr(torch, dtype.split(".", 1)[1])
        if dtype in ("fp16", "bf16"):
            return torch.float16
        if dtype in ("fp32", "fp64", "int16", "int32", "int64", "bool"):
            return torch.float32
        if dtype == "int8":
            return torch.int8
    if isinstance(dtype, torch.dtype):
        return dtype
    raise NotImplementedError(f"unsupported dtype for constant materialization: {dtype}")


def _torch_dtype_for_graph_dtype(dtype: int) -> torch.dtype:
    if dtype == Graph.FP16:
        return torch.float16
    if dtype == Graph.FP32:
        return torch.float32
    if dtype == Graph.INT8:
        return torch.int8
    raise NotImplementedError(f"unsupported graph dtype for constant materialization: {dtype}")
