from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from . import constants
from . import models
from .errors import UnsupportedLoweringError
from .lowering_utils import *
from ..IR import models as IRModels

LFM_GROUPED_MOE_TARGET_RE = re.compile(r"^layers\.(\d+)\.feed_forward\.experts\.(gate_up_proj|down_proj)$")

def bind_weight_placeholder(context: models.GenerationContext, node: IRModels.Node, tensor: Any, logical_name: str | None = None) -> None:
    resolver = context.component.weight_resolver
    if resolver is None:
        context.component.warn(f"{node.name}: weight lowered as graph input because no weights_dir was provided")
        context.component.add_runtime_input(tensor, logical_name)
        return
    source_target = resolver.source_target_for(node.name) or node.target
    record = resolver.resolve(node.name)
    node_id = models.tensor_node_id(tensor)
    if node_id is None:
        message = f"{node.name}: lowered weight tensor does not expose a Cactus node id"
        if context.config.strict and not context.config.allow_unsupported_ops:
            raise UnsupportedLoweringError(message)
        context.component.warn(message)
        return
    if record is None or record.output_name is None:
        generated = generated_weight_placeholder_values(context, node, source_target)
        if generated is not None:
            values, shape, dtype = generated
            path = write_constant_tensor(context, node, values, shape, dtype)
            context.component.add_weight_binding(
                models.WeightBinding(
                    placeholder=node.name,
                    source_target=source_target,
                    node_id=node_id,
                    path=path,
                    output_name=path,
                    source_name=source_target,
                    value_id=node.name,
                    precision=precision_name(context.graph, dtype),
                    component=context.component.name,
                    binding_kind="generated_constant",
                )
            )
            return
        message = f"{node.name}: could not resolve converted weight for source target {source_target}"
        if context.config.strict and not context.config.allow_unsupported_ops:
            raise UnsupportedLoweringError(message)
        context.component.warn(message)
        context.component.add_runtime_input(tensor, logical_name)
        return
    if should_dequantize_int8_weight_placeholder(record):
        path = write_dequantized_int8_weight_as_fp16(context, node, record)
        context.component.add_weight_binding(
            models.WeightBinding(
                placeholder=node.name,
                source_target=source_target,
                node_id=node_id,
                path=path,
                output_name=path,
                source_name=record.source_name or record.hf_name or source_target,
                value_id=node.name,
                precision="FP16",
                component=record.component,
                scale_factor=record.scale_factor,
                adapter_family=record.adapter_family,
                transform=record.transform,
                qdq_restore=record.qdq_restore,
                binding_kind="generated_dequant_fp16",
            )
        )
        return
    context.component.add_weight_binding(
        models.WeightBinding(
            placeholder=node.name,
            source_target=source_target,
            node_id=node_id,
            path=record.output_name,
            output_name=record.output_name,
            source_name=record.source_name or record.hf_name or source_target,
            value_id=node.name,
            precision=record.precision,
            component=record.component,
            scale_factor=record.scale_factor,
            adapter_family=record.adapter_family,
            transform=record.transform,
            qdq_restore=record.qdq_restore,
        )
    )

def generated_weight_placeholder_values(
    context: models.GenerationContext,
    node: IRModels.Node,
    source_target: str,
) -> tuple[list[float], tuple[int, ...], int] | None:
    if not source_target.endswith((".pos_emb.inv_freq", ".rotary_emb.inv_freq", ".rotary_emb.original_inv_freq")):
        return None
    shape = concrete_shape(meta_shape(node))
    if shape is None or len(shape) != 1:
        return None
    rope_theta = model_rope_theta(context)
    if rope_theta is None:
        return None
    rotary_dim = int(shape[0]) * 2
    values = [1.0 / (float(rope_theta) ** (index / rotary_dim)) for index in range(0, rotary_dim, 2)]
    return values, tuple(shape), cactus_precision(context.graph, tensor_dtype(node))

def model_rope_theta(context: models.GenerationContext) -> float | None:
    weights_dir = context.config.weights_dir
    if weights_dir is None:
        return None
    config_path = weights_dir / "config.json"
    if not config_path.is_file():
        config_path = weights_dir / "hf_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else {}
    rope_parameters = config.get("rope_parameters") or text_config.get("rope_parameters")
    if isinstance(rope_parameters, Mapping):
        value = rope_parameters.get("rope_theta") or rope_parameters.get("theta")
        if value is not None:
            return float(value)
    value = config.get("rope_theta") or config.get("rotary_emb_base") or text_config.get("rope_theta") or text_config.get("rotary_emb_base")
    return float(value) if value is not None else None

def lower_lfm_grouped_moe_placeholder(context: models.GenerationContext, node: IRModels.Node) -> models.LfmMoeWeightBundle | None:
    resolver = context.component.weight_resolver
    if resolver is None:
        return None
    source_target = resolver.source_target_for(node.name) or node.target
    parsed = parse_lfm_grouped_moe_target(source_target)
    if parsed is None:
        return None
    layer_index, grouped_kind = parsed
    placeholder_shape = concrete_shape(meta_shape(node))
    num_experts = placeholder_shape[0] if placeholder_shape is not None and placeholder_shape else None
    if not isinstance(num_experts, int):
        num_experts = count_lfm_moe_experts(resolver, layer_index)
    if num_experts <= 0:
        raise UnsupportedLoweringError(f"{node.name}: could not infer LFM MoE expert count for {source_target}")
    if grouped_kind == "gate_up_proj":
        return models.LfmMoeWeightBundle(
            w1_weights=tuple(
                lfm_expert_weight_input(context, node, layer_index, expert_index, "w1")
                for expert_index in range(num_experts)
            ),
            w3_weights=tuple(
                lfm_expert_weight_input(context, node, layer_index, expert_index, "w3")
                for expert_index in range(num_experts)
            ),
        )
    return models.LfmMoeWeightBundle(
        w2_weights=tuple(
            lfm_expert_weight_input(context, node, layer_index, expert_index, "w2")
            for expert_index in range(num_experts)
        )
    )

def parse_lfm_grouped_moe_target(source_target: str) -> tuple[int, str] | None:
    target = source_target
    while target.startswith("model."):
        target = target[len("model."):]
    match = LFM_GROUPED_MOE_TARGET_RE.fullmatch(target)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)

def count_lfm_moe_experts(resolver: models.WeightResolver, layer_index: int) -> int:
    prefix = f"model.layers.{layer_index}.feed_forward.experts."
    expert_indices: set[int] = set()
    for record in resolver.records:
        for alias in record.aliases:
            if not alias.startswith(prefix) or not alias.endswith(".w1.weight"):
                continue
            suffix = alias[len(prefix):]
            expert = suffix.split(".", 1)[0]
            if expert.isdigit():
                expert_indices.add(int(expert))
    return max(expert_indices) + 1 if expert_indices else 0

def lfm_expert_weight_input(
    context: models.GenerationContext,
    grouped_node: IRModels.Node,
    layer_index: int,
    expert_index: int,
    role: str,
) -> Any:
    source_target = f"model.layers.{layer_index}.feed_forward.experts.{expert_index}.{role}.weight"
    record = resolve_required_source_weight(context, grouped_node, source_target)
    tensor = context.graph.input(tuple(record.shape), dtype=weight_record_precision(context, record))
    bind_weight_record(
        context,
        grouped_node,
        tensor,
        source_target,
        record,
        value_id=f"{grouped_node.name}:{role}:{expert_index}",
    )
    return tensor

def resolve_required_source_weight(
    context: models.GenerationContext,
    node: IRModels.Node,
    source_target: str,
) -> models.WeightRecord:
    resolver = context.component.weight_resolver
    if resolver is None:
        raise UnsupportedLoweringError(f"{node.name}: no weight resolver available for {source_target}")
    for candidate in (source_target, source_target.removeprefix("model.")):
        record = resolver.resolve_source_target(candidate)
        if record is not None and record.output_name is not None:
            return record
    raise UnsupportedLoweringError(f"{node.name}: could not resolve converted weight for source target {source_target}")

def weight_record_precision(context: models.GenerationContext, record: models.WeightRecord) -> int:
    if record.precision and hasattr(context.graph, record.precision):
        return int(getattr(context.graph, record.precision))
    return int(getattr(context.graph, constants.DEFAULT_INPUT_PRECISION))

def should_dequantize_int8_weight_placeholder(record: models.WeightRecord | None) -> bool:
    if record is None or record.precision != "INT8" or record.output_name is None:
        return False
    adapter_family = (record.adapter_family or "").lower()
    return adapter_family.startswith("lfm") and "conv_depthwise.weights" in record.output_name

def bind_weight_record(
    context: models.GenerationContext,
    node: IRModels.Node,
    tensor: Any,
    source_target: str,
    record: models.WeightRecord,
    *,
    value_id: str,
) -> None:
    node_id = models.tensor_node_id(tensor)
    if node_id is None:
        raise UnsupportedLoweringError(f"{node.name}: lowered weight tensor does not expose a Cactus node id")
    context.component.add_weight_binding(
        models.WeightBinding(
            placeholder=node.name,
            source_target=source_target,
            node_id=node_id,
            path=str(record.output_name),
            output_name=str(record.output_name),
            source_name=record.source_name or record.hf_name or source_target,
            value_id=value_id,
            precision=record.precision,
            component=record.component,
            scale_factor=record.scale_factor,
            adapter_family=record.adapter_family,
            transform=record.transform,
            qdq_restore=record.qdq_restore,
        )
    )
