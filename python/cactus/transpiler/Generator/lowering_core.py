from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import constants
from . import models
from .errors import UnsupportedLoweringError
from .lowering_basic_ops import *
from .lowering_cache import cache_attention_generation_plan, lower_cache, lower_cache_placeholder, should_lower_cache_placeholder_as_state, lower_attention
from .lowering_constant_ops import *
from .lowering_nn_ops import lower_conv, lower_norm
from .lowering_special_ops import lower_moe, lower_special_cactus, lower_unsupported_semantic
from .lowering_utils import *
from .lowering_weights import bind_weight_placeholder, lower_lfm_grouped_moe_placeholder
from ..IR import models as IRModels


def create_cactus_graph() -> Any:
    try:
        from cactus import Graph
    except ModuleNotFoundError:
        from bindings.cactus import Graph

    return Graph()


def component_weight_resolver(
    component: models.ComponentGraph,
    config: models.GeneratorConfig,
) -> models.WeightResolver | None:
    if config.weights_dir is None:
        return None

    resolver = models.WeightResolver.from_graph(
        component.ir_graph,
        config.weights_dir,
        config.weights_manifest_path,
    )

    if not resolver.records:
        component.warn(f"{component.name}: no converted weight records found in {config.weights_dir}")

    return resolver


def build_lowering_rules(
    extra_lowerings: Mapping[str, models.LoweringRule | models.LoweringFn] | None = None,
) -> dict[str, models.LoweringRule]:
    rules: dict[str, models.LoweringRule] = {}

    add_rules(rules, constants.BINARY_TARGETS, lower_binary)
    add_rules(rules, constants.SCALAR_TARGETS, lower_scalar)
    add_rules(rules, constants.UNARY_TARGETS, lower_unary)
    add_rules(rules, constants.LOG1P_TARGETS, lower_log1p)
    add_rules(rules, constants.POW_TARGETS, lower_pow)
    add_rules(rules, constants.REDUCE_TARGETS, lower_reduce)
    add_rules(rules, constants.SHAPE_TARGETS, lower_shape)
    add_rules(rules, constants.EXPAND_TARGETS, lower_expand)
    add_rules(rules, constants.UNSQUEEZE_TARGETS, lower_unsqueeze)
    add_rules(rules, constants.FLATTEN_TARGETS, lower_flatten)
    add_rules(rules, constants.REPEAT_TARGETS, lower_repeat)
    add_rules(rules, constants.TRANSPOSE_TARGETS, lower_transpose)
    add_rules(rules, constants.SLICE_TARGETS, lower_slice)
    add_rules(rules, constants.INDEX_TARGETS, lower_index)
    add_rules(rules, constants.WHERE_TARGETS, lower_where)
    add_rules(rules, constants.UNFOLD_TARGETS, lower_unfold)
    add_rules(rules, constants.CAT_TARGETS, lower_cat)
    add_rules(rules, constants.MATMUL_TARGETS, lower_matmul)
    add_rules(rules, constants.ADDM_CONST_TARGETS, lower_addmm)
    add_rules(rules, constants.SPLIT_TARGETS, lower_split)
    add_rules(rules, constants.GETITEM_TARGETS, lower_getitem)
    add_rules(rules, constants.TO_COPY_TARGETS, lower_to_copy)
    add_rules(rules, constants.PASS_THROUGH_TARGETS, lower_pass_through)
    add_rules(rules, constants.COPY_TARGETS, lower_copy)
    add_rules(rules, constants.PAD_TARGETS, lower_constant_pad_nd)
    add_rules(rules, constants.CONSTANT_INPUT_TARGETS, lower_constant_input)
    add_rules(rules, constants.NEG_TARGETS, lower_neg)
    add_rules(rules, constants.CUMSUM_TARGETS, lower_cumsum)
    add_rules(rules, constants.SOFTMAX_TARGETS, lower_softmax)
    add_rules(rules, constants.TOPK_TARGETS, lower_topk)
    add_rules(rules, constants.GATHER_TARGETS, lower_gather)
    add_rules(rules, constants.EMBEDDING_TARGETS, lower_embedding)
    add_rules(rules, constants.CLAMP_TARGETS, lower_clamp)
    add_rules(rules, constants.NORM_TARGETS, lower_norm)
    add_rules(rules, constants.CONV_TARGETS, lower_conv)
    add_rules(rules, constants.ATTENTION_TARGETS, lower_attention)
    add_rules(rules, constants.CACHE_TARGETS, lower_cache)
    add_rules(rules, constants.MOE_TARGETS, lower_moe)
    add_rules(rules, constants.SPECIAL_CACTUS_TARGETS, lower_special_cactus)
    add_rules(rules, constants.UNSUPPORTED_SEMANTIC_TARGETS, lower_unsupported_semantic)

    if extra_lowerings:
        for target, rule in extra_lowerings.items():
            if isinstance(rule, models.LoweringRule):
                rules[target] = rule
            else:
                rules[target] = models.LoweringRule(target, rule)

    return rules


def add_rules(rules: dict[str, models.LoweringRule], targets: Any, lower: models.LoweringFn) -> None:
    target_names = targets.keys() if isinstance(targets, dict) else targets

    for target in target_names:
        rules[target] = models.LoweringRule(target, lower)


def lower_component(
    component: models.ComponentGraph,
    config: models.GeneratorConfig,
    lowering_rules: dict[str, models.LoweringRule],
) -> models.ComponentGraph:
    component.graph = create_cactus_graph()
    component.weight_resolver = component_weight_resolver(component, config)
    context = models.GenerationContext(component=component, config=config, lowerings=lowering_rules)
    (
        context.skip_node_names,
        context.cache_state_placeholder_names,
        context.prefill_cache_cat_annotations,
    ) = cache_attention_generation_plan(component.ir_graph)

    for node in IRModels.topological_sort(component.ir_graph):
        try:
            lower_node(context, node)
        except UnsupportedLoweringError as e:
            component.mark_unsupported(node)
            component.warn(str(e))

            if config.strict and not config.allow_unsupported_ops:
                raise
        except KeyError as e:
            component.mark_unsupported(node)
            component.warn(f"{node.name}: missing lowered input {e}")

            if config.strict and not config.allow_unsupported_ops:
                raise UnsupportedLoweringError(f"{node.name}: missing lowered input {e}") from e

    component.save()
    return component


def lower_node(context: models.GenerationContext, node: IRModels.Node) -> None:
    if node.name in context.skip_node_names:
        return

    if node.is_placeholder:
        context.bind(node, lower_placeholder(context, node))
        return

    if node.is_output:
        context.bind(node, lower_output(context, node))
        return

    rule = context.lowering_for(node)

    if rule is None:
        raise UnsupportedLoweringError(f"{node.name}: no lowering rule for {node.target}")

    result = rule.lower(context, node)

    if result is not None:
        context.bind(node, result)


def lower_placeholder(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if should_lower_cache_placeholder_as_state(context, node):
        return lower_cache_placeholder(context, node)

    if node.value_kind in constants.WEIGHT_VALUE_KINDS:
        grouped_moe_weights = lower_lfm_grouped_moe_placeholder(context, node)
        if grouped_moe_weights is not None:
            return grouped_moe_weights

    shape, dynamic_dims = graph_input_shape(node)
    dtype = placeholder_precision(context, node)
    tensor = context.graph.input(shape, dtype=dtype, dynamic_dims=dynamic_dims if any(dynamic_dims) else None)
    logical_name = logical_input_name(context.component.ir_graph, node)

    if node.value_kind in constants.WEIGHT_VALUE_KINDS:
        bind_weight_placeholder(context, node, tensor, logical_name)
        return tensor

    context.component.add_runtime_input(tensor, logical_name)
    return tensor


def placeholder_precision(context: models.GenerationContext, node: IRModels.Node) -> int:
    if node.value_kind in constants.WEIGHT_VALUE_KINDS and context.component.weight_resolver is not None:
        record = context.component.weight_resolver.resolve(node.name)
        if record is not None and record.precision and hasattr(context.graph, record.precision):
            return int(getattr(context.graph, record.precision))

    logical_name = logical_input_name(context.component.ir_graph, node)
    if is_fp16_runtime_input(logical_name) and str(tensor_dtype(node)) in {"torch.float32", "torch.float"}:
        return int(context.graph.FP16)

    return cactus_precision(context.graph, tensor_dtype(node))


def is_fp16_runtime_input(logical_name: str) -> bool:
    return logical_name in constants.FP16_RUNTIME_INPUTS or logical_name.startswith(("cross_k_", "cross_v_"))


def lower_output(context: models.GenerationContext, node: IRModels.Node) -> Any:
    refs = IRModels.extract_node_refs((node.args, node.kwargs))
    values = tuple(context.require(ref) for ref in refs)
    output_index = 0

    for ref, value in zip(refs, values):
        if isinstance(value, (tuple, list)):
            for item in value:
                context.component.add_output(item, logical_output_name(context.component.ir_graph, output_index, ref))
                output_index += 1
        else:
            context.component.add_output(value, logical_output_name(context.component.ir_graph, output_index, ref))
            output_index += 1

    if len(values) == 1:
        return values[0]

    return values


def logical_input_name(graph: IRModels.Graph, node: IRModels.Node) -> str:
    metadata_name = logical_name_from_metadata(node.ir_metadata, ("logical_input", "logical_name", "runtime_input"))

    if metadata_name:
        return metadata_name

    spec_name = logical_input_name_from_specs(graph, node.name)

    if spec_name:
        return spec_name

    if node.cache is not None:
        return cache_logical_name(node)

    target_name = clean_logical_name(node.target)

    if target_name is not None:
        return target_name

    return node.name


def logical_output_name(graph: IRModels.Graph, output_index: int, source_ref: str | None = None) -> str:
    source = graph.nodes_map.get(source_ref) if source_ref is not None else None

    if source is not None:
        metadata_name = logical_name_from_metadata(source.ir_metadata, ("logical_output", "logical_name", "runtime_output"))

        if metadata_name:
            return metadata_name

        if source.cache is not None:
            return cache_logical_name(source)

        if "logits" in source.name or "lm_head" in source.target:
            return "logits"

    spec_name = logical_output_name_from_specs(graph, output_index)

    if spec_name:
        return spec_name

    return f"output_{output_index}"


def logical_input_name_from_specs(graph: IRModels.Graph, node_name: str) -> str | None:
    for spec in graph.input_specs:
        if spec.arg_name != node_name:
            continue

        return clean_logical_name(spec.target) or clean_logical_name(spec.arg_name)

    return None


def logical_output_name_from_specs(graph: IRModels.Graph, output_index: int) -> str | None:
    if output_index < 0 or output_index >= len(graph.output_specs):
        return None

    spec = graph.output_specs[output_index]
    return clean_logical_name(spec.target) or clean_logical_name(spec.arg_name)


def logical_name_from_metadata(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = clean_logical_name(metadata.get(key))

        if value is not None:
            return value

    return None


def clean_logical_name(value: Any) -> str | None:
    if value is None:
        return None

    name = str(value)

    if not name or name == "None":
        return None

    return name


def cache_logical_name(node: IRModels.Node) -> str:
    if node.cache is not None and node.cache.tensor_index is not None:
        return f"past_key_values_{node.cache.tensor_index}"

    return node.name
