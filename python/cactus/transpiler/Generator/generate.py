from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import component_split
from . import constants
from . import models
from ..Converter import models as CModels
from ..Fusions import models as FModels
from ..IR import models as IRModels
from ..RuntimePlan import models as RPModels


class UnsupportedLoweringError(NotImplementedError):
    pass


GraphInput = CModels.LayerMap | IRModels.Graph
ComponentInput = GraphInput | Mapping[str, GraphInput]


def generate(
    ir_output: ComponentInput,
    output_dir: str | Path | None = None,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    config: models.GeneratorConfig | None = None,
    lowerings: Mapping[str, models.LoweringRule | models.LoweringFn] | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    if config is None and output_dir is None:
        raise ValueError("generate requires output_dir unless a GeneratorConfig is provided")

    generator_config = config or models.GeneratorConfig(
        output_dir=Path(output_dir),
        weights_dir=Path(weights_dir) if weights_dir is not None else None,
        weights_manifest_path=Path(weights_manifest_path) if weights_manifest_path is not None else None,
        graph_suffix=constants.DEFAULT_GRAPH_SUFFIX,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )
    components = component_graphs_from_input(ir_output, generator_config, component_name, model_profile)
    lowering_rules = build_lowering_rules(lowerings)

    for component in components:
        lower_component(component, generator_config, lowering_rules)

    return models.GenerationResult.from_components(components)


def generate_from_json(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    layer_map = read_layer_map(input_path)
    return generate(
        layer_map,
        output_dir,
        model_profile=model_profile,
        component_name=component_name,
        weights_dir=weights_dir,
        weights_manifest_path=weights_manifest_path,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )


def generate_bundle(
    ir_output: ComponentInput,
    bundle_dir: str | Path,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    metadata: dict[str, str] | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    bundle_path = Path(bundle_dir)
    components_dir = bundle_path / "components"
    result = generate(
        ir_output,
        output_dir=components_dir,
        model_profile=model_profile,
        component_name=component_name,
        weights_dir=weights_dir,
        weights_manifest_path=weights_manifest_path,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )
    plan = RPModels.runtime_plan_from_generation_result(
        result,
        bundle_dir=bundle_path,
        model_profile=model_profile,
        metadata=metadata,
    )
    engine_manifest_path, runtime_plan_path = plan.write(bundle_path)
    return replace(result, engine_manifest_path=engine_manifest_path, runtime_plan_path=runtime_plan_path)


################################################# Generator Utils!!!!!!! #################################################


def read_layer_map(input_path: str | Path) -> CModels.LayerMap:
    path = Path(input_path)
    return CModels.LayerMap.model_validate_json(path.read_text(encoding="utf-8"))


def component_graphs_from_input(
    ir_output: ComponentInput,
    config: models.GeneratorConfig,
    component_name: str | None,
    model_profile: Any | None,
) -> tuple[models.ComponentGraph, ...]:
    if isinstance(ir_output, Mapping):
        graphs = {name: graph_from_input(component_input) for name, component_input in ir_output.items()}
        split_graphs = component_split.split_component_graphs(graphs, model_profile)

        if split_graphs is not None:
            return tuple(
                models.ComponentGraph.from_ir(name, graph, config)
                for name, graph in split_graphs.items()
            )

        return tuple(
            models.ComponentGraph.from_ir(name, graph, config)
            for name, graph in graphs.items()
        )

    graph = graph_from_input(ir_output)
    name = component_name or default_component_name(graph, model_profile)
    return (models.ComponentGraph.from_ir(name, graph, config),)


def graph_from_input(ir_output: GraphInput) -> IRModels.Graph:
    if isinstance(ir_output, IRModels.Graph):
        return ir_output

    if isinstance(ir_output, CModels.LayerMap):
        return IRModels.Graph.from_map(ir_output)

    raise TypeError(f"Unsupported generator input: {type(ir_output).__name__}")


def default_component_name(graph: IRModels.Graph, model_profile: Any | None) -> str:
    profile_name = getattr(model_profile, "model_profiles", None)
    model_name = profile_name or graph.model_name or constants.DEFAULT_COMPONENT_NAME
    task = graph.task

    if task:
        return f"{model_name}_{task}"

    return model_name


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
    if should_lower_cache_placeholder_as_state(node):
        return lower_cache_placeholder(context, node)

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

    return cactus_precision(context.graph, tensor_dtype(node))


def should_lower_cache_placeholder_as_state(node: IRModels.Node) -> bool:
    if node.cache is None:
        return False

    if node.cache.kind == FModels.CacheKind.CONV:
        return any(child.target in {"cactus.conv_cache_append", "cactus.conv_cache_initialize"} for child in node.children)

    if node.cache.kind == FModels.CacheKind.KV:
        return bool(node.children) and all(child.target in {"cactus.kv_cache_append", "cactus.attention_cached"} for child in node.children)

    return False


def lower_cache_placeholder(context: models.GenerationContext, node: IRModels.Node) -> Any:
    annotation = require_cache_annotation(node)

    if annotation.kind == FModels.CacheKind.KV:
        cache_state = context.graph.kv_cache_state(
            kv_cache_capacity(context, annotation),
            required_cache_int(annotation.num_kv_heads, node, "num_kv_heads"),
            required_cache_int(annotation.head_dim, node, "head_dim"),
            window_size=int(annotation.window_size or 0),
            sink_size=0,
            num_slots=1,
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    if annotation.kind == FModels.CacheKind.CONV:
        cache_state = context.graph.conv_cache_state(
            required_cache_int(annotation.window_size, node, "window_size"),
            required_cache_int(annotation.hidden_dim, node, "hidden_dim"),
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    raise UnsupportedLoweringError(f"{node.name}: unsupported cache placeholder kind {annotation.kind}")


def bind_weight_placeholder(context: models.GenerationContext, node: IRModels.Node, tensor: Any, logical_name: str | None = None) -> None:
    resolver = context.component.weight_resolver

    if resolver is None:
        context.component.warn(f"{node.name}: weight lowered as graph input because no weights_dir was provided")
        context.component.add_runtime_input(tensor, logical_name)
        return

    source_target = resolver.source_target_for(node.name) or node.target
    record = resolver.resolve(node.name)

    if record is None or record.output_name is None:
        message = f"{node.name}: could not resolve converted weight for source target {source_target}"

        if context.config.strict and not context.config.allow_unsupported_ops:
            raise UnsupportedLoweringError(message)

        context.component.warn(message)
        context.component.add_runtime_input(tensor, logical_name)
        return

    node_id = models.tensor_node_id(tensor)

    if node_id is None:
        message = f"{node.name}: lowered weight tensor does not expose a Cactus node id"

        if context.config.strict and not context.config.allow_unsupported_ops:
            raise UnsupportedLoweringError(message)

        context.component.warn(message)
        return

    context.component.add_weight_binding(
        models.WeightBinding(
            placeholder=node.name,
            source_target=source_target,
            node_id=node_id,
            path=record.output_name,
            output_name=record.output_name,
            precision=record.precision,
            component=record.component,
        )
    )


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


def lower_binary(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    method = constants.BINARY_TARGETS[node.target]

    if len(inputs) == 2:
        left, right = align_binary_inputs(context, node, inputs)
        return getattr(context.graph, method)(left, right)

    scalar = scalar_attr(node, "other")

    if len(inputs) == 1 and scalar is not None:
        scalar_method = binary_method_to_scalar_method(method)
        return getattr(context.graph, scalar_method)(inputs[0], scalar)

    raise unsupported_arity(node, len(inputs), "two tensor inputs or one tensor plus scalar attr")


def lower_scalar(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method, attr_name = constants.SCALAR_TARGETS[node.target]
    value = scalar_attr(node, attr_name)

    if value is None:
        value = scalar_attr(node, "arg_1")

    if value is None:
        if method == "scalar_equal":
            return context.graph.scalar_multiply(inputs[0], 0.0)

        if method == "scalar_not_equal":
            return context.graph.scalar_add(context.graph.scalar_multiply(inputs[0], 0.0), 1.0)

        raise UnsupportedLoweringError(f"{node.name}: scalar lowering {node.target} missing attr {attr_name}")

    return getattr(context.graph, method)(inputs[0], value)


def align_binary_inputs(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    target_shape = output_shape(node)
    left = align_input_to_shape(context, inputs[0], node.parents[0], target_shape)
    right = align_input_to_shape(context, inputs[1], node.parents[1], target_shape)
    return align_binary_precision(context, left, right)


def align_binary_precision(context: models.GenerationContext, left: Any, right: Any) -> tuple[Any, Any]:
    left_precision = getattr(left, "dtype", None)
    right_precision = getattr(right, "dtype", None)

    if left_precision is None or right_precision is None or left_precision == right_precision:
        return left, right

    return left, cast_to_precision(context, right, left_precision)


def align_input_to_shape(context: models.GenerationContext, value: Any, source_node: IRModels.Node, target_shape: tuple[Any, ...]) -> Any:
    value = align_value_to_declared_shape(context, value, source_node)
    source_shape = meta_shape(source_node)

    if shape_matches_tensor(source_node, target_shape):
        return value

    if element_count(source_shape) == element_count(target_shape):
        return context.graph.reshape(value, target_shape)

    return value


def align_value_to_declared_shape(context: models.GenerationContext, value: Any, source_node: IRModels.Node) -> Any:
    actual_shape = tuple(getattr(value, "shape", ()))
    declared_shape = meta_shape(source_node)

    if not actual_shape or not declared_shape or len(declared_shape) <= len(actual_shape):
        return value

    resolved_shape = resolve_declared_shape_from_actual(declared_shape, actual_shape)

    if resolved_shape is None or tuple(resolved_shape) == actual_shape:
        return value

    return context.graph.reshape(value, resolved_shape)


def resolve_declared_shape_from_actual(declared_shape: tuple[Any, ...], actual_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    actual_index = len(actual_shape) - 1
    resolved = [1 for _ in declared_shape]

    for declared_index in range(len(declared_shape) - 1, -1, -1):
        declared_dim = concrete_dim(declared_shape[declared_index])

        if actual_index >= 0 and (declared_dim is None or declared_dim == actual_shape[actual_index]):
            resolved[declared_index] = actual_shape[actual_index]
            actual_index -= 1
            continue

        if declared_dim == 1:
            resolved[declared_index] = 1
            continue

        return None

    if actual_index >= 0:
        return None

    return tuple(resolved)


def lower_unary(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.UNARY_TARGETS[node.target]
    return getattr(context.graph, method)(inputs[0])


def lower_log1p(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.scalar_log(context.graph.scalar_add(inputs[0], 1.0))


def lower_pow(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    exponent = scalar_attr(node, "exponent")

    if exponent is None:
        raise UnsupportedLoweringError(f"{node.name}: pow lowering missing exponent attr")

    return context.graph.pow(inputs[0], exponent)


def lower_reduce(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.REDUCE_TARGETS[node.target]
    axis = axis_attr(node)

    if axis is None:
        raise UnsupportedLoweringError(f"{node.name}: reduce lowering missing axis/dim attr")

    normalized_axis = normalize_dim(axis, len(meta_shape(node.parents[0])))
    reduced = getattr(context.graph, method)(inputs[0], normalized_axis)
    target_shape = output_shape(node)
    dropped_shape = reduction_dropped_shape(meta_shape(node.parents[0]), normalized_axis)

    if bool(node.attrs.get("keepdim", False)) or (
        tuple(dropped_shape) != tuple(target_shape)
        and element_count(dropped_shape) == element_count(target_shape)
    ):
        return context.graph.reshape(reduced, target_shape)

    return reduced


def lower_shape(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    method = constants.SHAPE_TARGETS[node.target]
    shape = shape_attr(node)

    return getattr(context.graph, method)(inputs[0], shape)


def lower_expand(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    target_shape = shape_attr(node)

    if shape_matches_tensor(node.parents[0], target_shape):
        return inputs[0]

    if element_count(meta_shape(node.parents[0])) == element_count(target_shape):
        return context.graph.reshape(inputs[0], target_shape)

    try:
        return context.graph.expand(inputs[0], target_shape)
    except AttributeError as e:
        raise UnsupportedLoweringError(
            f"{node.name}: native Cactus expand is unavailable for broadcast shape "
            f"{meta_shape(node.parents[0])} -> {target_shape}"
        ) from e


def lower_unsqueeze(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.reshape(inputs[0], output_shape(node))


def lower_flatten(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.flatten(
        inputs[0],
        start_dim=int(node.attrs.get("start_dim", 0)),
        end_dim=int(node.attrs.get("end_dim", -1)),
    )


def lower_transpose(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)

    if can_alias_quantized_weight_transpose(context, node, inputs[0]):
        return inputs[0]

    permutation = node.attrs.get("permutation")

    if permutation is not None:
        return context.graph.permute(inputs[0], tuple_ints(permutation))

    if node.target == "aten.t.default":
        return context.graph.transpose(inputs[0])

    dim0 = node.attrs.get("dim0")
    dim1 = node.attrs.get("dim1")

    if dim0 is None or dim1 is None:
        raise UnsupportedLoweringError(f"{node.name}: transpose lowering missing permutation or dim attrs")

    return context.graph.permute(inputs[0], swap_permutation(parent_rank(node), int(dim0), int(dim1)))


def can_alias_quantized_weight_transpose(context: models.GenerationContext, node: IRModels.Node, value: Any) -> bool:
    return (
        is_weight_transpose_node(node)
        and is_cq_tensor(context, value)
        and bool(node.children)
        and all(child.target in constants.MATMUL_TARGETS for child in node.children)
    )


def is_weight_transpose_node(node: IRModels.Node) -> bool:
    if not node.parents:
        return False

    parent = node.parents[0]
    if not parent.is_placeholder or parent.value_kind not in constants.WEIGHT_VALUE_KINDS:
        return False

    if node.target == "aten.t.default":
        return True

    rank = len(meta_shape(parent))
    if rank < 2:
        return False

    permutation = node.attrs.get("permutation")
    if permutation is not None:
        return tuple_ints(permutation) == swap_permutation(rank, rank - 2, rank - 1)

    dim0 = node.attrs.get("dim0")
    dim1 = node.attrs.get("dim1")
    if dim0 is None or dim1 is None:
        return False

    return {
        normalize_dim(int(dim0), rank),
        normalize_dim(int(dim1), rank),
    } == {rank - 2, rank - 1}


def is_cq_tensor(context: models.GenerationContext, value: Any) -> bool:
    return getattr(value, "dtype", None) in {
        context.graph.CQ1,
        context.graph.CQ2,
        context.graph.CQ3,
        context.graph.CQ4,
    }


def lower_slice(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    axis_value = node.attrs.get("axis")

    if axis_value is None:
        axis_value = node.attrs.get("dim")

    if axis_value is None:
        if shape_matches_tensor(node.parents[0], output_shape(node)):
            return inputs[0]

        raise UnsupportedLoweringError(f"{node.name}: slice lowering missing axis/dim attr")

    axis = normalize_dim(int(axis_value), len(meta_shape(node.parents[0])))

    if node.target == "aten.select.int" or "index_value" in node.attrs:
        return context.graph.index(inputs[0], int(node.attrs.get("index_value", node.attrs.get("index", 0))), axis=axis)

    start = int(node.attrs.get("start", 0) or 0)
    step = node.attrs.get("step", 1)

    if step is not None and int(step) != 1:
        target_shape = output_shape(node)
        length = target_shape[axis] if axis < len(target_shape) else slice_length(node, start)
        return context.graph.strided_slice(inputs[0], axis, start, length, int(step))

    length = slice_length(node, start)
    return context.graph.slice(inputs[0], axis, start, length=length)


def lower_index(context: models.GenerationContext, node: IRModels.Node) -> Any:
    if node.target == "aten.index.Tensor":
        return lower_index_tensor(context, node)

    inputs = require_input_count(context, node, 1)
    index_value = node.attrs.get("index_value", node.attrs.get("index"))

    if index_value is None:
        raise UnsupportedLoweringError(f"{node.name}: index lowering missing index_value/index attr")

    return context.graph.index(inputs[0], int(index_value), axis=int(node.attrs.get("axis", node.attrs.get("dim", 0))))


def lower_index_tensor(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if len(inputs) < 2:
        raise unsupported_arity(node, len(inputs), "source tensor plus at least one index tensor")

    source = inputs[0]
    source_shape = meta_shape(node.parents[0])
    target_shape = meta_shape(node)

    if len(source_shape) == 3 and len(target_shape) == 2 and concrete_dim(source_shape[0]) == 1:
        feature_count = concrete_dim(source_shape[1])
        feature_dim = concrete_dim(source_shape[2])

        if feature_count is not None and feature_dim is not None:
            return context.graph.reshape(source, (feature_count, feature_dim))

    if len(source_shape) == 2 and len(target_shape) == 4 and concrete_dim(source_shape[0]) == 1:
        source_length = concrete_dim(source_shape[1])
        target_length = concrete_dim(target_shape[-1])

        if source_length is not None and target_length is not None:
            reshaped = context.graph.reshape(source, (1, 1, 1, source_length))

            if target_length < source_length:
                return context.graph.slice(reshaped, 3, 0, length=target_length)

            if target_length == source_length:
                return reshaped

    concrete_target_shape = concrete_shape(target_shape)

    if concrete_target_shape is not None and element_count(source_shape) == element_count(concrete_target_shape):
        return context.graph.reshape(source, concrete_target_shape)

    raise UnsupportedLoweringError(f"{node.name}: unsupported aten.index.Tensor pattern")


def lower_where(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 3)
    output_precision = cactus_precision(context.graph, tensor_dtype(node))
    true_value = cast_to_precision(context, inputs[1], output_precision)
    false_value = cast_to_precision(context, inputs[2], output_precision)
    return context.graph.where(inputs[0], true_value, false_value)


def lower_unfold(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    dimension = int(node.attrs.get("dimension", node.attrs.get("dim", node.attrs.get("arg_1", 0))))
    size = int(node.attrs.get("size", node.attrs.get("arg_2", 1)))
    step = int(node.attrs.get("step", node.attrs.get("arg_3", 1)))
    return context.graph.unfold(inputs[0], dimension, size, step)


def lower_cat(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if not inputs:
        raise unsupported_arity(node, 0, "at least one input")

    passthrough = empty_cat_passthrough(node, inputs)

    if passthrough is not None:
        return passthrough

    axis = normalize_dim(int(node.attrs.get("axis", node.attrs.get("dim", 0))), cat_rank(node))
    return context.graph.cat(inputs, axis=axis)


def lower_matmul(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)

    if node.target == "aten.bmm.default":
        return lower_bmm(context, node, inputs)

    lhs = matmul_activation_operand(context, inputs[0])
    rhs = inputs[1]

    if len(node.parents) >= 2 and is_weight_transpose_node(node.parents[1]):
        original_weight_node = node.parents[1].parents[0]
        original_weight = context.lookup(original_weight_node.name)

        if original_weight is not None and is_cq_tensor(context, original_weight):
            return context.graph.matmul(lhs, original_weight, pretransposed_rhs=True)

    if not is_cq_tensor(context, rhs):
        rhs = matmul_activation_operand(context, rhs)

    return context.graph.matmul(
        lhs,
        rhs,
        pretransposed_rhs=bool(node.attrs.get("pretransposed_rhs", False)),
    )


def matmul_activation_operand(context: models.GenerationContext, value: Any) -> Any:
    if getattr(value, "dtype", None) == context.graph.FP32:
        return cast_to_precision(context, value, context.graph.FP16)

    return value


def empty_cat_passthrough(node: IRModels.Node, inputs: tuple[Any, ...]) -> Any | None:
    target_shape = meta_shape(node)
    non_empty_inputs = []

    for input_value, parent in zip(inputs, node.parents):
        parent_shape = meta_shape(parent)
        parent_elements = element_count(parent_shape)

        if parent_elements == 0:
            continue

        non_empty_inputs.append((input_value, parent_shape))

    if len(non_empty_inputs) != 1:
        return None

    input_value, input_shape = non_empty_inputs[0]

    if tuple(input_shape) == tuple(target_shape):
        return input_value

    return None


def cat_rank(node: IRModels.Node) -> int:
    target_shape = meta_shape(node)

    if target_shape:
        return len(target_shape)

    return len(meta_shape(node.parents[0])) if node.parents else 0


def lower_bmm(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> Any:
    left_shape = meta_shape(node.parents[0])
    right_shape = meta_shape(node.parents[1])

    if len(left_shape) != 3 or len(right_shape) != 3:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires rank-3 inputs")

    if left_shape[0] != right_shape[0]:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires matching batch dimensions")

    batch_size = concrete_dim(left_shape[0])

    if batch_size is None:
        raise UnsupportedLoweringError(f"{node.name}: bmm lowering requires a concrete batch size")

    if batch_size == 1:
        left_2d = matmul_activation_operand(context, context.graph.reshape(inputs[0], (left_shape[1], left_shape[2])))
        right_2d = matmul_activation_operand(context, context.graph.reshape(inputs[1], (right_shape[1], right_shape[2])))
        product = context.graph.matmul(left_2d, right_2d)
        return context.graph.reshape(product, output_shape(node))

    batch_outputs = []

    for batch_index in range(batch_size):
        left_2d = matmul_activation_operand(context, context.graph.index(inputs[0], batch_index, axis=0))
        right_2d = matmul_activation_operand(context, context.graph.index(inputs[1], batch_index, axis=0))
        product = context.graph.matmul(left_2d, right_2d)
        batch_outputs.append(context.graph.reshape(product, (1, left_shape[1], right_shape[2])))

    return context.graph.cat(batch_outputs, axis=0)


def lower_addmm(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 3)
    product = context.graph.matmul(
        matmul_activation_operand(context, inputs[1]),
        matmul_activation_operand(context, inputs[2]),
    )
    return context.graph.add(inputs[0], product)


def lower_split(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    split_sizes = node.attrs.get("split_sizes")

    if not isinstance(split_sizes, list):
        raise UnsupportedLoweringError(f"{node.name}: split lowering requires split_sizes list")

    axis = normalize_dim(int(node.attrs.get("axis", node.attrs.get("dim", 0))), len(meta_shape(node.parents[0])))
    offset = 0
    outputs = []

    for size in split_sizes:
        size_int = int(size)
        outputs.append(context.graph.slice(inputs[0], axis, offset, length=size_int))
        offset += size_int

    return tuple(outputs)


def lower_getitem(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    index = int(node.attrs.get("index", 0))
    source = inputs[0]

    if isinstance(source, (tuple, list)):
        return source[index]

    if index == 0:
        return source

    raise UnsupportedLoweringError(f"{node.name}: getitem index {index} requires tuple-producing parent")


def lower_to_copy(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    dtype = node.attrs.get("dtype")

    if dtype is None or dtype == "torch.bool":
        return inputs[0]

    precision_name = constants.DTYPE_TO_PRECISION.get(str(dtype))

    if precision_name is None:
        return inputs[0]

    return context.graph.precision_cast(inputs[0], cactus_precision(context.graph, str(dtype)))


def lower_pass_through(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_at_least_one_input(context, node)
    return inputs[0]


def lower_copy(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_at_least_one_input(context, node)
    return inputs[-1]


def lower_constant_pad_nd(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    pads = node.attrs.get("pad", node.attrs.get("arg_1"))
    value = scalar_attr(node, "value")

    if value is None:
        value = scalar_attr(node, "arg_2")

    if value is None:
        value = 0.0

    if not isinstance(pads, (list, tuple)) or len(pads) % 2 != 0:
        raise UnsupportedLoweringError(f"{node.name}: constant_pad_nd lowering requires an even pad list")

    result = inputs[0]
    parent_shape = meta_shape(node.parents[0])
    current_shape = list(parent_shape)
    rank = len(parent_shape)
    positive_pads = [0 for _ in pads]

    for pair_index in range(0, len(pads), 2):
        left_pad = int(pads[pair_index])
        right_pad = int(pads[pair_index + 1])
        positive_pads[pair_index] = max(left_pad, 0)
        positive_pads[pair_index + 1] = max(right_pad, 0)

        if left_pad >= 0 and right_pad >= 0:
            continue

        axis = rank - 1 - (pair_index // 2)

        if axis < 0:
            raise UnsupportedLoweringError(f"{node.name}: pad rank exceeds input rank")

        axis_size = current_shape[axis]

        if not isinstance(axis_size, int):
            raise UnsupportedLoweringError(f"{node.name}: cannot crop symbolic pad dimension {axis_size}")

        start = max(-left_pad, 0)
        end_crop = max(-right_pad, 0)
        length = axis_size - start - end_crop

        if length < 0:
            raise UnsupportedLoweringError(f"{node.name}: pad crop removes more than the full axis")

        result = context.graph.slice(result, axis, start, length=length)
        current_shape[axis] = length

    if any(positive_pads):
        return context.graph.pad(result, positive_pads, float(value))

    return result


def lower_constant_input(context: models.GenerationContext, node: IRModels.Node) -> Any:
    context.component.warn(f"{node.name}: constant-producing op {node.target} lowered as graph input")
    shape, dynamic_dims = graph_input_shape(node)
    dtype = cactus_precision(context.graph, tensor_dtype(node))
    tensor = context.graph.input(shape, dtype=dtype, dynamic_dims=dynamic_dims if any(dynamic_dims) else None)
    context.component.add_runtime_input(tensor, node.name)
    return tensor


def lower_neg(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    return context.graph.scalar_multiply(inputs[0], -1.0)


def lower_softmax(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    axis = axis_attr(node, default=-1)

    if axis is None:
        raise UnsupportedLoweringError(f"{node.name}: softmax lowering missing axis/dim attr")

    return context.graph.softmax(inputs[0], axis=normalize_dim(axis, len(meta_shape(node.parents[0]))))


def lower_topk(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    k = node.attrs.get("k")

    if k is None:
        raise UnsupportedLoweringError(f"{node.name}: topk lowering missing k attr")

    return context.graph.topk(inputs[0], int(k))


def lower_gather(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)
    axis = axis_attr(node, default=0)
    return context.graph.gather(inputs[0], inputs[1], axis=axis if axis is not None else 0)


def lower_embedding(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 2)
    return context.graph.embedding_from_tensor(inputs[0], inputs[1])


def lower_clamp(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = require_input_count(context, node, 1)
    lo = node.attrs.get("lo", node.attrs.get("min", node.attrs.get("arg_1")))
    hi = node.attrs.get("hi", node.attrs.get("max", node.attrs.get("arg_2")))

    if lo is None and hi is None:
        return inputs[0]

    result = inputs[0]

    if lo is not None:
        keep = context.graph.scalar_greater_equal(result, lo)
        result = context.graph.add(
            context.graph.multiply(result, keep),
            context.graph.scalar_multiply(context.graph.logical_not(keep), lo),
        )

    if hi is not None:
        keep = context.graph.scalar_less_equal(result, hi)
        result = context.graph.add(
            context.graph.multiply(result, keep),
            context.graph.scalar_multiply(context.graph.logical_not(keep), hi),
        )

    return result


def lower_norm(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.rms_norm":
        require_len(node, inputs, 2)
        return context.graph.rms_norm(inputs[0], inputs[1], eps=epsilon_attr(node))

    if target in {"cactus.layernorm", "cactus.layer_norm", "aten.native_layer_norm.default", "aten.layer_norm.default"}:
        require_len(node, inputs, 2)
        bias = inputs[2] if len(inputs) > 2 else None
        return context.graph.layernorm(inputs[0], inputs[1], bias=bias, eps=epsilon_attr(node))

    if target in {"cactus.groupnorm", "cactus.group_norm", "aten.native_group_norm.default", "aten.group_norm.default"}:
        require_len(node, inputs, 3)
        return context.graph.groupnorm(inputs[0], inputs[1], inputs[2], node.attrs.get("num_groups", 1), eps=epsilon_attr(node))

    if target in {"cactus.batchnorm", "cactus.batch_norm", "aten.native_batch_norm.default", "aten.batch_norm.default"}:
        require_len(node, inputs, 5)
        return context.graph.batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], axis=int(node.attrs.get("axis", 1)), eps=epsilon_attr(node))

    raise UnsupportedLoweringError(f"{node.name}: unsupported norm target {target}")


def lower_conv(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target.startswith(constants.CACTUS_TARGET_PREFIX):
        method = target.removeprefix(constants.CACTUS_TARGET_PREFIX)
        return lower_named_conv(context, node, method, inputs)

    require_len(node, inputs, 2)
    bias = inputs[2] if len(inputs) > 2 else None
    rank = output_rank(node)

    if target == "aten.conv1d.default" or rank == 3:
        return lower_generic_conv1d(context, node, inputs, bias)

    if target == "aten.conv2d.default" or rank == 4:
        specialized = lower_specialized_conv2d(context, node, inputs, bias)

        if specialized is not None:
            return specialized

        return context.graph.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=first_int(node.attrs.get("stride"), 1),
            padding=first_int(node.attrs.get("padding"), 0),
            dilation=first_int(node.attrs.get("dilation"), 1),
            groups=int(node.attrs.get("groups", 1)),
        )

    raise UnsupportedLoweringError(f"{node.name}: cannot infer conv rank for {target}")


def lower_named_conv(context: models.GenerationContext, node: IRModels.Node, method: str, inputs: tuple[Any, ...]) -> Any:
    require_len(node, inputs, 2)
    bias = inputs[2] if len(inputs) > 2 else None

    if method == "conv1d":
        return lower_generic_conv1d(context, node, inputs, bias)

    if method == "conv1d_causal":
        x = inputs[0]

        if node.attrs.get("layout") == "batch_hidden_sequence":
            x = context.graph.permute(x, (0, 2, 1))

        output = context.graph.conv1d_causal(x, inputs[1], kernel_size=int(node.attrs.get("kernel_size", 0)), dilation=first_int(node.attrs.get("dilation"), 1))

        if node.attrs.get("layout") == "batch_hidden_sequence":
            return context.graph.permute(output, (0, 2, 1))

        return output

    if method == "conv1d_k3":
        require_plain_conv1d_attrs(node, method)
        return context.graph.conv1d_k3(inputs[0], inputs[1], stride=first_int(node.attrs.get("stride"), 1))

    if method == "conv1d_k7s3":
        require_len(node, inputs, 3)
        return context.graph.conv1d_k7s3(inputs[0], inputs[1], inputs[2])

    if method in {"conv1d_same_depthwise_k9", "conv1d_pointwise", "conv2d_k3s2p1", "conv2d_depthwise_k3s2p1", "conv2d_pointwise_1x1", "conv2d_k3s1p1"}:
        return getattr(context.graph, method)(inputs[0], inputs[1], bias=bias)

    if method == "conv2d":
        specialized = lower_specialized_conv2d(context, node, inputs, bias)

        if specialized is not None:
            return specialized

        return context.graph.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=first_int(node.attrs.get("stride"), 1),
            padding=first_int(node.attrs.get("padding"), 0),
            dilation=first_int(node.attrs.get("dilation"), 1),
            groups=int(node.attrs.get("groups", 1)),
        )

    raise UnsupportedLoweringError(f"{node.name}: unsupported Cactus conv method {method}")


def lower_generic_conv1d(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...], bias: Any | None) -> Any:
    stride = first_int(node.attrs.get("stride"), 1)
    padding = first_int(node.attrs.get("padding"), 0)
    dilation = first_int(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)
    input_shape = meta_shape(node.parents[0]) if node.parents else ()
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()

    if (
        padding == 0
        and dilation == 1
        and len(input_shape) == 3
        and len(weight_shape) == 3
        and groups == input_shape[1]
        and weight_shape[0] == input_shape[1]
        and weight_shape[1] == 1
    ):
        return context.graph.conv1d_depthwise(inputs[0], inputs[1], bias=bias, stride=stride)

    require_plain_conv1d_attrs(node, "conv1d")
    return context.graph.conv1d(inputs[0], inputs[1], bias=bias, stride=stride)


def lower_specialized_conv2d(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...], bias: Any | None) -> Any | None:
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()
    stride = tuple_int_values(node.attrs.get("stride"), 1)
    padding = tuple_int_values(node.attrs.get("padding"), 0)
    dilation = tuple_int_values(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)

    if len(weight_shape) != 4 or dilation != (1, 1):
        return None

    kernel = tuple(int(dim) for dim in weight_shape[2:])

    if kernel == (3, 3) and stride == (2, 2) and padding == (1, 1):
        if groups != 1 and weight_shape[1] == 1:
            return context.graph.conv2d_depthwise_k3s2p1(inputs[0], inputs[1], bias=bias)

        return context.graph.conv2d_k3s2p1(inputs[0], inputs[1], bias=bias)

    if kernel == (3, 3) and stride == (1, 1) and padding == (1, 1) and groups == 1:
        return context.graph.conv2d_k3s1p1(inputs[0], inputs[1], bias=bias)

    if kernel == (1, 1) and stride == (1, 1) and padding == (0, 0) and groups == 1:
        return context.graph.conv2d_pointwise_1x1(inputs[0], inputs[1], bias=bias)

    return None


def require_plain_conv1d_attrs(node: IRModels.Node, method: str) -> None:
    padding = first_int(node.attrs.get("padding"), 0)
    dilation = first_int(node.attrs.get("dilation"), 1)
    groups = first_int(node.attrs.get("groups"), 1)

    if padding != 0 or dilation != 1 or groups != 1:
        raise UnsupportedLoweringError(
            f"{node.name}: {method} lowering only supports padding=0, dilation=1, groups=1; "
            f"got padding={padding}, dilation={dilation}, groups={groups}"
        )


def lower_attention(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if node.target == "cactus.attention" or node.target == "aten.scaled_dot_product_attention.default":
        require_len(node, inputs, 3)
        query, key, value, mask = attention_inputs_for_layout(context, node, inputs)
        query = cast_to_precision(context, query, context.graph.FP16)
        key = cast_to_precision(context, key, context.graph.FP16)
        value = cast_to_precision(context, value, context.graph.FP16)
        output = context.graph.attention(
            query,
            key,
            value,
            scale=float(node.attrs.get("scale", 1.0)),
            is_causal=bool(node.attrs.get("is_causal", True)),
            position_offset=int(node.attrs.get("position_offset", 0)),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask,
            additive_mask=bool(node.attrs.get("additive_mask", False)),
        )
        return attention_output_for_layout(context, node, output)

    if node.target == "cactus.attention_cached":
        require_len(node, inputs, 5)
        return context.graph.attention_cached(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            scale=float(node.attrs.get("scale", 1.0)),
            position_offset=int(node.attrs.get("position_offset", 0)),
            window_size=int(node.attrs.get("window_size", 0)),
            v_head_dim=int(node.attrs.get("v_head_dim", 0)),
        )

    raise UnsupportedLoweringError(f"{node.name}: unsupported attention target {node.target}")


def attention_inputs_for_layout(context: models.GenerationContext, node: IRModels.Node, inputs: tuple[Any, ...]) -> tuple[Any, Any, Any, Any | None]:
    mask = inputs[3] if len(inputs) > 3 else None

    if node.attrs.get("input_layout") == "bhqd_bhds_bhsd":
        return (
            context.graph.permute(inputs[0], (0, 2, 1, 3)),
            context.graph.permute(inputs[1], (0, 3, 1, 2)),
            context.graph.permute(inputs[2], (0, 2, 1, 3)),
            mask,
        )

    return inputs[0], inputs[1], inputs[2], mask


def attention_output_for_layout(context: models.GenerationContext, node: IRModels.Node, output: Any) -> Any:
    if node.attrs.get("output_layout") == "bhqd":
        return context.graph.permute(output, (0, 2, 1, 3))

    return output


def lower_cache(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.kv_cache_append":
        if len(inputs) == 1:
            cache_state = create_cache_state_for_cache_output(context, node)
            context.graph.kv_cache_append(inputs[0], cache_state, window_size=int(node.attrs.get("window_size", 0)), sink_size=int(node.attrs.get("sink_size", 0)))
            return cache_state

        require_len(node, inputs, 2)
        require_cache_state_parent(node, 1, "cactus.kv_cache_state")
        context.graph.kv_cache_append(inputs[0], inputs[1], window_size=int(node.attrs.get("window_size", 0)), sink_size=int(node.attrs.get("sink_size", 0)))
        return inputs[1]

    if target == "cactus.conv_cache_append":
        require_len(node, inputs, 2)
        require_cache_state_parent(node, 1, "cactus.conv_cache_state")
        window = context.graph.conv_cache_append(inputs[0], inputs[1])
        return conv_cache_window_to_ir_layout(context, node, window)

    if target == "cactus.conv_cache_initialize":
        if len(inputs) == 1:
            cache_state = create_cache_state_for_cache_output(context, node)
            rows = conv_cache_rows_for_native(context, node, inputs[0])
            context.graph.conv_cache_initialize(rows, cache_state)
            return cache_state

        require_len(node, inputs, 2)
        rows = conv_cache_rows_for_native(context, node, inputs[0])
        context.graph.conv_cache_initialize(rows, inputs[1])
        return inputs[1]

    if target == "cactus.recurrent_cache_write":
        require_len(node, inputs, 2)
        return context.graph.recurrent_cache_write(inputs[0], inputs[1])

    if target == "cactus.kv_cache_state":
        return context.graph.kv_cache_state(
            required_int_attr(node, "max_seq_len"),
            required_int_attr(node, "num_kv_heads"),
            required_int_attr(node, "head_dim"),
            window_size=int(node.attrs.get("window_size", 0)),
            sink_size=int(node.attrs.get("sink_size", 0)),
            num_slots=int(node.attrs.get("num_slots", 1)),
        )

    if target == "cactus.conv_cache_state":
        return context.graph.conv_cache_state(required_int_attr(node, "window_size"), required_int_attr(node, "hidden_dim"))

    if target == "cactus.recurrent_cache_state":
        return context.graph.recurrent_cache_state(shape_attr(node), dtype=cactus_precision(context.graph, tensor_dtype(node)))

    raise UnsupportedLoweringError(f"{node.name}: unsupported cache target {target}")


def require_cache_state_parent(node: IRModels.Node, parent_index: int, expected_target: str) -> None:
    if parent_index >= len(node.parents):
        raise UnsupportedLoweringError(f"{node.name}: {node.target} missing cache parent {parent_index}")

    parent = node.parents[parent_index]

    if parent.target == expected_target:
        return

    if expected_target == "cactus.conv_cache_state" and parent.cache is not None and parent.cache.kind == FModels.CacheKind.CONV:
        return

    if expected_target == "cactus.kv_cache_state" and parent.cache is not None and parent.cache.kind == FModels.CacheKind.KV:
        return

    raise UnsupportedLoweringError(
        f"{node.name}: {node.target} requires parent {parent_index} to be {expected_target}; got {parent.target}. "
        "Raw HF cache tensors need a cache-state bridge before native cache append lowering."
    )


def require_cache_annotation(node: IRModels.Node) -> IRModels.CacheAnnotation:
    if node.cache is None:
        raise UnsupportedLoweringError(f"{node.name}: cache lowering requires cache annotation metadata")

    return node.cache


def required_cache_int(value: int | None, node: IRModels.Node, name: str) -> int:
    if value is None:
        raise UnsupportedLoweringError(f"{node.name}: cache lowering missing {name}")

    return int(value)


def kv_cache_capacity(context: models.GenerationContext, annotation: IRModels.CacheAnnotation) -> int:
    capacity = int(annotation.sequence_length or 1)

    if context.component.ir_graph.task == "decode_with_cache":
        capacity += 1

    return max(capacity, 1)


def create_cache_state_for_cache_output(context: models.GenerationContext, node: IRModels.Node) -> Any:
    annotation = require_cache_annotation(node)

    if annotation.kind == FModels.CacheKind.KV:
        cache_state = context.graph.kv_cache_state(
            kv_cache_capacity(context, annotation),
            required_cache_int(annotation.num_kv_heads, node, "num_kv_heads"),
            required_cache_int(annotation.head_dim, node, "head_dim"),
            window_size=int(annotation.window_size or 0),
            sink_size=0,
            num_slots=1,
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    if annotation.kind == FModels.CacheKind.CONV:
        cache_state = context.graph.conv_cache_state(
            required_cache_int(annotation.window_size, node, "window_size"),
            required_cache_int(annotation.hidden_dim, node, "hidden_dim"),
        )
        record_cache_state_binding(context, node, cache_state, annotation)
        return cache_state

    raise UnsupportedLoweringError(f"{node.name}: unsupported cache output kind {annotation.kind}")


def record_cache_state_binding(context: models.GenerationContext, node: IRModels.Node, cache_state: Any, annotation: IRModels.CacheAnnotation) -> None:
    node_id = models.tensor_node_id(cache_state)

    if node_id is None:
        raise UnsupportedLoweringError(f"{node.name}: cache state tensor does not expose a Cactus node id")

    context.component.add_cache_state_binding(RPModels.cache_state_binding_from_annotation(annotation, node_id))


def conv_cache_rows_for_native(context: models.GenerationContext, node: IRModels.Node, rows: Any) -> Any:
    rows_shape = meta_shape(node.parents[0]) if node.parents else output_shape(node)

    if len(rows_shape) == 3:
        batch, hidden_dim, window_size = rows_shape

        if batch != 1:
            raise UnsupportedLoweringError(f"{node.name}: conv cache initialize currently supports batch size 1")

        return context.graph.reshape(context.graph.permute(rows, (0, 2, 1)), (int(window_size), int(hidden_dim)))

    if len(rows_shape) == 2:
        return rows

    raise UnsupportedLoweringError(f"{node.name}: conv cache initialize rows must be rank 2 or 3, got shape {rows_shape}")


def conv_cache_window_to_ir_layout(context: models.GenerationContext, node: IRModels.Node, window: Any) -> Any:
    target_shape = output_shape(node)

    if len(target_shape) != 3:
        return window

    batch, hidden_dim, window_size = target_shape

    if batch != 1:
        raise UnsupportedLoweringError(f"{node.name}: conv cache append layout bridge currently supports batch size 1")

    return context.graph.reshape(context.graph.transpose(window), (int(batch), int(hidden_dim), int(window_size)))


def lower_moe(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if node.target == "cactus.dense_mlp_tq_fused":
        require_len(node, inputs, 4)
        return context.graph.dense_mlp_tq_fused(inputs[0], inputs[1], inputs[2], inputs[3], product_scale=float(node.attrs.get("product_scale", 1.0)))

    if node.target == "cactus.moe_layer_gated":
        if len(inputs) < 6:
            raise UnsupportedLoweringError(
                f"{node.name}: cactus.moe_layer_gated requires hidden, routing_probs, topk_indices, w1, w3, and w2 inputs; "
                "packed/grouped MoE lowering still needs a weight-unpacking policy"
            )

        return context.graph.moe_layer_gated(
            inputs[0],
            inputs[1],
            inputs[2],
            ensure_tensor_sequence(inputs[3]),
            ensure_tensor_sequence(inputs[4]),
            ensure_tensor_sequence(inputs[5]),
            required_int_attr(node, "num_experts"),
            required_int_attr(node, "num_experts_per_tok"),
            normalize_routing=bool(node.attrs.get("normalize_routing", True)),
            epsilon=float(node.attrs.get("epsilon", 1e-6)),
            routed_scaling_factor=float(node.attrs.get("routed_scaling_factor", 1.0)),
        )

    if node.target == "cactus.moe_layer_ungated":
        if len(inputs) < 5:
            raise unsupported_arity(node, len(inputs), "hidden, routing_probs, topk_indices, w1, and w2")

        return context.graph.moe_layer_ungated(
            inputs[0],
            inputs[1],
            inputs[2],
            ensure_tensor_sequence(inputs[3]),
            ensure_tensor_sequence(inputs[4]),
            required_int_attr(node, "num_experts"),
            required_int_attr(node, "num_experts_per_tok"),
            normalize_routing=bool(node.attrs.get("normalize_routing", True)),
            epsilon=float(node.attrs.get("epsilon", 1e-6)),
            routed_scaling_factor=float(node.attrs.get("routed_scaling_factor", 1.0)),
        )

    raise UnsupportedLoweringError(f"{node.name}: unsupported MoE target {node.target}")


def lower_special_cactus(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.rope":
        require_len(node, inputs, 1)
        return context.graph.rope(
            inputs[0],
            theta=float(attr_value(node, "theta", 10_000.0)),
            position_offset=int(attr_value(node, "position_offset", 0)),
        )

    if target == "cactus.rope_gptj":
        require_len(node, inputs, 1)
        return context.graph.rope_gptj(
            inputs[0],
            theta=float(attr_value(node, "theta", 10_000.0)),
            position_offset=int(attr_value(node, "position_offset", 0)),
            rot_dim=int(attr_value(node, "rot_dim", 0)),
        )

    if target == "cactus.glu":
        require_len(node, inputs, 1)
        return context.graph.glu(inputs[0], axis=axis_attr(node, default=-1))

    if target == "cactus.lstm_cell":
        require_len(node, inputs, 7)
        return context.graph.lstm_cell(*inputs[:7])

    if target == "cactus.gated_deltanet_decode":
        require_len(node, inputs, 6)
        return context.graph.gated_deltanet_decode(*inputs[:6], scale=float(node.attrs.get("scale", 1.0)))

    if target == "cactus.gated_deltanet_prefill":
        require_len(node, inputs, 6)
        return context.graph.gated_deltanet_prefill(*inputs[:6], chunk_size=int(node.attrs.get("chunk_size", 1)), scale=float(node.attrs.get("scale", 1.0)))

    if target == "cactus.lfm_short_conv_decode":
        require_len(node, inputs, 2)
        weight = lfm_short_conv_decode_weight(context, node, inputs[1])
        product = context.graph.multiply(inputs[0], weight)
        return context.graph.sum(product, axis=normalize_dim(-1, len(meta_shape(node.parents[0]))))

    if target == "cactus.rel_pos_bias":
        require_len(node, inputs, 2)
        return context.graph.rel_pos_bias(inputs[0], inputs[1], scale=float(node.attrs.get("scale") or 1.0))

    if target == "cactus.sample":
        require_len(node, inputs, 1)
        return context.graph.sample(inputs[0], temperature=float(node.attrs.get("temperature", 0.6)), top_p=float(node.attrs.get("top_p", 0.95)), top_k=int(node.attrs.get("top_k", 20)))

    if target == "cactus.scatter_topk":
        require_len(node, inputs, 2)
        return context.graph.scatter_topk(inputs[0], inputs[1], num_classes=required_int_attr(node, "num_classes"))

    if target == "cactus.gaussian_topk":
        require_len(node, inputs, 1)
        return context.graph.gaussian_topk(inputs[0], ppf=float(node.attrs.get("ppf", 0.0)))

    if target == "cactus.altup_predict":
        require_len(node, inputs, 2)
        return context.graph.altup_predict(inputs[0], list(inputs[1:]))

    if target == "cactus.altup_correct":
        require_len(node, inputs, 3)
        return context.graph.altup_correct(inputs[0], inputs[1], list(inputs[2:]))

    if target == "cactus.stft":
        require_len(node, inputs, 2)
        return context.graph.stft(inputs[0], inputs[1], stride=required_int_attr(node, "stride"), num_fft_bins=required_int_attr(node, "num_fft_bins"))

    if target == "cactus.rfft":
        require_len(node, inputs, 1)
        return context.graph.rfft(inputs[0])

    if target == "cactus.irfft":
        require_len(node, inputs, 1)
        return context.graph.irfft(inputs[0], output_length=required_int_attr(node, "output_length"))

    if target == "cactus.spectrogram":
        require_len(node, inputs, 2)
        return context.graph.spectrogram(
            inputs[0],
            inputs[1],
            frame_length=required_int_attr(node, "frame_length"),
            hop_length=required_int_attr(node, "hop_length"),
            fft_length=required_int_attr(node, "fft_length"),
            power=float(node.attrs.get("power", 2.0)),
        )

    if target == "cactus.image_preprocess":
        require_len(node, inputs, 1)
        if not has_attrs(node, ("src_width", "src_height", "target_width", "target_height", "patch_size", "channels")):
            return context.graph.reshape(inputs[0], output_shape(node))

        return context.graph.image_preprocess(
            inputs[0],
            src_width=required_int_attr(node, "src_width"),
            src_height=required_int_attr(node, "src_height"),
            target_width=required_int_attr(node, "target_width"),
            target_height=required_int_attr(node, "target_height"),
            patch_size=required_int_attr(node, "patch_size"),
            channels=required_int_attr(node, "channels"),
            rescale_factor=float(node.attrs.get("rescale_factor", 1.0)),
            mean=node.attrs.get("mean", (0.0, 0.0, 0.0)),
            std_dev=node.attrs.get("std_dev", (1.0, 1.0, 1.0)),
        )

    if target == "cactus.bilinear_interpolation":
        require_len(node, inputs, 1)
        return context.graph.bilinear_interpolation(inputs[0], required_int_attr(node, "dst_height"), required_int_attr(node, "dst_width"))

    raise UnsupportedLoweringError(f"{node.name}: unsupported special Cactus target {target}")


def lfm_short_conv_decode_weight(context: models.GenerationContext, node: IRModels.Node, weight: Any) -> Any:
    cache_window_shape = meta_shape(node.parents[0]) if node.parents else ()
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()

    if len(cache_window_shape) != 3:
        return weight

    batch_size, hidden_dim, window_size = cache_window_shape

    if not all(isinstance(dim, int) for dim in cache_window_shape):
        return weight

    if tuple(weight_shape) == tuple(cache_window_shape):
        return weight

    if len(weight_shape) == 2 and tuple(weight_shape) == (hidden_dim, window_size):
        return context.graph.reshape(weight, (batch_size, hidden_dim, window_size))

    if len(weight_shape) == 3 and tuple(weight_shape) in {
        (1, hidden_dim, window_size),
        (hidden_dim, 1, window_size),
    }:
        return context.graph.reshape(weight, (batch_size, hidden_dim, window_size))

    raise UnsupportedLoweringError(
        f"{node.name}: lfm short conv decode weight shape {weight_shape} is incompatible with cache window {cache_window_shape}"
    )


def lower_unsupported_semantic(context: models.GenerationContext, node: IRModels.Node) -> Any:
    raise UnsupportedLoweringError(f"{node.name}: {node.target} has no safe Cactus lowering yet")


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
        if isinstance(dim, int) and dim >= 0:
            dims.append(dim)
        elif isinstance(dim, str) and dim.isdigit():
            dims.append(int(dim))
        elif index < len(resolved_output_shape):
            dims.append(resolved_output_shape[index])
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


def dump_result_manifest(result: models.GenerationResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "component_paths": {name: str(path_) for name, path_ in result.component_paths.items()},
                "component_manifest_paths": {name: str(path_) for name, path_ in result.component_manifest_paths.items()},
                "engine_manifest_path": str(result.engine_manifest_path) if result.engine_manifest_path is not None else None,
                "runtime_plan_path": str(result.runtime_plan_path) if result.runtime_plan_path is not None else None,
                "unsupported_nodes": list(result.unsupported_nodes),
                "warnings": list(result.warnings),
                "ok": result.ok,
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    return path
