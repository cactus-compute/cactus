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


def _note(*items: str) -> dict[str, M.ConstraintSpec]:
    return {"note": {"items": items}}


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


DIRECT_GRAPHS: dict[str, M.FusionGraph] = {
    "add": _single_node_graph("add", "add", ("a", "b")),
    "subtract": _single_node_graph("subtract", "subtract", ("a", "b")),
    "multiply": _single_node_graph("multiply", "multiply", ("a", "b")),
    "divide": _single_node_graph("divide", "divide", ("a", "b")),
    "abs": _single_node_graph("abs", "abs", ("x",)),
    "pow": _single_node_graph(
        "pow",
        "pow",
        ("x",),
        attr_captures=(M.AttrCapture("exponent", "pow", "exponent", required=False),),
    ),
    "sqrt": _single_node_graph("sqrt", "sqrt", ("x",)),
    "scalar_add": _single_node_graph(
        "scalar_add",
        "scalar_add",
        ("x",),
        attr_captures=(M.AttrCapture("value", "scalar_add", "other", required=False),),
    ),
    "scalar_subtract": _single_node_graph(
        "scalar_subtract",
        "scalar_subtract",
        ("x",),
        attr_captures=(M.AttrCapture("value", "scalar_subtract", "other", required=False),),
    ),
    "scalar_multiply": _single_node_graph(
        "scalar_multiply",
        "scalar_multiply",
        ("x",),
        attr_captures=(M.AttrCapture("value", "scalar_multiply", "other", required=False),),
    ),
    "scalar_divide": _single_node_graph(
        "scalar_divide",
        "scalar_divide",
        ("x",),
        attr_captures=(M.AttrCapture("value", "scalar_divide", "other", required=False),),
    ),
    "scalar_floor_divide": _single_node_graph(
        "scalar_floor_divide",
        "scalar_floor_divide",
        ("x",),
        attr_captures=(M.AttrCapture("value", "scalar_floor_divide", "other", required=False),),
    ),
    "scalar_exp": _single_node_graph("scalar_exp", "scalar_exp", ("x",)),
    "scalar_sqrt": _single_node_graph("scalar_sqrt", "scalar_sqrt", ("x",)),
    "scalar_cos": _single_node_graph("scalar_cos", "scalar_cos", ("x",)),
    "scalar_sin": _single_node_graph("scalar_sin", "scalar_sin", ("x",)),
    "scalar_log": _single_node_graph("scalar_log", "scalar_log", ("x",)),
    "mean": _single_node_graph(
        "mean",
        "mean",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "mean", "dim", default=-1, required=False),),
    ),
    "sum": _single_node_graph(
        "sum",
        "sum",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "sum", "dim", default=-1, required=False),),
    ),
    "variance": _single_node_graph(
        "variance",
        "variance",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "variance", "dim", default=-1, required=False),),
    ),
    "min": _single_node_graph(
        "min",
        "min",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "min", "dim", default=-1, required=False),),
    ),
    "max": _single_node_graph(
        "max",
        "max",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "max", "dim", default=-1, required=False),),
    ),
    "softmax": _single_node_graph(
        "softmax",
        "softmax",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "softmax", "dim", default=-1, required=False),),
    ),
    "topk": _single_node_graph(
        "topk",
        "topk_direct",
        ("x",),
        attr_captures=(M.AttrCapture("k", "topk_direct", "k", required=True),),
    ),
    "view": _single_node_graph(
        "view",
        "view",
        ("x",),
        attr_captures=(M.AttrCapture("shape", "view", "shape", required=False),),
    ),
    "reshape": _single_node_graph(
        "reshape",
        "reshape",
        ("x",),
        attr_captures=(M.AttrCapture("shape", "reshape", "shape", required=False),),
    ),
    "flatten": _single_node_graph(
        "flatten",
        "flatten",
        ("x",),
        attr_captures=(
            M.AttrCapture("start_dim", "flatten", "start_dim", default=0, required=False),
            M.AttrCapture("end_dim", "flatten", "end_dim", default=-1, required=False),
        ),
    ),
    "transpose": _single_node_graph(
        "transpose",
        "transpose",
        ("x",),
        attr_captures=(M.AttrCapture("permutation", "transpose", "permutation", required=False),),
    ),
    "slice": _single_node_graph(
        "slice",
        "slice",
        ("x",),
        attr_captures=(
            M.AttrCapture("axis", "slice", "dim", required=False),
            M.AttrCapture("start", "slice", "start", required=False),
            M.AttrCapture("end", "slice", "end", required=False),
            M.AttrCapture("step", "slice", "step", required=False),
        ),
    ),
    "index": _single_node_graph(
        "index",
        "index",
        ("x",),
        attr_captures=(
            M.AttrCapture("index_value", "index", "index", required=False),
            M.AttrCapture("axis", "index", "dim", default=0, required=False),
        ),
    ),
    "cat": _graph("cat", "cat", ("cat",), inputs=(_variadic_input("values", "cat", 0, min_count=2),), attr_captures=(M.AttrCapture("axis", "cat", "dim", default=0, required=False),)),
    "gather": _single_node_graph("gather", "gather", ("tensor", "indices")),
    "relu": _single_node_graph("relu", "relu", ("x",)),
    "silu": _single_node_graph("silu", "silu", ("x",)),
    "gelu": _single_node_graph("gelu", "gelu", ("x",)),
    "gelu_erf": _single_node_graph("gelu_erf", "gelu_erf", ("x",)),
    "sigmoid": _single_node_graph("sigmoid", "sigmoid", ("x",)),
    "tanh": _single_node_graph("tanh", "tanh", ("x",)),
    "clamp": _single_node_graph(
        "clamp",
        "clamp",
        ("x",),
        attr_captures=(
            M.AttrCapture("lo", "clamp", "min", required=False),
            M.AttrCapture("hi", "clamp", "max", required=False),
        ),
    ),
    "glu": _single_node_graph(
        "glu",
        "glu",
        ("x",),
        attr_captures=(M.AttrCapture("axis", "glu", "dim", default=-1, required=False),),
    ),
    "groupnorm": _single_node_graph(
        "groupnorm",
        "groupnorm",
        ("x", "weight", "bias"),
        attr_captures=(
            M.AttrCapture("num_groups", "groupnorm", "num_groups", required=False),
            M.AttrCapture("epsilon", "groupnorm", "eps", default=1e-5, required=False),
        ),
    ),
    "batchnorm": _single_node_graph(
        "batchnorm",
        "batchnorm",
        ("x", "weight", "bias", "running_mean", "running_var"),
        attr_captures=(
            M.AttrCapture("axis", "batchnorm", "axis", default=1, required=False),
            M.AttrCapture("epsilon", "batchnorm", "eps", default=1e-5, required=False),
        ),
    ),
    "embedding_from_tensor": _single_node_graph(
        "embedding_from_tensor",
        "embedding",
        ("embedding_tensor", "indices"),
    ),
}


LINEAR_GRAPH = _graph(
    "linear",
    "linear_mm",
    ("linear_mm",),
    inputs=(_input("x", "linear_mm", 0), _input("weight", "linear_mm", 1)),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=False, required=False),),
    constraints={
        "linear_weight_layout": {"input_role": "x", "weight_role": "weight", "output_node": "linear_mm"},
    },
)

LINEAR_BIAS_GRAPH = _graph(
    "linear_bias",
    "linear_bias_add",
    ("linear_mm", "linear_bias_add"),
    edge_names=("linear_mm_to_bias_add",),
    inputs=(_input("x", "linear_mm", 0), _input("weight", "linear_mm", 1), _input("bias", "linear_bias_add", 1)),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=False, required=False),),
    constraints={
        "linear_weight_layout": {"input_role": "x", "weight_role": "weight", "output_node": "linear_mm"},
        "input_value_kind": {"role": "bias", "allowed_value_kinds": (M.ValueKind.PARAMETER, M.ValueKind.BUFFER)},
    },
)

RMS_NORM_GRAPH = _graph(
    "rms_norm",
    "rms_weight_mul",
    ("rms_square", "rms_mean", "rms_eps_add", "rms_inv", "rms_scale", "rms_weight_mul"),
    edge_names=E.EDGE_GROUPS["rms_norm"],
    inputs=(_input("x", "rms_square", 0), _input("weight", "rms_weight_mul", 1)),
    shared_inputs=(_shared_input("rms_square", 0, "rms_scale", 0),),
    attr_captures=(M.AttrCapture("epsilon", "rms_eps_add", "other", default=1e-6, required=False),),
    constraints={
        "node_attr_equals": {"node": "rms_mean", "attr": "dim", "value": -1},
    },
)

LAYERNORM_NO_BIAS_GRAPH = _graph(
    "layernorm_no_bias",
    "ln_weight_mul",
    ("ln_mean", "ln_center", "ln_square", "ln_var", "ln_eps_add", "ln_inv", "ln_norm", "ln_weight_mul"),
    edge_names=(
        "ln_mean_to_center",
        "ln_center_to_square",
        "ln_square_to_var",
        "ln_var_to_eps_add",
        "ln_eps_add_to_inv",
        "ln_center_to_norm",
        "ln_inv_to_norm",
        "ln_norm_to_weight_mul",
    ),
    inputs=(_input("x", "ln_mean", 0), _input("weight", "ln_weight_mul", 1)),
    shared_inputs=(_shared_input("ln_mean", 0, "ln_center", 0),),
    attr_captures=(M.AttrCapture("epsilon", "ln_eps_add", "other", default=1e-5, required=False),),
    constraints={
        "node_attrs_equal": {"left_node": "ln_mean", "left_attr": "dim", "right_node": "ln_var", "right_attr": "dim"},
    },
)

LAYERNORM_GRAPH = _graph(
    "layernorm",
    "ln_bias_add",
    ("ln_mean", "ln_center", "ln_square", "ln_var", "ln_eps_add", "ln_inv", "ln_norm", "ln_weight_mul", "ln_bias_add"),
    edge_names=E.EDGE_GROUPS["layernorm"],
    inputs=(_input("x", "ln_mean", 0), _input("weight", "ln_weight_mul", 1), _input("bias", "ln_bias_add", 1)),
    shared_inputs=(_shared_input("ln_mean", 0, "ln_center", 0),),
    attr_captures=(M.AttrCapture("epsilon", "ln_eps_add", "other", default=1e-5, required=False),),
    constraints={
        "node_attrs_equal": {"left_node": "ln_mean", "left_attr": "dim", "right_node": "ln_var", "right_attr": "dim"},
        "input_value_kind": {"role": "bias", "allowed_value_kinds": (M.ValueKind.PARAMETER, M.ValueKind.BUFFER)},
    },
)

SWIGLU_MLP_GRAPH = _graph(
    "swiglu_mlp",
    "down_proj",
    ("gate_proj", "up_proj", "mlp_activation", "mlp_product", "down_proj"),
    edge_names=E.EDGE_GROUPS["swiglu_mlp"],
    inputs=(_input("hidden", "gate_proj", 0), _input("gate_weight", "gate_proj", 1), _input("up_weight", "up_proj", 1), _input("down_weight", "down_proj", 1)),
    shared_inputs=(_shared_input("gate_proj", 0, "up_proj", 0),),
    attr_captures=(M.AttrCapture("product_scale", default=1.0, required=False),),
)

GELU_MLP_GRAPH = _graph(
    "gelu_mlp",
    "down_proj",
    ("up_proj", "gelu", "down_proj"),
    inputs=(_input("hidden", "up_proj", 0), _input("up_weight", "up_proj", 1), _input("down_weight", "down_proj", 1)),
    edge_names=E.EDGE_GROUPS["gelu_mlp"],
)

ATTENTION_DIRECT_GRAPH = _graph(
    "scaled_dot_product_attention",
    "attn_sdp",
    ("attn_sdp",),
    inputs=(_input("query", "attn_sdp", 0), _input("key", "attn_sdp", 1), _input("value", "attn_sdp", 2), _input("mask", "attn_sdp", 3, optional=True)),
    attr_captures=(M.AttrCapture("scale", "attn_sdp", "scale", required=False), M.AttrCapture("is_causal", "attn_sdp", "is_causal", default=True, required=False)),
)

ATTENTION_CORE_GRAPH = _graph(
    "attention_core",
    "attn_value",
    ("attn_qk", "attn_scale", "attn_softmax", "attn_value"),
    edge_names=("attn_qk_to_scale", "attn_scale_to_softmax", "attn_softmax_to_value"),
    inputs=(_input("query", "attn_qk", 0), _input("key", "attn_qk", 1), _input("value", "attn_value", 1)),
    attr_captures=(M.AttrCapture("scale", "attn_scale", "other", required=False), M.AttrCapture("is_causal", default=True, required=False), M.AttrCapture("window_size", default=0, required=False)),
    constraints={
        "node_attr_equals": {"node": "attn_softmax", "attr": "dim", "value": -1},
        "parent_tensor_dim_equal": {"left_node": "attn_qk", "left_parent_index": 0, "left_dim": -1, "right_node": "attn_qk", "right_parent_index": 1, "right_dim": -1, "allow_missing": True},
    },
)

ATTENTION_MASKED_GRAPH = _graph(
    "attention_masked",
    "attn_value",
    ("attn_qk", "attn_scale", "attn_mask_add", "attn_softmax", "attn_value"),
    edge_names=("attn_qk_to_scale", "attn_scale_to_mask_add", "attn_mask_add_to_softmax", "attn_softmax_to_value"),
    inputs=(_input("query", "attn_qk", 0), _input("key", "attn_qk", 1), _input("mask", "attn_mask_add", 1), _input("value", "attn_value", 1)),
    attr_captures=(M.AttrCapture("scale", "attn_scale", "other", required=False), M.AttrCapture("use_mask", default=True), M.AttrCapture("additive_mask", default=True, required=False)),
    constraints={
        "input_broadcastable_to_node": {"role": "mask", "node": "attn_qk", "allow_missing": True},
    },
)

ROPE_GRAPH = _graph(
    "rope",
    "rope_add",
    ("rope_slice_even", "rope_slice_odd", "rope_neg", "rope_rotate_cat", "rope_cos_mul", "rope_sin_mul", "rope_add"),
    edge_names=E.EDGE_GROUPS["rope"],
    inputs=(_input("x", "rope_slice_even", 0), _input("cos", "rope_cos_mul", 1), _input("sin", "rope_sin_mul", 1)),
    shared_inputs=(
        _shared_input("rope_slice_even", 0, "rope_slice_odd", 0),
        _shared_input("rope_slice_even", 0, "rope_cos_mul", 0),
    ),
    attr_captures=(M.AttrCapture("theta", default=None, required=False), M.AttrCapture("position_offset", default=0, required=False)),
    constraints={
        "rope_tables_compatible": {"x_role": "x", "cos_role": "cos", "sin_role": "sin"},
    },
)

CONV_GRAPH = _graph(
    "conv",
    "conv",
    ("conv",),
    inputs=(_input("x", "conv", 0), _input("weight", "conv", 1), _input("bias", "conv", 2, optional=True)),
    attr_captures=(M.AttrCapture("stride", "conv", "stride", default=1, required=False), M.AttrCapture("padding", "conv", "padding", default=0, required=False), M.AttrCapture("dilation", "conv", "dilation", default=1, required=False), M.AttrCapture("groups", "conv", "groups", default=1, required=False)),
)

CONV_BIAS_GRAPH = _graph(
    "conv_bias",
    "conv_bias_add",
    ("conv", "conv_bias_add"),
    edge_names=("conv_to_bias_add",),
    inputs=(_input("x", "conv", 0), _input("weight", "conv", 1), _input("bias", "conv_bias_add", 1)),
    attr_captures=(M.AttrCapture("stride", "conv", "stride", default=1, required=False), M.AttrCapture("padding", "conv", "padding", default=0, required=False), M.AttrCapture("dilation", "conv", "dilation", default=1, required=False), M.AttrCapture("groups", "conv", "groups", default=1, required=False)),
)

KV_CACHE_APPEND_GRAPH = _graph(
    "kv_cache_append",
    "kv_cache_append",
    ("kv_cache_append",),
    inputs=(_input("new_kv", "kv_cache_append", 0), _input("cache_state", "kv_cache_append", 1)),
    cache_inputs=(_cache_input("cache_state", "kv_cache_append", 1, tensor_role=M.CacheTensorRole.STATE),),
    cache_outputs=(_cache_output("updated_cache_state", "kv_cache_append", tensor_role=M.CacheTensorRole.STATE),),
    cache_mutations=(_cache_mutation("append_kv", read_roles=("cache_state",), write_roles=("updated_cache_state",)),),
)

ATTENTION_CACHED_GRAPH = _graph(
    "attention_cached",
    "attn_value",
    ("attn_qk", "attn_scale", "attn_softmax", "attn_value"),
    edge_names=("attn_qk_to_scale", "attn_scale_to_softmax", "attn_softmax_to_value"),
    inputs=(_input("query", "attn_qk", 0), _input("key_new", "attn_qk", 1), _input("value_new", "attn_value", 1)),
    cache_inputs=(
        _cache_input("key_cache_state", "attn_qk", 1, tensor_role=M.CacheTensorRole.KEY, optional=True),
        _cache_input("value_cache_state", "attn_value", 1, tensor_role=M.CacheTensorRole.VALUE, optional=True),
    ),
    attr_captures=(M.AttrCapture("scale", "attn_scale", "other", required=False), M.AttrCapture("position_offset", default=0, required=False), M.AttrCapture("window_size", default=0, required=False)),
    constraints={
        "same_layer": {"nodes": ("attn_qk", "attn_value"), "allow_missing": True},
    },
)

EXPERT_BRANCH_GRAPH = _graph(
    "moe_expert_branch",
    "expert_weighted",
    ("expert_gate", "expert_up", "expert_activation", "expert_product", "expert_down", "expert_weighted"),
    edge_names=("expert_gate_to_activation", "expert_activation_to_product", "expert_up_to_product", "expert_product_to_down", "expert_down_to_weighted"),
    inputs=(_input("hidden", "expert_gate", 0), _input("gate_weight", "expert_gate", 1), _input("up_weight", "expert_up", 1), _input("down_weight", "expert_down", 1), _input("routing_weight", "expert_weighted", 1)),
    shared_inputs=(_shared_input("expert_gate", 0, "expert_up", 0),),
    constraints=_note("Each repeated expert branch must correspond to the same expert index selected by routing."),
    allow_root_external_children=False,
)

MOE_GATED_GRAPH = _graph(
    "moe_layer_gated",
    "moe_combine",
    ("router_logits", "routing_probs", "topk", "moe_combine"),
    edge_names=("router_logits_to_probs", "routing_probs_to_topk"),
    inputs=(_input("hidden", "router_logits", 0), _input("router_weight", "router_logits", 1)),
    repeated_subgraphs=(M.RepeatedSubgraph("experts", EXPERT_BRANCH_GRAPH, min_count=1, anchor_node="topk"),),
    attr_captures=(M.AttrCapture("num_experts", default=None, required=False), M.AttrCapture("num_experts_per_tok", "topk", "k", required=False), M.AttrCapture("normalize_routing", default=True, required=False), M.AttrCapture("epsilon", default=1e-6, required=False), M.AttrCapture("routed_scaling_factor", default=1.0, required=False)),
    constraints=_note("Top-k indices must select the repeated expert branches.", "Routing probabilities must be the weights used for expert output combination."),
)

LSTM_CELL_GRAPH = _graph(
    "lstm_cell",
    "lstm_tanh",
    ("lstm_gate_mm", "lstm_recurrent_mm", "lstm_gate_add", "lstm_sigmoid", "lstm_tanh"),
    edge_names=E.EDGE_GROUPS["lstm_cell"],
    inputs=(_input("input", "lstm_gate_mm", 0), _input("h_prev", "lstm_recurrent_mm", 0), _input("weight_ih", "lstm_gate_mm", 1), _input("weight_hh", "lstm_recurrent_mm", 1)),
    constraints={
        "lstm_gate_structure": {"gate_node": "lstm_gate_add", "gate_count": 4, "require_explicit_gate_split": True},
    },
)

DELTANET_DECODE_GRAPH = _graph(
    "gated_deltanet_decode",
    "delta_gate",
    ("delta_q", "delta_k", "delta_v", "delta_gate"),
    inputs=(_input("query", "delta_q", 0), _input("key", "delta_k", 0), _input("value", "delta_v", 0), _input("gate_log", "delta_gate", 0), _input("beta", "delta_gate", 1), _input("initial_state", "delta_gate", 2)),
    attr_captures=(M.AttrCapture("scale", default=None, required=False),),
    cache_inputs=(_cache_input("initial_state", "delta_gate", 2, cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
)

DELTANET_PREFILL_GRAPH = _graph(
    "gated_deltanet_prefill",
    "delta_prefill",
    ("delta_q", "delta_k", "delta_v", "delta_gate", "delta_prefill"),
    inputs=(_input("query", "delta_q", 0), _input("key", "delta_k", 0), _input("value", "delta_v", 0), _input("gate_log", "delta_gate", 0), _input("beta", "delta_gate", 1), _input("initial_state", "delta_prefill", 2)),
    attr_captures=(M.AttrCapture("chunk_size", "delta_prefill", "chunk_size", required=False), M.AttrCapture("scale", default=None, required=False),),
    cache_inputs=(_cache_input("initial_state", "delta_prefill", 2, cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
)

REL_POS_BIAS_GRAPH = _graph(
    "rel_pos_bias",
    "rel_pos_bias",
    ("rel_pos_bias",),
    inputs=(_input("query", "rel_pos_bias", 0), _input("relative_key", "rel_pos_bias", 1)),
    attr_captures=(M.AttrCapture("scale", "rel_pos_bias", "scale", required=False),),
)

KV_CACHE_STATE_GRAPH = _graph(
    "kv_cache_state",
    "kv_cache_state",
    ("kv_cache_state",),
    outputs=(M.FusionOutput("cache_state", "kv_cache_state"),),
    attr_captures=(M.AttrCapture("max_seq_len", default=None, required=False), M.AttrCapture("num_kv_heads", default=None, required=False), M.AttrCapture("head_dim", default=None, required=False), M.AttrCapture("window_size", default=0, required=False), M.AttrCapture("sink_size", default=0, required=False), M.AttrCapture("num_slots", default=1, required=False)),
)

CONV_CACHE_STATE_GRAPH = _graph(
    "conv_cache_state",
    "conv_cache_state",
    ("conv_cache_state",),
    outputs=(M.FusionOutput("cache_state", "conv_cache_state"),),
    attr_captures=(M.AttrCapture("window_size", default=None, required=False), M.AttrCapture("hidden_dim", default=None, required=False)),
)

CONV_CACHE_APPEND_GRAPH = _graph(
    "conv_cache_append",
    "conv_cache_append",
    ("conv_cache_append",),
    inputs=(_input("new_data", "conv_cache_append", 0), _input("cache_state", "conv_cache_append", 1)),
    cache_inputs=(_cache_input("cache_state", "conv_cache_append", 1, cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_outputs=(_cache_output("updated_cache_state", "conv_cache_append", cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_mutations=(_cache_mutation("append_conv_state", cache_kind=M.CacheKind.CONV, read_roles=("cache_state",), write_roles=("updated_cache_state",)),),
)

CONV_CACHE_INITIALIZE_GRAPH = _graph(
    "conv_cache_initialize",
    "conv_cache_initialize",
    ("conv_cache_initialize",),
    inputs=(_input("rows", "conv_cache_initialize", 0), _input("cache_state", "conv_cache_initialize", 1)),
    cache_inputs=(_cache_input("cache_state", "conv_cache_initialize", 1, cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_outputs=(_cache_output("initialized_cache_state", "conv_cache_initialize", cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
)

RECURRENT_CACHE_STATE_GRAPH = _graph(
    "recurrent_cache_state",
    "recurrent_cache_state",
    ("recurrent_cache_state",),
    outputs=(M.FusionOutput("cache_state", "recurrent_cache_state"),),
    attr_captures=(M.AttrCapture("shape", default=None, required=False), M.AttrCapture("precision", default=None, required=False)),
)

RECURRENT_CACHE_WRITE_GRAPH = _graph(
    "recurrent_cache_write",
    "recurrent_cache_write",
    ("recurrent_cache_write",),
    inputs=(_input("new_value", "recurrent_cache_write", 0), _input("cache_input", "recurrent_cache_write", 1)),
    cache_inputs=(_cache_input("cache_input", "recurrent_cache_write", 1, cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
    cache_outputs=(_cache_output("updated_cache", "recurrent_cache_write", cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
    cache_mutations=(_cache_mutation("write_recurrent_state", cache_kind=M.CacheKind.RECURRENT, read_roles=("cache_input",), write_roles=("updated_cache",)),),
)

SAMPLE_GRAPH = _graph(
    "sample",
    "sample_softmax",
    ("sample_topk", "sample_softmax"),
    edge_names=("sample_topk_to_softmax",),
    inputs=(_input("logits", "sample_topk", 0),),
    attr_captures=(M.AttrCapture("temperature", default=0.6, required=False), M.AttrCapture("top_p", default=0.95, required=False), M.AttrCapture("top_k", "sample_topk", "k", default=20, required=False)),
)

SCATTER_TOPK_GRAPH = _graph(
    "scatter_topk",
    "scatter",
    ("scatter",),
    inputs=(_input("indices", "scatter", 0), _input("values", "scatter", 1)),
    attr_captures=(M.AttrCapture("num_classes", "scatter", "num_classes", required=False),),
)

GAUSSIAN_TOPK_GRAPH = _graph(
    "gaussian_topk",
    "gaussian_topk",
    ("gaussian_topk",),
    inputs=(_input("x", "gaussian_topk", 0),),
    attr_captures=(M.AttrCapture("ppf", "gaussian_topk", "ppf", required=False),),
)

ALTUP_PREDICT_GRAPH = _graph(
    "altup_predict",
    "altup_predict",
    ("altup_predict",),
    inputs=(_input("coefs", "altup_predict", 0), _variadic_input("streams", "altup_predict", 1)),
)

ALTUP_CORRECT_GRAPH = _graph(
    "altup_correct",
    "altup_correct",
    ("altup_correct",),
    inputs=(_input("coefs", "altup_correct", 0), _input("innovation", "altup_correct", 1), _variadic_input("predictions", "altup_correct", 2)),
)

DSP_GRAPHS: dict[str, M.FusionGraph] = {
    "stft": _graph("stft", "stft", ("stft",), inputs=(_input("x", "stft", 0), _input("weight", "stft", 1)), attr_captures=(M.AttrCapture("stride", "stft", "stride", required=False), M.AttrCapture("num_fft_bins", "stft", "num_fft_bins", required=False))),
    "rfft": _graph("rfft", "rfft", ("rfft",), inputs=(_input("x", "rfft", 0),)),
    "irfft": _graph("irfft", "irfft", ("irfft",), inputs=(_input("x", "irfft", 0),), attr_captures=(M.AttrCapture("output_length", "irfft", "output_length", required=False),)),
    "spectrogram": _graph("spectrogram", "spectrogram", ("spectrogram",), inputs=(_input("waveform", "spectrogram", 0), _input("mel_filters", "spectrogram", 1)), metadata={"requires_preprocessing_attrs": True}),
    "image_preprocess": _graph("image_preprocess", "image_preprocess", ("image_preprocess",), inputs=(_input("pixel_input", "image_preprocess", 0),), metadata={"requires_preprocessing_attrs": True}),
    "bilinear_interpolation": _graph("bilinear_interpolation", "bilinear_interpolation", ("bilinear_interpolation",), inputs=(_input("pos_embeds", "bilinear_interpolation", 0),), attr_captures=(M.AttrCapture("dst_height", "bilinear_interpolation", "dst_height", required=False), M.AttrCapture("dst_width", "bilinear_interpolation", "dst_width", required=False))),
}


GRAPH_BY_NAME: dict[str, M.FusionGraph] = {
    **DIRECT_GRAPHS,
    **DSP_GRAPHS,
    "linear": LINEAR_GRAPH,
    "linear_bias": LINEAR_BIAS_GRAPH,
    "rms_norm": RMS_NORM_GRAPH,
    "layernorm_no_bias": LAYERNORM_NO_BIAS_GRAPH,
    "layernorm": LAYERNORM_GRAPH,
    "swiglu_mlp": SWIGLU_MLP_GRAPH,
    "gelu_mlp": GELU_MLP_GRAPH,
    "scaled_dot_product_attention": ATTENTION_DIRECT_GRAPH,
    "attention_core": ATTENTION_CORE_GRAPH,
    "attention_masked": ATTENTION_MASKED_GRAPH,
    "rope": ROPE_GRAPH,
    "conv": CONV_GRAPH,
    "conv_bias": CONV_BIAS_GRAPH,
    "kv_cache_append": KV_CACHE_APPEND_GRAPH,
    "attention_cached": ATTENTION_CACHED_GRAPH,
    "moe_layer_gated": MOE_GATED_GRAPH,
    "lstm_cell": LSTM_CELL_GRAPH,
    "gated_deltanet_decode": DELTANET_DECODE_GRAPH,
    "gated_deltanet_prefill": DELTANET_PREFILL_GRAPH,
    "rel_pos_bias": REL_POS_BIAS_GRAPH,
    "kv_cache_state": KV_CACHE_STATE_GRAPH,
    "conv_cache_state": CONV_CACHE_STATE_GRAPH,
    "conv_cache_append": CONV_CACHE_APPEND_GRAPH,
    "conv_cache_initialize": CONV_CACHE_INITIALIZE_GRAPH,
    "recurrent_cache_state": RECURRENT_CACHE_STATE_GRAPH,
    "recurrent_cache_write": RECURRENT_CACHE_WRITE_GRAPH,
    "sample": SAMPLE_GRAPH,
    "scatter_topk": SCATTER_TOPK_GRAPH,
    "gaussian_topk": GAUSSIAN_TOPK_GRAPH,
    "altup_predict": ALTUP_PREDICT_GRAPH,
    "altup_correct": ALTUP_CORRECT_GRAPH,
}


DIRECT_CACTUS_OPS: dict[str, str] = {
    "sqrt": "scalar_sqrt",
}

DIRECT_DEFINITIONS: dict[str, M.FusionDefinition] = {
    name: _definition(name, cactus_op=DIRECT_CACTUS_OPS.get(name, name), graph=graph, fusion_fields=("generic", "direct"))
    for name, graph in DIRECT_GRAPHS.items()
}


FUSIONS: dict[str, M.FusionDefinition] = {
    **DIRECT_DEFINITIONS,
    "linear": _definition("linear", "matmul", LINEAR_GRAPH, fusion_fields=("generic", "linear")),
    "linear_bias": _definition("linear_bias", "linear", LINEAR_BIAS_GRAPH, fusion_fields=("generic", "linear")),
    "rms_norm": _definition("rms_norm", "rms_norm", RMS_NORM_GRAPH, fusion_fields=("generic", "rmsnorm", "gemma4_rmsnorm", "qwen2_5_rmsnorm")),
    "layernorm_no_bias": _definition("layernorm_no_bias", "layernorm", LAYERNORM_NO_BIAS_GRAPH, fusion_fields=("generic", "normalization")),
    "layernorm": _definition("layernorm", "layernorm", LAYERNORM_GRAPH, fusion_fields=("generic", "normalization")),
    "swiglu_mlp": _definition("swiglu_mlp", "dense_mlp_tq_fused", SWIGLU_MLP_GRAPH, fusion_fields=("generic", "mlp", "gemma4_mlp", "qwen2_5_mlp", "lfm_mlp")),
    "gelu_mlp": _definition("gelu_mlp", "matmul", GELU_MLP_GRAPH, fusion_fields=("generic", "mlp")),
    "scaled_dot_product_attention": _definition("scaled_dot_product_attention", "attention", ATTENTION_DIRECT_GRAPH, fusion_fields=("generic", "attention")),
    "attention_core": _definition("attention_core", "attention", ATTENTION_CORE_GRAPH, fusion_fields=("generic", "attention", "gemma4_attention", "qwen2_5_attention", "lfm_attention")),
    "attention_masked": _definition("attention_masked", "attention", ATTENTION_MASKED_GRAPH, fusion_fields=("generic", "attention", "gemma4_attention")),
    "rope": _definition("rope", "rope", ROPE_GRAPH, fusion_fields=("generic", "rope", "gemma4_rope", "qwen2_5_rope")),
    "conv": _definition("conv", "conv1d", CONV_GRAPH, fusion_fields=("generic", "conv", "audio", "vision")),
    "conv_bias": _definition("conv_bias", "conv1d", CONV_BIAS_GRAPH, fusion_fields=("generic", "conv", "audio", "vision")),
    "conv1d_k3": _definition("conv1d_k3", "conv1d_k3", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=3)),
    "conv1d_k7s3": _definition("conv1d_k7s3", "conv1d_k7s3", CONV_BIAS_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=7, stride=3)),
    "conv1d_causal": _definition("conv1d_causal", "conv1d_causal", CONV_GRAPH, fusion_fields=("generic", "conv", "cache"), metadata=_required_attrs(causal=True)),
    "conv1d_pointwise": _definition("conv1d_pointwise", "conv1d_pointwise", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=1)),
    "conv1d_same_depthwise_k9": _definition("conv1d_same_depthwise_k9", "conv1d_same_depthwise_k9", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=9, depthwise=True)),
    "conv2d_k3s2p1": _definition("conv2d_k3s2p1", "conv2d_k3s2p1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=3, stride=2, padding=1)),
    "conv2d_depthwise_k3s2p1": _definition("conv2d_depthwise_k3s2p1", "conv2d_depthwise_k3s2p1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=3, stride=2, padding=1, depthwise=True)),
    "conv2d_pointwise_1x1": _definition("conv2d_pointwise_1x1", "conv2d_pointwise_1x1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=1)),
    "kv_cache_append": _definition("kv_cache_append", "kv_cache_append", KV_CACHE_APPEND_GRAPH, fusion_fields=("generic", "cache"), supported_inference_modes=("prefill_with_cache", "decode_with_cache")),
    "attention_cached": _definition("attention_cached", "attention_cached", ATTENTION_CACHED_GRAPH, fusion_fields=("generic", "attention", "cache"), supported_inference_modes=("decode_with_cache",)),
    "moe_layer_gated": _definition("moe_layer_gated", "moe_layer_gated", MOE_GATED_GRAPH, fusion_fields=("generic", "moe")),
    "lstm_cell": _definition("lstm_cell", "lstm_cell", LSTM_CELL_GRAPH, fusion_fields=("generic", "recurrent", "audio")),
    "gated_deltanet_decode": _definition("gated_deltanet_decode", "gated_deltanet_decode", DELTANET_DECODE_GRAPH, fusion_fields=("generic", "recurrent", "cache")),
    "gated_deltanet_prefill": _definition("gated_deltanet_prefill", "gated_deltanet_prefill", DELTANET_PREFILL_GRAPH, fusion_fields=("generic", "recurrent", "cache")),
    "rel_pos_bias": _definition("rel_pos_bias", "rel_pos_bias", REL_POS_BIAS_GRAPH, fusion_fields=("generic", "attention")),
    "kv_cache_state": _definition("kv_cache_state", "kv_cache_state", KV_CACHE_STATE_GRAPH, fusion_fields=("generic", "cache"), supported_inference_modes=("prefill_with_cache", "decode_with_cache")),
    "conv_cache_state": _definition("conv_cache_state", "conv_cache_state", CONV_CACHE_STATE_GRAPH, fusion_fields=("generic", "cache", "conv")),
    "conv_cache_append": _definition("conv_cache_append", "conv_cache_append", CONV_CACHE_APPEND_GRAPH, fusion_fields=("generic", "cache", "conv")),
    "conv_cache_initialize": _definition("conv_cache_initialize", "conv_cache_initialize", CONV_CACHE_INITIALIZE_GRAPH, fusion_fields=("generic", "cache", "conv")),
    "recurrent_cache_state": _definition("recurrent_cache_state", "recurrent_cache_state", RECURRENT_CACHE_STATE_GRAPH, fusion_fields=("generic", "cache", "recurrent")),
    "recurrent_cache_write": _definition("recurrent_cache_write", "recurrent_cache_write", RECURRENT_CACHE_WRITE_GRAPH, fusion_fields=("generic", "cache", "recurrent")),
    "sample": _definition("sample", "sample", SAMPLE_GRAPH, fusion_fields=("generic", "sample")),
    "scatter_topk": _definition("scatter_topk", "scatter_topk", SCATTER_TOPK_GRAPH, fusion_fields=("generic", "sample")),
    "gaussian_topk": _definition("gaussian_topk", "gaussian_topk", GAUSSIAN_TOPK_GRAPH, fusion_fields=("generic", "sample")),
    "altup_predict": _definition("altup_predict", "altup_predict", ALTUP_PREDICT_GRAPH, fusion_fields=("generic", "altup")),
    "altup_correct": _definition("altup_correct", "altup_correct", ALTUP_CORRECT_GRAPH, fusion_fields=("generic", "altup")),
    "stft": _definition("stft", "stft", DSP_GRAPHS["stft"], fusion_fields=("generic", "audio", "dsp")),
    "rfft": _definition("rfft", "rfft", DSP_GRAPHS["rfft"], fusion_fields=("generic", "audio", "dsp")),
    "irfft": _definition("irfft", "irfft", DSP_GRAPHS["irfft"], fusion_fields=("generic", "audio", "dsp")),
    "spectrogram": _definition("spectrogram", "spectrogram", DSP_GRAPHS["spectrogram"], fusion_fields=("generic", "audio", "dsp")),
    "image_preprocess": _definition("image_preprocess", "image_preprocess", DSP_GRAPHS["image_preprocess"], fusion_fields=("generic", "vision", "preprocess")),
    "bilinear_interpolation": _definition("bilinear_interpolation", "bilinear_interpolation", DSP_GRAPHS["bilinear_interpolation"], fusion_fields=("generic", "vision")),
}


FUSION_CATALOG = M.FusionCatalog(fusions=tuple(FUSIONS.values()))

FUSIONS_BY_TARGET: dict[str, M.FusionDefinition] = _index_by_target(FUSIONS)
FUSIONS_BY_CACTUS_OP: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_cactus_op(FUSIONS)
FUSIONS_BY_FIELD: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_field(FUSIONS)
FUSIONS_BY_ROOT_OP: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_root_op(FUSIONS)
