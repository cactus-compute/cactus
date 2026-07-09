from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FusionNode:
    name: str
    op: str | tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    optional: bool = False
    transparent: bool = False
    repeated: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FusionEdge:
    source: str
    dest: str
    dest_input_index: int
    source_output_index: int | None = None
    optional: bool = False
    repeated: bool = False


@dataclass(frozen=True, slots=True)
class FusionInput:
    role: str
    node: str
    parent_index: int | None = None
    variadic: bool = False
    optional: bool = False
    repeated_group: str | None = None


@dataclass(frozen=True, slots=True)
class AttrCapture:
    name: str
    source_node: str | None = None
    source_attr: str | None = None
    default: Any = None
    required: bool = True
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FusionGraph:
    name: str
    target: str
    cactus_inputs: tuple[str, ...]
    cactus_attrs: tuple[str, ...]
    nodes: dict[str, FusionNode] = field(default_factory=dict)
    edges: tuple[FusionEdge, ...] = ()
    root: str | None = None
    outputs: tuple[str, ...] = ()
    inputs: tuple[FusionInput, ...] = ()
    output_attrs: tuple[AttrCapture, ...] = ()
    constraints: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    repeated_groups: tuple[str, ...] = ()
    notes: str = ""


def n(
    name: str,
    op: str | tuple[str, ...],
    *,
    attrs: dict[str, Any] | None = None,
    optional: bool = False,
    transparent: bool = False,
    repeated: bool = False,
    notes: str = "",
) -> FusionNode:
    return FusionNode(
        name=name,
        op=op,
        attrs=attrs or {},
        optional=optional,
        transparent=transparent,
        repeated=repeated,
        notes=notes,
    )


def e(
    source: str,
    dest: str,
    dest_input_index: int,
    *,
    source_output_index: int | None = None,
    optional: bool = False,
    repeated: bool = False,
) -> FusionEdge:
    return FusionEdge(
        source=source,
        dest=dest,
        dest_input_index=dest_input_index,
        source_output_index=source_output_index,
        optional=optional,
        repeated=repeated,
    )


def inp(
    role: str,
    node: str,
    parent_index: int | None = None,
    *,
    variadic: bool = False,
    optional: bool = False,
    repeated_group: str | None = None,
) -> FusionInput:
    return FusionInput(
        role=role,
        node=node,
        parent_index=parent_index,
        variadic=variadic,
        optional=optional,
        repeated_group=repeated_group,
    )


def cap(
    name: str,
    source_node: str | None = None,
    source_attr: str | None = None,
    *,
    default: Any = None,
    required: bool = True,
    notes: str = "",
) -> AttrCapture:
    return AttrCapture(
        name=name,
        source_node=source_node,
        source_attr=source_attr,
        default=default,
        required=required,
        notes=notes,
    )


def direct(
    target: str,
    cactus_inputs: tuple[str, ...],
    cactus_attrs: tuple[str, ...],
    raw_ops: str | tuple[str, ...],
    *,
    attr_map: dict[str, str] | None = None,
    optional_attr_defaults: dict[str, Any] | None = None,
    notes: str = "",
) -> FusionGraph:
    root = f"{target}_root"
    output_attrs = []

    for cactus_attr in cactus_attrs:
        source_attr = (attr_map or {}).get(cactus_attr, cactus_attr)
        output_attrs.append(
            cap(
                cactus_attr,
                root,
                source_attr,
                default=(optional_attr_defaults or {}).get(cactus_attr),
                required=cactus_attr not in (optional_attr_defaults or {}),
            )
        )

    return FusionGraph(
        name=f"{target}_direct",
        target=target,
        cactus_inputs=cactus_inputs,
        cactus_attrs=cactus_attrs,
        nodes={root: n(root, raw_ops)},
        root=root,
        outputs=(root,),
        inputs=tuple(inp(role, root, i) for i, role in enumerate(cactus_inputs)),
        output_attrs=tuple(output_attrs),
        notes=notes,
    )


def source_op(target: str, cactus_attrs: tuple[str, ...], notes: str) -> FusionGraph:
    root = f"{target}_source"
    return FusionGraph(
        name=f"{target}_source",
        target=target,
        cactus_inputs=(),
        cactus_attrs=cactus_attrs,
        nodes={root: n(root, "cactus.source_op", notes=notes)},
        root=root,
        outputs=(root,),
        output_attrs=tuple(cap(attr, default=None, notes="Caller/runtime supplied attr") for attr in cactus_attrs),
        notes=notes,
    )


FUSION_GRAPHS: dict[str, FusionGraph] = {
    "add": direct("add", ("a", "b"), (), "aten.add.Tensor"),
    "add_clipped": direct(
        "add_clipped",
        ("a", "b"),
        (),
        "aten.add.Tensor",
        notes="Needs post-add clipping semantics from a trace before this can be safely fused.",
    ),
    "subtract": direct("subtract", ("a", "b"), (), "aten.sub.Tensor"),
    "multiply": direct("multiply", ("a", "b"), (), "aten.mul.Tensor"),
    "divide": direct("divide", ("a", "b"), (), "aten.div.Tensor"),
    "not_equal": direct("not_equal", ("a", "b"), (), "aten.ne.Tensor"),
    "abs": direct("abs", ("x",), (), "aten.abs.default"),
    "pow": direct("pow", ("x",), ("exponent",), "aten.pow.Tensor_Scalar"),
    "precision_cast": direct(
        "precision_cast",
        ("x",),
        ("dtype",),
        "aten._to_copy.default",
        attr_map={"dtype": "dtype"},
    ),
    "quantize_activations": direct(
        "quantize_activations",
        ("x",),
        (),
        ("aten.quantize_per_tensor.default", "aten._to_copy.default"),
        notes="Needs quantization-specific attrs/scales once a quantized export is available.",
    ),
    "scalar_add": direct("scalar_add", ("x",), ("value",), "aten.add.Scalar", attr_map={"value": "scalar"}),
    "scalar_subtract": direct("scalar_subtract", ("x",), ("value",), "aten.sub.Scalar", attr_map={"value": "scalar"}),
    "scalar_multiply": direct("scalar_multiply", ("x",), ("value",), ("aten.mul.Scalar", "aten.neg.default"), attr_map={"value": "scalar"}),
    "scalar_divide": direct("scalar_divide", ("x",), ("value",), "aten.div.Scalar", attr_map={"value": "scalar"}),
    "scalar_floor_divide": direct("scalar_floor_divide", ("x",), ("value",), "aten.floor_divide.default", attr_map={"value": "scalar"}),
    "scalar_not_equal": direct("scalar_not_equal", ("x",), ("value",), "aten.ne.Scalar", attr_map={"value": "scalar"}),
    "scalar_exp": direct("scalar_exp", ("x",), (), "aten.exp.default"),
    "scalar_sqrt": direct("scalar_sqrt", ("x",), (), "aten.sqrt.default"),
    "scalar_cos": direct("scalar_cos", ("x",), (), "aten.cos.default"),
    "scalar_sin": direct("scalar_sin", ("x",), (), "aten.sin.default"),
    "scalar_log": direct("scalar_log", ("x",), (), "aten.log.default"),
    "clamp": direct("clamp", ("x",), ("lo", "hi"), "aten.clamp.default", attr_map={"lo": "min", "hi": "max"}),
    "masked_select_prefix": direct(
        "masked_select_prefix",
        ("x", "mask"),
        (),
        ("aten.masked_select.default", "aten.index.Tensor"),
        notes="Needs prefix semantics check, not just generic masked_select.",
    ),
    "masked_scatter": direct("masked_scatter", ("x", "mask", "source"), (), "aten.masked_scatter.default"),
    "view": direct("view", ("x",), ("shape",), "aten.view.default"),
    "reshape": direct("reshape", ("x",), ("shape",), "aten.reshape.default"),
    "expand": direct("expand", ("x",), ("shape",), "aten.expand.default"),
    "flatten": direct(
        "flatten",
        ("x",),
        ("start_dim", "end_dim"),
        "aten.flatten.using_ints",
        optional_attr_defaults={"start_dim": 0, "end_dim": -1},
    ),
    "slice": direct("slice", ("x",), ("axis", "start", "end", "step"), "aten.slice.Tensor", attr_map={"axis": "dim"}),
    "index": direct("index", ("x",), ("index_value", "axis"), "aten.select.int", attr_map={"axis": "dim", "index_value": "index"}),
    "transpose": direct(
        "transpose",
        ("x",),
        ("backend",),
        "aten.transpose.int",
        optional_attr_defaults={"backend": 0},
        notes="Needs dim0/dim1 capture if Cactus transpose grows beyond backend-only attrs.",
    ),
    "permute": direct(
        "permute",
        ("x",),
        ("permutation", "backend"),
        "aten.permute.default",
        attr_map={"permutation": "dims"},
        optional_attr_defaults={"backend": 0},
    ),
    "matmul": direct(
        "matmul",
        ("a", "b"),
        ("pretransposed_rhs", "backend", "output_dtype"),
        ("aten.mm.default", "aten.matmul.default", "aten.bmm.default"),
        optional_attr_defaults={"pretransposed_rhs": False, "backend": 0, "output_dtype": None},
    ),
    "gather": direct("gather", ("tensor", "indices"), (), "aten.gather.default"),
    "embedding_from_tensor": direct("embedding_from_tensor", ("embedding_tensor", "indices"), (), "aten.embedding.default"),
    "embedding_from_file": source_op("embedding_from_file", ("filename",), "Runtime/source op. No ATen subgraph unless file loading is represented in IR."),
    "mmap_embeddings": source_op("mmap_embeddings", ("filename",), "Runtime/source op for memory-mapped embedding weights."),
    "mmap_weights": source_op("mmap_weights", ("filename",), "Runtime/source op for memory-mapped weights."),
    "bilinear_interpolation": direct(
        "bilinear_interpolation",
        ("pos_embeds",),
        ("dst_height", "dst_width"),
        ("aten.upsample_bilinear2d.vec", "aten._upsample_bilinear2d_aa.vec"),
    ),
    "concat": direct("concat", ("a", "b"), ("axis",), "aten.cat.default", attr_map={"axis": "dim"}),
    "cat": FusionGraph(
        name="cat_variadic",
        target="cat",
        cactus_inputs=("tensors",),
        cactus_attrs=("axis",),
        nodes={"cat": n("cat", "aten.cat.default")},
        root="cat",
        outputs=("cat",),
        inputs=(inp("tensors", "cat", variadic=True),),
        output_attrs=(cap("axis", "cat", "dim"),),
    ),
    "groupnorm": direct("groupnorm", ("x", "weight", "bias"), ("num_groups", "eps"), "aten.group_norm.default"),
    "layernorm": direct("layernorm", ("x", "weight", "bias"), ("eps",), ("aten.layer_norm.default", "aten.native_layer_norm.default")),
    "batchnorm": direct("batchnorm", ("x", "weight", "bias", "running_mean", "running_var"), ("axis", "eps"), ("aten.batch_norm.default", "aten.native_batch_norm.default"), optional_attr_defaults={"axis": 1}),
    "rms_norm": FusionGraph(
        name="rms_norm_decomposed_or_direct",
        target="rms_norm",
        cactus_inputs=("x", "weight"),
        cactus_attrs=("eps",),
        nodes={
            "squared_x": n("squared_x", "aten.pow.Tensor_Scalar", attrs={"exponent": 2}),
            "variance": n("variance", "aten.mean.dim"),
            "variance_with_eps": n("variance_with_eps", "aten.add.Tensor"),
            "inverse_rms": n("inverse_rms", "aten.pow.Tensor_Scalar", attrs={"exponent": -0.5}),
            "normalized_x": n("normalized_x", "aten.mul.Tensor"),
            "weighted_output": n("weighted_output", "aten.mul.Tensor"),
        },
        edges=(
            e("squared_x", "variance", 0),
            e("variance", "variance_with_eps", 0),
            e("variance_with_eps", "inverse_rms", 0),
            e("inverse_rms", "normalized_x", 1),
            e("normalized_x", "weighted_output", 0),
        ),
        root="weighted_output",
        outputs=("weighted_output",),
        inputs=(inp("x", "normalized_x", 0), inp("weight", "weighted_output", 1)),
        output_attrs=(cap("eps", "variance_with_eps", "scalar"),),
        constraints=("normalized_x input 0 and squared_x input 0 must be the same node", "mean axis must be hidden dimension"),
        variants=("aten.rms_norm.default", "pow_mean_add_pow_mul_mul"),
    ),
    "topk": direct("topk", ("x",), ("k",), "aten.topk.default"),
    "rope": FusionGraph(
        name="rope_decomposed",
        target="rope",
        cactus_inputs=("x",),
        cactus_attrs=("theta", "position_offset", "backend"),
        nodes={
            "x": n("x", "external"),
            "sin": n("sin", ("aten.sin.default", "constant_sin_cache")),
            "cos": n("cos", ("aten.cos.default", "constant_cos_cache")),
            "rotate_half": n("rotate_half", "rope_rotate_half_subgraph"),
            "scaled_x": n("scaled_x", "aten.mul.Tensor"),
            "scaled_rot": n("scaled_rot", "aten.mul.Tensor"),
            "out": n("out", "aten.add.Tensor"),
        },
        edges=(e("x", "scaled_x", 0), e("cos", "scaled_x", 1), e("x", "rotate_half", 0), e("rotate_half", "scaled_rot", 0), e("sin", "scaled_rot", 1), e("scaled_x", "out", 0), e("scaled_rot", "out", 1)),
        root="out",
        outputs=("out",),
        inputs=(inp("x", "scaled_x", 0),),
        output_attrs=(cap("theta", default=None, required=False), cap("position_offset", default=0, required=False), cap("backend", default=0, required=False)),
        constraints=("pairwise rotary dims must match Cactus layout", "sin/cos cache or generated constants must be identified"),
    ),
    "rope_gptj": FusionGraph(
        name="rope_gptj_decomposed",
        target="rope_gptj",
        cactus_inputs=("x",),
        cactus_attrs=("theta", "position_offset", "rot_dim", "backend"),
        nodes={"rope": n("rope", "rope_gptj_subgraph")},
        root="rope",
        outputs=("rope",),
        inputs=(inp("x", "rope", 0),),
        output_attrs=(cap("theta", default=None, required=False), cap("position_offset", default=0), cap("rot_dim", default=0), cap("backend", default=0)),
        constraints=("GPT-J interleaved rotary layout check",),
    ),
    "sum": direct("sum", ("x",), ("axis",), "aten.sum.dim_IntList", attr_map={"axis": "dim"}),
    "mean": direct("mean", ("x",), ("axis", "keepdim"), "aten.mean.dim", attr_map={"axis": "dim"}),
    "variance": direct("variance", ("x",), ("axis", "keepdim"), "aten.var.dim", attr_map={"axis": "dim"}),
    "min": direct("min", ("x",), ("axis",), "aten.min.dim", attr_map={"axis": "dim"}),
    "max": direct("max", ("x",), ("axis",), "aten.max.dim", attr_map={"axis": "dim"}),
    "cumsum": direct("cumsum", ("x",), ("axis",), "aten.cumsum.default", attr_map={"axis": "dim"}),
    "softmax": direct("softmax", ("x",), ("axis",), "aten._softmax.default", attr_map={"axis": "dim"}),
    "attention": FusionGraph(
        name="attention_direct_or_decomposed",
        target="attention",
        cactus_inputs=("query", "key", "value", "mask"),
        cactus_attrs=("scale", "is_causal", "position_offset", "window_size", "backend", "use_mask", "additive_mask"),
        nodes={
            "query": n("query", "external"),
            "key": n("key", "external"),
            "value": n("value", "external"),
            "k_transpose": n("k_transpose", ("aten.transpose.int", "aten.permute.default"), optional=True, transparent=True),
            "scores": n("scores", ("aten.matmul.default", "aten.bmm.default")),
            "scale": n("scale", ("aten.mul.Scalar", "aten.div.Scalar"), optional=True),
            "mask_apply": n("mask_apply", ("aten.add.Tensor", "aten.where.self"), optional=True),
            "softmax": n("softmax", "aten._softmax.default"),
            "context": n("context", ("aten.matmul.default", "aten.bmm.default")),
            "direct_sdpa": n("direct_sdpa", "aten.scaled_dot_product_attention.default", optional=True),
        },
        edges=(e("key", "k_transpose", 0, optional=True), e("query", "scores", 0), e("k_transpose", "scores", 1, optional=True), e("scores", "scale", 0, optional=True), e("scale", "mask_apply", 0, optional=True), e("mask_apply", "softmax", 0, optional=True), e("scores", "softmax", 0, optional=True), e("softmax", "context", 0), e("value", "context", 1)),
        root="context",
        outputs=("context",),
        inputs=(inp("query", "scores", 0), inp("key", "scores", 1), inp("value", "context", 1), inp("mask", "mask_apply", 1, optional=True)),
        output_attrs=(cap("scale", "scale", "scalar", default=None, required=False), cap("is_causal", "direct_sdpa", "is_causal", default=False, required=False), cap("position_offset", default=0), cap("window_size", default=0), cap("backend", default=0), cap("use_mask", default=False), cap("additive_mask", default=False)),
        constraints=("q/k head_dim match", "softmax axis is score last dimension", "mask semantics are boolean/additive/causal", "RoPE, if fused separately, must already be applied to q and k"),
        variants=("aten.scaled_dot_product_attention.default", "matmul_scale_mask_softmax_matmul", "bmm_scale_mask_softmax_bmm"),
    ),
    "kv_cache_state": source_op("kv_cache_state", ("max_seq_len", "num_kv_heads", "head_dim", "window_size", "sink_size", "num_slots"), "State allocation op; not discovered from ordinary ATen math subgraph."),
    "kv_cache_append": direct("kv_cache_append", ("new_kv", "cache_state"), ("window_size", "sink_size"), "kv_cache_append_subgraph", optional_attr_defaults={"window_size": 0, "sink_size": 0}),
    "attention_cached": FusionGraph(
        name="attention_cached",
        target="attention_cached",
        cactus_inputs=("query", "key_new", "value_new", "k_cache_state", "v_cache_state"),
        cactus_attrs=("scale", "position_offset", "window_size", "v_head_dim"),
        nodes={"attention_cached": n("attention_cached", "cached_attention_subgraph")},
        root="attention_cached",
        outputs=("attention_cached",),
        inputs=tuple(inp(role, "attention_cached", i) for i, role in enumerate(("query", "key_new", "value_new", "k_cache_state", "v_cache_state"))),
        output_attrs=(cap("scale", default=None, required=False), cap("position_offset", default=0), cap("window_size", default=0), cap("v_head_dim", default=None, required=False)),
        constraints=("must identify cache state nodes", "must verify append/read ordering"),
    ),
    "conv_cache_state": source_op("conv_cache_state", ("window_size", "hidden_dim"), "State allocation op for streaming conv."),
    "conv_cache_append": direct("conv_cache_append", ("new_data", "cache_state"), (), "conv_cache_append_subgraph"),
    "conv_cache_initialize": direct("conv_cache_initialize", ("rows", "cache_state"), (), "conv_cache_initialize_subgraph"),
    "recurrent_cache_state": source_op("recurrent_cache_state", ("shape", "dtype"), "State allocation op for recurrent kernels."),
    "recurrent_cache_write": direct("recurrent_cache_write", ("new_value", "cache_input"), (), "recurrent_cache_write_subgraph"),
    "rel_pos_bias": direct("rel_pos_bias", ("query", "relative_key"), ("scale",), "relative_position_bias_subgraph"),
    "attention_int8_hybrid": direct(
        "attention_int8_hybrid",
        ("query", "key_new", "value_new"),
        ("scale", "position_offset", "cache_len", "num_kv_heads", "head_dim", "window_size"),
        "attention_int8_hybrid_subgraph",
        notes="Requires quantized/cache-aware attention trace.",
    ),
    "relu": direct("relu", ("x",), (), "aten.relu.default"),
    "silu": FusionGraph(
        name="silu_direct_or_sigmoid_mul",
        target="silu",
        cactus_inputs=("x",),
        cactus_attrs=(),
        nodes={"sigmoid": n("sigmoid", "aten.sigmoid.default", optional=True), "mul": n("mul", "aten.mul.Tensor")},
        edges=(e("sigmoid", "mul", 0, optional=True),),
        root="mul",
        outputs=("mul",),
        inputs=(inp("x", "mul", 1),),
        constraints=("mul other input and sigmoid input must be same x",),
        variants=("aten.silu.default", "sigmoid_mul"),
    ),
    "gelu": direct("gelu", ("x",), (), "aten.gelu.default"),
    "gelu_erf": direct("gelu_erf", ("x",), (), "gelu_erf_subgraph", notes="Detect erf-based GELU decomposition."),
    "sigmoid": direct("sigmoid", ("x",), (), "aten.sigmoid.default"),
    "tanh": direct("tanh", ("x",), (), "aten.tanh.default"),
    "glu": direct("glu", ("x",), ("axis",), "aten.glu.default", attr_map={"axis": "dim"}),
    "conv1d_causal": direct("conv1d_causal", ("x", "weight"), ("kernel_size", "dilation"), "aten.convolution.default", notes="Requires padding/causal layout check."),
    "conv1d_k3": direct("conv1d_k3", ("x", "weight"), ("stride",), "aten.convolution.default", optional_attr_defaults={"stride": 1}, notes="Requires weight kernel_size == 3."),
    "conv1d_k7s3": direct("conv1d_k7s3", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires kernel_size == 7 and stride == 3."),
    "conv1d": direct("conv1d", ("x", "weight", "bias"), ("stride",), ("aten.conv1d.default", "aten.convolution.default"), optional_attr_defaults={"stride": 1}),
    "conv1d_same_depthwise_k9": direct("conv1d_same_depthwise_k9", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires groups == channels, kernel_size == 9, same padding."),
    "conv1d_pointwise": direct("conv1d_pointwise", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires kernel_size == 1."),
    "conv2d_k3s2p1": direct("conv2d_k3s2p1", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires kernel_size == 3, stride == 2, padding == 1."),
    "conv2d_depthwise_k3s2p1": direct("conv2d_depthwise_k3s2p1", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires depthwise groups and k3s2p1."),
    "conv2d_pointwise_1x1": direct("conv2d_pointwise_1x1", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires kernel_size == 1x1."),
    "conv2d_k3s1p1": direct("conv2d_k3s1p1", ("x", "weight", "bias"), (), "aten.convolution.default", notes="Requires kernel_size == 3, stride == 1, padding == 1."),
    "conv2d": direct("conv2d", ("x", "weight", "bias"), ("stride", "padding", "dilation", "groups"), ("aten.conv2d.default", "aten.convolution.default"), optional_attr_defaults={"stride": 1, "padding": 0, "dilation": 1, "groups": 1}),
    "lstm_cell": direct("lstm_cell", ("input", "h_prev", "c_prev", "weight_ih", "weight_hh", "bias_ih", "bias_hh"), (), "lstm_cell_subgraph", notes="Requires gate split/order checks: input, forget, cell, output."),
    "gated_deltanet_decode": direct("gated_deltanet_decode", ("query", "key", "value", "gate_log", "beta", "initial_state"), ("scale",), "gated_deltanet_decode_subgraph"),
    "gated_deltanet_prefill": direct("gated_deltanet_prefill", ("query", "key", "value", "gate_log", "beta", "initial_state"), ("chunk_size", "scale"), "gated_deltanet_prefill_subgraph"),
    "stft": direct("stft", ("x", "weight"), ("stride", "num_fft_bins"), ("aten.stft.default", "stft_subgraph")),
    "altup_predict": direct("altup_predict", ("coefs", "streams"), (), "altup_predict_subgraph"),
    "altup_correct": direct("altup_correct", ("coefs", "innovation", "predictions"), (), "altup_correct_subgraph"),
    "gaussian_topk": direct("gaussian_topk", ("x",), ("ppf",), "gaussian_topk_subgraph"),
    "moe_layer_gated": FusionGraph(
        name="moe_layer_gated_topk_experts",
        target="moe_layer_gated",
        cactus_inputs=("hidden", "routing_probs", "topk_indices", "w1_weights", "w3_weights", "w2_weights"),
        cactus_attrs=("num_experts", "num_experts_per_tok", "normalize_routing", "epsilon", "routed_scaling_factor"),
        nodes={
            "router_logits": n("router_logits", ("aten.mm.default", "aten.matmul.default", "aten.linear.default")),
            "routing_probs": n("routing_probs", ("aten._softmax.default", "aten.sigmoid.default")),
            "topk": n("topk", "aten.topk.default"),
            "topk_values": n("topk_values", ("operator.getitem", "<built-in function getitem>")),
            "topk_indices": n("topk_indices", ("operator.getitem", "<built-in function getitem>")),
            "renorm": n("renorm", ("aten.sum.dim_IntList", "aten.div.Tensor"), optional=True),
            "expert_gate": n("expert_gate", ("aten.mm.default", "aten.matmul.default", "aten.linear.default"), repeated=True),
            "expert_up": n("expert_up", ("aten.mm.default", "aten.matmul.default", "aten.linear.default"), repeated=True),
            "expert_act": n("expert_act", ("aten.silu.default", "aten.gelu.default", "aten.relu.default"), repeated=True),
            "expert_mul": n("expert_mul", "aten.mul.Tensor", repeated=True),
            "expert_down": n("expert_down", ("aten.mm.default", "aten.matmul.default", "aten.linear.default"), repeated=True),
            "weighted_expert": n("weighted_expert", "aten.mul.Tensor", repeated=True),
            "combine": n("combine", ("aten.add.Tensor", "aten.index_put.default", "aten.scatter_add.default")),
        },
        edges=(e("router_logits", "routing_probs", 0), e("routing_probs", "topk", 0), e("topk", "topk_values", 0, source_output_index=0), e("topk", "topk_indices", 0, source_output_index=1), e("expert_gate", "expert_act", 0, repeated=True), e("expert_act", "expert_mul", 0, repeated=True), e("expert_up", "expert_mul", 1, repeated=True), e("expert_mul", "expert_down", 0, repeated=True), e("expert_down", "weighted_expert", 0, repeated=True), e("topk_values", "weighted_expert", 1, repeated=True), e("weighted_expert", "combine", 0, repeated=True)),
        root="combine",
        outputs=("combine",),
        inputs=(inp("hidden", "router_logits", 0), inp("routing_probs", "routing_probs", 0), inp("topk_indices", "topk_indices", 0), inp("w1_weights", "expert_gate", 1, variadic=True, repeated_group="experts"), inp("w3_weights", "expert_up", 1, variadic=True, repeated_group="experts"), inp("w2_weights", "expert_down", 1, variadic=True, repeated_group="experts")),
        output_attrs=(cap("num_experts", default=None, required=False), cap("num_experts_per_tok", "topk", "k"), cap("normalize_routing", default=False), cap("epsilon", default=1e-6), cap("routed_scaling_factor", default=1.0)),
        constraints=("topk getitem 0/1 must map values/indices", "expert branch count and weight order must be stable", "router hidden input must be shared with expert branches", "combine/scatter must preserve token order"),
        repeated_groups=("experts",),
    ),
    "dense_mlp_tq_fused": direct("dense_mlp_tq_fused", ("hidden", "gate_weight", "up_weight", "down_weight"), ("product_scale",), "dense_mlp_tq_fused_subgraph", optional_attr_defaults={"product_scale": 1.0}),
    "moe_layer_ungated": FusionGraph(
        name="moe_layer_ungated_topk_experts",
        target="moe_layer_ungated",
        cactus_inputs=("hidden", "routing_probs", "topk_indices", "w1_weights", "w2_weights"),
        cactus_attrs=("num_experts", "num_experts_per_tok", "normalize_routing", "epsilon", "routed_scaling_factor", "activation"),
        nodes={
            "router_logits": n("router_logits", ("aten.mm.default", "aten.matmul.default", "aten.linear.default")),
            "routing_probs": n("routing_probs", ("aten._softmax.default", "aten.sigmoid.default")),
            "topk": n("topk", "aten.topk.default"),
            "topk_values": n("topk_values", ("operator.getitem", "<built-in function getitem>")),
            "topk_indices": n("topk_indices", ("operator.getitem", "<built-in function getitem>")),
            "expert_up": n("expert_up", ("aten.mm.default", "aten.matmul.default", "aten.linear.default"), repeated=True),
            "expert_act": n("expert_act", ("aten.gelu.default", "aten.relu.default", "aten.silu.default"), repeated=True),
            "expert_down": n("expert_down", ("aten.mm.default", "aten.matmul.default", "aten.linear.default"), repeated=True),
            "weighted_expert": n("weighted_expert", "aten.mul.Tensor", repeated=True),
            "combine": n("combine", ("aten.add.Tensor", "aten.index_put.default", "aten.scatter_add.default")),
        },
        edges=(e("router_logits", "routing_probs", 0), e("routing_probs", "topk", 0), e("topk", "topk_values", 0, source_output_index=0), e("topk", "topk_indices", 0, source_output_index=1), e("expert_up", "expert_act", 0, repeated=True), e("expert_act", "expert_down", 0, repeated=True), e("expert_down", "weighted_expert", 0, repeated=True), e("topk_values", "weighted_expert", 1, repeated=True), e("weighted_expert", "combine", 0, repeated=True)),
        root="combine",
        outputs=("combine",),
        inputs=(inp("hidden", "router_logits", 0), inp("routing_probs", "routing_probs", 0), inp("topk_indices", "topk_indices", 0), inp("w1_weights", "expert_up", 1, variadic=True, repeated_group="experts"), inp("w2_weights", "expert_down", 1, variadic=True, repeated_group="experts")),
        output_attrs=(cap("num_experts", default=None, required=False), cap("num_experts_per_tok", "topk", "k"), cap("normalize_routing", default=False), cap("epsilon", default=1e-6), cap("routed_scaling_factor", default=1.0), cap("activation", "expert_act", "op")),
        constraints=("topk tuple outputs must be distinguished", "activation enum must be derived from expert_act op", "expert order must match weight lists"),
        repeated_groups=("experts",),
    ),
    "sample": direct("sample", ("logits",), ("temperature", "top_p", "top_k"), "sampling_subgraph", optional_attr_defaults={"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    "scatter_topk": direct("scatter_topk", ("indices", "values"), ("num_classes",), "scatter_topk_subgraph"),
    "persistent": direct("persistent", ("source_node",), (), "persistent_subgraph"),
    "rfft": direct("rfft", ("x",), (), "aten.fft_rfft.default"),
    "irfft": direct("irfft", ("x",), ("output_length",), "aten.fft_irfft.default"),
    "mel_filter_bank": source_op("mel_filter_bank", ("num_frequency_bins", "num_mel_filters", "min_frequency", "max_frequency", "sampling_rate", "norm_type", "scale_type"), "Generated constant/source op for audio preprocessing."),
    "spectrogram": direct("spectrogram", ("waveform", "mel_filters"), ("frame_length", "hop_length", "fft_length", "power", "center", "pad_mode", "mel_floor", "log_mel_mode", "dither", "preemphasis", "remove_dc_offset"), "spectrogram_subgraph"),
    "image_preprocess": direct("image_preprocess", ("pixel_input",), ("src_width", "src_height", "target_width", "target_height", "patch_size", "channels", "rescale_factor", "mean", "std_dev"), "image_preprocess_subgraph"),
}


ALL_FUSION_GRAPHS: tuple[FusionGraph, ...] = tuple(FUSION_GRAPHS.values())
