from . import models as M

EXPERT_GROUP = "experts"


def _edge(
    source: str,
    dest: str,
    dest_input_index: int | None = 0,
    *,
    source_output_index: int | None = None,
    optional: bool = False,
    transparent: bool = False,
    repeated: bool = False,
    repeated_group: str | None = None,
    metadata: dict | None = None,
) -> M.FusionEdge:
    return M.FusionEdge(
        source=source,
        dest=dest,
        dest_input_index=dest_input_index,
        source_output_index=source_output_index,
        optional=optional,
        transparent=transparent,
        repeated=repeated,
        repeated_group=repeated_group,
        metadata=metadata or {},
    )


def _expert_edge(
    source: str,
    dest: str,
    dest_input_index: int | None = 0,
) -> M.FusionEdge:
    return _edge(
        source,
        dest,
        dest_input_index,
        repeated=True,
        repeated_group=EXPERT_GROUP,
    )


def _index_edges_by_source(edges: dict[str, M.FusionEdge]) -> dict[str, tuple[str, ...]]:
    index: dict[str, tuple[str, ...]] = {}

    for edge_name, edge in edges.items():
        index[edge.source] = (*index.get(edge.source, ()), edge_name)

    return index


def _index_edges_by_dest(edges: dict[str, M.FusionEdge]) -> dict[str, tuple[str, ...]]:
    index: dict[str, tuple[str, ...]] = {}

    for edge_name, edge in edges.items():
        index[edge.dest] = (*index.get(edge.dest, ()), edge_name)

    return index


EDGES: dict[str, M.FusionEdge] = {
    "linear_mm_to_bias_add": _edge("linear_mm", "linear_bias_add", 0),
    "rms_square_to_mean": _edge("rms_square", "rms_mean", 0),
    "rms_mean_to_eps_add": _edge("rms_mean", "rms_eps_add", 0),
    "rms_eps_add_to_inv": _edge("rms_eps_add", "rms_inv", 0),
    "rms_inv_to_scale": _edge("rms_inv", "rms_scale", 1),
    "rms_scale_to_weight_mul": _edge("rms_scale", "rms_weight_mul", 0),
    "ln_mean_to_center": _edge("ln_mean", "ln_center", 1),
    "ln_center_to_square": _edge("ln_center", "ln_square", 0),
    "ln_square_to_var": _edge("ln_square", "ln_var", 0),
    "ln_var_to_eps_add": _edge("ln_var", "ln_eps_add", 0),
    "ln_eps_add_to_inv": _edge("ln_eps_add", "ln_inv", 0),
    "ln_center_to_norm": _edge("ln_center", "ln_norm", 0),
    "ln_inv_to_norm": _edge("ln_inv", "ln_norm", 1),
    "ln_norm_to_weight_mul": _edge("ln_norm", "ln_weight_mul", 0),
    "ln_weight_mul_to_bias_add": _edge("ln_weight_mul", "ln_bias_add", 0, optional=True),
    "gate_proj_to_activation": _edge("gate_proj", "mlp_activation", 0),
    "mlp_activation_to_product": _edge("mlp_activation", "mlp_product", 0),
    "up_proj_to_product": _edge("up_proj", "mlp_product", 1),
    "mlp_product_to_down_proj": _edge("mlp_product", "down_proj", 0),
    "up_proj_to_gelu": _edge("up_proj", "gelu", 0),
    "gelu_to_down_proj": _edge("gelu", "down_proj", 0),
    "q_proj_to_q_view": _edge("q_proj", "q_view", 0, optional=True, transparent=True),
    "k_proj_to_k_view": _edge("k_proj", "k_view", 0, optional=True, transparent=True),
    "v_proj_to_v_view": _edge("v_proj", "v_view", 0, optional=True, transparent=True),
    "q_view_to_q_rope": _edge("q_view", "q_rope", 0, optional=True, transparent=True),
    "k_view_to_k_rope": _edge("k_view", "k_rope", 0, optional=True, transparent=True),
    "q_rope_to_qk": _edge("q_rope", "attn_qk", 0),
    "k_rope_to_qk": _edge("k_rope", "attn_qk", 1),
    "attn_qk_to_scale": _edge("attn_qk", "attn_scale", 0),
    "attn_scale_to_mask_add": _edge("attn_scale", "attn_mask_add", 0, optional=True),
    "attn_scale_to_softmax": _edge("attn_scale", "attn_softmax", 0),
    "attn_mask_add_to_softmax": _edge("attn_mask_add", "attn_softmax", 0, optional=True),
    "attn_softmax_to_value": _edge("attn_softmax", "attn_value", 0),
    "v_view_to_value": _edge("v_view", "attn_value", 1, optional=True, transparent=True),
    "rope_slice_odd_to_neg": _edge("rope_slice_odd", "rope_neg", 0),
    "rope_neg_to_rotate_cat": _edge("rope_neg", "rope_rotate_cat", 0),
    "rope_slice_even_to_rotate_cat": _edge("rope_slice_even", "rope_rotate_cat", 1),
    "rope_cos_mul_to_add": _edge("rope_cos_mul", "rope_add", 0),
    "rope_sin_mul_to_add": _edge("rope_sin_mul", "rope_add", 1),
    "rope_rotate_cat_to_sin_mul": _edge("rope_rotate_cat", "rope_sin_mul", 0),
    "conv_to_bias_add": _edge("conv", "conv_bias_add", 0, optional=True),
    "lstm_gate_mm_to_add": _edge("lstm_gate_mm", "lstm_gate_add", 0),
    "lstm_recurrent_mm_to_add": _edge("lstm_recurrent_mm", "lstm_gate_add", 1),
    "lstm_add_to_sigmoid": _edge("lstm_gate_add", "lstm_sigmoid", 0),
    "lstm_add_to_tanh": _edge("lstm_gate_add", "lstm_tanh", 0),
    "delta_q_to_decode": _edge("delta_q", "delta_gate", 0, optional=True),
    "delta_k_to_decode": _edge("delta_k", "delta_gate", 1, optional=True),
    "delta_v_to_decode": _edge("delta_v", "delta_gate", 2, optional=True),
    "router_logits_to_probs": _edge("router_logits", "routing_probs", 0),
    "routing_probs_to_topk": _edge("routing_probs", "topk", 0),
    "expert_gate_to_activation": _expert_edge("expert_gate", "expert_activation", 0),
    "expert_activation_to_product": _expert_edge("expert_activation", "expert_product", 0),
    "expert_up_to_product": _expert_edge("expert_up", "expert_product", 1),
    "expert_product_to_down": _expert_edge("expert_product", "expert_down", 0),
    "expert_down_to_weighted": _expert_edge("expert_down", "expert_weighted", 0),
    "expert_weighted_to_combine": _expert_edge("expert_weighted", "moe_combine", 0),
    "sample_topk_to_softmax": _edge("sample_topk", "sample_softmax", 0),
}


EDGE_GROUPS: dict[str, tuple[str, ...]] = {
    "linear_bias": ("linear_mm_to_bias_add",),
    "rms_norm": (
        "rms_square_to_mean",
        "rms_mean_to_eps_add",
        "rms_eps_add_to_inv",
        "rms_inv_to_scale",
        "rms_scale_to_weight_mul",
    ),
    "layernorm": (
        "ln_mean_to_center",
        "ln_center_to_square",
        "ln_square_to_var",
        "ln_var_to_eps_add",
        "ln_eps_add_to_inv",
        "ln_center_to_norm",
        "ln_inv_to_norm",
        "ln_norm_to_weight_mul",
        "ln_weight_mul_to_bias_add",
    ),
    "swiglu_mlp": (
        "gate_proj_to_activation",
        "mlp_activation_to_product",
        "up_proj_to_product",
        "mlp_product_to_down_proj",
    ),
    "gelu_mlp": (
        "up_proj_to_gelu",
        "gelu_to_down_proj",
    ),
    "attention_core": (
        "attn_qk_to_scale",
        "attn_scale_to_mask_add",
        "attn_scale_to_softmax",
        "attn_mask_add_to_softmax",
        "attn_softmax_to_value",
    ),
    "projected_attention": (
        "q_proj_to_q_view",
        "k_proj_to_k_view",
        "v_proj_to_v_view",
        "q_view_to_q_rope",
        "k_view_to_k_rope",
        "q_rope_to_qk",
        "k_rope_to_qk",
        "attn_qk_to_scale",
        "attn_scale_to_mask_add",
        "attn_mask_add_to_softmax",
        "attn_softmax_to_value",
        "v_view_to_value",
    ),
    "rope": (
        "rope_slice_odd_to_neg",
        "rope_neg_to_rotate_cat",
        "rope_slice_even_to_rotate_cat",
        "rope_rotate_cat_to_sin_mul",
        "rope_cos_mul_to_add",
        "rope_sin_mul_to_add",
    ),
    "conv_bias": ("conv_to_bias_add",),
    "lstm_cell": (
        "lstm_gate_mm_to_add",
        "lstm_recurrent_mm_to_add",
        "lstm_add_to_sigmoid",
        "lstm_add_to_tanh",
    ),
    "moe_expert": (
        "expert_gate_to_activation",
        "expert_activation_to_product",
        "expert_up_to_product",
        "expert_product_to_down",
        "expert_down_to_weighted",
        "expert_weighted_to_combine",
    ),
}


EDGE_NAMES_BY_SOURCE: dict[str, tuple[str, ...]] = _index_edges_by_source(EDGES)
EDGE_NAMES_BY_DEST: dict[str, tuple[str, ...]] = _index_edges_by_dest(EDGES)
