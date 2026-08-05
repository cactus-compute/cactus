from __future__ import annotations

from . import edges as E
from . import models as M
from .fusion_builders import _graph, _input, _shared_input, _single_node_graph

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

GEMMA4_ATTENTION_LAYOUT_GRAPH = _single_node_graph(
    "gemma4_attention_layout",
    "view",
    ("x",),
    attr_captures=(
        M.AttrCapture("scale", default=1.0, required=False),
        M.AttrCapture("input_layout", default="bhqd_bhds_bhsd", required=False),
        M.AttrCapture("output_layout", default="bhqd", required=False),
        M.AttrCapture("window_size", default=0, required=False),
    ),
    metadata={"special_matcher": "gemma4_attention_layout"},
)

WHISPER_ATTENTION_LAYOUT_GRAPH = _single_node_graph(
    "whisper_attention_layout",
    "view",
    ("x",),
    attr_captures=(
        M.AttrCapture("scale", default=1.0, required=False),
        M.AttrCapture("is_causal", default=False, required=False),
        M.AttrCapture("output_layout", default="bthd_flat", required=False),
        M.AttrCapture("window_size", default=0, required=False),
    ),
    metadata={"special_matcher": "whisper_attention_layout"},
)

GENERIC_CACHED_ATTENTION_GRAPH = _single_node_graph(
    "generic_cached_attention",
    "view",
    ("x",),
    attr_captures=(
        M.AttrCapture("scale", default=None, required=False),
        M.AttrCapture("is_causal", default=True, required=False),
        M.AttrCapture("position_offset", default=0, required=False),
        M.AttrCapture("window_size", default=0, required=False),
        M.AttrCapture("input_layout", default="bhqd_bhsd_bhsd", required=False),
        M.AttrCapture("output_layout", default="bhqd", required=False),
    ),
    metadata={"special_matcher": "generic_cached_attention"},
)

LFM_BMM_MASKED_ATTENTION_GRAPH = _graph(
    "lfm_bmm_masked_attention",
    "lfm_attn_output_view",
    (
        "lfm_attn_query_expand",
        "lfm_attn_key_expand",
        "lfm_attn_value_expand",
        "lfm_attn_query_flat",
        "lfm_attn_key_flat",
        "lfm_attn_value_flat",
        "lfm_attn_qk",
        "lfm_attn_logits_view",
        "lfm_attn_mask_where",
        "lfm_attn_mask_zero",
        "lfm_attn_mask_fill",
        "lfm_attn_mask_add",
        "lfm_attn_softmax",
        "lfm_attn_null_eq",
        "lfm_attn_valid_not",
        "lfm_attn_any_valid",
        "lfm_attn_invalid_not",
        "lfm_attn_zero_like",
        "lfm_attn_probs_where",
        "lfm_attn_probs_expand",
        "lfm_attn_probs_flat",
        "lfm_attn_output_bmm",
        "lfm_attn_output_view",
    ),
    edge_names=E.EDGE_GROUPS["lfm_bmm_masked_attention"],
    inputs=(
        _input("query", "lfm_attn_query_expand", 0),
        _input("key", "lfm_attn_key_expand", 0),
        _input("value", "lfm_attn_value_expand", 0),
        _input("mask_predicate", "lfm_attn_mask_where", 0, metadata={"drop_after_fusion": True}),
    ),
    attr_captures=(
        M.AttrCapture("scale", default=None, required=False),
        M.AttrCapture("is_causal", default=True, required=False),
        M.AttrCapture("position_offset", default=0, required=False),
        M.AttrCapture("window_size", default=0, required=False),
        M.AttrCapture("input_layout", default="bhqd_bhds_bhsd", required=False),
        M.AttrCapture("output_layout", default="bhqd", required=False),
        M.AttrCapture("dropped_mask_builder", default=True, required=False),
    ),
    constraints={
        "node_attr_equals": (
            {"node": "lfm_attn_softmax", "attr": "axis", "value": -1, "allow_missing": True},
            {"node": "lfm_attn_null_eq", "attr": "other", "value": None},
            {"node": "lfm_attn_any_valid", "attr": "arg_1", "value": -1},
        ),
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
