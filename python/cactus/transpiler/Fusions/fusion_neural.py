from __future__ import annotations

from . import edges as E
from . import models as M
from .fusion_builders import _graph, _input, _shared_input

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

LINEAR_NATIVE_GRAPH = _graph(
    "linear_native",
    "linear_native",
    ("linear_native",),
    inputs=(
        _input("x", "linear_native", 0),
        _input("weight", "linear_native", 1, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
        _input("bias", "linear_native", 2, optional=True, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=True, required=False),),
)

LINEAR_TRANSPOSED_GRAPH = _graph(
    "linear_transposed",
    "linear_mm",
    ("linear_weight_transpose", "linear_mm"),
    edge_names=E.EDGE_GROUPS["linear_transposed"],
    inputs=(
        _input("x", "linear_mm", 0),
        _input("weight", "linear_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=True, required=False),),
)

LINEAR_ADDMM_BIAS_GRAPH = _graph(
    "linear_addmm_bias",
    "linear_addmm",
    ("linear_addmm",),
    inputs=(
        _input("x", "linear_addmm", 1),
        _input("weight", "linear_addmm", 2),
        _input("bias", "linear_addmm", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=False, required=False),),
    constraints={
        "linear_weight_layout": {"input_role": "x", "weight_role": "weight", "output_node": "linear_addmm"},
    },
)

LINEAR_ADDMM_TRANSPOSED_BIAS_GRAPH = _graph(
    "linear_addmm_transposed_bias",
    "linear_addmm",
    ("linear_weight_transpose", "linear_addmm"),
    edge_names=("linear_weight_transpose_to_addmm",),
    inputs=(
        _input("x", "linear_addmm", 1),
        _input("weight", "linear_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
        _input("bias", "linear_addmm", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    attr_captures=(M.AttrCapture("pretransposed_rhs", default=True, required=False),),
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

LAYERNORM_DIRECT_GRAPH = _graph(
    "layernorm_direct",
    "layernorm_direct",
    ("layernorm_direct",),
    inputs=(
        _input("x", "layernorm_direct", 0),
        _input("weight", "layernorm_direct", 1, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
        _input("bias", "layernorm_direct", 2, optional=True, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    attr_captures=(M.AttrCapture("epsilon", "layernorm_direct", "eps", default=1e-5, required=False),),
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

RMS_NORM_CAST_WEIGHT_LEFT_GRAPH = _graph(
    "rms_norm_cast_weight_left",
    "rms_weight_mul",
    ("rms_square", "rms_mean", "rms_eps_add", "rms_inv", "rms_scale", "precision_cast", "rms_weight_mul"),
    edge_names=(
        "rms_square_to_mean",
        "rms_mean_to_eps_add",
        "rms_eps_add_to_inv",
        "rms_inv_to_scale",
        "rms_scale_to_precision_cast",
        "rms_precision_cast_to_weight_mul",
    ),
    inputs=(_input("x", "rms_square", 0), _input("weight", "rms_weight_mul", 0)),
    shared_inputs=(_shared_input("rms_square", 0, "rms_scale", 0),),
    attr_captures=(M.AttrCapture("epsilon", "rms_eps_add", "other", default=1e-6, required=False),),
    constraints={
        "node_attr_equals": {"node": "rms_mean", "attr": "dim", "value": -1},
    },
)

RMS_NORM_POW_GRAPH = _graph(
    "rms_norm_pow",
    "rms_weight_mul",
    ("rms_square", "rms_mean", "rms_eps_add", "rms_inv_pow", "rms_scale", "rms_weight_mul"),
    edge_names=E.EDGE_GROUPS["rms_norm_pow"],
    inputs=(_input("x", "rms_square", 0), _input("weight", "rms_weight_mul", 1)),
    shared_inputs=(_shared_input("rms_square", 0, "rms_scale", 0),),
    attr_captures=(M.AttrCapture("epsilon", "rms_eps_add", "other", default=1e-6, required=False),),
    constraints={
        "node_attr_equals": {"node": "rms_mean", "attr": "dim", "value": -1},
    },
)

RMS_NORM_NO_WEIGHT_GRAPH = _graph(
    "rms_norm_no_weight",
    "rms_scale",
    ("rms_square", "rms_mean", "rms_eps_add", "rms_inv", "rms_scale"),
    edge_names=E.EDGE_GROUPS["rms_norm_no_weight"],
    inputs=(_input("x", "rms_square", 0),),
    shared_inputs=(_shared_input("rms_square", 0, "rms_scale", 0),),
    attr_captures=(M.AttrCapture("epsilon", "rms_eps_add", "other", default=1e-6, required=False),),
)

RMS_NORM_POW_NO_WEIGHT_GRAPH = _graph(
    "rms_norm_pow_no_weight",
    "rms_scale",
    ("rms_square", "rms_mean", "rms_eps_add", "rms_inv_pow", "rms_scale"),
    edge_names=E.EDGE_GROUPS["rms_norm_pow_no_weight"],
    inputs=(_input("x", "rms_square", 0),),
    shared_inputs=(_shared_input("rms_square", 0, "rms_scale", 0),),
    attr_captures=(M.AttrCapture("epsilon", "rms_eps_add", "other", default=1e-6, required=False),),
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

SILU_DECOMPOSED_GRAPH = _graph(
    "silu_decomposed",
    "silu_product",
    ("silu_sigmoid", "silu_product"),
    edge_names=E.EDGE_GROUPS["silu_decomposed"],
    inputs=(_input("x", "silu_product", 0),),
    shared_inputs=(_shared_input("silu_product", 0, "silu_sigmoid", 0),),
)

GLU_DECOMPOSED_GRAPH = _graph(
    "glu_decomposed",
    "glu_product",
    ("glu_left_slice", "glu_right_slice", "glu_sigmoid", "glu_product"),
    edge_names=E.EDGE_GROUPS["glu_decomposed"],
    inputs=(_input("x", "glu_left_slice", 0),),
    shared_inputs=(_shared_input("glu_left_slice", 0, "glu_right_slice", 0),),
    attr_captures=(M.AttrCapture("axis", "glu_left_slice", "axis", default=-1, required=False),),
    constraints={
        "slice_halves": {
            "left_node": "glu_left_slice",
            "right_node": "glu_right_slice",
        },
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

GEMMA4_GEGLU_MLP_GRAPH = _graph(
    "gemma4_geglu_mlp",
    "down_proj_view",
    (
        "gate_input_view",
        "gate_weight_transpose",
        "gate_proj",
        "gate_proj_view",
        "gelu",
        "up_input_view",
        "up_weight_transpose",
        "up_proj",
        "up_proj_view",
        "mlp_product",
        "mlp_product_view",
        "down_weight_transpose",
        "down_proj",
        "down_proj_view",
    ),
    edge_names=E.EDGE_GROUPS["gemma4_geglu_mlp"],
    inputs=(
        _input("hidden", "gate_input_view", 0),
        _input("gate_weight", "gate_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
        _input("up_weight", "up_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
        _input("down_weight", "down_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.LIFTED_CONSTANT)),
    ),
    shared_inputs=(_shared_input("gate_input_view", 0, "up_input_view", 0),),
    attr_captures=(M.AttrCapture("product_scale", default=1.0, required=False),),
    constraints={
        "node_attr_equals": {"node": "gelu", "attr": "approximate", "value": "tanh"},
    },
)
