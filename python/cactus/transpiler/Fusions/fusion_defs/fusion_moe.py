from __future__ import annotations

from .. import edges as E
from .. import models as M
from ..fusion_builders import _graph, _input, _shared_input, _variadic_input

EXPERT_BRANCH_GRAPH = _graph(
    "moe_expert_branch",
    "expert_weighted",
    ("expert_gate", "expert_up", "expert_activation", "expert_product", "expert_down", "expert_weighted"),
    edge_names=("expert_gate_to_activation", "expert_activation_to_product", "expert_up_to_product", "expert_product_to_down", "expert_down_to_weighted"),
    inputs=(_input("hidden", "expert_gate", 0), _input("gate_weight", "expert_gate", 1), _input("up_weight", "expert_up", 1), _input("down_weight", "expert_down", 1), _input("routing_weight", "expert_weighted", 1)),
    shared_inputs=(_shared_input("expert_gate", 0, "expert_up", 0),),
)

MOE_GATED_GRAPH = _graph(
    "moe_layer_gated",
    "moe_combine",
    ("moe_combine",),
    inputs=(_variadic_input("expert_outputs", "moe_combine", 0),),
    repeated_subgraphs=(M.RepeatedSubgraph("experts", EXPERT_BRANCH_GRAPH),),
    attr_captures=(M.AttrCapture("num_experts", default=None, required=False), M.AttrCapture("num_experts_per_tok", "topk", "k", required=False), M.AttrCapture("normalize_routing", default=True, required=False), M.AttrCapture("epsilon", default=1e-6, required=False), M.AttrCapture("routed_scaling_factor", default=1.0, required=False)),
    constraints={
        "moe_expert_branch_routing": {"repeated_subgraph": "experts", "combine_node": "moe_combine", "routing_weight_role": "routing_weight", "max_depth": 64},
        "moe_routing_weights_combine": {"repeated_subgraph": "experts", "combine_node": "moe_combine", "weighted_node": "expert_weighted", "routing_weight_role": "routing_weight", "max_depth": 64},
    },
)

LFM_GROUPED_MOE_GRAPH = _graph(
    "lfm_grouped_moe",
    "moe_grouped_combine",
    (
        "moe_hidden_view",
        "moe_router_weight_transpose",
        "moe_router_logits",
        "moe_router_sigmoid",
        "moe_router_bias",
        "moe_topk",
        "moe_topk_indices",
        "moe_routing_gather",
        "moe_routing_sum",
        "moe_routing_eps",
        "moe_routing_norm",
        "moe_routing_scale",
        "moe_token_arange",
        "moe_token_unsqueeze",
        "moe_token_expand",
        "moe_token_clone",
        "moe_token_flat",
        "moe_routing_flat",
        "moe_topk_flat",
        "moe_sort",
        "moe_sort_indices",
        "moe_unsort_empty",
        "moe_unsort_arange",
        "moe_unsort_index_put",
        "moe_sorted_expert_index",
        "moe_sorted_expert_copy",
        "moe_histc",
        "moe_offsets",
        "moe_sorted_routing_index",
        "moe_token_positions_index",
        "moe_sorted_hidden_index",
        "moe_gate_up_weight_transpose",
        "moe_grouped_gate_up",
        "moe_gate_up_split",
        "moe_gate_tensor",
        "moe_up_tensor",
        "moe_gate_copy",
        "moe_gate_sigmoid",
        "moe_gate_activation",
        "moe_gate_activation_copy",
        "moe_expert_product",
        "moe_down_weight_transpose",
        "moe_grouped_down",
        "moe_routing_unsqueeze",
        "moe_grouped_weighted",
        "moe_unsort_index",
        "moe_output_view",
        "moe_grouped_combine",
    ),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe"],
    inputs=(
        _input("hidden", "moe_hidden_view", 0),
        _input("router_weight", "moe_router_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER)),
        _input("router_bias", "moe_router_bias", 1, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER)),
        _input("gate_up_weight", "moe_gate_up_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER)),
        _input("down_weight", "moe_down_weight_transpose", 0, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER)),
    ),
    attr_captures=(
        M.AttrCapture("num_experts", "moe_histc", "bins", required=False),
        M.AttrCapture("num_experts_per_tok", "moe_topk", "k", required=True),
        M.AttrCapture("router_activation", default="sigmoid", required=False),
        M.AttrCapture("use_expert_bias", default=True, required=False),
        M.AttrCapture("normalize_routing", default=True, required=False),
        M.AttrCapture("activation", default="silu", required=False),
        M.AttrCapture("epsilon", "moe_routing_eps", "other", default=1e-6, required=False),
        M.AttrCapture("routed_scaling_factor", "moe_routing_scale", "other", default=1.0, required=False),
    ),
    constraints={
        "grouped_moe_structure": {
            "topk_node": "moe_topk",
            "topk_indices_node": "moe_topk_indices",
            "router_bias_role": "router_bias",
            "gate_up_weight_role": "gate_up_weight",
            "down_weight_role": "down_weight",
            "gate_up_grouped_node": "moe_grouped_gate_up",
            "down_grouped_node": "moe_grouped_down",
            "split_node": "moe_gate_up_split",
            "offsets_node": "moe_offsets",
            "combine_node": "moe_grouped_combine",
        },
    },
)

LFM_GROUPED_MOE_NO_TOKEN_CLONE_GRAPH = _graph(
    "lfm_grouped_moe_no_token_clone",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_GRAPH.nodes if node_name != "moe_token_clone"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_no_token_clone"],
    inputs=LFM_GROUPED_MOE_GRAPH.inputs,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_SILU_GRAPH = _graph(
    "lfm_grouped_moe_silu",
    "moe_grouped_combine",
    tuple(
        "moe_gate_silu" if node_name == "moe_gate_activation" else node_name
        for node_name in LFM_GROUPED_MOE_GRAPH.nodes
        if node_name != "moe_gate_sigmoid"
    ),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_silu"],
    inputs=LFM_GROUPED_MOE_GRAPH.inputs,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_SILU_NO_TOKEN_CLONE_GRAPH = _graph(
    "lfm_grouped_moe_silu_no_token_clone",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_SILU_GRAPH.nodes if node_name != "moe_token_clone"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_silu_no_token_clone"],
    inputs=LFM_GROUPED_MOE_GRAPH.inputs,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_DIRECT_ROUTER_INPUTS = tuple(
    input_spec
    for input_spec in LFM_GROUPED_MOE_GRAPH.inputs
    if input_spec.role != "router_weight"
) + (
    _input("router_weight", "moe_router_logits", 1, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER)),
)

LFM_GROUPED_MOE_DIRECT_ROUTER_GRAPH = _graph(
    "lfm_grouped_moe_direct_router",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_GRAPH.nodes if node_name != "moe_router_weight_transpose"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_direct_router"],
    inputs=LFM_GROUPED_MOE_DIRECT_ROUTER_INPUTS,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_DIRECT_ROUTER_NO_TOKEN_CLONE_GRAPH = _graph(
    "lfm_grouped_moe_direct_router_no_token_clone",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_DIRECT_ROUTER_GRAPH.nodes if node_name != "moe_token_clone"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_direct_router_no_token_clone"],
    inputs=LFM_GROUPED_MOE_DIRECT_ROUTER_INPUTS,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_SILU_DIRECT_ROUTER_GRAPH = _graph(
    "lfm_grouped_moe_silu_direct_router",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_SILU_GRAPH.nodes if node_name != "moe_router_weight_transpose"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_silu_direct_router"],
    inputs=LFM_GROUPED_MOE_DIRECT_ROUTER_INPUTS,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)

LFM_GROUPED_MOE_SILU_DIRECT_ROUTER_NO_TOKEN_CLONE_GRAPH = _graph(
    "lfm_grouped_moe_silu_direct_router_no_token_clone",
    "moe_grouped_combine",
    tuple(node_name for node_name in LFM_GROUPED_MOE_SILU_DIRECT_ROUTER_GRAPH.nodes if node_name != "moe_token_clone"),
    edge_names=E.EDGE_GROUPS["lfm_grouped_moe_silu_direct_router_no_token_clone"],
    inputs=LFM_GROUPED_MOE_DIRECT_ROUTER_INPUTS,
    attr_captures=LFM_GROUPED_MOE_GRAPH.attr_captures,
    constraints=LFM_GROUPED_MOE_GRAPH.constraints,
)
