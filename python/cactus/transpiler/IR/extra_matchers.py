from __future__ import annotations

from typing import Any

from . import models
from .extra_matcher_cache import (
    match_cache_output_consumers,
    match_cache_roll_append_structure,
    match_conv_cache_initialize_structure,
    match_empty_cache_initializer,
    match_short_conv_decode_structure,
    match_short_conv_prefill_structure,
)
from .extra_matcher_common import ExtraMatcher
from .extra_matcher_generic import (
    match_input_broadcastable_to_node,
    match_input_value_kind,
    match_node_attr_equals,
    match_node_attrs_equal,
    match_parent_tensor_dim_equal,
    match_same_layer,
    match_slice_halves,
)
from .extra_matcher_linear import (
    match_linear_weight_layout,
    match_lstm_gate_structure,
    match_rope_tables_compatible,
)
from .extra_matcher_moe import (
    match_grouped_moe_structure,
    match_moe_expert_branch_routing,
    match_moe_routing_weights_combine,
)
from ..Fusions import models as FModels


def match_extra_constraints(source: models.Node, graph: models.Graph, fusion: FModels.FusionGraph, bindings: dict[str, models.Node]) -> bool:
    """
    Dispatches every structured FusionGraph constraint to its registered matcher.

    `fusion.constraints` is a dict keyed by constraint name. This function looks
    up each name in `EXTRA_MATCHERS`, normalizes each raw spec into one or more
    spec dictionaries, and requires every matcher invocation to return True.
    Unknown constraint names or malformed specs fail closed.
    """
    for constraint_name, raw_spec in fusion.constraints.items():
        matcher = EXTRA_MATCHERS.get(constraint_name)

        if matcher is None:
            return False

        specs = normalize_constraint_specs(raw_spec)

        if specs is None:
            return False

        for spec in specs:
            if not matcher(source, graph, fusion, bindings, spec):
                return False

    return True


def normalize_constraint_specs(raw_spec: Any) -> tuple[dict[str, Any], ...] | None:
    """
    Normalizes one constraint value into a tuple of spec dictionaries.

    A constraint key can map to a single dict or a list/tuple of dicts. The
    latter lets one matcher type be applied multiple times without duplicating
    registry keys. Non-dict specs return None so the caller can fail safely.
    """
    if isinstance(raw_spec, dict):
        return (raw_spec,)

    if isinstance(raw_spec, (list, tuple)) and all(isinstance(spec, dict) for spec in raw_spec):
        return tuple(raw_spec)

    return None


EXTRA_MATCHERS: dict[str, ExtraMatcher] = {
    "linear_weight_layout": match_linear_weight_layout,
    "rope_tables_compatible": match_rope_tables_compatible,
    "lstm_gate_structure": match_lstm_gate_structure,
    "moe_expert_branch_routing": match_moe_expert_branch_routing,
    "moe_routing_weights_combine": match_moe_routing_weights_combine,
    "grouped_moe_structure": match_grouped_moe_structure,
    "cache_roll_append_structure": match_cache_roll_append_structure,
    "empty_cache_initializer": match_empty_cache_initializer,
    "conv_cache_initialize_structure": match_conv_cache_initialize_structure,
    "short_conv_prefill_structure": match_short_conv_prefill_structure,
    "short_conv_decode_structure": match_short_conv_decode_structure,
    "cache_output_consumers": match_cache_output_consumers,
    "node_attr_equals": match_node_attr_equals,
    "node_attrs_equal": match_node_attrs_equal,
    "input_value_kind": match_input_value_kind,
    "parent_tensor_dim_equal": match_parent_tensor_dim_equal,
    "input_broadcastable_to_node": match_input_broadcastable_to_node,
    "same_layer": match_same_layer,
    "slice_halves": match_slice_halves,
}

