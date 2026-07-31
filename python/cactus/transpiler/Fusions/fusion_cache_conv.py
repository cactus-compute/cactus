from __future__ import annotations

from . import edges as E
from . import models as M
from .fusion_builders import _cache_input, _cache_mutation, _cache_output, _graph, _input


CONV_GRAPH = _graph(
    "conv",
    "conv",
    ("conv",),
    inputs=(_input("x", "conv", 0), _input("weight", "conv", 1), _input("bias", "conv", 2, optional=True)),
    attr_captures=(M.AttrCapture("stride", "conv", "stride", default=1, required=False), M.AttrCapture("padding", "conv", "padding", default=0, required=False), M.AttrCapture("dilation", "conv", "dilation", default=1, required=False), M.AttrCapture("groups", "conv", "groups", default=1, required=False)),
)


LFM_SHORT_CONV_PREFILL_GRAPH = _graph(
    "lfm_short_conv_prefill",
    "slice",
    ("conv", "slice"),
    edge_names=E.EDGE_GROUPS["short_conv_prefill"],
    inputs=(_input("x", "conv", 0), _input("weight", "conv", 1), _input("bias", "conv", 2, optional=True)),
    attr_captures=(
        M.AttrCapture("kernel_size", default=3, required=False),
        M.AttrCapture("dilation", "conv", "dilation", default=1, required=False),
        M.AttrCapture("causal", default=True, required=False),
        M.AttrCapture("layout", default="batch_hidden_sequence", required=False),
    ),
    constraints={
        "short_conv_prefill_structure": {"conv_node": "conv", "slice_node": "slice", "x_role": "x", "weight_role": "weight"},
    },
)


LFM_SHORT_CONV_DECODE_GRAPH = _graph(
    "lfm_short_conv_decode",
    "sum",
    ("multiply", "sum"),
    edge_names=E.EDGE_GROUPS["short_conv_decode"],
    inputs=(
        _input("cache_window", "multiply", 0, allowed_value_kinds=(M.ValueKind.CACHE_OUTPUT, M.ValueKind.CACHE_STATE, M.ValueKind.ACTIVATION)),
        _input("weight", "multiply", 1, allowed_value_kinds=(M.ValueKind.PARAMETER, M.ValueKind.BUFFER, M.ValueKind.ACTIVATION)),
    ),
    attr_captures=(
        M.AttrCapture("window_size", default=3, required=False),
        M.AttrCapture("layout", default="batch_hidden_sequence", required=False),
    ),
    constraints={
        "short_conv_decode_structure": {"cache_window_role": "cache_window", "multiply_node": "multiply", "sum_node": "sum", "weight_role": "weight"},
    },
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
    inputs=(
        _input("new_kv", "kv_cache_append", 1),
        _input("cache_state", "kv_cache_append", 0, allowed_value_kinds=(M.ValueKind.CACHE_INPUT, M.ValueKind.CACHE_STATE, M.ValueKind.CACHE_OUTPUT)),
    ),
    cache_inputs=(_cache_input("cache_state", "kv_cache_append", 0),),
    cache_outputs=(_cache_output("updated_cache_state", "kv_cache_append"),),
    cache_mutations=(_cache_mutation("append_kv", read_roles=("cache_state",), write_roles=("updated_cache_state",)),),
    constraints={
        "cache_output_consumers": {"node": "kv_cache_append", "allow_output_child": True},
    },
)


KV_CACHE_INITIAL_APPEND_GRAPH = _graph(
    "kv_cache_initial_append",
    "kv_cache_append",
    ("kv_cache_append",),
    inputs=(
        _input("new_kv", "kv_cache_append", 1),
        _input("empty_cache", "kv_cache_append", 0, metadata={"drop_after_fusion": True}),
    ),
    cache_outputs=(_cache_output("initialized_cache_state", "kv_cache_append"),),
    cache_mutations=(_cache_mutation("initialize_append_kv", write_roles=("initialized_cache_state",)),),
    constraints={
        "empty_cache_initializer": {"empty_role": "empty_cache", "new_value_role": "new_kv", "output_node": "kv_cache_append"},
        "cache_output_consumers": {"node": "kv_cache_append", "allow_output_child": True},
    },
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
    inputs=(
        _input("new_data", "conv_cache_append", 0),
        _input("cache_state", "conv_cache_append", 1, allowed_value_kinds=(M.ValueKind.CACHE_INPUT, M.ValueKind.CACHE_STATE, M.ValueKind.CACHE_OUTPUT)),
    ),
    cache_inputs=(_cache_input("cache_state", "conv_cache_append", 1, cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_outputs=(_cache_output("updated_cache_state", "conv_cache_append", cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_mutations=(_cache_mutation("append_conv_state", cache_kind=M.CacheKind.CONV, read_roles=("cache_state",), write_roles=("updated_cache_state",)),),
)


LFM_CONV_CACHE_ROLL_APPEND_GRAPH = _graph(
    "lfm_conv_cache_roll_append",
    "lfm_conv_cache_output_copy",
    (
        "lfm_conv_cache_clone",
        "lfm_conv_cache_roll_arange",
        "lfm_conv_cache_roll_add",
        "lfm_conv_cache_roll_mod",
        "lfm_conv_cache_gather",
        "lfm_conv_cache_scatter_slice_dim0",
        "lfm_conv_cache_scatter_slice_dim1",
        "lfm_conv_cache_value_slice_dim0",
        "lfm_conv_cache_value_slice_dim1",
        "lfm_conv_cache_value_slice_last",
        "lfm_conv_cache_value_copy",
        "lfm_conv_cache_scatter_dim2",
        "lfm_conv_cache_scatter_dim1",
        "lfm_conv_cache_scatter_dim0",
        "lfm_conv_cache_output_copy",
    ),
    edge_names=E.EDGE_GROUPS["lfm_conv_cache_roll_append"],
    inputs=(
        _input("new_data", "lfm_conv_cache_value_copy", 1),
        _input("cache_state", "lfm_conv_cache_clone", 0, allowed_value_kinds=(M.ValueKind.CACHE_INPUT, M.ValueKind.CACHE_STATE, M.ValueKind.CACHE_OUTPUT)),
    ),
    cache_inputs=(
        _cache_input("cache_state", "lfm_conv_cache_clone", 0, cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),
    ),
    cache_outputs=(
        _cache_output("updated_cache_state", "lfm_conv_cache_output_copy", cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),
    ),
    cache_mutations=(
        _cache_mutation("roll_append_conv_state", cache_kind=M.CacheKind.CONV, read_roles=("cache_state",), write_roles=("updated_cache_state",)),
    ),
    attr_captures=(
        M.AttrCapture("window_size", default=3, required=False),
    ),
    constraints={
        "cache_roll_append_structure": {
            "cache_role": "cache_state",
            "new_data_role": "new_data",
            "clone_node": "lfm_conv_cache_clone",
            "gather_node": "lfm_conv_cache_gather",
            "arange_node": "lfm_conv_cache_roll_arange",
            "add_node": "lfm_conv_cache_roll_add",
            "mod_node": "lfm_conv_cache_roll_mod",
            "scatter_slice_dim0_node": "lfm_conv_cache_scatter_slice_dim0",
            "scatter_slice_dim1_node": "lfm_conv_cache_scatter_slice_dim1",
            "value_slice_dim0_node": "lfm_conv_cache_value_slice_dim0",
            "value_slice_dim1_node": "lfm_conv_cache_value_slice_dim1",
            "value_slice_last_node": "lfm_conv_cache_value_slice_last",
            "scatter_dim2_node": "lfm_conv_cache_scatter_dim2",
            "scatter_dim1_node": "lfm_conv_cache_scatter_dim1",
            "scatter_dim0_node": "lfm_conv_cache_scatter_dim0",
        },
    },
)


CONV_CACHE_INITIALIZE_GRAPH = _graph(
    "conv_cache_initialize",
    "conv_cache_initialize",
    ("conv_cache_initialize",),
    inputs=(
        _input("rows", "conv_cache_initialize", 1),
        _input("empty_cache", "conv_cache_initialize", 0, metadata={"drop_after_fusion": True}),
    ),
    cache_outputs=(_cache_output("initialized_cache_state", "conv_cache_initialize", cache_kind=M.CacheKind.CONV, tensor_role=M.CacheTensorRole.STATE),),
    cache_mutations=(_cache_mutation("initialize_conv_state", cache_kind=M.CacheKind.CONV, write_roles=("initialized_cache_state",)),),
    constraints={
        "conv_cache_initialize_structure": {"rows_role": "rows", "empty_role": "empty_cache", "output_node": "conv_cache_initialize"},
    },
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
