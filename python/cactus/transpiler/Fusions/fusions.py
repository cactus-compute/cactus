from __future__ import annotations

from . import models as M
from .fusion_attention import *
from .fusion_builders import (
    _definition,
    _index_by_cactus_op,
    _index_by_field,
    _index_by_root_op,
    _index_by_target,
    _required_attrs,
)
from .fusion_cache_conv import *
from .fusion_direct import *
from .fusion_moe import *
from .fusion_neural import *
from .fusion_special import *


GRAPH_BY_NAME: dict[str, M.FusionGraph] = {
    **DIRECT_GRAPHS,
    **DSP_GRAPHS,
    "linear": LINEAR_GRAPH,
    "linear_transposed": LINEAR_TRANSPOSED_GRAPH,
    "linear_bias": LINEAR_BIAS_GRAPH,
    "rms_norm": RMS_NORM_GRAPH,
    "rms_norm_pow": RMS_NORM_POW_GRAPH,
    "rms_norm_no_weight": RMS_NORM_NO_WEIGHT_GRAPH,
    "rms_norm_pow_no_weight": RMS_NORM_POW_NO_WEIGHT_GRAPH,
    "layernorm_no_bias": LAYERNORM_NO_BIAS_GRAPH,
    "layernorm": LAYERNORM_GRAPH,
    "silu_decomposed": SILU_DECOMPOSED_GRAPH,
    "glu_decomposed": GLU_DECOMPOSED_GRAPH,
    "swiglu_mlp": SWIGLU_MLP_GRAPH,
    "gelu_mlp": GELU_MLP_GRAPH,
    "gemma4_geglu_mlp": GEMMA4_GEGLU_MLP_GRAPH,
    "scaled_dot_product_attention": ATTENTION_DIRECT_GRAPH,
    "attention_core": ATTENTION_CORE_GRAPH,
    "attention_masked": ATTENTION_MASKED_GRAPH,
    "gemma4_attention_layout": GEMMA4_ATTENTION_LAYOUT_GRAPH,
    "lfm_bmm_masked_attention": LFM_BMM_MASKED_ATTENTION_GRAPH,
    "rope": ROPE_GRAPH,
    "conv": CONV_GRAPH,
    "lfm_short_conv_prefill": LFM_SHORT_CONV_PREFILL_GRAPH,
    "lfm_short_conv_decode": LFM_SHORT_CONV_DECODE_GRAPH,
    "conv_bias": CONV_BIAS_GRAPH,
    "kv_cache_append": KV_CACHE_APPEND_GRAPH,
    "kv_cache_initial_append": KV_CACHE_INITIAL_APPEND_GRAPH,
    "attention_cached": ATTENTION_CACHED_GRAPH,
    "moe_layer_gated": MOE_GATED_GRAPH,
    "lfm_grouped_moe": LFM_GROUPED_MOE_GRAPH,
    "lfm_grouped_moe_no_token_clone": LFM_GROUPED_MOE_NO_TOKEN_CLONE_GRAPH,
    "lstm_cell": LSTM_CELL_GRAPH,
    "gated_deltanet_decode": DELTANET_DECODE_GRAPH,
    "gated_deltanet_prefill": DELTANET_PREFILL_GRAPH,
    "rel_pos_bias": REL_POS_BIAS_GRAPH,
    "kv_cache_state": KV_CACHE_STATE_GRAPH,
    "conv_cache_state": CONV_CACHE_STATE_GRAPH,
    "conv_cache_append": CONV_CACHE_APPEND_GRAPH,
    "lfm_conv_cache_roll_append": LFM_CONV_CACHE_ROLL_APPEND_GRAPH,
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
    "linear_transposed": _definition("linear_transposed", "matmul", LINEAR_TRANSPOSED_GRAPH, fusion_fields=("generic", "linear")),
    "linear_bias": _definition("linear_bias", "linear", LINEAR_BIAS_GRAPH, fusion_fields=("generic", "linear")),
    "rms_norm": _definition("rms_norm", "rms_norm", RMS_NORM_GRAPH, fusion_fields=("generic", "rmsnorm", "gemma4_rmsnorm", "qwen2_5_rmsnorm")),
    "rms_norm_pow": _definition("rms_norm_pow", "rms_norm", RMS_NORM_POW_GRAPH, fusion_fields=("generic", "rmsnorm", "gemma4_rmsnorm", "qwen2_5_rmsnorm")),
    "rms_norm_no_weight": _definition("rms_norm_no_weight", "rms_norm", RMS_NORM_NO_WEIGHT_GRAPH, fusion_fields=("generic", "rmsnorm", "gemma4_rmsnorm", "qwen2_5_rmsnorm")),
    "rms_norm_pow_no_weight": _definition("rms_norm_pow_no_weight", "rms_norm", RMS_NORM_POW_NO_WEIGHT_GRAPH, fusion_fields=("generic", "rmsnorm", "gemma4_rmsnorm", "qwen2_5_rmsnorm")),
    "layernorm_no_bias": _definition("layernorm_no_bias", "layernorm", LAYERNORM_NO_BIAS_GRAPH, fusion_fields=("generic", "normalization")),
    "layernorm": _definition("layernorm", "layernorm", LAYERNORM_GRAPH, fusion_fields=("generic", "normalization")),
    "silu_decomposed": _definition("silu_decomposed", "silu", SILU_DECOMPOSED_GRAPH, fusion_fields=("generic", "activation", "audio", "mlp")),
    "glu_decomposed": _definition("glu_decomposed", "glu", GLU_DECOMPOSED_GRAPH, fusion_fields=("generic", "activation", "audio")),
    "swiglu_mlp": _definition("swiglu_mlp", "dense_mlp_tq_fused", SWIGLU_MLP_GRAPH, fusion_fields=("generic", "mlp", "gemma4_mlp", "qwen2_5_mlp", "lfm_mlp")),
    "gelu_mlp": _definition("gelu_mlp", "matmul", GELU_MLP_GRAPH, fusion_fields=("generic", "mlp")),
    "gemma4_geglu_mlp": _definition("gemma4_geglu_mlp", "dense_mlp_tq_fused", GEMMA4_GEGLU_MLP_GRAPH, fusion_fields=("gemma4_mlp",)),
    "scaled_dot_product_attention": _definition("scaled_dot_product_attention", "attention", ATTENTION_DIRECT_GRAPH, fusion_fields=("generic", "attention")),
    "attention_core": _definition("attention_core", "attention", ATTENTION_CORE_GRAPH, fusion_fields=("generic", "attention", "gemma4_attention", "qwen2_5_attention", "lfm_attention")),
    "attention_masked": _definition("attention_masked", "attention", ATTENTION_MASKED_GRAPH, fusion_fields=("generic", "attention", "gemma4_attention")),
    "gemma4_attention_layout": _definition(
        "gemma4_attention_layout",
        "attention",
        GEMMA4_ATTENTION_LAYOUT_GRAPH,
        fusion_fields=("attention", "gemma4_attention"),
        supported_inference_modes=("prefill_with_cache", "decode_with_cache"),
        metadata={"special_matcher": "gemma4_attention_layout"},
    ),
    "gemma4_bmm_masked_attention": _definition("gemma4_bmm_masked_attention", "attention", LFM_BMM_MASKED_ATTENTION_GRAPH, fusion_fields=("generic", "attention", "gemma4_attention")),
    "lfm_bmm_masked_attention": _definition("lfm_bmm_masked_attention", "attention", LFM_BMM_MASKED_ATTENTION_GRAPH, fusion_fields=("generic", "attention", "lfm_attention", "lfm_moe")),
    "rope": _definition("rope", "rope", ROPE_GRAPH, fusion_fields=("generic", "rope", "gemma4_rope", "qwen2_5_rope")),
    "conv": _definition("conv", "conv1d", CONV_GRAPH, fusion_fields=("generic", "conv", "audio", "vision"), metadata=_required_attrs(padding=0, dilation=1, groups=1)),
    "conv_bias": _definition("conv_bias", "conv1d", CONV_BIAS_GRAPH, fusion_fields=("generic", "conv", "audio", "vision")),
    "lfm_short_conv_prefill": _definition("lfm_short_conv_prefill", "conv1d_causal", LFM_SHORT_CONV_PREFILL_GRAPH, fusion_fields=("generic", "conv", "cache", "lfm_moe"), supported_inference_modes=("prefill_with_cache",)),
    "lfm_short_conv_decode": _definition("lfm_short_conv_decode", "lfm_short_conv_decode", LFM_SHORT_CONV_DECODE_GRAPH, fusion_fields=("generic", "conv", "cache", "lfm_moe"), supported_inference_modes=("decode_with_cache",)),
    "conv1d_k3": _definition("conv1d_k3", "conv1d_k3", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=3, stride=1, padding=0, dilation=1, groups=1)),
    "conv1d_k7s3": _definition("conv1d_k7s3", "conv1d_k7s3", CONV_BIAS_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=7, stride=3)),
    "conv1d_causal": _definition("conv1d_causal", "conv1d_causal", CONV_GRAPH, fusion_fields=("generic", "conv", "cache"), metadata=_required_attrs(causal=True)),
    "conv1d_pointwise": _definition("conv1d_pointwise", "conv1d_pointwise", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=1)),
    "conv1d_same_depthwise_k9": _definition("conv1d_same_depthwise_k9", "conv1d_same_depthwise_k9", CONV_GRAPH, fusion_fields=("generic", "conv", "audio"), metadata=_required_attrs(kernel_size=9, depthwise=True)),
    "conv2d_k3s2p1": _definition("conv2d_k3s2p1", "conv2d_k3s2p1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=3, stride=2, padding=1)),
    "conv2d_depthwise_k3s2p1": _definition("conv2d_depthwise_k3s2p1", "conv2d_depthwise_k3s2p1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=3, stride=2, padding=1, depthwise=True)),
    "conv2d_pointwise_1x1": _definition("conv2d_pointwise_1x1", "conv2d_pointwise_1x1", CONV_GRAPH, fusion_fields=("generic", "conv", "vision"), metadata=_required_attrs(kernel_size=1)),
    "kv_cache_append": _definition("kv_cache_append", "kv_cache_append", KV_CACHE_APPEND_GRAPH, fusion_fields=("generic", "cache"), supported_inference_modes=("prefill_with_cache", "decode_with_cache")),
    "kv_cache_initial_append": _definition("kv_cache_initial_append", "kv_cache_append", KV_CACHE_INITIAL_APPEND_GRAPH, fusion_fields=("generic", "cache"), supported_inference_modes=("prefill_with_cache",)),
    "attention_cached": _definition("attention_cached", "attention_cached", ATTENTION_CACHED_GRAPH, fusion_fields=("generic", "attention", "cache"), supported_inference_modes=("decode_with_cache",)),
    "moe_layer_gated": _definition("moe_layer_gated", "moe_layer_gated", MOE_GATED_GRAPH, fusion_fields=("generic", "moe")),
    "lfm_grouped_moe": _definition("lfm_grouped_moe", "moe_layer_gated", LFM_GROUPED_MOE_GRAPH, fusion_fields=("generic", "moe", "lfm_moe")),
    "lfm_grouped_moe_no_token_clone": _definition("lfm_grouped_moe_no_token_clone", "moe_layer_gated", LFM_GROUPED_MOE_NO_TOKEN_CLONE_GRAPH, fusion_fields=("generic", "moe", "lfm_moe")),
    "lstm_cell": _definition("lstm_cell", "lstm_cell", LSTM_CELL_GRAPH, fusion_fields=("generic", "recurrent", "audio")),
    "gated_deltanet_decode": _definition("gated_deltanet_decode", "gated_deltanet_decode", DELTANET_DECODE_GRAPH, fusion_fields=("generic", "recurrent", "cache")),
    "gated_deltanet_prefill": _definition("gated_deltanet_prefill", "gated_deltanet_prefill", DELTANET_PREFILL_GRAPH, fusion_fields=("generic", "recurrent", "cache")),
    "rel_pos_bias": _definition("rel_pos_bias", "rel_pos_bias", REL_POS_BIAS_GRAPH, fusion_fields=("generic", "attention")),
    "kv_cache_state": _definition("kv_cache_state", "kv_cache_state", KV_CACHE_STATE_GRAPH, fusion_fields=("generic", "cache"), supported_inference_modes=("prefill_with_cache", "decode_with_cache")),
    "conv_cache_state": _definition("conv_cache_state", "conv_cache_state", CONV_CACHE_STATE_GRAPH, fusion_fields=("generic", "cache", "conv")),
    "conv_cache_append": _definition("conv_cache_append", "conv_cache_append", CONV_CACHE_APPEND_GRAPH, fusion_fields=("generic", "cache", "conv")),
    "lfm_conv_cache_roll_append": _definition("lfm_conv_cache_roll_append", "conv_cache_append", LFM_CONV_CACHE_ROLL_APPEND_GRAPH, fusion_fields=("generic", "cache", "conv", "lfm_moe"), supported_inference_modes=("decode_with_cache",)),
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
}


FUSION_CATALOG = M.FusionCatalog(fusions=tuple(FUSIONS.values()))


FUSIONS_BY_TARGET: dict[str, M.FusionDefinition] = _index_by_target(FUSIONS)


FUSIONS_BY_CACTUS_OP: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_cactus_op(FUSIONS)


FUSIONS_BY_FIELD: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_field(FUSIONS)


FUSIONS_BY_ROOT_OP: dict[str, tuple[M.FusionDefinition, ...]] = _index_by_root_op(FUSIONS)
