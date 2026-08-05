CACTUS_TARGET_PREFIX = "cactus."
DEFAULT_COMPONENT_NAME = "model"
DEFAULT_GRAPH_SUFFIX = ".cactus"
DEFAULT_INPUT_PRECISION = "FP16"
GEMMA4_MLP_PRODUCT_SCALE = 1.0 / 64.0

GENERATED_BUNDLE_METADATA_FILES = {
    "runtime_plan.json",
}

DTYPE_TO_PRECISION: dict[str, str] = {
    "torch.float16": "FP16",
    "torch.bfloat16": "FP16",
    "torch.float32": "FP32",
    "torch.float": "FP32",
    "torch.int8": "FP32",
    "torch.uint8": "FP32",
    "torch.int16": "FP32",
    "torch.int32": "FP32",
    "torch.int64": "FP32",
    "torch.long": "FP32",
    "torch.bool": "FP32",
}

WEIGHT_VALUE_KINDS = {
    "parameter",
    "buffer",
}

INPUT_VALUE_KINDS = {
    "unknown",
    "user_input",
    "lifted_constant",
    "cache_input",
    "cache_output",
    "cache_state",
}

FP16_RUNTIME_INPUTS = {
    "input_features",
    "encoder_hidden_states",
    "inputs_embeds",
    "pixel_values",
}

GEMMA_ADD_CLIPPED_COMPONENTS = {
    "audio_encoder",
    "vision_encoder",
    "lm_encoder_step",
    "lm_encoder_text_chunk",
    "lm_encoder_media_step",
    "lm_encoder_media_chunk",
    "decoder_step",
    "decoder_prefill_chunk",
}

STATIC_CACHE_OUTPUT_COMPONENTS = {
    "decoder_cross_kv",
}

PASS_THROUGH_TARGETS = {
    "aten._assert_tensor_metadata.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.detach.default",
}

COPY_TARGETS = {
    "aten.copy.default",
}

PAD_TARGETS = {
    "cactus.pad",
    "aten.constant_pad_nd.default",
}

CONSTANT_INPUT_TARGETS = {
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.empty_permuted.default",
    "aten.full.default",
    "aten.full_like.default",
    "aten.scalar_tensor.default",
}

BINARY_TARGETS: dict[str, str] = {
    "cactus.add": "add",
    "cactus.add_clipped": "add_clipped",
    "cactus.subtract": "subtract",
    "cactus.multiply": "multiply",
    "cactus.divide": "divide",
    "cactus.not_equal": "not_equal",
    "cactus.equal": "equal",
    "cactus.less": "less",
    "cactus.less_equal": "less_equal",
    "cactus.greater": "greater",
    "cactus.greater_equal": "greater_equal",
    "cactus.bitwise_and": "bitwise_and",
    "cactus.bitwise_or": "bitwise_or",
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "subtract",
    "aten.mul.Tensor": "multiply",
    "aten.div.Tensor": "divide",
    "aten.ne.Tensor": "not_equal",
    "aten.eq.Tensor": "equal",
    "aten.lt.Tensor": "less",
    "aten.le.Tensor": "less_equal",
    "aten.gt.Tensor": "greater",
    "aten.ge.Tensor": "greater_equal",
    "aten.bitwise_and.Tensor": "bitwise_and",
    "aten.bitwise_or.Tensor": "bitwise_or",
}

SCALAR_TARGETS: dict[str, tuple[str, str]] = {
    "cactus.scalar_add": ("scalar_add", "value"),
    "cactus.scalar_subtract": ("scalar_subtract", "value"),
    "cactus.scalar_multiply": ("scalar_multiply", "value"),
    "cactus.scalar_divide": ("scalar_divide", "value"),
    "cactus.scalar_floor_divide": ("scalar_floor_divide", "value"),
    "cactus.scalar_not_equal": ("scalar_not_equal", "value"),
    "cactus.scalar_equal": ("scalar_equal", "value"),
    "cactus.scalar_less": ("scalar_less", "value"),
    "cactus.scalar_less_equal": ("scalar_less_equal", "value"),
    "cactus.scalar_greater": ("scalar_greater", "value"),
    "cactus.scalar_greater_equal": ("scalar_greater_equal", "value"),
    "aten.add.Scalar": ("scalar_add", "other"),
    "aten.sub.Scalar": ("scalar_subtract", "other"),
    "aten.mul.Scalar": ("scalar_multiply", "other"),
    "aten.div.Scalar": ("scalar_divide", "other"),
    "aten.floor_divide.default": ("scalar_floor_divide", "other"),
    "aten.ne.Scalar": ("scalar_not_equal", "other"),
    "aten.eq.Scalar": ("scalar_equal", "other"),
    "aten.lt.Scalar": ("scalar_less", "other"),
    "aten.le.Scalar": ("scalar_less_equal", "other"),
    "aten.gt.Scalar": ("scalar_greater", "other"),
    "aten.ge.Scalar": ("scalar_greater_equal", "other"),
}

UNARY_TARGETS: dict[str, str] = {
    "cactus.abs": "abs",
    "cactus.scalar_exp": "scalar_exp",
    "cactus.scalar_sqrt": "scalar_sqrt",
    "cactus.scalar_rsqrt": "scalar_rsqrt",
    "cactus.scalar_cos": "scalar_cos",
    "cactus.scalar_sin": "scalar_sin",
    "cactus.scalar_log": "scalar_log",
    "cactus.relu": "relu",
    "cactus.silu": "silu",
    "cactus.gelu": "gelu",
    "cactus.gelu_erf": "gelu_erf",
    "cactus.sigmoid": "sigmoid",
    "cactus.tanh": "tanh",
    "cactus.logical_not": "logical_not",
    "cactus.bitwise_not": "bitwise_not",
    "aten.abs.default": "abs",
    "aten.exp.default": "scalar_exp",
    "aten.sqrt.default": "scalar_sqrt",
    "aten.rsqrt.default": "scalar_rsqrt",
    "aten.cos.default": "scalar_cos",
    "aten.sin.default": "scalar_sin",
    "aten.log.default": "scalar_log",
    "aten.relu.default": "relu",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
    "aten.logical_not.default": "logical_not",
    "aten.bitwise_not.default": "bitwise_not",
}

FP16_UNARY_METHODS = {
    "relu",
    "silu",
    "gelu",
    "gelu_erf",
    "sigmoid",
    "tanh",
}

LOG1P_TARGETS = {
    "aten.log1p.default",
}

POW_TARGETS = {
    "cactus.pow",
    "aten.pow.Tensor_Scalar",
}

REDUCE_TARGETS: dict[str, str] = {
    "cactus.mean": "mean",
    "cactus.sum": "sum",
    "cactus.variance": "variance",
    "cactus.min": "min",
    "cactus.max": "max",
    "aten.mean.dim": "mean",
    "aten.sum.dim_IntList": "sum",
    "aten.var.dim": "variance",
    "aten.var.correction": "variance",
    "aten.amax.default": "max",
    "aten.amin.default": "min",
    "aten.any.dim": "max",
}

SHAPE_TARGETS: dict[str, str] = {
    "cactus.view": "view",
    "cactus.reshape": "reshape",
    "aten.view.default": "view",
    "aten.reshape.default": "reshape",
}

EXPAND_TARGETS = {
    "cactus.expand",
    "aten.expand.default",
}

UNSQUEEZE_TARGETS = {
    "aten.unsqueeze.default",
    "aten.squeeze.dims",
}

FLATTEN_TARGETS = {
    "aten.flatten.using_ints",
}

REPEAT_TARGETS = {
    "aten.repeat.default",
}

TRANSPOSE_TARGETS = {
    "cactus.transpose",
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
}

SLICE_TARGETS = {
    "cactus.slice",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.narrow.default",
}

INDEX_TARGETS = {
    "cactus.index",
    "aten.index.Tensor",
}

WHERE_TARGETS = {
    "cactus.where",
    "aten.where.self",
}

MASKED_SCATTER_TARGETS = {
    "cactus.masked_scatter",
    "aten.masked_scatter.default",
}

UNFOLD_TARGETS = {
    "cactus.unfold",
    "aten.unfold.default",
}

CAT_TARGETS = {
    "cactus.cat",
    "aten.cat.default",
}

MATMUL_TARGETS = {
    "cactus.matmul",
    "aten.mm.default",
    "aten.matmul.default",
    "aten.bmm.default",
}

LINEAR_TARGETS = {
    "cactus.linear",
    "aten.linear.default",
}

ADDM_CONST_TARGETS = {
    "aten.addmm.default",
}

SPLIT_TARGETS = {
    "aten.split_with_sizes.default",
}

GETITEM_TARGETS = {
    "operator.getitem",
    "<built-in function getitem>",
}

TO_COPY_TARGETS = {
    "cactus.precision_cast",
    "aten._to_copy.default",
}

NEG_TARGETS = {
    "aten.neg.default",
}

CUMSUM_TARGETS = {
    "cactus.cumsum",
    "aten.cumsum.default",
}

SOFTMAX_TARGETS = {
    "cactus.softmax",
    "aten.softmax.int",
    "aten._softmax.default",
}

TOPK_TARGETS = {
    "cactus.topk",
    "aten.topk.default",
}

GATHER_TARGETS = {
    "cactus.gather",
    "aten.gather.default",
}

EMBEDDING_TARGETS = {
    "cactus.embedding_from_tensor",
    "aten.embedding.default",
}

CLAMP_TARGETS = {
    "cactus.clamp",
    "aten.clamp.default",
    "aten.clamp.Tensor",
}

NORM_TARGETS = {
    "cactus.rms_norm",
    "cactus.layernorm",
    "cactus.layer_norm",
    "cactus.groupnorm",
    "cactus.group_norm",
    "cactus.batchnorm",
    "cactus.batch_norm",
    "aten.native_layer_norm.default",
    "aten.layer_norm.default",
    "aten.native_group_norm.default",
    "aten.group_norm.default",
    "aten.native_batch_norm.default",
    "aten.batch_norm.default",
}

CONV_TARGETS = {
    "cactus.conv1d",
    "cactus.conv1d_causal",
    "cactus.conv1d_k3",
    "cactus.conv1d_k7s3",
    "cactus.conv1d_same_depthwise_k9",
    "cactus.conv1d_pointwise",
    "cactus.conv2d",
    "cactus.conv2d_depthwise_k3s2p1",
    "cactus.conv2d_k3s1p1",
    "cactus.conv2d_k3s2p1",
    "cactus.conv2d_pointwise_1x1",
    "aten.convolution.default",
    "aten.conv1d.default",
    "aten.conv2d.default",
}

ATTENTION_TARGETS = {
    "cactus.attention",
    "cactus.attention_cached",
    "aten.scaled_dot_product_attention.default",
}

CACHE_TARGETS = {
    "cactus.kv_cache_append",
    "cactus.kv_cache_state",
    "cactus.conv_cache_append",
    "cactus.conv_cache_initialize",
    "cactus.conv_cache_state",
    "cactus.recurrent_cache_state",
    "cactus.recurrent_cache_write",
}

MOE_TARGETS = {
    "cactus.moe_layer_gated",
    "cactus.moe_layer_ungated",
    "cactus.dense_mlp_tq_fused",
    "cactus.qkv_tq_fused",
    "cactus.projection_pair_tq_fused",
}

SPECIAL_CACTUS_TARGETS = {
    "cactus.rope",
    "cactus.rope_gptj",
    "cactus.gemma4_rope_table_lookup",
    "cactus.glu",
    "cactus.lstm_cell",
    "cactus.gated_deltanet_decode",
    "cactus.gated_deltanet_prefill",
    "cactus.lfm_short_conv_decode",
    "cactus.rel_pos_bias",
    "cactus.sample",
    "cactus.scatter_topk",
    "cactus.gaussian_topk",
    "cactus.altup_predict",
    "cactus.altup_correct",
    "cactus.stft",
    "cactus.rfft",
    "cactus.irfft",
    "cactus.spectrogram",
    "cactus.image_preprocess",
    "cactus.bilinear_interpolation",
}

UNSUPPORTED_SEMANTIC_TARGETS = {
    "aten.fmod.Scalar",
    "aten.slice_scatter.default",
}

OPEN_SLICE_END = 9_223_372_036_854_775_807
