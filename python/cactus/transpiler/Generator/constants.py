CACTUS_TARGET_PREFIX = "cactus."
DEFAULT_COMPONENT_NAME = "model"
DEFAULT_GRAPH_SUFFIX = ".cactus"
DEFAULT_INPUT_PRECISION = "FP16"

DTYPE_TO_PRECISION: dict[str, str] = {
    "torch.float16": "FP16",
    "torch.bfloat16": "FP16",
    "torch.float32": "FP32",
    "torch.float": "FP32",
    "torch.int8": "INT8",
    "torch.uint8": "INT8",
    "torch.int16": "INT8",
    "torch.int32": "INT8",
    "torch.int64": "INT8",
    "torch.long": "INT8",
    "torch.bool": "INT8",
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

PASS_THROUGH_TARGETS = {
    "aten._assert_tensor_metadata.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.detach.default",
}

COPY_TARGETS = {
    "aten.copy.default",
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
    "cactus.subtract": "subtract",
    "cactus.multiply": "multiply",
    "cactus.divide": "divide",
    "cactus.not_equal": "not_equal",
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "subtract",
    "aten.mul.Tensor": "multiply",
    "aten.div.Tensor": "divide",
    "aten.ne.Tensor": "not_equal",
}

SCALAR_TARGETS: dict[str, tuple[str, str]] = {
    "cactus.scalar_add": ("scalar_add", "value"),
    "cactus.scalar_subtract": ("scalar_subtract", "value"),
    "cactus.scalar_multiply": ("scalar_multiply", "value"),
    "cactus.scalar_divide": ("scalar_divide", "value"),
    "cactus.scalar_floor_divide": ("scalar_floor_divide", "value"),
    "cactus.scalar_not_equal": ("scalar_not_equal", "value"),
    "aten.add.Scalar": ("scalar_add", "other"),
    "aten.sub.Scalar": ("scalar_subtract", "other"),
    "aten.mul.Scalar": ("scalar_multiply", "other"),
    "aten.div.Scalar": ("scalar_divide", "other"),
    "aten.floor_divide.default": ("scalar_floor_divide", "other"),
    "aten.ne.Scalar": ("scalar_not_equal", "other"),
}

UNARY_TARGETS: dict[str, str] = {
    "cactus.abs": "abs",
    "cactus.scalar_exp": "scalar_exp",
    "cactus.scalar_sqrt": "scalar_sqrt",
    "cactus.scalar_cos": "scalar_cos",
    "cactus.scalar_sin": "scalar_sin",
    "cactus.scalar_log": "scalar_log",
    "cactus.relu": "relu",
    "cactus.silu": "silu",
    "cactus.gelu": "gelu",
    "cactus.gelu_erf": "gelu_erf",
    "cactus.sigmoid": "sigmoid",
    "cactus.tanh": "tanh",
    "aten.abs.default": "abs",
    "aten.exp.default": "scalar_exp",
    "aten.sqrt.default": "scalar_sqrt",
    "aten.cos.default": "scalar_cos",
    "aten.sin.default": "scalar_sin",
    "aten.log.default": "scalar_log",
    "aten.relu.default": "relu",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
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
}

SHAPE_TARGETS: dict[str, str] = {
    "cactus.view": "view",
    "cactus.reshape": "reshape",
    "aten.view.default": "view",
    "aten.reshape.default": "reshape",
}

EXPAND_TARGETS = {
    "aten.expand.default",
}

UNSQUEEZE_TARGETS = {
    "aten.unsqueeze.default",
}

FLATTEN_TARGETS = {
    "aten.flatten.using_ints",
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
    "aten._to_copy.default",
}

NEG_TARGETS = {
    "aten.neg.default",
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
}

SPECIAL_CACTUS_TARGETS = {
    "cactus.rope",
    "cactus.rope_gptj",
    "cactus.glu",
    "cactus.lstm_cell",
    "cactus.gated_deltanet_decode",
    "cactus.gated_deltanet_prefill",
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
    "aten.any.dim",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.constant_pad_nd.default",
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.fmod.Scalar",
    "aten.index.Tensor",
    "aten.le.Tensor",
    "aten.logical_not.default",
    "aten.slice_scatter.default",
    "aten.where.self",
}

OPEN_SLICE_END = 9_223_372_036_854_775_807
