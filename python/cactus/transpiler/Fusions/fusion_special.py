from __future__ import annotations

from . import edges as E
from . import models as M
from .fusion_builders import _cache_input, _graph, _input, _single_node_graph, _variadic_input


LSTM_CELL_GRAPH = _graph(
    "lstm_cell",
    "lstm_tanh",
    ("lstm_gate_mm", "lstm_recurrent_mm", "lstm_gate_add", "lstm_sigmoid", "lstm_tanh"),
    edge_names=E.EDGE_GROUPS["lstm_cell"],
    inputs=(_input("input", "lstm_gate_mm", 0), _input("h_prev", "lstm_recurrent_mm", 0), _input("weight_ih", "lstm_gate_mm", 1), _input("weight_hh", "lstm_recurrent_mm", 1)),
    constraints={
        "lstm_gate_structure": {"gate_node": "lstm_gate_add", "gate_count": 4, "require_explicit_gate_split": True},
    },
)


DELTANET_DECODE_GRAPH = _graph(
    "gated_deltanet_decode",
    "delta_gate",
    ("delta_q", "delta_k", "delta_v", "delta_gate"),
    inputs=(_input("query", "delta_q", 0), _input("key", "delta_k", 0), _input("value", "delta_v", 0), _input("gate_log", "delta_gate", 0), _input("beta", "delta_gate", 1), _input("initial_state", "delta_gate", 2)),
    attr_captures=(M.AttrCapture("scale", default=None, required=False),),
    cache_inputs=(_cache_input("initial_state", "delta_gate", 2, cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
)


DELTANET_PREFILL_GRAPH = _graph(
    "gated_deltanet_prefill",
    "delta_prefill",
    ("delta_q", "delta_k", "delta_v", "delta_gate", "delta_prefill"),
    inputs=(_input("query", "delta_q", 0), _input("key", "delta_k", 0), _input("value", "delta_v", 0), _input("gate_log", "delta_gate", 0), _input("beta", "delta_gate", 1), _input("initial_state", "delta_prefill", 2)),
    attr_captures=(M.AttrCapture("chunk_size", "delta_prefill", "chunk_size", required=False), M.AttrCapture("scale", default=None, required=False),),
    cache_inputs=(_cache_input("initial_state", "delta_prefill", 2, cache_kind=M.CacheKind.RECURRENT, tensor_role=M.CacheTensorRole.STATE),),
)


REL_POS_BIAS_GRAPH = _graph(
    "rel_pos_bias",
    "rel_pos_bias",
    ("rel_pos_bias",),
    inputs=(
        _input("query", "rel_pos_bias", 0, tensor_constraints=(M.TensorConstraint(rank=4),)),
        _input("relative_key", "rel_pos_bias", 1, tensor_constraints=(M.TensorConstraint(rank=4),)),
    ),
    attr_captures=(M.AttrCapture("scale", "rel_pos_bias", "scale", required=False),),
)


GEMMA4_ROPE_COS_TABLE_GRAPH = _single_node_graph(
    "gemma4_rope_cos_table",
    "scalar_cos",
    ("position_ids",),
    metadata={"special_matcher": "gemma4_rope_table_lookup", "table_kind": "cos"},
)


GEMMA4_ROPE_SIN_TABLE_GRAPH = _single_node_graph(
    "gemma4_rope_sin_table",
    "scalar_sin",
    ("position_ids",),
    metadata={"special_matcher": "gemma4_rope_table_lookup", "table_kind": "sin"},
)


SAMPLE_GRAPH = _graph(
    "sample",
    "sample_softmax",
    ("sample_topk", "sample_softmax"),
    edge_names=("sample_topk_to_softmax",),
    inputs=(_input("logits", "sample_topk", 0),),
    attr_captures=(M.AttrCapture("temperature", default=0.6, required=False), M.AttrCapture("top_p", default=0.95, required=False), M.AttrCapture("top_k", "sample_topk", "k", default=20, required=False)),
)


SCATTER_TOPK_GRAPH = _graph(
    "scatter_topk",
    "scatter",
    ("scatter",),
    inputs=(_input("indices", "scatter", 0), _input("values", "scatter", 1)),
    attr_captures=(M.AttrCapture("num_classes", "scatter", "num_classes", required=False),),
)


GAUSSIAN_TOPK_GRAPH = _graph(
    "gaussian_topk",
    "gaussian_topk",
    ("gaussian_topk",),
    inputs=(_input("x", "gaussian_topk", 0),),
    attr_captures=(M.AttrCapture("ppf", "gaussian_topk", "ppf", required=False),),
)


ALTUP_PREDICT_GRAPH = _graph(
    "altup_predict",
    "altup_predict",
    ("altup_predict",),
    inputs=(_input("coefs", "altup_predict", 0), _variadic_input("streams", "altup_predict", 1)),
)


ALTUP_CORRECT_GRAPH = _graph(
    "altup_correct",
    "altup_correct",
    ("altup_correct",),
    inputs=(_input("coefs", "altup_correct", 0), _input("innovation", "altup_correct", 1), _variadic_input("predictions", "altup_correct", 2)),
)


DSP_GRAPHS: dict[str, M.FusionGraph] = {
    "stft": _graph("stft", "stft", ("stft",), inputs=(_input("x", "stft", 0), _input("weight", "stft", 1)), attr_captures=(M.AttrCapture("stride", "stft", "stride", required=False), M.AttrCapture("num_fft_bins", "stft", "num_fft_bins", required=False))),
    "rfft": _graph("rfft", "rfft", ("rfft",), inputs=(_input("x", "rfft", 0),)),
    "irfft": _graph("irfft", "irfft", ("irfft",), inputs=(_input("x", "irfft", 0),), attr_captures=(M.AttrCapture("output_length", "irfft", "output_length", required=False),)),
    "spectrogram": _graph("spectrogram", "spectrogram", ("spectrogram",), inputs=(_input("waveform", "spectrogram", 0), _input("mel_filters", "spectrogram", 1)), metadata={"requires_preprocessing_attrs": True}),
    "image_preprocess": _graph("image_preprocess", "image_preprocess", ("image_preprocess",), inputs=(_input("pixel_input", "image_preprocess", 0),), metadata={"requires_preprocessing_attrs": True}),
    "bilinear_interpolation": _graph("bilinear_interpolation", "bilinear_interpolation", ("bilinear_interpolation",), inputs=(_input("pos_embeds", "bilinear_interpolation", 0),), attr_captures=(M.AttrCapture("dst_height", "bilinear_interpolation", "dst_height", required=False), M.AttrCapture("dst_width", "bilinear_interpolation", "dst_width", required=False))),
}
