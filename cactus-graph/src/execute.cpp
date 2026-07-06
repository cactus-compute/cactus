#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

static int g_backend_override = -1;
int cactus_backend_select(const char* backend) {
    if (!backend) return -1;
    if (std::strcmp(backend, "auto") == 0) { g_backend_override = -1; return 0; }
    if (std::strcmp(backend, "metal") == 0) {
        if (!cactus_metal_available()) return -1;
        g_backend_override = 1;
        return 0;
    }
    if (std::strcmp(backend, "cpu") == 0) { g_backend_override = 0; return 0; }
    return -1;
}
bool cactus_backend_metal() { return g_backend_override != 0 && cactus_metal_available(); }

using ComputeFn = void(*)(GraphNode&, const nodes_vector&, const node_index_map_t&);

#define DECLARE_COMPUTE(name) \
    extern void name(GraphNode&, const nodes_vector&, const node_index_map_t&)

DECLARE_COMPUTE(compute_binary_op_node);
DECLARE_COMPUTE(compute_unary_op_node);
DECLARE_COMPUTE(compute_activation_node);
DECLARE_COMPUTE(compute_reduce_node);
DECLARE_COMPUTE(compute_reshape_node);
DECLARE_COMPUTE(compute_precision_cast_node);
DECLARE_COMPUTE(compute_matmul_node);
DECLARE_COMPUTE(compute_rms_norm_node);
DECLARE_COMPUTE(compute_rope_node);
DECLARE_COMPUTE(compute_softmax_node);
DECLARE_COMPUTE(compute_attention_node);
DECLARE_COMPUTE(compute_attention_int8_hybrid_node);
DECLARE_COMPUTE(compute_rel_pos_bias_node);
DECLARE_COMPUTE(compute_layernorm_node);
DECLARE_COMPUTE(compute_conv1d_causal_node);
DECLARE_COMPUTE(compute_conv1d_k3_node);
DECLARE_COMPUTE(compute_conv1d_k7s3_node);
DECLARE_COMPUTE(compute_conv1d_node);
DECLARE_COMPUTE(compute_conv1d_same_depthwise_k9_node);
DECLARE_COMPUTE(compute_conv1d_pointwise_node);
DECLARE_COMPUTE(compute_conv2d_k3s2p1_node);
DECLARE_COMPUTE(compute_conv2d_depthwise_k3s2p1_node);
DECLARE_COMPUTE(compute_conv2d_pointwise_1x1_node);
DECLARE_COMPUTE(compute_glu_node);
DECLARE_COMPUTE(compute_batchnorm_node);
DECLARE_COMPUTE(compute_groupnorm_node);
DECLARE_COMPUTE(compute_rope_gptj_node);
DECLARE_COMPUTE(compute_lstm_cell_node);
DECLARE_COMPUTE(compute_gated_deltanet_decode_node);
DECLARE_COMPUTE(compute_gated_deltanet_prefill_node);
DECLARE_COMPUTE(compute_stft_node);
DECLARE_COMPUTE(compute_altup_predict_node);
DECLARE_COMPUTE(compute_altup_correct_node);
DECLARE_COMPUTE(compute_gaussian_topk_node);
DECLARE_COMPUTE(compute_maxpool1d_node);
DECLARE_COMPUTE(compute_bilstm_sequence_node);
DECLARE_COMPUTE(compute_conv2d_k3s1p1_node);
DECLARE_COMPUTE(compute_stats_pool_node);
DECLARE_COMPUTE(compute_weighted_stats_pool_node);
DECLARE_COMPUTE(compute_transpose_node);
DECLARE_COMPUTE(compute_gather_node);
DECLARE_COMPUTE(compute_slice_node);
DECLARE_COMPUTE(compute_embedding_node);
DECLARE_COMPUTE(compute_concat_node);
DECLARE_COMPUTE(compute_cat_node);
DECLARE_COMPUTE(compute_index_node);
DECLARE_COMPUTE(compute_bilinear_interpolation_node);
DECLARE_COMPUTE(compute_sample_node);
DECLARE_COMPUTE(compute_topk_node);
DECLARE_COMPUTE(compute_scatter_topk_node);
DECLARE_COMPUTE(compute_moe_layer_node);
DECLARE_COMPUTE(compute_dense_mlp_tq_fused_node);
DECLARE_COMPUTE(compute_persistent_node);
DECLARE_COMPUTE(compute_kv_cache_state_node);
DECLARE_COMPUTE(compute_kv_cache_append_node);
DECLARE_COMPUTE(compute_attention_cached_node);
DECLARE_COMPUTE(compute_conv_cache_state_node);
DECLARE_COMPUTE(compute_conv_cache_append_node);
DECLARE_COMPUTE(compute_recurrent_cache_state_node);
DECLARE_COMPUTE(compute_recurrent_cache_write_node);
DECLARE_COMPUTE(compute_conv_cache_initialize_node);
DECLARE_COMPUTE(compute_image_preprocess_node);
DECLARE_COMPUTE(compute_rfft_node);
DECLARE_COMPUTE(compute_irfft_node);
DECLARE_COMPUTE(compute_mel_filter_bank_node);
DECLARE_COMPUTE(compute_spectrogram_node);
extern void shrink_thread_local_buffers();
#undef DECLARE_COMPUTE

static constexpr int OP_TYPE_COUNT = static_cast<int>(OpType::CONV_CACHE_INITIALIZE) + 1;
static_assert(OP_TYPE_COUNT <= 256, "OpType dispatch table overflow");
static ComputeFn dispatch_flat[OP_TYPE_COUNT] = {};

static bool init_dispatch() {
    dispatch_flat[static_cast<int>(OpType::ADD)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::ADD_CLIPPED)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::SUBTRACT)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::MULTIPLY)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::DIVIDE)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::NOT_EQUAL)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_ADD)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_SUBTRACT)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_MULTIPLY)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_DIVIDE)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_NOT_EQUAL)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_EXP)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_SQRT)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_COS)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_SIN)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_LOG)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::ABS)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::POW)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::RELU)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::SILU)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::GELU)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::GELU_ERF)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::SIGMOID)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::TANH)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::LEAKY_RELU)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::CLAMP)] = compute_activation_node;
    dispatch_flat[static_cast<int>(OpType::SUM)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::MEAN)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::VARIANCE)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::MIN)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::MAX)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::CUMSUM)] = compute_reduce_node;
    dispatch_flat[static_cast<int>(OpType::FLATTEN)] = compute_reshape_node;
    dispatch_flat[static_cast<int>(OpType::VIEW)] = compute_reshape_node;
    dispatch_flat[static_cast<int>(OpType::RESHAPE)] = compute_reshape_node;
    dispatch_flat[static_cast<int>(OpType::PRECISION_CAST)] = compute_precision_cast_node;
    dispatch_flat[static_cast<int>(OpType::MATMUL)] = compute_matmul_node;
    dispatch_flat[static_cast<int>(OpType::RMS_NORM)] = compute_rms_norm_node;
    dispatch_flat[static_cast<int>(OpType::LAYERNORM)] = compute_layernorm_node;
    dispatch_flat[static_cast<int>(OpType::GROUPNORM)] = compute_groupnorm_node;
    dispatch_flat[static_cast<int>(OpType::BATCHNORM)] = compute_batchnorm_node;
    dispatch_flat[static_cast<int>(OpType::ROPE)] = compute_rope_node;
    dispatch_flat[static_cast<int>(OpType::ROPE_GPTJ)] = compute_rope_gptj_node;
    dispatch_flat[static_cast<int>(OpType::SOFTMAX)] = compute_softmax_node;
    dispatch_flat[static_cast<int>(OpType::ATTENTION)] = compute_attention_node;
    dispatch_flat[static_cast<int>(OpType::ATTENTION_INT8_HYBRID)] = compute_attention_int8_hybrid_node;
    dispatch_flat[static_cast<int>(OpType::REL_POS_BIAS)] = compute_rel_pos_bias_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D_CAUSAL)] = compute_conv1d_causal_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D_K3)] = compute_conv1d_k3_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D_K7S3)] = compute_conv1d_k7s3_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D)] = compute_conv1d_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D_SAME_DEPTHWISE_K9)] = compute_conv1d_same_depthwise_k9_node;
    dispatch_flat[static_cast<int>(OpType::CONV1D_POINTWISE)] = compute_conv1d_pointwise_node;
    dispatch_flat[static_cast<int>(OpType::CONV2D_K3S2P1)] = compute_conv2d_k3s2p1_node;
    dispatch_flat[static_cast<int>(OpType::CONV2D_DEPTHWISE_K3S2P1)] = compute_conv2d_depthwise_k3s2p1_node;
    dispatch_flat[static_cast<int>(OpType::CONV2D_POINTWISE_1X1)] = compute_conv2d_pointwise_1x1_node;
    dispatch_flat[static_cast<int>(OpType::CONV2D_K3S1P1)] = compute_conv2d_k3s1p1_node;
    dispatch_flat[static_cast<int>(OpType::GLU)] = compute_glu_node;
    dispatch_flat[static_cast<int>(OpType::TRANSPOSE)] = compute_transpose_node;
    dispatch_flat[static_cast<int>(OpType::GATHER)] = compute_gather_node;
    dispatch_flat[static_cast<int>(OpType::SLICE)] = compute_slice_node;
    dispatch_flat[static_cast<int>(OpType::EMBEDDING)] = compute_embedding_node;
    dispatch_flat[static_cast<int>(OpType::CONCAT)] = compute_concat_node;
    dispatch_flat[static_cast<int>(OpType::CAT)] = compute_cat_node;
    dispatch_flat[static_cast<int>(OpType::INDEX)] = compute_index_node;
    dispatch_flat[static_cast<int>(OpType::BILINEAR_INTERPOLATION)] = compute_bilinear_interpolation_node;
    dispatch_flat[static_cast<int>(OpType::SAMPLE)] = compute_sample_node;
    dispatch_flat[static_cast<int>(OpType::TOPK)] = compute_topk_node;
    dispatch_flat[static_cast<int>(OpType::SCATTER_TOPK)] = compute_scatter_topk_node;
    dispatch_flat[static_cast<int>(OpType::MOE_LAYER)] = compute_moe_layer_node;
    dispatch_flat[static_cast<int>(OpType::DENSE_MLP_TQ_FUSED)] = compute_dense_mlp_tq_fused_node;
    dispatch_flat[static_cast<int>(OpType::PERSISTENT)] = compute_persistent_node;
    dispatch_flat[static_cast<int>(OpType::LSTM_CELL)] = compute_lstm_cell_node;
    dispatch_flat[static_cast<int>(OpType::GATED_DELTANET_DECODE)] = compute_gated_deltanet_decode_node;
    dispatch_flat[static_cast<int>(OpType::GATED_DELTANET_PREFILL)] = compute_gated_deltanet_prefill_node;
    dispatch_flat[static_cast<int>(OpType::STFT)] = compute_stft_node;
    dispatch_flat[static_cast<int>(OpType::ALTUP_PREDICT)] = compute_altup_predict_node;
    dispatch_flat[static_cast<int>(OpType::ALTUP_CORRECT)] = compute_altup_correct_node;
    dispatch_flat[static_cast<int>(OpType::GAUSSIAN_TOPK)] = compute_gaussian_topk_node;
    dispatch_flat[static_cast<int>(OpType::MAXPOOL1D)] = compute_maxpool1d_node;
    dispatch_flat[static_cast<int>(OpType::BILSTM_SEQUENCE)] = compute_bilstm_sequence_node;
    dispatch_flat[static_cast<int>(OpType::STATS_POOL)] = compute_stats_pool_node;
    dispatch_flat[static_cast<int>(OpType::WEIGHTED_STATS_POOL)] = compute_weighted_stats_pool_node;
    dispatch_flat[static_cast<int>(OpType::KV_CACHE_STATE)] = compute_kv_cache_state_node;
    dispatch_flat[static_cast<int>(OpType::KV_CACHE_APPEND)] = compute_kv_cache_append_node;
    dispatch_flat[static_cast<int>(OpType::ATTENTION_CACHED)] = compute_attention_cached_node;
    dispatch_flat[static_cast<int>(OpType::CONV_CACHE_STATE)] = compute_conv_cache_state_node;
    dispatch_flat[static_cast<int>(OpType::CONV_CACHE_APPEND)] = compute_conv_cache_append_node;
    dispatch_flat[static_cast<int>(OpType::RECURRENT_CACHE_STATE)] = compute_recurrent_cache_state_node;
    dispatch_flat[static_cast<int>(OpType::RECURRENT_CACHE_WRITE)] = compute_recurrent_cache_write_node;
    dispatch_flat[static_cast<int>(OpType::CONV_CACHE_INITIALIZE)] = compute_conv_cache_initialize_node;
    dispatch_flat[static_cast<int>(OpType::IMAGE_PREPROCESS)] = compute_image_preprocess_node;
    dispatch_flat[static_cast<int>(OpType::RFFT)] = compute_rfft_node;
    dispatch_flat[static_cast<int>(OpType::IRFFT)] = compute_irfft_node;
    dispatch_flat[static_cast<int>(OpType::MEL_FILTER_BANK)] = compute_mel_filter_bank_node;
    dispatch_flat[static_cast<int>(OpType::SPECTROGRAM)] = compute_spectrogram_node;
    return true;
}

static const bool dispatch_initialized = init_dispatch();

static inline void dispatch_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    int op = static_cast<int>(node.op_type);
    ComputeFn fn = dispatch_flat[op];
    if (fn) {
        fn(node, nodes, node_index_map);
    } else {
        throw std::runtime_error("Unknown operation type: " + std::to_string(op));
    }
}

static const char* op_type_names[] = {
    "INPUT", "PRECISION_CAST",
    "ADD", "ADD_CLIPPED", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "ABS", "POW", "FLATTEN", "VIEW",
    "MATMUL", "TRANSPOSE", "RESHAPE", "SLICE", "GATHER", "EMBEDDING",
    "BILINEAR_INTERPOLATION",
    "SUM", "MEAN", "VARIANCE", "MIN", "MAX", "CUMSUM",
    "RMS_NORM", "ROPE", "ROPE_GPTJ", "SOFTMAX",
    "ATTENTION", "ATTENTION_INT8_HYBRID", "REL_POS_BIAS",
    "CONV1D_CAUSAL", "CONV1D_K3", "CONV1D_K7S3", "CONV1D",
    "CONV1D_SAME_DEPTHWISE_K9", "CONV1D_POINTWISE",
    "CONV2D_K3S2P1", "CONV2D_DEPTHWISE_K3S2P1", "CONV2D_POINTWISE_1X1",
    "GLU", "BATCHNORM",
    "SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY", "SCALAR_DIVIDE",
    "SCALAR_EXP", "SCALAR_SQRT", "SCALAR_COS", "SCALAR_SIN", "SCALAR_LOG",
    "RELU", "SILU", "GELU", "GELU_ERF", "SIGMOID", "TANH",
    "SAMPLE", "CONCAT", "CAT",
    "SCATTER_TOPK", "TOPK", "LAYERNORM", "GROUPNORM",
    "MOE_LAYER", "INDEX", "PERSISTENT",
    "LSTM_CELL", "GATED_DELTANET_DECODE", "GATED_DELTANET_PREFILL",
    "STFT", "ALTUP_PREDICT", "ALTUP_CORRECT", "GAUSSIAN_TOPK",
    "MAXPOOL1D", "BILSTM_SEQUENCE", "LEAKY_RELU",
    "CONV2D_K3S1P1", "STATS_POOL", "WEIGHTED_STATS_POOL",
    "KV_CACHE_STATE", "KV_CACHE_APPEND", "ATTENTION_CACHED",
    "CONV_CACHE_STATE", "CONV_CACHE_APPEND",
    "RFFT", "IRFFT", "MEL_FILTER_BANK", "SPECTROGRAM",
    "IMAGE_PREPROCESS", "CLAMP", "DENSE_MLP_TQ_FUSED",
    "NOT_EQUAL", "SCALAR_NOT_EQUAL",
    "RECURRENT_CACHE_STATE",
    "RECURRENT_CACHE_WRITE",
    "CONV_CACHE_INITIALIZE"
};

static const char* get_op_name(OpType op) {
    return op_type_names[static_cast<int>(op)];
}

void compute_node_optimized(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    if (node.op_type == OpType::INPUT) return;
    dispatch_node(node, nodes, node_index_map);
}

void CactusGraph::set_input(size_t node_id, const void* data, Precision) {
    auto it = node_index_map_.find(node_id);
    if (it == node_index_map_.end()) {
        throw std::out_of_range("Unknown input node id: " + std::to_string(node_id));
    }

    auto& node = *nodes_[it->second];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only set data on input nodes");
    }

    if (!node.output_buffer.data && !node.output_buffer.external_data) {
        node.output_buffer.allocate();
    }

    if (node.output_buffer.external_data) {
        node.output_buffer.external_data = nullptr;
        node.output_buffer.allocate();
    }

    std::memcpy(node.output_buffer.get_data(), data, node.output_buffer.byte_size);
}

void CactusGraph::set_external_input(size_t node_id, void* data, Precision) {
    auto it = node_index_map_.find(node_id);
    if (it == node_index_map_.end()) {
        throw std::out_of_range("Unknown input node id: " + std::to_string(node_id));
    }

    auto& node = *nodes_[it->second];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only set data on input nodes");
    }

    node.output_buffer.set_external(data);
    embedded_input_node_ids_.erase(node_id);
}

void* CactusGraph::get_output(size_t node_id) {
    auto it = node_index_map_.find(node_id);
    if (it == node_index_map_.end()) {
        throw std::out_of_range("Unknown output node id: " + std::to_string(node_id));
    }

    auto& buffer = nodes_[it->second]->output_buffer;
    if (!buffer.get_data()) {
        buffer.allocate();
    }
    return buffer.get_data();
}

static bool check_debug_env() {
    const char* v1 = std::getenv("CACTUS_CAPTURE_ENABLE");
    const char* v2 = std::getenv("CACTUS_CAPTURE_STDOUT");
    const char* v3 = std::getenv("CACTUS_CAPTURE_FILE");
    const char* v4 = std::getenv("CACTUS_CAPTURE_DIR");
    const char* v5 = std::getenv("CACTUS_PROFILE_FILE");
    const char* v6 = std::getenv("CACTUS_PROFILE");
    return (v1 && v1[0] != '0') || (v2 && v2[0] != '0') ||
           (v3 && v3[0]) || (v4 && v4[0]) || (v5 && v5[0]) || (v6 && v6[0]);
}

namespace {
std::vector<size_t> infer_output_shape(const GraphNode& node, const nodes_vector& nodes, const node_index_map_t& idx) {
    auto in = [&](size_t i) -> const std::vector<size_t>& { return get_input(node, i, nodes, idx).shape; };
    switch (node.op_type) {
        case OpType::MATMUL: {
            const auto& lhs = in(0);
            const auto& rhs = in(1);
            std::vector<size_t> out = lhs;
            out.back() = node.params.pretransposed_rhs ? rhs[rhs.size() - 2] : rhs[rhs.size() - 1];
            return out;
        }
        case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
        case OpType::MULTIPLY: case OpType::DIVIDE: case OpType::NOT_EQUAL:
            return BroadcastInfo::compute(in(0), in(1)).output_shape;
        case OpType::ATTENTION: case OpType::ATTENTION_CACHED: case OpType::ATTENTION_INT8_HYBRID: {
            std::vector<size_t> out = in(0);
            if (node.params.v_head_dim > 0) out.back() = node.params.v_head_dim;
            return out;
        }
        case OpType::TRANSPOSE: {
            const auto& x = in(0);
            const auto& perm = node.params.permutation;
            if (perm.size() != x.size()) return x;
            std::vector<size_t> out(x.size());
            for (size_t i = 0; i < perm.size(); ++i) out[i] = x[perm[i]];
            return out;
        }
        case OpType::RESHAPE: case OpType::VIEW: case OpType::FLATTEN: {
            const auto& x = in(0);
            std::vector<size_t> out = node.params.new_shape;
            if (out.empty()) return x;
            size_t in_total = 1;
            for (size_t d : x) in_total *= d;
            size_t rest = 1;
            for (size_t i = 1; i < out.size(); ++i) rest *= out[i];
            if (rest > 0 && in_total % rest == 0) out[0] = in_total / rest;
            return out;
        }
        case OpType::CONCAT: case OpType::CAT: {
            std::vector<size_t> out = in(0);
            size_t axis = node.params.axis < 0 ? out.size() + static_cast<size_t>(node.params.axis)
                                               : static_cast<size_t>(node.params.axis);
            size_t sum = 0;
            for (size_t i = 0; i < node.input_ids.size(); ++i) sum += in(i)[axis];
            out[axis] = sum;
            return out;
        }
        case OpType::SLICE: {
            std::vector<size_t> out = in(0);
            size_t axis = node.params.axis < 0 ? out.size() + static_cast<size_t>(node.params.axis)
                                               : static_cast<size_t>(node.params.axis);
            out[axis] = node.params.slice_length;
            return out;
        }
        default: {
            std::vector<size_t> out = node.output_buffer.shape;
            if (out.empty()) return in(0);
            for (size_t i = 0; i < node.input_ids.size(); ++i) {
                const auto& inp = get_input(node, i, nodes, idx);
                if (inp.has_dynamic_dims() && !inp.shape.empty()) {
                    out[0] = inp.shape[0];
                    return out;
                }
            }
            return out;
        }
    }
}

bool skip_shape_infer(OpType op) {
    switch (op) {
        case OpType::KV_CACHE_STATE: case OpType::KV_CACHE_APPEND:
        case OpType::CONV_CACHE_STATE: case OpType::CONV_CACHE_APPEND: case OpType::CONV_CACHE_INITIALIZE:
        case OpType::RECURRENT_CACHE_STATE: case OpType::RECURRENT_CACHE_WRITE: case OpType::PERSISTENT:
            return true;
        default:
            return false;
    }
}
}

void CactusGraph::set_runtime_input_shape(size_t node_id, const std::vector<size_t>& shape) {
    GraphNode& node = *nodes_[node_index_map_.at(node_id)];
    node.output_buffer.set_shape(shape);
    node.output_buffer.dynamic_dims.assign(shape.size(), 1);
    node.output_buffer.data.reset();
    has_dynamic_shapes_ = true;
    runtime_shapes_dirty_ = true;
}

void CactusGraph::set_input_dynamic_dims(size_t node_id, const std::vector<uint8_t>& dynamic_dims) {
    GraphNode& node = *nodes_[node_index_map_.at(node_id)];
    node.output_buffer.dynamic_dims = dynamic_dims;
    if (!dynamic_dims.empty()) has_dynamic_shapes_ = true;
}

void CactusGraph::infer_shapes() {
    if (!has_dynamic_shapes_ || !runtime_shapes_dirty_) return;
    for (auto& np : nodes_) {
        GraphNode& node = *np;
        if (node.op_type == OpType::INPUT || persistent_node_ids_.count(node.id) || skip_shape_infer(node.op_type)) continue;
        bool dyn = false;
        for (size_t i = 0; i < node.input_ids.size() && !dyn; ++i) {
            dyn = get_input(node, i, nodes_, node_index_map_).has_dynamic_dims();
        }
        if (!dyn) continue;
        node.output_buffer.set_shape(infer_output_shape(node, nodes_, node_index_map_));
        switch (node.op_type) {
            case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
            case OpType::MULTIPLY: case OpType::DIVIDE: case OpType::NOT_EQUAL:
                node.params.broadcast_info = BroadcastInfo::compute(
                    get_input(node, 0, nodes_, node_index_map_).shape,
                    get_input(node, 1, nodes_, node_index_map_).shape);
                break;
            default:
                break;
        }
        node.output_buffer.dynamic_dims.assign(node.output_buffer.shape.size(), 1);
    }
    runtime_shapes_dirty_ = false;
}

static void row_strides(const std::vector<size_t>& shape, size_t* out) {
    size_t s=1; for (int k=(int)shape.size()-1; k>=0; --k){ out[k]=s; s*=shape[k]; }
}
static void bcast_strides(const std::vector<size_t>& in_shape, const std::vector<size_t>& out_shape, uint32_t* out) {
    size_t off = out_shape.size() - in_shape.size();
    size_t istr[8]; row_strides(in_shape, istr);
    for (size_t d=0; d<out_shape.size(); ++d)
        out[d] = (d < off) ? 0u : (in_shape[d-off]==1 ? 0u : (uint32_t)istr[d-off]);
}

bool cactus_kv_cache_grow(BufferDesc&, size_t, size_t);
std::vector<__fp16> dequantize_int8_weights_to_fp16(const BufferDesc& W, size_t rows, size_t cols,
                                                    const char* op_name);

static const __fp16* metal_conv_weight_f16(const BufferDesc& w, size_t rows, size_t cols) {
    if (w.precision == Precision::FP16) return w.data_as<__fp16>();
    if (w.precision != Precision::INT8) return nullptr;
    const int8_t* p = w.data_as<int8_t>();
    size_t n = rows * cols;
    if (!p || n == 0 || w.total_size < n) return nullptr;
    uint64_t h = 1469598103934665603ull;
    size_t take = n < 64 ? n : 64;
    for (size_t i = 0; i < take; ++i) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; }
    for (size_t i = n - take; i < n; ++i) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; }
    h ^= n; h *= 1099511628211ull;
    struct Entry { std::vector<__fp16> data; uint64_t fp; };
    static std::mutex mu;
    static std::unordered_map<const void*, Entry> cache;
    std::lock_guard<std::mutex> lk(mu);
    Entry& e = cache[p];
    if (e.data.size() != n || e.fp != h) {
        e.data = dequantize_int8_weights_to_fp16(w, rows, cols, "conv_metal");
        e.fp = h;
    }
    return e.data.data();
}

static bool try_encode_metal(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& map) {
    BufferDesc& out = node.output_buffer;
    auto fp16 = [](const BufferDesc& b){ return b.precision == Precision::FP16; };
    switch (node.op_type) {
        case OpType::MATMUL: {
            const auto& lhs = get_input(node, 0, nodes, map);
            const auto& rhs = get_input(node, 1, nodes, map);
            size_t M = lhs.shape[lhs.shape.size() - 2];
            if (!fp16(lhs)) return false;
            if (PrecisionTraits::is_cq(rhs.precision) && rhs.group_size > 0) {
                CactusQuantMatrix mat = rhs.to_cq_matrix();
                if (M == 1 && !cactus_graph_prefill_consistent())
                    return cactus_metal_encode_quant_matmul(out.get_data(), lhs.get_data(), &mat);
                return cactus_metal_encode_quant_matmul_m(out.get_data(), lhs.get_data(), &mat, (uint32_t)M);
            }
            if (!fp16(rhs) || !fp16(out) || rhs.shape.size() != 2) return false;
            size_t K = lhs.shape.back();
            size_t N = node.params.pretransposed_rhs ? rhs.shape[0] : rhs.shape[1];
            size_t rhs_k = node.params.pretransposed_rhs ? rhs.shape[1] : rhs.shape[0];
            if (K == 0 || rhs_k != K || lhs.total_size % K != 0) return false;
            size_t rows = lhs.total_size / K;
            if (rows != M) {
                size_t batch = rows / M;
                if (batch * M != rows) return false;
            }
            return cactus_metal_encode_gemm_f16(out.get_data(), lhs.get_data(), rhs.get_data(),
                (uint32_t)rows, (uint32_t)K, (uint32_t)N, node.params.pretransposed_rhs ? 1 : 0);
        }
        case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
        case OpType::MULTIPLY: case OpType::DIVIDE: case OpType::NOT_EQUAL: {
            const auto& a = get_input(node, 0, nodes, map);
            const auto& b = get_input(node, 1, nodes, map);
            int code = node.op_type==OpType::ADD?0: node.op_type==OpType::ADD_CLIPPED?1:
                       node.op_type==OpType::SUBTRACT?2: node.op_type==OpType::MULTIPLY?3:
                       node.op_type==OpType::DIVIDE?4:5;
            bool f32 = a.precision == Precision::FP32 && b.precision == Precision::FP32
                    && out.precision == Precision::FP32;
            if (f32 && a.total_size == out.total_size && b.total_size == out.total_size)
                return cactus_metal_encode_binary_f32(code, out.get_data(), a.get_data(), b.get_data(), out.total_size);
            if (!fp16(a) || !fp16(b) || !fp16(out)) return false;
            if (a.total_size == out.total_size && b.total_size == out.total_size)
                return cactus_metal_encode_binary(code, out.get_data(), a.get_data(), b.get_data(), out.total_size);
            const auto& osh = out.shape;
            uint32_t nd = (uint32_t)osh.size();
            if (nd == 0 || nd > 8 || a.shape.size() > nd || b.shape.size() > nd) return false;
            uint32_t oshape[8], astr[8], bstr[8];
            for (uint32_t d=0; d<nd; ++d) oshape[d]=(uint32_t)osh[d];
            bcast_strides(a.shape, osh, astr); bcast_strides(b.shape, osh, bstr);
            return cactus_metal_encode_bcast_binary(code, out.get_data(), a.get_data(), b.get_data(),
                oshape, astr, bstr, nd, (uint32_t)out.total_size, a.byte_size, b.byte_size, out.byte_size);
        }
        case OpType::TRANSPOSE: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out)) return false;
            const auto& perm = node.params.permutation;
            uint32_t nd = (uint32_t)perm.size();
            if (nd == 0 || nd > 8 || in.shape.size() != nd || out.shape.size() != nd) return false;
            bool tailswap = nd >= 2;
            for (uint32_t d = 0; d + 2 < nd && tailswap; ++d) if (perm[d] != d) tailswap = false;
            if (tailswap && (perm[nd-2] != nd-1 || perm[nd-1] != nd-2)) tailswap = false;
            if (tailswap) {
                size_t batch = 1;
                for (uint32_t d = 0; d + 2 < nd; ++d) batch *= in.shape[d];
                return cactus_metal_encode_transpose2d(out.get_data(), in.get_data(),
                    (uint32_t)batch, (uint32_t)in.shape[nd-2], (uint32_t)in.shape[nd-1]);
            }
            size_t istr[8]; row_strides(in.shape, istr);
            uint32_t oshape[8], sstride[8];
            for (uint32_t d=0; d<nd; ++d){ oshape[d]=(uint32_t)out.shape[d]; sstride[d]=(uint32_t)istr[perm[d]]; }
            return cactus_metal_encode_strided_copy(out.get_data(), in.get_data(), oshape, sstride, nd,
                (uint32_t)out.total_size, 0, in.byte_size, out.byte_size);
        }
        case OpType::SLICE: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out)) return false;
            size_t axis = (size_t)node.params.axis;
            uint32_t nd = (uint32_t)in.shape.size();
            if (axis == 0 || nd == 0 || nd > 8 || axis >= nd || out.shape.size() != nd) return false;
            size_t istr[8]; row_strides(in.shape, istr);
            uint32_t oshape[8], sstride[8];
            for (uint32_t d=0; d<nd; ++d){ oshape[d]=(uint32_t)out.shape[d]; sstride[d]=(uint32_t)istr[d]; }
            return cactus_metal_encode_strided_copy(out.get_data(), in.get_data(), oshape, sstride, nd,
                (uint32_t)out.total_size, (uint32_t)(node.params.slice_start * istr[axis]), in.byte_size, out.byte_size);
        }
        case OpType::INDEX: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out)) return false;
            size_t axis = (size_t)node.params.axis;
            if (axis == 0 || axis >= in.shape.size()) return false;
            size_t istr[8]; row_strides(in.shape, istr);
            size_t slice = istr[axis], block = istr[axis-1];
            uint32_t oshape[2] = { (uint32_t)(in.total_size/block), (uint32_t)slice };
            uint32_t sstride[2] = { (uint32_t)block, 1u };
            return cactus_metal_encode_strided_copy(out.get_data(), in.get_data(), oshape, sstride, 2,
                (uint32_t)out.total_size, (uint32_t)(node.params.index_value * slice), in.byte_size, out.byte_size);
        }
        case OpType::CAT: {
            if (node.input_ids.size() < 2 || out.shape.empty()) return false;
            size_t axis = (size_t)node.params.axis;
            uint32_t nd = (uint32_t)out.shape.size();
            if (axis >= nd || nd > 8) return false;
            for (size_t ii = 0; ii < node.input_ids.size(); ++ii) {
                const auto& cin = get_input(node, ii, nodes, map);
                if (!fp16(cin) || cin.shape.size() != nd) return false;
            }
            size_t ostr[8]; row_strides(out.shape, ostr);
            size_t axis_off = 0;
            for (size_t ii = 0; ii < node.input_ids.size(); ++ii) {
                const auto& cin = get_input(node, ii, nodes, map);
                uint32_t ishape[8], ostride[8];
                for (uint32_t d=0; d<nd; ++d){ ishape[d]=(uint32_t)cin.shape[d]; ostride[d]=(uint32_t)ostr[d]; }
                size_t bcast = (axis > 0 && cin.shape[0] == 1 && out.shape[0] > 1) ? out.shape[0] : 1;
                for (size_t r = 0; r < bcast; ++r) {
                    if (!cactus_metal_encode_strided_scatter(out.get_data(), cin.get_data(), ishape, ostride, nd,
                            (uint32_t)cin.total_size, (uint32_t)(axis_off * ostr[axis] + r * ostr[0]), cin.byte_size, out.byte_size))
                        return false;
                }
                axis_off += cin.shape[axis];
            }
            return true;
        }
        case OpType::SCALAR_ADD: case OpType::SCALAR_SUBTRACT:
        case OpType::SCALAR_MULTIPLY: case OpType::SCALAR_DIVIDE:
        case OpType::SCALAR_EXP: case OpType::SCALAR_SQRT: case OpType::SCALAR_COS:
        case OpType::SCALAR_SIN: case OpType::SCALAR_LOG: case OpType::ABS:
        case OpType::POW: case OpType::SCALAR_NOT_EQUAL: case OpType::LEAKY_RELU: {
            const auto& in = get_input(node, 0, nodes, map);
            int code;
            switch (node.op_type) {
                case OpType::SCALAR_ADD: code = 0; break;
                case OpType::SCALAR_SUBTRACT: code = 1; break;
                case OpType::SCALAR_MULTIPLY: code = 2; break;
                case OpType::SCALAR_DIVIDE: code = 3; break;
                case OpType::SCALAR_EXP: code = 4; break;
                case OpType::SCALAR_SQRT: code = 5; break;
                case OpType::SCALAR_COS: code = 6; break;
                case OpType::SCALAR_SIN: code = 7; break;
                case OpType::SCALAR_LOG: code = 8; break;
                case OpType::ABS: code = 9; break;
                case OpType::POW: code = 10; break;
                case OpType::SCALAR_NOT_EQUAL: code = 11; break;
                default: code = 12; break;
            }
            if (in.precision == Precision::FP32 && out.precision == Precision::FP32)
                return cactus_metal_encode_scalar_f32(code, out.get_data(), in.get_data(), out.total_size, node.params.scalar);
            if (!fp16(in) || !fp16(out)) return false;
            return cactus_metal_encode_scalar(code, out.get_data(), in.get_data(), out.total_size, node.params.scalar);
        }
        case OpType::GELU: case OpType::TANH: case OpType::SILU: case OpType::RELU:
        case OpType::GELU_ERF: case OpType::SIGMOID: {
            const auto& in = get_input(node, 0, nodes, map);
            int code = node.op_type==OpType::GELU?0: node.op_type==OpType::TANH?1: node.op_type==OpType::SILU?2:
                       node.op_type==OpType::GELU_ERF?4: node.op_type==OpType::SIGMOID?5:3;
            if (in.precision == Precision::FP32 && out.precision == Precision::FP32)
                return cactus_metal_encode_unary_f32(code, out.get_data(), in.get_data(), out.total_size);
            if (!fp16(in) || !fp16(out)) return false;
            return cactus_metal_encode_unary(code, out.get_data(), in.get_data(), out.total_size);
        }
        case OpType::SUM: case OpType::MEAN: case OpType::VARIANCE:
        case OpType::MIN: case OpType::MAX: {
            const auto& in = get_input(node, 0, nodes, map);
            int axis = node.params.axis;
            if (axis < 0 || (size_t)axis >= in.shape.size()) return false;
            bool f32 = in.precision == Precision::FP32 && out.precision == Precision::FP32;
            if (!f32 && (!fp16(in) || !fp16(out))) return false;
            int code = node.op_type==OpType::SUM?0: node.op_type==OpType::MEAN?1:
                       node.op_type==OpType::VARIANCE?2: node.op_type==OpType::MIN?3:4;
            size_t outer = 1, inner = 1;
            for (size_t d = 0; d < (size_t)axis; ++d) outer *= in.shape[d];
            for (size_t d = (size_t)axis + 1; d < in.shape.size(); ++d) inner *= in.shape[d];
            return cactus_metal_encode_reduce_axis(code, out.get_data(), in.get_data(),
                (uint32_t)outer, (uint32_t)in.shape[(size_t)axis], (uint32_t)inner, f32 ? 1 : 0);
        }
        case OpType::CUMSUM: {
            const auto& in = get_input(node, 0, nodes, map);
            int axis = node.params.axis;
            if (axis < 0 || (size_t)axis >= in.shape.size()) return false;
            bool f32 = in.precision == Precision::FP32 && out.precision == Precision::FP32;
            if (!f32 && (!fp16(in) || !fp16(out))) return false;
            size_t outer = 1, inner = 1;
            for (size_t d = 0; d < (size_t)axis; ++d) outer *= in.shape[d];
            for (size_t d = (size_t)axis + 1; d < in.shape.size(); ++d) inner *= in.shape[d];
            return cactus_metal_encode_cumsum(out.get_data(), in.get_data(),
                (uint32_t)outer, (uint32_t)in.shape[(size_t)axis], (uint32_t)inner, f32 ? 1 : 0);
        }
        case OpType::CONCAT: {
            if (node.input_ids.size() != 2) return false;
            const auto& a = get_input(node, 0, nodes, map);
            const auto& b = get_input(node, 1, nodes, map);
            if (!fp16(a) || !fp16(b) || !fp16(out)) return false;
            int axis = node.params.axis;
            if (axis < 0) axis += (int)a.shape.size();
            if (axis < 0 || (size_t)axis >= a.shape.size() || a.shape.size() != b.shape.size()) return false;
            size_t a_outer = 1, b_outer = 1, inner = 1;
            for (size_t d = 0; d < (size_t)axis; ++d) { a_outer *= a.shape[d]; b_outer *= b.shape[d]; }
            for (size_t d = (size_t)axis + 1; d < a.shape.size(); ++d) inner *= a.shape[d];
            return cactus_metal_encode_concat2(out.get_data(), a.get_data(), b.get_data(),
                (uint32_t)a_outer, (uint32_t)b_outer,
                (uint32_t)a.shape[(size_t)axis], (uint32_t)b.shape[(size_t)axis], (uint32_t)inner);
        }
        case OpType::GATHER: {
            const auto& t = get_input(node, 0, nodes, map);
            const auto& idx = get_input(node, 1, nodes, map);
            if (!fp16(t) || !fp16(out) || idx.precision != Precision::FP32) return false;
            if (t.shape.size() < 2) return false;
            size_t D = t.total_size / t.shape[0];
            return cactus_metal_encode_gather_f32idx(out.get_data(), t.get_data(), idx.get_data(),
                (uint32_t)idx.total_size, (uint32_t)D, t.byte_size);
        }
        case OpType::ROPE: case OpType::ROPE_GPTJ: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out) || in.shape.size() < 4) return false;
            size_t S = in.shape[1], H = in.shape[2], D = in.shape[3];
            bool gptj = node.op_type == OpType::ROPE_GPTJ;
            size_t rot = gptj ? (size_t)node.params.scalar : D;
            if (rot == 0 || rot > D || (rot % 2) != 0) return false;
            size_t tokens = in.total_size / D;
            return cactus_metal_encode_rope_full(out.get_data(), in.get_data(),
                (uint32_t)tokens, (uint32_t)S, (uint32_t)H, (uint32_t)D, (uint32_t)rot,
                (uint32_t)node.params.position_offset, node.params.theta, gptj ? 1 : 0);
        }
        case OpType::MAXPOOL1D: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out) || in.shape.size() != 3 || out.shape.size() != 3) return false;
            return cactus_metal_encode_maxpool1d(out.get_data(), in.get_data(),
                (uint32_t)(in.shape[0] * in.shape[1]), (uint32_t)in.shape[2],
                (uint32_t)out.shape[2], (uint32_t)node.params.kernel_size, (uint32_t)node.params.stride);
        }
        case OpType::BILINEAR_INTERPOLATION: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out) || in.shape.size() != 2) return false;
            size_t src = (size_t)std::sqrt((double)in.shape[0]);
            if (src * src != in.shape[0]) return false;
            return cactus_metal_encode_bilinear(out.get_data(), in.get_data(),
                (uint32_t)src, (uint32_t)src, (uint32_t)node.params.dst_height,
                (uint32_t)node.params.dst_width, (uint32_t)in.shape[1],
                node.params.align_corners ? 1 : 0);
        }
        case OpType::CONV1D: case OpType::CONV1D_K7S3: {
            if (node.input_ids.size() < 2) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            const BufferDesc* b = node.input_ids.size() > 2 ? &get_input(node, 2, nodes, map) : nullptr;
            if (!fp16(x) || !fp16(w) || !fp16(out)) return false;
            if (b && b->precision != Precision::FP16) return false;
            if (x.shape.size() != 3 || w.shape.size() != 3 || out.shape.size() != 3) return false;
            bool ck_co = node.op_type == OpType::CONV1D_K7S3;
            size_t K = ck_co ? w.shape[1] : w.shape[2];
            size_t Cout = ck_co ? w.shape[2] : w.shape[0];
            return cactus_metal_encode_conv1d_gen(out.get_data(), x.get_data(), w.get_data(),
                b ? b->get_data() : nullptr, (uint32_t)x.shape[0], (uint32_t)x.shape[1],
                (uint32_t)x.shape[2], (uint32_t)Cout, (uint32_t)out.shape[2], (uint32_t)K,
                (uint32_t)(node.params.stride ? node.params.stride : 1), ck_co ? 1 : 0);
        }
        case OpType::CONV1D_CAUSAL: case OpType::CONV1D_SAME_DEPTHWISE_K9: {
            if (node.input_ids.size() < 2) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            bool causal = node.op_type == OpType::CONV1D_CAUSAL;
            const BufferDesc* b = (!causal && node.input_ids.size() > 2) ? &get_input(node, 2, nodes, map) : nullptr;
            if (!fp16(x) || !fp16(out)) return false;
            if (b && b->precision != Precision::FP16) return false;
            if (x.shape.size() != 3) return false;
            size_t C = x.shape[2];
            size_t K;
            if (causal) {
                if (w.shape.size() != 3 || w.shape[0] != C || w.shape[1] != 1) return false;
                K = w.shape[2];
            } else {
                K = w.shape.back();
                if (w.total_size != C * K) return false;
            }
            const __fp16* wptr = metal_conv_weight_f16(w, C, K);
            if (!wptr) return false;
            size_t dil = causal ? (node.params.dilation ? node.params.dilation : 1) : 1;
            size_t pad = causal ? (K - 1) * dil : K / 2;
            return cactus_metal_encode_conv1d_nlc_dw(out.get_data(), x.get_data(), wptr,
                b ? b->get_data() : nullptr, (uint32_t)x.shape[0], (uint32_t)x.shape[1],
                (uint32_t)C, (uint32_t)K, (uint32_t)dil, (uint32_t)pad);
        }
        case OpType::CONV2D_K3S2P1: case OpType::CONV2D_K3S1P1:
        case OpType::CONV2D_DEPTHWISE_K3S2P1: case OpType::CONV2D_POINTWISE_1X1: {
            if (node.input_ids.size() < 2) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            const BufferDesc* b = node.input_ids.size() > 2 ? &get_input(node, 2, nodes, map) : nullptr;
            if (!fp16(x) || !fp16(w) || !fp16(out)) return false;
            if (b && b->precision != Precision::FP16) return false;
            if (x.shape.size() != 4 || out.shape.size() != 4) return false;
            bool dw = node.op_type == OpType::CONV2D_DEPTHWISE_K3S2P1;
            bool pw = node.op_type == OpType::CONV2D_POINTWISE_1X1;
            uint32_t K = pw ? 1 : 3;
            uint32_t stride = (node.op_type == OpType::CONV2D_K3S1P1 || pw) ? 1 : 2;
            uint32_t pad = pw ? 0 : 1;
            return cactus_metal_encode_conv2d(out.get_data(), x.get_data(), w.get_data(),
                b ? b->get_data() : nullptr, (uint32_t)x.shape[0], (uint32_t)x.shape[1],
                (uint32_t)x.shape[2], (uint32_t)x.shape[3], (uint32_t)out.shape[1],
                (uint32_t)out.shape[2], (uint32_t)out.shape[3], K, stride, pad, dw ? 1 : 0);
        }
        case OpType::CONV1D_POINTWISE: {
            if (node.input_ids.size() < 2) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            const BufferDesc* b = node.input_ids.size() > 2 ? &get_input(node, 2, nodes, map) : nullptr;
            if (!fp16(x) || !fp16(out)) return false;
            if (b && b->precision != Precision::FP16) return false;
            if (x.shape.size() != 3 || out.shape.size() != 3) return false;
            size_t Cin = x.shape[2], Cout = out.shape[2];
            if (w.total_size != Cin * Cout) return false;
            const __fp16* wptr = metal_conv_weight_f16(w, Cout, Cin);
            if (!wptr) return false;
            size_t rows = x.shape[0] * x.shape[1];
            if (!cactus_metal_encode_gemm_f16(out.get_data(), x.get_data(), wptr,
                    (uint32_t)rows, (uint32_t)Cin, (uint32_t)Cout, 1)) return false;
            if (b) return cactus_metal_encode_bias_add_rows(out.get_data(), b->get_data(),
                    (uint32_t)Cout, (uint32_t)out.total_size);
            return true;
        }
        case OpType::REL_POS_BIAS: {
            if (node.input_ids.size() != 2) return false;
            const auto& q = get_input(node, 0, nodes, map);
            const auto& r = get_input(node, 1, nodes, map);
            if (!fp16(q) || !fp16(r) || !fp16(out)) return false;
            if (q.shape.size() != 4 || r.shape.size() != 4) return false;
            size_t B = q.shape[0], T = q.shape[1], H = q.shape[2], D = q.shape[3];
            size_t Rb = r.shape[0], R = r.shape[1];
            if (B == 0 || T == 0 || (Rb != 1 && Rb != B)) return false;
            if (r.shape[2] != H || r.shape[3] != D) return false;
            if (R < 2 * T - 1 || out.total_size != B * H * T * T) return false;
            return cactus_metal_encode_rel_pos_bias(out.get_data(), q.get_data(), r.get_data(),
                (uint32_t)B, (uint32_t)T, (uint32_t)H, (uint32_t)D, (uint32_t)R,
                Rb == 1 ? 0 : 1, node.params.scale);
        }
        case OpType::BATCHNORM: {
            if (node.input_ids.size() != 5) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            const auto& b = get_input(node, 2, nodes, map);
            const auto& rm = get_input(node, 3, nodes, map);
            const auto& rv = get_input(node, 4, nodes, map);
            if (!fp16(x) || !fp16(out) || !fp16(w) || !fp16(b) || !fp16(rm) || !fp16(rv)) return false;
            int axis = node.params.axis;
            if (axis < 0) axis += (int)x.shape.size();
            if (axis < 0 || (size_t)axis >= x.shape.size()) return false;
            size_t inner = 1;
            for (size_t d = (size_t)axis + 1; d < x.shape.size(); ++d) inner *= x.shape[d];
            return cactus_metal_encode_batchnorm(out.get_data(), x.get_data(), w.get_data(),
                b.get_data(), rm.get_data(), rv.get_data(), (uint32_t)x.shape[(size_t)axis],
                (uint32_t)inner, (uint32_t)x.total_size, node.params.epsilon);
        }
        case OpType::GROUPNORM: {
            if (node.input_ids.size() < 3) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            const auto& b = get_input(node, 2, nodes, map);
            if (!fp16(x) || !fp16(out) || !fp16(w) || !fp16(b) || x.shape.size() < 2) return false;
            size_t spatial = 1;
            for (size_t d = 2; d < x.shape.size(); ++d) spatial *= x.shape[d];
            size_t groups = node.params.num_groups ? node.params.num_groups : 32;
            return cactus_metal_encode_groupnorm(out.get_data(), x.get_data(), w.get_data(),
                b.get_data(), (uint32_t)x.shape[0], (uint32_t)x.shape[1], (uint32_t)spatial,
                (uint32_t)groups, node.params.epsilon);
        }
        case OpType::CLAMP: {
            const auto& in = get_input(node, 0, nodes, map);
            bool f32 = in.precision == Precision::FP32 && out.precision == Precision::FP32;
            if (!f32 && (!fp16(in) || !fp16(out))) return false;
            return cactus_metal_encode_clamp(out.get_data(), in.get_data(), out.total_size,
                node.params.scalar, node.params.scale, f32 ? 1 : 0);
        }
        case OpType::GLU: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out) || out.shape.empty()) return false;
            int axis = node.params.axis;
            if (axis < 0) axis += (int)in.shape.size();
            if (axis < 0 || (size_t)axis >= in.shape.size()) return false;
            size_t split = in.shape[(size_t)axis] / 2;
            size_t inner = 1;
            for (size_t i = (size_t)axis + 1; i < in.shape.size(); ++i) inner *= in.shape[i];
            return cactus_metal_encode_glu(out.get_data(), in.get_data(), split, inner, out.total_size);
        }
        case OpType::SOFTMAX: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out) || out.shape.empty()) return false;
            size_t cols = out.shape.back();
            if (cols == 0) return false;
            return cactus_metal_encode_softmax_rows(out.get_data(), in.get_data(), out.total_size / cols, cols);
        }
        case OpType::GATED_DELTANET_DECODE: {
            if (node.input_ids.size() != 6) return false;
            const auto& q = get_input(node, 0, nodes, map);
            const auto& k = get_input(node, 1, nodes, map);
            const auto& v = get_input(node, 2, nodes, map);
            const auto& g = get_input(node, 3, nodes, map);
            const auto& b = get_input(node, 4, nodes, map);
            const auto& st = get_input(node, 5, nodes, map);
            if (!fp16(q) || !fp16(k) || !fp16(v) || !fp16(g) || !fp16(b) || !fp16(st) || !fp16(out)) return false;
            if (q.shape.size() != 4 || v.shape.size() != 4 || st.shape.size() != 4) return false;
            if (q.shape[1] != 1) return false;
            size_t B = q.shape[0], Hq = q.shape[2], K = q.shape[3];
            size_t Hv = v.shape[2], V = v.shape[3];
            if (Hq == 0 || (Hv % Hq) != 0) return false;
            if (st.shape[0] != B || st.shape[1] != K || st.shape[2] != Hv || st.shape[3] != V) return false;
            if (out.total_size != B * (1 + K) * Hv * V) return false;
            return cactus_metal_encode_deltanet_decode(out.get_data(), q.get_data(), k.get_data(),
                v.get_data(), g.get_data(), b.get_data(), st.get_data(),
                (uint32_t)B, (uint32_t)Hq, (uint32_t)Hv, (uint32_t)K, (uint32_t)V, node.params.scale);
        }
        case OpType::GATED_DELTANET_PREFILL: {
            if (node.input_ids.size() != 6) return false;
            const auto& q = get_input(node, 0, nodes, map);
            const auto& k = get_input(node, 1, nodes, map);
            const auto& v = get_input(node, 2, nodes, map);
            const auto& g = get_input(node, 3, nodes, map);
            const auto& b = get_input(node, 4, nodes, map);
            const auto& st = get_input(node, 5, nodes, map);
            if (!fp16(q) || !fp16(k) || !fp16(v) || !fp16(g) || !fp16(b) || !fp16(st) || !fp16(out)) return false;
            if (q.shape.size() != 4 || v.shape.size() != 4 || st.shape.size() != 4) return false;
            size_t B = q.shape[0], T = q.shape[1], Hq = q.shape[2], K = q.shape[3];
            size_t Hv = v.shape[2], V = v.shape[3];
            if (Hq == 0 || (Hv % Hq) != 0) return false;
            if (st.shape[0] != B || st.shape[1] != K || st.shape[2] != Hv || st.shape[3] != V) return false;
            if (out.total_size != B * (T + K) * Hv * V) return false;
            return cactus_metal_encode_deltanet_prefill(out.get_data(), q.get_data(), k.get_data(),
                v.get_data(), g.get_data(), b.get_data(), st.get_data(),
                (uint32_t)B, (uint32_t)T, (uint32_t)Hq, (uint32_t)Hv, (uint32_t)K, (uint32_t)V, node.params.scale);
        }
        case OpType::RECURRENT_CACHE_WRITE: {
            if (node.input_ids.size() < 2) return false;
            const auto& src = get_input(node, 0, nodes, map);
            BufferDesc& cache = nodes[map.at(node.input_ids[1])]->output_buffer;
            if (!src.get_data() || !cache.get_data() || src.byte_size == 0) return false;
            if (src.byte_size > cache.byte_size) return false;
            return cactus_metal_encode_copy(cache.get_data(), src.get_data(), src.byte_size);
        }
        case OpType::TOPK: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || out.precision != Precision::FP32) return false;
            if (in.shape.size() != 2 || node.params.top_k == 0 || node.params.top_k > 16) return false;
            return cactus_metal_encode_topk_rows(out.get_data(), in.get_data(),
                in.shape[0], in.shape[1], node.params.top_k);
        }
        case OpType::MOE_LAYER: {
            const size_t num_experts = node.params.num_experts;
            const size_t top_k = node.params.num_experts_per_tok;
            if (!node.params.moe_gated || num_experts == 0 || top_k == 0 || top_k > 16) return false;
            if (node.input_ids.size() != 3 + 3 * num_experts) return false;
            const auto& hidden = get_input(node, 0, nodes, map);
            const auto& probs = get_input(node, 1, nodes, map);
            const auto& topk = get_input(node, 2, nodes, map);
            if (!fp16(hidden) || !fp16(out) || !fp16(probs)) return false;
            if (topk.precision != Precision::FP32) return false;
            if (hidden.shape.size() != 2 || probs.shape.size() != 2 || probs.shape[1] != num_experts) return false;
            uint32_t act;
            switch (node.params.activation) {
                case Activation::SILU: act = 0u; break;
                case Activation::GELU: act = 1u; break;
                default: return false;
            }
            const auto& w1_0 = get_input(node, 3, nodes, map);
            if (!PrecisionTraits::is_cq(w1_0.precision) || w1_0.group_size == 0) return false;
            CactusQuantMatrix key = w1_0.to_cq_matrix();
            if (!cactus_metal_moe_cq4_ready(&key)) {
                std::vector<CactusQuantMatrix> w1s(num_experts), w3s(num_experts), w2s(num_experts);
                for (size_t e = 0; e < num_experts; ++e) {
                    const auto& b1 = get_input(node, 3 + e, nodes, map);
                    const auto& b3 = get_input(node, 3 + num_experts + e, nodes, map);
                    const auto& b2 = get_input(node, 3 + 2 * num_experts + e, nodes, map);
                    if (!PrecisionTraits::is_cq(b1.precision) || b1.group_size == 0 ||
                        !PrecisionTraits::is_cq(b3.precision) || b3.group_size == 0 ||
                        !PrecisionTraits::is_cq(b2.precision) || b2.group_size == 0) return false;
                    w1s[e] = b1.to_cq_matrix();
                    w3s[e] = b3.to_cq_matrix();
                    w2s[e] = b2.to_cq_matrix();
                }
                if (!cactus_metal_moe_cq4_build(w1s.data(), w3s.data(), w2s.data(), (uint32_t)num_experts))
                    return false;
            }
            return cactus_metal_encode_moe_gated_cq4(out.get_data(), hidden.get_data(), probs.get_data(),
                topk.get_data(), &key, (uint32_t)num_experts, (uint32_t)top_k, (uint32_t)hidden.shape[0],
                act, node.params.normalize_routing ? 1u : 0u, node.params.epsilon, node.params.scalar);
        }
        case OpType::LAYERNORM: {
            if (node.input_ids.size() < 2 || out.shape.empty()) return false;
            const auto& in = get_input(node, 0, nodes, map);
            const auto& w  = get_input(node, 1, nodes, map);
            const BufferDesc* b = node.input_ids.size() > 2 ? &get_input(node, 2, nodes, map) : nullptr;
            if (!fp16(in) || !fp16(out) || !fp16(w) || (b && b->precision != Precision::FP16)) return false;
            size_t dim = out.shape.back();
            if (dim == 0 || w.total_size != dim) return false;
            return cactus_metal_encode_layer_norm(out.get_data(), in.get_data(), w.get_data(),
                b ? b->get_data() : nullptr, out.total_size / dim, dim, node.params.epsilon);
        }
        case OpType::CONV1D_K3: {
            if (node.input_ids.size() < 2) return false;
            const auto& x = get_input(node, 0, nodes, map);
            const auto& w = get_input(node, 1, nodes, map);
            if (!fp16(x) || !fp16(out)) return false;
            if (x.shape.size() != 3 || w.shape.size() != 3 || x.shape[0] != 1 || w.shape[2] != 3) return false;
            size_t Cin = x.shape[1], L = x.shape[2], Cout = w.shape[0];
            size_t stride = node.params.stride ? node.params.stride : 1;
            size_t Lout = ((L - 1) / stride) + 1;
            int w_int8 = w.precision == Precision::INT8 ? 1 : 0;
            if (!w_int8 && w.precision != Precision::FP16) return false;
            if (out.total_size != Cout * Lout) return false;
            return cactus_metal_encode_conv1d_k3(out.get_data(), x.get_data(), w.get_data(), w_int8,
                w.activation_scales_data, (uint32_t)w.group_size,
                (uint32_t)Cin, (uint32_t)L, (uint32_t)Cout, (uint32_t)Lout, (uint32_t)stride);
        }
        case OpType::ATTENTION: {
            if (node.input_ids.size() < 3) return false;
            const auto& q = get_input(node, 0, nodes, map);
            const auto& k = get_input(node, 1, nodes, map);
            const auto& v = get_input(node, 2, nodes, map);
            const BufferDesc* mk = node.input_ids.size() > 3 ? &get_input(node, 3, nodes, map) : nullptr;
            if (!fp16(q) || !fp16(k) || !fp16(v) || !fp16(out)) return false;
            if (q.shape.size() != 4 || k.shape.size() != 4 || v.shape.size() != 4) return false;
            size_t B = q.shape[0], T = q.shape[1], HQ = q.shape[2], D = q.shape[3];
            size_t S = k.shape[1], HKV = k.shape[2], DV = v.shape[3];
            if (HKV == 0 || HQ % HKV != 0) return false;
            uint32_t mask_mode = 0;
            if (mk) {
                if (mk->precision != Precision::FP16) return false;
                bool per_head = mk->shape.size() == 4;
                if (!per_head && mk->shape.size() != 3) return false;
                mask_mode = (node.params.attention_mask_is_additive ? 1u : 2u) + (per_head ? 2u : 0u);
            }
            float scale = node.params.scale != 0.0f ? node.params.scale : 1.0f/std::sqrt((float)D);
            return cactus_metal_encode_attention_f16(out.get_data(), q.get_data(), k.get_data(), v.get_data(),
                mk ? mk->get_data() : nullptr,
                (uint32_t)B, (uint32_t)T, (uint32_t)S, (uint32_t)HQ, (uint32_t)HKV, (uint32_t)D, (uint32_t)DV,
                scale, node.params.is_causal ? 1u : 0u, (uint32_t)node.params.position_offset,
                (uint32_t)node.params.window_size, node.params.logit_cap, mask_mode);
        }
        case OpType::RMS_NORM: {
            if (node.input_ids.size() < 2 || out.shape.empty()) return false;
            const auto& in = get_input(node, 0, nodes, map);
            const auto& w  = get_input(node, 1, nodes, map);
            if (!fp16(in) || !fp16(out) || !fp16(w)) return false;
            size_t dim = out.shape.back();
            if (dim == 0) return false;
            return cactus_metal_encode_rms_norm(out.get_data(), in.get_data(), w.get_data(),
                                                out.total_size / dim, dim, node.params.epsilon);
        }
        case OpType::PRECISION_CAST: {
            const auto& in = get_input(node, 0, nodes, map);
            return cactus_metal_encode_cast(out.get_data(), static_cast<int>(out.precision),
                                            in.get_data(), static_cast<int>(in.precision), in.total_size);
        }
        case OpType::ATTENTION_CACHED: {
            if (node.input_ids.size() < 5) return false;
            const auto& qb = get_input(node, 0, nodes, map);
            const auto& knew = get_input(node, 1, nodes, map);
            const auto& vnew = get_input(node, 2, nodes, map);
            const auto& kcache = get_input(node, 3, nodes, map);
            const auto& vcache = get_input(node, 4, nodes, map);
            if (qb.shape.size() < 3) return false;
            size_t batch=qb.shape[0], seq=qb.shape[1], nqh=qb.shape[2];
            if (batch != 1) return false;
            if (kcache.precision == Precision::FP16 || vcache.precision == Precision::FP16) return false;
            const uint64_t* km = reinterpret_cast<const uint64_t*>(kcache.get_data());
            const uint64_t* vm = reinterpret_cast<const uint64_t*>(vcache.get_data());
            if (!km || !vm) return false;
            size_t cache_len=km[0], max_seq=km[1], kv_heads=km[2], hdim=km[3], v_max=vm[1];
            if (kv_heads == 0 || hdim == 0 || nqh % kv_heads != 0) return false;
            size_t v_hdim = node.params.v_head_dim > 0 ? node.params.v_head_dim : hdim;
            size_t new_seq_len = knew.total_size / (kv_heads * hdim);
            size_t history_len = cache_len >= new_seq_len ? cache_len - new_seq_len : 0;
            size_t po = node.params.position_offset, pos, new_len = seq;
            if (po == std::numeric_limits<size_t>::max()) pos = history_len;
            else if (po == std::numeric_limits<size_t>::max() - 1) {
                pos = (cache_len >= seq) ? cache_len - seq : 0; history_len = cache_len; new_len = 0;
            } else pos = po;
            size_t total_keys = history_len + new_len;
            size_t win = node.params.window_size;
            size_t kv_start = (win > 0 && pos > win) ? pos - win : 0;
            if (pos > history_len) kv_start = 0;
            size_t kv_end = node.params.is_causal ? std::min(total_keys, pos + 1) : total_keys;
            float scale = node.params.scale != 0.0f ? node.params.scale : 1.0f/std::sqrt((float)hdim);
            const char* bk = static_cast<const char*>(kcache.get_data());
            const char* bv = static_cast<const char*>(vcache.get_data());
            size_t ngK=(hdim+31)/32, ngV=(v_hdim+31)/32;
            if (seq > 1) {
                size_t sink = km[4];
                uint32_t ringv = (win > 0 && max_seq > 2*sink + 1) ? (uint32_t)(max_seq - 2*sink - 1) : 0u;
                return cactus_metal_encode_attention_i8_prefill(
                    out.get_data(), qb.get_data(), knew.get_data(), vnew.get_data(),
                    bk + 64, bv + 64,
                    bk + 64 + max_seq*kv_heads*hdim, bv + 64 + v_max*kv_heads*v_hdim,
                    (uint32_t)nqh, (uint32_t)kv_heads, (uint32_t)hdim, (uint32_t)v_hdim,
                    (uint32_t)history_len, (uint32_t)new_len, (uint32_t)pos,
                    (uint32_t)win, node.params.is_causal ? 1u : 0u, (uint32_t)seq, scale,
                    max_seq*kv_heads*hdim, v_max*kv_heads*v_hdim,
                    max_seq*kv_heads*ngK*sizeof(float), v_max*kv_heads*ngV*sizeof(float),
                    (uint32_t)sink, ringv);
            }
            size_t cache_ceiling = nodes[map.at(node.input_ids[3])]->params.max_cache_seq_len;
            if (win > 0 && win < cache_ceiling) {
                size_t sink = km[4];
                size_t ringW = max_seq > sink + 1 ? max_seq - sink - 1 : 0;
                if (ringW == 0) return false;
                if (total_keys > ringW) {
                    history_len = ringW;
                    total_keys = ringW;
                    kv_start = 0;
                    kv_end = ringW;
                }
            }
            bool ok = cactus_metal_encode_attention_i8(
                out.get_data(), qb.get_data(), knew.get_data(), vnew.get_data(),
                bk + 64, bv + 64,
                bk + 64 + max_seq*kv_heads*hdim, bv + 64 + v_max*kv_heads*v_hdim,
                (uint32_t)nqh, (uint32_t)kv_heads, (uint32_t)hdim, (uint32_t)v_hdim,
                (uint32_t)history_len, (uint32_t)total_keys, (uint32_t)kv_start, (uint32_t)kv_end, scale,
                history_len*kv_heads*hdim, history_len*kv_heads*v_hdim,
                history_len*kv_heads*ngK*sizeof(float), history_len*kv_heads*ngV*sizeof(float));
            return ok;
        }
        case OpType::KV_CACHE_APPEND: {
            if (node.input_ids.size() < 2) return false;
            const auto& new_kv = get_input(node, 0, nodes, map);
            GraphNode& cache_node = *nodes[map.at(node.input_ids[1])];
            BufferDesc& cache = cache_node.output_buffer;
            if (!fp16(new_kv) || cache.precision == Precision::FP16 || !cache.get_data()) return false;
            uint64_t* km = reinterpret_cast<uint64_t*>(cache.get_data());
            size_t current_len=km[0], max_len=km[1], kv_heads=km[2], hdim=km[3], sink=km[4], num_slots=km[5];
            if (num_slots != 1 || kv_heads == 0 || hdim == 0) return false;
            size_t new_seq_len = new_kv.total_size / (kv_heads * hdim);
            size_t ceiling = cache_node.params.max_cache_seq_len, ws = node.params.window_size;
            bool sliding = ws > 0 && ws < ceiling;
            size_t new_total = current_len + new_seq_len;
            char* base = static_cast<char*>(cache.get_data());
            size_t ngK = (hdim + 31)/32;
            if (new_seq_len > 1) {
                if (sliding) {
                    if (max_len <= sink + 1) return false;
                    uint32_t W = (uint32_t)(max_len - sink - 1);
                    if (!cactus_metal_encode_kv_append_ring_i8_m(new_kv.get_data(), base + 64,
                            base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                            (uint32_t)current_len, 32, (uint32_t)new_seq_len, (uint32_t)sink, W,
                            new_kv.byte_size, max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                        return false;
                    km[0] = new_total;
                } else if (ws > 0 && new_total > max_len) {
                    size_t window = max_len;
                    size_t keep_sink = std::min({sink, current_len, window});
                    size_t tail_capacity = window - keep_sink;
                    if (new_seq_len >= tail_capacity) return false;
                    size_t remaining = std::min(tail_capacity - new_seq_len, current_len - keep_sink);
                    size_t shift_src = current_len - remaining;
                    if (!cactus_metal_encode_kv_append_sliding_i8_m(new_kv.get_data(), base + 64,
                            base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                            (uint32_t)keep_sink, (uint32_t)remaining, (uint32_t)shift_src, 32, (uint32_t)new_seq_len,
                            new_kv.byte_size, max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                        return false;
                    km[0] = keep_sink + remaining + new_seq_len;
                } else {
                    if (new_total > max_len) {
                        if (!cactus_kv_cache_grow(cache, new_total, ceiling)) return false;
                        km = reinterpret_cast<uint64_t*>(cache.get_data());
                        max_len = km[1]; base = static_cast<char*>(cache.get_data());
                        if (new_total > max_len) return false;
                    }
                    if (!cactus_metal_encode_kv_append_i8_m(new_kv.get_data(), base + 64,
                            base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                            (uint32_t)current_len, 32, (uint32_t)new_seq_len,
                            new_kv.byte_size, max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                        return false;
                    km[0] = new_total;
                }
                if (out.get_data()) *out.data_as<float>() = static_cast<float>(km[0]);
                return true;
            }
            if (sliding) {
                if (max_len <= sink + 1) return false;
                uint32_t W = (uint32_t)(max_len - sink - 1);
                if (!cactus_metal_encode_kv_append_ring_i8_m(new_kv.get_data(), base + 64,
                        base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                        (uint32_t)current_len, 32, 1u, (uint32_t)sink, W,
                        new_kv.byte_size, max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                    return false;
                km[0] = new_total;
                if (out.get_data()) *out.data_as<float>() = static_cast<float>(new_total);
                return true;
            }
            if (new_total > max_len) {
                if (!cactus_kv_cache_grow(cache, new_total, ceiling)) return false;
                km = reinterpret_cast<uint64_t*>(cache.get_data());
                max_len = km[1]; base = static_cast<char*>(cache.get_data());
                if (new_total > max_len) return false;
            }
            size_t window = max_len;
            if (new_total > window) {
                size_t keep_sink = std::min({sink, current_len, window});
                size_t tail_capacity = window - keep_sink;
                if (new_seq_len >= tail_capacity) return false;
                size_t remaining = std::min(tail_capacity - new_seq_len, current_len - keep_sink);
                size_t shift_src = current_len - remaining;
                if (!cactus_metal_encode_kv_append_sliding_i8(new_kv.get_data(), base + 64,
                        base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                        (uint32_t)keep_sink, (uint32_t)remaining, (uint32_t)shift_src, 32, new_kv.byte_size,
                        max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                    return false;
                km[0] = keep_sink + remaining + new_seq_len;
                if (out.get_data()) *out.data_as<float>() = static_cast<float>(km[0]);
                return true;
            }
            if (!cactus_metal_encode_kv_append_i8(new_kv.get_data(), base + 64,
                    base + 64 + max_len*kv_heads*hdim, (uint32_t)kv_heads, (uint32_t)hdim,
                    (uint32_t)current_len, 32, new_kv.byte_size,
                    max_len*kv_heads*hdim, max_len*kv_heads*ngK*sizeof(float)))
                return false;
            km[0] = new_total;
            if (out.get_data()) *out.data_as<float>() = static_cast<float>(new_total);
            return true;
        }
        case OpType::CONV_CACHE_APPEND: {
            if (node.input_ids.size() < 2) return false;
            const auto& new_data = get_input(node, 0, nodes, map);
            BufferDesc& cache = nodes[map.at(node.input_ids[1])]->output_buffer;
            if (!fp16(out) || !cache.get_data() || !out.get_data()) return false;
            if (new_data.precision != Precision::FP16 && new_data.precision != Precision::FP32) return false;
            uint64_t* cm = reinterpret_cast<uint64_t*>(cache.get_data());
            uint64_t head = cm[0], count = cm[1], ws = cm[2], hd = cm[3];
            if (ws == 0 || hd == 0 || head >= ws) return false;
            size_t num_rows = new_data.total_size / hd;
            if (num_rows == 0 || out.total_size != ws * hd) return false;
            uint32_t nnew = (uint32_t)std::min<uint64_t>(num_rows, ws);
            uint64_t count_new = std::min<uint64_t>(ws, count + num_rows);
            if (!cactus_metal_encode_conv_cache_append(out.get_data(), new_data.get_data(),
                    static_cast<char*>(cache.get_data()) + 64,
                    (uint32_t)hd, (uint32_t)ws, nnew, (uint32_t)head, (uint32_t)count_new,
                    (uint32_t)num_rows, new_data.precision == Precision::FP32 ? 1 : 0))
                return false;
            cm[0] = (head + nnew) % ws;
            cm[1] = count_new;
            return true;
        }
        case OpType::EMBEDDING: {
            if (node.input_ids.size() < 2) return false;
            const auto& emb = get_input(node, 0, nodes, map);
            const auto& idxb = get_input(node, 1, nodes, map);
            if (!fp16(out)) return false;
            size_t M = idxb.total_size;
            if (M == 0 || M > 4096) return false;
            std::vector<uint32_t> rows(M);
            if (idxb.precision == Precision::FP32) { const float* p = idxb.data_as<float>(); for (size_t i=0;i<M;++i) rows[i]=(uint32_t)p[i]; }
            else if (idxb.precision == Precision::FP16) { const __fp16* p = idxb.data_as<__fp16>(); for (size_t i=0;i<M;++i) rows[i]=(uint32_t)(float)p[i]; }
            else { const int8_t* p = idxb.data_as<int8_t>(); for (size_t i=0;i<M;++i) rows[i]=(uint32_t)p[i]; }
            if (PrecisionTraits::is_cq(emb.precision) && emb.group_size > 0) {
                CactusQuantMatrix W = emb.to_cq_matrix();
                if (W.flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
                    return cactus_metal_encode_embedding_ortho_m(out.get_data(), &W, rows.data(), (uint32_t)M);
                return cactus_metal_encode_embedding_hadamard_m(out.get_data(), &W, rows.data(), (uint32_t)M);
            }
            if (emb.precision == Precision::FP16 && emb.shape.size() >= 2) {
                uint32_t D = (uint32_t)emb.shape[1];
                return cactus_metal_encode_gather_f16(out.get_data(), emb.get_data(),
                           emb.total_size * sizeof(__fp16), rows.data(), (uint32_t)M, D);
            }
            return false;
        }
        default: return false;
    }
}

struct MetalFusePlan;
MetalFusePlan* cactus_metal_plan_build(const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                       const std::unordered_map<size_t, size_t>& map,
                                       const std::unordered_set<size_t>& pinned_ids,
                                       const std::vector<uint8_t>& retype,
                                       const std::unordered_set<size_t>* banned = nullptr);
void cactus_metal_plan_free(MetalFusePlan* p);
bool cactus_metal_plan_fold(MetalFusePlan* p, const std::vector<std::unique_ptr<GraphNode>>& nodes);
const std::vector<uint32_t>* cactus_metal_plan_exec_list(const MetalFusePlan* p);
void* cactus_metal_plan_arena_ptr(const MetalFusePlan* p, size_t i);
bool cactus_metal_plan_has_arena(const MetalFusePlan* p);
void cactus_metal_plan_extend_last_use(const MetalFusePlan* p, std::vector<size_t>& last_use);
int32_t cactus_metal_plan_action(const MetalFusePlan* p, size_t i);
bool cactus_metal_plan_encode(MetalFusePlan* p, int32_t cid,
                              const std::vector<std::unique_ptr<GraphNode>>& nodes,
                              const std::unordered_map<size_t, size_t>& map);

void CactusGraph::build_metal_retype_plan() {
    metal_retype_built_ = true;
    const size_t n = nodes_.size();
    metal_retype_plan_.assign(n, 0);
    auto idxof = [&](size_t id) -> long {
        auto it = node_index_map_.find(id);
        return it == node_index_map_.end() ? -1 : (long)it->second;
    };
    std::vector<std::vector<size_t>> cons(n);
    for (size_t i = 0; i < n; ++i)
        for (size_t id : nodes_[i]->input_ids) { long j = idxof(id); if (j >= 0) cons[(size_t)j].push_back(i); }
    auto interior = [&](const GraphNode& nd) {
        if (nd.output_buffer.precision != Precision::FP32) return false;
        if (nd.output_buffer.shape.size() > 8) return false;
        switch (nd.op_type) {
            case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
            case OpType::MULTIPLY: case OpType::DIVIDE:
            case OpType::SCALAR_ADD: case OpType::SCALAR_SUBTRACT:
            case OpType::SCALAR_MULTIPLY: case OpType::SCALAR_DIVIDE:
            case OpType::CLAMP: case OpType::GELU: case OpType::TANH:
            case OpType::SILU: case OpType::RELU:
            case OpType::VIEW: case OpType::RESHAPE: case OpType::FLATTEN:
            case OpType::TRANSPOSE: case OpType::CAT:
                return std::fabs(nd.params.scalar) < 60000.0f && std::fabs(nd.params.scale) < 60000.0f;
            default: return false;
        }
    };
    std::vector<uint8_t> cand(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const GraphNode& nd = *nodes_[i];
        if (nd.op_type == OpType::PRECISION_CAST && nd.output_buffer.precision == Precision::FP32
            && !nd.input_ids.empty()) {
            long j = idxof(nd.input_ids[0]);
            if (j >= 0 && nodes_[(size_t)j]->output_buffer.precision == Precision::FP16) cand[i] = 1;
            continue;
        }
        if (!interior(nd) || nd.input_ids.empty()) continue;
        bool all = true;
        for (size_t id : nd.input_ids) {
            long j = idxof(id);
            if (j < 0 || !cand[(size_t)j]) { all = false; break; }
        }
        cand[i] = all ? 1 : 0;
    }
    std::vector<uint8_t> ok = cand;
    for (size_t ri = n; ri-- > 0;) {
        if (!ok[ri]) continue;
        const GraphNode& nd = *nodes_[ri];
        if (retained_output_node_ids_.count(nd.id) || persistent_node_ids_.count(nd.id)) { ok[ri] = 0; continue; }
        if (cons[ri].empty()) { ok[ri] = 0; continue; }
        for (size_t c : cons[ri]) {
            const GraphNode& cn = *nodes_[c];
            bool absorbs = (cn.op_type == OpType::PRECISION_CAST
                            && cn.output_buffer.precision == Precision::FP16)
                        || (cand[c] && ok[c]);
            if (!absorbs) { ok[ri] = 0; break; }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!ok[i]) continue;
        metal_retype_plan_[i] = nodes_[i]->op_type == OpType::PRECISION_CAST ? 1 : 2;
    }
    for (size_t i = 0; i < n; ++i) {
        const GraphNode& nd = *nodes_[i];
        if (nd.op_type != OpType::PRECISION_CAST || nd.output_buffer.precision != Precision::FP16
            || nd.input_ids.empty()) continue;
        long j = idxof(nd.input_ids[0]);
        if (j >= 0 && metal_retype_plan_[(size_t)j] != 0) metal_retype_plan_[i] = 1;
    }
}

void CactusGraph::execute(const std::string& profile_file) {
    cactus_graph_mark_unadjusted();
    BufferPool& pool = buffer_pool_;
    const size_t n = nodes_.size();
    infer_shapes();

    auto get_env_int = [](const char* name, int fallback) -> int {
        const char* val = std::getenv(name);
        return val ? std::atoi(val) : fallback;
    };

    bool trace_execution = get_env_int("CACTUS_TRACE_EXECUTE", 0) != 0;
    static const size_t flush_cadence = (size_t)get_env_int("CACTUS_FLUSH_CADENCE", 48);
    bool trace_nan = get_env_int("CACTUS_TRACE_NAN", 0) != 0;
    bool need_debug = !profile_file.empty();
    if (!need_debug) {
        static const bool env_debug = check_debug_env();
        need_debug = env_debug;
    }
    if (trace_execution) {
        need_debug = true;
    }

    auto trace_nonfinite = [&](size_t node_idx, const GraphNode& node) {
        if (!trace_nan) return;
        const BufferDesc& buffer = node.output_buffer;
        const void* data = buffer.get_data();
        if (!data || buffer.total_size == 0) return;

        auto report = [&](size_t elem_idx, float value) {
            std::cerr << "[cactus:nan] idx=" << node_idx
                      << " id=" << node.id
                      << " op=" << get_op_name(node.op_type)
                      << " elem=" << elem_idx
                      << " value=" << value
                      << " shape=[";
            for (size_t dim_idx = 0; dim_idx < buffer.shape.size(); ++dim_idx) {
                if (dim_idx > 0) std::cerr << ",";
                std::cerr << buffer.shape[dim_idx];
            }
            std::cerr << "]" << std::endl;
        };

        if (buffer.precision == Precision::FP16) {
            const __fp16* values = buffer.data_as<__fp16>();
            for (size_t i = 0; i < buffer.total_size; ++i) {
                float value = static_cast<float>(values[i]);
                if (!std::isfinite(value)) {
                    report(i, value);
                    return;
                }
            }
        } else if (buffer.precision == Precision::FP32) {
            const float* values = buffer.data_as<float>();
            for (size_t i = 0; i < buffer.total_size; ++i) {
                float value = values[i];
                if (!std::isfinite(value)) {
                    report(i, value);
                    return;
                }
            }
        }
    };

    auto can_release_node = [&](size_t node_idx) {
        const auto& node = nodes_[node_idx];
        if (node->op_type == OpType::INPUT) return false;
        if (node->op_type == OpType::KV_CACHE_STATE
            || node->op_type == OpType::CONV_CACHE_STATE
            || node->op_type == OpType::RECURRENT_CACHE_STATE) return false;
        if (persistent_node_ids_.count(node->id)) return false;
        if (retained_output_node_ids_.count(node->id)) return false;
        return true;
    };

    bool metal_mode = cactus_backend_metal();
    if (metal_mode && n < 100) {
        for (size_t i = 0; i < n; ++i) {
            OpType t = nodes_[i]->op_type;
            if (t == OpType::LSTM_CELL || t == OpType::BILSTM_SEQUENCE) {
                metal_mode = false;
                break;
            }
        }
    }
    if (metal_mode && !need_debug && !metal_retype_built_) build_metal_retype_plan();
    MetalFusePlan* fplan = nullptr;
    if (metal_mode && !need_debug) {
        uint64_t sig = 1469598103934665603ull;
        sig ^= (uint64_t)n; sig *= 1099511628211ull;
        for (auto& np : nodes_) {
            sig ^= (uint64_t)np->op_type + 0x9e3779b97f4a7c15ull;
            sig *= 1099511628211ull;
            if (np->op_type != OpType::INPUT) continue;
            for (size_t d : np->output_buffer.shape) {
                sig ^= (uint64_t)d + 0x9e3779b97f4a7c15ull;
                sig *= 1099511628211ull;
            }
        }
        auto it = metal_plans_.find(sig);
        if (it == metal_plans_.end()) {
            std::unordered_set<size_t> pinned(retained_output_node_ids_.begin(), retained_output_node_ids_.end());
            pinned.insert(persistent_node_ids_.begin(), persistent_node_ids_.end());
            static const std::vector<uint8_t> no_retype;
            const std::vector<uint8_t>& rt = (!metal_retype_disabled_ && metal_retype_plan_.size() == n)
                ? metal_retype_plan_ : no_retype;
            auto bit = metal_plan_banned_.find(sig);
            const std::unordered_set<size_t>* banned =
                bit == metal_plan_banned_.end() ? nullptr : &bit->second;
            it = metal_plans_.emplace(sig, cactus_metal_plan_build(nodes_, node_index_map_, pinned, rt, banned)).first;
        }
        fplan = it->second;
        metal_plan_sig_ = sig;
    }
    const uint8_t* rplan = (metal_mode && !need_debug && !metal_retype_disabled_
                            && metal_retype_plan_.size() == n) ? metal_retype_plan_.data() : nullptr;
    auto plan_of = [&](const GraphNode& node) -> uint8_t {
        if (!rplan) return 0;
        auto it = node_index_map_.find(node.id);
        return it == node_index_map_.end() ? 0 : rplan[it->second];
    };
    auto aliases_input = [&](const GraphNode& node) -> bool {
        if (plan_of(node) == 1) return true;
        if (node.op_type == OpType::SLICE && !node.input_ids.empty()) {
            auto it = node_index_map_.find(node.input_ids[0]);
            if (it != node_index_map_.end()) {
                const auto& ish = nodes_[it->second]->output_buffer.shape;
                size_t ax = static_cast<size_t>(node.params.axis);
                if (ax < ish.size()) {
                    size_t outer = 1;
                    for (size_t d = 0; d < ax; ++d) outer *= ish[d];
                    if (outer == 1) return true;
                }
            }
        }
        if (node.op_type == OpType::INDEX && node.params.axis == 0) return true;
        if (!metal_mode) return false;
        if (node.op_type == OpType::VIEW || node.op_type == OpType::RESHAPE || node.op_type == OpType::FLATTEN)
            return true;
        if (node.op_type == OpType::PRECISION_CAST && !node.input_ids.empty()) {
            auto it = node_index_map_.find(node.input_ids[0]);
            if (it != node_index_map_.end() &&
                nodes_[it->second]->output_buffer.precision == node.output_buffer.precision) return true;
        }
        return false;
    };
    auto preallocates_output = [&](const GraphNode& node) { return !aliases_input(node); };

    std::vector<size_t> last_use(n, 0);
    std::vector<size_t> use_count(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t input_id : nodes_[i]->input_ids) {
            auto it = node_index_map_.find(input_id);
            if (it != node_index_map_.end()) {
                last_use[it->second] = std::max(last_use[it->second], i);
                ++use_count[it->second];
            }
        }
    }

    std::vector<bool> keep_until_graph_cleanup(n, false);
    for (size_t i = n; i-- > 0;) {
        const auto& node = *nodes_[i];
        if (!aliases_input(node) || node.input_ids.empty()) continue;
        auto it = node_index_map_.find(node.input_ids[0]);
        if (it == node_index_map_.end()) continue;
        size_t base_idx = it->second;
        if (use_count[i] == 0 || keep_until_graph_cleanup[i]) {
            keep_until_graph_cleanup[base_idx] = true;
        } else {
            last_use[base_idx] = std::max(last_use[base_idx], last_use[i]);
        }
    }

    if (fplan) cactus_metal_plan_extend_last_use(fplan, last_use);

    std::vector<std::vector<size_t>> release_after(n);
    for (size_t i = 0; i < n; ++i) {
        if (!can_release_node(i) || use_count[i] == 0 || keep_until_graph_cleanup[i]) continue;
        release_after[last_use[i]].push_back(i);
    }

    if (metal_mode && !need_debug) {
        const bool transient_acts = n >= 1500 && !cactus_metal_plan_has_arena(fplan);
        std::vector<void*> transient_ptr(transient_acts ? n : 0, nullptr);
        std::vector<void*> transient_dead;
        std::vector<uint8_t> transient_ok(transient_acts ? n : 0, 0);
        auto release_transients = [&]() {
            for (void* p : transient_dead) cactus_metal_free_shared(p);
            transient_dead.clear();
            for (size_t i = 0; i < transient_ptr.size(); ++i)
                if (transient_ptr[i]) { cactus_metal_free_shared(transient_ptr[i]); transient_ptr[i] = nullptr; }
        };
        struct CacheWordSnap { size_t idx; size_t word; uint64_t value; };
        std::vector<CacheWordSnap> kv_snapshot;
        auto kv_restore = [&]() {
            for (const auto& s : kv_snapshot) {
                uint64_t* km = reinterpret_cast<uint64_t*>(nodes_[s.idx]->output_buffer.get_data());
                if (km) km[s.word] = s.value;
            }
        };
        auto metal_abort_cleanup = [&]() {
            cactus_metal_session_sync();
            kv_restore();
            release_transients();
            cactus_metal_set_active(false);
            cactus_metal_session_end();
        };
        struct MetalExecGuard {
            decltype(metal_abort_cleanup)& cleanup;
            bool armed = true;
            ~MetalExecGuard() { if (armed) cleanup(); }
        } metal_guard{metal_abort_cleanup};
        cactus_metal_session_begin();
        cactus_metal_set_active(true);
        if (transient_acts) {
            for (size_t i = 0; i < n; ++i) {
                size_t id = nodes_[i]->id;
                transient_ok[i] = !retained_output_node_ids_.count(id)
                    && !persistent_node_ids_.count(id)
                    && !keep_until_graph_cleanup[i]
                    && nodes_[i]->output_buffer.byte_size >= (256u << 10);
            }
        }
        auto metal_release = [&](size_t idx) {
            GraphNode& nd = *nodes_[idx];
            if (aliases_input(nd)) nd.output_buffer.external_data = nullptr;
            if (transient_acts && transient_ptr[idx]) {
                transient_dead.push_back(transient_ptr[idx]);
                transient_ptr[idx] = nullptr;
            }
        };
        auto assign_persistent_act = [&](GraphNode& nd) {
            size_t need = nd.output_buffer.byte_size;
            auto pit = metal_persistent_acts_.find(nd.id);
            void* p = nullptr;
            if (pit != metal_persistent_acts_.end() && pit->second.second >= need) {
                p = pit->second.first;
            } else {
                if (pit != metal_persistent_acts_.end()) cactus_metal_free_shared(pit->second.first);
                p = cactus_metal_alloc_pooled(need);
                if (p) metal_persistent_acts_[nd.id] = { p, need };
            }
            if (p) { nd.output_buffer.release_to_pool(pool); nd.output_buffer.set_external(p); }
            else nd.output_buffer.resize_from_pool(pool);
        };
        size_t since_flush = 0, since_recycle = 0;
        if (fplan) cactus_metal_plan_fold(fplan, nodes_);
        std::vector<uint8_t> metal_live(n, 0);
        auto maybe_recycle = [&]() {
            if (!transient_acts || ++since_recycle < 256 || transient_dead.empty()) return;
            for (void* p : transient_dead) cactus_metal_free_shared(p);
            transient_dead.clear();
            cactus_metal_session_sync();
            std::fill(metal_live.begin(), metal_live.end(), 0);
            since_flush = 0;
            since_recycle = 0;
        };
        for (size_t i = 0; i < n; ++i) {
            const GraphNode& nd = *nodes_[i];
            bool kv = nd.op_type == OpType::KV_CACHE_APPEND;
            bool conv = nd.op_type == OpType::CONV_CACHE_APPEND;
            if ((!kv && !conv) || nd.input_ids.size() < 2) continue;
            auto ci = node_index_map_.find(nd.input_ids[1]);
            if (ci == node_index_map_.end()) continue;
            const uint64_t* km = reinterpret_cast<const uint64_t*>(nodes_[ci->second]->output_buffer.get_data());
            if (!km) continue;
            kv_snapshot.push_back({ci->second, 0, km[0]});
            if (conv) kv_snapshot.push_back({ci->second, 1, km[1]});
        }
        auto input_metal_live = [&](const GraphNode& nd) -> bool {
            for (size_t id : nd.input_ids) {
                auto it = node_index_map_.find(id);
                if (it != node_index_map_.end() && metal_live[it->second]) return true;
            }
            return false;
        };
        const std::vector<uint32_t>* elist = fplan ? cactus_metal_plan_exec_list(fplan) : nullptr;
        size_t iter_count = elist ? elist->size() : n;
        if (fplan) {
            for (size_t ii = 0; ii < iter_count; ++ii) {
                size_t i = elist ? (size_t)(*elist)[ii] : ii;
                if (cactus_metal_plan_action(fplan, i) == -3) assign_persistent_act(*nodes_[i]);
            }
        }
        for (size_t ii = 0; ii < iter_count; ++ii) {
            size_t i = elist ? (size_t)(*elist)[ii] : ii;
            auto& node = nodes_[i];
            const OpType ot = node->op_type;
            if (ot == OpType::INPUT) continue;
            if (ot == OpType::KV_CACHE_STATE || ot == OpType::CONV_CACHE_STATE
                || ot == OpType::RECURRENT_CACHE_STATE) {
                dispatch_node(*node, nodes_, node_index_map_);
                populated_node_ids_.insert(node->id);
                for (size_t r : release_after[i]) metal_release(r);
                continue;
            }
            int32_t fact = fplan ? cactus_metal_plan_action(fplan, i) : -1;
            if (fact == -3) {
                assign_persistent_act(*node);
                metal_live[i] = 1;
                for (size_t r : release_after[i]) metal_release(r);
                continue;
            }
            if (aliases_input(*node)) {
                if (rplan && rplan[i] == 1) {
                    const auto& src = get_input(*node, 0, nodes_, node_index_map_);
                    node->output_buffer.set_external(const_cast<void*>(static_cast<const void*>(src.get_data())));
                } else {
                    dispatch_node(*node, nodes_, node_index_map_);
                }
                metal_live[i] = input_metal_live(*node) ? 1 : 0;
                for (size_t r : release_after[i]) metal_release(r);
                continue;
            }
            if (preallocates_output(*node)) {
                void* ap = fplan ? cactus_metal_plan_arena_ptr(fplan, i) : nullptr;
                if (ap) {
                    node->output_buffer.release_to_pool(pool);
                    node->output_buffer.set_external(ap);
                } else if (transient_acts && transient_ok[i]) {
                    void* tp = cactus_metal_alloc_shared(node->output_buffer.byte_size);
                    if (tp) {
                        node->output_buffer.release_to_pool(pool);
                        node->output_buffer.set_external(tp);
                        transient_ptr[i] = tp;
                    } else {
                        assign_persistent_act(*node);
                    }
                } else {
                    assign_persistent_act(*node);
                }
            }
            if (fact >= 0) {
                if (cactus_metal_plan_encode(fplan, fact, nodes_, node_index_map_)) {
                    metal_live[i] = 1;
                    for (size_t r : release_after[i]) metal_release(r);
                    if (++since_flush >= flush_cadence) { cactus_metal_session_flush(); since_flush = 0; }
                    maybe_recycle();
                    continue;
                }
                metal_guard.armed = false;
                metal_abort_cleanup();
                metal_plan_banned_[metal_plan_sig_].insert(i);
                metal_plans_.erase(metal_plan_sig_);
                cactus_metal_plan_free(fplan);
                execute(profile_file);
                return;
            }
            std::vector<std::pair<BufferDesc*, Precision>> flipped;
            if (rplan && rplan[i] == 2) {
                auto flip = [&](BufferDesc& b) {
                    if (b.precision == Precision::FP32) { flipped.push_back({&b, b.precision}); b.precision = Precision::FP16; }
                };
                flip(node->output_buffer);
                for (size_t id : node->input_ids) {
                    auto it = node_index_map_.find(id);
                    if (it != node_index_map_.end() && rplan[it->second] != 0) flip(nodes_[it->second]->output_buffer);
                }
            }
            bool force_cpu = (ot == OpType::PRECISION_CAST
                && node->output_buffer.total_size <= 8 && !input_metal_live(*node))
                || (ot == OpType::EMBEDDING && input_metal_live(*node));
            bool encoded = !force_cpu && try_encode_metal(*node, nodes_, node_index_map_);
            for (auto& fp : flipped) fp.first->precision = fp.second;
            if (!encoded && rplan && rplan[i] == 2) {
                metal_retype_disabled_ = true;
                metal_guard.armed = false;
                metal_abort_cleanup();
                execute(profile_file);
                return;
            }
            if (!encoded) {
                if (input_metal_live(*node)) {
                    cactus_metal_session_sync();
                    std::fill(metal_live.begin(), metal_live.end(), 0);
                }
                dispatch_node(*node, nodes_, node_index_map_);
            } else {
                metal_live[i] = 1;
            }
            if (ot == OpType::PERSISTENT) populated_node_ids_.insert(node->id);
            for (size_t r : release_after[i]) metal_release(r);
            if (encoded && ++since_flush >= flush_cadence) { cactus_metal_session_flush(); since_flush = 0; }
            if (encoded) maybe_recycle();
        }
        metal_guard.armed = false;
        release_transients();
        cactus_metal_set_active(false);
        cactus_metal_session_end();
        cactus_graph_metal_tail_commit();
        return;
    }

    if (!need_debug) {
        auto run = [&](auto& nd) { dispatch_node(nd, nodes_, node_index_map_); };
        for (size_t i = 0; i < n; ++i) {
            auto& node = nodes_[i];
            if (node->op_type == OpType::INPUT) continue;
            if (node->op_type == OpType::KV_CACHE_STATE
                || node->op_type == OpType::CONV_CACHE_STATE
                || node->op_type == OpType::RECURRENT_CACHE_STATE) {
                run(*node);
                populated_node_ids_.insert(node->id);
                for (size_t release_idx : release_after[i]) {
                    nodes_[release_idx]->output_buffer.release_memory(pool);
                }
                continue;
            }
            if (preallocates_output(*node)) {
                node->output_buffer.resize_from_pool(pool);
            }
            run(*node);
            trace_nonfinite(i, *node);
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }
            for (size_t release_idx : release_after[i]) {
                nodes_[release_idx]->output_buffer.release_memory(pool);
            }
        }
        return;
    }

    auto get_env_str = [](const char* name) -> std::string {
        const char* val = std::getenv(name);
        return val ? std::string(val) : std::string();
    };

    bool capture_to_stdout = get_env_int("CACTUS_CAPTURE_STDOUT", 0) != 0;
    std::string capture_file_path = get_env_str("CACTUS_CAPTURE_FILE");
    bool capture_requested = get_env_int("CACTUS_CAPTURE_ENABLE", 0) != 0;
    std::string capture_dir = get_env_str("CACTUS_CAPTURE_DIR");

    if (!capture_requested) {
        capture_requested = capture_to_stdout || !capture_file_path.empty() || !capture_dir.empty();
    } else if (capture_file_path.empty() && !capture_to_stdout && capture_dir.empty()) {
        capture_to_stdout = true;
    }

    size_t capture_preview_count = static_cast<size_t>(get_env_int("CACTUS_CAPTURE_PREVIEW_COUNT", 8));
    size_t capture_max_elements = static_cast<size_t>(get_env_int("CACTUS_CAPTURE_MAX_ELEMENTS", 65536));

    std::string env_profile = get_env_str("CACTUS_PROFILE_FILE");
    if (env_profile.empty()) env_profile = get_env_str("CACTUS_PROFILE");

    std::string target_profile = profile_file;
    if (target_profile.empty() && !env_profile.empty()) {
        target_profile = env_profile;
    }

    bool enable_profiling = !target_profile.empty();
    bool to_stdout = (target_profile == "stdout" || target_profile == "-");

    std::ofstream profile_out;
    std::ostream* out = &std::cout;

    if (enable_profiling && !to_stdout) {
        profile_out.open(target_profile, std::ios::app);
        if (profile_out.is_open()) {
            out = &profile_out;
        }
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    if (enable_profiling) {
        *out << "=== Graph Execution Profile ===" << std::endl;
        *out << std::left << std::setw(24) << "Operation"
             << std::setw(12) << "Time (ms)"
             << std::setw(20) << "Output Shape"
             << "Backend" << std::endl;
        *out << std::string(72, '-') << std::endl;
    }

    for (size_t node_idx = 0; node_idx < n; ++node_idx) {
        auto& node = nodes_[node_idx];

        if (node->op_type == OpType::INPUT) {
            continue;
        }

        if (node->op_type == OpType::KV_CACHE_STATE
            || node->op_type == OpType::CONV_CACHE_STATE
            || node->op_type == OpType::RECURRENT_CACHE_STATE) {
            dispatch_node(*node, nodes_, node_index_map_);
            if (trace_execution) {
                std::cerr << "[cactus:execute] cache-state idx=" << node_idx
                          << " id=" << node->id
                          << " op=" << get_op_name(node->op_type)
                          << std::endl;
            }
            trace_nonfinite(node_idx, *node);
            populated_node_ids_.insert(node->id);
            continue;
        }

        node->output_buffer.allocate_from_pool(pool);

        if (trace_execution) {
            std::cerr << "[cactus:execute] begin idx=" << node_idx
                      << " id=" << node->id
                      << " op=" << get_op_name(node->op_type)
                      << " shape=[";
            for (size_t dim_idx = 0; dim_idx < node->output_buffer.shape.size(); ++dim_idx) {
                if (dim_idx > 0) std::cerr << ",";
                std::cerr << node->output_buffer.shape[dim_idx];
            }
            std::cerr << "]" << std::endl;
        }

        if (enable_profiling) {
            auto start = std::chrono::high_resolution_clock::now();
            dispatch_node(*node, nodes_, node_index_map_);
            trace_nonfinite(node_idx, *node);
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            std::string shape_str = "[";
            for (size_t i = 0; i < node->output_buffer.shape.size(); ++i) {
                if (i > 0) shape_str += ",";
                shape_str += std::to_string(node->output_buffer.shape[i]);
            }
            shape_str += "]";

            *out << std::left << std::setw(24) << get_op_name(node->op_type)
                 << std::setw(12) << std::fixed << std::setprecision(3) << ms
                 << std::setw(20) << shape_str << std::endl;
        } else {
            dispatch_node(*node, nodes_, node_index_map_);
            trace_nonfinite(node_idx, *node);
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }
        }

        if (trace_execution) {
            std::cerr << "[cactus:execute] done idx=" << node_idx
                      << " id=" << node->id
                      << " op=" << get_op_name(node->op_type)
                      << std::endl;
        }
    }

    std::unique_ptr<std::ofstream> capture_file_stream;
    std::vector<std::ostream*> capture_outputs;

    if (capture_requested) {
        if (capture_to_stdout) {
            capture_outputs.push_back(&std::cout);
        }

        if (!capture_file_path.empty()) {
            std::filesystem::path capture_path(capture_file_path);
            if (capture_path.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(capture_path.parent_path(), ec);
            }

            auto stream_ptr = std::make_unique<std::ofstream>(capture_path, std::ios::out | std::ios::app);
            if (stream_ptr->is_open()) {
                capture_outputs.push_back(stream_ptr.get());
                capture_file_stream = std::move(stream_ptr);
            } else {
                std::cerr << "Failed to open capture file: " << capture_path << std::endl;
            }
        }

        if (!capture_dir.empty()) {
            std::filesystem::path dir_path(capture_dir);
            std::error_code ec;
            std::filesystem::create_directories(dir_path, ec);
        }

        if (capture_outputs.empty() && capture_dir.empty()) {
            capture_requested = false;
        }
    }

    if (capture_requested) {
        auto precision_to_string = [](Precision p) -> const char* {
            switch (p) {
                case Precision::FP32: return "FP32";
                case Precision::FP16: return "FP16";
                case Precision::INT8: return "INT8";
                default: return "UNKNOWN";
            }
        };

        auto format_double = [](double value) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << value;
            return oss.str();
        };

        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm time_info{};
#if defined(_WIN32)
        localtime_s(&time_info, &now_time);
#else
        localtime_r(&now_time, &time_info);
#endif

        auto write_header = [&](std::ostream& stream) {
            stream << "=== Graph Debug Capture ===" << std::endl;
            stream << "Timestamp: " << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S") << std::endl;
            stream << "Captured nodes: " << debug_nodes_.size() << std::endl;
            stream << std::string(60, '-') << std::endl;
        };

        auto write_separator = [](std::ostream& stream) {
            stream << std::string(60, '-') << std::endl;
        };

        if (debug_nodes_.empty()) {
            for (auto* stream : capture_outputs) {
                write_header(*stream);
                *stream << "No debug nodes registered on this graph." << std::endl;
                write_separator(*stream);
                stream->flush();
            }
        } else {
            for (auto* stream : capture_outputs) {
                write_header(*stream);
            }

            for (const auto& entry : debug_nodes_) {
                auto node_it = node_index_map_.find(entry.node_id);
                const GraphNode* node_ptr = nullptr;
                if (node_it != node_index_map_.end()) {
                    node_ptr = nodes_[node_it->second].get();
                }

                if (!node_ptr) {
                    for (auto* stream : capture_outputs) {
                        *stream << "Layer " << entry.layer_idx << " - " << entry.name
                                << " (node " << entry.node_id << ")" << std::endl;
                        *stream << "  Data: <unavailable; node not present in graph>" << std::endl;
                        write_separator(*stream);
                    }
                    continue;
                }

                const BufferDesc& buffer = node_ptr->output_buffer;
                const void* data_ptr = buffer.get_data();
                size_t total_size = buffer.total_size;

                std::ostringstream shape_ss;
                shape_ss << "[";
                for (size_t i = 0; i < buffer.shape.size(); ++i) {
                    if (i > 0) {
                        shape_ss << ",";
                    }
                    shape_ss << buffer.shape[i];
                }
                shape_ss << "]";
                std::string shape_str = shape_ss.str();

                bool has_data = data_ptr != nullptr && total_size > 0;
                size_t elements_to_process = total_size;
                bool truncated = false;
                if (has_data && elements_to_process > capture_max_elements && capture_max_elements > 0) {
                    elements_to_process = capture_max_elements;
                    truncated = true;
                }

                std::vector<float> preview_values;
                if (capture_preview_count > 0) {
                    preview_values.reserve(std::min(capture_preview_count, elements_to_process));
                }

                double min_val = std::numeric_limits<double>::infinity();
                double max_val = -std::numeric_limits<double>::infinity();
                long double sum = 0.0L;
                long double sum_sq = 0.0L;

                if (has_data && elements_to_process > 0) {
                    auto accumulate = [&](float value, size_t index) {
                        double v = static_cast<double>(value);
                        min_val = std::min(min_val, v);
                        max_val = std::max(max_val, v);
                        sum += static_cast<long double>(value);
                        sum_sq += static_cast<long double>(value) * static_cast<long double>(value);
                        if (capture_preview_count > 0 && index < capture_preview_count) {
                            preview_values.push_back(value);
                        }
                    };

                    if (buffer.precision == Precision::FP32) {
                        const float* typed = static_cast<const float*>(data_ptr);
                        for (size_t i = 0; i < elements_to_process; ++i) {
                            accumulate(typed[i], i);
                        }
                    } else if (buffer.precision == Precision::FP16) {
                        const __fp16* typed = reinterpret_cast<const __fp16*>(data_ptr);
                        for (size_t i = 0; i < elements_to_process; ++i) {
                            accumulate(static_cast<float>(typed[i]), i);
                        }
                    } else if (buffer.precision == Precision::INT8) {
                        const int8_t* typed = reinterpret_cast<const int8_t*>(data_ptr);
                        for (size_t i = 0; i < elements_to_process; ++i) {
                            accumulate(static_cast<float>(typed[i]), i);
                        }
                    } else {
                        has_data = false;
                    }
                } else {
                    has_data = false;
                }

                if (!capture_dir.empty() && has_data) {
                    std::string safe_name = entry.name;
                    std::string filename = capture_dir + "/" + safe_name + ".bin";
                    std::ofstream bin_file(filename, std::ios::binary);
                    if (bin_file.is_open()) {
                        size_t bytes_to_write = buffer.byte_size;
                        if (truncated) {
                             bytes_to_write = PrecisionTraits::packed_size_of(buffer.precision, elements_to_process);
                        }
                        bin_file.write(reinterpret_cast<const char*>(data_ptr), bytes_to_write);
                    }
                }

                size_t processed_count = has_data ? elements_to_process : 0;
                long double mean_ld = processed_count > 0 ? sum / processed_count : 0.0L;
                long double variance_ld = processed_count > 0 ? (sum_sq / processed_count) - (mean_ld * mean_ld) : 0.0L;
                if (variance_ld < 0.0L) {
                    variance_ld = 0.0L;
                }
                double mean_val = static_cast<double>(mean_ld);
                double stddev_val = processed_count > 0 ? std::sqrt(static_cast<double>(variance_ld)) : 0.0;

                std::ostringstream preview_ss;
                if (capture_preview_count > 0 && !preview_values.empty()) {
                    preview_ss << "[";
                    for (size_t i = 0; i < preview_values.size(); ++i) {
                        if (i > 0) {
                            preview_ss << ", ";
                        }
                        preview_ss << format_double(static_cast<double>(preview_values[i]));
                    }
                    if (processed_count > preview_values.size()) {
                        if (!preview_values.empty()) {
                            preview_ss << ", ...";
                        } else {
                            preview_ss << "...";
                        }
                    }
                    preview_ss << "]";
                }

                for (auto* stream : capture_outputs) {
                    *stream << "Layer " << entry.layer_idx << " - " << entry.name
                            << " (node " << entry.node_id << ")" << std::endl;
                    *stream << "  Shape: " << shape_str << "  Precision: " << precision_to_string(buffer.precision) << std::endl;
                    if (!has_data) {
                        *stream << "  Data: <unavailable>" << std::endl;
                    } else {
                        *stream << "  Stats: min=" << format_double(min_val)
                                << " max=" << format_double(max_val)
                                << " mean=" << format_double(mean_val)
                                << " std=" << format_double(stddev_val) << std::endl;
                        if (truncated || processed_count < total_size) {
                            *stream << "  Note: stats computed on first " << processed_count
                                    << " of " << total_size << " values" << std::endl;
                        }
                        if (capture_preview_count > 0 && !preview_values.empty()) {
                            *stream << "  Preview: " << preview_ss.str() << std::endl;
                        }
                    }
                    write_separator(*stream);
                }
            }

            for (auto* stream : capture_outputs) {
                stream->flush();
            }
        }
    }

    if (enable_profiling) {
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        double total_ms = total_duration.count() / 1000.0;

        *out << std::string(72, '-') << std::endl;
        *out << "Total execution time: " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
        *out << "================================" << std::endl;

        if (profile_out.is_open()) {
            profile_out.close();
        }
    }
}

void CactusGraph::invalidate_metal_state() {
    for (auto& kv : metal_plans_) cactus_metal_plan_free(kv.second);
    metal_plans_.clear();
    for (auto& kv : metal_persistent_acts_) cactus_metal_free_shared(kv.second.first);
    metal_persistent_acts_.clear();
    metal_retype_plan_.clear();
    metal_retype_built_ = false;
}

void CactusGraph::hard_reset() {
    invalidate_metal_state();
    nodes_.clear();
    node_index_map_.clear();
    mapped_files_.clear();
    weight_cache_.clear();
    next_node_id_ = 1;
    debug_nodes_.clear();
    buffer_pool_.clear();
}

void CactusGraph::soft_reset() {
    invalidate_metal_state();
    std::set<size_t> cached_node_ids;
    for (const auto& cache_entry : weight_cache_) {
        cached_node_ids.insert(cache_entry.second);
    }
    
    for (size_t pid : persistent_node_ids_) {
        cached_node_ids.insert(pid);
    }

    size_t max_preserved_id = 0;
    for (const auto& node : nodes_) {
        if ((node->op_type == OpType::INPUT && node->output_buffer.external_data) ||
            cached_node_ids.count(node->id)) {
            max_preserved_id = std::max(max_preserved_id, node->id);
        }
    }

    auto preserved_nodes = std::move(nodes_);
    auto preserved_index_map = std::move(node_index_map_);

    nodes_.clear();
    node_index_map_.clear();

    for (auto& node : preserved_nodes) {
        if ((node->op_type == OpType::INPUT && node->output_buffer.external_data) ||
            cached_node_ids.count(node->id)) {
            size_t index = nodes_.size();
            node_index_map_[node->id] = index;
            nodes_.push_back(std::move(node));
        }
    }

    next_node_id_ = max_preserved_id + 1;
    debug_nodes_.clear();
    if (!prefill_mode_) {
        buffer_pool_.clear();
        shrink_thread_local_buffers();
    }
}

void CactusGraph::soft_reset_keep_pool() {
    invalidate_metal_state();
    std::set<size_t> cached_node_ids;
    for (const auto& cache_entry : weight_cache_) {
        cached_node_ids.insert(cache_entry.second);
    }

    for (size_t pid : persistent_node_ids_) {
        cached_node_ids.insert(pid);
    }

    size_t max_preserved_id = 0;
    for (const auto& node : nodes_) {
        if ((node->op_type == OpType::INPUT && node->output_buffer.external_data) ||
            cached_node_ids.count(node->id)) {
            max_preserved_id = std::max(max_preserved_id, node->id);
        }
    }

    auto preserved_nodes = std::move(nodes_);

    nodes_.clear();
    node_index_map_.clear();

    for (auto& node : preserved_nodes) {
        if ((node->op_type == OpType::INPUT && node->output_buffer.external_data) ||
            cached_node_ids.count(node->id)) {
            size_t index = nodes_.size();
            node_index_map_[node->id] = index;
            nodes_.push_back(std::move(node));
        }
    }

    next_node_id_ = max_preserved_id + 1;
    debug_nodes_.clear();
}

void CactusGraph::prewarm_metal_quant_weights() {
    if (!cactus_backend_metal()) return;
    for (auto& np : nodes_) {
        GraphNode& node = *np;
        if (node.op_type != OpType::MATMUL || node.input_ids.size() < 2) continue;
        auto it = node_index_map_.find(node.input_ids[1]);
        if (it == node_index_map_.end()) continue;
        const BufferDesc& rhs = nodes_[it->second]->output_buffer;
        if (!(PrecisionTraits::is_cq(rhs.precision) && rhs.group_size > 0)) continue;
        CactusQuantMatrix mat = rhs.to_cq_matrix();
        cactus_metal_prewarm_quant(&mat);
    }
}
