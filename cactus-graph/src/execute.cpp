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
#include <map>
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
bool cactus_backend_fused() { return g_backend_override != 0 && cactus_metal_available(); }

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
bool cactus_kv_cache_grow(BufferDesc&, size_t, size_t);
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

static bool gpu_op_enabled(const char* name) {
    static const std::string spec = []{ const char* e=std::getenv("CACTUS_GPU_OPS"); return e?std::string(e):std::string(); }();
    if (spec.empty()) return true;
    return spec.find(name) != std::string::npos;
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

static bool try_encode_gpu(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& map) {
    BufferDesc& out = node.output_buffer;
    auto fp16 = [](const BufferDesc& b){ return b.precision == Precision::FP16; };
    if (std::getenv("CACTUS_GPU_FALLBACK")) return false;
    {
        const char* g = nullptr;
        switch (node.op_type) {
            case OpType::VIEW: case OpType::RESHAPE: case OpType::FLATTEN: g="copy"; break;
            case OpType::MATMUL: g="matmul"; break;
            case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
            case OpType::MULTIPLY: case OpType::DIVIDE: g="binary"; break;
            case OpType::SCALAR_ADD: case OpType::SCALAR_SUBTRACT:
            case OpType::SCALAR_MULTIPLY: case OpType::SCALAR_DIVIDE: g="scalar"; break;
            case OpType::GELU: case OpType::TANH: case OpType::SILU: case OpType::RELU: g="unary"; break;
            case OpType::RMS_NORM: g="rms"; break;
            case OpType::PRECISION_CAST: g="cast"; break;
            case OpType::ATTENTION_CACHED: g="attn"; break;
            case OpType::TRANSPOSE: g="transpose"; break;
            case OpType::SLICE: g="slice"; break;
            case OpType::INDEX: g="index"; break;
            case OpType::CAT: g="cat"; break;
            case OpType::KV_CACHE_APPEND: g="kvappend"; break;
            default: g=nullptr;
        }
        if (g && !gpu_op_enabled(g)) return false;
    }
    switch (node.op_type) {
        case OpType::VIEW: case OpType::RESHAPE: case OpType::FLATTEN: {
            const auto& in = get_input(node, 0, nodes, map);
            if (in.byte_size != out.byte_size) return false;
            return cactus_metal_encode_copy(out.get_data(), in.get_data(), in.byte_size);
        }
        case OpType::MATMUL: {
            const auto& lhs = get_input(node, 0, nodes, map);
            const auto& rhs = get_input(node, 1, nodes, map);
            size_t M = lhs.shape[lhs.shape.size() - 2];
            if (!(PrecisionTraits::is_cq(rhs.precision) && rhs.group_size > 0)) return false;
            if (!fp16(lhs)) return false;
            CactusQuantMatrix mat = rhs.to_cq_matrix();
            if (M == 1) return cactus_metal_encode_quant_matmul(out.get_data(), lhs.get_data(), &mat);
            return cactus_metal_encode_quant_matmul_m(out.get_data(), lhs.get_data(), &mat, (uint32_t)M);
        }
        case OpType::ADD: case OpType::ADD_CLIPPED: case OpType::SUBTRACT:
        case OpType::MULTIPLY: case OpType::DIVIDE: {
            const auto& a = get_input(node, 0, nodes, map);
            const auto& b = get_input(node, 1, nodes, map);
            if (!fp16(a) || !fp16(b) || !fp16(out)) return false;
            int code = node.op_type==OpType::ADD?0: node.op_type==OpType::ADD_CLIPPED?1:
                       node.op_type==OpType::SUBTRACT?2: node.op_type==OpType::MULTIPLY?3:4;
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
            size_t ostr[8]; row_strides(out.shape, ostr);
            size_t axis_off = 0;
            for (size_t ii = 0; ii < node.input_ids.size(); ++ii) {
                const auto& cin = get_input(node, ii, nodes, map);
                if (!fp16(cin) || cin.shape.size() != nd) return false;
                uint32_t ishape[8], ostride[8];
                for (uint32_t d=0; d<nd; ++d){ ishape[d]=(uint32_t)cin.shape[d]; ostride[d]=(uint32_t)ostr[d]; }
                if (!cactus_metal_encode_strided_scatter(out.get_data(), cin.get_data(), ishape, ostride, nd,
                        (uint32_t)cin.total_size, (uint32_t)(axis_off * ostr[axis]), cin.byte_size, out.byte_size))
                    return false;
                axis_off += cin.shape[axis];
            }
            return true;
        }
        case OpType::SCALAR_ADD: case OpType::SCALAR_SUBTRACT:
        case OpType::SCALAR_MULTIPLY: case OpType::SCALAR_DIVIDE: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out)) return false;
            int code = node.op_type==OpType::SCALAR_ADD?0: node.op_type==OpType::SCALAR_SUBTRACT?1:
                       node.op_type==OpType::SCALAR_MULTIPLY?2:3;
            return cactus_metal_encode_scalar(code, out.get_data(), in.get_data(), out.total_size, node.params.scalar);
        }
        case OpType::GELU: case OpType::TANH: case OpType::SILU: case OpType::RELU: {
            const auto& in = get_input(node, 0, nodes, map);
            if (!fp16(in) || !fp16(out)) return false;
            int code = node.op_type==OpType::GELU?0: node.op_type==OpType::TANH?1: node.op_type==OpType::SILU?2:3;
            return cactus_metal_encode_unary(code, out.get_data(), in.get_data(), out.total_size);
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
            if (std::getenv("CACTUS_GPU_DEBUG")) {
                static int dq=0;
                if (qb.shape.size()<2 || qb.shape[1]==1) if (dq++ < 4) std::cerr << "[attn] q.shape.size=" << qb.shape.size()
                    << " dims=[" << (qb.shape.size()>0?qb.shape[0]:0) << "," << (qb.shape.size()>1?qb.shape[1]:0)
                    << "," << (qb.shape.size()>2?qb.shape[2]:0) << "] kc.prec=" << (int)kcache.precision << "\n";
            }
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
            bool ok = cactus_metal_encode_attention_i8(
                out.get_data(), qb.get_data(), knew.get_data(), vnew.get_data(),
                bk + 64, bv + 64,
                bk + 64 + max_seq*kv_heads*hdim, bv + 64 + v_max*kv_heads*v_hdim,
                (uint32_t)nqh, (uint32_t)kv_heads, (uint32_t)hdim, (uint32_t)v_hdim,
                (uint32_t)history_len, (uint32_t)total_keys, (uint32_t)kv_start, (uint32_t)kv_end, scale,
                history_len*kv_heads*hdim, history_len*kv_heads*v_hdim,
                history_len*kv_heads*ngK*sizeof(float), history_len*kv_heads*ngV*sizeof(float));
            if (std::getenv("CACTUS_GPU_DEBUG")) { static int d2=0; if (d2++ < 3)
                std::cerr << "[attn2] nqh="<<nqh<<" kvh="<<kv_heads<<" hd="<<hdim<<" vhd="<<v_hdim
                << " hist="<<history_len<<" tot="<<total_keys<<" ks="<<kv_start<<" ke="<<kv_end
                << " cache_len="<<cache_len<<" -> ok="<<ok<<"\n"; }
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
            if (new_seq_len > 1) {
                size_t ceiling = cache_node.params.max_cache_seq_len, ws = node.params.window_size;
                bool sliding = ws > 0 && ws < ceiling;
                size_t new_total = current_len + new_seq_len;
                char* base = static_cast<char*>(cache.get_data());
                size_t ngK = (hdim + 31)/32;
                if (sliding) {
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
            size_t ceiling = cache_node.params.max_cache_seq_len, ws = node.params.window_size;
            bool sliding = ws > 0 && ws < ceiling;
            size_t new_total = current_len + new_seq_len;
            char* base = static_cast<char*>(cache.get_data());
            size_t ngK = (hdim + 31)/32;
            if (!sliding && new_total > max_len) {
                if (!cactus_kv_cache_grow(cache, new_total, ceiling)) return false;
                km = reinterpret_cast<uint64_t*>(cache.get_data());
                max_len = km[1]; base = static_cast<char*>(cache.get_data());
                if (new_total > max_len) return false;
            }
            size_t window = sliding ? ws : max_len;
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

namespace {
struct CpuOpProf { double ns = 0.0; uint64_t inv = 0; };
std::map<size_t, std::map<int, CpuOpProf>> g_cpu_prof;
std::map<size_t, uint64_t> g_cpu_prof_calls;
void cpu_prof_dump() {
    for (auto& nc : g_cpu_prof) {
        size_t nn = nc.first; uint64_t calls = g_cpu_prof_calls[nn];
        if (!calls) continue;
        double tot = 0.0; for (auto& op : nc.second) tot += op.second.ns;
        std::vector<std::pair<int,CpuOpProf>> v(nc.second.begin(), nc.second.end());
        std::sort(v.begin(), v.end(), [](const std::pair<int,CpuOpProf>& a, const std::pair<int,CpuOpProf>& b){
            return a.second.ns > b.second.ns; });
        std::cerr << "\n=== CPU op profile: graph n=" << nn << " | " << calls << " call(s) | "
                  << std::fixed << std::setprecision(3) << (tot/calls/1e6) << " ms/call total ===\n";
        std::cerr << "  " << std::left << std::setw(24) << "op" << std::right
                  << std::setw(11) << "ms/call" << std::setw(11) << "cnt/call" << std::setw(9) << "share\n";
        for (auto& e : v) {
            std::cerr << "  " << std::left << std::setw(24) << get_op_name((OpType)e.first) << std::right
                      << std::fixed << std::setprecision(4) << std::setw(11) << (e.second.ns/calls/1e6)
                      << std::setprecision(1) << std::setw(11) << ((double)e.second.inv/calls)
                      << std::setw(8) << (100.0*e.second.ns/tot) << "%\n";
        }
    }
}
inline void cpu_prof_begin(size_t n) {
    static bool reg = false;
    if (!reg) { std::atexit(cpu_prof_dump); reg = true; }
    g_cpu_prof_calls[n]++;
}
}

void CactusGraph::execute(const std::string& profile_file) {
    BufferPool& pool = buffer_pool_;
    const size_t n = nodes_.size();
    static const bool gpu_timing = std::getenv("CACTUS_GPU_TIMING") != nullptr;
    auto _tt = [](){ return std::chrono::high_resolution_clock::now(); };
    auto _t0 = _tt();
    infer_shapes();
    auto _t_infer = _tt();

    if (std::getenv("CACTUS_OP_HIST")) {
        static std::unordered_set<size_t> seen_sizes;
        if (seen_sizes.insert(n).second) {
            std::unordered_map<int,int> hist;
            for (size_t i = 0; i < n; ++i) hist[static_cast<int>(nodes_[i]->op_type)]++;
            std::cerr << "[ophist] graph nodes=" << n << "\n";
            for (auto& kv : hist)
                std::cerr << "  " << get_op_name(static_cast<OpType>(kv.first))
                          << " : " << kv.second << "\n";
        }
    }

    if (std::getenv("CACTUS_GRAPH_DUMP")) {
        static std::unordered_set<size_t> dumped;
        if (dumped.insert(n).second) {
            auto shp=[](const BufferDesc&b){ std::string s="["; for(size_t d=0;d<b.shape.size();++d){ if(d)s+="x"; s+=std::to_string(b.shape[d]); } return s+"]"; };
            std::cerr << "[graph-dump] n=" << n << "\n";
            for (size_t i=0;i<n;++i){
                GraphNode& nd=*nodes_[i];
                std::cerr << i << "\t" << get_op_name(nd.op_type) << "\tout=" << shp(nd.output_buffer)
                          << " pr" << (int)nd.output_buffer.precision;
                if (nd.output_buffer.group_size) std::cerr << " gs" << nd.output_buffer.group_size << " cq" << nd.output_buffer.cq_flags;
                std::cerr << " in=[";
                for (size_t j=0;j<nd.input_ids.size();++j){ auto it=node_index_map_.find(nd.input_ids[j]); size_t idx=(it!=node_index_map_.end()?it->second:SIZE_MAX); if(j)std::cerr<<","; std::cerr<<(long)idx; }
                std::cerr << "]";
                if (nd.params.axis) std::cerr << " axis=" << nd.params.axis;
                if (nd.params.epsilon != 0.0f) std::cerr << " eps=" << nd.params.epsilon;
                if (nd.params.scalar != 0.0f) std::cerr << " scalar=" << nd.params.scalar;
                if (!nd.params.permutation.empty()){ std::cerr<<" perm=["; for(size_t d=0;d<nd.params.permutation.size();++d){if(d)std::cerr<<",";std::cerr<<nd.params.permutation[d];} std::cerr<<"]"; }
                std::cerr << "\n";
            }
            std::cerr << "[graph-dump] end\n";
        }
    }

    auto get_env_int = [](const char* name, int fallback) -> int {
        const char* val = std::getenv(name);
        return val ? std::atoi(val) : fallback;
    };

    bool trace_execution = get_env_int("CACTUS_TRACE_EXECUTE", 0) != 0;
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

    const bool gpu_mode = cactus_backend_metal();
    const bool gpu_mode_req = gpu_mode && !g_cactus_force_cpu;
    auto aliases_input = [&](const GraphNode& node) -> bool {
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
        if (!gpu_mode_req) return false;
        if (node.op_type == OpType::VIEW || node.op_type == OpType::RESHAPE || node.op_type == OpType::FLATTEN)
            return true;
        if (node.op_type == OpType::PRECISION_CAST && !node.input_ids.empty()) {
            auto it = node_index_map_.find(node.input_ids[0]);
            if (it != node_index_map_.end() &&
                nodes_[it->second]->output_buffer.precision == node.output_buffer.precision) return true;
        }
        return false;
    };
    auto may_alias_input = aliases_input;
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
    for (size_t i = 0; i < n; ++i) {
        const auto& node = *nodes_[i];
        if (!may_alias_input(node) || node.input_ids.empty()) continue;
        auto it = node_index_map_.find(node.input_ids[0]);
        if (it == node_index_map_.end()) continue;
        size_t base_idx = it->second;
        if (use_count[i] == 0) {
            keep_until_graph_cleanup[base_idx] = true;
        } else {
            last_use[base_idx] = std::max(last_use[base_idx], last_use[i]);
        }
    }

    std::vector<std::vector<size_t>> release_after(n);
    for (size_t i = 0; i < n; ++i) {
        if (!can_release_node(i) || use_count[i] == 0 || keep_until_graph_cleanup[i]) continue;
        release_after[last_use[i]].push_back(i);
    }

    static const bool gpu_verify = std::getenv("CACTUS_GPU_VERIFY") != nullptr;
    if (gpu_mode_req && !need_debug) {
        auto _t_live = _tt();
        cactus_metal_session_begin();
        cactus_metal_set_active(true);
        std::unordered_map<int,int> gpu_fallback_hist;
        auto gpu_release = [&](size_t idx) {
            GraphNode& nd = *nodes_[idx];
            if (aliases_input(nd)) nd.output_buffer.external_data = nullptr;
        };
        const bool concurrent = cactus_metal_concurrent();
        struct Acc { uintptr_t lo, hi; bool w; };
        std::vector<Acc> accessed;
        auto conflicts = [&](uintptr_t lo, uintptr_t hi, bool w) {
            for (auto& a : accessed) if ((w || a.w) && lo < a.hi && a.lo < hi) return true;
            return false;
        };
        auto hazard_barrier = [&](GraphNode& nd) {
            if (!concurrent) return;
            const bool allW = (nd.op_type == OpType::KV_CACHE_APPEND);
            std::vector<Acc> mine; bool hit = false;
            for (size_t j = 0; j < nd.input_ids.size(); ++j) {
                const auto& ib = get_input(nd, j, nodes_, node_index_map_);
                if (!ib.get_data()) continue;
                uintptr_t lo = (uintptr_t)ib.get_data(), hi = lo + ib.byte_size;
                if (conflicts(lo, hi, allW)) hit = true;
                mine.push_back({lo, hi, allW});
            }
            if (nd.output_buffer.get_data()) {
                uintptr_t lo = (uintptr_t)nd.output_buffer.get_data(), hi = lo + nd.output_buffer.byte_size;
                if (conflicts(lo, hi, true)) hit = true;
                mine.push_back({lo, hi, true});
            }
            if (hit) { cactus_metal_barrier(); accessed.clear(); }
            for (auto& m : mine) accessed.push_back(m);
        };
        for (size_t i = 0; i < n; ++i) {
            auto& node = nodes_[i];
            const OpType ot = node->op_type;
            if (ot == OpType::INPUT) continue;
            if (ot == OpType::KV_CACHE_STATE || ot == OpType::CONV_CACHE_STATE
                || ot == OpType::RECURRENT_CACHE_STATE) {
                dispatch_node(*node, nodes_, node_index_map_);
                populated_node_ids_.insert(node->id);
                for (size_t r : release_after[i]) gpu_release(r);
                continue;
            }
            if (aliases_input(*node)) {
                dispatch_node(*node, nodes_, node_index_map_);
                for (size_t r : release_after[i]) gpu_release(r);
                continue;
            }
            if (preallocates_output(*node)) {
                size_t need = node->output_buffer.byte_size;
                auto pit = gpu_persistent_acts_.find(node->id);
                void* p = nullptr;
                if (pit != gpu_persistent_acts_.end() && pit->second.second >= need) {
                    p = pit->second.first;
                } else {
                    if (pit != gpu_persistent_acts_.end()) cactus_metal_free_shared(pit->second.first);
                    p = cactus_metal_alloc_shared(need);
                    if (p) gpu_persistent_acts_[node->id] = { p, need };
                }
                if (p) { node->output_buffer.release_to_pool(pool); node->output_buffer.set_external(p); }
                else node->output_buffer.resize_from_pool(pool);
            }
            hazard_barrier(*node);
            bool encoded = try_encode_gpu(*node, nodes_, node_index_map_);
            if (!encoded) {
                gpu_fallback_hist[(int)ot]++;
                cactus_metal_session_sync();
                accessed.clear();
                dispatch_node(*node, nodes_, node_index_map_);
            } else if (gpu_verify) {
                cactus_metal_session_sync();
                BufferDesc& ob = node->output_buffer;
                std::vector<char> gpu_copy(ob.byte_size);
                std::memcpy(gpu_copy.data(), ob.get_data(), ob.byte_size);
                dispatch_node(*node, nodes_, node_index_map_);
                if (ob.precision == Precision::FP16) {
                    const __fp16* g = reinterpret_cast<const __fp16*>(gpu_copy.data());
                    const __fp16* c = ob.data_as<__fp16>();
                    double maxd=0; size_t bad=0;
                    for (size_t k=0;k<ob.total_size;++k){ double d=std::fabs((double)g[k]-(double)c[k]); if(d>maxd)maxd=d; if(d>0.05)bad++; }
                    if (maxd > 0.05) {
                        static std::set<std::string> seen;
                        std::string key = std::string(get_op_name(node->op_type));
                        if (seen.insert(key).second)
                            std::cerr << "[gpu-verify] MISMATCH op=" << get_op_name(node->op_type)
                                      << " id=" << node->id << " maxdiff=" << maxd << " bad=" << bad
                                      << "/" << ob.total_size << " shape0=" << (ob.shape.empty()?0:ob.shape[0]) << "\n";
                    }
                }
            }
            if (ot == OpType::PERSISTENT) populated_node_ids_.insert(node->id);
            for (size_t r : release_after[i]) gpu_release(r);
        }
        if (std::getenv("CACTUS_GPU_STATS") && n > 1000) {
            static std::set<size_t> seen;
            if (seen.insert(n).second) { std::cerr << "[gpu-fallback] graph n=" << n << ":\n";
                for (auto& kv : gpu_fallback_hist) std::cerr << "    " << get_op_name((OpType)kv.first) << " : " << kv.second << "\n"; }
        }
        cactus_metal_set_active(false);
        auto _t_loop = _tt();
        cactus_metal_session_end();
        if (gpu_timing && n > 1000) {
            using us = std::chrono::microseconds;
            auto _t_done = _tt();
            static int _c = 0; static long _sum_total=0, _sum_loop=0; static int _cnt=0;
            _sum_total += std::chrono::duration_cast<us>(_t_done-_t0).count();
            _sum_loop  += std::chrono::duration_cast<us>(_t_loop-_t_live).count();
            _cnt++;
            if (_c++ % 32 == 31)
                std::cerr << "[gpu-timing] avg total execute=" << _sum_total/_cnt
                          << "us  loop(encode+inner-sync)=" << _sum_loop/_cnt
                          << "us  (CPU encode = total - on-GPU)\n";
        }
        return;
    }

    if (!need_debug) {
        static const bool cpu_prof = std::getenv("CACTUS_CPU_PROFILE") != nullptr;
        if (cpu_prof) cpu_prof_begin(n);
        auto run = [&](auto& nd) {
            if (!cpu_prof) { dispatch_node(nd, nodes_, node_index_map_); return; }
            auto t0 = std::chrono::high_resolution_clock::now();
            dispatch_node(nd, nodes_, node_index_map_);
            auto& pr = g_cpu_prof[n][(int)nd.op_type];
            pr.ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::high_resolution_clock::now() - t0).count();
            pr.inv++;
        };
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
            if (n == 3861) { static const char* dn = std::getenv("CACTUS_DUMP_NODE");
                if (dn && i == (size_t)std::atoi(dn)) { static int dc=0; if (dc++<1) {
                    BufferDesc& b = node->output_buffer; const __fp16* p = b.data_as<__fp16>();
                    if (p) std::cerr << "[dump] node " << i << " " << get_op_name(node->op_type) << ": "
                        << (float)p[0] << " " << (float)p[1] << " " << (float)p[2] << " " << (float)p[3]
                        << " | " << (float)p[4] << " " << (float)p[5] << "\n"; } } }
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

void CactusGraph::hard_reset() {
    nodes_.clear();
    node_index_map_.clear();
    mapped_files_.clear();
    weight_cache_.clear();
    next_node_id_ = 1;
    debug_nodes_.clear();
    buffer_pool_.clear();
}

void CactusGraph::soft_reset() {
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

void CactusGraph::prewarm_gpu_quant_weights() {
    const bool gpu = cactus_backend_metal() && std::getenv("CACTUS_NO_PREWARM") == nullptr;
    if (!gpu) return;
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
