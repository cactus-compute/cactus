#include "../cactus_graph.h"
#include "cactus_kernels.h"
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
DECLARE_COMPUTE(compute_persistent_node);
DECLARE_COMPUTE(compute_kv_cache_state_node);
DECLARE_COMPUTE(compute_kv_cache_append_node);
DECLARE_COMPUTE(compute_attention_cached_node);
DECLARE_COMPUTE(compute_conv_cache_state_node);
DECLARE_COMPUTE(compute_conv_cache_append_node);
DECLARE_COMPUTE(compute_image_preprocess_node);
DECLARE_COMPUTE(compute_rfft_node);
DECLARE_COMPUTE(compute_irfft_node);
DECLARE_COMPUTE(compute_mel_filter_bank_node);
DECLARE_COMPUTE(compute_spectrogram_node);
extern void shrink_thread_local_buffers();
#undef DECLARE_COMPUTE

static constexpr int OP_TYPE_COUNT = static_cast<int>(OpType::CLAMP) + 1;
static_assert(OP_TYPE_COUNT <= 256, "OpType dispatch table overflow");
static ComputeFn dispatch_flat[OP_TYPE_COUNT] = {};

static bool init_dispatch() {
    dispatch_flat[static_cast<int>(OpType::ADD)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::ADD_CLIPPED)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::SUBTRACT)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::MULTIPLY)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::DIVIDE)] = compute_binary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_ADD)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_SUBTRACT)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_MULTIPLY)] = compute_unary_op_node;
    dispatch_flat[static_cast<int>(OpType::SCALAR_DIVIDE)] = compute_unary_op_node;
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
    "MATMUL", "TRANSPOSE", "RESHAPE", "SLICE", "GATHER", "EMBEDDING", "VIEW", "FLATTEN",
    "BILINEAR_INTERPOLATION",
    "SUM", "MEAN", "VARIANCE", "MIN", "MAX",
    "RMS_NORM", "ROPE", "ROPE_GPTJ", "SOFTMAX", "ATTENTION", "ATTENTION_INT8_HYBRID", "REL_POS_BIAS", "CONV1D_CAUSAL", "CONV1D_K3", "CONV1D_K7S3", "CONV1D", "CONV1D_SAME_DEPTHWISE_K9", "CONV1D_POINTWISE", "CONV2D_K3S2P1", "CONV2D_DEPTHWISE_K3S2P1", "CONV2D_POINTWISE_1X1", "GLU", "BATCHNORM",
    "SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY", "SCALAR_DIVIDE",
    "SCALAR_EXP", "SCALAR_SQRT", "SCALAR_COS", "SCALAR_SIN", "SCALAR_LOG",
    "ABS", "POW", 
    "RELU", "SILU", "GELU", "GELU_ERF", "SIGMOID", "TANH",
    "SAMPLE", "CONCAT",
    "SCATTER_TOPK",
    "TOPK", "LAYERNORM", "GROUPNORM",
    "MOE_LAYER",
    "INDEX",
    "PERSISTENT",
    "LSTM_CELL",
    "GATED_DELTANET_DECODE",
    "GATED_DELTANET_PREFILL",
    "STFT",
    "ALTUP_PREDICT",
    "ALTUP_CORRECT",
    "GAUSSIAN_TOPK",
    "MAXPOOL1D",
    "BILSTM_SEQUENCE",
    "LEAKY_RELU",
    "CONV2D_K3S1P1",
    "STATS_POOL",
    "WEIGHTED_STATS_POOL"
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

void CactusGraph::execute(const std::string& profile_file) {
    BufferPool& pool = buffer_pool_;
    const size_t n = nodes_.size();

    bool need_debug = !profile_file.empty();
    if (!need_debug) {
        static const bool env_debug = check_debug_env();
        need_debug = env_debug;
    }

    if (!need_debug) {
        for (size_t i = 0; i < n; ++i) {
            auto& node = nodes_[i];
            if (node->op_type == OpType::INPUT) continue;
            if (node->op_type == OpType::KV_CACHE_STATE || node->op_type == OpType::CONV_CACHE_STATE) {
                dispatch_node(*node, nodes_, node_index_map_);
                populated_node_ids_.insert(node->id);
                continue;
            }
            node->output_buffer.allocate_from_pool(pool);
            dispatch_node(*node, nodes_, node_index_map_);
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }
        }
        return;
    }

    std::vector<size_t> last_use(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t input_id : nodes_[i]->input_ids) {
            auto it = node_index_map_.find(input_id);
            if (it != node_index_map_.end()) {
                last_use[it->second] = std::max(last_use[it->second], i);
            }
        }
    }

    auto get_env_int = [](const char* name, int fallback) -> int {
        const char* val = std::getenv(name);
        return val ? std::atoi(val) : fallback;
    };

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

        if (node->op_type != OpType::INPUT) {
            node->output_buffer.allocate_from_pool(pool);
        }

        if (enable_profiling && node->op_type != OpType::INPUT) {
            auto start = std::chrono::high_resolution_clock::now();
            dispatch_node(*node, nodes_, node_index_map_);
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
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }
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
