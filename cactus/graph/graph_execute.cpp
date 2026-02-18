#include "graph.h"
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

extern void compute_binary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_unary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_activation_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_reshape_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_precision_cast_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

extern void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_rms_norm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_rope_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_softmax_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_attention_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_attention_int8_hybrid_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_conv1d_causal_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_conv1d_k3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_conv1d_k7s3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_conv1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_groupnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_rope_gptj_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void shrink_thread_local_buffers();
extern void compute_lstm_cell_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_stft_magnitude_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

extern void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_gather_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_slice_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_embedding_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_concat_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_bilinear_interpolation_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

extern void compute_sample_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_scatter_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_persistent_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
extern void compute_quantize_activations_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

static const char* op_type_names[] = {
    "INPUT", "PRECISION_CAST",
    "ADD", "ADD_CLIPPED", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "MATMUL", "TRANSPOSE", "RESHAPE", "SLICE", "GATHER", "EMBEDDING",
    "BILINEAR_INTERPOLATION",
    "SUM", "MEAN", "VARIANCE", "MIN", "MAX",
    "RMS_NORM", "ROPE", "ROPE_GPTJ", "SOFTMAX", "ATTENTION", "ATTENTION_INT8_HYBRID", "CONV1D_CAUSAL", "CONV1D_K3", "CONV1D_K7S3", "CONV1D",
    "SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY", "SCALAR_DIVIDE",
    "SCALAR_EXP", "SCALAR_SQRT", "SCALAR_COS", "SCALAR_SIN",
    "RELU", "SILU", "GELU", "GELU_ERF", "SIGMOID", "TANH",
    "SAMPLE", "CONCAT",
    "SCATTER_TOPK",
    "TOPK", "LAYERNORM", "GROUPNORM",
    "INDEX",
    "PERSISTENT",
    "QUANTIZE_ACTIVATIONS",
    "LSTM_CELL",
    "STFT_MAGNITUDE"
};

static const char* get_op_name(OpType op) {
    return op_type_names[static_cast<int>(op)];
}

void compute_node_optimized(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    switch (node.op_type) {
        case OpType::INPUT:
            break;

        case OpType::ADD:
        case OpType::ADD_CLIPPED:
        case OpType::SUBTRACT:
        case OpType::MULTIPLY:
        case OpType::DIVIDE:
            compute_binary_op_node(node, nodes, node_index_map);
            break;

        case OpType::SCALAR_ADD:
        case OpType::SCALAR_SUBTRACT:
        case OpType::SCALAR_MULTIPLY:
        case OpType::SCALAR_DIVIDE:
        case OpType::SCALAR_EXP:
        case OpType::SCALAR_SQRT:
        case OpType::SCALAR_COS:
        case OpType::SCALAR_SIN:
            compute_unary_op_node(node, nodes, node_index_map);
            break;

        case OpType::RELU:
        case OpType::SILU:
        case OpType::GELU:
        case OpType::GELU_ERF:
        case OpType::SIGMOID:
        case OpType::TANH:
            compute_activation_node(node, nodes, node_index_map);
            break;

        case OpType::SUM:
        case OpType::MEAN:
        case OpType::VARIANCE:
        case OpType::MIN:
        case OpType::MAX:
            compute_reduce_node(node, nodes, node_index_map);
            break;

        case OpType::RESHAPE:
            compute_reshape_node(node, nodes, node_index_map);
            break;

        case OpType::PRECISION_CAST:
            compute_precision_cast_node(node, nodes, node_index_map);
            break;

        case OpType::MATMUL:
            compute_matmul_node(node, nodes, node_index_map);
            break;

        case OpType::RMS_NORM:
            compute_rms_norm_node(node, nodes, node_index_map);
            break;

        case OpType::ROPE:
            compute_rope_node(node, nodes, node_index_map);
            break;

        case OpType::ROPE_GPTJ:
            compute_rope_gptj_node(node, nodes, node_index_map);
            break;

        case OpType::SOFTMAX:
            compute_softmax_node(node, nodes, node_index_map);
            break;

        case OpType::ATTENTION:
            compute_attention_node(node, nodes, node_index_map);
            break;

        case OpType::ATTENTION_INT8_HYBRID:
            compute_attention_int8_hybrid_node(node, nodes, node_index_map);
            break;

        case OpType::LAYERNORM:
            compute_layernorm_node(node, nodes, node_index_map);
            break;

        case OpType::GROUPNORM:
            compute_groupnorm_node(node, nodes, node_index_map);
            break;

        case OpType::PERSISTENT:
            compute_persistent_node(node, nodes, node_index_map);
            break;

        case OpType::CONV1D_CAUSAL:
            compute_conv1d_causal_node(node, nodes, node_index_map);
            break;

        case OpType::CONV1D_K3:
            compute_conv1d_k3_node(node, nodes, node_index_map);
            break;

        case OpType::CONV1D_K7S3:
            compute_conv1d_k7s3_node(node, nodes, node_index_map);
            break;

        case OpType::CONV1D:
            compute_conv1d_node(node, nodes, node_index_map);
            break;

        case OpType::TRANSPOSE:
            compute_transpose_node(node, nodes, node_index_map);
            break;

        case OpType::GATHER:
            compute_gather_node(node, nodes, node_index_map);
            break;

        case OpType::SLICE:
            compute_slice_node(node, nodes, node_index_map);
            break;

        case OpType::EMBEDDING:
            compute_embedding_node(node, nodes, node_index_map);
            break;

        case OpType::CONCAT:
            compute_concat_node(node, nodes, node_index_map);
            break;

        case OpType::INDEX:
            compute_index_node(node, nodes, node_index_map);
            break;

        case OpType::BILINEAR_INTERPOLATION:
            compute_bilinear_interpolation_node(node, nodes, node_index_map);
            break;

        case OpType::SAMPLE:
            compute_sample_node(node, nodes, node_index_map);
            break;

        case OpType::TOPK:
            compute_topk_node(node, nodes, node_index_map);
            break;

        case OpType::SCATTER_TOPK:
            compute_scatter_topk_node(node, nodes, node_index_map);
            break;

        case OpType::QUANTIZE_ACTIVATIONS:
            compute_quantize_activations_node(node, nodes, node_index_map);
            break;

        case OpType::LSTM_CELL:
            compute_lstm_cell_node(node, nodes, node_index_map);
            break;

        case OpType::STFT_MAGNITUDE:
            compute_stft_magnitude_node(node, nodes, node_index_map);
            break;

        default:
            throw std::runtime_error("Unknown operation type: " + std::to_string(static_cast<int>(node.op_type)));
    }
}

void CactusGraph::set_input(size_t node_id, const void* data, Precision) {
    auto& node = *nodes_[node_index_map_[node_id]];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only set data on input nodes");
    }

    if (!node.output_buffer.data && !node.output_buffer.external_data) {
        node.output_buffer.allocate();
    }

    std::memcpy(node.output_buffer.get_data(), data, node.output_buffer.byte_size);
}

void CactusGraph::set_external_input(size_t node_id, void* data, Precision) {
    auto& node = *nodes_[node_index_map_[node_id]];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only set data on input nodes");
    }

    node.output_buffer.set_external(data);
}

void* CactusGraph::get_output(size_t node_id) {
    auto& buffer = nodes_[node_index_map_[node_id]]->output_buffer;
    if (!buffer.get_data()) {
        buffer.allocate();
    }
    return buffer.get_data();
}

void CactusGraph::execute(const std::string& profile_file) {
    std::vector<size_t> last_use(nodes_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (size_t input_id : nodes_[i]->input_ids) {
            auto it = node_index_map_.find(input_id);
            if (it != node_index_map_.end()) {
                last_use[it->second] = std::max(last_use[it->second], i);
            }
        }
    }

    BufferPool& pool = buffer_pool_;

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
        *out << std::left << std::setw(15) << "Operation"
             << std::setw(12) << "Time (ms)"
             << std::setw(20) << "Output Shape"
             << "Backend" << std::endl;
        *out << std::string(60, '-') << std::endl;
    }

    for (size_t node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
        auto& node = nodes_[node_idx];

        if (node->op_type != OpType::INPUT) {
            node->output_buffer.allocate_from_pool(pool);
        }

        if (enable_profiling && node->op_type != OpType::INPUT) {
            auto start = std::chrono::high_resolution_clock::now();

            compute_node_optimized(*node, nodes_, node_index_map_);
            
            if (node->op_type == OpType::PERSISTENT) {
                populated_node_ids_.insert(node->id);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double ms = duration.count() / 1000.0;

            std::string shape_str = "[";
            for (size_t i = 0; i < node->output_buffer.shape.size(); ++i) {
                if (i > 0) shape_str += ",";
                shape_str += std::to_string(node->output_buffer.shape[i]);
            }
            shape_str += "]";

            std::string values_str = "";
            if (node->output_buffer.get_data()) {
                size_t num_values = std::min(size_t(5), node->output_buffer.total_size);
                values_str = " values=[";

                if (node->output_buffer.precision == Precision::FP32) {
                    if (node->op_type == OpType::SAMPLE) {
                        uint32_t* uint32_data = reinterpret_cast<uint32_t*>(node->output_buffer.get_data());
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) values_str += ",";
                            values_str += std::to_string(uint32_data[i]);
                        }
                    } else {
                        float* float_data = reinterpret_cast<float*>(node->output_buffer.get_data());
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) values_str += ",";
                            values_str += std::to_string(float_data[i]).substr(0, 6);
                        }
                    }
                } else if (node->output_buffer.precision == Precision::FP16) {
                    __fp16* fp16_data = reinterpret_cast<__fp16*>(node->output_buffer.get_data());
                    for (size_t i = 0; i < num_values; ++i) {
                        if (i > 0) values_str += ",";
                        values_str += std::to_string(static_cast<float>(fp16_data[i])).substr(0, 6);
                    }
                } else if (node->output_buffer.precision == Precision::INT8) {
                    int8_t* int8_data = reinterpret_cast<int8_t*>(node->output_buffer.get_data());
                    for (size_t i = 0; i < num_values; ++i) {
                        if (i > 0) values_str += ",";
                        values_str += std::to_string(static_cast<int>(int8_data[i]));
                    }
                }

                if (node->output_buffer.total_size > 5) {
                    values_str += ",...";
                }
                values_str += "]";
            }

            std::string weights_str = "";
            if ((node->op_type == OpType::RMS_NORM || node->op_type == OpType::MATMUL ||
                 node->op_type == OpType::GATHER || node->op_type == OpType::EMBEDDING ||
                 node->op_type == OpType::ATTENTION || node->op_type == OpType::CONCAT) &&
                node->input_ids.size() >= 2) {
                const auto& weight_node = nodes_[node_index_map_.at(node->input_ids[1])];
                if (weight_node->output_buffer.get_data()) {
                    size_t num_values = std::min(size_t(5), weight_node->output_buffer.total_size);
                    weights_str = " weights=[";

                    if (weight_node->output_buffer.precision == Precision::FP32) {
                        const float* float_data = weight_node->output_buffer.data_as<float>();
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) weights_str += ",";
                            weights_str += std::to_string(float_data[i]).substr(0, 6);
                        }
                    } else if (weight_node->output_buffer.precision == Precision::FP16) {
                        const __fp16* fp16_data = weight_node->output_buffer.data_as<__fp16>();
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) weights_str += ",";
                            weights_str += std::to_string(static_cast<float>(fp16_data[i])).substr(0, 6);
                        }
                    } else if (weight_node->output_buffer.precision == Precision::INT8) {
                        const int8_t* int8_data = weight_node->output_buffer.data_as<int8_t>();
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) weights_str += ",";
                            weights_str += std::to_string(static_cast<int>(int8_data[i]));
                        }
                    }

                    if (weight_node->output_buffer.total_size > 5) {
                        weights_str += ",...";
                    }
                    weights_str += "]";
                }
            }

            *out << std::left << std::setw(15) << get_op_name(node->op_type)
                 << std::setw(12) << std::fixed << std::setprecision(3) << ms
                 << std::setw(20) << shape_str
                 << values_str << weights_str << std::endl;
        } else {
            compute_node_optimized(*node, nodes_, node_index_map_);
            
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
                             bytes_to_write = buffer.total_size * PrecisionTraits::size_of(buffer.precision);
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

        *out << std::string(60, '-') << std::endl;
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
    next_node_id_ = 0;
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
