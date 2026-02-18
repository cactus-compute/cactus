#include "graph.h"
#include "../kernel/kernel.h"
#include "../kernel/kernel_utils.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <assert.h>

namespace {
    thread_local std::vector<__fp16> transpose_buffer_fp16;
    thread_local std::vector<int8_t> quant_activation_buffer;
    thread_local std::vector<float> quant_scales_buffer;

    thread_local const __fp16* cached_quant_src = nullptr;
    thread_local size_t cached_quant_M = 0;
    thread_local size_t cached_quant_K = 0;

    void ensure_transpose_buffer_fp16(size_t required_size) {
        if (transpose_buffer_fp16.size() < required_size) {
            transpose_buffer_fp16.resize(required_size);
        }
    }

    void ensure_quant_buffers(size_t M, size_t K) {
        size_t required_data = M * K;
        if (quant_activation_buffer.size() < required_data) {
            quant_activation_buffer.resize(required_data);
        }
        if (quant_scales_buffer.size() < M) {
            quant_scales_buffer.resize(M);
        }
    }

    void quantize_activations_fp16_to_int8(const __fp16* src, int8_t* dst, float* scales, size_t M, size_t K) {
        if (src == cached_quant_src && M == cached_quant_M && K == cached_quant_K) {
            return;
        }

        constexpr size_t PARALLEL_THRESHOLD = 16;

        if (M >= PARALLEL_THRESHOLD) {
            CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
                [src, dst, scales, K](size_t m_start, size_t m_end) {
                    for (size_t m = m_start; m < m_end; m++) {
                        float max_abs = cactus_fp16_max_abs(src + m * K, K);
                        float scale = max_abs / 127.0f;
                        if (scale < 1e-10f) scale = 1e-10f;
                        scales[m] = scale;
                        cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
                    }
                });
        } else {
            for (size_t m = 0; m < M; m++) {
                float max_abs = cactus_fp16_max_abs(src + m * K, K);
                float scale = max_abs / 127.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                scales[m] = scale;
                cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
            }
        }

        cached_quant_src = src;
        cached_quant_M = M;
        cached_quant_K = K;
    }
}

void shrink_thread_local_buffers() {
    std::vector<__fp16>().swap(transpose_buffer_fp16);
    std::vector<int8_t>().swap(quant_activation_buffer);
    std::vector<float>().swap(quant_scales_buffer);
    cached_quant_src = nullptr;
    cached_quant_M = 0;
    cached_quant_K = 0;
}

void compute_quantize_activations_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& shape = input_buffer.shape;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("QUANTIZE_ACTIVATIONS requires FP16 input");
    }

    if (shape.size() < 2) {
        throw std::runtime_error("QUANTIZE_ACTIVATIONS requires at least 2D tensor");
    }

    size_t K = shape.back();
    size_t M = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        M *= shape[i];
    }

    if (!node.output_buffer.has_activation_scales() ||
        node.output_buffer.num_rows_for_activation_scales != M) {
        node.output_buffer.allocate_activation_scales(M);
    }

    const __fp16* src = input_buffer.data_as<__fp16>();
    int8_t* dst = node.output_buffer.data_as<int8_t>();
    float* scales = node.output_buffer.activation_scales_as_float();

    constexpr size_t PARALLEL_THRESHOLD = 16;

    if (M >= PARALLEL_THRESHOLD) {
        CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
            [src, dst, scales, K](size_t m_start, size_t m_end) {
                for (size_t m = m_start; m < m_end; m++) {
                    float max_abs = cactus_fp16_max_abs(src + m * K, K);
                    float scale = max_abs / 127.0f;
                    if (scale < 1e-10f) scale = 1e-10f;
                    scales[m] = scale;
                    cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
                }
            });
    } else {
        for (size_t m = 0; m < M; m++) {
            float max_abs = cactus_fp16_max_abs(src + m * K, K);
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[m] = scale;
            cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
        }
    }
}

void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& rhs_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& lhs_shape = lhs_buffer.shape;
    const auto& rhs_shape = rhs_buffer.shape;

    size_t M = lhs_shape[lhs_shape.size() - 2];
    size_t K = lhs_shape[lhs_shape.size() - 1];
    size_t N;
    if (rhs_buffer.is_interleaved && rhs_buffer.original_N > 0) {
        N = rhs_buffer.original_N;
    } else {
        N = node.params.pretransposed_rhs ?
            rhs_shape[rhs_shape.size() - 2] : rhs_shape[rhs_shape.size() - 1];
    }

    bool pretransposed_rhs = node.params.pretransposed_rhs;

    ComputeBackend backend = node.params.backend;

    if (backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU matrix multiplication not yet implemented");
    }

    const bool lhs_is_prequantized_int8 = (lhs_buffer.precision == Precision::INT8 &&
                                            lhs_buffer.has_activation_scales());

    if (rhs_buffer.is_grouped_int8()) {
        const int8_t* rhs = rhs_buffer.data_as<int8_t>();
        const __fp16* rhs_scales = rhs_buffer.scales_as_fp16();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (!pretransposed_rhs) {
            throw std::runtime_error("Group-wise INT8 matmul requires pretransposed weights");
        }

        const int8_t* lhs_int8;
        const float* lhs_scales;

        if (lhs_is_prequantized_int8) {
            lhs_int8 = lhs_buffer.data_as<int8_t>();
            lhs_scales = lhs_buffer.activation_scales_as_float();
        } else if (lhs_buffer.precision == Precision::FP16) {
            const __fp16* lhs = lhs_buffer.data_as<__fp16>();
            ensure_quant_buffers(M, K);
            quantize_activations_fp16_to_int8(lhs, quant_activation_buffer.data(),
                                              quant_scales_buffer.data(), M, K);
            lhs_int8 = quant_activation_buffer.data();
            lhs_scales = quant_scales_buffer.data();
        } else {
            throw std::runtime_error("INT8 matmul requires INT8 (pre-quantized) or FP16 activations");
        }

        cactus_matmul_int8(lhs_int8, lhs_scales,
                           rhs, rhs_scales, output,
                           M, K, N, rhs_buffer.group_size);

    } else {
        if (lhs_buffer.precision != Precision::FP16) {
            throw std::runtime_error("FP16 matmul requires FP16 activations");
        }

        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        const __fp16* rhs = rhs_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (pretransposed_rhs) {
            cactus_matmul_f16(lhs, rhs, output, M, K, N);
        } else {
            size_t transpose_size = rhs_shape[0] * rhs_shape[1];
            ensure_transpose_buffer_fp16(transpose_size);

            cactus_transpose_2d_f16(rhs, transpose_buffer_fp16.data(),
                                    rhs_shape[0], rhs_shape[1], 0, rhs_shape[0]);
            cactus_matmul_f16(lhs, transpose_buffer_fp16.data(), output, M, K, N);
        }
    }
}

void compute_rms_norm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    if (input_buffer.shape.size() != 2) {
        throw std::runtime_error("RMS normalization requires 2D input tensor [batch_size, dims], got " +
                                std::to_string(input_buffer.shape.size()) + "D tensor");
    }

    size_t batch_size = input_buffer.shape[0];
    size_t dims = input_buffer.shape[1];

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("RMS normalization only supports FP16 precision");
    }

    cactus_rms_norm_f16(input_buffer.data_as<__fp16>(), weight_buffer.data_as<__fp16>(),
       node.output_buffer.data_as<__fp16>(), batch_size, dims, node.params.epsilon);
}

void compute_rope_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU RoPE operation not yet implemented");
    }

    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& shape = input_buffer.shape;

    if (shape.size() < 4) {
        throw std::runtime_error("RoPE operation requires 4D tensor with shape [batch, seq_len, num_heads, head_dim], got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    if (input_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
        throw std::runtime_error("RoPE operation only supports FP16 precision");
    }

    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t num_heads = shape[2];
    size_t head_dim = shape[3];

    cactus_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                   batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta);
}

void compute_softmax_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& shape = input_buffer.shape;

    if (shape.size() < 2) {
        throw std::runtime_error("Softmax operation requires at least 2D tensor, got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Softmax operation only supports FP16 precision");
    }

    size_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        batch_size *= shape[i];
    }
    size_t vocab_size = shape[shape.size() - 1];

    cactus_softmax_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                      batch_size, 1, vocab_size);
}

void compute_attention_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU attention operation not yet implemented");
    }

    if (node.input_ids.size() < 3) {
        throw std::runtime_error("Attention operation requires 3 inputs (query, key, value), got " +
                                std::to_string(node.input_ids.size()) + " inputs");
    }

    const auto& query_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& key_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& value_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    const auto& q_shape = query_buffer.shape;
    const auto& k_shape = key_buffer.shape;

    if (q_shape.size() < 4) {
        throw std::runtime_error("Attention operation requires 4D tensors [batch, seq_len, num_heads, head_dim], got " +
                                std::to_string(q_shape.size()) + "D tensor");
    }

    if (query_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Attention operation only supports FP16 precision");
    }

    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];
    size_t head_dim = q_shape[3];
    size_t num_kv_heads = k_shape[2];
    size_t kv_seq_len = key_buffer.shape[1];

    cactus_attention_f16(query_buffer.data_as<__fp16>(), key_buffer.data_as<__fp16>(),
                         value_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                         batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, nullptr,
                         node.params.position_offset, node.params.window_size, node.params.is_causal);
}

void compute_attention_int8_hybrid_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& query_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& key_new_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& value_new_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    const auto& q_shape = query_buffer.shape;

    if (q_shape.size() < 4) {
        throw std::runtime_error("ATTENTION_INT8_HYBRID requires 4D query tensor");
    }

    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];
    size_t head_dim = node.params.head_dim;
    size_t num_kv_heads = node.params.num_kv_heads;
    size_t cache_len = node.params.cache_seq_len;
    size_t new_len = key_new_buffer.shape[1];

    cactus_attention_hybrid_int8_fp16(
        query_buffer.data_as<__fp16>(),
        node.params.cached_keys_int8,
        node.params.cached_values_int8,
        node.params.cached_k_scales,
        node.params.cached_v_scales,
        key_new_buffer.data_as<__fp16>(),
        value_new_buffer.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, seq_len, cache_len, new_len,
        num_q_heads, num_kv_heads, head_dim,
        node.params.scale, node.params.position_offset, true,
        node.params.window_size
    );
}

void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    bool has_bias = node.input_ids.size() > 2;
    float epsilon = node.params.epsilon;

    if (input_buffer.shape.empty()) {
        throw std::runtime_error("LayerNorm requires non-empty input tensor");
    }

    size_t feature_size = input_buffer.shape.back();
    size_t batch_size = input_buffer.total_size / feature_size;

    std::vector<float> input_float(input_buffer.total_size);
    std::vector<float> weight_float(feature_size);
    std::vector<float> bias_float(feature_size, 0.0f);

    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        std::memcpy(input_float.data(), input_buffer.data_as<float>(), input_buffer.total_size * sizeof(float));
    }

    if (weight_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 weight");
    } else if (weight_buffer.precision == Precision::FP16) {
        const __fp16* weight_fp16 = weight_buffer.data_as<__fp16>();
        for (size_t i = 0; i < feature_size; ++i) {
            weight_float[i] = static_cast<float>(weight_fp16[i]);
        }
    } else {
        std::memcpy(weight_float.data(), weight_buffer.data_as<float>(), feature_size * sizeof(float));
    }

    if (has_bias) {
        const auto& bias_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
        if (bias_buffer.precision == Precision::INT8) {
            throw std::runtime_error("LayerNorm currently does not support INT8 bias");
        } else if (bias_buffer.precision == Precision::FP16) {
            const __fp16* bias_fp16 = bias_buffer.data_as<__fp16>();
            for (size_t i = 0; i < feature_size; ++i) {
                bias_float[i] = static_cast<float>(bias_fp16[i]);
            }
        } else {
            std::memcpy(bias_float.data(), bias_buffer.data_as<float>(), feature_size * sizeof(float));
        }
    }

    std::vector<float> output_float(input_buffer.total_size);
    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input_float.data() + b * feature_size;
        float* output_row = output_float.data() + b * feature_size;

        float mean = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            mean += input_row[i];
        }
        mean /= feature_size;

        float variance = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;

        float std_inv = 1.0f / std::sqrt(variance + epsilon);
        for (size_t i = 0; i < feature_size; ++i) {
            output_row[i] = (input_row[i] - mean) * std_inv * weight_float[i] + bias_float[i];
        }
    }

    if (node.output_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 output");
    } else if (node.output_buffer.precision == Precision::FP16) {
        __fp16* output_fp16 = node.output_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            output_fp16[i] = static_cast<__fp16>(output_float[i]);
        }
    } else {
        std::memcpy(node.output_buffer.data_as<float>(), output_float.data(), input_buffer.total_size * sizeof(float));
    }
}

void compute_conv1d_causal_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("Causal conv requires 3D input [batch, seq_len, in_channels]");
    }
    if (W.shape.size() != 3) {
        throw std::runtime_error("Weight must be 3D");
    }

    const size_t N     = X.shape[0];
    const size_t L     = X.shape[1];
    const size_t C_in  = X.shape[2];
    const size_t W0    = W.shape[0];
    const size_t W1    = W.shape[1];
    const size_t K     = W.shape[2];
    const size_t dil   = node.params.dilation;
    if (dil < 1) throw std::runtime_error("dilation must be >= 1");

    size_t M = 1;
    size_t C_out = 0;

    assert((W1 == 1) && (W0 % C_in == 0) && "Only depthwise causal convolution is supported currently");
    M = W0 / C_in;
    C_out = C_in * M;

    Y.shape = { N, L, C_out };
    Y.precision = X.precision;

    if (W.precision == Precision::INT8) {
        const size_t W_size = W0 * W1 * K;
        const int8_t* W_int8 = W.data_as<int8_t>();

        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t K_total = W1 * K;
            const size_t group_size = W.group_size;
            const size_t num_groups = K_total / group_size;

            for (size_t row = 0; row < W0; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv1d_causal_depthwise_f16(
            X.data_as<__fp16>(), W_fp16.data(), Y.data_as<__fp16>(),
            N, L, C_in, K, dil);
    } else if (W.precision == Precision::FP16) {
        cactus_conv1d_causal_depthwise_f16(
            X.data_as<__fp16>(), W.data_as<__fp16>(), Y.data_as<__fp16>(),
            N, L, C_in, K, dil);
    } else {
        throw std::runtime_error("Depthwise causal conv supports INT8/FP16 weights");
    }
}

void compute_conv1d_k3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3)
        throw std::runtime_error("Conv requires 3D input [N, C_in, L]!");

    if (W.shape.size() != 3)
        throw std::runtime_error("Weight must be [C_out, C_in, 3]!");

    const size_t N    = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L    = X.shape[2];

    const size_t C_out = W.shape[0];
    const size_t K     = W.shape[2];
    const size_t stride = node.params.stride;

    if (K != 3)
        throw std::runtime_error("Conv1d_k3 only supports K=3!");

    size_t L_out = ((L - 1) / stride) + 1;
    Y.shape     = { N, C_out, L_out };
    Y.precision = X.precision;

    if (X.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d_k3 only supports FP16 activations");
    }

    if (W.precision == Precision::INT8) {
        const size_t W_size = C_out * C_in * K;
        const int8_t* W_int8 = W.data_as<int8_t>();

        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t K_total = C_in * K;
            const size_t group_size = W.group_size;
            const size_t num_groups = K_total / group_size;

            for (size_t row = 0; row < C_out; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W_fp16.data(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else if (W.precision == Precision::FP16) {
        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else {
        throw std::runtime_error("Conv1d_k3 only supports FP16 and INT8 weights");
    }
}

void compute_conv1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    const __fp16* bias_ptr = (node.input_ids.size() > 2) ?
        nodes[node_index_map.at(node.input_ids[2])]->output_buffer.data_as<__fp16>() : nullptr;

    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d only supports FP16");
    }

    cactus_conv1d_f16(X.data_as<__fp16>(), W.data_as<__fp16>(), bias_ptr,
                      Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride);
}

void compute_stft_magnitude_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                 const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;
    const size_t num_fft_bins = node.params.num_fft_bins;

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("stft_magnitude only supports FP16");
    }

    cactus_stft_magnitude_f16(X.data_as<__fp16>(), W.data_as<__fp16>(),
                              Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride, num_fft_bins);
}

void compute_conv1d_k7s3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer; // Expected packed [C_in, K, C_out]

    const __fp16* bias_ptr = (node.input_ids.size() > 2) ?
        nodes[node_index_map.at(node.input_ids[2])]->output_buffer.data_as<__fp16>() : nullptr;

    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    
    if (W.shape.size() != 3) throw std::runtime_error("Weight must be 3D");
    const size_t C_in_W = W.shape[0];
    const size_t K = W.shape[1];
    const size_t C_out = W.shape[2];
    const size_t stride = node.params.stride;

    if (C_in != C_in_W) throw std::runtime_error("Channel mismatch in conv1d_k7s3");
    if (K != 7 || stride != 3) throw std::runtime_error("conv1d_k7s3 requires K=7, stride=3");

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d specialized only supports FP16");
    }
    
    size_t L_out = (L < 7) ? 0 : (L - 7) / 3 + 1;
    Y.shape = {N, C_out, L_out};
    Y.precision = Precision::FP16;

    cactus_conv1d_f16_k7s3_oc8(
        X.data_as<__fp16>(), 
        W.data_as<__fp16>(), 
        bias_ptr,
        Y.data_as<__fp16>(), 
        N, L, C_in, C_out
    );
}

void compute_rope_gptj_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& shape = input_buffer.shape;

    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t num_heads = shape[2];
    size_t head_dim = shape[3];
    size_t rot_dim = static_cast<size_t>(node.params.scalar);

    cactus_gpt_j_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                          batch_size, seq_len, num_heads, head_dim, rot_dim,
                          node.params.position_offset, node.params.theta);
}

void compute_groupnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& weight = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& bias = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    float epsilon = node.params.epsilon;

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t spatial_size = 1;
    for (size_t i = 2; i < input.shape.size(); ++i) spatial_size *= input.shape[i];

    size_t num_groups = node.params.num_groups;
    if (num_groups == 0) num_groups = 32;
    
    if (channels % num_groups != 0) {
        throw std::runtime_error("GroupNorm: channels must be divisible by num_groups");
    }

    size_t channels_per_group = channels / num_groups;

    const __fp16* src = input.data_as<__fp16>();
    const __fp16* w = weight.data_as<__fp16>();
    const __fp16* b = bias.data_as<__fp16>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float sum = 0.0f, sum_sq = 0.0f;
            size_t count = 0;

            for (size_t c = 0; c < channels_per_group; ++c) {
                size_t ch = g * channels_per_group + c;
                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + ch * spatial_size + s;
                    float val = static_cast<float>(src[idx]);
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }

            float mean = sum / count;
            float var = (sum_sq / count) - (mean * mean);
            float inv_std = 1.0f / std::sqrt(var + epsilon);

            for (size_t c = 0; c < channels_per_group; ++c) {
                size_t ch = g * channels_per_group + c;
                float wt = static_cast<float>(w[ch]);
                float bi = static_cast<float>(b[ch]);

                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + ch * spatial_size + s;
                    float val = static_cast<float>(src[idx]);
                    dst[idx] = static_cast<__fp16>((val - mean) * inv_std * wt + bi);
                }
            }
        }
    }
}

void compute_lstm_cell_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& h_prev_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& c_prev_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    const auto& weight_ih_buffer = nodes[node_index_map.at(node.input_ids[3])]->output_buffer;
    const auto& weight_hh_buffer = nodes[node_index_map.at(node.input_ids[4])]->output_buffer;
    const auto& bias_ih_buffer = nodes[node_index_map.at(node.input_ids[5])]->output_buffer;
    const auto& bias_hh_buffer = nodes[node_index_map.at(node.input_ids[6])]->output_buffer;

    if (input_buffer.precision != Precision::FP16 || h_prev_buffer.precision != Precision::FP16 ||
        c_prev_buffer.precision != Precision::FP16 || weight_ih_buffer.precision != Precision::FP16 ||
        weight_hh_buffer.precision != Precision::FP16 || bias_ih_buffer.precision != Precision::FP16 ||
        bias_hh_buffer.precision != Precision::FP16) {
        throw std::runtime_error("LSTM cell requires all inputs to be FP16");
    }

    if (input_buffer.shape.size() != 2 || h_prev_buffer.shape.size() != 2 || c_prev_buffer.shape.size() != 2) {
        throw std::runtime_error("LSTM cell input/state shapes must be 2D [batch, features]");
    }

    const size_t batch_size = input_buffer.shape[0];
    const size_t input_size = input_buffer.shape[1];
    const size_t hidden_size = h_prev_buffer.shape[1];

    const __fp16* x_input = input_buffer.data_as<__fp16>();
    const __fp16* h_prev = h_prev_buffer.data_as<__fp16>();
    const __fp16* c_prev = c_prev_buffer.data_as<__fp16>();
    const __fp16* weight_ih = weight_ih_buffer.data_as<__fp16>();
    const __fp16* weight_hh = weight_hh_buffer.data_as<__fp16>();
    const __fp16* bias_ih = bias_ih_buffer.data_as<__fp16>();
    const __fp16* bias_hh = bias_hh_buffer.data_as<__fp16>();

    node.output_buffer.shape = {batch_size, hidden_size, 2};
    node.output_buffer.total_size = batch_size * hidden_size * 2;
    node.output_buffer.precision = Precision::FP16;
    node.output_buffer.allocate();

    std::vector<__fp16> h_new_temp(batch_size * hidden_size);
    std::vector<__fp16> c_new_temp(batch_size * hidden_size);

    cactus_lstm_cell_f16(
        x_input, h_prev, c_prev,
        weight_ih, weight_hh,
        bias_ih, bias_hh,
        h_new_temp.data(), c_new_temp.data(),
        batch_size, input_size, hidden_size
    );

    __fp16* output = node.output_buffer.data_as<__fp16>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < hidden_size; ++i) {
            const size_t idx = b * hidden_size + i;
            output[b * hidden_size * 2 + i * 2] = h_new_temp[idx];
            output[b * hidden_size * 2 + i * 2 + 1] = c_new_temp[idx];
        }
    }
}
