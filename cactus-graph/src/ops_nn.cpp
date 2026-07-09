#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <algorithm>
#include <limits>
#include <atomic>
#include <iostream>

namespace {
    thread_local std::vector<__fp16> transpose_buffer_fp16;

    void ensure_transpose_buffer_fp16(size_t required_size) {
        if (transpose_buffer_fp16.size() < required_size) {
            transpose_buffer_fp16.resize(required_size);
        }
    }
}

void shrink_thread_local_buffers() {
    std::vector<__fp16>().swap(transpose_buffer_fp16);
}

void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& rhs_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& lhs_shape = lhs_buffer.shape;
    const auto& rhs_shape = rhs_buffer.shape;

    size_t M = lhs_shape[lhs_shape.size() - 2];
    size_t K = lhs_shape[lhs_shape.size() - 1];
    size_t N;
    N = node.params.pretransposed_rhs ?
        rhs_shape[rhs_shape.size() - 2] : rhs_shape[rhs_shape.size() - 1];

    bool pretransposed_rhs = node.params.pretransposed_rhs;

    if (PrecisionTraits::is_cq(rhs_buffer.precision) && rhs_buffer.group_size > 0) {
        if (lhs_buffer.precision != Precision::FP16) {
            throw std::runtime_error("TQ matmul requires FP16 activations");
        }

        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        CactusQuantMatrix mat = rhs_buffer.to_cq_matrix();
        if (rhs_buffer.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
            cactus_quant_orthogonal_matmul(&mat, lhs, static_cast<uint32_t>(M), output);
        else if (node.params.backend == ComputeBackend::METAL && cactus_metal_available())
            cactus_metal_quant_matmul(&mat, lhs, static_cast<uint32_t>(M), output);
        else
            cactus_quant_matmul(&mat, lhs, static_cast<uint32_t>(M), output);
    } else {
        if (lhs_buffer.precision != Precision::FP16) {
            throw std::runtime_error("FP16 matmul requires FP16 activations (got precision " + std::to_string(static_cast<int>(lhs_buffer.precision)) + ")");
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

namespace {
    thread_local std::vector<__fp16> moe_compact_hidden_buf;
    thread_local std::vector<__fp16> moe_gate_buf;
    thread_local std::vector<__fp16> moe_gate_pad_buf;
    thread_local std::vector<__fp16> moe_up_buf;
    thread_local std::vector<__fp16> moe_expert_out_buf;
    thread_local std::vector<size_t> moe_expert_offsets_buf;
    thread_local std::vector<size_t> moe_expert_tokens_buf;
    thread_local std::vector<float> moe_routing_denom_buf;

    void ensure_moe_buffers(size_t max_tokens, size_t hidden_dim, size_t intermediate_dim,
                            size_t num_experts, size_t top_k) {
        size_t hidden_size = max_tokens * hidden_dim;
        size_t inter_size = max_tokens * intermediate_dim;
        if (moe_compact_hidden_buf.size() < hidden_size) moe_compact_hidden_buf.resize(hidden_size);
        if (moe_gate_buf.size() < inter_size) moe_gate_buf.resize(inter_size);
        if (moe_up_buf.size() < inter_size) moe_up_buf.resize(inter_size);
        if (moe_expert_out_buf.size() < hidden_size) moe_expert_out_buf.resize(hidden_size);
        size_t total_assignments = max_tokens * top_k;
        if (moe_expert_offsets_buf.size() < num_experts + 1) moe_expert_offsets_buf.resize(num_experts + 1);
        if (moe_expert_tokens_buf.size() < total_assignments) moe_expert_tokens_buf.resize(total_assignments);
        if (moe_routing_denom_buf.size() < max_tokens) moe_routing_denom_buf.resize(max_tokens);
    }

    void moe_matmul(const __fp16* lhs,
                    size_t M,
                    size_t K,
                    const BufferDesc& rhs_buffer,
                    __fp16* output,
                    size_t N) {
        if (rhs_buffer.precision == Precision::FP16) {
            cactus_matmul_f16(lhs, rhs_buffer.data_as<__fp16>(), output, M, K, N);
            return;
        }

        if (PrecisionTraits::is_cq(rhs_buffer.precision) && rhs_buffer.group_size > 0) {
            CactusQuantMatrix mat = rhs_buffer.to_cq_matrix();
            if (rhs_buffer.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
                cactus_quant_orthogonal_matmul(&mat, lhs, static_cast<uint32_t>(M), output);
            else
                cactus_quant_matmul(&mat, lhs, static_cast<uint32_t>(M), output);
            return;
        }

        throw std::runtime_error("moe_layer only supports FP16 or TQ expert weights");
    }
}

void compute_moe_layer_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const size_t num_experts = node.params.num_experts;
    const size_t top_k = node.params.num_experts_per_tok;
    const bool normalize_routing = node.params.normalize_routing;
    const float eps = node.params.epsilon;
    const float routed_scaling_factor = node.params.scalar;
    const bool gated = node.params.moe_gated;
    const Activation activation = node.params.activation;
    const size_t base_inputs = gated ? (3 + 3 * num_experts) : (3 + 2 * num_experts);
    bool has_per_expert_scale = node.input_ids.size() == base_inputs + 1;
    if (node.input_ids.size() != base_inputs && node.input_ids.size() != base_inputs + 1) {
        throw std::runtime_error("moe_layer expects " + std::to_string(base_inputs) + " or " + std::to_string(base_inputs + 1) + " inputs, got " + std::to_string(node.input_ids.size()));
    }

    const auto& hidden_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& routing_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& topk_idx_buffer = get_input(node, 2, nodes, node_index_map);

    if (hidden_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
        throw std::runtime_error("moe_layer expects FP16 hidden/output");
    }
    if (topk_idx_buffer.precision != Precision::FP32) {
        throw std::runtime_error("moe_layer expects FP32 topk indices");
    }

    const __fp16* expert_scales_fp16 = nullptr;
    if (has_per_expert_scale) {
        const auto& scale_buffer = get_input(node, base_inputs, nodes, node_index_map);
        if (scale_buffer.precision != Precision::FP16) {
            throw std::runtime_error("moe_layer expects FP16 per_expert_scale");
        }
        expert_scales_fp16 = scale_buffer.data_as<__fp16>();
    }

    const size_t token_count = hidden_buffer.shape[0];
    const size_t hidden_dim = hidden_buffer.shape[1];
    const size_t total_num_experts = routing_buffer.shape[1];

    const auto& w1_0_buffer = get_input(node, 3, nodes, node_index_map);
    const size_t expert_intermediate_dim = w1_0_buffer.shape[0];

    const auto* hidden = hidden_buffer.data_as<__fp16>();
    auto* output = node.output_buffer.data_as<__fp16>();
    const auto* topk_idx = topk_idx_buffer.data_as<float>();
    const auto* routing_fp16 = routing_buffer.precision == Precision::FP16 ? routing_buffer.data_as<__fp16>() : nullptr;
    const auto* routing_fp32 = routing_buffer.precision == Precision::FP32 ? routing_buffer.data_as<float>() : nullptr;

    auto routing_prob = [&](size_t tok, size_t exp) -> float {
        const size_t offset = tok * total_num_experts + exp;
        if (routing_fp16) return static_cast<float>(routing_fp16[offset]);
        return routing_fp32[offset];
    };

    ensure_moe_buffers(token_count, hidden_dim, expert_intermediate_dim, num_experts, top_k);

    size_t* expert_offsets = moe_expert_offsets_buf.data(); 
    size_t* expert_tokens_flat = moe_expert_tokens_buf.data();  

    auto expert_index = [&](float raw_idx) -> size_t {
        if (!std::isfinite(raw_idx) || raw_idx < 0.0f) {
            throw std::runtime_error("moe_layer got invalid expert index");
        }
        size_t idx = static_cast<size_t>(raw_idx + 0.5f);
        if (idx >= num_experts) {
            throw std::runtime_error("moe_layer got expert index out of range");
        }
        return idx;
    };

    std::memset(expert_offsets, 0, (num_experts + 1) * sizeof(size_t));
    for (size_t tok = 0; tok < token_count; ++tok) {
        for (size_t k = 0; k < top_k; ++k) {
            expert_offsets[expert_index(topk_idx[tok * top_k + k]) + 1]++;
        }
    }
    
    for (size_t e = 0; e < num_experts; ++e) {
        expert_offsets[e + 1] += expert_offsets[e];
    }
    
    thread_local std::vector<size_t> moe_write_cursors;
    if (moe_write_cursors.size() < num_experts) moe_write_cursors.resize(num_experts);
    std::memcpy(moe_write_cursors.data(), expert_offsets, num_experts * sizeof(size_t));

    for (size_t tok = 0; tok < token_count; ++tok) {
        for (size_t k = 0; k < top_k; ++k) {
            size_t idx = expert_index(topk_idx[tok * top_k + k]);
            expert_tokens_flat[moe_write_cursors[idx]++] = tok;
        }
    }

    float* routing_denom = moe_routing_denom_buf.data();
    if (normalize_routing) {
        for (size_t tok = 0; tok < token_count; ++tok) {
            float sum_probs = 0.0f;
            for (size_t k = 0; k < top_k; ++k) {
                size_t idx = expert_index(topk_idx[tok * top_k + k]);
                sum_probs += routing_prob(tok, idx);
            }
            routing_denom[tok] = sum_probs + eps;
        }
    }

    std::memset(output, 0, token_count * hidden_dim * sizeof(__fp16));

    for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const size_t start = expert_offsets[expert_idx];
        const size_t end = expert_offsets[expert_idx + 1];
        if (start == end) continue;

        const size_t selected_count = end - start;
        const size_t* selected_tokens = expert_tokens_flat + start;

        const auto& w1_buffer = get_input(node, 3 + expert_idx, nodes, node_index_map);
        const auto& w2_buffer = gated
            ? get_input(node, 3 + 2 * num_experts + expert_idx, nodes, node_index_map)
            : get_input(node, 3 + num_experts + expert_idx, nodes, node_index_map);

        __fp16* compact_hidden = moe_compact_hidden_buf.data();
        for (size_t i = 0; i < selected_count; ++i) {
            std::memcpy(compact_hidden + i * hidden_dim,
                        hidden + selected_tokens[i] * hidden_dim,
                        hidden_dim * sizeof(__fp16));
        }

        __fp16* gate = moe_gate_buf.data();
        __fp16* up = moe_up_buf.data();
        __fp16* expert_out = moe_expert_out_buf.data();

        moe_matmul(compact_hidden, selected_count, hidden_dim, w1_buffer, gate, expert_intermediate_dim);

        switch (activation) {
            case Activation::GELU:
                cactus_gelu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::GELU_ERF:
                cactus_gelu_f16_erf(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::RELU:
                cactus_relu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::SIGMOID:
                cactus_sigmoid_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::TANH:
                cactus_tanh_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::SILU:
            default:
                cactus_silu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
        }

        if (gated) {
            const auto& w3_buffer = get_input(node, 3 + num_experts + expert_idx, nodes, node_index_map);
            moe_matmul(compact_hidden, selected_count, hidden_dim, w3_buffer, up, expert_intermediate_dim);
            cactus_multiply_f16(gate, up, gate, selected_count * expert_intermediate_dim);
        }

        const size_t w2_k = w2_buffer.shape.size() == 2 ? w2_buffer.shape[1] : 0;
        if (w2_k < expert_intermediate_dim) {
            throw std::runtime_error("moe_layer down-proj weight K smaller than expert intermediate dim");
        }
        const __fp16* w2_input = gate;
        if (w2_k != expert_intermediate_dim) {
            if (moe_gate_pad_buf.size() < selected_count * w2_k) moe_gate_pad_buf.resize(selected_count * w2_k);
            std::memset(moe_gate_pad_buf.data(), 0, selected_count * w2_k * sizeof(__fp16));
            for (size_t i = 0; i < selected_count; ++i) {
                std::memcpy(moe_gate_pad_buf.data() + i * w2_k,
                            gate + i * expert_intermediate_dim,
                            expert_intermediate_dim * sizeof(__fp16));
            }
            w2_input = moe_gate_pad_buf.data();
        }

        moe_matmul(w2_input, selected_count, w2_k, w2_buffer, expert_out, hidden_dim);

        for (size_t i = 0; i < selected_count; ++i) {
            const size_t tok = selected_tokens[i];
            float expert_prob = routing_prob(tok, expert_idx);
            if (expert_prob <= 0.0f) continue;

            float route_weight = expert_prob;
            if (normalize_routing) {
                route_weight = expert_prob / routing_denom[tok];
            }
            route_weight *= routed_scaling_factor;
            if (expert_scales_fp16) {
                route_weight *= static_cast<float>(expert_scales_fp16[expert_idx]);
            }

            auto* out_row = output + tok * hidden_dim;
            const auto* expert_row = expert_out + i * hidden_dim;
            cactus_add_scaled_f16(out_row, expert_row, out_row, hidden_dim, route_weight);
        }
    }
}

void compute_dense_mlp_tq_fused_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& hidden_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& gate_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& up_buffer = get_input(node, 2, nodes, node_index_map);
    const auto& down_buffer = get_input(node, 3, nodes, node_index_map);

    if (hidden_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
        throw std::runtime_error("dense_mlp_tq_fused expects FP16 hidden/output");
    }
    if (!PrecisionTraits::is_cq(gate_buffer.precision) || !PrecisionTraits::is_cq(up_buffer.precision) ||
        !PrecisionTraits::is_cq(down_buffer.precision) ||
        gate_buffer.group_size == 0 || up_buffer.group_size == 0 || down_buffer.group_size == 0) {
        throw std::runtime_error("dense_mlp_tq_fused expects TQ gate/up/down weights");
    }
    if (hidden_buffer.shape.empty() || gate_buffer.shape.size() != 2 || up_buffer.shape.size() != 2 || down_buffer.shape.size() != 2) {
        throw std::runtime_error("dense_mlp_tq_fused expects rank >=1 hidden and rank-2 weights");
    }

    const size_t hidden_dim = hidden_buffer.shape.back();
    size_t M = 1;
    for (size_t i = 0; i + 1 < hidden_buffer.shape.size(); ++i) M *= hidden_buffer.shape[i];

    const size_t d_ffn = gate_buffer.shape[0];
    if (up_buffer.shape[0] != d_ffn || gate_buffer.shape[1] != hidden_dim || up_buffer.shape[1] != hidden_dim ||
        down_buffer.shape[1] != d_ffn) {
        throw std::runtime_error("dense_mlp_tq_fused weight dimensions do not match hidden");
    }

    thread_local std::vector<__fp16> gate_buf;
    thread_local std::vector<__fp16> up_buf;
    thread_local std::vector<__fp16> prod_buf;
    const size_t inter_size = M * d_ffn;
    if (gate_buf.size() < inter_size) gate_buf.resize(inter_size);
    if (up_buf.size() < inter_size) up_buf.resize(inter_size);
    if (prod_buf.size() < inter_size) prod_buf.resize(inter_size);

    const __fp16* hidden = hidden_buffer.data_as<__fp16>();
    __fp16* gate = gate_buf.data();
    __fp16* up = up_buf.data();
    __fp16* prod = prod_buf.data();
    __fp16* output = node.output_buffer.data_as<__fp16>();

    CactusQuantMatrix gate_mat = gate_buffer.to_cq_matrix();
    CactusQuantMatrix up_mat = up_buffer.to_cq_matrix();
    CactusQuantMatrix down_mat = down_buffer.to_cq_matrix();
    const bool use_safe_product_scale = node.params.scalar != 0.0f && node.params.scalar != 1.0f;
    const bool trace_dense_mlp = std::getenv("CACTUS_TRACE_DENSE_MLP") != nullptr;

    cactus_quant_matmul(&gate_mat, hidden, static_cast<uint32_t>(M), gate);
    cactus_gelu_f16(gate, gate, inter_size);
    float max_gate_abs = 0.0f;
    size_t gate_nonfinite = 0;
    if (use_safe_product_scale) {
        const __fp16 scale = static_cast<__fp16>(node.params.scalar);
        for (size_t i = 0; i < inter_size; ++i) {
            float value = static_cast<float>(gate[i]);
            if (!std::isfinite(value)) {
                gate_nonfinite += 1;
                value = std::copysign(65504.0f, value);
            }
            value *= static_cast<float>(scale);
            max_gate_abs = std::max(max_gate_abs, std::abs(value));
            gate[i] = static_cast<__fp16>(value);
        }
    }
    cactus_quant_matmul(&up_mat, hidden, static_cast<uint32_t>(M), up);
    float max_up_abs = 0.0f;
    size_t up_nonfinite = 0;
    if (use_safe_product_scale) {
        for (size_t i = 0; i < inter_size; ++i) {
            float value = static_cast<float>(up[i]);
            if (!std::isfinite(value)) {
                up_nonfinite += 1;
                value = std::copysign(65504.0f, value);
            }
            max_up_abs = std::max(max_up_abs, std::abs(value));
            up[i] = static_cast<__fp16>(value);
        }
        const float product_bound = max_gate_abs * max_up_abs;
        constexpr float kSafeHadamardProductBound = 1024.0f;
        if (std::isfinite(product_bound) && product_bound > kSafeHadamardProductBound) {
            const __fp16 extra_scale = static_cast<__fp16>(kSafeHadamardProductBound / product_bound);
            for (size_t i = 0; i < inter_size; ++i) {
                gate[i] = static_cast<__fp16>(gate[i] * extra_scale);
            }
            max_gate_abs *= static_cast<float>(extra_scale);
        }
    }
    cactus_multiply_f16(gate, up, prod, inter_size);
    if (trace_dense_mlp) {
        size_t prod_nonfinite = 0;
        float max_prod_abs = 0.0f;
        for (size_t i = 0; i < inter_size; ++i) {
            float value = static_cast<float>(prod[i]);
            if (!std::isfinite(value)) {
                prod_nonfinite += 1;
                continue;
            }
            max_prod_abs = std::max(max_prod_abs, std::abs(value));
        }
        std::cerr << "[cactus:dense_mlp] id=" << node.id
                  << " scale=" << node.params.scalar
                  << " gate_nonfinite=" << gate_nonfinite
                  << " up_nonfinite=" << up_nonfinite
                  << " prod_nonfinite=" << prod_nonfinite
                  << " max_gate=" << max_gate_abs
                  << " max_up=" << max_up_abs
                  << " max_prod=" << max_prod_abs
                  << " shape=[";
        for (size_t i = 0; i < node.output_buffer.shape.size(); ++i) {
            if (i) std::cerr << ",";
            std::cerr << node.output_buffer.shape[i];
        }
        std::cerr << "]" << std::endl;
    }
    cactus_quant_matmul(&down_mat, prod, static_cast<uint32_t>(M), output);
    if (trace_dense_mlp) {
        size_t output_nonfinite = 0;
        float max_output_abs = 0.0f;
        const size_t output_count = node.output_buffer.total_size;
        for (size_t i = 0; i < output_count; ++i) {
            float value = static_cast<float>(output[i]);
            if (!std::isfinite(value)) {
                output_nonfinite += 1;
                continue;
            }
            max_output_abs = std::max(max_output_abs, std::abs(value));
        }
        std::cerr << "[cactus:dense_mlp_out] id=" << node.id
                  << " output_nonfinite=" << output_nonfinite
                  << " max_output=" << max_output_abs
                  << std::endl;
    }
}

void compute_rms_norm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buffer = get_input(node, 1, nodes, node_index_map);

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
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
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
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
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

void compute_rel_pos_bias_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                               const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 2) {
        throw std::runtime_error("REL_POS_BIAS requires 2 inputs (query, relative_key)");
    }

    const auto& q_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& r_buffer = get_input(node, 1, nodes, node_index_map);
    auto& y_buffer = node.output_buffer;

    if (q_buffer.shape.size() != 4) {
        throw std::runtime_error("REL_POS_BIAS query must be [B, T, H, D]");
    }
    if (r_buffer.shape.size() != 4) {
        throw std::runtime_error("REL_POS_BIAS relative_key must be [B, R, H, D]");
    }
    if (q_buffer.precision != Precision::FP16 || r_buffer.precision != Precision::FP16) {
        throw std::runtime_error("REL_POS_BIAS currently only supports FP16 tensors");
    }

    const size_t B = q_buffer.shape[0];
    const size_t T = q_buffer.shape[1];
    const size_t H = q_buffer.shape[2];
    const size_t D = q_buffer.shape[3];
    const size_t Rb = r_buffer.shape[0];
    const size_t R = r_buffer.shape[1];

    if (Rb != 1 && Rb != B) {
        throw std::runtime_error("REL_POS_BIAS relative_key batch must be 1 or match query batch");
    }
    if (r_buffer.shape[2] != H || r_buffer.shape[3] != D) {
        throw std::runtime_error("REL_POS_BIAS expects matching [H, D] between query and relative_key");
    }
    if (R < (2 * T - 1)) {
        throw std::runtime_error("REL_POS_BIAS requires relative_key length >= 2*T-1");
    }

    const __fp16* q = q_buffer.data_as<__fp16>();
    const __fp16* r = r_buffer.data_as<__fp16>();
    __fp16* y = y_buffer.data_as<__fp16>();

    const float scale = node.params.scale;

    const size_t q_batch_stride = T * H * D;
    const size_t r_batch_stride = R * H * D;
    const size_t y_batch_stride = H * T * T;
    const size_t q_head_stride = D;
    const size_t r_head_stride = D;
    const size_t q_time_stride = H * D;
    const size_t r_time_stride = H * D;

    CactusThreading::parallel_for(B * H * T, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t b = work_idx / (H * T);
                const size_t rem = work_idx % (H * T);
                const size_t h = rem / T;
                const size_t t = rem % T;

                const size_t rb = (Rb == 1) ? 0 : b;
                const __fp16* q_vec = q + b * q_batch_stride + t * q_time_stride + h * q_head_stride;
                const __fp16* r_base = r + rb * r_batch_stride + h * r_head_stride;
                __fp16* y_row = y + b * y_batch_stride + h * (T * T) + t * T;

                for (size_t j = 0; j < T; ++j) {
                    const size_t rel_idx = (T - 1) - t + j;
                    const __fp16* r_vec = r_base + rel_idx * r_time_stride;

                    float acc = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        acc += static_cast<float>(q_vec[d]) * static_cast<float>(r_vec[d]);
                    }
                    y_row[j] = static_cast<__fp16>(acc * scale);
                }
            }
        });
}

void compute_attention_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() < 3 || node.input_ids.size() > 4) {
        throw std::runtime_error("Attention operation requires 3 or 4 inputs (query, key, value[, mask]), got " +
                                std::to_string(node.input_ids.size()) + " inputs");
    }

    const auto& query_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& key_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& value_buffer = get_input(node, 2, nodes, node_index_map);
    const BufferDesc* mask_buffer = nullptr;
    if (node.input_ids.size() == 4) {
        mask_buffer = &get_input(node, 3, nodes, node_index_map);
    }
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
    size_t v_head_dim = value_buffer.shape[3];
    bool mask_per_head = false;
    const __fp16* mask_ptr = nullptr;

    if (mask_buffer) {
        if (mask_buffer->precision != Precision::FP16) {
            throw std::runtime_error("Attention mask tensor must be FP16");
        }

        if (mask_buffer->shape.size() == 3) {
            if (mask_buffer->shape[0] != batch_size ||
                mask_buffer->shape[1] != seq_len ||
                mask_buffer->shape[2] != kv_seq_len) {
                throw std::runtime_error("Attention mask [B, T, S] shape mismatch");
            }
            mask_per_head = false;
        } else if (mask_buffer->shape.size() == 4) {
            if (mask_buffer->shape[0] != batch_size ||
                mask_buffer->shape[1] != num_q_heads ||
                mask_buffer->shape[2] != seq_len ||
                mask_buffer->shape[3] != kv_seq_len) {
                throw std::runtime_error("Attention mask [B, H, T, S] shape mismatch");
            }
            mask_per_head = true;
        } else {
            throw std::runtime_error("Attention mask must be rank 3 or 4");
        }

        mask_ptr = mask_buffer->data_as<__fp16>();
    }

    cactus_attention_f16(query_buffer.data_as<__fp16>(), key_buffer.data_as<__fp16>(),
                         value_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                         batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, mask_ptr,
                         node.params.position_offset, node.params.window_size, node.params.is_causal,
                         node.params.attention_mask_is_additive, mask_per_head, v_head_dim, node.params.logit_cap);
}

void compute_attention_int8_hybrid_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& query_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& key_new_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& value_new_buffer = get_input(node, 2, nodes, node_index_map);
    const auto& q_shape = query_buffer.shape;

    if (q_shape.size() < 4) {
        throw std::runtime_error("ATTENTION_INT8_HYBRID requires 4D query tensor");
    }

    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];
    size_t head_dim = node.params.head_dim;
    size_t v_head_dim = node.params.v_head_dim;
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
        node.params.window_size, KV_QUANT_GROUP_SIZE, v_head_dim
    );
}

void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buffer = get_input(node, 1, nodes, node_index_map);
    bool has_bias = node.input_ids.size() > 2;
    float epsilon = node.params.epsilon;

    if (input_buffer.shape.empty()) {
        throw std::runtime_error("LayerNorm requires non-empty input tensor");
    }

    size_t feature_size = input_buffer.shape.back();
    size_t batch_size = input_buffer.total_size / feature_size;

    if (weight_buffer.total_size != feature_size) {
        throw std::runtime_error("LayerNorm weight size mismatch with input feature dimension");
    }

    using BufferDesc = std::remove_reference_t<decltype(weight_buffer)>;
    const BufferDesc* bias_buffer_ptr = nullptr;
    if (has_bias) {
        const auto& bias_buffer = get_input(node, 2, nodes, node_index_map);
        if (bias_buffer.total_size != feature_size) {
            throw std::runtime_error("LayerNorm bias size mismatch with input feature dimension");
        }
        bias_buffer_ptr = &bias_buffer;
    }

    if (input_buffer.precision == Precision::FP16 &&
        weight_buffer.precision == Precision::FP16 &&
        node.output_buffer.precision == Precision::FP16 &&
        (!has_bias || bias_buffer_ptr->precision == Precision::FP16)) {
        cactus_layer_norm_f16(
            input_buffer.data_as<__fp16>(),
            weight_buffer.data_as<__fp16>(),
            has_bias ? bias_buffer_ptr->data_as<__fp16>() : nullptr,
            node.output_buffer.data_as<__fp16>(),
            batch_size,
            feature_size,
            epsilon);
        return;
    }

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
        const auto& bias_buffer = *bias_buffer_ptr;
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









void compute_glu_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                      const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.empty()) {
        throw std::runtime_error("GLU expects non-scalar input");
    }

    int axis = node.params.axis;
    if (axis < 0) axis += static_cast<int>(X.shape.size());
    if (axis < 0 || static_cast<size_t>(axis) >= X.shape.size()) {
        throw std::runtime_error("GLU axis out of range");
    }

    const size_t axis_size = X.shape[static_cast<size_t>(axis)];
    if ((axis_size % 2) != 0) {
        throw std::runtime_error("GLU split dimension must be even");
    }
    const size_t split = axis_size / 2;

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= X.shape[static_cast<size_t>(i)];
    }
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < X.shape.size(); ++i) {
        inner *= X.shape[i];
    }

    std::vector<size_t> out_shape = X.shape;
    out_shape[static_cast<size_t>(axis)] = split;
    Y.shape = out_shape;
    Y.precision = X.precision;

    if (X.precision == Precision::FP16) {
        cactus_glu_f16(X.data_as<__fp16>(), Y.data_as<__fp16>(), outer, split, inner);
        return;
    }

    if (X.precision == Precision::FP32) {
        cactus_glu_f32(X.data_as<float>(), Y.data_as<float>(), outer, split, inner);
        return;
    }

    throw std::runtime_error("GLU only supports FP16/FP32");
}





void compute_groupnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& weight = get_input(node, 1, nodes, node_index_map);
    const auto& bias = get_input(node, 2, nodes, node_index_map);
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
