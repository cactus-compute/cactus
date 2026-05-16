#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace {
    const __fp16* as_fp16_ptr(const BufferDesc& buffer, std::vector<__fp16>& scratch) {
        if (buffer.precision == Precision::FP16) {
            return buffer.data_as<__fp16>();
        }
        if (buffer.precision == Precision::FP32) {
            scratch.resize(buffer.total_size);
            cactus_fp32_to_fp16(buffer.data_as<float>(), scratch.data(), buffer.total_size);
            return scratch.data();
        }
        throw std::runtime_error("GATED_DELTANET unsupported precision (expected FP16/FP32)");
    }

    void validate_gated_deltanet_inputs(
        const BufferDesc& q,
        const BufferDesc& k,
        const BufferDesc& v,
        const BufferDesc& g,
        const BufferDesc& b,
        const BufferDesc& s) {
        auto is_supported_precision = [](Precision p) {
            return p == Precision::FP16 || p == Precision::FP32;
        };
        if (!is_supported_precision(q.precision) || !is_supported_precision(k.precision) ||
            !is_supported_precision(v.precision) || !is_supported_precision(g.precision) ||
            !is_supported_precision(b.precision) || !is_supported_precision(s.precision)) {
            throw std::runtime_error("GATED_DELTANET requires FP16/FP32 inputs");
        }

        if (q.shape.size() != 4 || k.shape.size() != 4 || v.shape.size() != 4) {
            throw std::runtime_error("GATED_DELTANET expects query/key/value rank 4 [B, T, H, D]");
        }
        if (g.shape.size() != 3 || b.shape.size() != 3) {
            throw std::runtime_error("GATED_DELTANET expects gate_log/beta rank 3 [B, T, H]");
        }
        if (s.shape.size() != 4) {
            throw std::runtime_error("GATED_DELTANET expects state rank 4 [B, K, H, V]");
        }

        const size_t B = q.shape[0];
        const size_t T = q.shape[1];
        const size_t Hq = q.shape[2];
        const size_t K = q.shape[3];

        if (k.shape[0] != B || k.shape[1] != T || k.shape[2] != Hq || k.shape[3] != K) {
            throw std::runtime_error("GATED_DELTANET query/key shape mismatch");
        }
        if (v.shape[0] != B || v.shape[1] != T) {
            throw std::runtime_error("GATED_DELTANET value shape mismatch");
        }
        const size_t Hv = v.shape[2];
        if (g.shape[0] != B || g.shape[1] != T || g.shape[2] != Hv ||
            b.shape[0] != B || b.shape[1] != T || b.shape[2] != Hv) {
            throw std::runtime_error("GATED_DELTANET gate_log/beta shape mismatch");
        }
        if (Hq == 0 || Hv == 0 || (Hv % Hq) != 0) {
            throw std::runtime_error("GATED_DELTANET expects value heads divisible by q/k heads");
        }
        const size_t V = v.shape[3];
        if (s.shape[0] != B || s.shape[1] != K || s.shape[2] != Hv || s.shape[3] != V) {
            throw std::runtime_error("GATED_DELTANET state shape mismatch");
        }
    }


}

void compute_gated_deltanet_decode_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                        const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 6) {
        throw std::runtime_error("GATED_DELTANET_DECODE expects 6 inputs");
    }

    const auto& q = get_input(node, 0, nodes, node_index_map);
    const auto& k = get_input(node, 1, nodes, node_index_map);
    const auto& v = get_input(node, 2, nodes, node_index_map);
    const auto& g = get_input(node, 3, nodes, node_index_map);
    const auto& b = get_input(node, 4, nodes, node_index_map);
    const auto& s = get_input(node, 5, nodes, node_index_map);

    validate_gated_deltanet_inputs(q, k, v, g, b, s);
    if (q.shape[1] != 1) {
        throw std::runtime_error("GATED_DELTANET_DECODE expects T=1");
    }

    const size_t B = q.shape[0];
    const size_t Hq = q.shape[2];
    const size_t K = q.shape[3];
    const size_t Hv = v.shape[2];
    const size_t V = v.shape[3];
    const size_t qk_heads_from_params = node.params.num_kv_heads;
    if (qk_heads_from_params != 0 && qk_heads_from_params != Hq) {
        throw std::runtime_error("GATED_DELTANET_DECODE num_qk_heads param mismatch");
    }

    std::vector<__fp16> q_cast;
    std::vector<__fp16> k_cast;
    std::vector<__fp16> v_cast;
    std::vector<__fp16> g_cast;
    std::vector<__fp16> b_cast;
    std::vector<__fp16> s_cast;
    const __fp16* q_data = as_fp16_ptr(q, q_cast);
    const __fp16* k_data = as_fp16_ptr(k, k_cast);
    const __fp16* v_data = as_fp16_ptr(v, v_cast);
    const __fp16* g_data = as_fp16_ptr(g, g_cast);
    const __fp16* b_data = as_fp16_ptr(b, b_cast);
    const __fp16* s_data = as_fp16_ptr(s, s_cast);
    __fp16* out = node.output_buffer.data_as<__fp16>();

    cactus_gated_deltanet_decode_f16(
        q_data, k_data, v_data, g_data, b_data, s_data, out,
        B, Hq, Hv, K, V, node.params.scale);
}


void compute_gated_deltanet_prefill_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                         const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 6) {
        throw std::runtime_error("GATED_DELTANET_PREFILL expects 6 inputs");
    }

    const auto& q = get_input(node, 0, nodes, node_index_map);
    const auto& k = get_input(node, 1, nodes, node_index_map);
    const auto& v = get_input(node, 2, nodes, node_index_map);
    const auto& g = get_input(node, 3, nodes, node_index_map);
    const auto& b = get_input(node, 4, nodes, node_index_map);
    const auto& s = get_input(node, 5, nodes, node_index_map);

    validate_gated_deltanet_inputs(q, k, v, g, b, s);

    const size_t B = q.shape[0];
    const size_t T = q.shape[1];
    const size_t Hq = q.shape[2];
    const size_t K = q.shape[3];
    const size_t Hv = v.shape[2];
    const size_t V = v.shape[3];
    const size_t qk_heads_from_params = node.params.num_kv_heads;
    if (qk_heads_from_params != 0 && qk_heads_from_params != Hq) {
        throw std::runtime_error("GATED_DELTANET_PREFILL num_qk_heads param mismatch");
    }

    std::vector<__fp16> q_cast;
    std::vector<__fp16> k_cast;
    std::vector<__fp16> v_cast;
    std::vector<__fp16> g_cast;
    std::vector<__fp16> b_cast;
    std::vector<__fp16> s_cast;
    const __fp16* q_data = as_fp16_ptr(q, q_cast);
    const __fp16* k_data = as_fp16_ptr(k, k_cast);
    const __fp16* v_data = as_fp16_ptr(v, v_cast);
    const __fp16* g_data = as_fp16_ptr(g, g_cast);
    const __fp16* b_data = as_fp16_ptr(b, b_cast);
    const __fp16* s_data = as_fp16_ptr(s, s_cast);
    __fp16* out = node.output_buffer.data_as<__fp16>();

    cactus_gated_deltanet_prefill_f16(
        q_data, k_data, v_data, g_data, b_data, s_data, out,
        B, T, Hq, Hv, K, V, node.params.chunk_size, node.params.scale);
}


void compute_rope_gptj_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
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

void compute_lstm_cell_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& h_prev_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& c_prev_buffer = get_input(node, 2, nodes, node_index_map);
    const auto& weight_ih_buffer = get_input(node, 3, nodes, node_index_map);
    const auto& weight_hh_buffer = get_input(node, 4, nodes, node_index_map);
    const auto& bias_ih_buffer = get_input(node, 5, nodes, node_index_map);
    const auto& bias_hh_buffer = get_input(node, 6, nodes, node_index_map);

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

void compute_altup_predict_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    size_t n = node.params.num_altup_inputs;
    const auto& coefs_buf = get_input(node, 0, nodes, node_index_map);

    std::vector<const __fp16*> stream_ptrs(n);
    for (size_t i = 0; i < n; i++) {
        stream_ptrs[i] = get_input(node, 1 + i, nodes, node_index_map).data_as<__fp16>();
    }

    const auto& stream0_buf = get_input(node, 1, nodes, node_index_map);
    size_t seq_len = stream0_buf.shape[0];
    size_t hidden_dim = stream0_buf.shape[1];

    cactus_altup_predict_f16(
        coefs_buf.data_as<__fp16>(),
        stream_ptrs.data(),
        node.output_buffer.data_as<__fp16>(),
        n, seq_len, hidden_dim);
}

void compute_gaussian_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buf = get_input(node, 0, nodes, node_index_map);
    const __fp16* input = input_buf.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();

    size_t rows = input_buf.shape[0];
    size_t cols = input_buf.shape[1];
    float ppf = node.params.scalar;

    cactus_gaussian_topk_f16(input, output, rows, cols, ppf);
}

void compute_altup_correct_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    size_t n = node.params.num_altup_inputs;
    const auto& coefs_buf = get_input(node, 0, nodes, node_index_map);
    const auto& innov_buf = get_input(node, 1, nodes, node_index_map);

    std::vector<const __fp16*> pred_ptrs(n);
    for (size_t i = 0; i < n; i++) {
        pred_ptrs[i] = get_input(node, 2 + i, nodes, node_index_map).data_as<__fp16>();
    }

    size_t seq_len = innov_buf.shape[0];
    size_t hidden_dim = innov_buf.shape[1];

    cactus_altup_correct_f16(
        coefs_buf.data_as<__fp16>(),
        innov_buf.data_as<__fp16>(),
        pred_ptrs.data(),
        node.output_buffer.data_as<__fp16>(),
        n, seq_len, hidden_dim);
}

void compute_bilstm_sequence_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                   const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& w_ih_fwd = get_input(node, 1, nodes, node_index_map);
    const auto& w_hh_fwd = get_input(node, 2, nodes, node_index_map);
    const auto& b_ih_fwd = get_input(node, 3, nodes, node_index_map);
    const auto& b_hh_fwd = get_input(node, 4, nodes, node_index_map);
    const auto& w_ih_bwd = get_input(node, 5, nodes, node_index_map);
    const auto& w_hh_bwd = get_input(node, 6, nodes, node_index_map);
    const auto& b_ih_bwd = get_input(node, 7, nodes, node_index_map);
    const auto& b_hh_bwd = get_input(node, 8, nodes, node_index_map);

    size_t batch_size = input.shape[0];
    size_t seq_len = input.shape[1];
    size_t input_size = input.shape[2];
    size_t hidden_size = w_ih_fwd.shape[0] / 4;

    cactus_bilstm_sequence_f16(
        input.data_as<__fp16>(),
        w_ih_fwd.data_as<__fp16>(), w_hh_fwd.data_as<__fp16>(),
        b_ih_fwd.data_as<__fp16>(), b_hh_fwd.data_as<__fp16>(),
        w_ih_bwd.data_as<__fp16>(), w_hh_bwd.data_as<__fp16>(),
        b_ih_bwd.data_as<__fp16>(), b_hh_bwd.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, seq_len, input_size, hidden_size);
}

void compute_stats_pool_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                              const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const __fp16* src = input.data_as<__fp16>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    size_t batch = input.shape[0];
    size_t total_per_batch = input.total_size / batch;
    size_t T = input.shape.back();
    size_t features = total_per_batch / T;

    for (size_t b = 0; b < batch; ++b) {
        const __fp16* batch_src = src + b * total_per_batch;
        __fp16* batch_dst = dst + b * features * 2;

        for (size_t f = 0; f < features; ++f) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                float v = static_cast<float>(batch_src[f * T + t]);
                sum += v;
                sum_sq += v * v;
            }
            float mean = sum / static_cast<float>(T);
            float var = T > 1 ? (sum_sq - static_cast<float>(T) * mean * mean) / static_cast<float>(T - 1) : 0.0f;
            float std_val = sqrtf(fmaxf(var, 0.0f));
            batch_dst[f] = static_cast<__fp16>(mean);
            batch_dst[features + f] = static_cast<__fp16>(std_val);
        }
    }
}

void compute_weighted_stats_pool_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                       const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buf = get_input(node, 1, nodes, node_index_map);
    const __fp16* src = input.data_as<__fp16>();
    const float* weights = weight_buf.data_as<float>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    size_t batch = input.shape[0];
    size_t total_per_batch = input.total_size / batch;
    size_t T = input.shape.back();
    size_t features = total_per_batch / T;

    constexpr float eps = 1e-8f;

    for (size_t b = 0; b < batch; ++b) {
        const __fp16* batch_src = src + b * total_per_batch;
        const float* batch_w = weights + b * T;
        __fp16* batch_dst = dst + b * features * 2;

        float v1 = 0.0f, v2 = 0.0f;
        for (size_t t = 0; t < T; ++t) {
            float w = batch_w[t];
            v1 += w;
            v2 += w * w;
        }
        float v1_safe = v1 + eps;
        float var_denom = v1_safe - v2 / v1_safe + eps;

        for (size_t f = 0; f < features; ++f) {
            float wsum = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                wsum += static_cast<float>(batch_src[f * T + t]) * batch_w[t];
            }
            float mean = wsum / v1_safe;

            float wvar = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                float dx = static_cast<float>(batch_src[f * T + t]) - mean;
                wvar += batch_w[t] * dx * dx;
            }
            float std_val = sqrtf(fmaxf(wvar / var_denom, 0.0f));

            batch_dst[f] = static_cast<__fp16>(mean);
            batch_dst[features + f] = static_cast<__fp16>(std_val);
        }
    }
}

