#include "../cactus_graph.h"
#include "cactus_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t DSV4_HC = 4;
constexpr size_t DSV4_HC_MIX_SIZE = (2 + DSV4_HC) * DSV4_HC;
constexpr float PI_F = 3.14159265358979323846f;

float read_scalar(const BufferDesc& buffer, size_t idx);

void read_hidden_token_fp16(const BufferDesc& hidden, size_t token, std::vector<__fp16>& out) {
    const size_t hidden_dim = hidden.shape[1];
    out.resize(hidden_dim);
    for (size_t d = 0; d < hidden_dim; ++d) {
        out[d] = static_cast<__fp16>(read_scalar(hidden, token * hidden_dim + d));
    }
}

void matmul_vec_weight(const std::vector<__fp16>& x, const BufferDesc& weight, std::vector<float>& out) {
    if (weight.shape.size() != 2 || weight.shape[1] != x.size()) {
        throw std::runtime_error("DeepSeek vec matmul weight shape mismatch");
    }
    const size_t rows = weight.shape[0];
    out.assign(rows, 0.0f);
    if (PrecisionTraits::is_cq(weight.precision) && weight.group_size > 0) {
        std::vector<__fp16> y(rows);
        CactusQuantMatrix mat = weight.to_cq_matrix();
        if (weight.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
            cactus_quant_orthogonal_matmul(&mat, x.data(), 1, y.data());
        else
            cactus_quant_matmul(&mat, x.data(), 1, y.data());
        for (size_t i = 0; i < rows; ++i) out[i] = static_cast<float>(y[i]);
        return;
    }
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t d = 0; d < x.size(); ++d) {
            acc += static_cast<float>(x[d]) * read_scalar(weight, r * x.size() + d);
        }
        out[r] = acc;
    }
}

void matmul_batch_weight(const std::vector<__fp16>& x, size_t rows, size_t cols,
                         const BufferDesc& weight, std::vector<__fp16>& out) {
    if (weight.shape.size() != 2 || weight.shape[1] != cols) {
        throw std::runtime_error("DeepSeek batch matmul weight shape mismatch");
    }
    const size_t n = weight.shape[0];
    out.assign(rows * n, static_cast<__fp16>(0.0f));
    if (PrecisionTraits::is_cq(weight.precision) && weight.group_size > 0) {
        CactusQuantMatrix mat = weight.to_cq_matrix();
        if (weight.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
            cactus_quant_orthogonal_matmul(&mat, x.data(), static_cast<uint32_t>(rows), out.data());
        else
            cactus_quant_matmul(&mat, x.data(), static_cast<uint32_t>(rows), out.data());
        return;
    }
    if (weight.precision == Precision::FP16) {
        cactus_matmul_f16(x.data(), weight.data_as<__fp16>(), out.data(), rows, cols, n);
        return;
    }
    for (size_t m = 0; m < rows; ++m) {
        for (size_t r = 0; r < n; ++r) {
            float acc = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                acc += static_cast<float>(x[m * cols + c]) * read_scalar(weight, r * cols + c);
            }
            out[m * n + r] = static_cast<__fp16>(acc);
        }
    }
}

uint8_t read_packed_bits(const uint8_t* data, size_t bit_offset, uint32_t bits) {
    uint32_t value = 0;
    for (uint32_t b = 0; b < bits; ++b) {
        const size_t absolute = bit_offset + b;
        value |= ((data[absolute / 8] >> (absolute % 8)) & 1u) << b;
    }
    return static_cast<uint8_t>(value);
}

float read_grouped_int8_scalar(const BufferDesc& buffer, size_t row, size_t col) {
    const int8_t* data = buffer.data_as<int8_t>();
    const __fp16* scales = reinterpret_cast<const __fp16*>(buffer.activation_scales_data);
    const size_t group = col / buffer.group_size;
    if ((buffer.cq_flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0 && buffer.shape.size() == 2) {
        const size_t k = buffer.group_size * buffer.num_groups;
        const size_t block = row / 4;
        const size_t lane = row % 4;
        const size_t k4 = col / 4;
        const size_t kk = col % 4;
        const int8_t q = data[((block * (k / 4) + k4) * 4 + lane) * 4 + kk];
        return static_cast<float>(q) * static_cast<float>(scales[(block * buffer.num_groups + group) * 4 + lane]);
    }
    return static_cast<float>(data[row * buffer.shape[1] + col]) *
           static_cast<float>(scales[row * buffer.num_groups + group]);
}

float read_cq_scalar(const BufferDesc& buffer, size_t row, size_t col) {
    const uint32_t bits = PrecisionTraits::cq_bits(buffer.precision);
    const size_t pgb = (buffer.group_size * bits + 7) / 8;
    const size_t group = col / buffer.group_size;
    const size_t local = col % buffer.group_size;
    const uint8_t* packed = buffer.data_as<uint8_t>() + (row * buffer.num_groups + group) * pgb;
    const uint8_t code = read_packed_bits(packed, local * bits, bits);
    const float codebook = buffer.cq_codebook ? static_cast<float>(buffer.cq_codebook[code]) : static_cast<float>(code);
    const float norm = buffer.cq_norms ? static_cast<float>(buffer.cq_norms[row * buffer.num_groups + group]) : 1.0f;
    return codebook * norm;
}

float read_scalar(const BufferDesc& buffer, size_t idx) {
    if (buffer.precision == Precision::FP32) {
        return buffer.data_as<float>()[idx];
    }
    if (buffer.precision == Precision::FP16) {
        return static_cast<float>(buffer.data_as<__fp16>()[idx]);
    }
    if (buffer.precision == Precision::INT8 && buffer.group_size > 0 && buffer.activation_scales_data != nullptr) {
        if (buffer.shape.size() == 1) return read_grouped_int8_scalar(buffer, 0, idx);
        return read_grouped_int8_scalar(buffer, idx / buffer.shape[1], idx % buffer.shape[1]);
    }
    if (PrecisionTraits::is_cq(buffer.precision) && buffer.group_size > 0) {
        if (buffer.shape.size() == 1) return read_cq_scalar(buffer, 0, idx);
        return read_cq_scalar(buffer, idx / buffer.shape[1], idx % buffer.shape[1]);
    }
    std::ostringstream os;
    os << "DeepSeek V4 op expects FP16, FP32, grouped INT8, or CQ input; got precision="
       << static_cast<int>(buffer.precision)
       << " group_size=" << buffer.group_size
       << " num_groups=" << buffer.num_groups
       << " scales=" << (buffer.activation_scales_data ? 1 : 0)
       << " shape=[";
    for (size_t i = 0; i < buffer.shape.size(); ++i) {
        if (i) os << ",";
        os << buffer.shape[i];
    }
    os << "]";
    throw std::runtime_error(os.str());
}

void write_scalar(BufferDesc& buffer, size_t idx, float value) {
    if (buffer.precision == Precision::FP32) {
        buffer.data_as<float>()[idx] = value;
        return;
    }
    if (buffer.precision == Precision::FP16) {
        buffer.data_as<__fp16>()[idx] = static_cast<__fp16>(value);
        return;
    }
    throw std::runtime_error("DeepSeek V4 op expects FP16 or FP32 output");
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

float silu(float x) {
    return x * sigmoid(x);
}

float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float sqrt_softplus(float x) {
    return std::sqrt(softplus(x));
}

void validate_hidden_and_router_weight(const BufferDesc& hidden,
                                       const BufferDesc& weight,
                                       size_t num_experts,
                                       const char* op_name) {
    if (hidden.shape.size() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects hidden [T,D]");
    }
    if (weight.shape != std::vector<size_t>{num_experts, hidden.shape[1]}) {
        throw std::runtime_error(std::string(op_name) + " router weight shape mismatch");
    }
    if (hidden.precision != Precision::FP16 && hidden.precision != Precision::FP32) {
        throw std::runtime_error(std::string(op_name) + " expects FP16 or FP32 hidden");
    }
    if (weight.precision != Precision::FP16 && weight.precision != Precision::FP32 &&
        weight.precision != Precision::INT8 && !PrecisionTraits::is_cq(weight.precision)) {
        throw std::runtime_error(std::string(op_name) + " expects FP16, FP32, INT8, or CQ router weight");
    }
}

float router_logit(const BufferDesc& hidden, const BufferDesc& weight, size_t token, size_t expert) {
    const size_t hidden_dim = hidden.shape[1];
    float acc = 0.0f;
    for (size_t d = 0; d < hidden_dim; ++d) {
        acc += read_scalar(hidden, token * hidden_dim + d) *
               read_scalar(weight, expert * hidden_dim + d);
    }
    return acc;
}

void write_route(BufferDesc& route, size_t tokens, size_t token, size_t k, size_t expert, float weight) {
    const size_t top_k = route.shape[2];
    write_scalar(route, token * top_k + k, static_cast<float>(expert));
    write_scalar(route, tokens * top_k + token * top_k + k, weight);
}

void sinkhorn_4x4(float comb[DSV4_HC][DSV4_HC], size_t iters, float eps) {
    for (size_t row = 0; row < DSV4_HC; ++row) {
        float max_value = comb[row][0];
        for (size_t col = 1; col < DSV4_HC; ++col) {
            max_value = std::max(max_value, comb[row][col]);
        }
        float sum = 0.0f;
        for (size_t col = 0; col < DSV4_HC; ++col) {
            comb[row][col] = std::exp(comb[row][col] - max_value);
            sum += comb[row][col];
        }
        for (size_t col = 0; col < DSV4_HC; ++col) {
            comb[row][col] = comb[row][col] / sum + eps;
        }
    }

    float col_sum[DSV4_HC] = {};
    for (size_t col = 0; col < DSV4_HC; ++col) {
        for (size_t row = 0; row < DSV4_HC; ++row) {
            col_sum[col] += comb[row][col];
        }
    }
    for (size_t row = 0; row < DSV4_HC; ++row) {
        for (size_t col = 0; col < DSV4_HC; ++col) {
            comb[row][col] /= (col_sum[col] + eps);
        }
    }

    for (size_t iter = 1; iter < iters; ++iter) {
        for (size_t row = 0; row < DSV4_HC; ++row) {
            float row_sum = 0.0f;
            for (size_t col = 0; col < DSV4_HC; ++col) row_sum += comb[row][col];
            for (size_t col = 0; col < DSV4_HC; ++col) comb[row][col] /= (row_sum + eps);
        }
        std::fill(std::begin(col_sum), std::end(col_sum), 0.0f);
        for (size_t col = 0; col < DSV4_HC; ++col) {
            for (size_t row = 0; row < DSV4_HC; ++row) {
                col_sum[col] += comb[row][col];
            }
        }
        for (size_t row = 0; row < DSV4_HC; ++row) {
            for (size_t col = 0; col < DSV4_HC; ++col) {
                comb[row][col] /= (col_sum[col] + eps);
            }
        }
    }
}

void validate_mix_shape(const BufferDesc& mix, size_t tokens) {
    if (mix.shape != std::vector<size_t>{tokens, DSV4_HC_MIX_SIZE}) {
        throw std::runtime_error("DeepSeek V4 HC op expects mix shape [T,24]");
    }
}

} // namespace

void compute_dsv4_hc_mix_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& streams = get_input(node, 0, nodes, node_index_map);
    const auto& fn = get_input(node, 1, nodes, node_index_map);
    const auto& base = get_input(node, 2, nodes, node_index_map);
    const auto& scale = get_input(node, 3, nodes, node_index_map);

    if (streams.shape.size() != 3 || streams.shape[1] != DSV4_HC) {
        throw std::runtime_error("DSV4_HC_MIX expects streams [T,4,D]");
    }
    const size_t tokens = streams.shape[0];
    const size_t hidden = streams.shape[2];
    const size_t flat_dim = DSV4_HC * hidden;
    if (fn.shape != std::vector<size_t>{DSV4_HC_MIX_SIZE, flat_dim} ||
        base.shape != std::vector<size_t>{DSV4_HC_MIX_SIZE} ||
        scale.shape != std::vector<size_t>{3}) {
        throw std::runtime_error("DSV4_HC_MIX weight shape mismatch");
    }
    if (node.output_buffer.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_HC_MIX output must be FP32");
    }

    const float eps = node.params.epsilon;
    const size_t sinkhorn_iters = std::max<size_t>(1, node.params.chunk_size);
    auto* output = node.output_buffer.data_as<float>();
    std::vector<float> normed(flat_dim);

    for (size_t t = 0; t < tokens; ++t) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat_dim; ++i) {
            float value = read_scalar(streams, t * flat_dim + i);
            mean_square += static_cast<double>(value) * value;
        }
        float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / flat_dim) + eps);
        for (size_t i = 0; i < flat_dim; ++i) {
            normed[i] = read_scalar(streams, t * flat_dim + i) * rsqrt;
        }

        float mixes[DSV4_HC_MIX_SIZE] = {};
        for (size_t m = 0; m < DSV4_HC_MIX_SIZE; ++m) {
            float acc = 0.0f;
            for (size_t i = 0; i < flat_dim; ++i) {
                acc += read_scalar(fn, m * flat_dim + i) * normed[i];
            }
            mixes[m] = acc;
        }

        const float pre_scale = read_scalar(scale, 0);
        const float post_scale = read_scalar(scale, 1);
        const float comb_scale = read_scalar(scale, 2);
        for (size_t i = 0; i < DSV4_HC; ++i) {
            output[t * DSV4_HC_MIX_SIZE + i] = sigmoid(mixes[i] * pre_scale + read_scalar(base, i)) + eps;
            output[t * DSV4_HC_MIX_SIZE + DSV4_HC + i] =
                2.0f * sigmoid(mixes[DSV4_HC + i] * post_scale + read_scalar(base, DSV4_HC + i));
        }

        float comb[DSV4_HC][DSV4_HC] = {};
        for (size_t row = 0; row < DSV4_HC; ++row) {
            for (size_t col = 0; col < DSV4_HC; ++col) {
                size_t idx = 2 * DSV4_HC + row * DSV4_HC + col;
                comb[row][col] = mixes[idx] * comb_scale + read_scalar(base, idx);
            }
        }
        sinkhorn_4x4(comb, sinkhorn_iters, eps);
        for (size_t row = 0; row < DSV4_HC; ++row) {
            for (size_t col = 0; col < DSV4_HC; ++col) {
                size_t idx = 2 * DSV4_HC + row * DSV4_HC + col;
                output[t * DSV4_HC_MIX_SIZE + idx] = comb[row][col];
            }
        }
    }
}

void compute_dsv4_hc_collapse_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& streams = get_input(node, 0, nodes, node_index_map);
    const auto& mix = get_input(node, 1, nodes, node_index_map);
    if (streams.shape.size() != 3 || streams.shape[1] != DSV4_HC) {
        throw std::runtime_error("DSV4_HC_COLLAPSE expects streams [T,4,D]");
    }
    const size_t tokens = streams.shape[0];
    const size_t hidden = streams.shape[2];
    validate_mix_shape(mix, tokens);

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0.0f;
            for (size_t h = 0; h < DSV4_HC; ++h) {
                acc += read_scalar(mix, t * DSV4_HC_MIX_SIZE + h) *
                       read_scalar(streams, (t * DSV4_HC + h) * hidden + d);
            }
            write_scalar(node.output_buffer, t * hidden + d, acc);
        }
    }
}

void compute_dsv4_hc_post_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& sublayer = get_input(node, 0, nodes, node_index_map);
    const auto& residual = get_input(node, 1, nodes, node_index_map);
    const auto& mix = get_input(node, 2, nodes, node_index_map);
    if (residual.shape.size() != 3 || residual.shape[1] != DSV4_HC) {
        throw std::runtime_error("DSV4_HC_POST expects residual streams [T,4,D]");
    }
    const size_t tokens = residual.shape[0];
    const size_t hidden = residual.shape[2];
    if (sublayer.shape != std::vector<size_t>{tokens, hidden}) {
        throw std::runtime_error("DSV4_HC_POST expects sublayer output [T,D]");
    }
    validate_mix_shape(mix, tokens);

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t k = 0; k < DSV4_HC; ++k) {
            const float post = read_scalar(mix, t * DSV4_HC_MIX_SIZE + DSV4_HC + k);
            for (size_t d = 0; d < hidden; ++d) {
                float acc = post * read_scalar(sublayer, t * hidden + d);
                for (size_t j = 0; j < DSV4_HC; ++j) {
                    const size_t comb_idx = t * DSV4_HC_MIX_SIZE + 2 * DSV4_HC + j * DSV4_HC + k;
                    acc += read_scalar(mix, comb_idx) *
                           read_scalar(residual, (t * DSV4_HC + j) * hidden + d);
                }
                write_scalar(node.output_buffer, (t * DSV4_HC + k) * hidden + d, acc);
            }
        }
    }
}

void compute_dsv4_hc_head_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& streams = get_input(node, 0, nodes, node_index_map);
    const auto& fn = get_input(node, 1, nodes, node_index_map);
    const auto& base = get_input(node, 2, nodes, node_index_map);
    const auto& scale = get_input(node, 3, nodes, node_index_map);
    if (streams.shape.size() != 3 || streams.shape[1] != DSV4_HC) {
        throw std::runtime_error("DSV4_HC_HEAD expects streams [T,4,D]");
    }
    const size_t tokens = streams.shape[0];
    const size_t hidden = streams.shape[2];
    const size_t flat_dim = DSV4_HC * hidden;
    if (fn.shape != std::vector<size_t>{DSV4_HC, flat_dim} ||
        base.shape != std::vector<size_t>{DSV4_HC} ||
        scale.shape != std::vector<size_t>{1}) {
        throw std::runtime_error("DSV4_HC_HEAD weight shape mismatch");
    }

    const float eps = node.params.epsilon;
    const float hc_scale = read_scalar(scale, 0);
    std::vector<float> normed(flat_dim);
    float pre[DSV4_HC] = {};

    for (size_t t = 0; t < tokens; ++t) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat_dim; ++i) {
            float value = read_scalar(streams, t * flat_dim + i);
            mean_square += static_cast<double>(value) * value;
        }
        float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / flat_dim) + eps);
        for (size_t i = 0; i < flat_dim; ++i) {
            normed[i] = read_scalar(streams, t * flat_dim + i) * rsqrt;
        }
        for (size_t h = 0; h < DSV4_HC; ++h) {
            float acc = 0.0f;
            for (size_t i = 0; i < flat_dim; ++i) {
                acc += read_scalar(fn, h * flat_dim + i) * normed[i];
            }
            pre[h] = sigmoid(acc * hc_scale + read_scalar(base, h)) + eps;
        }
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0.0f;
            for (size_t h = 0; h < DSV4_HC; ++h) {
                acc += pre[h] * read_scalar(streams, (t * DSV4_HC + h) * hidden + d);
            }
            write_scalar(node.output_buffer, t * hidden + d, acc);
        }
    }
}

void compute_dsv4_rope_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    if (input.shape.size() != 4) {
        throw std::runtime_error("DSV4_ROPE expects [batch,seq,heads,head_dim]");
    }
    if (input.precision != Precision::FP16 && input.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_ROPE expects FP16 or FP32 input");
    }

    const size_t batch = input.shape[0];
    const size_t seq = input.shape[1];
    const size_t heads = input.shape[2];
    const size_t head_dim = input.shape[3];
    const size_t rope_dim = node.params.head_dim;
    if (rope_dim == 0 || (rope_dim % 2) != 0 || rope_dim > head_dim) {
        throw std::runtime_error("DSV4_ROPE got invalid rope_dim");
    }
    const size_t nope_dim = head_dim - rope_dim;
    const float theta = node.params.theta;
    const float factor = node.params.scalar;
    const bool use_yarn = node.params.use_yarn && factor != 1.0f;
    const float beta_fast = node.params.yarn_beta_fast;
    const float beta_slow = node.params.yarn_beta_slow;
    const size_t original_max = node.params.yarn_original_max_position_embeddings;

    auto correction_dim = [](float rotations, size_t dim, float base, size_t max_seq_len) {
        return static_cast<float>(dim) *
               std::log(static_cast<float>(max_seq_len) / (rotations * 2.0f * PI_F)) /
               (2.0f * std::log(base));
    };
    auto ramp = [](float low, float high, size_t idx) {
        if (low == high) high += 0.001f;
        float value = (static_cast<float>(idx) - low) / (high - low);
        return std::clamp(value, 0.0f, 1.0f);
    };

    std::vector<float> freqs(rope_dim / 2);
    float low = 0.0f;
    float high = 0.0f;
    if (use_yarn) {
        low = std::floor(correction_dim(beta_fast, rope_dim, theta, original_max));
        high = std::ceil(correction_dim(beta_slow, rope_dim, theta, original_max));
        low = std::max(0.0f, low);
        high = std::min(static_cast<float>(rope_dim - 1), high);
    }
    for (size_t pair = 0; pair < rope_dim / 2; ++pair) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(2 * pair) / static_cast<float>(rope_dim));
        if (use_yarn) {
            float smooth = 1.0f - ramp(low, high, pair);
            freq = freq / factor * (1.0f - smooth) + freq * smooth;
        }
        freqs[pair] = freq;
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            const size_t stride = node.params.window_size == 0 ? 1 : node.params.window_size;
            const float pos = static_cast<float>(node.params.position_offset + t * stride);
            for (size_t h = 0; h < heads; ++h) {
                const size_t base_idx = ((b * seq + t) * heads + h) * head_dim;
                for (size_t d = 0; d < nope_dim; ++d) {
                    write_scalar(node.output_buffer, base_idx + d, read_scalar(input, base_idx + d));
                }
                for (size_t pair = 0; pair < rope_dim / 2; ++pair) {
                    const size_t d0 = nope_dim + 2 * pair;
                    const size_t d1 = d0 + 1;
                    const float x0 = read_scalar(input, base_idx + d0);
                    const float x1 = read_scalar(input, base_idx + d1);
                    const float angle = pos * freqs[pair];
                    const float c = std::cos(angle);
                    float s = std::sin(angle);
                    if (node.params.inverse) s = -s;
                    write_scalar(node.output_buffer, base_idx + d0, x0 * c - x1 * s);
                    write_scalar(node.output_buffer, base_idx + d1, x0 * s + x1 * c);
                }
            }
        }
    }
}

void compute_dsv4_sparse_attention_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& query = get_input(node, 0, nodes, node_index_map);
    const auto& kv = get_input(node, 1, nodes, node_index_map);
    const auto& attn_sink = get_input(node, 2, nodes, node_index_map);
    const auto& topk_indices = get_input(node, 3, nodes, node_index_map);

    if (query.shape.size() != 4) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION expects query [B,S,H,D]");
    }
    const size_t batch = query.shape[0];
    const size_t seq = query.shape[1];
    const size_t heads = query.shape[2];
    const size_t head_dim = query.shape[3];
    if (kv.shape.size() != 3 || kv.shape[0] != batch || kv.shape[2] != head_dim) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION expects kv [B,N,D]");
    }
    if (attn_sink.shape != std::vector<size_t>{heads}) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION expects attn_sink [H]");
    }
    if (topk_indices.shape.size() != 3 || topk_indices.shape[0] != batch || topk_indices.shape[1] != seq) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION expects topk_indices [B,S,K]");
    }
    if (topk_indices.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION expects FP32 topk_indices");
    }
    if (node.output_buffer.shape != query.shape) {
        throw std::runtime_error("DSV4_SPARSE_ATTENTION output shape mismatch");
    }

    const size_t kv_len = kv.shape[1];
    const size_t top_k = topk_indices.shape[2];
    const float scale = node.params.scalar;
    std::vector<float> logits(top_k);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            for (size_t h = 0; h < heads; ++h) {
                float max_logit = read_scalar(attn_sink, h);
                for (size_t k = 0; k < top_k; ++k) {
                    const float raw_idx = read_scalar(topk_indices, (b * seq + t) * top_k + k);
                    if (raw_idx < 0.0f) {
                        logits[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    if (!std::isfinite(raw_idx)) {
                        throw std::runtime_error("DSV4_SPARSE_ATTENTION got non-finite topk index");
                    }
                    const size_t idx = static_cast<size_t>(raw_idx + 0.5f);
                    if (idx >= kv_len) {
                        throw std::runtime_error("DSV4_SPARSE_ATTENTION topk index out of kv range");
                    }
                    float logit = 0.0f;
                    const size_t q_base = ((b * seq + t) * heads + h) * head_dim;
                    const size_t kv_base = (b * kv_len + idx) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        logit += read_scalar(query, q_base + d) * read_scalar(kv, kv_base + d);
                    }
                    logit *= scale;
                    logits[k] = logit;
                    max_logit = std::max(max_logit, logit);
                }

                float denom = std::exp(read_scalar(attn_sink, h) - max_logit);
                for (size_t k = 0; k < top_k; ++k) {
                    if (std::isfinite(logits[k])) {
                        denom += std::exp(logits[k] - max_logit);
                    }
                }

                const size_t out_base = ((b * seq + t) * heads + h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < top_k; ++k) {
                        if (!std::isfinite(logits[k])) continue;
                        const size_t idx = static_cast<size_t>(read_scalar(topk_indices, (b * seq + t) * top_k + k) + 0.5f);
                        const size_t kv_base = (b * kv_len + idx) * head_dim;
                        const float weight = std::exp(logits[k] - max_logit) / denom;
                        acc += weight * read_scalar(kv, kv_base + d);
                    }
                    write_scalar(node.output_buffer, out_base + d, acc);
                }
            }
        }
    }
}

void compute_dsv4_compress_hca_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& kv = get_input(node, 0, nodes, node_index_map);
    const auto& score = get_input(node, 1, nodes, node_index_map);
    const auto& ape = get_input(node, 2, nodes, node_index_map);
    const auto& norm_weight = get_input(node, 3, nodes, node_index_map);

    if (kv.shape.size() != 3) {
        throw std::runtime_error("DSV4_COMPRESS_HCA expects kv [B,S,D]");
    }
    const size_t batch = kv.shape[0];
    const size_t seq = kv.shape[1];
    const size_t head_dim = kv.shape[2];
    const size_t ratio = node.params.chunk_size;
    if (ratio == 0) {
        throw std::runtime_error("DSV4_COMPRESS_HCA got invalid ratio");
    }
    const size_t chunks = seq / ratio;
    if (score.shape != kv.shape ||
        ape.shape != std::vector<size_t>{ratio, head_dim} ||
        norm_weight.shape != std::vector<size_t>{head_dim} ||
        node.output_buffer.shape != std::vector<size_t>{batch, chunks, head_dim}) {
        throw std::runtime_error("DSV4_COMPRESS_HCA shape mismatch");
    }

    const float eps = node.params.epsilon;
    std::vector<float> compressed(head_dim);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t d = 0; d < head_dim; ++d) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t r = 0; r < ratio; ++r) {
                    const size_t src = (b * seq + c * ratio + r) * head_dim + d;
                    const float s = read_scalar(score, src) + read_scalar(ape, r * head_dim + d);
                    max_score = std::max(max_score, s);
                }
                float denom = 0.0f;
                float acc = 0.0f;
                for (size_t r = 0; r < ratio; ++r) {
                    const size_t src = (b * seq + c * ratio + r) * head_dim + d;
                    const float weight = std::exp(read_scalar(score, src) + read_scalar(ape, r * head_dim + d) - max_score);
                    denom += weight;
                    acc += weight * read_scalar(kv, src);
                }
                compressed[d] = acc / denom;
            }

            double mean_square = 0.0;
            for (float value : compressed) {
                mean_square += static_cast<double>(value) * value;
            }
            const float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / head_dim) + eps);
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t dst = (b * chunks + c) * head_dim + d;
                write_scalar(node.output_buffer, dst, compressed[d] * rsqrt * read_scalar(norm_weight, d));
            }
        }
    }
}

void compute_dsv4_compress_csa_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& kv = get_input(node, 0, nodes, node_index_map);
    const auto& score = get_input(node, 1, nodes, node_index_map);
    const auto& ape = get_input(node, 2, nodes, node_index_map);
    const auto& norm_weight = get_input(node, 3, nodes, node_index_map);

    if (kv.shape.size() != 3 || kv.shape[2] % 2 != 0) {
        throw std::runtime_error("DSV4_COMPRESS_CSA expects kv [B,S,2D]");
    }
    const size_t batch = kv.shape[0];
    const size_t seq = kv.shape[1];
    const size_t head_dim = kv.shape[2] / 2;
    const size_t ratio = node.params.chunk_size;
    if (ratio == 0) {
        throw std::runtime_error("DSV4_COMPRESS_CSA got invalid ratio");
    }
    const size_t chunks = seq / ratio;
    if (score.shape != kv.shape ||
        ape.shape != std::vector<size_t>{ratio, 2 * head_dim} ||
        norm_weight.shape != std::vector<size_t>{head_dim} ||
        node.output_buffer.shape != std::vector<size_t>{batch, chunks, head_dim}) {
        throw std::runtime_error("DSV4_COMPRESS_CSA shape mismatch");
    }

    const float eps = node.params.epsilon;
    std::vector<float> compressed(head_dim);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t d = 0; d < head_dim; ++d) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t slot = 0; slot < 2 * ratio; ++slot) {
                    if (slot < ratio && c == 0) continue;
                    const size_t src_chunk = slot < ratio ? c - 1 : c;
                    const size_t r = slot % ratio;
                    const size_t dim = slot < ratio ? d : head_dim + d;
                    const size_t src = (b * seq + src_chunk * ratio + r) * (2 * head_dim) + dim;
                    const float s = read_scalar(score, src) + read_scalar(ape, r * (2 * head_dim) + dim);
                    max_score = std::max(max_score, s);
                }
                float denom = 0.0f;
                float acc = 0.0f;
                for (size_t slot = 0; slot < 2 * ratio; ++slot) {
                    if (slot < ratio && c == 0) continue;
                    const size_t src_chunk = slot < ratio ? c - 1 : c;
                    const size_t r = slot % ratio;
                    const size_t dim = slot < ratio ? d : head_dim + d;
                    const size_t src = (b * seq + src_chunk * ratio + r) * (2 * head_dim) + dim;
                    const float weight = std::exp(read_scalar(score, src) + read_scalar(ape, r * (2 * head_dim) + dim) - max_score);
                    denom += weight;
                    acc += weight * read_scalar(kv, src);
                }
                compressed[d] = acc / denom;
            }

            double mean_square = 0.0;
            for (float value : compressed) {
                mean_square += static_cast<double>(value) * value;
            }
            const float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / head_dim) + eps);
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t dst = (b * chunks + c) * head_dim + d;
                write_scalar(node.output_buffer, dst, compressed[d] * rsqrt * read_scalar(norm_weight, d));
            }
        }
    }
}

void compute_dsv4_indexer_topk_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& query = get_input(node, 0, nodes, node_index_map);
    const auto& compressed_kv = get_input(node, 1, nodes, node_index_map);
    const auto& weights = get_input(node, 2, nodes, node_index_map);
    const auto& position_ids = get_input(node, 3, nodes, node_index_map);

    if (query.shape.size() != 4) {
        throw std::runtime_error("DSV4_INDEXER_TOPK expects query [B,S,H,D]");
    }
    const size_t batch = query.shape[0];
    const size_t seq = query.shape[1];
    const size_t heads = query.shape[2];
    const size_t head_dim = query.shape[3];
    const size_t top_k = node.params.top_k;
    const size_t ratio = node.params.chunk_size;
    const size_t offset = node.params.index_value;
    if (ratio == 0 || top_k == 0) {
        throw std::runtime_error("DSV4_INDEXER_TOPK got invalid top_k or ratio");
    }
    if (compressed_kv.shape.size() != 3 || compressed_kv.shape[0] != batch || compressed_kv.shape[2] != head_dim) {
        throw std::runtime_error("DSV4_INDEXER_TOPK expects compressed_kv [B,N,D]");
    }
    const size_t compressed_len = compressed_kv.shape[1];
    if (top_k > compressed_len) {
        throw std::runtime_error("DSV4_INDEXER_TOPK top_k exceeds compressed length");
    }
    if (weights.shape != std::vector<size_t>{batch, seq, heads} ||
        position_ids.shape != std::vector<size_t>{batch, seq} ||
        node.output_buffer.shape != std::vector<size_t>{batch, seq, top_k} ||
        node.output_buffer.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_INDEXER_TOPK shape mismatch");
    }

    std::vector<std::pair<float, size_t>> scores(compressed_len);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            const float raw_pos = read_scalar(position_ids, b * seq + t);
            if (!std::isfinite(raw_pos) || raw_pos < 0.0f) {
                throw std::runtime_error("DSV4_INDEXER_TOPK got invalid position id");
            }
            const size_t position = static_cast<size_t>(raw_pos + 0.5f);
            const size_t causal_threshold = (position + 1) / ratio;
            for (size_t n = 0; n < compressed_len; ++n) {
                if (n >= causal_threshold) {
                    scores[n] = {-std::numeric_limits<float>::infinity(), n};
                    continue;
                }
                float score = 0.0f;
                for (size_t h = 0; h < heads; ++h) {
                    float dot = 0.0f;
                    const size_t q_base = ((b * seq + t) * heads + h) * head_dim;
                    const size_t kv_base = (b * compressed_len + n) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += read_scalar(query, q_base + d) * read_scalar(compressed_kv, kv_base + d);
                    }
                    score += std::max(dot, 0.0f) *
                             read_scalar(weights, (b * seq + t) * heads + h) *
                             node.params.scalar;
                }
                scores[n] = {score, n};
            }
            std::partial_sort(scores.begin(),
                              scores.begin() + static_cast<std::ptrdiff_t>(top_k),
                              scores.end(),
                              [](const auto& a, const auto& b) {
                                  if (a.first != b.first) return a.first > b.first;
                                  return a.second < b.second;
                              });
            for (size_t k = 0; k < top_k; ++k) {
                const size_t idx = scores[k].second;
                const float value = idx >= causal_threshold ? -1.0f : static_cast<float>(idx + offset);
                write_scalar(node.output_buffer, (b * seq + t) * top_k + k, value);
            }
        }
    }
}

void compute_dsv4_router_topk_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& hidden = get_input(node, 0, nodes, node_index_map);
    const auto& weight = get_input(node, 1, nodes, node_index_map);
    const auto& bias = get_input(node, 2, nodes, node_index_map);

    const size_t num_experts = node.params.num_experts;
    const size_t top_k = node.params.num_experts_per_tok;
    validate_hidden_and_router_weight(hidden, weight, num_experts, "DSV4_ROUTER_TOPK");
    if (top_k == 0 || top_k > num_experts) {
        throw std::runtime_error("DSV4_ROUTER_TOPK got invalid top_k");
    }
    if (bias.shape != std::vector<size_t>{num_experts}) {
        throw std::runtime_error("DSV4_ROUTER_TOPK bias shape mismatch");
    }
    if (node.output_buffer.shape != std::vector<size_t>{2, hidden.shape[0], top_k} ||
        node.output_buffer.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_ROUTER_TOPK output must be FP32 [2,T,K]");
    }

    const size_t tokens = hidden.shape[0];
    const float eps = node.params.epsilon;
    const float route_scale = node.params.scalar;
    std::vector<float> original_scores(num_experts);
    std::vector<std::pair<float, size_t>> selection_scores(num_experts);

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t e = 0; e < num_experts; ++e) {
            const float score = sqrt_softplus(router_logit(hidden, weight, t, e));
            original_scores[e] = score;
            selection_scores[e] = {score + read_scalar(bias, e), e};
        }
        std::partial_sort(selection_scores.begin(),
                          selection_scores.begin() + static_cast<std::ptrdiff_t>(top_k),
                          selection_scores.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        float denom = eps;
        for (size_t k = 0; k < top_k; ++k) {
            denom += original_scores[selection_scores[k].second];
        }
        for (size_t k = 0; k < top_k; ++k) {
            const size_t expert = selection_scores[k].second;
            const float route_weight = original_scores[expert] / denom * route_scale;
            write_route(node.output_buffer, tokens, t, k, expert, route_weight);
        }
    }
}

void compute_dsv4_hash_router_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& hidden = get_input(node, 0, nodes, node_index_map);
    const auto& input_ids = get_input(node, 1, nodes, node_index_map);
    const auto& weight = get_input(node, 2, nodes, node_index_map);
    const auto& tid2eid = get_input(node, 3, nodes, node_index_map);

    const size_t num_experts = node.params.num_experts;
    const size_t top_k = node.params.num_experts_per_tok;
    validate_hidden_and_router_weight(hidden, weight, num_experts, "DSV4_HASH_ROUTER");
    if (top_k == 0) {
        throw std::runtime_error("DSV4_HASH_ROUTER got invalid top_k");
    }
    if (input_ids.shape != std::vector<size_t>{hidden.shape[0]}) {
        throw std::runtime_error("DSV4_HASH_ROUTER expects input_ids [T]");
    }
    if (tid2eid.shape.size() != 2 || tid2eid.shape[1] != top_k) {
        throw std::runtime_error("DSV4_HASH_ROUTER expects tid2eid [vocab,K]");
    }
    if (node.output_buffer.shape != std::vector<size_t>{2, hidden.shape[0], top_k} ||
        node.output_buffer.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_HASH_ROUTER output must be FP32 [2,T,K]");
    }

    const size_t tokens = hidden.shape[0];
    const size_t vocab = tid2eid.shape[0];
    const float eps = node.params.epsilon;
    const float route_scale = node.params.scalar;
    std::vector<float> original_scores(num_experts);
    std::vector<size_t> selected(top_k);

    for (size_t t = 0; t < tokens; ++t) {
        const float raw_id = read_scalar(input_ids, t);
        if (!std::isfinite(raw_id) || raw_id < 0.0f) {
            throw std::runtime_error("DSV4_HASH_ROUTER got invalid token id");
        }
        const size_t token_id = static_cast<size_t>(raw_id + 0.5f);
        if (token_id >= vocab) {
            throw std::runtime_error("DSV4_HASH_ROUTER token id out of tid2eid range");
        }
        for (size_t e = 0; e < num_experts; ++e) {
            original_scores[e] = sqrt_softplus(router_logit(hidden, weight, t, e));
        }
        float denom = eps;
        for (size_t k = 0; k < top_k; ++k) {
            const float raw_expert = read_scalar(tid2eid, token_id * top_k + k);
            if (!std::isfinite(raw_expert) || raw_expert < 0.0f) {
                throw std::runtime_error("DSV4_HASH_ROUTER got invalid expert id");
            }
            const size_t expert = static_cast<size_t>(raw_expert + 0.5f);
            if (expert >= num_experts) {
                throw std::runtime_error("DSV4_HASH_ROUTER expert id out of range");
            }
            selected[k] = expert;
            denom += original_scores[expert];
        }
        for (size_t k = 0; k < top_k; ++k) {
            const size_t expert = selected[k];
            const float route_weight = original_scores[expert] / denom * route_scale;
            write_route(node.output_buffer, tokens, t, k, expert, route_weight);
        }
    }
}

void compute_dsv4_moe_layer_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& hidden = get_input(node, 0, nodes, node_index_map);
    const auto& route = get_input(node, 1, nodes, node_index_map);

    const size_t num_experts = node.params.num_experts;
    const size_t top_k = node.params.num_experts_per_tok;
    if (hidden.shape.size() != 2) {
        throw std::runtime_error("DSV4_MOE_LAYER expects hidden [T,D]");
    }
    if (hidden.precision != Precision::FP16 && hidden.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_MOE_LAYER expects FP16 or FP32 hidden");
    }
    if (route.shape != std::vector<size_t>{2, hidden.shape[0], top_k} || route.precision != Precision::FP32) {
        throw std::runtime_error("DSV4_MOE_LAYER expects FP32 route [2,T,K]");
    }
    if (node.input_ids.size() != 2 + 3 * num_experts) {
        throw std::runtime_error("DSV4_MOE_LAYER input count mismatch");
    }

    const size_t tokens = hidden.shape[0];
    const size_t hidden_dim = hidden.shape[1];
    const auto& gate0 = get_input(node, 2, nodes, node_index_map);
    if (gate0.shape.size() != 2 || gate0.shape[1] != hidden_dim) {
        throw std::runtime_error("DSV4_MOE_LAYER gate weight shape mismatch");
    }
    const size_t inter_dim = gate0.shape[0];
    for (size_t e = 0; e < num_experts; ++e) {
        const auto& gate = get_input(node, 2 + e, nodes, node_index_map);
        const auto& up = get_input(node, 2 + num_experts + e, nodes, node_index_map);
        const auto& down = get_input(node, 2 + 2 * num_experts + e, nodes, node_index_map);
        if (gate.shape != std::vector<size_t>{inter_dim, hidden_dim} ||
            up.shape != std::vector<size_t>{inter_dim, hidden_dim} ||
            down.shape != std::vector<size_t>{hidden_dim, inter_dim}) {
            throw std::runtime_error("DSV4_MOE_LAYER expert weight shape mismatch");
        }
    }

    for (size_t i = 0; i < node.output_buffer.total_size; ++i) {
        write_scalar(node.output_buffer, i, 0.0f);
    }

    const float limit = node.params.scalar;
    bool all_cq_experts = true;
    for (size_t e = 0; e < num_experts; ++e) {
        const auto& gate = get_input(node, 2 + e, nodes, node_index_map);
        const auto& up = get_input(node, 2 + num_experts + e, nodes, node_index_map);
        const auto& down = get_input(node, 2 + 2 * num_experts + e, nodes, node_index_map);
        all_cq_experts = all_cq_experts &&
                         PrecisionTraits::is_cq(gate.precision) &&
                         PrecisionTraits::is_cq(up.precision) &&
                         PrecisionTraits::is_cq(down.precision);
    }
    struct Assignment { size_t token; float weight; };
    if (all_cq_experts) {
        std::vector<std::vector<Assignment>> by_expert(num_experts);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t k = 0; k < top_k; ++k) {
                const float raw_expert = read_scalar(route, t * top_k + k);
                const float route_weight = read_scalar(route, tokens * top_k + t * top_k + k);
                if (route_weight == 0.0f) continue;
                if (!std::isfinite(raw_expert) || raw_expert < 0.0f) {
                    throw std::runtime_error("DSV4_MOE_LAYER got invalid route expert");
                }
                const size_t expert = static_cast<size_t>(raw_expert + 0.5f);
                if (expert >= num_experts) {
                    throw std::runtime_error("DSV4_MOE_LAYER route expert out of range");
                }
                by_expert[expert].push_back({t, route_weight});
            }
        }

        std::vector<__fp16> compact_hidden;
        std::vector<__fp16> gate_batch;
        std::vector<__fp16> up_batch;
        std::vector<__fp16> product_batch;
        std::vector<__fp16> down_batch;
        for (size_t expert = 0; expert < num_experts; ++expert) {
            const auto& assignments = by_expert[expert];
            if (assignments.empty()) continue;
            const auto& gate_w = get_input(node, 2 + expert, nodes, node_index_map);
            const auto& up_w = get_input(node, 2 + num_experts + expert, nodes, node_index_map);
            const auto& down_w = get_input(node, 2 + 2 * num_experts + expert, nodes, node_index_map);
            const size_t batch = assignments.size();
            compact_hidden.resize(batch * hidden_dim);
            for (size_t m = 0; m < batch; ++m) {
                const size_t t = assignments[m].token;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    compact_hidden[m * hidden_dim + d] = static_cast<__fp16>(read_scalar(hidden, t * hidden_dim + d));
                }
            }
            matmul_batch_weight(compact_hidden, batch, hidden_dim, gate_w, gate_batch);
            matmul_batch_weight(compact_hidden, batch, hidden_dim, up_w, up_batch);
            product_batch.resize(batch * inter_dim);
            for (size_t m = 0; m < batch; ++m) {
                for (size_t i = 0; i < inter_dim; ++i) {
                    float gate = static_cast<float>(gate_batch[m * inter_dim + i]);
                    float up = static_cast<float>(up_batch[m * inter_dim + i]);
                    if (limit > 0.0f) {
                        gate = std::min(gate, limit);
                        up = std::clamp(up, -limit, limit);
                    }
                    product_batch[m * inter_dim + i] = static_cast<__fp16>(silu(gate) * up);
                }
            }
            matmul_batch_weight(product_batch, batch, inter_dim, down_w, down_batch);
            for (size_t m = 0; m < batch; ++m) {
                const size_t t = assignments[m].token;
                const float route_weight = assignments[m].weight;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    const size_t out_idx = t * hidden_dim + d;
                    const float current = read_scalar(node.output_buffer, out_idx);
                    write_scalar(node.output_buffer, out_idx, current + route_weight * static_cast<float>(down_batch[m * hidden_dim + d]));
                }
            }
        }
        return;
    }

    std::vector<__fp16> x;
    std::vector<__fp16> product_fp16(inter_dim);
    std::vector<float> gate_values;
    std::vector<float> up_values;
    std::vector<float> down_values;
    for (size_t t = 0; t < tokens; ++t) {
        read_hidden_token_fp16(hidden, t, x);
        for (size_t k = 0; k < top_k; ++k) {
            const float raw_expert = read_scalar(route, t * top_k + k);
            const float route_weight = read_scalar(route, tokens * top_k + t * top_k + k);
            if (route_weight == 0.0f) continue;
            if (!std::isfinite(raw_expert) || raw_expert < 0.0f) {
                throw std::runtime_error("DSV4_MOE_LAYER got invalid route expert");
            }
            const size_t expert = static_cast<size_t>(raw_expert + 0.5f);
            if (expert >= num_experts) {
                throw std::runtime_error("DSV4_MOE_LAYER route expert out of range");
            }

            const auto& gate_w = get_input(node, 2 + expert, nodes, node_index_map);
            const auto& up_w = get_input(node, 2 + num_experts + expert, nodes, node_index_map);
            const auto& down_w = get_input(node, 2 + 2 * num_experts + expert, nodes, node_index_map);

            const bool use_cq_fast =
                PrecisionTraits::is_cq(gate_w.precision) ||
                PrecisionTraits::is_cq(up_w.precision) ||
                PrecisionTraits::is_cq(down_w.precision);
            if (use_cq_fast) {
                matmul_vec_weight(x, gate_w, gate_values);
                matmul_vec_weight(x, up_w, up_values);
                for (size_t i = 0; i < inter_dim; ++i) {
                    float gate = gate_values[i];
                    float up = up_values[i];
                    if (limit > 0.0f) {
                        gate = std::min(gate, limit);
                        up = std::clamp(up, -limit, limit);
                    }
                    product_fp16[i] = static_cast<__fp16>(silu(gate) * up);
                }
                matmul_vec_weight(product_fp16, down_w, down_values);
                for (size_t d = 0; d < hidden_dim; ++d) {
                    const size_t out_idx = t * hidden_dim + d;
                    const float current = read_scalar(node.output_buffer, out_idx);
                    write_scalar(node.output_buffer, out_idx, current + route_weight * down_values[d]);
                }
            } else {
                std::vector<float> product(inter_dim);
                for (size_t i = 0; i < inter_dim; ++i) {
                    float gate = 0.0f;
                    float up = 0.0f;
                    for (size_t d = 0; d < hidden_dim; ++d) {
                        const float xv = read_scalar(hidden, t * hidden_dim + d);
                        gate += xv * read_scalar(gate_w, i * hidden_dim + d);
                        up += xv * read_scalar(up_w, i * hidden_dim + d);
                    }
                    if (limit > 0.0f) {
                        gate = std::min(gate, limit);
                        up = std::clamp(up, -limit, limit);
                    }
                    product[i] = silu(gate) * up;
                }
                for (size_t d = 0; d < hidden_dim; ++d) {
                    float expert_out = 0.0f;
                    for (size_t i = 0; i < inter_dim; ++i) {
                        expert_out += product[i] * read_scalar(down_w, d * inter_dim + i);
                    }
                    const size_t out_idx = t * hidden_dim + d;
                    const float current = read_scalar(node.output_buffer, out_idx);
                    write_scalar(node.output_buffer, out_idx, current + route_weight * expert_out);
                }
            }
        }
    }
}

void compute_dsv4_shared_expert_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& hidden = get_input(node, 0, nodes, node_index_map);
    const auto& gate_w = get_input(node, 1, nodes, node_index_map);
    const auto& up_w = get_input(node, 2, nodes, node_index_map);
    const auto& down_w = get_input(node, 3, nodes, node_index_map);

    if (hidden.shape.size() != 2) {
        throw std::runtime_error("DSV4_SHARED_EXPERT expects hidden [T,D]");
    }
    const size_t tokens = hidden.shape[0];
    const size_t hidden_dim = hidden.shape[1];
    if (gate_w.shape.size() != 2 || gate_w.shape[1] != hidden_dim) {
        throw std::runtime_error("DSV4_SHARED_EXPERT gate weight shape mismatch");
    }
    const size_t inter_dim = gate_w.shape[0];
    if (up_w.shape != std::vector<size_t>{inter_dim, hidden_dim} ||
        down_w.shape != std::vector<size_t>{hidden_dim, inter_dim} ||
        node.output_buffer.shape != hidden.shape) {
        throw std::runtime_error("DSV4_SHARED_EXPERT shape mismatch");
    }

    const float limit = node.params.scalar;
    std::vector<__fp16> x;
    std::vector<__fp16> product_fp16(inter_dim);
    std::vector<float> gate_values;
    std::vector<float> up_values;
    std::vector<float> down_values;
    const bool use_cq_fast =
        PrecisionTraits::is_cq(gate_w.precision) ||
        PrecisionTraits::is_cq(up_w.precision) ||
        PrecisionTraits::is_cq(down_w.precision);
    if (use_cq_fast && tokens > 1) {
        std::vector<__fp16> compact_hidden(tokens * hidden_dim);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t d = 0; d < hidden_dim; ++d) {
                compact_hidden[t * hidden_dim + d] = static_cast<__fp16>(read_scalar(hidden, t * hidden_dim + d));
            }
        }
        std::vector<__fp16> gate_batch;
        std::vector<__fp16> up_batch;
        std::vector<__fp16> product_batch(tokens * inter_dim);
        std::vector<__fp16> down_batch;
        matmul_batch_weight(compact_hidden, tokens, hidden_dim, gate_w, gate_batch);
        matmul_batch_weight(compact_hidden, tokens, hidden_dim, up_w, up_batch);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < inter_dim; ++i) {
                float gate = static_cast<float>(gate_batch[t * inter_dim + i]);
                float up = static_cast<float>(up_batch[t * inter_dim + i]);
                if (limit > 0.0f) {
                    gate = std::min(gate, limit);
                    up = std::clamp(up, -limit, limit);
                }
                product_batch[t * inter_dim + i] = static_cast<__fp16>(silu(gate) * up);
            }
        }
        matmul_batch_weight(product_batch, tokens, inter_dim, down_w, down_batch);
        for (size_t i = 0; i < node.output_buffer.total_size; ++i) {
            write_scalar(node.output_buffer, i, static_cast<float>(down_batch[i]));
        }
        return;
    }
    for (size_t t = 0; t < tokens; ++t) {
        if (use_cq_fast) {
            read_hidden_token_fp16(hidden, t, x);
            matmul_vec_weight(x, gate_w, gate_values);
            matmul_vec_weight(x, up_w, up_values);
            for (size_t i = 0; i < inter_dim; ++i) {
                float gate = gate_values[i];
                float up = up_values[i];
                if (limit > 0.0f) {
                    gate = std::min(gate, limit);
                    up = std::clamp(up, -limit, limit);
                }
                product_fp16[i] = static_cast<__fp16>(silu(gate) * up);
            }
            matmul_vec_weight(product_fp16, down_w, down_values);
            for (size_t d = 0; d < hidden_dim; ++d) write_scalar(node.output_buffer, t * hidden_dim + d, down_values[d]);
        } else {
            std::vector<float> product(inter_dim);
            for (size_t i = 0; i < inter_dim; ++i) {
                float gate = 0.0f;
                float up = 0.0f;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    const float xv = read_scalar(hidden, t * hidden_dim + d);
                    gate += xv * read_scalar(gate_w, i * hidden_dim + d);
                    up += xv * read_scalar(up_w, i * hidden_dim + d);
                }
                if (limit > 0.0f) {
                    gate = std::min(gate, limit);
                    up = std::clamp(up, -limit, limit);
                }
                product[i] = silu(gate) * up;
            }
            for (size_t d = 0; d < hidden_dim; ++d) {
                float acc = 0.0f;
                for (size_t i = 0; i < inter_dim; ++i) {
                    acc += product[i] * read_scalar(down_w, d * inter_dim + i);
                }
                write_scalar(node.output_buffer, t * hidden_dim + d, acc);
            }
        }
    }
}

void compute_dsv4_grouped_linear_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& weight = get_input(node, 1, nodes, node_index_map);
    const size_t groups = node.params.num_groups;
    if (input.shape.size() != 2 || weight.shape.size() != 2 || groups == 0) {
        throw std::runtime_error("DSV4_GROUPED_LINEAR expects input [T,D] and weight [N,D/groups]");
    }
    const size_t tokens = input.shape[0];
    const size_t input_dim = input.shape[1];
    const size_t out_dim = weight.shape[0];
    if (input_dim % groups != 0 || out_dim % groups != 0 ||
        weight.shape[1] != input_dim / groups ||
        node.output_buffer.shape != std::vector<size_t>{tokens, out_dim}) {
        throw std::runtime_error("DSV4_GROUPED_LINEAR shape mismatch");
    }
    const size_t in_per_group = input_dim / groups;
    const size_t out_per_group = out_dim / groups;
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t g = 0; g < groups; ++g) {
            for (size_t o = 0; o < out_per_group; ++o) {
                const size_t row = g * out_per_group + o;
                float acc = 0.0f;
                for (size_t c = 0; c < in_per_group; ++c) {
                    acc += read_scalar(input, t * input_dim + g * in_per_group + c) *
                           read_scalar(weight, row * in_per_group + c);
                }
                write_scalar(node.output_buffer, t * out_dim + row, acc);
            }
        }
    }
}

void compute_dsv4_rms_norm_node(GraphNode& node, const nodes_vector& nodes, const node_index_map_t& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    if (input.shape.empty()) {
        throw std::runtime_error("DSV4_RMS_NORM expects non-scalar input");
    }
    if (node.output_buffer.shape != input.shape) {
        throw std::runtime_error("DSV4_RMS_NORM output shape mismatch");
    }
    const size_t last_dim = input.shape.back();
    const size_t rows = input.total_size / last_dim;
    const float eps = node.params.epsilon;
    for (size_t row = 0; row < rows; ++row) {
        double mean_square = 0.0;
        const size_t base = row * last_dim;
        for (size_t d = 0; d < last_dim; ++d) {
            const float value = read_scalar(input, base + d);
            mean_square += static_cast<double>(value) * value;
        }
        const float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / last_dim) + eps);
        for (size_t d = 0; d < last_dim; ++d) {
            write_scalar(node.output_buffer, base + d, read_scalar(input, base + d) * rsqrt);
        }
    }
}
