#include "graph.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

size_t CactusGraph::input(const std::vector<size_t>& shape, Precision precision) {
    return add_node(OpType::INPUT, {}, shape, {.output_precision = precision});
}

size_t CactusGraph::add(size_t input1, size_t input2) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);                  
    BroadcastInfo broadcast_info = BroadcastInfo::compute(lhs_buffer.shape, rhs_buffer.shape);
    OpParams params{.broadcast_info = broadcast_info};

    return add_node(OpType::ADD, {input1, input2}, broadcast_info.output_shape, params);
}

size_t CactusGraph::add_clipped(size_t input1, size_t input2) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);

    BroadcastInfo broadcast_info = BroadcastInfo::compute(lhs_buffer.shape, rhs_buffer.shape);
    OpParams params{.broadcast_info = broadcast_info};

    return add_node(OpType::ADD_CLIPPED, {input1, input2}, broadcast_info.output_shape, params);
}

size_t CactusGraph::subtract(size_t input1, size_t input2) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);

    BroadcastInfo broadcast_info = BroadcastInfo::compute(lhs_buffer.shape, rhs_buffer.shape);
    OpParams params{.broadcast_info = broadcast_info};

    return add_node(OpType::SUBTRACT, {input1, input2}, broadcast_info.output_shape, params);
}

size_t CactusGraph::multiply(size_t input1, size_t input2) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);

    BroadcastInfo broadcast_info = BroadcastInfo::compute(lhs_buffer.shape, rhs_buffer.shape);
    OpParams params{.broadcast_info = broadcast_info};

    return add_node(OpType::MULTIPLY, {input1, input2}, broadcast_info.output_shape, params);
}

size_t CactusGraph::divide(size_t input1, size_t input2) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);

    BroadcastInfo broadcast_info = BroadcastInfo::compute(lhs_buffer.shape, rhs_buffer.shape);
    OpParams params{.broadcast_info = broadcast_info};

    return add_node(OpType::DIVIDE, {input1, input2}, broadcast_info.output_shape, params);
}

size_t CactusGraph::matmul(size_t input1, size_t input2, bool pretransposed_rhs, ComputeBackend backend) {
    const auto& lhs_buffer = get_output_buffer(input1);
    const auto& rhs_buffer = get_output_buffer(input2);
      
    if (lhs_buffer.shape.size() != 2 || rhs_buffer.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }

    size_t M = lhs_buffer.shape[0];
    size_t K = lhs_buffer.shape[1];
    size_t rhs_K = pretransposed_rhs ? rhs_buffer.shape[1] : rhs_buffer.shape[0];

    // For interleaved weights, use original_N (shape stores padded N)
    size_t N;
    if (rhs_buffer.is_interleaved && rhs_buffer.original_N > 0) {
        N = rhs_buffer.original_N;
    } else {
        N = pretransposed_rhs ? rhs_buffer.shape[0] : rhs_buffer.shape[1];
    }

    if (K != rhs_K) {
        std::cout << "Matrix dimensions incompatible for multiplication: " << K << " != " << rhs_K << std::endl;
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    std::vector<size_t> output_shape = {M, N};
    OpParams params{.pretransposed_rhs = pretransposed_rhs, .backend = backend};
    return add_node(OpType::MATMUL, {input1, input2}, output_shape, params);
}

size_t CactusGraph::transpose(size_t input, ComputeBackend backend) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape = input_buffer.shape;

    if (output_shape.size() >= 2) {
        std::swap(output_shape[output_shape.size()-2], output_shape[output_shape.size()-1]);
    }

    std::vector<size_t> permutation;
    for (size_t i = 0; i < input_buffer.shape.size(); ++i) {
        permutation.push_back(i);
    }
    if (permutation.size() >= 2) {
        std::swap(permutation[permutation.size()-2], permutation[permutation.size()-1]);
    }

    OpParams params{.permutation = permutation, .backend = backend};
    return add_node(OpType::TRANSPOSE, {input}, output_shape, params);
}

size_t CactusGraph::transposeN(size_t input, const std::vector<size_t>& permutation, ComputeBackend backend) {
    const auto& input_buffer = get_output_buffer(input);
    if (permutation.size() != input_buffer.shape.size()) {
        throw std::runtime_error("transposeN permutation size must match tensor rank");
    }
    std::vector<size_t> output_shape(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        size_t p = permutation[i];
        if (p >= input_buffer.shape.size()) {
            throw std::runtime_error("transposeN permutation index out of range");
        }
        output_shape[i] = input_buffer.shape[p];
    }
    OpParams params{.permutation = permutation, .backend = backend};
    return add_node(OpType::TRANSPOSE, {input}, output_shape, params);
}

size_t CactusGraph::reshape(size_t input, const std::vector<size_t>& new_shape) {
    OpParams params{.new_shape = new_shape};
    return add_node(OpType::RESHAPE, {input}, new_shape, params);
}

size_t CactusGraph::index(size_t input, size_t index_value, int dim) {
    const auto& input_buffer = get_output_buffer(input);
    const auto& shape = input_buffer.shape;

    if (shape.empty()) {
        throw std::invalid_argument("Cannot index a scalar tensor");
    }

    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim += static_cast<int>(shape.size());
    }

    if (actual_dim < 0 || static_cast<size_t>(actual_dim) >= shape.size()) {
        throw std::invalid_argument("Index dimension out of bounds");
    }

    if (index_value >= shape[actual_dim]) {
        throw std::invalid_argument("Index value " + std::to_string(index_value) +
                                    " out of bounds for dimension " + std::to_string(actual_dim) +
                                    " with size " + std::to_string(shape[actual_dim]));
    }

    std::vector<size_t> output_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) != actual_dim) {
            output_shape.push_back(shape[i]);
        }
    }

    if (output_shape.empty()) {
        output_shape = {1};
    }

    OpParams params{.axis = actual_dim, .output_precision = input_buffer.precision, .index_value = index_value};
    return add_node(OpType::INDEX, {input}, output_shape, params);
}

size_t CactusGraph::sum(size_t input, int axis) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape;

    if (axis == -1) {
        output_shape = {1};
    } else {
        output_shape = input_buffer.shape;
        output_shape.erase(output_shape.begin() + axis);
        if (output_shape.empty()) {
            output_shape = {1};
        }
    }

    OpParams params{.axis = axis, .output_precision = input_buffer.precision};
    return add_node(OpType::SUM, {input}, output_shape, params);
}

size_t CactusGraph::mean(size_t input, int axis) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape;

    if (axis == -1) {
        output_shape = {1};
    } else {
        output_shape = input_buffer.shape;
        output_shape.erase(output_shape.begin() + axis);
        if (output_shape.empty()) {
            output_shape = {1};
        }
    }

    OpParams params{.axis = axis, .output_precision = input_buffer.precision};
    return add_node(OpType::MEAN, {input}, output_shape, params);
}

size_t CactusGraph::variance(size_t input, int axis) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape;

    if (axis == -1) {
        output_shape = {1};
    } else {
        output_shape = input_buffer.shape;
        output_shape.erase(output_shape.begin() + axis);
        if (output_shape.empty()) {
            output_shape = {1};
        }
    }

    OpParams params{.axis = axis, .output_precision = input_buffer.precision};
    return add_node(OpType::VARIANCE, {input}, output_shape, params);
}

size_t CactusGraph::min(size_t input, int axis) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape;

    if (axis == -1) {
        output_shape = {1};
    } else {
        output_shape = input_buffer.shape;
        output_shape.erase(output_shape.begin() + axis);
        if (output_shape.empty()) {
            output_shape = {1};
        }
    }

    OpParams params{.axis = axis, .output_precision = input_buffer.precision};
    return add_node(OpType::MIN, {input}, output_shape, params);
}

size_t CactusGraph::max(size_t input, int axis) {
    const auto& input_buffer = get_output_buffer(input);
    std::vector<size_t> output_shape;

    if (axis == -1) {
        output_shape = {1};
    } else {
        output_shape = input_buffer.shape;
        output_shape.erase(output_shape.begin() + axis);
        if (output_shape.empty()) {
            output_shape = {1};
        }
    }

    OpParams params{.axis = axis, .output_precision = input_buffer.precision};
    return add_node(OpType::MAX, {input}, output_shape, params);
}

size_t CactusGraph::rms_norm(size_t input, size_t weight, float epsilon) {
    OpParams params{.epsilon = epsilon};
    return add_node(OpType::RMS_NORM, {input, weight}, {}, params);
}

size_t CactusGraph::rope(size_t input, float theta, size_t position_offset, ComputeBackend backend) {
    OpParams params{.theta = theta, .position_offset = position_offset, .backend = backend};
    return add_node(OpType::ROPE, {input}, {}, params);
}

size_t CactusGraph::softmax(size_t input, int axis) {
    OpParams params{.axis = axis};
    return add_node(OpType::SOFTMAX, {input}, {}, params);
}

size_t CactusGraph::topk(size_t input, size_t k) {
    const auto& input_buffer = get_output_buffer(input);

    if (input_buffer.shape.empty()) {
        throw std::runtime_error("TopK requires non-empty input tensor");
    }

    std::vector<size_t> output_shape = {2, input_buffer.shape[0], k};
    OpParams params{.output_precision = Precision::FP32, .top_k = k};

    return add_node(OpType::TOPK, {input}, output_shape, params);
}

size_t CactusGraph::layernorm(size_t input, size_t weight, size_t bias, float epsilon) {
    OpParams params{.epsilon = epsilon};
    return add_node(OpType::LAYERNORM, {input, weight, bias}, {}, params);
}

size_t CactusGraph::layernorm(size_t input, size_t weight, float epsilon) {
    OpParams params{.epsilon = epsilon};
    return add_node(OpType::LAYERNORM, {input, weight}, {}, params);
}

size_t CactusGraph::groupnorm(size_t input, size_t weight, size_t bias, size_t num_groups, float epsilon) {
    OpParams params{.epsilon = epsilon, .num_groups = num_groups};
    return add_node(OpType::GROUPNORM, {input, weight, bias}, {}, params);
}

size_t CactusGraph::attention(size_t query, size_t key, size_t value, float scale, bool is_causal, ComputeBackend backend) {
    OpParams params{.scale = scale, .is_causal = is_causal, .backend = backend};
    return add_node(OpType::ATTENTION, {query, key, value}, {}, params);
}

size_t CactusGraph::attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, ComputeBackend backend) {
    OpParams params{.scale = scale, .position_offset = position_offset, .backend = backend};
    return add_node(OpType::ATTENTION, {query, key, value}, {}, params);
}

size_t CactusGraph::attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, size_t window_size, ComputeBackend backend) {
    OpParams params{.scale = scale, .position_offset = position_offset, .window_size = window_size, .backend = backend};
    return add_node(OpType::ATTENTION, {query, key, value}, {}, params);
}

size_t CactusGraph::attention_int8_hybrid(size_t query, size_t key_new, size_t value_new, float scale, size_t position_offset,
                                          const int8_t* cached_keys, const int8_t* cached_values,
                                          const float* k_scales, const float* v_scales,
                                          size_t cache_len, size_t num_kv_heads, size_t head_dim, size_t window_size) {
    OpParams params;
    params.scale = scale;
    params.position_offset = position_offset;
    params.window_size = window_size;
    params.cached_keys_int8 = cached_keys;
    params.cached_values_int8 = cached_values;
    params.cached_k_scales = k_scales;
    params.cached_v_scales = v_scales;
    params.cache_seq_len = cache_len;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    return add_node(OpType::ATTENTION_INT8_HYBRID, {query, key_new, value_new}, {}, params);
}

size_t CactusGraph::conv1d_causal(size_t input, size_t weight, size_t, size_t dilation) {
    OpParams params{.dilation = dilation};
    return add_node(OpType::CONV1D_CAUSAL, {input, weight}, {}, params);
}

size_t CactusGraph::conv1d_k3(size_t input, size_t weight, size_t stride) {
    const auto& xin = get_output_buffer(input);
    const auto& w   = get_output_buffer(weight);

    if (xin.shape.size() != 3) throw std::runtime_error("conv1d_k3 expects N,C,L");
    if (w.shape.size()   != 3) throw std::runtime_error("weight must be [C_out,C_in,3]");
    if (w.shape[1] != xin.shape[1]) throw std::runtime_error("C_in mismatch in conv1d_k3");
    if (w.shape[2] != 3) throw std::runtime_error("K=3 expected in conv1d_k3");

    const size_t N    = xin.shape[0];
    const size_t L    = xin.shape[2];
    const size_t C_out= w.shape[0];
    const size_t K    = w.shape[2];

    const size_t pad = 1;
    const size_t L_out = (L + 2 * pad - K) / stride + 1;

    OpParams params{};
    params.stride = stride;
    params.output_precision = xin.precision;

    std::vector<size_t> out_shape{N, C_out, L_out};
    return add_node(OpType::CONV1D_K3, {input, weight}, out_shape, params);
}

size_t CactusGraph::conv1d_k7s3(size_t input, size_t weight, size_t bias) {
    const auto& xin = get_output_buffer(input);
    const auto& w   = get_output_buffer(weight);
    const auto& b   = get_output_buffer(bias);

    if (xin.shape.size() != 3) throw std::runtime_error("conv1d_k7s3 expects N,C,L");
    if (w.shape.size() != 3) throw std::runtime_error("weight must be [C_in, 7, C_out]");
    if (w.shape[0] != xin.shape[1]) throw std::runtime_error("C_in mismatch in conv1d_k7s3");
    if (w.shape[1] != 7) throw std::runtime_error("K=7 expected in conv1d_k7s3");
    
    size_t C_out = w.shape[2];
    if (b.total_size != C_out) throw std::runtime_error("Bias size mismatch");

    const size_t N    = xin.shape[0];
    const size_t L    = xin.shape[2];
    const size_t K    = 7;
    const size_t stride = 3;

    const size_t L_out = (L < K) ? 0 : (L - K) / stride + 1;

    OpParams params{};
    params.stride = stride;
    params.output_precision = xin.precision;

    std::vector<size_t> out_shape{N, C_out, L_out};
    return add_node(OpType::CONV1D_K7S3, {input, weight, bias}, out_shape, params);
}

size_t CactusGraph::conv1d(size_t input, size_t weight, size_t stride) {
    const auto& xin = get_output_buffer(input);
    const auto& w   = get_output_buffer(weight);
    
    if (xin.shape.size() != 3) throw std::runtime_error("conv1d expects N,C,L");
    if (w.shape.size() != 3) throw std::runtime_error("conv1d weight expects [C_out, C_in, K]");
    
    size_t N = xin.shape[0];
    size_t C_out = w.shape[0];
    size_t L = xin.shape[2];
    size_t K = w.shape[2];
    size_t L_out = (L - K) / stride + 1;
    
    OpParams params{.stride = stride};
    return add_node(OpType::CONV1D, {input, weight}, {N, C_out, L_out}, params);
}

size_t CactusGraph::conv1d(size_t input, size_t weight, size_t bias, size_t stride) {
    const auto& xin = get_output_buffer(input);
    const auto& w   = get_output_buffer(weight);
    
    if (xin.shape.size() != 3) throw std::runtime_error("conv1d expects N,C,L");
    if (w.shape.size() != 3) throw std::runtime_error("conv1d weight expects [C_out, C_in, K]");
    
    size_t N = xin.shape[0];
    size_t C_out = w.shape[0];
    size_t L = xin.shape[2];
    size_t K = w.shape[2];
    size_t L_out = (L - K) / stride + 1;
    
    OpParams params{.stride = stride};
    return add_node(OpType::CONV1D, {input, weight, bias}, {N, C_out, L_out}, params);
}

size_t CactusGraph::lstm_cell(size_t input, size_t h_prev, size_t c_prev, size_t weight_ih, size_t weight_hh, size_t bias_ih, size_t bias_hh) {
    const auto& h_buffer = get_output_buffer(h_prev);
    std::vector<size_t> output_shape = {h_buffer.shape[0], h_buffer.shape[1], 2};
    return add_node(OpType::LSTM_CELL, {input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh}, output_shape, {});
}

size_t CactusGraph::stft_magnitude(size_t input, size_t weight, size_t stride, size_t num_fft_bins) {
    const auto& xin = get_output_buffer(input);
    const auto& w = get_output_buffer(weight);

    if (xin.shape.size() != 3) throw std::runtime_error("stft_magnitude expects N,C,L input");
    if (w.shape.size() != 3) throw std::runtime_error("stft_magnitude weight expects [C_out, C_in, K]");

    size_t N = xin.shape[0];
    size_t L = xin.shape[2];
    size_t K = w.shape[2];
    size_t L_out = (L - K) / stride + 1;

    OpParams params{};
    params.stride = stride;
    params.num_fft_bins = num_fft_bins;

    return add_node(OpType::STFT_MAGNITUDE, {input, weight}, {N, num_fft_bins, L_out}, params);
}

size_t CactusGraph::concat(size_t input1, size_t input2, int axis) {
    const auto& buffer1 = get_output_buffer(input1);
    const auto& buffer2 = get_output_buffer(input2);

    if (buffer1.shape.size() != buffer2.shape.size()) {
        throw std::runtime_error("Concat requires inputs with same number of dimensions");
    }

    std::vector<size_t> output_shape = buffer1.shape;
    size_t ndims = output_shape.size();

    if (axis < 0) axis += ndims;
    if (axis < 0 || static_cast<size_t>(axis) >= ndims) {
        throw std::runtime_error("Invalid axis for concat operation");
    }

    for (size_t i = 0; i < ndims; ++i) {
        if (i != static_cast<size_t>(axis) && buffer1.shape[i] != buffer2.shape[i]) {
            throw std::runtime_error("Concat inputs must have same shape except on concat axis");
        }
    }

    output_shape[axis] = buffer1.shape[axis] + buffer2.shape[axis];

    OpParams params;
    params.axis = axis;
    return add_node(OpType::CONCAT, {input1, input2}, output_shape, params);
}

size_t CactusGraph::scatter_topk(size_t indices, size_t values, size_t num_classes) {
    const auto& indices_buffer = get_output_buffer(indices);
    const auto& values_buffer = get_output_buffer(values);

    if (indices_buffer.shape != values_buffer.shape) {
        throw std::runtime_error("ScatterTopK requires indices and values with identical shapes");
    }
    if (indices_buffer.shape.size() != 2) {
        throw std::runtime_error("ScatterTopK currently supports 2D tensors [batch, top_k]");
    }
    if (indices_buffer.precision != Precision::FP32 || values_buffer.precision != Precision::FP32) {
        throw std::runtime_error("ScatterTopK expects FP32 indices and values");
    }

    std::vector<size_t> output_shape = {num_classes, indices_buffer.shape[0]};
    OpParams params{.output_precision = Precision::FP32, .num_classes = num_classes};
    return add_node(OpType::SCATTER_TOPK, {indices, values}, output_shape, params);
}

size_t CactusGraph::sample(size_t logits, float temperature, float top_p, size_t top_k,
                           const std::unordered_map<uint32_t, float>& logit_bias) {
    const auto& logits_buffer = get_output_buffer(logits);

    if (logits_buffer.shape.empty()) {
        throw std::runtime_error("Sample requires non-empty logits tensor");
    }

    OpParams params;
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.random_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    params.output_precision = Precision::FP32;

    if (!logit_bias.empty()) {
        params.bias_indices.reserve(logit_bias.size());
        params.bias_values.reserve(logit_bias.size());
        for (const auto& [idx, val] : logit_bias) {
            params.bias_indices.push_back(idx);
            params.bias_values.push_back(val);
        }
    }

    std::vector<size_t> output_shape = {1};
    return add_node(OpType::SAMPLE, {logits}, output_shape, params);
}

size_t CactusGraph::scalar_add(size_t input, float value) {
    OpParams params{};
    params.scalar = value;
    params.output_precision = get_output_buffer(input).precision;
    return add_node(OpType::SCALAR_ADD, {input}, {}, params);
}

size_t CactusGraph::scalar_subtract(size_t input, float value) {
    OpParams params{};
    params.scalar = value;
    params.output_precision = get_output_buffer(input).precision;
    return add_node(OpType::SCALAR_SUBTRACT, {input}, {}, params);
}

size_t CactusGraph::scalar_multiply(size_t input, float value) {
    OpParams params{};
    params.scalar = value;
    params.output_precision = get_output_buffer(input).precision;
    return add_node(OpType::SCALAR_MULTIPLY, {input}, {}, params);
}

size_t CactusGraph::scalar_divide(size_t input, float value) {
    OpParams params{};
    params.scalar = value;
    params.output_precision = get_output_buffer(input).precision;
    return add_node(OpType::SCALAR_DIVIDE, {input}, {}, params);
}

size_t CactusGraph::scalar_exp(size_t input) {
    return add_node(OpType::SCALAR_EXP, {input}, {});
}

size_t CactusGraph::scalar_sqrt(size_t input) {
    return add_node(OpType::SCALAR_SQRT, {input}, {});
}

size_t CactusGraph::scalar_cos(size_t input) {
    return add_node(OpType::SCALAR_COS, {input}, {});
}

size_t CactusGraph::scalar_sin(size_t input) {
    return add_node(OpType::SCALAR_SIN, {input}, {});
}

size_t CactusGraph::relu(size_t input) {
    return add_node(OpType::RELU, {input}, {});
}

size_t CactusGraph::silu(size_t input) {
    return add_node(OpType::SILU, {input}, {});
}

size_t CactusGraph::gelu(size_t input) {
    return add_node(OpType::GELU, {input}, {});
}

size_t CactusGraph::gelu_erf(size_t input) {
    return add_node(OpType::GELU_ERF, {input}, {});
}

size_t CactusGraph::sigmoid(size_t input) {
    return add_node(OpType::SIGMOID, {input}, {});
}

size_t CactusGraph::tanh(size_t input) {
    return add_node(OpType::TANH, {input}, {});
}

size_t CactusGraph::rope_gptj(size_t input, float theta, size_t position_offset, size_t rot_dim, ComputeBackend backend) {
    OpParams params;
    params.theta = theta;
    params.position_offset = position_offset;
    params.scalar = static_cast<float>(rot_dim);
    params.backend = backend;
    return add_node(OpType::ROPE_GPTJ, {input}, {}, params);
}

size_t CactusGraph::gather(size_t tensor, size_t indices) {
    const auto& tensor_buffer = get_output_buffer(tensor);
    const auto& idx_shape = get_output_buffer(indices).shape;

    if (tensor_buffer.shape.empty()) {
        throw std::runtime_error("Cannot gather from scalar tensor");
    }

    std::vector<size_t> output_shape = idx_shape;
    for (size_t i = 1; i < tensor_buffer.shape.size(); i++) {
        output_shape.push_back(tensor_buffer.shape[i]);
    }

    OpParams params;
    params.output_precision = tensor_buffer.precision;

    return add_node(OpType::GATHER, {tensor, indices}, output_shape, params);
}

size_t CactusGraph::slice(size_t input, int axis, size_t start, size_t length) {
    const auto& input_buffer = get_output_buffer(input);
    if (input_buffer.shape.empty()) {
        throw std::runtime_error("Cannot slice a scalar tensor");
    }

    size_t axis_index = static_cast<size_t>(axis);
    size_t axis_size = input_buffer.shape[axis_index];

    if (start + length > axis_size) {
        throw std::runtime_error("Slice range extends beyond axis size");
    }

    std::vector<size_t> output_shape = input_buffer.shape;
    output_shape[axis_index] = length;

    OpParams params;
    params.axis = axis_index;
    params.slice_start = start;
    params.slice_length = length;
    params.output_precision = input_buffer.precision;

    return add_node(OpType::SLICE, {input}, output_shape, params);
}

size_t CactusGraph::embedding(size_t embedding_tensor, size_t indices) {
    const auto& emb_buffer = get_output_buffer(embedding_tensor);
    const auto& idx_shape = get_output_buffer(indices).shape;

    if (emb_buffer.shape.size() != 2) {
        std::cerr << "Error: Embedding tensor " << embedding_tensor << " has invalid shape: [";
        for(auto d : emb_buffer.shape) std::cerr << d << ",";
        std::cerr << "]. OpType=" << (int)nodes_[node_index_map_[embedding_tensor]]->op_type
                  << " ExtData=" << emb_buffer.external_data << std::endl;
        throw std::runtime_error("Embedding tensor must be 2D [vocab_size, hidden_dim]");
    }

    std::vector<size_t> output_shape = idx_shape;
    output_shape.push_back(emb_buffer.shape[1]);

    OpParams params;
    params.output_precision = (emb_buffer.precision == Precision::INT8) ? Precision::FP16 : emb_buffer.precision;

    return add_node(OpType::EMBEDDING, {embedding_tensor, indices}, output_shape, params);
}

size_t CactusGraph::bilinear_interpolation(size_t pos_embeds, size_t dst_height, size_t dst_width) {
    const auto& pos_embeds_buffer = get_output_buffer(pos_embeds);
    size_t embed_dim = pos_embeds_buffer.shape[1];
    std::vector<size_t> output_shape = {dst_height * dst_width, embed_dim};

    OpParams params;
    params.dst_height = dst_height;
    params.dst_width = dst_width;
    params.output_precision = Precision::FP16;

    return add_node(OpType::BILINEAR_INTERPOLATION, {pos_embeds}, output_shape, params);
}

size_t CactusGraph::precision_cast(size_t input, Precision target_precision) {
    OpParams params{};
    params.output_precision = target_precision;
    return add_node(OpType::PRECISION_CAST, {input}, {}, params);
}

size_t CactusGraph::quantize_activations(size_t input) {
    const auto& input_buffer = get_output_buffer(input);

    if (input_buffer.precision != Precision::FP16) {
        throw std::invalid_argument("quantize_activations requires FP16 input");
    }
    
    OpParams params{};
    params.output_precision = Precision::INT8;
    return add_node(OpType::QUANTIZE_ACTIVATIONS, {input}, input_buffer.shape, params);
}

size_t CactusGraph::add_node(OpType op_type, const std::vector<size_t>& inputs, const std::vector<size_t>& output_shape, const OpParams& params) {
    auto node = std::make_unique<GraphNode>(next_node_id_, op_type);
    node->input_ids = inputs;
    node->params = params;

    std::vector<size_t> result_shape = output_shape;
    if (result_shape.empty() && !inputs.empty()) {
        result_shape = nodes_[node_index_map_[inputs[0]]]->output_buffer.shape;
    }

    Precision result_precision = params.output_precision;
    if (op_type == OpType::PRECISION_CAST) {
        result_precision = params.output_precision;
    } else if (result_precision == Precision::INT8 && !inputs.empty()) {
        result_precision = nodes_[node_index_map_[inputs[0]]]->output_buffer.precision;
    }

    node->output_buffer = BufferDesc(result_shape, result_precision);

    size_t node_id = next_node_id_++;
    node_index_map_[node_id] = nodes_.size();
    nodes_.push_back(std::move(node));

    return node_id;
}

const BufferDesc& CactusGraph::get_output_buffer(size_t node_id) const {
    return nodes_[node_index_map_.at(node_id)]->output_buffer;
}

size_t CactusGraph::persistent(size_t source_node) {
    const auto& source_buffer = get_output_buffer(source_node);
    OpParams params;
    params.output_precision = source_buffer.precision;
    size_t node_id = add_node(OpType::PERSISTENT, {source_node}, source_buffer.shape, params);
    persistent_node_ids_.insert(node_id);
    return node_id;
}

bool CactusGraph::is_populated(size_t persistent_node_id) const {
    return populated_node_ids_.count(persistent_node_id) > 0;
}

void CactusGraph::invalidate_persistent(size_t persistent_node_id) {
    populated_node_ids_.erase(persistent_node_id);
    persistent_node_ids_.erase(persistent_node_id);
}
