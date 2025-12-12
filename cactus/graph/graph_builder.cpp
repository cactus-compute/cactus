#include "graph.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <system_error>
#include <limits>
#include <set>
#include <sstream>

static const char* op_type_names[] = {
    "INPUT", "PRECISION_CAST",
    "ADD", "ADD_CLIPPED", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "MATMUL", "TRANSPOSE", "RESHAPE", "SLICE", "GATHER", "EMBEDDING",
    "BILINEAR_INTERPOLATION",
    "SUM", "MEAN", "VARIANCE", "MIN", "MAX",
    "RMS_NORM", "ROPE", "SOFTMAX", "ATTENTION", "CONV1D_CAUSAL", "CONV1D_K3",
    "SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY", "SCALAR_DIVIDE",
    "SCALAR_EXP", "SCALAR_SQRT", "SCALAR_COS", "SCALAR_SIN",
    "SILU", "GELU", "GELU_ERF", "SAMPLE", "CONCAT",
    "SCATTER_TOPK",
    "TOPK", "LAYERNORM",
    "INDEX"
};

static const char* get_op_name(OpType op) {
    return op_type_names[static_cast<int>(op)];
}

BroadcastInfo BroadcastInfo::compute(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs) {
    BroadcastInfo info;
    size_t max_dims = std::max(lhs.size(), rhs.size());
    info.output_shape.resize(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t lhs_dim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
        size_t rhs_dim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;
        
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument("Shapes are not compatible for broadcasting");
        }
        
        info.output_shape[max_dims - 1 - i] = std::max(lhs_dim, rhs_dim);
    }
    
    info.needs_broadcasting = (lhs != info.output_shape || rhs != info.output_shape);
    return info;
}

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
    size_t N = pretransposed_rhs ? rhs_buffer.shape[0] : rhs_buffer.shape[1];
    
    if (K != rhs_K) {
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

size_t CactusGraph::conv1d_causal(size_t input, size_t weight, size_t, size_t dilation) {
    OpParams params{.dilation = dilation};
    return add_node(OpType::CONV1D_CAUSAL, {input, weight}, {}, params);
}

size_t CactusGraph::conv1d_k3(size_t input, size_t weight, size_t stride){
    const auto& xin = get_output_buffer(input);   // [N, C_in, L]
    const auto& w   = get_output_buffer(weight);  // [C_out, C_in, 3]

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

size_t CactusGraph::sample(size_t logits, float temperature, float top_p, size_t top_k) {
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

size_t CactusGraph::silu(size_t input) {
    return add_node(OpType::SILU, {input}, {});
}

size_t CactusGraph::gelu(size_t input) {
    return add_node(OpType::GELU, {input}, {});
}

size_t CactusGraph::gelu_erf(size_t input){
    return add_node(OpType::GELU_ERF, {input}, {});
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


size_t CactusGraph::mmap_embeddings(const std::string& filename) {
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);
    
    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Memory-mapped embeddings must be 2D [vocab_size, embedding_dim]");
    }
    
    Precision precision = mapped_file->precision();
    
    float scale = 1.0f;
    if (precision == Precision::INT8) {
        std::string scale_filename = filename;
        size_t dot_pos = scale_filename.find_last_of('.');
        if (dot_pos != std::string::npos) {
            scale_filename = scale_filename.substr(0, dot_pos) + ".scale";
            std::ifstream scale_file(scale_filename);
            if (scale_file.is_open()) {
                scale_file >> scale;
                scale_file.close();
            }
        }
    }
    
    size_t node_id = input(shape, precision);
    set_quantization_scale(node_id, scale);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);
    mapped_files_.push_back(std::move(mapped_file));
    return node_id;
}

size_t CactusGraph::mmap_weights(const std::string& filename) {
    auto it = weight_cache_.find(filename);
    if (it != weight_cache_.end()) {
        return it->second;
    }
    
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);
    
    const auto& shape = mapped_file->shape();
    Precision precision = mapped_file->precision();
    
    float scale = 1.0f;
    if (precision == Precision::INT8) {
        std::string scale_filename = filename;
        size_t dot_pos = scale_filename.find_last_of('.');
        if (dot_pos != std::string::npos) {
            scale_filename = scale_filename.substr(0, dot_pos) + ".scale";
            std::ifstream scale_file(scale_filename);
            if (scale_file.is_open()) {
                scale_file >> scale;
                scale_file.close();
            }
        }
    }
    
    size_t node_id = input(shape, precision);
    set_quantization_scale(node_id, scale);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);
    mapped_files_.push_back(std::move(mapped_file));
    weight_cache_[filename] = node_id;
    return node_id;
}

size_t CactusGraph::load_weights(const std::string& filename) {
    auto it = weight_cache_.find(filename);
    if (it != weight_cache_.end()) {
        return it->second;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filename);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(file_size);
    if (!file.read(buffer.get(), file_size)) {
        throw std::runtime_error("Failed to read weight file: " + filename);
    }
    file.close();

    const char* ptr = buffer.get();
    size_t offset = 0;

    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        shape[i] = static_cast<size_t>(dim_val);
        offset += sizeof(uint64_t);
    }

    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    Precision precision = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);

    offset += sizeof(uint64_t);

    float scale = 1.0f;
    if (precision == Precision::INT8) {
        scale = *reinterpret_cast<const float*>(ptr + offset);
        offset += sizeof(float);

        std::string scale_filename = filename;
        size_t dot_pos = scale_filename.find_last_of('.');
        if (dot_pos != std::string::npos) {
            scale_filename = scale_filename.substr(0, dot_pos) + ".scale";
            std::ifstream scale_file(scale_filename);
            if (scale_file.is_open()) {
                float external_scale;
                scale_file >> external_scale;
                scale = external_scale;  
                scale_file.close();
            }
        }
    }

    size_t node_id = input(shape, precision);
    set_quantization_scale(node_id, scale);

    const void* weight_data = ptr + offset;
    set_input(node_id, weight_data, precision);

    weight_cache_[filename] = node_id;

    return node_id;
}

void CactusGraph::set_quantization_scale(size_t node_id, float scale) {
    auto it = node_index_map_.find(node_id);
    if (it != node_index_map_.end()) {
        nodes_[it->second]->output_buffer.quantization_scale = scale;
    }
}


size_t CactusGraph::embedding(const std::string& filename, size_t indices) {
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);
    
    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Embedding file must contain 2D tensor [vocab_size, hidden_dim]");
    }
    
    Precision precision = mapped_file->precision();
    size_t embeddings_node = input(shape, precision);
    set_external_input(embeddings_node, const_cast<void*>(mapped_file->data()), precision);
    mapped_files_.push_back(std::move(mapped_file));
    
    const auto& idx_shape = get_output_buffer(indices).shape;
    std::vector<size_t> output_shape = idx_shape;
    output_shape.push_back(shape[1]);  
    
    OpParams params;
    params.output_precision = (precision == Precision::INT8) ? Precision::FP16 : precision;
    
    return add_node(OpType::EMBEDDING, {embeddings_node, indices}, output_shape, params);
}

size_t CactusGraph::embedding(size_t embedding_tensor, size_t indices) {
    const auto& emb_buffer = get_output_buffer(embedding_tensor);
    const auto& idx_shape = get_output_buffer(indices).shape;
    
    if (emb_buffer.shape.size() != 2) {
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
    params.output_precision = Precision::FP32;
    
    return add_node(OpType::BILINEAR_INTERPOLATION, {pos_embeds}, output_shape, params);
}   

size_t CactusGraph::precision_cast(size_t input, Precision target_precision) {
    OpParams params{};
    params.output_precision = target_precision;
    return add_node(OpType::PRECISION_CAST, {input}, {}, params);
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
    
    if (!capture_requested) {
        capture_requested = capture_to_stdout || !capture_file_path.empty();
    } else if (capture_file_path.empty() && !capture_to_stdout) {
        capture_to_stdout = true;
    }

    size_t capture_preview_count = static_cast<size_t>(get_env_int("CACTUS_CAPTURE_PREVIEW_COUNT", 8));
    size_t capture_max_elements = static_cast<size_t>(get_env_int("CACTUS_CAPTURE_MAX_ELEMENTS", 65536));

    bool enable_profiling = !profile_file.empty();
    std::ofstream profile_out;
    std::ostream* out = &std::cout;
    
    if (enable_profiling) {
        profile_out.open(profile_file);
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

        if (capture_outputs.empty()) {
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
                        float scale = buffer.quantization_scale;
                        for (size_t i = 0; i < elements_to_process; ++i) {
                            accumulate(static_cast<float>(typed[i]) * scale, i);
                        }
                    } else {
                        has_data = false;
                    }
                } else {
                    has_data = false;
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
    buffer_pool_.clear();
    shrink_thread_local_buffers();
}
