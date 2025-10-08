#include "graph.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <set>

static const char* op_type_names[] = {
    "INPUT", "PRECISION_CAST",
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", 
    "MATMUL", "TRANSPOSE", "RESHAPE", "GATHER", "EMBEDDING",
    "SUM", "MEAN", "VARIANCE", "MIN", "MAX",
    "RMS_NORM", "ROPE", "SOFTMAX", "ATTENTION",
    "SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY", "SCALAR_DIVIDE", 
    "SCALAR_EXP", "SCALAR_SQRT", "SCALAR_COS", "SCALAR_SIN",
    "SILU", "GELU", "SAMPLE", "CONCAT"
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


size_t CactusGraph::reshape(size_t input, const std::vector<size_t>& new_shape) {
    OpParams params{.new_shape = new_shape};
    return add_node(OpType::RESHAPE, {input}, new_shape, params);
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

size_t CactusGraph::attention(size_t query, size_t key, size_t value, float scale, ComputeBackend backend) {
    OpParams params{.scale = scale, .backend = backend};
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
    OpParams params{.scalar = value, .output_precision = get_output_buffer(input).precision};
    return add_node(OpType::SCALAR_ADD, {input}, {}, params);
}

size_t CactusGraph::scalar_subtract(size_t input, float value) {
    OpParams params{.scalar = value, .output_precision = get_output_buffer(input).precision};
    return add_node(OpType::SCALAR_SUBTRACT, {input}, {}, params);
}

size_t CactusGraph::scalar_multiply(size_t input, float value) {
    OpParams params{.scalar = value, .output_precision = get_output_buffer(input).precision};
    return add_node(OpType::SCALAR_MULTIPLY, {input}, {}, params);
}

size_t CactusGraph::scalar_divide(size_t input, float value) {
    OpParams params{.scalar = value, .output_precision = get_output_buffer(input).precision};
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

size_t CactusGraph::precision_cast(size_t input, Precision target_precision) {
    OpParams params{.output_precision = target_precision};
    return add_node(OpType::PRECISION_CAST, {input}, {}, params);
}

void CactusGraph::set_input(size_t node_id, void* data, Precision) {
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
    allocate_buffers();
    
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
    
    for (auto& node : nodes_) {
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
            if (node->output_buffer.data) {
                size_t num_values = std::min(size_t(5), node->output_buffer.total_size);
                values_str = " values=[";
                
                if (node->output_buffer.precision == Precision::FP32) {
                    if (node->op_type == OpType::SAMPLE) {
                        uint32_t* uint32_data = reinterpret_cast<uint32_t*>(node->output_buffer.data.get());
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) values_str += ",";
                            values_str += std::to_string(uint32_data[i]);
                        }
                    } else {
                        float* float_data = reinterpret_cast<float*>(node->output_buffer.data.get());
                        for (size_t i = 0; i < num_values; ++i) {
                            if (i > 0) values_str += ",";
                            values_str += std::to_string(float_data[i]).substr(0, 6);
                        }
                    }
                } else if (node->output_buffer.precision == Precision::FP16) {
                    __fp16* fp16_data = reinterpret_cast<__fp16*>(node->output_buffer.data.get());
                    for (size_t i = 0; i < num_values; ++i) {
                        if (i > 0) values_str += ",";
                        values_str += std::to_string(static_cast<float>(fp16_data[i])).substr(0, 6);
                    }
                } else if (node->output_buffer.precision == Precision::INT8) {
                    int8_t* int8_data = reinterpret_cast<int8_t*>(node->output_buffer.data.get());
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
}


