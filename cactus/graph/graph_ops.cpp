#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <cmath>

namespace {
    thread_local std::vector<int8_t> transpose_buffer_int8;
    thread_local std::vector<__fp16> transpose_buffer_fp16;
    thread_local std::vector<float> transpose_buffer_fp32;
    thread_local std::vector<int8_t> quantization_buffer_int8;
    std::mutex buffer_mutex;
    
    void ensure_transpose_buffer_int8(size_t required_size) {
        if (transpose_buffer_int8.size() < required_size) {
            transpose_buffer_int8.resize(required_size);
        }
    }
    
    void ensure_transpose_buffer_fp16(size_t required_size) {
        if (transpose_buffer_fp16.size() < required_size) {
            transpose_buffer_fp16.resize(required_size);
        }
    }
    
    void ensure_transpose_buffer_fp32(size_t required_size) {
        if (transpose_buffer_fp32.size() < required_size) {
            transpose_buffer_fp32.resize(required_size);
        }
    }
    
    void ensure_quantization_buffer_int8(size_t required_size) {
        if (quantization_buffer_int8.size() < required_size) {
            quantization_buffer_int8.resize(required_size);
        }
    }
}

void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    int axis = node.params.axis;
    
    if (axis == -1) {
        switch (node.op_type) {
            case OpType::SUM:
                if (input_buffer.precision == Precision::INT8) {
                    int64_t result = cactus_sum_all_int8(input_buffer.data_as<int8_t>(), input_buffer.total_size);
                    node.output_buffer.data_as<int8_t>()[0] = static_cast<int8_t>(std::max(static_cast<int64_t>(-128), std::min(static_cast<int64_t>(127), result)));
                } else if (input_buffer.precision == Precision::FP16) {
                    throw std::runtime_error("FP16 sum not yet implemented");
                } else {
                    double result = cactus_sum_all_f32(input_buffer.data_as<float>(), input_buffer.total_size);
                    node.output_buffer.data_as<float>()[0] = static_cast<float>(result);
                }
                break;
            case OpType::MEAN:
                if (input_buffer.precision == Precision::INT8) {
                    double result = cactus_mean_all_int8(input_buffer.data_as<int8_t>(), input_buffer.total_size);
                    node.output_buffer.data_as<int8_t>()[0] = static_cast<int8_t>(std::max(-128.0, std::min(127.0, result)));
                } else if (input_buffer.precision == Precision::FP16) {
                    double result = cactus_mean_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                    node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                } else {
                    double result = cactus_mean_all_f32(input_buffer.data_as<float>(), input_buffer.total_size);
                    node.output_buffer.data_as<float>()[0] = static_cast<float>(result);
                }
                break;
            case OpType::VARIANCE:
                if (input_buffer.precision == Precision::INT8) {
                    double result = cactus_variance_all_int8(input_buffer.data_as<int8_t>(), input_buffer.total_size);
                    node.output_buffer.data_as<int8_t>()[0] = static_cast<int8_t>(std::max(-128.0, std::min(127.0, result)));
                } else {
                    double result = cactus_variance_all_f32(input_buffer.data_as<float>(), input_buffer.total_size);
                    node.output_buffer.data_as<float>()[0] = static_cast<float>(result);
                }
                break;
            case OpType::MIN:
                if (input_buffer.precision == Precision::INT8) {
                    int64_t result = cactus_min_all_int8(input_buffer.data_as<int8_t>(), input_buffer.total_size);
                    node.output_buffer.data_as<int8_t>()[0] = static_cast<int8_t>(result);
                } else {
                    float result = cactus_min_all_f32(input_buffer.data_as<float>(), input_buffer.total_size);
                    node.output_buffer.data_as<float>()[0] = result;
                }
                break;
            case OpType::MAX:
                if (input_buffer.precision == Precision::INT8) {
                    int64_t result = cactus_max_all_int8(input_buffer.data_as<int8_t>(), input_buffer.total_size);
                    node.output_buffer.data_as<int8_t>()[0] = static_cast<int8_t>(result);
                } else {
                    float result = cactus_max_all_f32(input_buffer.data_as<float>(), input_buffer.total_size);
                    node.output_buffer.data_as<float>()[0] = result;
                }
                break;
            default: break;
        }
    } else {
        const auto& shape = input_buffer.shape;
        size_t axis_idx = static_cast<size_t>(axis);
        
        size_t outer_size = 1;
        for (size_t i = 0; i < axis_idx; i++) {
            outer_size *= shape[i];
        }
        
        size_t axis_size = shape[axis_idx];
        
        size_t inner_size = 1;
        for (size_t i = axis_idx + 1; i < shape.size(); i++) {
            inner_size *= shape[i];
        }
        
        
        switch (node.op_type) {
            case OpType::SUM:
                if (input_buffer.precision == Precision::INT8) {
                    cactus_sum_axis_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                        outer_size, axis_size, inner_size);
                } else {
                    cactus_sum_axis_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(), 
                                        outer_size, axis_size, inner_size);
                }
                break;
            case OpType::MEAN:
                if (input_buffer.precision == Precision::INT8) {
                    cactus_mean_axis_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                         outer_size, axis_size, inner_size);
                } else if (input_buffer.precision == Precision::FP16) {
                    cactus_mean_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(), 
                                        outer_size, axis_size, inner_size);
                } else {
                    cactus_mean_axis_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(), 
                                         outer_size, axis_size, inner_size);
                }
                break;
            case OpType::VARIANCE:
                if (input_buffer.precision == Precision::INT8) {
                    cactus_variance_axis_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                             outer_size, axis_size, inner_size);
                } else {
                    cactus_variance_axis_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(), 
                                             outer_size, axis_size, inner_size);
                }
                break;
            case OpType::MIN:
                if (input_buffer.precision == Precision::INT8) {
                    cactus_min_axis_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                        outer_size, axis_size, inner_size);
                } else {
                    cactus_min_axis_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(), 
                                        outer_size, axis_size, inner_size);
                }
                break;
            case OpType::MAX:
                if (input_buffer.precision == Precision::INT8) {
                    cactus_max_axis_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                        outer_size, axis_size, inner_size);
                } else {
                    cactus_max_axis_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(), 
                                        outer_size, axis_size, inner_size);
                }
                break;
            default: break;
        }
    }
}

void compute_fused_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    switch (node.op_type) {
        case OpType::GATHER: {
            const auto& tensor_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            size_t first_dim = tensor_buffer.shape[0];
            size_t element_size = 1;
            for (size_t i = 1; i < tensor_buffer.shape.size(); i++) {
                element_size *= tensor_buffer.shape[i];
            }
            
            size_t num_indices = indices_buffer.total_size;
            size_t bytes_per_element = element_size * PrecisionTraits::size_of(tensor_buffer.precision);
            
            if (tensor_buffer.precision == Precision::INT8) {
                const int8_t* tensor_data = tensor_buffer.data_as<int8_t>();
                const int8_t* indices = indices_buffer.data_as<int8_t>();
                int8_t* output = node.output_buffer.data_as<int8_t>();
                
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                }
            } else if (tensor_buffer.precision == Precision::FP16) {
                const __fp16* tensor_data = tensor_buffer.data_as<__fp16>();
                __fp16* output = node.output_buffer.data_as<__fp16>();
                
                if (indices_buffer.precision == Precision::INT8) {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                } else {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                }
            } else {
                const float* tensor_data = tensor_buffer.data_as<float>();
                float* output = node.output_buffer.data_as<float>();
                
                if (indices_buffer.precision == Precision::INT8) {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                } else {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                }
            }
            break;
        }
        case OpType::EMBEDDING: {
            const auto& embeddings_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            size_t vocab_size = embeddings_buffer.shape[0];
            size_t hidden_dim = embeddings_buffer.shape[1];
            size_t num_indices = indices_buffer.total_size;
            
            if (embeddings_buffer.precision == Precision::INT8) {
                const int8_t* embeddings = embeddings_buffer.data_as<int8_t>();
                __fp16* output = node.output_buffer.data_as<__fp16>();
                float scale = embeddings_buffer.quantization_scale;
                
                if (indices_buffer.precision == Precision::FP32) {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = static_cast<__fp16>(embeddings[idx * hidden_dim + j] * scale);
                        }
                    }
                } else {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = static_cast<__fp16>(embeddings[idx * hidden_dim + j] * scale);
                        }
                    }
                }
            } else if (embeddings_buffer.precision == Precision::FP16) {
                const __fp16* embeddings = embeddings_buffer.data_as<__fp16>();
                __fp16* output = node.output_buffer.data_as<__fp16>();
                
                if (indices_buffer.precision == Precision::FP32) {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                } else {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                }
            } else {
                const float* embeddings = embeddings_buffer.data_as<float>();
                float* output = node.output_buffer.data_as<float>();
                
                if (indices_buffer.precision == Precision::FP32) {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                } else {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                }
            }
            break;
        }
        case OpType::RMS_NORM: {
            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            if (input_buffer.shape.size() != 2) {
                throw std::runtime_error("RMS normalization requires 2D input tensor [batch_size, dims], got " + 
                                        std::to_string(input_buffer.shape.size()) + "D tensor");
            }
            
            size_t batch_size = input_buffer.shape[0];
            size_t dims = input_buffer.shape[1];
            
            if (input_buffer.precision == Precision::FP32) {
                cactus_rms_norm_f32(input_buffer.data_as<float>(), weight_buffer.data_as<float>(), 
                   node.output_buffer.data_as<float>(), batch_size, dims, node.params.epsilon);
            } else if (input_buffer.precision == Precision::FP16) {
                cactus_rms_norm_f16(input_buffer.data_as<__fp16>(), weight_buffer.data_as<__fp16>(), 
                   node.output_buffer.data_as<__fp16>(), batch_size, dims, node.params.epsilon);
            } else if (input_buffer.precision == Precision::INT8) {
                float input_scale = input_buffer.quantization_scale;
                
                std::vector<float> fp32_temp(batch_size * dims);
                
                if (weight_buffer.precision == Precision::FP16) {
                    std::vector<float> fp32_weights(dims);
                    const __fp16* fp16_weights = weight_buffer.data_as<__fp16>();
                    for (size_t i = 0; i < dims; i++) {
                        fp32_weights[i] = static_cast<float>(fp16_weights[i]);
                    }
                    cactus_rms_norm_i8_f32(input_buffer.data_as<int8_t>(), fp32_weights.data(), 
                                           fp32_temp.data(), batch_size, dims, node.params.epsilon, 
                                           input_scale);
                } else if (weight_buffer.precision == Precision::FP32) {
                    cactus_rms_norm_i8_f32(input_buffer.data_as<int8_t>(), weight_buffer.data_as<float>(), 
                                           fp32_temp.data(), batch_size, dims, node.params.epsilon, 
                                           input_scale);
                } else {
                    throw std::runtime_error("INT8 RMS normalization requires FP16 or FP32 weight precision");
                }
                
                float fast_scale = 2.0f / 127.0f; 
                
                for (size_t i = 0; i < batch_size * dims; ++i) {
                    float quantized = fp32_temp[i] / fast_scale;
                    node.output_buffer.data_as<int8_t>()[i] = static_cast<int8_t>(
                        std::round(std::max(-128.0f, std::min(127.0f, quantized))));
                }
                
                node.output_buffer.quantization_scale = fast_scale;
            } else {
                throw std::runtime_error("RMS normalization only supports FP32, FP16, and INT8 precision");
            }
            break;
        }
        case OpType::ROPE: {
            if (node.params.backend == ComputeBackend::NPU) {
                throw std::runtime_error("NPU RoPE operation not yet implemented");
            }
            
            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& shape = input_buffer.shape;
            
            if (shape.size() >= 4) {
                size_t batch_size = shape[0];
                size_t seq_len = shape[1];
                size_t num_heads = shape[2];
                size_t head_dim = shape[3];
                
                if (input_buffer.precision == Precision::INT8 && node.output_buffer.precision == Precision::INT8) {
                    float input_scale = 1.0f / input_buffer.quantization_scale;
                    float output_scale = 1.0f / node.output_buffer.quantization_scale;
                    cactus_rope_i8_f32_i8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(),
                                         batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta,
                                         input_scale, output_scale);
                } else if (input_buffer.precision == Precision::FP16 && node.output_buffer.precision == Precision::FP16) {
                    cactus_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                   batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta);
                } else if (input_buffer.precision == Precision::FP32 && node.output_buffer.precision == Precision::FP32) {
                    cactus_rope_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(),
                                   batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta);
                } else {
                    throw std::runtime_error("RoPE operation only supports FP32->FP32, FP16->FP16, or INT8->INT8 precision");
                }
            } else {
                throw std::runtime_error("RoPE operation requires 4D tensor with shape [batch, seq_len, num_heads, head_dim], got " + 
                                        std::to_string(shape.size()) + "D tensor");
            }
            break;
        }
        case OpType::SOFTMAX: {
            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& shape = input_buffer.shape;
            
            if (shape.size() >= 2) {
                size_t batch_size = 1;
                for (size_t i = 0; i < shape.size() - 1; i++) {
                    batch_size *= shape[i];
                }
                size_t vocab_size = shape[shape.size() - 1];
                
                if (input_buffer.precision == Precision::FP16) {
                    cactus_softmax_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                      batch_size, 1, vocab_size);
                } else {
                    cactus_softmax_f32(input_buffer.data_as<float>(), node.output_buffer.data_as<float>(),
                                      batch_size, 1, vocab_size);
                }
            } else {
                throw std::runtime_error("Softmax operation requires at least 2D tensor, got " + 
                                        std::to_string(shape.size()) + "D tensor");
            }
            break;
        }
        case OpType::ATTENTION: {
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
            
            size_t batch_size = q_shape[0];
            size_t seq_len = q_shape[1];
            size_t num_q_heads = q_shape[2];
            size_t head_dim = q_shape[3];
            size_t num_kv_heads = k_shape[2];  
            size_t kv_seq_len = key_buffer.shape[1]; 
            
            if (query_buffer.precision == Precision::INT8) {
                float q_scale = 1.0f / query_buffer.quantization_scale;
                float k_scale = 1.0f / key_buffer.quantization_scale;
                float v_scale = 1.0f / value_buffer.quantization_scale;
                float output_scale = 1.0f / node.output_buffer.quantization_scale;
                cactus_attention_int8(query_buffer.data_as<int8_t>(), key_buffer.data_as<int8_t>(),
                                      value_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(),
                                      batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, nullptr,
                                      q_scale, k_scale, v_scale, output_scale, node.params.position_offset, node.params.window_size,
                                      node.params.is_causal);
            } else if (query_buffer.precision == Precision::FP16) {
                cactus_attention_f16(query_buffer.data_as<__fp16>(), key_buffer.data_as<__fp16>(),
                                     value_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                     batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, nullptr,
                                     node.params.position_offset, node.params.window_size, node.params.is_causal);
            } else if (query_buffer.precision == Precision::FP32) {
                cactus_attention_f32(query_buffer.data_as<float>(), key_buffer.data_as<float>(),
                                     value_buffer.data_as<float>(), node.output_buffer.data_as<float>(),
                                     batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, nullptr,
                                     node.params.position_offset, node.params.window_size, node.params.is_causal);
            } else {
                throw std::runtime_error("Attention operation only supports INT8, FP16, and FP32 precision, got " + 
                                       std::to_string(static_cast<int>(query_buffer.precision)));
            }
            break;
        }
        case OpType::CONCAT: {
            const auto& input1_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& input2_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            std::vector<size_t> shape1 = input1_buffer.shape;
            std::vector<size_t> shape2 = input2_buffer.shape;
            std::vector<size_t> output_shape = node.output_buffer.shape;
            
            if (input1_buffer.precision == Precision::INT8) {
                cactus_concat_int8(input1_buffer.data_as<int8_t>(), input2_buffer.data_as<int8_t>(),
                                  node.output_buffer.data_as<int8_t>(),
                                  shape1.data(), shape2.data(), output_shape.data(),
                                  shape1.size(), node.params.axis);
            } else if (input1_buffer.precision == Precision::FP16) {
                cactus_concat_f16(input1_buffer.data_as<__fp16>(), input2_buffer.data_as<__fp16>(),
                                 node.output_buffer.data_as<__fp16>(),
                                 shape1.data(), shape2.data(), output_shape.data(),
                                 shape1.size(), node.params.axis);
            } else if (input1_buffer.precision == Precision::FP32) {
                cactus_concat_f32(input1_buffer.data_as<float>(), input2_buffer.data_as<float>(),
                                 node.output_buffer.data_as<float>(),
                                 shape1.data(), shape2.data(), output_shape.data(),
                                 shape1.size(), node.params.axis);
            }
            break;
        }
        default: break;
    }
}

void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU transpose operation not yet implemented");
    }
    
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    
    const auto& permutation = node.params.permutation;
    
    switch (input_buffer.precision) {
        case Precision::INT8:
            cactus_transpose_int8(input_buffer.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                 input_buffer.shape.data(), permutation.data(), permutation.size(),
                                 0, input_buffer.total_size);
            break;
        case Precision::FP16: {
            throw std::runtime_error("FP16 transpose not yet implemented");
            break;
        }
        case Precision::FP32: {
            const float* input = input_buffer.data_as<float>();
            float* output = node.output_buffer.data_as<float>();
            cactus_transpose_f32(input, output, input_buffer.shape.data(), permutation.data(), permutation.size(), 0, input_buffer.total_size);
            break;
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
    size_t N = node.params.pretransposed_rhs ? 
               rhs_shape[rhs_shape.size() - 2] : rhs_shape[rhs_shape.size() - 1];
    
    bool pretransposed_rhs = node.params.pretransposed_rhs;
    
    ComputeBackend backend = node.params.backend;
    
    if (backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU matrix multiplication not yet implemented");
    }
    
    if (lhs_buffer.precision == Precision::FP16 && rhs_buffer.precision == Precision::INT8) {
        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        const int8_t* rhs = rhs_buffer.data_as<int8_t>();
        __fp16* output = node.output_buffer.data_as<__fp16>();
        
        size_t lhs_size = M * K;
        size_t output_size = M * N;
        ensure_quantization_buffer_int8(lhs_size);
        
        float max_abs = cactus_fp16_max_abs(lhs, lhs_size);
        float lhs_scale = max_abs / 127.0f;
        if (lhs_scale == 0.0f) lhs_scale = 1.0f;
        
        Quantization::fp16_to_int8(lhs, quantization_buffer_int8.data(), lhs_size, lhs_scale);
        
        float rhs_scale = rhs_buffer.quantization_scale;
        
        std::vector<int32_t> int32_output(output_size);
        
        if (pretransposed_rhs) {
            cactus_matmul_int8_to_int32(quantization_buffer_int8.data(), rhs, 
                                           int32_output.data(), M, K, N);
        } else {
            size_t transpose_size = rhs_shape[0] * rhs_shape[1];
            ensure_transpose_buffer_int8(transpose_size);
            
            size_t rhs_perm[] = {1, 0};
            cactus_transpose_int8(rhs, transpose_buffer_int8.data(), rhs_shape.data(), rhs_perm, 2, 0, rhs_shape[0]);
            cactus_matmul_int8_to_int32(quantization_buffer_int8.data(), transpose_buffer_int8.data(), 
                                     int32_output.data(), M, K, N);
        }
        
        float combined_scale = lhs_scale * rhs_scale;
        cactus_int32_to_fp16_scaled(int32_output.data(), output, output_size, combined_scale);
        
    } else {
        switch (lhs_buffer.precision) {
            case Precision::INT8: {
                const int8_t* lhs = lhs_buffer.data_as<int8_t>();
                const int8_t* rhs = rhs_buffer.data_as<int8_t>();
                
                float lhs_scale = lhs_buffer.quantization_scale;
                float rhs_scale = rhs_buffer.quantization_scale;
                
                if (node.output_buffer.quantization_scale == 1.0f) {
                    float estimated_scale = lhs_scale * rhs_scale;
                    estimated_scale = std::max(0.001f, std::min(estimated_scale, 10.0f));
                    
                    node.output_buffer.quantization_scale = estimated_scale;
                    
                }
                
                int8_t* output = node.output_buffer.data_as<int8_t>();
                
                if (pretransposed_rhs) {
                    cactus_matmul_int8(lhs, rhs, output, M, K, N, lhs_scale, rhs_scale, node.output_buffer.quantization_scale);
                } else {
                    size_t transpose_size = rhs_shape[0] * rhs_shape[1];
                    ensure_transpose_buffer_int8(transpose_size);
                    
                    size_t rhs_perm[] = {1, 0};
                    cactus_transpose_int8(rhs, transpose_buffer_int8.data(), rhs_shape.data(), rhs_perm, 2, 0, rhs_shape[0]);
                    cactus_matmul_int8(lhs, transpose_buffer_int8.data(), output, M, K, N, lhs_scale, rhs_scale, node.output_buffer.quantization_scale);
                }
                
                break;
            }
            case Precision::FP16: {
                const __fp16* lhs = lhs_buffer.data_as<__fp16>();
                const __fp16* rhs = rhs_buffer.data_as<__fp16>();
                __fp16* output = node.output_buffer.data_as<__fp16>();
                
                if (pretransposed_rhs) {
                    cactus_matmul_f16(lhs, rhs, output, M, K, N);
                } else {
                    size_t transpose_size = rhs_shape[0] * rhs_shape[1];
                    ensure_transpose_buffer_fp16(transpose_size);
                    
                    cactus_transpose_2d_f32(reinterpret_cast<const float*>(rhs), 
                                            reinterpret_cast<float*>(transpose_buffer_fp16.data()), 
                                            rhs_shape[0], rhs_shape[1], 0, rhs_shape[0]);
                    cactus_matmul_f16(lhs, transpose_buffer_fp16.data(), output, M, K, N);
                }
                
                break;
            }
            case Precision::FP32: {
                const float* lhs = lhs_buffer.data_as<float>();
                const float* rhs = rhs_buffer.data_as<float>();
                float* output = node.output_buffer.data_as<float>();
                
                if (pretransposed_rhs) {
                    cactus_matmul_f32(lhs, rhs, output, M, K, N);
                } else {
                    size_t transpose_size = rhs_shape[0] * rhs_shape[1];
                    ensure_transpose_buffer_fp32(transpose_size);
                    
                    size_t rhs_perm[] = {1, 0};
                    cactus_transpose_f32(rhs, transpose_buffer_fp32.data(), rhs_shape.data(), rhs_perm, 2, 0, rhs_shape[0]);
                    cactus_matmul_f32(lhs, transpose_buffer_fp32.data(), output, M, K, N);
                }
                break;
            }
        }
    }
}

void compute_sample_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& logits_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    
    float temperature = node.params.temperature;
    float top_p = node.params.top_p;
    size_t top_k = node.params.top_k;
    size_t random_seed = node.params.random_seed;
    
    if (logits_buffer.shape.size() != 2) {
        throw std::runtime_error("Sample expects 2D logits tensor [seq_len, vocab_size]");
    }
    
    size_t seq_len = logits_buffer.shape[0];
    size_t vocab_size = logits_buffer.shape[1];
    size_t last_token_offset = (seq_len - 1) * vocab_size;
    
    if (logits_buffer.precision == Precision::INT8) {
        const int8_t* logits_int8 = logits_buffer.data_as<int8_t>();
        float scale = logits_buffer.quantization_scale;
        
        std::vector<float> probs(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = logits_int8[last_token_offset + i] * scale;
        }
        
        cactus_sample_f32(probs.data(), node.output_buffer.data_as<uint32_t>(), 
                         vocab_size, temperature, top_p, top_k, random_seed);
    } else if (logits_buffer.precision == Precision::FP16) {
        const __fp16* logits_fp16 = logits_buffer.data_as<__fp16>();
        cactus_sample_f16(logits_fp16 + last_token_offset, node.output_buffer.data_as<uint32_t>(), 
                         vocab_size, temperature, top_p, top_k, random_seed);
    } else {
        const float* logits_fp32 = logits_buffer.data_as<float>();
        cactus_sample_f32(logits_fp32 + last_token_offset, node.output_buffer.data_as<uint32_t>(), 
                         vocab_size, temperature, top_p, top_k, random_seed);
    }
}

void compute_scatter_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& values_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    if (indices_buffer.shape != values_buffer.shape) {
        throw std::runtime_error("ScatterTopK requires indices and values with identical shapes");
    }
    if (indices_buffer.shape.size() != 2) {
        throw std::runtime_error("ScatterTopK currently supports 2D tensors");
    }

    size_t batch_size = indices_buffer.shape[0];
    size_t top_k = indices_buffer.shape[1];
    size_t num_classes = node.params.num_classes;

    if (num_classes == 0) {
        throw std::runtime_error("ScatterTopK requires num_classes > 0");
    }

    float* output = node.output_buffer.data_as<float>();

    if (indices_buffer.precision != Precision::FP32 || values_buffer.precision != Precision::FP32) {
        throw std::runtime_error("ScatterTopK currently expects FP32 inputs");
    }

    cactus_scatter_topk_f32(indices_buffer.data_as<float>(), values_buffer.data_as<float>(), output, batch_size, top_k, num_classes);
}

void compute_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;    
    if (input_buffer.shape.size() != 2) {
        throw std::runtime_error("TopK currently only supports 2D tensors [batch, features]");
    }
    
    size_t k = node.params.top_k;
    size_t batch_size = input_buffer.shape[0];
    size_t feature_size = input_buffer.shape[1];
    size_t block_size = batch_size * k;
    
    std::vector<float> input_float(input_buffer.total_size);
    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("TopK currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        const float* input_fp32 = input_buffer.data_as<float>();
        std::memcpy(input_float.data(), input_fp32, input_buffer.total_size * sizeof(float));
    }
    
    float* output = node.output_buffer.data_as<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = input_float.data() + b * feature_size;
        
        std::vector<std::pair<size_t, float>> indexed_values(feature_size);
        for (size_t i = 0; i < feature_size; ++i) {
            indexed_values[i] = {i, row[i]};
        }
        
        std::partial_sort(indexed_values.begin(), 
                         indexed_values.begin() + k, 
                         indexed_values.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        float* idx_out_row = output + b * k;
        float* val_out_row = output + block_size + b * k;
        for (size_t i = 0; i < k; ++i) {
            idx_out_row[i] = static_cast<float>(indexed_values[i].first);
            val_out_row[i] = indexed_values[i].second;
        }
    }
}

void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& input_shape = input_buffer.shape;
    
    int dim = node.params.axis;
    size_t index_value = node.params.index_value;

    size_t element_size = PrecisionTraits::size_of(input_buffer.precision);
    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());

    if (dim == 0) {
        size_t slice_size = input_buffer.total_size / input_shape[0];
        size_t offset_bytes = index_value * slice_size * element_size;
        node.output_buffer.set_external(const_cast<char*>(input_data) + offset_bytes);

        if (input_buffer.precision == Precision::INT8) {
            node.output_buffer.quantization_scale = input_buffer.quantization_scale;
        }
        return;
    }
    
    std::vector<size_t> input_strides(input_shape.size());
    input_strides[input_shape.size() - 1] = 1;
    for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }
    
    size_t slice_size = input_strides[dim];
    size_t outer_size = input_buffer.total_size / input_strides[dim - 1];
    size_t dim_stride = input_strides[dim];
    size_t block_size = dim_stride * input_shape[dim];
    
    size_t output_idx = 0;
    for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        size_t input_base = outer_idx * block_size + index_value * dim_stride;
        
        std::memcpy(output_data + output_idx * element_size,
                    input_data + input_base * element_size,
                    slice_size * element_size);
        
        output_idx += slice_size;
    }
    
    if (input_buffer.precision == Precision::INT8) {
        node.output_buffer.quantization_scale = input_buffer.quantization_scale;
    }
}
