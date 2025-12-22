#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <assert.h>

namespace {
    thread_local std::vector<int8_t> transpose_buffer_int8;
    thread_local std::vector<__fp16> transpose_buffer_fp16;
    thread_local std::vector<float> transpose_buffer_fp32;
    thread_local std::vector<int8_t> quantization_buffer_int8;
    
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

void shrink_thread_local_buffers() {
    std::vector<int8_t>().swap(transpose_buffer_int8);
    std::vector<__fp16>().swap(transpose_buffer_fp16);
    std::vector<float>().swap(transpose_buffer_fp32);
    std::vector<int8_t>().swap(quantization_buffer_int8);
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
                    double result = cactus_sum_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                    node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
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
        case OpType::SLICE: {
            auto* input_node = nodes[node_index_map.at(node.input_ids[0])].get();
            auto& input_buffer = input_node->output_buffer;

            const size_t axis_index = static_cast<size_t>(node.params.axis);

            const size_t axis_size = input_buffer.shape[axis_index];
            const size_t slice_start = node.params.slice_start;
            size_t slice_length = node.params.slice_length;

            if (slice_length == 0) {
                slice_length = axis_size - slice_start;
            }

            const size_t element_size = PrecisionTraits::size_of(input_buffer.precision);

            if (axis_index == 0) {
                size_t inner_elements = 1;
                for (size_t i = 1; i < input_buffer.shape.size(); ++i) {
                    inner_elements *= input_buffer.shape[i];
                }

                auto* base_ptr = static_cast<char*>(input_buffer.get_data());
                if (!base_ptr) {
                    throw std::runtime_error("Slice input buffer is not available");
                }

                const size_t byte_offset = slice_start * inner_elements * element_size;

                node.output_buffer.set_external(base_ptr + byte_offset);
                node.output_buffer.precision = input_buffer.precision;
                node.output_buffer.quantization_scale = input_buffer.quantization_scale;
                break;
            }

            const char* input_ptr = static_cast<const char*>(input_buffer.get_data());
            if (!input_ptr) {
                throw std::runtime_error("Slice input buffer is not available");
            }

            size_t inner_elements = 1;
            for (size_t i = axis_index + 1; i < input_buffer.shape.size(); ++i) {
                inner_elements *= input_buffer.shape[i];
            }

            size_t outer_elements = 1;
            for (size_t i = 0; i < axis_index; ++i) {
                outer_elements *= input_buffer.shape[i];
            }

            const size_t copy_block_elements = slice_length * inner_elements;
            const size_t axis_stride_elements = axis_size * inner_elements;
            const size_t copy_block_bytes = copy_block_elements * element_size;
            const size_t axis_stride_bytes = axis_stride_elements * element_size;

            node.output_buffer.external_data = nullptr;
            node.output_buffer.allocate();
            node.output_buffer.precision = input_buffer.precision;
            node.output_buffer.quantization_scale = input_buffer.quantization_scale;

            auto* output_ptr = static_cast<char*>(node.output_buffer.get_data());
            if (!output_ptr) {
                throw std::runtime_error("Slice output buffer could not be allocated");
            }

            for (size_t outer = 0; outer < outer_elements; ++outer) {
                const char* src = input_ptr + outer * axis_stride_bytes + slice_start * inner_elements * element_size;
                char* dst = output_ptr + outer * copy_block_bytes;
                std::memcpy(dst, src, copy_block_bytes);
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
        case OpType::BILINEAR_INTERPOLATION: {
            const auto& pos_embeds_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            
            int total_pos_embeds = static_cast<int>(pos_embeds_buffer.shape[0]);
            int embed_dim = static_cast<int>(pos_embeds_buffer.shape[1]);
            
            int src_height = static_cast<int>(std::sqrt(total_pos_embeds));
            int src_width = src_height;
            
            int dst_height = static_cast<int>(node.params.dst_height);
            int dst_width = static_cast<int>(node.params.dst_width);

            std::vector<float> pos_embed_fp32(total_pos_embeds * embed_dim);
            
            if (pos_embeds_buffer.precision == Precision::INT8) {
                const int8_t* pos_embeds_data = pos_embeds_buffer.data_as<int8_t>();
                for (int i = 0; i < total_pos_embeds * embed_dim; ++i) {
                    pos_embed_fp32[i] = static_cast<float>(pos_embeds_data[i]) * pos_embeds_buffer.quantization_scale;
                }
            }
            else if (pos_embeds_buffer.precision == Precision::FP16) {
                const __fp16* pos_embed_fp16 = pos_embeds_buffer.data_as<__fp16>();
                for (int i = 0; i < total_pos_embeds * embed_dim; ++i) {
                    pos_embed_fp32[i] = static_cast<float>(pos_embed_fp16[i]);
                }
            }
            else {
                const float* pos_embed_data = pos_embeds_buffer.data_as<float>();
                std::memcpy(pos_embed_fp32.data(), pos_embed_data, total_pos_embeds * embed_dim * sizeof(float));
            }
            
            
            float* output = node.output_buffer.data_as<float>();
            cactus_bilinear_interpolation_fp32(pos_embed_fp32.data(), output, 
                                              src_height, src_width, embed_dim,
                                              dst_height, dst_width);
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
        case OpType::CONV1D_CAUSAL: {
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
                const float w_s   = W.quantization_scale;
                const size_t W_size = W0 * W1 * K;
                const int8_t* W_int8 = W.data_as<int8_t>();

                std::vector<__fp16> W_fp16(W_size);
                for (size_t i = 0; i < W_size; ++i) {
                    W_fp16[i] = static_cast<__fp16>(W_int8[i] * w_s);
                }

                cactus_conv1d_causal_depthwise_f16(
                    X.data_as<__fp16>(), W_fp16.data(), Y.data_as<__fp16>(),
                    N, L, C_in, K, dil);
            } else if (W.precision == Precision::FP16) {
                cactus_conv1d_causal_depthwise_f16(
                    X.data_as<__fp16>(), W.data_as<__fp16>(), Y.data_as<__fp16>(),
                    N, L, C_in, K, dil);
            } else if (W.precision == Precision::FP32) {
                cactus_conv1d_causal_depthwise_f32(
                    X.data_as<float>(), W.data_as<float>(), Y.data_as<float>(),
                    N, L, C_in, K, dil);
            } else {
                throw std::runtime_error("Depthwise causal conv supports INT8/FP16/FP32");
            }
            break;
        }

        case OpType::CONV1D_K3: {

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

                if (W.precision == Precision::INT8) {
                    if (X.precision != Precision::FP16) {
                        throw std::runtime_error(
                            "Conv1d_k3 with INT8 weights currently requires FP16 activations");
                    }

                    const float w_s = W.quantization_scale;
                    const size_t W_size = C_out * C_in * K;
                    const int8_t* W_int8 = W.data_as<int8_t>();

                    std::vector<__fp16> W_fp16(W_size);
                    for (size_t i = 0; i < W_size; ++i) {
                        W_fp16[i] = static_cast<__fp16>(W_int8[i] * w_s);
                    }

                    cactus_conv1d_f16_k3(
                        X.data_as<__fp16>(),
                        W_fp16.data(),
                        Y.data_as<__fp16>(),
                        N, L, C_in, C_out, stride
                    );
                }
                else if (X.precision == Precision::FP32) {
                    cactus_conv1d_f32_k3(
                        X.data_as<float>(),
                        W.data_as<float>(),
                        Y.data_as<float>(),
                        N, L, C_in, C_out, stride
                    );
                }
                else if (X.precision == Precision::FP16) {
                    cactus_conv1d_f16_k3(
                        X.data_as<__fp16>(),
                        W.data_as<__fp16>(),
                        Y.data_as<__fp16>(),
                        N, L, C_in, C_out, stride
                    );
                }
                else {
                    throw std::runtime_error("Conv1d_k3 only supports FP32, FP16, and INT8 weights with FP16 activations!");
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
            const __fp16* input = input_buffer.data_as<__fp16>();
            __fp16* output = node.output_buffer.data_as<__fp16>();
            cactus_transpose_f16(input, output, input_buffer.shape.data(), permutation.data(), permutation.size(), 0, input_buffer.total_size);
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

                    cactus_transpose_2d_f16(rhs, transpose_buffer_fp16.data(),
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

    const float* bias_values = node.params.bias_values.empty() ? nullptr : node.params.bias_values.data();
    const uint32_t* bias_indices = node.params.bias_indices.empty() ? nullptr : node.params.bias_indices.data();
    size_t bias_count = node.params.bias_values.size();

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
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count);
    } else if (logits_buffer.precision == Precision::FP16) {
        const __fp16* logits_fp16 = logits_buffer.data_as<__fp16>();
        cactus_sample_f16(logits_fp16 + last_token_offset, node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count);
    } else {
        const float* logits_fp32 = logits_buffer.data_as<float>();
        cactus_sample_f32(logits_fp32 + last_token_offset, node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count);
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
    std::fill(output, output + num_classes * batch_size, 0.0f);

    if (indices_buffer.precision != Precision::FP32 || values_buffer.precision != Precision::FP32) {
        throw std::runtime_error("ScatterTopK currently expects FP32 inputs");
    }

    const float* indices_data = indices_buffer.data_as<float>();
    const float* values_data = values_buffer.data_as<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t k = 0; k < top_k; ++k) {
            float raw_index = indices_data[b * top_k + k];
            if (!std::isfinite(raw_index)) {
                throw std::runtime_error("ScatterTopK index is not finite");
            }
            size_t expert_index = static_cast<size_t>(raw_index + 0.5f);
            if (expert_index >= num_classes) {
                throw std::runtime_error("ScatterTopK index out of range");
            }
            float weight = values_data[b * top_k + k];
            output[expert_index * batch_size + b] = weight;
        }
    }
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

void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& bias_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    float epsilon = node.params.epsilon;
    
    if (input_buffer.shape.empty()) {
        throw std::runtime_error("LayerNorm requires non-empty input tensor");
    }
    
    size_t feature_size = input_buffer.shape.back();
    size_t batch_size = input_buffer.total_size / feature_size;
    
    std::vector<float> input_float(input_buffer.total_size);
    std::vector<float> weight_float(feature_size);
    std::vector<float> bias_float(feature_size);
    
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
