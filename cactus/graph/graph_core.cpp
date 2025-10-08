#include "graph.h"
#include "../kernel/kernel.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace Quantization {
    void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale) {
        cactus_int8_to_fp32(src, dst, count, scale);
    }
    
    void fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale) {
        cactus_fp32_to_int8(src, dst, count, scale);
    }
    
    void dynamic_quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count, 
                                       float* computed_scale) {
        cactus_dynamic_quantize_fp32_to_int8(src, dst, count, computed_scale);
    }
    
    void fp16_to_fp32(const __fp16* src, float* dst, size_t count) {
        cactus_fp16_to_fp32(src, dst, count);
    }
    
    void fp32_to_fp16(const float* src, __fp16* dst, size_t count) {
        cactus_fp32_to_fp16(src, dst, count);
    }
    
    void int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale) {
        cactus_int8_to_fp16(src, dst, count, scale);
    }
    
    void fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale) {
        cactus_fp16_to_int8(src, dst, count, scale);
    }
}

static std::vector<size_t> compute_strides(const std::vector<size_t>& shape, const std::vector<size_t>& target_shape) {
    std::vector<size_t> strides(target_shape.size());
    
    size_t shape_offset = target_shape.size() - shape.size();
    
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (i < shape_offset) {
            strides[i] = 0;
        } else {
            size_t dim_idx = i - shape_offset;
            if (shape[dim_idx] == 1) {
                strides[i] = 0;
            } else {
                strides[i] = 1;
                for (size_t j = dim_idx + 1; j < shape.size(); ++j) {
                    strides[i] *= shape[j];
                }
            }
        }
    }
    
    return strides;
}


template<typename T>
void dispatch_binary_op(OpType op, const T* lhs, const T* rhs, T* output, size_t count) {
    switch (op) {
        case OpType::ADD:
            if constexpr (std::is_same_v<T, int8_t>) {
                cactus_add_int8(lhs, rhs, output, count);
            } else if constexpr (std::is_same_v<T, __fp16>) {
                cactus_add_f16(lhs, rhs, output, count);
            } else {
                cactus_add_f32(lhs, rhs, output, count);
            }
            break;
        case OpType::ADD_CLIPPED:
            if constexpr (std::is_same_v<T, int8_t>) {
                cactus_add_int8(lhs, rhs, output, count);  
            } else if constexpr (std::is_same_v<T, __fp16>) {
                cactus_add_f16_clipped(lhs, rhs, output, count);
            } else {
                cactus_add_f32(lhs, rhs, output, count); 
            }
            break;
        case OpType::SUBTRACT:
            if constexpr (std::is_same_v<T, int8_t>) {
                cactus_subtract_int8(lhs, rhs, output, count);
            } else if constexpr (std::is_same_v<T, __fp16>) {
                cactus_subtract_f16(lhs, rhs, output, count);
            } else {
                cactus_subtract_f32(lhs, rhs, output, count);
            }
            break;
        case OpType::MULTIPLY:
            if constexpr (std::is_same_v<T, int8_t>) {
                cactus_multiply_int8(lhs, rhs, output, count);
            } else if constexpr (std::is_same_v<T, __fp16>) {
                cactus_multiply_f16(lhs, rhs, output, count);
            } else {
                cactus_multiply_f32(lhs, rhs, output, count);
            }
            break;
        case OpType::DIVIDE:
            if constexpr (std::is_same_v<T, int8_t>) {
                cactus_divide_int8(lhs, rhs, output, count);
            } else if constexpr (std::is_same_v<T, __fp16>) {
                cactus_divide_f16(lhs, rhs, output, count);
            } else {
                cactus_divide_f32(lhs, rhs, output, count);
            }
            break;
        default:
            break;
    }
}

template void dispatch_binary_op<int8_t>(OpType, const int8_t*, const int8_t*, int8_t*, size_t);
template void dispatch_binary_op<__fp16>(OpType, const __fp16*, const __fp16*, __fp16*, size_t);
template void dispatch_binary_op<float>(OpType, const float*, const float*, float*, size_t);

template<typename T>
void dispatch_unary_op(OpType op, const T* input, T* output, size_t count, float param) {
    ScalarOpType scalar_op;
    switch (op) {
        case OpType::SCALAR_ADD: scalar_op = ScalarOpType::ADD; break;
        case OpType::SCALAR_SUBTRACT: scalar_op = ScalarOpType::SUBTRACT; break;
        case OpType::SCALAR_MULTIPLY: scalar_op = ScalarOpType::MULTIPLY; break;
        case OpType::SCALAR_DIVIDE: scalar_op = ScalarOpType::DIVIDE; break;
        case OpType::SCALAR_EXP: scalar_op = ScalarOpType::EXP; break;
        case OpType::SCALAR_SQRT: scalar_op = ScalarOpType::SQRT; break;
        case OpType::SCALAR_COS: scalar_op = ScalarOpType::COS; break;
        case OpType::SCALAR_SIN: scalar_op = ScalarOpType::SIN; break;
        default: return;
    }
    
    if constexpr (std::is_same_v<T, int8_t>) {
        cactus_scalar_op_int8(input, output, count, param, scalar_op);
    } else if constexpr (std::is_same_v<T, __fp16>) {
        cactus_scalar_op_f16(input, output, count, param, scalar_op);
    } else {
        cactus_scalar_op_f32(input, output, count, param, scalar_op);
    }
}

template void dispatch_unary_op<int8_t>(OpType, const int8_t*, int8_t*, size_t, float);
template void dispatch_unary_op<__fp16>(OpType, const __fp16*, __fp16*, size_t, float);
template void dispatch_unary_op<float>(OpType, const float*, float*, size_t, float);

void compute_node_optimized(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    switch (node.op_type) {
        case OpType::INPUT:
            break;
        case OpType::ADD:
        case OpType::ADD_CLIPPED:
        case OpType::SUBTRACT:
        case OpType::MULTIPLY:
        case OpType::DIVIDE: {
            const auto& lhs = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& rhs = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            if (node.params.broadcast_info.needs_broadcasting) {
                std::vector<size_t> lhs_strides = compute_strides(lhs.shape, node.params.broadcast_info.output_shape);
                std::vector<size_t> rhs_strides = compute_strides(rhs.shape, node.params.broadcast_info.output_shape);
                
                if (lhs.precision == Precision::INT8) {
                    switch (node.op_type) {
                        case OpType::ADD:
                        case OpType::ADD_CLIPPED:  
                            cactus_add_broadcast_int8(lhs.data_as<int8_t>(), rhs.data_as<int8_t>(),
                                                     node.output_buffer.data_as<int8_t>(),
                                                     lhs_strides.data(), rhs_strides.data(),
                                                     node.params.broadcast_info.output_shape.data(),
                                                     node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::SUBTRACT:
                            cactus_subtract_broadcast_int8(lhs.data_as<int8_t>(), rhs.data_as<int8_t>(), 
                                                          node.output_buffer.data_as<int8_t>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::MULTIPLY:
                            cactus_multiply_broadcast_int8(lhs.data_as<int8_t>(), rhs.data_as<int8_t>(), 
                                                          node.output_buffer.data_as<int8_t>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::DIVIDE:
                            cactus_divide_broadcast_int8(lhs.data_as<int8_t>(), rhs.data_as<int8_t>(), 
                                                        node.output_buffer.data_as<int8_t>(),
                                                        lhs_strides.data(), rhs_strides.data(),
                                                        node.params.broadcast_info.output_shape.data(),
                                                        node.params.broadcast_info.output_shape.size());
                            break;
                        default: break;
                    }
                } else if (lhs.precision == Precision::FP16) {
                    switch (node.op_type) {
                        case OpType::ADD:
                        case OpType::ADD_CLIPPED:  
                            cactus_add_broadcast_f16(lhs.data_as<__fp16>(), rhs.data_as<__fp16>(),
                                                     node.output_buffer.data_as<__fp16>(),
                                                     lhs_strides.data(), rhs_strides.data(),
                                                     node.params.broadcast_info.output_shape.data(),
                                                     node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::SUBTRACT:
                            cactus_subtract_broadcast_f16(lhs.data_as<__fp16>(), rhs.data_as<__fp16>(), 
                                                          node.output_buffer.data_as<__fp16>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::MULTIPLY:
                            cactus_multiply_broadcast_f16(lhs.data_as<__fp16>(), rhs.data_as<__fp16>(), 
                                                          node.output_buffer.data_as<__fp16>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::DIVIDE:
                            cactus_divide_broadcast_f16(lhs.data_as<__fp16>(), rhs.data_as<__fp16>(), 
                                                        node.output_buffer.data_as<__fp16>(),
                                                        lhs_strides.data(), rhs_strides.data(),
                                                        node.params.broadcast_info.output_shape.data(),
                                                        node.params.broadcast_info.output_shape.size());
                            break;
                        default: break;
                    }
                } else {
                    switch (node.op_type) {
                        case OpType::ADD:
                        case OpType::ADD_CLIPPED: 
                            cactus_add_broadcast_f32(lhs.data_as<float>(), rhs.data_as<float>(),
                                                     node.output_buffer.data_as<float>(),
                                                     lhs_strides.data(), rhs_strides.data(),
                                                     node.params.broadcast_info.output_shape.data(),
                                                     node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::SUBTRACT:
                            cactus_subtract_broadcast_f32(lhs.data_as<float>(), rhs.data_as<float>(), 
                                                          node.output_buffer.data_as<float>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::MULTIPLY:
                            cactus_multiply_broadcast_f32(lhs.data_as<float>(), rhs.data_as<float>(), 
                                                          node.output_buffer.data_as<float>(),
                                                          lhs_strides.data(), rhs_strides.data(),
                                                          node.params.broadcast_info.output_shape.data(),
                                                          node.params.broadcast_info.output_shape.size());
                            break;
                        case OpType::DIVIDE:
                            cactus_divide_broadcast_f32(lhs.data_as<float>(), rhs.data_as<float>(), 
                                                        node.output_buffer.data_as<float>(),
                                                        lhs_strides.data(), rhs_strides.data(),
                                                        node.params.broadcast_info.output_shape.data(),
                                                        node.params.broadcast_info.output_shape.size());
                            break;
                        default: break;
                    }
                }
            } else {
                if (lhs.precision == Precision::INT8) {
                    dispatch_binary_op<int8_t>(node.op_type, lhs.data_as<int8_t>(), 
                                              rhs.data_as<int8_t>(), node.output_buffer.data_as<int8_t>(), 
                                              node.output_buffer.total_size);
                } else if (lhs.precision == Precision::FP16) {
                    dispatch_binary_op<__fp16>(node.op_type, lhs.data_as<__fp16>(), 
                                              rhs.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(), 
                                              node.output_buffer.total_size);
                } else {
                    dispatch_binary_op<float>(node.op_type, lhs.data_as<float>(), 
                                             rhs.data_as<float>(), node.output_buffer.data_as<float>(), 
                                             node.output_buffer.total_size);
                }
            }
            break;
        }
        case OpType::SCALAR_ADD:
        case OpType::SCALAR_SUBTRACT:
        case OpType::SCALAR_MULTIPLY:
        case OpType::SCALAR_DIVIDE:
        case OpType::SCALAR_EXP:
        case OpType::SCALAR_SQRT:
        case OpType::SCALAR_COS:
        case OpType::SCALAR_SIN: {
            const auto& input = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            
            if (input.precision == Precision::INT8) {
                dispatch_unary_op<int8_t>(node.op_type, input.data_as<int8_t>(),
                                         node.output_buffer.data_as<int8_t>(),
                                         node.output_buffer.total_size, node.params.scalar);
            } else if (input.precision == Precision::FP16) {
                dispatch_unary_op<__fp16>(node.op_type, input.data_as<__fp16>(),
                                        node.output_buffer.data_as<__fp16>(),
                                        node.output_buffer.total_size, node.params.scalar);
            } else {
                dispatch_unary_op<float>(node.op_type, input.data_as<float>(),
                                        node.output_buffer.data_as<float>(),
                                        node.output_buffer.total_size, node.params.scalar);
            }
            break;
        }
        case OpType::SILU: {
            const auto& input = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            
            if (input.precision == Precision::INT8) {
                cactus_silu_int8(input.data_as<int8_t>(), 
                                node.output_buffer.data_as<int8_t>(), 
                                node.output_buffer.total_size,
                                input.quantization_scale,
                                node.output_buffer.quantization_scale);
            } else if (input.precision == Precision::FP16) {
                cactus_silu_f16(input.data_as<__fp16>(), 
                               node.output_buffer.data_as<__fp16>(), 
                               node.output_buffer.total_size);
            } else {
                cactus_silu_f32(input.data_as<float>(), 
                               node.output_buffer.data_as<float>(), 
                               node.output_buffer.total_size);
            }
            break;
        }
        case OpType::GELU: {
            const auto& input = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

            if (input.precision == Precision::INT8) {
                cactus_gelu_int8(input.data_as<int8_t>(),
                                node.output_buffer.data_as<int8_t>(),
                                node.output_buffer.total_size,
                                input.quantization_scale,
                                node.output_buffer.quantization_scale);
            } else if (input.precision == Precision::FP16) {
                cactus_gelu_f16(input.data_as<__fp16>(),
                               node.output_buffer.data_as<__fp16>(),
                               node.output_buffer.total_size);
            } else {
                cactus_gelu_f32(input.data_as<float>(),
                               node.output_buffer.data_as<float>(),
                               node.output_buffer.total_size);
            }
            break;
        }
        case OpType::MATMUL:
            compute_matmul_node(node, nodes, node_index_map);
            break;
        case OpType::TRANSPOSE:
            compute_transpose_node(node, nodes, node_index_map);
            break;
        case OpType::SUM:
        case OpType::MEAN:
        case OpType::VARIANCE:
        case OpType::MIN:
        case OpType::MAX:
            compute_reduce_node(node, nodes, node_index_map);
            break;
        case OpType::RMS_NORM:
        case OpType::ROPE:
        case OpType::SOFTMAX:
        case OpType::ATTENTION:
        case OpType::GATHER:
        case OpType::EMBEDDING:
            compute_fused_node(node, nodes, node_index_map);
            break;
        case OpType::SAMPLE:
            compute_sample_node(node, nodes, node_index_map);
            break;
        case OpType::CONCAT:
            compute_fused_node(node, nodes, node_index_map);
            break;
        case OpType::PRECISION_CAST:
            compute_precision_cast_node(node, nodes, node_index_map);
            break;
        case OpType::RESHAPE:
            compute_reshape_node(node, nodes, node_index_map);
            break;
    }
}


void CactusGraph::allocate_buffers() {
    for (auto& node : nodes_) {
        if (node->op_type != OpType::INPUT) {
            node->output_buffer.allocate();
        }
    }
}



namespace ValidationUtils {
    void validate_tensor_dims(const std::vector<size_t>& shape, size_t required_dims, const std::string& op_name) {
        if (shape.size() != required_dims) {
            throw std::runtime_error(op_name + " requires " + std::to_string(required_dims) + 
                                    "D tensor, got " + std::to_string(shape.size()) + "D tensor");
        }
    }
    
    void validate_precision(Precision actual, Precision required, const std::string& op_name) {
        if (actual != required) {
            std::string actual_str = (actual == Precision::INT8) ? "INT8" : "FP32";
            std::string required_str = (required == Precision::INT8) ? "INT8" : "FP32";
            throw std::runtime_error(op_name + " requires " + required_str + " precision, got " + actual_str);
        }
    }
    
    void validate_input_count(size_t actual, size_t required, const std::string& op_name) {
        if (actual < required) {
            throw std::runtime_error(op_name + " requires " + std::to_string(required) + 
                                    " inputs, got " + std::to_string(actual) + " inputs");
        }
    }
}

void compute_reshape_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    
    size_t input_total_elements = input_buffer.total_size;
    size_t output_total_elements = node.output_buffer.total_size;
    
    if (input_total_elements != output_total_elements) {
        throw std::runtime_error("Reshape operation: input elements (" + std::to_string(input_total_elements) + 
                                ") must match output elements (" + std::to_string(output_total_elements) + ")");
    }
    
    std::memcpy(node.output_buffer.get_data(), input_buffer.get_data(), input_buffer.byte_size);
}

void compute_precision_cast_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_node = *nodes[node_index_map.at(node.input_ids[0])];
    
    if (input_node.output_buffer.precision == node.output_buffer.precision) {
        std::memcpy(node.output_buffer.get_data(), input_node.output_buffer.get_data(), input_node.output_buffer.byte_size);
        return;
    }
    
    size_t count = input_node.output_buffer.total_size;
    
    if (input_node.output_buffer.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP32) {
        Quantization::int8_to_fp32(input_node.output_buffer.data_as<int8_t>(), node.output_buffer.data_as<float>(), count);
    } else if (input_node.output_buffer.precision == Precision::FP32 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp32_to_int8(input_node.output_buffer.data_as<float>(), node.output_buffer.data_as<int8_t>(), count);
    } else if (input_node.output_buffer.precision == Precision::FP16 && node.output_buffer.precision == Precision::FP32) {
        Quantization::fp16_to_fp32(input_node.output_buffer.data_as<__fp16>(), node.output_buffer.data_as<float>(), count);
    } else if (input_node.output_buffer.precision == Precision::FP32 && node.output_buffer.precision == Precision::FP16) {
        Quantization::fp32_to_fp16(input_node.output_buffer.data_as<float>(), node.output_buffer.data_as<__fp16>(), count);
    } else if (input_node.output_buffer.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP16) {
        Quantization::int8_to_fp16(input_node.output_buffer.data_as<int8_t>(), node.output_buffer.data_as<__fp16>(), count);
    } else if (input_node.output_buffer.precision == Precision::FP16 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp16_to_int8(input_node.output_buffer.data_as<__fp16>(), node.output_buffer.data_as<int8_t>(), count);
    } else {
        throw std::runtime_error("Unsupported precision conversion from " + 
                                std::to_string(static_cast<int>(input_node.output_buffer.precision)) + 
                                " to " + std::to_string(static_cast<int>(node.output_buffer.precision)));
    }
}


TensorConfig& TensorConfig::global() {
    static TensorConfig instance;
    return instance;
}

BufferDesc::BufferDesc() 
    : total_size(0), byte_size(0), external_data(nullptr), 
      precision(Precision::INT8), quantization_scale(1.0f) {}

BufferDesc::BufferDesc(const std::vector<size_t>& s, Precision prec, float scale)
    : shape(s), external_data(nullptr), precision(prec), 
      quantization_scale(scale) {
    total_size = 1;
    for (size_t dim : shape) total_size *= dim;
    byte_size = total_size * PrecisionTraits::size_of(prec);
}

void* BufferDesc::get_data() { 
    return external_data ? external_data : data.get(); 
}

const void* BufferDesc::get_data() const { 
    return external_data ? external_data : data.get(); 
}

void BufferDesc::allocate() {
    if (!data && !external_data) {
        data = std::make_unique<char[]>(byte_size);
    }
}

void BufferDesc::set_external(void* ptr) {
    external_data = ptr;
    data.reset();
}

GraphNode::GraphNode(size_t node_id, OpType type) : id(node_id), op_type(type) {}

CactusGraph::CactusGraph() : next_node_id_(0) {}

size_t CactusGraph::get_node_count() const { 
    return nodes_.size(); 
}

 