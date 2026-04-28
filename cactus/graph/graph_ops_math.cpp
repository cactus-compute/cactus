#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <stdexcept>

namespace Quantization {
    void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale) {
        cactus_int8_to_fp32(src, dst, count, scale);
    }

    void fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale) {
        cactus_fp32_to_int8(src, dst, count, scale);
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

void dispatch_binary_op_f16(OpType op, const __fp16* lhs, const __fp16* rhs, __fp16* output, size_t count) {
    switch (op) {
        case OpType::ADD:
            cactus_add_f16(lhs, rhs, output, count);
            break;
        case OpType::ADD_CLIPPED:
            cactus_add_f16_clipped(lhs, rhs, output, count);
            break;
        case OpType::SUBTRACT:
            cactus_subtract_f16(lhs, rhs, output, count);
            break;
        case OpType::MULTIPLY:
            cactus_multiply_f16(lhs, rhs, output, count);
            break;
        case OpType::DIVIDE:
            cactus_divide_f16(lhs, rhs, output, count);
            break;
        default:
            break;
    }
}

void dispatch_unary_op_f16(OpType op, const __fp16* input, __fp16* output, size_t count, float param) {
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
        case OpType::SCALAR_LOG: scalar_op = ScalarOpType::LOG; break;
        case OpType::ABS: scalar_op = ScalarOpType::ABS; break;
        case OpType::POW: scalar_op = ScalarOpType::POW; break;
        default: return;
    }

    cactus_scalar_op_f16(input, output, count, param, scalar_op);
}

void compute_binary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs = get_input(node, 0, nodes, node_index_map);
    const auto& rhs = get_input(node, 1, nodes, node_index_map);

    if (lhs.precision != Precision::FP16) {
        throw std::runtime_error("Binary operations only support FP16 precision (got " + std::to_string(static_cast<int>(lhs.precision)) + ")");
    }

    if (node.params.broadcast_info.needs_broadcasting) {
        std::vector<size_t> lhs_strides = compute_strides(lhs.shape, node.params.broadcast_info.output_shape);
        std::vector<size_t> rhs_strides = compute_strides(rhs.shape, node.params.broadcast_info.output_shape);

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
        dispatch_binary_op_f16(node.op_type, lhs.data_as<__fp16>(),
                               rhs.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                               node.output_buffer.total_size);
    }
}

void compute_unary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);

    if (input.precision != Precision::FP16) {
        throw std::runtime_error("Scalar operations only support FP16 precision");
    }

    dispatch_unary_op_f16(node.op_type, input.data_as<__fp16>(),
                          node.output_buffer.data_as<__fp16>(),
                          node.output_buffer.total_size, node.params.scalar);
}

void compute_activation_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);

    if (input.precision != Precision::FP16) {
        throw std::runtime_error("Activation operations only support FP16 precision");
    }

    switch (node.op_type) {
        case OpType::RELU:
            cactus_relu_f16(input.data_as<__fp16>(),
                            node.output_buffer.data_as<__fp16>(),
                            node.output_buffer.total_size);
            break;
        case OpType::SILU:
            cactus_silu_f16(input.data_as<__fp16>(),
                           node.output_buffer.data_as<__fp16>(),
                           node.output_buffer.total_size);
            break;
        case OpType::GELU:
            cactus_gelu_f16(input.data_as<__fp16>(),
                           node.output_buffer.data_as<__fp16>(),
                           node.output_buffer.total_size);
            break;
        case OpType::GELU_ERF:
            cactus_gelu_f16_erf(input.data_as<__fp16>(),
                                node.output_buffer.data_as<__fp16>(),
                                node.output_buffer.total_size);
            break;
        case OpType::SIGMOID:
            cactus_sigmoid_f16(input.data_as<__fp16>(),
                            node.output_buffer.data_as<__fp16>(),
                            node.output_buffer.total_size);
            break;
        case OpType::TANH:
            cactus_tanh_f16(input.data_as<__fp16>(),
                            node.output_buffer.data_as<__fp16>(),
                            node.output_buffer.total_size);
            break;
        case OpType::LEAKY_RELU:
            cactus_leaky_relu_f16(input.data_as<__fp16>(),
                                  node.output_buffer.data_as<__fp16>(),
                                  node.output_buffer.total_size, node.params.scalar);
            break;
        default:
            break;
    }
}

void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    int axis = node.params.axis;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Reduction operations only support FP16 precision");
    }

    if (axis == -1) {
        switch (node.op_type) {
            case OpType::SUM: {
                double result = cactus_sum_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::MEAN: {
                double result = cactus_mean_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::VARIANCE: {
                double result = cactus_variance_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::MIN: {
                __fp16 result = cactus_min_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = result;
                break;
            }
            case OpType::MAX: {
                __fp16 result = cactus_max_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = result;
                break;
            }
            default: break;
        }
    } else {
        auto dims = AxisDims::from_shape(input_buffer.shape, static_cast<size_t>(axis));

        switch (node.op_type) {
            case OpType::SUM:
                cactus_sum_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    dims.outer, dims.axis_size, dims.inner);
                break;
            case OpType::MEAN:
                cactus_mean_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    dims.outer, dims.axis_size, dims.inner);
                break;
            case OpType::VARIANCE:
                cactus_variance_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                         dims.outer, dims.axis_size, dims.inner);
                break;
            case OpType::MIN:
                cactus_min_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    dims.outer, dims.axis_size, dims.inner);
                break;
            case OpType::MAX:
                cactus_max_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    dims.outer, dims.axis_size, dims.inner);
                break;
            default: break;
        }
    }
}

void compute_reshape_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);

    size_t input_total_elements = input_buffer.total_size;
    size_t output_total_elements = node.output_buffer.total_size;

    if (input_total_elements != output_total_elements) {
        throw std::runtime_error("Reshape operation: input elements (" + std::to_string(input_total_elements) +
                                ") must match output elements (" + std::to_string(output_total_elements) + ")");
    }

    std::memcpy(node.output_buffer.get_data(), input_buffer.get_data(), input_buffer.byte_size);
}

template<typename OutT>
static void dequant_grouped_int8(const int8_t* src, OutT* dst, const __fp16* scales,
                                 const std::vector<size_t>& shape, size_t group_size) {
    if (shape.size() == 2) {
        size_t N = shape[0], K = shape[1];
        size_t num_groups = K / group_size;
        for (size_t row = 0; row < N; ++row) {
            for (size_t col = 0; col < K; ++col) {
                size_t idx = row * K + col;
                float scale = static_cast<float>(scales[row * num_groups + col / group_size]);
                dst[idx] = static_cast<OutT>(static_cast<float>(src[idx]) * scale);
            }
        }
    } else if (shape.size() == 1) {
        size_t K = shape[0];
        for (size_t col = 0; col < K; ++col) {
            float scale = static_cast<float>(scales[col / group_size]);
            dst[col] = static_cast<OutT>(static_cast<float>(src[col]) * scale);
        }
    }
}

void compute_precision_cast_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buf = get_input(node, 0, nodes, node_index_map);

    if (input_buf.precision == node.output_buffer.precision) {
        std::memcpy(node.output_buffer.get_data(), input_buf.get_data(), input_buf.byte_size);
        return;
    }

    size_t count = input_buf.total_size;

    if (input_buf.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP32) {
        if (input_buf.is_grouped_int8()) {
            dequant_grouped_int8<float>(input_buf.data_as<int8_t>(), node.output_buffer.data_as<float>(),
                                        input_buf.scales_as_fp16(), input_buf.shape, input_buf.group_size);
        } else {
            Quantization::int8_to_fp32(input_buf.data_as<int8_t>(), node.output_buffer.data_as<float>(), count, 1.0f);
        }
    } else if (input_buf.precision == Precision::FP32 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp32_to_int8(input_buf.data_as<float>(), node.output_buffer.data_as<int8_t>(), count, 1.0f);
    } else if (input_buf.precision == Precision::FP16 && node.output_buffer.precision == Precision::FP32) {
        Quantization::fp16_to_fp32(input_buf.data_as<__fp16>(), node.output_buffer.data_as<float>(), count);
    } else if (input_buf.precision == Precision::FP32 && node.output_buffer.precision == Precision::FP16) {
        Quantization::fp32_to_fp16(input_buf.data_as<float>(), node.output_buffer.data_as<__fp16>(), count);
    } else if (input_buf.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP16) {
        if (input_buf.is_grouped_int8()) {
            dequant_grouped_int8<__fp16>(input_buf.data_as<int8_t>(), node.output_buffer.data_as<__fp16>(),
                                         input_buf.scales_as_fp16(), input_buf.shape, input_buf.group_size);
        } else {
            Quantization::int8_to_fp16(input_buf.data_as<int8_t>(), node.output_buffer.data_as<__fp16>(), count, 1.0f);
        }
    } else if (input_buf.precision == Precision::FP16 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp16_to_int8(input_buf.data_as<__fp16>(), node.output_buffer.data_as<int8_t>(), count, 1.0f);
    } else {
        throw std::runtime_error("Unsupported precision conversion from " +
                                std::to_string(static_cast<int>(input_buf.precision)) +
                                " to " + std::to_string(static_cast<int>(node.output_buffer.precision)));
    }
}
