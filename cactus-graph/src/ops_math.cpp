#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <cmath>
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

static size_t broadcast_linear_index(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t index = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

static void increment_broadcast_coords(std::vector<size_t>& coords, const std::vector<size_t>& shape) {
    for (int axis = static_cast<int>(coords.size()) - 1; axis >= 0; --axis) {
        size_t idx = static_cast<size_t>(axis);
        coords[idx]++;
        if (coords[idx] < shape[idx]) {
            break;
        }
        coords[idx] = 0;
    }
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
        case OpType::NOT_EQUAL:
            CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        output[i] = static_cast<__fp16>(
                            static_cast<float>(lhs[i]) != static_cast<float>(rhs[i]) ? 1.0f : 0.0f);
                    }
                });
            break;
        default:
            break;
    }
}

static float apply_binary_op_f32(OpType op, float lhs, float rhs) {
    switch (op) {
        case OpType::ADD:
        case OpType::ADD_CLIPPED:
            return lhs + rhs;
        case OpType::SUBTRACT:
            return lhs - rhs;
        case OpType::MULTIPLY:
            return lhs * rhs;
        case OpType::DIVIDE:
            return lhs / rhs;
        case OpType::NOT_EQUAL:
            return lhs != rhs ? 1.0f : 0.0f;
        default:
            return lhs;
    }
}

void dispatch_binary_op_f32(OpType op, const float* lhs, const float* rhs, float* output, size_t count) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                output[i] = apply_binary_op_f32(op, lhs[i], rhs[i]);
            }
        });
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

static float apply_scalar_op_f32(OpType op, float value, float param) {
    switch (op) {
        case OpType::SCALAR_ADD: return value + param;
        case OpType::SCALAR_SUBTRACT: return value - param;
        case OpType::SCALAR_MULTIPLY: return value * param;
        case OpType::SCALAR_DIVIDE: return value / param;
        case OpType::SCALAR_EXP: return std::exp(value);
        case OpType::SCALAR_SQRT: return std::sqrt(value);
        case OpType::SCALAR_COS: return std::cos(value);
        case OpType::SCALAR_SIN: return std::sin(value);
        case OpType::SCALAR_LOG: return std::log(value);
        case OpType::ABS: return std::fabs(value);
        case OpType::POW: return std::pow(value, param);
        default: return value;
    }
}

void dispatch_unary_op_f32(OpType op, const float* input, float* output, size_t count, float param) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                output[i] = apply_scalar_op_f32(op, input[i], param);
            }
        });
}

void compute_binary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs = get_input(node, 0, nodes, node_index_map);
    const auto& rhs = get_input(node, 1, nodes, node_index_map);

    if (lhs.precision != rhs.precision) {
        throw std::runtime_error("Binary operations require matching precision");
    }
    if (lhs.precision != Precision::FP16 && lhs.precision != Precision::FP32) {
        throw std::runtime_error("Binary operations only support FP16/FP32 precision");
    }

    if (node.params.broadcast_info.needs_broadcasting) {
        std::vector<size_t> lhs_strides = compute_strides(lhs.shape, node.params.broadcast_info.output_shape);
        std::vector<size_t> rhs_strides = compute_strides(rhs.shape, node.params.broadcast_info.output_shape);
        const auto& output_shape = node.params.broadcast_info.output_shape;
        size_t total_elements = 1;
        for (size_t dim : output_shape) {
            total_elements *= dim;
        }
        if (lhs.precision == Precision::FP32) {
            const float* lhs_data = lhs.data_as<float>();
            const float* rhs_data = rhs.data_as<float>();
            float* out_data = node.output_buffer.data_as<float>();
            CactusThreading::parallel_for(total_elements, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    std::vector<size_t> coords(output_shape.size());
                    size_t tmp = start_idx;
                    for (int axis = static_cast<int>(output_shape.size()) - 1; axis >= 0; --axis) {
                        coords[static_cast<size_t>(axis)] = tmp % output_shape[static_cast<size_t>(axis)];
                        tmp /= output_shape[static_cast<size_t>(axis)];
                    }
                    for (size_t linear_idx = start_idx; linear_idx < end_idx; ++linear_idx) {
                        size_t lhs_idx = broadcast_linear_index(coords, lhs_strides);
                        size_t rhs_idx = broadcast_linear_index(coords, rhs_strides);
                        out_data[linear_idx] = apply_binary_op_f32(node.op_type, lhs_data[lhs_idx], rhs_data[rhs_idx]);
                        increment_broadcast_coords(coords, output_shape);
                    }
                });
            return;
        }

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
            case OpType::NOT_EQUAL: {
                const __fp16* lhs_data = lhs.data_as<__fp16>();
                const __fp16* rhs_data = rhs.data_as<__fp16>();
                __fp16* out_data = node.output_buffer.data_as<__fp16>();
                CactusThreading::parallel_for(total_elements, CactusThreading::Thresholds::ELEMENT_WISE,
                    [&](size_t start_idx, size_t end_idx) {
                        std::vector<size_t> coords(output_shape.size());
                        size_t tmp = start_idx;
                        for (int axis = static_cast<int>(output_shape.size()) - 1; axis >= 0; --axis) {
                            coords[static_cast<size_t>(axis)] = tmp % output_shape[static_cast<size_t>(axis)];
                            tmp /= output_shape[static_cast<size_t>(axis)];
                        }
                        for (size_t linear_idx = start_idx; linear_idx < end_idx; ++linear_idx) {
                            size_t lhs_idx = broadcast_linear_index(coords, lhs_strides);
                            size_t rhs_idx = broadcast_linear_index(coords, rhs_strides);
                            out_data[linear_idx] = static_cast<__fp16>(
                                static_cast<float>(lhs_data[lhs_idx]) != static_cast<float>(rhs_data[rhs_idx]) ? 1.0f : 0.0f);
                            increment_broadcast_coords(coords, output_shape);
                        }
                    });
                break;
            }
            default: break;
        }
    } else {
        if (lhs.precision == Precision::FP16) {
            dispatch_binary_op_f16(node.op_type, lhs.data_as<__fp16>(),
                                   rhs.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                   node.output_buffer.total_size);
        } else {
            dispatch_binary_op_f32(node.op_type, lhs.data_as<float>(),
                                   rhs.data_as<float>(), node.output_buffer.data_as<float>(),
                                   node.output_buffer.total_size);
        }
    }
}

void compute_unary_op_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);

    if (input.precision != Precision::FP16 && input.precision != Precision::FP32) {
        throw std::runtime_error("Scalar operations only support FP16/FP32 precision");
    }

    if (node.op_type == OpType::SCALAR_NOT_EQUAL) {
        const float scalar = node.params.scalar;
        if (input.precision == Precision::FP16) {
            const __fp16* input_data = input.data_as<__fp16>();
            __fp16* output_data = node.output_buffer.data_as<__fp16>();
            CactusThreading::parallel_for(node.output_buffer.total_size, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        output_data[i] = static_cast<__fp16>(
                            static_cast<float>(input_data[i]) != scalar ? 1.0f : 0.0f);
                    }
                });
        } else {
            const float* input_data = input.data_as<float>();
            float* output_data = node.output_buffer.data_as<float>();
            CactusThreading::parallel_for(node.output_buffer.total_size, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        output_data[i] = input_data[i] != scalar ? 1.0f : 0.0f;
                    }
                });
        }
        return;
    }

    if (input.precision == Precision::FP16) {
        dispatch_unary_op_f16(node.op_type, input.data_as<__fp16>(),
                              node.output_buffer.data_as<__fp16>(),
                              node.output_buffer.total_size, node.params.scalar);
    } else {
        dispatch_unary_op_f32(node.op_type, input.data_as<float>(),
                              node.output_buffer.data_as<float>(),
                              node.output_buffer.total_size, node.params.scalar);
    }
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
        case OpType::CLAMP:
            cactus_clamp_f16(input.data_as<__fp16>(),
                             node.output_buffer.data_as<__fp16>(),
                             node.output_buffer.total_size,
                             node.params.scalar, node.params.scale);
            break;
        default:
            break;
    }
}

void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    int axis = node.params.axis;

    if (input_buffer.precision != Precision::FP16 && input_buffer.precision != Precision::FP32) {
        throw std::runtime_error("Reduction operations only support FP16/FP32 precision");
    }

    if (node.op_type == OpType::CUMSUM) {
        if (axis < 0 || static_cast<size_t>(axis) >= input_buffer.shape.size()) {
            throw std::runtime_error("Cumsum axis out of range");
        }

        auto dims = AxisDims::from_shape(input_buffer.shape, static_cast<size_t>(axis));
        if (input_buffer.precision == Precision::FP32) {
            const float* input = input_buffer.data_as<float>();
            float* output = node.output_buffer.data_as<float>();
            for (size_t outer = 0; outer < dims.outer; ++outer) {
                for (size_t inner = 0; inner < dims.inner; ++inner) {
                    float running = 0.0f;
                    for (size_t axis_index = 0; axis_index < dims.axis_size; ++axis_index) {
                        size_t flat_index = ((outer * dims.axis_size) + axis_index) * dims.inner + inner;
                        running += input[flat_index];
                        output[flat_index] = running;
                    }
                }
            }
            return;
        }

        const __fp16* input = input_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        for (size_t outer = 0; outer < dims.outer; ++outer) {
            for (size_t inner = 0; inner < dims.inner; ++inner) {
                float running = 0.0f;
                for (size_t axis_index = 0; axis_index < dims.axis_size; ++axis_index) {
                    size_t flat_index = ((outer * dims.axis_size) + axis_index) * dims.inner + inner;
                    running += static_cast<float>(input[flat_index]);
                    output[flat_index] = static_cast<__fp16>(running);
                }
            }
        }
        return;
    }

    if (input_buffer.precision == Precision::FP32) {
        const float* input = input_buffer.data_as<float>();
        float* output = node.output_buffer.data_as<float>();
        if (axis == -1) {
            float result = 0.0f;
            switch (node.op_type) {
                case OpType::SUM:
                case OpType::MEAN:
                    for (size_t i = 0; i < input_buffer.total_size; ++i) result += input[i];
                    if (node.op_type == OpType::MEAN && input_buffer.total_size > 0) {
                        result /= static_cast<float>(input_buffer.total_size);
                    }
                    output[0] = result;
                    break;
                case OpType::VARIANCE: {
                    float mean = 0.0f;
                    for (size_t i = 0; i < input_buffer.total_size; ++i) mean += input[i];
                    mean /= static_cast<float>(std::max<size_t>(input_buffer.total_size, 1));
                    float variance = 0.0f;
                    for (size_t i = 0; i < input_buffer.total_size; ++i) {
                        float diff = input[i] - mean;
                        variance += diff * diff;
                    }
                    output[0] = variance / static_cast<float>(std::max<size_t>(input_buffer.total_size, 1));
                    break;
                }
                case OpType::MIN:
                    result = input_buffer.total_size == 0 ? 0.0f : input[0];
                    for (size_t i = 1; i < input_buffer.total_size; ++i) result = std::min(result, input[i]);
                    output[0] = result;
                    break;
                case OpType::MAX:
                    result = input_buffer.total_size == 0 ? 0.0f : input[0];
                    for (size_t i = 1; i < input_buffer.total_size; ++i) result = std::max(result, input[i]);
                    output[0] = result;
                    break;
                default:
                    break;
            }
            return;
        }

        auto dims = AxisDims::from_shape(input_buffer.shape, static_cast<size_t>(axis));
        CactusThreading::parallel_for(dims.outer * dims.inner, CactusThreading::Thresholds::ELEMENT_WISE,
            [&](size_t start_idx, size_t end_idx) {
                for (size_t out_idx = start_idx; out_idx < end_idx; ++out_idx) {
                    size_t outer = out_idx / dims.inner;
                    size_t inner = out_idx % dims.inner;
                    size_t base = outer * dims.axis_size * dims.inner + inner;
                    float result = 0.0f;
                    switch (node.op_type) {
                        case OpType::SUM:
                        case OpType::MEAN:
                            for (size_t axis_index = 0; axis_index < dims.axis_size; ++axis_index) {
                                result += input[base + axis_index * dims.inner];
                            }
                            if (node.op_type == OpType::MEAN && dims.axis_size > 0) {
                                result /= static_cast<float>(dims.axis_size);
                            }
                            output[out_idx] = result;
                            break;
                        case OpType::VARIANCE: {
                            float mean = 0.0f;
                            for (size_t axis_index = 0; axis_index < dims.axis_size; ++axis_index) {
                                mean += input[base + axis_index * dims.inner];
                            }
                            mean /= static_cast<float>(std::max<size_t>(dims.axis_size, 1));
                            float variance = 0.0f;
                            for (size_t axis_index = 0; axis_index < dims.axis_size; ++axis_index) {
                                float diff = input[base + axis_index * dims.inner] - mean;
                                variance += diff * diff;
                            }
                            output[out_idx] = variance / static_cast<float>(std::max<size_t>(dims.axis_size, 1));
                            break;
                        }
                        case OpType::MIN:
                            result = dims.axis_size == 0 ? 0.0f : input[base];
                            for (size_t axis_index = 1; axis_index < dims.axis_size; ++axis_index) {
                                result = std::min(result, input[base + axis_index * dims.inner]);
                            }
                            output[out_idx] = result;
                            break;
                        case OpType::MAX:
                            result = dims.axis_size == 0 ? 0.0f : input[base];
                            for (size_t axis_index = 1; axis_index < dims.axis_size; ++axis_index) {
                                result = std::max(result, input[base + axis_index * dims.inner]);
                            }
                            output[out_idx] = result;
                            break;
                        default:
                            break;
                    }
                }
            });
        return;
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

    if (cactus_metal_active_mode()) {
        node.output_buffer.set_external(const_cast<void*>(input_buffer.get_data()));
    } else {
        std::memcpy(node.output_buffer.get_data(), input_buffer.get_data(), input_buffer.byte_size);
    }
}

void compute_precision_cast_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buf = get_input(node, 0, nodes, node_index_map);

    if (input_buf.precision == node.output_buffer.precision) {
        if (cactus_metal_active_mode()) {
            node.output_buffer.set_external(const_cast<void*>(input_buf.get_data()));
        } else {
            std::memcpy(node.output_buffer.get_data(), input_buf.get_data(), input_buf.byte_size);
        }
        return;
    }

    size_t count = input_buf.total_size;

    if (input_buf.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP32) {
        Quantization::int8_to_fp32(input_buf.data_as<int8_t>(), node.output_buffer.data_as<float>(), count, 1.0f);
    } else if (input_buf.precision == Precision::FP32 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp32_to_int8(input_buf.data_as<float>(), node.output_buffer.data_as<int8_t>(), count, 1.0f);
    } else if (input_buf.precision == Precision::FP16 && node.output_buffer.precision == Precision::FP32) {
        Quantization::fp16_to_fp32(input_buf.data_as<__fp16>(), node.output_buffer.data_as<float>(), count);
    } else if (input_buf.precision == Precision::FP32 && node.output_buffer.precision == Precision::FP16) {
        Quantization::fp32_to_fp16(input_buf.data_as<float>(), node.output_buffer.data_as<__fp16>(), count);
    } else if (input_buf.precision == Precision::INT8 && node.output_buffer.precision == Precision::FP16) {
        Quantization::int8_to_fp16(input_buf.data_as<int8_t>(), node.output_buffer.data_as<__fp16>(), count, 1.0f);
    } else if (input_buf.precision == Precision::FP16 && node.output_buffer.precision == Precision::INT8) {
        Quantization::fp16_to_int8(input_buf.data_as<__fp16>(), node.output_buffer.data_as<int8_t>(), count, 1.0f);
    } else {
        throw std::runtime_error("Unsupported precision conversion from " +
                                std::to_string(static_cast<int>(input_buf.precision)) +
                                " to " + std::to_string(static_cast<int>(node.output_buffer.precision)));
    }
}
