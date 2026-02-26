#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <cmath>
#include <stdexcept>

void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU transpose operation not yet implemented");
    }

    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Transpose only supports FP16 precision");
    }

    const auto& permutation = node.params.permutation;

    const __fp16* input = input_buffer.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();
    cactus_transpose_f16(input, output, input_buffer.shape.data(), permutation.data(), permutation.size(), 0, input_buffer.total_size);
}

void compute_gather_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& tensor_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    size_t first_dim = tensor_buffer.shape[0];
    size_t element_size = 1;
    for (size_t i = 1; i < tensor_buffer.shape.size(); i++) {
        element_size *= tensor_buffer.shape[i];
    }

    size_t num_indices = indices_buffer.total_size;
    size_t bytes_per_element = PrecisionTraits::packed_size_of(tensor_buffer.precision, element_size);

    if (PrecisionTraits::is_integer(tensor_buffer.precision)) {
        const char* tensor_data = static_cast<const char*>(tensor_buffer.get_data());
        char* output = static_cast<char*>(node.output_buffer.get_data());
        Precision prec = tensor_buffer.precision;

        const bool is_grouped = tensor_buffer.group_size > 0;
        __fp16* gathered_scales = nullptr;
        const __fp16* src_scales = nullptr;
        size_t num_groups = 0;

        if (is_grouped) {
            num_groups = tensor_buffer.num_groups;
            src_scales = tensor_buffer.scales_as_fp16();
            size_t scales_bytes = num_indices * num_groups * sizeof(__fp16);
            node.output_buffer.owned_scales = std::make_unique<char[]>(scales_bytes);
            gathered_scales = reinterpret_cast<__fp16*>(node.output_buffer.owned_scales.get());
        }

        const int8_t* indices = indices_buffer.data_as<int8_t>();
        for (size_t i = 0; i < num_indices; i++) {
            size_t idx = static_cast<size_t>(indices[i]);
            if (idx >= first_dim) {
                throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
            }
            std::memcpy(output + PrecisionTraits::byte_offset_of(prec, i * element_size),
                        tensor_data + PrecisionTraits::byte_offset_of(prec, idx * element_size),
                        bytes_per_element);
            if (is_grouped) {
                for (size_t g = 0; g < num_groups; g++) {
                    gathered_scales[i * num_groups + g] = src_scales[idx * num_groups + g];
                }
            }
        }

        if (is_grouped) {
            node.output_buffer.group_size = tensor_buffer.group_size;
            node.output_buffer.num_groups = num_groups;
            node.output_buffer.scales_data = gathered_scales;
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
}

void compute_slice_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    auto* input_node = nodes[node_index_map.at(node.input_ids[0])].get();
    auto& input_buffer = input_node->output_buffer;

    const size_t axis_index = static_cast<size_t>(node.params.axis);

    const size_t axis_size = input_buffer.shape[axis_index];
    const size_t slice_start = node.params.slice_start;
    size_t slice_length = node.params.slice_length;

    if (slice_length == 0) {
        slice_length = axis_size - slice_start;
    }

    if (axis_index == 0) {
        size_t inner_elements = 1;
        for (size_t i = 1; i < input_buffer.shape.size(); ++i) {
            inner_elements *= input_buffer.shape[i];
        }

        auto* base_ptr = static_cast<char*>(input_buffer.get_data());
        if (!base_ptr) {
            throw std::runtime_error("Slice input buffer is not available");
        }

        const size_t byte_offset = PrecisionTraits::byte_offset_of(input_buffer.precision, slice_start * inner_elements);

        node.output_buffer.set_external(base_ptr + byte_offset);
        node.output_buffer.precision = input_buffer.precision;

        if (input_buffer.is_grouped_int8()) {
            size_t num_groups = input_buffer.num_groups;
            size_t scales_bytes = slice_length * num_groups * sizeof(__fp16);
            node.output_buffer.owned_scales = std::make_unique<char[]>(scales_bytes);
            __fp16* sliced_scales = reinterpret_cast<__fp16*>(node.output_buffer.owned_scales.get());
            const __fp16* input_scales = input_buffer.scales_as_fp16();

            for (size_t i = 0; i < slice_length; i++) {
                for (size_t g = 0; g < num_groups; g++) {
                    sliced_scales[i * num_groups + g] = input_scales[(slice_start + i) * num_groups + g];
                }
            }

            node.output_buffer.group_size = input_buffer.group_size;
            node.output_buffer.num_groups = num_groups;
            node.output_buffer.scales_data = sliced_scales;
        }
        return;
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

    node.output_buffer.external_data = nullptr;
    node.output_buffer.allocate();
    node.output_buffer.precision = input_buffer.precision;

    auto* output_ptr = static_cast<char*>(node.output_buffer.get_data());
    if (!output_ptr) {
        throw std::runtime_error("Slice output buffer could not be allocated");
    }

    const size_t copy_block_elements = slice_length * inner_elements;
    const size_t axis_stride_elements = axis_size * inner_elements;
    const size_t copy_block_bytes = PrecisionTraits::byte_offset_of(input_buffer.precision, copy_block_elements);
    const size_t axis_stride_bytes = PrecisionTraits::byte_offset_of(input_buffer.precision, axis_stride_elements);

    for (size_t outer = 0; outer < outer_elements; ++outer) {
        const char* src = input_ptr + outer * axis_stride_bytes + PrecisionTraits::byte_offset_of(input_buffer.precision, slice_start * inner_elements);
        char* dst = output_ptr + outer * copy_block_bytes;
        std::memcpy(dst, src, copy_block_bytes);
    }
}

void compute_embedding_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& embeddings_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    size_t hidden_dim = embeddings_buffer.shape[1];
    size_t num_indices = indices_buffer.total_size;
    size_t vocab_size = embeddings_buffer.original_N > 0
                      ? embeddings_buffer.original_N
                      : embeddings_buffer.shape[0];

    std::vector<float> indices_float;
    const float* indices_ptr;
    if (indices_buffer.precision == Precision::FP32) {
        indices_ptr = indices_buffer.data_as<float>();
    } else {
        indices_float.resize(num_indices);
        const int8_t* int_indices = indices_buffer.data_as<int8_t>();
        for (size_t i = 0; i < num_indices; i++) {
            indices_float[i] = static_cast<float>(int_indices[i]);
        }
        indices_ptr = indices_float.data();
    }

    __fp16* output = node.output_buffer.data_as<__fp16>();

    if (embeddings_buffer.precision == Precision::INT8 && embeddings_buffer.is_grouped_int8()) {
        const int8_t* embeddings = embeddings_buffer.data_as<int8_t>();
        const __fp16* scales = embeddings_buffer.scales_as_fp16();
        size_t group_size = embeddings_buffer.group_size;
        size_t num_groups = embeddings_buffer.num_groups;

        static const uint8_t gather_indices[4][16] = {
            {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51},  // lane 0
            {4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55},  // lane 1
            {8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59}, // lane 2
            {12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63} // lane 3
        };

        for (size_t i = 0; i < num_indices; i++) {
            size_t idx = static_cast<size_t>(indices_ptr[i]);
            if (idx >= vocab_size) {
                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
            }

            size_t block = idx / 4;
            size_t lane = idx % 4;
            __fp16* out_row = output + i * hidden_dim;

            uint8x16_t indices_vec = vld1q_u8(gather_indices[lane]);

            for (size_t g = 0; g < num_groups; g++) {
                float scale = (float)scales[(block * num_groups + g) * 4 + lane];
                float32x4_t scale_vec = vdupq_n_f32(scale);

                size_t k_start = g * group_size;
                size_t k_end = std::min(k_start + group_size, hidden_dim);

                const int8_t* group_base = embeddings + (block * hidden_dim + k_start) * 4;

                size_t k = k_start;
                for (; k + 16 <= k_end; k += 16) {
                    const int8_t* chunk_base = group_base + (k - k_start) * 4;
                    int8x16x4_t table = vld1q_s8_x4(chunk_base);

                    int8x16_t values = vqtbl4q_s8(table, indices_vec);

                    int16x8_t lo16 = vmovl_s8(vget_low_s8(values));
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(values));

                    float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), scale_vec);
                    float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), scale_vec);
                    float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), scale_vec);
                    float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), scale_vec);

                    vst1q_f16(out_row + k, vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1)));
                    vst1q_f16(out_row + k + 8, vcombine_f16(vcvt_f16_f32(f2), vcvt_f16_f32(f3)));
                }

                for (; k < k_end; k++) {
                    size_t k_group = k / 4;
                    size_t k_within = k % 4;
                    int8_t val = embeddings[(block * (hidden_dim / 4) + k_group) * 16 + lane * 4 + k_within];
                    out_row[k] = static_cast<__fp16>(val * scale);
                }
            }
        }
    } else if (embeddings_buffer.precision == Precision::FP16) {
        const __fp16* embeddings = embeddings_buffer.data_as<__fp16>();
        for (size_t i = 0; i < num_indices; i++) {
            size_t idx = static_cast<size_t>(indices_ptr[i]);
            if (idx >= vocab_size) {
                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
            }
            std::memcpy(output + i * hidden_dim, embeddings + idx * hidden_dim, hidden_dim * sizeof(__fp16));
        }
    } else {
        throw std::runtime_error("Embedding requires interleaved grouped INT8 or FP16");
    }
}

void compute_concat_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input1_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& input2_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    std::vector<size_t> shape1 = input1_buffer.shape;
    std::vector<size_t> shape2 = input2_buffer.shape;
    std::vector<size_t> output_shape = node.output_buffer.shape;

    if (input1_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Concat operation only supports FP16 precision");
    }
    cactus_concat_f16(input1_buffer.data_as<__fp16>(), input2_buffer.data_as<__fp16>(),
                     node.output_buffer.data_as<__fp16>(),
                     shape1.data(), shape2.data(), output_shape.data(),
                     shape1.size(), node.params.axis);
}

void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& input_shape = input_buffer.shape;

    int dim = node.params.axis;
    size_t index_value = node.params.index_value;

    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());

    if (dim == 0) {
        size_t slice_size = input_buffer.total_size / input_shape[0];
        size_t offset_bytes = PrecisionTraits::byte_offset_of(input_buffer.precision, index_value * slice_size);
        node.output_buffer.set_external(const_cast<char*>(input_data) + offset_bytes);
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

        char* output_offset_bytes = output_data + PrecisionTraits::byte_offset_of(input_buffer.precision, output_idx);
        const char* input_offset_bytes = input_data + PrecisionTraits::byte_offset_of(input_buffer.precision, input_base);
        size_t length = PrecisionTraits::byte_offset_of(input_buffer.precision, slice_size);
        std::memcpy(output_offset_bytes, input_offset_bytes, length);

        output_idx += slice_size;
    }
}

void compute_bilinear_interpolation_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& pos_embeds_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

    size_t total_pos_embeds = pos_embeds_buffer.shape[0];
    size_t embed_dim = pos_embeds_buffer.shape[1];

    size_t src_height = static_cast<size_t>(std::sqrt(total_pos_embeds));
    size_t src_width = src_height;

    size_t dst_height = node.params.dst_height;
    size_t dst_width = node.params.dst_width;

    __fp16* output = node.output_buffer.data_as<__fp16>();

    if (pos_embeds_buffer.precision == Precision::FP16) {
        const __fp16* input = pos_embeds_buffer.data_as<__fp16>();
        cactus_bilinear_interpolation_f16(input, output, src_height, src_width, embed_dim,
                                          dst_height, dst_width);
    }
    else if (pos_embeds_buffer.precision == Precision::INT8) {
        std::vector<__fp16> input_fp16(total_pos_embeds * embed_dim);
        cactus_int8_to_fp16(pos_embeds_buffer.data_as<int8_t>(), input_fp16.data(),
                            total_pos_embeds * embed_dim);
        cactus_bilinear_interpolation_f16(input_fp16.data(), output, src_height, src_width, embed_dim,
                                          dst_height, dst_width);
    }
    else {
        throw std::runtime_error("BILINEAR_INTERPOLATION only supports INT8 and FP16 input precision");
    }
}

void compute_persistent_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.empty()) {
        return;
    }

    auto it = node_index_map.find(node.input_ids[0]);
    
    if (it != node_index_map.end()) {
        const auto& input_buffer = nodes[it->second]->output_buffer;
        
        if (!node.output_buffer.get_data()) {
            node.output_buffer.allocate();
        }
        
        std::memcpy(node.output_buffer.get_data(), 
                    input_buffer.get_data(), 
                    input_buffer.byte_size);
    } else {
        if (node.output_buffer.get_data()) {
            return;
        }
        throw std::runtime_error("PERSISTENT node input not found and not populated - this should not happen");
    }
}
