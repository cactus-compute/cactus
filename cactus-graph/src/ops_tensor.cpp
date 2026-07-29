#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <unordered_map>

static std::vector<size_t> contiguous_strides(const std::vector<size_t>& shape);

template <typename T>
static void transpose_generic(
    const T* input,
    T* output,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& permutation) {
    std::vector<size_t> input_strides = contiguous_strides(input_shape);

    size_t total = 1;
    for (size_t dim : output_shape) total *= dim;

    for (size_t output_linear = 0; output_linear < total; ++output_linear) {
        size_t tmp = output_linear;
        size_t input_linear = 0;

        for (int dim = static_cast<int>(output_shape.size()) - 1; dim >= 0; --dim) {
            size_t dim_index = static_cast<size_t>(dim);
            size_t coord = tmp % output_shape[dim_index];
            tmp /= output_shape[dim_index];
            input_linear += coord * input_strides[permutation[dim_index]];
        }

        output[output_linear] = input[input_linear];
    }
}

void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU transpose operation not yet implemented");
    }

    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& permutation = node.params.permutation;

    if (input_buffer.precision == Precision::FP16) {
        const __fp16* input = input_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();
        cactus_transpose_f16(input, output, input_buffer.shape.data(), permutation.data(), permutation.size(), 0, input_buffer.total_size);
        return;
    }

    if (input_buffer.precision == Precision::FP32) {
        transpose_generic(
            input_buffer.data_as<float>(),
            node.output_buffer.data_as<float>(),
            input_buffer.shape,
            node.output_buffer.shape,
            permutation);
        return;
    }

    if (input_buffer.precision == Precision::INT8) {
        transpose_generic(
            input_buffer.data_as<int8_t>(),
            node.output_buffer.data_as<int8_t>(),
            input_buffer.shape,
            node.output_buffer.shape,
            permutation);
        return;
    }

    throw std::runtime_error("Transpose only supports FP32/FP16/INT8 precision");
}

void compute_gather_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& tensor_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& indices_buffer = get_input(node, 1, nodes, node_index_map);

    if (tensor_buffer.shape.size() == indices_buffer.shape.size()) {
        int axis_value = node.params.axis;
        if (axis_value < 0) axis_value += static_cast<int>(tensor_buffer.shape.size());
        if (axis_value < 0 || static_cast<size_t>(axis_value) >= tensor_buffer.shape.size()) {
            throw std::runtime_error("Gather axis out of range");
        }

        const size_t axis = static_cast<size_t>(axis_value);
        std::vector<size_t> tensor_strides = contiguous_strides(tensor_buffer.shape);
        std::vector<size_t> output_strides = contiguous_strides(node.output_buffer.shape);
        const size_t elem_bytes = PrecisionTraits::packed_size_of(tensor_buffer.precision, 1);
        const char* tensor_data = static_cast<const char*>(tensor_buffer.get_data());
        char* output_data = static_cast<char*>(node.output_buffer.get_data());

        auto read_index = [&](size_t i) -> size_t {
            if (indices_buffer.precision == Precision::INT8) {
                return static_cast<size_t>(indices_buffer.data_as<int8_t>()[i]);
            }
            if (indices_buffer.precision == Precision::FP16) {
                return static_cast<size_t>(indices_buffer.data_as<__fp16>()[i]);
            }
            return static_cast<size_t>(indices_buffer.data_as<float>()[i]);
        };

        for (size_t output_linear = 0; output_linear < node.output_buffer.total_size; ++output_linear) {
            size_t tmp = output_linear;
            size_t input_linear = 0;
            size_t gathered_index = read_index(output_linear);

            if (gathered_index >= tensor_buffer.shape[axis]) {
                throw std::runtime_error("Gather index out of bounds");
            }

            for (size_t dim = 0; dim < node.output_buffer.shape.size(); ++dim) {
                size_t coord = tmp / output_strides[dim];
                tmp %= output_strides[dim];
                input_linear += (dim == axis ? gathered_index : coord) * tensor_strides[dim];
            }

            std::memcpy(
                output_data + PrecisionTraits::byte_offset_of(node.output_buffer.precision, output_linear),
                tensor_data + PrecisionTraits::byte_offset_of(tensor_buffer.precision, input_linear),
                elem_bytes);
        }
        return;
    }

    size_t first_dim = tensor_buffer.shape[0];
    size_t element_size = 1;
    for (size_t i = 1; i < tensor_buffer.shape.size(); i++) {
        element_size *= tensor_buffer.shape[i];
    }

    size_t num_indices = indices_buffer.total_size;
    size_t bytes_per_element = PrecisionTraits::packed_size_of(tensor_buffer.precision, element_size);

    {
        const char* tensor_data = static_cast<const char*>(tensor_buffer.get_data());
        char* output = static_cast<char*>(node.output_buffer.get_data());

        auto gather_loop = [&](auto get_idx) {
            for (size_t i = 0; i < num_indices; i++) {
                size_t idx = get_idx(i);
                if (idx >= first_dim)
                    throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                std::memcpy(output + PrecisionTraits::byte_offset_of(tensor_buffer.precision, i * element_size),
                            tensor_data + PrecisionTraits::byte_offset_of(tensor_buffer.precision, idx * element_size),
                            bytes_per_element);
            }
        };

        if (indices_buffer.precision == Precision::INT8) {
            const int8_t* indices = indices_buffer.data_as<int8_t>();
            gather_loop([&](size_t i) { return static_cast<size_t>(indices[i]); });
        } else {
            const float* indices = indices_buffer.data_as<float>();
            gather_loop([&](size_t i) { return static_cast<size_t>(indices[i]); });
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
        auto dims0 = AxisDims::from_shape(input_buffer.shape, 0);

        auto* base_ptr = static_cast<char*>(input_buffer.get_data());
        if (!base_ptr) {
            throw std::runtime_error("Slice input buffer is not available");
        }

        const size_t byte_offset = PrecisionTraits::byte_offset_of(input_buffer.precision, slice_start * dims0.inner);

        node.output_buffer.set_external(base_ptr + byte_offset);
        node.output_buffer.precision = input_buffer.precision;

        if (input_buffer.is_cq()) {
            node.output_buffer.group_size = input_buffer.group_size;
            node.output_buffer.num_groups = input_buffer.num_groups;
        }
        return;
    }

    const char* input_ptr = static_cast<const char*>(input_buffer.get_data());
    if (!input_ptr) {
        throw std::runtime_error("Slice input buffer is not available");
    }

    auto dims = AxisDims::from_shape(input_buffer.shape, axis_index);

    node.output_buffer.external_data = nullptr;
    node.output_buffer.allocate();
    node.output_buffer.precision = input_buffer.precision;

    auto* output_ptr = static_cast<char*>(node.output_buffer.get_data());
    if (!output_ptr) {
        throw std::runtime_error("Slice output buffer could not be allocated");
    }

    const size_t copy_block_elements = slice_length * dims.inner;
    const size_t axis_stride_elements = axis_size * dims.inner;
    const size_t copy_block_bytes = PrecisionTraits::byte_offset_of(input_buffer.precision, copy_block_elements);
    const size_t axis_stride_bytes = PrecisionTraits::byte_offset_of(input_buffer.precision, axis_stride_elements);

    for (size_t outer = 0; outer < dims.outer; ++outer) {
        const char* src = input_ptr + outer * axis_stride_bytes + PrecisionTraits::byte_offset_of(input_buffer.precision, slice_start * dims.inner);
        char* dst = output_ptr + outer * copy_block_bytes;
        std::memcpy(dst, src, copy_block_bytes);
    }
}

void compute_strided_slice_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& input_shape = input_buffer.shape;
    const auto& output_shape = node.output_buffer.shape;

    int axis_value = node.params.axis;
    if (axis_value < 0) axis_value += static_cast<int>(input_shape.size());
    if (axis_value < 0 || static_cast<size_t>(axis_value) >= input_shape.size()) {
        throw std::runtime_error("strided_slice axis out of range");
    }
    if (node.params.stride == 0) {
        throw std::runtime_error("strided_slice step must be greater than zero");
    }

    const size_t axis = static_cast<size_t>(axis_value);
    const size_t start = node.params.slice_start;
    const size_t step = node.params.stride;
    const size_t length = node.params.slice_length;

    if (output_shape.size() != input_shape.size() || output_shape[axis] != length) {
        throw std::runtime_error("strided_slice shape mismatch");
    }

    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());
    if (!input_data || !output_data) {
        throw std::runtime_error("strided_slice input/output buffer is not available");
    }

    auto dims = AxisDims::from_shape(input_shape, axis);
    const size_t elem_bytes = PrecisionTraits::packed_size_of(input_buffer.precision, dims.inner);

    for (size_t outer = 0; outer < dims.outer; ++outer) {
        for (size_t axis_index = 0; axis_index < length; ++axis_index) {
            size_t input_axis = start + axis_index * step;
            if (input_axis >= dims.axis_size) {
                throw std::runtime_error("strided_slice index out of bounds");
            }

            const size_t input_offset = (outer * dims.axis_size + input_axis) * dims.inner;
            const size_t output_offset = (outer * length + axis_index) * dims.inner;
            std::memcpy(
                output_data + PrecisionTraits::byte_offset_of(input_buffer.precision, output_offset),
                input_data + PrecisionTraits::byte_offset_of(input_buffer.precision, input_offset),
                elem_bytes);
        }
    }
}

void compute_embedding_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& embeddings_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& indices_buffer = get_input(node, 1, nodes, node_index_map);

    size_t hidden_dim = embeddings_buffer.shape[1];
    size_t num_indices = indices_buffer.total_size;
    size_t vocab_size = embeddings_buffer.shape[0];

    std::vector<float> indices_float;
    const float* indices_ptr;
    if (indices_buffer.precision == Precision::FP32) {
        indices_ptr = indices_buffer.data_as<float>();
    } else if (indices_buffer.precision == Precision::FP16) {
        indices_float.resize(num_indices);
        const __fp16* half_indices = indices_buffer.data_as<__fp16>();
        for (size_t i = 0; i < num_indices; i++) {
            indices_float[i] = static_cast<float>(half_indices[i]);
        }
        indices_ptr = indices_float.data();
    } else {
        indices_float.resize(num_indices);
        const int8_t* int_indices = indices_buffer.data_as<int8_t>();
        for (size_t i = 0; i < num_indices; i++) {
            indices_float[i] = static_cast<float>(int_indices[i]);
        }
        indices_ptr = indices_float.data();
    }

    __fp16* output = node.output_buffer.data_as<__fp16>();

    Precision emb_prec = embeddings_buffer.precision;
    if (PrecisionTraits::is_cq(emb_prec) && embeddings_buffer.group_size > 0) {
        bool orthogonal = (embeddings_buffer.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL) != 0;
        std::unordered_map<size_t, size_t> row_cache;
        std::vector<__fp16> cached_rows;
        if (num_indices > 16) {
            row_cache.reserve(std::min<size_t>(num_indices, 256));
        }
        for (size_t i = 0; i < num_indices; i++) {
            size_t idx = static_cast<size_t>(indices_ptr[i]);
            if (idx >= vocab_size) {
                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
            }
            if (num_indices > 16) {
                auto it = row_cache.find(idx);
                if (it != row_cache.end()) {
                    std::memcpy(output + i * hidden_dim,
                                cached_rows.data() + it->second * hidden_dim,
                                hidden_dim * sizeof(__fp16));
                    continue;
                }
            }
            if (orthogonal) {
                cactus_quant_dequantize_orthogonal_embedding_row(
                    PrecisionTraits::cq_bits(emb_prec),
                    static_cast<uint32_t>(hidden_dim),
                    idx,
                    embeddings_buffer.data_as<uint8_t>(),
                    embeddings_buffer.cq_codebook,
                    embeddings_buffer.cq_norms,
                    embeddings_buffer.cq_input_scale_recip,
                    embeddings_buffer.cq_rotation,
                    embeddings_buffer.cq_flags,
                    output + i * hidden_dim);
            } else {
                cactus_quant_dequantize_hadamard_embedding_row(
                    PrecisionTraits::cq_bits(emb_prec),
                    static_cast<uint32_t>(hidden_dim),
                    static_cast<uint32_t>(embeddings_buffer.group_size),
                    static_cast<uint32_t>(embeddings_buffer.num_groups),
                    idx,
                    embeddings_buffer.data_as<uint8_t>(),
                    embeddings_buffer.cq_codebook,
                    embeddings_buffer.cq_norms,
                    embeddings_buffer.cq_input_scale_recip,
                    embeddings_buffer.cq_left_signs,
                    embeddings_buffer.cq_right_signs,
                    embeddings_buffer.cq_permutation,
                    output + i * hidden_dim);
            }
            if (num_indices > 16) {
                const size_t cache_slot = cached_rows.size() / hidden_dim;
                row_cache.emplace(idx, cache_slot);
                cached_rows.insert(cached_rows.end(), output + i * hidden_dim, output + (i + 1) * hidden_dim);
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
        throw std::runtime_error("Embedding requires CQ quantized or FP16 data");
    }
}

void compute_concat_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input1_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& input2_buffer = get_input(node, 1, nodes, node_index_map);

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

void compute_cat_node(
    GraphNode& node,
    const std::vector<std::unique_ptr<GraphNode>>& nodes,
    const std::unordered_map<size_t, size_t>& node_index_map
) {
    if (node.params.axis < 0) {
        throw std::runtime_error("Cat operation does not support negative axis");
    }
    if (node.input_ids.size() < 2) {
        throw std::runtime_error("Cat operation requires at least 2 input tensors");
    }

    const auto& first_buffer = get_input(node, 0, nodes, node_index_map);

    if (first_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Cat operation only supports FP16 precision");
    }

    std::vector<const __fp16*> input_data_ptrs(node.input_ids.size());
    std::vector<const size_t*> input_shape_ptrs(node.input_ids.size());

    for (size_t i = 0; i < node.input_ids.size(); i++) {
        const auto& buffer = get_input(node, i, nodes, node_index_map);

        if (buffer.precision != Precision::FP16) {
            throw std::runtime_error("Cat operation only supports FP16 precision");
        }

        input_data_ptrs[i] = buffer.data_as<__fp16>();
        input_shape_ptrs[i] = buffer.shape.data();
    }

    cactus_cat_f16(input_data_ptrs.data(),
                   node.output_buffer.data_as<__fp16>(),
                   input_shape_ptrs.data(),
                   node.output_buffer.shape.data(),
                   node.input_ids.size(),
                   node.output_buffer.shape.size(),
                   node.params.axis);
}

void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
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

void compute_unfold_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& input_shape = input_buffer.shape;
    const auto& output_shape = node.output_buffer.shape;

    if (input_shape.empty() || output_shape.size() != input_shape.size() + 1) {
        throw std::runtime_error("unfold shape mismatch");
    }

    int axis_value = node.params.axis;
    if (axis_value < 0) axis_value += static_cast<int>(input_shape.size());
    if (axis_value < 0 || static_cast<size_t>(axis_value) >= input_shape.size()) {
        throw std::runtime_error("unfold dimension out of range");
    }

    const size_t axis = static_cast<size_t>(axis_value);
    const size_t window_size = node.params.kernel_size;
    const size_t step = node.params.stride;
    if (window_size == 0 || step == 0) {
        throw std::runtime_error("unfold window size and step must be greater than zero");
    }

    std::vector<size_t> input_strides(input_shape.size(), 1);
    for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; --i) {
        input_strides[static_cast<size_t>(i)] =
            input_strides[static_cast<size_t>(i + 1)] * input_shape[static_cast<size_t>(i + 1)];
    }

    const size_t elem_bytes = PrecisionTraits::packed_size_of(input_buffer.precision, 1);
    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());

    if (!input_data || !output_data) {
        throw std::runtime_error("unfold input/output buffer is not available");
    }

    for (size_t out_linear = 0; out_linear < node.output_buffer.total_size; ++out_linear) {
        size_t tmp = out_linear;
        std::vector<size_t> coords(output_shape.size(), 0);
        for (int dim = static_cast<int>(output_shape.size()) - 1; dim >= 0; --dim) {
            coords[static_cast<size_t>(dim)] = tmp % output_shape[static_cast<size_t>(dim)];
            tmp /= output_shape[static_cast<size_t>(dim)];
        }

        size_t input_linear = 0;
        for (size_t dim = 0; dim < input_shape.size(); ++dim) {
            size_t input_coord = coords[dim];
            if (dim == axis) {
                input_coord = coords[dim] * step + coords.back();
            }
            input_linear += input_coord * input_strides[dim];
        }

        std::memcpy(
            output_data + PrecisionTraits::byte_offset_of(node.output_buffer.precision, out_linear),
            input_data + PrecisionTraits::byte_offset_of(input_buffer.precision, input_linear),
            elem_bytes);
    }
}

static std::vector<size_t> contiguous_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

void compute_expand_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& input_shape = input_buffer.shape;
    const auto& output_shape = node.output_buffer.shape;

    if (output_shape.size() < input_shape.size()) {
        throw std::runtime_error("expand output rank is smaller than input rank");
    }

    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());
    if (!input_data || !output_data) {
        throw std::runtime_error("expand input/output buffer is not available");
    }

    std::vector<size_t> input_strides = contiguous_strides(input_shape);
    const size_t rank_offset = output_shape.size() - input_shape.size();
    const size_t elem_bytes = PrecisionTraits::packed_size_of(input_buffer.precision, 1);

    for (size_t output_linear = 0; output_linear < node.output_buffer.total_size; ++output_linear) {
        size_t tmp = output_linear;
        size_t input_linear = 0;

        for (int dim = static_cast<int>(output_shape.size()) - 1; dim >= 0; --dim) {
            size_t dim_index = static_cast<size_t>(dim);
            size_t coord = tmp % output_shape[dim_index];
            tmp /= output_shape[dim_index];

            if (dim_index < rank_offset) {
                continue;
            }

            size_t input_dim_index = dim_index - rank_offset;
            size_t input_coord = input_shape[input_dim_index] == 1 ? 0 : coord;
            input_linear += input_coord * input_strides[input_dim_index];
        }

        std::memcpy(
            output_data + PrecisionTraits::byte_offset_of(node.output_buffer.precision, output_linear),
            input_data + PrecisionTraits::byte_offset_of(input_buffer.precision, input_linear),
            elem_bytes);
    }
}

static void fill_buffer(BufferDesc& buffer, float value) {
    if (value == 0.0f) {
        std::memset(buffer.get_data(), 0, buffer.byte_size);
        return;
    }

    if (buffer.precision == Precision::FP32) {
        float* data = buffer.data_as<float>();
        for (size_t i = 0; i < buffer.total_size; ++i) data[i] = value;
        return;
    }

    if (buffer.precision == Precision::FP16) {
        __fp16* data = buffer.data_as<__fp16>();
        for (size_t i = 0; i < buffer.total_size; ++i) data[i] = static_cast<__fp16>(value);
        return;
    }

    if (buffer.precision == Precision::INT8) {
        int8_t* data = buffer.data_as<int8_t>();
        for (size_t i = 0; i < buffer.total_size; ++i) data[i] = static_cast<int8_t>(value);
        return;
    }

    throw std::runtime_error("pad only supports FP32/FP16/INT8 precision");
}

void compute_pad_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& input_shape = input_buffer.shape;
    const auto& output_shape = node.output_buffer.shape;
    const auto& pads = node.params.new_shape;

    if (input_shape.empty() || output_shape.size() != input_shape.size()) {
        throw std::runtime_error("pad shape mismatch");
    }
    if (pads.size() % 2 != 0 || (pads.size() / 2) > input_shape.size()) {
        throw std::runtime_error("pad requires left/right pad pairs within input rank");
    }
    if (input_buffer.precision != Precision::FP32 &&
        input_buffer.precision != Precision::FP16 &&
        input_buffer.precision != Precision::INT8) {
        throw std::runtime_error("pad only supports FP32/FP16/INT8 precision");
    }

    fill_buffer(node.output_buffer, node.params.scalar);

    std::vector<size_t> left_offsets(input_shape.size(), 0);
    for (size_t pair = 0; pair < pads.size(); pair += 2) {
        size_t axis = input_shape.size() - 1 - (pair / 2);
        left_offsets[axis] = pads[pair];
    }

    std::vector<size_t> input_strides = contiguous_strides(input_shape);
    std::vector<size_t> output_strides = contiguous_strides(output_shape);
    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());
    const size_t elem_bytes = PrecisionTraits::packed_size_of(input_buffer.precision, 1);

    if (!input_data || !output_data) {
        throw std::runtime_error("pad input/output buffer is not available");
    }

    for (size_t input_linear = 0; input_linear < input_buffer.total_size; ++input_linear) {
        size_t tmp = input_linear;
        size_t output_linear = 0;

        for (size_t dim = 0; dim < input_shape.size(); ++dim) {
            size_t coord = tmp / input_strides[dim];
            tmp %= input_strides[dim];
            output_linear += (coord + left_offsets[dim]) * output_strides[dim];
        }

        std::memcpy(
            output_data + PrecisionTraits::byte_offset_of(node.output_buffer.precision, output_linear),
            input_data + PrecisionTraits::byte_offset_of(input_buffer.precision, input_linear),
            elem_bytes);
    }
}

void compute_bilinear_interpolation_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& pos_embeds_buffer = get_input(node, 0, nodes, node_index_map);

    size_t total_pos_embeds = pos_embeds_buffer.shape[0];
    size_t embed_dim = pos_embeds_buffer.shape[1];

    size_t src_height = static_cast<size_t>(std::sqrt(total_pos_embeds));
    size_t src_width = src_height;

    size_t dst_height = node.params.dst_height;
    size_t dst_width = node.params.dst_width;
    bool align_corners = node.params.align_corners;

    __fp16* output = node.output_buffer.data_as<__fp16>();

    if (pos_embeds_buffer.precision == Precision::FP16) {
        const __fp16* input = pos_embeds_buffer.data_as<__fp16>();
        cactus_bilinear_interpolation_f16(input, output, src_height, src_width, embed_dim,
                                          dst_height, dst_width, align_corners);
    }
    else if (pos_embeds_buffer.precision == Precision::INT8) {
        std::vector<__fp16> input_fp16(total_pos_embeds * embed_dim);
        cactus_int8_to_fp16(pos_embeds_buffer.data_as<int8_t>(), input_fp16.data(),
                            total_pos_embeds * embed_dim);
        cactus_bilinear_interpolation_f16(input_fp16.data(), output, src_height, src_width, embed_dim,
                                          dst_height, dst_width, align_corners);
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
