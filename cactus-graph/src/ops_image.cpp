#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <vector>

void compute_image_preprocess_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& input = get_input(node, 0, nodes, node_index_map);

    int src_w = static_cast<int>(node.params.dst_width);
    int src_h = static_cast<int>(node.params.dst_height);
    int dst_w = node.params.target_width;
    int dst_h = node.params.target_height;
    int ps = node.params.patch_size;
    int ch = node.params.image_channels;
    float rescale = node.params.rescale_factor;

    std::vector<float> src_float(static_cast<size_t>(src_w) * src_h * ch);
    if (input.precision == Precision::FP32) {
        std::memcpy(src_float.data(), input.get_data(), src_float.size() * sizeof(float));
    } else if (input.precision == Precision::INT8) {
        const uint8_t* raw = static_cast<const uint8_t*>(input.get_data());
        for (size_t i = 0; i < src_float.size(); i++)
            src_float[i] = static_cast<float>(raw[i]);
    } else if (input.precision == Precision::FP16) {
        Quantization::fp16_to_fp32(input.data_as<__fp16>(), src_float.data(), src_float.size());
    }

    std::vector<float> resized;
    const float* norm_input;

    if (dst_w != src_w || dst_h != src_h) {
        resized.resize(static_cast<size_t>(dst_w) * dst_h * ch);
        cactus_image_resize_float(src_float.data(), src_w, src_h,
                                   resized.data(), dst_w, dst_h, ch);
        norm_input = resized.data();
    } else {
        norm_input = src_float.data();
    }

    std::vector<float> normalized(static_cast<size_t>(dst_w) * dst_h * ch);
    cactus_image_normalize(norm_input, normalized.data(),
                            dst_w, dst_h, ch,
                            rescale, node.params.image_mean, node.params.image_std);

    cactus_image_to_patches(normalized.data(),
                             node.output_buffer.data_as<float>(),
                             dst_w, dst_h, ch, ps);
}
