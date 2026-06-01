#include "engine.h"
#include "cactus_graph.h"
#include "cactus_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cactus {
namespace engine {

bool Model::load_npu_audio_encoder(const std::string& model_path) {
    auto encoder = npu::create_encoder();
    if (!encoder) return false;
    if (!encoder->load(model_path)) return false;
    if (!encoder->is_available()) return false;
    npu_audio_encoder_ = std::move(encoder);
    CACTUS_LOG_INFO("model", "NPU audio encoder loaded from: " << model_path);
    return true;
}

bool Model::load_npu_vision_encoder(const std::string& model_path) {
    auto encoder = npu::create_encoder();
    if (!encoder) return false;
    if (!encoder->load(model_path)) return false;
    if (!encoder->is_available()) return false;
    npu_vision_encoder_ = std::move(encoder);
    CACTUS_LOG_INFO("model", "NPU vision encoder loaded from: " << model_path);
    return true;
}

bool Model::audio_encode_via_npu(const std::vector<float>& audio_features) {
    if (!npu_audio_encoder_ || !npu_audio_encoder_->is_available() || !audio_encoder_) {
        return false;
    }
    const std::vector<int> input_shape = npu_audio_encoder_->get_input_shape();
    if (input_shape.empty()) return false;

    size_t expected_elems = 1;
    for (int d : input_shape) {
        if (d <= 0) return false;
        expected_elems *= static_cast<size_t>(d);
    }
    if (audio_features.size() > expected_elems) return false;

    std::vector<__fp16> input_fp16(expected_elems, __fp16(0));
    for (size_t i = 0; i < audio_features.size(); ++i) {
        input_fp16[i] = static_cast<__fp16>(audio_features[i]);
    }

    const std::vector<int> output_shape = npu_audio_encoder_->get_output_shape();
    size_t output_elems = 1;
    for (int d : output_shape) {
        if (d <= 0) { output_elems = 0; break; }
        output_elems *= static_cast<size_t>(d);
    }
    if (output_elems == 0) {
        output_elems = npu_audio_encoder_->get_output_buffer_size();
    }
    std::vector<__fp16> output_fp16(output_elems, __fp16(0));

    size_t written = npu_audio_encoder_->encode(
        input_fp16.data(), output_fp16.data(), input_shape, "x", "encoded");
    if (written == 0) return false;

    for (size_t i = 0; i < audio_encoder_->output_node_ids.size()
                      && i < audio_encoder_->logical_outputs.size(); ++i) {
        const std::string& name = audio_encoder_->logical_outputs[i];
        size_t node_id = static_cast<size_t>(audio_encoder_->output_node_ids[i]);
        const auto& desc = audio_encoder_->graph->get_output_buffer(node_id);
        const size_t copy_bytes = std::min(desc.byte_size, written * sizeof(__fp16));
        auto& slot = media_features_[name];
        const size_t prev = slot.size();
        slot.resize(prev + copy_bytes);
        if (desc.precision == Precision::FP16) {
            std::memcpy(slot.data() + prev, output_fp16.data(), copy_bytes);
        } else if (desc.precision == Precision::FP32) {
            const size_t n = copy_bytes / sizeof(__fp16);
            float* dst = reinterpret_cast<float*>(slot.data() + prev);
            for (size_t k = 0; k < n; ++k) dst[k] = static_cast<float>(output_fp16[k]);
        } else {
            std::memcpy(slot.data() + prev, output_fp16.data(), copy_bytes);
        }
        auto shape_it = media_feature_shapes_.find(name);
        if (shape_it == media_feature_shapes_.end() || shape_it->second.empty()) {
            std::vector<size_t> shape;
            for (int d : output_shape) shape.push_back(static_cast<size_t>(d));
            media_feature_shapes_[name] = std::move(shape);
        }
        media_feature_precisions_[name] = desc.precision;
        break;
    }
    return true;
}

bool Model::vision_encode_via_npu(const std::vector<float>& pixel_values) {
    if (!npu_vision_encoder_ || !npu_vision_encoder_->is_available() || !vision_encoder_) {
        return false;
    }
    const std::vector<int> input_shape = npu_vision_encoder_->get_input_shape();
    if (input_shape.empty()) return false;

    size_t expected_elems = 1;
    for (int d : input_shape) {
        if (d <= 0) return false;
        expected_elems *= static_cast<size_t>(d);
    }
    if (pixel_values.size() > expected_elems) return false;

    std::vector<__fp16> input_fp16(expected_elems, __fp16(0));
    for (size_t i = 0; i < pixel_values.size(); ++i) {
        input_fp16[i] = static_cast<__fp16>(pixel_values[i]);
    }

    const std::vector<int> output_shape = npu_vision_encoder_->get_output_shape();
    size_t output_elems = 1;
    for (int d : output_shape) {
        if (d <= 0) { output_elems = 0; break; }
        output_elems *= static_cast<size_t>(d);
    }
    if (output_elems == 0) {
        output_elems = npu_vision_encoder_->get_output_buffer_size();
    }
    std::vector<__fp16> output_fp16(output_elems, __fp16(0));

    size_t written = npu_vision_encoder_->encode(
        input_fp16.data(), output_fp16.data(), input_shape, "x", "encoded");
    if (written == 0) return false;

    for (size_t i = 0; i < vision_encoder_->output_node_ids.size()
                      && i < vision_encoder_->logical_outputs.size(); ++i) {
        const std::string& name = vision_encoder_->logical_outputs[i];
        size_t node_id = static_cast<size_t>(vision_encoder_->output_node_ids[i]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(node_id);
        const Precision cpu_prec = desc.precision;
        const size_t cpu_byte_size = desc.byte_size;
        const size_t copy_elems = std::min(static_cast<size_t>(written),
                                           cpu_byte_size / sizeof(__fp16));
        auto& slot = media_features_[name];
        if (cpu_prec == Precision::FP16) {
            const size_t prev = slot.size();
            slot.resize(prev + copy_elems * sizeof(__fp16));
            std::memcpy(slot.data() + prev, output_fp16.data(), copy_elems * sizeof(__fp16));
        } else if (cpu_prec == Precision::FP32) {
            const size_t prev = slot.size();
            slot.resize(prev + copy_elems * sizeof(float));
            float* dst = reinterpret_cast<float*>(slot.data() + prev);
            for (size_t k = 0; k < copy_elems; ++k) dst[k] = static_cast<float>(output_fp16[k]);
        } else {
            const size_t prev = slot.size();
            slot.resize(prev + copy_elems * sizeof(__fp16));
            std::memcpy(slot.data() + prev, output_fp16.data(), copy_elems * sizeof(__fp16));
        }
        auto shape_it = media_feature_shapes_.find(name);
        std::vector<size_t> npu_shape;
        for (int d : output_shape) npu_shape.push_back(static_cast<size_t>(d));
        if (shape_it == media_feature_shapes_.end() || shape_it->second.empty()) {
            media_feature_shapes_[name] = npu_shape;
        } else if (npu_shape.size() >= 2 && shape_it->second.size() == npu_shape.size()) {
            shape_it->second[shape_it->second.size() - 2] += npu_shape[npu_shape.size() - 2];
        }
        media_feature_precisions_[name] = cpu_prec;
        break;
    }
    return true;
}


}
}
