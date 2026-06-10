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

bool Model::load_npu_source_encoder(const std::string& model_path) {
    auto encoder = npu::create_encoder();
    if (!encoder) return false;
    if (!encoder->load(model_path)) return false;
    if (!encoder->is_available()) return false;
    npu_source_encoder_ = std::move(encoder);
    CACTUS_LOG_INFO("model", "NPU source encoder loaded from: " << model_path);
    return true;
}

bool Model::source_encode_via_npu(const std::vector<uint32_t>& tokens) {
    if (!npu_source_encoder_ || !npu_source_encoder_->is_available() ||
        !source_encoder_ || !decoder_cross_kv_) {
        return false;
    }

    int ids_idx = input_index(*source_encoder_, "input_ids");
    if (ids_idx < 0) return false;
    size_t ids_node = static_cast<size_t>(source_encoder_->runtime_input_node_ids[ids_idx]);
    const auto& ids_desc = source_encoder_->graph->get_output_buffer(ids_node);
    if (tokens.size() > ids_desc.total_size) return false;

    std::vector<int> input_shape;
    input_shape.reserve(ids_desc.shape.size());
    for (size_t dim : ids_desc.shape) input_shape.push_back(static_cast<int>(dim));

    std::vector<int32_t> input_ids(ids_desc.total_size, 0);
    std::vector<int32_t> attention_mask(ids_desc.total_size, 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        input_ids[i] = static_cast<int32_t>(tokens[i]);
        attention_mask[i] = 1;
    }

    std::vector<npu::NPUNamedInput> npu_inputs;
    npu_inputs.push_back({
        "input_ids",
        input_ids.data(),
        npu::NPUNamedInput::DataType::INT32,
        input_shape,
    });
    if (input_index(*source_encoder_, "attention_mask") >= 0) {
        npu_inputs.push_back({
            "attention_mask",
            attention_mask.data(),
            npu::NPUNamedInput::DataType::INT32,
            input_shape,
        });
    }

    int hidden_idx = input_index(*decoder_cross_kv_, "encoder_hidden_states");
    if (hidden_idx < 0) return false;
    size_t hidden_node = static_cast<size_t>(decoder_cross_kv_->runtime_input_node_ids[hidden_idx]);
    const auto& hidden_desc = decoder_cross_kv_->graph->get_output_buffer(hidden_node);
    std::vector<__fp16> hidden(hidden_desc.total_size, __fp16(0));
    size_t written = npu_source_encoder_->encode_multimodal_input(
        npu_inputs,
        hidden.data(),
        "encoder_hidden_states");
    if (written == 0) return false;

    auto copy_fp16_to_component_input = [&](Component& comp, int input_idx, const __fp16* src, size_t src_elems) {
        if (input_idx < 0 || !src) return false;
        size_t node = static_cast<size_t>(comp.runtime_input_node_ids[static_cast<size_t>(input_idx)]);
        const auto& desc = comp.graph->get_output_buffer(node);
        auto& dst = comp.input_buffers[static_cast<size_t>(input_idx)];
        std::fill(dst.begin(), dst.end(), 0);
        const size_t elems = std::min(src_elems, desc.total_size);
        if (desc.precision == Precision::FP16) {
            std::memcpy(dst.data(), src, elems * sizeof(__fp16));
            return true;
        }
        if (desc.precision == Precision::FP32) {
            float* out = reinterpret_cast<float*>(dst.data());
            for (size_t i = 0; i < elems; ++i) out[i] = static_cast<float>(src[i]);
            return true;
        }
        CACTUS_LOG_WARN("model", "NPU source encoder output precision mismatch for encoder_hidden_states");
        return false;
    };

    if (!copy_fp16_to_component_input(*decoder_cross_kv_, hidden_idx, hidden.data(), written)) {
        return false;
    }

    auto fill_mask = [&](Component& comp) {
        int idx = input_index(comp, "encoder_attention_mask");
        if (idx < 0) return;
        std::fill(comp.input_buffers[static_cast<size_t>(idx)].begin(),
                  comp.input_buffers[static_cast<size_t>(idx)].end(),
                  0);
        for (size_t i = 0; i < tokens.size(); ++i) {
            write_int_input_at(comp, "encoder_attention_mask", i, 1);
        }
    };
    fill_mask(*decoder_cross_kv_);
    if (decoder_) fill_mask(*decoder_);

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
