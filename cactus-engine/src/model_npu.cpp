#include "engine.h"
#include "cactus_graph.h"
#include "cactus_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cactus {
namespace engine {

bool Model::load_npu_audio_encoder(const std::string& model_path, const std::string& compute_units) {
    auto encoder = npu::create_encoder();
    if (!encoder) return false;
    if (!encoder->load(model_path, compute_units)) return false;
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
        const size_t elem_size = (desc.precision == Precision::FP32) ? sizeof(float) : sizeof(__fp16);
        const size_t copy_elems = std::min(static_cast<size_t>(written), desc.byte_size / elem_size);
        auto& slot = media_features_[name];
        if (desc.precision == Precision::FP32) {
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

bool Model::vision_encode_via_npu(const std::vector<float>& pixel_values,
                                  const std::vector<int64_t>* pixel_position_ids) {
    if (!npu_vision_encoder_ || !npu_vision_encoder_->is_available() || !vision_encoder_) {
        return false;
    }
    const std::vector<int> input_shape = npu_vision_encoder_->get_input_shape_for("x");
    if (input_shape.empty()) return false;

    size_t expected_elems = 1;
    for (int d : input_shape) {
        if (d <= 0) return false;
        expected_elems *= static_cast<size_t>(d);
    }
    if (pixel_values.size() > expected_elems) return false;

    const bool package_takes_positions = npu_vision_encoder_->has_input("pixel_position_ids");
    if (package_takes_positions != (pixel_position_ids != nullptr)) {
        CACTUS_LOG_WARN("model", "NPU vision encoder and pixel_position_ids mismatch; "
            "falling back to CPU vision encoder (re-transpile with --npu to fix)");
        return false;
    }

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

    size_t written = 0;
    if (package_takes_positions) {
        const std::vector<int> pos_shape =
            npu_vision_encoder_->get_input_shape_for("pixel_position_ids");
        if (pos_shape.empty()) return false;
        size_t pos_elems = 1;
        for (int d : pos_shape) {
            if (d <= 0) return false;
            pos_elems *= static_cast<size_t>(d);
        }
        if (pixel_position_ids->size() > pos_elems) return false;

        std::vector<int32_t> positions_i32(pos_elems, -1);
        for (size_t i = 0; i < pixel_position_ids->size(); ++i) {
            positions_i32[i] = static_cast<int32_t>((*pixel_position_ids)[i]);
        }

        const std::vector<npu::NPUNamedInput> inputs = {
            {"x", input_fp16.data(), npu::NPUNamedInput::DataType::FP16, input_shape},
            {"pixel_position_ids", positions_i32.data(), npu::NPUNamedInput::DataType::INT32, pos_shape},
        };
        written = npu_vision_encoder_->encode_multimodal_input(
            inputs, output_fp16.data(), "encoded");
    } else {
        written = npu_vision_encoder_->encode(
            input_fp16.data(), output_fp16.data(), input_shape, "x", "encoded");
    }
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

bool Model::lfm2_vl_use_npu_vision() const {
    return npu_vision_encoder_ != nullptr
        && npu_vision_encoder_->is_available()
        && npu_vision_encoder_->has_input("positional_embeddings");
}

bool Model::lfm2_vl_encode_tile_npu(const float* pixel_values, const int64_t* mask,
                                    const float* pos_embeds, size_t max_patches,
                                    int dim, size_t patch_dim, std::vector<float>& enc_out) {
    if (!npu_vision_encoder_ || !npu_vision_encoder_->is_available()) return false;

    const std::vector<int> x_shape = npu_vision_encoder_->get_input_shape_for("x");
    const std::vector<int> m_shape = npu_vision_encoder_->get_input_shape_for("pixel_attention_mask");
    const std::vector<int> p_shape = npu_vision_encoder_->get_input_shape_for("positional_embeddings");
    if (x_shape.empty() || p_shape.empty()) return false;

    auto elem_count = [](const std::vector<int>& s) -> size_t {
        size_t e = 1;
        for (int d : s) { if (d <= 0) return 0; e *= static_cast<size_t>(d); }
        return e;
    };
    const size_t x_elems = elem_count(x_shape);
    const size_t p_elems = elem_count(p_shape);
    const size_t m_elems = m_shape.empty() ? 0 : elem_count(m_shape);
    const size_t pv_count = max_patches * patch_dim;
    const size_t pe_count = max_patches * static_cast<size_t>(dim);
    if (x_elems < pv_count || p_elems < pe_count) return false;

    std::vector<__fp16> x_fp16(x_elems, __fp16(0));
    for (size_t i = 0; i < pv_count; ++i) x_fp16[i] = static_cast<__fp16>(pixel_values[i]);
    std::vector<__fp16> p_fp16(p_elems, __fp16(0));
    for (size_t i = 0; i < pe_count; ++i) p_fp16[i] = static_cast<__fp16>(pos_embeds[i]);
    std::vector<int32_t> m_i32(m_elems ? m_elems : max_patches, 0);
    for (size_t i = 0; i < max_patches && i < m_i32.size(); ++i) m_i32[i] = static_cast<int32_t>(mask[i]);

    const std::vector<int> out_shape = npu_vision_encoder_->get_output_shape();
    size_t out_elems = elem_count(out_shape);
    if (out_elems == 0) out_elems = npu_vision_encoder_->get_output_buffer_size();
    std::vector<__fp16> out_fp16(out_elems, __fp16(0));

    std::vector<npu::NPUNamedInput> inputs = {
        {"x", x_fp16.data(), npu::NPUNamedInput::DataType::FP16, x_shape},
        {"positional_embeddings", p_fp16.data(), npu::NPUNamedInput::DataType::FP16, p_shape},
    };
    if (!m_shape.empty()) {
        inputs.push_back({"pixel_attention_mask", m_i32.data(), npu::NPUNamedInput::DataType::INT32, m_shape});
    }

    size_t written = npu_vision_encoder_->encode_multimodal_input(inputs, out_fp16.data(), "encoded");
    if (written == 0) return false;

    const size_t copy = std::min(enc_out.size(), static_cast<size_t>(written));
    for (size_t i = 0; i < copy; ++i) enc_out[i] = static_cast<float>(out_fp16[i]);
    return true;
}

}
}
