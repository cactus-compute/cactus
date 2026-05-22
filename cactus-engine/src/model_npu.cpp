#include "engine.h"
#include "cactus_graph.h"
#include "cactus_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cactus {
namespace engine {

namespace {

struct NPUCacheHeader {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t reserved[3];
};
constexpr size_t kNPUCacheHeaderBytes = 64;
static_assert(sizeof(NPUCacheHeader) == kNPUCacheHeaderBytes, "NPUCacheHeader layout mismatch");

bool is_gemma_family(const std::string& family) {
    return family.find("gemma") != std::string::npos;
}

}  // namespace

bool Model::load_npu_prefill(const std::string& model_path) {
    auto prefill = npu::create_prefill();
    if (!prefill) return false;
    if (!prefill->load(model_path)) return false;
    if (!prefill->is_available()) return false;
    npu_prefill_ = std::move(prefill);
    CACTUS_LOG_INFO("model", "NPU prefill loaded: chunk=" << npu_prefill_->get_chunk_size()
                              << " hidden=" << npu_prefill_->get_hidden_dim()
                              << " layers=" << npu_prefill_->get_num_layers()
                              << " kv_heads=" << npu_prefill_->get_num_kv_heads()
                              << " head_dim=" << npu_prefill_->get_head_dim());
    return true;
}

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

    // Copy NPU output into the CPU audio_encoder's primary output buffer so
    // downstream code (dynamic walker / media_features_) sees the same path.
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
        break;  // audio encoder only has one logical output
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

    // Copy NPU output into the CPU vision_encoder's primary output buffer so
    // downstream code (dynamic walker / media_features_) sees the same path.
    for (size_t i = 0; i < vision_encoder_->output_node_ids.size()
                      && i < vision_encoder_->logical_outputs.size(); ++i) {
        const std::string& name = vision_encoder_->logical_outputs[i];
        size_t node_id = static_cast<size_t>(vision_encoder_->output_node_ids[i]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(node_id);
        const size_t copy_bytes = std::min(desc.byte_size, written * sizeof(__fp16));
        auto& slot = media_features_[name];
        slot.assign(copy_bytes, 0);
        if (desc.precision == Precision::FP16) {
            std::memcpy(slot.data(), output_fp16.data(), copy_bytes);
        } else if (desc.precision == Precision::FP32) {
            const size_t n = copy_bytes / sizeof(__fp16);
            slot.assign(n * sizeof(float), 0);
            float* dst = reinterpret_cast<float*>(slot.data());
            for (size_t k = 0; k < n; ++k) dst[k] = static_cast<float>(output_fp16[k]);
        } else {
            std::memcpy(slot.data(), output_fp16.data(), copy_bytes);
        }
        std::vector<size_t> shape;
        for (int d : output_shape) shape.push_back(static_cast<size_t>(d));
        media_feature_shapes_[name] = std::move(shape);
        media_feature_precisions_[name] = desc.precision;
        break;  // vision encoder has one logical output
    }
    return true;
}

bool Model::prefill_via_npu(const std::vector<uint32_t>& tokens) {
    if (!npu_prefill_ || !npu_prefill_->is_available() || !decoder_) {
        return false;
    }
    if (tokens.empty()) return true;

    const int chunk_size = npu_prefill_->get_chunk_size();
    if (chunk_size <= 0) return false;

    const size_t num_tokens = tokens.size();
    const size_t num_chunks = (num_tokens + chunk_size - 1) / chunk_size;

    std::vector<int32_t> chunk_ids(chunk_size, 0);
    std::vector<int32_t> chunk_positions(chunk_size, 0);

    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t start = c * static_cast<size_t>(chunk_size);
        const size_t actual_tokens = std::min(static_cast<size_t>(chunk_size), num_tokens - start);

        std::fill(chunk_ids.begin(), chunk_ids.end(), 0);
        std::fill(chunk_positions.begin(), chunk_positions.end(), 0);
        for (size_t i = 0; i < actual_tokens; ++i) {
            chunk_ids[i] = static_cast<int32_t>(tokens[start + i]);
            chunk_positions[i] = static_cast<int32_t>(cache_total_seq_len_ + start + i);
        }

        auto direct = npu_prefill_->prefill_chunk_tokens(chunk_ids, chunk_positions);
        if (!direct.valid) {
            CACTUS_LOG_WARN("model", "NPU prefill chunk " << c << " returned invalid result");
            return false;
        }
        write_npu_kv_to_cache(direct, actual_tokens);
    }
    cache_total_seq_len_ += num_tokens;
    (void)is_gemma_family(family_);  // family info still useful for future scaling work
    return true;
}

void Model::write_npu_kv_to_cache(const npu::NPUPrefillDirectResult& result, size_t actual_tokens) {
    if (!decoder_) return;
    if (actual_tokens == 0) return;

    const size_t n_layers = std::min(result.k_caches.size(), decoder_->cache_states.size());
    if (n_layers == 0) return;

    for (size_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        const auto& cs = decoder_->cache_states[layer_idx];
        const auto& k_ref = result.k_caches[layer_idx];
        const auto& v_ref = result.v_caches[layer_idx];
        if (!k_ref.data || !v_ref.data) continue;

        const int node_ids[2] = {cs.key_node_id, cs.value_node_id};
        const npu::NPUBufferRef refs[2] = {k_ref, v_ref};
        for (int side = 0; side < 2; ++side) {
            const int node_id = node_ids[side];
            if (node_id < 0) continue;
            const auto& desc = decoder_->graph->get_output_buffer(static_cast<size_t>(node_id));
            if (desc.byte_size <= kNPUCacheHeaderBytes || !desc.get_data()) continue;

            void* raw = decoder_->graph->get_output(static_cast<size_t>(node_id));
            if (!raw) continue;
            auto* hdr = static_cast<NPUCacheHeader*>(raw);

            const size_t kv_heads = hdr->num_kv_heads;
            const size_t hdim = hdr->head_dim;
            const size_t max_len = hdr->max_seq_len;
            const size_t sink = hdr->sink_size;
            if (kv_heads == 0 || hdim == 0 || max_len == 0) continue;

            const size_t current_len = hdr->current_seq_len;
            const size_t new_total = current_len + actual_tokens;
            const size_t window = max_len;
            const size_t token_elems = kv_heads * hdim;
            const __fp16* source = refs[side].data;

            if (desc.precision == Precision::FP16) {
                __fp16* base = reinterpret_cast<__fp16*>(static_cast<char*>(raw) + kNPUCacheHeaderBytes);
                if (new_total <= window) {
                    std::memcpy(base + current_len * token_elems, source,
                                actual_tokens * token_elems * sizeof(__fp16));
                    hdr->current_seq_len = new_total;
                } else if (actual_tokens >= window) {
                    const size_t skip = actual_tokens - window;
                    std::memcpy(base, source + skip * token_elems,
                                window * token_elems * sizeof(__fp16));
                    hdr->current_seq_len = window;
                } else {
                    const size_t sink_eff = std::min(sink, window);
                    const size_t remaining = (actual_tokens + sink_eff < window)
                                                 ? window - sink_eff - actual_tokens : 0;
                    const size_t shift_src = (current_len > remaining) ? current_len - remaining : 0;
                    if (remaining > 0 && shift_src > sink_eff) {
                        std::memmove(base + sink_eff * token_elems,
                                     base + shift_src * token_elems,
                                     remaining * token_elems * sizeof(__fp16));
                    }
                    const size_t append_offset = window - actual_tokens;
                    std::memcpy(base + append_offset * token_elems, source,
                                actual_tokens * token_elems * sizeof(__fp16));
                    hdr->current_seq_len = window;
                }
            } else {
                int8_t* int8_base = reinterpret_cast<int8_t*>(static_cast<char*>(raw) + kNPUCacheHeaderBytes);
                const size_t num_groups = (hdim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
                const size_t scale_stride = kv_heads * num_groups;
                float* scale_base = reinterpret_cast<float*>(static_cast<char*>(raw) + kNPUCacheHeaderBytes +
                                                              max_len * kv_heads * hdim);
                if (new_total <= window) {
                    cactus_quantize_kv_fp16_to_int8(
                        source,
                        int8_base + current_len * token_elems,
                        scale_base + current_len * scale_stride,
                        actual_tokens, kv_heads, hdim);
                    hdr->current_seq_len = new_total;
                } else if (actual_tokens >= window) {
                    const size_t skip = actual_tokens - window;
                    cactus_quantize_kv_fp16_to_int8(
                        source + skip * token_elems,
                        int8_base,
                        scale_base,
                        window, kv_heads, hdim);
                    hdr->current_seq_len = window;
                } else {
                    const size_t sink_eff = std::min(sink, window);
                    const size_t remaining = (actual_tokens + sink_eff < window)
                                                 ? window - sink_eff - actual_tokens : 0;
                    const size_t shift_src = (current_len > remaining) ? current_len - remaining : 0;
                    if (remaining > 0 && shift_src > sink_eff) {
                        std::memmove(int8_base + sink_eff * token_elems,
                                     int8_base + shift_src * token_elems,
                                     remaining * token_elems);
                        std::memmove(scale_base + sink_eff * scale_stride,
                                     scale_base + shift_src * scale_stride,
                                     remaining * scale_stride * sizeof(float));
                    }
                    const size_t append_offset = window - actual_tokens;
                    cactus_quantize_kv_fp16_to_int8(
                        source,
                        int8_base + append_offset * token_elems,
                        scale_base + append_offset * scale_stride,
                        actual_tokens, kv_heads, hdim);
                    hdr->current_seq_len = window;
                }
            }
        }
    }
}

}  // namespace engine
}  // namespace cactus
