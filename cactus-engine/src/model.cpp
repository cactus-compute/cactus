#include "engine.h"
#include "cactus_graph.h"
#include "cactus_kernels.h"

#define PICOJSON_USE_INT64
#include "picojson.h"

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>

namespace cactus {
namespace engine {

std::vector<uint32_t> parse_config_uint_list(const std::string& value) {
    std::vector<uint32_t> numbers;
    size_t pos = 0;
    while (pos < value.size()) {
        while (pos < value.size() && !std::isdigit(static_cast<unsigned char>(value[pos]))) {
            ++pos;
        }
        if (pos >= value.size()) break;
        size_t end = pos;
        while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end]))) {
            ++end;
        }
        numbers.push_back(static_cast<uint32_t>(std::stoul(value.substr(pos, end - pos))));
        pos = end;
    }
    return numbers;
}

float read_scalar_value(Precision precision, const uint8_t* data, size_t index) {
    const uint8_t* ptr = data + PrecisionTraits::byte_offset_of(precision, index);
    switch (precision) {
        case Precision::FP32:
            return *reinterpret_cast<const float*>(ptr);
        case Precision::FP16:
            return static_cast<float>(*reinterpret_cast<const __fp16*>(ptr));
        case Precision::INT8:
            return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
        default:
            return 0.0f;
    }
}

void write_scalar_value(Precision precision, uint8_t* data, size_t index, float value) {
    uint8_t* ptr = data + PrecisionTraits::byte_offset_of(precision, index);
    switch (precision) {
        case Precision::FP32:
            *reinterpret_cast<float*>(ptr) = value;
            break;
        case Precision::FP16:
            *reinterpret_cast<__fp16*>(ptr) = static_cast<__fp16>(value);
            break;
        case Precision::INT8:
            *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(value);
            break;
        default:
            break;
    }
}

bool copy_component_tensor(CactusGraph& source_graph,
                           const BufferDesc& src_desc,
                           size_t src_node,
                           const BufferDesc& dst_desc,
                           std::vector<uint8_t>& dst_buffer,
                           size_t dst_element_offset,
                           size_t element_count,
                           const std::string& name) {
    const auto* src_ptr = static_cast<const uint8_t*>(source_graph.get_output(src_node));
    if (src_desc.precision == dst_desc.precision) {
        size_t dst_offset = PrecisionTraits::byte_offset_of(dst_desc.precision, dst_element_offset);
        std::memcpy(
            dst_buffer.data() + dst_offset,
            src_ptr,
            PrecisionTraits::packed_size_of(src_desc.precision, element_count));
        return true;
    }
    if (name != "position_ids" && name != "attention_mask") return false;
    for (size_t i = 0; i < element_count; ++i) {
        write_scalar_value(
            dst_desc.precision,
            dst_buffer.data(),
            dst_element_offset + i,
            read_scalar_value(src_desc.precision, src_ptr, i));
    }
    return true;
}

struct CrossKVCacheMetadata {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t reserved[3];
};

static_assert(sizeof(CrossKVCacheMetadata) == 64, "CrossKVCacheMetadata must be 64 bytes");

size_t cross_kv_cache_buffer_size(size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    return sizeof(CrossKVCacheMetadata) + max_seq * kv_heads * head_dim +
           max_seq * kv_heads * num_groups * sizeof(float);
}

bool write_cross_kv_cache_buffer(const BufferDesc& src_desc,
                                 const void* src_ptr,
                                 const BufferDesc& dst_desc,
                                 std::vector<uint8_t>& dst_buffer,
                                 size_t source_len,
                                 const std::string& name) {
    if (src_desc.precision != Precision::FP16 || dst_desc.precision != Precision::INT8) return false;
    if (src_desc.shape.size() != 4) return false;

    const size_t max_seq = src_desc.shape[1];
    const size_t kv_heads = src_desc.shape[2];
    const size_t head_dim = src_desc.shape[3];
    const size_t expected_bytes = cross_kv_cache_buffer_size(max_seq, kv_heads, head_dim);
    if (dst_buffer.size() < expected_bytes || dst_desc.byte_size < expected_bytes) {
        CACTUS_LOG_ERROR("model", "cross-KV cache input buffer is too small for " << name);
        return false;
    }

    source_len = std::min(source_len, max_seq);
    std::fill(dst_buffer.begin(), dst_buffer.end(), 0);
    auto* meta = reinterpret_cast<CrossKVCacheMetadata*>(dst_buffer.data());
    meta->current_seq_len = static_cast<uint64_t>(source_len);
    meta->max_seq_len = static_cast<uint64_t>(max_seq);
    meta->num_kv_heads = static_cast<uint64_t>(kv_heads);
    meta->head_dim = static_cast<uint64_t>(head_dim);
    meta->sink_size = 0;

    if (source_len == 0) return true;
    auto* int8_base = reinterpret_cast<int8_t*>(dst_buffer.data() + sizeof(CrossKVCacheMetadata));
    const size_t int8_bytes = max_seq * kv_heads * head_dim;
    auto* scale_base = reinterpret_cast<float*>(dst_buffer.data() + sizeof(CrossKVCacheMetadata) + int8_bytes);
    cactus_quantize_kv_fp16_to_int8(
        static_cast<const __fp16*>(src_ptr),
        int8_base,
        scale_base,
        source_len,
        kv_heads,
        head_dim);
    return true;
}

void ConvCache::init(size_t layers, size_t hidden_dim, size_t window_len, Precision model_precision) {
    num_layers = layers;
    hidden_size = hidden_dim;
    window_size = window_len;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    size_t state_bytes = window_size * hidden_size * element_size;
    layer_states.resize(num_layers);
    for (auto& state : layer_states) {
        state.data.resize(state_bytes);
        std::memset(state.data.data(), 0, state_bytes);
        state.head = 0;
        state.count = 0;
    }
}

ConvCache::CircularView ConvCache::get_window(size_t layer) const {
    CircularView view{};
    if (layer >= num_layers) {
        return view;
    }

    const auto& state = layer_states[layer];
    if (state.count == 0) {
        return view;
    }

    size_t stride = hidden_size * element_size;
    if (state.count < window_size) {
        view.ptr1 = state.data.data();
        view.len1 = state.count;
        view.total_len = state.count;
        return view;
    }

    view.ptr1 = state.data.data();
    view.len1 = state.head;
    view.ptr2 = state.data.data() + state.head * stride;
    view.len2 = window_size - state.head;
    view.total_len = window_size;
    return view;
}

void ConvCache::update(CactusGraph* gb, size_t layer, const size_t bx_node) {
    if (layer >= num_layers || !bx_node || window_size == 0 || hidden_size == 0) {
        return;
    }

    auto& state = layer_states[layer];
    const void* output_ptr = gb->get_output(bx_node);
    if (!output_ptr) {
        return;
    }

    const auto& buffer = gb->get_output_buffer(bx_node);
    const size_t stride_bytes = hidden_size * element_size;

    size_t rows = 1;
    if (!buffer.shape.empty()) {
        rows = buffer.shape.size() == 1 ? 1 : buffer.shape[0];
    }

    if (buffer.total_size > 0 && hidden_size > 0) {
        size_t inferred = buffer.total_size / hidden_size;
        if (inferred > 0) {
            rows = inferred;
        }
    }

    if (rows == 0) {
        return;
    }

    size_t copy_rows = std::min(rows, window_size);
    size_t start_row = rows > window_size ? rows - window_size : 0;
    const auto* src = static_cast<const uint8_t*>(output_ptr) + start_row * stride_bytes;

    for (size_t i = 0; i < copy_rows; ++i) {
        std::memcpy(state.data.data() + state.head * stride_bytes, src + i * stride_bytes, stride_bytes);
        state.head = (state.head + 1) % window_size;
        if (state.count < window_size) {
            ++state.count;
        }
    }
}

void ConvCache::reset() {
    for (auto& state : layer_states) {
        std::fill(state.data.begin(), state.data.end(), 0);
        state.head = 0;
        state.count = 0;
    }
}


namespace fs = std::filesystem;

Model::Model() : config_() {}

Model::Model(const Config& config) : config_(config) {}

Model::~Model() = default;

namespace {

bool read_exact(std::ifstream& in, void* data, size_t bytes) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(in.gcount()) == bytes;
}

bool read_float_vector(std::ifstream& in, std::vector<float>& out, size_t count) {
    out.resize(count);
    return read_exact(in, out.data(), count * sizeof(float));
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

} // namespace

bool Model::load_handoff_probe() {
    fs::path path = fs::path(bundle_dir_) / "handoff_probe.bin";
    if (!fs::exists(path)) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[8] = {};
    uint32_t version = 0;
    if (!read_exact(in, magic, sizeof(magic)) || std::string(magic, sizeof(magic)) != std::string("CHP10P6\0", 8)) {
        CACTUS_LOG_WARN("cloud_handoff", "Ignoring invalid handoff probe header at " << path);
        return false;
    }
    if (!read_exact(in, &version, sizeof(version))
        || !read_exact(in, &handoff_probe_feat_dim_, sizeof(handoff_probe_feat_dim_))
        || !read_exact(in, &handoff_probe_t_h_, sizeof(handoff_probe_t_h_))
        || !read_exact(in, &handoff_probe_h1_, sizeof(handoff_probe_h1_))
        || !read_exact(in, &handoff_probe_h2_, sizeof(handoff_probe_h2_))) {
        CACTUS_LOG_WARN("cloud_handoff", "Ignoring truncated handoff probe header at " << path);
        return false;
    }
    if (version != 1 || handoff_probe_feat_dim_ == 0 || handoff_probe_t_h_ == 0
        || handoff_probe_h1_ == 0 || handoff_probe_h2_ == 0) {
        CACTUS_LOG_WARN("cloud_handoff", "Ignoring unsupported handoff probe metadata at " << path);
        return false;
    }

    const size_t feat = handoff_probe_feat_dim_;
    const size_t th = handoff_probe_t_h_;
    const size_t h1 = handoff_probe_h1_;
    const size_t h2 = handoff_probe_h2_;
    bool ok = true;
    ok = ok && read_float_vector(in, handoff_probe_norm_weight_, feat);
    ok = ok && read_float_vector(in, handoff_probe_norm_bias_, feat);
    ok = ok && read_float_vector(in, handoff_probe_proj_weight_, th * feat);
    ok = ok && read_float_vector(in, handoff_probe_proj_bias_, th);
    ok = ok && read_float_vector(in, handoff_probe_attn_query_, th);
    ok = ok && read_float_vector(in, handoff_probe_head0_weight_, h1 * th);
    ok = ok && read_float_vector(in, handoff_probe_head0_bias_, h1);
    ok = ok && read_float_vector(in, handoff_probe_head2_weight_, h2 * h1);
    ok = ok && read_float_vector(in, handoff_probe_head2_bias_, h2);
    ok = ok && read_float_vector(in, handoff_probe_head4_weight_, h2);
    ok = ok && read_float_vector(in, handoff_probe_head4_bias_, 1);
    if (!ok) {
        CACTUS_LOG_WARN("cloud_handoff", "Ignoring truncated handoff probe weights at " << path);
        return false;
    }

    handoff_probe_hidden_.clear();
    handoff_probe_loaded_ = true;
    CACTUS_LOG_INFO("cloud_handoff", "Loaded handoff probe from " << path);
    return true;
}

bool Model::has_handoff_probe_rollout() const {
    return handoff_probe_loaded_
        && handoff_probe_feat_dim_ > 0
        && handoff_probe_hidden_.size() >= static_cast<size_t>(handoff_probe_feat_dim_);
}

void Model::maybe_capture_handoff_probe_hidden(const Component& comp, const std::string& output_name) {
    if (!handoff_probe_loaded_ || handoff_probe_feat_dim_ == 0) return;
    int idx = output_index(comp, output_name);
    if (idx < 0 || static_cast<size_t>(idx) >= comp.output_node_ids.size()) return;
    size_t node = static_cast<size_t>(comp.output_node_ids[idx]);
    const auto& desc = comp.graph->get_output_buffer(node);
    if (desc.total_size < handoff_probe_feat_dim_) return;
    if (!desc.shape.empty() &&
        static_cast<size_t>(desc.shape.back()) != static_cast<size_t>(handoff_probe_feat_dim_)) return;
    size_t rows = desc.total_size / handoff_probe_feat_dim_;
    if (rows == 0) return;
    const auto* data = static_cast<const uint8_t*>(comp.graph->get_output(node));
    if (!data) return;
    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * static_cast<size_t>(handoff_probe_feat_dim_);
        for (size_t i = 0; i < handoff_probe_feat_dim_; ++i) {
            handoff_probe_hidden_.push_back(read_scalar_value(desc.precision, data, base + i));
        }
    }
}

float Model::handoff_probe_wrong_probability() const {
    if (!has_handoff_probe_rollout()) return std::numeric_limits<float>::quiet_NaN();

    const size_t feat = handoff_probe_feat_dim_;
    const size_t th = handoff_probe_t_h_;
    const size_t h1 = handoff_probe_h1_;
    const size_t h2 = handoff_probe_h2_;
    const size_t tokens = std::min<size_t>(handoff_probe_hidden_.size() / feat, 1024);
    if (tokens == 0) return std::numeric_limits<float>::quiet_NaN();

    std::vector<float> projected(tokens * th);
    std::vector<float> scores(tokens);
    for (size_t t = 0; t < tokens; ++t) {
        const float* x = handoff_probe_hidden_.data() + t * feat;
        double mean = 0.0;
        for (size_t i = 0; i < feat; ++i) mean += x[i];
        mean /= static_cast<double>(feat);
        double var = 0.0;
        for (size_t i = 0; i < feat; ++i) {
            double d = static_cast<double>(x[i]) - mean;
            var += d * d;
        }
        var /= static_cast<double>(feat);
        float inv_std = static_cast<float>(1.0 / std::sqrt(var + 1e-5));

        for (size_t j = 0; j < th; ++j) {
            double acc = handoff_probe_proj_bias_[j];
            const float* w = handoff_probe_proj_weight_.data() + j * feat;
            for (size_t i = 0; i < feat; ++i) {
                float xn = (x[i] - static_cast<float>(mean)) * inv_std;
                xn = xn * handoff_probe_norm_weight_[i] + handoff_probe_norm_bias_[i];
                acc += static_cast<double>(w[i]) * xn;
            }
            float u = relu(static_cast<float>(acc));
            projected[t * th + j] = u;
            scores[t] += u * handoff_probe_attn_query_[j];
        }
        scores[t] /= std::sqrt(static_cast<float>(th));
    }

    float max_score = *std::max_element(scores.begin(), scores.end());
    double denom = 0.0;
    for (float s : scores) denom += std::exp(static_cast<double>(s - max_score));
    std::vector<float> pooled(th, 0.0f);
    for (size_t t = 0; t < tokens; ++t) {
        float alpha = static_cast<float>(std::exp(static_cast<double>(scores[t] - max_score)) / denom);
        const float* u = projected.data() + t * th;
        for (size_t j = 0; j < th; ++j) pooled[j] += alpha * u[j];
    }

    std::vector<float> y1(h1);
    for (size_t i = 0; i < h1; ++i) {
        double acc = handoff_probe_head0_bias_[i];
        const float* w = handoff_probe_head0_weight_.data() + i * th;
        for (size_t j = 0; j < th; ++j) acc += static_cast<double>(w[j]) * pooled[j];
        y1[i] = relu(static_cast<float>(acc));
    }
    std::vector<float> y2(h2);
    for (size_t i = 0; i < h2; ++i) {
        double acc = handoff_probe_head2_bias_[i];
        const float* w = handoff_probe_head2_weight_.data() + i * h1;
        for (size_t j = 0; j < h1; ++j) acc += static_cast<double>(w[j]) * y1[j];
        y2[i] = relu(static_cast<float>(acc));
    }
    double logit = handoff_probe_head4_bias_[0];
    for (size_t j = 0; j < h2; ++j) logit += static_cast<double>(handoff_probe_head4_weight_[j]) * y2[j];
    return static_cast<float>(1.0 / (1.0 + std::exp(-logit)));
}

bool Model::init(const std::string& bundle_dir, size_t context_size,
                 const std::string& /*system_prompt*/, bool /*do_warmup*/) {
    if (initialized_) return true;
    bundle_dir_ = bundle_dir;

    if (!config_.from_json(bundle_dir + "/config.txt")) {
        CACTUS_LOG_ERROR("model", "Failed to load config.txt from: " << bundle_dir);
        return false;
    }
    apply_kv_compress_env_override();
    if (!load_manifest()) {
        CACTUS_LOG_ERROR("model", "Failed to load bundle manifest from: " << bundle_dir);
        return false;
    }
    if (!setup_tokenizer()) {
        CACTUS_LOG_ERROR("model", "Tokenizer init failed for bundle: " << bundle_dir);
        return false;
    }
    const bool is_text_embedding =
        components_.count("text_embedding")
        && !components_.count("decoder")
        && !components_.count("decoder_step")
        && !components_.count("lm_encoder_step");
    if (is_text_embedding) {
        cache_max_seq_len_ = context_size;
        initialized_ = true;
        return true;
    }
    std::string encoder_name;
    std::string decoder_name;
    std::string source_encoder_name;
    std::string decoder_cross_kv_name;
    std::unordered_set<std::string> required_components;
    for (const auto& [name, comp] : components_) {
        auto route_it = comp.metadata.find("runtime_route");
        if (route_it == comp.metadata.end() || route_it->second != "encoder_cross_kv_decoder_step") {
            continue;
        }
        auto role_it = comp.metadata.find("runtime_role");
        if (role_it == comp.metadata.end()) {
            continue;
        }
        if (role_it->second == "source_encoder") {
            source_encoder_name = name;
            auto source_kind_it = comp.metadata.find("source_kind");
            if (source_kind_it != comp.metadata.end()) {
                encoder_cross_kv_source_kind_ = source_kind_it->second;
            }
        } else if (role_it->second == "decoder_cross_kv") {
            decoder_cross_kv_name = name;
        } else if (role_it->second == "decoder_step") {
            decoder_name = name;
        }
    }
    const bool has_metadata_encoder_cross_kv_route =
        !source_encoder_name.empty() &&
        !decoder_cross_kv_name.empty() &&
        !decoder_name.empty();
    bool has_chunked_prefill = components_.count("lm_encoder_step")
        && components_.count("decoder_media_step")
        && components_.count("lm_encoder_text_chunk")
        && components_.count("decoder_prefill_chunk");
    if (has_metadata_encoder_cross_kv_route) {
        decode_route_ = DecodeRoute::ENCODER_CROSS_KV_STEP;
        required_components = {source_encoder_name, decoder_cross_kv_name, decoder_name};
    } else if (has_chunked_prefill) {
        encoder_name = "lm_encoder_step";
        decoder_name = "decoder_media_step";
        decode_route_ = DecodeRoute::CACHED_STEP;
        required_components = {
            encoder_name,
            decoder_name,
            "lm_encoder_text_chunk",
            "decoder_prefill_chunk",
        };
    } else if (components_.count("decoder_step")
        && input_index(components_.at("decoder_step"), "input_ids") >= 0
        && input_index(components_.at("decoder_step"), "position_ids") >= 0) {
        decoder_name = "decoder_step";
        decode_route_ = DecodeRoute::DIRECT_DECODER_STEP;
        required_components = {decoder_name};
    } else if (components_.count("lm_encoder_step") && components_.count("decoder_step")) {
        encoder_name = "lm_encoder_step";
        decoder_name = "decoder_step";
        decode_route_ = DecodeRoute::CACHED_STEP;
        required_components = {encoder_name, decoder_name};
        if (components_.count("decoder_prefill_chunk")) {
            required_components.insert("decoder_prefill_chunk");
        }
        if (components_.count("lm_encoder_text_chunk")) {
            required_components.insert("lm_encoder_text_chunk");
        }
    } else if (components_.count("text_lm_encoder") && components_.count("decoder")) {
        encoder_name = "text_lm_encoder";
        decoder_name = "decoder";
        decode_route_ = DecodeRoute::FULL_CONTEXT_TEXT;
        required_components = {encoder_name, decoder_name};
    } else if (components_.count("audio_encoder") &&
               (components_.count("decoder") || components_.count("decoder_joint"))) {
        decoder_name = components_.count("decoder") ? "decoder" : "decoder_joint";
        decode_route_ = DecodeRoute::DIRECT_DECODER_STEP;
        required_components = {"audio_encoder", decoder_name};
    } else {
        CACTUS_LOG_ERROR("model", "Bundle missing required components: need lm_encoder_step+decoder_step (LM), text_lm_encoder+decoder, audio_encoder+decoder (transcription), or source/audio_encoder+decoder_cross_kv+decoder_step");
        return false;
    }
    for (const auto& optional : {
             "vision_encoder",
             "audio_encoder",
             "lm_encoder_media_step",
             "decoder_prefill_chunk",
             "decoder_embed_chunk",
             "lm_encoder",
         }) {
        if (components_.count(optional)) {
            required_components.insert(optional);
        }
    }
    if (!load_components(required_components)) return false;
    if (!encoder_name.empty()) encoder_ = &components_.at(encoder_name);
    if (!decoder_name.empty()) decoder_ = &components_.at(decoder_name);
    if (!source_encoder_name.empty()) source_encoder_ = &components_.at(source_encoder_name);
    if (!decoder_cross_kv_name.empty()) decoder_cross_kv_ = &components_.at(decoder_cross_kv_name);
    if (components_.count("decoder_prefill_chunk") && components_.at("decoder_prefill_chunk").graph) {
        decoder_prefill_ = &components_.at("decoder_prefill_chunk");
        decoder_prefill_chunk_ = decoder_prefill_;
    } else if (decode_route_ != DecodeRoute::ENCODER_CROSS_KV_STEP && !components_.count("audio_encoder")) {
        CACTUS_LOG_WARN("model", "Bundle has no decoder_prefill_chunk component; prompts will prefill token-by-token (prefill speed ~= decode speed).");
    }
    if (decoder_ && decoder_->graph) decoder_->graph->prewarm_metal_quant_weights();
    if (decoder_prefill_ && decoder_prefill_->graph) decoder_prefill_->graph->prewarm_metal_quant_weights();
    if (components_.count("decoder_embed_chunk") && components_.at("decoder_embed_chunk").graph) {
        decoder_embed_ = &components_.at("decoder_embed_chunk");
    }
    if (components_.count("lm_encoder_text_chunk") && components_.at("lm_encoder_text_chunk").graph) {
        prefill_encoder_ = &components_.at("lm_encoder_text_chunk");
    }
    if (const char* env = std::getenv("CACTUS_DISABLE_PREFILL_TAIL_PAD")) {
        prefill_tail_pad_disabled_ = std::atoi(env) != 0;
    }
    vision_encoder_ = components_.count("vision_encoder") ? &components_.at("vision_encoder") : nullptr;
    vision_projector_ = components_.count("vision_projector") ? &components_.at("vision_projector") : nullptr;
    audio_encoder_ = components_.count("audio_encoder") ? &components_.at("audio_encoder") : nullptr;
    lm_encoder_media_step_ = components_.count("lm_encoder_media_step") ? &components_.at("lm_encoder_media_step") : nullptr;
    lm_encoder_ = components_.count("lm_encoder") ? &components_.at("lm_encoder") : nullptr;
    lm_encoder_text_chunk_ = components_.count("lm_encoder_text_chunk") ? &components_.at("lm_encoder_text_chunk") : nullptr;
    lm_encoder_media_chunk_ = components_.count("lm_encoder_media_chunk") ? &components_.at("lm_encoder_media_chunk") : nullptr;
    std::vector<Component*> to_bind = {
        encoder_,
        source_encoder_,
        prefill_encoder_,
        decoder_,
        decoder_cross_kv_,
        decoder_prefill_,
        vision_encoder_,
        vision_projector_,
        audio_encoder_,
        lm_encoder_media_step_,
        lm_encoder_,
        lm_encoder_text_chunk_,
        lm_encoder_media_chunk_,
    };
    std::unordered_set<Component*> bound;
    for (Component* comp : to_bind) {
        if (!comp || !comp->graph || bound.count(comp)) continue;
        if (!bind_runtime_buffers(*comp)) return false;
        bound.insert(comp);
    }

    if (decoder_prefill_ && decoder_prefill_->graph) {
        try {
            if (prefill_encoder_ && prefill_encoder_->graph) {
                for (auto& b : prefill_encoder_->input_buffers) std::fill(b.begin(), b.end(), 0);
                prefill_encoder_->graph->execute();
                reset_component_cache_states(*prefill_encoder_);
            }
            for (auto& b : decoder_prefill_->input_buffers) std::fill(b.begin(), b.end(), 0);
            decoder_prefill_->graph->execute();
            reset_component_cache_states(*decoder_prefill_);
        } catch (const std::exception& e) {
            CACTUS_LOG_WARN("model", std::string("prefill warmup skipped: ") + e.what());
        }
    }

    if (vision_encoder_ && tokenizer_ && !vision_encoder_->output_node_ids.empty()) {
        size_t out_node = static_cast<size_t>(vision_encoder_->output_node_ids[0]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(out_node);
        size_t n = 0;
        if (desc.shape.size() >= 3) n = desc.shape[desc.shape.size() - 2];
        else if (desc.shape.size() >= 2) n = desc.shape[0];
        if (n > 0) tokenizer_->set_image_soft_token_count(n);
    }

    if (family_ == "lfm2_vl" && tokenizer_) {
        tokenizer_->set_lfm2_vision_config(config_);
    }

    cache_max_seq_len_ = context_size;

    if (load_handoff_probe()) {
        const bool is_parakeet = config_.model_type == Config::ModelType::PARAKEET_TDT;
        const Component* probe_comp = is_parakeet ? audio_encoder_ : decoder_;
        const char* probe_output = is_parakeet ? "encoder_hidden_states" : "probe_hidden";
        if (!probe_comp || output_index(*probe_comp, probe_output) < 0) {
            CACTUS_LOG_WARN("cloud_handoff", "Handoff probe is packaged, but its probe input is not exposed; "
                "reconvert to enable probe-based handoff");
            handoff_probe_loaded_ = false;
        }
    }

    initialized_ = true;
    return true;
}

bool Model::load_manifest() {
    std::ifstream in(fs::path(bundle_dir_) / "components" / "manifest.json");
    if (!in.is_open()) return false;
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty() || !root.is<picojson::object>()) {
        CACTUS_LOG_ERROR("model", "manifest parse: " << err);
        return false;
    }
    const auto& obj = root.get<picojson::object>();
    if (obj.count("family") && obj.at("family").is<std::string>()) {
        family_ = obj.at("family").get<std::string>();
    }
    if (!obj.count("components")) return false;
    for (const auto& cv : obj.at("components").get<picojson::array>()) {
        const auto& c = cv.get<picojson::object>();
        Component comp;
        comp.name = c.at("component").get<std::string>();
        comp.graph_path = c.count("graph") ? c.at("graph").get<std::string>() : "";
        if (c.count("runtime_input_node_ids")) {
            for (const auto& v : c.at("runtime_input_node_ids").get<picojson::array>())
                comp.runtime_input_node_ids.push_back(static_cast<int>(v.get<int64_t>()));
        }
        if (c.count("logical_inputs")) {
            for (const auto& v : c.at("logical_inputs").get<picojson::array>())
                comp.logical_inputs.push_back(v.get<std::string>());
        }
        if (c.count("output_node_ids")) {
            for (const auto& v : c.at("output_node_ids").get<picojson::array>())
                comp.output_node_ids.push_back(static_cast<int>(v.get<int64_t>()));
        }
        if (c.count("logical_outputs")) {
            for (const auto& v : c.at("logical_outputs").get<picojson::array>())
                comp.logical_outputs.push_back(v.get<std::string>());
        }
        if (c.count("metadata") && c.at("metadata").is<picojson::object>()) {
            for (const auto& mv : c.at("metadata").get<picojson::object>()) {
                if (mv.second.is<std::string>()) {
                    comp.metadata[mv.first] = mv.second.get<std::string>();
                }
            }
        }
        if (c.count("bound_constant_bindings")) {
            for (const auto& bv : c.at("bound_constant_bindings").get<picojson::array>()) {
                const auto& b = bv.get<picojson::object>();
                Binding bd;
                bd.node_id = static_cast<int>(b.at("node_id").get<int64_t>());
                bd.path = b.at("path").get<std::string>();
                comp.bindings.push_back(std::move(bd));
            }
        }
        if (c.count("cache_state_node_ids")) {
            for (const auto& sv : c.at("cache_state_node_ids").get<picojson::array>()) {
                if (!sv.is<picojson::object>()) continue;
                const auto& s = sv.get<picojson::object>();
                CacheStateBinding cs;
                if (s.count("layer_key")) cs.layer_key = s.at("layer_key").get<std::string>();
                if (s.count("key") && s.at("key").is<int64_t>())
                    cs.key_node_id = static_cast<int>(s.at("key").get<int64_t>());
                if (s.count("value") && s.at("value").is<int64_t>())
                    cs.value_node_id = static_cast<int>(s.at("value").get<int64_t>());
                if (cs.key_node_id >= 0 && cs.value_node_id >= 0) {
                    comp.cache_states.push_back(std::move(cs));
                }
            }
        }
        components_[comp.name] = std::move(comp);
    }
    return true;
}

bool Model::setup_tokenizer() {
    std::string vocab = bundle_dir_ + "/vocab.txt";
    std::string merges = bundle_dir_ + "/merges.txt";
    std::string cfg = bundle_dir_ + "/tokenizer_config.txt";
    if (!fs::exists(vocab)) return false;
    auto rt = load_tokenizer_runtime_config(cfg);
    bool use_bpe = rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::BPE
                   || (rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::UNKNOWN
                       && fs::exists(merges));
    if (use_bpe) tokenizer_ = std::make_unique<BPETokenizer>();
    else        tokenizer_ = std::make_unique<SPTokenizer>();
    return tokenizer_->load_vocabulary_with_config(vocab, merges, cfg);
}

bool Model::load_components(const std::unordered_set<std::string>& required_components) {
    for (auto& [name, comp] : components_) {
        if (!required_components.empty() && !required_components.count(name)) continue;
        if (!load_component_graph(comp)) return false;
    }
    return true;
}

bool Model::load_component_graph(Component& comp) {
    if (comp.graph) return true;
    if (comp.graph_path.empty()) return true;
    fs::path full = fs::path(bundle_dir_) / comp.graph_path;
    try {
        comp.graph = std::make_unique<CactusGraph>(CactusGraph::load(full.string()));
        comp.graph->retain_outputs(comp.output_node_ids);
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("model", "load " << comp.graph_path << ": " << e.what());
        return false;
    }
    for (const auto& b : comp.bindings) {
        if (b.node_id < 0) continue;
        try {
            fs::path weight_path(b.path);
            if (weight_path.is_absolute()) {
                fs::path local = fs::path(bundle_dir_) / weight_path.filename();
                if (fs::exists(local)) weight_path = local;
            } else {
                weight_path = fs::path(bundle_dir_) / weight_path;
            }
            comp.graph->bind_mmap_weights(static_cast<size_t>(b.node_id), weight_path.string());
        } catch (const std::exception& e) {
            CACTUS_LOG_ERROR("model", "bind " << b.path << ": " << e.what());
            return false;
        }
    }
    return bind_runtime_buffers(comp);
}

void Model::unload_component_graph(Component& comp) {
    if (cactus_default_backend() == ComputeBackend::METAL) return;
    if (comp.graph) {
        comp.graph->release_runtime_buffers();
        comp.graph->release_all_weight_pages();
    }
    comp.input_buffers.clear();
    comp.graph.reset();
}

bool Model::bind_runtime_buffers(Component& comp) {
    comp.input_buffers.resize(comp.runtime_input_node_ids.size());
    for (size_t i = 0; i < comp.runtime_input_node_ids.size(); ++i) {
        size_t node_id = static_cast<size_t>(comp.runtime_input_node_ids[i]);
        const auto& desc = comp.graph->get_output_buffer(node_id);
        comp.input_buffers[i].assign(desc.byte_size, 0);
        comp.graph->set_external_input(node_id, comp.input_buffers[i].data(), desc.precision);
    }
    return true;
}

int Model::input_index(const Component& comp, const std::string& name) const {
    for (size_t i = 0; i < comp.logical_inputs.size(); ++i) {
        if (comp.logical_inputs[i] == name) return static_cast<int>(i);
    }
    return -1;
}

void Model::write_int_input(Component& comp, const std::string& name, int64_t value) {
    write_int_input_at(comp, name, 0, value);
}

void Model::write_int_input_at(Component& comp, const std::string& name, size_t index, int64_t value) {
    int idx = input_index(comp, name);
    if (idx < 0) return;
    size_t node_id = static_cast<size_t>(comp.runtime_input_node_ids[idx]);
    const auto& desc = comp.graph->get_output_buffer(node_id);
    auto& buf = comp.input_buffers[idx];
    if (index >= desc.total_size) return;
    size_t offset = PrecisionTraits::byte_offset_of(desc.precision, index);
    auto* dst = buf.data() + offset;
    switch (desc.precision) {
        case Precision::FP32:
            *reinterpret_cast<float*>(dst) = static_cast<float>(value);
            break;
        case Precision::FP16:
            *reinterpret_cast<__fp16*>(dst) = static_cast<__fp16>(value);
            break;
        case Precision::INT8:
            *reinterpret_cast<int8_t*>(dst) = static_cast<int8_t>(value);
            break;
        default:
            *reinterpret_cast<int32_t*>(dst) = static_cast<int32_t>(value);
            break;
    }
}

void Model::write_bytes_input(Component& comp, const std::string& name, const void* data, size_t byte_size) {
    int idx = input_index(comp, name);
    if (idx < 0) return;
    auto& buf = comp.input_buffers[idx];
    size_t to_copy = std::min(byte_size, buf.size());
    std::memcpy(buf.data(), data, to_copy);
    if (to_copy < buf.size()) {
        std::memset(buf.data() + to_copy, 0, buf.size() - to_copy);
    }
}

int Model::output_index(const Component& comp, const std::string& name) const {
    for (size_t i = 0; i < comp.logical_outputs.size(); ++i) {
        if (comp.logical_outputs[i] == name) return static_cast<int>(i);
    }
    return -1;
}

void Model::copy_encoder_outputs_to_decoder(const Component& enc) {
    for (size_t i = 0; i < enc.output_node_ids.size() && i < enc.logical_outputs.size(); ++i) {
        const std::string& out_name = enc.logical_outputs[i];
        int dst_idx = input_index(*decoder_, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(enc.output_node_ids[i]);
        const auto& src_desc = enc.graph->get_output_buffer(src_node);
        void* src_ptr = enc.graph->get_output(src_node);
        auto& dst_buf = decoder_->input_buffers[dst_idx];
        size_t to_copy = std::min(src_desc.byte_size, dst_buf.size());
        std::memcpy(dst_buf.data(), src_ptr, to_copy);
    }
}

void Model::run_step(uint32_t token_id, size_t position, bool /*read_logits*/, bool use_fused) {
    if (decode_route_ == DecodeRoute::DIRECT_DECODER_STEP) {
        write_int_input(*decoder_, "input_ids", static_cast<int64_t>(token_id));
        write_int_input(*decoder_, "position_ids", static_cast<int64_t>(position));
        if (!use_fused || cactus_default_backend() != ComputeBackend::METAL) cactus_graph_set_prefill_consistent(true);
        decoder_->graph->execute();
        cactus_graph_set_prefill_consistent(false);
        maybe_capture_handoff_probe_hidden(*decoder_);
        return;
    }
    if (!use_fused) cactus_graph_set_prefill_consistent(true);
    if (use_fused && cactus_default_backend() == ComputeBackend::METAL) {
        if (ple_probe_state_ == 0)
            ple_probe_state_ = encoder_->graph->extract_ple_pathway(fused_embed_ctx_) ? 1 : 2;
        if (ple_probe_state_ == 1) {
            fused_embed_ctx_.token_id = static_cast<int>(token_id);
            fused_embed_ctx_.position = static_cast<int>(position);
            cactus_graph_set_fused_embed(&fused_embed_ctx_);
            write_int_input(*decoder_, "position_ids", static_cast<int64_t>(position));
            decoder_->graph->execute();
            cactus_graph_set_fused_embed(nullptr);
            cactus_graph_set_prefill_consistent(false);
            maybe_capture_handoff_probe_hidden(*decoder_);
            return;
        }
    }
    run_encoder_step(token_id, position);
    copy_component_outputs_to_inputs(*encoder_, *decoder_);
    decoder_->graph->execute();
    cactus_graph_set_prefill_consistent(false);
    maybe_capture_handoff_probe_hidden(*decoder_);
}

void Model::run_encoder_step(uint32_t token_id, size_t position) {
    write_int_input(*encoder_, "input_ids", static_cast<int64_t>(token_id));
    write_int_input(*encoder_, "position_ids", static_cast<int64_t>(position));
    encoder_->graph->execute();
}

void Model::set_component_batch(Component& comp, size_t batch) {
    for (size_t i = 0; i < comp.runtime_input_node_ids.size(); ++i) {
        size_t node_id = static_cast<size_t>(comp.runtime_input_node_ids[i]);
        const auto& desc = comp.graph->get_output_buffer(node_id);
        if (!desc.has_dynamic_dims() || desc.shape.empty() || desc.shape[0] == batch) continue;
        std::vector<size_t> shape = desc.shape;
        shape[0] = batch;
        comp.graph->set_runtime_input_shape(node_id, shape);
        const auto& resized = comp.graph->get_output_buffer(node_id);
        comp.input_buffers[i].assign(resized.byte_size, 0);
        comp.graph->set_external_input(node_id, comp.input_buffers[i].data(), resized.precision);
    }
}

size_t Model::decoder_cache_num_slots() {
    if (!decoder_ || !decoder_->graph) return 1;
    for (const auto& state : decoder_->cache_states) {
        int node_id = state.key_node_id;
        if (node_id < 0) continue;
        if (decoder_->graph->get_node_op_type(static_cast<size_t>(node_id)) != OpType::KV_CACHE_STATE) continue;
        size_t num_slots = decoder_->graph->get_node_cache_num_slots(static_cast<size_t>(node_id));
        return num_slots ? num_slots : 1;
    }
    return 1;
}

void Model::run_step_batch(const std::vector<uint32_t>& token_ids, const std::vector<size_t>& positions) {
    if (decode_route_ == DecodeRoute::DIRECT_DECODER_STEP) {
        for (size_t b = 0; b < token_ids.size(); ++b) {
            write_int_input_at(*decoder_, "input_ids", b, static_cast<int64_t>(token_ids[b]));
            write_int_input_at(*decoder_, "position_ids", b, static_cast<int64_t>(positions[b]));
        }
        decoder_->graph->execute();
        maybe_capture_handoff_probe_hidden(*decoder_);
        return;
    }
    for (size_t b = 0; b < token_ids.size(); ++b) {
        write_int_input_at(*encoder_, "input_ids", b, static_cast<int64_t>(token_ids[b]));
        write_int_input_at(*encoder_, "position_ids", b, static_cast<int64_t>(positions[b]));
    }
    encoder_->graph->execute();
    copy_component_outputs_to_inputs(*encoder_, *decoder_);
    decoder_->graph->execute();
    maybe_capture_handoff_probe_hidden(*decoder_);
}

std::vector<uint32_t> Model::batch_stop_token_ids() const {
    std::vector<uint32_t> stops;
    stops.push_back(config_.eos_token_id);
    Tokenizer* tk = get_tokenizer();
    auto add_if_single = [&](const std::string& s) {
        if (!tk || s.empty()) return;
        std::vector<uint32_t> t = tk->encode(s);
        if (t.size() == 1) stops.push_back(t[0]);
    };
    if (tk) add_if_single(tk->get_default_stop_sequence());
    if (config_.model_type == Config::ModelType::GEMMA4) add_if_single("<turn|>");
    return stops;
}

std::vector<std::vector<uint32_t>> Model::generate_batch(const std::vector<std::vector<uint32_t>>& prompts,
                                                         size_t max_new_tokens, bool stop_on_eos) {
    size_t batch = prompts.size();
    const bool cached = decode_route_ == DecodeRoute::CACHED_STEP;
    const bool direct = decode_route_ == DecodeRoute::DIRECT_DECODER_STEP;
    if (batch == 0 || !decoder_ || (!cached && !direct)) return {};
    if (cached && !encoder_) return {};
    for (const auto& p : prompts) if (p.empty()) return {};
    if (cached && !load_component_graph(*encoder_)) return {};
    if (!load_component_graph(*decoder_)) return {};
    if (batch > decoder_cache_num_slots()) return {};
    auto has_dynamic_input = [](Component& c) {
        for (int node_id : c.runtime_input_node_ids) {
            if (node_id < 0) continue;
            if (c.graph->get_output_buffer(static_cast<size_t>(node_id)).has_dynamic_dims()) return true;
        }
        return false;
    };
    if (batch > 1) {
        if (!has_dynamic_input(*decoder_)) return {};
        if (cached && !has_dynamic_input(*encoder_)) return {};
        for (const auto& np : decoder_->graph->nodes_) {
            if (np->op_type == OpType::CONV_CACHE_STATE || np->op_type == OpType::RECURRENT_CACHE_STATE) return {};
        }
    }

    size_t exec_batch = batch;
    if (batch > 1 || cactus_default_backend() == ComputeBackend::METAL) {
        exec_batch = std::max(batch, decoder_cache_num_slots());
    }
    reset_component_cache_states(*decoder_);
    if (cached) set_component_batch(*encoder_, exec_batch);
    set_component_batch(*decoder_, exec_batch);

    std::vector<std::vector<uint32_t>> out(batch);
    std::vector<size_t> fed(batch, 0);
    std::vector<uint32_t> last(batch, 0);
    std::vector<bool> done(batch, false);
    size_t remaining = batch;

    const std::vector<uint32_t> stop_ids = stop_on_eos ? batch_stop_token_ids() : std::vector<uint32_t>{};
    auto is_stop = [&](uint32_t t) {
        for (uint32_t s : stop_ids) if (s == t) return true;
        return false;
    };

    std::vector<uint32_t> tokens(exec_batch);
    std::vector<size_t> positions(exec_batch);
    while (remaining > 0) {
        for (size_t b = 0; b < batch; ++b) {
            positions[b] = fed[b];
            tokens[b] = (fed[b] < prompts[b].size()) ? prompts[b][fed[b]] : last[b];
        }
        for (size_t b = batch; b < exec_batch; ++b) {
            positions[b] = positions[0];
            tokens[b] = tokens[0];
        }
        run_step_batch(tokens, positions);
        std::vector<uint32_t> sampled = argmax_component_logits_batch(*decoder_, batch);
        for (size_t b = 0; b < batch; ++b) {
            ++fed[b];
            if (done[b]) continue;
            if (fed[b] >= prompts[b].size()) {
                last[b] = sampled[b];
                if (is_stop(sampled[b])) {
                    done[b] = true;
                    --remaining;
                    continue;
                }
                out[b].push_back(sampled[b]);
                if (out[b].size() >= max_new_tokens) {
                    done[b] = true;
                    --remaining;
                }
            }
        }
    }
    return out;
}

std::vector<std::vector<uint32_t>> Model::decode_batch(const std::vector<uint32_t>& seed_tokens,
                                                       size_t max_new_tokens) {
    std::vector<std::vector<uint32_t>> prompts;
    prompts.reserve(seed_tokens.size());
    for (uint32_t seed : seed_tokens) prompts.push_back({seed});
    return generate_batch(prompts, max_new_tokens);
}

bool Model::supports_dynamic_batch() {
    if (!decoder_) return false;
    if (!decoder_->graph && !load_component_graph(*decoder_)) return false;
    return decoder_->graph->has_dynamic_shapes();
}

void Model::set_decode_slots(size_t num_slots) {
    if (num_slots == 0) num_slots = 1;
    if (!decoder_) return;
    if (!decoder_->graph && !load_component_graph(*decoder_)) return;
    for (const auto& state : decoder_->cache_states) {
        for (int node_id : {state.key_node_id, state.value_node_id}) {
            if (node_id < 0) continue;
            if (decoder_->graph->get_node_op_type(static_cast<size_t>(node_id)) != OpType::KV_CACHE_STATE) continue;
            decoder_->graph->resize_cache_slots(static_cast<size_t>(node_id), num_slots);
        }
    }
}

#define FOR_EACH_MATCHED_OUTPUT(source, target, body) \
    for (size_t _i = 0; _i < (source).output_node_ids.size() && _i < (source).logical_outputs.size(); ++_i) { \
        const std::string& out_name = (source).logical_outputs[_i]; \
        int dst_idx = input_index((target), out_name); \
        if (dst_idx < 0) continue; \
        size_t src_node = static_cast<size_t>((source).output_node_ids[_i]); \
        const auto& src_desc = (source).graph->get_output_buffer(src_node); \
        size_t dst_node = static_cast<size_t>((target).runtime_input_node_ids[dst_idx]); \
        const auto& dst_desc = (target).graph->get_output_buffer(dst_node); \
        auto& dst_buf = (target).input_buffers[dst_idx]; \
        body \
    }

void Model::copy_component_outputs_to_inputs(const Component& source, Component& target) {
    FOR_EACH_MATCHED_OUTPUT(source, target, {
        std::fill(dst_buf.begin(), dst_buf.end(), 0);
        size_t elements = std::min(src_desc.total_size, dst_desc.total_size);
        if (!copy_component_tensor(*source.graph, src_desc, src_node, dst_desc, dst_buf, 0, elements, out_name))
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
    })
}

bool Model::copy_cross_kv_outputs_to_decoder_cache_inputs(const Component& source, Component& target, size_t source_len) {
    bool copied_any = false;
    for (size_t i = 0; i < source.output_node_ids.size() && i < source.logical_outputs.size(); ++i) {
        const std::string& out_name = source.logical_outputs[i];
        if (out_name.rfind("cross_k_", 0) != 0 && out_name.rfind("cross_v_", 0) != 0) {
            continue;
        }
        int dst_idx = input_index(target, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(source.output_node_ids[i]);
        const auto& src_desc = source.graph->get_output_buffer(src_node);
        const void* src_ptr = source.graph->get_output(src_node);
        size_t dst_node = static_cast<size_t>(target.runtime_input_node_ids[dst_idx]);
        const auto& dst_desc = target.graph->get_output_buffer(dst_node);
        if (!write_cross_kv_cache_buffer(src_desc, src_ptr, dst_desc, target.input_buffers[dst_idx], source_len, out_name)) {
            return false;
        }
        copied_any = true;
    }
    return copied_any;
}

void Model::copy_component_outputs_to_chunk_inputs(const Component& source, Component& target, size_t token_index) {
    FOR_EACH_MATCHED_OUTPUT(source, target, {
        size_t chunk_tokens = component_chunk_tokens(target, out_name);
        if (chunk_tokens <= token_index || chunk_tokens == 0)
            throw std::runtime_error("chunk prefill token index exceeds input capacity for " + out_name);
        if (dst_desc.total_size % chunk_tokens != 0)
            throw std::runtime_error("chunk prefill input shape is not token-aligned for " + out_name);
        size_t elements_per_token = dst_desc.total_size / chunk_tokens;
        if (src_desc.total_size != elements_per_token)
            throw std::runtime_error("component output/input token shape mismatch for " + out_name);
        if (!copy_component_tensor(*source.graph, src_desc, src_node, dst_desc,
                dst_buf, token_index * elements_per_token, src_desc.total_size, out_name))
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
    })
}

void Model::copy_component_outputs_to_chunk_inputs_range(const Component& source, Component& target, size_t token_offset) {
    FOR_EACH_MATCHED_OUTPUT(source, target, {
        size_t src_tokens = component_output_tokens(source, out_name);
        size_t dst_tokens = component_chunk_tokens(target, out_name);
        if (src_tokens == 0 || dst_tokens == 0 || token_offset + src_tokens > dst_tokens)
            throw std::runtime_error("chunk prefill output range exceeds input capacity for " + out_name);
        if (src_desc.total_size % src_tokens != 0 || dst_desc.total_size % dst_tokens != 0)
            throw std::runtime_error("chunk prefill output/input shape is not token-aligned for " + out_name);
        size_t src_elements_per_token = src_desc.total_size / src_tokens;
        size_t dst_elements_per_token = dst_desc.total_size / dst_tokens;
        if (src_elements_per_token != dst_elements_per_token)
            throw std::runtime_error("component output/input token shape mismatch for " + out_name);
        if (!copy_component_tensor(*source.graph, src_desc, src_node, dst_desc,
                dst_buf, token_offset * dst_elements_per_token, src_desc.total_size, out_name))
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
    })
}

#undef FOR_EACH_MATCHED_OUTPUT

bool Model::cache_states_compatible(const Component& source, const Component& target) const {
    if (source.cache_states.empty() || source.cache_states.size() != target.cache_states.size()) return false;
    for (size_t i = 0; i < source.cache_states.size(); ++i) {
        const auto& src = source.cache_states[i];
        const auto& dst = target.cache_states[i];
        if (src.layer_key != dst.layer_key) return false;
        if (src.key_node_id < 0 || src.value_node_id < 0 || dst.key_node_id < 0 || dst.value_node_id < 0) return false;
    }
    return true;
}

void Model::move_cache_states(Component& source, Component& target, size_t logical_current) {
    if (source.cache_states.empty() || source.cache_states.size() != target.cache_states.size()) {
        throw std::runtime_error("prefill and step cache states are not compatible");
    }
    for (size_t i = 0; i < source.cache_states.size(); ++i) {
        const auto& src = source.cache_states[i];
        const auto& dst = target.cache_states[i];
        if (src.layer_key != dst.layer_key) {
            throw std::runtime_error("prefill and step cache layer mismatch: " + src.layer_key + " != " + dst.layer_key);
        }
        int moved_src = -1, moved_dst = -1;
        for (auto [src_node, dst_node] : {std::pair<int, int>{src.key_node_id, dst.key_node_id}, std::pair<int, int>{src.value_node_id, dst.value_node_id}}) {
            if (src_node < 0 || dst_node < 0) continue;
            // Conv/recurrent caches share one node for key and value; move it only once.
            if (src_node == moved_src && dst_node == moved_dst) continue;
            moved_src = src_node;
            moved_dst = dst_node;
            target.graph->steal_cache_buffer(static_cast<size_t>(dst_node), *source.graph, static_cast<size_t>(src_node));
            if (target.graph->get_node_op_type(static_cast<size_t>(dst_node)) == OpType::KV_CACHE_STATE &&
                logical_current != std::numeric_limits<size_t>::max()) {
                auto* meta = static_cast<uint64_t*>(target.graph->get_output(static_cast<size_t>(dst_node)));
                if (meta && logical_current < meta[0]) {
                    meta[0] = logical_current;
                }
            }
        }
    }
}

void Model::set_cache_current_len(Component& comp, size_t len) {
    for (const auto& state : comp.cache_states) {
        for (int node_id : {state.key_node_id, state.value_node_id}) {
            if (node_id < 0) continue;
            if (comp.graph->get_node_op_type(static_cast<size_t>(node_id)) != OpType::KV_CACHE_STATE) continue;
            auto* meta = static_cast<uint64_t*>(comp.graph->get_output(static_cast<size_t>(node_id)));
            if (!meta || len >= meta[0]) continue;
            meta[0] = len;
        }
    }
}

void Model::reset_component_cache_states(Component& comp) {
    for (const auto& state : comp.cache_states) {
        for (int node_id : {state.key_node_id, state.value_node_id}) {
            if (node_id < 0) continue;
            const auto& desc = comp.graph->get_output_buffer(static_cast<size_t>(node_id));
            if (desc.byte_size == 0 || !desc.get_data()) continue;
            void* ptr = comp.graph->get_output(static_cast<size_t>(node_id));
            if (!ptr) continue;
            const OpType op_type = comp.graph->get_node_op_type(static_cast<size_t>(node_id));
            switch (op_type) {
                case OpType::KV_CACHE_STATE:
                    if (desc.byte_size >= sizeof(uint64_t)) {
                        auto* meta = static_cast<uint64_t*>(ptr);
                        uint64_t num_slots = (desc.byte_size >= 6 * sizeof(uint64_t) && meta[5] > 0) ? meta[5] : 1;
                        size_t slot_stride = num_slots ? desc.byte_size / num_slots : desc.byte_size;
                        for (uint64_t s = 0; s < num_slots; ++s) {
                            *reinterpret_cast<uint64_t*>(static_cast<char*>(ptr) + s * slot_stride) = 0;
                        }
                    }
                    break;
                case OpType::CONV_CACHE_STATE:
                    if (desc.byte_size >= 2 * sizeof(uint64_t)) {
                        auto* meta = static_cast<uint64_t*>(ptr);
                        meta[0] = 0;  // head
                        meta[1] = 0;  // count
                    }
                    break;
                case OpType::RECURRENT_CACHE_STATE:
                    std::memset(ptr, 0, desc.byte_size);
                    break;
                default:
                    break;
            }
        }
    }
}

size_t Model::component_chunk_tokens(const Component& comp, const std::string& input_name) const {
    int idx = input_index(comp, input_name);
    if (idx < 0) return 0;
    const auto& desc = comp.graph->get_output_buffer(static_cast<size_t>(comp.runtime_input_node_ids[idx]));
    if (desc.shape.size() >= 2 && desc.shape[0] == 1) return desc.shape[1];
    return desc.shape.empty() ? 0 : desc.shape[0];
}

size_t Model::component_output_tokens(const Component& comp, const std::string& output_name) const {
    for (size_t i = 0; i < comp.logical_outputs.size() && i < comp.output_node_ids.size(); ++i) {
        if (comp.logical_outputs[i] != output_name) continue;
        const auto& desc = comp.graph->get_output_buffer(static_cast<size_t>(comp.output_node_ids[i]));
        if (desc.shape.size() >= 2 && desc.shape[0] == 1) return desc.shape[1];
        return desc.shape.empty() ? 0 : desc.shape[0];
    }
    return 0;
}

void Model::execute_prefill_chunk(Component& chunk_comp, Component* enc_comp, size_t encoder_chunk,
                                  size_t chunk_tokens, const std::vector<uint32_t>& tokens,
                                  size_t processed, size_t start_position) {
    for (size_t i = 0; i < chunk_comp.input_buffers.size(); ++i) {
        std::fill(chunk_comp.input_buffers[i].begin(), chunk_comp.input_buffers[i].end(), 0);
    }
    if (enc_comp && encoder_chunk > 0) {
        for (size_t chunk_offset = 0; chunk_offset < chunk_tokens; chunk_offset += encoder_chunk) {
            for (size_t i = 0; i < enc_comp->input_buffers.size(); ++i) {
                std::fill(enc_comp->input_buffers[i].begin(), enc_comp->input_buffers[i].end(), 0);
            }
            for (size_t i = 0; i < encoder_chunk; ++i) {
                size_t index = processed + chunk_offset + i;
                uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
                write_int_input_at(*enc_comp, "input_ids", i, static_cast<int64_t>(token));
                write_int_input_at(*enc_comp, "position_ids", i, static_cast<int64_t>(start_position + processed + chunk_offset + i));
            }
            enc_comp->graph->execute();
            copy_component_outputs_to_chunk_inputs_range(*enc_comp, chunk_comp, chunk_offset);
        }
    } else {
        for (size_t i = 0; i < chunk_tokens; ++i) {
            size_t index = processed + i;
            uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
            run_encoder_step(token, start_position + processed + i);
            copy_component_outputs_to_chunk_inputs(*encoder_, chunk_comp, i);
        }
    }
    chunk_comp.graph->execute();
}

void Model::reset_prefill_stats() {
    last_prefill_cache_copy_ms_ = 0.0;
    last_prefill_padding_tokens_ = 0;
    last_prefill_scalar_tail_tokens_ = 0;
    last_prefill_tail_chunk_tokens_ = 0;
    last_prefill_tail_padding_tokens_ = 0;
}

Model::ChunkedPrefillResult Model::run_chunked_prefill(const std::vector<uint32_t>& tokens, size_t start_position, size_t chunk_size, bool prepare_decode) {
    ChunkedPrefillResult result;
    reset_prefill_stats();
    if (decode_route_ != DecodeRoute::CACHED_STEP || !encoder_ || !decoder_ || !decoder_prefill_) return result;
    if (start_position != 0) return result;
    if (!load_component_graph(*decoder_prefill_)) return result;
    if (prefill_encoder_ && !load_component_graph(*prefill_encoder_)) return result;
    if (!cache_states_compatible(*decoder_prefill_, *decoder_)) return result;
    size_t component_tokens = component_chunk_tokens(*decoder_prefill_, "inputs_embeds");
    if (component_tokens <= 1) return result;
    size_t effective_chunk = chunk_size > 0 ? std::min(chunk_size, component_tokens) : component_tokens;
    if (effective_chunk != component_tokens) effective_chunk = component_tokens;
    size_t whole_chunks_end = (tokens.size() / effective_chunk) * effective_chunk;
    auto any_cache_node = [&](auto predicate) {
        if (!decoder_prefill_->graph) return false;
        for (const auto& state : decoder_prefill_->cache_states) {
            for (int node_id : {state.key_node_id, state.value_node_id}) {
                if (node_id < 0) continue;
                if (predicate(static_cast<size_t>(node_id))) return true;
            }
        }
        return false;
    };
    const bool has_recurrent_state = any_cache_node([&](size_t id) {
        return decoder_prefill_->graph->get_node_op_type(id) == OpType::RECURRENT_CACHE_STATE;
    });
    if (has_recurrent_state && whole_chunks_end > effective_chunk) {
        whole_chunks_end = effective_chunk;
    }
    const bool has_sliding_window_cache = any_cache_node([&](size_t id) {
        return decoder_prefill_->graph->get_node_op_type(id) == OpType::KV_CACHE_STATE
            && decoder_prefill_->graph->get_node_window_size(id) > 0;
    });
    const size_t tail_tokens = tokens.size() - whole_chunks_end;
    const size_t padding_cutoff = std::max<size_t>(1, effective_chunk / 16);
    const bool has_conv_state = any_cache_node([&](size_t id) {
        return decoder_prefill_->graph->get_node_op_type(id) == OpType::CONV_CACHE_STATE;
    });
    const bool pad_tail = !has_conv_state
        && !has_recurrent_state
        && !has_sliding_window_cache
        && tail_tokens >= padding_cutoff;
    const bool padded_window_too_small = any_cache_node([&](size_t id) {
        if (decoder_prefill_->graph->get_node_op_type(id) != OpType::KV_CACHE_STATE) return false;
        size_t window = decoder_prefill_->graph->get_node_window_size(id);
        size_t sink = decoder_prefill_->graph->get_node_sink_size(id);
        return window > 0 && (window < effective_chunk * 4 || window <= effective_chunk + sink);
    });
    const bool use_padded_tail = !pad_tail && !prefill_tail_pad_disabled_
        && has_sliding_window_cache && !has_recurrent_state && !has_conv_state
        && tail_tokens > 8 && !padded_window_too_small;
    const size_t executable_tokens = whole_chunks_end + (pad_tail ? effective_chunk : 0);
    if (executable_tokens == 0 && !use_padded_tail) {
        result.scalar_tail_tokens = tail_tokens;
        last_prefill_scalar_tail_tokens_ = tail_tokens;
        return result;
    }

    size_t encoder_chunk = 0;
    if (prefill_encoder_ && input_index(*prefill_encoder_, "input_ids") >= 0 && input_index(*prefill_encoder_, "position_ids") >= 0) {
        encoder_chunk = component_chunk_tokens(*prefill_encoder_, "input_ids");
        if (encoder_chunk == 0 || effective_chunk % encoder_chunk != 0) {
            encoder_chunk = 0;
        }
    }

    size_t processed = 0;
    while (processed + effective_chunk <= executable_tokens) {
        execute_prefill_chunk(*decoder_prefill_, prefill_encoder_, encoder_chunk,
                              effective_chunk, tokens, processed, start_position);
        processed += effective_chunk;
    }

    size_t tail_executed = 0;
    size_t tail_padding = 0;
    if (use_padded_tail) {
        const size_t pads = effective_chunk - tail_tokens;
        const size_t kept_real = tail_tokens - 1;
        std::vector<std::pair<size_t, std::vector<uint8_t>>> backups;
        for (const auto& state : decoder_prefill_->cache_states) {
            for (int node_id : {state.key_node_id, state.value_node_id}) {
                if (node_id < 0) continue;
                size_t id = static_cast<size_t>(node_id);
                if (decoder_prefill_->graph->get_node_op_type(id) != OpType::KV_CACHE_STATE) continue;
                backups.emplace_back(id, decoder_prefill_->graph->snapshot_cache_padded_append(id, kept_real, pads + 1));
            }
        }
        execute_prefill_chunk(*decoder_prefill_, prefill_encoder_, encoder_chunk,
                              effective_chunk, tokens, processed, start_position);
        for (auto& [id, backup] : backups) {
            decoder_prefill_->graph->rollback_cache_padded_append(id, kept_real, pads + 1, backup);
        }
        processed += kept_real;
        tail_executed = kept_real;
        tail_padding = pads;
    }

    result.executed_tokens = processed;
    result.logical_tokens = std::min(tokens.size(), processed);
    if (result.logical_tokens > 0) {
        result.last_logit_row = (result.logical_tokens - 1) % effective_chunk;
    }
    result.padding_tokens = processed > tokens.size() ? processed - tokens.size() : 0;
    result.scalar_tail_tokens = tokens.size() - result.logical_tokens;
    last_prefill_padding_tokens_ = result.padding_tokens;
    last_prefill_scalar_tail_tokens_ = result.scalar_tail_tokens;
    last_prefill_tail_chunk_tokens_ = tail_executed;
    last_prefill_tail_padding_tokens_ = tail_padding;

    if (processed > 0 && prepare_decode) {
        for (size_t i = 0; i < decoder_->input_buffers.size(); ++i) {
            std::fill(decoder_->input_buffers[i].begin(), decoder_->input_buffers[i].end(), 0);
        }
        auto copy_start = std::chrono::high_resolution_clock::now();
        move_cache_states(*decoder_prefill_, *decoder_, start_position + result.logical_tokens);
        auto copy_end = std::chrono::high_resolution_clock::now();
        last_prefill_cache_copy_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count() / 1000.0;
    }
    return result;
}

void Model::run_full_context_text() {
    if (!encoder_ || !decoder_ || context_tokens_.empty()) return;
    int input_ids_idx = input_index(*encoder_, "input_ids");
    int attention_mask_idx = input_index(*encoder_, "attention_mask");
    if (input_ids_idx < 0 || attention_mask_idx < 0) {
        throw std::runtime_error("text_lm_encoder requires input_ids and attention_mask inputs");
    }
    size_t input_node = static_cast<size_t>(encoder_->runtime_input_node_ids[input_ids_idx]);
    const auto& input_desc = encoder_->graph->get_output_buffer(input_node);
    if (context_tokens_.size() > input_desc.total_size) {
        throw std::runtime_error("context exceeds transpiled text_lm_encoder capacity");
    }
    std::fill(encoder_->input_buffers[input_ids_idx].begin(), encoder_->input_buffers[input_ids_idx].end(), 0);
    std::fill(encoder_->input_buffers[attention_mask_idx].begin(), encoder_->input_buffers[attention_mask_idx].end(), 0);
    for (size_t i = 0; i < context_tokens_.size(); ++i) {
        write_int_input_at(*encoder_, "input_ids", i, static_cast<int64_t>(context_tokens_[i]));
        write_int_input_at(*encoder_, "attention_mask", i, 1);
    }
    encoder_->graph->execute();
    for (size_t i = 0; i < encoder_->output_node_ids.size() && i < encoder_->logical_outputs.size(); ++i) {
        const std::string& out_name = encoder_->logical_outputs[i];
        int dst_idx = input_index(*decoder_, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(encoder_->output_node_ids[i]);
        const auto& src_desc = encoder_->graph->get_output_buffer(src_node);
        void* src_ptr = encoder_->graph->get_output(src_node);
        std::memcpy(decoder_->input_buffers[dst_idx].data(), src_ptr, src_desc.byte_size);
    }
    last_logit_position_ = context_tokens_.empty() ? 0 : context_tokens_.size() - 1;
    decoder_->graph->execute();
}

void Model::write_media_embeds_row(Component& comp, int embeds_idx, const uint8_t* feature_row,
                                   size_t feature_row_bytes, Precision feature_precision) {
    auto& embeds_buf = comp.input_buffers[embeds_idx];
    size_t node_id = static_cast<size_t>(comp.runtime_input_node_ids[embeds_idx]);
    const auto& desc = comp.graph->get_output_buffer(node_id);
    if (desc.precision == feature_precision) {
        size_t to_copy = std::min(feature_row_bytes, embeds_buf.size());
        std::memcpy(embeds_buf.data(), feature_row, to_copy);
        if (to_copy < embeds_buf.size()) {
            std::memset(embeds_buf.data() + to_copy, 0, embeds_buf.size() - to_copy);
        }
        return;
    }
    size_t src_elem = PrecisionTraits::size_of(feature_precision);
    size_t dst_elem = PrecisionTraits::size_of(desc.precision);
    size_t src_count = src_elem ? feature_row_bytes / src_elem : 0;
    size_t dst_count = dst_elem ? embeds_buf.size() / dst_elem : 0;
    size_t n = std::min(src_count, dst_count);
    auto load_float = [&](size_t i) -> float {
        if (feature_precision == Precision::FP16) return static_cast<float>(reinterpret_cast<const __fp16*>(feature_row)[i]);
        if (feature_precision == Precision::FP32) return reinterpret_cast<const float*>(feature_row)[i];
        return static_cast<float>(reinterpret_cast<const int8_t*>(feature_row)[i]);
    };
    for (size_t i = 0; i < n; ++i) {
        float v = load_float(i);
        if (desc.precision == Precision::FP16) reinterpret_cast<__fp16*>(embeds_buf.data())[i] = static_cast<__fp16>(v);
        else if (desc.precision == Precision::FP32) reinterpret_cast<float*>(embeds_buf.data())[i] = v;
        else reinterpret_cast<int8_t*>(embeds_buf.data())[i] = static_cast<int8_t>(v);
    }
    if (n < dst_count) {
        std::memset(embeds_buf.data() + n * dst_elem, 0, (dst_count - n) * dst_elem);
    }
}

void Model::run_media_step(size_t position, const uint8_t* feature_row, size_t feature_row_bytes,
                           Precision feature_precision) {
    if (lm_encoder_media_step_) {
        int embeds_idx = input_index(*lm_encoder_media_step_, "inputs_embeds");
        if (embeds_idx >= 0) {
            write_media_embeds_row(*lm_encoder_media_step_, embeds_idx, feature_row, feature_row_bytes, feature_precision);
            write_int_input(*lm_encoder_media_step_, "input_ids", 0);
            write_int_input(*lm_encoder_media_step_, "position_ids", static_cast<int64_t>(position));
            lm_encoder_media_step_->graph->execute();
            copy_encoder_outputs_to_decoder(*lm_encoder_media_step_);
            decoder_->graph->execute();
            return;
        }
    } else if (encoder_ != nullptr && decoder_ != nullptr) {
        int dec_embeds_idx = input_index(*decoder_, "inputs_embeds");
        if (dec_embeds_idx >= 0) {
            run_encoder_step(static_cast<uint32_t>(config_.pad_token_id), position);
            copy_component_outputs_to_inputs(*encoder_, *decoder_);
            write_media_embeds_row(*decoder_, dec_embeds_idx, feature_row, feature_row_bytes, feature_precision);
            decoder_->graph->execute();
            return;
        }
    }
    run_step(static_cast<uint32_t>(config_.pad_token_id), position, false);
}

namespace {
void write_typed_buffer(std::vector<uint8_t>& buf, Precision dst_prec,
                        const void* src_data, size_t src_bytes, Precision src_prec);
}  // namespace

void Model::run_vision_encoder(const std::string& image_path) {
    if (!vision_encoder_) return;
    if (family_ == "lfm2_vl") {
        run_vision_encoder_lfm2_vl(image_path);
        return;
    }
    if (!load_component_graph(*vision_encoder_)) {
        throw std::runtime_error("failed to load vision_encoder");
    }

    auto write_int_buffer_typed = [&](int idx, const int64_t* src, size_t src_count) {
        auto& buf = vision_encoder_->input_buffers[idx];
        size_t node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[idx]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(node);
        const size_t elem = PrecisionTraits::size_of(desc.precision);
        const size_t cap = elem ? buf.size() / elem : 0;
        const size_t n = std::min(cap, src_count);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = src[i];
            switch (desc.precision) {
                case Precision::FP32: reinterpret_cast<float*>(buf.data())[i] = static_cast<float>(v); break;
                case Precision::FP16: reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(v); break;
                case Precision::INT8: reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(v); break;
                default:
                    if (elem == 8) reinterpret_cast<int64_t*>(buf.data())[i] = v;
                    else if (elem == 4) reinterpret_cast<int32_t*>(buf.data())[i] = static_cast<int32_t>(v);
                    break;
            }
        }
        if (n < cap) std::memset(buf.data() + n * elem, 0, (cap - n) * elem);
    };

    if (family_ == "qwen3_5" || family_ == "qwen3_vl" || config_.model_type == Config::ModelType::QWEN) {
        Qwen3VlImagePreprocessed prep = preprocess_qwen3_vl_image(image_path, config_);
        int pv_idx = input_index(*vision_encoder_, "pixel_values");
        if (pv_idx < 0) {
            throw std::runtime_error("Qwen3-VL vision_encoder missing pixel_values input");
        }
        auto& pv_buf = vision_encoder_->input_buffers[pv_idx];
        size_t pv_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pv_idx]);
        const auto& pv_desc = vision_encoder_->graph->get_output_buffer(pv_node);
        write_typed_buffer(pv_buf, pv_desc.precision,
                           prep.pixel_values.data(),
                           prep.pixel_values.size() * sizeof(float),
                           Precision::FP32);
    } else {
        Gemma4ImagePreprocessed prep = preprocess_gemma4_image(image_path, config_);
        int pv_idx = input_index(*vision_encoder_, "pixel_values");
        if (pv_idx >= 0) {
            auto& pv_buf = vision_encoder_->input_buffers[pv_idx];
            size_t pv_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pv_idx]);
            const auto& pv_desc = vision_encoder_->graph->get_output_buffer(pv_node);
            write_typed_buffer(pv_buf, pv_desc.precision,
                               prep.pixel_values.data(),
                               prep.pixel_values.size() * sizeof(float),
                               Precision::FP32);
        }
        int pp_idx = input_index(*vision_encoder_, "pixel_position_ids");
        if (pp_idx >= 0) {
            write_int_buffer_typed(pp_idx, prep.pixel_position_ids.data(),
                                   prep.pixel_position_ids.size());
        }
    }

    vision_encoder_->graph->execute();
    for (size_t i = 0; i < vision_encoder_->output_node_ids.size() && i < vision_encoder_->logical_outputs.size(); ++i) {
        const std::string& name = vision_encoder_->logical_outputs[i];
        size_t node_id = static_cast<size_t>(vision_encoder_->output_node_ids[i]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(node_id);
        void* ptr = vision_encoder_->graph->get_output(node_id);
        auto& slot = media_features_[name];
        slot.assign(desc.byte_size, 0);
        std::memcpy(slot.data(), ptr, desc.byte_size);
        media_feature_shapes_[name] = desc.shape;
        media_feature_precisions_[name] = desc.precision;
    }
    if (cactus_default_backend() != ComputeBackend::METAL) {
        vision_encoder_->graph->release_runtime_buffers();
        vision_encoder_->graph->release_all_weight_pages();
    }
    unload_component_graph(*vision_encoder_);
}

namespace {
void typed_buffer_to_float(const void* src, Precision prec, float* dst, size_t n) {
    switch (prec) {
        case Precision::FP32: std::memcpy(dst, src, n * sizeof(float)); break;
        case Precision::FP16: {
            const __fp16* s = reinterpret_cast<const __fp16*>(src);
            for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(s[i]);
            break;
        }
        case Precision::INT8: {
            const int8_t* s = reinterpret_cast<const int8_t*>(src);
            for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(s[i]);
            break;
        }
        default: std::memset(dst, 0, n * sizeof(float)); break;
    }
}
}  // namespace

bool Model::load_lfm2_vl_position_grid() {
    if (lfm2_pos_grid_loaded_) return !lfm2_pos_grid_.empty();
    lfm2_pos_grid_loaded_ = true;
    if (!vision_encoder_) return false;
    auto path_it = vision_encoder_->metadata.find("position_embedding_grid_path");
    auto shape_it = vision_encoder_->metadata.find("position_embedding_grid_shape");
    if (path_it == vision_encoder_->metadata.end() || shape_it == vision_encoder_->metadata.end()) return false;
    int gh = 0, gw = 0, gd = 0;
    if (std::sscanf(shape_it->second.c_str(), "%d,%d,%d", &gh, &gw, &gd) != 3) return false;
    if (gh <= 0 || gw <= 0 || gd <= 0) return false;
    fs::path full = fs::path(bundle_dir_) / path_it->second;
    std::ifstream f(full, std::ios::binary);
    if (!f.is_open()) return false;
    const size_t count = static_cast<size_t>(gh) * gw * gd;
    lfm2_pos_grid_.resize(count);
    f.read(reinterpret_cast<char*>(lfm2_pos_grid_.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) { lfm2_pos_grid_.clear(); return false; }
    lfm2_pos_grid_h_ = gh;
    lfm2_pos_grid_w_ = gw;
    lfm2_pos_grid_dim_ = gd;
    return true;
}

void Model::run_vision_encoder_lfm2_vl(const std::string& image_path) {
    if (!vision_encoder_ || !vision_projector_) {
        throw std::runtime_error("lfm2_vl requires vision_encoder and vision_projector components");
    }
    if (!load_lfm2_vl_position_grid()) {
        throw std::runtime_error("lfm2_vl vision position-embedding grid is missing from the bundle");
    }
    if (!load_component_graph(*vision_encoder_)) throw std::runtime_error("failed to load vision_encoder");
    if (!load_component_graph(*vision_projector_)) throw std::runtime_error("failed to load vision_projector");

    media_features_.erase("image_features");
    media_feature_shapes_.erase("image_features");
    media_feature_precisions_.erase("image_features");

    encode_lfm2_vl_image_into_features(image_path);

    unload_component_graph(*vision_encoder_);
    unload_component_graph(*vision_projector_);
}

void Model::encode_lfm2_vl_image_into_features(const std::string& image_path) {
    Lfm2VlImagePreprocessed prep = preprocess_lfm2_vl_image(image_path, config_);
    const int dim = lfm2_pos_grid_dim_;
    const int factor = config_.downsample_factor ? static_cast<int>(config_.downsample_factor) : 2;
    const size_t max_patches = prep.max_num_patches;
    const size_t patch_dim = prep.patch_dim;
    const int cff = dim * factor * factor;

    const int pv_idx = input_index(*vision_encoder_, "pixel_values");
    const int pm_idx = input_index(*vision_encoder_, "pixel_attention_mask");
    const int pe_idx = input_index(*vision_encoder_, "positional_embeddings");
    const int vf_idx = input_index(*vision_projector_, "vision_features");
    if (pv_idx < 0 || pe_idx < 0 || vf_idx < 0) {
        throw std::runtime_error("lfm2_vl vision components are missing expected inputs");
    }

    const size_t enc_out_node = static_cast<size_t>(vision_encoder_->output_node_ids[0]);
    const size_t proj_out_node = static_cast<size_t>(vision_projector_->output_node_ids[0]);
    const auto& proj_out_desc = vision_projector_->graph->get_output_buffer(proj_out_node);
    const size_t proj_rows = proj_out_desc.shape.size() >= 2
        ? proj_out_desc.shape[proj_out_desc.shape.size() - 2] : 0;
    const size_t text_hidden = proj_out_desc.shape.empty() ? 0 : proj_out_desc.shape.back();
    const Precision proj_prec = proj_out_desc.precision;
    const size_t proj_elem = PrecisionTraits::size_of(proj_prec);
    if (proj_rows == 0 || text_hidden == 0) {
        throw std::runtime_error("lfm2_vl vision_projector has an invalid output shape");
    }
    const auto& enc_out_desc0 = vision_encoder_->graph->get_output_buffer(enc_out_node);
    const size_t enc_elem = PrecisionTraits::size_of(enc_out_desc0.precision);
    const size_t enc_out_elems = enc_elem ? enc_out_desc0.byte_size / enc_elem : 0;
    if (enc_out_elems < max_patches * static_cast<size_t>(dim)) {
        throw std::runtime_error("lfm2_vl vision_encoder output is smaller than the preprocessed patch grid");
    }

    auto write_mask = [&](int comp_idx, const int64_t* src, size_t count) {
        auto& buf = vision_encoder_->input_buffers[comp_idx];
        size_t node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[comp_idx]);
        const auto& d = vision_encoder_->graph->get_output_buffer(node);
        const size_t elem = PrecisionTraits::size_of(d.precision);
        const size_t cap = elem ? buf.size() / elem : 0;
        const size_t n = std::min(cap, count);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = src[i];
            switch (d.precision) {
                case Precision::FP32: reinterpret_cast<float*>(buf.data())[i] = static_cast<float>(v); break;
                case Precision::FP16: reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(v); break;
                case Precision::INT8: reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(v); break;
                default:
                    if (elem == 8) reinterpret_cast<int64_t*>(buf.data())[i] = v;
                    else if (elem == 4) reinterpret_cast<int32_t*>(buf.data())[i] = static_cast<int32_t>(v);
                    break;
            }
        }
        if (n < cap) std::memset(buf.data() + n * elem, 0, (cap - n) * elem);
    };

    auto write_float_input = [](Component& comp, int idx, const float* src, size_t count) {
        auto& buf = comp.input_buffers[idx];
        size_t node = static_cast<size_t>(comp.runtime_input_node_ids[idx]);
        const auto& d = comp.graph->get_output_buffer(node);
        write_typed_buffer(buf, d.precision, src, count * sizeof(float), Precision::FP32);
    };

    std::vector<uint8_t> image_features;
    size_t total_tokens = 0;

    std::vector<float> pos_buf(max_patches * static_cast<size_t>(dim));
    std::vector<float> enc_out_f(max_patches * static_cast<size_t>(dim));
    std::vector<float> unshuf;
    std::vector<float> proj_in;

    for (size_t t = 0; t < prep.spatial_shapes.size(); ++t) {
        const int h = prep.spatial_shapes[t].first;
        const int w = prep.spatial_shapes[t].second;
        const int num_tokens = (h / factor) * (w / factor);
        if (static_cast<size_t>(num_tokens) > proj_rows) {
            throw std::runtime_error("lfm2_vl sub-image exceeds the traced projector capacity");
        }

        std::fill(pos_buf.begin(), pos_buf.end(), 0.0f);
        interpolate_position_embeddings(
            lfm2_pos_grid_.data(), lfm2_pos_grid_h_, lfm2_pos_grid_w_, dim, h, w, pos_buf.data());

        write_float_input(*vision_encoder_, pv_idx,
                          prep.pixel_values.data() + t * max_patches * patch_dim, max_patches * patch_dim);
        if (pm_idx >= 0) {
            write_mask(pm_idx, prep.pixel_attention_mask.data() + t * max_patches, max_patches);
        }
        write_float_input(*vision_encoder_, pe_idx, pos_buf.data(), max_patches * static_cast<size_t>(dim));
        vision_encoder_->graph->execute();
        const auto& enc_desc = vision_encoder_->graph->get_output_buffer(enc_out_node);
        const void* enc_ptr = vision_encoder_->graph->get_output(enc_out_node);
        typed_buffer_to_float(enc_ptr, enc_desc.precision, enc_out_f.data(),
                              max_patches * static_cast<size_t>(dim));

        unshuf.assign(static_cast<size_t>(num_tokens) * cff, 0.0f);
        pixel_unshuffle(enc_out_f.data(), h, w, dim, factor, unshuf.data());

        proj_in.assign(proj_rows * static_cast<size_t>(cff), 0.0f);
        std::copy(unshuf.begin(), unshuf.end(), proj_in.begin());
        write_float_input(*vision_projector_, vf_idx, proj_in.data(), proj_in.size());

        vision_projector_->graph->execute();
        const void* proj_ptr = vision_projector_->graph->get_output(proj_out_node);
        const size_t row_bytes = text_hidden * proj_elem;
        const size_t append_bytes = static_cast<size_t>(num_tokens) * row_bytes;
        const size_t base = image_features.size();
        image_features.resize(base + append_bytes);
        std::memcpy(image_features.data() + base, proj_ptr, append_bytes);
        total_tokens += static_cast<size_t>(num_tokens);
    }

    auto& slot = media_features_["image_features"];
    const size_t prev_bytes = slot.size();
    slot.resize(prev_bytes + image_features.size());
    std::memcpy(slot.data() + prev_bytes, image_features.data(), image_features.size());
    auto& shape = media_feature_shapes_["image_features"];
    if (shape.size() != 2) shape = { total_tokens, text_hidden };
    else shape[0] += total_tokens;
    media_feature_precisions_["image_features"] = proj_prec;
}

void Model::run_audio_encoder_messages(const std::vector<std::vector<float>>& audio_features_per_message) {
    if (!audio_encoder_) return;
    if (audio_features_per_message.empty()) return;
    if (!load_component_graph(*audio_encoder_)) {
        throw std::runtime_error("failed to load audio_encoder");
    }
    for (const std::string& logical : audio_encoder_->logical_outputs) {
        media_features_.erase(logical);
        media_feature_shapes_.erase(logical);
        media_feature_precisions_.erase(logical);
    }
    for (const auto& mel : audio_features_per_message) {
        if (mel.empty()) continue;
        run_audio_encoder(mel);
    }
    if (cactus_default_backend() != ComputeBackend::METAL) {
        audio_encoder_->graph->release_runtime_buffers();
        audio_encoder_->graph->release_all_weight_pages();
    }
    unload_component_graph(*audio_encoder_);
}

void Model::run_audio_encoder(const std::vector<float>& audio_features) {
    if (!audio_encoder_) return;
    const std::vector<std::string> candidate_input_names = {"input_features", "audio_features"};
    int feature_idx = -1;
    for (const auto& name : candidate_input_names) {
        int idx = input_index(*audio_encoder_, name);
        if (idx >= 0) { feature_idx = idx; break; }
    }
    if (feature_idx < 0) {
        CACTUS_LOG_WARN("model", "audio_encoder has no input named input_features/audio_features; skipping");
        return;
    }
    const size_t feature_node = static_cast<size_t>(audio_encoder_->runtime_input_node_ids[feature_idx]);
    const auto& feature_desc = audio_encoder_->graph->get_output_buffer(feature_node);
    const size_t mel_bins = feature_desc.shape.size() >= 3
        ? feature_desc.shape[2]
        : static_cast<size_t>(config_.audio_input_feat_size);
    const size_t max_frames_per_chunk = feature_desc.shape.size() >= 2 ? feature_desc.shape[1] : 0;
    if (max_frames_per_chunk == 0 || mel_bins == 0) {
        CACTUS_LOG_WARN("model", "audio_encoder feature input has unexpected shape; skipping");
        return;
    }
    const size_t total_frames = audio_features.size() / mel_bins;
    const size_t num_chunks = total_frames == 0
        ? 1
        : (total_frames + max_frames_per_chunk - 1) / max_frames_per_chunk;

    const int mask_idx = input_index(*audio_encoder_, "input_features_mask");

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const size_t frame_start = chunk_idx * max_frames_per_chunk;
        const size_t frames_in_chunk = frame_start >= total_frames
            ? 0
            : std::min(max_frames_per_chunk, total_frames - frame_start);
        const size_t chunk_feature_elems = frames_in_chunk * mel_bins;
        const float* chunk_src = audio_features.data() + frame_start * mel_bins;

        auto& buf = audio_encoder_->input_buffers[feature_idx];
        if (feature_desc.precision == Precision::FP32) {
            const size_t n_bytes = std::min(chunk_feature_elems * sizeof(float), buf.size());
            std::memcpy(buf.data(), chunk_src, n_bytes);
            if (n_bytes < buf.size()) std::memset(buf.data() + n_bytes, 0, buf.size() - n_bytes);
        } else if (feature_desc.precision == Precision::FP16) {
            const size_t cap_elems = buf.size() / sizeof(__fp16);
            const size_t n_elems = std::min(chunk_feature_elems, cap_elems);
            __fp16* dst = reinterpret_cast<__fp16*>(buf.data());
            for (size_t i = 0; i < n_elems; ++i) dst[i] = static_cast<__fp16>(chunk_src[i]);
            if (n_elems < cap_elems) {
                std::memset(buf.data() + n_elems * sizeof(__fp16), 0, (cap_elems - n_elems) * sizeof(__fp16));
            }
        } else {
            const size_t n_elems = std::min(chunk_feature_elems, buf.size());
            int8_t* dst = reinterpret_cast<int8_t*>(buf.data());
            for (size_t i = 0; i < n_elems; ++i) dst[i] = static_cast<int8_t>(chunk_src[i]);
            if (n_elems < buf.size()) std::memset(buf.data() + n_elems, 0, buf.size() - n_elems);
        }

        if (mask_idx >= 0) {
            auto& mb = audio_encoder_->input_buffers[mask_idx];
            const size_t mask_node = static_cast<size_t>(audio_encoder_->runtime_input_node_ids[mask_idx]);
            const auto& mask_desc = audio_encoder_->graph->get_output_buffer(mask_node);
            const size_t elem = PrecisionTraits::size_of(mask_desc.precision);
            const size_t cap = elem ? mb.size() / elem : 0;
            const size_t n = std::min(cap, frames_in_chunk);
            for (size_t i = 0; i < n; ++i) {
                switch (mask_desc.precision) {
                    case Precision::FP32: reinterpret_cast<float*>(mb.data())[i] = 1.0f; break;
                    case Precision::FP16: reinterpret_cast<__fp16*>(mb.data())[i] = static_cast<__fp16>(1.0f); break;
                    case Precision::INT8: reinterpret_cast<int8_t*>(mb.data())[i] = 1; break;
                    default: reinterpret_cast<int8_t*>(mb.data())[i] = 1; break;
                }
            }
            if (n < cap) std::memset(mb.data() + n * elem, 0, (cap - n) * elem);
        }

        audio_encoder_->graph->execute();

        for (size_t i = 0; i < audio_encoder_->output_node_ids.size() && i < audio_encoder_->logical_outputs.size(); ++i) {
            const std::string& name = audio_encoder_->logical_outputs[i];
            const size_t node_id = static_cast<size_t>(audio_encoder_->output_node_ids[i]);
            const auto& desc = audio_encoder_->graph->get_output_buffer(node_id);
            void* ptr = audio_encoder_->graph->get_output(node_id);
            auto& slot = media_features_[name];
            const size_t prev_bytes = slot.size();
            slot.resize(prev_bytes + desc.byte_size);
            std::memcpy(slot.data() + prev_bytes, ptr, desc.byte_size);
            auto shape_it = media_feature_shapes_.find(name);
            if (shape_it == media_feature_shapes_.end() || shape_it->second.empty()) {
                media_feature_shapes_[name] = desc.shape;
            } else if (desc.shape.size() >= 2 && shape_it->second.size() == desc.shape.size()) {
                shape_it->second[shape_it->second.size() - 2] += desc.shape[desc.shape.size() - 2];
            }
            media_feature_precisions_[name] = desc.precision;
        }
    }
}

static float uncertainty_from_margin(float best, float second) {
    float confidence = 1.0f;
    if (std::isfinite(best) && std::isfinite(second)) {
        float margin = std::max(-60.0f, std::min(60.0f, best - second));
        confidence = 1.0f / (1.0f + std::exp(-margin));
    }
    return std::max(0.0f, std::min(1.0f, 1.0f - confidence));
}

uint32_t Model::argmax_logits_at(const BufferDesc& desc, void* ptr, size_t row_off, float* out_uncertainty) {
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    uint32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    float second_v = -std::numeric_limits<float>::infinity();
    const auto& tool_bias = tool_constrainer_.get_bias();
    const std::vector<float>* tool_dense = tool_constrainer_.get_dense_bias();
    {
        uint32_t gidx; float gbest, gsecond;
        if (!tool_dense && tool_bias.empty() && vocab_bias_.empty() && suppressed_token_id_ < 0 &&
            cactus_graph_metal_argmax(&gidx, &gbest, &gsecond)) {
            if (out_uncertainty) *out_uncertainty = uncertainty_from_margin(gbest, gsecond);
            return gidx;
        }
    }
    auto score_with_bias = [&](size_t token_id, float value) {
        if (tool_dense && token_id < tool_dense->size()) value += (*tool_dense)[token_id];
        auto tool_it = tool_bias.find(static_cast<uint32_t>(token_id));
        if (tool_it != tool_bias.end()) value += tool_it->second;
        auto vocab_it = vocab_bias_.find(static_cast<uint32_t>(token_id));
        if (vocab_it != vocab_bias_.end()) value += vocab_it->second;
        return value;
    };
    auto observe_logit = [&](size_t i, float v) {
        if (static_cast<int64_t>(i) == suppressed_token_id_) return;
        v = score_with_bias(i, v);
        if (v > best_v) {
            second_v = best_v;
            best_v = v;
            best = static_cast<uint32_t>(i);
        } else if (v > second_v) {
            second_v = v;
        }
    };
    if (desc.precision == Precision::FP32) {
        float* p = static_cast<float*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) observe_logit(i, p[i]);
    } else if (desc.precision == Precision::FP16) {
        __fp16* p = static_cast<__fp16*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) observe_logit(i, static_cast<float>(p[i]));
    } else {
        int8_t* p = static_cast<int8_t*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) observe_logit(i, static_cast<float>(p[i]));
    }
    if (out_uncertainty) *out_uncertainty = uncertainty_from_margin(best_v, second_v);
    return best;
}

uint32_t Model::argmax_component_logits(Component& comp, size_t logit_row, float* out_uncertainty) {
    size_t out_node = static_cast<size_t>(comp.output_node_ids.empty() ? 0 : comp.output_node_ids[0]);
    const auto& desc = comp.graph->get_output_buffer(out_node);
    void* ptr = comp.graph->get_output(out_node);
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    size_t seq = desc.shape.size() >= 2 ? desc.shape[desc.shape.size() - 2] : 1;
    size_t row = seq > 0 ? seq - 1 : 0;
    if (logit_row != std::numeric_limits<size_t>::max()) {
        row = std::min(logit_row, seq > 0 ? seq - 1 : 0);
    } else if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        row = std::min(last_logit_position_, seq > 0 ? seq - 1 : 0);
    }
    return argmax_logits_at(desc, ptr, row * vocab, out_uncertainty);
}

std::vector<uint32_t> Model::argmax_component_logits_batch(Component& comp, size_t batch) {
    std::vector<uint32_t> out(batch, 0);
    if (batch == 0) return out;
    size_t out_node = static_cast<size_t>(comp.output_node_ids.empty() ? 0 : comp.output_node_ids[0]);
    const auto& desc = comp.graph->get_output_buffer(out_node);
    void* ptr = comp.graph->get_output(out_node);
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    if (vocab == 0) return out;
    size_t total_rows = desc.total_size / vocab;
    size_t seq = total_rows / batch;
    for (size_t b = 0; b < batch; ++b) {
        size_t row = b * seq + (seq > 0 ? seq - 1 : 0);
        out[b] = argmax_logits_at(desc, ptr, row * vocab, nullptr);
    }
    return out;
}

uint32_t Model::argmax_last_logits(float* out_uncertainty) {
    return argmax_component_logits(*decoder_, std::numeric_limits<size_t>::max(), out_uncertainty);
}

void Model::prepare_sampling_context(float repetition_penalty) {
    samp_recent_.clear();
    samp_has_bias_ = false;
    samp_penalty_ = repetition_penalty;
    constexpr size_t kPenaltyWindow = 64;
    size_t start = token_history_.size() > kPenaltyWindow ? token_history_.size() - kPenaltyWindow : 0;
    for (size_t i = start; i < token_history_.size(); ++i) {
        uint32_t id = token_history_[i];
        bool dup = false;
        for (uint32_t r : samp_recent_) if (r == id) { dup = true; break; }
        if (!dup) samp_recent_.push_back(id);
    }
    const std::vector<float>* tool_dense = tool_constrainer_.get_dense_bias();
    const auto& tool_bias = tool_constrainer_.get_bias();
    if (tool_dense || !tool_bias.empty() || !vocab_bias_.empty()) {
        size_t n = tokenizer_ ? tokenizer_->get_vocab_size() : 0;
        if (tool_dense && tool_dense->size() > n) n = tool_dense->size();
        if (n > 0) {
            if (tool_dense) samp_bias_dense_.assign(tool_dense->begin(), tool_dense->end());
            else samp_bias_dense_.assign(n, 0.0f);
            if (samp_bias_dense_.size() < n) samp_bias_dense_.resize(n, 0.0f);
            for (const auto& kv : tool_bias)
                if (kv.first < samp_bias_dense_.size()) samp_bias_dense_[kv.first] += kv.second;
            for (const auto& kv : vocab_bias_)
                if (kv.first < samp_bias_dense_.size()) samp_bias_dense_[kv.first] += kv.second;
            samp_has_bias_ = true;
        }
    }
    samp_ctx_active_ = (repetition_penalty != 1.0f && !samp_recent_.empty()) ||
                       samp_has_bias_ || suppressed_token_id_ >= 0;
    if (samp_ctx_active_) {
        cactus_graph_set_sampling(samp_recent_.data(), (int)samp_recent_.size(), repetition_penalty,
                                  samp_has_bias_ ? samp_bias_dense_.data() : nullptr,
                                  samp_has_bias_ ? samp_bias_dense_.size() : 0,
                                  suppressed_token_id_);
    } else {
        cactus_graph_clear_sampling();
    }
}

uint32_t Model::sample_component_logits(Component& comp, float temperature, float top_p, size_t top_k,
                                        float min_p, bool greedy, float* out_uncertainty) {
    size_t out_node = static_cast<size_t>(comp.output_node_ids.empty() ? 0 : comp.output_node_ids[0]);
    const auto& desc = comp.graph->get_output_buffer(out_node);
    void* ptr = comp.graph->get_output(out_node);
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    size_t seq = desc.shape.size() >= 2 ? desc.shape[desc.shape.size() - 2] : 1;
    size_t row = seq > 0 ? seq - 1 : 0;
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        row = std::min(last_logit_position_, seq > 0 ? seq - 1 : 0);
    }
    size_t off = row * vocab;
    if (vocab == 0 || (desc.precision != Precision::FP16 && desc.precision != Precision::FP32)) {
        return argmax_component_logits(comp, std::numeric_limits<size_t>::max(), out_uncertainty);
    }
    const bool fp16 = desc.precision == Precision::FP16;
    __fp16* h = fp16 ? static_cast<__fp16*>(ptr) + off : nullptr;
    float*  f = fp16 ? nullptr : static_cast<float*>(ptr) + off;
    auto get = [&](size_t i) -> float { return fp16 ? (float)h[i] : f[i]; };
    auto put = [&](size_t i, float v) { if (fp16) h[i] = (__fp16)v; else f[i] = v; };

    if (samp_ctx_active_ && !cactus_graph_metal_adjusted()) {
        if (samp_penalty_ != 1.0f) {
            for (uint32_t id : samp_recent_) {
                if (id >= vocab) continue;
                float v = get(id);
                put(id, v > 0.0f ? v / samp_penalty_ : v * samp_penalty_);
            }
        }
        if (suppressed_token_id_ >= 0 && (size_t)suppressed_token_id_ < vocab)
            put((size_t)suppressed_token_id_, -65504.0f);
    }
    if (greedy) {
        uint32_t gidx; float gbest, gsecond;
        if ((!samp_has_bias_ || cactus_graph_metal_argmax_biased()) &&
            cactus_graph_metal_argmax(&gidx, &gbest, &gsecond)) {
            if (out_uncertainty) *out_uncertainty = uncertainty_from_margin(gbest, gsecond);
            return gidx;
        }
    }
    const float* bd = samp_has_bias_ ? samp_bias_dense_.data() : nullptr;
    const size_t bn = samp_has_bias_ ? samp_bias_dense_.size() : 0;
    auto biased = [&](size_t i) -> float {
        float v = get(i);
        if (i < bn) v += bd[i];
        return v;
    };

    if (greedy) {
        uint32_t best = 0;
        float bv = -std::numeric_limits<float>::infinity(), sv = bv;
        for (size_t i = 0; i < vocab; ++i) {
            float v = biased(i);
            if (v > bv) { sv = bv; bv = v; best = (uint32_t)i; }
            else if (v > sv) sv = v;
        }
        if (out_uncertainty) *out_uncertainty = uncertainty_from_margin(bv, sv);
        return best;
    }

    size_t K = std::min<size_t>(std::max<size_t>(top_k, 1), 512);
    std::vector<std::pair<float, uint32_t>> cand;
    cand.reserve(2 * K + 16);
    float kmin = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        float v = biased(i);
        if (v <= kmin) continue;
        cand.emplace_back(v, (uint32_t)i);
        if (cand.size() >= 2 * K) {
            std::nth_element(cand.begin(), cand.begin() + (K - 1), cand.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            cand.resize(K);
            kmin = cand.back().first;
            for (const auto& c : cand) if (c.first < kmin) kmin = c.first;
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (cand.size() > K) cand.resize(K);
    if (cand.empty()) return 0;
    if (out_uncertainty) {
        float sv = cand.size() > 1 ? cand[1].first : -std::numeric_limits<float>::infinity();
        *out_uncertainty = uncertainty_from_margin(cand[0].first, sv);
    }
    if (cand.size() == 1) return cand[0].second;

    float t = std::max(temperature, 1e-3f);
    float maxl = cand[0].first;
    std::vector<float> probs(cand.size());
    double denom = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
        probs[i] = std::exp((cand[i].first - maxl) / t);
        denom += probs[i];
    }
    size_t keep = cand.size();
    if (top_p > 0.0f && top_p < 1.0f) {
        double cum = 0.0;
        for (size_t i = 0; i < cand.size(); ++i) {
            cum += probs[i] / denom;
            if (cum >= top_p) { keep = i + 1; break; }
        }
    }
    if (min_p > 0.0f) {
        float cut = min_p * probs[0];
        while (keep > 1 && probs[keep - 1] < cut) --keep;
    }
    double mass = 0.0;
    for (size_t i = 0; i < keep; ++i) mass += probs[i];
    double r = std::uniform_real_distribution<double>(0.0, mass)(sample_rng_);
    double cum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        cum += probs[i];
        if (r <= cum) return cand[i].second;
    }
    return cand[keep - 1].second;
}

bool Model::prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token,
                                           float* out_uncertainty) {
    reset_prefill_stats();
    if (out_uncertainty) *out_uncertainty = 0.0f;
    if (tokens.empty() || !decoder_ || cache_total_seq_len_ != 0) {
        return false;
    }
    if (decode_route_ == DecodeRoute::ENCODER_CROSS_KV_STEP && encoder_cross_kv_source_kind_ == "text_tokens") {
        std::vector<uint32_t> source_tokens = tokens;
        std::vector<uint32_t> decoder_seed = {config_.decoder_start_token_id};
        if (!source_tokens.empty() && source_tokens.back() == config_.decoder_start_token_id) {
            decoder_seed = {source_tokens.back()};
            source_tokens.pop_back();
        }
        if (!prepare_encoder_cross_kv_from_text(source_tokens)) {
            return false;
        }
        std::vector<uint32_t> emitted = run_encoder_cross_kv_decode_loop(
            decoder_seed,
            1,
            {},
            nullptr);
        if (emitted.empty()) return false;
        out_token = emitted.front();
        return true;
    }
    cache_token_ids_ = tokens;
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        out_token = argmax_last_logits(out_uncertainty);
        record_sampled_token(out_token);
        return true;
    }
    if (decode_route_ == DecodeRoute::DIRECT_DECODER_STEP) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            run_step(tokens[i], i, i + 1 == tokens.size());
        }
        cache_total_seq_len_ = tokens.size();
        out_token = argmax_last_logits(out_uncertainty);
        record_sampled_token(out_token);
        return true;
    }
    if (!encoder_) {
        return false;
    }
    ChunkedPrefillResult chunked;
    if (decoder_prefill_) {
        chunked = run_chunked_prefill(tokens, cache_total_seq_len_, get_prefill_chunk_size(), true);
        if (chunked.logical_tokens == tokens.size() && chunked.padding_tokens > 0 && tokens.size() > 0) {
            // Cache already moved into the step component; drop the padded last row and re-run it for logits.
            set_cache_current_len(*decoder_, tokens.size() - 1);
            cache_total_seq_len_ = tokens.size() - 1;
            run_step(tokens.back(), cache_total_seq_len_, true, /*use_fused=*/false);
            ++cache_total_seq_len_;
            out_token = argmax_last_logits(out_uncertainty);
            record_sampled_token(out_token);
            last_prefill_scalar_tail_tokens_ = 1;
            maybe_roll_compact();
            return true;
        }
        cache_total_seq_len_ += chunked.logical_tokens;
    }
    for (size_t i = chunked.logical_tokens; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_, i + 1 == tokens.size(), /*use_fused=*/false);
        ++cache_total_seq_len_;
    }
    last_prefill_scalar_tail_tokens_ = tokens.size() - chunked.logical_tokens;
    if (chunked.logical_tokens == tokens.size() && chunked.logical_tokens > 0 && decoder_prefill_) {
        out_token = argmax_component_logits(*decoder_prefill_, chunked.last_logit_row, out_uncertainty);
    } else {
        out_token = argmax_last_logits(out_uncertainty);
    }
    record_sampled_token(out_token);
    maybe_roll_compact();
    return true;
}

void Model::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& /*profile_file*/, bool prepare_decode) {
    reset_prefill_stats();
    if (decode_route_ == DecodeRoute::ENCODER_CROSS_KV_STEP && encoder_cross_kv_source_kind_ == "text_tokens") {
        (void)prepare_decode;
        prepare_encoder_cross_kv_from_text(tokens);
        return;
    }
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        if (!context_tokens_.empty()) run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        cache_token_ids_ = context_tokens_;
        return;
    }
    ChunkedPrefillResult chunked = run_chunked_prefill(tokens, cache_total_seq_len_, get_prefill_chunk_size(), prepare_decode);
    cache_total_seq_len_ += chunked.logical_tokens;
    for (size_t i = chunked.logical_tokens; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_, /*read_logits=*/false, /*use_fused=*/false);
        ++cache_total_seq_len_;
    }
    cache_token_ids_.insert(cache_token_ids_.end(), tokens.begin(), tokens.end());
    last_prefill_scalar_tail_tokens_ = tokens.size() - chunked.logical_tokens;

    if (prepare_decode) {
        // After the prompt reaches full length here -- never mid-chunk -- bound it to target_len.
        maybe_roll_compact();
    }
}

void Model::prefill_with_images(const std::vector<uint32_t>& tokens,
                                const std::vector<std::string>& image_paths,
                                const std::string& profile_file) {
    prefill_with_media(tokens, image_paths, {}, profile_file);
}

void Model::prefill_with_audio(const std::vector<uint32_t>& tokens,
                               const std::vector<std::vector<float>>& audio_features_per_message,
                               const std::string& profile_file) {
    prefill_with_media(tokens, {}, audio_features_per_message, profile_file);
}

namespace {

void write_typed_buffer(std::vector<uint8_t>& buf, Precision dst_prec,
                        const void* src_data, size_t src_bytes, Precision src_prec) {
    if (dst_prec == src_prec) {
        size_t to_copy = std::min(src_bytes, buf.size());
        std::memcpy(buf.data(), src_data, to_copy);
        if (to_copy < buf.size()) std::memset(buf.data() + to_copy, 0, buf.size() - to_copy);
        return;
    }
    const size_t src_elem = PrecisionTraits::size_of(src_prec);
    const size_t dst_elem = PrecisionTraits::size_of(dst_prec);
    const size_t src_count = src_elem ? src_bytes / src_elem : 0;
    const size_t dst_count = dst_elem ? buf.size() / dst_elem : 0;
    const size_t n = std::min(src_count, dst_count);
    auto load_float = [&](size_t i) -> float {
        if (src_prec == Precision::FP16) return static_cast<float>(reinterpret_cast<const __fp16*>(src_data)[i]);
        if (src_prec == Precision::FP32) return reinterpret_cast<const float*>(src_data)[i];
        return static_cast<float>(reinterpret_cast<const int8_t*>(src_data)[i]);
    };
    for (size_t i = 0; i < n; ++i) {
        float v = load_float(i);
        if (dst_prec == Precision::FP16) reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(v);
        else if (dst_prec == Precision::FP32) reinterpret_cast<float*>(buf.data())[i] = v;
        else reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(v);
    }
    if (n < dst_count) {
        std::memset(buf.data() + n * dst_elem, 0, (dst_count - n) * dst_elem);
    }
}

static inline void write_int_element(uint8_t* buf, Precision prec, size_t index, int64_t v) {
    switch (prec) {
        case Precision::FP32: reinterpret_cast<float*>(buf)[index] = static_cast<float>(v); break;
        case Precision::FP16: reinterpret_cast<__fp16*>(buf)[index] = static_cast<__fp16>(v); break;
        case Precision::INT8: reinterpret_cast<int8_t*>(buf)[index] = static_cast<int8_t>(v); break;
        default: {
            size_t elem = PrecisionTraits::size_of(prec);
            if (elem == 8) reinterpret_cast<int64_t*>(buf)[index] = v;
            else if (elem == 4) reinterpret_cast<int32_t*>(buf)[index] = static_cast<int32_t>(v);
            break;
        }
    }
}

static inline size_t typed_buf_capacity(const std::vector<uint8_t>& buf, Precision prec) {
    size_t elem = PrecisionTraits::size_of(prec);
    return elem ? buf.size() / elem : 0;
}

static inline void zero_fill_remainder(std::vector<uint8_t>& buf, Precision prec, size_t written, size_t cap) {
    if (written < cap) {
        size_t elem = PrecisionTraits::size_of(prec);
        std::memset(buf.data() + written * elem, 0, (cap - written) * elem);
    }
}

void fill_int_buffer(std::vector<uint8_t>& buf, Precision prec, int64_t value, size_t count) {
    const size_t cap = typed_buf_capacity(buf, prec);
    const size_t n = std::min(cap, count);
    for (size_t i = 0; i < n; ++i) write_int_element(buf.data(), prec, i, value);
    zero_fill_remainder(buf, prec, n, cap);
}

void write_tokens_buffer(std::vector<uint8_t>& buf, Precision prec,
                         const std::vector<uint32_t>& tokens, size_t offset) {
    const size_t cap = typed_buf_capacity(buf, prec);
    const size_t avail = (offset < tokens.size()) ? (tokens.size() - offset) : 0;
    const size_t n = std::min(cap, avail);
    for (size_t i = 0; i < n; ++i) write_int_element(buf.data(), prec, i, static_cast<int64_t>(tokens[offset + i]));
    zero_fill_remainder(buf, prec, n, cap);
}

void write_int_vector_buffer(std::vector<uint8_t>& buf, Precision prec, const std::vector<int64_t>& values) {
    const size_t cap = typed_buf_capacity(buf, prec);
    const size_t n = std::min(cap, values.size());
    for (size_t i = 0; i < n; ++i) write_int_element(buf.data(), prec, i, values[i]);
    zero_fill_remainder(buf, prec, n, cap);
}

std::vector<int64_t> qwen3_vl_position_ids(const std::vector<uint32_t>& tokens,
                                           size_t capacity,
                                           const std::vector<Qwen3VlImagePreprocessed>& images,
                                           uint32_t image_token_id) {
    std::vector<int64_t> positions(3 * capacity, 0);
    size_t token_index = 0;
    size_t image_index = 0;
    int64_t current_pos = 0;
    while (token_index < tokens.size() && token_index < capacity) {
        if (image_token_id != 0 && tokens[token_index] == image_token_id) {
            if (image_index >= images.size()) {
                throw std::runtime_error("Qwen3-VL prompt contains more image token groups than image inputs");
            }
            const auto& image = images[image_index++];
            const size_t merge_size = 2;
            const size_t grid_t = image.grid_t;
            const size_t llm_grid_h = image.grid_h / merge_size;
            const size_t llm_grid_w = image.grid_w / merge_size;
            const size_t image_seq = grid_t * llm_grid_h * llm_grid_w;
            size_t count = 0;
            while (token_index + count < tokens.size()
                   && token_index + count < capacity
                   && tokens[token_index + count] == image_token_id) {
                ++count;
            }
            if (count != image_seq) {
                throw std::runtime_error("Qwen3-VL image token count does not match vision feature grid");
            }
            size_t local = 0;
            for (size_t t = 0; t < grid_t; ++t) {
                (void)t;
                for (size_t h = 0; h < llm_grid_h; ++h) {
                    for (size_t w = 0; w < llm_grid_w; ++w) {
                        size_t pos = token_index + local++;
                        positions[pos] = current_pos;
                        positions[capacity + pos] = current_pos + static_cast<int64_t>(h);
                        positions[2 * capacity + pos] = current_pos + static_cast<int64_t>(w);
                    }
                }
            }
            current_pos += static_cast<int64_t>(std::max(image.grid_h, image.grid_w) / merge_size);
            token_index += count;
            continue;
        }

        size_t text_count = 0;
        while (token_index + text_count < tokens.size()
               && token_index + text_count < capacity
               && (image_token_id == 0 || tokens[token_index + text_count] != image_token_id)) {
            size_t pos = token_index + text_count;
            int64_t value = current_pos + static_cast<int64_t>(text_count);
            positions[pos] = value;
            positions[capacity + pos] = value;
            positions[2 * capacity + pos] = value;
            ++text_count;
        }
        current_pos += static_cast<int64_t>(text_count);
        token_index += text_count;
    }
    return positions;
}

} // namespace

void Model::reset_encoder_cross_kv_route_state() {
    if (!decoder_) return;
    reset_component_cache_states(*decoder_);
    cache_total_seq_len_ = 0;
    token_history_.clear();
    encoder_cross_kv_ready_ = false;
    encoder_cross_kv_source_len_ = 0;
}

bool Model::finish_encoder_cross_kv_prepare() {
    if (!source_encoder_ || !decoder_cross_kv_ || !decoder_) return false;
    source_encoder_->graph->execute();
    copy_component_outputs_to_inputs(*source_encoder_, *decoder_cross_kv_);
    decoder_cross_kv_->graph->execute();
    copy_component_outputs_to_inputs(*source_encoder_, *decoder_);
    if (!copy_cross_kv_outputs_to_decoder_cache_inputs(*decoder_cross_kv_, *decoder_, encoder_cross_kv_source_len_)) {
        copy_component_outputs_to_inputs(*decoder_cross_kv_, *decoder_);
    }
    encoder_cross_kv_ready_ = true;
    return true;
}

bool Model::prepare_encoder_cross_kv_from_text(const std::vector<uint32_t>& tokens) {
    if (!source_encoder_ || !decoder_cross_kv_ || !decoder_ || tokens.empty()) return false;
    if (encoder_cross_kv_source_kind_ != "text_tokens") return false;

    reset_encoder_cross_kv_route_state();
    for (auto& buf : source_encoder_->input_buffers) {
        std::fill(buf.begin(), buf.end(), 0);
    }

    int ids_idx = input_index(*source_encoder_, "input_ids");
    if (ids_idx < 0) {
        CACTUS_LOG_ERROR("model", "source encoder missing input_ids input");
        return false;
    }
    size_t ids_node = static_cast<size_t>(source_encoder_->runtime_input_node_ids[ids_idx]);
    const auto& ids_desc = source_encoder_->graph->get_output_buffer(ids_node);
    size_t count = tokens.size();
    if (count > ids_desc.total_size) {
        CACTUS_LOG_WARN("model", "source token count " << count
            << " exceeds source encoder capacity " << ids_desc.total_size << "; truncating");
        count = ids_desc.total_size;
    }
    encoder_cross_kv_source_len_ = count;
    for (size_t i = 0; i < count; ++i) {
        write_int_input_at(*source_encoder_, "input_ids", i, static_cast<int64_t>(tokens[i]));
    }

    int mask_idx = input_index(*source_encoder_, "attention_mask");
    if (mask_idx >= 0) {
        for (size_t i = 0; i < count; ++i) {
            write_int_input_at(*source_encoder_, "attention_mask", i, 1);
        }
    }

    return finish_encoder_cross_kv_prepare();
}

bool Model::prepare_encoder_cross_kv_from_audio(const std::vector<float>& audio_features) {
    if (!source_encoder_ || !decoder_cross_kv_ || !decoder_) return false;
    int feature_idx = input_index(*source_encoder_, "input_features");
    if (feature_idx < 0) {
        feature_idx = input_index(*source_encoder_, "audio_features");
    }
    if (feature_idx < 0) {
        CACTUS_LOG_ERROR("model", "audio source encoder missing input_features/audio_features input");
        return false;
    }

    reset_encoder_cross_kv_route_state();

    auto& feature_buf = source_encoder_->input_buffers[feature_idx];
    size_t feature_node = static_cast<size_t>(source_encoder_->runtime_input_node_ids[feature_idx]);
    const auto& feature_desc = source_encoder_->graph->get_output_buffer(feature_node);
    write_typed_buffer(
        feature_buf,
        feature_desc.precision,
        audio_features.data(),
        audio_features.size() * sizeof(float),
        Precision::FP32);

    return finish_encoder_cross_kv_prepare();
}

bool Model::run_encoder_cross_kv_decoder_step(uint32_t token_id, size_t position) {
    if (!encoder_cross_kv_ready_ || !decoder_) return false;
    int ids_idx = input_index(*decoder_, "decoder_input_ids");
    const char* ids_name = "decoder_input_ids";
    if (ids_idx < 0) {
        ids_idx = input_index(*decoder_, "input_ids");
        ids_name = "input_ids";
    }
    int pos_idx = input_index(*decoder_, "position_ids");
    if (ids_idx < 0 || pos_idx < 0) {
        CACTUS_LOG_ERROR("model", "decoder_step missing decoder_input_ids/input_ids or position_ids input");
        return false;
    }
    write_int_input(*decoder_, ids_name, static_cast<int64_t>(token_id));
    write_int_input(*decoder_, "position_ids", static_cast<int64_t>(position));
    decoder_->graph->execute();
    return true;
}

std::vector<uint32_t> Model::run_encoder_cross_kv_decode_loop(
    const std::vector<uint32_t>& decoder_prompt_tokens,
    size_t max_tokens,
    const std::vector<std::vector<uint32_t>>& stop_token_sequences,
    const std::atomic<bool>* should_stop) {
    std::vector<uint32_t> emitted;
    if (!encoder_cross_kv_ready_ || decoder_prompt_tokens.empty() || max_tokens == 0) return emitted;

    std::vector<uint32_t> tokens = decoder_prompt_tokens;
    auto stopped = [&]() {
        for (const auto& stop_seq : stop_token_sequences) {
            if (stop_seq.empty() || emitted.size() < stop_seq.size()) continue;
            if (std::equal(stop_seq.rbegin(), stop_seq.rend(), emitted.rbegin())) return true;
        }
        return false;
    };

    for (size_t i = 0; i < max_tokens; ++i) {
        if (should_stop && should_stop->load()) break;
        const size_t start = cache_total_seq_len_ < tokens.size() ? cache_total_seq_len_ : tokens.size() - 1;
        for (size_t pos = start; pos < tokens.size(); ++pos) {
            if (!run_encoder_cross_kv_decoder_step(tokens[pos], pos)) return emitted;
        }
        cache_total_seq_len_ = tokens.size();

        uint32_t next_token = argmax_last_logits();
        record_sampled_token(next_token);
        emitted.push_back(next_token);
        if (stopped()) break;
        tokens.push_back(next_token);
    }

    return emitted;
}

bool Model::build_lm_encoder_outputs_dynamic_gemma4(
    const std::vector<uint32_t>& tokens,
    std::map<std::string, std::vector<uint8_t>>& store_bytes,
    std::map<std::string, Precision>& store_prec,
    std::map<std::string, std::vector<size_t>>& store_shape) {
    if (!encoder_ || !lm_encoder_media_step_ || tokens.empty()) return false;

    const uint32_t image_tok = config_.image_token_id;
    const uint32_t audio_tok = config_.audio_token_id;

    auto audio_it = media_features_.find("audio_features");
    const bool have_audio_features = audio_it != media_features_.end() && !audio_it->second.empty();
    size_t audio_rows = 0;
    size_t audio_row_bytes = 0;
    Precision audio_prec = Precision::FP16;
    if (have_audio_features) {
        const auto& shape = media_feature_shapes_["audio_features"];
        audio_rows = shape.size() >= 2 ? shape[shape.size() - 2] : 0;
        audio_prec = media_feature_precisions_["audio_features"];
        audio_row_bytes = audio_rows > 0 ? audio_it->second.size() / audio_rows : audio_it->second.size();
    }

    auto image_it = media_features_.find("image_features");
    const bool have_image_features = image_it != media_features_.end() && !image_it->second.empty();
    size_t image_rows = 0;
    size_t image_row_bytes = 0;
    Precision image_prec = Precision::FP16;
    if (have_image_features) {
        const auto& shape = media_feature_shapes_["image_features"];
        image_rows = shape.size() >= 2 ? shape[shape.size() - 2] : 0;
        image_prec = media_feature_precisions_["image_features"];
        image_row_bytes = image_rows > 0 ? image_it->second.size() / image_rows : image_it->second.size();
    }

    struct OutputInfo {
        std::string name;
        int text_idx = -1;
        int media_idx = -1;
        size_t per_token_bytes = 0;
        Precision precision = Precision::FP16;
        std::vector<size_t> shape_template;
    };

    std::vector<OutputInfo> outputs;
    for (size_t i = 0; i < encoder_->logical_outputs.size() && i < encoder_->output_node_ids.size(); ++i) {
        OutputInfo info;
        info.name = encoder_->logical_outputs[i];
        info.text_idx = static_cast<int>(i);
        info.media_idx = output_index(*lm_encoder_media_step_, info.name);
        if (info.media_idx < 0) {
            throw std::runtime_error("lm_encoder_media_step missing output " + info.name);
        }
        size_t node_id = static_cast<size_t>(encoder_->output_node_ids[i]);
        const auto& desc = encoder_->graph->get_output_buffer(node_id);
        info.per_token_bytes = desc.byte_size;
        info.precision = desc.precision;
        info.shape_template = desc.shape;
        outputs.push_back(std::move(info));
    }
    if (outputs.empty()) return false;

    const size_t token_count = tokens.size();
    for (const auto& info : outputs) {
        store_bytes[info.name].assign(token_count * info.per_token_bytes, 0);
        store_prec[info.name] = info.precision;
        std::vector<size_t> shape = info.shape_template;
        if (shape.size() >= 2 && shape[shape.size() - 2] == 1) {
            shape[shape.size() - 2] = token_count;
        } else if (shape.size() == 1) {
            shape[0] = token_count;
        }
        store_shape[info.name] = std::move(shape);
    }

    size_t audio_idx = 0;
    size_t image_idx = 0;
    for (size_t pos = 0; pos < token_count; ++pos) {
        const uint32_t token = tokens[pos];
        Component* component = encoder_;
        const uint8_t* media_row = nullptr;
        size_t media_row_bytes = 0;
        Precision media_prec = Precision::FP16;

        if (audio_tok != 0 && token == audio_tok && have_audio_features) {
            if (audio_idx >= audio_rows) {
                throw std::runtime_error("Gemma4 prompt contains more audio tokens than audio feature rows");
            }
            component = lm_encoder_media_step_;
            media_row = audio_it->second.data() + audio_idx * audio_row_bytes;
            media_row_bytes = audio_row_bytes;
            media_prec = audio_prec;
            ++audio_idx;
        } else if (image_tok != 0 && token == image_tok && have_image_features) {
            if (image_idx >= image_rows) {
                throw std::runtime_error("Gemma4 prompt contains more image tokens than image feature rows");
            }
            component = lm_encoder_media_step_;
            media_row = image_it->second.data() + image_idx * image_row_bytes;
            media_row_bytes = image_row_bytes;
            media_prec = image_prec;
            ++image_idx;
        }

        if (component == lm_encoder_media_step_) {
            int embeds_idx = input_index(*component, "inputs_embeds");
            if (embeds_idx < 0) {
                throw std::runtime_error("lm_encoder_media_step missing inputs_embeds input");
            }
            auto& buf = component->input_buffers[embeds_idx];
            size_t node_id = static_cast<size_t>(component->runtime_input_node_ids[embeds_idx]);
            const auto& desc = component->graph->get_output_buffer(node_id);
            write_typed_buffer(buf, desc.precision, media_row, media_row_bytes, media_prec);
            write_int_input(*component, "input_ids", 0);
            write_int_input(*component, "position_ids", static_cast<int64_t>(pos));
        } else {
            write_int_input(*component, "input_ids", static_cast<int64_t>(token));
            write_int_input(*component, "position_ids", static_cast<int64_t>(pos));
        }
        component->graph->execute();

        for (const auto& info : outputs) {
            int out_idx = component == encoder_ ? info.text_idx : info.media_idx;
            size_t node_id = static_cast<size_t>(component->output_node_ids[out_idx]);
            const auto& desc = component->graph->get_output_buffer(node_id);
            if (desc.byte_size != info.per_token_bytes || desc.precision != info.precision) {
                throw std::runtime_error("Gemma4 dynamic output shape mismatch for " + info.name);
            }
            const void* ptr = component->graph->get_output(node_id);
            std::memcpy(store_bytes[info.name].data() + pos * info.per_token_bytes, ptr, info.per_token_bytes);
        }
        component->graph->release_runtime_buffers();
    }
    encoder_->graph->release_all_weight_pages();
    lm_encoder_media_step_->graph->release_all_weight_pages();
    return true;
}


bool Model::run_chunk_prefill_path(const std::vector<uint32_t>& tokens,
                                   const std::vector<std::string>& image_paths,
                                   const std::vector<std::vector<float>>& audio_features_per_message) {
    if (cache_total_seq_len_ > 0) return false;
    const bool have_images = !image_paths.empty() && vision_encoder_ != nullptr;
    bool any_audio = false;
    for (const auto& mel : audio_features_per_message) { if (!mel.empty()) { any_audio = true; break; } }
    const bool have_audio = any_audio && audio_encoder_ != nullptr;
    std::vector<Qwen3VlImagePreprocessed> qwen_images;

    if (have_images) {
        if (!load_component_graph(*vision_encoder_)) {
            throw std::runtime_error("failed to load vision_encoder");
        }
        for (const std::string& logical : vision_encoder_->logical_outputs) {
            media_features_.erase(logical);
            media_feature_shapes_.erase(logical);
            media_feature_precisions_.erase(logical);
        }
        const bool lfm2_vision = family_ == "lfm2_vl";
        if (lfm2_vision) {
            if (!vision_projector_) throw std::runtime_error("lfm2_vl requires a vision_projector component");
            if (!load_lfm2_vl_position_grid()) {
                throw std::runtime_error("lfm2_vl vision position-embedding grid is missing from the bundle");
            }
            if (!load_component_graph(*vision_projector_)) throw std::runtime_error("failed to load vision_projector");
            media_features_.erase("image_features");
            media_feature_shapes_.erase("image_features");
            media_feature_precisions_.erase("image_features");
        }
        for (const auto& path : image_paths) {
            if (family_ == "lfm2_vl") {
                encode_lfm2_vl_image_into_features(path);
                continue;
            } else if (family_ == "qwen3_5" || family_ == "qwen3_vl" || config_.model_type == Config::ModelType::QWEN) {
                Qwen3VlImagePreprocessed prep = preprocess_qwen3_vl_image(path, config_);
                int pv_idx = input_index(*vision_encoder_, "pixel_values");
                if (pv_idx < 0) {
                    throw std::runtime_error("Qwen3-VL vision_encoder missing pixel_values input");
                }
                auto& pv_buf = vision_encoder_->input_buffers[pv_idx];
                size_t pv_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pv_idx]);
                const auto& pv_desc = vision_encoder_->graph->get_output_buffer(pv_node);
                write_typed_buffer(pv_buf, pv_desc.precision,
                                   prep.pixel_values.data(),
                                   prep.pixel_values.size() * sizeof(float),
                                   Precision::FP32);
                qwen_images.push_back(std::move(prep));
            } else {
                Gemma4ImagePreprocessed prep = preprocess_gemma4_image(path, config_);
                int pv_idx = input_index(*vision_encoder_, "pixel_values");
                if (pv_idx >= 0) {
                    auto& pv_buf = vision_encoder_->input_buffers[pv_idx];
                    size_t pv_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pv_idx]);
                    const auto& pv_desc = vision_encoder_->graph->get_output_buffer(pv_node);
                    write_typed_buffer(pv_buf, pv_desc.precision,
                                       prep.pixel_values.data(),
                                       prep.pixel_values.size() * sizeof(float),
                                       Precision::FP32);
                }
                int pp_idx = input_index(*vision_encoder_, "pixel_position_ids");
                if (pp_idx >= 0) {
                    auto& pp_buf = vision_encoder_->input_buffers[pp_idx];
                    size_t pp_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pp_idx]);
                    const auto& pp_desc = vision_encoder_->graph->get_output_buffer(pp_node);
                    const size_t elem = PrecisionTraits::size_of(pp_desc.precision);
                    const size_t cap = elem ? pp_buf.size() / elem : 0;
                    const size_t n = std::min(cap, prep.pixel_position_ids.size());
                    for (size_t i = 0; i < n; ++i) {
                        int64_t v = prep.pixel_position_ids[i];
                        switch (pp_desc.precision) {
                            case Precision::FP32: reinterpret_cast<float*>(pp_buf.data())[i] = static_cast<float>(v); break;
                            case Precision::FP16: reinterpret_cast<__fp16*>(pp_buf.data())[i] = static_cast<__fp16>(v); break;
                            case Precision::INT8: reinterpret_cast<int8_t*>(pp_buf.data())[i] = static_cast<int8_t>(v); break;
                            default:
                                if (elem == 8) reinterpret_cast<int64_t*>(pp_buf.data())[i] = v;
                                else if (elem == 4) reinterpret_cast<int32_t*>(pp_buf.data())[i] = static_cast<int32_t>(v);
                                break;
                        }
                    }
                    if (n < cap) std::memset(pp_buf.data() + n * elem, 0, (cap - n) * elem);
                }
            }
            vision_encoder_->graph->execute();
            for (size_t i = 0; i < vision_encoder_->output_node_ids.size()
                              && i < vision_encoder_->logical_outputs.size(); ++i) {
                const std::string& name = vision_encoder_->logical_outputs[i];
                size_t node_id = static_cast<size_t>(vision_encoder_->output_node_ids[i]);
                const auto& desc = vision_encoder_->graph->get_output_buffer(node_id);
                void* ptr = vision_encoder_->graph->get_output(node_id);
                auto& slot = media_features_[name];
                const size_t prev_bytes = slot.size();
                slot.resize(prev_bytes + desc.byte_size);
                std::memcpy(slot.data() + prev_bytes, ptr, desc.byte_size);
                auto shape_it = media_feature_shapes_.find(name);
                if (shape_it == media_feature_shapes_.end() || shape_it->second.empty()) {
                    media_feature_shapes_[name] = desc.shape;
                } else if (desc.shape.size() >= 2 && shape_it->second.size() == desc.shape.size()) {
                    shape_it->second[shape_it->second.size() - 2] += desc.shape[desc.shape.size() - 2];
                }
                media_feature_precisions_[name] = desc.precision;
            }
            vision_encoder_->graph->release_runtime_buffers();
            vision_encoder_->graph->release_all_weight_pages();
        }
        if (lfm2_vision) {
            unload_component_graph(*vision_projector_);
        }
        unload_component_graph(*vision_encoder_);
    }

    if (have_audio) {
        run_audio_encoder_messages(audio_features_per_message);
    }

    std::map<std::string, std::vector<uint8_t>> store_bytes;
    std::map<std::string, Precision> store_prec;
    std::map<std::string, std::vector<size_t>> store_shape;
    const bool needs_dynamic_walk = family_ == "gemma4" && (have_images || have_audio) && lm_encoder_media_step_ != nullptr;

    if (needs_dynamic_walk) {
        if (!build_lm_encoder_outputs_dynamic_gemma4(tokens, store_bytes, store_prec, store_shape)) {
            return false;
        }
    } else {
        if (!load_component_graph(*lm_encoder_)) {
            throw std::runtime_error("failed to load lm_encoder");
        }
        int ids_idx = input_index(*lm_encoder_, "input_ids");
        if (ids_idx >= 0) {
            auto& ids_buf = lm_encoder_->input_buffers[ids_idx];
            size_t ids_node = static_cast<size_t>(lm_encoder_->runtime_input_node_ids[ids_idx]);
            const auto& ids_desc = lm_encoder_->graph->get_output_buffer(ids_node);
            write_tokens_buffer(ids_buf, ids_desc.precision, tokens, 0);
        }

        int mask_idx = input_index(*lm_encoder_, "attention_mask");
        if (mask_idx >= 0) {
            auto& mb = lm_encoder_->input_buffers[mask_idx];
            size_t mnode = static_cast<size_t>(lm_encoder_->runtime_input_node_ids[mask_idx]);
            const auto& mdesc = lm_encoder_->graph->get_output_buffer(mnode);
            fill_int_buffer(mb, mdesc.precision, 1, tokens.size());
        }

        int pos_idx = input_index(*lm_encoder_, "position_ids");
        if (pos_idx >= 0) {
            auto& pos_buf = lm_encoder_->input_buffers[pos_idx];
            size_t pos_node = static_cast<size_t>(lm_encoder_->runtime_input_node_ids[pos_idx]);
            const auto& pos_desc = lm_encoder_->graph->get_output_buffer(pos_node);
            if (pos_desc.shape.size() >= 3 && pos_desc.shape[0] == 3 && !qwen_images.empty()) {
                size_t capacity = pos_desc.shape[pos_desc.shape.size() - 1];
                auto positions = qwen3_vl_position_ids(tokens, capacity, qwen_images, config_.image_token_id);
                write_int_vector_buffer(pos_buf, pos_desc.precision, positions);
            }
        }

        for (const auto& kv : media_features_) {
            const std::string& name = kv.first;
            int idx = input_index(*lm_encoder_, name);
            if (idx < 0) continue;
            auto& dst_buf = lm_encoder_->input_buffers[idx];
            size_t node_id = static_cast<size_t>(lm_encoder_->runtime_input_node_ids[idx]);
            const auto& desc = lm_encoder_->graph->get_output_buffer(node_id);
            Precision src_prec = media_feature_precisions_[name];
            write_typed_buffer(dst_buf, desc.precision,
                               kv.second.data(), kv.second.size(), src_prec);
        }
        lm_encoder_->graph->execute();

        for (size_t i = 0; i < lm_encoder_->output_node_ids.size()
                          && i < lm_encoder_->logical_outputs.size(); ++i) {
            const std::string& name = lm_encoder_->logical_outputs[i];
            size_t node_id = static_cast<size_t>(lm_encoder_->output_node_ids[i]);
            const auto& desc = lm_encoder_->graph->get_output_buffer(node_id);
            void* ptr = lm_encoder_->graph->get_output(node_id);
            auto& slot = store_bytes[name];
            slot.assign(desc.byte_size, 0);
            std::memcpy(slot.data(), ptr, desc.byte_size);
            store_prec[name] = desc.precision;
            store_shape[name] = desc.shape;
        }

        if (have_images && family_ == "lfm2_vl") {
            auto feat_it = media_features_.find("image_features");
            auto emb_it = store_bytes.find("inputs_embeds");
            if (feat_it != media_features_.end() && emb_it != store_bytes.end() && tokenizer_) {
                const auto& emb_shape = store_shape["inputs_embeds"];
                const auto& feat_shape = media_feature_shapes_["image_features"];
                const size_t hidden = emb_shape.empty() ? 0 : emb_shape.back();
                const size_t emb_seq = emb_shape.size() >= 2 ? emb_shape[emb_shape.size() - 2] : 0;
                const size_t feat_rows = feat_shape.empty() ? 0 : feat_shape[0];
                const Precision emb_prec = store_prec["inputs_embeds"];
                const Precision feat_prec = media_feature_precisions_["image_features"];
                const size_t emb_elem = PrecisionTraits::size_of(emb_prec);
                const size_t feat_elem = PrecisionTraits::size_of(feat_prec);
                const uint32_t image_tok = tokenizer_->get_image_token_id();
                uint8_t* emb_data = emb_it->second.data();
                const uint8_t* feat_data = feat_it->second.data();
                std::vector<float> frow(hidden);
                size_t fcur = 0;
                const size_t limit = std::min(emb_seq, tokens.size());
                for (size_t p = 0; p < limit && fcur < feat_rows; ++p) {
                    if (tokens[p] != image_tok) continue;
                    typed_buffer_to_float(feat_data + fcur * hidden * feat_elem, feat_prec, frow.data(), hidden);
                    uint8_t* dst = emb_data + p * hidden * emb_elem;
                    for (size_t d = 0; d < hidden; ++d) {
                        switch (emb_prec) {
                            case Precision::FP32: reinterpret_cast<float*>(dst)[d] = frow[d]; break;
                            case Precision::FP16: reinterpret_cast<__fp16*>(dst)[d] = static_cast<__fp16>(frow[d]); break;
                            default: reinterpret_cast<float*>(dst)[d] = frow[d]; break;
                        }
                    }
                    ++fcur;
                }
            }
        }

        lm_encoder_->graph->release_runtime_buffers();
        lm_encoder_->graph->release_all_weight_pages();
        unload_component_graph(*lm_encoder_);
    }

    auto embeds_shape_it = store_shape.find("inputs_embeds");
    if (embeds_shape_it == store_shape.end()) {
        return false;
    }
    size_t full_seq = 0;
    {
        const auto& sh = embeds_shape_it->second;
        if (sh.size() >= 3) full_seq = sh[sh.size() - 2];
        else if (!sh.empty()) full_seq = sh[0];
    }
    if (full_seq == 0) return false;

    size_t chunk_seq = 0;
    {
        if (!load_component_graph(*decoder_prefill_chunk_)) {
            throw std::runtime_error("failed to load decoder_prefill_chunk");
        }
        int idx = input_index(*decoder_prefill_chunk_, "inputs_embeds");
        if (idx < 0) return false;
        size_t node_id = static_cast<size_t>(decoder_prefill_chunk_->runtime_input_node_ids[idx]);
        const auto& desc = decoder_prefill_chunk_->graph->get_output_buffer(node_id);
        const auto& sh = desc.shape;
        if (sh.size() >= 3) chunk_seq = sh[sh.size() - 2];
        else if (!sh.empty()) chunk_seq = sh[0];
    }
    if (chunk_seq == 0) return false;

    std::map<std::string, size_t> per_pos_bytes;
    for (const auto& kv : store_bytes) {
        per_pos_bytes[kv.first] = kv.second.size() / full_seq;
    }

    size_t valid_seq = tokens.size();
    auto mask_it = store_bytes.find("attention_mask");
    if (mask_it != store_bytes.end() && per_pos_bytes.count("attention_mask")) {
        Precision mp = store_prec["attention_mask"];
        size_t per = per_pos_bytes["attention_mask"];
        const uint8_t* mp_data = mask_it->second.data();
        size_t count = 0;
        for (size_t i = 0; i < full_seq; ++i) {
            const uint8_t* pos = mp_data + i * per;
            bool nonzero = false;
            switch (mp) {
                case Precision::INT8:
                    nonzero = (*reinterpret_cast<const int8_t*>(pos) != 0); break;
                case Precision::FP16:
                    nonzero = (static_cast<float>(*reinterpret_cast<const __fp16*>(pos)) != 0.0f); break;
                case Precision::FP32:
                    nonzero = (*reinterpret_cast<const float*>(pos) != 0.0f); break;
                default:
                    if (per == 8) nonzero = (*reinterpret_cast<const int64_t*>(pos) != 0);
                    else if (per == 4) nonzero = (*reinterpret_cast<const int32_t*>(pos) != 0);
                    else nonzero = (*pos != 0);
                    break;
            }
            if (nonzero) ++count;
        }
        if (count > 0) valid_seq = count;
    }
    valid_seq = std::min(valid_seq, full_seq);
    const size_t whole_chunks_end = (valid_seq / chunk_seq) * chunk_seq;
    for (size_t chunk_start = 0; chunk_start < whole_chunks_end; chunk_start += chunk_seq) {
        for (const auto& kv : store_bytes) {
            const std::string& name = kv.first;
            int idx = input_index(*decoder_prefill_chunk_, name);
            if (idx < 0) continue;
            auto& dst_buf = decoder_prefill_chunk_->input_buffers[idx];
            size_t node_id = static_cast<size_t>(decoder_prefill_chunk_->runtime_input_node_ids[idx]);
            const auto& desc = decoder_prefill_chunk_->graph->get_output_buffer(node_id);
            Precision src_prec = store_prec[name];
            size_t src_per_pos = per_pos_bytes[name];
            const uint8_t* src_ptr = kv.second.data() + chunk_start * src_per_pos;
            size_t src_slice_bytes = chunk_seq * src_per_pos;
            write_typed_buffer(dst_buf, desc.precision, src_ptr, src_slice_bytes, src_prec);
        }
        decoder_prefill_chunk_->graph->execute();
    }
    if (whole_chunks_end > 0 && decoder_ != nullptr) {
        move_cache_states(*decoder_prefill_chunk_, *decoder_);
        decoder_prefill_chunk_->graph->release_runtime_buffers();
        unload_component_graph(*decoder_prefill_chunk_);
    }
    for (size_t pos = whole_chunks_end; pos < valid_seq; ++pos) {
        for (const auto& kv : store_bytes) {
            const std::string& name = kv.first;
            int idx = input_index(*decoder_, name);
            if (idx < 0) continue;
            auto& dst_buf = decoder_->input_buffers[idx];
            size_t node_id = static_cast<size_t>(decoder_->runtime_input_node_ids[idx]);
            const auto& desc = decoder_->graph->get_output_buffer(node_id);
            Precision src_prec = store_prec[name];
            size_t src_per_pos = per_pos_bytes[name];
            const uint8_t* src_ptr = kv.second.data() + pos * src_per_pos;
            write_typed_buffer(dst_buf, desc.precision, src_ptr, src_per_pos, src_prec);
        }
        decoder_->graph->execute();
    }
    cache_total_seq_len_ += valid_seq;
    return true;
}

void Model::prefill_with_media(const std::vector<uint32_t>& tokens,
                               const std::vector<std::string>& image_paths,
                               const std::vector<std::vector<float>>& audio_features_per_message,
                               const std::string& profile_file) {
    if (tokens.empty()) return;
    if (!image_paths.empty() && vision_encoder_ == nullptr) {
        throw std::runtime_error("Model bundle does not include a vision_encoder for image input");
    }
    bool any_audio = false;
    for (const auto& mel : audio_features_per_message) { if (!mel.empty()) { any_audio = true; break; } }
    if (any_audio && audio_encoder_ == nullptr) {
        throw std::runtime_error("Model bundle does not include an audio_encoder for audio input");
    }
    const bool have_images = !image_paths.empty();
    const bool have_audio = any_audio;
    if (!have_images && !have_audio) {
        prefill(tokens, get_prefill_chunk_size(), profile_file);
        return;
    }

    const bool can_chunk_prefill =
        lm_encoder_ != nullptr && decoder_prefill_chunk_ != nullptr &&
        (vision_encoder_ != nullptr || audio_encoder_ != nullptr);
    if (can_chunk_prefill) {
        if (run_chunk_prefill_path(tokens, image_paths, audio_features_per_message)) {
            (void)profile_file;
            return;
        }
    }
    if (!supports_warm_media_injection()) {
        CACTUS_LOG_WARN("model", "Bundle supports neither chunk-prefill nor warm media injection; falling back to text-only prefill");
        prefill(tokens, get_prefill_chunk_size(), profile_file);
        return;
    }

    if (have_images) {
        for (const auto& path : image_paths) {
            run_vision_encoder(path);
        }
    }
    if (have_audio) {
        run_audio_encoder_messages(audio_features_per_message);
    }

    std::string image_feature_name;
    Precision image_feature_prec = Precision::FP16;
    size_t image_row_bytes = 0;
    if (have_images) {
        const std::vector<std::string> candidates = {"image_features", "image_embeddings", "vision_features", "inputs_embeds"};
        for (const auto& name : candidates) {
            if (media_features_.count(name)) { image_feature_name = name; break; }
        }
        if (image_feature_name.empty() && !media_features_.empty()) {
            image_feature_name = media_features_.begin()->first;
        }
        if (!image_feature_name.empty()) {
            const auto& shape = media_feature_shapes_[image_feature_name];
            image_feature_prec = media_feature_precisions_[image_feature_name];
            if (shape.size() >= 2) {
                size_t rows = (shape.size() >= 3) ? shape[shape.size() - 2] : shape[0];
                size_t total = media_features_[image_feature_name].size();
                image_row_bytes = rows > 0 ? total / rows : total;
            } else {
                image_row_bytes = media_features_[image_feature_name].size();
            }
        }
    }

    std::string audio_feature_name;
    Precision audio_feature_prec = Precision::FP16;
    size_t audio_row_bytes = 0;
    if (have_audio) {
        const std::vector<std::string> candidates = {"audio_features", "audio_embeddings", "encoder_hidden_states", "inputs_embeds"};
        for (const auto& name : candidates) {
            if (media_features_.count(name) && name != image_feature_name) { audio_feature_name = name; break; }
        }
        if (audio_feature_name.empty()) {
            for (const auto& kv : media_features_) {
                if (kv.first != image_feature_name) { audio_feature_name = kv.first; break; }
            }
        }
        if (!audio_feature_name.empty()) {
            const auto& shape = media_feature_shapes_[audio_feature_name];
            audio_feature_prec = media_feature_precisions_[audio_feature_name];
            if (shape.size() >= 2) {
                size_t rows = (shape.size() >= 3) ? shape[shape.size() - 2] : shape[0];
                size_t total = media_features_[audio_feature_name].size();
                audio_row_bytes = rows > 0 ? total / rows : total;
            } else {
                audio_row_bytes = media_features_[audio_feature_name].size();
            }
        }
    }

    size_t image_consumed = 0;
    size_t audio_consumed = 0;
    const uint32_t image_tok = config_.image_token_id;
    const uint32_t audio_tok = config_.audio_token_id;
    const bool warm_media = supports_warm_media_injection();

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t t = tokens[i];
        size_t pos = cache_total_seq_len_ + i;
        if (image_tok != 0 && t == image_tok && !image_feature_name.empty() && warm_media) {
            const auto& feat = media_features_[image_feature_name];
            const uint8_t* row = feat.data() + image_consumed * image_row_bytes;
            if (image_consumed * image_row_bytes + image_row_bytes <= feat.size()) {
                run_media_step(pos, row, image_row_bytes, image_feature_prec);
                ++image_consumed;
                continue;
            }
        }
        if (audio_tok != 0 && t == audio_tok && !audio_feature_name.empty() && warm_media) {
            const auto& feat = media_features_[audio_feature_name];
            const uint8_t* row = feat.data() + audio_consumed * audio_row_bytes;
            if (audio_consumed * audio_row_bytes + audio_row_bytes <= feat.size()) {
                run_media_step(pos, row, audio_row_bytes, audio_feature_prec);
                ++audio_consumed;
                continue;
            }
        }
        run_step(t, pos, false);
    }
    cache_total_seq_len_ += tokens.size();
    cache_token_ids_.insert(cache_token_ids_.end(), tokens.begin(), tokens.end());
    (void)profile_file;
}

uint32_t Model::decode(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                        size_t top_k, const std::string& /*profile_file*/, float* out_entropy,
                        float min_p, float repetition_penalty) {
    if (tokens.empty()) return 0;
    float temp = temperature < 0.0f ? config_.default_temperature : temperature;
    float tp = (top_p <= 0.0f || top_p > 1.0f) ? config_.default_top_p : top_p;
    size_t tk = top_k == 0 ? config_.default_top_k : top_k;
    const bool greedy = temp <= 0.011f;
    prepare_sampling_context(repetition_penalty);
    struct SampClearGuard {
        Model* m;
        ~SampClearGuard() { cactus_graph_clear_sampling(); m->samp_ctx_active_ = false; }
    } samp_guard{this};
    if (decode_route_ == DecodeRoute::ENCODER_CROSS_KV_STEP) {
        if (!encoder_cross_kv_ready_ && encoder_cross_kv_source_kind_ == "text_tokens") {
            std::vector<uint32_t> source_tokens = tokens;
            std::vector<uint32_t> decoder_seed = {config_.decoder_start_token_id};
            if (!source_tokens.empty() && source_tokens.back() == config_.decoder_start_token_id) {
                decoder_seed = {source_tokens.back()};
                source_tokens.pop_back();
            }
            if (!prepare_encoder_cross_kv_from_text(source_tokens)) return 0;
            std::vector<uint32_t> emitted = run_encoder_cross_kv_decode_loop(
                decoder_seed,
                1,
                {},
                nullptr);
            if (out_entropy) *out_entropy = 0.0f;
            return emitted.empty() ? 0 : emitted.front();
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (!run_encoder_cross_kv_decoder_step(tokens[i], cache_total_seq_len_ + i)) return 0;
        }
        cache_total_seq_len_ += tokens.size();
        if (out_entropy) *out_entropy = 0.0f;
        uint32_t result = argmax_last_logits();
        record_sampled_token(result);
        return result;
    }
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        cache_token_ids_ = context_tokens_;
        uint32_t result = (greedy && !samp_ctx_active_)
            ? argmax_last_logits(out_entropy)
            : sample_component_logits(*decoder_, temp, tp, tk, min_p, greedy, out_entropy);
        record_sampled_token(result);
        return result;
    }
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_ + i, /*read_logits=*/false);
    }
    run_step(tokens.back(), cache_total_seq_len_ + tokens.size() - 1, /*read_logits=*/true);
    cache_total_seq_len_ += tokens.size();
    cache_token_ids_.insert(cache_token_ids_.end(), tokens.begin(), tokens.end());
    maybe_roll_compact();
    uint32_t result = (greedy && !samp_ctx_active_)
        ? argmax_last_logits(out_entropy)
        : sample_component_logits(*decoder_, temp, tp, tk, min_p, greedy, out_entropy);
    record_sampled_token(result);
    return result;
}

uint32_t Model::decode_with_audio(const std::vector<uint32_t>& tokens,
                                  const std::vector<std::vector<float>>& /*audio_features_per_message*/,
                                  float temperature, float top_p, size_t top_k, const std::string& profile_file,
                                  float* out_entropy, float min_p, float repetition_penalty,
                                  float* /*out_token_time_start*/, float* /*out_token_time_end*/) {
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

std::vector<uint32_t> Model::transcribe_whisper_seq2seq(
    const std::vector<float>& audio_features,
    const std::vector<uint32_t>& decoder_prompt_tokens,
    size_t max_tokens,
    const std::vector<std::vector<uint32_t>>& stop_token_sequences,
    const std::atomic<bool>* should_stop,
    int64_t suppress_token_id) {
    if (decoder_prompt_tokens.empty() || max_tokens == 0) return {};
    if (decode_route_ != DecodeRoute::ENCODER_CROSS_KV_STEP || encoder_cross_kv_source_kind_ != "audio_features") {
        CACTUS_LOG_ERROR("model", "Whisper bundle missing encoder_cross_kv_decoder_step route metadata");
        return {};
    }
    if (!prepare_encoder_cross_kv_from_audio(audio_features)) return {};
    struct SuppressGuard { int64_t& id; ~SuppressGuard() { id = -1; } } guard{suppressed_token_id_};
    suppressed_token_id_ = suppress_token_id;
    return run_encoder_cross_kv_decode_loop(
        decoder_prompt_tokens,
        max_tokens,
        stop_token_sequences,
        should_stop);
}

std::vector<uint32_t> Model::transcribe_parakeet_tdt(const std::vector<float>& audio_features,
                                                     ParakeetTdtStreamState* stream, bool is_final,
                                                     size_t end_frame,
                                                     const std::atomic<bool>* should_stop) {
    std::vector<uint32_t> emitted;
    double raw_decode_ms = 0.0;

    reset_handoff_probe_rollout();

    Component* audio_enc = components_.count("audio_encoder") ? &components_.at("audio_encoder") : nullptr;
    Component* dec = components_.count("decoder") ? &components_.at("decoder") : nullptr;
    if (!audio_enc || !dec) {
        CACTUS_LOG_ERROR("model", "Parakeet TDT bundle missing audio_encoder or decoder component");
        return emitted;
    }
    if (!bind_runtime_buffers(*audio_enc)) return emitted;
    if (!bind_runtime_buffers(*dec)) return emitted;

    int feat_idx = input_index(*audio_enc, "input_features");
    if (feat_idx < 0) {
        CACTUS_LOG_ERROR("model", "audio_encoder has no input_features input");
        return emitted;
    }
    auto& feat_buf = audio_enc->input_buffers[feat_idx];
    size_t feat_node = static_cast<size_t>(audio_enc->runtime_input_node_ids[feat_idx]);
    const auto& feat_desc = audio_enc->graph->get_output_buffer(feat_node);
    if (feat_desc.shape.size() != 3) {
        CACTUS_LOG_ERROR("model", "audio_encoder expects [1, frames, mels] input shape");
        return emitted;
    }
    const size_t expected_frames = feat_desc.shape[1];
    const size_t expected_mels = feat_desc.shape[2];
    const size_t source_frames = expected_mels > 0 ? audio_features.size() / expected_mels : 0;
    const size_t copy_frames = std::min(source_frames, expected_frames);
    std::vector<float> transposed(expected_frames * expected_mels, 0.0f);
    for (size_t t = 0; t < copy_frames; ++t) {
        for (size_t m = 0; m < expected_mels; ++m) {
            transposed[t * expected_mels + m] = audio_features[m * source_frames + t];
        }
    }
    write_typed_buffer(feat_buf, feat_desc.precision, transposed.data(),
                       transposed.size() * sizeof(float), Precision::FP32);

    if (should_stop && should_stop->load()) return emitted;
    audio_enc->graph->execute();
    maybe_capture_handoff_probe_hidden(*audio_enc, "encoder_hidden_states");

    int hidden_idx = output_index(*audio_enc, "encoder_hidden_states");
    if (hidden_idx < 0) {
        CACTUS_LOG_ERROR("model", "audio_encoder has no encoder_hidden_states output");
        return emitted;
    }
    size_t hidden_node = static_cast<size_t>(audio_enc->output_node_ids[hidden_idx]);
    const auto& hidden_desc = audio_enc->graph->get_output_buffer(hidden_node);
    const uint8_t* hidden_ptr = static_cast<const uint8_t*>(audio_enc->graph->get_output(hidden_node));
    if (hidden_desc.shape.size() < 3 || hidden_ptr == nullptr) {
        CACTUS_LOG_ERROR("model", "encoder_hidden_states must be 3D [B, T, D]");
        return emitted;
    }
    const size_t T = hidden_desc.shape[1];
    const size_t D = hidden_desc.shape[2];
    const Precision hidden_precision = hidden_desc.precision;
    const size_t hidden_elem = PrecisionTraits::size_of(hidden_precision);
    const size_t frame_bytes = D * hidden_elem;

    auto zero_state = [&](const std::string& name) {
        int idx = input_index(*dec, name);
        if (idx < 0) return;
        auto& buf = dec->input_buffers[idx];
        std::memset(buf.data(), 0, buf.size());
    };
    if (stream && stream->initialized && stream->dec_state.size() == 4) {
        const char* sn[4] = {"state_h_0", "state_c_0", "state_h_1", "state_c_1"};
        for (size_t i = 0; i < 4; ++i) {
            int idx = input_index(*dec, sn[i]);
            if (idx < 0) continue;
            auto& buf = dec->input_buffers[idx];
            std::memcpy(buf.data(), stream->dec_state[i].data(),
                        std::min(buf.size(), stream->dec_state[i].size()));
        }
    } else {
        zero_state("state_h_0");
        zero_state("state_c_0");
        zero_state("state_h_1");
        zero_state("state_c_1");
    }

    std::vector<uint32_t> durations = config_.tdt_durations;
    if (durations.empty()) {
        for (uint32_t i = 0; i < config_.tdt_num_durations; ++i) durations.push_back(i);
    }
    if (durations.empty()) durations.push_back(1);

    const uint32_t configured_blank = config_.tdt_blank_id;
    uint32_t last_token = (stream && stream->initialized) ? stream->last_token : configured_blank;
    size_t time_index = (stream && stream->initialized) ? stream->time_index : 0;

    const int ef_idx = input_index(*dec, "encoder_frame");
    const int tok_in_idx = input_index(*dec, "token_ids");
    const int logits_idx = output_index(*dec, "step_logits");
    if (ef_idx < 0 || tok_in_idx < 0 || logits_idx < 0) {
        CACTUS_LOG_ERROR("model", "decoder missing encoder_frame / token_ids / step_logits ports");
        return emitted;
    }
    auto& ef_buf = dec->input_buffers[ef_idx];
    const auto& ef_desc = dec->graph->get_output_buffer(static_cast<size_t>(dec->runtime_input_node_ids[ef_idx]));
    auto& tok_buf = dec->input_buffers[tok_in_idx];
    const Precision tok_prec = dec->graph->get_output_buffer(static_cast<size_t>(dec->runtime_input_node_ids[tok_in_idx])).precision;
    void* tok_data = tok_buf.data();
    const size_t logits_node = static_cast<size_t>(dec->output_node_ids[logits_idx]);
    const auto& logits_desc = dec->graph->get_output_buffer(logits_node);
    const Precision logits_prec = logits_desc.precision;
    const size_t total_classes = logits_desc.shape.empty() ? 0 : logits_desc.shape.back();
    const size_t num_durations = durations.size();
    const size_t token_class_count = (total_classes > num_durations) ? (total_classes - num_durations) : total_classes;
    if (token_class_count == 0) return emitted;
    uint32_t effective_blank = configured_blank;
    if (effective_blank >= token_class_count) effective_blank = static_cast<uint32_t>(token_class_count - 1);

    const std::array<const char*, 4> state_names = {"state_h_0", "state_c_0", "state_h_1", "state_c_1"};
    struct StateCopy { void* in_data; const void* out_ptr; size_t bytes; };
    std::array<StateCopy, 4> state_copies{};
    size_t state_copy_count = 0;
    for (const char* state_name : state_names) {
        int out_idx = output_index(*dec, state_name);
        int in_idx = input_index(*dec, state_name);
        if (out_idx < 0 || in_idx < 0) continue;
        size_t out_node = static_cast<size_t>(dec->output_node_ids[out_idx]);
        const auto& out_desc = dec->graph->get_output_buffer(out_node);
        auto& in_buf = dec->input_buffers[in_idx];
        state_copies[state_copy_count++] = {
            in_buf.data(),
            dec->graph->get_output(out_node),
            std::min(out_desc.byte_size, in_buf.size())
        };
    }

    size_t commit_to = T;
    if (stream) {
        size_t valid_hidden = T;
        if (expected_frames > 0)
            valid_hidden = std::min<size_t>(T, (copy_frames * T) / expected_frames);
        commit_to = (end_frame > 0) ? std::min(end_frame, valid_hidden) : valid_hidden;
    }
    Tokenizer* stream_tok = stream ? get_tokenizer() : nullptr;
    const auto& tdt_vocab_bias = get_vocab_bias();
    const float frame_sec = (160.0f / 16000.0f) *
        static_cast<float>(std::max<uint32_t>(1, config_.subsampling_factor));
    constexpr uint32_t kMaxStreamDurationSkipFrames = 2;

    auto snapshot_state = [&]() {
        std::vector<std::vector<uint8_t>> snap(state_copy_count);
        for (size_t s = 0; s < state_copy_count; ++s) {
            const uint8_t* p = static_cast<const uint8_t*>(state_copies[s].in_data);
            snap[s].assign(p, p + state_copies[s].bytes);
        }
        return snap;
    };

    std::vector<std::vector<uint8_t>> snap_state;
    uint32_t snap_last_token = last_token;
    size_t snap_time = time_index;
    size_t confirmed_count = 0;
    if (stream) snap_state = snapshot_state();

    while (time_index < commit_to) {
        if (should_stop && should_stop->load()) break;
        const uint8_t* frame_ptr = hidden_ptr + time_index * frame_bytes;
        write_typed_buffer(ef_buf, ef_desc.precision, frame_ptr, frame_bytes, hidden_precision);

        size_t symbols_added = 0;
        bool advanced = false;
        while (symbols_added < 10) {
            switch (tok_prec) {
                case Precision::FP32: *reinterpret_cast<float*>(tok_data) = static_cast<float>(last_token); break;
                case Precision::FP16: *reinterpret_cast<__fp16*>(tok_data) = static_cast<__fp16>(last_token); break;
                case Precision::INT8: *reinterpret_cast<int8_t*>(tok_data) = static_cast<int8_t>(last_token); break;
                default: *reinterpret_cast<int32_t*>(tok_data) = static_cast<int32_t>(last_token); break;
            }
            const auto exec_t0 = std::chrono::steady_clock::now();
            dec->graph->execute();
            raw_decode_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - exec_t0).count();

            const void* logits_ptr = dec->graph->get_output(logits_node);
            auto get_logit = [&](size_t i) -> float {
                if (logits_prec == Precision::FP32) return reinterpret_cast<const float*>(logits_ptr)[i];
                if (logits_prec == Precision::FP16) return static_cast<float>(reinterpret_cast<const __fp16*>(logits_ptr)[i]);
                return static_cast<float>(reinterpret_cast<const int8_t*>(logits_ptr)[i]);
            };

            uint32_t next_token = 0;
            float best_token_score = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < token_class_count; ++i) {
                float v = get_logit(i);
                if (stream && !tdt_vocab_bias.empty()) {
                    auto it = tdt_vocab_bias.find(static_cast<uint32_t>(i));
                    if (it != tdt_vocab_bias.end()) v += it->second;
                }
                if (v > best_token_score) { best_token_score = v; next_token = static_cast<uint32_t>(i); }
            }
            uint32_t best_duration_idx = 0;
            float best_duration_score = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < num_durations; ++i) {
                float v = get_logit(token_class_count + i);
                if (v > best_duration_score) { best_duration_score = v; best_duration_idx = static_cast<uint32_t>(i); }
            }

            uint32_t skip = durations[std::min<uint32_t>(best_duration_idx, static_cast<uint32_t>(durations.size() - 1))];
            if (stream && skip > kMaxStreamDurationSkipFrames) skip = kMaxStreamDurationSkipFrames;

            if (next_token != effective_blank) {
                if (stream && !is_final && stream_tok && !emitted.empty()) {
                    std::string piece = stream_tok->decode({next_token});
                    if (!piece.empty() && piece[0] == ' ') {
                        snap_state = snapshot_state();
                        snap_last_token = last_token;
                        snap_time = time_index;
                        confirmed_count = emitted.size();
                    }
                }
                emitted.push_back(next_token);
                last_token = next_token;
                for (size_t s = 0; s < state_copy_count; ++s) {
                    std::memcpy(state_copies[s].in_data, state_copies[s].out_ptr, state_copies[s].bytes);
                }
            }

            ++symbols_added;

            if (skip > 0) {
                time_index += skip;
                advanced = true;
                break;
            }
            if (next_token == effective_blank) {
                time_index += 1;
                advanced = true;
                break;
            }
        }

        if (!advanced) time_index += 1;
    }

    if (stream) {
        stream->initialized = true;
        stream->decoded_tokens = emitted.size();
        stream->raw_decode_ms = raw_decode_ms;
        stream->pending.clear();
        if (is_final) {
            stream->dec_state = snapshot_state();
            stream->last_token = last_token;
            stream->time_index = time_index;
            stream->confirmed_sec = static_cast<float>(time_index) * frame_sec;
        } else {
            stream->dec_state = std::move(snap_state);
            stream->last_token = snap_last_token;
            stream->time_index = snap_time;
            stream->confirmed_sec = static_cast<float>(snap_time) * frame_sec;
            stream->pending.assign(emitted.begin() + confirmed_count, emitted.end());
            emitted.resize(confirmed_count);
        }
    }

    return emitted;
}

uint32_t Model::decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& /*image_paths*/,
                                     float temperature, float top_p, size_t top_k, const std::string& profile_file,
                                     float* out_entropy, float min_p, float repetition_penalty) {
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

namespace {

std::vector<float> pool_and_normalize_media_feature(
    const std::vector<uint8_t>& bytes,
    const std::vector<size_t>& shape,
    Precision precision,
    const std::string& source
) {
    const size_t elem_size = PrecisionTraits::size_of(precision);
    if (elem_size == 0 || bytes.empty() || shape.empty()) {
        throw std::runtime_error(source + " produced empty feature output");
    }
    const size_t total_elems = bytes.size() / elem_size;
    const size_t hidden_dim = shape.back();
    if (hidden_dim == 0 || total_elems == 0 || total_elems % hidden_dim != 0) {
        throw std::runtime_error(source + " feature shape inconsistent with hidden_dim");
    }

    std::vector<float> fp32(total_elems);
    switch (precision) {
        case Precision::FP32:
            std::memcpy(fp32.data(), bytes.data(), total_elems * sizeof(float));
            break;
        case Precision::FP16:
            Quantization::fp16_to_fp32(reinterpret_cast<const __fp16*>(bytes.data()), fp32.data(), total_elems);
            break;
        case Precision::INT8:
            Quantization::int8_to_fp32(reinterpret_cast<const int8_t*>(bytes.data()), fp32.data(), total_elems, 1.0f);
            break;
        default:
            throw std::runtime_error(source + " feature precision not supported for embeddings");
    }

    const size_t n_rows = total_elems / hidden_dim;
    std::vector<float> pooled(hidden_dim, 0.0f);
    for (size_t r = 0; r < n_rows; ++r) {
        const float* src = fp32.data() + r * hidden_dim;
        for (size_t d = 0; d < hidden_dim; ++d) pooled[d] += src[d];
    }
    const float inv = 1.0f / static_cast<float>(n_rows);
    for (float& v : pooled) v *= inv;

    float norm_sq = 0.0f;
    for (float v : pooled) norm_sq += v * v;
    if (norm_sq > 1e-12f) {
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (float& v : pooled) v *= inv_norm;
    }
    return pooled;
}

}  // namespace

std::vector<float> Model::get_image_embeddings(const std::string& image_path) {
    if (!vision_encoder_) {
        throw std::runtime_error("Model has no vision_encoder component");
    }
    if (vision_encoder_->logical_outputs.empty()) {
        throw std::runtime_error("vision_encoder has no logical outputs");
    }
    std::string output_name = vision_encoder_->logical_outputs[0];

    run_vision_encoder(image_path);

    if (!media_features_.count(output_name)) {
        for (const char* name : {"image_features", "image_embeddings", "vision_features"}) {
            if (media_features_.count(name)) { output_name = name; break; }
        }
    }
    auto bytes_it = media_features_.find(output_name);
    auto shape_it = media_feature_shapes_.find(output_name);
    auto prec_it = media_feature_precisions_.find(output_name);
    if (bytes_it == media_features_.end() || shape_it == media_feature_shapes_.end()
        || prec_it == media_feature_precisions_.end()) {
        throw std::runtime_error("vision_encoder produced no output for '" + output_name + "'");
    }

    std::vector<float> embedding = pool_and_normalize_media_feature(
        bytes_it->second, shape_it->second, prec_it->second, "vision_encoder");

    for (const std::string& name : vision_encoder_->logical_outputs) {
        media_features_.erase(name);
        media_feature_shapes_.erase(name);
        media_feature_precisions_.erase(name);
    }
    media_features_.erase(output_name);
    media_feature_shapes_.erase(output_name);
    media_feature_precisions_.erase(output_name);
    // run_vision_encoder unloads the graph; restore so subsequent paths that
    // assume the encoder is loaded (e.g. transcribe_*) keep working.
    load_component_graph(*vision_encoder_);
    return embedding;
}

std::vector<float> Model::get_audio_embeddings(const std::vector<float>& mel_bins) {
    if (!audio_encoder_) {
        throw std::runtime_error("Model has no audio_encoder component");
    }
    if (mel_bins.empty()) {
        throw std::runtime_error("Empty audio features");
    }
    if (audio_encoder_->logical_outputs.empty()) {
        throw std::runtime_error("audio_encoder has no logical outputs");
    }
    const std::string output_name = audio_encoder_->logical_outputs[0];

    run_audio_encoder_messages({mel_bins});

    auto bytes_it = media_features_.find(output_name);
    auto shape_it = media_feature_shapes_.find(output_name);
    auto prec_it = media_feature_precisions_.find(output_name);
    if (bytes_it == media_features_.end() || shape_it == media_feature_shapes_.end()
        || prec_it == media_feature_precisions_.end()) {
        throw std::runtime_error("audio_encoder produced no output for '" + output_name + "'");
    }

    std::vector<float> embedding = pool_and_normalize_media_feature(
        bytes_it->second, shape_it->second, prec_it->second, "audio_encoder");

    for (const std::string& name : audio_encoder_->logical_outputs) {
        media_features_.erase(name);
        media_feature_shapes_.erase(name);
        media_feature_precisions_.erase(name);
    }
    load_component_graph(*audio_encoder_);
    return embedding;
}

void Model::reset_cache() {
    cache_total_seq_len_ = 0;
    last_logit_position_ = 0;
    encoder_cross_kv_ready_ = false;
    context_tokens_.clear();
    cache_token_ids_.clear();
    special_rows_.clear();
    token_history_.clear();
    media_features_.clear();
    media_feature_shapes_.clear();
    media_feature_precisions_.clear();
    for (auto& kv : components_) {
        Component& comp = kv.second;
        if (!comp.graph) continue;
        reset_component_cache_states(comp);
    }
}

void Model::set_cache_window(size_t /*window_size*/, size_t /*sink_size*/) {}

void Model::apply_kv_compress_env_override() {
    config_.parse_kv_compress_override(std::getenv("CACTUS_KV_COMPRESS_AT"),
                                       std::getenv("CACTUS_KV_COMPRESS_TO"));
}

std::vector<size_t> Model::compressible_layers() const {
    size_t shared = (config_.num_kv_shared_layers == Config::UNSET_U32)
                        ? 0 : config_.num_kv_shared_layers;
    return cactus::kvcompress::physical_compressible_layers(
        config_.layer_types, config_.num_layers, shared);
}

void Model::compress_kv_cache_keydiff(const cactus::kvcompress::Params& params) {
    if (!decoder_) return;
    using cactus::kvcompress::CacheHeader;
    constexpr size_t kHeaderBytes = sizeof(CacheHeader);

    std::vector<size_t> layers = compressible_layers();
    if (layers.empty()) return;
    std::set<size_t> compressible(layers.begin(), layers.end());
    const double rope_theta = static_cast<double>(config_.rope_theta);
    const double rope_local_theta = (config_.rope_local_base_freq == Config::UNSET_F32)
        ? rope_theta : static_cast<double>(config_.rope_local_base_freq);
    const size_t old_total = cache_total_seq_len_;

    cactus::kvcompress::Params params_local = params;
    bool preserve = config_.kv_compress_preserve_special;
    if (const char* e = std::getenv("CACTUS_KV_PRESERVE_SPECIAL")) preserve = (std::atoi(e) != 0);
    const bool map_valid = cache_token_ids_.size() == old_total && media_features_.empty();
    const bool per_head_protect = preserve && map_valid && special_rows_.valid();
    // cache_token_ids_ past tracked_len is still head-aligned, so specials there apply to every head.
    std::vector<int> appended_special;
    if (per_head_protect) {
        if (special_ids_.empty() && tokenizer_) special_ids_ = tokenizer_->special_token_ids();
        for (size_t r = special_rows_.tracked_len(); r < old_total && r < cache_token_ids_.size(); ++r)
            if (special_ids_.count(cache_token_ids_[r])) appended_special.push_back(static_cast<int>(r));
    }

    Component& comp = *decoder_;
    if (!comp.graph) return;
    // Skip the whole pass (before mutating) if any layer's V dim differs from K (MLA), or a head
    // can't fit sink + all its specials in the budget.
    const size_t protect_budget = params.abs_budget > 0 ? static_cast<size_t>(params.abs_budget) : 0;
    for (size_t li = 0; li < comp.cache_states.size(); ++li) {
        if (!compressible.count(li)) continue;
        const auto& cs = comp.cache_states[li];
        if (cs.key_node_id < 0 || cs.value_node_id < 0) continue;
        if (comp.graph->get_node_op_type(static_cast<size_t>(cs.key_node_id)) != OpType::KV_CACHE_STATE) continue;
        void* kraw = comp.graph->get_output(static_cast<size_t>(cs.key_node_id));
        void* vraw = comp.graph->get_output(static_cast<size_t>(cs.value_node_id));
        if (!kraw || !vraw) continue;
        if (static_cast<CacheHeader*>(vraw)->head_dim != static_cast<CacheHeader*>(kraw)->head_dim) return;
        if (per_head_protect && protect_budget > 0 &&
            special_rows_.max_reserved(li, params.sink, appended_special) > protect_budget) return;
    }
    size_t new_seq_len = 0;
    bool have_new_seq_len = false;
    std::vector<int> canonical_keep;
    bool canonical_captured = false;
    std::vector<cactus::kvcompress::RopeRotation> unrope;
    size_t shrink_cap = 1;
    while (shrink_cap < static_cast<size_t>(config_.kv_compress_trigger_len)) shrink_cap <<= 1;
    for (size_t li = 0; li < comp.cache_states.size(); ++li) {
        if (!compressible.count(li)) continue;
        const auto& cs = comp.cache_states[li];
        if (cs.key_node_id < 0 || cs.value_node_id < 0) continue;
        if (comp.graph->get_node_op_type(static_cast<size_t>(cs.key_node_id)) != OpType::KV_CACHE_STATE) continue;

        const auto& kdesc = comp.graph->get_output_buffer(static_cast<size_t>(cs.key_node_id));
        const auto& vdesc = comp.graph->get_output_buffer(static_cast<size_t>(cs.value_node_id));
        if (kdesc.byte_size <= kHeaderBytes || vdesc.byte_size <= kHeaderBytes) continue;
        void* kraw = comp.graph->get_output(static_cast<size_t>(cs.key_node_id));
        void* vraw = comp.graph->get_output(static_cast<size_t>(cs.value_node_id));
        if (!kraw || !vraw) continue;

        auto* khdr = static_cast<CacheHeader*>(kraw);
        auto* vhdr = static_cast<CacheHeader*>(vraw);
        size_t n = khdr->current_seq_len;
        size_t kv_heads = khdr->num_kv_heads;
        size_t head_dim = khdr->head_dim;
        if (kv_heads == 0 || head_dim == 0) continue;
        if (unrope.empty()) unrope = cactus::kvcompress::unrope_table(n, head_dim, rope_theta);

        static const std::vector<std::vector<int>> kNoProtect;
        if (per_head_protect) special_rows_.add_appended(li, kv_heads, appended_special);
        const std::vector<std::vector<int>>& pph = per_head_protect ? special_rows_.protect(li) : kNoProtect;

        if (kdesc.precision == Precision::FP16) {
            auto* kbase = reinterpret_cast<uint16_t*>(static_cast<char*>(kraw) + kHeaderBytes);
            auto* vbase = reinterpret_cast<uint16_t*>(static_cast<char*>(vraw) + kHeaderBytes);
            auto kept = cactus::kvcompress::keepsets_from_fp16(
                kbase, n, kv_heads, head_dim, unrope, params_local, pph);
            if (!canonical_captured) { canonical_keep = kept.empty() ? std::vector<int>{} : kept[0]; canonical_captured = true; }
            cactus::kvcompress::compact_fp16(kbase, vbase, kv_heads, head_dim, kept, unrope);
            if (per_head_protect) special_rows_.remap(li, kept);
            size_t B = kept.empty() ? 0 : kept[0].size();
            khdr->current_seq_len = B;
            vhdr->current_seq_len = B;
            new_seq_len = B;
            have_new_seq_len = true;
        } else if (kdesc.precision == Precision::INT8) {
            size_t max_seq = khdr->max_seq_len;
            auto* k_i8 = reinterpret_cast<int8_t*>(static_cast<char*>(kraw) + kHeaderBytes);
            auto* k_sc = reinterpret_cast<float*>(static_cast<char*>(kraw) + kHeaderBytes +
                                                  max_seq * kv_heads * head_dim);
            auto* v_i8 = reinterpret_cast<int8_t*>(static_cast<char*>(vraw) + kHeaderBytes);
            auto* v_sc = reinterpret_cast<float*>(static_cast<char*>(vraw) + kHeaderBytes +
                                                  max_seq * kv_heads * head_dim);
            auto kept = cactus::kvcompress::keepsets_from_int8(
                k_i8, k_sc, n, kv_heads, head_dim, KV_QUANT_GROUP_SIZE, unrope, params_local, pph);
            if (!canonical_captured) { canonical_keep = kept.empty() ? std::vector<int>{} : kept[0]; canonical_captured = true; }
            cactus::kvcompress::compact_int8(k_i8, k_sc, kv_heads, head_dim, KV_QUANT_GROUP_SIZE,
                                             kept, unrope, /*renumber=*/true);
            cactus::kvcompress::compact_int8(v_i8, v_sc, kv_heads, head_dim, KV_QUANT_GROUP_SIZE,
                                             kept, unrope, /*renumber=*/false);
            if (per_head_protect) special_rows_.remap(li, kept);
            size_t B = kept.empty() ? 0 : kept[0].size();
            khdr->current_seq_len = B;
            vhdr->current_seq_len = B;
            new_seq_len = B;
            have_new_seq_len = true;
        }
        comp.graph->shrink_cache_buffer(static_cast<size_t>(cs.key_node_id), shrink_cap);
        comp.graph->shrink_cache_buffer(static_cast<size_t>(cs.value_node_id), shrink_cap);
    }

    const size_t Delta = (have_new_seq_len && old_total >= new_seq_len) ? old_total - new_seq_len : 0;
    if (Delta > 0) {
        const double dpos = -static_cast<double>(Delta);
        for (size_t li = 0; li < comp.cache_states.size(); ++li) {
            if (compressible.count(li)) continue;
            const auto& cs = comp.cache_states[li];
            if (cs.key_node_id < 0) continue;
            if (comp.graph->get_node_op_type(static_cast<size_t>(cs.key_node_id)) != OpType::KV_CACHE_STATE) continue;
            const auto& kdesc = comp.graph->get_output_buffer(static_cast<size_t>(cs.key_node_id));
            if (kdesc.byte_size <= kHeaderBytes) continue;
            void* kraw = comp.graph->get_output(static_cast<size_t>(cs.key_node_id));
            if (!kraw) continue;
            auto* khdr = static_cast<CacheHeader*>(kraw);
            size_t kv_heads = khdr->num_kv_heads, head_dim = khdr->head_dim;
            if (kv_heads == 0 || head_dim == 0) continue;
            size_t hi = khdr->current_seq_len;
            size_t lo = std::min<size_t>(khdr->sink_size, hi);
            const double layer_theta = cactus::kvcompress::is_sliding_layer(config_.layer_types, li)
                ? rope_local_theta : rope_theta;
            if (kdesc.precision == Precision::FP16) {
                auto* kbase = reinterpret_cast<uint16_t*>(static_cast<char*>(kraw) + kHeaderBytes);
                cactus::kvcompress::rerope_recent_fp16(kbase, kv_heads, head_dim, lo, hi,
                                                       layer_theta, dpos);
            } else if (kdesc.precision == Precision::INT8) {
                size_t max_seq = khdr->max_seq_len;
                auto* k_i8 = reinterpret_cast<int8_t*>(static_cast<char*>(kraw) + kHeaderBytes);
                auto* k_sc = reinterpret_cast<float*>(static_cast<char*>(kraw) + kHeaderBytes +
                                                      max_seq * kv_heads * head_dim);
                cactus::kvcompress::rerope_recent_int8(k_i8, k_sc, kv_heads, head_dim,
                                                       KV_QUANT_GROUP_SIZE, lo, hi, layer_theta, dpos);
            }
        }
    }

    if (have_new_seq_len) {
        cache_total_seq_len_ = new_seq_len;
        if (per_head_protect) special_rows_.set_tracked_len(new_seq_len);
        else special_rows_.invalidate();
        if (map_valid && canonical_captured) {
            std::vector<uint32_t> compacted;
            compacted.reserve(canonical_keep.size());
            for (int idx : canonical_keep)
                if (idx >= 0 && idx < static_cast<int>(cache_token_ids_.size())) compacted.push_back(cache_token_ids_[idx]);
            cache_token_ids_ = std::move(compacted);
        } else {
            cache_token_ids_.clear();
        }
    }
}

void Model::maybe_roll_compact() {
    if (!config_.kv_compress || config_.kv_compress_trigger_len <= 0) return;
    if (cache_total_seq_len_ < static_cast<size_t>(config_.kv_compress_trigger_len)) return;

    cactus::kvcompress::Params p;
    p.recent_frac = config_.kv_compress_recent_frac;
    p.sink = config_.kv_compress_sink;
    p.abs_budget = config_.kv_compress_target_len;
    compress_kv_cache_keydiff(p);
}

static void l2_normalize_inplace(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    float inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1e-12));
    for (float& x : v) x *= inv;
}

static std::vector<float> finalize_pooled_embedding(const std::vector<double>& sum, size_t count, bool normalize) {
    std::vector<float> out(sum.size(), 0.0f);
    if (count > 0) {
        for (size_t h = 0; h < sum.size(); ++h) out[h] = static_cast<float>(sum[h] / static_cast<double>(count));
    }
    if (normalize) l2_normalize_inplace(out);
    return out;
}

std::vector<float> Model::get_text_embeddings(const std::vector<uint32_t>& tokens, bool normalize) {
    if (has_text_embedding()) {
        return get_embeddings(tokens, /*pooled=*/true, normalize);
    }
    if (has_lm_embedding()) {
        return get_lm_embeddings(tokens, normalize);
    }
    throw std::runtime_error("get_text_embeddings: bundle has neither a text_embedding nor a decoder_embed_chunk component");
}

std::vector<float> Model::get_lm_embeddings(const std::vector<uint32_t>& tokens, bool normalize) {
    if (!decoder_embed_) {
        throw std::runtime_error("get_lm_embeddings: bundle has no decoder_embed_chunk component");
    }
    if (tokens.empty()) {
        throw std::runtime_error("get_lm_embeddings: empty token sequence");
    }
    if (decode_route_ != DecodeRoute::CACHED_STEP || !encoder_) {
        throw std::runtime_error("get_lm_embeddings: model does not support chunked LM embeddings");
    }
    if (!load_component_graph(*decoder_embed_)) {
        throw std::runtime_error("get_lm_embeddings: failed to load decoder_embed_chunk graph");
    }
    if (prefill_encoder_ && !load_component_graph(*prefill_encoder_)) {
        throw std::runtime_error("get_lm_embeddings: failed to load prefill encoder graph");
    }
    reset_component_cache_states(*decoder_embed_);

    const size_t component_tokens = component_chunk_tokens(*decoder_embed_, "inputs_embeds");
    if (component_tokens <= 1) {
        throw std::runtime_error("get_lm_embeddings: decoder_embed_chunk is not chunk-shaped");
    }
    const size_t effective_chunk = component_tokens;

    bool recurrent_state = false;
    if (decoder_embed_->graph) {
        for (const auto& state : decoder_embed_->cache_states) {
            for (int node_id : {state.key_node_id, state.value_node_id}) {
                if (node_id < 0) continue;
                if (decoder_embed_->graph->get_node_op_type(static_cast<size_t>(node_id)) == OpType::RECURRENT_CACHE_STATE) {
                    recurrent_state = true;
                }
            }
        }
    }
    const size_t embed_limit = recurrent_state ? std::min(tokens.size(), effective_chunk) : tokens.size();

    size_t encoder_chunk = 0;
    if (prefill_encoder_ && input_index(*prefill_encoder_, "input_ids") >= 0 && input_index(*prefill_encoder_, "position_ids") >= 0) {
        encoder_chunk = component_chunk_tokens(*prefill_encoder_, "input_ids");
        if (encoder_chunk == 0 || effective_chunk % encoder_chunk != 0) encoder_chunk = 0;
    }

    const int out_idx = output_index(*decoder_embed_, "last_hidden_state");
    if (out_idx < 0 || static_cast<size_t>(out_idx) >= decoder_embed_->output_node_ids.size()) {
        throw std::runtime_error("get_lm_embeddings: decoder_embed_chunk missing last_hidden_state output");
    }
    const size_t out_node = static_cast<size_t>(decoder_embed_->output_node_ids[out_idx]);

    std::vector<double> sum;
    size_t count = 0;

    size_t processed = 0;
    while (processed < embed_limit) {
        for (auto& buf : decoder_embed_->input_buffers) std::fill(buf.begin(), buf.end(), 0);
        if (encoder_chunk > 0) {
            for (size_t chunk_offset = 0; chunk_offset < effective_chunk; chunk_offset += encoder_chunk) {
                for (auto& buf : prefill_encoder_->input_buffers) std::fill(buf.begin(), buf.end(), 0);
                for (size_t i = 0; i < encoder_chunk; ++i) {
                    const size_t index = processed + chunk_offset + i;
                    const uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
                    write_int_input_at(*prefill_encoder_, "input_ids", i, static_cast<int64_t>(token));
                    write_int_input_at(*prefill_encoder_, "position_ids", i, static_cast<int64_t>(processed + chunk_offset + i));
                }
                prefill_encoder_->graph->execute();
                copy_component_outputs_to_chunk_inputs_range(*prefill_encoder_, *decoder_embed_, chunk_offset);
            }
        } else {
            for (size_t i = 0; i < effective_chunk; ++i) {
                const size_t index = processed + i;
                const uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
                run_encoder_step(token, processed + i);
                copy_component_outputs_to_chunk_inputs(*encoder_, *decoder_embed_, i);
            }
        }
        decoder_embed_->graph->execute();

        const auto& desc = decoder_embed_->graph->get_output_buffer(out_node);
        void* ptr = decoder_embed_->graph->get_output(out_node);
        const size_t hidden = desc.shape.empty() ? 0 : desc.shape.back();
        const size_t seq = (desc.shape.size() >= 2) ? desc.shape[desc.shape.size() - 2] : 1;
        if (hidden == 0 || !ptr) {
            throw std::runtime_error("get_lm_embeddings: decoder_embed_chunk produced no last_hidden_state");
        }
        if (sum.empty()) sum.assign(hidden, 0.0);
        const bool is_fp16 = desc.precision == Precision::FP16;
        auto read_at = [&](size_t i) -> float {
            return is_fp16 ? static_cast<float>(reinterpret_cast<const __fp16*>(ptr)[i])
                           : reinterpret_cast<const float*>(ptr)[i];
        };
        size_t real_rows = std::min(effective_chunk, tokens.size() - processed);
        real_rows = std::min(real_rows, seq);
        for (size_t t = 0; t < real_rows; ++t) {
            for (size_t h = 0; h < hidden; ++h) sum[h] += static_cast<double>(read_at(t * hidden + h));
        }
        count += real_rows;
        processed += effective_chunk;
    }

    reset_component_cache_states(*decoder_embed_);
    decoder_embed_->graph->release_runtime_buffers();
    decoder_embed_->graph->release_all_weight_pages();
    unload_component_graph(*decoder_embed_);
    return finalize_pooled_embedding(sum, count, normalize);
}

std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& tokens, bool pooled,
                                          bool normalize, const std::string& /*profile_file*/) {
    if (!components_.count("text_embedding")) {
        throw std::runtime_error("get_embeddings: bundle has no text_embedding component");
    }
    if (tokens.empty()) {
        throw std::runtime_error("get_embeddings: empty token sequence");
    }
    Component* comp = &components_.at("text_embedding");
    if (!load_component_graph(*comp)) {
        throw std::runtime_error("get_embeddings: failed to load embedding component graph");
    }

    std::vector<uint32_t> wrapped;
    wrapped.reserve(tokens.size() + 2);
    if (tokenizer_) wrapped.push_back(tokenizer_->get_bos_token());
    wrapped.insert(wrapped.end(), tokens.begin(), tokens.end());
    if (tokenizer_) wrapped.push_back(tokenizer_->get_eos_token());

    int ids_idx = input_index(*comp, "input_ids");
    if (ids_idx < 0) {
        throw std::runtime_error("get_embeddings: embedding component missing input_ids");
    }
    auto& ids_buf = comp->input_buffers[ids_idx];
    size_t ids_node = static_cast<size_t>(comp->runtime_input_node_ids[ids_idx]);
    const auto& ids_desc = comp->graph->get_output_buffer(ids_node);
    size_t capacity = PrecisionTraits::size_of(ids_desc.precision)
                        ? ids_buf.size() / PrecisionTraits::size_of(ids_desc.precision) : wrapped.size();
    size_t n_real = std::min(capacity, wrapped.size());
    write_tokens_buffer(ids_buf, ids_desc.precision, wrapped, 0);

    int mask_idx = input_index(*comp, "attention_mask");
    if (mask_idx >= 0) {
        auto& mb = comp->input_buffers[mask_idx];
        size_t mnode = static_cast<size_t>(comp->runtime_input_node_ids[mask_idx]);
        const auto& mdesc = comp->graph->get_output_buffer(mnode);
        fill_int_buffer(mb, mdesc.precision, 1, n_real);
    }

    comp->graph->execute();

    if (comp->output_node_ids.empty()) {
        throw std::runtime_error("get_embeddings: embedding component produced no outputs");
    }
    size_t out_node = static_cast<size_t>(comp->output_node_ids[0]);
    const auto& desc = comp->graph->get_output_buffer(out_node);
    void* ptr = comp->graph->get_output(out_node);
    size_t hidden = desc.shape.empty() ? 0 : desc.shape.back();
    size_t seq = (desc.shape.size() >= 2) ? desc.shape[desc.shape.size() - 2] : 1;
    if (hidden == 0) {
        throw std::runtime_error("get_embeddings: embedding output has zero hidden dim");
    }

    const bool is_fp16 = desc.precision == Precision::FP16;
    auto read_at = [&](size_t i) -> float {
        return is_fp16 ? static_cast<float>(reinterpret_cast<const __fp16*>(ptr)[i])
                       : reinterpret_cast<const float*>(ptr)[i];
    };

    std::vector<float> result(hidden, 0.0f);
    if (pooled) {
        size_t pool_rows = std::min(seq, std::max<size_t>(1, n_real));
        for (size_t t = 0; t < pool_rows; ++t) {
            for (size_t h = 0; h < hidden; ++h) result[h] += read_at(t * hidden + h);
        }
        for (size_t h = 0; h < hidden; ++h) result[h] /= static_cast<float>(pool_rows);
    } else {
        for (size_t h = 0; h < hidden; ++h) result[h] = read_at(h);
    }

    if (normalize) l2_normalize_inplace(result);

    comp->graph->release_runtime_buffers();
    comp->graph->release_all_weight_pages();
    unload_component_graph(*comp);
    return result;
}

bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        CACTUS_LOG_ERROR("config", "Failed to open config file: " << config_path);
        return false;
    }
    
    std::string line;
    bool decoder_start_token_seen = false;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        if (key == "vocab_size") vocab_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "bos_token_id") bos_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "eos_token_id") eos_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "decoder_start_token_id") {
            decoder_start_token_id = static_cast<uint32_t>(std::stoul(value));
            decoder_start_token_seen = true;
        }
        else if (key == "decoder_prompt_token_ids") decoder_prompt_token_ids = parse_config_uint_list(value);
        else if (key == "num_layers") num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "hidden_dim") hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "ffn_intermediate_dim") ffn_intermediate_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_heads") attention_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_kv_heads") attention_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_head_dim") attention_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "layer_norm_eps") layer_norm_eps = std::stof(value);
        else if (key == "rope_theta") rope_theta = std::stof(value);
        else if (key == "num_experts") num_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_shared_experts") num_shared_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_top_experts") num_top_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "moe_every_n_layers") moe_every_n_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "moe_intermediate_dim" || key == "moe_intermediate_size") moe_intermediate_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_dense_layers") num_dense_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_experts_per_tok") num_experts_per_tok = static_cast<uint32_t>(std::stoul(value));
        else if (key == "norm_topk_prob") norm_topk_prob = (value == "true" || value == "1");
        else if (key == "use_expert_bias") use_expert_bias = (value == "true" || value == "1");
        else if (key == "routed_scaling_factor") routed_scaling_factor = std::stof(value);
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "vision_hidden_dim" || key == "vision_hidden_size") vision_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_num_layers") vision_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_attention_heads") vision_attention_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_image_size") vision_image_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_patch_size") vision_patch_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_num_channels") vision_num_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_embed_dim") vision_embed_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "visual_tokens_per_img") visual_tokens_per_img = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_pixel_shuffle") use_pixel_shuffle = (value == "true" || value == "1");
        else if (key == "pixel_shuffle_factor") pixel_shuffle_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_image_tokens") use_image_tokens = (value == "true" || value == "1");
        else if (key == "image_token_id") image_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_layout_tags") use_layout_tags = (value == "true" || value == "1");
        else if (key == "image_seq_len") image_seq_len = static_cast<uint32_t>(std::stoul(value));
        else if (key == "global_image_size") global_image_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_tile_size") max_tile_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rescale_factor") rescale_factor = std::stof(value);
        else if (key == "image_mean") image_mean = std::stof(value);
        else if (key == "image_std") image_std = std::stof(value);
        else if (key == "downsample_factor") downsample_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "min_tiles") min_tiles = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_tiles") max_tiles = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_thumbnail") use_thumbnail = (value == "true" || value == "1");
        else if (key == "min_image_tokens") min_image_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_image_tokens") max_image_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tile_size") tile_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_pixels_tolerance") max_pixels_tolerance = std::stof(value);
        else if (key == "do_image_splitting") do_image_splitting = (value == "true" || value == "1");
        else if (key == "precision") {
            if (value == "INT8") precision = Precision::INT8;
            else if (value == "FP16") precision = Precision::FP16;
            else precision = Precision::FP32;
        }
        else if (key == "model_type") {
            std::string mt = value;
            std::transform(mt.begin(), mt.end(), mt.begin(), ::tolower);
            if (mt == "qwen") model_type = ModelType::QWEN;
            else if (mt == "qwen3p5" || mt == "qwen3_5") model_type = ModelType::QWEN3P5;
            else if (mt == "gemma" || mt == "gemma3") model_type = ModelType::GEMMA;
            else if (mt == "gemma3n") model_type = ModelType::GEMMA3N;
            else if (mt == "lfm2") model_type = ModelType::LFM2;
            else if (mt == "whisper") model_type = ModelType::WHISPER;
            else if (mt == "parakeet_tdt" || mt == "parakeet-tdt") model_type = ModelType::PARAKEET_TDT;
            else if (mt == "youtu") model_type = ModelType::YOUTU;
            else if (mt == "needle") model_type = ModelType::NEEDLE;
            else if (mt == "bert" || mt == "nomic") model_type = ModelType::NOMIC;
            else model_type = ModelType::GEMMA4;
        }
        else if (key == "model_variant") {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            if (v == "vlm") model_variant = ModelVariant::VLM;
            else if (v == "extract") model_variant = ModelVariant::EXTRACT;
            else if (v == "rag") model_variant = ModelVariant::RAG;
            else model_variant = ModelVariant::DEFAULT;
        }
        else if (key == "conv_L_cache") conv_L_cache = static_cast<size_t>(std::stoul(value));
        else if (key == "kv_compress") kv_compress = (value == "true" || value == "1");
        else if (key == "kv_compress_recent_frac") kv_compress_recent_frac = std::stof(value);
        else if (key == "kv_compress_sink") kv_compress_sink = static_cast<uint32_t>(std::stoul(value));
        else if (key == "kv_compress_trigger_len") kv_compress_trigger_len = static_cast<int32_t>(std::stol(value));
        else if (key == "kv_compress_target_len") kv_compress_target_len = static_cast<int32_t>(std::stol(value));
        else if (key == "kv_compress_preserve_special") kv_compress_preserve_special = (value == "true" || value == "1");
        else if (key == "layer_types") {
            layer_types.clear();
            std::string sanitized;
            sanitized.reserve(value.size());
            for (char c : value) {
                if (c == '[' || c == ']' || c == '\'' || c == '"') {
                    continue;
                }
                sanitized.push_back(c);
            }
            std::stringstream ss(sanitized);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty()) layer_types.push_back(item);
                }
            }
        }
        else if (key == "enc_hidden_act") encoder_act_gelu = (value == "gelu");
        else if (key == "dec_hidden_act") decoder_act_gelu = (value == "gelu");
        else if (key == "num_encoder_layers") num_encoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_decoder_layers") num_decoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "partial_rotary_factor") partial_rotary_factor = std::stof(value);
        else if (key == "pad_token_id") pad_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "conv_kernel_size") conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_kernel_size") subsampling_conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_stride") subsampling_conv_stride = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_channels") subsampling_conv_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_factor") subsampling_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_mel_bins") num_mel_bins = static_cast<uint32_t>(std::stoul(value));
        else if (key == "encoder_hidden_act") encoder_hidden_act = value;
        else if (key == "linear_num_key_heads") linear_num_key_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_key_head_dim") linear_key_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_num_value_heads") linear_num_value_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_value_head_dim") linear_value_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_q_proj_dim") linear_q_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "kv_lora_rank") kv_lora_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "q_lora_rank") q_lora_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_head_dim") qk_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_nope_head_dim") qk_nope_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_rope_head_dim") qk_rope_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "v_head_dim") v_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rope_interleave") rope_interleave = (value == "true" || value == "1");
        else if (key == "attention_bias") attention_bias = (value == "true" || value == "1");
        else if (key == "rope_scaling_factor") rope_scaling_factor = std::stof(value);
        else if (key == "rope_mscale_all_dim") rope_mscale_all_dim = std::stof(value);
        else if (key == "linear_k_proj_dim") linear_k_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_v_proj_dim") linear_v_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "predictor_hidden_dim") predictor_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "predictor_num_layers") predictor_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_joint_dim") tdt_joint_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_num_durations") tdt_num_durations = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_blank_id") tdt_blank_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_durations") {
            tdt_durations.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t first = item.find_first_not_of(" \t");
                if (first == std::string::npos) continue;
                size_t last = item.find_last_not_of(" \t");
                item = item.substr(first, last - first + 1);
                tdt_durations.push_back(static_cast<uint32_t>(std::stoul(item)));
            }
        }
        else if (key == "altup_num_inputs") altup_num_inputs = static_cast<uint32_t>(std::stoul(value));
        else if (key == "laurel_rank") laurel_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "hidden_size_per_layer_input") hidden_size_per_layer_input = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_kv_shared_layers") num_kv_shared_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "sliding_window") sliding_window = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rope_local_base_freq") rope_local_base_freq = std::stof(value);
        else if (key == "final_logit_softcapping") final_logit_softcapping = std::stof(value);
        else if (key == "global_partial_rotary_factor") global_partial_rotary_factor = std::stof(value);
        else if (key == "expert_intermediate_size") expert_intermediate_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "global_head_dim") global_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_global_kv_heads" || key == "num_global_key_value_heads") num_global_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_k_eq_v") attention_k_eq_v = (value == "true" || value == "1");
        else if (key == "enable_moe_block") enable_moe_block = (value == "true" || value == "1");
        else if (key == "vision_head_dim") vision_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_kv_heads") vision_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_intermediate_size") vision_intermediate_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_position_embedding_size") vision_position_embedding_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_pooling_kernel_size") vision_pooling_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_default_output_length") vision_default_output_length = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_rope_theta") vision_rope_theta = std::stof(value);
        else if (key == "audio_hidden_dim") audio_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_num_layers") audio_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_num_heads") audio_num_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_head_dim") audio_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_input_feat_size") audio_input_feat_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_conf_conv_kernel_size") audio_conf_conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_chunk_size") audio_chunk_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_context_left") audio_context_left = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_context_right") audio_context_right = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_logit_cap") audio_logit_cap = std::stof(value);
        else if (key == "audio_residual_weight") audio_residual_weight = std::stof(value);
        else if (key == "audio_output_proj_dims") audio_output_proj_dims = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_vocab_size") audio_vocab_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_vocab_offset") audio_vocab_offset = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_soft_tokens") audio_soft_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv0_channels") audio_sscp_conv0_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv1_channels") audio_sscp_conv1_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv_eps") audio_sscp_conv_eps = std::stof(value);
        else if (key == "audio_rms_norm_eps") audio_rms_norm_eps = std::stof(value);
        else if (key == "audio_fft_length") audio_fft_length = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_fft_overdrive") {
            audio_fft_overdrive = (value == "true" || value == "1");
            audio_fft_length = audio_fft_overdrive ? 1024u : 512u;
        }
        else if (key == "audio_token_id") audio_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "activation_sparsity_ppf") {
            activation_sparsity_ppf.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t first = item.find_first_not_of(" \t");
                if (first == std::string::npos) continue;
                size_t last = item.find_last_not_of(" \t");
                item = item.substr(first, last - first + 1);
                activation_sparsity_ppf.push_back(std::stof(item));
            }
        }
    }

    if (is_gemma_family(model_type)) {
        default_temperature = 1.0f;
        default_top_p = 0.95f;
        default_top_k = 64;
        if (model_type == ModelType::GEMMA4) {
            default_cloud_handoff_threshold = 0.81f;
        }
    } else if (model_type == ModelType::LFM2) {
        default_temperature = 0.3f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN) {
        default_temperature = 0.6f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN3P5) {
        default_temperature = 0.7f;
        default_top_p = 0.8f;
        default_top_k = 20;
    }

    if (model_type == ModelType::GEMMA4) {
        auto missing_u32 = [](uint32_t v) { return v == UNSET_U32; };
        auto missing_f32 = [](float v) { return v == UNSET_F32; };
        std::string missing;
        if (missing_u32(hidden_size_per_layer_input)) missing += " hidden_size_per_layer_input";
        if (missing_u32(num_kv_shared_layers)) missing += " num_kv_shared_layers";
        if (missing_u32(sliding_window)) missing += " sliding_window";
        if (missing_u32(global_head_dim)) missing += " global_head_dim";
        if (missing_f32(rope_local_base_freq)) missing += " rope_local_base_freq";
        if (missing_f32(final_logit_softcapping)) missing += " final_logit_softcapping";
        if (missing_f32(global_partial_rotary_factor)) missing += " global_partial_rotary_factor";
        if (layer_types.empty()) missing += " layer_types";
        if (!missing.empty()) {
            CACTUS_LOG_ERROR("config", "Gemma4 config missing required fields:" << missing);
            return false;
        }
    }

    if (!decoder_start_token_seen) {
        decoder_start_token_id = bos_token_id;
    }

    validate_kv_compress();
    return true;
}

bool Config::parse_kv_compress_override(const char* trigger_env, const char* target_env) {
    const bool has_trigger = trigger_env && *trigger_env;
    const bool has_target = target_env && *target_env;
    if (!has_trigger && !has_target) return false;
    if (has_trigger) kv_compress_trigger_len = static_cast<int32_t>(std::stol(trigger_env));
    if (has_target) kv_compress_target_len = static_cast<int32_t>(std::stol(target_env));
    if (kv_compress_trigger_len <= 0) {
        kv_compress = false;
        kv_compress_trigger_len = 0;
        kv_compress_target_len = 0;
        CACTUS_LOG_INFO("kv_compress", "rolling compaction disabled (CACTUS_KV_COMPRESS_AT <= 0)");
        return true;
    }
    kv_compress = true;
    validate_kv_compress();
    CACTUS_LOG_INFO("kv_compress", "rolling override: trigger_len=" << kv_compress_trigger_len
        << " target_len=" << kv_compress_target_len);
    return true;
}

void Config::validate_kv_compress() {
    if (kv_compress_trigger_len <= 0) return;
    if (kv_compress_target_len <= 0 || kv_compress_target_len >= kv_compress_trigger_len) {
        CACTUS_LOG_WARN("kv_compress", "invalid rolling config (target_len=" << kv_compress_target_len
            << ", trigger_len=" << kv_compress_trigger_len
            << "): require 0 < target_len < trigger_len; disabling rolling compaction");
        kv_compress = false;
        kv_compress_trigger_len = 0;
        kv_compress_target_len = 0;
    }
}

std::string Config::to_json() const {
    return "{}";
}

std::unique_ptr<Model> create_model(const std::string& bundle_dir) {
    CACTUS_LOG_DEBUG("model", "Creating model from: " << bundle_dir);
    fs::path manifest = fs::path(bundle_dir) / "components" / "manifest.json";
    if (!fs::exists(manifest)) {
        CACTUS_LOG_ERROR("model",
            "Not a transpiled bundle (no components/manifest.json at " << bundle_dir << "). "
            "Run `cactus convert <hf_model>` to produce one.");
        return nullptr;
    }
    return std::make_unique<Model>();
}

const std::vector<Model::DebugNode>& Model::get_debug_nodes() const {
    debug_nodes_.clear();
    return debug_nodes_;
}


double Model::score_tokens_window_logprob(const std::vector<uint32_t>& /*tokens*/, size_t /*start*/,
                                            size_t /*end*/, size_t /*context*/, size_t* tokens_scored) {
    if (tokens_scored) *tokens_scored = 0;
    return 0.0;
}

}
}
