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
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace cactus {
namespace engine {

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

bool Model::init(const std::string& bundle_dir, size_t context_size,
                 const std::string& /*system_prompt*/, bool /*do_warmup*/) {
    if (initialized_) return true;
    bundle_dir_ = bundle_dir;

    if (!config_.from_json(bundle_dir + "/config.txt")) {
        CACTUS_LOG_ERROR("model", "Failed to load config.txt from: " << bundle_dir);
        return false;
    }
    if (!load_manifest()) {
        CACTUS_LOG_ERROR("model", "Failed to load bundle manifest from: " << bundle_dir);
        return false;
    }
    if (!setup_tokenizer()) {
        CACTUS_LOG_ERROR("model", "Tokenizer init failed for bundle: " << bundle_dir);
        return false;
    }
    if (!load_components()) return false;

    encoder_ = components_.count("lm_encoder_step") ? &components_.at("lm_encoder_step") : nullptr;
    decoder_ = components_.count("decoder_step") ? &components_.at("decoder_step") : nullptr;
    const bool is_lm = (encoder_ != nullptr && decoder_ != nullptr);
    const bool is_transcription = components_.count("audio_encoder") &&
                                  (components_.count("decoder") || components_.count("decoder_joint"));
    if (!is_lm && !is_transcription) {
        CACTUS_LOG_ERROR("model", "Bundle missing required components: need lm_encoder_step+decoder_step (LM) or audio_encoder+decoder (transcription)");
        return false;
    }
    if (encoder_ && !bind_runtime_buffers(*encoder_)) return false;
    if (decoder_ && !bind_runtime_buffers(*decoder_)) return false;

    vision_encoder_ = components_.count("vision_encoder") ? &components_.at("vision_encoder") : nullptr;
    audio_encoder_ = components_.count("audio_encoder") ? &components_.at("audio_encoder") : nullptr;
    lm_encoder_media_step_ = components_.count("lm_encoder_media_step") ? &components_.at("lm_encoder_media_step") : nullptr;
    decoder_prefill_chunk_ = components_.count("decoder_prefill_chunk") ? &components_.at("decoder_prefill_chunk") : nullptr;
    lm_encoder_ = components_.count("lm_encoder") ? &components_.at("lm_encoder") : nullptr;
    if (vision_encoder_ && !bind_runtime_buffers(*vision_encoder_)) return false;
    if (audio_encoder_ && !bind_runtime_buffers(*audio_encoder_)) return false;
    if (lm_encoder_media_step_ && !bind_runtime_buffers(*lm_encoder_media_step_)) return false;
    if (decoder_prefill_chunk_ && !bind_runtime_buffers(*decoder_prefill_chunk_)) return false;
    if (lm_encoder_ && !bind_runtime_buffers(*lm_encoder_)) return false;

    if (vision_encoder_ && tokenizer_ && !vision_encoder_->output_node_ids.empty()) {
        size_t out_node = static_cast<size_t>(vision_encoder_->output_node_ids[0]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(out_node);
        size_t n = 0;
        if (desc.shape.size() >= 3) n = desc.shape[desc.shape.size() - 2];
        else if (desc.shape.size() >= 2) n = desc.shape[0];
        if (n > 0) tokenizer_->set_image_soft_token_count(n);
    }

    cache_max_seq_len_ = context_size;
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
            for (const auto& ev : c.at("cache_state_node_ids").get<picojson::array>()) {
                if (!ev.is<picojson::object>()) continue;
                const auto& e = ev.get<picojson::object>();
                CacheStateEntry cs;
                if (e.count("layer_key")) cs.layer_key = e.at("layer_key").get<std::string>();
                if (e.count("key") && e.at("key").is<int64_t>())
                    cs.key_node_id = static_cast<int>(e.at("key").get<int64_t>());
                if (e.count("value") && e.at("value").is<int64_t>())
                    cs.value_node_id = static_cast<int>(e.at("value").get<int64_t>());
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

bool Model::load_components() {
    for (auto& [name, comp] : components_) {
        if (comp.graph_path.empty()) continue;
        fs::path full = fs::path(bundle_dir_) / comp.graph_path;
        try {
            comp.graph = std::make_unique<CactusGraph>(CactusGraph::load(full.string()));
        } catch (const std::exception& e) {
            CACTUS_LOG_ERROR("model", "load " << comp.graph_path << ": " << e.what());
            return false;
        }
        for (const auto& b : comp.bindings) {
            if (b.node_id < 0) continue;
            try {
                comp.graph->bind_mmap_weights(static_cast<size_t>(b.node_id),
                                              (fs::path(bundle_dir_) / b.path).string());
            } catch (const std::exception& e) {
                CACTUS_LOG_ERROR("model", "bind " << b.path << ": " << e.what());
                return false;
            }
        }
    }
    return true;
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
    int idx = input_index(comp, name);
    if (idx < 0) return;
    size_t node_id = static_cast<size_t>(comp.runtime_input_node_ids[idx]);
    const auto& desc = comp.graph->get_output_buffer(node_id);
    auto& buf = comp.input_buffers[idx];
    switch (desc.precision) {
        case Precision::FP32:
            *reinterpret_cast<float*>(buf.data()) = static_cast<float>(value);
            break;
        case Precision::FP16:
            *reinterpret_cast<__fp16*>(buf.data()) = static_cast<__fp16>(value);
            break;
        case Precision::INT8:
            *reinterpret_cast<int8_t*>(buf.data()) = static_cast<int8_t>(value);
            break;
        default:
            *reinterpret_cast<int32_t*>(buf.data()) = static_cast<int32_t>(value);
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

void Model::run_step(uint32_t token_id, size_t position, bool /*read_logits*/) {
    write_int_input(*encoder_, "input_ids", static_cast<int64_t>(token_id));
    write_int_input(*encoder_, "position_ids", static_cast<int64_t>(position));
    encoder_->graph->execute();
    copy_encoder_outputs_to_decoder(*encoder_);
    decoder_->graph->execute();
}

void Model::run_media_step(size_t position, const uint8_t* feature_row, size_t feature_row_bytes,
                           Precision feature_precision) {
    if (!lm_encoder_media_step_) {
        run_step(static_cast<uint32_t>(config_.pad_token_id), position, false);
        return;
    }
    int embeds_idx = input_index(*lm_encoder_media_step_, "inputs_embeds");
    if (embeds_idx < 0) {
        run_step(static_cast<uint32_t>(config_.pad_token_id), position, false);
        return;
    }
    auto& embeds_buf = lm_encoder_media_step_->input_buffers[embeds_idx];
    size_t node_id = static_cast<size_t>(lm_encoder_media_step_->runtime_input_node_ids[embeds_idx]);
    const auto& desc = lm_encoder_media_step_->graph->get_output_buffer(node_id);
    if (desc.precision == feature_precision) {
        size_t to_copy = std::min(feature_row_bytes, embeds_buf.size());
        std::memcpy(embeds_buf.data(), feature_row, to_copy);
        if (to_copy < embeds_buf.size()) {
            std::memset(embeds_buf.data() + to_copy, 0, embeds_buf.size() - to_copy);
        }
    } else {
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
    write_int_input(*lm_encoder_media_step_, "input_ids", 0);
    write_int_input(*lm_encoder_media_step_, "position_ids", static_cast<int64_t>(position));
    lm_encoder_media_step_->graph->execute();
    copy_encoder_outputs_to_decoder(*lm_encoder_media_step_);
    decoder_->graph->execute();
}

void Model::run_vision_encoder(const std::string& image_path) {
    if (!vision_encoder_) return;
    Gemma4ImagePreprocessed prep = preprocess_gemma4_image(image_path, config_);
    write_bytes_input(*vision_encoder_, "pixel_values", prep.pixel_values.data(),
                      prep.pixel_values.size() * sizeof(float));
    write_bytes_input(*vision_encoder_, "pixel_position_ids", prep.pixel_position_ids.data(),
                      prep.pixel_position_ids.size() * sizeof(int64_t));
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
}

void Model::run_audio_encoder(const std::vector<float>& audio_features) {
    if (!audio_encoder_) return;
    std::vector<std::string> candidate_input_names = {"input_features", "audio_features"};
    bool wrote = false;
    for (const auto& name : candidate_input_names) {
        int idx = input_index(*audio_encoder_, name);
        if (idx < 0) continue;
        size_t node_id = static_cast<size_t>(audio_encoder_->runtime_input_node_ids[idx]);
        const auto& desc = audio_encoder_->graph->get_output_buffer(node_id);
        auto& buf = audio_encoder_->input_buffers[idx];
        if (desc.precision == Precision::FP32) {
            size_t n = std::min(audio_features.size() * sizeof(float), buf.size());
            std::memcpy(buf.data(), audio_features.data(), n);
            if (n < buf.size()) std::memset(buf.data() + n, 0, buf.size() - n);
        } else if (desc.precision == Precision::FP16) {
            size_t n_elems = std::min(audio_features.size(), buf.size() / sizeof(__fp16));
            __fp16* dst = reinterpret_cast<__fp16*>(buf.data());
            for (size_t i = 0; i < n_elems; ++i) dst[i] = static_cast<__fp16>(audio_features[i]);
            if (n_elems * sizeof(__fp16) < buf.size()) {
                std::memset(buf.data() + n_elems * sizeof(__fp16), 0, buf.size() - n_elems * sizeof(__fp16));
            }
        } else {
            size_t n_elems = std::min(audio_features.size(), buf.size());
            int8_t* dst = reinterpret_cast<int8_t*>(buf.data());
            for (size_t i = 0; i < n_elems; ++i) dst[i] = static_cast<int8_t>(audio_features[i]);
            if (n_elems < buf.size()) std::memset(buf.data() + n_elems, 0, buf.size() - n_elems);
        }
        wrote = true;
        break;
    }
    if (!wrote) {
        CACTUS_LOG_WARN("model", "audio_encoder has no input named input_features/audio_features; skipping");
        return;
    }
    int mask_idx = input_index(*audio_encoder_, "input_features_mask");
    if (mask_idx >= 0) {
        auto& mb = audio_encoder_->input_buffers[mask_idx];
        std::fill(mb.begin(), mb.end(), static_cast<uint8_t>(1));
    }
    audio_encoder_->graph->execute();
    for (size_t i = 0; i < audio_encoder_->output_node_ids.size() && i < audio_encoder_->logical_outputs.size(); ++i) {
        const std::string& name = audio_encoder_->logical_outputs[i];
        size_t node_id = static_cast<size_t>(audio_encoder_->output_node_ids[i]);
        const auto& desc = audio_encoder_->graph->get_output_buffer(node_id);
        void* ptr = audio_encoder_->graph->get_output(node_id);
        auto& slot = media_features_[name];
        slot.assign(desc.byte_size, 0);
        std::memcpy(slot.data(), ptr, desc.byte_size);
        media_feature_shapes_[name] = desc.shape;
        media_feature_precisions_[name] = desc.precision;
    }
}

uint32_t Model::argmax_last_logits() {
    size_t out_node = static_cast<size_t>(decoder_->output_node_ids.empty() ? 0 : decoder_->output_node_ids[0]);
    const auto& desc = decoder_->graph->get_output_buffer(out_node);
    void* ptr = decoder_->graph->get_output(out_node);
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    size_t seq = desc.shape.size() >= 2 ? desc.shape[desc.shape.size() - 2] : 1;
    size_t row_off = (seq > 0 ? (seq - 1) * vocab : 0);
    uint32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    if (desc.precision == Precision::FP32) {
        float* p = static_cast<float*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) if (p[i] > best_v) { best_v = p[i]; best = static_cast<uint32_t>(i); }
    } else if (desc.precision == Precision::FP16) {
        __fp16* p = static_cast<__fp16*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) {
            float v = static_cast<float>(p[i]);
            if (v > best_v) { best_v = v; best = static_cast<uint32_t>(i); }
        }
    } else {
        int8_t* p = static_cast<int8_t*>(ptr) + row_off;
        for (size_t i = 0; i < vocab; ++i) if (p[i] > best_v) { best_v = static_cast<float>(p[i]); best = static_cast<uint32_t>(i); }
    }
    return best;
}

void Model::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& /*profile_file*/) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_ + i, /*read_logits=*/false);
    }
    cache_total_seq_len_ += tokens.size();
}

void Model::prefill_with_images(const std::vector<uint32_t>& tokens,
                                const std::vector<std::string>& image_paths,
                                const std::string& profile_file) {
    prefill_with_media(tokens, image_paths, {}, profile_file);
}

void Model::prefill_with_audio(const std::vector<uint32_t>& tokens,
                               const std::vector<float>& audio_features,
                               const std::string& profile_file) {
    prefill_with_media(tokens, {}, audio_features, profile_file);
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

void fill_int_buffer(std::vector<uint8_t>& buf, Precision prec, int64_t value, size_t count) {
    const size_t elem = PrecisionTraits::size_of(prec);
    const size_t cap = elem ? buf.size() / elem : 0;
    const size_t n = std::min(cap, count);
    for (size_t i = 0; i < n; ++i) {
        switch (prec) {
            case Precision::FP32: reinterpret_cast<float*>(buf.data())[i] = static_cast<float>(value); break;
            case Precision::FP16: reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(value); break;
            case Precision::INT8: reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(value); break;
            default: reinterpret_cast<int64_t*>(buf.data())[i] = value; break;
        }
    }
    if (n < cap) {
        std::memset(buf.data() + n * elem, 0, (cap - n) * elem);
    }
}

void write_tokens_buffer(std::vector<uint8_t>& buf, Precision prec,
                         const std::vector<uint32_t>& tokens, size_t offset) {
    const size_t elem = PrecisionTraits::size_of(prec);
    const size_t cap = elem ? buf.size() / elem : 0;
    const size_t avail = (offset < tokens.size()) ? (tokens.size() - offset) : 0;
    const size_t n = std::min(cap, avail);
    for (size_t i = 0; i < n; ++i) {
        int64_t v = static_cast<int64_t>(tokens[offset + i]);
        switch (prec) {
            case Precision::FP32: reinterpret_cast<float*>(buf.data())[i] = static_cast<float>(v); break;
            case Precision::FP16: reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(v); break;
            case Precision::INT8: reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(v); break;
            default: reinterpret_cast<int64_t*>(buf.data())[i] = v; break;
        }
    }
    if (n < cap) {
        std::memset(buf.data() + n * elem, 0, (cap - n) * elem);
    }
}

} // namespace

void Model::copy_cache_state(const Component& src, Component& dst) {
    if (src.cache_states.empty() || dst.cache_states.empty()) return;
    if (src.cache_states.size() != dst.cache_states.size()) {
        throw std::runtime_error("cache state count mismatch between " + src.name + " and " + dst.name);
    }
    for (size_t i = 0; i < src.cache_states.size(); ++i) {
        const auto& s = src.cache_states[i];
        const auto& d = dst.cache_states[i];
        if (s.layer_key != d.layer_key) {
            throw std::runtime_error("cache layer mismatch: " + s.layer_key + " vs " + d.layer_key);
        }
        for (auto pair : {std::pair<int,int>{s.key_node_id, d.key_node_id},
                          std::pair<int,int>{s.value_node_id, d.value_node_id}}) {
            const auto& sd = src.graph->get_output_buffer(static_cast<size_t>(pair.first));
            const auto& dd = dst.graph->get_output_buffer(static_cast<size_t>(pair.second));
            if (sd.byte_size != dd.byte_size) {
                throw std::runtime_error("cache byte size mismatch for layer " + s.layer_key);
            }
            std::memcpy(dst.graph->get_output(static_cast<size_t>(pair.second)),
                        src.graph->get_output(static_cast<size_t>(pair.first)),
                        sd.byte_size);
        }
    }
}

bool Model::run_chunk_prefill_path(const std::vector<uint32_t>& tokens,
                                   const std::vector<std::string>& image_paths,
                                   const std::vector<float>& audio_features) {
    const bool have_images = !image_paths.empty() && vision_encoder_ != nullptr;
    const bool have_audio = !audio_features.empty() && audio_encoder_ != nullptr;

    if (have_images) {
        for (const auto& path : image_paths) {
            if (family_ == "lfm2_vl") {
                Lfm2VlImagePreprocessed prep = preprocess_lfm2_vl_image(path, config_);
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
                int pm_idx = input_index(*vision_encoder_, "pixel_attention_mask");
                if (pm_idx >= 0) {
                    auto& pm_buf = vision_encoder_->input_buffers[pm_idx];
                    size_t pm_node = static_cast<size_t>(vision_encoder_->runtime_input_node_ids[pm_idx]);
                    const auto& pm_desc = vision_encoder_->graph->get_output_buffer(pm_node);
                    const size_t elem = PrecisionTraits::size_of(pm_desc.precision);
                    const size_t cap = elem ? pm_buf.size() / elem : 0;
                    const size_t n = std::min(cap, prep.pixel_attention_mask.size());
                    for (size_t i = 0; i < n; ++i) {
                        int64_t v = prep.pixel_attention_mask[i];
                        switch (pm_desc.precision) {
                            case Precision::FP32: reinterpret_cast<float*>(pm_buf.data())[i] = static_cast<float>(v); break;
                            case Precision::FP16: reinterpret_cast<__fp16*>(pm_buf.data())[i] = static_cast<__fp16>(v); break;
                            case Precision::INT8: reinterpret_cast<int8_t*>(pm_buf.data())[i] = static_cast<int8_t>(v); break;
                            default: reinterpret_cast<int64_t*>(pm_buf.data())[i] = v; break;
                        }
                    }
                    if (n < cap) std::memset(pm_buf.data() + n * elem, 0, (cap - n) * elem);
                }
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
                slot.assign(desc.byte_size, 0);
                std::memcpy(slot.data(), ptr, desc.byte_size);
                media_feature_shapes_[name] = desc.shape;
                media_feature_precisions_[name] = desc.precision;
            }
        }
    }

    if (have_audio) {
        run_audio_encoder(audio_features);
    }

    {
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
    }
    lm_encoder_->graph->execute();

    std::map<std::string, std::vector<uint8_t>> store_bytes;
    std::map<std::string, Precision> store_prec;
    std::map<std::string, std::vector<size_t>> store_shape;
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
        copy_cache_state(*decoder_prefill_chunk_, *decoder_);
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
                               const std::vector<float>& audio_features,
                               const std::string& profile_file) {
    if (tokens.empty()) return;
    const bool have_images = !image_paths.empty() && vision_encoder_ != nullptr;
    const bool have_audio = !audio_features.empty() && audio_encoder_ != nullptr;
    if (!have_images && !have_audio) {
        prefill(tokens, get_prefill_chunk_size(), profile_file);
        return;
    }

    const bool can_chunk_prefill =
        lm_encoder_ != nullptr && decoder_prefill_chunk_ != nullptr &&
        (vision_encoder_ != nullptr || audio_encoder_ != nullptr);
    if (can_chunk_prefill) {
        if (run_chunk_prefill_path(tokens, image_paths, audio_features)) {
            (void)profile_file;
            return;
        }
    }
    if (!lm_encoder_media_step_) {
        CACTUS_LOG_WARN("model", "Bundle has neither chunk-prefill nor lm_encoder_media_step; falling back to text-only prefill");
        prefill(tokens, get_prefill_chunk_size(), profile_file);
        return;
    }

    if (have_images) {
        for (const auto& path : image_paths) {
            run_vision_encoder(path);
        }
    }
    if (have_audio) {
        run_audio_encoder(audio_features);
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

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t t = tokens[i];
        size_t pos = cache_total_seq_len_ + i;
        if (image_tok != 0 && t == image_tok && !image_feature_name.empty() && lm_encoder_media_step_) {
            const auto& feat = media_features_[image_feature_name];
            const uint8_t* row = feat.data() + image_consumed * image_row_bytes;
            if (image_consumed * image_row_bytes + image_row_bytes <= feat.size()) {
                run_media_step(pos, row, image_row_bytes, image_feature_prec);
                ++image_consumed;
                continue;
            }
        }
        if (audio_tok != 0 && t == audio_tok && !audio_feature_name.empty() && lm_encoder_media_step_) {
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
    (void)profile_file;
}

uint32_t Model::decode(const std::vector<uint32_t>& tokens, float /*temperature*/, float /*top_p*/,
                        size_t /*top_k*/, const std::string& /*profile_file*/, float* out_entropy,
                        float /*min_p*/, float /*repetition_penalty*/) {
    if (tokens.empty()) return 0;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_ + i, /*read_logits=*/false);
    }
    run_step(tokens.back(), cache_total_seq_len_ + tokens.size() - 1, /*read_logits=*/true);
    cache_total_seq_len_ += tokens.size();
    if (out_entropy) *out_entropy = 0.0f;
    uint32_t result = argmax_last_logits();
    record_sampled_token(result);
    return result;
}

uint32_t Model::decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& /*mel_bins*/,
                                  float temperature, float top_p, size_t top_k, const std::string& profile_file,
                                  float* out_entropy, float min_p, float repetition_penalty,
                                  float* /*out_token_time_start*/, float* /*out_token_time_end*/) {
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

std::vector<uint32_t> Model::transcribe_parakeet_tdt(const std::vector<float>& audio_features) {
    std::vector<uint32_t> emitted;

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

    audio_enc->graph->execute();

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
    const size_t hidden_elem = PrecisionTraits::size_of(hidden_desc.precision);
    const size_t frame_bytes = D * hidden_elem;

    auto zero_state = [&](const std::string& name) {
        int idx = input_index(*dec, name);
        if (idx < 0) return;
        auto& buf = dec->input_buffers[idx];
        std::memset(buf.data(), 0, buf.size());
    };
    zero_state("state_h_0");
    zero_state("state_c_0");
    zero_state("state_h_1");
    zero_state("state_c_1");

    std::vector<uint32_t> durations = config_.tdt_durations;
    if (durations.empty()) {
        for (uint32_t i = 0; i < config_.tdt_num_durations; ++i) durations.push_back(i);
    }
    if (durations.empty()) durations.push_back(1);

    const uint32_t configured_blank = config_.tdt_blank_id;
    uint32_t last_token = configured_blank;
    size_t time_index = 0;

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

    while (time_index < T) {
        const uint8_t* frame_ptr = hidden_ptr + time_index * frame_bytes;
        write_typed_buffer(ef_buf, ef_desc.precision, frame_ptr, frame_bytes, hidden_desc.precision);

        size_t symbols_added = 0;
        bool advanced = false;
        while (symbols_added < 10) {
            switch (tok_prec) {
                case Precision::FP32: *reinterpret_cast<float*>(tok_data) = static_cast<float>(last_token); break;
                case Precision::FP16: *reinterpret_cast<__fp16*>(tok_data) = static_cast<__fp16>(last_token); break;
                case Precision::INT8: *reinterpret_cast<int8_t*>(tok_data) = static_cast<int8_t>(last_token); break;
                default: *reinterpret_cast<int32_t*>(tok_data) = static_cast<int32_t>(last_token); break;
            }
            dec->graph->execute();

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
                if (v > best_token_score) { best_token_score = v; next_token = static_cast<uint32_t>(i); }
            }
            uint32_t best_duration_idx = 0;
            float best_duration_score = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < num_durations; ++i) {
                float v = get_logit(token_class_count + i);
                if (v > best_duration_score) { best_duration_score = v; best_duration_idx = static_cast<uint32_t>(i); }
            }

            const uint32_t skip = durations[std::min<uint32_t>(best_duration_idx, static_cast<uint32_t>(durations.size() - 1))];

            if (next_token != effective_blank) {
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

    return emitted;
}

uint32_t Model::decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& /*image_paths*/,
                                     float temperature, float top_p, size_t top_k, const std::string& profile_file,
                                     float* out_entropy, float min_p, float repetition_penalty) {
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

std::vector<float> Model::get_image_embeddings(const std::string& /*image_path*/) {
    throw std::runtime_error("Image embeddings not wired up for transpiled bundles yet");
}

std::vector<float> Model::get_audio_embeddings(const std::vector<float>& /*mel_bins*/) {
    throw std::runtime_error("Audio embeddings not wired up for transpiled bundles yet");
}

void Model::reset_cache() {
    cache_total_seq_len_ = 0;
    token_history_.clear();
    for (auto& kv : components_) {
        Component& comp = kv.second;
        if (!comp.graph) continue;
        for (const auto& cs : comp.cache_states) {
            for (int node_id : {cs.key_node_id, cs.value_node_id}) {
                if (node_id < 0) continue;
                const auto& desc = comp.graph->get_output_buffer(static_cast<size_t>(node_id));
                if (desc.byte_size < sizeof(uint64_t) || !desc.get_data()) continue;
                void* data = comp.graph->get_output(static_cast<size_t>(node_id));
                if (!data) continue;
                *static_cast<uint64_t*>(data) = 0;
            }
        }
    }
}

void Model::set_cache_window(size_t /*window_size*/, size_t /*sink_size*/) {}

void Model::remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges) {
    size_t total_removed = 0;
    for (const auto& r : ranges) total_removed += r.second;
    if (cache_total_seq_len_ >= total_removed)
        cache_total_seq_len_ -= total_removed;
    else
        cache_total_seq_len_ = 0;

    struct CacheHeader {
        uint64_t current_seq_len;
        uint64_t max_seq_len;
        uint64_t num_kv_heads;
        uint64_t head_dim;
        uint64_t sink_size;
        uint64_t reserved[3];
    };
    constexpr size_t kHeaderBytes = 64;
    static_assert(sizeof(CacheHeader) == kHeaderBytes, "CacheHeader layout mismatch");

    auto sorted_ranges = ranges;
    std::sort(sorted_ranges.begin(), sorted_ranges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& kv : components_) {
        Component& comp = kv.second;
        if (!comp.graph) continue;
        for (const auto& cs : comp.cache_states) {
            for (int node_id : {cs.key_node_id, cs.value_node_id}) {
                if (node_id < 0) continue;
                const auto& desc = comp.graph->get_output_buffer(static_cast<size_t>(node_id));
                if (desc.byte_size <= kHeaderBytes || !desc.get_data()) continue;
                void* raw = comp.graph->get_output(static_cast<size_t>(node_id));
                if (!raw) continue;
                auto* hdr = static_cast<CacheHeader*>(raw);
                size_t cur = hdr->current_seq_len;
                if (cur == 0) continue;
                size_t kv_heads = hdr->num_kv_heads;
                size_t hdim = hdr->head_dim;
                if (kv_heads == 0 || hdim == 0) continue;
                size_t token_elems = kv_heads * hdim;
                size_t num_groups = (hdim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
                size_t token_scales = kv_heads * num_groups;
                size_t max_seq = hdr->max_seq_len;

                size_t new_len = cur;
                if (desc.precision == Precision::FP16) {
                    auto* base = reinterpret_cast<__fp16*>(static_cast<char*>(raw) + kHeaderBytes);
                    for (auto it = sorted_ranges.rbegin(); it != sorted_ranges.rend(); ++it) {
                        size_t start = it->first;
                        if (start >= new_len) continue;
                        size_t count = std::min(it->second, new_len - start);
                        size_t tail_start = start + count;
                        size_t tail_count = new_len - tail_start;
                        if (tail_count > 0) {
                            std::memmove(base + start * token_elems,
                                         base + tail_start * token_elems,
                                         tail_count * token_elems * sizeof(__fp16));
                        }
                        new_len -= count;
                    }
                } else {
                    auto* int8_base = reinterpret_cast<int8_t*>(static_cast<char*>(raw) + kHeaderBytes);
                    auto* scale_base = reinterpret_cast<float*>(static_cast<char*>(raw) + kHeaderBytes +
                                                                max_seq * kv_heads * hdim);
                    for (auto it = sorted_ranges.rbegin(); it != sorted_ranges.rend(); ++it) {
                        size_t start = it->first;
                        if (start >= new_len) continue;
                        size_t count = std::min(it->second, new_len - start);
                        size_t tail_start = start + count;
                        size_t tail_count = new_len - tail_start;
                        if (tail_count > 0) {
                            std::memmove(int8_base + start * token_elems,
                                         int8_base + tail_start * token_elems,
                                         tail_count * token_elems);
                            std::memmove(scale_base + start * token_scales,
                                         scale_base + tail_start * token_scales,
                                         tail_count * token_scales * sizeof(float));
                        }
                        new_len -= count;
                    }
                }
                hdr->current_seq_len = new_len;
            }
        }
    }
}

std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& /*tokens*/, bool /*pooled*/,
                                          bool /*normalize*/, const std::string& /*profile_file*/) {
    return {};
}

bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        CACTUS_LOG_ERROR("config", "Failed to open config file: " << config_path);
        return false;
    }
    
    std::string line;
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
            else if (mt == "gemma") model_type = ModelType::GEMMA;
            else if (mt == "gemma3n") model_type = ModelType::GEMMA3N;
            else if (mt == "lfm2") model_type = ModelType::LFM2;
            else if (mt == "whisper") model_type = ModelType::WHISPER;
            else if (mt == "parakeet_tdt" || mt == "parakeet-tdt") model_type = ModelType::PARAKEET_TDT;
            else if (mt == "youtu") model_type = ModelType::YOUTU;
            else if (mt == "needle") model_type = ModelType::NEEDLE;
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
        else if (key == "channel_open_token_id") channel_open_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "channel_close_token_id") channel_close_token_id = static_cast<uint32_t>(std::stoul(value));
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
            default_cloud_handoff_threshold = 0.92f;
            default_rolling_entropy_window = 16;
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

    return true;
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

bool Model::load_npu_prefill(const std::string& /*model_path*/) {
    return false;
}

double Model::score_tokens_window_logprob(const std::vector<uint32_t>& /*tokens*/, size_t /*start*/,
                                            size_t /*end*/, size_t /*context*/, size_t* tokens_scored) {
    if (tokens_scored) *tokens_scored = 0;
    return 0.0;
}

}
}
