#include "engine.h"
#include "cactus_graph.h"

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
    if (!encoder_ || !decoder_) {
        CACTUS_LOG_ERROR("model", "Bundle missing lm_encoder_step or decoder_step components");
        return false;
    }
    if (!bind_runtime_buffers(*encoder_)) return false;
    if (!bind_runtime_buffers(*decoder_)) return false;

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

void Model::run_step(uint32_t token_id, size_t position, bool /*read_logits*/) {
    write_int_input(*encoder_, "input_ids", static_cast<int64_t>(token_id));
    write_int_input(*encoder_, "position_ids", static_cast<int64_t>(position));
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
    decoder_->graph->execute();
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
                                const std::vector<std::string>& /*image_paths*/,
                                const std::string& profile_file) {
    prefill(tokens, get_prefill_chunk_size(), profile_file);
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
}

void Model::set_cache_window(size_t /*window_size*/, size_t /*sink_size*/) {}

void Model::remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges) {
    size_t total_removed = 0;
    for (const auto& r : ranges) total_removed += r.second;
    if (cache_total_seq_len_ >= total_removed)
        cache_total_seq_len_ -= total_removed;
    else
        cache_total_seq_len_ = 0;
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
