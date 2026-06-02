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
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>

namespace cactus {
namespace engine {

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
    const bool is_text_embedding =
        components_.count("text_embedding")
        && !components_.count("decoder")
        && !components_.count("decoder_step")
        && !components_.count("lm_encoder_step");
    if (is_text_embedding) {
        // Embedding-only bundle: no decode route; get_embeddings loads the
        // text_embedding component on demand.
        cache_max_seq_len_ = context_size;
        initialized_ = true;
        return true;
    }
    std::string encoder_name;
    std::string decoder_name;
    std::unordered_set<std::string> required_components;
    const bool is_whisper_transcription =
        config_.model_type == Config::ModelType::WHISPER &&
        components_.count("audio_encoder") &&
        components_.count("decoder_cross_kv") &&
        components_.count("decoder_step");
    bool has_chunked_prefill = components_.count("lm_encoder_step")
        && components_.count("decoder_media_step")
        && components_.count("lm_encoder_text_chunk")
        && components_.count("decoder_prefill_chunk");
    if (is_whisper_transcription) {
        decoder_name = "decoder_step";
        decode_route_ = DecodeRoute::DIRECT_DECODER_STEP;
        required_components = {"audio_encoder", "decoder_cross_kv", decoder_name};
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
        CACTUS_LOG_ERROR("model", "Bundle missing required components: need lm_encoder_step+decoder_step (LM), audio_encoder+decoder (transcription), or Whisper audio_encoder+decoder_cross_kv+decoder_step");
        return false;
    }
    for (const auto& optional : {
             "vision_encoder",
             "audio_encoder",
             "lm_encoder_media_step",
             "decoder_prefill_chunk",
             "lm_encoder",
         }) {
        if (components_.count(optional)) {
            required_components.insert(optional);
        }
    }
    if (!load_components(required_components)) return false;
    if (!encoder_name.empty()) encoder_ = &components_.at(encoder_name);
    if (!decoder_name.empty()) decoder_ = &components_.at(decoder_name);
    if (components_.count("decoder_prefill_chunk") && components_.at("decoder_prefill_chunk").graph) {
        decoder_prefill_ = &components_.at("decoder_prefill_chunk");
        decoder_prefill_chunk_ = decoder_prefill_;
    }
    if (components_.count("lm_encoder_text_chunk") && components_.at("lm_encoder_text_chunk").graph) {
        prefill_encoder_ = &components_.at("lm_encoder_text_chunk");
    }
    vision_encoder_ = components_.count("vision_encoder") ? &components_.at("vision_encoder") : nullptr;
    audio_encoder_ = components_.count("audio_encoder") ? &components_.at("audio_encoder") : nullptr;
    lm_encoder_media_step_ = components_.count("lm_encoder_media_step") ? &components_.at("lm_encoder_media_step") : nullptr;
    lm_encoder_ = components_.count("lm_encoder") ? &components_.at("lm_encoder") : nullptr;
    lm_encoder_text_chunk_ = components_.count("lm_encoder_text_chunk") ? &components_.at("lm_encoder_text_chunk") : nullptr;
    lm_encoder_media_chunk_ = components_.count("lm_encoder_media_chunk") ? &components_.at("lm_encoder_media_chunk") : nullptr;
    std::vector<Component*> to_bind = {
        encoder_,
        prefill_encoder_,
        decoder_,
        decoder_prefill_,
        vision_encoder_,
        audio_encoder_,
        lm_encoder_media_step_,
        lm_encoder_,
        lm_encoder_text_chunk_,
        lm_encoder_media_chunk_,
    };
    if (components_.count("decoder_cross_kv")) {
        to_bind.push_back(&components_.at("decoder_cross_kv"));
    }
    std::unordered_set<Component*> bound;
    for (Component* comp : to_bind) {
        if (!comp || !comp->graph || bound.count(comp)) continue;
        if (!bind_runtime_buffers(*comp)) return false;
        bound.insert(comp);
    }

    if (vision_encoder_ && tokenizer_ && !vision_encoder_->output_node_ids.empty()) {
        size_t out_node = static_cast<size_t>(vision_encoder_->output_node_ids[0]);
        const auto& desc = vision_encoder_->graph->get_output_buffer(out_node);
        size_t n = 0;
        if (desc.shape.size() >= 3) n = desc.shape[desc.shape.size() - 2];
        else if (desc.shape.size() >= 2) n = desc.shape[0];
        if (n > 0) tokenizer_->set_image_soft_token_count(n);
    }

    cache_max_seq_len_ = context_size;

    if (!npu_audio_encoder_mlpackage_.empty()) {
        std::string full_path = bundle_dir + "/" + npu_audio_encoder_mlpackage_;
        if (!load_npu_audio_encoder(full_path)) {
            CACTUS_LOG_WARN("model", "NPU audio encoder load failed for " << full_path << "; falling back to CPU");
        }
    }
    if (!npu_vision_encoder_mlpackage_.empty()) {
        std::string full_path = bundle_dir + "/" + npu_vision_encoder_mlpackage_;
        if (!load_npu_vision_encoder(full_path)) {
            CACTUS_LOG_WARN("model", "NPU vision encoder load failed for " << full_path << "; falling back to CPU");
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
    if (obj.count("npu_audio_encoder") && obj.at("npu_audio_encoder").is<std::string>()) {
        npu_audio_encoder_mlpackage_ = obj.at("npu_audio_encoder").get<std::string>();
    }
    if (obj.count("npu_vision_encoder") && obj.at("npu_vision_encoder").is<std::string>()) {
        npu_vision_encoder_mlpackage_ = obj.at("npu_vision_encoder").get<std::string>();
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

void Model::run_step(uint32_t token_id, size_t position, bool /*read_logits*/) {
    if (decode_route_ == DecodeRoute::DIRECT_DECODER_STEP) {
        write_int_input(*decoder_, "input_ids", static_cast<int64_t>(token_id));
        write_int_input(*decoder_, "position_ids", static_cast<int64_t>(position));
        decoder_->graph->execute();
        return;
    }
    run_encoder_step(token_id, position);
    copy_component_outputs_to_inputs(*encoder_, *decoder_);
    decoder_->graph->execute();
}

void Model::run_encoder_step(uint32_t token_id, size_t position) {
    write_int_input(*encoder_, "input_ids", static_cast<int64_t>(token_id));
    write_int_input(*encoder_, "position_ids", static_cast<int64_t>(position));
    encoder_->graph->execute();
}

void Model::copy_component_outputs_to_inputs(const Component& source, Component& target) {
    for (size_t i = 0; i < source.output_node_ids.size() && i < source.logical_outputs.size(); ++i) {
        const std::string& out_name = source.logical_outputs[i];
        int dst_idx = input_index(target, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(source.output_node_ids[i]);
        const auto& src_desc = source.graph->get_output_buffer(src_node);
        size_t dst_node = static_cast<size_t>(target.runtime_input_node_ids[dst_idx]);
        const auto& dst_desc = target.graph->get_output_buffer(dst_node);
        std::fill(target.input_buffers[dst_idx].begin(), target.input_buffers[dst_idx].end(), 0);
        size_t elements = std::min(src_desc.total_size, dst_desc.total_size);
        if (!copy_component_tensor(*source.graph, src_desc, src_node, dst_desc, target.input_buffers[dst_idx], 0, elements, out_name)) {
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
        }
    }
}

void Model::copy_component_outputs_to_chunk_inputs(const Component& source, Component& target, size_t token_index) {
    for (size_t i = 0; i < source.output_node_ids.size() && i < source.logical_outputs.size(); ++i) {
        const std::string& out_name = source.logical_outputs[i];
        int dst_idx = input_index(target, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(source.output_node_ids[i]);
        const auto& src_desc = source.graph->get_output_buffer(src_node);
        size_t dst_node = static_cast<size_t>(target.runtime_input_node_ids[dst_idx]);
        const auto& dst_desc = target.graph->get_output_buffer(dst_node);
        size_t chunk_tokens = component_chunk_tokens(target, out_name);
        if (chunk_tokens <= token_index || chunk_tokens == 0) {
            throw std::runtime_error("chunk prefill token index exceeds input capacity for " + out_name);
        }
        if (dst_desc.total_size % chunk_tokens != 0) {
            throw std::runtime_error("chunk prefill input shape is not token-aligned for " + out_name);
        }
        size_t elements_per_token = dst_desc.total_size / chunk_tokens;
        if (src_desc.total_size != elements_per_token) {
            throw std::runtime_error("component output/input token shape mismatch for " + out_name);
        }
        if (!copy_component_tensor(
                *source.graph,
                src_desc,
                src_node,
                dst_desc,
                target.input_buffers[dst_idx],
                token_index * elements_per_token,
                src_desc.total_size,
                out_name)) {
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
        }
    }
}

void Model::copy_component_outputs_to_chunk_inputs_range(const Component& source, Component& target, size_t token_offset) {
    for (size_t i = 0; i < source.output_node_ids.size() && i < source.logical_outputs.size(); ++i) {
        const std::string& out_name = source.logical_outputs[i];
        int dst_idx = input_index(target, out_name);
        if (dst_idx < 0) continue;
        size_t src_node = static_cast<size_t>(source.output_node_ids[i]);
        const auto& src_desc = source.graph->get_output_buffer(src_node);
        size_t dst_node = static_cast<size_t>(target.runtime_input_node_ids[dst_idx]);
        const auto& dst_desc = target.graph->get_output_buffer(dst_node);
        size_t src_tokens = component_output_tokens(source, out_name);
        size_t dst_tokens = component_chunk_tokens(target, out_name);
        if (src_tokens == 0 || dst_tokens == 0 || token_offset + src_tokens > dst_tokens) {
            throw std::runtime_error("chunk prefill output range exceeds input capacity for " + out_name);
        }
        if (src_desc.total_size % src_tokens != 0 || dst_desc.total_size % dst_tokens != 0) {
            throw std::runtime_error("chunk prefill output/input shape is not token-aligned for " + out_name);
        }
        size_t src_elements_per_token = src_desc.total_size / src_tokens;
        size_t dst_elements_per_token = dst_desc.total_size / dst_tokens;
        if (src_elements_per_token != dst_elements_per_token) {
            throw std::runtime_error("component output/input token shape mismatch for " + out_name);
        }
        if (!copy_component_tensor(
                *source.graph,
                src_desc,
                src_node,
                dst_desc,
                target.input_buffers[dst_idx],
                token_offset * dst_elements_per_token,
                src_desc.total_size,
                out_name)) {
            throw std::runtime_error("component output/input precision mismatch for " + out_name);
        }
    }
}

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

void Model::copy_cache_states(const Component& source, Component& target, size_t logical_current) {
    if (source.cache_states.empty() || source.cache_states.size() != target.cache_states.size()) {
        throw std::runtime_error("prefill and step cache states are not compatible");
    }
    for (size_t i = 0; i < source.cache_states.size(); ++i) {
        const auto& src = source.cache_states[i];
        const auto& dst = target.cache_states[i];
        if (src.layer_key != dst.layer_key) {
            throw std::runtime_error("prefill and step cache layer mismatch: " + src.layer_key + " != " + dst.layer_key);
        }
        for (auto [src_node, dst_node] : {std::pair<int, int>{src.key_node_id, dst.key_node_id}, std::pair<int, int>{src.value_node_id, dst.value_node_id}}) {
            const auto& src_desc = source.graph->get_output_buffer(static_cast<size_t>(src_node));
            const auto& dst_desc = target.graph->get_output_buffer(static_cast<size_t>(dst_node));
            if (src_desc.precision != dst_desc.precision) {
                std::ostringstream oss;
                oss << "prefill and step cache precision mismatch at layer " << src.layer_key
                    << ": " << static_cast<int>(src_desc.precision)
                    << " vs " << static_cast<int>(dst_desc.precision);
                throw std::runtime_error(oss.str());
            }
            void* src_ptr = source.graph->get_output(static_cast<size_t>(src_node));
            void* dst_ptr = target.graph->get_output(static_cast<size_t>(dst_node));

            const OpType src_op = source.graph->get_node_op_type(static_cast<size_t>(src_node));
            const OpType dst_op = target.graph->get_node_op_type(static_cast<size_t>(dst_node));
            if (src_op != dst_op) {
                throw std::runtime_error(
                    "cache state op_type mismatch between prefill and step at layer "
                    + src.layer_key);
            }
            if (src_op == OpType::RECURRENT_CACHE_STATE) {
                if (src_desc.byte_size != dst_desc.byte_size) {
                    throw std::runtime_error(
                        "recurrent cache buffer shape mismatch between prefill and step at layer "
                        + src.layer_key);
                }
                std::memcpy(dst_ptr, src_ptr, src_desc.byte_size);
                continue;
            }
            if (src_op == OpType::CONV_CACHE_STATE) {
                if (src_desc.byte_size != dst_desc.byte_size) {
                    throw std::runtime_error(
                        "conv cache buffer shape mismatch between prefill and step at layer "
                        + src.layer_key);
                }
                std::memcpy(dst_ptr, src_ptr, src_desc.byte_size);
                continue;
            }
            if (src_desc.byte_size == dst_desc.byte_size && logical_current == std::numeric_limits<size_t>::max()) {
                std::memcpy(dst_ptr, src_ptr, src_desc.byte_size);
                continue;
            }
            if (src_desc.precision == Precision::INT8) {
                auto* src_meta = static_cast<uint64_t*>(src_ptr);
                auto* dst_meta = static_cast<uint64_t*>(dst_ptr);
                const size_t src_current = static_cast<size_t>(src_meta[0]);
                const size_t effective_current = std::min(src_current, logical_current);
                const size_t src_max = static_cast<size_t>(src_meta[1]);
                const size_t kv_heads = static_cast<size_t>(src_meta[2]);
                const size_t head_dim = static_cast<size_t>(src_meta[3]);
                const size_t sink = static_cast<size_t>(src_meta[4]);
                if (src_max == 0 || kv_heads == 0 || head_dim == 0) {
                    throw std::runtime_error("prefill cache metadata is not initialized for layer " + src.layer_key);
                }
                const size_t groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
                const size_t int8_stride = kv_heads * head_dim;
                const size_t scale_stride = kv_heads * groups;
                const size_t row_bytes = int8_stride + scale_stride * sizeof(float);
                const size_t dst_max = (dst_desc.byte_size - 64) / row_bytes;
                if (dst_max == 0) {
                    throw std::runtime_error("step cache capacity is zero for layer " + src.layer_key);
                }
                const size_t dst_current = std::min(effective_current, dst_max);
                dst_meta[0] = dst_current;
                dst_meta[1] = dst_max;
                dst_meta[2] = kv_heads;
                dst_meta[3] = head_dim;
                dst_meta[4] = std::min(sink, dst_current);
                std::memset(static_cast<char*>(dst_ptr) + 64, 0, dst_desc.byte_size - 64);

                const auto* src_i8 = static_cast<const int8_t*>(src_ptr) + 64;
                const auto* src_scales = reinterpret_cast<const float*>(
                    static_cast<const char*>(src_ptr) + 64 + src_max * int8_stride);
                auto* dst_i8 = static_cast<int8_t*>(dst_ptr) + 64;
                auto* dst_scales = reinterpret_cast<float*>(
                    static_cast<char*>(dst_ptr) + 64 + dst_max * int8_stride);
                auto copy_rows = [&](size_t dst_row, size_t src_row, size_t rows) {
                    if (rows == 0) return;
                    std::memcpy(
                        dst_i8 + dst_row * int8_stride,
                        src_i8 + src_row * int8_stride,
                        rows * int8_stride);
                    std::memcpy(
                        dst_scales + dst_row * scale_stride,
                        src_scales + src_row * scale_stride,
                        rows * scale_stride * sizeof(float));
                };
                if (effective_current <= dst_max) {
                    copy_rows(0, 0, effective_current);
                } else {
                    const size_t copied_sink = std::min(sink, dst_max);
                    const size_t tail_rows = dst_max - copied_sink;
                    copy_rows(0, 0, copied_sink);
                    if (tail_rows > 0) copy_rows(copied_sink, effective_current - tail_rows, tail_rows);
                }
                continue;
            }
            if (PrecisionTraits::is_cq(src_desc.precision) || src_desc.byte_size < 64 || dst_desc.byte_size < 64) {
                std::ostringstream oss;
                oss << "prefill and step cache buffer mismatch at layer " << src.layer_key
                    << ": " << src_desc.byte_size << " bytes vs " << dst_desc.byte_size << " bytes";
                throw std::runtime_error(oss.str());
            }

            auto* src_meta = static_cast<uint64_t*>(src_ptr);
            auto* dst_meta = static_cast<uint64_t*>(dst_ptr);
            const size_t src_current = static_cast<size_t>(src_meta[0]);
            const size_t effective_current = std::min(src_current, logical_current);
            const size_t src_max = static_cast<size_t>(src_meta[1]);
            const size_t kv_heads = static_cast<size_t>(src_meta[2]);
            const size_t head_dim = static_cast<size_t>(src_meta[3]);
            const size_t sink = static_cast<size_t>(src_meta[4]);
            if (src_max == 0 || kv_heads == 0 || head_dim == 0) {
                throw std::runtime_error("prefill cache metadata is not initialized for layer " + src.layer_key);
            }
            const size_t row_bytes = kv_heads * head_dim * PrecisionTraits::size_of(src_desc.precision);
            const size_t dst_max = (dst_desc.byte_size - 64) / row_bytes;
            if (dst_max == 0) {
                throw std::runtime_error("step cache capacity is zero for layer " + src.layer_key);
            }
            const size_t dst_current = std::min(effective_current, dst_max);
            dst_meta[0] = dst_current;
            dst_meta[1] = dst_max;
            dst_meta[2] = kv_heads;
            dst_meta[3] = head_dim;
            dst_meta[4] = std::min(sink, dst_current);
            std::memset(static_cast<char*>(dst_ptr) + 64, 0, dst_desc.byte_size - 64);
            const auto* src_rows = static_cast<const char*>(src_ptr) + 64;
            auto* dst_rows = static_cast<char*>(dst_ptr) + 64;
            if (effective_current <= dst_max) {
                std::memcpy(dst_rows, src_rows, effective_current * row_bytes);
            } else {
                const size_t copied_sink = std::min(sink, dst_max);
                const size_t tail_rows = dst_max - copied_sink;
                if (copied_sink > 0) {
                    std::memcpy(dst_rows, src_rows, copied_sink * row_bytes);
                }
                if (tail_rows > 0) {
                    std::memcpy(
                        dst_rows + copied_sink * row_bytes,
                        src_rows + (effective_current - tail_rows) * row_bytes,
                        tail_rows * row_bytes);
                }
            }
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
                        static_cast<uint64_t*>(ptr)[0] = 0;
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

Model::ChunkedPrefillResult Model::run_chunked_prefill(const std::vector<uint32_t>& tokens, size_t start_position, size_t chunk_size, bool prepare_decode) {
    ChunkedPrefillResult result;
    last_prefill_cache_copy_ms_ = 0.0;
    last_prefill_padding_tokens_ = 0;
    last_prefill_scalar_tail_tokens_ = 0;
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
    const bool has_recurrent_state = [&]() {
        if (!decoder_prefill_->graph) return false;
        for (const auto& state : decoder_prefill_->cache_states) {
            for (int node_id : {state.key_node_id, state.value_node_id}) {
                if (node_id < 0) continue;
                if (decoder_prefill_->graph->get_node_op_type(static_cast<size_t>(node_id))
                    == OpType::RECURRENT_CACHE_STATE) {
                    return true;
                }
            }
        }
        return false;
    }();
    if (has_recurrent_state && whole_chunks_end > effective_chunk) {
        whole_chunks_end = effective_chunk;
    }
    const size_t tail_tokens = tokens.size() - whole_chunks_end;
    const size_t padding_cutoff = std::max<size_t>(1, effective_chunk / 16);
    const bool pad_tail = family_ != "lfm2_vl"
        && !has_recurrent_state
        && tail_tokens >= padding_cutoff;
    const size_t executable_tokens = whole_chunks_end + (pad_tail ? effective_chunk : 0);
    if (executable_tokens == 0) {
        result.scalar_tail_tokens = tail_tokens;
        last_prefill_scalar_tail_tokens_ = tail_tokens;
        return result;
    }
    result.padding_tokens = executable_tokens > tokens.size() ? executable_tokens - tokens.size() : 0;
    result.scalar_tail_tokens = tokens.size() - std::min(tokens.size(), executable_tokens);
    last_prefill_padding_tokens_ = result.padding_tokens;
    last_prefill_scalar_tail_tokens_ = result.scalar_tail_tokens;

    size_t encoder_chunk = 0;
    if (prefill_encoder_ && input_index(*prefill_encoder_, "input_ids") >= 0 && input_index(*prefill_encoder_, "position_ids") >= 0) {
        encoder_chunk = component_chunk_tokens(*prefill_encoder_, "input_ids");
        if (encoder_chunk == 0 || effective_chunk % encoder_chunk != 0) {
            encoder_chunk = 0;
        }
    }

    size_t processed = 0;
    while (processed + effective_chunk <= executable_tokens) {
        for (size_t i = 0; i < decoder_prefill_->input_buffers.size(); ++i) {
            std::fill(decoder_prefill_->input_buffers[i].begin(), decoder_prefill_->input_buffers[i].end(), 0);
        }
        if (encoder_chunk > 0) {
            for (size_t chunk_offset = 0; chunk_offset < effective_chunk; chunk_offset += encoder_chunk) {
                for (size_t i = 0; i < prefill_encoder_->input_buffers.size(); ++i) {
                    std::fill(prefill_encoder_->input_buffers[i].begin(), prefill_encoder_->input_buffers[i].end(), 0);
                }
                for (size_t i = 0; i < encoder_chunk; ++i) {
                    size_t index = processed + chunk_offset + i;
                    uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
                    write_int_input_at(*prefill_encoder_, "input_ids", i, static_cast<int64_t>(token));
                    write_int_input_at(*prefill_encoder_, "position_ids", i, static_cast<int64_t>(start_position + processed + chunk_offset + i));
                }
                prefill_encoder_->graph->execute();
                copy_component_outputs_to_chunk_inputs_range(*prefill_encoder_, *decoder_prefill_, chunk_offset);
            }
        } else {
            for (size_t i = 0; i < effective_chunk; ++i) {
                size_t index = processed + i;
                uint32_t token = index < tokens.size() ? tokens[index] : static_cast<uint32_t>(config_.pad_token_id);
                run_encoder_step(token, start_position + processed + i);
                copy_component_outputs_to_chunk_inputs(*encoder_, *decoder_prefill_, i);
            }
        }
        decoder_prefill_->graph->execute();
        processed += effective_chunk;
    }
    result.executed_tokens = processed;
    result.logical_tokens = std::min(tokens.size(), processed);
    if (result.logical_tokens > 0) {
        result.last_logit_row = (result.logical_tokens - 1) % effective_chunk;
    }
    if (processed > 0 && prepare_decode) {
        for (size_t i = 0; i < decoder_->input_buffers.size(); ++i) {
            std::fill(decoder_->input_buffers[i].begin(), decoder_->input_buffers[i].end(), 0);
        }
        auto copy_start = std::chrono::high_resolution_clock::now();
        copy_cache_states(*decoder_prefill_, *decoder_, start_position + result.logical_tokens);
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

namespace {
void write_typed_buffer(std::vector<uint8_t>& buf, Precision dst_prec,
                        const void* src_data, size_t src_bytes, Precision src_prec);
}  // namespace

void Model::run_vision_encoder(const std::string& image_path) {
    if (!vision_encoder_) return;
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

    if (family_ == "lfm2_vl") {
        Lfm2VlImagePreprocessed prep = preprocess_lfm2_vl_image(image_path, config_);
        if (has_npu_vision_encoder() && vision_encode_via_npu(prep.pixel_values)) {
            return;
        }
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
            write_int_buffer_typed(pm_idx, prep.pixel_attention_mask.data(),
                                   prep.pixel_attention_mask.size());
        }
    } else if (family_ == "qwen3_5" || family_ == "qwen3_vl" || config_.model_type == Config::ModelType::QWEN) {
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
        if (has_npu_vision_encoder() && vision_encode_via_npu(prep.pixel_values)) {
            return;
        }
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
    vision_encoder_->graph->release_runtime_buffers();
    vision_encoder_->graph->release_all_weight_pages();
    unload_component_graph(*vision_encoder_);
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
    audio_encoder_->graph->release_runtime_buffers();
    audio_encoder_->graph->release_all_weight_pages();
    unload_component_graph(*audio_encoder_);
}

void Model::run_audio_encoder(const std::vector<float>& audio_features) {
    if (!audio_encoder_) return;
    if (has_npu_audio_encoder() && audio_encode_via_npu(audio_features)) {
        return;
    }
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

uint32_t Model::argmax_component_logits(Component& comp, size_t logit_row) {
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
    size_t row_off = row * vocab;
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

uint32_t Model::argmax_last_logits() {
    return argmax_component_logits(*decoder_);
}

bool Model::prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token) {
    last_prefill_cache_copy_ms_ = 0.0;
    last_prefill_padding_tokens_ = 0;
    last_prefill_scalar_tail_tokens_ = 0;
    if (tokens.empty() || !decoder_ || cache_total_seq_len_ != 0) {
        return false;
    }
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        out_token = argmax_last_logits();
        record_sampled_token(out_token);
        return true;
    }
    if (decode_route_ == DecodeRoute::DIRECT_DECODER_STEP) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            run_step(tokens[i], i, i + 1 == tokens.size());
        }
        cache_total_seq_len_ = tokens.size();
        out_token = argmax_last_logits();
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
            copy_cache_states(*decoder_prefill_, *decoder_, tokens.size() - 1);
            cache_total_seq_len_ = tokens.size() - 1;
            run_step(tokens.back(), cache_total_seq_len_, true);
            ++cache_total_seq_len_;
            out_token = argmax_last_logits();
            record_sampled_token(out_token);
            last_prefill_scalar_tail_tokens_ = 1;
            return true;
        }
        cache_total_seq_len_ += chunked.logical_tokens;
    }
    for (size_t i = chunked.logical_tokens; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_, i + 1 == tokens.size());
        ++cache_total_seq_len_;
    }
    last_prefill_scalar_tail_tokens_ = tokens.size() - chunked.logical_tokens;
    if (chunked.logical_tokens == tokens.size() && chunked.logical_tokens > 0 && decoder_prefill_) {
        out_token = argmax_component_logits(*decoder_prefill_, chunked.last_logit_row);
    } else {
        out_token = argmax_last_logits();
    }
    record_sampled_token(out_token);
    return true;
}

void Model::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& /*profile_file*/, bool prepare_decode) {
    last_prefill_cache_copy_ms_ = 0.0;
    last_prefill_padding_tokens_ = 0;
    last_prefill_scalar_tail_tokens_ = 0;
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        if (!context_tokens_.empty()) run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        return;
    }
    ChunkedPrefillResult chunked = run_chunked_prefill(tokens, cache_total_seq_len_, get_prefill_chunk_size(), prepare_decode);
    cache_total_seq_len_ += chunked.logical_tokens;
    for (size_t i = chunked.logical_tokens; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_, /*read_logits=*/false);
        ++cache_total_seq_len_;
    }
    last_prefill_scalar_tail_tokens_ = tokens.size() - chunked.logical_tokens;
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

void write_int_vector_buffer(std::vector<uint8_t>& buf, Precision prec, const std::vector<int64_t>& values) {
    const size_t elem = PrecisionTraits::size_of(prec);
    const size_t cap = elem ? buf.size() / elem : 0;
    const size_t n = std::min(cap, values.size());
    for (size_t i = 0; i < n; ++i) {
        int64_t v = values[i];
        switch (prec) {
            case Precision::FP32: reinterpret_cast<float*>(buf.data())[i] = static_cast<float>(v); break;
            case Precision::FP16: reinterpret_cast<__fp16*>(buf.data())[i] = static_cast<__fp16>(v); break;
            case Precision::INT8: reinterpret_cast<int8_t*>(buf.data())[i] = static_cast<int8_t>(v); break;
            default:
                if (elem == 8) reinterpret_cast<int64_t*>(buf.data())[i] = v;
                else if (elem == 4) reinterpret_cast<int32_t*>(buf.data())[i] = static_cast<int32_t>(v);
                break;
        }
    }
    if (n < cap) {
        std::memset(buf.data() + n * elem, 0, (cap - n) * elem);
    }
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
        for (const auto& path : image_paths) {
            if (family_ == "lfm2_vl") {
                Lfm2VlImagePreprocessed prep = preprocess_lfm2_vl_image(path, config_);
                if (has_npu_vision_encoder() && vision_encode_via_npu(prep.pixel_values)) {
                    continue;
                }
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
        copy_cache_states(*decoder_prefill_chunk_, *decoder_);
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
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        if (out_entropy) *out_entropy = 0.0f;
        uint32_t result = argmax_last_logits();
        record_sampled_token(result);
        return result;
    }
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
    const std::atomic<bool>* should_stop) {
    std::vector<uint32_t> emitted;
    if (decoder_prompt_tokens.empty() || max_tokens == 0) return emitted;

    Component* audio_enc = components_.count("audio_encoder") ? &components_.at("audio_encoder") : nullptr;
    Component* cross_kv = components_.count("decoder_cross_kv") ? &components_.at("decoder_cross_kv") : nullptr;
    Component* step = components_.count("decoder_step") ? &components_.at("decoder_step") : nullptr;
    if (!audio_enc || !cross_kv || !step) {
        CACTUS_LOG_ERROR("model", "Whisper bundle missing audio_encoder, decoder_cross_kv, or decoder_step component");
        return emitted;
    }
    if (!bind_runtime_buffers(*audio_enc)) return emitted;
    if (!bind_runtime_buffers(*cross_kv)) return emitted;
    if (!bind_runtime_buffers(*step)) return emitted;

    reset_component_cache_states(*step);
    cache_total_seq_len_ = 0;
    token_history_.clear();

    const int feat_idx = input_index(*audio_enc, "input_features");
    if (feat_idx < 0) {
        CACTUS_LOG_ERROR("model", "Whisper audio_encoder has no input_features input");
        return emitted;
    }
    auto& feat_buf = audio_enc->input_buffers[feat_idx];
    const size_t feat_node = static_cast<size_t>(audio_enc->runtime_input_node_ids[feat_idx]);
    const auto& feat_desc = audio_enc->graph->get_output_buffer(feat_node);
    write_typed_buffer(
        feat_buf,
        feat_desc.precision,
        audio_features.data(),
        audio_features.size() * sizeof(float),
        Precision::FP32);
    audio_enc->graph->execute();

    const int hidden_idx = output_index(*audio_enc, "encoder_hidden_states");
    if (hidden_idx < 0) {
        CACTUS_LOG_ERROR("model", "Whisper audio_encoder has no encoder_hidden_states output");
        return emitted;
    }
    const size_t hidden_node = static_cast<size_t>(audio_enc->output_node_ids[hidden_idx]);
    const auto& hidden_desc = audio_enc->graph->get_output_buffer(hidden_node);
    const void* hidden_ptr = audio_enc->graph->get_output(hidden_node);
    if (hidden_ptr == nullptr || hidden_desc.byte_size == 0) {
        CACTUS_LOG_ERROR("model", "Whisper encoder_hidden_states output is empty");
        return emitted;
    }

    const int cross_hidden_idx = input_index(*cross_kv, "encoder_hidden_states");
    if (cross_hidden_idx < 0) {
        CACTUS_LOG_ERROR("model", "Whisper decoder_cross_kv missing encoder_hidden_states input");
        return emitted;
    }
    auto& cross_hidden_buf = cross_kv->input_buffers[cross_hidden_idx];
    const size_t cross_hidden_node = static_cast<size_t>(cross_kv->runtime_input_node_ids[cross_hidden_idx]);
    const auto& cross_hidden_desc = cross_kv->graph->get_output_buffer(cross_hidden_node);
    write_typed_buffer(
        cross_hidden_buf,
        cross_hidden_desc.precision,
        hidden_ptr,
        hidden_desc.byte_size,
        hidden_desc.precision);
    cross_kv->graph->execute();

    for (size_t i = 0; i < cross_kv->output_node_ids.size() && i < cross_kv->logical_outputs.size(); ++i) {
        const std::string& name = cross_kv->logical_outputs[i];
        int idx = input_index(*step, name);
        if (idx < 0) continue;
        const size_t src_node = static_cast<size_t>(cross_kv->output_node_ids[i]);
        const auto& src_desc = cross_kv->graph->get_output_buffer(src_node);
        const void* src_ptr = cross_kv->graph->get_output(src_node);
        if (src_ptr == nullptr || src_desc.byte_size == 0) continue;
        auto& dst_buf = step->input_buffers[idx];
        const size_t dst_node = static_cast<size_t>(step->runtime_input_node_ids[idx]);
        const auto& dst_desc = step->graph->get_output_buffer(dst_node);
        write_typed_buffer(dst_buf, dst_desc.precision, src_ptr, src_desc.byte_size, src_desc.precision);
    }

    const int ids_idx = input_index(*step, "decoder_input_ids");
    const int pos_idx = input_index(*step, "position_ids");
    if (ids_idx < 0 || pos_idx < 0) {
        CACTUS_LOG_ERROR("model", "Whisper decoder_step missing decoder_input_ids or position_ids input");
        return emitted;
    }

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
            write_int_input(*step, "decoder_input_ids", static_cast<int64_t>(tokens[pos]));
            write_int_input(*step, "position_ids", static_cast<int64_t>(pos));
            step->graph->execute();
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

    std::vector<__fp16> npu_hidden_storage;
    bool used_npu = false;
    size_t npu_hidden_T = 0;
    if (has_npu_audio_encoder()) {
        const std::vector<int> in_shape = npu_audio_encoder_->get_input_shape();
        const std::vector<int> out_shape = npu_audio_encoder_->get_output_shape();
        if (in_shape.size() >= 3 && out_shape.size() >= 3 &&
            in_shape[1] > 0 && in_shape[2] > 0 && out_shape[1] > 0 && out_shape[2] > 0 &&
            static_cast<size_t>(in_shape[2]) == expected_mels) {
            const size_t window_frames = static_cast<size_t>(in_shape[1]);
            const size_t window_hidden = static_cast<size_t>(out_shape[1]);
            const size_t hidden_dim_npu = static_cast<size_t>(out_shape[2]);
            const size_t chunk_input_elems = window_frames * expected_mels;
            const size_t chunk_output_elems = window_hidden * hidden_dim_npu;
            const size_t num_chunks = (copy_frames + window_frames - 1) / window_frames;
            const size_t total_hidden_T = num_chunks * window_hidden;
            npu_hidden_storage.assign(total_hidden_T * hidden_dim_npu, __fp16(0));
            std::vector<__fp16> input_fp16(chunk_input_elems);
            bool all_ok = num_chunks > 0;
            for (size_t c = 0; c < num_chunks && all_ok; ++c) {
                const size_t frame_start = c * window_frames;
                const size_t frame_end = std::min(frame_start + window_frames, copy_frames);
                std::fill(input_fp16.begin(), input_fp16.end(), __fp16(0));
                for (size_t t = frame_start; t < frame_end; ++t) {
                    const size_t local = (t - frame_start) * expected_mels;
                    const size_t src = t * expected_mels;
                    for (size_t m = 0; m < expected_mels; ++m) {
                        input_fp16[local + m] = static_cast<__fp16>(transposed[src + m]);
                    }
                }
                __fp16* out_ptr = npu_hidden_storage.data() + c * chunk_output_elems;
                size_t written = npu_audio_encoder_->encode(
                    input_fp16.data(), out_ptr, in_shape, "x", "encoded");
                if (written == 0) { all_ok = false; break; }
            }
            if (all_ok) {
                used_npu = true;
                const size_t valid_input = copy_frames;
                npu_hidden_T = (valid_input * window_hidden + window_frames - 1) / window_frames;
                if (npu_hidden_T > total_hidden_T) npu_hidden_T = total_hidden_T;
                CACTUS_LOG_INFO("model", "Parakeet audio encoder ran on NPU ("
                                << num_chunks << " chunks, " << npu_hidden_T << " valid hidden frames)");
            } else {
                CACTUS_LOG_WARN("model", "NPU audio encoder chunk failed; falling back to CPU graph");
            }
        }
    }
    if (!used_npu) {
        audio_enc->graph->execute();
    }

    int hidden_idx = output_index(*audio_enc, "encoder_hidden_states");
    if (hidden_idx < 0) {
        CACTUS_LOG_ERROR("model", "audio_encoder has no encoder_hidden_states output");
        return emitted;
    }
    size_t hidden_node = static_cast<size_t>(audio_enc->output_node_ids[hidden_idx]);
    const auto& hidden_desc = audio_enc->graph->get_output_buffer(hidden_node);
    const uint8_t* hidden_ptr;
    if (used_npu) {
        hidden_ptr = reinterpret_cast<const uint8_t*>(npu_hidden_storage.data());
    } else {
        hidden_ptr = static_cast<const uint8_t*>(audio_enc->graph->get_output(hidden_node));
    }
    if (hidden_desc.shape.size() < 3 || hidden_ptr == nullptr) {
        CACTUS_LOG_ERROR("model", "encoder_hidden_states must be 3D [B, T, D]");
        return emitted;
    }
    const size_t T = used_npu ? npu_hidden_T : hidden_desc.shape[1];
    const size_t D = hidden_desc.shape[2];
    const Precision hidden_precision = used_npu ? Precision::FP16 : hidden_desc.precision;
    const size_t hidden_elem = PrecisionTraits::size_of(hidden_precision);
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
    const std::string output_name = vision_encoder_->logical_outputs[0];

    run_vision_encoder(image_path);

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
    // run_audio_encoder_messages unloads the graph; restore so subsequent
    // transcribe_* paths (which assume the encoder stays loaded) work.
    load_component_graph(*audio_encoder_);
    return embedding;
}

void Model::reset_cache() {
    cache_total_seq_len_ = 0;
    last_logit_position_ = 0;
    context_tokens_.clear();
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

    // Embedding encoders (nomic / XLM-R) wrap the sequence with BOS/EOS, matching
    // the reference tokenizer's add_special_tokens behavior.
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

    if (normalize) {
        double norm = 0.0;
        for (float v : result) norm += static_cast<double>(v) * v;
        float inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1e-12));
        for (float& v : result) v *= inv;
    }

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


double Model::score_tokens_window_logprob(const std::vector<uint32_t>& /*tokens*/, size_t /*start*/,
                                            size_t /*end*/, size_t /*context*/, size_t* tokens_scored) {
    if (tokens_scored) *tokens_scored = 0;
    return 0.0;
}

}
}
