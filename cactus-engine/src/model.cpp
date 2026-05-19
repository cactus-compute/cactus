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
    std::string encoder_name;
    std::string decoder_name;
    std::unordered_set<std::string> required_components;
    bool has_chunked_prefill = components_.count("lm_encoder_step")
        && components_.count("decoder_media_step")
        && components_.count("lm_encoder_text_chunk")
        && components_.count("decoder_prefill_chunk");
    if (has_chunked_prefill) {
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
    } else {
        CACTUS_LOG_ERROR("model", "Bundle missing lm_encoder_step+decoder_step or text_lm_encoder+decoder components");
        return false;
    }
    if (!load_components(required_components)) return false;
    if (!encoder_name.empty()) encoder_ = &components_.at(encoder_name);
    decoder_ = &components_.at(decoder_name);
    if (components_.count("decoder_prefill_chunk") && components_.at("decoder_prefill_chunk").graph) {
        decoder_prefill_ = &components_.at("decoder_prefill_chunk");
    }
    if (components_.count("lm_encoder_text_chunk") && components_.at("lm_encoder_text_chunk").graph) {
        prefill_encoder_ = &components_.at("lm_encoder_text_chunk");
    }
    if (encoder_ && !bind_runtime_buffers(*encoder_)) return false;
    if (prefill_encoder_ && !bind_runtime_buffers(*prefill_encoder_)) return false;
    if (!bind_runtime_buffers(*decoder_)) return false;
    if (decoder_prefill_ && !bind_runtime_buffers(*decoder_prefill_)) return false;

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
        if (c.count("cache_state_node_ids")) {
            for (const auto& sv : c.at("cache_state_node_ids").get<picojson::array>()) {
                const auto& s = sv.get<picojson::object>();
                CacheStateBinding cs;
                if (s.count("layer_key")) cs.layer_key = s.at("layer_key").get<std::string>();
                if (s.count("key")) cs.key_node_id = static_cast<int>(s.at("key").get<int64_t>());
                if (s.count("value")) cs.value_node_id = static_cast<int>(s.at("value").get<int64_t>());
                comp.cache_states.push_back(std::move(cs));
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

void Model::copy_cache_states(const Component& source, Component& target) {
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
            if (src_desc.byte_size == dst_desc.byte_size) {
                std::memcpy(dst_ptr, src_ptr, src_desc.byte_size);
                continue;
            }
            if (src_desc.precision == Precision::INT8) {
                auto* src_meta = static_cast<uint64_t*>(src_ptr);
                auto* dst_meta = static_cast<uint64_t*>(dst_ptr);
                const size_t src_current = static_cast<size_t>(src_meta[0]);
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
                const size_t dst_current = std::min(src_current, dst_max);
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
                if (src_current <= dst_max) {
                    copy_rows(0, 0, src_current);
                } else {
                    const size_t copied_sink = std::min(sink, dst_max);
                    const size_t tail_rows = dst_max - copied_sink;
                    copy_rows(0, 0, copied_sink);
                    if (tail_rows > 0) copy_rows(copied_sink, src_current - tail_rows, tail_rows);
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
            const size_t dst_current = std::min(src_current, dst_max);
            dst_meta[0] = dst_current;
            dst_meta[1] = dst_max;
            dst_meta[2] = kv_heads;
            dst_meta[3] = head_dim;
            dst_meta[4] = std::min(sink, dst_current);
            std::memset(static_cast<char*>(dst_ptr) + 64, 0, dst_desc.byte_size - 64);
            const auto* src_rows = static_cast<const char*>(src_ptr) + 64;
            auto* dst_rows = static_cast<char*>(dst_ptr) + 64;
            if (src_current <= dst_max) {
                std::memcpy(dst_rows, src_rows, src_current * row_bytes);
            } else {
                const size_t copied_sink = std::min(sink, dst_max);
                const size_t tail_rows = dst_max - copied_sink;
                if (copied_sink > 0) {
                    std::memcpy(dst_rows, src_rows, copied_sink * row_bytes);
                }
                if (tail_rows > 0) {
                    std::memcpy(
                        dst_rows + copied_sink * row_bytes,
                        src_rows + (src_current - tail_rows) * row_bytes,
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
            if (!desc.get_data()) continue;
            void* ptr = comp.graph->get_output(static_cast<size_t>(node_id));
            if (!ptr) continue;
            auto* metadata = static_cast<uint64_t*>(ptr);
            metadata[0] = 0;
            metadata[1] = 0;
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

size_t Model::run_chunked_prefill(const std::vector<uint32_t>& tokens, size_t start_position, size_t chunk_size, bool prepare_decode) {
    if (decode_route_ != DecodeRoute::CACHED_STEP || !encoder_ || !decoder_ || !decoder_prefill_) return 0;
    if (start_position != 0) return 0;
    if (!cache_states_compatible(*decoder_prefill_, *decoder_)) return 0;
    size_t component_tokens = component_chunk_tokens(*decoder_prefill_, "inputs_embeds");
    if (component_tokens <= 1) return 0;
    size_t effective_chunk = chunk_size > 0 ? std::min(chunk_size, component_tokens) : component_tokens;
    if (effective_chunk != component_tokens) effective_chunk = component_tokens;
    if (tokens.size() < effective_chunk) return 0;

    size_t encoder_chunk = 0;
    if (prefill_encoder_ && input_index(*prefill_encoder_, "input_ids") >= 0 && input_index(*prefill_encoder_, "position_ids") >= 0) {
        encoder_chunk = component_chunk_tokens(*prefill_encoder_, "input_ids");
        if (encoder_chunk == 0 || effective_chunk % encoder_chunk != 0) {
            encoder_chunk = 0;
        }
    }

    size_t processed = 0;
    while (processed + effective_chunk <= tokens.size()) {
        for (size_t i = 0; i < decoder_prefill_->input_buffers.size(); ++i) {
            std::fill(decoder_prefill_->input_buffers[i].begin(), decoder_prefill_->input_buffers[i].end(), 0);
        }
        if (encoder_chunk > 0) {
            for (size_t chunk_offset = 0; chunk_offset < effective_chunk; chunk_offset += encoder_chunk) {
                for (size_t i = 0; i < prefill_encoder_->input_buffers.size(); ++i) {
                    std::fill(prefill_encoder_->input_buffers[i].begin(), prefill_encoder_->input_buffers[i].end(), 0);
                }
                for (size_t i = 0; i < encoder_chunk; ++i) {
                    write_int_input_at(*prefill_encoder_, "input_ids", i, static_cast<int64_t>(tokens[processed + chunk_offset + i]));
                    write_int_input_at(*prefill_encoder_, "position_ids", i, static_cast<int64_t>(start_position + processed + chunk_offset + i));
                }
                prefill_encoder_->graph->execute();
                copy_component_outputs_to_chunk_inputs_range(*prefill_encoder_, *decoder_prefill_, chunk_offset);
            }
        } else {
            for (size_t i = 0; i < effective_chunk; ++i) {
                run_encoder_step(tokens[processed + i], start_position + processed + i);
                copy_component_outputs_to_chunk_inputs(*encoder_, *decoder_prefill_, i);
            }
        }
        decoder_prefill_->graph->execute();
        processed += effective_chunk;
    }
    if (processed > 0 && prepare_decode) {
        for (size_t i = 0; i < decoder_->input_buffers.size(); ++i) {
            std::fill(decoder_->input_buffers[i].begin(), decoder_->input_buffers[i].end(), 0);
        }
        copy_cache_states(*decoder_prefill_, *decoder_);
    }
    return processed;
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

uint32_t Model::argmax_last_logits() {
    size_t out_node = static_cast<size_t>(decoder_->output_node_ids.empty() ? 0 : decoder_->output_node_ids[0]);
    const auto& desc = decoder_->graph->get_output_buffer(out_node);
    void* ptr = decoder_->graph->get_output(out_node);
    size_t vocab = desc.shape.empty() ? 0 : desc.shape.back();
    size_t seq = desc.shape.size() >= 2 ? desc.shape[desc.shape.size() - 2] : 1;
    size_t row = decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT ? std::min(last_logit_position_, seq > 0 ? seq - 1 : 0) : (seq > 0 ? seq - 1 : 0);
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

void Model::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& /*profile_file*/, bool prepare_decode) {
    if (decode_route_ == DecodeRoute::FULL_CONTEXT_TEXT) {
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        if (!context_tokens_.empty()) run_full_context_text();
        cache_total_seq_len_ = context_tokens_.size();
        return;
    }
    size_t processed = run_chunked_prefill(tokens, cache_total_seq_len_, get_prefill_chunk_size(), prepare_decode);
    cache_total_seq_len_ += processed;
    for (size_t i = processed; i < tokens.size(); ++i) {
        run_step(tokens[i], cache_total_seq_len_, /*read_logits=*/false);
        ++cache_total_seq_len_;
    }
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
    last_logit_position_ = 0;
    context_tokens_.clear();
    token_history_.clear();
    if (decoder_) reset_component_cache_states(*decoder_);
    if (decoder_prefill_) reset_component_cache_states(*decoder_prefill_);
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
