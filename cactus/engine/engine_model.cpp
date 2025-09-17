#include "engine.h"
#include "../graph/graph.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <set>

namespace cactus {
namespace engine {


Model::Model() 
    : tokenizer_(nullptr), 
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f) {
    weight_nodes_.layers.resize(config_.num_layers);
}

Model::Model(const Config& config) 
    : config_(config),
      tokenizer_(nullptr),
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f) {
    weight_nodes_.layers.resize(config.num_layers);
}

Model::~Model() {
    if (graph_handle_) {
        delete static_cast<CactusGraph*>(graph_handle_);
    }
}

bool Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt) {
    if (initialized_) {
        return true;
    }
    
    model_folder_path_ = model_folder;
    std::string config_path = model_folder + "/config.txt";
    
    if (!config_.from_json(config_path)) {
        return false;
    }
    
    std::string vocab_file = model_folder + "/vocab.txt";
    std::string merges_file = model_folder + "/merges.txt";
    std::string tokenizer_config_file = model_folder + "/tokenizer_config.txt";
    
    std::ifstream merges_check(merges_file);
    bool has_merges = false;
    if (merges_check.is_open()) {
        std::string line;
        int line_count = 0;
        while (std::getline(merges_check, line) && line_count < 10) {
            if (!line.empty() && line[0] != '#') {
                has_merges = true;
                break;
            }
            line_count++;
        }
        merges_check.close();
    }
    
    if (has_merges) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    } else {
        tokenizer_ = std::make_unique<SPTokenizer>();
    }
    
    if (!tokenizer_->load_vocabulary_with_config(vocab_file, merges_file, tokenizer_config_file)) {
        return false;
    }
    
    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    
    embedding_file_path_ = model_folder + "/token_embeddings.weights";
    weight_nodes_.layers.resize(config_.num_layers);
    
    load_weights_to_graph(gb);
    
    attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));
    
    Precision cache_precision;
    std::string precision_name;
    switch (config_.precision) {
        case Config::Precision::INT8:
            cache_precision = Precision::INT8;
            precision_name = "INT8";
            break;
        case Config::Precision::FP16:
            cache_precision = Precision::FP16;
            precision_name = "FP16";
            break;
        case Config::Precision::FP32:
            cache_precision = Precision::FP32;
            precision_name = "FP32";
            break;
    }
    kv_cache_.init(config_.num_layers, context_size, config_.attention_kv_heads, config_.attention_head_dim, cache_precision);
    
    size_t window_size = std::min(context_size, size_t(1024));
    size_t sink_size = 4;
    const char* env_window = std::getenv("CACTUS_KV_WINDOW_SIZE");
    const char* env_sink = std::getenv("CACTUS_KV_SINK_SIZE");
    if (env_window) {
        window_size = std::stoul(env_window);
    }
    if (env_sink) {
        sink_size = std::stoul(env_sink);
    }
    kv_cache_.set_window_size(window_size, sink_size);
    cache_k_output_nodes_.resize(config_.num_layers);
    cache_v_output_nodes_.resize(config_.num_layers);
    
    initialized_ = true;
    
    std::string warmup_text = system_prompt.empty() ? "Henry" : system_prompt;
    auto warmup_tokens = tokenizer_->encode(warmup_text);
    forward(warmup_tokens);
    kv_cache_.reset();
    return true;
}


void Model::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    
    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;  
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
    }
    
    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        layer.attn_q_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_norm.weights");
        layer.attn_k_norm_weight = gb->mmap_weights(layer_prefix + "attn_k_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
    }
}



size_t Model::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx, 
                             ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    
    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    
    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    q_proj = gb->reshape(q_proj, {batch_seq * num_heads, head_dim});
    q_proj = gb->rms_norm(q_proj, layer.attn_q_norm_weight, config_.layer_norm_eps);
    q_proj = gb->reshape(q_proj, {batch_seq, num_heads * head_dim});
    
    size_t num_kv_heads = config_.attention_kv_heads;
    k_proj = gb->reshape(k_proj, {batch_seq * num_kv_heads, head_dim});
    k_proj = gb->rms_norm(k_proj, layer.attn_k_norm_weight, config_.layer_norm_eps);
    k_proj = gb->reshape(k_proj, {batch_seq, num_kv_heads * head_dim});
    
    size_t seq_len = batch_seq;
    
    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    
    if (config_.rope_theta > 0) {
        q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, position_offset);
        k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, position_offset);
    }
    
    size_t final_k = k_proj_4d;
    size_t final_v = v_proj_4d;
    
    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);
        
        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            size_t cache_k_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            size_t cache_v_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            
            gb->set_input(cache_k_node, k_view.ptr1, kv_cache_.precision);
            gb->set_input(cache_v_node, v_view.ptr1, kv_cache_.precision);
            
            final_k = gb->concat(cache_k_node, k_proj_4d, 1);
            final_v = gb->concat(cache_v_node, v_proj_4d, 1);
        } else {
            size_t cache_k_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            size_t cache_v_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            
            gb->set_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
            
            final_k = gb->concat(cache_k_node, k_proj_4d, 1);
            final_v = gb->concat(cache_v_node, v_proj_4d, 1);
        }
    }
    
    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    }
    
    
    auto attn_output_4d = gb->attention(q_proj_4d, final_k, final_v, attention_scale_, position_offset);
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}

size_t Model::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx, 
                       ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}


size_t Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, 
                                     ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    auto after_attention = gb->add(hidden, attn_output);
    auto normalized_after_attention = gb->rms_norm(after_attention, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_attention, layer_idx, backend);
    return gb->add(after_attention, mlp_output);
}


size_t Model::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    
    auto seq_len = static_cast<size_t>(tokens.size());
    
    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    
    auto backend = config_.default_backend == Config::Backend::CPU 
        ? ComputeBackend::CPU 
        : ComputeBackend::NPU;
    
    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);
    
    static std::set<uint32_t> skip_layers = {}; 
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }
    
    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    
    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);
    
    return final_hidden;
}


uint32_t Model::generate(const std::vector<uint32_t>& tokens, float temperature, float top_p, 
                        size_t top_k, const std::string& profile_file) {
    auto final_hidden = forward(tokens, true);
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU 
        ? ComputeBackend::CPU 
        : ComputeBackend::NPU;
    
    auto logits_node_id = gb->matmul(final_hidden, weight_nodes_.output_weight, true, backend);
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k);
    
    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }
    
    update_kv_cache(gb, tokens.size());
    
    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

void Model::update_kv_cache(CactusGraph* gb, size_t seq_len) {
    kv_cache_.update_from_graph(gb, cache_k_output_nodes_, cache_v_output_nodes_, 
                               seq_len, config_.num_layers, config_.attention_kv_heads, 
                               config_.attention_head_dim);
}


std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& tokens, bool pooled) {
    auto final_hidden = forward(tokens);
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto* output_ptr = gb->get_output(final_hidden);
    const auto& output_buffer = gb->get_output_buffer(final_hidden);
    
    std::vector<float> embeddings;
    
    if (pooled) {
        auto pooled_hidden = gb->mean(final_hidden, 0);
        gb->execute();
        
        auto* pooled_ptr = gb->get_output(pooled_hidden);
        const auto& pooled_buffer = gb->get_output_buffer(pooled_hidden);
        
        size_t hidden_dim = pooled_buffer.total_size;
        embeddings.resize(hidden_dim);
        
        if (pooled_buffer.precision == Precision::FP32) {
            float* pooled_data = static_cast<float*>(pooled_ptr);
            std::copy(pooled_data, pooled_data + hidden_dim, embeddings.begin());
        } else if (pooled_buffer.precision == Precision::FP16) {
            __fp16* pooled_data = static_cast<__fp16*>(pooled_ptr);
            Quantization::fp16_to_fp32(pooled_data, embeddings.data(), hidden_dim);
        } else if (pooled_buffer.precision == Precision::INT8) {
            int8_t* pooled_data = static_cast<int8_t*>(pooled_ptr);
            float scale = pooled_buffer.quantization_scale;
            Quantization::int8_to_fp32(pooled_data, embeddings.data(), hidden_dim, scale);
        }
    } else {
        gb->execute();
        
        size_t total_size = output_buffer.total_size;
        embeddings.resize(total_size);
        
        if (output_buffer.precision == Precision::FP32) {
            float* hidden_states = static_cast<float*>(output_ptr);
            std::copy(hidden_states, hidden_states + total_size, embeddings.begin());
        } else if (output_buffer.precision == Precision::FP16) {
            __fp16* hidden_states = static_cast<__fp16*>(output_ptr);
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = static_cast<float>(hidden_states[i]);
            }
        } else if (output_buffer.precision == Precision::INT8) {
            int8_t* hidden_states = static_cast<int8_t*>(output_ptr);
            float scale = output_buffer.quantization_scale;
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = hidden_states[i] * scale;
            }
        }
    }
    
    kv_cache_.reset();
    
    return embeddings;
}


bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
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
        
        if (key == "vocab_size") vocab_size = std::stoul(value);
        else if (key == "bos_token_id") bos_token_id = std::stoul(value);
        else if (key == "eos_token_id") eos_token_id = std::stoul(value);
        else if (key == "num_layers") num_layers = std::stoul(value);
        else if (key == "hidden_dim") hidden_dim = std::stoul(value);
        else if (key == "ffn_intermediate_dim") ffn_intermediate_dim = std::stoul(value);
        else if (key == "attention_heads") attention_heads = std::stoul(value);
        else if (key == "attention_kv_heads") attention_kv_heads = std::stoul(value);
        else if (key == "attention_head_dim") attention_head_dim = std::stoul(value);
        else if (key == "layer_norm_eps") layer_norm_eps = std::stof(value);
        else if (key == "rope_theta") rope_theta = std::stof(value);
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "precision") {
            if (value == "INT8") precision = Precision::INT8;
            else if (value == "FP16") precision = Precision::FP16;
            else precision = Precision::FP32;
        }
    }
    
    return true;
}

std::string Config::to_json() const {
    return "{}";
}

}
}