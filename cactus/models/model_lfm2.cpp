#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <set>
#include <limits>
#include <iostream>
#include <algorithm>
namespace cactus {
namespace engine {
LFM2Model::LFM2Model() : Model() {
    weight_nodes_.layers.resize(config_.num_layers);
    conv_cache_bx_nodes_.assign(config_.num_layers, 0);
}
LFM2Model::LFM2Model(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config_.num_layers);
    conv_cache_bx_nodes_.assign(config_.num_layers, 0);
}
void LFM2Model::post_init() {
    if (config_.conv_L_cache > 0) {
        Precision cache_precision;
        switch (config_.precision) {
            case Config::Precision::INT8:
                cache_precision = Precision::INT8;
                break;
            case Config::Precision::FP16:
                cache_precision = Precision::FP16;
                break;
            case Config::Precision::FP32:
                cache_precision = Precision::FP32;
                break;
        }
        size_t conv_window = config_.conv_L_cache > 0 ? config_.conv_L_cache - 1 : 0;
        conv_cache_.init(config_.num_layers, config_.hidden_dim, conv_window, cache_precision);
        
    }
    last_forward_used_cache_ = false;
    
}
void LFM2Model::reset_cache() {
    Model::reset_cache(); 
    
    if (conv_cache_.window_size > 0) {
        conv_cache_.reset();
        
    }
}
bool LFM2Model::is_cache_empty() const {
    return kv_cache_.is_empty();
}
bool LFM2Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt, bool do_warmup) {
    if (!Model::init(model_folder, context_size, system_prompt, do_warmup)) {
        return false;
    }
    
    if (weight_nodes_.layers.size() != config_.num_layers) {
        weight_nodes_.layers.resize(config_.num_layers);
        
    }
    conv_cache_bx_nodes_.assign(config_.num_layers, 0);
    
    return true;
}
bool LFM2Model::init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
                     const std::string& system_prompt, bool do_warmup) {
    if (!Model::init(external_graph, model_folder, context_size, system_prompt, do_warmup)) {
        return false;
    }
    
    if (weight_nodes_.layers.size() != config_.num_layers) {
        weight_nodes_.layers.resize(config_.num_layers);
        
    }
    conv_cache_bx_nodes_.assign(config_.num_layers, 0);
    
    return true;
}
void LFM2Model::load_weights_to_graph(CactusGraph* gb) {
    if (weight_nodes_.layers.size() != config_.num_layers) {
        weight_nodes_.layers.resize(config_.num_layers);
    }
    if (conv_cache_bx_nodes_.size() != config_.num_layers) {
        conv_cache_bx_nodes_.assign(config_.num_layers, 0);
    }
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    
    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
        
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
        
    }
    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer_entry = weight_nodes_.layers[i];
        auto& layer = layer_entry.weights;
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        
        bool is_conv_layer = false;
        if (i < config_.layer_types.size()) {
            std::string layer_type = config_.layer_types[i];
            is_conv_layer = (layer_type == "conv" || layer_type == "CONV");
            
        }
        if (is_conv_layer) {
            layer_entry.type = WeightNodeIDs::LayerType::CONV;
            layer.conv_in_proj_weight = gb->mmap_weights(layer_prefix + "conv_in_proj.weights");
            layer.conv_out_proj_weight = gb->mmap_weights(layer_prefix + "conv_out_proj.weights");
            layer.conv_depthwise_weight = gb->mmap_weights(layer_prefix + "conv_depthwise.weights");
            
        } else {
            layer_entry.type = WeightNodeIDs::LayerType::ATTENTION;
            layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
            layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
            layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
            layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
            layer.attn_q_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_norm.weights");
            layer.attn_k_norm_weight = gb->mmap_weights(layer_prefix + "attn_k_norm.weights");
            
        }
        layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
        
    }
    
}
size_t LFM2Model::build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                               ComputeBackend backend, bool use_cache) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer       = layer_entry.weights;
    size_t in_proj = gb->matmul(input, layer.conv_in_proj_weight, true, backend);
    const auto& in_proj_buf = gb->get_output_buffer(in_proj);
    if (in_proj_buf.shape.size() != 2 || (in_proj_buf.shape[1] % 3) != 0) {
        throw std::runtime_error("Conv in_proj output must be [L, 3*C]");
    }
    
    const size_t L = in_proj_buf.shape[0];
    const size_t C = in_proj_buf.shape[1] / 3;
    
    size_t triplet = gb->reshape(in_proj, {L, static_cast<size_t>(3), C});
    size_t B = gb->slice(triplet, 1, 0, 1);
    size_t Cg = gb->slice(triplet, 1, 1, 1);
    size_t X = gb->slice(triplet, 1, 2, 1);
    
    B  = gb->reshape(B,  {L, C});
    
    Cg = gb->reshape(Cg, {L, C});
    
    X  = gb->reshape(X,  {L, C});
    size_t Bx = gb->multiply(B, X);
    if (use_cache) {
        conv_cache_bx_nodes_[layer_idx] = Bx;
        
    } else {
        conv_cache_bx_nodes_[layer_idx] = 0;
        
    }
    const auto& wbuf = gb->get_output_buffer(layer.conv_depthwise_weight);
    size_t K = wbuf.shape.back(); 
    
    size_t conv_w = layer.conv_depthwise_weight;
    if (wbuf.shape.size() == 2) { 
        K = wbuf.shape[1];
        auto conv_w_quantization_scale = gb->get_output_buffer(conv_w).quantization_scale;
        conv_w = gb->reshape(conv_w, {wbuf.shape[0], static_cast<size_t>(1), K});
        gb->set_quantization_scale(conv_w, conv_w_quantization_scale);
        
    } else if (wbuf.shape.size() != 3){
        throw std::runtime_error("Unexpected depthwise weight rank");
    }
    K = wbuf.shape.back();
    
    size_t conv_input_lc = Bx;
    if (use_cache && conv_cache_.window_size > 0) {
        auto view = conv_cache_.get_window(layer_idx);
        std::vector<size_t> segments;
        if (view.len2 > 0) {
            size_t left_node = gb->input({view.len2, C}, conv_cache_.precision);
            gb->set_input(left_node, view.ptr2, conv_cache_.precision);
            segments.push_back(left_node);
            
        }
        if (view.len1 > 0) {
            size_t right_node = gb->input({view.len1, C}, conv_cache_.precision);
            gb->set_input(right_node, view.ptr1, conv_cache_.precision);
            segments.push_back(right_node);
            
        }
        if (!segments.empty()) {
            conv_input_lc = segments[0];
            for (size_t idx = 1; idx < segments.size(); ++idx) {
                conv_input_lc = gb->concat(conv_input_lc, segments[idx], 0);
                
            }
            conv_input_lc = gb->concat(conv_input_lc, Bx, 0);
            
        }
    }
    
    const auto& conv_input_buf = gb->get_output_buffer(conv_input_lc);
    size_t total_len = conv_input_buf.shape[0];
    
    size_t x_nlc = gb->reshape(conv_input_lc, {static_cast<size_t>(1), total_len, C});
    const size_t dilation = 1;
    
    size_t y_nlc = gb->conv1d_causal(x_nlc, conv_w, K, dilation); 
    size_t start = total_len > L ? total_len - L : 0;
    size_t y_slice = gb->slice(y_nlc, 1, start, L);
    
    size_t y_lc = gb->reshape(y_slice, {L, C});
    size_t gated = gb->multiply(Cg, y_lc); 
    size_t projected = gb->matmul(gated, layer.conv_out_proj_weight, true, backend); 
    return projected;
}

size_t LFM2Model::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    auto q_proj_linear = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj_linear = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj_linear = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    const auto& q_shape = gb->get_output_buffer(q_proj_linear).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    
    auto q_proj_reshaped = gb->reshape(q_proj_linear, {batch_seq * num_heads, head_dim});
    
    auto q_proj_norm = gb->rms_norm(q_proj_reshaped, layer.attn_q_norm_weight, config_.layer_norm_eps);
    
    auto q_proj = gb->reshape(q_proj_norm, {batch_seq, num_heads * head_dim});
    size_t num_kv_heads = config_.attention_kv_heads;
    auto k_proj_reshaped = gb->reshape(k_proj_linear, {batch_seq * num_kv_heads, head_dim});
    
    auto k_proj_norm = gb->rms_norm(k_proj_reshaped, layer.attn_k_norm_weight, config_.layer_norm_eps);
    
    auto k_proj = gb->reshape(k_proj_norm, {batch_seq, num_kv_heads * head_dim});
    size_t seq_len = batch_seq;

    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});    
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});    
    auto v_proj_4d = gb->reshape(v_proj_linear, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});

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
    auto projected = gb->matmul(attn_output, layer.attn_output_weight, true, backend);
    return projected;
}

size_t LFM2Model::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    auto gate = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    auto up = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    auto activated = gb->multiply(gb->silu(gate), up);
    
    auto down = gb->matmul(activated, layer.ffn_down_weight, true, backend);
    
    return down;
}

size_t LFM2Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    size_t block_output;
    if (layer_entry.type == WeightNodeIDs::LayerType::CONV) {
        block_output = build_conv1d(gb, normalized_input, layer_idx, backend, use_cache);
    } else {
        if (layer_idx < conv_cache_bx_nodes_.size()) {
            conv_cache_bx_nodes_[layer_idx] = 0;
        }
        block_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    }
    
    auto after_block = gb->add(hidden, block_output);
    auto normalized_after_block = gb->rms_norm(after_block, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_block, layer_idx, backend);
    auto block_result = gb->add(after_block, mlp_output);
    return block_result;
}

size_t LFM2Model::forward(CactusGraph* gb, size_t input_embeddings, size_t seq_len,
                         ComputeBackend backend, bool use_cache) {
    if (seq_len == 0) {
        throw std::runtime_error("Sequence length must be greater than zero");
    }
    
    if (conv_cache_bx_nodes_.size() != config_.num_layers) {
        conv_cache_bx_nodes_.assign(config_.num_layers, 0);
        
    }
    std::fill(conv_cache_bx_nodes_.begin(), conv_cache_bx_nodes_.end(), 0);
    
    last_forward_used_cache_ = use_cache;
    
    if (!use_cache && conv_cache_.window_size > 0) {
        conv_cache_.reset();
        
    }
    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    
    size_t hidden = input_embeddings;
    
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
        
    }
    
    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    
    return final_hidden;
}
size_t LFM2Model::forward(CactusGraph* gb, const std::vector<uint32_t>& tokens, 
                         ComputeBackend backend, bool use_cache) {
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }
    auto seq_len = static_cast<size_t>(tokens.size());
    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    const auto& embedding_buffer = gb->get_output_buffer(embedding_node_id_);
    
    for (size_t i = 0; i < embedding_buffer.shape.size(); ++i) {
        
        if (i + 1 < embedding_buffer.shape.size()) {
            
        }
    }
    
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);
    auto final_hidden = forward(gb, hidden, seq_len, backend, use_cache);
    
    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);
    
    return final_hidden;
}
size_t LFM2Model::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    return forward(gb, tokens, backend, use_cache);
}
void LFM2Model::post_execute_updates(CactusGraph* gb, size_t seq_len) {
    if (conv_cache_bx_nodes_.empty()) {
        return;
    }
    
    if (!last_forward_used_cache_ || conv_cache_.window_size == 0) {
        std::fill(conv_cache_bx_nodes_.begin(), conv_cache_bx_nodes_.end(), 0);
        
        return;
    }
    size_t layer_count = std::min(conv_cache_bx_nodes_.size(), weight_nodes_.layers.size());
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        if (weight_nodes_.layers[layer_idx].type != WeightNodeIDs::LayerType::CONV) {
            conv_cache_bx_nodes_[layer_idx] = 0;
            continue;
        }
        size_t bx_node = conv_cache_bx_nodes_[layer_idx];
        if (bx_node != 0) {
            conv_cache_.update(gb, layer_idx, bx_node);
        }
        conv_cache_bx_nodes_[layer_idx] = 0;
    }
    last_forward_used_cache_ = false;
    
}
}
}