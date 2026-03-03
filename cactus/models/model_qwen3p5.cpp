#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <vector>

namespace cactus {
namespace engine {

Qwen3p5Model::Qwen3p5Model() : Model() {
    weight_nodes_.layers.resize(config_.num_layers);
    conv_cache_state_nodes_.assign(config_.num_layers, 0);
}

Qwen3p5Model::Qwen3p5Model(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
    conv_cache_state_nodes_.assign(config.num_layers, 0);
}

void Qwen3p5Model::post_init() {
    bool has_deltanet_layer = false;
    const WeightNodeIDs::LayerWeights* first_linear = nullptr;
    for (const auto& layer : weight_nodes_.layers) {
        if (layer.type == WeightNodeIDs::LayerType::DELTANET) {
            has_deltanet_layer = true;
            if (!first_linear) {
                first_linear = &layer.weights;
            }
            break;
        }
    }

    deltanet_heads_ = static_cast<size_t>(config_.attention_heads);
    deltanet_key_dim_ = static_cast<size_t>(config_.attention_head_dim);
    deltanet_value_dim_ = static_cast<size_t>(config_.attention_head_dim);

    if (has_deltanet_layer &&
        config_.linear_num_key_heads > 0 &&
        config_.linear_key_head_dim > 0) {
        const size_t num_k_heads = static_cast<size_t>(config_.linear_num_key_heads);
        const size_t key_head_dim = static_cast<size_t>(config_.linear_key_head_dim);
        size_t num_v_heads = static_cast<size_t>(config_.linear_num_value_heads > 0
            ? config_.linear_num_value_heads
            : config_.linear_num_key_heads);
        size_t value_head_dim = static_cast<size_t>(config_.linear_value_head_dim);

        if (num_v_heads == 0) {
            throw std::runtime_error("Qwen3p5Model invalid linear-attention value head count in config");
        }
        if (value_head_dim == 0) {
            if (config_.linear_v_proj_dim > 0 && (config_.linear_v_proj_dim % num_v_heads == 0)) {
                value_head_dim = static_cast<size_t>(config_.linear_v_proj_dim / num_v_heads);
            } else {
                throw std::runtime_error("Qwen3p5Model cannot derive linear-attention value head dim from config");
            }
        }
        if (config_.linear_q_proj_dim > 0 &&
            static_cast<size_t>(config_.linear_q_proj_dim) != num_k_heads * key_head_dim) {
            throw std::runtime_error("Qwen3p5Model linear-attention q projection dim mismatch with key-head topology");
        }
        if (config_.linear_v_proj_dim > 0 &&
            static_cast<size_t>(config_.linear_v_proj_dim) != num_v_heads * value_head_dim) {
            throw std::runtime_error("Qwen3p5Model linear-attention v projection dim mismatch with value-head topology");
        }

        deltanet_heads_ = num_v_heads;
        deltanet_key_dim_ = key_head_dim;
        deltanet_value_dim_ = value_head_dim;
    } else if (has_deltanet_layer) {
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        if (!gb) {
            throw std::runtime_error("Qwen3p5Model cannot infer linear-attention dims: graph unavailable");
        }
        if (!first_linear) {
            throw std::runtime_error("Qwen3p5Model expected deltanet layer weights for dim inference");
        }

        const auto& q_buf = gb->get_output_buffer(first_linear->attn_q_weight);
        const auto& v_buf = gb->get_output_buffer(first_linear->attn_v_weight);
        const auto& n_buf = gb->get_output_buffer(first_linear->attn_q_norm_weight);
        if (q_buf.shape.size() < 2 || v_buf.shape.size() < 2 || n_buf.shape.empty()) {
            throw std::runtime_error("Qwen3p5Model invalid linear-attention weight shapes");
        }

        const size_t q_proj_dim = q_buf.shape[0];
        const size_t v_proj_dim = v_buf.shape[0];
        const size_t inferred_key_dim = n_buf.shape[0];
        if (inferred_key_dim == 0 || q_proj_dim == 0 || v_proj_dim == 0) {
            throw std::runtime_error("Qwen3p5Model failed to infer non-zero linear-attention dims");
        }
        if (q_proj_dim % inferred_key_dim != 0) {
            throw std::runtime_error("Qwen3p5Model q projection dim is not divisible by inferred key dim");
        }

        const size_t inferred_k_heads = q_proj_dim / inferred_key_dim;
        const size_t inferred_v_heads = v_proj_dim / inferred_key_dim;
        if (inferred_k_heads == 0 || inferred_v_heads == 0 || inferred_v_heads % inferred_k_heads != 0) {
            throw std::runtime_error("Qwen3p5Model failed to infer compatible linear-attention head topology");
        }

        deltanet_heads_ = inferred_v_heads;
        deltanet_key_dim_ = inferred_key_dim;
        deltanet_value_dim_ = inferred_key_dim;
    }

    if (deltanet_heads_ == 0 || deltanet_key_dim_ == 0 || deltanet_value_dim_ == 0) {
        throw std::runtime_error("Qwen3p5Model requires non-zero deltanet heads/key/value dims");
    }

    deltanet_state_flat_dim_ = deltanet_heads_ * deltanet_key_dim_ * deltanet_value_dim_;

    deltanet_mixed_dim_ = 0;
    if (config_.linear_q_proj_dim > 0 && config_.linear_v_proj_dim > 0) {
        deltanet_mixed_dim_ = static_cast<size_t>(2 * config_.linear_q_proj_dim + config_.linear_v_proj_dim);
    } else if (has_deltanet_layer) {
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        if (!gb || !first_linear) {
            throw std::runtime_error("Qwen3p5Model cannot infer linear-attention mixed projection dim");
        }
        const auto& q_buf = gb->get_output_buffer(first_linear->attn_q_weight);
        const auto& v_buf = gb->get_output_buffer(first_linear->attn_v_weight);
        if (q_buf.shape.size() < 2 || v_buf.shape.size() < 2) {
            throw std::runtime_error("Qwen3p5Model invalid q/v projection shapes for mixed dim inference");
        }
        deltanet_mixed_dim_ = (2 * q_buf.shape[0]) + v_buf.shape[0];
    }

    deltanet_conv_history_len_ = 0;
    if (has_deltanet_layer) {
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        if (!gb) {
            throw std::runtime_error("Qwen3p5Model cannot infer linear-attention conv history: graph unavailable");
        }
        for (const auto& layer : weight_nodes_.layers) {
            if (layer.type != WeightNodeIDs::LayerType::DELTANET || layer.weights.deltanet_conv_weight == 0) {
                continue;
            }
            const auto& conv_buf = gb->get_output_buffer(layer.weights.deltanet_conv_weight);
            if (conv_buf.shape.empty()) {
                continue;
            }
            const size_t kernel = conv_buf.shape.back();
            if (kernel > 1) {
                deltanet_conv_history_len_ = std::max(deltanet_conv_history_len_, kernel - 1);
            }
        }
    }

    if (deltanet_conv_history_len_ > 0 && deltanet_mixed_dim_ == 0) {
        throw std::runtime_error("Qwen3p5Model linear-attention conv history requires non-zero mixed dim");
    }
    deltanet_conv_flat_dim_ = deltanet_conv_history_len_ * deltanet_mixed_dim_;
    deltanet_cache_row_dim_ = deltanet_state_flat_dim_ + deltanet_conv_flat_dim_;
    if (deltanet_cache_row_dim_ == 0) {
        throw std::runtime_error("Qwen3p5Model computed zero cache row dim");
    }

    conv_cache_.init(config_.num_layers, deltanet_cache_row_dim_, 1, Precision::FP32);
    last_forward_used_cache_ = false;
    deltanet_total_seq_len_ = 0;
}

void Qwen3p5Model::reset_cache() {
    Model::reset_cache();
    if (conv_cache_.window_size > 0) {
        conv_cache_.reset();
    }
    std::fill(conv_cache_state_nodes_.begin(), conv_cache_state_nodes_.end(), 0);
    deltanet_total_seq_len_ = 0;
    last_forward_used_cache_ = false;
}

void Qwen3p5Model::post_execute_updates(CactusGraph* gb, size_t seq_len) {
    if (!last_forward_used_cache_) {
        std::fill(conv_cache_state_nodes_.begin(), conv_cache_state_nodes_.end(), 0);
        return;
    }

    const size_t layer_count = std::min(conv_cache_state_nodes_.size(), weight_nodes_.layers.size());
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        if (weight_nodes_.layers[layer_idx].type != WeightNodeIDs::LayerType::DELTANET) {
            conv_cache_state_nodes_[layer_idx] = 0;
            continue;
        }

        const size_t state_node = conv_cache_state_nodes_[layer_idx];
        if (state_node != 0) {
            conv_cache_.update(gb, layer_idx, state_node);
        }
        conv_cache_state_nodes_[layer_idx] = 0;
    }

    deltanet_total_seq_len_ += seq_len;
    last_forward_used_cache_ = false;
}

void Qwen3p5Model::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    auto choose_existing_weight = [&](const std::vector<std::string>& candidates) -> std::string {
        for (const auto& p : candidates) {
            if (std::filesystem::exists(p)) {
                return p;
            }
        }
        return std::string();
    };

    auto mmap_required = [&](const std::vector<std::string>& candidates, const std::string& weight_name) -> size_t {
        const std::string path = choose_existing_weight(candidates);
        if (path.empty()) {
            throw std::runtime_error("Qwen3p5Model missing required weight: " + weight_name);
        }
        return gb->mmap_weights(path);
    };

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer_entry = weight_nodes_.layers[i];
        auto& layer = layer_entry.weights;

        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        bool is_deltanet = false;
        if (i < config_.layer_types.size()) {
            std::string lt = config_.layer_types[i];
            std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
            is_deltanet = (lt.find("deltanet") != std::string::npos ||
                           lt.find("linear_attention") != std::string::npos ||
                           lt.find("linear_attn") != std::string::npos);
        }
        layer_entry.type = is_deltanet ? WeightNodeIDs::LayerType::DELTANET : WeightNodeIDs::LayerType::ATTENTION;

        layer.attn_q_weight = mmap_required(
            {layer_prefix + "attn_q.weights", layer_prefix + "linear_attn_q.weights"},
            "layer_" + std::to_string(i) + "_attn_q/linear_attn_q");
        layer.attn_k_weight = mmap_required(
            {layer_prefix + "attn_k.weights", layer_prefix + "linear_attn_k.weights"},
            "layer_" + std::to_string(i) + "_attn_k/linear_attn_k");
        layer.attn_v_weight = mmap_required(
            {layer_prefix + "attn_v.weights", layer_prefix + "linear_attn_v.weights"},
            "layer_" + std::to_string(i) + "_attn_v/linear_attn_v");
        layer.attn_output_weight = mmap_required(
            {layer_prefix + "attn_output.weights", layer_prefix + "linear_attn_output.weights"},
            "layer_" + std::to_string(i) + "_attn_output/linear_attn_output");
        layer.input_layernorm_weight = mmap_required(
            {layer_prefix + "input_norm.weights"},
            "layer_" + std::to_string(i) + "_input_norm");
        layer.attn_q_norm_weight = mmap_required(
            {layer_prefix + "attn_q_norm.weights", layer_prefix + "linear_attn_norm.weights"},
            "layer_" + std::to_string(i) + "_attn_q_norm/linear_attn_norm");
        layer.attn_k_norm_weight = mmap_required(
            {layer_prefix + "attn_k_norm.weights", layer_prefix + "linear_attn_norm.weights"},
            "layer_" + std::to_string(i) + "_attn_k_norm/linear_attn_norm");

        if (is_deltanet) {
            const std::string qkv_path = choose_existing_weight({
                layer_prefix + "linear_attn_qkv.weights"
            });
            layer.deltanet_qkv_weight = qkv_path.empty() ? 0 : gb->mmap_weights(qkv_path);

            const std::string gate_path = choose_existing_weight({
                layer_prefix + "deltanet_gate.weights",
                layer_prefix + "attn_gate.weights",
                layer_prefix + "attn_f_gate.weights",
                layer_prefix + "linear_attn_a.weights"
            });
            const std::string beta_path = choose_existing_weight({
                layer_prefix + "deltanet_beta.weights",
                layer_prefix + "attn_beta.weights",
                layer_prefix + "attn_f_beta.weights",
                layer_prefix + "linear_attn_b.weights"
            });
            if (gate_path.empty() || beta_path.empty()) {
                throw std::runtime_error(
                    "Qwen3p5Model deltanet layer missing gate/beta weights at layer " + std::to_string(i));
            }
            layer.deltanet_gate_weight = gb->mmap_weights(gate_path);
            layer.deltanet_beta_weight = gb->mmap_weights(beta_path);

            const std::string gate_bias_path = choose_existing_weight({
                layer_prefix + "linear_attn_A_log.weights"
            });
            const std::string beta_bias_path = choose_existing_weight({
                layer_prefix + "linear_attn_dt_bias.weights"
            });
            layer.deltanet_gate_bias = gate_bias_path.empty() ? 0 : gb->mmap_weights(gate_bias_path);
            layer.deltanet_beta_bias = beta_bias_path.empty() ? 0 : gb->mmap_weights(beta_bias_path);
            const std::string z_path = choose_existing_weight({
                layer_prefix + "linear_attn_z.weights"
            });
            const std::string conv_path = choose_existing_weight({
                layer_prefix + "linear_attn_conv1d.weights"
            });
            layer.deltanet_z_weight = z_path.empty() ? 0 : gb->mmap_weights(z_path);
            layer.deltanet_conv_weight = conv_path.empty() ? 0 : gb->mmap_weights(conv_path);
        }

        layer.ffn_gate_weight = mmap_required(
            {layer_prefix + "ffn_gate.weights"},
            "layer_" + std::to_string(i) + "_ffn_gate");
        layer.ffn_up_weight = mmap_required(
            {layer_prefix + "ffn_up.weights"},
            "layer_" + std::to_string(i) + "_ffn_up");
        layer.ffn_down_weight = mmap_required(
            {layer_prefix + "ffn_down.weights"},
            "layer_" + std::to_string(i) + "_ffn_down");
        layer.post_attention_layernorm_weight = mmap_required(
            {layer_prefix + "post_attn_norm.weights"},
            "layer_" + std::to_string(i) + "_post_attn_norm");
    }

    if (npu::is_npu_available()) {
        std::string npu_prefill_path = model_folder_path_ + "/model.mlpackage";
        load_npu_prefill(npu_prefill_path);
    }
}

size_t Qwen3p5Model::build_gated_deltanet(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                          ComputeBackend backend, bool use_cache, size_t position_offset) {
    (void)position_offset;
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;

    const auto& in_shape = gb->get_output_buffer(normalized_input).shape;
    if (in_shape.size() < 2) {
        throw std::runtime_error("Qwen3p5Model linear-attention expects rank-2 normalized input");
    }
    const size_t seq_len = in_shape[0];

    size_t q_proj = 0;
    size_t k_proj = 0;
    size_t v_proj = 0;
    size_t mixed_qkv = 0;
    size_t z_proj = layer.deltanet_z_weight ? gb->matmul(normalized_input, layer.deltanet_z_weight, true, backend) : 0;

    const size_t num_heads = deltanet_heads_;
    const size_t key_dim = deltanet_key_dim_;
    const size_t value_dim = deltanet_value_dim_;
    const size_t v_proj_dim = num_heads * value_dim;
    size_t qk_proj_dim = 0;

    if (layer.deltanet_qkv_weight != 0) {
        mixed_qkv = gb->matmul(normalized_input, layer.deltanet_qkv_weight, true, backend);
        const auto& mixed_shape = gb->get_output_buffer(mixed_qkv).shape;
        if (mixed_shape.size() < 2 || mixed_shape[0] != seq_len) {
            throw std::runtime_error("Qwen3p5Model invalid linear-attention joint qkv projection shape");
        }
        if (mixed_shape[1] <= v_proj_dim || ((mixed_shape[1] - v_proj_dim) % 2) != 0) {
            throw std::runtime_error("Qwen3p5Model invalid linear-attention joint qkv projection width");
        }
        qk_proj_dim = (mixed_shape[1] - v_proj_dim) / 2;
    } else {
        q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
        k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
        v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
        const auto& q_shape = gb->get_output_buffer(q_proj).shape;
        if (q_shape.size() < 2 || q_shape[0] != seq_len) {
            throw std::runtime_error("Qwen3p5Model invalid linear-attention q projection shape");
        }
        qk_proj_dim = q_shape[1];
        size_t mixed_qk = gb->concat(q_proj, k_proj, 1);
        mixed_qkv = gb->concat(mixed_qk, v_proj, 1);
    }

    if (qk_proj_dim == 0 || qk_proj_dim % key_dim != 0 || v_proj_dim == 0) {
        throw std::runtime_error("Qwen3p5Model invalid linear-attention projection dimensions");
    }
    const size_t num_k_heads = qk_proj_dim / key_dim;
    if (num_heads % num_k_heads != 0) {
        throw std::runtime_error("Qwen3p5Model requires num_v_heads divisible by num_k_heads");
    }

    const size_t mixed_proj_dim = qk_proj_dim + qk_proj_dim + v_proj_dim;
    if (deltanet_mixed_dim_ != 0 && mixed_proj_dim != deltanet_mixed_dim_) {
        throw std::runtime_error("Qwen3p5Model linear-attention mixed projection dim mismatch");
    }
    const size_t state_flat_dim = deltanet_state_flat_dim_;
    size_t prev_state_flat = 0;
    size_t prev_conv_flat = 0;
    if (use_cache && conv_cache_.window_size > 0) {
        const auto view = conv_cache_.get_window(layer_idx);
        if (conv_cache_.window_size == 1 && view.total_len == 1) {
            const uint8_t* cache_row = nullptr;
            if (view.len2 == 1 && view.ptr2 != nullptr) {
                cache_row = static_cast<const uint8_t*>(view.ptr2);
            } else if (view.len1 == 1 && view.ptr1 != nullptr) {
                cache_row = static_cast<const uint8_t*>(view.ptr1);
            }
            if (cache_row != nullptr) {
                prev_state_flat = gb->input({1, state_flat_dim}, conv_cache_.precision);
                gb->set_external_input(prev_state_flat, const_cast<void*>(static_cast<const void*>(cache_row)),
                                       conv_cache_.precision);
                if (deltanet_conv_flat_dim_ > 0) {
                    const size_t state_bytes = state_flat_dim * conv_cache_.element_size;
                    prev_conv_flat = gb->input({1, deltanet_conv_flat_dim_}, conv_cache_.precision);
                    gb->set_external_input(
                        prev_conv_flat,
                        const_cast<void*>(static_cast<const void*>(cache_row + state_bytes)),
                        conv_cache_.precision);
                }
            }
        }

        if (prev_state_flat == 0) {
            std::vector<size_t> segments;
            if (view.len2 > 0) {
                size_t left_node = gb->input({view.len2, deltanet_cache_row_dim_}, conv_cache_.precision);
                gb->set_external_input(left_node, const_cast<void*>(view.ptr2), conv_cache_.precision);
                segments.push_back(left_node);
            }
            if (view.len1 > 0) {
                size_t right_node = gb->input({view.len1, deltanet_cache_row_dim_}, conv_cache_.precision);
                gb->set_external_input(right_node, const_cast<void*>(view.ptr1), conv_cache_.precision);
                segments.push_back(right_node);
            }

            if (!segments.empty()) {
                size_t stacked = segments[0];
                for (size_t i = 1; i < segments.size(); ++i) {
                    stacked = gb->concat(stacked, segments[i], 0);
                }
                if (view.total_len > 1) {
                    stacked = gb->slice(stacked, 0, view.total_len - 1, 1);
                }
                prev_state_flat = gb->slice(stacked, 1, 0, state_flat_dim);
                if (deltanet_conv_flat_dim_ > 0) {
                    prev_conv_flat = gb->slice(stacked, 1, state_flat_dim, deltanet_conv_flat_dim_);
                }
            }
        }
    }

    size_t conv_source_with_history = mixed_qkv;
    if (layer.deltanet_conv_weight) {
        const auto& conv_wbuf = gb->get_output_buffer(layer.deltanet_conv_weight);
        const size_t kernel = conv_wbuf.shape.back();
        if (kernel > 1 && use_cache && deltanet_conv_history_len_ > 0) {
            size_t history_2d = 0;
            if (prev_conv_flat != 0) {
                history_2d = gb->reshape(prev_conv_flat, {deltanet_conv_history_len_, mixed_proj_dim});
                history_2d = gb->precision_cast(history_2d, Precision::FP16);
            } else {
                history_2d = gb->input({deltanet_conv_history_len_, mixed_proj_dim}, Precision::FP16);
                std::vector<__fp16> zeros(deltanet_conv_flat_dim_, static_cast<__fp16>(0.0f));
                gb->set_input(history_2d, zeros.data(), Precision::FP16);
            }
            conv_source_with_history = gb->concat(history_2d, mixed_qkv, 0);
        }

        const size_t total_conv_len = gb->get_output_buffer(conv_source_with_history).shape[0];
        size_t conv_input = gb->reshape(conv_source_with_history, {1, total_conv_len, mixed_proj_dim});
        size_t conv_out = gb->conv1d_causal(conv_input, layer.deltanet_conv_weight, kernel, 1);
        const size_t conv_start = total_conv_len > seq_len ? total_conv_len - seq_len : 0;
        conv_out = gb->slice(conv_out, 1, conv_start, seq_len);
        mixed_qkv = gb->reshape(gb->silu(conv_out), {seq_len, mixed_proj_dim});
    }

    q_proj = gb->slice(mixed_qkv, 1, 0, qk_proj_dim);
    k_proj = gb->slice(mixed_qkv, 1, qk_proj_dim, qk_proj_dim);
    v_proj = gb->slice(mixed_qkv, 1, qk_proj_dim + qk_proj_dim, v_proj_dim);

    size_t q_4d = gb->reshape(q_proj, {1, seq_len, num_k_heads, key_dim});
    size_t k_4d = gb->reshape(k_proj, {1, seq_len, num_k_heads, key_dim});
    size_t v_4d = gb->reshape(v_proj, {1, seq_len, num_heads, value_dim});

    size_t q_norm = gb->sum(gb->multiply(q_4d, q_4d), 3);
    q_norm = gb->scalar_sqrt(gb->scalar_add(q_norm, 1e-6f));
    q_norm = gb->reshape(q_norm, {1, seq_len, num_k_heads, 1});
    q_4d = gb->divide(q_4d, q_norm);

    size_t k_norm = gb->sum(gb->multiply(k_4d, k_4d), 3);
    k_norm = gb->scalar_sqrt(gb->scalar_add(k_norm, 1e-6f));
    k_norm = gb->reshape(k_norm, {1, seq_len, num_k_heads, 1});
    k_4d = gb->divide(k_4d, k_norm);

    size_t a_logits = gb->matmul(normalized_input, layer.deltanet_gate_weight, true, backend);
    size_t b_logits = gb->matmul(normalized_input, layer.deltanet_beta_weight, true, backend);
    a_logits = gb->precision_cast(a_logits, Precision::FP16);
    b_logits = gb->precision_cast(b_logits, Precision::FP16);
    if (layer.deltanet_beta_bias) {
        size_t dt_bias_vec = layer.deltanet_beta_bias;
        const auto& dt_buf = gb->get_output_buffer(dt_bias_vec);
        if (!dt_buf.shape.empty() && dt_buf.shape[0] > num_heads) {
            dt_bias_vec = gb->slice(dt_bias_vec, 0, 0, num_heads);
        }
        size_t dt_bias_2d = gb->reshape(dt_bias_vec, {1, num_heads});
        dt_bias_2d = gb->precision_cast(dt_bias_2d, Precision::FP16);
        a_logits = gb->add(a_logits, dt_bias_2d);
    }
    size_t a_softplus = gb->scalar_log(gb->scalar_add(gb->scalar_exp(a_logits), 1.0f));

    size_t gate_log;
    if (layer.deltanet_gate_bias) {
        size_t a_log_vec = layer.deltanet_gate_bias;
        const auto& a_log_buf = gb->get_output_buffer(a_log_vec);
        if (!a_log_buf.shape.empty() && a_log_buf.shape[0] > num_heads) {
            a_log_vec = gb->slice(a_log_vec, 0, 0, num_heads);
        }
        size_t a_log_2d = gb->reshape(a_log_vec, {1, num_heads});
        a_log_2d = gb->precision_cast(a_log_2d, Precision::FP16);
        size_t neg_exp_a = gb->scalar_multiply(gb->scalar_exp(a_log_2d), -1.0f);
        gate_log = gb->multiply(neg_exp_a, a_softplus);
    } else {
        gate_log = gb->scalar_multiply(a_softplus, -1.0f);
    }
    size_t beta = gb->sigmoid(b_logits);

    size_t gate_3d = gb->reshape(gate_log, {1, seq_len, num_heads});
    size_t beta_3d = gb->reshape(beta, {1, seq_len, num_heads});

    size_t initial_state;

    if (prev_state_flat != 0) {
        initial_state = gb->reshape(prev_state_flat, {1, key_dim, num_heads, value_dim});
    } else {
        initial_state = gb->input({1, key_dim, num_heads, value_dim}, Precision::FP16);
        std::vector<__fp16> zeros(state_flat_dim, static_cast<__fp16>(0.0f));
        gb->set_input(initial_state, zeros.data(), Precision::FP16);
    }

    size_t deltanet_out;
    if (use_cache && seq_len == 1) {
        deltanet_out = gb->gated_deltanet_decode(q_4d, k_4d, v_4d, gate_3d, beta_3d, initial_state, 0.0f);
    } else {
        const size_t chunk_for_op = std::min<size_t>(64, std::max<size_t>(1, seq_len));
        deltanet_out = gb->gated_deltanet_prefill(
            q_4d, k_4d, v_4d, gate_3d, beta_3d, initial_state, chunk_for_op, 0.0f);
    }

    size_t y_4d = gb->slice(deltanet_out, 1, 0, seq_len);
    size_t state_tail = gb->slice(deltanet_out, 1, seq_len, key_dim);

    if (use_cache) {
        size_t packed_cache = gb->reshape(state_tail, {1, state_flat_dim});
        if (deltanet_conv_flat_dim_ > 0) {
            size_t history_2d = 0;
            if (layer.deltanet_conv_weight) {
                const size_t total_history_len = gb->get_output_buffer(conv_source_with_history).shape[0];
                if (total_history_len >= deltanet_conv_history_len_) {
                    const size_t hist_start = total_history_len - deltanet_conv_history_len_;
                    history_2d = gb->slice(conv_source_with_history, 0, hist_start, deltanet_conv_history_len_);
                } else {
                    size_t pad_rows = deltanet_conv_history_len_ - total_history_len;
                    size_t pad = gb->input({pad_rows, mixed_proj_dim}, Precision::FP16);
                    std::vector<__fp16> zeros(pad_rows * mixed_proj_dim, static_cast<__fp16>(0.0f));
                    gb->set_input(pad, zeros.data(), Precision::FP16);
                    history_2d = gb->concat(pad, conv_source_with_history, 0);
                }
            } else if (prev_conv_flat != 0) {
                history_2d = gb->reshape(prev_conv_flat, {deltanet_conv_history_len_, mixed_proj_dim});
                history_2d = gb->precision_cast(history_2d, Precision::FP16);
            } else {
                history_2d = gb->input({deltanet_conv_history_len_, mixed_proj_dim}, Precision::FP16);
                std::vector<__fp16> zeros(deltanet_conv_flat_dim_, static_cast<__fp16>(0.0f));
                gb->set_input(history_2d, zeros.data(), Precision::FP16);
            }

            size_t history_flat = gb->reshape(history_2d, {1, deltanet_conv_flat_dim_});
            packed_cache = gb->concat(packed_cache, history_flat, 1);
        }
        packed_cache = gb->precision_cast(packed_cache, conv_cache_.precision);
        conv_cache_state_nodes_[layer_idx] = packed_cache;
        cache_k_output_nodes_[layer_idx] = 0;
        cache_v_output_nodes_[layer_idx] = 0;
    } else {
        conv_cache_state_nodes_[layer_idx] = 0;
    }

    size_t y_2d = gb->reshape(y_4d, {seq_len * num_heads, value_dim});
    if (z_proj != 0) {
        size_t z_3d = gb->reshape(z_proj, {seq_len, num_heads, value_dim});
        size_t z_2d = gb->reshape(z_3d, {seq_len * num_heads, value_dim});
        y_2d = gb->multiply(gb->rms_norm(y_2d, layer.attn_q_norm_weight, config_.layer_norm_eps), gb->silu(z_2d));
    }
    y_2d = gb->reshape(y_2d, {seq_len, num_heads * value_dim});
    return gb->matmul(y_2d, layer.attn_output_weight, true, backend);
}

size_t Qwen3p5Model::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                     ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;

    if (layer_entry.type == WeightNodeIDs::LayerType::DELTANET) {
        return build_gated_deltanet(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    }

    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    const size_t q_target_dim = num_heads * head_dim;

    size_t q_gate = 0;
    const size_t q_proj_dim = q_shape.size() > 1 ? q_shape[1] : 0;
    if (q_proj_dim == 2 * q_target_dim) {
        size_t q_packed = gb->reshape(q_proj, {batch_seq, num_heads, 2 * head_dim});
        q_gate = gb->slice(q_packed, 2, head_dim, head_dim);
        q_proj = gb->slice(q_packed, 2, 0, head_dim);
        q_gate = gb->reshape(q_gate, {batch_seq, q_target_dim});
        q_proj = gb->reshape(q_proj, {batch_seq, q_target_dim});
    }

    const size_t q_norm_weight = gb->scalar_add(layer.attn_q_norm_weight, 1.0f);
    q_proj = gb->reshape(q_proj, {batch_seq * num_heads, head_dim});
    q_proj = gb->rms_norm(q_proj, q_norm_weight, config_.layer_norm_eps);
    q_proj = gb->reshape(q_proj, {batch_seq, num_heads * head_dim});

    size_t num_kv_heads = config_.attention_kv_heads;
    const size_t k_norm_weight = gb->scalar_add(layer.attn_k_norm_weight, 1.0f);
    k_proj = gb->reshape(k_proj, {batch_seq * num_kv_heads, head_dim});
    k_proj = gb->rms_norm(k_proj, k_norm_weight, config_.layer_norm_eps);
    k_proj = gb->reshape(k_proj, {batch_seq, num_kv_heads * head_dim});

    size_t seq_len = batch_seq;

    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});

    if (config_.rope_theta > 0) {
        float rotary_factor = config_.partial_rotary_factor;
        if (rotary_factor <= 0.0f) {
            rotary_factor = 0.25f;
        }
        size_t rot_dim = static_cast<size_t>(static_cast<float>(head_dim) * rotary_factor);
        rot_dim = std::min(rot_dim, head_dim);
        if ((rot_dim & 1u) != 0) {
            rot_dim -= 1;
        }

        auto apply_partial_rope = [&](size_t x_4d) -> size_t {
            if (rot_dim == 0) {
                return x_4d;
            }
            if (rot_dim >= head_dim) {
                return gb->rope(x_4d, config_.rope_theta, position_offset);
            }
            size_t x_rot = gb->slice(x_4d, 3, 0, rot_dim);
            size_t x_tail = gb->slice(x_4d, 3, rot_dim, head_dim - rot_dim);
            x_rot = gb->rope(x_rot, config_.rope_theta, position_offset);
            return gb->concat(x_rot, x_tail, 3);
        };

        q_proj_4d = apply_partial_rope(q_proj_4d);
        k_proj_4d = apply_partial_rope(k_proj_4d);
    }

    size_t attn_output_4d;

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = k_proj_4d;
        cache_v_output_nodes_[layer_idx] = v_proj_4d;
        conv_cache_state_nodes_[layer_idx] = 0;
    }

    if (use_cache && !kv_cache_.is_empty()) {
        attn_output_4d = gb->attention_int8_hybrid(
            q_proj_4d, k_proj_4d, v_proj_4d,
            attention_scale_, position_offset,
            kv_cache_.get_keys_int8(layer_idx),
            kv_cache_.get_values_int8(layer_idx),
            kv_cache_.get_key_scales(layer_idx),
            kv_cache_.get_value_scales(layer_idx),
            kv_cache_.current_seq_len, num_kv_heads, head_dim
        );
    } else {
        attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, position_offset);
    }

    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    if (q_gate != 0) {
        attn_output = gb->multiply(attn_output, gb->sigmoid(q_gate));
    }
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}

size_t Qwen3p5Model::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                               ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx].weights;
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}

size_t Qwen3p5Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                             ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx].weights;
    const size_t input_norm_weight = gb->scalar_add(layer.input_layernorm_weight, 1.0f);
    auto normalized_input = gb->rms_norm(hidden, input_norm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    auto after_attention = gb->add(hidden, attn_output);
    const size_t post_attn_norm_weight = gb->scalar_add(layer.post_attention_layernorm_weight, 1.0f);
    auto normalized_after_attention = gb->rms_norm(after_attention, post_attn_norm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_attention, layer_idx, backend);
    return gb->add(after_attention, mlp_output);
}

size_t Qwen3p5Model::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    if (conv_cache_state_nodes_.size() != config_.num_layers) {
        conv_cache_state_nodes_.assign(config_.num_layers, 0);
    }
    std::fill(conv_cache_state_nodes_.begin(), conv_cache_state_nodes_.end(), 0);
    last_forward_used_cache_ = use_cache;
    if (!use_cache && conv_cache_.window_size > 0) {
        conv_cache_.reset();
        deltanet_total_seq_len_ = 0;
    }

    auto seq_len = static_cast<size_t>(tokens.size());
    const size_t kv_pos = use_cache ? kv_cache_.get_total_seq_len() : 0;
    const size_t position_offset = use_cache ? std::max(kv_pos, deltanet_total_seq_len_) : 0;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    const size_t final_norm_weight = gb->scalar_add(weight_nodes_.output_norm_weight, 1.0f);
    auto final_hidden = gb->rms_norm(hidden, final_norm_weight, config_.layer_norm_eps);
    return final_hidden;
}

}
}
