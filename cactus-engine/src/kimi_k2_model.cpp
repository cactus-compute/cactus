#include "kimi_k2_model.h"
#include "cactus_graph.h"

#define PICOJSON_USE_INT64
#include "picojson.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cactus {
namespace engine {

namespace fs = std::filesystem;

namespace {

void maybe_print_kimi_logits_topk(CactusGraph& graph, size_t logits_node, size_t top_k) {
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") == nullptr) return;

    const auto& buffer = graph.get_output_buffer(logits_node);
    if (buffer.shape.size() != 2 || buffer.shape[1] == 0) return;
    const size_t seq_len = buffer.shape[0];
    const size_t vocab = buffer.shape[1];
    const size_t offset = (seq_len - 1) * vocab;
    top_k = std::min(top_k, vocab);

    std::vector<std::pair<float, uint32_t>> values;
    values.reserve(vocab);
    if (buffer.precision == Precision::FP16) {
        const auto* logits = static_cast<const __fp16*>(graph.get_output(logits_node));
        for (uint32_t i = 0; i < vocab; ++i) {
            values.emplace_back(static_cast<float>(logits[offset + i]), i);
        }
    } else if (buffer.precision == Precision::FP32) {
        const auto* logits = static_cast<const float*>(graph.get_output(logits_node));
        for (uint32_t i = 0; i < vocab; ++i) {
            values.emplace_back(logits[offset + i], i);
        }
    } else {
        return;
    }

    size_t nonzero = 0;
    size_t nan_count = 0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto& [value, id] : values) {
        if (value != 0.0f) ++nonzero;
        if (std::isnan(value)) ++nan_count;
        if (std::isfinite(value)) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }

    std::partial_sort(values.begin(), values.begin() + top_k, values.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::cerr << "[kimi_logits_topk nonzero=" << nonzero
              << " nan=" << nan_count
              << " min=" << min_value
              << " max=" << max_value << "]";
    for (size_t i = 0; i < top_k; ++i) {
        std::cerr << " " << values[i].second << ":" << values[i].first;
    }
    std::cerr << "\n";
}

void maybe_print_node_stats(CactusGraph& graph, size_t node_id, const char* label) {
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") == nullptr) return;
    const auto& buffer = graph.get_output_buffer(node_id);
    size_t nonzero = 0;
    size_t nan_count = 0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    double abs_sum = 0.0;

    if (buffer.precision == Precision::FP16) {
        const auto* data = static_cast<const __fp16*>(graph.get_output(node_id));
        for (size_t i = 0; i < buffer.total_size; ++i) {
            float value = static_cast<float>(data[i]);
            if (value != 0.0f) ++nonzero;
            if (std::isnan(value)) ++nan_count;
            if (std::isfinite(value)) {
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
                abs_sum += std::abs(value);
            }
        }
    } else if (buffer.precision == Precision::FP32) {
        const auto* data = static_cast<const float*>(graph.get_output(node_id));
        for (size_t i = 0; i < buffer.total_size; ++i) {
            float value = data[i];
            if (value != 0.0f) ++nonzero;
            if (std::isnan(value)) ++nan_count;
            if (std::isfinite(value)) {
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
                abs_sum += std::abs(value);
            }
        }
    } else {
        return;
    }

    std::cerr << "[kimi_node_stats " << label
              << " shape=";
    for (size_t i = 0; i < buffer.shape.size(); ++i) {
        if (i > 0) std::cerr << "x";
        std::cerr << buffer.shape[i];
    }
    std::cerr << " nonzero=" << nonzero
              << " nan=" << nan_count
              << " min=" << min_value
              << " max=" << max_value
              << " abs_sum=" << abs_sum << "]\n";
}

}  // namespace

KimiK2Model::KimiK2Model() : Model(), config_copy_() {}

KimiK2Model::KimiK2Model(const Config& config) : Model(config), config_copy_(config) {}

bool KimiK2Model::init(const std::string& model_dir, size_t context_size,
                       const std::string& /*system_prompt*/, bool /*do_warmup*/) {
    model_dir_ = model_dir;
    context_size_ = context_size;

    Config parsed;
    if (!load_config(model_dir, parsed)) {
        CACTUS_LOG_ERROR("kimi_k2", "Failed to read config.txt or config.json from " << model_dir);
        return false;
    }
    config_copy_ = parsed;
    validate_architecture();

    if (!setup_tokenizer(model_dir)) {
        CACTUS_LOG_ERROR("kimi_k2", "Failed to initialize tokenizer from " << model_dir);
        return false;
    }
    if (!load_weight_manifest(model_dir)) {
        CACTUS_LOG_ERROR("kimi_k2", "Failed to load Kimi weight manifest from " << model_dir);
        return false;
    }
    validate_required_weights();
    graph_ = std::make_unique<CactusGraph>();
    load_weights_to_graph();
    initialize_cache_states();
    kimi_cache_seq_len_ = 0;

    return true;
}

uint32_t KimiK2Model::decode(const std::vector<uint32_t>& tokens,
                             float temperature, float top_p, size_t top_k, const std::string& profile_file,
                             float* out_entropy, float min_p, float repetition_penalty) {
    if (!graph_) throw std::runtime_error("Kimi graph is not initialized");
    if (tokens.empty()) throw std::runtime_error("Kimi decode requires at least one token");
    if (kimi_cache_seq_len_ + tokens.size() > context_size_) {
        throw std::runtime_error("Kimi decode input exceeds configured context");
    }
    if (out_entropy) *out_entropy = 0.0f;

    const size_t position_offset = kimi_cache_seq_len_;
    graph_->soft_reset_keep_pool();
    size_t hidden = build_forward(*graph_, tokens, true, position_offset);
    size_t logits_hidden = tokens.size() == 1 ? hidden : graph_->slice(hidden, 0, tokens.size() - 1, 1);
    size_t normed = graph_->rms_norm(logits_hidden, output_norm_node_, config_copy_.layer_norm_eps);
    size_t logits = graph_->matmul(normed, output_weight_node_, true, ComputeBackend::CPU);
    graph_->register_debug_node(config_copy_.num_layers, "kimi_logits", logits);
    if (temperature < 0.0f) temperature = 0.0f;
    if (top_p < 0.0f) top_p = 1.0f;
    size_t sample = graph_->sample_with_options(logits, temperature, top_p, min_p, repetition_penalty, top_k);
    graph_->execute(profile_file);
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr) {
        for (const auto& entry : graph_->get_debug_nodes()) {
            if (entry.name == "kimi_embedding" || entry.name == "kimi_layer_hidden" ||
                entry.name == "kimi_layer_post_attn" || entry.name == "kimi_layer_mlp_out" ||
                entry.name == "kimi_moe_routed" || entry.name == "kimi_moe_shared" ||
                entry.name == "kimi_moe_topk_indices") {
                std::string label = entry.name + "_" + std::to_string(entry.layer_idx);
                maybe_print_node_stats(*graph_, entry.node_id, label.c_str());
            }
        }
    }
    maybe_print_node_stats(*graph_, hidden, "hidden");
    maybe_print_node_stats(*graph_, normed, "normed");
    maybe_print_kimi_logits_topk(*graph_, logits, 8);

    const auto* output = static_cast<const uint32_t*>(graph_->get_output(sample));
    uint32_t token = output[0];
    kimi_cache_seq_len_ += tokens.size();
    record_sampled_token(token);
    return token;
}

bool KimiK2Model::prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token) {
    if (tokens.empty()) return false;
    out_token = decode(tokens, 0.0f, 1.0f, 1);
    return true;
}

void KimiK2Model::prefill(const std::vector<uint32_t>& tokens, size_t, const std::string& profile_file, bool) {
    if (!graph_) throw std::runtime_error("Kimi graph is not initialized");
    if (tokens.empty()) return;
    if (kimi_cache_seq_len_ + tokens.size() > context_size_) {
        throw std::runtime_error("Kimi prefill input exceeds configured context");
    }

    const size_t position_offset = kimi_cache_seq_len_;
    graph_->soft_reset_keep_pool();
    build_forward(*graph_, tokens, true, position_offset);
    graph_->execute(profile_file);
    kimi_cache_seq_len_ += tokens.size();
}

void KimiK2Model::reset_cache() {
    if (!graph_) return;
    graph_->hard_reset();
    load_weights_to_graph();
    initialize_cache_states();
    kimi_cache_seq_len_ = 0;
}

void KimiK2Model::prefetch_moe_expert_pages() {
    if (!graph_) throw std::runtime_error("Kimi graph is not initialized");
    for (uint32_t layer_idx = config_copy_.num_dense_layers; layer_idx < config_copy_.num_layers; ++layer_idx) {
        const auto& moe = layers_[layer_idx].moe;
        for (const auto& expert : moe.experts) {
            graph_->prefetch_weight_pages(expert.gate);
            graph_->prefetch_weight_pages(expert.up);
            graph_->prefetch_weight_pages(expert.down);
        }
    }
}

void KimiK2Model::load_weights_to_graph() {
    if (!graph_) throw std::runtime_error("Kimi graph is not initialized");
    auto& gb = *graph_;
    embedding_node_ = mmap_weight(gb, "model.embed_tokens.weight");
    output_norm_node_ = mmap_weight(gb, "model.norm.weight");
    output_weight_node_ = mmap_weight(gb, "lm_head.weight");

    layers_.clear();
    layers_.resize(config_copy_.num_layers);
    for (uint32_t i = 0; i < config_copy_.num_layers; ++i) {
        auto& layer = layers_[i];
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        layer.input_norm = mmap_weight(gb, prefix + "input_layernorm.weight");
        layer.q_a = mmap_weight(gb, prefix + "self_attn.q_a_proj.weight");
        layer.q_a_norm = mmap_weight(gb, prefix + "self_attn.q_a_layernorm.weight");
        layer.q_b = mmap_weight(gb, prefix + "self_attn.q_b_proj.weight");
        layer.kv_a = mmap_weight(gb, prefix + "self_attn.kv_a_proj_with_mqa.weight");
        layer.kv_a_norm = mmap_weight(gb, prefix + "self_attn.kv_a_layernorm.weight");
        layer.kv_b = mmap_weight(gb, prefix + "self_attn.kv_b_proj.weight");
        layer.o = mmap_weight(gb, prefix + "self_attn.o_proj.weight");
        layer.post_attn_norm = mmap_weight(gb, prefix + "post_attention_layernorm.weight");

        if (i < config_copy_.num_dense_layers) {
            layer.dense.gate = mmap_weight(gb, prefix + "mlp.gate_proj.weight");
            layer.dense.up = mmap_weight(gb, prefix + "mlp.up_proj.weight");
            layer.dense.down = mmap_weight(gb, prefix + "mlp.down_proj.weight");
        } else {
            layer.moe.router = mmap_weight(gb, prefix + "mlp.gate.weight");
            layer.moe.router_bias = mmap_weight(gb, prefix + "mlp.gate.e_score_correction_bias");
            layer.moe.experts.resize(config_copy_.num_experts);
            for (uint32_t e = 0; e < config_copy_.num_experts; ++e) {
                const std::string ep = prefix + "mlp.experts." + std::to_string(e) + ".";
                layer.moe.experts[e].gate = mmap_weight(gb, ep + "gate_proj.weight");
                layer.moe.experts[e].up = mmap_weight(gb, ep + "up_proj.weight");
                layer.moe.experts[e].down = mmap_weight(gb, ep + "down_proj.weight");
            }
            layer.moe.shared.gate = mmap_weight(gb, prefix + "mlp.shared_experts.gate_proj.weight");
            layer.moe.shared.up = mmap_weight(gb, prefix + "mlp.shared_experts.up_proj.weight");
            layer.moe.shared.down = mmap_weight(gb, prefix + "mlp.shared_experts.down_proj.weight");
        }
    }
}

void KimiK2Model::initialize_cache_states() {
    if (!graph_) throw std::runtime_error("Kimi graph is not initialized");
    const size_t heads = config_copy_.attention_heads;
    const size_t key_dim = config_copy_.qk_nope_head_dim + config_copy_.qk_rope_head_dim;
    const size_t value_dim = config_copy_.v_head_dim;
    cache_nodes_.clear();
    cache_nodes_.resize(config_copy_.num_layers);
    for (uint32_t i = 0; i < config_copy_.num_layers; ++i) {
        cache_nodes_[i].key = graph_->kv_cache_state(context_size_, heads, key_dim, context_size_, 0);
        cache_nodes_[i].value = graph_->kv_cache_state(context_size_, heads, value_dim, context_size_, 0);
    }
}

size_t KimiK2Model::mmap_weight(CactusGraph& gb, const std::string& logical_name) {
    return gb.mmap_weights(weight_path(logical_name));
}

size_t KimiK2Model::build_forward(CactusGraph& gb, const std::vector<uint32_t>& tokens,
                                  bool use_cache, size_t position_offset) {
    std::vector<float> token_ids(tokens.begin(), tokens.end());
    size_t token_input = gb.input({tokens.size()}, Precision::FP32);
    gb.set_input(token_input, token_ids.data(), Precision::FP32);

    size_t hidden = gb.embedding(embedding_node_, token_input);
    hidden = gb.reshape(hidden, {tokens.size(), config_copy_.hidden_dim});
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr) {
        gb.register_debug_node(0, "kimi_embedding", hidden);
    }

    for (uint32_t layer_idx = 0; layer_idx < config_copy_.num_layers; ++layer_idx) {
        size_t attn_input = gb.rms_norm(hidden, layers_[layer_idx].input_norm, config_copy_.layer_norm_eps);
        size_t attn_out = build_attention(gb, attn_input, layer_idx, tokens.size(), use_cache, position_offset);
        hidden = gb.add(hidden, attn_out);
        if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr && layer_idx < 4) {
            gb.register_debug_node(layer_idx, "kimi_layer_post_attn", hidden);
        }

        size_t mlp_input = gb.rms_norm(hidden, layers_[layer_idx].post_attn_norm, config_copy_.layer_norm_eps);
        size_t mlp_out = layer_idx < config_copy_.num_dense_layers
            ? build_dense_mlp(gb, mlp_input, layer_idx)
            : build_moe(gb, mlp_input, layer_idx);
        if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr && layer_idx < 4) {
            gb.register_debug_node(layer_idx, "kimi_layer_mlp_out", mlp_out);
        }
        hidden = gb.add(hidden, mlp_out);
        if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr) {
            gb.register_debug_node(layer_idx, "kimi_layer_hidden", hidden);
        }
    }

    return hidden;
}

size_t KimiK2Model::build_attention(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx, size_t seq_len,
                                    bool use_cache, size_t position_offset) {
    const auto& layer = layers_[layer_idx];
    const size_t heads = config_copy_.attention_heads;
    const size_t q_nope_dim = config_copy_.qk_nope_head_dim;
    const size_t q_rope_dim = config_copy_.qk_rope_head_dim;
    const size_t q_head_dim = q_nope_dim + q_rope_dim;
    const size_t kv_rank = config_copy_.kv_lora_rank;
    const size_t v_dim = config_copy_.v_head_dim;

    size_t q = gb.matmul(normalized_input, layer.q_a, true, ComputeBackend::CPU);
    q = gb.rms_norm(q, layer.q_a_norm, config_copy_.layer_norm_eps);
    q = gb.matmul(q, layer.q_b, true, ComputeBackend::CPU);
    q = gb.reshape(q, {1, seq_len, heads, q_head_dim});
    size_t q_nope = gb.slice(q, 3, 0, q_nope_dim);
    size_t q_pe = gb.slice(q, 3, q_nope_dim, q_rope_dim);
    q_pe = gb.kimi_yarn_rope(q_pe, config_copy_.rope_theta, position_offset, config_copy_.rope_scaling_factor,
                             config_copy_.rope_original_max_position_embeddings,
                             config_copy_.rope_yarn_beta_fast, config_copy_.rope_yarn_beta_slow,
                             config_copy_.rope_yarn_mscale, config_copy_.rope_mscale_all_dim,
                             ComputeBackend::CPU);

    size_t kv_a = gb.matmul(normalized_input, layer.kv_a, true, ComputeBackend::CPU);
    size_t compressed_kv = gb.slice(kv_a, 1, 0, kv_rank);
    size_t k_pe = gb.slice(kv_a, 1, kv_rank, q_rope_dim);
    k_pe = gb.reshape(k_pe, {1, seq_len, 1, q_rope_dim});
    k_pe = gb.kimi_yarn_rope(k_pe, config_copy_.rope_theta, position_offset, config_copy_.rope_scaling_factor,
                             config_copy_.rope_original_max_position_embeddings,
                             config_copy_.rope_yarn_beta_fast, config_copy_.rope_yarn_beta_slow,
                             config_copy_.rope_yarn_mscale, config_copy_.rope_mscale_all_dim,
                             ComputeBackend::CPU);

    compressed_kv = gb.rms_norm(compressed_kv, layer.kv_a_norm, config_copy_.layer_norm_eps);
    size_t kv = gb.matmul(compressed_kv, layer.kv_b, true, ComputeBackend::CPU);
    kv = gb.reshape(kv, {1, seq_len, heads, q_nope_dim + v_dim});
    size_t k_nope = gb.slice(kv, 3, 0, q_nope_dim);
    size_t value = gb.slice(kv, 3, q_nope_dim, v_dim);

    std::vector<size_t> repeated_k_pe(heads, k_pe);
    size_t k_pe_all_heads = gb.cat(repeated_k_pe, 2);
    size_t query = gb.cat({q_nope, q_pe}, 3);
    size_t key = gb.cat({k_nope, k_pe_all_heads}, 3);
    size_t attn = 0;
    if (use_cache) {
        gb.kv_cache_append(key, cache_nodes_[layer_idx].key, context_size_, 0);
        gb.kv_cache_append(value, cache_nodes_[layer_idx].value, context_size_, 0);
        attn = gb.attention_cached(query, key, value,
                                   cache_nodes_[layer_idx].key,
                                   cache_nodes_[layer_idx].value,
                                   attention_softmax_scale(), position_offset,
                                   context_size_, v_dim);
    } else {
        attn = gb.attention(query, key, value, attention_softmax_scale(), true, ComputeBackend::CPU);
    }
    attn = gb.reshape(attn, {seq_len, heads * v_dim});
    return gb.matmul(attn, layer.o, true, ComputeBackend::CPU);
}

size_t KimiK2Model::build_dense_mlp(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx) {
    const auto& dense = layers_[layer_idx].dense;
    size_t gate = gb.matmul(normalized_input, dense.gate, true, ComputeBackend::CPU);
    gate = gb.silu(gate);
    size_t up = gb.matmul(normalized_input, dense.up, true, ComputeBackend::CPU);
    size_t product = gb.multiply(gate, up);
    return gb.matmul(product, dense.down, true, ComputeBackend::CPU);
}

size_t KimiK2Model::build_moe(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx) {
    const auto& moe = layers_[layer_idx].moe;
    size_t router_logits = gb.matmul(normalized_input, moe.router, true, ComputeBackend::CPU);
    size_t routing_probs = gb.sigmoid(router_logits);
    size_t choice_scores = gb.add(routing_probs, moe.router_bias);
    size_t topk = gb.topk(choice_scores, config_copy_.num_experts_per_tok);
    size_t topk_indices = gb.index(topk, 0, 0);
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr && layer_idx < 4) {
        gb.register_debug_node(layer_idx, "kimi_moe_topk_indices", topk_indices);
    }

    std::vector<size_t> gate_weights;
    std::vector<size_t> up_weights;
    std::vector<size_t> down_weights;
    gate_weights.reserve(config_copy_.num_experts);
    up_weights.reserve(config_copy_.num_experts);
    down_weights.reserve(config_copy_.num_experts);
    for (const auto& expert : moe.experts) {
        gate_weights.push_back(expert.gate);
        up_weights.push_back(expert.up);
        down_weights.push_back(expert.down);
    }

    size_t routed = gb.moe_layer(normalized_input, routing_probs, topk_indices,
                                gate_weights, up_weights, down_weights,
                                config_copy_.num_experts, config_copy_.num_experts_per_tok,
                                config_copy_.norm_topk_prob, 1e-20f,
                                config_copy_.routed_scaling_factor, Activation::SILU);
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr && layer_idx < 4) {
        gb.register_debug_node(layer_idx, "kimi_moe_routed", routed);
    }
    size_t shared_gate = gb.matmul(normalized_input, moe.shared.gate, true, ComputeBackend::CPU);
    shared_gate = gb.silu(shared_gate);
    size_t shared_up = gb.matmul(normalized_input, moe.shared.up, true, ComputeBackend::CPU);
    size_t shared_product = gb.multiply(shared_gate, shared_up);
    size_t shared = gb.matmul(shared_product, moe.shared.down, true, ComputeBackend::CPU);
    if (std::getenv("CACTUS_KIMI_PRINT_TOPK") != nullptr && layer_idx < 4) {
        gb.register_debug_node(layer_idx, "kimi_moe_shared", shared);
    }
    return gb.add(routed, shared);
}

float KimiK2Model::attention_softmax_scale() const {
    const float q_head_dim = static_cast<float>(config_copy_.qk_nope_head_dim + config_copy_.qk_rope_head_dim);
    float scale = 1.0f / std::sqrt(q_head_dim);
    if (config_copy_.rope_mscale_all_dim > 0.0f) {
        float mscale = 0.1f * config_copy_.rope_mscale_all_dim *
                       std::log(config_copy_.rope_scaling_factor) + 1.0f;
        scale *= mscale * mscale;
    }
    return scale;
}

bool KimiK2Model::setup_tokenizer(const std::string& model_dir) {
    std::string vocab = (fs::path(model_dir) / "vocab.txt").string();
    std::string merges = (fs::path(model_dir) / "merges.txt").string();
    std::string cfg = (fs::path(model_dir) / "tokenizer_config.txt").string();
    if (!fs::exists(cfg)) {
        cfg = (fs::path(model_dir) / "tokenizer_config.json").string();
    }
    if (!fs::exists(vocab)) {
        CACTUS_LOG_WARN("kimi_k2", "No vocab.txt found; raw token-id execution is available, but text tokenization is not");
        return true;
    }

    auto rt = load_tokenizer_runtime_config(cfg);
    bool use_bpe = rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::BPE ||
                   (rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::UNKNOWN && fs::exists(merges));
    if (use_bpe) tokenizer_ = std::make_unique<BPETokenizer>();
    else tokenizer_ = std::make_unique<SPTokenizer>();
    return tokenizer_->load_vocabulary_with_config(vocab, merges, cfg);
}

bool KimiK2Model::load_config(const std::string& model_dir, Config& out_config) const {
    const fs::path config_txt = fs::path(model_dir) / "config.txt";
    if (fs::exists(config_txt) && out_config.from_json(config_txt.string())) {
        return true;
    }

    const fs::path config_json = fs::path(model_dir) / "config.json";
    if (!fs::exists(config_json)) return false;

    std::ifstream in(config_json);
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty() || !root.is<picojson::object>()) {
        throw std::runtime_error("Kimi config.json parse failed: " + err);
    }

    const picojson::object* cfg = &root.get<picojson::object>();
    if (auto it = cfg->find("text_config"); it != cfg->end() && it->second.is<picojson::object>()) {
        cfg = &it->second.get<picojson::object>();
    }

    auto number = [&](const char* key, double fallback) {
        auto it = cfg->find(key);
        return (it != cfg->end() && it->second.is<double>()) ? it->second.get<double>() : fallback;
    };
    auto boolean = [&](const char* key, bool fallback) {
        auto it = cfg->find(key);
        return (it != cfg->end() && it->second.is<bool>()) ? it->second.get<bool>() : fallback;
    };
    auto string_value = [&](const char* key, const std::string& fallback) {
        auto it = cfg->find(key);
        return (it != cfg->end() && it->second.is<std::string>()) ? it->second.get<std::string>() : fallback;
    };
    auto u32 = [&](const char* key, uint32_t fallback) {
        return static_cast<uint32_t>(number(key, static_cast<double>(fallback)));
    };

    out_config.model_type = Config::ModelType::KIMI_K2;
    out_config.vocab_size = u32("vocab_size", out_config.vocab_size);
    out_config.bos_token_id = u32("bos_token_id", out_config.bos_token_id);
    out_config.eos_token_id = u32("eos_token_id", out_config.eos_token_id);
    out_config.num_layers = u32("num_hidden_layers", out_config.num_layers);
    out_config.hidden_dim = u32("hidden_size", out_config.hidden_dim);
    out_config.ffn_intermediate_dim = u32("intermediate_size", out_config.ffn_intermediate_dim);
    out_config.attention_heads = u32("num_attention_heads", out_config.attention_heads);
    out_config.attention_kv_heads = u32("num_key_value_heads", out_config.attention_kv_heads);
    out_config.layer_norm_eps = static_cast<float>(number("rms_norm_eps", out_config.layer_norm_eps));
    out_config.rope_theta = static_cast<float>(number("rope_theta", out_config.rope_theta));
    out_config.q_lora_rank = u32("q_lora_rank", out_config.q_lora_rank);
    out_config.kv_lora_rank = u32("kv_lora_rank", out_config.kv_lora_rank);
    out_config.qk_nope_head_dim = u32("qk_nope_head_dim", out_config.qk_nope_head_dim);
    out_config.qk_rope_head_dim = u32("qk_rope_head_dim", out_config.qk_rope_head_dim);
    out_config.qk_head_dim = out_config.qk_nope_head_dim + out_config.qk_rope_head_dim;
    out_config.attention_head_dim = out_config.qk_head_dim;
    out_config.v_head_dim = u32("v_head_dim", out_config.v_head_dim);
    out_config.max_position_embeddings = u32("max_position_embeddings", out_config.max_position_embeddings);
    out_config.num_experts = u32("n_routed_experts", out_config.num_experts);
    out_config.num_shared_experts = u32("n_shared_experts", out_config.num_shared_experts);
    out_config.num_dense_layers = u32("first_k_dense_replace", out_config.num_dense_layers);
    out_config.num_experts_per_tok = u32("num_experts_per_tok", out_config.num_experts_per_tok);
    out_config.moe_intermediate_dim = u32("moe_intermediate_size", out_config.moe_intermediate_dim);
    out_config.moe_every_n_layers = u32("moe_layer_freq", out_config.moe_every_n_layers);
    out_config.moe_n_group = u32("n_group", out_config.moe_n_group);
    out_config.moe_topk_group = u32("topk_group", out_config.moe_topk_group);
    out_config.norm_topk_prob = boolean("norm_topk_prob", out_config.norm_topk_prob);
    out_config.tie_word_embeddings = boolean("tie_word_embeddings", out_config.tie_word_embeddings);
    out_config.routed_scaling_factor = static_cast<float>(number("routed_scaling_factor", out_config.routed_scaling_factor));
    out_config.moe_topk_method = string_value("topk_method", out_config.moe_topk_method);
    out_config.moe_scoring_func = string_value("scoring_func", out_config.moe_scoring_func);
    out_config.attention_bias = boolean("attention_bias", out_config.attention_bias);

    if (auto it = cfg->find("rope_scaling"); it != cfg->end() && it->second.is<picojson::object>()) {
        const auto& rope = it->second.get<picojson::object>();
        auto rope_number = [&](const char* key, double fallback) {
            auto jt = rope.find(key);
            return (jt != rope.end() && jt->second.is<double>()) ? jt->second.get<double>() : fallback;
        };
        out_config.rope_scaling_factor = static_cast<float>(rope_number("factor", out_config.rope_scaling_factor));
        out_config.rope_yarn_beta_fast = static_cast<float>(rope_number("beta_fast", out_config.rope_yarn_beta_fast));
        out_config.rope_yarn_beta_slow = static_cast<float>(rope_number("beta_slow", out_config.rope_yarn_beta_slow));
        out_config.rope_yarn_mscale = static_cast<float>(rope_number("mscale", out_config.rope_yarn_mscale));
        out_config.rope_mscale_all_dim = static_cast<float>(rope_number("mscale_all_dim", out_config.rope_mscale_all_dim));
        out_config.rope_original_max_position_embeddings =
            static_cast<uint32_t>(rope_number("original_max_position_embeddings",
                                              out_config.rope_original_max_position_embeddings));
    }

    return true;
}

bool KimiK2Model::load_weight_manifest(const std::string& model_dir) {
    const std::vector<fs::path> candidates = {
        fs::path(model_dir) / "weights_manifest.json",
        fs::path(model_dir) / "manifest.json",
    };

    fs::path manifest_path;
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            manifest_path = candidate;
            break;
        }
    }
    if (manifest_path.empty()) return false;

    std::ifstream in(manifest_path);
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty() || !root.is<picojson::object>()) {
        throw std::runtime_error("Kimi weight manifest parse failed: " + err);
    }

    const auto& obj = root.get<picojson::object>();
    auto add_manifest_entry = [&](const picojson::object& entry) {
        auto name_it = entry.find("hf_name");
        if (name_it == entry.end() || !name_it->second.is<std::string>()) {
            name_it = entry.find("adapter_name");
        }
        auto path_it = entry.find("path");
        if (path_it == entry.end() || !path_it->second.is<std::string>()) {
            path_it = entry.find("output_name");
        }
        if (name_it == entry.end() || path_it == entry.end() ||
            !name_it->second.is<std::string>() || !path_it->second.is<std::string>()) {
            return;
        }
        std::string logical = name_it->second.get<std::string>();
        const std::string prefix = "language_model.";
        if (logical.rfind(prefix, 0) == 0) {
            logical = logical.substr(prefix.size());
        }
        weight_manifest_[logical] = path_it->second.get<std::string>();
    };
    auto read_map_object = [&](const picojson::object& values) {
        for (const auto& [name, value] : values) {
            if (value.is<std::string>()) {
                weight_manifest_[name] = value.get<std::string>();
            } else if (value.is<picojson::object>()) {
                const auto& entry = value.get<picojson::object>();
                auto it = entry.find("path");
                if (it != entry.end() && it->second.is<std::string>()) {
                    weight_manifest_[name] = it->second.get<std::string>();
                }
            }
        }
    };

    if (auto it = obj.find("weights"); it != obj.end() && it->second.is<picojson::array>()) {
        for (const auto& value : it->second.get<picojson::array>()) {
            if (value.is<picojson::object>()) add_manifest_entry(value.get<picojson::object>());
        }
    } else if (auto it = obj.find("weights"); it != obj.end() && it->second.is<picojson::object>()) {
        read_map_object(it->second.get<picojson::object>());
    } else if (auto it = obj.find("tensors"); it != obj.end() && it->second.is<picojson::object>()) {
        read_map_object(it->second.get<picojson::object>());
    } else {
        read_map_object(obj);
    }

    return !weight_manifest_.empty();
}

void KimiK2Model::validate_architecture() const {
    if (config_copy_.model_type != Config::ModelType::KIMI_K2) {
        throw std::runtime_error("KimiK2Model requires model_type=kimi_k2");
    }
    if (config_copy_.q_lora_rank == 0 || config_copy_.kv_lora_rank == 0 ||
        config_copy_.qk_nope_head_dim == 0 || config_copy_.qk_rope_head_dim == 0 ||
        config_copy_.v_head_dim == 0) {
        throw std::runtime_error("KimiK2Model requires MLA q/kv LoRA ranks and qk/v head dims");
    }
    if (config_copy_.moe_scoring_func != "sigmoid") {
        throw std::runtime_error("KimiK2Model implements only Moonshot Kimi sigmoid MoE scoring");
    }
    if (config_copy_.moe_topk_method != "noaux_tc") {
        throw std::runtime_error("KimiK2Model implements only Moonshot Kimi noaux_tc routing");
    }
    if (config_copy_.moe_n_group != 1 || config_copy_.moe_topk_group != 1) {
        throw std::runtime_error("KimiK2Model text config requires n_group=1 and topk_group=1");
    }
    if (!config_copy_.norm_topk_prob) {
        throw std::runtime_error("KimiK2Model requires norm_topk_prob=true");
    }
    if (config_copy_.num_dense_layers != 1) {
        throw std::runtime_error("KimiK2Model requires first_k_dense_replace=1");
    }
}

void KimiK2Model::validate_required_weights() const {
    std::vector<std::string> required = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    };

    for (uint32_t i = 0; i < config_copy_.num_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        required.push_back(prefix + "input_layernorm.weight");
        required.push_back(prefix + "self_attn.q_a_proj.weight");
        required.push_back(prefix + "self_attn.q_a_layernorm.weight");
        required.push_back(prefix + "self_attn.q_b_proj.weight");
        required.push_back(prefix + "self_attn.kv_a_proj_with_mqa.weight");
        required.push_back(prefix + "self_attn.kv_a_layernorm.weight");
        required.push_back(prefix + "self_attn.kv_b_proj.weight");
        required.push_back(prefix + "self_attn.o_proj.weight");
        required.push_back(prefix + "post_attention_layernorm.weight");

        if (i < config_copy_.num_dense_layers) {
            required.push_back(prefix + "mlp.gate_proj.weight");
            required.push_back(prefix + "mlp.up_proj.weight");
            required.push_back(prefix + "mlp.down_proj.weight");
        } else {
            required.push_back(prefix + "mlp.gate.weight");
            required.push_back(prefix + "mlp.gate.e_score_correction_bias");
            for (uint32_t e = 0; e < config_copy_.num_experts; ++e) {
                const std::string ep = prefix + "mlp.experts." + std::to_string(e) + ".";
                required.push_back(ep + "gate_proj.weight");
                required.push_back(ep + "up_proj.weight");
                required.push_back(ep + "down_proj.weight");
            }
            required.push_back(prefix + "mlp.shared_experts.gate_proj.weight");
            required.push_back(prefix + "mlp.shared_experts.up_proj.weight");
            required.push_back(prefix + "mlp.shared_experts.down_proj.weight");
        }
    }

    std::vector<std::string> missing_manifest;
    std::vector<std::string> missing_files;
    for (const auto& logical_name : required) {
        auto it = weight_manifest_.find(logical_name);
        if (it == weight_manifest_.end()) {
            missing_manifest.push_back(logical_name);
            continue;
        }
        fs::path p(it->second);
        if (!p.is_absolute()) p = fs::path(model_dir_) / p;
        if (!fs::exists(p)) {
            missing_files.push_back(logical_name + " -> " + p.string());
        }
    }

    if (missing_manifest.empty() && missing_files.empty()) return;

    std::ostringstream message;
    message << "Kimi weight bundle is incomplete:";
    if (!missing_manifest.empty()) {
        message << " missing manifest entries [";
        const size_t limit = std::min<size_t>(missing_manifest.size(), 16);
        for (size_t i = 0; i < limit; ++i) {
            if (i > 0) message << ", ";
            message << missing_manifest[i];
        }
        if (missing_manifest.size() > limit) {
            message << ", ... +" << (missing_manifest.size() - limit) << " more";
        }
        message << "]";
    }
    if (!missing_files.empty()) {
        message << " missing files [";
        const size_t limit = std::min<size_t>(missing_files.size(), 16);
        for (size_t i = 0; i < limit; ++i) {
            if (i > 0) message << ", ";
            message << missing_files[i];
        }
        if (missing_files.size() > limit) {
            message << ", ... +" << (missing_files.size() - limit) << " more";
        }
        message << "]";
    }
    throw std::runtime_error(message.str());
}

std::string KimiK2Model::weight_path(const std::string& logical_name) const {
    auto it = weight_manifest_.find(logical_name);
    if (it == weight_manifest_.end()) {
        throw std::runtime_error("Missing Kimi tensor in manifest: " + logical_name);
    }
    fs::path p(it->second);
    if (p.is_absolute()) return p.string();
    return (fs::path(model_dir_) / p).string();
}

} // namespace engine
} // namespace cactus
