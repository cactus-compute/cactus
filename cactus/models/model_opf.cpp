#include "model.h"
#include "../graph/graph.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cactus {
namespace engine {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

char parse_boundary(const std::string& name, std::string& out_span) {
    if (name == "O") { out_span.clear(); return 'O'; }
    auto dash = name.find('-');
    if (dash == std::string::npos || dash == 0) { out_span.clear(); return 'O'; }
    out_span = name.substr(dash + 1);
    return name[0];
}

}

OPFModel::OPFModel() : Model() {}

OPFModel::OPFModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
    for (auto& layer : weight_nodes_.layers) {
        layer.expert_w1_weight.assign(config.num_experts, 0);
        layer.expert_w3_weight.assign(config.num_experts, 0);
        layer.expert_w2_weight.assign(config.num_experts, 0);
        layer.expert_w1_bias.assign(config.num_experts, 0);
        layer.expert_w3_bias.assign(config.num_experts, 0);
        layer.expert_w2_bias.assign(config.num_experts, 0);
    }
}

void OPFModel::post_init() {
    const size_t n = config_.label_names.size();
    if (n == 0 || n != config_.num_labels) {
        throw std::runtime_error("OPFModel: label_names is empty or disagrees with num_labels");
    }
    auto& t = label_tables_;
    t.class_names = config_.label_names;
    t.token_to_span_label.assign(n, 0);
    t.token_boundary_tag.assign(n, 'O');
    t.span_class_names = {"O"};
    t.background_token_idx = -1;

    for (size_t i = 0; i < n; ++i) {
        std::string span;
        const char tag = parse_boundary(t.class_names[i], span);
        if (tag == 'O') {
            if (t.background_token_idx < 0) t.background_token_idx = static_cast<int>(i);
            continue;
        }
        int span_idx = -1;
        for (size_t s = 0; s < t.span_class_names.size(); ++s) {
            if (t.span_class_names[s] == span) { span_idx = static_cast<int>(s); break; }
        }
        if (span_idx < 0) {
            span_idx = static_cast<int>(t.span_class_names.size());
            t.span_class_names.push_back(span);
        }
        t.token_to_span_label[i] = span_idx;
        t.token_boundary_tag[i] = tag;
    }
    if (t.background_token_idx < 0) {
        throw std::runtime_error("OPFModel: label list must include 'O' (background)");
    }
    rebuild_crf_tables();
}

void OPFModel::rebuild_crf_tables() {
    const auto& t = label_tables_;
    const size_t n = t.class_names.size();
    crf_start_scores_.assign(n, kNegInf);
    crf_end_scores_.assign(n, kNegInf);
    crf_transition_scores_.assign(n * n, kNegInf);

    const int bg = t.background_token_idx;
    auto can_start = [&](size_t i, char tag) { return static_cast<int>(i) == bg || tag == 'B' || tag == 'S'; };
    auto can_end   = [&](size_t i, char tag) { return static_cast<int>(i) == bg || tag == 'E' || tag == 'S'; };

    for (size_t i = 0; i < n; ++i) {
        const char tag_i = t.token_boundary_tag[i];
        if (can_start(i, tag_i)) crf_start_scores_[i] = 0.0f;
        if (can_end(i, tag_i))   crf_end_scores_[i]   = 0.0f;

        const bool i_starts_or_ends_span = (static_cast<int>(i) == bg || tag_i == 'E' || tag_i == 'S');
        const bool i_inside_span = (tag_i == 'B' || tag_i == 'I');

        for (size_t j = 0; j < n; ++j) {
            const char tag_j = t.token_boundary_tag[j];
            const bool j_is_bg = (static_cast<int>(j) == bg);
            bool allowed = false;
            if (i_starts_or_ends_span) {
                allowed = j_is_bg || tag_j == 'B' || tag_j == 'S';
            } else if (i_inside_span) {
                allowed = (t.token_to_span_label[i] == t.token_to_span_label[j]) &&
                          (tag_j == 'I' || tag_j == 'E');
            }
            if (allowed) crf_transition_scores_[i * n + j] = 0.0f;
        }
    }
}

void OPFModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    weight_nodes_.classifier_weight  = gb->mmap_weights(model_folder_path_ + "/classifier.weights");
    weight_nodes_.classifier_bias    = gb->mmap_weights(model_folder_path_ + "/classifier.bias");

    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        auto& layer = weight_nodes_.layers[i];
        const std::string p = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        layer.input_norm_weight     = gb->mmap_weights(p + "input_norm.weights");
        layer.post_attn_norm_weight = gb->mmap_weights(p + "post_attn_norm.weights");
        layer.attn_q_weight         = gb->mmap_weights(p + "attn_q.weights");
        layer.attn_k_weight         = gb->mmap_weights(p + "attn_k.weights");
        layer.attn_v_weight         = gb->mmap_weights(p + "attn_v.weights");
        layer.attn_output_weight    = gb->mmap_weights(p + "attn_output.weights");
        layer.attn_q_bias           = gb->mmap_weights(p + "attn_q.bias");
        layer.attn_k_bias           = gb->mmap_weights(p + "attn_k.bias");
        layer.attn_v_bias           = gb->mmap_weights(p + "attn_v.bias");
        layer.attn_output_bias      = gb->mmap_weights(p + "attn_output.bias");
        layer.attn_sinks            = gb->mmap_weights(p + "attn_sinks.weights");
        layer.moe_router_weight     = gb->mmap_weights(p + "moe_router.weights");
        layer.moe_router_bias       = gb->mmap_weights(p + "moe_router.bias");

        for (uint32_t e = 0; e < config_.num_experts; ++e) {
            const std::string ep = p + "moe_expert_" + std::to_string(e) + "_";
            layer.expert_w1_weight[e] = gb->mmap_weights(ep + "w1.weights");
            layer.expert_w3_weight[e] = gb->mmap_weights(ep + "w3.weights");
            layer.expert_w2_weight[e] = gb->mmap_weights(ep + "w2.weights");
            layer.expert_w1_bias[e]   = gb->mmap_weights(ep + "w1.bias");
            layer.expert_w3_bias[e]   = gb->mmap_weights(ep + "w3.bias");
            layer.expert_w2_bias[e]   = gb->mmap_weights(ep + "w2.bias");
        }
    }
}

size_t OPFModel::build_bidirectional_band_mask(CactusGraph* gb, size_t seq_len) const {
    const size_t w = config_.sliding_window;
    std::vector<__fp16> buf(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            const size_t d = i > j ? i - j : j - i;
            buf[i * seq_len + j] = static_cast<__fp16>(d <= w ? 0.0f : kNegInf);
        }
    }
    const size_t node = gb->input({1, seq_len, seq_len}, Precision::FP16);
    gb->set_input(node, buf.data(), Precision::FP16);
    return node;
}

size_t OPFModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) throw std::runtime_error("OPFModel is an encoder; KV cache is not supported");
    (void)position_offset;

    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q = gb->add(gb->matmul(normalized_input, layer.attn_q_weight, true, backend), layer.attn_q_bias);
    auto k = gb->add(gb->matmul(normalized_input, layer.attn_k_weight, true, backend), layer.attn_k_bias);
    auto v = gb->add(gb->matmul(normalized_input, layer.attn_v_weight, true, backend), layer.attn_v_bias);

    const size_t seq_len = gb->get_output_buffer(q).shape[0];
    const size_t num_q = config_.attention_heads;
    const size_t num_kv = config_.attention_kv_heads;
    const size_t head_dim = config_.attention_head_dim;

    auto q4 = gb->reshape(q, {1, seq_len, num_q, head_dim});
    auto k4 = gb->reshape(k, {1, seq_len, num_kv, head_dim});
    auto v4 = gb->reshape(v, {1, seq_len, num_kv, head_dim});

    if (config_.rope_theta > 0) {
        const bool use_yarn = config_.rope_scaling_factor > 1.0f && config_.rope_original_ctx > 0;
        auto rope = [&](size_t x) {
            return use_yarn
                ? gb->rope_gptj_yarn(x, config_.rope_theta, config_.rope_scaling_factor,
                                     config_.rope_beta_fast, config_.rope_beta_slow,
                                     config_.rope_original_ctx, 0, head_dim, backend)
                : gb->rope_gptj(x, config_.rope_theta, 0, head_dim, backend);
        };
        q4 = rope(q4);
        k4 = rope(k4);
    }

    auto attn = gb->attention_masked(q4, k4, v4,
                                     build_bidirectional_band_mask(gb, seq_len),
                                     attention_scale_, false, backend, true, 0, 0, 0.0f,
                                     layer.attn_sinks);
    auto out = gb->matmul(gb->reshape(attn, {seq_len, num_q * head_dim}),
                          layer.attn_output_weight, true, backend);
    return gb->add(out, layer.attn_output_bias);
}

size_t OPFModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    const float top_k_f = static_cast<float>(config_.num_experts_per_tok);

    auto logits = gb->add(gb->matmul(normalized_h, layer.moe_router_weight, true, backend),
                          layer.moe_router_bias);
    auto topk = gb->topk(logits, config_.num_experts_per_tok);
    auto topk_idx = gb->index(topk, 0, 0);
    auto topk_probs = gb->scalar_divide(
        gb->softmax(gb->precision_cast(gb->index(topk, 1, 0), Precision::FP16)), top_k_f);
    auto routing = gb->scatter_topk(topk_idx, topk_probs, config_.num_experts);

    auto moe = gb->moe_layer_openai(
        normalized_h, routing, topk_idx,
        layer.expert_w1_weight, layer.expert_w3_weight, layer.expert_w2_weight,
        layer.expert_w1_bias,   layer.expert_w3_bias,   layer.expert_w2_bias,
        config_.num_experts, config_.num_experts_per_tok);
    return gb->scalar_multiply(moe, top_k_f);
}

size_t OPFModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    (void)use_cache; (void)position_offset;
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto after_attn = gb->add(hidden,
                              build_attention(gb, gb->rms_norm(hidden, layer.input_norm_weight, config_.layer_norm_eps),
                                              layer_idx, backend));
    return gb->add(after_attn,
                   build_mlp(gb, gb->rms_norm(after_attn, layer.post_attn_norm_weight, config_.layer_norm_eps),
                             layer_idx, backend));
}

size_t OPFModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (use_cache) throw std::runtime_error("OPFModel is an encoder; KV cache is not supported");
    if (tokens.empty()) throw std::runtime_error("OPFModel::forward: empty token sequence");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    const auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    const size_t seq_len = tokens.size();
    auto input_ids = gb->input({seq_len}, Precision::FP32);
    std::vector<float> id_buf(tokens.begin(), tokens.end());
    gb->set_input(input_ids, id_buf.data(), Precision::FP32);

    auto hidden = gb->embedding(embedding_node_id_, input_ids);
    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        hidden = build_transformer_block(gb, hidden, i, backend);
    }
    hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    return gb->add(gb->matmul(hidden, weight_nodes_.classifier_weight, true, backend),
                   weight_nodes_.classifier_bias);
}

std::vector<int> OPFModel::viterbi_decode(const float* logits, size_t seq_len, size_t num_classes) const {
    std::vector<int> path(seq_len, 0);
    if (seq_len == 0) return path;

    const float* trans = crf_transition_scores_.data();
    std::vector<float> scores(num_classes), next_scores(num_classes);
    std::vector<int> backptr((seq_len - 1) * num_classes, 0);

    for (size_t c = 0; c < num_classes; ++c) scores[c] = logits[c] + crf_start_scores_[c];

    for (size_t t = 1; t < seq_len; ++t) {
        const float* emit = logits + t * num_classes;
        for (size_t j = 0; j < num_classes; ++j) {
            float best = kNegInf;
            int best_prev = 0;
            for (size_t i = 0; i < num_classes; ++i) {
                const float cand = scores[i] + trans[i * num_classes + j];
                if (cand > best) { best = cand; best_prev = static_cast<int>(i); }
            }
            next_scores[j] = best + emit[j];
            backptr[(t - 1) * num_classes + j] = best_prev;
        }
        scores.swap(next_scores);
    }

    bool any_finite = false;
    for (float s : scores) if (std::isfinite(s)) { any_finite = true; break; }
    if (!any_finite) {
        for (size_t t = 0; t < seq_len; ++t) {
            const float* row = logits + t * num_classes;
            path[t] = static_cast<int>(std::max_element(row, row + num_classes) - row);
        }
        return path;
    }

    size_t last = 0;
    float best = kNegInf;
    for (size_t c = 0; c < num_classes; ++c) {
        const float v = scores[c] + crf_end_scores_[c];
        if (v > best) { best = v; last = c; }
    }
    path[seq_len - 1] = static_cast<int>(last);
    for (size_t t = seq_len - 1; t > 0; --t) {
        last = backptr[(t - 1) * num_classes + last];
        path[t - 1] = static_cast<int>(last);
    }
    return path;
}

std::vector<OPFModel::Span> OPFModel::classify(const std::string& text) {
    if (!initialized_ || !graph_handle_) throw std::runtime_error("OPFModel::classify called before init()");
    if (!tokenizer_) throw std::runtime_error("OPFModel::classify: tokenizer not loaded");

    const auto tokens = tokenizer_->encode(text);
    if (tokens.empty()) return {};

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const size_t logits_node = forward(tokens, false);
    gb->execute();

    const auto& buf = gb->get_output_buffer(logits_node);
    if (buf.shape.size() != 2 || buf.shape[1] != config_.num_labels) {
        throw std::runtime_error("OPFModel::classify: unexpected logits shape");
    }
    const size_t seq_len = buf.shape[0];
    const size_t num_classes = buf.shape[1];

    const auto* raw = reinterpret_cast<const __fp16*>(gb->get_output(logits_node));
    std::vector<float> logits(seq_len * num_classes);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = static_cast<float>(raw[i]);

    const auto path = viterbi_decode(logits.data(), seq_len, num_classes);

    std::vector<Span> spans;
    const auto& t = label_tables_;
    for (size_t i = 0; i < path.size(); ) {
        const int cls = path[i];
        const char tag = t.token_boundary_tag[cls];
        const int span_idx = t.token_to_span_label[cls];
        if (span_idx == 0) { ++i; continue; }

        if (tag == 'S') {
            spans.push_back({i, i + 1, t.span_class_names[span_idx]});
            ++i;
        } else if (tag == 'B') {
            size_t j = i + 1;
            while (j < path.size() && t.token_to_span_label[path[j]] == span_idx) {
                const char tj = t.token_boundary_tag[path[j]];
                if (tj == 'E') { ++j; break; }
                if (tj != 'I') break;
                ++j;
            }
            spans.push_back({i, j, t.span_class_names[span_idx]});
            i = j;
        } else {
            ++i;
        }
    }
    return spans;
}

}
}
