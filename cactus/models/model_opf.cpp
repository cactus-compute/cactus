// OpenAI privacy-filter (openai/privacy-filter): bidirectional sliding-window
// encoder with GQA + per-head attention sinks + MoE FFN + BIOES token-classifier
// head, decoded with a constrained-Viterbi CRF over BIOES transitions.
//
// Parity gaps vs HF reference (documented; see TODOs below):
//   - Attention sinks are dropped (softmax without the extra sink column).
//   - MoE activation is SiLU SwiGLU; HF uses quick-GELU GLU with
//     gate=min(gate,7), up=clamp(up,-7,7), and a `(up + 1)` bias. Per-expert
//     biases on gate/up/down are not threaded through `moe_layer` yet.
//   - RoPE is GPT-J interleaved base RoPE; HF uses YaRN-scaled RoPE.
// The structure here lets those be dropped in one-at-a-time without refactor.

#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

namespace {

constexpr float kNegInf = -1e9f;

char boundary_tag_of(const std::string& name, std::string& out_span) {
    if (name == "O") { out_span.clear(); return 'O'; }
    auto dash = name.find('-');
    if (dash == std::string::npos || dash == 0) { out_span.clear(); return 'O'; }
    out_span = name.substr(dash + 1);
    return name[0];
}

}  // namespace

OPFModel::OPFModel() : Model() {}

OPFModel::OPFModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
    for (auto& layer : weight_nodes_.layers) {
        layer.expert_w1_weight.resize(config.num_experts, 0);
        layer.expert_w3_weight.resize(config.num_experts, 0);
        layer.expert_w2_weight.resize(config.num_experts, 0);
        layer.expert_w1_bias.resize(config.num_experts, 0);
        layer.expert_w3_bias.resize(config.num_experts, 0);
        layer.expert_w2_bias.resize(config.num_experts, 0);
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
        const std::string& name = t.class_names[i];
        std::string span;
        char tag = boundary_tag_of(name, span);
        if (tag == 'O') {
            if (t.background_token_idx < 0) t.background_token_idx = static_cast<int>(i);
            t.token_to_span_label[i] = 0;
            t.token_boundary_tag[i] = 'O';
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

    auto is_bg = [&](size_t i) { return static_cast<int>(i) == t.background_token_idx; };

    for (size_t i = 0; i < n; ++i) {
        char tag = t.token_boundary_tag[i];
        // A valid path starts/ends on background, a single-token span, or a span boundary tag.
        if (is_bg(i) || tag == 'B' || tag == 'S') crf_start_scores_[i] = 0.0f;
        if (is_bg(i) || tag == 'E' || tag == 'S') crf_end_scores_[i] = 0.0f;

        for (size_t j = 0; j < n; ++j) {
            char next_tag = t.token_boundary_tag[j];
            bool prev_bg = is_bg(i);
            bool next_bg = is_bg(j);
            bool allowed = false;
            if (prev_bg) {
                allowed = next_bg || next_tag == 'B' || next_tag == 'S';
            } else if (tag == 'E' || tag == 'S') {
                allowed = next_bg || next_tag == 'B' || next_tag == 'S';
            } else if (tag == 'B' || tag == 'I') {
                // Inside a span: must stay on the same span label, and can only
                // continue with I or close with E.
                allowed = (t.token_to_span_label[i] == t.token_to_span_label[j]) &&
                          (next_tag == 'I' || next_tag == 'E');
            }
            if (allowed) crf_transition_scores_[i * n + j] = 0.0f;
        }
    }
    // TODO: plumb runtime-configurable Viterbi bias knobs
    // (viterbi_calibration.json) into the allowed edges once we have a
    // deployment reason to shift precision/recall operating points.
}

void OPFModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    weight_nodes_.classifier_weight = gb->mmap_weights(model_folder_path_ + "/classifier.weights");
    weight_nodes_.classifier_bias = gb->mmap_weights(model_folder_path_ + "/classifier.bias");

    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        auto& layer = weight_nodes_.layers[i];
        std::string prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        layer.input_norm_weight = gb->mmap_weights(prefix + "input_norm.weights");
        layer.post_attn_norm_weight = gb->mmap_weights(prefix + "post_attn_norm.weights");

        layer.attn_q_weight = gb->mmap_weights(prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(prefix + "attn_v.weights");
        layer.attn_output_weight = gb->mmap_weights(prefix + "attn_output.weights");
        layer.attn_q_bias = gb->mmap_weights(prefix + "attn_q.bias");
        layer.attn_k_bias = gb->mmap_weights(prefix + "attn_k.bias");
        layer.attn_v_bias = gb->mmap_weights(prefix + "attn_v.bias");
        layer.attn_output_bias = gb->mmap_weights(prefix + "attn_output.bias");
        layer.attn_sinks = gb->mmap_weights(prefix + "attn_sinks.weights");

        layer.moe_router_weight = gb->mmap_weights(prefix + "moe_router.weights");
        layer.moe_router_bias = gb->mmap_weights(prefix + "moe_router.bias");

        for (uint32_t e = 0; e < config_.num_experts; ++e) {
            std::string ep = prefix + "moe_expert_" + std::to_string(e) + "_";
            layer.expert_w1_weight[e] = gb->mmap_weights(ep + "w1.weights");
            layer.expert_w3_weight[e] = gb->mmap_weights(ep + "w3.weights");
            layer.expert_w2_weight[e] = gb->mmap_weights(ep + "w2.weights");
            layer.expert_w1_bias[e] = gb->mmap_weights(ep + "w1.bias");
            layer.expert_w3_bias[e] = gb->mmap_weights(ep + "w3.bias");
            layer.expert_w2_bias[e] = gb->mmap_weights(ep + "w2.bias");
        }
    }
}

size_t OPFModel::build_bidirectional_band_mask(CactusGraph* gb, size_t seq_len) const {
    // Additive mask [1, T, T]; mask[0, i, j] = 0 if |i-j| <= sliding_window else -inf.
    // attention_masked accepts rank-3 [B, T, S] masks (broadcast across heads).
    const size_t w = config_.sliding_window;
    std::vector<__fp16> buf(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            size_t d = i > j ? i - j : j - i;
            buf[i * seq_len + j] = (d <= w) ? static_cast<__fp16>(0.0f) : static_cast<__fp16>(kNegInf);
        }
    }
    size_t node = gb->input({1, seq_len, seq_len}, Precision::FP16);
    gb->set_input(node, buf.data(), Precision::FP16);
    return node;
}

size_t OPFModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("OPFModel is an encoder; KV cache is not supported");
    }
    (void)position_offset;

    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    q = gb->add(q, layer.attn_q_bias);
    auto k = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    k = gb->add(k, layer.attn_k_bias);
    auto v = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    v = gb->add(v, layer.attn_v_bias);

    const size_t seq_len = gb->get_output_buffer(q).shape[0];
    const size_t num_q = config_.attention_heads;
    const size_t num_kv = config_.attention_kv_heads;
    const size_t head_dim = config_.attention_head_dim;

    auto q4 = gb->reshape(q, {1, seq_len, num_q, head_dim});
    auto k4 = gb->reshape(k, {1, seq_len, num_kv, head_dim});
    auto v4 = gb->reshape(v, {1, seq_len, num_kv, head_dim});

    // OPF uses YaRN-scaled GPT-J-layout RoPE. `rope_gptj_yarn` falls back to
    // base RoPE when `scaling_factor <= 1`, matching the reference's behavior
    // in the non-scaled regime.
    if (config_.rope_theta > 0) {
        if (config_.rope_scaling_factor > 1.0f && config_.rope_original_ctx > 0) {
            q4 = gb->rope_gptj_yarn(q4, config_.rope_theta, config_.rope_scaling_factor,
                                     config_.rope_beta_fast, config_.rope_beta_slow,
                                     config_.rope_original_ctx, 0, head_dim, backend);
            k4 = gb->rope_gptj_yarn(k4, config_.rope_theta, config_.rope_scaling_factor,
                                     config_.rope_beta_fast, config_.rope_beta_slow,
                                     config_.rope_original_ctx, 0, head_dim, backend);
        } else {
            q4 = gb->rope_gptj(q4, config_.rope_theta, 0, head_dim, backend);
            k4 = gb->rope_gptj(k4, config_.rope_theta, 0, head_dim, backend);
        }
    }

    auto mask = build_bidirectional_band_mask(gb, seq_len);
    auto attn = gb->attention_masked_with_sinks(q4, k4, v4, mask, layer.attn_sinks,
                                                attention_scale_,
                                                /*is_causal=*/false, backend,
                                                /*additive_mask=*/true);

    auto attn_flat = gb->reshape(attn, {seq_len, num_q * head_dim});
    auto out = gb->matmul(attn_flat, layer.attn_output_weight, true, backend);
    out = gb->add(out, layer.attn_output_bias);
    return out;
}

size_t OPFModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    // Router: logits = x @ W^T + b, then softmax over the topk subset (HF pattern).
    auto router_logits = gb->matmul(normalized_h, layer.moe_router_weight, true, backend);
    router_logits = gb->add(router_logits, layer.moe_router_bias);
    auto topk = gb->topk(router_logits, config_.num_experts_per_tok);
    auto topk_idx = gb->index(topk, 0, 0);
    auto topk_vals = gb->index(topk, 1, 0);
    // topk emits FP32; softmax kernel requires FP16 input. Cast down, softmax,
    // divide by top_k (HF balanced-average convention), then cast back to FP32
    // so scatter_topk accepts it as values alongside the FP32 indices.
    auto topk_vals_fp16 = gb->precision_cast(topk_vals, Precision::FP16);
    auto topk_probs_fp16 = gb->softmax(topk_vals_fp16);
    topk_probs_fp16 = gb->scalar_divide(topk_probs_fp16, static_cast<float>(config_.num_experts_per_tok));
    auto topk_probs = gb->precision_cast(topk_probs_fp16, Precision::FP32);

    // `moe_layer_openai` reads normalized top-k weights from a full [T, E]
    // routing tensor. `scatter_topk` conveniently scatters our top-k weights
    // into a dense grid, but its declared output layout is [E, T] — the kernel
    // wants [T, E], so transpose.
    auto routing_sparse_et = gb->scatter_topk(topk_idx, topk_probs, config_.num_experts);
    // scatter_topk emits FP32; cast to FP16 so the transpose kernel (FP16-only)
    // and downstream moe_layer can consume it.
    auto routing_sparse_et_fp16 = gb->precision_cast(routing_sparse_et, Precision::FP16);
    auto routing_full = gb->transpose(routing_sparse_et_fp16, backend);

    auto moe = gb->moe_layer_openai(
        normalized_h, routing_full, topk_idx,
        layer.expert_w1_weight, layer.expert_w3_weight, layer.expert_w2_weight,
        layer.expert_w1_bias, layer.expert_w3_bias, layer.expert_w2_bias,
        config_.num_experts, config_.num_experts_per_tok,
        config_.swiglu_alpha, config_.swiglu_limit);

    // HF multiplies the MoE output by top_k after the weighted sum.
    moe = gb->scalar_multiply(moe, static_cast<float>(config_.num_experts_per_tok));
    return moe;
}

size_t OPFModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("OPFModel is an encoder; KV cache is not supported");
    }
    (void)position_offset;
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto normed = gb->rms_norm(hidden, layer.input_norm_weight, config_.layer_norm_eps);
    auto attn = build_attention(gb, normed, layer_idx, backend);
    auto after_attn = gb->add(hidden, attn);

    auto normed2 = gb->rms_norm(after_attn, layer.post_attn_norm_weight, config_.layer_norm_eps);
    auto mlp = build_mlp(gb, normed2, layer_idx, backend);
    return gb->add(after_attn, mlp);
}

size_t OPFModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (use_cache) {
        throw std::runtime_error("OPFModel is an encoder; KV cache is not supported");
    }
    if (tokens.empty()) {
        throw std::runtime_error("OPFModel::forward: empty token sequence");
    }
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    const size_t seq_len = tokens.size();
    auto input_ids = gb->input({seq_len}, Precision::FP32);
    std::vector<float> id_buf(seq_len);
    for (size_t i = 0; i < seq_len; ++i) id_buf[i] = static_cast<float>(tokens[i]);
    gb->set_input(input_ids, id_buf.data(), Precision::FP32);

    auto hidden = gb->embedding(embedding_node_id_, input_ids);
    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        hidden = build_transformer_block(gb, hidden, i, backend);
    }
    hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);

    // Classifier head: [T, hidden] @ [num_labels, hidden]^T + [num_labels].
    auto logits = gb->matmul(hidden, weight_nodes_.classifier_weight, true, backend);
    logits = gb->add(logits, weight_nodes_.classifier_bias);
    return logits;
}

std::vector<int> OPFModel::viterbi_decode(const float* logits, size_t seq_len, size_t num_classes) const {
    std::vector<int> path(seq_len, 0);
    if (seq_len == 0) return path;

    const float* trans = crf_transition_scores_.data();
    std::vector<float> scores(num_classes);
    std::vector<float> next_scores(num_classes);
    std::vector<int> backptr((seq_len > 0 ? seq_len - 1 : 0) * num_classes, 0);

    // Start: emission[0] + start_scores.
    for (size_t c = 0; c < num_classes; ++c) {
        scores[c] = logits[c] + crf_start_scores_[c];
    }

    for (size_t t = 1; t < seq_len; ++t) {
        const float* emit = logits + t * num_classes;
        for (size_t j = 0; j < num_classes; ++j) {
            float best = kNegInf;
            int best_prev = 0;
            for (size_t i = 0; i < num_classes; ++i) {
                float cand = scores[i] + trans[i * num_classes + j];
                if (cand > best) { best = cand; best_prev = static_cast<int>(i); }
            }
            next_scores[j] = best + emit[j];
            backptr[(t - 1) * num_classes + j] = best_prev;
        }
        scores.swap(next_scores);
    }

    // Add end-scores; fall back to per-step argmax if every path is masked out
    // (happens only if the transition table is completely empty).
    bool any_finite = false;
    for (size_t c = 0; c < num_classes; ++c) {
        if (std::isfinite(scores[c])) { any_finite = true; break; }
    }
    if (!any_finite) {
        for (size_t t = 0; t < seq_len; ++t) {
            size_t best_c = 0;
            float best_v = logits[t * num_classes];
            for (size_t c = 1; c < num_classes; ++c) {
                float v = logits[t * num_classes + c];
                if (v > best_v) { best_v = v; best_c = c; }
            }
            path[t] = static_cast<int>(best_c);
        }
        return path;
    }

    size_t last = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (size_t c = 0; c < num_classes; ++c) {
        float v = scores[c] + crf_end_scores_[c];
        if (v > best_v) { best_v = v; last = c; }
    }
    path[seq_len - 1] = static_cast<int>(last);
    for (size_t t = seq_len - 1; t > 0; --t) {
        last = backptr[(t - 1) * num_classes + last];
        path[t - 1] = static_cast<int>(last);
    }
    return path;
}

std::vector<OPFModel::Span> OPFModel::classify(const std::string& text) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("OPFModel::classify called before init()");
    }
    if (!tokenizer_) {
        throw std::runtime_error("OPFModel::classify: tokenizer not loaded");
    }

    auto tokens = tokenizer_->encode(text);
    if (tokens.empty()) return {};

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    size_t logits_node = forward(tokens, /*use_cache=*/false);
    gb->execute();

    const auto& logits_buf = gb->get_output_buffer(logits_node);
    if (logits_buf.shape.size() != 2 || logits_buf.shape[1] != config_.num_labels) {
        throw std::runtime_error("OPFModel::classify: unexpected logits shape");
    }
    const size_t seq_len = logits_buf.shape[0];
    const size_t num_classes = logits_buf.shape[1];

    // Logits are fp16 in-graph; copy to float for stable Viterbi math.
    const auto* raw = reinterpret_cast<const __fp16*>(gb->get_output(logits_node));
    std::vector<float> logits(seq_len * num_classes);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = static_cast<float>(raw[i]);

    auto path = viterbi_decode(logits.data(), seq_len, num_classes);

    // Walk the decoded tag sequence, collapsing B(I*)E and S runs into spans.
    std::vector<Span> spans;
    const auto& t = label_tables_;
    size_t i = 0;
    while (i < path.size()) {
        int cls = path[i];
        char tag = t.token_boundary_tag[cls];
        int span_idx = t.token_to_span_label[cls];
        if (tag == 'O' || span_idx == 0) { ++i; continue; }

        if (tag == 'S') {
            spans.push_back({i, i + 1, t.span_class_names[span_idx]});
            ++i;
            continue;
        }
        if (tag == 'B') {
            size_t start = i;
            size_t j = i + 1;
            while (j < path.size()) {
                char jtag = t.token_boundary_tag[path[j]];
                int jspan = t.token_to_span_label[path[j]];
                if (jspan != span_idx) break;
                if (jtag == 'E') { ++j; break; }
                if (jtag == 'I') { ++j; continue; }
                break;
            }
            spans.push_back({start, j, t.span_class_names[span_idx]});
            i = j;
            continue;
        }
        // An I/E tag that didn't follow a B is structurally invalid under
        // Viterbi; skip it defensively.
        ++i;
    }
    return spans;
}

}  // namespace engine
}  // namespace cactus
