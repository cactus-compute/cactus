#include "gemma4_mtp_assistant.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace cactus {
namespace engine {

namespace {

constexpr size_t kTargetHiddenDim = 1536;
constexpr size_t kCentroidCount = 2048;
constexpr size_t kTopCentroidCount = 32;
constexpr size_t kTokensPerCentroid = 128;
constexpr size_t kTokenOrderingSize = kCentroidCount * kTokensPerCentroid;

std::vector<float> probs_from_logits(std::vector<float> logits, float temperature, float top_p, size_t top_k, float min_p) {
    if (logits.empty()) return {};
    if (temperature == 0.0f) {
        std::vector<float> probs(logits.size(), 0.0f);
        auto it = std::max_element(logits.begin(), logits.end());
        probs[std::distance(logits.begin(), it)] = 1.0f;
        return probs;
    }
    if (temperature > 0.0f) {
        for (float& value : logits) value /= temperature;
    }
    if (top_k > 0 && top_k < logits.size()) {
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(), sorted.begin() + top_k - 1, sorted.end(), std::greater<float>());
        float kth = sorted[top_k - 1];
        for (float& value : logits) {
            if (value < kth) value = -std::numeric_limits<float>::infinity();
        }
    }
    auto normalize = [](const std::vector<float>& values) {
        std::vector<float> probs(values.size(), 0.0f);
        float max_value = -std::numeric_limits<float>::infinity();
        for (float value : values) max_value = std::max(max_value, value);
        if (!std::isfinite(max_value)) return probs;
        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i])) {
                probs[i] = std::exp(static_cast<double>(values[i] - max_value));
                sum += probs[i];
            }
        }
        if (sum <= 0.0 || !std::isfinite(sum)) return probs;
        for (float& prob : probs) prob = static_cast<float>(prob / sum);
        return probs;
    };
    if (min_p > 0.0f) {
        auto probs = normalize(logits);
        float threshold = (*std::max_element(probs.begin(), probs.end())) * min_p;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (probs[i] < threshold) logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted;
        sorted.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isfinite(logits[i])) sorted.emplace_back(logits[i], i);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<float> sorted_logits;
        sorted_logits.reserve(sorted.size());
        for (const auto& item : sorted) sorted_logits.push_back(item.first);
        auto sorted_probs = normalize(sorted_logits);
        float cumulative = 0.0f;
        bool remove = false;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted_probs[i];
            if (remove) logits[sorted[i].second] = -std::numeric_limits<float>::infinity();
            if (cumulative > top_p) remove = true;
        }
    }
    auto probs = normalize(logits);
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f || !std::isfinite(sum)) {
        std::fill(probs.begin(), probs.end(), 1.0f / static_cast<float>(probs.size()));
    }
    return probs;
}

std::vector<std::pair<uint32_t, float>> sparse_probs_from_logits_row(
    const void* logits_ptr, Precision precision, size_t row_offset, size_t vocab_size,
    float temperature, float top_p, size_t top_k, float min_p) {
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> top;
    size_t k = std::min(top_k, vocab_size);
    auto push_value = [&](float value, uint32_t token) {
        if (top.size() < k) {
            top.emplace(value, token);
        } else if (value > top.top().first) {
            top.pop();
            top.emplace(value, token);
        }
    };

    if (precision == Precision::FP32) {
        const float* src = static_cast<const float*>(logits_ptr) + row_offset;
        for (size_t i = 0; i < vocab_size; ++i) push_value(src[i], static_cast<uint32_t>(i));
    } else if (precision == Precision::FP16) {
        const __fp16* src = static_cast<const __fp16*>(logits_ptr) + row_offset;
        for (size_t i = 0; i < vocab_size; ++i) push_value(static_cast<float>(src[i]), static_cast<uint32_t>(i));
    } else {
        const int8_t* src = static_cast<const int8_t*>(logits_ptr) + row_offset;
        for (size_t i = 0; i < vocab_size; ++i) push_value(static_cast<float>(src[i]), static_cast<uint32_t>(i));
    }

    std::vector<std::pair<uint32_t, float>> logits;
    logits.reserve(top.size());
    while (!top.empty()) {
        logits.emplace_back(top.top().second, top.top().first);
        top.pop();
    }
    if (temperature > 0.0f) {
        for (auto& item : logits) item.second /= temperature;
    }
    std::sort(logits.begin(), logits.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    auto normalize = [](const std::vector<std::pair<uint32_t, float>>& values) {
        std::vector<std::pair<uint32_t, float>> probs;
        probs.reserve(values.size());
        float max_value = -std::numeric_limits<float>::infinity();
        for (const auto& item : values) max_value = std::max(max_value, item.second);
        if (!std::isfinite(max_value)) return probs;
        double sum = 0.0;
        for (const auto& item : values) {
            float prob = std::exp(static_cast<double>(item.second - max_value));
            probs.emplace_back(item.first, prob);
            sum += prob;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) return std::vector<std::pair<uint32_t, float>>{};
        for (auto& item : probs) item.second = static_cast<float>(item.second / sum);
        return probs;
    };

    if (min_p > 0.0f) {
        auto probs = normalize(logits);
        if (!probs.empty()) {
            float max_prob = 0.0f;
            for (const auto& item : probs) max_prob = std::max(max_prob, item.second);
            float threshold = max_prob * min_p;
            std::vector<std::pair<uint32_t, float>> filtered;
            filtered.reserve(logits.size());
            for (size_t i = 0; i < logits.size(); ++i) {
                if (probs[i].second >= threshold) filtered.push_back(logits[i]);
            }
            logits = std::move(filtered);
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        auto probs = normalize(logits);
        std::vector<std::pair<uint32_t, float>> filtered;
        filtered.reserve(logits.size());
        float cumulative = 0.0f;
        bool remove = false;
        for (size_t i = 0; i < logits.size(); ++i) {
            cumulative += i < probs.size() ? probs[i].second : 0.0f;
            if (!remove) filtered.push_back(logits[i]);
            if (cumulative > top_p) remove = true;
        }
        logits = std::move(filtered);
    }

    auto probs = normalize(logits);
    float sum = 0.0f;
    for (const auto& item : probs) sum += item.second;
    if (sum <= 0.0f || !std::isfinite(sum)) {
        float uniform = probs.empty() ? 0.0f : 1.0f / static_cast<float>(probs.size());
        for (auto& item : probs) item.second = uniform;
    }
    return probs;
}

}

bool Gemma4MtpAssistant::init(CactusGraph* graph, const std::string& assistant_path) {
    if (!graph) {
        throw std::invalid_argument("Gemma4MtpAssistant requires a graph");
    }
    std::filesystem::path resolved_path;
    const std::filesystem::path base_path(assistant_path);
    const std::vector<std::filesystem::path> candidates = {
        base_path,
        base_path / "assistant",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate / "assistant_cactus_manifest.json")) {
            resolved_path = candidate;
            break;
        }
    }
    if (resolved_path.empty()) {
        return false;
    }

    graph_ = graph;
    path_ = resolved_path.string();
    token_embeddings_ = graph_->mmap_embeddings(path_ + "/token_embeddings.weights");
    masked_centroids_ = graph_->mmap_weights(path_ + "/masked_embedding_centroids.weights");
    pre_projection_ = graph_->mmap_weights(path_ + "/pre_projection.weights");
    post_projection_ = graph_->mmap_weights(path_ + "/post_projection.weights");
    output_norm_ = graph_->mmap_weights(path_ + "/output_norm.weights");

    layers_.resize(4);
    for (uint32_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        std::string prefix = path_ + "/layer_" + std::to_string(i) + "_";
        layer.input_norm = graph_->mmap_weights(prefix + "input_norm.weights");
        layer.attn_q = graph_->mmap_weights(prefix + "attn_q.weights");
        layer.attn_q_norm = graph_->mmap_weights(prefix + "attn_q_norm.weights");
        layer.attn_output = graph_->mmap_weights(prefix + "attn_output.weights");
        layer.post_attn_norm = graph_->mmap_weights(prefix + "post_attn_norm.weights");
        layer.pre_ffn_norm = graph_->mmap_weights(prefix + "pre_ffn_norm.weights");
        layer.ffn_gate = graph_->mmap_weights(prefix + "ffn_gate.weights");
        layer.ffn_up = graph_->mmap_weights(prefix + "ffn_up.weights");
        layer.ffn_down = graph_->mmap_weights(prefix + "ffn_down.weights");
        layer.post_ffn_norm = graph_->mmap_weights(prefix + "post_ffn_norm.weights");
        layer.layer_scalar = graph_->mmap_weights(prefix + "layer_scalar.weights");
    }

    std::ifstream ordering_file(path_ + "/masked_embedding_token_ordering.json");
    std::string ordering_json((std::istreambuf_iterator<char>(ordering_file)), std::istreambuf_iterator<char>());
    size_t values_pos = ordering_json.find("\"values\"");
    size_t array_start = ordering_json.find('[', values_pos);
    size_t array_end = ordering_json.find(']', array_start);
    if (values_pos == std::string::npos || array_start == std::string::npos || array_end == std::string::npos) {
        throw std::runtime_error("Invalid masked_embedding_token_ordering.json");
    }
    token_ordering_.clear();
    token_ordering_.reserve(kTokenOrderingSize);
    size_t pos = array_start + 1;
    while (pos < array_end) {
        while (pos < array_end
               && (ordering_json[pos] == ',' || std::isspace(static_cast<unsigned char>(ordering_json[pos])))) {
            pos++;
        }
        if (pos >= array_end) break;
        char* end_ptr = nullptr;
        unsigned long value = std::strtoul(ordering_json.c_str() + pos, &end_ptr, 10);
        if (end_ptr == ordering_json.c_str() + pos) {
            throw std::runtime_error("Invalid Gemma 4 assistant token ordering value");
        }
        token_ordering_.push_back(static_cast<uint32_t>(value));
        pos = static_cast<size_t>(end_ptr - ordering_json.c_str());
    }
    if (token_ordering_.size() != kTokenOrderingSize) {
        throw std::runtime_error("Unexpected Gemma 4 assistant token ordering size");
    }

    initialized_ = true;
    return true;
}

size_t Gemma4MtpAssistant::apply_rope(CactusGraph* gb, size_t tensor, size_t head_dim, size_t rot_dim,
                                      float rope_freq, size_t position) {
    if (rot_dim < head_dim) {
        size_t half_dim = head_dim / 2;
        size_t half_rot = rot_dim / 2;
        size_t pass_len = half_dim - half_rot;
        float adjusted_theta = std::pow(rope_freq, static_cast<float>(rot_dim) / static_cast<float>(head_dim));

        auto left_rot = gb->slice(tensor, 3, 0, half_rot);
        auto left_pass = gb->slice(tensor, 3, half_rot, pass_len);
        auto right_rot = gb->slice(tensor, 3, half_dim, half_rot);
        auto right_pass = gb->slice(tensor, 3, half_dim + half_rot, pass_len);

        auto rotated = gb->rope(gb->concat(left_rot, right_rot, 3), adjusted_theta, position);
        auto rotated_left = gb->slice(rotated, 3, 0, half_rot);
        auto rotated_right = gb->slice(rotated, 3, half_rot, half_rot);

        auto new_left = gb->concat(rotated_left, left_pass, 3);
        auto new_right = gb->concat(rotated_right, right_pass, 3);
        return gb->concat(new_left, new_right, 3);
    }
    return gb->rope(tensor, rope_freq, position);
}

size_t Gemma4MtpAssistant::build_layer(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                       const Gemma4Model::SharedCacheNodes& cache_nodes, size_t position) {
    const auto& layer = layers_.at(layer_idx);
    const bool is_full = layer_idx == 3;
    const size_t head_dim = is_full ? 512 : 256;
    const size_t num_heads = 4;
    const size_t rot_dim = is_full ? 128 : head_dim;
    const float rope_freq = is_full ? 1000000.0f : 10000.0f;
    const size_t window = is_full ? 0 : 512;
    const size_t cache_k = is_full ? cache_nodes.full_k : cache_nodes.sliding_k;
    const size_t cache_v = is_full ? cache_nodes.full_v : cache_nodes.sliding_v;
    if (cache_k == 0 || cache_v == 0) {
        throw std::runtime_error("Gemma 4 MTP target shared KV cache is unavailable");
    }

    auto normed = gb->rms_norm(hidden, layer.input_norm, 1e-6f);
    auto q = gb->matmul(normed, layer.attn_q, true, ComputeBackend::CPU);
    q = gb->reshape(q, {1, 1, num_heads, head_dim});
    q = gb->reshape(gb->rms_norm(gb->reshape(q, {num_heads, head_dim}), layer.attn_q_norm, 1e-6f),
                    {1, 1, num_heads, head_dim});
    q = apply_rope(gb, q, head_dim, rot_dim, rope_freq, position);

    auto attn = gb->attention_cache_only(q, cache_k, cache_v, 1.0f, position, window);
    attn = gb->matmul(gb->reshape(attn, {1, num_heads * head_dim}), layer.attn_output, true, ComputeBackend::CPU);
    attn = gb->rms_norm(attn, layer.post_attn_norm, 1e-6f);
    auto residual = gb->add(hidden, attn);

    auto mlp_in = gb->rms_norm(residual, layer.pre_ffn_norm, 1e-6f);
    auto gate = gb->gelu(gb->matmul(mlp_in, layer.ffn_gate, true, ComputeBackend::CPU));
    auto up = gb->matmul(mlp_in, layer.ffn_up, true, ComputeBackend::CPU);
    auto mlp = gb->matmul(gb->multiply(gate, up), layer.ffn_down, true, ComputeBackend::CPU);
    mlp = gb->rms_norm(mlp, layer.post_ffn_norm, 1e-6f);
    auto out = gb->add(residual, mlp);
    return gb->multiply(out, layer.layer_scalar);
}

std::vector<uint32_t> Gemma4MtpAssistant::candidate_tokens_from_centroids(CactusGraph* gb,
                                                                          size_t centroid_logits_node) const {
    const auto& centroid_buf = gb->get_output_buffer(centroid_logits_node);
    void* centroid_ptr = gb->get_output(centroid_logits_node);
    const size_t centroid_count = centroid_buf.total_size;
    if (centroid_count == 0 || centroid_count * kTokensPerCentroid > token_ordering_.size()) {
        throw std::runtime_error("Gemma 4 MTP assistant centroid/token ordering shape mismatch");
    }

    std::vector<std::pair<float, uint32_t>> centroids;
    centroids.reserve(centroid_count);
    for (uint32_t i = 0; i < centroid_count; ++i) {
        float value = 0.0f;
        if (centroid_buf.precision == Precision::FP32) {
            value = static_cast<const float*>(centroid_ptr)[i];
        } else if (centroid_buf.precision == Precision::FP16) {
            value = static_cast<float>(static_cast<const __fp16*>(centroid_ptr)[i]);
        } else {
            value = static_cast<float>(static_cast<const int8_t*>(centroid_ptr)[i]);
        }
        centroids.emplace_back(value, i);
    }
    const size_t top_count = std::min(kTopCentroidCount, centroids.size());
    std::partial_sort(centroids.begin(), centroids.begin() + top_count, centroids.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<uint32_t> candidates;
    candidates.reserve(top_count * kTokensPerCentroid);
    for (size_t c = 0; c < top_count; ++c) {
        size_t offset = static_cast<size_t>(centroids[c].second) * kTokensPerCentroid;
        for (size_t i = 0; i < kTokensPerCentroid; ++i) {
            candidates.push_back(token_ordering_[offset + i]);
        }
    }
    return candidates;
}

uint32_t Gemma4MtpAssistant::argmax_masked_logits(CactusGraph* gb, size_t logits_node, size_t centroid_logits_node) const {
    const auto& logits_buf = gb->get_output_buffer(logits_node);
    const size_t vocab_size = logits_buf.shape.back();
    const size_t row_offset = (logits_buf.total_size / vocab_size - 1) * vocab_size;
    void* logits_ptr = gb->get_output(logits_node);
    auto candidates = candidate_tokens_from_centroids(gb, centroid_logits_node);

    auto logit_at = [&](uint32_t token) -> float {
        if (logits_buf.precision == Precision::FP32) {
            return static_cast<const float*>(logits_ptr)[row_offset + token];
        }
        if (logits_buf.precision == Precision::FP16) {
            return static_cast<float>(static_cast<const __fp16*>(logits_ptr)[row_offset + token]);
        }
        return static_cast<float>(static_cast<const int8_t*>(logits_ptr)[row_offset + token]);
    };

    uint32_t best = candidates[0];
    float best_value = logit_at(best);
    for (uint32_t token : candidates) {
        float value = logit_at(token);
        if (value > best_value) {
            best_value = value;
            best = token;
        }
    }
    return best;
}

uint32_t Gemma4MtpAssistant::argmax_masked_logits_from_hidden(CactusGraph* gb, size_t hidden_node,
                                                              size_t centroid_logits_node) const {
    const auto& hidden_buf = gb->get_output_buffer(hidden_node);
    if (hidden_buf.shape.size() != 2 || hidden_buf.shape[0] != 1 || hidden_buf.precision != Precision::FP16) {
        throw std::runtime_error("Gemma 4 MTP assistant candidate argmax requires hidden [1, dim] FP16");
    }
    size_t logits = gb->matmul(hidden_node, token_embeddings_, true, ComputeBackend::CPU);
    gb->execute();
    return argmax_masked_logits(gb, logits, centroid_logits_node);
}

std::pair<uint32_t, uint32_t> Gemma4MtpAssistant::top2_masked_logits_from_hidden(
    CactusGraph* gb, size_t hidden_node, size_t centroid_logits_node) const {
    const auto& hidden_buf = gb->get_output_buffer(hidden_node);
    if (hidden_buf.shape.size() != 2 || hidden_buf.shape[0] != 1 || hidden_buf.precision != Precision::FP16) {
        throw std::runtime_error("Gemma 4 MTP assistant candidate top2 requires hidden [1, dim] FP16");
    }

    size_t logits = gb->matmul(hidden_node, token_embeddings_, true, ComputeBackend::CPU);
    gb->execute();
    const auto& logits_buf = gb->get_output_buffer(logits);
    const size_t vocab_size = logits_buf.shape.back();
    const size_t row_offset = (logits_buf.total_size / vocab_size - 1) * vocab_size;
    void* logits_ptr = gb->get_output(logits);
    auto candidates = candidate_tokens_from_centroids(gb, centroid_logits_node);

    auto logit_at = [&](uint32_t token) -> float {
        if (logits_buf.precision == Precision::FP32) {
            return static_cast<const float*>(logits_ptr)[row_offset + token];
        }
        if (logits_buf.precision == Precision::FP16) {
            return static_cast<float>(static_cast<const __fp16*>(logits_ptr)[row_offset + token]);
        }
        return static_cast<float>(static_cast<const int8_t*>(logits_ptr)[row_offset + token]);
    };

    uint32_t first = candidates[0];
    uint32_t second = candidates[0];
    float first_value = -std::numeric_limits<float>::infinity();
    float second_value = -std::numeric_limits<float>::infinity();
    for (uint32_t token : candidates) {
        float value = logit_at(token);
        if (value > first_value) {
            if (token != first) {
                second = first;
                second_value = first_value;
            }
            first = token;
            first_value = value;
        } else if (token != first && value > second_value) {
            second = token;
            second_value = value;
        }
    }
    return {first, second};
}

Gemma4MtpAssistant::StepResult Gemma4MtpAssistant::draft_one(
    uint32_t target_token,
    size_t target_embedding_node,
    const std::vector<__fp16>& target_hidden,
    const Gemma4Model::SharedCacheNodes& cache_nodes,
    size_t position,
    float hidden_scale,
    float temperature,
    float top_p,
    size_t top_k,
    float min_p,
    bool return_hidden,
    bool return_second_token) {
    if (!initialized_ || !graph_) {
        throw std::runtime_error("Gemma4MtpAssistant is not initialized");
    }
    if (target_hidden.size() != kTargetHiddenDim) {
        throw std::runtime_error("Gemma4MtpAssistant expects 1536-wide target hidden vectors");
    }

    auto* gb = graph_;
    gb->soft_reset_keep_pool();
    float embedding_scale = std::sqrt(static_cast<float>(kTargetHiddenDim)) * 16.0f;
    if (const char* value = std::getenv("CACTUS_GEMMA4_MTP_EMBED_SCALE")) {
        embedding_scale *= std::strtof(value, nullptr);
    }
    if (const char* value = std::getenv("CACTUS_GEMMA4_MTP_HIDDEN_SCALE")) {
        hidden_scale = std::strtof(value, nullptr);
    }
    auto token_input = gb->input({1}, Precision::FP32);
    float token_value = static_cast<float>(target_token);
    gb->set_input(token_input, &token_value, Precision::FP32);
    auto target_embedding = gb->scalar_multiply(gb->embedding(target_embedding_node, token_input), embedding_scale);

    auto hidden_input = gb->input({1, kTargetHiddenDim}, Precision::FP16);
    gb->set_input(hidden_input, target_hidden.data(), Precision::FP16);
    if (hidden_scale != 1.0f) {
        hidden_input = gb->scalar_multiply(hidden_input, hidden_scale);
    }
    auto input = gb->concat(target_embedding, hidden_input, 1);

    auto hidden = gb->matmul(input, pre_projection_, true, ComputeBackend::CPU);
    for (uint32_t i = 0; i < layers_.size(); ++i) {
        hidden = build_layer(gb, hidden, i, cache_nodes, position);
    }
    hidden = gb->rms_norm(hidden, output_norm_, 1e-6f);

    const bool sparse_sampled = temperature > 0.0f && top_k > 0;
    const bool greedy_candidate_logits = temperature == 0.0f && return_second_token;
    size_t logits = 0;
    if (!greedy_candidate_logits) {
        logits = gb->matmul(hidden, token_embeddings_, true, ComputeBackend::CPU);
    }
    size_t centroid_logits = 0;
    if (!sparse_sampled) {
        centroid_logits = gb->matmul(hidden, masked_centroids_, true, ComputeBackend::CPU);
    }
    size_t projected = 0;
    if (return_hidden) {
        projected = gb->matmul(hidden, post_projection_, true, ComputeBackend::CPU);
    }
    gb->execute();

    StepResult result;
    if (!sparse_sampled) {
        if (greedy_candidate_logits && return_second_token) {
            auto top2 = top2_masked_logits_from_hidden(gb, hidden, centroid_logits);
            result.token = top2.first;
            result.second_token = top2.second;
            result.has_second_token = top2.second != top2.first;
        } else {
            result.token = argmax_masked_logits(gb, logits, centroid_logits);
        }
    }
    if (temperature > 0.0f) {
        const auto& logits_buf = gb->get_output_buffer(logits);
        const size_t vocab_size = logits_buf.shape.back();
        const size_t row_offset = (logits_buf.total_size / vocab_size - 1) * vocab_size;
        void* logits_ptr = gb->get_output(logits);
        if (top_k > 0) {
            result.sparse_probabilities = sparse_probs_from_logits_row(
                logits_ptr, logits_buf.precision, row_offset, vocab_size,
                temperature, top_p, top_k, min_p);
            if (!result.sparse_probabilities.empty()) {
                result.token = result.sparse_probabilities.front().first;
            }
        } else {
            std::vector<float> logits_values(vocab_size);
            if (logits_buf.precision == Precision::FP32) {
                const float* src = static_cast<const float*>(logits_ptr) + row_offset;
                std::copy(src, src + vocab_size, logits_values.begin());
            } else if (logits_buf.precision == Precision::FP16) {
                const __fp16* src = static_cast<const __fp16*>(logits_ptr) + row_offset;
                for (size_t i = 0; i < vocab_size; ++i) logits_values[i] = static_cast<float>(src[i]);
            } else {
                const int8_t* src = static_cast<const int8_t*>(logits_ptr) + row_offset;
                for (size_t i = 0; i < vocab_size; ++i) logits_values[i] = static_cast<float>(src[i]);
            }
            result.probabilities = probs_from_logits(std::move(logits_values), temperature, top_p, top_k, min_p);
        }
    }
    if (return_hidden) {
        result.hidden.resize(kTargetHiddenDim);
        const auto& projected_buf = gb->get_output_buffer(projected);
        void* projected_ptr = gb->get_output(projected);
        if (projected_buf.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(projected_ptr);
            std::copy(src, src + kTargetHiddenDim, result.hidden.begin());
        } else if (projected_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(projected_ptr);
            for (size_t i = 0; i < kTargetHiddenDim; ++i) result.hidden[i] = static_cast<__fp16>(src[i]);
        } else {
            const int8_t* src = static_cast<const int8_t*>(projected_ptr);
            for (size_t i = 0; i < kTargetHiddenDim; ++i) result.hidden[i] = static_cast<__fp16>(src[i]);
        }
    }
    return result;
}

}
}
