#include "gemma4_mtp_assistant.h"
#include "src/mtp_sampler.h"

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
    return mtp_sparse_distribution_from_logits(std::move(logits), MtpSamplingOptions{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
    });
}

size_t json_size_field(const std::string& json, const std::string& key, size_t fallback) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    char* end = nullptr;
    unsigned long value = std::strtoul(json.c_str() + pos + 1, &end, 10);
    return end == json.c_str() + pos + 1 ? fallback : static_cast<size_t>(value);
}

float json_float_field(const std::string& json, const std::string& key, float fallback) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    char* end = nullptr;
    float value = std::strtof(json.c_str() + pos + 1, &end);
    return end == json.c_str() + pos + 1 ? fallback : value;
}

bool json_bool_field(const std::string& json, const std::string& key, bool fallback) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    while (++pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {}
    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return fallback;
}

std::vector<std::string> json_object_array(const std::string& json, const std::string& key) {
    std::vector<std::string> objects;
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return objects;
    size_t array_start = json.find('[', pos);
    if (array_start == std::string::npos) return objects;
    int depth = 0;
    size_t object_start = std::string::npos;
    for (size_t i = array_start + 1; i < json.size(); ++i) {
        if (json[i] == '{') {
            if (depth == 0) object_start = i;
            depth++;
        } else if (json[i] == '}') {
            depth--;
            if (depth == 0 && object_start != std::string::npos) {
                objects.push_back(json.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        } else if (json[i] == ']' && depth == 0) {
            break;
        }
    }
    return objects;
}

}

Gemma4MtpAssistantConfig default_gemma4_mtp_assistant_config() {
    Gemma4MtpAssistantConfig config;
    config.layers = {
        Gemma4MtpAssistantLayerConfig{},
        Gemma4MtpAssistantLayerConfig{},
        Gemma4MtpAssistantLayerConfig{},
        Gemma4MtpAssistantLayerConfig{
            .head_dim = 512,
            .num_heads = 4,
            .rot_dim = 128,
            .rope_freq = 1000000.0f,
            .window = 0,
            .full_attention = true,
        },
    };
    return config;
}

Gemma4MtpAssistantConfig parse_gemma4_mtp_assistant_config(const std::string& manifest_json) {
    Gemma4MtpAssistantConfig config = default_gemma4_mtp_assistant_config();
    config.target_hidden_dim = json_size_field(manifest_json, "target_hidden_dim", config.target_hidden_dim);
    config.centroid_count = json_size_field(manifest_json, "centroid_count", config.centroid_count);
    config.top_centroid_count = json_size_field(manifest_json, "top_centroid_count", config.top_centroid_count);
    config.tokens_per_centroid = json_size_field(manifest_json, "tokens_per_centroid", config.tokens_per_centroid);

    auto layer_objects = json_object_array(manifest_json, "layers");
    if (!layer_objects.empty()) {
        auto defaults = default_gemma4_mtp_assistant_config().layers;
        config.layers.clear();
        config.layers.reserve(layer_objects.size());
        for (size_t i = 0; i < layer_objects.size(); ++i) {
            Gemma4MtpAssistantLayerConfig layer = i < defaults.size()
                ? defaults[i]
                : Gemma4MtpAssistantLayerConfig{};
            const auto& object = layer_objects[i];
            layer.head_dim = json_size_field(object, "head_dim", layer.head_dim);
            layer.num_heads = json_size_field(object, "num_heads", layer.num_heads);
            layer.rot_dim = json_size_field(object, "rot_dim", layer.rot_dim);
            layer.rope_freq = json_float_field(object, "rope_freq", layer.rope_freq);
            layer.window = json_size_field(object, "window", layer.window);
            layer.full_attention = json_bool_field(object, "full_attention", layer.full_attention);
            config.layers.push_back(layer);
        }
    }

    return config;
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
    std::ifstream manifest_file(path_ + "/assistant_cactus_manifest.json");
    std::string manifest_json((std::istreambuf_iterator<char>(manifest_file)), std::istreambuf_iterator<char>());
    config_ = parse_gemma4_mtp_assistant_config(manifest_json);

    token_embeddings_ = graph_->mmap_embeddings(path_ + "/token_embeddings.weights");
    masked_centroids_ = graph_->mmap_weights(path_ + "/masked_embedding_centroids.weights");
    pre_projection_ = graph_->mmap_weights(path_ + "/pre_projection.weights");
    post_projection_ = graph_->mmap_weights(path_ + "/post_projection.weights");
    output_norm_ = graph_->mmap_weights(path_ + "/output_norm.weights");

    layers_.resize(config_.layers.size());
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
    const size_t expected_ordering_size = config_.centroid_count * config_.tokens_per_centroid;
    token_ordering_.reserve(expected_ordering_size);
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
    if (token_ordering_.size() != expected_ordering_size) {
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
    const auto& layer_config = config_.layers.at(layer_idx);
    const size_t head_dim = layer_config.head_dim;
    const size_t num_heads = layer_config.num_heads;
    const size_t rot_dim = layer_config.rot_dim;
    const float rope_freq = layer_config.rope_freq;
    const size_t window = layer_config.window;
    const size_t cache_k = layer_config.full_attention ? cache_nodes.full_k : cache_nodes.sliding_k;
    const size_t cache_v = layer_config.full_attention ? cache_nodes.full_v : cache_nodes.sliding_v;
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
    if (centroid_count == 0 || centroid_count * config_.tokens_per_centroid > token_ordering_.size()) {
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
    const size_t top_count = std::min(config_.top_centroid_count, centroids.size());
    std::partial_sort(centroids.begin(), centroids.begin() + top_count, centroids.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<uint32_t> candidates;
    candidates.reserve(top_count * config_.tokens_per_centroid);
    for (size_t c = 0; c < top_count; ++c) {
        size_t offset = static_cast<size_t>(centroids[c].second) * config_.tokens_per_centroid;
        for (size_t i = 0; i < config_.tokens_per_centroid; ++i) {
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
    if (target_hidden.size() != config_.target_hidden_dim) {
        throw std::runtime_error("Gemma4MtpAssistant target hidden vector width mismatch");
    }

    auto* gb = graph_;
    gb->soft_reset_keep_pool();
    float embedding_scale = std::sqrt(static_cast<float>(config_.target_hidden_dim)) * 16.0f;
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

    auto hidden_input = gb->input({1, config_.target_hidden_dim}, Precision::FP16);
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
            result.probabilities = mtp_distribution_from_logits(std::move(logits_values), MtpSamplingOptions{
                .temperature = temperature,
                .top_p = top_p,
                .top_k = top_k,
                .min_p = min_p,
            }).probabilities;
        }
    }
    if (return_hidden) {
        result.hidden.resize(config_.target_hidden_dim);
        const auto& projected_buf = gb->get_output_buffer(projected);
        void* projected_ptr = gb->get_output(projected);
        if (projected_buf.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(projected_ptr);
            std::copy(src, src + config_.target_hidden_dim, result.hidden.begin());
        } else if (projected_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(projected_ptr);
            for (size_t i = 0; i < config_.target_hidden_dim; ++i) result.hidden[i] = static_cast<__fp16>(src[i]);
        } else {
            const int8_t* src = static_cast<const int8_t*>(projected_ptr);
            for (size_t i = 0; i < config_.target_hidden_dim; ++i) result.hidden[i] = static_cast<__fp16>(src[i]);
        }
    }
    return result;
}

}
}
