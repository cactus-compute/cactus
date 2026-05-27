#include "wrongness_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace cactus {
namespace engine {
namespace {

template <typename T>
bool read_exact(std::ifstream& in, T& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(in);
}

bool read_bytes(std::ifstream& in, char* out, size_t n) {
    in.read(out, static_cast<std::streamsize>(n));
    return static_cast<bool>(in);
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

float linear_relu(const float* input, const std::vector<float>& weight,
                  const std::vector<float>& bias, size_t in_dim, size_t row) {
    float acc = bias[row];
    const float* w = weight.data() + row * in_dim;
    for (size_t i = 0; i < in_dim; ++i) {
        acc += input[i] * w[i];
    }
    return std::max(acc, 0.0f);
}

} // namespace

bool WrongnessProbe::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[8] = {};
    if (!read_bytes(in, magic, sizeof(magic)) || std::memcmp(magic, "CCHP10P6", 8) != 0) {
        return false;
    }
    uint32_t version = 0;
    uint32_t tensor_count = 0;
    if (!read_exact(in, version) || !read_exact(in, tensor_count) || version != 1) {
        return false;
    }

    std::map<std::string, std::vector<float>> tensors;
    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint16_t name_len = 0;
        if (!read_exact(in, name_len) || name_len == 0 || name_len > 512) return false;
        std::string name(name_len, '\0');
        if (!read_bytes(in, name.data(), name.size())) return false;
        uint32_t count = 0;
        if (!read_exact(in, count)) return false;
        std::vector<float> values(count);
        if (count > 0 && !read_bytes(in, reinterpret_cast<char*>(values.data()), count * sizeof(float))) {
            return false;
        }
        tensors[name] = std::move(values);
    }

    auto take = [&](const char* name, size_t expected) -> std::vector<float> {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.size() != expected) {
            throw std::runtime_error(std::string("invalid wrongness probe tensor: ") + name);
        }
        return std::move(it->second);
    };

    try {
        norm_weight_ = take("norm.weight", FEAT_DIM);
        norm_bias_ = take("norm.bias", FEAT_DIM);
        proj_weight_ = take("proj.weight", PROJ_DIM * FEAT_DIM);
        proj_bias_ = take("proj.bias", PROJ_DIM);
        attn_query_ = take("attn_query", PROJ_DIM);
        head0_weight_ = take("head.0.weight", HIDDEN0 * PROJ_DIM);
        head0_bias_ = take("head.0.bias", HIDDEN0);
        head1_weight_ = take("head.2.weight", HIDDEN1 * HIDDEN0);
        head1_bias_ = take("head.2.bias", HIDDEN1);
        head2_weight_ = take("head.4.weight", HIDDEN1);
        head2_bias_ = take("head.4.bias", 1);
    } catch (const std::exception&) {
        loaded_ = false;
        return false;
    }

    loaded_ = true;
    return true;
}

WrongnessProbeScore WrongnessProbe::score(const std::vector<float>& hidden_states, size_t token_count) const {
    if (!loaded_ || token_count == 0 || hidden_states.size() < token_count * FEAT_DIM) {
        return {};
    }
    size_t t_count = std::min(token_count, MAX_TOKENS);
    const size_t token_offset = token_count > t_count ? token_count - t_count : 0;
    std::vector<float> projected(t_count * PROJ_DIM, 0.0f);
    std::vector<float> scores(t_count, 0.0f);
    std::array<float, FEAT_DIM> normed{};

    constexpr float eps = 1.0e-5f;
    constexpr float inv_sqrt_proj = 1.0f / 5.656854249492381f; // 1 / sqrt(32)
    for (size_t t = 0; t < t_count; ++t) {
        const float* x = hidden_states.data() + (token_offset + t) * FEAT_DIM;
        double mean = 0.0;
        for (size_t i = 0; i < FEAT_DIM; ++i) mean += static_cast<double>(x[i]);
        mean /= static_cast<double>(FEAT_DIM);
        double var = 0.0;
        for (size_t i = 0; i < FEAT_DIM; ++i) {
            double d = static_cast<double>(x[i]) - mean;
            var += d * d;
        }
        var /= static_cast<double>(FEAT_DIM);
        float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (size_t i = 0; i < FEAT_DIM; ++i) {
            normed[i] = (x[i] - static_cast<float>(mean)) * inv_std * norm_weight_[i] + norm_bias_[i];
        }

        float* u = projected.data() + t * PROJ_DIM;
        float score = 0.0f;
        for (size_t j = 0; j < PROJ_DIM; ++j) {
            u[j] = linear_relu(normed.data(), proj_weight_, proj_bias_, FEAT_DIM, j);
            score += u[j] * attn_query_[j];
        }
        scores[t] = score * inv_sqrt_proj;
    }

    float max_score = *std::max_element(scores.begin(), scores.end());
    double denom = 0.0;
    for (float s : scores) denom += std::exp(static_cast<double>(s - max_score));
    std::array<float, PROJ_DIM> pooled{};
    for (size_t t = 0; t < t_count; ++t) {
        float alpha = static_cast<float>(std::exp(static_cast<double>(scores[t] - max_score)) / denom);
        const float* u = projected.data() + t * PROJ_DIM;
        for (size_t j = 0; j < PROJ_DIM; ++j) pooled[j] += alpha * u[j];
    }

    std::array<float, HIDDEN0> h0{};
    for (size_t j = 0; j < HIDDEN0; ++j) {
        h0[j] = linear_relu(pooled.data(), head0_weight_, head0_bias_, PROJ_DIM, j);
    }
    std::array<float, HIDDEN1> h1{};
    for (size_t j = 0; j < HIDDEN1; ++j) {
        h1[j] = linear_relu(h0.data(), head1_weight_, head1_bias_, HIDDEN0, j);
    }
    float logit = head2_bias_[0];
    for (size_t i = 0; i < HIDDEN1; ++i) logit += h1[i] * head2_weight_[i];

    WrongnessProbeScore result;
    result.logit = logit;
    result.probability_wrong = sigmoid(logit);
    result.confidence = 1.0f - result.probability_wrong;
    return result;
}

} // namespace engine
} // namespace cactus
