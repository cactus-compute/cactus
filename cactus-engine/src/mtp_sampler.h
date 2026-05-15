#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

struct MtpDistribution {
    std::vector<float> probabilities;
};

struct MtpSamplingOptions {
    float temperature = 0.0f;
    float top_p = 0.0f;
    size_t top_k = 0;
    float min_p = 0.0f;
};

inline uint32_t mtp_argmax(const MtpDistribution& dist) {
    if (dist.probabilities.empty()) {
        throw std::invalid_argument("MTP distribution must not be empty");
    }
    return static_cast<uint32_t>(std::distance(
        dist.probabilities.begin(),
        std::max_element(dist.probabilities.begin(), dist.probabilities.end())));
}

inline uint32_t mtp_sample(const MtpDistribution& dist, float u) {
    if (dist.probabilities.empty()) {
        throw std::invalid_argument("MTP distribution must not be empty");
    }
    float cumulative = 0.0f;
    for (size_t i = 0; i < dist.probabilities.size(); ++i) {
        cumulative += std::max(0.0f, dist.probabilities[i]);
        if (u <= cumulative) {
            return static_cast<uint32_t>(i);
        }
    }
    return static_cast<uint32_t>(dist.probabilities.size() - 1);
}

inline uint32_t mtp_sample_sparse(const std::vector<std::pair<uint32_t, float>>& probabilities, float u) {
    if (probabilities.empty()) {
        throw std::invalid_argument("MTP sparse distribution must not be empty");
    }
    float cumulative = 0.0f;
    for (const auto& item : probabilities) {
        cumulative += std::max(0.0f, item.second);
        if (u <= cumulative) {
            return item.first;
        }
    }
    return probabilities.back().first;
}

inline float mtp_sparse_probability_at(const std::vector<std::pair<uint32_t, float>>& probabilities,
                                       uint32_t token) {
    for (const auto& item : probabilities) {
        if (item.first == token) return item.second;
    }
    return 0.0f;
}

inline std::vector<float> mtp_normalize_logits(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size(), 0.0f);
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : logits) max_value = std::max(max_value, value);
    if (!std::isfinite(max_value)) return probs;

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i])) {
            probs[i] = static_cast<float>(std::exp(static_cast<double>(logits[i] - max_value)));
            sum += probs[i];
        }
    }
    if (sum <= 0.0 || !std::isfinite(sum)) return probs;

    for (float& prob : probs) prob = static_cast<float>(prob / sum);
    return probs;
}

inline MtpDistribution mtp_distribution_from_logits(std::vector<float> logits,
                                                    const MtpSamplingOptions& options) {
    MtpDistribution result;
    if (logits.empty()) return result;

    if (options.temperature == 0.0f) {
        result.probabilities.assign(logits.size(), 0.0f);
        auto it = std::max_element(logits.begin(), logits.end());
        result.probabilities[std::distance(logits.begin(), it)] = 1.0f;
        return result;
    }

    if (options.temperature > 0.0f) {
        for (float& value : logits) value /= options.temperature;
    }

    if (options.top_k > 0 && options.top_k < logits.size()) {
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(), sorted.begin() + options.top_k - 1, sorted.end(), std::greater<float>());
        float kth = sorted[options.top_k - 1];
        for (float& value : logits) {
            if (value < kth) value = -std::numeric_limits<float>::infinity();
        }
    }

    if (options.min_p > 0.0f) {
        auto probs = mtp_normalize_logits(logits);
        if (!probs.empty()) {
            float threshold = (*std::max_element(probs.begin(), probs.end())) * options.min_p;
            for (size_t i = 0; i < logits.size(); ++i) {
                if (probs[i] < threshold) logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    if (options.top_p > 0.0f && options.top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted;
        sorted.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isfinite(logits[i])) sorted.emplace_back(logits[i], i);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        std::vector<float> sorted_logits;
        sorted_logits.reserve(sorted.size());
        for (const auto& item : sorted) sorted_logits.push_back(item.first);
        auto sorted_probs = mtp_normalize_logits(sorted_logits);

        float cumulative = 0.0f;
        bool remove = false;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted_probs[i];
            if (remove) logits[sorted[i].second] = -std::numeric_limits<float>::infinity();
            if (cumulative > options.top_p) remove = true;
        }
    }

    result.probabilities = mtp_normalize_logits(logits);
    float sum = std::accumulate(result.probabilities.begin(), result.probabilities.end(), 0.0f);
    if (sum <= 0.0f || !std::isfinite(sum)) {
        float uniform = result.probabilities.empty() ? 0.0f : 1.0f / static_cast<float>(result.probabilities.size());
        std::fill(result.probabilities.begin(), result.probabilities.end(), uniform);
    }
    return result;
}

inline std::vector<std::pair<uint32_t, float>> mtp_sparse_distribution_from_logits(
    std::vector<std::pair<uint32_t, float>> logits,
    const MtpSamplingOptions& options) {
    if (logits.empty()) return {};

    if (options.temperature > 0.0f) {
        for (auto& item : logits) item.second /= options.temperature;
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
            float prob = static_cast<float>(std::exp(static_cast<double>(item.second - max_value)));
            probs.emplace_back(item.first, prob);
            sum += prob;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) return std::vector<std::pair<uint32_t, float>>{};
        for (auto& item : probs) item.second = static_cast<float>(item.second / sum);
        return probs;
    };

    if (options.min_p > 0.0f) {
        auto probs = normalize(logits);
        if (!probs.empty()) {
            float max_prob = 0.0f;
            for (const auto& item : probs) max_prob = std::max(max_prob, item.second);
            float threshold = max_prob * options.min_p;
            std::vector<std::pair<uint32_t, float>> filtered;
            filtered.reserve(logits.size());
            for (size_t i = 0; i < logits.size(); ++i) {
                if (probs[i].second >= threshold) filtered.push_back(logits[i]);
            }
            logits = std::move(filtered);
        }
    }

    if (options.top_p > 0.0f && options.top_p < 1.0f) {
        auto probs = normalize(logits);
        std::vector<std::pair<uint32_t, float>> filtered;
        filtered.reserve(logits.size());
        float cumulative = 0.0f;
        bool remove = false;
        for (size_t i = 0; i < logits.size(); ++i) {
            cumulative += i < probs.size() ? probs[i].second : 0.0f;
            if (!remove) filtered.push_back(logits[i]);
            if (cumulative > options.top_p) remove = true;
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

inline MtpDistribution rejection_adjusted_distribution(const MtpDistribution& target,
                                                       const MtpDistribution& assistant) {
    if (target.probabilities.size() != assistant.probabilities.size()) {
        throw std::invalid_argument("MTP target and assistant distributions must have same vocabulary size");
    }

    MtpDistribution adjusted;
    adjusted.probabilities.resize(target.probabilities.size());
    float sum = 0.0f;
    for (size_t i = 0; i < target.probabilities.size(); ++i) {
        float value = std::max(target.probabilities[i] - assistant.probabilities[i], 0.0f);
        adjusted.probabilities[i] = value;
        sum += value;
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        adjusted = target;
        sum = std::accumulate(adjusted.probabilities.begin(), adjusted.probabilities.end(), 0.0f);
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        float uniform = adjusted.probabilities.empty() ? 0.0f : 1.0f / static_cast<float>(adjusted.probabilities.size());
        std::fill(adjusted.probabilities.begin(), adjusted.probabilities.end(), uniform);
        return adjusted;
    }

    for (float& value : adjusted.probabilities) {
        value /= sum;
    }
    return adjusted;
}

inline std::vector<std::pair<uint32_t, float>> mtp_rejection_adjusted_sparse_distribution(
    const std::vector<std::pair<uint32_t, float>>& target,
    const std::vector<std::pair<uint32_t, float>>& assistant) {
    std::vector<std::pair<uint32_t, float>> adjusted;
    adjusted.reserve(target.size());
    float sum = 0.0f;
    for (const auto& item : target) {
        float value = std::max(item.second - mtp_sparse_probability_at(assistant, item.first), 0.0f);
        if (value > 0.0f) {
            adjusted.emplace_back(item.first, value);
            sum += value;
        }
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        return target;
    }
    for (auto& item : adjusted) item.second /= sum;
    return adjusted;
}
