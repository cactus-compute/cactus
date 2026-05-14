#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

struct MtpDistribution {
    std::vector<float> probabilities;
};

struct MtpDraftBatch {
    std::vector<uint32_t> tokens;
    std::vector<MtpDistribution> probabilities;
};

struct MtpVerificationResult {
    std::vector<uint32_t> output_tokens;
    size_t accepted_draft_tokens = 0;
    bool rejected = false;
};

class MtpDeterministicRng {
public:
    explicit MtpDeterministicRng(std::initializer_list<float> values) : values_(values) {}

    float uniform() {
        if (values_.empty()) return 0.0f;
        float value = values_[index_ % values_.size()];
        index_++;
        return std::clamp(value, 0.0f, 1.0f);
    }

private:
    std::vector<float> values_;
    size_t index_ = 0;
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

inline MtpVerificationResult verify_greedy_mtp_draft(const MtpDraftBatch& draft,
                                                     const std::vector<MtpDistribution>& target) {
    if (target.size() < draft.tokens.size() + 1) {
        throw std::invalid_argument("MTP target distributions must include one extra next-token distribution");
    }

    MtpVerificationResult result;
    for (size_t i = 0; i < draft.tokens.size(); ++i) {
        uint32_t target_token = mtp_argmax(target[i]);
        if (draft.tokens[i] != target_token) {
            result.output_tokens.push_back(target_token);
            result.rejected = true;
            return result;
        }
        result.output_tokens.push_back(draft.tokens[i]);
        result.accepted_draft_tokens++;
    }

    result.output_tokens.push_back(mtp_argmax(target[draft.tokens.size()]));
    return result;
}

inline MtpVerificationResult verify_sampled_mtp_draft(const MtpDraftBatch& draft,
                                                      const std::vector<MtpDistribution>& target,
                                                      MtpDeterministicRng& rng) {
    if (target.size() < draft.tokens.size() + 1) {
        throw std::invalid_argument("MTP target distributions must include one extra next-token distribution");
    }
    if (draft.probabilities.size() < draft.tokens.size()) {
        throw std::invalid_argument("MTP draft must include assistant probabilities for each token");
    }

    MtpVerificationResult result;
    for (size_t i = 0; i < draft.tokens.size(); ++i) {
        uint32_t y = draft.tokens[i];
        const auto& q = draft.probabilities[i];
        const auto& p = target[i];
        if (y >= q.probabilities.size() || y >= p.probabilities.size()) {
            throw std::out_of_range("MTP draft token outside distribution vocabulary");
        }

        float q_y = q.probabilities[y];
        float p_y = p.probabilities[y];
        float accept_probability = q_y <= 0.0f ? 1.0f : std::min(1.0f, p_y / q_y);

        if (rng.uniform() <= accept_probability) {
            result.output_tokens.push_back(y);
            result.accepted_draft_tokens++;
            continue;
        }

        MtpDistribution adjusted = rejection_adjusted_distribution(p, q);
        result.output_tokens.push_back(mtp_sample(adjusted, rng.uniform()));
        result.rejected = true;
        return result;
    }

    result.output_tokens.push_back(mtp_sample(target[draft.tokens.size()], rng.uniform()));
    return result;
}
