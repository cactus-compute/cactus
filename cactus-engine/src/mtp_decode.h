#pragma once

#include "mtp_sampler.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

struct MtpDraftBatch {
    std::vector<uint32_t> tokens;
    std::vector<MtpDistribution> probabilities;
};

struct MtpVerificationResult {
    std::vector<uint32_t> output_tokens;
    size_t accepted_draft_tokens = 0;
    bool rejected = false;
    bool emitted_extra_target_token = false;
};

class MtpDeterministicRng {
public:
    explicit MtpDeterministicRng(std::initializer_list<float> values) : values_(values) {}

    float uniform() {
        if (values_.empty()) {
            return 0.0f;
        }
        float value = values_[index_ % values_.size()];
        index_++;
        return std::clamp(value, 0.0f, 1.0f);
    }

private:
    std::vector<float> values_;
    size_t index_ = 0;
};

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
    result.emitted_extra_target_token = true;
    return result;
}

template <typename Rng>
inline MtpVerificationResult verify_sampled_mtp_draft(const MtpDraftBatch& draft,
                                                      const std::vector<MtpDistribution>& target,
                                                      Rng& rng) {
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
    result.emitted_extra_target_token = true;
    return result;
}
