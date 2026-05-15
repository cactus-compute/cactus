#pragma once

#include "mtp_sampler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

struct MtpDraftStep {
    uint32_t token = 0;
    uint32_t alt_first_token = 0;
    bool has_alt_first_token = false;
    std::vector<float> probabilities;
    std::vector<std::pair<uint32_t, float>> sparse_probabilities;
    std::vector<__fp16> hidden;
};

struct MtpDraftBatch {
    std::vector<uint32_t> tokens;
    uint32_t alt_first_token = 0;
    bool has_alt_first_token = false;
    std::vector<MtpDistribution> probabilities;
    std::vector<std::vector<std::pair<uint32_t, float>>> sparse_probabilities;
    std::vector<double> assistant_step_ms;
    double assistant_draft_ms = 0.0;
    double sampling_or_argmax_ms = 0.0;
};

struct MtpVerificationResult {
    std::vector<uint32_t> output_tokens;
    std::vector<uint32_t> verify_tokens;
    std::vector<uint32_t> target_tokens;
    std::vector<size_t> committed_verify_indices;
    std::vector<__fp16> next_hidden;
    size_t next_hidden_row = 0;
    size_t accepted_draft_tokens = 0;
    bool rejected = false;
    bool emitted_extra_target_token = false;
    size_t target_cache_crop_length = 0;
    double target_verify_ms = 0.0;
    double sampling_or_argmax_ms = 0.0;
    double kv_transaction_ms = 0.0;
};

struct MtpRoundResult {
    MtpDraftBatch draft;
    MtpVerificationResult verification;
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

inline double mtp_elapsed_ms(std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline uint32_t mtp_sample_distribution(const std::vector<float>& probabilities, std::mt19937& rng) {
    if (probabilities.empty()) return 0;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return mtp_sample(MtpDistribution{probabilities}, dist(rng));
}

inline uint32_t mtp_sample_sparse_distribution(const std::vector<std::pair<uint32_t, float>>& probabilities,
                                               std::mt19937& rng) {
    if (probabilities.empty()) return 0;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return mtp_sample_sparse(probabilities, dist(rng));
}

template <typename DraftOne>
MtpDraftBatch mtp_draft_tokens(uint32_t input_token,
                               const std::vector<__fp16>& previous_hidden,
                               size_t draft_limit,
                               const MtpSamplingOptions& options,
                               bool sparse_sampled,
                               std::mt19937& rng,
                               DraftOne&& draft_one) {
    MtpDraftBatch batch;
    batch.tokens.reserve(draft_limit);
    batch.assistant_step_ms.reserve(draft_limit);
    if (options.temperature > 0.0f) {
        if (sparse_sampled) {
            batch.sparse_probabilities.reserve(draft_limit);
        } else {
            batch.probabilities.reserve(draft_limit);
        }
    }

    std::vector<__fp16> assistant_hidden = previous_hidden;
    uint32_t assistant_input_token = input_token;
    for (size_t i = 0; i < draft_limit; ++i) {
        auto draft_start = std::chrono::steady_clock::now();
        MtpDraftStep step = draft_one(assistant_input_token, assistant_hidden, i, i + 1 < draft_limit);
        auto draft_end = std::chrono::steady_clock::now();
        double step_ms = mtp_elapsed_ms(draft_start, draft_end);
        batch.assistant_draft_ms += step_ms;
        batch.assistant_step_ms.push_back(step_ms);

        uint32_t draft_token = step.token;
        if (i == 0 && step.has_alt_first_token) {
            batch.alt_first_token = step.alt_first_token;
            batch.has_alt_first_token = true;
        }
        if (options.temperature > 0.0f) {
            auto sample_start = std::chrono::steady_clock::now();
            if (sparse_sampled) {
                draft_token = mtp_sample_sparse_distribution(step.sparse_probabilities, rng);
                batch.sparse_probabilities.push_back(std::move(step.sparse_probabilities));
            } else {
                draft_token = mtp_sample_distribution(step.probabilities, rng);
                batch.probabilities.push_back(MtpDistribution{std::move(step.probabilities)});
            }
            auto sample_end = std::chrono::steady_clock::now();
            batch.sampling_or_argmax_ms += mtp_elapsed_ms(sample_start, sample_end);
        }

        batch.tokens.push_back(draft_token);
        if (i + 1 < draft_limit) {
            assistant_hidden = std::move(step.hidden);
        }
        assistant_input_token = draft_token;
    }
    return batch;
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
    result.emitted_extra_target_token = true;
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
    result.emitted_extra_target_token = true;
    return result;
}
