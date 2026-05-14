#pragma once

#include "../models/gemma4/gemma4_mtp_assistant.h"
#include "../models/gemma4/model_gemma4.h"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace cactus {
namespace engine {

struct Gemma4MtpSamplingOptions {
    float temperature = 0.0f;
    float top_p = 0.0f;
    size_t top_k = 0;
    float min_p = 0.0f;
};

struct Gemma4MtpDraftBatch {
    std::vector<uint32_t> tokens;
    uint32_t alt_first_token = 0;
    bool has_alt_first_token = false;
    std::vector<std::vector<float>> probabilities;
    std::vector<std::vector<std::pair<uint32_t, float>>> sparse_probabilities;
    std::vector<double> assistant_step_ms;
    double assistant_draft_ms = 0.0;
    double sampling_or_argmax_ms = 0.0;
};

struct Gemma4MtpVerifyResult {
    std::vector<uint32_t> output_tokens;
    std::vector<uint32_t> verify_tokens;
    std::vector<uint32_t> target_tokens;
    std::vector<size_t> committed_verify_indices;
    std::vector<__fp16> next_hidden;
    size_t next_hidden_row = 0;
    size_t accepted = 0;
    bool rejected = false;
    bool emitted_extra_target_token = false;
    size_t target_cache_crop_length = 0;
    double target_verify_ms = 0.0;
    double sampling_or_argmax_ms = 0.0;
    double kv_transaction_ms = 0.0;
};

struct Gemma4MtpRoundResult {
    Gemma4MtpDraftBatch draft;
    Gemma4MtpVerifyResult verification;
};

uint32_t gemma4_mtp_sample_distribution(const std::vector<float>& probabilities, std::mt19937& rng);
uint32_t gemma4_mtp_sample_sparse_distribution(
    const std::vector<std::pair<uint32_t, float>>& probabilities,
    std::mt19937& rng);

class Gemma4AssistantDraftProvider {
public:
    Gemma4AssistantDraftProvider(Gemma4Model& target, Gemma4MtpAssistant& assistant);

    Gemma4MtpDraftBatch draft(uint32_t input_token,
                              const std::vector<__fp16>& previous_hidden,
                              const Gemma4Model::SharedCacheNodes& cache_nodes,
                              size_t assistant_position,
                              size_t draft_limit,
                              const Gemma4MtpSamplingOptions& options,
                              bool sparse_sampled,
                              std::mt19937& rng);

private:
    Gemma4Model& target_;
    Gemma4MtpAssistant& assistant_;
};

class Gemma4SpeculativeVerifier {
public:
    explicit Gemma4SpeculativeVerifier(Gemma4Model& target);

    Gemma4MtpVerifyResult verify_and_commit(uint32_t input_token,
                                            const Gemma4MtpDraftBatch& draft,
                                            size_t remaining_tokens,
                                            const Gemma4MtpSamplingOptions& options,
                                            bool sparse_sampled,
                                            std::mt19937& rng);

private:
    Gemma4Model& target_;
};

class Gemma4MtpDecodeStrategy {
public:
    Gemma4MtpDecodeStrategy(Gemma4Model& target, Gemma4MtpAssistant& assistant);

    Gemma4MtpRoundResult decode_round(uint32_t input_token,
                                      const std::vector<__fp16>& previous_hidden,
                                      const Gemma4Model::SharedCacheNodes& cache_nodes,
                                      size_t assistant_position,
                                      size_t draft_limit,
                                      size_t remaining_tokens,
                                      const Gemma4MtpSamplingOptions& options,
                                      bool sparse_sampled,
                                      std::mt19937& rng);

private:
    Gemma4AssistantDraftProvider draft_provider_;
    Gemma4SpeculativeVerifier verifier_;
};

}
}
