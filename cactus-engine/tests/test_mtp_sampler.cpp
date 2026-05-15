#include "test_utils.h"
#include "src/mtp_decode.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace TestUtils;

static bool tokens_eq(const std::vector<uint32_t>& actual, const std::vector<uint32_t>& expected) {
    return actual == expected;
}

static bool approx_eq(float actual, float expected, float tolerance = 1e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

static MtpDistribution dist(std::initializer_list<float> probs) {
    return MtpDistribution{std::vector<float>(probs)};
}

bool test_greedy_accepts_matching_prefix_and_extra_target_token() {
    MtpDraftBatch draft{
        .tokens = {1, 2, 3},
        .probabilities = {dist({0.05f, 0.90f, 0.03f, 0.02f}),
                          dist({0.04f, 0.03f, 0.88f, 0.05f}),
                          dist({0.01f, 0.02f, 0.03f, 0.94f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.01f, 0.97f, 0.01f, 0.01f}),
        dist({0.01f, 0.01f, 0.97f, 0.01f}),
        dist({0.01f, 0.01f, 0.01f, 0.97f}),
        dist({0.02f, 0.03f, 0.90f, 0.05f}),
    };

    MtpVerificationResult result = verify_greedy_mtp_draft(draft, target);

    return result.accepted_draft_tokens == 3
        && !result.rejected
        && tokens_eq(result.output_tokens, {1, 2, 3, 2});
}

bool test_greedy_replaces_first_mismatch_with_target_argmax() {
    MtpDraftBatch draft{
        .tokens = {1, 2, 3},
        .probabilities = {dist({0.05f, 0.90f, 0.03f, 0.02f}),
                          dist({0.04f, 0.03f, 0.88f, 0.05f}),
                          dist({0.01f, 0.02f, 0.03f, 0.94f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.01f, 0.02f, 0.96f, 0.01f}),
        dist({0.01f, 0.01f, 0.97f, 0.01f}),
        dist({0.01f, 0.01f, 0.01f, 0.97f}),
        dist({0.02f, 0.03f, 0.90f, 0.05f}),
    };

    MtpVerificationResult result = verify_greedy_mtp_draft(draft, target);

    return result.accepted_draft_tokens == 0
        && result.rejected
        && tokens_eq(result.output_tokens, {2});
}

bool test_greedy_accepts_prefix_then_replaces_mismatch() {
    MtpDraftBatch draft{
        .tokens = {1, 2, 3},
        .probabilities = {dist({0.05f, 0.90f, 0.03f, 0.02f}),
                          dist({0.04f, 0.03f, 0.88f, 0.05f}),
                          dist({0.01f, 0.02f, 0.03f, 0.94f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.01f, 0.97f, 0.01f, 0.01f}),
        dist({0.01f, 0.01f, 0.97f, 0.01f}),
        dist({0.90f, 0.02f, 0.03f, 0.05f}),
        dist({0.02f, 0.03f, 0.90f, 0.05f}),
    };

    MtpVerificationResult result = verify_greedy_mtp_draft(draft, target);

    return result.accepted_draft_tokens == 2
        && result.rejected
        && tokens_eq(result.output_tokens, {1, 2, 0});
}

bool test_rejection_adjusted_distribution_clamps_and_normalizes() {
    MtpDistribution target = dist({0.30f, 0.30f, 0.40f});
    MtpDistribution assistant = dist({0.60f, 0.30f, 0.10f});

    MtpDistribution adjusted = rejection_adjusted_distribution(target, assistant);

    return adjusted.probabilities.size() == 3
        && approx_eq(adjusted.probabilities[0], 0.0f)
        && approx_eq(adjusted.probabilities[1], 0.0f)
        && approx_eq(adjusted.probabilities[2], 1.0f);
}

bool test_sparse_rejection_adjustment_uses_target_support() {
    std::vector<std::pair<uint32_t, float>> target = {
        {10, 0.30f},
        {11, 0.30f},
        {12, 0.40f},
    };
    std::vector<std::pair<uint32_t, float>> assistant = {
        {10, 0.60f},
        {11, 0.30f},
        {99, 0.10f},
    };

    auto adjusted = mtp_rejection_adjusted_sparse_distribution(target, assistant);

    return adjusted.size() == 1
        && adjusted[0].first == 12
        && approx_eq(adjusted[0].second, 1.0f)
        && mtp_sparse_probability_at(adjusted, 10) == 0.0f
        && mtp_sample_sparse(adjusted, 0.5f) == 12;
}

bool test_dense_logits_to_probs_applies_temperature_top_k_top_p_min_p() {
    MtpSamplingOptions options{
        .temperature = 1.0f,
        .top_p = 0.80f,
        .top_k = 3,
        .min_p = 0.0f,
    };

    MtpDistribution probs = mtp_distribution_from_logits({1.0f, 4.0f, 3.0f, 2.0f}, options);

    return probs.probabilities.size() == 4
        && probs.probabilities[0] == 0.0f
        && probs.probabilities[3] == 0.0f
        && probs.probabilities[1] > probs.probabilities[2]
        && approx_eq(probs.probabilities[1] + probs.probabilities[2], 1.0f);
}

bool test_sparse_logits_to_probs_matches_dense_on_selected_support() {
    MtpSamplingOptions options{
        .temperature = 1.0f,
        .top_p = 1.0f,
        .top_k = 2,
        .min_p = 0.0f,
    };

    auto sparse = mtp_sparse_distribution_from_logits({{10, 4.0f}, {11, 2.0f}}, options);

    return sparse.size() == 2
        && sparse[0].first == 10
        && sparse[1].first == 11
        && sparse[0].second > sparse[1].second
        && approx_eq(sparse[0].second + sparse[1].second, 1.0f);
}

bool test_sampled_accepts_when_target_probability_exceeds_assistant_probability() {
    MtpDraftBatch draft{
        .tokens = {1},
        .probabilities = {dist({0.50f, 0.20f, 0.30f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.10f, 0.70f, 0.20f}),
        dist({0.10f, 0.20f, 0.70f}),
    };
    MtpDeterministicRng rng({0.99f, 0.99f});

    MtpVerificationResult result = verify_sampled_mtp_draft(draft, target, rng);

    return result.accepted_draft_tokens == 1
        && !result.rejected
        && tokens_eq(result.output_tokens, {1, 2});
}

bool test_sampled_rejects_and_samples_adjusted_replacement() {
    MtpDraftBatch draft{
        .tokens = {0},
        .probabilities = {dist({0.60f, 0.30f, 0.10f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.30f, 0.30f, 0.40f}),
        dist({0.20f, 0.20f, 0.60f}),
    };
    MtpDeterministicRng rng({0.75f, 0.20f});

    MtpVerificationResult result = verify_sampled_mtp_draft(draft, target, rng);

    return result.accepted_draft_tokens == 0
        && result.rejected
        && tokens_eq(result.output_tokens, {2});
}

bool test_zero_assistant_probability_does_not_create_nan() {
    MtpDraftBatch draft{
        .tokens = {1},
        .probabilities = {dist({0.50f, 0.0f, 0.50f})},
    };
    std::vector<MtpDistribution> target = {
        dist({0.20f, 0.30f, 0.50f}),
        dist({0.20f, 0.20f, 0.60f}),
    };
    MtpDeterministicRng rng({0.99f, 0.99f});

    MtpVerificationResult result = verify_sampled_mtp_draft(draft, target, rng);

    return result.accepted_draft_tokens == 1
        && !result.rejected
        && tokens_eq(result.output_tokens, {1, 2});
}

bool test_draft_loop_advances_token_and_hidden_state() {
    MtpSamplingOptions options;
    std::mt19937 rng(7);
    std::vector<uint32_t> inputs;
    std::vector<size_t> hidden_sizes;
    std::vector<bool> needs_hidden_values;

    MtpDraftBatch batch = mtp_draft_tokens(
        5,
        {static_cast<__fp16>(1)},
        3,
        options,
        false,
        rng,
        [&](uint32_t input_token, const std::vector<__fp16>& hidden, size_t i, bool needs_hidden) {
            inputs.push_back(input_token);
            hidden_sizes.push_back(hidden.size());
            needs_hidden_values.push_back(needs_hidden);
            MtpDraftStep step;
            step.token = input_token + 1;
            step.hidden = {static_cast<__fp16>(10 + i)};
            return step;
        });

    return tokens_eq(batch.tokens, {6, 7, 8})
        && tokens_eq(inputs, {5, 6, 7})
        && hidden_sizes == std::vector<size_t>({1, 1, 1})
        && needs_hidden_values == std::vector<bool>({true, true, false})
        && batch.assistant_step_ms.size() == 3;
}

int main() {
    TestRunner runner("MTP Sampler Tests");

    runner.run_test("greedy_accepts_all", test_greedy_accepts_matching_prefix_and_extra_target_token());
    runner.run_test("greedy_rejects_first", test_greedy_replaces_first_mismatch_with_target_argmax());
    runner.run_test("greedy_rejects_after_prefix", test_greedy_accepts_prefix_then_replaces_mismatch());
    runner.run_test("rejection_adjustment", test_rejection_adjusted_distribution_clamps_and_normalizes());
    runner.run_test("sparse_rejection_adjustment", test_sparse_rejection_adjustment_uses_target_support());
    runner.run_test("dense_logits_to_probs", test_dense_logits_to_probs_applies_temperature_top_k_top_p_min_p());
    runner.run_test("sparse_logits_to_probs", test_sparse_logits_to_probs_matches_dense_on_selected_support());
    runner.run_test("sampled_accepts_p_ge_q", test_sampled_accepts_when_target_probability_exceeds_assistant_probability());
    runner.run_test("sampled_rejects_adjusted", test_sampled_rejects_and_samples_adjusted_replacement());
    runner.run_test("zero_q_no_nan", test_zero_assistant_probability_does_not_create_nan());
    runner.run_test("draft_loop_state", test_draft_loop_advances_token_and_hidden_state());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
