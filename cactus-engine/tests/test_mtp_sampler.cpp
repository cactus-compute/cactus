#include "test_utils.h"
#include "src/mtp_sampler.h"

#include <cmath>
#include <cstdint>
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

int main() {
    TestRunner runner("MTP Sampler Tests");

    runner.run_test("greedy_accepts_all", test_greedy_accepts_matching_prefix_and_extra_target_token());
    runner.run_test("greedy_rejects_first", test_greedy_replaces_first_mismatch_with_target_argmax());
    runner.run_test("greedy_rejects_after_prefix", test_greedy_accepts_prefix_then_replaces_mismatch());
    runner.run_test("rejection_adjustment", test_rejection_adjusted_distribution_clamps_and_normalizes());
    runner.run_test("sparse_rejection_adjustment", test_sparse_rejection_adjustment_uses_target_support());
    runner.run_test("sampled_accepts_p_ge_q", test_sampled_accepts_when_target_probability_exceeds_assistant_probability());
    runner.run_test("sampled_rejects_adjusted", test_sampled_rejects_and_samples_adjusted_replacement());
    runner.run_test("zero_q_no_nan", test_zero_assistant_probability_does_not_create_nan());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
