#include "test_utils.h"
#include "src/mtp_decode.h"

static bool test_assistant_proposal_accepted_when_verifier_agrees() {
    MtpDraftBatch draft;
    draft.tokens = {1, 2};

    std::vector<MtpDistribution> target = {
        MtpDistribution{{0.0f, 1.0f, 0.0f}},
        MtpDistribution{{0.0f, 0.0f, 1.0f}},
        MtpDistribution{{1.0f, 0.0f, 0.0f}},
    };

    auto result = verify_greedy_mtp_draft(draft, target);
    return result.accepted_draft_tokens == 2 &&
           result.output_tokens == std::vector<uint32_t>({1, 2, 0}) &&
           result.emitted_extra_target_token &&
           !result.rejected;
}

static bool test_assistant_proposal_rejected_when_verifier_disagrees() {
    MtpDraftBatch draft;
    draft.tokens = {1, 2};

    std::vector<MtpDistribution> target = {
        MtpDistribution{{0.0f, 1.0f, 0.0f}},
        MtpDistribution{{1.0f, 0.0f, 0.0f}},
        MtpDistribution{{0.0f, 0.0f, 1.0f}},
    };

    auto result = verify_greedy_mtp_draft(draft, target);
    return result.accepted_draft_tokens == 1 &&
           result.output_tokens == std::vector<uint32_t>({1, 0}) &&
           result.rejected &&
           !result.emitted_extra_target_token;
}

static bool test_sampler_is_deterministic_under_greedy_settings() {
    MtpSamplingOptions options;
    options.temperature = 0.0f;

    auto dist = mtp_distribution_from_logits({0.1f, 4.0f, 3.0f}, options);
    return mtp_argmax(dist) == 1 &&
           dist.probabilities == std::vector<float>({0.0f, 1.0f, 0.0f});
}

int main() {
    TestUtils::TestRunner runner("Generic MTP Decode Tests");
    runner.run_test("assistant_proposal_accepted_when_verifier_agrees", test_assistant_proposal_accepted_when_verifier_agrees());
    runner.run_test("assistant_proposal_rejected_when_verifier_disagrees", test_assistant_proposal_rejected_when_verifier_disagrees());
    runner.run_test("sampler_is_deterministic_under_greedy_settings", test_sampler_is_deterministic_under_greedy_settings());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
