#include "test_utils.h"
#include "src/mtp_completion.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace TestUtils;

bool test_streams_each_accepted_token_in_order() {
    MtpCompletionAccumulator accumulator;
    std::vector<uint32_t> streamed_ids;
    std::string streamed_text;

    accumulator.set_stream_callback([&](const AcceptedToken& token) {
        streamed_ids.push_back(token.id);
        streamed_text += token.text;
    });

    AcceptedTokenBatch batch{
        .tokens = {10, 11, 12},
        .texts = {"desert", " ", "rain"},
    };

    accumulator.consume_accepted_batch(batch);

    return streamed_ids == std::vector<uint32_t>({10, 11, 12})
        && streamed_text == "desert rain";
}

bool test_stop_sequence_checked_after_each_token() {
    MtpCompletionAccumulator accumulator;
    accumulator.set_stop_sequences({"<stop>"});

    AcceptedTokenBatch batch{
        .tokens = {1, 2, 3},
        .texts = {"hello", "<stop>", "SHOULD_NOT_STREAM"},
    };

    MtpCompletionResult result = accumulator.consume_accepted_batch(batch);

    return result.stopped
        && result.tokens_consumed == 2
        && accumulator.output_text() == "hello<stop>"
        && accumulator.token_history() == std::vector<uint32_t>({1, 2});
}

bool test_rejected_draft_tokens_do_not_enter_history() {
    MtpCompletionAccumulator accumulator;

    AcceptedTokenBatch batch{
        .tokens = {4, 9},
        .texts = {"accepted", " replacement"},
        .drafted_tokens = 4,
        .accepted_draft_tokens = 1,
        .rejected_tokens = 3,
        .rejected_draft_token_ids = {5, 6, 7},
    };

    accumulator.consume_accepted_batch(batch);

    return accumulator.token_history() == std::vector<uint32_t>({4, 9});
}

bool test_token_history_updates_once_per_accepted_token() {
    MtpCompletionAccumulator accumulator;

    accumulator.consume_accepted_batch(AcceptedTokenBatch{
        .tokens = {1, 2},
        .texts = {"a", "b"},
    });
    accumulator.consume_accepted_batch(AcceptedTokenBatch{
        .tokens = {3},
        .texts = {"c"},
    });

    return accumulator.token_history() == std::vector<uint32_t>({1, 2, 3});
}

bool test_missing_assistant_records_fallback_reason() {
    MtpCompletionAccumulator accumulator;
    MtpFallbackDecision decision = decide_mtp_fallback(MtpAvailability{
        .requested = true,
        .required = false,
        .assistant_loaded = false,
        .supports_prompt = true,
    });

    accumulator.record_fallback(decision);

    return decision.use_standard_decode
        && accumulator.metrics().fallback_reason == "assistant_unavailable";
}

bool test_non_gemma_target_reports_generic_unsupported_reason() {
    MtpFallbackDecision decision = decide_mtp_fallback(MtpAvailability{
        .requested = true,
        .required = false,
        .assistant_loaded = false,
        .supports_prompt = false,
        .supports_target = false,
    });

    return decision.use_standard_decode
        && !decision.error
        && decision.reason == "unsupported_target";
}

int main() {
    TestRunner runner("MTP Completion Semantics Tests");

    runner.run_test("streams_each_token", test_streams_each_accepted_token_in_order());
    runner.run_test("stop_after_each_token", test_stop_sequence_checked_after_each_token());
    runner.run_test("rejected_not_history", test_rejected_draft_tokens_do_not_enter_history());
    runner.run_test("history_once", test_token_history_updates_once_per_accepted_token());
    runner.run_test("fallback_reason", test_missing_assistant_records_fallback_reason());
    runner.run_test("unsupported_target", test_non_gemma_target_reports_generic_unsupported_reason());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
