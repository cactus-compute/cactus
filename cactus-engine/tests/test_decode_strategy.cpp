#include "test_utils.h"
#include "src/decode_strategy.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace TestUtils;

class FakeTargetDecoder final : public TargetDecoder {
public:
    explicit FakeTargetDecoder(std::vector<uint32_t> tokens) : tokens_(std::move(tokens)) {}

    DecodeStepResult decode_one(const std::vector<uint32_t>& input_tokens,
                                const DecodeOptions& options) override {
        seen_inputs.push_back(input_tokens);
        seen_options.push_back(options);
        uint32_t token = tokens_.at(index_++);
        return DecodeStepResult{.token = token, .entropy = 0.0f};
    }

    std::vector<std::vector<uint32_t>> seen_inputs;
    std::vector<DecodeOptions> seen_options;

private:
    std::vector<uint32_t> tokens_;
    size_t index_ = 0;
};

class FakeAssistantDraftProvider final : public AssistantDraftProvider {
public:
    explicit FakeAssistantDraftProvider(std::vector<DraftBatch> drafts) : drafts_(std::move(drafts)) {}

    DraftBatch draft(const TargetDecodeState& state, size_t max_draft_tokens) override {
        seen_states.push_back(state);
        seen_limits.push_back(max_draft_tokens);
        return drafts_.at(index_++);
    }

    std::vector<TargetDecodeState> seen_states;
    std::vector<size_t> seen_limits;

private:
    std::vector<DraftBatch> drafts_;
    size_t index_ = 0;
};

class FakeVerifier final : public SpeculativeVerifier {
public:
    explicit FakeVerifier(std::vector<AcceptedTokenBatch> batches) : batches_(std::move(batches)) {}

    AcceptedTokenBatch verify(const DraftBatch& draft,
                              const TargetDecodeState& state,
                              const DecodeOptions& options) override {
        seen_drafts.push_back(draft);
        seen_states.push_back(state);
        seen_options.push_back(options);
        return batches_.at(index_++);
    }

    std::vector<DraftBatch> seen_drafts;
    std::vector<TargetDecodeState> seen_states;
    std::vector<DecodeOptions> seen_options;

private:
    std::vector<AcceptedTokenBatch> batches_;
    size_t index_ = 0;
};

bool test_standard_strategy_is_null_object_fallback() {
    FakeTargetDecoder target({42});
    StandardDecodeStrategy strategy(target);

    DecodeOptions options;
    options.temperature = 0.0f;
    AcceptedTokenBatch batch = strategy.decode_next({7, 8}, options);

    return batch.tokens == std::vector<uint32_t>{42}
        && batch.entropies.size() == 1
        && target.seen_inputs == std::vector<std::vector<uint32_t>>{{7, 8}};
}

bool test_mtp_strategy_returns_accepted_token_batch() {
    FakeTargetDecoder target({99});
    FakeAssistantDraftProvider assistant({
        DraftBatch{.tokens = {10, 11, 12}},
    });
    FakeVerifier verifier({
        AcceptedTokenBatch{
            .tokens = {10, 11, 7},
            .entropies = {0.1f, 0.2f, 0.3f},
            .drafted_tokens = 3,
            .accepted_draft_tokens = 2,
            .rejected_tokens = 1,
        },
    });
    Gemma4MtpDecodeStrategy strategy(target, assistant, verifier);

    DecodeOptions options;
    options.temperature = 0.0f;
    options.max_draft_tokens = 3;
    AcceptedTokenBatch batch = strategy.decode_next({1, 2, 3}, options);

    return batch.tokens == std::vector<uint32_t>({10, 11, 7})
        && batch.drafted_tokens == 3
        && batch.accepted_draft_tokens == 2
        && batch.rejected_tokens == 1
        && assistant.seen_limits == std::vector<size_t>{3}
        && verifier.seen_drafts.size() == 1;
}

bool test_strategy_factory_selects_standard_when_mtp_disabled() {
    FakeTargetDecoder target({5});
    FakeAssistantDraftProvider assistant({});
    FakeVerifier verifier({});

    DecodeOptions options;
    options.mtp_enabled = false;
    std::unique_ptr<DecodeStrategy> strategy = make_decode_strategy(target, &assistant, &verifier, options);

    return strategy->name() == std::string("standard");
}

bool test_strategy_factory_selects_mtp_when_available() {
    FakeTargetDecoder target({5});
    FakeAssistantDraftProvider assistant({});
    FakeVerifier verifier({});

    DecodeOptions options;
    options.mtp_enabled = true;
    options.max_draft_tokens = 6;
    std::unique_ptr<DecodeStrategy> strategy = make_decode_strategy(target, &assistant, &verifier, options);

    return strategy->name() == std::string("gemma4_mtp");
}

bool test_strategy_factory_errors_when_mtp_required_but_unavailable() {
    FakeTargetDecoder target({5});

    DecodeOptions options;
    options.mtp_enabled = true;
    options.mtp_required = true;

    try {
        (void)make_decode_strategy(target, nullptr, nullptr, options);
    } catch (const MissingAssistantError& err) {
        return std::string(err.what()).find("assistant") != std::string::npos;
    }

    return false;
}

int main() {
    TestRunner runner("Decode Strategy Tests");

    runner.run_test("standard_null_object", test_standard_strategy_is_null_object_fallback());
    runner.run_test("mtp_returns_batch", test_mtp_strategy_returns_accepted_token_batch());
    runner.run_test("factory_standard", test_strategy_factory_selects_standard_when_mtp_disabled());
    runner.run_test("factory_mtp", test_strategy_factory_selects_mtp_when_available());
    runner.run_test("factory_missing_required", test_strategy_factory_errors_when_mtp_required_but_unavailable());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
