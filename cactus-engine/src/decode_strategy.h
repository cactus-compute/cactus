#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct DecodeOptions {
    bool mtp_enabled = false;
    bool mtp_required = false;
    float temperature = -1.0f;
    float top_p = -1.0f;
    float min_p = 0.15f;
    float repetition_penalty = 1.1f;
    size_t top_k = 0;
    size_t max_draft_tokens = 6;
};

struct DecodeStepResult {
    uint32_t token = 0;
    float entropy = 0.0f;
};

struct DraftBatch {
    std::vector<uint32_t> tokens;
};

struct TargetDecodeState {
    std::vector<uint32_t> input_tokens;
};

struct AcceptedTokenBatch {
    std::vector<uint32_t> tokens;
    std::vector<std::string> texts;
    std::vector<float> entropies;
    size_t drafted_tokens = 0;
    size_t accepted_draft_tokens = 0;
    size_t rejected_tokens = 0;
    std::vector<uint32_t> rejected_draft_token_ids;
};

class MissingAssistantError : public std::runtime_error {
public:
    explicit MissingAssistantError(const std::string& message) : std::runtime_error(message) {}
};

class TargetDecoder {
public:
    virtual ~TargetDecoder() = default;
    virtual DecodeStepResult decode_one(const std::vector<uint32_t>& input_tokens,
                                        const DecodeOptions& options) = 0;
};

class AssistantDraftProvider {
public:
    virtual ~AssistantDraftProvider() = default;
    virtual DraftBatch draft(const TargetDecodeState& state, size_t max_draft_tokens) = 0;
};

class SpeculativeVerifier {
public:
    virtual ~SpeculativeVerifier() = default;
    virtual AcceptedTokenBatch verify(const DraftBatch& draft,
                                      const TargetDecodeState& state,
                                      const DecodeOptions& options) = 0;
};

class DecodeStrategy {
public:
    virtual ~DecodeStrategy() = default;
    virtual AcceptedTokenBatch decode_next(const std::vector<uint32_t>& input_tokens,
                                           const DecodeOptions& options) = 0;
    virtual std::string name() const = 0;
};

class StandardDecodeStrategy final : public DecodeStrategy {
public:
    explicit StandardDecodeStrategy(TargetDecoder& target) : target_(target) {}

    AcceptedTokenBatch decode_next(const std::vector<uint32_t>& input_tokens,
                                   const DecodeOptions& options) override {
        DecodeStepResult step = target_.decode_one(input_tokens, options);
        return AcceptedTokenBatch{
            .tokens = {step.token},
            .entropies = {step.entropy},
        };
    }

    std::string name() const override { return "standard"; }

private:
    TargetDecoder& target_;
};

class Gemma4MtpDecodeStrategy final : public DecodeStrategy {
public:
    Gemma4MtpDecodeStrategy(TargetDecoder& target,
                            AssistantDraftProvider& assistant,
                            SpeculativeVerifier& verifier)
        : target_(target), assistant_(assistant), verifier_(verifier) {}

    AcceptedTokenBatch decode_next(const std::vector<uint32_t>& input_tokens,
                                   const DecodeOptions& options) override {
        (void)target_;
        TargetDecodeState state{.input_tokens = input_tokens};
        DraftBatch draft = assistant_.draft(state, options.max_draft_tokens);
        return verifier_.verify(draft, state, options);
    }

    std::string name() const override { return "gemma4_mtp"; }

private:
    TargetDecoder& target_;
    AssistantDraftProvider& assistant_;
    SpeculativeVerifier& verifier_;
};

inline std::unique_ptr<DecodeStrategy> make_decode_strategy(TargetDecoder& target,
                                                            AssistantDraftProvider* assistant,
                                                            SpeculativeVerifier* verifier,
                                                            const DecodeOptions& options) {
    if (!options.mtp_enabled) {
        return std::make_unique<StandardDecodeStrategy>(target);
    }
    if (!assistant || !verifier) {
        if (options.mtp_required) {
            throw MissingAssistantError("Gemma 4 MTP assistant is unavailable");
        }
        return std::make_unique<StandardDecodeStrategy>(target);
    }
    return std::make_unique<Gemma4MtpDecodeStrategy>(target, *assistant, *verifier);
}
