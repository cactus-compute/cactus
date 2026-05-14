#pragma once

#include "decode_strategy.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct AcceptedToken {
    uint32_t id = 0;
    std::string text;
};

struct MtpCompletionResult {
    bool stopped = false;
    size_t tokens_consumed = 0;
};

struct MtpCompletionMetrics {
    std::string fallback_reason;
    size_t drafted_tokens = 0;
    size_t accepted_tokens = 0;
    size_t rejected_tokens = 0;
    size_t mtp_rounds = 0;
};

struct MtpAvailability {
    bool requested = false;
    bool required = false;
    bool assistant_loaded = false;
    bool supports_prompt = true;
};

struct MtpFallbackDecision {
    bool use_standard_decode = false;
    bool error = false;
    std::string reason;
};

inline MtpFallbackDecision decide_mtp_fallback(const MtpAvailability& availability) {
    if (!availability.requested) {
        return MtpFallbackDecision{.use_standard_decode = true, .reason = "mtp_disabled"};
    }
    if (!availability.supports_prompt) {
        return MtpFallbackDecision{
            .use_standard_decode = !availability.required,
            .error = availability.required,
            .reason = "unsupported_prompt",
        };
    }
    if (!availability.assistant_loaded) {
        return MtpFallbackDecision{
            .use_standard_decode = !availability.required,
            .error = availability.required,
            .reason = "assistant_unavailable",
        };
    }
    return MtpFallbackDecision{.reason = ""};
}

class MtpCompletionAccumulator {
public:
    void set_stream_callback(std::function<void(const AcceptedToken&)> callback) {
        callback_ = std::move(callback);
    }

    void set_stop_sequences(std::vector<std::string> stop_sequences) {
        stop_sequences_ = std::move(stop_sequences);
    }

    MtpCompletionResult consume_accepted_batch(const AcceptedTokenBatch& batch) {
        metrics_.drafted_tokens += batch.drafted_tokens;
        metrics_.rejected_tokens += batch.rejected_tokens;
        metrics_.mtp_rounds++;

        MtpCompletionResult result;
        for (size_t i = 0; i < batch.tokens.size(); ++i) {
            AcceptedToken token{
                .id = batch.tokens[i],
                .text = i < batch.texts.size() ? batch.texts[i] : "",
            };

            token_history_.push_back(token.id);
            output_text_ += token.text;
            metrics_.accepted_tokens++;
            result.tokens_consumed++;

            if (callback_) {
                callback_(token);
            }

            if (has_stop_sequence()) {
                result.stopped = true;
                break;
            }
        }

        return result;
    }

    void record_fallback(const MtpFallbackDecision& decision) {
        metrics_.fallback_reason = decision.reason;
    }

    const std::string& output_text() const { return output_text_; }
    const std::vector<uint32_t>& token_history() const { return token_history_; }
    const MtpCompletionMetrics& metrics() const { return metrics_; }

private:
    bool has_stop_sequence() const {
        for (const auto& stop : stop_sequences_) {
            if (!stop.empty() && output_text_.find(stop) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    std::function<void(const AcceptedToken&)> callback_;
    std::vector<std::string> stop_sequences_;
    std::string output_text_;
    std::vector<uint32_t> token_history_;
    MtpCompletionMetrics metrics_;
};
