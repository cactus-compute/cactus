#pragma once

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct AcceptedToken {
    uint32_t id = 0;
    std::string text;
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
    bool supports_target = true;
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
    if (!availability.supports_target) {
        return MtpFallbackDecision{
            .use_standard_decode = !availability.required,
            .error = availability.required,
            .reason = "unsupported_target",
        };
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

inline std::string mtp_env_string(const char* key, const std::string& fallback = "") {
    const char* value = std::getenv(key);
    return value && value[0] ? std::string(value) : fallback;
}

inline std::string mtp_csv_escape_field(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

inline std::string mtp_bool_field(bool value) {
    return value ? "true" : "false";
}

inline std::string mtp_double_field(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

inline std::string mtp_join_ms(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ";";
        out << values[i];
    }
    return out.str();
}

inline void mtp_append_csv_line(const std::filesystem::path& path,
                                const std::string& header,
                                const std::vector<std::string>& fields) {
    bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Unable to open diagnostic trace CSV: " + path.string());
    }
    if (write_header) out << header << "\n";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) out << ",";
        out << mtp_csv_escape_field(fields[i]);
    }
    out << "\n";
}

struct DiagnosticTraceContext {
    bool enabled = false;
    std::filesystem::path dir;
    std::string run_id;
    std::string model;
    std::string prompt_name;
    std::string shape;
    std::string rep;

    static DiagnosticTraceContext from_env() {
        DiagnosticTraceContext ctx;
        std::string dir = mtp_env_string("CACTUS_GEMMA4_MTP_TRACE_CSV_DIR");
        if (dir.empty()) return ctx;
        ctx.run_id = mtp_env_string("CACTUS_MTP_DIAG_RUN_ID");
        if (ctx.run_id.empty()) return ctx;
        ctx.enabled = true;
        ctx.dir = dir;
        std::filesystem::create_directories(ctx.dir);
        ctx.model = mtp_env_string("CACTUS_MTP_DIAG_MODEL", "model");
        ctx.prompt_name = mtp_env_string("CACTUS_MTP_DIAG_PROMPT", "prompt");
        ctx.shape = mtp_env_string("CACTUS_MTP_DIAG_SHAPE", "unknown");
        ctx.rep = mtp_env_string("CACTUS_MTP_DIAG_REP", "0");
        return ctx;
    }

    void write_round(size_t round_index,
                     size_t generated_start,
                     size_t generated_end,
                     size_t target_batch_m,
                     size_t assistant_pass_count,
                     size_t drafted_tokens,
                     size_t accepted_drafts,
                     bool rejected,
                     bool alt_branch_accepted,
                     bool emitted_extra_target_token,
                     double target_forward_ms,
                     double assistant_total_ms,
                     const std::vector<double>& assistant_step_ms,
                     double sampling_or_argmax_ms,
                     double kv_transaction_ms,
                     double callback_stream_ms,
                     double loop_overhead_ms,
                     double round_total_ms) const {
        if (!enabled || generated_end <= generated_start) return;
        static const std::string header =
            "run_id,model,prompt_name,shape,rep,round_index,generated_token_start,generated_token_end,"
            "generated_tokens_emitted,target_batch_m,assistant_pass_count,drafted_tokens,accepted_drafts,"
            "rejected,alt_branch_accepted,emitted_extra_target_token,target_forward_ms,assistant_total_ms,"
            "assistant_step_ms,sampling_or_argmax_ms,kv_transaction_ms,callback_stream_ms,loop_overhead_ms,round_total_ms";
        mtp_append_csv_line(dir / "round_trace.csv", header, {
            run_id, model, prompt_name, shape, rep,
            std::to_string(round_index),
            std::to_string(generated_start),
            std::to_string(generated_end),
            std::to_string(generated_end - generated_start),
            std::to_string(target_batch_m),
            std::to_string(assistant_pass_count),
            std::to_string(drafted_tokens),
            std::to_string(accepted_drafts),
            mtp_bool_field(rejected),
            mtp_bool_field(alt_branch_accepted),
            mtp_bool_field(emitted_extra_target_token),
            mtp_double_field(target_forward_ms),
            mtp_double_field(assistant_total_ms),
            mtp_join_ms(assistant_step_ms),
            mtp_double_field(sampling_or_argmax_ms),
            mtp_double_field(kv_transaction_ms),
            mtp_double_field(callback_stream_ms),
            mtp_double_field(loop_overhead_ms),
            mtp_double_field(round_total_ms),
        });
    }

    void write_token(size_t token_position,
                     uint32_t token_id,
                     size_t round_index,
                     size_t token_index_in_round,
                     const std::string& source,
                     size_t round_emitted,
                     double target_forward_ms,
                     double assistant_total_ms,
                     double other_ms,
                     double round_total_ms) const {
        if (!enabled || round_emitted == 0) return;
        static const std::string header =
            "run_id,model,prompt_name,shape,rep,token_position,token_id,round_index,token_index_in_round,"
            "source,round_generated_tokens_emitted,round_target_forward_ms,round_assistant_total_ms,"
            "round_other_ms,round_total_ms,allocated_target_forward_ms,allocated_assistant_ms,"
            "allocated_other_ms,allocated_total_ms";
        double denom = static_cast<double>(round_emitted);
        mtp_append_csv_line(dir / "token_trace.csv", header, {
            run_id, model, prompt_name, shape, rep,
            std::to_string(token_position),
            std::to_string(token_id),
            std::to_string(round_index),
            std::to_string(token_index_in_round),
            source,
            std::to_string(round_emitted),
            mtp_double_field(target_forward_ms),
            mtp_double_field(assistant_total_ms),
            mtp_double_field(other_ms),
            mtp_double_field(round_total_ms),
            mtp_double_field(target_forward_ms / denom),
            mtp_double_field(assistant_total_ms / denom),
            mtp_double_field(other_ms / denom),
            mtp_double_field(round_total_ms / denom),
        });
    }
};
