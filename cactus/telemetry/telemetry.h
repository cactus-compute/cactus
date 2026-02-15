#pragma once
#include <string>
#include <cstdint>

namespace cactus {
namespace telemetry {

struct CompletionMetrics {
    bool success;
    bool cloud_handoff;
    double ttft_ms;
    double prefill_tps;
    double decode_tps;
    double response_time_ms;
    double confidence;
    double ram_usage_mb;
    size_t prefill_tokens;
    size_t decode_tokens;
    const char* error_message;
    const char* function_calls_json;
};

void init(const char* project_id = nullptr, const char* project_scope = nullptr, const char* cloud_key = nullptr);
void setEnabled(bool enabled);
void setCloudDisabled(bool disabled);
void recordInit(const char* model, bool success, double response_time_ms, const char* message);
void recordCompletion(const char* model, const CompletionMetrics& metrics);
void recordCompletion(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void recordEmbedding(const char* model, bool success, const char* message);
void recordTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void recordStreamTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, double session_ttft_ms, double session_tps, double session_time_ms, int session_tokens, const char* message);
void setStreamMode(bool in_stream);
void markInference(bool active);
void flush();
void shutdown();

} // namespace telemetry
} // namespace cactus
