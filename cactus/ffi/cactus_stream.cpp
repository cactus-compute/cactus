#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "telemetry/telemetry.h"
#include <cstring>
#include <regex>
#include <chrono>

using namespace cactus::ffi;

double json_number(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return 0.0;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;
    try { return std::stod(json.substr(start, end - start)); }
    catch (...) { return 0.0; }
}

std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t start = pos + pattern.size();

    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '"') return {};
    size_t q1 = start;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

inline std::string escape_json(const std::string& s) {
    return escape_json_string(s);
}

bool json_bool(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start + 4 <= json.size() && json.substr(start, 4) == "true") return true;
    if (start + 5 <= json.size() && json.substr(start, 5) == "false") return false;
    return false;
}

std::string json_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "[]";
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '[') return "[]";
    int depth = 1;
    size_t end = start + 1;
    while (end < json.size() && depth > 0) {
        if (json[end] == '[') depth++;
        else if (json[end] == ']') depth--;
        end++;
    }
    return json.substr(start, end - start);
}

static bool fuzzy_match(const std::string& a, const std::string& b, size_t n, double threshold) {
    if (!n) return false;
    if (a.size() < n || b.size() < n) return false;

    std::vector<size_t> dp(n + 1);
    size_t dp_im1_jm1;

    for (size_t j = 0; j <= n; ++j) dp[j] = j;

    for (size_t i = 1; i <= n; ++i) {
        dp_im1_jm1 = dp[0];
        dp[0] = i;

        for (size_t j = 1; j <= n; ++j) {
            size_t dp_im1_j = dp[j];

            if (a[i - 1] == b[j - 1]) {
                dp[j] = dp_im1_jm1;
            } else {
                dp[j] = std::min({
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    dp_im1_jm1 + 1
                });
            }
            
            dp_im1_jm1 = dp_im1_j;
        }
    }

    return 1.0 - static_cast<double>(dp[n]) / static_cast<double>(n) >= threshold;
}

static std::string suppress_unwanted_text(const std::string& text) {
    static const std::regex pattern(R"(\([^)]*\)|\[[^\]]*\]|\.\.\.)");
    std::string result = std::regex_replace(text, pattern, "");

    size_t start = result.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = result.find_last_not_of(" \t\n\r");
    return result.substr(start, end - start + 1);
}

static void parse_stream_transcribe_init_options(const std::string& json,
                                                 double& confirmation_threshold,
                                                 size_t& min_chunk_size,
                                                 bool& telemetry_enabled) {
    confirmation_threshold = 0.99;
    min_chunk_size = 32000;
    telemetry_enabled = true;

    if (json.empty()) {
        return;
    }

    size_t pos = json.find("\"confirmation_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        confirmation_threshold = std::stod(json.substr(pos));
    }

    pos = json.find("\"min_chunk_size\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        min_chunk_size = static_cast<size_t>(std::stod(json.substr(pos)));
    }

    pos = json.find("\"telemetry_enabled\"");
    if (pos != std::string::npos) {
        telemetry_enabled = json_bool(json, "telemetry_enabled");
    }
}

struct CactusStreamTranscribeHandle {
    CactusModelHandle* model_handle;

    struct CactusStreamTranscribeOptions {
        double confirmation_threshold;
        size_t min_chunk_size;
    } options;

    std::vector<uint8_t> audio_buffer;

    std::string previous_transcription;
    size_t previous_audio_buffer_size;

    char transcribe_response_buffer[8192];

    std::chrono::steady_clock::time_point stream_start;
    bool stream_first_token_seen;
    double stream_first_token_ms;
    int stream_total_tokens;

    std::chrono::steady_clock::time_point stream_session_start;
    bool stream_session_first_token_seen;
    double stream_session_first_token_ms;
    int stream_cumulative_tokens;
};

extern "C" {

cactus_stream_transcribe_t cactus_stream_transcribe_start(cactus_model_t model, const char* options_json) {
    if (!model) {
        last_error_message = "Model not initialized. Check model path and files.";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }

    try {
        auto* model_handle = static_cast<CactusModelHandle*>(model);
        if (!model_handle->model) {
            last_error_message = "Invalid model handle.";
            CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
            return nullptr;
        }

        auto* stream_handle = new CactusStreamTranscribeHandle();
        stream_handle->model_handle = model_handle;
        stream_handle->previous_audio_buffer_size = 0;
        stream_handle->transcribe_response_buffer[0] = '\0';

        auto session_start_time = std::chrono::steady_clock::now();
        stream_handle->stream_start = session_start_time;
        stream_handle->stream_first_token_seen = false;
        stream_handle->stream_first_token_ms = 0.0;
        stream_handle->stream_total_tokens = 0;

        stream_handle->stream_session_start = session_start_time;
        stream_handle->stream_session_first_token_seen = false;
        stream_handle->stream_session_first_token_ms = 0.0;
        stream_handle->stream_cumulative_tokens = 0;

        double confirmation_threshold;
        size_t min_chunk_size;
        bool telemetry_enabled;
        parse_stream_transcribe_init_options(
            options_json ? options_json : "",
            confirmation_threshold,
            min_chunk_size,
            telemetry_enabled
        );

        stream_handle->options = { confirmation_threshold, min_chunk_size };

        CACTUS_LOG_INFO("stream_transcribe_start",
            "Stream transcription initialized for model: " << model_handle->model_name);

        return stream_handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_start: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription initialization";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }
}

int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    if (!pcm_buffer || pcm_buffer_size == 0) {
        last_error_message = "Invalid parameters: pcm_buffer or pcm_buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        last_error_message = "Invalid parameters: response_buffer or buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    try {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);

        handle->audio_buffer.insert(
            handle->audio_buffer.end(),
            pcm_buffer,
            pcm_buffer + pcm_buffer_size
        );
        CACTUS_LOG_DEBUG("stream_transcribe_process",
            "Inserted " << pcm_buffer_size << " bytes, buffer size: " << handle->audio_buffer.size());

        if (handle->audio_buffer.size() < handle->options.min_chunk_size * sizeof(int16_t)) {
            std::string json_response = "{\"success\":true,\"confirmed\":\"\",\"pending\":\"\"}";

            if (json_response.length() >= buffer_size) {
                last_error_message = "Response buffer too small";
                CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
                return -1;
            }

            std::strcpy(response_buffer, json_response.c_str());
            return static_cast<int>(json_response.length());
        }

        bool is_moonshine = handle->model_handle->model->get_config().model_type == cactus::engine::Config::ModelType::MOONSHINE;

        cactus::telemetry::setStreamMode(true);
        const int result = cactus_transcribe(
            handle->model_handle,
            nullptr,
            is_moonshine ? "" : "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            handle->transcribe_response_buffer,
            sizeof(handle->transcribe_response_buffer),
            nullptr,
            nullptr,
            nullptr,
            handle->audio_buffer.data(),
            handle->audio_buffer.size());
        cactus::telemetry::setStreamMode(false);

        cactus_reset(handle->model_handle);

        if (result < 0) {
            last_error_message = "Transcription failed in stream process.";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
            return -1;
        }

        std::string json_str(handle->transcribe_response_buffer);
        std::string response = suppress_unwanted_text(json_string(json_str, "response"));

        std::string error = json_string(json_str, "error");
        bool cloud_handoff = json_bool(json_str, "cloud_handoff");
        std::string function_calls = json_array(json_str, "function_calls");
        double confidence = json_number(json_str, "confidence");
        double time_to_first_token_ms = json_number(json_str, "time_to_first_token_ms");
        double total_time_ms = json_number(json_str, "total_time_ms");
        double prefill_tps = json_number(json_str, "prefill_tps");
        double decode_tps = json_number(json_str, "decode_tps");
        double ram_usage_mb = json_number(json_str, "ram_usage_mb");
        double prefill_tokens = json_number(json_str, "prefill_tokens");
        double decode_tokens = json_number(json_str, "decode_tokens");

        double chunk_decode_tokens = decode_tokens;
        if (chunk_decode_tokens < 0.0) chunk_decode_tokens = 0.0;

        std::string confirmed;
        const size_t n = std::min(handle->previous_transcription.size(), response.size());
        if (fuzzy_match(handle->previous_transcription, response, n, handle->options.confirmation_threshold)) {
            handle->audio_buffer.erase(
                handle->audio_buffer.begin(),
                handle->audio_buffer.begin() + handle->previous_audio_buffer_size
            );
            confirmed = suppress_unwanted_text(handle->previous_transcription);
            handle->previous_transcription.clear();
            handle->previous_audio_buffer_size = 0;

            if (chunk_decode_tokens > 0.0) {
                handle->stream_total_tokens += static_cast<int>(std::round(chunk_decode_tokens));
                handle->stream_cumulative_tokens += static_cast<int>(std::round(chunk_decode_tokens));
            }

            if (!handle->stream_first_token_seen) {
                auto now = std::chrono::steady_clock::now();
                handle->stream_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
                handle->stream_first_token_seen = true;
            }

            if (!handle->stream_session_first_token_seen) {
                auto now = std::chrono::steady_clock::now();
                handle->stream_session_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
                handle->stream_session_first_token_seen = true;
            }
        } else {
            handle->previous_transcription = response;
            handle->previous_audio_buffer_size = handle->audio_buffer.size();
        }

        constexpr int STREAM_TOKENS_CAP = 20000;
        constexpr double STREAM_DURATION_CAP_MS = 600000.0;
        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
        if (handle->stream_total_tokens >= STREAM_TOKENS_CAP || elapsed_ms >= STREAM_DURATION_CAP_MS) {
            double period_tps = (elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_total_tokens) * 1000.0) / elapsed_ms : 0.0;

            double cumulative_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
            double cumulative_tps = (cumulative_elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_cumulative_tokens) * 1000.0) / cumulative_elapsed_ms : 0.0;

            cactus::telemetry::recordStreamTranscription(
                handle->model_handle->model_name.c_str(),
                true,
                handle->stream_first_token_ms,
                period_tps,
                elapsed_ms,
                handle->stream_total_tokens,
                handle->stream_session_first_token_ms,
                cumulative_tps,
                cumulative_elapsed_ms,
                handle->stream_cumulative_tokens,
                ""
            );

            handle->stream_start = std::chrono::steady_clock::now();
            handle->stream_first_token_seen = false;
            handle->stream_first_token_ms = 0.0;
            handle->stream_total_tokens = 0;
        }

        double total_tokens = prefill_tokens + decode_tokens;
        std::ostringstream json_builder;
        json_builder << "{";
        json_builder << "\"success\":true,";
        json_builder << "\"error\":" << (error.empty() ? "null" : "\"" + escape_json(error) + "\"") << ",";
        json_builder << "\"cloud_handoff\":" << (cloud_handoff ? "true" : "false") << ",";
        json_builder << "\"confirmed\":\"" << escape_json(confirmed) << "\",";
        json_builder << "\"pending\":\"" << escape_json(response) << "\",";
        json_builder << "\"function_calls\":" << function_calls << ",";
        json_builder << "\"confidence\":" << confidence << ",";
        json_builder << "\"time_to_first_token_ms\":" << time_to_first_token_ms << ",";
        json_builder << "\"total_time_ms\":" << total_time_ms << ",";
        json_builder << "\"prefill_tps\":" << prefill_tps << ",";
        json_builder << "\"decode_tps\":" << decode_tps << ",";
        json_builder << "\"ram_usage_mb\":" << ram_usage_mb << ",";
        json_builder << "\"prefill_tokens\":" << prefill_tokens << ",";
        json_builder << "\"decode_tokens\":" << decode_tokens << ",";
        json_builder << "\"total_tokens\":" << total_tokens;
        json_builder << "}";

        std::string json_response = json_builder.str();

        if (json_response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
            return -1;
        }

        std::strcpy(response_buffer, json_response.c_str());
        return static_cast<int>(json_response.length());
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_process: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription processing";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response("Unknown error during stream processing", response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, "Unknown error during stream processing");
        return -1;
    }
}

int cactus_stream_transcribe_stop(
    cactus_stream_transcribe_t stream,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        return -1;
    }

    auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);

    if (!response_buffer || buffer_size == 0) {
        delete handle;
        return 0;
    }

    try {
        std::string suppressed = suppress_unwanted_text(handle->previous_transcription);

        std::string json_response = "{\"success\":true,\"confirmed\":\"" +
            escape_json(suppressed) + "\"}";

        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
        double period_tps = (elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_total_tokens) * 1000.0) / elapsed_ms : 0.0;

        double cumulative_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
        double cumulative_tps = (cumulative_elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_cumulative_tokens) * 1000.0) / cumulative_elapsed_ms : 0.0;

        cactus::telemetry::recordStreamTranscription(
            handle->model_handle->model_name.c_str(),
            true,
            handle->stream_first_token_ms,
            period_tps,
            elapsed_ms,
            handle->stream_total_tokens,
            handle->stream_session_first_token_ms,
            cumulative_tps,
            cumulative_elapsed_ms,
            handle->stream_cumulative_tokens,
            ""
        );

        if (json_response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, last_error_message.c_str());
            delete handle;
            return -1;
        }

        std::strcpy(response_buffer, json_response.c_str());
        delete handle;
        return static_cast<int>(json_response.length());
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_stop: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, e.what());
        delete handle;
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription stop";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response("Unknown error during stream stop", response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, "Unknown error during stream stop");
        delete handle;
        return -1;
    }
}

}
