#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "net.h"
#include <cstring>

using namespace cactus::net;

// Forward declarations for functions defined later in this file
namespace cactus {
namespace ffi {
bool perform_cloud_fallback(const std::string& messages_json,
                            const std::string& options_json,
                            std::string& cloud_response,
                            std::string& error_message);
std::string convert_cloud_response_to_cactus_format(
    const std::string& cloud_response,
    double local_time_to_first_token,
    double local_prefill_tps,
    size_t local_prompt_tokens,
    float local_confidence);
std::string construct_cloud_fallback_error_json(
    const std::string& error_message,
    float confidence,
    double time_to_first_token,
    double prefill_tps,
    size_t prompt_tokens);
} // namespace ffi
} // namespace cactus

extern "C" {

void cactus_set_cloud_config(
    const char* provider,
    const char* api_key,
    const char* endpoint_url,
    const char* model
) {
    CloudConfig config;
    config.provider = provider ? provider : "";
    config.api_key = api_key ? api_key : "";
    config.endpoint_url = endpoint_url ? endpoint_url : "";
    config.model = model ? model : "";

    set_cloud_config(config);
}

bool cactus_is_cloud_configured(void) {
    return is_cloud_configured();
}

bool cactus_is_network_available(void) {
    return is_network_available();
}

int cactus_test_http(const char* url, char* response_buffer, size_t buffer_size) {
    if (!url || !response_buffer || buffer_size == 0) {
        return -1;
    }

    HttpRequest request;
    request.url = url;
    request.timeout_ms = 10000;  // 10 second timeout for test

    HttpResponse response = http_post(request);

    // For GET-style test, we send empty POST which some endpoints accept
    // Or we could add http_get, but POST to httpbin.org/post works fine

    std::string result;
    if (response.success) {
        result = response.body.empty() ? "{\"status\":\"ok\"}" : response.body;
    } else {
        result = "{\"error\":\"" + cactus::ffi::escape_json_string(response.error) + "\",\"status_code\":" +
                 std::to_string(response.status_code) + "}";
    }

    if (result.length() >= buffer_size) {
        result = result.substr(0, buffer_size - 1);
    }
    std::strcpy(response_buffer, result.c_str());

    return response.status_code;
}

} // extern "C"

namespace cactus {
namespace ffi {

// Called internally when cloud handoff is triggered
// Returns true if cloud request succeeded, false otherwise
// On success, cloud_response contains the cloud API response
bool perform_cloud_fallback(const std::string& messages_json,
                            const std::string& options_json,
                            std::string& cloud_response,
                            std::string& error_message) {
    if (!is_cloud_configured()) {
        error_message = "Cloud fallback triggered but no cloud config set";
        return false;
    }

    if (!is_network_available()) {
        error_message = "Cloud fallback triggered but network not available";
        return false;
    }

    HttpResponse response = cloud_complete(messages_json, options_json);

    if (!response.success) {
        error_message = response.error.empty()
            ? "Cloud request failed with status " + std::to_string(response.status_code)
            : response.error;
        return false;
    }

    cloud_response = response.body;
    return true;
}

// Parse cloud API response and convert to cactus response format
std::string convert_cloud_response_to_cactus_format(
    const std::string& cloud_response,
    double local_time_to_first_token,
    double local_prefill_tps,
    size_t local_prompt_tokens,
    float local_confidence
) {
    // Extract response text from cloud API response
    // This handles OpenAI-compatible format: {"choices":[{"message":{"content":"..."}}]}
    std::string response_text;

    // Check if response looks like JSON (should start with '{' or '[')
    size_t first_non_space = cloud_response.find_first_not_of(" \t\n\r");
    bool looks_like_json = first_non_space != std::string::npos &&
        (cloud_response[first_non_space] == '{' || cloud_response[first_non_space] == '[');

    if (!looks_like_json) {
        // Likely an HTML error page or other non-JSON response
        response_text = "Cloud API returned non-JSON response (possibly an error page)";
    }

    // Helper lambda to extract string value starting at a position after opening quote
    auto extract_json_string = [&cloud_response](size_t start) -> std::string {
        std::string result;
        size_t end = start;
        while (end < cloud_response.size()) {
            if (cloud_response[end] == '\\' && end + 1 < cloud_response.size()) {
                end += 2; // Skip escaped character
            } else if (cloud_response[end] == '"') {
                break;
            } else {
                end++;
            }
        }
        result = cloud_response.substr(start, end - start);

        // Unescape common sequences
        std::string unescaped;
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] == '\\' && i + 1 < result.size()) {
                char next = result[i + 1];
                if (next == 'n') { unescaped += '\n'; i++; }
                else if (next == 't') { unescaped += '\t'; i++; }
                else if (next == '"') { unescaped += '"'; i++; }
                else if (next == '\\') { unescaped += '\\'; i++; }
                else { unescaped += result[i]; }
            } else {
                unescaped += result[i];
            }
        }
        return unescaped;
    };

    if (looks_like_json) {
        // Try OpenAI format first: {"choices":[{"message":{"content":"..."}}]}
        size_t choices_pos = cloud_response.find("\"choices\"");
        if (choices_pos != std::string::npos) {
            size_t content_pos = cloud_response.find("\"content\"", choices_pos);
            if (content_pos != std::string::npos) {
                size_t start = cloud_response.find("\"", content_pos + 10);
                if (start != std::string::npos) {
                    response_text = extract_json_string(start + 1);
                }
            }
        }

        // Try Anthropic format: {"content":[{"type":"text","text":"..."}]}
        if (response_text.empty()) {
            size_t content_pos = cloud_response.find("\"content\"");
            if (content_pos != std::string::npos) {
                // Look for "text" field within content array (Anthropic format)
                size_t text_pos = cloud_response.find("\"text\"", content_pos);
                if (text_pos != std::string::npos) {
                    // Skip past "text": to find the value
                    size_t colon_pos = cloud_response.find(":", text_pos + 6);
                    if (colon_pos != std::string::npos) {
                        size_t start = cloud_response.find("\"", colon_pos);
                        if (start != std::string::npos) {
                            response_text = extract_json_string(start + 1);
                        }
                    }
                }
            }
        }

        // Fallback: look for any "content" with a direct string value
        if (response_text.empty()) {
            size_t content_pos = cloud_response.find("\"content\"");
            if (content_pos != std::string::npos) {
                size_t colon_pos = cloud_response.find(":", content_pos + 9);
                if (colon_pos != std::string::npos) {
                    // Skip whitespace after colon
                    size_t val_start = cloud_response.find_first_not_of(" \t\n\r", colon_pos + 1);
                    if (val_start != std::string::npos && cloud_response[val_start] == '"') {
                        response_text = extract_json_string(val_start + 1);
                    }
                }
            }
        }
    }

    // If we couldn't parse, check for error field or return indication
    if (response_text.empty()) {
        // Check if there's an error field in the response
        size_t error_pos = cloud_response.find("\"error\"");
        if (error_pos != std::string::npos) {
            // Try to extract error message
            size_t msg_pos = cloud_response.find("\"message\"", error_pos);
            if (msg_pos != std::string::npos) {
                size_t start = cloud_response.find("\"", msg_pos + 9);
                if (start != std::string::npos) {
                    start++;
                    size_t end = cloud_response.find("\"", start);
                    if (end != std::string::npos) {
                        response_text = "Cloud API error: " + cloud_response.substr(start, end - start);
                    }
                }
            }
        }
        // Still empty? Use a placeholder indicating parse failure
        if (response_text.empty()) {
            response_text = "[Unable to parse cloud API response]";
        }
    }

    // Build cactus-format response
    std::ostringstream json;
    json << "{";
    json << "\"success\":true,";
    json << "\"error\":null,";
    json << "\"cloud_handoff\":true,";
    json << "\"cloud_completed\":true,";
    json << "\"response\":\"";

    // Escape response text for JSON
    for (char c : response_text) {
        switch (c) {
            case '"': json << "\\\""; break;
            case '\\': json << "\\\\"; break;
            case '\n': json << "\\n"; break;
            case '\r': json << "\\r"; break;
            case '\t': json << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 32) {
                    json << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    json << c;
                }
        }
    }

    json << "\",";
    json << "\"function_calls\":[],";
    json << "\"confidence\":" << std::fixed << std::setprecision(4) << local_confidence << ",";
    json << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << local_time_to_first_token << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << local_prefill_tps << ",";
    json << "\"prefill_tokens\":" << local_prompt_tokens << ",";
    json << "\"decode_tokens\":0,";
    json << "\"total_tokens\":" << local_prompt_tokens;
    json << "}";

    return json.str();
}

// Build error response when cloud fallback fails
// Note: Uses escape_json_string from cactus_utils.h
std::string construct_cloud_fallback_error_json(
    const std::string& error_message,
    float confidence,
    double time_to_first_token,
    double prefill_tps,
    size_t prompt_tokens
) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":false,";
    json << "\"error\":\"" << escape_json_string(error_message) << "\",";
    json << "\"cloud_handoff\":true,";
    json << "\"cloud_completed\":false,";
    json << "\"response\":null,";
    json << "\"function_calls\":[],";
    json << "\"confidence\":" << std::fixed << std::setprecision(4) << confidence << ",";
    json << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ",";
    json << "\"prefill_tokens\":" << prompt_tokens << ",";
    json << "\"decode_tokens\":0,";
    json << "\"total_tokens\":" << prompt_tokens;
    json << "}";
    return json.str();
}

} // namespace ffi
} // namespace cactus
