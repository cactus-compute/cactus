#include "cactus_cloud.h"
#include "telemetry/telemetry.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef CACTUS_USE_CURL
#include <curl/curl.h>
#endif

namespace cactus {
namespace ffi {

namespace {

std::string compact_error_detail(std::string detail) {
    detail = trim_string(detail);
    if (detail.empty()) return {};

    for (char& c : detail) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    while (detail.find("  ") != std::string::npos) {
        detail.erase(detail.find("  "), 1);
    }

    constexpr size_t kMaxLen = 160;
    if (detail.size() > kMaxLen) {
        detail = detail.substr(0, kMaxLen) + "...";
    }
    return detail;
}

} // namespace

std::string resolve_cloud_api_key(const char* cloud_key_param) {
    const char* env_cloud = std::getenv("CACTUS_CLOUD_KEY");
    const char* env_cloud_legacy = std::getenv("CACTUS_CLOUD_API_KEY"); // keeping this cuz Matthew Hayes had some code reliant on Cactus_cloud_api_key
    std::string resolved_cloud_key;
    bool key_from_param_or_env = false;

    if (cloud_key_param && *cloud_key_param) {
        resolved_cloud_key = cloud_key_param;
        key_from_param_or_env = true;
    } else if (env_cloud && *env_cloud) {
        resolved_cloud_key = env_cloud;
        key_from_param_or_env = true;
    } else if (env_cloud_legacy && *env_cloud_legacy) {
        resolved_cloud_key = env_cloud_legacy;
        key_from_param_or_env = true;
    } else {
        resolved_cloud_key = cactus::telemetry::loadCachedCloudApiKey();
    }

    resolved_cloud_key = trim_string(resolved_cloud_key);

    const bool should_cache =
        !resolved_cloud_key.empty() &&
        key_from_param_or_env;
    if (should_cache) {
        cactus::telemetry::cacheCloudApiKey(resolved_cloud_key.c_str());
    }

    return resolved_cloud_key;
}

namespace {

#ifdef CACTUS_USE_CURL
static std::atomic<bool> g_warned_missing_cloud_api_key{false};

static void apply_curl_tls_trust(CURL* curl) {
    if (!curl) return;
    const char* ca_bundle = std::getenv("CACTUS_CA_BUNDLE");
    if (ca_bundle && ca_bundle[0] != '\0') {
        curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle);
    }
#if defined(__ANDROID__)
    const char* ca_path = std::getenv("CACTUS_CA_PATH");
    if (ca_path && ca_path[0] != '\0') {
        curl_easy_setopt(curl, CURLOPT_CAPATH, ca_path);
    } else {
        curl_easy_setopt(curl, CURLOPT_CAPATH, "/system/etc/security/cacerts");
    }
#endif
}

static size_t curl_write_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

static bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    if (size <= 0) return false;
    in.seekg(0, std::ios::beg);

    out.resize(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(out.data()), size)) return false;
    return true;
}

static std::string infer_mime_type(const std::string& path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return "image/jpeg";
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == "png") return "image/png";
    if (ext == "webp") return "image/webp";
    if (ext == "gif") return "image/gif";
    return "image/jpeg";
}

static std::string build_cloud_text_prompt(const CloudCompletionRequest& request) {
    std::ostringstream oss;
    oss << "You are continuing an existing assistant conversation.\\n";
    oss << "Output contract:\\n";
    oss << "1) Never include role prefixes like 'assistant:'.\\n";
    oss << "2) Never include markdown/code fences/backticks.\\n";
    oss << "3) Return only the final assistant answer text unless a tool call is required.\\n";
    oss << "4) If a tool call is required, return ONLY JSON with this exact shape:\\n";
    oss << "[{\"name\":\"tool_name\",\"arguments\":{\"arg\":\"value\"}}]\\n";
    oss << "5) Do not include any prose before or after that JSON tool-call output.\\n";
    oss << "\\nConversation:\\n";
    for (const auto& m : request.messages) {
        oss << "[" << m.role << "] " << m.content << "\\n";
    }

    if (!request.tools.empty()) {
        oss << "\\nAvailable tools JSON (use only these tool names and arguments):\\n";
        oss << serialize_tools_json(request.tools) << "\\n";
        oss << "If tools are relevant, prefer the strict JSON tool-call output contract above.\\n";
    }

    if (!request.local_output.empty()) {
        oss << "\\nLocal model draft (useful fallback reference, may be low confidence):\\n";
        oss << request.local_output << "\\n";
    }

    return oss.str();
}

static std::string call_cloud_endpoint(const std::string& url,
                                       const std::string& payload,
                                       long timeout_ms,
                                       const char* cloud_key_param,
                                       std::string& err_out) {
    std::string api_key = resolve_cloud_api_key(cloud_key_param);
    if (api_key.empty()) {
        if (!g_warned_missing_cloud_api_key.exchange(true)) {
            CACTUS_LOG_WARN("cloud_handoff", "No cloud key found (cloud_key param, CACTUS_CLOUD_KEY, or CACTUS_CLOUD_API_KEY env); cloud handoff will fall back to local output");
        }
        err_out = "missing_api_key";
        return {};
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        err_out = "curl_init_failed";
        return {};
    }

    std::string response_body;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("X-API-Key: " + api_key).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);

    if (!env_flag_enabled("CACTUS_CLOUD_STRICT_SSL")) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    } else {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        apply_curl_tls_trust(curl);
    }

    CURLcode res = curl_easy_perform(curl);
    long http_status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        err_out = curl_easy_strerror(res);
        return {};
    }

    if (http_status >= 400) {
        std::string detail = json_string_field(response_body, "error");
        if (detail.empty()) detail = json_string_field(response_body, "message");
        if (detail.empty()) detail = json_string_field(response_body, "detail");
        detail = compact_error_detail(detail);

        err_out = "http_" + std::to_string(http_status);
        if (!detail.empty()) {
            err_out += ":" + detail;
        }
        return {};
    }

    return response_body;
}
#endif

} // namespace

std::string cloud_base64_encode(const uint8_t* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<uint32_t>(data[i + 2]);
        out += table[(n >> 18) & 0x3F];
        out += table[(n >> 12) & 0x3F];
        out += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? table[n & 0x3F] : '=';
    }
    return out;
}

std::vector<uint8_t> cloud_build_wav(const uint8_t* pcm, size_t pcm_bytes) {
    constexpr uint32_t sample_rate = 16000;
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits = 16;
    const uint32_t byte_rate = sample_rate * channels * bits / 8;
    const uint16_t block_align = channels * bits / 8;
    const uint32_t data_size = static_cast<uint32_t>(pcm_bytes);
    const uint32_t file_size = 36 + data_size;

    std::vector<uint8_t> wav(44 + pcm_bytes);
    auto w16 = [&](size_t off, uint16_t v) {
        wav[off] = v & 0xFF;
        wav[off + 1] = v >> 8;
    };
    auto w32 = [&](size_t off, uint32_t v) {
        wav[off] = v & 0xFF;
        wav[off + 1] = (v >> 8) & 0xFF;
        wav[off + 2] = (v >> 16) & 0xFF;
        wav[off + 3] = (v >> 24) & 0xFF;
    };

    std::memcpy(wav.data(), "RIFF", 4);
    w32(4, file_size);
    std::memcpy(wav.data() + 8, "WAVE", 4);
    std::memcpy(wav.data() + 12, "fmt ", 4);
    w32(16, 16);
    w16(20, 1);
    w16(22, channels);
    w32(24, sample_rate);
    w32(28, byte_rate);
    w16(32, block_align);
    w16(34, bits);
    std::memcpy(wav.data() + 36, "data", 4);
    w32(40, data_size);
    std::memcpy(wav.data() + 44, pcm, pcm_bytes);
    return wav;
}

CloudResponse cloud_transcribe_request(const std::string& audio_b64,
                                       const std::string& fallback_text,
                                       long timeout_seconds,
                                       const char* cloud_key) {
#ifdef CACTUS_USE_CURL
    std::string endpoint = "https://104.198.76.3/api/v1/transcribe";

    std::string payload = "{\"audio\":\"" + audio_b64 + "\",\"mime_type\":\"audio/wav\",\"language\":\"en-US\"}";

    std::string err;
    std::string body = call_cloud_endpoint(endpoint, payload, timeout_seconds * 1000L, cloud_key, err);
    if (body.empty()) {
        return {fallback_text, "", false, err.empty() ? "request_failed" : err};
    }

    std::string transcript = json_string_field(body, "transcript");
    if (transcript.empty()) {
        transcript = json_string_field(body, "text");
    }
    if (transcript.empty()) {
        transcript = json_string_field(body, "response");
    }
    if (transcript.empty()) {
        transcript = json_string_field(body, "analysis");
    }
    if (transcript.empty()) {
        std::string detail = json_string_field(body, "error");
        if (detail.empty()) detail = json_string_field(body, "message");
        if (detail.empty()) detail = json_string_field(body, "detail");
        detail = compact_error_detail(detail);
        if (!detail.empty()) {
            return {fallback_text, "", false, "missing_transcript:" + detail};
        }
        return {fallback_text, "", false, "missing_transcript"};
    }

    std::string api_key_hash = json_string_field(body, "api_key_hash");
    return {transcript, api_key_hash, true, ""};
#else
    (void)audio_b64;
    (void)timeout_seconds;
    return {fallback_text, "", false, "curl_not_enabled"};
#endif
}

CloudCompletionResult cloud_complete_request(const CloudCompletionRequest& request,
                                             long timeout_ms) {
#ifdef CACTUS_USE_CURL
    std::string text_model = env_or_default("CACTUS_CLOUD_TEXT_MODEL", "gemini-2.5-flash");
    std::string vlm_model = env_or_default("CACTUS_CLOUD_VLM_MODEL", "gemini-2.5-flash");

    std::string endpoint;
    std::string payload;

    if (request.has_images) {
        std::vector<std::string> image_paths;
        for (const auto& message : request.messages) {
            for (const auto& image : message.images) {
                if (!image.empty()) {
                    image_paths.push_back(image);
                }
            }
        }

        if (image_paths.size() > 1) {
            return {false, false, "", {}, "multiple_images_not_supported"};
        }

        std::string image_path;
        if (!image_paths.empty()) {
            image_path = image_paths.back();
        }
        if (image_path.empty()) {
            return {false, false, "", {}, "missing_image_path"};
        }

        std::vector<uint8_t> img_bytes;
        if (!read_file_bytes(image_path, img_bytes)) {
            return {false, false, "", {}, "image_read_failed"};
        }

        std::string img_b64 = cloud_base64_encode(img_bytes.data(), img_bytes.size());
        std::string mime = infer_mime_type(image_path);
        std::string prompt = build_cloud_text_prompt(request);

        endpoint = "https://104.198.76.3/api/v1/vlm";
        payload = "{"
                  "\"image\":\"" + img_b64 + "\"," 
                  "\"mime_type\":\"" + mime + "\"," 
                  "\"prompt\":\"" + escape_json_string(prompt) + "\"," 
                  "\"language\":\"en-US\"," 
                  "\"model\":\"" + escape_json_string(vlm_model) + "\""
                  "}";
    } else {
        endpoint = "https://104.198.76.3/api/v1/text";
        std::string text = build_cloud_text_prompt(request);
        payload = "{"
                  "\"text\":\"" + escape_json_string(text) + "\"," 
                  "\"language\":\"en-US\"," 
                  "\"model\":\"" + escape_json_string(text_model) + "\""
                  "}";
    }

    std::string err;
    const char* cloud_key = request.cloud_key.empty() ? nullptr : request.cloud_key.c_str();
    std::string body = call_cloud_endpoint(endpoint, payload, timeout_ms, cloud_key, err);
    if (body.empty()) {
        return {false, false, "", {}, err.empty() ? "request_failed" : err};
    }

    std::vector<std::string> function_calls;
    std::string calls_json = json_array_field(body, "function_calls");
    if (calls_json != "[]") {
        function_calls = split_json_array(calls_json);
    }

    std::string response = json_string_field(body, "text");
    if (response.empty()) {
        response = json_string_field(body, "analysis");
    }

    if (!response.empty() && function_calls.empty()) {
        std::string regular_response;
        parse_function_calls_from_response(response, regular_response, function_calls);
        response = regular_response;
    }

    if (!response.empty() && function_calls.empty()) {
        auto first = response.find_first_not_of(" \t\n\r");
        auto last = response.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos) {
            std::string trimmed = response.substr(first, last - first + 1);
            if (!trimmed.empty() && trimmed.front() == '[' && trimmed.back() == ']' &&
                trimmed.find("\"name\"") != std::string::npos) {
                function_calls = split_json_array(trimmed);
                response.clear();
            }
        }
    }

    if (response.empty() && function_calls.empty()) {
        return {false, false, "", {}, "missing_text"};
    }

    CloudCompletionResult out;
    out.ok = true;
    out.used_cloud = true;
    out.response = response;
    out.function_calls = std::move(function_calls);
    return out;
#else
    (void)request;
    (void)timeout_ms;
    return {false, false, "", {}, "curl_not_enabled"};
#endif
}

} // namespace ffi
} // namespace cactus
