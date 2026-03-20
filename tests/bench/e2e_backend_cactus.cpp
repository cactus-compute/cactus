#include "e2e_driver.h"

#include "ffi/cactus_ffi.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

namespace {

struct CactusHandle {
    cactus_model_t model;
};

bool available() {
    return true;
}

void* load(const char* model_path, int /*threads*/) {
    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "[cactus] cactus_init failed: " << (cactus_get_last_error() ? cactus_get_last_error() : "unknown") << "\n";
        return nullptr;
    }
    auto* h = new CactusHandle{model};
    return h;
}

// Simple JSON value extractor for numeric fields
static double json_double(const char* json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = strstr(json, needle.c_str());
    if (!pos) return 0.0;
    pos += needle.size();
    // Skip whitespace and colon
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;
    return atof(pos);
}

static int json_int(const char* json, const char* key) {
    return static_cast<int>(json_double(json, key));
}

static std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

e2e::E2EResult generate(void* handle, const char* prompt, int max_tokens) {
    auto* h = static_cast<CactusHandle*>(handle);
    e2e::E2EResult result = {};

    // Reset state for clean generation
    cactus_reset(h->model);

    // Build messages JSON (escape prompt for safety)
    std::string messages = R"([{"role": "user", "content": ")" + escape_json(prompt) + R"("}])";

    // Build options JSON with greedy decoding
    std::string options = R"({"max_tokens": )" + std::to_string(max_tokens) +
                          R"(, "temperature": 0.0, "top_p": 0.0, "top_k": 1, "telemetry_enabled": false,)"
                          R"( "stop_sequences": ["<|im_end|>", "<end_of_turn>"]})";

    char response[8192] = {};
    auto wall_start = std::chrono::steady_clock::now();
    int ret = cactus_complete(h->model, messages.c_str(), response, sizeof(response),
                              options.c_str(), nullptr, nullptr, nullptr);
    auto wall_end = std::chrono::steady_clock::now();

    if (ret <= 0) {
        std::cerr << "[cactus] cactus_complete failed\n";
        return result;
    }

    // Parse metrics from response JSON
    result.prefill_tokens = json_int(response, "prefill_tokens");
    result.decode_tokens = json_int(response, "decode_tokens");
    result.prefill_tps = json_double(response, "prefill_tps");
    result.decode_tps = json_double(response, "decode_tps");
    result.ttft_ms = json_double(response, "time_to_first_token_ms");
    result.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    return result;
}

void unload(void* handle) {
    auto* h = static_cast<CactusHandle*>(handle);
    cactus_destroy(h->model);
    delete h;
}

static int reg = [] {
    e2e::register_e2e_backend({
        "cactus", "cactus",
        available, load, generate, unload
    });
    return 0;
}();

} // namespace
