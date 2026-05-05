// Fast decode diagnostic: loads model, runs 10 tokens on a short prompt.
// Target: <10s total. Use with CACTUS_TEST_MODEL and optionally CACTUS_TENSOR_DUMP.
//
// Usage:
//   CACTUS_TEST_MODEL=/path/to/weights CACTUS_TENSOR_DUMP=/tmp/tensors ./test_fast
//
// Output on stderr: [step N] sampled_tok=ID (text=...)
// Token IDs go to stdout for easy diffing against HF reference.

#include "../cactus_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <cctype>

static const int N_TOKENS = 50;

static int prompt_repeats() {
    const char* env = std::getenv("CACTUS_TEST_PROMPT_REPEATS");
    if (!env || !*env) return 130;
    int value = std::atoi(env);
    return value > 0 ? value : 1;
}

static std::string build_prompt() {
    std::string content;
    content.reserve(6400);
    int repeats = prompt_repeats();
    for (int i = 0; i < repeats; ++i) {
        content += "The quick brown fox jumps over the lazy dog. ";
    }

    return std::string("[")
        + "{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
        + "{\"role\":\"user\",\"content\":\"" + content + "\"}"
        + "]";
}

static const char* OPTIONS = R"({
    "max_tokens": 50,
    "temperature": 0.0,
    "stop_sequences": ["<end_of_turn>", "<|im_end|>"],
    "telemetry_enabled": false
})";

static std::string g_last_token_text;
static int g_tok_count = 0;
static std::chrono::steady_clock::time_point g_first_tok_time;

static void token_callback(const char* text, uint32_t token_id, void* /*ud*/) {
    if (g_tok_count == 0) g_first_tok_time = std::chrono::steady_clock::now();
    g_tok_count++;
    g_last_token_text = text ? text : "";
    printf("%u\n", token_id);
    fflush(stdout);
}

static double json_number(const char* json, const char* key) {
    if (!json || !key) return -1.0;

    std::string needle = std::string("\"") + key + "\"";
    const char* pos = std::strstr(json, needle.c_str());
    if (!pos) return -1.0;

    pos = std::strchr(pos + needle.size(), ':');
    if (!pos) return -1.0;
    ++pos;

    while (*pos && std::isspace(static_cast<unsigned char>(*pos))) ++pos;

    char* end = nullptr;
    double value = std::strtod(pos, &end);
    return end && end != pos ? value : -1.0;
}

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        fprintf(stderr, "CACTUS_TEST_MODEL not set\n");
        return 1;
    }

    fprintf(stderr, "Loading model from %s\n", model_path);
    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    if (const char* profile_after_init = std::getenv("CACTUS_PROFILE_AFTER_INIT")) {
        if (*profile_after_init) {
            setenv("CACTUS_PROFILE", profile_after_init, 1);
        }
    }
    std::string prompt = build_prompt();
    fprintf(stderr, "Model loaded. Running prompt_repeats=%d, max_tokens=%d...\n", prompt_repeats(), N_TOKENS);

    auto t0 = std::chrono::steady_clock::now();
    char response[65536] = {};
    int result = cactus_complete(model, prompt.c_str(), response, sizeof(response),
                                 OPTIONS, nullptr, token_callback, nullptr, nullptr, 0);
    auto t1 = std::chrono::steady_clock::now();

    if (g_tok_count > 1) {
        double decode_ms = std::chrono::duration<double, std::milli>(t1 - g_first_tok_time).count();
        double ttft_ms   = std::chrono::duration<double, std::milli>(g_first_tok_time - t0).count();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double json_ttft = json_number(response, "time_to_first_token_ms");
        double prefill_tokens = json_number(response, "prefill_tokens");
        double prefill_tps = json_number(response, "prefill_tps");
        double decode_tokens = json_number(response, "decode_tokens");
        double decode_tps = json_number(response, "decode_tps");

        if (json_ttft > 0.0) ttft_ms = json_ttft;
        if (decode_tps <= 0.0) decode_tps = (g_tok_count - 1) * 1000.0 / decode_ms;
        if (decode_tokens <= 0.0) decode_tokens = g_tok_count;
        if (prefill_tps <= 0.0 && prefill_tokens > 0.0 && ttft_ms > 0.0) {
            prefill_tps = prefill_tokens * 1000.0 / ttft_ms;
        }

        fprintf(stderr,
                "TTFT: %.2f ms | prefill_tokens: %.0f | prefill_tps: %.2f tok/s | decode_tokens: %.0f | decode_tps: %.2f tok/s | wall: %.2f ms\n",
                ttft_ms, prefill_tokens, prefill_tps, decode_tokens, decode_tps, wall_ms);
    }
    cactus_destroy(model);
    return result > 0 ? 0 : 1;
}
