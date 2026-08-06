#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace EngineTestUtils;

static const char* g_model_path         = std::getenv("CACTUS_TEST_MODEL");
static const char* g_transcription_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path        = std::getenv("CACTUS_TEST_ASSETS");

static constexpr size_t kPrefillTokens = 1000;
static constexpr size_t kDecodeTokens  = 100;
static constexpr int    kMeasuredRuns  = 3;

static void print_mean(const char* label, const char* unit, const std::vector<double>& samples) {
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::cout << "├─ " << std::left << std::setw(17) << label << std::right
              << std::fixed << std::setprecision(2) << std::setw(12) << mean
              << " " << unit << "\n";
}

static void print_header(const char* title) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("  ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";
}

static std::string build_context_messages(int filler) {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < filler; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < filler; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";
    return msg;
}

static bool build_exact_prompt_tokens(cactus_model_t model, std::vector<uint32_t>& tokens) {
    for (int filler = 50; filler <= 3200 && tokens.size() < kPrefillTokens; filler *= 2) {
        std::string msg = build_context_messages(filler);
        std::vector<char> prompt(msg.size() + 4096);
        if (cactus_render_prompt(model, msg.c_str(), nullptr, nullptr,
                                 prompt.data(), prompt.size()) < 0) {
            std::cerr << "[✗] Failed to render prompt\n";
            return false;
        }
        size_t token_len = 0;
        std::vector<uint32_t> buffer(prompt.size());
        if (cactus_tokenize(model, prompt.data(), buffer.data(), buffer.size(), &token_len) != 0) {
            std::cerr << "[✗] Failed to tokenize prompt\n";
            return false;
        }
        buffer.resize(token_len);
        tokens = std::move(buffer);
    }
    if (tokens.size() < kPrefillTokens) {
        std::cerr << "[✗] Could not build a " << kPrefillTokens << "-token prompt\n";
        return false;
    }
    std::cout << "├─ Prompt tokens: " << kPrefillTokens
              << " (rendered " << tokens.size() << ", truncated)\n"
              << "├─ Decode tokens: " << kDecodeTokens << "\n";
    tokens.resize(kPrefillTokens);
    return true;
}

static bool llm_benchmark() {
    print_header("LLM BENCHMARK (1k prefill / 100 decode)");

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::vector<uint32_t> tokens;
    if (!build_exact_prompt_tokens(model, tokens)) {
        cactus_destroy(model);
        return false;
    }

    std::vector<double> prefill_tps, decode_tps, peak_ram;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        char response[16384];
        int rc = cactus_benchmark_tokens(model, tokens.data(), tokens.size(), kDecodeTokens,
                                         response, sizeof(response));
        std::string json(response);
        bool exact = rc >= 0 &&
            json.find("\"success\":true") != std::string::npos &&
            json_number(json, "prompt_tokens")     == static_cast<double>(kPrefillTokens) &&
            json_number(json, "completion_tokens") == static_cast<double>(kDecodeTokens);
        if (!exact) {
            std::cerr << "[✗] Benchmark run failed: " << json << "\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: prefill_tps=" << std::fixed << std::setprecision(2)
                      << json_number(json, "prefill_tps")
                      << " decode_tps=" << json_number(json, "decode_tps") << "\n";
            continue;
        }
        prefill_tps.push_back(json_number(json, "prefill_tps"));
        decode_tps.push_back(json_number(json, "decode_tps"));
        peak_ram.push_back(json_number(json, "peak_ram_usage_mb"));
    }
    cactus_destroy(model);

    std::cout << "\n[Mean of " << kMeasuredRuns << " runs]\n";
    print_mean("prefill_tps", "tok/s", prefill_tps);
    print_mean("decode_tps",  "tok/s", decode_tps);
    print_mean("peak_ram",    "MB",    peak_ram);
    std::cout << "└─ Status: PASSED ✓\n";
    return true;
}

static bool vlm_benchmark() {
    print_header("VLM BENCHMARK (image embed / decode)");

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string image_path = std::string(g_assets_path) + "/test_monkey.png";
    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    const size_t embed_buffer_size = 16 * 1024 * 1024;
    std::vector<float> embeddings(embed_buffer_size / sizeof(float));

    std::vector<double> embed_ms;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        size_t dim = 0;
        Timer t;
        int rc = cactus_image_embed(model, image_path.c_str(), embeddings.data(),
                                    embed_buffer_size, &dim);
        double elapsed = t.elapsed_ms();
        if (rc <= 0 || dim == 0) {
            std::cerr << "[✗] Image embedding failed\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: image_embed=" << std::fixed << std::setprecision(2)
                      << elapsed << "ms (dim " << dim << ")\n";
            continue;
        }
        embed_ms.push_back(elapsed);
    }

    std::vector<double> audio_ms;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        size_t dim = 0;
        Timer t;
        int rc = cactus_audio_embed(model, audio_path.c_str(), embeddings.data(),
                                    embed_buffer_size, &dim);
        double elapsed = t.elapsed_ms();
        if (rc <= 0 || dim == 0) {
            std::cerr << "[✗] Audio embedding failed\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: audio_embed=" << std::fixed << std::setprecision(2)
                      << elapsed << "ms (dim " << dim << ")\n";
            continue;
        }
        audio_ms.push_back(elapsed);
    }

    std::string messages = "[{\"role\": \"user\", "
        "\"content\": \"Describe this image in detail.\", "
        "\"images\": [\"" + image_path + "\"]}]";
    const char* options = R"({
        "max_tokens": 100,
        "temperature": 0,
        "top_k": 1,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "telemetry_enabled": false,
        "auto_handoff": false
    })";

    std::vector<double> decode_tps;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        cactus_reset(model);
        char response[8192];
        int rc = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                 options, nullptr, nullptr, nullptr, nullptr, 0);
        std::string json(response);
        double tps = json_number(json, "decode_tps");
        if (rc <= 0 || tps <= 0.0) {
            std::cerr << "[✗] VLM completion failed: " << json << "\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: decode_tps=" << tps << " (decode_tokens="
                      << static_cast<int>(json_number(json, "decode_tokens")) << ")\n";
            continue;
        }
        decode_tps.push_back(tps);
    }
    cactus_destroy(model);

    std::cout << "\n[Mean of " << kMeasuredRuns << " runs]\n";
    print_mean("image_embed", "ms",    embed_ms);
    print_mean("audio_embed", "ms",    audio_ms);
    print_mean("decode_tps",  "tok/s", decode_tps);
    std::cout << "└─ Status: PASSED ✓\n";
    return true;
}

static bool transcribe_benchmark() {
    print_header("TRANSCRIBE BENCHMARK (audio embed / stt)");

    cactus_model_t model = cactus_init(g_transcription_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize transcription model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    const size_t embed_buffer_size = 1024 * 1024;
    std::vector<float> embeddings(embed_buffer_size / sizeof(float));

    std::vector<double> embed_ms;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        size_t dim = 0;
        Timer t;
        int rc = cactus_audio_embed(model, audio_path.c_str(), embeddings.data(),
                                    embed_buffer_size, &dim);
        double elapsed = t.elapsed_ms();
        if (rc <= 0 || dim == 0) {
            std::cerr << "[✗] Audio embedding failed\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: audio_embed=" << std::fixed << std::setprecision(2)
                      << elapsed << "ms (dim " << dim << ")\n";
            continue;
        }
        embed_ms.push_back(elapsed);
    }

    const char* options = R"({"max_tokens": 500, "telemetry_enabled": false, "auto_handoff": false})";
    std::vector<double> decode_tps, transcribe_ms;
    for (int run = 0; run <= kMeasuredRuns; run++) {
        char response[1 << 15] = {0};
        Timer t;
        int rc = cactus_transcribe(model, audio_path.c_str(), nullptr,
                                   response, sizeof(response), options,
                                   nullptr, nullptr, nullptr, 0);
        double elapsed = t.elapsed_ms();
        std::string json(response);
        if (rc <= 0 || json_string(json, "response").length() <= 5) {
            std::cerr << "[✗] Transcription failed: " << json << "\n";
            cactus_destroy(model);
            return false;
        }
        if (run == 0) {
            std::cout << "├─ Warmup: transcribe=" << std::fixed << std::setprecision(2)
                      << elapsed << "ms decode_tps=" << json_number(json, "decode_tps") << "\n";
            continue;
        }
        decode_tps.push_back(json_number(json, "decode_tps"));
        transcribe_ms.push_back(elapsed);
    }
    cactus_destroy(model);

    std::cout << "\n[Mean of " << kMeasuredRuns << " runs]\n";
    print_mean("audio_embed",      "ms",    embed_ms);
    print_mean("transcribe_total", "ms",    transcribe_ms);
    print_mean("decode_tps",       "tok/s", decode_tps);
    std::cout << "└─ Status: PASSED ✓\n";
    return true;
}

int main() {
    TestUtils::TestRunner runner("Benchmark");
    runner.run_test("llm_benchmark",        llm_benchmark());
    runner.run_test("vlm_benchmark",        vlm_benchmark());
    runner.run_test("transcribe_benchmark", transcribe_benchmark());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
