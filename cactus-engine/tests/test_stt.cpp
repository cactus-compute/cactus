#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

bool test_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        TRANSCRIPTION TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }
    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    char response[1 << 15] = {0};

    Timer timer;
    int rc = cactus_transcribe(model, audio_path.c_str(), nullptr,
                               response, sizeof(response),
                               R"({"max_tokens": 200, "telemetry_enabled": false})",
                               nullptr, nullptr, nullptr, 0);
    double elapsed = timer.elapsed_ms();

    if (rc <= 0) {
        std::cerr << "[✗] Transcription failed: " << response << "\n";
        cactus_destroy(model);
        return false;
    }

    std::string response_str(response);
    std::string transcript = json_string(response_str, "response");

    std::cout << "├─ Transcript: " << transcript << "\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n";

    Metrics m;
    m.parse(response);
    m.print_json();

    cactus_destroy(model);

    bool passed = rc > 0 && !transcript.empty() && transcript.length() > 5;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

bool test_transcription_pcm() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║      TRANSCRIPTION PCM TEST               ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }
    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    FILE* wav_file = fopen(audio_path.c_str(), "rb");
    if (!wav_file) {
        std::cerr << "[✗] Failed to open audio file\n";
        cactus_destroy(model);
        return false;
    }

    // Skip 44-byte WAV header
    fseek(wav_file, 44, SEEK_SET);
    std::vector<uint8_t> pcm_data;
    uint8_t buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), wav_file)) > 0) {
        pcm_data.insert(pcm_data.end(), buf, buf + n);
    }
    fclose(wav_file);

    char response[1 << 15] = {0};

    Timer timer;
    int rc = cactus_transcribe(model, nullptr, nullptr,
                               response, sizeof(response),
                               R"({"max_tokens": 200, "telemetry_enabled": false})",
                               nullptr, nullptr,
                               pcm_data.data(), pcm_data.size());
    double elapsed = timer.elapsed_ms();

    if (rc <= 0) {
        std::cerr << "[✗] PCM transcription failed: " << response << "\n";
        cactus_destroy(model);
        return false;
    }

    std::string response_str(response);
    std::string transcript = json_string(response_str, "response");

    std::cout << "├─ Transcript: " << transcript << "\n"
              << "├─ PCM size: " << pcm_data.size() << " bytes\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n";

    cactus_destroy(model);

    bool passed = rc > 0 && !transcript.empty() && transcript.length() > 5;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

int main() {
    TestUtils::TestRunner runner("STT Tests");
    runner.run_test("transcription", test_transcription());
    runner.run_test("transcription_pcm", test_transcription_pcm());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
