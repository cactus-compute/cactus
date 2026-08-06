#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace EngineTestUtils;

static const char* g_transcription_model_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* g_options = R"({"max_tokens": 200, "telemetry_enabled": false, "auto_handoff": false})";

bool test_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        TRANSCRIPTION TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    char response[1 << 15] = {0};

    int rc = cactus_transcribe(model, audio_path.c_str(), nullptr,
                               response, sizeof(response), g_options,
                               nullptr, nullptr, nullptr, 0);
    std::string file_transcript = json_string(std::string(response), "response");
    std::cout << "├─ File transcript: " << file_transcript << "\n";
    bool file_ok = rc > 0 && file_transcript.length() > 5;
    if (!file_ok) std::cerr << "[✗] File transcription failed: " << response << "\n";

    FILE* wav_file = fopen(audio_path.c_str(), "rb");
    if (!wav_file) {
        std::cerr << "[✗] Failed to open audio file\n";
        cactus_destroy(model);
        return false;
    }
    fseek(wav_file, 44, SEEK_SET);
    std::vector<uint8_t> pcm_data;
    uint8_t buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), wav_file)) > 0) {
        pcm_data.insert(pcm_data.end(), buf, buf + n);
    }
    fclose(wav_file);

    char pcm_response[1 << 15] = {0};
    int pcm_rc = cactus_transcribe(model, nullptr, nullptr,
                                   pcm_response, sizeof(pcm_response), g_options,
                                   nullptr, nullptr,
                                   pcm_data.data(), pcm_data.size());
    std::string pcm_transcript = json_string(std::string(pcm_response), "response");
    std::cout << "├─ PCM transcript:  " << pcm_transcript << "\n";
    bool pcm_ok = pcm_rc > 0 && pcm_transcript.length() > 5;
    if (!pcm_ok) std::cerr << "[✗] PCM transcription failed: " << pcm_response << "\n";

    cactus_destroy(model);

    bool passed = file_ok && pcm_ok;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

int main() {
    TestUtils::TestRunner runner("STT Tests");
    runner.run_test("transcription", test_transcription());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
