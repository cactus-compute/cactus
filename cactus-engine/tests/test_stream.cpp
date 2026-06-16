#include "test_utils.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

using namespace EngineTestUtils;

static const char* g_transcription_model_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static std::vector<std::string> normalized_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) {
        std::string n;
        for (char c : w) if (std::isalnum((unsigned char)c)) n += (char)std::tolower((unsigned char)c);
        if (!n.empty()) words.push_back(n);
    }
    return words;
}

static double word_recall(const std::vector<std::string>& ref, const std::vector<std::string>& hyp) {
    if (ref.empty()) return 1.0;
    size_t found = 0;
    for (const auto& w : ref)
        if (std::find(hyp.begin(), hyp.end(), w) != hyp.end()) ++found;
    return (double)found / (double)ref.size();
}

static std::vector<int16_t> load_wav(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    char tag[4];
    uint32_t sz;
    if (fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "RIFF", 4)) { fclose(f); return {}; }
    fread(&sz, 4, 1, f);
    if (fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "WAVE", 4)) { fclose(f); return {}; }
    uint16_t ch = 1, bits = 16, fmt = 0;
    bool have_fmt = false;
    std::vector<int16_t> mono;
    while (fread(tag, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (!std::strncmp(tag, "fmt ", 4)) {
            uint16_t af, c, al, bps; uint32_t sr, br;
            fread(&af, 2, 1, f); fread(&c, 2, 1, f); fread(&sr, 4, 1, f);
            fread(&br, 4, 1, f); fread(&al, 2, 1, f); fread(&bps, 2, 1, f);
            fmt = af; ch = c ? c : 1; bits = bps; have_fmt = true;
            if (sz > 16) fseek(f, (long)sz - 16, SEEK_CUR);
        } else if (!std::strncmp(tag, "data", 4)) {
            if (!have_fmt || fmt != 1 || bits != 16) { fclose(f); return {}; }
            size_t frames = sz / (2u * ch);
            std::vector<int16_t> raw(sz / 2);
            fread(raw.data(), 2, raw.size(), f);
            mono.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                int acc = 0;
                for (uint16_t k = 0; k < ch; ++k) acc += raw[i * ch + k];
                mono[i] = (int16_t)(acc / ch);
            }
            break;
        } else {
            fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
        }
    }
    fclose(f);
    return mono;
}

static std::string run_stream(cactus_model_t model, const std::vector<int16_t>& pcm, size_t chunk_samples) {
    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, nullptr);
    if (!stream) return "";

    std::string transcript;
    std::vector<char> resp(1 << 16);
    auto append = [&]() {
        std::string c = json_string(std::string(resp.data()), "confirmed");
        if (c.empty()) return;
        if (!transcript.empty()) transcript += ' ';
        transcript += c;
    };

    for (size_t off = 0; off < pcm.size(); off += chunk_samples) {
        size_t n = std::min(chunk_samples, pcm.size() - off);
        resp[0] = '\0';
        if (cactus_stream_transcribe_process(
                stream, reinterpret_cast<const uint8_t*>(pcm.data() + off), n * sizeof(int16_t),
                resp.data(), resp.size()) < 0) {
            cactus_stream_transcribe_stop(stream, nullptr, 0);
            return transcript;
        }
        append();
    }

    resp[0] = '\0';
    cactus_stream_transcribe_stop(stream, resp.data(), resp.size());
    append();
    return transcript;
}

static std::string transcribe_full(cactus_model_t model, const std::vector<int16_t>& pcm) {
    char response[1 << 15] = {0};
    int rc = cactus_transcribe(model, nullptr, nullptr, response, sizeof(response),
                               R"({"telemetry_enabled": false, "auto_handoff": false})",
                               nullptr, nullptr,
                               reinterpret_cast<const uint8_t*>(pcm.data()),
                               pcm.size() * sizeof(int16_t));
    if (rc <= 0) return "";
    return json_string(std::string(response), "response");
}

static bool test_stream_basic() {
    std::cout << "\n=== STREAM TEST ===\n";
    if (!g_transcription_model_path) { std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n"; return true; }
    if (!g_assets_path) { std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n"; return true; }

    std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + "/test.wav");
    if (pcm.empty()) { std::cerr << "[✗] Failed to load test.wav\n"; return false; }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::string reference = transcribe_full(model, pcm);
    std::string streamed = run_stream(model, pcm, 16000);
    cactus_destroy(model);

    auto ref_words = normalized_words(reference);
    auto stream_words = normalized_words(streamed);
    double recall = word_recall(ref_words, stream_words);

    std::cout << "├─ reference: " << reference << "\n"
              << "├─ streamed:  " << streamed << "\n"
              << "├─ recall=" << recall << " (ref=" << ref_words.size()
              << " streamed=" << stream_words.size() << ")\n";

    bool passed = !stream_words.empty() && recall >= 0.6;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

int main() {
    TestUtils::TestRunner runner("Stream Tests");
    runner.run_test("stream_basic", test_stream_basic());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
