#include "test_utils.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace EngineTestUtils;

static const char* g_model = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets = std::getenv("CACTUS_TEST_ASSETS");

static std::vector<std::string> words_of(const std::string& text) {
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

static double recall(const std::vector<std::string>& ref, const std::vector<std::string>& hyp) {
    if (ref.empty()) return 1.0;
    size_t found = 0;
    for (const auto& w : ref) if (std::find(hyp.begin(), hyp.end(), w) != hyp.end()) ++found;
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
    uint32_t rate = 16000;
    bool have_fmt = false;
    std::vector<int16_t> mono;
    while (fread(tag, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (!std::strncmp(tag, "fmt ", 4)) {
            uint16_t af, c, al, bps;
            uint32_t sr, br;
            fread(&af, 2, 1, f); fread(&c, 2, 1, f); fread(&sr, 4, 1, f);
            fread(&br, 4, 1, f); fread(&al, 2, 1, f); fread(&bps, 2, 1, f);
            fmt = af; ch = c ? c : 1; bits = bps; rate = sr; have_fmt = true;
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
    if (mono.empty() || rate == 16000) return mono;
    double ratio = 16000.0 / (double)rate;
    size_t on = (size_t)(mono.size() * ratio);
    std::vector<int16_t> out(on);
    for (size_t i = 0; i < on; ++i) {
        double pos = (double)i / ratio;
        size_t i0 = (size_t)pos;
        double fr = pos - (double)i0;
        int16_t a = mono[std::min(i0, mono.size() - 1)];
        int16_t b = mono[std::min(i0 + 1, mono.size() - 1)];
        out[i] = (int16_t)((double)a + ((double)b - (double)a) * fr);
    }
    return out;
}

static std::string transcribe_oneshot(cactus_model_t model, const std::vector<int16_t>& pcm) {
    char resp[1 << 16] = {0};
    int rc = cactus_transcribe(model, nullptr, nullptr, resp, sizeof(resp),
                               R"({"telemetry_enabled": false, "auto_handoff": false, "timestamps": true})",
                               nullptr, nullptr,
                               reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size() * sizeof(int16_t));
    return rc <= 0 ? "" : json_string(std::string(resp), "response");
}

static bool test_stream_matches_oneshot() {
    std::cout << "\n=== STREAM vs ONE-SHOT ===\n";
    if (!g_model || !g_assets) { std::cout << "SKIP (model/assets not set)\n"; return true; }
    cactus_model_t model = cactus_init(g_model, nullptr, false);
    if (!model) { std::cerr << "[x] model init failed\n"; return false; }

    std::vector<int16_t> pcm = load_wav(std::string(g_assets) + "/test.wav");
    if (pcm.size() > 20 * 16000) pcm.resize(20 * 16000);
    std::string oneshot_text = transcribe_oneshot(model, pcm);
    auto golden = words_of(oneshot_text);
    if (golden.size() < 4) { cactus_destroy(model); std::cout << "  SKIP (test.wav has no usable speech)\n"; return true; }

    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, nullptr);
    if (!stream) { cactus_destroy(model); std::cerr << "[x] stream start failed\n"; return false; }
    std::string streamed;
    std::vector<char> resp(1 << 16);
    auto append = [&]() {
        std::string c = json_string(std::string(resp.data()), "confirmed");
        if (c.empty()) return;
        if (!streamed.empty() && streamed.back() != ' ' && c.front() != ' ') streamed += ' ';
        streamed += c;
    };
    for (size_t off = 0; off < pcm.size(); off += 16000) {
        size_t n = std::min<size_t>(16000, pcm.size() - off);
        resp[0] = '\0';
        if (cactus_stream_transcribe_process(stream, reinterpret_cast<const uint8_t*>(pcm.data() + off),
                                             n * sizeof(int16_t), resp.data(), resp.size()) < 0) {
            cactus_stream_transcribe_stop(stream, nullptr, 0);
            cactus_destroy(model);
            std::cerr << "[x] process failed\n";
            return false;
        }
        append();
    }
    resp[0] = '\0';
    cactus_stream_transcribe_stop(stream, resp.data(), resp.size());
    append();
    cactus_destroy(model);

    auto sw = words_of(streamed);
    std::cout << "  ONE-SHOT: " << oneshot_text << "\n";
    std::cout << "  STREAM:   " << streamed << "\n";
    double rc = recall(golden, sw), pr = recall(sw, golden);
    bool ok = rc >= 0.85 && pr >= 0.85;
    std::cout << "  recall=" << std::fixed << std::setprecision(3) << rc << " precision=" << pr
              << "  " << (ok ? "OK" : "FAIL") << "\n  Status: " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Stream Transcribe Tests");
    runner.run_test("stream_transcribe_matches_oneshot", test_stream_matches_oneshot());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
