#include "test_utils.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

using namespace EngineTestUtils;

static const char* g_transcription_model_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");
static constexpr size_t kLiveChunkSamples = 4000;

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

static std::vector<int16_t> make_silence(int seconds) {
    return std::vector<int16_t>((size_t)seconds * 16000, 0);
}

static std::vector<int16_t> sine_tone(int seconds, double freq, double amp) {
    std::vector<int16_t> s((size_t)seconds * 16000);
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (int16_t)(amp * 32767.0 * std::sin(2.0 * 3.14159265358979 * freq * (double)i / 16000.0));
    return s;
}

static std::vector<int16_t> white_noise(int seconds, double amp, uint32_t seed) {
    std::vector<int16_t> s((size_t)seconds * 16000);
    uint32_t st = seed;
    for (auto& v : s) {
        st = st * 1664525u + 1013904223u;
        double r = ((double)(st >> 9) / 4194304.0) - 1.0;
        v = (int16_t)(amp * 32767.0 * std::clamp(r, -1.0, 1.0));
    }
    return s;
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
            uint16_t af, c, al, bps; uint32_t sr, br;
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

static std::string transcribe_full(cactus_model_t model, const std::vector<int16_t>& pcm) {
    char response[1 << 16] = {0};
    int rc = cactus_transcribe(model, nullptr, nullptr, response, sizeof(response),
                               R"({"telemetry_enabled": false, "auto_handoff": false, "timestamps": true})",
                               nullptr, nullptr,
                               reinterpret_cast<const uint8_t*>(pcm.data()),
                               pcm.size() * sizeof(int16_t));
    if (rc <= 0) return "";
    return json_string(std::string(response), "response");
}

static std::string run_stream(cactus_model_t model, const std::vector<int16_t>& pcm, size_t chunk_samples) {
    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, nullptr);
    if (!stream) return "";
    std::string transcript;
    std::vector<char> resp(1 << 16);
    auto append = [&]() {
        std::string c = json_string(std::string(resp.data()), "confirmed");
        if (c.empty()) return;
        if (!transcript.empty() && transcript.back() != ' ' && c.front() != ' ') transcript += ' ';
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

static bool test_stream_matches_oneshot() {
    std::cout << "\n=== STREAM vs ONE-SHOT (all assets) ===\n";
    if (!g_transcription_model_path) { std::cout << "SKIP CACTUS_TEST_TRANSCRIPTION_MODEL not set\n"; return true; }
    if (!g_assets_path) { std::cout << "SKIP CACTUS_TEST_ASSETS not set\n"; return true; }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[x] init failed\n"; return false; }

    const char* files[] = {"/test.wav", "/record.wav", "/test_long.wav", "/hotword.wav"};
    const size_t chunk_samples = 5 * 16000;
    bool all_ok = true;
    int tested = 0;
    for (const char* file : files) {
        std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + file);
        if (pcm.empty()) continue;
        if (pcm.size() > 20 * 16000) pcm.resize(20 * 16000);

        std::string golden = transcribe_full(model, pcm);
        std::string streamed = run_stream(model, pcm, chunk_samples);
        auto gw = normalized_words(golden);
        auto sw = normalized_words(streamed);
        double recall = word_recall(gw, sw);
        double precision = word_recall(sw, gw);
        bool substantive = gw.size() >= 4;
        bool ok = !substantive || (recall >= 0.9 && precision >= 0.9);
        all_ok = all_ok && ok;
        if (substantive) ++tested;
        std::cout << "  " << file << " (" << std::fixed << std::setprecision(1) << (pcm.size() / 16000.0)
                  << "s) recall=" << std::setprecision(3) << recall << " precision=" << precision
                  << "  " << (ok ? "OK" : "FAIL") << "\n"
                  << "    golden:   " << golden << "\n    streamed: " << streamed << "\n";
    }
    cactus_destroy(model);
    bool passed = all_ok && tested > 0;
    std::cout << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

static bool test_stream_chunk_sizes() {
    std::cout << "\n=== STREAM CHUNK SIZES (0.1s / 1s / 5s) ===\n";
    if (!g_transcription_model_path || !g_assets_path) { std::cout << "SKIP\n"; return true; }
    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[x] init failed\n"; return false; }

    std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + "/test.wav");
    if (pcm.empty()) { cactus_destroy(model); std::cout << "  SKIP (no test.wav)\n"; return true; }
    if (pcm.size() > 20 * 16000) pcm.resize(20 * 16000);
    auto gw = normalized_words(transcribe_full(model, pcm));

    const size_t chunks[] = {1600, 16000, 80000};
    const char* labels[] = {"0.1s", "1.0s", "5.0s"};
    bool all_ok = true;
    for (int i = 0; i < 3; ++i) {
        auto sw = normalized_words(run_stream(model, pcm, chunks[i]));
        double recall = word_recall(gw, sw);
        bool ok = recall >= 0.85;
        all_ok = all_ok && ok;
        std::cout << "  chunk " << labels[i] << ": recall=" << std::fixed << std::setprecision(3)
                  << recall << "  " << (ok ? "OK" : "FAIL") << "\n";
    }
    cactus_destroy(model);
    std::cout << "  Status: " << (all_ok ? "PASSED" : "FAILED") << "\n";
    return all_ok;
}

static bool test_stream_edge_cases() {
    std::cout << "\n=== STREAM EDGE CASES ===\n";
    if (!g_transcription_model_path) { std::cout << "SKIP\n"; return true; }
    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[x] init failed\n"; return false; }

    struct Case { const char* label; std::vector<int16_t> pcm; size_t chunk; size_t max_words; };
    std::vector<Case> cases;
    cases.push_back({"empty input", {}, 16000, 0});
    cases.push_back({"2s silence", make_silence(2), 8000, 2});
    cases.push_back({"5s silence", make_silence(5), 16000, 2});
    cases.push_back({"3s pure tone 440Hz", sine_tone(3, 440.0, 0.6), 16000, 20});
    cases.push_back({"3s white noise", white_noise(3, 0.5, 7), 16000, 30});
    cases.push_back({"tiny 0.05s chunk", make_silence(1), 800, 2});

    bool all_ok = true;
    for (auto& c : cases) {
        std::string streamed = run_stream(model, c.pcm, c.chunk);
        auto w = normalized_words(streamed);
        bool ok = w.size() <= c.max_words;
        all_ok = all_ok && ok;
        std::cout << "  " << c.label << ": words=" << w.size() << " (<=" << c.max_words << ")  "
                  << (ok ? "OK" : "FAIL") << (streamed.empty() ? "" : ("  '" + streamed + "'")) << "\n";
    }
    cactus_destroy(model);
    std::cout << "  Status: " << (all_ok ? "PASSED" : "FAILED") << "\n";
    return all_ok;
}

static std::vector<int16_t> concat(std::vector<std::vector<int16_t>> parts) {
    std::vector<int16_t> out;
    for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

static bool test_stream_silence_gap() {
    std::cout << "\n=== STREAM SILENCE GAPS (speech separated by silence) ===\n";
    if (!g_transcription_model_path || !g_assets_path) { std::cout << "SKIP\n"; return true; }
    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[x] init failed\n"; return false; }

    std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + "/test_long.wav");
    if (pcm.size() < 20 * 16000) { cactus_destroy(model); std::cout << "  SKIP (test_long.wav too short)\n"; return true; }

    std::vector<int16_t> seg_a(pcm.begin(), pcm.begin() + 10 * 16000);
    std::vector<int16_t> seg_b(pcm.begin() + 10 * 16000, pcm.begin() + 20 * 16000);
    auto golden_a = normalized_words(transcribe_full(model, seg_a));
    auto golden_b = normalized_words(transcribe_full(model, seg_b));

    struct Case { std::string label; std::vector<int16_t> pcm; bool expect_a; bool expect_b; };
    std::vector<Case> cases;
    for (int gap : {2, 5, 10, 20, 30})
        cases.push_back({"a/" + std::to_string(gap) + "s/b", concat({seg_a, make_silence(gap), seg_b}), true, true});
    cases.push_back({"10s_sil/a", concat({make_silence(10), seg_a}), true, false});
    cases.push_back({"a/10s_sil", concat({seg_a, make_silence(10)}), true, false});
    cases.push_back({"2gap a/b/a", concat({seg_a, make_silence(15), seg_b, make_silence(15), seg_a}), true, true});
    cases.push_back({"3gap varied", concat({seg_a, make_silence(5), seg_b, make_silence(12), seg_a, make_silence(20), seg_b}), true, true});
    cases.push_back({"4gap a/b...", concat({seg_a, make_silence(8), seg_b, make_silence(8), seg_a, make_silence(8), seg_b, make_silence(8), seg_a}), true, true});
    cases.push_back({"5gap a/b...", concat({seg_a, make_silence(6), seg_b, make_silence(6), seg_a, make_silence(6), seg_b, make_silence(6), seg_a, make_silence(6), seg_b}), true, true});

    bool all_ok = true;
    for (auto& c : cases) {
        std::string streamed = run_stream(model, c.pcm, kLiveChunkSamples);
        auto sw = normalized_words(streamed);
        double ra = word_recall(golden_a, sw), rb = word_recall(golden_b, sw);
        bool ok = (!c.expect_a || ra >= 0.5) && (!c.expect_b || rb >= 0.5);
        all_ok = all_ok && ok;
        std::cout << "  " << std::setw(18) << std::left << c.label
                  << " a-recall=" << std::fixed << std::setprecision(3) << ra
                  << " b-recall=" << rb << "  " << (ok ? "OK" : "FAIL") << "\n";
        if (!ok) std::cout << "      streamed: " << streamed << "\n";
    }
    cactus_destroy(model);
    std::cout << "  Status: " << (all_ok ? "PASSED" : "FAILED") << "\n";
    return all_ok;
}

static bool test_timestamps_segments() {
    std::cout << "\n=== WHISPER TIMESTAMP SEGMENTS ===\n";
    if (!g_transcription_model_path || !g_assets_path) { std::cout << "SKIP\n"; return true; }
    const bool is_whisper = std::string(g_transcription_model_path).find("whisper") != std::string::npos;
    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[x] init failed\n"; return false; }

    std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + "/test.wav");
    if (pcm.size() > 20 * 16000) pcm.resize(20 * 16000);
    std::vector<char> resp(1 << 16, 0);
    cactus_transcribe(model, nullptr, nullptr, resp.data(), resp.size(),
                      R"({"timestamps": true})", nullptr, nullptr,
                      reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size() * sizeof(int16_t));
    cactus_destroy(model);

    std::string json(resp.data());
    size_t count = 0;
    for (size_t p = json.find("\"start\":"); p != std::string::npos; p = json.find("\"start\":", p + 1)) ++count;
    bool ok = is_whisper ? count > 0 : count == 0;
    std::cout << "  segments=" << count << " (" << (is_whisper ? "Whisper: expect >0" : "Parakeet: expect 0") << ")  "
              << (ok ? "OK" : "FAIL") << "\n  Status: " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Stream Tests");
    runner.run_test("stream_matches_oneshot", test_stream_matches_oneshot());
    runner.run_test("stream_chunk_sizes", test_stream_chunk_sizes());
    runner.run_test("stream_edge_cases", test_stream_edge_cases());
    runner.run_test("stream_silence_gap", test_stream_silence_gap());
    runner.run_test("timestamps_segments", test_timestamps_segments());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
