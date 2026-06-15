#include "test_utils.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>

using namespace EngineTestUtils;

static const char* g_transcription_model_path = std::getenv("CACTUS_TEST_TRANSCRIPTION_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static std::vector<int16_t> load_wav(const std::string& path);

static std::string normalize_word(const std::string& w) {
    std::string out;
    for (char c : w) if (std::isalnum((unsigned char)c)) out += (char)std::tolower((unsigned char)c);
    return out;
}

static std::vector<std::string> normalized_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) {
        std::string n = normalize_word(w);
        if (!n.empty()) words.push_back(n);
    }
    return words;
}

static double word_recall(const std::vector<std::string>& ref, const std::vector<std::string>& hyp) {
    if (ref.empty()) return 1.0;
    std::vector<std::string> ref_set = ref;
    std::sort(ref_set.begin(), ref_set.end());
    ref_set.erase(std::unique(ref_set.begin(), ref_set.end()), ref_set.end());
    size_t found = 0;
    for (const auto& w : ref_set) {
        if (std::find(hyp.begin(), hyp.end(), w) != hyp.end()) ++found;
    }
    return (double)found / (double)ref_set.size();
}

static std::string run_stream(cactus_model_t model, const std::vector<int16_t>& pcm,
                              size_t chunk_samples, const char* options) {
    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, options);
    if (!stream) return "";

    std::string transcript;
    auto append = [&](const std::string& confirmed) {
        if (confirmed.empty()) return;
        if (!transcript.empty()) transcript += ' ';
        transcript += confirmed;
    };

    std::vector<char> resp(1 << 16);
    for (size_t off = 0; off < pcm.size(); off += chunk_samples) {
        size_t n = std::min(chunk_samples, pcm.size() - off);
        resp[0] = '\0';
        int rc = cactus_stream_transcribe_process(
            stream, reinterpret_cast<const uint8_t*>(pcm.data() + off), n * sizeof(int16_t),
            resp.data(), resp.size());
        if (rc < 0) {
            cactus_stream_transcribe_stop(stream, nullptr, 0);
            return transcript;
        }
        append(json_string(std::string(resp.data()), "confirmed"));
    }

    resp[0] = '\0';
    cactus_stream_transcribe_stop(stream, resp.data(), resp.size());
    append(json_string(std::string(resp.data()), "confirmed"));
    return transcript;
}

static std::string transcribe_pcm_full(cactus_model_t model, const std::vector<int16_t>& pcm) {
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
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        STREAM BASIC TEST                  ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    const char* files[] = {"/test.wav", "/record.wav", "/test_long.wav"};
    bool all_ok = true;
    int tested = 0;
    for (const char* file : files) {
        std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + file);
        if (pcm.empty()) continue;
        if (pcm.size() > 30 * 16000) pcm.resize(30 * 16000);

        std::string reference = transcribe_pcm_full(model, pcm);
        Timer timer;
        std::string streamed = run_stream(model, pcm, 16000, R"({"min_chunk_sec": 4.0})");
        double elapsed = timer.elapsed_ms();

        auto ref_words = normalized_words(reference);
        auto stream_words = normalized_words(streamed);
        double recall = word_recall(ref_words, stream_words);
        bool substantive = ref_words.size() >= 8;
        bool ok = !substantive || recall >= 0.6;
        all_ok = all_ok && ok;
        if (substantive) ++tested;

        std::cout << "├─ " << file << " (" << std::fixed << std::setprecision(1)
                  << (pcm.size() / 16000.0) << "s)\n"
                  << "│   reference: " << reference << "\n"
                  << "│   streamed:  " << streamed << "\n"
                  << "│   words ref=" << ref_words.size() << " streamed=" << stream_words.size()
                  << " recall=" << std::setprecision(2) << recall
                  << " time=" << std::setprecision(0) << elapsed << "ms "
                  << (substantive ? (ok ? "✓" : "✗") : "— (no baseline)") << "\n";
    }

    cactus_destroy(model);
    bool passed = all_ok && tested > 0;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

static bool test_stream_speech_silence_speech() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║   STREAM SPEECH / SILENCE / SPEECH TEST   ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::vector<int16_t> full = load_wav(std::string(g_assets_path) + "/test.wav");
    if (full.empty()) { std::cerr << "[✗] Failed to load test.wav\n"; cactus_destroy(model); return false; }

    size_t clip_len = std::min<size_t>(full.size(), 8 * 16000);
    std::vector<int16_t> clip(full.begin(), full.begin() + clip_len);
    std::vector<int16_t> combined = clip;
    combined.insert(combined.end(), 5 * 16000, 0);
    combined.insert(combined.end(), clip.begin(), clip.end());

    std::string reference = transcribe_pcm_full(model, clip);
    auto ref_words = normalized_words(reference);

    std::string key;
    for (const auto& w : ref_words) if (w.size() >= 4 && w.size() > key.size()) key = w;
    if (key.empty()) for (const auto& w : ref_words) if (w.size() > key.size()) key = w;

    Timer timer;
    std::string streamed = run_stream(model, combined, 8000,
                                      R"({"min_chunk_sec": 3.0, "silence_sec": 0.7})");
    double elapsed = timer.elapsed_ms();

    auto stream_words = normalized_words(streamed);
    size_t key_count = key.empty() ? 0 : (size_t)std::count(stream_words.begin(), stream_words.end(), key);

    std::cout << "├─ Reference (one clip): " << reference << "\n"
              << "├─ Streamed (clip|silence|clip): " << streamed << "\n"
              << "├─ Distinctive word: '" << key << "' x" << key_count << " (expect >= 2)\n"
              << "├─ Words: one-clip=" << ref_words.size() << " streamed=" << stream_words.size() << "\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n";

    cactus_destroy(model);
    bool passed = !stream_words.empty() && key_count >= 2 &&
                  stream_words.size() >= (size_t)(1.5 * ref_words.size());
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

static bool test_stream_silence_only() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        STREAM SILENCE-ONLY TEST           ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::vector<int16_t> silence(5 * 16000, 0);
    std::string streamed = run_stream(model, silence, 8000, nullptr);
    auto words = normalized_words(streamed);

    std::cout << "├─ Streamed: '" << streamed << "'\n"
              << "├─ Words: " << words.size() << " (expect 0)\n";

    cactus_destroy(model);
    bool passed = words.empty();
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

static std::vector<int16_t> make_silence(int seconds) {
    return std::vector<int16_t>((size_t)seconds * 16000, 0);
}

static std::vector<int16_t> concat(std::initializer_list<const std::vector<int16_t>*> parts) {
    std::vector<int16_t> out;
    for (const auto* p : parts) out.insert(out.end(), p->begin(), p->end());
    return out;
}

static bool test_stream_long_silence() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        STREAM LONG-SILENCE TEST           ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::vector<int16_t> full = load_wav(std::string(g_assets_path) + "/test.wav");
    if (full.empty()) { std::cerr << "[✗] Failed to load test.wav\n"; cactus_destroy(model); return false; }

    std::vector<int16_t> clip(full.begin(), full.begin() + std::min<size_t>(full.size(), 8 * 16000));
    std::vector<int16_t> sil20 = make_silence(20);
    std::vector<int16_t> sil30 = make_silence(30);

    std::string reference = transcribe_pcm_full(model, clip);
    auto ref_words = normalized_words(reference);
    std::string key;
    for (const auto& w : ref_words) if (w.size() >= 4 && w.size() > key.size()) key = w;
    if (key.empty()) { std::cerr << "[✗] No reference key word\n"; cactus_destroy(model); return false; }

    struct Case { const char* label; std::vector<int16_t> pcm; size_t chunk; int expect; };
    std::vector<Case> cases;
    cases.push_back({"20s silence -> speech",                       concat({&sil20, &clip}),                  8000, 1});
    cases.push_back({"speech -> 20s silence -> speech",             concat({&clip, &sil20, &clip}),           8000, 2});
    cases.push_back({"speech -> 20s -> speech -> 20s -> speech",    concat({&clip, &sil20, &clip, &sil20, &clip}), 8000, 3});
    cases.push_back({"speech -> 20s trailing silence",              concat({&clip, &sil20}),                  8000, 1});
    cases.push_back({"30s silence only",                            sil30,                                    8000, 0});
    cases.push_back({"speech -> 20s -> speech (0.15s live chunks)", concat({&clip, &sil20, &clip}),           2400, 2});

    bool all_ok = true;
    for (auto& c : cases) {
        std::string streamed = run_stream(model, c.pcm, c.chunk, R"({"min_chunk_sec": 2.0, "silence_sec": 0.7})");
        auto words = normalized_words(streamed);
        size_t kc = (size_t)std::count(words.begin(), words.end(), key);
        bool ok = c.expect == 0
            ? words.empty()
            : ((int)kc >= c.expect && words.size() <= (size_t)c.expect * (ref_words.size() + 4));
        all_ok = all_ok && ok;
        std::cout << "├─ " << c.label << "\n"
                  << "│   '" << key << "' x" << kc << " (expect " << c.expect << "), words=" << words.size()
                  << "  " << (ok ? "✓" : "✗") << "\n"
                  << "│   streamed: " << streamed << "\n";
    }

    cactus_destroy(model);
    std::cout << "└─ Status: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return all_ok;
}

static std::vector<int16_t> sine_tone(int seconds, double freq, double amp) {
    const double pi = 3.14159265358979323846;
    std::vector<int16_t> s((size_t)seconds * 16000);
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (int16_t)(amp * 32767.0 * std::sin(2.0 * pi * freq * (double)i / 16000.0));
    return s;
}

static std::vector<int16_t> white_noise(int seconds, double amp, uint32_t seed) {
    std::vector<int16_t> s((size_t)seconds * 16000);
    uint32_t state = seed;
    for (auto& v : s) {
        state = state * 1664525u + 1013904223u;
        double r = ((double)(state >> 9) / 4194304.0) - 1.0;
        v = (int16_t)(amp * 32767.0 * std::clamp(r, -1.0, 1.0));
    }
    return s;
}

static std::vector<int16_t> scale_clip(const std::vector<int16_t>& in, double gain) {
    std::vector<int16_t> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = (int16_t)std::clamp((double)in[i] * gain, -32768.0, 32767.0);
    return out;
}

static std::vector<int16_t> with_micro_pauses(const std::vector<int16_t>& clip) {
    std::vector<int16_t> out;
    std::vector<int16_t> gap(4800, 0); // 0.3s < silence_sec
    size_t piece = 2 * 16000;
    for (size_t off = 0; off < clip.size(); off += piece) {
        size_t n = std::min(piece, clip.size() - off);
        out.insert(out.end(), clip.begin() + off, clip.begin() + off + n);
        if (off + n < clip.size()) out.insert(out.end(), gap.begin(), gap.end());
    }
    return out;
}

static bool test_stream_edge_cases() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          STREAM EDGE CASES TEST           ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::vector<int16_t> full = load_wav(std::string(g_assets_path) + "/test.wav");
    if (full.empty()) { std::cerr << "[✗] Failed to load test.wav\n"; cactus_destroy(model); return false; }
    std::vector<int16_t> clip(full.begin(), full.begin() + std::min<size_t>(full.size(), 8 * 16000));

    std::string reference = transcribe_pcm_full(model, clip);
    auto ref_words = normalized_words(reference);
    int R = (int)ref_words.size();
    std::string key;
    for (const auto& w : ref_words) if (w.size() >= 4 && w.size() > key.size()) key = w;
    if (R < 4 || key.empty()) { std::cerr << "[✗] Weak reference\n"; cactus_destroy(model); return false; }

    std::vector<int16_t> sil20 = make_silence(20);

    std::vector<int16_t> continuous = concat({&clip, &clip, &clip, &clip});
    std::vector<int16_t> sil1 = make_silence(1);
    std::vector<int16_t> bursts = concat({&clip, &sil1, &clip, &sil1, &clip});
    std::vector<int16_t> micro = with_micro_pauses(clip);
    std::vector<int16_t> gapped = concat({&clip, &sil20, &clip});

    struct Case {
        const char* label;
        std::vector<int16_t> pcm;
        size_t chunk;
        int min_key, min_words, max_words;
    };
    std::vector<Case> cases;
    cases.push_back({"1. empty input",                 {},                          8000, 0, 0, 0});
    cases.push_back({"2. 0.2s silence only",           make_silence(1),             8000, 0, 0, 0});
    cases.push_back({"3. 8s pure tone 440Hz",          sine_tone(8, 440.0, 0.6),    8000, 0, 0, 40});
    cases.push_back({"4. 8s white noise",              white_noise(8, 0.5, 9173),   8000, 0, 0, 40});
    cases.push_back({"5. very quiet speech x0.05",     scale_clip(clip, 0.05),      8000, 0, 0, R + 10});
    cases.push_back({"6. loud clipped speech x4",      scale_clip(clip, 4.0),       8000, 0, 0, 3 * R});
    cases.push_back({"7. 32s continuous (>cap)",       continuous,                  8000, 1, R, 8 * R});
    cases.push_back({"8. speech|1s|speech|1s|speech",  bursts,                      8000, 2, 0, 5 * R});
    cases.push_back({"9. micro-pauses (0.3s gaps)",    micro,                       8000, 1, 0, 3 * R});
    cases.push_back({"10. 20s gap, 0.1s live chunks",  gapped,                      1600, 2, 0, 4 * R});

    bool all_ok = true;
    for (auto& c : cases) {
        std::string streamed = run_stream(model, c.pcm, c.chunk, R"({"min_chunk_sec": 2.0, "silence_sec": 0.7})");
        auto words = normalized_words(streamed);
        int kc = (int)std::count(words.begin(), words.end(), key);
        bool ok = kc >= c.min_key && (int)words.size() >= c.min_words && (int)words.size() <= c.max_words;
        all_ok = all_ok && ok;
        std::cout << "├─ " << c.label << "\n"
                  << "│   words=" << words.size() << " '" << key << "'x" << kc
                  << "  (need key>=" << c.min_key << ", words " << c.min_words << ".." << c.max_words << ")  "
                  << (ok ? "✓" : "✗") << "\n"
                  << "│   streamed: " << streamed << "\n";
    }

    cactus_destroy(model);
    std::cout << "└─ Status: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return all_ok;
}

static std::vector<int16_t> load_wav(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    auto rd32 = [&](uint32_t& v) { return fread(&v, 4, 1, f) == 1; };
    auto rd16 = [&](uint16_t& v) { return fread(&v, 2, 1, f) == 1; };
    char tag[4];
    uint32_t rsz;
    if (fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "RIFF", 4)) { fclose(f); return {}; }
    rd32(rsz);
    if (fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "WAVE", 4)) { fclose(f); return {}; }
    uint16_t fmt = 0, ch = 1, bits = 16;
    uint32_t rate = 16000;
    bool have_fmt = false;
    std::vector<int16_t> mono;
    while (fread(tag, 1, 4, f) == 4) {
        uint32_t sz;
        if (!rd32(sz)) break;
        if (!std::strncmp(tag, "fmt ", 4)) {
            uint16_t af, c, al, bps;
            uint32_t sr, br;
            rd16(af); rd16(c); rd32(sr); rd32(br); rd16(al); rd16(bps);
            fmt = af; ch = c ? c : 1; rate = sr; bits = bps; have_fmt = true;
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
        double s = (double)i / ratio;
        size_t i0 = (size_t)s;
        double fr = s - (double)i0;
        int16_t a = mono[std::min(i0, mono.size() - 1)];
        int16_t b = mono[std::min(i0 + 1, mono.size() - 1)];
        out[i] = (int16_t)((double)a + ((double)b - (double)a) * fr);
    }
    return out;
}

static bool contains_word(const std::vector<std::string>& words, const std::string& w) {
    return std::find(words.begin(), words.end(), normalize_word(w)) != words.end();
}

static bool test_stream_say() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║      STREAM SAY (KNOWN CONTENT) TEST      ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }
    std::string dir = std::string(g_assets_path) + "/say/";
    std::vector<int16_t> a = load_wav(dir + "say_a.wav");
    std::vector<int16_t> b = load_wav(dir + "say_b.wav");
    std::vector<int16_t> c = load_wav(dir + "say_c.wav");
    std::vector<int16_t> lng = load_wav(dir + "say_long.wav");
    if (a.empty() || b.empty() || c.empty() || lng.empty()) {
        std::cout << "⊘ SKIP │ say/*.wav not generated (run the say command)\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    std::vector<int16_t> sil10 = make_silence(10);
    std::vector<int16_t> sil20 = make_silence(20);

    bool all_ok = true;
    auto check = [&](const std::string& label, const std::vector<int16_t>& pcm, size_t chunk,
                     std::vector<std::string> must_have) {
        std::string s = run_stream(model, pcm, chunk, R"({"min_chunk_sec": 1.5, "silence_sec": 0.7})");
        auto w = normalized_words(s);
        std::string missing;
        for (auto& m : must_have) if (!contains_word(w, m)) missing += m + " ";
        bool ok = missing.empty();
        all_ok = all_ok && ok;
        std::cout << "├─ " << label << "  " << (ok ? "✓" : "✗")
                  << (missing.empty() ? "" : "  MISSING: " + missing) << "\n"
                  << "│   streamed: " << s << "\n";
    };

    check("say_a plain (1s chunks)", a, 16000, {"fox", "lazy", "dog"});
    check("say_a | 20s silence | say_b (0.5s chunks)", concat({&a, &sil20, &b}), 8000, {"fox", "dog", "box"});
    check("10s sil + say_c + 10s sil (0.1s live chunks)", concat({&sil10, &c, &sil10}), 1600, {"speech", "device"});
    check("say_long continuous (1s chunks)", lng, 16000, {"transcription", "sentence"});
    check("say_a + say_b back-to-back (1s chunks)", concat({&a, &b}), 16000, {"fox", "box"});

    cactus_destroy(model);
    std::cout << "└─ Status: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return all_ok;
}

static bool test_stream_chunking() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        STREAM CHUNKING TEST               ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_transcription_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIPTION_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcription_model_path, nullptr, false);
    if (!model) { std::cerr << "[✗] Failed to initialize model\n"; return false; }

    const char* files[] = {"/test.wav", "/test_long.wav", "/record.wav", "/hotword.wav"};
    const size_t chunks[] = {1600, 16000, 48000};
    const char* chunk_labels[] = {"0.1s", "1.0s", "3.0s"};

    bool all_ok = true;
    for (const char* file : files) {
        std::vector<int16_t> pcm = load_wav(std::string(g_assets_path) + file);
        if (pcm.empty()) continue;
        if (pcm.size() > 18 * 16000) pcm.resize(18 * 16000);

        auto base_words = normalized_words(transcribe_pcm_full(model, pcm));
        std::cout << "├─ " << file << " (" << std::fixed << std::setprecision(1)
                  << (pcm.size() / 16000.0) << "s, baseline " << base_words.size() << " words)\n";
        if (base_words.size() < 8) {
            std::cout << "│   (no usable speech baseline — exercised only)\n";
            for (size_t ci = 0; ci < 3; ++ci) run_stream(model, pcm, chunks[ci], R"({"min_chunk_sec": 3.0})");
            continue;
        }
        for (size_t ci = 0; ci < 3; ++ci) {
            std::string streamed = run_stream(model, pcm, chunks[ci], R"({"min_chunk_sec": 3.0})");
            double recall = word_recall(base_words, normalized_words(streamed));
            bool ok = recall >= 0.55;
            all_ok = all_ok && ok;
            std::cout << "│   chunk " << chunk_labels[ci] << ": recall=" << std::setprecision(2)
                      << recall << "  " << (ok ? "✓" : "✗") << "\n";
        }
    }

    cactus_destroy(model);
    std::cout << "└─ Status: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return all_ok;
}

int main() {
    TestUtils::TestRunner runner("Stream Tests");
    runner.run_test("stream_basic", test_stream_basic());
    runner.run_test("stream_speech_silence_speech", test_stream_speech_silence_speech());
    runner.run_test("stream_silence_only", test_stream_silence_only());
    runner.run_test("stream_long_silence", test_stream_long_silence());
    runner.run_test("stream_edge_cases", test_stream_edge_cases());
    runner.run_test("stream_say", test_stream_say());
    runner.run_test("stream_chunking", test_stream_chunking());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
