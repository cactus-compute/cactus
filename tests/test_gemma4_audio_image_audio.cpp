#include "../cactus/ffi/cactus_ffi.h"
#include "../libs/audio/wav.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string unescape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"'; i++; break;
                case '\\': out += '\\'; i++; break;
                case 'n':  out += '\n'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case 't':  out += '\t'; i++; break;
                case 'b':  out += '\b'; i++; break;
                case 'f':  out += '\f'; i++; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::string extract_response_field(const std::string& json) {
    const std::string key = "\"response\":\"";
    size_t start = json.find(key);
    if (start == std::string::npos) return "";
    start += key.size();
    size_t end = start;
    while (end < json.size()) {
        end = json.find('"', end);
        if (end == std::string::npos) return "";
        size_t back = 0;
        for (size_t i = end; i > start && json[i - 1] == '\\'; i--) back++;
        if (back % 2 == 0) break;
        end++;
    }
    return unescape_json(json.substr(start, end - start));
}

static std::string ensure_wav_from_mp3(const std::string& mp3_path) {
    fs::path src(mp3_path);
    fs::path out = fs::temp_directory_path() / (src.stem().string() + "_cactus_test_16k.wav");
    if (fs::exists(out)) return out.string();
    std::string cmd = "afconvert -f WAVE -d LEI16@16000 -c 1 "
                      "'" + mp3_path + "' '" + out.string() + "' >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) return "";
    return fs::exists(out) ? out.string() : "";
}

static std::vector<uint8_t> wav_to_pcm16(const std::string& wav_path) {
    AudioFP32 wav = load_wav(wav_path);
    std::vector<int16_t> pcm(wav.samples.size());
    for (size_t i = 0; i < wav.samples.size(); i++) {
        float s = wav.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }
    std::vector<uint8_t> bytes(pcm.size() * sizeof(int16_t));
    std::memcpy(bytes.data(), pcm.data(), bytes.size());
    return bytes;
}

struct TurnResult {
    bool ok = false;
    std::string response;
};

static TurnResult run_turn(cactus_model_t model,
                           const std::string& messages_json,
                           const std::vector<uint8_t>& pcm) {
    const char* options = R"({"max_tokens":128,"temperature":0.0,"top_p":1.0,"top_k":1,"auto_handoff":false,"telemetry_enabled":false})";
    std::vector<char> buf(64 * 1024, 0);
    int rc = cactus_complete(
        model, messages_json.c_str(), buf.data(), buf.size(),
        options, nullptr, nullptr, nullptr,
        pcm.empty() ? nullptr : pcm.data(), pcm.size()
    );
    TurnResult r;
    if (rc < 0) {
        std::cerr << "  cactus_complete failed: " << buf.data() << "\n";
        return r;
    }
    r.response = extract_response_field(std::string(buf.data()));
    r.ok = !r.response.empty();
    return r;
}

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (!model_path) {
        std::cerr << "SKIP: CACTUS_TEST_GEMMA4_MODEL not set\n";
        return 0;
    }
    const char* repo_root_env = std::getenv("CACTUS_TEST_REPO_ROOT");
    fs::path repo_root = repo_root_env ? fs::path(repo_root_env) : fs::current_path() / "..";

    fs::path who_mp3 = repo_root / "who_are_you.mp3";
    fs::path math_mp3 = repo_root / "2+2.mp3";
    fs::path banner = repo_root / "assets" / "banner.jpg";

    for (const auto& p : {who_mp3, math_mp3, banner}) {
        if (!fs::exists(p)) {
            std::cerr << "SKIP: missing asset " << p << "\n";
            return 0;
        }
    }

    std::string who_wav = ensure_wav_from_mp3(who_mp3.string());
    std::string math_wav = ensure_wav_from_mp3(math_mp3.string());
    if (who_wav.empty() || math_wav.empty()) {
        std::cerr << "SKIP: afconvert failed (needed to transcode mp3 -> wav)\n";
        return 0;
    }

    auto who_pcm = wav_to_pcm16(who_wav);
    auto math_pcm = wav_to_pcm16(math_wav);

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "FAIL: cactus_init\n";
        return 1;
    }

    std::string m1 = R"([{"role":"user","content":""}])";
    std::cout << "Turn 1 (audio who_are_you.mp3):\n";
    auto t1 = run_turn(model, m1, who_pcm);
    if (!t1.ok) { cactus_destroy(model); return 1; }
    std::cout << "  -> " << t1.response << "\n";

    std::string m2 =
        std::string(R"([{"role":"user","content":""},)") +
        R"({"role":"assistant","content":")" + escape_json(t1.response) + R"("},)" +
        R"({"role":"user","content":"describe this","images":[")" + escape_json(banner.string()) + R"("]}])";
    std::cout << "Turn 2 (image + 'describe this'):\n";
    auto t2 = run_turn(model, m2, {});
    if (!t2.ok) { cactus_destroy(model); return 1; }
    std::cout << "  -> " << t2.response << "\n";

    std::string m3 =
        std::string(R"([{"role":"user","content":""},)") +
        R"({"role":"assistant","content":")" + escape_json(t1.response) + R"("},)" +
        R"({"role":"user","content":"describe this","images":[")" + escape_json(banner.string()) + R"("]},)" +
        R"({"role":"assistant","content":")" + escape_json(t2.response) + R"("},)" +
        R"({"role":"user","content":"","images":[")" + escape_json(banner.string()) + R"("]}])";
    std::cout << "Turn 3 (audio 2+2.mp3, image still attached):\n";
    auto t3 = run_turn(model, m3, math_pcm);
    if (!t3.ok) { cactus_destroy(model); return 1; }
    std::cout << "  -> " << t3.response << "\n";

    cactus_destroy(model);

    if (t3.response == t2.response) {
        std::cerr << "FAIL: turn 3 response identical to turn 2 — new audio is being ignored\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
