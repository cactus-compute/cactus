#include "test_utils.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <cmath>

using namespace EngineTestUtils;

static const char* g_model_path      = std::getenv("CACTUS_TEST_MODEL");
static const char* g_transcribe_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
static const char* g_assets_path     = std::getenv("CACTUS_TEST_ASSETS");
static const char* g_golden_file     = std::getenv("CACTUS_TEST_GOLDEN_FILE");
static const char* g_golden_family   = std::getenv("CACTUS_TEST_GOLDEN_FAMILY");
static const char* g_golden_prec     = std::getenv("CACTUS_TEST_GOLDEN_PRECISION");
static bool g_generate = std::getenv("CACTUS_GOLDEN_GENERATE") &&
                          std::string(std::getenv("CACTUS_GOLDEN_GENERATE")) == "1";

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::string find_golden_entry(const std::string& s, const std::string& family, const std::string& precision) {
    size_t i = 0;
    for (; i < s.size() && s[i] != '{'; i++);
    while (i < s.size()) {
        size_t start = i;
        int depth = 1;
        i++;
        for (;i < s.size() && depth > 0; i++) {
            if (s[i] == '"') {
                i++;
                for (;i < s.size() && s[i] != '"'; i++) {
                    if (s[i] == '\\') i++;
                }
            }
            else if (s[i] == '{') depth++;
            else if (s[i] == '}') depth--;
        }
        std::string entry = s.substr(start, i - start);
        if (json_string(entry, "model_family") == family && json_string(entry, "precision") == precision)
            return entry;
        for (; i < s.size() && s[i] != '{'; i++);
    }
    return "";
}

static std::string tolower(const std::string& s) {
    std::string o = s;
    std::transform(o.begin(), o.end(), o.begin(), [](unsigned char c) { return std::tolower(c); });
    return o;
}

static std::string stream_text(const StreamingData& d) {
    std::string t;
    for (const auto& s : d.tokens) t += s;
    return t;
}

static float compute_wer(const std::string& hyp, const std::string& ref) {
    auto split = [](const std::string& s) {
        std::vector<std::string> w;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) {
            std::string c;
            for (unsigned char ch : tok) if (std::isalnum(ch)) c += std::tolower(ch);
            if (!c.empty()) w.push_back(c);
        }
        return w;
    };
    auto h = split(hyp), r = split(ref);
    if (r.empty()) return h.empty() ? 0.0f : 1.0f;
    size_t n = r.size(), m = h.size();
    std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(m + 1));
    for (size_t i = 0; i <= n; i++) dp[i][0] = i;
    for (size_t j = 0; j <= m; j++) dp[0][j] = j;
    for (size_t i = 1; i <= n; i++)
        for (size_t j = 1; j <= m; j++)
            dp[i][j] = r[i-1] == h[j-1] ? dp[i-1][j-1]
                        : 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
    return float(dp[n][m]) / float(n);
}

static float cosine_sim(const float* a, const float* b, size_t n) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0f : dot / (std::sqrt(na) * std::sqrt(nb));
}

static std::vector<float> parse_float_array(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\":";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return {};
    size_t s = pos + pat.size();
    while (s < json.size() && json[s] == ' ') s++;
    if (s >= json.size() || json[s] != '[') return {};
    int depth = 1; size_t e = s + 1;
    while (e < json.size() && depth > 0) { if (json[e] == '[') depth++; else if (json[e] == ']') depth--; e++; }
    std::vector<float> r;
    std::istringstream iss(json.substr(s + 1, e - s - 2));
    std::string tok;
    while (std::getline(iss, tok, ',')) { try { r.push_back(std::stof(tok)); } catch (...) {} }
    return r;
}

static std::string resolve_asset_paths(const std::string& messages) {
    if (!g_assets_path) return messages;
    std::string prefix = std::string(g_assets_path) + "/";
    std::string result = messages;
    for (const char* ext : {".png", ".jpg", ".jpeg", ".wav", ".mp3"}) {
        size_t pos = 0;
        while ((pos = result.find(ext, pos)) != std::string::npos) {
            size_t name_start = pos;
            while (name_start > 0 && result[name_start - 1] != '"') name_start--;
            std::string name = result.substr(name_start, pos + strlen(ext) - name_start);
            if (name.find('/') == std::string::npos) {
                result.replace(name_start, name.size(), prefix + name);
                pos = name_start + prefix.size() + name.size();
            } else {
                pos += strlen(ext);
            }
        }
    }
    return result;
}

static bool test_completion_golden(const std::string& golden) {
    std::string messages = resolve_asset_paths(json_string(golden, "input_messages"));
    std::string options  = json_string(golden, "options");
    std::string expected = tolower(json_string(golden, "expected_output"));
    int min_tok = (int)json_number(golden, "token_count_min", 1);
    int max_tok = (int)json_number(golden, "token_count_max", 256);

    if (!g_model_path) { std::cerr << "CACTUS_TEST_MODEL not set\n"; return false; }
    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cerr << "Failed to init model\n"; return false; }

    StreamingData data; data.model = model;
    char resp[8192] = {};
    std::cout << "Response: ";
    int rc = cactus_complete(model, messages.c_str(), resp, sizeof(resp),
                              options.c_str(), nullptr, stream_callback, &data);
    std::cout << "\n";
    cactus_destroy(model);
    if (rc <= 0) return false;

    std::string gen = tolower(stream_text(data));

    if (g_generate) {
        std::cout << "  [GOLDEN] output: " << stream_text(data) << "\n"
                  << "  [GOLDEN] tokens: " << data.token_count << "\n";
        return true;
    }

    bool match = gen.find(expected) != std::string::npos;
    bool range = data.token_count >= min_tok && data.token_count <= max_tok;
    std::cout << "  expected: \"" << expected << "\" match=" << (match?"Y":"N")
              << " tokens=" << data.token_count << " [" << min_tok << "," << max_tok << "]\n";
    return match && range;
}

static bool test_embedding_golden(const std::string& golden) {
    std::string text = json_string(golden, "input_text");
    int exp_dim = (int)json_number(golden, "expected_embedding_dim", 0);
    float threshold = (float)json_number(golden, "cosine_similarity_threshold", 0.99);
    auto ref = parse_float_array(golden, "reference_embedding");

    if (!g_model_path) { std::cerr << "CACTUS_TEST_MODEL not set\n"; return false; }
    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cerr << "Failed to init model\n"; return false; }

    std::vector<float> emb(4096); size_t dim = 0;
    int rc = cactus_embed(model, text.c_str(), emb.data(), emb.size() * sizeof(float), &dim, true);
    cactus_destroy(model);
    if (rc <= 0 || dim == 0) return false;

    if (g_generate) {
        std::cout << "  [GOLDEN] dim: " << dim << "\n";
        return true;
    }

    bool dim_ok = exp_dim == 0 || (int)dim == exp_dim;
    float sim = (!ref.empty() && ref.size() == dim) ? cosine_sim(emb.data(), ref.data(), dim) : 0.0f;
    bool sim_ok = ref.empty() || (ref.size() == dim && sim >= threshold);
    std::cout << "  dim=" << dim << " sim=" << std::fixed << std::setprecision(4) << sim
              << " threshold=" << threshold << "\n";
    return dim_ok && sim_ok;
}

static bool test_stt_golden(const std::string& golden, const std::string& family) {
    std::string audio = json_string(golden, "audio_file");
    std::string reference = json_string(golden, "reference_transcript");
    float threshold = (float)json_number(golden, "wer_threshold", 0.15);

    if (!g_transcribe_path) { std::cerr << "CACTUS_TEST_TRANSCRIBE_MODEL not set\n"; return false; }
    if (!g_assets_path) { std::cerr << "CACTUS_TEST_ASSETS not set\n"; return false; }
    cactus_model_t model = cactus_init(g_transcribe_path, nullptr, false);
    if (!model) { std::cerr << "Failed to init model\n"; return false; }

    std::string path = std::string(g_assets_path) + "/" + audio;
    const char* prompt = family == "whisper" ? "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" : "";
    StreamingData data; data.model = model;
    char resp[1 << 15] = {};
    std::cout << "Transcript: ";
    int rc = cactus_transcribe(model, path.c_str(), prompt, resp, sizeof(resp),
                                R"({"max_tokens":100,"telemetry_enabled":false})",
                                stream_callback, &data, nullptr, 0);
    std::cout << "\n";
    cactus_destroy(model);
    if (rc <= 0) return false;

    std::string transcript = stream_text(data);
    if (g_generate) {
        std::cout << "  [GOLDEN] transcript: " << transcript << "\n";
        return true;
    }

    float wer = compute_wer(transcript, reference);
    std::cout << "  ref=\"" << reference << "\" wer=" << std::fixed << std::setprecision(4) << wer
              << " threshold=" << threshold << "\n";
    return wer <= threshold;
}

int main() {
    if (!g_golden_file || !g_golden_family || !g_golden_prec) {
        std::cout << "Skipping exhaustive tests (CACTUS_TEST_GOLDEN_FILE, CACTUS_TEST_GOLDEN_FAMILY, CACTUS_TEST_GOLDEN_PRECISION not set)\n";
        return 0;
    }

    std::string all = read_file(g_golden_file);
    if (all.empty()) { std::cerr << "Cannot read: " << g_golden_file << "\n"; return 1; }
    std::string json = find_golden_entry(all, g_golden_family, g_golden_prec);
    if (json.empty()) { std::cerr << "No golden entry for " << g_golden_family << "/" << g_golden_prec << "\n"; return 1; }

    std::string type = json_string(json, "test_type");
    std::string family(g_golden_family);
    std::string label = type + "_" + family + "_" + g_golden_prec;

    TestUtils::TestRunner runner("Exhaustive Tests");

    if (type == "llm" || type == "vlm")  runner.run_test(label, test_completion_golden(json));
    else if (type == "embedding")        runner.run_test(label, test_embedding_golden(json));
    else if (type == "stt")              runner.run_test(label, test_stt_golden(json, family));
    else { std::cerr << "Unknown test_type: " << type << "\n"; return 1; }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
