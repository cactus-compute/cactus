#include "test_utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>


namespace TestUtils {

void apply_backend() {
    static bool applied = false;
    if (applied) return;
    applied = true;
    const char* b = std::getenv("CACTUS_TEST_BACKEND");
    if (!b || !*b) return;
    if (cactus_set_backend(b) == 0)
        std::cout << "Backend: " << b << "\n";
    else
        std::cout << "Backend '" << b << "' unavailable; using default\n";
}

TestRunner::TestRunner(const std::string& suite_name)
    : suite_name_(suite_name), passed_count_(0), total_count_(0) {
    apply_backend();
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║ Running " << std::left << std::setw(76) << suite_name_ << " ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";
}

void TestRunner::run_test(const std::string& test_name, bool result) {
    total_count_++;
    if (result) {
        passed_count_++;
        std::cout << "✓ PASS │ " << std::left << std::setw(25) << test_name << "\n";
    } else {
        std::cout << "✗ FAIL │ " << std::left << std::setw(25) << test_name << "\n";
    }
}

void TestRunner::log_skip(const std::string& test_name, const std::string& reason) {
    std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << test_name << " │ " << reason << "\n";
}

void TestRunner::print_summary() {
    std::cout << "────────────────────────────────────────────────────────────────────────────────────────\n";
    if (all_passed())
        std::cout << "✓ All " << total_count_ << " tests passed!\n";
    else
        std::cout << "✗ " << (total_count_ - passed_count_) << " of " << total_count_ << " tests failed!\n";
    std::cout << "\n";
}

bool TestRunner::all_passed() const {
    return passed_count_ == total_count_;
}

}

namespace EngineTestUtils {

Timer::Timer() : start(std::chrono::high_resolution_clock::now()) {}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

double json_number(const std::string& json, const std::string& key, double def) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return def;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;
    try { return std::stod(json.substr(start, end - start)); }
    catch (...) { return def; }
}

std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '"') return {};
    ++start;

    std::string out;
    out.reserve(128);
    bool escaped = false;
    for (size_t i = start; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(c); break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') { escaped = true; continue; }
        if (c == '"') return out;
        out.push_back(c);
    }
    return {};
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (auto c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            default:   o << c;      break;
        }
    }
    return o.str();
}

void stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token ? token : "");
    data->token_ids.push_back(token_id);
    data->token_count++;

    std::string out = token ? token : "";
    for (char& c : out) if (c == '\n') c = ' ';
    std::cout << out << std::flush;

    if (data->stop_at > 0 && data->token_count >= data->stop_at) {
        std::cout << " [-> stopped]" << std::flush;
        cactus_stop(data->model);
    }
}

static bool json_bool(const std::string& json, const std::string& key, bool def = false) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return def;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start + 4 <= json.size() && json.substr(start, 4) == "true") return true;
    if (start + 5 <= json.size() && json.substr(start, 5) == "false") return false;
    return def;
}

static std::string json_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "[]";
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '[') return "[]";
    int depth = 1;
    size_t end = start + 1;
    while (end < json.size() && depth > 0) {
        if (json[end] == '[') depth++;
        else if (json[end] == ']') depth--;
        end++;
    }
    return json.substr(start, end - start);
}

void Metrics::parse(const std::string& json) {
    success = json_bool(json, "success", false);
    error = json_string(json, "error");
    cloud_handoff = json_bool(json, "cloud_handoff", false);
    response = json_string(json, "response");
    thinking = json_string(json, "thinking");
    function_calls = json_array(json, "function_calls");
    confidence = json_number(json, "confidence", -1.0);
    ttft = json_number(json, "time_to_first_token_ms");
    total_ms = json_number(json, "total_time_ms");
    prefill_tps = json_number(json, "prefill_tps");
    decode_tps = json_number(json, "decode_tps");
    ram_mb = json_number(json, "ram_usage_mb");
    prefill_tokens = json_number(json, "prefill_tokens");
    completion_tokens = json_number(json, "decode_tokens");
    total_tokens = json_number(json, "total_tokens");
    segments = json_array(json, "segments");
}

void Metrics::print_json() const {
    std::cout << "  \"success\": " << (success ? "true" : "false") << ",\n"
              << "  \"error\": " << (error.empty() ? "null" : "\"" + error + "\"") << ",\n"
              << "  \"cloud_handoff\": " << (cloud_handoff ? "true" : "false") << ",\n"
              << "  \"response\": \"" << response << "\",\n"
              << "  \"thinking\": " << (thinking.empty() ? "null" : "\"" + thinking + "\"") << ",\n"
              << "  \"function_calls\": " << function_calls << ",\n"
              << "  \"segments\": " << segments << ",\n"
              << "  \"confidence\": " << std::fixed << std::setprecision(4) << confidence << ",\n"
              << "  \"time_to_first_token_ms\": " << std::setprecision(2) << ttft << ",\n"
              << "  \"total_time_ms\": " << total_ms << ",\n"
              << "  \"prefill_tps\": " << prefill_tps << ",\n"
              << "  \"decode_tps\": " << decode_tps << ",\n"
              << "  \"ram_usage_mb\": " << ram_mb << ",\n"
              << "  \"prefill_tokens\": " << std::setprecision(0) << prefill_tokens << ",\n"
              << "  \"decode_tokens\": " << completion_tokens << ",\n"
              << "  \"total_tokens\": " << total_tokens << std::endl;
}

void PrefillMetrics::parse(const std::string& json) {
    success = json_bool(json, "success", false);
    error = json_string(json, "error");
    prefill_tokens = json_number(json, "prefill_tokens");
    prefill_tps = json_number(json, "prefill_tps");
    total_ms = json_number(json, "total_time_ms");
    ram_mb = json_number(json, "ram_usage_mb");
}

std::string PrefillMetrics::line() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "prefill_tokens=" << std::setprecision(0) << prefill_tokens
        << ", prefill_tps=" << std::setprecision(2) << prefill_tps
        << ", total_time_ms=" << std::setprecision(2) << total_ms
        << ", ram_usage_mb=" << std::setprecision(2) << ram_mb;
    return oss.str();
}

void PrefillMetrics::print_line() const {
    std::cout << line();
}

static std::mt19937 g_rng(7);

std::vector<__fp16> rand_halfs(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<__fp16> v(n);
    for (auto& x : v) x = (__fp16)d(g_rng);
    return v;
}

bool close_all(const std::vector<__fp16>& got, const std::vector<float>& want,
               float tol, const char* what) {
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::fabs((float)got[i] - want[i]) > tol) {
            std::cerr << "  [✗] " << what << " diverged at " << i << ": got "
                      << (float)got[i] << " want " << want[i] << "\n";
            return false;
        }
    }
    return true;
}

float gelu_ref(float x) {
    float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(c));
}

CQ4Fixture::CQ4Fixture(bool interleaved, uint32_t K_, uint32_t N_, uint32_t gs_)
    : K(K_), N(N_), gs(gs_), ng(K_ / gs_), pgb(gs_ / 2) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> pos(0.5f, 1.5f);
    std::uniform_int_distribution<int> coin(0, 1);

    codebook.resize(16);
    for (int i = 0; i < 16; ++i) codebook[i] = (__fp16)((i - 7.5f) / 7.5f);
    recip.resize(K);
    for (auto& v : recip) v = (__fp16)pos(g_rng);
    ls.resize(gs); rs.resize(gs);
    for (auto& s : ls) s = coin(g_rng) ? 1 : -1;
    for (auto& s : rs) s = coin(g_rng) ? 1 : -1;
    perm.resize(gs);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), g_rng);

    idx.resize((size_t)N * K);
    for (auto& v : idx) v = (uint8_t)nib(g_rng);
    norms.resize((size_t)N * ng);
    packed.assign((size_t)N * ng * pgb, 0);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t g = 0; g < ng; ++g) {
            float nv = pos(g_rng);
            if (interleaved) norms[(((size_t)(n >> 2) * ng + g) << 2) + (n & 3u)] = (__fp16)nv;
            else norms[(size_t)n * ng + g] = (__fp16)nv;
            for (uint32_t e = 0; e < gs; ++e) {
                uint8_t v = idx[(size_t)n * K + g * gs + e];
                size_t byte;
                uint32_t shift;
                if (interleaved) {
                    uint32_t blk = e >> 4, j = e & 15u, sub = j >> 2, b = j & 3u;
                    byte = ((size_t)(n >> 2) * ng + g) * 4u * pgb
                         + (2u * blk + (sub >> 1)) * 16u + (n & 3u) * 4u + b;
                    shift = (sub & 1u) * 4u;
                } else {
                    byte = ((size_t)n * ng + g) * pgb + (e >> 1);
                    shift = (e & 1u) * 4u;
                }
                packed[byte] |= (uint8_t)(v << shift);
            }
        }

    W.bits = 4; W.K = K; W.N = N; W.group_size = gs; W.num_groups = ng;
    W.flags = interleaved ? CACTUS_QUANT_FLAG_INTERLEAVED_4ROW : 0;
    W.codebook = codebook.data();
    W.input_scale_recip = recip.data();
    W.norms = norms.data();
    W.packed_indices = packed.data();
    W.left_signs = ls.data();
    W.right_signs = rs.data();
    W.permutation = perm.data();
}

std::vector<float> CQ4Fixture::oracle(const std::vector<__fp16>& x) const {
    std::vector<__fp16> code(K);
    for (uint32_t g = 0; g < ng; ++g) {
        std::vector<float> z(gs);
        for (uint32_t k = 0; k < gs; ++k) {
            uint32_t gk = g * gs + k;
            z[k] = (float)x[gk] * (float)recip[gk] * (float)ls[k];
        }
        for (uint32_t h = 1; h < gs; h <<= 1)
            for (uint32_t k = 0; k < gs; ++k)
                if ((k & h) == 0) { float a = z[k], b = z[k + h]; z[k] = a + b; z[k + h] = a - b; }
        float s = 1.0f / std::sqrt((float)gs);
        for (uint32_t k = 0; k < gs; ++k) z[k] *= s * (float)rs[k];
        for (uint32_t k = 0; k < gs; ++k) code[g * gs + k] = (__fp16)z[perm[k]];
    }
    const bool il = (W.flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    std::vector<float> y(N);
    for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t g = 0; g < ng; ++g) {
            float nm = il ? (float)norms[(((size_t)(n >> 2) * ng + g) << 2) + (n & 3u)]
                          : (float)norms[(size_t)n * ng + g];
            double p = 0;
            for (uint32_t e = 0; e < gs; ++e)
                p += (float)code[g * gs + e] * (float)codebook[idx[(size_t)n * K + g * gs + e]];
            acc += nm * p;
        }
        y[n] = (float)acc;
    }
    return y;
}

}
