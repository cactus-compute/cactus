#include "test_utils.h"
#include <sstream>

namespace TestUtils {

TestRunner::TestRunner(const std::string& suite_name)
    : suite_name_(suite_name), passed_count_(0), total_count_(0) {
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

void TestRunner::log_performance(const std::string& test_name, const std::string& details) {
    std::cout << "⚡PERF │ " << std::left << std::setw(38) << test_name << " │ " << details << "\n";
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

}
