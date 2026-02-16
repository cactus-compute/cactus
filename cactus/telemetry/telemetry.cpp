#include "telemetry/telemetry.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>
#include <curl/curl.h>
#include <dirent.h>
#include <functional>
#include <cmath>
#include <limits>

namespace cactus {
namespace telemetry {

enum EventType { INIT = 0, COMPLETION = 1, EMBEDDING = 2, TRANSCRIPTION = 3, STREAM_TRANSCRIBE = 4 };

struct Event {
    EventType type;
    char model[128];
    bool success;
    bool cloud_handoff;
    double ttft_ms;
    double prefill_tps;
    double decode_tps;
    double tps;
    double response_time_ms;
    double confidence;
    double ram_usage_mb;
    int tokens;
    int prefill_tokens;
    int decode_tokens;
    double session_ttft_ms;
    double session_tps;
    double session_time_ms;
    int session_tokens;
    char message[256];
    char error[256];
    char function_calls[1024];
    std::chrono::system_clock::time_point timestamp;
};
static std::atomic<bool> enabled{false};
static std::atomic<int> inference_active{0};
static std::atomic<bool> shutdown_called{false};
static std::atomic<bool> atexit_registered{false};
static std::atomic<bool> cloud_disabled{false};
static std::string supabase_url = "https://vlqqczxwyaodtcdmdmlw.supabase.co";
static std::string supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscXFjenh3eWFvZHRjZG1kbWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MTg2MzIsImV4cCI6MjA2NzA5NDYzMn0.nBzqGuK9j6RZ6mOPWU2boAC_5H9XDs-fPpo5P3WZYbI";
static std::string device_id;
static std::string project_id;
static std::string cloud_key;
static std::string project_scope;
static std::string device_model;
static std::string device_os;
static std::string device_os_version;
static std::string device_brand;
static std::string device_registered_file;
static std::string project_registered_file;
static std::atomic<bool> device_registered{false};
static std::atomic<bool> project_registered{false};
static std::atomic<bool> ids_ready{false};
static std::atomic<bool> in_stream_mode{false};

static std::string new_uuid();
static std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
static std::string model_basename(const char* model_path) {
    if (!model_path) return {};
    std::string m(model_path);
    size_t pos = m.find_last_of("/\\");
    if (pos != std::string::npos && pos + 1 < m.size()) {
        m = m.substr(pos + 1);
    }
    return m;
}

// Forward declarations for helpers used before definition
static std::string event_type_to_string(EventType t);
static bool event_type_from_string(const std::string& s, EventType& t);
static bool extract_string_field(const std::string& line, const std::string& key, std::string& out);
static bool extract_bool_field(const std::string& line, const std::string& key, bool& out);
static bool extract_double_field(const std::string& line, const std::string& key, double& out);
static bool extract_int_field(const std::string& line, const std::string& key, int& out);
static bool extract_double_field_raw(const std::string& line, const std::string& key, double& out);

static std::string get_telemetry_dir() {
    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/Library/Caches/cactus/telemetry";
    mkdir((std::string(home) + "/Library").c_str(), 0755);
    mkdir((std::string(home) + "/Library/Caches").c_str(), 0755);
    mkdir((std::string(home) + "/Library/Caches/cactus").c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    return dir;
}

static std::string scoped_file_name(const std::string& prefix, const std::string& scope) {
    std::hash<std::string> hasher;
    size_t h = hasher(scope);
    std::ostringstream oss;
    oss << prefix << std::hex << h;
    return oss.str();
}

static std::string load_or_create_id(const std::string& file) {
    std::ifstream in(file);
    if (in.is_open()) {
        std::string line;
        if (std::getline(in, line) && !line.empty()) {
            return line;
        }
    }
    std::string id = new_uuid();
    std::ofstream out(file, std::ios::trunc);
    if (out.is_open()) {
        out << id;
    }
    return id;
}

static bool load_registered_flag(const std::string& file) {
    std::ifstream in(file);
    if (!in.is_open()) return false;
    std::string line;
    if (std::getline(in, line) && !line.empty()) {
        return line[0] == '1';
    }
    return false;
}

static void persist_registered_flag(const std::string& file) {
    std::ofstream out(file, std::ios::trunc);
    if (out.is_open()) {
        out << "1";
    }
}

static std::string sysctl_string(const char* key) {
    size_t size = 0;
    if (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) return {};
    std::string out(size, '\0');
    if (sysctlbyname(key, out.data(), &size, nullptr, 0) != 0) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

static Event make_event(EventType type, const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    Event e;
    e.type = type;
    e.success = success;
    e.cloud_handoff = false;
    e.ttft_ms = ttft_ms;
    e.prefill_tps = 0.0;
    e.decode_tps = tps;
    e.tps = tps;
    e.response_time_ms = response_time_ms;
    e.confidence = 0.0;
    e.ram_usage_mb = 0.0;
    e.tokens = tokens;
    e.prefill_tokens = 0;
    e.decode_tokens = tokens;
    e.session_ttft_ms = 0.0;
    e.session_tps = 0.0;
    e.session_time_ms = 0.0;
    e.session_tokens = 0;
    e.timestamp = std::chrono::system_clock::now();
    std::memset(e.model, 0, sizeof(e.model));
    std::memset(e.message, 0, sizeof(e.message));
    std::memset(e.error, 0, sizeof(e.error));
    std::memset(e.function_calls, 0, sizeof(e.function_calls));
    std::string safe_model = model_basename(model);
    if (!safe_model.empty()) std::strncpy(e.model, safe_model.c_str(), sizeof(e.model)-1);
    if (message) std::strncpy(e.message, message, sizeof(e.message)-1);
    return e;
}

static Event make_event_extended(EventType type, const char* model, const CompletionMetrics& metrics) {
    Event e;
    e.type = type;
    e.success = metrics.success;
    e.cloud_handoff = metrics.cloud_handoff;
    e.ttft_ms = metrics.ttft_ms;
    e.prefill_tps = metrics.prefill_tps;
    e.decode_tps = metrics.decode_tps;
    e.tps = metrics.decode_tps;
    e.response_time_ms = metrics.response_time_ms;
    e.confidence = metrics.confidence;
    e.ram_usage_mb = metrics.ram_usage_mb;
    e.tokens = static_cast<int>(metrics.prefill_tokens + metrics.decode_tokens);
    e.prefill_tokens = static_cast<int>(metrics.prefill_tokens);
    e.decode_tokens = static_cast<int>(metrics.decode_tokens);
    e.timestamp = std::chrono::system_clock::now();
    std::memset(e.model, 0, sizeof(e.model));
    std::memset(e.message, 0, sizeof(e.message));
    std::memset(e.error, 0, sizeof(e.error));
    std::memset(e.function_calls, 0, sizeof(e.function_calls));
    std::string safe_model = model_basename(model);
    if (!safe_model.empty()) std::strncpy(e.model, safe_model.c_str(), sizeof(e.model)-1);
    if (!metrics.success && metrics.error_message) std::strncpy(e.error, metrics.error_message, sizeof(e.error)-1);
    if (metrics.function_calls_json) std::strncpy(e.function_calls, metrics.function_calls_json, sizeof(e.function_calls)-1);
    return e;
}

static bool parse_event_line(const std::string& line, Event& out) {
    std::string type_str;
    if (!extract_string_field(line, "event_type", type_str)) return false;
    EventType et;
    if (!event_type_from_string(type_str, et)) return false;

    std::string model;
    extract_string_field(line, "model", model);
    bool success = false;
    extract_bool_field(line, "success", success);
    bool cloud_handoff = false;
    extract_bool_field(line, "cloud_handoff", cloud_handoff);
    double ttft = 0.0;
    extract_double_field(line, "ttft", ttft);
    double prefill_tps = 0.0;
    extract_double_field(line, "prefill_tps", prefill_tps);
    double decode_tps = 0.0;
    extract_double_field(line, "decode_tps", decode_tps);
    double tps = 0.0;
    extract_double_field(line, "tps", tps);
    if (decode_tps == 0.0 && tps != 0.0) decode_tps = tps;
    double response_time = 0.0;
    extract_double_field(line, "response_time", response_time);
    double confidence = 0.0;
    extract_double_field(line, "confidence", confidence);
    double ram_usage_mb = 0.0;
    extract_double_field(line, "ram_usage_mb", ram_usage_mb);
    int tokens = 0;
    extract_int_field(line, "tokens", tokens);
    int prefill_tokens = 0;
    extract_int_field(line, "prefill_tokens", prefill_tokens);
    int decode_tokens = 0;
    extract_int_field(line, "decode_tokens", decode_tokens);
    double session_ttft_ms = 0.0;
    extract_double_field(line, "session_ttft", session_ttft_ms);
    double session_tps = 0.0;
    extract_double_field(line, "session_tps", session_tps);
    double session_time_ms = 0.0;
    extract_double_field(line, "session_time_ms", session_time_ms);
    int session_tokens = 0;
    extract_int_field(line, "session_tokens", session_tokens);
    std::string message;
    extract_string_field(line, "message", message);
    std::string error;
    extract_string_field(line, "error", error);
    std::string function_calls;
    extract_string_field(line, "function_calls", function_calls);
    double ts_ms = 0.0;
    if (!extract_double_field_raw(line, "ts_ms", ts_ms)) {
        extract_double_field(line, "ts_ms", ts_ms);
    }
    auto ts_point = std::chrono::system_clock::now();
    if (ts_ms > 0.0) {
        ts_point = std::chrono::system_clock::time_point(std::chrono::milliseconds(static_cast<long long>(ts_ms)));
    }

    out = make_event(et,
                     model.empty() ? nullptr : model.c_str(),
                     success,
                     ttft,
                     tps,
                     response_time,
                     tokens,
                     message.empty() ? nullptr : message.c_str());
    out.timestamp = ts_point;
    out.cloud_handoff = cloud_handoff;
    out.prefill_tps = prefill_tps;
    out.decode_tps = decode_tps;
    out.confidence = confidence;
    out.ram_usage_mb = ram_usage_mb;
    out.prefill_tokens = prefill_tokens;
    out.decode_tokens = decode_tokens;
    out.session_ttft_ms = session_ttft_ms;
    out.session_tps = session_tps;
    out.session_time_ms = session_time_ms;
    out.session_tokens = session_tokens;
    if (!error.empty()) std::strncpy(out.error, error.c_str(), sizeof(out.error)-1);
    if (!function_calls.empty()) std::strncpy(out.function_calls, function_calls.c_str(), sizeof(out.function_calls)-1);
    return true;
}

static std::string event_type_to_string(EventType t) {
    switch (t) {
        case INIT: return "init";
        case COMPLETION: return "completion";
        case EMBEDDING: return "embedding";
        case TRANSCRIPTION: return "transcription";
        case STREAM_TRANSCRIBE: return "stream_transcribe";
        default: return "unknown";
    }
}

static bool event_type_from_string(const std::string& s, EventType& t) {
    if (s == "init") { t = INIT; return true; }
    if (s == "completion") { t = COMPLETION; return true; }
    if (s == "embedding") { t = EMBEDDING; return true; }
    if (s == "transcription") { t = TRANSCRIPTION; return true; }
    if (s == "stream_transcribe") { t = STREAM_TRANSCRIBE; return true; }
    return false;
}

static bool extract_string_field(const std::string& line, const std::string& key, std::string& out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size()) return false;
    if (line[pos] == '"') {
        pos++;
        size_t end = line.find('"', pos);
        if (end == std::string::npos) return false;
        out = line.substr(pos, end - pos);
        return true;
    }
    size_t end = line.find_first_of(",}", pos);
    if (end == std::string::npos) end = line.size();
    out = line.substr(pos, end - pos);
    if (out == "null") out.clear();
    return true;
}

static bool extract_bool_field(const std::string& line, const std::string& key, bool& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    if (raw == "true") { out = true; return true; }
    if (raw == "false") { out = false; return true; }
    return false;
}

static bool extract_double_field(const std::string& line, const std::string& key, double& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    try {
        out = std::stod(raw);
        return true;
    } catch (...) {
        return false;
    }
}

// Helper used when the field is already numeric (no quotes) in the log cache.
static bool extract_double_field_raw(const std::string& line, const std::string& key, double& out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size()) return false;
    size_t end = line.find_first_of(",}", pos);
    if (end == std::string::npos) end = line.size();
    std::string raw = line.substr(pos, end - pos);
    try {
        out = std::stod(raw);
        return true;
    } catch (...) {
        return false;
    }
}

static bool extract_int_field(const std::string& line, const std::string& key, int& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    try {
        out = std::stoi(raw);
        return true;
    } catch (...) {
        return false;
    }
}

static std::string new_uuid() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    uint64_t a = rng();
    uint64_t b = rng();
    a = (a & 0xffffffffffff0fffULL) | 0x0000000000004000ULL;
    b = (b & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(8) << ((a >> 32) & 0xffffffffULL);
    oss << "-" << std::setw(4) << ((a >> 16) & 0xffffULL);
    oss << "-" << std::setw(4) << (a & 0xffffULL);
    oss << "-" << std::setw(4) << ((b >> 48) & 0xffffULL);
    oss << "-" << std::setw(12) << (b & 0xffffffffffffULL);
    return oss.str();
}

static std::string format_timestamp(const std::chrono::system_clock::time_point& tp) {
    using namespace std::chrono;
    auto secs = time_point_cast<std::chrono::seconds>(tp);
    auto ms = duration_cast<std::chrono::milliseconds>(tp - secs).count();
    std::time_t t = system_clock::to_time_t(secs);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<long long>(ms));
    return std::string(buf);
}

[[maybe_unused]] static bool is_valid_uuid(const std::string& s) {
    // Simple check used only for validation when accepting user-provided IDs.
    if (s.size() != 36) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (c != '-') return false;
        } else {
            bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            if (!hex) return false;
        }
    }
    return true;
}

static void collect_device_info() {
    struct utsname u;
    if (uname(&u) == 0) {
        // Prefer user-facing macOS info over Darwin kernel version.
        std::string hw_model = sysctl_string("hw.model");
        std::string os_product = sysctl_string("kern.osproductversion");
        device_os = (std::string(u.sysname) == "Darwin") ? "macos" : u.sysname;
        device_os_version = !os_product.empty() ? os_product : u.release;
        device_model = !hw_model.empty() ? hw_model : u.machine;
        device_brand = "apple";
    }
}

static bool ensure_project_row(CURL* curl) {
    if (project_id.empty() || project_registered.load()) return true;
    std::string url = supabase_url + "/rest/v1/projects";
    std::ostringstream payload;
    payload << "[{";
    payload << "\"id\":\"" << project_id << "\"";
    payload << ",\"project_key\":\"" << project_id << "\"";
    if (!cloud_key.empty()) {
        payload << ",\"cloud_key\":\"" << cloud_key << "\"";
    }
    if (!project_scope.empty()) {
        payload << ",\"name\":\"" << project_scope << "\"";
    }
    payload << "}]";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string apikey_hdr = std::string("apikey: ") + supabase_key;
    std::string auth_hdr = std::string("Authorization: Bearer ") + supabase_key;
    headers = curl_slist_append(headers, apikey_hdr.c_str());
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "Prefer: resolution=ignore-duplicates");
    headers = curl_slist_append(headers, "Content-Profile: cactus");
    headers = curl_slist_append(headers, "Accept-Profile: cactus");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    std::string body = payload.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (headers) curl_slist_free_all(headers);
    bool ok = (res == CURLE_OK) && ((code >= 200 && code < 300) || code == 409);
    if (ok) {
        if (!project_registered_file.empty()) {
            persist_registered_flag(project_registered_file);
        }
        project_registered.store(true);
    }
    return ok;
}

static bool ensure_device_row(CURL* curl) {
    if (device_registered.load()) return true;
    if (device_id.empty()) device_id = new_uuid();
    if (device_os.empty()) collect_device_info();
    std::string url = supabase_url + "/rest/v1/devices";
    std::ostringstream payload;
    payload << "[{";
    payload << "\"id\":\"" << device_id << "\"";
    payload << ",\"device_id\":\"" << device_id << "\"";
    if (!device_model.empty()) payload << ",\"model\":\"" << device_model << "\"";
    if (!device_os.empty()) payload << ",\"os\":\"" << device_os << "\"";
    if (!device_os_version.empty()) payload << ",\"os_version\":\"" << device_os_version << "\"";
    if (!device_brand.empty()) payload << ",\"brand\":\"" << device_brand << "\"";
    payload << "}]";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string apikey_hdr = std::string("apikey: ") + supabase_key;
    std::string auth_hdr = std::string("Authorization: Bearer ") + supabase_key;
    headers = curl_slist_append(headers, apikey_hdr.c_str());
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "Prefer: resolution=ignore-duplicates");
    headers = curl_slist_append(headers, "Content-Profile: cactus");
    headers = curl_slist_append(headers, "Accept-Profile: cactus");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    std::string body = payload.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (headers) curl_slist_free_all(headers);
    bool ok = (res == CURLE_OK) && ((code >= 200 && code < 300) || code == 409);
    if (ok) {
        persist_registered_flag(device_registered_file);
        device_registered.store(true);
    }
    return ok;
}

static bool send_payload(CURL* curl, const std::string& url, const std::string& body) {
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string apikey_hdr = std::string("apikey: ") + supabase_key;
    std::string auth_hdr = std::string("Authorization: Bearer ") + supabase_key;
    headers = curl_slist_append(headers, apikey_hdr.c_str());
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "Prefer: return=minimal");
    headers = curl_slist_append(headers, "Content-Profile: cactus");
    headers = curl_slist_append(headers, "Accept-Profile: cactus");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (headers) curl_slist_free_all(headers);
    return (res == CURLE_OK) && (code >= 200 && code < 300);
}

static bool send_batch_to_cloud(const std::vector<Event>& local) {
    if (!enabled.load()) return false;
    if (local.empty()) return true;
    if (cloud_disabled.load()) return false;
    if (device_id.empty()) device_id = new_uuid();
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    ensure_project_row(curl);
    if (!device_registered.load()) {
        ensure_device_row(curl);
    }
    std::string url = supabase_url + "/rest/v1/logs";
    std::ostringstream payload;
    payload << "[";
    for (size_t i = 0; i < local.size(); ++i) {
        const Event& e = local[i];
        payload << "{";
        payload << "\"event_type\":\"" << event_type_to_string(e.type) << "\",";
        payload << "\"model\":\"" << e.model << "\",";
        payload << "\"success\":" << (e.success ? "true" : "false") << ",";
        payload << "\"cloud_handoff\":" << (e.cloud_handoff ? "true" : "false") << ",";
        if (e.type == INIT || !e.success) {
            payload << "\"ttft\":null,";
            payload << "\"prefill_tps\":null,";
            payload << "\"decode_tps\":null,";
            payload << "\"tps\":null,";
        } else {
            payload << "\"ttft\":" << e.ttft_ms << ",";
            payload << "\"prefill_tps\":" << e.prefill_tps << ",";
            payload << "\"decode_tps\":" << e.decode_tps << ",";
            payload << "\"tps\":" << e.tps << ",";
        }
        if (!e.success) {
            payload << "\"response_time\":null,";
            payload << "\"ram_usage_mb\":null,";
        } else {
            payload << "\"response_time\":" << e.response_time_ms << ",";
            payload << "\"ram_usage_mb\":" << e.ram_usage_mb << ",";
        }
        if (e.type == INIT || !e.success) {
            payload << "\"confidence\":null,";
            payload << "\"tokens\":null,";
            payload << "\"prefill_tokens\":null,";
            payload << "\"decode_tokens\":null,";
            payload << "\"session_ttft\":null,";
            payload << "\"session_tps\":null,";
            payload << "\"session_time_ms\":null,";
            payload << "\"session_tokens\":null,";
        } else {
            payload << "\"confidence\":" << e.confidence << ",";
            payload << "\"tokens\":" << e.tokens << ",";
            payload << "\"prefill_tokens\":" << e.prefill_tokens << ",";
            payload << "\"decode_tokens\":" << e.decode_tokens << ",";
            payload << "\"session_ttft\":" << e.session_ttft_ms << ",";
            payload << "\"session_tps\":" << e.session_tps << ",";
            payload << "\"session_time_ms\":" << e.session_time_ms << ",";
            payload << "\"session_tokens\":" << e.session_tokens << ",";
        }
        payload << "\"created_at\":\"" << format_timestamp(e.timestamp) << "\",";
        if (!project_id.empty()) {
            payload << "\"project_id\":\"" << project_id << "\",";
        }
        if (!cloud_key.empty()) {
            payload << "\"cloud_key\":\"" << cloud_key << "\",";
        }
        payload << "\"framework\":\"cpp\",";
        payload << "\"device_id\":\"" << device_id << "\"";
        if (e.message[0] != '\0') {
            payload << ",\"message\":\"" << e.message << "\"";
        }
        if (e.error[0] != '\0') {
            payload << ",\"error\":\"" << e.error << "\"";
        } else {
            payload << ",\"error\":null";
        }
        if (e.function_calls[0] != '\0') {
            payload << ",\"function_calls\":" << e.function_calls;
        } else {
            payload << ",\"function_calls\":[]";
        }
        payload << "}";
        if (i + 1 < local.size()) payload << ",";
    }
    payload << "]";
    bool ok = send_payload(curl, url, payload.str());
    curl_easy_cleanup(curl);
    return ok;
}

static void write_events_to_cache(const std::vector<Event>& local) {
    std::string dir = get_telemetry_dir();
    for (const auto &e : local) {
        std::ostringstream oss;
        oss << "{\"event_type\":\"" << event_type_to_string(e.type) << "\",";
        oss << "\"model\":\"" << e.model << "\",";
        oss << "\"success\":" << (e.success ? "true" : "false") << ",";
        oss << "\"cloud_handoff\":" << (e.cloud_handoff ? "true" : "false") << ",";
        if (e.type == INIT || !e.success) {
            oss << "\"ttft\":null,";
            oss << "\"prefill_tps\":null,";
            oss << "\"decode_tps\":null,";
            oss << "\"tps\":null,";
        } else {
            oss << "\"ttft\":" << e.ttft_ms << ",";
            oss << "\"prefill_tps\":" << e.prefill_tps << ",";
            oss << "\"decode_tps\":" << e.decode_tps << ",";
            oss << "\"tps\":" << e.tps << ",";
        }
        if (!e.success) {
            oss << "\"response_time\":null,";
            oss << "\"ram_usage_mb\":null,";
        } else {
            oss << "\"response_time\":" << e.response_time_ms << ",";
            oss << "\"ram_usage_mb\":" << e.ram_usage_mb << ",";
        }
        if (e.type == INIT || !e.success) {
            oss << "\"confidence\":null,";
            oss << "\"tokens\":null,";
            oss << "\"prefill_tokens\":null,";
            oss << "\"decode_tokens\":null,";
            oss << "\"session_ttft\":null,";
            oss << "\"session_tps\":null,";
            oss << "\"session_time_ms\":null,";
            oss << "\"session_tokens\":null";
        } else {
            oss << "\"confidence\":" << e.confidence << ",";
            oss << "\"tokens\":" << e.tokens << ",";
            oss << "\"prefill_tokens\":" << e.prefill_tokens << ",";
            oss << "\"decode_tokens\":" << e.decode_tokens << ",";
            oss << "\"session_ttft\":" << e.session_ttft_ms << ",";
            oss << "\"session_tps\":" << e.session_tps << ",";
            oss << "\"session_time_ms\":" << e.session_time_ms << ",";
            oss << "\"session_tokens\":" << e.session_tokens;
        }
        oss << ",\"ts_ms\":" << std::chrono::duration_cast<std::chrono::milliseconds>(e.timestamp.time_since_epoch()).count();
        if (e.message[0] != '\0') {
            oss << ",\"message\":\"" << e.message << "\"";
        }
        if (e.error[0] != '\0') {
            oss << ",\"error\":\"" << e.error << "\"";
        } else {
            oss << ",\"error\":null";
        }
        if (e.function_calls[0] != '\0') {
            oss << ",\"function_calls\":" << e.function_calls;
        } else {
            oss << ",\"function_calls\":[]";
        }
        oss << "}";
        std::string file = dir + "/" + event_type_to_string(e.type) + ".log";
        std::ofstream out(file, std::ios::app);
        if (out.is_open()) {
            out << oss.str() << "\n";
            out.close();
        }
    }
}
static std::vector<Event> load_cached_events() {
    std::vector<Event> events;
    std::string dir = get_telemetry_dir();
    DIR* d = opendir(dir.c_str());
    if (!d) return events;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".log") continue;
        std::ifstream in(dir + "/" + name);
        std::string line;
        while (std::getline(in, line)) {
            Event e;
            if (parse_event_line(line, e)) {
                events.push_back(e);
            }
        }
    }
    closedir(d);
    return events;
}

static void clear_cache_files() {
    std::string dir = get_telemetry_dir();
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".log") continue;
        std::remove((dir + "/" + name).c_str());
    }
    closedir(d);
}

static void flush_logs_with_event(const Event* latest) {
    std::vector<Event> events = load_cached_events();
    if (latest) events.push_back(*latest);
    if (events.empty()) return;
    bool cloud_ok = send_batch_to_cloud(events);
    if (cloud_ok) {
        clear_cache_files();
    } else if (latest) {
        write_events_to_cache({*latest});
    }
}

void init(const char* project_id_param, const char* project_scope, const char* cloud_key_param) {
    std::string scope = project_scope ? project_scope : "default";
    if (const char* env = std::getenv("CACTUS_NO_CLOUD_TELE")) {
        if (env[0] != '\0' && !(env[0] == '0' && env[1] == '\0')) {
            cloud_disabled.store(true);
        }
    }
    const char* env_url = std::getenv("CACTUS_SUPABASE_URL");
    const char* env_key = std::getenv("CACTUS_SUPABASE_KEY");
    if (env_url && *env_url) supabase_url = env_url;
    if (env_key && *env_key) supabase_key = env_key;

    const char* env_project = std::getenv("CACTUS_PROJECT_ID");
    std::string dir = get_telemetry_dir();
    if (project_id_param && *project_id_param) {
        project_id = project_id_param;
    } else if (env_project && *env_project) {
        project_id = env_project;
    } else {
        std::string file = dir + "/" + scoped_file_name("project_", scope);
        project_id = load_or_create_id(file);
    }

    std::string project_flag_file = dir + "/" + scoped_file_name("project_reg_", scope);
    project_registered_file = project_flag_file;
    project_registered.store(load_registered_flag(project_flag_file));

    const char* env_cloud = std::getenv("CACTUS_CLOUD_KEY");
    if (cloud_key_param && *cloud_key_param) {
        cloud_key = cloud_key_param;
    } else if (env_cloud && *env_cloud) {
        cloud_key = env_cloud;
    }

    std::string device_file = dir + "/device_id";
    std::string device_flag_file = dir + "/device_registered";
    device_registered_file = device_flag_file;
    device_registered.store(load_registered_flag(device_flag_file));
    device_id = load_or_create_id(device_file);
    collect_device_info();

    curl_global_init(CURL_GLOBAL_DEFAULT);
    if (!atexit_registered.exchange(true)) {
        std::atexit([](){ shutdown(); });
    }
    ids_ready.store(true);
    setEnabled(true);
}

void setEnabled(bool en) {
    enabled.store(en);
}

void setCloudDisabled(bool disabled) {
    cloud_disabled.store(disabled);
}

void recordInit(const char* model, bool success, double response_time_ms, const char* message) {
    if (!enabled.load() || !ids_ready.load()) return;
    double nan = std::numeric_limits<double>::quiet_NaN();
    Event e = make_event(INIT, model, success, nan, nan, response_time_ms, 0, message);
    if (success) {
        write_events_to_cache({e});
    } else {
        flush_logs_with_event(&e);
    }
}

void recordCompletion(const char* model, const CompletionMetrics& metrics) {
    if (!enabled.load() || !ids_ready.load()) return;
    Event e = make_event_extended(COMPLETION, model, metrics);
    flush_logs_with_event(&e);
}

void recordCompletion(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    if (!enabled.load() || !ids_ready.load()) return;
    Event e = make_event(COMPLETION, model, success, ttft_ms, tps, response_time_ms, tokens, message);
    flush_logs_with_event(&e);
}

void recordEmbedding(const char* model, bool success, const char* message) {
    if (!enabled.load() || !ids_ready.load()) return;
    Event e = make_event(EMBEDDING, model, success, 0.0, 0.0, 0.0, 0, message);
    flush_logs_with_event(&e);
}

void recordTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    if (!enabled.load() || !ids_ready.load()) return;
    if (in_stream_mode.load()) return;
    Event e = make_event(TRANSCRIPTION, model, success, ttft_ms, tps, response_time_ms, tokens, message);
    flush_logs_with_event(&e);
}

void recordStreamTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, double session_ttft_ms, double session_tps, double session_time_ms, int session_tokens, const char* message) {
    if (!enabled.load() || !ids_ready.load()) return;
    Event e = make_event(STREAM_TRANSCRIBE, model, success, ttft_ms, tps, response_time_ms, tokens, message);
    e.session_ttft_ms = session_ttft_ms;
    e.session_tps = session_tps;
    e.session_time_ms = session_time_ms;
    e.session_tokens = session_tokens;
    flush_logs_with_event(&e);
}

void setStreamMode(bool in_stream) {
    in_stream_mode.store(in_stream);
}

void markInference(bool active) {
    if (active) inference_active.fetch_add(1);
    else {
        int v = inference_active.load();
        if (v > 0) inference_active.fetch_sub(1);
    }
}

void flush() {
    flush_logs_with_event(nullptr);
}

void shutdown() {
    if (!shutdown_called.exchange(true)) {
        flush();
    }
    curl_global_cleanup();
}

} // namespace telemetry
} // namespace cactus
