#ifndef CACTUS_TELEMETRY_H
#define CACTUS_TELEMETRY_H

#include <string>
#include <thread>
#include <sstream>
#include <iostream>
#include <chrono>
#include <mutex>
#include <map>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include "cactus_utils.h"

#if defined(__APPLE__) && !defined(TARGET_OS_IPHONE) && !defined(__ANDROID__)
#define CACTUS_TELEMETRY_ENABLED
#include <curl/curl.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace cactus {
namespace ffi {

enum class TelemetryEventType {
    Init,
    Completion,
    Embedding,
    Transcription
};

struct TelemetryMetrics {
    TelemetryEventType event_type;
    std::string model;

    double ttft_ms = 0.0;
    double tps = 0.0;
    double response_time_ms = 0.0;
    int tokens = 0;

    bool success = false;
    std::string message;

    std::chrono::system_clock::time_point timestamp;
};


class HttpClient {
public:
    struct Response {
        bool success;
        int status_code;
        std::string body;
    };

    static Response postJson(
        const std::string& url,
        const std::map<std::string, std::string>& headers,
        const std::string& json_body
    );

private:
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

inline size_t HttpClient::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

inline HttpClient::Response HttpClient::postJson(
    [[maybe_unused]] const std::string& url,
    [[maybe_unused]] const std::map<std::string, std::string>& headers,
    [[maybe_unused]] const std::string& json_body
) {
#ifdef CACTUS_TELEMETRY_ENABLED
    Response response;
    response.success = false;
    response.status_code = 0;

    CURL* curl = curl_easy_init();
    if (!curl) {
        return response;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_body.length());

    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);

    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        long response_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        response.status_code = static_cast<int>(response_code);
        response.success = (response_code >= 200 && response_code < 300);

        if (!response.success && !response.body.empty()) {
            std::cerr << "[Telemetry] Response body: " << response.body << std::endl;
        }
    } else {
        std::cerr << "[Telemetry] HTTP POST failed: " << curl_easy_strerror(res) << std::endl;
    }

    if (header_list) {
        curl_slist_free_all(header_list);
    }
    curl_easy_cleanup(curl);

    return response;
#else
    (void)url;
    (void)headers;
    (void)json_body;
    Response response;
    response.success = false;
    response.status_code = 0;
    return response;
#endif
}

class DeviceManager {
public:
    static std::string getDeviceId();
    static std::string getProjectId();
    static std::map<std::string, std::string> getDeviceMetadata();
    static std::string registerDevice();

private:
    static std::string getConfigPath();
    static std::map<std::string, std::string> readConfig();
    static void writeConfig(const std::map<std::string, std::string>& config);
};

inline std::string DeviceManager::getConfigPath() {
    const char* home = getenv("HOME");
    if (!home) {
        home = "/tmp";
    }

    std::string cactus_dir = std::string(home) + "/.cactus";

    mkdir(cactus_dir.c_str(), 0755);

    return cactus_dir + "/telemetry_config.json";
}

inline std::map<std::string, std::string> DeviceManager::readConfig() {
    std::map<std::string, std::string> config;
    std::string path = getConfigPath();
    std::ifstream file(path);

    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string content = buffer.str();

        const std::string device_id_key = "\"device_id\":\"";
        size_t device_pos = content.find(device_id_key);
        if (device_pos != std::string::npos) {
            size_t start = device_pos + device_id_key.length();
            size_t end = content.find("\"", start);
            if (end != std::string::npos) {
                config["device_id"] = content.substr(start, end - start);
            }
        }

        const std::string project_id_key = "\"project_id\":\"";
        size_t project_pos = content.find(project_id_key);
        if (project_pos != std::string::npos) {
            size_t start = project_pos + project_id_key.length();
            size_t end = content.find("\"", start);
            if (end != std::string::npos) {
                config["project_id"] = content.substr(start, end - start);
            }
        }
    }

    return config;
}

inline void DeviceManager::writeConfig(const std::map<std::string, std::string>& config) {
    std::string path = getConfigPath();
    std::ofstream file(path);

    if (file.is_open()) {
        file << "{\n";

        auto device_it = config.find("device_id");
        if (device_it != config.end()) {
            file << "  \"device_id\":\"" << device_it->second << "\"";
        }

        auto project_it = config.find("project_id");
        if (project_it != config.end()) {
            if (device_it != config.end()) {
                file << ",\n";
            }
            file << "  \"project_id\":\"" << project_it->second << "\"";
        }

        file << "\n}\n";
        file.close();
    }
}

inline std::string DeviceManager::getDeviceId() {
    auto config = readConfig();
    std::string device_id = config["device_id"];

    if (!device_id.empty()) {
        std::cerr << "[Device Manager] Using cached device ID: " << device_id << std::endl;
        return device_id;
    }

    std::cerr << "[Device Manager] No cached device ID found, registering new device..." << std::endl;

    device_id = registerDevice();

    if (!device_id.empty()) {
        std::cerr << "[Device Manager] Registration successful, caching config" << std::endl;

        std::string project_id = config["project_id"];
        if (project_id.empty()) {
            project_id = generateUUID();
            std::cerr << "[Device Manager] Generated new project ID: " << project_id << std::endl;
        }

        config["device_id"] = device_id;
        config["project_id"] = project_id;
        writeConfig(config);
    } else {
        std::cerr << "[Device Manager] ERROR: Device registration failed!" << std::endl;
        std::cerr << "[Device Manager] No device ID will be cached. Telemetry will not work." << std::endl;
    }

    return device_id;
}

inline std::string DeviceManager::getProjectId() {
    auto config = readConfig();
    std::string project_id = config["project_id"];

    if (!project_id.empty()) {
        return project_id;
    }

    project_id = generateUUID();
    std::cerr << "[Device Manager] Generated new project ID: " << project_id << std::endl;

    config["project_id"] = project_id;
    writeConfig(config);

    return project_id;
}

inline std::string DeviceManager::registerDevice() {
#ifdef CACTUS_TELEMETRY_ENABLED
    static const std::string SUPABASE_URL = "https://vlqqczxwyaodtcdmdmlw.supabase.co";
    static const std::string SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscXFjenh3eWFvZHRjZG1kbWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MTg2MzIsImV4cCI6MjA2NzA5NDYzMn0.nBzqGuK9j6RZ6mOPWU2boAC_5H9XDs-fPpo5P3WZYbI";

    std::string device_id = generateUUID();

    auto metadata = getDeviceMetadata();

    std::ostringstream json;
    json << "{";
    json << "\"device_id\":\"" << device_id << "\",";
    json << "\"model\":\"" << metadata["model"] << "\",";
    json << "\"os\":\"" << metadata["os"] << "\",";
    json << "\"os_version\":\"" << metadata["os_version"] << "\",";
    json << "\"brand\":\"" << metadata["brand"] << "\"";
    json << "}";

    std::string payload = json.str();

    std::map<std::string, std::string> headers;
    headers["apikey"] = SUPABASE_KEY;
    headers["Authorization"] = "Bearer " + SUPABASE_KEY;
    headers["Content-Type"] = "application/json";
    headers["Accept-Profile"] = "cactus";
    headers["Content-Profile"] = "cactus";
    headers["Prefer"] = "return=representation";

    std::string url = SUPABASE_URL + "/rest/v1/devices";

    auto response = HttpClient::postJson(url, headers, payload);

    if (response.success && !response.body.empty()) {
        std::string response_id;

        const std::string id_key = "\"id\":\"";
        size_t id_pos = response.body.find(id_key);
        if (id_pos != std::string::npos) {
            size_t start = id_pos + id_key.length();
            size_t end = response.body.find("\"", start);
            if (end != std::string::npos) {
                response_id = response.body.substr(start, end - start);
            }
        }

        if (!response_id.empty()) {
            std::cerr << "[Device Registration] SUCCESS - Device registered!" << std::endl;
            return response_id;
        } else {
            std::cerr << "[Device Registration] FAILED - Could not parse ID from response" << std::endl;
            return "";
        }
    } else {
        std::cerr << "[Device Registration] FAILED - Direct table insertion unsuccessful" << std::endl;
        return "";
    }
#else
    return "";
#endif
}

inline std::map<std::string, std::string> DeviceManager::getDeviceMetadata() {
    std::map<std::string, std::string> metadata;

#ifdef CACTUS_TELEMETRY_ENABLED
    struct utsname system_info;
    if (uname(&system_info) == 0) {
        metadata["os"] = "macOS";
        metadata["os_version"] = system_info.release;
        metadata["architecture"] = system_info.machine;
        metadata["model"] = system_info.machine;
        metadata["brand"] = "apple";
    }
#else
    metadata["os"] = "unknown";
    metadata["os_version"] = "unknown";
    metadata["architecture"] = "unknown";
    metadata["model"] = "unknown";
    metadata["brand"] = "unknown";
#endif

    return metadata;
}

class LogRecord {
public:
    static std::string buildJson(
        const TelemetryMetrics& metrics,
        const std::string& project_id,
        const std::string& device_id,
        const std::string& telemetry_token
    );

private:
    static std::string escapeJson(const std::string& input);
    static std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp);
    static std::string eventTypeToString(TelemetryEventType type);
};

inline std::string LogRecord::escapeJson(const std::string& input) {
    std::ostringstream output;
    for (char c : input) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                          << static_cast<int>(c);
                } else {
                    output << c;
                }
        }
    }
    return output.str();
}

inline std::string LogRecord::formatTimestamp(const std::chrono::system_clock::time_point& timestamp) {
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()) % 1000;

    std::tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &time_t);
#else
    gmtime_r(&time_t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

inline std::string LogRecord::eventTypeToString(TelemetryEventType type) {
    switch (type) {
        case TelemetryEventType::Init:
            return "init";
        case TelemetryEventType::Completion:
            return "completion";
        case TelemetryEventType::Embedding:
            return "embedding";
        case TelemetryEventType::Transcription:
            return "transcription";
        default:
            return "unknown";
    }
}

inline std::string LogRecord::buildJson(
    const TelemetryMetrics& metrics,
    const std::string& project_id,
    const std::string& device_id,
    const std::string& telemetry_token
) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);

    json << "{";
    json << "\"event_type\":\"" << eventTypeToString(metrics.event_type) << "\",";
    json << "\"model\":\"" << escapeJson(metrics.model) << "\",";
    json << "\"success\":" << (metrics.success ? "true" : "false") << ",";
    json << "\"project_id\":\"" << project_id << "\",";
    json << "\"device_id\":\"" << device_id << "\",";
    json << "\"telemetry_token\":\"" << telemetry_token << "\",";
    json << "\"framework\":\"cpp\",";
    json << "\"framework_version\":\"" << getVersion() << "\"";

    json << ",\"ttft\":" << metrics.ttft_ms;
    json << ",\"tps\":" << metrics.tps;
    json << ",\"response_time\":" << metrics.response_time_ms;
    json << ",\"tokens\":" << metrics.tokens;

    if (!metrics.message.empty()) {
        json << ",\"message\":\"" << escapeJson(metrics.message) << "\"";
    }

    json << "}";
    return json.str();
}

class CactusTelemetry {
public:
    static CactusTelemetry& getInstance();

    void setEnabled(bool enabled);
    void setTelemetryToken(const std::string& token);
    void setProjectId(const std::string& project_id);

    void recordEvent(const TelemetryMetrics& metrics);

    void recordInit(const std::string& model, bool success, const std::string& message = "");

    void recordCompletion(const std::string& model, bool success,
                         double ttft_ms, double tps, double response_time_ms,
                         int tokens, const std::string& message = "");

    void recordEmbedding(const std::string& model, bool success,
                        const std::string& message = "");

    void recordTranscription(const std::string& model, bool success,
                           double ttft_ms, double tps, double response_time_ms,
                           int tokens, const std::string& message = "");

    bool isEnabled() const;

private:
    CactusTelemetry();
    ~CactusTelemetry() = default;

    CactusTelemetry(const CactusTelemetry&) = delete;
    CactusTelemetry& operator=(const CactusTelemetry&) = delete;

    void sendToSupabase(const TelemetryMetrics& metrics);

    bool enabled_ = false;
    std::string telemetry_token_;
    std::string project_id_;
    std::string device_id_;

    mutable std::mutex mutex_;
};

static const std::string SUPABASE_URL = "https://vlqqczxwyaodtcdmdmlw.supabase.co";
static const std::string SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscXFjenh3eWFvZHRjZG1kbWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MTg2MzIsImV4cCI6MjA2NzA5NDYzMn0.nBzqGuK9j6RZ6mOPWU2boAC_5H9XDs-fPpo5P3WZYbI";

inline CactusTelemetry& CactusTelemetry::getInstance() {
    static CactusTelemetry instance;
    return instance;
}

inline CactusTelemetry::CactusTelemetry() {
#ifdef CACTUS_TELEMETRY_ENABLED
    device_id_ = DeviceManager::getDeviceId();
    project_id_ = DeviceManager::getProjectId();
#endif
}

inline void CactusTelemetry::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

inline void CactusTelemetry::setTelemetryToken(const std::string& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_token_ = token;
}

inline void CactusTelemetry::setProjectId(const std::string& project_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    project_id_ = project_id;
}

inline bool CactusTelemetry::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && !telemetry_token_.empty();
}

inline void CactusTelemetry::sendToSupabase([[maybe_unused]] const TelemetryMetrics& metrics) {
#ifdef CACTUS_TELEMETRY_ENABLED
    std::string telemetry_token;
    std::string project_id;
    std::string device_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        telemetry_token = telemetry_token_;
        project_id = project_id_;
        device_id = device_id_;
    }

    std::string log_json = LogRecord::buildJson(metrics, project_id, device_id, telemetry_token);

    std::string payload = "[" + log_json + "]";

    std::map<std::string, std::string> headers;
    headers["apikey"] = SUPABASE_KEY;
    headers["Authorization"] = "Bearer " + SUPABASE_KEY;
    headers["Content-Type"] = "application/json";
    headers["Prefer"] = "return=minimal";
    headers["Content-Profile"] = "cactus";

    std::string url = SUPABASE_URL + "/rest/v1/logs";
    HttpClient::postJson(url, headers, payload);
#else
    (void)metrics;
#endif
}

inline void CactusTelemetry::recordEvent(const TelemetryMetrics& metrics) {
    if (!isEnabled()) {
        return;
    }

    std::thread([this, metrics]() {
        sendToSupabase(metrics);
    }).detach();
}

inline void CactusTelemetry::recordInit(const std::string& model, bool success,
                                   const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Init;
    metrics.model = model;
    metrics.success = success;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

inline void CactusTelemetry::recordCompletion(const std::string& model, bool success,
                                         double ttft_ms, double tps, double response_time_ms,
                                         int tokens, const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Completion;
    metrics.model = model;
    metrics.success = success;
    metrics.ttft_ms = ttft_ms;
    metrics.tps = tps;
    metrics.response_time_ms = response_time_ms;
    metrics.tokens = tokens;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

inline void CactusTelemetry::recordEmbedding(const std::string& model, bool success,
                                        const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Embedding;
    metrics.model = model;
    metrics.success = success;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

inline void CactusTelemetry::recordTranscription(const std::string& model, bool success,
                            double ttft_ms, double tps, double response_time_ms,
                            int tokens, const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Transcription;
    metrics.model = model;
    metrics.success = success;
    metrics.response_time_ms = response_time_ms;
    metrics.ttft_ms = ttft_ms;
    metrics.tps = tps;
    metrics.tokens = tokens;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

} // namespace ffi
} // namespace cactus

#ifdef __cplusplus
extern "C" {
#endif

inline void cactus_set_telemetry_enabled(int enabled) {
    cactus::ffi::CactusTelemetry::getInstance().setEnabled(enabled != 0);
}

inline void cactus_set_telemetry_token(const char* token) {
    if (token) {
        cactus::ffi::CactusTelemetry::getInstance().setTelemetryToken(token);
    }
}

#ifdef __cplusplus
}
#endif

#endif // CACTUS_TELEMETRY_H
