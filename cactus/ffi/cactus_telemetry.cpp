#include <thread>
#include <sstream>
#include <iostream>
#include <chrono>
#include <string>
#include <mutex>
#include <map>
#include <iomanip>
#include <ctime>

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

    int audio_duration_ms = 0;

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
};

class DeviceManager {
public:
    static std::string getDeviceId();
    static std::string getProjectId();
};

class LogRecord {
public:
    static std::string buildJson(
        const TelemetryMetrics& metrics,
        const std::string& project_id,
        const std::string& device_id,
        const std::string& telemetry_token
    );

private:
    static std::string escapeJson(const std::string& input) {
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

    static std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp) {
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

    static std::string eventTypeToString(TelemetryEventType type) {
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
};

std::string LogRecord::buildJson(
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
    json << "\"framework_version\":\"1.0.0\"";

    if (metrics.event_type == TelemetryEventType::Completion) {
        json << ",\"ttft\":" << metrics.ttft_ms;
        json << ",\"tps\":" << metrics.tps;
        json << ",\"response_time\":" << metrics.response_time_ms;
        json << ",\"tokens\":" << metrics.tokens;
    }

    if (metrics.event_type == TelemetryEventType::Transcription) {
        json << ",\"response_time\":" << metrics.response_time_ms;
        json << ",\"audio_duration\":" << metrics.audio_duration_ms;
    }

    if (!metrics.message.empty()) {
        json << ",\"message\":\"" << escapeJson(metrics.message) << "\"";
    }

    json << "}";
    return json.str();
}

class TelemetryCollector {
public:
    static TelemetryCollector& getInstance();

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
                           double response_time_ms, int audio_duration_ms,
                           const std::string& message = "");

    bool isEnabled() const;

private:
    TelemetryCollector();
    ~TelemetryCollector() = default;

    TelemetryCollector(const TelemetryCollector&) = delete;
    TelemetryCollector& operator=(const TelemetryCollector&) = delete;

    void sendToSupabase(const TelemetryMetrics& metrics);

    bool enabled_ = false;
    std::string telemetry_token_;
    std::string project_id_;
    std::string device_id_;

    mutable std::mutex mutex_;
};

static const std::string SUPABASE_URL = "https://vlqqczxwyaodtcdmdmlw.supabase.co";
static const std::string SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscXFjenh3eWFvZHRjZG1kbWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MTg2MzIsImV4cCI6MjA2NzA5NDYzMn0.nBzqGuK9j6RZ6mOPWU2boAC_5H9XDs-fPpo5P3WZYbI";

TelemetryCollector& TelemetryCollector::getInstance() {
    static TelemetryCollector instance;
    return instance;
}

TelemetryCollector::TelemetryCollector() {
#ifdef __APPLE__
    device_id_ = DeviceManager::getDeviceId();
    project_id_ = DeviceManager::getProjectId();
#endif
}

void TelemetryCollector::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

void TelemetryCollector::setTelemetryToken(const std::string& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_token_ = token;
}

void TelemetryCollector::setProjectId(const std::string& project_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    project_id_ = project_id;
}

bool TelemetryCollector::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && !telemetry_token_.empty();
}

void TelemetryCollector::sendToSupabase(const TelemetryMetrics& metrics) {
#ifdef __APPLE__
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
    auto response = HttpClient::postJson(url, headers, payload);

    if (response.success) {
        std::cerr << "[Telemetry] Log sent successfully" << std::endl;
    }
#endif
}

void TelemetryCollector::recordEvent(const TelemetryMetrics& metrics) {
    if (!isEnabled()) {
        return;
    }

    std::thread([this, metrics]() {
        sendToSupabase(metrics);
    }).detach();
}

void TelemetryCollector::recordInit(const std::string& model, bool success,
                                   const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Init;
    metrics.model = model;
    metrics.success = success;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

void TelemetryCollector::recordCompletion(const std::string& model, bool success,
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

void TelemetryCollector::recordEmbedding(const std::string& model, bool success,
                                        const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Embedding;
    metrics.model = model;
    metrics.success = success;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

void TelemetryCollector::recordTranscription(const std::string& model, bool success,
                                            double response_time_ms, int audio_duration_ms,
                                            const std::string& message) {
    TelemetryMetrics metrics;
    metrics.event_type = TelemetryEventType::Transcription;
    metrics.model = model;
    metrics.success = success;
    metrics.response_time_ms = response_time_ms;
    metrics.audio_duration_ms = audio_duration_ms;
    metrics.message = message;
    metrics.timestamp = std::chrono::system_clock::now();

    recordEvent(metrics);
}

} // namespace ffi
} // namespace cactus

extern "C" {

void cactus_set_telemetry_enabled(int enabled) {
    cactus::ffi::TelemetryCollector::getInstance().setEnabled(enabled != 0);
}

void cactus_set_telemetry_token(const char* token) {
    if (token) {
        cactus::ffi::TelemetryCollector::getInstance().setTelemetryToken(token);
    }
}
}
