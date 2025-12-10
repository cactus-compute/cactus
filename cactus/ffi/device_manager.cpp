#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <map>

#ifdef __APPLE__
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace cactus {
namespace ffi {

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
    static std::map<std::string, std::string> getDeviceMetadata();
    static std::string registerDevice();
    static std::string generateUUID();

private:
    static std::string getConfigPath();
    static std::map<std::string, std::string> readConfig();
    static void writeConfig(const std::map<std::string, std::string>& config);
};

std::string DeviceManager::getConfigPath() {
    const char* home = getenv("HOME");
    if (!home) {
        home = "/tmp";
    }

    std::string cactus_dir = std::string(home) + "/.cactus";

    mkdir(cactus_dir.c_str(), 0755);

    return cactus_dir + "/telemetry_config.json";
}

std::map<std::string, std::string> DeviceManager::readConfig() {
    std::map<std::string, std::string> config;
    std::string path = getConfigPath();
    std::ifstream file(path);

    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string content = buffer.str();

        size_t device_pos = content.find("\"device_id\":\"");
        if (device_pos != std::string::npos) {
            size_t start = device_pos + 13;
            size_t end = content.find("\"", start);
            if (end != std::string::npos) {
                config["device_id"] = content.substr(start, end - start);
            }
        }

        size_t project_pos = content.find("\"project_id\":\"");
        if (project_pos != std::string::npos) {
            size_t start = project_pos + 14;
            size_t end = content.find("\"", start);
            if (end != std::string::npos) {
                config["project_id"] = content.substr(start, end - start);
            }
        }
    }

    return config;
}

void DeviceManager::writeConfig(const std::map<std::string, std::string>& config) {
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

std::string DeviceManager::generateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;

    for (int i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";

    for (int i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";

    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";

    ss << dis2(gen);

    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";

    for (int i = 0; i < 12; i++) {
        ss << dis(gen);
    }

    return ss.str();
}

std::string DeviceManager::getDeviceId() {
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

std::string DeviceManager::getProjectId() {
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

std::string DeviceManager::registerDevice() {
#ifdef __APPLE__
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

        size_t id_pos = response.body.find("\"id\":\"");
        if (id_pos != std::string::npos) {
            size_t start = id_pos + 6;
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

std::map<std::string, std::string> DeviceManager::getDeviceMetadata() {
    std::map<std::string, std::string> metadata;

#ifdef __APPLE__
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

} // namespace ffi
} // namespace cactus
