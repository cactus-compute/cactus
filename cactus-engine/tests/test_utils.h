#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../cactus_engine.h"
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>

namespace TestUtils {

class TestRunner {
public:
    TestRunner(const std::string& suite_name);
    void run_test(const std::string& test_name, bool result);
    void log_performance(const std::string& test_name, const std::string& details);
    void log_skip(const std::string& test_name, const std::string& reason);
    void print_summary();
    bool all_passed() const;

private:
    std::string suite_name_;
    int passed_count_;
    int total_count_;
};

}

namespace EngineTestUtils {

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer();
    double elapsed_ms() const;
};

double json_number(const std::string& json, const std::string& key, double def = 0.0);
std::string json_string(const std::string& json, const std::string& key);
std::string escape_json(const std::string& s);

struct StreamingData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count = 0;
    cactus_model_t model = nullptr;
    int stop_at = -1;
};

void stream_callback(const char* token, uint32_t token_id, void* user_data);

struct Metrics {
    bool success = false;
    std::string error;
    bool cloud_handoff = false;
    std::string response;
    std::string thinking;
    std::string function_calls;
    double confidence = -1.0;
    double ttft = 0.0;
    double total_ms = 0.0;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    double ram_mb = 0.0;
    double prefill_tokens = 0.0;
    double completion_tokens = 0.0;
    double total_tokens = 0.0;
    std::string segments;

    void parse(const std::string& json);
    void print_json() const;
};

struct PrefillMetrics {
    bool success = false;
    std::string error;
    double prefill_tokens = 0.0;
    double prefill_tps = 0.0;
    double total_ms = 0.0;
    double ram_mb = 0.0;

    void parse(const std::string& json);
    std::string line() const;
    void print_line() const;
};

template<typename TestFunc>
bool run_test(const char* title, const char* model_path, const char* messages,
              const char* options, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1,
              const char* user_prompt = nullptr) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (user_prompt) {
        std::cout << "├─ User prompt: " << user_prompt << "\n";
    }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    StreamingData data;
    data.model = model;
    data.stop_at = stop_at;

    char response[4096];
    std::cout << "Response: ";

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 options, tools, stream_callback, &data, nullptr, 0);

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);

    bool success = test_logic(result, data, response, metrics);
    std::cout << "└─ Status: " << (success ? "PASSED ✓" : "FAILED ✗") << std::endl;

    cactus_destroy(model);
    return success;
}

}

#endif
