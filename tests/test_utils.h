#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../cactus/cactus.h"
#include "../cactus/ffi/cactus_ffi.h"
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <cmath>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <fstream>
#include <unistd.h>
#endif

namespace TestUtils {
    
    size_t random_graph_input(CactusGraph& graph, const std::vector<size_t>& shape, Precision precision = Precision::INT8);
    bool verify_graph_outputs(CactusGraph& graph, size_t node_a, size_t node_b, float tolerance = 1e-6f);
    bool verify_graph_against_data(CactusGraph& graph, size_t node_id, const void* expected_data, size_t byte_size, float tolerance = 1e-6f);
    
    void fill_random_int8(std::vector<int8_t>& data);
    void fill_random_float(std::vector<float>& data);
    
    template<typename Func>
    double time_function(Func&& func, int iterations = 1);
    
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

    template<typename T>
    class TestFixture {
    public:
        TestFixture(const std::string& test_name = "");
        ~TestFixture() {
            graph_.hard_reset();
        }

        CactusGraph& graph() { return graph_; }

        size_t create_input(const std::vector<size_t>& shape, Precision precision = Precision::INT8);
        void set_input_data(size_t input_id, const std::vector<T>& data, Precision precision);
        void execute();
        T* get_output(size_t node_id);
        bool verify_output(size_t node_id, const std::vector<T>& expected, float tolerance = 1e-6f);

    private:
        CactusGraph graph_;
    };

    using Int8TestFixture = TestFixture<int8_t>;
    using FloatTestFixture = TestFixture<float>;

    template<typename T>
    bool compare_arrays(const T* actual, const T* expected, size_t count, float tolerance = 1e-6f);
    
    template<typename T>
    std::vector<T> create_test_data(size_t count);
    
    bool test_basic_operation(const std::string& op_name, 
                             std::function<size_t(CactusGraph&, size_t, size_t)> op_func,
                             const std::vector<int8_t>& data_a,
                             const std::vector<int8_t>& data_b,
                             const std::vector<int8_t>& expected,
                             const std::vector<size_t>& shape = {4});
                             
    bool test_scalar_operation(const std::string& op_name,
                              std::function<size_t(CactusGraph&, size_t, float)> op_func,
                              const std::vector<int8_t>& data,
                              float scalar,
                              const std::vector<int8_t>& expected,
                              const std::vector<size_t>& shape = {4});
}

template<typename Func>
double TestUtils::time_function(Func&& func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template<typename T>
TestUtils::TestFixture<T>::TestFixture(const std::string& test_name) {
    (void)test_name;
}

template<typename T>
size_t TestUtils::TestFixture<T>::create_input(const std::vector<size_t>& shape, Precision precision) {
    return graph_.input(shape, precision);
}

template<typename T>
void TestUtils::TestFixture<T>::set_input_data(size_t input_id, const std::vector<T>& data, Precision precision) {
    graph_.set_input(input_id, const_cast<void*>(static_cast<const void*>(data.data())), precision);
}

template<typename T>
void TestUtils::TestFixture<T>::execute() {
    graph_.execute();
}

template<typename T>
T* TestUtils::TestFixture<T>::get_output(size_t node_id) {
    return static_cast<T*>(graph_.get_output(node_id));
}

template<typename T>
bool TestUtils::TestFixture<T>::verify_output(size_t node_id, const std::vector<T>& expected, float tolerance) {
    T* output = get_output(node_id);
    return compare_arrays(output, expected.data(), expected.size(), tolerance);
}



template<typename T>
bool TestUtils::compare_arrays(const T* actual, const T* expected, size_t count, float tolerance) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(actual[i] - expected[i]) > tolerance) return false;
        } else {
            if (actual[i] != expected[i]) return false;
        }
    }
    return true;
}

template<typename T>
std::vector<T> TestUtils::create_test_data(size_t count) {
    std::vector<T> data(count);
    if constexpr (std::is_same_v<T, int8_t>) {
        fill_random_int8(data);
    } else if constexpr (std::is_same_v<T, float>) {
        fill_random_float(data);
    }
    return data;
}


namespace EngineTestUtils {

inline size_t get_memory_footprint_bytes() {
#ifdef __APPLE__
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count) == KERN_SUCCESS) {
        return vm_info.phys_footprint;
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage;
    }
#elif defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size, resident;
        statm >> size >> resident;
        return resident * sysconf(_SC_PAGESIZE);
    }
#endif
    return 0;
}

class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }

    void capture_baseline() {
        baseline_memory_ = get_memory_footprint_bytes();
        peak_memory_ = baseline_memory_;
    }

    double get_usage_mb() {
        size_t current = get_memory_footprint_bytes();
        if (current > peak_memory_) {
            peak_memory_ = current;
        }
        size_t model_mem = (current > baseline_memory_) ? (current - baseline_memory_) : 0;
        return model_mem / (1024.0 * 1024.0);
    }

    double get_peak_mb() {
        size_t model_peak = (peak_memory_ > baseline_memory_) ? (peak_memory_ - baseline_memory_) : 0;
        return model_peak / (1024.0 * 1024.0);
    }

private:
    MemoryTracker() : baseline_memory_(0), peak_memory_(0) {}
    size_t baseline_memory_;
    size_t peak_memory_;
};

inline void capture_memory_baseline() {
    MemoryTracker::instance().capture_baseline();
}

inline double get_memory_usage_mb() {
    return MemoryTracker::instance().get_usage_mb();
}

inline double get_peak_model_memory_mb() {
    return MemoryTracker::instance().get_peak_mb();
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

inline double json_number(const std::string& json, const std::string& key, double def = 0.0) {
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

inline std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t q1 = json.find('"', pos + pattern.size());
    if (q1 == std::string::npos) return {};
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

inline std::string escape_json(const std::string& s) {
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

struct StreamingData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count = 0;
    cactus_model_t model = nullptr;
    int stop_at = -1;
};

inline void stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token ? token : "");
    data->token_ids.push_back(token_id);
    data->token_count++;

    std::string out = token ? token : "";
    for (char& c : out) if (c == '\n') c = ' ';
    std::cout << out << std::flush;

    if (data->stop_at > 0 && data->token_count >= data->stop_at) {
        std::cout << " [→ stopped]" << std::flush;
        cactus_stop(data->model);
    }
}

struct Metrics {
    double ttft = 0.0;
    double tps = 0.0;
    double total_ms = 0.0;
    double prompt_tokens = 0.0;
    double completion_tokens = 0.0;

    void parse(const std::string& response) {
        ttft = json_number(response, "time_to_first_token_ms");
        tps = json_number(response, "tokens_per_second");
        total_ms = json_number(response, "total_time_ms");
        prompt_tokens = json_number(response, "prompt_tokens", json_number(response, "prefill_tokens"));
        completion_tokens = json_number(response, "completion_tokens", json_number(response, "decode_tokens"));
    }

    void print() const {
        std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2)
                  << ttft << " ms\n"
                  << "├─ Tokens per second: " << tps << std::endl;
    }

    void print_full() const {
        std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << ttft << " ms\n"
                  << "├─ Tokens per second:  " << tps << "\n"
                  << "├─ Total time:         " << total_ms << " ms\n"
                  << "├─ Prompt tokens:      " << prompt_tokens << "\n"
                  << "├─ Completion tokens:  " << completion_tokens << std::endl;
    }

    void print_perf(double ram_mb = 0.0) const {
        double prefill_tps = (prompt_tokens > 0 && ttft > 0) ? (prompt_tokens * 1000.0 / ttft) : 0.0;
        double ttft_sec = ttft / 1000.0;
        std::cout << "├─ TTFT: " << std::fixed << std::setprecision(2) << ttft_sec << " sec\n"
                  << "├─ Prefill: " << std::setprecision(1) << prefill_tps << " toks/sec\n"
                  << "├─ Decode: " << tps << " toks/sec\n"
                  << "└─ RAM: " << std::setprecision(1) << ram_mb << " MB" << std::endl;
    }
};

template<typename TestFunc>
bool run_test(const char* title, const char* model_path, const char* messages,
              const char* options, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(model_path, 2048, nullptr);
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
                                 options, tools, stream_callback, &data);

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);

    bool success = test_logic(result, data, response, metrics);

    std::cout << "└─ Status: " << (success ? "PASSED ✓" : "FAILED ✗") << std::endl;

    cactus_destroy(model);
    return success;
}

} // namespace EngineTestUtils

#endif