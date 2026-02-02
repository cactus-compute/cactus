#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../cactus/cactus.h"
#include "../cactus/ffi/cactus_ffi.h"
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <atomic>
#include <mutex>


namespace TestUtils {

size_t random_graph_input(CactusGraph& graph, const std::vector<size_t>& shape, Precision precision = Precision::INT8);
bool verify_graph_outputs(CactusGraph& graph, size_t node_a, size_t node_b, float tolerance = 1e-6f);
bool verify_graph_against_data(CactusGraph& graph, size_t node_id, const void* expected_data, size_t byte_size, float tolerance = 1e-6f);
void fill_random_int8(std::vector<int8_t>& data);
void fill_random_float(std::vector<float>& data);

template<typename Func>
double time_function(Func&& func, int iterations = 1) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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
constexpr Precision default_precision() {
    if constexpr (std::is_same_v<T, int8_t>) return Precision::INT8;
    else if constexpr (std::is_same_v<T, __fp16>) return Precision::FP16;
    else return Precision::FP32;
}

template<typename T>
constexpr float default_tolerance() {
    if constexpr (std::is_same_v<T, __fp16>) return 1e-2f;
    else return 1e-6f;
}

template<typename T>
bool compare_arrays(const T* actual, const T* expected, size_t count, float tolerance = default_tolerance<T>()) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, __fp16>) {
            if (std::abs(static_cast<float>(actual[i]) - static_cast<float>(expected[i])) > tolerance) return false;
        } else if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(actual[i] - expected[i]) > tolerance) return false;
        } else {
            if (actual[i] != expected[i]) return false;
        }
    }
    return true;
}

template<typename T>
class TestFixture {
public:
    TestFixture(const std::string& = "") {}
    ~TestFixture() { graph_.hard_reset(); }

    CactusGraph& graph() { return graph_; }

    size_t create_input(const std::vector<size_t>& shape, Precision precision = default_precision<T>()) {
        return graph_.input(shape, precision);
    }

    void set_input_data(size_t input_id, const std::vector<T>& data, Precision precision = default_precision<T>()) {
        graph_.set_input(input_id, const_cast<void*>(static_cast<const void*>(data.data())), precision);
    }

    void execute() { graph_.execute(); }

    T* get_output(size_t node_id) {
        return static_cast<T*>(graph_.get_output(node_id));
    }

    bool verify_output(size_t node_id, const std::vector<T>& expected, float tolerance = default_tolerance<T>()) {
        return compare_arrays(get_output(node_id), expected.data(), expected.size(), tolerance);
    }

private:
    CactusGraph graph_;
};

using Int8TestFixture = TestFixture<int8_t>;
using FP16TestFixture = TestFixture<__fp16>;

void fill_random_fp16(std::vector<__fp16>& data);

bool test_basic_operation(const std::string& op_name,
                          std::function<size_t(CactusGraph&, size_t, size_t)> op_func,
                          const std::vector<__fp16>& data_a,
                          const std::vector<__fp16>& data_b,
                          const std::vector<__fp16>& expected,
                          const std::vector<size_t>& shape = {4});

bool test_scalar_operation(const std::string& op_name,
                           std::function<size_t(CactusGraph&, size_t, float)> op_func,
                           const std::vector<__fp16>& data,
                           float scalar,
                           const std::vector<__fp16>& expected,
                           const std::vector<size_t>& shape = {4});

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

    void parse(const std::string& json);
    void print_json() const;
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
                                 options, tools, stream_callback, &data);

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);

    bool success = test_logic(result, data, response, metrics);
    std::cout << "└─ Status: " << (success ? "PASSED ✓" : "FAILED ✗") << std::endl;

    cactus_destroy(model);
    return success;
}

}

#ifdef HAVE_SDL2

#include <SDL.h>
#include <SDL_audio.h>

class AudioCapture {
public:
    AudioCapture(int len_ms = 10000);
    ~AudioCapture();

    bool init(int capture_id, int sample_rate);
    void resume();
    void pause();
    void clear();
    size_t get(int duration_ms, std::vector<float>& result);
    size_t get_all(std::vector<float>& result);
    bool is_running() const { return m_running; }
    size_t get_total_samples_received() const { return m_total_samples_received; }
    size_t get_buffer_length() const;

private:
    void callback(uint8_t* stream, int len);

    int m_len_ms;
    std::atomic<bool> m_running;
    SDL_AudioDeviceID m_dev_id_in;
    bool m_sdl_initialized;
    std::vector<float> m_audio;
    size_t m_audio_pos;
    size_t m_audio_len;
    std::atomic<size_t> m_total_samples_received;
    mutable std::mutex m_mutex;
};

#endif

#endif
