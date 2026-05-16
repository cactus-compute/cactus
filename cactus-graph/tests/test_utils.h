#pragma once

#include "../cactus_graph.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <cstring>
#include <cstdio>

namespace TestUtils {

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class TestRunner {
public:
    explicit TestRunner(const std::string& suite_name) : suite_(suite_name), passed_(0), failed_(0) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
                  << "║ Running " << std::left << std::setw(73) << suite_name << "║\n"
                  << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";
    }

    void run_test(const std::string& name, bool result) {
        if (result) {
            std::cout << "✓ PASS │ " << std::left << std::setw(25) << name << std::endl;
            passed_++;
        } else {
            std::cout << "✗ FAIL │ " << std::left << std::setw(25) << name << std::endl;
            failed_++;
        }
    }

    void run_bench(const std::string&, bool result) {
        if (!result) failed_++;
    }

    void print_benchmarks_header() const {
        std::cout << "── benchmarks ──────────────────────────────────────────────────────────────────────────\n";
    }

    void print_summary() const {
        std::cout << "────────────────────────────────────────────────────────────────────────────────────────\n";
        if (failed_ == 0) {
            std::cout << "✓ All " << passed_ << " tests passed!\n";
        } else {
            std::cout << "✗ " << failed_ << " of " << (passed_ + failed_) << " tests failed!\n";
        }
    }

    bool all_passed() const { return failed_ == 0; }

private:
    std::string suite_;
    int passed_;
    int failed_;
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

inline void fill_random_fp16(std::vector<__fp16>& data) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : data) x = static_cast<__fp16>(dis(gen));
}

inline size_t runtime_id_from_serialized_index(uint32_t serialized_index) {
    return static_cast<size_t>(serialized_index);
}

inline bool test_basic_operation(const std::string& op_name,
                                 std::function<size_t(CactusGraph&, size_t, size_t)> op_func,
                                 const std::vector<__fp16>& data_a,
                                 const std::vector<__fp16>& data_b,
                                 const std::vector<__fp16>& expected,
                                 const std::vector<size_t>& shape = {4}) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::FP16);
    size_t input_b = graph.input(shape, Precision::FP16);
    size_t result_id = op_func(graph, input_a, input_b);
    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), Precision::FP16);
    graph.set_input(input_b, const_cast<void*>(static_cast<const void*>(data_b.data())), Precision::FP16);
    graph.execute();
    __fp16* output = static_cast<__fp16*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-2f) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

inline bool test_scalar_operation(const std::string& op_name,
                                  std::function<size_t(CactusGraph&, size_t, float)> op_func,
                                  const std::vector<__fp16>& data,
                                  float scalar,
                                  const std::vector<__fp16>& expected,
                                  const std::vector<size_t>& shape = {4}) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::FP16);
    size_t result_id = op_func(graph, input_a, scalar);
    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data.data())), Precision::FP16);
    graph.execute();
    __fp16* output = static_cast<__fp16*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-2f) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

} // namespace TestUtils
