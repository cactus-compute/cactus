#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../cactus/cactus.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>

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

// Removed run_test method - tests handle their own runner calls

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

#endif 