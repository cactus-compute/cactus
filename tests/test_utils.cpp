#include "test_utils.h"
#include <random>
#include <cstring>
#include <algorithm>

namespace TestUtils {
    
    std::mt19937 gen(42);
    
    size_t random_graph_input(CactusGraph& graph, const std::vector<size_t>& shape, Precision precision) {
        size_t node_id = graph.input(shape, precision);
        
        size_t total_elements = 1;
        for (size_t dim : shape) {
            total_elements *= dim;
        }
        
        if (precision == Precision::INT8) {
            std::uniform_int_distribution<int> dist(-50, 50);
            std::vector<int8_t> data(total_elements);
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = static_cast<int8_t>(dist(gen));
            }
            graph.set_input(node_id, data.data(), precision);
        } else {
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
            std::vector<float> data(total_elements);
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = dist(gen);
            }
            graph.set_input(node_id, data.data(), precision);
        }
        
        return node_id;
    }
    
    bool verify_graph_outputs(CactusGraph& graph, size_t node_a, size_t node_b, float tolerance) {
        graph.execute();
        
        const auto& buffer_a = graph.get_output_buffer(node_a);
        const auto& buffer_b = graph.get_output_buffer(node_b);
        
        if (buffer_a.shape != buffer_b.shape || buffer_a.precision != buffer_b.precision) {
            return false;
        }
        
        void* data_a = graph.get_output(node_a);
        void* data_b = graph.get_output(node_b);
        
        size_t total_elements = 1;
        for (size_t dim : buffer_a.shape) {
            total_elements *= dim;
        }
        
        if (buffer_a.precision == Precision::INT8) {
            const int8_t* ptr_a = static_cast<const int8_t*>(data_a);
            const int8_t* ptr_b = static_cast<const int8_t*>(data_b);
            for (size_t i = 0; i < total_elements; ++i) {
                if (std::abs(ptr_a[i] - ptr_b[i]) > tolerance) {
                    return false;
                }
            }
        } else {
            const float* ptr_a = static_cast<const float*>(data_a);
            const float* ptr_b = static_cast<const float*>(data_b);
            for (size_t i = 0; i < total_elements; ++i) {
                if (std::abs(ptr_a[i] - ptr_b[i]) > tolerance) {
                    return false;
                }
            }
        }
        
        graph.hard_reset();
    return true;
    }
    
    bool verify_graph_against_data(CactusGraph& graph, size_t node_id, const void* expected_data, size_t byte_size, float tolerance) {
        graph.execute();
        
        void* actual_data = graph.get_output(node_id);
        const auto& buffer = graph.get_output_buffer(node_id);
        
        if (buffer.precision == Precision::INT8) {
            const int8_t* actual = static_cast<const int8_t*>(actual_data);
            const int8_t* expected = static_cast<const int8_t*>(expected_data);
            size_t count = byte_size;
            for (size_t i = 0; i < count; ++i) {
                if (std::abs(actual[i] - expected[i]) > tolerance) {
                    return false;
                }
            }
        } else {
            const float* actual = static_cast<const float*>(actual_data);
            const float* expected = static_cast<const float*>(expected_data);
            size_t count = byte_size / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                if (std::abs(actual[i] - expected[i]) > tolerance) {
                    return false;
                }
            }
        }
        
        graph.hard_reset();
    return true;
    }
    
    void fill_random_int8(std::vector<int8_t>& data) {
        std::uniform_int_distribution<int> dist(-50, 50);
        for (auto& val : data) {
            val = static_cast<int8_t>(dist(gen));
        }
    }
    
    void fill_random_float(std::vector<float>& data) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& val : data) {
            val = dist(gen);
        }
    }
    
    TestRunner::TestRunner(const std::string& suite_name) 
        : suite_name_(suite_name), passed_count_(0), total_count_(0) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Running " << std::left << std::setw(73) << suite_name_ << " ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";
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
        std::cout << "⚡PERF │ " << std::left << std::setw(25) << test_name << " │ " << details << "\n";
    }
    
    void TestRunner::log_skip(const std::string& test_name, const std::string& reason) {
        std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << test_name << " │ " << reason << "\n";
    }
    
    void TestRunner::print_summary() {
        std::cout << "────────────────────────────────────────────────────────────────────────────────────────\n";
        if (all_passed()) {
            std::cout << "✓ All " << total_count_ << " tests passed!\n";
        } else {
            std::cout << "✗ " << (total_count_ - passed_count_) << " of " << total_count_ << " tests failed!\n";
        }
        std::cout << "\n";
    }
    
    bool TestRunner::all_passed() const {
        return passed_count_ == total_count_;
    }
    
    bool test_basic_operation(const std::string& op_name,
                             std::function<size_t(CactusGraph&, size_t, size_t)> op_func,
                             const std::vector<int8_t>& data_a,
                             const std::vector<int8_t>& data_b,
                             const std::vector<int8_t>& expected,
                             const std::vector<size_t>& shape) {
        (void)op_name;  
        CactusGraph graph;
        
        size_t input_a = graph.input(shape, Precision::INT8);
        size_t input_b = graph.input(shape, Precision::INT8);
        size_t result_id = op_func(graph, input_a, input_b);
        
        graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), Precision::INT8);
        graph.set_input(input_b, const_cast<void*>(static_cast<const void*>(data_b.data())), Precision::INT8);
        
        graph.execute();
        
        int8_t* output = static_cast<int8_t*>(graph.get_output(result_id));
        
        for (size_t i = 0; i < expected.size(); ++i) {
            if (output[i] != expected[i]) {
                graph.hard_reset();
                return false;
            }
        }
        
        graph.hard_reset();
        return true;
    }
    
    bool test_scalar_operation(const std::string& op_name,
                              std::function<size_t(CactusGraph&, size_t, float)> op_func,
                              const std::vector<int8_t>& data,
                              float scalar,
                              const std::vector<int8_t>& expected,
                              const std::vector<size_t>& shape) {
        (void)op_name;  
        CactusGraph graph;
        
        size_t input_a = graph.input(shape, Precision::INT8);
        size_t result_id = op_func(graph, input_a, scalar);
        
        graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data.data())), Precision::INT8);
        
        graph.execute();
        
        int8_t* output = static_cast<int8_t*>(graph.get_output(result_id));
        
        for (size_t i = 0; i < expected.size(); ++i) {
            if (output[i] != expected[i]) {
                graph.hard_reset();
                return false;
            }
        }
        
        graph.hard_reset();
        return true;
    }
} 