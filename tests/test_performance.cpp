#include "test_utils.h"
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include <cstdio>
#include <random>
#include <filesystem>
#include <fstream>

struct BenchmarkConfig {
    std::vector<size_t> dimensions = {1024};
    std::vector<Precision> precisions = {Precision::INT8, Precision::FP32};
    std::vector<ComputeBackend> backends = {ComputeBackend::CPU};
    int iterations = 1;
    
    BenchmarkConfig() {
    }
};

template<typename T>
double time_operation(std::function<void()> operation, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        operation();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; 
}

template<typename T>
void setup_random_data(std::vector<T>& data) {
    if constexpr (std::is_same_v<T, int8_t>) {
        TestUtils::fill_random_int8(data);
    } else {
        TestUtils::fill_random_float(data);
    }
}

std::string precision_to_string(Precision prec) {
    return (prec == Precision::INT8) ? "INT8" : "FP32";
}

std::string backend_to_string(ComputeBackend backend) {
    return (backend == ComputeBackend::CPU) ? "CPU" : "NPU";
}

double calculate_gflops(size_t ops, double time_ms) {
    return ops / (time_ms * 1e6);
}

template<typename T>
void benchmark_binary_elementwise_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t, size_t)>>> ops = {
        {"Add", [](CactusGraph& b, size_t a, size_t c) { return b.add(a, c); }},
        {"Subtract", [](CactusGraph& b, size_t a, size_t c) { return b.subtract(a, c); }},
        {"Multiply", [](CactusGraph& b, size_t a, size_t c) { return b.multiply(a, c); }},
        {"Divide", [](CactusGraph& b, size_t a, size_t c) { return b.divide(a, c); }}
    };
    
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (const auto& [op_name, op_func] : ops) {
        for (size_t dim : config.dimensions) {
            size_t total_elements = dim * dim;
            
            TestUtils::TestFixture<T> fixture(op_name);
            size_t input_a = fixture.create_input({dim, dim}, precision);
            size_t input_b = fixture.create_input({dim, dim}, precision);
            
            std::vector<T> data_a(total_elements), data_b(total_elements);
            setup_random_data(data_a);
            setup_random_data(data_b);
            
            fixture.set_input_data(input_a, data_a, precision);
            fixture.set_input_data(input_b, data_b, precision);
            
            op_func(fixture.graph(), input_a, input_b);

            double time_ms = time_operation<T>([&]() {
                fixture.execute();
            }, config.iterations);
            
            double gflops = calculate_gflops(total_elements, time_ms);
            
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                                 std::to_string(time_ms) + "ms, " + std::to_string(gflops) + " GFLOPS");
        }
    }
}

template<typename T>
void benchmark_scalar_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t)>>> ops = {
        {"ScalarAdd", [](CactusGraph& b, size_t a) { return b.scalar_add(a, 2.5f); }},
        {"ScalarSubtract", [](CactusGraph& b, size_t a) { return b.scalar_subtract(a, 2.5f); }},
        {"ScalarMultiply", [](CactusGraph& b, size_t a) { return b.scalar_multiply(a, 2.5f); }},
        {"ScalarDivide", [](CactusGraph& b, size_t a) { return b.scalar_divide(a, 2.5f); }},
        {"ScalarExp", [](CactusGraph& b, size_t a) { return b.scalar_exp(a); }},
        {"ScalarSqrt", [](CactusGraph& b, size_t a) { return b.scalar_sqrt(a); }},
        {"ScalarCos", [](CactusGraph& b, size_t a) { return b.scalar_cos(a); }},
        {"ScalarSin", [](CactusGraph& b, size_t a) { return b.scalar_sin(a); }}
    };
    
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (const auto& [op_name, op_func] : ops) {
        for (size_t dim : config.dimensions) {
            size_t total_elements = dim * dim;
            
            TestUtils::TestFixture<T> fixture(op_name);
            size_t input_a = fixture.create_input({dim, dim}, precision);
            
            std::vector<T> data_a(total_elements);
            setup_random_data(data_a);
            fixture.set_input_data(input_a, data_a, precision);
            
            op_func(fixture.graph(), input_a);

            double time_ms = time_operation<T>([&]() {
                fixture.execute();
            }, config.iterations);
            
            double gflops = calculate_gflops(total_elements, time_ms);
            
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                                 std::to_string(time_ms) + "ms, " + std::to_string(gflops) + " GFLOPS");
        }
    }
}

template<typename T>
void benchmark_matmul_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (ComputeBackend backend : config.backends) {
        std::string backend_str = backend_to_string(backend);
        
        for (size_t dim : config.dimensions) {
            try {
                TestUtils::TestFixture<T> fixture("MatMul");
                size_t input_a = fixture.create_input({dim, dim}, precision);
                size_t input_b = fixture.create_input({dim, dim}, precision);
                
                std::vector<T> data_a(dim * dim), data_b(dim * dim);
                setup_random_data(data_a);
                setup_random_data(data_b);
                
                fixture.set_input_data(input_a, data_a, precision);
                fixture.set_input_data(input_b, data_b, precision);
                
                fixture.graph().matmul(input_a, input_b, false, backend);

                double time_ms = time_operation<T>([&]() {
                    fixture.execute();
                }, config.iterations);
                
                double gflops = calculate_gflops(2ULL * dim * dim * dim, time_ms);
                
                runner.log_performance("MatMul " + std::to_string(dim) + "x" + std::to_string(dim) + "x" + std::to_string(dim) + " " + backend_str + " " + prec_str,
                                     std::to_string(time_ms) + "ms, " + std::to_string(gflops) + " GFLOPS");
            } catch (const std::exception& e) {
                runner.log_performance("MatMul " + std::to_string(dim) + "x" + std::to_string(dim) + "x" + std::to_string(dim) + " " + backend_str + " " + prec_str,
                                     "SKIP: " + std::string(e.what()));
            }
        }
    }
}

template<typename T>
void benchmark_unary_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t)>>> ops = {
        {"Transpose", [](CactusGraph& b, size_t a) { return b.transpose(a); }}
    };
    
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (const auto& [op_name, op_func] : ops) {
        for (size_t dim : config.dimensions) {
            size_t total_elements = dim * dim;
            
            TestUtils::TestFixture<T> fixture(op_name);
            size_t input_a = fixture.create_input({dim, dim}, precision);
            
            std::vector<T> data_a(total_elements);
            setup_random_data(data_a);
            fixture.set_input_data(input_a, data_a, precision);
            
            op_func(fixture.graph(), input_a);

            double time_ms = time_operation<T>([&]() {
                fixture.execute();
            }, config.iterations);
            
            double gb_per_sec = (total_elements * PrecisionTraits::size_of(precision) * 2) / (time_ms * 1e6);
            
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                                 std::to_string(time_ms) + "ms, " + std::to_string(gb_per_sec) + " GB/s");
        }
    }
}

template<typename T>
void benchmark_reduction_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t)>>> ops = {
        {"Sum", [](CactusGraph& b, size_t a) { return b.sum(a, -1); }},
        {"Mean", [](CactusGraph& b, size_t a) { return b.mean(a, -1); }},
        {"Variance", [](CactusGraph& b, size_t a) { return b.variance(a, -1); }},
        {"Min", [](CactusGraph& b, size_t a) { return b.min(a, -1); }},
        {"Max", [](CactusGraph& b, size_t a) { return b.max(a, -1); }}
    };
    
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (const auto& [op_name, op_func] : ops) {
        for (size_t dim : config.dimensions) {
            size_t total_elements = dim * dim;
            
            TestUtils::TestFixture<T> fixture(op_name);
            size_t input_a = fixture.create_input({dim, dim}, precision);
            
            std::vector<T> data_a(total_elements);
            setup_random_data(data_a);
            fixture.set_input_data(input_a, data_a, precision);
            
            op_func(fixture.graph(), input_a);

            double time_ms = time_operation<T>([&]() {
                fixture.execute();
            }, config.iterations);
            
            double gb_per_sec = (total_elements * PrecisionTraits::size_of(precision)) / (time_ms * 1e6);
            
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                                 std::to_string(time_ms) + "ms, " + std::to_string(gb_per_sec) + " GB/s");
        }
    }
}

template<typename T>
void benchmark_advanced_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (size_t dim : config.dimensions) {
        size_t total_elements = dim * dim;
        
        TestUtils::TestFixture<T> fixture("Softmax");
        size_t input_a = fixture.create_input({dim, dim}, precision);
        
        std::vector<T> data_a(total_elements);
        setup_random_data(data_a);
        fixture.set_input_data(input_a, data_a, precision);
        
        fixture.graph().softmax(input_a, -1);

        double time_ms = time_operation<T>([&]() {
            fixture.execute();
        }, config.iterations);
        
        double gb_per_sec = (total_elements * PrecisionTraits::size_of(precision)) / (time_ms * 1e6);
        
        runner.log_performance("Softmax " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                             std::to_string(time_ms) + "ms, " + std::to_string(gb_per_sec) + " GB/s");
    }
}

template<typename T>
void benchmark_rms_norm(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (size_t dim : config.dimensions) {
        size_t total_elements = dim * dim;
        
        CactusGraph graph;
        size_t input_a = graph.input({dim, dim}, precision);
        size_t weight = graph.input({dim}, Precision::FP32);  
        
        std::vector<T> data_a(total_elements);
        std::vector<float> weight_data(dim, 1.0f);
        setup_random_data(data_a);
        
        graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), precision);
        graph.set_input(weight, const_cast<void*>(static_cast<const void*>(weight_data.data())), Precision::FP32);
        
        graph.rms_norm(input_a, weight);

        double time_ms = time_operation<T>([&]() {
            graph.execute();
        }, config.iterations);
        
        double gb_per_sec = (total_elements * PrecisionTraits::size_of(precision)) / (time_ms * 1e6);
        
        runner.log_performance("RMSNorm " + std::to_string(dim) + "x" + std::to_string(dim) + " " + prec_str,
                             std::to_string(time_ms) + "ms, " + std::to_string(gb_per_sec) + " GB/s");
        
        graph.hard_reset();
    }
}

template<typename T>
void benchmark_rope(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    if constexpr (!std::is_same_v<T, float>) {
    }
    
    std::string prec_str = "FP32";
    
    for (size_t dim : config.dimensions) {
        size_t batch_size = 1;
        size_t seq_len = dim / 4;
        size_t num_heads = 4;
        size_t head_dim = dim / 4;
        size_t total_elements = batch_size * seq_len * num_heads * head_dim;
        
        TestUtils::FloatTestFixture fixture("RoPE");
        size_t input_a = fixture.create_input({batch_size, seq_len, num_heads, head_dim}, Precision::FP32);
        
        std::vector<float> data_a(total_elements);
        setup_random_data(data_a);
        fixture.set_input_data(input_a, data_a, Precision::FP32);
        
        fixture.graph().rope(input_a, 10000.0f);

        double time_ms = time_operation<float>([&]() {
            fixture.execute();
        }, config.iterations);
        
        double gb_per_sec = (total_elements * 4 * 2) / (time_ms * 1e6); 
        
        runner.log_performance("RoPE " + std::to_string(seq_len) + "x" + std::to_string(num_heads) + "x" + std::to_string(head_dim) + " " + prec_str,
                             std::to_string(time_ms) + "ms, " + std::to_string(gb_per_sec) + " GB/s");
    }
}

template<typename T>
void benchmark_attention(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
    std::string prec_str = precision_to_string(precision);
    
    for (size_t dim : config.dimensions) {
        size_t batch_size = 1;
        size_t seq_len = std::min(dim / 8, 64UL); 
        size_t num_heads = 8;
        size_t head_dim = dim / 8;
        size_t total_elements = batch_size * seq_len * num_heads * head_dim;
        
        CactusGraph graph;
        size_t query = graph.input({batch_size, seq_len, num_heads, head_dim}, precision);
        size_t key = graph.input({batch_size, seq_len, num_heads, head_dim}, precision);
        size_t value = graph.input({batch_size, seq_len, num_heads, head_dim}, precision);
        
        std::vector<T> q_data(total_elements), k_data(total_elements), v_data(total_elements);
        setup_random_data(q_data);
        setup_random_data(k_data);
        setup_random_data(v_data);
        
        graph.set_input(query, const_cast<void*>(static_cast<const void*>(q_data.data())), precision);
        graph.set_input(key, const_cast<void*>(static_cast<const void*>(k_data.data())), precision);
        graph.set_input(value, const_cast<void*>(static_cast<const void*>(v_data.data())), precision);
        
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        graph.attention(query, key, value, scale);
        
        double time_ms = time_operation<T>([&]() {
            graph.execute();
        }, config.iterations);
        
        double gflops = calculate_gflops(2ULL * batch_size * num_heads * seq_len * seq_len * head_dim, time_ms);
        
        runner.log_performance("Attention " + std::to_string(seq_len) + "x" + std::to_string(num_heads) + "x" + std::to_string(head_dim) + " " + prec_str,
                             std::to_string(time_ms) + "ms, " + std::to_string(gflops) + " GFLOPS");
        
        graph.hard_reset();
    }
}


template<typename T>
void benchmark_embedding_ops(TestUtils::TestRunner& runner, BenchmarkConfig& config) {
    std::vector<size_t> vocab_sizes = {127};
    std::vector<size_t> embedding_dims = {128};
    std::vector<size_t> sequence_lengths = {64};
    
    std::string precision_str = precision_to_string(std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32);
    
    for (size_t vocab_size : vocab_sizes) {
        for (size_t embedding_dim : embedding_dims) {
            for (size_t seq_len : sequence_lengths) {
                CactusGraph graph;
                
                std::vector<T> embeddings_data(vocab_size * embedding_dim);
                setup_random_data(embeddings_data);
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(0, vocab_size - 1);
                
                std::vector<int8_t> indices_data(seq_len);
                for (auto& idx : indices_data) {
                    int val = dis(gen);
                    idx = static_cast<int8_t>(std::min(val, 127));
                }
                
                Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
                
                size_t embeddings_id = graph.input({vocab_size, embedding_dim}, precision);
                size_t indices_id = graph.input({seq_len}, Precision::INT8);
                graph.embedding(embeddings_id, indices_id);

                graph.set_input(embeddings_id, embeddings_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);
                
                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);
                
                double throughput = (seq_len * embedding_dim * sizeof(T)) / (time_ms * 1e3);
                
                runner.log_performance(
                    "Embedding " + std::to_string(vocab_size) + " vocab x" + 
                    std::to_string(embedding_dim) + " dim, seq=" + std::to_string(seq_len) + " " + precision_str,
                    std::to_string(time_ms) + "ms, " + std::to_string(throughput) + " GB/s"
                );
            }
        }
    }
}

void benchmark_mmap_embedding(TestUtils::TestRunner& runner, BenchmarkConfig& config) {
    std::vector<size_t> vocab_sizes = {100};
    std::vector<size_t> embedding_dims = {64};
    std::vector<size_t> sequence_lengths = {32};
    
    for (size_t vocab_size : vocab_sizes) {
        for (size_t embedding_dim : embedding_dims) {
            for (size_t seq_len : sequence_lengths) {
                CactusGraph graph;
                
                std::vector<float> embeddings_data(vocab_size * embedding_dim);
                setup_random_data(embeddings_data);
                
                size_t temp_embeddings = graph.input({vocab_size, embedding_dim}, Precision::FP32);
                graph.set_input(temp_embeddings, embeddings_data.data(), Precision::FP32);
                
                const std::string temp_file = "/tmp/perf_embeddings_" + 
                    std::to_string(vocab_size) + "_" + std::to_string(embedding_dim) + ".bin";
                
                GraphFile::save_node(graph, temp_embeddings, temp_file);
                graph.hard_reset();
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(0, vocab_size - 1);
                
                std::vector<int8_t> indices_data(seq_len);
                for (auto& idx : indices_data) {
                    int val = dis(gen);
                    idx = static_cast<int8_t>(std::min(val, 127));
                }
                
                size_t indices_id = graph.input({seq_len}, Precision::INT8);
                graph.embedding(temp_file, indices_id);

                graph.set_input(indices_id, indices_data.data(), Precision::INT8);
                
                double time_ms = time_operation<float>([&]() {
                    graph.execute();
                }, config.iterations);
                
                double throughput = (seq_len * embedding_dim * sizeof(float)) / (time_ms * 1e3);
                
                runner.log_performance(
                    "MMap Embedding " + std::to_string(vocab_size) + " vocab x" + 
                    std::to_string(embedding_dim) + " dim, seq=" + std::to_string(seq_len) + " FP32",
                    std::to_string(time_ms) + "ms, " + std::to_string(throughput) + " GB/s"
                );
                
                std::remove(temp_file.c_str());
            }
        }
    }
}

template<typename T>
void benchmark_gather_ops(TestUtils::TestRunner& runner, BenchmarkConfig& config) {
    std::string precision_str = precision_to_string(std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32);
    
    {
        std::vector<size_t> tensor_sizes = {127};
        std::vector<size_t> index_counts = {132};
        
        for (size_t tensor_size : tensor_sizes) {
            for (size_t index_count : index_counts) {
                CactusGraph graph;
                
                std::vector<T> tensor_data(tensor_size);
                setup_random_data(tensor_data);
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(0, tensor_size - 1);
                
                std::vector<int8_t> indices_data(index_count);
                for (auto& idx : indices_data) {
                    int val = dis(gen);
                    idx = static_cast<int8_t>(std::min(val, 127));
                }
                
                Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
                
                size_t tensor_id = graph.input({tensor_size}, precision);
                size_t indices_id = graph.input({index_count}, Precision::INT8);
                graph.gather(tensor_id, indices_id);

                graph.set_input(tensor_id, tensor_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);
                
                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);
                
                double throughput = (index_count * sizeof(T)) / (time_ms * 1e3);
                
                runner.log_performance(
                    "Gather 1D " + std::to_string(tensor_size) + " → " + std::to_string(index_count) + " " + precision_str,
                    std::to_string(time_ms) + "ms, " + std::to_string(throughput) + " GB/s"
                );
            }
        }
    }
    
    {
        std::vector<std::vector<size_t>> tensor_shapes = {{64, 16, 8}};
        std::vector<size_t> index_counts = {12};
        
        for (const auto& shape : tensor_shapes) {
            for (size_t index_count : index_counts) {
                CactusGraph graph;
                
                size_t total_elements = shape[0] * shape[1] * shape[2];
                std::vector<T> tensor_data(total_elements);
                setup_random_data(tensor_data);
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(0, shape[0] - 1);
                
                std::vector<int8_t> indices_data(index_count);
                for (auto& idx : indices_data) {
                    int val = dis(gen);
                    idx = static_cast<int8_t>(std::min(val, 127));
                }
                
                Precision precision = std::is_same_v<T, int8_t> ? Precision::INT8 : Precision::FP32;
                
                size_t tensor_id = graph.input(shape, precision);
                size_t indices_id = graph.input({index_count}, Precision::INT8);
                graph.gather(tensor_id, indices_id);

                graph.set_input(tensor_id, tensor_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);
                
                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);
                
                double throughput = (index_count * shape[1] * shape[2] * sizeof(T)) / (time_ms * 1e3);
                
                runner.log_performance(
                    "Gather 3D " + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + "x" + std::to_string(shape[2]) + 
                    " → " + std::to_string(index_count) + " " + precision_str,
                    std::to_string(time_ms) + "ms, " + std::to_string(throughput) + " GB/s"
                );
            }
        }
    }
}

bool test_binary_elementwise_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_binary_elementwise_ops<int8_t>(runner, config);
    benchmark_binary_elementwise_ops<float>(runner, config);
    
    return true;
}

bool test_scalar_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_scalar_ops<int8_t>(runner, config);
    benchmark_scalar_ops<float>(runner, config);
    
    return true;
}

bool test_matrix_multiplication_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_matmul_ops<int8_t>(runner, config);
    benchmark_matmul_ops<float>(runner, config);
    
    return true;
}

bool test_unary_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_unary_ops<int8_t>(runner, config);
    benchmark_unary_ops<float>(runner, config);
    
    return true;
}

bool test_reduction_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_reduction_ops<int8_t>(runner, config);
    benchmark_reduction_ops<float>(runner, config);
    
    return true;
}

bool test_advanced_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_advanced_ops<int8_t>(runner, config);
    benchmark_advanced_ops<float>(runner, config);
    
    return true;
}

bool test_engine_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    
    benchmark_rms_norm<float>(runner, config);
    benchmark_rope<int8_t>(runner, config);
    benchmark_rope<float>(runner, config);
    benchmark_attention<int8_t>(runner, config);
    benchmark_attention<float>(runner, config);
    
    return true;
}

bool test_gather_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    config.iterations = 10;
    
    benchmark_gather_ops<int8_t>(runner, config);
    benchmark_gather_ops<float>(runner, config);
    
    return true;
}

bool test_embedding_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    config.iterations = 10;
    
    benchmark_embedding_ops<int8_t>(runner, config);
    benchmark_embedding_ops<float>(runner, config);
    benchmark_mmap_embedding(runner, config);
    
    return true;
}


int main() {
    TestUtils::TestRunner runner("Performance Benchmarks");
    
    runner.run_test("Binary Element-wise Operations", test_binary_elementwise_performance(runner));
    runner.run_test("Scalar Operations", test_scalar_operations_performance(runner));
    runner.run_test("Matrix Multiplication", test_matrix_multiplication_performance(runner));
    runner.run_test("Unary Operations", test_unary_operations_performance(runner));
    runner.run_test("Reduction Operations", test_reduction_operations_performance(runner));
    runner.run_test("Advanced Operations", test_advanced_operations_performance(runner));
    runner.run_test("Engine Operations", test_engine_operations_performance(runner));
    runner.run_test("Gather Operations", test_gather_operations_performance(runner));
    runner.run_test("Embedding Operations", test_embedding_operations_performance(runner));
    
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}