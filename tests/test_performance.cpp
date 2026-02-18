#include "test_utils.h"
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cstdio>
#include <random>
#include <filesystem>
#include <fstream>

struct BenchmarkConfig {
    std::vector<size_t> dimensions = {1024};
    std::vector<Precision> precisions = {Precision::FP16};
    std::vector<ComputeBackend> backends = {ComputeBackend::CPU};
    int iterations = 10;

    BenchmarkConfig() {
    }
};

template<typename T>
double time_operation(std::function<void()> operation, int iterations) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        operation();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        operation();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / (1000.0 * iterations);
}

template<typename T>
void setup_random_data(std::vector<T>& data) {
    if constexpr (std::is_same_v<T, int8_t>) {
        TestUtils::fill_random_int8(data);
    } else if constexpr (std::is_same_v<T, __fp16>) {
        TestUtils::fill_random_fp16(data);
    } else {
        TestUtils::fill_random_float(data);
    }
}

std::string precision_to_string(Precision prec) {
    switch (prec) {
        case Precision::INT8: return "INT8";
        case Precision::FP16: return "FP16";
        case Precision::FP32: return "FP32";
        default: return "UNKNOWN";
    }
}

std::string backend_to_string(ComputeBackend backend) {
    return (backend == ComputeBackend::CPU) ? "CPU" : "NPU";
}

double calculate_gflops(size_t ops, double time_ms) {
    return ops / (time_ms * 1e6);
}

void benchmark_streaming_stores(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    std::vector<size_t> sizes = {1024*1024, 4*1024*1024};

    for (size_t num_elements : sizes) {
        std::vector<__fp16> A(num_elements), B(num_elements), C(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            A[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
            B[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }

        double time_ms = time_operation<__fp16>([&]() {
            cactus_add_f16(A.data(), B.data(), C.data(), num_elements);
        }, config.iterations);

        double gb_per_sec = (num_elements * 2 * 3) / (time_ms * 1e6);
        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gb_per_sec << " GB/s";
        runner.log_performance(
            "Add (stream) " + std::to_string(num_elements / (1024*1024)) + "M elements",
            details.str());
    }
}

template<typename T>
void benchmark_binary_elementwise_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t, size_t)>>> ops = {
        {"Add", [](CactusGraph& b, size_t a, size_t c) { return b.add(a, c); }},
        {"Subtract", [](CactusGraph& b, size_t a, size_t c) { return b.subtract(a, c); }},
        {"Multiply", [](CactusGraph& b, size_t a, size_t c) { return b.multiply(a, c); }},
        {"Divide", [](CactusGraph& b, size_t a, size_t c) { return b.divide(a, c); }}
    };

    Precision precision = TestUtils::default_precision<T>();
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
            
            std::ostringstream details;
            details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                    << std::setprecision(2) << gflops << " GFLOPS";
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim),
                                 details.str());
        }
    }
}

void benchmark_conv1d_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> shapes = {
        {1, 3000, 80, 512},
        {1, 1500, 512, 512},
    };

    for (const auto& [N, L, C_in, C_out] : shapes) {
        size_t stride = 1;
        size_t out_len = ((L - 1) / stride) + 1;

        std::vector<__fp16> input(N * C_in * L);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }

        std::vector<__fp16> weight(C_out * C_in * 3);
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
        }

        std::vector<__fp16> output(N * C_out * out_len);

        double time_ms = time_operation<__fp16>([&]() {
            cactus_conv1d_f16_k3(input.data(), weight.data(), output.data(),
                                  N, L, C_in, C_out, stride);
        }, config.iterations);

        size_t flops = 2ULL * N * out_len * C_out * C_in * 3;
        double gflops = flops / (time_ms * 1e6);

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gflops << " GFLOPS";
        runner.log_performance(
            "Conv1D k3 " + std::to_string(N) + "x" + std::to_string(L) + "x" + std::to_string(C_in) + "→" + std::to_string(C_out),
            details.str());
    }
}

void benchmark_broadcast_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    for (size_t dim : config.dimensions) {
        size_t rows = dim;
        size_t cols = dim;
        size_t total_elements = rows * cols;

        std::vector<__fp16> A(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            A[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }

        std::vector<__fp16> B_vec(cols);
        for (size_t i = 0; i < cols; ++i) {
            B_vec[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
        }

        std::vector<__fp16> C(total_elements);

        {
            size_t a_strides[2] = {cols, 1};
            size_t b_strides[2] = {0, 1};
            size_t output_shape[2] = {rows, cols};

            double time_ms = time_operation<__fp16>([&]() {
                cactus_add_broadcast_f16(A.data(), B_vec.data(), C.data(),
                                          a_strides, b_strides, output_shape, 2);
            }, config.iterations);

            double gflops = calculate_gflops(total_elements, time_ms);
            double gb_per_sec = (total_elements * 2 * 3) / (time_ms * 1e6);

            std::ostringstream details;
            details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                    << std::setprecision(2) << gflops << " GFLOPS, "
                    << std::setprecision(2) << gb_per_sec << " GB/s";
            runner.log_performance(
                "Broadcast Add " + std::to_string(dim) + "x" + std::to_string(dim) + "+[" + std::to_string(dim) + "]",
                details.str());
        }
    }
}

template<typename T>
void benchmark_scalar_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t)>>> ops = {
        {"Scalar Add", [](CactusGraph& b, size_t a) { return b.scalar_add(a, 2.5f); }},
        {"Scalar Sub", [](CactusGraph& b, size_t a) { return b.scalar_subtract(a, 2.5f); }},
        {"Scalar Mul", [](CactusGraph& b, size_t a) { return b.scalar_multiply(a, 2.5f); }},
        {"Scalar Div", [](CactusGraph& b, size_t a) { return b.scalar_divide(a, 2.5f); }},
        {"Scalar Exp", [](CactusGraph& b, size_t a) { return b.scalar_exp(a); }},
        {"Scalar Sqrt", [](CactusGraph& b, size_t a) { return b.scalar_sqrt(a); }},
        {"Scalar Cos", [](CactusGraph& b, size_t a) { return b.scalar_cos(a); }},
        {"Scalar Sin", [](CactusGraph& b, size_t a) { return b.scalar_sin(a); }}
    };

    Precision precision = TestUtils::default_precision<T>();

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

            std::ostringstream details;
            details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                    << std::setprecision(2) << gflops << " GFLOPS";
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim),
                                 details.str());
        }
    }
}

template<typename T>
void benchmark_matmul_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = TestUtils::default_precision<T>();

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

                std::ostringstream details;
                details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                        << std::setprecision(2) << gflops << " GFLOPS";
                runner.log_performance("MatMul " + std::to_string(dim) + "³ " + backend_str,
                                     details.str());
            } catch (const std::exception& e) {
                runner.log_performance("MatMul " + std::to_string(dim) + "³ " + backend_str,
                                     "SKIP: " + std::string(e.what()));
            }
        }
    }
}

void benchmark_matmul_int8_grouped(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const size_t group_size = 64; 

    std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
        {1, 1024, 1024},
        {1024, 1024, 1024},
    };

    for (const auto& [M, K, N] : shapes) {
        size_t K_aligned = ((K + group_size - 1) / group_size) * group_size;
        size_t num_groups = K_aligned / group_size;

        std::vector<__fp16> A(M * K_aligned);
        for (size_t i = 0; i < M * K_aligned; ++i) {
            A[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }

        std::vector<int8_t> B(N * K_aligned);
        for (size_t i = 0; i < N * K_aligned; ++i) {
            B[i] = static_cast<int8_t>((rand() % 256) - 128);
        }

        std::vector<__fp16> B_scales(N * num_groups);
        for (size_t i = 0; i < N * num_groups; ++i) {
            B_scales[i] = static_cast<__fp16>(0.01f + (static_cast<float>(rand()) / RAND_MAX) * 0.05f);
        }

        std::vector<int8_t> A_quant(M * K_aligned);
        std::vector<float> A_scales(M);
        for (size_t m = 0; m < M; ++m) {
            float max_abs = cactus_fp16_max_abs(A.data() + m * K_aligned, K_aligned);
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            A_scales[m] = scale;
            cactus_fp16_to_int8(A.data() + m * K_aligned, A_quant.data() + m * K_aligned, K_aligned, scale);
        }

        std::vector<__fp16> C(M * N);

        double time_ms = time_operation<__fp16>([&]() {
            cactus_matmul_int8(A_quant.data(), A_scales.data(), B.data(), B_scales.data(), C.data(),
                                       M, K_aligned, N, group_size);
        }, config.iterations);

        double gflops = calculate_gflops(2ULL * M * K_aligned * N, time_ms);

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gflops << " GFLOPS";
        runner.log_performance(
            "MatMul INT8 " + std::to_string(M) + "x" + std::to_string(K_aligned) + "x" + std::to_string(N),
            details.str());
    }
}

template<typename T>
void benchmark_unary_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    const std::vector<std::pair<std::string, std::function<size_t(CactusGraph&, size_t)>>> ops = {
        {"Transpose", [](CactusGraph& b, size_t a) { return b.transpose(a); }}
    };

    Precision precision = TestUtils::default_precision<T>();

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

            std::ostringstream details;
            details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                    << std::setprecision(2) << gb_per_sec << " GB/s";
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim),
                                 details.str());
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

    Precision precision = TestUtils::default_precision<T>();

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

            std::ostringstream details;
            details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                    << std::setprecision(2) << gb_per_sec << " GB/s";
            runner.log_performance(op_name + " " + std::to_string(dim) + "x" + std::to_string(dim),
                                 details.str());
        }
    }
}

template<typename T>
void benchmark_advanced_ops(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = TestUtils::default_precision<T>();

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

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gb_per_sec << " GB/s";
        runner.log_performance("Softmax " + std::to_string(dim) + "x" + std::to_string(dim),
                             details.str());
    }
}

template<typename T>
void benchmark_rms_norm(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = Precision::FP16;

    for (size_t dim : config.dimensions) {
        size_t total_elements = dim * dim;

        CactusGraph graph;
        size_t input_a = graph.input({dim, dim}, precision);
        size_t weight = graph.input({dim}, precision);

        std::vector<__fp16> data_a(total_elements);
        std::vector<__fp16> weight_data(dim, static_cast<__fp16>(1.0f));
        for (size_t i = 0; i < total_elements; ++i) {
            data_a[i] = static_cast<__fp16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
        }

        graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), precision);
        graph.set_input(weight, const_cast<void*>(static_cast<const void*>(weight_data.data())), precision);

        graph.rms_norm(input_a, weight);

        double time_ms = time_operation<__fp16>([&]() {
            graph.execute();
        }, config.iterations);

        double gb_per_sec = (total_elements * 2) / (time_ms * 1e6);

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gb_per_sec << " GB/s";
        runner.log_performance("RMSNorm " + std::to_string(dim) + "x" + std::to_string(dim),
                             details.str());
        
        graph.hard_reset();
    }
}

template<typename T>
void benchmark_rope(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = Precision::FP16;

    for (size_t dim : config.dimensions) {
        size_t batch_size = 1;
        size_t seq_len = dim / 4;
        size_t num_heads = 4;
        size_t head_dim = dim / 4;
        size_t total_elements = batch_size * seq_len * num_heads * head_dim;

        TestUtils::FP16TestFixture fixture("RoPE");
        size_t input_a = fixture.create_input({batch_size, seq_len, num_heads, head_dim}, precision);

        std::vector<__fp16> data_a(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            data_a[i] = static_cast<__fp16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
        }
        fixture.set_input_data(input_a, data_a, precision);

        fixture.graph().rope(input_a, 10000.0f);

        double time_ms = time_operation<__fp16>([&]() {
            fixture.execute();
        }, config.iterations);

        double gb_per_sec = (total_elements * 2 * 2) / (time_ms * 1e6);

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gb_per_sec << " GB/s";
        runner.log_performance("RoPE " + std::to_string(seq_len) + "x" + std::to_string(num_heads) + "x" + std::to_string(head_dim),
                             details.str());
    }
}

template<typename T>
void benchmark_attention(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    Precision precision = TestUtils::default_precision<T>();

    for (size_t dim : config.dimensions) {
        size_t batch_size = 1;
        size_t seq_len = 1024;
        size_t num_heads = 16;
        size_t head_dim = dim / 16;
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

        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gflops << " GFLOPS";
        runner.log_performance("Attention " + std::to_string(seq_len) + "x" + std::to_string(num_heads) + "x" + std::to_string(head_dim),
                             details.str());

        graph.hard_reset();
    }
}


template<typename T>
void benchmark_embedding_ops(TestUtils::TestRunner& runner, BenchmarkConfig& config) {
    std::vector<size_t> vocab_sizes = {65000};
    std::vector<size_t> embedding_dims = {1024};
    std::vector<size_t> sequence_lengths = {1000};

    std::string precision_str = precision_to_string(TestUtils::default_precision<T>());

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

                Precision precision = TestUtils::default_precision<T>();

                size_t embeddings_id = graph.input({vocab_size, embedding_dim}, precision);
                size_t indices_id = graph.input({seq_len}, Precision::INT8);
                graph.embedding(embeddings_id, indices_id);

                graph.set_input(embeddings_id, embeddings_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);

                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);

                double throughput = (seq_len * embedding_dim * sizeof(T)) / (time_ms * 1e3);

                std::ostringstream details;
                details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                        << std::setprecision(2) << throughput << " GB/s";
                runner.log_performance(
                    "Embedding " + std::to_string(vocab_size/1000) + "k vocab x" +
                    std::to_string(embedding_dim) + " dim " + precision_str,
                    details.str());
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

                std::vector<__fp16> embeddings_data(vocab_size * embedding_dim);
                for (size_t i = 0; i < embeddings_data.size(); ++i) {
                    embeddings_data[i] = static_cast<__fp16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
                }

                size_t temp_embeddings = graph.input({vocab_size, embedding_dim}, Precision::FP16);
                graph.set_input(temp_embeddings, embeddings_data.data(), Precision::FP16);

                std::filesystem::path tmpdir;
                try {
                    tmpdir = std::filesystem::temp_directory_path();
                } catch (...) {
                    tmpdir = std::filesystem::current_path();
                }
                std::filesystem::path tmpfile = tmpdir / (std::string("perf_embeddings_") +
                    std::to_string(vocab_size) + "_" + std::to_string(embedding_dim) + ".bin");
                const std::string temp_file = tmpfile.string();

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

                double time_ms = time_operation<__fp16>([&]() {
                    graph.execute();
                }, config.iterations);

                double throughput = (seq_len * embedding_dim * sizeof(__fp16)) / (time_ms * 1e3);

                std::ostringstream details;
                details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                        << std::setprecision(2) << throughput << " GB/s";
                runner.log_performance(
                    "MMap Embedding " + std::to_string(vocab_size) + " vocab x" +
                    std::to_string(embedding_dim) + " dim",
                    details.str());

                std::error_code ec;
                std::filesystem::remove(tmpfile, ec);
            }
        }
    }
}

template<typename T>
void benchmark_gather_ops(TestUtils::TestRunner& runner, BenchmarkConfig& config) {
    std::string precision_str = precision_to_string(TestUtils::default_precision<T>());

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

                Precision precision = TestUtils::default_precision<T>();

                size_t tensor_id = graph.input({tensor_size}, precision);
                size_t indices_id = graph.input({index_count}, Precision::INT8);
                graph.gather(tensor_id, indices_id);

                graph.set_input(tensor_id, tensor_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);

                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);

                double throughput = (index_count * sizeof(T)) / (time_ms * 1e3);

                std::ostringstream details;
                details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                        << std::setprecision(2) << throughput << " GB/s";
                runner.log_performance(
                    "Gather 1D " + std::to_string(tensor_size) + "→" + std::to_string(index_count) + " " + precision_str,
                    details.str());
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

                Precision precision = TestUtils::default_precision<T>();

                size_t tensor_id = graph.input(shape, precision);
                size_t indices_id = graph.input({index_count}, Precision::INT8);
                graph.gather(tensor_id, indices_id);

                graph.set_input(tensor_id, tensor_data.data(), precision);
                graph.set_input(indices_id, indices_data.data(), Precision::INT8);

                double time_ms = time_operation<T>([&]() {
                    graph.execute();
                }, config.iterations);

                double throughput = (index_count * shape[1] * shape[2] * sizeof(T)) / (time_ms * 1e3);

                std::ostringstream details;
                details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                        << std::setprecision(2) << throughput << " GB/s";
                runner.log_performance(
                    "Gather 3D " + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + "x" + std::to_string(shape[2]) +
                    "→" + std::to_string(index_count) + " " + precision_str,
                    details.str());
            }
        }
    }
}

void benchmark_mel_filter_bank(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    using namespace cactus::engine;

    const size_t sampling_rate = 16000;
    const size_t feature_size = 80;
    const size_t num_frequency_bins = 201;

    AudioProcessor audio_proc;

    double time_ms = time_operation<float>([&]() {
        audio_proc.init_mel_filters(num_frequency_bins, feature_size, 0.0f, 8000.0f, sampling_rate);
    }, config.iterations);

    std::ostringstream details;
    details << std::fixed << std::setprecision(3) << time_ms << "ms";
    runner.log_performance(
        "Mel Filter Bank " + std::to_string(feature_size) + "x" + std::to_string(num_frequency_bins),
        details.str());
}

void benchmark_spectrogram(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    using namespace cactus::engine;

    const size_t n_fft = 400;
    const size_t hop_length = 160;
    const size_t chunk_length = 30;
    const size_t sampling_rate = 16000;
    const size_t feature_size = 80;
    const size_t num_frequency_bins = 1 + n_fft / 2;

    const size_t n_samples = chunk_length * sampling_rate;

    AudioProcessor audio_proc;
    audio_proc.init_mel_filters(num_frequency_bins, feature_size, 0.0f, 8000.0f, sampling_rate);

    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * M_PI * 440.0f * i / sampling_rate);
    }

    AudioProcessor::SpectrogramConfig spec_config;
    spec_config.n_fft = n_fft;
    spec_config.hop_length = hop_length;
    spec_config.frame_length = n_fft;
    spec_config.power = 2.0f;
    spec_config.center = true;
    spec_config.log_mel = "log10";

    double time_ms = time_operation<float>([&]() {
        audio_proc.compute_spectrogram(waveform, spec_config);
    }, config.iterations);

    const size_t pad_length = n_fft / 2;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - n_fft) / hop_length;

    double real_time_factor = (chunk_length * 1000.0) / time_ms;

    size_t bytes_processed = (n_samples * sizeof(float)) + (feature_size * num_frames * sizeof(float));
    double gb_per_sec = bytes_processed / (time_ms * 1e6);

    std::ostringstream details;
    details << std::fixed << std::setprecision(3) << time_ms << "ms, "
            << std::setprecision(1) << real_time_factor << "x RT, "
            << std::setprecision(2) << gb_per_sec << " GB/s";
    runner.log_performance(
        "Spectrogram " + std::to_string(chunk_length) + "s (" + std::to_string(num_frames) + " frames)",
        details.str());
}

void benchmark_gemm_f16_direct(TestUtils::TestRunner& runner, const BenchmarkConfig& config) {
    std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
        {1, 1024, 1024},
        {1024, 1024, 1024},
    };

    for (const auto& [M, K, N] : shapes) {
        std::vector<__fp16> A(M * K), B_T(N * K), C(M * N);
        for (size_t i = 0; i < M * K; ++i) {
            A[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }
        for (size_t i = 0; i < N * K; ++i) {
            B_T[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
        }

        double time_ms = time_operation<__fp16>([&]() {
            cactus_matmul_f16(A.data(), B_T.data(), C.data(), M, K, N);
        }, config.iterations);

        double gflops = calculate_gflops(2ULL * M * K * N, time_ms);
        std::ostringstream details;
        details << std::fixed << std::setprecision(3) << time_ms << "ms, "
                << std::setprecision(2) << gflops << " GFLOPS";
        runner.log_performance("MatMul F16 " + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N),
                             details.str());
    }
}

bool test_gemm_f16_direct_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_gemm_f16_direct(runner, config);
    return true;
}

bool test_streaming_stores_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_streaming_stores(runner, config);
    return true;
}

bool test_conv1d_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_conv1d_ops(runner, config);
    return true;
}

bool test_broadcast_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_broadcast_ops(runner, config);
    return true;
}

bool test_binary_elementwise_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_binary_elementwise_ops<__fp16>(runner, config);
    return true;
}

bool test_scalar_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_scalar_ops<__fp16>(runner, config);
    return true;
}

bool test_matrix_multiplication_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_matmul_ops<__fp16>(runner, config);
    return true;
}

bool test_grouped_int8_matmul_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_matmul_int8_grouped(runner, config);
    return true;
}

bool test_unary_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_unary_ops<__fp16>(runner, config);
    return true;
}

bool test_reduction_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_reduction_ops<__fp16>(runner, config);
    return true;
}

bool test_advanced_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_advanced_ops<__fp16>(runner, config);
    return true;
}

bool test_engine_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    benchmark_rms_norm<__fp16>(runner, config);
    benchmark_rope<__fp16>(runner, config);
    benchmark_attention<__fp16>(runner, config);
    return true;
}

bool test_gather_operations_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;
    // INT8 for storage, FP16 for computation
    benchmark_gather_ops<int8_t>(runner, config);
    benchmark_gather_ops<__fp16>(runner, config);
    return true;
}

bool test_signals_performance(TestUtils::TestRunner& runner) {
    BenchmarkConfig config;

    benchmark_mel_filter_bank(runner, config);
    benchmark_spectrogram(runner, config);

    return true;
}


int main() {
    TestUtils::TestRunner runner("Performance Benchmarks");

    runner.run_test("Streaming Stores", test_streaming_stores_performance(runner));
    runner.run_test("Conv1D Operations", test_conv1d_operations_performance(runner));
    runner.run_test("Broadcast Operations", test_broadcast_operations_performance(runner));
    runner.run_test("Binary Element-wise Operations", test_binary_elementwise_performance(runner));
    runner.run_test("Scalar Operations", test_scalar_operations_performance(runner));
    runner.run_test("Matrix Multiplication", test_matrix_multiplication_performance(runner));
    runner.run_test("F16 MatMul", test_gemm_f16_direct_performance(runner));
    runner.run_test("Grouped INT8 MatMul", test_grouped_int8_matmul_performance(runner));
    runner.run_test("Unary Operations", test_unary_operations_performance(runner));
    runner.run_test("Reduction Operations", test_reduction_operations_performance(runner));
    runner.run_test("Advanced Operations", test_advanced_operations_performance(runner));
    runner.run_test("Engine Operations", test_engine_operations_performance(runner));
    runner.run_test("Gather Operations", test_gather_operations_performance(runner));
    runner.run_test("Signals Operations", test_signals_performance(runner));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}