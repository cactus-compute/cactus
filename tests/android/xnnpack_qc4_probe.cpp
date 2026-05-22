#include <xnnpack.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

volatile float sink_f32 = 0.0f;
std::string shape_filter;

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Shape {
    const char* name;
    size_t K;
    size_t N;
};

void check_xnn(enum xnn_status status, const char* what) {
    if (status != xnn_status_success) {
        std::cerr << what << " failed status=" << static_cast<int>(status) << "\n";
        std::exit(2);
    }
}

void run_shape(const Shape& shape, int iterations) {
    if (!shape_filter.empty() && shape_filter != shape.name) return;

    std::vector<int8_t> input(shape.K);
    std::vector<uint8_t> kernel(shape.N * shape.K / 2);
    std::vector<int8_t> kernel_qc8(shape.N * shape.K);
    std::vector<float> kernel_scales(shape.N);
    std::vector<float> bias(shape.N, 0.0f);
    std::vector<float> output(shape.N);
    std::vector<xnn_quantization_params> quantization(1 + XNN_EXTRA_QUANTIZATION_PARAMS);

    std::mt19937 gen(123);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> act_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.25f);
    for (auto& v : input) v = static_cast<int8_t>(act_dist(gen));
    for (auto& v : kernel) v = static_cast<uint8_t>(byte_dist(gen));
    for (auto& v : kernel_qc8) v = static_cast<int8_t>(act_dist(gen));
    for (auto& v : kernel_scales) v = scale_dist(gen);
    for (auto& q : quantization) {
        q.zero_point = 0;
        q.scale = 0.01f;
    }

    xnn_operator_t op = nullptr;
    check_xnn(xnn_create_fully_connected_nc_qd8_f32_qc4w(
                  shape.K, shape.N, shape.K, shape.N, 8, kernel_scales.data(),
                  kernel.data(), bias.data(), -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(), 0, nullptr, &op),
              "xnn_create_fully_connected_nc_qd8_f32_qc4w");

    size_t workspace_size = 0;
    check_xnn(xnn_reshape_fully_connected_nc_qd8_f32_qc4w(op, 1, &workspace_size, nullptr),
              "xnn_reshape_fully_connected_nc_qd8_f32_qc4w");
    std::vector<uint8_t> workspace(workspace_size + 64);

    check_xnn(xnn_setup_fully_connected_nc_qd8_f32_qc4w(
                  op, input.data(), output.data(), workspace.data(), quantization.data()),
              "xnn_setup_fully_connected_nc_qd8_f32_qc4w");

    check_xnn(xnn_run_operator(op, nullptr), "xnn_run_operator warmup");
    const uint64_t start = now_ns();
    for (int i = 0; i < iterations; ++i) {
        check_xnn(xnn_run_operator(op, nullptr), "xnn_run_operator");
    }
    const double ms = static_cast<double>(now_ns() - start) / 1.0e6 / static_cast<double>(iterations);
    for (float v : output) sink_f32 += v;

    std::cout << "shape=" << shape.name
              << " K=" << shape.K
              << " N=" << shape.N
              << " workspace_bytes=" << workspace_size
              << " iterations=" << iterations << "\n"
              << "xnn_qd8_f32_qc4w_ms=" << std::fixed << std::setprecision(6) << ms << "\n";

    check_xnn(xnn_delete_operator(op), "xnn_delete_operator");

    op = nullptr;
    check_xnn(xnn_create_fully_connected_nc_qd8_f32_qc8w(
                  shape.K, shape.N, shape.K, shape.N, kernel_scales.data(),
                  kernel_qc8.data(), bias.data(), -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(), 0, nullptr, &op),
              "xnn_create_fully_connected_nc_qd8_f32_qc8w");

    workspace_size = 0;
    check_xnn(xnn_reshape_fully_connected_nc_qd8_f32_qc8w(op, 1, &workspace_size, nullptr),
              "xnn_reshape_fully_connected_nc_qd8_f32_qc8w");
    workspace.assign(workspace_size + 64, 0);

    check_xnn(xnn_setup_fully_connected_nc_qd8_f32_qc8w(
                  op, input.data(), output.data(), workspace.data(), quantization.data()),
              "xnn_setup_fully_connected_nc_qd8_f32_qc8w");

    check_xnn(xnn_run_operator(op, nullptr), "xnn_run_operator qc8 warmup");
    const uint64_t qc8_start = now_ns();
    for (int i = 0; i < iterations; ++i) {
        check_xnn(xnn_run_operator(op, nullptr), "xnn_run_operator qc8");
    }
    const double qc8_ms = static_cast<double>(now_ns() - qc8_start) / 1.0e6 / static_cast<double>(iterations);
    for (float v : output) sink_f32 += v;

    std::cout << "xnn_qd8_f32_qc8w_ms=" << std::fixed << std::setprecision(6) << qc8_ms << "\n";

    check_xnn(xnn_delete_operator(op), "xnn_delete_operator qc8");
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 200;
    if (argc > 1) iterations = std::max(1, std::atoi(argv[1]));
    if (argc > 2) shape_filter = argv[2];

    check_xnn(xnn_initialize(nullptr), "xnn_initialize");

    const Shape shapes[] = {
        {"gemma_ffn_up", 1536, 12288},
        {"gemma_ffn_down", 12288, 1536},
        {"gemma_ffn_mid_up", 1536, 6144},
        {"gemma_ffn_mid_down", 6144, 1536},
        {"gemma_attn_q", 1536, 2048},
        {"gemma_attn_in", 2048, 1536},
        {"gemma_output_head", 1536, 262144},
    };
    for (const Shape& shape : shapes) {
        run_shape(shape, iterations);
    }
    std::cout << "sink_f32=" << sink_f32 << "\n";
    return 0;
}
