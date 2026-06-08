#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace TestUtils;

bool test_conv1d_k3() {
    const size_t N = 1, L = 32, C_in = 8, C_out = 16, stride = 1;
    const size_t L_out = ((L - 1) / stride) + 1;
    std::vector<__fp16> input(N * L * C_in), weight(C_out * 3 * C_in), output(N * L_out * C_out);
    fill_random_fp16(input, -0.5f, 0.5f);
    fill_random_fp16(weight, -0.5f, 0.5f);

    cactus_conv1d_f16_k3(input.data(), weight.data(), output.data(), N, L, C_in, C_out, stride);

    for (size_t i = 0; i < N * L_out * C_out; i++) {
        if (!std::isfinite(static_cast<float>(output[i]))) {
            std::cerr << "  conv1d_k3: non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool test_conv1d_causal_depthwise() {
    const size_t N = 1, L = 16, C = 32, K = 3;
    std::vector<__fp16> input(N * L * C), weight(C * K), output(N * L * C);
    fill_random_fp16(input, -0.5f, 0.5f);
    fill_random_fp16(weight, -0.5f, 0.5f);

    cactus_conv1d_causal_depthwise_f16(input.data(), weight.data(), output.data(), N, L, C, K, 1);

    for (size_t i = 0; i < N * L * C; i++) {
        if (!std::isfinite(static_cast<float>(output[i]))) {
            std::cerr << "  conv1d_causal: non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool test_stft_complex() {
    const size_t N = 1, L = 128, C_in = 1, K = 64, stride = 32;
    const size_t num_fft_bins = K / 2;
    const size_t C_out = K;
    const size_t num_frames = (L - K) / stride + 1;
    std::vector<__fp16> input(N * L * C_in), weight(C_out * K * C_in);
    std::vector<__fp16> output(N * num_frames * num_fft_bins * 2);
    fill_random_fp16(input, -1.0f, 1.0f);
    fill_random_fp16(weight, -0.5f, 0.5f);

    cactus_stft_f16(input.data(), weight.data(), output.data(), N, L, C_in, C_out, K, stride, num_fft_bins);

    for (size_t i = 0; i < output.size(); i++) {
        if (!std::isfinite(static_cast<float>(output[i]))) {
            std::cerr << "  stft: non-finite at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool test_maxpool1d() {
    const size_t batch = 1, channels = 4, length = 16, kernel = 3, stride = 2;
    const size_t out_len = (length - kernel) / stride + 1;
    std::vector<__fp16> input(batch * channels * length), output(batch * channels * out_len);
    fill_random_fp16(input, -5.0f, 5.0f);

    cactus_maxpool1d_f16(input.data(), output.data(), batch, channels, length, kernel, stride);

    for (size_t c = 0; c < channels; c++) {
        for (size_t o = 0; o < out_len; o++) {
            float expected_max = -1e9f;
            for (size_t k = 0; k < kernel; k++) {
                float v = static_cast<float>(input[c * length + o * stride + k]);
                if (v > expected_max) expected_max = v;
            }
            float actual = static_cast<float>(output[c * out_len + o]);
            if (std::abs(actual - expected_max) > 0.01f) {
                std::cerr << "  maxpool1d mismatch [c=" << c << ",o=" << o << "]: "
                          << actual << " vs " << expected_max << "\n";
                return false;
            }
        }
    }
    return true;
}

bool run_benchmarks() {
    {
        const size_t N = 1, L = 3000, C_in = 80, C_out = 512, stride = 1;
        const size_t L_out = ((L - 1) / stride) + 1;
        std::vector<__fp16> input(N * L * C_in), weight(C_out * 3 * C_in), output(N * L_out * C_out);
        fill_random_fp16(input, -0.5f, 0.5f);
        fill_random_fp16(weight, -0.5f, 0.5f);
        cactus_conv1d_f16_k3(input.data(), weight.data(), output.data(), N, L, C_in, C_out, stride);
        Timer t;
        for (int i = 0; i < 100; i++)
            cactus_conv1d_f16_k3(input.data(), weight.data(), output.data(), N, L, C_in, C_out, stride);
        double ms = t.elapsed_ms() / 100.0;
        double gflops = (2.0 * N * L_out * C_out * 3 * C_in) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "conv1d_k3 1x3000x80->512"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    }
    {
        const size_t N = 1, L = 256, C = 128, K = 3, stride = 1;
        std::vector<__fp16> input(N * L * C), weight(C * K), output(N * L * C);
        fill_random_fp16(input, -0.5f, 0.5f);
        fill_random_fp16(weight, -0.5f, 0.5f);
        cactus_conv1d_causal_depthwise_f16(input.data(), weight.data(), output.data(), N, L, C, K, stride);
        Timer t;
        for (int i = 0; i < 100; i++)
            cactus_conv1d_causal_depthwise_f16(input.data(), weight.data(), output.data(), N, L, C, K, stride);
        double ms = t.elapsed_ms() / 100.0;
        double gflops = (2.0 * N * L * C * K) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "conv1d_causal 1x256x128 k3"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    }
    {
        const size_t N = 1, L = 1024, C_in = 1, K = 64, stride = 32;
        const size_t num_fft_bins = K / 2;
        const size_t C_out = K;
        const size_t num_frames = (L - K) / stride + 1;
        std::vector<__fp16> input(N * L * C_in), weight(C_out * K * C_in);
        std::vector<__fp16> output(N * num_frames * num_fft_bins * 2);
        fill_random_fp16(input, -1.0f, 1.0f);
        fill_random_fp16(weight, -0.5f, 0.5f);
        cactus_stft_f16(input.data(), weight.data(), output.data(), N, L, C_in, C_out, K, stride, num_fft_bins);
        Timer t;
        for (int i = 0; i < 100; i++)
            cactus_stft_f16(input.data(), weight.data(), output.data(), N, L, C_in, C_out, K, stride, num_fft_bins);
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(28) << "stft 1x1024x1 k64"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << ms << " ms\n";
    }
    return true;
}

int main() {
    TestRunner runner("Convolution & Pooling");
    runner.run_test("conv1d_k3", test_conv1d_k3());
    runner.run_test("conv1d_causal_depthwise", test_conv1d_causal_depthwise());
    runner.run_test("stft_complex", test_stft_complex());
    runner.run_test("maxpool1d", test_maxpool1d());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
