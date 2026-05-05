#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace TestUtils;

bool test_image_preprocess_basic() {
    CactusGraph g;

    const int w = 32, h = 32, ch = 3, ps = 16;
    size_t pixel_count = static_cast<size_t>(w) * h * ch;

    size_t input = g.input({pixel_count}, Precision::FP32);
    std::vector<float> pixels(pixel_count);
    for (size_t i = 0; i < pixel_count; i++)
        pixels[i] = static_cast<float>(i % 256);
    g.set_input(input, pixels.data(), Precision::FP32);

    size_t result = g.image_preprocess(input, w, h, w, h, ps, ch);
    g.execute();

    const auto& buf = g.get_output_buffer(result);
    int ph = h / ps;
    int pw = w / ps;
    size_t expected_patches = static_cast<size_t>(ph) * pw;
    size_t patch_dim = static_cast<size_t>(ps) * ps * ch;

    if (buf.shape[0] != expected_patches) return false;
    if (buf.shape[1] != patch_dim) return false;

    float* out = static_cast<float*>(g.get_output(result));
    bool has_nonzero = false;
    for (size_t i = 0; i < buf.total_size; i++) {
        if (!std::isfinite(out[i])) return false;
        if (std::abs(out[i]) > 1e-6f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_image_preprocess_resize() {
    CactusGraph g;

    const int src_w = 64, src_h = 64, ch = 3, ps = 16;
    const int dst_w = 32, dst_h = 32;
    size_t pixel_count = static_cast<size_t>(src_w) * src_h * ch;

    size_t input = g.input({pixel_count}, Precision::FP32);
    std::vector<float> pixels(pixel_count, 128.0f);
    g.set_input(input, pixels.data(), Precision::FP32);

    size_t result = g.image_preprocess(input, src_w, src_h, dst_w, dst_h, ps, ch);
    g.execute();

    const auto& buf = g.get_output_buffer(result);
    int ph = dst_h / ps;
    int pw = dst_w / ps;
    if (buf.shape[0] != static_cast<size_t>(ph * pw)) return false;

    float* out = static_cast<float*>(g.get_output(result));
    for (size_t i = 0; i < buf.total_size; i++) {
        if (!std::isfinite(out[i])) return false;
    }
    return true;
}

bool test_image_preprocess_normalize() {
    CactusGraph g;

    const int w = 16, h = 16, ch = 3, ps = 16;
    size_t pixel_count = static_cast<size_t>(w) * h * ch;

    size_t input = g.input({pixel_count}, Precision::FP32);
    std::vector<float> pixels(pixel_count, 127.5f);
    g.set_input(input, pixels.data(), Precision::FP32);

    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_dev[3] = {0.5f, 0.5f, 0.5f};
    size_t result = g.image_preprocess(input, w, h, w, h, ps, ch, 1.0f / 255.0f, mean, std_dev);
    g.execute();

    float* out = static_cast<float*>(g.get_output(result));
    float expected = (127.5f / 255.0f - 0.5f) / 0.5f;
    if (std::abs(out[0] - expected) > 0.02f) return false;

    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, auto setup, auto run) {
        setup();
        run();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) run();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    {
        const int w = 512, h = 512, ch = 3, ps = 16;
        size_t pixel_count = static_cast<size_t>(w) * h * ch;
        std::vector<float> pixels(pixel_count, 128.0f);

        bench("image_preprocess 512x512", []{}, [&]{
            CactusGraph g;
            size_t inp = g.input({pixel_count}, Precision::FP32);
            g.set_input(inp, pixels.data(), Precision::FP32);
            g.image_preprocess(inp, w, h, w, h, ps, ch);
            g.execute();
        });
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("Image Preprocessing Tests");

    runner.run_test("Image Preprocess Basic", test_image_preprocess_basic());
    runner.run_test("Image Preprocess Resize", test_image_preprocess_resize());
    runner.run_test("Image Preprocess Normalize", test_image_preprocess_normalize());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
