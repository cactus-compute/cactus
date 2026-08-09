#include "test_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

static const char* g_bundle_path = std::getenv("CACTUS_SD_BUNDLE");

static bool check_tokens(cactus_model_t model, const char* text, const std::vector<uint32_t>& expected) {
    uint32_t buffer[128] = {0};
    size_t count = 0;
    int rc = cactus_tokenize(model, text, buffer, 128, &count);
    if (rc < 0 || count != expected.size()) {
        std::cerr << "[✗] tokenize \"" << text << "\": rc=" << rc << " count=" << count
                  << " expected " << expected.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (buffer[i] != expected[i]) {
            std::cerr << "[✗] tokenize \"" << text << "\": token[" << i << "]=" << buffer[i]
                      << " expected " << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

static bool test_clip_tokenizer(cactus_model_t model) {
    bool ok = check_tokens(model, "a photograph of an astronaut riding a horse",
                           {320, 8853, 539, 550, 18376, 6765, 320, 4558});
    ok = check_tokens(model, "Hello, WORLD!! (test-123)",
                      {3306, 267, 1002, 748, 263, 1628, 268, 272, 273, 274, 264}) && ok;
    std::cout << "├─ CLIP tokenizer parity: " << (ok ? "ok" : "FAILED") << "\n";
    return ok;
}

static bool test_generate(cactus_model_t model) {
    std::vector<uint8_t> rgb(512 * 512 * 3, 0);
    unsigned int width = 0;
    unsigned int height = 0;
    int rc = cactus_generate_image(model, "a photograph of an astronaut riding a horse",
                                   rgb.data(), rgb.size(), &width, &height, 2, 8.5f, 1234ULL);
    if (rc != static_cast<int>(rgb.size()) || width != 512 || height != 512) {
        std::cerr << "[✗] generate_image rc=" << rc << " " << width << "x" << height << "\n";
        return false;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    for (uint8_t v : rgb) {
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    const double mean = sum / rgb.size();
    const double stddev = std::sqrt(sum_sq / rgb.size() - mean * mean);
    std::cout << "├─ image 512x512: mean=" << mean << " stddev=" << stddev << "\n";
    if (mean < 5.0 || mean > 250.0 || stddev < 10.0) {
        std::cerr << "[✗] generated image looks degenerate\n";
        return false;
    }
    return true;
}

static bool test_diffusion() {
    cactus_model_t model = cactus_init(g_bundle_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize bundle: " << g_bundle_path << "\n";
        return false;
    }
    bool passed = test_clip_tokenizer(model);
    passed = test_generate(model) && passed;
    cactus_destroy(model);
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

int main() {
    if (!g_bundle_path || !*g_bundle_path) {
        std::cout << "CACTUS_SD_BUNDLE not set; skipping diffusion tests\n";
        return 0;
    }
    TestUtils::TestRunner runner("Diffusion Tests");
    runner.run_test("text_to_image", test_diffusion());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
