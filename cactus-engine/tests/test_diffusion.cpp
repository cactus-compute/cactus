#include "test_utils.h"
#include "../src/engine.h"
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

static bool test_lcm_scheduler() {
    cactus::engine::DiffusionParams params;
    const std::vector<uint32_t> expected_four = {999, 759, 499, 259};
    if (cactus::engine::diffusion_lcm_timesteps(params, 4) != expected_four) {
        std::cerr << "[✗] 4-step LCM schedule does not match diffusers\n";
        return false;
    }
    const std::vector<uint32_t> expected_two = {999, 499};
    if (cactus::engine::diffusion_lcm_timesteps(params, 2) != expected_two) {
        std::cerr << "[✗] 2-step LCM schedule does not match diffusers\n";
        return false;
    }
    if (!cactus::engine::diffusion_lcm_timesteps(params, 0).empty()) {
        std::cerr << "[✗] zero-step schedule should be empty\n";
        return false;
    }

    const std::vector<float> alphas = cactus::engine::diffusion_alphas_cumprod(params);
    if (alphas.size() != params.num_train_timesteps) {
        std::cerr << "[✗] alphas_cumprod has the wrong length\n";
        return false;
    }
    if (!(alphas.front() < 1.0f && alphas.back() > 0.0f && alphas.back() < alphas.front())) {
        std::cerr << "[✗] alphas_cumprod is not a decreasing schedule inside (0, 1)\n";
        return false;
    }
    if (std::abs(alphas[999] - 0.00466f) > 1e-4f) {
        std::cerr << "[✗] alphas_cumprod[999]=" << alphas[999] << " expected ~0.00466\n";
        return false;
    }

    for (size_t dim : {2u, 256u}) {
        const std::vector<float> embedding = cactus::engine::diffusion_guidance_embedding(7.5f, dim);
        if (embedding.size() != dim) {
            std::cerr << "[✗] guidance embedding has the wrong length for dim=" << dim << "\n";
            return false;
        }
        for (float value : embedding) {
            if (!std::isfinite(value)) {
                std::cerr << "[✗] guidance embedding is not finite for dim=" << dim << "\n";
                return false;
            }
        }
    }
    std::cout << "├─ LCM scheduler: ok\n";
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
    TestUtils::TestRunner runner("Diffusion Tests");
    runner.run_test("lcm_scheduler", test_lcm_scheduler());
    if (g_bundle_path && *g_bundle_path) {
        runner.run_test("text_to_image", test_diffusion());
    } else {
        std::cout << "CACTUS_SD_BUNDLE not set; skipping text-to-image generation\n";
    }
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
