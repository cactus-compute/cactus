#include "test_utils.h"
#include "../cactus/kernel/kernel.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

static float compute_mse(const __fp16* a, const __fp16* b, size_t n) {
    float mse = 0;
    for (size_t i = 0; i < n; ++i) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        mse += diff * diff;
    }
    return mse / n;
}

// Run encode/decode roundtrip MSE for many seeds and report variance.
// If the implementation is rotation-invariant, MSE should be ~constant across seeds.
static bool seed_variance_test(size_t head_dim, size_t angle_bits, const char* label) {
    const size_t seq_len = 256;
    const size_t kv_heads = 4;
    const std::vector<uint64_t> seeds = {42, 7, 1337, 2026, 99, 12345, 3, 100, 555, 7777};

    std::vector<__fp16> src(seq_len * kv_heads * head_dim);
    TestUtils::fill_random_fp16(src);

    const size_t num_vectors = seq_len * kv_heads;
    const size_t angles_bytes_total = num_vectors * turboquant_angles_bytes_per_head(head_dim, angle_bits);

    std::vector<uint8_t> rotation_signs(turboquant_rotation_signs_bytes(head_dim));
    std::vector<float> radii(num_vectors);
    std::vector<uint8_t> angles(angles_bytes_total);
    std::vector<__fp16> dst(seq_len * kv_heads * head_dim);

    std::cout << "  " << label << " (head_dim=" << head_dim << ", " << angle_bits << "-bit):\n";
    std::cout << "    seed       MSE\n";

    float min_mse = 1e10f, max_mse = 0.0f, sum_mse = 0.0f;
    for (uint64_t seed : seeds) {
        cactus_turboquant_init(rotation_signs.data(), head_dim, seed);

        cactus_turboquant_encode_kv_fp16(
            src.data(), radii.data(), angles.data(),
            rotation_signs.data(),
            seq_len, kv_heads, head_dim, angle_bits);

        cactus_turboquant_decode_kv_fp16(
            radii.data(), angles.data(), rotation_signs.data(),
            dst.data(), seq_len, kv_heads, head_dim, angle_bits);

        float mse = compute_mse(src.data(), dst.data(), src.size());
        std::cout << "    " << std::setw(7) << seed << "    "
                  << std::fixed << std::setprecision(6) << mse << "\n";
        min_mse = std::min(min_mse, mse);
        max_mse = std::max(max_mse, mse);
        sum_mse += mse;
    }
    float mean = sum_mse / seeds.size();
    float spread = (max_mse - min_mse) / mean * 100.0f;
    std::cout << "    mean=" << std::fixed << std::setprecision(6) << mean
              << "  spread=" << std::setprecision(2) << spread << "% of mean\n\n";

    return spread < 5.0f;  // expect <5% spread for rotation-invariant
}

bool test_seed_variance_d64_4bit() { return seed_variance_test(64,  4, "lfm2-vl-like"); }
bool test_seed_variance_d256_4bit(){ return seed_variance_test(256, 4, "gemma-3-4b-like"); }
bool test_seed_variance_d256_2bit(){ return seed_variance_test(256, 2, "V-side at d=256"); }

int main() {
    TestUtils::TestRunner runner("Seed variance");
    runner.run_test("d=64 4-bit MSE stable across seeds",  test_seed_variance_d64_4bit());
    runner.run_test("d=256 4-bit MSE stable across seeds", test_seed_variance_d256_4bit());
    runner.run_test("d=256 2-bit MSE stable across seeds", test_seed_variance_d256_2bit());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
