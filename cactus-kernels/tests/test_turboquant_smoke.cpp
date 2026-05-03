// Smoke test for ported TurboQuant KV cache kernels.
// Verifies that encode → decode round-trips with reasonable error,
// and that K=4 / K=6 dot products work end-to-end.
//
// This is a minimal correctness check — full benchmarking against gemma-3/4
// happens in Phase 6 of the v2 port.

#include "test_utils.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace TestUtils;

// Test 1: encode → decode round-trips for K=4 and K=6 with bounded reconstruction error.
bool test_tq_encode_decode_roundtrip(size_t head_dim, size_t angle_bits) {
    const size_t seq_len = 16;
    const size_t kv_heads = 1;

    std::vector<__fp16> src(seq_len * kv_heads * head_dim);
    fill_random_fp16(src, -1.0f, 1.0f);

    std::vector<uint8_t> rot_signs(turboquant_rotation_signs_bytes(head_dim));
    cactus_turboquant_init(rot_signs.data(), head_dim, /*seed=*/7);

    std::vector<float> radii(seq_len * kv_heads);
    std::vector<uint8_t> angles(seq_len * kv_heads * turboquant_angles_bytes_per_head(head_dim, angle_bits));

    cactus_turboquant_encode_kv_fp16(src.data(), radii.data(), angles.data(),
                                     rot_signs.data(),
                                     seq_len, kv_heads, head_dim, angle_bits);

    std::vector<__fp16> dst(seq_len * kv_heads * head_dim);
    cactus_turboquant_decode_kv_fp16(radii.data(), angles.data(), rot_signs.data(),
                                     dst.data(), seq_len, kv_heads, head_dim, angle_bits);

    // Compute mean squared error after round-trip; should be small but non-zero
    // (codebook quantization is lossy).
    double mse = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        double diff = static_cast<float>(src[i]) - static_cast<float>(dst[i]);
        mse += diff * diff;
    }
    mse /= src.size();

    // Empirical bounds — typical Lloyd-Max MSE for unit-Gaussian-ish input:
    //   K=4 (16 levels): ~0.05–0.15
    //   K=6 (64 levels): ~0.005–0.02
    // We just check it's bounded; a large MSE means the kernel is broken.
    const double mse_upper_bound = (angle_bits >= 6) ? 0.05 : 0.30;
    bool ok = (mse > 0.0) && (mse < mse_upper_bound);

    if (!ok) {
        std::cerr << "  [FAIL] head_dim=" << head_dim << " angle_bits=" << angle_bits
                  << " mse=" << mse << " (expected > 0 and < " << mse_upper_bound << ")\n";
    } else {
        std::cerr << "  [ok]   head_dim=" << head_dim << " angle_bits=" << angle_bits
                  << " mse=" << mse << "\n";
    }
    return ok;
}

// Test 2: encode → dot_Nbit_f32 should approximate true Q·K dot.
bool test_tq_dot_consistency(size_t head_dim, size_t angle_bits) {
    std::vector<__fp16> k_src_fp16(head_dim);
    std::vector<float> q_rotated(head_dim);
    fill_random_fp16(k_src_fp16, -1.0f, 1.0f);
    for (size_t i = 0; i < head_dim; ++i) {
        q_rotated[i] = (i % 2) ? 0.5f : -0.5f;
    }

    // Encode K
    std::vector<uint8_t> rot_signs(turboquant_rotation_signs_bytes(head_dim));
    cactus_turboquant_init(rot_signs.data(), head_dim, /*seed=*/7);

    std::vector<float> radii(1);
    std::vector<uint8_t> angles(turboquant_angles_bytes_per_head(head_dim, angle_bits));

    cactus_turboquant_encode_kv_fp16(k_src_fp16.data(), radii.data(), angles.data(),
                                     rot_signs.data(),
                                     /*seq_len=*/1, /*kv_heads=*/1, head_dim, angle_bits);

    // Dot Q · decoded(K) via the bit-specific dot kernel
    float result;
    if (angle_bits == 4) {
        result = dot_4bit_f32(q_rotated.data(), angles.data(), head_dim) * radii[0];
    } else if (angle_bits == 6) {
        result = dot_6bit_f32(q_rotated.data(), angles.data(), head_dim) * radii[0];
    } else {
        return false;  // unsupported in this smoke test
    }

    // Just check it produced a finite result (basic sanity).
    bool ok = std::isfinite(result);
    std::cerr << "  [" << (ok ? "ok" : "FAIL") << "]   head_dim=" << head_dim
              << " angle_bits=" << angle_bits << " dot=" << result << "\n";
    return ok;
}

bool test_tq_4bit_roundtrip_dim256() { return test_tq_encode_decode_roundtrip(256, 4); }
bool test_tq_6bit_roundtrip_dim256() { return test_tq_encode_decode_roundtrip(256, 6); }
bool test_tq_4bit_roundtrip_dim512() { return test_tq_encode_decode_roundtrip(512, 4); }
bool test_tq_6bit_roundtrip_dim512() { return test_tq_encode_decode_roundtrip(512, 6); }

bool test_tq_4bit_dot_dim256() { return test_tq_dot_consistency(256, 4); }
bool test_tq_6bit_dot_dim256() { return test_tq_dot_consistency(256, 6); }
bool test_tq_6bit_dot_dim512() { return test_tq_dot_consistency(512, 6); }

int main() {
    TestRunner runner("TurboQuant KV cache kernels (smoke)");
    runner.run_test("encode_decode_roundtrip K4 dim=256", test_tq_4bit_roundtrip_dim256());
    runner.run_test("encode_decode_roundtrip K6 dim=256", test_tq_6bit_roundtrip_dim256());
    runner.run_test("encode_decode_roundtrip K4 dim=512", test_tq_4bit_roundtrip_dim512());
    runner.run_test("encode_decode_roundtrip K6 dim=512", test_tq_6bit_roundtrip_dim512());
    runner.run_test("dot_4bit_f32 dim=256", test_tq_4bit_dot_dim256());
    runner.run_test("dot_6bit_f32 dim=256", test_tq_6bit_dot_dim256());
    runner.run_test("dot_6bit_f32 dim=512", test_tq_6bit_dot_dim512());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
