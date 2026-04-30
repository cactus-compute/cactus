#include "../cactus_kernels.h"
#include "tq_weight_loader.h"

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// ══════════════════════════════════════════════════════════════════════════════
// §2  Reference implementation
// ══════════════════════════════════════════════════════════════════════════════

static void fwht(float* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1)
        for (uint32_t i = 0; i < n; i += h << 1)
            for (uint32_t j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    float s = 1.f / std::sqrt((float)n);
    for (uint32_t i = 0; i < n; ++i) x[i] *= s;
}

static uint8_t unpack_index(const uint8_t* base, uint32_t bits, uint32_t k) {
    if (bits == 4) return (k & 1u) ? (base[k / 2] >> 4) : (base[k / 2] & 0x0Fu);
    return (base[k / 4] >> ((k & 3u) * 2u)) & 0x3u;
}

static void tq_reference_gemv(const TQWeightData& w, const float* x, float* y) {
    uint32_t pgb = (w.group_size * w.bits) / 8;
    for (uint32_t n = 0; n < w.N; ++n) {
        for (uint32_t g = 0; g < w.num_groups; ++g) {
            uint32_t base_k = g * w.group_size;
            std::vector<float> z(w.group_size);
            for (uint32_t k = 0; k < w.group_size; ++k)
                z[k] = x[base_k + k] / float(w.input_scale[base_k + k]) * float(w.left_signs[k]);
            fwht(z.data(), w.group_size);
            for (uint32_t k = 0; k < w.group_size; ++k)
                z[k] *= float(w.right_signs[k]);

            const uint8_t* packed_row = w.packed.data() + (size_t(n) * w.num_groups + g) * pgb;
            float gsum = 0.f;
            for (uint32_t k = 0; k < w.group_size; ++k) {
                uint8_t idx = unpack_index(packed_row, w.bits, k);
                float basis = (w.flags & CACTUS_TQ_FLAG_CODE_ORDERED_INDICES)
                    ? z[k] : z[w.permutation[k]];
                gsum += basis * float(w.codebook[idx]);
            }
            y[n] += float(w.norms[size_t(n) * w.num_groups + g]) * gsum;
        }
    }
}

static std::vector<float> tq_reference_gemm(const TQWeightData& w, const float* A, uint32_t M) {
    std::vector<float> out(size_t(M) * w.N, 0.f);
    for (uint32_t m = 0; m < M; ++m)
        tq_reference_gemv(w, A + size_t(m) * w.K, out.data() + size_t(m) * w.N);
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// §3  Test runner
// ══════════════════════════════════════════════════════════════════════════════

static std::vector<float> random_activations(uint32_t M, uint32_t K, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    std::vector<float> A(size_t(M) * K);
    for (auto& v : A) v = dis(gen);
    return A;
}

static bool check(const char* name, const float* expected, const __fp16* actual, size_t n) {
    float max_err = 0.f;
    size_t fail_i = SIZE_MAX;
    for (size_t i = 0; i < n; ++i) {
        float err = std::abs(float(actual[i]) - expected[i]);
        float tol = 0.1f + 0.02f * std::abs(expected[i]);
        if (err > tol && fail_i == SIZE_MAX) fail_i = i;
        max_err = std::max(max_err, err);
    }
    if (fail_i != SIZE_MAX) {
        std::cerr << "FAIL " << name << " [" << fail_i << "]"
                  << " expected=" << expected[fail_i]
                  << " got=" << float(actual[fail_i])
                  << " max_err=" << max_err << "\n";
        return false;
    }
    std::cout << "PASS " << name << "  max_err=" << max_err << "\n";
    return true;
}

static bool run(const char* name, const char* path, uint32_t max_rows, uint32_t M) {
    TQWeightData w;
    if (!tq_load_weights(path, max_rows, w)) {
        std::cerr << "SKIP " << name << ": could not load " << path << "\n";
        return true;
    }

    auto A_f  = random_activations(M, w.K, 42);
    auto ref  = tq_reference_gemm(w, A_f.data(), M);

    std::vector<__fp16> A_h(A_f.size());
    for (size_t i = 0; i < A_f.size(); ++i) A_h[i] = __fp16(A_f[i]);

    CactusTQMatrix mat = w.matrix();
    std::vector<__fp16> out(size_t(M) * w.N);

    if (M == 1) {
        if (w.bits == 4) cactus_tq4_gemv(&mat, A_h.data(), out.data());
        else             cactus_tq2_gemv(&mat, A_h.data(), out.data());
    } else {
        if (w.bits == 4) cactus_tq4_gemm(&mat, A_h.data(), M, out.data());
        else             cactus_tq2_gemm(&mat, A_h.data(), M, out.data());
    }

    return check(name, ref.data(), out.data(), out.size());
}

int main() {
    const char* tq4_path = CACTUS_TEST_ASSETS_DIR "/tq4_ffn_gate_64rows.weights";
    const char* tq2_path = CACTUS_TEST_ASSETS_DIR "/tq2_embed_tokens_64rows.weights";

    bool ok = true;
    ok &= run("tq4_gemv",  tq4_path, 64, 1);
    ok &= run("tq4_gemm",  tq4_path, 64, 8);
    ok &= run("tq2_gemv",  tq2_path, 64, 1);
    ok &= run("tq2_gemm",  tq2_path, 64, 8);
    return ok ? 0 : 1;
}
