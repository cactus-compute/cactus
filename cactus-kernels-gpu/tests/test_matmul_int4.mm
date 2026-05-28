/* Smoke test: INT4 (CQ4-style) mat-vec GPU vs CPU reference.
 * Uses synthetic CQ4-shaped weights (no rotation; codebook 0..15 → -8..7).
 *
 * Goal: prove the kernel dispatches and produces SOMETHING — full numeric
 * validation against the cactus quant pipeline lands when we wire the
 * Python transpiler. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

using namespace cactus::gpu;

static uint16_t f32_to_f16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp >  15) return (uint16_t)((sign << 15) | 0x7C00);
    if (exp < -14) return (uint16_t)(sign << 15);
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0)       out = (sign << 31);
    else if (exp == 31) out = (sign << 31) | (0xFF << 23) | (mant << 13);
    else                out = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s path/to/cactus_kernels.metallib\n", argv[0]);
        return 1;
    }

    Context* ctx = context_create(argv[1]);
    if (!ctx) { std::fprintf(stderr, "no context\n"); return 1; }

    // Small matmul: y = W (NxK, int4) * x (K, fp16), N=128, K=256
    // K must be a multiple of 128 (CQ4 group size). N must be a multiple of 4.
    const uint32_t K = 256;
    const uint32_t N = 128;
    const uint32_t GROUPS = K / 128;            // 2 groups

    // Build a deterministic quant-then-dequant W and reference outputs.
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> nibd(0, 15);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Pack: qs is laid out [K/4 × N] uint16, with row = (k/4), col = n.
    std::vector<uint16_t> qs(K / 4 * N, 0);
    std::vector<uint16_t> scales(GROUPS * N, 0);
    std::vector<float>    W_f32(K * N, 0.0f);  // dequantized reference

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < GROUPS; ++g) {
            float scale = 0.05f + 0.01f * float(n & 7);
            scales[g * N + n] = f32_to_f16(scale);
            for (uint32_t i = 0; i < 128; i += 4) {
                int q0 = nibd(rng), q1 = nibd(rng), q2 = nibd(rng), q3 = nibd(rng);
                uint16_t packed = (uint16_t)((q0 & 0xF) |
                                              ((q1 & 0xF) << 4) |
                                              ((q2 & 0xF) << 8) |
                                              ((q3 & 0xF) << 12));
                uint32_t k4_row = (g * 128 + i) / 4;
                qs[k4_row * N + n] = packed;
                // q4_0 dequant semantics: weight = scale * (nibble - 8).
                W_f32[(g * 128 + i + 0) * N + n] = scale * float(q0 - 8);
                W_f32[(g * 128 + i + 1) * N + n] = scale * float(q1 - 8);
                W_f32[(g * 128 + i + 2) * N + n] = scale * float(q2 - 8);
                W_f32[(g * 128 + i + 3) * N + n] = scale * float(q3 - 8);
            }
        }
    }

    std::vector<float>    x_f32(K, 0.0f);
    std::vector<uint16_t> x_f16(K, 0);
    for (uint32_t k = 0; k < K; ++k) { x_f32[k] = nd(rng) * 0.1f; x_f16[k] = f32_to_f16(x_f32[k]); }

    // CPU reference: y[n] = sum_k W[k][n] * x[k]
    std::vector<float> y_ref(N, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        double acc = 0.0;
        for (uint32_t k = 0; k < K; ++k) acc += double(W_f32[k * N + n]) * double(x_f32[k]);
        y_ref[n] = (float)acc;
    }

    Buffer* qs_buf     = buffer_wrap_host_memory(ctx, qs.data(),     qs.size() * 2);
    Buffer* scales_buf = buffer_wrap_host_memory(ctx, scales.data(), scales.size() * 2);
    Buffer* x_buf      = buffer_wrap_host_memory(ctx, x_f16.data(),  x_f16.size() * 2);
    Buffer* y_buf      = buffer_create(ctx, N * 2, StorageMode::SHARED);

    Pipeline* p = pipeline_mul_mv_int4_fp16(ctx, K, N);
    if (!p) { std::fprintf(stderr, "mv pipeline failed\n"); return 1; }

    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[4] = {
        {qs_buf, 0},
        {scales_buf, 0},
        {x_buf, 0},
        {y_buf, 0},
    };
    // Grid: N/4 threadgroups of 32 threads each.
    command_buffer_dispatch(cb, p, bb, 4,
                            N / 4, 1, 1,
                            32, 1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);

    uint16_t* y_gpu = (uint16_t*)buffer_contents(y_buf);
    double max_err = 0.0, mean_err = 0.0, ref_norm = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        float gpu = f16_to_f32(y_gpu[n]);
        double err = std::fabs(gpu - y_ref[n]);
        if (err > max_err) max_err = err;
        mean_err += err;
        ref_norm += std::fabs(y_ref[n]);
    }
    mean_err /= double(N);
    std::printf("matmul_int4 GPU vs CPU: mean_err=%.4g  max_err=%.4g  ref_avg=%.4g\n",
                mean_err, max_err, ref_norm / double(N));

    pipeline_destroy(p);
    buffer_destroy(qs_buf); buffer_destroy(scales_buf);
    buffer_destroy(x_buf);  buffer_destroy(y_buf);
    delete cb;
    context_destroy(ctx);

    // Relative tolerance: int4 + fp16 is noisy. Aim for relative < 2%.
    double avg_ref = ref_norm / double(N);
    return (max_err < 0.1 * avg_ref + 1e-2) ? 0 : 1;
}
