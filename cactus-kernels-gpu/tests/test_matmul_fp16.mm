/* Smoke + correctness for mul_mv_fp16 (fp16 × fp16 → fp16 LM head).
 * Reference: CPU fp32 matmul. */
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
    if (argc < 2) { std::fprintf(stderr, "usage: %s metallib\n", argv[0]); return 1; }
    Context* ctx = context_create(argv[1]);
    if (!ctx) return 1;

    // Realistic LM-head shape (Gemma 4 E2B: K=1536, N=262144 is too large for
    // the test; pick K=1024, N=512 = a modest mat-vec)
    const uint32_t K = 1024, N = 512;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> Wf(N*K), xf(K);
    for (auto& v : Wf) v = nd(rng) * 0.02f;
    for (auto& v : xf) v = nd(rng) * 0.3f;

    std::vector<uint16_t> Wh(N*K), xh(K);
    for (size_t i = 0; i < N*K; ++i) Wh[i] = f32_to_f16(Wf[i]);
    for (size_t i = 0; i < K;   ++i) xh[i] = f32_to_f16(xf[i]);

    // CPU reference: y[n] = sum_k W[n,k] * x[k]
    std::vector<float> y_ref(N, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        double acc = 0.0;
        for (uint32_t k = 0; k < K; ++k) acc += double(Wf[n*K + k]) * double(xf[k]);
        y_ref[n] = (float)acc;
    }

    Buffer* W_b = buffer_wrap_host_memory(ctx, Wh.data(), Wh.size() * 2);
    Buffer* x_b = buffer_wrap_host_memory(ctx, xh.data(), xh.size() * 2);
    Buffer* y_b = buffer_create(ctx, N * 2, StorageMode::SHARED);

    Pipeline* p = pipeline_mul_mv_fp16(ctx, K, N);
    if (!p) { std::fprintf(stderr, "pipeline_mul_mv_fp16 failed\n"); return 1; }

    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[3] = {{W_b, 0}, {x_b, 0}, {y_b, 0}};
    // N/4 threadgroups of 32 threads each (matches kernel: MV_FP16_ROWS=4)
    command_buffer_dispatch(cb, p, bb, 3, N/4, 1, 1, 32, 1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);
    command_buffer_destroy(cb);

    uint16_t* y_gpu = (uint16_t*)buffer_contents(y_b);
    double max_err = 0.0, mean_err = 0.0, ref_max_abs = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        float g = f16_to_f32(y_gpu[n]);
        double e = std::fabs(g - y_ref[n]);
        if (e > max_err) max_err = e;
        mean_err += e;
        if (std::fabs(y_ref[n]) > ref_max_abs) ref_max_abs = std::fabs(y_ref[n]);
    }
    mean_err /= double(N);
    std::printf("mul_mv_fp16 GPU vs CPU: mean_err=%.4g max_err=%.4g ref_max_abs=%.4g\n",
                mean_err, max_err, ref_max_abs);
    // Sample a few elements so we can sanity-check
    std::printf("  y[0]   gpu=%.4f  ref=%.4f\n", f16_to_f32(y_gpu[0]),   y_ref[0]);
    std::printf("  y[N-1] gpu=%.4f  ref=%.4f\n", f16_to_f32(y_gpu[N-1]), y_ref[N-1]);

    pipeline_destroy(p);
    buffer_destroy(W_b); buffer_destroy(x_b); buffer_destroy(y_b);
    context_destroy(ctx);
    // fp16 accumulation noise: max_err ~ 1% of ref_max_abs is acceptable.
    return (max_err < 0.05 * ref_max_abs + 1e-3) ? 0 : 1;
}
