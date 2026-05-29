/* Smoke test: RMSNorm GPU vs CPU reference.
 *
 *   out[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps)
 *
 * Builds a small (B=4 rows, D=512) random input, runs both CPU and GPU,
 * checks the GPU output is within tolerance. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

using namespace cactus::gpu;

// fp32-on-CPU reference.
static void rms_norm_cpu_ref(const std::vector<float>& x,
                             const std::vector<float>& w,
                             std::vector<float>& out,
                             size_t B, size_t D, float eps) {
    out.assign(B * D, 0.0f);
    for (size_t b = 0; b < B; ++b) {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double v = x[b*D + i];
            sum += v * v;
        }
        double inv = 1.0 / std::sqrt(sum / double(D) + eps);
        for (size_t i = 0; i < D; ++i) {
            out[b*D + i] = (float)(w[i] * x[b*D + i] * inv);
        }
    }
}

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp >  15) return (uint16_t)((sign << 15) | 0x7C00); // inf
    if (exp < -14) return (uint16_t)(sign << 15);            // 0 (subnormal not handled)
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0)        out = (sign << 31);
    else if (exp == 31)  out = (sign << 31) | (0xFF << 23) | (mant << 13);
    else                 out = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s path/to/cactus_kernels.metallib\n", argv[0]);
        return 1;
    }

    Context* ctx = context_create(argv[1]);
    if (!ctx) { std::fprintf(stderr, "no context\n"); return 1; }

    const size_t B = 4, D = 512;
    const float eps = 1e-6f;

    // Synthetic input.
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x_f32(B*D), w_f32(D), ref(B*D);
    for (auto& v : x_f32) v = nd(rng);
    for (auto& v : w_f32) v = nd(rng) * 0.1f + 1.0f;  // weight near 1.0

    rms_norm_cpu_ref(x_f32, w_f32, ref, B, D, eps);

    // fp16 buffers.
    std::vector<uint16_t> x_f16(B*D), w_f16(D);
    for (size_t i = 0; i < B*D; ++i) x_f16[i] = f32_to_f16(x_f32[i]);
    for (size_t i = 0; i < D;   ++i) w_f16[i] = f32_to_f16(w_f32[i]);

    Buffer* x_buf   = buffer_wrap_host_memory(ctx, x_f16.data(), x_f16.size() * 2);
    Buffer* w_buf   = buffer_wrap_host_memory(ctx, w_f16.data(), w_f16.size() * 2);
    Buffer* out_buf = buffer_create(ctx, B * D * 2, StorageMode::SHARED);
    Buffer* eps_buf = buffer_wrap_host_memory(ctx, (void*)&eps, sizeof(float));

    Pipeline* p = pipeline_rms_norm_fp16(ctx, (uint32_t)D);
    if (!p) { std::fprintf(stderr, "rms_norm pipeline failed\n"); return 1; }

    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[] = {
        {x_buf, 0},
        {w_buf, 0},
        {out_buf, 0},
        {eps_buf, 0},
    };
    // One threadgroup per row. Each thread covers 4 elements (N_READS=4).
    const uint32_t tg_threads = (uint32_t)(D / 4);
    command_buffer_dispatch(cb, p, bb, 4,
                            (uint32_t)B, 1, 1,
                            tg_threads, 1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);

    // Compare.
    uint16_t* gpu_f16 = (uint16_t*)buffer_contents(out_buf);
    double max_err = 0.0, mean_err = 0.0;
    for (size_t i = 0; i < B*D; ++i) {
        float gpu = f16_to_f32(gpu_f16[i]);
        float err = std::fabs(gpu - ref[i]);
        if (err > max_err) max_err = err;
        mean_err += err;
    }
    mean_err /= double(B*D);
    std::printf("RMSNorm GPU vs CPU: mean_err=%.4g  max_err=%.4g\n", mean_err, max_err);

    pipeline_destroy(p);
    buffer_destroy(x_buf);
    buffer_destroy(w_buf);
    buffer_destroy(out_buf);
    buffer_destroy(eps_buf);
    delete cb;
    context_destroy(ctx);

    return (max_err < 1e-2) ? 0 : 1;
}
