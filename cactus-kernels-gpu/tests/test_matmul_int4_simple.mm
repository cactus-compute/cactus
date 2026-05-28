/* Even simpler matmul test: K=128 (one group), N=4, weights all = 1, x all = 1.
 * Expected y = K = 128 for every output element. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
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
    if (argc < 2) return 1;
    Context* ctx = context_create(argv[1]);

    // K=128, N=4. One CQ4 group. Scale=1, all nibbles = 1.
    const uint32_t K = 128, N = 4;
    std::vector<uint16_t> qs(K/4 * N, 0);   // 32 × 4 uint16s
    std::vector<uint16_t> scales(1 * N, 0); // 1 group × 4 columns
    // All nibbles = 1 → packed = 0x1111. Scale = 1.0.
    for (auto& q : qs)     q = 0x1111;
    for (auto& s : scales) s = f32_to_f16(1.0f);

    std::vector<uint16_t> x_f16(K, 0);
    for (auto& v : x_f16) v = f32_to_f16(1.0f);

    Buffer* qs_b     = buffer_wrap_host_memory(ctx, qs.data(),     qs.size() * 2);
    Buffer* scales_b = buffer_wrap_host_memory(ctx, scales.data(), scales.size() * 2);
    Buffer* x_b      = buffer_wrap_host_memory(ctx, x_f16.data(),  x_f16.size() * 2);
    Buffer* y_b      = buffer_create(ctx, N * 2, StorageMode::SHARED);

    Pipeline* p = pipeline_mul_mv_int4_fp16(ctx, K, N);
    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[4] = {{qs_b,0},{scales_b,0},{x_b,0},{y_b,0}};
    command_buffer_dispatch(cb, p, bb, 4,
                            N/4, 1, 1,   // 1 threadgroup
                            32,  1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);

    uint16_t* y = (uint16_t*)buffer_contents(y_b);
    std::printf("y = [");
    for (uint32_t n = 0; n < N; ++n) std::printf("%6.2f ", f16_to_f32(y[n]));
    std::printf("]  expected: each = %u\n", K);
    // expected y[n] = sum over K of W[k,n] * x[k] = K * 1 * 1 = 128
    return 0;
}
