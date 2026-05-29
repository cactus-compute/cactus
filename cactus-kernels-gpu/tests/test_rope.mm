/* CPU-reference test for the rope_apply kernel.
 *
 * Mirrors decode_one: 1 token, fresh random Q/K, a fixed position. Compare
 * GPU rotation against numpy-style CPU reference for both NeoX (LLaMA/Gemma)
 * and interleaved (GPT-J) layouts.
 *
 * NeoX:        pairs are (q[i], q[i + head_dim/2])
 * Interleaved: pairs are (q[2i], q[2i+1])
 *   theta_i = pos * theta_base^(-2i/head_dim)
 *   q0' = q0*cos - q1*sin
 *   q1' = q0*sin + q1*cos
 */
#import <Metal/Metal.h>
#include "cactus_gpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

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

struct RopeCase {
    const char* name;
    uint32_t head_dim;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    bool     is_neox;
    float    theta;
    int32_t  position;
};

static int run_case(Context* ctx, const RopeCase& tc) {
    const uint32_t HEAD_DIM = tc.head_dim;
    const uint32_t HQ       = tc.num_q_heads;
    const uint32_t HKV      = tc.num_kv_heads;
    const bool     NEOX     = tc.is_neox;
    const float    THETA    = tc.theta;
    const int32_t  POS      = tc.position;

    std::mt19937 rng(0xC0DE + (uint32_t)tc.position);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> Qf(HQ * HEAD_DIM), Kf(HKV * HEAD_DIM);
    for (auto& v : Qf) v = nd(rng) * 0.5f;
    for (auto& v : Kf) v = nd(rng) * 0.5f;

    // CPU reference rotation. We store back into Qref/Kref in the same layout
    // and compare after the GPU pass.
    std::vector<float> Qref = Qf, Kref = Kf;
    auto rotate = [&](float* row) {
        for (uint32_t pair = 0; pair < HEAD_DIM / 2u; ++pair) {
            float exponent = -2.0f * (float)pair / (float)HEAD_DIM;
            float freq     = std::pow(THETA, exponent);
            float angle    = (float)POS * freq;
            float c = std::cos(angle), s = std::sin(angle);
            uint32_t i0, i1;
            if (NEOX) { i0 = pair;       i1 = pair + HEAD_DIM/2u; }
            else      { i0 = pair * 2u;  i1 = pair * 2u + 1u;     }
            float a = row[i0], b = row[i1];
            row[i0] = a * c - b * s;
            row[i1] = a * s + b * c;
        }
    };
    for (uint32_t h = 0; h < HQ;  ++h) rotate(&Qref[h * HEAD_DIM]);
    for (uint32_t h = 0; h < HKV; ++h) rotate(&Kref[h * HEAD_DIM]);

    // Pack fp16 Q/K and bind buffers. fp16 storage matches the kernel.
    std::vector<uint16_t> Qh(Qf.size()), Kh(Kf.size());
    for (size_t i = 0; i < Qf.size(); ++i) Qh[i] = f32_to_f16(Qf[i]);
    for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = f32_to_f16(Kf[i]);

    Buffer* qbuf = buffer_create(ctx, Qh.size() * 2, StorageMode::SHARED);
    Buffer* kbuf = buffer_create(ctx, Kh.size() * 2, StorageMode::SHARED);
    std::memcpy(buffer_contents(qbuf), Qh.data(), Qh.size() * 2);
    std::memcpy(buffer_contents(kbuf), Kh.data(), Kh.size() * 2);

    Buffer* pbuf = buffer_create(ctx, 4, StorageMode::SHARED);
    std::memcpy(buffer_contents(pbuf), &POS, 4);

    Buffer* nqbuf  = buffer_create(ctx, 4, StorageMode::SHARED);
    Buffer* nkvbuf = buffer_create(ctx, 4, StorageMode::SHARED);
    std::memcpy(buffer_contents(nqbuf),  &HQ,  4);
    std::memcpy(buffer_contents(nkvbuf), &HKV, 4);

    Pipeline* p = pipeline_rope_apply(ctx, HEAD_DIM, NEOX, THETA);
    if (!p) { std::fprintf(stderr, "[%s] pipeline_rope_apply failed\n", tc.name); return 1; }

    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[5] = {{qbuf, 0}, {kbuf, 0}, {pbuf, 0}, {nqbuf, 0}, {nkvbuf, 0}};
    uint32_t head_max = std::max(HQ, std::max(HKV, 1u));
    command_buffer_dispatch(cb, p, bb, 5, 1, head_max, 1, HEAD_DIM / 2u, 1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);
    command_buffer_destroy(cb);

    uint16_t* qgpu = (uint16_t*)buffer_contents(qbuf);
    uint16_t* kgpu = (uint16_t*)buffer_contents(kbuf);

    double q_max = 0, q_mean = 0, q_ref_max = 0;
    for (uint32_t i = 0; i < HQ * HEAD_DIM; ++i) {
        float g = f16_to_f32(qgpu[i]);
        double e = std::fabs(g - Qref[i]);
        if (e > q_max) q_max = e;
        q_mean += e;
        if (std::fabs(Qref[i]) > q_ref_max) q_ref_max = std::fabs(Qref[i]);
    }
    q_mean /= double(HQ * HEAD_DIM);

    double k_max = 0, k_mean = 0, k_ref_max = 0;
    for (uint32_t i = 0; i < HKV * HEAD_DIM; ++i) {
        float g = f16_to_f32(kgpu[i]);
        double e = std::fabs(g - Kref[i]);
        if (e > k_max) k_max = e;
        k_mean += e;
        if (std::fabs(Kref[i]) > k_ref_max) k_ref_max = std::fabs(Kref[i]);
    }
    k_mean /= double(HKV * HEAD_DIM);

    std::printf("[%s] head_dim=%u hq=%u hkv=%u neox=%d theta=%g pos=%d\n",
                tc.name, HEAD_DIM, HQ, HKV, (int)NEOX, THETA, POS);
    std::printf("    Q: mean_err=%.4g max_err=%.4g ref_max=%.4g\n", q_mean, q_max, q_ref_max);
    std::printf("    K: mean_err=%.4g max_err=%.4g ref_max=%.4g\n", k_mean, k_max, k_ref_max);
    std::printf("    Q[0][0..3] gpu=%.4f %.4f %.4f %.4f\n",
                f16_to_f32(qgpu[0]), f16_to_f32(qgpu[1]),
                f16_to_f32(qgpu[2]), f16_to_f32(qgpu[3]));
    std::printf("              ref=%.4f %.4f %.4f %.4f\n",
                Qref[0], Qref[1], Qref[2], Qref[3]);

    pipeline_destroy(p);
    buffer_destroy(qbuf); buffer_destroy(kbuf); buffer_destroy(pbuf);
    buffer_destroy(nqbuf); buffer_destroy(nkvbuf);

    // fp16 round-trip + sin/cos noise: 1% of ref_max is comfortable.
    double tol_q = 0.02 * q_ref_max + 5e-3;
    double tol_k = 0.02 * k_ref_max + 5e-3;
    return (q_max <= tol_q && k_max <= tol_k) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s metallib\n", argv[0]); return 1; }
    Context* ctx = context_create(argv[1]);
    if (!ctx) return 1;

    int rc = 0;
    // Common variants we hit in real models.
    RopeCase cases[] = {
        {"neox_dim64_mha_pos0",   64,  4, 4, true,  10000.0f, 0  },
        {"neox_dim64_mha_pos5",   64,  4, 4, true,  10000.0f, 5  },
        {"neox_dim128_mha_pos17", 128, 4, 4, true,  10000.0f, 17 },
        {"neox_dim256_gqa_pos1",  256, 8, 1, true,  10000.0f, 1  },
        {"neox_dim256_gqa_pos31", 256, 8, 1, true,  10000.0f, 31 },
        {"gptj_dim64_pos7",       64,  2, 2, false, 10000.0f, 7  },
    };
    for (const auto& tc : cases) rc |= run_case(ctx, tc);

    context_destroy(ctx);
    return rc;
}
