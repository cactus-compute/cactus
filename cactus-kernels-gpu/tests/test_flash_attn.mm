/* CPU-reference test for the flash_attn_decode kernel.
 *
 * Test setup mirrors a real decode call: 1 token's worth of Q, walk the
 * full KV cache. Compare against numpy-style softmax(Q K^T / sqrt(d)) @ V.
 *
 * Common shape: head_dim 128, num_query_groups 1 (MHA), no softcap, causal. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"

#include <cmath>
#include <cstdio>
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

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s metallib\n", argv[0]); return 1; }
    Context* ctx = context_create(argv[1]);
    if (!ctx) return 1;

    const uint32_t HEADS_Q = 4;
    const uint32_t HEADS_KV = 4;          // MHA — NUM_GROUPS = 1
    const uint32_t NUM_GROUPS = HEADS_Q / HEADS_KV;  // 1
    const uint32_t DK = 128;
    const uint32_t DV = 128;
    const uint32_t SEQ_K = 8;
    const uint32_t MAX_SEQ = 32;

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Q[HEADS_Q × DK], cache_k/v laid out [HEADS_KV, MAX_SEQ, DK/DV]
    std::vector<float> Qf(HEADS_Q * DK);
    std::vector<float> Kf(HEADS_KV * MAX_SEQ * DK, 0.0f);
    std::vector<float> Vf(HEADS_KV * MAX_SEQ * DV, 0.0f);
    for (auto& v : Qf) v = nd(rng) * 0.3f;
    for (uint32_t h = 0; h < HEADS_KV; ++h) {
        for (uint32_t s = 0; s < SEQ_K; ++s) {
            for (uint32_t d = 0; d < DK; ++d) Kf[(h*MAX_SEQ + s)*DK + d] = nd(rng) * 0.3f;
            for (uint32_t d = 0; d < DV; ++d) Vf[(h*MAX_SEQ + s)*DV + d] = nd(rng) * 0.3f;
        }
    }

    // Convert to fp16.
    std::vector<uint16_t> Qh(Qf.size()), Kh(Kf.size()), Vh(Vf.size());
    for (size_t i = 0; i < Qf.size(); ++i) Qh[i] = f32_to_f16(Qf[i]);
    for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = f32_to_f16(Kf[i]);
    for (size_t i = 0; i < Vf.size(); ++i) Vh[i] = f32_to_f16(Vf[i]);

    // CPU reference: softmax(QK^T / sqrt(d_k)) @ V per query head.
    std::vector<float> Oref(HEADS_Q * DV, 0.0f);
    const float scale = 1.0f / std::sqrt((float)DK);
    for (uint32_t qh = 0; qh < HEADS_Q; ++qh) {
        uint32_t kh = qh / NUM_GROUPS;
        std::vector<double> scores(SEQ_K);
        double m = -INFINITY;
        for (uint32_t s = 0; s < SEQ_K; ++s) {
            double dot = 0.0;
            for (uint32_t d = 0; d < DK; ++d) dot += (double)Qf[qh*DK + d] * (double)Kf[(kh*MAX_SEQ + s)*DK + d];
            scores[s] = dot * scale;
            if (scores[s] > m) m = scores[s];
        }
        double sum = 0.0;
        for (uint32_t s = 0; s < SEQ_K; ++s) { scores[s] = std::exp(scores[s] - m); sum += scores[s]; }
        for (uint32_t s = 0; s < SEQ_K; ++s) scores[s] /= sum;
        for (uint32_t d = 0; d < DV; ++d) {
            double acc = 0.0;
            for (uint32_t s = 0; s < SEQ_K; ++s) acc += scores[s] * (double)Vf[(kh*MAX_SEQ + s)*DV + d];
            Oref[qh*DV + d] = (float)acc;
        }
    }

    // GPU dispatch.
    Buffer* Qb = buffer_wrap_host_memory(ctx, Qh.data(), Qh.size() * 2);
    Buffer* Kb = buffer_wrap_host_memory(ctx, Kh.data(), Kh.size() * 2);
    Buffer* Vb = buffer_wrap_host_memory(ctx, Vh.data(), Vh.size() * 2);
    Buffer* Ob = buffer_create(ctx, HEADS_Q * DV * 2, StorageMode::SHARED);
    // Dummy mask buffer (zero-filled). Bind a non-null buffer so the kernel's
    // `if (mask_or_null)` check has well-defined memory to read from.
    Buffer* Mb = buffer_create(ctx, SEQ_K * 2, StorageMode::SHARED);
    std::memset(buffer_contents(Mb), 0, SEQ_K * 2);
    float s_val = scale;
    uint32_t sk = SEQ_K, ms = MAX_SEQ;
    float softcap = 0.0f;
    Buffer* s_buf  = buffer_create(ctx, 4, StorageMode::SHARED); std::memcpy(buffer_contents(s_buf),  &s_val,    4);
    Buffer* sk_buf = buffer_create(ctx, 4, StorageMode::SHARED); std::memcpy(buffer_contents(sk_buf), &sk,       4);
    Buffer* mx_buf = buffer_create(ctx, 4, StorageMode::SHARED); std::memcpy(buffer_contents(mx_buf), &ms,       4);
    Buffer* sc_buf = buffer_create(ctx, 4, StorageMode::SHARED); std::memcpy(buffer_contents(sc_buf), &softcap,  4);

    Pipeline* p = pipeline_flash_attn(ctx, DK, DV, NUM_GROUPS, /*causal*/true, /*has_softcap*/false);
    if (!p) { std::fprintf(stderr, "pipeline_flash_attn failed\n"); return 1; }

    CommandBuffer* cb = command_buffer_begin(ctx);
    BufferBinding bb[9] = {{Qb,0},{Kb,0},{Vb,0},{Mb,0},{Ob,0},{s_buf,0},{sk_buf,0},{mx_buf,0},{sc_buf,0}};
    command_buffer_dispatch(cb, p, bb, 9, HEADS_Q, 1, 1, 32, 1, 1);
    command_buffer_commit(cb);
    command_buffer_wait(cb);
    command_buffer_destroy(cb);

    // Compare.
    uint16_t* Og = (uint16_t*)buffer_contents(Ob);
    double max_err = 0, mean_err = 0, ref_max = 0;
    for (uint32_t i = 0; i < HEADS_Q * DV; ++i) {
        float g = f16_to_f32(Og[i]);
        double e = std::fabs(g - Oref[i]);
        if (e > max_err) max_err = e;
        mean_err += e;
        if (std::fabs(Oref[i]) > ref_max) ref_max = std::fabs(Oref[i]);
    }
    mean_err /= double(HEADS_Q * DV);
    std::printf("flash_attn GPU vs CPU: mean_err=%.4g max_err=%.4g ref_max=%.4g\n",
                mean_err, max_err, ref_max);
    std::printf("  O[0][0..3]   gpu=%.4f %.4f %.4f %.4f\n",
                f16_to_f32(Og[0]), f16_to_f32(Og[1]), f16_to_f32(Og[2]), f16_to_f32(Og[3]));
    std::printf("            ref=%.4f %.4f %.4f %.4f\n", Oref[0], Oref[1], Oref[2], Oref[3]);

    pipeline_destroy(p);
    buffer_destroy(Qb); buffer_destroy(Kb); buffer_destroy(Vb); buffer_destroy(Ob); buffer_destroy(Mb);
    buffer_destroy(s_buf); buffer_destroy(sk_buf); buffer_destroy(mx_buf); buffer_destroy(sc_buf);
    context_destroy(ctx);

    return (max_err < 0.05 * ref_max + 5e-3) ? 0 : 1;
}
