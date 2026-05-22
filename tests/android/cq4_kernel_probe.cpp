#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

volatile int64_t sink_i64 = 0;
volatile float sink_f32 = 0.0f;
std::string shape_filter;
std::string bench_filter;

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Shape {
    const char* name;
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
};

struct Buffers {
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t n_blocks;
    std::vector<int8_t> act_i8;
    std::vector<float> act_scales;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> tile_packed;
    std::vector<uint8_t> litert_packed;
    std::vector<int8_t> expanded;
    std::vector<int8_t> i8mm_packed;
    std::vector<float> channel_scales;
    std::vector<__fp16> norms;
    std::vector<__fp16> out;
    int8x16_t cb_lut;

    Buffers(uint32_t k, uint32_t n, uint32_t gs)
        : K(k), N(n), group_size(gs), num_groups(k / gs), n_blocks(n / 4) {
        const uint32_t pgb = gs / 2;
        const size_t panel_bytes = 4 * pgb;
        const size_t tile_group_bytes = static_cast<size_t>(gs / 16) * 8 * 16;
        const size_t expanded_panel_bytes = static_cast<size_t>(gs / 16) * 4 * 16;
        const size_t i8mm_tile_count = N / 16;
        const size_t i8mm_group_bytes = static_cast<size_t>(gs / 16) * 16 * 16;

        act_i8.resize(K);
        act_scales.resize(num_groups);
        packed.resize(static_cast<size_t>(n_blocks) * num_groups * panel_bytes);
        tile_packed.resize(static_cast<size_t>(N / 16) * num_groups * tile_group_bytes);
        litert_packed.resize(static_cast<size_t>(N) * K / 2);
        expanded.resize(static_cast<size_t>(n_blocks) * num_groups * expanded_panel_bytes);
        i8mm_packed.resize(i8mm_tile_count * num_groups * i8mm_group_bytes);
        channel_scales.resize(N);
        norms.resize(static_cast<size_t>(n_blocks) * num_groups * 4);
        out.resize(N);

        std::mt19937 gen(123);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        std::uniform_int_distribution<int> act_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.25f);
        std::uniform_int_distribution<int> index_dist(0, 15);

        for (auto& v : act_i8) v = static_cast<int8_t>(act_dist(gen));
        for (auto& v : act_scales) v = scale_dist(gen);
        for (auto& v : packed) v = static_cast<uint8_t>(byte_dist(gen));
        for (auto& v : litert_packed) v = static_cast<uint8_t>(byte_dist(gen));
        for (auto& v : channel_scales) v = scale_dist(gen);
        for (auto& v : norms) v = static_cast<__fp16>(scale_dist(gen));

        alignas(16) int8_t cb[16];
        for (int i = 0; i < 16; ++i) cb[i] = static_cast<int8_t>((i - 8) * 8);
        cb_lut = vld1q_s8(cb);

        const uint8x16_t lo_mask = vdupq_n_u8(0x0F);
        for (uint32_t nb = 0; nb < n_blocks; ++nb) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p_base = packed.data() + (static_cast<size_t>(nb) * num_groups + g) * panel_bytes;
                int8_t* e_base = expanded.data() + (static_cast<size_t>(nb) * num_groups + g) * expanded_panel_bytes;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const size_t off = static_cast<size_t>(kb / 16) * 64;
                    uint8x16_t b0 = vld1q_u8(p_base + (kb / 8 + 0) * 16);
                    uint8x16_t b1 = vld1q_u8(p_base + (kb / 8 + 1) * 16);
                    vst1q_s8(e_base + off + 0,  vqtbl1q_s8(cb_lut, vreinterpretq_s8_u8(vandq_u8(b0, lo_mask))));
                    vst1q_s8(e_base + off + 16, vqtbl1q_s8(cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(b0, 4))));
                    vst1q_s8(e_base + off + 32, vqtbl1q_s8(cb_lut, vreinterpretq_s8_u8(vandq_u8(b1, lo_mask))));
                    vst1q_s8(e_base + off + 48, vqtbl1q_s8(cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(b1, 4))));
                }
            }
        }

        for (uint32_t tile = 0; tile < N / 16; ++tile) {
            const uint32_t nb0 = tile * 4;
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p0 = packed.data() + (static_cast<size_t>(nb0 + 0) * num_groups + g) * panel_bytes;
                const uint8_t* p1 = packed.data() + (static_cast<size_t>(nb0 + 1) * num_groups + g) * panel_bytes;
                const uint8_t* p2 = packed.data() + (static_cast<size_t>(nb0 + 2) * num_groups + g) * panel_bytes;
                const uint8_t* p3 = packed.data() + (static_cast<size_t>(nb0 + 3) * num_groups + g) * panel_bytes;
                uint8_t* dst = tile_packed.data() + (static_cast<size_t>(tile) * num_groups + g) * tile_group_bytes;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    uint8_t* q = dst + static_cast<size_t>(kb / 16) * 8 * 16;
                    std::memcpy(q + 0, p0 + (kb / 8 + 0) * 16, 16);
                    std::memcpy(q + 16, p0 + (kb / 8 + 1) * 16, 16);
                    std::memcpy(q + 32, p1 + (kb / 8 + 0) * 16, 16);
                    std::memcpy(q + 48, p1 + (kb / 8 + 1) * 16, 16);
                    std::memcpy(q + 64, p2 + (kb / 8 + 0) * 16, 16);
                    std::memcpy(q + 80, p2 + (kb / 8 + 1) * 16, 16);
                    std::memcpy(q + 96, p3 + (kb / 8 + 0) * 16, 16);
                    std::memcpy(q + 112, p3 + (kb / 8 + 1) * 16, 16);
                }
            }
        }

        alignas(16) int8_t cb_scalar[16];
        vst1q_s8(cb_scalar, cb_lut);
        for (uint32_t tile = 0; tile < i8mm_tile_count; ++tile) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                int8_t* group_base = i8mm_packed.data()
                    + (static_cast<size_t>(tile) * num_groups + g) * i8mm_group_bytes;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8_t* k_base = group_base + static_cast<size_t>(kb / 16) * 16 * 16;
                    for (uint32_t pair = 0; pair < 8; ++pair) {
                        int8_t* lo = k_base + pair * 16;
                        int8_t* hi = k_base + 128 + pair * 16;
                        for (uint32_t lane = 0; lane < 8; ++lane) {
                            lo[lane] = cb_scalar[index_dist(gen)];
                            lo[lane + 8] = cb_scalar[index_dist(gen)];
                            hi[lane] = cb_scalar[index_dist(gen)];
                            hi[lane + 8] = cb_scalar[index_dist(gen)];
                        }
                    }
                }
            }
        }
    }
};

double bench_ms(const std::string& label, int iterations, const std::function<void()>& fn) {
    if (!bench_filter.empty() && bench_filter != label) return 0.0;
    fn();
    const uint64_t start = now_ns();
    for (int i = 0; i < iterations; ++i) fn();
    const double ms = static_cast<double>(now_ns() - start) / 1.0e6 / static_cast<double>(iterations);
    std::cout << label << "_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
    return ms;
}

void run_probe(const Shape& shape, int iterations) {
    if (!shape_filter.empty() && shape_filter != shape.name) return;
    Buffers b(shape.K, shape.N, shape.group_size);
    const uint32_t gs = b.group_size;
    const uint32_t num_groups = b.num_groups;
    const uint32_t pgb = gs / 2;
    const size_t panel_bytes = 4 * pgb;
    const size_t tile_group_bytes = static_cast<size_t>(gs / 16) * 8 * 16;
    const size_t expanded_panel_bytes = static_cast<size_t>(gs / 16) * 4 * 16;
    const size_t i8mm_group_bytes = static_cast<size_t>(gs / 16) * 16 * 16;
    const size_t litert_row_bytes = static_cast<size_t>(shape.K) / 2;
    const uint8x16_t lo_mask = vdupq_n_u8(0x0F);
    const int8x16_t i4_zp = vdupq_n_s8(8);

    std::cout << "shape=" << shape.name
              << " K=" << shape.K
              << " N=" << shape.N
              << " groups=" << num_groups
              << " iterations=" << iterations << "\n";

    bench_ms("packed_tbl_dot_norm", iterations, [&] {
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p_base = b.packed.data() + (static_cast<size_t>(nb) * num_groups + g) * panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t dot_a = vdupq_n_s32(0);
                int32x4_t dot_b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    uint8x16_t p0 = vld1q_u8(p_base + (kb / 8 + 0) * 16);
                    int8x16_t w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0, lo_mask)));
                    int8x16_t w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w0, a_v, 0);
                    dot_b = vdotq_laneq_s32(dot_b, w1, a_v, 1);
                    uint8x16_t p1 = vld1q_u8(p_base + (kb / 8 + 1) * 16);
                    int8x16_t w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1, lo_mask)));
                    int8x16_t w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w2, a_v, 2);
                    dot_b = vdotq_laneq_s32(dot_b, w3, a_v, 3);
                }
                int32x4_t dot = vaddq_s32(dot_a, dot_b);
                float32x4_t norm = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb) * num_groups + g) * 4));
                norm = vmulq_n_f32(norm, b.act_scales[g]);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(dot), norm);
            }
            vst1_f16(b.out.data() + nb * 4, vcvt_f16_f32(acc));
        }
        sink_f32 += static_cast<float>(b.out[0]);
    });

    bench_ms("litert_i4pc_dot4_f32", iterations, [&] {
        for (uint32_t n = 0; n < shape.N; n += 4) {
            const uint8_t* row0 = b.litert_packed.data() + static_cast<size_t>(n + 0) * litert_row_bytes;
            const uint8_t* row1 = b.litert_packed.data() + static_cast<size_t>(n + 1) * litert_row_bytes;
            const uint8_t* row2 = b.litert_packed.data() + static_cast<size_t>(n + 2) * litert_row_bytes;
            const uint8_t* row3 = b.litert_packed.data() + static_cast<size_t>(n + 3) * litert_row_bytes;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            for (uint32_t kb = 0; kb < shape.K; kb += 32) {
                int8x16x2_t a = vuzpq_s8(vld1q_s8(b.act_i8.data() + kb), vld1q_s8(b.act_i8.data() + kb + 16));
                uint8x16_t raw0 = vld1q_u8(row0 + kb / 2);
                uint8x16_t raw1 = vld1q_u8(row1 + kb / 2);
                uint8x16_t raw2 = vld1q_u8(row2 + kb / 2);
                uint8x16_t raw3 = vld1q_u8(row3 + kb / 2);
                int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, lo_mask)), i4_zp);
                int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), i4_zp);
                int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, lo_mask)), i4_zp);
                int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), i4_zp);
                int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw2, lo_mask)), i4_zp);
                int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw2, 4)), i4_zp);
                int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw3, lo_mask)), i4_zp);
                int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw3, 4)), i4_zp);
                acc0 = vdotq_s32(acc0, w0l, a.val[0]);
                acc0 = vdotq_s32(acc0, w0h, a.val[1]);
                acc1 = vdotq_s32(acc1, w1l, a.val[0]);
                acc1 = vdotq_s32(acc1, w1h, a.val[1]);
                acc2 = vdotq_s32(acc2, w2l, a.val[0]);
                acc2 = vdotq_s32(acc2, w2h, a.val[1]);
                acc3 = vdotq_s32(acc3, w3l, a.val[0]);
                acc3 = vdotq_s32(acc3, w3h, a.val[1]);
            }
            b.out[n + 0] = static_cast<__fp16>(static_cast<float>(vaddvq_s32(acc0)) * b.channel_scales[n + 0]);
            b.out[n + 1] = static_cast<__fp16>(static_cast<float>(vaddvq_s32(acc1)) * b.channel_scales[n + 1]);
            b.out[n + 2] = static_cast<__fp16>(static_cast<float>(vaddvq_s32(acc2)) * b.channel_scales[n + 2]);
            b.out[n + 3] = static_cast<__fp16>(static_cast<float>(vaddvq_s32(acc3)) * b.channel_scales[n + 3]);
        }
        sink_f32 += static_cast<float>(b.out[0]);
    });

    bench_ms("litert_i4pc_dot4_int_only", iterations, [&] {
        int32x4_t acc_all = vdupq_n_s32(0);
        for (uint32_t n = 0; n < shape.N; n += 4) {
            const uint8_t* row0 = b.litert_packed.data() + static_cast<size_t>(n + 0) * litert_row_bytes;
            const uint8_t* row1 = b.litert_packed.data() + static_cast<size_t>(n + 1) * litert_row_bytes;
            const uint8_t* row2 = b.litert_packed.data() + static_cast<size_t>(n + 2) * litert_row_bytes;
            const uint8_t* row3 = b.litert_packed.data() + static_cast<size_t>(n + 3) * litert_row_bytes;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            for (uint32_t kb = 0; kb < shape.K; kb += 32) {
                int8x16x2_t a = vuzpq_s8(vld1q_s8(b.act_i8.data() + kb), vld1q_s8(b.act_i8.data() + kb + 16));
                uint8x16_t raw0 = vld1q_u8(row0 + kb / 2);
                uint8x16_t raw1 = vld1q_u8(row1 + kb / 2);
                uint8x16_t raw2 = vld1q_u8(row2 + kb / 2);
                uint8x16_t raw3 = vld1q_u8(row3 + kb / 2);
                int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, lo_mask)), i4_zp);
                int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), i4_zp);
                int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, lo_mask)), i4_zp);
                int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), i4_zp);
                int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw2, lo_mask)), i4_zp);
                int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw2, 4)), i4_zp);
                int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw3, lo_mask)), i4_zp);
                int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw3, 4)), i4_zp);
                acc0 = vdotq_s32(acc0, w0l, a.val[0]);
                acc0 = vdotq_s32(acc0, w0h, a.val[1]);
                acc1 = vdotq_s32(acc1, w1l, a.val[0]);
                acc1 = vdotq_s32(acc1, w1h, a.val[1]);
                acc2 = vdotq_s32(acc2, w2l, a.val[0]);
                acc2 = vdotq_s32(acc2, w2h, a.val[1]);
                acc3 = vdotq_s32(acc3, w3l, a.val[0]);
                acc3 = vdotq_s32(acc3, w3h, a.val[1]);
            }
            acc_all = vaddq_s32(acc_all, acc0);
            acc_all = vaddq_s32(acc_all, acc1);
            acc_all = vaddq_s32(acc_all, acc2);
            acc_all = vaddq_s32(acc_all, acc3);
        }
        sink_i64 += vaddvq_s32(acc_all);
    });

    bench_ms("packed_pair_tbl_dot_norm", iterations, [&] {
        uint32_t nb = 0;
        for (; nb + 2 <= b.n_blocks; nb += 2) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p0 = b.packed.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * panel_bytes;
                const uint8_t* p1 = b.packed.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t d0a = vdupq_n_s32(0);
                int32x4_t d0b = vdupq_n_s32(0);
                int32x4_t d1a = vdupq_n_s32(0);
                int32x4_t d1b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8x16_t a_v = vld1q_s8(a_grp + kb);

                    uint8x16_t p0b0 = vld1q_u8(p0 + (kb / 8 + 0) * 16);
                    int8x16_t p0w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b0, lo_mask)));
                    int8x16_t p0w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b0, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w0, a_v, 0);
                    d0b = vdotq_laneq_s32(d0b, p0w1, a_v, 1);
                    uint8x16_t p0b1 = vld1q_u8(p0 + (kb / 8 + 1) * 16);
                    int8x16_t p0w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b1, lo_mask)));
                    int8x16_t p0w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b1, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w2, a_v, 2);
                    d0b = vdotq_laneq_s32(d0b, p0w3, a_v, 3);

                    uint8x16_t p1b0 = vld1q_u8(p1 + (kb / 8 + 0) * 16);
                    int8x16_t p1w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b0, lo_mask)));
                    int8x16_t p1w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b0, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w0, a_v, 0);
                    d1b = vdotq_laneq_s32(d1b, p1w1, a_v, 1);
                    uint8x16_t p1b1 = vld1q_u8(p1 + (kb / 8 + 1) * 16);
                    int8x16_t p1w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b1, lo_mask)));
                    int8x16_t p1w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b1, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w2, a_v, 2);
                    d1b = vdotq_laneq_s32(d1b, p1w3, a_v, 3);
                }
                int32x4_t dot0 = vaddq_s32(d0a, d0b);
                int32x4_t dot1 = vaddq_s32(d1a, d1b);
                float32x4_t norm0 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * 4));
                float32x4_t norm1 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * 4));
                norm0 = vmulq_n_f32(norm0, b.act_scales[g]);
                norm1 = vmulq_n_f32(norm1, b.act_scales[g]);
                acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(dot0), norm0);
                acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(dot1), norm1);
            }
            vst1_f16(b.out.data() + (nb + 0) * 4, vcvt_f16_f32(acc0));
            vst1_f16(b.out.data() + (nb + 1) * 4, vcvt_f16_f32(acc1));
        }
        sink_f32 += static_cast<float>(b.out[2]);
    });

    bench_ms("packed_tbl_x16_dot_norm", iterations, [&] {
        for (uint32_t tile = 0; tile < shape.N / 16; ++tile) {
            const uint32_t nb0 = tile * 4;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p0 = b.packed.data() + (static_cast<size_t>(nb0 + 0) * num_groups + g) * panel_bytes;
                const uint8_t* p1 = b.packed.data() + (static_cast<size_t>(nb0 + 1) * num_groups + g) * panel_bytes;
                const uint8_t* p2 = b.packed.data() + (static_cast<size_t>(nb0 + 2) * num_groups + g) * panel_bytes;
                const uint8_t* p3 = b.packed.data() + (static_cast<size_t>(nb0 + 3) * num_groups + g) * panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t d0a = vdupq_n_s32(0);
                int32x4_t d0b = vdupq_n_s32(0);
                int32x4_t d1a = vdupq_n_s32(0);
                int32x4_t d1b = vdupq_n_s32(0);
                int32x4_t d2a = vdupq_n_s32(0);
                int32x4_t d2b = vdupq_n_s32(0);
                int32x4_t d3a = vdupq_n_s32(0);
                int32x4_t d3b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8x16_t a_v = vld1q_s8(a_grp + kb);

                    uint8x16_t p0b0 = vld1q_u8(p0 + (kb / 8 + 0) * 16);
                    int8x16_t p0w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b0, lo_mask)));
                    int8x16_t p0w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b0, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w0, a_v, 0);
                    d0b = vdotq_laneq_s32(d0b, p0w1, a_v, 1);
                    uint8x16_t p0b1 = vld1q_u8(p0 + (kb / 8 + 1) * 16);
                    int8x16_t p0w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b1, lo_mask)));
                    int8x16_t p0w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b1, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w2, a_v, 2);
                    d0b = vdotq_laneq_s32(d0b, p0w3, a_v, 3);

                    uint8x16_t p1b0 = vld1q_u8(p1 + (kb / 8 + 0) * 16);
                    int8x16_t p1w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b0, lo_mask)));
                    int8x16_t p1w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b0, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w0, a_v, 0);
                    d1b = vdotq_laneq_s32(d1b, p1w1, a_v, 1);
                    uint8x16_t p1b1 = vld1q_u8(p1 + (kb / 8 + 1) * 16);
                    int8x16_t p1w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b1, lo_mask)));
                    int8x16_t p1w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b1, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w2, a_v, 2);
                    d1b = vdotq_laneq_s32(d1b, p1w3, a_v, 3);

                    uint8x16_t p2b0 = vld1q_u8(p2 + (kb / 8 + 0) * 16);
                    int8x16_t p2w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p2b0, lo_mask)));
                    int8x16_t p2w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p2b0, 4)));
                    d2a = vdotq_laneq_s32(d2a, p2w0, a_v, 0);
                    d2b = vdotq_laneq_s32(d2b, p2w1, a_v, 1);
                    uint8x16_t p2b1 = vld1q_u8(p2 + (kb / 8 + 1) * 16);
                    int8x16_t p2w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p2b1, lo_mask)));
                    int8x16_t p2w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p2b1, 4)));
                    d2a = vdotq_laneq_s32(d2a, p2w2, a_v, 2);
                    d2b = vdotq_laneq_s32(d2b, p2w3, a_v, 3);

                    uint8x16_t p3b0 = vld1q_u8(p3 + (kb / 8 + 0) * 16);
                    int8x16_t p3w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p3b0, lo_mask)));
                    int8x16_t p3w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p3b0, 4)));
                    d3a = vdotq_laneq_s32(d3a, p3w0, a_v, 0);
                    d3b = vdotq_laneq_s32(d3b, p3w1, a_v, 1);
                    uint8x16_t p3b1 = vld1q_u8(p3 + (kb / 8 + 1) * 16);
                    int8x16_t p3w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p3b1, lo_mask)));
                    int8x16_t p3w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p3b1, 4)));
                    d3a = vdotq_laneq_s32(d3a, p3w2, a_v, 2);
                    d3b = vdotq_laneq_s32(d3b, p3w3, a_v, 3);
                }
                const float scale_grp = b.act_scales[g];
                float32x4_t n0 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 0) * num_groups + g) * 4));
                float32x4_t n1 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 1) * num_groups + g) * 4));
                float32x4_t n2 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 2) * num_groups + g) * 4));
                float32x4_t n3 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 3) * num_groups + g) * 4));
                acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(vaddq_s32(d0a, d0b)), vmulq_n_f32(n0, scale_grp));
                acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(vaddq_s32(d1a, d1b)), vmulq_n_f32(n1, scale_grp));
                acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(vaddq_s32(d2a, d2b)), vmulq_n_f32(n2, scale_grp));
                acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(vaddq_s32(d3a, d3b)), vmulq_n_f32(n3, scale_grp));
            }
            vst1_f16(b.out.data() + nb0 * 4 + 0, vcvt_f16_f32(acc0));
            vst1_f16(b.out.data() + nb0 * 4 + 4, vcvt_f16_f32(acc1));
            vst1_f16(b.out.data() + nb0 * 4 + 8, vcvt_f16_f32(acc2));
            vst1_f16(b.out.data() + nb0 * 4 + 12, vcvt_f16_f32(acc3));
        }
        sink_f32 += static_cast<float>(b.out[6]);
    });

    bench_ms("packed_tbl_x16_tilepack_dot_norm", iterations, [&] {
        for (uint32_t tile = 0; tile < shape.N / 16; ++tile) {
            const uint32_t nb0 = tile * 4;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* tp = b.tile_packed.data() + (static_cast<size_t>(tile) * num_groups + g) * tile_group_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t d0a = vdupq_n_s32(0);
                int32x4_t d0b = vdupq_n_s32(0);
                int32x4_t d1a = vdupq_n_s32(0);
                int32x4_t d1b = vdupq_n_s32(0);
                int32x4_t d2a = vdupq_n_s32(0);
                int32x4_t d2b = vdupq_n_s32(0);
                int32x4_t d3a = vdupq_n_s32(0);
                int32x4_t d3b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const uint8_t* q = tp + static_cast<size_t>(kb / 16) * 8 * 16;
                    int8x16_t a_v = vld1q_s8(a_grp + kb);

                    uint8x16_t p0b0 = vld1q_u8(q + 0);
                    int8x16_t p0w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b0, lo_mask)));
                    int8x16_t p0w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b0, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w0, a_v, 0);
                    d0b = vdotq_laneq_s32(d0b, p0w1, a_v, 1);
                    uint8x16_t p0b1 = vld1q_u8(q + 16);
                    int8x16_t p0w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b1, lo_mask)));
                    int8x16_t p0w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b1, 4)));
                    d0a = vdotq_laneq_s32(d0a, p0w2, a_v, 2);
                    d0b = vdotq_laneq_s32(d0b, p0w3, a_v, 3);

                    uint8x16_t p1b0 = vld1q_u8(q + 32);
                    int8x16_t p1w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b0, lo_mask)));
                    int8x16_t p1w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b0, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w0, a_v, 0);
                    d1b = vdotq_laneq_s32(d1b, p1w1, a_v, 1);
                    uint8x16_t p1b1 = vld1q_u8(q + 48);
                    int8x16_t p1w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b1, lo_mask)));
                    int8x16_t p1w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b1, 4)));
                    d1a = vdotq_laneq_s32(d1a, p1w2, a_v, 2);
                    d1b = vdotq_laneq_s32(d1b, p1w3, a_v, 3);

                    uint8x16_t p2b0 = vld1q_u8(q + 64);
                    int8x16_t p2w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p2b0, lo_mask)));
                    int8x16_t p2w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p2b0, 4)));
                    d2a = vdotq_laneq_s32(d2a, p2w0, a_v, 0);
                    d2b = vdotq_laneq_s32(d2b, p2w1, a_v, 1);
                    uint8x16_t p2b1 = vld1q_u8(q + 80);
                    int8x16_t p2w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p2b1, lo_mask)));
                    int8x16_t p2w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p2b1, 4)));
                    d2a = vdotq_laneq_s32(d2a, p2w2, a_v, 2);
                    d2b = vdotq_laneq_s32(d2b, p2w3, a_v, 3);

                    uint8x16_t p3b0 = vld1q_u8(q + 96);
                    int8x16_t p3w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p3b0, lo_mask)));
                    int8x16_t p3w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p3b0, 4)));
                    d3a = vdotq_laneq_s32(d3a, p3w0, a_v, 0);
                    d3b = vdotq_laneq_s32(d3b, p3w1, a_v, 1);
                    uint8x16_t p3b1 = vld1q_u8(q + 112);
                    int8x16_t p3w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p3b1, lo_mask)));
                    int8x16_t p3w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p3b1, 4)));
                    d3a = vdotq_laneq_s32(d3a, p3w2, a_v, 2);
                    d3b = vdotq_laneq_s32(d3b, p3w3, a_v, 3);
                }
                const float scale_grp = b.act_scales[g];
                float32x4_t n0 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 0) * num_groups + g) * 4));
                float32x4_t n1 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 1) * num_groups + g) * 4));
                float32x4_t n2 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 2) * num_groups + g) * 4));
                float32x4_t n3 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb0 + 3) * num_groups + g) * 4));
                acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(vaddq_s32(d0a, d0b)), vmulq_n_f32(n0, scale_grp));
                acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(vaddq_s32(d1a, d1b)), vmulq_n_f32(n1, scale_grp));
                acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(vaddq_s32(d2a, d2b)), vmulq_n_f32(n2, scale_grp));
                acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(vaddq_s32(d3a, d3b)), vmulq_n_f32(n3, scale_grp));
            }
            vst1_f16(b.out.data() + nb0 * 4 + 0, vcvt_f16_f32(acc0));
            vst1_f16(b.out.data() + nb0 * 4 + 4, vcvt_f16_f32(acc1));
            vst1_f16(b.out.data() + nb0 * 4 + 8, vcvt_f16_f32(acc2));
            vst1_f16(b.out.data() + nb0 * 4 + 12, vcvt_f16_f32(acc3));
        }
        sink_f32 += static_cast<float>(b.out[6]);
    });

    bench_ms("packed_pair_serial_dot_norm", iterations, [&] {
        uint32_t nb = 0;
        for (; nb + 2 <= b.n_blocks; nb += 2) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p_base = b.packed.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t dot_a = vdupq_n_s32(0);
                int32x4_t dot_b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    uint8x16_t p0 = vld1q_u8(p_base + (kb / 8 + 0) * 16);
                    int8x16_t w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0, lo_mask)));
                    int8x16_t w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w0, a_v, 0);
                    dot_b = vdotq_laneq_s32(dot_b, w1, a_v, 1);
                    uint8x16_t p1 = vld1q_u8(p_base + (kb / 8 + 1) * 16);
                    int8x16_t w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1, lo_mask)));
                    int8x16_t w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w2, a_v, 2);
                    dot_b = vdotq_laneq_s32(dot_b, w3, a_v, 3);
                }
                int32x4_t dot = vaddq_s32(dot_a, dot_b);
                float32x4_t norm = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * 4));
                norm = vmulq_n_f32(norm, b.act_scales[g]);
                acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(dot), norm);
            }

            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p_base = b.packed.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t dot_a = vdupq_n_s32(0);
                int32x4_t dot_b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    uint8x16_t p0 = vld1q_u8(p_base + (kb / 8 + 0) * 16);
                    int8x16_t w0 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0, lo_mask)));
                    int8x16_t w1 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w0, a_v, 0);
                    dot_b = vdotq_laneq_s32(dot_b, w1, a_v, 1);
                    uint8x16_t p1 = vld1q_u8(p_base + (kb / 8 + 1) * 16);
                    int8x16_t w2 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1, lo_mask)));
                    int8x16_t w3 = vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)));
                    dot_a = vdotq_laneq_s32(dot_a, w2, a_v, 2);
                    dot_b = vdotq_laneq_s32(dot_b, w3, a_v, 3);
                }
                int32x4_t dot = vaddq_s32(dot_a, dot_b);
                float32x4_t norm = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * 4));
                norm = vmulq_n_f32(norm, b.act_scales[g]);
                acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(dot), norm);
            }
            vst1_f16(b.out.data() + (nb + 0) * 4, vcvt_f16_f32(acc0));
            vst1_f16(b.out.data() + (nb + 1) * 4, vcvt_f16_f32(acc1));
        }
        sink_f32 += static_cast<float>(b.out[4]);
    });

    bench_ms("packed_tbl_only", iterations, [&] {
        int32x4_t acc = vdupq_n_s32(0);
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p_base = b.packed.data() + (static_cast<size_t>(nb) * num_groups + g) * panel_bytes;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    uint8x16_t p0 = vld1q_u8(p_base + (kb / 8 + 0) * 16);
                    uint8x16_t p1 = vld1q_u8(p_base + (kb / 8 + 1) * 16);
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)))));
                }
            }
        }
        sink_i64 += vaddvq_s32(acc);
    });

    bench_ms("packed_pair_tbl_only", iterations, [&] {
        int32x4_t acc = vdupq_n_s32(0);
        uint32_t nb = 0;
        for (; nb + 2 <= b.n_blocks; nb += 2) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint8_t* p0 = b.packed.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * panel_bytes;
                const uint8_t* p1 = b.packed.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * panel_bytes;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    uint8x16_t p0b0 = vld1q_u8(p0 + (kb / 8 + 0) * 16);
                    uint8x16_t p0b1 = vld1q_u8(p0 + (kb / 8 + 1) * 16);
                    uint8x16_t p1b0 = vld1q_u8(p1 + (kb / 8 + 0) * 16);
                    uint8x16_t p1b1 = vld1q_u8(p1 + (kb / 8 + 1) * 16);
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b0, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b0, 4)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p0b1, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p0b1, 4)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b0, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b0, 4)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vandq_u8(p1b1, lo_mask)))));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vqtbl1q_s8(b.cb_lut, vreinterpretq_s8_u8(vshrq_n_u8(p1b1, 4)))));
                }
            }
        }
        sink_i64 += vaddvq_s32(acc);
    });

    bench_ms("expanded_dot_norm", iterations, [&] {
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const int8_t* e_base = b.expanded.data() + (static_cast<size_t>(nb) * num_groups + g) * expanded_panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t dot_a = vdupq_n_s32(0);
                int32x4_t dot_b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const size_t off = static_cast<size_t>(kb / 16) * 64;
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    dot_a = vdotq_laneq_s32(dot_a, vld1q_s8(e_base + off + 0),  a_v, 0);
                    dot_b = vdotq_laneq_s32(dot_b, vld1q_s8(e_base + off + 16), a_v, 1);
                    dot_a = vdotq_laneq_s32(dot_a, vld1q_s8(e_base + off + 32), a_v, 2);
                    dot_b = vdotq_laneq_s32(dot_b, vld1q_s8(e_base + off + 48), a_v, 3);
                }
                int32x4_t dot = vaddq_s32(dot_a, dot_b);
                float32x4_t norm = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb) * num_groups + g) * 4));
                norm = vmulq_n_f32(norm, b.act_scales[g]);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(dot), norm);
            }
            vst1_f16(b.out.data() + nb * 4, vcvt_f16_f32(acc));
        }
        sink_f32 += static_cast<float>(b.out[1]);
    });

    bench_ms("expanded_pair_dot_norm", iterations, [&] {
        uint32_t nb = 0;
        for (; nb + 2 <= b.n_blocks; nb += 2) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const int8_t* e0 = b.expanded.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * expanded_panel_bytes;
                const int8_t* e1 = b.expanded.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * expanded_panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t d0a = vdupq_n_s32(0);
                int32x4_t d0b = vdupq_n_s32(0);
                int32x4_t d1a = vdupq_n_s32(0);
                int32x4_t d1b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const size_t off = static_cast<size_t>(kb / 16) * 64;
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    d0a = vdotq_laneq_s32(d0a, vld1q_s8(e0 + off + 0),  a_v, 0);
                    d0b = vdotq_laneq_s32(d0b, vld1q_s8(e0 + off + 16), a_v, 1);
                    d0a = vdotq_laneq_s32(d0a, vld1q_s8(e0 + off + 32), a_v, 2);
                    d0b = vdotq_laneq_s32(d0b, vld1q_s8(e0 + off + 48), a_v, 3);
                    d1a = vdotq_laneq_s32(d1a, vld1q_s8(e1 + off + 0),  a_v, 0);
                    d1b = vdotq_laneq_s32(d1b, vld1q_s8(e1 + off + 16), a_v, 1);
                    d1a = vdotq_laneq_s32(d1a, vld1q_s8(e1 + off + 32), a_v, 2);
                    d1b = vdotq_laneq_s32(d1b, vld1q_s8(e1 + off + 48), a_v, 3);
                }
                int32x4_t dot0 = vaddq_s32(d0a, d0b);
                int32x4_t dot1 = vaddq_s32(d1a, d1b);
                float32x4_t norm0 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * 4));
                float32x4_t norm1 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * 4));
                norm0 = vmulq_n_f32(norm0, b.act_scales[g]);
                norm1 = vmulq_n_f32(norm1, b.act_scales[g]);
                acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(dot0), norm0);
                acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(dot1), norm1);
            }
            vst1_f16(b.out.data() + (nb + 0) * 4, vcvt_f16_f32(acc0));
            vst1_f16(b.out.data() + (nb + 1) * 4, vcvt_f16_f32(acc1));
        }
        sink_f32 += static_cast<float>(b.out[3]);
    });

    bench_ms("expanded_i8mm_x16_dot_norm", iterations, [&] {
        auto load8 = [](const int8_t* p) {
            return vreinterpretq_s8_u64(vld1q_lane_u64(reinterpret_cast<const uint64_t*>(p), vdupq_n_u64(0), 0));
        };
        for (uint32_t tile = 0; tile < shape.N / 16; ++tile) {
            float32x4_t acc0123 = vdupq_n_f32(0.0f);
            float32x4_t acc4567 = vdupq_n_f32(0.0f);
            float32x4_t acc89ab = vdupq_n_f32(0.0f);
            float32x4_t acccdef = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                const int8_t* w_base = b.i8mm_packed.data()
                    + (static_cast<size_t>(tile) * num_groups + g) * i8mm_group_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;

                int32x4_t v01 = vdupq_n_s32(0);
                int32x4_t v23 = vdupq_n_s32(0);
                int32x4_t v45 = vdupq_n_s32(0);
                int32x4_t v67 = vdupq_n_s32(0);
                int32x4_t v89 = vdupq_n_s32(0);
                int32x4_t vab = vdupq_n_s32(0);
                int32x4_t vcd = vdupq_n_s32(0);
                int32x4_t vef = vdupq_n_s32(0);

                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const int8_t* wk = w_base + static_cast<size_t>(kb / 16) * 16 * 16;
                    int8x16_t a_lo = load8(a_grp + kb);
                    int8x16_t a_hi = load8(a_grp + kb + 8);

                    v01 = vmmlaq_s32(v01, a_lo, vld1q_s8(wk + 0));
                    v23 = vmmlaq_s32(v23, a_lo, vld1q_s8(wk + 16));
                    v45 = vmmlaq_s32(v45, a_lo, vld1q_s8(wk + 32));
                    v67 = vmmlaq_s32(v67, a_lo, vld1q_s8(wk + 48));
                    v89 = vmmlaq_s32(v89, a_lo, vld1q_s8(wk + 64));
                    vab = vmmlaq_s32(vab, a_lo, vld1q_s8(wk + 80));
                    vcd = vmmlaq_s32(vcd, a_lo, vld1q_s8(wk + 96));
                    vef = vmmlaq_s32(vef, a_lo, vld1q_s8(wk + 112));
                    v01 = vmmlaq_s32(v01, a_hi, vld1q_s8(wk + 128));
                    v23 = vmmlaq_s32(v23, a_hi, vld1q_s8(wk + 144));
                    v45 = vmmlaq_s32(v45, a_hi, vld1q_s8(wk + 160));
                    v67 = vmmlaq_s32(v67, a_hi, vld1q_s8(wk + 176));
                    v89 = vmmlaq_s32(v89, a_hi, vld1q_s8(wk + 192));
                    vab = vmmlaq_s32(vab, a_hi, vld1q_s8(wk + 208));
                    vcd = vmmlaq_s32(vcd, a_hi, vld1q_s8(wk + 224));
                    vef = vmmlaq_s32(vef, a_hi, vld1q_s8(wk + 240));
                }

                int32x4_t dot0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(v01), vreinterpretq_u64_s32(v23)));
                int32x4_t dot4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(v45), vreinterpretq_u64_s32(v67)));
                int32x4_t dot89ab = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(v89), vreinterpretq_u64_s32(vab)));
                int32x4_t dotcdef = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vcd), vreinterpretq_u64_s32(vef)));
                const uint32_t nb = tile * 4;
                const float scale_grp = b.act_scales[g];
                float32x4_t n0123 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 0) * num_groups + g) * 4));
                float32x4_t n4567 = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 1) * num_groups + g) * 4));
                float32x4_t n89ab = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 2) * num_groups + g) * 4));
                float32x4_t ncdef = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb + 3) * num_groups + g) * 4));
                acc0123 = vfmaq_f32(acc0123, vcvtq_f32_s32(dot0123), vmulq_n_f32(n0123, scale_grp));
                acc4567 = vfmaq_f32(acc4567, vcvtq_f32_s32(dot4567), vmulq_n_f32(n4567, scale_grp));
                acc89ab = vfmaq_f32(acc89ab, vcvtq_f32_s32(dot89ab), vmulq_n_f32(n89ab, scale_grp));
                acccdef = vfmaq_f32(acccdef, vcvtq_f32_s32(dotcdef), vmulq_n_f32(ncdef, scale_grp));
            }
            vst1_f16(b.out.data() + tile * 16 + 0, vcvt_f16_f32(acc0123));
            vst1_f16(b.out.data() + tile * 16 + 4, vcvt_f16_f32(acc4567));
            vst1_f16(b.out.data() + tile * 16 + 8, vcvt_f16_f32(acc89ab));
            vst1_f16(b.out.data() + tile * 16 + 12, vcvt_f16_f32(acccdef));
        }
        sink_f32 += static_cast<float>(b.out[5]);
    });

    bench_ms("expanded_dot_int_only", iterations, [&] {
        int32x4_t acc_all = vdupq_n_s32(0);
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                const int8_t* e_base = b.expanded.data() + (static_cast<size_t>(nb) * num_groups + g) * expanded_panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                int32x4_t dot_a = vdupq_n_s32(0);
                int32x4_t dot_b = vdupq_n_s32(0);
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const size_t off = static_cast<size_t>(kb / 16) * 64;
                    int8x16_t a_v = vld1q_s8(a_grp + kb);
                    dot_a = vdotq_laneq_s32(dot_a, vld1q_s8(e_base + off + 0),  a_v, 0);
                    dot_b = vdotq_laneq_s32(dot_b, vld1q_s8(e_base + off + 16), a_v, 1);
                    dot_a = vdotq_laneq_s32(dot_a, vld1q_s8(e_base + off + 32), a_v, 2);
                    dot_b = vdotq_laneq_s32(dot_b, vld1q_s8(e_base + off + 48), a_v, 3);
                }
                acc_all = vaddq_s32(acc_all, vaddq_s32(dot_a, dot_b));
            }
        }
        sink_i64 += vaddvq_s32(acc_all);
    });

    bench_ms("expanded_load_only", iterations, [&] {
        int32x4_t acc = vdupq_n_s32(0);
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            for (uint32_t g = 0; g < num_groups; ++g) {
                const int8_t* e_base = b.expanded.data() + (static_cast<size_t>(nb) * num_groups + g) * expanded_panel_bytes;
                const int8_t* a_grp = b.act_i8.data() + static_cast<size_t>(g) * gs;
                for (uint32_t kb = 0; kb < gs; kb += 16) {
                    const size_t off = static_cast<size_t>(kb / 16) * 64;
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vld1q_s8(a_grp + kb)));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vld1q_s8(e_base + off + 0)));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vld1q_s8(e_base + off + 16)));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vld1q_s8(e_base + off + 32)));
                    acc = vaddq_s32(acc, vreinterpretq_s32_s8(vld1q_s8(e_base + off + 48)));
                }
            }
        }
        sink_i64 += vaddvq_s32(acc);
    });

    bench_ms("norm_fma_only", iterations, [&] {
        float32x4_t acc_all = vdupq_n_f32(0.0f);
        int32x4_t dot = vdupq_n_s32(12345);
        for (uint32_t nb = 0; nb < b.n_blocks; ++nb) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (uint32_t g = 0; g < num_groups; ++g) {
                float32x4_t norm = vcvt_f32_f16(vld1_f16(b.norms.data() + (static_cast<size_t>(nb) * num_groups + g) * 4));
                norm = vmulq_n_f32(norm, b.act_scales[g]);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(dot), norm);
            }
            acc_all = vaddq_f32(acc_all, acc);
        }
        sink_f32 += vaddvq_f32(acc_all);
    });

    bench_ms("vdot_register_pressure", iterations * 50, [&] {
        int8x16_t a0 = vld1q_s8(b.act_i8.data());
        int8x16_t w0 = vld1q_s8(b.expanded.data());
        int8x16_t w1 = vld1q_s8(b.expanded.data() + 16);
        int8x16_t w2 = vld1q_s8(b.expanded.data() + 32);
        int8x16_t w3 = vld1q_s8(b.expanded.data() + 48);
        int32x4_t d0 = vdupq_n_s32(0);
        int32x4_t d1 = vdupq_n_s32(0);
        for (int i = 0; i < 4096; ++i) {
            d0 = vdotq_laneq_s32(d0, w0, a0, 0);
            d1 = vdotq_laneq_s32(d1, w1, a0, 1);
            d0 = vdotq_laneq_s32(d0, w2, a0, 2);
            d1 = vdotq_laneq_s32(d1, w3, a0, 3);
        }
        sink_i64 += vaddvq_s32(vaddq_s32(d0, d1));
    });
}

struct OrthogonalBuffers {
    uint32_t K = 1536;
    uint32_t N = 262144;
    uint32_t pgb = K / 2;
    std::vector<uint8_t> packed;
    std::vector<__fp16> ar;
    std::vector<float> ar32;
    std::vector<int8_t> act_i8;
    std::vector<int8_t> qd8_qc8_weights;
    std::vector<float> qd8_qc8_scales;
    std::vector<__fp16> norms;
    std::vector<__fp16> out;
    std::vector<float> out_f32;
    uint8x16_t cb_lo_tbl;
    uint8x16_t cb_hi_tbl;

    OrthogonalBuffers() {
        packed.resize(static_cast<size_t>(N) * pgb);
        ar.resize(K);
        ar32.resize(K);
        act_i8.resize(K);
        qd8_qc8_weights.resize(static_cast<size_t>(N) * K);
        qd8_qc8_scales.resize(N);
        norms.resize(N);
        out.resize(N);
        out_f32.resize(N);

        std::mt19937 gen(123);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        std::uniform_int_distribution<int> i8_dist(-127, 127);
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> scale_dist(0.00002f, 0.00012f);
        for (auto& v : packed) v = static_cast<uint8_t>(byte_dist(gen));
        for (auto& v : ar) v = static_cast<__fp16>(value_dist(gen));
        for (uint32_t i = 0; i < K; ++i) ar32[i] = static_cast<float>(ar[i]);
        for (auto& v : act_i8) v = static_cast<int8_t>(i8_dist(gen));
        for (auto& v : qd8_qc8_weights) v = static_cast<int8_t>(i8_dist(gen));
        for (auto& v : qd8_qc8_scales) v = scale_dist(gen);
        for (auto& v : norms) v = static_cast<__fp16>(0.05f + 0.01f * value_dist(gen));

        __fp16 cb_f16[16];
        uint8_t cb_lo_arr[16];
        uint8_t cb_hi_arr[16];
        for (uint32_t c = 0; c < 16; ++c) {
            cb_f16[c] = static_cast<__fp16>((static_cast<int>(c) - 8) * 0.125f);
            cb_lo_arr[c] = reinterpret_cast<const uint8_t*>(&cb_f16[c])[0];
            cb_hi_arr[c] = reinterpret_cast<const uint8_t*>(&cb_f16[c])[1];
        }
        cb_lo_tbl = vld1q_u8(cb_lo_arr);
        cb_hi_tbl = vld1q_u8(cb_hi_arr);
    }
};

void run_orthogonal_probe(int iterations) {
    if (shape_filter != "gemma_orthogonal") return;
    OrthogonalBuffers b;

    std::cout << "shape=gemma_orthogonal"
              << " K=" << b.K
              << " N=" << b.N
              << " iterations=" << iterations << "\n";

    auto decode_packed = [&](uint8x8_t raw8, float16x8_t& cb_lo, float16x8_t& cb_hi) {
        uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
        uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
        uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
        uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);
        uint8x16_t lo_bytes = vqtbl1q_u8(b.cb_lo_tbl, indices16);
        uint8x16_t hi_bytes = vqtbl1q_u8(b.cb_hi_tbl, indices16);
        uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
        cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
        cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);
    };

    bench_ms("orthogonal_dot_full", iterations, [&] {
        for (uint32_t n = 0; n < b.N; ++n) {
            const uint8_t* packed = b.packed.data() + static_cast<size_t>(n) * b.pgb;
            const __fp16* ar = b.ar.data();
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t cb_lo;
                float16x8_t cb_hi;
                decode_packed(vld1_u8(packed + i / 2), cb_lo, cb_hi);
                float16x8_t ar_lo = vld1q_f16(ar + i);
                float16x8_t ar_hi = vld1q_f16(ar + i + 8);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)), vcvt_f32_f16(vget_low_f16(ar_lo)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)), vcvt_f32_f16(vget_high_f16(ar_lo)));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)), vcvt_f32_f16(vget_low_f16(ar_hi)));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)), vcvt_f32_f16(vget_high_f16(ar_hi)));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[0]);
    });

    bench_ms("orthogonal_dot_full_ar32", iterations, [&] {
        for (uint32_t n = 0; n < b.N; ++n) {
            const uint8_t* packed = b.packed.data() + static_cast<size_t>(n) * b.pgb;
            const float* ar = b.ar32.data();
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t cb_lo;
                float16x8_t cb_hi;
                decode_packed(vld1_u8(packed + i / 2), cb_lo, cb_hi);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)), vld1q_f32(ar + i));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)), vld1q_f32(ar + i + 4));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)), vld1q_f32(ar + i + 8));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)), vld1q_f32(ar + i + 12));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[1]);
    });

    bench_ms("orthogonal_qd8_qc8_dot4", iterations, [&] {
        const float act_scale = 0.03125f;
        const int8_t* a = b.act_i8.data();
        for (uint32_t n = 0; n < b.N; n += 4) {
            const int8_t* w0 = b.qd8_qc8_weights.data() + static_cast<size_t>(n + 0) * b.K;
            const int8_t* w1 = b.qd8_qc8_weights.data() + static_cast<size_t>(n + 1) * b.K;
            const int8_t* w2 = b.qd8_qc8_weights.data() + static_cast<size_t>(n + 2) * b.K;
            const int8_t* w3 = b.qd8_qc8_weights.data() + static_cast<size_t>(n + 3) * b.K;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            for (uint32_t k = 0; k < b.K; k += 16) {
                const int8x16_t av = vld1q_s8(a + k);
                acc0 = vdotq_s32(acc0, vld1q_s8(w0 + k), av);
                acc1 = vdotq_s32(acc1, vld1q_s8(w1 + k), av);
                acc2 = vdotq_s32(acc2, vld1q_s8(w2 + k), av);
                acc3 = vdotq_s32(acc3, vld1q_s8(w3 + k), av);
            }
            b.out_f32[n + 0] = static_cast<float>(vaddvq_s32(acc0)) * act_scale * b.qd8_qc8_scales[n + 0];
            b.out_f32[n + 1] = static_cast<float>(vaddvq_s32(acc1)) * act_scale * b.qd8_qc8_scales[n + 1];
            b.out_f32[n + 2] = static_cast<float>(vaddvq_s32(acc2)) * act_scale * b.qd8_qc8_scales[n + 2];
            b.out_f32[n + 3] = static_cast<float>(vaddvq_s32(acc3)) * act_scale * b.qd8_qc8_scales[n + 3];
        }
        sink_f32 += b.out_f32[0];
    });

    bench_ms("orthogonal_decode_only", iterations, [&] {
        uint32x4_t acc = vdupq_n_u32(0);
        for (uint32_t n = 0; n < b.N; ++n) {
            const uint8_t* packed = b.packed.data() + static_cast<size_t>(n) * b.pgb;
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t cb_lo;
                float16x8_t cb_hi;
                decode_packed(vld1_u8(packed + i / 2), cb_lo, cb_hi);
                acc = vaddq_u32(acc, vreinterpretq_u32_f16(cb_lo));
                acc = vaddq_u32(acc, vreinterpretq_u32_f16(cb_hi));
            }
        }
        sink_i64 += static_cast<int64_t>(vaddvq_u32(acc));
    });

    bench_ms("orthogonal_ar_fma_only", iterations, [&] {
        const float16x8_t cb_lo = vdupq_n_f16(0.125f);
        const float16x8_t cb_hi = vdupq_n_f16(-0.25f);
        for (uint32_t n = 0; n < b.N; ++n) {
            const __fp16* ar = b.ar.data();
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t ar_lo = vld1q_f16(ar + i);
                float16x8_t ar_hi = vld1q_f16(ar + i + 8);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)), vcvt_f32_f16(vget_low_f16(ar_lo)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)), vcvt_f32_f16(vget_high_f16(ar_lo)));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)), vcvt_f32_f16(vget_low_f16(ar_hi)));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)), vcvt_f32_f16(vget_high_f16(ar_hi)));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[2]);
    });

    bench_ms("orthogonal_fcvt_fma_const", iterations, [&] {
        const float16x8_t cb_lo = vdupq_n_f16(0.125f);
        const float16x8_t cb_hi = vdupq_n_f16(-0.25f);
        const float16x8_t ar_lo = vdupq_n_f16(0.03125f);
        const float16x8_t ar_hi = vdupq_n_f16(-0.0625f);
        for (uint32_t n = 0; n < b.N; ++n) {
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t cb_lo_i = cb_lo;
                float16x8_t cb_hi_i = cb_hi;
                float16x8_t ar_lo_i = ar_lo;
                float16x8_t ar_hi_i = ar_hi;
                asm volatile("" : "+w"(cb_lo_i), "+w"(cb_hi_i), "+w"(ar_lo_i), "+w"(ar_hi_i));
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo_i)), vcvt_f32_f16(vget_low_f16(ar_lo_i)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo_i)), vcvt_f32_f16(vget_high_f16(ar_lo_i)));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi_i)), vcvt_f32_f16(vget_low_f16(ar_hi_i)));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi_i)), vcvt_f32_f16(vget_high_f16(ar_hi_i)));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[3]);
    });

    bench_ms("orthogonal_ar_fcvt_reduce_only", iterations, [&] {
        for (uint32_t n = 0; n < b.N; ++n) {
            const __fp16* ar = b.ar.data();
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                float16x8_t ar_lo = vld1q_f16(ar + i);
                float16x8_t ar_hi = vld1q_f16(ar + i + 8);
                acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(ar_lo)));
                acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(ar_lo)));
                acc2 = vaddq_f32(acc2, vcvt_f32_f16(vget_low_f16(ar_hi)));
                acc3 = vaddq_f32(acc3, vcvt_f32_f16(vget_high_f16(ar_hi)));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[4]);
    });

    bench_ms("orthogonal_ar32_fma_only", iterations, [&] {
        const float32x4_t cb_lo = vdupq_n_f32(0.125f);
        const float32x4_t cb_hi = vdupq_n_f32(-0.25f);
        for (uint32_t n = 0; n < b.N; ++n) {
            const float* ar = b.ar32.data();
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                acc0 = vfmaq_f32(acc0, cb_lo, vld1q_f32(ar + i));
                acc1 = vfmaq_f32(acc1, cb_lo, vld1q_f32(ar + i + 4));
                acc2 = vfmaq_f32(acc2, cb_hi, vld1q_f32(ar + i + 8));
                acc3 = vfmaq_f32(acc3, cb_hi, vld1q_f32(ar + i + 12));
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[5]);
    });

    bench_ms("orthogonal_fma_const", iterations, [&] {
        const float32x4_t cb_lo = vdupq_n_f32(0.125f);
        const float32x4_t cb_hi = vdupq_n_f32(-0.25f);
        const float32x4_t ar_lo = vdupq_n_f32(0.03125f);
        const float32x4_t ar_hi = vdupq_n_f32(-0.0625f);
        for (uint32_t n = 0; n < b.N; ++n) {
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);
            for (uint32_t i = 0; i < b.K; i += 16) {
                acc0 = vfmaq_f32(acc0, cb_lo, ar_lo);
                acc1 = vfmaq_f32(acc1, cb_lo, ar_lo);
                acc2 = vfmaq_f32(acc2, cb_hi, ar_hi);
                acc3 = vfmaq_f32(acc3, cb_hi, ar_hi);
            }
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            b.out[n] = static_cast<__fp16>(acc * static_cast<float>(b.norms[n]));
        }
        sink_f32 += static_cast<float>(b.out[6]);
    });
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 200;
    if (argc > 1) iterations = std::max(1, std::atoi(argv[1]));
    if (argc > 2) shape_filter = argv[2];
    if (argc > 3) bench_filter = argv[3];

    const Shape shapes[] = {
        {"qwen_like", 6144, 2048, 128},
        {"lfm_like", 8192, 2048, 128},
        {"gemma_like", 2304, 9216, 128},
        {"gemma_ffn_up", 1536, 12288, 128},
        {"gemma_ffn_down", 12288, 1536, 128},
        {"gemma_ffn_mid_up", 1536, 6144, 128},
        {"gemma_ffn_mid_down", 6144, 1536, 128},
        {"gemma_attn_q", 1536, 2048, 128},
        {"gemma_attn_in", 2048, 1536, 128},
    };
    for (const Shape& shape : shapes) {
        run_probe(shape, iterations);
    }
    run_orthogonal_probe(iterations);
    std::cout << "sink_i64=" << sink_i64 << " sink_f32=" << sink_f32 << "\n";
    return 0;
}
