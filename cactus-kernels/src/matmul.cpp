#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
constexpr size_t ACCELERATE_M_THRESHOLD = 4;
constexpr size_t ACCELERATE_K_THRESHOLD = 256;
#endif

struct CactusQuantMatmulTrace {
    const CactusQuantMatrix* W = nullptr;
    uint32_t M = 0;
    const char* path = "unknown";
    bool enabled = false;
    std::chrono::steady_clock::time_point start;

    CactusQuantMatmulTrace(const CactusQuantMatrix* matrix, uint32_t rows)
        : W(matrix), M(rows) {
        static const bool trace_enabled = [] {
            const char* file = std::getenv("CACTUS_QUANT_MATMUL_TRACE_FILE");
            return file && file[0];
        }();
        enabled = trace_enabled;
        if (enabled) start = std::chrono::steady_clock::now();
    }

    ~CactusQuantMatmulTrace() {
        if (!enabled) return;
        const char* file = std::getenv("CACTUS_QUANT_MATMUL_TRACE_FILE");
        if (!file || !file[0] || !W) return;
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        bool write_header = false;
        {
            std::ifstream in(file, std::ios::binary | std::ios::ate);
            write_header = !in.good() || in.tellg() == 0;
        }
        std::ofstream out(file, std::ios::app);
        if (!out) return;
        if (write_header) {
            out << "path,M,K,N,bits,group_size,num_groups,flags,ms\n";
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        out << path << ","
            << M << ","
            << W->K << ","
            << W->N << ","
            << W->bits << ","
            << W->group_size << ","
            << W->num_groups << ","
            << W->flags << ","
            << std::fixed << std::setprecision(3) << ms << "\n";
    }
};

struct CactusQuantArgmaxTrace {
    const CactusQuantMatrix* W = nullptr;
    uint32_t M = 0;
    bool enabled = false;
    std::chrono::steady_clock::time_point start;

    CactusQuantArgmaxTrace(const CactusQuantMatrix* matrix, uint32_t rows)
        : W(matrix), M(rows) {
        static const bool trace_enabled = [] {
            const char* file = std::getenv("CACTUS_QUANT_ARGMAX_TRACE_FILE");
            return file && file[0];
        }();
        enabled = trace_enabled;
        if (enabled) start = std::chrono::steady_clock::now();
    }

    ~CactusQuantArgmaxTrace() {
        if (!enabled) return;
        const char* file = std::getenv("CACTUS_QUANT_ARGMAX_TRACE_FILE");
        if (!file || !file[0] || !W) return;
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        bool write_header = false;
        {
            std::ifstream in(file, std::ios::binary | std::ios::ate);
            write_header = !in.good() || in.tellg() == 0;
        }
        std::ofstream out(file, std::ios::app);
        if (!out) return;
        if (write_header) {
            out << "M,K,N,bits,flags,ms\n";
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        out << M << ","
            << W->K << ","
            << W->N << ","
            << W->bits << ","
            << W->flags << ","
            << std::fixed << std::setprecision(3) << ms << "\n";
    }
};

// Do NOT Remove: Uncomment for testing on various paths
// -----
// TEMPORARY: Force fallback path for testing on DOTPROD devices
// #undef __ARM_FEATURE_DOTPROD

#if defined(__ARM_FEATURE_DOTPROD)
    #define CACTUS_DOTQ_LANE(acc, b, a, lane) vdotq_laneq_s32(acc, b, a, lane)
#else
    static inline int32x4_t cactus_dotq_with_pattern(int32x4_t acc, int8x16_t b, int8x8_t a_pattern) {
        int8x8_t b_lo = vget_low_s8(b);
        int8x8_t b_hi = vget_high_s8(b);

        int16x8_t prod_lo = vmull_s8(b_lo, a_pattern);
        int16x8_t prod_hi = vmull_s8(b_hi, a_pattern);

        int32x4_t sum_lo = vpaddlq_s16(prod_lo);
        int32x4_t sum_hi = vpaddlq_s16(prod_hi);

        int32x2_t final_lo = vpadd_s32(vget_low_s32(sum_lo), vget_high_s32(sum_lo));
        int32x2_t final_hi = vpadd_s32(vget_low_s32(sum_hi), vget_high_s32(sum_hi));

        return vaddq_s32(acc, vcombine_s32(final_lo, final_hi));
    }

    static inline int32x4_t cactus_dotq_lane0(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_lo = vget_low_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_lo), 0));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane1(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_lo = vget_low_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_lo), 1));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane2(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_hi = vget_high_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_hi), 0));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane3(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_hi = vget_high_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_hi), 1));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    #define CACTUS_DOTQ_LANE(acc, b, a, lane) cactus_dotq_lane##lane(acc, b, a)
#endif

static inline __fp16 hsum_f16x8(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t sum4 = vadd_f16(lo, hi);
    float16x4_t sum2 = vadd_f16(sum4, vext_f16(sum4, sum4, 2));
    float16x4_t sum1 = vadd_f16(sum2, vext_f16(sum2, sum2, 1));
    return vget_lane_f16(sum1, 0);
}

namespace {

static inline float16x8_t cactus_quant_signs_to_f16(const int8_t* signs, uint32_t offset) {
    if (signs == nullptr) return vdupq_n_f16(1);
    return vcvtq_f16_s16(vmovl_s8(vld1_s8(signs + offset)));
}

static inline float16x8_t cactus_quant_input_scale_recip8(const CactusQuantMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) {
        return vld1q_f16(W.input_scale_recip + offset);
    }
    if (W.input_scale != nullptr) {
        return vdivq_f16(vdupq_n_f16(1), vld1q_f16(W.input_scale + offset));
    }
    return vdupq_n_f16(1);
}

static inline __fp16 cactus_quant_input_scale_recip1(const CactusQuantMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) return W.input_scale_recip[offset];
    if (W.input_scale != nullptr) return static_cast<__fp16>(1.0f / static_cast<float>(W.input_scale[offset]));
    return static_cast<__fp16>(1);
}

static inline const __fp16* cactus_quant_scale_ptr(const CactusQuantMatrix& W, uint32_t row, uint32_t group) {
    return W.norms + static_cast<size_t>(row) * W.num_groups + group;
}

static inline const uint8_t* cactus_quant_packed_chunk_ptr(
    const CactusQuantMatrix& W,
    uint32_t row,
    uint32_t group,
    uint32_t k) {
    return W.packed_indices
        + (((static_cast<size_t>(row) * W.num_groups + group)
            * cactus_quant_packed_group_bytes(W.bits, W.group_size))
           + (static_cast<size_t>(k) * W.bits) / 8);
}

static inline void tq_interleave_4x_s8(const int8x16_t row0, const int8x16_t row1,
                                       const int8x16_t row2, const int8x16_t row3,
                                       int8_t* dst) {
    int32x4_t r0 = vreinterpretq_s32_s8(row0), r1 = vreinterpretq_s32_s8(row1);
    int32x4_t r2 = vreinterpretq_s32_s8(row2), r3 = vreinterpretq_s32_s8(row3);
    int32x4_t t01l = vzip1q_s32(r0, r1), t01h = vzip2q_s32(r0, r1);
    int32x4_t t23l = vzip1q_s32(r2, r3), t23h = vzip2q_s32(r2, r3);
    vst1q_s8(dst,      vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01l), vreinterpretq_s64_s32(t23l))));
    vst1q_s8(dst + 16, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01l), vreinterpretq_s64_s32(t23l))));
    vst1q_s8(dst + 32, vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01h), vreinterpretq_s64_s32(t23h))));
    vst1q_s8(dst + 48, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01h), vreinterpretq_s64_s32(t23h))));
}

static inline void tq_interleave_4x_s8_vecs(const int8x16_t row0, const int8x16_t row1,
                                            const int8x16_t row2, const int8x16_t row3,
                                            int8x16_t& out0, int8x16_t& out1,
                                            int8x16_t& out2, int8x16_t& out3) {
    int32x4_t r0 = vreinterpretq_s32_s8(row0), r1 = vreinterpretq_s32_s8(row1);
    int32x4_t r2 = vreinterpretq_s32_s8(row2), r3 = vreinterpretq_s32_s8(row3);
    int32x4_t t01l = vzip1q_s32(r0, r1), t01h = vzip2q_s32(r0, r1);
    int32x4_t t23l = vzip1q_s32(r2, r3), t23h = vzip2q_s32(r2, r3);
    out0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01l), vreinterpretq_s64_s32(t23l)));
    out1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01l), vreinterpretq_s64_s32(t23l)));
    out2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01h), vreinterpretq_s64_s32(t23h)));
    out3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01h), vreinterpretq_s64_s32(t23h)));
}

static inline float16x8_t cactus_tq4_lookup_codebook8(uint8x8_t nibbles, uint8x16x2_t cb_bytes) {
    uint8x8_t byte_offsets = vshl_n_u8(nibbles, 1);
    uint8x8_t byte_offsets_hi = vadd_u8(byte_offsets, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(byte_offsets, byte_offsets_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vreinterpretq_f16_u8(vqtbl2q_u8(cb_bytes, byte_idx));
}

static inline uint8x8_t cactus_tq2_unpack_8x2bit_le(uint8_t b0, uint8_t b1) {
    uint64_t idx_word =
        ((uint64_t)((b0     ) & 0x3)      ) |
        ((uint64_t)((b0 >> 2) & 0x3) <<  8) |
        ((uint64_t)((b0 >> 4) & 0x3) << 16) |
        ((uint64_t)((b0 >> 6) & 0x3) << 24) |
        ((uint64_t)((b1     ) & 0x3) << 32) |
        ((uint64_t)((b1 >> 2) & 0x3) << 40) |
        ((uint64_t)((b1 >> 4) & 0x3) << 48) |
        ((uint64_t)((b1 >> 6) & 0x3) << 56);
    return vcreate_u8(idx_word);
}

static inline float16x8_t cactus_tq2_lookup_codebook8(uint8x8_t indices, uint8x8_t cb_bytes) {
    uint8x8_t off_lo = vshl_n_u8(indices, 1);
    uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    uint8x16_t lut = vcombine_u8(cb_bytes, cb_bytes);
    return vreinterpretq_f16_u8(vqtbl1q_u8(lut, byte_idx));
}

static void cactus_quant_fwht128_f16(__fp16* x) {
    float16x8_t v[16];
    for (int i = 0; i < 16; ++i) v[i] = vld1q_f16(x + i * 8);
    for (int i = 0; i < 16; ++i) {
        float16x8_t r = vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(v[i])));
        float16x8_t s = vaddq_f16(v[i], r);
        float16x8_t d = vsubq_f16(v[i], r);
        v[i] = vreinterpretq_f16_u16(vtrn1q_u16(vreinterpretq_u16_f16(s), vreinterpretq_u16_f16(d)));
    }
    for (int i = 0; i < 16; ++i) {
        float32x4_t f32 = vreinterpretq_f32_f16(v[i]);
        float16x8_t a = vreinterpretq_f16_f32(vtrn1q_f32(f32, f32));
        float16x8_t b = vreinterpretq_f16_f32(vtrn2q_f32(f32, f32));
        float16x8_t s = vaddq_f16(a, b);
        float16x8_t d = vsubq_f16(a, b);
        v[i] = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(s), vreinterpretq_f32_f16(d)));
    }
    for (int i = 0; i < 16; ++i) {
        float16x4_t lo = vget_low_f16(v[i]);
        float16x4_t hi = vget_high_f16(v[i]);
        v[i] = vcombine_f16(vadd_f16(lo, hi), vsub_f16(lo, hi));
    }

    auto pass = [&](int s) {
        for (int base = 0; base < 16; base += (s << 1)) {
            for (int j = 0; j < s; ++j) {
                float16x8_t a = v[base + j];
                float16x8_t b = v[base + j + s];
                v[base + j] = vaddq_f16(a, b);
                v[base + j + s] = vsubq_f16(a, b);
            }
        }
    };
    pass(1);
    pass(2);
    pass(4);
    pass(8);

    float16x8_t inv = vdupq_n_f16(static_cast<__fp16>(1.0f / std::sqrt(128.0f)));
    for (int i = 0; i < 16; ++i) {
        vst1q_f16(x + i * 8, vmulq_f16(v[i], inv));
    }
}

static void cactus_quant_fwht_f16(__fp16* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; j += 8) {
                if (j + 8 <= i + h) {
                    float16x8_t a = vld1q_f16(x + j);
                    float16x8_t b = vld1q_f16(x + j + h);
                    vst1q_f16(x + j, vaddq_f16(a, b));
                    vst1q_f16(x + j + h, vsubq_f16(a, b));
                } else {
                    for (uint32_t k = j; k < i + h; ++k) {
                        __fp16 a = x[k];
                        __fp16 b = x[k + h];
                        x[k] = static_cast<__fp16>(a + b);
                        x[k + h] = static_cast<__fp16>(a - b);
                    }
                }
            }
        }
    }

    const float16x8_t inv_v = vdupq_n_f16(static_cast<__fp16>(1.0f / std::sqrt(static_cast<float>(n))));
    uint32_t k = 0;
    for (; k + 8 <= n; k += 8) {
        vst1q_f16(x + k, vmulq_f16(vld1q_f16(x + k), inv_v));
    }
    const __fp16 inv = static_cast<__fp16>(1.0f / std::sqrt(static_cast<float>(n)));
    for (; k < n; ++k) {
        x[k] = static_cast<__fp16>(x[k] * inv);
    }
}

static void cactus_quant_transform_hadamard_group(
    const CactusQuantMatrix& W,
    const __fp16* x_group,
    uint32_t group,
    __fp16* code_basis) {
    const uint32_t gs = W.group_size;
    __fp16 tmp[256];
    __fp16* work = (W.permutation == nullptr) ? code_basis : tmp;

    uint32_t k = 0;
    for (; k + 8 <= gs; k += 8) {
        const uint32_t offset = group * gs + k;
        float16x8_t x_v = vld1q_f16(x_group + k);
        x_v = vmulq_f16(x_v, cactus_quant_input_scale_recip8(W, offset));
        float16x8_t s_v = cactus_quant_signs_to_f16(W.left_signs, k);
        vst1q_f16(work + k, vmulq_f16(x_v, s_v));
    }
    for (; k < gs; ++k) {
        const uint32_t offset = group * gs + k;
        const float sign = W.left_signs ? static_cast<float>(W.left_signs[k]) : 1.0f;
        const float scale = static_cast<float>(cactus_quant_input_scale_recip1(W, offset));
        work[k] = static_cast<__fp16>(static_cast<float>(x_group[k]) * scale * sign);
    }

    if (gs == 128) {
        cactus_quant_fwht128_f16(work);
    } else {
        cactus_quant_fwht_f16(work, gs);
    }

    k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t w_v = vld1q_f16(work + k);
        float16x8_t s_v = cactus_quant_signs_to_f16(W.right_signs, k);
        vst1q_f16(work + k, vmulq_f16(w_v, s_v));
    }
    for (; k < gs; ++k) {
        const float sign = W.right_signs ? static_cast<float>(W.right_signs[k]) : 1.0f;
        work[k] = static_cast<__fp16>(static_cast<float>(work[k]) * sign);
    }

    if (work != code_basis) {
        for (uint32_t j = 0; j < gs; ++j) {
            code_basis[j] = work[W.permutation[j]];
        }
    }
}

static void cactus_quant_transform_hadamard_activations(
    const CactusQuantMatrix& W,
    const __fp16* A,
    uint32_t M,
    __fp16* code_basis) {
    const size_t work_items = static_cast<size_t>(M) * W.num_groups;
    CactusThreading::parallel_for(
        work_items,
        CactusThreading::ParallelConfig{16, 1},
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t m = idx / W.num_groups;
                const size_t g = idx - m * W.num_groups;
                cactus_quant_transform_hadamard_group(
                    W,
                    A + m * W.K + g * W.group_size,
                    static_cast<uint32_t>(g),
                    code_basis + m * W.K + g * W.group_size);
            }
        });
}

template<typename WorkFunc>
static void cactus_quant_parallel_ranges(size_t total_work, size_t work_per_thread, WorkFunc work_func) {
    if (total_work == 0) return;
    if (work_per_thread == 0) work_per_thread = 1;

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = std::min(pool.num_workers(), (total_work + work_per_thread - 1) / work_per_thread);
    num_threads = std::min(num_threads, total_work);
    if (num_threads <= 1) {
        work_func(0, total_work);
        return;
    }

    pool.enqueue_n_threads(total_work, num_threads, work_func);
    pool.wait_all();
}

static void cactus_quant_matmul_f32_segment_accum(
    const __fp16* __restrict__ A,
    size_t a_stride,
    const __fp16* __restrict__ B_tile,
    float* __restrict__ C_f32,
    size_t M,
    size_t Kseg,
    size_t actual_n) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N_MAX = 16;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);

        float16x8_t acc[TILE_M][TILE_N_MAX];
        for (size_t mi = 0; mi < actual_m; ++mi)
            for (size_t ni = 0; ni < actual_n; ++ni)
                acc[mi][ni] = vdupq_n_f16(0);

        for (size_t k = 0; k < Kseg; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < actual_m; ++mi) {
                const __fp16* ap = A + (m_start + mi) * a_stride + k;
                a_lo[mi] = vld1q_f16(ap);
                a_hi[mi] = vld1q_f16(ap + 8);
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                const __fp16* bp = B_tile + ni * Kseg + k;
                float16x8_t b_lo = vld1q_f16(bp);
                float16x8_t b_hi = vld1q_f16(bp + 8);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_lo[mi], b_lo);
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_hi[mi], b_hi);
                }
            }
        }

        for (size_t mi = 0; mi < actual_m; ++mi)
            for (size_t ni = 0; ni < actual_n; ++ni)
                C_f32[(m_start + mi) * TILE_N_MAX + ni] += static_cast<float>(hsum_f16x8(acc[mi][ni]));
    }
}

struct CactusTQ4ScaledDecoder {
    uint8x16x2_t cb_bytes;

    explicit CactusTQ4ScaledDecoder(const CactusQuantMatrix& W) {
        cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook));
        cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook) + 16);
    }

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 16) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
            uint8x8_t bytes = vld1_u8(packed);
            uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
            uint8x8_t hi = vshr_n_u8(bytes, 4);
            vst1q_f16(dst + k,
                      vmulq_f16(cactus_tq4_lookup_codebook8(vzip1_u8(lo, hi), cb_bytes), rn_v));
            vst1q_f16(dst + k + 8,
                      vmulq_f16(cactus_tq4_lookup_codebook8(vzip2_u8(lo, hi), cb_bytes), rn_v));
        }
    }
};

struct CactusTQ2ScaledDecoder {
    uint8x8_t cb_bytes;

    explicit CactusTQ2ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
            uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(packed[0], packed[1]);
            vst1q_f16(dst + k,
                      vmulq_f16(cactus_tq2_lookup_codebook8(indices, cb_bytes), rn_v));
        }
    }
};

template<uint32_t Bits, typename DecodeGroup>
static void cactus_quant_group_gemm(
    const CactusQuantMatrix& W,
    const __fp16* A,
    uint32_t M,
    __fp16* C,
    DecodeGroup decode_group) {
    static_assert(Bits >= 1 && Bits <= 4);
    if (W.bits != Bits) return;

    constexpr size_t TILE_N = 16;
    const size_t n_blocks = (W.N + TILE_N - 1) / TILE_N;


    CactusThreading::parallel_gemm_tiles(M, n_blocks,
        [&, decode_group](size_t block_start, size_t block_end) {
            thread_local std::vector<__fp16> b_tile;
            thread_local std::vector<float> c_accum;
            if (b_tile.size() < TILE_N * W.group_size)
                b_tile.resize(TILE_N * W.group_size);
            if (c_accum.size() < M * TILE_N)
                c_accum.resize(M * TILE_N);

            for (size_t block = block_start; block < block_end; ++block) {
                const size_t n_start = block * TILE_N;
                const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W.N) - n_start);

                std::fill(c_accum.begin(), c_accum.begin() + M * TILE_N, 0.0f);

                for (uint32_t g = 0; g < W.num_groups; ++g) {
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        decode_group(W, static_cast<uint32_t>(n_start + ni), g,
                                     b_tile.data() + ni * W.group_size);
                    }
                    cactus_quant_matmul_f32_segment_accum(
                        A + static_cast<size_t>(g) * W.group_size,
                        W.K,
                        b_tile.data(),
                        c_accum.data(),
                        M,
                        W.group_size,
                        actual_n);
                }

                for (size_t m = 0; m < M; ++m)
                    for (size_t ni = 0; ni < actual_n; ++ni)
                        C[m * W.N + n_start + ni] = static_cast<__fp16>(c_accum[m * TILE_N + ni]);
            }
        });
}

static bool cactus_quant_valid_common(const CactusQuantMatrix* W, const void* A, void* C) {
    if (W == nullptr || A == nullptr || C == nullptr) return false;
    if (W->K == 0 || W->N == 0 || W->group_size == 0 || W->num_groups == 0) return false;
    if (W->group_size > 256) return false;
    if ((W->group_size & (W->group_size - 1)) != 0) return false;
    if (W->K != W->group_size * W->num_groups) return false;
    if (W->codebook == nullptr || W->norms == nullptr || W->packed_indices == nullptr) return false;
    return true;
}

}  // namespace

static void cactus_quant_dispatch_group_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    switch (W->bits) {
        case 1: cactus_quant_1bit_gemm(W, A, M, C); return;
        case 2: cactus_quant_2bit_gemm(W, A, M, C); return;
        case 3: cactus_quant_3bit_gemm(W, A, M, C); return;
        case 4: cactus_quant_4bit_gemm(W, A, M, C); return;
        default: return;
    }
}

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t /*M*/,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        const size_t m_end = std::min(row_block + TILE_M, end_row);

        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            const size_t n_end = std::min(col_block + TILE_N, N);

            float16x8_t acc[TILE_M][TILE_N];
            for (size_t m = 0; m < TILE_M; ++m)
                for (size_t n = 0; n < TILE_N; ++n)
                    acc[m][n] = vdupq_n_f16(0);

            for (size_t k = 0; k < K16; k += 16) {
                float16x8_t a0_lo = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a0_hi = (row_block < m_end) ? vld1q_f16(a + row_block * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a1_lo = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a1_hi = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a2_lo = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_hi = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a3_lo = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_hi = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k + 8) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_lo = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    float16x8_t b_hi = vld1q_f16(b_transposed + (col_block + ni) * K + k + 8);

                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_lo, b_lo);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_hi, b_hi);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_lo, b_lo);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_hi, b_hi);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_lo, b_lo);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_hi, b_hi);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_lo, b_lo);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_hi, b_hi);
                }
            }

            for (size_t k = K16; k < K8; k += 8) {
                float16x8_t a0_v = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a1_v = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_v = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_v = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_v = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_v, b_v);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_v, b_v);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_v, b_v);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_v, b_v);
                }
            }

            for (size_t k = K8; k < K; ++k) {
                for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                    __fp16 av = a[(row_block + mi) * K + k];
                    for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                        __fp16 bv = b_transposed[(col_block + ni) * K + k];
                        acc[mi][ni] = vsetq_lane_f16(vgetq_lane_f16(acc[mi][ni], 0) + av * bv, acc[mi][ni], 0);
                    }
                }
            }

            for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    c[(row_block + mi) * N + col_block + ni] = hsum_f16x8(acc[mi][ni]);
                }
            }
        }
    }
}

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {

#ifdef __APPLE__
    if (K >= ACCELERATE_K_THRESHOLD && M >= ACCELERATE_M_THRESHOLD) {
        const size_t a_len = M * K;
        const size_t b_len = N * K;
        const size_t c_len = M * N;

        std::vector<float> A_f32(a_len);
        std::vector<float> BT_f32(b_len);
        std::vector<float> C_f32(c_len);

        for (size_t i = 0; i < a_len; i++) A_f32[i] = (float)a[i];
        for (size_t i = 0; i < b_len; i++) BT_f32[i] = (float)b_transposed[i];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)N, (int)K,
                    1.0f, A_f32.data(), (int)K,
                    BT_f32.data(), (int)K,
                    0.0f, C_f32.data(), (int)N);

        for (size_t i = 0; i < c_len; i++) {
            float v = C_f32[i];
            if (v > 65504.f) v = 65504.f;
            else if (v < -65504.f) v = -65504.f;
            c[i] = (__fp16)v;
        }
        return;
    }
#endif

    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );

            }
        });
}

uint32_t cactus_quant_packed_group_bytes(uint32_t bits, uint32_t group_size) {
    if (bits == 0 || bits > 4) return 0;
    return (group_size * bits + 7) / 8;
}

static inline float tq_quantize_group_i8(const __fp16* src, int8_t* dst, uint32_t gs);
static inline float tq_quantize_codebook_i8(const __fp16* codebook, int8_t* cb_i8, uint32_t cb_size);
static inline int8x16_t tq_expand_i8_16(const uint8_t* packed, uint32_t bits, int8x16_t cb_lut);

void cactus_quant_4bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x16x2_t cb_bytes;
    cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook));
    cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook) + 16);

    const uint32_t pgb = cactus_quant_packed_group_bytes(4, gs);
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < W->K) code_basis_buf.resize(W->K);
    cactus_quant_transform_hadamard_activations(*W, x, 1, code_basis_buf.data());
    const __fp16* code_basis = code_basis_buf.data();

    if ((gs % 32) == 0 && gs <= 256) {
        constexpr size_t INT8_TILE_N = 16;
        const size_t int8_n_blocks = (W->N + INT8_TILE_N - 1) / INT8_TILE_N;

        thread_local std::vector<int8_t> act_i8_buf;
        thread_local std::vector<float> act_scales_buf;
        if (act_i8_buf.size() < W->K) act_i8_buf.resize(W->K);
        if (act_scales_buf.size() < W->num_groups) act_scales_buf.resize(W->num_groups);

        for (uint32_t g = 0; g < W->num_groups; ++g) {
            const __fp16* src = code_basis + static_cast<size_t>(g) * gs;
            int8_t* dst = act_i8_buf.data() + static_cast<size_t>(g) * gs;
            act_scales_buf[g] = tq_quantize_group_i8(src, dst, gs);
        }
        const int8_t* act_i8 = act_i8_buf.data();
        const float* act_scales = act_scales_buf.data();

        int8_t cb_i8[16] = {};
        const float cb_scale = tq_quantize_codebook_i8(W->codebook, cb_i8, 16);
        const int8x16_t cb_lut = vld1q_s8(cb_i8);

        auto expand_group4 = [&](size_t cache_block, uint32_t g, int8_t* dst, float* norm_scale) {
            int8x16_t exp4[4][16];
            const uint32_t n_vecs = gs / 16;
            const size_t n_start4 = cache_block * 4;
            for (size_t ni = 0; ni < 4; ++ni) {
                const size_t n_abs = n_start4 + ni;
                if (n_abs < W->N) {
                    const uint8_t* p = W->packed_indices
                        + (n_abs * W->num_groups + g) * pgb;
                    for (uint32_t v = 0; v < n_vecs; ++v) {
                        exp4[ni][v] = tq_expand_i8_16(p + (v * 16 * 4) / 8, 4, cb_lut);
                    }
                    norm_scale[ni] = static_cast<float>(W->norms[n_abs * W->num_groups + g]) * cb_scale;
                } else {
                    for (uint32_t v = 0; v < n_vecs; ++v) exp4[ni][v] = vdupq_n_s8(0);
                    norm_scale[ni] = 0.0f;
                }
            }

            for (uint32_t v = 0; v < n_vecs; ++v)
                tq_interleave_4x_s8(exp4[0][v], exp4[1][v], exp4[2][v], exp4[3][v], dst + v * 64);
        };

        cactus_quant_parallel_ranges(int8_n_blocks, 16, [&](size_t block_start, size_t block_end) {
            for (size_t block = block_start; block < block_end; ++block) {
                const size_t n_start = block * INT8_TILE_N;
                const size_t actual_n = std::min(INT8_TILE_N, static_cast<size_t>(W->N) - n_start);
                float acc[INT8_TILE_N] = {};

                for (uint32_t g = 0; g < W->num_groups; ++g) {
                    const int8_t* a_group = act_i8 + static_cast<size_t>(g) * gs;
                    const float act_scale = act_scales[g];

                    if (actual_n == INT8_TILE_N) {
                        int32x4_t dot0 = vdupq_n_s32(0);
                        int32x4_t dot1 = vdupq_n_s32(0);
                        int32x4_t dot2 = vdupq_n_s32(0);
                        int32x4_t dot3 = vdupq_n_s32(0);

                        const size_t cache_block0 = n_start / 4;
                        alignas(16) int8_t b_stack0[256 * 4];
                        alignas(16) int8_t b_stack1[256 * 4];
                        alignas(16) int8_t b_stack2[256 * 4];
                        alignas(16) int8_t b_stack3[256 * 4];
                        float norm_stack0[4], norm_stack1[4], norm_stack2[4], norm_stack3[4];

                        const int8_t* b_base0;
                        const int8_t* b_base1;
                        const int8_t* b_base2;
                        const int8_t* b_base3;
                        const float* norm_base0;
                        const float* norm_base1;
                        const float* norm_base2;
                        const float* norm_base3;
                        expand_group4(cache_block0 + 0, g, b_stack0, norm_stack0);
                        expand_group4(cache_block0 + 1, g, b_stack1, norm_stack1);
                        expand_group4(cache_block0 + 2, g, b_stack2, norm_stack2);
                        expand_group4(cache_block0 + 3, g, b_stack3, norm_stack3);
                        b_base0 = b_stack0;
                        b_base1 = b_stack1;
                        b_base2 = b_stack2;
                        b_base3 = b_stack3;
                        norm_base0 = norm_stack0;
                        norm_base1 = norm_stack1;
                        norm_base2 = norm_stack2;
                        norm_base3 = norm_stack3;

                        for (uint32_t k = 0; k < gs; k += 32) {
                            int8x16_t a_lo = vld1q_s8(a_group + k);
                            int8x16_t a_hi = vld1q_s8(a_group + k + 16);

                            #define TQ_SDOT_PANEL(DOT, BASE) do { \
                                const int8_t* bk = (BASE) + k * 4; \
                                int8x16_t b00 = vld1q_s8(bk); \
                                int8x16_t b01 = vld1q_s8(bk + 16); \
                                int8x16_t b02 = vld1q_s8(bk + 32); \
                                int8x16_t b03 = vld1q_s8(bk + 48); \
                                int8x16_t b10 = vld1q_s8(bk + 64); \
                                int8x16_t b11 = vld1q_s8(bk + 80); \
                                int8x16_t b12 = vld1q_s8(bk + 96); \
                                int8x16_t b13 = vld1q_s8(bk + 112); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b00, a_lo, 0); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b01, a_lo, 1); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b02, a_lo, 2); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b03, a_lo, 3); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b10, a_hi, 0); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b11, a_hi, 1); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b12, a_hi, 2); \
                                DOT = CACTUS_DOTQ_LANE(DOT, b13, a_hi, 3); \
                            } while (0)

                            TQ_SDOT_PANEL(dot0, b_base0);
                            TQ_SDOT_PANEL(dot1, b_base1);
                            TQ_SDOT_PANEL(dot2, b_base2);
                            TQ_SDOT_PANEL(dot3, b_base3);
                            #undef TQ_SDOT_PANEL
                        }

                        const float32x4_t scale0 = vmulq_n_f32(vld1q_f32(norm_base0), act_scale);
                        const float32x4_t scale1 = vmulq_n_f32(vld1q_f32(norm_base1), act_scale);
                        const float32x4_t scale2 = vmulq_n_f32(vld1q_f32(norm_base2), act_scale);
                        const float32x4_t scale3 = vmulq_n_f32(vld1q_f32(norm_base3), act_scale);

                        float tmp[16];
                        vst1q_f32(tmp,      vmulq_f32(vcvtq_f32_s32(dot0), scale0));
                        vst1q_f32(tmp + 4,  vmulq_f32(vcvtq_f32_s32(dot1), scale1));
                        vst1q_f32(tmp + 8,  vmulq_f32(vcvtq_f32_s32(dot2), scale2));
                        vst1q_f32(tmp + 12, vmulq_f32(vcvtq_f32_s32(dot3), scale3));
                        for (size_t ni = 0; ni < INT8_TILE_N; ++ni) acc[ni] += tmp[ni];
                        continue;
                    }

                    for (size_t ni4 = 0; ni4 < actual_n; ni4 += 4) {
                        const size_t n_abs = n_start + ni4;
                        const size_t cache_block = n_abs / 4;
                        alignas(16) int8_t b_stack[256 * 4];
                        float norm_stack[4];
                        const int8_t* b_base;
                        const float* norm_base;
                        expand_group4(cache_block, g, b_stack, norm_stack);
                        b_base = b_stack;
                        norm_base = norm_stack;
                        const float32x4_t scale_v = vmulq_n_f32(vld1q_f32(norm_base), act_scale);

                        int32x4_t dot = vdupq_n_s32(0);
                        for (uint32_t k = 0; k < gs; k += 32) {
                            const int8_t* bk = b_base + k * 4;
                            int8x16_t b00 = vld1q_s8(bk);
                            int8x16_t b01 = vld1q_s8(bk + 16);
                            int8x16_t b02 = vld1q_s8(bk + 32);
                            int8x16_t b03 = vld1q_s8(bk + 48);
                            int8x16_t b10 = vld1q_s8(bk + 64);
                            int8x16_t b11 = vld1q_s8(bk + 80);
                            int8x16_t b12 = vld1q_s8(bk + 96);
                            int8x16_t b13 = vld1q_s8(bk + 112);

                            int8x16_t a_lo = vld1q_s8(a_group + k);
                            dot = CACTUS_DOTQ_LANE(dot, b00, a_lo, 0);
                            dot = CACTUS_DOTQ_LANE(dot, b01, a_lo, 1);
                            dot = CACTUS_DOTQ_LANE(dot, b02, a_lo, 2);
                            dot = CACTUS_DOTQ_LANE(dot, b03, a_lo, 3);

                            int8x16_t a_hi = vld1q_s8(a_group + k + 16);
                            dot = CACTUS_DOTQ_LANE(dot, b10, a_hi, 0);
                            dot = CACTUS_DOTQ_LANE(dot, b11, a_hi, 1);
                            dot = CACTUS_DOTQ_LANE(dot, b12, a_hi, 2);
                            dot = CACTUS_DOTQ_LANE(dot, b13, a_hi, 3);
                        }

                        float32x4_t contrib = vmulq_f32(vcvtq_f32_s32(dot), scale_v);
                        float tmp[4];
                        vst1q_f32(tmp, contrib);
                        const size_t lane_count = std::min<size_t>(4, actual_n - ni4);
                        for (size_t lane = 0; lane < lane_count; ++lane) {
                            acc[ni4 + lane] += tmp[lane];
                        }
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    y[n_start + ni] = static_cast<__fp16>(acc[ni]);
                }
            }
        });
        return;
    }

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                const __fp16* z = code_basis + static_cast<size_t>(g) * gs;

                float16x8_t acc0[TILE_N];
                float16x8_t acc1[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    acc0[ni] = vdupq_n_f16(0);
                    acc1[ni] = vdupq_n_f16(0);
                }

                for (uint32_t k = 0; k < gs; k += 16) {
                    float16x8_t z0 = vld1q_f16(z + k);
                    float16x8_t z1 = vld1q_f16(z + k + 8);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 2;
                        uint8x8_t bytes = vld1_u8(p);
                        uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                        uint8x8_t hi = vshr_n_u8(bytes, 4);

                        float16x8_t cv0 = cactus_tq4_lookup_codebook8(vzip1_u8(lo, hi), cb_bytes);
                        float16x8_t cv1 = cactus_tq4_lookup_codebook8(vzip2_u8(lo, hi), cb_bytes);
                        acc0[ni] = vfmaq_f16(acc0[ni], z0, cv0);
                        acc1[ni] = vfmaq_f16(acc1[ni], z1, cv1);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn *
                        (static_cast<float>(hsum_f16x8(acc0[ni])) +
                         static_cast<float>(hsum_f16x8(acc1[ni])));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_2bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 16;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x8_t cb_bytes = vld1_u8(reinterpret_cast<const uint8_t*>(W->codebook));

    const uint32_t pgb = cactus_quant_packed_group_bytes(2, gs);
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < W->K) code_basis_buf.resize(W->K);
    cactus_quant_transform_hadamard_activations(*W, x, 1, code_basis_buf.data());
    const __fp16* code_basis = code_basis_buf.data();

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                const __fp16* z = code_basis + static_cast<size_t>(g) * gs;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    accv[ni] = vdupq_n_f16(0);
                }

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 4;
                        uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(p[0], p[1]);
                        float16x8_t cv = cactus_tq2_lookup_codebook8(indices, cb_bytes);
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_4bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<4>(*W, code_basis_buf.data(), M, C, CactusTQ4ScaledDecoder(*W));
}

void cactus_quant_2bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<2>(*W, code_basis_buf.data(), M, C, CactusTQ2ScaledDecoder(*W));
}



struct CactusTQ1ScaledDecoder {
    uint8x8_t cb_bytes;

    explicit CactusTQ1ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);

            uint8_t byte = packed[0];
            uint64_t idx_word =
                ((uint64_t)((byte >> 0) & 1)      ) |
                ((uint64_t)((byte >> 1) & 1) <<  8) |
                ((uint64_t)((byte >> 2) & 1) << 16) |
                ((uint64_t)((byte >> 3) & 1) << 24) |
                ((uint64_t)((byte >> 4) & 1) << 32) |
                ((uint64_t)((byte >> 5) & 1) << 40) |
                ((uint64_t)((byte >> 6) & 1) << 48) |
                ((uint64_t)((byte >> 7) & 1) << 56);
            uint8x8_t indices = vcreate_u8(idx_word);

            uint8x8_t off_lo = vshl_n_u8(indices, 1);
            uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
            uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
            uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
            uint8x16_t lut = vcombine_u8(cb_bytes, cb_bytes);
            float16x8_t cv = vreinterpretq_f16_u8(vqtbl1q_u8(lut, byte_idx));
            vst1q_f16(dst + k, vmulq_f16(cv, rn_v));
        }
    }
};

void cactus_quant_1bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 1 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x8_t cb_bytes = vld1_u8(reinterpret_cast<const uint8_t*>(W->codebook));
    uint8x16_t cb_lut = vcombine_u8(cb_bytes, cb_bytes);

    const uint32_t pgb = cactus_quant_packed_group_bytes(1, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) accv[ni] = vdupq_n_f16(0);

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 8;
                        uint8_t byte = p[0];
                        uint64_t idx_word =
                            ((uint64_t)((byte >> 0) & 1)      ) |
                            ((uint64_t)((byte >> 1) & 1) <<  8) |
                            ((uint64_t)((byte >> 2) & 1) << 16) |
                            ((uint64_t)((byte >> 3) & 1) << 24) |
                            ((uint64_t)((byte >> 4) & 1) << 32) |
                            ((uint64_t)((byte >> 5) & 1) << 40) |
                            ((uint64_t)((byte >> 6) & 1) << 48) |
                            ((uint64_t)((byte >> 7) & 1) << 56);
                        uint8x8_t indices = vcreate_u8(idx_word);
                        uint8x8_t off_lo = vshl_n_u8(indices, 1);
                        uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
                        uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
                        uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
                        float16x8_t cv = vreinterpretq_f16_u8(vqtbl1q_u8(cb_lut, byte_idx));
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_1bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 1 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<1>(*W, code_basis_buf.data(), M, C, CactusTQ1ScaledDecoder(*W));
}



static inline uint8x8_t cactus_tq3_unpack_8x3bit(const uint8_t* packed) {

    uint32_t b0 = packed[0], b1 = packed[1], b2 = packed[2];
    uint32_t word = b0 | (b1 << 8) | (b2 << 16);
    uint64_t idx_word =
        ((uint64_t)((word >> 0)  & 0x7)      ) |
        ((uint64_t)((word >> 3)  & 0x7) <<  8) |
        ((uint64_t)((word >> 6)  & 0x7) << 16) |
        ((uint64_t)((word >> 9)  & 0x7) << 24) |
        ((uint64_t)((word >> 12) & 0x7) << 32) |
        ((uint64_t)((word >> 15) & 0x7) << 40) |
        ((uint64_t)((word >> 18) & 0x7) << 48) |
        ((uint64_t)((word >> 21) & 0x7) << 56);
    return vcreate_u8(idx_word);
}

static inline float16x8_t cactus_tq3_lookup_codebook8(uint8x8_t indices, uint8x16_t cb_bytes) {
    uint8x8_t off_lo = vshl_n_u8(indices, 1);
    uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vreinterpretq_f16_u8(vqtbl1q_u8(cb_bytes, byte_idx));
}

struct CactusTQ3ScaledDecoder {
    uint8x16_t cb_bytes;

    explicit CactusTQ3ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
            uint8x8_t indices = cactus_tq3_unpack_8x3bit(packed);
            float16x8_t cv = cactus_tq3_lookup_codebook8(indices, cb_bytes);
            vst1q_f16(dst + k, vmulq_f16(cv, rn_v));
        }
    }
};

void cactus_quant_3bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 3 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x16_t cb_bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook));

    const uint32_t pgb = cactus_quant_packed_group_bytes(3, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) accv[ni] = vdupq_n_f16(0);

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + (k * 3) / 8;
                        uint8x8_t indices = cactus_tq3_unpack_8x3bit(p);
                        float16x8_t cv = cactus_tq3_lookup_codebook8(indices, cb_bytes);
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_3bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 3 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<3>(*W, code_basis_buf.data(), M, C, CactusTQ3ScaledDecoder(*W));
}

static inline float tq_quantize_codebook_i8(const __fp16* codebook, int8_t* cb_i8, uint32_t cb_size) {
    float max_abs = 0.f;
    for (uint32_t i = 0; i < cb_size; i++) {
        float v = std::abs(static_cast<float>(codebook[i]));
        if (v > max_abs) max_abs = v;
    }
    float scale = max_abs / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.f / scale;
    for (uint32_t i = 0; i < cb_size; i++)
        cb_i8[i] = static_cast<int8_t>(std::round(static_cast<float>(codebook[i]) * inv));
    return scale;
}

static inline float tq_quantize_group_i8(const __fp16* src, int8_t* dst, uint32_t gs) {
    float32x4_t max_vec = vdupq_n_f32(0.f);
    uint32_t k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t v = vld1q_f16(src + k);
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_low_f16(v))));
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_high_f16(v))));
    }
    float max_abs = vmaxvq_f32(max_vec);
    for (; k < gs; k++) {
        float v = std::abs(static_cast<float>(src[k]));
        if (v > max_abs) max_abs = v;
    }
    float scale = max_abs / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float32x4_t inv_vec = vdupq_n_f32(1.f / scale);
    k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t v = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v)), inv_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v)), inv_vec));
        vst1_s8(dst + k, vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))));
    }
    for (; k < gs; k++)
        dst[k] = static_cast<int8_t>(std::round(static_cast<float>(src[k]) / scale));
    return scale;
}


static inline uint32_t tq_extract_idx(const uint8_t* packed, uint32_t k, uint32_t bits) {
    if (bits == 4) return (packed[k / 2] >> ((k & 1) * 4)) & 0xF;
    if (bits == 2) return (packed[k / 4] >> ((k & 3) * 2)) & 0x3;
    if (bits == 1) return (packed[k / 8] >> (k & 7)) & 0x1;
    const uint32_t bit_pos = k * 3;
    const uint32_t byte_idx = bit_pos / 8;
    const uint32_t bit_idx = bit_pos & 7;
    uint32_t val = static_cast<uint32_t>(packed[byte_idx]) >> bit_idx;
    if (bit_idx > 5) {
        val |= static_cast<uint32_t>(packed[byte_idx + 1]) << (8 - bit_idx);
    }
    return val & 0x7;
}

static inline int8x16_t tq_expand_i8_16(const uint8_t* packed, uint32_t bits, int8x16_t cb_lut) {
    if (bits == 4) {
        uint8x8_t bytes = vld1_u8(packed);
        uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
        uint8x8_t hi = vshr_n_u8(bytes, 4);
        uint8x16_t idx = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
        return vqtbl1q_s8(cb_lut, idx);
    } else if (bits == 2) {
        uint8_t b0 = packed[0], b1 = packed[1], b2 = packed[2], b3 = packed[3];
        uint64_t lo_w =
            ((uint64_t)(b0&3)) | ((uint64_t)((b0>>2)&3)<<8) |
            ((uint64_t)((b0>>4)&3)<<16) | ((uint64_t)((b0>>6)&3)<<24) |
            ((uint64_t)(b1&3)<<32) | ((uint64_t)((b1>>2)&3)<<40) |
            ((uint64_t)((b1>>4)&3)<<48) | ((uint64_t)((b1>>6)&3)<<56);
        uint64_t hi_w =
            ((uint64_t)(b2&3)) | ((uint64_t)((b2>>2)&3)<<8) |
            ((uint64_t)((b2>>4)&3)<<16) | ((uint64_t)((b2>>6)&3)<<24) |
            ((uint64_t)(b3&3)<<32) | ((uint64_t)((b3>>2)&3)<<40) |
            ((uint64_t)((b3>>4)&3)<<48) | ((uint64_t)((b3>>6)&3)<<56);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    } else if (bits == 1) {
        uint8_t b0 = packed[0], b1 = packed[1];
        uint64_t lo_w =
            ((uint64_t)((b0>>0)&1)) | ((uint64_t)((b0>>1)&1)<<8) |
            ((uint64_t)((b0>>2)&1)<<16) | ((uint64_t)((b0>>3)&1)<<24) |
            ((uint64_t)((b0>>4)&1)<<32) | ((uint64_t)((b0>>5)&1)<<40) |
            ((uint64_t)((b0>>6)&1)<<48) | ((uint64_t)((b0>>7)&1)<<56);
        uint64_t hi_w =
            ((uint64_t)((b1>>0)&1)) | ((uint64_t)((b1>>1)&1)<<8) |
            ((uint64_t)((b1>>2)&1)<<16) | ((uint64_t)((b1>>3)&1)<<24) |
            ((uint64_t)((b1>>4)&1)<<32) | ((uint64_t)((b1>>5)&1)<<40) |
            ((uint64_t)((b1>>6)&1)<<48) | ((uint64_t)((b1>>7)&1)<<56);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    } else { // bits == 3
        uint64_t raw = 0;
        std::memcpy(&raw, packed, 6);
        uint64_t lo_w = 0, hi_w = 0;
        for (int i = 0; i < 8; i++) lo_w |= ((raw >> (i*3)) & 7ULL) << (i*8);
        for (int i = 0; i < 8; i++) hi_w |= ((raw >> ((i+8)*3)) & 7ULL) << (i*8);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    }
}

static void tq_preexpand_weights(
    const CactusQuantMatrix* W,
    uint32_t bits, uint32_t num_groups, uint32_t gs, uint32_t pgb,
    const int8x16_t& cb_lut, float cb_scale,
    size_t N_blocks,
    int8_t* w_il, float* n_f32) {
    const uint32_t n_vecs = gs / 16;
    for (size_t nb = 0; nb < N_blocks; ++nb) {
        size_t n_start = nb * 4;
        size_t valid_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);
        for (uint32_t g = 0; g < num_groups; ++g) {
            int8x16_t exp4[4][16];
            for (size_t ni = 0; ni < valid_n; ++ni) {
                const uint8_t* p = W->packed_indices + (static_cast<size_t>(n_start+ni)*num_groups+g)*pgb;
                for (uint32_t v = 0; v < n_vecs; ++v)
                    exp4[ni][v] = tq_expand_i8_16(p + (v*16*bits)/8, bits, cb_lut);
            }
            for (size_t ni = valid_n; ni < 4; ++ni)
                for (uint32_t v = 0; v < n_vecs; ++v) exp4[ni][v] = vdupq_n_s8(0);

            int8_t* dst = w_il + (nb*num_groups+g)*gs*4;
            for (uint32_t v = 0; v < n_vecs; ++v)
                tq_interleave_4x_s8(exp4[0][v], exp4[1][v], exp4[2][v], exp4[3][v], dst + v*64);
            float* nd = n_f32 + (nb*num_groups+g)*4;
            for (size_t ni = 0; ni < 4; ++ni)
                nd[ni] = (n_start+ni < W->N) ? static_cast<float>(W->norms[(n_start+ni)*num_groups+g]) * cb_scale : 0.f;
        }
    }
}

static void cactus_quant_small_m_sdot_gemm_no_preexpand(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C,
    uint32_t bits,
    uint32_t num_groups,
    uint32_t gs,
    uint32_t pgb,
    uint32_t codebook_size,
    size_t N_blocks) {
    const size_t act_size = static_cast<size_t>(M) * W->K;
    static thread_local std::vector<__fp16> tl_code_basis;
    static thread_local std::vector<int8_t> tl_act_i8;
    static thread_local std::vector<float> tl_act_scales;
    if (tl_code_basis.size() < act_size) tl_code_basis.resize(act_size);
    if (tl_act_i8.size() < act_size) tl_act_i8.resize(act_size);
    const size_t scales_size = static_cast<size_t>(M) * num_groups;
    if (tl_act_scales.size() < scales_size) tl_act_scales.resize(scales_size);
    __fp16* code_basis = tl_code_basis.data();
    int8_t* act_i8 = tl_act_i8.data();
    float* act_scales = tl_act_scales.data();
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis);

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t g = 0; g < num_groups; ++g) {
            act_scales[static_cast<size_t>(m) * num_groups + g] = tq_quantize_group_i8(
                code_basis + static_cast<size_t>(m) * W->K + static_cast<size_t>(g) * gs,
                act_i8 + static_cast<size_t>(m) * W->K + static_cast<size_t>(g) * gs,
                gs);
        }
    }

    int8_t cb_i8[16] = {};
    const float cb_scale = tq_quantize_codebook_i8(W->codebook, cb_i8, codebook_size);
    const int8x16_t cb_lut = vld1q_s8(cb_i8);
    const uint32_t n_vecs = gs / 16;
    constexpr uint32_t MAX_M = 5;

    if (M == 2 && (W->N % 8) == 0) {
        const size_t N_blocks8 = W->N / 8;
        CactusThreading::parallel_gemm_tiles(M, N_blocks8,
            [&](size_t block_start, size_t block_end) {
                for (size_t n_block = block_start; n_block < block_end; ++n_block) {
                    const size_t n_start = n_block * 8;
                    float32x4_t running_sum00 = vdupq_n_f32(0.f);
                    float32x4_t running_sum01 = vdupq_n_f32(0.f);
                    float32x4_t running_sum10 = vdupq_n_f32(0.f);
                    float32x4_t running_sum11 = vdupq_n_f32(0.f);

                    for (uint32_t g = 0; g < num_groups; ++g) {
                        const uint8_t* p0 = W->packed_indices
                            + ((static_cast<size_t>(n_start) * num_groups + g) * pgb);
                        const uint8_t* p1 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 1) * num_groups + g) * pgb);
                        const uint8_t* p2 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 2) * num_groups + g) * pgb);
                        const uint8_t* p3 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 3) * num_groups + g) * pgb);
                        const uint8_t* p4 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 4) * num_groups + g) * pgb);
                        const uint8_t* p5 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 5) * num_groups + g) * pgb);
                        const uint8_t* p6 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 6) * num_groups + g) * pgb);
                        const uint8_t* p7 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 7) * num_groups + g) * pgb);
                        float norms_arr0[4] = {
                            static_cast<float>(W->norms[(n_start + 0) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 1) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 2) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 3) * num_groups + g]) * cb_scale,
                        };
                        float norms_arr1[4] = {
                            static_cast<float>(W->norms[(n_start + 4) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 5) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 6) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 7) * num_groups + g]) * cb_scale,
                        };
                        const float32x4_t norms0_v = vld1q_f32(norms_arr0);
                        const float32x4_t norms1_v = vld1q_f32(norms_arr1);
                        const int8_t* a_group0 = act_i8 + static_cast<size_t>(g) * gs;
                        const int8_t* a_group1 = act_i8 + W->K + static_cast<size_t>(g) * gs;
                        int32x4_t acc00 = vdupq_n_s32(0);
                        int32x4_t acc01 = vdupq_n_s32(0);
                        int32x4_t acc10 = vdupq_n_s32(0);
                        int32x4_t acc11 = vdupq_n_s32(0);

                        for (uint32_t v = 0; v < n_vecs; v += 2) {
                            const size_t p_offset0 = (static_cast<size_t>(v) * 16 * bits) / 8;
                            const size_t p_offset1 = (static_cast<size_t>(v + 1) * 16 * bits) / 8;
                            int8x16_t b0, b1, b2, b3, b4, b5, b6, b7;
                            int8x16_t c0, c1, c2, c3, c4, c5, c6, c7;
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p0 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p1 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p2 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p3 + p_offset0, bits, cb_lut),
                                b0, b1, b2, b3);
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p0 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p1 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p2 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p3 + p_offset1, bits, cb_lut),
                                b4, b5, b6, b7);
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p4 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p5 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p6 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p7 + p_offset0, bits, cb_lut),
                                c0, c1, c2, c3);
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p4 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p5 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p6 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p7 + p_offset1, bits, cb_lut),
                                c4, c5, c6, c7);

                            const uint32_t k = v * 16;
                            int8x16_t av = vld1q_s8(a_group0 + k);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b0, av, 0);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b1, av, 1);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b2, av, 2);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b3, av, 3);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c0, av, 0);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c1, av, 1);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c2, av, 2);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c3, av, 3);
                            av = vld1q_s8(a_group0 + k + 16);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b4, av, 0);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b5, av, 1);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b6, av, 2);
                            acc00 = CACTUS_DOTQ_LANE(acc00, b7, av, 3);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c4, av, 0);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c5, av, 1);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c6, av, 2);
                            acc01 = CACTUS_DOTQ_LANE(acc01, c7, av, 3);

                            av = vld1q_s8(a_group1 + k);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b0, av, 0);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b1, av, 1);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b2, av, 2);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b3, av, 3);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c0, av, 0);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c1, av, 1);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c2, av, 2);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c3, av, 3);
                            av = vld1q_s8(a_group1 + k + 16);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b4, av, 0);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b5, av, 1);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b6, av, 2);
                            acc10 = CACTUS_DOTQ_LANE(acc10, b7, av, 3);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c4, av, 0);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c5, av, 1);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c6, av, 2);
                            acc11 = CACTUS_DOTQ_LANE(acc11, c7, av, 3);
                        }

                        const float scale0 = act_scales[g];
                        const float scale1 = act_scales[num_groups + g];
                        running_sum00 = vmlaq_f32(running_sum00,
                            vcvtq_f32_s32(acc00), vmulq_n_f32(norms0_v, scale0));
                        running_sum01 = vmlaq_f32(running_sum01,
                            vcvtq_f32_s32(acc01), vmulq_n_f32(norms1_v, scale0));
                        running_sum10 = vmlaq_f32(running_sum10,
                            vcvtq_f32_s32(acc10), vmulq_n_f32(norms0_v, scale1));
                        running_sum11 = vmlaq_f32(running_sum11,
                            vcvtq_f32_s32(acc11), vmulq_n_f32(norms1_v, scale1));
                    }

                    vst1_f16(C + n_start, vcvt_f16_f32(running_sum00));
                    vst1_f16(C + n_start + 4, vcvt_f16_f32(running_sum01));
                    vst1_f16(C + W->N + n_start, vcvt_f16_f32(running_sum10));
                    vst1_f16(C + W->N + n_start + 4, vcvt_f16_f32(running_sum11));
                }
            });
        return;
    }

    CactusThreading::parallel_gemm_tiles(M, N_blocks,
        [&](size_t block_start, size_t block_end) {
            thread_local std::vector<int8_t> b_interleaved;
            if (b_interleaved.size() < static_cast<size_t>(gs) * 4)
                b_interleaved.resize(static_cast<size_t>(gs) * 4);

            for (size_t n_block = block_start; n_block < block_end; ++n_block) {
                const size_t n_start = n_block * 4;
                const size_t actual_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);

                float32x4_t running_sum[MAX_M];
                for (uint32_t m = 0; m < M; ++m) running_sum[m] = vdupq_n_f32(0.f);

                for (uint32_t g = 0; g < num_groups; ++g) {
                    if ((M == 2 || M == 3) && actual_n == 4) {
                        const uint8_t* p0 = W->packed_indices
                            + ((static_cast<size_t>(n_start) * num_groups + g) * pgb);
                        const uint8_t* p1 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 1) * num_groups + g) * pgb);
                        const uint8_t* p2 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 2) * num_groups + g) * pgb);
                        const uint8_t* p3 = W->packed_indices
                            + ((static_cast<size_t>(n_start + 3) * num_groups + g) * pgb);
                        float norms_arr[4] = {
                            static_cast<float>(W->norms[n_start * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 1) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 2) * num_groups + g]) * cb_scale,
                            static_cast<float>(W->norms[(n_start + 3) * num_groups + g]) * cb_scale,
                        };
                        const float32x4_t norms_v = vld1q_f32(norms_arr);
                        const int8_t* a_group0 = act_i8 + static_cast<size_t>(g) * gs;
                        const int8_t* a_group1 = act_i8 + W->K + static_cast<size_t>(g) * gs;
                        const int8_t* a_group2 = act_i8 + static_cast<size_t>(2) * W->K + static_cast<size_t>(g) * gs;
                        int32x4_t acc0 = vdupq_n_s32(0);
                        int32x4_t acc1 = vdupq_n_s32(0);
                        int32x4_t acc2 = vdupq_n_s32(0);
                        for (uint32_t v = 0; v < n_vecs; v += 2) {
                            const size_t p_offset0 = (static_cast<size_t>(v) * 16 * bits) / 8;
                            const size_t p_offset1 = (static_cast<size_t>(v + 1) * 16 * bits) / 8;
                            int8x16_t b0, b1, b2, b3, b4, b5, b6, b7;
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p0 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p1 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p2 + p_offset0, bits, cb_lut),
                                tq_expand_i8_16(p3 + p_offset0, bits, cb_lut),
                                b0, b1, b2, b3);
                            tq_interleave_4x_s8_vecs(
                                tq_expand_i8_16(p0 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p1 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p2 + p_offset1, bits, cb_lut),
                                tq_expand_i8_16(p3 + p_offset1, bits, cb_lut),
                                b4, b5, b6, b7);

                            const uint32_t k = v * 16;
                            int8x16_t av = vld1q_s8(a_group0 + k);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b0, av, 0);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b1, av, 1);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b2, av, 2);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b3, av, 3);
                            av = vld1q_s8(a_group0 + k + 16);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b4, av, 0);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b5, av, 1);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b6, av, 2);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b7, av, 3);

                            av = vld1q_s8(a_group1 + k);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b0, av, 0);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b1, av, 1);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b2, av, 2);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b3, av, 3);
                            av = vld1q_s8(a_group1 + k + 16);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b4, av, 0);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b5, av, 1);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b6, av, 2);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b7, av, 3);
                            if (M == 3) {
                                av = vld1q_s8(a_group2 + k);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b0, av, 0);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b1, av, 1);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b2, av, 2);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b3, av, 3);
                                av = vld1q_s8(a_group2 + k + 16);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b4, av, 0);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b5, av, 1);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b6, av, 2);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b7, av, 3);
                            }
                        }
                        running_sum[0] = vmlaq_f32(running_sum[0],
                            vcvtq_f32_s32(acc0), vmulq_n_f32(norms_v, act_scales[g]));
                        running_sum[1] = vmlaq_f32(running_sum[1],
                            vcvtq_f32_s32(acc1), vmulq_n_f32(norms_v, act_scales[num_groups + g]));
                        if (M == 3) {
                            running_sum[2] = vmlaq_f32(running_sum[2],
                                vcvtq_f32_s32(acc2), vmulq_n_f32(norms_v, act_scales[static_cast<size_t>(2) * num_groups + g]));
                        }
                        continue;
                    }

                    int8x16_t exp4[4][16];
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + ((static_cast<size_t>(n_start + ni) * num_groups + g) * pgb);
                        for (uint32_t v = 0; v < n_vecs; ++v)
                            exp4[ni][v] = tq_expand_i8_16(p + (v * 16 * bits) / 8, bits, cb_lut);
                    }
                    for (size_t ni = actual_n; ni < 4; ++ni)
                        for (uint32_t v = 0; v < n_vecs; ++v) exp4[ni][v] = vdupq_n_s8(0);

                    float norms_arr[4];
                    for (size_t ni = 0; ni < 4; ++ni) {
                        norms_arr[ni] = (n_start + ni < W->N)
                            ? static_cast<float>(W->norms[(n_start + ni) * num_groups + g]) * cb_scale
                            : 0.f;
                    }
                    const float32x4_t norms_v = vld1q_f32(norms_arr);

                    if (M == 2 || M == 3) {
                        const int8_t* a_group0 = act_i8 + static_cast<size_t>(g) * gs;
                        const int8_t* a_group1 = act_i8 + W->K + static_cast<size_t>(g) * gs;
                        const int8_t* a_group2 = act_i8 + static_cast<size_t>(2) * W->K + static_cast<size_t>(g) * gs;
                        int32x4_t acc0 = vdupq_n_s32(0);
                        int32x4_t acc1 = vdupq_n_s32(0);
                        int32x4_t acc2 = vdupq_n_s32(0);
                        for (uint32_t v = 0; v < n_vecs; v += 2) {
                            int8x16_t b0, b1, b2, b3, b4, b5, b6, b7;
                            tq_interleave_4x_s8_vecs(
                                exp4[0][v], exp4[1][v], exp4[2][v], exp4[3][v],
                                b0, b1, b2, b3);
                            tq_interleave_4x_s8_vecs(
                                exp4[0][v + 1], exp4[1][v + 1], exp4[2][v + 1], exp4[3][v + 1],
                                b4, b5, b6, b7);

                            const uint32_t k = v * 16;
                            int8x16_t av = vld1q_s8(a_group0 + k);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b0, av, 0);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b1, av, 1);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b2, av, 2);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b3, av, 3);
                            av = vld1q_s8(a_group0 + k + 16);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b4, av, 0);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b5, av, 1);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b6, av, 2);
                            acc0 = CACTUS_DOTQ_LANE(acc0, b7, av, 3);

                            av = vld1q_s8(a_group1 + k);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b0, av, 0);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b1, av, 1);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b2, av, 2);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b3, av, 3);
                            av = vld1q_s8(a_group1 + k + 16);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b4, av, 0);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b5, av, 1);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b6, av, 2);
                            acc1 = CACTUS_DOTQ_LANE(acc1, b7, av, 3);
                            if (M == 3) {
                                av = vld1q_s8(a_group2 + k);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b0, av, 0);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b1, av, 1);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b2, av, 2);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b3, av, 3);
                                av = vld1q_s8(a_group2 + k + 16);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b4, av, 0);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b5, av, 1);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b6, av, 2);
                                acc2 = CACTUS_DOTQ_LANE(acc2, b7, av, 3);
                            }
                        }
                        running_sum[0] = vmlaq_f32(running_sum[0],
                            vcvtq_f32_s32(acc0), vmulq_n_f32(norms_v, act_scales[g]));
                        running_sum[1] = vmlaq_f32(running_sum[1],
                            vcvtq_f32_s32(acc1), vmulq_n_f32(norms_v, act_scales[num_groups + g]));
                        if (M == 3) {
                            running_sum[2] = vmlaq_f32(running_sum[2],
                                vcvtq_f32_s32(acc2), vmulq_n_f32(norms_v, act_scales[static_cast<size_t>(2) * num_groups + g]));
                        }
                    } else {
                        for (uint32_t v = 0; v < n_vecs; ++v) {
                            tq_interleave_4x_s8(exp4[0][v], exp4[1][v], exp4[2][v], exp4[3][v],
                                                b_interleaved.data() + static_cast<size_t>(v) * 64);
                        }
                        for (uint32_t m = 0; m < M; ++m) {
                            const int8_t* a_group = act_i8
                                + static_cast<size_t>(m) * W->K + static_cast<size_t>(g) * gs;
                            int32x4_t acc = vdupq_n_s32(0);
                            for (uint32_t k = 0; k < gs; k += 32) {
                                const int8_t* b_base = b_interleaved.data() + static_cast<size_t>(k) * 4;
                                int8x16_t av = vld1q_s8(a_group + k);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base),      av, 0);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 16), av, 1);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 32), av, 2);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 48), av, 3);
                                av = vld1q_s8(a_group + k + 16);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 64), av, 0);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 80), av, 1);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 96), av, 2);
                                acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + 112), av, 3);
                            }
                            const float scale = act_scales[static_cast<size_t>(m) * num_groups + g];
                            running_sum[m] = vmlaq_f32(running_sum[m],
                                vcvtq_f32_s32(acc), vmulq_n_f32(norms_v, scale));
                        }
                    }
                }

                for (uint32_t m = 0; m < M; ++m) {
                    float16x4_t r = vcvt_f16_f32(running_sum[m]);
                    if (actual_n == 4) {
                        vst1_f16(C + static_cast<size_t>(m) * W->N + n_start, r);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ++ni) {
                            C[static_cast<size_t>(m) * W->N + n_start + ni] = vget_lane_f16(r, 0);
                            r = vext_f16(r, r, 1);
                        }
                    }
                }
            }
        });
}

void cactus_quant_matmul(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    CactusQuantMatmulTrace trace(W, M);
    if (W != nullptr && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) != 0) {
        trace.path = "orthogonal_dispatch";
        cactus_quant_orthogonal_matmul(W, A, M, C);
        return;
    }
    if (!cactus_quant_valid_common(W, A, C) || M == 0) {
        trace.path = "invalid";
        return;
    }

    const uint32_t gs = W->group_size;
    const uint32_t bits = W->bits;
    const uint32_t num_groups = W->num_groups;
    const uint32_t pgb = cactus_quant_packed_group_bytes(bits, gs);
    const bool has_expanded_layout = W->expanded != nullptr && W->norm_f32 != nullptr;
    const bool has_sdot_group_layout = (gs % 32) == 0;
    const size_t N_blocks = (W->N + 3) / 4;
    const uint32_t codebook_size = 1u << bits;
    const size_t expanded_size = N_blocks * num_groups * gs * 4;
    const size_t expanded_norms_size = N_blocks * num_groups * 4;

    if (M == 1) {
        if (!has_expanded_layout || !has_sdot_group_layout) {
            if (bits == 4 && (gs % 16) == 0) { trace.path = "m1_tq4_gemv"; cactus_quant_4bit_gemv(W, A, C); return; }
            if (bits == 2 && (gs % 8) == 0) { trace.path = "m1_tq2_gemv"; cactus_quant_2bit_gemv(W, A, C); return; }
            if (bits == 1 && (gs % 8) == 0) { trace.path = "m1_tq1_gemv"; cactus_quant_1bit_gemv(W, A, C); return; }
            if (bits == 3 && (gs % 8) == 0) { trace.path = "m1_tq3_gemv"; cactus_quant_3bit_gemv(W, A, C); return; }
        }

        if (!has_sdot_group_layout) {
            trace.path = "m1_group_dispatch";
            cactus_quant_dispatch_group_gemm(W, A, M, C);
            return;
        }

        const size_t act_size = W->K;
        static thread_local std::vector<__fp16> tl_code_basis;
        if (tl_code_basis.size() < act_size) tl_code_basis.resize(act_size);
        __fp16* code_basis_ptr = tl_code_basis.data();
        cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_ptr);

        // INT8 SDOT GEMV
        thread_local std::vector<int8_t> tl_act_i8;
        thread_local std::vector<float> tl_act_scales;
        if (tl_act_i8.size() < act_size) tl_act_i8.resize(act_size);
        if (tl_act_scales.size() < num_groups) tl_act_scales.resize(num_groups);

        for (uint32_t g = 0; g < num_groups; g++) {
            tl_act_scales[g] = tq_quantize_group_i8(
                code_basis_ptr + g * gs,
                tl_act_i8.data() + g * gs, gs);
        }

        const int8_t* act_i8 = tl_act_i8.data();
        const float* act_scales = tl_act_scales.data();

        // Use pre-cached expanded weights if available, otherwise build inline
        const int8_t* w_il = has_expanded_layout ? W->expanded : nullptr;
        const float* n_f32 = has_expanded_layout ? W->norm_f32 : nullptr;

        int8_t cb_i8[16] = {};
        const float cb_scale = tq_quantize_codebook_i8(W->codebook, cb_i8, codebook_size);
        const int8x16_t cb_lut = vld1q_s8(cb_i8);

        std::vector<int8_t> w_il_buf;
        std::vector<float> n_f32_buf;
        if (!w_il) {
            trace.path = "m1_sdot_inline_expand";
            w_il_buf.resize(expanded_size);
            n_f32_buf.resize(expanded_norms_size);
            tq_preexpand_weights(W, bits, num_groups, gs, pgb, cb_lut, cb_scale,
                                 N_blocks, w_il_buf.data(), n_f32_buf.data());
            w_il = w_il_buf.data();
            n_f32 = n_f32_buf.data();
        } else {
            trace.path = "m1_sdot_preexpanded";
        }

        auto& pool = CactusThreading::get_thread_pool();
        size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
        num_threads = std::min(num_threads, N_blocks);

        auto process_blocks = [&](size_t block_start, size_t block_end) {
            for (size_t n_block = block_start; n_block < block_end; ++n_block) {
                float32x4_t running_sum = vdupq_n_f32(0.f);
                const size_t nb_off = n_block * num_groups;

                uint32_t g = 0;
                for (; g + 1 < num_groups; g += 2) {
                    const int8_t* a0 = act_i8 + g * gs;
                    const int8_t* a1 = act_i8 + (g + 1) * gs;
                    const int8_t* b0 = w_il + (nb_off + g) * gs * 4;
                    const int8_t* b1 = w_il + (nb_off + g + 1) * gs * 4;

                    __builtin_prefetch(b1 + gs * 4, 0, 3);

                    int32x4_t acc0 = vdupq_n_s32(0);
                    int32x4_t acc1 = vdupq_n_s32(0);

                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a0 + k);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4),      av, 0);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 16), av, 1);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 32), av, 2);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 48), av, 3);
                        av = vld1q_s8(a0 + k + 16);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 64), av, 0);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 80), av, 1);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 96), av, 2);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 112),av, 3);
                    }

                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a1 + k);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4),      av, 0);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 16), av, 1);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 32), av, 2);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 48), av, 3);
                        av = vld1q_s8(a1 + k + 16);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 64), av, 0);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 80), av, 1);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 96), av, 2);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 112),av, 3);
                    }

                    float32x4_t s0 = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g) * 4), act_scales[g]);
                    float32x4_t s1 = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g + 1) * 4), act_scales[g + 1]);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc0), s0);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc1), s1);
                }

                for (; g < num_groups; ++g) {
                    const int8_t* a_group = act_i8 + g * gs;
                    const int8_t* b_base = w_il + (nb_off + g) * gs * 4;

                    int32x4_t acc = vdupq_n_s32(0);
                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a_group + k);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4),      av, 0);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 16), av, 1);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 32), av, 2);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 48), av, 3);
                        av = vld1q_s8(a_group + k + 16);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 64), av, 0);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 80), av, 1);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 96), av, 2);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 112),av, 3);
                    }

                    float32x4_t scale = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g) * 4), act_scales[g]);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scale);
                }

                const size_t n_start = n_block * 4;
                const size_t actual_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);
                float16x4_t result = vcvt_f16_f32(running_sum);
                if (actual_n == 4) {
                    vst1_f16(C + n_start, result);
                } else {
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        C[n_start + ni] = vget_lane_f16(result, 0);
                        result = vext_f16(result, result, 1);
                    }
                }
            }
        };

        if (num_threads <= 1) {
            process_blocks(0, N_blocks);
        } else {
            pool.enqueue_n_threads(N_blocks, num_threads, process_blocks);
            pool.wait_all();
        }

        return;
    }

    if (!has_sdot_group_layout) {
        trace.path = "group_dispatch";
        cactus_quant_dispatch_group_gemm(W, A, M, C);
        return;
    }

    if (!has_expanded_layout && M > 1 && M <= 5) {
        trace.path = "small_m_no_preexpand";
        cactus_quant_small_m_sdot_gemm_no_preexpand(
            W, A, M, C, bits, num_groups, gs, pgb, codebook_size, N_blocks);
        return;
    }

    if (!has_expanded_layout && M <= 8) {
        trace.path = "small_m_loop_m1";
        for (uint32_t m = 0; m < M; ++m) {
            cactus_quant_matmul(W, A + static_cast<size_t>(m) * W->K, 1, C + static_cast<size_t>(m) * W->N);
        }
        return;
    }

    trace.path = "gemm_preexpand_amortized";
    const size_t act_size = static_cast<size_t>(M) * W->K;
    std::vector<__fp16> heap_code_basis(act_size);
    __fp16* code_basis_ptr = heap_code_basis.data();
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_ptr);

    // M > 1: full pre-expansion amortized over M rows
    const size_t scales_size = static_cast<size_t>(M) * num_groups;
    std::vector<int8_t> heap_act_i8(act_size);
    std::vector<float> heap_act_scales(scales_size);
    int8_t* act_i8_ptr = heap_act_i8.data();
    float* act_scales_ptr = heap_act_scales.data();

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t g = 0; g < num_groups; g++) {
            act_scales_ptr[m * num_groups + g] = tq_quantize_group_i8(
                code_basis_ptr + m * W->K + g * gs,
                act_i8_ptr + m * W->K + g * gs, gs);
        }
    }

    constexpr size_t TILE_M = 8;

    int8_t cb_i8[16] = {};
    const float cb_scale = tq_quantize_codebook_i8(W->codebook, cb_i8, codebook_size);
    const int8x16_t cb_lut = vld1q_s8(cb_i8);

    std::vector<int8_t> w_il_buf;
    std::vector<float> n_f32_buf;
    w_il_buf.resize(expanded_size);
    n_f32_buf.resize(expanded_norms_size);
    tq_preexpand_weights(W, bits, num_groups, gs, pgb, cb_lut, cb_scale,
                         N_blocks, w_il_buf.data(), n_f32_buf.data());
    const int8_t* w_il = w_il_buf.data();
    const float* n_f32 = n_f32_buf.data();

    const size_t M_blocks = (M + TILE_M - 1) / TILE_M;
    const size_t total_tiles = M_blocks * N_blocks;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [&](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                const size_t m_block = tile_idx / N_blocks;
                const size_t n_block = tile_idx % N_blocks;
                const size_t m_start = m_block * TILE_M;
                const size_t m_end = std::min(m_start + TILE_M, static_cast<size_t>(M));
                const size_t actual_m = m_end - m_start;
                const size_t n_start = n_block * 4;
                const size_t actual_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);

                const int8_t* a_rows[TILE_M];
                for (size_t mi = 0; mi < TILE_M; mi++) {
                    size_t row = m_start + (mi < actual_m ? mi : actual_m - 1);
                    a_rows[mi] = act_i8_ptr + row * W->K;
                }

                float32x4_t running_sum[TILE_M];
                for (size_t mi = 0; mi < TILE_M; mi++)
                    running_sum[mi] = vdupq_n_f32(0.f);

                for (uint32_t g = 0; g < num_groups; ++g) {
                    const int8_t* b_base = w_il + (n_block * num_groups + g) * gs * 4;
                    float32x4_t norms_v = vld1q_f32(n_f32 + (n_block * num_groups + g) * 4);

                    __builtin_prefetch(b_base + gs * 4, 0, 3);

                    int32x4_t row_acc[TILE_M];
                    for (size_t mi = 0; mi < TILE_M; mi++) row_acc[mi] = vdupq_n_s32(0);

                    for (uint32_t k = 0; k < gs; k += 32) {
                        const int8_t* bk = b_base + k * 4;
                        int8x16_t b00 = vld1q_s8(bk);
                        int8x16_t b01 = vld1q_s8(bk + 16);
                        int8x16_t b02 = vld1q_s8(bk + 32);
                        int8x16_t b03 = vld1q_s8(bk + 48);
                        int8x16_t b10 = vld1q_s8(bk + 64);
                        int8x16_t b11 = vld1q_s8(bk + 80);
                        int8x16_t b12 = vld1q_s8(bk + 96);
                        int8x16_t b13 = vld1q_s8(bk + 112);

#define TQ_GEMM_ROW(ROW) do { \
                            const int8_t* ap = a_rows[ROW] + g * gs + k; \
                            int8x16_t a_lo = vld1q_s8(ap); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b00, a_lo, 0); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b01, a_lo, 1); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b02, a_lo, 2); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b03, a_lo, 3); \
                            int8x16_t a_hi = vld1q_s8(ap + 16); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b10, a_hi, 0); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b11, a_hi, 1); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b12, a_hi, 2); \
                            row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b13, a_hi, 3); \
                        } while(0)

                        TQ_GEMM_ROW(0);
                        TQ_GEMM_ROW(1);
                        TQ_GEMM_ROW(2);
                        TQ_GEMM_ROW(3);
                        TQ_GEMM_ROW(4);
                        TQ_GEMM_ROW(5);
                        TQ_GEMM_ROW(6);
                        TQ_GEMM_ROW(7);
                        #undef TQ_GEMM_ROW
                    }

                    for (size_t mi = 0; mi < actual_m; ++mi) {
                        float a_sc = act_scales_ptr[(m_start + mi) * num_groups + g];
                        running_sum[mi] = vmlaq_f32(running_sum[mi],
                            vcvtq_f32_s32(row_acc[mi]), vmulq_n_f32(norms_v, a_sc));
                    }
                }

                for (size_t mi = 0; mi < actual_m; ++mi) {
                    float16x4_t r = vcvt_f16_f32(running_sum[mi]);
                    if (actual_n == 4) {
                        vst1_f16(C + (m_start + mi) * W->N + n_start, r);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            C[(m_start + mi) * W->N + n_start + ni] = vget_lane_f16(r, 0);
                            r = vext_f16(r, r, 1);
                        }
                    }
                }
            }
        });
}

static void tq_fwht_normalized_f32(std::vector<float>& x) {
    const uint32_t n = static_cast<uint32_t>(x.size());
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; ++j) {
                float a = x[j];
                float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (float& v : x) v *= inv;
}

void cactus_quant_dequantize_hadamard_embedding_row(
    uint32_t bits,
    uint32_t hidden_dim,
    uint32_t group_size,
    uint32_t num_groups,
    size_t row,
    const uint8_t* packed_base,
    const __fp16* codebook,
    const __fp16* norms,
    const __fp16* input_scale_recip,
    const int8_t* left_signs,
    const int8_t* right_signs,
    const uint32_t* permutation,
    __fp16* out_row) {
    if (!packed_base || !codebook || !norms || !out_row || bits == 0 || bits > 4) return;
    if (hidden_dim == 0 || group_size == 0 || num_groups == 0) return;
    if ((group_size & (group_size - 1)) != 0) return;
    if (group_size > 256) return;

    const uint32_t packed_group_bytes = cactus_quant_packed_group_bytes(bits, group_size);
    std::vector<float> rotated(group_size);
    for (uint32_t g = 0; g < num_groups; ++g) {
        std::fill(rotated.begin(), rotated.end(), 0.0f);
        const uint8_t* packed = packed_base + (static_cast<size_t>(row) * num_groups + g) * packed_group_bytes;

        for (uint32_t k = 0; k < group_size; ++k) {
            uint8_t idx = static_cast<uint8_t>(tq_extract_idx(packed, k, bits));
            uint32_t dst = permutation ? permutation[k] : k;
            float rs = right_signs ? static_cast<float>(right_signs[dst]) : 1.0f;
            rotated[dst] = static_cast<float>(codebook[idx]) * rs;
        }

        tq_fwht_normalized_f32(rotated);

        const float norm = static_cast<float>(norms[static_cast<size_t>(row) * num_groups + g]);
        for (uint32_t k = 0; k < group_size; ++k) {
            uint32_t col = g * group_size + k;
            float ls = left_signs ? static_cast<float>(left_signs[k]) : 1.0f;
            float scale = input_scale_recip ? static_cast<float>(input_scale_recip[col]) : 1.0f;
            out_row[col] = static_cast<__fp16>(rotated[k] * ls * norm * scale);
        }
    }

    for (uint32_t col = num_groups * group_size; col < hidden_dim; ++col) {
        out_row[col] = static_cast<__fp16>(0);
    }
}

void cactus_quant_dequantize_orthogonal_embedding_row(
    uint32_t bits,
    uint32_t K,
    size_t row,
    const uint8_t* packed_base,
    const __fp16* codebook,
    const __fp16* norms,
    const __fp16* input_scale_recip,
    const __fp16* rotation,
    __fp16* out_row) {
    if (!packed_base || !codebook || !norms || !rotation || !out_row || bits == 0 || bits > 4) return;
    if (K == 0) return;

    const uint32_t packed_group_bytes = cactus_quant_packed_group_bytes(bits, K);
    const uint8_t* packed = packed_base + static_cast<size_t>(row) * packed_group_bytes;
    const float norm = static_cast<float>(norms[row]);

    std::vector<float> dq(K);
    for (uint32_t i = 0; i < K; ++i) {
        dq[i] = static_cast<float>(codebook[tq_extract_idx(packed, i, bits)]);
    }

    for (uint32_t j = 0; j < K; ++j) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            acc += dq[i] * static_cast<float>(rotation[static_cast<size_t>(j) * K + i]);
        }
        float scale = input_scale_recip ? static_cast<float>(input_scale_recip[j]) : 1.0f;
        out_row[j] = static_cast<__fp16>(acc * norm * scale);
    }
}

static void cactus_quant_orthogonal_rotate_input(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    float* A_rot) {
    const uint32_t K = W->K;
    std::fill(A_rot, A_rot + static_cast<size_t>(M) * K, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        const __fp16* a_row = A + static_cast<size_t>(m) * K;
        float* ar_row = A_rot + static_cast<size_t>(m) * K;
        for (uint32_t k = 0; k < K; ++k) {
            float a_val = static_cast<float>(a_row[k]);
            if (W->input_scale_recip) a_val *= static_cast<float>(W->input_scale_recip[k]);
            if (a_val == 0.0f) continue;
            const __fp16* R_row_k = W->rotation + static_cast<size_t>(k) * K;
            float32x4_t av = vdupq_n_f32(a_val);
            uint32_t i = 0;
            for (; i + 8 <= K; i += 8) {
                float16x8_t rv = vld1q_f16(R_row_k + i);
                float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(rv));
                float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(rv));
                vst1q_f32(ar_row + i,     vfmaq_f32(vld1q_f32(ar_row + i),     av, r_lo));
                vst1q_f32(ar_row + i + 4, vfmaq_f32(vld1q_f32(ar_row + i + 4), av, r_hi));
            }
            for (; i < K; ++i)
                ar_row[i] += a_val * static_cast<float>(R_row_k[i]);
        }
    }
}

static void cactus_f32_to_f16(const float* src, __fp16* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
    }
    for (; i < count; ++i) dst[i] = static_cast<__fp16>(src[i]);
}

static void cactus_tq4_codebook_byte_tables(const float* cb_f32, uint8_t* cb_lo_arr, uint8_t* cb_hi_arr) {
    __fp16 cb_f16[16];
    for (uint32_t c = 0; c < 16; ++c) {
        cb_f16[c] = static_cast<__fp16>(cb_f32[c]);
        cb_lo_arr[c] = reinterpret_cast<const uint8_t*>(&cb_f16[c])[0];
        cb_hi_arr[c] = reinterpret_cast<const uint8_t*>(&cb_f16[c])[1];
    }
}

void cactus_quant_orthogonal_matmul(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    CactusQuantMatmulTrace trace(W, M);
    trace.path = "orthogonal_matmul";
    if (!W || !A || !C || M == 0) return;
    if (!W->rotation || !W->packed_indices || !W->codebook || !W->norms) return;

    const uint32_t K    = W->K;
    const uint32_t N    = W->N;
    const uint32_t bits = W->bits;
    const uint32_t pgb  = cactus_quant_packed_group_bytes(bits, K);

    if (K == 0 || N == 0 || pgb == 0 || bits == 0 || bits > 4) return;

    const uint32_t n_cb = 1u << bits;

    float cb_f32[16] = {};
    for (uint32_t c = 0; c < n_cb; ++c)
        cb_f32[c] = static_cast<float>(W->codebook[c]);

    std::vector<float> A_rot(static_cast<size_t>(M) * K);
    cactus_quant_orthogonal_rotate_input(W, A, M, A_rot.data());

    if (bits == 4 && (K % 16) == 0) {
        uint8_t cb_lo_arr[16], cb_hi_arr[16];
        cactus_tq4_codebook_byte_tables(cb_f32, cb_lo_arr, cb_hi_arr);
        uint8x16_t cb_lo_tbl = vld1q_u8(cb_lo_arr);
        uint8x16_t cb_hi_tbl = vld1q_u8(cb_hi_arr);

        std::vector<__fp16> A_rot_f16(static_cast<size_t>(M) * K);
        for (uint32_t m = 0; m < M; ++m) {
            const float* ar = A_rot.data() + static_cast<size_t>(m) * K;
            __fp16* arf = A_rot_f16.data() + static_cast<size_t>(m) * K;
            cactus_f32_to_f16(ar, arf, K);
        }

        CactusThreading::parallel_for(
            static_cast<size_t>(N),
            CactusThreading::ParallelConfig{64, 1},
            [&](size_t n_start, size_t n_end) {
                for (size_t n = n_start; n < n_end; ++n) {
                    const uint8_t* packed = W->packed_indices + n * pgb;
                    float norm_n = static_cast<float>(W->norms[n]);
                    for (uint32_t m = 0; m < M; ++m) {
                        const __fp16* arf = A_rot_f16.data() + static_cast<size_t>(m) * K;
                        float32x4_t acc0 = vdupq_n_f32(0.f);
                        float32x4_t acc1 = vdupq_n_f32(0.f);
                        float32x4_t acc2 = vdupq_n_f32(0.f);
                        float32x4_t acc3 = vdupq_n_f32(0.f);

                        for (uint32_t i = 0; i < K; i += 16) {
                            uint8x8_t raw8 = vld1_u8(packed + i / 2);
                            uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                            uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                            uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                            uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);

                            uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                            uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);

                            uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                            float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                            float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);

                            float16x8_t ar_lo = vld1q_f16(arf + i);
                            float16x8_t ar_hi = vld1q_f16(arf + i + 8);

                            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)),
                                                   vcvt_f32_f16(vget_low_f16(ar_lo)));
                            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)),
                                                   vcvt_f32_f16(vget_high_f16(ar_lo)));
                            acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)),
                                                   vcvt_f32_f16(vget_low_f16(ar_hi)));
                            acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)),
                                                   vcvt_f32_f16(vget_high_f16(ar_hi)));
                        }

                        float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                                          vaddq_f32(acc2, acc3)));
                        C[static_cast<size_t>(m) * N + n] = static_cast<__fp16>(acc * norm_n);
                    }
                }
            });
        return;
    }

    CactusThreading::parallel_for(
        static_cast<size_t>(N),
        CactusThreading::ParallelConfig{64, 1},
        [&](size_t n_start, size_t n_end) {
            for (size_t n = n_start; n < n_end; ++n) {
                const uint8_t* packed = W->packed_indices + n * pgb;
                float norm_n = static_cast<float>(W->norms[n]);
                for (uint32_t m = 0; m < M; ++m) {
                    const float* ar = A_rot.data() + static_cast<size_t>(m) * K;
                    float acc = 0.0f;
                    for (uint32_t i = 0; i < K; ++i) {
                        uint8_t idx = static_cast<uint8_t>(tq_extract_idx(packed, i, bits));
                        acc += cb_f32[idx] * ar[i];
                    }
                    C[static_cast<size_t>(m) * N + n] = static_cast<__fp16>(acc * norm_n);
                }
            }
        });
}

bool cactus_quant_orthogonal_argmax(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    uint32_t* indices) {
    CactusQuantArgmaxTrace trace(W, M);
    if (!W || !A || !indices || M == 0) return false;
    if (!W->rotation || !W->packed_indices || !W->codebook || !W->norms) return false;

    const uint32_t K    = W->K;
    const uint32_t N    = W->N;
    const uint32_t bits = W->bits;
    const uint32_t pgb  = cactus_quant_packed_group_bytes(bits, K);

    if (K == 0 || N == 0 || pgb == 0 || bits == 0 || bits > 4) return false;

    const uint32_t n_cb = 1u << bits;

    float cb_f32[16] = {};
    for (uint32_t c = 0; c < n_cb; ++c)
        cb_f32[c] = static_cast<float>(W->codebook[c]);

    static thread_local std::vector<float> argmax_a_rot;
    const size_t a_rot_size = static_cast<size_t>(M) * K;
    if (argmax_a_rot.size() < a_rot_size) argmax_a_rot.resize(a_rot_size);
    float* a_rot = argmax_a_rot.data();
    cactus_quant_orthogonal_rotate_input(W, A, M, a_rot);

    static thread_local std::vector<__fp16> argmax_a_rot_f16;
    __fp16* a_rot_f16 = nullptr;
    if (bits == 4 && (K % 16) == 0) {
        if (argmax_a_rot_f16.size() < a_rot_size) argmax_a_rot_f16.resize(a_rot_size);
        a_rot_f16 = argmax_a_rot_f16.data();
        for (uint32_t m = 0; m < M; ++m) {
            const float* ar = a_rot + static_cast<size_t>(m) * K;
            __fp16* arf = a_rot_f16 + static_cast<size_t>(m) * K;
            cactus_f32_to_f16(ar, arf, K);
        }
    }

    auto& pool = CactusThreading::get_thread_pool();
    constexpr CactusThreading::ParallelConfig projection_config{64, 1};
    size_t num_threads = CactusThreading::get_optimal_thread_count(N, projection_config);

    std::vector<float> best_values(num_threads * M, -std::numeric_limits<float>::infinity());
    std::vector<uint32_t> best_indices(num_threads * M, 0);
    std::atomic<size_t> next_slot{0};
    uint8_t cb_lo_arr[16] = {}, cb_hi_arr[16] = {};
    if (bits == 4 && (K % 16) == 0) {
        cactus_tq4_codebook_byte_tables(cb_f32, cb_lo_arr, cb_hi_arr);
    }
    uint8x16_t cb_lo_tbl = vld1q_u8(cb_lo_arr);
    uint8x16_t cb_hi_tbl = vld1q_u8(cb_hi_arr);

    auto update_best = [&](size_t slot, uint32_t m, float value, uint32_t idx) {
        float& best_value = best_values[slot * M + m];
        uint32_t& best_index = best_indices[slot * M + m];
        if (value > best_value || (value == best_value && idx < best_index)) {
            best_value = value;
            best_index = idx;
        }
    };

    auto process_range = [&](size_t n_start, size_t n_end) {
        const size_t slot = next_slot.fetch_add(1, std::memory_order_relaxed);
        if (bits == 4 && (K % 16) == 0) {
            if (M == 2) {
                const __fp16* arf0 = a_rot_f16;
                const __fp16* arf1 = a_rot_f16 + K;
                size_t n = n_start;
                for (; n + 1 < n_end; n += 2) {
                    const uint8_t* packed0 = W->packed_indices + n * pgb;
                    const uint8_t* packed1 = W->packed_indices + (n + 1) * pgb;
                    float norm0 = static_cast<float>(W->norms[n]);
                    float norm1 = static_cast<float>(W->norms[n + 1]);
                    float32x4_t acc00 = vdupq_n_f32(0.f);
                    float32x4_t acc01 = vdupq_n_f32(0.f);
                    float32x4_t acc02 = vdupq_n_f32(0.f);
                    float32x4_t acc03 = vdupq_n_f32(0.f);
                    float32x4_t acc04 = vdupq_n_f32(0.f);
                    float32x4_t acc05 = vdupq_n_f32(0.f);
                    float32x4_t acc06 = vdupq_n_f32(0.f);
                    float32x4_t acc07 = vdupq_n_f32(0.f);
                    float32x4_t acc10 = vdupq_n_f32(0.f);
                    float32x4_t acc11 = vdupq_n_f32(0.f);
                    float32x4_t acc12 = vdupq_n_f32(0.f);
                    float32x4_t acc13 = vdupq_n_f32(0.f);
                    float32x4_t acc14 = vdupq_n_f32(0.f);
                    float32x4_t acc15 = vdupq_n_f32(0.f);
                    float32x4_t acc16 = vdupq_n_f32(0.f);
                    float32x4_t acc17 = vdupq_n_f32(0.f);

                    for (uint32_t i = 0; i < K; i += 16) {
                        uint8x8_t raw0 = vld1_u8(packed0 + i / 2);
                        uint8x8_t raw1 = vld1_u8(packed1 + i / 2);
                        uint8x8_t lo0_nibs = vand_u8(raw0, vdup_n_u8(0x0F));
                        uint8x8_t hi0_nibs = vshr_n_u8(raw0, 4);
                        uint8x8_t lo1_nibs = vand_u8(raw1, vdup_n_u8(0x0F));
                        uint8x8_t hi1_nibs = vshr_n_u8(raw1, 4);
                        uint8x8x2_t zipped0 = vzip_u8(lo0_nibs, hi0_nibs);
                        uint8x8x2_t zipped1 = vzip_u8(lo1_nibs, hi1_nibs);
                        uint8x16_t indices0 = vcombine_u8(zipped0.val[0], zipped0.val[1]);
                        uint8x16_t indices1 = vcombine_u8(zipped1.val[0], zipped1.val[1]);

                        uint8x16_t lo0_bytes = vqtbl1q_u8(cb_lo_tbl, indices0);
                        uint8x16_t hi0_bytes = vqtbl1q_u8(cb_hi_tbl, indices0);
                        uint8x16_t lo1_bytes = vqtbl1q_u8(cb_lo_tbl, indices1);
                        uint8x16_t hi1_bytes = vqtbl1q_u8(cb_hi_tbl, indices1);

                        uint8x16x2_t fp160_bytes = vzipq_u8(lo0_bytes, hi0_bytes);
                        uint8x16x2_t fp161_bytes = vzipq_u8(lo1_bytes, hi1_bytes);
                        float16x8_t cb0_lo = vreinterpretq_f16_u8(fp160_bytes.val[0]);
                        float16x8_t cb0_hi = vreinterpretq_f16_u8(fp160_bytes.val[1]);
                        float16x8_t cb1_lo = vreinterpretq_f16_u8(fp161_bytes.val[0]);
                        float16x8_t cb1_hi = vreinterpretq_f16_u8(fp161_bytes.val[1]);

                        float16x8_t ar0_lo = vld1q_f16(arf0 + i);
                        float16x8_t ar0_hi = vld1q_f16(arf0 + i + 8);
                        float16x8_t ar1_lo = vld1q_f16(arf1 + i);
                        float16x8_t ar1_hi = vld1q_f16(arf1 + i + 8);

                        float32x4_t ar0_lo0 = vcvt_f32_f16(vget_low_f16(ar0_lo));
                        float32x4_t ar0_lo1 = vcvt_f32_f16(vget_high_f16(ar0_lo));
                        float32x4_t ar0_hi0 = vcvt_f32_f16(vget_low_f16(ar0_hi));
                        float32x4_t ar0_hi1 = vcvt_f32_f16(vget_high_f16(ar0_hi));
                        float32x4_t ar1_lo0 = vcvt_f32_f16(vget_low_f16(ar1_lo));
                        float32x4_t ar1_lo1 = vcvt_f32_f16(vget_high_f16(ar1_lo));
                        float32x4_t ar1_hi0 = vcvt_f32_f16(vget_low_f16(ar1_hi));
                        float32x4_t ar1_hi1 = vcvt_f32_f16(vget_high_f16(ar1_hi));
                        float32x4_t cb0_lo0 = vcvt_f32_f16(vget_low_f16(cb0_lo));
                        float32x4_t cb0_lo1 = vcvt_f32_f16(vget_high_f16(cb0_lo));
                        float32x4_t cb0_hi0 = vcvt_f32_f16(vget_low_f16(cb0_hi));
                        float32x4_t cb0_hi1 = vcvt_f32_f16(vget_high_f16(cb0_hi));
                        float32x4_t cb1_lo0 = vcvt_f32_f16(vget_low_f16(cb1_lo));
                        float32x4_t cb1_lo1 = vcvt_f32_f16(vget_high_f16(cb1_lo));
                        float32x4_t cb1_hi0 = vcvt_f32_f16(vget_low_f16(cb1_hi));
                        float32x4_t cb1_hi1 = vcvt_f32_f16(vget_high_f16(cb1_hi));

                        acc00 = vfmaq_f32(acc00, cb0_lo0, ar0_lo0);
                        acc01 = vfmaq_f32(acc01, cb0_lo1, ar0_lo1);
                        acc02 = vfmaq_f32(acc02, cb0_hi0, ar0_hi0);
                        acc03 = vfmaq_f32(acc03, cb0_hi1, ar0_hi1);
                        acc04 = vfmaq_f32(acc04, cb1_lo0, ar0_lo0);
                        acc05 = vfmaq_f32(acc05, cb1_lo1, ar0_lo1);
                        acc06 = vfmaq_f32(acc06, cb1_hi0, ar0_hi0);
                        acc07 = vfmaq_f32(acc07, cb1_hi1, ar0_hi1);
                        acc10 = vfmaq_f32(acc10, cb0_lo0, ar1_lo0);
                        acc11 = vfmaq_f32(acc11, cb0_lo1, ar1_lo1);
                        acc12 = vfmaq_f32(acc12, cb0_hi0, ar1_hi0);
                        acc13 = vfmaq_f32(acc13, cb0_hi1, ar1_hi1);
                        acc14 = vfmaq_f32(acc14, cb1_lo0, ar1_lo0);
                        acc15 = vfmaq_f32(acc15, cb1_lo1, ar1_lo1);
                        acc16 = vfmaq_f32(acc16, cb1_hi0, ar1_hi0);
                        acc17 = vfmaq_f32(acc17, cb1_hi1, ar1_hi1);
                    }

                    float acc0 = vaddvq_f32(vaddq_f32(vaddq_f32(acc00, acc01),
                                                       vaddq_f32(acc02, acc03)));
                    float acc1 = vaddvq_f32(vaddq_f32(vaddq_f32(acc10, acc11),
                                                       vaddq_f32(acc12, acc13)));
                    float acc2 = vaddvq_f32(vaddq_f32(vaddq_f32(acc04, acc05),
                                                       vaddq_f32(acc06, acc07)));
                    float acc3 = vaddvq_f32(vaddq_f32(vaddq_f32(acc14, acc15),
                                                       vaddq_f32(acc16, acc17)));
                    update_best(slot, 0, static_cast<float>(static_cast<__fp16>(acc0 * norm0)), static_cast<uint32_t>(n));
                    update_best(slot, 1, static_cast<float>(static_cast<__fp16>(acc1 * norm0)), static_cast<uint32_t>(n));
                    update_best(slot, 0, static_cast<float>(static_cast<__fp16>(acc2 * norm1)), static_cast<uint32_t>(n + 1));
                    update_best(slot, 1, static_cast<float>(static_cast<__fp16>(acc3 * norm1)), static_cast<uint32_t>(n + 1));
                }
                for (; n < n_end; ++n) {
                    const uint8_t* packed = W->packed_indices + n * pgb;
                    float norm_n = static_cast<float>(W->norms[n]);
                    float32x4_t acc00 = vdupq_n_f32(0.f);
                    float32x4_t acc01 = vdupq_n_f32(0.f);
                    float32x4_t acc02 = vdupq_n_f32(0.f);
                    float32x4_t acc03 = vdupq_n_f32(0.f);
                    float32x4_t acc10 = vdupq_n_f32(0.f);
                    float32x4_t acc11 = vdupq_n_f32(0.f);
                    float32x4_t acc12 = vdupq_n_f32(0.f);
                    float32x4_t acc13 = vdupq_n_f32(0.f);
                    for (uint32_t i = 0; i < K; i += 16) {
                        uint8x8_t raw8 = vld1_u8(packed + i / 2);
                        uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                        uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                        uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                        uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);
                        uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                        uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);
                        uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                        float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                        float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);
                        float16x8_t ar0_lo = vld1q_f16(arf0 + i);
                        float16x8_t ar0_hi = vld1q_f16(arf0 + i + 8);
                        float16x8_t ar1_lo = vld1q_f16(arf1 + i);
                        float16x8_t ar1_hi = vld1q_f16(arf1 + i + 8);
                        float32x4_t cb_lo_lo = vcvt_f32_f16(vget_low_f16(cb_lo));
                        float32x4_t cb_lo_hi = vcvt_f32_f16(vget_high_f16(cb_lo));
                        float32x4_t cb_hi_lo = vcvt_f32_f16(vget_low_f16(cb_hi));
                        float32x4_t cb_hi_hi = vcvt_f32_f16(vget_high_f16(cb_hi));
                        acc00 = vfmaq_f32(acc00, cb_lo_lo, vcvt_f32_f16(vget_low_f16(ar0_lo)));
                        acc01 = vfmaq_f32(acc01, cb_lo_hi, vcvt_f32_f16(vget_high_f16(ar0_lo)));
                        acc02 = vfmaq_f32(acc02, cb_hi_lo, vcvt_f32_f16(vget_low_f16(ar0_hi)));
                        acc03 = vfmaq_f32(acc03, cb_hi_hi, vcvt_f32_f16(vget_high_f16(ar0_hi)));
                        acc10 = vfmaq_f32(acc10, cb_lo_lo, vcvt_f32_f16(vget_low_f16(ar1_lo)));
                        acc11 = vfmaq_f32(acc11, cb_lo_hi, vcvt_f32_f16(vget_high_f16(ar1_lo)));
                        acc12 = vfmaq_f32(acc12, cb_hi_lo, vcvt_f32_f16(vget_low_f16(ar1_hi)));
                        acc13 = vfmaq_f32(acc13, cb_hi_hi, vcvt_f32_f16(vget_high_f16(ar1_hi)));
                    }
                    float acc0 = vaddvq_f32(vaddq_f32(vaddq_f32(acc00, acc01),
                                                       vaddq_f32(acc02, acc03)));
                    float acc1 = vaddvq_f32(vaddq_f32(vaddq_f32(acc10, acc11),
                                                       vaddq_f32(acc12, acc13)));
                    update_best(slot, 0, static_cast<float>(static_cast<__fp16>(acc0 * norm_n)), static_cast<uint32_t>(n));
                    update_best(slot, 1, static_cast<float>(static_cast<__fp16>(acc1 * norm_n)), static_cast<uint32_t>(n));
                }
                return;
            }
            if (M == 3) {
                const __fp16* arf0 = a_rot_f16;
                const __fp16* arf1 = a_rot_f16 + K;
                const __fp16* arf2 = a_rot_f16 + static_cast<size_t>(2) * K;
                for (size_t n = n_start; n < n_end; ++n) {
                    const uint8_t* packed = W->packed_indices + n * pgb;
                    float norm_n = static_cast<float>(W->norms[n]);
                    float32x4_t acc00 = vdupq_n_f32(0.f);
                    float32x4_t acc01 = vdupq_n_f32(0.f);
                    float32x4_t acc02 = vdupq_n_f32(0.f);
                    float32x4_t acc03 = vdupq_n_f32(0.f);
                    float32x4_t acc10 = vdupq_n_f32(0.f);
                    float32x4_t acc11 = vdupq_n_f32(0.f);
                    float32x4_t acc12 = vdupq_n_f32(0.f);
                    float32x4_t acc13 = vdupq_n_f32(0.f);
                    float32x4_t acc20;
                    float32x4_t acc21;
                    float32x4_t acc22;
                    float32x4_t acc23;
                    if (M == 3) {
                        acc20 = vdupq_n_f32(0.f);
                        acc21 = vdupq_n_f32(0.f);
                        acc22 = vdupq_n_f32(0.f);
                        acc23 = vdupq_n_f32(0.f);
                    }

                    for (uint32_t i = 0; i < K; i += 16) {
                        uint8x8_t raw8 = vld1_u8(packed + i / 2);
                        uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                        uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                        uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                        uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);

                        uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                        uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);

                        uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                        float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                        float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);

                        float16x8_t ar0_lo = vld1q_f16(arf0 + i);
                        float16x8_t ar0_hi = vld1q_f16(arf0 + i + 8);
                        float16x8_t ar1_lo = vld1q_f16(arf1 + i);
                        float16x8_t ar1_hi = vld1q_f16(arf1 + i + 8);
                        float16x8_t ar2_lo;
                        float16x8_t ar2_hi;
                        if (M == 3) {
                            ar2_lo = vld1q_f16(arf2 + i);
                            ar2_hi = vld1q_f16(arf2 + i + 8);
                        }

                        float32x4_t cb_lo_lo = vcvt_f32_f16(vget_low_f16(cb_lo));
                        float32x4_t cb_lo_hi = vcvt_f32_f16(vget_high_f16(cb_lo));
                        float32x4_t cb_hi_lo = vcvt_f32_f16(vget_low_f16(cb_hi));
                        float32x4_t cb_hi_hi = vcvt_f32_f16(vget_high_f16(cb_hi));

                        acc00 = vfmaq_f32(acc00, cb_lo_lo, vcvt_f32_f16(vget_low_f16(ar0_lo)));
                        acc01 = vfmaq_f32(acc01, cb_lo_hi, vcvt_f32_f16(vget_high_f16(ar0_lo)));
                        acc02 = vfmaq_f32(acc02, cb_hi_lo, vcvt_f32_f16(vget_low_f16(ar0_hi)));
                        acc03 = vfmaq_f32(acc03, cb_hi_hi, vcvt_f32_f16(vget_high_f16(ar0_hi)));
                        acc10 = vfmaq_f32(acc10, cb_lo_lo, vcvt_f32_f16(vget_low_f16(ar1_lo)));
                        acc11 = vfmaq_f32(acc11, cb_lo_hi, vcvt_f32_f16(vget_high_f16(ar1_lo)));
                        acc12 = vfmaq_f32(acc12, cb_hi_lo, vcvt_f32_f16(vget_low_f16(ar1_hi)));
                        acc13 = vfmaq_f32(acc13, cb_hi_hi, vcvt_f32_f16(vget_high_f16(ar1_hi)));
                        if (M == 3) {
                            acc20 = vfmaq_f32(acc20, cb_lo_lo, vcvt_f32_f16(vget_low_f16(ar2_lo)));
                            acc21 = vfmaq_f32(acc21, cb_lo_hi, vcvt_f32_f16(vget_high_f16(ar2_lo)));
                            acc22 = vfmaq_f32(acc22, cb_hi_lo, vcvt_f32_f16(vget_low_f16(ar2_hi)));
                            acc23 = vfmaq_f32(acc23, cb_hi_hi, vcvt_f32_f16(vget_high_f16(ar2_hi)));
                        }
                    }

                    float acc0 = vaddvq_f32(vaddq_f32(vaddq_f32(acc00, acc01),
                                                       vaddq_f32(acc02, acc03)));
                    float acc1 = vaddvq_f32(vaddq_f32(vaddq_f32(acc10, acc11),
                                                       vaddq_f32(acc12, acc13)));
                    update_best(slot, 0, static_cast<float>(static_cast<__fp16>(acc0 * norm_n)), static_cast<uint32_t>(n));
                    update_best(slot, 1, static_cast<float>(static_cast<__fp16>(acc1 * norm_n)), static_cast<uint32_t>(n));
                    if (M == 3) {
                        float acc2 = vaddvq_f32(vaddq_f32(vaddq_f32(acc20, acc21),
                                                           vaddq_f32(acc22, acc23)));
                        update_best(slot, 2, static_cast<float>(static_cast<__fp16>(acc2 * norm_n)), static_cast<uint32_t>(n));
                    }
                }
                return;
            }
            for (size_t n = n_start; n < n_end; ++n) {
                const uint8_t* packed = W->packed_indices + n * pgb;
                float norm_n = static_cast<float>(W->norms[n]);
                for (uint32_t m = 0; m < M; ++m) {
                    const __fp16* arf = a_rot_f16 + static_cast<size_t>(m) * K;
                    float32x4_t acc0 = vdupq_n_f32(0.f);
                    float32x4_t acc1 = vdupq_n_f32(0.f);
                    float32x4_t acc2 = vdupq_n_f32(0.f);
                    float32x4_t acc3 = vdupq_n_f32(0.f);

                    for (uint32_t i = 0; i < K; i += 16) {
                        uint8x8_t raw8 = vld1_u8(packed + i / 2);
                        uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                        uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                        uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                        uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);

                        uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                        uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);

                        uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                        float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                        float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);

                        float16x8_t ar_lo = vld1q_f16(arf + i);
                        float16x8_t ar_hi = vld1q_f16(arf + i + 8);

                        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)),
                                               vcvt_f32_f16(vget_low_f16(ar_lo)));
                        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)),
                                               vcvt_f32_f16(vget_high_f16(ar_lo)));
                        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)),
                                               vcvt_f32_f16(vget_low_f16(ar_hi)));
                        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)),
                                               vcvt_f32_f16(vget_high_f16(ar_hi)));
                    }

                    float acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                                      vaddq_f32(acc2, acc3)));
                    float value = static_cast<float>(static_cast<__fp16>(acc * norm_n));
                    update_best(slot, m, value, static_cast<uint32_t>(n));
                }
            }
            return;
        }

        for (size_t n = n_start; n < n_end; ++n) {
            const uint8_t* packed = W->packed_indices + n * pgb;
            float norm_n = static_cast<float>(W->norms[n]);
            for (uint32_t m = 0; m < M; ++m) {
                const float* ar = a_rot + static_cast<size_t>(m) * K;
                float acc = 0.0f;
                for (uint32_t i = 0; i < K; ++i) {
                    uint8_t idx = static_cast<uint8_t>(tq_extract_idx(packed, i, bits));
                    acc += cb_f32[idx] * ar[i];
                }
                float value = static_cast<float>(static_cast<__fp16>(acc * norm_n));
                update_best(slot, m, value, static_cast<uint32_t>(n));
            }
        }
    };

    if (num_threads == 1) {
        process_range(0, N);
    } else {
        pool.enqueue_n_threads(N, num_threads, process_range);
        pool.wait_all();
    }

    for (uint32_t m = 0; m < M; ++m) {
        float best_value = -std::numeric_limits<float>::infinity();
        uint32_t best_index = 0;
        for (size_t slot = 0; slot < num_threads; ++slot) {
            float value = best_values[slot * M + m];
            uint32_t idx = best_indices[slot * M + m];
            if (value > best_value || (value == best_value && idx < best_index)) {
                best_value = value;
                best_index = idx;
            }
        }
        indices[m] = best_index;
    }

    return true;
}

bool cactus_quant_orthogonal_argmax_candidates(
    const CactusQuantMatrix* W,
    const __fp16* A,
    const uint32_t* candidates,
    uint32_t candidate_count,
    uint32_t* index) {
    if (!W || !A || !candidates || !index || candidate_count == 0) return false;
    if (!W->rotation || !W->packed_indices || !W->codebook || !W->norms) return false;

    const uint32_t K    = W->K;
    const uint32_t N    = W->N;
    const uint32_t bits = W->bits;
    const uint32_t pgb  = cactus_quant_packed_group_bytes(bits, K);
    if (K == 0 || N == 0 || pgb == 0 || bits == 0 || bits > 4) return false;

    const uint32_t n_cb = 1u << bits;
    float cb_f32[16] = {};
    for (uint32_t c = 0; c < n_cb; ++c)
        cb_f32[c] = static_cast<float>(W->codebook[c]);

    static thread_local std::vector<float> candidate_a_rot;
    if (candidate_a_rot.size() < K) candidate_a_rot.resize(K);
    float* a_rot = candidate_a_rot.data();
    cactus_quant_orthogonal_rotate_input(W, A, 1, a_rot);

    static thread_local std::vector<__fp16> candidate_a_rot_f16;
    __fp16* a_rot_f16 = nullptr;
    if (bits == 4 && (K % 16) == 0) {
        if (candidate_a_rot_f16.size() < K) candidate_a_rot_f16.resize(K);
        a_rot_f16 = candidate_a_rot_f16.data();
        cactus_f32_to_f16(a_rot, a_rot_f16, K);
    }

    uint8_t cb_lo_arr[16] = {}, cb_hi_arr[16] = {};
    if (bits == 4 && (K % 16) == 0) {
        cactus_tq4_codebook_byte_tables(cb_f32, cb_lo_arr, cb_hi_arr);
    }
    uint8x16_t cb_lo_tbl = vld1q_u8(cb_lo_arr);
    uint8x16_t cb_hi_tbl = vld1q_u8(cb_hi_arr);

    uint32_t best_index = candidates[0];
    float best_value = -std::numeric_limits<float>::infinity();

    for (uint32_t c = 0; c < candidate_count; ++c) {
        uint32_t n = candidates[c];
        if (n >= N) continue;
        const uint8_t* packed = W->packed_indices + static_cast<size_t>(n) * pgb;
        float norm_n = static_cast<float>(W->norms[n]);
        float acc = 0.0f;

        if (bits == 4 && (K % 16) == 0) {
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);

            for (uint32_t i = 0; i < K; i += 16) {
                uint8x8_t raw8 = vld1_u8(packed + i / 2);
                uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);

                uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);
                uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);

                float16x8_t ar_lo = vld1q_f16(a_rot_f16 + i);
                float16x8_t ar_hi = vld1q_f16(a_rot_f16 + i + 8);

                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)),
                                       vcvt_f32_f16(vget_low_f16(ar_lo)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)),
                                       vcvt_f32_f16(vget_high_f16(ar_lo)));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)),
                                       vcvt_f32_f16(vget_low_f16(ar_hi)));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)),
                                       vcvt_f32_f16(vget_high_f16(ar_hi)));
            }

            acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                        vaddq_f32(acc2, acc3)));
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                uint8_t idx = static_cast<uint8_t>(tq_extract_idx(packed, i, bits));
                acc += cb_f32[idx] * a_rot[i];
            }
        }

        float value = static_cast<float>(static_cast<__fp16>(acc * norm_n));
        if (value > best_value) {
            best_value = value;
            best_index = n;
        }
    }

    *index = best_index;
    return std::isfinite(best_value);
}

bool cactus_quant_orthogonal_top2_candidates(
    const CactusQuantMatrix* W,
    const __fp16* A,
    const uint32_t* candidates,
    uint32_t candidate_count,
    uint32_t* first,
    uint32_t* second) {
    if (!W || !A || !candidates || !first || !second || candidate_count == 0) return false;
    if (!W->rotation || !W->packed_indices || !W->codebook || !W->norms) return false;

    const uint32_t K    = W->K;
    const uint32_t N    = W->N;
    const uint32_t bits = W->bits;
    const uint32_t pgb  = cactus_quant_packed_group_bytes(bits, K);
    if (K == 0 || N == 0 || pgb == 0 || bits == 0 || bits > 4) return false;

    const uint32_t n_cb = 1u << bits;
    float cb_f32[16] = {};
    for (uint32_t c = 0; c < n_cb; ++c)
        cb_f32[c] = static_cast<float>(W->codebook[c]);

    static thread_local std::vector<float> top2_a_rot;
    if (top2_a_rot.size() < K) top2_a_rot.resize(K);
    float* a_rot = top2_a_rot.data();
    cactus_quant_orthogonal_rotate_input(W, A, 1, a_rot);

    static thread_local std::vector<__fp16> top2_a_rot_f16;
    __fp16* a_rot_f16 = nullptr;
    if (bits == 4 && (K % 16) == 0) {
        if (top2_a_rot_f16.size() < K) top2_a_rot_f16.resize(K);
        a_rot_f16 = top2_a_rot_f16.data();
        cactus_f32_to_f16(a_rot, a_rot_f16, K);
    }

    uint8_t cb_lo_arr[16] = {}, cb_hi_arr[16] = {};
    if (bits == 4 && (K % 16) == 0) {
        cactus_tq4_codebook_byte_tables(cb_f32, cb_lo_arr, cb_hi_arr);
    }
    uint8x16_t cb_lo_tbl = vld1q_u8(cb_lo_arr);
    uint8x16_t cb_hi_tbl = vld1q_u8(cb_hi_arr);

    uint32_t best_index = candidates[0];
    uint32_t second_index = candidates[0];
    float best_value = -std::numeric_limits<float>::infinity();
    float second_value = -std::numeric_limits<float>::infinity();

    for (uint32_t c = 0; c < candidate_count; ++c) {
        uint32_t n = candidates[c];
        if (n >= N) continue;
        const uint8_t* packed = W->packed_indices + static_cast<size_t>(n) * pgb;
        float norm_n = static_cast<float>(W->norms[n]);
        float acc = 0.0f;

        if (bits == 4 && (K % 16) == 0) {
            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f);
            float32x4_t acc3 = vdupq_n_f32(0.f);

            for (uint32_t i = 0; i < K; i += 16) {
                uint8x8_t raw8 = vld1_u8(packed + i / 2);
                uint8x8_t lo_nibs = vand_u8(raw8, vdup_n_u8(0x0F));
                uint8x8_t hi_nibs = vshr_n_u8(raw8, 4);
                uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
                uint8x16_t indices16 = vcombine_u8(zipped.val[0], zipped.val[1]);

                uint8x16_t lo_bytes = vqtbl1q_u8(cb_lo_tbl, indices16);
                uint8x16_t hi_bytes = vqtbl1q_u8(cb_hi_tbl, indices16);
                uint8x16x2_t fp16_bytes = vzipq_u8(lo_bytes, hi_bytes);
                float16x8_t cb_lo = vreinterpretq_f16_u8(fp16_bytes.val[0]);
                float16x8_t cb_hi = vreinterpretq_f16_u8(fp16_bytes.val[1]);

                float16x8_t ar_lo = vld1q_f16(a_rot_f16 + i);
                float16x8_t ar_hi = vld1q_f16(a_rot_f16 + i + 8);

                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(cb_lo)),
                                       vcvt_f32_f16(vget_low_f16(ar_lo)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(cb_lo)),
                                       vcvt_f32_f16(vget_high_f16(ar_lo)));
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(cb_hi)),
                                       vcvt_f32_f16(vget_low_f16(ar_hi)));
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(cb_hi)),
                                       vcvt_f32_f16(vget_high_f16(ar_hi)));
            }

            acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                        vaddq_f32(acc2, acc3)));
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                uint8_t idx = static_cast<uint8_t>(tq_extract_idx(packed, i, bits));
                acc += cb_f32[idx] * a_rot[i];
            }
        }

        float value = static_cast<float>(static_cast<__fp16>(acc * norm_n));
        if (value > best_value) {
            if (n != best_index) {
                second_value = best_value;
                second_index = best_index;
            }
            best_value = value;
            best_index = n;
        } else if (n != best_index && value > second_value) {
            second_value = value;
            second_index = n;
        }
    }

    *first = best_index;
    *second = std::isfinite(second_value) ? second_index : best_index;
    return std::isfinite(best_value);
}
