#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>

template<typename FinalizeFn>
static void axis_reduce_f32_impl(const __fp16* input, __fp16* output,
                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                 FinalizeFn finalize) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            constexpr size_t W = SIMD_F16_WIDTH;
            const size_t vec_axis = simd_align(axis_size);

            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (size_t a = 0; a < vec_axis; a += W) {
                __fp16 values[W];
                for (size_t j = 0; j < W; j++) {
                    values[j] = input[outer * axis_size * inner_size + (a + j) * inner_size + inner];
                }
                float32x4_t lo, hi;
                f16x8_split_f32(vld1q_f16(values), lo, hi);
                sum_lo = vaddq_f32(sum_lo, lo);
                sum_hi = vaddq_f32(sum_hi, hi);
            }

            float total = vaddvq_f32(vaddq_f32(sum_lo, sum_hi));

            for (size_t a = vec_axis; a < axis_size; a++) {
                total += static_cast<float>(input[outer * axis_size * inner_size + a * inner_size + inner]);
            }

            output[outer * inner_size + inner] = static_cast<__fp16>(finalize(total, axis_size));
        });
}

template<typename SimdReduceOp, typename ScalarReduceOp>
static void axis_reduce_f16_impl(const __fp16* input, __fp16* output,
                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                 float16x8_t init_vec, __fp16 init_scalar,
                                 SimdReduceOp simd_reduce, ScalarReduceOp scalar_reduce) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            constexpr size_t W = SIMD_F16_WIDTH;
            const size_t vec_axis = simd_align(axis_size);
            float16x8_t acc = init_vec;

            for (size_t a = 0; a < vec_axis; a += W) {
                __fp16 values[W];
                for (size_t j = 0; j < W; j++) {
                    values[j] = input[outer * axis_size * inner_size + (a + j) * inner_size + inner];
                }
                acc = simd_reduce(acc, vld1q_f16(values));
            }

            __fp16 result = init_scalar;
            __fp16 arr[8];
            vst1q_f16(arr, acc);
            for (int j = 0; j < 8; j++) {
                result = scalar_reduce(result, arr[j]);
            }

            for (size_t a = vec_axis; a < axis_size; a++) {
                result = scalar_reduce(result, input[outer * axis_size * inner_size + a * inner_size + inner]);
            }

            output[outer * inner_size + inner] = result;
        });
}

double cactus_sum_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start, size_t end) -> double {
            const size_t vec_end = start + simd_align(end - start);
            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (size_t i = start; i < vec_end; i += SIMD_F16_WIDTH) {
                float32x4_t lo, hi;
                f16x8_split_f32(vld1q_f16(&data[i]), lo, hi);
                sum_lo = vaddq_f32(sum_lo, lo);
                sum_hi = vaddq_f32(sum_hi, hi);
            }

            double s = static_cast<double>(vaddvq_f32(vaddq_f32(sum_lo, sum_hi)));
            for (size_t i = vec_end; i < end; ++i) s += static_cast<double>(data[i]);
            return s;
        },
        0.0, [](double a, double b) { return a + b; }
    );
}

void cactus_sum_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    axis_reduce_f32_impl(input, output, outer_size, axis_size, inner_size,
        [](float total, size_t) { return total; });
}

double cactus_mean_all_f16(const __fp16* data, size_t num_elements) {
    return cactus_sum_all_f16(data, num_elements) / static_cast<double>(num_elements);
}

void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    axis_reduce_f32_impl(input, output, outer_size, axis_size, inner_size,
        [](float total, size_t axis_size) { return total / static_cast<float>(axis_size); });
}

struct VarianceState {
    double sum;
    double sum_sq;
    VarianceState() : sum(0.0), sum_sq(0.0) {}
    VarianceState(double s, double sq) : sum(s), sum_sq(sq) {}
};

double cactus_variance_all_f16(const __fp16* data, size_t num_elements) {
    VarianceState result = CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start, size_t end) -> VarianceState {
            const size_t vec_end = start + simd_align(end - start);

            float32x4_t sum_lo = vdupq_n_f32(0.0f), sum_hi = vdupq_n_f32(0.0f);
            float32x4_t sq_lo = vdupq_n_f32(0.0f), sq_hi = vdupq_n_f32(0.0f);

            for (size_t i = start; i < vec_end; i += SIMD_F16_WIDTH) {
                float32x4_t lo, hi;
                f16x8_split_f32(vld1q_f16(&data[i]), lo, hi);
                sum_lo = vaddq_f32(sum_lo, lo);
                sum_hi = vaddq_f32(sum_hi, hi);
                sq_lo = vfmaq_f32(sq_lo, lo, lo);
                sq_hi = vfmaq_f32(sq_hi, hi, hi);
            }

            double sum = static_cast<double>(vaddvq_f32(vaddq_f32(sum_lo, sum_hi)));
            double sum_sq = static_cast<double>(vaddvq_f32(vaddq_f32(sq_lo, sq_hi)));

            for (size_t i = vec_end; i < end; ++i) {
                double x = static_cast<double>(data[i]);
                sum += x;
                sum_sq += x * x;
            }
            return VarianceState(sum, sum_sq);
        },
        VarianceState(),
        [](const VarianceState& a, const VarianceState& b) {
            return VarianceState(a.sum + b.sum, a.sum_sq + b.sum_sq);
        }
    );

    double mean = result.sum / static_cast<double>(num_elements);
    double mean_sq = result.sum_sq / static_cast<double>(num_elements);
    return mean_sq - mean * mean;
}

void cactus_variance_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (size_t a = 0; a < axis_size; a++) {
                float x = static_cast<float>(input[outer * axis_size * inner_size + a * inner_size + inner]);
                sum += x;
                sum_sq += x * x;
            }
            float mean = sum / static_cast<float>(axis_size);
            float mean_sq = sum_sq / static_cast<float>(axis_size);
            output[outer * inner_size + inner] = static_cast<__fp16>(mean_sq - mean * mean);
        });
}

__fp16 cactus_min_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start, size_t end) -> __fp16 {
            const size_t vec_end = start + simd_align(end - start);
            float16x8_t acc = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            for (size_t i = start; i < vec_end; i += SIMD_F16_WIDTH) {
                acc = vminq_f16(acc, vld1q_f16(&data[i]));
            }

            __fp16 result = static_cast<__fp16>(65504.0f);
            __fp16 arr[8];
            vst1q_f16(arr, acc);
            for (int j = 0; j < 8; j++) result = std::min(result, arr[j]);
            for (size_t i = vec_end; i < end; ++i) result = std::min(result, data[i]);
            return result;
        },
        static_cast<__fp16>(65504.0f),
        [](__fp16 a, __fp16 b) { return std::min(a, b); }
    );
}

void cactus_min_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    axis_reduce_f16_impl(input, output, outer_size, axis_size, inner_size,
        vdupq_n_f16(static_cast<__fp16>(65504.0f)), static_cast<__fp16>(65504.0f),
        [](float16x8_t a, float16x8_t b) { return vminq_f16(a, b); },
        [](__fp16 a, __fp16 b) { return std::min(a, b); });
}

__fp16 cactus_max_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start, size_t end) -> __fp16 {
            const size_t vec_end = start + simd_align(end - start);
            float16x8_t acc = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            for (size_t i = start; i < vec_end; i += SIMD_F16_WIDTH) {
                acc = vmaxq_f16(acc, vld1q_f16(&data[i]));
            }

            __fp16 result = static_cast<__fp16>(-65504.0f);
            __fp16 arr[8];
            vst1q_f16(arr, acc);
            for (int j = 0; j < 8; j++) result = std::max(result, arr[j]);
            for (size_t i = vec_end; i < end; ++i) result = std::max(result, data[i]);
            return result;
        },
        static_cast<__fp16>(-65504.0f),
        [](__fp16 a, __fp16 b) { return std::max(a, b); }
    );
}

void cactus_max_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    axis_reduce_f16_impl(input, output, outer_size, axis_size, inner_size,
        vdupq_n_f16(static_cast<__fp16>(-65504.0f)), static_cast<__fp16>(-65504.0f),
        [](float16x8_t a, float16x8_t b) { return vmaxq_f16(a, b); },
        [](__fp16 a, __fp16 b) { return std::max(a, b); });
}
