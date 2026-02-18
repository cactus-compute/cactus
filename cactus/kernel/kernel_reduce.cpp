#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

double cactus_sum_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(input_vec)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(input_vec)));
            }

            double thread_sum = static_cast<double>(vaddvq_f32(vaddq_f32(sum_lo, sum_hi)));

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<double>(data[i]);
            }

            return thread_sum;
        },
        0.0,
        [](double a, double b) { return a + b; }
    );
}

void cactus_sum_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(input_vec)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(input_vec)));
            }

            float total_sum_f32 = vaddvq_f32(vaddq_f32(sum_lo, sum_hi));

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum_f32 += static_cast<float>(input[idx]);
            }

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = static_cast<__fp16>(total_sum_f32);
        });
}

double cactus_mean_all_f16(const __fp16* data, size_t num_elements) {
    double sum = cactus_sum_all_f16(data, num_elements);
    return sum / static_cast<double>(num_elements);
}

void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(input_vec)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(input_vec)));
            }

            float total_sum_f32 = vaddvq_f32(vaddq_f32(sum_lo, sum_hi));

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum_f32 += static_cast<float>(input[idx]);
            }

            const float mean_f32 = total_sum_f32 / static_cast<float>(axis_size);

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = static_cast<__fp16>(mean_f32);
        });
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
        [&](size_t start_idx, size_t end_idx) -> VarianceState {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float32x4_t sum_vec_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_vec_hi = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_vec_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_vec_hi = vdupq_n_f32(0.0f);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x = vld1q_f16(&data[i]);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x));

                sum_vec_lo = vaddq_f32(sum_vec_lo, x_lo);
                sum_vec_hi = vaddq_f32(sum_vec_hi, x_hi);
                sum_sq_vec_lo = vfmaq_f32(sum_sq_vec_lo, x_lo, x_lo);
                sum_sq_vec_hi = vfmaq_f32(sum_sq_vec_hi, x_hi, x_hi);
            }

            double sum = static_cast<double>(vaddvq_f32(vaddq_f32(sum_vec_lo, sum_vec_hi)));
            double sum_sq = static_cast<double>(vaddvq_f32(vaddq_f32(sum_sq_vec_lo, sum_sq_vec_hi)));

            for (size_t i = vectorized_end; i < end_idx; ++i) {
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
            float sum = 0.0f;
            float sum_sq = 0.0f;

            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                float x = static_cast<float>(input[idx]);
                sum += x;
                sum_sq += x * x;
            }

            float mean = sum / static_cast<float>(axis_size);
            float mean_sq = sum_sq / static_cast<float>(axis_size);
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = static_cast<__fp16>(mean_sq - mean * mean);
        });
}

__fp16 cactus_min_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> __fp16 {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float16x8_t min_vec = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                min_vec = vminq_f16(min_vec, input_vec);
            }

            __fp16 thread_min = static_cast<__fp16>(65504.0f);
            __fp16 min_array[8];
            vst1q_f16(min_array, min_vec);
            for (int j = 0; j < 8; j++) {
                thread_min = std::min(thread_min, min_array[j]);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_min = std::min(thread_min, data[i]);
            }

            return thread_min;
        },
        static_cast<__fp16>(65504.0f),
        [](__fp16 a, __fp16 b) { return std::min(a, b); }
    );
}

void cactus_min_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t min_vec = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                min_vec = vminq_f16(min_vec, input_vec);
            }

            __fp16 min_val = static_cast<__fp16>(65504.0f);
            __fp16 min_array[8];
            vst1q_f16(min_array, min_vec);
            for (int j = 0; j < 8; j++) {
                min_val = std::min(min_val, min_array[j]);
            }

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                min_val = std::min(min_val, input[idx]);
            }

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = min_val;
        });
}

__fp16 cactus_max_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> __fp16 {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float16x8_t max_vec = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                max_vec = vmaxq_f16(max_vec, input_vec);
            }

            __fp16 thread_max = static_cast<__fp16>(-65504.0f);
            __fp16 max_array[8];
            vst1q_f16(max_array, max_vec);
            for (int j = 0; j < 8; j++) {
                thread_max = std::max(thread_max, max_array[j]);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_max = std::max(thread_max, data[i]);
            }

            return thread_max;
        },
        static_cast<__fp16>(-65504.0f),
        [](__fp16 a, __fp16 b) { return std::max(a, b); }
    );
}

void cactus_max_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t max_vec = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                max_vec = vmaxq_f16(max_vec, input_vec);
            }

            __fp16 max_val = static_cast<__fp16>(-65504.0f);
            __fp16 max_array[8];
            vst1q_f16(max_array, max_vec);
            for (int j = 0; j < 8; j++) {
                max_val = std::max(max_val, max_array[j]);
            }

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                max_val = std::max(max_val, input[idx]);
            }

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = max_val;
        });
}
