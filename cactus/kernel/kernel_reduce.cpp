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

void cactus_sum_axis_f16(const __fp16* input,
                         __fp16* output,
                         size_t outer_size,
                         size_t axis_size,
                         size_t inner_size) {
    CactusThreading::parallel_for_2d(
        outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            if (inner_size == 1) {
                const __fp16* ptr = input + outer * axis_size;

                size_t a = 0;
                for (; a < vectorized_axis; a += SIMD_WIDTH) {
                    float16x8_t v = vld1q_f16(ptr + a);
                    sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(v)));
                    sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(v)));
                }

                float tail_sum = 0.0f;
                for (; a < axis_size; ++a) {
                    tail_sum += static_cast<float>(ptr[a]);
                }

                float total_sum = tail_sum + vaddvq_f32(sum_lo) + vaddvq_f32(sum_hi);

                output[outer] = static_cast<__fp16>(total_sum);
                return;
            }

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                const size_t base = outer * axis_size * inner_size + (a * inner_size) + inner;

                for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                    values[j] = input[base + j * inner_size];
                }

                float16x8_t v = vld1q_f16(values);
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(v)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(v)));
            }

            float total_sum_f32 = vaddvq_f32(sum_lo) + vaddvq_f32(sum_hi);

            for (size_t a = vectorized_axis; a < axis_size; ++a) {
                const size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum_f32 += static_cast<float>(input[idx]);
            }

            const size_t output_idx = outer * inner_size + inner;
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

            if (inner_size == 1) {
                const __fp16* ptr = input + outer * axis_size;
                float tail_sum = 0.0f;
                size_t a = 0;

                for (; a < vectorized_axis; a += SIMD_WIDTH) {
                    float16x8_t v = vld1q_f16(ptr + a);
                    sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(v)));
                    sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(v)));
                }
                for (; a < axis_size; ++a) {
                    tail_sum += static_cast<float>(ptr[a]);
                }
                float total_sum = tail_sum + vaddvq_f32(sum_lo) + vaddvq_f32(sum_hi);

                output[outer] = static_cast<__fp16>(total_sum / static_cast<float>(axis_size));
                return;
            }

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
            constexpr size_t SIMD_WIDTH = 8;
            float sum = 0.0f;
            float sum_sq = 0.0f;

            if (inner_size == 1) {
                const __fp16* ptr = input + outer * axis_size * inner_size + inner;
                float32x4_t sum_lo = vdupq_n_f32(0.0f);
                float32x4_t sum_hi = vdupq_n_f32(0.0f);
                float32x4_t sum_sq_lo = vdupq_n_f32(0.0f);
                float32x4_t sum_sq_hi = vdupq_n_f32(0.0f);
                size_t vec_end = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
                size_t a = 0;

                for (; a < vec_end; a += SIMD_WIDTH) {
                    float16x8_t x = vld1q_f16(ptr + a);
                    float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x));
                    float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x));

                    sum_lo = vaddq_f32(sum_lo, x_lo);
                    sum_hi = vaddq_f32(sum_hi, x_hi);
                    sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
                    sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
                }

                sum = vaddvq_f32(vaddq_f32(sum_lo, sum_hi));
                sum_sq = vaddvq_f32(vaddq_f32(sum_sq_lo, sum_sq_hi));

                for (; a < axis_size; ++a) {
                    float x = static_cast<float>(ptr[a]);
                    sum += x;
                    sum_sq += x * x;
                }

                float mean = sum / static_cast<float>(axis_size);
                float mean_sq = sum_sq / static_cast<float>(axis_size);
                size_t output_idx = outer * inner_size + inner;
                output[output_idx] = static_cast<__fp16>(mean_sq - mean * mean);
                return;
            }

            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_hi = vdupq_n_f32(0.0f);

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t x = vld1q_f16(values);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x));

                sum_lo = vaddq_f32(sum_lo, x_lo);
                sum_hi = vaddq_f32(sum_hi, x_hi);
                sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
                sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
            }

            sum = vaddvq_f32(vaddq_f32(sum_lo, sum_hi));
            sum_sq = vaddvq_f32(vaddq_f32(sum_sq_lo, sum_sq_hi));

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                float x = static_cast<float>(input[idx]);
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
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t min_vec = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            if (inner_size == 1) {
                const __fp16* ptr = input + outer * axis_size;
                float tail_min = static_cast<float>(65504.0f);
                size_t a = 0;

                for (; a < vectorized_axis; a += SIMD_WIDTH) {
                    min_vec = vminq_f16(min_vec, vld1q_f16(ptr + a));
                }
                for (; a < axis_size; ++a) {
                    tail_min = std::min(tail_min, static_cast<float>(ptr[a]));
                }

                __fp16 min_array[8];
                vst1q_f16(min_array, min_vec);
                float total_min = tail_min;
                for (int j = 0; j < 8; j++) {
                    total_min = std::min(total_min, static_cast<float>(min_array[j]));
                }
                output[outer] = static_cast<__fp16>(total_min);
                return;
            }

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
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t max_vec = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            if (inner_size == 1) {
                const __fp16* ptr = input + outer * axis_size;
                float tail_max = static_cast<float>(-65504.0f);
                size_t a = 0;

                for (; a < vectorized_axis; a += SIMD_WIDTH) {
                    max_vec = vmaxq_f16(max_vec, vld1q_f16(ptr + a));
                }
                for (; a < axis_size; ++a) {
                    tail_max = std::max(tail_max, static_cast<float>(ptr[a]));
                }

                __fp16 max_array[8];
                vst1q_f16(max_array, max_vec);
                float total_max = tail_max;
                for (int j = 0; j < 8; j++) {
                    total_max = std::max(total_max, static_cast<float>(max_array[j]));
                }
                output[outer] = static_cast<__fp16>(total_max);
                return;
            }

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
