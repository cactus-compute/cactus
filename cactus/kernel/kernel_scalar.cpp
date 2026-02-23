#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    const __fp16 scalar_f16 = static_cast<__fp16>(scalar_value);
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;

    switch (op_type) {
        case ScalarOpType::ADD: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 4;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 256], 0, 0);
                            float16x8_t v0 = vaddq_f16(vld1q_f16(&input[i]), scalar_vec);
                            float16x8_t v1 = vaddq_f16(vld1q_f16(&input[i + 8]), scalar_vec);
                            float16x8_t v2 = vaddq_f16(vld1q_f16(&input[i + 16]), scalar_vec);
                            float16x8_t v3 = vaddq_f16(vld1q_f16(&input[i + 24]), scalar_vec);
                            stream_store_f16x8(&output[i], v0);
                            stream_store_f16x8(&output[i + 8], v1);
                            stream_store_f16x8(&output[i + 16], v2);
                            stream_store_f16x8(&output[i + 24], v3);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            stream_store_f16x8(&output[i], vaddq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            vst1q_f16(&output[i], vaddq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] + scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::SUBTRACT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 4;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 256], 0, 0);
                            float16x8_t v0 = vsubq_f16(vld1q_f16(&input[i]), scalar_vec);
                            float16x8_t v1 = vsubq_f16(vld1q_f16(&input[i + 8]), scalar_vec);
                            float16x8_t v2 = vsubq_f16(vld1q_f16(&input[i + 16]), scalar_vec);
                            float16x8_t v3 = vsubq_f16(vld1q_f16(&input[i + 24]), scalar_vec);
                            stream_store_f16x8(&output[i], v0);
                            stream_store_f16x8(&output[i + 8], v1);
                            stream_store_f16x8(&output[i + 16], v2);
                            stream_store_f16x8(&output[i + 24], v3);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            stream_store_f16x8(&output[i], vsubq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            vst1q_f16(&output[i], vsubq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] - scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::MULTIPLY: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 4;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 256], 0, 0);
                            float16x8_t v0 = vmulq_f16(vld1q_f16(&input[i]), scalar_vec);
                            float16x8_t v1 = vmulq_f16(vld1q_f16(&input[i + 8]), scalar_vec);
                            float16x8_t v2 = vmulq_f16(vld1q_f16(&input[i + 16]), scalar_vec);
                            float16x8_t v3 = vmulq_f16(vld1q_f16(&input[i + 24]), scalar_vec);
                            stream_store_f16x8(&output[i], v0);
                            stream_store_f16x8(&output[i + 8], v1);
                            stream_store_f16x8(&output[i + 16], v2);
                            stream_store_f16x8(&output[i + 24], v3);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            stream_store_f16x8(&output[i], vmulq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            vst1q_f16(&output[i], vmulq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] * scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::DIVIDE: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 4;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 256], 0, 0);
                            float16x8_t v0 = vdivq_f16(vld1q_f16(&input[i]), scalar_vec);
                            float16x8_t v1 = vdivq_f16(vld1q_f16(&input[i + 8]), scalar_vec);
                            float16x8_t v2 = vdivq_f16(vld1q_f16(&input[i + 16]), scalar_vec);
                            float16x8_t v3 = vdivq_f16(vld1q_f16(&input[i + 24]), scalar_vec);
                            stream_store_f16x8(&output[i], v0);
                            stream_store_f16x8(&output[i + 8], v1);
                            stream_store_f16x8(&output[i + 16], v2);
                            stream_store_f16x8(&output[i + 24], v3);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            stream_store_f16x8(&output[i], vdivq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            vst1q_f16(&output[i], vdivq_f16(vld1q_f16(&input[i]), scalar_vec));
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] / scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::EXP: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 2;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 128], 0, 0);
                            float16x8_t in0 = vld1q_f16(&input[i]);
                            float16x8_t in1 = vld1q_f16(&input[i + 8]);
                            float16x8_t r0 = vcombine_f16(
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_low_f16(in0)))),
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_high_f16(in0)))));
                            float16x8_t r1 = vcombine_f16(
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_low_f16(in1)))),
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_high_f16(in1)))));
                            stream_store_f16x8(&output[i], r0);
                            stream_store_f16x8(&output[i + 8], r1);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            float16x8_t in_vec = vld1q_f16(&input[i]);
                            float16x8_t result = vcombine_f16(
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_low_f16(in_vec)))),
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_high_f16(in_vec)))));
                            stream_store_f16x8(&output[i], result);
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            float16x8_t in_vec = vld1q_f16(&input[i]);
                            float16x8_t result = vcombine_f16(
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_low_f16(in_vec)))),
                                vcvt_f16_f32(fast_exp_f32x4(vcvt_f32_f16(vget_high_f16(in_vec)))));
                            vst1q_f16(&output[i], result);
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = static_cast<__fp16>(std::exp(static_cast<float>(input[i])));
                    }
                });
            break;
        }

        case ScalarOpType::SQRT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    constexpr size_t UNROLL = 2;
                    const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    if (use_streaming) {
                        for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                            __builtin_prefetch(&input[i + 128], 0, 0);
                            float16x8_t in0 = vld1q_f16(&input[i]);
                            float16x8_t in1 = vld1q_f16(&input[i + 8]);
                            float16x8_t r0 = vcombine_f16(
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_low_f16(in0)))),
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_high_f16(in0)))));
                            float16x8_t r1 = vcombine_f16(
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_low_f16(in1)))),
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_high_f16(in1)))));
                            stream_store_f16x8(&output[i], r0);
                            stream_store_f16x8(&output[i + 8], r1);
                        }
                        for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                            float16x8_t in_vec = vld1q_f16(&input[i]);
                            float16x8_t result = vcombine_f16(
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_low_f16(in_vec)))),
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_high_f16(in_vec)))));
                            stream_store_f16x8(&output[i], result);
                        }
                    } else {
                        for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                            float16x8_t in_vec = vld1q_f16(&input[i]);
                            float16x8_t result = vcombine_f16(
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_low_f16(in_vec)))),
                                vcvt_f16_f32(vsqrtq_f32(vcvt_f32_f16(vget_high_f16(in_vec)))));
                            vst1q_f16(&output[i], result);
                        }
                    }
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = static_cast<__fp16>(std::sqrt(static_cast<float>(input[i])));
                    }
                });
            break;
        }

        case ScalarOpType::COS:
        case ScalarOpType::SIN: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        float val = static_cast<float>(input[i]);
                        float result;
                        switch (op_type) {
                            case ScalarOpType::COS: result = std::cos(val); break;
                            case ScalarOpType::SIN: result = std::sin(val); break;
                            default: result = val; break;
                        }
                        output[i] = static_cast<__fp16>(result);
                    }
                });
            break;
        }

        case ScalarOpType::LOG: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        float val = static_cast<float>(input[i]);
                        output[i] = static_cast<__fp16>(std::log(val));
                    }
                });
            break;
        }
    }
}
