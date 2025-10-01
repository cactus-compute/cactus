#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

void cactus_scalar_op_int8(const int8_t* input, int8_t* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    switch (op_type) {
        case ScalarOpType::ADD: {
            const int8_t scalar_int8 = clamp_to_int8(scalar_value);
            const size_t vectorized_elements = (num_elements / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
            
            int8x16_t scalar_vec = vdupq_n_s8(scalar_int8);
            
            for (size_t i = 0; i < vectorized_elements; i += NEON_VECTOR_SIZE) {
                int8x16_t input_vec = vld1q_s8(&input[i]);
                
                int16x8_t input_low = vmovl_s8(vget_low_s8(input_vec));
                int16x8_t input_high = vmovl_s8(vget_high_s8(input_vec));
                int16x8_t scalar_low = vmovl_s8(vget_low_s8(scalar_vec));
                int16x8_t scalar_high = vmovl_s8(vget_high_s8(scalar_vec));
                
                int16x8_t result_low = vaddq_s16(input_low, scalar_low);
                int16x8_t result_high = vaddq_s16(input_high, scalar_high);
                
                int8x16_t result_vec = vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
                vst1q_s8(&output[i], result_vec);
            }
            
            for (size_t i = vectorized_elements; i < num_elements; ++i) {
                int32_t sum = static_cast<int32_t>(input[i]) + static_cast<int32_t>(scalar_int8);
                output[i] = clamp_to_int8(sum);
            }
            break;
        }
        
        case ScalarOpType::SUBTRACT: {
            const int8_t scalar_int8 = clamp_to_int8(scalar_value);
            const size_t vectorized_elements = (num_elements / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
            
            int8x16_t scalar_vec = vdupq_n_s8(scalar_int8);
            
            for (size_t i = 0; i < vectorized_elements; i += NEON_VECTOR_SIZE) {
                int8x16_t input_vec = vld1q_s8(&input[i]);
                
                int16x8_t input_low = vmovl_s8(vget_low_s8(input_vec));
                int16x8_t input_high = vmovl_s8(vget_high_s8(input_vec));
                int16x8_t scalar_low = vmovl_s8(vget_low_s8(scalar_vec));
                int16x8_t scalar_high = vmovl_s8(vget_high_s8(scalar_vec));
                
                int16x8_t result_low = vsubq_s16(input_low, scalar_low);
                int16x8_t result_high = vsubq_s16(input_high, scalar_high);
                
                int8x16_t result_vec = vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
                vst1q_s8(&output[i], result_vec);
            }
            
            for (size_t i = vectorized_elements; i < num_elements; ++i) {
                int32_t diff = static_cast<int32_t>(input[i]) - static_cast<int32_t>(scalar_int8);
                output[i] = clamp_to_int8(diff);
            }
            break;
        }
        
        case ScalarOpType::MULTIPLY: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t scalar_f32 = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int8x8_t input_s8 = vld1_s8(&input[i]);
                        int16x8_t input_s16 = vmovl_s8(input_s8);
                        
                        float32x4_t input_low_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));
                        float32x4_t input_high_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16)));
                        
                        float32x4_t result_low_f32 = vmulq_f32(input_low_f32, scalar_f32);
                        float32x4_t result_high_f32 = vmulq_f32(input_high_f32, scalar_f32);
                        
                        int32x4_t result_low_s32 = vcvtq_s32_f32(result_low_f32);
                        int32x4_t result_high_s32 = vcvtq_s32_f32(result_high_f32);
                        
                        int16x8_t result_s16 = vcombine_s16(vqmovn_s32(result_low_s32), vqmovn_s32(result_high_s32));
                        vst1_s8(&output[i], vqmovn_s16(result_s16));
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = input_float * scalar_value;
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
        
        case ScalarOpType::DIVIDE: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t scalar_f32 = vdupq_n_f32(scalar_value);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int8x8_t input_s8 = vld1_s8(&input[i]);
                        int16x8_t input_s16 = vmovl_s8(input_s8);
                        
                        float32x4_t input_low_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));
                        float32x4_t input_high_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16)));
                        
                        float32x4_t result_low_f32 = vdivq_f32(input_low_f32, scalar_f32);
                        float32x4_t result_high_f32 = vdivq_f32(input_high_f32, scalar_f32);
                        
                        int32x4_t result_low_s32 = vcvtq_s32_f32(result_low_f32);
                        int32x4_t result_high_s32 = vcvtq_s32_f32(result_high_f32);
                        
                        int16x8_t result_s16 = vcombine_s16(vqmovn_s32(result_low_s32), vqmovn_s32(result_high_s32));
                        vst1_s8(&output[i], vqmovn_s16(result_s16));
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = input_float / scalar_value;
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
        
        case ScalarOpType::EXP: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int16x8_t input_s16 = vmovl_s8(vld1_s8(&input[i]));
                        float32x4_t input_low_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));
                        
                        const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
                        const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
                        const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
                        const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);

                        float32x4_t x = vmulq_f32(input_low_f32, log2e);
                        int32x4_t xi = vcvtq_s32_f32(x);
                        float32x4_t xf = vsubq_f32(x, vcvtq_f32_s32(xi));
                        
                        float32x4_t p = vmlaq_f32(c2, c3, xf);
                        p = vmlaq_f32(c1, p, xf);
                        p = vmlaq_f32(vdupq_n_f32(1.0f), p, xf);

                        int32x4_t exponent = vaddq_s32(xi, vdupq_n_s32(127));
                        exponent = vshlq_n_s32(exponent, 23);
                        float32x4_t scale = vreinterpretq_f32_s32(exponent);
                        
                        float32x4_t result_f32 = vmulq_f32(p, scale);
                        
                        int32x4_t result_s32 = vcvtq_s32_f32(result_f32);
                        int16x4_t result_s16 = vqmovn_s32(result_s32);
                        vst1_s8(&output[i], vqmovn_s16(vcombine_s16(result_s16, result_s16)));
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = expf(input_float);
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
        
        case ScalarOpType::SQRT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int8x8_t input_s8 = vld1_s8(&input[i]);
                        int16x8_t input_s16 = vmovl_s8(input_s8);
                        
                        float32x4_t input_low_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));
                        float32x4_t input_high_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16)));
                        
                        float32x4_t rsqrt_low = vrsqrteq_f32(input_low_f32);
                        float32x4_t rsqrt_high = vrsqrteq_f32(input_high_f32);
                        
                        rsqrt_low = vmulq_f32(rsqrt_low, vrsqrtsq_f32(vmulq_f32(input_low_f32, rsqrt_low), rsqrt_low));
                        rsqrt_high = vmulq_f32(rsqrt_high, vrsqrtsq_f32(vmulq_f32(input_high_f32, rsqrt_high), rsqrt_high));
                        
                        float32x4_t result_low_f32 = vmulq_f32(input_low_f32, rsqrt_low);
                        float32x4_t result_high_f32 = vmulq_f32(input_high_f32, rsqrt_high);
                        
                        int32x4_t result_low_s32 = vcvtq_s32_f32(result_low_f32);
                        int32x4_t result_high_s32 = vcvtq_s32_f32(result_high_f32);
                        
                        int16x8_t result_s16 = vcombine_s16(vqmovn_s32(result_low_s32), vqmovn_s32(result_high_s32));
                        vst1_s8(&output[i], vqmovn_s16(result_s16));
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = sqrtf(input_float);
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
        
        case ScalarOpType::COS: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    const float32x4_t c1 = vdupq_n_f32(-0.5f);
                    const float32x4_t c2 = vdupq_n_f32(0.04166666666f);
                    const float32x4_t c3 = vdupq_n_f32(-0.00138888888f);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int16x8_t input_s16 = vmovl_s8(vld1_s8(&input[i]));
                        float32x4_t x = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));

                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f)))), vdupq_n_f32(2.0f * 3.14159265358979323846f)));
                        
                        float32x4_t x_abs = vabsq_f32(x);
                        
                        float32x4_t y = vdupq_n_f32(1.0f);
                        float32x4_t x2 = vmulq_f32(x_abs, x_abs);
                        y = vmlaq_f32(y, x2, c1);
                        y = vmlaq_f32(y, vmulq_f32(x2, x2), c2);
                        y = vmlaq_f32(y, vmulq_f32(vmulq_f32(x2, x2), x2), c3);

                        int32x4_t result_s32 = vcvtq_s32_f32(vmulq_f32(y, vdupq_n_f32(127.0f)));
                        int16x4_t result_s16 = vqmovn_s32(result_s32);
                        vst1_s8(&output[i], vqmovn_s16(vcombine_s16(result_s16, result_s16)));
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = cosf(input_float);
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
        
        case ScalarOpType::SIN: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    const float32x4_t c1 = vdupq_n_f32(1.0f);
                    const float32x4_t c3 = vdupq_n_f32(-0.16666666666f);
                    const float32x4_t c5 = vdupq_n_f32(0.00833333333f);
                    const float32x4_t c7 = vdupq_n_f32(-0.00019841269f);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        int16x8_t input_s16 = vmovl_s8(vld1_s8(&input[i]));
                        float32x4_t x = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16)));

                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f)))), vdupq_n_f32(2.0f * 3.14159265358979323846f)));
                        
                        float32x4_t x2 = vmulq_f32(x, x);
                        float32x4_t x3 = vmulq_f32(x2, x);
                        float32x4_t x5 = vmulq_f32(x3, x2);
                        float32x4_t x7 = vmulq_f32(x5, x2);
                        
                        float32x4_t y = vmulq_f32(x, c1);
                        y = vmlaq_f32(y, x3, c3);
                        y = vmlaq_f32(y, x5, c5);
                        y = vmlaq_f32(y, x7, c7);

                        int32x4_t result_s32 = vcvtq_s32_f32(vmulq_f32(y, vdupq_n_f32(127.0f)));
                        int16x4_t result_s16 = vqmovn_s32(result_s32);
                        vst1_s8(&output[i], vqmovn_s16(vcombine_s16(result_s16, result_s16)));
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        float input_float = static_cast<float>(input[i]);
                        float result_float = sinf(input_float);
                        output[i] = clamp_to_int8(result_float);
                    }
                });
            break;
        }
    }
}

void cactus_scalar_op_f32(const float* input, float* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    switch (op_type) {
        case ScalarOpType::ADD: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vaddq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] + scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::SUBTRACT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vsubq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] - scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::MULTIPLY: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vmulq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] * scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::DIVIDE: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vdivq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] / scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::EXP: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_f32 = vld1q_f32(&input[i]);
                        
                        const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
                        const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
                        const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
                        const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);

                        float32x4_t x = vmulq_f32(input_f32, log2e);
                        int32x4_t xi = vcvtq_s32_f32(x);
                        float32x4_t xf = vsubq_f32(x, vcvtq_f32_s32(xi));
                        
                        float32x4_t p = vmlaq_f32(c2, c3, xf);
                        p = vmlaq_f32(c1, p, xf);
                        p = vmlaq_f32(vdupq_n_f32(1.0f), p, xf);

                        int32x4_t exponent = vaddq_s32(xi, vdupq_n_s32(127));
                        exponent = vshlq_n_s32(exponent, 23);
                        float32x4_t scale = vreinterpretq_f32_s32(exponent);
                        
                        float32x4_t result_f32 = vmulq_f32(p, scale);
                        
                        vst1q_f32(&output[i], result_f32);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = std::exp(input[i]);
                    }
                });
            break;
        }
        
        case ScalarOpType::SQRT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_f32 = vld1q_f32(&input[i]);
                        
                        input_f32 = vmaxq_f32(input_f32, vdupq_n_f32(0.0f));
                        
                        uint32x4_t zero_mask = vceqq_f32(input_f32, vdupq_n_f32(0.0f));
                        
                        float32x4_t rsqrt = vrsqrteq_f32(input_f32);
                        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(input_f32, rsqrt), rsqrt));
                        float32x4_t result_f32 = vmulq_f32(input_f32, rsqrt);
                        
                        result_f32 = vbslq_f32(zero_mask, vdupq_n_f32(0.0f), result_f32);
                        
                        vst1q_f32(&output[i], result_f32);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = std::sqrt(std::max(0.0f, input[i]));
                    }
                });
            break;
        }
        
        case ScalarOpType::COS: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t two_pi = vdupq_n_f32(2.0f * 3.14159265358979323846f);
                    const float32x4_t inv_two_pi = vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f));
                    const float32x4_t c0 = vdupq_n_f32(1.0f);
                    const float32x4_t c2 = vdupq_n_f32(-0.5f);
                    const float32x4_t c4 = vdupq_n_f32(0.04166666666f);
                    const float32x4_t c6 = vdupq_n_f32(-0.00138888888f);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t x = vld1q_f32(&input[i]);
                        
                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));
                        
                        float32x4_t x2 = vmulq_f32(x, x);
                        float32x4_t x4 = vmulq_f32(x2, x2);
                        float32x4_t x6 = vmulq_f32(x4, x2);
                        
                        float32x4_t result = c0;
                        result = vmlaq_f32(result, x2, c2);
                        result = vmlaq_f32(result, x4, c4);
                        result = vmlaq_f32(result, x6, c6);
                        
                        vst1q_f32(&output[i], result);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = cosf(input[i]);
                    }
                });
            break;
        }
        
        case ScalarOpType::SIN: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t two_pi = vdupq_n_f32(2.0f * 3.14159265358979323846f);
                    const float32x4_t inv_two_pi = vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f));
                    const float32x4_t c1 = vdupq_n_f32(1.0f);
                    const float32x4_t c3 = vdupq_n_f32(-0.16666666666f);
                    const float32x4_t c5 = vdupq_n_f32(0.00833333333f);
                    const float32x4_t c7 = vdupq_n_f32(-0.00019841269f);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t x = vld1q_f32(&input[i]);
                        
                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));
                        
                        float32x4_t x2 = vmulq_f32(x, x);
                        float32x4_t x3 = vmulq_f32(x2, x);
                        float32x4_t x5 = vmulq_f32(x3, x2);
                        float32x4_t x7 = vmulq_f32(x5, x2);
                        
                        float32x4_t result = vmulq_f32(x, c1);
                        result = vmlaq_f32(result, x3, c3);
                        result = vmlaq_f32(result, x5, c5);
                        result = vmlaq_f32(result, x7, c7);
                        
                        vst1q_f32(&output[i], result);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = sinf(input[i]);
                    }
                });
            break;
        }
    }
}

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    const __fp16 scalar_f16 = static_cast<__fp16>(scalar_value);

    switch (op_type) {
        case ScalarOpType::ADD: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vaddq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
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
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vsubq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
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
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vmulq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
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
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vdivq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] / scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::EXP:
        case ScalarOpType::SQRT:
        case ScalarOpType::COS:
        case ScalarOpType::SIN: {
            // For complex operations, convert to float32, compute, then convert back
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        float val = static_cast<float>(input[i]);
                        float result;
                        switch (op_type) {
                            case ScalarOpType::EXP: result = std::exp(val); break;
                            case ScalarOpType::SQRT: result = std::sqrt(val); break;
                            case ScalarOpType::COS: result = std::cos(val); break;
                            case ScalarOpType::SIN: result = std::sin(val); break;
                            default: result = val; break;
                        }
                        output[i] = static_cast<__fp16>(result);
                    }
                });
            break;
        }
    }
}