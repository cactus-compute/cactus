#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <limits>
#include <random>


void cactus_silu_f32(const float* input, float* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            const float32x4_t one = vdupq_n_f32(1.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t x = vld1q_f32(&input[i]);
                
                float32x4_t neg_x = vnegq_f32(x);
                
                float32x4_t exp_vals[4];
                float neg_vals[4], result_vals[4];
                vst1q_f32(neg_vals, neg_x);
                
                for (int j = 0; j < 4; j++) {
                    result_vals[j] = expf(neg_vals[j]);
                }
                exp_vals[0] = vld1q_f32(result_vals);
                
                float32x4_t one_plus_exp = vaddq_f32(one, exp_vals[0]);
                
                float32x4_t sigmoid = vdivq_f32(one, one_plus_exp);
                
                float32x4_t silu = vmulq_f32(x, sigmoid);
                
                vst1q_f32(&output[i], silu);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float sigmoid = 1.0f / (1.0f + expf(-input[i]));
                output[i] = input[i] * sigmoid;
            }
        });
}

void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));

                float32x4_t neg_x_low = vnegq_f32(x_low);
                float32x4_t neg_x_high = vnegq_f32(x_high);

                float exp_vals[8];
                vst1q_f32(&exp_vals[0], neg_x_low);
                vst1q_f32(&exp_vals[4], neg_x_high);

                for (int j = 0; j < 8; j++) {
                    exp_vals[j] = expf(exp_vals[j]);
                }

                float32x4_t exp_low = vld1q_f32(&exp_vals[0]);
                float32x4_t exp_high = vld1q_f32(&exp_vals[4]);

                float32x4_t one_f32 = vdupq_n_f32(1.0f);
                float32x4_t one_plus_exp_low = vaddq_f32(one_f32, exp_low);
                float32x4_t one_plus_exp_high = vaddq_f32(one_f32, exp_high);

                float32x4_t sigmoid_low = vdivq_f32(one_f32, one_plus_exp_low);
                float32x4_t sigmoid_high = vdivq_f32(one_f32, one_plus_exp_high);

                float16x4_t sigmoid_low_f16 = vcvt_f16_f32(sigmoid_low);
                float16x4_t sigmoid_high_f16 = vcvt_f16_f32(sigmoid_high);
                float16x8_t sigmoid = vcombine_f16(sigmoid_low_f16, sigmoid_high_f16);

                float16x8_t silu = vmulq_f16(x, sigmoid);

                vst1q_f16(&output[i], silu);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x_f32 = static_cast<float>(input[i]);
                float sigmoid = 1.0f / (1.0f + expf(-x_f32));
                output[i] = static_cast<__fp16>(x_f32 * sigmoid);
            }
        });
}


void cactus_silu_int8(const int8_t* input, int8_t* output, size_t num_elements, 
                      float input_scale, float output_scale) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                float x = input[i] * input_scale;
                float sigmoid = 1.0f / (1.0f + expf(-x));
                float silu = x * sigmoid;
                float scaled = silu / output_scale;
                output[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, roundf(scaled))));
            }
        });
}

void cactus_gelu_f32(const float* input, float* output, size_t num_elements) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            const float32x4_t half = vdupq_n_f32(0.5f);
            const float32x4_t one = vdupq_n_f32(1.0f);
            const float32x4_t sqrt_2_pi_vec = vdupq_n_f32(sqrt_2_over_pi);
            const float32x4_t coeff_vec = vdupq_n_f32(coeff);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t x = vld1q_f32(&input[i]);
                
                float32x4_t x_cubed = vmulq_f32(vmulq_f32(x, x), x);
                float32x4_t inner = vmlaq_f32(x, coeff_vec, x_cubed);
                float32x4_t tanh_arg = vmulq_f32(sqrt_2_pi_vec, inner);
                
                float tanh_vals[4], arg_vals[4];
                vst1q_f32(arg_vals, tanh_arg);
                for (int j = 0; j < 4; j++) {
                    tanh_vals[j] = tanhf(arg_vals[j]);
                }
                float32x4_t tanh_result = vld1q_f32(tanh_vals);
                
                float32x4_t gelu = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_result));
                
                vst1q_f32(&output[i], gelu);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x = input[i];
                float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                output[i] = 0.5f * x * (1.0f + tanhf(inner));
            }
        });
}

void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            const float32x4_t half = vdupq_n_f32(0.5f);
            const float32x4_t one = vdupq_n_f32(1.0f);
            const float32x4_t sqrt_2_pi_vec = vdupq_n_f32(sqrt_2_over_pi);
            const float32x4_t coeff_vec = vdupq_n_f32(coeff);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x_f16 = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_f16));

                float32x4_t x_cubed_low = vmulq_f32(vmulq_f32(x_low, x_low), x_low);
                float32x4_t x_cubed_high = vmulq_f32(vmulq_f32(x_high, x_high), x_high);

                float32x4_t inner_low = vmlaq_f32(x_low, coeff_vec, x_cubed_low);
                float32x4_t inner_high = vmlaq_f32(x_high, coeff_vec, x_cubed_high);
                inner_low = vmulq_f32(sqrt_2_pi_vec, inner_low);
                inner_high = vmulq_f32(sqrt_2_pi_vec, inner_high);

                float tanh_vals[8];
                vst1q_f32(&tanh_vals[0], inner_low);
                vst1q_f32(&tanh_vals[4], inner_high);
                for (int j = 0; j < 8; j++) {
                    tanh_vals[j] = tanhf(tanh_vals[j]);
                }
                float32x4_t tanh_low = vld1q_f32(&tanh_vals[0]);
                float32x4_t tanh_high = vld1q_f32(&tanh_vals[4]);

                float32x4_t one_plus_tanh_low = vaddq_f32(one, tanh_low);
                float32x4_t one_plus_tanh_high = vaddq_f32(one, tanh_high);
                float32x4_t gelu_low = vmulq_f32(vmulq_f32(half, x_low), one_plus_tanh_low);
                float32x4_t gelu_high = vmulq_f32(vmulq_f32(half, x_high), one_plus_tanh_high);

                float16x4_t gelu_low_f16 = vcvt_f16_f32(gelu_low);
                float16x4_t gelu_high_f16 = vcvt_f16_f32(gelu_high);
                float16x8_t gelu_f16 = vcombine_f16(gelu_low_f16, gelu_high_f16);

                vst1q_f16(&output[i], gelu_f16);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x = static_cast<float>(input[i]);
                float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                float gelu = 0.5f * x * (1.0f + tanhf(inner));
                output[i] = static_cast<__fp16>(gelu);
            }
        });
}

void cactus_gelu_int8(const int8_t* input, int8_t* output, size_t num_elements,
                      float input_scale, float output_scale) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                float x = input[i] * input_scale;
                float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                float gelu = 0.5f * x * (1.0f + tanhf(inner));
                float scaled = gelu / output_scale;
                output[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, roundf(scaled))));
            }
        });
}


namespace CactusSoftmax {

inline float32x4_t fast_exp_neon(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
    const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);
    const float32x4_t c4 = vdupq_n_f32(0.009618129842071803f);
    const float32x4_t c5 = vdupq_n_f32(0.001333355814670656f);
    
    const float32x4_t clamp_min = vdupq_n_f32(-87.0f);
    const float32x4_t clamp_max = vdupq_n_f32(87.0f);
    
    x = vmaxq_f32(x, clamp_min);
    x = vminq_f32(x, clamp_max);
    
    x = vmulq_f32(x, log2e);
    
    int32x4_t xi = vcvtq_s32_f32(x);
    float32x4_t xf = vsubq_f32(x, vcvtq_f32_s32(xi));
    
    float32x4_t p = vdupq_n_f32(1.0f);
    p = vfmaq_f32(p, c1, xf);
    
    float32x4_t xf2 = vmulq_f32(xf, xf);
    p = vfmaq_f32(p, c2, xf2);
    
    float32x4_t xf3 = vmulq_f32(xf2, xf);
    p = vfmaq_f32(p, c3, xf3);
    
    float32x4_t xf4 = vmulq_f32(xf3, xf);
    p = vfmaq_f32(p, c4, xf4);
    
    float32x4_t xf5 = vmulq_f32(xf4, xf);
    p = vfmaq_f32(p, c5, xf5);
    
    int32x4_t exponent = vaddq_s32(xi, vdupq_n_s32(127));
    exponent = vshlq_n_s32(exponent, 23);
    float32x4_t scale = vreinterpretq_f32_s32(exponent);
    
    return vmulq_f32(p, scale);
}

inline float32x4_t fast_reciprocal_neon(float32x4_t x) {
    float32x4_t recip = vrecpeq_f32(x);
    recip = vmulq_f32(recip, vrecpsq_f32(x, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(x, recip));
    return recip;
}

void kernel_softmax_neon_optimized_single(const float* input, float* output, size_t vocab_size) {
    constexpr size_t SIMD_WIDTH = 4;
    constexpr size_t UNROLL_FACTOR = 8;
    constexpr size_t VECTORIZED_WIDTH = SIMD_WIDTH * UNROLL_FACTOR;
    const size_t vocab_vectorized = (vocab_size / VECTORIZED_WIDTH) * VECTORIZED_WIDTH;
    
    float32x4_t max_vec[UNROLL_FACTOR];
    for (size_t u = 0; u < UNROLL_FACTOR; u++) {
        max_vec[u] = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    }
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float32x4_t x_vec = vld1q_f32(&input[i + u * SIMD_WIDTH]);
            max_vec[u] = vmaxq_f32(max_vec[u], x_vec);
        }
    }
    
    float32x4_t final_max = max_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR; u++) {
        final_max = vmaxq_f32(final_max, max_vec[u]);
    }
    
    float max_val = vmaxvq_f32(final_max);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    const float32x4_t max_broadcast = vdupq_n_f32(max_val);
    
    float32x4_t sum_vec[UNROLL_FACTOR];
    for (size_t u = 0; u < UNROLL_FACTOR; u++) {
        sum_vec[u] = vdupq_n_f32(0.0f);
    }
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        float32x4_t x_vec[UNROLL_FACTOR];
        float32x4_t exp_vec[UNROLL_FACTOR];
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            x_vec[u] = vld1q_f32(&input[i + u * SIMD_WIDTH]);
        }
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            exp_vec[u] = fast_exp_neon(vsubq_f32(x_vec[u], max_broadcast));
            sum_vec[u] = vaddq_f32(sum_vec[u], exp_vec[u]);
        }
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            vst1q_f32(&output[i + u * SIMD_WIDTH], exp_vec[u]);
        }
    }
    
    float32x4_t final_sum = sum_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR; u++) {
        final_sum = vaddq_f32(final_sum, sum_vec[u]);
    }
    
    float sum = vaddvq_f32(final_sum);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    const float32x4_t inv_sum_vec = fast_reciprocal_neon(vdupq_n_f32(sum));
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        float32x4_t exp_vec[UNROLL_FACTOR];
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            exp_vec[u] = vld1q_f32(&output[i + u * SIMD_WIDTH]);
        }
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            exp_vec[u] = vmulq_f32(exp_vec[u], inv_sum_vec);
        }
        
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            vst1q_f32(&output[i + u * SIMD_WIDTH], exp_vec[u]);
        }
    }
    
    const float inv_sum = vgetq_lane_f32(inv_sum_vec, 0);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        output[i] *= inv_sum;
    }
}

}

void cactus_softmax_f32(
    const float* input,
    float* output,
    size_t batch_size,
    size_t seq_len,
    size_t vocab_size
) {
    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t offset = idx * vocab_size;
                CactusSoftmax::kernel_softmax_neon_optimized_single(input + offset, output + offset, vocab_size);
            }
        });
}

void kernel_softmax_f16_single(const __fp16* input, __fp16* output, size_t vocab_size) {
    constexpr size_t SIMD_WIDTH = 8;  
    constexpr size_t UNROLL_FACTOR = 4; 
    constexpr size_t VECTORIZED_WIDTH = SIMD_WIDTH * UNROLL_FACTOR;
    const size_t vocab_vectorized = (vocab_size / VECTORIZED_WIDTH) * VECTORIZED_WIDTH;
    
    float32x4_t max_vec[UNROLL_FACTOR * 2];  
    for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
        max_vec[u] = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    }
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t x_vec_f16 = vld1q_f16(&input[i + u * SIMD_WIDTH]);
            float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_vec_f16));
            float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_vec_f16));
            max_vec[u * 2] = vmaxq_f32(max_vec[u * 2], x_low);
            max_vec[u * 2 + 1] = vmaxq_f32(max_vec[u * 2 + 1], x_high);
        }
    }
    
    float32x4_t final_max = max_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
        final_max = vmaxq_f32(final_max, max_vec[u]);
    }
    
    float max_val = vmaxvq_f32(final_max);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        max_val = std::max(max_val, static_cast<float>(input[i]));
    }
    
    const float32x4_t max_broadcast = vdupq_n_f32(max_val);
    const float16x8_t max_broadcast_f16 = vcombine_f16(
        vcvt_f16_f32(max_broadcast), 
        vcvt_f16_f32(max_broadcast)
    );
    
    float32x4_t sum_vec[UNROLL_FACTOR * 2];
    for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
        sum_vec[u] = vdupq_n_f32(0.0f);
    }
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t x_vec_f16 = vld1q_f16(&input[i + u * SIMD_WIDTH]);
            
            float16x8_t centered_f16 = vsubq_f16(x_vec_f16, max_broadcast_f16);
            
            float32x4_t centered_low = vcvt_f32_f16(vget_low_f16(centered_f16));
            float32x4_t centered_high = vcvt_f32_f16(vget_high_f16(centered_f16));
            
            float32x4_t exp_low = CactusSoftmax::fast_exp_neon(centered_low);
            float32x4_t exp_high = CactusSoftmax::fast_exp_neon(centered_high);
            
            float16x8_t exp_f16 = vcombine_f16(vcvt_f16_f32(exp_low), vcvt_f16_f32(exp_high));
            vst1q_f16(&output[i + u * SIMD_WIDTH], exp_f16);
            
            sum_vec[u * 2] = vaddq_f32(sum_vec[u * 2], exp_low);
            sum_vec[u * 2 + 1] = vaddq_f32(sum_vec[u * 2 + 1], exp_high);
        }
    }
    
    float32x4_t final_sum = sum_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
        final_sum = vaddq_f32(final_sum, sum_vec[u]);
    }
    
    float sum = vaddvq_f32(final_sum);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        float exp_val = expf(static_cast<float>(input[i]) - max_val);
        output[i] = static_cast<__fp16>(exp_val);
        sum += exp_val;
    }
    
    const float inv_sum = 1.0f / sum;
    const float16x8_t inv_sum_vec_f16 = vdupq_n_f16(static_cast<__fp16>(inv_sum));
    
    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t exp_vec = vld1q_f16(&output[i + u * SIMD_WIDTH]);
            float16x8_t result = vmulq_f16(exp_vec, inv_sum_vec_f16);
            vst1q_f16(&output[i + u * SIMD_WIDTH], result);
        }
    }
    
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(output[i]) * inv_sum);
    }
}

void cactus_softmax_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t vocab_size
) {
    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t offset = idx * vocab_size;
                kernel_softmax_f16_single(input + offset, output + offset, vocab_size);
            }
        });
}

void cactus_sample_f32(const float* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed) {
    std::vector<float> filtered_logits(vocab_size);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        filtered_logits[i] = logits[i];
    }
    
    
    if (temperature > 0) {
        for (size_t i = 0; i < vocab_size; ++i) {
            filtered_logits[i] /= temperature;
        }
    }
    
    if (top_k > 0) {
        std::vector<std::pair<float, size_t>> logit_pairs;
        logit_pairs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_pairs.emplace_back(filtered_logits[i], i);
        }
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        if (top_k < vocab_size) {
            float kth_value = logit_pairs[top_k - 1].first;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (filtered_logits[i] < kth_value) {
                    filtered_logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted_logits;
        sorted_logits.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!std::isinf(filtered_logits[i])) {
                sorted_logits.emplace_back(filtered_logits[i], i);
            }
        }
        std::sort(sorted_logits.begin(), sorted_logits.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float max_logit = sorted_logits.empty() ? 0.0f : sorted_logits[0].first;
        std::vector<float> temp_probs;
        temp_probs.reserve(sorted_logits.size());
        float sum = 0.0f;
        for (const auto& pair : sorted_logits) {
            float prob = std::exp(pair.first - max_logit);
            temp_probs.push_back(prob);
            sum += prob;
        }
        
        for (float& prob : temp_probs) {
            prob /= sum;
        }
        
        float cumulative_prob = 0.0f;
        std::vector<bool> indices_to_remove(sorted_logits.size(), false);
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            cumulative_prob += temp_probs[i];
            if (cumulative_prob > top_p) {
                indices_to_remove[i] = true;
            }
        }
        
        if (!indices_to_remove.empty()) {
            for (size_t i = 1; i < indices_to_remove.size(); ++i) {
                indices_to_remove[i] = indices_to_remove[i-1] || indices_to_remove[i];
            }
            indices_to_remove[0] = false;
        }
        
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            if (indices_to_remove[i]) {
                filtered_logits[sorted_logits[i].second] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    float max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
    if (std::isinf(max_logit)) {
        output[0] = 0;
        return;
    }
    
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isinf(filtered_logits[i])) {
            probs[i] = 0.0f;
        } else {
            probs[i] = std::exp(filtered_logits[i] - max_logit);
            sum += probs[i];
        }
    }
    
    if (sum == 0.0f) {
        output[0] = 0;
        return;
    }
    
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }
    
    uint32_t actual_seed = (random_seed == 0) ? std::random_device{}() : random_seed;
    std::mt19937 gen(actual_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);
    
    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (cumulative >= sample) {
            output[0] = static_cast<uint32_t>(i);
            return;
        }
    }
    
    for (size_t i = vocab_size; i > 0; --i) {
        if (probs[i-1] > 0.0f) {
            output[0] = static_cast<uint32_t>(i-1);
            return;
        }
    }
    
    output[0] = 0;
}

void cactus_sample_f16(const __fp16* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed) {
    std::vector<__fp16> filtered_logits(vocab_size);

    if (temperature > 0) {
        __fp16 inv_temp = static_cast<__fp16>(1.0f / temperature);
        float16x8_t inv_temp_vec = vdupq_n_f16(inv_temp);
        size_t i = 0;
        for (; i + 8 <= vocab_size; i += 8) {
            float16x8_t logits_vec = vld1q_f16(&logits[i]);
            float16x8_t scaled = vmulq_f16(logits_vec, inv_temp_vec);
            vst1q_f16(&filtered_logits[i], scaled);
        }
        for (; i < vocab_size; ++i) {
            filtered_logits[i] = logits[i] * inv_temp;
        }
    } else {
        std::memcpy(filtered_logits.data(), logits, vocab_size * sizeof(__fp16));
    }

    static std::vector<uint32_t> token_history;
    static const size_t MAX_HISTORY = 128; 
    static const float REPETITION_PENALTY = 1.1f;

    if (!token_history.empty() && REPETITION_PENALTY != 1.0f) {
        const __fp16 penalty_inv = static_cast<__fp16>(1.0f / REPETITION_PENALTY);
        const __fp16 penalty = static_cast<__fp16>(REPETITION_PENALTY);

        for (uint32_t prev_token : token_history) {
            if (prev_token < vocab_size) {
                filtered_logits[prev_token] = (filtered_logits[prev_token] > static_cast<__fp16>(0))
                    ? static_cast<__fp16>(filtered_logits[prev_token] * penalty_inv)
                    : static_cast<__fp16>(filtered_logits[prev_token] * penalty);
            }
        }
    }

    if (top_k > 0) {
        std::vector<std::pair<__fp16, size_t>> logit_pairs;
        logit_pairs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_pairs.emplace_back(filtered_logits[i], i);
        }
        std::partial_sort(logit_pairs.begin(),
                         logit_pairs.begin() + std::min(top_k, vocab_size),
                         logit_pairs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        if (top_k < vocab_size) {
            __fp16 kth_value = logit_pairs[top_k - 1].first;
            __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < vocab_size; ++i) {
                if (filtered_logits[i] < kth_value) {
                    filtered_logits[i] = neg_inf;
                }
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<__fp16, size_t>> sorted_logits;
        sorted_logits.reserve(vocab_size);
        __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vocab_size; ++i) {
            if (filtered_logits[i] != neg_inf) {
                sorted_logits.emplace_back(filtered_logits[i], i);
            }
        }
        std::sort(sorted_logits.begin(), sorted_logits.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        __fp16 max_logit = sorted_logits.empty() ? static_cast<__fp16>(0.0f) : sorted_logits[0].first;
        std::vector<float> temp_probs;
        temp_probs.reserve(sorted_logits.size());
        float sum = 0.0f;
        for (const auto& pair : sorted_logits) {
            float prob = std::exp(static_cast<float>(pair.first - max_logit));
            temp_probs.push_back(prob);
            sum += prob;
        }

        for (float& prob : temp_probs) {
            prob /= sum;
        }

        float cumulative_prob = 0.0f;
        std::vector<bool> indices_to_remove(sorted_logits.size(), false);
        bool threshold_reached = false;
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            cumulative_prob += temp_probs[i];
            if (cumulative_prob > top_p && i > 0) { 
                threshold_reached = true;
            }
            if (threshold_reached) {
                indices_to_remove[i] = true;
            }
        }

        if (!indices_to_remove.empty()) {
            for (size_t i = 1; i < indices_to_remove.size(); ++i) {
                indices_to_remove[i] = indices_to_remove[i-1] || indices_to_remove[i];
            }
            indices_to_remove[0] = false;
        }
        
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            if (indices_to_remove[i]) {
                filtered_logits[sorted_logits[i].second] = neg_inf;
            }
        }
    }
    
    __fp16 max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
    __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
    if (max_logit == neg_inf) {
        output[0] = 0;
        return;
    }
    
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (filtered_logits[i] == neg_inf) {
            probs[i] = 0.0f;
        } else {
            probs[i] = std::exp(static_cast<float>(filtered_logits[i] - max_logit));
            sum += probs[i];
        }
    }

    if (sum == 0.0f) {
        output[0] = 0;
        return;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    uint32_t actual_seed = (random_seed == 0) ? std::random_device{}() : random_seed;
    std::mt19937 gen(actual_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);

    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (cumulative >= sample) {
            output[0] = static_cast<uint32_t>(i);
            token_history.push_back(output[0]);
            if (token_history.size() > MAX_HISTORY) {
                token_history.erase(token_history.begin());
            }
            return;
        }
    }

    for (size_t i = vocab_size; i > 0; --i) {
        if (probs[i-1] > 0.0f) {
            output[0] = static_cast<uint32_t>(i-1);
            token_history.push_back(output[0]);
            if (token_history.size() > MAX_HISTORY) {
                token_history.erase(token_history.begin());
            }
            return;
        }
    }

    output[0] = 0;
    token_history.push_back(output[0]);
    if (token_history.size() > MAX_HISTORY) {
        token_history.erase(token_history.begin());
    }
}
