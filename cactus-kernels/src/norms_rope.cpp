#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <vector>

void cactus_rms_norm_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t batch_size,
    size_t dims,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 8;
    constexpr size_t UNROLL_FACTOR = 2;
    constexpr size_t TILE_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    
    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* input_row = input + b * dims;
        __fp16* output_row = output + b * dims;
        
        float32x4_t sum_squares_vec[UNROLL_FACTOR * 2];
        for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
            sum_squares_vec[u] = vdupq_n_f32(0.0f);
        }
        
        size_t i = 0;
        const size_t tile_end = (dims >= TILE_SIZE) ? dims - TILE_SIZE + 1 : 0;
        
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float16x8_t input_vec = vld1q_f16(&input_row[i + u * SIMD_WIDTH]);
                float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input_vec));
                float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input_vec));
                sum_squares_vec[u * 2] = vfmaq_f32(sum_squares_vec[u * 2], input_low, input_low);
                sum_squares_vec[u * 2 + 1] = vfmaq_f32(sum_squares_vec[u * 2 + 1], input_high, input_high);
            }
        }
        
        const size_t simd_end = (dims >= SIMD_WIDTH) ? dims - SIMD_WIDTH + 1 : 0;
        for (; i < simd_end; i += SIMD_WIDTH) {
            float16x8_t input_vec = vld1q_f16(&input_row[i]);
            float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input_vec));
            float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input_vec));
            sum_squares_vec[0] = vfmaq_f32(sum_squares_vec[0], input_low, input_low);
            sum_squares_vec[1] = vfmaq_f32(sum_squares_vec[1], input_high, input_high);
        }
        
        float32x4_t total_sum = sum_squares_vec[0];
        for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
            total_sum = vaddq_f32(total_sum, sum_squares_vec[u]);
        }
        float sum_squares = vaddvq_f32(total_sum);
        
        for (; i < dims; ++i) {
            float val = static_cast<float>(input_row[i]);
            sum_squares += val * val;
        }
        
        float rms = sqrtf(sum_squares / static_cast<float>(dims) + eps);
        float inv_rms = 1.0f / rms;
        float16x8_t inv_rms_vec = vdupq_n_f16(static_cast<__fp16>(inv_rms));
        
        i = 0;
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float16x8_t input_vec = vld1q_f16(&input_row[i + u * SIMD_WIDTH]);
                float16x8_t weight_vec = vld1q_f16(&weight[i + u * SIMD_WIDTH]);
                float16x8_t norm_vec = vmulq_f16(vmulq_f16(input_vec, inv_rms_vec), weight_vec);
                vst1q_f16(&output_row[i + u * SIMD_WIDTH], norm_vec);
            }
        }
        
        for (; i < simd_end; i += SIMD_WIDTH) {
            float16x8_t input_vec = vld1q_f16(&input_row[i]);
            float16x8_t weight_vec = vld1q_f16(&weight[i]);
            float16x8_t norm_vec = vmulq_f16(vmulq_f16(input_vec, inv_rms_vec), weight_vec);
            vst1q_f16(&output_row[i], norm_vec);
        }
        
        for (; i < dims; ++i) {
            output_row[i] = static_cast<__fp16>(static_cast<float>(input_row[i]) * inv_rms * static_cast<float>(weight[i]));
        }
    }
}

void cactus_layer_norm_f16(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t batch_size,
    size_t dims,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 8;
    constexpr size_t UNROLL_FACTOR = 3;
    constexpr size_t TILE_SIZE = SIMD_WIDTH * UNROLL_FACTOR;

    const size_t tile_end = (dims >= TILE_SIZE) ? dims - TILE_SIZE + 1 : 0;
    const size_t simd_end = (dims >= SIMD_WIDTH) ? dims - SIMD_WIDTH + 1 : 0;

    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* input_row = input + b * dims;
        __fp16* output_row = output + b * dims;

        float32x4_t sum_input_vec[UNROLL_FACTOR * 2];
        float32x4_t sum_squares_vec[UNROLL_FACTOR * 2];
        for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
            sum_input_vec[u] = vdupq_n_f32(0.0f);
            sum_squares_vec[u] = vdupq_n_f32(0.0f);
        }

        size_t i = 0;

        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float16x8_t input_vec = vld1q_f16(&input_row[i + u * SIMD_WIDTH]);
                float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input_vec));
                float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input_vec));

                sum_input_vec[u * 2] = vaddq_f32(sum_input_vec[u * 2], input_low);
                sum_input_vec[u * 2 + 1] = vaddq_f32(sum_input_vec[u * 2 + 1], input_high);

                sum_squares_vec[u * 2] = vfmaq_f32(sum_squares_vec[u * 2], input_low, input_low);
                sum_squares_vec[u * 2 + 1] = vfmaq_f32(sum_squares_vec[u * 2 + 1], input_high, input_high);
            }
        }

        for (; i < simd_end; i += SIMD_WIDTH) {
            float16x8_t input_vec = vld1q_f16(&input_row[i]);
            float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input_vec));
            float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input_vec));
            sum_input_vec[0] = vaddq_f32(sum_input_vec[0], input_low);
            sum_input_vec[1] = vaddq_f32(sum_input_vec[1], input_high);
            sum_squares_vec[0] = vfmaq_f32(sum_squares_vec[0], input_low, input_low);
            sum_squares_vec[1] = vfmaq_f32(sum_squares_vec[1], input_high, input_high);
        }

        float32x4_t total_sum_inputs = sum_input_vec[0];
        float32x4_t total_sum_squares = sum_squares_vec[0];
        for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
            total_sum_inputs = vaddq_f32(total_sum_inputs, sum_input_vec[u]);
            total_sum_squares = vaddq_f32(total_sum_squares, sum_squares_vec[u]);
        }

        float sum_inputs = vaddvq_f32(total_sum_inputs);
        float sum_squares = vaddvq_f32(total_sum_squares);
        for (; i < dims; ++i) {
            float val = static_cast<float>(input_row[i]);
            sum_inputs += val;
            sum_squares += val * val;
        }

        float mean = sum_inputs / static_cast<float>(dims);
        float mean_squares = sum_squares / static_cast<float>(dims);
        float variance = mean_squares - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        float inv_std = 1.0f / sqrtf(variance + eps);

        float16x8_t mean_vec = vdupq_n_f16(static_cast<__fp16>(mean));
        float16x8_t inv_std_vec = vdupq_n_f16(static_cast<__fp16>(inv_std));

        i = 0;
        if (bias) {
            for (; i < tile_end; i += TILE_SIZE) {
                for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                    float16x8_t input_vec = vld1q_f16(&input_row[i + u * SIMD_WIDTH]);
                    float16x8_t weight_vec = vld1q_f16(&weight[i + u * SIMD_WIDTH]);
                    float16x8_t bias_vec = vld1q_f16(&bias[i + u * SIMD_WIDTH]);
                    float16x8_t out_vec = vmulq_f16(vmulq_f16(vsubq_f16(input_vec, mean_vec), inv_std_vec), weight_vec);
                    out_vec = vaddq_f16(out_vec, bias_vec);
                    vst1q_f16(&output_row[i + u * SIMD_WIDTH], out_vec);
                }
            }

            for (; i < simd_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&input_row[i]);
                float16x8_t weight_vec = vld1q_f16(&weight[i]);
                float16x8_t bias_vec = vld1q_f16(&bias[i]);
                float16x8_t out_vec = vmulq_f16(vmulq_f16(vsubq_f16(input_vec, mean_vec), inv_std_vec), weight_vec);
                out_vec = vaddq_f16(out_vec, bias_vec);
                vst1q_f16(&output_row[i], out_vec);
            }

            for (; i < dims; ++i) {
                output_row[i] = static_cast<__fp16>((static_cast<float>(input_row[i]) - mean) * inv_std * static_cast<float>(weight[i]) + static_cast<float>(bias[i]));
            }
        } else {
            for (; i < tile_end; i += TILE_SIZE) {
                for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                    float16x8_t input_vec = vld1q_f16(&input_row[i + u * SIMD_WIDTH]);
                    float16x8_t weight_vec = vld1q_f16(&weight[i + u * SIMD_WIDTH]);
                    float16x8_t out_vec = vmulq_f16(vmulq_f16(vsubq_f16(input_vec, mean_vec), inv_std_vec), weight_vec);
                    vst1q_f16(&output_row[i + u * SIMD_WIDTH], out_vec);
                }
            }

            for (; i < simd_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&input_row[i]);
                float16x8_t weight_vec = vld1q_f16(&weight[i]);
                float16x8_t out_vec = vmulq_f16(vmulq_f16(vsubq_f16(input_vec, mean_vec), inv_std_vec), weight_vec);
                vst1q_f16(&output_row[i], out_vec);
            }

            for (; i < dims; ++i) {
                output_row[i] = static_cast<__fp16>((static_cast<float>(input_row[i]) - mean) * inv_std * static_cast<float>(weight[i]));
            }
        }
    }
}

namespace CactusRoPEF16 {

struct RoPECacheF16 {
    std::vector<__fp16> cos_table;
    std::vector<__fp16> sin_table;
    size_t max_seq_len;
    size_t head_dim;
    float theta;
    bool initialized;
    
    RoPECacheF16() : max_seq_len(0), head_dim(0), theta(0.0f), initialized(false) {}
};

static thread_local std::vector<RoPECacheF16> rope_caches_f16;
static thread_local RoPECacheF16* active_rope_cache_f16 = nullptr;

void precompute_rope_tables_f16(size_t seq_len, size_t head_dim, float theta) {
    RoPECacheF16* cache = nullptr;
    for (auto& candidate : rope_caches_f16) {
        if (candidate.initialized && candidate.head_dim == head_dim && candidate.theta == theta) {
            cache = &candidate;
            break;
        }
    }
    if (!cache) {
        rope_caches_f16.emplace_back();
        cache = &rope_caches_f16.back();
        cache->head_dim = head_dim;
        cache->theta = theta;
    }

    active_rope_cache_f16 = cache;
    if (cache->initialized && cache->max_seq_len >= seq_len) {
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t table_size = seq_len * half_dim;

    size_t start_pos = 0;
    if (cache->initialized) {
        start_pos = cache->max_seq_len;
    }

    cache->cos_table.resize(table_size);
    cache->sin_table.resize(table_size);

    for (size_t pos = start_pos; pos < seq_len; ++pos) {
        const float pos_float = static_cast<float>(pos);
        for (size_t i = 0; i < half_dim; ++i) {
            const float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            const float angle = pos_float * freq;

            const size_t idx = pos * half_dim + i;
            cache->cos_table[idx] = static_cast<__fp16>(cosf(angle));
            cache->sin_table[idx] = static_cast<__fp16>(sinf(angle));
        }
    }

    cache->max_seq_len = seq_len;
    cache->initialized = true;
}

}

void cactus_rope_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t start_pos,
    float theta
) {
    const size_t half_dim = head_dim / 2;
    
    CactusRoPEF16::precompute_rope_tables_f16(seq_len + start_pos, head_dim, theta);
    
    const auto& cache = *CactusRoPEF16::active_rope_cache_f16;
    const __fp16* cos_cache = cache.cos_table.data() + start_pos * half_dim;
    const __fp16* sin_cache = cache.sin_table.data() + start_pos * half_dim;

    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t batch_idx = idx / seq_len;
                const size_t seq_idx = idx % seq_len;
                
                for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    const size_t offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
                    const __fp16* input_ptr = input + offset;
                    __fp16* output_ptr = output + offset;
                    
                    const __fp16* cos_ptr = cos_cache + seq_idx * half_dim;
                    const __fp16* sin_ptr = sin_cache + seq_idx * half_dim;
                    
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_half_dim = (half_dim / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    for (size_t i = 0; i < vectorized_half_dim; i += SIMD_WIDTH) {
                        float16x8_t cos_vec = vld1q_f16(&cos_ptr[i]);
                        float16x8_t sin_vec = vld1q_f16(&sin_ptr[i]);
                        
                        float16x8_t x_first_half = vld1q_f16(&input_ptr[i]);
                        float16x8_t x_second_half = vld1q_f16(&input_ptr[i + half_dim]);
                        
                        float16x8_t first_result = vfmsq_f16(vmulq_f16(x_first_half, cos_vec), x_second_half, sin_vec);
                        float16x8_t second_result = vfmaq_f16(vmulq_f16(x_second_half, cos_vec), x_first_half, sin_vec);
                        
                        vst1q_f16(&output_ptr[i], first_result);
                        vst1q_f16(&output_ptr[i + half_dim], second_result);
                    }
                    
                    for (size_t i = vectorized_half_dim; i < half_dim; ++i) {
                        const __fp16 cos_val = cos_ptr[i];
                        const __fp16 sin_val = sin_ptr[i];
                        
                        const __fp16 x_first_half = input_ptr[i];
                        const __fp16 x_second_half = input_ptr[i + half_dim];
                        
                        output_ptr[i] = x_first_half * cos_val - x_second_half * sin_val;
                        
                        output_ptr[i + half_dim] = x_second_half * cos_val + x_first_half * sin_val;
                    }
                }
            }
        });
} 

void cactus_gpt_j_rope_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t rot_dim,
    size_t start_pos,
    float theta
) {
    const size_t half_rot_dim = rot_dim / 2;
    
    CactusRoPEF16::precompute_rope_tables_f16(seq_len + start_pos, rot_dim, theta);
    
    const auto& cache = *CactusRoPEF16::active_rope_cache_f16;
    const __fp16* cos_cache = cache.cos_table.data() + start_pos * half_rot_dim;
    const __fp16* sin_cache = cache.sin_table.data() + start_pos * half_rot_dim;

    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t batch_idx = idx / seq_len;
                const size_t seq_idx = idx % seq_len;
                
                for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    const size_t offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
                    const __fp16* input_ptr = input + offset;
                    __fp16* output_ptr = output + offset;
                    
                    const __fp16* cos_ptr = cos_cache + seq_idx * half_rot_dim;
                    const __fp16* sin_ptr = sin_cache + seq_idx * half_rot_dim;
                    
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_half_rot_dim = (half_rot_dim / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    for (size_t i = 0; i < vectorized_half_rot_dim; i += SIMD_WIDTH) {
                        float16x8_t cos_vec = vld1q_f16(&cos_ptr[i]);
                        float16x8_t sin_vec = vld1q_f16(&sin_ptr[i]);
                        
                        float16x8x2_t x_vec = vld2q_f16(&input_ptr[2*i]);
                        float16x8_t x_first_half = x_vec.val[0];
                        float16x8_t x_second_half = x_vec.val[1];
                        
                        float16x8_t first_result = vfmsq_f16(vmulq_f16(x_first_half, cos_vec), x_second_half, sin_vec);
                        float16x8_t second_result = vfmaq_f16(vmulq_f16(x_second_half, cos_vec), x_first_half, sin_vec);
                        
                        float16x8x2_t t;
                        t.val[0] = first_result;
                        t.val[1] = second_result;
                        vst2q_f16(&output_ptr[2*i], t);
                    }
                    
                    for (size_t i = vectorized_half_rot_dim; i < half_rot_dim; ++i) {
                        const __fp16 cos_val = cos_ptr[i];
                        const __fp16 sin_val = sin_ptr[i];
                        
                        const __fp16 x_first_half = input_ptr[2*i];
                        const __fp16 x_second_half = input_ptr[2*i + 1];
                        
                        output_ptr[2*i] = x_first_half * cos_val - x_second_half * sin_val;
                        
                        output_ptr[2*i + 1] = x_second_half * cos_val + x_first_half * sin_val;
                    }

                    constexpr size_t TAIL_SIMD_WIDTH = 8;
                    size_t copy_idx = rot_dim;
                    const size_t copy_end_vec = (head_dim / TAIL_SIMD_WIDTH) * TAIL_SIMD_WIDTH;

                    for (; copy_idx + TAIL_SIMD_WIDTH <= copy_end_vec; copy_idx += TAIL_SIMD_WIDTH) {
                        float16x8_t v = vld1q_f16(&input_ptr[copy_idx]);
                        vst1q_f16(&output_ptr[copy_idx], v);
                    }
                    for (; copy_idx < head_dim; ++copy_idx) {
                        output_ptr[copy_idx] = input_ptr[copy_idx];
                    }
                }
            }
        });
}
