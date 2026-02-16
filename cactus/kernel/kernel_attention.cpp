#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>

static inline void cactus_attention_f16_h64(
    const __fp16* queries,
    const __fp16* keys,
    const __fp16* values,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    float scale,
    size_t position_offset,
    bool is_causal
) {
    constexpr size_t HEAD_DIM = 64;
    constexpr size_t BLOCK_SIZE = 32;
    constexpr float NEG_INF = -INFINITY;

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * HEAD_DIM;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * HEAD_DIM;
    const size_t o_batch_stride = q_batch_stride;
    const size_t q_seq_stride = num_q_heads * HEAD_DIM;
    const size_t kv_seq_stride = num_kv_heads * HEAD_DIM;
    const size_t o_seq_stride = q_seq_stride;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start, size_t end) {

        float block_scores[BLOCK_SIZE];

        for (size_t work = start; work < end; ++work) {
            const size_t batch = work / (num_q_heads * seq_len);
            const size_t rem = work % (num_q_heads * seq_len);
            const size_t q_head = rem / seq_len;
            const size_t q_pos = rem % seq_len;
            const size_t kv_head = q_head / group_size;

            const __fp16* q = queries + batch*q_batch_stride + q_pos*q_seq_stride + q_head*HEAD_DIM;
            __fp16* o = output + batch*o_batch_stride + q_pos*o_seq_stride + q_head*HEAD_DIM;

            float32x4_t acc_lo[8], acc_hi[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                acc_lo[i] = vdupq_n_f32(0.f);
                acc_hi[i] = vdupq_n_f32(0.f);
            }

            float running_max = NEG_INF;
            float running_sum = 0.f;

            const size_t abs_q = position_offset + q_pos;
            size_t kv_end = is_causal ? std::min(kv_seq_len, abs_q + 1) : kv_seq_len;

            for (size_t kv0 = 0; kv0 < kv_end; kv0 += BLOCK_SIZE) {
                const size_t kv1 = std::min(kv0 + BLOCK_SIZE, kv_end);
                float block_max = NEG_INF;

                for (size_t i = kv0; i < kv1; i++) {
                    float32x4_t s0 = vdupq_n_f32(0.f);
                    float32x4_t s1 = vdupq_n_f32(0.f);

                    const __fp16* k = keys + batch*kv_batch_stride + i*kv_seq_stride + kv_head*HEAD_DIM;

                    #pragma unroll
                    for (int d = 0; d < 8; d++) {
                        float16x8_t qv = vld1q_f16(q + d*8);
                        float16x8_t kv = vld1q_f16(k + d*8);

                        float32x4_t ql = vcvt_f32_f16(vget_low_f16(qv));
                        float32x4_t qh = vcvt_f32_f16(vget_high_f16(qv));
                        float32x4_t kl = vcvt_f32_f16(vget_low_f16(kv));
                        float32x4_t kh = vcvt_f32_f16(vget_high_f16(kv));

                        s0 = vfmaq_f32(s0, ql, kl);
                        s1 = vfmaq_f32(s1, qh, kh);
                    }

                    float score = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
                    block_scores[i - kv0] = score;
                    block_max = std::max(block_max, score);
                }

                float scale_corr = expf(running_max - block_max);
                running_sum *= scale_corr;

                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    acc_lo[d] = vmulq_n_f32(acc_lo[d], scale_corr);
                    acc_hi[d] = vmulq_n_f32(acc_hi[d], scale_corr);
                }

                float block_sum = 0.f;
                for (size_t i = 0; i < kv1 - kv0; i++) {
                    block_scores[i] = expf(block_scores[i] - block_max);
                    block_sum += block_scores[i];
                }

                for (size_t i = 0; i < kv1 - kv0; i++) {
                    float w = block_scores[i];
                    if (w == 0.f) continue;

                    const __fp16* v = values + batch*kv_batch_stride + (kv0+i)*kv_seq_stride + kv_head*HEAD_DIM;
                    float32x4_t wv = vdupq_n_f32(w);

                    #pragma unroll
                    for (int d = 0; d < 8; d++) {
                        float16x8_t vv = vld1q_f16(v + d*8);
                        acc_lo[d] = vfmaq_f32(acc_lo[d], vcvt_f32_f16(vget_low_f16(vv)), wv);
                        acc_hi[d] = vfmaq_f32(acc_hi[d], vcvt_f32_f16(vget_high_f16(vv)), wv);
                    }
                }

                running_sum += block_sum;
                running_max = block_max;
            }

            if (running_sum == 0.f) {
                memset(o, 0, HEAD_DIM * sizeof(__fp16));
                continue;
            }

            float inv = 1.f / running_sum;
            float32x4_t invv = vdupq_n_f32(inv);

            #pragma unroll
            for (int d = 0; d < 8; d++) {
                float16x8_t out = vcombine_f16(
                    vcvt_f16_f32(vmulq_f32(acc_lo[d], invv)),
                    vcvt_f16_f32(vmulq_f32(acc_hi[d], invv))
                );
                vst1q_f16(o + d*8, out);
            }
        }
    });
}

void cactus_attention_f16(
    const __fp16* queries,
    const __fp16* keys,
    const __fp16* values,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    const __fp16* mask,
    size_t position_offset,
    size_t window_size,
    bool is_causal
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    if (head_dim == 64 && mask == nullptr && window_size == 0) {
        cactus_attention_f16_h64(
            queries, keys, values, output,
            batch_size, seq_len, kv_seq_len,
            num_q_heads, num_kv_heads,
            scale, position_offset, is_causal
        );
        return;
    }

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;

    const size_t group_size = num_q_heads / num_kv_heads;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = num_q_heads * head_dim;
    const size_t mask_batch_stride = mask ? seq_len * kv_seq_len : 0;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            std::vector<float> block_scores(BLOCK_SIZE);
            std::vector<float32x4_t> output_accum_low(head_dim_aligned / VECTOR_WIDTH * 2);
            std::vector<float32x4_t> output_accum_high(head_dim_aligned / VECTOR_WIDTH * 2);
            
            const size_t tail_dims = head_dim - head_dim_aligned;
            std::vector<float> output_accum_tail(tail_dims, 0.0f);

            const float NEG_INF = -std::numeric_limits<float>::infinity();
            const size_t used_vec_blocks = head_dim_aligned / VECTOR_WIDTH;

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;

                const size_t kv_head_idx = q_head_idx / group_size;

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const __fp16* K_base = keys + batch_idx * kv_batch_stride;
                const __fp16* V_base = values + batch_idx * kv_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;
                const __fp16* M = mask ? (mask + batch_idx * mask_batch_stride) : nullptr;
                    const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                    __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;
                    
                    float running_max = -std::numeric_limits<float>::infinity();
                    float running_sum = 0.0f;
                    
                    for (size_t i = 0; i < output_accum_low.size(); ++i) {
                        output_accum_low[i] = vdupq_n_f32(0.0f);
                        output_accum_high[i] = vdupq_n_f32(0.0f);
                    }
                    for (size_t i = 0; i < tail_dims; ++i) {
                        output_accum_tail[i] = 0.0f;
                    }
                    
                    const bool is_decode = (q_pos == seq_len - 1) && seq_len > 1;
                    const size_t absolute_q_pos = position_offset + q_pos;

                    size_t kv_start = 0;
                    size_t kv_end = kv_seq_len;

                    if (window_size > 0 && window_size < kv_seq_len) {
                        if (absolute_q_pos > window_size) {
                            kv_start = absolute_q_pos - window_size;
                        }
                        if (is_causal) {
                            kv_end = std::min(kv_end, absolute_q_pos + 1);
                        }
                    } else if (is_causal) {
                        kv_end = std::min(kv_end, absolute_q_pos + 1);
                    }

                    for (size_t kv_block_start = kv_start; kv_block_start < kv_end; kv_block_start += BLOCK_SIZE) {
                        const size_t kv_block_end = std::min(kv_block_start + BLOCK_SIZE, kv_end);
                        const size_t block_size = kv_block_end - kv_block_start;

                        float block_max = -std::numeric_limits<float>::infinity();

                        if (!is_decode && is_causal && kv_block_start > absolute_q_pos) {
                            for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                                block_scores[kv_idx] = NEG_INF;
                            }
                            continue; 
                        }

                        for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                            const size_t kv_pos = kv_block_start + kv_idx;

                            if (!is_decode && is_causal && kv_pos > absolute_q_pos) {
                                block_scores[kv_idx] = NEG_INF;
                                continue;
                            }

                            const __fp16* k_vec = K_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                            if (kv_idx + 1 < block_size) {
                                const __fp16* next_k_vec = K_base + (kv_pos + 1) * kv_seq_stride + kv_head_idx * head_dim;
                                __builtin_prefetch(next_k_vec, 0, 1);
                            }

                            float32x4_t score_accum_low = vdupq_n_f32(0.0f);
                            float32x4_t score_accum_high = vdupq_n_f32(0.0f);
                            
                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t q_vec_f16 = vld1q_f16(&q_vec[dim_block]);
                                float16x8_t k_vec_f16 = vld1q_f16(&k_vec[dim_block]);
                                
                                float32x4_t q_low = vcvt_f32_f16(vget_low_f16(q_vec_f16));
                                float32x4_t q_high = vcvt_f32_f16(vget_high_f16(q_vec_f16));
                                float32x4_t k_low = vcvt_f32_f16(vget_low_f16(k_vec_f16));
                                float32x4_t k_high = vcvt_f32_f16(vget_high_f16(k_vec_f16));
                                
                                score_accum_low = vfmaq_f32(score_accum_low, q_low, k_low);
                                score_accum_high = vfmaq_f32(score_accum_high, q_high, k_high);
                            }
                            
                            float score = vaddvq_f32(vaddq_f32(score_accum_low, score_accum_high));
                            
                            for (size_t dim = head_dim_aligned; dim < head_dim; ++dim) {
                                score += static_cast<float>(q_vec[dim]) * static_cast<float>(k_vec[dim]);
                            }
                            
                            score *= scale;
                            
                            size_t absolute_q_pos = position_offset + q_pos;

                            if (is_causal && kv_pos > absolute_q_pos) {
                                score = NEG_INF;
                            }
                            else if (window_size > 0 && kv_pos < absolute_q_pos && (absolute_q_pos - kv_pos) > window_size) {
                                score = NEG_INF;
                            }
                            else if (M && static_cast<float>(M[q_pos * kv_seq_len + kv_pos]) == 0.0f) {
                                score = NEG_INF;
                            }
                            
                            block_scores[kv_idx] = score;
                            block_max = std::max(block_max, score);
                        }
                        
                        float current_block_scale = 1.0f;

                        if (block_max > NEG_INF) {
                            if (block_max > running_max) {
                            float scale_correction = expf(running_max - block_max);
                            running_sum *= scale_correction;
                            
                            for (size_t i = 0; i < used_vec_blocks; ++i) {
                                output_accum_low[i] = vmulq_n_f32(output_accum_low[i], scale_correction);
                                output_accum_high[i] = vmulq_n_f32(output_accum_high[i], scale_correction);
                            }
                            for (size_t i = 0; i < tail_dims; ++i) {
                                output_accum_tail[i] *= scale_correction;
                            }
                            running_max = block_max;
                            } else {
                                current_block_scale = expf(block_max - running_max);
                            }
                        }
                        
                        float block_sum = 0.0f;
                        const size_t vec_size = (block_size / 4) * 4;

                        for (size_t kv_idx = 0; kv_idx < vec_size; kv_idx += 4) {
                            float32x4_t scores = vld1q_f32(&block_scores[kv_idx]);
                            uint32x4_t inf_mask = vceqq_f32(scores, vdupq_n_f32(NEG_INF));

                            float32x4_t x = vsubq_f32(scores, vdupq_n_f32(block_max));
                            x = vmulq_n_f32(x, 1.442695f); 
                            float32x4_t x_floor = vrndmq_f32(x);
                            int32x4_t xi = vcvtq_s32_f32(x_floor);
                            float32x4_t xf = vsubq_f32(x, x_floor);

                            float32x4_t y = vfmaq_n_f32(vdupq_n_f32(1.0f), xf, 0.6931472f);
                            y = vfmaq_f32(y, vmulq_f32(xf, xf), vdupq_n_f32(0.2402265f));

                            xi = vaddq_s32(xi, vdupq_n_s32(127));
                            xi = vshlq_n_s32(xi, 23);
                            y = vmulq_f32(y, vreinterpretq_f32_s32(xi));

                            uint32x4_t underflow_mask = vcltq_f32(x, vdupq_n_f32(-126.0f));
                            uint32x4_t zero_mask = vorrq_u32(inf_mask, underflow_mask);
                            y = vbslq_f32(zero_mask, vdupq_n_f32(0.0f), y);

                            vst1q_f32(&block_scores[kv_idx], y);
                            block_sum += vaddvq_f32(y);
                        }

                        for (size_t kv_idx = vec_size; kv_idx < block_size; ++kv_idx) {
                            if (block_scores[kv_idx] != NEG_INF) {
                                block_scores[kv_idx] = expf(block_scores[kv_idx] - block_max);
                                block_sum += block_scores[kv_idx];
                            } else {
                                block_scores[kv_idx] = 0.0f;
                            }
                        }
                        
                        for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                            const float attn_weight = block_scores[kv_idx] * current_block_scale;
                            if (attn_weight == 0.0f) continue;
                            
                            const size_t kv_pos = kv_block_start + kv_idx;
                            const __fp16* v_vec = V_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                            
                            const float32x4_t weight_vec = vdupq_n_f32(attn_weight);
                            
                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t v_vec_f16 = vld1q_f16(&v_vec[dim_block]);
                                float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v_vec_f16));
                                float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v_vec_f16));
                                
                                size_t idx = dim_block / VECTOR_WIDTH;
                                output_accum_low[idx] = vfmaq_f32(output_accum_low[idx], v_low, weight_vec);
                                output_accum_high[idx] = vfmaq_f32(output_accum_high[idx], v_high, weight_vec);
                            }
                            
                            for (size_t dim = head_dim_aligned; dim < head_dim; ++dim) {
                                float val = attn_weight * static_cast<float>(v_vec[dim]);
                                output_accum_tail[dim - head_dim_aligned] += val;
                            }
                        }
                        
                        running_sum += block_sum * current_block_scale;
                    }
                    
                    if (running_sum > 0.0f) {
                        const float inv_sum = 1.0f / running_sum;
                        const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
                        
                        for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                            size_t idx = dim_block / VECTOR_WIDTH;
                            float32x4_t final_low = vmulq_f32(output_accum_low[idx], inv_sum_vec);
                            float32x4_t final_high = vmulq_f32(output_accum_high[idx], inv_sum_vec);
                            
                            float16x4_t low_f16 = vcvt_f16_f32(final_low);
                            float16x4_t high_f16 = vcvt_f16_f32(final_high);
                            float16x8_t combined = vcombine_f16(low_f16, high_f16);
                            
                            vst1q_f16(&o_vec[dim_block], combined);
                        }
                        
                        for (size_t dim = head_dim_aligned; dim < head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(output_accum_tail[dim - head_dim_aligned] * inv_sum);
                        }
                    } else {
                        for (size_t dim = 0; dim < head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(0.0f);
                        }
                    }
            }
        });
}

void cactus_attention_hybrid_int8_fp16(
    const __fp16* queries,    
    const int8_t* keys_cached, 
    const int8_t* values_cached, 
    const float* k_scales,   
    const float* v_scales, 
    const __fp16* keys_new,  
    const __fp16* values_new, 
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t cache_len,    
    size_t new_len,   
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    size_t position_offset,
    bool is_causal,
    size_t window_size,
    size_t quant_group_size
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;

    const size_t gqa_group_size = num_q_heads / num_kv_heads;  // GQA group size
    const size_t num_quant_groups = (head_dim + quant_group_size - 1) / quant_group_size;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_cached_batch_stride = cache_len * num_kv_heads * head_dim;
    const size_t kv_new_batch_stride = new_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = num_q_heads * head_dim;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            std::vector<float> block_scores(BLOCK_SIZE);
            std::vector<float32x4_t> output_accum_low(head_dim_aligned / VECTOR_WIDTH * 2);
            std::vector<float32x4_t> output_accum_high(head_dim_aligned / VECTOR_WIDTH * 2);

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;

                const size_t kv_head_idx = q_head_idx / gqa_group_size;

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const int8_t* K_cached_base = keys_cached + batch_idx * kv_cached_batch_stride;
                const int8_t* V_cached_base = values_cached + batch_idx * kv_cached_batch_stride;
                const __fp16* K_new_base = keys_new + batch_idx * kv_new_batch_stride;
                const __fp16* V_new_base = values_new + batch_idx * kv_new_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;

                const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;

                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                for (size_t i = 0; i < output_accum_low.size(); ++i) {
                    output_accum_low[i] = vdupq_n_f32(0.0f);
                    output_accum_high[i] = vdupq_n_f32(0.0f);
                }

                const size_t absolute_q_pos = position_offset + q_pos;
                size_t kv_end = is_causal ? std::min(kv_seq_len, absolute_q_pos + 1) : kv_seq_len;

                size_t kv_start = 0;
                if (window_size > 0 && absolute_q_pos > window_size) {
                    kv_start = absolute_q_pos - window_size;
                }

                size_t kv_block_start0 = (kv_start / BLOCK_SIZE) * BLOCK_SIZE;

                for (size_t kv_block_start = kv_block_start0; kv_block_start < kv_end; kv_block_start += BLOCK_SIZE) {
                    const size_t kv_block_end = std::min(kv_block_start + BLOCK_SIZE, kv_end);
                    const size_t block_size = kv_block_end - kv_block_start;

                    float block_max = -std::numeric_limits<float>::infinity();

                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        const size_t kv_pos = kv_block_start + kv_idx;

                        if ((is_causal && kv_pos > absolute_q_pos) || (window_size > 0 && kv_pos < kv_start)) {
                            block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
                            continue;
                        }

                        float32x4_t score_accum_low = vdupq_n_f32(0.0f);
                        float32x4_t score_accum_high = vdupq_n_f32(0.0f);

                        if (kv_pos < cache_len) {
                            const int8_t* k_vec = K_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                            const float* k_scale_base = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;

                            for (size_t quant_group = 0; quant_group < num_quant_groups; quant_group++) {
                                const size_t dim_base = quant_group * quant_group_size;
                                const float k_scale = k_scale_base[quant_group];
                                const float32x4_t k_scale_vec = vdupq_n_f32(k_scale);

                                #pragma unroll
                                for (size_t i = 0; i < 4; i++) {
                                    const size_t dim_block = dim_base + i * VECTOR_WIDTH;
                                    if (dim_block >= head_dim_aligned) break;

                                    float16x8_t q_vec_f16 = vld1q_f16(&q_vec[dim_block]);
                                    float32x4_t q_low = vcvt_f32_f16(vget_low_f16(q_vec_f16));
                                    float32x4_t q_high = vcvt_f32_f16(vget_high_f16(q_vec_f16));

                                    int8x8_t k_vec_i8 = vld1_s8(&k_vec[dim_block]);
                                    int16x8_t k_vec_i16 = vmovl_s8(k_vec_i8);
                                    float32x4_t k_low = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_vec_i16))), k_scale_vec);
                                    float32x4_t k_high = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(k_vec_i16))), k_scale_vec);

                                    score_accum_low = vfmaq_f32(score_accum_low, q_low, k_low);
                                    score_accum_high = vfmaq_f32(score_accum_high, q_high, k_high);
                                }
                            }
                        } else {
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* k_vec = K_new_base + new_pos * kv_seq_stride + kv_head_idx * head_dim;

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t q_vec_f16 = vld1q_f16(&q_vec[dim_block]);
                                float16x8_t k_vec_f16 = vld1q_f16(&k_vec[dim_block]);

                                float32x4_t q_low = vcvt_f32_f16(vget_low_f16(q_vec_f16));
                                float32x4_t q_high = vcvt_f32_f16(vget_high_f16(q_vec_f16));
                                float32x4_t k_low = vcvt_f32_f16(vget_low_f16(k_vec_f16));
                                float32x4_t k_high = vcvt_f32_f16(vget_high_f16(k_vec_f16));

                                score_accum_low = vfmaq_f32(score_accum_low, q_low, k_low);
                                score_accum_high = vfmaq_f32(score_accum_high, q_high, k_high);
                            }
                        }

                        float score = vaddvq_f32(vaddq_f32(score_accum_low, score_accum_high)) * scale;
                        block_scores[kv_idx] = score;
                        block_max = std::max(block_max, score);
                    }

                    if (block_max > -std::numeric_limits<float>::infinity()) {
                        float scale_correction = expf(running_max - block_max);
                        running_sum *= scale_correction;

                        for (size_t i = 0; i < output_accum_low.size() / 2; ++i) {
                            output_accum_low[i] = vmulq_n_f32(output_accum_low[i], scale_correction);
                            output_accum_high[i] = vmulq_n_f32(output_accum_high[i], scale_correction);
                        }
                        running_max = block_max;
                    }

                    float block_sum = 0.0f;
                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        if (block_scores[kv_idx] != -std::numeric_limits<float>::infinity()) {
                            block_scores[kv_idx] = expf(block_scores[kv_idx] - block_max);
                            block_sum += block_scores[kv_idx];
                        } else {
                            block_scores[kv_idx] = 0.0f;
                        }
                    }

                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        const float attn_weight = block_scores[kv_idx];
                        if (attn_weight == 0.0f) continue;

                        const size_t kv_pos = kv_block_start + kv_idx;
                        const float32x4_t weight_vec = vdupq_n_f32(attn_weight);

                        if (kv_pos < cache_len) {
                            const int8_t* v_vec = V_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                            const float* v_scale_base = v_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;

                            for (size_t quant_group = 0; quant_group < num_quant_groups; quant_group++) {
                                const size_t dim_base = quant_group * quant_group_size;
                                const float v_scale = v_scale_base[quant_group];
                                const float32x4_t v_scale_vec = vdupq_n_f32(v_scale);

                                #pragma unroll
                                for (size_t i = 0; i < 4; i++) {
                                    const size_t dim_block = dim_base + i * VECTOR_WIDTH;
                                    if (dim_block >= head_dim_aligned) break;

                                    int8x8_t v_vec_i8 = vld1_s8(&v_vec[dim_block]);
                                    int16x8_t v_vec_i16 = vmovl_s8(v_vec_i8);
                                    float32x4_t v_low = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_vec_i16))), v_scale_vec);
                                    float32x4_t v_high = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_vec_i16))), v_scale_vec);

                                    size_t idx = dim_block / VECTOR_WIDTH;
                                    output_accum_low[idx] = vfmaq_f32(output_accum_low[idx], v_low, weight_vec);
                                    output_accum_high[idx] = vfmaq_f32(output_accum_high[idx], v_high, weight_vec);
                                }
                            }
                        } else {
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* v_vec = V_new_base + new_pos * kv_seq_stride + kv_head_idx * head_dim;

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t v_vec_f16 = vld1q_f16(&v_vec[dim_block]);
                                float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v_vec_f16));
                                float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v_vec_f16));

                                size_t idx = dim_block / VECTOR_WIDTH;
                                output_accum_low[idx] = vfmaq_f32(output_accum_low[idx], v_low, weight_vec);
                                output_accum_high[idx] = vfmaq_f32(output_accum_high[idx], v_high, weight_vec);
                            }
                        }
                    }

                    running_sum += block_sum;
                }

                if (running_sum > 0.0f) {
                    const float inv_sum = 1.0f / running_sum;
                    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

                    for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                        size_t idx = dim_block / VECTOR_WIDTH;
                        float32x4_t final_low = vmulq_f32(output_accum_low[idx], inv_sum_vec);
                        float32x4_t final_high = vmulq_f32(output_accum_high[idx], inv_sum_vec);

                        float16x4_t low_f16 = vcvt_f16_f32(final_low);
                        float16x4_t high_f16 = vcvt_f16_f32(final_high);
                        float16x8_t combined = vcombine_f16(low_f16, high_f16);

                        vst1q_f16(&o_vec[dim_block], combined);
                    }
                } else {
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        o_vec[dim] = static_cast<__fp16>(0.0f);
                    }
                }
            }
        });
}

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

static thread_local RoPECacheF16 rope_cache_f16;

void precompute_rope_tables_f16(size_t seq_len, size_t head_dim, float theta) {
    if (rope_cache_f16.initialized && 
        rope_cache_f16.max_seq_len >= seq_len && 
        rope_cache_f16.head_dim == head_dim && 
        rope_cache_f16.theta == theta) {
        return;
    }
        
    const size_t half_dim = head_dim / 2;
    const size_t table_size = seq_len * half_dim;
    
    rope_cache_f16.cos_table.resize(table_size);
    rope_cache_f16.sin_table.resize(table_size);
        
    for (size_t pos = 0; pos < seq_len; ++pos) {
        const float pos_float = static_cast<float>(pos);
        for (size_t i = 0; i < half_dim; ++i) {
            const float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            const float angle = pos_float * freq;
            
            const size_t idx = pos * half_dim + i;
            rope_cache_f16.cos_table[idx] = static_cast<__fp16>(cosf(angle));
            rope_cache_f16.sin_table[idx] = static_cast<__fp16>(sinf(angle));
        }
    }
    
    rope_cache_f16.max_seq_len = seq_len;
    rope_cache_f16.head_dim = head_dim;
    rope_cache_f16.theta = theta;
    rope_cache_f16.initialized = true;
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
    
    const __fp16* cos_cache = CactusRoPEF16::rope_cache_f16.cos_table.data() + start_pos * half_dim;
    const __fp16* sin_cache = CactusRoPEF16::rope_cache_f16.sin_table.data() + start_pos * half_dim;

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
    
    const __fp16* cos_cache = CactusRoPEF16::rope_cache_f16.cos_table.data() + start_pos * half_rot_dim;
    const __fp16* sin_cache = CactusRoPEF16::rope_cache_f16.sin_table.data() + start_pos * half_rot_dim;

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