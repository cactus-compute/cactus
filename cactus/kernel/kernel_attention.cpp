#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>


void cactus_attention_int8(
    const int8_t* queries,
    const int8_t* keys,
    const int8_t* values,
    int8_t* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    const int8_t* mask,
    float q_scale,
    float k_scale,
    float v_scale,
    float output_scale,
    size_t position_offset,
    size_t window_size,
    bool is_causal
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    constexpr size_t VECTOR_WIDTH = 16;
    constexpr size_t TILE_Q = 4;
    constexpr size_t TILE_K = 8;
    constexpr size_t VECTOR_UNROLL = 2;
    const size_t head_dim_aligned = (head_dim / (VECTOR_WIDTH * VECTOR_UNROLL)) * (VECTOR_WIDTH * VECTOR_UNROLL);
    
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
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;
                const size_t kv_head_idx = q_head_idx / group_size;

                const int8_t* Q_base = queries + batch_idx * q_batch_stride;
                const int8_t* K_base = keys + batch_idx * kv_batch_stride;
                const int8_t* V_base = values + batch_idx * kv_batch_stride;
                int8_t* O_base = output + batch_idx * o_batch_stride;
                const int8_t* M = mask ? (mask + batch_idx * mask_batch_stride) : nullptr;

                for (size_t q_start = q_pos; q_start <= q_pos; q_start += TILE_Q) {
                    const size_t q_end = std::min(q_start + TILE_Q, seq_len);
                    
                    std::vector<float> attention_scores(TILE_Q * kv_seq_len, -std::numeric_limits<float>::infinity());

                    for (size_t q_offset = 0; q_offset < (q_end - q_start); ++q_offset) {
                        const size_t q_pos = q_start + q_offset;
                        const int8_t* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;

                        for (size_t kv_start = 0; kv_start < kv_seq_len; kv_start += TILE_K) {
                            const size_t kv_end = std::min(kv_start + TILE_K, kv_seq_len);
                            
                            std::vector<int32x4_t> accumulators(TILE_K, vdupq_n_s32(0));

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH * VECTOR_UNROLL) {
                                int8x16_t q_vec_low = vld1q_s8(&q_vec[dim_block]);
                                int8x16_t q_vec_high = vld1q_s8(&q_vec[dim_block + VECTOR_WIDTH]);

                                for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                    const size_t kv_pos = kv_start + kv_idx;
                                    const int8_t* k_vec = K_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                                    int8x16_t k_vec_low = vld1q_s8(&k_vec[dim_block]);
                                    int8x16_t k_vec_high = vld1q_s8(&k_vec[dim_block + VECTOR_WIDTH]);

                                    accumulators[kv_idx] = accum_i8mm(accumulators[kv_idx], q_vec_low, k_vec_low);
                                    accumulators[kv_idx] = accum_i8mm(accumulators[kv_idx], q_vec_high, k_vec_high);
                                }
                            }

                            for (size_t dim_block = head_dim_aligned; dim_block < head_dim; dim_block += VECTOR_WIDTH) {
                                size_t remaining = head_dim - dim_block;
                                
                                int8_t q_tmp[VECTOR_WIDTH] = {};
                                if (remaining >= VECTOR_WIDTH) {
                                    memcpy(q_tmp, &q_vec[dim_block], VECTOR_WIDTH);
                                } else {
                                    memcpy(q_tmp, &q_vec[dim_block], remaining);
                                }
                                int8x16_t q_vec_remainder = vld1q_s8(q_tmp);

                                for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                    const size_t kv_pos = kv_start + kv_idx;
                                    const int8_t* k_vec = K_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                                    int8_t k_tmp[VECTOR_WIDTH] = {};
                                    if (remaining >= VECTOR_WIDTH) {
                                        memcpy(k_tmp, &k_vec[dim_block], VECTOR_WIDTH);
                                    } else {
                                        memcpy(k_tmp, &k_vec[dim_block], remaining);
                                    }
                                    int8x16_t k_vec_remainder = vld1q_s8(k_tmp);

                                    accumulators[kv_idx] = accum_i8mm(accumulators[kv_idx], q_vec_remainder, k_vec_remainder);
                                }
                            }

                            for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                const size_t kv_pos = kv_start + kv_idx;
                                int32_t score = vaddvq_s32(accumulators[kv_idx]);
                                float attention_score = static_cast<float>(score) * q_scale * k_scale * scale;

                                size_t absolute_q_pos = position_offset + q_pos;

                                if (is_causal && kv_pos > absolute_q_pos) {
                                    attention_score = -std::numeric_limits<float>::infinity();
                                }
                                else if (window_size > 0 && kv_pos < absolute_q_pos && (absolute_q_pos - kv_pos) > window_size) {
                                    attention_score = -std::numeric_limits<float>::infinity();
                                }
                                else if (M) {
                                    const int8_t mask_val = M[q_pos * kv_seq_len + kv_pos];
                                    if (mask_val == 0) {
                                        attention_score = -std::numeric_limits<float>::infinity();
                                    }
                                }
                                
                                attention_scores[q_offset * kv_seq_len + kv_pos] = attention_score;
                            }
                        }
                    }

                    for (size_t q_offset = 0; q_offset < (q_end - q_start); ++q_offset) {
                        const size_t q_pos = q_start + q_offset;
                        float* scores_row = &attention_scores[q_offset * kv_seq_len];

                        float max_score = -std::numeric_limits<float>::infinity();
                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            max_score = std::max(max_score, scores_row[kv_pos]);
                        }

                        float sum_exp = 0.0f;
                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            if (scores_row[kv_pos] != -std::numeric_limits<float>::infinity()) {
                                scores_row[kv_pos] = expf(scores_row[kv_pos] - max_score);
                                sum_exp += scores_row[kv_pos];
                            } else {
                                scores_row[kv_pos] = 0.0f;
                            }
                        }

                        if (sum_exp > 0.0f) {
                            const float inv_sum = 1.0f / sum_exp;
                            for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                                scores_row[kv_pos] *= inv_sum;
                            }
                        }

                        int8_t* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;
                        std::fill(o_vec, o_vec + head_dim, 0);

                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            const float attn_weight = scores_row[kv_pos];
                            if (attn_weight == 0.0f) continue;

                            const int8_t* v_vec = V_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                            for (size_t dim = 0; dim < head_dim; ++dim) {
                                float weighted_val_fp32 = attn_weight * static_cast<float>(v_vec[dim]) * v_scale;
                                float current_fp32 = static_cast<float>(o_vec[dim]) * output_scale;
                                float result_fp32 = current_fp32 + weighted_val_fp32;
                                
                                int32_t quantized_result = static_cast<int32_t>(result_fp32 / output_scale + (result_fp32 >= 0 ? 0.5f : -0.5f));
                                quantized_result = std::max(-128, std::min(127, quantized_result));
                                o_vec[dim] = static_cast<int8_t>(quantized_result);
                            }
                        }
                    }
                }
            }
        });
}

void cactus_attention_f32(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    const float* mask,
    size_t position_offset,
    size_t window_size,
    bool is_causal
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    constexpr size_t VECTOR_WIDTH = 4;
    constexpr size_t TILE_Q = 4;
    constexpr size_t TILE_K = 8;
    constexpr size_t VECTOR_UNROLL = 2;
    const size_t head_dim_aligned = (head_dim / (VECTOR_WIDTH * VECTOR_UNROLL)) * (VECTOR_WIDTH * VECTOR_UNROLL);
    
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
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;
                const size_t kv_head_idx = q_head_idx / group_size;

                const float* Q_base = queries + batch_idx * q_batch_stride;
                const float* K_base = keys + batch_idx * kv_batch_stride;
                const float* V_base = values + batch_idx * kv_batch_stride;
                float* O_base = output + batch_idx * o_batch_stride;
                const float* M = mask ? (mask + batch_idx * mask_batch_stride) : nullptr;

                for (size_t q_start = q_pos; q_start <= q_pos; q_start += TILE_Q) {
                    const size_t q_end = std::min(q_start + TILE_Q, seq_len);
                    
                    std::vector<float> attention_scores(TILE_Q * kv_seq_len, -std::numeric_limits<float>::infinity());

                    for (size_t q_offset = 0; q_offset < (q_end - q_start); ++q_offset) {
                        const size_t q_pos = q_start + q_offset;
                        const float* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;

                        for (size_t kv_start = 0; kv_start < kv_seq_len; kv_start += TILE_K) {
                            const size_t kv_end = std::min(kv_start + TILE_K, kv_seq_len);
                            
                            std::vector<float32x4_t> accumulators(TILE_K, vdupq_n_f32(0.0f));

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH * VECTOR_UNROLL) {
                                float32x4_t q_vec_low = vld1q_f32(&q_vec[dim_block]);
                                float32x4_t q_vec_high = vld1q_f32(&q_vec[dim_block + VECTOR_WIDTH]);

                                for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                    const size_t kv_pos = kv_start + kv_idx;
                                    const float* k_vec = K_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                                    if (kv_idx + 1 < (kv_end - kv_start)) {
                                        const float* next_k_vec = K_base + (kv_pos + 1) * kv_seq_stride + kv_head_idx * head_dim;
                                        __builtin_prefetch(next_k_vec + dim_block, 0, 1);
                                    }

                                    float32x4_t k_vec_low = vld1q_f32(&k_vec[dim_block]);
                                    float32x4_t k_vec_high = vld1q_f32(&k_vec[dim_block + VECTOR_WIDTH]);

                                    accumulators[kv_idx] = vfmaq_f32(accumulators[kv_idx], q_vec_low, k_vec_low);
                                    accumulators[kv_idx] = vfmaq_f32(accumulators[kv_idx], q_vec_high, k_vec_high);
                                }
                            }

                            for (size_t dim_block = head_dim_aligned; dim_block < head_dim; dim_block += VECTOR_WIDTH) {
                                size_t remaining = head_dim - dim_block;
                                
                                float q_tmp[VECTOR_WIDTH] = {};
                                if (remaining >= VECTOR_WIDTH) {
                                    memcpy(q_tmp, &q_vec[dim_block], VECTOR_WIDTH * sizeof(float));
                                } else {
                                    memcpy(q_tmp, &q_vec[dim_block], remaining * sizeof(float));
                                }
                                float32x4_t q_vec_remainder = vld1q_f32(q_tmp);

                                for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                    const size_t kv_pos = kv_start + kv_idx;
                                    const float* k_vec = K_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                                    float k_tmp[VECTOR_WIDTH] = {};
                                    if (remaining >= VECTOR_WIDTH) {
                                        memcpy(k_tmp, &k_vec[dim_block], VECTOR_WIDTH * sizeof(float));
                                    } else {
                                        memcpy(k_tmp, &k_vec[dim_block], remaining * sizeof(float));
                                    }
                                    float32x4_t k_vec_remainder = vld1q_f32(k_tmp);

                                    accumulators[kv_idx] = vfmaq_f32(accumulators[kv_idx], q_vec_remainder, k_vec_remainder);
                                }
                            }

                            for (size_t kv_idx = 0; kv_idx < (kv_end - kv_start); ++kv_idx) {
                                const size_t kv_pos = kv_start + kv_idx;
                                float score = vaddvq_f32(accumulators[kv_idx]);
                                float attention_score = score * scale;

                                size_t absolute_q_pos = position_offset + q_pos;

                                if (is_causal && kv_pos > absolute_q_pos) {
                                    attention_score = -std::numeric_limits<float>::infinity();
                                }
                                else if (window_size > 0 && kv_pos < absolute_q_pos && (absolute_q_pos - kv_pos) > window_size) {
                                    attention_score = -std::numeric_limits<float>::infinity();
                                }
                                else if (M) {
                                    const float mask_val = M[q_pos * kv_seq_len + kv_pos];
                                    if (mask_val == 0.0f) {
                                        attention_score = -std::numeric_limits<float>::infinity();
                                    }
                                }
                                
                                attention_scores[q_offset * kv_seq_len + kv_pos] = attention_score;
                            }
                        }
                    }

                    for (size_t q_offset = 0; q_offset < (q_end - q_start); ++q_offset) {
                        const size_t q_pos = q_start + q_offset;
                        float* scores_row = &attention_scores[q_offset * kv_seq_len];

                        float max_score = -std::numeric_limits<float>::infinity();
                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            max_score = std::max(max_score, scores_row[kv_pos]);
                        }

                        float sum_exp = 0.0f;
                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            if (scores_row[kv_pos] != -std::numeric_limits<float>::infinity()) {
                                scores_row[kv_pos] = expf(scores_row[kv_pos] - max_score);
                                sum_exp += scores_row[kv_pos];
                            } else {
                                scores_row[kv_pos] = 0.0f;
                            }
                        }

                        if (sum_exp > 0.0f) {
                            const float inv_sum = 1.0f / sum_exp;
                            for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                                scores_row[kv_pos] *= inv_sum;
                            }
                        }
                        

                        float* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;
                        std::fill(o_vec, o_vec + head_dim, 0.0f);

                        for (size_t kv_pos = 0; kv_pos < kv_seq_len; ++kv_pos) {
                            const float attn_weight = scores_row[kv_pos];
                            if (attn_weight == 0.0f) continue;

                            const float* v_vec = V_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;

                            if (kv_pos + 1 < kv_seq_len) {
                                const float* next_v_vec = V_base + (kv_pos + 1) * kv_seq_stride + kv_head_idx * head_dim;
                                __builtin_prefetch(next_v_vec, 0, 1);
                            }

                            size_t dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;
                            float32x4_t weight_vec = vdupq_n_f32(attn_weight);

                            for (size_t dim = 0; dim < dim_aligned; dim += VECTOR_WIDTH) {
                                float32x4_t v_values = vld1q_f32(&v_vec[dim]);
                                float32x4_t o_values = vld1q_f32(&o_vec[dim]);
                                float32x4_t weighted = vmulq_f32(v_values, weight_vec);
                                float32x4_t result = vaddq_f32(o_values, weighted);
                                vst1q_f32(&o_vec[dim], result);
                            }

                            for (size_t dim = dim_aligned; dim < head_dim; ++dim) {
                                o_vec[dim] += v_vec[dim] * attn_weight;
                            }
                        }
                    }
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
                        const size_t kv_block_end = std::min(kv_block_start + BLOCK_SIZE, kv_seq_len);
                        const size_t block_size = kv_block_end - kv_block_start;

                        float block_max = -std::numeric_limits<float>::infinity();

                        if (!is_decode && is_causal && kv_block_start > absolute_q_pos) {
                            for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                                block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
                            }
                            continue; 
                        }

                        for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                            const size_t kv_pos = kv_block_start + kv_idx;

                            if (!is_decode && is_causal && kv_pos > absolute_q_pos) {
                                block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
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
                                score = -std::numeric_limits<float>::infinity();
                            }
                            else if (window_size > 0 && kv_pos < absolute_q_pos && (absolute_q_pos - kv_pos) > window_size) {
                                score = -std::numeric_limits<float>::infinity();
                            }
                            else if (M && static_cast<float>(M[q_pos * kv_seq_len + kv_pos]) == 0.0f) {
                                score = -std::numeric_limits<float>::infinity();
                            }
                            
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
                        const size_t vec_size = (block_size / 4) * 4;

                        for (size_t kv_idx = 0; kv_idx < vec_size; kv_idx += 4) {
                            float32x4_t scores = vld1q_f32(&block_scores[kv_idx]);
                            uint32x4_t inf_mask = vceqq_f32(scores, vdupq_n_f32(-std::numeric_limits<float>::infinity()));

                            float32x4_t x = vsubq_f32(scores, vdupq_n_f32(block_max));
                            x = vmulq_n_f32(x, 1.442695f); 

                            int32x4_t xi = vcvtq_s32_f32(x);
                            float32x4_t xf = vsubq_f32(x, vcvtq_f32_s32(xi));

                            float32x4_t y = vfmaq_n_f32(vdupq_n_f32(1.0f), xf, 0.6931472f);
                            y = vfmaq_f32(y, vmulq_f32(xf, xf), vdupq_n_f32(0.2402265f));

                            xi = vaddq_s32(xi, vdupq_n_s32(127));
                            xi = vshlq_n_s32(xi, 23);
                            y = vmulq_f32(y, vreinterpretq_f32_s32(xi));

                            y = vbslq_f32(inf_mask, vdupq_n_f32(0.0f), y);

                            vst1q_f32(&block_scores[kv_idx], y);
                            block_sum += vaddvq_f32(y);
                        }

                        for (size_t kv_idx = vec_size; kv_idx < block_size; ++kv_idx) {
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
                        
                        for (size_t dim = head_dim_aligned; dim < head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(0.0f);
                        }
                    } else {
                        for (size_t dim = 0; dim < head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(0.0f);
                        }
                    }
            }
        });
}


void cactus_rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_size,
    size_t dims,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 4;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t TILE_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    
    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input + b * dims;
        float* output_row = output + b * dims;
        
        float32x4_t sum_squares_vec[UNROLL_FACTOR];
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            sum_squares_vec[u] = vdupq_n_f32(0.0f);
        }
        
        size_t i = 0;
        const size_t tile_end = (dims >= TILE_SIZE) ? dims - TILE_SIZE + 1 : 0;
        
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float32x4_t input_vec = vld1q_f32(&input_row[i + u * SIMD_WIDTH]);
                sum_squares_vec[u] = vfmaq_f32(sum_squares_vec[u], input_vec, input_vec);
            }
        }
        
        const size_t simd_end = (dims >= SIMD_WIDTH) ? dims - SIMD_WIDTH + 1 : 0;
        for (; i < simd_end; i += SIMD_WIDTH) {
            float32x4_t input_vec = vld1q_f32(&input_row[i]);
            sum_squares_vec[0] = vfmaq_f32(sum_squares_vec[0], input_vec, input_vec);
        }
        
        float32x4_t total_sum = sum_squares_vec[0];
        for (size_t u = 1; u < UNROLL_FACTOR; u++) {
            total_sum = vaddq_f32(total_sum, sum_squares_vec[u]);
        }
        float sum_squares = vaddvq_f32(total_sum);
        
        for (; i < dims; ++i) {
            float val = input_row[i];
            sum_squares += val * val;
        }
        
        float rms = sqrtf(sum_squares / static_cast<float>(dims) + eps);
        float inv_rms = 1.0f / rms;
        float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
        
        i = 0;
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float32x4_t input_vec = vld1q_f32(&input_row[i + u * SIMD_WIDTH]);
                float32x4_t weight_vec = vld1q_f32(&weight[i + u * SIMD_WIDTH]);
                float32x4_t norm_vec = vmulq_f32(vmulq_f32(input_vec, inv_rms_vec), weight_vec);
                vst1q_f32(&output_row[i + u * SIMD_WIDTH], norm_vec);
            }
        }
        
        for (; i < simd_end; i += SIMD_WIDTH) {
            float32x4_t input_vec = vld1q_f32(&input_row[i]);
            float32x4_t weight_vec = vld1q_f32(&weight[i]);
            float32x4_t norm_vec = vmulq_f32(vmulq_f32(input_vec, inv_rms_vec), weight_vec);
            vst1q_f32(&output_row[i], norm_vec);
        }
        
        for (; i < dims; ++i) {
            output_row[i] = input_row[i] * inv_rms * weight[i];
        }
    }
}

void cactus_rms_norm_i8_f32(
    const int8_t* input,
    const float* weight,
    float* output,
    size_t batch_size,
    size_t dims,
    float eps,
    float input_scale
) {
    constexpr size_t SIMD_WIDTH = 4;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t TILE_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    
    const float32x4_t input_scale_vec = vdupq_n_f32(input_scale);
    
    for (size_t b = 0; b < batch_size; ++b) {
        const int8_t* input_row = input + b * dims;
        float* output_row = output + b * dims;
        
        float32x4_t sum_squares_vec[UNROLL_FACTOR];
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            sum_squares_vec[u] = vdupq_n_f32(0.0f);
        }
        
        size_t i = 0;
        const size_t tile_end = (dims >= TILE_SIZE) ? dims - TILE_SIZE + 1 : 0;
        
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                int8x8_t input_i8 = vld1_s8(&input_row[i + u * SIMD_WIDTH]);
                int16x4_t input_i16 = vget_low_s16(vmovl_s8(input_i8));
                int32x4_t input_i32 = vmovl_s16(input_i16);
                float32x4_t input_f32 = vmulq_f32(vcvtq_f32_s32(input_i32), input_scale_vec);
                sum_squares_vec[u] = vfmaq_f32(sum_squares_vec[u], input_f32, input_f32);
            }
        }
        
        const size_t simd_end = (dims >= SIMD_WIDTH) ? dims - SIMD_WIDTH + 1 : 0;
        for (; i < simd_end; i += SIMD_WIDTH) {
            int8x8_t input_i8 = vld1_s8(&input_row[i]);
            int16x4_t input_i16 = vget_low_s16(vmovl_s8(input_i8));
            int32x4_t input_i32 = vmovl_s16(input_i16);
            float32x4_t input_f32 = vmulq_f32(vcvtq_f32_s32(input_i32), input_scale_vec);
            sum_squares_vec[0] = vfmaq_f32(sum_squares_vec[0], input_f32, input_f32);
        }
        
        float32x4_t total_sum = sum_squares_vec[0];
        for (size_t u = 1; u < UNROLL_FACTOR; u++) {
            total_sum = vaddq_f32(total_sum, sum_squares_vec[u]);
        }
        float sum_squares = vaddvq_f32(total_sum);
        
        for (; i < dims; ++i) {
            float val = static_cast<float>(input_row[i]) * input_scale;
            sum_squares += val * val;
        }
        
        float rms = sqrtf(sum_squares / static_cast<float>(dims) + eps);
        float inv_rms = 1.0f / rms;
        float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
        
        i = 0;
        for (; i < tile_end; i += TILE_SIZE) {
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                int8x8_t input_i8 = vld1_s8(&input_row[i + u * SIMD_WIDTH]);
                int16x4_t input_i16 = vget_low_s16(vmovl_s8(input_i8));
                int32x4_t input_i32 = vmovl_s16(input_i16);
                float32x4_t input_f32 = vmulq_f32(vcvtq_f32_s32(input_i32), input_scale_vec);
                
                float32x4_t weight_vec = vld1q_f32(&weight[i + u * SIMD_WIDTH]);
                float32x4_t norm_f32 = vmulq_f32(vmulq_f32(input_f32, inv_rms_vec), weight_vec);
                
                vst1q_f32(&output_row[i + u * SIMD_WIDTH], norm_f32);
            }
        }
        
        for (; i < simd_end; i += SIMD_WIDTH) {
            int8x8_t input_i8 = vld1_s8(&input_row[i]);
            int16x4_t input_i16 = vget_low_s16(vmovl_s8(input_i8));
            int32x4_t input_i32 = vmovl_s16(input_i16);
            float32x4_t input_f32 = vmulq_f32(vcvtq_f32_s32(input_i32), input_scale_vec);
            
            float32x4_t weight_vec = vld1q_f32(&weight[i]);
            float32x4_t norm_f32 = vmulq_f32(vmulq_f32(input_f32, inv_rms_vec), weight_vec);
            
            vst1q_f32(&output_row[i], norm_f32);
        }
        
        for (; i < dims; ++i) {
            float val = static_cast<float>(input_row[i]) * input_scale;
            output_row[i] = (val * inv_rms) * weight[i];
        }
    }
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

namespace CactusRoPE {

struct RoPECache {
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    size_t max_seq_len;
    size_t head_dim;
    float theta;
    bool initialized;
    
    RoPECache() : max_seq_len(0), head_dim(0), theta(0.0f), initialized(false) {}
};

static thread_local RoPECache rope_cache;

void precompute_rope_tables(size_t seq_len, size_t head_dim, float theta) {
    if (rope_cache.initialized && 
        rope_cache.max_seq_len >= seq_len && 
        rope_cache.head_dim == head_dim && 
        rope_cache.theta == theta) {
        return;
    }
        
    const size_t half_dim = head_dim / 2;
    const size_t table_size = seq_len * half_dim;
    
    rope_cache.cos_table.resize(table_size);
    rope_cache.sin_table.resize(table_size);
        
    for (size_t pos = 0; pos < seq_len; ++pos) {
        const float pos_float = static_cast<float>(pos);
        for (size_t dim_idx = 0; dim_idx < half_dim; ++dim_idx) {
            const float inv_freq = 1.0f / powf(theta, 2.0f * static_cast<float>(dim_idx) / static_cast<float>(head_dim));
            const float angle = pos_float * inv_freq;
            const size_t cache_idx = pos * half_dim + dim_idx;
            rope_cache.cos_table[cache_idx] = cosf(angle);
            rope_cache.sin_table[cache_idx] = sinf(angle);
        }
    }
    
    rope_cache.max_seq_len = seq_len;
    rope_cache.head_dim = head_dim;
    rope_cache.theta = theta;
    rope_cache.initialized = true;
}

void kernel_rope_neon_optimized_head(
    const float* input_head,
    float* output_head,
    const float* cos_cache,
    const float* sin_cache,
    size_t seq_len,
    size_t head_dim,
    size_t pairs_per_head
) {
    constexpr size_t SIMD_WIDTH = 4;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t VECTORIZED_PAIRS = SIMD_WIDTH * UNROLL_FACTOR;
    
    const size_t pairs_vectorized = (pairs_per_head / VECTORIZED_PAIRS) * VECTORIZED_PAIRS;
    
    for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        const float* x = input_head + seq_idx * head_dim;
        float* y = output_head + seq_idx * head_dim;
        const size_t cache_base = seq_idx * pairs_per_head;
        
        size_t pair_idx = 0;
        
        for (; pair_idx < pairs_vectorized; pair_idx += VECTORIZED_PAIRS) {
            const size_t cache_offset = cache_base + pair_idx;
            
            float32x4_t cos_vec[UNROLL_FACTOR];
            float32x4_t sin_vec[UNROLL_FACTOR];
            float32x4x2_t input_pairs[UNROLL_FACTOR];
            
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                cos_vec[u] = vld1q_f32(&cos_cache[cache_offset + u * SIMD_WIDTH]);
                sin_vec[u] = vld1q_f32(&sin_cache[cache_offset + u * SIMD_WIDTH]);
                input_pairs[u] = vld2q_f32(&x[2 * (pair_idx + u * SIMD_WIDTH)]);
            }
            
            for (size_t u = 0; u < UNROLL_FACTOR; u++) {
                float32x4_t x1_vec = input_pairs[u].val[0];
                float32x4_t x2_vec = input_pairs[u].val[1];
                
                float32x4_t y1_vec = vfmsq_f32(vmulq_f32(x1_vec, cos_vec[u]), x2_vec, sin_vec[u]);
                float32x4_t y2_vec = vfmaq_f32(vmulq_f32(x2_vec, cos_vec[u]), x1_vec, sin_vec[u]);
                
                float32x4x2_t output_pairs;
                output_pairs.val[0] = y1_vec;
                output_pairs.val[1] = y2_vec;
                vst2q_f32(&y[2 * (pair_idx + u * SIMD_WIDTH)], output_pairs);
            }
        }
        
        for (; pair_idx < pairs_per_head; ++pair_idx) {
            const size_t cache_offset = cache_base + pair_idx;
            const float cos_val = cos_cache[cache_offset];
            const float sin_val = sin_cache[cache_offset];
            
            const float x1 = x[2 * pair_idx];
            const float x2 = x[2 * pair_idx + 1];
            
            y[2 * pair_idx] = x1 * cos_val - x2 * sin_val;
            y[2 * pair_idx + 1] = x1 * sin_val + x2 * cos_val;
        }
    }
}

}

void cactus_rope_f32(
    const float* input,
    float* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t start_pos,
    float theta
) {
    const size_t half_dim = head_dim / 2;
    
    CactusRoPE::precompute_rope_tables(seq_len + start_pos, head_dim, theta);
    
    const float* cos_cache = CactusRoPE::rope_cache.cos_table.data() + start_pos * half_dim;
    const float* sin_cache = CactusRoPE::rope_cache.sin_table.data() + start_pos * half_dim;

    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t batch_idx = idx / seq_len;
                const size_t seq_idx = idx % seq_len;
                
                for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    const size_t offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
                    const float* input_ptr = input + offset;
                    float* output_ptr = output + offset;
                    
                    const float* cos_ptr = cos_cache + seq_idx * half_dim;
                    const float* sin_ptr = sin_cache + seq_idx * half_dim;
                    
                    
                    for (size_t i = 0; i < half_dim; ++i) {
                        const float cos_val = cos_ptr[i];
                        const float sin_val = sin_ptr[i];
                        
                        const float x_first_half = input_ptr[i];
                        const float x_second_half = input_ptr[i + half_dim];
                        
                        output_ptr[i] = x_first_half * cos_val - x_second_half * sin_val;
                        
                        output_ptr[i + half_dim] = x_second_half * cos_val + x_first_half * sin_val;
                    }
                }
            }
        });
}

void cactus_rope_i8_f32_i8(
    const int8_t* input,
    int8_t* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t start_pos,
    float theta,
    float input_scale,
    float output_scale
) {
    const size_t half_dim = head_dim / 2;
    
    CactusRoPE::precompute_rope_tables(seq_len + start_pos, head_dim, theta);
    
    const float* cos_cache = CactusRoPE::rope_cache.cos_table.data() + start_pos * half_dim;
    const float* sin_cache = CactusRoPE::rope_cache.sin_table.data() + start_pos * half_dim;

    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t batch_idx = idx / seq_len;
                const size_t seq_idx = idx % seq_len;
                
                for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    const size_t offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
                    const int8_t* x = input + offset;
                    int8_t* y = output + offset;
                    const size_t cache_base = seq_idx * half_dim;
                    
                    for (size_t i = 0; i < half_dim; ++i) {
                        const float cos_val = cos_cache[cache_base + i];
                        const float sin_val = sin_cache[cache_base + i];
                        
                        const float x_first_half = static_cast<float>(x[i]) * input_scale;
                        const float x_second_half = static_cast<float>(x[i + half_dim]) * input_scale;
                        
                        const float y0 = x_first_half * cos_val - x_second_half * sin_val;
                        const float y1 = x_second_half * cos_val + x_first_half * sin_val;
                        
                        const float scaled_y0 = y0 / output_scale;
                        const float scaled_y1 = y1 / output_scale;
                        
                        y[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(scaled_y0))));
                        y[i + half_dim] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(scaled_y1))));
                    }
                }
            }
        });
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