#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __APPLE__
static void cactus_attention_f16_accelerate(
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
    size_t position_offset,
    bool is_causal
) {
    constexpr size_t BLOCK_SIZE = 64;

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = q_batch_stride;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = q_seq_stride;

    static constexpr CactusThreading::ParallelConfig ATTENTION_BATCHED{1, 1};
    CactusThreading::parallel_for(batch_size * num_q_heads, ATTENTION_BATCHED,
        [&](size_t start, size_t end) {

        std::vector<float> Q_f32(seq_len * head_dim);
        std::vector<float> K_f32(BLOCK_SIZE * head_dim);
        std::vector<float> V_f32(BLOCK_SIZE * head_dim);
        std::vector<float> scores(seq_len * BLOCK_SIZE);
        std::vector<float> acc(seq_len * head_dim);
        std::vector<float> row_max(seq_len);
        std::vector<float> row_sum(seq_len);

        for (size_t work = start; work < end; ++work) {
            const size_t batch = work / num_q_heads;
            const size_t q_head = work % num_q_heads;
            const size_t kv_head = q_head / group_size;

            for (size_t q = 0; q < seq_len; ++q) {
                const __fp16* q_src = queries + batch*q_batch_stride + q*q_seq_stride + q_head*head_dim;
                float* q_dst = Q_f32.data() + q * head_dim;
                for (size_t d = 0; d < head_dim; d += 8) {
                    float16x8_t v = vld1q_f16(q_src + d);
                    vst1q_f32(q_dst + d,     vcvt_f32_f16(vget_low_f16(v)));
                    vst1q_f32(q_dst + d + 4, vcvt_f32_f16(vget_high_f16(v)));
                }
            }

            std::fill(row_max.begin(), row_max.begin() + seq_len, -INFINITY);
            std::fill(row_sum.begin(), row_sum.begin() + seq_len, 0.0f);
            memset(acc.data(), 0, seq_len * head_dim * sizeof(float));

            for (size_t kv0 = 0; kv0 < kv_seq_len; kv0 += BLOCK_SIZE) {
                const size_t block_len = std::min(BLOCK_SIZE, kv_seq_len - kv0);

                size_t q_start = 0;
                size_t active_rows = seq_len;
                if (is_causal) {
                    if (kv0 > position_offset) {
                        q_start = kv0 - position_offset;
                    }
                    if (q_start >= seq_len) continue;
                    active_rows = seq_len - q_start;
                }

                for (size_t i = 0; i < block_len; ++i) {
                    const __fp16* k_src = keys + batch*kv_batch_stride + (kv0+i)*kv_seq_stride + kv_head*head_dim;
                    float* k_dst = K_f32.data() + i * head_dim;
                    for (size_t d = 0; d < head_dim; d += 8) {
                        float16x8_t v = vld1q_f16(k_src + d);
                        vst1q_f32(k_dst + d,     vcvt_f32_f16(vget_low_f16(v)));
                        vst1q_f32(k_dst + d + 4, vcvt_f32_f16(vget_high_f16(v)));
                    }
                }

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)active_rows, (int)block_len, (int)head_dim,
                            scale,
                            Q_f32.data() + q_start * head_dim, (int)head_dim,
                            K_f32.data(), (int)head_dim,
                            0.0f,
                            scores.data(), (int)block_len);

                for (size_t r = 0; r < active_rows; ++r) {
                    const size_t q_pos = q_start + r;
                    const size_t abs_q = position_offset + q_pos;
                    float* s_row = scores.data() + r * block_len;

                    size_t valid_len = block_len;
                    if (is_causal) {
                        if (abs_q < kv0) {
                            memset(s_row, 0, block_len * sizeof(float));
                            continue;
                        }
                        valid_len = std::min(block_len, abs_q - kv0 + 1);
                    }

                    float32x4_t vmax = vdupq_n_f32(-INFINITY);
                    size_t j = 0;
                    for (; j + 4 <= valid_len; j += 4) {
                        vmax = vmaxq_f32(vmax, vld1q_f32(s_row + j));
                    }
                    float block_max = vmaxvq_f32(vmax);
                    for (; j < valid_len; ++j) {
                        block_max = std::max(block_max, s_row[j]);
                    }

                    float prev_max = row_max[q_pos];
                    float new_max = std::max(prev_max, block_max);
                    float scale_old = expf(prev_max - new_max);

                    if (prev_max != -INFINITY) {
                        float* acc_row = acc.data() + q_pos * head_dim;
                        float32x4_t sv = vdupq_n_f32(scale_old);
                        for (size_t d = 0; d < head_dim; d += 4) {
                            float32x4_t a = vld1q_f32(acc_row + d);
                            vst1q_f32(acc_row + d, vmulq_f32(a, sv));
                        }
                    }
                    row_sum[q_pos] = row_sum[q_pos] * scale_old;
                    row_max[q_pos] = new_max;

                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    float32x4_t vnmax = vdupq_n_f32(new_max);
                    float32x4_t log2e = vdupq_n_f32(1.442695f);
                    j = 0;
                    for (; j + 4 <= valid_len; j += 4) {
                        float32x4_t x = vmulq_f32(vsubq_f32(vld1q_f32(s_row + j), vnmax), log2e);
                        float32x4_t x_floor = vrndmq_f32(x);
                        int32x4_t xi = vcvtq_s32_f32(x_floor);
                        float32x4_t xf = vsubq_f32(x, x_floor);
                        float32x4_t y = vfmaq_n_f32(vdupq_n_f32(1.0f), xf, 0.6931472f);
                        y = vfmaq_f32(y, vmulq_f32(xf, xf), vdupq_n_f32(0.2402265f));
                        xi = vshlq_n_s32(vaddq_s32(xi, vdupq_n_s32(127)), 23);
                        y = vmulq_f32(y, vreinterpretq_f32_s32(xi));
                        uint32x4_t underflow = vcltq_f32(x, vdupq_n_f32(-126.0f));
                        y = vbslq_f32(underflow, vdupq_n_f32(0.0f), y);
                        vst1q_f32(s_row + j, y);
                        vsum = vaddq_f32(vsum, y);
                    }
                    float block_sum = vaddvq_f32(vsum);
                    for (; j < valid_len; ++j) {
                        s_row[j] = expf(s_row[j] - new_max);
                        block_sum += s_row[j];
                    }
                    if (valid_len < block_len) {
                        memset(s_row + valid_len, 0, (block_len - valid_len) * sizeof(float));
                    }
                    row_sum[q_pos] += block_sum;
                }

                for (size_t i = 0; i < block_len; ++i) {
                    const __fp16* v_src = values + batch*kv_batch_stride + (kv0+i)*kv_seq_stride + kv_head*head_dim;
                    float* v_dst = V_f32.data() + i * head_dim;
                    for (size_t d = 0; d < head_dim; d += 8) {
                        float16x8_t v = vld1q_f16(v_src + d);
                        vst1q_f32(v_dst + d,     vcvt_f32_f16(vget_low_f16(v)));
                        vst1q_f32(v_dst + d + 4, vcvt_f32_f16(vget_high_f16(v)));
                    }
                }

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)active_rows, (int)head_dim, (int)block_len,
                            1.0f,
                            scores.data(), (int)block_len,
                            V_f32.data(), (int)head_dim,
                            1.0f,
                            acc.data() + q_start * head_dim, (int)head_dim);
            }

            for (size_t q = 0; q < seq_len; ++q) {
                __fp16* o = output + batch*o_batch_stride + q*o_seq_stride + q_head*head_dim;
                float sum = row_sum[q];
                if (sum == 0.0f) {
                    memset(o, 0, head_dim * sizeof(__fp16));
                    continue;
                }
                float inv = 1.0f / sum;
                float32x4_t invv = vdupq_n_f32(inv);
                float* acc_row = acc.data() + q * head_dim;
                for (size_t d = 0; d < head_dim; d += 8) {
                    float32x4_t a0 = vmulq_f32(vld1q_f32(acc_row + d), invv);
                    float32x4_t a1 = vmulq_f32(vld1q_f32(acc_row + d + 4), invv);
                    vst1q_f16(o + d, vcombine_f16(vcvt_f16_f32(a0), vcvt_f16_f32(a1)));
                }
            }
        }
    });
}
#endif

static inline void cactus_attention_f16_fast(
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
    size_t position_offset,
    bool is_causal
) {
    constexpr size_t BLOCK_SIZE = 32;
    constexpr size_t MAX_ACCUM_SLOTS = 256 / 8;
    const size_t nblocks = head_dim / 8;

#ifdef __APPLE__
    if (seq_len >= 64) {
        cactus_attention_f16_accelerate(
            queries, keys, values, output,
            batch_size, seq_len, kv_seq_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, position_offset, is_causal
        );
        return;
    }
#endif

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = q_batch_stride;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = q_seq_stride;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start, size_t end) {

        float block_scores[BLOCK_SIZE];
        float16x8_t vacc[MAX_ACCUM_SLOTS];

        for (size_t work = start; work < end; ++work) {
            const size_t batch = work / (num_q_heads * seq_len);
            const size_t rem = work % (num_q_heads * seq_len);
            const size_t q_head = rem / seq_len;
            const size_t q_pos = rem % seq_len;
            const size_t kv_head = q_head / group_size;

            const __fp16* q = queries + batch*q_batch_stride + q_pos*q_seq_stride + q_head*head_dim;
            __fp16* o = output + batch*o_batch_stride + q_pos*o_seq_stride + q_head*head_dim;

            for (size_t i = 0; i < nblocks; i++)
                vacc[i] = vdupq_n_f16((__fp16)0);

            float running_max = -INFINITY;
            float running_sum = 0.f;

            const size_t abs_q = position_offset + q_pos;
            size_t kv_end = is_causal ? std::min(kv_seq_len, abs_q + 1) : kv_seq_len;

            for (size_t kv0 = 0; kv0 < kv_end; kv0 += BLOCK_SIZE) {
                const size_t kv1 = std::min(kv0 + BLOCK_SIZE, kv_end);
                float block_max = -INFINITY;

                for (size_t i = kv0; i < kv1; i++) {
                    float16x8_t s = vdupq_n_f16((__fp16)0);
                    const __fp16* k = keys + batch*kv_batch_stride + i*kv_seq_stride + kv_head*head_dim;

                    for (size_t d = 0; d < nblocks; d++) {
                        float16x8_t qv = vld1q_f16(q + d*8);
                        float16x8_t kv = vld1q_f16(k + d*8);
                        s = vfmaq_f16(s, qv, kv);
                    }

                    float score = (vaddvq_f32(vcvt_f32_f16(vget_low_f16(s))) +
                                   vaddvq_f32(vcvt_f32_f16(vget_high_f16(s)))) * scale;
                    block_scores[i - kv0] = score;
                    block_max = std::max(block_max, score);
                }

                float scale_corr = expf(running_max - block_max);
                running_sum *= scale_corr;

                float16x8_t corr16 = vdupq_n_f16((__fp16)scale_corr);
                for (size_t d = 0; d < nblocks; d++)
                    vacc[d] = vmulq_f16(vacc[d], corr16);

                float block_sum = 0.f;
                for (size_t i = 0; i < kv1 - kv0; i++) {
                    block_scores[i] = expf(block_scores[i] - block_max);
                    block_sum += block_scores[i];
                }

                for (size_t i = 0; i < kv1 - kv0; i++) {
                    float w = block_scores[i];
                    if (w == 0.f) continue;

                    const __fp16* v = values + batch*kv_batch_stride + (kv0+i)*kv_seq_stride + kv_head*head_dim;
                    float16x8_t wv = vdupq_n_f16((__fp16)w);

                    for (size_t d = 0; d < nblocks; d++) {
                        float16x8_t vv = vld1q_f16(v + d*8);
                        vacc[d] = vfmaq_f16(vacc[d], vv, wv);
                    }
                }

                running_sum += block_sum;
                running_max = block_max;
            }

            if (running_sum == 0.f) {
                memset(o, 0, head_dim * sizeof(__fp16));
                continue;
            }

            float inv = 1.f / running_sum;
            float32x4_t invv = vdupq_n_f32(inv);

            for (size_t d = 0; d < nblocks; d++) {
                float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(vacc[d])), invv);
                float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(vacc[d])), invv);
                vst1q_f16(o + d*8, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
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
    bool is_causal,
    bool mask_is_additive,
    bool mask_per_head
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    if (head_dim % 8 == 0 && mask == nullptr && window_size == 0) {
        cactus_attention_f16_fast(
            queries, keys, values, output,
            batch_size, seq_len, kv_seq_len,
            num_q_heads, num_kv_heads, head_dim,
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
    const size_t mask_batch_stride = mask
        ? (mask_per_head ? (num_q_heads * seq_len * kv_seq_len) : (seq_len * kv_seq_len))
        : 0;

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
                            else if (M) {
                                const size_t mask_index = mask_per_head
                                    ? ((q_head_idx * seq_len + q_pos) * kv_seq_len + kv_pos)
                                    : (q_pos * kv_seq_len + kv_pos);
                                const float mask_value = static_cast<float>(M[mask_index]);
                                if (mask_is_additive) {
                                    if (!std::isfinite(mask_value)) {
                                        score = NEG_INF;
                                    } else {
                                        score += mask_value;
                                    }
                                } else if (mask_value == 0.0f) {
                                    score = NEG_INF;
                                }
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

static void cactus_attention_hybrid_int8_fp16_decode(
    const __fp16* queries,
    const int8_t* keys_cached,
    const int8_t* values_cached,
    const float* k_scales,
    const float* v_scales,
    const __fp16* keys_new,
    const __fp16* values_new,
    __fp16* output,
    size_t batch_size,
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
    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    constexpr size_t MAX_HEAD_DIM = 256;
    constexpr size_t MAX_ACCUM_SLOTS = MAX_HEAD_DIM / VECTOR_WIDTH;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;
    const size_t num_accum_slots = head_dim_aligned / VECTOR_WIDTH;

    const size_t gqa_group_size = num_q_heads / num_kv_heads;
    const size_t num_quant_groups = (head_dim + quant_group_size - 1) / quant_group_size;

    const size_t q_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_stride = num_q_heads * head_dim;

    CactusThreading::parallel_for(batch_size * num_q_heads, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            float block_scores[BLOCK_SIZE];
            float32x4_t output_accum_low[MAX_ACCUM_SLOTS];
            float32x4_t output_accum_high[MAX_ACCUM_SLOTS];
            float16x8_t block_accum[MAX_ACCUM_SLOTS];

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / num_q_heads;
                const size_t q_head_idx = work_idx % num_q_heads;
                const size_t kv_head_idx = q_head_idx / gqa_group_size;

                const __fp16* q_vec = queries + batch_idx * q_stride + q_head_idx * head_dim;
                const int8_t* K_cached_base = keys_cached + batch_idx * cache_len * kv_seq_stride;
                const int8_t* V_cached_base = values_cached + batch_idx * cache_len * kv_seq_stride;
                const __fp16* K_new_base = keys_new + batch_idx * new_len * kv_seq_stride;
                const __fp16* V_new_base = values_new + batch_idx * new_len * kv_seq_stride;
                __fp16* o_vec = output + batch_idx * o_stride + q_head_idx * head_dim;

                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                for (size_t i = 0; i < num_accum_slots; ++i) {
                    output_accum_low[i] = vdupq_n_f32(0.0f);
                    output_accum_high[i] = vdupq_n_f32(0.0f);
                }

                const size_t absolute_q_pos = position_offset;
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

                    // K scoring: interleaved 2-position + deferred scale

                    size_t kv_idx = 0;
                    for (; kv_idx + 1 < block_size; kv_idx += 2) {
                        const size_t kv_pos_a = kv_block_start + kv_idx;
                        const size_t kv_pos_b = kv_pos_a + 1;

                        bool skip_a = (is_causal && kv_pos_a > absolute_q_pos) || (window_size > 0 && kv_pos_a < kv_start);
                        bool skip_b = (is_causal && kv_pos_b > absolute_q_pos) || (window_size > 0 && kv_pos_b < kv_start);

                        if (skip_a && skip_b) {
                            block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
                            block_scores[kv_idx + 1] = -std::numeric_limits<float>::infinity();
                            continue;
                        }

                        float score_a = 0.0f, score_b = 0.0f;

                        if (kv_pos_a < cache_len && kv_pos_b < cache_len && !skip_a && !skip_b) {
                            const int8_t* k_a = K_cached_base + kv_pos_a * kv_seq_stride + kv_head_idx * head_dim;
                            const int8_t* k_b = K_cached_base + kv_pos_b * kv_seq_stride + kv_head_idx * head_dim;
                            const float* ks_a = k_scales + (kv_pos_a * num_kv_heads + kv_head_idx) * num_quant_groups;
                            const float* ks_b = k_scales + (kv_pos_b * num_kv_heads + kv_head_idx) * num_quant_groups;

                            for (size_t qg = 0; qg < num_quant_groups; qg++) {
                                const size_t dim_base = qg * quant_group_size;

                                float32x4_t raw_lo_a = vdupq_n_f32(0.0f), raw_hi_a = vdupq_n_f32(0.0f);
                                float32x4_t raw_lo_b = vdupq_n_f32(0.0f), raw_hi_b = vdupq_n_f32(0.0f);

                                #pragma unroll
                                for (size_t i = 0; i < 4; i++) {
                                    const size_t d = dim_base + i * VECTOR_WIDTH;
                                    if (d >= head_dim_aligned) break;

                                    float16x8_t qf = vld1q_f16(&q_vec[d]);
                                    float32x4_t q_lo = vcvt_f32_f16(vget_low_f16(qf));
                                    float32x4_t q_hi = vcvt_f32_f16(vget_high_f16(qf));

                                    int16x8_t ka_i16 = vmovl_s8(vld1_s8(&k_a[d]));
                                    int16x8_t kb_i16 = vmovl_s8(vld1_s8(&k_b[d]));

                                    raw_lo_a = vfmaq_f32(raw_lo_a, q_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ka_i16))));
                                    raw_hi_a = vfmaq_f32(raw_hi_a, q_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ka_i16))));
                                    raw_lo_b = vfmaq_f32(raw_lo_b, q_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(kb_i16))));
                                    raw_hi_b = vfmaq_f32(raw_hi_b, q_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(kb_i16))));
                                }

                                score_a += ks_a[qg] * vaddvq_f32(vaddq_f32(raw_lo_a, raw_hi_a));
                                score_b += ks_b[qg] * vaddvq_f32(vaddq_f32(raw_lo_b, raw_hi_b));
                            }
                            score_a *= scale; score_b *= scale;
                        } else {
                            auto score_one = [&](size_t kv_pos) -> float {
                                float s = 0.0f;
                                if (kv_pos < cache_len) {
                                    const int8_t* kv = K_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                                    const float* ks = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;
                                    for (size_t qg = 0; qg < num_quant_groups; qg++) {
                                        float32x4_t rlo = vdupq_n_f32(0.0f), rhi = vdupq_n_f32(0.0f);
                                        const size_t db = qg * quant_group_size;
                                        for (size_t ii = 0; ii < 4; ii++) {
                                            const size_t d = db + ii * VECTOR_WIDTH;
                                            if (d >= head_dim_aligned) break;
                                            float16x8_t qf = vld1q_f16(&q_vec[d]);
                                            int16x8_t ki16 = vmovl_s8(vld1_s8(&kv[d]));
                                            rlo = vfmaq_f32(rlo, vcvt_f32_f16(vget_low_f16(qf)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(ki16))));
                                            rhi = vfmaq_f32(rhi, vcvt_f32_f16(vget_high_f16(qf)), vcvtq_f32_s32(vmovl_s16(vget_high_s16(ki16))));
                                        }
                                        s += ks[qg] * vaddvq_f32(vaddq_f32(rlo, rhi));
                                    }
                                } else {
                                    const __fp16* kfp = K_new_base + (kv_pos - cache_len) * kv_seq_stride + kv_head_idx * head_dim;
                                    float32x4_t slo = vdupq_n_f32(0.0f), shi = vdupq_n_f32(0.0f);
                                    for (size_t d = 0; d < head_dim_aligned; d += VECTOR_WIDTH) {
                                        float16x8_t qf = vld1q_f16(&q_vec[d]);
                                        float16x8_t kf = vld1q_f16(&kfp[d]);
                                        slo = vfmaq_f32(slo, vcvt_f32_f16(vget_low_f16(qf)), vcvt_f32_f16(vget_low_f16(kf)));
                                        shi = vfmaq_f32(shi, vcvt_f32_f16(vget_high_f16(qf)), vcvt_f32_f16(vget_high_f16(kf)));
                                    }
                                    s = vaddvq_f32(vaddq_f32(slo, shi));
                                }
                                return s * scale;
                            };
                            if (!skip_a) score_a = score_one(kv_pos_a);
                            if (!skip_b) score_b = score_one(kv_pos_b);
                        }

                        block_scores[kv_idx] = skip_a ? -std::numeric_limits<float>::infinity() : score_a;
                        block_scores[kv_idx + 1] = skip_b ? -std::numeric_limits<float>::infinity() : score_b;
                        if (!skip_a) block_max = std::max(block_max, score_a);
                        if (!skip_b) block_max = std::max(block_max, score_b);
                    }

                    for (; kv_idx < block_size; ++kv_idx) {
                        const size_t kv_pos = kv_block_start + kv_idx;
                        if ((is_causal && kv_pos > absolute_q_pos) || (window_size > 0 && kv_pos < kv_start)) {
                            block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        float score = 0.0f;
                        if (kv_pos < cache_len) {
                            const int8_t* kv = K_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                            const float* ks = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;
                            for (size_t qg = 0; qg < num_quant_groups; qg++) {
                                float32x4_t rlo = vdupq_n_f32(0.0f), rhi = vdupq_n_f32(0.0f);
                                const size_t db = qg * quant_group_size;
                                for (size_t ii = 0; ii < 4; ii++) {
                                    const size_t d = db + ii * VECTOR_WIDTH;
                                    if (d >= head_dim_aligned) break;
                                    float16x8_t qf = vld1q_f16(&q_vec[d]);
                                    int16x8_t ki16 = vmovl_s8(vld1_s8(&kv[d]));
                                    rlo = vfmaq_f32(rlo, vcvt_f32_f16(vget_low_f16(qf)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(ki16))));
                                    rhi = vfmaq_f32(rhi, vcvt_f32_f16(vget_high_f16(qf)), vcvtq_f32_s32(vmovl_s16(vget_high_s16(ki16))));
                                }
                                score += ks[qg] * vaddvq_f32(vaddq_f32(rlo, rhi));
                            }
                        } else {
                            const __fp16* kfp = K_new_base + (kv_pos - cache_len) * kv_seq_stride + kv_head_idx * head_dim;
                            float32x4_t slo = vdupq_n_f32(0.0f), shi = vdupq_n_f32(0.0f);
                            for (size_t d = 0; d < head_dim_aligned; d += VECTOR_WIDTH) {
                                float16x8_t qf = vld1q_f16(&q_vec[d]);
                                float16x8_t kf = vld1q_f16(&kfp[d]);
                                slo = vfmaq_f32(slo, vcvt_f32_f16(vget_low_f16(qf)), vcvt_f32_f16(vget_low_f16(kf)));
                                shi = vfmaq_f32(shi, vcvt_f32_f16(vget_high_f16(qf)), vcvt_f32_f16(vget_high_f16(kf)));
                            }
                            score = vaddvq_f32(vaddq_f32(slo, shi));
                        }
                        score *= scale;
                        block_scores[kv_idx] = score;
                        block_max = std::max(block_max, score);
                    }

                    if (block_max > -std::numeric_limits<float>::infinity()) {
                        float scale_correction = expf(running_max - block_max);
                        running_sum *= scale_correction;
                        for (size_t i = 0; i < num_accum_slots; ++i) {
                            output_accum_low[i] = vmulq_n_f32(output_accum_low[i], scale_correction);
                            output_accum_high[i] = vmulq_n_f32(output_accum_high[i], scale_correction);
                        }
                        running_max = block_max;
                    }

                    float block_sum = 0.0f;
                    for (size_t j = 0; j < block_size; ++j) {
                        if (block_scores[j] != -std::numeric_limits<float>::infinity()) {
                            block_scores[j] = expf(block_scores[j] - block_max);
                            block_sum += block_scores[j];
                        } else {
                            block_scores[j] = 0.0f;
                        }
                    }

                    // V accumulation: FP16 block accumulators, widened to FP32 at block end

                    for (size_t i = 0; i < num_accum_slots; ++i)
                        block_accum[i] = vdupq_n_f16((__fp16)0.0f);

                    for (size_t j = 0; j < block_size; ++j) {
                        const float w = block_scores[j];
                        if (w == 0.0f) continue;

                        const size_t kv_pos = kv_block_start + j;

                        if (kv_pos < cache_len) {
                            const int8_t* v_ptr = V_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                            const float* vs = v_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;

                            for (size_t qg = 0; qg < num_quant_groups; qg++) {
                                const size_t dim_base = qg * quant_group_size;
                                const float16x8_t ws_vec = vdupq_n_f16(static_cast<__fp16>(w * vs[qg]));

                                #pragma unroll
                                for (size_t i = 0; i < 4; i++) {
                                    const size_t d = dim_base + i * VECTOR_WIDTH;
                                    if (d >= head_dim_aligned) break;

                                    float16x8_t v_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(&v_ptr[d])));
                                    block_accum[d / VECTOR_WIDTH] = vfmaq_f16(block_accum[d / VECTOR_WIDTH], v_f16, ws_vec);
                                }
                            }
                        } else {
                            const __fp16* v_ptr = V_new_base + (kv_pos - cache_len) * kv_seq_stride + kv_head_idx * head_dim;
                            const float16x8_t w_vec = vdupq_n_f16(static_cast<__fp16>(w));

                            for (size_t d = 0; d < head_dim_aligned; d += VECTOR_WIDTH) {
                                block_accum[d / VECTOR_WIDTH] = vfmaq_f16(block_accum[d / VECTOR_WIDTH], vld1q_f16(&v_ptr[d]), w_vec);
                            }
                        }
                    }

                    for (size_t i = 0; i < num_accum_slots; ++i) {
                        output_accum_low[i] = vaddq_f32(output_accum_low[i], vcvt_f32_f16(vget_low_f16(block_accum[i])));
                        output_accum_high[i] = vaddq_f32(output_accum_high[i], vcvt_f32_f16(vget_high_f16(block_accum[i])));
                    }

                    running_sum += block_sum;
                }

                if (running_sum > 0.0f) {
                    const float inv_sum = 1.0f / running_sum;
                    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

                    for (size_t d = 0; d < head_dim_aligned; d += VECTOR_WIDTH) {
                        size_t idx = d / VECTOR_WIDTH;
                        vst1q_f16(&o_vec[d], vcombine_f16(
                            vcvt_f16_f32(vmulq_f32(output_accum_low[idx], inv_sum_vec)),
                            vcvt_f16_f32(vmulq_f32(output_accum_high[idx], inv_sum_vec))));
                    }
                } else {
                    memset(o_vec, 0, head_dim * sizeof(__fp16));
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

    if (seq_len == 1 && head_dim % 8 == 0) {
        cactus_attention_hybrid_int8_fp16_decode(
            queries, keys_cached, values_cached, k_scales, v_scales,
            keys_new, values_new, output,
            batch_size, cache_len, new_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, position_offset, is_causal, window_size, quant_group_size);
        return;
    }

    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;

    const size_t gqa_group_size = num_q_heads / num_kv_heads;
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