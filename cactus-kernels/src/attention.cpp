#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>
#include <cassert>

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
    size_t v_head_dim,
    float scale,
    size_t position_offset,
    bool is_causal
) {
    constexpr size_t BLOCK_SIZE = 64;

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t k_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t v_batch_stride = kv_seq_len * num_kv_heads * v_head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * v_head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t k_seq_stride = num_kv_heads * head_dim;
    const size_t v_seq_stride = num_kv_heads * v_head_dim;
    const size_t o_seq_stride = num_q_heads * v_head_dim;

    static constexpr CactusThreading::ParallelConfig ATTENTION_BATCHED{1, 1};
    CactusThreading::parallel_for(batch_size * num_q_heads, ATTENTION_BATCHED,
        [&](size_t start, size_t end) {

        std::vector<float> Q_f32(seq_len * head_dim);
        std::vector<float> K_f32(BLOCK_SIZE * head_dim);
        std::vector<float> V_f32(BLOCK_SIZE * v_head_dim);
        std::vector<float> scores(seq_len * BLOCK_SIZE);
        std::vector<float> acc(seq_len * v_head_dim);
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
            memset(acc.data(), 0, seq_len * v_head_dim * sizeof(float));

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
                    const __fp16* k_src = keys + batch*k_batch_stride + (kv0+i)*k_seq_stride + kv_head*head_dim;
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
                        float* acc_row = acc.data() + q_pos * v_head_dim;
                        float32x4_t sv = vdupq_n_f32(scale_old);
                        for (size_t d = 0; d < v_head_dim; d += 4) {
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
                        float32x4_t t = vfmaq_n_f32(vdupq_n_f32(0.2246932f), xf, 0.0789673f);
                        t = vfmaq_f32(vdupq_n_f32(0.6963248f), t, xf);
                        float32x4_t y = vfmaq_f32(vdupq_n_f32(0.9999003f), t, xf);
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
                    const __fp16* v_src = values + batch*v_batch_stride + (kv0+i)*v_seq_stride + kv_head*v_head_dim;
                    float* v_dst = V_f32.data() + i * v_head_dim;
                    for (size_t d = 0; d < v_head_dim; d += 8) {
                        float16x8_t v = vld1q_f16(v_src + d);
                        vst1q_f32(v_dst + d,     vcvt_f32_f16(vget_low_f16(v)));
                        vst1q_f32(v_dst + d + 4, vcvt_f32_f16(vget_high_f16(v)));
                    }
                }

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)active_rows, (int)v_head_dim, (int)block_len,
                            1.0f,
                            scores.data(), (int)block_len,
                            V_f32.data(), (int)v_head_dim,
                            1.0f,
                            acc.data() + q_start * v_head_dim, (int)v_head_dim);
            }

            for (size_t q = 0; q < seq_len; ++q) {
                __fp16* o = output + batch*o_batch_stride + q*o_seq_stride + q_head*v_head_dim;
                float sum = row_sum[q];
                if (sum == 0.0f) {
                    memset(o, 0, v_head_dim * sizeof(__fp16));
                    continue;
                }
                float inv = 1.0f / sum;
                float32x4_t invv = vdupq_n_f32(inv);
                float* acc_row = acc.data() + q * v_head_dim;
                for (size_t d = 0; d < v_head_dim; d += 8) {
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
    bool is_causal,
    size_t window_size,
    size_t v_head_dim
) {
    constexpr size_t BLOCK_SIZE = 32;
    const size_t qk_nblocks = head_dim / 8;
    const size_t v_nblocks = v_head_dim / 8;

#ifdef __APPLE__
    if (seq_len >= 64 && window_size == 0) {
        cactus_attention_f16_accelerate(
            queries, keys, values, output,
            batch_size, seq_len, kv_seq_len,
            num_q_heads, num_kv_heads, head_dim, v_head_dim,
            scale, position_offset, is_causal
        );
        return;
    }
#endif

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t v_batch_stride = kv_seq_len * num_kv_heads * v_head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * v_head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t v_seq_stride = num_kv_heads * v_head_dim;
    const size_t o_seq_stride = num_q_heads * v_head_dim;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start, size_t end) {

        float block_scores[BLOCK_SIZE];
        std::vector<float32x4_t> acc_lo(v_nblocks), acc_hi(v_nblocks);

        for (size_t work = start; work < end; ++work) {
            const size_t batch = work / (num_q_heads * seq_len);
            const size_t rem = work % (num_q_heads * seq_len);
            const size_t q_head = rem / seq_len;
            const size_t q_pos = rem % seq_len;
            const size_t kv_head = q_head / group_size;

            const __fp16* q = queries + batch*q_batch_stride + q_pos*q_seq_stride + q_head*head_dim;
            __fp16* o = output + batch*o_batch_stride + q_pos*o_seq_stride + q_head*v_head_dim;

            for (size_t i = 0; i < v_nblocks; i++) {
                acc_lo[i] = vdupq_n_f32(0.f);
                acc_hi[i] = vdupq_n_f32(0.f);
            }

            float running_max = -INFINITY;
            float running_sum = 0.f;

            const size_t abs_q = position_offset + q_pos;
            size_t kv_end = is_causal ? std::min(kv_seq_len, abs_q + 1) : kv_seq_len;
            size_t kv_start = (window_size > 0 && abs_q > window_size) ? abs_q - window_size : 0;

            for (size_t kv0 = kv_start; kv0 < kv_end; kv0 += BLOCK_SIZE) {
                const size_t kv1 = std::min(kv0 + BLOCK_SIZE, kv_end);
                float block_max = -INFINITY;

                for (size_t i = kv0; i < kv1; i++) {
                    float32x4_t s0 = vdupq_n_f32(0.f);
                    float32x4_t s1 = vdupq_n_f32(0.f);

                    const __fp16* k = keys + batch*kv_batch_stride + i*kv_seq_stride + kv_head*head_dim;

                    for (size_t d = 0; d < qk_nblocks; d++) {
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

                float current_block_scale = 1.0f;
                if (block_max > running_max) {
                    float scale_correction = expf(running_max - block_max);
                    running_sum *= scale_correction;

                    for (size_t d = 0; d < v_nblocks; d++) {
                        acc_lo[d] = vmulq_n_f32(acc_lo[d], scale_correction);
                        acc_hi[d] = vmulq_n_f32(acc_hi[d], scale_correction);
                    }
                    running_max = block_max;
                } else {
                    current_block_scale = expf(block_max - running_max);
                }

                float block_sum = 0.f;
                for (size_t i = 0; i < kv1 - kv0; i++) {
                    block_scores[i] = expf(block_scores[i] - block_max);
                    block_sum += block_scores[i];
                }

                for (size_t i = 0; i < kv1 - kv0; i++) {
                    const float attn_weight = block_scores[i] * current_block_scale;
                    if (attn_weight == 0.f) continue;

                    const __fp16* v = values + batch*v_batch_stride + (kv0+i)*v_seq_stride + kv_head*v_head_dim;
                    float32x4_t wv = vdupq_n_f32(attn_weight);

                    for (size_t d = 0; d < v_nblocks; d++) {
                        float16x8_t vv = vld1q_f16(v + d*8);
                        acc_lo[d] = vfmaq_f32(acc_lo[d], vcvt_f32_f16(vget_low_f16(vv)), wv);
                        acc_hi[d] = vfmaq_f32(acc_hi[d], vcvt_f32_f16(vget_high_f16(vv)), wv);
                    }
                }

                running_sum += block_sum * current_block_scale;
            }

            if (running_sum == 0.f) {
                memset(o, 0, v_head_dim * sizeof(__fp16));
                continue;
            }

            float inv = 1.f / running_sum;
            float32x4_t invv = vdupq_n_f32(inv);

            for (size_t d = 0; d < v_nblocks; d++) {
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
    bool is_causal,
    bool mask_is_additive,
    bool mask_per_head,
    size_t v_head_dim,
    float logit_cap
) {
    if (v_head_dim == 0) v_head_dim = head_dim;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    if (mask == nullptr && head_dim % 8 == 0 && v_head_dim % 8 == 0 && logit_cap == 0.0f) {
        cactus_attention_f16_fast(
            queries, keys, values, output,
            batch_size, seq_len, kv_seq_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, position_offset, is_causal, window_size, v_head_dim
        );
        return;
    }

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;
    const size_t v_head_dim_aligned = (v_head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;

    const size_t group_size = num_q_heads / num_kv_heads;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads * head_dim;
    const size_t v_batch_stride = kv_seq_len * num_kv_heads * v_head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * v_head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t v_seq_stride = num_kv_heads * v_head_dim;
    const size_t o_seq_stride = num_q_heads * v_head_dim;
    const size_t mask_batch_stride = mask
        ? (mask_per_head ? (num_q_heads * seq_len * kv_seq_len) : (seq_len * kv_seq_len))
        : 0;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            std::vector<float> block_scores(BLOCK_SIZE);
            std::vector<float32x4_t> output_accum_low(v_head_dim_aligned / VECTOR_WIDTH * 2);
            std::vector<float32x4_t> output_accum_high(v_head_dim_aligned / VECTOR_WIDTH * 2);
            
            const size_t v_tail_dims = v_head_dim - v_head_dim_aligned;
            std::vector<float> output_accum_tail(v_tail_dims, 0.0f);

            const float NEG_INF = -std::numeric_limits<float>::infinity();
            const size_t used_vec_blocks = v_head_dim_aligned / VECTOR_WIDTH;

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;

                const size_t kv_head_idx = q_head_idx / group_size;

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const __fp16* K_base = keys + batch_idx * kv_batch_stride;
                const __fp16* V_base = values + batch_idx * v_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;
                const __fp16* M = mask ? (mask + batch_idx * mask_batch_stride) : nullptr;
                    const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                    __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * v_head_dim;
                    
                    float running_max = -std::numeric_limits<float>::infinity();
                    float running_sum = 0.0f;
                    
                    for (size_t i = 0; i < output_accum_low.size(); ++i) {
                        output_accum_low[i] = vdupq_n_f32(0.0f);
                        output_accum_high[i] = vdupq_n_f32(0.0f);
                    }
                    for (size_t i = 0; i < v_tail_dims; ++i) {
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
                            
                            if (logit_cap > 0.0f && std::isfinite(score)) {
                                score = logit_cap * tanhf(score / logit_cap);
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
                            for (size_t i = 0; i < v_tail_dims; ++i) {
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

                            float32x4_t t = vfmaq_n_f32(vdupq_n_f32(0.2246932f), xf, 0.0789673f);
                            t = vfmaq_f32(vdupq_n_f32(0.6963248f), t, xf);
                            float32x4_t y = vfmaq_f32(vdupq_n_f32(0.9999003f), t, xf);

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
                            const __fp16* v_vec = V_base + kv_pos * v_seq_stride + kv_head_idx * v_head_dim;
                            
                            const float32x4_t weight_vec = vdupq_n_f32(attn_weight);
                            
                            for (size_t dim_block = 0; dim_block < v_head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t v_vec_f16 = vld1q_f16(&v_vec[dim_block]);
                                float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v_vec_f16));
                                float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v_vec_f16));
                                
                                size_t idx = dim_block / VECTOR_WIDTH;
                                output_accum_low[idx] = vfmaq_f32(output_accum_low[idx], v_low, weight_vec);
                                output_accum_high[idx] = vfmaq_f32(output_accum_high[idx], v_high, weight_vec);
                            }
                            
                            for (size_t dim = v_head_dim_aligned; dim < v_head_dim; ++dim) {
                                float val = attn_weight * static_cast<float>(v_vec[dim]);
                                output_accum_tail[dim - v_head_dim_aligned] += val;
                            }
                        }
                        
                        running_sum += block_sum * current_block_scale;
                    }
                    
                    if (running_sum > 0.0f) {
                        const float inv_sum = 1.0f / running_sum;
                        const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
                        
                        for (size_t dim_block = 0; dim_block < v_head_dim_aligned; dim_block += VECTOR_WIDTH) {
                            size_t idx = dim_block / VECTOR_WIDTH;
                            float32x4_t final_low = vmulq_f32(output_accum_low[idx], inv_sum_vec);
                            float32x4_t final_high = vmulq_f32(output_accum_high[idx], inv_sum_vec);
                            
                            float16x4_t low_f16 = vcvt_f16_f32(final_low);
                            float16x4_t high_f16 = vcvt_f16_f32(final_high);
                            float16x8_t combined = vcombine_f16(low_f16, high_f16);
                            
                            vst1q_f16(&o_vec[dim_block], combined);
                        }
                        
                        for (size_t dim = v_head_dim_aligned; dim < v_head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(output_accum_tail[dim - v_head_dim_aligned] * inv_sum);
                        }
                    } else {
                        for (size_t dim = 0; dim < v_head_dim; ++dim) {
                            o_vec[dim] = static_cast<__fp16>(0.0f);
                        }
                    }
            }
        });
}

static void cactus_attention_hybrid_int8_fp16_decode_dot(
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
    size_t window_size
) {
    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 64;
    constexpr size_t QGROUP = 32;
    constexpr size_t MAX_HEAD_DIM = 512;
    constexpr size_t MAX_QUANT_GROUPS = MAX_HEAD_DIM / QGROUP;
    constexpr size_t MAX_ACCUM_SLOTS = MAX_HEAD_DIM / VECTOR_WIDTH;

    const size_t num_quant_groups = head_dim / QGROUP;
    const size_t num_accum_slots = head_dim / VECTOR_WIDTH;
    const size_t gqa_group_size = num_q_heads / num_kv_heads;

    const size_t q_batch_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t k_cached_batch_stride = cache_len * kv_seq_stride;
    const size_t v_cached_batch_stride = cache_len * kv_seq_stride;
    const size_t k_new_batch_stride = new_len * kv_seq_stride;
    const size_t v_new_batch_stride = new_len * kv_seq_stride;
    const size_t o_batch_stride = num_q_heads * head_dim;

    CactusThreading::parallel_for(batch_size * num_q_heads, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            alignas(16) int8_t q_int8[MAX_HEAD_DIM];
            float q_scales[MAX_QUANT_GROUPS];
            float block_scores[BLOCK_SIZE];
            float32x4_t output_accum_low[MAX_ACCUM_SLOTS];
            float32x4_t output_accum_high[MAX_ACCUM_SLOTS];
            float16x8_t block_accum[MAX_ACCUM_SLOTS];

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / num_q_heads;
                const size_t q_head_idx = work_idx % num_q_heads;
                const size_t kv_head_idx = q_head_idx / gqa_group_size;

                const __fp16* q_vec = queries + batch_idx * q_batch_stride + q_head_idx * head_dim;
                const int8_t* K_cached_base = keys_cached + batch_idx * k_cached_batch_stride;
                const int8_t* V_cached_base = values_cached + batch_idx * v_cached_batch_stride;
                const __fp16* K_new_base = keys_new + batch_idx * k_new_batch_stride;
                const __fp16* V_new_base = values_new + batch_idx * v_new_batch_stride;
                __fp16* o_vec = output + batch_idx * o_batch_stride + q_head_idx * head_dim;

                for (size_t qg = 0; qg < num_quant_groups; ++qg) {
                    const __fp16* q_grp = q_vec + qg * QGROUP;
                    float16x8_t amax_v = vabsq_f16(vld1q_f16(q_grp));
                    for (size_t i = 1; i < QGROUP / VECTOR_WIDTH; ++i)
                        amax_v = vmaxq_f16(amax_v, vabsq_f16(vld1q_f16(q_grp + i * VECTOR_WIDTH)));
                    float amax = static_cast<float>(vmaxvq_f16(amax_v));
                    float q_scale = amax / 127.0f;
                    float inv = q_scale > 0.0f ? 127.0f / amax : 0.0f;
                    q_scales[qg] = q_scale;

                    int8_t* qd = q_int8 + qg * QGROUP;
                    for (size_t i = 0; i < QGROUP / VECTOR_WIDTH; ++i) {
                        float16x8_t qf = vld1q_f16(q_grp + i * VECTOR_WIDTH);
                        float32x4_t lo = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(qf)), inv);
                        float32x4_t hi = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(qf)), inv);
                        int32x4_t lo_i = vcvtaq_s32_f32(lo);
                        int32x4_t hi_i = vcvtaq_s32_f32(hi);
                        int16x8_t pack16 = vcombine_s16(vqmovn_s32(lo_i), vqmovn_s32(hi_i));
                        vst1_s8(qd + i * VECTOR_WIDTH, vqmovn_s16(pack16));
                    }
                }

                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                for (size_t i = 0; i < num_accum_slots; ++i) {
                    output_accum_low[i] = vdupq_n_f32(0.0f);
                    output_accum_high[i] = vdupq_n_f32(0.0f);
                }

                const size_t absolute_q_pos = position_offset;
                const size_t kv_end = is_causal ? std::min(kv_seq_len, absolute_q_pos + 1) : kv_seq_len;
                const size_t kv_start_abs = (window_size > 0 && absolute_q_pos > window_size)
                                            ? absolute_q_pos - window_size : 0;
                const size_t kv_start = (position_offset > cache_len) ? 0 : kv_start_abs;

                for (size_t kv_block_start = kv_start; kv_block_start < kv_end; kv_block_start += BLOCK_SIZE) {
                    const size_t kv_block_end = std::min(kv_block_start + BLOCK_SIZE, kv_end);
                    const size_t block_size = kv_block_end - kv_block_start;

                    float block_max = -std::numeric_limits<float>::infinity();

                    const size_t cached_kv_end = std::min(kv_block_end, cache_len);
                    const size_t new_kv_start = std::max(kv_block_start, cache_len);

                    size_t kv_pos = kv_block_start;
                    for (; kv_pos + 3 < cached_kv_end; kv_pos += 4) {
                        const int8_t* k1 = K_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                        const int8_t* k2 = k1 + kv_seq_stride;
                        const int8_t* k3 = k2 + kv_seq_stride;
                        const int8_t* k4 = k3 + kv_seq_stride;
                        const float* ks1 = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;
                        const float* ks2 = ks1 + num_kv_heads * num_quant_groups;
                        const float* ks3 = ks2 + num_kv_heads * num_quant_groups;
                        const float* ks4 = ks3 + num_kv_heads * num_quant_groups;
                        if (kv_pos + 8 < cached_kv_end) {
                            __builtin_prefetch(k1 + 4 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(k1 + 5 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(k1 + 6 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(k1 + 7 * kv_seq_stride, 0, 0);
                        }

                        float32x4_t sumv1 = vdupq_n_f32(0.0f);
                        float32x4_t sumv2 = vdupq_n_f32(0.0f);
                        float32x4_t sumv3 = vdupq_n_f32(0.0f);
                        float32x4_t sumv4 = vdupq_n_f32(0.0f);

                        for (size_t qg = 0; qg < num_quant_groups; ++qg) {
                            int8x16_t q_lo = vld1q_s8(q_int8 + qg * QGROUP);
                            int8x16_t q_hi = vld1q_s8(q_int8 + qg * QGROUP + 16);

                            int32x4_t d1 = vdupq_n_s32(0);
                            int32x4_t d2 = vdupq_n_s32(0);
                            int32x4_t d3 = vdupq_n_s32(0);
                            int32x4_t d4 = vdupq_n_s32(0);

                            d1 = vdotq_s32(d1, q_lo, vld1q_s8(k1 + qg * QGROUP));
                            d2 = vdotq_s32(d2, q_lo, vld1q_s8(k2 + qg * QGROUP));
                            d3 = vdotq_s32(d3, q_lo, vld1q_s8(k3 + qg * QGROUP));
                            d4 = vdotq_s32(d4, q_lo, vld1q_s8(k4 + qg * QGROUP));
                            d1 = vdotq_s32(d1, q_hi, vld1q_s8(k1 + qg * QGROUP + 16));
                            d2 = vdotq_s32(d2, q_hi, vld1q_s8(k2 + qg * QGROUP + 16));
                            d3 = vdotq_s32(d3, q_hi, vld1q_s8(k3 + qg * QGROUP + 16));
                            d4 = vdotq_s32(d4, q_hi, vld1q_s8(k4 + qg * QGROUP + 16));

                            float qg_q = q_scales[qg];
                            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(d1), qg_q * ks1[qg]);
                            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(d2), qg_q * ks2[qg]);
                            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(d3), qg_q * ks3[qg]);
                            sumv4 = vmlaq_n_f32(sumv4, vcvtq_f32_s32(d4), qg_q * ks4[qg]);
                        }
                        float s1 = vaddvq_f32(sumv1) * scale;
                        float s2 = vaddvq_f32(sumv2) * scale;
                        float s3 = vaddvq_f32(sumv3) * scale;
                        float s4 = vaddvq_f32(sumv4) * scale;
                        block_scores[kv_pos - kv_block_start] = s1;
                        block_scores[kv_pos - kv_block_start + 1] = s2;
                        block_scores[kv_pos - kv_block_start + 2] = s3;
                        block_scores[kv_pos - kv_block_start + 3] = s4;
                        float local_max = std::max(std::max(s1, s2), std::max(s3, s4));
                        if (local_max > block_max) block_max = local_max;
                    }
                    for (; kv_pos < cached_kv_end; ++kv_pos) {
                        const int8_t* k_vec = K_cached_base + kv_pos * kv_seq_stride + kv_head_idx * head_dim;
                        const float* k_scale_base = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups;

                        float32x4_t sumv = vdupq_n_f32(0.0f);
                        for (size_t qg = 0; qg < num_quant_groups; ++qg) {
                            int8x16_t q_lo = vld1q_s8(q_int8 + qg * QGROUP);
                            int8x16_t q_hi = vld1q_s8(q_int8 + qg * QGROUP + 16);
                            int8x16_t k_lo = vld1q_s8(k_vec + qg * QGROUP);
                            int8x16_t k_hi = vld1q_s8(k_vec + qg * QGROUP + 16);
                            int32x4_t dot_acc = vdupq_n_s32(0);
                            dot_acc = vdotq_s32(dot_acc, q_lo, k_lo);
                            dot_acc = vdotq_s32(dot_acc, q_hi, k_hi);
                            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(dot_acc), q_scales[qg] * k_scale_base[qg]);
                        }
                        float score = vaddvq_f32(sumv) * scale;
                        block_scores[kv_pos - kv_block_start] = score;
                        block_max = std::max(block_max, score);
                    }

                    for (kv_pos = std::max(kv_pos, new_kv_start); kv_pos < kv_block_end; ++kv_pos) {
                        if (is_causal && kv_pos > absolute_q_pos) {
                            block_scores[kv_pos - kv_block_start] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        const size_t new_pos = kv_pos - cache_len;
                        const __fp16* k_vec = K_new_base + new_pos * kv_seq_stride + kv_head_idx * head_dim;
                        float16x8_t s_acc = vdupq_n_f16((__fp16)0.0f);
                        for (size_t d = 0; d < head_dim; d += VECTOR_WIDTH) {
                            s_acc = vfmaq_f16(s_acc, vld1q_f16(q_vec + d), vld1q_f16(k_vec + d));
                        }
                        float score = (vaddvq_f32(vcvt_f32_f16(vget_low_f16(s_acc))) +
                                       vaddvq_f32(vcvt_f32_f16(vget_high_f16(s_acc)))) * scale;
                        block_scores[kv_pos - kv_block_start] = score;
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
                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        if (block_scores[kv_idx] != -std::numeric_limits<float>::infinity()) {
                            block_scores[kv_idx] = expf(block_scores[kv_idx] - block_max);
                            block_sum += block_scores[kv_idx];
                        } else {
                            block_scores[kv_idx] = 0.0f;
                        }
                    }

                    for (size_t i = 0; i < num_accum_slots; ++i)
                        block_accum[i] = vdupq_n_f16((__fp16)0.0f);

                    const size_t cached_block_end = std::min(kv_block_end, cache_len);
                    size_t v_kv = kv_block_start;
                    for (; v_kv + 3 < cached_block_end; v_kv += 4) {
                        const float w1 = block_scores[v_kv - kv_block_start];
                        const float w2 = block_scores[v_kv + 1 - kv_block_start];
                        const float w3 = block_scores[v_kv + 2 - kv_block_start];
                        const float w4 = block_scores[v_kv + 3 - kv_block_start];
                        if (w1 == 0.0f && w2 == 0.0f && w3 == 0.0f && w4 == 0.0f) continue;

                        const int8_t* v1 = V_cached_base + v_kv * kv_seq_stride + kv_head_idx * head_dim;
                        const int8_t* v2 = v1 + kv_seq_stride;
                        const int8_t* v3 = v2 + kv_seq_stride;
                        const int8_t* v4 = v3 + kv_seq_stride;
                        const float* vs1 = v_scales + (v_kv * num_kv_heads + kv_head_idx) * num_quant_groups;
                        const float* vs2 = vs1 + num_kv_heads * num_quant_groups;
                        const float* vs3 = vs2 + num_kv_heads * num_quant_groups;
                        const float* vs4 = vs3 + num_kv_heads * num_quant_groups;
                        if (v_kv + 8 < cached_block_end) {
                            __builtin_prefetch(v1 + 4 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(v1 + 5 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(v1 + 6 * kv_seq_stride, 0, 0);
                            __builtin_prefetch(v1 + 7 * kv_seq_stride, 0, 0);
                        }

                        for (size_t qg = 0; qg < num_quant_groups; ++qg) {
                            const float16x8_t ws1_vec = vdupq_n_f16(static_cast<__fp16>(w1 * vs1[qg]));
                            const float16x8_t ws2_vec = vdupq_n_f16(static_cast<__fp16>(w2 * vs2[qg]));
                            const float16x8_t ws3_vec = vdupq_n_f16(static_cast<__fp16>(w3 * vs3[qg]));
                            const float16x8_t ws4_vec = vdupq_n_f16(static_cast<__fp16>(w4 * vs4[qg]));
                            #pragma unroll
                            for (size_t i = 0; i < QGROUP / VECTOR_WIDTH; ++i) {
                                const size_t d = qg * QGROUP + i * VECTOR_WIDTH;
                                float16x8_t v1_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(v1 + d)));
                                float16x8_t v2_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(v2 + d)));
                                float16x8_t v3_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(v3 + d)));
                                float16x8_t v4_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(v4 + d)));
                                float16x8_t acc = block_accum[d / VECTOR_WIDTH];
                                acc = vfmaq_f16(acc, v1_f16, ws1_vec);
                                acc = vfmaq_f16(acc, v2_f16, ws2_vec);
                                acc = vfmaq_f16(acc, v3_f16, ws3_vec);
                                acc = vfmaq_f16(acc, v4_f16, ws4_vec);
                                block_accum[d / VECTOR_WIDTH] = acc;
                            }
                        }
                    }
                    for (; v_kv < cached_block_end; ++v_kv) {
                        const float w = block_scores[v_kv - kv_block_start];
                        if (w == 0.0f) continue;
                        const int8_t* v_vec = V_cached_base + v_kv * kv_seq_stride + kv_head_idx * head_dim;
                        const float* v_scale_base = v_scales + (v_kv * num_kv_heads + kv_head_idx) * num_quant_groups;
                        for (size_t qg = 0; qg < num_quant_groups; ++qg) {
                            const float16x8_t ws_vec = vdupq_n_f16(static_cast<__fp16>(w * v_scale_base[qg]));
                            #pragma unroll
                            for (size_t i = 0; i < QGROUP / VECTOR_WIDTH; ++i) {
                                const size_t d = qg * QGROUP + i * VECTOR_WIDTH;
                                float16x8_t v_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(v_vec + d)));
                                block_accum[d / VECTOR_WIDTH] = vfmaq_f16(block_accum[d / VECTOR_WIDTH], v_f16, ws_vec);
                            }
                        }
                    }
                    for (size_t kv_idx = std::max(v_kv, std::max(kv_block_start, cache_len)); kv_idx < kv_block_end; ++kv_idx) {
                        const float w = block_scores[kv_idx - kv_block_start];
                        if (w == 0.0f) continue;
                        const size_t new_pos = kv_idx - cache_len;
                        const __fp16* v_vec = V_new_base + new_pos * kv_seq_stride + kv_head_idx * head_dim;
                        const float16x8_t w_vec = vdupq_n_f16(static_cast<__fp16>(w));
                        for (size_t d = 0; d < head_dim; d += VECTOR_WIDTH) {
                            block_accum[d / VECTOR_WIDTH] = vfmaq_f16(block_accum[d / VECTOR_WIDTH], vld1q_f16(v_vec + d), w_vec);
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
                    for (size_t d = 0; d < head_dim; d += VECTOR_WIDTH) {
                        size_t idx = d / VECTOR_WIDTH;
                        vst1q_f16(o_vec + d, vcombine_f16(
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
    size_t quant_group_size,
    size_t v_head_dim
) {
    if (v_head_dim == 0) v_head_dim = head_dim;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    if (seq_len == 1 &&
        head_dim == v_head_dim &&
        head_dim <= 512 &&
        head_dim % 32 == 0 &&
        quant_group_size == 32) {
        cactus_attention_hybrid_int8_fp16_decode_dot(
            queries, keys_cached, values_cached, k_scales, v_scales,
            keys_new, values_new, output,
            batch_size, cache_len, new_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, position_offset, is_causal, window_size);
        return;
    }

    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;
    const size_t v_head_dim_aligned = (v_head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;
    const size_t num_accum_slots = v_head_dim_aligned / VECTOR_WIDTH;

    const size_t gqa_group_size = num_q_heads / num_kv_heads;
    const size_t num_quant_groups_k = (head_dim + quant_group_size - 1) / quant_group_size;
    const size_t num_quant_groups_v = (v_head_dim + quant_group_size - 1) / quant_group_size;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t k_cached_batch_stride = cache_len * num_kv_heads * head_dim;
    const size_t v_cached_batch_stride = cache_len * num_kv_heads * v_head_dim;
    const size_t k_new_batch_stride = new_len * num_kv_heads * head_dim;
    const size_t v_new_batch_stride = new_len * num_kv_heads * v_head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * v_head_dim;
    const size_t q_seq_stride = num_q_heads  * head_dim;
    const size_t k_seq_stride = num_kv_heads * head_dim;
    const size_t v_seq_stride = num_kv_heads * v_head_dim;
    const size_t o_seq_stride = num_q_heads * v_head_dim;

    CactusThreading::parallel_for(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            float block_scores[BLOCK_SIZE];
            std::vector<float32x4_t> output_accum_low(num_accum_slots);
            std::vector<float32x4_t> output_accum_high(num_accum_slots);
            std::vector<float16x8_t> block_accum(num_accum_slots);

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;

                const size_t kv_head_idx = q_head_idx / gqa_group_size;

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const int8_t* K_cached_base = keys_cached + batch_idx * k_cached_batch_stride;
                const int8_t* V_cached_base = values_cached + batch_idx * v_cached_batch_stride;
                const __fp16* K_new_base = keys_new + batch_idx * k_new_batch_stride;
                const __fp16* V_new_base = values_new + batch_idx * v_new_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;

                const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * v_head_dim;

                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                for (size_t i = 0; i < num_accum_slots; ++i) {
                    output_accum_low[i] = vdupq_n_f32(0.0f);
                    output_accum_high[i] = vdupq_n_f32(0.0f);
                }

                const size_t absolute_q_pos = position_offset + q_pos;
                size_t kv_end = is_causal ? std::min(kv_seq_len, cache_len + q_pos + 1) : kv_seq_len;

                size_t kv_start = 0;
                if (window_size > 0 && absolute_q_pos > window_size) {
                    kv_start = absolute_q_pos - window_size;
                }

                constexpr size_t SINK_SIZE = 4;
                const size_t cache_abs_offset = (position_offset >= cache_len) ? (position_offset - cache_len) : 0;

                const size_t kv_block_start0 = (window_size > 0 && kv_start > 0) ? 0
                    : (kv_start / BLOCK_SIZE) * BLOCK_SIZE;

                for (size_t kv_block_start = kv_block_start0; kv_block_start < kv_end; kv_block_start += BLOCK_SIZE) {
                    const size_t kv_block_end = std::min(kv_block_start + BLOCK_SIZE, kv_end);
                    const size_t block_size = kv_block_end - kv_block_start;

                    float block_max = -std::numeric_limits<float>::infinity();

                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        const size_t kv_pos = kv_block_start + kv_idx;

                        bool window_masked = false;
                        if (window_size > 0 && kv_start > 0) {
                            if (kv_pos < cache_len) {
                                if (cache_abs_offset == 0 || kv_pos >= SINK_SIZE) {
                                    window_masked = (cache_abs_offset + kv_pos < kv_start);
                                }
                            } else {
                                window_masked = (kv_pos + cache_abs_offset < kv_start);
                            }
                        }

                        if ((is_causal && kv_pos > absolute_q_pos) || window_masked) {
                            block_scores[kv_idx] = -std::numeric_limits<float>::infinity();
                            continue;
                        }

                        float score = 0.0f;

                        if (kv_pos < cache_len) {
                            if (k_scales != nullptr) {
                                const int8_t* k_vec = K_cached_base + kv_pos * k_seq_stride + kv_head_idx * head_dim;
                                const float* k_scale_base = k_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups_k;

                                for (size_t quant_group = 0; quant_group < num_quant_groups_k; quant_group++) {
                                    const size_t dim_base = quant_group * quant_group_size;
                                    float16x8_t s_acc = vdupq_n_f16((__fp16)0.0f);

                                    #pragma unroll
                                    for (size_t i = 0; i < 4; i++) {
                                        const size_t dim_block = dim_base + i * VECTOR_WIDTH;
                                        if (dim_block >= head_dim_aligned) break;

                                        float16x8_t q_f16 = vld1q_f16(&q_vec[dim_block]);
                                        float16x8_t k_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(&k_vec[dim_block])));
                                        s_acc = vfmaq_f16(s_acc, q_f16, k_f16);
                                    }

                                    float partial = vaddvq_f32(vcvt_f32_f16(vget_low_f16(s_acc))) +
                                                    vaddvq_f32(vcvt_f32_f16(vget_high_f16(s_acc)));
                                    score += k_scale_base[quant_group] * partial;
                                }
                            } else {
                                const __fp16* k_vec = reinterpret_cast<const __fp16*>(K_cached_base) +
                                    kv_pos * k_seq_stride + kv_head_idx * head_dim;
                                float16x8_t s_acc = vdupq_n_f16((__fp16)0.0f);

                                for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                    float16x8_t q_f16 = vld1q_f16(&q_vec[dim_block]);
                                    float16x8_t k_f16 = vld1q_f16(&k_vec[dim_block]);
                                    s_acc = vfmaq_f16(s_acc, q_f16, k_f16);
                                }

                                score = vaddvq_f32(vcvt_f32_f16(vget_low_f16(s_acc))) +
                                        vaddvq_f32(vcvt_f32_f16(vget_high_f16(s_acc)));
                            }
                        } else {
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* k_vec = K_new_base + new_pos * k_seq_stride + kv_head_idx * head_dim;

                            float16x8_t s_acc = vdupq_n_f16((__fp16)0.0f);

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t q_f16 = vld1q_f16(&q_vec[dim_block]);
                                float16x8_t k_f16 = vld1q_f16(&k_vec[dim_block]);
                                s_acc = vfmaq_f16(s_acc, q_f16, k_f16);
                            }

                            score = vaddvq_f32(vcvt_f32_f16(vget_low_f16(s_acc))) +
                                    vaddvq_f32(vcvt_f32_f16(vget_high_f16(s_acc)));
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
                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        if (block_scores[kv_idx] != -std::numeric_limits<float>::infinity()) {
                            block_scores[kv_idx] = expf(block_scores[kv_idx] - block_max);
                            block_sum += block_scores[kv_idx];
                        } else {
                            block_scores[kv_idx] = 0.0f;
                        }
                    }

                    for (size_t i = 0; i < num_accum_slots; ++i)
                        block_accum[i] = vdupq_n_f16((__fp16)0.0f);

                    for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                        const float attn_weight = block_scores[kv_idx];
                        if (attn_weight == 0.0f) continue;

                        const size_t kv_pos = kv_block_start + kv_idx;

                        if (kv_pos < cache_len) {
                            if (v_scales != nullptr) {
                                const int8_t* v_vec = V_cached_base + kv_pos * v_seq_stride + kv_head_idx * v_head_dim;
                                const float* v_scale_base = v_scales + (kv_pos * num_kv_heads + kv_head_idx) * num_quant_groups_v;

                                for (size_t quant_group = 0; quant_group < num_quant_groups_v; quant_group++) {
                                    const size_t dim_base = quant_group * quant_group_size;
                                    const float16x8_t ws_vec = vdupq_n_f16(static_cast<__fp16>(attn_weight * v_scale_base[quant_group]));

                                    #pragma unroll
                                    for (size_t i = 0; i < 4; i++) {
                                        const size_t dim_block = dim_base + i * VECTOR_WIDTH;
                                        if (dim_block >= v_head_dim_aligned) break;

                                        float16x8_t v_f16 = vcvtq_f16_s16(vmovl_s8(vld1_s8(&v_vec[dim_block])));
                                        block_accum[dim_block / VECTOR_WIDTH] = vfmaq_f16(block_accum[dim_block / VECTOR_WIDTH], v_f16, ws_vec);
                                    }
                                }
                            } else {
                                const __fp16* v_vec = reinterpret_cast<const __fp16*>(V_cached_base) +
                                    kv_pos * v_seq_stride + kv_head_idx * v_head_dim;
                                const float16x8_t w_vec = vdupq_n_f16(static_cast<__fp16>(attn_weight));

                                for (size_t dim_block = 0; dim_block < v_head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                    block_accum[dim_block / VECTOR_WIDTH] =
                                        vfmaq_f16(block_accum[dim_block / VECTOR_WIDTH], vld1q_f16(&v_vec[dim_block]), w_vec);
                                }
                            }
                        } else {
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* v_vec = V_new_base + new_pos * v_seq_stride + kv_head_idx * v_head_dim;
                            const float16x8_t w_vec = vdupq_n_f16(static_cast<__fp16>(attn_weight));

                            for (size_t dim_block = 0; dim_block < v_head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                block_accum[dim_block / VECTOR_WIDTH] = vfmaq_f16(block_accum[dim_block / VECTOR_WIDTH], vld1q_f16(&v_vec[dim_block]), w_vec);
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

                    for (size_t dim_block = 0; dim_block < v_head_dim_aligned; dim_block += VECTOR_WIDTH) {
                        size_t idx = dim_block / VECTOR_WIDTH;
                        vst1q_f16(&o_vec[dim_block], vcombine_f16(
                            vcvt_f16_f32(vmulq_f32(output_accum_low[idx], inv_sum_vec)),
                            vcvt_f16_f32(vmulq_f32(output_accum_high[idx], inv_sum_vec))));
                    }
                } else {
                    memset(o_vec, 0, v_head_dim * sizeof(__fp16));
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
