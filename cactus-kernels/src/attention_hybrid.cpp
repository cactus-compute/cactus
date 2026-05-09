#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>

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
