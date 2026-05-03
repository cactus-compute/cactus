#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>
#include <cassert>

void cactus_attention_hybrid_turboquant_fp16(
    const __fp16 *queries,
    const float *cached_key_radii,      const uint8_t *cached_key_angles,
    const float *cached_value_radii,    const uint8_t *cached_value_angles,
    const uint8_t *rotation_signs,
    const __fp16 *keys_new, const __fp16 *values_new, __fp16 *output,
    size_t batch_size, size_t seq_len, size_t cache_len, size_t new_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale,
    size_t angle_bits, size_t value_angle_bits,
    size_t position_offset, bool is_causal, size_t window_size
) {
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    const size_t kv_seq_len = cache_len + new_len;

    constexpr size_t VECTOR_WIDTH = 8;
    constexpr size_t BLOCK_SIZE = 32;
    const size_t head_dim_aligned = (head_dim / VECTOR_WIDTH) * VECTOR_WIDTH;

    const size_t gqa_group_size = num_q_heads / num_kv_heads;

    const size_t key_angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    const size_t val_angles_bytes = turboquant_angles_bytes_per_head(head_dim, value_angle_bits);

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_new_batch_stride = new_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = num_q_heads * head_dim;
    const size_t cache_batch_stride = cache_len * num_kv_heads;

    CactusThreading::parallel_for_static(batch_size * num_q_heads * seq_len, CactusThreading::Thresholds::ATTENTION,
        [=](size_t start_idx, size_t end_idx) {
            std::vector<float> _q_rot_f32(head_dim);
            std::vector<float> _cached_acc(head_dim);
            std::vector<float> _new_acc(head_dim);
            std::vector<float> block_scores(BLOCK_SIZE);
            float* q_rot_f32 = _q_rot_f32.data();
            float* cached_acc = _cached_acc.data();
            float* new_acc = _new_acc.data();

            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t batch_idx = work_idx / (num_q_heads * seq_len);
                const size_t remainder = work_idx % (num_q_heads * seq_len);
                const size_t q_head_idx = remainder / seq_len;
                const size_t q_pos = remainder % seq_len;

                const size_t kv_head_idx = q_head_idx / gqa_group_size;

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const __fp16* K_new_base = keys_new + batch_idx * kv_new_batch_stride;
                const __fp16* V_new_base = values_new + batch_idx * kv_new_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;

                const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;

                float q_norm;
                {
                    float32x4_t nacc0 = vdupq_n_f32(0.f);
                    float32x4_t nacc1 = vdupq_n_f32(0.f);
                    size_t d = 0;
                    for (; d + 16 <= head_dim; d += 16) {
                        float16x8_t h0 = vld1q_f16(q_vec + d);
                        float16x8_t h1 = vld1q_f16(q_vec + d + 8);
                        float32x4_t a  = vcvt_f32_f16(vget_low_f16(h0));
                        float32x4_t b  = vcvt_f32_f16(vget_high_f16(h0));
                        float32x4_t c  = vcvt_f32_f16(vget_low_f16(h1));
                        float32x4_t e  = vcvt_f32_f16(vget_high_f16(h1));
                        vst1q_f32(q_rot_f32 + d,      a);
                        vst1q_f32(q_rot_f32 + d + 4,  b);
                        vst1q_f32(q_rot_f32 + d + 8,  c);
                        vst1q_f32(q_rot_f32 + d + 12, e);
                        nacc0 = vfmaq_f32(vfmaq_f32(nacc0, a, a), b, b);
                        nacc1 = vfmaq_f32(vfmaq_f32(nacc1, c, c), e, e);
                    }
                    for (; d + 8 <= head_dim; d += 8) {
                        float16x8_t h  = vld1q_f16(q_vec + d);
                        float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                        float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                        vst1q_f32(q_rot_f32 + d,     lo);
                        vst1q_f32(q_rot_f32 + d + 4, hi);
                        nacc0 = vfmaq_f32(vfmaq_f32(nacc0, lo, lo), hi, hi);
                    }
                    float norm_sq = vaddvq_f32(vaddq_f32(nacc0, nacc1));
                    for (; d < head_dim; d++) {
                        float v = static_cast<float>(q_vec[d]);
                        q_rot_f32[d] = v;
                        norm_sq += v * v;
                    }
                    q_norm = sqrtf(norm_sq);
                }

                const bool has_cache = (cache_len > 0);
                float q_sum = 0.f;
                if (has_cache) {
                    const float inv_norm = (q_norm > 1e-10f) ? 1.f / q_norm : 0.f;
                    float32x4_t inv_v = vdupq_n_f32(inv_norm);
                    size_t d = 0;
                    for (; d + 16 <= head_dim; d += 16) {
                        vst1q_f32(q_rot_f32 + d,      vmulq_f32(vld1q_f32(q_rot_f32 + d),      inv_v));
                        vst1q_f32(q_rot_f32 + d + 4,  vmulq_f32(vld1q_f32(q_rot_f32 + d + 4),  inv_v));
                        vst1q_f32(q_rot_f32 + d + 8,  vmulq_f32(vld1q_f32(q_rot_f32 + d + 8),  inv_v));
                        vst1q_f32(q_rot_f32 + d + 12, vmulq_f32(vld1q_f32(q_rot_f32 + d + 12), inv_v));
                    }
                    for (; d < head_dim; d++) q_rot_f32[d] *= inv_norm;

                    rotate_forward(q_rot_f32, rotation_signs, head_dim);

                    {
                        float32x4_t sacc0 = vdupq_n_f32(0.f);
                        float32x4_t sacc1 = vdupq_n_f32(0.f);
                        d = 0;
                        for (; d + 8 <= head_dim; d += 8) {
                            sacc0 = vaddq_f32(sacc0, vld1q_f32(q_rot_f32 + d));
                            sacc1 = vaddq_f32(sacc1, vld1q_f32(q_rot_f32 + d + 4));
                        }
                        q_sum = vaddvq_f32(vaddq_f32(sacc0, sacc1));
                        for (; d < head_dim; d++) q_sum += q_rot_f32[d];
                    }

                }

                memset(cached_acc, 0, head_dim * sizeof(float));
                memset(new_acc,    0, head_dim * sizeof(float));
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

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
                            const size_t ci = batch_idx * cache_batch_stride
                                            + kv_pos * num_kv_heads + kv_head_idx;
                            const float radius   = cached_key_radii[ci];
                            float polar_dot;
                            if (angle_bits == 2) {
                                polar_dot = dot_2bit_f32(q_rot_f32, q_sum,
                                                         cached_key_angles + ci * key_angles_bytes,
                                                         head_dim) * radius;
                            } else if (angle_bits == 6) {
                                polar_dot = dot_6bit_f32(q_rot_f32,
                                                         cached_key_angles + ci * key_angles_bytes,
                                                         head_dim) * radius;
                            } else if (angle_bits == 8) {
                                polar_dot = dot_8bit_f32(q_rot_f32,
                                                         cached_key_angles + ci * key_angles_bytes,
                                                         head_dim) * radius;
                            } else {
                                polar_dot = dot_4bit_f32(q_rot_f32,
                                                         cached_key_angles + ci * key_angles_bytes,
                                                         head_dim) * radius;
                            }
                            float score = q_norm * polar_dot * scale;
                            block_scores[kv_idx] = score;
                            block_max = std::max(block_max, score);
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

                            float score = vaddvq_f32(vaddq_f32(score_accum_low, score_accum_high)) * scale;
                            block_scores[kv_idx] = score;
                            block_max = std::max(block_max, score);
                        }
                    }

                    if (block_max > -std::numeric_limits<float>::infinity()) {
                        float scale_correction = expf(running_max - block_max);
                        running_sum *= scale_correction;

                        float32x4_t sv = vdupq_n_f32(scale_correction);
                        for (size_t d = 0; d + 16 <= head_dim; d += 16) {
                            vst1q_f32(cached_acc + d,      vmulq_f32(vld1q_f32(cached_acc + d),      sv));
                            vst1q_f32(cached_acc + d + 4,  vmulq_f32(vld1q_f32(cached_acc + d + 4),  sv));
                            vst1q_f32(cached_acc + d + 8,  vmulq_f32(vld1q_f32(cached_acc + d + 8),  sv));
                            vst1q_f32(cached_acc + d + 12, vmulq_f32(vld1q_f32(cached_acc + d + 12), sv));
                            vst1q_f32(new_acc + d,         vmulq_f32(vld1q_f32(new_acc + d),         sv));
                            vst1q_f32(new_acc + d + 4,     vmulq_f32(vld1q_f32(new_acc + d + 4),     sv));
                            vst1q_f32(new_acc + d + 8,     vmulq_f32(vld1q_f32(new_acc + d + 8),     sv));
                            vst1q_f32(new_acc + d + 12,    vmulq_f32(vld1q_f32(new_acc + d + 12),    sv));
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
                            const size_t ci = batch_idx * cache_batch_stride
                                            + kv_pos * num_kv_heads + kv_head_idx;
                            if (value_angle_bits == 2) {
                                accumulate_2bit_f32(
                                    cached_value_angles + ci * val_angles_bytes,
                                    attn_weight * cached_value_radii[ci],
                                    cached_acc, head_dim);
                            } else if (value_angle_bits == 6) {
                                accumulate_6bit_f32(
                                    cached_value_angles + ci * val_angles_bytes,
                                    attn_weight * cached_value_radii[ci],
                                    cached_acc, head_dim);
                            } else if (value_angle_bits == 8) {
                                accumulate_8bit_f32(
                                    cached_value_angles + ci * val_angles_bytes,
                                    attn_weight * cached_value_radii[ci],
                                    cached_acc, head_dim);
                            } else {
                                accumulate_4bit_f32(
                                    cached_value_angles + ci * val_angles_bytes,
                                    attn_weight * cached_value_radii[ci],
                                    cached_acc, head_dim);
                            }
                        } else {
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* v_vec = V_new_base + new_pos * kv_seq_stride + kv_head_idx * head_dim;

                            for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                                float16x8_t v_vec_f16 = vld1q_f16(&v_vec[dim_block]);
                                float32x4_t v_low = vcvt_f32_f16(vget_low_f16(v_vec_f16));
                                float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v_vec_f16));

                                vst1q_f32(new_acc + dim_block,
                                    vfmaq_f32(vld1q_f32(new_acc + dim_block), v_low, weight_vec));
                                vst1q_f32(new_acc + dim_block + 4,
                                    vfmaq_f32(vld1q_f32(new_acc + dim_block + 4), v_high, weight_vec));
                            }
                        }
                    }

                    running_sum += block_sum;
                }

                if (running_sum > 0.0f) {
                    if (has_cache) {
                        rotate_inverse(cached_acc, rotation_signs, head_dim);
                    }

                    const float inv_sum = 1.0f / running_sum;
                    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

                    for (size_t dim_block = 0; dim_block < head_dim_aligned; dim_block += VECTOR_WIDTH) {
                        float32x4_t final_low = vmulq_f32(
                            vaddq_f32(vld1q_f32(cached_acc + dim_block), vld1q_f32(new_acc + dim_block)), inv_sum_vec);
                        float32x4_t final_high = vmulq_f32(
                            vaddq_f32(vld1q_f32(cached_acc + dim_block + 4), vld1q_f32(new_acc + dim_block + 4)), inv_sum_vec);

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
