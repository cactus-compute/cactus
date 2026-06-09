#include "../cactus_kernels.h"
#include "threading.h"
#include "matmul_simd.h"
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

// ── SME2 prefill attention over the cached-int8 KV segment ──────────────────────────────────
// Two-pass design: when position_offset >= cache_len the cached segment is FULLY causally visible
// to every prefill query row (kv_end = cache_len + q_pos + 1 > cache_len), so no online softmax is
// needed there. Per 16-query-row tile: (1) SME SMOPA computes ALL cached QK scores (64-kv blocks,
// raw int32 partials, fp rescale in NEON outside streaming mode), (2) one masked softmax pass over
// the whole cached row fixes GLOBAL per-(row, v-scale-group) P quantization scales, (3) SME SMOPA
// AV accumulates across ALL cached blocks inside ZA with a single readout per 64-dim slice — the
// per-(kv, 32-dim-group) V scales cannot factor out of the kv contraction, so they are folded into
// the int8 P operand (two pack variants per slice; see docs/sme/working-examples/avprobe.cpp).
// The fp16 keys_new/values_new segment then continues per row through the incumbent flash-softmax
// loop seeded from the cached state (flash softmax is partition-invariant). Window/sink masking
// replicates the incumbent's cached-segment semantics exactly (contiguous masked range per row).
static void cactus_attention_sme_prefill(
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
    constexpr size_t QTILE = 16;
    constexpr size_t KVBLK = 64;
    constexpr size_t SINK_SIZE = 4;
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    const size_t kv_seq_len = cache_len + new_len;
    const size_t n_qgroups = head_dim / quant_group_size;
    const size_t dim_groups = quant_group_size / 4;
    const size_t num_passes = head_dim / KVBLK;
    const size_t num_blocks = (cache_len + KVBLK - 1) / KVBLK;
    const size_t gqa_group_size = num_q_heads / num_kv_heads;
    const size_t num_qtiles = (seq_len + QTILE - 1) / QTILE;
    const size_t num_accum_slots = head_dim / 8;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t k_cached_batch_stride = cache_len * num_kv_heads * head_dim;
    const size_t v_cached_batch_stride = cache_len * num_kv_heads * head_dim;
    const size_t k_new_batch_stride = new_len * num_kv_heads * head_dim;
    const size_t v_new_batch_stride = new_len * num_kv_heads * head_dim;
    const size_t o_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t k_seq_stride = num_kv_heads * head_dim;
    const size_t v_seq_stride = num_kv_heads * head_dim;
    const size_t o_seq_stride = num_q_heads * head_dim;

    const size_t cache_abs_offset = position_offset - cache_len;   // gate guarantees >= 0

    // Shared packed-KV scratch, built once per call and reused by every q-head tile in the GQA
    // group: SMOPA operand layouts for K (per 64-kv block) and V (per 64-dim slice), plus
    // kv-transposed scale planes [g][64] for vectorized rescale / P folding. Tail kv beyond
    // cache_len pack as zeros with zero scales (their scores are forced to -inf below).
    const size_t BH = batch_size * num_kv_heads;
    const size_t kpack_blk = head_dim * KVBLK;
    const size_t vpack_pass_blk = KVBLK * KVBLK;
    const size_t scale_blk = n_qgroups * KVBLK;
    std::vector<int8_t> kpack_all(BH * num_blocks * kpack_blk);
    std::vector<int8_t> vpack_all(BH * num_passes * num_blocks * vpack_pass_blk);
    std::vector<float> kst_all(BH * num_blocks * KVBLK);     // flat per-kv K scale
    std::vector<float> vst_all(BH * num_blocks * scale_blk);

    const size_t total_pack_items = BH * num_blocks;
    const size_t n_pack_slots = std::min(total_pack_items,
        CactusThreading::get_thread_pool().num_workers());
    CactusThreading::parallel_for(n_pack_slots, CactusThreading::ParallelConfig{1, 1},
        [&](size_t slot_start, size_t slot_end) {
            alignas(16) int8_t zero_row[512] = {0};
            alignas(16) int8_t krq[512];
            for (size_t slot = slot_start; slot < slot_end; ++slot)
            for (size_t item = slot; item < total_pack_items; item += n_pack_slots) {
                const size_t bh = item / num_blocks;
                const size_t blk = item % num_blocks;
                const size_t batch_idx = bh / num_kv_heads;
                const size_t kv_head_idx = bh % num_kv_heads;
                const int8_t* K_base = keys_cached + batch_idx * k_cached_batch_stride;
                const int8_t* V_base = values_cached + batch_idx * v_cached_batch_stride;
                int8_t* kp = kpack_all.data() + (bh * num_blocks + blk) * kpack_blk;
                float* kst = kst_all.data() + (bh * num_blocks + blk) * KVBLK;
                float* vst = vst_all.data() + (bh * num_blocks + blk) * scale_blk;

                // K: requantize each kv row to ONE flat scale (ksflat = max_g ks[g]; per-group
                // ratio folded into the int8 values — costs <=1 bit on low-scale groups, buys the
                // single-readout QK leaf), then gather into [dim-quad][vec(c/16)][c%16][4]
                const size_t words = head_dim / 4;
                for (size_t c = 0; c < KVBLK; ++c) {
                    const size_t kvp = blk * KVBLK + c;
                    const bool valid = kvp < cache_len;
                    const int8_t* krow = zero_row;
                    if (valid) {
                        const int8_t* ksrc = K_base + kvp * k_seq_stride + kv_head_idx * head_dim;
                        const float* ks = k_scales + (kvp * num_kv_heads + kv_head_idx) * n_qgroups;
                        const float* vs = v_scales + (kvp * num_kv_heads + kv_head_idx) * n_qgroups;
                        float ksflat = 0.0f;
                        for (size_t g = 0; g < n_qgroups; ++g) ksflat = std::max(ksflat, ks[g]);
                        kst[c] = ksflat;
                        const float kinv = ksflat > 0.0f ? 1.0f / ksflat : 0.0f;
                        for (size_t g = 0; g < n_qgroups; ++g) {
                            vst[g * KVBLK + c] = vs[g];
                            const float32x4_t rho = vdupq_n_f32(ks[g] * kinv);
                            for (size_t i = g * quant_group_size; i < (g + 1) * quant_group_size; i += 8) {
                                int16x8_t k16 = vmovl_s8(vld1_s8(ksrc + i));
                                float32x4_t lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(k16))), rho);
                                float32x4_t hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(k16))), rho);
                                int16x4_t l16 = vqmovn_s32(vcvtnq_s32_f32(lo));
                                int16x4_t h16 = vqmovn_s32(vcvtnq_s32_f32(hi));
                                vst1_s8(krq + i, vqmovn_s16(vcombine_s16(l16, h16)));
                            }
                        }
                        krow = krq;
                    } else {
                        kst[c] = 0.0f;
                        for (size_t g = 0; g < n_qgroups; ++g) vst[g * KVBLK + c] = 0.0f;
                    }
                    int8_t* dstc = kp + (c >> 4) * 64 + (c & 15) * 4;
                    for (size_t w = 0; w < words; ++w)
                        memcpy(dstc + w * 256, krow + w * 4, 4);
                }

                // V -> per 64-dim slice: [kg][tile][16 dims][4 kv] via 4-way byte interleave
                for (size_t kg = 0; kg < 16; ++kg) {
                    const int8_t* vrow[4];
                    for (size_t c4 = 0; c4 < 4; ++c4) {
                        const size_t kvp = blk * KVBLK + kg * 4 + c4;
                        vrow[c4] = (kvp < cache_len)
                            ? V_base + kvp * v_seq_stride + kv_head_idx * head_dim : zero_row;
                    }
                    for (size_t pass = 0; pass < num_passes; ++pass) {
                        int8_t* vp = vpack_all.data() +
                            ((bh * num_passes + pass) * num_blocks + blk) * vpack_pass_blk + kg * 256;
                        for (size_t t = 0; t < 4; ++t) {
                            const size_t off = pass * 64 + t * 16;
                            int8x16x4_t q;
                            q.val[0] = vld1q_s8(vrow[0] + off);
                            q.val[1] = vld1q_s8(vrow[1] + off);
                            q.val[2] = vld1q_s8(vrow[2] + off);
                            q.val[3] = vld1q_s8(vrow[3] + off);
                            vst4q_s8(vp + t * 64, q);
                        }
                    }
                }
            }
        });

    // Strided round-robin over tiles: parallel_for's static splitter floors work_per_thread and
    // dumps the remainder on the LAST worker (64 tiles / 14 workers -> 13x4 + 1x12, a 3x straggler
    // measured as ~60% pool idle). Each pseudo-item below is one worker slot; workers walk items
    // slot, slot+nw, ... for +-1 balance (and unequal-cost windowed tiles get mixed).
    const size_t total_tiles = batch_size * num_q_heads * num_qtiles;
    const size_t n_slots = std::min(total_tiles,
        CactusThreading::get_thread_pool().num_workers());
    CactusThreading::parallel_for(n_slots,
        CactusThreading::ParallelConfig{1, 1},
        [&](size_t slot_start, size_t slot_end) {
            std::vector<int8_t> qpack(n_qgroups * dim_groups * 64);
            std::vector<float> qs_scaled(QTILE);
            std::vector<int32_t> qk_partials(num_blocks * 16 * 64);
            const size_t srow_stride = num_blocks * KVBLK;
            std::vector<float> scores(QTILE * srow_stride);
            std::vector<float> m_c(QTILE), sum_c(QTILE);
            std::vector<float> ps_blk(QTILE * n_qgroups * num_blocks);
            std::vector<float> inv_ps_blk(QTILE * n_qgroups * num_blocks);
            std::vector<uint8_t> ppack(num_blocks * 2048);
            std::vector<int32_t> av_out(num_blocks * 16 * 64);
            std::vector<float> oacc(QTILE * head_dim);
            size_t mask_lo[QTILE], mask_hi[QTILE];
            std::vector<float32x4_t> output_accum_low(num_accum_slots);
            std::vector<float32x4_t> output_accum_high(num_accum_slots);
            std::vector<float16x8_t> block_accum(num_accum_slots);
            float block_scores[32];

            for (size_t slot = slot_start; slot < slot_end; ++slot)
            for (size_t work_idx = slot; work_idx < total_tiles; work_idx += n_slots) {
                const size_t batch_idx = work_idx / (num_q_heads * num_qtiles);
                const size_t rem = work_idx % (num_q_heads * num_qtiles);
                const size_t q_head_idx = rem / num_qtiles;
                const size_t tile_idx = rem % num_qtiles;
                const size_t kv_head_idx = q_head_idx / gqa_group_size;
                const size_t bh = batch_idx * num_kv_heads + kv_head_idx;
                const size_t tile0 = tile_idx * QTILE;
                const size_t rows = std::min(QTILE, seq_len - tile0);

                const __fp16* Q_base = queries + batch_idx * q_batch_stride;
                const __fp16* K_new_base = keys_new + batch_idx * k_new_batch_stride;
                const __fp16* V_new_base = values_new + batch_idx * v_new_batch_stride;
                __fp16* O_base = output + batch_idx * o_batch_stride;
                const int8_t* kpack_bh = kpack_all.data() + bh * num_blocks * kpack_blk;
                const float* kst_bh = kst_all.data() + bh * num_blocks * KVBLK;
                const float* vst_bh = vst_all.data() + bh * num_blocks * scale_blk;

                // 1) quantize + pack Q with ONE scale per row (flat scales let the QK leaf
                //    accumulate the whole head_dim in ZA); `scale` folds into the rescale factor
                memset(qpack.data(), 0, qpack.size());
                for (size_t r = 0; r < rows; ++r) {
                    const __fp16* q_vec = Q_base + (tile0 + r) * q_seq_stride + q_head_idx * head_dim;
                    float32x4_t amx = vdupq_n_f32(0.0f);
                    for (size_t i = 0; i < head_dim; i += 8) {
                        float16x8_t v = vld1q_f16(q_vec + i);
                        amx = vmaxq_f32(amx, vmaxq_f32(
                            vabsq_f32(vcvt_f32_f16(vget_low_f16(v))),
                            vabsq_f32(vcvt_f32_f16(vget_high_f16(v)))));
                    }
                    const float amax = vmaxvq_f32(amx);
                    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                    qs_scaled[r] = scale * (amax / 127.0f);
                    const float32x4_t invv = vdupq_n_f32(inv);
                    int8_t* qdst = qpack.data() + r * 4;
                    for (size_t i = 0; i < head_dim; i += 8) {
                        float16x8_t v = vld1q_f16(q_vec + i);
                        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v)), invv));
                        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v)), invv));
                        int8x8_t s8 = vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
                        vst1_lane_s32(reinterpret_cast<int32_t*>(qdst + (i / 4) * 64),
                                      vreinterpret_s32_s8(s8), 0);
                        vst1_lane_s32(reinterpret_cast<int32_t*>(qdst + (i / 4 + 1) * 64),
                                      vreinterpret_s32_s8(s8), 1);
                    }
                }

                // 2) per-row window-mask ranges over the cached segment (incumbent semantics:
                //    masked iff (cache_abs_offset==0 || c >= SINK_SIZE) && cache_abs_offset+c < kv_start
                //    => one contiguous masked range [lo, hi) per row)
                size_t skip_lo_max = 0, skip_hi_min = 0;
                bool any_mask = false;
                for (size_t r = 0; r < rows; ++r) {
                    const size_t absolute_q_pos = position_offset + tile0 + r;
                    size_t kv_start = 0;
                    if (window_size > 0 && absolute_q_pos > window_size)
                        kv_start = absolute_q_pos - window_size;
                    size_t lo = 0, hi = 0;
                    if (window_size > 0 && kv_start > 0) {
                        lo = (cache_abs_offset > 0) ? SINK_SIZE : 0;
                        hi = (kv_start > cache_abs_offset) ? kv_start - cache_abs_offset : 0;
                        hi = std::min(hi, cache_len);
                        if (hi < lo) hi = lo;
                    }
                    mask_lo[r] = lo;
                    mask_hi[r] = hi;
                    if (r == 0) { skip_lo_max = lo; skip_hi_min = hi; }
                    else {
                        skip_lo_max = std::max(skip_lo_max, lo);
                        skip_hi_min = std::min(skip_hi_min, hi);
                    }
                    if (hi > lo) any_mask = true;
                }

                // 3) SME QK over the cached segment -> fp32 scores. The fully-masked block run is
                //    contiguous, so the leaf runs once on the prefix and once on the suffix.
                size_t skipA = num_blocks, skipB = num_blocks;
                if (any_mask) {
                    const size_t a = (skip_lo_max + KVBLK - 1) / KVBLK;
                    const size_t b = skip_hi_min / KVBLK;
                    if (a < b) { skipA = a; skipB = b; }
                }
                if (skipA > 0)
                    cactus_sme_attn_qk_seg(qpack.data(), kpack_bh, qk_partials.data(),
                                           (uint32_t)(head_dim / 4), (uint32_t)skipA);
                if (skipB < num_blocks)
                    cactus_sme_attn_qk_seg(qpack.data(), kpack_bh + skipB * kpack_blk,
                                           qk_partials.data() + skipB * 1024,
                                           (uint32_t)(head_dim / 4), (uint32_t)(num_blocks - skipB));
                for (size_t blk = 0; blk < num_blocks; ++blk) {
                    const size_t c0 = blk * KVBLK;
                    if (blk >= skipA && blk < skipB) {
                        for (size_t r = 0; r < rows; ++r) {
                            float* srow = scores.data() + r * srow_stride + c0;
                            for (size_t c = 0; c < KVBLK; ++c) srow[c] = NEG_INF;
                        }
                        continue;
                    }
                    const float* kstf = kst_bh + blk * KVBLK;
                    for (size_t r = 0; r < rows; ++r) {
                        float* srow = scores.data() + r * srow_stride + c0;
                        const float32x4_t qsv = vdupq_n_f32(qs_scaled[r]);
                        const int32_t* p64 = qk_partials.data() + blk * 1024 + r * 64;
                        for (size_t c = 0; c < KVBLK; c += 4)
                            vst1q_f32(srow + c, vmulq_f32(
                                vmulq_f32(vcvtq_f32_s32(vld1q_s32(p64 + c)), qsv),
                                vld1q_f32(kstf + c)));
                    }
                }
                for (size_t r = 0; r < rows; ++r) {
                    float* srow = scores.data() + r * srow_stride;
                    for (size_t c = cache_len; c < srow_stride; ++c) srow[c] = NEG_INF;
                    for (size_t c = mask_lo[r]; c < mask_hi[r]; ++c) srow[c] = NEG_INF;
                }

                // 4) masked softmax stats per row over the whole cached segment (exp in place)
                for (size_t r = 0; r < rows; ++r) {
                    float* srow = scores.data() + r * srow_stride;
                    float32x4_t mv = vdupq_n_f32(NEG_INF);
                    for (size_t c = 0; c < srow_stride; c += 4)
                        mv = vmaxq_f32(mv, vld1q_f32(srow + c));
                    const float m = vmaxvq_f32(mv);
                    m_c[r] = m;
                    if (!(m > NEG_INF)) {
                        sum_c[r] = 0.0f;
                        memset(srow, 0, srow_stride * sizeof(float));
                        continue;
                    }
                    // vectorized exp (scalar expf here is ~half the incumbent's attention core
                    // time); masked -inf lanes clamp to exp(-87) ~ 1.6e-38, which is negligible
                    // in the sum and rounds to 0 in the u8 P quantization — no branch needed
                    const float32x4_t mv4 = vdupq_n_f32(m);
                    float32x4_t sumv = vdupq_n_f32(0.0f);
                    for (size_t c = 0; c < srow_stride; c += 4) {
                        const float32x4_t e = fast_exp_f32x4(vsubq_f32(vld1q_f32(srow + c), mv4));
                        vst1q_f32(srow + c, e);
                        sumv = vaddq_f32(sumv, e);
                    }
                    sum_c[r] = vaddvq_f32(sumv);
                }

                // 5) per-(row, v-group, BLOCK) P quantization scales: ps = max_c(P*vs)/255 over
                //    each 64-kv block — per-block scales keep full u8 resolution in blocks far
                //    below the row max (peaked attention), at the cost of a per-block AV readout
                for (size_t r = 0; r < rows; ++r) {
                    const float* prow = scores.data() + r * srow_stride;
                    for (size_t g = 0; g < n_qgroups; ++g) {
                        float* psd = ps_blk.data() + (r * n_qgroups + g) * num_blocks;
                        float* ipsd = inv_ps_blk.data() + (r * n_qgroups + g) * num_blocks;
                        for (size_t blk = 0; blk < num_blocks; ++blk) {
                            const float* vs64 = vst_bh + blk * scale_blk + g * KVBLK;
                            const float* p64 = prow + blk * KVBLK;
                            float32x4_t mx = vdupq_n_f32(0.0f);
                            for (size_t c = 0; c < KVBLK; c += 4)
                                mx = vmaxq_f32(mx, vmulq_f32(vld1q_f32(p64 + c), vld1q_f32(vs64 + c)));
                            const float m = vmaxvq_f32(mx);
                            psd[blk] = m / 255.0f;
                            ipsd[blk] = m > 0.0f ? 255.0f / m : 0.0f;
                        }
                    }
                }

                // 6) SME AV: one pass per 64-dim slice; P folded with vs and the per-block scale,
                //    u8-quantized (USMOPA); per-block int32 tiles rescaled+accumulated in fp32
                for (size_t pass = 0; pass < num_passes; ++pass) {
                    memset(ppack.data(), 0, ppack.size());
                    for (size_t grp = 0; grp < 2; ++grp) {
                        const size_t g = pass * 2 + grp;
                        for (size_t r = 0; r < rows; ++r) {
                            const float* prow = scores.data() + r * srow_stride;
                            const float* ipsd = inv_ps_blk.data() + (r * n_qgroups + g) * num_blocks;
                            for (size_t blk = 0; blk < num_blocks; ++blk) {
                                const float32x4_t invv = vdupq_n_f32(ipsd[blk]);
                                const float* vs64 = vst_bh + blk * scale_blk + g * KVBLK;
                                const float* p64 = prow + blk * KVBLK;
                                uint8_t* pdst = ppack.data() + blk * 2048 + grp * 1024 + r * 4;
                                for (size_t kg = 0; kg < 16; ++kg) {
                                    float32x4_t f = vmulq_f32(vmulq_f32(
                                        vld1q_f32(p64 + kg * 4), vld1q_f32(vs64 + kg * 4)), invv);
                                    int16x4_t i16 = vqmovn_s32(vcvtnq_s32_f32(f));
                                    uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, i16));
                                    vst1_lane_u32(reinterpret_cast<uint32_t*>(pdst + kg * 64),
                                                  vreinterpret_u32_u8(u8), 0);
                                }
                            }
                        }
                    }
                    cactus_sme_attn_av_pass(ppack.data(),
                        vpack_all.data() + (bh * num_passes + pass) * num_blocks * vpack_pass_blk,
                        av_out.data(), (uint32_t)num_blocks);
                    for (size_t r = 0; r < rows; ++r) {
                        for (size_t grp = 0; grp < 2; ++grp) {
                            const size_t g = pass * 2 + grp;
                            const float* psd = ps_blk.data() + (r * n_qgroups + g) * num_blocks;
                            float32x4_t acc[8];
                            for (size_t i = 0; i < 8; ++i) acc[i] = vdupq_n_f32(0.0f);
                            for (size_t blk = 0; blk < num_blocks; ++blk) {
                                const float32x4_t psv = vdupq_n_f32(psd[blk]);
                                const int32_t* src = av_out.data() + blk * 1024 + r * 64 + grp * 32;
                                for (size_t d = 0; d < 32; d += 4)
                                    acc[d / 4] = vfmaq_f32(acc[d / 4],
                                        vcvtq_f32_s32(vld1q_s32(src + d)), psv);
                            }
                            float* dst = oacc.data() + r * head_dim + pass * 64 + grp * 32;
                            for (size_t d = 0; d < 32; d += 4) vst1q_f32(dst + d, acc[d / 4]);
                        }
                    }
                }

                // 7) per row: seed the flash state from the cached segment, then the incumbent
                //    loop over the fp16 new segment [cache_len, kv_end)
                for (size_t r = 0; r < rows; ++r) {
                    const size_t q_pos = tile0 + r;
                    const size_t absolute_q_pos = position_offset + q_pos;
                    const __fp16* q_vec = Q_base + q_pos * q_seq_stride + q_head_idx * head_dim;
                    __fp16* o_vec = O_base + q_pos * o_seq_stride + q_head_idx * head_dim;

                    float running_max = m_c[r];
                    float running_sum = sum_c[r];
                    const float* orow = oacc.data() + r * head_dim;
                    for (size_t i = 0; i < num_accum_slots; ++i) {
                        output_accum_low[i] = vld1q_f32(orow + i * 8);
                        output_accum_high[i] = vld1q_f32(orow + i * 8 + 4);
                    }

                    size_t kv_start = 0;
                    if (window_size > 0 && absolute_q_pos > window_size)
                        kv_start = absolute_q_pos - window_size;
                    const size_t kv_end = is_causal
                        ? std::min(kv_seq_len, cache_len + q_pos + 1) : kv_seq_len;

                    for (size_t kv_block_start = cache_len; kv_block_start < kv_end; kv_block_start += 32) {
                        const size_t kv_block_end = std::min(kv_block_start + 32, kv_end);
                        const size_t block_size = kv_block_end - kv_block_start;

                        float block_max = NEG_INF;
                        for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                            const size_t kv_pos = kv_block_start + kv_idx;
                            bool window_masked = false;
                            if (window_size > 0 && kv_start > 0)
                                window_masked = (kv_pos + cache_abs_offset < kv_start);
                            if ((is_causal && kv_pos > absolute_q_pos) || window_masked) {
                                block_scores[kv_idx] = NEG_INF;
                                continue;
                            }
                            const size_t new_pos = kv_pos - cache_len;
                            const __fp16* k_vec = K_new_base + new_pos * k_seq_stride + kv_head_idx * head_dim;
                            float16x8_t s_acc = vdupq_n_f16((__fp16)0.0f);
                            for (size_t d = 0; d < head_dim; d += 8)
                                s_acc = vfmaq_f16(s_acc, vld1q_f16(q_vec + d), vld1q_f16(k_vec + d));
                            float score = vaddvq_f32(vcvt_f32_f16(vget_low_f16(s_acc))) +
                                          vaddvq_f32(vcvt_f32_f16(vget_high_f16(s_acc)));
                            score *= scale;
                            block_scores[kv_idx] = score;
                            block_max = std::max(block_max, score);
                        }

                        if (block_max > NEG_INF) {
                            const float scale_correction = expf(running_max - block_max);
                            running_sum *= scale_correction;
                            for (size_t i = 0; i < num_accum_slots; ++i) {
                                output_accum_low[i] = vmulq_n_f32(output_accum_low[i], scale_correction);
                                output_accum_high[i] = vmulq_n_f32(output_accum_high[i], scale_correction);
                            }
                            running_max = block_max;
                        }

                        float block_sum = 0.0f;
                        for (size_t kv_idx = 0; kv_idx < block_size; ++kv_idx) {
                            if (block_scores[kv_idx] != NEG_INF) {
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
                            const size_t new_pos = kv_block_start + kv_idx - cache_len;
                            const __fp16* v_vec = V_new_base + new_pos * v_seq_stride + kv_head_idx * head_dim;
                            const float16x8_t w_vec = vdupq_n_f16((__fp16)attn_weight);
                            for (size_t d = 0; d < head_dim; d += 8)
                                block_accum[d / 8] = vfmaq_f16(block_accum[d / 8], vld1q_f16(v_vec + d), w_vec);
                        }

                        for (size_t i = 0; i < num_accum_slots; ++i) {
                            output_accum_low[i] = vaddq_f32(output_accum_low[i], vcvt_f32_f16(vget_low_f16(block_accum[i])));
                            output_accum_high[i] = vaddq_f32(output_accum_high[i], vcvt_f32_f16(vget_high_f16(block_accum[i])));
                        }
                        running_sum += block_sum;
                    }

                    if (running_sum > 0.0f) {
                        const float32x4_t inv_sum_vec = vdupq_n_f32(1.0f / running_sum);
                        for (size_t d = 0; d < head_dim; d += 8) {
                            const size_t idx = d / 8;
                            vst1q_f16(o_vec + d, vcombine_f16(
                                vcvt_f16_f32(vmulq_f32(output_accum_low[idx], inv_sum_vec)),
                                vcvt_f16_f32(vmulq_f32(output_accum_high[idx], inv_sum_vec))));
                        }
                    } else {
                        memset(o_vec, 0, head_dim * sizeof(__fp16));
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

    // SME2 prefill path: cached-int8 segment via SMOPA tiles (16 q-rows x 64 kv), new fp16
    // segment via the incumbent per-row flash loop. Requires the cached segment to be fully
    // causally visible (position_offset >= cache_len, the chunked-prefill invariant) and the
    // engine's 32-wide KV quantization groups. cache_len <= 8192 bounds the packed-KV scratch
    // (~2.3 MB at Gemma shapes); larger caches fall back to the incumbent.
    if (seq_len >= 2 &&
        k_scales != nullptr && v_scales != nullptr &&
        head_dim == v_head_dim &&
        head_dim % 64 == 0 && head_dim <= 512 &&
        quant_group_size == 32 &&
        cache_len >= 64 && cache_len <= 8192 &&
        position_offset >= cache_len &&
        num_kv_heads > 0 && num_q_heads % num_kv_heads == 0 &&
        cactus_quant_sme_attn_enabled()) {
        cactus_attention_sme_prefill(
            queries, keys_cached, values_cached, k_scales, v_scales,
            keys_new, values_new, output,
            batch_size, seq_len, cache_len, new_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, position_offset, is_causal, window_size, quant_group_size);
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
