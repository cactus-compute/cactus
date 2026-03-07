#include "kernel.h"
#include <arm_neon.h>
#include <cstddef>

void cactus_altup_predict_f16(
    const __fp16* coefs,
    const __fp16* const* streams,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim
) {
    const size_t coef_stride = n * n;

    for (size_t i = 0; i < n; i++) {
        for (size_t s = 0; s < seq_len; s++) {
            const __fp16* coef_row = coefs + s * coef_stride + i * n;
            __fp16* out_row = output + (i * seq_len + s) * hidden_dim;
            const __fp16* src_i = streams[i] + s * hidden_dim;

            float c[8];
            for (size_t j = 0; j < n; j++) {
                c[j] = static_cast<float>(coef_row[j]);
            }

            size_t d = 0;
            for (; d + 8 <= hidden_dim; d += 8) {
                float32x4_t acc_lo = vcvt_f32_f16(vget_low_f16(vld1q_f16(src_i + d)));
                float32x4_t acc_hi = vcvt_f32_f16(vget_high_f16(vld1q_f16(src_i + d)));

                for (size_t j = 0; j < n; j++) {
                    float16x8_t sj = vld1q_f16(streams[j] + s * hidden_dim + d);
                    float32x4_t cj = vdupq_n_f32(c[j]);
                    acc_lo = vfmaq_f32(acc_lo, vcvt_f32_f16(vget_low_f16(sj)), cj);
                    acc_hi = vfmaq_f32(acc_hi, vcvt_f32_f16(vget_high_f16(sj)), cj);
                }

                vst1q_f16(out_row + d, vcombine_f16(vcvt_f16_f32(acc_lo), vcvt_f16_f32(acc_hi)));
            }
            for (; d < hidden_dim; d++) {
                float acc = static_cast<float>(src_i[d]);
                for (size_t j = 0; j < n; j++) {
                    acc += c[j] * static_cast<float>(streams[j][s * hidden_dim + d]);
                }
                out_row[d] = static_cast<__fp16>(acc);
            }
        }
    }
}

void cactus_altup_correct_f16(
    const __fp16* coefs,
    const __fp16* innovation,
    const __fp16* const* predictions,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t s = 0; s < seq_len; s++) {
            float ci = static_cast<float>(coefs[s * n + i]);
            float32x4_t ci_vec = vdupq_n_f32(ci);

            const __fp16* pred_row = predictions[i] + s * hidden_dim;
            const __fp16* innov_row = innovation + s * hidden_dim;
            __fp16* out_row = output + (i * seq_len + s) * hidden_dim;

            size_t d = 0;
            for (; d + 8 <= hidden_dim; d += 8) {
                float16x8_t pred = vld1q_f16(pred_row + d);
                float16x8_t innov = vld1q_f16(innov_row + d);

                float32x4_t p_lo = vcvt_f32_f16(vget_low_f16(pred));
                float32x4_t p_hi = vcvt_f32_f16(vget_high_f16(pred));
                float32x4_t i_lo = vcvt_f32_f16(vget_low_f16(innov));
                float32x4_t i_hi = vcvt_f32_f16(vget_high_f16(innov));

                float32x4_t out_lo = vfmaq_f32(p_lo, i_lo, ci_vec);
                float32x4_t out_hi = vfmaq_f32(p_hi, i_hi, ci_vec);

                vst1q_f16(out_row + d, vcombine_f16(vcvt_f16_f32(out_lo), vcvt_f16_f32(out_hi)));
            }
            for (; d < hidden_dim; d++) {
                out_row[d] = static_cast<__fp16>(
                    static_cast<float>(pred_row[d]) + ci * static_cast<float>(innov_row[d])
                );
            }
        }
    }
}
