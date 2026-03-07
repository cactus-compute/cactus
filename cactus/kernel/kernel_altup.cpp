#include "kernel.h"
#include <arm_neon.h>
#include <cstddef>
#include <cmath>

void cactus_gaussian_topk_f16(
    const __fp16* input,
    __fp16* output,
    size_t rows,
    size_t cols,
    float ppf
) {
    static constexpr float MAX_SAFE_ABS = 240.0f;

    for (size_t r = 0; r < rows; r++) {
        const __fp16* row_in = input + r * cols;
        __fp16* row_out = output + r * cols;

        // Pass 1: compute mean and max absolute deviation
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t d = 0;
        for (; d + 8 <= cols; d += 8) {
            float16x8_t v = vld1q_f16(row_in + d);
            sum_vec = vaddq_f32(sum_vec, vcvt_f32_f16(vget_low_f16(v)));
            sum_vec = vaddq_f32(sum_vec, vcvt_f32_f16(vget_high_f16(v)));
        }
        float sum = vaddvq_f32(sum_vec);
        for (; d < cols; d++) {
            sum += static_cast<float>(row_in[d]);
        }
        float mu = sum / static_cast<float>(cols);

        float32x4_t mu_vec = vdupq_n_f32(mu);
        float32x4_t max_abs_vec = vdupq_n_f32(0.0f);
        d = 0;
        for (; d + 8 <= cols; d += 8) {
            float16x8_t v = vld1q_f16(row_in + d);
            float32x4_t lo = vsubq_f32(vcvt_f32_f16(vget_low_f16(v)), mu_vec);
            float32x4_t hi = vsubq_f32(vcvt_f32_f16(vget_high_f16(v)), mu_vec);
            max_abs_vec = vmaxq_f32(max_abs_vec, vabsq_f32(lo));
            max_abs_vec = vmaxq_f32(max_abs_vec, vabsq_f32(hi));
        }
        float max_abs = vmaxvq_f32(max_abs_vec);
        for (; d < cols; d++) {
            float diff = static_cast<float>(row_in[d]) - mu;
            float a = diff < 0.0f ? -diff : diff;
            if (a > max_abs) max_abs = a;
        }

        // Safe scaling: scale = max(max_abs / 240, 1)
        float scale = max_abs * (1.0f / MAX_SAFE_ABS);
        if (scale < 1.0f) scale = 1.0f;
        float inv_scale = 1.0f / scale;

        // Pass 2: compute variance of scaled deviations
        float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
        float32x4_t var_sum_vec = vdupq_n_f32(0.0f);
        d = 0;
        for (; d + 8 <= cols; d += 8) {
            float16x8_t v = vld1q_f16(row_in + d);
            float32x4_t lo = vmulq_f32(vsubq_f32(vcvt_f32_f16(vget_low_f16(v)), mu_vec), inv_scale_vec);
            float32x4_t hi = vmulq_f32(vsubq_f32(vcvt_f32_f16(vget_high_f16(v)), mu_vec), inv_scale_vec);
            var_sum_vec = vfmaq_f32(var_sum_vec, lo, lo);
            var_sum_vec = vfmaq_f32(var_sum_vec, hi, hi);
        }
        float var_sum = vaddvq_f32(var_sum_vec);
        for (; d < cols; d++) {
            float ds = (static_cast<float>(row_in[d]) - mu) * inv_scale;
            var_sum += ds * ds;
        }
        float variance = var_sum / static_cast<float>(cols);
        float sigma = sqrtf(variance) * scale;

        // cutoff = mu + ppf * sigma, output = relu(input - cutoff)
        float cutoff = mu + ppf * sigma;
        float32x4_t cutoff_vec = vdupq_n_f32(cutoff);
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        d = 0;
        for (; d + 8 <= cols; d += 8) {
            float16x8_t v = vld1q_f16(row_in + d);
            float32x4_t lo = vmaxq_f32(vsubq_f32(vcvt_f32_f16(vget_low_f16(v)), cutoff_vec), zero_vec);
            float32x4_t hi = vmaxq_f32(vsubq_f32(vcvt_f32_f16(vget_high_f16(v)), cutoff_vec), zero_vec);
            vst1q_f16(row_out + d, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
        }
        for (; d < cols; d++) {
            float val = static_cast<float>(row_in[d]) - cutoff;
            row_out[d] = static_cast<__fp16>(val > 0.0f ? val : 0.0f);
        }
    }
}

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
