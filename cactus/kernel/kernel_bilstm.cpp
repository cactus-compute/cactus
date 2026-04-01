#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <vector>
#include <cstring>
#include <cmath>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

static void apply_lstm_gates_f32(
    const float* __restrict gates,
    float* __restrict c,
    float* __restrict h,
    size_t hidden_size
) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    size_t i = 0;

    for (; i + 4 <= hidden_size; i += 4) {
        float32x4_t i_raw = vld1q_f32(gates + i);
        float32x4_t f_raw = vld1q_f32(gates + hidden_size + i);
        float32x4_t g_raw = vld1q_f32(gates + 2 * hidden_size + i);
        float32x4_t o_raw = vld1q_f32(gates + 3 * hidden_size + i);

        float32x4_t i_act = vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(i_raw))));
        float32x4_t f_act = vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(f_raw))));
        float32x4_t g_act = fast_tanh_f32x4(g_raw);
        float32x4_t o_act = vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(o_raw))));

        float32x4_t c_prev = vld1q_f32(c + i);
        float32x4_t c_new = vfmaq_f32(vmulq_f32(f_act, c_prev), i_act, g_act);
        vst1q_f32(c + i, c_new);

        vst1q_f32(h + i, vmulq_f32(o_act, fast_tanh_f32x4(c_new)));
    }

    for (; i < hidden_size; ++i) {
        float ig = 1.0f / (1.0f + expf(-gates[i]));
        float fg = 1.0f / (1.0f + expf(-gates[hidden_size + i]));
        float gg = tanhf(gates[2 * hidden_size + i]);
        float og = 1.0f / (1.0f + expf(-gates[3 * hidden_size + i]));
        float cv = fg * c[i] + ig * gg;
        c[i] = cv;
        h[i] = og * tanhf(cv);
    }
}

#ifndef __APPLE__
static void apply_lstm_gates_f16(
    const __fp16* __restrict gates,
    __fp16* __restrict c,
    __fp16* __restrict h,
    size_t hidden_size
) {
    constexpr size_t W = 8;
    const size_t simd_end = (hidden_size / W) * W;
    const float32x4_t one = vdupq_n_f32(1.0f);

    for (size_t i = 0; i < simd_end; i += W) {
        float16x8_t i_raw = vld1q_f16(gates + i);
        float16x8_t f_raw = vld1q_f16(gates + hidden_size + i);
        float16x8_t g_raw = vld1q_f16(gates + 2 * hidden_size + i);
        float16x8_t o_raw = vld1q_f16(gates + 3 * hidden_size + i);

        float32x4_t il = vcvt_f32_f16(vget_low_f16(i_raw));
        float32x4_t ih = vcvt_f32_f16(vget_high_f16(i_raw));
        float16x8_t i_act = vcombine_f16(
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(il))))),
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(ih))))));

        float32x4_t fl = vcvt_f32_f16(vget_low_f16(f_raw));
        float32x4_t fh = vcvt_f32_f16(vget_high_f16(f_raw));
        float16x8_t f_act = vcombine_f16(
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(fl))))),
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(fh))))));

        float32x4_t gl = vcvt_f32_f16(vget_low_f16(g_raw));
        float32x4_t gh = vcvt_f32_f16(vget_high_f16(g_raw));
        float16x8_t g_act = vcombine_f16(
            vcvt_f16_f32(fast_tanh_f32x4(gl)),
            vcvt_f16_f32(fast_tanh_f32x4(gh)));

        float32x4_t ol = vcvt_f32_f16(vget_low_f16(o_raw));
        float32x4_t oh = vcvt_f32_f16(vget_high_f16(o_raw));
        float16x8_t o_act = vcombine_f16(
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(ol))))),
            vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(oh))))));

        float16x8_t c_prev = vld1q_f16(c + i);
        float16x8_t c_new = vfmaq_f16(vmulq_f16(f_act, c_prev), i_act, g_act);
        vst1q_f16(c + i, c_new);

        float32x4_t cl = vcvt_f32_f16(vget_low_f16(c_new));
        float32x4_t ch = vcvt_f32_f16(vget_high_f16(c_new));
        float16x8_t c_tanh = vcombine_f16(
            vcvt_f16_f32(fast_tanh_f32x4(cl)),
            vcvt_f16_f32(fast_tanh_f32x4(ch)));
        vst1q_f16(h + i, vmulq_f16(o_act, c_tanh));
    }

    for (size_t i = simd_end; i < hidden_size; ++i) {
        float ig = 1.0f / (1.0f + expf(-static_cast<float>(gates[i])));
        float fg = 1.0f / (1.0f + expf(-static_cast<float>(gates[hidden_size + i])));
        float gg = tanhf(static_cast<float>(gates[2 * hidden_size + i]));
        float og = 1.0f / (1.0f + expf(-static_cast<float>(gates[3 * hidden_size + i])));
        float cv = fg * static_cast<float>(c[i]) + ig * gg;
        c[i] = static_cast<__fp16>(cv);
        h[i] = static_cast<__fp16>(og * tanhf(cv));
    }
}

static inline float hsum_f16x8_f32(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t s4 = vadd_f16(lo, hi);
    float16x4_t s2 = vadd_f16(s4, vext_f16(s4, s4, 2));
    float16x4_t s1 = vadd_f16(s2, vext_f16(s2, s2, 1));
    return static_cast<float>(vget_lane_f16(s1, 0));
}

static void lstm_gemv_f16_neon(
    const __fp16* __restrict x,
    const __fp16* __restrict W,
    const __fp16* __restrict bias,
    __fp16* __restrict out,
    size_t K, size_t N
) {
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        const __fp16* w0 = W + n * K;
        const __fp16* w1 = W + (n + 1) * K;
        const __fp16* w2 = W + (n + 2) * K;
        const __fp16* w3 = W + (n + 3) * K;
        float16x8_t a0 = vdupq_n_f16(0);
        float16x8_t a1 = vdupq_n_f16(0);
        float16x8_t a2 = vdupq_n_f16(0);
        float16x8_t a3 = vdupq_n_f16(0);
        for (size_t k = 0; k < K16; k += 16) {
            float16x8_t xlo = vld1q_f16(x + k);
            float16x8_t xhi = vld1q_f16(x + k + 8);
            a0 = vfmaq_f16(a0, xlo, vld1q_f16(w0 + k));
            a0 = vfmaq_f16(a0, xhi, vld1q_f16(w0 + k + 8));
            a1 = vfmaq_f16(a1, xlo, vld1q_f16(w1 + k));
            a1 = vfmaq_f16(a1, xhi, vld1q_f16(w1 + k + 8));
            a2 = vfmaq_f16(a2, xlo, vld1q_f16(w2 + k));
            a2 = vfmaq_f16(a2, xhi, vld1q_f16(w2 + k + 8));
            a3 = vfmaq_f16(a3, xlo, vld1q_f16(w3 + k));
            a3 = vfmaq_f16(a3, xhi, vld1q_f16(w3 + k + 8));
        }
        for (size_t k = K16; k < K8; k += 8) {
            float16x8_t xv = vld1q_f16(x + k);
            a0 = vfmaq_f16(a0, xv, vld1q_f16(w0 + k));
            a1 = vfmaq_f16(a1, xv, vld1q_f16(w1 + k));
            a2 = vfmaq_f16(a2, xv, vld1q_f16(w2 + k));
            a3 = vfmaq_f16(a3, xv, vld1q_f16(w3 + k));
        }
        float s0 = hsum_f16x8_f32(a0);
        float s1 = hsum_f16x8_f32(a1);
        float s2 = hsum_f16x8_f32(a2);
        float s3 = hsum_f16x8_f32(a3);
        for (size_t k = K8; k < K; ++k) {
            float xv = static_cast<float>(x[k]);
            s0 += xv * static_cast<float>(w0[k]);
            s1 += xv * static_cast<float>(w1[k]);
            s2 += xv * static_cast<float>(w2[k]);
            s3 += xv * static_cast<float>(w3[k]);
        }
        out[n] = static_cast<__fp16>(s0 + static_cast<float>(bias[n]));
        out[n + 1] = static_cast<__fp16>(s1 + static_cast<float>(bias[n + 1]));
        out[n + 2] = static_cast<__fp16>(s2 + static_cast<float>(bias[n + 2]));
        out[n + 3] = static_cast<__fp16>(s3 + static_cast<float>(bias[n + 3]));
    }
    for (; n < N; ++n) {
        const __fp16* w = W + n * K;
        float16x8_t acc = vdupq_n_f16(0);
        for (size_t k = 0; k < K8; k += 8)
            acc = vfmaq_f16(acc, vld1q_f16(x + k), vld1q_f16(w + k));
        float s = hsum_f16x8_f32(acc);
        for (size_t k = K8; k < K; ++k)
            s += static_cast<float>(x[k]) * static_cast<float>(w[k]);
        out[n] = static_cast<__fp16>(s + static_cast<float>(bias[n]));
    }
}
#endif

static inline void fp16_to_fp32_neon(const __fp16* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t v = vld1q_f16(src + i);
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(v)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(v)));
    }
    for (; i < n; ++i)
        dst[i] = static_cast<float>(src[i]);
}

static inline void fp32_to_fp16_neon(const float* src, __fp16* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
    }
    for (; i < n; ++i)
        dst[i] = static_cast<__fp16>(src[i]);
}

void cactus_bilstm_sequence_f16(
    const __fp16* input,
    const __fp16* weight_ih_fwd,
    const __fp16* weight_hh_fwd,
    const __fp16* bias_ih_fwd,
    const __fp16* bias_hh_fwd,
    const __fp16* weight_ih_bwd,
    const __fp16* weight_hh_bwd,
    const __fp16* bias_ih_bwd,
    const __fp16* bias_hh_bwd,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    size_t hidden_size
) {
    const size_t gate_size = 4 * hidden_size;
    const size_t combined_K = input_size + hidden_size;
    const size_t output_size = 2 * hidden_size;

    std::vector<float> bias_fwd_f32(gate_size);
    std::vector<float> bias_bwd_f32(gate_size);
    for (size_t g = 0; g < gate_size; ++g) {
        bias_fwd_f32[g] = static_cast<float>(bias_ih_fwd[g]) + static_cast<float>(bias_hh_fwd[g]);
        bias_bwd_f32[g] = static_cast<float>(bias_ih_bwd[g]) + static_cast<float>(bias_hh_bwd[g]);
    }

#ifdef __APPLE__
    std::vector<float> W_fwd_f32(gate_size * combined_K);
    std::vector<float> W_bwd_f32(gate_size * combined_K);
    for (size_t g = 0; g < gate_size; ++g) {
        fp16_to_fp32_neon(weight_ih_fwd + g * input_size,
                          W_fwd_f32.data() + g * combined_K, input_size);
        fp16_to_fp32_neon(weight_hh_fwd + g * hidden_size,
                          W_fwd_f32.data() + g * combined_K + input_size, hidden_size);
        fp16_to_fp32_neon(weight_ih_bwd + g * input_size,
                          W_bwd_f32.data() + g * combined_K, input_size);
        fp16_to_fp32_neon(weight_hh_bwd + g * hidden_size,
                          W_bwd_f32.data() + g * combined_K + input_size, hidden_size);
    }

    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* batch_in = input + b * seq_len * input_size;
        __fp16* batch_out = output + b * seq_len * output_size;

        auto& pool = CactusThreading::get_thread_pool();
        auto bwd_future = pool.enqueue([&, batch_in, batch_out]() {
            std::vector<float> xh(combined_K);
            std::vector<float> gates(gate_size);
            std::vector<float> h(hidden_size, 0.0f);
            std::vector<float> c(hidden_size, 0.0f);

            for (size_t idx = 0; idx < seq_len; ++idx) {
                const size_t t = seq_len - 1 - idx;
                fp16_to_fp32_neon(batch_in + t * input_size, xh.data(), input_size);
                memcpy(xh.data() + input_size, h.data(), hidden_size * sizeof(float));

                memcpy(gates.data(), bias_bwd_f32.data(), gate_size * sizeof(float));
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            static_cast<int>(gate_size), static_cast<int>(combined_K),
                            1.0f, W_bwd_f32.data(), static_cast<int>(combined_K),
                            xh.data(), 1, 1.0f, gates.data(), 1);

                apply_lstm_gates_f32(gates.data(), c.data(), h.data(), hidden_size);

                fp32_to_fp16_neon(h.data(), batch_out + t * output_size + hidden_size, hidden_size);
            }
        });

        {
            std::vector<float> xh(combined_K);
            std::vector<float> gates(gate_size);
            std::vector<float> h(hidden_size, 0.0f);
            std::vector<float> c(hidden_size, 0.0f);

            for (size_t t = 0; t < seq_len; ++t) {
                fp16_to_fp32_neon(batch_in + t * input_size, xh.data(), input_size);
                memcpy(xh.data() + input_size, h.data(), hidden_size * sizeof(float));

                memcpy(gates.data(), bias_fwd_f32.data(), gate_size * sizeof(float));
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            static_cast<int>(gate_size), static_cast<int>(combined_K),
                            1.0f, W_fwd_f32.data(), static_cast<int>(combined_K),
                            xh.data(), 1, 1.0f, gates.data(), 1);

                apply_lstm_gates_f32(gates.data(), c.data(), h.data(), hidden_size);

                fp32_to_fp16_neon(h.data(), batch_out + t * output_size, hidden_size);
            }
        }

        bwd_future.get();
    }

#else
    std::vector<__fp16> W_fwd_f16(gate_size * combined_K);
    std::vector<__fp16> W_bwd_f16(gate_size * combined_K);
    for (size_t g = 0; g < gate_size; ++g) {
        memcpy(W_fwd_f16.data() + g * combined_K,
               weight_ih_fwd + g * input_size, input_size * sizeof(__fp16));
        memcpy(W_fwd_f16.data() + g * combined_K + input_size,
               weight_hh_fwd + g * hidden_size, hidden_size * sizeof(__fp16));
        memcpy(W_bwd_f16.data() + g * combined_K,
               weight_ih_bwd + g * input_size, input_size * sizeof(__fp16));
        memcpy(W_bwd_f16.data() + g * combined_K + input_size,
               weight_hh_bwd + g * hidden_size, hidden_size * sizeof(__fp16));
    }

    std::vector<__fp16> bias_fwd_f16(gate_size);
    std::vector<__fp16> bias_bwd_f16(gate_size);
    for (size_t g = 0; g < gate_size; ++g) {
        bias_fwd_f16[g] = static_cast<__fp16>(bias_fwd_f32[g]);
        bias_bwd_f16[g] = static_cast<__fp16>(bias_bwd_f32[g]);
    }

    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* batch_in = input + b * seq_len * input_size;
        __fp16* batch_out = output + b * seq_len * output_size;

        auto& pool = CactusThreading::get_thread_pool();
        auto bwd_future = pool.enqueue([&, batch_in, batch_out]() {
            std::vector<__fp16> xh(combined_K);
            std::vector<__fp16> gates(gate_size);
            std::vector<__fp16> h(hidden_size, static_cast<__fp16>(0));
            std::vector<__fp16> c(hidden_size, static_cast<__fp16>(0));

            for (size_t idx = 0; idx < seq_len; ++idx) {
                const size_t t = seq_len - 1 - idx;
                memcpy(xh.data(), batch_in + t * input_size, input_size * sizeof(__fp16));
                memcpy(xh.data() + input_size, h.data(), hidden_size * sizeof(__fp16));
                lstm_gemv_f16_neon(xh.data(), W_bwd_f16.data(), bias_bwd_f16.data(),
                                   gates.data(), combined_K, gate_size);
                apply_lstm_gates_f16(gates.data(), c.data(), h.data(), hidden_size);
                memcpy(batch_out + t * output_size + hidden_size,
                       h.data(), hidden_size * sizeof(__fp16));
            }
        });

        {
            std::vector<__fp16> xh(combined_K);
            std::vector<__fp16> gates(gate_size);
            std::vector<__fp16> h(hidden_size, static_cast<__fp16>(0));
            std::vector<__fp16> c(hidden_size, static_cast<__fp16>(0));

            for (size_t t = 0; t < seq_len; ++t) {
                memcpy(xh.data(), batch_in + t * input_size, input_size * sizeof(__fp16));
                memcpy(xh.data() + input_size, h.data(), hidden_size * sizeof(__fp16));
                lstm_gemv_f16_neon(xh.data(), W_fwd_f16.data(), bias_fwd_f16.data(),
                                   gates.data(), combined_K, gate_size);
                apply_lstm_gates_f16(gates.data(), c.data(), h.data(), hidden_size);
                memcpy(batch_out + t * output_size,
                       h.data(), hidden_size * sizeof(__fp16));
            }
        }

        bwd_future.get();
    }
#endif
}
