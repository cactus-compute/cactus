#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <cstddef>
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

constexpr size_t T_TILE_F16 = 2;
constexpr size_t ACCELERATE_K_THRESHOLD = 32;
constexpr size_t ACCELERATE_L_THRESHOLD = 128;

#ifdef __APPLE__
static void conv1d_causal_depthwise_f16_accelerate(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;
    CactusThreading::parallel_for_2d(N, C, CactusThreading::Thresholds::ATTENTION, [&](size_t n, size_t c) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16*       Yb = output + n * out_bs;
        const __fp16* Wc = weight + c * K;

        if (dilation == 1) {

            size_t i = 0;
            for(; i + 7 < L; i += 8){
                float16x8_t acc = vdupq_n_f16(0.0f);

                for (size_t k = 0; k < K; ++k) {
                    float16x8_t input_vec = vld1q_f16(&Xb[(i + k) * C + c]);
                    float16x8_t weight_vec = vdupq_n_f16(Wc[k]);
                    acc = vfmaq_f16(acc, input_vec, weight_vec);
                }
                vst1q_f16(&Yb[i * C + c], acc);
            }

            for (; i < L; ++i) {
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += float(Xb[(i + k) * C + c]) * float(Wc[k]);
                }
                Yb[i * C + c] = (__fp16)acc;
            }
        } else {

            for (size_t i = 0; i < L; ++i) {
                float acc = 0.0f;

                for (size_t k = 0; k < K; ++k) {
                    size_t idx = i + k * dilation;
                    acc += float(Xb[idx * C + c]) * float(Wc[k]);
                }

                Yb[i * C + c] = (__fp16)acc;
            }
        }
    });
}
#endif

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
#ifdef __APPLE__
    if (K >= ACCELERATE_K_THRESHOLD && L >= ACCELERATE_L_THRESHOLD) {
        conv1d_causal_depthwise_f16_accelerate(input, weight, output, N, L, C, K, dilation);
        return;
    }
#endif

    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;

    CactusThreading::parallel_for_2d(N, C, CactusThreading::Thresholds::ATTENTION, [&](size_t n, size_t c) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16*       Yb = output + n * out_bs;

        std::vector<float> wrev(K);
        const __fp16* Wc = weight + c * K;
        for (size_t k = 0; k < K; ++k) wrev[k] = (float)Wc[K - 1 - k];

        for (size_t t0 = 0; t0 < L; t0 += T_TILE_F16) {
            const size_t t1 = std::min(t0 + 1, L - 1);

            float32x4_t vacc0 = vdupq_n_f32(0.f);
            float32x4_t vacc1 = vdupq_n_f32(0.f);

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                
                float x0_0=0, x1_0=0, x2_0=0, x3_0=0;
                float x0_1=0, x1_1=0, x2_1=0, x3_1=0;
                {
                    ptrdiff_t a0=(ptrdiff_t)t0-(ptrdiff_t)((k+0)*dilation);
                    ptrdiff_t a1=(ptrdiff_t)t0-(ptrdiff_t)((k+1)*dilation);
                    ptrdiff_t a2=(ptrdiff_t)t0-(ptrdiff_t)((k+2)*dilation);
                    ptrdiff_t a3=(ptrdiff_t)t0-(ptrdiff_t)((k+3)*dilation);
                    if (a0>=0) x0_0 = (float)Xb[(size_t)a0*C + c];
                    if (a1>=0) x1_0 = (float)Xb[(size_t)a1*C + c];
                    if (a2>=0) x2_0 = (float)Xb[(size_t)a2*C + c];
                    if (a3>=0) x3_0 = (float)Xb[(size_t)a3*C + c];

                    ptrdiff_t b0=(ptrdiff_t)t1-(ptrdiff_t)((k+0)*dilation);
                    ptrdiff_t b1=(ptrdiff_t)t1-(ptrdiff_t)((k+1)*dilation);
                    ptrdiff_t b2=(ptrdiff_t)t1-(ptrdiff_t)((k+2)*dilation);
                    ptrdiff_t b3=(ptrdiff_t)t1-(ptrdiff_t)((k+3)*dilation);
                    if (b0>=0) x0_1 = (float)Xb[(size_t)b0*C + c];
                    if (b1>=0) x1_1 = (float)Xb[(size_t)b1*C + c];
                    if (b2>=0) x2_1 = (float)Xb[(size_t)b2*C + c];
                    if (b3>=0) x3_1 = (float)Xb[(size_t)b3*C + c];
                }
                float32x4_t xv0 = {x0_0,x1_0,x2_0,x3_0};
                float32x4_t yv0 = {x0_1,x1_1,x2_1,x3_1};
                float32x4_t wv0 = {wrev[k+0],wrev[k+1],wrev[k+2],wrev[k+3]};
                vacc0 = vfmaq_f32(vacc0, xv0, wv0);
                vacc1 = vfmaq_f32(vacc1, yv0, wv0);

                float a0_0=0, a1_0=0, a2_0=0, a3_0=0;
                float a0_1=0, a1_1=0, a2_1=0, a3_1=0;
                {
                    ptrdiff_t a0i=(ptrdiff_t)t0-(ptrdiff_t)((k+4)*dilation);
                    ptrdiff_t a1i=(ptrdiff_t)t0-(ptrdiff_t)((k+5)*dilation);
                    ptrdiff_t a2i=(ptrdiff_t)t0-(ptrdiff_t)((k+6)*dilation);
                    ptrdiff_t a3i=(ptrdiff_t)t0-(ptrdiff_t)((k+7)*dilation);
                    if (a0i>=0) a0_0 = (float)Xb[(size_t)a0i*C + c];
                    if (a1i>=0) a1_0 = (float)Xb[(size_t)a1i*C + c];
                    if (a2i>=0) a2_0 = (float)Xb[(size_t)a2i*C + c];
                    if (a3i>=0) a3_0 = (float)Xb[(size_t)a3i*C + c];

                    ptrdiff_t b0i=(ptrdiff_t)t1-(ptrdiff_t)((k+4)*dilation);
                    ptrdiff_t b1i=(ptrdiff_t)t1-(ptrdiff_t)((k+5)*dilation);
                    ptrdiff_t b2i=(ptrdiff_t)t1-(ptrdiff_t)((k+6)*dilation);
                    ptrdiff_t b3i=(ptrdiff_t)t1-(ptrdiff_t)((k+7)*dilation);
                    if (b0i>=0) a0_1 = (float)Xb[(size_t)b0i*C + c];
                    if (b1i>=0) a1_1 = (float)Xb[(size_t)b1i*C + c];
                    if (b2i>=0) a2_1 = (float)Xb[(size_t)b2i*C + c];
                    if (b3i>=0) a3_1 = (float)Xb[(size_t)b3i*C + c];
                }
                float32x4_t xv1 = {a0_0,a1_0,a2_0,a3_0};
                float32x4_t yv1 = {a0_1,a1_1,a2_1,a3_1};
                float32x4_t wv1 = {wrev[k+4],wrev[k+5],wrev[k+6],wrev[k+7]};
                vacc0 = vfmaq_f32(vacc0, xv1, wv1);
                vacc1 = vfmaq_f32(vacc1, yv1, wv1);
            }

            float acc0 = vaddvq_f32(vacc0);
            float acc1 = vaddvq_f32(vacc1);

            for (; k < K; ++k) {
                ptrdiff_t a=(ptrdiff_t)t0-(ptrdiff_t)(k*dilation);
                if (a>=0) acc0 += wrev[k] * (float)Xb[(size_t)a*C + c];
                ptrdiff_t b=(ptrdiff_t)t1-(ptrdiff_t)(k*dilation);
                if (b>=0) acc1 += wrev[k] * (float)Xb[(size_t)b*C + c];
            }

            Yb[t0*C + c] = (__fp16)acc0;
            if (t0 + 1 < L) Yb[t1*C + c] = (__fp16)acc1;
        }
    });
}

void cactus_conv1d_f16_k3(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t stride
){
    const size_t out_len = ((L - 1) / stride) + 1;
    const size_t in_bs  = C_in * L;
    const size_t out_bs = C_out * out_len;

    const size_t total_compute = N * C_out * out_len * C_in * 3;
    CactusThreading::ParallelConfig config = (total_compute < 100000)
        ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
        : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(N, C_out, config, [&](size_t n, size_t oc) {
        const __fp16* Xb = input + n * in_bs;
        __fp16* Yoc = output + n * out_bs + oc * out_len;
        const __fp16* Woc = weight + oc * (C_in * 3);

        for (size_t out_idx = 0; out_idx < out_len; out_idx += 2) {
            const size_t out_t0 = out_idx;
            const bool have_t1 = (out_idx + 1) < out_len;
            const size_t out_t1 = have_t1 ? (out_idx + 1) : out_idx;

            const size_t t0 = out_t0 * stride;
            const size_t t1 = have_t1 ? (out_t1 * stride) : t0;

            float32x4_t acc0 = vdupq_n_f32(0.f);
            float32x4_t acc1 = vdupq_n_f32(0.f);

            size_t ic = 0;
            for (; ic + 16 <= C_in; ic += 16) {
                for (size_t u = 0; u < 16; ++u) {
                    const __fp16* Xc = Xb + (ic + u) * L;
                    const __fp16* Wc = Woc + (ic + u) * 3;

                    const float16x8_t wv = {
                        Wc[0], Wc[1], Wc[2], (__fp16)0,
                        Wc[0], Wc[1], Wc[2], (__fp16)0
                    };

                    const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                    const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;
                    const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                    const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;

                    const __fp16 x0m = (tm0 >= 0) ? Xc[tm0] : (__fp16)0;
                    const __fp16 x00 = Xc[t0];
                    const __fp16 x0p = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : (__fp16)0;

                    __fp16 x1m = 0, x10 = 0, x1p = 0;
                    if (have_t1) {
                        x1m = (tm1 >= 0) ? Xc[tm1] : (__fp16)0;
                        x10 = Xc[t1];
                        x1p = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : (__fp16)0;
                    }

                    const float16x8_t xv = {
                        x0m, x00, x0p, (__fp16)0,
                        x1m, x10, x1p, (__fp16)0
                    };

                    const float16x4_t xv0_h = vget_low_f16(xv);
                    const float16x4_t wv0_h = vget_low_f16(wv);
                    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(xv0_h), vcvt_f32_f16(wv0_h));

                    if (have_t1) {
                        const float16x4_t xv1_h = vget_high_f16(xv);
                        const float16x4_t wv1_h = vget_high_f16(wv);
                        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(xv1_h), vcvt_f32_f16(wv1_h));
                    }
                }
            }

            for (; ic < C_in; ++ic) {
                const __fp16* Xc = Xb + ic * L;
                const __fp16* Wc = Woc + ic * 3;

                const float16x8_t wv = {
                    Wc[0], Wc[1], Wc[2], (__fp16)0,
                    Wc[0], Wc[1], Wc[2], (__fp16)0
                };

                const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;
                const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;

                const __fp16 x0m = (tm0 >= 0) ? Xc[tm0] : (__fp16)0;
                const __fp16 x00 = Xc[t0];
                const __fp16 x0p = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : (__fp16)0;

                __fp16 x1m = 0, x10 = 0, x1p = 0;
                if (have_t1) {
                    x1m = (tm1 >= 0) ? Xc[tm1] : (__fp16)0;
                    x10 = Xc[t1];
                    x1p = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : (__fp16)0;
                }

                const float16x8_t xv = {
                    x0m, x00, x0p, (__fp16)0,
                    x1m, x10, x1p, (__fp16)0
                };

                const float16x4_t xv0_h = vget_low_f16(xv);
                const float16x4_t wv0_h = vget_low_f16(wv);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(xv0_h), vcvt_f32_f16(wv0_h));

                if (have_t1) {
                    const float16x4_t xv1_h = vget_high_f16(xv);
                    const float16x4_t wv1_h = vget_high_f16(wv);
                    acc1 = vfmaq_f32(acc1, vcvt_f32_f16(xv1_h), vcvt_f32_f16(wv1_h));
                }
            }

            float32x2_t s0 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
            float sum0 = vget_lane_f32(s0, 0) + vget_lane_f32(s0, 1);
            Yoc[out_t0] = (__fp16)sum0;

            if (have_t1) {
                float32x2_t s1 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
                float sum1 = vget_lane_f32(s1, 0) + vget_lane_f32(s1, 1);
                Yoc[out_t1] = (__fp16)sum1;
            }
        }
    });
}

#ifdef __APPLE__
static void conv1d_f16_accelerate(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t K,
    size_t stride
){
    const size_t out_len = ((L - K) / stride) + 1;
    const size_t in_bs   = C_in * L;
    const size_t out_bs  = C_out * out_len;

    const size_t total_compute = N * C_out * out_len * C_in * K;
    CactusThreading::ParallelConfig config = (total_compute < 100000)
        ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
        : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(
        N, C_out, config,
        [&](size_t n, size_t oc) {

        const __fp16* Xb  = input  + n * in_bs;
        __fp16*       Yoc = output + n * out_bs + oc * out_len;
        const __fp16* Woc = weight + oc * (C_in * K);
        const float   b   = bias ? (float)bias[oc] : 0.f;

        std::vector<float> out_f32(out_len, b);
        std::vector<float> input_f32(L);
        std::vector<float> weight_f32(K);

        for (size_t ic = 0; ic < C_in; ++ic) {
            const __fp16* Xc = Xb + ic * L;
            const __fp16* Wc = Woc + ic * K;

            for (size_t i = 0; i < L; ++i) input_f32[i] = (float)Xc[i];
            for (size_t k = 0; k < K; ++k) weight_f32[k] = (float)Wc[k];

            if (stride == 1) {
                std::vector<float> conv_out(out_len);
                vDSP_conv(input_f32.data(), 1, weight_f32.data(), 1,
                          conv_out.data(), 1, out_len, K);
                vDSP_vadd(out_f32.data(), 1, conv_out.data(), 1,
                          out_f32.data(), 1, out_len);
            } else {
                std::vector<float> full_conv(L - K + 1);
                vDSP_conv(input_f32.data(), 1, weight_f32.data(), 1,
                          full_conv.data(), 1, L - K + 1, K);
                for (size_t out_t = 0; out_t < out_len; ++out_t) {
                    out_f32[out_t] += full_conv[out_t * stride];
                }
            }
        }

        for (size_t out_t = 0; out_t < out_len; ++out_t) {
            Yoc[out_t] = (__fp16)out_f32[out_t];
        }
    });
}
#endif

static void conv1d_f16_neon(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t K,
    size_t stride
){
    const size_t out_len = ((L - K) / stride) + 1;
    const size_t in_bs   = C_in  * L;
    const size_t out_bs  = C_out * out_len;

    const size_t total_compute = N * C_out * out_len * C_in * K;
    CactusThreading::ParallelConfig config = (total_compute < 100000)
        ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
        : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(
        N, C_out, config,
        [&](size_t n, size_t oc) {

        const __fp16* Xb  = input  + n * in_bs;
        __fp16*       Yoc = output + n * out_bs + oc * out_len;
        const __fp16* Woc = weight + oc * (C_in * K);
        const float   b   = bias ? (float)bias[oc] : 0.f;

        for (size_t out_t = 0; out_t < out_len; ++out_t) {
            const size_t t = out_t * stride;
            float sum = b;

            for (size_t ic = 0; ic < C_in; ++ic) {
                const __fp16* Xc = Xb  + ic * L + t;
                const __fp16* Wc = Woc + ic * K;

                float32x4_t acc0 = vdupq_n_f32(0.f);
                float32x4_t acc1 = vdupq_n_f32(0.f);

                size_t k = 0;

                for (; k + 8 <= K; k += 8) {
                    const float16x8_t xv = vld1q_f16(Xc + k);
                    const float16x8_t wv = vld1q_f16(Wc + k);

                    acc0 = vfmaq_f32(acc0,
                                     vcvt_f32_f16(vget_low_f16(xv)),
                                     vcvt_f32_f16(vget_low_f16(wv)));
                    acc1 = vfmaq_f32(acc1,
                                     vcvt_f32_f16(vget_high_f16(xv)),
                                     vcvt_f32_f16(vget_high_f16(wv)));
                }

                float32x2_t s1 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
                sum += vget_lane_f32(s1, 0) + vget_lane_f32(s1, 1);

                float32x2_t s2 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
                sum += vget_lane_f32(s2, 0) + vget_lane_f32(s2, 1);

                for (; k < K; ++k) {
                    sum += (float)Xc[k] * (float)Wc[k];
                }
            }

            Yoc[out_t] = (__fp16)sum;
        }
    });
}

#ifdef __APPLE__
static void conv1d_f16_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t K,
    size_t stride
) {
    const size_t out_len = (L - K) / stride + 1;
    const size_t col_K = C_in * K;

    std::vector<float> W_f32(C_out * col_K);
    for (size_t i = 0; i < C_out * col_K; ++i)
        W_f32[i] = static_cast<float>(weight[i]);

    std::vector<float> bias_f32;
    if (bias) {
        bias_f32.resize(C_out);
        for (size_t i = 0; i < C_out; ++i)
            bias_f32[i] = static_cast<float>(bias[i]);
    }

    std::vector<float> col(col_K * out_len);
    std::vector<float> Y_f32(C_out * out_len);

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xn = input + n * C_in * L;
        __fp16* Yn = output + n * C_out * out_len;

        if (stride == 1) {
            for (size_t ic = 0; ic < C_in; ++ic) {
                const __fp16* Xc = Xn + ic * L;
                for (size_t k = 0; k < K; ++k) {
                    float* dst = col.data() + (ic * K + k) * out_len;
                    const __fp16* src = Xc + k;
                    size_t t = 0;
                    for (; t + 8 <= out_len; t += 8) {
                        float16x8_t v = vld1q_f16(src + t);
                        vst1q_f32(dst + t,     vcvt_f32_f16(vget_low_f16(v)));
                        vst1q_f32(dst + t + 4, vcvt_f32_f16(vget_high_f16(v)));
                    }
                    for (; t < out_len; ++t)
                        dst[t] = static_cast<float>(src[t]);
                }
            }
        } else {
            for (size_t ic = 0; ic < C_in; ++ic) {
                const __fp16* Xc = Xn + ic * L;
                for (size_t k = 0; k < K; ++k) {
                    float* dst = col.data() + (ic * K + k) * out_len;
                    for (size_t t = 0; t < out_len; ++t)
                        dst[t] = static_cast<float>(Xc[t * stride + k]);
                }
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(C_out), static_cast<int>(out_len), static_cast<int>(col_K),
                    1.0f, W_f32.data(), static_cast<int>(col_K),
                    col.data(), static_cast<int>(out_len),
                    0.0f, Y_f32.data(), static_cast<int>(out_len));

        if (bias) {
            for (size_t oc = 0; oc < C_out; ++oc) {
                float b = bias_f32[oc];
                const float* src = Y_f32.data() + oc * out_len;
                __fp16* dst = Yn + oc * out_len;
                size_t t = 0;
                for (; t + 8 <= out_len; t += 8) {
                    float32x4_t v0 = vaddq_f32(vld1q_f32(src + t),     vdupq_n_f32(b));
                    float32x4_t v1 = vaddq_f32(vld1q_f32(src + t + 4), vdupq_n_f32(b));
                    vst1q_f16(dst + t, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                }
                for (; t < out_len; ++t)
                    dst[t] = static_cast<__fp16>(src[t] + b);
            }
        } else {
            for (size_t oc = 0; oc < C_out; ++oc) {
                const float* src = Y_f32.data() + oc * out_len;
                __fp16* dst = Yn + oc * out_len;
                size_t t = 0;
                for (; t + 8 <= out_len; t += 8) {
                    float32x4_t v0 = vld1q_f32(src + t);
                    float32x4_t v1 = vld1q_f32(src + t + 4);
                    vst1q_f16(dst + t, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                }
                for (; t < out_len; ++t)
                    dst[t] = static_cast<__fp16>(src[t]);
            }
        }
    }
}
#endif

void cactus_conv1d_f16(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t K,
    size_t stride
){
#ifdef __APPLE__
    if (stride > 1 || K < ACCELERATE_K_THRESHOLD) {
        conv1d_f16_gemm(input, weight, bias, output, N, L, C_in, C_out, K, stride);
        return;
    }
    if (K >= ACCELERATE_K_THRESHOLD && L >= ACCELERATE_L_THRESHOLD) {
        conv1d_f16_accelerate(input, weight, bias, output, N, L, C_in, C_out, K, stride);
        return;
    }
#endif
    conv1d_f16_neon(input, weight, bias, output, N, L, C_in, C_out, K, stride);
}

inline void conv1d_k7s3_oc8_t4(
    const __fp16* Xb,
    const __fp16* Wpack,
    const __fp16* bias,
    __fp16* Yb,
    size_t L,
    size_t out_len,
    size_t C_in,
    size_t C_out,
    size_t out_t,
    size_t oc0
){
    float32x4_t acc[4][2];
    float32x4_t b0 = vdupq_n_f32(0.f);
    float32x4_t b1 = vdupq_n_f32(0.f);
    if (bias) {
        float16x8_t bv = vld1q_f16(bias + oc0);
        b0 = vcvt_f32_f16(vget_low_f16(bv));
        b1 = vcvt_f32_f16(vget_high_f16(bv));
    }

    for (int j = 0; j < 4; ++j) {
        acc[j][0] = b0;
        acc[j][1] = b1;
    }

    const size_t t_base = out_t * 3;

    for (size_t ic = 0; ic < C_in; ++ic) {
        const __fp16* Wic = Wpack + (ic * 7) * C_out + oc0;
        const __fp16* Xic = Xb + ic * L + t_base;

        for (int k = 0; k < 7; ++k) {
            float16x8_t w_half = vld1q_f16(Wic + k * C_out);
            float32x4_t w0 = vcvt_f32_f16(vget_low_f16(w_half));
            float32x4_t w1 = vcvt_f32_f16(vget_high_f16(w_half));
            for (int j = 0; j < 4; ++j) {
                float x_val = (float)Xic[j * 3 + k];
                float32x4_t xv = vdupq_n_f32(x_val);

                acc[j][0] = vfmaq_f32(acc[j][0], w0, xv);
                acc[j][1] = vfmaq_f32(acc[j][1], w1, xv);
            }
        }
    }

    float tmp0[4], tmp1[4];
    for(int j=0; j<4; ++j) {
        vst1q_f32(tmp0, acc[j][0]);
        vst1q_f32(tmp1, acc[j][1]);
        
        for(int i=0; i<4; ++i) {
             Yb[(oc0 + i) * out_len + out_t + j] = (__fp16)tmp0[i];
        }
        for(int i=0; i<4; ++i) {
             Yb[(oc0 + 4 + i) * out_len + out_t + j] = (__fp16)tmp1[i];
        }
    }
}

inline void conv1d_k7s3_oc8_scalar(
    const __fp16* Xb,
    const __fp16* Wpack,
    const __fp16* bias,
    __fp16* Yb,
    size_t L,
    size_t out_len,
    size_t C_in,
    size_t C_out,
    size_t out_t,
    size_t oc0
){
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);

    if (bias) {
        float16x8_t bv = vld1q_f16(bias + oc0);
        acc0 = vcvt_f32_f16(vget_low_f16(bv));
        acc1 = vcvt_f32_f16(vget_high_f16(bv));
    }

    const size_t t_base = out_t * 3;
    
    for (size_t ic = 0; ic < C_in; ++ic) {
        const __fp16* Wic = Wpack + (ic * 7) * C_out + oc0;
        const __fp16* Xic = Xb + ic * L + t_base;
        for (int k = 0; k < 7; ++k) {
            float x_val = (float)Xic[k];
            float32x4_t xv = vdupq_n_f32(x_val);
            
            float16x8_t w_half = vld1q_f16(Wic + k * C_out);
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(w_half)), xv);
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(w_half)), xv);
        }
    }
    
    float tmp[8];
    vst1q_f32(tmp, acc0);
    vst1q_f32(tmp+4, acc1);
    
    for(int i=0; i<8; ++i) {
        if (oc0 + i < C_out) {
             Yb[(oc0 + i) * out_len + out_t] = (__fp16)tmp[i];
        }
    }
}

void cactus_conv1d_f16_k7s3_oc8(
    const __fp16* input,
    const __fp16* Wpack,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out)
{
    if (L < 7) return;
    size_t out_len = (L - 7) / 3 + 1;
    size_t num_oc_blocks = (C_out + 7) / 8;

    CactusThreading::parallel_for_2d(N, num_oc_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE, [=](size_t n, size_t ob) {
        size_t oc0 = ob * 8;
        const __fp16* Xb = input + n * (C_in * L);
        __fp16* Yb = output + n * (C_out * out_len);

        size_t out_t = 0;
        for (; out_t + 4 <= out_len; out_t += 4) {
            conv1d_k7s3_oc8_t4(Xb, Wpack, bias, Yb, L, out_len, C_in, C_out, out_t, oc0);
        }

        for (; out_t < out_len; ++out_t) {
            conv1d_k7s3_oc8_scalar(Xb, Wpack, bias, Yb, L, out_len, C_in, C_out, out_t, oc0);
        }
    });
}



void cactus_bilinear_interpolation_f16(const __fp16* input, __fp16* output, size_t src_height, size_t src_width, size_t embed_dim,
                                       size_t dst_height, size_t dst_width, bool align_corners)
{
    float scale_h, scale_w;
    if (align_corners) {
        scale_h = (src_height > 1 && dst_height > 1)
                      ? static_cast<float>(src_height - 1) / static_cast<float>(dst_height - 1)
                      : 0.0f;
        scale_w = (src_width > 1 && dst_width > 1)
                      ? static_cast<float>(src_width - 1) / static_cast<float>(dst_width - 1)
                      : 0.0f;
    } else {
        scale_h = (dst_height > 0)
                      ? static_cast<float>(src_height) / static_cast<float>(dst_height)
                      : 0.0f;
        scale_w = (dst_width > 0)
                      ? static_cast<float>(src_width) / static_cast<float>(dst_width)
                      : 0.0f;
    }

    for (size_t dst_y = 0; dst_y < dst_height; ++dst_y) {
        for (size_t dst_x = 0; dst_x < dst_width; ++dst_x) {
            float src_y_float, src_x_float;
            if (align_corners) {
                src_y_float = static_cast<float>(dst_y) * scale_h;
                src_x_float = static_cast<float>(dst_x) * scale_w;
            } else {
                src_y_float = (static_cast<float>(dst_y) + 0.5f) * scale_h - 0.5f;
                src_x_float = (static_cast<float>(dst_x) + 0.5f) * scale_w - 0.5f;
                if (src_y_float < 0.0f) src_y_float = 0.0f;
                if (src_x_float < 0.0f) src_x_float = 0.0f;
                const float max_y = static_cast<float>(src_height) - 1.0f;
                const float max_x = static_cast<float>(src_width) - 1.0f;
                if (src_y_float > max_y) src_y_float = max_y;
                if (src_x_float > max_x) src_x_float = max_x;
            }

            int y0 = static_cast<int>(std::floor(src_y_float));
            int x0 = static_cast<int>(std::floor(src_x_float));

            int y1 = ((y0 + 1) < static_cast<int>(src_height)) ? (y0 + 1) : (static_cast<int>(src_height) - 1);
            int x1 = ((x0 + 1) < static_cast<int>(src_width)) ? (x0 + 1) : (static_cast<int>(src_width) - 1);

            float dy = src_y_float - y0;
            float dx = src_x_float - x0;

            float w00 = (1.0f - dx) * (1.0f - dy);
            float w01 = dx * (1.0f - dy);
            float w10 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            size_t idx00 = (y0 * static_cast<int>(src_width) + x0) * static_cast<int>(embed_dim);
            size_t idx01 = (y0 * static_cast<int>(src_width) + x1) * static_cast<int>(embed_dim);
            size_t idx10 = (y1 * static_cast<int>(src_width) + x0) * static_cast<int>(embed_dim);
            size_t idx11 = (y1 * static_cast<int>(src_width) + x1) * static_cast<int>(embed_dim);

            size_t out_idx = (dst_y * dst_width + dst_x) * embed_dim;

            size_t d = 0;
            for(; d + 8 <= embed_dim; d += 8){
                float16x8_t v00 = vld1q_f16(input + idx00 + d);
                float16x8_t v01 = vld1q_f16(input + idx01 + d);
                float16x8_t v10 = vld1q_f16(input + idx10 + d);
                float16x8_t v11 = vld1q_f16(input + idx11 + d);

                float16x8_t result = vfmaq_f16(
                    vfmaq_f16(
                        vfmaq_f16(vdupq_n_f16(0.0f), v00, vdupq_n_f16(w00)),
                        v01, vdupq_n_f16(w01)),
                    v10, vdupq_n_f16(w10));
                result = vfmaq_f16(result, v11, vdupq_n_f16(w11));

                vst1q_f16(output + out_idx + d, result);
            }

            for (; d < embed_dim; ++d) {
                output[out_idx + d] =
                    input[idx00 + d] * w00 +
                    input[idx01 + d] * w01 +
                    input[idx10 + d] * w10 +
                    input[idx11 + d] * w11;
            }
        }
    }
}

void cactus_stft_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t /*C_out*/,
    size_t K, size_t stride,
    size_t num_fft_bins
) {
    const size_t out_len = ((L - K) / stride) + 1;
    const size_t in_bs  = C_in * L;
    const size_t out_bs = 2 * num_fft_bins * out_len;

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xb = input + n * in_bs;

        for (size_t bin = 0; bin < num_fft_bins; ++bin) {
            const __fp16* Wr = weight + bin * (C_in * K);
            const __fp16* Wi = weight + (bin + num_fft_bins) * (C_in * K);

            for (size_t out_t = 0; out_t < out_len; ++out_t) {
                const size_t t = out_t * stride;
                float sum_real = 0.0f;
                float sum_imag = 0.0f;

                for (size_t ic = 0; ic < C_in; ++ic) {
                    const __fp16* Xc  = Xb  + ic * L + t;
                    const __fp16* Wrc = Wr  + ic * K;
                    const __fp16* Wic = Wi  + ic * K;

                    float32x4_t acc_r0 = vdupq_n_f32(0.f);
                    float32x4_t acc_r1 = vdupq_n_f32(0.f);
                    float32x4_t acc_i0 = vdupq_n_f32(0.f);
                    float32x4_t acc_i1 = vdupq_n_f32(0.f);

                    size_t k = 0;
                    for (; k + 8 <= K; k += 8) {
                        const float16x8_t xv = vld1q_f16(Xc + k);
                        const float16x8_t wr = vld1q_f16(Wrc + k);
                        const float16x8_t wi = vld1q_f16(Wic + k);

                        acc_r0 = vfmaq_f32(acc_r0, vcvt_f32_f16(vget_low_f16(xv)),  vcvt_f32_f16(vget_low_f16(wr)));
                        acc_r1 = vfmaq_f32(acc_r1, vcvt_f32_f16(vget_high_f16(xv)), vcvt_f32_f16(vget_high_f16(wr)));
                        acc_i0 = vfmaq_f32(acc_i0, vcvt_f32_f16(vget_low_f16(xv)),  vcvt_f32_f16(vget_low_f16(wi)));
                        acc_i1 = vfmaq_f32(acc_i1, vcvt_f32_f16(vget_high_f16(xv)), vcvt_f32_f16(vget_high_f16(wi)));
                    }

                    sum_real += vaddvq_f32(acc_r0) + vaddvq_f32(acc_r1);
                    sum_imag += vaddvq_f32(acc_i0) + vaddvq_f32(acc_i1);

                    for (; k < K; ++k) {
                        float x = (float)Xc[k];
                        sum_real += x * (float)Wrc[k];
                        sum_imag += x * (float)Wic[k];
                    }
                }

                output[n * out_bs + bin * out_len + out_t]                    = (__fp16)sum_real;
                output[n * out_bs + (bin + num_fft_bins) * out_len + out_t]  = (__fp16)sum_imag;
            }
        }
    }
}

void cactus_conv1d_same_depthwise_f16_k9(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t L, size_t C)
{
    constexpr int K = 9;

    const size_t bs = L * C;

    CactusThreading::parallel_for_2d(
        N, C, CactusThreading::Thresholds::ATTENTION,
        [&](size_t n, size_t c) {

        const __fp16* Xb = input  + n * bs;
        __fp16* Yb = output + n * bs;

        const __fp16* Wc = weight + c * K;
        const float b = bias ? (float)bias[c] : 0.0f;

        const float w0 = (float)Wc[0];
        const float w1 = (float)Wc[1];
        const float w2 = (float)Wc[2];
        const float w3 = (float)Wc[3];
        const float w4 = (float)Wc[4];
        const float w5 = (float)Wc[5];
        const float w6 = (float)Wc[6];
        const float w7 = (float)Wc[7];
        const float w8 = (float)Wc[8];

        const float32x4_t wv0 = {w0, w1, w2, w3};
        const float32x4_t wv1 = {w4, w5, w6, w7};

        for (size_t t0 = 0; t0 < L; t0 += T_TILE_F16) {
            const bool have_t1 = (t0 + 1 < L);
            const size_t t1 = have_t1 ? (t0 + 1) : t0;

            float x0_0=0, x1_0=0, x2_0=0, x3_0=0;
            float x4_0=0, x5_0=0, x6_0=0, x7_0=0;
            float x8_0=0;

            {
                const ptrdiff_t i0 = (ptrdiff_t)t0 - 4;
                const ptrdiff_t i1 = (ptrdiff_t)t0 - 3;
                const ptrdiff_t i2 = (ptrdiff_t)t0 - 2;
                const ptrdiff_t i3 = (ptrdiff_t)t0 - 1;
                if (i0 >= 0 && i0 < (ptrdiff_t)L) x0_0 = (float)Xb[(size_t)i0 * C + c];
                if (i1 >= 0 && i1 < (ptrdiff_t)L) x1_0 = (float)Xb[(size_t)i1 * C + c];
                if (i2 >= 0 && i2 < (ptrdiff_t)L) x2_0 = (float)Xb[(size_t)i2 * C + c];
                if (i3 >= 0 && i3 < (ptrdiff_t)L) x3_0 = (float)Xb[(size_t)i3 * C + c];

                const ptrdiff_t j4 = (ptrdiff_t)t0 + 0;
                const ptrdiff_t j5 = (ptrdiff_t)t0 + 1;
                const ptrdiff_t j6 = (ptrdiff_t)t0 + 2;
                const ptrdiff_t j7 = (ptrdiff_t)t0 + 3;
                const ptrdiff_t j8 = (ptrdiff_t)t0 + 4;
                if (j4 >= 0 && j4 < (ptrdiff_t)L) x4_0 = (float)Xb[(size_t)j4 * C + c];
                if (j5 >= 0 && j5 < (ptrdiff_t)L) x5_0 = (float)Xb[(size_t)j5 * C + c];
                if (j6 >= 0 && j6 < (ptrdiff_t)L) x6_0 = (float)Xb[(size_t)j6 * C + c];
                if (j7 >= 0 && j7 < (ptrdiff_t)L) x7_0 = (float)Xb[(size_t)j7 * C + c];
                if (j8 >= 0 && j8 < (ptrdiff_t)L) x8_0 = (float)Xb[(size_t)j8 * C + c];
            }

            float32x4_t acc0v = vdupq_n_f32(0.f);
            const float32x4_t xv0_0 = {x0_0, x1_0, x2_0, x3_0};
            const float32x4_t xv1_0 = {x4_0, x5_0, x6_0, x7_0};
            acc0v = vfmaq_f32(acc0v, xv0_0, wv0);
            acc0v = vfmaq_f32(acc0v, xv1_0, wv1);
            float acc0 = vaddvq_f32(acc0v) + (w8 * x8_0) + b;

            float acc1 = 0.f;
            if (have_t1) {
                float x0_1=0, x1_1=0, x2_1=0, x3_1=0;
                float x4_1=0, x5_1=0, x6_1=0, x7_1=0;
                float x8_1=0;

                {
                    const ptrdiff_t i0 = (ptrdiff_t)t1 - 4;
                    const ptrdiff_t i1 = (ptrdiff_t)t1 - 3;
                    const ptrdiff_t i2 = (ptrdiff_t)t1 - 2;
                    const ptrdiff_t i3 = (ptrdiff_t)t1 - 1;
                    if (i0 >= 0 && i0 < (ptrdiff_t)L) x0_1 = (float)Xb[(size_t)i0 * C + c];
                    if (i1 >= 0 && i1 < (ptrdiff_t)L) x1_1 = (float)Xb[(size_t)i1 * C + c];
                    if (i2 >= 0 && i2 < (ptrdiff_t)L) x2_1 = (float)Xb[(size_t)i2 * C + c];
                    if (i3 >= 0 && i3 < (ptrdiff_t)L) x3_1 = (float)Xb[(size_t)i3 * C + c];

                    const ptrdiff_t j4 = (ptrdiff_t)t1 + 0;
                    const ptrdiff_t j5 = (ptrdiff_t)t1 + 1;
                    const ptrdiff_t j6 = (ptrdiff_t)t1 + 2;
                    const ptrdiff_t j7 = (ptrdiff_t)t1 + 3;
                    const ptrdiff_t j8 = (ptrdiff_t)t1 + 4;
                    if (j4 >= 0 && j4 < (ptrdiff_t)L) x4_1 = (float)Xb[(size_t)j4 * C + c];
                    if (j5 >= 0 && j5 < (ptrdiff_t)L) x5_1 = (float)Xb[(size_t)j5 * C + c];
                    if (j6 >= 0 && j6 < (ptrdiff_t)L) x6_1 = (float)Xb[(size_t)j6 * C + c];
                    if (j7 >= 0 && j7 < (ptrdiff_t)L) x7_1 = (float)Xb[(size_t)j7 * C + c];
                    if (j8 >= 0 && j8 < (ptrdiff_t)L) x8_1 = (float)Xb[(size_t)j8 * C + c];
                }

                float32x4_t acc1v = vdupq_n_f32(0.f);
                const float32x4_t xv0_1 = {x0_1, x1_1, x2_1, x3_1};
                const float32x4_t xv1_1 = {x4_1, x5_1, x6_1, x7_1};
                acc1v = vfmaq_f32(acc1v, xv0_1, wv0);
                acc1v = vfmaq_f32(acc1v, xv1_1, wv1);
                acc1 = vaddvq_f32(acc1v) + (w8 * x8_1) + b;
            }

            Yb[t0 * C + c] = (__fp16)acc0;
            if (have_t1) Yb[t1 * C + c] = (__fp16)acc1;
        }
    });
}


