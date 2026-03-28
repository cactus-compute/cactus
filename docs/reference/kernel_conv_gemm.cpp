/*
 * Alternative conv1d implementation using im2col + cblas_sgemm.
 *
 * 15x faster than the vDSP path for small kernels (K<8) and strided
 * convolutions. Not active pending Henry's review of vDSP path optimization.
 *
 * Benchmarks (M4 Pro, pyannote segmentation model):
 *   SincConv (1→80, K=251, s=10): vDSP 15.0ms → GEMM 1.9ms (7.8x)
 *   Conv1 (80→60, K=5, s=1):      vDSP 23.0ms → GEMM 0.5ms (43x)
 *   Conv2 (60→60, K=5, s=1):      vDSP 4.9ms  → GEMM 0.2ms (20x)
 *   Total:                         vDSP 42.9ms → GEMM 2.7ms (16x)
 *
 * Root causes of vDSP slowness:
 *   1. K<8 falls entirely to scalar in conv1d_f16_neon (SIMD loop needs K>=8)
 *   2. stride>1 computes full convolution then subsamples (10x wasted FLOPs)
 *   3. Per-channel heap allocation (~56MB total for SincConv)
 *
 * This implementation converts conv1d into a single cblas_sgemm call via
 * im2col unfolding. Weights converted to FP32 once, im2col handles stride
 * natively, GEMM uses AMX coprocessor.
 *
 * See Slack thread 2026-03-27 for full benchmarks and profiling data.
 */

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include <arm_neon.h>
#include <vector>
#include <cstring>

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
                        vst1q_f32(dst + t, vcvt_f32_f16(vget_low_f16(v)));
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
                    float32x4_t v0 = vaddq_f32(vld1q_f32(src + t), vdupq_n_f32(b));
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
