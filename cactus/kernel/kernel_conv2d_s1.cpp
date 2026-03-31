#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <vector>
#include <cstring>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

void cactus_conv2d_f16_k3s1p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in, size_t H, size_t W,
    size_t C_out
) {
    const size_t H_out = H;
    const size_t W_out = W;

#ifdef __APPLE__
    const size_t col_K = C_in * 9;
    std::vector<float> W_f32(C_out * col_K);
    for (size_t oc = 0; oc < C_out; ++oc) {
        for (size_t ic = 0; ic < C_in; ++ic) {
            for (size_t kh = 0; kh < 3; ++kh) {
                for (size_t kw = 0; kw < 3; ++kw) {
                    W_f32[oc * col_K + ic * 9 + kh * 3 + kw] =
                        static_cast<float>(weight[((oc * C_in + ic) * 3 + kh) * 3 + kw]);
                }
            }
        }
    }

    std::vector<float> bias_f32(C_out, 0.0f);
    if (bias) {
        for (size_t i = 0; i < C_out; ++i)
            bias_f32[i] = static_cast<float>(bias[i]);
    }

    std::vector<float> col(col_K * H_out * W_out);
    std::vector<float> Y_f32(C_out * H_out * W_out);

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xn = input + n * C_in * H * W;
        __fp16* Yn = output + n * C_out * H_out * W_out;

        for (size_t ic = 0; ic < C_in; ++ic) {
            for (size_t kh = 0; kh < 3; ++kh) {
                for (size_t kw = 0; kw < 3; ++kw) {
                    float* dst = col.data() + (ic * 9 + kh * 3 + kw) * H_out * W_out;
                    for (size_t oh = 0; oh < H_out; ++oh) {
                        ptrdiff_t ih = static_cast<ptrdiff_t>(oh) + static_cast<ptrdiff_t>(kh) - 1;
                        float* dst_row = dst + oh * W_out;
                        if (ih < 0 || ih >= static_cast<ptrdiff_t>(H)) {
                            memset(dst_row, 0, W_out * sizeof(float));
                            continue;
                        }
                        const __fp16* src_row = Xn + ic * H * W + static_cast<size_t>(ih) * W;
                        const ptrdiff_t iw_offset = static_cast<ptrdiff_t>(kw) - 1;
                        for (size_t ow = 0; ow < W_out; ++ow) {
                            ptrdiff_t iw = static_cast<ptrdiff_t>(ow) + iw_offset;
                            dst_row[ow] = (iw < 0 || iw >= static_cast<ptrdiff_t>(W))
                                ? 0.0f : static_cast<float>(src_row[iw]);
                        }
                    }
                }
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(C_out), static_cast<int>(H_out * W_out), static_cast<int>(col_K),
                    1.0f, W_f32.data(), static_cast<int>(col_K),
                    col.data(), static_cast<int>(H_out * W_out),
                    0.0f, Y_f32.data(), static_cast<int>(H_out * W_out));

        for (size_t oc = 0; oc < C_out; ++oc) {
            float b = bias_f32[oc];
            const float* src = Y_f32.data() + oc * H_out * W_out;
            __fp16* dst = Yn + oc * H_out * W_out;
            size_t i = 0;
            for (; i + 8 <= H_out * W_out; i += 8) {
                float32x4_t v0 = vaddq_f32(vld1q_f32(src + i), vdupq_n_f32(b));
                float32x4_t v1 = vaddq_f32(vld1q_f32(src + i + 4), vdupq_n_f32(b));
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
            }
            for (; i < H_out * W_out; ++i)
                dst[i] = static_cast<__fp16>(src[i] + b);
        }
    }

#else
    const size_t total_compute = N * C_out * H_out * W_out * C_in * 9;
    CactusThreading::ParallelConfig cfg =
        (total_compute < 100000) ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
                                 : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(N, C_out, cfg, [&](size_t n, size_t oc) {
        const float b0 = bias ? static_cast<float>(bias[oc]) : 0.0f;
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                float acc = b0;
                for (size_t ic = 0; ic < C_in; ++ic) {
                    for (size_t kh = 0; kh < 3; ++kh) {
                        for (size_t kw = 0; kw < 3; ++kw) {
                            const ptrdiff_t ih = static_cast<ptrdiff_t>(oh) + kh - 1;
                            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) + kw - 1;
                            if (ih >= 0 && ih < static_cast<ptrdiff_t>(H) &&
                                iw >= 0 && iw < static_cast<ptrdiff_t>(W)) {
                                acc += static_cast<float>(input[((n * C_in + ic) * H + ih) * W + iw]) *
                                       static_cast<float>(weight[((oc * C_in + ic) * 3 + kh) * 3 + kw]);
                            }
                        }
                    }
                }
                output[((n * C_out + oc) * H_out + oh) * W_out + ow] = static_cast<__fp16>(acc);
            }
        }
    });
#endif
}
