#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

void cactus_conv2d_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in, size_t H, size_t W,
    size_t C_out
){
    if (H + 2 < 3 || W + 2 < 3) return;
    const size_t H_out = (H - 1) / 2 + 1;
    const size_t W_out = (W - 1) / 2 + 1;

#ifdef __APPLE__
    {
        const size_t spatial = H_out * W_out;
        const size_t col_K = C_in * 9;
        std::vector<float> W_f32(C_out * col_K);
        for (size_t oc = 0; oc < C_out; ++oc)
            for (size_t ic = 0; ic < C_in; ++ic)
                for (size_t kh = 0; kh < 3; ++kh)
                    for (size_t kw = 0; kw < 3; ++kw)
                        W_f32[oc * col_K + ic * 9 + kh * 3 + kw] =
                            static_cast<float>(weight[((oc * C_in + ic) * 3 + kh) * 3 + kw]);

        std::vector<float> bias_f32(C_out, 0.0f);
        if (bias)
            for (size_t i = 0; i < C_out; ++i)
                bias_f32[i] = static_cast<float>(bias[i]);

        std::vector<float> col(col_K * spatial);
        std::vector<float> Y_f32(C_out * spatial);

        for (size_t n = 0; n < N; ++n) {
            const __fp16* Xn = input + n * C_in * H * W;
            __fp16* Yn = output + n * C_out * spatial;

            for (size_t ic = 0; ic < C_in; ++ic) {
                for (size_t kh = 0; kh < 3; ++kh) {
                    for (size_t kw = 0; kw < 3; ++kw) {
                        float* dst = col.data() + (ic * 9 + kh * 3 + kw) * spatial;
                        for (size_t oh = 0; oh < H_out; ++oh) {
                            ptrdiff_t ih = static_cast<ptrdiff_t>(oh) * 2 + static_cast<ptrdiff_t>(kh) - 1;
                            float* dst_row = dst + oh * W_out;
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(H)) {
                                memset(dst_row, 0, W_out * sizeof(float));
                                continue;
                            }
                            const __fp16* src_row = Xn + ic * H * W + static_cast<size_t>(ih) * W;
                            const ptrdiff_t iw_offset = static_cast<ptrdiff_t>(kw) - 1;
                            for (size_t ow = 0; ow < W_out; ++ow) {
                                ptrdiff_t iw = static_cast<ptrdiff_t>(ow) * 2 + iw_offset;
                                dst_row[ow] = (iw < 0 || iw >= static_cast<ptrdiff_t>(W))
                                    ? 0.0f : static_cast<float>(src_row[iw]);
                            }
                        }
                    }
                }
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(C_out), static_cast<int>(spatial), static_cast<int>(col_K),
                        1.0f, W_f32.data(), static_cast<int>(col_K),
                        col.data(), static_cast<int>(spatial),
                        0.0f, Y_f32.data(), static_cast<int>(spatial));

            for (size_t oc = 0; oc < C_out; ++oc) {
                const float b = bias_f32[oc];
                const float* src = Y_f32.data() + oc * spatial;
                __fp16* dst = Yn + oc * spatial;
                size_t i = 0;
                for (; i + 8 <= spatial; i += 8) {
                    float32x4_t v0 = vaddq_f32(vld1q_f32(src + i),     vdupq_n_f32(b));
                    float32x4_t v1 = vaddq_f32(vld1q_f32(src + i + 4), vdupq_n_f32(b));
                    vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                }
                for (; i < spatial; ++i)
                    dst[i] = static_cast<__fp16>(src[i] + b);
            }
        }
        return;
    }
#endif

    auto in_idx = [&](size_t n, size_t ic, size_t ih, size_t iw) -> size_t {
        return ((n * C_in + ic) * H + ih) * W + iw;
    };
    auto w_idx = [&](size_t oc, size_t ic, size_t kh, size_t kw) -> size_t {
        return (((oc * C_in + ic) * 3 + kh) * 3 + kw);
    };
    auto out_idx = [&](size_t n, size_t oc, size_t oh, size_t ow) -> size_t {
        return ((n * C_out + oc) * H_out + oh) * W_out + ow;
    };

    constexpr size_t OW_VEC = 4;

    const size_t total_compute = N * C_out * H_out * W_out * C_in * 9;
    CactusThreading::ParallelConfig cfg =
        (total_compute < 100000) ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
                                 : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(N, C_out, cfg, [&](size_t n, size_t oc) {
        const float b0 = bias ? (float)bias[oc] : 0.0f;

        for (size_t oh = 0; oh < H_out; ++oh) {
            const ptrdiff_t ih0 = (ptrdiff_t)oh * 2 - 1;

            const bool h_interior = (ih0 >= 0) && ((size_t)(ih0 + 2) < H);

            for (size_t ow = 0; ow < W_out; ) {
                const size_t rem = W_out - ow;

                const bool do_vec = (rem >= OW_VEC);
                const bool w_interior_vec =
                    do_vec &&
                    (ow >= 1) &&
                    (((ow + (OW_VEC - 1)) * 2 + 1) < W);

                float32x4_t vacc = vdupq_n_f32(b0);

                if (h_interior && w_interior_vec) {
                    for (size_t ic = 0; ic < C_in; ++ic) {

                        const __fp16* row0 = input + in_idx(n, ic, (size_t)(ih0 + 0), 0);
                        const __fp16* row1 = input + in_idx(n, ic, (size_t)(ih0 + 1), 0);
                        const __fp16* row2 = input + in_idx(n, ic, (size_t)(ih0 + 2), 0);
                        const ptrdiff_t iw_base0 = (ptrdiff_t)ow * 2 - 1;
                        {
                            const float w00 = (float)weight[w_idx(oc, ic, 0, 0)];
                            const float w01 = (float)weight[w_idx(oc, ic, 0, 1)];
                            const float w02 = (float)weight[w_idx(oc, ic, 0, 2)];

                            float x0 = (float)row0[(size_t)(iw_base0 + 0)];
                            float x1 = (float)row0[(size_t)(iw_base0 + 2)];
                            float x2 = (float)row0[(size_t)(iw_base0 + 4)];
                            float x3 = (float)row0[(size_t)(iw_base0 + 6)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w00));

                            x0 = (float)row0[(size_t)(iw_base0 + 1)];
                            x1 = (float)row0[(size_t)(iw_base0 + 3)];
                            x2 = (float)row0[(size_t)(iw_base0 + 5)];
                            x3 = (float)row0[(size_t)(iw_base0 + 7)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w01));

                            x0 = (float)row0[(size_t)(iw_base0 + 2)];
                            x1 = (float)row0[(size_t)(iw_base0 + 4)];
                            x2 = (float)row0[(size_t)(iw_base0 + 6)];
                            x3 = (float)row0[(size_t)(iw_base0 + 8)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w02));
                        }

                        {
                            const float w10 = (float)weight[w_idx(oc, ic, 1, 0)];
                            const float w11 = (float)weight[w_idx(oc, ic, 1, 1)];
                            const float w12 = (float)weight[w_idx(oc, ic, 1, 2)];
                            const ptrdiff_t iw_base0 = (ptrdiff_t)ow * 2 - 1;

                            float x0 = (float)row1[(size_t)(iw_base0 + 0)];
                            float x1 = (float)row1[(size_t)(iw_base0 + 2)];
                            float x2 = (float)row1[(size_t)(iw_base0 + 4)];
                            float x3 = (float)row1[(size_t)(iw_base0 + 6)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w10));

                            x0 = (float)row1[(size_t)(iw_base0 + 1)];
                            x1 = (float)row1[(size_t)(iw_base0 + 3)];
                            x2 = (float)row1[(size_t)(iw_base0 + 5)];
                            x3 = (float)row1[(size_t)(iw_base0 + 7)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w11));

                            x0 = (float)row1[(size_t)(iw_base0 + 2)];
                            x1 = (float)row1[(size_t)(iw_base0 + 4)];
                            x2 = (float)row1[(size_t)(iw_base0 + 6)];
                            x3 = (float)row1[(size_t)(iw_base0 + 8)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w12));
                        }

                        {
                            const float w20 = (float)weight[w_idx(oc, ic, 2, 0)];
                            const float w21 = (float)weight[w_idx(oc, ic, 2, 1)];
                            const float w22 = (float)weight[w_idx(oc, ic, 2, 2)];
                            const ptrdiff_t iw_base0 = (ptrdiff_t)ow * 2 - 1;

                            float x0 = (float)row2[(size_t)(iw_base0 + 0)];
                            float x1 = (float)row2[(size_t)(iw_base0 + 2)];
                            float x2 = (float)row2[(size_t)(iw_base0 + 4)];
                            float x3 = (float)row2[(size_t)(iw_base0 + 6)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w20));

                            x0 = (float)row2[(size_t)(iw_base0 + 1)];
                            x1 = (float)row2[(size_t)(iw_base0 + 3)];
                            x2 = (float)row2[(size_t)(iw_base0 + 5)];
                            x3 = (float)row2[(size_t)(iw_base0 + 7)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w21));

                            x0 = (float)row2[(size_t)(iw_base0 + 2)];
                            x1 = (float)row2[(size_t)(iw_base0 + 4)];
                            x2 = (float)row2[(size_t)(iw_base0 + 6)];
                            x3 = (float)row2[(size_t)(iw_base0 + 8)];
                            vacc = vfmaq_f32(vacc, (float32x4_t){x0,x1,x2,x3}, vdupq_n_f32(w22));
                        }
                    }

                    __fp16* Y = output + out_idx(n, oc, oh, ow);
                    const float16x4_t yv = vcvt_f16_f32(vacc);
                    vst1_f16(Y, yv);

                    ow += OW_VEC;
                    continue;
                }

                const size_t tile = do_vec ? OW_VEC : 1;
                float acc_s[OW_VEC] = {b0, b0, b0, b0};

                for (size_t ic = 0; ic < C_in; ++ic) {
                    for (size_t kh = 0; kh < 3; ++kh) {
                        const ptrdiff_t ih = ih0 + (ptrdiff_t)kh;
                        if (ih < 0 || ih >= (ptrdiff_t)H) continue;

                        const __fp16* row = input + in_idx(n, ic, (size_t)ih, 0);

                        for (size_t kw = 0; kw < 3; ++kw) {
                            const float wv = (float)weight[w_idx(oc, ic, kh, kw)];
                            for (size_t t = 0; t < tile; ++t) {
                                const size_t ow_t = ow + t;
                                const ptrdiff_t iw = (ptrdiff_t)ow_t * 2 - 1 + (ptrdiff_t)kw;
                                if (iw < 0 || iw >= (ptrdiff_t)W) continue;
                                acc_s[t] += (float)row[(size_t)iw] * wv;
                            }
                        }
                    }
                }

                __fp16* Y = output + out_idx(n, oc, oh, ow);
                for (size_t t = 0; t < tile; ++t) {
                    Y[t] = (__fp16)acc_s[t];
                }
                ow += tile;
            }
        }
    });
}

void cactus_conv2d_depthwise_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C,
    size_t H,
    size_t W
){
    if (H + 2 < 3 || W + 2 < 3) return;
    const size_t H_out = (H - 1) / 2 + 1;
    const size_t W_out = (W - 1) / 2 + 1;

    auto in_idx = [&](size_t n, size_t c, size_t ih, size_t iw) -> size_t {
        return ((n * C + c) * H + ih) * W + iw;
    };
    auto out_idx = [&](size_t n, size_t c, size_t oh, size_t ow) -> size_t {
        return ((n * C + c) * H_out + oh) * W_out + ow;
    };

    constexpr size_t OW_VEC = 4;
    const size_t total_compute = N * C * H_out * W_out * 9;
    CactusThreading::ParallelConfig cfg =
        (total_compute < 100000) ? CactusThreading::ParallelConfig{SIZE_MAX, SIZE_MAX}
                                 : CactusThreading::Thresholds::ATTENTION;

    CactusThreading::parallel_for_2d(N, C, cfg, [&](size_t n, size_t c) {
        const __fp16* Wc = weight + c * 9;
        const float w00 = (float)Wc[0], w01 = (float)Wc[1], w02 = (float)Wc[2];
        const float w10 = (float)Wc[3], w11 = (float)Wc[4], w12 = (float)Wc[5];
        const float w20 = (float)Wc[6], w21 = (float)Wc[7], w22 = (float)Wc[8];
        const float b0 = bias ? (float)bias[c] : 0.0f;

        for (size_t oh = 0; oh < H_out; ++oh) {
            const ptrdiff_t ih0 = (ptrdiff_t)oh * 2 - 1;
            const bool h_interior = (ih0 >= 0) && ((size_t)(ih0 + 2) < H);

            for (size_t ow = 0; ow < W_out; ) {
                const size_t rem = W_out - ow;
                const bool do_vec = (rem >= OW_VEC);
                const bool w_interior_vec =
                    do_vec &&
                    (ow >= 1) &&
                    (((ow + (OW_VEC - 1)) * 2 + 1) < W);

                if (h_interior && w_interior_vec) {
                    const __fp16* row0 = input + in_idx(n, c, (size_t)(ih0 + 0), 0);
                    const __fp16* row1 = input + in_idx(n, c, (size_t)(ih0 + 1), 0);
                    const __fp16* row2 = input + in_idx(n, c, (size_t)(ih0 + 2), 0);
                    const ptrdiff_t iw_base0 = (ptrdiff_t)ow * 2 - 1;

                    float32x4_t vacc = vdupq_n_f32(b0);
                    float x0, x1, x2, x3;

                    x0 = (float)row0[(size_t)(iw_base0 + 0)];
                    x1 = (float)row0[(size_t)(iw_base0 + 2)];
                    x2 = (float)row0[(size_t)(iw_base0 + 4)];
                    x3 = (float)row0[(size_t)(iw_base0 + 6)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w00));

                    x0 = (float)row0[(size_t)(iw_base0 + 1)];
                    x1 = (float)row0[(size_t)(iw_base0 + 3)];
                    x2 = (float)row0[(size_t)(iw_base0 + 5)];
                    x3 = (float)row0[(size_t)(iw_base0 + 7)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w01));

                    x0 = (float)row0[(size_t)(iw_base0 + 2)];
                    x1 = (float)row0[(size_t)(iw_base0 + 4)];
                    x2 = (float)row0[(size_t)(iw_base0 + 6)];
                    x3 = (float)row0[(size_t)(iw_base0 + 8)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w02));

                    x0 = (float)row1[(size_t)(iw_base0 + 0)];
                    x1 = (float)row1[(size_t)(iw_base0 + 2)];
                    x2 = (float)row1[(size_t)(iw_base0 + 4)];
                    x3 = (float)row1[(size_t)(iw_base0 + 6)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w10));

                    x0 = (float)row1[(size_t)(iw_base0 + 1)];
                    x1 = (float)row1[(size_t)(iw_base0 + 3)];
                    x2 = (float)row1[(size_t)(iw_base0 + 5)];
                    x3 = (float)row1[(size_t)(iw_base0 + 7)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w11));

                    x0 = (float)row1[(size_t)(iw_base0 + 2)];
                    x1 = (float)row1[(size_t)(iw_base0 + 4)];
                    x2 = (float)row1[(size_t)(iw_base0 + 6)];
                    x3 = (float)row1[(size_t)(iw_base0 + 8)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w12));

                    x0 = (float)row2[(size_t)(iw_base0 + 0)];
                    x1 = (float)row2[(size_t)(iw_base0 + 2)];
                    x2 = (float)row2[(size_t)(iw_base0 + 4)];
                    x3 = (float)row2[(size_t)(iw_base0 + 6)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w20));

                    x0 = (float)row2[(size_t)(iw_base0 + 1)];
                    x1 = (float)row2[(size_t)(iw_base0 + 3)];
                    x2 = (float)row2[(size_t)(iw_base0 + 5)];
                    x3 = (float)row2[(size_t)(iw_base0 + 7)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w21));

                    x0 = (float)row2[(size_t)(iw_base0 + 2)];
                    x1 = (float)row2[(size_t)(iw_base0 + 4)];
                    x2 = (float)row2[(size_t)(iw_base0 + 6)];
                    x3 = (float)row2[(size_t)(iw_base0 + 8)];
                    vacc = vfmaq_f32(vacc, (float32x4_t){x0, x1, x2, x3}, vdupq_n_f32(w22));

                    __fp16* Y = output + out_idx(n, c, oh, ow);
                    vst1_f16(Y, vcvt_f16_f32(vacc));
                    ow += OW_VEC;
                    continue;
                }

                const size_t tile = do_vec ? OW_VEC : 1;
                float acc_s[OW_VEC] = {b0, b0, b0, b0};
                for (size_t t = 0; t < tile; ++t) {
                    const size_t ow_t = ow + t;
                    for (size_t kh = 0; kh < 3; ++kh) {
                        const ptrdiff_t ih = ih0 + (ptrdiff_t)kh;
                        if (ih < 0 || ih >= (ptrdiff_t)H) continue;

                        const __fp16* row = input + in_idx(n, c, (size_t)ih, 0);
                        for (size_t kw = 0; kw < 3; ++kw) {
                            const ptrdiff_t iw = (ptrdiff_t)ow_t * 2 - 1 + (ptrdiff_t)kw;
                            if (iw < 0 || iw >= (ptrdiff_t)W) continue;
                            const float wv = (float)Wc[kh * 3 + kw];
                            acc_s[t] += (float)row[(size_t)iw] * wv;
                        }
                    }
                }

                __fp16* Y = output + out_idx(n, c, oh, ow);
                for (size_t t = 0; t < tile; ++t) {
                    Y[t] = (__fp16)acc_s[t];
                }
                ow += tile;
            }
        }
    });
}

void cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out
){
    const size_t HW = H * W;
    if (HW == 0 || N == 0) return;

    CactusThreading::parallel_for(
        N,
        CactusThreading::Thresholds::ATTENTION,
        [&](size_t n_start, size_t n_end) {
            std::vector<__fp16> packed_in(HW * C_in);
            std::vector<__fp16> packed_out(HW * C_out);

            for (size_t n = n_start; n < n_end; ++n) {
                const __fp16* Xn = input + n * C_in * HW;
                __fp16* Yn = output + n * C_out * HW;

                for (size_t hw = 0; hw < HW; ++hw) {
                    const size_t h = hw / W;
                    const size_t w = hw - h * W;
                    __fp16* row = packed_in.data() + hw * C_in;
                    for (size_t ic = 0; ic < C_in; ++ic) {
                        row[ic] = Xn[(ic * H + h) * W + w];
                    }
                }

                cactus_matmul_f16(
                    packed_in.data(),
                    weight,
                    packed_out.data(),
                    HW,
                    C_in,
                    C_out
                );

                for (size_t hw = 0; hw < HW; ++hw) {
                    const size_t h = hw / W;
                    const size_t w = hw - h * W;
                    const __fp16* row = packed_out.data() + hw * C_out;
                    for (size_t oc = 0; oc < C_out; ++oc) {
                        float outv = (float)row[oc];
                        if (bias) outv += (float)bias[oc];
                        Yn[(oc * H + h) * W + w] = (__fp16)outv;
                    }
                }
            }
        });
}

void cactus_conv1d_pointwise_f16_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out
){
    const size_t M = N * L;
    if (M == 0) return;

    cactus_matmul_f16(input, weight, output, M, C_in, C_out);

    if (bias) {
        for (size_t m = 0; m < M; ++m) {
            __fp16* row = output + m * C_out;
            for (size_t oc = 0; oc < C_out; ++oc) {
                row[oc] = (__fp16)((float)row[oc] + (float)bias[oc]);
            }
        }
    }
}

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
                        size_t ow_start = 0, ow_end = W_out;
                        if (iw_offset < 0) { dst_row[0] = 0.0f; ow_start = 1; }
                        if (iw_offset > 0) { dst_row[W_out - 1] = 0.0f; ow_end = W_out - 1; }
                        size_t ow = ow_start;
                        for (; ow + 8 <= ow_end; ow += 8) {
                            float16x8_t v = vld1q_f16(src_row + ow + iw_offset);
                            vst1q_f32(dst_row + ow,     vcvt_f32_f16(vget_low_f16(v)));
                            vst1q_f32(dst_row + ow + 4, vcvt_f32_f16(vget_high_f16(v)));
                        }
                        for (; ow < ow_end; ++ow)
                            dst_row[ow] = static_cast<float>(src_row[ow + iw_offset]);
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

void cactus_maxpool1d_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t channels,
    size_t input_length,
    size_t kernel_size,
    size_t stride
) {
    const size_t output_length = (input_length - kernel_size) / stride + 1;

    CactusThreading::parallel_for(batch_size * channels, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t bc = start; bc < end; ++bc) {
                const size_t b = bc / channels;
                const size_t c = bc % channels;

                const __fp16* src = input + b * channels * input_length + c * input_length;
                __fp16* dst = output + b * channels * output_length + c * output_length;

                for (size_t i = 0; i < output_length; ++i) {
                    const size_t in_start = i * stride;
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t k = 0; k < kernel_size; ++k) {
                        float val = static_cast<float>(src[in_start + k]);
                        if (val > max_val) max_val = val;
                    }
                    dst[i] = static_cast<__fp16>(max_val);
                }
            }
        });
}
