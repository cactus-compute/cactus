// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"

#if defined(LM_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double lm_ggml_float;

#define LM_GGML_GELU_FP16
#define LM_GGML_GELU_QUICK_FP16

#define LM_GGML_SOFT_MAX_UNROLL 4
#define LM_GGML_VEC_DOT_UNROLL  2
#define LM_GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
extern "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
extern lm_ggml_fp16_t lm_ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern lm_ggml_fp16_t lm_ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void lm_ggml_vec_dot_f32(int n, float * LM_GGML_RESTRICT s, size_t bs, const float * LM_GGML_RESTRICT x, size_t bx, const float * LM_GGML_RESTRICT y, size_t by, int nrc);
void lm_ggml_vec_dot_bf16(int n, float * LM_GGML_RESTRICT s, size_t bs, lm_ggml_bf16_t * LM_GGML_RESTRICT x, size_t bx, lm_ggml_bf16_t * LM_GGML_RESTRICT y, size_t by, int nrc);
void lm_ggml_vec_dot_f16(int n, float * LM_GGML_RESTRICT s, size_t bs, lm_ggml_fp16_t * LM_GGML_RESTRICT x, size_t bx, lm_ggml_fp16_t * LM_GGML_RESTRICT y, size_t by, int nrc);

void lm_ggml_vec_silu_f32(const int n, float * y, const float * x);
lm_ggml_float lm_ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
lm_ggml_float lm_ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max);

inline static void lm_ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void lm_ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_set_i32(const int n, int32_t * x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v;    }
inline static void lm_ggml_vec_cpy_i32(const int n, int32_t * y, const int32_t * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void lm_ggml_vec_set_f16(const int n, lm_ggml_fp16_t * x, const lm_ggml_fp16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void lm_ggml_vec_set_bf16(const int n, lm_ggml_bf16_t * x, const lm_ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void lm_ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void lm_ggml_vec_add_f16 (const int n, lm_ggml_fp16_t * z, const lm_ggml_fp16_t * x, const lm_ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(x[i]) + LM_GGML_FP16_TO_FP32(y[i]));
    }
}
inline static void lm_ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void lm_ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void lm_ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void lm_ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void lm_ggml_vec_sub_f16 (const int n, lm_ggml_fp16_t * z, const lm_ggml_fp16_t * x, const lm_ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(x[i]) - LM_GGML_FP16_TO_FP32(y[i]));
    }
}
inline static void lm_ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void lm_ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void lm_ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void lm_ggml_vec_neg_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(-LM_GGML_FP16_TO_FP32(x[i]));
    }
}

inline static void lm_ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void lm_ggml_vec_mul_f16 (const int n, lm_ggml_fp16_t * z, const lm_ggml_fp16_t * x, const lm_ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(x[i]) * LM_GGML_FP16_TO_FP32(y[i]));
    }
}
inline static void lm_ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
inline static void lm_ggml_vec_div_f16 (const int n, lm_ggml_fp16_t * z, const lm_ggml_fp16_t * x, const lm_ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(x[i]) / LM_GGML_FP16_TO_FP32(y[i]));
    }
}

// compute LM_GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void lm_ggml_vec_dot_f16_unroll(const int n, const int xs, float * LM_GGML_RESTRICT s, void * LM_GGML_RESTRICT xv, lm_ggml_fp16_t * LM_GGML_RESTRICT y) {
    lm_ggml_float sumf[LM_GGML_VEC_DOT_UNROLL] = { 0.0 };

    lm_ggml_fp16_t * LM_GGML_RESTRICT x[LM_GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < LM_GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (lm_ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC sum[LM_GGML_VEC_DOT_UNROLL][LM_GGML_F16_ARR] = { { LM_GGML_F16_VEC_ZERO } };

    LM_GGML_F16_VEC ax[LM_GGML_F16_ARR];
    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);

            for (int k = 0; k < LM_GGML_VEC_DOT_UNROLL; ++k) {
                ax[j] = LM_GGML_F16_VEC_LOAD(x[k] + i + j*LM_GGML_F16_EPR, j);

                sum[k][j] = LM_GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
            }
        }
    }

    // reduce sum0..sum3 to sum0
    for (int k = 0; k < LM_GGML_VEC_DOT_UNROLL; ++k) {
        LM_GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < LM_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[j][i])*LM_GGML_FP16_TO_FP32(y[i]));
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < LM_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[j][i])*LM_GGML_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < LM_GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float)sumf[i];
    }
}

inline static void lm_ggml_vec_mad_f32(const int n, float * LM_GGML_RESTRICT y, const float * LM_GGML_RESTRICT x, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx = LM_GGML_F32_VEC_SET1(v);

    LM_GGML_F32_VEC ax[LM_GGML_F32_ARR];
    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ax[j] = LM_GGML_F32_VEC_LOAD(x + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void lm_ggml_vec_mad_f16(const int n, lm_ggml_fp16_t * LM_GGML_RESTRICT y, const lm_ggml_fp16_t * LM_GGML_RESTRICT x, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC vx = LM_GGML_F16_VEC_SET1(v);

    LM_GGML_F16_VEC ax[LM_GGML_F16_ARR];
    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ax[j] = LM_GGML_F16_VEC_LOAD(x + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            LM_GGML_F16_VEC_STORE(y + i + j*LM_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i]) + LM_GGML_FP16_TO_FP32(x[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i]) + LM_GGML_FP16_TO_FP32(x[i])*v);
    }
#endif
}

// xs and vs are byte strides of x and v
inline static void lm_ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * LM_GGML_RESTRICT y, const float * LM_GGML_RESTRICT xv, const float * LM_GGML_RESTRICT vv) {

    const float * LM_GGML_RESTRICT x[LM_GGML_VEC_MAD_UNROLL];
    const float * LM_GGML_RESTRICT v[LM_GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < LM_GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx[LM_GGML_VEC_MAD_UNROLL];

    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        vx[k] = LM_GGML_F32_VEC_SET1(v[k][0]);
    }

    LM_GGML_F32_VEC ax[LM_GGML_VEC_MAD_UNROLL][LM_GGML_F32_ARR];
    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);

            for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
                ax[k][j] = LM_GGML_F32_VEC_LOAD(x[k] + i + j*LM_GGML_F32_EPR);
                ay[j] = LM_GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
            }

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = np; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#else
    // scalar
    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

//inline static void lm_ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void lm_ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(LM_GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx = LM_GGML_F32_VEC_SET1(v);

    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_MUL(ay[j], vx);

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void lm_ggml_vec_scale_f16(const int n, lm_ggml_fp16_t * y, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC vx = LM_GGML_F16_VEC_SET1(v);

    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_MUL(ay[j], vx);

            LM_GGML_F16_VEC_STORE(y + i + j*LM_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void lm_ggml_vec_norm_f32 (const int n, float * s, const float * x) { lm_ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void lm_ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void lm_ggml_vec_sqr_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16(v*v);
    }
}
inline static void lm_ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void lm_ggml_vec_sqrt_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(sqrtf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void lm_ggml_vec_log_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(logf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void lm_ggml_vec_sin_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(sinf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void lm_ggml_vec_cos_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(cosf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void lm_ggml_vec_abs_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(fabsf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void lm_ggml_vec_sgn_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16((v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f));
    }
}
inline static void lm_ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void lm_ggml_vec_step_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16((LM_GGML_FP16_TO_FP32(x[i]) > 0.f) ? 1.f : 0.f);
    }
}
inline static void lm_ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void lm_ggml_vec_tanh_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(tanhf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void lm_ggml_vec_elu_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(expm1f(LM_GGML_FP16_TO_FP32(x[i])));
    }
}
inline static void lm_ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void lm_ggml_vec_relu_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16((v > 0.f) ? v : 0.f);
    }
}
inline static void lm_ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void lm_ggml_vec_leaky_relu_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x, const float ns) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
    }
}
inline static void lm_ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
inline static void lm_ggml_vec_sigmoid_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(1.f / (1.f + expf(-LM_GGML_FP16_TO_FP32(x[i]))));
    }
}
// TODO: optimize performance
inline static void lm_ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void lm_ggml_vec_hardswish_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16(v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)));
    }
}
inline static void lm_ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void lm_ggml_vec_hardsigmoid_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (LM_GGML_FP16_TO_FP32(x[i]) + 3.0f) / 6.0f)));
    }
}
inline static void lm_ggml_vec_exp_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
inline static void lm_ggml_vec_exp_f16 (const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(expf(LM_GGML_FP16_TO_FP32(x[i])));
    }
}

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
static const float SQRT_2_INV      = 0.70710678118654752440084436210484f;

inline static float lm_ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void lm_ggml_vec_gelu_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_table_gelu_f16[i16[i]];
    }
}

inline static void lm_ggml_vec_gelu_erf_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float xi = LM_GGML_FP16_TO_FP32(x[i]);
        float res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
        y[i] = LM_GGML_FP32_TO_FP16(res);
    }
}

#ifdef LM_GGML_GELU_FP16
inline static void lm_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            lm_ggml_fp16_t fp16 = LM_GGML_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = LM_GGML_FP16_TO_FP32(lm_ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void lm_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_gelu_f32(x[i]);
    }
}
#endif

inline static void lm_ggml_vec_gelu_erf_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
    }
}

inline static float lm_ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

//inline static void lm_ggml_vec_gelu_quick_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = lm_ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

#ifdef LM_GGML_GELU_QUICK_FP16
inline static void lm_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        lm_ggml_fp16_t fp16 = LM_GGML_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = LM_GGML_FP16_TO_FP32(lm_ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void lm_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_gelu_quick_f32(x[i]);
    }
}
#endif

inline static void lm_ggml_vec_gelu_quick_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = LM_GGML_FP16_TO_FP32(x[i]);
        y[i] = LM_GGML_FP32_TO_FP16(v*(1.0f/(1.0f+expf(GELU_QUICK_COEF*v))));
    }
}

// Sigmoid Linear Unit (SiLU) function
inline static float lm_ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}
inline static lm_ggml_fp16_t lm_ggml_silu_f16(lm_ggml_fp16_t x) {
    float v = LM_GGML_FP16_TO_FP32(x);
    return LM_GGML_FP32_TO_FP16(v/(1.0f + expf(-v)));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t lm_ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t lm_ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = lm_ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 lm_ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 lm_ggml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 lm_ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 lm_ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 lm_ggml_v_expf(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c))
        return MADD128(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 lm_ggml_v_silu(__m128 x) {
    const __m128 one = _mm_set1_ps(1);
    const __m128 zero = _mm_setzero_ps();
    const __m128 neg_x = _mm_sub_ps(zero, x);
    const __m128 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    return _mm_div_ps(x, one_plus_exp_neg_x);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__

inline static void lm_ggml_vec_silu_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_silu_f16(x[i]);
    }
}

inline static float lm_ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static lm_ggml_fp16_t lm_ggml_silu_backward_f16(lm_ggml_fp16_t x, lm_ggml_fp16_t dy) {
    const float v = LM_GGML_FP16_TO_FP32(x);
    const float s = 1.0f/(1.0f + expf(-v));
    return LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(dy)*s*(1.0f + v*(1.0f - s)));
}

inline static void lm_ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = lm_ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void lm_ggml_vec_silu_backward_f16(const int n, lm_ggml_fp16_t * dx, const lm_ggml_fp16_t * x, const lm_ggml_fp16_t * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = lm_ggml_silu_backward_f16(x[i], dy[i]);
    }
}

inline static void lm_ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef LM_GGML_USE_ACCELERATE
    lm_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (lm_ggml_float)x[i];
    }
    *s = (float)sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void lm_ggml_vec_sum_f32_ggf(const int n, lm_ggml_float * s, const float * x) {
    lm_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (lm_ggml_float)x[i];
    }
    *s = sum;
}

inline static void lm_ggml_vec_sum_f16_ggf(const int n, float * s, const lm_ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += LM_GGML_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void lm_ggml_vec_sum_bf16_ggf(const int n, float * s, const lm_ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += LM_GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void lm_ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef LM_GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void lm_ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    lm_ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void lm_ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

#ifdef __cplusplus
}
#endif
