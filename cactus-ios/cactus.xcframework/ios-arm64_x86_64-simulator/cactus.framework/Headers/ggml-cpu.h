#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for lm_ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct lm_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `lm_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `lm_ggml_graph_compute()`

        int n_threads;
        struct lm_ggml_threadpool * threadpool;

        // abort lm_ggml_graph_compute when true
        lm_ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // numa strategies
    enum lm_ggml_numa_strategy {
        LM_GGML_NUMA_STRATEGY_DISABLED   = 0,
        LM_GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        LM_GGML_NUMA_STRATEGY_ISOLATE    = 2,
        LM_GGML_NUMA_STRATEGY_NUMACTL    = 3,
        LM_GGML_NUMA_STRATEGY_MIRROR     = 4,
        LM_GGML_NUMA_STRATEGY_COUNT
    };

    LM_GGML_BACKEND_API void    lm_ggml_numa_init(enum lm_ggml_numa_strategy numa); // call once for better performance on NUMA systems
    LM_GGML_BACKEND_API bool    lm_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_new_i32(struct lm_ggml_context * ctx, int32_t value);
    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_new_f32(struct lm_ggml_context * ctx, float value);

    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_set_i32 (struct lm_ggml_tensor * tensor, int32_t value);
    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_set_f32 (struct lm_ggml_tensor * tensor, float value);

    LM_GGML_BACKEND_API int32_t lm_ggml_get_i32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_BACKEND_API void    lm_ggml_set_i32_1d(const struct lm_ggml_tensor * tensor, int i, int32_t value);

    LM_GGML_BACKEND_API int32_t lm_ggml_get_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_BACKEND_API void    lm_ggml_set_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    LM_GGML_BACKEND_API float   lm_ggml_get_f32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_BACKEND_API void    lm_ggml_set_f32_1d(const struct lm_ggml_tensor * tensor, int i, float value);

    LM_GGML_BACKEND_API float   lm_ggml_get_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_BACKEND_API void    lm_ggml_set_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    LM_GGML_BACKEND_API struct lm_ggml_threadpool *      lm_ggml_threadpool_new           (struct lm_ggml_threadpool_params  * params);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_free          (struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API int                           lm_ggml_threadpool_get_n_threads (struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_pause         (struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_resume        (struct lm_ggml_threadpool * threadpool);

    // lm_ggml_graph_plan() has to be called before lm_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    LM_GGML_BACKEND_API struct lm_ggml_cplan lm_ggml_graph_plan(
                  const struct lm_ggml_cgraph * cgraph,
                                       int   n_threads, /* = LM_GGML_DEFAULT_N_THREADS */
                    struct lm_ggml_threadpool * threadpool /* = NULL */ );
    LM_GGML_BACKEND_API enum lm_ggml_status  lm_ggml_graph_compute(struct lm_ggml_cgraph * cgraph, struct lm_ggml_cplan * cplan);

    // same as lm_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    LM_GGML_BACKEND_API enum lm_ggml_status  lm_ggml_graph_compute_with_ctx(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_sse3       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_ssse3      (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx_vnni   (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx2       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_bmi2       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_f16c       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_fma        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512     (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_vbmi(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_vnni(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_bf16(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_amx_int8   (void);
    // ARM
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_neon       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_arm_fma    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_fp16_va    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_dotprod    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_matmul_int8(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_sve        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_sme        (void);
    // other
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_riscv_v    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_vsx        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_vxe        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_wasm_simd  (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*lm_ggml_vec_dot_t)  (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT x, size_t bx,
                                       const void * LM_GGML_RESTRICT y, size_t by, int nrc);

    struct lm_ggml_type_traits_cpu {
        lm_ggml_from_float_t        from_float;
        lm_ggml_vec_dot_t           vec_dot;
        enum lm_ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    LM_GGML_BACKEND_API const struct lm_ggml_type_traits_cpu * lm_ggml_get_type_traits_cpu(enum lm_ggml_type type);

    LM_GGML_BACKEND_API void lm_ggml_cpu_init(void);

    //
    // CPU backend
    //

    LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_cpu_init(void);

    LM_GGML_BACKEND_API bool lm_ggml_backend_is_cpu                (lm_ggml_backend_t backend);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_n_threads     (lm_ggml_backend_t backend_cpu, int n_threads);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_threadpool    (lm_ggml_backend_t backend_cpu, lm_ggml_threadpool_t threadpool);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_abort_callback(lm_ggml_backend_t backend_cpu, lm_ggml_abort_callback abort_callback, void * abort_callback_data);

    LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_cpu_reg(void);

    LM_GGML_BACKEND_API void lm_ggml_cpu_fp32_to_fp16(const float *, lm_ggml_fp16_t *, int64_t);
    LM_GGML_BACKEND_API void lm_ggml_cpu_fp16_to_fp32(const lm_ggml_fp16_t *, float *, int64_t);
    LM_GGML_BACKEND_API void lm_ggml_cpu_fp32_to_bf16(const float *, lm_ggml_bf16_t *, int64_t);
    LM_GGML_BACKEND_API void lm_ggml_cpu_bf16_to_fp32(const lm_ggml_bf16_t *, float *, int64_t);

#ifdef __cplusplus
}
#endif
