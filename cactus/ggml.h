#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct lm_ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct lm_ggml_context * ctx = lm_ggml_init(params);
//
//       struct lm_ggml_tensor * x = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//
//       lm_ggml_set_param(ctx, x); // x is an input variable
//
//       struct lm_ggml_tensor * a  = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//       struct lm_ggml_tensor * b  = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//       struct lm_ggml_tensor * x2 = lm_ggml_mul(ctx, x, x);
//       struct lm_ggml_tensor * f  = lm_ggml_add(ctx, lm_ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct lm_ggml_cgraph * gf = lm_ggml_new_graph(ctx);
//       lm_ggml_build_forward_expand(gf, f);
//
//       // set the input variable and parameter values
//       lm_ggml_set_f32(x, 2.0f);
//       lm_ggml_set_f32(a, 3.0f);
//       lm_ggml_set_f32(b, 4.0f);
//
//       lm_ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", lm_ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the lm_ggml_graph_compute() function.
//
// The lm_ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// lm_ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the lm_ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The lm_ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the lm_ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - lm_ggml_permute()
//   - lm_ggml_conv_1d_1s()
//   - lm_ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct lm_ggml_tensor)
//
// The tensors are stored in memory via the lm_ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct lm_ggml_tensor * c = lm_ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The lm_ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct lm_ggml_tensor * a = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as lm_ggml_get_f32_1d() and lm_ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (lm_ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef LM_GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LM_GGML_BUILD
#            define LM_GGML_API __declspec(dllexport) extern
#        else
#            define LM_GGML_API __declspec(dllimport) extern
#        endif
#    else
#        define LM_GGML_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define LM_GGML_API extern
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define LM_GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define LM_GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define LM_GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define LM_GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define LM_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LM_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define LM_GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define LM_GGML_FILE_VERSION 2

#define LM_GGML_QNT_VERSION        2    // bump this on quantization format changes
#define LM_GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define LM_GGML_MAX_DIMS           4
#define LM_GGML_MAX_PARAMS         2048
#define LM_GGML_MAX_SRC            10
#define LM_GGML_MAX_N_THREADS      512
#define LM_GGML_MAX_OP_PARAMS      64

#ifndef LM_GGML_MAX_NAME
#   define LM_GGML_MAX_NAME        64
#endif

#define LM_GGML_DEFAULT_N_THREADS  4
#define LM_GGML_DEFAULT_GRAPH_SIZE 2048

#if UINTPTR_MAX == 0xFFFFFFFF
    #define LM_GGML_MEM_ALIGN 4
#else
    #define LM_GGML_MEM_ALIGN 16
#endif

#define LM_GGML_EXIT_SUCCESS 0
#define LM_GGML_EXIT_ABORTED 1

#define LM_GGML_ROPE_TYPE_NEOX   2
#define LM_GGML_ROPE_TYPE_MROPE  8
#define LM_GGML_ROPE_TYPE_VISION 24

#define LM_GGML_UNUSED(x) (void)(x)

#define LM_GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
#   define LM_GGML_UNREACHABLE() do { fprintf(stderr, "statement should be unreachable\n"); abort(); } while(0)
#elif defined(__GNUC__)
#   define LM_GGML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#   define LM_GGML_UNREACHABLE() __assume(0)
#else
#   define LM_GGML_UNREACHABLE() ((void) 0)
#endif

#ifdef __cplusplus
#   define LM_GGML_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#   define LM_GGML_NORETURN __declspec(noreturn)
#else
#   define LM_GGML_NORETURN _Noreturn
#endif

#define LM_GGML_ABORT(...) lm_ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define LM_GGML_ASSERT(x) if (!(x)) LM_GGML_ABORT("LM_GGML_ASSERT(%s) failed", #x)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define LM_GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    LM_GGML_UNUSED(prefix##0);
#define LM_GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    LM_GGML_UNUSED(prefix##1);
#define LM_GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    LM_GGML_UNUSED(prefix##2);
#define LM_GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    LM_GGML_UNUSED(prefix##3);

#define LM_GGML_TENSOR_UNARY_OP_LOCALS \
    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define LM_GGML_TENSOR_BINARY_OP_LOCALS \
    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define LM_GGML_TENSOR_BINARY_OP_LOCALS01 \
    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

#ifdef  __cplusplus
extern "C" {
#endif

    LM_GGML_NORETURN LM_GGML_ATTRIBUTE_FORMAT(3, 4)
    LM_GGML_API void lm_ggml_abort(const char * file, int line, const char * fmt, ...);

    enum lm_ggml_status {
        LM_GGML_STATUS_ALLOC_FAILED = -2,
        LM_GGML_STATUS_FAILED = -1,
        LM_GGML_STATUS_SUCCESS = 0,
        LM_GGML_STATUS_ABORTED = 1,
    };

    // get lm_ggml_status name string
    LM_GGML_API const char * lm_ggml_status_to_string(enum lm_ggml_status status);

    // ieee 754-2008 half-precision float16
    // todo: make this not an integral type
    typedef uint16_t lm_ggml_fp16_t;
    LM_GGML_API float       lm_ggml_fp16_to_fp32(lm_ggml_fp16_t);
    LM_GGML_API lm_ggml_fp16_t lm_ggml_fp32_to_fp16(float);
    LM_GGML_API void        lm_ggml_fp16_to_fp32_row(const lm_ggml_fp16_t *, float *, int64_t);
    LM_GGML_API void        lm_ggml_fp32_to_fp16_row(const float *, lm_ggml_fp16_t *, int64_t);

    // google brain half-precision bfloat16
    typedef struct { uint16_t bits; } lm_ggml_bf16_t;
    LM_GGML_API lm_ggml_bf16_t lm_ggml_fp32_to_bf16(float);
    LM_GGML_API float       lm_ggml_bf16_to_fp32(lm_ggml_bf16_t);  // consider just doing << 16
    LM_GGML_API void        lm_ggml_bf16_to_fp32_row(const lm_ggml_bf16_t *, float *, int64_t);
    LM_GGML_API void        lm_ggml_fp32_to_bf16_row_ref(const float *, lm_ggml_bf16_t *, int64_t);
    LM_GGML_API void        lm_ggml_fp32_to_bf16_row(const float *, lm_ggml_bf16_t *, int64_t);

    struct lm_ggml_object;
    struct lm_ggml_context;
    struct lm_ggml_cgraph;

    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum lm_ggml_type {
        LM_GGML_TYPE_F32     = 0,
        LM_GGML_TYPE_F16     = 1,
        LM_GGML_TYPE_Q4_0    = 2,
        LM_GGML_TYPE_Q4_1    = 3,
        // LM_GGML_TYPE_Q4_2 = 4, support has been removed
        // LM_GGML_TYPE_Q4_3 = 5, support has been removed
        LM_GGML_TYPE_Q5_0    = 6,
        LM_GGML_TYPE_Q5_1    = 7,
        LM_GGML_TYPE_Q8_0    = 8,
        LM_GGML_TYPE_Q8_1    = 9,
        LM_GGML_TYPE_Q2_K    = 10,
        LM_GGML_TYPE_Q3_K    = 11,
        LM_GGML_TYPE_Q4_K    = 12,
        LM_GGML_TYPE_Q5_K    = 13,
        LM_GGML_TYPE_Q6_K    = 14,
        LM_GGML_TYPE_Q8_K    = 15,
        LM_GGML_TYPE_IQ2_XXS = 16,
        LM_GGML_TYPE_IQ2_XS  = 17,
        LM_GGML_TYPE_IQ3_XXS = 18,
        LM_GGML_TYPE_IQ1_S   = 19,
        LM_GGML_TYPE_IQ4_NL  = 20,
        LM_GGML_TYPE_IQ3_S   = 21,
        LM_GGML_TYPE_IQ2_S   = 22,
        LM_GGML_TYPE_IQ4_XS  = 23,
        LM_GGML_TYPE_I8      = 24,
        LM_GGML_TYPE_I16     = 25,
        LM_GGML_TYPE_I32     = 26,
        LM_GGML_TYPE_I64     = 27,
        LM_GGML_TYPE_F64     = 28,
        LM_GGML_TYPE_IQ1_M   = 29,
        LM_GGML_TYPE_BF16    = 30,
        // LM_GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // LM_GGML_TYPE_Q4_0_4_8 = 32,
        // LM_GGML_TYPE_Q4_0_8_8 = 33,
        LM_GGML_TYPE_TQ1_0   = 34,
        LM_GGML_TYPE_TQ2_0   = 35,
        // LM_GGML_TYPE_IQ4_NL_4_4 = 36,
        // LM_GGML_TYPE_IQ4_NL_4_8 = 37,
        // LM_GGML_TYPE_IQ4_NL_8_8 = 38,
        LM_GGML_TYPE_COUNT   = 39,
    };

    // precision
    enum lm_ggml_prec {
        LM_GGML_PREC_DEFAULT =  0, // stored as lm_ggml_tensor.op_params, 0 by default
        LM_GGML_PREC_F32     = 10,
    };

    // model file types
    enum lm_ggml_ftype {
        LM_GGML_FTYPE_UNKNOWN        = -1,
        LM_GGML_FTYPE_ALL_F32        = 0,
        LM_GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        LM_GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
    };

    // available tensor operations:
    enum lm_ggml_op {
        LM_GGML_OP_NONE = 0,

        LM_GGML_OP_DUP,
        LM_GGML_OP_ADD,
        LM_GGML_OP_ADD1,
        LM_GGML_OP_ACC,
        LM_GGML_OP_SUB,
        LM_GGML_OP_MUL,
        LM_GGML_OP_DIV,
        LM_GGML_OP_SQR,
        LM_GGML_OP_SQRT,
        LM_GGML_OP_LOG,
        LM_GGML_OP_SIN,
        LM_GGML_OP_COS,
        LM_GGML_OP_SUM,
        LM_GGML_OP_SUM_ROWS,
        LM_GGML_OP_MEAN,
        LM_GGML_OP_ARGMAX,
        LM_GGML_OP_COUNT_EQUAL,
        LM_GGML_OP_REPEAT,
        LM_GGML_OP_REPEAT_BACK,
        LM_GGML_OP_CONCAT,
        LM_GGML_OP_SILU_BACK,
        LM_GGML_OP_NORM, // normalize
        LM_GGML_OP_RMS_NORM,
        LM_GGML_OP_RMS_NORM_BACK,
        LM_GGML_OP_GROUP_NORM,
        LM_GGML_OP_L2_NORM,

        LM_GGML_OP_MUL_MAT,
        LM_GGML_OP_MUL_MAT_ID,
        LM_GGML_OP_OUT_PROD,

        LM_GGML_OP_SCALE,
        LM_GGML_OP_SET,
        LM_GGML_OP_CPY,
        LM_GGML_OP_CONT,
        LM_GGML_OP_RESHAPE,
        LM_GGML_OP_VIEW,
        LM_GGML_OP_PERMUTE,
        LM_GGML_OP_TRANSPOSE,
        LM_GGML_OP_GET_ROWS,
        LM_GGML_OP_GET_ROWS_BACK,
        LM_GGML_OP_DIAG,
        LM_GGML_OP_DIAG_MASK_INF,
        LM_GGML_OP_DIAG_MASK_ZERO,
        LM_GGML_OP_SOFT_MAX,
        LM_GGML_OP_SOFT_MAX_BACK,
        LM_GGML_OP_ROPE,
        LM_GGML_OP_ROPE_BACK,
        LM_GGML_OP_CLAMP,
        LM_GGML_OP_CONV_TRANSPOSE_1D,
        LM_GGML_OP_IM2COL,
        LM_GGML_OP_IM2COL_BACK,
        LM_GGML_OP_CONV_2D_DW,
        LM_GGML_OP_CONV_TRANSPOSE_2D,
        LM_GGML_OP_POOL_1D,
        LM_GGML_OP_POOL_2D,
        LM_GGML_OP_POOL_2D_BACK,
        LM_GGML_OP_UPSCALE, // nearest interpolate
        LM_GGML_OP_PAD,
        LM_GGML_OP_PAD_REFLECT_1D,
        LM_GGML_OP_ARANGE,
        LM_GGML_OP_TIMESTEP_EMBEDDING,
        LM_GGML_OP_ARGSORT,
        LM_GGML_OP_LEAKY_RELU,

        LM_GGML_OP_FLASH_ATTN_EXT,
        LM_GGML_OP_FLASH_ATTN_BACK,
        LM_GGML_OP_SSM_CONV,
        LM_GGML_OP_SSM_SCAN,
        LM_GGML_OP_WIN_PART,
        LM_GGML_OP_WIN_UNPART,
        LM_GGML_OP_GET_REL_POS,
        LM_GGML_OP_ADD_REL_POS,
        LM_GGML_OP_RWKV_WKV6,
        LM_GGML_OP_GATED_LINEAR_ATTN,
        LM_GGML_OP_RWKV_WKV7,

        LM_GGML_OP_UNARY,

        LM_GGML_OP_MAP_CUSTOM1,
        LM_GGML_OP_MAP_CUSTOM2,
        LM_GGML_OP_MAP_CUSTOM3,

        LM_GGML_OP_CUSTOM,

        LM_GGML_OP_CROSS_ENTROPY_LOSS,
        LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        LM_GGML_OP_OPT_STEP_ADAMW,

        LM_GGML_OP_COUNT,
    };

    enum lm_ggml_unary_op {
        LM_GGML_UNARY_OP_ABS,
        LM_GGML_UNARY_OP_SGN,
        LM_GGML_UNARY_OP_NEG,
        LM_GGML_UNARY_OP_STEP,
        LM_GGML_UNARY_OP_TANH,
        LM_GGML_UNARY_OP_ELU,
        LM_GGML_UNARY_OP_RELU,
        LM_GGML_UNARY_OP_SIGMOID,
        LM_GGML_UNARY_OP_GELU,
        LM_GGML_UNARY_OP_GELU_QUICK,
        LM_GGML_UNARY_OP_SILU,
        LM_GGML_UNARY_OP_HARDSWISH,
        LM_GGML_UNARY_OP_HARDSIGMOID,
        LM_GGML_UNARY_OP_EXP,
        LM_GGML_UNARY_OP_GELU_ERF,

        LM_GGML_UNARY_OP_COUNT,
    };

    enum lm_ggml_object_type {
        LM_GGML_OBJECT_TYPE_TENSOR,
        LM_GGML_OBJECT_TYPE_GRAPH,
        LM_GGML_OBJECT_TYPE_WORK_BUFFER
    };

    enum lm_ggml_log_level {
        LM_GGML_LOG_LEVEL_NONE  = 0,
        LM_GGML_LOG_LEVEL_DEBUG = 1,
        LM_GGML_LOG_LEVEL_INFO  = 2,
        LM_GGML_LOG_LEVEL_WARN  = 3,
        LM_GGML_LOG_LEVEL_ERROR = 4,
        LM_GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };

    // this tensor...
    enum lm_ggml_tensor_flag {
        LM_GGML_TENSOR_FLAG_INPUT  =  1, // ...is an input for the GGML compute graph
        LM_GGML_TENSOR_FLAG_OUTPUT =  2, // ...is an output for the GGML compute graph
        LM_GGML_TENSOR_FLAG_PARAM  =  4, // ...contains trainable parameters
        LM_GGML_TENSOR_FLAG_LOSS   =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
    };

    struct lm_ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    // n-dimensional tensor
    struct lm_ggml_tensor {
        enum lm_ggml_type type;

        struct lm_ggml_backend_buffer * buffer;

        int64_t ne[LM_GGML_MAX_DIMS]; // number of elements
        size_t  nb[LM_GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = lm_ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / lm_ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum lm_ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[LM_GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct lm_ggml_tensor * src[LM_GGML_MAX_SRC];

        // source tensor and offset for views
        struct lm_ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[LM_GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };

    static const size_t LM_GGML_TENSOR_SIZE = sizeof(struct lm_ggml_tensor);

    // Abort callback
    // If not NULL, called before ggml computation
    // If it returns true, the computation is aborted
    typedef bool (*lm_ggml_abort_callback)(void * data);


    //
    // GUID
    //

    // GUID types
    typedef uint8_t lm_ggml_guid[16];
    typedef lm_ggml_guid * lm_ggml_guid_t;

    LM_GGML_API bool lm_ggml_guid_matches(lm_ggml_guid_t guid_a, lm_ggml_guid_t guid_b);

    // misc

    LM_GGML_API void    lm_ggml_time_init(void); // call this once at the beginning of the program
    LM_GGML_API int64_t lm_ggml_time_ms(void);
    LM_GGML_API int64_t lm_ggml_time_us(void);
    LM_GGML_API int64_t lm_ggml_cycles(void);
    LM_GGML_API int64_t lm_ggml_cycles_per_ms(void);

    // accepts a UTF-8 path, even on Windows
    LM_GGML_API FILE *  lm_ggml_fopen(const char * fname, const char * mode);

    LM_GGML_API void    lm_ggml_print_object (const struct lm_ggml_object * obj);
    LM_GGML_API void    lm_ggml_print_objects(const struct lm_ggml_context * ctx);

    LM_GGML_API int64_t lm_ggml_nelements (const struct lm_ggml_tensor * tensor);
    LM_GGML_API int64_t lm_ggml_nrows     (const struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t  lm_ggml_nbytes    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t  lm_ggml_nbytes_pad(const struct lm_ggml_tensor * tensor); // same as lm_ggml_nbytes() but padded to LM_GGML_MEM_ALIGN

    LM_GGML_API int64_t lm_ggml_blck_size(enum lm_ggml_type type);
    LM_GGML_API size_t  lm_ggml_type_size(enum lm_ggml_type type);             // size in bytes for all elements in a block
    LM_GGML_API size_t  lm_ggml_row_size (enum lm_ggml_type type, int64_t ne); // size in bytes for all elements in a row

    LM_GGML_DEPRECATED(
    LM_GGML_API double lm_ggml_type_sizef(enum lm_ggml_type type), // lm_ggml_type_size()/lm_ggml_blck_size() as float
    "use lm_ggml_row_size() instead");

    LM_GGML_API const char * lm_ggml_type_name(enum lm_ggml_type type);
    LM_GGML_API const char * lm_ggml_op_name  (enum lm_ggml_op   op);
    LM_GGML_API const char * lm_ggml_op_symbol(enum lm_ggml_op   op);

    LM_GGML_API const char * lm_ggml_unary_op_name(enum lm_ggml_unary_op op);
    LM_GGML_API const char * lm_ggml_op_desc(const struct lm_ggml_tensor * t); // unary or op name

    LM_GGML_API size_t  lm_ggml_element_size(const struct lm_ggml_tensor * tensor);

    LM_GGML_API bool    lm_ggml_is_quantized(enum lm_ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    LM_GGML_API enum lm_ggml_type lm_ggml_ftype_to_lm_ggml_type(enum lm_ggml_ftype ftype);

    LM_GGML_API bool lm_ggml_is_transposed(const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_permuted  (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_empty     (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_scalar    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_vector    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_matrix    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_3d        (const struct lm_ggml_tensor * tensor);
    LM_GGML_API int  lm_ggml_n_dims       (const struct lm_ggml_tensor * tensor); // returns 1 for scalars

    // returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
    LM_GGML_API bool lm_ggml_is_contiguous  (const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_contiguous_0(const struct lm_ggml_tensor * tensor); // same as lm_ggml_is_contiguous()
    LM_GGML_API bool lm_ggml_is_contiguous_1(const struct lm_ggml_tensor * tensor); // contiguous for dims >= 1
    LM_GGML_API bool lm_ggml_is_contiguous_2(const struct lm_ggml_tensor * tensor); // contiguous for dims >= 2

    // returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
    LM_GGML_API bool lm_ggml_is_contiguously_allocated(const struct lm_ggml_tensor * tensor);

    // true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
    LM_GGML_API bool lm_ggml_is_contiguous_channels(const struct lm_ggml_tensor * tensor);

    LM_GGML_API bool lm_ggml_are_same_shape (const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1);
    LM_GGML_API bool lm_ggml_are_same_stride(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1);

    LM_GGML_API bool lm_ggml_can_repeat(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    LM_GGML_API size_t lm_ggml_tensor_overhead(void);

    LM_GGML_API bool lm_ggml_validate_row_data(enum lm_ggml_type type, const void * data, size_t nbytes);

    // main

    LM_GGML_API struct lm_ggml_context * lm_ggml_init (struct lm_ggml_init_params params);
    LM_GGML_API void                  lm_ggml_reset(struct lm_ggml_context * ctx);
    LM_GGML_API void                  lm_ggml_free (struct lm_ggml_context * ctx);

    LM_GGML_API size_t  lm_ggml_used_mem(const struct lm_ggml_context * ctx);

    LM_GGML_API bool    lm_ggml_get_no_alloc(struct lm_ggml_context * ctx);
    LM_GGML_API void    lm_ggml_set_no_alloc(struct lm_ggml_context * ctx, bool no_alloc);

    LM_GGML_API void *  lm_ggml_get_mem_buffer     (const struct lm_ggml_context * ctx);
    LM_GGML_API size_t  lm_ggml_get_mem_size       (const struct lm_ggml_context * ctx);
    LM_GGML_API size_t  lm_ggml_get_max_tensor_size(const struct lm_ggml_context * ctx);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int    n_dims,
            const int64_t *ne);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_1d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_2d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_3d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_4d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    LM_GGML_API void * lm_ggml_new_buffer(struct lm_ggml_context * ctx, size_t nbytes);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup_tensor (struct lm_ggml_context * ctx, const struct lm_ggml_tensor * src);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_tensor(struct lm_ggml_context * ctx, struct lm_ggml_tensor * src);

    // Context tensor enumeration and lookup
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_first_tensor(const struct lm_ggml_context * ctx);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_next_tensor (const struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_tensor(struct lm_ggml_context * ctx, const char * name);

    // Converts a flat index into coordinates
    LM_GGML_API void lm_ggml_unravel_index(const struct lm_ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);

    LM_GGML_API enum lm_ggml_unary_op lm_ggml_get_unary_op(const struct lm_ggml_tensor * tensor);

    LM_GGML_API void *  lm_ggml_get_data    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API float * lm_ggml_get_data_f32(const struct lm_ggml_tensor * tensor);

    LM_GGML_API const char *         lm_ggml_get_name   (const struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_name   (      struct lm_ggml_tensor * tensor, const char * name);
    LM_GGML_ATTRIBUTE_FORMAT(2, 3)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_format_name(      struct lm_ggml_tensor * tensor, const char * fmt, ...);

    // Tensor flags
    LM_GGML_API void lm_ggml_set_input(struct lm_ggml_tensor * tensor);
    LM_GGML_API void lm_ggml_set_output(struct lm_ggml_tensor * tensor);
    LM_GGML_API void lm_ggml_set_param(struct lm_ggml_tensor * tensor);
    LM_GGML_API void lm_ggml_set_loss(struct lm_ggml_tensor * tensor);

    //
    // operations on tensors with backpropagation
    //

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_cast(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            enum   lm_ggml_type      type);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add1(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add1_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // dst = a
    // view(dst, nb1, nb2, nb3, offset) += b
    // return dst
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_acc(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_acc_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sub(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sub_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_div(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_div_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqr(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqr_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqrt(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqrt_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_log(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_log_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sin(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sin_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cos(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cos_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // return scalar
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sum(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sum_rows(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // mean along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mean(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // argmax along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_argmax(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // count number of equal elements in a and b
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_count_equal(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_repeat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // sums repetitions in a into shape of b
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_repeat_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b); // sum up values that are adjacent in dims > 0 instead of repeated with same stride

    // concat a and b along dim
    // used in stable-diffusion
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_concat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   dim);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_abs(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_abs_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sgn(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sgn_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_neg(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_neg_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_step(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_step_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_tanh(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_tanh_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_elu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_elu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_relu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_leaky_relu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a, float negative_slope, bool inplace);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_relu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sigmoid(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sigmoid_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // GELU using erf (error function) when possible
    // some backends may fallback to approximation based on Abramowitz and Stegun formula
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_erf(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_erf_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_quick(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_quick_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // a - x
    // b - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // hardswish(x) = x * relu6(x + 3) / 6
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_hardswish(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // hardsigmoid(x) = relu6(x + 3) / 6
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_hardsigmoid(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_exp(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_exp_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // normalize along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    // group normalize along ne0*ne1*n_groups
    // used in stable-diffusion
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_group_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_groups,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_group_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_groups,
            float                 eps);

    // l2 normalize along rows
    // used in rwkv v7
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_l2_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_l2_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    // a - x
    // b - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            float                 eps);

    // A: k columns, n rows => [ne03, ne02, n, k]
    // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul_mat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // change the precision of a matrix multiplication
    // set to LM_GGML_PREC_F32 for higher precision (useful for phi-2)
    LM_GGML_API void lm_ggml_mul_mat_set_prec(
            struct lm_ggml_tensor * a,
            enum lm_ggml_prec       prec);

    // indirect matrix multiplication
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul_mat_id(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * as,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * ids);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_out_prod(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_scale(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 s);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_scale_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 s);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset); // in bytes

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset); // in bytes

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                offset); // in bytes

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_1d_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                offset); // in bytes

    // b -> view(a,offset,nb1,nb2,3), return modified a
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset); // in bytes

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_2d_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset); // in bytes

    // a -> b, return view(b)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cpy(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cast(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum   lm_ggml_type      type);

    // make contiguous
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // make contiguous, with new shape
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_permute(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for lm_ggml_permute(ctx, a, 1, 0, 2, 3)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_transpose(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // supports 3D: a->ne[2] == b->ne[1]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rows(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // data
            struct lm_ggml_tensor  * b); // row indices

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rows_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // gradients of lm_ggml_get_rows result
            struct lm_ggml_tensor  * b,  // row indices
            struct lm_ggml_tensor  * c); // data for lm_ggml_get_rows, only used for its shape

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag(
        struct lm_ggml_context     * ctx,
        struct lm_ggml_tensor      * a);

    // set elements above the diagonal to -INF
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_inf(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_inf_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_zero(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_zero_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // fused soft_max(a*scale + mask*(ALiBi slope))
    // mask is optional
    // max_bias = 0.0f for no ALiBi
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_ext(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * mask,
            float                 scale,
            float                 max_bias);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_ext_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            float                 scale,
            float                 max_bias);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_ext_back_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            float                 scale,
            float                 max_bias);

    // rotary position embedding
    // if (mode & 1) - skip n_past elements (NOT SUPPORTED)
    // if (mode & LM_GGML_ROPE_TYPE_NEOX) - GPT-NeoX style
    //
    // b is an int32 vector with size a->ne[2], it contains the positions
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode);

    // custom RoPE
    // c is freq factors (e.g. phi3-128k), (optional)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_ext(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * c,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_multi(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * c,
            int                   n_dims,
            int                   sections[4],
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_ext_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * c,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_custom(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow),
        "use lm_ggml_rope_ext instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_custom_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow),
        "use lm_ggml_rope_ext_inplace instead");

    // compute correction dims for YaRN RoPE scaling
    LM_GGML_API void lm_ggml_rope_yarn_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_ext_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a, // gradients of lm_ggml_rope result
            struct lm_ggml_tensor  * b, // positions
            struct lm_ggml_tensor  * c, // freq factors
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_multi_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * c,
            int                   n_dims,
            int                   sections[4],
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);


    // clamp
    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_clamp(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 min,
            float                 max);

    // im2col
    // converts data into a format that effectively results in a convolution when combined with matrix multiplication
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_im2col(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // convolution kernel
            struct lm_ggml_tensor  * b,  // data
            int                   s0, // stride dimension 0
            int                   s1, // stride dimension 1
            int                   p0, // padding dimension 0
            int                   p1, // padding dimension 1
            int                   d0, // dilation dimension 0
            int                   d1, // dilation dimension 1
            bool                  is_2D,
            enum lm_ggml_type        dst_type);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_im2col_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,  // convolution kernel
        struct lm_ggml_tensor  * b,  // gradient of im2col output
        int64_t             * ne, // shape of im2col input
        int                   s0, // stride dimension 0
        int                   s1, // stride dimension 1
        int                   p0, // padding dimension 0
        int                   p1, // padding dimension 1
        int                   d0, // dilation dimension 0
        int                   d1, // dilation dimension 1
        bool                  is_2D);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,   // convolution kernel
            struct lm_ggml_tensor  * b,   // data
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    // conv_1d with padding = half
    // alias for lm_ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    LM_GGML_API struct lm_ggml_tensor* lm_ggml_conv_1d_ph(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // convolution kernel
            struct lm_ggml_tensor  * b,  // data
            int                   s,  // stride
            int                   d); // dilation

    // depthwise
    // TODO: this is very likely wrong for some cases! - needs more testing
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_1d_dw(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,   // convolution kernel
            struct lm_ggml_tensor  * b,   // data
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_1d_dw_ph(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,   // convolution kernel
            struct lm_ggml_tensor  * b,   // data
            int                   s0,  // stride
            int                   d0); // dilation

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_transpose_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,   // convolution kernel
            struct lm_ggml_tensor  * b,   // data
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,   // convolution kernel
            struct lm_ggml_tensor  * b,   // data
            int                   s0,  // stride dimension 0
            int                   s1,  // stride dimension 1
            int                   p0,  // padding dimension 0
            int                   p1,  // padding dimension 1
            int                   d0,  // dilation dimension 0
            int                   d1); // dilation dimension 1

    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_sk_p0(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_s1_ph(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // depthwise (via im2col and mul_mat)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_dw(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // convolution kernel
            struct lm_ggml_tensor  * b,  // data
            int                  s0,  // stride dimension 0
            int                  s1,  // stride dimension 1
            int                  p0,  // padding dimension 0
            int                  p1,  // padding dimension 1
            int                  d0,  // dilation dimension 0
            int                  d1); // dilation dimension 1

    // Depthwise 2D convolution
    // may be faster than lm_ggml_conv_2d_dw, but not available in all backends
    // a:   KW    KH    1    C    convolution kernel
    // b:   W     H     C    N    input data
    // res: W_out H_out C    N
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_dw_direct(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   stride0,
            int                   stride1,
            int                   pad0,
            int                   pad1,
            int                   dilation0,
            int                   dilation1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_transpose_2d_p0(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   stride);

    enum lm_ggml_op_pool {
        LM_GGML_OP_POOL_MAX,
        LM_GGML_OP_POOL_AVG,
        LM_GGML_OP_POOL_COUNT,
    };

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pool_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum lm_ggml_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    // the result will have 2*p0 padding for the first dimension
    // and 2*p1 padding for the second dimension
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pool_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum lm_ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            float                 p0,
            float                 p1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pool_2d_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * af, // "a"/input used in forward pass
            enum lm_ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            float                 p0,
            float                 p1);

    enum lm_ggml_scale_mode {
        LM_GGML_SCALE_MODE_NEAREST  = 0,
        LM_GGML_SCALE_MODE_BILINEAR = 1,
    };

    // interpolate
    // multiplies ne0 and ne1 by scale factor
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_upscale(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   scale_factor,
            enum lm_ggml_scale_mode  mode);

    // interpolate
    // interpolate scale to specified dimensions
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_upscale_ext(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   ne0,
            int                   ne1,
            int                   ne2,
            int                   ne3,
            enum lm_ggml_scale_mode  mode);

    // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pad(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                  p0,
            int                  p1,
            int                  p2,
            int                  p3);

    // pad each dimension with reflection: [a, b, c, d] -> [b, a, b, c, d, c]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pad_reflect_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   p0,
            int                   p1);

    // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
    // timesteps: [N,]
    // return: [N, dim]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_timestep_embedding(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * timesteps,
            int                   dim,
            int                   max_period);

    // sort rows
    enum lm_ggml_sort_order {
        LM_GGML_SORT_ORDER_ASC,
        LM_GGML_SORT_ORDER_DESC,
    };

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_argsort(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum lm_ggml_sort_order  order);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_arange(
            struct lm_ggml_context * ctx,
            float                 start,
            float                 stop,
            float                 step);

    // top k elements per row
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_top_k(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   k);

#define LM_GGML_KQ_MASK_PAD 64

    // q:    [n_embd_k, n_batch,     n_head,    1]
    // k:    [n_embd_k, n_kv,        n_head_kv, 1]
    // v:    [n_embd_v, n_kv,        n_head_kv, 1] !! not transposed !!
    // mask: [n_kv,     n_batch_pad, 1,         1] !! n_batch_pad = LM_GGML_PAD(n_batch, LM_GGML_KQ_MASK_PAD) !!
    // res:  [n_embd_v, n_head,      n_batch,   1] !! permuted !!
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_flash_attn_ext(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * q,
            struct lm_ggml_tensor  * k,
            struct lm_ggml_tensor  * v,
            struct lm_ggml_tensor  * mask,
            float                 scale,
            float                 max_bias,
            float                 logit_softcap);

    LM_GGML_API void lm_ggml_flash_attn_ext_set_prec(
            struct lm_ggml_tensor * a,
            enum lm_ggml_prec       prec);

    LM_GGML_API enum lm_ggml_prec lm_ggml_flash_attn_ext_get_prec(
            const struct lm_ggml_tensor * a);

    // TODO: needs to be adapted to lm_ggml_flash_attn_ext
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_flash_attn_back(
           struct lm_ggml_context * ctx,
           struct lm_ggml_tensor  * q,
           struct lm_ggml_tensor  * k,
           struct lm_ggml_tensor  * v,
           struct lm_ggml_tensor  * d,
           bool                  masked);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_ssm_conv(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * sx,
            struct lm_ggml_tensor  * c);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_ssm_scan(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * s,
            struct lm_ggml_tensor  * x,
            struct lm_ggml_tensor  * dt,
            struct lm_ggml_tensor  * A,
            struct lm_ggml_tensor  * B,
            struct lm_ggml_tensor  * C);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_win_part(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   w);

    // reverse of lm_ggml_win_part
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_win_unpart(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_unary(
            struct lm_ggml_context * ctx,
             struct lm_ggml_tensor * a,
             enum lm_ggml_unary_op op);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_unary_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op op);

    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rel_pos(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   qh,
            int                   kh);

    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_rel_pos(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * pw,
            struct lm_ggml_tensor  * ph);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_rel_pos_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * pw,
            struct lm_ggml_tensor  * ph);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rwkv_wkv6(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * k,
            struct lm_ggml_tensor  * v,
            struct lm_ggml_tensor  * r,
            struct lm_ggml_tensor  * tf,
            struct lm_ggml_tensor  * td,
            struct lm_ggml_tensor  * state);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gated_linear_attn(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * k,
            struct lm_ggml_tensor  * v,
            struct lm_ggml_tensor  * q,
            struct lm_ggml_tensor  * g,
            struct lm_ggml_tensor  * state,
            float scale);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rwkv_wkv7(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * r,
            struct lm_ggml_tensor  * w,
            struct lm_ggml_tensor  * k,
            struct lm_ggml_tensor  * v,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * state);

    // custom operators

    typedef void (*lm_ggml_custom1_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*lm_ggml_custom2_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, const struct lm_ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*lm_ggml_custom3_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, const struct lm_ggml_tensor * b, const struct lm_ggml_tensor * c, int ith, int nth, void * userdata);

#define LM_GGML_N_TASKS_MAX (-1)
    // n_tasks == LM_GGML_N_TASKS_MAX means to use max number of tasks

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            lm_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            lm_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            lm_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            lm_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            struct lm_ggml_tensor    * c,
            lm_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            struct lm_ggml_tensor    * c,
            lm_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    typedef void (*lm_ggml_custom_op_t)(struct lm_ggml_tensor * dst , int ith, int nth, void * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_custom_4d(
            struct lm_ggml_context * ctx,
            enum lm_ggml_type        type,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            struct lm_ggml_tensor ** args,
            int                   n_args,
            lm_ggml_custom_op_t      fun,
            int                   n_tasks,
            void                * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_custom_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor ** args,
            int                   n_args,
            lm_ggml_custom_op_t      fun,
            int                   n_tasks,
            void                * userdata);

    // loss function

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cross_entropy_loss(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // logits
            struct lm_ggml_tensor  * b); // labels

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cross_entropy_loss_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,  // logits
            struct lm_ggml_tensor  * b,  // labels
            struct lm_ggml_tensor  * c); // gradients of cross_entropy_loss result

    // AdamW optimizer step
    // Paper: https://arxiv.org/pdf/1711.05101v3.pdf
    // PyTorch: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_opt_step_adamw(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * grad,
            struct lm_ggml_tensor  * m,
            struct lm_ggml_tensor  * v,
            struct lm_ggml_tensor  * adamw_params); // parameters such a the learning rate

    //
    // automatic differentiation
    //

    LM_GGML_API void lm_ggml_build_forward_expand(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor);
    LM_GGML_API void lm_ggml_build_backward_expand(
        struct lm_ggml_context *  ctx,        // context for gradient computation
        struct lm_ggml_cgraph  *  cgraph,
        struct lm_ggml_tensor  ** grad_accs);

    // graph allocation in a context
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_new_graph       (struct lm_ggml_context * ctx); // size = LM_GGML_DEFAULT_GRAPH_SIZE, grads = false
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_new_graph_custom(struct lm_ggml_context * ctx, size_t size, bool grads);
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_graph_dup       (struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, bool force_grads);
    LM_GGML_API void                 lm_ggml_graph_cpy       (struct lm_ggml_cgraph * src, struct lm_ggml_cgraph * dst);
    LM_GGML_API void                 lm_ggml_graph_reset     (struct lm_ggml_cgraph * cgraph); // set regular grads + optimizer momenta to 0, set loss grad to 1
    LM_GGML_API void                 lm_ggml_graph_clear     (struct lm_ggml_cgraph * cgraph);

    LM_GGML_API int                   lm_ggml_graph_size   (struct lm_ggml_cgraph * cgraph);
    LM_GGML_API struct lm_ggml_tensor *  lm_ggml_graph_node   (struct lm_ggml_cgraph * cgraph, int i); // if i < 0, returns nodes[n_nodes + i]
    LM_GGML_API struct lm_ggml_tensor ** lm_ggml_graph_nodes  (struct lm_ggml_cgraph * cgraph);
    LM_GGML_API int                   lm_ggml_graph_n_nodes(struct lm_ggml_cgraph * cgraph);

    LM_GGML_API void   lm_ggml_graph_add_node(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor);

    LM_GGML_API size_t lm_ggml_graph_overhead(void);
    LM_GGML_API size_t lm_ggml_graph_overhead_custom(size_t size, bool grads);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_graph_get_tensor  (const struct lm_ggml_cgraph * cgraph, const char * name);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_graph_get_grad    (const struct lm_ggml_cgraph * cgraph, const struct lm_ggml_tensor * node);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_graph_get_grad_acc(const struct lm_ggml_cgraph * cgraph, const struct lm_ggml_tensor * node);

    LM_GGML_API void                 lm_ggml_graph_export(const struct lm_ggml_cgraph * cgraph, const char * fname);
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_graph_import(const char * fname, struct lm_ggml_context ** ctx_data, struct lm_ggml_context ** ctx_eval);

    // print info and performance information for the graph
    LM_GGML_API void lm_ggml_graph_print(const struct lm_ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    LM_GGML_API void lm_ggml_graph_dump_dot(const struct lm_ggml_cgraph * gb, const struct lm_ggml_cgraph * gf, const char * filename);

    // TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
    typedef void (*lm_ggml_log_callback)(enum lm_ggml_log_level level, const char * text, void * user_data);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LM_GGML_API void lm_ggml_log_set(lm_ggml_log_callback log_callback, void * user_data);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_zero(struct lm_ggml_tensor * tensor);

    //
    // quantization
    //

    // - lm_ggml_quantize_init can be called multiple times with the same type
    //   it will only initialize the quantization tables for the first call or after lm_ggml_quantize_free
    //   automatically called by lm_ggml_quantize_chunk for convenience
    //
    // - lm_ggml_quantize_free will free any memory allocated by lm_ggml_quantize_init
    //   call this at the end of the program to avoid memory leaks
    //
    // note: these are thread-safe
    //
    LM_GGML_API void lm_ggml_quantize_init(enum lm_ggml_type type);
    LM_GGML_API void lm_ggml_quantize_free(void);

    // some quantization type cannot be used without an importance matrix
    LM_GGML_API bool lm_ggml_quantize_requires_imatrix(enum lm_ggml_type type);

    // calls lm_ggml_quantize_init internally (i.e. can allocate memory)
    LM_GGML_API size_t lm_ggml_quantize_chunk(
            enum lm_ggml_type   type,
               const float * src,
                      void * dst,
                   int64_t   start,
                   int64_t   nrows,
                   int64_t   n_per_row,
               const float * imatrix);

#ifdef __cplusplus
    // restrict not standard in C++
#    if defined(__GNUC__)
#        define LM_GGML_RESTRICT __restrict__
#    elif defined(__clang__)
#        define LM_GGML_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define LM_GGML_RESTRICT __restrict
#    else
#        define LM_GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define LM_GGML_RESTRICT __restrict
#    else
#        define LM_GGML_RESTRICT restrict
#    endif
#endif
    typedef void (*lm_ggml_to_float_t)  (const void  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
    typedef void (*lm_ggml_from_float_t)(const float * LM_GGML_RESTRICT x, void  * LM_GGML_RESTRICT y, int64_t k);

    struct lm_ggml_type_traits {
        const char             * type_name;
        int64_t                  blck_size;
        int64_t                  blck_size_interleave; // interleave elements in blocks
        size_t                   type_size;
        bool                     is_quantized;
        lm_ggml_to_float_t          to_float;
        lm_ggml_from_float_t        from_float_ref;
    };

    LM_GGML_API const struct lm_ggml_type_traits * lm_ggml_get_type_traits(enum lm_ggml_type type);

    // ggml threadpool
    // TODO: currently, only a few functions are in the base ggml API, while the rest are in the CPU backend
    // the goal should be to create an API that other backends can use move everything to the ggml base

    // scheduling priorities
    enum lm_ggml_sched_priority {
        LM_GGML_SCHED_PRIO_NORMAL,
        LM_GGML_SCHED_PRIO_MEDIUM,
        LM_GGML_SCHED_PRIO_HIGH,
        LM_GGML_SCHED_PRIO_REALTIME
    };

    // threadpool params
    // Use lm_ggml_threadpool_params_default() or lm_ggml_threadpool_params_init() to populate the defaults
    struct lm_ggml_threadpool_params {
        bool                cpumask[LM_GGML_MAX_N_THREADS]; // mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads;                   // number of threads
        enum lm_ggml_sched_priority prio;                   // thread priority
        uint32_t            poll;                        // polling level (0 - no polling, 100 - aggressive polling)
        bool                strict_cpu;                  // strict cpu placement
        bool                paused;                      // start in paused state
    };

    struct lm_ggml_threadpool;     // forward declaration, see ggml.c

    typedef struct lm_ggml_threadpool * lm_ggml_threadpool_t;

    LM_GGML_API struct lm_ggml_threadpool_params lm_ggml_threadpool_params_default(int n_threads);
    LM_GGML_API void                          lm_ggml_threadpool_params_init   (struct lm_ggml_threadpool_params * p, int n_threads);
    LM_GGML_API bool                          lm_ggml_threadpool_params_match  (const struct lm_ggml_threadpool_params * p0, const struct lm_ggml_threadpool_params * p1);

#ifdef  __cplusplus
}
#endif
