#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

#include "bench_common.h"

#include <string>
#include <vector>

namespace bench {

struct MatmulBackendVariant {
    const char* name;
    const char* framework;

    void* (*prepare_weights)(const float* fp32, size_t N, size_t K);

    // May be null. Backends that quantize activations once per call set it.
    void* (*prepare_activations)(const float* fp32, size_t M, size_t K, void* weights);

    // If `output` is non-null, fill it with FP32 output for accuracy. If
    // `reference` is non-null, the backend may fill it with its own internal
    // ground truth (used by backends whose precision differs from the FP32
    // reference enough to need a tighter, kernel-internal tolerance).
    void (*run_kernel)(size_t M, size_t K, size_t N,
                       void* weights, void* activations,
                       const int8_t* act_int8, const float* act_scales,
                       float* output, float* reference);

    void (*cleanup)(void* weights, void* activations);
};

void register_matmul_backend(MatmulBackendVariant v);
const std::vector<MatmulBackendVariant>& get_matmul_backends();

struct AttnBackendVariant {
    const char* name;
    const char* framework;
    AttnMode mode;

    void* (*prepare)(const AttnDims& dims, size_t seq_len, size_t cache_len,
                     const float* fp32_q, const float* fp32_k, const float* fp32_v);
    void  (*run)(void* state, float* output);
    void  (*cleanup)(void* state);
};

void register_attn_backend(AttnBackendVariant v);
const std::vector<AttnBackendVariant>& get_attn_backends();

bool run_matmul_benchmark(const MatmulBenchOptions& opt);
bool run_attn_benchmark(const AttnBenchOptions& opt);

} // namespace bench

#endif
