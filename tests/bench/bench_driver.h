#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

#include "bench_common.h"

#include <string>
#include <vector>

namespace bench {

// ── Matmul backend interface ────────────────────────────────────────────────

struct MatmulBackendVariant {
    const char* name;
    const char* framework;

    // Prepare an opaque weights handle for the (N,K) shape from raw fp32 weights.
    void* (*prepare_weights)(const float* fp32, size_t N, size_t K);

    // Optional per-backend activation prep (some backends quantize once per call).
    void* (*prepare_activations)(const float* fp32, size_t M, size_t K, void* weights);

    // Run the kernel. If `output` is non-null, fill it with FP32 output for accuracy.
    // If `reference` is non-null, the backend may fill it with its own internal
    // ground-truth (some backends compare against their own reference).
    void (*run_kernel)(size_t M, size_t K, size_t N,
                       void* weights, void* activations,
                       const int8_t* act_int8, const float* act_scales,
                       float* output, float* reference);

    void (*cleanup)(void* weights, void* activations);
};

void register_matmul_backend(MatmulBackendVariant v);
const std::vector<MatmulBackendVariant>& get_matmul_backends();

// ── Attention backend interface ─────────────────────────────────────────────

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

// ── Drivers ─────────────────────────────────────────────────────────────────

bool run_matmul_benchmark(const MatmulBenchOptions& opt);
bool run_attn_benchmark(const AttnBenchOptions& opt);

} // namespace bench

#endif
