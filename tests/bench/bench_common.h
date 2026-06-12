#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../cactus-kernels/cactus_kernels.h"

namespace bench {

constexpr size_t kGroupSize = 32;
constexpr size_t kBlockSize = 4;

inline const std::vector<size_t>& default_dim_sweep() {
    static const std::vector<size_t> dims = {128, 256, 512, 1024, 2048, 4096};
    return dims;
}

enum class MatmulGraph {
    GEMV_D,    // (1 x d) @ (d x N), with N=d swept
    GEMM_D,    // (M x d) @ (d x N), M=N=512, d swept
    GEMM_MN,   // (M x d) @ (d x N), M=N swept, d=512
};

struct MatmulConfig {
    MatmulGraph graph;
    size_t M;
    size_t K;
    size_t N;
    size_t sweep_dim;
};

struct MatmulBenchOptions {
    int warmup = 50;
    int iterations = 200;
    int num_threads = 0;
    std::vector<MatmulGraph> graphs = {MatmulGraph::GEMV_D, MatmulGraph::GEMM_D, MatmulGraph::GEMM_MN};
    std::vector<size_t> dims;
    std::string backends_filter;
    std::string csv_path;
};

enum class AttnMode { PREFILL, DECODE };

enum class AttnGraph {
    PREFILL_S,
    DECODE_CACHE,
};

struct AttnDims {
    size_t head_dim = 128;
    size_t num_q_heads = 8;
    size_t num_kv_heads = 8;
};

struct AttnConfig {
    AttnGraph graph;
    AttnMode mode;
    AttnDims dims;
    size_t seq_len;
    size_t cache_len;
    size_t sweep_dim;
};

struct AttnBenchOptions {
    int warmup = 50;
    int iterations = 200;
    int num_threads = 0;
    std::vector<AttnGraph> graphs = {AttnGraph::PREFILL_S, AttnGraph::DECODE_CACHE};
    std::vector<size_t> dims;
    size_t model_dim = 1024;
    size_t num_heads = 8;
    size_t num_kv_heads = 0;  // 0 = use num_heads (MHA); >0 enables GQA/MQA
    size_t head_dim_override = 0;  // 0 = derive from model_dim/num_heads
    std::string backends_filter;
    std::string csv_path;
};

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double compute_matmul_gops(size_t M, size_t K, size_t N, int iters, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    return (2.0 * M * K * N * iters) / (total_ms * 1e6);
}

inline double compute_attention_gflops(size_t num_q_heads, size_t seq_len,
                                        size_t kv_seq_len, size_t head_dim,
                                        int iterations, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    double flops_per_call = static_cast<double>(num_q_heads) *
        (4.0 * seq_len * kv_seq_len * head_dim + 5.0 * seq_len * kv_seq_len);
    return (flops_per_call * iterations) / (total_ms * 1e6);
}

void quantize_int8_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales);

std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K);
std::vector<__fp16> interleave_scales_n4(const std::vector<float>& scales, size_t N, size_t num_groups);

struct CactusActivations {
    std::vector<int8_t> int8;
    std::vector<float> scales;
    std::vector<__fp16> fp16;
    std::vector<float> fp32;
};

CactusActivations prepare_cactus_activations(size_t M, size_t K, std::mt19937& gen);

void reference_matmul_fp32(const float* A, const float* B_rowmajor_NK,
                            float* C, size_t M, size_t K, size_t N);

void reference_attention_fp32(const float* Q, const float* K, const float* V,
                               float* output,
                               size_t num_q_heads, size_t num_kv_heads,
                               size_t seq_len, size_t kv_seq_len,
                               size_t head_dim, float scale,
                               size_t window_size = 0);

struct AccuracyResult {
    float max_abs_error = 0.0f;
    float nrmse = 0.0f;
    bool passed = false;
};

AccuracyResult check_accuracy(const float* reference, const float* actual,
                               size_t count, float nrmse_tolerance);

bool framework_matches_filter(const char* framework, const std::string& filter);

void fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void fp16_to_fp32(const __fp16* src, float* dst, size_t count);

void quantize_rows_int8(const float* src, int8_t* dst, float* scales,
                         size_t rows, size_t cols);

void transpose_2d(const float* src, float* dst, size_t rows, size_t cols);

void set_thread_override(int n);
int  get_thread_override();
int  get_effective_threads(int backend_default);

bool parse_matmul_bench_args(int argc, char** argv, MatmulBenchOptions& opt, std::string& err);
bool parse_attn_bench_args(int argc, char** argv, AttnBenchOptions& opt, std::string& err);

const char* matmul_graph_name(MatmulGraph g);
const char* attn_graph_name(AttnGraph g);

std::vector<MatmulConfig> build_matmul_configs(const MatmulBenchOptions& opt);
std::vector<AttnConfig>   build_attn_configs(const AttnBenchOptions& opt);

} // namespace bench

#endif
