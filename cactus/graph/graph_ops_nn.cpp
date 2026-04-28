#include "graph.h"
#include "../kernel/kernel.h"
#include "../kernel/kernel_utils.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstdio>

namespace {
    thread_local std::vector<__fp16> transpose_buffer_fp16;
    thread_local std::vector<int8_t> quant_activation_buffer;
    thread_local std::vector<float> quant_scales_buffer;

    thread_local const __fp16* cached_quant_src = nullptr;
    thread_local size_t cached_quant_M = 0;
    thread_local size_t cached_quant_K = 0;

    void ensure_transpose_buffer_fp16(size_t required_size) {
        if (transpose_buffer_fp16.size() < required_size) {
            transpose_buffer_fp16.resize(required_size);
        }
    }

    void ensure_quant_buffers(size_t M, size_t K) {
        size_t required_data = M * K;
        if (quant_activation_buffer.size() < required_data) {
            quant_activation_buffer.resize(required_data);
        }
        if (quant_scales_buffer.size() < M) {
            quant_scales_buffer.resize(M);
        }
    }

    void quantize_activations_fp16_to_int8(const __fp16* src, int8_t* dst, float* scales, size_t M, size_t K) {
        if (src == cached_quant_src && M == cached_quant_M && K == cached_quant_K) {
            return;
        }

        constexpr size_t PARALLEL_THRESHOLD = 16;

        if (M >= PARALLEL_THRESHOLD) {
            CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
                [src, dst, scales, K](size_t m_start, size_t m_end) {
                    for (size_t m = m_start; m < m_end; m++) {
                        float max_abs = cactus_fp16_max_abs(src + m * K, K);
                        float scale = max_abs / 127.0f;
                        if (scale < 1e-10f) scale = 1e-10f;
                        scales[m] = scale;
                        cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
                    }
                });
        } else {
            for (size_t m = 0; m < M; m++) {
                float max_abs = cactus_fp16_max_abs(src + m * K, K);
                float scale = max_abs / 127.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                scales[m] = scale;
                cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
            }
        }

        cached_quant_src = src;
        cached_quant_M = M;
        cached_quant_K = K;
    }

    const __fp16* as_fp16_ptr(const BufferDesc& buffer, std::vector<__fp16>& scratch) {
        if (buffer.precision == Precision::FP16) {
            return buffer.data_as<__fp16>();
        }
        if (buffer.precision == Precision::FP32) {
            scratch.resize(buffer.total_size);
            cactus_fp32_to_fp16(buffer.data_as<float>(), scratch.data(), buffer.total_size);
            return scratch.data();
        }
        throw std::runtime_error("GATED_DELTANET unsupported precision (expected FP16/FP32)");
    }

    void validate_gated_deltanet_inputs(
        const BufferDesc& q,
        const BufferDesc& k,
        const BufferDesc& v,
        const BufferDesc& g,
        const BufferDesc& b,
        const BufferDesc& s) {
        auto is_supported_precision = [](Precision p) {
            return p == Precision::FP16 || p == Precision::FP32;
        };
        if (!is_supported_precision(q.precision) || !is_supported_precision(k.precision) ||
            !is_supported_precision(v.precision) || !is_supported_precision(g.precision) ||
            !is_supported_precision(b.precision) || !is_supported_precision(s.precision)) {
            throw std::runtime_error("GATED_DELTANET requires FP16/FP32 inputs");
        }

        if (q.shape.size() != 4 || k.shape.size() != 4 || v.shape.size() != 4) {
            throw std::runtime_error("GATED_DELTANET expects query/key/value rank 4 [B, T, H, D]");
        }
        if (g.shape.size() != 3 || b.shape.size() != 3) {
            throw std::runtime_error("GATED_DELTANET expects gate_log/beta rank 3 [B, T, H]");
        }
        if (s.shape.size() != 4) {
            throw std::runtime_error("GATED_DELTANET expects state rank 4 [B, K, H, V]");
        }

        const size_t B = q.shape[0];
        const size_t T = q.shape[1];
        const size_t Hq = q.shape[2];
        const size_t K = q.shape[3];

        if (k.shape[0] != B || k.shape[1] != T || k.shape[2] != Hq || k.shape[3] != K) {
            throw std::runtime_error("GATED_DELTANET query/key shape mismatch");
        }
        if (v.shape[0] != B || v.shape[1] != T) {
            throw std::runtime_error("GATED_DELTANET value shape mismatch");
        }
        const size_t Hv = v.shape[2];
        if (g.shape[0] != B || g.shape[1] != T || g.shape[2] != Hv ||
            b.shape[0] != B || b.shape[1] != T || b.shape[2] != Hv) {
            throw std::runtime_error("GATED_DELTANET gate_log/beta shape mismatch");
        }
        if (Hq == 0 || Hv == 0 || (Hv % Hq) != 0) {
            throw std::runtime_error("GATED_DELTANET expects value heads divisible by q/k heads");
        }
        const size_t V = v.shape[3];
        if (s.shape[0] != B || s.shape[1] != K || s.shape[2] != Hv || s.shape[3] != V) {
            throw std::runtime_error("GATED_DELTANET state shape mismatch");
        }
    }


}

void shrink_thread_local_buffers() {
    std::vector<__fp16>().swap(transpose_buffer_fp16);
    std::vector<int8_t>().swap(quant_activation_buffer);
    std::vector<float>().swap(quant_scales_buffer);
    cached_quant_src = nullptr;
    cached_quant_M = 0;
    cached_quant_K = 0;
}

void compute_quantize_activations_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& shape = input_buffer.shape;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("QUANTIZE_ACTIVATIONS requires FP16 input");
    }

    if (shape.size() < 2) {
        throw std::runtime_error("QUANTIZE_ACTIVATIONS requires at least 2D tensor");
    }

    size_t K = shape.back();
    size_t M = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        M *= shape[i];
    }

    if (!node.output_buffer.has_activation_scales() ||
        node.output_buffer.num_rows_for_activation_scales != M) {
        node.output_buffer.allocate_activation_scales(M);
    }

    const __fp16* src = input_buffer.data_as<__fp16>();
    int8_t* dst = node.output_buffer.data_as<int8_t>();
    float* scales = node.output_buffer.activation_scales_as_float();

    constexpr size_t PARALLEL_THRESHOLD = 16;

    if (M >= PARALLEL_THRESHOLD) {
        CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
            [src, dst, scales, K](size_t m_start, size_t m_end) {
                for (size_t m = m_start; m < m_end; m++) {
                    float max_abs = cactus_fp16_max_abs(src + m * K, K);
                    float scale = max_abs / 127.0f;
                    if (scale < 1e-10f) scale = 1e-10f;
                    scales[m] = scale;
                    cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
                }
            });
    } else {
        for (size_t m = 0; m < M; m++) {
            float max_abs = cactus_fp16_max_abs(src + m * K, K);
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[m] = scale;
            cactus_fp16_to_int8(src + m * K, dst + m * K, K, scale);
        }
    }
}

void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& rhs_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& lhs_shape = lhs_buffer.shape;
    const auto& rhs_shape = rhs_buffer.shape;

    size_t M = lhs_shape[lhs_shape.size() - 2];
    size_t K = lhs_shape[lhs_shape.size() - 1];
    size_t N;
    if (rhs_buffer.is_interleaved && rhs_buffer.original_N > 0) {
        N = rhs_buffer.original_N;
    } else {
        N = node.params.pretransposed_rhs ?
            rhs_shape[rhs_shape.size() - 2] : rhs_shape[rhs_shape.size() - 1];
    }

    bool pretransposed_rhs = node.params.pretransposed_rhs;

    ComputeBackend backend = node.params.backend;

    if (backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU matrix multiplication not yet implemented");
    }

    const bool lhs_is_prequantized_int8 = (lhs_buffer.precision == Precision::INT8 &&
                                            lhs_buffer.has_activation_scales());

    if (PrecisionTraits::is_integer(rhs_buffer.precision) && rhs_buffer.group_size > 0) {
        const int8_t* rhs = rhs_buffer.data_as<int8_t>();
        const __fp16* rhs_scales = rhs_buffer.scales_as_fp16();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (!pretransposed_rhs) {
            throw std::runtime_error("Group-wise quantized matmul requires pretransposed weights");
        }

        const int8_t* lhs_int8;
        const float* lhs_scales;

        if (lhs_is_prequantized_int8) {
            lhs_int8 = lhs_buffer.data_as<int8_t>();
            lhs_scales = lhs_buffer.activation_scales_as_float();
        } else if (lhs_buffer.precision == Precision::FP16) {
            ensure_quant_buffers(M, K);
            cached_quant_src = nullptr;
            quantize_activations_fp16_to_int8(lhs_buffer.data_as<__fp16>(), quant_activation_buffer.data(),
                                              quant_scales_buffer.data(), M, K);
            lhs_int8 = quant_activation_buffer.data();
            lhs_scales = quant_scales_buffer.data();
        } else {
            throw std::runtime_error("Quantized matmul requires INT8 (pre-quantized) or FP16 activations");
        }

        cactus_matmul_integer(rhs_buffer.precision,
                        lhs_int8, lhs_scales,
                        rhs, rhs_scales, output,
                        M, K, N, rhs_buffer.group_size);
    } else {
        if (lhs_buffer.precision != Precision::FP16) {
            throw std::runtime_error("FP16 matmul requires FP16 activations (got precision " + std::to_string(static_cast<int>(lhs_buffer.precision)) + ")");
        }

        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        const __fp16* rhs = rhs_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (pretransposed_rhs) {
            cactus_matmul_f16(lhs, rhs, output, M, K, N);
        } else {
            size_t transpose_size = rhs_shape[0] * rhs_shape[1];
            ensure_transpose_buffer_fp16(transpose_size);

            cactus_transpose_2d_f16(rhs, transpose_buffer_fp16.data(),
                                    rhs_shape[0], rhs_shape[1], 0, rhs_shape[0]);
            cactus_matmul_f16(lhs, transpose_buffer_fp16.data(), output, M, K, N);
        }
    }
}

// ─── Grouped MLP K=96 (gemma4-e2b-grouped-k96) ───────────────────────────────
namespace {
    // Per-token scratch reused across decode calls. Sized for the larger
    // Gemma-4 wide layer (D_FFN=12288).
    thread_local std::vector<__fp16> gmlp_gate_batched; // M * d_ffn (prefill batched gate proj)
    thread_local std::vector<__fp16> gmlp_up_partial;   // d_ffn
    thread_local std::vector<__fp16> gmlp_h_full;       // d_ffn
    thread_local std::vector<int8_t> gmlp_h_int8;       // d_ffn
    thread_local std::vector<int8_t> gmlp_x_int8_batched; // M * hidden_dim
    thread_local std::vector<float>  gmlp_x_scales_batched; // M
    thread_local std::vector<float>  gmlp_group_max;    // k_groups
    thread_local std::vector<uint16_t> gmlp_block_runs;  // 2 * k_active pairs
    thread_local std::vector<uint16_t> gmlp_kgroup_runs; // 2 * k_active pairs
    thread_local std::vector<uint16_t> gmlp_active_groups; // k_active

    inline void ensure_gmlp_buffers(size_t M, size_t hidden_dim, size_t d_ffn,
                                     size_t k_groups, size_t k_active) {
        if (gmlp_gate_batched.size()    < M * d_ffn)      gmlp_gate_batched.resize(M * d_ffn);
        if (gmlp_up_partial.size()      < d_ffn)          gmlp_up_partial.resize(d_ffn);
        if (gmlp_h_full.size()          < d_ffn)          gmlp_h_full.resize(d_ffn);
        if (gmlp_h_int8.size()          < d_ffn)          gmlp_h_int8.resize(d_ffn);
        if (gmlp_x_int8_batched.size()  < M * hidden_dim) gmlp_x_int8_batched.resize(M * hidden_dim);
        if (gmlp_x_scales_batched.size()< M)              gmlp_x_scales_batched.resize(M);
        if (gmlp_group_max.size()       < k_groups)       gmlp_group_max.resize(k_groups);
        if (gmlp_active_groups.size()   < k_active)       gmlp_active_groups.resize(k_active);
        if (gmlp_block_runs.capacity()  < 2 * k_active)   gmlp_block_runs.reserve(2 * k_active);
        if (gmlp_kgroup_runs.capacity() < 2 * k_active)   gmlp_kgroup_runs.reserve(2 * k_active);
    }

    // ─── Optional profiling for the grouped MLP ──────────────────────────────
    // Set K96_PROF=1 to dump per-step accumulated nanoseconds to stderr at the
    // end of the run. Designed to localize the bottleneck inside the per-token
    // grouped-MLP loop on decode (M=1).
    struct GmlpProf {
        bool enabled = false;
        bool dump_registered = false;
        uint64_t ns_quantize = 0;
        uint64_t ns_gate     = 0;
        uint64_t ns_gelu_max = 0;
        uint64_t ns_topk     = 0;
        uint64_t ns_runs     = 0;
        uint64_t ns_up       = 0;
        uint64_t ns_h        = 0;
        uint64_t ns_quant_h  = 0;
        uint64_t ns_down     = 0;
        uint64_t ns_reduce   = 0;
        uint64_t ns_dispatch = 0; // sum across workers of per-cluster work (worker-side wall-time aggregate)
        std::atomic<uint64_t> ns_pcl_up{0};
        std::atomic<uint64_t> ns_pcl_h{0};
        std::atomic<uint64_t> ns_pcl_quant{0};
        std::atomic<uint64_t> ns_pcl_down{0};
        std::atomic<uint64_t> ns_pcl_acc{0};
        uint64_t calls       = 0;
        void check_env() {
            const char* e = getenv("K96_PROF");
            enabled = (e && e[0] && e[0] != '0');
        }
        void dump() const {
            if (!enabled || calls == 0) return;
            uint64_t total = ns_quantize + ns_gate + ns_gelu_max + ns_topk +
                             ns_runs + ns_up + ns_h + ns_quant_h + ns_down + ns_reduce;
            auto pct = [&](uint64_t v) { return total ? (100.0 * double(v) / double(total)) : 0.0; };
            uint64_t pup    = ns_pcl_up.load();
            uint64_t ph     = ns_pcl_h.load();
            uint64_t pq     = ns_pcl_quant.load();
            uint64_t pdn    = ns_pcl_down.load();
            uint64_t pacc   = ns_pcl_acc.load();
            uint64_t ptot   = pup + ph + pq + pdn + pacc;
            auto ppct = [&](uint64_t v) { return ptot ? (100.0 * double(v) / double(ptot)) : 0.0; };
            fprintf(stderr,
                "[K96_PROF] calls=%llu total=%.3f ms\n"
                "  quant_x      %8.3f ms (%5.2f%%)\n"
                "  gate_proj    %8.3f ms (%5.2f%%)\n"
                "  gelu+max     %8.3f ms (%5.2f%%)\n"
                "  topk         %8.3f ms (%5.2f%%)\n"
                "  build_runs   %8.3f ms (%5.2f%%)\n"
                "  up_proj      %8.3f ms (%5.2f%%)\n"
                "  h_compute    %8.3f ms (%5.2f%%)\n"
                "  quant_h      %8.3f ms (%5.2f%%)\n"
                "  down_proj    %8.3f ms (%5.2f%%)\n"
                "  reduce       %8.3f ms (%5.2f%%)\n"
                "  -- per-cluster (sum across workers) --\n"
                "  pcl_up       %8.3f ms (%5.2f%%)\n"
                "  pcl_h        %8.3f ms (%5.2f%%)\n"
                "  pcl_quant    %8.3f ms (%5.2f%%)\n"
                "  pcl_down     %8.3f ms (%5.2f%%)\n"
                "  pcl_acc      %8.3f ms (%5.2f%%)\n",
                (unsigned long long)calls,
                total / 1e6,
                ns_quantize / 1e6, pct(ns_quantize),
                ns_gate     / 1e6, pct(ns_gate),
                ns_gelu_max / 1e6, pct(ns_gelu_max),
                ns_topk     / 1e6, pct(ns_topk),
                ns_runs     / 1e6, pct(ns_runs),
                ns_up       / 1e6, pct(ns_up),
                ns_h        / 1e6, pct(ns_h),
                ns_quant_h  / 1e6, pct(ns_quant_h),
                ns_down     / 1e6, pct(ns_down),
                ns_reduce   / 1e6, pct(ns_reduce),
                pup  / 1e6, ppct(pup),
                ph   / 1e6, ppct(ph),
                pq   / 1e6, ppct(pq),
                pdn  / 1e6, ppct(pdn),
                pacc / 1e6, ppct(pacc));
            fflush(stderr);
        }
    };
    static GmlpProf& gmlp_prof() {
        static GmlpProf p;
        static bool inited = false;
        if (!inited) {
            p.check_env();
            if (p.enabled) {
                std::atexit([](){ /* will dump via the static instance below */ });
            }
            inited = true;
        }
        return p;
    }
    // Register a single atexit dumper.
    struct GmlpProfDumper {
        ~GmlpProfDumper() { gmlp_prof().dump(); }
    };
    static GmlpProfDumper gmlp_prof_dumper;

    using gmlp_clk = std::chrono::steady_clock;
    static inline uint64_t gmlp_now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   gmlp_clk::now().time_since_epoch()).count();
    }

    // GELU(tanh) approximation over a fp16 range, in-place; tracks the max-abs.
    // Returns the max-abs of the (post-GELU) values. Operates on `n` elements
    // starting at `p`. Assumes n % 8 falls back to scalar tail.
    static inline float gelu_inplace_maxabs_fp16(__fp16* p, size_t n) {
        const float32x4_t v_half    = vdupq_n_f32(0.5f);
        const float32x4_t v_one     = vdupq_n_f32(1.0f);
        const float32x4_t v_sqrt2pi = vdupq_n_f32(0.7978845608028654f);
        const float32x4_t v_coeff   = vdupq_n_f32(0.044715f);
        float32x4_t v_max = vdupq_n_f32(0.f);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            float16x8_t xh = vld1q_f16(p + i);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(xh));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(xh));
            auto gelu = [&](float32x4_t x) {
                float32x4_t x2 = vmulq_f32(x, x);
                float32x4_t x3 = vmulq_f32(x2, x);
                float32x4_t z  = vmulq_f32(v_sqrt2pi, vaddq_f32(x, vmulq_f32(v_coeff, x3)));
                float32x4_t zc = vmaxq_f32(vdupq_n_f32(-4.5f), vminq_f32(vdupq_n_f32(4.5f), z));
                float32x4_t z2 = vmulq_f32(zc, zc);
                float32x4_t num = vmulq_f32(zc, vaddq_f32(vdupq_n_f32(27.f), z2));
                float32x4_t den = vaddq_f32(vdupq_n_f32(27.f), vmulq_f32(vdupq_n_f32(9.f), z2));
                float32x4_t tanh_z = vdivq_f32(num, den);
                return vmulq_f32(vmulq_f32(v_half, x), vaddq_f32(v_one, tanh_z));
            };
            float32x4_t g_lo = gelu(x_lo);
            float32x4_t g_hi = gelu(x_hi);
            float16x8_t out = vcombine_f16(vcvt_f16_f32(g_lo), vcvt_f16_f32(g_hi));
            vst1q_f16(p + i, out);
            v_max = vmaxq_f32(v_max, vabsq_f32(g_lo));
            v_max = vmaxq_f32(v_max, vabsq_f32(g_hi));
        }
        float m = vmaxvq_f32(v_max);
        for (; i < n; ++i) {
            float x = static_cast<float>(p[i]);
            float x3 = x * x * x;
            float z  = 0.7978845608028654f * (x + 0.044715f * x3);
            if (z > 4.5f) z = 4.5f; else if (z < -4.5f) z = -4.5f;
            float z2 = z * z;
            float tanh_z = (z * (27.f + z2)) / (27.f + 9.f * z2);
            float gv = 0.5f * x * (1.f + tanh_z);
            p[i] = static_cast<__fp16>(gv);
            float a = std::fabs(gv);
            if (a > m) m = a;
        }
        return m;
    }

    // Vectorized GELU(tanh) over fp16 values + per-cluster max-abs reduction.
    // gate_full is mutated to gelu(gate_full); group_max[c] = max |gate_full[i]|
    // over neurons i in cluster c. Cluster c spans [offsets[c], offsets[c+1]).
    void fused_gelu_and_cluster_max(__fp16* gate_full, size_t /*d_ffn*/,
                                     const uint32_t* offsets, size_t k_groups,
                                     float* group_max) {
        const float32x4_t v_half    = vdupq_n_f32(0.5f);
        const float32x4_t v_one     = vdupq_n_f32(1.0f);
        const float32x4_t v_sqrt2pi = vdupq_n_f32(0.7978845608028654f);
        const float32x4_t v_coeff   = vdupq_n_f32(0.044715f);
        for (size_t g = 0; g < k_groups; ++g) {
            uint32_t s = offsets[g];
            uint32_t e = offsets[g + 1];
            if (e <= s) { group_max[g] = -std::numeric_limits<float>::infinity(); continue; }
            float32x4_t v_max = vdupq_n_f32(0.f);
            __fp16* p = gate_full + s;
            size_t n = static_cast<size_t>(e - s);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                float16x8_t xh = vld1q_f16(p + i);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(xh));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(xh));
                auto gelu = [&](float32x4_t x) {
                    float32x4_t x2 = vmulq_f32(x, x);
                    float32x4_t x3 = vmulq_f32(x2, x);
                    float32x4_t z  = vmulq_f32(v_sqrt2pi, vaddq_f32(x, vmulq_f32(v_coeff, x3)));
                    float32x4_t zc = vmaxq_f32(vdupq_n_f32(-4.5f), vminq_f32(vdupq_n_f32(4.5f), z));
                    float32x4_t z2 = vmulq_f32(zc, zc);
                    float32x4_t num = vmulq_f32(zc, vaddq_f32(vdupq_n_f32(27.f), z2));
                    float32x4_t den = vaddq_f32(vdupq_n_f32(27.f), vmulq_f32(vdupq_n_f32(9.f), z2));
                    float32x4_t tanh_z = vdivq_f32(num, den);
                    return vmulq_f32(vmulq_f32(v_half, x), vaddq_f32(v_one, tanh_z));
                };
                float32x4_t g_lo = gelu(x_lo);
                float32x4_t g_hi = gelu(x_hi);
                float16x8_t out = vcombine_f16(vcvt_f16_f32(g_lo), vcvt_f16_f32(g_hi));
                vst1q_f16(p + i, out);
                v_max = vmaxq_f32(v_max, vabsq_f32(g_lo));
                v_max = vmaxq_f32(v_max, vabsq_f32(g_hi));
            }
            float m = vmaxvq_f32(v_max);
            // Tail: scalar fallback for clusters whose size isn't a multiple of 8.
            for (; i < n; ++i) {
                float x = static_cast<float>(p[i]);
                float x3 = x * x * x;
                float z  = 0.7978845608028654f * (x + 0.044715f * x3);
                if (z > 4.5f) z = 4.5f; else if (z < -4.5f) z = -4.5f;
                float z2 = z * z;
                float tanh_z = (z * (27.f + z2)) / (27.f + 9.f * z2);
                float gv = 0.5f * x * (1.f + tanh_z);
                p[i] = static_cast<__fp16>(gv);
                float a = std::fabs(gv);
                if (a > m) m = a;
            }
            group_max[g] = m;
        }
    }
}

// Fwd-declare: packed (per-cluster contiguous) variant.
static void compute_grouped_mlp_int8_packed(GraphNode& node,
                                             const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                             const std::unordered_map<size_t, size_t>& node_index_map);

void compute_grouped_mlp_int8_node(GraphNode& node,
                                    const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                    const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.k96_packed_base != nullptr) {
        compute_grouped_mlp_int8_packed(node, nodes, node_index_map);
        return;
    }
    const auto& hidden_buf = get_input(node, 0, nodes, node_index_map);
    const auto& gate_buf   = get_input(node, 1, nodes, node_index_map);
    const auto& up_buf     = get_input(node, 2, nodes, node_index_map);
    const auto& down_buf   = get_input(node, 3, nodes, node_index_map);

    if (hidden_buf.precision != Precision::FP16) {
        throw std::runtime_error("grouped_mlp_int8: hidden must be FP16");
    }
    if (!PrecisionTraits::is_integer(gate_buf.precision) || gate_buf.group_size == 0 ||
        !PrecisionTraits::is_integer(up_buf.precision)   || up_buf.group_size == 0 ||
        !PrecisionTraits::is_integer(down_buf.precision) || down_buf.group_size == 0) {
        throw std::runtime_error("grouped_mlp_int8: weights must be group-quantized integer");
    }
    if (gate_buf.precision != up_buf.precision || gate_buf.precision != down_buf.precision) {
        throw std::runtime_error("grouped_mlp_int8: gate/up/down weight precisions must match");
    }
    if (gate_buf.precision != Precision::INT8 && gate_buf.precision != Precision::INT4) {
        throw std::runtime_error("grouped_mlp_int8: only INT8 or INT4 weights supported");
    }
    const bool is_int4 = (gate_buf.precision == Precision::INT4);

    const auto& shape = hidden_buf.shape;
    size_t K = shape.back();   // hidden_dim
    size_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) M *= shape[i];

    size_t d_ffn = gate_buf.is_interleaved ? gate_buf.original_N : gate_buf.shape[0];
    size_t hidden_dim = down_buf.is_interleaved ? down_buf.original_N : down_buf.shape[0];

    const size_t k_groups = node.params.k_groups;
    const size_t k_active = node.params.k_active;
    const auto& offsets = node.params.cluster_offsets;
    if (offsets.size() != k_groups + 1) {
        throw std::runtime_error("grouped_mlp_int8: cluster_offsets size mismatch");
    }
    if (offsets.back() != d_ffn) {
        throw std::runtime_error("grouped_mlp_int8: last offset != D_FFN");
    }

    ensure_gmlp_buffers(M, K, d_ffn, k_groups, k_active);

    const __fp16* hidden_fp16 = hidden_buf.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();

    const int8_t*  gate_w = gate_buf.data_as<int8_t>();
    const __fp16*  gate_s = gate_buf.scales_as_fp16();
    const int8_t*  up_w   = up_buf.data_as<int8_t>();
    const __fp16*  up_s   = up_buf.scales_as_fp16();
    const int8_t*  down_w = down_buf.data_as<int8_t>();
    const __fp16*  down_s = down_buf.scales_as_fp16();

    GmlpProf& prof = gmlp_prof();
    const bool prof_on = prof.enabled;
    uint64_t t0 = prof_on ? gmlp_now_ns() : 0;

    // 1. Quantise all M activation rows to int8 (one pass).
    for (size_t t = 0; t < M; ++t) {
        const __fp16* x = hidden_fp16 + t * K;
        float maxabs = cactus_fp16_max_abs(x, K);
        float x_scale = std::max(maxabs / 127.0f, 1e-10f);
        gmlp_x_scales_batched[t] = x_scale;
        cactus_fp16_to_int8(x, gmlp_x_int8_batched.data() + t * K, K, x_scale);
    }
    if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_quantize += t1 - t0; t0 = t1; }

    // 2. Batched gate_proj: one GEMM call covers all M tokens. For decode (M=1)
    //    this dispatches to gemv; for prefill (M>1) we get the much faster gemm
    //    path instead of M serial gemv calls. Use the precision-aware dispatcher
    //    so INT4 weights take the int4 path.
    cactus_matmul_integer(gate_buf.precision,
                          gmlp_x_int8_batched.data(), gmlp_x_scales_batched.data(),
                          gate_w, gate_s,
                          gmlp_gate_batched.data(),
                          M, K, d_ffn, gate_buf.group_size);
    if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_gate += t1 - t0; t0 = t1; }

    // The per-token routing + selective up/down still loop. This is fine for
    // decode (M=1). For long prefill it's still serial but the gate_proj —
    // typically the largest single matmul in the MLP — has been batched.
    for (size_t t = 0; t < M; ++t) {
        const int8_t* x_int8 = gmlp_x_int8_batched.data() + t * K;
        float x_scale = gmlp_x_scales_batched[t];
        __fp16* gate_full = gmlp_gate_batched.data() + t * d_ffn;
        __fp16* y = output + t * hidden_dim;

        // 3. Fused vectorised GELU(tanh) + per-cluster max-abs (in place on gate row).
        fused_gelu_and_cluster_max(gate_full, d_ffn,
                                    offsets.data(), k_groups,
                                    gmlp_group_max.data());
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_gelu_max += t1 - t0; t0 = t1; }

        // 4. Top-k_active groups.
        std::vector<std::pair<float, uint16_t>> idx(k_groups);
        for (size_t g = 0; g < k_groups; ++g) idx[g] = { gmlp_group_max[g], static_cast<uint16_t>(g) };
        std::partial_sort(idx.begin(), idx.begin() + k_active, idx.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        for (size_t i = 0; i < k_active; ++i) gmlp_active_groups[i] = idx[i].second;
        std::sort(gmlp_active_groups.begin(), gmlp_active_groups.begin() + k_active);
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_topk += t1 - t0; t0 = t1; }

        // 5. Build N-block & K-group runs (cluster spans are 32-aligned by the
        //    converter, so they map cleanly to 4-row N-blocks (32/8=4) and
        //    32-element K-groups).
        gmlp_block_runs.clear();
        gmlp_kgroup_runs.clear();
        size_t i = 0;
        while (i < k_active) {
            uint16_t g0 = gmlp_active_groups[i];
            size_t j = i;
            while (j + 1 < k_active && gmlp_active_groups[j + 1] == gmlp_active_groups[j] + 1) ++j;
            uint32_t s = offsets[g0];
            uint32_t e = offsets[gmlp_active_groups[j] + 1];
            // s,e are multiples of 32 ⇒ multiples of 4 ⇒ multiples of GROUP_SIZE.
            uint32_t blk_start = s / 4;
            uint32_t blk_count = (e - s) / 4;
            if (blk_count > 0) {
                gmlp_block_runs.push_back(static_cast<uint16_t>(blk_start));
                gmlp_block_runs.push_back(static_cast<uint16_t>(blk_count));
                uint32_t kg_start = s / 32;
                uint32_t kg_count = (e - s) / 32;
                gmlp_kgroup_runs.push_back(static_cast<uint16_t>(kg_start));
                gmlp_kgroup_runs.push_back(static_cast<uint16_t>(kg_count));
            }
            i = j + 1;
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_runs += t1 - t0; t0 = t1; }

        // 6. Selective up_proj. INT4/INT8 share layout; only B byte-stride differs.
        if (is_int4) {
            cactus_gemv_int4_active_block_runs(
                x_int8, x_scale,
                up_w, up_s,
                gmlp_up_partial.data(),
                K, d_ffn, up_buf.group_size,
                gmlp_block_runs.data(), gmlp_block_runs.size() / 2);
        } else {
            cactus_gemv_int8_active_block_runs(
                x_int8, x_scale,
                up_w, up_s,
                gmlp_up_partial.data(),
                K, d_ffn, up_buf.group_size,
                gmlp_block_runs.data(), gmlp_block_runs.size() / 2);
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_up += t1 - t0; t0 = t1; }

        // 7. h_active = gate_act * up at active positions; vectorised max-abs.
        // Each run length n_pos = blk_count*4 is a multiple of 32 (cluster
        // spans are 32-aligned by the converter), so we can stride by 8 fp16
        // lanes safely without scalar tail.
        float h_max;
        {
            float16x8_t v_max = vdupq_n_f16(static_cast<__fp16>(1e-6f));
            const __fp16* gp = gate_full;
            const __fp16* up = gmlp_up_partial.data();
            __fp16* hp = gmlp_h_full.data();
            for (size_t r = 0; r < gmlp_block_runs.size(); r += 2) {
                uint32_t pos = uint32_t(gmlp_block_runs[r]) * 4u;
                uint32_t n_pos = uint32_t(gmlp_block_runs[r + 1]) * 4u;
                for (uint32_t k = 0; k < n_pos; k += 8) {
                    float16x8_t a = vld1q_f16(gp + pos + k);
                    float16x8_t b = vld1q_f16(up + pos + k);
                    float16x8_t v = vmulq_f16(a, b);
                    vst1q_f16(hp + pos + k, v);
                    v_max = vmaxq_f16(v_max, vabsq_f16(v));
                }
            }
            // Reduce max across the lanes via fp32 (avoids fp16 reduce intrinsic
            // platform variance).
            float32x4_t m_lo = vcvt_f32_f16(vget_low_f16(v_max));
            float32x4_t m_hi = vcvt_f32_f16(vget_high_f16(v_max));
            float32x4_t mm = vmaxq_f32(m_lo, m_hi);
            h_max = vmaxvq_f32(mm);
            if (h_max < 1e-6f) h_max = 1e-6f;
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_h += t1 - t0; t0 = t1; }
        float h_scale = h_max / 127.0f;
        float inv = 1.0f / h_scale;
        {
            float32x4_t v_inv = vdupq_n_f32(inv);
            const __fp16* hp = gmlp_h_full.data();
            int8_t* qp = gmlp_h_int8.data();
            for (size_t r = 0; r < gmlp_block_runs.size(); r += 2) {
                uint32_t pos = uint32_t(gmlp_block_runs[r]) * 4u;
                uint32_t n_pos = uint32_t(gmlp_block_runs[r + 1]) * 4u;
                for (uint32_t k = 0; k < n_pos; k += 16) {
                    float16x8_t f0 = vld1q_f16(hp + pos + k);
                    float16x8_t f1 = vld1q_f16(hp + pos + k + 8);
                    float32x4_t a0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f0)), v_inv);
                    float32x4_t a1 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f0)), v_inv);
                    float32x4_t a2 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f1)), v_inv);
                    float32x4_t a3 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f1)), v_inv);
                    int32x4_t q0 = vcvtnq_s32_f32(a0);
                    int32x4_t q1 = vcvtnq_s32_f32(a1);
                    int32x4_t q2 = vcvtnq_s32_f32(a2);
                    int32x4_t q3 = vcvtnq_s32_f32(a3);
                    int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                    int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                    int8x16_t out = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                    vst1q_s8(qp + pos + k, out);
                }
            }
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_quant_h += t1 - t0; t0 = t1; }

        // 8. Selective down_proj. Output written for ALL hidden_dim positions.
        if (is_int4) {
            cactus_gemv_int4_active_kgroup_runs(
                gmlp_h_int8.data(), h_scale,
                down_w, down_s,
                y,
                d_ffn, hidden_dim, down_buf.group_size,
                gmlp_kgroup_runs.data(), gmlp_kgroup_runs.size() / 2);
        } else {
            cactus_gemv_int8_active_kgroup_runs(
                gmlp_h_int8.data(), h_scale,
                down_w, down_s,
                y,
                d_ffn, hidden_dim, down_buf.group_size,
                gmlp_kgroup_runs.data(), gmlp_kgroup_runs.size() / 2);
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_down += t1 - t0; t0 = t1; }
        if (prof_on) {
            prof.calls++;
            // Dump every N calls (~one per token across 30 layers ≈ 30 calls/token).
            // 600 calls ≈ 20 tokens of data.
            if (prof.calls % 600 == 0) prof.dump();
        }
    }
}

// ─── K96 PACKED grouped MLP ────────────────────────────────────────────────
//
// Per-cluster contiguous packed format: one mmapped blob holds, for each of
// the K=96 clusters, the cluster's up_w + up_s + down_w + down_s back-to-back.
// For decode (M=1) this lets the kernel walk one contiguous file region per
// active cluster instead of jumping between two separate up_w_full / down_w_full
// arrays. Avoids the d_ffn-sized fp16 intermediate buffer entirely — each
// cluster's `h` slice (≤cluster_size_max ~256 elements) lives in tiny per-thread
// scratch buffers.
//
namespace {
    // Per-thread scratch for the packed kernel. Indexed by an explicit thread
    // index passed from the dispatch lambda (NOT thread_local — the global pool
    // workers vary in count, and we want stable indexing into per-thread y
    // accumulators).
    struct GmlpPackedScratch {
        std::vector<float>  y_acc;    // hidden_dim, fp32 partial sum for y
        std::vector<__fp16> up_part;  // max cluster_size, up_proj output then h
        std::vector<int8_t> h_int8;   // max cluster_size
        std::vector<__fp16> tmp_y;    // hidden_dim, fp16 down-proj output
    };
    thread_local std::vector<__fp16> gmp_gate_batched;     // M * d_ffn
    thread_local std::vector<int8_t> gmp_x_int8_batched;   // M * hidden
    thread_local std::vector<float>  gmp_x_scales_batched; // M
    thread_local std::vector<float>  gmp_group_max;        // k_groups
    thread_local std::vector<uint16_t> gmp_active_groups;  // k_active

    inline void ensure_packed_buffers(size_t M, size_t hidden, size_t d_ffn,
                                       size_t k_groups, size_t k_active) {
        if (gmp_gate_batched.size()      < M * d_ffn)   gmp_gate_batched.resize(M * d_ffn);
        if (gmp_x_int8_batched.size()    < M * hidden)  gmp_x_int8_batched.resize(M * hidden);
        if (gmp_x_scales_batched.size()  < M)           gmp_x_scales_batched.resize(M);
        if (gmp_group_max.size()         < k_groups)    gmp_group_max.resize(k_groups);
        if (gmp_active_groups.size()     < k_active)    gmp_active_groups.resize(k_active);
    }

    // Fused INT4 gate_proj + GeLU(tanh) + per-cluster max-abs. For decode (M=1)
    // only — performs GEMV on int4 gate_w, applies GELU in place to the gate
    // output, and finalizes per-cluster max-abs in registers, without ever
    // re-reading gate_full from main memory after the GEMV stores it.
    //
    // Parallelizes over CONTIGUOUS N-block slabs aligned to actual cluster
    // boundaries (cluster sizes vary 32..256, so 32-element alignment alone is
    // not sufficient — we snap each split point to the nearest cluster offset).
    // Each slab contains 1+ clusters but no cluster ever spans two slabs, so
    // group_max[c] has a single owner and needs no atomic update.
    static inline void fused_gate_proj_gelu_cluster_max_int4(
        const int8_t* x_int8, float x_scale,
        const int8_t* gate_w, const __fp16* gate_s,
        __fp16* gate_full,
        size_t hidden_dim, size_t d_ffn, size_t group_size,
        const uint32_t* offsets, size_t k_groups,
        float* group_max)
    {
        auto& pool = CactusThreading::get_thread_pool();
        const size_t num_workers = pool.num_workers();
        const size_t total_blocks = d_ffn / 4;
        // Match (and slightly exceed) the per-precision gemv thread cap. The
        // generic GEMV path caps to ~5 on macOS/Linux for moderate N. We can
        // afford more here since each worker also folds in the GELU+max post-
        // pass, increasing per-block work, but more than ~8 threads dispatches
        // out-pace the data they process. Override via K96_FUSED_GATE_THREADS.
        static const size_t fused_gate_threads_env = []() -> size_t {
            const char* e = getenv("K96_FUSED_GATE_THREADS");
            if (!e) return 0;
            int v = atoi(e);
            return v > 0 ? static_cast<size_t>(v) : 0;
        }();
        const size_t cap = (fused_gate_threads_env > 0) ? fused_gate_threads_env : 5;
        const size_t threads = std::max<size_t>(1,
            std::min({num_workers, total_blocks / 8, cap, k_groups}));
        // Compute per-worker [block_lo, block_hi) ranges. Slab boundaries MUST
        // sit on actual cluster boundaries — only some 32-element boundaries
        // are cluster boundaries (cluster sizes vary 32..256). We pick a target
        // offset for each split (proportional share of d_ffn) and SNAP to the
        // nearest cluster boundary.
        std::vector<size_t> blk_starts(threads + 1, 0);
        for (size_t w = 1; w < threads; ++w) {
            size_t target_off = (d_ffn * w) / threads;
            // Find the first cluster boundary >= target_off.
            size_t c = 0;
            while (c <= k_groups && offsets[c] < target_off) ++c;
            if (c > k_groups) c = k_groups;
            size_t off = offsets[c];
            blk_starts[w] = off / 4;
        }
        blk_starts[threads] = total_blocks;
        // Discard empty slabs (multiple cluster boundaries could collapse to
        // identical offsets if some clusters are empty). Re-pack.
        size_t kept = 1;
        for (size_t w = 1; w <= threads; ++w) {
            if (blk_starts[w] > blk_starts[kept - 1]) {
                blk_starts[kept++] = blk_starts[w];
            }
        }
        const size_t real_threads = kept - 1;
        // Initialise group_max for empty clusters; non-empty clusters will be
        // exclusively overwritten by their owning worker (slabs are cluster-
        // aligned so a cluster never spans two workers).
        for (size_t c = 0; c < k_groups; ++c) {
            if (offsets[c] == offsets[c + 1]) {
                group_max[c] = -std::numeric_limits<float>::infinity();
            }
        }
        auto worker = [&](size_t wid) {
            const size_t blk_lo = blk_starts[wid];
            const size_t blk_hi = blk_starts[wid + 1];
            if (blk_hi <= blk_lo) return;
            const size_t n_lo = blk_lo * 4;
            const size_t n_hi = blk_hi * 4;
            // GEMV the worker's slab.
            cactus_gemv_int4_block_range(x_int8, x_scale,
                                          gate_w, gate_s,
                                          gate_full,
                                          hidden_dim, /*N=*/n_hi,
                                          group_size,
                                          blk_lo, blk_hi,
                                          /*accumulate=*/false);
            // Walk the clusters that fall fully inside [n_lo, n_hi). Find
            // first cluster c with offsets[c] >= n_lo via linear scan.
            size_t c = 0;
            while (c < k_groups && offsets[c] < n_lo) ++c;
            for (; c < k_groups; ++c) {
                uint32_t cs = offsets[c];
                uint32_t ce = offsets[c + 1];
                if (cs >= n_hi) break;          // out of this slab
                if (ce > n_hi) break;           // safety: never happens since slabs are 32-aligned
                if (ce <= cs) continue;         // empty
                __fp16* p = gate_full + cs;
                float m = gelu_inplace_maxabs_fp16(p, ce - cs);
                group_max[c] = m;
            }
        };
        if (real_threads <= 1) { worker(0); return; }
        // Single-mutex-acquisition dispatch: emplace `real_threads-1` tasks
        // under one mutex lock instead of `real_threads-1` separate enqueues.
        pool.enqueue_n_threads(real_threads - 1, real_threads - 1,
            [&worker](size_t start, size_t end) {
                for (size_t w = start; w < end; ++w) worker(w + 1);
            });
        worker(0);
        pool.wait_all();
    }
}

static void compute_grouped_mlp_int8_packed(GraphNode& node,
                                             const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                             const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& hidden_buf = get_input(node, 0, nodes, node_index_map);
    const auto& gate_buf   = get_input(node, 1, nodes, node_index_map);

    if (hidden_buf.precision != Precision::FP16) {
        throw std::runtime_error("grouped_mlp_int8_packed: hidden must be FP16");
    }
    if (!PrecisionTraits::is_integer(gate_buf.precision) || gate_buf.group_size == 0) {
        throw std::runtime_error("grouped_mlp_int8_packed: gate must be group-quantized");
    }
    const bool is_int4 = node.params.k96_packed_is_int4;
    const size_t group_size = gate_buf.group_size;

    const auto& shape = hidden_buf.shape;
    size_t hidden_dim = shape.back();
    size_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) M *= shape[i];

    size_t d_ffn = gate_buf.is_interleaved ? gate_buf.original_N : gate_buf.shape[0];

    const size_t k_groups = node.params.k_groups;
    size_t k_active = node.params.k_active;
    // Optional runtime override for INT4 K96-packed: lower k_active to reduce
    // up/down bytes touched per token. Validated for accuracy at 42 (~12.5%
    // memory savings vs 48). Setting via K96_K_ACTIVE_OVERRIDE.
    if (is_int4) {
        static const size_t k_active_override = []() -> size_t {
            const char* env = getenv("K96_K_ACTIVE_OVERRIDE");
            if (!env) return 0;
            int v = atoi(env);
            if (v <= 0) return 0;
            return static_cast<size_t>(v);
        }();
        if (k_active_override > 0 && k_active_override < k_active) {
            k_active = k_active_override;
        }
    }
    const auto& offsets = node.params.cluster_offsets;
    if (offsets.size() != k_groups + 1) {
        throw std::runtime_error("grouped_mlp_int8_packed: cluster_offsets size mismatch");
    }

    const char* base = node.params.k96_packed_base;
    const uint32_t* cluster_sizes = node.params.k96_cluster_sizes;
    const uint64_t* up_w_off = node.params.k96_up_w_offsets;
    const uint64_t* up_s_off = node.params.k96_up_s_offsets;
    const uint64_t* dn_w_off = node.params.k96_down_w_offsets;
    const uint64_t* dn_s_off = node.params.k96_down_s_offsets;

    ensure_packed_buffers(M, hidden_dim, d_ffn, k_groups, k_active);

    const __fp16* hidden_fp16 = hidden_buf.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();
    const int8_t* gate_w = gate_buf.data_as<int8_t>();
    const __fp16* gate_s = gate_buf.scales_as_fp16();

    GmlpProf& prof = gmlp_prof();
    const bool prof_on = prof.enabled;
    uint64_t t0 = prof_on ? gmlp_now_ns() : 0;

    // 1. Quantise activations.
    for (size_t t = 0; t < M; ++t) {
        const __fp16* x = hidden_fp16 + t * hidden_dim;
        float maxabs = cactus_fp16_max_abs(x, hidden_dim);
        float xs = std::max(maxabs / 127.0f, 1e-10f);
        gmp_x_scales_batched[t] = xs;
        cactus_fp16_to_int8(x, gmp_x_int8_batched.data() + t * hidden_dim, hidden_dim, xs);
    }
    if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_quantize += t1 - t0; t0 = t1; }

    // 2. Batched gate_proj. For decode (M=1) on INT4 we can use a fused kernel
    //    that rolls gate_proj + GeLU + per-cluster max-abs into a single sweep
    //    over gate weights — eliminating the post-gate pass over gate_full
    //    (`fused_gelu_and_cluster_max`). This is gated by K96_FUSED_GATE_ROUTE
    //    (default OFF: on M-series the parallelization granularity of the fused
    //    path costs more than the saved post-pass for d_ffn ~6k; the saving is
    //    small enough to fall within run-to-run noise).
    static const bool fused_gate_route_enabled = []() {
        const char* env = getenv("K96_FUSED_GATE_ROUTE");
        if (!env) return false;  // default OFF: empirically a slight regression vs legacy
        return atoi(env) != 0;
    }();
    const bool use_fused_gate_route = is_int4 && M == 1 && fused_gate_route_enabled;
    if (use_fused_gate_route) {
        fused_gate_proj_gelu_cluster_max_int4(
            gmp_x_int8_batched.data(), gmp_x_scales_batched[0],
            gate_w, gate_s,
            gmp_gate_batched.data(),
            hidden_dim, d_ffn, gate_buf.group_size,
            offsets.data(), k_groups,
            gmp_group_max.data());
    } else {
        cactus_matmul_integer(gate_buf.precision,
                              gmp_x_int8_batched.data(), gmp_x_scales_batched.data(),
                              gate_w, gate_s,
                              gmp_gate_batched.data(),
                              M, hidden_dim, d_ffn, gate_buf.group_size);
    }
    if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_gate += t1 - t0; t0 = t1; }

    auto& pool = CactusThreading::get_thread_pool();
    const size_t num_workers = pool.num_workers();

    // Per-worker scratch storage. Reused across decode steps.
    // NOT thread_local — we index by an explicit worker id assigned by the
    // dispatcher, and the underlying physical thread varies per launch.
    static std::vector<GmlpPackedScratch> scratch;
    static std::mutex scratch_mu;
    {
        std::lock_guard<std::mutex> lk(scratch_mu);
        if (scratch.size() < num_workers + 1) scratch.resize(num_workers + 1);
        for (size_t w = 0; w < num_workers + 1; ++w) {
            if (scratch[w].y_acc.size()    < hidden_dim) scratch[w].y_acc.resize(hidden_dim);
            if (scratch[w].up_part.size()  < 512)        scratch[w].up_part.resize(512);
            if (scratch[w].h_int8.size()   < 512)        scratch[w].h_int8.resize(512);
            if (scratch[w].tmp_y.size()    < hidden_dim) scratch[w].tmp_y.resize(hidden_dim);
        }
    }

    for (size_t t = 0; t < M; ++t) {
        const int8_t* x_int8 = gmp_x_int8_batched.data() + t * hidden_dim;
        float x_scale = gmp_x_scales_batched[t];
        __fp16* gate_full = gmp_gate_batched.data() + t * d_ffn;
        __fp16* y = output + t * hidden_dim;

        // 3. Fused vectorised GELU(tanh) + per-cluster max-abs.
        // If fused gate-routing was used above, gate_full is already post-GELU
        // and gmp_group_max is already populated — skip this pass.
        if (!use_fused_gate_route) {
            fused_gelu_and_cluster_max(gate_full, d_ffn,
                                        offsets.data(), k_groups,
                                        gmp_group_max.data());
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_gelu_max += t1 - t0; t0 = t1; }

        // 4. Top-k_active groups (by gelu max-abs).
        std::vector<std::pair<float, uint16_t>> idx(k_groups);
        for (size_t g = 0; g < k_groups; ++g) idx[g] = { gmp_group_max[g], static_cast<uint16_t>(g) };
        std::partial_sort(idx.begin(), idx.begin() + k_active, idx.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        for (size_t i = 0; i < k_active; ++i) gmp_active_groups[i] = idx[i].second;
        // Sort active groups so memory walk is monotonic where possible (cache-friendlier).
        std::sort(gmp_active_groups.begin(), gmp_active_groups.begin() + k_active);
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_topk += t1 - t0; t0 = t1; }

        // 5+6+7+8. Fused per-cluster up→h→down. Parallelized across active clusters.
        // Each worker holds a private fp32 y_acc[hidden_dim] partial sum; we
        // sum them into the final y at the end.
        const size_t num_threads = std::min(num_workers, k_active);

        // Runtime toggle for the INT4 two-phase split-down path.
        // Empirically slower than the per-cluster path on M-series CPUs
        // (extra dispatch costs and lost streaming locality more than offset
        // the cluster-tail recovery), so default OFF. Set `K96_SPLIT_DOWN=1`
        // to opt in for benchmarking.
        static const bool int4_split_down_enabled = []() {
            const char* env = getenv("K96_SPLIT_DOWN");
            if (!env) return false;
            return atoi(env) != 0;
        }();
        const bool use_int4_split_down = is_int4 && int4_split_down_enabled;

        // Zero touched scratch regions for the workers we'll dispatch (only used by the
        // legacy single-phase path; INT4 split-down writes y directly).
        if (!use_int4_split_down) {
            for (size_t w = 0; w < num_threads; ++w) {
                std::fill(scratch[w].y_acc.begin(),
                          scratch[w].y_acc.begin() + hidden_dim, 0.0f);
            }
        }

        // Copy thread_local data into stack-local arrays so worker threads can
        // safely access them without depending on the originating thread's TLS.
        // (Thread-local addresses are stable while the originating thread is
        // alive — but only if the kernel is invoked from the same thread that
        // owns the TLS slot; under nested pool dispatch this isn't guaranteed.)
        std::vector<uint16_t> active_groups_copy(gmp_active_groups.begin(),
                                                  gmp_active_groups.begin() + k_active);
        const uint16_t* active_groups_ptr = active_groups_copy.data();
        const __fp16* gate_full_ptr = gate_full;

        const bool pcl_prof = prof_on && (getenv("K96_PROF_PCL") != nullptr);

        // ---- INT4 two-phase split-down path -----------------------------------
        // Phase 1 (cluster-parallel, work-stealing): up_proj + h = gate*up +
        //   per-cluster h_max + h quantize. h_int8 / h_scale stored in a shared
        //   per-cluster registry indexed by active-slot.
        // Phase 2 (N-block-parallel): for each n_block in [0, hidden_dim/4), sum
        //   contributions from all active clusters into a single fp32 accumulator
        //   and write fp16 directly to y. Eliminates the cluster-tail tail since
        //   the unit of work (N-block) has identical cost regardless of cluster.
        if (use_int4_split_down) {
            // Per-token shared registry: contiguous int8 storage for h, plus
            // per-cluster offsets / scales / weight pointers. Sized for k_active
            // clusters of up to 256 elements each (pessimistic but tiny).
            std::vector<int8_t>          h_storage(k_active * 256);
            std::vector<uint32_t>        h_offsets(k_active + 1, 0);
            std::vector<float>           h_scale_per(k_active, 0.0f);
            std::vector<const uint8_t*>  down_w_ptrs(k_active, nullptr);
            std::vector<const __fp16*>   down_s_ptrs(k_active, nullptr);
            std::vector<uint32_t>        N_c_per(k_active, 0);

            // Pre-compute per-cluster offsets into shared h_storage and weight
            // pointers (cheap; serial; lets phase 1 workers just write).
            uint32_t cum = 0;
            for (size_t ai = 0; ai < k_active; ++ai) {
                const uint32_t c = active_groups_ptr[ai];
                const uint32_t N_c = cluster_sizes[c];
                h_offsets[ai] = cum;
                cum += (N_c + 15) & ~15u;  // 16B alignment for vectorised loads
                N_c_per[ai] = N_c;
                if (N_c > 0) {
                    down_w_ptrs[ai] = reinterpret_cast<const uint8_t*>(base + dn_w_off[c]);
                    down_s_ptrs[ai] = reinterpret_cast<const __fp16*>(base + dn_s_off[c]);
                }
            }
            h_offsets[k_active] = cum;
            if (h_storage.size() < cum) h_storage.resize(cum);

            // Phase 1 atomic counter (work-stealing on clusters).
            const size_t N_blocks = hidden_dim / 4;
            // CHUNK=8: each chunk = 8 N-blocks = 32 columns of hidden_dim. With
            // 48 chunks total and 14 workers, ~3.4 chunks/worker — balances
            // dispatch overhead vs work-stealing granularity.
            constexpr size_t CHUNK = 8;
            const size_t num_chunks = (N_blocks + CHUNK - 1) / CHUNK;
            std::atomic<size_t> next_cluster{0};
            std::atomic<size_t> next_chunk{0};

            auto phase1_task = [&, active_groups_ptr, gate_full_ptr, pcl_prof](size_t worker_id) {
                auto& sc = scratch[worker_id];
                uint64_t l_up = 0, l_h = 0, l_q = 0;
                uint64_t lt0 = pcl_prof ? gmlp_now_ns() : 0;
                while (true) {
                    size_t ai = next_cluster.fetch_add(1, std::memory_order_relaxed);
                    if (ai >= k_active) break;
                    const uint32_t c = active_groups_ptr[ai];
                    const uint32_t N_c = cluster_sizes[c];
                    if (N_c == 0) {
                        h_scale_per[ai] = 0.0f;
                        continue;
                    }
                    if (sc.up_part.size() < N_c) sc.up_part.resize(N_c);

                    const int8_t* up_w_c = reinterpret_cast<const int8_t*>(base + up_w_off[c]);
                    const __fp16* up_s_c = reinterpret_cast<const __fp16*>(base + up_s_off[c]);
                    const uint32_t s     = offsets[c];

                    // (a) up_proj on the cluster: K=hidden_dim, N=N_c
                    cactus_gemv_int4_st(x_int8, x_scale, up_w_c, up_s_c,
                                         sc.up_part.data(),
                                         hidden_dim, N_c, group_size);
                    if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_up += lt1 - lt0; lt0 = lt1; }

                    // (b) h = gate_slice * up_partial; track per-cluster max-abs.
                    const __fp16* gp = gate_full_ptr + s;
                    __fp16* up = sc.up_part.data();
                    float h_max;
                    {
                        float16x8_t v_max = vdupq_n_f16(static_cast<__fp16>(1e-6f));
                        for (uint32_t k = 0; k < N_c; k += 8) {
                            float16x8_t a = vld1q_f16(gp + k);
                            float16x8_t b = vld1q_f16(up + k);
                            float16x8_t v = vmulq_f16(a, b);
                            vst1q_f16(up + k, v);  // overwrite up_part in place with h
                            v_max = vmaxq_f16(v_max, vabsq_f16(v));
                        }
                        float32x4_t m_lo = vcvt_f32_f16(vget_low_f16(v_max));
                        float32x4_t m_hi = vcvt_f32_f16(vget_high_f16(v_max));
                        float32x4_t mm = vmaxq_f32(m_lo, m_hi);
                        h_max = vmaxvq_f32(mm);
                        if (h_max < 1e-6f) h_max = 1e-6f;
                    }
                    if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_h += lt1 - lt0; lt0 = lt1; }
                    float h_scale = h_max / 127.0f;
                    float inv = 1.0f / h_scale;
                    h_scale_per[ai] = h_scale;

                    // (c) Quantize h into the SHARED registry slot for this cluster.
                    int8_t* qp = h_storage.data() + h_offsets[ai];
                    {
                        float32x4_t v_inv = vdupq_n_f32(inv);
                        for (uint32_t k = 0; k < N_c; k += 16) {
                            float16x8_t f0 = vld1q_f16(up + k);
                            float16x8_t f1 = vld1q_f16(up + k + 8);
                            float32x4_t a0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f0)), v_inv);
                            float32x4_t a1 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f0)), v_inv);
                            float32x4_t a2 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f1)), v_inv);
                            float32x4_t a3 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f1)), v_inv);
                            int32x4_t q0 = vcvtnq_s32_f32(a0);
                            int32x4_t q1 = vcvtnq_s32_f32(a1);
                            int32x4_t q2 = vcvtnq_s32_f32(a2);
                            int32x4_t q3 = vcvtnq_s32_f32(a3);
                            int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                            int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                            int8x16_t out = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                            vst1q_s8(qp + k, out);
                        }
                    }
                    if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_q += lt1 - lt0; lt0 = lt1; }
                }
                if (pcl_prof) {
                    prof.ns_pcl_up.fetch_add(l_up, std::memory_order_relaxed);
                    prof.ns_pcl_h.fetch_add(l_h, std::memory_order_relaxed);
                    prof.ns_pcl_quant.fetch_add(l_q, std::memory_order_relaxed);
                }
            };

            // Per-task fp32 chunk accumulator; cluster-outer order preserves
            // cluster-contiguous memory access (critical for L1/L2 hit rate).
            auto phase2_task = [&]() {
                alignas(16) float chunk_acc[CHUNK * 4];
                while (true) {
                    size_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= num_chunks) break;
                    const size_t nb_lo = ci * CHUNK;
                    const size_t nb_hi = std::min(nb_lo + CHUNK, N_blocks);
                    const size_t span = nb_hi - nb_lo;
                    // Zero chunk accumulator.
                    for (size_t k = 0; k < span * 4; k += 4) {
                        vst1q_f32(chunk_acc + k, vdupq_n_f32(0.0f));
                    }
                    // Cluster-outer: walk each cluster's down_w contiguously
                    // for our n_block range (cache-friendly).
                    for (size_t ai = 0; ai < k_active; ++ai) {
                        const uint32_t N_c = N_c_per[ai];
                        if (N_c == 0) continue;
                        const float h_scale = h_scale_per[ai];
                        const float32x4_t v_h_scale = vdupq_n_f32(h_scale);
                        const int8_t* A = h_storage.data() + h_offsets[ai];
                        const uint8_t* B_packed = down_w_ptrs[ai];
                        const __fp16* B_scales = down_s_ptrs[ai];
                        const size_t num_groups = N_c / group_size;
                        const int8x16_t a_lo = vld1q_s8(A);
                        const int8x16_t a_hi = vld1q_s8(A + 16);
                        // For N_c > 32, we have multiple groups; each group
                        // updates `a_lo`/`a_hi` from its own A slab below.
                        for (size_t bi = 0; bi < span; ++bi) {
                            const size_t n_block = nb_lo + bi;
                            float32x4_t cluster_sum = vdupq_n_f32(0.0f);
                            // Walk K-groups of this cluster; consecutive
                            // n_blocks are stride-of-N_c*2 apart for B_packed
                            // and stride-of-num_groups*4 for B_scales.
                            int32x4_t acc;
                            int8x16_t b0, b1, b2, b3;
                            int8x16_t a_lo_g, a_hi_g;
                            for (size_t g = 0; g < num_groups; ++g) {
                                const size_t k_base = g * group_size;
                                const uint8_t* b_base =
                                    B_packed + (n_block * N_c + k_base) * 2;
                                if (g == 0 && k_base == 0) {
                                    a_lo_g = a_lo; a_hi_g = a_hi;
                                } else {
                                    const int8_t* ap = A + k_base;
                                    a_lo_g = vld1q_s8(ap);
                                    a_hi_g = vld1q_s8(ap + 16);
                                }
                                acc = vdupq_n_s32(0);
                                unpack_int4_as_int8x16x2(b_base, b1, b0);
                                unpack_int4_as_int8x16x2(b_base + 16, b3, b2);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b0, a_lo_g, 0);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b1, a_lo_g, 1);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b2, a_lo_g, 2);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b3, a_lo_g, 3);
                                unpack_int4_as_int8x16x2(b_base + 32, b1, b0);
                                unpack_int4_as_int8x16x2(b_base + 48, b3, b2);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b0, a_hi_g, 0);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b1, a_hi_g, 1);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b2, a_hi_g, 2);
                                acc = CACTUS_KU_DOTQ_LANE(acc, b3, a_hi_g, 3);
                                float32x4_t scales = vcvt_f32_f16(
                                    vld1_f16(B_scales + (n_block * num_groups + g) * 4));
                                cluster_sum = vmlaq_f32(cluster_sum,
                                                         vcvtq_f32_s32(acc), scales);
                            }
                            float32x4_t prev = vld1q_f32(chunk_acc + bi * 4);
                            prev = vmlaq_f32(prev, cluster_sum, v_h_scale);
                            vst1q_f32(chunk_acc + bi * 4, prev);
                        }
                    }
                    // Write fp16 directly to y for this chunk (no inter-worker
                    // reduction — each n_block is owned exclusively by its worker).
                    for (size_t bi = 0; bi < span; ++bi) {
                        float32x4_t v = vld1q_f32(chunk_acc + bi * 4);
                        vst1_f16(y + (nb_lo + bi) * 4, vcvt_f16_f32(v));
                    }
                }
            };

            uint64_t tp1_0 = prof_on ? gmlp_now_ns() : 0;
            const size_t p1_threads = std::min(num_workers, k_active);
            if (p1_threads <= 1) {
                phase1_task(0);
            } else {
                pool.enqueue_n_threads(p1_threads - 1, p1_threads - 1,
                    [&](size_t start, size_t end) {
                        for (size_t w = start; w < end; ++w) phase1_task(w + 1);
                    });
                phase1_task(0);
                pool.wait_all();
            }
            if (prof_on) { uint64_t tp1_1 = gmlp_now_ns(); prof.ns_up += tp1_1 - tp1_0; }

            uint64_t tp2_0 = prof_on ? gmlp_now_ns() : 0;
            const size_t p2_threads = std::min(num_workers, num_chunks);
            if (p2_threads <= 1) {
                phase2_task();
            } else {
                pool.enqueue_n_threads(p2_threads - 1, p2_threads - 1,
                    [&](size_t /*start*/, size_t /*end*/) {
                        phase2_task();
                    });
                phase2_task();
                pool.wait_all();
            }
            if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_down += t1 - tp2_0; t0 = t1; }
            if (prof_on) {
                prof.calls++;
                if (prof.calls % 600 == 0) prof.dump();
            }
            continue;  // skip legacy path below
        }
        // ---- end INT4 split-down ----------------------------------------------

        // ---- INT4 sub-cluster-split path --------------------------------------
        // Splits BIG clusters (N_c > SUB_SPLIT_THRESHOLD) into 32-aligned
        // sub-row-ranges of ~SUB_SPLIT_TARGET rows each. Each sub-task is still
        // a sequential stream of weights for one cluster's row subrange (up_w
        // slab) and one cluster's k-slice (down_w with one k-group run), so L2
        // streaming locality is preserved. Total work units expand from
        // k_active (~48) to ~50-70 of more uniform size, dropping the
        // straggler tail without adding a global barrier.
        static const size_t sub_split_threshold = []() -> size_t {
            const char* env = getenv("K96_SUB_SPLIT_THRESH");
            // Default 192: only split very large clusters (>192 rows).
            // Empirically best on M-series is (THRESH, TARGET) = (192, 128).
            // Lower thresholds (96/64) split too aggressively — extra
            // dispatch + h_max/quantize duplication + per-sub-task fp32
            // accumulator writes outweigh tail recovery.
            if (!env) return 192;
            int v = atoi(env);
            return v > 0 ? static_cast<size_t>(v) : 192;
        }();
        static const size_t sub_split_target = []() -> size_t {
            const char* env = getenv("K96_SUB_SPLIT_TARGET");
            if (!env) return 128;
            int v = atoi(env);
            return v > 0 ? static_cast<size_t>(v) : 128;
        }();
        static const bool sub_split_enabled = []() {
            const char* env = getenv("K96_SUB_SPLIT");
            // Default OFF: empirically slightly slower than legacy per-cluster
            // dispatch on M-series (the down-GEMV K-slice doesn't shrink its
            // 384-N-block iteration with smaller K, so per-byte arithmetic
            // intensity drops). Set K96_SUB_SPLIT=1 to opt in for benchmarking.
            if (!env) return false;
            return atoi(env) != 0;
        }();

        if (is_int4 && sub_split_enabled) {
            // Find max cluster size among active clusters to decide whether to
            // bother building sub-tasks at all.
            uint32_t max_N_c = 0;
            for (size_t ai = 0; ai < k_active; ++ai) {
                uint32_t N_c = cluster_sizes[active_groups_ptr[ai]];
                if (N_c > max_N_c) max_N_c = N_c;
            }
            if (max_N_c > sub_split_threshold) {
                // Build sub_tasks: each entry is (active_idx, row_off, row_cnt).
                // active_idx indexes active_groups_ptr (so we can recover c).
                // Row offsets and counts are 32-multiples (clusters are 32-aligned).
                struct SubTask { uint16_t ai; uint16_t row_off; uint16_t row_cnt; };
                std::vector<SubTask> sub_tasks;
                sub_tasks.reserve(k_active * 2);
                for (size_t ai = 0; ai < k_active; ++ai) {
                    const uint32_t N_c = cluster_sizes[active_groups_ptr[ai]];
                    if (N_c == 0) continue;
                    if (N_c <= sub_split_threshold) {
                        sub_tasks.push_back({static_cast<uint16_t>(ai), 0,
                                              static_cast<uint16_t>(N_c)});
                    } else {
                        // Split into chunks of ~sub_split_target rows, 32-aligned.
                        size_t n_chunks = (N_c + sub_split_target - 1) / sub_split_target;
                        if (n_chunks < 2) n_chunks = 2;
                        size_t chunk = (N_c + n_chunks - 1) / n_chunks;
                        // Round chunk up to multiple of 32.
                        chunk = (chunk + 31) & ~size_t(31);
                        if (chunk == 0) chunk = 32;
                        for (size_t r = 0; r < N_c; r += chunk) {
                            size_t c_cnt = std::min(chunk, size_t(N_c) - r);
                            sub_tasks.push_back({static_cast<uint16_t>(ai),
                                                  static_cast<uint16_t>(r),
                                                  static_cast<uint16_t>(c_cnt)});
                        }
                    }
                }
                const size_t num_sub = sub_tasks.size();
                const SubTask* sub_tasks_ptr = sub_tasks.data();
                if (getenv("K96_SUB_SPLIT_DEBUG") != nullptr) {
                    static std::atomic<int> printed{0};
                    if (printed.fetch_add(1) < 3) {
                        fprintf(stderr, "[k96 sub-split] k_active=%zu num_sub=%zu max_N_c=%u sizes=",
                                k_active, num_sub, max_N_c);
                        for (size_t ai = 0; ai < k_active && ai < 16; ++ai) {
                            fprintf(stderr, "%u ", cluster_sizes[active_groups_ptr[ai]]);
                        }
                        fprintf(stderr, "\n");
                    }
                }

                const size_t num_threads_sub = std::min(num_workers, num_sub);
                // Zero per-worker fp32 accumulators.
                for (size_t w = 0; w < num_threads_sub; ++w) {
                    std::fill(scratch[w].y_acc.begin(),
                              scratch[w].y_acc.begin() + hidden_dim, 0.0f);
                }

                std::atomic<size_t> next_sub{0};
                auto sub_task_fn = [&, active_groups_ptr, gate_full_ptr, sub_tasks_ptr]
                                    (size_t worker_id) {
                    auto& sc = scratch[worker_id];
                    while (true) {
                        size_t si = next_sub.fetch_add(1, std::memory_order_relaxed);
                        if (si >= num_sub) break;
                        const SubTask st = sub_tasks_ptr[si];
                        const uint32_t c = active_groups_ptr[st.ai];
                        const uint32_t N_c = cluster_sizes[c];
                        const uint32_t row_off = st.row_off;
                        const uint32_t row_cnt = st.row_cnt;
                        if (sc.up_part.size() < row_cnt) sc.up_part.resize(row_cnt);
                        if (sc.h_int8.size()  < row_cnt) sc.h_int8.resize(row_cnt);

                        const int8_t* up_w_c   = reinterpret_cast<const int8_t*>(base + up_w_off[c]);
                        const __fp16* up_s_c   = reinterpret_cast<const __fp16*>(base + up_s_off[c]);
                        const int8_t* down_w_c = reinterpret_cast<const int8_t*>(base + dn_w_off[c]);
                        const __fp16* down_s_c = reinterpret_cast<const __fp16*>(base + dn_s_off[c]);

                        const uint32_t s = offsets[c];
                        const size_t up_num_groups = hidden_dim / group_size;

                        // (a) up_proj on row subrange: shift B base/scales to
                        // start at n_block = row_off/4. N=row_cnt.
                        const size_t n_block_off = row_off / 4;
                        const int8_t* up_w_sub = up_w_c + (n_block_off * hidden_dim) * 2;  // INT4: *2
                        const __fp16* up_s_sub = up_s_c + n_block_off * up_num_groups * 4;
                        cactus_gemv_int4_st(x_int8, x_scale,
                                             up_w_sub, up_s_sub,
                                             sc.up_part.data(),
                                             hidden_dim, row_cnt, group_size);

                        // (b) h = gate_slice * up_partial + max-abs.
                        const __fp16* gp = gate_full_ptr + s + row_off;
                        __fp16* up = sc.up_part.data();
                        float h_max;
                        {
                            float16x8_t v_max = vdupq_n_f16(static_cast<__fp16>(1e-6f));
                            for (uint32_t k = 0; k < row_cnt; k += 8) {
                                float16x8_t a = vld1q_f16(gp + k);
                                float16x8_t b = vld1q_f16(up + k);
                                float16x8_t v = vmulq_f16(a, b);
                                vst1q_f16(up + k, v);
                                v_max = vmaxq_f16(v_max, vabsq_f16(v));
                            }
                            float32x4_t m_lo = vcvt_f32_f16(vget_low_f16(v_max));
                            float32x4_t m_hi = vcvt_f32_f16(vget_high_f16(v_max));
                            float32x4_t mm = vmaxq_f32(m_lo, m_hi);
                            h_max = vmaxvq_f32(mm);
                            if (h_max < 1e-6f) h_max = 1e-6f;
                        }
                        float h_scale = h_max / 127.0f;
                        float inv = 1.0f / h_scale;

                        // (c) Quantize h into sc.h_int8.
                        {
                            float32x4_t v_inv = vdupq_n_f32(inv);
                            int8_t* qp = sc.h_int8.data();
                            for (uint32_t k = 0; k < row_cnt; k += 16) {
                                float16x8_t f0 = vld1q_f16(up + k);
                                float16x8_t f1 = vld1q_f16(up + k + 8);
                                float32x4_t a0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f0)), v_inv);
                                float32x4_t a1 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f0)), v_inv);
                                float32x4_t a2 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f1)), v_inv);
                                float32x4_t a3 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f1)), v_inv);
                                int32x4_t q0 = vcvtnq_s32_f32(a0);
                                int32x4_t q1 = vcvtnq_s32_f32(a1);
                                int32x4_t q2 = vcvtnq_s32_f32(a2);
                                int32x4_t q3 = vcvtnq_s32_f32(a3);
                                int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                                int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                                int8x16_t out = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                                vst1q_s8(qp + k, out);
                            }
                        }

                        // (d+e) down_proj k-slice with FP32 accumulation
                        // directly into per-worker y_acc. Avoids the fp16
                        // tmp_y round-trip — saves a hidden_dim load+store
                        // per sub-task, which matters because sub-task count
                        // is higher than cluster count.
                        cactus_gemv_int4_st_kslice_fp32acc(
                            sc.h_int8.data(), h_scale,
                            down_w_c, down_s_c,
                            sc.y_acc.data(),
                            N_c, hidden_dim, group_size,
                            row_off, row_cnt);
                    }
                };

                // Dispatch: main thread runs as worker 0, others enqueued.
                if (num_threads_sub <= 1) {
                    sub_task_fn(0);
                } else {
                    std::vector<std::future<void>> futures;
                    futures.reserve(num_threads_sub - 1);
                    for (size_t w = 1; w < num_threads_sub; ++w) {
                        futures.push_back(pool.enqueue([&, w](){ sub_task_fn(w); }));
                    }
                    sub_task_fn(0);
                    for (auto& f : futures) f.get();
                }

                if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_down += t1 - t0; t0 = t1; }

                // Reduce per-worker y_acc into final y (fp16). Vectorised.
                {
                    size_t k = 0;
                    for (; k + 4 <= hidden_dim; k += 4) {
                        float32x4_t s = vld1q_f32(scratch[0].y_acc.data() + k);
                        for (size_t w = 1; w < num_threads_sub; ++w) {
                            s = vaddq_f32(s, vld1q_f32(scratch[w].y_acc.data() + k));
                        }
                        vst1_f16(y + k, vcvt_f16_f32(s));
                    }
                    for (; k < hidden_dim; ++k) {
                        float ssum = scratch[0].y_acc[k];
                        for (size_t w = 1; w < num_threads_sub; ++w) ssum += scratch[w].y_acc[k];
                        y[k] = static_cast<__fp16>(ssum);
                    }
                }
                if (prof_on) {
                    prof.calls++;
                    if (prof.calls % 600 == 0) prof.dump();
                }
                continue;  // skip legacy per-cluster path
            }
        }
        // ---- end INT4 sub-cluster-split ---------------------------------------

        // Atomic next-cluster index for work-stealing load balance. Cluster
        // sizes vary from ~24 to ~256, so static partitioning is highly
        // imbalanced. Each worker grabs the next cluster from this counter.
        std::atomic<size_t> next_cluster{0};
        auto cluster_task = [&, active_groups_ptr, gate_full_ptr, pcl_prof](size_t /*cluster_lo*/, size_t /*cluster_hi*/, size_t worker_id) {
            auto& sc = scratch[worker_id];
            uint64_t l_up = 0, l_h = 0, l_q = 0, l_dn = 0, l_acc = 0;
            uint64_t lt0 = pcl_prof ? gmlp_now_ns() : 0;
            while (true) {
                size_t ai = next_cluster.fetch_add(1, std::memory_order_relaxed);
                if (ai >= k_active) break;
                const uint32_t c = active_groups_ptr[ai];
                const uint32_t N_c = cluster_sizes[c];
                if (N_c == 0) continue;
                if (sc.up_part.size() < N_c) sc.up_part.resize(N_c);
                if (sc.h_int8.size()  < N_c) sc.h_int8.resize(N_c);

                const int8_t* up_w_c   = reinterpret_cast<const int8_t*>(base + up_w_off[c]);
                const __fp16* up_s_c   = reinterpret_cast<const __fp16*>(base + up_s_off[c]);
                const int8_t* down_w_c = reinterpret_cast<const int8_t*>(base + dn_w_off[c]);
                const __fp16* down_s_c = reinterpret_cast<const __fp16*>(base + dn_s_off[c]);

                const uint32_t s = offsets[c];

                // (a) up_proj on the cluster: K=hidden_dim, N=N_c
                if (is_int4) {
                    cactus_gemv_int4_st(x_int8, x_scale,
                                         up_w_c, up_s_c,
                                         sc.up_part.data(),
                                         hidden_dim, N_c, group_size);
                } else if (cpu_has_i8mm()) {
                    cactus_gemv_int8_i8mm_st(x_int8, x_scale,
                                              up_w_c, up_s_c,
                                              sc.up_part.data(),
                                              hidden_dim, N_c, group_size);
                } else {
                    cactus_gemv_int8_st(x_int8, x_scale,
                                         up_w_c, up_s_c,
                                         sc.up_part.data(),
                                         hidden_dim, N_c, group_size);
                }
                if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_up += lt1 - lt0; lt0 = lt1; }

                // (b) h = gate_slice * up_partial + per-cluster max-abs.
                const __fp16* gp = gate_full_ptr + s;
                __fp16* up = sc.up_part.data();
                float h_max;
                {
                    float16x8_t v_max = vdupq_n_f16(static_cast<__fp16>(1e-6f));
                    for (uint32_t k = 0; k < N_c; k += 8) {
                        float16x8_t a = vld1q_f16(gp + k);
                        float16x8_t b = vld1q_f16(up + k);
                        float16x8_t v = vmulq_f16(a, b);
                        vst1q_f16(up + k, v);  // overwrite up_part in place with h
                        v_max = vmaxq_f16(v_max, vabsq_f16(v));
                    }
                    float32x4_t m_lo = vcvt_f32_f16(vget_low_f16(v_max));
                    float32x4_t m_hi = vcvt_f32_f16(vget_high_f16(v_max));
                    float32x4_t mm = vmaxq_f32(m_lo, m_hi);
                    h_max = vmaxvq_f32(mm);
                    if (h_max < 1e-6f) h_max = 1e-6f;
                }
                if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_h += lt1 - lt0; lt0 = lt1; }
                float h_scale = h_max / 127.0f;
                float inv = 1.0f / h_scale;
                {
                    float32x4_t v_inv = vdupq_n_f32(inv);
                    int8_t* qp = sc.h_int8.data();
                    for (uint32_t k = 0; k < N_c; k += 16) {
                        float16x8_t f0 = vld1q_f16(up + k);
                        float16x8_t f1 = vld1q_f16(up + k + 8);
                        float32x4_t a0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f0)), v_inv);
                        float32x4_t a1 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f0)), v_inv);
                        float32x4_t a2 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f1)), v_inv);
                        float32x4_t a3 = vmulq_f32(vcvt_f32_f16(vget_high_f16(f1)), v_inv);
                        int32x4_t q0 = vcvtnq_s32_f32(a0);
                        int32x4_t q1 = vcvtnq_s32_f32(a1);
                        int32x4_t q2 = vcvtnq_s32_f32(a2);
                        int32x4_t q3 = vcvtnq_s32_f32(a3);
                        int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                        int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                        int8x16_t out = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                        vst1q_s8(qp + k, out);
                    }
                }
                if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_q += lt1 - lt0; lt0 = lt1; }

                // (c) down_proj into per-worker fp16 tmp_y, then accumulate
                // fp16->fp32 into y_acc.
                __fp16* tmp_y_buf = sc.tmp_y.data();
                if (is_int4) {
                    cactus_gemv_int4_st(sc.h_int8.data(), h_scale,
                                         down_w_c, down_s_c,
                                         tmp_y_buf,
                                         N_c, hidden_dim, group_size);
                } else if (cpu_has_i8mm()) {
                    cactus_gemv_int8_i8mm_st(sc.h_int8.data(), h_scale,
                                              down_w_c, down_s_c,
                                              tmp_y_buf,
                                              N_c, hidden_dim, group_size);
                } else {
                    cactus_gemv_int8_st(sc.h_int8.data(), h_scale,
                                         down_w_c, down_s_c,
                                         tmp_y_buf,
                                         N_c, hidden_dim, group_size);
                }
                if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_dn += lt1 - lt0; lt0 = lt1; }

                {
                    float* ya = sc.y_acc.data();
                    const __fp16* yp = tmp_y_buf;
                    size_t k = 0;
                    for (; k + 8 <= hidden_dim; k += 8) {
                        float32x4_t a0 = vld1q_f32(ya + k);
                        float32x4_t a1 = vld1q_f32(ya + k + 4);
                        float16x8_t v  = vld1q_f16(yp + k);
                        a0 = vaddq_f32(a0, vcvt_f32_f16(vget_low_f16(v)));
                        a1 = vaddq_f32(a1, vcvt_f32_f16(vget_high_f16(v)));
                        vst1q_f32(ya + k, a0);
                        vst1q_f32(ya + k + 4, a1);
                    }
                    for (; k < hidden_dim; ++k) ya[k] += float(yp[k]);
                }
                if (pcl_prof) { uint64_t lt1 = gmlp_now_ns(); l_acc += lt1 - lt0; lt0 = lt1; }
            }
            if (pcl_prof) {
                prof.ns_pcl_up.fetch_add(l_up, std::memory_order_relaxed);
                prof.ns_pcl_h.fetch_add(l_h, std::memory_order_relaxed);
                prof.ns_pcl_quant.fetch_add(l_q, std::memory_order_relaxed);
                prof.ns_pcl_down.fetch_add(l_dn, std::memory_order_relaxed);
                prof.ns_pcl_acc.fetch_add(l_acc, std::memory_order_relaxed);
            }
        };

        if (num_threads <= 1) {
            cluster_task(0, k_active, 0);
        } else {
            // Single-mutex batched dispatch: emplace N-1 tasks in one critical
            // section, then have main thread do worker 0 inline. Avoids the
            // 13× packaged_task heap alloc + mutex acquire of pool.enqueue().
            const size_t per = k_active / num_threads;
            const size_t rem = k_active % num_threads;
            std::atomic<size_t> remaining{num_threads - 1};
            // Use a condition variable + atomic for completion.
            // For simplicity we use the existing pool.enqueue but reduce overhead
            // via shared completion counter.
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads - 1);
            for (size_t w = 1; w < num_threads; ++w) {
                size_t lo = w * per + std::min(w, rem);
                size_t hi = lo + per + (w < rem ? 1 : 0);
                futures.push_back(pool.enqueue([&, lo, hi, w](){
                    cluster_task(lo, hi, w);
                }));
            }
            {
                size_t lo = 0;
                size_t hi = (0 < rem ? per + 1 : per);
                cluster_task(lo, hi, 0);
            }
            for (auto& f : futures) f.get();
        }
        if (prof_on) { uint64_t t1 = gmlp_now_ns(); prof.ns_down += t1 - t0; t0 = t1; }

        // 9. Reduce per-worker y_acc into final y (fp16). Vectorised.
        {
            size_t k = 0;
            for (; k + 4 <= hidden_dim; k += 4) {
                float32x4_t s = vld1q_f32(scratch[0].y_acc.data() + k);
                for (size_t w = 1; w < num_threads; ++w) {
                    s = vaddq_f32(s, vld1q_f32(scratch[w].y_acc.data() + k));
                }
                vst1_f16(y + k, vcvt_f16_f32(s));
            }
            for (; k < hidden_dim; ++k) {
                float s = scratch[0].y_acc[k];
                for (size_t w = 1; w < num_threads; ++w) s += scratch[w].y_acc[k];
                y[k] = static_cast<__fp16>(s);
            }
        }
        if (prof_on) {
            prof.calls++;
            if (prof.calls % 600 == 0) prof.dump();
        }
    }
}

namespace {
    thread_local std::vector<__fp16> moe_compact_hidden_buf;
    thread_local std::vector<__fp16> moe_gate_buf;
    thread_local std::vector<__fp16> moe_up_buf;
    thread_local std::vector<__fp16> moe_expert_out_buf;
    thread_local std::vector<int8_t> moe_lhs_q_buf;
    thread_local std::vector<float> moe_lhs_scales_buf;
    thread_local std::vector<size_t> moe_expert_offsets_buf;  
    thread_local std::vector<size_t> moe_expert_tokens_buf; 
    thread_local std::vector<float> moe_routing_denom_buf; 

    void ensure_moe_buffers(size_t max_tokens, size_t hidden_dim, size_t intermediate_dim,
                            size_t num_experts, size_t top_k) {
        size_t hidden_size = max_tokens * hidden_dim;
        size_t inter_size = max_tokens * intermediate_dim;
        if (moe_compact_hidden_buf.size() < hidden_size) moe_compact_hidden_buf.resize(hidden_size);
        if (moe_gate_buf.size() < inter_size) moe_gate_buf.resize(inter_size);
        if (moe_up_buf.size() < inter_size) moe_up_buf.resize(inter_size);
        if (moe_expert_out_buf.size() < hidden_size) moe_expert_out_buf.resize(hidden_size);
        size_t max_k = std::max(hidden_dim, intermediate_dim);
        size_t quant_size = max_tokens * max_k;
        if (moe_lhs_q_buf.size() < quant_size) moe_lhs_q_buf.resize(quant_size);
        if (moe_lhs_scales_buf.size() < max_tokens) moe_lhs_scales_buf.resize(max_tokens);
        size_t total_assignments = max_tokens * top_k;
        if (moe_expert_offsets_buf.size() < num_experts + 1) moe_expert_offsets_buf.resize(num_experts + 1);
        if (moe_expert_tokens_buf.size() < total_assignments) moe_expert_tokens_buf.resize(total_assignments);
        if (moe_routing_denom_buf.size() < max_tokens) moe_routing_denom_buf.resize(max_tokens);
    }

    void moe_matmul(const __fp16* lhs,
                                            size_t M,
                                            size_t K,
                                            const BufferDesc& rhs_buffer,
                                            __fp16* output,
                                            size_t N,
                                            bool lhs_prequantized = false) {
        if (rhs_buffer.precision == Precision::FP16) {
            cactus_matmul_f16(lhs, rhs_buffer.data_as<__fp16>(), output, M, K, N);
            return;
        }

        if (PrecisionTraits::is_integer(rhs_buffer.precision) && rhs_buffer.group_size > 0) {
            int8_t* lhs_q = moe_lhs_q_buf.data();
            float* lhs_scales = moe_lhs_scales_buf.data();
            if (!lhs_prequantized) {
                for (size_t row = 0; row < M; ++row) {
                    float scale = cactus_fp16_max_abs(lhs + row * K, K) / 127.0f;
                    if (scale < 1e-10f) scale = 1e-10f;
                    lhs_scales[row] = scale;
                    cactus_fp16_to_int8(lhs + row * K, lhs_q + row * K, K, scale);
                }
            }
            cactus_matmul_integer(rhs_buffer.precision,
                           lhs_q, lhs_scales,
                           rhs_buffer.data_as<int8_t>(),
                           rhs_buffer.scales_as_fp16(),
                           output, M, K, N, rhs_buffer.group_size);
            return;
        }

        throw std::runtime_error("moe_layer only supports FP16 or grouped INT4/INT8 expert weights");
    }
}

void compute_moe_layer_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const size_t num_experts = node.params.num_experts;
    const size_t top_k = node.params.num_experts_per_tok;
    const bool normalize_routing = node.params.normalize_routing;
    const float eps = node.params.epsilon;
    const float routed_scaling_factor = node.params.scalar;
    const bool gated = node.params.moe_gated;
    const Activation activation = node.params.activation;
    const size_t base_inputs = gated ? (3 + 3 * num_experts) : (3 + 2 * num_experts);
    bool has_per_expert_scale = node.input_ids.size() == base_inputs + 1;
    if (node.input_ids.size() != base_inputs && node.input_ids.size() != base_inputs + 1) {
        throw std::runtime_error("moe_layer expects " + std::to_string(base_inputs) + " or " + std::to_string(base_inputs + 1) + " inputs, got " + std::to_string(node.input_ids.size()));
    }

    const auto& hidden_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& routing_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& topk_idx_buffer = get_input(node, 2, nodes, node_index_map);

    if (hidden_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
        throw std::runtime_error("moe_layer expects FP16 hidden/output");
    }
    if (topk_idx_buffer.precision != Precision::FP32) {
        throw std::runtime_error("moe_layer expects FP32 topk indices");
    }

    const __fp16* expert_scales_fp16 = nullptr;
    if (has_per_expert_scale) {
        const auto& scale_buffer = get_input(node, base_inputs, nodes, node_index_map);
        if (scale_buffer.precision != Precision::FP16) {
            throw std::runtime_error("moe_layer expects FP16 per_expert_scale");
        }
        expert_scales_fp16 = scale_buffer.data_as<__fp16>();
    }

    const size_t token_count = hidden_buffer.shape[0];
    const size_t hidden_dim = hidden_buffer.shape[1];
    const size_t total_num_experts = routing_buffer.shape[1];

    const auto& w1_0_buffer = get_input(node, 3, nodes, node_index_map);
    const size_t expert_intermediate_dim = w1_0_buffer.shape[0];

    const auto* hidden = hidden_buffer.data_as<__fp16>();
    auto* output = node.output_buffer.data_as<__fp16>();
    const auto* topk_idx = topk_idx_buffer.data_as<float>();
    const auto* routing_fp16 = routing_buffer.precision == Precision::FP16 ? routing_buffer.data_as<__fp16>() : nullptr;
    const auto* routing_fp32 = routing_buffer.precision == Precision::FP32 ? routing_buffer.data_as<float>() : nullptr;

    auto routing_prob = [&](size_t tok, size_t exp) -> float {
        const size_t offset = tok * total_num_experts + exp;
        if (routing_fp16) return static_cast<float>(routing_fp16[offset]);
        return routing_fp32[offset];
    };

    ensure_moe_buffers(token_count, hidden_dim, expert_intermediate_dim, num_experts, top_k);

    size_t* expert_offsets = moe_expert_offsets_buf.data(); 
    size_t* expert_tokens_flat = moe_expert_tokens_buf.data();  

    std::memset(expert_offsets, 0, (num_experts + 1) * sizeof(size_t));
    for (size_t tok = 0; tok < token_count; ++tok) {
        for (size_t k = 0; k < top_k; ++k) {
            float raw_idx = topk_idx[tok * top_k + k];
            if (!std::isfinite(raw_idx)) {
                throw std::runtime_error("moe_layer got non-finite expert index");
            }
            size_t idx = static_cast<size_t>(raw_idx + 0.5f);
            if (idx >= num_experts) {
                throw std::runtime_error("moe_layer got expert index out of range");
            }
            expert_offsets[idx + 1]++;
        }
    }
    
    for (size_t e = 0; e < num_experts; ++e) {
        expert_offsets[e + 1] += expert_offsets[e];
    }
    
    thread_local std::vector<size_t> moe_write_cursors;
    if (moe_write_cursors.size() < num_experts) moe_write_cursors.resize(num_experts);
    std::memcpy(moe_write_cursors.data(), expert_offsets, num_experts * sizeof(size_t));

    for (size_t tok = 0; tok < token_count; ++tok) {
        for (size_t k = 0; k < top_k; ++k) {
            size_t idx = static_cast<size_t>(topk_idx[tok * top_k + k] + 0.5f);
            expert_tokens_flat[moe_write_cursors[idx]++] = tok;
        }
    }

    float* routing_denom = moe_routing_denom_buf.data();
    if (normalize_routing) {
        for (size_t tok = 0; tok < token_count; ++tok) {
            float sum_probs = 0.0f;
            for (size_t k = 0; k < top_k; ++k) {
                size_t idx = static_cast<size_t>(topk_idx[tok * top_k + k] + 0.5f);
                sum_probs += routing_prob(tok, idx);
            }
            routing_denom[tok] = sum_probs + eps;
        }
    }

    std::memset(output, 0, token_count * hidden_dim * sizeof(__fp16));

    for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const size_t start = expert_offsets[expert_idx];
        const size_t end = expert_offsets[expert_idx + 1];
        if (start == end) continue;

        const size_t selected_count = end - start;
        const size_t* selected_tokens = expert_tokens_flat + start;

        const auto& w1_buffer = get_input(node, 3 + expert_idx, nodes, node_index_map);
        const auto& w2_buffer = gated
            ? get_input(node, 3 + 2 * num_experts + expert_idx, nodes, node_index_map)
            : get_input(node, 3 + num_experts + expert_idx, nodes, node_index_map);

        __fp16* compact_hidden = moe_compact_hidden_buf.data();
        for (size_t i = 0; i < selected_count; ++i) {
            std::memcpy(compact_hidden + i * hidden_dim,
                        hidden + selected_tokens[i] * hidden_dim,
                        hidden_dim * sizeof(__fp16));
        }

        __fp16* gate = moe_gate_buf.data();
        __fp16* up = moe_up_buf.data();
        __fp16* expert_out = moe_expert_out_buf.data();

        moe_matmul(compact_hidden, selected_count, hidden_dim, w1_buffer, gate, expert_intermediate_dim);
        const bool w1_was_int8 = w1_buffer.is_grouped_int8();

        switch (activation) {
            case Activation::GELU:
                cactus_gelu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::GELU_ERF:
                cactus_gelu_f16_erf(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::RELU:
                cactus_relu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
            case Activation::SILU:
            default:
                cactus_silu_f16(gate, gate, selected_count * expert_intermediate_dim);
                break;
        }

        if (gated) {
            const auto& w3_buffer = get_input(node, 3 + num_experts + expert_idx, nodes, node_index_map);
            moe_matmul(compact_hidden, selected_count, hidden_dim, w3_buffer, up, expert_intermediate_dim, w1_was_int8);
            cactus_multiply_f16(gate, up, gate, selected_count * expert_intermediate_dim);
        }

        moe_matmul(gate, selected_count, expert_intermediate_dim, w2_buffer, expert_out, hidden_dim);

        for (size_t i = 0; i < selected_count; ++i) {
            const size_t tok = selected_tokens[i];
            float expert_prob = routing_prob(tok, expert_idx);
            if (expert_prob <= 0.0f) continue;

            float route_weight = expert_prob;
            if (normalize_routing) {
                route_weight = expert_prob / routing_denom[tok];
            }
            route_weight *= routed_scaling_factor;
            if (expert_scales_fp16) {
                route_weight *= static_cast<float>(expert_scales_fp16[expert_idx]);
            }

            auto* out_row = output + tok * hidden_dim;
            const auto* expert_row = expert_out + i * hidden_dim;
            cactus_add_scaled_f16(out_row, expert_row, out_row, hidden_dim, route_weight);
        }
    }
}

void compute_rms_norm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buffer = get_input(node, 1, nodes, node_index_map);

    if (input_buffer.shape.size() != 2) {
        throw std::runtime_error("RMS normalization requires 2D input tensor [batch_size, dims], got " +
                                std::to_string(input_buffer.shape.size()) + "D tensor");
    }

    size_t batch_size = input_buffer.shape[0];
    size_t dims = input_buffer.shape[1];

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("RMS normalization only supports FP16 precision");
    }

    cactus_rms_norm_f16(input_buffer.data_as<__fp16>(), weight_buffer.data_as<__fp16>(),
       node.output_buffer.data_as<__fp16>(), batch_size, dims, node.params.epsilon);
}

void compute_rope_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU RoPE operation not yet implemented");
    }

    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& shape = input_buffer.shape;

    if (shape.size() < 4) {
        throw std::runtime_error("RoPE operation requires 4D tensor with shape [batch, seq_len, num_heads, head_dim], got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    if (input_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
        throw std::runtime_error("RoPE operation only supports FP16 precision");
    }

    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t num_heads = shape[2];
    size_t head_dim = shape[3];

    cactus_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                   batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta);
}

void compute_softmax_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& shape = input_buffer.shape;

    if (shape.size() < 2) {
        throw std::runtime_error("Softmax operation requires at least 2D tensor, got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Softmax operation only supports FP16 precision");
    }

    size_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        batch_size *= shape[i];
    }
    size_t vocab_size = shape[shape.size() - 1];

    cactus_softmax_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                      batch_size, 1, vocab_size);
}

void compute_rel_pos_bias_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                               const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 2) {
        throw std::runtime_error("REL_POS_BIAS requires 2 inputs (query, relative_key)");
    }

    const auto& q_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& r_buffer = get_input(node, 1, nodes, node_index_map);
    auto& y_buffer = node.output_buffer;

    if (q_buffer.shape.size() != 4) {
        throw std::runtime_error("REL_POS_BIAS query must be [B, T, H, D]");
    }
    if (r_buffer.shape.size() != 4) {
        throw std::runtime_error("REL_POS_BIAS relative_key must be [B, R, H, D]");
    }
    if (q_buffer.precision != Precision::FP16 || r_buffer.precision != Precision::FP16) {
        throw std::runtime_error("REL_POS_BIAS currently only supports FP16 tensors");
    }

    const size_t B = q_buffer.shape[0];
    const size_t T = q_buffer.shape[1];
    const size_t H = q_buffer.shape[2];
    const size_t D = q_buffer.shape[3];
    const size_t Rb = r_buffer.shape[0];
    const size_t R = r_buffer.shape[1];

    if (Rb != 1 && Rb != B) {
        throw std::runtime_error("REL_POS_BIAS relative_key batch must be 1 or match query batch");
    }
    if (r_buffer.shape[2] != H || r_buffer.shape[3] != D) {
        throw std::runtime_error("REL_POS_BIAS expects matching [H, D] between query and relative_key");
    }
    if (R < (2 * T - 1)) {
        throw std::runtime_error("REL_POS_BIAS requires relative_key length >= 2*T-1");
    }

    const __fp16* q = q_buffer.data_as<__fp16>();
    const __fp16* r = r_buffer.data_as<__fp16>();
    __fp16* y = y_buffer.data_as<__fp16>();

    const float scale = node.params.scale;

    const size_t q_batch_stride = T * H * D;
    const size_t r_batch_stride = R * H * D;
    const size_t y_batch_stride = H * T * T;
    const size_t q_head_stride = D;
    const size_t r_head_stride = D;
    const size_t q_time_stride = H * D;
    const size_t r_time_stride = H * D;

    CactusThreading::parallel_for(B * H * T, CactusThreading::Thresholds::ATTENTION,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t b = work_idx / (H * T);
                const size_t rem = work_idx % (H * T);
                const size_t h = rem / T;
                const size_t t = rem % T;

                const size_t rb = (Rb == 1) ? 0 : b;
                const __fp16* q_vec = q + b * q_batch_stride + t * q_time_stride + h * q_head_stride;
                const __fp16* r_base = r + rb * r_batch_stride + h * r_head_stride;
                __fp16* y_row = y + b * y_batch_stride + h * (T * T) + t * T;

                for (size_t j = 0; j < T; ++j) {
                    const size_t rel_idx = (T - 1) - t + j;
                    const __fp16* r_vec = r_base + rel_idx * r_time_stride;

                    float acc = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        acc += static_cast<float>(q_vec[d]) * static_cast<float>(r_vec[d]);
                    }
                    y_row[j] = static_cast<__fp16>(acc * scale);
                }
            }
        });
}

void compute_attention_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU attention operation not yet implemented");
    }

    if (node.input_ids.size() < 3 || node.input_ids.size() > 4) {
        throw std::runtime_error("Attention operation requires 3 or 4 inputs (query, key, value[, mask]), got " +
                                std::to_string(node.input_ids.size()) + " inputs");
    }

    const auto& query_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& key_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& value_buffer = get_input(node, 2, nodes, node_index_map);
    const BufferDesc* mask_buffer = nullptr;
    if (node.input_ids.size() == 4) {
        mask_buffer = &get_input(node, 3, nodes, node_index_map);
    }
    const auto& q_shape = query_buffer.shape;
    const auto& k_shape = key_buffer.shape;

    if (q_shape.size() < 4) {
        throw std::runtime_error("Attention operation requires 4D tensors [batch, seq_len, num_heads, head_dim], got " +
                                std::to_string(q_shape.size()) + "D tensor");
    }

    if (query_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Attention operation only supports FP16 precision");
    }

    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];
    size_t head_dim = q_shape[3];
    size_t num_kv_heads = k_shape[2];
    size_t kv_seq_len = key_buffer.shape[1];
    size_t v_head_dim = value_buffer.shape[3];
    bool mask_per_head = false;
    const __fp16* mask_ptr = nullptr;

    if (mask_buffer) {
        if (mask_buffer->precision != Precision::FP16) {
            throw std::runtime_error("Attention mask tensor must be FP16");
        }

        if (mask_buffer->shape.size() == 3) {
            if (mask_buffer->shape[0] != batch_size ||
                mask_buffer->shape[1] != seq_len ||
                mask_buffer->shape[2] != kv_seq_len) {
                throw std::runtime_error("Attention mask [B, T, S] shape mismatch");
            }
            mask_per_head = false;
        } else if (mask_buffer->shape.size() == 4) {
            if (mask_buffer->shape[0] != batch_size ||
                mask_buffer->shape[1] != num_q_heads ||
                mask_buffer->shape[2] != seq_len ||
                mask_buffer->shape[3] != kv_seq_len) {
                throw std::runtime_error("Attention mask [B, H, T, S] shape mismatch");
            }
            mask_per_head = true;
        } else {
            throw std::runtime_error("Attention mask must be rank 3 or 4");
        }

        mask_ptr = mask_buffer->data_as<__fp16>();
    }

    cactus_attention_f16(query_buffer.data_as<__fp16>(), key_buffer.data_as<__fp16>(),
                         value_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                         batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, mask_ptr,
                         node.params.position_offset, node.params.window_size, node.params.is_causal,
                         node.params.attention_mask_is_additive, mask_per_head, v_head_dim, node.params.logit_cap);
}

void compute_attention_int8_hybrid_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& query_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& key_new_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& value_new_buffer = get_input(node, 2, nodes, node_index_map);
    const auto& q_shape = query_buffer.shape;

    if (q_shape.size() < 4) {
        throw std::runtime_error("ATTENTION_INT8_HYBRID requires 4D query tensor");
    }

    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];
    size_t head_dim = node.params.head_dim;
    size_t v_head_dim = node.params.v_head_dim;
    size_t num_kv_heads = node.params.num_kv_heads;
    size_t cache_len = node.params.cache_seq_len;
    size_t new_len = key_new_buffer.shape[1];

    cactus_attention_hybrid_int8_fp16(
        query_buffer.data_as<__fp16>(),
        node.params.cached_keys_int8,
        node.params.cached_values_int8,
        node.params.cached_k_scales,
        node.params.cached_v_scales,
        key_new_buffer.data_as<__fp16>(),
        value_new_buffer.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, seq_len, cache_len, new_len,
        num_q_heads, num_kv_heads, head_dim,
        node.params.scale, node.params.position_offset, true,
        node.params.window_size, KV_QUANT_GROUP_SIZE, v_head_dim
    );
}

void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buffer = get_input(node, 1, nodes, node_index_map);
    bool has_bias = node.input_ids.size() > 2;
    float epsilon = node.params.epsilon;

    if (input_buffer.shape.empty()) {
        throw std::runtime_error("LayerNorm requires non-empty input tensor");
    }

    size_t feature_size = input_buffer.shape.back();
    size_t batch_size = input_buffer.total_size / feature_size;

    if (weight_buffer.total_size != feature_size) {
        throw std::runtime_error("LayerNorm weight size mismatch with input feature dimension");
    }

    using BufferDesc = std::remove_reference_t<decltype(weight_buffer)>;
    const BufferDesc* bias_buffer_ptr = nullptr;
    if (has_bias) {
        const auto& bias_buffer = get_input(node, 2, nodes, node_index_map);
        if (bias_buffer.total_size != feature_size) {
            throw std::runtime_error("LayerNorm bias size mismatch with input feature dimension");
        }
        bias_buffer_ptr = &bias_buffer;
    }

    if (input_buffer.precision == Precision::FP16 &&
        weight_buffer.precision == Precision::FP16 &&
        node.output_buffer.precision == Precision::FP16 &&
        (!has_bias || bias_buffer_ptr->precision == Precision::FP16)) {
        cactus_layer_norm_f16(
            input_buffer.data_as<__fp16>(),
            weight_buffer.data_as<__fp16>(),
            has_bias ? bias_buffer_ptr->data_as<__fp16>() : nullptr,
            node.output_buffer.data_as<__fp16>(),
            batch_size,
            feature_size,
            epsilon);
        return;
    }

    std::vector<float> input_float(input_buffer.total_size);
    std::vector<float> weight_float(feature_size);
    std::vector<float> bias_float(feature_size, 0.0f);

    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        std::memcpy(input_float.data(), input_buffer.data_as<float>(), input_buffer.total_size * sizeof(float));
    }

    if (weight_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 weight");
    } else if (weight_buffer.precision == Precision::FP16) {
        const __fp16* weight_fp16 = weight_buffer.data_as<__fp16>();
        for (size_t i = 0; i < feature_size; ++i) {
            weight_float[i] = static_cast<float>(weight_fp16[i]);
        }
    } else {
        std::memcpy(weight_float.data(), weight_buffer.data_as<float>(), feature_size * sizeof(float));
    }

    if (has_bias) {
        const auto& bias_buffer = *bias_buffer_ptr;
        if (bias_buffer.precision == Precision::INT8) {
            throw std::runtime_error("LayerNorm currently does not support INT8 bias");
        } else if (bias_buffer.precision == Precision::FP16) {
            const __fp16* bias_fp16 = bias_buffer.data_as<__fp16>();
            for (size_t i = 0; i < feature_size; ++i) {
                bias_float[i] = static_cast<float>(bias_fp16[i]);
            }
        } else {
            std::memcpy(bias_float.data(), bias_buffer.data_as<float>(), feature_size * sizeof(float));
        }
    }

    std::vector<float> output_float(input_buffer.total_size);
    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input_float.data() + b * feature_size;
        float* output_row = output_float.data() + b * feature_size;

        float mean = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            mean += input_row[i];
        }
        mean /= feature_size;

        float variance = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;

        float std_inv = 1.0f / std::sqrt(variance + epsilon);
        for (size_t i = 0; i < feature_size; ++i) {
            output_row[i] = (input_row[i] - mean) * std_inv * weight_float[i] + bias_float[i];
        }
    }

    if (node.output_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 output");
    } else if (node.output_buffer.precision == Precision::FP16) {
        __fp16* output_fp16 = node.output_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            output_fp16[i] = static_cast<__fp16>(output_float[i]);
        }
    } else {
        std::memcpy(node.output_buffer.data_as<float>(), output_float.data(), input_buffer.total_size * sizeof(float));
    }
}

void compute_conv1d_causal_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("Causal conv requires 3D input [batch, seq_len, in_channels]");
    }
    if (W.shape.size() != 3) {
        throw std::runtime_error("Weight must be 3D");
    }

    const size_t N     = X.shape[0];
    const size_t L     = X.shape[1];
    const size_t C_in  = X.shape[2];
    const size_t W0    = W.shape[0];
    const size_t W1    = W.shape[1];
    const size_t W2    = W.shape[2];
    const size_t dil   = node.params.dilation;
    if (dil < 1) throw std::runtime_error("dilation must be >= 1");

    size_t M = 1;
    size_t C_out = 0;
    const bool standard_layout = (W1 == 1);
    const bool transposed_layout = (W2 == 1);
    if ((!standard_layout && !transposed_layout) || (W0 % C_in != 0)) {
        throw std::runtime_error("Only depthwise causal convolution is supported currently");
    }
    const size_t K = standard_layout ? W2 : W1;
    M = W0 / C_in;
    C_out = C_in * M;

    Y.shape = { N, L, C_out };
    Y.precision = X.precision;

    auto transpose_depthwise_weights_fp16 = [&](const __fp16* src) {
        std::vector<__fp16> transposed(W0 * K);
        for (size_t oc = 0; oc < W0; ++oc) {
            for (size_t k = 0; k < K; ++k) {
                transposed[oc * K + k] = src[(oc * W1 + k) * W2];
            }
        }
        return transposed;
    };

    if (W.precision == Precision::INT8) {
        const size_t W_size = W0 * W1 * W2;
        const int8_t* W_int8 = W.data_as<int8_t>();

        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t K_total = W1 * K;
            const size_t group_size = W.group_size;
            const size_t num_groups = K_total / group_size;

            for (size_t row = 0; row < W0; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        if (transposed_layout && !standard_layout) {
            auto fixed = transpose_depthwise_weights_fp16(W_fp16.data());
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), fixed.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        } else {
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), W_fp16.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        }
    } else if (W.precision == Precision::FP16) {
        if (transposed_layout && !standard_layout) {
            auto fixed = transpose_depthwise_weights_fp16(W.data_as<__fp16>());
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), fixed.data(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        } else {
            cactus_conv1d_causal_depthwise_f16(
                X.data_as<__fp16>(), W.data_as<__fp16>(), Y.data_as<__fp16>(),
                N, L, C_in, K, dil);
        }
    } else {
        throw std::runtime_error("Depthwise causal conv supports INT8/FP16 weights");
    }
}

void compute_conv1d_k3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU causal convolution operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3)
        throw std::runtime_error("Conv requires 3D input [N, C_in, L]!");

    if (W.shape.size() != 3)
        throw std::runtime_error("Weight must be [C_out, C_in, 3]!");

    const size_t N    = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L    = X.shape[2];

    const size_t C_out = W.shape[0];
    const size_t K     = W.shape[2];
    const size_t stride = node.params.stride;

    if (K != 3)
        throw std::runtime_error("Conv1d_k3 only supports K=3!");

    size_t L_out = ((L - 1) / stride) + 1;
    Y.shape     = { N, C_out, L_out };
    Y.precision = X.precision;

    if (X.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d_k3 only supports FP16 activations");
    }

    if (W.precision == Precision::INT8) {
        const size_t W_size = C_out * C_in * K;
        const int8_t* W_int8 = W.data_as<int8_t>();

        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t K_total = C_in * K;
            const size_t group_size = W.group_size;
            const size_t num_groups = K_total / group_size;

            for (size_t row = 0; row < C_out; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W_fp16.data(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else if (W.precision == Precision::FP16) {
        cactus_conv1d_f16_k3(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            Y.data_as<__fp16>(),
            N, L, C_in, C_out, stride
        );
    } else {
        throw std::runtime_error("Conv1d_k3 only supports FP16 and INT8 weights");
    }
}

void compute_conv1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }

    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d expects input [N, C_in, L]");
    }
    if (W.shape.size() != 3) {
        throw std::runtime_error("conv1d weight must be [C_out, C_in, K]");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;

    if (W.shape[1] != C_in) {
        throw std::runtime_error("conv1d weight C_in mismatch");
    }

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d only supports FP16");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d bias only supports FP16/FP32");
        }
    }

    cactus_conv1d_f16(X.data_as<__fp16>(), W.data_as<__fp16>(), bias_ptr,
                      Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride);
}

void compute_conv1d_same_depthwise_k9_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                           const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv1d_same_depthwise_k9 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d_same_depthwise_k9 expects input [N, L, C]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv1d_same_depthwise_k9 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t L = X.shape[1];
    const size_t C = X.shape[2];
    const size_t K = 9;

    if (W.shape.size() == 2) {
        if (W.shape[0] != C || W.shape[1] != K) {
            throw std::runtime_error("conv1d_same_depthwise_k9 weight must be [C, 9]");
        }
    } else if (W.shape.size() == 3) {
        if (W.shape[0] != C || W.shape[1] != 1 || W.shape[2] != K) {
            throw std::runtime_error("conv1d_same_depthwise_k9 weight must be [C, 1, 9]");
        }
    } else {
        throw std::runtime_error("conv1d_same_depthwise_k9 weight must be rank 2 or 3");
    }

    Y.shape = {N, L, C};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C) {
            throw std::runtime_error("conv1d_same_depthwise_k9 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_same_depthwise_k9 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv1d_same_depthwise_f16_k9(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        const size_t W_size = C * K;
        const int8_t* W_int8 = W.data_as<int8_t>();
        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t K_total = K;
            const size_t group_size = W.group_size;
            if (group_size == 0 || (K_total % group_size) != 0 || scales == nullptr) {
                throw std::runtime_error("Grouped INT8 conv1d_same_depthwise_k9 requires valid per-group scales");
            }

            const size_t num_groups = K_total / group_size;
            for (size_t row = 0; row < C; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv1d_same_depthwise_f16_k9(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C
        );
        return;
    }

    throw std::runtime_error("conv1d_same_depthwise_k9 only supports FP16/INT8 weights");
}

void compute_conv2d_k3s2p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_k3s2p1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_k3s2p1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_k3s2p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    const size_t C_out = Y.shape[1];

    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_k3s2p1 input spatial dimensions must be > 0");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_k3s2p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        if (W.shape.size() != 4) {
            throw std::runtime_error("conv2d_k3s2p1 FP16 weight must be [C_out, C_in, 3, 3]");
        }
        cactus_conv2d_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        const size_t K_total = C_in * 9;
        const size_t W_size = C_out * K_total;
        const int8_t* W_int8 = W.data_as<int8_t>();
        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t group_size = W.group_size;
            if (group_size == 0 || (K_total % group_size) != 0 || scales == nullptr) {
                throw std::runtime_error("Grouped INT8 conv2d_k3s2p1 requires valid per-group scales");
            }

            const size_t num_groups = K_total / group_size;
            for (size_t row = 0; row < C_out; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv2d_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    throw std::runtime_error("conv2d_k3s2p1 only supports FP16/INT8 weights");
}

void compute_conv2d_depthwise_k3s2p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                          const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_depthwise_k3s2p1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 expects input [N, C, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 input spatial dimensions must be > 0");
    }

    if (W.shape.size() == 3) {
        if (W.shape[0] != C || W.shape[1] != 3 || W.shape[2] != 3) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be [C, 3, 3]");
        }
    } else if (W.shape.size() == 4) {
        if (W.shape[0] != C || W.shape[1] != 1 || W.shape[2] != 3 || W.shape[3] != 3) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be [C, 1, 3, 3]");
        }
    } else {
        throw std::runtime_error("conv2d_depthwise_k3s2p1 weight must be rank 3 or 4");
    }

    const size_t H_out = (H - 1) / 2 + 1;
    const size_t W_out = (W_in - 1) / 2 + 1;
    Y.shape = {N, C, H_out, W_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C) {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_depthwise_k3s2p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv2d_depthwise_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C, H, W_in
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        const size_t K_total = 9;
        const size_t W_size = C * K_total;
        const int8_t* W_int8 = W.data_as<int8_t>();
        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t group_size = W.group_size;
            if (group_size == 0 || (K_total % group_size) != 0 || scales == nullptr) {
                throw std::runtime_error("Grouped INT8 conv2d_depthwise_k3s2p1 requires valid per-group scales");
            }

            const size_t num_groups = K_total / group_size;
            for (size_t row = 0; row < C; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv2d_depthwise_f16_k3s2p1_nchw(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C, H, W_in
        );
        return;
    }

    throw std::runtime_error("conv2d_depthwise_k3s2p1 only supports FP16/INT8 weights");
}

void compute_conv2d_pointwise_1x1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                       const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv2d_pointwise_1x1 operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_pointwise_1x1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_pointwise_1x1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_pointwise_1x1 input spatial dimensions must be > 0");
    }

    size_t C_out = 0;
    if (W.shape.size() == 2) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in) {
            throw std::runtime_error("conv2d_pointwise_1x1 weight must be [C_out, C_in]");
        }
    } else if (W.shape.size() == 4) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in || W.shape[2] != 1 || W.shape[3] != 1) {
            throw std::runtime_error("conv2d_pointwise_1x1 weight must be [C_out, C_in, 1, 1]");
        }
    } else {
        throw std::runtime_error("conv2d_pointwise_1x1 weight must be rank 2 or 4");
    }

    Y.shape = {N, C_out, H, W_in};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv2d_pointwise_1x1 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_pointwise_1x1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        const size_t K_total = C_in;
        const size_t W_size = C_out * K_total;
        const int8_t* W_int8 = W.data_as<int8_t>();
        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t group_size = W.group_size;
            if (group_size == 0 || (K_total % group_size) != 0 || scales == nullptr) {
                throw std::runtime_error("Grouped INT8 conv2d_pointwise_1x1 requires valid per-group scales");
            }

            const size_t num_groups = K_total / group_size;
            for (size_t row = 0; row < C_out; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out
        );
        return;
    }

    throw std::runtime_error("conv2d_pointwise_1x1 only supports FP16/INT8 weights");
}

void compute_conv1d_pointwise_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                   const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU conv1d_pointwise operation not yet implemented");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 3) {
        throw std::runtime_error("conv1d_pointwise expects input [N, L, C_in]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv1d_pointwise only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t L = X.shape[1];
    const size_t C_in = X.shape[2];

    size_t C_out = 0;
    if (W.shape.size() == 2) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in) {
            throw std::runtime_error("conv1d_pointwise weight must be [C_out, C_in]");
        }
    } else if (W.shape.size() == 3) {
        C_out = W.shape[0];
        if (W.shape[1] != C_in || W.shape[2] != 1) {
            throw std::runtime_error("conv1d_pointwise weight must be [C_out, C_in, 1]");
        }
    } else {
        throw std::runtime_error("conv1d_pointwise weight must be rank 2 or 3");
    }

    Y.shape = {N, L, C_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d_pointwise bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_pointwise bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        cactus_conv1d_pointwise_f16_gemm(
            X.data_as<__fp16>(),
            W.data_as<__fp16>(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C_in, C_out
        );
        return;
    }

    if (W.precision == Precision::INT8) {
        const size_t K_total = C_in;
        const size_t W_size = C_out * K_total;
        const int8_t* W_int8 = W.data_as<int8_t>();
        std::vector<__fp16> W_fp16(W_size);

        if (W.is_grouped_int8()) {
            const __fp16* scales = W.scales_as_fp16();
            const size_t group_size = W.group_size;
            if (group_size == 0 || (K_total % group_size) != 0 || scales == nullptr) {
                throw std::runtime_error("Grouped INT8 conv1d_pointwise requires valid per-group scales");
            }

            const size_t num_groups = K_total / group_size;
            for (size_t row = 0; row < C_out; ++row) {
                for (size_t col = 0; col < K_total; ++col) {
                    size_t idx = row * K_total + col;
                    size_t group_idx = col / group_size;
                    float scale = static_cast<float>(scales[row * num_groups + group_idx]);
                    W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                }
            }
        } else {
            for (size_t i = 0; i < W_size; ++i) {
                W_fp16[i] = static_cast<__fp16>(W_int8[i]);
            }
        }

        cactus_conv1d_pointwise_f16_gemm(
            X.data_as<__fp16>(),
            W_fp16.data(),
            bias_ptr,
            Y.data_as<__fp16>(),
            N, L, C_in, C_out
        );
        return;
    }

    throw std::runtime_error("conv1d_pointwise only supports FP16/INT8 weights");
}

void compute_glu_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                      const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.empty()) {
        throw std::runtime_error("GLU expects non-scalar input");
    }

    int axis = node.params.axis;
    if (axis < 0) axis += static_cast<int>(X.shape.size());
    if (axis < 0 || static_cast<size_t>(axis) >= X.shape.size()) {
        throw std::runtime_error("GLU axis out of range");
    }

    const size_t axis_size = X.shape[static_cast<size_t>(axis)];
    if ((axis_size % 2) != 0) {
        throw std::runtime_error("GLU split dimension must be even");
    }
    const size_t split = axis_size / 2;

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= X.shape[static_cast<size_t>(i)];
    }
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < X.shape.size(); ++i) {
        inner *= X.shape[i];
    }

    std::vector<size_t> out_shape = X.shape;
    out_shape[static_cast<size_t>(axis)] = split;
    Y.shape = out_shape;
    Y.precision = X.precision;

    if (X.precision == Precision::FP16) {
        cactus_glu_f16(X.data_as<__fp16>(), Y.data_as<__fp16>(), outer, split, inner);
        return;
    }

    if (X.precision == Precision::FP32) {
        cactus_glu_f32(X.data_as<float>(), Y.data_as<float>(), outer, split, inner);
        return;
    }

    throw std::runtime_error("GLU only supports FP16/FP32");
}

void compute_batchnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 5) {
        throw std::runtime_error("BatchNorm expects 5 inputs: input, weight, bias, running_mean, running_var");
    }

    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const auto& B = get_input(node, 2, nodes, node_index_map);
    const auto& RM = get_input(node, 3, nodes, node_index_map);
    const auto& RV = get_input(node, 4, nodes, node_index_map);
    auto& Y = node.output_buffer;

    if (X.shape.empty()) {
        throw std::runtime_error("BatchNorm expects non-scalar input");
    }

    int axis = node.params.axis;
    if (axis < 0) axis += static_cast<int>(X.shape.size());
    if (axis < 0 || static_cast<size_t>(axis) >= X.shape.size()) {
        throw std::runtime_error("BatchNorm axis out of range");
    }

    const size_t C = X.shape[static_cast<size_t>(axis)];
    if (W.total_size != C || B.total_size != C || RM.total_size != C || RV.total_size != C) {
        throw std::runtime_error("BatchNorm parameter size mismatch");
    }

    auto load_1d_float = [C](const BufferDesc& buf, const char* name) -> std::vector<float> {
        if (buf.total_size != C) {
            throw std::runtime_error(std::string("BatchNorm parameter size mismatch for ") + name);
        }
        std::vector<float> out(C);
        if (buf.precision == Precision::FP16) {
            const __fp16* p = buf.data_as<__fp16>();
            for (size_t i = 0; i < C; ++i) out[i] = static_cast<float>(p[i]);
        } else if (buf.precision == Precision::FP32) {
            std::memcpy(out.data(), buf.data_as<float>(), C * sizeof(float));
        } else {
            throw std::runtime_error(std::string("BatchNorm parameter ") + name + " must be FP16 or FP32");
        }
        return out;
    };

    const std::vector<float> gamma = load_1d_float(W, "weight");
    const std::vector<float> beta = load_1d_float(B, "bias");
    const std::vector<float> mean = load_1d_float(RM, "running_mean");
    const std::vector<float> var = load_1d_float(RV, "running_var");

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= X.shape[static_cast<size_t>(i)];
    }
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < X.shape.size(); ++i) {
        inner *= X.shape[i];
    }

    Y.shape = X.shape;
    Y.precision = X.precision;

    if (X.precision == Precision::FP16) {
        cactus_batchnorm_f16(
            X.data_as<__fp16>(),
            gamma.data(),
            beta.data(),
            mean.data(),
            var.data(),
            Y.data_as<__fp16>(),
            outer,
            C,
            inner,
            node.params.epsilon
        );
        return;
    }

    if (X.precision == Precision::FP32) {
        cactus_batchnorm_f32(
            X.data_as<float>(),
            gamma.data(),
            beta.data(),
            mean.data(),
            var.data(),
            Y.data_as<float>(),
            outer,
            C,
            inner,
            node.params.epsilon
        );
        return;
    }

    throw std::runtime_error("BatchNorm only supports FP16/FP32 activations");
}

void compute_stft_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                 const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    const size_t C_out = W.shape[0];
    const size_t K = W.shape[2];
    const size_t stride = node.params.stride;
    const size_t num_fft_bins = node.params.num_fft_bins;

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("stft only supports FP16");
    }

    cactus_stft_f16(X.data_as<__fp16>(), W.data_as<__fp16>(),
                            Y.data_as<__fp16>(), N, L, C_in, C_out, K, stride, num_fft_bins);
}

void compute_conv1d_k7s3_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                         const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }

    auto& Y = node.output_buffer;

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t L = X.shape[2];
    
    if (W.shape.size() != 3) throw std::runtime_error("Weight must be 3D");
    const size_t C_in_W = W.shape[0];
    const size_t K = W.shape[1];
    const size_t C_out = W.shape[2];
    const size_t stride = node.params.stride;

    if (C_in != C_in_W) throw std::runtime_error("Channel mismatch in conv1d_k7s3");
    if (K != 7 || stride != 3) throw std::runtime_error("conv1d_k7s3 requires K=7, stride=3");

    if (X.precision != Precision::FP16 || W.precision != Precision::FP16) {
        throw std::runtime_error("Conv1d specialized only supports FP16");
    }
    
    size_t L_out = (L < 7) ? 0 : (L - 7) / 3 + 1;
    Y.shape = {N, C_out, L_out};
    Y.precision = Precision::FP16;

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->total_size != C_out) {
            throw std::runtime_error("conv1d_k7s3 bias size mismatch");
        }
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv1d_k7s3 bias only supports FP16/FP32");
        }
    }

    cactus_conv1d_f16_k7s3_oc8(
        X.data_as<__fp16>(), 
        W.data_as<__fp16>(), 
        bias_ptr,
        Y.data_as<__fp16>(), 
        N, L, C_in, C_out
    );
}

void compute_gated_deltanet_decode_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                        const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 6) {
        throw std::runtime_error("GATED_DELTANET_DECODE expects 6 inputs");
    }

    const auto& q = get_input(node, 0, nodes, node_index_map);
    const auto& k = get_input(node, 1, nodes, node_index_map);
    const auto& v = get_input(node, 2, nodes, node_index_map);
    const auto& g = get_input(node, 3, nodes, node_index_map);
    const auto& b = get_input(node, 4, nodes, node_index_map);
    const auto& s = get_input(node, 5, nodes, node_index_map);

    validate_gated_deltanet_inputs(q, k, v, g, b, s);
    if (q.shape[1] != 1) {
        throw std::runtime_error("GATED_DELTANET_DECODE expects T=1");
    }

    const size_t B = q.shape[0];
    const size_t Hq = q.shape[2];
    const size_t K = q.shape[3];
    const size_t Hv = v.shape[2];
    const size_t V = v.shape[3];
    const size_t qk_heads_from_params = node.params.num_kv_heads;
    if (qk_heads_from_params != 0 && qk_heads_from_params != Hq) {
        throw std::runtime_error("GATED_DELTANET_DECODE num_qk_heads param mismatch");
    }

    std::vector<__fp16> q_cast;
    std::vector<__fp16> k_cast;
    std::vector<__fp16> v_cast;
    std::vector<__fp16> g_cast;
    std::vector<__fp16> b_cast;
    std::vector<__fp16> s_cast;
    const __fp16* q_data = as_fp16_ptr(q, q_cast);
    const __fp16* k_data = as_fp16_ptr(k, k_cast);
    const __fp16* v_data = as_fp16_ptr(v, v_cast);
    const __fp16* g_data = as_fp16_ptr(g, g_cast);
    const __fp16* b_data = as_fp16_ptr(b, b_cast);
    const __fp16* s_data = as_fp16_ptr(s, s_cast);
    __fp16* out = node.output_buffer.data_as<__fp16>();

    cactus_gated_deltanet_decode_f16(
        q_data, k_data, v_data, g_data, b_data, s_data, out,
        B, Hq, Hv, K, V, node.params.scale);
}

void compute_gated_deltanet_prefill_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                         const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.input_ids.size() != 6) {
        throw std::runtime_error("GATED_DELTANET_PREFILL expects 6 inputs");
    }

    const auto& q = get_input(node, 0, nodes, node_index_map);
    const auto& k = get_input(node, 1, nodes, node_index_map);
    const auto& v = get_input(node, 2, nodes, node_index_map);
    const auto& g = get_input(node, 3, nodes, node_index_map);
    const auto& b = get_input(node, 4, nodes, node_index_map);
    const auto& s = get_input(node, 5, nodes, node_index_map);

    validate_gated_deltanet_inputs(q, k, v, g, b, s);

    const size_t B = q.shape[0];
    const size_t T = q.shape[1];
    const size_t Hq = q.shape[2];
    const size_t K = q.shape[3];
    const size_t Hv = v.shape[2];
    const size_t V = v.shape[3];
    const size_t qk_heads_from_params = node.params.num_kv_heads;
    if (qk_heads_from_params != 0 && qk_heads_from_params != Hq) {
        throw std::runtime_error("GATED_DELTANET_PREFILL num_qk_heads param mismatch");
    }

    std::vector<__fp16> q_cast;
    std::vector<__fp16> k_cast;
    std::vector<__fp16> v_cast;
    std::vector<__fp16> g_cast;
    std::vector<__fp16> b_cast;
    std::vector<__fp16> s_cast;
    const __fp16* q_data = as_fp16_ptr(q, q_cast);
    const __fp16* k_data = as_fp16_ptr(k, k_cast);
    const __fp16* v_data = as_fp16_ptr(v, v_cast);
    const __fp16* g_data = as_fp16_ptr(g, g_cast);
    const __fp16* b_data = as_fp16_ptr(b, b_cast);
    const __fp16* s_data = as_fp16_ptr(s, s_cast);
    __fp16* out = node.output_buffer.data_as<__fp16>();

    cactus_gated_deltanet_prefill_f16(
        q_data, k_data, v_data, g_data, b_data, s_data, out,
        B, T, Hq, Hv, K, V, node.params.chunk_size, node.params.scale);
}

void compute_rope_gptj_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& shape = input_buffer.shape;

    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t num_heads = shape[2];
    size_t head_dim = shape[3];
    size_t rot_dim = static_cast<size_t>(node.params.scalar);

    cactus_gpt_j_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                          batch_size, seq_len, num_heads, head_dim, rot_dim,
                          node.params.position_offset, node.params.theta);
}

void compute_groupnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& weight = get_input(node, 1, nodes, node_index_map);
    const auto& bias = get_input(node, 2, nodes, node_index_map);
    float epsilon = node.params.epsilon;

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t spatial_size = 1;
    for (size_t i = 2; i < input.shape.size(); ++i) spatial_size *= input.shape[i];

    size_t num_groups = node.params.num_groups;
    if (num_groups == 0) num_groups = 32;
    
    if (channels % num_groups != 0) {
        throw std::runtime_error("GroupNorm: channels must be divisible by num_groups");
    }

    size_t channels_per_group = channels / num_groups;

    const __fp16* src = input.data_as<__fp16>();
    const __fp16* w = weight.data_as<__fp16>();
    const __fp16* b = bias.data_as<__fp16>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float sum = 0.0f, sum_sq = 0.0f;
            size_t count = 0;

            for (size_t c = 0; c < channels_per_group; ++c) {
                size_t ch = g * channels_per_group + c;
                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + ch * spatial_size + s;
                    float val = static_cast<float>(src[idx]);
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }

            float mean = sum / count;
            float var = (sum_sq / count) - (mean * mean);
            float inv_std = 1.0f / std::sqrt(var + epsilon);

            for (size_t c = 0; c < channels_per_group; ++c) {
                size_t ch = g * channels_per_group + c;
                float wt = static_cast<float>(w[ch]);
                float bi = static_cast<float>(b[ch]);

                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + ch * spatial_size + s;
                    float val = static_cast<float>(src[idx]);
                    dst[idx] = static_cast<__fp16>((val - mean) * inv_std * wt + bi);
                }
            }
        }
    }
}

void compute_lstm_cell_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = get_input(node, 0, nodes, node_index_map);
    const auto& h_prev_buffer = get_input(node, 1, nodes, node_index_map);
    const auto& c_prev_buffer = get_input(node, 2, nodes, node_index_map);
    const auto& weight_ih_buffer = get_input(node, 3, nodes, node_index_map);
    const auto& weight_hh_buffer = get_input(node, 4, nodes, node_index_map);
    const auto& bias_ih_buffer = get_input(node, 5, nodes, node_index_map);
    const auto& bias_hh_buffer = get_input(node, 6, nodes, node_index_map);

    if (input_buffer.precision != Precision::FP16 || h_prev_buffer.precision != Precision::FP16 ||
        c_prev_buffer.precision != Precision::FP16 || weight_ih_buffer.precision != Precision::FP16 ||
        weight_hh_buffer.precision != Precision::FP16 || bias_ih_buffer.precision != Precision::FP16 ||
        bias_hh_buffer.precision != Precision::FP16) {
        throw std::runtime_error("LSTM cell requires all inputs to be FP16");
    }

    if (input_buffer.shape.size() != 2 || h_prev_buffer.shape.size() != 2 || c_prev_buffer.shape.size() != 2) {
        throw std::runtime_error("LSTM cell input/state shapes must be 2D [batch, features]");
    }

    const size_t batch_size = input_buffer.shape[0];
    const size_t input_size = input_buffer.shape[1];
    const size_t hidden_size = h_prev_buffer.shape[1];

    const __fp16* x_input = input_buffer.data_as<__fp16>();
    const __fp16* h_prev = h_prev_buffer.data_as<__fp16>();
    const __fp16* c_prev = c_prev_buffer.data_as<__fp16>();
    const __fp16* weight_ih = weight_ih_buffer.data_as<__fp16>();
    const __fp16* weight_hh = weight_hh_buffer.data_as<__fp16>();
    const __fp16* bias_ih = bias_ih_buffer.data_as<__fp16>();
    const __fp16* bias_hh = bias_hh_buffer.data_as<__fp16>();

    node.output_buffer.shape = {batch_size, hidden_size, 2};
    node.output_buffer.total_size = batch_size * hidden_size * 2;
    node.output_buffer.precision = Precision::FP16;
    node.output_buffer.allocate();

    std::vector<__fp16> h_new_temp(batch_size * hidden_size);
    std::vector<__fp16> c_new_temp(batch_size * hidden_size);

    cactus_lstm_cell_f16(
        x_input, h_prev, c_prev,
        weight_ih, weight_hh,
        bias_ih, bias_hh,
        h_new_temp.data(), c_new_temp.data(),
        batch_size, input_size, hidden_size
    );

    __fp16* output = node.output_buffer.data_as<__fp16>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < hidden_size; ++i) {
            const size_t idx = b * hidden_size + i;
            output[b * hidden_size * 2 + i * 2] = h_new_temp[idx];
            output[b * hidden_size * 2 + i * 2 + 1] = c_new_temp[idx];
        }
    }
}

void compute_altup_predict_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    size_t n = node.params.num_altup_inputs;
    const auto& coefs_buf = get_input(node, 0, nodes, node_index_map);

    std::vector<const __fp16*> stream_ptrs(n);
    for (size_t i = 0; i < n; i++) {
        stream_ptrs[i] = get_input(node, 1 + i, nodes, node_index_map).data_as<__fp16>();
    }

    const auto& stream0_buf = get_input(node, 1, nodes, node_index_map);
    size_t seq_len = stream0_buf.shape[0];
    size_t hidden_dim = stream0_buf.shape[1];

    cactus_altup_predict_f16(
        coefs_buf.data_as<__fp16>(),
        stream_ptrs.data(),
        node.output_buffer.data_as<__fp16>(),
        n, seq_len, hidden_dim);
}

void compute_gaussian_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buf = get_input(node, 0, nodes, node_index_map);
    const __fp16* input = input_buf.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();

    size_t rows = input_buf.shape[0];
    size_t cols = input_buf.shape[1];
    float ppf = node.params.scalar;

    cactus_gaussian_topk_f16(input, output, rows, cols, ppf);
}

void compute_altup_correct_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    size_t n = node.params.num_altup_inputs;
    const auto& coefs_buf = get_input(node, 0, nodes, node_index_map);
    const auto& innov_buf = get_input(node, 1, nodes, node_index_map);

    std::vector<const __fp16*> pred_ptrs(n);
    for (size_t i = 0; i < n; i++) {
        pred_ptrs[i] = get_input(node, 2 + i, nodes, node_index_map).data_as<__fp16>();
    }

    size_t seq_len = innov_buf.shape[0];
    size_t hidden_dim = innov_buf.shape[1];

    cactus_altup_correct_f16(
        coefs_buf.data_as<__fp16>(),
        innov_buf.data_as<__fp16>(),
        pred_ptrs.data(),
        node.output_buffer.data_as<__fp16>(),
        n, seq_len, hidden_dim);
}

void compute_bilstm_sequence_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                   const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& w_ih_fwd = get_input(node, 1, nodes, node_index_map);
    const auto& w_hh_fwd = get_input(node, 2, nodes, node_index_map);
    const auto& b_ih_fwd = get_input(node, 3, nodes, node_index_map);
    const auto& b_hh_fwd = get_input(node, 4, nodes, node_index_map);
    const auto& w_ih_bwd = get_input(node, 5, nodes, node_index_map);
    const auto& w_hh_bwd = get_input(node, 6, nodes, node_index_map);
    const auto& b_ih_bwd = get_input(node, 7, nodes, node_index_map);
    const auto& b_hh_bwd = get_input(node, 8, nodes, node_index_map);

    size_t batch_size = input.shape[0];
    size_t seq_len = input.shape[1];
    size_t input_size = input.shape[2];
    size_t hidden_size = w_ih_fwd.shape[0] / 4;

    cactus_bilstm_sequence_f16(
        input.data_as<__fp16>(),
        w_ih_fwd.data_as<__fp16>(), w_hh_fwd.data_as<__fp16>(),
        b_ih_fwd.data_as<__fp16>(), b_hh_fwd.data_as<__fp16>(),
        w_ih_bwd.data_as<__fp16>(), w_hh_bwd.data_as<__fp16>(),
        b_ih_bwd.data_as<__fp16>(), b_hh_bwd.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, seq_len, input_size, hidden_size);
}

void compute_maxpool1d_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                            const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t input_length = input.shape[2];
    size_t kernel_size = node.params.kernel_size;
    size_t stride = node.params.stride;

    cactus_maxpool1d_f16(
        input.data_as<__fp16>(),
        node.output_buffer.data_as<__fp16>(),
        batch_size, channels, input_length,
        kernel_size, stride);
}

void compute_conv2d_k3s1p1_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                 const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& X = get_input(node, 0, nodes, node_index_map);
    const auto& W = get_input(node, 1, nodes, node_index_map);
    const BufferDesc* B = nullptr;
    if (node.input_ids.size() >= 3) {
        B = &get_input(node, 2, nodes, node_index_map);
    }
    auto& Y = node.output_buffer;

    if (X.shape.size() != 4) {
        throw std::runtime_error("conv2d_k3s1p1 expects input [N, C_in, H, W]");
    }
    if (X.precision != Precision::FP16) {
        throw std::runtime_error("conv2d_k3s1p1 only supports FP16 activations");
    }

    const size_t N = X.shape[0];
    const size_t C_in = X.shape[1];
    const size_t H = X.shape[2];
    const size_t W_in = X.shape[3];
    const size_t C_out = Y.shape[1];

    if (H == 0 || W_in == 0) {
        throw std::runtime_error("conv2d_k3s1p1 input spatial dimensions must be > 0");
    }

    const __fp16* bias_ptr = nullptr;
    std::vector<__fp16> bias_fp16;
    if (B) {
        if (B->precision == Precision::FP16) {
            bias_ptr = B->data_as<__fp16>();
        } else if (B->precision == Precision::FP32) {
            bias_fp16.resize(C_out);
            cactus_fp32_to_fp16(B->data_as<float>(), bias_fp16.data(), C_out);
            bias_ptr = bias_fp16.data();
        } else {
            throw std::runtime_error("conv2d_k3s1p1 bias only supports FP16/FP32");
        }
    }

    if (W.precision == Precision::FP16) {
        if (W.shape.size() != 4) {
            throw std::runtime_error("conv2d_k3s1p1 FP16 weight must be [C_out, C_in, 3, 3]");
        }
        cactus_conv2d_f16_k3s1p1_nchw(
            X.data_as<__fp16>(), W.data_as<__fp16>(), bias_ptr,
            Y.data_as<__fp16>(),
            N, C_in, H, W_in, C_out);
        return;
    }

    throw std::runtime_error("conv2d_k3s1p1 only supports FP16 weights");
}

void compute_stats_pool_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                              const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const __fp16* src = input.data_as<__fp16>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    size_t batch = input.shape[0];
    size_t total_per_batch = input.total_size / batch;
    size_t T = input.shape.back();
    size_t features = total_per_batch / T;

    for (size_t b = 0; b < batch; ++b) {
        const __fp16* batch_src = src + b * total_per_batch;
        __fp16* batch_dst = dst + b * features * 2;

        for (size_t f = 0; f < features; ++f) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                float v = static_cast<float>(batch_src[f * T + t]);
                sum += v;
                sum_sq += v * v;
            }
            float mean = sum / static_cast<float>(T);
            float var = T > 1 ? (sum_sq - static_cast<float>(T) * mean * mean) / static_cast<float>(T - 1) : 0.0f;
            float std_val = sqrtf(fmaxf(var, 0.0f));
            batch_dst[f] = static_cast<__fp16>(mean);
            batch_dst[features + f] = static_cast<__fp16>(std_val);
        }
    }
}

void compute_weighted_stats_pool_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                                       const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input = get_input(node, 0, nodes, node_index_map);
    const auto& weight_buf = get_input(node, 1, nodes, node_index_map);
    const __fp16* src = input.data_as<__fp16>();
    const float* weights = weight_buf.data_as<float>();
    __fp16* dst = node.output_buffer.data_as<__fp16>();

    size_t batch = input.shape[0];
    size_t total_per_batch = input.total_size / batch;
    size_t T = input.shape.back();
    size_t features = total_per_batch / T;

    constexpr float eps = 1e-8f;

    for (size_t b = 0; b < batch; ++b) {
        const __fp16* batch_src = src + b * total_per_batch;
        const float* batch_w = weights + b * T;
        __fp16* batch_dst = dst + b * features * 2;

        float v1 = 0.0f, v2 = 0.0f;
        for (size_t t = 0; t < T; ++t) {
            float w = batch_w[t];
            v1 += w;
            v2 += w * w;
        }
        float v1_safe = v1 + eps;
        float var_denom = v1_safe - v2 / v1_safe + eps;

        for (size_t f = 0; f < features; ++f) {
            float wsum = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                wsum += static_cast<float>(batch_src[f * T + t]) * batch_w[t];
            }
            float mean = wsum / v1_safe;

            float wvar = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                float dx = static_cast<float>(batch_src[f * T + t]) - mean;
                wvar += batch_w[t] * dx * dx;
            }
            float std_val = sqrtf(fmaxf(wvar / var_denom, 0.0f));

            batch_dst[f] = static_cast<__fp16>(mean);
            batch_dst[features + f] = static_cast<__fp16>(std_val);
        }
    }
}

