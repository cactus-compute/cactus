#include "bench_common.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <thread>

namespace bench {

static int s_thread_override = 0;

void set_thread_override(int n) { s_thread_override = n; }
int get_thread_override() { return s_thread_override; }
int get_effective_threads(int backend_default) {
    return s_thread_override > 0 ? s_thread_override : backend_default;
}

const char* matmul_graph_name(MatmulGraph g) {
    switch (g) {
        case MatmulGraph::GEMV_D:  return "gemv_d";
        case MatmulGraph::GEMM_D:  return "gemm_d";
        case MatmulGraph::GEMM_MN: return "gemm_mn";
    }
    return "unknown";
}

const char* attn_graph_name(AttnGraph g) {
    switch (g) {
        case AttnGraph::PREFILL_S:    return "attn_prefill_s";
        case AttnGraph::DECODE_CACHE: return "attn_decode_cache";
    }
    return "unknown";
}

std::vector<MatmulConfig> build_matmul_configs(const MatmulBenchOptions& opt) {
    std::vector<size_t> dims = opt.dims;
    if (dims.empty()) dims = default_dim_sweep();

    std::vector<MatmulConfig> out;
    for (auto g : opt.graphs) {
        for (size_t d : dims) {
            MatmulConfig c{};
            c.graph = g;
            c.sweep_dim = d;
            switch (g) {
                case MatmulGraph::GEMV_D:
                    c.M = 1; c.K = d; c.N = d; break;
                case MatmulGraph::GEMM_D:
                    c.M = 512; c.K = d; c.N = 512; break;
                case MatmulGraph::GEMM_MN:
                    c.M = d; c.K = 512; c.N = d; break;
            }
            out.push_back(c);
        }
    }
    return out;
}

std::vector<AttnConfig> build_attn_configs(const AttnBenchOptions& opt) {
    std::vector<size_t> dims = opt.dims;
    if (dims.empty()) dims = default_dim_sweep();

    AttnDims base{};
    base.head_dim = opt.head_dim_override > 0
                    ? opt.head_dim_override
                    : opt.model_dim / opt.num_heads;
    base.num_q_heads = opt.num_heads;
    base.num_kv_heads = opt.num_kv_heads > 0 ? opt.num_kv_heads : opt.num_heads;

    std::vector<AttnConfig> out;
    for (auto g : opt.graphs) {
        for (size_t d : dims) {
            AttnConfig c{};
            c.graph = g;
            c.dims = base;
            c.sweep_dim = d;
            if (g == AttnGraph::PREFILL_S) {
                c.mode = AttnMode::PREFILL;
                c.seq_len = d;
                c.cache_len = 0;
            } else {
                c.mode = AttnMode::DECODE;
                c.seq_len = 1;
                c.cache_len = d;
            }
            out.push_back(c);
        }
    }
    return out;
}

static void quantize_per_group(const std::vector<float>& src, size_t N, size_t K,
                                std::vector<int8_t>& dst, std::vector<float>& scales,
                                int qmax, int qmin) {
    const size_t num_groups = K / kGroupSize;
    dst.resize(N * K);
    scales.resize(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            const size_t base = n * K + g * kGroupSize;
            for (size_t k = 0; k < kGroupSize; ++k)
                max_abs = std::max(max_abs, std::abs(src[base + k]));
            float scale = std::max(max_abs / static_cast<float>(qmax), 1e-10f);
            scales[n * num_groups + g] = scale;
            for (size_t k = 0; k < kGroupSize; ++k) {
                int q = static_cast<int>(std::round(src[base + k] / scale));
                dst[base + k] = static_cast<int8_t>(std::max(qmin, std::min(qmax, q)));
            }
        }
    }
}

void quantize_int8_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales) {
    quantize_per_group(src, N, K, dst, scales, 127, -128);
}

std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K) {
    const size_t N_blocks = N / kBlockSize;
    const size_t K_blocks = K / kBlockSize;
    std::vector<int8_t> out(N * K);
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t kb = 0; kb < K_blocks; ++kb)
            for (size_t ni = 0; ni < kBlockSize; ++ni)
                for (size_t ki = 0; ki < kBlockSize; ++ki)
                    out[(nb * K_blocks + kb) * kBlockSize * kBlockSize + ni * kBlockSize + ki] =
                        rowmajor[(nb * kBlockSize + ni) * K + kb * kBlockSize + ki];
    return out;
}

std::vector<__fp16> interleave_scales_n4(const std::vector<float>& scales, size_t N, size_t num_groups) {
    const size_t N_blocks = N / kBlockSize;
    std::vector<__fp16> out(N * num_groups);
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t g = 0; g < num_groups; ++g)
            for (size_t ni = 0; ni < kBlockSize; ++ni)
                out[(nb * num_groups + g) * kBlockSize + ni] =
                    static_cast<__fp16>(scales[(nb * kBlockSize + ni) * num_groups + g]);
    return out;
}

CactusActivations prepare_cactus_activations(size_t M, size_t K, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CactusActivations a;

    a.fp32.resize(M * K);
    for (auto& v : a.fp32) v = dist(gen);

    a.fp16.resize(M * K);
    for (size_t i = 0; i < M * K; ++i)
        a.fp16[i] = static_cast<__fp16>(a.fp32[i]);

    a.int8.resize(M * K);
    a.scales.resize(M);
    for (size_t m = 0; m < M; ++m) {
        float max_abs = cactus_fp16_max_abs(a.fp16.data() + m * K, K);
        float scale = std::max(max_abs / 127.0f, 1e-10f);
        a.scales[m] = scale;
        cactus_fp16_to_int8(a.fp16.data() + m * K, a.int8.data() + m * K, K, scale);
    }
    return a;
}

void reference_matmul_fp32(const float* A, const float* B_rowmajor_NK,
                            float* C, size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; m++)
        for (size_t n = 0; n < N; n++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++)
                sum += static_cast<double>(A[m * K + k]) *
                       static_cast<double>(B_rowmajor_NK[n * K + k]);
            C[m * N + n] = static_cast<float>(sum);
        }
}

AccuracyResult check_accuracy(const float* reference, const float* actual,
                               size_t count, float nrmse_tolerance) {
    AccuracyResult r;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (size_t i = 0; i < count; i++) {
        float err = std::abs(reference[i] - actual[i]);
        if (err > r.max_abs_error) r.max_abs_error = err;
        sum_sq_err += static_cast<double>(err) * err;
        sum_sq_ref += static_cast<double>(reference[i]) * reference[i];
    }
    float rms_ref = static_cast<float>(std::sqrt(sum_sq_ref / std::max(count, size_t(1))));
    float rms_err = static_cast<float>(std::sqrt(sum_sq_err / std::max(count, size_t(1))));
    r.nrmse = (rms_ref > 1e-6f) ? rms_err / rms_ref : rms_err;
    r.passed = r.nrmse <= nrmse_tolerance;
    return r;
}

bool framework_matches_filter(const char* framework, const std::string& filter) {
    if (filter.empty()) return true;
    std::string fw(framework);
    size_t pos = 0;
    while (pos < filter.size()) {
        size_t comma = filter.find(',', pos);
        if (comma == std::string::npos) comma = filter.size();
        std::string token = filter.substr(pos, comma - pos);
        if (token == fw) return true;
        pos = comma + 1;
    }
    return false;
}

void reference_attention_fp32(const float* Q, const float* K, const float* V,
                               float* output,
                               size_t num_q_heads, size_t num_kv_heads,
                               size_t seq_len, size_t kv_seq_len,
                               size_t head_dim, float scale,
                               size_t window_size) {
    size_t gqa_ratio = num_q_heads / num_kv_heads;

    for (size_t qh = 0; qh < num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;

        for (size_t sq = 0; sq < seq_len; ++sq) {
            std::vector<double> scores(kv_seq_len);
            double max_score = -1e30;

            const size_t abs_q = sq + (kv_seq_len - seq_len);
            for (size_t sk = 0; sk < kv_seq_len; ++sk) {
                bool causal_masked = sk > abs_q;
                bool window_masked = (window_size > 0 && abs_q > window_size && sk < abs_q - window_size);
                if (causal_masked || window_masked) { scores[sk] = -1e30; continue; }
                double dot = 0.0;
                for (size_t d = 0; d < head_dim; ++d)
                    dot += static_cast<double>(Q[qh * seq_len * head_dim + sq * head_dim + d]) *
                           static_cast<double>(K[kvh * kv_seq_len * head_dim + sk * head_dim + d]);
                scores[sk] = dot * scale;
                if (scores[sk] > max_score) max_score = scores[sk];
            }

            double sum_exp = 0.0;
            for (size_t sk = 0; sk < kv_seq_len; ++sk) {
                scores[sk] = std::exp(scores[sk] - max_score);
                sum_exp += scores[sk];
            }
            for (size_t sk = 0; sk < kv_seq_len; ++sk) scores[sk] /= sum_exp;

            for (size_t d = 0; d < head_dim; ++d) {
                double val = 0.0;
                for (size_t sk = 0; sk < kv_seq_len; ++sk)
                    val += scores[sk] *
                           static_cast<double>(V[kvh * kv_seq_len * head_dim + sk * head_dim + d]);
                output[qh * seq_len * head_dim + sq * head_dim + d] = static_cast<float>(val);
            }
        }
    }
}

void fp32_to_fp16(const float* src, __fp16* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        vst1_f16(reinterpret_cast<float16_t*>(dst + i), vcvt_f16_f32(lo));
        vst1_f16(reinterpret_cast<float16_t*>(dst + i + 4), vcvt_f16_f32(hi));
    }
    for (; i < count; ++i) dst[i] = static_cast<__fp16>(src[i]);
}

void fp16_to_fp32(const __fp16* src, float* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        float16x8_t v = vld1q_f16(reinterpret_cast<const float16_t*>(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(v)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(v)));
    }
    for (; i < count; ++i) dst[i] = static_cast<float>(src[i]);
}

void quantize_rows_int8(const float* src, int8_t* dst, float* scales,
                         size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row = src + r * cols;
        int8_t* out = dst + r * cols;

        float32x4_t vmax = vdupq_n_f32(0.0f);
        size_t c = 0;
        for (; c + 4 <= cols; c += 4)
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(row + c)));
        float max_abs = vmaxvq_f32(vmax);
        for (; c < cols; ++c)
            max_abs = std::max(max_abs, std::abs(row[c]));

        float scale = std::max(max_abs / 127.0f, 1e-10f);
        scales[r] = scale;
        float inv_scale = 1.0f / scale;
        float32x4_t vinv = vdupq_n_f32(inv_scale);

        c = 0;
        for (; c + 8 <= cols; c += 8) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(row + c), vinv);
            float32x4_t f1 = vmulq_f32(vld1q_f32(row + c + 4), vinv);
            int32x4_t i0 = vcvtnq_s32_f32(f0);
            int32x4_t i1 = vcvtnq_s32_f32(f1);
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t b = vqmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(out + c, b);
        }
        for (; c < cols; ++c) {
            int q = static_cast<int>(std::round(row[c] * inv_scale));
            out[c] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    }
}

void transpose_2d(const float* src, float* dst, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            dst[c * rows + r] = src[r * cols + c];
}

static bool parse_dim_list(const std::string& csv, std::vector<size_t>& out) {
    out.clear();
    size_t pos = 0;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        if (comma == std::string::npos) comma = csv.size();
        std::string tok = csv.substr(pos, comma - pos);
        if (tok.empty()) return false;
        try { out.push_back(static_cast<size_t>(std::stoull(tok))); }
        catch (...) { return false; }
        pos = comma + 1;
    }
    return !out.empty();
}

static bool parse_matmul_graphs(const std::string& csv, std::vector<MatmulGraph>& out) {
    out.clear();
    size_t pos = 0;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        if (comma == std::string::npos) comma = csv.size();
        std::string tok = csv.substr(pos, comma - pos);
        if      (tok == "gemv_d")  out.push_back(MatmulGraph::GEMV_D);
        else if (tok == "gemm_d")  out.push_back(MatmulGraph::GEMM_D);
        else if (tok == "gemm_mn") out.push_back(MatmulGraph::GEMM_MN);
        else return false;
        pos = comma + 1;
    }
    return !out.empty();
}

static bool parse_attn_graphs(const std::string& csv, std::vector<AttnGraph>& out) {
    out.clear();
    size_t pos = 0;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        if (comma == std::string::npos) comma = csv.size();
        std::string tok = csv.substr(pos, comma - pos);
        if      (tok == "attn_prefill_s")    out.push_back(AttnGraph::PREFILL_S);
        else if (tok == "attn_decode_cache") out.push_back(AttnGraph::DECODE_CACHE);
        else return false;
        pos = comma + 1;
    }
    return !out.empty();
}

bool parse_matmul_bench_args(int argc, char** argv, MatmulBenchOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--iterations") {
            if (++i >= argc) { err = "Missing --iterations value"; return false; }
            opt.iterations = std::max(1, std::stoi(argv[i]));
        } else if (a == "--warmup") {
            if (++i >= argc) { err = "Missing --warmup value"; return false; }
            opt.warmup = std::max(0, std::stoi(argv[i]));
        } else if (a == "--backends") {
            if (++i >= argc) { err = "Missing --backends value"; return false; }
            opt.backends_filter = argv[i];
        } else if (a == "--graphs") {
            if (++i >= argc) { err = "Missing --graphs value"; return false; }
            if (!parse_matmul_graphs(argv[i], opt.graphs)) {
                err = "Invalid --graphs value (use gemv_d,gemm_d,gemm_mn)";
                return false;
            }
        } else if (a == "--dims") {
            if (++i >= argc) { err = "Missing --dims value"; return false; }
            if (!parse_dim_list(argv[i], opt.dims)) {
                err = "Invalid --dims value";
                return false;
            }
        } else if (a == "--csv") {
            if (++i >= argc) { err = "Missing --csv value"; return false; }
            opt.csv_path = argv[i];
        } else if (a == "--threads") {
            if (++i >= argc) { err = "Missing --threads value"; return false; }
            std::string val(argv[i]);
            if (val == "max")
                opt.num_threads = static_cast<int>(std::thread::hardware_concurrency());
            else
                opt.num_threads = std::max(1, std::stoi(val));
        } else {
            err = "Unknown argument: " + a;
            return false;
        }
    }
    return true;
}

bool parse_attn_bench_args(int argc, char** argv, AttnBenchOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--iterations") {
            if (++i >= argc) { err = "Missing --iterations value"; return false; }
            opt.iterations = std::max(1, std::stoi(argv[i]));
        } else if (a == "--warmup") {
            if (++i >= argc) { err = "Missing --warmup value"; return false; }
            opt.warmup = std::max(0, std::stoi(argv[i]));
        } else if (a == "--backends") {
            if (++i >= argc) { err = "Missing --backends value"; return false; }
            opt.backends_filter = argv[i];
        } else if (a == "--graphs") {
            if (++i >= argc) { err = "Missing --graphs value"; return false; }
            if (!parse_attn_graphs(argv[i], opt.graphs)) {
                err = "Invalid --graphs value (use attn_prefill_s,attn_decode_cache)";
                return false;
            }
        } else if (a == "--dims") {
            if (++i >= argc) { err = "Missing --dims value"; return false; }
            if (!parse_dim_list(argv[i], opt.dims)) {
                err = "Invalid --dims value";
                return false;
            }
        } else if (a == "--model_dim") {
            if (++i >= argc) { err = "Missing --model_dim value"; return false; }
            opt.model_dim = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--heads") {
            if (++i >= argc) { err = "Missing --heads value"; return false; }
            opt.num_heads = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--kv_heads") {
            if (++i >= argc) { err = "Missing --kv_heads value"; return false; }
            opt.num_kv_heads = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--head_dim") {
            if (++i >= argc) { err = "Missing --head_dim value"; return false; }
            opt.head_dim_override = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--csv") {
            if (++i >= argc) { err = "Missing --csv value"; return false; }
            opt.csv_path = argv[i];
        } else if (a == "--threads") {
            if (++i >= argc) { err = "Missing --threads value"; return false; }
            std::string val(argv[i]);
            if (val == "max")
                opt.num_threads = static_cast<int>(std::thread::hardware_concurrency());
            else
                opt.num_threads = std::max(1, std::stoi(val));
        } else {
            err = "Unknown argument: " + a;
            return false;
        }
    }
    return true;
}

} // namespace bench
