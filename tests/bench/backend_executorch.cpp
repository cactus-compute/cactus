#include "bench_driver.h"

#ifdef WITH_EXECUTORCH

#include <xnnpack.h>
#include <pthreadpool.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

static bool s_initialized = false;
static pthreadpool_t s_threadpool = nullptr;
static size_t s_threadpool_threads = 0;

static bool ensure_init() {
    if (!s_initialized) {
        if (xnn_initialize(nullptr) != xnn_status_success) {
            fprintf(stderr, "[executorch] xnn_initialize failed\n");
            return false;
        }
        s_threadpool = pthreadpool_create(0);
        s_threadpool_threads = 0;
        s_initialized = true;
    }
    return true;
}

static void ensure_threadpool(int num_threads) {
    size_t target = (num_threads > 0) ? static_cast<size_t>(num_threads) : 0;
    if (target == s_threadpool_threads) return;
    if (s_threadpool) pthreadpool_destroy(s_threadpool);
    s_threadpool = pthreadpool_create(target);
    s_threadpool_threads = target;
}

static void* aligned_alloc_workspace(size_t size) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
}

static void fill_qparams(struct xnn_quantization_params* qp, size_t M,
                          const float* act_scales) {
    for (size_t m = 0; m < M; ++m) {
        qp[m].zero_point = 0;
        qp[m].scale = act_scales[m];
    }
    for (size_t m = M; m < M + XNN_EXTRA_QUANTIZATION_PARAMS; ++m)
        qp[m] = qp[M ? M - 1 : 0];
}

struct XnnWeights {
    size_t K = 0, N = 0;
    xnn_operator_t op = nullptr;
    size_t current_M = 0;
    size_t workspace_size = 0;
    void* workspace = nullptr;
    std::vector<struct xnn_quantization_params> qp_buf;
    std::vector<float> output_buf;
};

static void reshape(XnnWeights* w, size_t M) {
    size_t ws = 0;
    xnn_reshape_fully_connected_nc_qd8_f32_qc8w(w->op, M, &ws, s_threadpool);
    if (ws > w->workspace_size) {
        free(w->workspace);
        w->workspace = aligned_alloc_workspace(ws);
        w->workspace_size = ws;
    }
    if (w->qp_buf.size() < M + XNN_EXTRA_QUANTIZATION_PARAMS)
        w->qp_buf.resize(M + XNN_EXTRA_QUANTIZATION_PARAMS);
    w->current_M = M;
}

void run_kernel(size_t M, size_t /*K*/, size_t /*N*/,
                void* weights, void*,
                const int8_t* act, const float* act_scales,
                float* output, float*) {
    auto* w = static_cast<XnnWeights*>(weights);
    if (!w || !w->op) return;

    int thr = bench::get_thread_override();
    if (thr > 0) ensure_threadpool(thr);

    w->output_buf.resize(M * w->N);
    if (w->current_M != M) reshape(w, M);
    fill_qparams(w->qp_buf.data(), M, act_scales);
    xnn_setup_fully_connected_nc_qd8_f32_qc8w(
        w->op, act, w->output_buf.data(), w->workspace, w->qp_buf.data());
    xnn_run_operator(w->op, s_threadpool);
    if (output)
        std::memcpy(output, w->output_buf.data(), M * w->N * sizeof(float));
}

void cleanup(void* weights, void*) {
    auto* w = static_cast<XnnWeights*>(weights);
    if (w) {
        if (w->op) xnn_delete_operator(w->op);
        free(w->workspace);
        delete w;
    }
}

void* int8_prepare(const float* fp32, size_t N, size_t K) {
    if (!ensure_init()) return nullptr;

    std::vector<int8_t> qw(N * K);
    std::vector<float> scales(N);
    bench::quantize_rows_int8(fp32, qw.data(), scales.data(), N, K);

    auto* w = new XnnWeights();
    w->K = K;
    w->N = N;
    xnn_status st = xnn_create_fully_connected_nc_qd8_f32_qc8w(
        K, N, K, N,
        scales.data(), qw.data(), nullptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0, nullptr, &w->op);
    if (st != xnn_status_success) {
        fprintf(stderr, "[executorch_int8] create failed (%d)\n", static_cast<int>(st));
        delete w;
        return nullptr;
    }
    return w;
}

static int reg = [] {
    bench::register_matmul_backend({
        "executorch_int8", "executorch",
        int8_prepare, nullptr, run_kernel, cleanup
    });
    return 0;
}();

} // namespace

#else  // !WITH_EXECUTORCH

namespace { [[maybe_unused]] static int reg_xnn = []{ return 0; }(); }

#endif // WITH_EXECUTORCH

// WITH_EXECUTORCH_LLM is separate from WITH_EXECUTORCH because the LLM custom
// ops require a built ExecuTorch tree, not just XNNPACK. cactus prefill is
// FP16 vs ET custom_sdpa FP32-only; cactus decode is FP16 Q + INT8 KV
// per-group(32) vs ET custom_quantized_sdpa INT8 Q/K/V per-channel — the
// asymmetry is reflected in the backend names.

#ifdef WITH_EXECUTORCH_LLM

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

namespace et = ::executorch;
namespace etext = ::executorch::extension;
using et::aten::ScalarType;
using et::aten::Tensor;
using et::runtime::KernelRuntimeContext;
using etext::TensorPtr;
using etext::make_tensor_ptr;

// custom_sdpa_out has no is_seq_at_dim_1 flag (the default BSNH is hard-coded
// per op_sdpa.cpp), so we must transpose driver's BNSH buffers to BSNH.

namespace et_attn_prefill {

struct State {
    bench::AttnDims dims;
    size_t seq_len = 0;
    std::vector<float> q, k, v;
    std::vector<float> out;
};

static void transpose_bnsh_to_bsnh(const float* src, float* dst,
                                    size_t heads, size_t seq, size_t head_dim) {
    for (size_t s = 0; s < seq; ++s)
        for (size_t h = 0; h < heads; ++h) {
            const float* in = src + h * seq * head_dim + s * head_dim;
            float* out = dst + s * heads * head_dim + h * head_dim;
            std::memcpy(out, in, head_dim * sizeof(float));
        }
}
static void transpose_bsnh_to_bnsh(const float* src, float* dst,
                                    size_t heads, size_t seq, size_t head_dim) {
    for (size_t h = 0; h < heads; ++h)
        for (size_t s = 0; s < seq; ++s) {
            const float* in = src + s * heads * head_dim + h * head_dim;
            float* out = dst + h * seq * head_dim + s * head_dim;
            std::memcpy(out, in, head_dim * sizeof(float));
        }
}

void* prepare(const bench::AttnDims& dims, size_t seq_len, size_t /*cache_len*/,
              const float* fp32_q, const float* fp32_k, const float* fp32_v) {
    auto* s = new State();
    s->dims = dims;
    s->seq_len = seq_len;
    size_t q_count = dims.num_q_heads  * seq_len * dims.head_dim;
    size_t k_count = dims.num_kv_heads * seq_len * dims.head_dim;
    s->q.resize(q_count);
    s->k.resize(k_count);
    s->v.resize(k_count);
    s->out.assign(q_count, 0.0f);
    transpose_bnsh_to_bsnh(fp32_q, s->q.data(), dims.num_q_heads,  seq_len, dims.head_dim);
    transpose_bnsh_to_bsnh(fp32_k, s->k.data(), dims.num_kv_heads, seq_len, dims.head_dim);
    transpose_bnsh_to_bsnh(fp32_v, s->v.data(), dims.num_kv_heads, seq_len, dims.head_dim);
    return s;
}

void run(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    const auto& d = s->dims;
    using SZ = ::executorch::aten::SizesType;
    SZ B = 1;
    SZ Hq = static_cast<SZ>(d.num_q_heads);
    SZ Hkv = static_cast<SZ>(d.num_kv_heads);
    SZ S = static_cast<SZ>(s->seq_len);
    SZ D = static_cast<SZ>(d.head_dim);

    auto q_t = make_tensor_ptr({B, S, Hq,  D}, s->q.data(),  ScalarType::Float);
    auto k_t = make_tensor_ptr({B, S, Hkv, D}, s->k.data(),  ScalarType::Float);
    auto v_t = make_tensor_ptr({B, S, Hkv, D}, s->v.data(),  ScalarType::Float);
    auto o_t = make_tensor_ptr({B, S, Hq,  D}, s->out.data(), ScalarType::Float);

    KernelRuntimeContext ctx;
    et::aten::optional<Tensor> no_mask;
    et::aten::optional<double> scale = static_cast<double>(1.0 / std::sqrt(static_cast<float>(d.head_dim)));

    ::torch::executor::native::custom_sdpa_out(
        ctx, *q_t, *k_t, *v_t,
        /*start_pos*/ 0,
        no_mask,
        /*dropout_p*/ 0.0,
        /*is_causal*/ true,
        scale,
        *o_t);

    if (output)
        transpose_bsnh_to_bnsh(s->out.data(), output,
                                d.num_q_heads, s->seq_len, d.head_dim);
}

void cleanup(void* state) { delete static_cast<State*>(state); }

} // namespace et_attn_prefill

namespace et_attn_decode_q8pc {

struct State {
    bench::AttnDims dims;
    size_t kv_seq_len = 0;
    // The header (op_sdpa.h) declares the last bool parameter as
    // `is_seq_at_dim_1`, but the implementation (op_sdpa.cpp) renames the
    // same parameter to `is_seq_at_dim_2` in its body. The body's name
    // describes the actual behavior: passing `true` selects `SeqDim::TWO`
    // (BNSH = seq at dim 2). We pass `true` to match our BNSH buffer.
    // Scales/zp must be 4D matching the first N-1 dims of the data tensor
    // ([B, N, S, 1]); zero-points must be Char (INT8).
    std::vector<int8_t> q_int8, k_int8, v_int8;
    std::vector<float>  q_scales, k_scales, v_scales;
    std::vector<int8_t> q_zp, k_zp, v_zp;
    std::vector<float>  out;
};

static void quantize_bnsh(const float* src, size_t heads, size_t seq, size_t head_dim,
                           int8_t* q_out, float* scales_out, int8_t* zp_out) {
    for (size_t h = 0; h < heads; ++h) {
        for (size_t s = 0; s < seq; ++s) {
            const float* in = src + (h * seq + s) * head_dim;
            const size_t row_idx = h * seq + s;
            int8_t* qrow = q_out + row_idx * head_dim;
            float max_abs = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) max_abs = std::max(max_abs, std::abs(in[d]));
            float scale = std::max(max_abs / 127.0f, 1e-10f);
            float inv = 1.0f / scale;
            scales_out[row_idx] = scale;
            zp_out[row_idx] = 0;
            for (size_t d = 0; d < head_dim; ++d) {
                int q = static_cast<int>(std::round(in[d] * inv));
                qrow[d] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
            }
        }
    }
}

void* prepare(const bench::AttnDims& dims, size_t /*seq_len*/, size_t cache_len,
              const float* fp32_q, const float* fp32_k, const float* fp32_v) {
    auto* s = new State();
    s->dims = dims;
    s->kv_seq_len = cache_len + 1;

    const size_t hd = dims.head_dim;
    const size_t q_rows  = 1 * dims.num_q_heads;
    const size_t kv_rows = s->kv_seq_len * dims.num_kv_heads;

    s->q_int8.resize(q_rows  * hd);   s->q_scales.resize(q_rows);   s->q_zp.assign(q_rows, 0);
    s->k_int8.resize(kv_rows * hd);   s->k_scales.resize(kv_rows);  s->k_zp.assign(kv_rows, 0);
    s->v_int8.resize(kv_rows * hd);   s->v_scales.resize(kv_rows);  s->v_zp.assign(kv_rows, 0);

    quantize_bnsh(fp32_q, dims.num_q_heads,  1,            hd,
                   s->q_int8.data(), s->q_scales.data(), s->q_zp.data());
    quantize_bnsh(fp32_k, dims.num_kv_heads, s->kv_seq_len, hd,
                   s->k_int8.data(), s->k_scales.data(), s->k_zp.data());
    quantize_bnsh(fp32_v, dims.num_kv_heads, s->kv_seq_len, hd,
                   s->v_int8.data(), s->v_scales.data(), s->v_zp.data());

    s->out.assign(dims.num_q_heads * hd, 0.0f);
    return s;
}

void run(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    const auto& d = s->dims;
    using SZ = ::executorch::aten::SizesType;
    SZ B = 1;
    SZ Hq = static_cast<SZ>(d.num_q_heads);
    SZ Hkv = static_cast<SZ>(d.num_kv_heads);
    SZ S = 1;
    SZ KVL = static_cast<SZ>(s->kv_seq_len);
    SZ D = static_cast<SZ>(d.head_dim);

    auto q_t  = make_tensor_ptr({B, Hq,  S,   D}, s->q_int8.data(), ScalarType::Char);
    auto k_t  = make_tensor_ptr({B, Hkv, KVL, D}, s->k_int8.data(), ScalarType::Char);
    auto v_t  = make_tensor_ptr({B, Hkv, KVL, D}, s->v_int8.data(), ScalarType::Char);
    auto qs_t = make_tensor_ptr({B, Hq,  S,   1}, s->q_scales.data(), ScalarType::Float);
    auto ks_t = make_tensor_ptr({B, Hkv, KVL, 1}, s->k_scales.data(), ScalarType::Float);
    auto vs_t = make_tensor_ptr({B, Hkv, KVL, 1}, s->v_scales.data(), ScalarType::Float);
    auto qz_t = make_tensor_ptr({B, Hq,  S,   1}, s->q_zp.data(),     ScalarType::Char);
    auto kz_t = make_tensor_ptr({B, Hkv, KVL, 1}, s->k_zp.data(),     ScalarType::Char);
    auto vz_t = make_tensor_ptr({B, Hkv, KVL, 1}, s->v_zp.data(),     ScalarType::Char);
    auto o_t  = make_tensor_ptr({B, Hq,  S,   D}, s->out.data(), ScalarType::Float);

    KernelRuntimeContext ctx;
    et::aten::optional<Tensor> no_mask;
    et::aten::optional<double> scale = static_cast<double>(1.0 / std::sqrt(static_cast<float>(d.head_dim)));
    et::aten::optional<Tensor> qz(*qz_t), qs(*qs_t);
    et::aten::optional<Tensor> kz(*kz_t), ks(*ks_t);
    et::aten::optional<Tensor> vz(*vz_t), vs(*vs_t);

    ::torch::executor::native::custom_quantized_sdpa_out(
        ctx, *q_t, *k_t, *v_t,
        /*start_pos*/ static_cast<int64_t>(s->kv_seq_len - 1),
        no_mask,
        /*dropout_p*/ 0.0,
        /*is_causal*/ true,
        scale,
        qz, qs,
        kz, ks,
        vz, vs,
        /*is_seq_at_dim_1*/ true,    // BNSH; header param name vs impl body
                                     // disagree (see struct comment above).
        *o_t);

    if (output)
        std::memcpy(output, s->out.data(), s->out.size() * sizeof(float));
}

void cleanup(void* state) { delete static_cast<State*>(state); }

} // namespace et_attn_decode_q8pc

static int reg_attn = [] {
    bench::register_attn_backend({
        "executorch_sdpa_prefill_fp32", "executorch", bench::AttnMode::PREFILL,
        et_attn_prefill::prepare, et_attn_prefill::run, et_attn_prefill::cleanup
    });
    bench::register_attn_backend({
        "executorch_qsdpa_decode_int8pc", "executorch", bench::AttnMode::DECODE,
        et_attn_decode_q8pc::prepare, et_attn_decode_q8pc::run, et_attn_decode_q8pc::cleanup
    });
    return 0;
}();

} // namespace

#else  // !WITH_EXECUTORCH_LLM

namespace { [[maybe_unused]] static int reg_attn = []{ return 0; }(); }

#endif // WITH_EXECUTORCH_LLM
