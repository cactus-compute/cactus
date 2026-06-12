#include "bench_driver.h"

#ifdef WITH_LITERT

#include <arm_neon.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "tflite/kernels/internal/optimized/neon_tensor_utils.h"
#include "ruy/ruy.h"

namespace {

static constexpr int kRuyGemvThreads = 1;
static constexpr int kRuyGemmThreads = 4;

struct Int8Weights {
    size_t K = 0, N = 0;
    std::vector<int8_t> int8_rowmajor;
    std::vector<float> weight_scales;
    std::vector<float> neon_output;
    std::vector<int32_t> ruy_output;
};

// Shared ruy::Context — without this, each matrix creates its own Context
// with thread pool, leading to 4k+ worker threads at small dims.
static ruy::Context* get_ruy_context(int max_threads) {
    static std::unique_ptr<ruy::Context> ctx;
    static int last_threads = -1;
    if (!ctx || max_threads != last_threads) {
        ctx = std::make_unique<ruy::Context>();
        ctx->set_max_num_threads(max_threads);
        last_threads = max_threads;
    }
    return ctx.get();
}

void* int8_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new Int8Weights();
    w->K = K;
    w->N = N;
    w->int8_rowmajor.resize(N * K);
    w->weight_scales.resize(N);
    bench::quantize_rows_int8(fp32, w->int8_rowmajor.data(), w->weight_scales.data(), N, K);
    return w;
}

void int8_cleanup(void* weights, void*) {
    delete static_cast<Int8Weights*>(weights);
}

void neon_run_kernel(size_t M, size_t /*K*/, size_t /*N*/,
                     void* weights, void*,
                     const int8_t* act_int8, const float* act_scales,
                     float* output, float*) {
    auto* w = static_cast<Int8Weights*>(weights);
    w->neon_output.resize(M * w->N);
    std::memset(w->neon_output.data(), 0, M * w->N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        w->int8_rowmajor.data(), static_cast<int>(w->N), static_cast<int>(w->K),
        act_int8, act_scales, static_cast<int>(M), w->neon_output.data());

    const float* ws = w->weight_scales.data();
    for (size_t m = 0; m < M; m++) {
        float* row = w->neon_output.data() + m * w->N;
        size_t n = 0;
        for (; n + 4 <= w->N; n += 4) {
            float32x4_t v = vld1q_f32(row + n);
            float32x4_t s = vld1q_f32(ws + n);
            vst1q_f32(row + n, vmulq_f32(v, s));
        }
        for (; n < w->N; n++) row[n] *= ws[n];
    }

    if (output) std::memcpy(output, w->neon_output.data(), M * w->N * sizeof(float));
}

static void ruy_run_kernel_impl(size_t M, Int8Weights* w,
                                const int8_t* act_int8, ruy::Context* ctx) {
    w->ruy_output.resize(M * w->N);

    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(act_int8); lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(w->K), static_cast<int>(w->N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(w->int8_rowmajor.data()); rhs.set_zero_point(0);
    rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> dst;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->N),
                          ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(w->ruy_output.data());

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(lhs, rhs, mul_params, ctx, &dst);
}

void ruy_run_kernel(size_t M, size_t /*K*/, size_t /*N*/,
                    void* weights, void*,
                    const int8_t* act_int8, const float* act_scales,
                    float* output, float*) {
    auto* w = static_cast<Int8Weights*>(weights);
    int default_threads = (M <= 1) ? kRuyGemvThreads : kRuyGemmThreads;
    int threads = bench::get_effective_threads(default_threads);
    ruy_run_kernel_impl(M, w, act_int8, get_ruy_context(threads));

    w->neon_output.resize(M * w->N);
    const float* ws = w->weight_scales.data();
    for (size_t m = 0; m < M; m++) {
        float32x4_t as = vdupq_n_f32(act_scales[m]);
        const int32_t* src = w->ruy_output.data() + m * w->N;
        float* dst = w->neon_output.data() + m * w->N;
        size_t n = 0;
        for (; n + 4 <= w->N; n += 4) {
            float32x4_t v = vcvtq_f32_s32(vld1q_s32(src + n));
            float32x4_t s = vld1q_f32(ws + n);
            vst1q_f32(dst + n, vmulq_f32(vmulq_f32(v, as), s));
        }
        for (; n < w->N; n++)
            dst[n] = static_cast<float>(src[n]) * act_scales[m] * ws[n];
    }

    if (output) std::memcpy(output, w->neon_output.data(), M * w->N * sizeof(float));
}

// LiteRT/TFLite has no fused attention op, so attention here is hand-composed
// from its INT8 matmul kernels (Q·Kᵀ → softmax → @V). Cactus and onnxrt are
// compared against fused kernels; this is the best LiteRT-primitive baseline.

namespace attn {

static void ruy_matmul_int32(const int8_t* A, size_t M, size_t K,
                              const int8_t* B_rowmajor, size_t N,
                              int32_t* dst, ruy::Context* ctx,
                              bool cache_rhs = false) {
    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(A); lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(K), static_cast<int>(N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(B_rowmajor); rhs.set_zero_point(0);
    if (cache_rhs) rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> out;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(N),
                          ruy::Order::kRowMajor, out.mutable_layout());
    out.set_data(dst);

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(lhs, rhs, mul_params, ctx, &out);
}

static void dequant_and_scale(const int32_t* int32_out,
                               const float* row_scales, const float* col_scales,
                               float extra_scale, float* dst,
                               size_t M, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        float rs = row_scales[m] * extra_scale;
        float32x4_t rs_v = vdupq_n_f32(rs);
        const int32_t* src = int32_out + m * N;
        float* out = dst + m * N;
        size_t n = 0;
        for (; n + 4 <= N; n += 4) {
            float32x4_t v = vcvtq_f32_s32(vld1q_s32(src + n));
            float32x4_t cs = vld1q_f32(col_scales + n);
            vst1q_f32(out + n, vmulq_f32(vmulq_f32(v, rs_v), cs));
        }
        for (; n < N; n++)
            out[n] = static_cast<float>(src[n]) * rs * col_scales[n];
    }
}

static void neon_mbvma_dequant(const int8_t* weight, size_t N, size_t K,
                                const float* weight_scales,
                                const int8_t* act_int8, const float* act_scales,
                                float* dst) {
    std::memset(dst, 0, N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weight, static_cast<int>(N), static_cast<int>(K),
        act_int8, act_scales, 1, dst);
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        float32x4_t v = vld1q_f32(dst + n);
        float32x4_t s = vld1q_f32(weight_scales + n);
        vst1q_f32(dst + n, vmulq_f32(v, s));
    }
    for (; n < N; n++) dst[n] *= weight_scales[n];
}

static void softmax_causal(float* scores, size_t seq_len, size_t kv_seq_len) {
    size_t offset = kv_seq_len - seq_len;
    for (size_t sq = 0; sq < seq_len; ++sq) {
        float* row = scores + sq * kv_seq_len;
        if (seq_len > 1)
            for (size_t sk = sq + offset + 1; sk < kv_seq_len; ++sk)
                row[sk] = -1e30f;

        float max_val = row[0];
        for (size_t sk = 1; sk < kv_seq_len; ++sk)
            if (row[sk] > max_val) max_val = row[sk];

        float sum = 0.0f;
        for (size_t sk = 0; sk < kv_seq_len; ++sk) {
            row[sk] = std::exp(row[sk] - max_val);
            sum += row[sk];
        }
        float inv_sum = 1.0f / sum;
        for (size_t sk = 0; sk < kv_seq_len; ++sk) row[sk] *= inv_sum;
    }
}

struct KVHead {
    std::vector<int8_t> k_int8;
    std::vector<float>  k_scales;
    std::vector<int8_t> vt_int8;
    std::vector<float>  vt_scales;
};

struct AttnState {
    bench::AttnDims dims;
    size_t seq_len = 0, kv_seq_len = 0;
    float scale = 0.0f;
    std::vector<KVHead> kv_heads;
    std::vector<float> fp32_q;
    int ruy_threads = 1;
    std::vector<int8_t>  q_int8_buf;
    std::vector<float>   q_scales_buf;
    std::vector<int32_t> int32_buf;
    std::vector<float>   scores_buf;
    std::vector<int8_t>  scores_int8_buf;
    std::vector<float>   scores_scales_buf;
    std::vector<float>   sv_buf;
};

static void* attn_prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                           const float* fp32_q, const float* fp32_k, const float* fp32_v,
                           bench::AttnMode mode) {
    auto* s = new AttnState();
    s->dims = dims;
    s->seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    s->kv_seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = dims.head_dim;
    s->fp32_q.assign(fp32_q, fp32_q + dims.num_q_heads * sl * hd);

    s->kv_heads.resize(dims.num_kv_heads);
    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        auto& head = s->kv_heads[h];
        const float* k_head = fp32_k + h * kvl * hd;
        const float* v_head = fp32_v + h * kvl * hd;

        head.k_int8.resize(kvl * hd);
        head.k_scales.resize(kvl);
        bench::quantize_rows_int8(k_head, head.k_int8.data(), head.k_scales.data(), kvl, hd);

        std::vector<float> vt(hd * kvl);
        bench::transpose_2d(v_head, vt.data(), kvl, hd);
        head.vt_int8.resize(hd * kvl);
        head.vt_scales.resize(hd);
        bench::quantize_rows_int8(vt.data(), head.vt_int8.data(), head.vt_scales.data(), hd, kvl);
    }

    s->q_int8_buf.resize(sl * hd);
    s->q_scales_buf.resize(sl);
    s->int32_buf.resize(std::max(sl * kvl, sl * hd));
    s->scores_buf.resize(sl * kvl);
    s->scores_int8_buf.resize(sl * kvl);
    s->scores_scales_buf.resize(sl);
    s->sv_buf.resize(sl * hd);

    s->ruy_threads = bench::get_effective_threads(sl > 1 ? kRuyGemmThreads : kRuyGemvThreads);
    return s;
}

static void run_ruy(void* state, float* output) {
    auto* s = static_cast<AttnState*>(state);
    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = s->dims.head_dim;
    size_t gqa_ratio = s->dims.num_q_heads / s->dims.num_kv_heads;
    ruy::Context* ctx = get_ruy_context(s->ruy_threads);

    for (size_t qh = 0; qh < s->dims.num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;
        const float* q_head = s->fp32_q.data() + qh * sl * hd;
        auto& kv = s->kv_heads[kvh];

        bench::quantize_rows_int8(q_head, s->q_int8_buf.data(), s->q_scales_buf.data(), sl, hd);

        ruy_matmul_int32(s->q_int8_buf.data(), sl, hd, kv.k_int8.data(), kvl,
                         s->int32_buf.data(), ctx, true);
        dequant_and_scale(s->int32_buf.data(), s->q_scales_buf.data(), kv.k_scales.data(),
                          s->scale, s->scores_buf.data(), sl, kvl);

        softmax_causal(s->scores_buf.data(), sl, kvl);

        bench::quantize_rows_int8(s->scores_buf.data(), s->scores_int8_buf.data(),
                                   s->scores_scales_buf.data(), sl, kvl);

        ruy_matmul_int32(s->scores_int8_buf.data(), sl, kvl, kv.vt_int8.data(), hd,
                         s->int32_buf.data(), ctx, true);
        dequant_and_scale(s->int32_buf.data(), s->scores_scales_buf.data(), kv.vt_scales.data(),
                          1.0f, s->sv_buf.data(), sl, hd);

        if (output)
            std::memcpy(output + qh * sl * hd, s->sv_buf.data(), sl * hd * sizeof(float));
    }
}

// TFLite's MatrixBatchVectorMultiplyAccumulate is vector-matrix only, so the
// NEON path is decode-only (query must have 1 row).
static void run_neon(void* state, float* output) {
    auto* s = static_cast<AttnState*>(state);
    size_t kvl = s->kv_seq_len, hd = s->dims.head_dim;
    size_t gqa_ratio = s->dims.num_q_heads / s->dims.num_kv_heads;

    for (size_t qh = 0; qh < s->dims.num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;
        const float* q_head = s->fp32_q.data() + qh * hd;
        auto& kv = s->kv_heads[kvh];

        bench::quantize_rows_int8(q_head, s->q_int8_buf.data(), s->q_scales_buf.data(), 1, hd);
        s->q_scales_buf[0] *= s->scale;

        neon_mbvma_dequant(kv.k_int8.data(), kvl, hd, kv.k_scales.data(),
                           s->q_int8_buf.data(), s->q_scales_buf.data(),
                           s->scores_buf.data());

        softmax_causal(s->scores_buf.data(), 1, kvl);

        bench::quantize_rows_int8(s->scores_buf.data(), s->scores_int8_buf.data(),
                                   s->scores_scales_buf.data(), 1, kvl);

        neon_mbvma_dequant(kv.vt_int8.data(), hd, kvl, kv.vt_scales.data(),
                           s->scores_int8_buf.data(), s->scores_scales_buf.data(),
                           s->sv_buf.data());

        if (output)
            std::memcpy(output + qh * hd, s->sv_buf.data(), hd * sizeof(float));
    }
}

static void cleanup(void* state) { delete static_cast<AttnState*>(state); }

void* prefill(const bench::AttnDims& d, size_t sl, size_t cl,
              const float* q, const float* k, const float* v) {
    return attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* decode(const bench::AttnDims& d, size_t sl, size_t cl,
             const float* q, const float* k, const float* v) {
    return attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}

} // namespace attn

static int reg = [] {
    bench::register_matmul_backend({
        "litert_neon", "litert",
        int8_prepare, nullptr, neon_run_kernel, int8_cleanup
    });
    bench::register_matmul_backend({
        "litert_ruy", "litert",
        int8_prepare, nullptr, ruy_run_kernel, int8_cleanup
    });

    using P = bench::AttnMode;
    bench::register_attn_backend({
        "litert_ruy_prefill", "litert", P::PREFILL,
        attn::prefill, attn::run_ruy, attn::cleanup
    });
    bench::register_attn_backend({
        "litert_ruy_decode", "litert", P::DECODE,
        attn::decode,  attn::run_ruy, attn::cleanup
    });
    bench::register_attn_backend({
        "litert_neon_decode", "litert", P::DECODE,
        attn::decode,  attn::run_neon, attn::cleanup
    });
    return 0;
}();

} // namespace

#else  // !WITH_LITERT
namespace { [[maybe_unused]] static int reg = []{ return 0; }(); }
#endif
