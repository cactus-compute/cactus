#include "bench_driver.h"

#ifdef WITH_GGML

#include "ggml.h"
#include "ggml-cpu.h"

#include "../../cactus-kernels/src/threading.h"

#include <cstring>
#include <cmath>
#include <thread>
#include <vector>

namespace {

namespace matmul {

struct Weights {
    size_t K = 0, N = 0;
    ggml_type type = GGML_TYPE_Q8_0;
    std::vector<uint8_t> quantized;
    size_t row_stride = 0;
    std::vector<float> output;
};

void* prepare_quantized(const float* fp32, size_t N, size_t K, ggml_type type) {
    auto* w = new Weights();
    w->K = K; w->N = N; w->type = type;
    w->row_stride = ggml_row_size(w->type, K);
    w->quantized.resize(w->row_stride * N);

    const auto* cpu_traits = ggml_get_type_traits_cpu(w->type);
    auto from_float = cpu_traits->from_float;
    if (!from_float) from_float = ggml_get_type_traits(w->type)->from_float_ref;

    for (size_t n = 0; n < N; n++)
        from_float(fp32 + n * K, w->quantized.data() + n * w->row_stride, static_cast<int64_t>(K));
    return w;
}

void* prepare_q4_0(const float* fp32, size_t N, size_t K) {
    return prepare_quantized(fp32, N, K, GGML_TYPE_Q4_0);
}

struct Activations {
    std::vector<uint8_t> q8_input;
    size_t q8_row_stride = 0;
};

void* prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new Activations();
    a->q8_row_stride = ggml_row_size(GGML_TYPE_Q8_0, K);
    a->q8_input.resize(a->q8_row_stride * M);
    auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (size_t m = 0; m < M; m++)
        quantize(fp32 + m * K, a->q8_input.data() + m * a->q8_row_stride, static_cast<int64_t>(K));
    return a;
}

static void gemv_slice(ggml_vec_dot_t vec_dot, int64_t nrows,
                        const uint8_t* wdata, size_t w_stride,
                        const uint8_t* q8_input, size_t q8_row_stride,
                        float* output, size_t M, size_t K, size_t N,
                        size_t n_begin, size_t n_end) {
    const int Kint = static_cast<int>(K);
    (void)N;
    for (size_t m = 0; m < M; m++) {
        const uint8_t* act_row = q8_input + m * q8_row_stride;
        float* out_row = output + m * N;
        bool can_pair_m = (nrows >= 2 && m + 1 < M);

        size_t n = n_begin;
        if (can_pair_m) {
            float* out_row_next = output + (m + 1) * N;
            for (; n + 1 < n_end; n += 2) {
                float tmp[4];
                vec_dot(Kint, tmp, 2,
                        wdata + n * w_stride, w_stride,
                        act_row, q8_row_stride, 2);
                out_row[n]          = tmp[0];
                out_row[n + 1]      = tmp[1];
                out_row_next[n]     = tmp[2];
                out_row_next[n + 1] = tmp[3];
            }
            for (; n < n_end; n++) {
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0, act_row, 0, 1);
                vec_dot(Kint, out_row_next + n, 0,
                        wdata + n * w_stride, 0, act_row + q8_row_stride, 0, 1);
            }
            m++;
        } else {
            for (; n < n_end; n++)
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0, act_row, 0, 1);
        }
    }
}

void run_kernel(size_t M, size_t K, size_t N,
                void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    (void)K; (void)N;
    auto* w = static_cast<Weights*>(weights);
    auto* a = static_cast<Activations*>(activations);
    w->output.resize(M * w->N);

    const auto* cpu_traits = ggml_get_type_traits_cpu(w->type);
    size_t target_threads = static_cast<size_t>(
        bench::get_effective_threads(static_cast<int>(std::thread::hardware_concurrency())));
    target_threads = std::max<size_t>(1, target_threads);

    // Serial path skips threadpool wakeup (saves tens of μs per call at tiny N).
    if (target_threads == 1 || w->N < 64) {
        gemv_slice(cpu_traits->vec_dot, cpu_traits->nrows,
                   w->quantized.data(), w->row_stride,
                   a->q8_input.data(), a->q8_row_stride,
                   w->output.data(), M, w->K, w->N, 0, w->N);
    } else {
        size_t chunk = std::max<size_t>(16, w->N / target_threads);
        CactusThreading::ParallelConfig par_cfg{chunk, 16};
        CactusThreading::parallel_for(w->N, par_cfg,
            [&](size_t n_begin, size_t n_end) {
                gemv_slice(cpu_traits->vec_dot, cpu_traits->nrows,
                           w->quantized.data(), w->row_stride,
                           a->q8_input.data(), a->q8_row_stride,
                           w->output.data(), M, w->K, w->N, n_begin, n_end);
            });
    }

    if (output) std::memcpy(output, w->output.data(), M * w->N * sizeof(float));
}

void cleanup(void* weights, void* activations) {
    delete static_cast<Weights*>(weights);
    if (activations) delete static_cast<Activations*>(activations);
}

} // namespace matmul

namespace attn {

struct State {
    bench::AttnDims dims;
    size_t seq_len;
    size_t output_count;
    bool permuted_output;
    ggml_context* ctx;
    ggml_cgraph*  graph;
    ggml_threadpool* threadpool;
    ggml_cplan plan;
    std::vector<uint8_t> work_buf;
    ggml_tensor* result;
};

static void fill_causal_mask(ggml_fp16_t* data, size_t kv_len, size_t seq_len) {
    size_t offset = kv_len - seq_len;
    for (size_t sq = 0; sq < seq_len; ++sq)
        for (size_t kv = 0; kv < kv_len; ++kv)
            data[sq * kv_len + kv] = (kv <= sq + offset)
                ? ggml_fp32_to_fp16(0.0f)
                : ggml_fp32_to_fp16(-INFINITY);
}

static void quantize_tensor(const float* fp32, void* dst, ggml_type type, size_t n) {
    if (type == GGML_TYPE_F32) { std::memcpy(dst, fp32, n * sizeof(float)); return; }
    auto from_float = ggml_get_type_traits_cpu(type)->from_float;
    if (!from_float) from_float = ggml_get_type_traits(type)->from_float_ref;
    from_float(fp32, (uint8_t*)dst, static_cast<int64_t>(n));
}

struct SetupResult {
    ggml_context* ctx;
    ggml_tensor* q_t;
    ggml_tensor* k_t;
    ggml_tensor* mask_t;
    size_t sl, kvl, hd, qh, kvh, kv_rows;
    float scale;
};

static SetupResult setup_common(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                                 const float* fp32_q, const float* fp32_k,
                                 bench::AttnMode mode, ggml_type kv_type, size_t extra_bytes) {
    SetupResult r;
    r.sl  = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    r.kvl = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    r.hd  = dims.head_dim;
    r.qh  = dims.num_q_heads;
    r.kvh = dims.num_kv_heads;
    r.kv_rows = r.kvh * r.kvl;
    r.scale = 1.0f / std::sqrt(static_cast<float>(r.hd));

    size_t kv_row_bytes = ggml_row_size(kv_type, r.hd);
    size_t data_bytes = r.qh * r.sl * r.hd * sizeof(float)
                      + 2 * r.kv_rows * kv_row_bytes
                      + r.kvl * r.sl * sizeof(ggml_fp16_t)
                      + r.qh * r.sl * r.hd * sizeof(float)
                      + extra_bytes;
    size_t ctx_size = data_bytes + ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 64 * 1024 * 1024;

    r.ctx = ggml_init({ ctx_size, nullptr, false });

    r.q_t = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, (int64_t)r.hd, (int64_t)r.sl, (int64_t)r.qh);
    r.k_t = ggml_new_tensor_3d(r.ctx, kv_type, (int64_t)r.hd, (int64_t)r.kvl, (int64_t)r.kvh);
    ggml_set_name(r.q_t, "Q"); ggml_set_input(r.q_t);
    ggml_set_name(r.k_t, "K"); ggml_set_input(r.k_t);

    std::memcpy(r.q_t->data, fp32_q, r.qh * r.sl * r.hd * sizeof(float));
    quantize_tensor(fp32_k, r.k_t->data, kv_type, r.kv_rows * r.hd);

    r.mask_t = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, (int64_t)r.kvl, (int64_t)r.sl);
    ggml_set_name(r.mask_t, "mask"); ggml_set_input(r.mask_t);
    fill_causal_mask((ggml_fp16_t*)r.mask_t->data, r.kvl, r.sl);
    return r;
}

// Shared ggml threadpool — without this, each state creates its own pool with
// `hardware_concurrency` threads, so at small cache (large NS) hundreds of
// threads sit on condition variables and the per-call cost balloons.
static ggml_threadpool* get_ggml_threadpool() {
    static ggml_threadpool* tp = nullptr;
    static int last_threads = -1;
    int target = bench::get_effective_threads(
        static_cast<int>(std::thread::hardware_concurrency()));
    if (target < 1) target = 1;
    if (tp == nullptr || target != last_threads) {
        if (tp) ggml_threadpool_free(tp);
        auto params = ggml_threadpool_params_default(target);
        tp = ggml_threadpool_new(&params);
        last_threads = target;
    }
    return tp;
}
static int get_ggml_threads() {
    return std::max(1, bench::get_effective_threads(
        static_cast<int>(std::thread::hardware_concurrency())));
}

static State* finalize(const bench::AttnDims& dims, SetupResult& r,
                        ggml_tensor* result, bool permuted) {
    auto* gf = ggml_new_graph_custom(r.ctx, 4096, false);
    ggml_build_forward_expand(gf, result);

    int n_threads = get_ggml_threads();
    auto* tp = get_ggml_threadpool();
    auto cplan = ggml_graph_plan(gf, n_threads, tp);
    std::vector<uint8_t> work_buf(cplan.work_size);
    cplan.work_data = work_buf.data();

    auto* s = new State{ dims, r.sl, r.qh * r.sl * r.hd, permuted,
                          r.ctx, gf, tp, cplan, std::move(work_buf), result };
    s->plan.work_data = s->work_buf.data();
    return s;
}

void* prepare_flash(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                    const float* fp32_q, const float* fp32_k, const float* fp32_v,
                    bench::AttnMode mode, ggml_type kv_type) {
    auto r = setup_common(dims, seq_len, cache_len, fp32_q, fp32_k, mode, kv_type, 0);

    auto* v_t = ggml_new_tensor_3d(r.ctx, kv_type, (int64_t)r.hd, (int64_t)r.kvl, (int64_t)r.kvh);
    ggml_set_name(v_t, "V"); ggml_set_input(v_t);
    quantize_tensor(fp32_v, v_t->data, kv_type, r.kv_rows * r.hd);

    auto* result = ggml_flash_attn_ext(r.ctx, r.q_t, r.k_t, v_t, r.mask_t, r.scale, 0.0f, 0.0f);
    ggml_set_output(result);
    return finalize(dims, r, result, true);
}

void* prepare_matmul(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                     const float* fp32_q, const float* fp32_k, const float* fp32_v,
                     bench::AttnMode mode, ggml_type kv_type) {
    size_t sl  = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    size_t kvl = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    size_t extra = dims.num_q_heads * sl * kvl * sizeof(float) * 2;
    auto r = setup_common(dims, seq_len, cache_len, fp32_q, fp32_k, mode, kv_type, extra);

    auto* v_t = ggml_new_tensor_3d(r.ctx, kv_type, (int64_t)r.kvl, (int64_t)r.hd, (int64_t)r.kvh);
    ggml_set_name(v_t, "V"); ggml_set_input(v_t);
    std::vector<float> v_transposed(r.kvh * r.kvl * r.hd);
    for (size_t h = 0; h < r.kvh; ++h)
        bench::transpose_2d(fp32_v + h * r.kvl * r.hd,
                             v_transposed.data() + h * r.hd * r.kvl,
                             r.kvl, r.hd);
    quantize_tensor(v_transposed.data(), v_t->data, kv_type, r.kv_rows * r.hd);

    auto* kq = ggml_mul_mat(r.ctx, r.k_t, r.q_t);
    kq = ggml_soft_max_ext(r.ctx, kq, r.mask_t, r.scale, 0.0f);
    auto* kqv = ggml_mul_mat(r.ctx, v_t, kq);
    ggml_set_output(kqv);
    return finalize(dims, r, kqv, false);
}

#define GGML_ATTN_VARIANT(name, prep_fn, mode, type) \
    void* name(const bench::AttnDims& d, size_t sl, size_t cl, const float* q, const float* k, const float* v) { \
        return prep_fn(d, sl, cl, q, k, v, bench::AttnMode::mode, type); \
    }
GGML_ATTN_VARIANT(fa_q8_prefill, prepare_flash,  PREFILL, GGML_TYPE_Q8_0)
GGML_ATTN_VARIANT(fa_q8_decode,  prepare_flash,  DECODE,  GGML_TYPE_Q8_0)
GGML_ATTN_VARIANT(mm_q8_prefill, prepare_matmul, PREFILL, GGML_TYPE_Q8_0)
GGML_ATTN_VARIANT(mm_q8_decode,  prepare_matmul, DECODE,  GGML_TYPE_Q8_0)
#undef GGML_ATTN_VARIANT

void run(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    ggml_graph_compute(s->graph, &s->plan);

    if (output) {
        const float* src = (const float*)s->result->data;
        size_t hd = s->dims.head_dim, qh = s->dims.num_q_heads, sl = s->seq_len;
        if (s->permuted_output) {
            for (size_t sq = 0; sq < sl; ++sq)
                for (size_t h = 0; h < qh; ++h)
                    std::memcpy(output + h * sl * hd + sq * hd,
                                src + (sq * qh + h) * hd, hd * sizeof(float));
        } else {
            std::memcpy(output, src, s->output_count * sizeof(float));
        }
    }
}

void cleanup(void* state) {
    auto* s = static_cast<State*>(state);
    // Threadpool is shared (see get_ggml_threadpool); do NOT free here.
    ggml_free(s->ctx);
    delete s;
}

} // namespace attn

static int reg = [] {
    bench::register_matmul_backend({
        "ggml_q4_0", "ggml",
        matmul::prepare_q4_0, matmul::prepare_act, matmul::run_kernel, matmul::cleanup
    });
    bench::register_attn_backend({
        "ggml_fa_q8_prefill", "ggml", bench::AttnMode::PREFILL,
        attn::fa_q8_prefill, attn::run, attn::cleanup
    });
    bench::register_attn_backend({
        "ggml_fa_q8_decode", "ggml", bench::AttnMode::DECODE,
        attn::fa_q8_decode, attn::run, attn::cleanup
    });
    return 0;
}();

} // namespace

#else
namespace { [[maybe_unused]] static int reg = []{ return 0; }(); }
#endif
