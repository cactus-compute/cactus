#include "bench_driver.h"

#include "../../cactus-kernels/cactus_kernels.h"
#include "../../cactus-kernels/src/threading.h"

#include <arm_neon.h>
#include <cmath>
#include <cstring>

namespace {

namespace matmul {

struct Weights {
    size_t K = 0, N = 0;
    std::vector<int8_t> int8_weights;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
};

void* prepare_int8(const float* fp32, size_t N, size_t K) {
    auto* w = new Weights();
    w->K = K;
    w->N = N;

    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;

    bench::quantize_int8_per_group(src, N, K, rowmajor, raw_scales);
    w->int8_weights = bench::interleave_weights_nk4(rowmajor, N, K);
    w->scales = bench::interleave_scales_n4(raw_scales, N, K / bench::kGroupSize);
    return w;
}

void run_kernel(size_t M, size_t K, size_t N,
                void* weights, void*,
                const int8_t* act_int8, const float* act_scales,
                float* output, float*) {
    auto* w = static_cast<Weights*>(weights);
    w->output_buf.assign(M * w->N, __fp16(0.0f));

    int thr = bench::get_thread_override();
    if (thr > 0) CactusThreading::set_gemm_threads(static_cast<size_t>(thr));

    if (M == 1) {
        cactus_gemv_int8(act_int8, act_scales[0],
                         w->int8_weights.data(), w->scales.data(),
                         w->output_buf.data(), K, N, bench::kGroupSize);
    } else {
        cactus_matmul_int8(act_int8, act_scales,
                           w->int8_weights.data(), w->scales.data(),
                           w->output_buf.data(), M, K, N, bench::kGroupSize);
    }

    if (thr > 0) CactusThreading::reset_gemm_threads();

    if (output) {
        size_t count = M * w->N;
        bench::fp16_to_fp32(w->output_buf.data(), output, count);
    }
}

void cleanup(void* weights, void*) {
    delete static_cast<Weights*>(weights);
}

} // namespace matmul

namespace attn {

struct State {
    bench::AttnDims dims;
    bench::AttnMode mode;
    size_t seq_len = 0;
    size_t cache_len = 0;
    float scale = 0.0f;
    std::vector<__fp16> q, k, v, output;
    std::vector<__fp16> k_new, v_new;
    std::vector<int8_t> k_cached, v_cached;
    std::vector<float> k_scales, v_scales;
};

// Driver layout [head, seq, head_dim] → cactus prefill layout [seq, head, head_dim] (batch=1).
static void transpose_seq_head(const float* src, __fp16* dst,
                                size_t heads, size_t seq, size_t head_dim) {
    for (size_t s = 0; s < seq; ++s)
        for (size_t h = 0; h < heads; ++h) {
            const float* in = src + h * seq * head_dim + s * head_dim;
            __fp16* out = dst + s * heads * head_dim + h * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = static_cast<__fp16>(in[d]);
        }
}

static void transpose_back(const __fp16* src, float* dst,
                            size_t heads, size_t seq, size_t head_dim) {
    for (size_t h = 0; h < heads; ++h)
        for (size_t s = 0; s < seq; ++s) {
            const __fp16* in = src + s * heads * head_dim + h * head_dim;
            float* out = dst + h * seq * head_dim + s * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = static_cast<float>(in[d]);
        }
}

static void* prepare_impl(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                           const float* fp32_q, const float* fp32_k, const float* fp32_v,
                           bench::AttnMode mode) {
    auto* s = new State();
    s->dims = dims;
    s->mode = mode;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    if (mode == bench::AttnMode::PREFILL) {
        s->seq_len = seq_len;
        size_t q_count = dims.num_q_heads * seq_len * dims.head_dim;
        size_t kv_count = dims.num_kv_heads * seq_len * dims.head_dim;

        s->q.resize(q_count);
        s->k.resize(kv_count);
        s->v.resize(kv_count);
        s->output.resize(q_count);

        transpose_seq_head(fp32_q, s->q.data(), dims.num_q_heads,  seq_len, dims.head_dim);
        transpose_seq_head(fp32_k, s->k.data(), dims.num_kv_heads, seq_len, dims.head_dim);
        transpose_seq_head(fp32_v, s->v.data(), dims.num_kv_heads, seq_len, dims.head_dim);
    } else {
        s->seq_len = 1;
        s->cache_len = cache_len;
        size_t q_count = dims.num_q_heads * dims.head_dim;
        size_t kv_per_token = dims.num_kv_heads * dims.head_dim;
        size_t kv_seq_len = cache_len + 1;
        size_t kv_total = dims.num_kv_heads * kv_seq_len * dims.head_dim;

        s->q.resize(q_count);
        bench::fp32_to_fp16(fp32_q, s->q.data(), q_count);

        s->k_new.resize(kv_per_token);
        s->v_new.resize(kv_per_token);
        s->output.resize(q_count);

        std::vector<__fp16> full_k(kv_total), full_v(kv_total);
        transpose_seq_head(fp32_k, full_k.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);
        transpose_seq_head(fp32_v, full_v.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);

        size_t cached_elements = cache_len * kv_per_token;
        s->k_cached.resize(cached_elements);
        s->v_cached.resize(cached_elements);

        size_t sc = kv_scales_count(cache_len, dims.num_kv_heads, dims.head_dim);
        s->k_scales.resize(sc);
        s->v_scales.resize(sc);

        cactus_quantize_kv_fp16_to_int8(full_k.data(), s->k_cached.data(), s->k_scales.data(),
                                         cache_len, dims.num_kv_heads, dims.head_dim);
        cactus_quantize_kv_fp16_to_int8(full_v.data(), s->v_cached.data(), s->v_scales.data(),
                                         cache_len, dims.num_kv_heads, dims.head_dim);

        size_t new_offset = cached_elements;
        std::memcpy(s->k_new.data(), full_k.data() + new_offset, kv_per_token * sizeof(__fp16));
        std::memcpy(s->v_new.data(), full_v.data() + new_offset, kv_per_token * sizeof(__fp16));
    }
    return s;
}

void* prepare_prefill(const bench::AttnDims& d, size_t sl, size_t cl,
                      const float* q, const float* k, const float* v) {
    return prepare_impl(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* prepare_decode(const bench::AttnDims& d, size_t sl, size_t cl,
                     const float* q, const float* k, const float* v) {
    return prepare_impl(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}

void run(void* state, float* output) {
    auto* s = static_cast<State*>(state);

    if (s->mode == bench::AttnMode::PREFILL) {
        cactus_attention_f16(s->q.data(), s->k.data(), s->v.data(), s->output.data(),
                              1, s->seq_len, s->seq_len,
                              s->dims.num_q_heads, s->dims.num_kv_heads,
                              s->dims.head_dim, s->scale, nullptr, 0, 0, true);
    } else {
        cactus_attention_hybrid_int8_fp16(
            s->q.data(),
            s->k_cached.data(), s->v_cached.data(),
            s->k_scales.data(), s->v_scales.data(),
            s->k_new.data(), s->v_new.data(),
            s->output.data(),
            1, 1, s->cache_len, 1,
            s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
            s->scale, s->cache_len, true, 0, bench::kGroupSize);
    }

    if (output)
        transpose_back(s->output.data(), output,
                       s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

void cleanup(void* state) { delete static_cast<State*>(state); }

} // namespace attn

static int reg = []{
    bench::register_matmul_backend({
        "cactus_int8", "cactus",
        matmul::prepare_int8, nullptr, matmul::run_kernel, matmul::cleanup
    });
    bench::register_attn_backend({
        "cactus_prefill", "cactus", bench::AttnMode::PREFILL,
        attn::prepare_prefill, attn::run, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode", "cactus", bench::AttnMode::DECODE,
        attn::prepare_decode, attn::run, attn::cleanup
    });
    return 0;
}();

} // namespace
