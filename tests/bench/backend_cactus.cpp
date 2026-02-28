#include "bench_driver.h"

#include <arm_neon.h>
#include <cmath>
#include <cstring>

namespace {

// ── Matmul ──────────────────────────────────────────────────────────────────────

namespace matmul {

enum class Quant { INT8, INT4 };

struct Weights {
    size_t K, N;
    Quant quant;
    std::vector<int8_t> int8_weights;
    std::vector<uint8_t> int4_packed;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;

    void run(size_t M, const int8_t* act_int8, const float* act_scales, __fp16* out) const {
        if (quant == Quant::INT8)
            cactus_matmul_int8(act_int8, act_scales, int8_weights.data(), scales.data(),
                               out, M, K, N, bench::kGroupSize);
        else
            cactus_matmul_int4(act_int8, act_scales,
                               reinterpret_cast<const int8_t*>(int4_packed.data()),
                               scales.data(), out, M, K, N, bench::kGroupSize);
    }
};

static void* prepare_impl(const float* fp32, size_t N, size_t K, Quant quant) {
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;

    auto* w = new Weights();
    w->K = K;
    w->N = N;
    w->quant = quant;

    if (quant == Quant::INT8) {
        bench::quantize_int8_per_group(src, N, K, rowmajor, raw_scales);
        w->int8_weights = bench::interleave_weights_nk4(rowmajor, N, K);
    } else {
        bench::quantize_int4_per_group(src, N, K, rowmajor, raw_scales);
        w->int4_packed = bench::pack_int4_pairs(bench::interleave_weights_nk4(rowmajor, N, K));
    }
    w->scales = bench::interleave_scales_n4(raw_scales, N, K / bench::kGroupSize);
    return w;
}

void* prepare_int8(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, Quant::INT8); }
void* prepare_int4(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, Quant::INT4); }

void run_kernel(size_t M, void* weights, void*,
                const int8_t* act_int8, const float* act_scales,
                float* output, float*) {
    auto* w = static_cast<Weights*>(weights);
    w->output_buf.resize(M * w->N);
    int thr = bench::get_thread_override();
    if (thr > 0) CactusThreading::set_gemm_threads(static_cast<size_t>(thr));
    w->run(M, act_int8, act_scales, w->output_buf.data());
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

// ── Attention ───────────────────────────────────────────────────────────────────

namespace attn {

struct State {
    bench::AttnDims dims;
    bench::AttnMode mode;
    size_t seq_len;
    size_t cache_len;
    float scale;
    std::vector<__fp16> q, k, v, output;
    std::vector<__fp16> k_new, v_new;
    std::vector<int8_t> k_cached, v_cached;
    std::vector<float> k_scales, v_scales;
};

static void transpose_and_convert(const float* src, __fp16* dst,
                                   size_t heads, size_t seq, size_t head_dim) {
    for (size_t s = 0; s < seq; ++s)
        for (size_t h = 0; h < heads; ++h) {
            const float* in = src + h * seq * head_dim + s * head_dim;
            __fp16* out = dst + s * heads * head_dim + h * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = static_cast<__fp16>(in[d]);
        }
}

static void transpose_and_convert_back(const __fp16* src, float* dst,
                                        size_t heads, size_t seq, size_t head_dim) {
    for (size_t h = 0; h < heads; ++h)
        for (size_t s = 0; s < seq; ++s) {
            const __fp16* in = src + s * heads * head_dim + h * head_dim;
            float* out = dst + h * seq * head_dim + s * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = static_cast<float>(in[d]);
        }
}

void* prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
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

        transpose_and_convert(fp32_q, s->q.data(), dims.num_q_heads, seq_len, dims.head_dim);
        transpose_and_convert(fp32_k, s->k.data(), dims.num_kv_heads, seq_len, dims.head_dim);
        transpose_and_convert(fp32_v, s->v.data(), dims.num_kv_heads, seq_len, dims.head_dim);
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
        transpose_and_convert(fp32_k, full_k.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);
        transpose_and_convert(fp32_v, full_v.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);

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

void* prefill(const bench::AttnDims& d, size_t sl, size_t cl,
              const float* q, const float* k, const float* v) {
    return prepare(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* decode(const bench::AttnDims& d, size_t sl, size_t cl,
             const float* q, const float* k, const float* v) {
    return prepare(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
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
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

void cleanup(void* state) { delete static_cast<State*>(state); }

void* decode_transposed(const bench::AttnDims& dims, size_t, size_t cache_len,
                         const float* fp32_q, const float* fp32_k, const float* fp32_v) {
    auto* s = new State();
    s->dims = dims;
    s->mode = bench::AttnMode::DECODE;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
    s->seq_len = 1;
    s->cache_len = cache_len;

    size_t q_count = dims.num_q_heads * dims.head_dim;
    size_t kv_per_token = dims.num_kv_heads * dims.head_dim;
    size_t kv_seq_len = cache_len + 1;

    s->q.resize(q_count);
    bench::fp32_to_fp16(fp32_q, s->q.data(), q_count);

    s->k_new.resize(kv_per_token);
    s->v_new.resize(kv_per_token);
    s->output.resize(q_count);

    size_t kv_total = dims.num_kv_heads * kv_seq_len * dims.head_dim;
    std::vector<__fp16> full_k(kv_total), full_v(kv_total);
    bench::fp32_to_fp16(fp32_k, full_k.data(), kv_total);
    bench::fp32_to_fp16(fp32_v, full_v.data(), kv_total);

    size_t cached_per_head = cache_len * dims.head_dim;
    size_t cached_elements = dims.num_kv_heads * cached_per_head;
    std::vector<__fp16> cached_k(cached_elements), cached_v(cached_elements);
    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        std::memcpy(cached_k.data() + h * cached_per_head,
                    full_k.data() + h * kv_seq_len * dims.head_dim,
                    cached_per_head * sizeof(__fp16));
        std::memcpy(cached_v.data() + h * cached_per_head,
                    full_v.data() + h * kv_seq_len * dims.head_dim,
                    cached_per_head * sizeof(__fp16));
    }

    s->k_cached.resize(cached_elements);
    s->v_cached.resize(cached_elements);
    size_t sc = kv_scales_count(cache_len, dims.num_kv_heads, dims.head_dim);
    s->k_scales.resize(sc);
    s->v_scales.resize(sc);

    cactus_quantize_kv_fp16_to_int8(cached_k.data(), s->k_cached.data(), s->k_scales.data(),
                                     cache_len, dims.num_kv_heads, dims.head_dim);
    cactus_quantize_kv_fp16_to_int8(cached_v.data(), s->v_cached.data(), s->v_scales.data(),
                                     cache_len, dims.num_kv_heads, dims.head_dim);

    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        std::memcpy(s->k_new.data() + h * dims.head_dim,
                    full_k.data() + h * kv_seq_len * dims.head_dim + cache_len * dims.head_dim,
                    dims.head_dim * sizeof(__fp16));
        std::memcpy(s->v_new.data() + h * dims.head_dim,
                    full_v.data() + h * kv_seq_len * dims.head_dim + cache_len * dims.head_dim,
                    dims.head_dim * sizeof(__fp16));
    }
    return s;
}

void run_transposed(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    cactus_attention_hybrid_int8_fp16_transposed(
        s->q.data(),
        s->k_cached.data(), s->v_cached.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

void run_highthread(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    CactusThreading::set_attention_config(1, 4);
    cactus_attention_hybrid_int8_fp16(
        s->q.data(),
        s->k_cached.data(), s->v_cached.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    CactusThreading::reset_attention_config();
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

struct PackedState {
    bench::AttnDims dims;
    size_t cache_len;
    float scale;
    std::vector<__fp16> q, output;
    std::vector<__fp16> k_new, v_new;
    std::vector<int8_t> kv_packed;
    std::vector<float> k_scales, v_scales;
};

void* decode_packed(const bench::AttnDims& dims, size_t, size_t cache_len,
                     const float* fp32_q, const float* fp32_k, const float* fp32_v) {
    auto* s = new PackedState();
    s->dims = dims;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
    s->cache_len = cache_len;

    size_t q_count = dims.num_q_heads * dims.head_dim;
    size_t kv_per_token = dims.num_kv_heads * dims.head_dim;
    size_t kv_seq_len = cache_len + 1;

    s->q.resize(q_count);
    bench::fp32_to_fp16(fp32_q, s->q.data(), q_count);

    s->k_new.resize(kv_per_token);
    s->v_new.resize(kv_per_token);
    s->output.resize(q_count);

    size_t kv_total = dims.num_kv_heads * kv_seq_len * dims.head_dim;
    std::vector<__fp16> full_k(kv_total), full_v(kv_total);
    bench::fp32_to_fp16(fp32_k, full_k.data(), kv_total);
    bench::fp32_to_fp16(fp32_v, full_v.data(), kv_total);

    size_t cached_per_head = cache_len * dims.head_dim;
    size_t cached_elements = dims.num_kv_heads * cached_per_head;
    std::vector<__fp16> cached_k(cached_elements), cached_v(cached_elements);
    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        std::memcpy(cached_k.data() + h * cached_per_head,
                    full_k.data() + h * kv_seq_len * dims.head_dim,
                    cached_per_head * sizeof(__fp16));
        std::memcpy(cached_v.data() + h * cached_per_head,
                    full_v.data() + h * kv_seq_len * dims.head_dim,
                    cached_per_head * sizeof(__fp16));
    }

    s->kv_packed.resize(dims.num_kv_heads * cache_len * 2 * dims.head_dim);
    size_t sc = kv_scales_count(cache_len, dims.num_kv_heads, dims.head_dim);
    s->k_scales.resize(sc);
    s->v_scales.resize(sc);

    cactus_quantize_kv_packed_fp16_to_int8(
        cached_k.data(), cached_v.data(),
        s->kv_packed.data(),
        s->k_scales.data(), s->v_scales.data(),
        cache_len, dims.num_kv_heads, dims.head_dim);

    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        std::memcpy(s->k_new.data() + h * dims.head_dim,
                    full_k.data() + h * kv_seq_len * dims.head_dim + cache_len * dims.head_dim,
                    dims.head_dim * sizeof(__fp16));
        std::memcpy(s->v_new.data() + h * dims.head_dim,
                    full_v.data() + h * kv_seq_len * dims.head_dim + cache_len * dims.head_dim,
                    dims.head_dim * sizeof(__fp16));
    }
    return s;
}

void run_packed(void* state, float* output) {
    auto* s = static_cast<PackedState*>(state);
    cactus_attention_hybrid_int8_fp16_packed(
        s->q.data(),
        s->kv_packed.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, 1, s->dims.head_dim);
}

void cleanup_packed(void* state) { delete static_cast<PackedState*>(state); }

void* decode_stdpacked(const bench::AttnDims& dims, size_t, size_t cache_len,
                        const float* fp32_q, const float* fp32_k, const float* fp32_v) {
    auto* s = new PackedState();
    s->dims = dims;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
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
    transpose_and_convert(fp32_k, full_k.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);
    transpose_and_convert(fp32_v, full_v.data(), dims.num_kv_heads, kv_seq_len, dims.head_dim);

    size_t cached_elements = cache_len * kv_per_token;
    s->kv_packed.resize(cached_elements * 2);
    size_t sc = kv_scales_count(cache_len, dims.num_kv_heads, dims.head_dim);
    s->k_scales.resize(sc);
    s->v_scales.resize(sc);

    cactus_quantize_kv_packed_fp16_to_int8(
        full_k.data(), full_v.data(),
        s->kv_packed.data(),
        s->k_scales.data(), s->v_scales.data(),
        cache_len, dims.num_kv_heads, dims.head_dim);

    size_t new_offset = cached_elements;
    std::memcpy(s->k_new.data(), full_k.data() + new_offset, kv_per_token * sizeof(__fp16));
    std::memcpy(s->v_new.data(), full_v.data() + new_offset, kv_per_token * sizeof(__fp16));
    return s;
}

void run_stdpacked(void* state, float* output) {
    auto* s = static_cast<PackedState*>(state);
    cactus_attention_hybrid_int8_fp16_stdpacked(
        s->q.data(),
        s->kv_packed.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, 1, s->dims.head_dim);
}

void run_interleaved(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    cactus_attention_hybrid_int8_fp16_interleaved(
        s->q.data(),
        s->k_cached.data(), s->v_cached.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

void run_deferscale(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    cactus_attention_hybrid_int8_fp16_deferscale(
        s->q.data(),
        s->k_cached.data(), s->v_cached.data(),
        s->k_scales.data(), s->v_scales.data(),
        s->k_new.data(), s->v_new.data(),
        s->output.data(),
        1, 1, s->cache_len, 1,
        s->dims.num_q_heads, s->dims.num_kv_heads, s->dims.head_dim,
        s->scale, s->cache_len, true, 0, bench::kGroupSize);
    if (output)
        transpose_and_convert_back(s->output.data(), output,
                                    s->dims.num_q_heads, s->seq_len, s->dims.head_dim);
}

} // namespace attn

// ── Registration ────────────────────────────────────────────────────────────────

static int reg = [] {
    bench::register_matmul_backend({
        "cactus_int8", "cactus", bench::QuantCategory::INT8, 0,
        matmul::prepare_int8, nullptr, matmul::run_kernel, matmul::cleanup
    });
    bench::register_matmul_backend({
        "cactus_int4", "cactus", bench::QuantCategory::INT4, 0,
        matmul::prepare_int4, nullptr, matmul::run_kernel, matmul::cleanup
    });
    bench::register_attn_backend({
        "cactus_prefill", "cactus", bench::AttnMode::PREFILL,
        attn::prefill, attn::run, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode", "cactus", bench::AttnMode::DECODE,
        attn::decode, attn::run, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode_transposed", "cactus", bench::AttnMode::DECODE,
        attn::decode_transposed, attn::run_transposed, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode_highthread", "cactus", bench::AttnMode::DECODE,
        attn::decode, attn::run_highthread, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode_packed", "cactus", bench::AttnMode::DECODE,
        attn::decode_packed, attn::run_packed, attn::cleanup_packed
    });
    bench::register_attn_backend({
        "cactus_decode_stdpacked", "cactus", bench::AttnMode::DECODE,
        attn::decode_stdpacked, attn::run_stdpacked, attn::cleanup_packed
    });
    bench::register_attn_backend({
        "cactus_decode_deferscale", "cactus", bench::AttnMode::DECODE,
        attn::decode, attn::run_deferscale, attn::cleanup
    });
    bench::register_attn_backend({
        "cactus_decode_interleaved", "cactus", bench::AttnMode::DECODE,
        attn::decode, attn::run_interleaved, attn::cleanup
    });
    return 0;
}();

} // namespace
