#include "bench_driver.h"

#include "../../cactus-kernels/cactus_kernels.h"
#include "../../cactus-kernels/src/threading.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstring>

namespace {

namespace matmul {

struct Cq4Weights {
    size_t K = 0, N = 0;
    uint32_t group_size = 128;
    uint32_t num_groups = 0;
    std::vector<__fp16> codebook;
    std::vector<__fp16> input_scale;
    std::vector<__fp16> input_scale_recip;
    std::vector<__fp16> norms;
    std::vector<uint8_t> packed_indices;
    std::vector<int8_t> left_signs;
    std::vector<int8_t> right_signs;
    std::vector<uint32_t> permutation;
    std::vector<__fp16> output_buf;
    CactusQuantMatrix matrix{};
};

struct Cq4Activations {
    std::vector<__fp16> fp16;
};

void* prepare_cq4(const float* fp32, size_t N, size_t K) {
    auto* w = new Cq4Weights();
    w->K = K;
    w->N = N;
    w->group_size = static_cast<uint32_t>(std::min<size_t>(128, K));
    w->num_groups = static_cast<uint32_t>((K + w->group_size - 1) / w->group_size);

    constexpr uint32_t bits = 4;
    constexpr uint32_t codebook_size = 16;
    w->codebook.resize(codebook_size);
    for (uint32_t i = 0; i < codebook_size; ++i) {
        float v = (static_cast<float>(i) - 7.5f) / 7.5f;
        w->codebook[i] = static_cast<__fp16>(v);
    }

    w->input_scale.assign(K, static_cast<__fp16>(1.0f));
    w->input_scale_recip.assign(K, static_cast<__fp16>(1.0f));
    w->left_signs.assign(w->group_size, 1);
    w->right_signs.assign(w->group_size, 1);
    w->permutation.resize(w->group_size);
    for (uint32_t i = 0; i < w->group_size; ++i) w->permutation[i] = i;

    const uint32_t packed_group_bytes = cactus_quant_packed_group_bytes(bits, w->group_size);
    w->norms.resize(N * w->num_groups);
    w->packed_indices.assign(N * w->num_groups * packed_group_bytes, 0);

    for (size_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < w->num_groups; ++g) {
            const size_t base = n * K + static_cast<size_t>(g) * w->group_size;
            const size_t end = std::min(base + w->group_size, (n + 1) * K);
            float max_abs = 1e-6f;
            for (size_t k = base; k < end; ++k)
                max_abs = std::max(max_abs, std::abs(fp32[k]));
            w->norms[n * w->num_groups + g] = static_cast<__fp16>(max_abs);

            uint8_t* packed = w->packed_indices.data() +
                (n * w->num_groups + g) * packed_group_bytes;
            for (uint32_t k = 0; k < w->group_size; ++k) {
                float v = 0.0f;
                if (base + k < end) v = fp32[base + k] / max_abs;
                int idx = static_cast<int>(std::round((std::max(-1.0f, std::min(1.0f, v)) + 1.0f) * 7.5f));
                idx = std::max(0, std::min(15, idx));
                if ((k & 1u) == 0)
                    packed[k >> 1] = static_cast<uint8_t>(idx);
                else
                    packed[k >> 1] = static_cast<uint8_t>(packed[k >> 1] | (idx << 4));
            }
        }
    }

    w->matrix = CactusQuantMatrix{
        bits, static_cast<uint32_t>(K), static_cast<uint32_t>(N),
        w->group_size, w->num_groups, 0,
        w->codebook.data(), w->input_scale.data(), w->input_scale_recip.data(),
        w->norms.data(), w->packed_indices.data(), w->left_signs.data(),
        w->right_signs.data(), w->permutation.data(), nullptr, nullptr, nullptr
    };
    return w;
}

void* prepare_cq4_activations(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new Cq4Activations();
    a->fp16.resize(M * K);
    bench::fp32_to_fp16(fp32, a->fp16.data(), M * K);
    return a;
}

void run_cq4_kernel(size_t M, size_t, size_t,
                    void* weights, void* activations,
                    const int8_t*, const float*,
                    float* output, float* reference) {
    auto* w = static_cast<Cq4Weights*>(weights);
    auto* a = static_cast<Cq4Activations*>(activations);
    w->output_buf.assign(M * w->N, __fp16(0.0f));

    int thr = bench::get_thread_override();
    if (thr > 0) CactusThreading::set_gemm_threads(static_cast<size_t>(thr));
    cactus_quant_matmul(&w->matrix, a->fp16.data(), static_cast<uint32_t>(M), w->output_buf.data());
    if (thr > 0) CactusThreading::reset_gemm_threads();

    if (output) bench::fp16_to_fp32(w->output_buf.data(), output, M * w->N);
    if (reference) bench::fp16_to_fp32(w->output_buf.data(), reference, M * w->N);
}

void cleanup_cq4(void* weights, void* activations) {
    delete static_cast<Cq4Weights*>(weights);
    delete static_cast<Cq4Activations*>(activations);
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
        "cactus_cq4", "cactus",
        matmul::prepare_cq4, matmul::prepare_cq4_activations,
        matmul::run_cq4_kernel, matmul::cleanup_cq4
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
