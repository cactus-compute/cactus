
#ifdef CACTUS_HAS_MLX

#include "mlx_gemma4.h"
#include "../../graph/graph.h"
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/backend/metal/metal.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>


namespace mx = mlx::core;
namespace mxf = mlx::core::fast;

using hrc = std::chrono::high_resolution_clock;
static double ms_since(hrc::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(hrc::now() - t).count() / 1000.0;
}

static bool mlx_profile_enabled() {
    static int v = -1;
    if (v < 0) v = std::getenv("CACTUS_MLX_PROFILE") ? 1 : 0;
    return v == 1;
}


struct QuantCache {
    mx::array w;
    mx::array scales;
    mx::array biases;
};

struct PairKey {
    const void* a;
    const void* b;
    bool operator==(const PairKey& o) const { return a == o.a && b == o.b; }
};

struct PairKeyHash {
    size_t operator()(const PairKey& k) const {
        size_t h = std::hash<const void*>{}(k.a);
        return h ^ (std::hash<const void*>{}(k.b) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
};

static std::unordered_map<const void*, QuantCache> s_quant_cache;
static std::unordered_map<PairKey, QuantCache, PairKeyHash> s_quant_pair_cache;
static std::mutex s_quant_mutex;

static size_t weight_rows(const BufferDesc& w) {
    return (w.is_interleaved && w.original_N > 0) ? w.original_N : w.shape[0];
}

static size_t weight_cols(const BufferDesc& w) {
    return w.shape.size() > 1 ? w.shape[1] : 0;
}

static QuantCache build_int4(const int8_t* W, const __fp16* scales_raw,
                              size_t N, size_t K, size_t group_size) {
    const size_t K_blocks  = K / 4;
    const size_t num_groups = K / group_size;
    const size_t K8        = K / 8;
    const uint8_t* B       = reinterpret_cast<const uint8_t*>(W);

    std::vector<uint8_t> w_uint(N * K);
    for (size_t n = 0; n < N; n++) {
        const size_t n_blk = n / 4, ni = n % 4;
        for (size_t k = 0; k < K; k++) {
            const size_t k_blk = k / 4, ki = k % 4;
            size_t p        = (n_blk * K_blocks + k_blk) * 16 + ni * 4 + ki;
            size_t byte_idx = (p / 32) * 16 + (p % 16);
            bool   is_hi    = (p % 32) >= 16;
            uint8_t nib     = is_hi ? (B[byte_idx] >> 4) & 0xF : B[byte_idx] & 0xF;
            w_uint[n * K + k] = (nib + 8) & 0xF;
        }
    }

    std::vector<uint32_t> w_mlx(N * K8);
    for (size_t n = 0; n < N; n++)
        for (size_t k8 = 0; k8 < K8; k8++) {
            uint32_t word = 0;
            for (int i = 0; i < 8; i++)
                word |= (uint32_t)(w_uint[n * K + k8 * 8 + i] & 0xF) << (i * 4);
            w_mlx[n * K8 + k8] = word;
        }

    std::vector<mx::float16_t> sc(N * num_groups), bi(N * num_groups);
    for (size_t n = 0; n < N; n++) {
        const size_t n_blk = n / 4, ni = n % 4;
        for (size_t g = 0; g < num_groups; g++) {
            mx::float16_t s = static_cast<mx::float16_t>(
                scales_raw[(n_blk * num_groups + g) * 4 + ni]);
            sc[n * num_groups + g] = s;
            bi[n * num_groups + g] = mx::float16_t(-8.0f * static_cast<float>(s));
        }
    }

    auto wa = mx::array(w_mlx.data(), {(int)N, (int)K8}, mx::uint32);
    auto sa = mx::array(sc.data(), {(int)N, (int)num_groups}, mx::float16);
    auto ba = mx::array(bi.data(), {(int)N, (int)num_groups}, mx::float16);
    auto deq = mx::dequantize(wa, sa, std::make_optional(ba),
        std::make_optional(static_cast<int>(group_size)), std::make_optional(4),
        "affine", std::nullopt, std::make_optional(mx::float16));
    auto repacked = mx::quantize(deq,
        std::make_optional(static_cast<int>(group_size)), std::make_optional(4), "affine");
    mx::eval(repacked[0], repacked[1], repacked[2]);
    return {std::move(repacked[0]), std::move(repacked[1]), std::move(repacked[2])};
}

static QuantCache build_int8(const int8_t* W, const __fp16* scales_raw,
                              size_t N, size_t K, size_t group_size) {
    const size_t K_blocks  = K / 4;
    const size_t num_groups = K / group_size;
    const size_t K4        = K / 4;

    std::vector<uint8_t> w_uint(N * K);
    for (size_t n = 0; n < N; n++) {
        const size_t n_blk = n / 4, ni = n % 4;
        for (size_t k = 0; k < K; k++) {
            const size_t k_blk = k / 4, ki = k % 4;
            size_t idx = (n_blk * K_blocks + k_blk) * 16 + ni * 4 + ki;
            w_uint[n * K + k] = static_cast<uint8_t>(static_cast<int>(W[idx]) + 128);
        }
    }

    std::vector<uint32_t> w_mlx(N * K4);
    for (size_t n = 0; n < N; n++)
        for (size_t k4 = 0; k4 < K4; k4++) {
            uint32_t word = 0;
            for (int i = 0; i < 4; i++)
                word |= (uint32_t)w_uint[n * K + k4 * 4 + i] << (i * 8);
            w_mlx[n * K4 + k4] = word;
        }

    std::vector<mx::float16_t> sc(N * num_groups), bi(N * num_groups);
    for (size_t n = 0; n < N; n++) {
        const size_t n_blk = n / 4, ni = n % 4;
        for (size_t g = 0; g < num_groups; g++) {
            mx::float16_t s = static_cast<mx::float16_t>(
                scales_raw[(n_blk * num_groups + g) * 4 + ni]);
            sc[n * num_groups + g] = s;
            bi[n * num_groups + g] = mx::float16_t(-128.0f * static_cast<float>(s));
        }
    }

    auto wa = mx::array(w_mlx.data(), {(int)N, (int)K4}, mx::uint32);
    auto sa = mx::array(sc.data(), {(int)N, (int)num_groups}, mx::float16);
    auto ba = mx::array(bi.data(), {(int)N, (int)num_groups}, mx::float16);
    auto deq = mx::dequantize(wa, sa, std::make_optional(ba),
        std::make_optional(static_cast<int>(group_size)), std::make_optional(8),
        "affine", std::nullopt, std::make_optional(mx::float16));
    auto repacked = mx::quantize(deq,
        std::make_optional(static_cast<int>(group_size)), std::make_optional(8), "affine");
    mx::eval(repacked[0], repacked[1], repacked[2]);
    return {std::move(repacked[0]), std::move(repacked[1]), std::move(repacked[2])};
}

static const QuantCache& get_quant_cache(const int8_t* W, const __fp16* scales,
                                          size_t N, size_t K, size_t group_size, bool is_int4) {
    std::lock_guard<std::mutex> lk(s_quant_mutex);
    auto it = s_quant_cache.find(static_cast<const void*>(W));
    if (it == s_quant_cache.end()) {
        auto cache = is_int4 ? build_int4(W, scales, N, K, group_size)
                             : build_int8(W, scales, N, K, group_size);
        it = s_quant_cache.emplace(static_cast<const void*>(W), std::move(cache)).first;
    }
    return it->second;
}

static const QuantCache& get_combined_quant_cache(const BufferDesc& gate_w, const BufferDesc& up_w) {
    PairKey key{gate_w.get_data(), up_w.get_data()};
    {
        std::lock_guard<std::mutex> lk(s_quant_mutex);
        auto it = s_quant_pair_cache.find(key);
        if (it != s_quant_pair_cache.end()) return it->second;
    }

    const size_t gate_N = weight_rows(gate_w), up_N = weight_rows(up_w);
    const size_t K = weight_cols(gate_w);
    const bool is_int4 = gate_w.precision == Precision::INT4;

    const auto& gate = get_quant_cache(gate_w.data_as<int8_t>(), gate_w.scales_as_fp16(),
                                        gate_N, K, gate_w.group_size, is_int4);
    const auto& up = get_quant_cache(up_w.data_as<int8_t>(), up_w.scales_as_fp16(),
                                      up_N, K, up_w.group_size, is_int4);

    auto w = mx::concatenate({gate.w, up.w}, 0);
    auto scales = mx::concatenate({gate.scales, up.scales}, 0);
    auto biases = mx::concatenate({gate.biases, up.biases}, 0);
    mx::eval(w, scales, biases);

    std::lock_guard<std::mutex> lk(s_quant_mutex);
    auto [it, _] = s_quant_pair_cache.emplace(key, QuantCache{
        std::move(w), std::move(scales), std::move(biases)});
    return it->second;
}


static std::unordered_map<const void*, mx::array> s_fp16_cache;
static std::mutex s_fp16_mutex;

static mx::array get_fp16_array(const __fp16* data, std::initializer_list<int> shape) {
    std::lock_guard<std::mutex> lk(s_fp16_mutex);
    auto it = s_fp16_cache.find(static_cast<const void*>(data));
    if (it == s_fp16_cache.end()) {
        std::vector<int> sv(shape);
        mx::array arr = (sv.size() == 1)
            ? mx::array(reinterpret_cast<const mx::float16_t*>(data), {sv[0]}, mx::float16)
            : mx::array(reinterpret_cast<const mx::float16_t*>(data), {sv[0], sv[1]}, mx::float16);
        mx::eval(arr);
        it = s_fp16_cache.emplace(static_cast<const void*>(data), std::move(arr)).first;
    }
    return it->second;
}


static mx::array mlx_proj(const mx::array& x, const BufferDesc& w_desc) {
    size_t N = weight_rows(w_desc);
    size_t K = w_desc.shape.size() > 1 ? w_desc.shape[1] : x.shape().back();

    auto xf16 = mx::astype(x, mx::float16);
    const bool is_quant = PrecisionTraits::is_integer(w_desc.precision) && w_desc.group_size > 0;
    if (!is_quant) {
        auto wt = get_fp16_array(w_desc.data_as<__fp16>(), {(int)N, (int)K});
        return mx::matmul(xf16, mx::transpose(wt));
    }
    const bool is_int4 = (w_desc.precision == Precision::INT4);
    const auto& c = get_quant_cache(w_desc.data_as<int8_t>(), w_desc.scales_as_fp16(),
                                    N, K, w_desc.group_size, is_int4);
    auto result = mx::quantized_matmul(xf16, c.w, c.scales,
        std::make_optional(c.biases), true,
        std::make_optional((int)w_desc.group_size),
        std::make_optional(is_int4 ? 4 : 8));
    return mx::astype(result, mx::float16);
}


static mx::array mlx_rms_norm(const mx::array& x, const BufferDesc& w_desc, float eps) {
    auto w = get_fp16_array(w_desc.data_as<__fp16>(), {(int)w_desc.shape[0]});
    return mxf::rms_norm(mx::astype(x, mx::float16), w, eps);
}

static mx::array mlx_rms_norm_ones(const mx::array& x, float eps) {
    return mxf::rms_norm(mx::astype(x, mx::float16), std::nullopt, eps);
}


static mx::array mlx_rope(const mx::array& x, float theta,
                            size_t head_dim, size_t rot_dim, size_t pos_offset) {
    int S = x.shape()[0], heads = x.shape()[1], half = static_cast<int>(rot_dim / 2);

    std::vector<float> pos_data(S);
    for (int i = 0; i < S; i++) pos_data[i] = static_cast<float>(pos_offset + i);
    auto pos = mx::array(pos_data.data(), {S, 1, 1}, mx::float32);

    std::vector<float> freq_data(half);
    for (int i = 0; i < half; i++)
        freq_data[i] = 1.0f / std::pow(theta, 2.0f * i / static_cast<float>(head_dim));
    auto freq = mx::array(freq_data.data(), {1, 1, half}, mx::float32);

    auto angles = mx::multiply(pos, freq);
    auto cos_a = mx::astype(mx::cos(angles), mx::float16);
    auto sin_a = mx::astype(mx::sin(angles), mx::float16);

    auto x1 = mx::slice(x, {0, 0, 0}, {S, heads, half});
    auto x2 = mx::slice(x, {0, 0, half}, {S, heads, (int)rot_dim});

    auto out1 = mx::subtract(mx::multiply(x1, cos_a), mx::multiply(x2, sin_a));
    auto out2 = mx::add(mx::multiply(x2, cos_a), mx::multiply(x1, sin_a));

    if (rot_dim == head_dim) return mx::concatenate({out1, out2}, 2);
    auto x_tail = mx::slice(x, {0, 0, (int)rot_dim}, {S, heads, (int)head_dim});
    return mx::concatenate({out1, out2, x_tail}, 2);
}


struct AttentionResult {
    mx::array out;
    mx::array next_k = mx::array(0.0f);
    mx::array next_v = mx::array(0.0f);
};

static AttentionResult mlx_attention_sliding_prefill(
        const mx::array& q, const mx::array& k_new, const mx::array& v_new,
        float scale, size_t window) {
    const int T = q.shape()[0], H = q.shape()[1], Hkv = k_new.shape()[1], d = q.shape()[2];
    const int chunk = 128;

    auto Q = mx::transpose(mx::expand_dims(q, 0), {0, 2, 1, 3});
    auto K = mx::transpose(mx::expand_dims(k_new, 0), {0, 2, 1, 3});
    auto V = mx::transpose(mx::expand_dims(v_new, 0), {0, 2, 1, 3});

    std::vector<mx::array> out_chunks;
    for (int q_start = 0; q_start < T; q_start += chunk) {
        const int q_end = std::min(T, q_start + chunk), Tc = q_end - q_start;
        const int k_start = std::max(0, q_start - static_cast<int>(window));
        const int k_end = q_end, Sc = k_end - k_start;

        auto q_chunk = mx::slice(Q, {0, 0, q_start, 0}, {1, H, q_end, d});
        auto k_chunk = mx::slice(K, {0, 0, k_start, 0}, {1, Hkv, k_end, d});
        auto v_chunk = mx::slice(V, {0, 0, k_start, 0}, {1, Hkv, k_end, d});

        std::vector<mx::float16_t> mask_data(static_cast<size_t>(Tc) * Sc, mx::float16_t(0.0f));
        const mx::float16_t neg_inf = mx::float16_t(-INFINITY);
        for (int qi = 0; qi < Tc; qi++)
            for (int kj = 0; kj < Sc; kj++) {
                int q_abs = q_start + qi, k_abs = k_start + kj;
                if (k_abs > q_abs || (q_abs - k_abs) > static_cast<int>(window))
                    mask_data[static_cast<size_t>(qi) * Sc + kj] = neg_inf;
            }

        auto mask_arr = mx::array(mask_data.data(), {1, 1, Tc, Sc}, mx::float16);
        out_chunks.push_back(mxf::scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, scale, "array", mask_arr));
    }

    auto out = out_chunks.size() == 1 ? out_chunks[0] : mx::concatenate(out_chunks, 2);
    auto out_thd = mx::reshape(mx::transpose(out, {0, 2, 1, 3}),
                                {T, out.shape()[1], out.shape()[3]});

    mx::array next_k = k_new, next_v = v_new;
    if (window > 0 && T > static_cast<int>(window)) {
        next_k = mx::slice(k_new, {T - static_cast<int>(window), 0, 0}, {T, Hkv, d});
        next_v = mx::slice(v_new, {T - static_cast<int>(window), 0, 0}, {T, Hkv, d});
    }
    return {out_thd, next_k, next_v};
}

static AttentionResult mlx_attention(const mx::array& q, const mx::array& k_new, const mx::array& v_new,
                                      const Gemma4MLXState::LayerKV& kv_state,
                                      float scale, size_t window, size_t pos_offset) {
    int T = q.shape()[0], d = q.shape()[2], Hkv = k_new.shape()[1];

    if (window > 0 && kv_state.cached_len == 0 && pos_offset == 0 && T > 1)
        return mlx_attention_sliding_prefill(q, k_new, v_new, scale, window);

    mx::array k_full = k_new, v_full = v_new;
    if (kv_state.cached_len > 0 && !kv_state.k.empty()) {
        auto k_cached = mx::array(reinterpret_cast<const mx::float16_t*>(kv_state.k.data()),
            {(int)kv_state.cached_len, Hkv, d}, mx::float16);
        auto v_cached = mx::array(reinterpret_cast<const mx::float16_t*>(kv_state.v.data()),
            {(int)kv_state.cached_len, Hkv, d}, mx::float16);
        k_full = mx::concatenate({k_cached, k_new}, 0);
        v_full = mx::concatenate({v_cached, v_new}, 0);
    }

    int S = k_full.shape()[0];
    auto Q = mx::transpose(mx::expand_dims(q, 0), {0, 2, 1, 3});
    auto K = mx::transpose(mx::expand_dims(k_full, 0), {0, 2, 1, 3});
    auto V = mx::transpose(mx::expand_dims(v_full, 0), {0, 2, 1, 3});

    std::string mask_mode;
    std::optional<mx::array> mask_arr;

    if (T > 1 && (pos_offset > 0 || window > 0)) {
        std::vector<mx::float16_t> mask_data(static_cast<size_t>(T) * S, mx::float16_t(0.0f));
        const int64_t k_abs_start = static_cast<int64_t>(pos_offset) - static_cast<int64_t>(kv_state.cached_len);
        const mx::float16_t neg_inf = mx::float16_t(-INFINITY);
        for (int qi = 0; qi < T; qi++)
            for (int kj = 0; kj < S; kj++) {
                int64_t q_abs = static_cast<int64_t>(pos_offset + qi);
                int64_t k_abs = k_abs_start + kj;
                if (k_abs > q_abs || (window > 0 && (q_abs - k_abs) > static_cast<int64_t>(window)))
                    mask_data[static_cast<size_t>(qi) * S + kj] = neg_inf;
            }
        mask_arr = mx::array(mask_data.data(), {1, 1, T, S}, mx::float16);
        mask_mode = "array";
    } else if (T > 1) {
        mask_mode = "causal";
    }

    mx::array out = mask_arr
        ? mxf::scaled_dot_product_attention(Q, K, V, scale, mask_mode, *mask_arr)
        : mxf::scaled_dot_product_attention(Q, K, V, scale, mask_mode);
    auto out_thd = mx::reshape(mx::transpose(out, {0, 2, 1, 3}),
                                {T, out.shape()[1], out.shape()[3]});

    mx::array next_k = mx::array(0.0f), next_v = mx::array(0.0f);
    if (window > 0 && S > (int)window) {
        next_k = mx::slice(k_full, {S - (int)window, 0, 0}, {S, Hkv, d});
        next_v = mx::slice(v_full, {S - (int)window, 0, 0}, {S, Hkv, d});
    } else {
        next_k = k_full;
        next_v = v_full;
    }
    return {out_thd, next_k, next_v};
}


static mx::array mlx_gelu_gated(const mx::array& gate_proj, const mx::array& up_proj) {
    auto gate_safe = mx::maximum(mx::minimum(gate_proj,
        mx::array(mx::float16_t(40.0f))), mx::array(mx::float16_t(-40.0f)));
    const auto kS = mx::array(mx::float16_t(0.7978845608f));
    auto x3    = mx::multiply(gate_safe, mx::multiply(gate_safe, gate_safe));
    auto inner = mx::add(gate_safe, mx::multiply(mx::array(mx::float16_t(0.044715f)), x3));
    auto gelu  = mx::multiply(gate_proj,
                   mx::multiply(mx::array(mx::float16_t(0.5f)),
                     mx::add(mx::array(mx::float16_t(1.0f)),
                       mx::tanh(mx::multiply(kS, inner)))));
    auto gated = mx::multiply(gelu, up_proj);
    return mx::maximum(mx::minimum(gated,
        mx::array(mx::float16_t(65504.0f))), mx::array(mx::float16_t(-65504.0f)));
}

static mx::array mlx_ffn(const mx::array& x,
                           const BufferDesc& gate_w, const BufferDesc& up_w,
                           const BufferDesc& down_w) {
    auto xf16 = mx::astype(x, mx::float16);
    const size_t ffn_dim = weight_rows(gate_w);
    const bool fuse_gate_up =
        PrecisionTraits::is_integer(gate_w.precision) &&
        PrecisionTraits::is_integer(up_w.precision) &&
        gate_w.group_size > 0 &&
        gate_w.precision == up_w.precision &&
        gate_w.group_size == up_w.group_size &&
        weight_cols(gate_w) == weight_cols(up_w) &&
        weight_rows(gate_w) == weight_rows(up_w);

    mx::array gate_proj = mx::array(0.0f), up_proj = mx::array(0.0f);
    if (fuse_gate_up) {
        const auto& combined = get_combined_quant_cache(gate_w, up_w);
        auto gu = mx::quantized_matmul(xf16, combined.w, combined.scales,
            std::make_optional(combined.biases), true,
            std::make_optional((int)gate_w.group_size),
            std::make_optional(gate_w.precision == Precision::INT4 ? 4 : 8));
        auto gu16 = mx::astype(gu, mx::float16);
        int T = gu16.shape()[0], D = static_cast<int>(ffn_dim);
        gate_proj = mx::slice(gu16, {0, 0}, {T, D});
        up_proj   = mx::slice(gu16, {0, D}, {T, 2 * D});
    } else {
        gate_proj = mlx_proj(xf16, gate_w);
        up_proj   = mlx_proj(xf16, up_w);
    }

    return mlx_proj(mlx_gelu_gated(gate_proj, up_proj), down_w);
}


static mx::array mlx_pli(const mx::array& hidden, const mx::array& pli_combined,
                           const BufferDesc& gate_w, const BufferDesc& proj_w,
                           const BufferDesc& norm_w, uint32_t layer_idx, uint32_t pli_dim, float eps) {
    int T = hidden.shape()[0];
    int start = static_cast<int>(layer_idx * pli_dim), end = start + static_cast<int>(pli_dim);
    auto pli_slice = mx::slice(pli_combined, {0, start}, {T, end});

    auto gate_raw = mlx_proj(hidden, gate_w);
    auto gf = mx::astype(gate_raw, mx::float32);
    auto x3 = mx::multiply(gf, mx::multiply(gf, gf));
    auto inner = mx::add(gf, mx::multiply(mx::array(0.044715f), x3));
    auto gate = mx::astype(mx::multiply(gf,
        mx::multiply(mx::array(0.5f),
          mx::add(mx::array(1.0f),
            mx::tanh(mx::multiply(mx::array(0.7978845608f), inner))))), mx::float16);

    auto gated = mx::multiply(gate, mx::astype(pli_slice, mx::float16));
    return mx::add(hidden, mlx_rms_norm(mlx_proj(gated, proj_w), norm_w, eps));
}


static mx::array mlx_layer_qkv(
        const mx::array& normed, CactusGraph& g, const Gemma4MLXLayerNodes& nodes,
        int T, int H, int Hkv, int d, float norm_eps, size_t pos_offset,
        const std::vector<mx::array>& shared_k, const std::vector<mx::array>& shared_v,
        mx::array* k_out, mx::array* v_out,
        mx::array& q_out) {
    auto q_flat = mlx_proj(normed, g.get_output_buffer(nodes.q_weight));
    auto q_normed = mlx_rms_norm(mx::reshape(q_flat, {T * H, d}),
                                  g.get_output_buffer(nodes.q_norm), norm_eps);
    auto q = mx::reshape(q_normed, {T, H, d});

    mx::array k = mx::array(0.0f), v = mx::array(0.0f);
    if (nodes.share_src < 0) {
        auto k_flat = mlx_proj(normed, g.get_output_buffer(nodes.k_weight));
        auto k_normed = mlx_rms_norm(mx::reshape(k_flat, {T * Hkv, d}),
                                      g.get_output_buffer(nodes.k_norm), norm_eps);
        k = mx::reshape(k_normed, {T, Hkv, d});

        auto v_flat = mlx_proj(normed, g.get_output_buffer(nodes.v_weight));
        v = mx::reshape(mlx_rms_norm_ones(mx::reshape(v_flat, {T * Hkv, d}), norm_eps), {T, Hkv, d});

        q = mlx_rope(q, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        k = mlx_rope(k, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        if (k_out) *k_out = k;
        if (v_out) *v_out = v;
    } else {
        q = mlx_rope(q, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        k = shared_k[nodes.share_src];
        v = shared_v[nodes.share_src];
    }
    q_out = q;
    return k; }

static mx::array mlx_transformer_layer(
        const mx::array& hidden, const mx::array& pli_combined,
        const Gemma4MLXLayerNodes& nodes, CactusGraph* gb,
        const Gemma4MLXState::LayerKV& kv_state,
        float norm_eps, float attn_scale, uint32_t pli_dim, bool has_pli,
        uint32_t layer_idx, size_t pos_offset,
        const std::vector<mx::array>& shared_k, const std::vector<mx::array>& shared_v,
        mx::array* current_k_out, mx::array* current_v_out,
        mx::array* cache_k_out, mx::array* cache_v_out) {

    auto& g = *gb;
    int T = hidden.shape()[0];
    int H = static_cast<int>(nodes.num_heads), Hkv = static_cast<int>(nodes.kv_heads);
    int d = static_cast<int>(nodes.head_dim);

    auto normed = mlx_rms_norm(hidden, g.get_output_buffer(nodes.input_ln), norm_eps);

    mx::array q = mx::array(0.0f), k = mx::array(0.0f), v = mx::array(0.0f);
    if (nodes.share_src < 0) {
        auto q_flat = mlx_proj(normed, g.get_output_buffer(nodes.q_weight));
        q = mx::reshape(mlx_rms_norm(mx::reshape(q_flat, {T*H, d}),
                         g.get_output_buffer(nodes.q_norm), norm_eps), {T, H, d});

        auto k_flat = mlx_proj(normed, g.get_output_buffer(nodes.k_weight));
        k = mx::reshape(mlx_rms_norm(mx::reshape(k_flat, {T*Hkv, d}),
                         g.get_output_buffer(nodes.k_norm), norm_eps), {T, Hkv, d});

        auto v_flat = mlx_proj(normed, g.get_output_buffer(nodes.v_weight));
        v = mx::reshape(mlx_rms_norm_ones(mx::reshape(v_flat, {T*Hkv, d}), norm_eps), {T, Hkv, d});

        q = mlx_rope(q, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        k = mlx_rope(k, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        if (current_k_out) *current_k_out = k;
        if (current_v_out) *current_v_out = v;
    } else {
        auto q_flat = mlx_proj(normed, g.get_output_buffer(nodes.q_weight));
        q = mx::reshape(mlx_rms_norm(mx::reshape(q_flat, {T*H, d}),
                         g.get_output_buffer(nodes.q_norm), norm_eps), {T, H, d});
        q = mlx_rope(q, nodes.rope_freq, d, nodes.rot_dim, pos_offset);
        k = shared_k[nodes.share_src];
        v = shared_v[nodes.share_src];
    }

    auto attn = mlx_attention(q, k, v, kv_state, attn_scale, nodes.window, pos_offset);
    if (nodes.share_src < 0) {
        if (cache_k_out) *cache_k_out = attn.next_k;
        if (cache_v_out) *cache_v_out = attn.next_v;
    }

    auto attn_proj = mlx_proj(mx::reshape(attn.out, {T, H * d}), g.get_output_buffer(nodes.o_weight));
    auto residual = mx::add(hidden, mlx_rms_norm(attn_proj, g.get_output_buffer(nodes.post_attn_ln), norm_eps));

    auto pre_ffn = mlx_rms_norm(residual, g.get_output_buffer(nodes.pre_ffn_ln), norm_eps);
    auto ffn_out = mlx_ffn(pre_ffn, g.get_output_buffer(nodes.gate_weight),
                            g.get_output_buffer(nodes.up_weight), g.get_output_buffer(nodes.down_weight));
    auto h2 = mx::add(residual, mlx_rms_norm(ffn_out, g.get_output_buffer(nodes.post_ffn_ln), norm_eps));

    if (has_pli && nodes.pli_gate != 0)
        h2 = mlx_pli(h2, pli_combined, g.get_output_buffer(nodes.pli_gate),
                      g.get_output_buffer(nodes.pli_proj), g.get_output_buffer(nodes.pli_norm),
                      layer_idx, pli_dim, norm_eps);

    if (nodes.layer_scalar != 0) {
        const BufferDesc& ls = g.get_output_buffer(nodes.layer_scalar);
        if (ls.get_data())
            h2 = mx::multiply(h2, get_fp16_array(ls.data_as<__fp16>(), {1}));
    }

    return h2;
}


static mx::array build_pli_combined(const mx::array& h, const mx::array& pli_embed,
                                     Gemma4MLXForwardParams& params, int T) {
    int nl = static_cast<int>(params.num_layers), pli_d = static_cast<int>(params.pli_dim);

    auto scale = mx::array(mx::float16_t(1.0f / std::sqrt(static_cast<float>(params.hidden_dim))));
    auto pli_proj = mx::multiply(mlx_proj(h, params.gb->get_output_buffer(params.per_layer_model_proj_node)), scale);
    auto pp_n = mlx_rms_norm(mx::reshape(pli_proj, {T * nl, pli_d}),
                              params.gb->get_output_buffer(params.per_layer_proj_norm_node), params.norm_eps);
    auto pp_back = mx::reshape(pp_n, {T, nl * pli_d});

    auto inv_sqrt2 = mx::array(mx::float16_t(1.0f / std::sqrt(2.0f)));
    return mx::multiply(mx::add(pp_back, pli_embed), inv_sqrt2);
}

static void update_kv_cache(Gemma4MLXState* state, const Gemma4MLXForwardParams& params,
                             const std::vector<mx::array>& next_cache_k,
                             const std::vector<mx::array>& next_cache_v, int T) {
    for (uint32_t i = 0; i < params.num_layers; i++) {
        const auto& ln = params.layers[i];
        if (ln.share_src >= 0) continue;

        const mx::array& k_next = next_cache_k[i];
        const mx::array& v_next = next_cache_v[i];
        int Hkv = static_cast<int>(ln.kv_heads), d = static_cast<int>(ln.head_dim);

        auto& kv = state->layer_kv[i];
        kv.kv_heads = Hkv;
        kv.head_dim = d;

        size_t keep = static_cast<size_t>(k_next.shape()[0]);
        size_t elem = static_cast<size_t>(Hkv) * d;
        kv.k.resize(keep * elem);
        kv.v.resize(keep * elem);
        std::memcpy(kv.k.data(), k_next.data<mx::float16_t>(), keep * elem * sizeof(uint16_t));
        std::memcpy(kv.v.data(), v_next.data<mx::float16_t>(), keep * elem * sizeof(uint16_t));
        kv.cached_len = keep;
    }
    state->cached_seq_len += static_cast<size_t>(T);
}

MLXFusedFn make_gemma4_full_forward_fn(
        Gemma4MLXState* state,
        std::shared_ptr<Gemma4MLXForwardParams> params) {

    return [state, params](const std::vector<const BufferDesc*>& inputs, BufferDesc& output) {
        const BufferDesc& embed_desc = *inputs[0];
        int T = static_cast<int>(embed_desc.shape[0]);
        int hidden = static_cast<int>(embed_desc.shape[1]);
        size_t pos_offset = state->cached_seq_len;

        auto h = mx::array(reinterpret_cast<const mx::float16_t*>(embed_desc.data_as<__fp16>()),
                           {T, hidden}, mx::float16);

        mx::array pli_combined = mx::zeros({1});
        if (params->has_pli && inputs.size() > 1) {
            const BufferDesc& pli_embed_desc = *inputs[1];
            int pli_total = static_cast<int>(params->num_layers * params->pli_dim);
            auto pli_embed = mx::array(
                reinterpret_cast<const mx::float16_t*>(pli_embed_desc.data_as<__fp16>()),
                {T, pli_total}, mx::float16);
            pli_combined = build_pli_combined(h, pli_embed, *params, T);
        }

        if (state->layer_kv.size() < (size_t)params->num_layers)
            state->layer_kv.resize(params->num_layers);

        std::vector<mx::array> to_eval;
        std::vector<mx::array> current_k(params->num_layers, mx::array(0.0f));
        std::vector<mx::array> current_v(params->num_layers, mx::array(0.0f));
        std::vector<mx::array> next_cache_k(params->num_layers, mx::array(0.0f));
        std::vector<mx::array> next_cache_v(params->num_layers, mx::array(0.0f));

        const bool profiling = mlx_profile_enabled();
        auto t_fwd = profiling ? hrc::now() : hrc::time_point{};

        for (uint32_t i = 0; i < params->num_layers; i++) {
            const auto& layer = params->layers[i];
            uint32_t cache_idx = layer.share_src >= 0 ? static_cast<uint32_t>(layer.share_src) : i;
            h = mlx_transformer_layer(
                h, pli_combined, layer, params->gb,
                state->layer_kv[cache_idx],
                params->norm_eps, params->attention_scale,
                params->pli_dim, params->has_pli,
                i, pos_offset, current_k, current_v,
                layer.share_src < 0 ? &current_k[i] : nullptr,
                layer.share_src < 0 ? &current_v[i] : nullptr,
                layer.share_src < 0 ? &next_cache_k[i] : nullptr,
                layer.share_src < 0 ? &next_cache_v[i] : nullptr);
        }

        for (uint32_t i = 0; i < params->num_layers; i++)
            if (params->layers[i].share_src < 0) {
                to_eval.push_back(next_cache_k[i]);
                to_eval.push_back(next_cache_v[i]);
            }
        to_eval.push_back(h);
        mx::eval(to_eval);

        if (profiling)
            std::cerr << "[MLX] T=" << T << " forward=" << ms_since(t_fwd) << " ms"
                      << " (" << (T * 1000.0 / ms_since(t_fwd)) << " tps)\n";

        std::memcpy(output.data_as<__fp16>(), h.data<mx::float16_t>(),
                    static_cast<size_t>(T) * hidden * sizeof(__fp16));

        update_kv_cache(state, *params, next_cache_k, next_cache_v, T);
    };
}

#else

#include "mlx_gemma4.h"

MLXFusedFn make_gemma4_full_forward_fn(Gemma4MLXState*, std::shared_ptr<Gemma4MLXForwardParams>) { return {}; }

#endif
