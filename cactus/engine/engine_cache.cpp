#include "engine.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {
using LC = cactus::engine::KVCache::LayerCache;

static void tq_resize(LC& c, size_t cap_tokens, size_t kv_heads,
                      size_t k_ang_bytes, size_t v_ang_bytes, size_t qjl_bytes,
                      bool use_qjl) {
    size_t n = cap_tokens * kv_heads;
    c.tq_key_radii.resize(n);
    c.tq_key_angles.resize(n * k_ang_bytes);
    c.tq_val_radii.resize(n);
    c.tq_val_angles.resize(n * v_ang_bytes);
    if (use_qjl) {
        c.tq_key_error_norms.resize(n);
        c.tq_key_qjl_bits.resize(n * qjl_bytes);
    } else {
        c.tq_key_error_norms.clear();
        c.tq_key_qjl_bits.clear();
    }
    // V never uses QJL in this codebase — drop these buffers always.
    c.tq_val_error_norms.clear();
    c.tq_val_qjl_bits.clear();
}

static void tq_encode_kv(LC& c,
                         const __fp16* k_fp16, const __fp16* v_fp16,
                         size_t dst_token_offset, size_t num_tokens,
                         size_t kv_heads, size_t head_dim,
                         size_t k_angle_bits, size_t v_angle_bits, size_t projection_dim,
                         size_t k_ang_bytes, size_t v_ang_bytes, size_t qjl_bytes,
                         bool use_qjl,
                         const uint8_t* rot_signs, const uint8_t* proj_mat) {
    size_t e = dst_token_offset * kv_heads;
    float*   k_err = use_qjl ? c.tq_key_error_norms.data() + e : nullptr;
    uint8_t* k_qjl = use_qjl ? c.tq_key_qjl_bits.data()   + e * qjl_bytes : nullptr;
    cactus_turboquant_encode_kv_fp16(
        k_fp16,
        c.tq_key_radii.data()       + e, c.tq_key_angles.data()      + e * k_ang_bytes,
        k_err, k_qjl,
        rot_signs, proj_mat, num_tokens, kv_heads, head_dim, k_angle_bits, projection_dim);
    cactus_turboquant_encode_kv_fp16(
        v_fp16,
        c.tq_val_radii.data()       + e, c.tq_val_angles.data()      + e * v_ang_bytes,
        nullptr, nullptr,
        rot_signs, proj_mat, num_tokens, kv_heads, head_dim, v_angle_bits, projection_dim);
}

static void tq_shift(LC& c, size_t dst_token, size_t src_token, size_t count,
                     size_t kv_heads, size_t k_ang_bytes, size_t v_ang_bytes, size_t qjl_bytes) {
    if (count == 0 || dst_token == src_token) return;
    size_t d = dst_token * kv_heads, s = src_token * kv_heads, n = count * kv_heads;
    std::memmove(c.tq_key_radii.data()  + d, c.tq_key_radii.data()  + s, n * sizeof(float));
    std::memmove(c.tq_key_angles.data() + d * k_ang_bytes, c.tq_key_angles.data() + s * k_ang_bytes, n * k_ang_bytes);
    std::memmove(c.tq_val_radii.data()  + d, c.tq_val_radii.data()  + s, n * sizeof(float));
    std::memmove(c.tq_val_angles.data() + d * v_ang_bytes, c.tq_val_angles.data() + s * v_ang_bytes, n * v_ang_bytes);
    if (!c.tq_key_error_norms.empty())
        std::memmove(c.tq_key_error_norms.data() + d, c.tq_key_error_norms.data() + s, n * sizeof(float));
    if (!c.tq_key_qjl_bits.empty())
        std::memmove(c.tq_key_qjl_bits.data() + d * qjl_bytes, c.tq_key_qjl_bits.data() + s * qjl_bytes, n * qjl_bytes);
}
} // anonymous namespace

namespace cactus {
namespace engine {

void KVCache::init(size_t layers, size_t max_seq, size_t kv_heads, size_t dim, Precision model_precision,
                   KVQuantMethod method, size_t tq_proj_dim,
                   size_t tq_k_bits, size_t tq_v_bits, bool tq_qjl, uint64_t tq_seed) {
    num_layers = layers;
    max_seq_len = max_seq;
    num_kv_heads = kv_heads;
    head_dim = dim;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);
    quant_method = method;
    tq_projection_dim = tq_proj_dim;
    tq_key_angle_bits = tq_k_bits;
    tq_value_angle_bits = tq_v_bits;
    tq_use_qjl = tq_qjl;

    layer_caches.resize(num_layers);

    current_seq_len = 0;
    total_seq_len = 0;

    if (quant_method == KVQuantMethod::TURBOQUANT && dim > 0) {
        tq_rotation_signs.resize((dim + 7) / 8);
        size_t proj_groups = (tq_projection_dim + 7) / 8;
        size_t rot_row_bytes = (dim + 7) / 8;
        tq_projection_matrix.resize(proj_groups * rot_row_bytes * 8);
        cactus_turboquant_init(tq_rotation_signs.data(), tq_projection_matrix.data(),
                               dim, tq_projection_dim, tq_seed);
    }
}

void KVCache::set_window_size(size_t window, size_t sink) {
    window_size = window;
    sink_size = sink;

    if (num_kv_heads > 0 && head_dim > 0 && window_size > 0) {
        size_t cache_bytes = window_size * num_kv_heads * head_dim * element_size;
        size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
        size_t num_scales = window_size * num_kv_heads * num_groups;

        for (auto& cache : layer_caches) {
            cache.keys.resize(cache_bytes);
            cache.values.resize(cache_bytes);
            std::memset(cache.keys.data(), 0, cache_bytes);
            std::memset(cache.values.data(), 0, cache_bytes);

            cache.key_scales.resize(num_scales);
            cache.value_scales.resize(num_scales);
            std::fill(cache.key_scales.begin(), cache.key_scales.end(), 1.0f);
            std::fill(cache.value_scales.begin(), cache.value_scales.end(), 1.0f);
        }

        if (quant_method == KVQuantMethod::TURBOQUANT) {
            size_t k_ang_bytes = turboquant_angles_bytes_per_head(head_dim, tq_key_angle_bits);
                size_t v_ang_bytes = turboquant_angles_bytes_per_head(head_dim, tq_value_angle_bits);
            size_t qjl_bytes = turboquant_qjl_bytes_per_head(tq_projection_dim);
            for (auto& cache : layer_caches)
                tq_resize(cache, window_size, num_kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
        }
    }
}

void KVCache::reset() {
    current_seq_len = 0;
    total_seq_len = 0;
}

void* KVCache::get_key_ptr(size_t layer) {
    if (current_seq_len == 0 || layer >= num_layers) return nullptr;
    return layer_caches[layer].keys.data();
}

void* KVCache::get_value_ptr(size_t layer) {
    if (current_seq_len == 0 || layer >= num_layers) return nullptr;
    return layer_caches[layer].values.data();
}

KVCache::CircularView KVCache::get_key_view(size_t layer) {
    CircularView view;
    if (layer >= num_layers || current_seq_len == 0) {
    view.ptr1 = nullptr;
        view.ptr2 = nullptr;
        view.len1 = 0;
        view.len2 = 0;
        view.total_len = 0;
        return view;
    }

    view.ptr1 = layer_caches[layer].keys.data();
    view.ptr2 = nullptr;
    view.len1 = current_seq_len;
    view.len2 = 0;
    view.total_len = current_seq_len;
    return view;
}

KVCache::CircularView KVCache::get_value_view(size_t layer) {
    CircularView view;
    if (layer >= num_layers || current_seq_len == 0) {
        view.ptr1 = nullptr;
        view.ptr2 = nullptr;
        view.len1 = 0;
        view.len2 = 0;
        view.total_len = 0;
        return view;
    }

    view.ptr1 = layer_caches[layer].values.data();
    view.ptr2 = nullptr;
    view.len1 = current_seq_len;
    view.len2 = 0;
    view.total_len = current_seq_len;
    return view;
}

void KVCache::update_from_graph(CactusGraph* gb, const std::vector<size_t>& k_nodes,
                               const std::vector<size_t>& v_nodes, size_t seq_len,
                               size_t layers, size_t kv_heads, size_t dim) {
    size_t old_seq_len = current_seq_len;
    size_t new_total_len = old_seq_len + seq_len;
    size_t elements_per_token = kv_heads * dim;
    size_t num_groups = (dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    size_t scales_per_token = kv_heads * num_groups;
    size_t bytes_per_token = elements_per_token * element_size;

    total_seq_len += seq_len;

    size_t effective_seq_len;
    bool use_sliding_window = (window_size > 0 && new_total_len > window_size);

    if (use_sliding_window) {
        effective_seq_len = window_size;
    } else {
        effective_seq_len = new_total_len;
    }

    bool any_layer_updated = false;

    for (size_t layer_idx = 0; layer_idx < layers; layer_idx++) {
        auto& cache = layer_caches[layer_idx];

        void* k_output = gb->get_output(k_nodes[layer_idx]);
        void* v_output = gb->get_output(v_nodes[layer_idx]);

        if (k_output && v_output) {
            const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
            const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);

            if (quant_method == KVQuantMethod::TURBOQUANT) {
                size_t k_ang_bytes = turboquant_angles_bytes_per_head(dim, tq_key_angle_bits);
                size_t v_ang_bytes = turboquant_angles_bytes_per_head(dim, tq_value_angle_bits);
                size_t qjl_bytes = turboquant_qjl_bytes_per_head(tq_projection_dim);
                const __fp16* k_fp16 = static_cast<const __fp16*>(k_output);
                const __fp16* v_fp16 = static_cast<const __fp16*>(v_output);
                const uint8_t* rot  = tq_rotation_signs.data();
                const uint8_t* proj = tq_projection_matrix.data();

                if (new_total_len * elements_per_token == k_buffer.total_size &&
                    new_total_len * elements_per_token == v_buffer.total_size) {
                    any_layer_updated = true;
                    if (!use_sliding_window) {
                        tq_resize(cache, new_total_len, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
                        tq_encode_kv(cache, k_fp16, v_fp16, 0, new_total_len, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
                    } else {
                        size_t remaining_window = window_size - sink_size;
                        size_t skip_tokens      = new_total_len - window_size;
                        bool first_slide = (cache.tq_key_radii.size() != window_size * kv_heads);
                        if (first_slide) {
                            tq_resize(cache, window_size, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
                            tq_encode_kv(cache, k_fp16, v_fp16, 0, sink_size, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
                        }
                        size_t src_elem = (sink_size + skip_tokens) * elements_per_token;
                        tq_encode_kv(cache, k_fp16 + src_elem, v_fp16 + src_elem, sink_size, remaining_window, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
                    }
                }
                else if (seq_len * elements_per_token == k_buffer.total_size &&
                         seq_len * elements_per_token == v_buffer.total_size) {
                    any_layer_updated = true;
                    if (!use_sliding_window) {
                        tq_resize(cache, new_total_len, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
                        tq_encode_kv(cache, k_fp16, v_fp16, old_seq_len, seq_len, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
                    } else {
                        if (cache.tq_key_radii.size() != window_size * kv_heads)
                            tq_resize(cache, window_size, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
                        size_t tokens_to_shift = window_size - sink_size - seq_len;
                        if (tokens_to_shift > 0 && old_seq_len > sink_size) {
                            size_t shift_src = old_seq_len - tokens_to_shift;
                            if (shift_src > sink_size)
                                tq_shift(cache, sink_size, shift_src, tokens_to_shift, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes);
                        }
                        tq_encode_kv(cache, k_fp16, v_fp16, window_size - seq_len, seq_len, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
                    }
                }
                continue;
            }

            size_t expected_elements = new_total_len * elements_per_token;

            if (k_buffer.total_size == expected_elements && v_buffer.total_size == expected_elements) {
                any_layer_updated = true;

                if (!use_sliding_window) {
                    size_t total_bytes = new_total_len * bytes_per_token;
                    cache.keys.resize(total_bytes);
                    cache.values.resize(total_bytes);

                    if (precision == Precision::INT8) {
                        size_t num_scales = new_total_len * scales_per_token;
                        cache.key_scales.resize(num_scales);
                        cache.value_scales.resize(num_scales);

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(k_output),
                            reinterpret_cast<int8_t*>(cache.keys.data()),
                            cache.key_scales.data(),
                            new_total_len, kv_heads, dim);

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(v_output),
                            reinterpret_cast<int8_t*>(cache.values.data()),
                            cache.value_scales.data(),
                            new_total_len, kv_heads, dim);
                    } else {
                        std::memcpy(cache.keys.data(), k_output, total_bytes);
                        std::memcpy(cache.values.data(), v_output, total_bytes);
                    }
                } else {
                    size_t cache_bytes = window_size * bytes_per_token;
                    size_t remaining_window = window_size - sink_size;
                    size_t skip_tokens = new_total_len - window_size;

                    bool first_slide = (cache.keys.size() != cache_bytes);

                    if (first_slide) {
                        cache.keys.resize(cache_bytes);
                        cache.values.resize(cache_bytes);
                    }

                    if (precision == Precision::INT8) {
                        size_t num_scales = window_size * scales_per_token;
                        if (first_slide) {
                            cache.key_scales.resize(num_scales);
                            cache.value_scales.resize(num_scales);
                        }

                        const __fp16* k_fp16 = static_cast<const __fp16*>(k_output);
                        const __fp16* v_fp16 = static_cast<const __fp16*>(v_output);

                        if (first_slide) {
                            cactus_quantize_kv_fp16_to_int8(
                                k_fp16,
                                reinterpret_cast<int8_t*>(cache.keys.data()),
                                cache.key_scales.data(),
                                sink_size, kv_heads, dim);

                            cactus_quantize_kv_fp16_to_int8(
                                v_fp16,
                                reinterpret_cast<int8_t*>(cache.values.data()),
                                cache.value_scales.data(),
                                sink_size, kv_heads, dim);
                        }

                        size_t src_offset = (sink_size + skip_tokens) * elements_per_token;
                        size_t dst_offset = sink_size * elements_per_token;
                        size_t scale_dst_offset = sink_size * scales_per_token;

                        cactus_quantize_kv_fp16_to_int8(
                            k_fp16 + src_offset,
                            reinterpret_cast<int8_t*>(cache.keys.data()) + dst_offset,
                            cache.key_scales.data() + scale_dst_offset,
                            remaining_window, kv_heads, dim);

                        cactus_quantize_kv_fp16_to_int8(
                            v_fp16 + src_offset,
                            reinterpret_cast<int8_t*>(cache.values.data()) + dst_offset,
                            cache.value_scales.data() + scale_dst_offset,
                            remaining_window, kv_heads, dim);
                    } else {
                        if (first_slide) {
                            size_t sink_bytes = sink_size * bytes_per_token;
                            std::memcpy(cache.keys.data(), k_output, sink_bytes);
                            std::memcpy(cache.values.data(), v_output, sink_bytes);
                        }

                        const uint8_t* k_src = static_cast<const uint8_t*>(k_output) +
                                              (sink_size + skip_tokens) * bytes_per_token;
                        const uint8_t* v_src = static_cast<const uint8_t*>(v_output) +
                                              (sink_size + skip_tokens) * bytes_per_token;
                        size_t sink_bytes = sink_size * bytes_per_token;
                        size_t recent_bytes = remaining_window * bytes_per_token;

                        std::memcpy(cache.keys.data() + sink_bytes, k_src, recent_bytes);
                        std::memcpy(cache.values.data() + sink_bytes, v_src, recent_bytes);
                    }
                }
            }
            else if (seq_len * elements_per_token == k_buffer.total_size &&
                      seq_len * elements_per_token == v_buffer.total_size) {
                any_layer_updated = true;

                if (!use_sliding_window) {
                    size_t old_bytes = old_seq_len * bytes_per_token;
                    size_t new_bytes = seq_len * bytes_per_token;
                    size_t total_bytes = old_bytes + new_bytes;
                    cache.keys.resize(total_bytes);
                    cache.values.resize(total_bytes);

                    if (precision == Precision::INT8) {
                        size_t num_scales = (old_seq_len + seq_len) * scales_per_token;
                        cache.key_scales.resize(num_scales);
                        cache.value_scales.resize(num_scales);

                        size_t dst_offset = old_seq_len * elements_per_token;
                        size_t scale_offset = old_seq_len * scales_per_token;

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(k_output),
                            reinterpret_cast<int8_t*>(cache.keys.data()) + dst_offset,
                            cache.key_scales.data() + scale_offset,
                            seq_len, kv_heads, dim);

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(v_output),
                            reinterpret_cast<int8_t*>(cache.values.data()) + dst_offset,
                            cache.value_scales.data() + scale_offset,
                            seq_len, kv_heads, dim);
                    } else {
                        std::memcpy(cache.keys.data() + old_bytes, k_output, new_bytes);
                        std::memcpy(cache.values.data() + old_bytes, v_output, new_bytes);
                    }
                } else {
                    size_t cache_bytes = window_size * bytes_per_token;

                    if (cache.keys.size() != cache_bytes) {
                        cache.keys.resize(cache_bytes);
                        cache.values.resize(cache_bytes);
                    }

                    size_t tokens_to_shift = window_size - sink_size - seq_len;
                    size_t sink_bytes = sink_size * bytes_per_token;

                    if (tokens_to_shift > 0 && old_seq_len > sink_size) {
                        size_t shift_src = old_seq_len - tokens_to_shift;
                        if (shift_src > sink_size) {
                            size_t shift_bytes = tokens_to_shift * bytes_per_token;

                            std::memmove(cache.keys.data() + sink_bytes,
                                       cache.keys.data() + shift_src * bytes_per_token,
                                       shift_bytes);
                            std::memmove(cache.values.data() + sink_bytes,
                                       cache.values.data() + shift_src * bytes_per_token,
                                       shift_bytes);

                            if (precision == Precision::INT8) {
                                size_t sink_scale_offset = sink_size * scales_per_token;
                                size_t shift_src_scale_offset = shift_src * scales_per_token;
                                size_t shift_scale_count = tokens_to_shift * scales_per_token;

                                std::memmove(cache.key_scales.data() + sink_scale_offset,
                                           cache.key_scales.data() + shift_src_scale_offset,
                                           shift_scale_count * sizeof(float));
                                std::memmove(cache.value_scales.data() + sink_scale_offset,
                                           cache.value_scales.data() + shift_src_scale_offset,
                                           shift_scale_count * sizeof(float));
                            }
                        }
                    }

                    size_t append_token_offset = window_size - seq_len;
                    size_t append_bytes_offset = append_token_offset * bytes_per_token;
                    size_t new_bytes = seq_len * bytes_per_token;

                    if (precision == Precision::INT8) {
                        size_t append_offset = append_token_offset * elements_per_token;
                        size_t scale_offset = append_token_offset * scales_per_token;

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(k_output),
                            reinterpret_cast<int8_t*>(cache.keys.data()) + append_offset,
                            cache.key_scales.data() + scale_offset,
                            seq_len, kv_heads, dim);

                        cactus_quantize_kv_fp16_to_int8(
                            static_cast<const __fp16*>(v_output),
                            reinterpret_cast<int8_t*>(cache.values.data()) + append_offset,
                            cache.value_scales.data() + scale_offset,
                            seq_len, kv_heads, dim);
                    } else {
                        std::memcpy(cache.keys.data() + append_bytes_offset, k_output, new_bytes);
                        std::memcpy(cache.values.data() + append_bytes_offset, v_output, new_bytes);
                    }
                }
            }
        }
    }

    if (any_layer_updated) {
        current_seq_len = effective_seq_len;
    }
}

void KVCache::update_from_npu(size_t layer_idx, const __fp16* k_data, const __fp16* v_data,
                               size_t num_tokens, size_t kv_heads, size_t dim) {
    if (layer_idx >= num_layers || !k_data || !v_data || num_tokens == 0) {
        return;
    }

    auto& cache = layer_caches[layer_idx];
    size_t old_seq_len = current_seq_len;
    size_t new_total_len = old_seq_len + num_tokens;
    size_t elements_per_token = kv_heads * dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t num_groups = (dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    size_t scales_per_token = kv_heads * num_groups;

    if (layer_idx == 0) {
        total_seq_len += num_tokens;
    }

    bool use_sliding_window = (window_size > 0 && new_total_len > window_size);
    size_t effective_seq_len = use_sliding_window ? window_size : new_total_len;

    if (quant_method == KVQuantMethod::TURBOQUANT) {
        size_t k_ang_bytes = turboquant_angles_bytes_per_head(dim, tq_key_angle_bits);
                size_t v_ang_bytes = turboquant_angles_bytes_per_head(dim, tq_value_angle_bits);
        size_t qjl_bytes = turboquant_qjl_bytes_per_head(tq_projection_dim);
        const uint8_t* rot  = tq_rotation_signs.data();
        const uint8_t* proj = tq_projection_matrix.data();

        if (!use_sliding_window) {
            tq_resize(cache, new_total_len, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
            tq_encode_kv(cache, k_data, v_data, old_seq_len, num_tokens, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
        } else {
            if (cache.tq_key_radii.size() != window_size * kv_heads)
                tq_resize(cache, window_size, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl);
            size_t remaining_window = window_size - sink_size;
            if (num_tokens >= remaining_window) {
                size_t skip = num_tokens - remaining_window;
                tq_encode_kv(cache, k_data + skip * elements_per_token, v_data + skip * elements_per_token,
                             sink_size, remaining_window, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
            } else {
                size_t tokens_to_shift = remaining_window - num_tokens;
                if (tokens_to_shift > 0 && old_seq_len > sink_size) {
                    size_t shift_src = old_seq_len - tokens_to_shift;
                    if (shift_src > sink_size)
                        tq_shift(cache, sink_size, shift_src, tokens_to_shift, kv_heads, k_ang_bytes, v_ang_bytes, qjl_bytes);
                }
                tq_encode_kv(cache, k_data, v_data, window_size - num_tokens, num_tokens, kv_heads, dim, tq_key_angle_bits, tq_value_angle_bits, tq_projection_dim, k_ang_bytes, v_ang_bytes, qjl_bytes, tq_use_qjl, rot, proj);
            }
        }
        if (layer_idx == num_layers - 1)
            current_seq_len = effective_seq_len;
        return;
    }

    if (!use_sliding_window) {
        size_t total_bytes = new_total_len * bytes_per_token;
        cache.keys.resize(total_bytes);
        cache.values.resize(total_bytes);

        size_t num_scales = new_total_len * scales_per_token;
        cache.key_scales.resize(num_scales);
        cache.value_scales.resize(num_scales);

        size_t dst_offset = old_seq_len * elements_per_token;
        size_t scale_offset = old_seq_len * scales_per_token;

        cactus_quantize_kv_fp16_to_int8(
            k_data,
            reinterpret_cast<int8_t*>(cache.keys.data()) + dst_offset,
            cache.key_scales.data() + scale_offset,
            num_tokens, kv_heads, dim);

        cactus_quantize_kv_fp16_to_int8(
            v_data,
            reinterpret_cast<int8_t*>(cache.values.data()) + dst_offset,
            cache.value_scales.data() + scale_offset,
            num_tokens, kv_heads, dim);
    } else {
        size_t cache_bytes = window_size * bytes_per_token;

        if (cache.keys.size() != cache_bytes) {
            cache.keys.resize(cache_bytes);
            cache.values.resize(cache_bytes);

            size_t num_scales = window_size * scales_per_token;
            cache.key_scales.resize(num_scales);
            cache.value_scales.resize(num_scales);
        }

        size_t remaining_window = window_size - sink_size;

        if (num_tokens >= remaining_window) {
            size_t skip_tokens = num_tokens - remaining_window;
            size_t dst_offset = sink_size * elements_per_token;
            size_t scale_offset = sink_size * scales_per_token;

            cactus_quantize_kv_fp16_to_int8(
                k_data + skip_tokens * elements_per_token,
                reinterpret_cast<int8_t*>(cache.keys.data()) + dst_offset,
                cache.key_scales.data() + scale_offset,
                remaining_window, kv_heads, dim);

            cactus_quantize_kv_fp16_to_int8(
                v_data + skip_tokens * elements_per_token,
                reinterpret_cast<int8_t*>(cache.values.data()) + dst_offset,
                cache.value_scales.data() + scale_offset,
                remaining_window, kv_heads, dim);
        } else {
            size_t tokens_to_shift = remaining_window - num_tokens;

            if (tokens_to_shift > 0 && old_seq_len > sink_size) {
                size_t shift_src = old_seq_len - tokens_to_shift;
                if (shift_src > sink_size) {
                    size_t sink_offset = sink_size * elements_per_token;
                    size_t shift_src_offset = shift_src * elements_per_token;
                    size_t shift_bytes = tokens_to_shift * bytes_per_token;

                    std::memmove(cache.keys.data() + sink_offset,
                               cache.keys.data() + shift_src_offset,
                               shift_bytes);
                    std::memmove(cache.values.data() + sink_offset,
                               cache.values.data() + shift_src_offset,
                               shift_bytes);

                    size_t sink_scale_offset = sink_size * scales_per_token;
                    size_t shift_src_scale_offset = shift_src * scales_per_token;
                    size_t shift_scale_count = tokens_to_shift * scales_per_token;

                    std::memmove(cache.key_scales.data() + sink_scale_offset,
                               cache.key_scales.data() + shift_src_scale_offset,
                               shift_scale_count * sizeof(float));
                    std::memmove(cache.value_scales.data() + sink_scale_offset,
                               cache.value_scales.data() + shift_src_scale_offset,
                               shift_scale_count * sizeof(float));
                }
            }

            size_t append_token_offset = window_size - num_tokens;
            size_t append_offset = append_token_offset * elements_per_token;
            size_t scale_offset = append_token_offset * scales_per_token;

            cactus_quantize_kv_fp16_to_int8(
                k_data,
                reinterpret_cast<int8_t*>(cache.keys.data()) + append_offset,
                cache.key_scales.data() + scale_offset,
                num_tokens, kv_heads, dim);

            cactus_quantize_kv_fp16_to_int8(
                v_data,
                reinterpret_cast<int8_t*>(cache.values.data()) + append_offset,
                cache.value_scales.data() + scale_offset,
                num_tokens, kv_heads, dim);
        }
    }

    if (layer_idx == num_layers - 1) {
        current_seq_len = effective_seq_len;
    }
}

const int8_t* KVCache::get_keys_int8(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) {
        return nullptr;
    }
    return reinterpret_cast<const int8_t*>(layer_caches[layer].keys.data());
}

const int8_t* KVCache::get_values_int8(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) {
        return nullptr;
    }
    return reinterpret_cast<const int8_t*>(layer_caches[layer].values.data());
}

const float* KVCache::get_key_scales(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) {
        return nullptr;
    }
    return layer_caches[layer].key_scales.data();
}

const float* KVCache::get_value_scales(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) {
        return nullptr;
    }
    return layer_caches[layer].value_scales.data();
}

const float* KVCache::get_tq_key_radii(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_key_radii.data();
}
const uint8_t* KVCache::get_tq_key_angles(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_key_angles.data();
}
const float* KVCache::get_tq_key_error_norms(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    if (layer_caches[layer].tq_key_error_norms.empty()) return nullptr;
    return layer_caches[layer].tq_key_error_norms.data();
}
const uint8_t* KVCache::get_tq_key_qjl_bits(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    if (layer_caches[layer].tq_key_qjl_bits.empty()) return nullptr;
    return layer_caches[layer].tq_key_qjl_bits.data();
}
const float* KVCache::get_tq_val_radii(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_val_radii.data();
}
const uint8_t* KVCache::get_tq_val_angles(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_val_angles.data();
}
const float* KVCache::get_tq_val_error_norms(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_val_error_norms.data();
}
const uint8_t* KVCache::get_tq_val_qjl_bits(size_t layer) const {
    if (layer >= num_layers || current_seq_len == 0) return nullptr;
    return layer_caches[layer].tq_val_qjl_bits.data();
}

void ConvCache::init(size_t layers, size_t hidden_dim, size_t window_len, Precision model_precision) {
    num_layers = layers;
    hidden_size = hidden_dim;
    window_size = window_len;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    size_t state_bytes = window_size * hidden_size * element_size;
    layer_states.resize(num_layers);
    for (auto& state : layer_states) {
        state.data.resize(state_bytes);
        std::memset(state.data.data(), 0, state_bytes);
        state.head = 0;
        state.count = 0;
    }
}

ConvCache::CircularView ConvCache::get_window(size_t layer) const {
    CircularView view;
    if (layer >= num_layers) {
        view.ptr1 = nullptr;
        view.len1 = 0;
        view.ptr2 = nullptr;
        view.len2 = 0;
        view.total_len = 0;
        return view;
    }

    const auto& state = layer_states[layer];
    if (state.count == 0) {
        view.ptr1 = nullptr;
        view.len1 = 0;
        view.ptr2 = nullptr;
        view.len2 = 0;
        view.total_len = 0;
        return view;
    }

    size_t stride = hidden_size * element_size;

    if (state.count < window_size) {
        view.ptr1 = state.data.data();
        view.len1 = state.count;
        view.ptr2 = nullptr;
        view.len2 = 0;
        view.total_len = state.count;
        return view;
    }

    view.ptr1 = state.data.data();
    view.len1 = state.head;
    view.ptr2 = state.data.data() + state.head * stride;
    view.len2 = window_size - state.head;
    view.total_len = window_size;
    return view;
}

void ConvCache::update(CactusGraph* gb, size_t layer, const size_t bx_node) {
    if (layer >= num_layers || !bx_node || window_size == 0 || hidden_size == 0) {
        return;
    }

    auto& state = layer_states[layer];
    const void* output_ptr = gb->get_output(bx_node);
    if (!output_ptr) {
        return;
    }

    const auto& buffer = gb->get_output_buffer(bx_node);
    const size_t stride_bytes = hidden_size * element_size;

    size_t rows = 1;
    if (!buffer.shape.empty()) {
        if (buffer.shape.size() == 1) {
            rows = 1;
        } else {
            rows = buffer.shape[0];
        }
    }

    if (buffer.total_size > 0 && hidden_size > 0) {
        size_t inferred = buffer.total_size / hidden_size;
        if (inferred > 0) {
            rows = inferred;
        }
    }

    if (rows == 0) {
        return;
    }

    size_t copy_rows = std::min(rows, window_size);
    size_t start_row = rows > window_size ? rows - window_size : 0;

    const uint8_t* src = static_cast<const uint8_t*>(output_ptr) + start_row * stride_bytes;

    for (size_t i = 0; i < copy_rows; ++i) {
        std::memcpy(state.data.data() + state.head * stride_bytes, src + i * stride_bytes, stride_bytes);
        state.head = (state.head + 1) % window_size;
        if (state.count < window_size) {
            ++state.count;
        }
    }
}

void ConvCache::reset() {
    for (auto& state : layer_states) {
        std::fill(state.data.begin(), state.data.end(), 0);
        state.head = 0;
        state.count = 0;
    }
}

}
}