#include "engine.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cactus {
namespace engine {

void KVCache::init(size_t layers, size_t max_seq, size_t kv_heads, const std::vector<size_t>& layer_dims, Precision model_precision) {
    num_layers = layers;
    max_seq_len = max_seq;
    num_kv_heads = kv_heads;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    layer_caches.resize(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        layer_caches[i].head_dim = layer_dims[i];
    }

    current_seq_len = 0;
    total_seq_len = 0;
}

void KVCache::set_window_size(size_t window, size_t sink) {
    window_size = window;
    sink_size = sink;

    if (num_kv_heads > 0 && window_size > 0) {
        for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            size_t head_dim = get_layer_head_dim(layer_idx);
            if (head_dim == 0) continue;
            size_t cache_bytes = window_size * num_kv_heads * head_dim * element_size;
            size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
            size_t num_scales = window_size * num_kv_heads * num_groups;
            auto& cache = layer_caches[layer_idx];

            cache.keys.resize(cache_bytes);
            cache.values.resize(cache_bytes);
            std::memset(cache.keys.data(), 0, cache_bytes);
            std::memset(cache.values.data(), 0, cache_bytes);

            cache.key_scales.resize(num_scales);
            cache.value_scales.resize(num_scales);
            std::fill(cache.key_scales.begin(), cache.key_scales.end(), 1.0f);
            std::fill(cache.value_scales.begin(), cache.value_scales.end(), 1.0f);
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
                               size_t layers, size_t kv_heads) {
    size_t old_seq_len = current_seq_len;
    size_t new_total_len = old_seq_len + seq_len;

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
        if (k_nodes[layer_idx] == 0 || v_nodes[layer_idx] == 0)
            continue;

        auto& cache = layer_caches[layer_idx];

        void* k_output = gb->get_output(k_nodes[layer_idx]);
        void* v_output = gb->get_output(v_nodes[layer_idx]);

        if (k_output && v_output) {
            const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
            const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);

            size_t dim = get_layer_head_dim(layer_idx);
            size_t elements_per_token = kv_heads * dim;
            size_t num_groups = (dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
            size_t scales_per_token = kv_heads * num_groups;
            size_t bytes_per_token = elements_per_token * element_size;

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

void KVCache::remove_token_range(size_t start, size_t count) {
    if (count == 0 || start >= current_seq_len || start + count > current_seq_len) return;

    size_t tail_tokens = current_seq_len - start - count;

    auto erase_bytes = [](std::vector<uint8_t>& buf, size_t offset, size_t remove, size_t tail) {
        std::memmove(buf.data() + offset, buf.data() + offset + remove, tail);
        buf.resize(buf.size() - remove);
    };

    auto erase_floats = [](std::vector<float>& buf, size_t offset, size_t remove, size_t tail) {
        std::memmove(buf.data() + offset, buf.data() + offset + remove, tail * sizeof(float));
        buf.resize(buf.size() - remove);
    };

    for (size_t i = 0; i < layer_caches.size(); i++) {
        size_t dim = get_layer_head_dim(i);
        if (dim == 0) continue;
        auto& layer = layer_caches[i];
        size_t bytes_per_tok = num_kv_heads * dim * element_size;

        erase_bytes(layer.keys,   start * bytes_per_tok, count * bytes_per_tok, tail_tokens * bytes_per_tok);
        erase_bytes(layer.values, start * bytes_per_tok, count * bytes_per_tok, tail_tokens * bytes_per_tok);

        if (precision == Precision::INT8 && !layer.key_scales.empty()) {
            size_t scaler_per_tok = num_kv_heads * dim / KV_QUANT_GROUP_SIZE;
            erase_floats(layer.key_scales, start * scaler_per_tok, count * scaler_per_tok, tail_tokens * scaler_per_tok);
            erase_floats(layer.value_scales, start * scaler_per_tok, count * scaler_per_tok, tail_tokens * scaler_per_tok);
        }
    }

    current_seq_len -= count;
    total_seq_len -= count;
}

void KVCache::compact_to_windows(const std::vector<size_t>& target_windows) {
    for (size_t i = 0; i < layer_caches.size(); i++) {
        size_t dim = get_layer_head_dim(i);
        if (dim == 0) continue;
        size_t target = (i < target_windows.size()) ? target_windows[i] : 0;
        if (target == 0) continue;

        auto& layer = layer_caches[i];
        size_t layer_tokens = layer.keys.size() / (num_kv_heads * dim * element_size);
        if (layer_tokens <= target) continue;

        size_t bytes_per_token = num_kv_heads * dim * element_size;
        size_t keep_bytes = target * bytes_per_token;
        size_t discard_bytes = (layer_tokens - target) * bytes_per_token;

        std::memmove(layer.keys.data(), layer.keys.data() + discard_bytes, keep_bytes);
        layer.keys.resize(keep_bytes);
        std::memmove(layer.values.data(), layer.values.data() + discard_bytes, keep_bytes);
        layer.values.resize(keep_bytes);

        if (precision == Precision::INT8 && !layer.key_scales.empty()) {
            size_t num_groups = (dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
            size_t scales_per_token = num_kv_heads * num_groups;
            size_t keep_scales = target * scales_per_token;
            size_t discard_scales = (layer_tokens - target) * scales_per_token;

            std::memmove(layer.key_scales.data(), layer.key_scales.data() + discard_scales, keep_scales * sizeof(float));
            layer.key_scales.resize(keep_scales);
            std::memmove(layer.value_scales.data(), layer.value_scales.data() + discard_scales, keep_scales * sizeof(float));
            layer.value_scales.resize(keep_scales);
        }
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