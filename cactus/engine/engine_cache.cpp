#include "engine.h"
#include "../graph/graph.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cactus {
namespace engine {

void KVCache::init(size_t layers, size_t max_seq, size_t kv_heads, size_t dim, Precision model_precision) {
    num_layers = layers;
    max_seq_len = max_seq;
    num_kv_heads = kv_heads;
    head_dim = dim;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    layer_caches.resize(num_layers);

    current_seq_len = 0;
    total_seq_len = 0;
}

void KVCache::set_window_size(size_t window, size_t sink) {
    window_size = window;
    sink_size = sink;
}

void KVCache::reset() {
    current_seq_len = 0;
    total_seq_len = 0;

    for (auto& cache : layer_caches) {
        cache.keys.clear();
        cache.values.clear();
    }
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
    size_t new_seq_len = old_seq_len + seq_len;
    size_t elements_per_token = kv_heads * dim;
    size_t bytes_per_token = elements_per_token * element_size;


    total_seq_len += seq_len;
    current_seq_len = new_seq_len;

    for (size_t layer_idx = 0; layer_idx < layers; layer_idx++) {
        auto& cache = layer_caches[layer_idx];

        void* k_output = gb->get_output(k_nodes[layer_idx]);
        void* v_output = gb->get_output(v_nodes[layer_idx]);

        if (k_output && v_output) {
            const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
            const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);

            size_t expected_elements = new_seq_len * elements_per_token;

            if (k_buffer.total_size == expected_elements && v_buffer.total_size == expected_elements) {
                size_t total_bytes = new_seq_len * bytes_per_token;

                cache.keys.resize(total_bytes);
                cache.values.resize(total_bytes);

                std::memcpy(cache.keys.data(), k_output, total_bytes);
                std::memcpy(cache.values.data(), v_output, total_bytes);
            }
        }
    }

}

void ConvCache::init(size_t layers, size_t hidden_dim, size_t window_len, Precision model_precision) {
    num_layers = layers;
    hidden_size = hidden_dim;
    window_size = window_len;  
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    layer_states.resize(num_layers);
    for (auto& state : layer_states) {
        state.data.assign(window_size * hidden_size * element_size, 0);  
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