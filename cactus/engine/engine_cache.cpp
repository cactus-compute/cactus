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

}
}