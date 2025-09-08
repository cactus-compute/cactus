#include "engine.h"
#include "../graph/graph.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <arm_neon.h>

namespace cactus {
namespace engine {

inline void simd_memcpy(void* dst, const void* src, size_t size) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    
    while (size >= 64) {
        uint8x16x4_t data = vld4q_u8(s);
        vst4q_u8(d, data);
        d += 64;
        s += 64;
        size -= 64;
    }
    
    while (size >= 16) {
        uint8x16_t data = vld1q_u8(s);
        vst1q_u8(d, data);
        d += 16;
        s += 16;
        size -= 16;
    }
    
    if (size > 0) {
        std::memcpy(d, s, size);
    }
}

void KVCache::init(size_t layers, size_t max_seq, size_t kv_heads, size_t dim, Precision model_precision) {
    num_layers = layers;
    max_seq_len = max_seq;
    num_kv_heads = kv_heads;
    head_dim = dim;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);
    
    if (window_size > max_seq_len) {
        window_size = max_seq_len;
    }
    
    size_t total_window = window_size + sink_size;
    if ((total_window & (total_window - 1)) == 0) {
        use_fast_indexing = true;
        buffer_mask = total_window - 1;
    } else {
        use_fast_indexing = false;
        buffer_mask = 0;
    }
    
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t buffer_size = total_window * bytes_per_token;
    
    layer_caches.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layer_caches[i].keys.resize(buffer_size, 0);
        layer_caches[i].values.resize(buffer_size, 0);
        layer_caches[i].start_idx = 0;
        layer_caches[i].end_idx = 0;
        layer_caches[i].total_seen = 0;
    }
    
    continuous_keys.resize(num_layers);
    continuous_values.resize(num_layers);
    
    current_seq_len = 0;
    total_seq_len = 0;
}

void KVCache::set_window_size(size_t window, size_t sink) {
    window_size = window;
    sink_size = sink;
    
    if (window_size > max_seq_len) {
        window_size = max_seq_len;
    }
}

void KVCache::reset() {
    current_seq_len = 0;
    total_seq_len = 0;
    
    for (auto& cache : layer_caches) {
        std::fill(cache.keys.begin(), cache.keys.end(), 0);
        std::fill(cache.values.begin(), cache.values.end(), 0);
        cache.start_idx = 0;
        cache.end_idx = 0;
        cache.total_seen = 0;
    }
    
    continuous_keys.clear();
    continuous_values.clear();
    continuous_keys.resize(num_layers);
    continuous_values.resize(num_layers);
}

size_t KVCache::get_circular_index(size_t logical_idx, size_t start, size_t buffer_size) const {
    if (use_fast_indexing) {
        return (start + logical_idx) & buffer_mask;
    }
    return (start + logical_idx) % buffer_size;
}

void KVCache::slide_window(size_t layer) {
    auto& cache = layer_caches[layer];
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t buffer_capacity = (window_size + sink_size);
    
    if (current_seq_len <= buffer_capacity) {
        return;
    }
    
    size_t tokens_to_remove = current_seq_len - buffer_capacity;
    
    size_t sink_bytes = sink_size * bytes_per_token;
    std::vector<uint8_t> sink_keys(sink_bytes);
    std::vector<uint8_t> sink_values(sink_bytes);
    
    if (cache.start_idx + sink_size <= buffer_capacity) {
        simd_memcpy(sink_keys.data(), 
                   cache.keys.data() + cache.start_idx * bytes_per_token,
                   sink_bytes);
        simd_memcpy(sink_values.data(),
                   cache.values.data() + cache.start_idx * bytes_per_token,
                   sink_bytes);
    } else {
        size_t first_part = buffer_capacity - cache.start_idx;
        size_t first_bytes = first_part * bytes_per_token;
        size_t second_bytes = sink_bytes - first_bytes;
        
        simd_memcpy(sink_keys.data(), cache.keys.data() + cache.start_idx * bytes_per_token, first_bytes);
        simd_memcpy(sink_keys.data() + first_bytes, cache.keys.data(), second_bytes);
        
        simd_memcpy(sink_values.data(), cache.values.data() + cache.start_idx * bytes_per_token, first_bytes);
        simd_memcpy(sink_values.data() + first_bytes, cache.values.data(), second_bytes);
    }
    
    cache.start_idx = (cache.start_idx + tokens_to_remove) % buffer_capacity;
    current_seq_len = buffer_capacity;
    
    size_t new_sink_start = cache.start_idx;
    if (new_sink_start + sink_size <= buffer_capacity) {
        simd_memcpy(cache.keys.data() + new_sink_start * bytes_per_token,
                   sink_keys.data(), sink_bytes);
        simd_memcpy(cache.values.data() + new_sink_start * bytes_per_token,
                   sink_values.data(), sink_bytes);
    } else {
        size_t first_part = buffer_capacity - new_sink_start;
        size_t first_bytes = first_part * bytes_per_token;
        size_t second_bytes = sink_bytes - first_bytes;
        
        simd_memcpy(cache.keys.data() + new_sink_start * bytes_per_token, sink_keys.data(), first_bytes);
        simd_memcpy(cache.keys.data(), sink_keys.data() + first_bytes, second_bytes);
        
        simd_memcpy(cache.values.data() + new_sink_start * bytes_per_token, sink_values.data(), first_bytes);
        simd_memcpy(cache.values.data(), sink_values.data() + first_bytes, second_bytes);
    }
}

void KVCache::copy_to_circular_buffer(LayerCache& cache, const void* new_data, size_t new_tokens, bool is_key) {
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t buffer_capacity = window_size + sink_size;
    
    const uint8_t* src = static_cast<const uint8_t*>(new_data);
    std::vector<uint8_t>& buffer = is_key ? cache.keys : cache.values;
    
    size_t total_bytes = new_tokens * bytes_per_token;
    size_t write_pos = cache.end_idx;
    size_t first_chunk_tokens = std::min(new_tokens, buffer_capacity - write_pos);
    size_t first_chunk_bytes = first_chunk_tokens * bytes_per_token;
    
    simd_memcpy(buffer.data() + write_pos * bytes_per_token, src, first_chunk_bytes);
    
    if (first_chunk_tokens < new_tokens) {
        size_t remaining_bytes = total_bytes - first_chunk_bytes;
        simd_memcpy(buffer.data(), src + first_chunk_bytes, remaining_bytes);
    }
}

void KVCache::materialize_continuous_buffer(size_t layer) {
    if (layer >= num_layers) return;
    
    auto& cache = layer_caches[layer];
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t buffer_capacity = window_size + sink_size;
    
    size_t required_size = current_seq_len * bytes_per_token;
    if (continuous_keys[layer].size() < required_size) {
        continuous_keys[layer].resize(required_size);
        continuous_values[layer].resize(required_size);
    }
    
    size_t wrap_point = buffer_capacity - cache.start_idx;
    
    if (current_seq_len <= wrap_point) {
        size_t total_bytes = current_seq_len * bytes_per_token;
        simd_memcpy(continuous_keys[layer].data(),
                   cache.keys.data() + cache.start_idx * bytes_per_token,
                   total_bytes);
        simd_memcpy(continuous_values[layer].data(),
                   cache.values.data() + cache.start_idx * bytes_per_token,
                   total_bytes);
    } else {
        size_t first_part_bytes = wrap_point * bytes_per_token;
        size_t second_part_bytes = (current_seq_len - wrap_point) * bytes_per_token;
        
        simd_memcpy(continuous_keys[layer].data(),
                   cache.keys.data() + cache.start_idx * bytes_per_token,
                   first_part_bytes);
        simd_memcpy(continuous_keys[layer].data() + first_part_bytes,
                   cache.keys.data(),
                   second_part_bytes);
        
        simd_memcpy(continuous_values[layer].data(),
                   cache.values.data() + cache.start_idx * bytes_per_token,
                   first_part_bytes);
        simd_memcpy(continuous_values[layer].data() + first_part_bytes,
                   cache.values.data(),
                   second_part_bytes);
    }
}

void* KVCache::get_key_ptr(size_t layer) {
    if (current_seq_len == 0) return nullptr;
    
    auto& cache = layer_caches[layer];
    size_t buffer_capacity = window_size + sink_size;
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    
    if (cache.start_idx == 0 || cache.start_idx + current_seq_len <= buffer_capacity) {
        return cache.keys.data() + cache.start_idx * bytes_per_token;
    }
    
    materialize_continuous_buffer(layer);
    return continuous_keys[layer].data();
}

void* KVCache::get_value_ptr(size_t layer) {
    if (current_seq_len == 0) return nullptr;
    
    auto& cache = layer_caches[layer];
    size_t buffer_capacity = window_size + sink_size;
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    
    if (cache.start_idx == 0 || cache.start_idx + current_seq_len <= buffer_capacity) {
        return cache.values.data() + cache.start_idx * bytes_per_token;
    }
    
    materialize_continuous_buffer(layer);
    return continuous_values[layer].data();
}

KVCache::CircularView KVCache::get_circular_view(const LayerCache& cache, bool is_key) const {
    CircularView view;
    view.total_len = current_seq_len;
    
    if (current_seq_len == 0) {
        view.ptr1 = nullptr;
        view.ptr2 = nullptr;
        view.len1 = 0;
        view.len2 = 0;
        return view;
    }
    
    size_t buffer_capacity = window_size + sink_size;
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    const std::vector<uint8_t>& buffer = is_key ? cache.keys : cache.values;
    
    if (cache.start_idx + current_seq_len <= buffer_capacity) {
        view.ptr1 = const_cast<uint8_t*>(buffer.data()) + cache.start_idx * bytes_per_token;
        view.ptr2 = nullptr;
        view.len1 = current_seq_len;
        view.len2 = 0;
    } else {
        size_t first_part_len = buffer_capacity - cache.start_idx;
        size_t second_part_len = current_seq_len - first_part_len;
        
        view.ptr1 = const_cast<uint8_t*>(buffer.data()) + cache.start_idx * bytes_per_token;
        view.ptr2 = const_cast<uint8_t*>(buffer.data());
        view.len1 = first_part_len;
        view.len2 = second_part_len;
    }
    
    return view;
}

KVCache::CircularView KVCache::get_key_view(size_t layer) {
    if (layer >= num_layers) {
        return CircularView{nullptr, nullptr, 0, 0, 0};
    }
    return get_circular_view(layer_caches[layer], true);
}

KVCache::CircularView KVCache::get_value_view(size_t layer) {
    if (layer >= num_layers) {
        return CircularView{nullptr, nullptr, 0, 0, 0};
    }
    return get_circular_view(layer_caches[layer], false);
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
                uint8_t* k_src = static_cast<uint8_t*>(k_output) + old_seq_len * bytes_per_token;
                uint8_t* v_src = static_cast<uint8_t*>(v_output) + old_seq_len * bytes_per_token;
                
                copy_to_circular_buffer(cache, k_src, seq_len, true);
                copy_to_circular_buffer(cache, v_src, seq_len, false);
                
                cache.end_idx = (cache.end_idx + seq_len) % (window_size + sink_size);
                cache.total_seen += seq_len;
                
                if (current_seq_len > window_size + sink_size) {
                    slide_window(layer_idx);
                }
            }
        }
    }
    
    if (current_seq_len > window_size + sink_size) {
        current_seq_len = window_size + sink_size;
    }
}

}
}