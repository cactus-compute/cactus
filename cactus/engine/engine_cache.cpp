#include "engine.h"
#include "../graph/graph.h"
#include <algorithm>
#include <cstring>

namespace cactus {
namespace engine {

void KVCache::init(size_t num_layers, size_t max_seq, size_t num_kv_heads, size_t head_dim, Precision model_precision) {
    max_seq_len = max_seq;
    keys.resize(num_layers);
    values.resize(num_layers);
    
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);
    
    size_t num_elements = 1 * max_seq_len * num_kv_heads * head_dim;
    size_t cache_byte_size = num_elements * element_size;
    for (size_t i = 0; i < num_layers; ++i) {
        keys[i].resize(cache_byte_size, 0);
        values[i].resize(cache_byte_size, 0);
    }
    current_seq_len = 0;
}

void KVCache::reset() {
    current_seq_len = 0;
    for (auto& k : keys) {
        std::fill(k.begin(), k.end(), 0);
    }
    for (auto& v : values) {
        std::fill(v.begin(), v.end(), 0);
    }
}

void KVCache::update_from_graph(CactusGraph* gb, const std::vector<size_t>& k_nodes, 
                               const std::vector<size_t>& v_nodes, size_t seq_len, 
                               size_t num_layers, size_t kv_heads, size_t head_dim) {
    size_t total_kv_seq = current_seq_len + seq_len;
    size_t kv_elements_per_seq = kv_heads * head_dim;
    size_t new_elements = seq_len * kv_elements_per_seq;
    size_t new_byte_size = new_elements * element_size;
    
    for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        void* k_output = gb->get_output(k_nodes[layer_idx]);
        void* v_output = gb->get_output(v_nodes[layer_idx]);
        
        if (k_output && v_output) {
            const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
            const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);
            
            size_t expected_elements = total_kv_seq * kv_elements_per_seq;
            
            if (k_buffer.total_size == expected_elements && v_buffer.total_size == expected_elements) {
                size_t cache_offset = current_seq_len * kv_elements_per_seq * element_size;
                size_t graph_offset = current_seq_len * kv_elements_per_seq * element_size;
                
                uint8_t* k_cache_ptr = static_cast<uint8_t*>(get_key_ptr(layer_idx)) + cache_offset;
                uint8_t* v_cache_ptr = static_cast<uint8_t*>(get_value_ptr(layer_idx)) + cache_offset;
                uint8_t* k_graph_ptr = static_cast<uint8_t*>(k_output) + graph_offset;
                uint8_t* v_graph_ptr = static_cast<uint8_t*>(v_output) + graph_offset;
                
                std::memcpy(k_cache_ptr, k_graph_ptr, new_byte_size);
                std::memcpy(v_cache_ptr, v_graph_ptr, new_byte_size);
            }
        }
    }
    current_seq_len = total_kv_seq;
}

}
}