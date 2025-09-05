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
    
    size_t elements_per_token = num_kv_heads * head_dim;
    bytes_per_page = tokens_per_page * elements_per_token * element_size;
    
    size_t max_pages_needed = (max_seq_len + tokens_per_page - 1) / tokens_per_page;
    size_t total_pages = max_pages_needed * num_layers * 2;
    
    page_pool.resize(total_pages);
    for (size_t i = 0; i < total_pages; ++i) {
        page_pool[i].data.resize(bytes_per_page, 0);
        page_pool[i].in_use = false;
        page_pool[i].ref_count = 0;
        free_pages.push_back(i);
    }
    
    layer_tables.resize(num_layers);
    continuous_keys.resize(num_layers);
    continuous_values.resize(num_layers);
    
    size_t continuous_buffer_size = max_seq_len * elements_per_token * element_size;
    for (size_t i = 0; i < num_layers; ++i) {
        continuous_keys[i].resize(continuous_buffer_size, 0);
        continuous_values[i].resize(continuous_buffer_size, 0);
    }
    
    current_seq_len = 0;
}

void KVCache::reset() {
    current_seq_len = 0;
    
    for (auto& table : layer_tables) {
        for (size_t page_idx : table.key_pages) {
            free_page(page_idx);
        }
        for (size_t page_idx : table.value_pages) {
            free_page(page_idx);
        }
        table.key_pages.clear();
        table.value_pages.clear();
    }
    
    for (auto& k : continuous_keys) {
        std::fill(k.begin(), k.end(), 0);
    }
    for (auto& v : continuous_values) {
        std::fill(v.begin(), v.end(), 0);
    }
}

size_t KVCache::allocate_page() {
    if (free_pages.empty()) {
        throw std::runtime_error("KV cache page pool exhausted");
    }
    
    size_t page_idx = free_pages.back();
    free_pages.pop_back();
    
    page_pool[page_idx].in_use = true;
    page_pool[page_idx].ref_count = 1;
    
    return page_idx;
}

void KVCache::free_page(size_t page_idx) {
    if (page_idx >= page_pool.size()) {
        return;
    }
    
    page_pool[page_idx].ref_count--;
    if (page_pool[page_idx].ref_count <= 0) {
        page_pool[page_idx].in_use = false;
        page_pool[page_idx].ref_count = 0;
        std::fill(page_pool[page_idx].data.begin(), page_pool[page_idx].data.end(), 0);
        free_pages.push_back(page_idx);
    }
}

size_t KVCache::get_num_pages_needed(size_t seq_len) const {
    return (seq_len + tokens_per_page - 1) / tokens_per_page;
}

void KVCache::materialize_continuous_buffer(size_t layer) {
    if (layer >= num_layers) return;
    
    const auto& table = layer_tables[layer];
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    
    for (size_t page_idx = 0; page_idx < table.key_pages.size(); ++page_idx) {
        size_t start_token = page_idx * tokens_per_page;
        size_t tokens_in_page = std::min(tokens_per_page, current_seq_len - start_token);
        size_t bytes_to_copy = tokens_in_page * bytes_per_token;
        
        if (start_token < current_seq_len) {
            const auto& key_page = page_pool[table.key_pages[page_idx]];
            const auto& value_page = page_pool[table.value_pages[page_idx]];
            
            std::memcpy(continuous_keys[layer].data() + start_token * bytes_per_token,
                       key_page.data.data(), bytes_to_copy);
            std::memcpy(continuous_values[layer].data() + start_token * bytes_per_token,
                       value_page.data.data(), bytes_to_copy);
        }
    }
}

void* KVCache::get_key_ptr(size_t layer) {
    materialize_continuous_buffer(layer);
    return continuous_keys[layer].data();
}

void* KVCache::get_value_ptr(size_t layer) {
    materialize_continuous_buffer(layer);
    return continuous_values[layer].data();
}

void KVCache::update_from_graph(CactusGraph* gb, const std::vector<size_t>& k_nodes, 
                               const std::vector<size_t>& v_nodes, size_t seq_len, 
                               size_t layers, size_t kv_heads, size_t dim) {
    size_t old_seq_len = current_seq_len;
    size_t new_seq_len = old_seq_len + seq_len;
    size_t elements_per_token = kv_heads * dim;
    size_t bytes_per_token = elements_per_token * element_size;
    
    for (size_t layer_idx = 0; layer_idx < layers; layer_idx++) {
        auto& table = layer_tables[layer_idx];
        
        size_t old_pages = get_num_pages_needed(old_seq_len);
        size_t new_pages = get_num_pages_needed(new_seq_len);
        
        for (size_t i = old_pages; i < new_pages; ++i) {
            table.key_pages.push_back(allocate_page());
            table.value_pages.push_back(allocate_page());
        }
        
        void* k_output = gb->get_output(k_nodes[layer_idx]);
        void* v_output = gb->get_output(v_nodes[layer_idx]);
        
        if (k_output && v_output) {
            const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
            const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);
            
            size_t expected_elements = new_seq_len * elements_per_token;
            
            if (k_buffer.total_size == expected_elements && v_buffer.total_size == expected_elements) {
                uint8_t* k_src = static_cast<uint8_t*>(k_output) + old_seq_len * bytes_per_token;
                uint8_t* v_src = static_cast<uint8_t*>(v_output) + old_seq_len * bytes_per_token;
                
                for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
                    size_t global_token_idx = old_seq_len + token_idx;
                    size_t page_idx = global_token_idx / tokens_per_page;
                    size_t token_in_page = global_token_idx % tokens_per_page;
                    
                    if (page_idx < table.key_pages.size()) {
                        auto& key_page = page_pool[table.key_pages[page_idx]];
                        auto& value_page = page_pool[table.value_pages[page_idx]];
                        
                        size_t page_offset = token_in_page * bytes_per_token;
                        
                        std::memcpy(key_page.data.data() + page_offset,
                                   k_src + token_idx * bytes_per_token,
                                   bytes_per_token);
                        std::memcpy(value_page.data.data() + page_offset,
                                   v_src + token_idx * bytes_per_token,
                                   bytes_per_token);
                    }
                }
            }
        }
    }
    
    current_seq_len = new_seq_len;
}

}
}