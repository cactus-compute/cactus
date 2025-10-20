#include "engine.h"
#include "../graph/graph.h"
#include "../kernel/kernel_utils.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <atomic>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>

namespace cactus {
namespace engine {

namespace fs = std::filesystem;

static void* create_mmap_file(const std::string& filepath, size_t size, int& fd) {
    fs::path dir = fs::path(filepath).parent_path();
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    fd = open(filepath.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        throw std::runtime_error("Failed to open cache file: " + filepath);
    }

    if (ftruncate(fd, size) != 0) {
        close(fd);
        throw std::runtime_error("Failed to resize cache file: " + filepath);
    }

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap cache file: " + filepath);
    }

    return ptr;
}

static void cleanup_mmap(void*& ptr, int& fd, size_t size) {
    if (ptr && ptr != MAP_FAILED) {
        munmap(ptr, size);
        ptr = nullptr;
    }
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

static inline void optimized_memcpy(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

static void sliding_window_copy(void* dst, const void* src,
                               size_t sink_bytes, size_t window_bytes, size_t skip_bytes) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);

    if (sink_bytes > 0) {
        std::memcpy(d, s, sink_bytes);
        d += sink_bytes;
    }

    s += skip_bytes;
    std::memcpy(d, s, window_bytes);
}

void KVCache::init(size_t layers, size_t max_seq, size_t kv_heads, size_t dim, Precision model_precision) {
    num_layers = layers;
    max_seq_len = max_seq;
    num_kv_heads = kv_heads;
    head_dim = dim;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    layer_caches.resize(num_layers);

    size_t max_cache_tokens = (window_size > 0 && window_size < max_seq_len) ? window_size : max_seq_len;
    size_t elements_per_token = num_kv_heads * head_dim;
    size_t bytes_per_token = elements_per_token * element_size;
    size_t buffer_size = max_cache_tokens * bytes_per_token;

    for (size_t i = 0; i < num_layers; i++) {
        auto& cache = layer_caches[i];

        if (cache.keys_mmap) {
            cleanup_mmap(cache.keys_mmap, cache.keys_fd, cache.mmap_size);
            cleanup_mmap(cache.values_mmap, cache.values_fd, cache.mmap_size);
        }

        std::string keys_file = cache_dir + "/layer_" + std::to_string(i) + "_keys.bin";
        std::string values_file = cache_dir + "/layer_" + std::to_string(i) + "_values.bin";

        cache.keys_mmap = create_mmap_file(keys_file, buffer_size, cache.keys_fd);
        cache.values_mmap = create_mmap_file(values_file, buffer_size, cache.values_fd);
        cache.mmap_size = buffer_size;
        cache.start_idx = 0;
        cache.cache_len = 0;
    }

    max_cache_size = max_cache_tokens;
    current_seq_len = 0;
    total_seq_len = 0;
    cache_start_pos = 0;
}

void KVCache::set_window_size(size_t window, size_t sink) {
    window_size = window;
    sink_size = sink;
}

void KVCache::set_cache_dir(const std::string& dir) {
    cache_dir = dir;
}

void KVCache::reset() {
    current_seq_len = 0;
    total_seq_len = 0;
    cache_start_pos = 0;

    for (auto& cache : layer_caches) {
        cache.start_idx = 0;
        cache.cache_len = 0;
    }
}

void* KVCache::get_key_ptr(size_t layer) {
    if (current_seq_len == 0 || layer >= num_layers) return nullptr;
    return layer_caches[layer].keys_mmap;
}

void* KVCache::get_value_ptr(size_t layer) {
    if (current_seq_len == 0 || layer >= num_layers) return nullptr;
    return layer_caches[layer].values_mmap;
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

    view.ptr1 = layer_caches[layer].keys_mmap;
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

    view.ptr1 = layer_caches[layer].values_mmap;
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

    size_t effective_max_cache = (window_size > 0 && window_size < max_seq_len) ? window_size : max_seq_len;

    if (seq_len == 1 && old_seq_len > 0 && new_seq_len <= effective_max_cache) {
        size_t append_bytes = bytes_per_token;

        CactusThreading::parallel_for(layers, CactusThreading::Thresholds::ELEMENT_WISE,
            [&](size_t start_layer, size_t end_layer) {
                for (size_t layer_idx = start_layer; layer_idx < end_layer; layer_idx++) {
                    auto& cache = layer_caches[layer_idx];

                    void* k_output = gb->get_output(k_nodes[layer_idx]);
                    void* v_output = gb->get_output(v_nodes[layer_idx]);

                    if (k_output && v_output) {
                        uint8_t* k_new = static_cast<uint8_t*>(k_output) + old_seq_len * bytes_per_token;
                        uint8_t* v_new = static_cast<uint8_t*>(v_output) + old_seq_len * bytes_per_token;

                        uint8_t* k_dst = static_cast<uint8_t*>(cache.keys_mmap) + old_seq_len * bytes_per_token;
                        uint8_t* v_dst = static_cast<uint8_t*>(cache.values_mmap) + old_seq_len * bytes_per_token;
                        optimized_memcpy(k_dst, k_new, append_bytes);
                        optimized_memcpy(v_dst, v_new, append_bytes);

                        cache.cache_len = new_seq_len;
                    }
                }
            });

        current_seq_len = new_seq_len;
        return;
    }

    std::atomic<size_t> first_layer_skip{0};

    CactusThreading::parallel_for(layers, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_layer, size_t end_layer) {
            for (size_t layer_idx = start_layer; layer_idx < end_layer; layer_idx++) {
                auto& cache = layer_caches[layer_idx];

                void* k_output = gb->get_output(k_nodes[layer_idx]);
                void* v_output = gb->get_output(v_nodes[layer_idx]);

                if (k_output && v_output) {
                    const auto& k_buffer = gb->get_output_buffer(k_nodes[layer_idx]);
                    const auto& v_buffer = gb->get_output_buffer(v_nodes[layer_idx]);

                    size_t expected_elements = new_seq_len * elements_per_token;

                    if (k_buffer.total_size == expected_elements && v_buffer.total_size == expected_elements) {

                        if (new_seq_len <= effective_max_cache) {
                            size_t total_bytes = new_seq_len * bytes_per_token;

                            optimized_memcpy(cache.keys_mmap, k_output, total_bytes);
                            optimized_memcpy(cache.values_mmap, v_output, total_bytes);

                            cache.cache_len = new_seq_len;
                            cache.start_idx = 0;
                        } else if (window_size > 0) {
                            size_t sink_bytes = sink_size * bytes_per_token;
                            size_t window_tokens = window_size - sink_size;
                            size_t window_bytes = window_tokens * bytes_per_token;

                            size_t tokens_to_skip = std::max(sink_size, new_seq_len - window_tokens);

                            sliding_window_copy(cache.keys_mmap, k_output,
                                              sink_bytes, window_bytes, tokens_to_skip * bytes_per_token);
                            sliding_window_copy(cache.values_mmap, v_output,
                                              sink_bytes, window_bytes, tokens_to_skip * bytes_per_token);

                            cache.cache_len = window_size;
                            cache.start_idx = 0;

                            if (layer_idx == 0) {
                                first_layer_skip.store(tokens_to_skip);
                            }
                        } else {
                            size_t tokens_to_keep = effective_max_cache;
                            size_t tokens_to_skip = new_seq_len - tokens_to_keep;
                            size_t total_bytes = tokens_to_keep * bytes_per_token;

                            uint8_t* k_src = static_cast<uint8_t*>(k_output) + tokens_to_skip * bytes_per_token;
                            uint8_t* v_src = static_cast<uint8_t*>(v_output) + tokens_to_skip * bytes_per_token;

                            optimized_memcpy(cache.keys_mmap, k_src, total_bytes);
                            optimized_memcpy(cache.values_mmap, v_src, total_bytes);

                            cache.cache_len = tokens_to_keep;
                            cache.start_idx = 0;

                            if (layer_idx == 0) {
                                first_layer_skip.store(total_seq_len - tokens_to_keep);
                            }
                        }
                    }
                }
            }
        });

    if (first_layer_skip.load() > 0) {
        cache_start_pos = first_layer_skip.load();
    }

    current_seq_len = std::min(new_seq_len, effective_max_cache);
}

}
}