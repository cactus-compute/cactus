#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <cstring>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cassert>
#include <stdexcept>

namespace {

struct ConvCacheMetadata {
    uint64_t head;
    uint64_t count;
    uint64_t window_size;
    uint64_t hidden_dim;
    uint64_t reserved[4];
};

static_assert(sizeof(ConvCacheMetadata) == 64, "ConvCacheMetadata must be 64 bytes");

inline ConvCacheMetadata* get_conv_meta(BufferDesc& buf) {
    return static_cast<ConvCacheMetadata*>(buf.get_data());
}

inline __fp16* get_conv_data(BufferDesc& buf) {
    return reinterpret_cast<__fp16*>(static_cast<char*>(buf.get_data()) + sizeof(ConvCacheMetadata));
}

struct CacheMetadata {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t num_slots;
    uint64_t reserved[2];
};

static_assert(sizeof(CacheMetadata) == 64, "CacheMetadata must be 64 bytes");

inline size_t cache_buffer_size(size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    return sizeof(CacheMetadata) + max_seq * kv_heads * head_dim + max_seq * kv_heads * num_groups * sizeof(float);
}

inline size_t fp16_cache_elements(size_t max_seq, size_t kv_heads, size_t head_dim) {
    return (sizeof(CacheMetadata) / sizeof(__fp16)) + max_seq * kv_heads * head_dim;
}

inline size_t kv_slot_off(const BufferDesc& buf, size_t slot) {
    if (slot == 0) return 0;
    const CacheMetadata* m = reinterpret_cast<const CacheMetadata*>(buf.get_data());
    size_t per_slot = (buf.precision == Precision::FP16)
        ? fp16_cache_elements(m->max_seq_len, m->num_kv_heads, m->head_dim) * sizeof(__fp16)
        : cache_buffer_size(m->max_seq_len, m->num_kv_heads, m->head_dim);
    return slot * per_slot;
}

inline CacheMetadata* get_meta(BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<CacheMetadata*>(static_cast<char*>(buf.get_data()) + kv_slot_off(buf, slot));
}

inline const CacheMetadata* get_meta(const BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<const CacheMetadata*>(static_cast<const char*>(buf.get_data()) + kv_slot_off(buf, slot));
}

inline int8_t* get_int8_data(BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<int8_t*>(static_cast<char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata));
}

inline const int8_t* get_int8_data(const BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<const int8_t*>(static_cast<const char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata));
}

inline float* get_scales(BufferDesc& buf, size_t max_seq, size_t kv_heads, size_t head_dim, size_t slot = 0) {
    size_t int8_bytes = max_seq * kv_heads * head_dim;
    return reinterpret_cast<float*>(static_cast<char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata) + int8_bytes);
}

inline const float* get_scales(const BufferDesc& buf, size_t max_seq, size_t kv_heads, size_t head_dim, size_t slot = 0) {
    size_t int8_bytes = max_seq * kv_heads * head_dim;
    return reinterpret_cast<const float*>(static_cast<const char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata) + int8_bytes);
}

inline __fp16* get_fp16_data(BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<__fp16*>(static_cast<char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata));
}

inline const __fp16* get_fp16_data(const BufferDesc& buf, size_t slot = 0) {
    return reinterpret_cast<const __fp16*>(static_cast<const char*>(buf.get_data()) + kv_slot_off(buf, slot) + sizeof(CacheMetadata));
}

inline bool use_fp16_kv_cache() {
    static const bool cached = [] {
        const char* value = std::getenv("CACTUS_KV_CACHE_FP16");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return cached;
}

constexpr size_t kInitialCacheEntries = 256;

inline bool kv_cache_resident() {
    return cactus_backend_metal();
}

inline bool resize_cache_buffer(BufferDesc& buf, size_t new_max) {
    auto* meta = get_meta(buf);
    size_t cur = meta->max_seq_len;
    const size_t current_seq = meta->current_seq_len;
    if (new_max == cur || new_max < current_seq) return false;

    const size_t kv_heads = meta->num_kv_heads;
    const size_t hdim = meta->head_dim;
    const bool fp16_cache = buf.precision == Precision::FP16;

    size_t total = fp16_cache ? fp16_cache_elements(new_max, kv_heads, hdim)
                              : cache_buffer_size(new_max, kv_heads, hdim);
    BufferDesc resized({total}, fp16_cache ? Precision::FP16 : Precision::INT8);
    const bool metal = kv_cache_resident();
    void* old_data = buf.get_data();
    if (metal) {
        cactus_metal_session_sync();
        void* p = cactus_metal_alloc_shared(resized.byte_size);
        if (p) resized.set_external(p); else resized.allocate();
    } else {
        resized.allocate();
    }
    std::memset(resized.get_data(), 0, resized.byte_size);

    std::memcpy(resized.get_data(), buf.get_data(), sizeof(CacheMetadata));
    if (fp16_cache) {
        std::memcpy(get_fp16_data(resized), get_fp16_data(buf),
                    current_seq * kv_heads * hdim * sizeof(__fp16));
    } else {
        std::memcpy(get_int8_data(resized), get_int8_data(buf), current_seq * kv_heads * hdim);
        const size_t groups = (hdim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
        std::memcpy(get_scales(resized, new_max, kv_heads, hdim),
                    get_scales(buf, cur, kv_heads, hdim),
                    current_seq * kv_heads * groups * sizeof(float));
    }
    get_meta(resized)->max_seq_len = new_max;
    buf = std::move(resized);
    cactus_metal_free_shared(old_data);
    return true;
}

inline bool grow_cache_buffer(BufferDesc& buf, size_t needed, size_t ceiling) {
    size_t cur = get_meta(buf)->max_seq_len;
    if (needed <= cur || cur >= ceiling) return false;
    size_t new_max = cur;
    while (new_max < needed) new_max <<= 1;
    if (new_max > ceiling) new_max = ceiling;
    if (new_max <= cur) return false;
    return resize_cache_buffer(buf, new_max);
}

} // namespace

bool cactus_kv_cache_grow(BufferDesc& buf, size_t needed, size_t ceiling) {
    return grow_cache_buffer(buf, needed, ceiling);
}

void compute_kv_cache_state_node(
    GraphNode& node,
    const nodes_vector&,
    const node_index_map_t&) {

    if (node.output_buffer.get_data()) return;

    size_t ceiling = node.params.max_cache_seq_len;
    size_t window = node.params.window_size;
    size_t num_slots = node.params.cache_num_slots > 0 ? node.params.cache_num_slots : 1;
    bool sliding = window > 0 && window < ceiling;
    size_t max_seq;
    if (sliding) max_seq = std::min(ceiling, window + node.params.cache_sink_size + 1);
    else if (num_slots > 1 || window > 0) max_seq = ceiling;
    else max_seq = std::min(ceiling, kInitialCacheEntries);
    size_t kv_heads = node.params.num_kv_heads;
    size_t hdim = node.params.head_dim;
    const bool fp16_cache = use_fp16_kv_cache();
    size_t per_slot = fp16_cache
        ? fp16_cache_elements(max_seq, kv_heads, hdim)
        : cache_buffer_size(max_seq, kv_heads, hdim);

    node.output_buffer = BufferDesc({num_slots * per_slot}, fp16_cache ? Precision::FP16 : Precision::INT8);
    if (kv_cache_resident()) {
        void* p = cactus_metal_alloc_shared(node.output_buffer.byte_size);
        if (p) node.output_buffer.set_external(p); else node.output_buffer.allocate();
    } else {
        node.output_buffer.allocate();
    }
    std::memset(node.output_buffer.get_data(), 0, node.output_buffer.byte_size);

    auto* meta0 = get_meta(node.output_buffer, 0);
    meta0->current_seq_len = 0;
    meta0->max_seq_len = max_seq;
    meta0->num_kv_heads = kv_heads;
    meta0->head_dim = hdim;
    meta0->sink_size = node.params.cache_sink_size;
    meta0->num_slots = num_slots;
    for (size_t s = 1; s < num_slots; ++s) {
        *get_meta(node.output_buffer, s) = *meta0;
    }
}

void kv_append_one_slot(BufferDesc& cache_buf, size_t slot, size_t num_slots,
                        const __fp16* source, size_t new_seq_len,
                        size_t window_size, size_t ceiling) {
    auto* meta = get_meta(cache_buf, slot);
    size_t current_len = meta->current_seq_len;
    size_t max_len = meta->max_seq_len;
    size_t kv_heads = meta->num_kv_heads;
    size_t hdim = meta->head_dim;
    size_t sink = meta->sink_size;
    size_t num_groups = (hdim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    size_t int8_stride = kv_heads * hdim;
    size_t scale_stride = kv_heads * num_groups;
    bool sliding = window_size > 0 && window_size < ceiling;

    if (num_slots <= 1 && !sliding && current_len + new_seq_len > max_len) {
        if (grow_cache_buffer(cache_buf, current_len + new_seq_len, ceiling)) {
            meta = get_meta(cache_buf, slot);
            max_len = meta->max_seq_len;
        }
    }

    if (cache_buf.precision == Precision::FP16) {
        size_t stride = kv_heads * hdim;
        __fp16* fp16_base = get_fp16_data(cache_buf, slot);
        size_t window = sliding ? window_size : max_len;
        size_t new_total = current_len + new_seq_len;
        if (new_total > window) {
            size_t keep_sink = std::min({sink, current_len, window});
            size_t tail_capacity = window - keep_sink;
            if (new_seq_len >= tail_capacity) {
                if (tail_capacity > 0) {
                    size_t source_offset = new_seq_len - tail_capacity;
                    std::memcpy(fp16_base + keep_sink * stride, source + source_offset * stride,
                                tail_capacity * stride * sizeof(__fp16));
                }
                meta->current_seq_len = keep_sink + tail_capacity;
                return;
            }
            size_t remaining = std::min(tail_capacity - new_seq_len, current_len - keep_sink);
            size_t shift_src = current_len - remaining;
            if (remaining > 0 && shift_src > keep_sink) {
                std::memmove(fp16_base + keep_sink * stride, fp16_base + shift_src * stride,
                             remaining * stride * sizeof(__fp16));
            }
            size_t append_offset = keep_sink + remaining;
            std::memcpy(fp16_base + append_offset * stride, source, new_seq_len * stride * sizeof(__fp16));
            meta->current_seq_len = append_offset + new_seq_len;
        } else {
            std::memcpy(fp16_base + current_len * stride, source, new_seq_len * stride * sizeof(__fp16));
            meta->current_seq_len = new_total;
        }
        return;
    }

    int8_t* int8_base = get_int8_data(cache_buf, slot);
    float* scale_base = get_scales(cache_buf, max_len, kv_heads, hdim, slot);
    size_t window = sliding ? window_size : max_len;
    size_t new_total = current_len + new_seq_len;
    if (new_total > window) {
        size_t keep_sink = std::min({sink, current_len, window});
        size_t tail_capacity = window - keep_sink;
        if (new_seq_len >= tail_capacity) {
            if (tail_capacity > 0) {
                size_t source_offset = new_seq_len - tail_capacity;
                cactus_quantize_kv_fp16_to_int8(source + source_offset * int8_stride,
                    int8_base + keep_sink * int8_stride, scale_base + keep_sink * scale_stride,
                    tail_capacity, kv_heads, hdim);
            }
            meta->current_seq_len = keep_sink + tail_capacity;
            return;
        }
        size_t remaining = std::min(tail_capacity - new_seq_len, current_len - keep_sink);
        size_t shift_src = current_len - remaining;
        if (remaining > 0 && shift_src > keep_sink) {
            std::memmove(int8_base + keep_sink * int8_stride, int8_base + shift_src * int8_stride,
                         remaining * int8_stride);
            std::memmove(scale_base + keep_sink * scale_stride, scale_base + shift_src * scale_stride,
                         remaining * scale_stride * sizeof(float));
        }
        size_t append_offset = keep_sink + remaining;
        cactus_quantize_kv_fp16_to_int8(source, int8_base + append_offset * int8_stride,
            scale_base + append_offset * scale_stride, new_seq_len, kv_heads, hdim);
        meta->current_seq_len = append_offset + new_seq_len;
    } else {
        cactus_quantize_kv_fp16_to_int8(source, int8_base + current_len * int8_stride,
            scale_base + current_len * scale_stride, new_seq_len, kv_heads, hdim);
        meta->current_seq_len = new_total;
    }
}

void compute_kv_cache_append_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& new_kv = get_input(node, 0, nodes, node_index_map);
    auto& cache_buf = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    size_t slot = node.params.cache_slot;
    const auto* m0 = get_meta(cache_buf, 0);
    size_t num_slots = m0->num_slots ? m0->num_slots : 1;
    size_t kv_heads = m0->num_kv_heads;
    size_t hdim = m0->head_dim;
    size_t ceiling = nodes[node_index_map.at(node.input_ids[1])]->params.max_cache_seq_len;
    size_t window_size = node.params.window_size;
    const __fp16* src = new_kv.data_as<__fp16>();

    size_t batch = new_kv.shape.empty() ? 1 : new_kv.shape[0];
    if (num_slots > 1 && batch > 1 && batch <= num_slots) {
        size_t row_elems = new_kv.total_size / batch;
        size_t per_row_tokens = row_elems / (kv_heads * hdim);
        for (size_t i = 0; i < batch; ++i) {
            kv_append_one_slot(cache_buf, i, num_slots, src + i * row_elems, per_row_tokens, window_size, ceiling);
        }
        *node.output_buffer.data_as<float>() = static_cast<float>(get_meta(cache_buf, batch - 1)->current_seq_len);
        return;
    }

    size_t new_seq_len = new_kv.total_size / (kv_heads * hdim);
    kv_append_one_slot(cache_buf, slot, num_slots, src, new_seq_len, window_size, ceiling);
    *node.output_buffer.data_as<float>() = static_cast<float>(get_meta(cache_buf, slot)->current_seq_len);
}

void compute_attention_cached_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& query_buf = get_input(node, 0, nodes, node_index_map);
    const auto& key_new_buf = get_input(node, 1, nodes, node_index_map);
    const auto& val_new_buf = get_input(node, 2, nodes, node_index_map);
    const auto& k_cache_buf = get_input(node, 3, nodes, node_index_map);
    const auto& v_cache_buf = get_input(node, 4, nodes, node_index_map);

    size_t slot = node.params.cache_slot;
    const auto* k_meta = get_meta(k_cache_buf, slot);
    size_t num_slots = k_meta->num_slots ? k_meta->num_slots : 1;
    size_t cache_len = k_meta->current_seq_len;
    size_t k_max = k_meta->max_seq_len;
    size_t kv_heads = k_meta->num_kv_heads;
    size_t hdim = k_meta->head_dim;

    const auto* v_meta = get_meta(v_cache_buf, slot);
    size_t v_hdim = node.params.v_head_dim > 0 ? node.params.v_head_dim : hdim;
    size_t v_max = v_meta->max_seq_len;

    const int8_t* cached_keys = get_int8_data(k_cache_buf, slot);
    const float* k_scales = get_scales(k_cache_buf, k_max, kv_heads, hdim, slot);
    const int8_t* cached_values = get_int8_data(v_cache_buf, slot);
    const float* v_scales = get_scales(v_cache_buf, v_max, kv_heads, v_hdim, slot);

    const auto& q_shape = query_buf.shape;
    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t num_q_heads = q_shape[2];

    size_t new_seq_len = key_new_buf.total_size / (kv_heads * hdim);
    size_t history_len = (cache_len >= new_seq_len) ? cache_len - new_seq_len : 0;
    bool cache_only_attention = false;
    size_t position_offset = node.params.position_offset;
    if (position_offset == std::numeric_limits<size_t>::max()) {
        position_offset = history_len;
    } else if (position_offset == std::numeric_limits<size_t>::max() - 1) {
        position_offset = (cache_len >= seq_len) ? cache_len - seq_len : 0;
        history_len = cache_len;
        new_seq_len = 0;
        cache_only_attention = true;
    }

    if (batch_size > 1 && num_slots > 1) {
        bool fp16_cache = (k_cache_buf.precision == Precision::FP16 || v_cache_buf.precision == Precision::FP16);
        size_t q_stride = query_buf.total_size / batch_size;
        size_t knew_stride = key_new_buf.total_size / batch_size;
        size_t vnew_stride = val_new_buf.total_size / batch_size;
        size_t out_stride = node.output_buffer.total_size / batch_size;
        const __fp16* q_all = query_buf.data_as<__fp16>();
        const __fp16* knew_all = key_new_buf.data_as<__fp16>();
        const __fp16* vnew_all = val_new_buf.data_as<__fp16>();
        __fp16* out_all = node.output_buffer.data_as<__fp16>();
        for (size_t i = 0; i < batch_size; ++i) {
            size_t ci = get_meta(k_cache_buf, i)->current_seq_len;
            size_t hist_i = (ci >= seq_len) ? ci - seq_len : 0;
            if (fp16_cache) {
                cactus_attention_f16(
                    q_all + i * q_stride,
                    get_fp16_data(k_cache_buf, i),
                    get_fp16_data(v_cache_buf, i),
                    out_all + i * out_stride,
                    1, seq_len, ci,
                    num_q_heads, kv_heads, hdim,
                    node.params.scale, nullptr, hist_i, node.params.window_size,
                    true, false, false, v_hdim);
            } else {
                cactus_attention_hybrid_int8_fp16(
                    q_all + i * q_stride,
                    get_int8_data(k_cache_buf, i),
                    get_int8_data(v_cache_buf, i),
                    get_scales(k_cache_buf, k_max, kv_heads, hdim, i),
                    get_scales(v_cache_buf, v_max, kv_heads, v_hdim, i),
                    knew_all + i * knew_stride,
                    vnew_all + i * vnew_stride,
                    out_all + i * out_stride,
                    1, seq_len, hist_i, seq_len,
                    num_q_heads, kv_heads, hdim,
                    node.params.scale, hist_i, true, node.params.window_size,
                    KV_QUANT_GROUP_SIZE, v_hdim);
            }
        }
        return;
    }

    if (k_cache_buf.precision == Precision::FP16 || v_cache_buf.precision == Precision::FP16) {
        cactus_attention_f16(
            query_buf.data_as<__fp16>(),
            get_fp16_data(k_cache_buf, slot),
            get_fp16_data(v_cache_buf, slot),
            node.output_buffer.data_as<__fp16>(),
            batch_size, seq_len, cache_len,
            num_q_heads, kv_heads, hdim,
            node.params.scale,
            nullptr,
            position_offset,
            node.params.window_size,
            true,
            false,
            false,
            v_hdim);
        return;
    }

        cactus_attention_hybrid_int8_fp16(
            query_buf.data_as<__fp16>(),
            cached_keys,
            cached_values,
            k_scales,
            v_scales,
            key_new_buf.data_as<__fp16>(),
            val_new_buf.data_as<__fp16>(),
            node.output_buffer.data_as<__fp16>(),
            batch_size, seq_len, history_len, cache_only_attention ? 0 : seq_len,
            num_q_heads, kv_heads, hdim,
            node.params.scale,
            position_offset,
        true,
        node.params.window_size,
        KV_QUANT_GROUP_SIZE,
        v_hdim);
}

void compute_conv_cache_state_node(
    GraphNode& node,
    const nodes_vector&,
    const node_index_map_t&) {

    if (node.output_buffer.get_data()) return;

    size_t ws = node.params.window_size;
    size_t hd = node.params.head_dim;
    size_t total = sizeof(ConvCacheMetadata) + ws * hd * sizeof(__fp16);

    node.output_buffer = BufferDesc({total}, Precision::INT8);
    node.output_buffer.allocate();
    std::memset(node.output_buffer.get_data(), 0, total);

    auto* meta = get_conv_meta(node.output_buffer);
    meta->head = 0;
    meta->count = 0;
    meta->window_size = ws;
    meta->hidden_dim = hd;
}

void compute_conv_cache_append_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& new_data = get_input(node, 0, nodes, node_index_map);
    auto& cache_buf = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto* meta = get_conv_meta(cache_buf);

    size_t ws = meta->window_size;
    size_t hd = meta->hidden_dim;
    size_t head = meta->head;
    size_t count = meta->count;

    size_t num_rows = new_data.total_size / hd;
    if (num_rows == 0) return;

    __fp16* cache_data = get_conv_data(cache_buf);

    const __fp16* src;
    std::vector<__fp16> converted;
    if (new_data.precision == Precision::FP16) {
        src = new_data.data_as<__fp16>();
    } else if (new_data.precision == Precision::FP32) {
        converted.resize(new_data.total_size);
        Quantization::fp32_to_fp16(new_data.data_as<float>(), converted.data(), new_data.total_size);
        src = converted.data();
    } else {
        converted.resize(new_data.total_size);
        Quantization::int8_to_fp16(new_data.data_as<int8_t>(), converted.data(), new_data.total_size);
        src = converted.data();
    }

    size_t copy_rows = std::min(num_rows, ws);
    size_t start_row = num_rows > ws ? num_rows - ws : 0;

    for (size_t i = 0; i < copy_rows; ++i) {
        std::memcpy(cache_data + head * hd, src + (start_row + i) * hd, hd * sizeof(__fp16));
        head = (head + 1) % ws;
        if (count < ws) ++count;
    }

    meta->head = head;
    meta->count = count;

    __fp16* out = node.output_buffer.data_as<__fp16>();
    if (count < ws) {
        const size_t pad_rows = ws - count;
        std::memset(out, 0, pad_rows * hd * sizeof(__fp16));
        std::memcpy(out + pad_rows * hd, cache_data, count * hd * sizeof(__fp16));
    } else {
        size_t tail_rows = ws - head;
        if (tail_rows > 0) {
            std::memcpy(out, cache_data + head * hd, tail_rows * hd * sizeof(__fp16));
        }
        if (head > 0) {
            std::memcpy(out + tail_rows * hd, cache_data, head * hd * sizeof(__fp16));
        }
    }
}

void compute_conv_cache_initialize_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& new_data = get_input(node, 0, nodes, node_index_map);
    auto& cache_buf = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto* meta = get_conv_meta(cache_buf);

    const size_t ws = meta->window_size;
    const size_t hd = meta->hidden_dim;

    __fp16* cache_data = get_conv_data(cache_buf);
    std::memset(cache_data, 0, ws * hd * sizeof(__fp16));
    meta->head = 0;
    meta->count = 0;

    const size_t num_rows = new_data.total_size / hd;
    if (num_rows == 0) return;

    const __fp16* src;
    std::vector<__fp16> converted;
    if (new_data.precision == Precision::FP16) {
        src = new_data.data_as<__fp16>();
    } else if (new_data.precision == Precision::FP32) {
        converted.resize(new_data.total_size);
        Quantization::fp32_to_fp16(new_data.data_as<float>(), converted.data(), new_data.total_size);
        src = converted.data();
    } else {
        converted.resize(new_data.total_size);
        Quantization::int8_to_fp16(new_data.data_as<int8_t>(), converted.data(), new_data.total_size);
        src = converted.data();
    }

    const size_t copy_rows = std::min(num_rows, ws);
    const size_t start_row = num_rows - copy_rows;
    std::memcpy(cache_data, src + start_row * hd, copy_rows * hd * sizeof(__fp16));
    meta->head = copy_rows % ws;
    meta->count = copy_rows;
}

void compute_recurrent_cache_state_node(
    GraphNode& node,
    const nodes_vector&,
    const node_index_map_t&) {

    if (node.output_buffer.get_data()) return;
    node.output_buffer.allocate();
    std::memset(node.output_buffer.get_data(), 0, node.output_buffer.byte_size);
}

void compute_recurrent_cache_write_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& src = get_input(node, 0, nodes, node_index_map);
    auto& cache_buf = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    std::memcpy(cache_buf.get_data(), src.get_data(), src.byte_size);
}

void CactusGraph::steal_cache_buffer(size_t dst_node, CactusGraph& src, size_t src_node) {
    auto& dst = nodes_[node_index_map_.at(dst_node)];
    auto& s = src.nodes_[src.node_index_map_.at(src_node)];
    // Buffer carries its own runtime precision (may be fp16); only op_type is invariant pre-move.
    assert(dst->op_type == s->op_type);
    auto shape = s->output_buffer.shape;
    auto prec = s->output_buffer.precision;
    dst->output_buffer = std::move(s->output_buffer);
    s->output_buffer = BufferDesc(shape, prec);
}

namespace {

struct PaddedAppendBackup {
    uint64_t overshoot;
    uint64_t keep_sink;
    uint64_t kept_padded;
    uint64_t ring_rows;
};

size_t ring_row_for_pos(size_t pos, size_t sink, size_t ring_capacity) {
    if (pos < ring_capacity) return pos;
    size_t recent = ring_capacity > sink ? ring_capacity - sink : 1;
    return sink + (pos - sink) % recent;
}

struct CacheRowRegion {
    size_t offset;
    size_t row_bytes;
};

std::vector<CacheRowRegion> cache_row_regions(const BufferDesc& buf, const CacheMetadata* meta) {
    const auto* base = static_cast<const uint8_t*>(buf.get_data());
    const size_t stride = meta->num_kv_heads * meta->head_dim;
    if (buf.precision == Precision::FP16) {
        return {{static_cast<size_t>(reinterpret_cast<const uint8_t*>(get_fp16_data(buf)) - base), stride * sizeof(__fp16)}};
    }
    const size_t num_groups = (meta->head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    return {
        {static_cast<size_t>(reinterpret_cast<const uint8_t*>(get_int8_data(buf)) - base), stride},
        {static_cast<size_t>(reinterpret_cast<const uint8_t*>(get_scales(buf, meta->max_seq_len, meta->num_kv_heads, meta->head_dim)) - base),
         meta->num_kv_heads * num_groups * sizeof(float)},
    };
}

} // namespace

std::vector<uint8_t> CactusGraph::snapshot_cache_padded_append(size_t node_id, size_t real_tokens, size_t pad_tokens) const {
    const auto& node = *nodes_[node_index_map_.at(node_id)];
    const auto& buf = node.output_buffer;
    if (node.op_type != OpType::KV_CACHE_STATE || !buf.get_data() || pad_tokens == 0) return {};
    const size_t window = node.params.window_size;
    const size_t ceiling = node.params.max_cache_seq_len;
    if (window == 0 || window >= ceiling) return {};
    const auto* meta = get_meta(buf);
    const size_t len0 = meta->current_seq_len;
    const size_t appended = real_tokens + pad_tokens;
    if (kv_cache_resident()) {
        const size_t ring_window = meta->max_seq_len - meta->sink_size - 1;
        if (ring_window <= meta->sink_size || appended >= ring_window - meta->sink_size) {
            throw std::runtime_error("padded cache append larger than the attention window is not supported");
        }
        std::vector<uint8_t> backup(sizeof(PaddedAppendBackup));
        auto* header = reinterpret_cast<PaddedAppendBackup*>(backup.data());
        header->overshoot = 0;
        header->keep_sink = meta->sink_size;
        header->kept_padded = ring_window;
        header->ring_rows = pad_tokens;
        const auto* base = static_cast<const uint8_t*>(buf.get_data());
        for (const auto& region : cache_row_regions(buf, meta)) {
            for (size_t t = 0; t < pad_tokens; ++t) {
                size_t row = ring_row_for_pos(len0 + real_tokens + t, meta->sink_size, ring_window);
                const uint8_t* src = base + region.offset + row * region.row_bytes;
                backup.insert(backup.end(), src, src + region.row_bytes);
            }
        }
        return backup;
    }
    if (len0 == 0 || len0 + appended <= window) return {};
    const size_t keep_sink = std::min({static_cast<size_t>(meta->sink_size), len0, window});
    const size_t tail_capacity = window - keep_sink;
    if (appended >= tail_capacity) {
        throw std::runtime_error("padded cache append larger than the attention window is not supported");
    }
    auto kept_after = [&](size_t n) { return std::min(tail_capacity - n, len0 - keep_sink); };
    const size_t kept_real = kept_after(real_tokens);
    const size_t kept_padded = kept_after(appended);
    const size_t overshoot = kept_real - kept_padded;
    if (overshoot == 0) return {};

    const size_t first_saved_row = len0 - kept_real;
    std::vector<uint8_t> backup(sizeof(PaddedAppendBackup));
    auto* header = reinterpret_cast<PaddedAppendBackup*>(backup.data());
    header->overshoot = overshoot;
    header->keep_sink = keep_sink;
    header->kept_padded = kept_padded;
    const auto* base = static_cast<const uint8_t*>(buf.get_data());
    for (const auto& region : cache_row_regions(buf, meta)) {
        const uint8_t* rows = base + region.offset + first_saved_row * region.row_bytes;
        backup.insert(backup.end(), rows, rows + overshoot * region.row_bytes);
    }
    return backup;
}

void CactusGraph::rollback_cache_padded_append(size_t node_id, size_t real_tokens, size_t pad_tokens,
                                               const std::vector<uint8_t>& backup) {
    auto& node = *nodes_[node_index_map_.at(node_id)];
    auto& buf = node.output_buffer;
    if (node.op_type != OpType::KV_CACHE_STATE || !buf.get_data() || pad_tokens == 0) return;
    auto* meta = get_meta(buf);
    if (backup.empty()) {
        meta->current_seq_len = meta->current_seq_len >= pad_tokens ? meta->current_seq_len - pad_tokens : 0;
        return;
    }
    const auto* header = reinterpret_cast<const PaddedAppendBackup*>(backup.data());
    if (header->ring_rows > 0) {
        const size_t ring_window = header->kept_padded;
        const size_t sink = header->keep_sink;
        const size_t len_after = meta->current_seq_len;
        const size_t len0 = len_after >= real_tokens + pad_tokens ? len_after - real_tokens - pad_tokens : 0;
        auto* base = static_cast<uint8_t*>(buf.get_data());
        const uint8_t* saved = backup.data() + sizeof(PaddedAppendBackup);
        for (const auto& region : cache_row_regions(buf, meta)) {
            for (size_t t = 0; t < header->ring_rows; ++t) {
                size_t row = ring_row_for_pos(len0 + real_tokens + t, sink, ring_window);
                std::memcpy(base + region.offset + row * region.row_bytes, saved, region.row_bytes);
                saved += region.row_bytes;
            }
        }
        meta->current_seq_len = len0 + real_tokens;
        return;
    }
    const size_t overshoot = header->overshoot;
    const size_t keep_sink = header->keep_sink;
    const size_t kept_padded = header->kept_padded;
    const size_t move_rows = kept_padded + real_tokens;
    auto* base = static_cast<uint8_t*>(buf.get_data());
    const uint8_t* saved = backup.data() + sizeof(PaddedAppendBackup);
    for (const auto& region : cache_row_regions(buf, meta)) {
        uint8_t* rows = base + region.offset;
        std::memmove(rows + (keep_sink + overshoot) * region.row_bytes,
                     rows + keep_sink * region.row_bytes, move_rows * region.row_bytes);
        std::memcpy(rows + keep_sink * region.row_bytes, saved, overshoot * region.row_bytes);
        saved += overshoot * region.row_bytes;
    }
    meta->current_seq_len = keep_sink + overshoot + kept_padded + real_tokens;
}

void CactusGraph::shrink_cache_buffer(size_t node_id, size_t new_capacity) {
    auto& buf = nodes_[node_index_map_.at(node_id)]->output_buffer;
    if (!buf.get_data()) return;
    auto* meta = get_meta(buf);
    size_t target = std::max<size_t>(new_capacity, meta->current_seq_len);
    if (target >= meta->max_seq_len) return;
    resize_cache_buffer(buf, target);
}

