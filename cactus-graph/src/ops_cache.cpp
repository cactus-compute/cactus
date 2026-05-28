#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <algorithm>
#include <limits>
#include <cstdlib>

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
    uint64_t reserved[3];
};

static_assert(sizeof(CacheMetadata) == 64, "CacheMetadata must be 64 bytes");

inline CacheMetadata* get_meta(BufferDesc& buf) {
    return static_cast<CacheMetadata*>(buf.get_data());
}

inline const CacheMetadata* get_meta(const BufferDesc& buf) {
    return static_cast<const CacheMetadata*>(buf.get_data());
}

inline int8_t* get_int8_data(BufferDesc& buf) {
    return reinterpret_cast<int8_t*>(static_cast<char*>(buf.get_data()) + sizeof(CacheMetadata));
}

inline const int8_t* get_int8_data(const BufferDesc& buf) {
    return reinterpret_cast<const int8_t*>(static_cast<const char*>(buf.get_data()) + sizeof(CacheMetadata));
}

inline float* get_scales(BufferDesc& buf, size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t int8_bytes = max_seq * kv_heads * head_dim;
    return reinterpret_cast<float*>(static_cast<char*>(buf.get_data()) + sizeof(CacheMetadata) + int8_bytes);
}

inline const float* get_scales(const BufferDesc& buf, size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t int8_bytes = max_seq * kv_heads * head_dim;
    return reinterpret_cast<const float*>(static_cast<const char*>(buf.get_data()) + sizeof(CacheMetadata) + int8_bytes);
}

inline size_t cache_buffer_size(size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    return sizeof(CacheMetadata) + max_seq * kv_heads * head_dim + max_seq * kv_heads * num_groups * sizeof(float);
}

inline size_t fp16_cache_elements(size_t max_seq, size_t kv_heads, size_t head_dim) {
    return (sizeof(CacheMetadata) / sizeof(__fp16)) + max_seq * kv_heads * head_dim;
}

inline __fp16* get_fp16_data(BufferDesc& buf) {
    return reinterpret_cast<__fp16*>(static_cast<char*>(buf.get_data()) + sizeof(CacheMetadata));
}

inline const __fp16* get_fp16_data(const BufferDesc& buf) {
    return reinterpret_cast<const __fp16*>(static_cast<const char*>(buf.get_data()) + sizeof(CacheMetadata));
}

inline bool use_fp16_kv_cache() {
    static const bool cached = [] {
        const char* value = std::getenv("CACTUS_KV_CACHE_FP16");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return cached;
}

} // namespace

void compute_kv_cache_state_node(
    GraphNode& node,
    const nodes_vector&,
    const node_index_map_t&) {

    if (node.output_buffer.get_data()) return;

    size_t max_seq = node.params.max_cache_seq_len;
    size_t kv_heads = node.params.num_kv_heads;
    size_t hdim = node.params.head_dim;
    const bool fp16_cache = use_fp16_kv_cache();
    size_t total = fp16_cache
        ? fp16_cache_elements(max_seq, kv_heads, hdim)
        : cache_buffer_size(max_seq, kv_heads, hdim);

    node.output_buffer = BufferDesc({total}, fp16_cache ? Precision::FP16 : Precision::INT8);
    node.output_buffer.allocate();
    std::memset(node.output_buffer.get_data(), 0, node.output_buffer.byte_size);

    auto* meta = get_meta(node.output_buffer);
    meta->current_seq_len = 0;
    meta->max_seq_len = max_seq;
    meta->num_kv_heads = kv_heads;
    meta->head_dim = hdim;
    meta->sink_size = node.params.cache_sink_size;
}

void compute_kv_cache_append_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& new_kv = get_input(node, 0, nodes, node_index_map);
    auto& cache_buf = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    auto* meta = get_meta(cache_buf);

    size_t current_len = meta->current_seq_len;
    size_t max_len = meta->max_seq_len;
    size_t kv_heads = meta->num_kv_heads;
    size_t hdim = meta->head_dim;
    size_t sink = meta->sink_size;
    size_t num_groups = (hdim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;

    size_t new_seq_len = new_kv.total_size / (kv_heads * hdim);
    size_t int8_stride = kv_heads * hdim;
    size_t scale_stride = kv_heads * num_groups;

    if (cache_buf.precision == Precision::FP16) {
        size_t stride = kv_heads * hdim;
        __fp16* fp16_base = get_fp16_data(cache_buf);
        const __fp16* source = new_kv.data_as<__fp16>();
        size_t window = node.params.window_size;
        if (window == 0) window = max_len;

        size_t new_total = current_len + new_seq_len;
        bool needs_eviction = new_total > window;
        if (needs_eviction) {
            size_t keep_sink = std::min({sink, current_len, window});
            size_t tail_capacity = window - keep_sink;
            if (new_seq_len >= tail_capacity) {
                if (tail_capacity > 0) {
                    size_t source_offset = new_seq_len - tail_capacity;
                    std::memcpy(
                        fp16_base + keep_sink * stride,
                        source + source_offset * stride,
                        tail_capacity * stride * sizeof(__fp16));
                }
                meta->current_seq_len = keep_sink + tail_capacity;
                *node.output_buffer.data_as<float>() = static_cast<float>(meta->current_seq_len);
                return;
            }

            size_t remaining = tail_capacity - new_seq_len;
            remaining = std::min(remaining, current_len - keep_sink);
            size_t shift_src = current_len - remaining;
            if (remaining > 0 && shift_src > keep_sink) {
                std::memmove(
                    fp16_base + keep_sink * stride,
                    fp16_base + shift_src * stride,
                    remaining * stride * sizeof(__fp16));
            }
            size_t append_offset = keep_sink + remaining;
            std::memcpy(fp16_base + append_offset * stride, source, new_seq_len * stride * sizeof(__fp16));
            meta->current_seq_len = append_offset + new_seq_len;
        } else {
            std::memcpy(fp16_base + current_len * stride, source, new_seq_len * stride * sizeof(__fp16));
            meta->current_seq_len = new_total;
        }

        *node.output_buffer.data_as<float>() = static_cast<float>(meta->current_seq_len);
        return;
    }

    int8_t* int8_base = get_int8_data(cache_buf);
    float* scale_base = get_scales(cache_buf, max_len, kv_heads, hdim);

    size_t window = node.params.window_size;
    if (window == 0) window = max_len;

    size_t new_total = current_len + new_seq_len;
    bool needs_eviction = new_total > window;

    if (needs_eviction) {
        size_t keep_sink = std::min({sink, current_len, window});
        size_t tail_capacity = window - keep_sink;
        if (new_seq_len >= tail_capacity) {
            if (tail_capacity > 0) {
                size_t source_offset = new_seq_len - tail_capacity;
                cactus_quantize_kv_fp16_to_int8(
                    new_kv.data_as<__fp16>() + source_offset * int8_stride,
                    int8_base + keep_sink * int8_stride,
                    scale_base + keep_sink * scale_stride,
                    tail_capacity, kv_heads, hdim);
            }
            meta->current_seq_len = keep_sink + tail_capacity;
            *node.output_buffer.data_as<float>() = static_cast<float>(meta->current_seq_len);
            return;
        }

        size_t remaining = tail_capacity - new_seq_len;
        remaining = std::min(remaining, current_len - keep_sink);
        size_t shift_src = current_len - remaining;

        if (remaining > 0 && shift_src > keep_sink) {
            std::memmove(int8_base + keep_sink * int8_stride,
                         int8_base + shift_src * int8_stride,
                         remaining * int8_stride);
            std::memmove(scale_base + keep_sink * scale_stride,
                         scale_base + shift_src * scale_stride,
                         remaining * scale_stride * sizeof(float));
        }

        size_t append_offset = keep_sink + remaining;
        cactus_quantize_kv_fp16_to_int8(
            new_kv.data_as<__fp16>(),
            int8_base + append_offset * int8_stride,
            scale_base + append_offset * scale_stride,
            new_seq_len, kv_heads, hdim);

        meta->current_seq_len = append_offset + new_seq_len;
    } else {
        cactus_quantize_kv_fp16_to_int8(
            new_kv.data_as<__fp16>(),
            int8_base + current_len * int8_stride,
            scale_base + current_len * scale_stride,
            new_seq_len, kv_heads, hdim);

        meta->current_seq_len = new_total;
    }

    *node.output_buffer.data_as<float>() = static_cast<float>(meta->current_seq_len);
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

    const auto* k_meta = get_meta(k_cache_buf);
    size_t cache_len = k_meta->current_seq_len;
    size_t k_max = k_meta->max_seq_len;
    size_t kv_heads = k_meta->num_kv_heads;
    size_t hdim = k_meta->head_dim;

    const auto* v_meta = get_meta(v_cache_buf);
    size_t v_hdim = node.params.v_head_dim > 0 ? node.params.v_head_dim : hdim;
    size_t v_max = v_meta->max_seq_len;

    const int8_t* cached_keys = get_int8_data(k_cache_buf);
    const float* k_scales = get_scales(k_cache_buf, k_max, kv_heads, hdim);
    const int8_t* cached_values = get_int8_data(v_cache_buf);
    const float* v_scales = get_scales(v_cache_buf, v_max, kv_heads, v_hdim);

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

    if (k_cache_buf.precision == Precision::FP16 || v_cache_buf.precision == Precision::FP16) {
        cactus_attention_f16(
            query_buf.data_as<__fp16>(),
            get_fp16_data(k_cache_buf),
            get_fp16_data(v_cache_buf),
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
