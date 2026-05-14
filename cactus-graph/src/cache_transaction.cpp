#include "../cactus_graph.h"
#include "cactus_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

struct KvCacheMetadata {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t reserved[3];
};

static_assert(sizeof(KvCacheMetadata) == 64, "KvCacheMetadata must be 64 bytes");

KvCacheMetadata* kv_meta(BufferDesc& buffer) {
    return static_cast<KvCacheMetadata*>(buffer.get_data());
}

const KvCacheMetadata* kv_meta(const BufferDesc& buffer) {
    return static_cast<const KvCacheMetadata*>(buffer.get_data());
}

int8_t* kv_int8_data(BufferDesc& buffer) {
    return reinterpret_cast<int8_t*>(static_cast<char*>(buffer.get_data()) + sizeof(KvCacheMetadata));
}

float* kv_scale_data(BufferDesc& buffer, size_t max_seq, size_t kv_heads, size_t head_dim) {
    size_t int8_bytes = max_seq * kv_heads * head_dim;
    return reinterpret_cast<float*>(static_cast<char*>(buffer.get_data()) + sizeof(KvCacheMetadata) + int8_bytes);
}

void copy_kv_row(BufferDesc& buffer, uint64_t dst, uint64_t src) {
    auto* meta = kv_meta(buffer);
    const size_t max_seq = meta->max_seq_len;
    const size_t kv_heads = meta->num_kv_heads;
    const size_t head_dim = meta->head_dim;
    const size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    if (dst >= max_seq || src >= max_seq) {
        throw std::out_of_range("KV cache selected commit row out of range");
    }

    const size_t int8_stride = kv_heads * head_dim;
    const size_t scale_stride = kv_heads * num_groups;
    int8_t* int8_base = kv_int8_data(buffer);
    float* scale_base = kv_scale_data(buffer, max_seq, kv_heads, head_dim);
    std::memmove(int8_base + dst * int8_stride,
                 int8_base + src * int8_stride,
                 int8_stride);
    std::memmove(scale_base + dst * scale_stride,
                 scale_base + src * scale_stride,
                 scale_stride * sizeof(float));
}

} // namespace

uint64_t CactusGraph::kv_cache_sequence_length(size_t cache_state_node) const {
    auto it = node_index_map_.find(cache_state_node);
    if (it == node_index_map_.end()) {
        throw std::invalid_argument("Unknown KV cache node");
    }
    const auto& buffer = nodes_.at(it->second)->output_buffer;
    if (!buffer.get_data()) {
        return 0;
    }
    return kv_meta(buffer)->current_seq_len;
}

void CactusGraph::set_kv_cache_sequence_length(size_t cache_state_node, uint64_t sequence_length) {
    auto it = node_index_map_.find(cache_state_node);
    if (it == node_index_map_.end()) {
        throw std::invalid_argument("Unknown KV cache node");
    }
    auto& buffer = nodes_.at(it->second)->output_buffer;
    if (!buffer.get_data()) {
        throw std::invalid_argument("KV cache node has no allocated buffer");
    }
    auto* meta = kv_meta(buffer);
    meta->current_seq_len = std::min<uint64_t>(sequence_length, meta->max_seq_len);
}

CactusGraph::KvCacheTransaction CactusGraph::begin_kv_cache_transaction(const std::vector<size_t>& cache_state_nodes) {
    std::vector<std::pair<size_t, uint64_t>> snapshots;
    snapshots.reserve(cache_state_nodes.size());
    for (size_t node : cache_state_nodes) {
        snapshots.emplace_back(node, kv_cache_sequence_length(node));
    }
    return KvCacheTransaction(this, std::move(snapshots));
}

void CactusGraph::KvCacheTransaction::rollback() {
    if (!graph_ || closed_) return;
    for (const auto& [node, length] : snapshots_) {
        graph_->set_kv_cache_sequence_length(node, length);
        graph_->pending_kv_cache_sequence_lengths_.emplace_back(node, length);
    }
    closed_ = true;
}

void CactusGraph::KvCacheTransaction::commit_all() {
    if (!graph_ || closed_) return;
    for (const auto& [node, _] : snapshots_) {
        uint64_t committed_length = graph_->kv_cache_sequence_length(node);
        graph_->pending_kv_cache_sequence_lengths_.emplace_back(node, committed_length);
    }
    closed_ = true;
}

void CactusGraph::KvCacheTransaction::commit_prefix(size_t accepted_tokens) {
    if (!graph_ || closed_) return;
    for (const auto& [node, length] : snapshots_) {
        uint64_t committed_length = length + accepted_tokens;
        graph_->set_kv_cache_sequence_length(node, committed_length);
        graph_->pending_kv_cache_sequence_lengths_.emplace_back(node, committed_length);
    }
    closed_ = true;
}

void CactusGraph::KvCacheTransaction::commit_selected(const std::vector<size_t>& token_indices) {
    if (!graph_ || closed_) return;
    size_t appended_tokens = 0;
    for (size_t index : token_indices) {
        appended_tokens = std::max(appended_tokens, index + 1);
    }
    for (const auto& [node, length] : snapshots_) {
        auto it = graph_->node_index_map_.find(node);
        if (it == graph_->node_index_map_.end()) {
            throw std::invalid_argument("Unknown KV cache node");
        }
        auto& buffer = graph_->nodes_.at(it->second)->output_buffer;
        auto* meta = kv_meta(buffer);
        if (appended_tokens > meta->max_seq_len) {
            throw std::out_of_range("KV cache selected commit exceeds cache capacity");
        }
        if (length + appended_tokens > meta->max_seq_len) {
            throw std::runtime_error("KV cache selected commit after sliding-window eviction is unsupported");
        }
        uint64_t append_start = length;
        for (size_t dst = 0; dst < token_indices.size(); ++dst) {
            const uint64_t src_row = append_start + token_indices[dst];
            const uint64_t dst_row = append_start + dst;
            if (src_row != dst_row) {
                copy_kv_row(buffer, dst_row, src_row);
            }
        }
        uint64_t committed_length = append_start + token_indices.size();
        graph_->set_kv_cache_sequence_length(node, committed_length);
        graph_->pending_kv_cache_sequence_lengths_.emplace_back(node, committed_length);
    }
    closed_ = true;
}

void CactusGraph::apply_pending_kv_cache_sequence_lengths() {
    if (pending_kv_cache_sequence_lengths_.empty()) return;
    auto pending = std::move(pending_kv_cache_sequence_lengths_);
    pending_kv_cache_sequence_lengths_.clear();
    for (const auto& [node, length] : pending) {
        set_kv_cache_sequence_length(node, length);
    }
}
