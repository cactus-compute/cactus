#include "graph.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>

size_t BufferPool::round_up_size(size_t size) const {
    constexpr size_t ALIGNMENT = 64;
    constexpr size_t MIN_BUCKET = 1024;

    if (size < MIN_BUCKET) return MIN_BUCKET;

    size_t aligned = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    size_t bucket = MIN_BUCKET;
    while (bucket < aligned) {
        bucket *= 2;
    }
    return bucket;
}

char* BufferPool::acquire(size_t byte_size) {
    if (byte_size == 0) return nullptr;

    size_t bucket_size = round_up_size(byte_size);

    auto it = free_buffers_.find(bucket_size);
    if (it != free_buffers_.end() && !it->second.empty()) {
        auto buffer = std::move(it->second.back());
        it->second.pop_back();
        pool_bytes_ -= bucket_size;
        active_bytes_ += bucket_size;
        if (active_bytes_ > peak_bytes_) {
            peak_bytes_ = active_bytes_;
        }
        return buffer.release();
    }

    active_bytes_ += bucket_size;
    if (active_bytes_ > peak_bytes_) {
        peak_bytes_ = active_bytes_;
    }
    return new char[bucket_size];
}

void BufferPool::release(char* ptr, size_t byte_size) {
    if (!ptr || byte_size == 0) return;

    size_t bucket_size = round_up_size(byte_size);
    active_bytes_ -= bucket_size;
    pool_bytes_ += bucket_size;

    free_buffers_[bucket_size].push_back(std::unique_ptr<char[]>(ptr));
}

void BufferPool::clear() {
    free_buffers_.clear();
    pool_bytes_ = 0;
}

BufferDesc::BufferDesc()
    : total_size(0), byte_size(0), external_data(nullptr), pooled_data(nullptr),
      precision(Precision::FP16) {}

BufferDesc::BufferDesc(const std::vector<size_t>& s, Precision prec)
    : shape(s), external_data(nullptr), pooled_data(nullptr), precision(prec) {
    total_size = 1;
    for (size_t dim : shape) total_size *= dim;
    byte_size = PrecisionTraits::packed_size_of(prec, total_size);
}

BufferDesc::~BufferDesc() {
    if (pooled_data) {
        delete[] pooled_data;
        pooled_data = nullptr;
    }
}

BufferDesc::BufferDesc(BufferDesc&& other) noexcept
    : shape(std::move(other.shape)),
      total_size(other.total_size),
      byte_size(other.byte_size),
      data(std::move(other.data)),
      external_data(other.external_data),
      pooled_data(other.pooled_data),
      precision(other.precision),
      group_size(other.group_size),
      num_groups(other.num_groups),
      scales_data(other.scales_data),
      owned_scales(std::move(other.owned_scales)),
      is_interleaved(other.is_interleaved),
      original_N(other.original_N),
      activation_scales_data(other.activation_scales_data),
      owned_activation_scales(std::move(other.owned_activation_scales)),
      num_rows_for_activation_scales(other.num_rows_for_activation_scales) {
    other.total_size = 0;
    other.byte_size = 0;
    other.external_data = nullptr;
    other.pooled_data = nullptr;
    other.group_size = 0;
    other.num_groups = 0;
    other.scales_data = nullptr;
    other.is_interleaved = false;
    other.original_N = 0;
    other.activation_scales_data = nullptr;
    other.num_rows_for_activation_scales = 0;
}

BufferDesc& BufferDesc::operator=(BufferDesc&& other) noexcept {
    if (this != &other) {
        if (pooled_data) {
            delete[] pooled_data;
        }

        shape = std::move(other.shape);
        total_size = other.total_size;
        byte_size = other.byte_size;
        data = std::move(other.data);
        external_data = other.external_data;
        pooled_data = other.pooled_data;
        precision = other.precision;
        group_size = other.group_size;
        num_groups = other.num_groups;
        scales_data = other.scales_data;
        owned_scales = std::move(other.owned_scales);
        is_interleaved = other.is_interleaved;
        original_N = other.original_N;
        activation_scales_data = other.activation_scales_data;
        owned_activation_scales = std::move(other.owned_activation_scales);
        num_rows_for_activation_scales = other.num_rows_for_activation_scales;

        other.total_size = 0;
        other.byte_size = 0;
        other.external_data = nullptr;
        other.pooled_data = nullptr;
        other.group_size = 0;
        other.num_groups = 0;
        other.scales_data = nullptr;
        other.is_interleaved = false;
        other.original_N = 0;
        other.activation_scales_data = nullptr;
        other.num_rows_for_activation_scales = 0;
    }
    return *this;
}

void* BufferDesc::get_data() {
    if (external_data) return external_data;
    if (pooled_data) return pooled_data;
    return data.get();
}

const void* BufferDesc::get_data() const {
    if (external_data) return external_data;
    if (pooled_data) return pooled_data;
    return data.get();
}

void BufferDesc::allocate() {
    if (!data && !external_data && !pooled_data) {
        data = std::make_unique<char[]>(byte_size);
    }
}

void BufferDesc::allocate_from_pool(BufferPool& pool) {
    if (!data && !external_data && !pooled_data && byte_size > 0) {
        pooled_data = pool.acquire(byte_size);
    }
}

void BufferDesc::release_to_pool(BufferPool& pool) {
    if (pooled_data && byte_size > 0) {
        pool.release(pooled_data, byte_size);
        pooled_data = nullptr;
    }
}

void BufferDesc::set_external(void* ptr) {
    external_data = ptr;
    data.reset();
    pooled_data = nullptr;
}

// GraphNode implementation
GraphNode::GraphNode(size_t node_id, OpType type) : id(node_id), op_type(type) {}

// TensorConfig implementation
TensorConfig& TensorConfig::global() {
    static TensorConfig instance;
    return instance;
}

// BroadcastInfo implementation
BroadcastInfo BroadcastInfo::compute(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs) {
    BroadcastInfo info;
    size_t max_dims = std::max(lhs.size(), rhs.size());
    info.output_shape.resize(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t lhs_dim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
        size_t rhs_dim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument("Shapes are not compatible for broadcasting");
        }

        info.output_shape[max_dims - 1 - i] = std::max(lhs_dim, rhs_dim);
    }

    info.needs_broadcasting = (lhs != info.output_shape || rhs != info.output_shape);
    return info;
}

namespace ValidationUtils {
    void validate_tensor_dims(const std::vector<size_t>& shape, size_t required_dims, const std::string& op_name) {
        if (shape.size() != required_dims) {
            throw std::runtime_error(op_name + " requires " + std::to_string(required_dims) +
                                    "D tensor, got " + std::to_string(shape.size()) + "D tensor");
        }
    }

    void validate_precision(Precision actual, Precision required, const std::string& op_name) {
        if (actual != required) {
            std::string actual_str = (actual == Precision::INT8) ? "INT8" : "FP32";
            std::string required_str = (required == Precision::INT8) ? "INT8" : "FP32";
            throw std::runtime_error(op_name + " requires " + required_str + " precision, got " + actual_str);
        }
    }

    void validate_input_count(size_t actual, size_t required, const std::string& op_name) {
        if (actual < required) {
            throw std::runtime_error(op_name + " requires " + std::to_string(required) +
                                    " inputs, got " + std::to_string(actual) + " inputs");
        }
    }
}

CactusGraph::CactusGraph() : next_node_id_(0) {}

size_t CactusGraph::get_node_count() const {
    return nodes_.size();
}

void CactusGraph::register_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id) {
    debug_nodes_.push_back({layer_idx, name, node_id});
}

void CactusGraph::capture_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id) {
    register_debug_node(layer_idx, name, node_id);
}

const std::vector<CactusGraph::DebugNodeEntry>& CactusGraph::get_debug_nodes() const {
    return debug_nodes_;
}

void CactusGraph::clear_debug_nodes() {
    debug_nodes_.clear();
}

void CactusGraph::allocate_buffers() {
    for (auto& node : nodes_) {
        if (node->op_type != OpType::INPUT) {
            node->output_buffer.allocate();
        }
    }
}
