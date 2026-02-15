#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>
#include <sstream>
#include <iostream>
#include <arm_neon.h>

namespace cactus {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NONE = 4
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { min_level_ = level; }
    LogLevel get_level() const { return min_level_; }

    void set_callback(std::function<void(LogLevel, const std::string&, const std::string&)> cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = cb;
    }

    void log(LogLevel level, const std::string& component, const std::string& message) {
        if (level < min_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        if (callback_) {
            callback_(level, component, message);
        } else {
            std::cerr << "[" << level_string(level) << "] [" << component << "] " << message << std::endl;
        }

        if (level == LogLevel::ERROR) {
            last_error_ = "[" + component + "] " + message;
        }
    }

    const std::string& last_error() const { return last_error_; }
    void clear_error() { last_error_.clear(); }

private:
    Logger() : min_level_(LogLevel::WARN) {}

    static const char* level_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "?";
        }
    }

    LogLevel min_level_;
    std::mutex mutex_;
    std::string last_error_;
    std::function<void(LogLevel, const std::string&, const std::string&)> callback_;
};

} // namespace cactus

#define CACTUS_LOG(level, component, msg) \
    do { \
        if (static_cast<int>(level) >= static_cast<int>(cactus::Logger::instance().get_level())) { \
            std::ostringstream _cactus_log_ss; \
            _cactus_log_ss << msg; \
            cactus::Logger::instance().log(level, component, _cactus_log_ss.str()); \
        } \
    } while(0)

#define CACTUS_LOG_DEBUG(component, msg) CACTUS_LOG(cactus::LogLevel::DEBUG, component, msg)
#define CACTUS_LOG_INFO(component, msg)  CACTUS_LOG(cactus::LogLevel::INFO, component, msg)
#define CACTUS_LOG_WARN(component, msg)  CACTUS_LOG(cactus::LogLevel::WARN, component, msg)
#define CACTUS_LOG_ERROR(component, msg) CACTUS_LOG(cactus::LogLevel::ERROR, component, msg)

namespace GraphFile {
    class MappedFile;
}

enum class Precision {
    INT8,
    FP16,
    FP32,
    INT4 
};

enum class ComputeBackend {
    CPU,
    NPU
};

enum class OpType {
    INPUT, PRECISION_CAST,
    ADD, ADD_CLIPPED, SUBTRACT, MULTIPLY, DIVIDE,
    MATMUL, TRANSPOSE, RESHAPE, SLICE, GATHER, EMBEDDING,
    BILINEAR_INTERPOLATION,
    SUM, MEAN, VARIANCE, MIN, MAX,
    RMS_NORM, ROPE, ROPE_GPTJ, SOFTMAX, ATTENTION, ATTENTION_INT8_HYBRID, CONV1D_CAUSAL, CONV1D_K3, CONV1D_K7S3, CONV1D,
    SCALAR_ADD, SCALAR_SUBTRACT, SCALAR_MULTIPLY, SCALAR_DIVIDE, SCALAR_EXP, SCALAR_SQRT, SCALAR_COS, SCALAR_SIN,
    SILU, GELU, GELU_ERF, TANH,
    SAMPLE, CONCAT,
    SCATTER_TOPK,
    TOPK, LAYERNORM, GROUPNORM,
    INDEX,
    PERSISTENT,
    QUANTIZE_ACTIVATIONS
};

struct PrecisionTraits {
    static constexpr size_t size_of(Precision prec) {
        switch (prec) {
            case Precision::INT8: return 1;
            case Precision::FP16: return 2;
            case Precision::FP32: return 4;
            case Precision::INT4: return 1; 
        }
        return 1;
    }

    static constexpr size_t packed_size_of(Precision prec, size_t count) {
        switch (prec) {
            case Precision::INT4: return (count + 1) / 2;  
            default: return count * size_of(prec);
        }
    }

    static constexpr bool is_integer(Precision prec) {
        switch (prec) {
            case Precision::INT8: return true;
            case Precision::INT4: return true;
            case Precision::FP16: return false;
            case Precision::FP32: return false;
        }
        return true;
    }

    static constexpr bool is_floating_point(Precision prec) {
        switch (prec) {
            case Precision::INT8: return false;
            case Precision::INT4: return false;
            case Precision::FP16: return true;
            case Precision::FP32: return true;
        }
        return false;
    }
};

namespace Quantization {
    void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
    void fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);
    void fp16_to_fp32(const __fp16* src, float* dst, size_t count);
    void fp32_to_fp16(const float* src, __fp16* dst, size_t count);
    void int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale = 1.0f);
    void fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale = 1.0f);
}

struct TensorConfig {
    Precision default_precision = Precision::INT8;
    Precision compute_precision = Precision::INT8;
    Precision output_precision = Precision::INT8;
    bool auto_mixed_precision = false;
    bool enable_int4_packing = true;
    
    static TensorConfig& global();
};

struct BroadcastInfo {
    std::vector<size_t> output_shape;
    bool needs_broadcasting;

    static BroadcastInfo compute(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs);
};

class BufferPool;

struct BufferDesc {
    std::vector<size_t> shape;
    size_t total_size;
    size_t byte_size;
    std::unique_ptr<char[]> data;
    void* external_data;
    char* pooled_data;
    Precision precision;

    size_t group_size = 0;
    size_t num_groups = 0;
    void* scales_data = nullptr;
    std::unique_ptr<char[]> owned_scales;

    bool is_interleaved = false;
    size_t original_N = 0;  

    void* activation_scales_data = nullptr;
    std::unique_ptr<char[]> owned_activation_scales;
    size_t num_rows_for_activation_scales = 0;

    BufferDesc();
    BufferDesc(const std::vector<size_t>& s, Precision prec = Precision::INT8);
    ~BufferDesc();

    BufferDesc(BufferDesc&& other) noexcept;
    BufferDesc& operator=(BufferDesc&& other) noexcept;

    BufferDesc(const BufferDesc&) = delete;
    BufferDesc& operator=(const BufferDesc&) = delete;

    void* get_data();
    const void* get_data() const;

    template<typename T>
    T* data_as() { return static_cast<T*>(get_data()); }

    template<typename T>
    const T* data_as() const { return static_cast<const T*>(get_data()); }

    const __fp16* scales_as_fp16() const {
        return reinterpret_cast<const __fp16*>(scales_data);
    }

    bool is_grouped_int8() const {
        return precision == Precision::INT8 && group_size > 0;
    }

    void set_grouped_scales(size_t gs, size_t ng, void* scales_ptr) {
        group_size = gs;
        num_groups = ng;
        scales_data = scales_ptr;
    }

    void set_interleaved(bool interleaved, size_t orig_n) {
        is_interleaved = interleaved;
        original_N = orig_n;
    }

    bool has_activation_scales() const {
        return activation_scales_data != nullptr && num_rows_for_activation_scales > 0;
    }
    const float* activation_scales_as_float() const {
        return reinterpret_cast<const float*>(activation_scales_data);
    }
    float* activation_scales_as_float() {
        return reinterpret_cast<float*>(activation_scales_data);
    }
    void allocate_activation_scales(size_t num_rows) {
        num_rows_for_activation_scales = num_rows;
        owned_activation_scales = std::make_unique<char[]>(num_rows * sizeof(float));
        activation_scales_data = owned_activation_scales.get();
    }
    void set_activation_scales(void* scales_ptr, size_t num_rows) {
        activation_scales_data = scales_ptr;
        num_rows_for_activation_scales = num_rows;
    }

    void allocate();
    void allocate_from_pool(BufferPool& pool);
    void release_to_pool(BufferPool& pool);
    void set_external(void* ptr);
};

struct OpParams {
    float scalar = 0.0f;
    float scale = 1.0f;
    float theta = 10000.0f;
    float epsilon = 1e-6f;
    int axis = -1;
    bool pretransposed_rhs = false;
    size_t position_offset = 0;
    size_t slice_start = 0;
    size_t slice_length = 0;
    size_t window_size = 0;
    bool is_causal = true;  
    std::vector<size_t> new_shape;
    std::vector<size_t> permutation;
    Precision output_precision = Precision::INT8;
    BroadcastInfo broadcast_info;
    ComputeBackend backend = ComputeBackend::CPU;

    size_t dilation = 1;
    size_t stride = 1;
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = 0;
    size_t random_seed = 0;
    
    size_t index_value = 0;  
    size_t num_classes = 0; 
    size_t num_groups = 0;
    size_t dst_height = 0;
    size_t dst_width = 0;

    std::vector<float> bias_values;
    std::vector<uint32_t> bias_indices;

    const int8_t* cached_keys_int8 = nullptr;
    const int8_t* cached_values_int8 = nullptr;
    const float* cached_k_scales = nullptr;
    const float* cached_v_scales = nullptr;
    size_t cache_seq_len = 0;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
};

struct GraphNode {
    size_t id;
    OpType op_type;
    std::vector<size_t> input_ids;
    BufferDesc output_buffer;
    OpParams params;
    
    GraphNode(size_t node_id, OpType type);
};

template<typename T>
void dispatch_binary_op(OpType op, const T* lhs, const T* rhs, T* output, size_t count);

template<typename T>
void dispatch_unary_op(OpType op, const T* input, T* output, size_t count, float param = 0.0f);

void compute_node_optimized(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_fused_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_reshape_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_precision_cast_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_sample_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_scatter_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_groupnorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_persistent_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);
void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

void shrink_thread_local_buffers();

class BufferPool {
public:
    BufferPool() = default;
    ~BufferPool() = default;

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    char* acquire(size_t byte_size);
    void release(char* ptr, size_t byte_size);
    void clear();

    size_t active_bytes() const { return active_bytes_; }
    size_t pool_bytes() const { return pool_bytes_; }
    size_t peak_bytes() const { return peak_bytes_; }

private:
    std::unordered_map<size_t, std::vector<std::unique_ptr<char[]>>> free_buffers_;
    size_t active_bytes_ = 0;
    size_t pool_bytes_ = 0;
    size_t peak_bytes_ = 0;

    size_t round_up_size(size_t size) const;
};

namespace ValidationUtils {
    void validate_tensor_dims(const std::vector<size_t>& shape, size_t required_dims, const std::string& op_name);
    void validate_precision(Precision actual, Precision required, const std::string& op_name);
    void validate_input_count(size_t actual, size_t required, const std::string& op_name);
}


class CactusGraph {
public:
    CactusGraph();

    struct DebugNodeEntry {
        uint32_t layer_idx;
        std::string name;
        size_t node_id;
    };
    
    size_t input(const std::vector<size_t>& shape, Precision precision = Precision::INT8);
    size_t precision_cast(size_t input, Precision target_precision);
    size_t quantize_activations(size_t input);  
    
    size_t add(size_t input1, size_t input2);
    size_t add_clipped(size_t input1, size_t input2);  
    size_t subtract(size_t input1, size_t input2);
    size_t multiply(size_t input1, size_t input2);
    size_t divide(size_t input1, size_t input2);
    
    
    size_t scalar_add(size_t input, float value);
    size_t scalar_subtract(size_t input, float value);
    size_t scalar_multiply(size_t input, float value);
    size_t scalar_divide(size_t input, float value);
    size_t scalar_exp(size_t input);
    size_t scalar_sqrt(size_t input);
    size_t scalar_cos(size_t input);
    size_t scalar_sin(size_t input);
    
    size_t silu(size_t input);
    size_t gelu(size_t input);
    size_t gelu_erf(size_t input);
    size_t tanh(size_t input);
    
    size_t matmul(size_t input1, size_t input2, bool pretransposed_rhs = false, ComputeBackend backend = ComputeBackend::CPU);
    size_t transpose(size_t input, ComputeBackend backend = ComputeBackend::CPU);
    size_t transposeN(size_t input, const std::vector<size_t>& permutation, ComputeBackend backend = ComputeBackend::CPU);
    size_t reshape(size_t input, const std::vector<size_t>& new_shape);
    size_t slice(size_t input, int axis, size_t start, size_t length);
    size_t index(size_t input, size_t index_value, int dim);
    
    size_t sum(size_t input, int axis);
    size_t mean(size_t input, int axis);
    size_t variance(size_t input, int axis);
    size_t min(size_t input, int axis);
    size_t max(size_t input, int axis);
    
    size_t gather(size_t embeddings, size_t indices);
    size_t mmap_embeddings(const std::string& filename);
    size_t mmap_weights(const std::string& filename);
    void set_grouped_scales(size_t node_id, size_t group_size, size_t num_groups, void* scales_ptr);
    void set_interleaved(size_t node_id, bool interleaved, size_t original_N);

    void release_weight_pages(size_t node_id);
    void prefetch_weight_pages(size_t node_id);
    void release_all_weight_pages();
    size_t embedding(const std::string& filename, size_t indices);
    size_t embedding(size_t embedding_tensor, size_t indices);
    size_t bilinear_interpolation(size_t pos_embeds, size_t dst_height, size_t dst_width);

    size_t layernorm(size_t input, size_t weight, size_t bias, float epsilon = 1e-5f);
    size_t layernorm(size_t input, size_t weight, float epsilon = 1e-5f);  // No bias version
    size_t groupnorm(size_t input, size_t weight, size_t bias, size_t num_groups = 32, float epsilon = 1e-5f);
    size_t topk(size_t input, size_t k);
    size_t rms_norm(size_t input, size_t weight, float epsilon = 1e-5f);
    size_t rope(size_t input, float theta, size_t position_offset = 0, ComputeBackend backend = ComputeBackend::CPU);
    size_t rope_gptj(size_t input, float theta, size_t position_offset = 0, size_t rot_dim = 0, ComputeBackend backend = ComputeBackend::CPU);
    size_t softmax(size_t input, int axis = -1);
    size_t attention(size_t query, size_t key, size_t value, float scale, bool is_causal = true, ComputeBackend backend = ComputeBackend::CPU);
    size_t attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, ComputeBackend backend = ComputeBackend::CPU);
    size_t attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, size_t window_size, ComputeBackend backend = ComputeBackend::CPU);

    size_t attention_int8_hybrid(size_t query, size_t key_new, size_t value_new, float scale, size_t position_offset,
                                 const int8_t* cached_keys, const int8_t* cached_values,
                                 const float* k_scales, const float* v_scales,
                                 size_t cache_len, size_t num_kv_heads, size_t head_dim, size_t window_size = 0);

    size_t conv1d_causal(size_t input, size_t weight, size_t kernel_size, size_t dilation = 1);
    size_t conv1d_k3(size_t input, size_t weight, size_t stride);
    size_t conv1d_k7s3(size_t input, size_t weight, size_t bias);
    size_t conv1d(size_t input, size_t weight, size_t stride);
    size_t conv1d(size_t input, size_t weight, size_t bias, size_t stride);
    
    size_t sample(size_t logits, float temperature = 0.6f, float top_p = 0.95f, size_t top_k = 20,
                  const std::unordered_map<uint32_t, float>& logit_bias = {});
    
    size_t concat(size_t input1, size_t input2, int axis = 0);
    size_t scatter_topk(size_t indices, size_t values, size_t num_classes);
    
    void set_input(size_t node_id, const void* data, Precision precision);
    void set_external_input(size_t node_id, void* data, Precision precision);
    void* get_output(size_t node_id);
    
    void execute(const std::string& profile_file = "");
    void hard_reset();
    void soft_reset();
    void soft_reset_keep_pool();
    void set_prefill_mode(bool enabled) { prefill_mode_ = enabled; }

    void register_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id);
    void capture_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id);
    const std::vector<DebugNodeEntry>& get_debug_nodes() const;
    void clear_debug_nodes();
    
    size_t add_node(OpType op_type, const std::vector<size_t>& inputs, const std::vector<size_t>& output_shape, const OpParams& params = {});
    const BufferDesc& get_output_buffer(size_t node_id) const;
    void allocate_buffers();
    size_t get_node_count() const;

    size_t persistent(size_t source_node);
    bool is_populated(size_t persistent_node_id) const;
    void invalidate_persistent(size_t persistent_node_id);

    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<size_t, size_t> node_index_map_;

private:
    size_t next_node_id_;
    std::vector<std::unique_ptr<GraphFile::MappedFile>> mapped_files_;
    std::unordered_map<std::string, size_t> weight_cache_;
    std::unordered_map<size_t, size_t> node_to_mapped_file_;
    std::vector<DebugNodeEntry> debug_nodes_;
    BufferPool buffer_pool_;
    bool prefill_mode_ = false;
    
    std::unordered_set<size_t> persistent_node_ids_;
    std::unordered_set<size_t> populated_node_ids_;
};


namespace GraphFile {
    struct LoadedNode {
        size_t node_id;
        std::vector<size_t> shape;
        Precision precision;
        size_t byte_size;
    };
    
    void save_node(CactusGraph& graph, size_t node_id, const std::string& filename);
    
    class MappedFile {
    public:
        MappedFile(const std::string& filename);
        ~MappedFile();

        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;
        MappedFile(MappedFile&& other) noexcept;
        MappedFile& operator=(MappedFile&& other) noexcept;

        const std::vector<size_t>& shape() const;
        Precision precision() const;
        size_t byte_size() const;

        size_t group_size() const { return group_size_; }
        size_t num_groups() const { return num_groups_; }
        const void* scales_data() const;

        bool is_interleaved() const { return is_interleaved_; }
        size_t original_N() const { return original_N_; }

        void* data();
        const void* data() const;

        template<typename T>
        const T* typed_data() const;

        void release_pages();
        void prefetch_pages();

    private:
        int fd_;
        void* mapped_data_;
        size_t file_size_, data_offset_;
        std::vector<size_t> shape_;
        Precision precision_;
        size_t byte_size_;
        size_t group_size_ = 0;
        size_t num_groups_ = 0;
        size_t scales_offset_ = 0;
        size_t scales_bytes_ = 0;
        uint32_t alignment_ = 32;

        bool is_interleaved_ = false;
        size_t original_N_ = 0;

        std::unique_ptr<int8_t[]> unpacked_data_;  

        void parse_header();
        void apply_madvise_hints();
        void unpack_int4_data();
    };
}

#endif 
