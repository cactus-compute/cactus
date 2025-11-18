#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GraphFile {
    class MappedFile;
}

enum class Precision {
    INT8, 
    FP16,
    FP32
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
    RMS_NORM, ROPE, SOFTMAX, ATTENTION, CONV1D_CAUSAL, CONV1D_K3,
    SCALAR_ADD, SCALAR_SUBTRACT, SCALAR_MULTIPLY, SCALAR_DIVIDE, SCALAR_EXP, SCALAR_SQRT, SCALAR_COS, SCALAR_SIN,
    SILU, GELU,
    SAMPLE, CONCAT,
    SCATTER_TOPK,
    TOPK, LAYERNORM,
    INDEX,
};

struct PrecisionTraits {
    static constexpr size_t size_of(Precision prec) {
        switch (prec) {
            case Precision::INT8: return 1;
            case Precision::FP16: return 2;
            case Precision::FP32: return 4;
        }
        return 1;
    }
    
    static constexpr bool is_integer(Precision prec) {
        switch (prec) {
            case Precision::INT8: return true;
            case Precision::FP16: return false;
            case Precision::FP32: return false;
        }
        return true;
    }
    
    static constexpr bool is_floating_point(Precision prec) {
        switch (prec) {
            case Precision::INT8: return false;
            case Precision::FP16: return true;
            case Precision::FP32: return true;
        }
        return false;
    }
};

namespace Quantization {
    void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
    void fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);
    void dynamic_quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count, 
                                       float* computed_scale);
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

struct BufferDesc {
    std::vector<size_t> shape;
    size_t total_size;
    size_t byte_size;
    std::unique_ptr<char[]> data;
    void* external_data;
    Precision precision;
    float quantization_scale;
    
    BufferDesc();
    BufferDesc(const std::vector<size_t>& s, Precision prec = Precision::INT8, float scale = 1.0f);
    
    void* get_data();
    const void* get_data() const;
    
    template<typename T>
    T* data_as() { return static_cast<T*>(get_data()); }
    
    template<typename T>
    const T* data_as() const { return static_cast<const T*>(get_data()); }
    
    void allocate();
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
    bool is_causal = true;  // Default to causal for backward compatibility
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
    
    size_t index_value = 0;  // For INDEX operation
    size_t num_classes = 0;  // For scatter operations
    size_t dst_height = 0;
    size_t dst_width = 0;   
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
void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map);

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
    
    size_t add(size_t input1, size_t input2);
    size_t add_clipped(size_t input1, size_t input2);  // For FP16 residual connections (Gemma)
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
    void set_quantization_scale(size_t node_id, float scale);
    size_t embedding(const std::string& filename, size_t indices);
    size_t embedding(size_t embedding_tensor, size_t indices);
    size_t bilinear_interpolation(size_t pos_embeds, size_t dst_height, size_t dst_width);

    size_t layernorm(size_t input, size_t weight, size_t bias, float epsilon = 1e-5f);
    size_t topk(size_t input, size_t k);
    size_t rms_norm(size_t input, size_t weight, float epsilon = 1e-5f);
    size_t rope(size_t input, float theta, size_t position_offset = 0, ComputeBackend backend = ComputeBackend::CPU);
    size_t softmax(size_t input, int axis = -1);
    size_t attention(size_t query, size_t key, size_t value, float scale, bool is_causal = true, ComputeBackend backend = ComputeBackend::CPU);
    size_t attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, ComputeBackend backend = ComputeBackend::CPU);
    size_t attention(size_t query, size_t key, size_t value, float scale, size_t position_offset, size_t window_size, ComputeBackend backend = ComputeBackend::CPU);

    size_t conv1d_causal(size_t input, size_t weight, size_t kernel_size, size_t dilation = 1);
    size_t conv1d_k3(size_t input, size_t weight, size_t stride);
    
    size_t sample(size_t logits, float temperature = 0.6f, float top_p = 0.95f, size_t top_k = 20);
    
    size_t concat(size_t input1, size_t input2, int axis = 0);
    size_t scatter_topk(size_t indices, size_t values, size_t num_classes);
    
    void set_input(size_t node_id, const void* data, Precision precision);
    void set_external_input(size_t node_id, void* data, Precision precision);
    void* get_output(size_t node_id);
    
    void execute(const std::string& profile_file = "");
    void hard_reset();
    void soft_reset();

    void register_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id);
    void capture_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id);
    const std::vector<DebugNodeEntry>& get_debug_nodes() const;
    void clear_debug_nodes();
    
    size_t add_node(OpType op_type, const std::vector<size_t>& inputs, const std::vector<size_t>& output_shape, const OpParams& params = {});
    const BufferDesc& get_output_buffer(size_t node_id) const;
    void allocate_buffers();
    size_t get_node_count() const;

    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<size_t, size_t> node_index_map_;

private:
    size_t next_node_id_;
    std::vector<std::unique_ptr<GraphFile::MappedFile>> mapped_files_;
    std::unordered_map<std::string, size_t> weight_cache_;
    std::vector<DebugNodeEntry> debug_nodes_;
};


namespace GraphFile {
    struct LoadedNode {
        size_t node_id;
        std::vector<size_t> shape;
        Precision precision;
        size_t byte_size;
    };
    
    void save_node(CactusGraph& graph, size_t node_id, const std::string& filename);
    LoadedNode load_into_graph(CactusGraph& graph, const std::string& filename);
    
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
        float quantization_scale() const;
        
        void* data();
        const void* data() const;
        
        template<typename T>
        const T* typed_data() const;
        
        LoadedNode load_into_graph(CactusGraph& graph) const;
        
    private:
        int fd_;
        void* mapped_data_;
        size_t file_size_, data_offset_;
        std::vector<size_t> shape_;
        Precision precision_;
        size_t byte_size_;
        float quantization_scale_;
        void parse_header();
    };
    
    MappedFile mmap_load(const std::string& filename);
}

#endif 
