#ifndef CACTUS_NPU_H
#define CACTUS_NPU_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace cactus {
namespace npu {


class NPUEncoder {
public:
    virtual ~NPUEncoder() = default;

    virtual bool load(const std::string& model_path) = 0;

    virtual bool preallocate(const std::vector<int>& input_shape,
                             const std::string& input_name = "x",
                             const std::string& output_name = "") = 0;

    virtual size_t encode(const __fp16* input,
                          __fp16* output,
                          const std::vector<int>& shape,
                          const std::string& input_name = "x",
                          const std::string& output_name = "") = 0;

    virtual bool is_available() const = 0;

    virtual std::vector<int> get_input_shape() const = 0;

    virtual std::vector<int> get_output_shape() const = 0;

    virtual __fp16* get_output_buffer() = 0;

    virtual size_t get_output_buffer_size() const = 0;
};

std::unique_ptr<NPUEncoder> create_encoder();

bool is_npu_available();

// Output from NPU prefill - contains hidden state and per-layer KV caches
struct NPUPrefillOutput {
    std::string name;
    std::vector<int> shape;
    std::vector<__fp16> data;
};

// Direct buffer reference for zero-copy access to NPU outputs
struct NPUBufferRef {
    const __fp16* data;
    size_t count;  // number of elements
};

// Result from prefill_chunk_direct - provides direct pointers to internal buffers
struct NPUPrefillDirectResult {
    NPUBufferRef hidden;
    std::vector<NPUBufferRef> k_caches;  // one per layer
    std::vector<NPUBufferRef> v_caches;  // one per layer
    bool valid;
};

// NPU Prefill class for LLM prefill acceleration
// Unlike NPUEncoder which has single output, this handles multiple outputs:
// - hidden: [chunk_size, hidden_dim] - final hidden states
// - k_0..k_N: [chunk_size, num_kv_heads, head_dim] - key caches per layer
// - v_0..v_N: [chunk_size, num_kv_heads, head_dim] - value caches per layer
class NPUPrefill {
public:
    virtual ~NPUPrefill() = default;

    // Load the CoreML model for prefill
    virtual bool load(const std::string& model_path) = 0;

    // Check if model is loaded and available
    virtual bool is_available() const = 0;

    // Get the fixed chunk size (typically 256 for ANE models)
    virtual int get_chunk_size() const = 0;

    // Get model dimensions (inferred from model description)
    virtual int get_hidden_dim() const = 0;
    virtual int get_num_layers() const = 0;
    virtual int get_num_kv_heads() const = 0;
    virtual int get_head_dim() const = 0;

    // Run prefill on a chunk of embeddings
    // Input: embeddings [chunk_size, hidden_dim]
    // Input: position_offset - starting position for RoPE (0 for first chunk, chunk_size for second, etc.)
    // Returns: vector of outputs (hidden, k_0, v_0, k_1, v_1, ...)
    virtual std::vector<NPUPrefillOutput> prefill_chunk(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") = 0;

    // Zero-copy version: returns direct pointers to internal pre-allocated buffers
    // IMPORTANT: The returned pointers are only valid until the next prefill call
    virtual NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") = 0;
};

std::unique_ptr<NPUPrefill> create_prefill();

} // namespace npu
} // namespace cactus

#endif // CACTUS_NPU_H