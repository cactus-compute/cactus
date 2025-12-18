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

struct NPUBufferRef {
    const __fp16* data;
    size_t count;
};

struct NPUPrefillDirectResult {
    NPUBufferRef hidden;
    std::vector<NPUBufferRef> k_caches;
    std::vector<NPUBufferRef> v_caches;
    bool valid;
};

class NPUPrefill {
public:
    virtual ~NPUPrefill() = default;
    virtual bool load(const std::string& model_path) = 0;
    virtual bool is_available() const = 0;
    virtual int get_chunk_size() const = 0;
    virtual int get_hidden_dim() const = 0;
    virtual int get_num_layers() const = 0;
    virtual int get_num_kv_heads() const = 0;
    virtual int get_head_dim() const = 0;

    virtual NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") = 0;
};


std::unique_ptr<NPUEncoder> create_encoder();
std::unique_ptr<NPUPrefill> create_prefill();
bool is_npu_available();

} // namespace npu
} // namespace cactus

#endif // CACTUS_NPU_H