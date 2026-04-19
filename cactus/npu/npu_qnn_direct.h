#pragma once
#include "npu.h"

#ifdef CACTUS_HAS_QNN_DIRECT

#include <string>
#include <memory>
#include <vector>

namespace cactus {
namespace npu {

class QNNDirectPrefill : public NPUPrefill {
public:
    QNNDirectPrefill();
    ~QNNDirectPrefill() override;

    bool load(const std::string& model_folder) override;
    bool is_available() const override;
    int get_chunk_size() const override;
    int get_hidden_dim() const override;
    int get_num_layers() const override;
    int get_num_kv_heads() const override;
    int get_head_dim() const override;

    NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset,
        const std::string& input_name) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class QNNDirectEncoder : public NPUEncoder {
public:
    QNNDirectEncoder();
    ~QNNDirectEncoder() override;

    bool load(const std::string& model_path) override;
    bool preallocate(const std::vector<int>& input_shape,
                     const std::string& input_name = "x",
                     const std::string& output_name = "") override;
    size_t encode(const __fp16* input, __fp16* output,
                  const std::vector<int>& shape,
                  const std::string& input_name = "x",
                  const std::string& output_name = "") override;
    bool is_available() const override;
    std::vector<int> get_input_shape() const override;
    std::vector<int> get_output_shape() const override;
    __fp16* get_output_buffer() override;
    size_t get_output_buffer_size() const override;
    size_t encode_multimodal_input(
        const std::vector<NPUNamedInput>& inputs,
        __fp16* output,
        const std::string& output_name = "") override;
    void reset_state() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace npu
} // namespace cactus

#endif // CACTUS_HAS_QNN_DIRECT
