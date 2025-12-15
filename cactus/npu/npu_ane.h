#ifndef CACTUS_NPU_ANE_H
#define CACTUS_NPU_ANE_H

#include "npu.h"

#if defined(__APPLE__)
#define CACTUS_HAS_ANE 1
#else
#define CACTUS_HAS_ANE 0
#endif

namespace cactus {
namespace npu {

#if CACTUS_HAS_ANE


class ANEEncoder : public NPUEncoder {
public:
    ANEEncoder();
    ~ANEEncoder() override;

    ANEEncoder(const ANEEncoder&) = delete;
    ANEEncoder& operator=(const ANEEncoder&) = delete;

    ANEEncoder(ANEEncoder&& other) noexcept;
    ANEEncoder& operator=(ANEEncoder&& other) noexcept;

    bool load(const std::string& model_path) override;

    bool preallocate(const std::vector<int>& input_shape,
                     const std::string& input_name = "x",
                     const std::string& output_name = "") override;

    size_t encode(const __fp16* input,
                  __fp16* output,
                  const std::vector<int>& shape,
                  const std::string& input_name = "x",
                  const std::string& output_name = "") override;

    bool is_available() const override;

    std::vector<int> get_input_shape() const override;

    std::vector<int> get_output_shape() const override;

    __fp16* get_output_buffer() override;

    size_t get_output_buffer_size() const override;

private:
    void* impl_;
};

class ANEPrefill : public NPUPrefill {
public:
    ANEPrefill();
    ~ANEPrefill() override;

    ANEPrefill(const ANEPrefill&) = delete;
    ANEPrefill& operator=(const ANEPrefill&) = delete;

    ANEPrefill(ANEPrefill&& other) noexcept;
    ANEPrefill& operator=(ANEPrefill&& other) noexcept;

    bool load(const std::string& model_path) override;
    bool is_available() const override;

    int get_chunk_size() const override;
    int get_hidden_dim() const override;
    int get_num_layers() const override;
    int get_num_kv_heads() const override;
    int get_head_dim() const override;

    NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") override;

private:
    void* impl_;
    int chunk_size_ = 256;
    int hidden_dim_ = 0;
    int num_layers_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
};

#else

class ANEEncoder : public NPUEncoder {
public:
    ANEEncoder() = default;
    ~ANEEncoder() override = default;

    bool load(const std::string&) override { return false; }

    bool preallocate(const std::vector<int>&,
                     const std::string& = "x",
                     const std::string& = "") override { return false; }

    size_t encode(const __fp16*,
                  __fp16*,
                  const std::vector<int>&,
                  const std::string& = "x",
                  const std::string& = "") override { return 0; }

    bool is_available() const override { return false; }

    std::vector<int> get_input_shape() const override { return {}; }

    std::vector<int> get_output_shape() const override { return {}; }

    __fp16* get_output_buffer() override { return nullptr; }

    size_t get_output_buffer_size() const override { return 0; }
};

class ANEPrefill : public NPUPrefill {
public:
    ANEPrefill() = default;
    ~ANEPrefill() override = default;

    bool load(const std::string&) override { return false; }
    bool is_available() const override { return false; }

    int get_chunk_size() const override { return 0; }
    int get_hidden_dim() const override { return 0; }
    int get_num_layers() const override { return 0; }
    int get_num_kv_heads() const override { return 0; }
    int get_head_dim() const override { return 0; }

    NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>&,
        int = 0,
        const std::string& = "x") override { return {}; }
};

#endif // CACTUS_HAS_ANE

} // namespace npu
} // namespace cactus

#endif // CACTUS_NPU_ANE_H