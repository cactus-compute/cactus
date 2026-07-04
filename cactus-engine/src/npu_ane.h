#ifndef CACTUS_NPU_ANE_H
#define CACTUS_NPU_ANE_H

#include "engine.h"

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

    bool load(const std::string& model_path, const std::string& compute_units = "") override;

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

    bool has_input(const std::string& name) const override;

    std::vector<int> get_input_shape_for(const std::string& name) const override;

    __fp16* get_output_buffer() override;

    size_t get_output_buffer_size() const override;

    size_t encode_multimodal_input(
        const std::vector<NPUNamedInput>& inputs,
        __fp16* output,
        const std::string& output_name = "") override;

private:
    void* impl_;
};

#else

class ANEEncoder : public NPUEncoder {
public:
    ANEEncoder() = default;
    ~ANEEncoder() override = default;

    bool load(const std::string&, const std::string& = "") override { return false; }

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

    bool has_input(const std::string&) const override { return false; }

    std::vector<int> get_input_shape_for(const std::string&) const override { return {}; }

    __fp16* get_output_buffer() override { return nullptr; }

    size_t get_output_buffer_size() const override { return 0; }

    size_t encode_multimodal_input(
        const std::vector<NPUNamedInput>&,
        __fp16*,
        const std::string& = "") override { return 0; }
};

#endif // CACTUS_HAS_ANE

} // namespace npu
} // namespace cactus

#endif // CACTUS_NPU_ANE_H