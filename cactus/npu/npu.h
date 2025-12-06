#ifndef CACTUS_NPU_H
#define CACTUS_NPU_H

#include <vector>
#include <string>
#include <memory>

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

} // namespace npu
} // namespace cactus

#endif // CACTUS_NPU_H