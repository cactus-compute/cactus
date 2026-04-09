#include "npu.h"

#if defined(CACTUS_USE_HAILO)

#include <hailo/hailort.hpp>
#include <hailo/quantization.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace cactus {
namespace npu {

namespace {

constexpr std::chrono::milliseconds kHailoInferTimeout(10000);

size_t shape_product(const std::vector<int>& shape) {
    size_t total = 1;
    for (int dim : shape) {
        total *= static_cast<size_t>(std::max(dim, 1));
    }
    return total;
}

size_t round_up_to_page(size_t bytes) {
    long page_size = ::sysconf(_SC_PAGESIZE);
    const size_t page = (page_size > 0) ? static_cast<size_t>(page_size) : static_cast<size_t>(4096);
    return ((bytes + page - 1) / page) * page;
}

struct AlignedBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    size_t capacity = 0;

    ~AlignedBuffer() {
        if (ptr) {
            std::free(ptr);
        }
    }

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    bool allocate(size_t requested_size) {
        if (ptr) {
            std::free(ptr);
            ptr = nullptr;
        }
        size = requested_size;
        capacity = round_up_to_page(requested_size);
        if (capacity == 0) {
            return false;
        }
        long page_size = ::sysconf(_SC_PAGESIZE);
        const size_t alignment = (page_size > 0) ? static_cast<size_t>(page_size) : static_cast<size_t>(4096);
        if (0 != ::posix_memalign(&ptr, alignment, capacity)) {
            ptr = nullptr;
            size = 0;
            capacity = 0;
            return false;
        }
        std::memset(ptr, 0, capacity);
        return true;
    }

    uint8_t* data() {
        return static_cast<uint8_t*>(ptr);
    }

    const uint8_t* data() const {
        return static_cast<const uint8_t*>(ptr);
    }
};

std::vector<int> infer_input_shape(const hailo_3d_image_shape_t& shape) {
    return {1, static_cast<int>(shape.height), static_cast<int>(shape.width), static_cast<int>(shape.features)};
}

std::vector<int> infer_output_shape(const hailo_3d_image_shape_t& shape) {
    return {static_cast<int>(shape.height), static_cast<int>(shape.width), static_cast<int>(shape.features)};
}

bool is_hef_path(const std::string& model_path) {
    namespace fs = std::filesystem;
    fs::path path(model_path);
    return fs::exists(path) && path.extension() == ".hef";
}

} // namespace

class HailoEncoder : public NPUEncoder {
public:
    bool load(const std::string& model_path) override {
        if (!is_hef_path(model_path)) {
            return false;
        }

        auto vdevice_exp = hailort::VDevice::create();
        if (!vdevice_exp) {
            std::cerr << "[cactus][hailo] Failed to create VDevice, status=" << vdevice_exp.status() << std::endl;
            return false;
        }
        vdevice_ = vdevice_exp.release();

        auto infer_model_exp = vdevice_->create_infer_model(model_path);
        if (!infer_model_exp) {
            std::cerr << "[cactus][hailo] Failed to create infer model from " << model_path
                      << ", status=" << infer_model_exp.status() << std::endl;
            return false;
        }
        infer_model_ = infer_model_exp.release();

        auto input_exp = infer_model_->input();
        auto output_exp = infer_model_->output();
        if (!input_exp || !output_exp) {
            std::cerr << "[cactus][hailo] Failed to query infer model streams" << std::endl;
            return false;
        }
        auto input_stream = input_exp.release();
        auto output_stream = output_exp.release();
        input_stream.set_format_type(HAILO_FORMAT_TYPE_UINT8);
        output_stream.set_format_type(HAILO_FORMAT_TYPE_UINT8);

        input_shape_ = infer_input_shape(input_stream.shape());
        output_shape_ = infer_output_shape(output_stream.shape());
        input_frame_size_ = input_stream.get_frame_size();
        output_frame_size_ = output_stream.get_frame_size();
        output_elements_ = shape_product(output_shape_);

        auto quant_infos = output_stream.get_quant_infos();
        if (!quant_infos.empty()) {
            output_quant_info_ = quant_infos.front();
            has_output_quant_info_ = hailort::Quantization::is_qp_valid(output_quant_info_);
        }

        auto configured_exp = infer_model_->configure();
        if (!configured_exp) {
            std::cerr << "[cactus][hailo] Failed to configure infer model, status="
                      << configured_exp.status() << std::endl;
            return false;
        }
        configured_ = std::make_unique<hailort::ConfiguredInferModel>(configured_exp.release());

        auto bindings_exp = configured_->create_bindings();
        if (!bindings_exp) {
            std::cerr << "[cactus][hailo] Failed to create bindings, status=" << bindings_exp.status() << std::endl;
            return false;
        }
        bindings_ = std::make_unique<hailort::ConfiguredInferModel::Bindings>(bindings_exp.release());

        available_ = preallocate(input_shape_, "x", "");
        if (available_) {
            std::cerr << "[cactus][hailo] Loaded HEF " << model_path
                      << " input=" << input_shape_[0] << "x" << input_shape_[1] << "x"
                      << input_shape_[2] << "x" << input_shape_[3]
                      << " output=" << output_shape_[0] << "x" << output_shape_[1] << "x"
                      << output_shape_[2] << std::endl;
        }
        return available_;
    }

    bool preallocate(const std::vector<int>&,
                     const std::string& input_name = "x",
                     const std::string& output_name = "") override {
        (void)input_name;
        (void)output_name;
        if (!configured_ || !bindings_) {
            return false;
        }
        if (!input_buffer_.allocate(input_frame_size_) || !output_buffer_u8_.allocate(output_frame_size_)) {
            return false;
        }
        output_buffer_fp16_.assign(output_elements_, static_cast<__fp16>(0));

        auto input_exp = bindings_->input();
        auto output_exp = bindings_->output();
        if (!input_exp || !output_exp) {
            return false;
        }

        auto status = input_exp.value().set_buffer(hailort::MemoryView(input_buffer_.data(), input_frame_size_));
        if (HAILO_SUCCESS != status) {
            std::cerr << "[cactus][hailo] Failed to bind input buffer, status=" << status << std::endl;
            return false;
        }
        status = output_exp.value().set_buffer(hailort::MemoryView(output_buffer_u8_.data(), output_frame_size_));
        if (HAILO_SUCCESS != status) {
            std::cerr << "[cactus][hailo] Failed to bind output buffer, status=" << status << std::endl;
            return false;
        }
        return true;
    }

    size_t encode(const __fp16*,
                  __fp16*,
                  const std::vector<int>&,
                  const std::string& = "x",
                  const std::string& = "") override {
        return 0;
    }

    bool supports_image_input() const override {
        return true;
    }

    size_t encode_image_uint8(const uint8_t* input,
                              __fp16* output,
                              const std::vector<int>& shape,
                              const std::string& input_name = "x",
                              const std::string& output_name = "") override {
        (void)input_name;
        (void)output_name;
        if (!available_ || !configured_ || !bindings_ || !input) {
            return 0;
        }
        if (shape_product(shape) != input_frame_size_) {
            std::cerr << "[cactus][hailo] Input image shape mismatch: expected " << input_frame_size_
                      << " bytes, got " << shape_product(shape) << std::endl;
            return 0;
        }

        std::memcpy(input_buffer_.data(), input, input_frame_size_);

        auto ready_status = configured_->wait_for_async_ready(kHailoInferTimeout);
        if (HAILO_SUCCESS != ready_status) {
            std::cerr << "[cactus][hailo] wait_for_async_ready failed, status=" << ready_status << std::endl;
            return 0;
        }

        auto job_exp = configured_->run_async(*bindings_, hailort::ASYNC_INFER_EMPTY_CALLBACK);
        if (!job_exp) {
            std::cerr << "[cactus][hailo] run_async failed, status=" << job_exp.status() << std::endl;
            return 0;
        }
        auto job = job_exp.release();
        auto wait_status = job.wait(kHailoInferTimeout);
        if (HAILO_SUCCESS != wait_status) {
            std::cerr << "[cactus][hailo] inference wait failed, status=" << wait_status << std::endl;
            return 0;
        }

        for (size_t i = 0; i < output_elements_; ++i) {
            float value = static_cast<float>(output_buffer_u8_.data()[i]);
            if (has_output_quant_info_) {
                value = hailort::Quantization::dequantize_output<float, uint8_t>(
                    output_buffer_u8_.data()[i], output_quant_info_);
            }
            output_buffer_fp16_[i] = static_cast<__fp16>(value);
        }

        if (output) {
            std::memcpy(output, output_buffer_fp16_.data(), output_elements_ * sizeof(__fp16));
        }
        return output_elements_;
    }

    bool is_available() const override {
        return available_;
    }

    std::vector<int> get_input_shape() const override {
        return input_shape_;
    }

    std::vector<int> get_output_shape() const override {
        return output_shape_;
    }

    __fp16* get_output_buffer() override {
        return output_buffer_fp16_.empty() ? nullptr : output_buffer_fp16_.data();
    }

    size_t get_output_buffer_size() const override {
        return output_elements_;
    }

    size_t encode_multimodal_input(const std::vector<NPUNamedInput>&,
                                   __fp16*,
                                   const std::string& = "") override {
        return 0;
    }

private:
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::InferModel> infer_model_;
    std::unique_ptr<hailort::ConfiguredInferModel> configured_;
    std::unique_ptr<hailort::ConfiguredInferModel::Bindings> bindings_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    size_t input_frame_size_ = 0;
    size_t output_frame_size_ = 0;
    size_t output_elements_ = 0;
    bool available_ = false;
    bool has_output_quant_info_ = false;
    hailo_quant_info_t output_quant_info_{};
    AlignedBuffer input_buffer_;
    AlignedBuffer output_buffer_u8_;
    std::vector<__fp16> output_buffer_fp16_;
};

std::unique_ptr<NPUEncoder> create_encoder() {
    return std::make_unique<HailoEncoder>();
}

std::unique_ptr<NPUPrefill> create_prefill() {
    return nullptr;
}

bool is_npu_available() {
    auto vdevice_exp = hailort::VDevice::create();
    return static_cast<bool>(vdevice_exp);
}

} // namespace npu
} // namespace cactus

#endif // defined(CACTUS_USE_HAILO)
