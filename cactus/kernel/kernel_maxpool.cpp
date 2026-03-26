#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <limits>

void cactus_maxpool1d_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t channels,
    size_t input_length,
    size_t kernel_size,
    size_t stride
) {
    const size_t output_length = (input_length - kernel_size) / stride + 1;

    CactusThreading::parallel_for(batch_size * channels, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t bc = start; bc < end; ++bc) {
                const size_t b = bc / channels;
                const size_t c = bc % channels;

                const __fp16* src = input + b * channels * input_length + c * input_length;
                __fp16* dst = output + b * channels * output_length + c * output_length;

                for (size_t i = 0; i < output_length; ++i) {
                    const size_t in_start = i * stride;
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (size_t k = 0; k < kernel_size; ++k) {
                        float val = static_cast<float>(src[in_start + k]);
                        if (val > max_val) max_val = val;
                    }

                    dst[i] = static_cast<__fp16>(max_val);
                }
            }
        });
}
