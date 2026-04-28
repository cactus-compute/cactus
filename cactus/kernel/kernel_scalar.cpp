#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    const __fp16 scalar_f16 = static_cast<__fp16>(scalar_value);
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;

    switch (op_type) {
        case ScalarOpType::ADD: {
            const float16x8_t sv = vdupq_n_f16(scalar_f16);
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_BASIC,
                [sv](float16x8_t v) { return vaddq_f16(v, sv); },
                [scalar_f16](__fp16 v) { return static_cast<__fp16>(v + scalar_f16); });
            break;
        }
        case ScalarOpType::SUBTRACT: {
            const float16x8_t sv = vdupq_n_f16(scalar_f16);
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_BASIC,
                [sv](float16x8_t v) { return vsubq_f16(v, sv); },
                [scalar_f16](__fp16 v) { return static_cast<__fp16>(v - scalar_f16); });
            break;
        }
        case ScalarOpType::MULTIPLY: {
            const float16x8_t sv = vdupq_n_f16(scalar_f16);
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_BASIC,
                [sv](float16x8_t v) { return vmulq_f16(v, sv); },
                [scalar_f16](__fp16 v) { return static_cast<__fp16>(v * scalar_f16); });
            break;
        }
        case ScalarOpType::DIVIDE: {
            const float16x8_t sv = vdupq_n_f16(scalar_f16);
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_BASIC,
                [sv](float16x8_t v) { return vdivq_f16(v, sv); },
                [scalar_f16](__fp16 v) { return static_cast<__fp16>(v / scalar_f16); });
            break;
        }
        case ScalarOpType::ABS:
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_BASIC,
                [](float16x8_t v) { return vabsq_f16(v); },
                [](__fp16 v) { return static_cast<__fp16>(std::abs(static_cast<float>(v))); });
            break;

        case ScalarOpType::EXP:
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [](float16x8_t v) { return apply_f32_op_on_f16x8(v, fast_exp_f32x4); },
                [](__fp16 v) { return static_cast<__fp16>(std::exp(static_cast<float>(v))); }, 2);
            break;

        case ScalarOpType::SQRT:
            elementwise_op_f16(input, output, num_elements, use_streaming,
                CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [](float16x8_t v) { return apply_f32_op_on_f16x8(v, vsqrtq_f32); },
                [](__fp16 v) { return static_cast<__fp16>(std::sqrt(static_cast<float>(v))); }, 2);
            break;

        case ScalarOpType::POW:
            CactusThreading::parallel_for(num_elements,
                CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start, size_t end) {
                    for (size_t i = start; i < end; ++i) {
                        output[i] = static_cast<__fp16>(
                            std::pow(static_cast<float>(input[i]), static_cast<float>(scalar_f16)));
                    }
                });
            break;

        case ScalarOpType::COS:
        case ScalarOpType::SIN:
            CactusThreading::parallel_for(num_elements,
                CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start, size_t end) {
                    for (size_t i = start; i < end; ++i) {
                        float val = static_cast<float>(input[i]);
                        float result = (op_type == ScalarOpType::COS) ? std::cos(val) : std::sin(val);
                        output[i] = static_cast<__fp16>(result);
                    }
                });
            break;

        case ScalarOpType::LOG:
            CactusThreading::parallel_for(num_elements,
                CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start, size_t end) {
                    for (size_t i = start; i < end; ++i) {
                        output[i] = static_cast<__fp16>(std::log(static_cast<float>(input[i])));
                    }
                });
            break;
    }
}
