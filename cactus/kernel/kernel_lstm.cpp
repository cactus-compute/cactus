#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <vector>

void cactus_lstm_cell_f16(
    const __fp16* x_input,
    const __fp16* h_prev,
    const __fp16* c_prev,
    const __fp16* weight_ih,
    const __fp16* weight_hh,
    const __fp16* bias_ih,
    const __fp16* bias_hh,
    __fp16* h_new,
    __fp16* c_new,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size
) {
    constexpr size_t SIMD_WIDTH = 8;
    const size_t gate_size = 4 * hidden_size;

    std::vector<__fp16> gates_ih(batch_size * gate_size);
    std::vector<__fp16> gates_hh(batch_size * gate_size);

    cactus_matmul_f16(x_input, weight_ih, gates_ih.data(), batch_size, input_size, gate_size);
    cactus_matmul_f16(h_prev, weight_hh, gates_hh.data(), batch_size, hidden_size, gate_size);

    // Pre-add biases into single buffer to eliminate redundant loads
    std::vector<__fp16> bias_combined(gate_size);
    for (size_t i = 0; i < gate_size; ++i) {
        bias_combined[i] = bias_ih[i] + bias_hh[i];
    }

    const size_t simd_end = (hidden_size / (2 * SIMD_WIDTH)) * (2 * SIMD_WIDTH);

    CactusThreading::parallel_for(batch_size, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t batch_start, size_t batch_end) {

            for (size_t b = batch_start; b < batch_end; ++b) {
                const size_t gate_offset = b * gate_size;
                const size_t hidden_offset = b * hidden_size;

                // Process 16 elements per iteration (2x unroll)
                for (size_t h = 0; h < simd_end; h += 2 * SIMD_WIDTH) {
                    // Load and add gates + biases for first 8 elements
                    float16x8_t i_gate_0 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + h]),
                                                               vld1q_f16(&gates_hh[gate_offset + h])),
                                                     vld1q_f16(&bias_combined[h]));

                    float16x8_t f_gate_0 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + hidden_size + h]),
                                                               vld1q_f16(&gates_hh[gate_offset + hidden_size + h])),
                                                     vld1q_f16(&bias_combined[hidden_size + h]));

                    float16x8_t g_gate_0 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 2 * hidden_size + h]),
                                                               vld1q_f16(&gates_hh[gate_offset + 2 * hidden_size + h])),
                                                     vld1q_f16(&bias_combined[2 * hidden_size + h]));

                    float16x8_t o_gate_0 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 3 * hidden_size + h]),
                                                               vld1q_f16(&gates_hh[gate_offset + 3 * hidden_size + h])),
                                                     vld1q_f16(&bias_combined[3 * hidden_size + h]));

                    // Load and add gates + biases for second 8 elements
                    float16x8_t i_gate_1 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + h + SIMD_WIDTH]),
                                                               vld1q_f16(&gates_hh[gate_offset + h + SIMD_WIDTH])),
                                                     vld1q_f16(&bias_combined[h + SIMD_WIDTH]));

                    float16x8_t f_gate_1 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + hidden_size + h + SIMD_WIDTH]),
                                                               vld1q_f16(&gates_hh[gate_offset + hidden_size + h + SIMD_WIDTH])),
                                                     vld1q_f16(&bias_combined[hidden_size + h + SIMD_WIDTH]));

                    float16x8_t g_gate_1 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 2 * hidden_size + h + SIMD_WIDTH]),
                                                               vld1q_f16(&gates_hh[gate_offset + 2 * hidden_size + h + SIMD_WIDTH])),
                                                     vld1q_f16(&bias_combined[2 * hidden_size + h + SIMD_WIDTH]));

                    float16x8_t o_gate_1 = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 3 * hidden_size + h + SIMD_WIDTH]),
                                                               vld1q_f16(&gates_hh[gate_offset + 3 * hidden_size + h + SIMD_WIDTH])),
                                                     vld1q_f16(&bias_combined[3 * hidden_size + h + SIMD_WIDTH]));

                    // Apply activations to first 8 elements using fast_sigmoid
                    float32x4_t i_low_0 = vcvt_f32_f16(vget_low_f16(i_gate_0));
                    float32x4_t i_high_0 = vcvt_f32_f16(vget_high_f16(i_gate_0));
                    float16x8_t i_act_0 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(i_low_0)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(i_high_0)));

                    float32x4_t f_low_0 = vcvt_f32_f16(vget_low_f16(f_gate_0));
                    float32x4_t f_high_0 = vcvt_f32_f16(vget_high_f16(f_gate_0));
                    float16x8_t f_act_0 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(f_low_0)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(f_high_0)));

                    float32x4_t g_low_0 = vcvt_f32_f16(vget_low_f16(g_gate_0));
                    float32x4_t g_high_0 = vcvt_f32_f16(vget_high_f16(g_gate_0));
                    float16x8_t g_act_0 = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(g_low_0)),
                                                       vcvt_f16_f32(fast_tanh_f32x4(g_high_0)));

                    float32x4_t o_low_0 = vcvt_f32_f16(vget_low_f16(o_gate_0));
                    float32x4_t o_high_0 = vcvt_f32_f16(vget_high_f16(o_gate_0));
                    float16x8_t o_act_0 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(o_low_0)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(o_high_0)));

                    // Apply activations to second 8 elements using fast_sigmoid
                    float32x4_t i_low_1 = vcvt_f32_f16(vget_low_f16(i_gate_1));
                    float32x4_t i_high_1 = vcvt_f32_f16(vget_high_f16(i_gate_1));
                    float16x8_t i_act_1 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(i_low_1)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(i_high_1)));

                    float32x4_t f_low_1 = vcvt_f32_f16(vget_low_f16(f_gate_1));
                    float32x4_t f_high_1 = vcvt_f32_f16(vget_high_f16(f_gate_1));
                    float16x8_t f_act_1 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(f_low_1)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(f_high_1)));

                    float32x4_t g_low_1 = vcvt_f32_f16(vget_low_f16(g_gate_1));
                    float32x4_t g_high_1 = vcvt_f32_f16(vget_high_f16(g_gate_1));
                    float16x8_t g_act_1 = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(g_low_1)),
                                                       vcvt_f16_f32(fast_tanh_f32x4(g_high_1)));

                    float32x4_t o_low_1 = vcvt_f32_f16(vget_low_f16(o_gate_1));
                    float32x4_t o_high_1 = vcvt_f32_f16(vget_high_f16(o_gate_1));
                    float16x8_t o_act_1 = vcombine_f16(vcvt_f16_f32(fast_sigmoid_f32x4(o_low_1)),
                                                       vcvt_f16_f32(fast_sigmoid_f32x4(o_high_1)));

                    // Update cell state and hidden state for first 8 elements
                    float16x8_t c_prev_vec_0 = vld1q_f16(&c_prev[hidden_offset + h]);
                    float16x8_t c_update_0 = vfmaq_f16(vmulq_f16(f_act_0, c_prev_vec_0), i_act_0, g_act_0);
                    vst1q_f16(&c_new[hidden_offset + h], c_update_0);

                    float32x4_t c_low_0 = vcvt_f32_f16(vget_low_f16(c_update_0));
                    float32x4_t c_high_0 = vcvt_f32_f16(vget_high_f16(c_update_0));
                    float16x8_t c_tanh_0 = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(c_low_0)),
                                                        vcvt_f16_f32(fast_tanh_f32x4(c_high_0)));
                    vst1q_f16(&h_new[hidden_offset + h], vmulq_f16(o_act_0, c_tanh_0));

                    // Update cell state and hidden state for second 8 elements
                    float16x8_t c_prev_vec_1 = vld1q_f16(&c_prev[hidden_offset + h + SIMD_WIDTH]);
                    float16x8_t c_update_1 = vfmaq_f16(vmulq_f16(f_act_1, c_prev_vec_1), i_act_1, g_act_1);
                    vst1q_f16(&c_new[hidden_offset + h + SIMD_WIDTH], c_update_1);

                    float32x4_t c_low_1 = vcvt_f32_f16(vget_low_f16(c_update_1));
                    float32x4_t c_high_1 = vcvt_f32_f16(vget_high_f16(c_update_1));
                    float16x8_t c_tanh_1 = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(c_low_1)),
                                                        vcvt_f16_f32(fast_tanh_f32x4(c_high_1)));
                    vst1q_f16(&h_new[hidden_offset + h + SIMD_WIDTH], vmulq_f16(o_act_1, c_tanh_1));
                }

                // Scalar cleanup for remaining elements
                for (size_t h = simd_end; h < hidden_size; ++h) {
                    float i_gate_val = static_cast<float>(gates_ih[gate_offset + h] +
                                                          gates_hh[gate_offset + h] +
                                                          bias_combined[h]);
                    float f_gate_val = static_cast<float>(gates_ih[gate_offset + hidden_size + h] +
                                                          gates_hh[gate_offset + hidden_size + h] +
                                                          bias_combined[hidden_size + h]);
                    float g_gate_val = static_cast<float>(gates_ih[gate_offset + 2 * hidden_size + h] +
                                                          gates_hh[gate_offset + 2 * hidden_size + h] +
                                                          bias_combined[2 * hidden_size + h]);
                    float o_gate_val = static_cast<float>(gates_ih[gate_offset + 3 * hidden_size + h] +
                                                          gates_hh[gate_offset + 3 * hidden_size + h] +
                                                          bias_combined[3 * hidden_size + h]);

                    float i_act = 1.0f / (1.0f + expf(-i_gate_val));
                    float f_act = 1.0f / (1.0f + expf(-f_gate_val));
                    float g_act = tanhf(g_gate_val);
                    float o_act = 1.0f / (1.0f + expf(-o_gate_val));

                    float c_val = f_act * static_cast<float>(c_prev[hidden_offset + h]) + i_act * g_act;
                    c_new[hidden_offset + h] = static_cast<__fp16>(c_val);
                    h_new[hidden_offset + h] = static_cast<__fp16>(o_act * tanhf(c_val));
                }
            }
        });
}
