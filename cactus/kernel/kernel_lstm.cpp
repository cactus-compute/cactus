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

    const size_t simd_end = (hidden_size / SIMD_WIDTH) * SIMD_WIDTH;

    CactusThreading::parallel_for(batch_size, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t batch_start, size_t batch_end) {
            const float32x4_t one = vdupq_n_f32(1.0f);

            for (size_t b = batch_start; b < batch_end; ++b) {
                const size_t gate_offset = b * gate_size;
                const size_t hidden_offset = b * hidden_size;

                for (size_t h = 0; h < simd_end; h += SIMD_WIDTH) {
                    float16x8_t i_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + h]),
                                                             vld1q_f16(&gates_hh[gate_offset + h])),
                                                   vaddq_f16(vld1q_f16(&bias_ih[h]),
                                                             vld1q_f16(&bias_hh[h])));

                    float16x8_t f_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + hidden_size + h]),
                                                             vld1q_f16(&gates_hh[gate_offset + hidden_size + h])),
                                                   vaddq_f16(vld1q_f16(&bias_ih[hidden_size + h]),
                                                             vld1q_f16(&bias_hh[hidden_size + h])));

                    float16x8_t g_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 2 * hidden_size + h]),
                                                             vld1q_f16(&gates_hh[gate_offset + 2 * hidden_size + h])),
                                                   vaddq_f16(vld1q_f16(&bias_ih[2 * hidden_size + h]),
                                                             vld1q_f16(&bias_hh[2 * hidden_size + h])));

                    float16x8_t o_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[gate_offset + 3 * hidden_size + h]),
                                                             vld1q_f16(&gates_hh[gate_offset + 3 * hidden_size + h])),
                                                   vaddq_f16(vld1q_f16(&bias_ih[3 * hidden_size + h]),
                                                             vld1q_f16(&bias_hh[3 * hidden_size + h])));

                    float32x4_t i_low = vcvt_f32_f16(vget_low_f16(i_gate));
                    float32x4_t i_high = vcvt_f32_f16(vget_high_f16(i_gate));
                    float16x8_t i_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(i_low))))),
                                                     vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(i_high))))));

                    float32x4_t f_low = vcvt_f32_f16(vget_low_f16(f_gate));
                    float32x4_t f_high = vcvt_f32_f16(vget_high_f16(f_gate));
                    float16x8_t f_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(f_low))))),
                                                     vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(f_high))))));

                    float32x4_t g_low = vcvt_f32_f16(vget_low_f16(g_gate));
                    float32x4_t g_high = vcvt_f32_f16(vget_high_f16(g_gate));
                    float16x8_t g_act = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(g_low)),
                                                     vcvt_f16_f32(fast_tanh_f32x4(g_high)));

                    float32x4_t o_low = vcvt_f32_f16(vget_low_f16(o_gate));
                    float32x4_t o_high = vcvt_f32_f16(vget_high_f16(o_gate));
                    float16x8_t o_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(o_low))))),
                                                     vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(o_high))))));

                    float16x8_t c_prev_vec = vld1q_f16(&c_prev[hidden_offset + h]);
                    float16x8_t c_update = vfmaq_f16(vmulq_f16(f_act, c_prev_vec), i_act, g_act);
                    vst1q_f16(&c_new[hidden_offset + h], c_update);

                    float32x4_t c_low = vcvt_f32_f16(vget_low_f16(c_update));
                    float32x4_t c_high = vcvt_f32_f16(vget_high_f16(c_update));
                    float16x8_t c_tanh = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(c_low)),
                                                      vcvt_f16_f32(fast_tanh_f32x4(c_high)));
                    vst1q_f16(&h_new[hidden_offset + h], vmulq_f16(o_act, c_tanh));
                }

                for (size_t h = simd_end; h < hidden_size; ++h) {
                    float i_gate_val = static_cast<float>(gates_ih[gate_offset + h] +
                                                          gates_hh[gate_offset + h] +
                                                          bias_ih[h] + bias_hh[h]);
                    float f_gate_val = static_cast<float>(gates_ih[gate_offset + hidden_size + h] +
                                                          gates_hh[gate_offset + hidden_size + h] +
                                                          bias_ih[hidden_size + h] + bias_hh[hidden_size + h]);
                    float g_gate_val = static_cast<float>(gates_ih[gate_offset + 2 * hidden_size + h] +
                                                          gates_hh[gate_offset + 2 * hidden_size + h] +
                                                          bias_ih[2 * hidden_size + h] + bias_hh[2 * hidden_size + h]);
                    float o_gate_val = static_cast<float>(gates_ih[gate_offset + 3 * hidden_size + h] +
                                                          gates_hh[gate_offset + 3 * hidden_size + h] +
                                                          bias_ih[3 * hidden_size + h] + bias_hh[3 * hidden_size + h]);

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
