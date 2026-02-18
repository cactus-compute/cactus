#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <limits>
#include <random>
#include <iostream>

void cactus_relu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        __fp16 x = input[i];
        output[i] = x > static_cast<__fp16>(0) ? x : static_cast<__fp16>(0);
    }
}

void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            const float32x4_t one_f32 = vdupq_n_f32(1.0f);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));

                float32x4_t exp_low = fast_exp_f32x4(vnegq_f32(x_low));
                float32x4_t exp_high = fast_exp_f32x4(vnegq_f32(x_high));

                float32x4_t sigmoid_low = vdivq_f32(one_f32, vaddq_f32(one_f32, exp_low));
                float32x4_t sigmoid_high = vdivq_f32(one_f32, vaddq_f32(one_f32, exp_high));

                float32x4_t silu_low = vmulq_f32(x_low, sigmoid_low);
                float32x4_t silu_high = vmulq_f32(x_high, sigmoid_high);

                float16x8_t silu = vcombine_f16(vcvt_f16_f32(silu_low), vcvt_f16_f32(silu_high));
                vst1q_f16(&output[i], silu);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x_f32 = static_cast<float>(input[i]);
                float sigmoid = 1.0f / (1.0f + expf(-x_f32));
                output[i] = static_cast<__fp16>(x_f32 * sigmoid);
            }
        });
}

void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            const float32x4_t half = vdupq_n_f32(0.5f);
            const float32x4_t one = vdupq_n_f32(1.0f);
            const float32x4_t sqrt_2_pi_vec = vdupq_n_f32(sqrt_2_over_pi);
            const float32x4_t coeff_vec = vdupq_n_f32(coeff);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x_f16 = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_f16));

                float32x4_t x_cubed_low = vmulq_f32(vmulq_f32(x_low, x_low), x_low);
                float32x4_t x_cubed_high = vmulq_f32(vmulq_f32(x_high, x_high), x_high);

                float32x4_t inner_low = vfmaq_f32(x_low, coeff_vec, x_cubed_low);
                float32x4_t inner_high = vfmaq_f32(x_high, coeff_vec, x_cubed_high);
                inner_low = vmulq_f32(sqrt_2_pi_vec, inner_low);
                inner_high = vmulq_f32(sqrt_2_pi_vec, inner_high);

                float32x4_t tanh_low = fast_tanh_f32x4(inner_low);
                float32x4_t tanh_high = fast_tanh_f32x4(inner_high);

                float32x4_t gelu_low = vmulq_f32(vmulq_f32(half, x_low), vaddq_f32(one, tanh_low));
                float32x4_t gelu_high = vmulq_f32(vmulq_f32(half, x_high), vaddq_f32(one, tanh_high));

                float16x8_t gelu_f16 = vcombine_f16(vcvt_f16_f32(gelu_low), vcvt_f16_f32(gelu_high));
                vst1q_f16(&output[i], gelu_f16);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x = static_cast<float>(input[i]);
                float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                float gelu = 0.5f * x * (1.0f + tanhf(inner));
                output[i] = static_cast<__fp16>(gelu);
            }
        });
}

void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements)
{
    const float inv_sqrt2 = 0.70710678118654752440f;

    CactusThreading::parallel_for(
        num_elements,
        CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {

            constexpr size_t SIMD = 8; 
            size_t vec_end = start_idx + ((end_idx - start_idx) / SIMD) * SIMD;

            for (size_t i = start_idx; i < vec_end; i += SIMD) {

                float16x8_t xh = vld1q_f16(&input[i]);

                float32x4_t x0 = vcvt_f32_f16(vget_low_f16(xh));
                float32x4_t x1 = vcvt_f32_f16(vget_high_f16(xh));

                float32x4_t arg0 = vmulq_n_f32(x0, inv_sqrt2);
                float32x4_t arg1 = vmulq_n_f32(x1, inv_sqrt2);

                float arg0_s[4], arg1_s[4];
                float erf0_s[4], erf1_s[4];
                vst1q_f32(arg0_s, arg0);
                vst1q_f32(arg1_s, arg1);

                for (int j = 0; j < 4; j++) {
                    erf0_s[j] = erff(arg0_s[j]);
                    erf1_s[j] = erff(arg1_s[j]);
                }

                float32x4_t erf0 = vld1q_f32(erf0_s);
                float32x4_t erf1 = vld1q_f32(erf1_s);

                float32x4_t t0 = vaddq_f32(vdupq_n_f32(1.0f), erf0);
                float32x4_t t1 = vaddq_f32(vdupq_n_f32(1.0f), erf1);

                float32x4_t gelu0 = vmulq_f32(vmulq_n_f32(x0, 0.5f), t0);
                float32x4_t gelu1 = vmulq_f32(vmulq_n_f32(x1, 0.5f), t1);

                float16x8_t gelu_h = vcombine_f16(
                    vcvt_f16_f32(gelu0),
                    vcvt_f16_f32(gelu1)
                );

                vst1q_f16(&output[i], gelu_h);
            }

            for (size_t i = vec_end; i < end_idx; i++) {
                float x = (float)input[i];
                float arg = x * inv_sqrt2;
                float res = 0.5f * x * (1.0f + erff(arg));
                output[i] = (__fp16)res;
            }
        }
    );
}

void cactus_sigmoid_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            const float32x4_t one_f32 = vdupq_n_f32(1.0f);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));

                float32x4_t exp_low = fast_exp_f32x4(vnegq_f32(x_low));
                float32x4_t exp_high = fast_exp_f32x4(vnegq_f32(x_high));

                float32x4_t sigmoid_low = vdivq_f32(one_f32, vaddq_f32(one_f32, exp_low));
                float32x4_t sigmoid_high = vdivq_f32(one_f32, vaddq_f32(one_f32, exp_high));

                float16x8_t sigmoid = vcombine_f16(vcvt_f16_f32(sigmoid_low), vcvt_f16_f32(sigmoid_high));
                vst1q_f16(&output[i], sigmoid);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x_f32 = static_cast<float>(input[i]);
                float sigmoid = 1.0f / (1.0f + expf(-x_f32));
                output[i] = static_cast<__fp16>(sigmoid);
            }
        });
}

void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t x_f16 = vld1q_f16(&input[i]);

                float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_f16));

                float32x4_t tanh_low = fast_tanh_f32x4(x_low);
                float32x4_t tanh_high = fast_tanh_f32x4(x_high);

                float16x8_t tanh_f16 = vcombine_f16(vcvt_f16_f32(tanh_low), vcvt_f16_f32(tanh_high));
                vst1q_f16(&output[i], tanh_f16);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float x = static_cast<float>(input[i]);
                output[i] = static_cast<__fp16>(std::tanh(x));
            }
        });
}

void kernel_softmax_f16_single(const __fp16* input, __fp16* output, size_t vocab_size) {

    constexpr size_t SIMD_WIDTH = 8;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t VECTORIZED_WIDTH = SIMD_WIDTH * UNROLL_FACTOR;
    const size_t vocab_vectorized = (vocab_size / VECTORIZED_WIDTH) * VECTORIZED_WIDTH;

    float32x4_t max_vec[UNROLL_FACTOR * 2];
    for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
        max_vec[u] = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    }

    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t x_vec_f16 = vld1q_f16(&input[i + u * SIMD_WIDTH]);
            float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_vec_f16));
            float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_vec_f16));
            max_vec[u * 2] = vmaxq_f32(max_vec[u * 2], x_low);
            max_vec[u * 2 + 1] = vmaxq_f32(max_vec[u * 2 + 1], x_high);
        }
    }

    float32x4_t final_max = max_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
        final_max = vmaxq_f32(final_max, max_vec[u]);
    }

    float max_val = vmaxvq_f32(final_max);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        max_val = std::max(max_val, static_cast<float>(input[i]));
    }

    const float32x4_t max_broadcast = vdupq_n_f32(max_val);

    float32x4_t sum_vec[UNROLL_FACTOR * 2];
    for (size_t u = 0; u < UNROLL_FACTOR * 2; u++) {
        sum_vec[u] = vdupq_n_f32(0.0f);
    }

    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t x_vec_f16 = vld1q_f16(&input[i + u * SIMD_WIDTH]);

            float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_vec_f16));
            float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_vec_f16));

            float32x4_t exp_low = fast_exp_f32x4(vsubq_f32(x_low, max_broadcast));
            float32x4_t exp_high = fast_exp_f32x4(vsubq_f32(x_high, max_broadcast));

            float16x8_t exp_f16 = vcombine_f16(vcvt_f16_f32(exp_low), vcvt_f16_f32(exp_high));
            vst1q_f16(&output[i + u * SIMD_WIDTH], exp_f16);

            sum_vec[u * 2] = vaddq_f32(sum_vec[u * 2], exp_low);
            sum_vec[u * 2 + 1] = vaddq_f32(sum_vec[u * 2 + 1], exp_high);
        }
    }

    float32x4_t final_sum = sum_vec[0];
    for (size_t u = 1; u < UNROLL_FACTOR * 2; u++) {
        final_sum = vaddq_f32(final_sum, sum_vec[u]);
    }

    float sum = vaddvq_f32(final_sum);
    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        float exp_val = expf(static_cast<float>(input[i]) - max_val);
        output[i] = static_cast<__fp16>(exp_val);
        sum += exp_val;
    }

    const float inv_sum = 1.0f / sum;
    const float16x8_t inv_sum_vec_f16 = vdupq_n_f16(static_cast<__fp16>(inv_sum));

    for (size_t i = 0; i < vocab_vectorized; i += VECTORIZED_WIDTH) {
        for (size_t u = 0; u < UNROLL_FACTOR; u++) {
            float16x8_t exp_vec = vld1q_f16(&output[i + u * SIMD_WIDTH]);
            float16x8_t result = vmulq_f16(exp_vec, inv_sum_vec_f16);
            vst1q_f16(&output[i + u * SIMD_WIDTH], result);
        }
    }

    for (size_t i = vocab_vectorized; i < vocab_size; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(output[i]) * inv_sum);
    }
}

void cactus_softmax_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t vocab_size
) {
    CactusThreading::parallel_for(batch_size * seq_len, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const size_t offset = idx * vocab_size;
                kernel_softmax_f16_single(input + offset, output + offset, vocab_size);
            }
        });
}

void cactus_sample_f32(const float* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values, const uint32_t* bias_indices,
                       size_t bias_count) {

    if (temperature == 0.0f && top_p <= 0.0f && top_k == 0) {
        if (vocab_size == 0) {
            output[0] = 0;
            return;
        }
        size_t best_idx = 0;
        float best_val = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            float val = logits[i];
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }
        output[0] = static_cast<uint32_t>(best_idx);
        return;
    }

    std::vector<float> filtered_logits(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        filtered_logits[i] = logits[i];
    }

    if (bias_values && bias_indices && bias_count > 0) {
        for (size_t i = 0; i < bias_count; ++i) {
            uint32_t idx = bias_indices[i];
            if (idx < vocab_size) {
                filtered_logits[idx] += bias_values[i];
            }
        }
    }

    if (temperature > 0) {
        for (size_t i = 0; i < vocab_size; ++i) {
            filtered_logits[i] /= temperature;
        }
    }
    
    if (top_k > 0) {
        std::vector<std::pair<float, size_t>> logit_pairs;
        logit_pairs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_pairs.emplace_back(filtered_logits[i], i);
        }
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        if (top_k < vocab_size) {
            float kth_value = logit_pairs[top_k - 1].first;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (filtered_logits[i] < kth_value) {
                    filtered_logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    
    constexpr float min_p = 0.15f;
    if (min_p > 0.0f) {
        float max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
        if (!std::isinf(max_logit)) {
            std::vector<float> temp_probs(vocab_size);
            float sum = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (!std::isinf(filtered_logits[i])) {
                    temp_probs[i] = std::exp(filtered_logits[i] - max_logit);
                    sum += temp_probs[i];
                } else {
                    temp_probs[i] = 0.0f;
                }
            }

            if (sum > 0.0f) {
                for (size_t i = 0; i < vocab_size; ++i) {
                    temp_probs[i] /= sum;
                }

                float max_prob = *std::max_element(temp_probs.begin(), temp_probs.end());
                float threshold = max_prob * min_p;

                for (size_t i = 0; i < vocab_size; ++i) {
                    if (temp_probs[i] < threshold) {
                        filtered_logits[i] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted_logits;
        sorted_logits.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!std::isinf(filtered_logits[i])) {
                sorted_logits.emplace_back(filtered_logits[i], i);
            }
        }
        std::sort(sorted_logits.begin(), sorted_logits.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float max_logit = sorted_logits.empty() ? 0.0f : sorted_logits[0].first;
        std::vector<float> temp_probs;
        temp_probs.reserve(sorted_logits.size());
        float sum = 0.0f;
        for (const auto& pair : sorted_logits) {
            float prob = std::exp(pair.first - max_logit);
            temp_probs.push_back(prob);
            sum += prob;
        }

        for (float& prob : temp_probs) {
            prob /= sum;
        }

        float cumulative_prob = 0.0f;
        std::vector<bool> indices_to_remove(sorted_logits.size(), false);
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            cumulative_prob += temp_probs[i];
            if (cumulative_prob > top_p) {
                indices_to_remove[i] = true;
            }
        }

        if (!indices_to_remove.empty()) {
            for (size_t i = 1; i < indices_to_remove.size(); ++i) {
                indices_to_remove[i] = indices_to_remove[i-1] || indices_to_remove[i];
            }
            indices_to_remove[0] = false;
        }

        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            if (indices_to_remove[i]) {
                filtered_logits[sorted_logits[i].second] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    float max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
    if (std::isinf(max_logit)) {
        output[0] = 0;
        return;
    }
    
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isinf(filtered_logits[i])) {
            probs[i] = 0.0f;
        } else {
            probs[i] = std::exp(filtered_logits[i] - max_logit);
            sum += probs[i];
        }
    }
    
    if (sum == 0.0f) {
        output[0] = 0;
        return;
    }
    
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }
    
    uint32_t actual_seed = (random_seed == 0) ? std::random_device{}() : random_seed;
    std::mt19937 gen(actual_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);
    
    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (cumulative >= sample) {
            output[0] = static_cast<uint32_t>(i);
            return;
        }
    }
    
    for (size_t i = vocab_size; i > 0; --i) {
        if (probs[i-1] > 0.0f) {
            output[0] = static_cast<uint32_t>(i-1);
            return;
        }
    }
    
    output[0] = 0;
}

void cactus_sample_f16(const __fp16* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values, const uint32_t* bias_indices,
                       size_t bias_count) {

    if (temperature == 0.0f && top_p <= 0.0f && top_k == 0) {
        if (vocab_size == 0) {
            output[0] = 0;
            return;
        }
        size_t best_idx = 0;
        float best_val = static_cast<float>(logits[0]);
        for (size_t i = 1; i < vocab_size; ++i) {
            float val = static_cast<float>(logits[i]);
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }
        output[0] = static_cast<uint32_t>(best_idx);
        return;
    }

    std::vector<__fp16> filtered_logits(vocab_size);

    std::memcpy(filtered_logits.data(), logits, vocab_size * sizeof(__fp16));

    if (bias_values && bias_indices && bias_count > 0) {
        for (size_t i = 0; i < bias_count; ++i) {
            uint32_t idx = bias_indices[i];
            if (idx < vocab_size) {
                filtered_logits[idx] = static_cast<__fp16>(static_cast<float>(filtered_logits[idx]) + bias_values[i]);
            }
        }
    }

    if (temperature > 0) {
        __fp16 inv_temp = static_cast<__fp16>(1.0f / temperature);
        float16x8_t inv_temp_vec = vdupq_n_f16(inv_temp);
        size_t i = 0;
        for (; i + 8 <= vocab_size; i += 8) {
            float16x8_t logits_vec = vld1q_f16(&logits[i]);
            float16x8_t scaled = vmulq_f16(logits_vec, inv_temp_vec);
            vst1q_f16(&filtered_logits[i], scaled);
        }
        for (; i < vocab_size; ++i) {
            filtered_logits[i] = logits[i] * inv_temp;
        }
    } else {
        std::memcpy(filtered_logits.data(), logits, vocab_size * sizeof(__fp16));
    }

    static std::vector<uint32_t> token_history;
    static const size_t MAX_HISTORY = 128; 
    static const float REPETITION_PENALTY = 1.1f;

    if (!token_history.empty() && REPETITION_PENALTY != 1.0f) {
        const __fp16 penalty_inv = static_cast<__fp16>(1.0f / REPETITION_PENALTY);
        const __fp16 penalty = static_cast<__fp16>(REPETITION_PENALTY);

        for (uint32_t prev_token : token_history) {
            if (prev_token < vocab_size) {
                filtered_logits[prev_token] = (filtered_logits[prev_token] > static_cast<__fp16>(0))
                    ? static_cast<__fp16>(filtered_logits[prev_token] * penalty_inv)
                    : static_cast<__fp16>(filtered_logits[prev_token] * penalty);
            }
        }
    }

    if (top_k > 0) {
        std::vector<std::pair<__fp16, size_t>> logit_pairs;
        logit_pairs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_pairs.emplace_back(filtered_logits[i], i);
        }
        std::partial_sort(logit_pairs.begin(),
                         logit_pairs.begin() + std::min(top_k, vocab_size),
                         logit_pairs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        if (top_k < vocab_size) {
            __fp16 kth_value = logit_pairs[top_k - 1].first;
            __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < vocab_size; ++i) {
                if (filtered_logits[i] < kth_value) {
                    filtered_logits[i] = neg_inf;
                }
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<__fp16, size_t>> sorted_logits;
        sorted_logits.reserve(vocab_size);
        __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vocab_size; ++i) {
            if (filtered_logits[i] != neg_inf) {
                sorted_logits.emplace_back(filtered_logits[i], i);
            }
        }
        std::sort(sorted_logits.begin(), sorted_logits.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        __fp16 max_logit = sorted_logits.empty() ? static_cast<__fp16>(0.0f) : sorted_logits[0].first;
        std::vector<float> temp_probs;
        temp_probs.reserve(sorted_logits.size());
        float sum = 0.0f;
        for (const auto& pair : sorted_logits) {
            float prob = std::exp(static_cast<float>(pair.first - max_logit));
            temp_probs.push_back(prob);
            sum += prob;
        }

        for (float& prob : temp_probs) {
            prob /= sum;
        }

        float cumulative_prob = 0.0f;
        std::vector<bool> indices_to_remove(sorted_logits.size(), false);
        bool threshold_reached = false;
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            cumulative_prob += temp_probs[i];
            if (cumulative_prob > top_p && i > 0) { 
                threshold_reached = true;
            }
            if (threshold_reached) {
                indices_to_remove[i] = true;
            }
        }

        if (!indices_to_remove.empty()) {
            for (size_t i = 1; i < indices_to_remove.size(); ++i) {
                indices_to_remove[i] = indices_to_remove[i-1] || indices_to_remove[i];
            }
            indices_to_remove[0] = false;
        }
        
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            if (indices_to_remove[i]) {
                filtered_logits[sorted_logits[i].second] = neg_inf;
            }
        }
    }
    
    __fp16 max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
    __fp16 neg_inf = static_cast<__fp16>(-std::numeric_limits<float>::infinity());
    if (max_logit == neg_inf) {
        output[0] = 0;
        return;
    }
    
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (filtered_logits[i] == neg_inf) {
            probs[i] = 0.0f;
        } else {
            probs[i] = std::exp(static_cast<float>(filtered_logits[i] - max_logit));
            sum += probs[i];
        }
    }

    if (sum == 0.0f) {
        output[0] = 0;
        return;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    uint32_t actual_seed = (random_seed == 0) ? std::random_device{}() : random_seed;
    std::mt19937 gen(actual_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);

    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (cumulative >= sample) {
            output[0] = static_cast<uint32_t>(i);
            token_history.push_back(output[0]);
            if (token_history.size() > MAX_HISTORY) {
                token_history.erase(token_history.begin());
            }
            return;
        }
    }

    for (size_t i = vocab_size; i > 0; --i) {
        if (probs[i-1] > 0.0f) {
            output[0] = static_cast<uint32_t>(i-1);
            token_history.push_back(output[0]);
            if (token_history.size() > MAX_HISTORY) {
                token_history.erase(token_history.begin());
            }
            return;
        }
    }

    output[0] = 0;
    token_history.push_back(output[0]);
    if (token_history.size() > MAX_HISTORY) {
        token_history.erase(token_history.begin());
    }
}
