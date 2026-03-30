#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cmath>

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE, 
        [src, dst, scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 16) * 16;
            float32x4_t scale_vec = vdupq_n_f32(scale);
            
            for (size_t i = start; i < simd_end; i += 16) {
                int8x16_t input = vld1q_s8(&src[i]);
                
                int16x8_t low = vmovl_s8(vget_low_s8(input));
                int16x8_t high = vmovl_s8(vget_high_s8(input));
                
                int32x4_t low_low = vmovl_s16(vget_low_s16(low));
                int32x4_t low_high = vmovl_s16(vget_high_s16(low));
                int32x4_t high_low = vmovl_s16(vget_low_s16(high));
                int32x4_t high_high = vmovl_s16(vget_high_s16(high));
                
                float32x4_t f_low_low = vcvtq_f32_s32(low_low);
                float32x4_t f_low_high = vcvtq_f32_s32(low_high);
                float32x4_t f_high_low = vcvtq_f32_s32(high_low);
                float32x4_t f_high_high = vcvtq_f32_s32(high_high);
                
                vst1q_f32(&dst[i], vmulq_f32(f_low_low, scale_vec));
                vst1q_f32(&dst[i + 4], vmulq_f32(f_low_high, scale_vec));
                vst1q_f32(&dst[i + 8], vmulq_f32(f_high_low, scale_vec));
                vst1q_f32(&dst[i + 12], vmulq_f32(f_high_high, scale_vec));
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<float>(src[i]) * scale;
            }
        });
}

void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, inv_scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 16) * 16;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_vec = vdupq_n_f32(-128.0f);
            float32x4_t max_vec = vdupq_n_f32(127.0f);
            
            for (size_t i = start; i < simd_end; i += 16) {
                float32x4_t input_0 = vld1q_f32(&src[i]);
                float32x4_t input_1 = vld1q_f32(&src[i + 4]);
                float32x4_t input_2 = vld1q_f32(&src[i + 8]);
                float32x4_t input_3 = vld1q_f32(&src[i + 12]);
                
                float32x4_t scaled_0 = vmulq_f32(input_0, inv_scale_vec);
                float32x4_t scaled_1 = vmulq_f32(input_1, inv_scale_vec);
                float32x4_t scaled_2 = vmulq_f32(input_2, inv_scale_vec);
                float32x4_t scaled_3 = vmulq_f32(input_3, inv_scale_vec);
                
                scaled_0 = vmaxq_f32(vminq_f32(scaled_0, max_vec), min_vec);
                scaled_1 = vmaxq_f32(vminq_f32(scaled_1, max_vec), min_vec);
                scaled_2 = vmaxq_f32(vminq_f32(scaled_2, max_vec), min_vec);
                scaled_3 = vmaxq_f32(vminq_f32(scaled_3, max_vec), min_vec);
                
                int32x4_t int_0 = vcvtnq_s32_f32(scaled_0); 
                int32x4_t int_1 = vcvtnq_s32_f32(scaled_1);
                int32x4_t int_2 = vcvtnq_s32_f32(scaled_2);
                int32x4_t int_3 = vcvtnq_s32_f32(scaled_3);
                
                int16x8_t int16_low = vcombine_s16(vqmovn_s32(int_0), vqmovn_s32(int_1));
                int16x8_t int16_high = vcombine_s16(vqmovn_s32(int_2), vqmovn_s32(int_3));
                
                int8x16_t result = vcombine_s8(vqmovn_s16(int16_low), vqmovn_s16(int16_high));
                vst1q_s8(&dst[i], result);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                float quantized = src[i] * inv_scale;
                dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
            }
        });
}

void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            
            for (size_t i = start; i < simd_end; i += 8) {
                float16x8_t input = vld1q_f16(&src[i]);
                
                float32x4_t output_low = vcvt_f32_f16(vget_low_f16(input));
                float32x4_t output_high = vcvt_f32_f16(vget_high_f16(input));
                
                vst1q_f32(&dst[i], output_low);
                vst1q_f32(&dst[i + 4], output_high);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        });
}

void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            
            for (size_t i = start; i < simd_end; i += 8) {
                float32x4_t input_low = vld1q_f32(&src[i]);
                float32x4_t input_high = vld1q_f32(&src[i + 4]);
                
                float16x4_t output_low = vcvt_f16_f32(input_low);
                float16x4_t output_high = vcvt_f16_f32(input_high);
                
                float16x8_t output = vcombine_f16(output_low, output_high);
                vst1q_f16(&dst[i], output);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<__fp16>(src[i]);
            }
        });
}

void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            float32x4_t scale_vec = vdupq_n_f32(scale);
            
            for (size_t i = start; i < simd_end; i += 8) {
                int8x8_t input = vld1_s8(&src[i]);
                
                int16x8_t int16 = vmovl_s8(input);
                int32x4_t int32_low = vmovl_s16(vget_low_s16(int16));
                int32x4_t int32_high = vmovl_s16(vget_high_s16(int16));
                
                float32x4_t float_low = vcvtq_f32_s32(int32_low);
                float32x4_t float_high = vcvtq_f32_s32(int32_high);
                
                float_low = vmulq_f32(float_low, scale_vec);
                float_high = vmulq_f32(float_high, scale_vec);
                
                float16x8_t output = vcombine_f16(vcvt_f16_f32(float_low), vcvt_f16_f32(float_high));
                vst1q_f16(&dst[i], output);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<__fp16>(static_cast<float>(src[i]) * scale);
            }
        });
}

void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, inv_scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_vec = vdupq_n_f32(-128.0f);
            float32x4_t max_vec = vdupq_n_f32(127.0f);
            
            for (size_t i = start; i < simd_end; i += 8) {
                float16x8_t input = vld1q_f16(&src[i]);
                
                float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input));
                float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input));
                
                float32x4_t scaled_low = vmulq_f32(input_low, inv_scale_vec);
                float32x4_t scaled_high = vmulq_f32(input_high, inv_scale_vec);
                
                scaled_low = vmaxq_f32(vminq_f32(scaled_low, max_vec), min_vec);
                scaled_high = vmaxq_f32(vminq_f32(scaled_high, max_vec), min_vec);
                
                int32x4_t int_low = vcvtnq_s32_f32(scaled_low);
                int32x4_t int_high = vcvtnq_s32_f32(scaled_high);
                
                int16x8_t int16_combined = vcombine_s16(vqmovn_s32(int_low), vqmovn_s32(int_high));
                int8x8_t result = vqmovn_s16(int16_combined);
                
                vst1_s8(&dst[i], result);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                float quantized = static_cast<float>(src[i]) * inv_scale;
                dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
            }
        });
}

float cactus_fp16_max_abs(const __fp16* src, size_t count) {
    float32x4_t abs_max_vec = vdupq_n_f32(0.0f);
    const size_t simd_end = (count / 8) * 8;

    for (size_t i = 0; i < simd_end; i += 8) {
        float16x8_t input = vld1q_f16(&src[i]);

        float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input));
        float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input));

        float32x4_t abs_low = vabsq_f32(input_low);
        float32x4_t abs_high = vabsq_f32(input_high);

        abs_max_vec = vmaxq_f32(abs_max_vec, abs_low);
        abs_max_vec = vmaxq_f32(abs_max_vec, abs_high);
    }

    float max_abs = vmaxvq_f32(abs_max_vec);

    for (size_t i = simd_end; i < count; ++i) {
        float abs_val = std::abs(static_cast<float>(src[i]));
        max_abs = std::max(max_abs, abs_val);
    }

    return max_abs;
}

static inline float quantize_group_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count) {

    float32x4_t max_vec = vdupq_n_f32(0.0f);
    size_t k = 0;

    for (; k + 8 <= count; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_low_f16(vals))));
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_high_f16(vals))));
    }

    float max_abs = vmaxvq_f32(max_vec);
    for (; k < count; k++) {
        float val = std::abs(static_cast<float>(src[k]));
        if (val > max_abs) max_abs = val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

    k = 0;
    for (; k + 8 <= count; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(vals)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(vals)), inv_scale_vec));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t result = vqmovn_s16(vcombine_s16(s0, s1));
        vst1_s8(dst + k, result);
    }

    for (; k < count; k++) {
        float val = static_cast<float>(src[k]) * inv_scale;
        int32_t q = static_cast<int32_t>(roundf(val));
        q = std::max(-128, std::min(127, q));
        dst[k] = static_cast<int8_t>(q);
    }

    return scale;
}

void cactus_quantize_kv_fp16_to_int8(
    const __fp16* src,
    int8_t* dst,
    float* scales,
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t group_size
) {
    const size_t num_groups = (head_dim + group_size - 1) / group_size;

    CactusThreading::parallel_for(seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, scales, head_dim, group_size, num_groups](size_t start, size_t end) {
            for (size_t idx = start; idx < end; idx++) {
                const __fp16* head_src = src + idx * head_dim;
                int8_t* head_dst = dst + idx * head_dim;
                float* head_scales = scales + idx * num_groups;

                for (size_t g = 0; g < num_groups; g++) {
                    size_t group_start = g * group_size;
                    size_t group_count = std::min(group_size, head_dim - group_start);
                    head_scales[g] = quantize_group_fp16_to_int8(
                        head_src + group_start,
                        head_dst + group_start,
                        group_count
                    );
                }
            }
        });
}

void cactus_unpack_int4_to_int8(const uint8_t* packed, int8_t* unpacked, size_t unpacked_count) {
    assert (unpacked_count % 32 == 0 && "Unpacked count must be a multiple of 32 for int4 unpacking");
    const size_t num_groups = unpacked_count / 32;

    CactusThreading::parallel_for(num_groups, CactusThreading::Thresholds::ELEMENT_WISE,
        [packed, unpacked](size_t start, size_t end) {
            for (size_t group = start; group < end; ++group) {
                const uint8_t* src = &packed[group * 16];
                int8_t* dst = &unpacked[group * 32];

                int8x16_t high_signed, low_signed;
                unpack_int4_as_int8x16x2(src, high_signed, low_signed);
                vst1q_s8(dst, low_signed);
                vst1q_s8(dst + 16, high_signed);
            }
        });
}
