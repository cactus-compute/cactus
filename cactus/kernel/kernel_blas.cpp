#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>

static inline size_t compute_linear_index(const size_t* coords, const size_t* strides, size_t ndim) {
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

static inline void increment_coords(size_t* coords, const size_t* shape, size_t ndim) {
    for (int i = ndim - 1; i >= 0; --i) {
        coords[i]++;
        if (coords[i] < shape[i]) {
            break;
        }
        coords[i] = 0;
    }
}

void cactus_add_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            const size_t vectorized_end = ((end_idx - start_idx) / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE + start_idx;
    
            for (size_t i = start_idx; i < vectorized_end; i += NEON_VECTOR_SIZE) {
        int8x16_t a_vec = vld1q_s8(&a[i]);
        int8x16_t b_vec = vld1q_s8(&b[i]);
        
        int16x8_t a_low = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_low = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t b_high = vmovl_s8(vget_high_s8(b_vec));
        
        int16x8_t result_low = vaddq_s16(a_low, b_low);
        int16x8_t result_high = vaddq_s16(a_high, b_high);
        
        int8x16_t result_vec = vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
        vst1q_s8(&output[i], result_vec);
    }
    
            for (size_t i = vectorized_end; i < end_idx; ++i) {
        int32_t sum = static_cast<int32_t>(a[i]) + static_cast<int32_t>(b[i]);
        output[i] = clamp_to_int8(sum);
    }
        });
}

void cactus_subtract_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            const size_t vectorized_end = ((end_idx - start_idx) / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE + start_idx;
    
            for (size_t i = start_idx; i < vectorized_end; i += NEON_VECTOR_SIZE) {
        int8x16_t a_vec = vld1q_s8(&a[i]);
        int8x16_t b_vec = vld1q_s8(&b[i]);
        
        int16x8_t a_low = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_low = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t b_high = vmovl_s8(vget_high_s8(b_vec));
        
        int16x8_t result_low = vsubq_s16(a_low, b_low);
        int16x8_t result_high = vsubq_s16(a_high, b_high);
        
        int8x16_t result_vec = vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
        vst1q_s8(&output[i], result_vec);
    }
    
            for (size_t i = vectorized_end; i < end_idx; ++i) {
        int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        output[i] = clamp_to_int8(diff);
    }
        });
}

void cactus_multiply_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            const size_t vectorized_end = ((end_idx - start_idx) / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE + start_idx;
            
            for (size_t i = start_idx; i < vectorized_end; i += NEON_VECTOR_SIZE) {
                int8x16_t a_vec = vld1q_s8(&a[i]);
                int8x16_t b_vec = vld1q_s8(&b[i]);
                
                int16x8_t a_low = vmovl_s8(vget_low_s8(a_vec));
                int16x8_t a_high = vmovl_s8(vget_high_s8(a_vec));
                int16x8_t b_low = vmovl_s8(vget_low_s8(b_vec));
                int16x8_t b_high = vmovl_s8(vget_high_s8(b_vec));
                
                int16x8_t result_low = vmulq_s16(a_low, b_low);
                int16x8_t result_high = vmulq_s16(a_high, b_high);
                
                int8x16_t result_vec = vcombine_s8(vqmovn_s16(result_low), vqmovn_s16(result_high));
                vst1q_s8(&output[i], result_vec);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                int32_t product = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
                output[i] = clamp_to_int8(product);
            }
        });
}

void cactus_divide_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx; ++i) {
                if (b[i] == 0) {
                    output[i] = (a[i] >= 0) ? 127 : -128;
                } else {
                    float result = static_cast<float>(a[i]) / static_cast<float>(b[i]);
                    output[i] = clamp_to_int8(result);
                }
            }
        });
}

void cactus_add_f32(const float* a, const float* b, float* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t a_vec = vld1q_f32(&a[i]);
                float32x4_t b_vec = vld1q_f32(&b[i]);
                float32x4_t result_vec = vaddq_f32(a_vec, b_vec);
                vst1q_f32(&output[i], result_vec);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] + b[i];
            }
        });
}

void cactus_subtract_f32(const float* a, const float* b, float* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t a_vec = vld1q_f32(&a[i]);
                float32x4_t b_vec = vld1q_f32(&b[i]);
                float32x4_t result_vec = vsubq_f32(a_vec, b_vec);
                vst1q_f32(&output[i], result_vec);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] - b[i];
                    }
                });
        }
        
void cactus_multiply_f32(const float* a, const float* b, float* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t a_vec = vld1q_f32(&a[i]);
                float32x4_t b_vec = vld1q_f32(&b[i]);
                float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
                vst1q_f32(&output[i], result_vec);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] * b[i];
            }
        });
}

void cactus_divide_f32(const float* a, const float* b, float* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t a_vec = vld1q_f32(&a[i]);
                float32x4_t b_vec = vld1q_f32(&b[i]);
                float32x4_t result_vec = vdivq_f32(a_vec, b_vec);
                vst1q_f32(&output[i], result_vec);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] / b[i];
            }
        });
}

void cactus_add_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                              const size_t* a_strides, const size_t* b_strides,
                              const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = clamp_to_int8(static_cast<int32_t>(a[a_idx]) + static_cast<int32_t>(b[b_idx]));
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_subtract_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = clamp_to_int8(static_cast<int32_t>(a[a_idx]) - static_cast<int32_t>(b[b_idx]));
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_multiply_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = clamp_to_int8(static_cast<int32_t>(a[a_idx]) * static_cast<int32_t>(b[b_idx]));
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_divide_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        int32_t b_val = static_cast<int32_t>(b[b_idx]);
        if (b_val == 0) {
            output[linear_idx] = 0;
        } else {
            output[linear_idx] = clamp_to_int8(static_cast<int32_t>(a[a_idx]) / b_val);
        }
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_add_broadcast_f32(const float* a, const float* b, float* output,
                                  const size_t* a_strides, const size_t* b_strides,
                                  const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = a[a_idx] + b[b_idx];
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_subtract_broadcast_f32(const float* a, const float* b, float* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = a[a_idx] - b[b_idx];
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_multiply_broadcast_f32(const float* a, const float* b, float* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = a[a_idx] * b[b_idx];
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_divide_broadcast_f32(const float* a, const float* b, float* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }
    
    std::vector<size_t> coords(ndim, 0);
    
    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);
        
        output[linear_idx] = a[a_idx] / b[b_idx];
        
        increment_coords(coords.data(), output_shape, ndim);
    }
}


void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            constexpr float FP16_MAX = 65500.0f;
            const float32x4_t max_val = vdupq_n_f32(FP16_MAX);
            const float32x4_t min_val = vdupq_n_f32(-FP16_MAX);

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t a_vec = vld1q_f16(&a[i]);
                float16x8_t b_vec = vld1q_f16(&b[i]);

                float32x4_t a_low = vcvt_f32_f16(vget_low_f16(a_vec));
                float32x4_t a_high = vcvt_f32_f16(vget_high_f16(a_vec));
                float32x4_t b_low = vcvt_f32_f16(vget_low_f16(b_vec));
                float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b_vec));

                float32x4_t result_low = vaddq_f32(a_low, b_low);
                float32x4_t result_high = vaddq_f32(a_high, b_high);

                result_low = vminq_f32(vmaxq_f32(result_low, min_val), max_val);
                result_high = vminq_f32(vmaxq_f32(result_high, min_val), max_val);

                float16x4_t result_low_f16 = vcvt_f16_f32(result_low);
                float16x4_t result_high_f16 = vcvt_f16_f32(result_high);
                float16x8_t result_vec = vcombine_f16(result_low_f16, result_high_f16);

                vst1q_f16(&output[i], result_vec);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                float result = static_cast<float>(a[i]) + static_cast<float>(b[i]);
                result = std::fmin(std::fmax(result, -FP16_MAX), FP16_MAX);
                output[i] = static_cast<__fp16>(result);
            }
        });
}

void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t a_vec = vld1q_f16(&a[i]);
                float16x8_t b_vec = vld1q_f16(&b[i]);
                float16x8_t result_vec = vsubq_f16(a_vec, b_vec);
                vst1q_f16(&output[i], result_vec);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] - b[i];
            }
        });
}

void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t a_vec = vld1q_f16(&a[i]);
                float16x8_t b_vec = vld1q_f16(&b[i]);
                float16x8_t result_vec = vmulq_f16(a_vec, b_vec);
                vst1q_f16(&output[i], result_vec);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] * b[i];
            }
        });
}

void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t a_vec = vld1q_f16(&a[i]);
                float16x8_t b_vec = vld1q_f16(&b[i]);
                float16x8_t result_vec = vdivq_f16(a_vec, b_vec);
                vst1q_f16(&output[i], result_vec);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] / b[i];
            }
        });
}

void cactus_add_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                              const size_t* a_strides, const size_t* b_strides,
                              const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }

    std::vector<size_t> coords(ndim, 0);
    constexpr float FP16_MAX = 65500.0f;

    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);

        float result = static_cast<float>(a[a_idx]) + static_cast<float>(b[b_idx]);
        result = std::fmin(std::fmax(result, -FP16_MAX), FP16_MAX);
        output[linear_idx] = static_cast<__fp16>(result);

        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_subtract_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }

    std::vector<size_t> coords(ndim, 0);

    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);

        output[linear_idx] = a[a_idx] - b[b_idx];

        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_multiply_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }

    std::vector<size_t> coords(ndim, 0);

    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);

        output[linear_idx] = a[a_idx] * b[b_idx];

        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_divide_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }

    std::vector<size_t> coords(ndim, 0);

    for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
        size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);

        output[linear_idx] = a[a_idx] / b[b_idx];

        increment_coords(coords.data(), output_shape, ndim);
    }
}

void cactus_transpose_f32(const float* source, float* destination, const size_t* shape, const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx) {
    if (ndim == 2 && permutation[0] == 1 && permutation[1] == 0) {
        size_t num_rows = shape[0];
        size_t num_cols = shape[1];
        
        constexpr size_t THRESHOLD = 8192;
        constexpr size_t TILE_ROWS = 32;
        if (num_rows * num_cols >= THRESHOLD) {
            const size_t num_row_blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;
            
            CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [=](size_t start_block, size_t end_block) {
                    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                        size_t start_row = block_idx * TILE_ROWS;
                        size_t end_row = std::min(start_row + TILE_ROWS, num_rows);
                        
                        cactus_transpose_2d_f32(source, destination, num_rows, num_cols, start_row, end_row);
                    }
                });
        } else {
            cactus_transpose_2d_f32(source, destination, num_rows, num_cols, 0, num_rows);
        }
    } else {
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t src_idx = 0;
            size_t tmp_idx = idx;
            
            for (size_t i = 0; i < ndim; ++i) {
                size_t coord = tmp_idx % shape[permutation[ndim - 1 - i]];
                tmp_idx /= shape[permutation[ndim - 1 - i]];
                
                size_t stride = 1;
                for (size_t j = permutation[ndim - 1 - i] + 1; j < ndim; ++j) {
                    stride *= shape[j];
                }
                src_idx += coord * stride;
            }
            
            destination[idx] = source[src_idx];
        }
    }
} 

void cactus_concat_f32(const float* input1, const float* input2, float* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis) {
    if (axis < 0) axis += ndims;
    
    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= output_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndims; ++i) {
        inner_size *= output_shape[i];
    }
    
    size_t axis_size1 = shape1[axis];
    size_t axis_size2 = shape2[axis];
    
    size_t input1_stride = axis_size1 * inner_size;
    size_t input2_stride = axis_size2 * inner_size;
    size_t output_stride = (axis_size1 + axis_size2) * inner_size;
    
    CactusThreading::parallel_for(outer_size, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t outer = start; outer < end; ++outer) {
                const float* in1_ptr = input1 + outer * input1_stride;
                const float* in2_ptr = input2 + outer * input2_stride;
                float* out_ptr = output + outer * output_stride;
                
                size_t copy_size1 = axis_size1 * inner_size;
                std::memcpy(out_ptr, in1_ptr, copy_size1 * sizeof(float));
                
                size_t copy_size2 = axis_size2 * inner_size;
                std::memcpy(out_ptr + copy_size1, in2_ptr, copy_size2 * sizeof(float));
            }
        });
}

void cactus_concat_f16(const __fp16* input1, const __fp16* input2, __fp16* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis) {
    if (axis < 0) axis += ndims;

    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= output_shape[i];
    }

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndims; ++i) {
        inner_size *= output_shape[i];
    }

    size_t axis_size1 = shape1[axis];
    size_t axis_size2 = shape2[axis];

    size_t input1_stride = axis_size1 * inner_size;
    size_t input2_stride = axis_size2 * inner_size;
    size_t output_stride = (axis_size1 + axis_size2) * inner_size;

    CactusThreading::parallel_for(outer_size, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t outer = start; outer < end; ++outer) {
                const __fp16* in1_ptr = input1 + outer * input1_stride;
                const __fp16* in2_ptr = input2 + outer * input2_stride;
                __fp16* out_ptr = output + outer * output_stride;

                size_t copy_size1 = axis_size1 * inner_size;
                std::memcpy(out_ptr, in1_ptr, copy_size1 * sizeof(__fp16));

                size_t copy_size2 = axis_size2 * inner_size;
                std::memcpy(out_ptr + copy_size1, in2_ptr, copy_size2 * sizeof(__fp16));
            }
        });
}


void cactus_concat_int8(const int8_t* input1, const int8_t* input2, int8_t* output,
                        const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                        size_t ndims, int axis) {
    if (axis < 0) axis += ndims;
    
    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= output_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndims; ++i) {
        inner_size *= output_shape[i];
    }
    
    size_t axis_size1 = shape1[axis];
    size_t axis_size2 = shape2[axis];
    
    size_t input1_stride = axis_size1 * inner_size;
    size_t input2_stride = axis_size2 * inner_size;
    size_t output_stride = (axis_size1 + axis_size2) * inner_size;
    
    CactusThreading::parallel_for(outer_size, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t outer = start; outer < end; ++outer) {
                const int8_t* in1_ptr = input1 + outer * input1_stride;
                const int8_t* in2_ptr = input2 + outer * input2_stride;
                int8_t* out_ptr = output + outer * output_stride;
                
                size_t copy_size1 = axis_size1 * inner_size;
                std::memcpy(out_ptr, in1_ptr, copy_size1 * sizeof(int8_t));
                
                size_t copy_size2 = axis_size2 * inner_size;
                std::memcpy(out_ptr + copy_size1, in2_ptr, copy_size2 * sizeof(int8_t));
            }
        });
}

void cactus_transpose_2d_int8(const int8_t* source, int8_t* destination, size_t num_rows, size_t num_cols, size_t start_row, size_t end_row) {
    constexpr size_t TILE_SIZE = 64;
    constexpr size_t VECTOR_WIDTH = 16;

    for (size_t row_tile_start = start_row; row_tile_start < end_row; row_tile_start += TILE_SIZE) {
        const size_t row_tile_end = std::min(row_tile_start + TILE_SIZE, end_row);
        
            for (size_t col_tile_start = 0; col_tile_start < num_cols; col_tile_start += TILE_SIZE) {
            const size_t col_tile_end = std::min(col_tile_start + TILE_SIZE, num_cols);
            
            for (size_t row_block = row_tile_start; row_block < row_tile_end; row_block += VECTOR_WIDTH) {
                const size_t row_block_end = std::min(row_block + VECTOR_WIDTH, row_tile_end);
                
                for (size_t col_block = col_tile_start; col_block < col_tile_end; col_block += VECTOR_WIDTH) {
                    const size_t col_block_end = std::min(col_block + VECTOR_WIDTH, col_tile_end);
                    
                    if (row_block_end - row_block >= 8 && col_block_end - col_block >= 8) {
                        int8x8_t rows[8];
                        for (int i = 0; i < 8; i++) {
                            if (row_block + i < row_block_end) {
                                rows[i] = vld1_s8(&source[(row_block + i) * num_cols + col_block]);
                            } else {
                                rows[i] = vdup_n_s8(0);
                            }
                        }
                        
                        int8x8x2_t r01 = vtrn_s8(rows[0], rows[1]);
                        int8x8x2_t r23 = vtrn_s8(rows[2], rows[3]);
                        int8x8x2_t r45 = vtrn_s8(rows[4], rows[5]);
                        int8x8x2_t r67 = vtrn_s8(rows[6], rows[7]);
                        
                        int16x4x2_t r0123_low = vtrn_s16(vreinterpret_s16_s8(r01.val[0]), vreinterpret_s16_s8(r23.val[0]));
                        int16x4x2_t r0123_high = vtrn_s16(vreinterpret_s16_s8(r01.val[1]), vreinterpret_s16_s8(r23.val[1]));
                        int16x4x2_t r4567_low = vtrn_s16(vreinterpret_s16_s8(r45.val[0]), vreinterpret_s16_s8(r67.val[0]));
                        int16x4x2_t r4567_high = vtrn_s16(vreinterpret_s16_s8(r45.val[1]), vreinterpret_s16_s8(r67.val[1]));
                        
                        int32x2x2_t final_0123 = vtrn_s32(vreinterpret_s32_s16(r0123_low.val[0]), vreinterpret_s32_s16(r4567_low.val[0]));
                        int32x2x2_t final_4567 = vtrn_s32(vreinterpret_s32_s16(r0123_low.val[1]), vreinterpret_s32_s16(r4567_low.val[1]));
                        int32x2x2_t final_89AB = vtrn_s32(vreinterpret_s32_s16(r0123_high.val[0]), vreinterpret_s32_s16(r4567_high.val[0]));
                        int32x2x2_t final_CDEF = vtrn_s32(vreinterpret_s32_s16(r0123_high.val[1]), vreinterpret_s32_s16(r4567_high.val[1]));
                        
                        int8x8_t transposed[8] = {
                            vreinterpret_s8_s32(final_0123.val[0]),
                            vreinterpret_s8_s32(final_4567.val[0]),
                            vreinterpret_s8_s32(final_0123.val[1]),
                            vreinterpret_s8_s32(final_4567.val[1]),
                            vreinterpret_s8_s32(final_89AB.val[0]),
                            vreinterpret_s8_s32(final_CDEF.val[0]),
                            vreinterpret_s8_s32(final_89AB.val[1]),
                            vreinterpret_s8_s32(final_CDEF.val[1])
                        };
                        
                        for (int col = 0; col < 8 && col_block + col < col_block_end; col++) {
                            if (col_block + col < num_cols) {
                                vst1_s8(&destination[(col_block + col) * num_rows + row_block], transposed[col]);
                            }
                        }
                    } else {
                        for (size_t row = row_block; row < row_block_end; row++) {
                            for (size_t col = col_block; col < col_block_end; col++) {
                                destination[col * num_rows + row] = source[row * num_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }
}

void cactus_transpose_2d_f32(const float* source, float* destination, size_t num_rows, size_t num_cols, size_t start_row, size_t end_row) {
    constexpr size_t TILE_SIZE = 32;
    constexpr size_t VECTOR_WIDTH = 4;

    for (size_t row_tile_start = start_row; row_tile_start < end_row; row_tile_start += TILE_SIZE) {
        const size_t row_tile_end = std::min(row_tile_start + TILE_SIZE, end_row);
        
        for (size_t col_tile_start = 0; col_tile_start < num_cols; col_tile_start += TILE_SIZE) {
            const size_t col_tile_end = std::min(col_tile_start + TILE_SIZE, num_cols);
            
            for (size_t row_block = row_tile_start; row_block < row_tile_end; row_block += VECTOR_WIDTH) {
                const size_t row_block_end = std::min(row_block + VECTOR_WIDTH, row_tile_end);
                
                for (size_t col_block = col_tile_start; col_block < col_tile_end; col_block += VECTOR_WIDTH) {
                    const size_t col_block_end = std::min(col_block + VECTOR_WIDTH, col_tile_end);
                    
                    if (row_block_end - row_block >= 4 && col_block_end - col_block >= 4) {
                        float32x4_t rows[4];
                        for (int i = 0; i < 4; i++) {
                            if (row_block + i < row_block_end) {
                                rows[i] = vld1q_f32(&source[(row_block + i) * num_cols + col_block]);
                            } else {
                                rows[i] = vdupq_n_f32(0.0f);
                            }
                        }
                        
                        float32x4x2_t r01 = vtrnq_f32(rows[0], rows[1]);
                        float32x4x2_t r23 = vtrnq_f32(rows[2], rows[3]);
                        
                        float32x4_t col0 = vcombine_f32(vget_low_f32(r01.val[0]), vget_low_f32(r23.val[0]));
                        float32x4_t col1 = vcombine_f32(vget_low_f32(r01.val[1]), vget_low_f32(r23.val[1]));
                        float32x4_t col2 = vcombine_f32(vget_high_f32(r01.val[0]), vget_high_f32(r23.val[0]));
                        float32x4_t col3 = vcombine_f32(vget_high_f32(r01.val[1]), vget_high_f32(r23.val[1]));
                        
                        if (col_block + 0 < num_cols) {
                            if (row_block_end - row_block >= 4) {
                                vst1q_f32(&destination[(col_block + 0) * num_rows + row_block], col0);
                            } else {
                                float temp[4];
                                vst1q_f32(temp, col0);
                                for (size_t i = 0; i < row_block_end - row_block; ++i) {
                                    destination[(col_block + 0) * num_rows + row_block + i] = temp[i];
                                }
                            }
                        }
                        if (col_block + 1 < num_cols) {
                            if (row_block_end - row_block >= 4) {
                                vst1q_f32(&destination[(col_block + 1) * num_rows + row_block], col1);
                            } else {
                                float temp[4];
                                vst1q_f32(temp, col1);
                                for (size_t i = 0; i < row_block_end - row_block; ++i) {
                                    destination[(col_block + 1) * num_rows + row_block + i] = temp[i];
                                }
                            }
                        }
                        if (col_block + 2 < num_cols) {
                            if (row_block_end - row_block >= 4) {
                                vst1q_f32(&destination[(col_block + 2) * num_rows + row_block], col2);
                            } else {
                                float temp[4];
                                vst1q_f32(temp, col2);
                                for (size_t i = 0; i < row_block_end - row_block; ++i) {
                                    destination[(col_block + 2) * num_rows + row_block + i] = temp[i];
                                }
                            }
                        }
                        if (col_block + 3 < num_cols) {
                            if (row_block_end - row_block >= 4) {
                                vst1q_f32(&destination[(col_block + 3) * num_rows + row_block], col3);
                            } else {
                                float temp[4];
                                vst1q_f32(temp, col3);
                                for (size_t i = 0; i < row_block_end - row_block; ++i) {
                                    destination[(col_block + 3) * num_rows + row_block + i] = temp[i];
                                }
                            }
                        }
                    } else {
                        for (size_t row = row_block; row < row_block_end; row++) {
                            for (size_t col = col_block; col < col_block_end; col++) {
                                destination[col * num_rows + row] = source[row * num_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }
}


void cactus_transpose_int8(const int8_t* source, int8_t* destination, const size_t* shape, const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx) {
    if (ndim == 2 && permutation[0] == 1 && permutation[1] == 0) {
        size_t num_rows = shape[0];
        size_t num_cols = shape[1];
        
        constexpr size_t THRESHOLD = 8192;
        constexpr size_t TILE_ROWS = 64;
        if (num_rows * num_cols >= THRESHOLD) {
            const size_t num_row_blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;
            
            CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [=](size_t start_block, size_t end_block) {
                    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                        size_t start_row = block_idx * TILE_ROWS;
                        size_t end_row = std::min(start_row + TILE_ROWS, num_rows);
                        
                        cactus_transpose_2d_int8(source, destination, num_rows, num_cols, start_row, end_row);
                    }
                });
        } else {
            cactus_transpose_2d_int8(source, destination, num_rows, num_cols, 0, num_rows);
        }
    } else {
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            size_t src_idx = 0;
            size_t tmp_idx = idx;
            
            for (size_t i = 0; i < ndim; ++i) {
                size_t coord = tmp_idx % shape[permutation[ndim - 1 - i]];
                tmp_idx /= shape[permutation[ndim - 1 - i]];
                
                size_t stride = 1;
                for (size_t j = permutation[ndim - 1 - i] + 1; j < ndim; ++j) {
                    stride *= shape[j];
                }
                src_idx += coord * stride;
            }
            
            destination[idx] = source[src_idx];
        }
    }
}