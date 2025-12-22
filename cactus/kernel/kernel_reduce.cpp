#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


int64_t cactus_sum_all_int8(const int8_t* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> int64_t {
            constexpr size_t VECTOR_WIDTH = 16;
            constexpr size_t TILE_SIZE = VECTOR_WIDTH * 4;
            const size_t tile_aligned = ((end_idx - start_idx) / TILE_SIZE) * TILE_SIZE + start_idx;
            
            int32x4_t sum_vec[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
            
            for (size_t i = start_idx; i < tile_aligned; i += TILE_SIZE) {
                int8x16_t input_vec[4];
                input_vec[0] = vld1q_s8(&data[i]);
                input_vec[1] = vld1q_s8(&data[i + VECTOR_WIDTH]);
                input_vec[2] = vld1q_s8(&data[i + VECTOR_WIDTH * 2]);
                input_vec[3] = vld1q_s8(&data[i + VECTOR_WIDTH * 3]);
                
                for (int j = 0; j < 4; ++j) {
                    int16x8_t low = vmovl_s8(vget_low_s8(input_vec[j]));
                    int16x8_t high = vmovl_s8(vget_high_s8(input_vec[j]));
                    
                    sum_vec[j] = vaddq_s32(sum_vec[j], vmovl_s16(vget_low_s16(low)));
                    sum_vec[j] = vaddq_s32(sum_vec[j], vmovl_s16(vget_high_s16(low)));
                    sum_vec[j] = vaddq_s32(sum_vec[j], vmovl_s16(vget_low_s16(high)));
                    sum_vec[j] = vaddq_s32(sum_vec[j], vmovl_s16(vget_high_s16(high)));
                }
            }
            
            const size_t vectorized_end = ((end_idx - start_idx) / VECTOR_WIDTH) * VECTOR_WIDTH + start_idx;
            for (size_t i = tile_aligned; i < vectorized_end; i += VECTOR_WIDTH) {
                int8x16_t input_vec = vld1q_s8(&data[i]);
                
                int16x8_t low = vmovl_s8(vget_low_s8(input_vec));
                int16x8_t high = vmovl_s8(vget_high_s8(input_vec));
                
                sum_vec[0] = vaddq_s32(sum_vec[0], vmovl_s16(vget_low_s16(low)));
                sum_vec[0] = vaddq_s32(sum_vec[0], vmovl_s16(vget_high_s16(low)));
                sum_vec[0] = vaddq_s32(sum_vec[0], vmovl_s16(vget_low_s16(high)));
                sum_vec[0] = vaddq_s32(sum_vec[0], vmovl_s16(vget_high_s16(high)));
            }
            
            int64_t thread_sum = 0;
            for (int j = 0; j < 4; ++j) {
                thread_sum += vaddvq_s32(sum_vec[j]);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<int64_t>(data[i]);
            }
            
            return thread_sum;
        },
        0LL,
        [](int64_t a, int64_t b) { return a + b; }
    );
}

void cactus_sum_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    if (inner_size > 1) {
        CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
            [&](size_t outer, size_t inner) {
                int32x4_t sum_vec = vdupq_n_s32(0);
                int32_t scalar_sum = 0;
                
                constexpr size_t CACHE_BLOCK_SIZE = 128;
                for (size_t block_start = 0; block_start < axis_size; block_start += CACHE_BLOCK_SIZE) {
                    size_t block_end = std::min(block_start + CACHE_BLOCK_SIZE, axis_size);
                    const size_t vectorized_axis = ((block_end - block_start) / 4) * 4 + block_start;
                    
                    for (size_t a = block_start; a < vectorized_axis; a += 4) {
                        int32_t values[4];
                        for (int j = 0; j < 4; j++) {
                            size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                            values[j] = static_cast<int32_t>(input[idx]);
                        }
                        int32x4_t input_vec = vld1q_s32(values);
                        sum_vec = vaddq_s32(sum_vec, input_vec);
                    }
                    
                    for (size_t a = vectorized_axis; a < block_end; a++) {
                        size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                        scalar_sum += static_cast<int32_t>(input[idx]);
                    }
                }
                
                int32_t total_sum = vaddvq_s32(sum_vec) + scalar_sum;
                total_sum = std::min(127, std::max(-128, total_sum));
                size_t output_idx = outer * inner_size + inner;
                output[output_idx] = static_cast<int8_t>(total_sum);
            });
    } else {
        CactusThreading::parallel_for(outer_size, CactusThreading::Thresholds::AXIS_REDUCE,
            [&](size_t start_outer, size_t end_outer) {
                for (size_t outer = start_outer; outer < end_outer; outer++) {
            int32x4_t sum_vec = vdupq_n_s32(0);
            size_t processed = 0;
            
                const size_t vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                    size_t base_idx = outer * axis_size + a;
                        int8x16_t input_vec = vld1q_s8(&input[base_idx]);
                    
                    int16x8_t low = vmovl_s8(vget_low_s8(input_vec));
                    int16x8_t high = vmovl_s8(vget_high_s8(input_vec));
                    
                    int32x4_t low_low = vmovl_s16(vget_low_s16(low));
                    int32x4_t low_high = vmovl_s16(vget_high_s16(low));
                    int32x4_t high_low = vmovl_s16(vget_low_s16(high));
                    int32x4_t high_high = vmovl_s16(vget_high_s16(high));
                    
                    sum_vec = vaddq_s32(sum_vec, low_low);
                    sum_vec = vaddq_s32(sum_vec, low_high);
                    sum_vec = vaddq_s32(sum_vec, high_low);
                    sum_vec = vaddq_s32(sum_vec, high_high);
                    
                    processed = a + NEON_VECTOR_SIZE;
            }
            
            int32_t total_sum = vaddvq_s32(sum_vec);
            
            for (size_t a = processed; a < axis_size; a++) {
                        size_t idx = outer * axis_size + a;
                        total_sum += static_cast<int32_t>(input[idx]);
            }
            
            total_sum = std::min(127, std::max(-128, total_sum));
                    output[outer] = static_cast<int8_t>(total_sum);
                }
        });
    }
}

double cactus_mean_all_int8(const int8_t* data, size_t num_elements) {
    int64_t sum = cactus_sum_all_int8(data, num_elements);
    return static_cast<double>(sum) / static_cast<double>(num_elements);
}

void cactus_mean_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    cactus_sum_axis_int8(input, output, outer_size, axis_size, inner_size);
    
    size_t result_size = outer_size * inner_size;
    for (size_t i = 0; i < result_size; i++) {
        double mean_val = static_cast<double>(output[i]) / static_cast<double>(axis_size);
        int8_t clamped_val = static_cast<int8_t>(std::round(mean_val));
        if (mean_val > 127) clamped_val = 127;
        if (mean_val < -128) clamped_val = -128;
        output[i] = clamped_val;
    }
}

double cactus_variance_all_int8(const int8_t* data, size_t num_elements) {
    double mean = cactus_mean_all_int8(data, num_elements);
    const size_t vectorized_elements = (num_elements / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
    
    float32x4_t sum_squared_diff_vec = vdupq_n_f32(0.0f);
    float32x4_t mean_vec = vdupq_n_f32(static_cast<float>(mean));
    
    for (size_t i = 0; i < vectorized_elements; i += NEON_VECTOR_SIZE) {
        int8x16_t input_vec = vld1q_s8(&data[i]);
        
        int16x8_t low = vmovl_s8(vget_low_s8(input_vec));
        int16x8_t high = vmovl_s8(vget_high_s8(input_vec));
        
        float32x4_t low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low)));
        float32x4_t low_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low)));
        float32x4_t high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high)));
        float32x4_t high_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high)));
        
        float32x4_t diff_low_low = vsubq_f32(low_low, mean_vec);
        float32x4_t diff_low_high = vsubq_f32(low_high, mean_vec);
        float32x4_t diff_high_low = vsubq_f32(high_low, mean_vec);
        float32x4_t diff_high_high = vsubq_f32(high_high, mean_vec);
        
        sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_low_low, diff_low_low);
        sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_low_high, diff_low_high);
        sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_high_low, diff_high_low);
        sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_high_high, diff_high_high);
    }
    
    double sum_squared_diff = vaddvq_f32(sum_squared_diff_vec);
    
    for (size_t i = vectorized_elements; i < num_elements; ++i) {
        double diff = static_cast<double>(data[i]) - mean;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / static_cast<double>(num_elements);
}

void cactus_variance_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    std::vector<int8_t> mean_output(outer_size * inner_size);
    cactus_sum_axis_int8(input, mean_output.data(), outer_size, axis_size, inner_size);
    
    for (size_t i = 0; i < outer_size * inner_size; i++) {
        double mean_val = static_cast<double>(mean_output[i]) / static_cast<double>(axis_size);
        mean_output[i] = static_cast<int8_t>(std::round(mean_val));
    }
    
    std::vector<double> sum_squared_diff(outer_size * inner_size, 0.0);
    
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {
            size_t mean_idx = outer * inner_size + inner;
            double mean_val = static_cast<double>(mean_output[mean_idx]);
            
            float32x4_t sum_squared_diff_vec = vdupq_n_f32(0.0f);
            float32x4_t mean_vec = vdupq_n_f32(static_cast<float>(mean_val));
            
            if (inner_size == 1) {
                const size_t vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                    size_t base_idx = outer * axis_size + a;
                    int8x16_t input_vec = vld1q_s8(&input[base_idx]);
                    
                    int16x8_t low = vmovl_s8(vget_low_s8(input_vec));
                    int16x8_t high = vmovl_s8(vget_high_s8(input_vec));
                    
                    float32x4_t low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low)));
                    float32x4_t low_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low)));
                    float32x4_t high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high)));
                    float32x4_t high_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high)));
                    
                    float32x4_t diff_low_low = vsubq_f32(low_low, mean_vec);
                    float32x4_t diff_low_high = vsubq_f32(low_high, mean_vec);
                    float32x4_t diff_high_low = vsubq_f32(high_low, mean_vec);
                    float32x4_t diff_high_high = vsubq_f32(high_high, mean_vec);
                    
                    sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_low_low, diff_low_low);
                    sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_low_high, diff_low_high);
                    sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_high_low, diff_high_low);
                    sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_high_high, diff_high_high);
                }
            } else {
                const size_t vectorized_axis = (axis_size / 4) * 4;
                for (size_t a = 0; a < vectorized_axis; a += 4) {
                    float values[4];
                    for (int j = 0; j < 4; j++) {
                        size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                        values[j] = static_cast<float>(input[idx]);
                    }
                    float32x4_t input_vec = vld1q_f32(values);
                    float32x4_t diff_vec = vsubq_f32(input_vec, mean_vec);
                    sum_squared_diff_vec = vfmaq_f32(sum_squared_diff_vec, diff_vec, diff_vec);
                }
            }
            
            double sum_sq_diff = vaddvq_f32(sum_squared_diff_vec);
            
            size_t vectorized_axis = (inner_size == 1) ? (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE : (axis_size / 4) * 4;
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                double diff = static_cast<double>(input[idx]) - mean_val;
                sum_sq_diff += diff * diff;
            }
            
            sum_squared_diff[mean_idx] = sum_sq_diff;
        }
    }
    
    for (size_t i = 0; i < sum_squared_diff.size(); i++) {
        double variance_val = sum_squared_diff[i] / static_cast<double>(axis_size);
        int8_t clamped_val = static_cast<int8_t>(std::round(variance_val));
        if (variance_val > 127) clamped_val = 127;
        if (variance_val < -128) clamped_val = -128;
        output[i] = clamped_val;
    }
}

int64_t cactus_min_all_int8(const int8_t* data, size_t num_elements) {
    if (num_elements == 0) return 0;
    
    constexpr size_t VECTOR_WIDTH = 16;
    constexpr size_t TILE_SIZE = VECTOR_WIDTH * 4;
    const size_t tile_aligned = (num_elements / TILE_SIZE) * TILE_SIZE;
    
    int8x16_t min_vec[4] = {
        vdupq_n_s8(std::numeric_limits<int8_t>::max()),
        vdupq_n_s8(std::numeric_limits<int8_t>::max()),
        vdupq_n_s8(std::numeric_limits<int8_t>::max()),
        vdupq_n_s8(std::numeric_limits<int8_t>::max())
    };
    
    for (size_t i = 0; i < tile_aligned; i += TILE_SIZE) {
        int8x16_t input_vec[4];
        input_vec[0] = vld1q_s8(&data[i]);
        input_vec[1] = vld1q_s8(&data[i + VECTOR_WIDTH]);
        input_vec[2] = vld1q_s8(&data[i + VECTOR_WIDTH * 2]);
        input_vec[3] = vld1q_s8(&data[i + VECTOR_WIDTH * 3]);
        
        min_vec[0] = vminq_s8(min_vec[0], input_vec[0]);
        min_vec[1] = vminq_s8(min_vec[1], input_vec[1]);
        min_vec[2] = vminq_s8(min_vec[2], input_vec[2]);
        min_vec[3] = vminq_s8(min_vec[3], input_vec[3]);
    }
    
    const size_t vectorized_elements = (num_elements / VECTOR_WIDTH) * VECTOR_WIDTH;
    for (size_t i = tile_aligned; i < vectorized_elements; i += VECTOR_WIDTH) {
        int8x16_t input_vec = vld1q_s8(&data[i]);
        min_vec[0] = vminq_s8(min_vec[0], input_vec);
    }
    
    int8x16_t final_min = vminq_s8(vminq_s8(min_vec[0], min_vec[1]), vminq_s8(min_vec[2], min_vec[3]));
    int8_t min_val = vminvq_s8(final_min);
    
    for (size_t i = vectorized_elements; i < num_elements; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    
    return static_cast<int64_t>(min_val);
}

void cactus_min_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
                int8x16_t min_vec = vdupq_n_s8(std::numeric_limits<int8_t>::max());
                size_t vectorized_axis;
                
                if (inner_size == 1) {
                    vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                    for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                        size_t base_idx = outer * axis_size + a;
                    int8x16_t input_vec = vld1q_s8(&input[base_idx]);
                        min_vec = vminq_s8(min_vec, input_vec);
                    }
                } else {
                    vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                    for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                        int8_t values[NEON_VECTOR_SIZE];
                        for (size_t j = 0; j < NEON_VECTOR_SIZE; j++) {
                            size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                        values[j] = input[idx];
                        }
                        int8x16_t input_vec = vld1q_s8(values);
                        min_vec = vminq_s8(min_vec, input_vec);
                    }
                }
                
                int8_t min_val = vminvq_s8(min_vec);
                
                for (size_t a = vectorized_axis; a < axis_size; a++) {
                    size_t idx;
                    if (inner_size == 1) {
                        idx = outer * axis_size + a;
                    } else {
                        idx = outer * axis_size * inner_size + a * inner_size + inner;
                    }
                if (input[idx] < min_val) {
                    min_val = input[idx];
                    }
                }
                
                size_t output_idx = outer * inner_size + inner;
            output[output_idx] = min_val;
            });
}

int64_t cactus_max_all_int8(const int8_t* data, size_t num_elements) {
    if (num_elements == 0) return 0;
    
    constexpr size_t VECTOR_WIDTH = 16;
    constexpr size_t TILE_SIZE = VECTOR_WIDTH * 4;
    const size_t tile_aligned = (num_elements / TILE_SIZE) * TILE_SIZE;
    
    int8x16_t max_vec[4] = {
        vdupq_n_s8(std::numeric_limits<int8_t>::min()),
        vdupq_n_s8(std::numeric_limits<int8_t>::min()),
        vdupq_n_s8(std::numeric_limits<int8_t>::min()),
        vdupq_n_s8(std::numeric_limits<int8_t>::min())
    };
    
    for (size_t i = 0; i < tile_aligned; i += TILE_SIZE) {
        int8x16_t input_vec[4];
        input_vec[0] = vld1q_s8(&data[i]);
        input_vec[1] = vld1q_s8(&data[i + VECTOR_WIDTH]);
        input_vec[2] = vld1q_s8(&data[i + VECTOR_WIDTH * 2]);
        input_vec[3] = vld1q_s8(&data[i + VECTOR_WIDTH * 3]);
        
        max_vec[0] = vmaxq_s8(max_vec[0], input_vec[0]);
        max_vec[1] = vmaxq_s8(max_vec[1], input_vec[1]);
        max_vec[2] = vmaxq_s8(max_vec[2], input_vec[2]);
        max_vec[3] = vmaxq_s8(max_vec[3], input_vec[3]);
    }
    
    const size_t vectorized_elements = (num_elements / VECTOR_WIDTH) * VECTOR_WIDTH;
    for (size_t i = tile_aligned; i < vectorized_elements; i += VECTOR_WIDTH) {
        int8x16_t input_vec = vld1q_s8(&data[i]);
        max_vec[0] = vmaxq_s8(max_vec[0], input_vec);
    }
    
    int8x16_t final_max = vmaxq_s8(vmaxq_s8(max_vec[0], max_vec[1]), vmaxq_s8(max_vec[2], max_vec[3]));
    int8_t max_val = vmaxvq_s8(final_max);
    
    for (size_t i = vectorized_elements; i < num_elements; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    return static_cast<int64_t>(max_val);
}

void cactus_max_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            int8x16_t max_vec = vdupq_n_s8(std::numeric_limits<int8_t>::min());
            size_t vectorized_axis;
            
            if (inner_size == 1) {
                vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                    size_t base_idx = outer * axis_size + a;
                    int8x16_t input_vec = vld1q_s8(&input[base_idx]);
                    max_vec = vmaxq_s8(max_vec, input_vec);
                }
            } else {
                vectorized_axis = (axis_size / NEON_VECTOR_SIZE) * NEON_VECTOR_SIZE;
                for (size_t a = 0; a < vectorized_axis; a += NEON_VECTOR_SIZE) {
                    int8_t values[NEON_VECTOR_SIZE];
                    for (size_t j = 0; j < NEON_VECTOR_SIZE; j++) {
                        size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                        values[j] = input[idx];
                    }
                    int8x16_t input_vec = vld1q_s8(values);
                    max_vec = vmaxq_s8(max_vec, input_vec);
                }
            }
            
            int8_t max_val = vmaxvq_s8(max_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx;
                if (inner_size == 1) {
                    idx = outer * axis_size + a;
                } else {
                    idx = outer * axis_size * inner_size + a * inner_size + inner;
                }
                if (input[idx] > max_val) {
                    max_val = input[idx];
                }
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = max_val;
        });
}

double cactus_sum_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                sum_vec = vaddq_f32(sum_vec, input_vec);
            }
            
            double thread_sum = static_cast<double>(vaddvq_f32(sum_vec));
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<double>(data[i]);
            }
            
            return thread_sum;
        },
        0.0,
        [](double a, double b) { return a + b; }
    );
}

void cactus_sum_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                sum_vec = vaddq_f32(sum_vec, input_vec);
            }
            
            float total_sum = vaddvq_f32(sum_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum += input[idx];
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = total_sum;
        });
}

double cactus_sum_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 8;  
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float16x8_t sum_vec = vdupq_n_f16(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                sum_vec = vaddq_f16(sum_vec, input_vec);
            }
            
            double thread_sum = 0.0;
            __fp16 sum_array[8];
            vst1q_f16(sum_array, sum_vec);
            for (int j = 0; j < 8; j++) {
                thread_sum += static_cast<double>(sum_array[j]);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<double>(data[i]);
            }
            
            return thread_sum;
        },
        0.0,
        [](double a, double b) { return a + b; }
    );
}

double cactus_mean_all_f16(const __fp16* data, size_t num_elements) {
    double sum = cactus_sum_all_f16(data, num_elements);
    return sum / static_cast<double>(num_elements);
}

void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t sum_vec = vdupq_n_f16(0.0f);
            
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                sum_vec = vaddq_f16(sum_vec, input_vec);
            }
            
            __fp16 total_sum = 0.0f;
            __fp16 sum_array[8];
            vst1q_f16(sum_array, sum_vec);
            for (int j = 0; j < 8; j++) {
                total_sum += sum_array[j];
            }
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum += input[idx];
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = total_sum / static_cast<__fp16>(axis_size);
        });
}

double cactus_mean_all_f32(const float* data, size_t num_elements) {
    double sum = cactus_sum_all_f32(data, num_elements);
    return sum / static_cast<double>(num_elements);
}

void cactus_mean_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    cactus_sum_axis_f32(input, output, outer_size, axis_size, inner_size);
    const float divisor = static_cast<float>(axis_size);
    
    CactusThreading::parallel_for(outer_size * inner_size, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t divisor_vec = vdupq_n_f32(divisor);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t output_vec = vld1q_f32(&output[i]);
                output_vec = vdivq_f32(output_vec, divisor_vec);
                vst1q_f32(&output[i], output_vec);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] /= divisor;
            }
        });
}

double cactus_variance_all_f32(const float* data, size_t num_elements) {
    double mean = cactus_mean_all_f32(data, num_elements);
    
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t mean_vec = vdupq_n_f32(static_cast<float>(mean));
            float32x4_t var_vec = vdupq_n_f32(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                float32x4_t diff = vsubq_f32(input_vec, mean_vec);
                var_vec = vmlaq_f32(var_vec, diff, diff);
            }
            
            double thread_var = static_cast<double>(vaddvq_f32(var_vec));
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                double diff = static_cast<double>(data[i]) - mean;
                thread_var += diff * diff;
            }
            
            return thread_var;
        },
        0.0,
        [](double a, double b) { return a + b; }
    ) / static_cast<double>(num_elements);
}

void cactus_variance_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    std::vector<float> means(outer_size * inner_size);
    cactus_mean_axis_f32(input, means.data(), outer_size, axis_size, inner_size);
    
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            size_t output_idx = outer * inner_size + inner;
            float mean_val = means[output_idx];
            
            float32x4_t mean_vec = vdupq_n_f32(mean_val);
            float32x4_t var_vec = vdupq_n_f32(0.0f);
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                float32x4_t diff = vsubq_f32(input_vec, mean_vec);
                var_vec = vmlaq_f32(var_vec, diff, diff);
            }
            
            float total_var = vaddvq_f32(var_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                float diff = input[idx] - mean_val;
                total_var += diff * diff;
            }
            
            output[output_idx] = total_var / static_cast<float>(axis_size);
        });
}

float cactus_min_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> float {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                min_vec = vminq_f32(min_vec, input_vec);
            }
            
            float thread_min = vminvq_f32(min_vec);
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_min = std::min(thread_min, data[i]);
            }
            
            return thread_min;
        },
        std::numeric_limits<float>::max(),
        [](float a, float b) { return std::min(a, b); }
    );
}

void cactus_min_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                min_vec = vminq_f32(min_vec, input_vec);
            }
            
            float min_val = vminvq_f32(min_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                min_val = std::min(min_val, input[idx]);
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = min_val;
        });
}

float cactus_max_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> float {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                max_vec = vmaxq_f32(max_vec, input_vec);
            }
            
            float thread_max = vmaxvq_f32(max_vec);
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_max = std::max(thread_max, data[i]);
            }
            
            return thread_max;
        },
        std::numeric_limits<float>::lowest(),
        [](float a, float b) { return std::max(a, b); }
    );
}

void cactus_max_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                max_vec = vmaxq_f32(max_vec, input_vec);
            }
            
            float max_val = vmaxvq_f32(max_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                max_val = std::max(max_val, input[idx]);
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = max_val;
        });
} 