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

enum class BroadcastOp { ADD, SUB, MUL, DIV };

template<BroadcastOp Op>
static inline float16x8_t broadcast_op_vec(float16x8_t a, float16x8_t b) {
    if constexpr (Op == BroadcastOp::ADD) return vaddq_f16(a, b);
    else if constexpr (Op == BroadcastOp::SUB) return vsubq_f16(a, b);
    else if constexpr (Op == BroadcastOp::MUL) return vmulq_f16(a, b);
    else return vdivq_f16(a, b);
}

template<BroadcastOp Op>
static inline __fp16 broadcast_op_scalar(__fp16 a, __fp16 b) {
    if constexpr (Op == BroadcastOp::ADD) return a + b;
    else if constexpr (Op == BroadcastOp::SUB) return a - b;
    else if constexpr (Op == BroadcastOp::MUL) return a * b;
    else return a / b;
}

template<BroadcastOp Op>
static void broadcast_op_optimized(const __fp16* a, const __fp16* b, __fp16* output,
                                    const size_t* a_strides, const size_t* b_strides,
                                    const size_t* output_shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= output_shape[i];
    }

    if (total_elements == 0) return;

    size_t inner_size = output_shape[ndim - 1];
    bool a_inner_contiguous = (a_strides[ndim - 1] == 1) || (a_strides[ndim - 1] == 0);
    bool b_inner_contiguous = (b_strides[ndim - 1] == 1) || (b_strides[ndim - 1] == 0);
    bool a_inner_broadcast = (a_strides[ndim - 1] == 0);
    bool b_inner_broadcast = (b_strides[ndim - 1] == 0);

    size_t outer_size = total_elements / inner_size;

    if (a_inner_contiguous && b_inner_contiguous && inner_size >= 8) {
        CactusThreading::parallel_for(outer_size, CactusThreading::Thresholds::ELEMENT_WISE,
            [&](size_t start_outer, size_t end_outer) {
                std::vector<size_t> coords(ndim, 0);

                size_t tmp = start_outer;
                for (int i = ndim - 2; i >= 0; --i) {
                    coords[i] = tmp % output_shape[i];
                    tmp /= output_shape[i];
                }

                for (size_t outer_idx = start_outer; outer_idx < end_outer; ++outer_idx) {
                    size_t a_base = 0, b_base = 0;
                    for (size_t i = 0; i < ndim - 1; ++i) {
                        a_base += coords[i] * a_strides[i];
                        b_base += coords[i] * b_strides[i];
                    }

                    __fp16* out_ptr = output + outer_idx * inner_size;
                    const size_t vec_end = (inner_size / 8) * 8;

                    const bool use_stream = total_elements >= STREAMING_STORE_THRESHOLD;

                    if (a_inner_broadcast && b_inner_broadcast) {
                        __fp16 result = broadcast_op_scalar<Op>(a[a_base], b[b_base]);
                        float16x8_t result_vec = vdupq_n_f16(result);
                        if (use_stream) {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                stream_store_f16x8(out_ptr + i, result_vec);
                            }
                        } else {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                vst1q_f16(out_ptr + i, result_vec);
                            }
                        }
                        for (size_t i = vec_end; i < inner_size; ++i) {
                            out_ptr[i] = result;
                        }
                    } else if (a_inner_broadcast) {
                        float16x8_t a_vec = vdupq_n_f16(a[a_base]);
                        const __fp16* b_ptr = b + b_base;
                        if (use_stream) {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t b_vec = vld1q_f16(b_ptr + i);
                                stream_store_f16x8(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        } else {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t b_vec = vld1q_f16(b_ptr + i);
                                vst1q_f16(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        }
                        for (size_t i = vec_end; i < inner_size; ++i) {
                            out_ptr[i] = broadcast_op_scalar<Op>(a[a_base], b_ptr[i]);
                        }
                    } else if (b_inner_broadcast) {
                        const __fp16* a_ptr = a + a_base;
                        float16x8_t b_vec = vdupq_n_f16(b[b_base]);
                        if (use_stream) {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t a_vec = vld1q_f16(a_ptr + i);
                                stream_store_f16x8(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        } else {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t a_vec = vld1q_f16(a_ptr + i);
                                vst1q_f16(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        }
                        for (size_t i = vec_end; i < inner_size; ++i) {
                            out_ptr[i] = broadcast_op_scalar<Op>(a_ptr[i], b[b_base]);
                        }
                    } else {
                        const __fp16* a_ptr = a + a_base;
                        const __fp16* b_ptr = b + b_base;
                        if (use_stream) {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t a_vec = vld1q_f16(a_ptr + i);
                                float16x8_t b_vec = vld1q_f16(b_ptr + i);
                                stream_store_f16x8(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        } else {
                            for (size_t i = 0; i < vec_end; i += 8) {
                                float16x8_t a_vec = vld1q_f16(a_ptr + i);
                                float16x8_t b_vec = vld1q_f16(b_ptr + i);
                                vst1q_f16(out_ptr + i, broadcast_op_vec<Op>(a_vec, b_vec));
                            }
                        }
                        for (size_t i = vec_end; i < inner_size; ++i) {
                            out_ptr[i] = broadcast_op_scalar<Op>(a_ptr[i], b_ptr[i]);
                        }
                    }

                    for (int i = ndim - 2; i >= 0; --i) {
                        coords[i]++;
                        if (coords[i] < output_shape[i]) break;
                        coords[i] = 0;
                    }
                }
            });
    } else {
        CactusThreading::parallel_for(total_elements, CactusThreading::Thresholds::ELEMENT_WISE,
            [&](size_t start_idx, size_t end_idx) {
                std::vector<size_t> coords(ndim);

                size_t tmp = start_idx;
                for (int i = ndim - 1; i >= 0; --i) {
                    coords[i] = tmp % output_shape[i];
                    tmp /= output_shape[i];
                }

                for (size_t linear_idx = start_idx; linear_idx < end_idx; ++linear_idx) {
                    size_t a_idx = compute_linear_index(coords.data(), a_strides, ndim);
                    size_t b_idx = compute_linear_index(coords.data(), b_strides, ndim);

                    output[linear_idx] = broadcast_op_scalar<Op>(a[a_idx], b[b_idx]);

                    increment_coords(coords.data(), output_shape, ndim);
                }
            });
    }
}

void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            constexpr size_t UNROLL = 4;
            const size_t unrolled_end = start_idx + ((end_idx - start_idx) / (SIMD_WIDTH * UNROLL)) * (SIMD_WIDTH * UNROLL);
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            if (use_streaming) {
                for (size_t i = start_idx; i < unrolled_end; i += SIMD_WIDTH * UNROLL) {
                    __builtin_prefetch(&a[i + 256], 0, 0);
                    __builtin_prefetch(&b[i + 256], 0, 0);
                    float16x8_t v0 = vaddq_f16(vld1q_f16(&a[i]), vld1q_f16(&b[i]));
                    float16x8_t v1 = vaddq_f16(vld1q_f16(&a[i + 8]), vld1q_f16(&b[i + 8]));
                    float16x8_t v2 = vaddq_f16(vld1q_f16(&a[i + 16]), vld1q_f16(&b[i + 16]));
                    float16x8_t v3 = vaddq_f16(vld1q_f16(&a[i + 24]), vld1q_f16(&b[i + 24]));
                    stream_store_f16x8(&output[i], v0);
                    stream_store_f16x8(&output[i + 8], v1);
                    stream_store_f16x8(&output[i + 16], v2);
                    stream_store_f16x8(&output[i + 24], v3);
                }
                for (size_t i = unrolled_end; i < vectorized_end; i += SIMD_WIDTH) {
                    stream_store_f16x8(&output[i], vaddq_f16(vld1q_f16(&a[i]), vld1q_f16(&b[i])));
                }
            } else {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    vst1q_f16(&output[i], vaddq_f16(vld1q_f16(&a[i]), vld1q_f16(&b[i])));
                }
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] + b[i];
            }
        });
}

void cactus_add_f16_clipped(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
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
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            if (use_streaming) {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    stream_store_f16x8(&output[i], vsubq_f16(a_vec, b_vec));
                }
            } else {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    vst1q_f16(&output[i], vsubq_f16(a_vec, b_vec));
                }
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] - b[i];
            }
        });
}

void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            if (use_streaming) {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    stream_store_f16x8(&output[i], vmulq_f16(a_vec, b_vec));
                }
            } else {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    vst1q_f16(&output[i], vmulq_f16(a_vec, b_vec));
                }
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] * b[i];
            }
        });
}

void cactus_add_scaled_f16(const __fp16* base, const __fp16* src, __fp16* output, size_t num_elements, float scale) {
    constexpr size_t SIMD_WIDTH = 8;
    const float32x4_t vscale = vdupq_n_f32(scale);
    const size_t vec_end = (num_elements / SIMD_WIDTH) * SIMD_WIDTH;

    for (size_t i = 0; i < vec_end; i += SIMD_WIDTH) {
        float16x8_t base_vec = vld1q_f16(base + i);
        float16x8_t src_vec = vld1q_f16(src + i);

        float32x4_t base_lo = vcvt_f32_f16(vget_low_f16(base_vec));
        float32x4_t base_hi = vcvt_f32_f16(vget_high_f16(base_vec));
        float32x4_t src_lo = vcvt_f32_f16(vget_low_f16(src_vec));
        float32x4_t src_hi = vcvt_f32_f16(vget_high_f16(src_vec));

        float32x4_t result_lo = vfmaq_f32(base_lo, src_lo, vscale);
        float32x4_t result_hi = vfmaq_f32(base_hi, src_hi, vscale);

        vst1q_f16(output + i, vcombine_f16(vcvt_f16_f32(result_lo), vcvt_f16_f32(result_hi)));
    }
    for (size_t i = vec_end; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(base[i])
                                        + static_cast<float>(src[i]) * scale);
    }
}

void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    const bool use_streaming = num_elements >= STREAMING_STORE_THRESHOLD;
    CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            if (use_streaming) {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    stream_store_f16x8(&output[i], vdivq_f16(a_vec, b_vec));
                }
            } else {
                for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                    float16x8_t a_vec = vld1q_f16(&a[i]);
                    float16x8_t b_vec = vld1q_f16(&b[i]);
                    vst1q_f16(&output[i], vdivq_f16(a_vec, b_vec));
                }
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] = a[i] / b[i];
            }
        });
}

void cactus_add_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                              const size_t* a_strides, const size_t* b_strides,
                              const size_t* output_shape, size_t ndim) {
    broadcast_op_optimized<BroadcastOp::ADD>(a, b, output, a_strides, b_strides, output_shape, ndim);
}

void cactus_subtract_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    broadcast_op_optimized<BroadcastOp::SUB>(a, b, output, a_strides, b_strides, output_shape, ndim);
}

void cactus_multiply_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    broadcast_op_optimized<BroadcastOp::MUL>(a, b, output, a_strides, b_strides, output_shape, ndim);
}

void cactus_divide_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim) {
    broadcast_op_optimized<BroadcastOp::DIV>(a, b, output, a_strides, b_strides, output_shape, ndim);
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

void cactus_transpose_2d_f16(const __fp16* source, __fp16* destination, size_t num_rows, size_t num_cols, size_t start_row, size_t end_row) {
    constexpr size_t TILE_SIZE = 32;
    constexpr size_t VECTOR_WIDTH = 8;

    for (size_t row_tile_start = start_row; row_tile_start < end_row; row_tile_start += TILE_SIZE) {
        const size_t row_tile_end = std::min(row_tile_start + TILE_SIZE, end_row);

        for (size_t col_tile_start = 0; col_tile_start < num_cols; col_tile_start += TILE_SIZE) {
            const size_t col_tile_end = std::min(col_tile_start + TILE_SIZE, num_cols);

            for (size_t row_block = row_tile_start; row_block < row_tile_end; row_block += VECTOR_WIDTH) {
                const size_t row_block_end = std::min(row_block + VECTOR_WIDTH, row_tile_end);

                for (size_t col_block = col_tile_start; col_block < col_tile_end; col_block += VECTOR_WIDTH) {
                    const size_t col_block_end = std::min(col_block + VECTOR_WIDTH, col_tile_end);

                    if (row_block_end - row_block >= 8 && col_block_end - col_block >= 8) {
                        float16x8_t rows[8];
                        for (int i = 0; i < 8; i++) {
                            if (row_block + i < row_block_end) {
                                rows[i] = vld1q_f16(&source[(row_block + i) * num_cols + col_block]);
                            } else {
                                rows[i] = vdupq_n_f16(0.0f);
                            }
                        }

                        float16x8x2_t r01 = vtrnq_f16(rows[0], rows[1]);
                        float16x8x2_t r23 = vtrnq_f16(rows[2], rows[3]);
                        float16x8x2_t r45 = vtrnq_f16(rows[4], rows[5]);
                        float16x8x2_t r67 = vtrnq_f16(rows[6], rows[7]);

                        float32x4x2_t r0123_lo = vtrnq_f32(
                            vreinterpretq_f32_f16(r01.val[0]),
                            vreinterpretq_f32_f16(r23.val[0]));
                        float32x4x2_t r0123_hi = vtrnq_f32(
                            vreinterpretq_f32_f16(r01.val[1]),
                            vreinterpretq_f32_f16(r23.val[1]));
                        float32x4x2_t r4567_lo = vtrnq_f32(
                            vreinterpretq_f32_f16(r45.val[0]),
                            vreinterpretq_f32_f16(r67.val[0]));
                        float32x4x2_t r4567_hi = vtrnq_f32(
                            vreinterpretq_f32_f16(r45.val[1]),
                            vreinterpretq_f32_f16(r67.val[1]));

                        float16x8_t col0 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_low_f32(r0123_lo.val[0]), vget_low_f32(r4567_lo.val[0])));
                        float16x8_t col1 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_low_f32(r0123_hi.val[0]), vget_low_f32(r4567_hi.val[0])));
                        float16x8_t col2 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_low_f32(r0123_lo.val[1]), vget_low_f32(r4567_lo.val[1])));
                        float16x8_t col3 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_low_f32(r0123_hi.val[1]), vget_low_f32(r4567_hi.val[1])));
                        float16x8_t col4 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_high_f32(r0123_lo.val[0]), vget_high_f32(r4567_lo.val[0])));
                        float16x8_t col5 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_high_f32(r0123_hi.val[0]), vget_high_f32(r4567_hi.val[0])));
                        float16x8_t col6 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_high_f32(r0123_lo.val[1]), vget_high_f32(r4567_lo.val[1])));
                        float16x8_t col7 = vreinterpretq_f16_f32(vcombine_f32(
                            vget_high_f32(r0123_hi.val[1]), vget_high_f32(r4567_hi.val[1])));

                        float16x8_t cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
                        for (int c = 0; c < 8 && col_block + c < col_block_end; c++) {
                            if (col_block + c < num_cols) {
                                if (row_block_end - row_block >= 8) {
                                    vst1q_f16(&destination[(col_block + c) * num_rows + row_block], cols[c]);
                                } else {
                                    __fp16 temp[8];
                                    vst1q_f16(temp, cols[c]);
                                    for (size_t i = 0; i < row_block_end - row_block; ++i) {
                                        destination[(col_block + c) * num_rows + row_block + i] = temp[i];
                                    }
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

void cactus_transpose_f16(const __fp16* source, __fp16* destination, const size_t* shape, const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx) {
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

                        cactus_transpose_2d_f16(source, destination, num_rows, num_cols, start_row, end_row);
                    }
                });
        } else {
            cactus_transpose_2d_f16(source, destination, num_rows, num_cols, 0, num_rows);
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