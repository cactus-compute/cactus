#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>
#include <arm_neon.h>

enum class ScalarOpType {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    EXP,
    SQRT,
    COS,
    SIN
};


void cactus_add_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements);
void cactus_subtract_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements);
void cactus_multiply_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements);
void cactus_divide_int8(const int8_t* a, const int8_t* b, int8_t* output, size_t num_elements);


void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_add_f16_clipped(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);


void cactus_add_f32(const float* a, const float* b, float* output, size_t num_elements);
void cactus_subtract_f32(const float* a, const float* b, float* output, size_t num_elements);
void cactus_multiply_f32(const float* a, const float* b, float* output, size_t num_elements);
void cactus_divide_f32(const float* a, const float* b, float* output, size_t num_elements);


void cactus_add_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                               const size_t* a_strides, const size_t* b_strides,
                               const size_t* output_shape, size_t ndim);
void cactus_subtract_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                    const size_t* a_strides, const size_t* b_strides,
                                    const size_t* output_shape, size_t ndim);
void cactus_multiply_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                    const size_t* a_strides, const size_t* b_strides,
                                    const size_t* output_shape, size_t ndim);
void cactus_divide_broadcast_int8(const int8_t* a, const int8_t* b, int8_t* output,
                                  const size_t* a_strides, const size_t* b_strides,
                                  const size_t* output_shape, size_t ndim);


void cactus_add_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                               const size_t* a_strides, const size_t* b_strides,
                               const size_t* output_shape, size_t ndim);
void cactus_subtract_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim);
void cactus_multiply_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim);
void cactus_divide_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim);


void cactus_add_broadcast_f32(const float* a, const float* b, float* output,
                               const size_t* a_strides, const size_t* b_strides,
                               const size_t* output_shape, size_t ndim);
void cactus_subtract_broadcast_f32(const float* a, const float* b, float* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim);
void cactus_multiply_broadcast_f32(const float* a, const float* b, float* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim);
void cactus_divide_broadcast_f32(const float* a, const float* b, float* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim);


void cactus_scalar_op_int8(const int8_t* input, int8_t* output, size_t num_elements, float scalar_value, ScalarOpType op_type);
void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type);
void cactus_scalar_op_f32(const float* input, float* output, size_t num_elements, float scalar_value, ScalarOpType op_type);


void cactus_matmul_int8(const int8_t* a, const int8_t* b_transposed, int8_t* c,
                        size_t M, size_t K, size_t N,
                        float a_scale, float b_scale, float c_scale);

#if defined(__ARM_FEATURE_MATMUL_INT8)
void cactus_matmul_int8_to_int32_i8mm(const int8_t* a, const int8_t* b_transposed, int32_t* c,
                                       size_t M, size_t K, size_t N);
#define cactus_matmul_int8_to_int32 cactus_matmul_int8_to_int32_i8mm
#else
void cactus_matmul_int8_to_int32(const int8_t* a, const int8_t* b_transposed, int32_t* c,
                                 size_t M, size_t K, size_t N);
#endif

void cactus_matmul_f16(const __fp16* a, const __fp16* b_transposed, __fp16* c,
                       size_t M, size_t K, size_t N);

void cactus_matmul_f32(const float* a, const float* b_transposed, float* c,
                       size_t M, size_t K, size_t N);


void cactus_transpose_2d_int8(const int8_t* source, int8_t* destination,
                               size_t num_rows, size_t num_cols, size_t start_row, size_t end_row);
void cactus_transpose_2d_f16(const __fp16* source, __fp16* destination,
                             size_t num_rows, size_t num_cols, size_t start_row, size_t end_row);
void cactus_transpose_2d_f32(const float* source, float* destination,
                             size_t num_rows, size_t num_cols, size_t start_row, size_t end_row);

void cactus_transpose_int8(const int8_t* source, int8_t* destination, const size_t* shape,
                           const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx);
void cactus_transpose_f16(const __fp16* source, __fp16* destination, const size_t* shape,
                          const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx);
void cactus_transpose_f32(const float* source, float* destination, const size_t* shape,
                          const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx);

int64_t cactus_sum_all_int8(const int8_t* data, size_t num_elements);
void cactus_sum_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size);
double cactus_sum_all_f16(const __fp16* data, size_t num_elements);
double cactus_sum_all_f32(const float* data, size_t num_elements);
void cactus_sum_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size);

double cactus_mean_all_int8(const int8_t* data, size_t num_elements);
void cactus_mean_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size);
double cactus_mean_all_f16(const __fp16* data, size_t num_elements);
void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);
double cactus_mean_all_f32(const float* data, size_t num_elements);
void cactus_mean_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size);

double cactus_variance_all_int8(const int8_t* data, size_t num_elements);
void cactus_variance_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size);
double cactus_variance_all_f32(const float* data, size_t num_elements);
void cactus_variance_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size);

int64_t cactus_min_all_int8(const int8_t* data, size_t num_elements);
void cactus_min_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size);
float cactus_min_all_f32(const float* data, size_t num_elements);
void cactus_min_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size);

int64_t cactus_max_all_int8(const int8_t* data, size_t num_elements);
void cactus_max_axis_int8(const int8_t* input, int8_t* output, size_t outer_size, size_t axis_size, size_t inner_size);
float cactus_max_all_f32(const float* data, size_t num_elements);
void cactus_max_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size);

void cactus_rms_norm_f16(const __fp16* input, const __fp16* weight, __fp16* output,
                          size_t batch_size, size_t dims, float eps);
                          
void cactus_rms_norm_f32(const float* input, const float* weight, float* output,
                          size_t batch_size, size_t dims, float eps);

void cactus_rms_norm_i8_f32(const int8_t* input, const float* weight, float* output,
                             size_t batch_size, size_t dims, float eps, float input_scale);

void cactus_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                      size_t num_heads, size_t head_dim, size_t start_pos, float theta);

void cactus_rope_f32(const float* input, float* output, size_t batch_size, size_t seq_len,
                      size_t num_heads, size_t head_dim, size_t start_pos, float theta);

void cactus_rope_i8_f32_i8(const int8_t* input, int8_t* output, size_t batch_size, size_t seq_len,
                           size_t num_heads, size_t head_dim, size_t start_pos, float theta,
                           float input_scale, float output_scale);

void cactus_softmax_f16(const __fp16* input, __fp16* output, size_t batch_size, 
                         size_t seq_len, size_t vocab_size);

void cactus_softmax_f32(const float* input, float* output, size_t batch_size, 
                         size_t seq_len, size_t vocab_size);

void cactus_silu_f32(const float* input, float* output, size_t num_elements);
void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_silu_int8(const int8_t* input, int8_t* output, size_t num_elements, 
                      float input_scale, float output_scale);

void cactus_gelu_f32(const float* input, float* output, size_t num_elements);
void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_gelu_int8(const int8_t* input, int8_t* output, size_t num_elements,
                      float input_scale, float output_scale);

void cactus_gelu_f32_erf(const float* input, float* output, size_t num_elements);
void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_gelu_int8_erf(
    const int8_t* input,
    int8_t* output,
    size_t num_elements,
    float scale_in,
    float scale_out);

                      
void cactus_attention_int8(const int8_t* queries, const int8_t* keys, const int8_t* values, int8_t* output,
                            size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                            size_t head_dim, float scale, const int8_t* mask,
                            float q_scale, float k_scale, float v_scale, float output_scale, size_t position_offset = 0, size_t window_size = 0,
                            bool is_causal = true);

void cactus_attention_f16(const __fp16* queries, const __fp16* keys, const __fp16* values, __fp16* output,
                          size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                          size_t head_dim, float scale, const __fp16* mask, size_t position_offset = 0, size_t window_size = 0,
                          bool is_causal = true);

void cactus_attention_f32(const float* queries, const float* keys, const float* values, float* output,
                          size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                          size_t head_dim, float scale, const float* mask, size_t position_offset = 0, size_t window_size = 0,
                          bool is_causal = true);


void cactus_conv1d_causal_depthwise_f32(
    const float* input,
    const float* weight,
    float* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation);

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation);

void cactus_conv1d_causal_depthwise_int8(
    const int8_t* input,
    const int8_t* weight,
    int8_t* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation,
    float input_scale,
    float weight_scale,
    float output_scale);

void cactus_conv1d_f32_k3(
    const float* input,
    const float* weight,
    float* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t stride
);

void cactus_conv1d_f16_k3(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t stride
);

void cactus_conv1d_f32_k3(
    const float* input,
    const float* weight,
    float* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t stride
);

void cactus_conv1d_f16_k3(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t stride
);

void cactus_bilinear_interpolation_fp32(const float* input, float* output, size_t src_height, size_t src_width, size_t embed_dim,
                                        size_t dst_height, size_t dst_width);

void cactus_sample_f32(const float* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);
void cactus_sample_f16(const __fp16* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);


void cactus_concat_f32(const float* input1, const float* input2, float* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis);
void cactus_concat_f16(const __fp16* input1, const __fp16* input2, __fp16* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis);
void cactus_concat_int8(const int8_t* input1, const int8_t* input2, int8_t* output,
                        const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                        size_t ndims, int axis);

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);
void cactus_dynamic_quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count, float* computed_scale);
void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale = 1.0f);
float cactus_fp16_max_abs(const __fp16* src, size_t count);
void cactus_int32_to_fp16_scaled(const int32_t* src, __fp16* dst, size_t count, float scale);

#endif 