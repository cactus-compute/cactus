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

constexpr size_t KV_QUANT_GROUP_SIZE = 32;

void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_add_f16_clipped(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);

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

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type);

void cactus_gemv_int8(const int8_t* A, float A_scale,
                      const int8_t* B, const __fp16* B_scales,
                      __fp16* C, size_t K, size_t N, size_t group_size);

void cactus_gemm_int8(const int8_t* A, const float* A_scales,
                      const int8_t* B, const __fp16* B_scales,
                      __fp16* C, size_t M, size_t K, size_t N, size_t group_size);

void cactus_matmul_int8(const int8_t* A, const float* A_scales,
                        const int8_t* B, const __fp16* B_scales,
                        __fp16* C, size_t M, size_t K, size_t N, size_t group_size);

void cactus_matmul_f16(const __fp16* a, const __fp16* b_transposed, __fp16* c,
                       size_t M, size_t K, size_t N);

void cactus_transpose_2d_f16(const __fp16* source, __fp16* destination,
                             size_t num_rows, size_t num_cols, size_t start_row, size_t end_row);
void cactus_transpose_f16(const __fp16* source, __fp16* destination, const size_t* shape,
                          const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx);

double cactus_sum_all_f16(const __fp16* data, size_t num_elements);
void cactus_sum_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);

double cactus_mean_all_f16(const __fp16* data, size_t num_elements);
void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);

double cactus_variance_all_f16(const __fp16* data, size_t num_elements);
void cactus_variance_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);

__fp16 cactus_min_all_f16(const __fp16* data, size_t num_elements);
void cactus_min_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);

__fp16 cactus_max_all_f16(const __fp16* data, size_t num_elements);
void cactus_max_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size);

void cactus_rms_norm_f16(const __fp16* input, const __fp16* weight, __fp16* output,
                          size_t batch_size, size_t dims, float eps);

void cactus_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                      size_t num_heads, size_t head_dim, size_t start_pos, float theta);

void cactus_gpt_j_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                           size_t num_heads, size_t head_dim, size_t rot_dim, size_t start_pos, float theta);

void cactus_softmax_f16(const __fp16* input, __fp16* output, size_t batch_size,
                         size_t seq_len, size_t vocab_size);

void cactus_relu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_sigmoid_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_attention_f16(const __fp16* queries, const __fp16* keys, const __fp16* values, __fp16* output,
                          size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                          size_t head_dim, float scale, const __fp16* mask, size_t position_offset = 0, size_t window_size = 0,
                          bool is_causal = true);

void cactus_attention_hybrid_int8_fp16(
    const __fp16* queries,        
    const int8_t* keys_cached, 
    const int8_t* values_cached, 
    const float* k_scales,
    const float* v_scales, 
    const __fp16* keys_new, 
    const __fp16* values_new, 
    __fp16* output,
    size_t batch_size, size_t seq_len, size_t cache_len, size_t new_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t position_offset = 0, bool is_causal = true, size_t window_size = 0,
    size_t group_size = KV_QUANT_GROUP_SIZE);

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation);

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

void cactus_conv1d_f16(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t K,
    size_t stride
);

void cactus_stft_magnitude_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t K, size_t stride,
    size_t num_fft_bins
);

void cactus_conv1d_f16_k7s3_oc8(
    const __fp16* input,
    const __fp16* Wpack,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out
);

void cactus_bilinear_interpolation_f16(const __fp16* input, __fp16* output, size_t src_height, size_t src_width, size_t embed_dim,
                                       size_t dst_height, size_t dst_width);

void cactus_sample_f32(const float* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);
void cactus_sample_f16(const __fp16* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);

void cactus_concat_f16(const __fp16* input1, const __fp16* input2, __fp16* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis);

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale = 1.0f);
float cactus_fp16_max_abs(const __fp16* src, size_t count);

void cactus_quantize_kv_fp16_to_int8(
    const __fp16* src,
    int8_t* dst,
    float* scales,
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t group_size = KV_QUANT_GROUP_SIZE);

inline size_t kv_scales_count(size_t seq_len, size_t kv_heads, size_t head_dim, size_t group_size = KV_QUANT_GROUP_SIZE) {
    size_t num_groups = (head_dim + group_size - 1) / group_size;
    return seq_len * kv_heads * num_groups;
}

void cactus_unpack_int4_to_int8(const uint8_t* packed, int8_t* unpacked, size_t unpacked_count);

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
);

#endif
