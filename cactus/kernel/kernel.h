#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>
#include <arm_neon.h>
#include <vector>

enum class Precision;

enum class ScalarOpType {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    ABS,
    EXP,
    POW,
    SQRT,
    COS,
    SIN,
    LOG
};

constexpr size_t KV_QUANT_GROUP_SIZE = 32;

void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_add_f16_clipped(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements);
void cactus_add_scaled_f16(const __fp16* base, const __fp16* src, __fp16* output, size_t num_elements, float scale);
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

void cactus_gemv_int8_i8mm(const int8_t* A, float A_scale,
                            const int8_t* B, const __fp16* B_scales,
                            __fp16* C, size_t K, size_t N, size_t group_size);

void cactus_gemm_int8_i8mm(const int8_t* A, const float* A_scales,
                            const int8_t* B, const __fp16* B_scales,
                            __fp16* C, size_t M, size_t K, size_t N, size_t group_size);

void cactus_gemv_int4(const int8_t* A, float A_scale,
                      const int8_t* B_packed, const __fp16* B_scales,
                      __fp16* C, size_t K, size_t N, size_t group_size);

void cactus_gemm_int4(const int8_t* A, const float* A_scales,
                      const int8_t* B_packed, const __fp16* B_scales,
                      __fp16* C, size_t M, size_t K, size_t N, size_t group_size);

void cactus_matmul_int4(const int8_t* A, const float* A_scales,
                        const int8_t* B_packed, const __fp16* B_scales,
                        __fp16* C, size_t M, size_t K, size_t N, size_t group_size);

void cactus_matmul_integer(Precision precision,
                            const int8_t* A, const float* A_scales,
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

void cactus_layer_norm_f16(const __fp16* input, const __fp16* weight, const __fp16* bias,
                            __fp16* output, size_t batch_size, size_t dims, float eps);

void cactus_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                      size_t num_heads, size_t head_dim, size_t start_pos, float theta);

void cactus_gpt_j_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                           size_t num_heads, size_t head_dim, size_t rot_dim, size_t start_pos, float theta);

void cactus_softmax_f16(const __fp16* input, __fp16* output, size_t batch_size,
                         size_t seq_len, size_t vocab_size);

void cactus_relu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_leaky_relu_f16(const __fp16* input, __fp16* output, size_t num_elements, float negative_slope);

void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_sigmoid_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_glu_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t split_size,
    size_t inner_size
);

void cactus_glu_f32(
    const float* input,
    float* output,
    size_t outer_size,
    size_t split_size,
    size_t inner_size
);

void cactus_batchnorm_f16(
    const __fp16* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    __fp16* output,
    size_t outer_size,
    size_t channels,
    size_t inner_size,
    float epsilon
);

void cactus_batchnorm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    size_t outer_size,
    size_t channels,
    size_t inner_size,
    float epsilon
);

void cactus_attention_f16(const __fp16* queries, const __fp16* keys, const __fp16* values, __fp16* output,
                          size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                          size_t head_dim, float scale, const __fp16* mask, size_t position_offset = 0, size_t window_size = 0,
                          bool is_causal = true, bool mask_is_additive = false, bool mask_per_head = false,
                          size_t v_head_dim = 0, float logit_cap = 0.0f);

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
    size_t group_size = KV_QUANT_GROUP_SIZE, size_t v_head_dim = 0);

void cactus_gated_deltanet_decode_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    float scale);

void cactus_gated_deltanet_prefill_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t T,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    size_t requested_chunk_size,
    float scale);

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

void cactus_stft_f16(
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

void cactus_conv1d_same_depthwise_f16_k9(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C
);

void cactus_conv2d_f16_k3s1p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out
);

void cactus_conv2d_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out
);

void cactus_conv2d_depthwise_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C,
    size_t H,
    size_t W
);

void cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out
);

void cactus_conv1d_pointwise_f16_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out
);

void cactus_bilinear_interpolation_f16(const __fp16* input, __fp16* output, size_t src_height, size_t src_width, size_t embed_dim,
                                       size_t dst_height, size_t dst_width, bool align_corners = true);

void cactus_sample_f32(const float* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);
void cactus_sample_f16(const __fp16* logits, uint32_t* output, size_t vocab_size,
                       float temperature, float top_p, size_t top_k, size_t random_seed,
                       const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                       size_t bias_count = 0);

void cactus_sample_f32_ex(const float* logits, uint32_t* output, size_t vocab_size,
                          float temperature, float top_p, float min_p, float repetition_penalty,
                          size_t top_k, size_t random_seed,
                          const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                          size_t bias_count = 0);
void cactus_sample_f16_ex(const __fp16* logits, uint32_t* output, size_t vocab_size,
                          float temperature, float top_p, float min_p, float repetition_penalty,
                          size_t top_k, size_t random_seed,
                          const float* bias_values = nullptr, const uint32_t* bias_indices = nullptr,
                          size_t bias_count = 0);

void cactus_concat_f16(const __fp16* input1, const __fp16* input2, __fp16* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis);
void cactus_cat_f16(const __fp16** inputs, __fp16* output, const size_t** input_shapes,
                      const size_t* output_shape, size_t num_inputs, size_t rank, int axis);

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

void cactus_gaussian_topk_f16(
    const __fp16* input,
    __fp16* output,
    size_t rows,
    size_t cols,
    float ppf);

void cactus_altup_predict_f16(
    const __fp16* coefs,
    const __fp16* const* streams,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim);

void cactus_altup_correct_f16(
    const __fp16* coefs,
    const __fp16* innovation,
    const __fp16* const* predictions,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim);

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

void cactus_bilstm_sequence_f16(
    const __fp16* input,
    const __fp16* weight_ih_fwd,
    const __fp16* weight_hh_fwd,
    const __fp16* bias_ih_fwd,
    const __fp16* bias_hh_fwd,
    const __fp16* weight_ih_bwd,
    const __fp16* weight_hh_bwd,
    const __fp16* bias_ih_bwd,
    const __fp16* bias_hh_bwd,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    size_t hidden_size
);

void cactus_maxpool1d_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t channels,
    size_t input_length,
    size_t kernel_size,
    size_t stride
);

// Precision::TQ2 descriptor. Pointers reference an mmap'd blob.
struct CactusTQ2 {
    uint32_t vocab;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t per_group_bytes;
    uint32_t flags;

    const float*    codebook;
    const __fp16*   input_scale;
    const __fp16*   input_scale_recip;
    const int8_t*   left_signs;
    const int8_t*   right_signs;
    const uint32_t* permutation;
    const __fp16*   scales;
    const uint8_t*  packed;

    uint32_t inv_permutation[256];
    std::vector<__fp16> input_scale_recip_storage;
};

int  cactus_tq2_load(CactusTQ2* out, const void* blob, size_t blob_size);
void cactus_tq2_dequant_row(const CactusTQ2* tq2, uint32_t token_id, __fp16* out);

// Materialize all [N, K] rows into a fp16 buffer for reference/debug use.
// Caller-provided buffer must hold N*K __fp16 elements.
void cactus_tq2_dequant_layer(const CactusTQ2* tq2, __fp16* out);


// Generic TQ-N descriptor. Backs Precision::TQ3 (3-bit) and Precision::TQ4
// (4-bit). The `bits` field tells the kernel which unpack to use. Pointers
// reference an mmap'd blob.
//
// Header layout (matches python/src/tqh_pack.py write_tq_weights):
//   136-byte fixed header, fields at:
//       4: flags             (bit 0 = hadamard indices stored in code/K order,
//                             bit 1 = 4-row panel-major indices/scales)
//     128: rotation_family   (0 = randomized hadamard gs<=256,
//                             1 = orthogonal full-width gs == K, num_groups == 1)
//     132: has_input_scale   (0/1)
//   Followed by 32-byte-aligned blocks:
//     codebook   fp32[2^bits]
//     input_scale fp16[K]                       (if has_input_scale)
//     rotation:  hadamard => int8[gs] left || int8[gs] right || u32[gs] perm
//                orth     => fp16[K*K] R
//     scales     fp16[N * num_groups]            (per-row L2 norms)
//                or fp16[ceil(N/4) * num_groups * 4] when panel-major
//     packed     uint8[N * num_groups * per_group_bytes]
//                or uint8[ceil(N/4) * num_groups * 4 * per_group_bytes]
//                when panel-major, ordered as [n_block, group, k16_chunk, lane, bytes]
//                LSB-first within each byte:
//                  bits == 3: 8 indices per 24-bit LE word (3 bytes)
//                  bits == 4: 2 indices per byte (low nibble, then high nibble)
struct CactusTQN {
    uint32_t dim0;           // rows (N)
    uint32_t dim1;           // cols (K)
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t bits;           // 3 or 4
    uint32_t rotation_family;
    uint32_t per_group_bytes; // group_size * bits / 8
    uint32_t has_input_scale;
    uint32_t flags;

    const float*    codebook;     // [2^bits]
    const __fp16*   input_scale;  // [K] or null
    const __fp16*   input_scale_recip; // [K] or null
    const __fp16*   scales;       // [N * num_groups]
    const uint8_t*  packed;       // packed indices

    // Hadamard rotation pointers (rotation_family == 0):
    const int8_t*   left_signs;
    const int8_t*   right_signs;
    const uint32_t* permutation;
    uint32_t        inv_permutation[256];

    // Orthogonal full-width rotation (rotation_family == 1):
    const __fp16*   orth_R;       // [K*K] or null
    std::vector<__fp16> input_scale_recip_storage;
};

// Loads a TQ3 (precision=11, bits=3) or TQ4 (precision=12, bits=4) blob.
int  cactus_tqn_load(CactusTQN* out, const void* blob, size_t blob_size);
// Per-row dequant — handles both rotation families and both bit-widths.
void cactus_tqn_dequant_row(const CactusTQN* tqn, uint32_t row_id, __fp16* out);
// Materialize the full [N, K] tensor as fp16 for reference/debug use.
// Caller buffer = N*K __fp16.
void cactus_tqn_dequant_layer(const CactusTQN* tqn, __fp16* out);

// Fused gemv for Precision::TQ4 with hadamard rotation: y[N] = TQ4(weight) @ x[K].
// Reads the packed indices + scales + signs/perm + codebook directly from the
// mmapped blob — never materializes the full fp16 weight matrix. Decode-only
// (M==1). The activation x is fp16[K]; output y is fp16[N].
//
// Internally folds tqn->input_scale into x once, then per row dequantizes one
// group at a time, immediately accumulates against the corresponding slice of
// x, and discards the dequantized values. Memory pressure is dominated by the
// packed indices stream (~K*N/2 bytes for the whole tensor).
void cactus_gemv_tq4_hadamard_f16(const CactusTQN* tqn, const __fp16* x,
                                   __fp16* y, size_t N);

// Packed TQ matmul paths. These never materialize the full weight matrix:
// weights are dequantized into per-row/per-group scratch inside the kernel.
// A is fp16 [M, K], packed weight rows are [N, K], C is fp16 [M, N].
void cactus_matmul_tq2_f16(const CactusTQ2* tq2, const __fp16* A, __fp16* C,
                           size_t M, size_t K, size_t N);
void cactus_matmul_tqn_f16(const CactusTQN* tqn, const __fp16* A, __fp16* C,
                           size_t M, size_t K, size_t N);

#endif
