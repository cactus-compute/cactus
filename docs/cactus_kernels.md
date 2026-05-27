---
title: "Cactus Kernels API Reference"
description: "ARM NEON SIMD kernel library for matrix multiplication, attention, convolution, quantization, DSP, and image processing. The computational foundation of the Cactus inference engine."
keywords: ["ARM NEON", "SIMD", "kernels", "matmul", "attention", "quantization", "CQ", "DSP", "image processing"]
---

# Cactus Kernels API Documentation

The Cactus Kernels layer (`cactus-kernels`) provides hand-tuned ARM NEON SIMD implementations of the core operations used by the graph and engine layers. All kernels operate on FP16 (`__fp16`) data by default and are designed for mobile ARM chips (Apple Silicon, Snapdragon, Exynos, Tensor, etc.).

Header: `cactus-kernels/cactus_kernels.h`

## Precision Types

```cpp
enum class Precision {
    INT8,   // 8-bit integer (KV cache quantization)
    FP16,   // 16-bit float (default compute precision)
    FP32,   // 32-bit float
    CQ1,    // Cactus Quant 1-bit
    CQ2,    // Cactus Quant 2-bit
    CQ3,    // Cactus Quant 3-bit
    CQ4     // Cactus Quant 4-bit
};
```

## Element-wise Arithmetic

```cpp
void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t n);
void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t n);
void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t n);
void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t n);
void cactus_add_clipped_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t n);
void cactus_add_scaled_f16(const __fp16* base, const __fp16* src, __fp16* output, size_t n, float scale);
```

### Broadcast Variants

For tensors with different shapes, broadcast versions handle stride-based indexing:

```cpp
void cactus_add_broadcast_f16(
    const __fp16* a, const __fp16* b, __fp16* output,
    const size_t* a_strides, const size_t* b_strides,
    const size_t* output_shape, size_t ndim);
// Also: subtract, multiply, divide broadcast variants
```

### Scalar Operations

```cpp
enum class ScalarOpType { ADD, SUBTRACT, MULTIPLY, DIVIDE, ABS, EXP, POW, SQRT, COS, SIN, LOG };

void cactus_scalar_op_f16(
    const __fp16* input, __fp16* output, size_t n,
    float scalar_value, ScalarOpType op_type);
```

## Reductions

```cpp
// Full reductions (all elements)
double cactus_sum_all_f16(const __fp16* data, size_t n);
double cactus_mean_all_f16(const __fp16* data, size_t n);
double cactus_variance_all_f16(const __fp16* data, size_t n);
__fp16 cactus_min_all_f16(const __fp16* data, size_t n);
__fp16 cactus_max_all_f16(const __fp16* data, size_t n);

// Axis reductions
void cactus_sum_axis_f16(const __fp16* input, __fp16* output,
    size_t outer_size, size_t axis_size, size_t inner_size);
// Also: mean, variance, min, max axis variants
```

## Matrix Multiplication

### FP16 Matmul

```cpp
void cactus_matmul_f16(
    const __fp16* a,             // (M, K)
    const __fp16* b_transposed,  // (N, K) — pre-transposed RHS
    __fp16* c,                   // (M, N) output
    size_t M, size_t K, size_t N);
```

### Cactus Quant (CQ) Matmul

CQ quantization uses Hadamard rotation + per-group codebooks for 1-4 bit weight compression:

```cpp
struct CactusQuantMatrix {
    uint32_t bits;           // 1, 2, 3, or 4
    uint32_t K, N;           // matrix dimensions
    uint32_t group_size;     // elements per codebook group
    uint32_t num_groups;
    uint32_t flags;          // ORTHOGONAL, INTERLEAVED_4ROW
    const __fp16* codebook;
    const __fp16* input_scale;
    const __fp16* norms;
    const uint8_t* packed_indices;
    const int8_t* left_signs;
    const int8_t* right_signs;
    const uint32_t* permutation;
    const __fp16* rotation;
    // ...
};

// Unified dispatch (picks gemv or gemm based on M)
void cactus_quant_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);

// Bit-width specific
void cactus_quant_4bit_gemv(const CactusQuantMatrix* W, const __fp16* x, __fp16* y);
void cactus_quant_4bit_gemm(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);
void cactus_quant_2bit_gemv(const CactusQuantMatrix* W, const __fp16* x, __fp16* y);
void cactus_quant_2bit_gemm(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);
void cactus_quant_1bit_gemv(const CactusQuantMatrix* W, const __fp16* x, __fp16* y);
void cactus_quant_1bit_gemm(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);
void cactus_quant_3bit_gemv(const CactusQuantMatrix* W, const __fp16* x, __fp16* y);
void cactus_quant_3bit_gemm(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);

// Orthogonal rotation variant
void cactus_quant_orthogonal_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C);
```

### Embedding Dequantization

```cpp
// Hadamard-rotated embedding row dequantization
void cactus_quant_dequantize_hadamard_embedding_row(
    uint32_t bits, uint32_t hidden_dim, uint32_t group_size, uint32_t num_groups,
    size_t row, const uint8_t* packed_base, const __fp16* codebook,
    const __fp16* norms, const __fp16* input_scale_recip,
    const int8_t* left_signs, const int8_t* right_signs,
    const uint32_t* permutation, __fp16* out_row);

// Orthogonal rotation variant
void cactus_quant_dequantize_orthogonal_embedding_row(
    uint32_t bits, uint32_t K, size_t row,
    const uint8_t* packed_base, const __fp16* codebook,
    const __fp16* norms, const __fp16* input_scale_recip,
    const __fp16* rotation, uint32_t flags, __fp16* out_row);
```

## Attention

### Standard FP16 Attention

```cpp
void cactus_attention_f16(
    const __fp16* queries,    // (batch, seq, num_q_heads, head_dim)
    const __fp16* keys,       // (batch, kv_seq, num_kv_heads, head_dim)
    const __fp16* values,     // (batch, kv_seq, num_kv_heads, head_dim)
    __fp16* output,
    size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale,
    const __fp16* mask = nullptr,
    size_t position_offset = 0,
    size_t window_size = 0,       // 0 = full context, >0 = sliding window
    bool is_causal = true,
    bool mask_is_additive = false,
    bool mask_per_head = false,
    size_t v_head_dim = 0,        // 0 = same as head_dim
    float logit_cap = 0.0f);
```

Supports grouped-query attention (GQA) when `num_q_heads > num_kv_heads`, sliding window attention, additive masks, and logit soft-capping.

### Hybrid INT8/FP16 Attention (KV Cache)

For models with INT8-quantized KV cache and FP16 new tokens:

```cpp
void cactus_attention_hybrid_int8_fp16(
    const __fp16* queries,
    const int8_t* keys_cached,      // INT8 quantized cache
    const int8_t* values_cached,
    const float* k_scales,          // per-group dequant scales
    const float* v_scales,
    const __fp16* keys_new,         // FP16 new tokens
    const __fp16* values_new,
    __fp16* output,
    size_t batch_size, size_t seq_len,
    size_t cache_len, size_t new_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale,
    size_t position_offset = 0,
    bool is_causal = true,
    size_t window_size = 0,
    size_t group_size = 32,         // KV_QUANT_GROUP_SIZE
    size_t v_head_dim = 0);
```

## Normalization

```cpp
void cactus_rms_norm_f16(
    const __fp16* input, const __fp16* weight, __fp16* output,
    size_t batch_size, size_t dims, float eps);

void cactus_layer_norm_f16(
    const __fp16* input, const __fp16* weight, const __fp16* bias, __fp16* output,
    size_t batch_size, size_t dims, float eps);

void cactus_batchnorm_f16(
    const __fp16* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, __fp16* output,
    size_t outer_size, size_t channels, size_t inner_size, float epsilon);

void cactus_softmax_f16(
    const __fp16* input, __fp16* output,
    size_t batch_size, size_t seq_len, size_t vocab_size);
```

## Positional Encoding

```cpp
void cactus_rope_f16(
    const __fp16* input, __fp16* output,
    size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim,
    size_t start_pos, float theta);

void cactus_gpt_j_rope_f16(
    const __fp16* input, __fp16* output,
    size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim,
    size_t rot_dim, size_t start_pos, float theta);
```

## Activation Functions

```cpp
void cactus_relu_f16(const __fp16* input, __fp16* output, size_t n);
void cactus_leaky_relu_f16(const __fp16* input, __fp16* output, size_t n, float negative_slope);
void cactus_clamp_f16(const __fp16* input, __fp16* output, size_t n, float lo, float hi);
void cactus_silu_f16(const __fp16* input, __fp16* output, size_t n);
void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t n);
void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t n);
void cactus_sigmoid_f16(const __fp16* input, __fp16* output, size_t n);
void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t n);

void cactus_glu_f16(const __fp16* input, __fp16* output,
    size_t outer_size, size_t split_size, size_t inner_size);
```

## Convolution

### 1D Convolution

```cpp
void cactus_conv1d_f16(const __fp16* input, const __fp16* weight, const __fp16* bias,
    __fp16* output, size_t N, size_t L, size_t C_in, size_t C_out, size_t K, size_t stride);

void cactus_conv1d_f16_k3(const __fp16* input, const __fp16* weight, __fp16* output,
    size_t N, size_t L, size_t C_in, size_t C_out, size_t stride);

void cactus_conv1d_causal_depthwise_f16(const __fp16* input, const __fp16* weight, __fp16* output,
    size_t N, size_t L, size_t C, size_t K, size_t dilation);

void cactus_conv1d_same_depthwise_f16_k9(const __fp16* input, const __fp16* weight,
    const __fp16* bias, __fp16* output, size_t N, size_t L, size_t C);

void cactus_conv1d_pointwise_f16_gemm(const __fp16* input, const __fp16* weight,
    const __fp16* bias, __fp16* output, size_t N, size_t L, size_t C_in, size_t C_out);
```

### 2D Convolution

```cpp
void cactus_conv2d_f16_k3s2p1_nchw(...);             // 3x3 stride-2 pad-1
void cactus_conv2d_depthwise_f16_k3s2p1_nchw(...);   // depthwise 3x3 stride-2
void cactus_conv2d_pointwise_f16_1x1_nchw_gemm(...); // 1x1 pointwise via GEMM
void cactus_conv2d_f16_k3s1p1_nchw(...);             // 3x3 stride-1 pad-1
```

## Recurrent Layers

```cpp
void cactus_lstm_cell_f16(
    const __fp16* x, const __fp16* h_prev, const __fp16* c_prev,
    const __fp16* weight_ih, const __fp16* weight_hh,
    const __fp16* bias_ih, const __fp16* bias_hh,
    __fp16* h_new, __fp16* c_new,
    size_t batch_size, size_t input_size, size_t hidden_size);

void cactus_bilstm_sequence_f16(
    const __fp16* input,
    const __fp16* w_ih_fwd, const __fp16* w_hh_fwd, const __fp16* b_ih_fwd, const __fp16* b_hh_fwd,
    const __fp16* w_ih_bwd, const __fp16* w_hh_bwd, const __fp16* b_ih_bwd, const __fp16* b_hh_bwd,
    __fp16* output, size_t batch_size, size_t seq_len, size_t input_size, size_t hidden_size);

void cactus_gated_deltanet_decode_f16(
    const __fp16* q, const __fp16* k, const __fp16* v,
    const __fp16* gate, const __fp16* beta, const __fp16* state,
    __fp16* out, size_t B, size_t Hq, size_t Hv, size_t K, size_t V, float scale);

void cactus_gated_deltanet_prefill_f16(
    const __fp16* q, const __fp16* k, const __fp16* v,
    const __fp16* gate, const __fp16* beta, const __fp16* state,
    __fp16* out, size_t B, size_t T, size_t Hq, size_t Hv,
    size_t K, size_t V, size_t chunk_size, float scale);
```

## Sampling

```cpp
void cactus_sample_f16(
    const __fp16* logits, uint32_t* output, size_t vocab_size,
    float temperature, float top_p, size_t top_k, size_t random_seed,
    const float* bias_values = nullptr,
    const uint32_t* bias_indices = nullptr,
    size_t bias_count = 0);

// Extended version with min_p and repetition penalty
void cactus_sample_f16_ex(
    const __fp16* logits, uint32_t* output, size_t vocab_size,
    float temperature, float top_p, float min_p, float repetition_penalty,
    size_t top_k, size_t random_seed,
    const float* bias_values = nullptr,
    const uint32_t* bias_indices = nullptr,
    size_t bias_count = 0);
```

## DSP (Digital Signal Processing)

```cpp
void cactus_rfft_f32_1d(const float* input, float* output, size_t n, const char* norm);
void cactus_irfft_f32_1d(const float* input, float* output, size_t n, const char* norm);
float cactus_hertz_to_mel(float freq, const char* mel_scale);
float cactus_mel_to_hertz(float mels, const char* mel_scale);

void cactus_generate_mel_filter_bank(
    float* mel_filters, int num_frequency_bins, int num_mel_filters,
    float min_frequency, float max_frequency, int sampling_rate,
    const char* norm, const char* mel_scale, bool triangularize_in_mel_space);

void cactus_compute_spectrogram_f32(
    const float* waveform, size_t waveform_length,
    const float* window, size_t window_length,
    size_t frame_length, size_t hop_length, const size_t* fft_length,
    float* spectrogram, float power,
    bool center, const char* pad_mode, bool onesided,
    float dither, const float* preemphasis,
    const float* mel_filters, size_t mel_filters_size,
    float mel_floor, const char* log_mel,
    float reference, float min_value, const float* db_range,
    bool remove_dc_offset);
```

## Image Processing

```cpp
// Load image from file (JPEG, PNG, BMP, etc.)
unsigned char* cactus_image_load(const char* path, int* width, int* height, int* channels, int desired_channels);
void cactus_image_free(unsigned char* data);

// Resize
void cactus_image_resize_uint8(const unsigned char* input, int src_w, int src_h,
    unsigned char* output, int dst_w, int dst_h, int channels);
void cactus_image_resize_float(const float* input, int src_w, int src_h,
    float* output, int dst_w, int dst_h, int channels);

// Normalize with mean/std
void cactus_image_normalize(const float* input, float* output,
    int width, int height, int channels,
    float rescale_factor, const float* mean, const float* std_dev);

// Convert to vision transformer patches
void cactus_image_to_patches(const float* image, float* patches,
    int width, int height, int channels, int patch_size);

// Channel conversion
void cactus_image_convert_to_rgb(const unsigned char* input, unsigned char* output,
    int width, int height, int channels);
```

## Precision Conversion

```cpp
void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale = 1.0f);
void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);
float cactus_fp16_max_abs(const __fp16* src, size_t count);
```

## KV Cache Quantization

```cpp
constexpr size_t KV_QUANT_GROUP_SIZE = 32;

void cactus_quantize_kv_fp16_to_int8(
    const __fp16* src, int8_t* dst, float* scales,
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t group_size = KV_QUANT_GROUP_SIZE);

inline size_t kv_scales_count(
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t group_size = KV_QUANT_GROUP_SIZE);
```

## Miscellaneous

```cpp
void cactus_stft_f16(const __fp16* input, const __fp16* weight, __fp16* output,
    size_t N, size_t L, size_t C_in, size_t C_out, size_t K, size_t stride, size_t num_fft_bins);

void cactus_bilinear_interpolation_f16(const __fp16* input, __fp16* output,
    size_t src_height, size_t src_width, size_t embed_dim,
    size_t dst_height, size_t dst_width, bool align_corners = true);

void cactus_maxpool1d_f16(const __fp16* input, __fp16* output,
    size_t batch_size, size_t channels, size_t input_length,
    size_t kernel_size, size_t stride);

void cactus_altup_predict_f16(const __fp16* coefs, const __fp16* const* streams,
    __fp16* output, size_t n, size_t seq_len, size_t hidden_dim);

void cactus_altup_correct_f16(const __fp16* coefs, const __fp16* innovation,
    const __fp16* const* predictions, __fp16* output,
    size_t n, size_t seq_len, size_t hidden_dim);

void cactus_gaussian_topk_f16(const __fp16* input, __fp16* output,
    size_t rows, size_t cols, float ppf);
```

## Directory Structure

```
cactus-kernels/
  cactus_kernels.h       # public API (this file)
  threading.h            # thread pool utilities
  libs/
    stb_image.h          # vendored image loading
    stb_image_resize2.h  # vendored image resizing
  src/
    arithmetic.cpp       # element-wise ops, broadcast, reductions
    matmul.cpp           # FP16 GEMM
    quant.cpp            # CQ 1-4 bit GEMV/GEMM, dequantization
    attention.cpp        # attention kernels (FP16, hybrid INT8/FP16)
    activations.cpp      # relu, silu, gelu, sigmoid, tanh, glu, clamp
    norm.cpp             # rms_norm, layer_norm, batchnorm, softmax
    conv.cpp             # conv1d, conv2d variants
    sample.cpp           # top-k/top-p/min-p sampling
    dsp.cpp              # rfft, irfft, mel filter bank, spectrogram
    image.cpp            # image load/resize/normalize/patches
    rope.cpp             # rotary position embedding
    transpose.cpp        # 2D and N-D transpose
    lstm.cpp             # LSTM cell, BiLSTM sequence
    deltanet.cpp         # gated DeltaNet decode/prefill
    wav.h                # WAV file loading + 16kHz resampling
  tests/
    test_kernels.cpp     # 36 tests across 7 suites
    test_utils.h         # test runner, fp16 comparison helpers
```

## See Also

- [Cactus Graph API](/docs/cactus_graph.md) — Computation graph built on top of these kernels
- [Cactus Engine API](/docs/cactus_engine.md) — High-level inference API
- [TurboQuant-H](/blog/turboquant-h.md) — 2-bit embedding quantization using the CQ kernel infrastructure
