#ifndef CACTUS_FFI_H
#define CACTUS_FFI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if __GNUC__ >= 4
    #define CACTUS_FFI_EXPORT __attribute__((visibility("default")))
    #define CACTUS_FFI_LOCAL  __attribute__((visibility("hidden")))
#else
    #define CACTUS_FFI_EXPORT
    #define CACTUS_FFI_LOCAL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cactus_model_t;
typedef void* cactus_index_t;
typedef void* cactus_stream_transcribe_t;

typedef void (*cactus_token_callback)(const char* token, uint32_t token_id, void* user_data);

CACTUS_FFI_EXPORT cactus_model_t cactus_init(
    const char* model_path,
    const char* corpus_dir,                 // optional: NULL if no RAG corpus
    bool cache_index                        // false = always rebuild index, true = load cached if available
);

CACTUS_FFI_EXPORT void cactus_destroy(cactus_model_t model);
CACTUS_FFI_EXPORT void cactus_reset(cactus_model_t model);
CACTUS_FFI_EXPORT void cactus_stop(cactus_model_t model);

CACTUS_FFI_EXPORT int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,               // optional
    const char* tools_json,                 // optional
    cactus_token_callback callback,         // optional
    void* user_data                         // optional
);

CACTUS_FFI_EXPORT int cactus_prefill(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,               // optional
    const char* tools_json                  // optional
);

CACTUS_FFI_EXPORT int cactus_tokenize(
    cactus_model_t model,
    const char* text,
    uint32_t* token_buffer,
    size_t token_buffer_len,
    size_t* out_token_len
);

CACTUS_FFI_EXPORT int cactus_score_window(
    cactus_model_t model,
    const uint32_t* tokens,
    size_t token_len,
    size_t start,
    size_t end,
    size_t context,
    char* response_buffer,
    size_t buffer_size
);

CACTUS_FFI_EXPORT int cactus_transcribe(
    cactus_model_t model,
    const char* audio_file_path,            // NULL if using pcm_buffer
    const char* prompt,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,               // optional
    cactus_token_callback callback,         // optional
    void* user_data,                        // optional
    const uint8_t* pcm_buffer,              // NULL if using audio_file_path
    size_t pcm_buffer_size
);

CACTUS_FFI_EXPORT int cactus_detect_language(
    cactus_model_t model,
    const char* audio_file_path,            // NULL if using pcm_buffer
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,               // optional
    const uint8_t* pcm_buffer,              // NULL if using audio_file_path
    size_t pcm_buffer_size
);

CACTUS_FFI_EXPORT cactus_stream_transcribe_t cactus_stream_transcribe_start(
    cactus_model_t model,
    const char* options_json                // optional
);

CACTUS_FFI_EXPORT int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size,
    char* response_buffer,
    size_t buffer_size
);

CACTUS_FFI_EXPORT int cactus_stream_transcribe_stop(
    cactus_stream_transcribe_t stream,
    char* response_buffer,
    size_t buffer_size
);

CACTUS_FFI_EXPORT int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim,
    bool normalize
);

CACTUS_FFI_EXPORT int cactus_image_embed(
    cactus_model_t model,
    const char* image_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
);

CACTUS_FFI_EXPORT int cactus_audio_embed(
    cactus_model_t model,
    const char* audio_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
);

CACTUS_FFI_EXPORT int cactus_vad(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
);

CACTUS_FFI_EXPORT int cactus_rag_query(
    cactus_model_t model,
    const char* query,
    char* response_buffer,
    size_t buffer_size,
    size_t top_k
);


CACTUS_FFI_EXPORT cactus_index_t cactus_index_init(
    const char* index_dir,
    size_t embedding_dim
);

CACTUS_FFI_EXPORT int cactus_index_add(
    cactus_index_t index,
    const int* ids,
    const char** documents,
    const char** metadatas,                 // optional: can be NULL
    const float** embeddings,
    size_t count,
    size_t embedding_dim
);

CACTUS_FFI_EXPORT int cactus_index_delete(
    cactus_index_t index,
    const int* ids,
    size_t ids_count
);

CACTUS_FFI_EXPORT int cactus_index_get(
    cactus_index_t index,
    const int* ids,
    size_t ids_count,
    char** document_buffers,
    size_t* document_buffer_sizes,
    char** metadata_buffers,
    size_t* metadata_buffer_sizes,
    float** embedding_buffers,
    size_t* embedding_buffer_sizes
);

CACTUS_FFI_EXPORT int cactus_index_query(
    cactus_index_t index,
    const float** embeddings,
    size_t embeddings_count,
    size_t embedding_dim,
    const char* options_json,               // optional
    int** id_buffers,
    size_t* id_buffer_sizes,
    float** score_buffers,
    size_t* score_buffer_sizes
);

CACTUS_FFI_EXPORT int cactus_index_compact(cactus_index_t index);
CACTUS_FFI_EXPORT void cactus_index_destroy(cactus_index_t index);

CACTUS_FFI_EXPORT const char* cactus_get_last_error(void);

// level: 0=DEBUG, 1=INFO, 2=WARN (default), 3=ERROR, 4=NONE
CACTUS_FFI_EXPORT void cactus_log_set_level(int level);

typedef void (*cactus_log_callback_t)(int level, const char* component, const char* message, void* user_data);
CACTUS_FFI_EXPORT void cactus_log_set_callback(cactus_log_callback_t callback, void* user_data);

CACTUS_FFI_EXPORT void cactus_set_telemetry_environment(const char* framework, const char* cache_location, const char* version);
CACTUS_FFI_EXPORT void cactus_set_app_id(const char* app_id);
CACTUS_FFI_EXPORT void cactus_telemetry_flush(void);
CACTUS_FFI_EXPORT void cactus_telemetry_shutdown(void);

// cactus graph export 
typedef void* cactus_graph_t;
typedef uint64_t cactus_node_t;

typedef struct {
    int32_t precision;
    size_t rank;
    size_t shape[8]; 
    size_t num_elements;
    size_t byte_size;
} cactus_tensor_info_t;

CACTUS_FFI_EXPORT cactus_graph_t cactus_graph_create(void);
CACTUS_FFI_EXPORT void cactus_graph_destroy(cactus_graph_t graph);
CACTUS_FFI_EXPORT int cactus_graph_hard_reset(cactus_graph_t graph);

CACTUS_FFI_EXPORT int cactus_graph_input(
    cactus_graph_t graph, const size_t* shape, size_t rank, int32_t precision,
cactus_node_t* out_node);

CACTUS_FFI_EXPORT int cactus_graph_set_input(
    cactus_graph_t graph, cactus_node_t node, const void* data, int32_t
precision);
CACTUS_FFI_EXPORT int cactus_graph_set_external_input(
    cactus_graph_t graph, cactus_node_t node, void* data, int32_t precision);

CACTUS_FFI_EXPORT int cactus_graph_precision_cast(
    cactus_graph_t graph, cactus_node_t input, int32_t target_precision, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_quantize_activations(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_add(cactus_graph_t graph, cactus_node_t a,
cactus_node_t b, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_add_clipped(cactus_graph_t graph, cactus_node_t a,
cactus_node_t b, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_subtract(cactus_graph_t graph, cactus_node_t
a, cactus_node_t b, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_multiply(cactus_graph_t graph, cactus_node_t
a, cactus_node_t b, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_divide(cactus_graph_t graph, cactus_node_t
a, cactus_node_t b, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_scalar_add(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_subtract(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_multiply(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_divide(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_exp(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_sqrt(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_cos(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_sin(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scalar_log(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_abs(cactus_graph_t graph, cactus_node_t x,
cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_pow(cactus_graph_t graph, cactus_node_t x,
float exponent, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_view(
    cactus_graph_t graph, cactus_node_t x, const size_t* shape, size_t rank,
cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_flatten(
    cactus_graph_t graph, cactus_node_t x, int32_t start_dim, int32_t end_dim,
cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_reshape(
    cactus_graph_t graph, cactus_node_t x, const size_t* shape, size_t rank, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_transpose(
    cactus_graph_t graph, cactus_node_t x, int32_t backend, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_transpose_n(
    cactus_graph_t graph, cactus_node_t x, const size_t* permutation, size_t rank, int32_t backend, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_slice(
    cactus_graph_t graph, cactus_node_t x, int32_t axis, size_t start, size_t length, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_index(
    cactus_graph_t graph, cactus_node_t x, size_t index_value, int32_t dim, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_sum(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_mean(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_variance(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_min(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_max(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_concat(
    cactus_graph_t graph, cactus_node_t a, cactus_node_t b, int32_t axis,
cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_cat(
    cactus_graph_t graph, const cactus_node_t* nodes, size_t count, int32_t
axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_matmul(
    cactus_graph_t graph, cactus_node_t a, cactus_node_t b, bool pretransposed_rhs, int32_t backend, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gather(
    cactus_graph_t graph, cactus_node_t tensor, cactus_node_t indices, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_embedding_from_tensor(
    cactus_graph_t graph, cactus_node_t embedding_tensor, cactus_node_t indices, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_embedding_from_file(
    cactus_graph_t graph, const char* filename, cactus_node_t indices, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_mmap_embeddings(
    cactus_graph_t graph, const char* filename, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_mmap_weights(
    cactus_graph_t graph, const char* filename, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_bilinear_interpolation(
    cactus_graph_t graph, cactus_node_t pos_embeds, size_t dst_height, size_t dst_width, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_set_grouped_scales(
    cactus_graph_t graph, cactus_node_t node, size_t group_size, size_t num_groups, void* scales_ptr);
CACTUS_FFI_EXPORT int cactus_graph_set_interleaved(
    cactus_graph_t graph, cactus_node_t node, bool interleaved, size_t original_n);
CACTUS_FFI_EXPORT int cactus_graph_release_weight_pages(cactus_graph_t graph, cactus_node_t node);
CACTUS_FFI_EXPORT int cactus_graph_prefetch_weight_pages(cactus_graph_t graph, cactus_node_t node);
CACTUS_FFI_EXPORT int cactus_graph_release_all_weight_pages(cactus_graph_t graph);

CACTUS_FFI_EXPORT int cactus_graph_relu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_silu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gelu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gelu_erf(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_sigmoid(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_tanh(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_glu(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_layernorm(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, float epsilon, bool has_bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_groupnorm(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, size_t num_groups, float epsilon, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_batchnorm(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, cactus_node_t running_mean, cactus_node_t running_var, int32_t axis, float epsilon, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_topk(cactus_graph_t graph, cactus_node_t input, size_t k, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_rms_norm(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, float epsilon, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_rope(
    cactus_graph_t graph, cactus_node_t input, float theta, size_t position_offset, int32_t backend, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_rope_gptj(
    cactus_graph_t graph, cactus_node_t input, float theta, size_t position_offset, size_t rot_dim, int32_t backend, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_softmax(cactus_graph_t graph, cactus_node_t input, int32_t axis, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_attention(
    cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, float scale, bool is_causal, size_t position_offset, size_t window_size, int32_t backend, bool use_mask, cactus_node_t mask, bool additive_mask, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_rel_pos_bias(
    cactus_graph_t graph, cactus_node_t query, cactus_node_t relative_key, float scale, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_attention_int8_hybrid(
    cactus_graph_t graph, cactus_node_t query, cactus_node_t key_new, cactus_node_t value_new, float scale, size_t position_offset,
    const int8_t* cached_keys, const int8_t* cached_values, const float* k_scales, const float* v_scales,
    size_t cache_len, size_t num_kv_heads, size_t head_dim, size_t window_size, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_conv1d_causal(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t kernel_size, size_t dilation, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv1d_k3(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t stride, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv1d_k7s3(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv1d(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, size_t stride, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv1d_same_depthwise_k9(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv1d_pointwise(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv2d_k3s2p1(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv2d_depthwise_k3s2p1(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_conv2d_pointwise_1x1(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_lstm_cell(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t h_prev, cactus_node_t c_prev, cactus_node_t weight_ih, cactus_node_t weight_hh, cactus_node_t bias_ih, cactus_node_t bias_hh, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gated_deltanet_decode(
    cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, cactus_node_t gate_log, cactus_node_t beta, cactus_node_t initial_state, float scale, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gated_deltanet_prefill(
    cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, cactus_node_t gate_log, cactus_node_t beta, cactus_node_t initial_state, size_t chunk_size, float scale, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_stft(
    cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t stride, size_t num_fft_bins, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_altup_predict(
    cactus_graph_t graph, cactus_node_t coefs, const cactus_node_t* streams, size_t num_streams, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_altup_correct(
    cactus_graph_t graph, cactus_node_t coefs, cactus_node_t innovation, const cactus_node_t* predictions, size_t num_predictions, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_gaussian_topk(
    cactus_graph_t graph, cactus_node_t input, float ppf, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_moe_layer_gated(
    cactus_graph_t graph, cactus_node_t hidden, cactus_node_t routing_probs, cactus_node_t topk_indices,
    const cactus_node_t* w1_weights, const cactus_node_t* w3_weights, const cactus_node_t* w2_weights,
    size_t num_experts, size_t num_experts_per_tok, bool normalize_routing, float epsilon, float routed_scaling_factor, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_moe_layer_ungated(
    cactus_graph_t graph, cactus_node_t hidden, cactus_node_t routing_probs, cactus_node_t topk_indices,
    const cactus_node_t* w1_weights, const cactus_node_t* w2_weights,
    size_t num_experts, size_t num_experts_per_tok, bool normalize_routing, float epsilon, float routed_scaling_factor, int32_t activation, cactus_node_t* out);

CACTUS_FFI_EXPORT int cactus_graph_sample(
    cactus_graph_t graph, cactus_node_t logits, float temperature, float top_p, size_t top_k, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_scatter_topk(
    cactus_graph_t graph, cactus_node_t indices, cactus_node_t values, size_t num_classes, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_persistent(
    cactus_graph_t graph, cactus_node_t source_node, cactus_node_t* out);
CACTUS_FFI_EXPORT int cactus_graph_is_populated(
    cactus_graph_t graph, cactus_node_t persistent_node, int32_t* out_is_populated);
CACTUS_FFI_EXPORT int cactus_graph_invalidate_persistent(
    cactus_graph_t graph, cactus_node_t persistent_node);

CACTUS_FFI_EXPORT int cactus_graph_execute(cactus_graph_t graph);
CACTUS_FFI_EXPORT int cactus_graph_get_output_ptr(cactus_graph_t graph,
cactus_node_t node, void** out_ptr);
CACTUS_FFI_EXPORT int cactus_graph_get_output_info(cactus_graph_t graph,
cactus_node_t node, cactus_tensor_info_t* out_info);

#ifdef __cplusplus
}
#endif

#endif // CACTUS_FFI_H
