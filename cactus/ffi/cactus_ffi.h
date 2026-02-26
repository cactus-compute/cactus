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

CACTUS_FFI_EXPORT void cactus_set_telemetry_environment(const char* framework, const char* cache_location);
CACTUS_FFI_EXPORT void cactus_telemetry_flush(void);
CACTUS_FFI_EXPORT void cactus_telemetry_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // CACTUS_FFI_H
