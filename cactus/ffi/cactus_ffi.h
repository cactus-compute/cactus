#ifndef CACTUS_FFI_H
#define CACTUS_FFI_H

#include <stddef.h>
#include <stdint.h>

#if __GNUC__ >= 4
  #define CACTUS_FFI_EXPORT __attribute__ ((visibility ("default")))
  #define CACTUS_FFI_LOCAL  __attribute__ ((visibility ("hidden")))
#else
  #define CACTUS_FFI_EXPORT
  #define CACTUS_FFI_LOCAL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cactus_model_t;

typedef void (*cactus_token_callback)(const char* token, uint32_t token_id, void* user_data);

CACTUS_FFI_EXPORT cactus_model_t cactus_init(const char* model_path, size_t context_size, const char* corpus_dir);

CACTUS_FFI_EXPORT int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data
);

CACTUS_FFI_EXPORT int cactus_transcribe(
    cactus_model_t model,
    const char* audio_file_path,
    const char* prompt,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    cactus_token_callback callback,
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
);


CACTUS_FFI_EXPORT int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
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

CACTUS_FFI_EXPORT void cactus_reset(cactus_model_t model);

CACTUS_FFI_EXPORT void cactus_stop(cactus_model_t model);

CACTUS_FFI_EXPORT void cactus_destroy(cactus_model_t model);

#ifdef __cplusplus
}
#endif

#endif 