#ifndef CACTUS_BRIDGE_H
#define CACTUS_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cactus_model_t;

cactus_model_t cactus_init(const char* model_path, const char* corpus_dir, bool cache_index);
void cactus_destroy(cactus_model_t model);
void cactus_reset(cactus_model_t model);

int cactus_prefill(cactus_model_t model,
                   const char* messages_json,
                   char* response_buffer,
                   size_t buffer_size,
                   const char* options_json,
                   const char* tools_json,
                   const uint8_t* pcm_buffer,
                   size_t pcm_buffer_size);

void cactus_npu_set_enabled(bool enabled);
bool cactus_npu_enabled(void);
void cactus_mps_set_enabled(bool enabled);
bool cactus_mps_enabled(void);

#ifdef __cplusplus
}
#endif

#endif
