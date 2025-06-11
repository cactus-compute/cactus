#ifndef CACTUS_FFI_H
#define CACTUS_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined _WIN32 || defined __CYGWIN__
  #ifdef CACTUS_FFI_BUILDING_DLL
    #ifdef __GNUC__
      #define CACTUS_FFI_EXPORT __attribute__ ((dllexport))
    #else
      #define CACTUS_FFI_EXPORT __declspec(dllexport)
    #endif
  #else
    #ifdef __GNUC__
      #define CACTUS_FFI_EXPORT __attribute__ ((dllimport))
    #else
      #define CACTUS_FFI_EXPORT __declspec(dllimport)
    #endif
  #endif
  #define CACTUS_FFI_LOCAL
#else
  #if __GNUC__ >= 4
    #define CACTUS_FFI_EXPORT __attribute__ ((visibility ("default")))
    #define CACTUS_FFI_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define CACTUS_FFI_EXPORT
    #define CACTUS_FFI_LOCAL
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cactus_context_opaque* cactus_context_handle_t;


typedef struct cactus_init_params_c {
    const char* model_path;
    const char* chat_template; 

    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_gpu_layers;
    int32_t n_threads;
    bool use_mmap;
    bool use_mlock;
    bool embedding; 
    int32_t pooling_type; 
    int32_t embd_normalize;
    bool flash_attn;
    const char* cache_type_k; 
    const char* cache_type_v; 
    void (*progress_callback)(float progress); 

} cactus_init_params_c_t;

typedef struct cactus_completion_params_c {
    const char* prompt;
    int32_t n_predict; 
    int32_t n_threads; 
    int32_t seed;
    double temperature;
    int32_t top_k;
    double top_p;
    double min_p;
    double typical_p;
    int32_t penalty_last_n;
    double penalty_repeat;
    double penalty_freq;
    double penalty_present;
    int32_t mirostat;
    double mirostat_tau;
    double mirostat_eta;
    bool ignore_eos;
    int32_t n_probs; 
    const char** stop_sequences; 
    int stop_sequence_count;
    const char* grammar; 
    bool (*token_callback)(const char* token_json);

} cactus_completion_params_c_t;


typedef struct cactus_token_array_c {
    int32_t* tokens;
    int32_t count;
} cactus_token_array_c_t;

typedef struct cactus_float_array_c {
    float* values;
    int32_t count;
} cactus_float_array_c_t;

typedef struct cactus_completion_result_c {
    char* text; 
    int32_t tokens_predicted;
    int32_t tokens_evaluated;
    bool truncated;
    bool stopped_eos;
    bool stopped_word;
    bool stopped_limit;
    char* stopping_word; 
} cactus_completion_result_c_t;

typedef struct cactus_tokenize_result_c {
    cactus_token_array_c_t tokens;
    bool has_media;
    char** bitmap_hashes;
    int bitmap_hash_count;
    size_t* chunk_positions;
    int chunk_position_count;
    size_t* chunk_positions_media;
    int chunk_position_media_count;
} cactus_tokenize_result_c_t;

CACTUS_FFI_EXPORT cactus_context_handle_t cactus_init_context_c(const cactus_init_params_c_t* params);

CACTUS_FFI_EXPORT void cactus_free_context_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT int cactus_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    cactus_completion_result_c_t* result
);

// **MULTIMODAL COMPLETION**
CACTUS_FFI_EXPORT int cactus_multimodal_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    const char** media_paths,
    int media_count,
    cactus_completion_result_c_t* result
);

CACTUS_FFI_EXPORT void cactus_stop_completion_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT cactus_token_array_c_t cactus_tokenize_c(cactus_context_handle_t handle, const char* text);

CACTUS_FFI_EXPORT char* cactus_detokenize_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count);

CACTUS_FFI_EXPORT cactus_float_array_c_t cactus_embedding_c(cactus_context_handle_t handle, const char* text);

CACTUS_FFI_EXPORT void cactus_free_string_c(char* str);

CACTUS_FFI_EXPORT void cactus_free_token_array_c(cactus_token_array_c_t arr);

CACTUS_FFI_EXPORT void cactus_free_float_array_c(cactus_float_array_c_t arr);

CACTUS_FFI_EXPORT void cactus_free_completion_result_members_c(cactus_completion_result_c_t* result);

CACTUS_FFI_EXPORT cactus_tokenize_result_c_t cactus_tokenize_with_media_c(cactus_context_handle_t handle, const char* text, const char** media_paths, int media_count);

CACTUS_FFI_EXPORT void cactus_free_tokenize_result_c(cactus_tokenize_result_c_t* result);

CACTUS_FFI_EXPORT void cactus_set_guide_tokens_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count);

CACTUS_FFI_EXPORT int cactus_init_multimodal_c(cactus_context_handle_t handle, const char* mmproj_path, bool use_gpu);

CACTUS_FFI_EXPORT bool cactus_is_multimodal_enabled_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT bool cactus_supports_vision_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT bool cactus_supports_audio_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT void cactus_release_multimodal_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT int cactus_init_vocoder_c(cactus_context_handle_t handle, const char* vocoder_model_path);

CACTUS_FFI_EXPORT bool cactus_is_vocoder_enabled_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT int cactus_get_tts_type_c(cactus_context_handle_t handle);

CACTUS_FFI_EXPORT char* cactus_get_formatted_audio_completion_c(cactus_context_handle_t handle, const char* speaker_json_str, const char* text_to_speak);

CACTUS_FFI_EXPORT cactus_token_array_c_t cactus_get_audio_guide_tokens_c(cactus_context_handle_t handle, const char* text_to_speak);

CACTUS_FFI_EXPORT cactus_float_array_c_t cactus_decode_audio_tokens_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count);

CACTUS_FFI_EXPORT void cactus_release_vocoder_c(cactus_context_handle_t handle);

// **HIGH PRIORITY ADDITIONS**

typedef struct {
    const char* path;
    float scale;
} cactus_lora_adapter_c_t;

typedef struct {
    cactus_lora_adapter_c_t* adapters;
    int32_t count;
} cactus_lora_adapters_c_t;

typedef struct {
    char* model_name;
    int64_t model_size;
    int64_t model_params;
    double pp_avg;
    double pp_std;
    double tg_avg;
    double tg_std;
} cactus_bench_result_c_t;

// **HIGH PRIORITY: Benchmarking**
CACTUS_FFI_EXPORT cactus_bench_result_c_t cactus_bench_c(cactus_context_handle_t handle, int pp, int tg, int pl, int nr);

// **HIGH PRIORITY: LoRA Adapter Support**
CACTUS_FFI_EXPORT int cactus_apply_lora_adapters_c(cactus_context_handle_t handle, const cactus_lora_adapters_c_t* adapters);
CACTUS_FFI_EXPORT void cactus_remove_lora_adapters_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT cactus_lora_adapters_c_t cactus_get_loaded_lora_adapters_c(cactus_context_handle_t handle);

// **HIGH PRIORITY: Chat Template Support**
CACTUS_FFI_EXPORT bool cactus_validate_chat_template_c(cactus_context_handle_t handle, bool use_jinja, const char* name);
CACTUS_FFI_EXPORT char* cactus_get_formatted_chat_c(cactus_context_handle_t handle, const char* messages, const char* chat_template);

// **ADVANCED: Chat with Jinja and Tools Support**
typedef struct {
    char* prompt;
    char* json_schema;
    char* tools;
    char* tool_choice;
    bool parallel_tool_calls;
} cactus_chat_result_c_t;

CACTUS_FFI_EXPORT cactus_chat_result_c_t cactus_get_formatted_chat_with_jinja_c(
    cactus_context_handle_t handle, 
    const char* messages,
    const char* chat_template,
    const char* json_schema,
    const char* tools,
    bool parallel_tool_calls,
    const char* tool_choice
);

// **HIGH PRIORITY: Context Management**
CACTUS_FFI_EXPORT void cactus_rewind_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT bool cactus_init_sampling_c(cactus_context_handle_t handle);

// **COMPLETION CONTROL**
CACTUS_FFI_EXPORT void cactus_begin_completion_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT void cactus_end_completion_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT void cactus_load_prompt_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT void cactus_load_prompt_with_media_c(cactus_context_handle_t handle, const char** media_paths, int media_count);

// **TOKEN PROCESSING**
CACTUS_FFI_EXPORT int cactus_do_completion_step_c(cactus_context_handle_t handle, char** token_text);
CACTUS_FFI_EXPORT size_t cactus_find_stopping_strings_c(cactus_context_handle_t handle, const char* text, size_t last_token_size, int stop_type);

// **HIGH PRIORITY: Model Information**
CACTUS_FFI_EXPORT int32_t cactus_get_n_ctx_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT int32_t cactus_get_n_embd_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT char* cactus_get_model_desc_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT int64_t cactus_get_model_size_c(cactus_context_handle_t handle);
CACTUS_FFI_EXPORT int64_t cactus_get_model_params_c(cactus_context_handle_t handle);

// Memory management functions
CACTUS_FFI_EXPORT void cactus_free_bench_result_members_c(cactus_bench_result_c_t* result);
CACTUS_FFI_EXPORT void cactus_free_lora_adapters_c(cactus_lora_adapters_c_t* adapters);
CACTUS_FFI_EXPORT void cactus_free_chat_result_members_c(cactus_chat_result_c_t* result);

#ifdef __cplusplus
}
#endif

#endif