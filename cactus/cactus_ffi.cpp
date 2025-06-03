#include "cactus_ffi.h"
#include "cactus.h"
#include "cactus_stt.h" // The C++ STT wrapper
#include "common.h"
#include "llama.h"
#include "llama-chat.h" // For llama_chat_message and llama_chat_apply_template
#include "json.hpp"     // For nlohmann::json

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring> 
#include <cstdlib> 
#include <sstream> 
#include <iostream> 
#include <cstdio> // For printf
#include <vector> // For std::vector in STT FFI
#include <new>    // For std::nothrow in STT FFI

// Bring symbols from the cactus namespace into the current scope for unqualified lookup.
// This will allow the 'log' call within the LOG_INFO macro to find 'cactus::log'.
using namespace cactus;

/**
 * @brief Converts a C-style array of strings to a C++ vector of strings.
 * @param arr The C-style array of strings.
 * @param count The number of strings in the array.
 * @return A std::vector<std::string> containing the strings.
 */
static std::vector<std::string> c_str_array_to_vector(const char** arr, int count) {
    std::vector<std::string> vec;
    if (arr != nullptr) {
        for (int i = 0; i < count; ++i) {
            if (arr[i] != nullptr) {
                vec.push_back(arr[i]);
            }
        }
    }
    return vec;
}


/**
 * @brief Safely duplicates a C string.
 * The caller is responsible for freeing the returned string using free().
 * @param str The std::string to duplicate.
 * @return A newly allocated C string, or nullptr if allocation fails. Returns an empty string if the input is empty.
 */
static char* safe_strdup(const std::string& str) {
    if (str.empty()) {
        char* empty_str = (char*)malloc(1);
        if (empty_str) empty_str[0] = '\0';
        return empty_str;
    }
    char* new_str = (char*)malloc(str.length() + 1);
    if (new_str) {
        std::strcpy(new_str, str.c_str());
    }
    return new_str; 
}

// Helper function to parse chat messages from JSON string to a temporary structure
// This temporary structure holds std::string to manage memory easily before converting to const char*
static std::vector<std::pair<std::string, std::string>> parse_raw_chat_messages_from_json(const std::string& json_str) {
    std::vector<std::pair<std::string, std::string>> raw_messages;
    try {
        nlohmann::json parsed_json = nlohmann::json::parse(json_str);
        if (parsed_json.is_array()) {
            for (const auto& item : parsed_json) {
                if (item.is_object() && item.contains("role") && item.contains("content")) {
                    raw_messages.push_back({item["role"].get<std::string>(), item["content"].get<std::string>()});
                } else {
                    LOG_WARNING("[FFI parse_raw_chat_messages_from_json] Skipping malformed message object in JSON array.");
                }
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERROR("[FFI parse_raw_chat_messages_from_json] JSON parse error: %s. Input: %s", e.what(), json_str.substr(0, 200).c_str());
    } catch (const std::exception& e) {
        LOG_ERROR("[FFI parse_raw_chat_messages_from_json] General exception during JSON parsing: %s", e.what());
    }
    return raw_messages;
}

extern "C" {

CACTUS_FFI_EXPORT cactus_init_params_c_t cactus_default_init_params_c() {
    common_params cpp_defaults; // C++ default construction
    cactus_init_params_c_t c_params = {}; // Zero-initialize C struct first

    // Manually translate relevant fields from cpp_defaults to c_params
    // Paths will be null by default, user must set them.
    // cpp_defaults.model.path and mmproj.path are std::string, convert to const char*
    // For strings, if the default is empty, it's fine, they'll be nullptr or empty string after strdup.
    // However, for most paths, the C++ default is an empty string, which is not useful for the C FFI user directly.
    // It's better to leave path char* as nullptr by default from zero-initialization.

    c_params.chat_template = nullptr; // Default to nullptr, model/library will pick internal default.

    c_params.n_ctx = cpp_defaults.n_ctx;
    c_params.n_batch = cpp_defaults.n_batch;
    c_params.n_ubatch = cpp_defaults.n_ubatch;
    c_params.n_gpu_layers = cpp_defaults.n_gpu_layers; // Key field for GPU offload default (-1)
    c_params.n_threads = cpp_defaults.cpuparams.n_threads;
    c_params.use_mmap = cpp_defaults.use_mmap;
    c_params.use_mlock = cpp_defaults.use_mlock;
    c_params.embedding = cpp_defaults.embedding;
    c_params.pooling_type = static_cast<int32_t>(cpp_defaults.pooling_type);
    c_params.embd_normalize = cpp_defaults.embd_normalize;
    c_params.flash_attn = cpp_defaults.flash_attn;
    
    // For cache_type_k and cache_type_v, common_params defaults to F16. 
    // We need to convert these enum values back to strings for the C API if needed,
    // or decide on a string default. For simplicity, let's default to nullptr, meaning
    // the underlying C++ layer will use its F16 default if these are not set by the user.
    c_params.cache_type_k = nullptr; 
    c_params.cache_type_v = nullptr;

    c_params.progress_callback = nullptr;
    c_params.warmup = cpp_defaults.warmup;
    c_params.mmproj_use_gpu = cpp_defaults.mmproj_use_gpu;
    c_params.main_gpu = cpp_defaults.main_gpu;

    return c_params;
}

/**
 * @brief Initializes a new cactus context with the given parameters.
 * This function loads the model and prepares it for use.
 * The caller is responsible for freeing the context using cactus_free_context_c.
 * @param params A pointer to the initialization parameters.
 * @return A handle to the created cactus context, or nullptr on failure.
 */
cactus_context_handle_t cactus_init_context_c(const cactus_init_params_c_t* params) {
    printf("[DEBUG] cactus_ffi.cpp: Entering cactus_init_context_c\n");
    if (!params) {
        printf("[ERROR] cactus_ffi.cpp: cactus_init_params_c_t is NULL\n");
        return nullptr;
    }

    cactus::cactus_context* context = nullptr;
    try {
        context = new cactus::cactus_context();
        printf("[DEBUG] cactus_ffi.cpp: cactus_context object created\n");

        common_params cpp_params;
        cpp_params.model.path = params->model_path;
        printf("[DEBUG] cactus_ffi.cpp: model_path = %s\n", params->model_path ? params->model_path : "NULL");
        if (params->mmproj_path && params->mmproj_path[0] != '\0') {
            cpp_params.mmproj.path = params->mmproj_path;
        } else {
            cpp_params.mmproj.path = "";
        }
        if (params->chat_template && params->chat_template[0] != '\0') {
            cpp_params.chat_template = params->chat_template;
        }
        cpp_params.n_ctx = params->n_ctx;
        cpp_params.n_batch = params->n_batch;
        cpp_params.n_ubatch = params->n_ubatch;
        cpp_params.n_gpu_layers = params->n_gpu_layers;
        cpp_params.cpuparams.n_threads = params->n_threads;
        cpp_params.use_mmap = params->use_mmap;
        cpp_params.use_mlock = params->use_mlock;
        cpp_params.embedding = params->embedding;
        cpp_params.pooling_type = static_cast<enum llama_pooling_type>(params->pooling_type);
        cpp_params.embd_normalize = params->embd_normalize;
        cpp_params.flash_attn = params->flash_attn;
        if (params->cache_type_k) {
             try {
                  cpp_params.cache_type_k = cactus::kv_cache_type_from_str(params->cache_type_k);
             } catch (const std::exception& e) {
                 std::cerr << "Warning: Invalid cache_type_k: " << params->cache_type_k << " Error: " << e.what() << std::endl;
                 delete context;
                 return nullptr;
             }
        }
        if (params->cache_type_v) {
            try {
                cpp_params.cache_type_v = cactus::kv_cache_type_from_str(params->cache_type_v);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid cache_type_v: " << params->cache_type_v << " Error: " << e.what() << std::endl;
                delete context;
                return nullptr;
            }
        }
        // TODO: Add translation for LoRA, RoPE params

        // Progress callback can be complex; this simple version might crash if the Dart function disappears
        if (params->progress_callback) {
            cpp_params.progress_callback = [](float progress, void* user_data) {
                auto callback = reinterpret_cast<void (*)(float)>(user_data);
                callback(progress);
                return true; 
            };
            cpp_params.progress_callback_user_data = reinterpret_cast<void*>(params->progress_callback);
        } else {
             cpp_params.progress_callback = nullptr;
             cpp_params.progress_callback_user_data = nullptr;
        }
        cpp_params.warmup = params->warmup;
        cpp_params.mmproj_use_gpu = params->mmproj_use_gpu;

        if (!context->loadModel(cpp_params)) {
            // loadModel logs errors internally
            delete context;
            printf("[ERROR] cactus_ffi.cpp: loadModel failed\n");
            return nullptr;
        }
        printf("[DEBUG] cactus_ffi.cpp: loadModel succeeded\n");

        return reinterpret_cast<cactus_context_handle_t>(context);

    } catch (const std::exception& e) {
        // It's good practice to log the exception message if possible
        printf("[ERROR] cactus_ffi.cpp: Exception during context initialization: %s\n", e.what());
        if (context) delete context;
        return nullptr;
    } catch (...) {
        printf("[ERROR] cactus_ffi.cpp: Unknown exception during context initialization\n");
        if (context) delete context;
        return nullptr;
    }
}


/**
 * @brief Frees a cactus context that was previously created with cactus_init_context_c.
 * @param handle The handle to the cactus context to free.
 */
void cactus_free_context_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        delete context;
    }
}


/**
 * @brief Performs text completion using the provided context and parameters.
 * This function can stream tokens back via a callback.
 * @param handle The handle to the cactus context.
 * @param params A pointer to the completion parameters.
 * @param result A pointer to a structure where the completion result will be stored.
 *               The caller is responsible for calling cactus_free_completion_result_members_c
 *               on the result structure to free allocated memory for text and stopping_word.
 * @return 0 on success, negative value on error.
 *         -1: Invalid arguments (handle, params, or result is null).
 *         -2: Failed to initialize sampling.
 *         -3: Exception occurred during completion.
 *         -4: Unknown exception occurred.
 */
CACTUS_FFI_EXPORT int cactus_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    cactus_completion_result_c_t* result // Output parameter
) {
    if (!handle || !params || !result) {
        // LOG_ERROR("[FFI] Invalid arguments to cactus_completion_c: handle, params, or result is null.");
        return CACTUS_COMPLETION_ERROR_INVALID_ARGUMENTS; // Or a specific error code for invalid args
    }

    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context) { // Should not happen if handle is valid from init, but good practice
        // LOG_ERROR("[FFI] Catastrophic: context handle is non-null but maps to null internal context.");
        return CACTUS_COMPLETION_ERROR_UNKNOWN; // Or a more specific critical error
    }

    // CRITICAL CHECK: Ensure the internal llama_context (ctx) is not null before proceeding.
    if (!context->ctx) {
        // LOG_ERROR("[FFI] cactus_completion_c called but internal LLaMA context (ctx) is NULL.");
        if (result) { // Still try to populate part of the result if possible
            result->text = nullptr;
            result->stopping_word = nullptr;
            result->tokens_predicted = 0;
            result->tokens_evaluated = 0;
            result->truncated = false;
            result->stopped_eos = false;
            result->stopped_word = false;
            result->stopped_limit = true; // Indicate failure due to limit/error
        }
        return CACTUS_COMPLETION_ERROR_NULL_CONTEXT;
    }

    // Initialize result fields to safe defaults
    memset(result, 0, sizeof(cactus_completion_result_c_t));

    try {
        context->rewind();

        // Set prompt for the context
        if (params->prompt) {
            context->params.prompt = params->prompt;
        } else {
            context->params.prompt = ""; // Ensure it's not uninitialized
        }

        // Set image path for the context
        if (params->image_path && params->image_path[0] != '\0') {
            context->params.image.clear();
            context->params.image.push_back(params->image_path);
        } else {
            context->params.image.clear();
        }

        // +++ Add Logging Here +++
        LOG_INFO("[FFI cactus_completion_c] After rewind and param setup:");
        LOG_INFO("[FFI cactus_completion_c]   Prompt: '%s'", context->params.prompt.c_str());
        if (!context->params.image.empty()) {
            LOG_INFO("[FFI cactus_completion_c]   Image[0]: '%s'", context->params.image[0].c_str());
        } else {
            LOG_INFO("[FFI cactus_completion_c]   Image: (empty)");
        }
        // +++ End Logging +++

        if (params->n_threads > 0) {
             context->params.cpuparams.n_threads = params->n_threads;
        }
        context->params.n_predict = params->n_predict;
        context->params.sampling.seed = params->seed;
        context->params.sampling.temp = params->temperature;
        context->params.sampling.top_k = params->top_k;
        context->params.sampling.top_p = params->top_p;
        context->params.sampling.min_p = params->min_p;
        context->params.sampling.typ_p = params->typical_p;
        context->params.sampling.penalty_last_n = params->penalty_last_n;
        context->params.sampling.penalty_repeat = params->penalty_repeat;
        context->params.sampling.penalty_freq = params->penalty_freq;
        context->params.sampling.penalty_present = params->penalty_present;
        context->params.sampling.mirostat = params->mirostat;
        context->params.sampling.mirostat_tau = params->mirostat_tau;
        context->params.sampling.mirostat_eta = params->mirostat_eta;
        context->params.sampling.ignore_eos = params->ignore_eos;
        context->params.sampling.n_probs = params->n_probs;
        context->params.antiprompt = c_str_array_to_vector(params->stop_sequences, params->stop_sequence_count);
        if (params->grammar) {
             context->params.sampling.grammar = params->grammar;
        }

        if (!context->initSampling()) {
            return -2; 
        }
        context->beginCompletion();
        context->loadPrompt();

        // --- Streaming loop --- 
        int64_t generation_start_time_us = llama_time_us(); // Record start time
        context->generation_time_us = 0; // Reset time for current completion

        while (context->has_next_token && !context->is_interrupted) {
            const cactus::completion_token_output token_with_probs = context->doCompletion();

            if (token_with_probs.tok == -1 && !context->has_next_token) {
                 break;
            }
            
            if (token_with_probs.tok != -1 && params->token_callback) {
                // Format token data (simple example: just the text)
                // A more complex implementation could create JSON here
                std::string token_text = common_token_to_piece(context->ctx, token_with_probs.tok);
                
                // Call the Dart callback
                bool continue_completion = params->token_callback(token_text.c_str());
                if (!continue_completion) {
                    context->is_interrupted = true; 
                    break;
                }
            }
        }

        int64_t generation_end_time_us = llama_time_us(); // Record end time
        context->generation_time_us = generation_end_time_us - generation_start_time_us;

        // --- Fill final result struct --- 
        result->text = safe_strdup(context->generated_text);
        result->tokens_predicted = context->num_tokens_predicted;
        result->tokens_evaluated = context->num_prompt_tokens;
        result->truncated = context->truncated;
        result->stopped_eos = context->stopped_eos;
        result->stopped_word = context->stopped_word;
        result->stopped_limit = context->stopped_limit;
        result->stopping_word = safe_strdup(context->stopping_word);
        result->generation_time_us = context->generation_time_us;
        // TODO: Populate timings 

        context->is_predicting = false;
        return 0; // Success

    } catch (const std::exception& e) {
        // Log error
        std::cerr << "Error during completion: " << e.what() << std::endl;

        // Cleanup state
        context->is_predicting = false;
        context->is_interrupted = true; 
        return -3; 

    } catch (...) {
        // Log error
        context->is_predicting = false;
        context->is_interrupted = true;
        return -4; // Unknown exception
    }
}


/**
 * @brief Stops an ongoing completion process.
 * Sets an interruption flag in the context.
 * @param handle The handle to the cactus context.
 */
void cactus_stop_completion_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        context->is_interrupted = true;
    }
}


/**
 * @brief Tokenizes a given text using the context's tokenizer.
 * The caller is responsible for freeing the returned token array using cactus_free_token_array_c.
 * @param handle The handle to the cactus context.
 * @param text The C string to tokenize.
 * @return A cactus_token_array_c_t structure containing the tokens and their count.
 *         The 'tokens' field will be nullptr and 'count' 0 on failure or if input is invalid.
 */
cactus_token_array_c_t cactus_tokenize_c(cactus_context_handle_t handle, const char* text) {
    cactus_token_array_c_t result = {nullptr, 0};
    if (!handle || !text) {
        return result;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx) { // Need the llama_context
        return result;
    }

    try {
        std::vector<llama_token> tokens_vec = ::common_tokenize(context->ctx, text, false, true);
        if (!tokens_vec.empty()) {
            result.count = tokens_vec.size();
            result.tokens = (int32_t*)malloc(result.count * sizeof(int32_t));
            if (result.tokens) {
                // Copy data
                std::copy(tokens_vec.begin(), tokens_vec.end(), result.tokens);
            } else {
                result.count = 0; // Malloc failed
            }
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error during tokenization: " << e.what() << std::endl;
        return {nullptr, 0};
    } catch (...) {
        std::cerr << "Unknown error during tokenization." << std::endl;
        return {nullptr, 0};
    }
}

/**
 * @brief Detokenizes an array of tokens into a string.
 * The caller is responsible for freeing the returned C string using cactus_free_string_c.
 * @param handle The handle to the cactus context.
 * @param tokens A pointer to an array of token IDs.
 * @param count The number of tokens in the array.
 * @return A newly allocated C string representing the detokenized text.
 *         Returns an empty string on failure or if input is invalid.
 */
char* cactus_detokenize_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count) {
    if (!handle || !tokens || count <= 0) {
        return safe_strdup(""); // Return empty string
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
     if (!context->ctx) {
        return safe_strdup("");
    }

    try {
        std::vector<llama_token> tokens_vec(tokens, tokens + count);
        std::string text = cactus::tokens_to_str(context->ctx, tokens_vec.cbegin(), tokens_vec.cend());
        // Print the intermediate C++ string for debugging
        std::cout << "[DEBUG cactus_detokenize_c] Intermediate std::string: [" << text << "]" << std::endl;
        return safe_strdup(text);
    } catch (const std::exception& e) {
        std::cerr << "Error during detokenization: " << e.what() << std::endl;
        return safe_strdup("");
    } catch (...) {
        std::cerr << "Unknown error during detokenization." << std::endl;
        return safe_strdup("");
    }
}


/**
 * @brief Generates an embedding for the given text.
 * Embedding mode must be enabled during context initialization.
 * The caller is responsible for freeing the returned float array using cactus_free_float_array_c.
 * @param handle The handle to the cactus context.
 * @param text The C string for which to generate the embedding.
 * @return A cactus_float_array_c_t structure containing the embedding values and their count.
 *         The 'values' field will be nullptr and 'count' 0 on failure or if embedding is not enabled.
 */
cactus_float_array_c_t cactus_embedding_c(cactus_context_handle_t handle, const char* text) {
    cactus_float_array_c_t result = {nullptr, 0};
     if (!handle || !text) {
        return result;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx || !context->params.embedding) { 
        std::cerr << "Error: Embedding mode not enabled or context not initialized." << std::endl;
        return result;
    }

    try {
        context->rewind();
        context->params.prompt = text;
        context->params.n_predict = 0; 

        if (!context->initSampling()) { return result; }
        context->beginCompletion();
        context->loadPrompt();
        context->doCompletion(); 

        common_params dummy_embd_params;
        dummy_embd_params.embd_normalize = context->params.embd_normalize;

        std::vector<float> embedding_vec = context->getEmbedding(dummy_embd_params);

        if (!embedding_vec.empty()) {
            result.count = embedding_vec.size();
            result.values = (float*)malloc(result.count * sizeof(float));
            if (result.values) {
                std::copy(embedding_vec.begin(), embedding_vec.end(), result.values);
            } else {
                result.count = 0; 
            }
        }
        context->is_predicting = false;
        return result;

    } catch (const std::exception& e) {
        std::cerr << "Error during embedding generation: " << e.what() << std::endl;
        context->is_predicting = false;
        return {nullptr, 0};
    } catch (...) {
        std::cerr << "Unknown error during embedding generation." << std::endl;
        context->is_predicting = false;
        return {nullptr, 0};
    }
}



/**
 * @brief Frees a C string that was allocated by one of the cactus_ffi functions.
 * @param str The C string to free.
 */
void cactus_free_string_c(char* str) {
    if (str) {
        free(str);
    }
}

/**
 * @brief Frees a token array structure (the 'tokens' field) allocated by cactus_tokenize_c.
 * @param arr The token array to free.
 */
void cactus_free_token_array_c(cactus_token_array_c_t arr) {
    if (arr.tokens) {
        free(arr.tokens);
    }
    // No need to zero out arr, caller owns it
}

/**
 * @brief Frees a float array structure (the 'values' field) allocated by cactus_embedding_c.
 * @param arr The float array to free.
 */
void cactus_free_float_array_c(cactus_float_array_c_t arr) {
    if (arr.values) {
        free(arr.values);
    }
}

/**
 * @brief Frees the members of a cactus_completion_result_c_t structure that were dynamically allocated.
 * Specifically, this frees the 'text' and 'stopping_word' C strings.
 * @param result A pointer to the completion result structure whose members are to be freed.
 */
void cactus_free_completion_result_members_c(cactus_completion_result_c_t* result) {
    if (result) {
        cactus_free_string_c(result->text);
        cactus_free_string_c(result->stopping_word);
        result->text = nullptr; // Prevent double free
        result->stopping_word = nullptr;
    }
}


/**
 * @brief Loads a vocoder model into the given cactus context.
 * @param handle The handle to the cactus context.
 * @param params A pointer to the vocoder loading parameters.
 * @return 0 on success, negative value on error.
 *         -1: Invalid arguments.
 *         -2: Vocoder model loading failed.
 *         -3: Exception occurred.
 *         -4: Unknown exception occurred.
 */
int cactus_load_vocoder_c(
    cactus_context_handle_t handle,
    const cactus_vocoder_load_params_c_t* params
) {
    if (!handle || !params || !params->model_params.path) {
        std::cerr << "Error: Invalid arguments to cactus_load_vocoder_c." << std::endl;
        return -1; // Invalid arguments
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);

    try {
        common_params_vocoder vocoder_cpp_params;
        vocoder_cpp_params.model.path = params->model_params.path;
        
        if (params->speaker_file) {
            vocoder_cpp_params.speaker_file = params->speaker_file;
        }
        vocoder_cpp_params.use_guide_tokens = params->use_guide_tokens;

        if (!context->loadVocoderModel(vocoder_cpp_params)) {
            std::cerr << "Error: Failed to load vocoder model." << std::endl;
            return -2; // Vocoder model loading failed
        }
        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "Exception in cactus_load_vocoder_c: " << e.what() << std::endl;
        return -3; // Exception occurred
    } catch (...) {
        std::cerr << "Unknown exception in cactus_load_vocoder_c." << std::endl;
        return -4; // Unknown exception
    }
}


/**
 * @brief Synthesizes speech from the given text input and saves it to a WAV file.
 * A vocoder model must be loaded first using cactus_load_vocoder_c.
 * @param handle The handle to the cactus context.
 * @param params A pointer to the speech synthesis parameters.
 * @return 0 on success, negative value on error.
 *         -1: Invalid arguments.
 *         -2: Speech synthesis failed.
 *         -3: Exception occurred.
 *         -4: Unknown exception occurred.
 */
int cactus_synthesize_speech_c(
    cactus_context_handle_t handle,
    const cactus_synthesize_speech_params_c_t* params
) {
    if (!handle || !params || !params->text_input || !params->output_wav_path) {
        std::cerr << "Error: Invalid arguments to cactus_synthesize_speech_c." << std::endl;
        return -1; // Invalid arguments
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);

    try {
        std::string text_input_str = params->text_input;
        std::string output_wav_path_str = params->output_wav_path;
        std::string speaker_id_str = params->speaker_id ? params->speaker_id : "";

        if (!context->synthesizeSpeech(text_input_str, output_wav_path_str, speaker_id_str)) {
            std::cerr << "Error: Speech synthesis failed." << std::endl;
            return -2; // Synthesis failed
        }
        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "Exception in cactus_synthesize_speech_c: " << e.what() << std::endl;
        return -3; // Exception occurred
    } catch (...) {
        std::cerr << "Unknown exception in cactus_synthesize_speech_c." << std::endl;
        return -4; // Unknown exception
    }
}

/**
 * @brief Formats a list of chat messages using the appropriate chat template.  
 * @param handle The handle to the cactus context.
 * @param messages_json A JSON string representing an array of chat messages (e.g., [{"role": "user", "content": "Hello"}]).
 * @param override_chat_template An optional chat template string to use. If NULL or empty,
 *                               the template from context initialization or the model's default will be used.
 * @param image_path An optional image path to include in the formatted prompt.
 * @return A newly allocated C string containing the fully formatted prompt. Caller must free using cactus_free_string_c.
 *         Returns an empty string on failure.
 */
CACTUS_FFI_EXPORT char* cactus_get_formatted_chat_c(
    cactus_context_handle_t handle,
    const char* messages_json_c_str,
    const char* override_chat_template_c_str,
    const char* image_path_c_str
) {
    if (!handle || !messages_json_c_str) {
        LOG_ERROR("[FFI cactus_get_formatted_chat_c] Invalid arguments: handle or messages_json is null.");
        return nullptr; 
    }

    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context) { 
        LOG_ERROR("[FFI cactus_get_formatted_chat_c] Context is null.");
        return nullptr;
    }

    std::string messages_json_std_str = messages_json_c_str;
    std::vector<std::pair<std::string, std::string>> raw_messages = parse_raw_chat_messages_from_json(messages_json_std_str);

    if (raw_messages.empty() && !messages_json_std_str.empty() && messages_json_std_str != "[]") {
        LOG_ERROR("[FFI cactus_get_formatted_chat_c] Failed to parse any messages from non-empty JSON. Returning original JSON.");
        return safe_strdup(messages_json_std_str);
    }

    std::string image_path_std_str = image_path_c_str ? image_path_c_str : "";

    std::vector<std::string> string_data_holder;
    string_data_holder.reserve(raw_messages.size() * 2 + (image_path_std_str.empty() ? 0 : 1)); // +1 for potentially modified content
    
    std::vector<llama_chat_message> llama_messages_objs;
    llama_messages_objs.reserve(raw_messages.size());

    for (size_t i = 0; i < raw_messages.size(); ++i) {
        const auto& raw_msg_pair = raw_messages[i];
        
        string_data_holder.push_back(raw_msg_pair.first); // role string
        std::string content_str = raw_msg_pair.second;   // content string

        // If it's the last message, user role, and image path is present, modify content string
        if (i == raw_messages.size() - 1 && strcmp(string_data_holder.back().c_str(), "user") == 0 && !image_path_std_str.empty()) {
            if (content_str.find("<__image__>") == std::string::npos) {
                content_str = "<__image__>\n" + content_str;
                LOG_INFO("[FFI cactus_get_formatted_chat_c] Prepended <__image__> tag to last user message content string.");
            }
        }
        string_data_holder.push_back(content_str); // (potentially modified) content string

        llama_chat_message l_msg_obj;
        l_msg_obj.role = string_data_holder[string_data_holder.size() - 2].c_str();
        l_msg_obj.content = string_data_holder.back().c_str();
        llama_messages_objs.push_back(l_msg_obj);
    }

    std::string chat_template_str;
    if (override_chat_template_c_str && override_chat_template_c_str[0] != '\0') {
        chat_template_str = override_chat_template_c_str;
        LOG_INFO("[FFI cactus_get_formatted_chat_c] Using override template: '%s'", chat_template_str.c_str());
    } else if (!context->params.chat_template.empty()) {
        chat_template_str = context->params.chat_template;
        LOG_INFO("[FFI cactus_get_formatted_chat_c] Using context template: '%s'", chat_template_str.c_str());
    } else {
        LOG_INFO("[FFI cactus_get_formatted_chat_c] No chat template override or init param; will pass nullptr to llama_chat_apply_template (model default).");
        // chat_template_str remains empty, c_template_for_api will be nullptr
    }

    std::string formatted_prompt_std_str;
    try {
        // Max buffer size for the formatted prompt. 
        // Consider context->n_ctx (max tokens) * avg chars/token + template overhead.
        size_t buffer_size = context->n_ctx * 10 + 1024; // Increased multiplier and base overhead
        if (buffer_size < 4096) buffer_size = 4096;
        std::vector<char> buf(buffer_size);

        const char* c_template_for_api = chat_template_str.empty() ? nullptr : chat_template_str.c_str();
        
        LOG_INFO("[FFI cactus_get_formatted_chat_c] Applying template. C Template string ptr: %p, Messages count: %zu", 
                 (void*)c_template_for_api, llama_messages_objs.size());
        if(c_template_for_api) {
             LOG_INFO("[FFI cactus_get_formatted_chat_c] Template string content: '%s'", c_template_for_api);
        }

        int n_written = llama_chat_apply_template(
            c_template_for_api,
            llama_messages_objs.empty() ? nullptr : llama_messages_objs.data(),
            llama_messages_objs.size(),
            true, // add_assistant_role
            buf.data(),
            static_cast<int32_t>(buf.size()) 
        );

        if (n_written < 0) {
            LOG_ERROR("[FFI cactus_get_formatted_chat_c] llama_chat_apply_template returned error code: %d. This usually means the template is malformed or incompatible.", n_written);
            nlohmann::json messages_json_array = nlohmann::json::array();
            for(const auto& msg_obj : llama_messages_objs) {
                messages_json_array.push_back({{"role", msg_obj.role ? msg_obj.role : ""}, {"content", msg_obj.content ? msg_obj.content : ""}});
            }
            formatted_prompt_std_str = messages_json_array.dump();
        } else if ((size_t)n_written >= buf.size()) { 
            LOG_ERROR("[FFI cactus_get_formatted_chat_c] llama_chat_apply_template output truncated or buffer too small. n_written: %d, buf_size: %zu. Template might be too verbose or messages too long.", n_written, buf.size());
            nlohmann::json messages_json_array = nlohmann::json::array();
            for(const auto& msg_obj : llama_messages_objs) {
                messages_json_array.push_back({{"role", msg_obj.role ? msg_obj.role : ""}, {"content", msg_obj.content ? msg_obj.content : ""}});
            }
            formatted_prompt_std_str = messages_json_array.dump();
        } else {
            formatted_prompt_std_str.assign(buf.data(), n_written);
            LOG_INFO("[FFI cactus_get_formatted_chat_c] Applied chat template successfully. Result starts with: '%s' (Length: %zu)", formatted_prompt_std_str.substr(0,200).c_str(), formatted_prompt_std_str.length());
        }

    } catch (const std::exception& e) {
        LOG_ERROR("[FFI cactus_get_formatted_chat_c] Exception during llama_chat_apply_template or surrounding logic: %s. Falling back to JSON.", e.what());
        nlohmann::json messages_json_array = nlohmann::json::array();
        for(const auto& msg_obj : llama_messages_objs) {
             messages_json_array.push_back({{"role", msg_obj.role ? msg_obj.role : ""}, {"content", msg_obj.content ? msg_obj.content : ""}});
        }
        formatted_prompt_std_str = messages_json_array.dump();
    }
    
    if (formatted_prompt_std_str.empty() && !messages_json_std_str.empty()){
        LOG_WARNING("[FFI cactus_get_formatted_chat_c] Formatted prompt is empty despite processing, returning original JSON string as fallback.");
        return safe_strdup(messages_json_std_str);
    }
    
    return safe_strdup(formatted_prompt_std_str);
}

CACTUS_FFI_EXPORT void cactus_free_formatted_chat_result_members_c(cactus_formatted_chat_result_c_t* result) {
    if (result) {
        if (result->prompt) free(result->prompt);
        if (result->grammar) free(result->grammar);
        // Initialize to safe defaults to prevent double free if called again
        result->prompt = nullptr;
        result->grammar = nullptr;
    }
}

// +++ Speech-to-Text (STT) FFI Implementations +++

// Define the opaque struct declared in the header
struct cactus_stt_context {
    cactus::STT* instance;
};

CACTUS_FFI_EXPORT cactus_stt_context_t* cactus_stt_init(const char* model_path, const char* language) {
    if (!model_path || !language) {
        // Consider logging an error here if a logging mechanism is available
        return nullptr;
    }
    cactus_stt_context_t* ctx = new (std::nothrow) cactus_stt_context_t;
    if (!ctx) {
        // Allocation failure for the context struct
        return nullptr;
    }
    ctx->instance = new (std::nothrow) cactus::STT();
    if (!ctx->instance) {
        // Allocation failure for the STT instance
        delete ctx;
        return nullptr;
    }
    // Assuming "en" and use_gpu=true as defaults if not passed or if complex logic is avoided in FFI layer
    // The prompt implies language is a parameter, so we use it. GPU usage is true by default in C++ impl.
    if (!ctx->instance->initialize(model_path, language)) {
        delete ctx->instance;
        delete ctx;
        return nullptr;
    }
    return ctx;
}

CACTUS_FFI_EXPORT bool cactus_stt_process_audio(cactus_stt_context_t* ctx, const float* samples, uint32_t num_samples) {
    if (!ctx || !ctx->instance || !samples) {
        return false;
    }
    if (!ctx->instance->isInitialized()) {
        // Or handle error, e.g., log "STT context not initialized"
        return false;
    }
    // Create a std::vector from the raw pointer and size.
    // This involves a copy, which might be a performance consideration for very large chunks.
    // Alternative: Modify cactus::STT::processAudio to accept const float* and size directly.
    std::vector<float> audio_vector(samples, samples + num_samples);
    return ctx->instance->processAudio(audio_vector);
}

CACTUS_FFI_EXPORT char* cactus_stt_get_transcription(cactus_stt_context_t* ctx) {
    if (!ctx || !ctx->instance || !ctx->instance->isInitialized()) {
        return nullptr;
    }
    std::string transcription = ctx->instance->getTranscription();
    // Caller must free this string using cactus_free_string_c.
    // strdup will return nullptr if transcription.c_str() is null (not possible for std::string)
    // or if memory allocation fails.
    // If transcription is empty, strdup("") correctly returns a pointer to a null-terminated empty string.
    char* c_transcription = strdup(transcription.c_str());
    return c_transcription;
}

CACTUS_FFI_EXPORT void cactus_stt_free(cactus_stt_context_t* ctx) {
    if (ctx) {
        if (ctx->instance) {
            delete ctx->instance; // This will call STT::~STT() which calls cleanup()
        }
        delete ctx;
    }
}

CACTUS_FFI_EXPORT void cactus_stt_set_user_vocabulary(cactus_stt_context_t* ctx, const char* vocabulary) {
    if (!ctx || !ctx->instance) {
        // Optional: Log error if context or instance is null
        // fprintf(stderr, "cactus_stt_set_user_vocabulary: STT context or instance is null.\n");
        return;
    }
    if (!vocabulary) {
        // Treat null vocabulary as a request to clear/reset the vocabulary
        ctx->instance->setUserVocabulary("");
        return;
    }
    ctx->instance->setUserVocabulary(vocabulary);
}

// --- End Speech-to-Text (STT) FFI Implementations ---

// +++ Benchmarking FFI Functions +++
/**
 * @brief Benchmarks the model performance using the C FFI.
 * The caller is responsible for freeing the returned JSON string using cactus_free_string_c.
 *
 * @param handle Handle to the cactus context.
 * @param pp Prompt processing tokens.
 * @param tg Text generation iterations.
 * @param pl Parallel tokens to predict.
 * @param nr Number of repetitions.
 * @return JSON string with benchmark results, or nullptr on error.
 */
CACTUS_FFI_EXPORT char* cactus_bench_c(
    cactus_context_handle_t handle,
    int32_t pp,
    int32_t tg,
    int32_t pl,
    int32_t nr
) {
    if (!handle) {
        LOG_ERROR("[FFI cactus_bench_c] Invalid context handle.");
        return nullptr;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx) {
        LOG_ERROR("[FFI cactus_bench_c] Internal LLaMA context (ctx) is NULL.");
        return nullptr;
    }

    try {
        std::string result_str = context->bench(pp, tg, pl, nr);
        if (result_str.empty() || result_str == "[]") {
             LOG_WARNING("[FFI cactus_bench_c] Benchmarking returned empty or '[]' result.");
            return safe_strdup("[]"); 
        }
        return safe_strdup(result_str.c_str());
    } catch (const std::exception& e) {
        LOG_ERROR("[FFI cactus_bench_c] Exception: %s", e.what());
        return nullptr;
    } catch (...) {
        LOG_ERROR("[FFI cactus_bench_c] Unknown exception.");
        return nullptr;
    }
}
// --- End Benchmarking FFI Functions ---


// +++ LoRA Adapter Management FFI Functions +++
// CACTUS_FFI_EXPORT int cactus_apply_lora_adapters_c(...); // Placeholder for actual LoRA functions
// ... other LoRA related C FFI functions ...
// --- End LoRA Adapter Management FFI Functions ---

} // extern "C" 