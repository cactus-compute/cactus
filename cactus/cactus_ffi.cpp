#include "cactus_ffi.h"
#include "cactus.h"
#include "common.h"
#include "llama.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iostream>

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


extern "C" {

cactus_context_handle_t cactus_init_context_c(const cactus_init_params_c_t* params) {
    if (!params || !params->model_path) {
        return nullptr;
    }

    cactus::cactus_context* context = nullptr;
    try {
        context = new cactus::cactus_context();

        common_params cpp_params;
        cpp_params.model.path = params->model_path;
        if (params->chat_template) {
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



        if (!context->loadModel(cpp_params)) {
            delete context;
            return nullptr;
        }

        return reinterpret_cast<cactus_context_handle_t>(context);

    } catch (const std::exception& e) {
        std::cerr << "Error initializing context: " << e.what() << std::endl;
        if (context) delete context;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error initializing context." << std::endl;
        if (context) delete context;
        return nullptr;
    }
}

void cactus_free_context_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        delete context;
    }
}

int cactus_completion_c(
    cactus_context_handle_t handle,
    const cactus_completion_params_c_t* params,
    cactus_completion_result_c_t* result // Output parameter
) {
    if (!handle || !params || !params->prompt || !result) {
        return -1; // Invalid arguments
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);

    memset(result, 0, sizeof(cactus_completion_result_c_t));

    try {
        context->rewind();

        context->params.prompt = params->prompt;
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

        while (context->has_next_token && !context->is_interrupted) {
            const cactus::completion_token_output token_with_probs = context->doCompletion();

            if (token_with_probs.tok == -1 && !context->has_next_token) {
                 break;
            }
            
            if (token_with_probs.tok != -1 && params->token_callback) {
                std::string token_text = common_token_to_piece(context->ctx, token_with_probs.tok);
                
                bool continue_completion = params->token_callback(token_text.c_str());
                if (!continue_completion) {
                    context->is_interrupted = true;
                    break;
                }
            }
        }

        result->text = safe_strdup(context->generated_text);
        result->tokens_predicted = context->num_tokens_predicted;
        result->tokens_evaluated = context->num_prompt_tokens;
        result->truncated = context->truncated;
        result->stopped_eos = context->stopped_eos;
        result->stopped_word = context->stopped_word;
        result->stopped_limit = context->stopped_limit;
        result->stopping_word = safe_strdup(context->stopping_word);

        context->is_predicting = false;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error during completion: " << e.what() << std::endl;
        context->is_predicting = false;
        context->is_interrupted = true;
        return -3;
    } catch (...) {
        context->is_predicting = false;
        context->is_interrupted = true;
        return -4;
    }
}

void cactus_stop_completion_c(cactus_context_handle_t handle) {
    if (handle) {
        cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
        context->is_interrupted = true;
    }
}

cactus_token_array_c_t cactus_tokenize_c(cactus_context_handle_t handle, const char* text) {
    cactus_token_array_c_t result = {nullptr, 0};
    if (!handle || !text) {
        return result;
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx) {
        return result;
    }

    try {
        std::vector<llama_token> tokens_vec = ::common_tokenize(context->ctx, text, false, true);
        if (!tokens_vec.empty()) {
            result.count = tokens_vec.size();
            result.tokens = (int32_t*)malloc(result.count * sizeof(int32_t));
            if (result.tokens) {
                std::copy(tokens_vec.begin(), tokens_vec.end(), result.tokens);
            } else {
                result.count = 0;
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

char* cactus_detokenize_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count) {
    if (!handle || !tokens || count <= 0) {
        return safe_strdup("");
    }
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
     if (!context->ctx) {
        return safe_strdup("");
    }

    try {
        std::vector<llama_token> tokens_vec(tokens, tokens + count);
        std::string text = cactus::tokens_to_str(context->ctx, tokens_vec.cbegin(), tokens_vec.cend());
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

void cactus_free_string_c(char* str) {
    if (str) {
        free(str);
    }
}

void cactus_free_token_array_c(cactus_token_array_c_t arr) {
    if (arr.tokens) {
        free(arr.tokens);
    }
}

void cactus_free_float_array_c(cactus_float_array_c_t arr) {
    if (arr.values) {
        free(arr.values);
    }
}

void cactus_free_completion_result_members_c(cactus_completion_result_c_t* result) {
    if (result) {
        cactus_free_string_c(result->text);
        cactus_free_string_c(result->stopping_word);
        result->text = nullptr;
        result->stopping_word = nullptr;
    }
}

cactus_tokenize_result_c_t cactus_tokenize_with_media_c(cactus_context_handle_t handle, const char* text, const char** media_paths, int media_count) {
    cactus_tokenize_result_c_t result = {0};
    if (!handle || !text) {
        return result;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    if (!context->ctx) {
        return result;
    }

    try {
        std::vector<std::string> media_vec;
        if (media_paths && media_count > 0) {
            for (int i = 0; i < media_count; ++i) {
                if (media_paths[i]) {
                    media_vec.push_back(media_paths[i]);
                }
            }
        }

        cactus::cactus_tokenize_result tokenize_result = context->tokenize(text, media_vec);

        result.tokens.count = tokenize_result.tokens.size();
        if (result.tokens.count > 0) {
            result.tokens.tokens = (int32_t*)malloc(result.tokens.count * sizeof(int32_t));
            if (result.tokens.tokens) {
                std::copy(tokenize_result.tokens.begin(), tokenize_result.tokens.end(), result.tokens.tokens);
            } else {
                result.tokens.count = 0;
            }
        }

        result.has_media = tokenize_result.has_media;

        if (!tokenize_result.bitmap_hashes.empty()) {
            result.bitmap_hash_count = tokenize_result.bitmap_hashes.size();
            result.bitmap_hashes = (char**)malloc(result.bitmap_hash_count * sizeof(char*));
            if (result.bitmap_hashes) {
                for (int i = 0; i < result.bitmap_hash_count; ++i) {
                    result.bitmap_hashes[i] = safe_strdup(tokenize_result.bitmap_hashes[i]);
                }
            } else {
                result.bitmap_hash_count = 0;
            }
        }

        if (!tokenize_result.chunk_pos.empty()) {
            result.chunk_position_count = tokenize_result.chunk_pos.size();
            result.chunk_positions = (size_t*)malloc(result.chunk_position_count * sizeof(size_t));
            if (result.chunk_positions) {
                std::copy(tokenize_result.chunk_pos.begin(), tokenize_result.chunk_pos.end(), result.chunk_positions);
            } else {
                result.chunk_position_count = 0;
            }
        }

        if (!tokenize_result.chunk_pos_media.empty()) {
            result.chunk_position_media_count = tokenize_result.chunk_pos_media.size();
            result.chunk_positions_media = (size_t*)malloc(result.chunk_position_media_count * sizeof(size_t));
            if (result.chunk_positions_media) {
                std::copy(tokenize_result.chunk_pos_media.begin(), tokenize_result.chunk_pos_media.end(), result.chunk_positions_media);
            } else {
                result.chunk_position_media_count = 0;
            }
        }

        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error during enhanced tokenization: " << e.what() << std::endl;
        return {0};
    } catch (...) {
        std::cerr << "Unknown error during enhanced tokenization." << std::endl;
        return {0};
    }
}

void cactus_free_tokenize_result_c(cactus_tokenize_result_c_t* result) {
    if (result) {
        cactus_free_token_array_c(result->tokens);
        
        if (result->bitmap_hashes) {
            for (int i = 0; i < result->bitmap_hash_count; ++i) {
                cactus_free_string_c(result->bitmap_hashes[i]);
            }
            free(result->bitmap_hashes);
            result->bitmap_hashes = nullptr;
        }
        
        if (result->chunk_positions) {
            free(result->chunk_positions);
            result->chunk_positions = nullptr;
        }
        
        if (result->chunk_positions_media) {
            free(result->chunk_positions_media);
            result->chunk_positions_media = nullptr;
        }
        
        result->bitmap_hash_count = 0;
        result->chunk_position_count = 0;
        result->chunk_position_media_count = 0;
        result->has_media = false;
    }
}

void cactus_set_guide_tokens_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count) {
    if (!handle || !tokens || count <= 0) {
        return;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::vector<llama_token> guide_tokens_vec(tokens, tokens + count);
        context->setGuideTokens(guide_tokens_vec);
    } catch (const std::exception& e) {
        std::cerr << "Error setting guide tokens: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error setting guide tokens." << std::endl;
    }
}

int cactus_init_multimodal_c(cactus_context_handle_t handle, const char* mmproj_path, bool use_gpu) {
    if (!handle || !mmproj_path) {
        return -1;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        bool success = context->initMultimodal(mmproj_path, use_gpu);
        return success ? 0 : -2;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing multimodal: " << e.what() << std::endl;
        return -3;
    } catch (...) {
        std::cerr << "Unknown error initializing multimodal." << std::endl;
        return -4;
    }
}

bool cactus_is_multimodal_enabled_c(cactus_context_handle_t handle) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->isMultimodalEnabled();
    } catch (...) {
        return false;
    }
}

bool cactus_supports_vision_c(cactus_context_handle_t handle) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->isMultimodalSupportVision();
    } catch (...) {
        return false;
    }
}

bool cactus_supports_audio_c(cactus_context_handle_t handle) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->isMultimodalSupportAudio();
    } catch (...) {
        return false;
    }
}

void cactus_release_multimodal_c(cactus_context_handle_t handle) {
    if (!handle) {
        return;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        context->releaseMultimodal();
    } catch (const std::exception& e) {
        std::cerr << "Error releasing multimodal: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error releasing multimodal." << std::endl;
    }
}

int cactus_init_vocoder_c(cactus_context_handle_t handle, const char* vocoder_model_path) {
    if (!handle || !vocoder_model_path) {
        return -1;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        bool success = context->initVocoder(vocoder_model_path);
        return success ? 0 : -2;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing vocoder: " << e.what() << std::endl;
        return -3;
    } catch (...) {
        std::cerr << "Unknown error initializing vocoder." << std::endl;
        return -4;
    }
}

bool cactus_is_vocoder_enabled_c(cactus_context_handle_t handle) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->isVocoderEnabled();
    } catch (...) {
        return false;
    }
}

int cactus_get_tts_type_c(cactus_context_handle_t handle) {
    if (!handle) {
        return -1;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->getTTSType();
    } catch (...) {
        return -1;
    }
}

char* cactus_get_formatted_audio_completion_c(cactus_context_handle_t handle, const char* speaker_json_str, const char* text_to_speak) {
    if (!handle || !text_to_speak) {
        return safe_strdup("");
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::string speaker_str = speaker_json_str ? speaker_json_str : "";
        std::string result = context->getFormattedAudioCompletion(speaker_str, text_to_speak);
        return safe_strdup(result);
    } catch (const std::exception& e) {
        std::cerr << "Error getting formatted audio completion: " << e.what() << std::endl;
        return safe_strdup("");
    } catch (...) {
        std::cerr << "Unknown error getting formatted audio completion." << std::endl;
        return safe_strdup("");
    }
}

cactus_token_array_c_t cactus_get_audio_guide_tokens_c(cactus_context_handle_t handle, const char* text_to_speak) {
    cactus_token_array_c_t result = {nullptr, 0};
    if (!handle || !text_to_speak) {
        return result;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::vector<llama_token> tokens = context->getAudioCompletionGuideTokens(text_to_speak);
        if (!tokens.empty()) {
            result.count = tokens.size();
            result.tokens = (int32_t*)malloc(result.count * sizeof(int32_t));
            if (result.tokens) {
                std::copy(tokens.begin(), tokens.end(), result.tokens);
            } else {
                result.count = 0;
            }
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error getting audio guide tokens: " << e.what() << std::endl;
        return {nullptr, 0};
    } catch (...) {
        std::cerr << "Unknown error getting audio guide tokens." << std::endl;
        return {nullptr, 0};
    }
}

cactus_float_array_c_t cactus_decode_audio_tokens_c(cactus_context_handle_t handle, const int32_t* tokens, int32_t count) {
    cactus_float_array_c_t result = {nullptr, 0};
    if (!handle || !tokens || count <= 0) {
        return result;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::vector<llama_token> token_vec(tokens, tokens + count);
        std::vector<float> audio = context->decodeAudioTokens(token_vec);
        
        if (!audio.empty()) {
            result.count = audio.size();
            result.values = (float*)malloc(result.count * sizeof(float));
            if (result.values) {
                std::copy(audio.begin(), audio.end(), result.values);
            } else {
                result.count = 0;
            }
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error decoding audio tokens: " << e.what() << std::endl;
        return {nullptr, 0};
    } catch (...) {
        std::cerr << "Unknown error decoding audio tokens." << std::endl;
        return {nullptr, 0};
    }
}

void cactus_release_vocoder_c(cactus_context_handle_t handle) {
    if (!handle) {
        return;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        context->releaseVocoder();
    } catch (const std::exception& e) {
        std::cerr << "Error releasing vocoder: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error releasing vocoder." << std::endl;
    }
}

} // extern "C"

// **HIGH PRIORITY FFI IMPLEMENTATIONS**

extern "C" {

cactus_bench_result_c_t cactus_bench_c(cactus_context_handle_t handle, int pp, int tg, int pl, int nr) {
    cactus_bench_result_c_t result = {0};
    if (!handle) {
        return result;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::string bench_json = context->bench(pp, tg, pl, nr);
        
        char model_desc[128];
        llama_model_desc(context->model, model_desc, sizeof(model_desc));
        result.model_name = safe_strdup(model_desc);
        result.model_size = llama_model_size(context->model);
        result.model_params = llama_model_n_params(context->model);
        
        std::istringstream ss(bench_json);
        std::string token;
        std::vector<std::string> tokens;
        
        ss.ignore(1);
        while (std::getline(ss, token, ',')) {
            if (token.back() == ']') token.pop_back();
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 6) {
            result.pp_avg = std::stod(tokens[3]);
            result.pp_std = std::stod(tokens[4]);
            result.tg_avg = std::stod(tokens[5]);
            result.tg_std = tokens.size() > 6 ? std::stod(tokens[6]) : 0.0;
        }
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmarking: " << e.what() << std::endl;
        return {0};
    }
}

int cactus_apply_lora_adapters_c(cactus_context_handle_t handle, const cactus_lora_adapters_c_t* adapters) {
    if (!handle || !adapters) {
        return -1;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::vector<common_adapter_lora_info> lora_adapters;
        
        for (int i = 0; i < adapters->count; ++i) {
            common_adapter_lora_info adapter;
            adapter.path = adapters->adapters[i].path;
            adapter.scale = adapters->adapters[i].scale;
            lora_adapters.push_back(adapter);
        }
        
        return context->applyLoraAdapters(lora_adapters);
    } catch (const std::exception& e) {
        std::cerr << "Error applying LoRA adapters: " << e.what() << std::endl;
        return -2;
    }
}

void cactus_remove_lora_adapters_c(cactus_context_handle_t handle) {
    if (!handle) {
        return;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        context->removeLoraAdapters();
    } catch (const std::exception& e) {
        std::cerr << "Error removing LoRA adapters: " << e.what() << std::endl;
    }
}

cactus_lora_adapters_c_t cactus_get_loaded_lora_adapters_c(cactus_context_handle_t handle) {
    cactus_lora_adapters_c_t result = {nullptr, 0};
    if (!handle) {
        return result;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        auto loaded_adapters = context->getLoadedLoraAdapters();
        
        if (!loaded_adapters.empty()) {
            result.count = loaded_adapters.size();
            result.adapters = (cactus_lora_adapter_c_t*)malloc(result.count * sizeof(cactus_lora_adapter_c_t));
            
            if (result.adapters) {
                for (int i = 0; i < result.count; ++i) {
                    result.adapters[i].path = safe_strdup(loaded_adapters[i].path);
                    result.adapters[i].scale = loaded_adapters[i].scale;
                }
            } else {
                result.count = 0;
            }
        }
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error getting loaded LoRA adapters: " << e.what() << std::endl;
        return {nullptr, 0};
    }
}

bool cactus_validate_chat_template_c(cactus_context_handle_t handle, bool use_jinja, const char* name) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->validateModelChatTemplate(use_jinja, name);
    } catch (const std::exception& e) {
        std::cerr << "Error validating chat template: " << e.what() << std::endl;
        return false;
    }
}

char* cactus_get_formatted_chat_c(cactus_context_handle_t handle, const char* messages, const char* chat_template) {
    if (!handle || !messages) {
        return safe_strdup("");
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        std::string template_str = chat_template ? chat_template : "";
        std::string result = context->getFormattedChat(messages, template_str);
        return safe_strdup(result);
    } catch (const std::exception& e) {
        std::cerr << "Error formatting chat: " << e.what() << std::endl;
        return safe_strdup("");
    }
}

void cactus_rewind_c(cactus_context_handle_t handle) {
    if (!handle) {
        return;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        context->rewind();
    } catch (const std::exception& e) {
        std::cerr << "Error rewinding context: " << e.what() << std::endl;
    }
}

bool cactus_init_sampling_c(cactus_context_handle_t handle) {
    if (!handle) {
        return false;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->initSampling();
    } catch (const std::exception& e) {
        std::cerr << "Error initializing sampling: " << e.what() << std::endl;
        return false;
    }
}

int32_t cactus_get_n_ctx_c(cactus_context_handle_t handle) {
    if (!handle) {
        return 0;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return context->n_ctx;
    } catch (const std::exception& e) {
        std::cerr << "Error getting n_ctx: " << e.what() << std::endl;
        return 0;
    }
}

int32_t cactus_get_n_embd_c(cactus_context_handle_t handle) {
    if (!handle) {
        return 0;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return llama_model_n_embd(context->model);
    } catch (const std::exception& e) {
        std::cerr << "Error getting n_embd: " << e.what() << std::endl;
        return 0;
    }
}

char* cactus_get_model_desc_c(cactus_context_handle_t handle) {
    if (!handle) {
        return safe_strdup("");
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        char model_desc[128];
        llama_model_desc(context->model, model_desc, sizeof(model_desc));
        return safe_strdup(model_desc);
    } catch (const std::exception& e) {
        std::cerr << "Error getting model description: " << e.what() << std::endl;
        return safe_strdup("");
    }
}

int64_t cactus_get_model_size_c(cactus_context_handle_t handle) {
    if (!handle) {
        return 0;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return llama_model_size(context->model);
    } catch (const std::exception& e) {
        std::cerr << "Error getting model size: " << e.what() << std::endl;
        return 0;
    }
}

int64_t cactus_get_model_params_c(cactus_context_handle_t handle) {
    if (!handle) {
        return 0;
    }
    
    cactus::cactus_context* context = reinterpret_cast<cactus::cactus_context*>(handle);
    try {
        return llama_model_n_params(context->model);
    } catch (const std::exception& e) {
        std::cerr << "Error getting model params: " << e.what() << std::endl;
        return 0;
    }
}

void cactus_free_bench_result_members_c(cactus_bench_result_c_t* result) {
    if (result) {
        cactus_free_string_c(result->model_name);
        result->model_name = nullptr;
    }
}

void cactus_free_lora_adapters_c(cactus_lora_adapters_c_t* adapters) {
    if (adapters && adapters->adapters) {
        for (int i = 0; i < adapters->count; ++i) {
            cactus_free_string_c((char*)adapters->adapters[i].path);
        }
        free(adapters->adapters);
        adapters->adapters = nullptr;
        adapters->count = 0;
    }
}

} // extern "C"