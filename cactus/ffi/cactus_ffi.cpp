#include "cactus_ffi.h"
#include "ffi_utils.h"
#include "../../libs/audio/wav.h"
#include "../engine/engine.h"
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <iostream>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

struct CactusModelHandle {
    std::unique_ptr<Model> model;
    std::atomic<bool> should_stop;
    std::vector<uint32_t> processed_tokens; 

    CactusModelHandle() : should_stop(false) {}
};

static bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                                   const std::vector<std::vector<uint32_t>>& stop_sequences) {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;

        if (generated_tokens.size() >= stop_seq.size()) {
            if (std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin())) {
                return true;
            }
        }
    }
    return false;
}

static std::vector<float> compute_whisper_mel_from_wav(const std::string& wav_path) {
    using namespace cactus::engine;
    AudioFP32 audio = load_wav(wav_path);
    std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);

    AudioProcessor::SpectrogramConfig cfg{};
    cfg.n_fft        = 400; 
    cfg.frame_length = 400;
    cfg.hop_length   = 160;
    cfg.power        = 2.0f;
    cfg.center       = true;
    cfg.pad_mode     = "reflect";
    cfg.onesided     = true;
    cfg.dither       = 0.0f;
    cfg.mel_floor    = 1e-10f;
    cfg.log_mel      = "log10";      // <- IMPORTANT: log10, NOT "dB"
    cfg.reference    = 1.0f;
    cfg.min_value    = 1e-10f;
    cfg.remove_dc_offset = true;

    const size_t num_mel_filters = 80;
    const size_t num_frequency_bins = cfg.n_fft / 2 + 1;

    AudioProcessor ap;
    ap.init_mel_filters(num_frequency_bins,num_mel_filters,0.0f,8000.0f,16000);

    std::vector<float> mel = ap.compute_spectrogram(waveform_16k, cfg);

    if (mel.empty()) {
        return mel;
    }

    size_t n_mels = num_mel_filters;
    size_t n_frames = mel.size() / n_mels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : mel) {
        if (v > max_val) max_val = v;
    }
    float min_allowed = max_val - 8.0f;

    for (float& v : mel) {
        if (v < min_allowed) v = min_allowed;
        v = (v + 4.0f) / 4.0f;
    }

    const size_t target_frames = 3000;
    if (n_frames != target_frames) {
        std::vector<float> fixed(n_mels * target_frames, 0.0f);

        size_t copy_frames = std::min(n_frames, target_frames);
        for (size_t m = 0; m < n_mels; ++m) {
            const float* src = &mel[m * n_frames];
            float* dst = &fixed[m * target_frames];
            std::copy(src, src + copy_frames, dst);
        }

        return fixed;
    }

    return mel;
}




extern "C" {

static std::string last_error_message;

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, size_t context_size, const char* corpus_dir) {
    try {
        
        auto* handle = new CactusModelHandle();
        handle->model = create_model(model_path);


        if (!handle->model) {
            last_error_message = "Failed to create model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }

        if (!handle->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }

        if (corpus_dir != nullptr) {
            Tokenizer* tok = handle->model->get_tokenizer();
            if (tok) {
                try {
                    cactus::engine::Tokenizer* etok = static_cast<cactus::engine::Tokenizer*>(tok);
                    etok->set_corpus_dir(std::string(corpus_dir));
                } catch (...) {
                    
                }
            }
        }

        return handle;
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error during model initialization";
        return nullptr;
    }
}

int cactus_transcribe(
    cactus_model_t model,
    const char* audio_file_path,
    const char* prompt,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    cactus_token_callback callback,
    void* user_data
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? "Model not initialized. Check model path and files." : last_error_message;
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path || !prompt || !response_buffer || buffer_size == 0) {
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto* handle = static_cast<CactusModelHandle*>(model);
        handle->should_stop = false;

        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        parse_options_json(options_json ? options_json : "",
                          temperature, top_p, top_k, max_tokens, stop_sequences);

        std::vector<float> mel_bins = compute_whisper_mel_from_wav(audio_file_path);

        if (mel_bins.empty()) {
            handle_error_response("Computed mel spectrogram is empty", response_buffer, buffer_size);
            return -1;
        }

        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            handle_error_response("Tokenizer unavailable for Whisper model", response_buffer, buffer_size);
            return -1;
        }

        std::string prompt_text(prompt);
        std::vector<uint32_t> tokens = tokenizer->encode(prompt_text);
        if (tokens.empty()) {
            handle_error_response("Decoder input tokens are empty after encoding prompt text", response_buffer, buffer_size);
            return -1;
        }

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({ tokenizer->get_eos_token() });

        double time_to_first_token = 0.0;
        size_t completion_tokens   = 0;
        std::vector<uint32_t> generated_tokens;
        std::string final_text;

        uint32_t next_token = handle->model->generate_with_audio(tokens, mel_bins,temperature, top_p, top_k, "profile.txt");{
            auto t_first = std::chrono::high_resolution_clock::now();
            time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
        }

        generated_tokens.push_back(next_token);
        tokens.push_back(next_token);
        completion_tokens++;

        if (callback) {
            std::string piece = tokenizer->decode({ next_token });
            final_text += piece;
            callback(piece.c_str(), next_token, user_data);
        } else {
            final_text += tokenizer->decode({ next_token });
        }

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            for (size_t i = 1; i < max_tokens; ++i) {
                if (handle->should_stop) break;

                next_token = handle->model->generate_with_audio(tokens,mel_bins,temperature, top_p, top_k, "profile.txt");

                generated_tokens.push_back(next_token);
                tokens.push_back(next_token);
                completion_tokens++;

                std::string piece = tokenizer->decode({ next_token });
                final_text += piece;

                if (callback) {
                    callback(piece.c_str(), next_token, user_data);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms =std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        double decode_time_ms = std::max(0.0, total_time_ms - time_to_first_token);
        double tokens_per_second =(completion_tokens > 1 && decode_time_ms > 0.0) ? ((completion_tokens - 1) * 1000.0) / decode_time_ms: 0.0;

        size_t prompt_tokens = 0;
        if (!tokens.empty() && completion_tokens <= tokens.size())
            prompt_tokens = tokens.size() - completion_tokens;

        std::string json = construct_response_json(final_text, {}, time_to_first_token,total_time_ms,tokens_per_second,prompt_tokens,completion_tokens);

        if (json.size() >= buffer_size) {
            std::cout << "Response buffer too small for Whisper output" << std::endl;
            handle_error_response("Response buffer too small for Whisper output", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, json.c_str());
        return static_cast<int>(json.size());
    }
    catch (...) {
        handle_error_response("Unknown fatal error inside Whisper FFI",
                              response_buffer, buffer_size);
        return -1;
    }
}

int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? 
            "Model not initialized. Check model path and files." : last_error_message;
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }
    
    if (!messages_json || !response_buffer || buffer_size == 0) {
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();
        handle->should_stop = false;
        
        std::vector<std::string> image_paths;
        auto messages = parse_messages_json(messages_json, image_paths);
        
        if (messages.empty()) {
            handle_error_response("No messages provided", response_buffer, buffer_size);
            return -1;
        }
        
        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        parse_options_json(options_json ? options_json : "", 
                          temperature, top_p, top_k, max_tokens, stop_sequences);
        
        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0) {
            tools = parse_tools_json(tools_json);
        }
        
        std::string formatted_tools = format_tools_for_prompt(tools);
        std::string full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools);

        if (full_prompt.find("ERROR:") == 0) {
            handle_error_response(full_prompt.substr(6), response_buffer, buffer_size);
            return -1;
        }

        // Use incremental processing approach from main
        std::vector<uint32_t> current_prompt_tokens = tokenizer->encode(full_prompt);
        
        std::vector<uint32_t> tokens_to_process;
        bool is_prefix = (current_prompt_tokens.size() >= handle->processed_tokens.size()) &&
                         std::equal(handle->processed_tokens.begin(), handle->processed_tokens.end(), current_prompt_tokens.begin());

        if (handle->processed_tokens.empty() || !is_prefix) {
            handle->model->reset_cache();
            tokens_to_process = current_prompt_tokens;
        } else {
            tokens_to_process.assign(current_prompt_tokens.begin() + handle->processed_tokens.size(), current_prompt_tokens.end());
        }
        
        size_t prompt_tokens = tokens_to_process.size();

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({tokenizer->get_eos_token()});
        for (const auto& stop_seq : stop_sequences) {
            stop_token_sequences.push_back(tokenizer->encode(stop_seq));
        }

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        // Generate first token - use image support from HEAD if images are present
        uint32_t next_token;
        if (tokens_to_process.empty()) {
            if (handle->processed_tokens.empty()) {
                 handle_error_response("Cannot generate from empty prompt", response_buffer, buffer_size);
                 return -1;
            }
            std::vector<uint32_t> last_token_vec = { handle->processed_tokens.back() };
            next_token = handle->model->generate(last_token_vec, temperature, top_p, top_k);
        } else {
            if (!image_paths.empty()) {
                next_token = handle->model->generate_with_images(tokens_to_process, image_paths, temperature, top_p, top_k, "profile.txt");
            } else {
                next_token = handle->model->generate(tokens_to_process, temperature, top_p, top_k, "profile.txt");
            }
        }
        
        handle->processed_tokens = current_prompt_tokens;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (callback) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (handle->should_stop) break;

                next_token = handle->model->generate({next_token}, temperature, top_p, top_k);
                generated_tokens.push_back(next_token);
                handle->processed_tokens.push_back(next_token);

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;

                if (callback) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        
        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double tokens_per_second = completion_tokens > 1 ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;
        
        std::string response_text = tokenizer->decode(generated_tokens);
        
        std::string regular_response;
        std::vector<std::string> function_calls;
        parse_function_calls_from_response(response_text, regular_response, function_calls);
        
        std::string result = construct_response_json(regular_response, function_calls, time_to_first_token,
                                                     total_time_ms, tokens_per_second, prompt_tokens,
                                                     completion_tokens);
        
        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }
        
        std::strcpy(response_buffer, result.c_str());
        
        return static_cast<int>(result.length());
        
    } catch (const std::exception& e) {
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) {
        delete static_cast<CactusModelHandle*>(model);
    }
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->model->reset_cache();
    handle->processed_tokens.clear();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->should_stop = true;
}

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) return -1;
    if (!text || !embeddings_buffer || buffer_size == 0) return -1;

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();

        std::vector<uint32_t> tokens = tokenizer->encode(text);
        if (tokens.empty()) return -1;

        std::vector<float> embeddings = handle->model->get_embeddings(tokens, true);
        if (embeddings.size() * sizeof(float) > buffer_size) return -2;

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) {
            *embedding_dim = embeddings.size();
        }

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding generation";
        return -1;
    }
}

int cactus_image_embed(
    cactus_model_t model,
    const char* image_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) return -1;
    if (!image_path || !embeddings_buffer || buffer_size == 0) return -1;

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);

        std::vector<float> embeddings = handle->model->get_image_embeddings(image_path);
        if (embeddings.empty()) return -1;
        if (embeddings.size() * sizeof(float) > buffer_size) return -2;

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) {
            *embedding_dim = embeddings.size();
        }

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image embedding generation";
        return -1;
    }
}

int cactus_audio_embed(
    cactus_model_t model,
    const char* audio_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) return -1;
    if (!audio_path || !embeddings_buffer || buffer_size == 0) return -1;

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);

        auto mel_bins = compute_whisper_mel_from_wav(audio_path);
        if (mel_bins.empty()) {
            last_error_message = "Failed to compute mel spectrogram from audio file";
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_audio_embeddings(mel_bins);
        if (embeddings.empty()) return -1;
        if (embeddings.size() * sizeof(float) > buffer_size) return -2;

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) {
            *embedding_dim = embeddings.size();
        }

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio embedding generation";
        return -1;
    }
}

}