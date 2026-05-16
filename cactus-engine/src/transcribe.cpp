#include "../cactus_engine.h"
#include "utils.h"
#include "cactus_kernels.h"
#include "wav.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <mutex>

using namespace cactus::ffi;

extern "C" {

int cactus_preprocess_audio_features(
    const char* audio_file_path,
    const char* model_type,
    size_t mel_bins,
    float* features_buffer,
    size_t buffer_size,
    size_t* feature_count,
    size_t* out_mel_bins,
    size_t* out_frames
) {
    if (!audio_file_path || !model_type || !features_buffer || !feature_count || !out_mel_bins || !out_frames) {
        last_error_message = "Invalid parameters for audio feature preprocessing";
        CACTUS_LOG_ERROR("audio_preprocess", last_error_message);
        return -1;
    }

    try {
        AudioFP32 audio = load_wav(audio_file_path);
        std::vector<float> audio_samples = resample_to_16k_fp32(audio.samples, audio.sample_rate);
        if (audio_samples.empty()) {
            last_error_message = "No audio samples available for preprocessing";
            CACTUS_LOG_ERROR("audio_preprocess", last_error_message);
            return -1;
        }

        const size_t bins = std::max<size_t>(1, mel_bins);
        std::string lowered(model_type);
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        std::vector<float> features;
        size_t frames = 0;

        if (lowered.find("gemma4") != std::string::npos || lowered.find("gemma-4") != std::string::npos) {
            cactus::engine::Config cfg;
            cfg.model_type = cactus::engine::Config::ModelType::GEMMA4;
            cfg.audio_input_feat_size = static_cast<uint32_t>(bins);
            cfg.audio_soft_tokens = 188;
            cfg.audio_fft_length = 512;
            auto prepared = cactus::audio::preprocess_audio_for_gemma4(std::move(audio_samples), cfg);
            features = std::move(prepared.features);
            frames = prepared.num_frames;
        } else if (lowered.find("parakeet") != std::string::npos) {
            auto cfg = cactus::audio::get_parakeet_spectrogram_config();
            const size_t waveform_samples = audio_samples.size();
            cactus::audio::apply_preemphasis(audio_samples, 0.97f);
            features = cactus::audio::compute_spectrogram_graph(
                audio_samples, cfg, bins, 0.0f, 8000.0f,
                cactus::audio::WHISPER_SAMPLE_RATE, 0, 0);
            cactus::audio::normalize_parakeet_log_mel(features, bins);
            size_t valid_frames = waveform_samples / cfg.hop_length;
            if (valid_frames == 0) valid_frames = 1;
            cactus::audio::trim_mel_frames(features, bins, valid_frames);
            frames = features.size() / bins;
        } else {
            auto cfg = cactus::audio::get_whisper_spectrogram_config();
            const bool is_whisper_v3 = bins > 80;
            if (is_whisper_v3) cactus::audio::apply_whisper_v3_overrides(cfg);
            int norm_type = 1;  // Whisper HF feature extractor uses Slaney-normalized mel filters.
            int scale_type = 2; // Whisper HF feature extractor uses the Slaney mel scale.
            std::vector<float> mel = cactus::audio::compute_spectrogram_graph(
                audio_samples, cfg, bins, 0.0f, 8000.0f,
                cactus::audio::WHISPER_SAMPLE_RATE, norm_type, scale_type);
            features = cactus::audio::normalize_whisper_mel(mel, bins, true);
            frames = features.size() / bins;
        }

        const size_t bytes_needed = features.size() * sizeof(float);
        if (bytes_needed > buffer_size) {
            last_error_message = "Audio feature output buffer too small";
            CACTUS_LOG_ERROR("audio_preprocess", last_error_message);
            return -2;
        }

        std::memcpy(features_buffer, features.data(), bytes_needed);
        *feature_count = features.size();
        *out_mel_bins = bins;
        *out_frames = frames;
        return static_cast<int>(features.size());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("audio_preprocess", last_error_message);
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio feature preprocessing";
        CACTUS_LOG_ERROR("audio_preprocess", last_error_message);
        return -1;
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
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (validate_audio_params("transcribe", model, response_buffer, buffer_size,
                              audio_file_path, pcm_buffer, pcm_buffer_size) != 0) {
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        const auto model_type = handle->model->get_config().model_type;
        const bool is_whisper = model_type == cactus::engine::Config::ModelType::WHISPER;
        const bool is_parakeet = model_type == cactus::engine::Config::ModelType::PARAKEET_TDT;

        if (!is_whisper && !is_parakeet) {
            const uint8_t* pcm_data = pcm_buffer;
            size_t pcm_size = pcm_buffer_size;
            std::vector<uint8_t> file_pcm;

            if (audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
                AudioFP32 audio = load_wav(audio_file_path);
                if (audio.samples.empty()) {
                    CACTUS_LOG_ERROR("transcribe", "Failed to load audio file: " << audio_file_path);
                    handle_error_response("Failed to load audio file", response_buffer, buffer_size);
                    return -1;
                }
                file_pcm.resize(audio.samples.size() * sizeof(int16_t));
                int16_t* out = reinterpret_cast<int16_t*>(file_pcm.data());
                for (size_t i = 0; i < audio.samples.size(); i++) {
                    float clamped = std::max(-1.0f, std::min(1.0f, audio.samples[i]));
                    out[i] = static_cast<int16_t>(clamped * 32767.0f);
                }
                pcm_data = file_pcm.data();
                pcm_size = file_pcm.size();
            }

            std::string user_content = "Transcribe the following audio.";
            if (prompt && prompt[0] != '\0') {
                user_content = prompt;
            }

            std::string messages_json = "[{\"role\": \"user\", \"content\": \""
                + escape_json_string(user_content) + "\"}]";

            return cactus_complete(
                model,
                messages_json.c_str(),
                response_buffer,
                buffer_size,
                options_json,
                nullptr,
                callback,
                user_data,
                pcm_data,
                pcm_size
            );
        }

        std::lock_guard<std::mutex> lock(handle->model_mutex);
        handle->should_stop = false;
        auto start_time = std::chrono::high_resolution_clock::now();
        InferenceOptions options = parse_inference_options_json(options_json ? options_json : "");

        std::vector<float> audio_samples;
        if (audio_file_path == nullptr) {
            auto waveform_fp32 = cactus::audio::pcm_buffer_to_float_samples(pcm_buffer, pcm_buffer_size);
            audio_samples = resample_to_16k_fp32(waveform_fp32, cactus::audio::WHISPER_SAMPLE_RATE);
        } else {
            AudioFP32 audio = load_wav(audio_file_path);
            audio_samples = resample_to_16k_fp32(audio.samples, audio.sample_rate);
        }
        if (audio_samples.empty()) {
            handle_error_response("No audio input provided", response_buffer, buffer_size);
            return -1;
        }

        const size_t mel_bins = std::max<size_t>(1, static_cast<size_t>(handle->model->get_config().num_mel_bins));
        std::vector<float> audio_features;
        if (is_parakeet) {
            auto cfg = cactus::audio::get_parakeet_spectrogram_config();
            const size_t waveform_samples = audio_samples.size();
            cactus::audio::apply_preemphasis(audio_samples, 0.97f);
            audio_features = cactus::audio::compute_spectrogram_graph(
                audio_samples, cfg, mel_bins, 0.0f, 8000.0f,
                cactus::audio::WHISPER_SAMPLE_RATE, 0, 0);
            cactus::audio::normalize_parakeet_log_mel(audio_features, mel_bins);
            size_t valid_frames = waveform_samples / cfg.hop_length;
            if (valid_frames == 0) valid_frames = 1;
            cactus::audio::trim_mel_frames(audio_features, mel_bins, valid_frames);
        } else {
            auto cfg = cactus::audio::get_whisper_spectrogram_config();
            const bool is_whisper_v3 = mel_bins > 80;
            if (is_whisper_v3) cactus::audio::apply_whisper_v3_overrides(cfg);
            int norm_type = 1;  // Whisper HF feature extractor uses Slaney-normalized mel filters.
            int scale_type = 2; // Whisper HF feature extractor uses the Slaney mel scale.
            std::vector<float> mel = cactus::audio::compute_spectrogram_graph(
                audio_samples, cfg, mel_bins, 0.0f, 8000.0f,
                cactus::audio::WHISPER_SAMPLE_RATE, norm_type, scale_type);
            audio_features = cactus::audio::normalize_whisper_mel(mel, mel_bins, true);
        }
        if (audio_features.empty()) {
            handle_error_response("Computed audio features are empty", response_buffer, buffer_size);
            return -1;
        }

        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            handle_error_response("Tokenizer unavailable", response_buffer, buffer_size);
            return -1;
        }

        std::vector<uint32_t> tokens;
        if (prompt && prompt[0] != '\0') {
            tokens = tokenizer->encode(prompt);
        } else if (is_whisper) {
            tokens = tokenizer->encode("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>");
        }

        if (tokens.empty() && is_whisper) {
            handle_error_response("Decoder input tokens empty", response_buffer, buffer_size);
            return -1;
        }

        std::vector<std::vector<uint32_t>> stop_token_sequences = {{tokenizer->get_eos_token()}};
        auto append_exact_stop_sequence = [&](const char* stop_text) {
            std::vector<uint32_t> seq = tokenizer->encode(stop_text);
            if (!seq.empty() && tokenizer->decode(seq) == stop_text) {
                stop_token_sequences.push_back(std::move(seq));
            }
        };
        append_exact_stop_sequence("<|endoftext|>");
        append_exact_stop_sequence("<|endoftranscript|>");
        append_exact_stop_sequence("</s>");
        append_exact_stop_sequence("<pad>");

        const float audio_length_sec =
            static_cast<float>(audio_samples.size()) / static_cast<float>(cactus::audio::WHISPER_SAMPLE_RATE);
        if (options.max_tokens == 100 && (!options_json || std::string(options_json).find("\"max_tokens\"") == std::string::npos)) {
            options.max_tokens = std::max<size_t>(100, static_cast<size_t>(audio_length_sec * (is_parakeet ? 30.0f : 20.0f)));
        }

        constexpr size_t WHISPER_MAX_DECODER_POSITIONS = 448;
        if (is_whisper && tokens.size() < WHISPER_MAX_DECODER_POSITIONS) {
            options.max_tokens = std::min(options.max_tokens, WHISPER_MAX_DECODER_POSITIONS - tokens.size());
        }

        std::string final_text;
        std::vector<uint32_t> generated_tokens;
        generated_tokens.reserve(options.max_tokens);
        const size_t prompt_tokens = tokens.size();
        double time_to_first_token = 0.0;
        float total_entropy_sum = 0.0f;

        for (size_t i = 0; i < options.max_tokens; ++i) {
            if (handle->should_stop) break;

            float token_entropy = 0.0f;
            float tok_time_start = 0.0f;
            float tok_time_end = 0.0f;
            uint32_t next_token = handle->model->decode_with_audio(
                tokens, audio_features,
                options.temperature, options.top_p, options.top_k,
                "", &token_entropy,
                options.min_p, options.repetition_penalty,
                is_parakeet ? &tok_time_start : nullptr,
                is_parakeet ? &tok_time_end : nullptr);

            if (generated_tokens.empty()) {
                auto t_first = std::chrono::high_resolution_clock::now();
                time_to_first_token =
                    std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
            }

            total_entropy_sum += token_entropy;
            generated_tokens.push_back(next_token);
            if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                break;
            }

            std::string piece = tokenizer->decode({next_token});
            if (piece == "<|endoftext|>" || piece == "<|endoftranscript|>" || piece == "</s>" || piece == "<pad>") {
                break;
            }

            tokens.push_back(next_token);
            final_text += piece;
            if (callback) callback(piece.c_str(), next_token, user_data);
        }

        handle->model->reset_cache();

        if (!final_text.empty() && final_text[0] == ' ') {
            final_text.erase(0, 1);
        }

        const size_t completion_tokens = generated_tokens.size();
        float mean_entropy = completion_tokens > 0
            ? total_entropy_sum / static_cast<float>(completion_tokens)
            : 0.0f;
        float confidence = 1.0f - mean_entropy;

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        double prefill_tps = time_to_first_token > 0 ? (prompt_tokens * 1000.0) / time_to_first_token : 0.0;
        double decode_time_ms = std::max(0.0, total_time_ms - time_to_first_token);
        double decode_tps =
            (completion_tokens > 1 && decode_time_ms > 0.0)
                ? ((completion_tokens - 1) * 1000.0) / decode_time_ms
                : 0.0;

        std::string json = construct_response_json(
            final_text, {}, time_to_first_token, total_time_ms,
            prefill_tps, decode_tps, prompt_tokens, completion_tokens,
            confidence);
        if (json.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }
        std::strcpy(response_buffer, json.c_str());
        return static_cast<int>(json.size());
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("transcribe", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    }
}

} // extern "C"
