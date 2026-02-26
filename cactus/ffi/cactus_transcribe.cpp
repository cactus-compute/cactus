#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model.h"
#include "telemetry/telemetry.h"
#include "../../libs/audio/wav.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cctype>

using namespace cactus::engine;
using namespace cactus::ffi;
using cactus::audio::WHISPER_TARGET_FRAMES;
using cactus::audio::WHISPER_SAMPLE_RATE;
using cactus::audio::apply_preemphasis;
using cactus::audio::get_parakeet_spectrogram_config;
using cactus::audio::get_whisper_spectrogram_config;
using cactus::audio::normalize_parakeet_log_mel;
using cactus::audio::trim_mel_frames;

static constexpr size_t WHISPER_MAX_DECODER_POSITIONS = 448;

static std::vector<float> normalize_mel(std::vector<float>& mel, size_t n_mels) {
    size_t n_frames = mel.size() / n_mels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : mel)
        if (v > max_val) max_val = v;

    float min_allowed = max_val - 8.0f;
    for (float& v : mel) {
        if (v < min_allowed) v = min_allowed;
        v = (v + 4.0f) / 4.0f;
    }

    if (n_frames != WHISPER_TARGET_FRAMES) {
        std::vector<float> fixed(n_mels * WHISPER_TARGET_FRAMES, 0.0f);
        size_t copy_frames = std::min(n_frames, WHISPER_TARGET_FRAMES);
        for (size_t m = 0; m < n_mels; ++m) {
            const float* src = &mel[m * n_frames];
            float* dst = &fixed[m * WHISPER_TARGET_FRAMES];
            std::copy(src, src + copy_frames, dst);
        }
        return fixed;
    }
    return mel;
}

extern "C" {

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
    if (!model) {
        std::string error_msg = last_error_message.empty() ? "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("transcribe", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, error_msg.c_str());
        return -1;
    }
    if (!prompt || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("transcribe", "Invalid parameters: prompt, response_buffer, or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, "Invalid parameters");
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("transcribe", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, "No audio input provided");
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR("transcribe", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, "Cannot provide both audio_file_path and pcm_buffer");
        return -1;
    }

    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR("transcribe", "Invalid pcm_buffer_size: " << pcm_buffer_size);
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, "pcm_buffer_size must be even and at least 2 bytes");
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::lock_guard<std::mutex> lock(handle->model_mutex);
        handle->should_stop = false;

        float temperature, top_p, confidence_threshold;
        size_t top_k, max_tokens, tool_rag_top_k;
        std::vector<std::string> stop_sequences;
        bool force_tools, include_stop_sequences, use_vad, telemetry_enabled;
        float cloud_handoff_threshold = handle->model->get_config().default_cloud_handoff_threshold;
        const std::string opts = options_json ? options_json : "";
        parse_options_json(
            options_json ? options_json : "", temperature,
            top_p, top_k, max_tokens, stop_sequences,
            force_tools, tool_rag_top_k, confidence_threshold,
            include_stop_sequences, use_vad, telemetry_enabled
        );
        {
            size_t pos = opts.find("\"cloud_handoff_threshold\"");
            if (pos != std::string::npos) {
                pos = opts.find(':', pos);
                if (pos != std::string::npos) {
                    ++pos;
                    while (pos < opts.size() && std::isspace(static_cast<unsigned char>(opts[pos]))) ++pos;
                    try {
                        cloud_handoff_threshold = std::stof(opts.substr(pos));
                    } catch (...) {}
                }
            }
        }

        const char* force_handoff_env = std::getenv("CACTUS_FORCE_HANDOFF");
        if (force_handoff_env && force_handoff_env[0] == '1' && force_handoff_env[1] == '\0') {
            cloud_handoff_threshold = 0.0001f;
        }

        (void)telemetry_enabled;

        bool is_moonshine = handle->model->get_config().model_type == cactus::engine::Config::ModelType::MOONSHINE;
        bool is_parakeet = handle->model->get_config().model_type == cactus::engine::Config::ModelType::PARAKEET;

        std::vector<float> audio_buffer;
        if (audio_file_path == nullptr) {
            const int16_t* pcm_samples = reinterpret_cast<const int16_t*>(pcm_buffer);
            size_t num_samples = pcm_buffer_size / 2;

            std::vector<float> waveform_fp32(num_samples);
            for (size_t i = 0; i < num_samples; i++)
                waveform_fp32[i] = static_cast<float>(pcm_samples[i]) / 32768.0f;

            audio_buffer = resample_to_16k_fp32(waveform_fp32, WHISPER_SAMPLE_RATE);
        } else {
             AudioFP32 audio = load_wav(audio_file_path);
             audio_buffer = resample_to_16k_fp32(audio.samples, audio.sample_rate);
        }

        if (use_vad) {
            auto* vad = static_cast<SileroVADModel*>(handle->vad_model.get());
            auto segments = vad->get_speech_timestamps(audio_buffer, {});

            std::vector<float> speech_audio;
            for (const auto& segment : segments) {
                speech_audio.insert(
                    speech_audio.end(),
                    audio_buffer.begin() + segment.start,
                    audio_buffer.begin() + std::min(segment.end, audio_buffer.size())
                );
            }
            audio_buffer = std::move(speech_audio);

            if (audio_buffer.empty()) {
                CACTUS_LOG_DEBUG("transcribe", "VAD detected only silence, returning empty transcription");

                auto vad_end_time = std::chrono::high_resolution_clock::now();
                double vad_total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(vad_end_time - start_time).count() / 1000.0;

                std::string json = construct_response_json("", {}, 0.0, vad_total_time_ms, 0.0, 0.0, 0, 0, 1.0f);

                if (json.size() >= buffer_size) {
                    handle_error_response("Response buffer too small", response_buffer, buffer_size);
                    cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, "Response buffer too small");
                    return -1;
                }

                cactus::telemetry::recordTranscription(handle->model_name.c_str(), true, 0.0, 0.0, vad_total_time_ms, 0, "");
                std::strcpy(response_buffer, json.c_str());
                return static_cast<int>(json.size());
            }
        }
        const float audio_length_sec = static_cast<float>(audio_buffer.size()) / static_cast<float>(WHISPER_SAMPLE_RATE);

        if (!is_moonshine) {
            AudioProcessor ap;
            if (is_parakeet) {
                auto cfg = get_parakeet_spectrogram_config();
                size_t mel_bins = std::max<size_t>(1, static_cast<size_t>(handle->model->get_config().num_mel_bins));
                const size_t waveform_samples = audio_buffer.size();
                ap.init_mel_filters(cfg.n_fft / 2 + 1, mel_bins, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
                apply_preemphasis(audio_buffer, 0.97f);
                audio_buffer = ap.compute_spectrogram(audio_buffer, cfg);
                normalize_parakeet_log_mel(audio_buffer, mel_bins);
                size_t valid_frames = waveform_samples / cfg.hop_length;
                if (valid_frames == 0) valid_frames = 1;
                trim_mel_frames(audio_buffer, mel_bins, valid_frames);
            } else {
                auto cfg = get_whisper_spectrogram_config();
                ap.init_mel_filters(cfg.n_fft / 2 + 1, 80, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
                std::vector<float> mel = ap.compute_spectrogram(audio_buffer, cfg);
                audio_buffer = normalize_mel(mel, 80);
            }
        }

        if (audio_buffer.empty()) {
            CACTUS_LOG_ERROR("transcribe", "Computed audio features are empty");
            handle_error_response("Computed audio features are empty", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, "Computed audio features are empty");
            return -1;
        }

        CACTUS_LOG_DEBUG("transcribe", "Audio features prepared, size: " << audio_buffer.size());

        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            CACTUS_LOG_ERROR("transcribe", "Tokenizer unavailable");
            handle_error_response("Tokenizer unavailable", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, "Tokenizer unavailable");
            return -1;
        }

        std::vector<uint32_t> tokens = tokenizer->encode(std::string(prompt));
        if (tokens.empty() && !is_moonshine && !is_parakeet) {
            CACTUS_LOG_ERROR("transcribe", "Decoder input tokens empty after encoding prompt");
            handle_error_response("Decoder input tokens empty", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, "Decoder input tokens empty");
            return -1;
        }

        if (!is_parakeet) {
            size_t max_allowed_tokens = 0;
            if (tokens.size() < WHISPER_MAX_DECODER_POSITIONS) {
                max_allowed_tokens = WHISPER_MAX_DECODER_POSITIONS - tokens.size();
            }
            if (max_tokens > max_allowed_tokens) {
                max_tokens = max_allowed_tokens;
            }
        }

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({ tokenizer->get_eos_token() });

        double time_to_first_token = 0.0;
        size_t completion_tokens = 0;
        std::vector<uint32_t> generated_tokens;
        std::string final_text;

        float first_token_entropy = 0.0f;
        float total_entropy_sum = 0.0f;
        size_t total_entropy_count = 0;
        float max_token_entropy_norm = 0.0f;

        float max_tps = handle->model->get_config().default_max_tps;
        if (max_tps < 0) {
            max_tps = 100;
        }

        size_t max_tps_tokens = std::max<size_t>(1, static_cast<size_t>(audio_length_sec * max_tps));
        if (max_tokens > max_tps_tokens) {
            max_tokens = max_tps_tokens;
        }

        uint32_t next_token = handle->model->decode_with_audio(tokens, audio_buffer, temperature, top_p, top_k, "", &first_token_entropy);
        {
            auto t_first = std::chrono::high_resolution_clock::now();
            time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
        }

        total_entropy_sum += first_token_entropy;
        total_entropy_count++;
        if (first_token_entropy > max_token_entropy_norm) max_token_entropy_norm = first_token_entropy;

        generated_tokens.push_back(next_token);
        tokens.push_back(next_token);
        completion_tokens++;

        std::string piece = tokenizer->decode({ next_token });
        final_text += piece;
        if (callback) callback(piece.c_str(), next_token, user_data);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            for (size_t i = 1; i < max_tokens; ++i) {
                if (handle->should_stop) break;

                float token_entropy = 0.0f;
                next_token = handle->model->decode_with_audio(tokens, audio_buffer, temperature, top_p, top_k, "", &token_entropy);

                total_entropy_sum += token_entropy;
                total_entropy_count++;
                if (token_entropy > max_token_entropy_norm) max_token_entropy_norm = token_entropy;

                generated_tokens.push_back(next_token);
                tokens.push_back(next_token);
                completion_tokens++;

                piece = tokenizer->decode({ next_token });
                final_text += piece;
                if (callback) callback(piece.c_str(), next_token, user_data);

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;
            }
        }

        float mean_entropy = total_entropy_count > 0 ? total_entropy_sum / static_cast<float>(total_entropy_count) : 0.0f;
        float confidence = 1.0f - mean_entropy;

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        double decode_time_ms = std::max(0.0, total_time_ms - time_to_first_token);

        size_t prompt_tokens = 0;
        if (!tokens.empty() && completion_tokens <= tokens.size())
            prompt_tokens = tokens.size() - completion_tokens;

        double prefill_tps = time_to_first_token > 0 ? (prompt_tokens * 1000.0) / time_to_first_token : 0.0;
        double decode_tps = (completion_tokens > 1 && decode_time_ms > 0.0) ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;

        std::string cleaned_text = final_text;
        
        const std::vector<std::string> tokens_to_remove = {
            "<|startoftranscript|>",
            "</s>",
            "<pad>"
        };
        for (const auto& token_to_remove : tokens_to_remove) {
            size_t pos = 0;
            while ((pos = cleaned_text.find(token_to_remove, pos)) != std::string::npos) {
                cleaned_text.erase(pos, token_to_remove.length());
            }
        }
        
        if (!cleaned_text.empty() && cleaned_text[0] == ' ') {
            cleaned_text.erase(0, 1);
        }

        bool entropy_handoff = false;
        if (!cleaned_text.empty() && cleaned_text.length() > 5) {
             if (cloud_handoff_threshold > 0.0f && max_token_entropy_norm > cloud_handoff_threshold) {
                 entropy_handoff = true;
             }
        }
        const bool cloud_handoff = entropy_handoff;

        std::string json = construct_response_json(cleaned_text, {}, time_to_first_token, total_time_ms, prefill_tps, decode_tps, prompt_tokens, completion_tokens, confidence, cloud_handoff);

        if (json.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, "Response buffer too small");
            return -1;
        }

        cactus::telemetry::recordTranscription(handle->model_name.c_str(), true, time_to_first_token, decode_tps, total_time_ms, static_cast<int>(completion_tokens), "");

        std::strcpy(response_buffer, json.c_str());

        return static_cast<int>(json.size());
    }
    catch (const std::exception& e) {
        CACTUS_LOG_ERROR("transcribe", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, e.what());
        return -1;
    }
    catch (...) {
        CACTUS_LOG_ERROR("transcribe", "Unknown exception during transcription");
        handle_error_response("Unknown error in transcribe", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, "Unknown error in transcribe");
        return -1;
    }
}

}