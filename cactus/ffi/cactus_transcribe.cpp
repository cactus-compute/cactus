#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_telemetry.h"
#include "../../libs/audio/wav.h"
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t WHISPER_TARGET_FRAMES = 3000;
static constexpr int WHISPER_SAMPLE_RATE = 16000;

static AudioProcessor::SpectrogramConfig get_whisper_spectrogram_config() {
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
    cfg.log_mel      = "log10";
    cfg.reference    = 1.0f;
    cfg.min_value    = 1e-10f;
    cfg.remove_dc_offset = true;
    return cfg;
}

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

static std::vector<float> compute_whisper_mel_from_pcm(const int16_t* pcm_samples, size_t num_samples, int sample_rate_in) {
    if (!pcm_samples || num_samples == 0) return {};

    std::vector<float> waveform_fp32(num_samples);
    for (size_t i = 0; i < num_samples; i++)
        waveform_fp32[i] = static_cast<float>(pcm_samples[i]) / 32768.0f;

    std::vector<float> waveform_16k = resample_to_16k_fp32(waveform_fp32, sample_rate_in);
    if (waveform_16k.empty()) return {};

    auto cfg = get_whisper_spectrogram_config();
    const size_t num_mel_filters = 80;
    const size_t num_frequency_bins = cfg.n_fft / 2 + 1;

    AudioProcessor ap;
    ap.init_mel_filters(num_frequency_bins, num_mel_filters, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
    std::vector<float> mel = ap.compute_spectrogram(waveform_16k, cfg);

    if (mel.empty()) return mel;
    return normalize_mel(mel, num_mel_filters);
}

static std::vector<float> compute_whisper_mel_from_wav(const std::string& wav_path) {
    AudioFP32 audio = load_wav(wav_path);
    std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);

    auto cfg = get_whisper_spectrogram_config();
    const size_t num_mel_filters = 80;
    const size_t num_frequency_bins = cfg.n_fft / 2 + 1;

    AudioProcessor ap;
    ap.init_mel_filters(num_frequency_bins, num_mel_filters, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
    std::vector<float> mel = ap.compute_spectrogram(waveform_16k, cfg);

    if (mel.empty()) return mel;
    return normalize_mel(mel, num_mel_filters);
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
        return -1;
    }

    if (!prompt || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("transcribe", "Invalid parameters: prompt, response_buffer, or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("transcribe", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR("transcribe", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        return -1;
    }

    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR("transcribe", "Invalid pcm_buffer_size: " << pcm_buffer_size);
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::lock_guard<std::mutex> lock(handle->model_mutex);
        handle->should_stop = false;

        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        bool force_tools = false;  
        parse_options_json(options_json ? options_json : "", temperature, top_p, top_k, max_tokens, stop_sequences, force_tools);

        std::vector<float> mel_bins;
        if (audio_file_path == nullptr) {
            const int16_t* pcm_samples = reinterpret_cast<const int16_t*>(pcm_buffer);
            size_t num_samples = pcm_buffer_size / 2;
            mel_bins = compute_whisper_mel_from_pcm(pcm_samples, num_samples, WHISPER_SAMPLE_RATE);
        } else {
            mel_bins = compute_whisper_mel_from_wav(audio_file_path);
        }

        if (mel_bins.empty()) {
            CACTUS_LOG_ERROR("transcribe", "Computed mel spectrogram is empty");
            handle_error_response("Computed mel spectrogram is empty", response_buffer, buffer_size);
            return -1;
        }

        CACTUS_LOG_DEBUG("transcribe", "Mel spectrogram computed, size: " << mel_bins.size());

        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            CACTUS_LOG_ERROR("transcribe", "Tokenizer unavailable");
            handle_error_response("Tokenizer unavailable", response_buffer, buffer_size);
            return -1;
        }

        std::vector<uint32_t> tokens = tokenizer->encode(std::string(prompt));
        if (tokens.empty()) {
            CACTUS_LOG_ERROR("transcribe", "Decoder input tokens empty after encoding prompt");
            handle_error_response("Decoder input tokens empty", response_buffer, buffer_size);
            return -1;
        }

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({ tokenizer->get_eos_token() });

        double time_to_first_token = 0.0;
        size_t completion_tokens = 0;
        std::vector<uint32_t> generated_tokens;
        std::string final_text;

        uint32_t next_token = handle->model->decode_with_audio(tokens, mel_bins, temperature, top_p, top_k, "profile.txt");
        {
            auto t_first = std::chrono::high_resolution_clock::now();
            time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
        }

        generated_tokens.push_back(next_token);
        tokens.push_back(next_token);
        completion_tokens++;

        std::string piece = tokenizer->decode({ next_token });
        final_text += piece;
        if (callback) callback(piece.c_str(), next_token, user_data);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            for (size_t i = 1; i < max_tokens; ++i) {
                if (handle->should_stop) break;

                next_token = handle->model->decode_with_audio(tokens, mel_bins, temperature, top_p, top_k, "profile.txt");
                generated_tokens.push_back(next_token);
                tokens.push_back(next_token);
                completion_tokens++;

                piece = tokenizer->decode({ next_token });
                final_text += piece;
                if (callback) callback(piece.c_str(), next_token, user_data);

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        double decode_time_ms = std::max(0.0, total_time_ms - time_to_first_token);
        double tokens_per_second = (completion_tokens > 1 && decode_time_ms > 0.0) ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;

        size_t prompt_tokens = 0;
        if (!tokens.empty() && completion_tokens <= tokens.size())
            prompt_tokens = tokens.size() - completion_tokens;

        std::string json = construct_response_json(final_text, {}, time_to_first_token, total_time_ms, tokens_per_second, prompt_tokens, completion_tokens);

        if (json.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, json.c_str());

        CactusTelemetry::getInstance().recordTranscription(
            handle->model_name,
            true,
            time_to_first_token,
            tokens_per_second,
            total_time_ms,
            completion_tokens,
            ""
        );

        return static_cast<int>(json.size());
    }
    catch (const std::exception& e) {
        CACTUS_LOG_ERROR("transcribe", "Exception: " << e.what());

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordTranscription(
            handle->model_name,
            false,
            0.0,
            0,
            0.0,
            0,
            std::string(e.what())
        );

        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    }
    catch (...) {
        CACTUS_LOG_ERROR("transcribe", "Unknown exception during transcription");

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordTranscription(
            handle->model_name,
            false,
            0.0,
            0,
            0.0,
            0,
            "Unknown error in transcribe"
        );

        handle_error_response("Unknown error in transcribe", response_buffer, buffer_size);
        return -1;
    }
}

}
