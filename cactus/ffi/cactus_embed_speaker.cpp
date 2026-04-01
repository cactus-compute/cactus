#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model.h"
#include "../../libs/audio/wav.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;
using cactus::audio::get_wespeaker_spectrogram_config;

extern "C" {

int cactus_embed_speaker(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    (void)options_json;
    if (!model) {
        std::string error_msg = last_error_message.empty() ? "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("embed_speaker", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("embed_speaker", "Invalid parameters: response_buffer or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("embed_speaker", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR("embed_speaker", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        return -1;
    }

    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR("embed_speaker", "Invalid pcm_buffer_size: " << pcm_buffer_size);
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* wespeaker = dynamic_cast<WeSpeakerModel*>(handle->model.get());
        if (!wespeaker) {
            CACTUS_LOG_ERROR("embed_speaker", "Model is not a WeSpeaker embedding model");
            handle_error_response("Model is not a WeSpeaker embedding model", response_buffer, buffer_size);
            return -1;
        }

        std::vector<float> audio;

        if (audio_file_path == nullptr) {
            const int16_t* pcm_samples = reinterpret_cast<const int16_t*>(pcm_buffer);
            const size_t num_samples = pcm_buffer_size / 2;
            audio.resize(num_samples);
            for (size_t i = 0; i < num_samples; ++i)
                audio[i] = static_cast<float>(pcm_samples[i]) / 32768.0f;
        } else {
            AudioFP32 wav_audio = load_wav(audio_file_path);
            audio = resample_to_16k_fp32(wav_audio.samples, wav_audio.sample_rate);
        }

        if (audio.empty()) {
            last_error_message = "Failed to load audio or audio is empty";
            CACTUS_LOG_ERROR("embed_speaker", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        static const auto s_cfg = get_wespeaker_spectrogram_config();
        static AudioProcessor s_ap = []() {
            AudioProcessor p;
            p.init_mel_filters(get_wespeaker_spectrogram_config().n_fft / 2 + 1,
                               80, 20.0f, 8000.0f, 16000, nullptr, "htk");
            return p;
        }();

        std::vector<float> mel = s_ap.compute_spectrogram(audio, s_cfg);
        audio = {};

        static constexpr size_t NUM_MEL_BINS = 80;
        const size_t num_frames = mel.size() / NUM_MEL_BINS;

        for (size_t m = 0; m < NUM_MEL_BINS; ++m) {
            float* row = mel.data() + m * num_frames;
            float sum = 0.0f;
            for (size_t t = 0; t < num_frames; ++t) sum += row[t];
            const float mean = sum / static_cast<float>(num_frames);
            for (size_t t = 0; t < num_frames; ++t) row[t] -= mean;
        }

        auto embedding = wespeaker->embed(mel.data(), mel.size());

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"embedding\":[";
        json << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) json << ",";
            json << embedding[i];
        }
        json << "],";
        json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
        json << "}";

        std::string response = json.str();
        if (response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("embed_speaker", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, response.c_str());
        return static_cast<int>(response.length());

    } catch (const std::exception& e) {
        last_error_message = "Exception during speaker embedding: " + std::string(e.what());
        CACTUS_LOG_ERROR("embed_speaker", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during speaker embedding";
        CACTUS_LOG_ERROR("embed_speaker", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
}

}
