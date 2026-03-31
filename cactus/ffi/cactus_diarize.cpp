#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model.h"
#include "../../libs/audio/wav.h"
#include <chrono>
#include <cstring>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr int DIARIZE_MAX_SPEAKERS = 3;

struct DiarizeOptions {
    size_t step_samples  = 16000;
    float  threshold     = -1.0f;
    int    num_speakers  = -1;
    int    min_speakers  = -1;
    int    max_speakers  = -1;
};

static DiarizeOptions parse_diarize_options(const std::string& json) {
    DiarizeOptions opts;
    if (json.empty()) return opts;

    size_t pos;

    pos = json.find("\"step_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        int v = std::stoi(json.substr(pos));
        if (v > 0) opts.step_samples = static_cast<size_t>(v) * 16;
    }

    pos = json.find("\"threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        opts.threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"num_speakers\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        opts.num_speakers = std::stoi(json.substr(pos));
    }

    pos = json.find("\"min_speakers\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        opts.min_speakers = std::stoi(json.substr(pos));
    }

    pos = json.find("\"max_speakers\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        opts.max_speakers = std::stoi(json.substr(pos));
    }

    return opts;
}

static void apply_speaker_filter(std::vector<float>& scores, size_t total_frames, const DiarizeOptions& opts) {
    int keep = DIARIZE_MAX_SPEAKERS;
    if (opts.num_speakers > 0) keep = std::min(opts.num_speakers, DIARIZE_MAX_SPEAKERS);
    if (opts.max_speakers > 0) keep = std::min(keep, opts.max_speakers);
    if (opts.min_speakers > 0) keep = std::max(keep, std::min(opts.min_speakers, DIARIZE_MAX_SPEAKERS));
    if (keep >= DIARIZE_MAX_SPEAKERS) return;

    float activity[DIARIZE_MAX_SPEAKERS] = {};
    for (size_t f = 0; f < total_frames; ++f)
        for (int s = 0; s < DIARIZE_MAX_SPEAKERS; ++s)
            activity[s] += scores[f * DIARIZE_MAX_SPEAKERS + s];

    int order[DIARIZE_MAX_SPEAKERS] = {0, 1, 2};
    std::sort(order, order + DIARIZE_MAX_SPEAKERS, [&](int a, int b) { return activity[a] > activity[b]; });

    bool active[DIARIZE_MAX_SPEAKERS] = {};
    for (int i = 0; i < keep; ++i) active[order[i]] = true;

    for (size_t f = 0; f < total_frames; ++f)
        for (int s = 0; s < DIARIZE_MAX_SPEAKERS; ++s)
            if (!active[s]) scores[f * DIARIZE_MAX_SPEAKERS + s] = 0.0f;
}

static void apply_threshold(std::vector<float>& scores, float threshold) {
    for (float& v : scores)
        v = (v >= threshold) ? v : 0.0f;
}

extern "C" {

int cactus_diarize(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("diarize", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("diarize", "Invalid parameters: response_buffer or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("diarize", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR("diarize", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        return -1;
    }

    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR("diarize", "Invalid pcm_buffer_size: " << pcm_buffer_size);
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* pyannote = dynamic_cast<PyAnnoteModel*>(handle->model.get());
        if (!pyannote) {
            CACTUS_LOG_ERROR("diarize", "Model is not a PyAnnote diarization model");
            handle_error_response("Model is not a PyAnnote diarization model", response_buffer, buffer_size);
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
            CACTUS_LOG_ERROR("diarize", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        const DiarizeOptions opts = parse_diarize_options(options_json ? options_json : "");
        auto scores = pyannote->diarize(audio.data(), audio.size(), opts.step_samples);
        audio = {};

        const size_t total_frames = scores.size() / DIARIZE_MAX_SPEAKERS;

        if (opts.num_speakers > 0 || opts.min_speakers > 0 || opts.max_speakers > 0)
            apply_speaker_filter(scores, total_frames, opts);

        if (opts.threshold >= 0.0f)
            apply_threshold(scores, opts.threshold);

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"num_speakers\":" << DIARIZE_MAX_SPEAKERS << ",";
        json << "\"scores\":[";
        json << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < scores.size(); ++i) {
            if (i > 0) json << ",";
            json << scores[i];
        }
        json << "],";
        json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
        json << "}";

        std::string response = json.str();
        if (response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("diarize", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, response.c_str());
        return static_cast<int>(response.length());

    } catch (const std::exception& e) {
        last_error_message = "Exception during diarization: " + std::string(e.what());
        CACTUS_LOG_ERROR("diarize", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during diarization";
        CACTUS_LOG_ERROR("diarize", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
}

}
