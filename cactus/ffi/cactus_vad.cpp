#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model.h"
#include "../../libs/audio/wav.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;


static SileroVADModel::SpeechTimestampsOptions parse_speech_timestamps_options(const std::string& json) {
    SileroVADModel::SpeechTimestampsOptions options;

    if (json.empty()) return options;

    size_t pos = json.find("\"threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"neg_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.neg_threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"min_speech_duration_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.min_speech_duration_ms = std::stoi(json.substr(pos));
    }

    pos = json.find("\"max_speech_duration_s\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.max_speech_duration_s = std::stof(json.substr(pos));
    }

    pos = json.find("\"min_silence_duration_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.min_silence_duration_ms = std::stoi(json.substr(pos));
    }

    pos = json.find("\"speech_pad_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.speech_pad_ms = std::stoi(json.substr(pos));
    }

    pos = json.find("\"window_size_samples\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.window_size_samples = std::stoi(json.substr(pos));
    }

    pos = json.find("\"min_silence_at_max_speech\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.min_silence_at_max_speech = std::stoi(json.substr(pos));
    }

    pos = json.find("\"use_max_poss_sil_at_max_speech\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        std::string value = json.substr(pos, 5);
        options.use_max_poss_sil_at_max_speech = (value.find("true") != std::string::npos);
    }

    pos = json.find("\"sampling_rate\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        options.sampling_rate = std::stoi(json.substr(pos));
    }

    return options;
}


extern "C" {

int cactus_vad(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (validate_audio_params("vad", model, response_buffer, buffer_size, audio_file_path, pcm_buffer, pcm_buffer_size) != 0)
        return -1;

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* vad_model = static_cast<SileroVADModel*>(handle->model.get());

        SileroVADModel::SpeechTimestampsOptions options = parse_speech_timestamps_options(options_json ? options_json : "");

        std::vector<float> audio;
        if (audio_file_path == nullptr) {
            audio = pcm_to_float(pcm_buffer, pcm_buffer_size);
        } else {
            AudioFP32 wav_audio = load_wav(audio_file_path);
            audio = resample_to_16k_fp32(wav_audio.samples, wav_audio.sample_rate);
        }

        if (audio.empty()) {
            last_error_message = "Failed to load audio or audio is empty";
            CACTUS_LOG_ERROR("vad", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        auto segments = vad_model->get_speech_timestamps(audio, options);

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"segments\":[";

        for (size_t i = 0; i < segments.size(); ++i) {
            if (i > 0) json << ",";
            json << "{";
            json << "\"start\":" << segments[i].start << ",";
            json << "\"end\":" << segments[i].end;
            json << "}";
        }

        json << "],";
        json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
        json << "}";

        std::string response = json.str();
        if (response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("vad", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, response.c_str());
        return static_cast<int>(response.length());

    } catch (const std::exception& e) {
        last_error_message = "Exception during VAD processing: " + std::string(e.what());
        CACTUS_LOG_ERROR("vad", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during VAD processing";
        CACTUS_LOG_ERROR("vad", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
}

}
