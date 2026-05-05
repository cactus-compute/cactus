#include "../cactus_engine.h"
#include "utils.h"
#include "cactus_kernels.h"
#include "wav.h"
#include <cstring>

using namespace cactus::ffi;

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
    if (!model || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("transcribe", "Invalid parameters");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("transcribe", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }

    // Load audio file into PCM if a file path was given
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
        // Convert float samples to int16 PCM
        file_pcm.resize(audio.samples.size() * sizeof(int16_t));
        int16_t* out = reinterpret_cast<int16_t*>(file_pcm.data());
        for (size_t i = 0; i < audio.samples.size(); i++) {
            float clamped = std::max(-1.0f, std::min(1.0f, audio.samples[i]));
            out[i] = static_cast<int16_t>(clamped * 32767.0f);
        }
        pcm_data = file_pcm.data();
        pcm_size = file_pcm.size();
    }

    // Build a transcription message and route through cactus_complete
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
        nullptr,        // no tools
        callback,
        user_data,
        pcm_data,
        pcm_size
    );
}

} // extern "C"
