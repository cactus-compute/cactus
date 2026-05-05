#include "../cactus_engine.h"
#include "utils.h"
#include "cactus_kernels.h"
#include "wav.h"
#include <cstring>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

static std::vector<float> compute_mel_from_wav(const std::string& wav_path, size_t mel_bins) {
    AudioFP32 audio = load_wav(wav_path);
    std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);

    auto cfg = cactus::audio::get_whisper_spectrogram_config();
    const size_t num_mel_filters = std::max<size_t>(1, mel_bins);
    const bool is_v3 = num_mel_filters > 80;
    if (is_v3) cactus::audio::apply_whisper_v3_overrides(cfg);

    int norm_type = is_v3 ? 1 : 1;
    int scale_type = is_v3 ? 2 : 2;
    std::vector<float> mel = cactus::audio::compute_spectrogram_graph(
        waveform_16k, cfg, num_mel_filters, 0.0f, 8000.0f, 16000, norm_type, scale_type);
    if (mel.empty()) return mel;

    return cactus::audio::normalize_whisper_mel(mel, num_mel_filters, is_v3);
}

extern "C" {

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim,
    bool normalize
) {
    if (!model || !text || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("embed", "Invalid parameters for text embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            CACTUS_LOG_ERROR("embed", "Model tokenizer not available");
            return -1;
        }

        auto tokens = tokenizer->encode(text);
        if (tokens.empty()) {
            CACTUS_LOG_ERROR("embed", "Failed to tokenize input text");
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_embeddings(tokens, true, normalize);
        if (embeddings.empty()) {
            CACTUS_LOG_ERROR("embed", "Embedding returned empty result");
            return -1;
        }

        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes");
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding";
        CACTUS_LOG_ERROR("embed", last_error_message);
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
    if (!model || !image_path || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("image_embed", "Invalid parameters for image embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::vector<float> embeddings = handle->model->get_image_embeddings(image_path);
        if (embeddings.empty()) {
            CACTUS_LOG_ERROR("image_embed", "Image embedding returned empty result");
            return -1;
        }
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("image_embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes");
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("image_embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image embedding";
        CACTUS_LOG_ERROR("image_embed", last_error_message);
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
    if (!model || !audio_path || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("audio_embed", "Invalid parameters for audio embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto mel_bins = compute_mel_from_wav(audio_path, handle->model->get_config().num_mel_bins);
        if (mel_bins.empty()) {
            last_error_message = "Failed to compute mel spectrogram";
            CACTUS_LOG_ERROR("audio_embed", last_error_message << " for: " << audio_path);
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_audio_embeddings(mel_bins);
        if (embeddings.empty()) {
            CACTUS_LOG_ERROR("audio_embed", "Audio embedding returned empty result");
            return -1;
        }
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("audio_embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes");
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("audio_embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio embedding";
        CACTUS_LOG_ERROR("audio_embed", last_error_message);
        return -1;
    }
}

}
