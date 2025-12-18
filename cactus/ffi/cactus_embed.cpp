#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_telemetry.h"
#include "../../libs/audio/wav.h"
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

static std::vector<float> compute_mel_from_wav(const std::string& wav_path) {
    AudioFP32 audio = load_wav(wav_path);
    std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);

    auto cfg = get_whisper_spectrogram_config();
    const size_t num_mel_filters = 80;
    const size_t num_frequency_bins = cfg.n_fft / 2 + 1;

    AudioProcessor ap;
    ap.init_mel_filters(num_frequency_bins, num_mel_filters, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
    std::vector<float> mel = ap.compute_spectrogram(waveform_16k, cfg);

    if (mel.empty()) return mel;

    size_t n_mels = num_mel_filters;
    size_t n_frames = mel.size() / n_mels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : mel) if (v > max_val) max_val = v;

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

        std::vector<uint32_t> tokens = tokenizer->encode(text);
        if (tokens.empty()) {
            CACTUS_LOG_ERROR("embed", "Tokenization produced empty result");
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_embeddings(tokens, true, normalize);
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes, got " << buffer_size);
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            true,
            ""
        );

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("embed", "Exception: " << e.what());

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding";
        CACTUS_LOG_ERROR("embed", last_error_message);

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

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

        CACTUS_LOG_DEBUG("image_embed", "Processing image: " << image_path);
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

        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            true,
            ""
        );

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("image_embed", "Exception: " << e.what());

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image embedding";
        CACTUS_LOG_ERROR("image_embed", last_error_message);

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

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

        CACTUS_LOG_DEBUG("audio_embed", "Processing audio: " << audio_path);
        auto mel_bins = compute_mel_from_wav(audio_path);
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

        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            true,
            ""
        );

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("audio_embed", "Exception: " << e.what());

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio embedding";
        CACTUS_LOG_ERROR("audio_embed", last_error_message);

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordEmbedding(
            handle->model_name,
            false,
            last_error_message
        );

        return -1;
    }
}

}
