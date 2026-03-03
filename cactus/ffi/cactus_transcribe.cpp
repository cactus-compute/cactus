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
static constexpr size_t MAX_CHUNK_SAMPLES = WHISPER_SAMPLE_RATE * 30;
static constexpr size_t MAX_CONTEXT_WORDS = 64;

static std::vector<float> normalize_mel(std::vector<float>& mel, size_t n_mels) {
    size_t n_frames = mel.size() / n_mels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : mel)
        if (v > max_val) max_val = v;

    float min_allowed = max_val - 8.0f;
    for (float& v : mel) {
        if (v < min_allowed) v = min_allowed;
        v = (v + 4.0f) * 0.25f;
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
    return std::move(mel);
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
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, error_msg.c_str());
        return -1;
    }
    if (!prompt || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("transcribe", "Invalid parameters: prompt, response_buffer, or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, "Invalid parameters");
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR("transcribe", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, "No audio input provided");
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR("transcribe", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, "Cannot provide both audio_file_path and pcm_buffer");
        return -1;
    }

    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR("transcribe", "Invalid pcm_buffer_size: " << pcm_buffer_size);
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, "pcm_buffer_size must be even and at least 2 bytes");
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
            opts.c_str(), temperature,
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
                        cloud_handoff_threshold = std::stof(opts.c_str() + pos);
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

        std::vector<float> audio_samples;
        if (audio_file_path == nullptr) {
            const int16_t* pcm_samples = reinterpret_cast<const int16_t*>(pcm_buffer);
            size_t num_samples = pcm_buffer_size / 2;
            std::vector<float> waveform_fp32(num_samples);
            constexpr float inv_32768 = 1.0f / 32768.0f;
            for (size_t i = 0; i < num_samples; i++)
                waveform_fp32[i] = static_cast<float>(pcm_samples[i]) * inv_32768;
            audio_samples = resample_to_16k_fp32(waveform_fp32, WHISPER_SAMPLE_RATE);
        } else {
            AudioFP32 audio = load_wav(audio_file_path);
            audio_samples = resample_to_16k_fp32(audio.samples, audio.sample_rate);
        }

        std::vector<std::vector<float>> chunks;

        if (use_vad) {
            auto* vad = static_cast<SileroVADModel*>(handle->vad_model.get());
            auto vad_segments = vad->get_speech_timestamps(audio_samples, {});
            chunks.reserve(vad_segments.size());

            std::vector<float> current;
            for (const auto& seg : vad_segments) {
                size_t end = std::min(seg.end, audio_samples.size());
                if (current.size() + (end - seg.start) > MAX_CHUNK_SAMPLES) {
                    chunks.emplace_back(std::move(current));
                    current = {};
                }
                current.insert(
                    current.end(),
                    audio_samples.begin() + seg.start,
                    audio_samples.begin() + end
                );
            }

            if (!current.empty()) {
                chunks.emplace_back(std::move(current));
            }

            if (chunks.empty()) {
                CACTUS_LOG_DEBUG("transcribe", "VAD detected only silence, returning empty transcription");
                auto vad_end_time = std::chrono::high_resolution_clock::now();
                double vad_total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(vad_end_time - start_time).count() / 1000.0;
                std::string json = construct_response_json("", {}, 0.0, vad_total_time_ms, 0.0, 0.0, 0, 0, 1.0f);
                if (json.size() >= buffer_size) {
                    handle_error_response("Response buffer too small", response_buffer, buffer_size);
                    cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, 0.0, "Response buffer too small");
                    return -1;
                }
                cactus::telemetry::recordTranscription(handle->model_name.c_str(), true, 0.0, 0.0, vad_total_time_ms, 0, get_ram_usage_mb(), "");
                std::strcpy(response_buffer, json.c_str());
                return static_cast<int>(json.size());
            }
        } else {
            chunks.reserve((audio_samples.size() + MAX_CHUNK_SAMPLES - 1) / MAX_CHUNK_SAMPLES);
            for (size_t start = 0; start < audio_samples.size(); start += MAX_CHUNK_SAMPLES) {
                size_t end = std::min(start + MAX_CHUNK_SAMPLES, audio_samples.size());
                chunks.emplace_back(audio_samples.begin() + start, audio_samples.begin() + end);
            }
        }

        auto cfg = is_parakeet ? get_parakeet_spectrogram_config() : get_whisper_spectrogram_config();
        size_t mel_bins = 0;
        AudioProcessor ap;
        if (!is_moonshine) {
            if (is_parakeet) {
                mel_bins = std::max<size_t>(1, static_cast<size_t>(handle->model->get_config().num_mel_bins));
                ap.init_mel_filters(cfg.n_fft / 2 + 1, mel_bins, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
            } else {
                ap.init_mel_filters(cfg.n_fft / 2 + 1, 80, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
            }
        }

        auto* tokenizer = handle->model->get_tokenizer();
        if (!tokenizer) {
            CACTUS_LOG_ERROR("transcribe", "Tokenizer unavailable");
            handle_error_response("Tokenizer unavailable", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, 0.0, "Tokenizer unavailable");
            return -1;
        }

        std::vector<uint32_t> initial_tokens = tokenizer->encode(std::string(prompt));
        if (initial_tokens.empty() && !is_moonshine && !is_parakeet) {
            CACTUS_LOG_ERROR("transcribe", "Decoder input tokens empty after encoding prompt");
            handle_error_response("Decoder input tokens empty", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, 0.0, "Decoder input tokens empty");
            return -1;
        }

        float max_tps = handle->model->get_config().default_max_tps;
        if (max_tps < 0) max_tps = 100;

        const std::vector<std::vector<uint32_t>> stop_token_sequences = {{ tokenizer->get_eos_token() }};

        double time_to_first_token = 0.0;
        size_t completion_tokens = 0;
        std::string final_text;
        float total_entropy_sum = 0.0f;
        float max_token_entropy_norm = 0.0f;

        auto sop = tokenizer->encode("<|startofprev|>");
        auto sot = tokenizer->encode("<|startoftranscript|>");
        auto sot_it = std::search(initial_tokens.begin(), initial_tokens.end(), sot.begin(), sot.end());
        const auto sot_begin = sot_it != initial_tokens.end() ? sot_it : initial_tokens.begin();

        for (auto& raw : chunks) {
            if (handle->should_stop || completion_tokens >= max_tokens) break;

            std::vector<float> chunk_audio = std::move(raw);
            const float chunk_length_sec = static_cast<float>(chunk_audio.size()) / static_cast<float>(WHISPER_SAMPLE_RATE);

            std::vector<uint32_t> tokens;
            if (final_text.empty() || is_parakeet || is_moonshine) {
                tokens = initial_tokens;
            } else {
                size_t word_count = 0, pos = final_text.size();
                while (pos > 0 && word_count < MAX_CONTEXT_WORDS) {
                    while (pos > 0 && std::isspace((unsigned char)final_text[pos - 1])) --pos;
                    while (pos > 0 && !std::isspace((unsigned char)final_text[pos - 1])) --pos;
                    ++word_count;
                }
                tokens = sop;
                auto ctx = tokenizer->encode(final_text.substr(pos));
                tokens.insert(tokens.end(), ctx.begin(), ctx.end());
                tokens.insert(tokens.end(), sot_begin, initial_tokens.end());
            }

            if (!is_moonshine) {
                if (is_parakeet) {
                    size_t waveform_samples = chunk_audio.size();
                    apply_preemphasis(chunk_audio, 0.97f);
                    chunk_audio = ap.compute_spectrogram(chunk_audio, cfg);
                    normalize_parakeet_log_mel(chunk_audio, mel_bins);
                    size_t valid_frames = waveform_samples / cfg.hop_length;
                    if (valid_frames == 0) valid_frames = 1;
                    trim_mel_frames(chunk_audio, mel_bins, valid_frames);
                } else {
                    std::vector<float> mel = ap.compute_spectrogram(chunk_audio, cfg);
                    chunk_audio = normalize_mel(mel, 80);
                }
            }

            if (chunk_audio.empty()) {
                CACTUS_LOG_DEBUG("transcribe", "Chunk audio features empty, skipping");
                continue;
            }

            CACTUS_LOG_DEBUG("transcribe", "Chunk audio features size: " << chunk_audio.size());

            size_t chunk_max_tokens = max_tokens - completion_tokens;
            if (!is_parakeet) {
                size_t max_allowed = tokens.size() < WHISPER_MAX_DECODER_POSITIONS ?
                    WHISPER_MAX_DECODER_POSITIONS - tokens.size() : 0;
                if (chunk_max_tokens > max_allowed) chunk_max_tokens = max_allowed;
            }
            size_t max_tps_tokens = std::max<size_t>(1, static_cast<size_t>(chunk_length_sec * max_tps));
            if (chunk_max_tokens > max_tps_tokens) chunk_max_tokens = max_tps_tokens;

            tokens.reserve(tokens.size() + chunk_max_tokens);
            std::vector<uint32_t> generated_tokens;
            generated_tokens.reserve(chunk_max_tokens);

            for (size_t i = 0; i < chunk_max_tokens; ++i) {
                if (handle->should_stop) break;

                float token_entropy = 0.0f;
                uint32_t next_token = handle->model->decode_with_audio(tokens, chunk_audio, temperature, top_p, top_k, "", &token_entropy);

                if (completion_tokens == 0) [[unlikely]] {
                    auto t_first = std::chrono::high_resolution_clock::now();
                    time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
                }

                total_entropy_sum += token_entropy;
                if (token_entropy > max_token_entropy_norm) max_token_entropy_norm = token_entropy;

                generated_tokens.emplace_back(next_token);
                tokens.emplace_back(next_token);
                completion_tokens++;

                std::string piece = tokenizer->decode({ next_token });
                final_text += piece;
                if (callback) callback(piece.c_str(), next_token, user_data);

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;
            }

            cactus_reset(model);
        }

        float mean_entropy = completion_tokens > 0 ? total_entropy_sum / static_cast<float>(completion_tokens) : 0.0f;
        float confidence = 1.0f - mean_entropy;

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        size_t prompt_tokens = initial_tokens.size();
        double prefill_tps = time_to_first_token > 0 ? (prompt_tokens * 1000.0) / time_to_first_token : 0.0;
        double decode_time_ms = std::max(0.0, total_time_ms - time_to_first_token);
        double decode_tps = (completion_tokens > 1 && decode_time_ms > 0.0) ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;

        const std::vector<std::string> tokens_to_remove = {
            "<|startoftranscript|>", "</s>", "<pad>"
        };
        for (const auto& token : tokens_to_remove) {
            size_t pos = 0;
            while ((pos = final_text.find(token, pos)) != std::string::npos)
                final_text.erase(pos, token.length());
        }
        if (!final_text.empty() && final_text[0] == ' ')
            final_text.erase(0, 1);

        const bool cloud_handoff = !final_text.empty() && final_text.length() > 5 &&
            cloud_handoff_threshold > 0.0f && max_token_entropy_norm > cloud_handoff_threshold;

        std::string json = construct_response_json(final_text, {}, time_to_first_token, total_time_ms, prefill_tps, decode_tps, prompt_tokens, completion_tokens, confidence, cloud_handoff);

        if (json.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            cactus::telemetry::recordTranscription(handle->model_name.c_str(), false, 0.0, 0.0, 0.0, 0, 0.0, "Response buffer too small");
            return -1;
        }

        cactus::telemetry::recordTranscription(handle->model_name.c_str(), true, time_to_first_token, decode_tps, total_time_ms, static_cast<int>(completion_tokens), get_ram_usage_mb(), "");

        std::strcpy(response_buffer, json.c_str());

        return static_cast<int>(json.size());
    }
    catch (const std::exception& e) {
        CACTUS_LOG_ERROR("transcribe", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, e.what());
        return -1;
    }
    catch (...) {
        CACTUS_LOG_ERROR("transcribe", "Unknown exception during transcription");
        handle_error_response("Unknown error in transcribe", response_buffer, buffer_size);
        cactus::telemetry::recordTranscription(model ? static_cast<CactusModelHandle*>(model)->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, "Unknown error in transcribe");
        return -1;
    }
}

}