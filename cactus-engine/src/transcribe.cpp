#include "../cactus_engine.h"
#include "utils.h"
#include "cloud.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include "wav.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <mutex>

using namespace cactus::ffi;

namespace {

constexpr float kWhisperTimestampStepSec = 0.02f;

uint32_t whisper_timestamp_begin(cactus::engine::Tokenizer* tokenizer) {
    return tokenizer->encode("<|notimestamps|>")[0] + 1;
}

std::vector<TranscriptSegment> parse_whisper_timestamp_segments(
    cactus::engine::Tokenizer* tokenizer, const std::vector<uint32_t>& tokens,
    uint32_t timestamp_begin, std::string& clean_text) {
    std::vector<TranscriptSegment> segments;
    std::vector<uint32_t> text_tokens;
    std::vector<uint32_t> flat;
    float seg_start = -1.0f;
    auto flush = [&](float end_sec) {
        if (!text_tokens.empty()) {
            std::string text = tokenizer->decode(text_tokens);
            if (!text.empty() && text[0] == ' ') text.erase(0, 1);
            segments.push_back({seg_start, end_sec, text});
        }
        text_tokens.clear();
    };
    for (uint32_t tok : tokens) {
        if (tok >= timestamp_begin) {
            const float t = static_cast<float>(tok - timestamp_begin) * kWhisperTimestampStepSec;
            if (seg_start < 0.0f) seg_start = t;
            else { flush(t); seg_start = t; }
        } else {
            text_tokens.push_back(tok);
            flat.push_back(tok);
        }
    }
    flush(seg_start);
    clean_text = tokenizer->decode(flat);
    return segments;
}

} // namespace

extern "C" {

CACTUS_FFI_EXPORT int cactus_preprocess_audio_features(
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
            // The Python transpiled runtime shape-specializes Gemma4 audio to
            // its requested capacity (currently capped at 30s). Do not apply
            // the native interactive model's shorter audio_soft_tokens limit
            // here, otherwise long-file multimodal prompts are silently
            // truncated before they reach the graph bundle.
            cfg.audio_soft_tokens = 0;
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
            cactus::audio::pad_or_trim_whisper_waveform(audio_samples);
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
    struct MetalTrimGuard {
        ~MetalTrimGuard() { cactus_metal_trim_prefill_cache(); }
    } metal_trim_guard;
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
        const size_t original_audio_sample_count = audio_samples.size();

        const size_t mel_bins = std::max<size_t>(1, static_cast<size_t>(handle->model->get_config().num_mel_bins));
        std::vector<float> audio_features;
        std::vector<float> handoff_audio_samples;
        if (is_parakeet) {
            auto cfg = cactus::audio::get_parakeet_spectrogram_config();
            const size_t waveform_samples = audio_samples.size();
            handoff_audio_samples = audio_samples;
            cactus::audio::apply_preemphasis(audio_samples, 0.97f);
            audio_features = cactus::audio::compute_spectrogram_graph(
                audio_samples, cfg, mel_bins, 0.0f, 8000.0f,
                cactus::audio::WHISPER_SAMPLE_RATE, 0, 0);
            cactus::audio::normalize_parakeet_log_mel(audio_features, mel_bins);
            size_t valid_frames = waveform_samples / cfg.hop_length;
            if (valid_frames == 0) valid_frames = 1;
            cactus::audio::trim_mel_frames(audio_features, mel_bins, valid_frames);
        } else {
            cactus::audio::pad_or_trim_whisper_waveform(audio_samples);
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

        const bool want_timestamps = is_whisper && options.timestamps;
        const uint32_t ts_begin = want_timestamps ? whisper_timestamp_begin(tokenizer) : 0;

        std::vector<uint32_t> tokens;
        if (prompt && prompt[0] != '\0') {
            tokens = tokenizer->encode(prompt);
        } else if (is_whisper) {
            tokens = handle->model->get_config().decoder_prompt_token_ids;
            std::string language = json_string_field(options_json ? options_json : "", "language");
            if (!language.empty()) {
                std::vector<uint32_t> language_token = tokenizer->encode("<|" + language + "|>");
                if (language_token.size() == 1) {
                    for (uint32_t& token : tokens) {
                        std::string piece = tokenizer->decode({token});
                        if (piece.size() == 6 && piece[0] == '<' && piece[1] == '|' &&
                            piece[4] == '|' && piece[5] == '>' &&
                            std::islower((unsigned char)piece[2]) && std::islower((unsigned char)piece[3])) {
                            token = language_token[0];
                            break;
                        }
                    }
                }
            }
            if (want_timestamps)
                tokens.erase(std::remove(tokens.begin(), tokens.end(), ts_begin - 1), tokens.end());
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
            static_cast<float>(original_audio_sample_count) / static_cast<float>(cactus::audio::WHISPER_SAMPLE_RATE);
        if (options.max_tokens == 100 && (!options_json || std::string(options_json).find("\"max_tokens\"") == std::string::npos)) {
            options.max_tokens = std::max<size_t>(100, static_cast<size_t>(audio_length_sec * (is_parakeet ? 30.0f : 20.0f)));
        }

        constexpr size_t WHISPER_MAX_DECODER_POSITIONS = 448;
        if (is_whisper && tokens.size() < WHISPER_MAX_DECODER_POSITIONS) {
            options.max_tokens = std::min(options.max_tokens, WHISPER_MAX_DECODER_POSITIONS - tokens.size());
        }

        std::string final_text;
        std::vector<TranscriptSegment> segments;
        std::vector<uint32_t> generated_tokens;
        generated_tokens.reserve(options.max_tokens);
        const size_t prompt_tokens = tokens.size();
        double time_to_first_token = 0.0;
        float total_entropy_sum = 0.0f;

        if (is_parakeet) {
            generated_tokens = handle->model->transcribe_parakeet_tdt(
                audio_features, nullptr, true, 0, &handle->should_stop);
            auto t_first = std::chrono::high_resolution_clock::now();
            time_to_first_token =
                std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
            final_text = tokenizer->decode(generated_tokens);
            if (callback) {
                for (uint32_t tok : generated_tokens) {
                    std::string piece = tokenizer->decode({tok});
                    callback(piece.c_str(), tok, user_data);
                }
            }
        } else if (is_whisper) {
            generated_tokens = handle->model->transcribe_whisper_seq2seq(
                audio_features,
                tokens,
                options.max_tokens,
                stop_token_sequences,
                &handle->should_stop,
                want_timestamps ? static_cast<int64_t>(ts_begin) - 1 : -1);
            auto t_first = std::chrono::high_resolution_clock::now();
            time_to_first_token =
                std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
            for (const auto& stop_seq : stop_token_sequences) {
                if (stop_seq.empty() || generated_tokens.size() < stop_seq.size()) continue;
                if (std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin())) {
                    generated_tokens.resize(generated_tokens.size() - stop_seq.size());
                    break;
                }
            }
            if (want_timestamps) {
                segments = parse_whisper_timestamp_segments(tokenizer, generated_tokens, ts_begin, final_text);
            } else {
                final_text = tokenizer->decode(generated_tokens);
            }
            if (callback) {
                for (uint32_t tok : generated_tokens) {
                    if (want_timestamps && tok >= ts_begin) continue;
                    std::string piece = tokenizer->decode({tok});
                    callback(piece.c_str(), tok, user_data);
                }
            }
        } else {
            for (size_t i = 0; i < options.max_tokens; ++i) {
                if (handle->should_stop) break;

                float token_entropy = 0.0f;
                uint32_t next_token = handle->model->decode_with_audio(
                    tokens, std::vector<std::vector<float>>{audio_features},
                    options.temperature, options.top_p, options.top_k,
                    "", &token_entropy,
                    options.min_p, options.repetition_penalty,
                    nullptr, nullptr);

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

        bool cloud_handoff_used = false;
        std::string handoff_reason;
        float reported_threshold = -1.0f;
        if (is_parakeet) {
            const bool cloud_disabled = env_flag_enabled("CACTUS_DISABLE_CLOUD_HANDOFF");
            const bool cloud_eligible =
                !cloud_disabled && !handle->cloud_handoff_disabled && options.auto_handoff;
            float threshold = options.confidence_threshold;
            if (threshold < 0.0f) {
                const float model_default = handle->model->get_config().default_cloud_handoff_threshold;
                threshold = handle->model->has_handoff_probe()
                    ? 0.50f
                    : (model_default > 0.0f ? model_default : 0.7f);
            }
            reported_threshold = threshold;

            if (!options.auto_handoff) {
                handoff_reason = "handoff off";
            } else if (cloud_disabled) {
                handoff_reason = "disabled (env)";
            } else if (handle->cloud_handoff_disabled) {
                handoff_reason = "disabled (auth failed earlier)";
            }

            if (cloud_eligible && handle->model->has_handoff_probe_rollout()) {
                float p_wrong = handle->model->handoff_probe_wrong_probability();
                if (std::isfinite(p_wrong)) {
                    confidence = std::max(0.0f, std::min(1.0f, 1.0f - p_wrong));
                    CACTUS_LOG_DEBUG("cloud_handoff", "Parakeet handoff probe p_wrong="
                        << p_wrong << " confidence=" << confidence);
                    if (confidence < threshold) {
                        CACTUS_LOG_WARN("cloud_handoff", "Cloud transcription handoff triggered: p_wrong="
                            << p_wrong << " confidence=" << confidence << " threshold=" << threshold);
                        std::vector<int16_t> pcm16(handoff_audio_samples.size());
                        for (size_t i = 0; i < handoff_audio_samples.size(); ++i) {
                            float clamped = std::max(-1.0f, std::min(1.0f, handoff_audio_samples[i]));
                            pcm16[i] = static_cast<int16_t>(std::lround(clamped * 32767.0f));
                        }
                        std::vector<uint8_t> wav = cactus::ffi::cloud_build_wav(
                            reinterpret_cast<const uint8_t*>(pcm16.data()), pcm16.size() * sizeof(int16_t));
                        std::string audio_b64 = cactus::ffi::cloud_base64_encode(wav.data(), wav.size());
                        std::string cloud_key = cactus::ffi::resolve_cloud_api_key(nullptr);
                        CloudResponse cloud = cactus::ffi::cloud_transcribe_request(
                            audio_b64, final_text,
                            static_cast<long>(options.cloud_timeout_ms / 1000),
                            cloud_key.empty() ? nullptr : cloud_key.c_str());
                        if (cloud.used_cloud && !cloud.transcript.empty()) {
                            final_text = cloud.transcript;
                            cloud_handoff_used = true;
                            handoff_reason = "low confidence (probe)";
                        } else {
                            if (cloud.error.rfind("http_401", 0) == 0 || cloud.error.rfind("http_403", 0) == 0) {
                                handle->cloud_handoff_disabled = true;
                            }
                            handoff_reason = "handoff failed: " + cloud.error;
                            CACTUS_LOG_WARN("cloud_handoff", "Cloud transcription failed, keeping local output: "
                                << cloud.error);
                        }
                    }
                }
            }
            if (handoff_reason.empty()) {
                handoff_reason = (confidence >= threshold) ? "above threshold" : "kept local";
            }
        }

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
            confidence, cloud_handoff_used, "", segments, "",
            reported_threshold, handoff_reason);
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
