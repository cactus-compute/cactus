#include "cactus_ffi.h"
#include "cactus_cloud.h"
#include "cactus_utils.h"
#include "telemetry/telemetry.h"
#include <algorithm>
#include <cstring>
#include <regex>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <future>
#include <chrono>
#include <deque>
#include <mutex>
#include <sstream>

using namespace cactus::ffi;
using cactus::audio::WHISPER_SAMPLE_RATE;
using cactus::audio::apply_preemphasis;
using cactus::audio::get_parakeet_spectrogram_config;
using cactus::audio::normalize_parakeet_log_mel;
using cactus::audio::trim_mel_frames;

double json_number(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return 0.0;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;
    try { return std::stod(json.substr(start, end - start)); }
    catch (...) { return 0.0; }
}

std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t start = pos + pattern.size();

    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '"') return {};
    size_t q1 = start;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

inline std::string escape_json(const std::string& s) {
    return escape_json_string(s);
}

bool json_bool(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start + 4 <= json.size() && json.substr(start, 4) == "true") return true;
    return false;
}

static std::string ensure_json_bool_option(const std::string& json,
                                           const std::string& key,
                                           bool value) {
    const std::string field = "\"" + key + "\"";
    if (json.find(field) != std::string::npos) return json;

    const std::string bool_str = value ? "true" : "false";
    if (json.empty()) {
        return "{\"" + key + "\":" + bool_str + "}";
    }

    std::string out = json;
    const size_t close = out.rfind('}');
    const size_t open = out.find('{');
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return json;
    }

    size_t content_start = open + 1;
    while (content_start < close && std::isspace(static_cast<unsigned char>(out[content_start]))) {
        ++content_start;
    }
    const bool has_entries = content_start < close;
    const std::string insertion = (has_entries ? "," : "") + field + ":" + bool_str;
    out.insert(close, insertion);
    return out;
}

std::string json_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "[]";
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '[') return "[]";
    int depth = 1;
    size_t end = start + 1;
    while (end < json.size() && depth > 0) {
        if (json[end] == '[') depth++;
        else if (json[end] == ']') depth--;
        end++;
    }
    return json.substr(start, end - start);
}

static std::string suppress_unwanted_text(const std::string& text) {
    static const std::regex pattern(R"(\([^)]*\)|\[[^\]]*\]|\.\.\.)");
    std::string result = std::regex_replace(text, pattern, "");

    size_t start = result.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = result.find_last_not_of(" \t\n\r");
    return result.substr(start, end - start + 1);
}

static bool is_pcm_chunk_silent(const uint8_t* pcm_buffer, size_t pcm_buffer_size, float rms_threshold) {
    if (!pcm_buffer || pcm_buffer_size < sizeof(int16_t)) return true;

    const size_t sample_count = pcm_buffer_size / sizeof(int16_t);
    const auto* samples = reinterpret_cast<const int16_t*>(pcm_buffer);
    double sum_sq = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        const double normalized = static_cast<double>(samples[i]) / 32768.0;
        sum_sq += normalized * normalized;
    }

    const double rms = std::sqrt(sum_sq / static_cast<double>(sample_count));
    return rms < static_cast<double>(rms_threshold);
}

static void parse_stream_transcribe_init_options(const std::string& json,
                                                 double& confirmation_threshold,
                                                 size_t& min_chunk_size,
                                                 bool& telemetry_enabled,
                                                 std::string& language) {
    confirmation_threshold = 0.99;
    min_chunk_size = 32000;
    telemetry_enabled = true;
    language = "en";

    if (json.empty()) return;

    size_t pos = json.find("\"confirmation_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        confirmation_threshold = std::stod(json.substr(pos));
    }

    pos = json.find("\"min_chunk_size\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        min_chunk_size = static_cast<size_t>(std::stod(json.substr(pos)));
    }

    pos = json.find("\"telemetry_enabled\"");
    if (pos != std::string::npos) {
        telemetry_enabled = json_bool(json, "telemetry_enabled");
    }

    language = json_string(json, "language");
    if (language.empty()) language = "en";
}

static bool build_parakeet_features_from_pcm(
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size,
    const cactus::engine::Model* model,
    std::vector<float>& out_features) {
    if (!pcm_buffer || pcm_buffer_size < sizeof(int16_t) || !model) {
        return false;
    }

    std::vector<float> waveform = cactus::audio::pcm_buffer_to_float_samples(
        pcm_buffer, pcm_buffer_size);
    if (waveform.empty()) {
        return false;
    }

    const auto cfg = get_parakeet_spectrogram_config();
    const size_t mel_bins = std::max<size_t>(
        1, static_cast<size_t>(model->get_config().num_mel_bins));
    cactus::engine::AudioProcessor processor;
    processor.init_mel_filters(
        cfg.n_fft / 2 + 1,
        mel_bins,
        0.0f,
        8000.0f,
        WHISPER_SAMPLE_RATE);

    const size_t waveform_samples = waveform.size();
    apply_preemphasis(waveform, 0.97f);
    out_features = processor.compute_spectrogram(waveform, cfg);
    normalize_parakeet_log_mel(out_features, mel_bins);
    size_t valid_frames = waveform_samples / cfg.hop_length;
    if (valid_frames == 0) valid_frames = 1;
    trim_mel_frames(out_features, mel_bins, valid_frames);
    return !out_features.empty();
}

static size_t parakeet_tdt_pcm_bytes_to_encoder_frames(
    const cactus::engine::Model* model,
    size_t pcm_bytes) {
    if (!model) return 0;
    const auto cfg = get_parakeet_spectrogram_config();
    const double sample_count = static_cast<double>(pcm_bytes) / 2.0;
    const double feature_frames = sample_count / static_cast<double>(cfg.hop_length);
    const double subsampling = std::max<uint32_t>(1, model->get_config().subsampling_factor);
    return static_cast<size_t>(std::llround(feature_frames / subsampling));
}

struct CactusStreamTranscribeHandle {
    CactusModelHandle* model_handle;

    struct CactusStreamTranscribeOptions {
        double confirmation_threshold;
        size_t min_chunk_size;
        std::string language;
    } options;
    std::string transcribe_options_json;
    bool has_custom_vocabulary_bias = false;
    std::vector<std::string> custom_vocabulary;

    std::vector<uint8_t> audio_buffer;
    std::string previous_parakeet_pending;
    size_t previous_parakeet_audio_buffer_size = 0;
    size_t previous_parakeet_pending_ticks = 0;
    size_t parakeet_silence_run_bytes = 0;
    size_t parakeet_speech_run_bytes = 0;
    bool parakeet_resume_guard_active = true;
    std::string parakeet_committed_text;
    size_t parakeet_chunk_cursor_bytes = 0;
    float parakeet_committed_until_sec = 0.0f;
    bool parakeet_onset_active = false;
    size_t parakeet_onset_start_bytes = 0;
    bool parakeet_tdt_chunked_stream = false;
    struct ParakeetTdtDecodeContext {
        std::vector<float> audio_samples;
        size_t samples_decoded_up_to = 0;
        float audio_time_offset_sec = 0.0f;
        cactus::engine::ParakeetTDTModel::ChunkStreamState decoder_state;
        cactus::engine::AudioProcessor audio_processor;
        size_t mel_bins = 0;
        bool initialized = false;
    } parakeet_tdt_decode_context;
    bool parakeet_tdt_chunked_blocked_until_pause = false;
    cactus::engine::ParakeetTDTModel::ChunkStreamState parakeet_tdt_decoder_state;
    std::string parakeet_tdt_pending_raw;
    struct ParakeetTdtResumeCheckpoint {
        size_t end_bytes = 0;
        cactus::engine::ParakeetTDTModel::ChunkStreamState state;
    };
    std::deque<ParakeetTdtResumeCheckpoint> parakeet_tdt_resume_checkpoints;

    std::vector<TranscriptSegment> previous_segments;
    bool previous_cloud_handoff = false;
    float stream_audio_offset_sec = 0.0f;
    uint64_t next_cloud_job_id = 1;

    struct CloudJob {
        uint64_t id;
        std::future<CloudResponse> result;
    };
    std::vector<CloudJob> pending_cloud_jobs;
    std::vector<std::pair<uint64_t, CloudResponse>> completed_cloud_results;

    char transcribe_response_buffer[65536];

    std::chrono::steady_clock::time_point stream_start;
    bool stream_first_token_seen;
    double stream_first_token_ms;
    int stream_total_tokens;

    std::chrono::steady_clock::time_point stream_session_start;
    bool stream_session_first_token_seen;
    double stream_session_first_token_ms;
    int stream_cumulative_tokens;
    double last_stream_decode_tps = 0.0;
    double last_stream_decode_tokens = 0.0;
    double last_stream_total_time_ms = 0.0;
    double last_stream_raw_decoder_tps = 0.0;
};

static std::vector<TranscriptSegment> parse_segments(const std::string& transcribe_json);

struct StreamWindowDecodeResult {
    std::string raw_json;
    std::string response;
    std::vector<TranscriptSegment> segments;
    double decode_tokens = 0.0;
    bool cloud_handoff = false;
};

static double pcm_bytes_to_seconds(size_t pcm_bytes) {
    return static_cast<double>(pcm_bytes) / (16000.0 * 2.0);
}

static size_t seconds_to_pcm_bytes(double seconds) {
    if (seconds <= 0.0) return 0;
    const size_t raw = static_cast<size_t>(std::llround(seconds * 16000.0 * 2.0));
    return raw - (raw % sizeof(int16_t));
}

static std::string join_segments_by_end_time(
    const std::vector<TranscriptSegment>& segments,
    float offset_sec,
    float min_end_exclusive_sec,
    float max_end_inclusive_sec,
    float* out_max_emitted_end_sec = nullptr) {
    std::string joined;
    float max_end_sec = min_end_exclusive_sec;
    for (const auto& seg : segments) {
        const float abs_end_sec = offset_sec + seg.end;
        if (abs_end_sec <= min_end_exclusive_sec || abs_end_sec > max_end_inclusive_sec) {
            continue;
        }
        if (!joined.empty()) joined += ' ';
        joined += seg.text;
        if (abs_end_sec > max_end_sec) {
            max_end_sec = abs_end_sec;
        }
    }

    if (out_max_emitted_end_sec) {
        *out_max_emitted_end_sec = max_end_sec;
    }
    return suppress_unwanted_text(joined);
}

static bool run_stream_window_transcribe(
    CactusStreamTranscribeHandle* handle,
    const char* transcribe_prompt,
    const std::string& effective_transcribe_options_json,
    const uint8_t* decode_pcm,
    size_t decode_pcm_size,
    StreamWindowDecodeResult& out) {
    if (!handle || !handle->model_handle || !decode_pcm || decode_pcm_size == 0) {
        return false;
    }

    const auto model_type = handle->model_handle->model->get_config().model_type;
    const bool use_model_stream_mode =
        model_type != cactus::engine::Config::ModelType::PARAKEET_TDT;
    cactus::telemetry::setStreamMode(use_model_stream_mode);
    const int result = cactus_transcribe(
        handle->model_handle,
        nullptr,
        transcribe_prompt,
        handle->transcribe_response_buffer,
        sizeof(handle->transcribe_response_buffer),
        effective_transcribe_options_json.empty() ? nullptr : effective_transcribe_options_json.c_str(),
        nullptr,
        nullptr,
        decode_pcm,
        decode_pcm_size);
    cactus::telemetry::setStreamMode(false);

    if (result < 0) {
        return false;
    }

    out.raw_json.assign(handle->transcribe_response_buffer);
    out.response = suppress_unwanted_text(json_string(out.raw_json, "response"));
    out.segments = parse_segments(out.raw_json);
    out.decode_tokens = std::max(0.0, json_number(out.raw_json, "decode_tokens"));
    out.cloud_handoff = json_bool(out.raw_json, "cloud_handoff");
    handle->last_stream_decode_tps = std::max(0.0, json_number(out.raw_json, "decode_tps"));
    handle->last_stream_decode_tokens = out.decode_tokens;
    handle->last_stream_total_time_ms = std::max(0.0, json_number(out.raw_json, "total_time_ms"));
    handle->last_stream_raw_decoder_tps = std::max(0.0, json_number(out.raw_json, "raw_decoder_tps"));
    return true;
}

static bool run_parakeet_tdt_chunk_decode(
    CactusStreamTranscribeHandle* handle,
    size_t window_start_bytes,
    size_t window_end_bytes,
    size_t chunk_start_bytes,
    size_t chunk_end_bytes,
    cactus::engine::ParakeetTDTModel::ChunkStreamResult& out,
    cactus::engine::ParakeetTDTModel::ChunkStreamState& out_state) {
    if (!handle || !handle->model_handle || !handle->model_handle->model) {
        return false;
    }
    if (chunk_end_bytes <= chunk_start_bytes ||
        window_end_bytes <= window_start_bytes ||
        window_end_bytes > handle->audio_buffer.size() ||
        chunk_end_bytes > window_end_bytes ||
        window_start_bytes > chunk_start_bytes) {
        return false;
    }

    auto* model = static_cast<cactus::engine::ParakeetTDTModel*>(
        handle->model_handle->model.get());
    size_t replay_start_bytes = window_start_bytes;
    cactus::engine::ParakeetTDTModel::ChunkStreamState base_state;
    bool found_checkpoint = false;
    for (auto it = handle->parakeet_tdt_resume_checkpoints.rbegin();
         it != handle->parakeet_tdt_resume_checkpoints.rend();
         ++it) {
        if (it->end_bytes <= window_start_bytes) {
            replay_start_bytes = std::max(window_start_bytes, it->end_bytes);
            base_state = it->state;
            found_checkpoint = true;
            break;
        }
    }
    if (!found_checkpoint) {
        for (auto it = handle->parakeet_tdt_resume_checkpoints.rbegin();
             it != handle->parakeet_tdt_resume_checkpoints.rend();
             ++it) {
            if (it->end_bytes <= chunk_start_bytes) {
                replay_start_bytes = std::max(window_start_bytes, it->end_bytes);
                base_state = it->state;
                found_checkpoint = true;
                break;
            }
        }
    }
    if (!found_checkpoint) {
        replay_start_bytes = window_start_bytes;
        base_state = {};
    }

    std::vector<float> audio_features;
    if (!build_parakeet_features_from_pcm(
            handle->audio_buffer.data() + window_start_bytes,
            window_end_bytes - window_start_bytes,
            model,
            audio_features)) {
        return false;
    }

    const size_t replay_start_frame = parakeet_tdt_pcm_bytes_to_encoder_frames(
        model, replay_start_bytes - window_start_bytes);
    const size_t start_frame = parakeet_tdt_pcm_bytes_to_encoder_frames(
        model, chunk_start_bytes - window_start_bytes);
    size_t end_frame = parakeet_tdt_pcm_bytes_to_encoder_frames(
        model, chunk_end_bytes - window_start_bytes);
    if (end_frame <= start_frame) {
        end_frame = start_frame + 1;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(handle->model_handle->model_mutex);
    cactus::telemetry::setStreamMode(true);
    try {
        out_state = base_state;
        out = model->decode_chunk_stream(
            audio_features,
            replay_start_frame,
            start_frame,
            end_frame,
            out_state);
    } catch (...) {
        cactus::telemetry::setStreamMode(false);
        throw;
    }
    cactus::telemetry::setStreamMode(false);
    const auto decode_end = std::chrono::steady_clock::now();
    const double decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        decode_end - decode_start).count() / 1000.0;
    handle->last_stream_total_time_ms = decode_ms;
    handle->last_stream_decode_tokens = static_cast<double>(out.token_count);
    handle->last_stream_decode_tps =
        (out.token_count > 1 && decode_ms > 0.0)
            ? ((static_cast<double>(out.token_count) - 1.0) * 1000.0) / decode_ms
            : 0.0;
    handle->last_stream_raw_decoder_tps = out.raw_decoder_tps;
    return true;
}

static void reset_parakeet_tdt_chunk_stream(CactusStreamTranscribeHandle* handle) {
    if (!handle) return;
    handle->parakeet_tdt_decoder_state = {};
    handle->parakeet_tdt_pending_raw.clear();
    handle->parakeet_tdt_resume_checkpoints.clear();
}

static std::vector<std::pair<size_t, size_t>> collect_parakeet_word_spans(const std::string& text);

static std::string strip_unwanted_text_preserving_edges(const std::string& text) {
    static const std::regex pattern(R"(\([^)]*\)|\[[^\]]*\]|\.\.\.)");
    return std::regex_replace(text, pattern, "");
}

static void parakeet_append_monotonic_text(std::string& dst, const std::string& piece) {
    const std::string normalized = suppress_unwanted_text(piece);
    if (normalized.empty()) return;
    if (dst.empty()) {
        dst = normalized;
        return;
    }

    const unsigned char first = static_cast<unsigned char>(normalized.front());
    const bool attach_without_space =
        std::ispunct(first) && normalized.front() != '"' &&
        normalized.front() != '\'' && normalized.front() != '(';
    if (!std::isspace(static_cast<unsigned char>(dst.back())) && !attach_without_space) {
        dst += " ";
    }
    dst += normalized;
}

static std::vector<TranscriptSegment> parse_segments(const std::string& transcribe_json) {
    std::vector<TranscriptSegment> segs;
    std::string arr = json_array(transcribe_json, "segments");
    size_t i = 1;
    while (i < arr.size() - 1) {
        while (i < arr.size() && (arr[i] == ',' || std::isspace((unsigned char)arr[i]))) ++i;
        if (i >= arr.size() - 1 || arr[i] != '{') break;
        int depth = 1;
        size_t obj_start = i++;
        while (i < arr.size() && depth > 0) {
            if (arr[i] == '{') ++depth;
            else if (arr[i] == '}') --depth;
            ++i;
        }
        std::string obj = arr.substr(obj_start, i - obj_start);
        segs.emplace_back(
            static_cast<float>(json_number(obj, "start")),
            static_cast<float>(json_number(obj, "end")),
            json_string(obj, "text")
        );
    }
    return segs;
}

static bool is_parakeet_word_char(unsigned char ch) {
    return std::isalnum(ch) || ch == '\'';
}

static std::vector<std::pair<size_t, size_t>> collect_parakeet_word_spans(const std::string& text) {
    std::vector<std::pair<size_t, size_t>> spans;
    bool in_word = false;
    size_t word_start = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (is_parakeet_word_char(ch)) {
            if (!in_word) {
                in_word = true;
                word_start = i;
            }
        } else if (in_word) {
            spans.emplace_back(word_start, i);
            in_word = false;
        }
    }

    if (in_word) {
        spans.emplace_back(word_start, text.size());
    }
    return spans;
}

static std::string parakeet_word_from_span(
    const std::string& text,
    const std::pair<size_t, size_t>& span) {
    if (span.second <= span.first || span.second > text.size()) return "";
    return text.substr(span.first, span.second - span.first);
}

static bool parakeet_words_fuzzy_equal(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;
    const size_t dist = levenshtein_ci(a, b);
    if (dist == 0) return true;
    const size_t max_len = std::max(a.size(), b.size());
    return max_len >= 5 && dist <= 1;
}

static size_t parakeet_common_prefix_words(const std::string& left, const std::string& right) {
    const auto left_spans = collect_parakeet_word_spans(left);
    const auto right_spans = collect_parakeet_word_spans(right);
    const size_t max_words = std::min(left_spans.size(), right_spans.size());
    size_t common = 0;
    for (; common < max_words; ++common) {
        const std::string left_word = parakeet_word_from_span(left, left_spans[common]);
        const std::string right_word = parakeet_word_from_span(right, right_spans[common]);
        if (!parakeet_words_fuzzy_equal(left_word, right_word)) {
            break;
        }
    }
    return common;
}

static std::string parakeet_take_first_words(const std::string& text, size_t words) {
    if (text.empty() || words == 0) return "";
    const auto spans = collect_parakeet_word_spans(text);
    if (spans.empty()) return "";
    if (words >= spans.size()) return suppress_unwanted_text(text);
    const size_t end_idx = spans[words].first;
    return suppress_unwanted_text(text.substr(0, end_idx));
}

static std::string parakeet_drop_first_words(const std::string& text, size_t words) {
    if (text.empty()) return "";
    const auto spans = collect_parakeet_word_spans(text);
    if (spans.empty()) return suppress_unwanted_text(text);
    if (words >= spans.size()) return "";
    const size_t start_idx = spans[words].first;
    return suppress_unwanted_text(text.substr(start_idx));
}

static size_t parakeet_word_count(const std::string& text) {
    return collect_parakeet_word_spans(text).size();
}

static size_t parakeet_overlap_suffix_prefix_words(
    const std::string& left,
    const std::string& right,
    size_t max_words = 64) {
    const auto left_spans = collect_parakeet_word_spans(left);
    const auto right_spans = collect_parakeet_word_spans(right);
    if (left_spans.empty() || right_spans.empty()) return 0;

    const size_t max_overlap = std::min({left_spans.size(), right_spans.size(), max_words});
    size_t best = 0;
    for (size_t k = 1; k <= max_overlap; ++k) {
        bool matched = true;
        for (size_t i = 0; i < k; ++i) {
            const auto& left_span = left_spans[left_spans.size() - k + i];
            const auto& right_span = right_spans[i];
            const std::string left_word = parakeet_word_from_span(left, left_span);
            const std::string right_word = parakeet_word_from_span(right, right_span);
            if (!parakeet_words_fuzzy_equal(left_word, right_word)) {
                matched = false;
                break;
            }
        }
        if (matched) best = k;
    }
    return best;
}

static std::string parakeet_strip_recent_aligned_overlap(
    const std::string& committed_text,
    const std::string& hypothesis,
    size_t max_recent_words = 24) {
    const std::string candidate = suppress_unwanted_text(hypothesis);
    if (candidate.empty() || committed_text.empty()) return candidate;

    const auto committed_spans = collect_parakeet_word_spans(committed_text);
    if (committed_spans.empty()) return candidate;

    const size_t recent_words = std::min<size_t>(committed_spans.size(), max_recent_words);
    const size_t recent_start = committed_spans[committed_spans.size() - recent_words].first;
    const std::string recent_tail = suppress_unwanted_text(committed_text.substr(recent_start));
    const auto recent_spans = collect_parakeet_word_spans(recent_tail);
    const auto candidate_spans = collect_parakeet_word_spans(candidate);
    const size_t recent_count = recent_spans.size();
    const size_t candidate_count = candidate_spans.size();
    if (recent_count == 0 || candidate_count == 0) return candidate;

    size_t best_run = 0;
    size_t best_candidate_start = candidate_count;
    size_t best_candidate_end = 0;
    size_t best_recent_start = recent_count;

    for (size_t i = 0; i < recent_count; ++i) {
        for (size_t j = 0; j < candidate_count; ++j) {
            size_t run = 0;
            while (i + run < recent_count && j + run < candidate_count) {
                const std::string recent_word = parakeet_word_from_span(
                    recent_tail, recent_spans[i + run]);
                const std::string candidate_word = parakeet_word_from_span(
                    candidate, candidate_spans[j + run]);
                if (!parakeet_words_fuzzy_equal(recent_word, candidate_word)) {
                    break;
                }
                ++run;
            }

            const size_t candidate_end = j + run;
            const bool better_run = run > best_run;
            const bool better_end = run == best_run && candidate_end > best_candidate_end;
            const bool better_start = run == best_run &&
                                      candidate_end == best_candidate_end &&
                                      j < best_candidate_start;
            if (better_run || better_end || better_start) {
                best_run = run;
                best_candidate_start = j;
                best_candidate_end = candidate_end;
                best_recent_start = i;
            }
        }
    }

    if (best_run == 0 || best_candidate_end == 0) return candidate;

    const size_t min_overlap_words = candidate_count <= 5 ? 2 : 3;
    if (best_run < min_overlap_words) return candidate;

    if (best_candidate_start > 1) return candidate;
    if (best_candidate_start == 1) {
        const std::string lead_word = parakeet_word_from_span(candidate, candidate_spans[0]);
        const std::string overlap_word = parakeet_word_from_span(candidate, candidate_spans[1]);
        const std::string recent_word = parakeet_word_from_span(
            recent_tail, recent_spans[best_recent_start]);
        if (!parakeet_words_fuzzy_equal(lead_word, overlap_word) &&
            !parakeet_words_fuzzy_equal(lead_word, recent_word)) {
            return candidate;
        }
    }

    return parakeet_drop_first_words(candidate, best_candidate_end);
}

static std::string parakeet_strip_recent_committed_prefix(
    const std::string& committed_text,
    const std::string& hypothesis);

static std::string parakeet_emit_delta(
    const std::string& committed_text,
    const std::string& newly_confirmed) {
    std::string candidate = parakeet_strip_recent_committed_prefix(
        committed_text, newly_confirmed);
    if (candidate.empty()) return "";
    if (committed_text.empty()) return candidate;

    const size_t overlap_words = parakeet_overlap_suffix_prefix_words(committed_text, candidate);
    if (overlap_words > 0) {
        return parakeet_drop_first_words(candidate, overlap_words);
    }

    return candidate;
}

static std::string parakeet_strip_recent_committed_prefix(
    const std::string& committed_text,
    const std::string& hypothesis) {
    const std::string candidate = suppress_unwanted_text(hypothesis);
    if (candidate.empty() || committed_text.empty()) return candidate;

    const auto committed_spans = collect_parakeet_word_spans(committed_text);
    if (committed_spans.empty()) return candidate;

    const size_t recent_words = std::min<size_t>(committed_spans.size(), 24);
    const size_t recent_start = committed_spans[committed_spans.size() - recent_words].first;
    const std::string recent_tail = suppress_unwanted_text(committed_text.substr(recent_start));
    const size_t candidate_words = parakeet_word_count(candidate);
    const size_t overlap_words = parakeet_overlap_suffix_prefix_words(recent_tail, candidate, recent_words);

    // Be conservative for tiny hypotheses, but strip even a single repeated
    // boundary word when the new hypothesis is a longer continuation.
    const size_t required_overlap = candidate_words >= 5 ? 1 : 2;
    if (overlap_words >= required_overlap) {
        return parakeet_drop_first_words(candidate, overlap_words);
    }

    const std::string aligned_trimmed = parakeet_strip_recent_aligned_overlap(
        committed_text, candidate, recent_words);
    if (aligned_trimmed != candidate) {
        return aligned_trimmed;
    }

    const auto recent_spans = collect_parakeet_word_spans(recent_tail);
    const auto candidate_spans = collect_parakeet_word_spans(candidate);
    const size_t recent_count = recent_spans.size();
    const size_t candidate_count = candidate_spans.size();
    if (recent_count == 0 || candidate_count < 4) return candidate;

    std::vector<std::vector<size_t>> dp(
        recent_count + 1, std::vector<size_t>(candidate_count + 1, 0));
    for (size_t i = 1; i <= recent_count; ++i) {
        for (size_t j = 1; j <= candidate_count; ++j) {
            const std::string recent_word = parakeet_word_from_span(recent_tail, recent_spans[i - 1]);
            const std::string candidate_word = parakeet_word_from_span(candidate, candidate_spans[j - 1]);
            if (parakeet_words_fuzzy_equal(recent_word, candidate_word)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    const size_t lcs_words = dp[recent_count][candidate_count];
    if (lcs_words < 3 || lcs_words * 2 < candidate_count) return candidate;

    size_t i = recent_count;
    size_t j = candidate_count;
    size_t last_matched_candidate_word = 0;
    bool found_match = false;
    bool captured_last_match = false;
    while (i > 0 && j > 0) {
        const std::string recent_word = parakeet_word_from_span(recent_tail, recent_spans[i - 1]);
        const std::string candidate_word = parakeet_word_from_span(candidate, candidate_spans[j - 1]);
        if (parakeet_words_fuzzy_equal(recent_word, candidate_word)) {
            if (!captured_last_match) {
                last_matched_candidate_word = j;
                captured_last_match = true;
            }
            found_match = true;
            --i;
            --j;
        } else if (dp[i - 1][j] >= dp[i][j - 1]) {
            --i;
        } else {
            --j;
        }
    }

    if (!found_match || last_matched_candidate_word >= candidate_count) return candidate;
    return parakeet_drop_first_words(candidate, last_matched_candidate_word);

    return candidate;
}

static std::string serialize_segments_with_offset(
    const std::vector<TranscriptSegment>& segs, float offset_sec) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < segs.size(); ++i) {
        if (i > 0) out << ",";
        out << std::fixed << std::setprecision(3)
            << "{\"start\":" << (segs[i].start + offset_sec)
            << ",\"end\":" << (segs[i].end + offset_sec)
            << ",\"text\":\"" << escape_json(segs[i].text) << "\"}";
    }
    out << "]";
    return out.str();
}

static std::string build_stream_response(
    const std::string& raw_json_str,
    const std::string& error_msg,
    const std::string& confirmed,
    const std::string& pending,
    const std::string& segments_json,
    bool cloud_handoff,
    double buffer_duration_ms,
    uint64_t cloud_job_id,
    uint64_t cloud_result_job_id,
    const CloudResponse& cloud_result,
    double override_time_to_first_token_ms = -1.0,
    double override_total_time_ms = -1.0,
    double override_decode_tps = -1.0,
    double override_decode_tokens = -1.0,
    double override_total_tokens = -1.0,
    double override_raw_decoder_tps = -1.0
) {
    std::string function_calls = json_array(raw_json_str, "function_calls");
    double confidence = json_number(raw_json_str, "confidence");
    double time_to_first_token_ms = json_number(raw_json_str, "time_to_first_token_ms");
    double total_time_ms = json_number(raw_json_str, "total_time_ms");
    double prefill_tps = json_number(raw_json_str, "prefill_tps");
    double decode_tps = json_number(raw_json_str, "decode_tps");
    double ram_usage_mb = json_number(raw_json_str, "ram_usage_mb");
    double prefill_tokens = json_number(raw_json_str, "prefill_tokens");
    double decode_tokens = json_number(raw_json_str, "decode_tokens");
    double total_tokens = json_number(raw_json_str, "total_tokens");
    double raw_decoder_tps = json_number(raw_json_str, "raw_decoder_tps");
    if (override_time_to_first_token_ms >= 0.0) {
        time_to_first_token_ms = override_time_to_first_token_ms;
    }
    if (override_total_time_ms >= 0.0) {
        total_time_ms = override_total_time_ms;
    }
    if (override_decode_tps >= 0.0) {
        decode_tps = override_decode_tps;
    }
    if (override_decode_tokens >= 0.0) {
        decode_tokens = override_decode_tokens;
    }
    if (override_total_tokens >= 0.0) {
        total_tokens = override_total_tokens;
    }
    if (override_raw_decoder_tps >= 0.0) {
        raw_decoder_tps = override_raw_decoder_tps;
    }
    std::string effective_confirmed = confirmed;
    if (cloud_result.used_cloud && !cloud_result.transcript.empty()) {
        effective_confirmed = cloud_result.transcript;
    }

    std::ostringstream json_builder;
    json_builder << "{";
    json_builder << "\"success\":true,";
    json_builder << "\"buffer_duration_ms\":" << buffer_duration_ms << ",";
    json_builder << "\"error\":" << (error_msg.empty() ? "null" : "\"" + escape_json(error_msg) + "\"") << ",";
    json_builder << "\"cloud_handoff\":" << (cloud_handoff ? "true" : "false") << ",";
    json_builder << "\"cloud_job_id\":" << cloud_job_id << ",";
    json_builder << "\"cloud_result_job_id\":" << cloud_result_job_id << ",";
    json_builder << "\"cloud_result\":\"" << escape_json(cloud_result.transcript) << "\",";
    json_builder << "\"cloud_result_used_cloud\":" << (cloud_result.used_cloud ? "true" : "false") << ",";
    json_builder << "\"cloud_result_error\":";
    if (cloud_result.error.empty()) {
        json_builder << "null,";
    } else {
        json_builder << "\"" << escape_json(cloud_result.error) << "\",";
    }
    json_builder << "\"cloud_result_source\":\"" << (cloud_result.used_cloud ? "cloud" : "fallback") << "\",";
    json_builder << "\"confirmed_local\":\"" << escape_json(confirmed) << "\",";
    json_builder << "\"confirmed\":\"" << escape_json(effective_confirmed) << "\",";
    json_builder << "\"pending\":\"" << escape_json(pending) << "\",";
    json_builder << "\"segments\":" << segments_json << ",";
    json_builder << "\"function_calls\":" << function_calls << ",";
    json_builder << "\"confidence\":" << confidence << ",";
    json_builder << "\"time_to_first_token_ms\":" << time_to_first_token_ms << ",";
    json_builder << "\"total_time_ms\":" << total_time_ms << ",";
    json_builder << "\"prefill_tps\":" << prefill_tps << ",";
    json_builder << "\"decode_tps\":" << decode_tps << ",";
    json_builder << "\"ram_usage_mb\":" << ram_usage_mb << ",";
    json_builder << "\"prefill_tokens\":" << prefill_tokens << ",";
    json_builder << "\"decode_tokens\":" << decode_tokens << ",";
    json_builder << "\"total_tokens\":" << total_tokens << ",";
    json_builder << "\"raw_decoder_tps\":" << raw_decoder_tps;
    json_builder << "}";
    return json_builder.str();
}

extern "C" {

cactus_stream_transcribe_t cactus_stream_transcribe_start(cactus_model_t model, const char* options_json) {
    if (!model) {
        last_error_message = "Model not initialized. Check model path and files.";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }

    try {
        auto* model_handle = static_cast<CactusModelHandle*>(model);
        if (!model_handle->model) {
            last_error_message = "Invalid model handle.";
            CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
            return nullptr;
        }

        auto* stream_handle = new CactusStreamTranscribeHandle();
        stream_handle->model_handle = model_handle;
        stream_handle->transcribe_response_buffer[0] = '\0';

        auto session_start_time = std::chrono::steady_clock::now();
        stream_handle->stream_start = session_start_time;
        stream_handle->stream_first_token_seen = false;
        stream_handle->stream_first_token_ms = 0.0;
        stream_handle->stream_total_tokens = 0;

        stream_handle->stream_session_start = session_start_time;
        stream_handle->stream_session_first_token_seen = false;
        stream_handle->stream_session_first_token_ms = 0.0;
        stream_handle->stream_cumulative_tokens = 0;

        double confirmation_threshold;
        size_t min_chunk_size;
        bool telemetry_enabled;
        std::string language;
        parse_stream_transcribe_init_options(
            options_json ? options_json : "",
            confirmation_threshold,
            min_chunk_size,
            telemetry_enabled,
            language
        );

        stream_handle->options = { confirmation_threshold, min_chunk_size, language };
        stream_handle->transcribe_options_json = options_json ? options_json : "";
        stream_handle->parakeet_tdt_chunked_stream =
            model_handle->model->get_config().model_type ==
                cactus::engine::Config::ModelType::PARAKEET_TDT;
        {
            float vocabulary_boost = 5.0f;
            parse_custom_vocabulary_options(stream_handle->transcribe_options_json,
                                           stream_handle->custom_vocabulary, vocabulary_boost);
            auto vocab_bias = build_custom_vocabulary_bias(
                model_handle->model->get_tokenizer(),
                stream_handle->custom_vocabulary,
                vocabulary_boost
            );
            stream_handle->has_custom_vocabulary_bias = !vocab_bias.empty();
            if (stream_handle->has_custom_vocabulary_bias) {
                model_handle->model->set_vocab_bias(vocab_bias);
            }
        }

        if (stream_handle->parakeet_tdt_chunked_stream) {
            auto& ctx = stream_handle->parakeet_tdt_decode_context;
            ctx.mel_bins = std::max<size_t>(
                1, static_cast<size_t>(model_handle->model->get_config().num_mel_bins));
            auto cfg = get_parakeet_spectrogram_config();
            ctx.audio_processor.init_mel_filters(
                cfg.n_fft / 2 + 1,
                ctx.mel_bins,
                0.0f,
                8000.0f,
                WHISPER_SAMPLE_RATE);
            ctx.initialized = true;
        }

        CACTUS_LOG_INFO("stream_transcribe_start",
            "Stream transcription initialized for model: " << model_handle->model_name);

        return stream_handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_start: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription initialization";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }
}

int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    if (!pcm_buffer || pcm_buffer_size == 0) {
        last_error_message = "Invalid parameters: pcm_buffer or pcm_buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        last_error_message = "Invalid parameters: response_buffer or buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
        return -1;
    }

    try {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);
        const auto model_type = handle->model_handle->model->get_config().model_type;
        bool is_moonshine = model_type == cactus::engine::Config::ModelType::MOONSHINE;
        bool is_parakeet_tdt =
            model_type == cactus::engine::Config::ModelType::PARAKEET_TDT;
        bool is_parakeet_ctc =
            model_type == cactus::engine::Config::ModelType::PARAKEET;
        bool is_parakeet =
            is_parakeet_ctc || is_parakeet_tdt;
        bool is_gemma4 = model_type == cactus::engine::Config::ModelType::GEMMA4;

        if (is_parakeet_tdt &&
            handle->parakeet_tdt_chunked_stream &&
            handle->parakeet_tdt_decode_context.initialized) {
            auto& ctx = handle->parakeet_tdt_decode_context;
            auto* tdt_model = static_cast<cactus::engine::ParakeetTDTModel*>(
                handle->model_handle->model.get());
            auto new_samples = cactus::audio::pcm_buffer_to_float_samples(
                pcm_buffer, pcm_buffer_size);
            ctx.audio_samples.insert(ctx.audio_samples.end(), new_samples.begin(), new_samples.end());

            constexpr size_t kTdtLeftContextSamples = 8 * 16000;
            constexpr size_t kTdtRightContextSamples = 16000;
            constexpr size_t kTdtChunkSamples = 16000;
            constexpr size_t kTdtColdStartSamples = 6 * 16000;

            const size_t total_samples = ctx.audio_samples.size();
            const size_t decodable_up_to = total_samples > kTdtRightContextSamples
                ? total_samples - kTdtRightContextSamples
                : 0;
            const size_t min_chunk_samples = ctx.samples_decoded_up_to == 0
                ? kTdtColdStartSamples
                : kTdtChunkSamples;

            if (decodable_up_to <= ctx.samples_decoded_up_to ||
                decodable_up_to - ctx.samples_decoded_up_to < min_chunk_samples) {
                const std::string json_response = build_stream_response(
                    "{}",
                    "",
                    "",
                    handle->previous_parakeet_pending,
                    "[]",
                    false,
                    0.0,
                    0,
                    0,
                    CloudResponse{},
                    handle->stream_first_token_seen ? handle->stream_first_token_ms : 0.0,
                    handle->last_stream_total_time_ms,
                    handle->last_stream_decode_tps,
                    handle->last_stream_decode_tokens,
                    handle->last_stream_decode_tokens,
                    handle->last_stream_raw_decoder_tps);
                if (json_response.length() >= buffer_size) {
                    handle_error_response("Response buffer too small", response_buffer, buffer_size);
                    return -1;
                }
                std::strcpy(response_buffer, json_response.c_str());
                return static_cast<int>(json_response.length());
            }

            const size_t window_start_sample = ctx.samples_decoded_up_to > kTdtLeftContextSamples
                ? ctx.samples_decoded_up_to - kTdtLeftContextSamples
                : 0;
            const size_t window_end_sample = std::min(
                total_samples, decodable_up_to + kTdtRightContextSamples);
            std::vector<float> window_audio(
                ctx.audio_samples.begin() + window_start_sample,
                ctx.audio_samples.begin() + window_end_sample);

            auto cfg = get_parakeet_spectrogram_config();
            const size_t waveform_samples = window_audio.size();
            apply_preemphasis(window_audio, 0.97f);
            window_audio = ctx.audio_processor.compute_spectrogram(window_audio, cfg);
            normalize_parakeet_log_mel(window_audio, ctx.mel_bins);
            size_t valid_frames = waveform_samples / cfg.hop_length;
            if (valid_frames == 0) valid_frames = 1;
            trim_mel_frames(window_audio, ctx.mel_bins, valid_frames);

            const uint32_t subsampling =
                std::max<uint32_t>(1, handle->model_handle->model->get_config().subsampling_factor);
            const size_t samples_per_enc_frame = cfg.hop_length * subsampling;
            const size_t context_samples = ctx.samples_decoded_up_to - window_start_sample;
            const size_t decode_start_frame = context_samples / samples_per_enc_frame;
            const size_t new_samples_count = decodable_up_to - ctx.samples_decoded_up_to;
            const size_t decode_end_frame =
                decode_start_frame + (new_samples_count / samples_per_enc_frame);

            if (decode_end_frame <= decode_start_frame) {
                const std::string json_response = build_stream_response(
                    "{}",
                    "",
                    "",
                    handle->previous_parakeet_pending,
                    "[]",
                    false,
                    0.0,
                    0,
                    0,
                    CloudResponse{},
                    handle->stream_first_token_seen ? handle->stream_first_token_ms : 0.0,
                    handle->last_stream_total_time_ms,
                    handle->last_stream_decode_tps,
                    handle->last_stream_decode_tokens,
                    handle->last_stream_decode_tokens,
                    handle->last_stream_raw_decoder_tps);
                if (json_response.length() >= buffer_size) {
                    handle_error_response("Response buffer too small", response_buffer, buffer_size);
                    return -1;
                }
                std::strcpy(response_buffer, json_response.c_str());
                return static_cast<int>(json_response.length());
            }

            cactus::engine::ParakeetTDTModel::ChunkStreamResult chunk_decode;
            {
                const auto decode_start = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(handle->model_handle->model_mutex);
                cactus::telemetry::setStreamMode(true);
                try {
                    chunk_decode = tdt_model->decode_chunk_stream(
                        window_audio,
                        decode_start_frame,
                        decode_start_frame,
                        decode_end_frame,
                        ctx.decoder_state);
                    cactus_reset(handle->model_handle);
                } catch (...) {
                    cactus::telemetry::setStreamMode(false);
                    throw;
                }
                cactus::telemetry::setStreamMode(false);
                const auto decode_end = std::chrono::steady_clock::now();
                const double decode_ms =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        decode_end - decode_start).count() / 1000.0;
                handle->last_stream_total_time_ms = decode_ms;
                handle->last_stream_decode_tokens = static_cast<double>(chunk_decode.token_count);
                handle->last_stream_decode_tps =
                    (chunk_decode.token_count > 1 && decode_ms > 0.0)
                        ? ((static_cast<double>(chunk_decode.token_count) - 1.0) * 1000.0) / decode_ms
                        : 0.0;
                handle->last_stream_raw_decoder_tps = chunk_decode.raw_decoder_tps;
            }

            const float window_offset_sec = ctx.audio_time_offset_sec +
                static_cast<float>(window_start_sample) / static_cast<float>(WHISPER_SAMPLE_RATE);
            std::vector<TranscriptSegment> confirmed_segments;
            if (!chunk_decode.confirmed_text.empty()) {
                confirmed_segments.emplace_back(
                    window_offset_sec + chunk_decode.start_sec,
                    window_offset_sec + chunk_decode.confirmed_end_sec,
                    suppress_unwanted_text(chunk_decode.confirmed_text));
            }
            std::string confirmed_text = suppress_unwanted_text(chunk_decode.confirmed_text);
            std::string pending_text = suppress_unwanted_text(chunk_decode.pending_text);
            if (!confirmed_text.empty() && !handle->custom_vocabulary.empty()) {
                apply_vocabulary_spelling_correction(confirmed_text, handle->custom_vocabulary);
            }
            if (!pending_text.empty() && !handle->custom_vocabulary.empty()) {
                apply_vocabulary_spelling_correction(pending_text, handle->custom_vocabulary);
            }

            if (!confirmed_text.empty()) {
                if (chunk_decode.token_count > 0) {
                    handle->stream_total_tokens += static_cast<int>(chunk_decode.token_count);
                    handle->stream_cumulative_tokens += static_cast<int>(chunk_decode.token_count);
                }
                if (!handle->stream_first_token_seen) {
                    auto now = std::chrono::steady_clock::now();
                    handle->stream_first_token_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - handle->stream_start).count();
                    handle->stream_first_token_seen = true;
                }
                if (!handle->stream_session_first_token_seen) {
                    auto now = std::chrono::steady_clock::now();
                    handle->stream_session_first_token_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - handle->stream_session_start).count();
                    handle->stream_session_first_token_seen = true;
                }
                parakeet_append_monotonic_text(handle->parakeet_committed_text, confirmed_text);
            }

            handle->previous_parakeet_pending = pending_text;
            handle->previous_parakeet_pending_ticks = pending_text.empty() ? 0 : 1;

            if (chunk_decode.confirmed_token_count > 0) {
                const size_t confirmed_samples_in_window = static_cast<size_t>(
                    chunk_decode.confirmed_end_sec * WHISPER_SAMPLE_RATE);
                ctx.samples_decoded_up_to = window_start_sample + confirmed_samples_in_window;
            } else if (chunk_decode.token_count == 0) {
                ctx.samples_decoded_up_to = decodable_up_to;
            }

            if (ctx.samples_decoded_up_to > kTdtLeftContextSamples * 2) {
                const size_t trim_samples = ctx.samples_decoded_up_to - kTdtLeftContextSamples;
                ctx.audio_samples.erase(
                    ctx.audio_samples.begin(),
                    ctx.audio_samples.begin() + trim_samples);
                ctx.samples_decoded_up_to -= trim_samples;
                ctx.audio_time_offset_sec += static_cast<float>(trim_samples) / 16000.0f;
            }

            handle->parakeet_committed_until_sec = std::max(
                handle->parakeet_committed_until_sec,
                window_offset_sec +
                    (chunk_decode.confirmed_token_count > 0 ? chunk_decode.confirmed_end_sec : 0.0f));

            const std::string json_response = build_stream_response(
                "{}",
                "",
                confirmed_text,
                pending_text,
                serialize_segments_with_offset(confirmed_segments, 0.0f),
                false,
                0.0,
                0,
                0,
                CloudResponse{},
                handle->stream_first_token_seen ? handle->stream_first_token_ms : 0.0,
                handle->last_stream_total_time_ms,
                handle->last_stream_decode_tps,
                handle->last_stream_decode_tokens,
                handle->last_stream_decode_tokens,
                handle->last_stream_raw_decoder_tps);
            if (json_response.length() >= buffer_size) {
                handle_error_response("Response buffer too small", response_buffer, buffer_size);
                return -1;
            }
            std::strcpy(response_buffer, json_response.c_str());
            return static_cast<int>(json_response.length());
        }

        handle->audio_buffer.insert(
            handle->audio_buffer.end(),
            pcm_buffer,
            pcm_buffer + pcm_buffer_size
        );
        CACTUS_LOG_DEBUG("stream_transcribe_process",
            "Inserted " << pcm_buffer_size << " bytes, buffer size: " << handle->audio_buffer.size());

        if (handle->audio_buffer.size() < handle->options.min_chunk_size * sizeof(int16_t)) {
            std::string json_response = "{\"success\":true,\"confirmed\":\"\",\"pending\":\"\"}";

            if (json_response.length() >= buffer_size) {
                last_error_message = "Response buffer too small";
                CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
                return -1;
            }

            std::strcpy(response_buffer, json_response.c_str());
            return static_cast<int>(json_response.length());
        }

        std::string whisper_prompt = "<|startoftranscript|><|" + handle->options.language + "|><|transcribe|><|notimestamps|>";
        const char* transcribe_prompt =
            is_gemma4 ? "Transcribe the audio." :
            (is_moonshine || is_parakeet) ? "" :
            whisper_prompt.c_str();
        std::string effective_transcribe_options_json = handle->transcribe_options_json;
        if (is_parakeet && !is_parakeet_tdt) {
            effective_transcribe_options_json = ensure_json_bool_option(
                effective_transcribe_options_json, "use_vad", false);
        }

        if (is_parakeet_tdt) {
            constexpr double kParakeetTdtLeftContextSec = 2.0;
            constexpr double kParakeetTdtChunkSec = 0.25;
            constexpr double kParakeetTdtRightContextSec = 1.0;
            constexpr size_t kParakeetTdtMaxChunksPerProcess = 3;

            const size_t left_context_bytes = seconds_to_pcm_bytes(kParakeetTdtLeftContextSec);
            const size_t chunk_bytes = std::max(
                seconds_to_pcm_bytes(kParakeetTdtChunkSec),
                handle->options.min_chunk_size * sizeof(int16_t));
            const size_t right_context_bytes = seconds_to_pcm_bytes(kParakeetTdtRightContextSec);
            const std::string tdt_vad_options_json = ensure_json_bool_option(
                handle->transcribe_options_json, "use_vad", true);
            const std::string tdt_no_vad_options_json = ensure_json_bool_option(
                handle->transcribe_options_json, "use_vad", false);

            if (handle->parakeet_tdt_chunked_stream &&
                !handle->parakeet_tdt_chunked_blocked_until_pause &&
                !handle->parakeet_committed_text.empty()) {
                const size_t encoder_frame_bytes =
                    get_parakeet_spectrogram_config().hop_length *
                    std::max<uint32_t>(1, handle->model_handle->model->get_config().subsampling_factor) *
                    sizeof(int16_t);
                std::string confirmed;
                size_t confirmed_audio_bytes = 0;
                size_t processed_chunks = 0;
                bool chunked_stalled_on_speech = false;

                while (processed_chunks < kParakeetTdtMaxChunksPerProcess) {
                    const bool chunked_warmup =
                        handle->parakeet_committed_text.empty() &&
                        handle->parakeet_tdt_pending_raw.empty();
                    const size_t available_chunk_bytes =
                        handle->audio_buffer.size() > handle->parakeet_chunk_cursor_bytes + right_context_bytes
                        ? handle->audio_buffer.size() - handle->parakeet_chunk_cursor_bytes - right_context_bytes
                        : 0;
                    const size_t chunk_decode_bytes = std::max(
                        encoder_frame_bytes,
                        ((chunked_warmup
                              ? available_chunk_bytes
                              : seconds_to_pcm_bytes(0.96)) /
                         encoder_frame_bytes) * encoder_frame_bytes);
                    if (handle->parakeet_chunk_cursor_bytes + chunk_decode_bytes + right_context_bytes >
                        handle->audio_buffer.size()) {
                        break;
                    }

                    const size_t chunk_start_bytes = handle->parakeet_chunk_cursor_bytes;
                    const size_t chunk_end_bytes = chunk_start_bytes + chunk_decode_bytes;
                    const size_t window_start_bytes = chunk_start_bytes > left_context_bytes
                        ? chunk_start_bytes - left_context_bytes
                        : 0;
                    const size_t window_end_bytes = chunk_end_bytes + right_context_bytes;

                    cactus::engine::ParakeetTDTModel::ChunkStreamResult chunk_decode;
                    cactus::engine::ParakeetTDTModel::ChunkStreamState chunk_state;
                    if (!run_parakeet_tdt_chunk_decode(
                            handle,
                            window_start_bytes,
                            window_end_bytes,
                            chunk_start_bytes,
                            chunk_end_bytes,
                            chunk_decode,
                            chunk_state)) {
                        last_error_message = "Chunked TDT decode failed.";
                        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                        handle_error_response(last_error_message, response_buffer, buffer_size);
                        cactus::telemetry::recordStreamTranscription(
                            handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr,
                            false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0,
                            last_error_message.c_str());
                        return -1;
                    }

                    const bool chunk_silent = is_pcm_chunk_silent(
                        handle->audio_buffer.data() + chunk_start_bytes,
                        chunk_end_bytes - chunk_start_bytes,
                        0.012f);
                    const std::string raw_chunk_text =
                        strip_unwanted_text_preserving_edges(chunk_decode.text);
                    const std::string raw_confirmed_text =
                        strip_unwanted_text_preserving_edges(chunk_decode.confirmed_text);
                    const std::string raw_pending_text =
                        strip_unwanted_text_preserving_edges(chunk_decode.pending_text);

                    if (raw_chunk_text.empty() && !chunk_silent) {
                        chunked_stalled_on_speech = true;
                        break;
                    }

                    if (!raw_chunk_text.empty()) {
                        std::string confirmed_delta = suppress_unwanted_text(raw_confirmed_text);
                        std::string pending_raw = raw_pending_text;
                        if (!confirmed_delta.empty() && !handle->custom_vocabulary.empty()) {
                            apply_vocabulary_spelling_correction(confirmed_delta, handle->custom_vocabulary);
                        }
                        handle->parakeet_tdt_pending_raw = pending_raw;
                        if (!confirmed_delta.empty()) {
                            if (chunk_decode.token_count > 0) {
                                handle->stream_total_tokens += static_cast<int>(chunk_decode.token_count);
                                handle->stream_cumulative_tokens += static_cast<int>(chunk_decode.token_count);
                            }

                            if (!handle->stream_first_token_seen) {
                                auto now = std::chrono::steady_clock::now();
                                handle->stream_first_token_ms =
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        now - handle->stream_start).count();
                                handle->stream_first_token_seen = true;
                            }
                            if (!handle->stream_session_first_token_seen) {
                                auto now = std::chrono::steady_clock::now();
                                handle->stream_session_first_token_ms =
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        now - handle->stream_session_start).count();
                                handle->stream_session_first_token_seen = true;
                            }

                            parakeet_append_monotonic_text(
                                handle->parakeet_committed_text,
                                confirmed_delta);
                            parakeet_append_monotonic_text(confirmed, confirmed_delta);
                            const size_t confirmed_boundary_bytes = std::min(
                                chunk_end_bytes,
                                std::max(
                                    chunk_start_bytes,
                                    window_start_bytes + seconds_to_pcm_bytes(chunk_decode.confirmed_end_sec)));
                            if (confirmed_boundary_bytes > chunk_start_bytes) {
                                confirmed_audio_bytes += confirmed_boundary_bytes - chunk_start_bytes;
                            }
                        }
                    }

                    handle->parakeet_chunk_cursor_bytes = chunk_end_bytes;
                    handle->parakeet_tdt_decoder_state = chunk_state;
                    const size_t checkpoint_end_bytes = std::min(
                        chunk_end_bytes,
                        std::max(
                            window_start_bytes,
                            window_start_bytes + seconds_to_pcm_bytes(chunk_decode.resume_end_sec)));
                    while (!handle->parakeet_tdt_resume_checkpoints.empty() &&
                           handle->parakeet_tdt_resume_checkpoints.back().end_bytes >= checkpoint_end_bytes) {
                        handle->parakeet_tdt_resume_checkpoints.pop_back();
                    }
                    if (handle->parakeet_tdt_resume_checkpoints.empty() ||
                        handle->parakeet_tdt_resume_checkpoints.back().end_bytes < checkpoint_end_bytes) {
                        handle->parakeet_tdt_resume_checkpoints.push_back({checkpoint_end_bytes, chunk_state});
                    }
                    while (handle->parakeet_tdt_resume_checkpoints.size() > 16) {
                        handle->parakeet_tdt_resume_checkpoints.pop_front();
                    }
                    handle->parakeet_committed_until_sec = std::max(
                        handle->parakeet_committed_until_sec,
                        static_cast<float>(
                            handle->stream_audio_offset_sec +
                            pcm_bytes_to_seconds(checkpoint_end_bytes)));
                    ++processed_chunks;
                }

                const bool fallback_to_legacy_tdt =
                    chunked_stalled_on_speech &&
                    confirmed.empty() &&
                    handle->parakeet_tdt_pending_raw.empty();
                if (fallback_to_legacy_tdt) {
                    const bool bootstrap_reset =
                        handle->parakeet_committed_text.empty() &&
                        handle->previous_parakeet_pending.empty();
                    reset_parakeet_tdt_chunk_stream(handle);
                    handle->parakeet_tdt_chunked_blocked_until_pause = true;
                    if (bootstrap_reset) {
                        handle->parakeet_chunk_cursor_bytes = 0;
                    }
                }

                if (!fallback_to_legacy_tdt &&
                    handle->parakeet_chunk_cursor_bytes > left_context_bytes) {
                    const size_t trim_bytes =
                        handle->parakeet_chunk_cursor_bytes - left_context_bytes;
                    handle->stream_audio_offset_sec += static_cast<float>(
                        pcm_bytes_to_seconds(trim_bytes));
                    handle->audio_buffer.erase(
                        handle->audio_buffer.begin(),
                        handle->audio_buffer.begin() + trim_bytes);
                    handle->parakeet_chunk_cursor_bytes -= trim_bytes;
                    while (!handle->parakeet_tdt_resume_checkpoints.empty() &&
                           handle->parakeet_tdt_resume_checkpoints.front().end_bytes <= trim_bytes) {
                        handle->parakeet_tdt_resume_checkpoints.pop_front();
                    }
                    for (auto& checkpoint : handle->parakeet_tdt_resume_checkpoints) {
                        checkpoint.end_bytes -= trim_bytes;
                    }
                }

                if (!fallback_to_legacy_tdt) {
                    std::string pending_output;
                    std::string raw_json = "{}";
                    if (!handle->parakeet_tdt_pending_raw.empty()) {
                        pending_output = suppress_unwanted_text(handle->parakeet_tdt_pending_raw);
                        if (!pending_output.empty() && !handle->custom_vocabulary.empty()) {
                            apply_vocabulary_spelling_correction(
                                pending_output, handle->custom_vocabulary);
                        }
                    }
                    if (handle->audio_buffer.size() > handle->parakeet_chunk_cursor_bytes) {
                        const size_t preview_start_bytes = handle->parakeet_chunk_cursor_bytes > left_context_bytes
                            ? handle->parakeet_chunk_cursor_bytes - left_context_bytes
                            : 0;
                        StreamWindowDecodeResult preview_decode;
                        if (run_stream_window_transcribe(
                                handle,
                                transcribe_prompt,
                                tdt_no_vad_options_json,
                                handle->audio_buffer.data() + preview_start_bytes,
                                handle->audio_buffer.size() - preview_start_bytes,
                                preview_decode)) {
                            raw_json = preview_decode.raw_json;
                            if (pending_output.empty()) {
                                pending_output = parakeet_strip_recent_committed_prefix(
                                    handle->parakeet_committed_text,
                                    preview_decode.response);
                                if (!pending_output.empty() && !handle->custom_vocabulary.empty()) {
                                    apply_vocabulary_spelling_correction(
                                        pending_output, handle->custom_vocabulary);
                                }
                            }
                        }
                    }

                    handle->previous_parakeet_pending = pending_output;
                    handle->previous_parakeet_pending_ticks = pending_output.empty() ? 0 : 1;
                    handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
                    handle->previous_segments.clear();
                    handle->previous_cloud_handoff = false;
                    handle->parakeet_onset_active = false;
                    handle->parakeet_resume_guard_active = false;

                    CloudResponse empty_cloud_result;
                    const std::string json_response = build_stream_response(
                        raw_json,
                        "",
                        confirmed,
                        pending_output,
                        "[]",
                        false,
                        pcm_bytes_to_seconds(confirmed_audio_bytes) * 1000.0,
                        0,
                        0,
                        empty_cloud_result,
                        handle->stream_first_token_seen ? handle->stream_first_token_ms : 0.0,
                        raw_json == "{}" ? handle->last_stream_total_time_ms : -1.0,
                        raw_json == "{}" ? handle->last_stream_decode_tps : -1.0,
                        raw_json == "{}" ? handle->last_stream_decode_tokens : -1.0,
                        raw_json == "{}" ? handle->last_stream_decode_tokens : -1.0,
                        raw_json == "{}" ? handle->last_stream_raw_decoder_tps : -1.0);
                    if (json_response.length() >= buffer_size) {
                        last_error_message = "Response buffer too small";
                        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                        handle_error_response(last_error_message, response_buffer, buffer_size);
                        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
                        return -1;
                    }

                    std::strcpy(response_buffer, json_response.c_str());
                    return static_cast<int>(json_response.length());
                }
            }

            auto decode_window = [&](size_t window_start_bytes,
                                     size_t window_end_bytes,
                                     StreamWindowDecodeResult& out,
                                     float& out_offset_sec) -> bool {
                if (window_end_bytes <= window_start_bytes || window_end_bytes > handle->audio_buffer.size()) {
                    return false;
                }
                out_offset_sec = static_cast<float>(
                    handle->stream_audio_offset_sec + pcm_bytes_to_seconds(window_start_bytes));
                StreamWindowDecodeResult vad_decode;
                const bool vad_ok = run_stream_window_transcribe(
                    handle,
                    transcribe_prompt,
                    tdt_vad_options_json,
                    handle->audio_buffer.data() + window_start_bytes,
                    window_end_bytes - window_start_bytes,
                    vad_decode);
                if (vad_ok && (!vad_decode.response.empty() || !vad_decode.segments.empty())) {
                    out = std::move(vad_decode);
                    return true;
                }

                return run_stream_window_transcribe(
                    handle,
                    transcribe_prompt,
                    tdt_no_vad_options_json,
                    handle->audio_buffer.data() + window_start_bytes,
                    window_end_bytes - window_start_bytes,
                    out);
            };

            constexpr size_t kParakeetHoldbackWords = 1;
            constexpr size_t kParakeetSilenceFlushBytes = 16000; // ~0.5s silence
            constexpr size_t kParakeetStableFlushWords = 10;
            constexpr size_t kParakeetStableFlushTicks = 2;
            constexpr size_t kParakeetResumeGuardBytes = 24000; // ~0.75s speech
            constexpr size_t kParakeetOnsetLeadSilenceBytes = 16000; // ~0.5s lead-in
            constexpr size_t kParakeetOnsetMinConfirmWords = 6;

            const bool chunk_silent = is_pcm_chunk_silent(pcm_buffer, pcm_buffer_size, 0.012f);
            const size_t prior_silence_run_bytes = handle->parakeet_silence_run_bytes;
            const bool speech_resumed =
                !chunk_silent && prior_silence_run_bytes >= kParakeetSilenceFlushBytes;
            const size_t speech_chunk_start_bytes =
                handle->audio_buffer.size() >= pcm_buffer_size
                ? handle->audio_buffer.size() - pcm_buffer_size
                : 0;

            if (chunk_silent) {
                handle->parakeet_silence_run_bytes += pcm_buffer_size;
                handle->parakeet_speech_run_bytes = 0;
                if (handle->parakeet_silence_run_bytes >= kParakeetSilenceFlushBytes) {
                    handle->parakeet_resume_guard_active = true;
                    handle->parakeet_onset_active = false;
                    reset_parakeet_tdt_chunk_stream(handle);
                    const size_t retained_silence_bytes = std::min(
                        handle->parakeet_silence_run_bytes, kParakeetOnsetLeadSilenceBytes);
                    handle->parakeet_onset_start_bytes =
                        handle->audio_buffer.size() > retained_silence_bytes
                        ? handle->audio_buffer.size() - retained_silence_bytes
                        : 0;
                }
            } else {
                if (!handle->parakeet_onset_active &&
                    handle->parakeet_committed_text.empty() &&
                    handle->previous_parakeet_pending.empty()) {
                    handle->parakeet_onset_active = true;
                    handle->parakeet_onset_start_bytes = 0;
                }
                if (speech_resumed) {
                    handle->parakeet_resume_guard_active = true;
                    handle->parakeet_speech_run_bytes = 0;
                    handle->parakeet_onset_active = true;
                    const size_t retained_silence_bytes = std::min(
                        prior_silence_run_bytes, kParakeetOnsetLeadSilenceBytes);
                    handle->parakeet_onset_start_bytes =
                        speech_chunk_start_bytes > retained_silence_bytes
                        ? speech_chunk_start_bytes - retained_silence_bytes
                        : 0;
                }
                handle->parakeet_silence_run_bytes = 0;
                if (handle->parakeet_resume_guard_active) {
                    handle->parakeet_speech_run_bytes += pcm_buffer_size;
                    if (handle->parakeet_speech_run_bytes >= kParakeetResumeGuardBytes) {
                        handle->parakeet_resume_guard_active = false;
                        handle->parakeet_speech_run_bytes = 0;
                    }
                }
            }

            const bool detected_pause =
                handle->parakeet_silence_run_bytes >= kParakeetSilenceFlushBytes;
            const bool should_hold_cursor = handle->parakeet_onset_active;

            size_t processed_chunks = 0;
            if (!should_hold_cursor) {
                while (handle->parakeet_chunk_cursor_bytes + chunk_bytes + right_context_bytes <=
                           handle->audio_buffer.size() &&
                       processed_chunks < kParakeetTdtMaxChunksPerProcess) {
                    handle->parakeet_chunk_cursor_bytes += chunk_bytes;
                    ++processed_chunks;
                }
            }

            if (!handle->parakeet_onset_active &&
                handle->parakeet_chunk_cursor_bytes > left_context_bytes) {
                const size_t trim_bytes =
                    handle->parakeet_chunk_cursor_bytes - left_context_bytes;
                handle->stream_audio_offset_sec += static_cast<float>(
                    pcm_bytes_to_seconds(trim_bytes));
                handle->audio_buffer.erase(
                    handle->audio_buffer.begin(),
                    handle->audio_buffer.begin() + trim_bytes);
                handle->parakeet_chunk_cursor_bytes -= trim_bytes;
            }

            const bool onset_mode_active = handle->parakeet_onset_active;
            const size_t preview_start_bytes = onset_mode_active
                ? std::min(handle->parakeet_onset_start_bytes, handle->audio_buffer.size())
                : (handle->parakeet_chunk_cursor_bytes > left_context_bytes
                   ? handle->parakeet_chunk_cursor_bytes - left_context_bytes
                   : 0);
            const size_t preview_end_bytes = handle->audio_buffer.size();

            StreamWindowDecodeResult response_decode;
            float response_segments_offset_sec = 0.0f;
            if (!decode_window(preview_start_bytes, preview_end_bytes, response_decode, response_segments_offset_sec)) {
                last_error_message = "Transcription failed in stream process.";
                CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                cactus::telemetry::recordStreamTranscription(
                    handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr,
                    false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0,
                    last_error_message.c_str());
                return -1;
            }

            std::string confirmed;
            const std::string previous_pending = handle->previous_parakeet_pending;
            std::string pending_tail = previous_pending;
            const std::string normalized_response = parakeet_strip_recent_committed_prefix(
                handle->parakeet_committed_text, response_decode.response);
            if (!normalized_response.empty()) {
                if (previous_pending.empty() || (onset_mode_active && handle->parakeet_resume_guard_active)) {
                    pending_tail = normalized_response;
                } else {
                    const size_t stable_words = parakeet_common_prefix_words(
                        previous_pending, normalized_response);
                    const size_t confirm_words =
                        stable_words > kParakeetHoldbackWords
                        ? stable_words - kParakeetHoldbackWords
                        : 0;
                    confirmed = parakeet_take_first_words(normalized_response, confirm_words);
                    pending_tail = confirm_words > 0
                        ? parakeet_drop_first_words(normalized_response, confirm_words)
                        : normalized_response;
                }
            }

            const bool onset_commit_too_small =
                onset_mode_active &&
                !confirmed.empty() &&
                !detected_pause &&
                parakeet_word_count(confirmed) < kParakeetOnsetMinConfirmWords;
            if (onset_commit_too_small) {
                confirmed.clear();
                pending_tail = normalized_response;
            }

            const bool pending_equivalent =
                pending_tail == previous_pending ||
                (!pending_tail.empty() &&
                 !previous_pending.empty() &&
                 levenshtein_ci(pending_tail, previous_pending) <= 1);
            const bool pending_changed = !pending_equivalent;
            if (pending_changed) {
                handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
            }

            const bool stable_short_pending =
                !pending_tail.empty() &&
                !handle->parakeet_resume_guard_active &&
                pending_equivalent &&
                handle->previous_parakeet_pending_ticks >= kParakeetStableFlushTicks &&
                parakeet_word_count(pending_tail) <= kParakeetStableFlushWords;
            if ((detected_pause || stable_short_pending) && !pending_tail.empty()) {
                if (!confirmed.empty()) confirmed += " ";
                confirmed += pending_tail;
                pending_tail.clear();
            }

            double buffer_duration_ms = 0.0;
            bool cloud_handoff_triggered = false;
            uint64_t cloud_job_id = 0;
            uint64_t cloud_result_job_id = 0;
            CloudResponse cloud_result;

            if (!confirmed.empty()) {
                if (!handle->custom_vocabulary.empty()) {
                    apply_vocabulary_spelling_correction(confirmed, handle->custom_vocabulary);
                }

                std::string confirmed_delta = parakeet_emit_delta(
                    handle->parakeet_committed_text, confirmed);
                if (!confirmed_delta.empty()) {
                    if (response_decode.decode_tokens > 0.0) {
                        handle->stream_total_tokens += static_cast<int>(std::round(response_decode.decode_tokens));
                        handle->stream_cumulative_tokens += static_cast<int>(std::round(response_decode.decode_tokens));
                    }

                    if (!handle->stream_first_token_seen) {
                        auto now = std::chrono::steady_clock::now();
                        handle->stream_first_token_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                now - handle->stream_start).count();
                        handle->stream_first_token_seen = true;
                    }

                    if (!handle->stream_session_first_token_seen) {
                        auto now = std::chrono::steady_clock::now();
                        handle->stream_session_first_token_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                now - handle->stream_session_start).count();
                        handle->stream_session_first_token_seen = true;
                    }

                    if (!handle->parakeet_committed_text.empty()) {
                        handle->parakeet_committed_text += " ";
                    }
                    handle->parakeet_committed_text += confirmed_delta;
                    confirmed = confirmed_delta;
                    if (onset_mode_active) {
                        handle->parakeet_onset_active = false;
                        handle->parakeet_onset_start_bytes = 0;
                        handle->parakeet_chunk_cursor_bytes =
                            handle->audio_buffer.size() > right_context_bytes
                            ? handle->audio_buffer.size() - right_context_bytes
                            : 0;
                    }
                    buffer_duration_ms =
                        static_cast<double>(processed_chunks) * kParakeetTdtChunkSec * 1000.0;
                } else {
                    confirmed.clear();
                }
            }

            if (pending_tail.empty()) {
                handle->previous_parakeet_pending_ticks = 0;
            } else if (pending_equivalent) {
                handle->previous_parakeet_pending_ticks = std::min<size_t>(
                    handle->previous_parakeet_pending_ticks + 1, 1000000);
            } else {
                handle->previous_parakeet_pending_ticks = 1;
            }
            handle->previous_parakeet_pending = pending_tail;
            if (pending_tail.empty()) {
                handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
            }

            std::vector<TranscriptSegment> current_segments = response_decode.segments;
            const float segments_offset_sec = response_segments_offset_sec;
            const std::string json_str = response_decode.raw_json;
            const std::string pending_output = pending_tail;
            handle->previous_cloud_handoff = false;

            for (auto it = handle->pending_cloud_jobs.begin(); it != handle->pending_cloud_jobs.end(); ) {
                if (it->result.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    CloudResponse job_result = it->result.get();
                    CACTUS_LOG_INFO(
                        "cloud_handoff",
                        "Completed stream cloud job id=" << it->id
                            << " used_cloud=" << (job_result.used_cloud ? "true" : "false")
                            << " error=" << (job_result.error.empty() ? "none" : job_result.error)
                            << " transcript_chars=" << job_result.transcript.size());
                    handle->completed_cloud_results.push_back({it->id, std::move(job_result)});
                    it = handle->pending_cloud_jobs.erase(it);
                } else {
                    ++it;
                }
            }

            if (!handle->completed_cloud_results.empty()) {
                cloud_result_job_id = handle->completed_cloud_results.front().first;
                cloud_result = handle->completed_cloud_results.front().second;
                handle->completed_cloud_results.erase(handle->completed_cloud_results.begin());

                if (!cloud_result.api_key_hash.empty()) {
                    cactus::telemetry::setCloudKey(cloud_result.api_key_hash.c_str());
                }
            }

            constexpr int STREAM_TOKENS_CAP = 20000;
            constexpr double STREAM_DURATION_CAP_MS = 600000.0;
            auto now = std::chrono::steady_clock::now();
            double elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - handle->stream_start).count();
            if (handle->stream_total_tokens >= STREAM_TOKENS_CAP ||
                elapsed_ms >= STREAM_DURATION_CAP_MS) {
                double period_tps = (elapsed_ms > 0.0)
                    ? (static_cast<double>(handle->stream_total_tokens) * 1000.0) / elapsed_ms
                    : 0.0;

                double cumulative_elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - handle->stream_session_start).count();
                double cumulative_tps = (cumulative_elapsed_ms > 0.0)
                    ? (static_cast<double>(handle->stream_cumulative_tokens) * 1000.0) /
                        cumulative_elapsed_ms
                    : 0.0;

                cactus::telemetry::recordStreamTranscription(
                    handle->model_handle->model_name.c_str(),
                    true,
                    handle->stream_first_token_ms,
                    period_tps,
                    elapsed_ms,
                    handle->stream_total_tokens,
                    handle->stream_session_first_token_ms,
                    cumulative_tps,
                    cumulative_elapsed_ms,
                    handle->stream_cumulative_tokens,
                    ""
                );

                handle->stream_start = std::chrono::steady_clock::now();
                handle->stream_first_token_seen = false;
                handle->stream_first_token_ms = 0.0;
                handle->stream_total_tokens = 0;
            }

            std::string json_response = build_stream_response(
                json_str,
                json_string(json_str, "error"),
                confirmed,
                pending_output,
                serialize_segments_with_offset(current_segments, segments_offset_sec),
                cloud_handoff_triggered,
                buffer_duration_ms,
                cloud_job_id,
                cloud_result_job_id,
                cloud_result
            );

            if (json_response.length() >= buffer_size) {
                last_error_message = "Response buffer too small";
                CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                cactus::telemetry::recordStreamTranscription(
                    handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr,
                    false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0,
                    last_error_message.c_str());
                return -1;
            }

            std::strcpy(response_buffer, json_response.c_str());
            return static_cast<int>(json_response.length());
        }

        const uint8_t* decode_pcm = handle->audio_buffer.data();
        size_t decode_pcm_size = handle->audio_buffer.size();
        if (is_parakeet_tdt) {
            constexpr size_t kParakeetDecodeWindowBytes = 160000; // ~5.0s
            if (decode_pcm_size > kParakeetDecodeWindowBytes) {
                decode_pcm += (decode_pcm_size - kParakeetDecodeWindowBytes);
                decode_pcm_size = kParakeetDecodeWindowBytes;
            }
        }

        cactus::telemetry::setStreamMode(true);
        int result = cactus_transcribe(
            handle->model_handle,
            nullptr,
            transcribe_prompt,
            handle->transcribe_response_buffer,
            sizeof(handle->transcribe_response_buffer),
            effective_transcribe_options_json.empty() ? nullptr : effective_transcribe_options_json.c_str(),
            nullptr,
            nullptr,
            decode_pcm,
            decode_pcm_size);
        cactus::telemetry::setStreamMode(false);
        if (result < 0) {
            last_error_message = "Transcription failed in stream process.";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
            return -1;
        }

        std::string json_str(handle->transcribe_response_buffer);
        std::string response = suppress_unwanted_text(json_string(json_str, "response"));

        cactus_reset(handle->model_handle);

        auto current_segments = parse_segments(json_str);
        const float segments_offset_sec = static_cast<float>(handle->stream_audio_offset_sec);

        std::string confirmed;
        double buffer_duration_ms = 0.0;
        bool cloud_handoff_triggered = false;
        uint64_t cloud_job_id = 0;
        uint64_t cloud_result_job_id = 0;
        CloudResponse cloud_result;
        double chunk_decode_tokens = json_number(json_str, "decode_tokens");
        if (chunk_decode_tokens < 0.0) {
            chunk_decode_tokens = 0.0;
        }

        if (is_parakeet_tdt) {
            constexpr size_t kParakeetHoldbackWords = 1;
            constexpr size_t kParakeetTailBytes = 64000;         // ~2.0s left context
            constexpr size_t kParakeetSilenceFlushBytes = 16000; // ~0.5s silence
            constexpr size_t kParakeetStableFlushWords = 4;
            constexpr size_t kParakeetStableFlushTicks = 2;
            constexpr size_t kParakeetMediumTrimWords = 3;
            constexpr size_t kParakeetAggressiveTrimWords = 6;

            const std::string previous_pending = handle->previous_parakeet_pending;
            std::string pending_tail = previous_pending;
            const bool chunk_silent = is_pcm_chunk_silent(pcm_buffer, pcm_buffer_size, 0.012f);
            if (chunk_silent) {
                handle->parakeet_silence_run_bytes += pcm_buffer_size;
            } else {
                handle->parakeet_silence_run_bytes = 0;
            }
            if (!response.empty()) {
                const std::string normalized_response = parakeet_strip_recent_committed_prefix(
                    handle->parakeet_committed_text, response);
                if (previous_pending.empty()) {
                    pending_tail = normalized_response;
                } else {
                    const size_t stable_words = parakeet_common_prefix_words(previous_pending, normalized_response);
                    size_t confirm_words = 0;
                    if (stable_words > kParakeetHoldbackWords) {
                        confirm_words = stable_words - kParakeetHoldbackWords;
                    }

                    confirmed = parakeet_take_first_words(normalized_response, confirm_words);
                    if (confirm_words > 0) {
                        pending_tail = parakeet_drop_first_words(normalized_response, confirm_words);
                    } else {
                        pending_tail = normalized_response;
                    }
                }
            }

            const bool pending_equivalent =
                pending_tail == previous_pending ||
                (!pending_tail.empty() &&
                 !previous_pending.empty() &&
                 levenshtein_ci(pending_tail, previous_pending) <= 1);
            const bool pending_changed = !pending_equivalent;
            if (pending_changed) {
                handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
            }

            const bool detected_pause =
                handle->parakeet_silence_run_bytes >= kParakeetSilenceFlushBytes;
            const bool stable_short_pending =
                !pending_tail.empty() &&
                pending_equivalent &&
                handle->previous_parakeet_pending_ticks >= kParakeetStableFlushTicks &&
                parakeet_word_count(pending_tail) <= kParakeetStableFlushWords &&
                handle->audio_buffer.size() >=
                    (handle->previous_parakeet_audio_buffer_size + (kParakeetSilenceFlushBytes / 2));
            if ((detected_pause || stable_short_pending) && !pending_tail.empty()) {
                if (!confirmed.empty()) confirmed += " ";
                confirmed += pending_tail;
                pending_tail.clear();
            }

            if (!confirmed.empty()) {
                if (!handle->custom_vocabulary.empty()) {
                    apply_vocabulary_spelling_correction(confirmed, handle->custom_vocabulary);
                }

                std::string confirmed_delta = parakeet_emit_delta(
                    handle->parakeet_committed_text, confirmed);

                const size_t max_trim = handle->audio_buffer.size() > kParakeetTailBytes
                    ? handle->audio_buffer.size() - kParakeetTailBytes
                    : 0;
                const size_t confirmed_words = parakeet_word_count(confirmed);
                const size_t hop_bytes = std::clamp<size_t>(
                    pcm_buffer_size, static_cast<size_t>(4000), static_cast<size_t>(16000));
                size_t confirmed_bytes = std::min(hop_bytes, max_trim);
                if (confirmed_words >= kParakeetAggressiveTrimWords || pending_tail.empty()) {
                    confirmed_bytes = max_trim;
                } else if (confirmed_words >= kParakeetMediumTrimWords) {
                    confirmed_bytes = std::min(max_trim, hop_bytes * 2);
                }
                buffer_duration_ms = (confirmed_bytes / 2.0) / 16000.0 * 1000.0;

                if (!confirmed_delta.empty()) {
                    if (chunk_decode_tokens > 0.0) {
                        handle->stream_total_tokens += static_cast<int>(std::round(chunk_decode_tokens));
                        handle->stream_cumulative_tokens += static_cast<int>(std::round(chunk_decode_tokens));
                    }

                    if (!handle->stream_first_token_seen) {
                        auto now = std::chrono::steady_clock::now();
                        handle->stream_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
                        handle->stream_first_token_seen = true;
                    }

                    if (!handle->stream_session_first_token_seen) {
                        auto now = std::chrono::steady_clock::now();
                        handle->stream_session_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
                        handle->stream_session_first_token_seen = true;
                    }

                    if (!handle->parakeet_committed_text.empty()) {
                        handle->parakeet_committed_text += " ";
                    }
                    handle->parakeet_committed_text += confirmed_delta;
                }

                if (confirmed_bytes > 0) {
                    if (handle->previous_cloud_handoff && !confirmed.empty()) {
                        cloud_handoff_triggered = true;
                        std::vector<uint8_t> confirmed_audio(
                            handle->audio_buffer.begin(),
                            handle->audio_buffer.begin() + confirmed_bytes
                        );
                        auto wav = cloud_build_wav(confirmed_audio.data(), confirmed_audio.size());
                        std::string b64 = cloud_base64_encode(wav.data(), wav.size());
                        cloud_job_id = handle->next_cloud_job_id++;
                        CACTUS_LOG_INFO(
                            "cloud_handoff",
                            "Queued stream cloud job id=" << cloud_job_id
                                << " confirmed_chars=" << confirmed.size()
                                << " audio_bytes=" << confirmed_audio.size());
                        handle->pending_cloud_jobs.push_back({
                            cloud_job_id,
                            std::async(std::launch::async, cloud_transcribe_request, b64, confirmed, 15L, nullptr)
                        });
                    }

                    handle->stream_audio_offset_sec += static_cast<float>(confirmed_bytes) / (16000.0f * 2.0f);
                    handle->audio_buffer.erase(
                        handle->audio_buffer.begin(),
                        handle->audio_buffer.begin() + confirmed_bytes
                    );
                }

                confirmed = confirmed_delta;
                handle->previous_cloud_handoff = false;
                handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
            } else {
                handle->previous_cloud_handoff = json_bool(json_str, "cloud_handoff");
            }

            if (pending_tail.empty()) {
                handle->previous_parakeet_pending_ticks = 0;
            } else if (pending_equivalent) {
                handle->previous_parakeet_pending_ticks = std::min<size_t>(
                    handle->previous_parakeet_pending_ticks + 1, 1000000);
            } else {
                handle->previous_parakeet_pending_ticks = 1;
            }
            handle->previous_parakeet_pending = pending_tail;
            if (pending_tail.empty()) {
                handle->previous_parakeet_audio_buffer_size = handle->audio_buffer.size();
            }
            handle->previous_segments = current_segments;
        } else {
            size_t confirmed_segments = 0;
            float confirmed_end_sec = 0.0f;
            for (size_t i = 0; i < std::min(handle->previous_segments.size(), current_segments.size()); ++i) {
                if (handle->previous_segments[i].text != current_segments[i].text) {
                    break;
                }
                confirmed_segments = i + 1;
                confirmed_end_sec = std::min(handle->previous_segments[i].end, current_segments[i].end);
            }

            if (confirmed_segments > 0 && confirmed_end_sec > 0.0f) {
                buffer_duration_ms = confirmed_end_sec * 1000.0;

                for (size_t i = 0; i < confirmed_segments; ++i) {
                    if (!confirmed.empty()) confirmed += ' ';
                    confirmed += handle->previous_segments[i].text;
                }

                if (!handle->custom_vocabulary.empty()) {
                    apply_vocabulary_spelling_correction(confirmed, handle->custom_vocabulary);
                }

                if (chunk_decode_tokens > 0.0) {
                    handle->stream_total_tokens += static_cast<int>(std::round(chunk_decode_tokens));
                    handle->stream_cumulative_tokens += static_cast<int>(std::round(chunk_decode_tokens));
                }

                if (!handle->stream_first_token_seen) {
                    auto now = std::chrono::steady_clock::now();
                    handle->stream_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
                    handle->stream_first_token_seen = true;
                }

                if (!handle->stream_session_first_token_seen) {
                    auto now = std::chrono::steady_clock::now();
                    handle->stream_session_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
                    handle->stream_session_first_token_seen = true;
                }

                const size_t confirmed_bytes = std::min(
                    static_cast<size_t>(confirmed_end_sec * 16000.0) * 2,
                    handle->audio_buffer.size()
                );

                if (handle->previous_cloud_handoff && !confirmed.empty()) {
                    cloud_handoff_triggered = true;
                    std::vector<uint8_t> confirmed_audio(
                        handle->audio_buffer.begin(),
                        handle->audio_buffer.begin() + confirmed_bytes
                    );
                    auto wav = cloud_build_wav(confirmed_audio.data(), confirmed_audio.size());
                    std::string b64 = cloud_base64_encode(wav.data(), wav.size());
                    cloud_job_id = handle->next_cloud_job_id++;
                    CACTUS_LOG_INFO(
                        "cloud_handoff",
                        "Queued stream cloud job id=" << cloud_job_id
                            << " confirmed_chars=" << confirmed.size()
                            << " audio_bytes=" << confirmed_audio.size());
                    handle->pending_cloud_jobs.push_back({
                        cloud_job_id,
                        std::async(std::launch::async, cloud_transcribe_request, b64, confirmed, 15L, nullptr)
                    });
                }

                handle->stream_audio_offset_sec += confirmed_end_sec;
                handle->audio_buffer.erase(
                    handle->audio_buffer.begin(),
                    handle->audio_buffer.begin() + confirmed_bytes
                );
                constexpr float kSegmentEpsilonSec = 0.02f;
                size_t current_tail_start = 0;
                while (current_tail_start < current_segments.size() &&
                       current_segments[current_tail_start].end <= confirmed_end_sec + kSegmentEpsilonSec) {
                    ++current_tail_start;
                }
                handle->previous_segments = std::vector<TranscriptSegment>(
                    current_segments.begin() + current_tail_start, current_segments.end());
                handle->previous_cloud_handoff = false;
            } else {
                handle->previous_segments = current_segments;
                handle->previous_cloud_handoff = json_bool(json_str, "cloud_handoff");
            }
        }

        for (auto it = handle->pending_cloud_jobs.begin(); it != handle->pending_cloud_jobs.end(); ) {
            if (it->result.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                CloudResponse job_result = it->result.get();
                CACTUS_LOG_INFO(
                    "cloud_handoff",
                    "Completed stream cloud job id=" << it->id
                        << " used_cloud=" << (job_result.used_cloud ? "true" : "false")
                        << " error=" << (job_result.error.empty() ? "none" : job_result.error)
                        << " transcript_chars=" << job_result.transcript.size());
                handle->completed_cloud_results.push_back({it->id, std::move(job_result)});
                it = handle->pending_cloud_jobs.erase(it);
            } else {
                ++it;
            }
        }

        if (!handle->completed_cloud_results.empty()) {
            cloud_result_job_id = handle->completed_cloud_results.front().first;
            cloud_result = handle->completed_cloud_results.front().second;
            handle->completed_cloud_results.erase(handle->completed_cloud_results.begin());

            if (!cloud_result.api_key_hash.empty()) {
                cactus::telemetry::setCloudKey(cloud_result.api_key_hash.c_str());
            }
        }

        constexpr int STREAM_TOKENS_CAP = 20000;
        constexpr double STREAM_DURATION_CAP_MS = 600000.0;
        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
        if (handle->stream_total_tokens >= STREAM_TOKENS_CAP || elapsed_ms >= STREAM_DURATION_CAP_MS) {
            double period_tps = (elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_total_tokens) * 1000.0) / elapsed_ms : 0.0;

            double cumulative_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
            double cumulative_tps = (cumulative_elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_cumulative_tokens) * 1000.0) / cumulative_elapsed_ms : 0.0;

            cactus::telemetry::recordStreamTranscription(
                handle->model_handle->model_name.c_str(),
                true,
                handle->stream_first_token_ms,
                period_tps,
                elapsed_ms,
                handle->stream_total_tokens,
                handle->stream_session_first_token_ms,
                cumulative_tps,
                cumulative_elapsed_ms,
                handle->stream_cumulative_tokens,
                ""
            );

            handle->stream_start = std::chrono::steady_clock::now();
            handle->stream_first_token_seen = false;
            handle->stream_first_token_ms = 0.0;
            handle->stream_total_tokens = 0;
        }

        std::string pending_output = response;
        if (is_parakeet_tdt && !handle->previous_parakeet_pending.empty()) {
            pending_output = handle->previous_parakeet_pending;
        }

        std::string json_response = build_stream_response(
            json_str,
            json_string(json_str, "error"),
            confirmed,
            pending_output,
            serialize_segments_with_offset(current_segments, segments_offset_sec),
            cloud_handoff_triggered,
            buffer_duration_ms,
            cloud_job_id,
            cloud_result_job_id,
            cloud_result
        );

        if (json_response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, last_error_message.c_str());
            return -1;
        }

        std::strcpy(response_buffer, json_response.c_str());
        return static_cast<int>(json_response.length());
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_process: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription processing";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response("Unknown error during stream processing", response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(nullptr, false, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, "Unknown error during stream processing");
        return -1;
    }
}

int cactus_stream_transcribe_stop(
    cactus_stream_transcribe_t stream,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        return -1;
    }

    auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);
    auto clear_stream_vocab_bias = [&]() {
        if (handle->has_custom_vocabulary_bias &&
            handle->model_handle &&
            handle->model_handle->model) {
            handle->model_handle->model->clear_vocab_bias();
        }
    };

    if (!response_buffer || buffer_size == 0) {
        clear_stream_vocab_bias();
        delete handle;
        return 0;
    }

    try {
        const auto model_type = handle->model_handle->model->get_config().model_type;
        bool is_moonshine = model_type == cactus::engine::Config::ModelType::MOONSHINE;
        bool is_parakeet_tdt =
            model_type == cactus::engine::Config::ModelType::PARAKEET_TDT;
        bool is_parakeet_ctc =
            model_type == cactus::engine::Config::ModelType::PARAKEET;
        bool is_parakeet =
            is_parakeet_ctc || is_parakeet_tdt;
        bool is_tinyllama = model_type == cactus::engine::Config::ModelType::GEMMA4;

        std::string final_confirmed;
        if (is_parakeet_tdt) {
            final_confirmed = handle->parakeet_committed_text;
            if (!(handle->parakeet_tdt_chunked_stream && handle->parakeet_tdt_decode_context.initialized) &&
                !handle->previous_parakeet_pending.empty()) {
                const std::string normalized_pending = parakeet_strip_recent_committed_prefix(
                    final_confirmed, handle->previous_parakeet_pending);
                const std::string pending_delta = parakeet_emit_delta(
                    final_confirmed, normalized_pending);
                if (!pending_delta.empty()) {
                    if (!final_confirmed.empty()) final_confirmed += " ";
                    final_confirmed += pending_delta;
                }
            }
        } else {
            for (const auto& seg : handle->previous_segments) {
                if (!final_confirmed.empty()) final_confirmed += ' ';
                final_confirmed += seg.text;
            }
        }

        if (is_parakeet_tdt &&
            handle->parakeet_tdt_chunked_stream &&
            handle->parakeet_tdt_decode_context.initialized) {
            auto& ctx = handle->parakeet_tdt_decode_context;
            auto* tdt_model = static_cast<cactus::engine::ParakeetTDTModel*>(
                handle->model_handle->model.get());
            const size_t total_samples = ctx.audio_samples.size();
            if (total_samples > ctx.samples_decoded_up_to) {
                constexpr size_t kTdtLeftContextSamples = 8 * 16000;
                const size_t window_start_sample = ctx.samples_decoded_up_to > kTdtLeftContextSamples
                    ? ctx.samples_decoded_up_to - kTdtLeftContextSamples
                    : 0;
                std::vector<float> window_audio(
                    ctx.audio_samples.begin() + window_start_sample,
                    ctx.audio_samples.end());

                auto cfg = get_parakeet_spectrogram_config();
                const size_t waveform_samples = window_audio.size();
                apply_preemphasis(window_audio, 0.97f);
                window_audio = ctx.audio_processor.compute_spectrogram(window_audio, cfg);
                normalize_parakeet_log_mel(window_audio, ctx.mel_bins);
                size_t valid_frames = waveform_samples / cfg.hop_length;
                if (valid_frames == 0) valid_frames = 1;
                trim_mel_frames(window_audio, ctx.mel_bins, valid_frames);

                const uint32_t subsampling =
                    std::max<uint32_t>(1, handle->model_handle->model->get_config().subsampling_factor);
                const size_t samples_per_enc_frame = cfg.hop_length * subsampling;
                const size_t context_samples = ctx.samples_decoded_up_to - window_start_sample;
                const size_t decode_start_frame = context_samples / samples_per_enc_frame;
                const size_t total_window_samples = total_samples - window_start_sample;
                const size_t decode_end_frame = total_window_samples / samples_per_enc_frame;

                if (decode_end_frame > decode_start_frame) {
                    cactus::engine::ParakeetTDTModel::ChunkStreamResult chunk_decode;
                    {
                        std::lock_guard<std::mutex> lock(handle->model_handle->model_mutex);
                        cactus::telemetry::setStreamMode(true);
                        try {
                            chunk_decode = tdt_model->decode_chunk_stream(
                                window_audio,
                                decode_start_frame,
                                decode_start_frame,
                                decode_end_frame,
                                ctx.decoder_state);
                            cactus_reset(handle->model_handle);
                        } catch (...) {
                            cactus::telemetry::setStreamMode(false);
                            throw;
                        }
                        cactus::telemetry::setStreamMode(false);
                    }

                    std::string confirmed_flush = suppress_unwanted_text(chunk_decode.confirmed_text);
                    std::string pending_flush = suppress_unwanted_text(chunk_decode.pending_text);
                    if (!confirmed_flush.empty() && !handle->custom_vocabulary.empty()) {
                        apply_vocabulary_spelling_correction(confirmed_flush, handle->custom_vocabulary);
                    }
                    if (!pending_flush.empty() && !handle->custom_vocabulary.empty()) {
                        apply_vocabulary_spelling_correction(pending_flush, handle->custom_vocabulary);
                    }
                    parakeet_append_monotonic_text(final_confirmed, confirmed_flush);
                    parakeet_append_monotonic_text(final_confirmed, pending_flush);
                }
            } else if (!handle->previous_parakeet_pending.empty()) {
                parakeet_append_monotonic_text(final_confirmed, handle->previous_parakeet_pending);
            }
        }

        if (!handle->audio_buffer.empty()) {
            std::string whisper_prompt = "<|startoftranscript|><|" + handle->options.language + "|><|transcribe|><|notimestamps|>";
            const char* transcribe_prompt =
                is_tinyllama ? "Transcribe the audio." :
                (is_moonshine || is_parakeet) ? "" :
                whisper_prompt.c_str();
            std::string effective_transcribe_options_json = handle->transcribe_options_json;
            if (is_parakeet && !is_parakeet_tdt) {
                effective_transcribe_options_json = ensure_json_bool_option(
                    effective_transcribe_options_json, "use_vad", false);
            }

            if (is_parakeet_tdt) {
                const bool use_chunked_stop =
                    handle->parakeet_tdt_chunked_stream &&
                    !handle->parakeet_tdt_chunked_blocked_until_pause &&
                    (handle->parakeet_tdt_decoder_state.initialized ||
                     !handle->parakeet_tdt_resume_checkpoints.empty() ||
                     !handle->parakeet_tdt_pending_raw.empty());
                if (use_chunked_stop) {
                    final_confirmed = handle->parakeet_committed_text;
                    constexpr double kParakeetTdtLeftContextSec = 2.0;
                    const size_t left_context_bytes = seconds_to_pcm_bytes(kParakeetTdtLeftContextSec);
                    if (handle->parakeet_chunk_cursor_bytes < handle->audio_buffer.size()) {
                        const size_t chunk_start_bytes = handle->parakeet_chunk_cursor_bytes;
                        const size_t chunk_end_bytes = handle->audio_buffer.size();
                        const size_t window_start_bytes = chunk_start_bytes > left_context_bytes
                            ? chunk_start_bytes - left_context_bytes
                            : 0;
                        cactus::engine::ParakeetTDTModel::ChunkStreamResult chunk_decode;
                        cactus::engine::ParakeetTDTModel::ChunkStreamState chunk_state;
                        if (run_parakeet_tdt_chunk_decode(
                                handle,
                                window_start_bytes,
                                chunk_end_bytes,
                                chunk_start_bytes,
                                chunk_end_bytes,
                                chunk_decode,
                                chunk_state)) {
                            std::string confirmed_chunk = suppress_unwanted_text(chunk_decode.confirmed_text);
                            std::string pending_chunk = suppress_unwanted_text(chunk_decode.pending_text);
                            if (!confirmed_chunk.empty() && !handle->custom_vocabulary.empty()) {
                                apply_vocabulary_spelling_correction(confirmed_chunk, handle->custom_vocabulary);
                            }
                            if (!pending_chunk.empty() && !handle->custom_vocabulary.empty()) {
                                apply_vocabulary_spelling_correction(pending_chunk, handle->custom_vocabulary);
                            }
                            parakeet_append_monotonic_text(final_confirmed, confirmed_chunk);
                            handle->parakeet_tdt_pending_raw =
                                strip_unwanted_text_preserving_edges(chunk_decode.pending_text);
                            handle->parakeet_chunk_cursor_bytes = chunk_end_bytes;
                            handle->parakeet_tdt_decoder_state = chunk_state;
                            handle->parakeet_committed_until_sec = std::max(
                                handle->parakeet_committed_until_sec,
                                static_cast<float>(
                                    handle->stream_audio_offset_sec +
                                    pcm_bytes_to_seconds(
                                        std::min(
                                            chunk_end_bytes,
                                            std::max(
                                                window_start_bytes,
                                                window_start_bytes + seconds_to_pcm_bytes(
                                                    chunk_decode.resume_end_sec))))));
                        }
                    } else if (!handle->previous_parakeet_pending.empty()) {
                        const std::string normalized_pending = parakeet_strip_recent_committed_prefix(
                            final_confirmed, handle->previous_parakeet_pending);
                        const std::string pending_delta = parakeet_emit_delta(
                            final_confirmed, normalized_pending);
                        if (!pending_delta.empty()) {
                            if (!final_confirmed.empty()) final_confirmed += " ";
                            final_confirmed += pending_delta;
                        }
                    }
                    if (!handle->parakeet_tdt_pending_raw.empty()) {
                        std::string pending_flush = suppress_unwanted_text(
                            handle->parakeet_tdt_pending_raw);
                        if (!pending_flush.empty() && !handle->custom_vocabulary.empty()) {
                            apply_vocabulary_spelling_correction(
                                pending_flush, handle->custom_vocabulary);
                        }
                        parakeet_append_monotonic_text(final_confirmed, pending_flush);
                        handle->parakeet_tdt_pending_raw.clear();
                    }
                    handle->parakeet_committed_text = final_confirmed;
                } else {
                    constexpr double kParakeetTdtLeftContextSec = 2.0;
                    constexpr float kParakeetTdtCommitEpsilonSec = 0.03f;
                    const size_t left_context_bytes = seconds_to_pcm_bytes(kParakeetTdtLeftContextSec);
                    const size_t flush_start_bytes = handle->parakeet_onset_active
                        ? std::min(handle->parakeet_onset_start_bytes, handle->audio_buffer.size())
                        : (handle->parakeet_chunk_cursor_bytes > left_context_bytes
                           ? handle->parakeet_chunk_cursor_bytes - left_context_bytes
                           : 0);
                    const std::string tdt_vad_options_json = ensure_json_bool_option(
                        handle->transcribe_options_json, "use_vad", true);
                    const std::string tdt_no_vad_options_json = ensure_json_bool_option(
                        handle->transcribe_options_json, "use_vad", false);

                    StreamWindowDecodeResult flush_decode;
                    bool flush_ok = run_stream_window_transcribe(
                            handle,
                            transcribe_prompt,
                            tdt_vad_options_json,
                            handle->audio_buffer.data() + flush_start_bytes,
                            handle->audio_buffer.size() - flush_start_bytes,
                            flush_decode);
                    if (flush_ok && flush_decode.response.empty() && flush_decode.segments.empty()) {
                        flush_ok = run_stream_window_transcribe(
                            handle,
                            transcribe_prompt,
                            tdt_no_vad_options_json,
                            handle->audio_buffer.data() + flush_start_bytes,
                            handle->audio_buffer.size() - flush_start_bytes,
                            flush_decode);
                    }
                    if (flush_ok) {
                        const float flush_offset_sec = static_cast<float>(
                            handle->stream_audio_offset_sec + pcm_bytes_to_seconds(flush_start_bytes));
                        float flushed_until_sec = handle->parakeet_committed_until_sec;
                        std::string flushed_text = join_segments_by_end_time(
                            flush_decode.segments,
                            flush_offset_sec,
                            handle->parakeet_committed_until_sec + kParakeetTdtCommitEpsilonSec,
                            std::numeric_limits<float>::max(),
                            &flushed_until_sec);
                        if (flushed_text.empty()) {
                            flushed_text = parakeet_strip_recent_committed_prefix(
                                final_confirmed, flush_decode.response);
                        }
                        const std::string effective_flush_delta =
                            parakeet_emit_delta(final_confirmed, flushed_text);
                        if (!effective_flush_delta.empty()) {
                            if (!final_confirmed.empty()) final_confirmed += " ";
                            final_confirmed += effective_flush_delta;
                        }
                        if (flushed_until_sec > handle->parakeet_committed_until_sec) {
                            handle->parakeet_committed_until_sec = flushed_until_sec;
                        }
                    } else {
                        CACTUS_LOG_WARN(
                            "stream_transcribe_stop",
                            "Final flush transcription failed; returning previously confirmed transcript.");
                    }
                }
            } else {
                cactus::telemetry::setStreamMode(true);
                const int flush_result = cactus_transcribe(
                    handle->model_handle,
                    nullptr,
                    transcribe_prompt,
                    handle->transcribe_response_buffer,
                    sizeof(handle->transcribe_response_buffer),
                    effective_transcribe_options_json.empty() ? nullptr : effective_transcribe_options_json.c_str(),
                    nullptr,
                    nullptr,
                    handle->audio_buffer.data(),
                    handle->audio_buffer.size());
                cactus::telemetry::setStreamMode(false);

                cactus_reset(handle->model_handle);

                if (flush_result >= 0) {
                    std::string flush_json(handle->transcribe_response_buffer);
                    std::string flushed_text = suppress_unwanted_text(json_string(flush_json, "response"));
                    if (flushed_text.empty()) {
                        auto flush_segments = parse_segments(flush_json);
                        for (const auto& seg : flush_segments) {
                            if (!flushed_text.empty()) flushed_text += ' ';
                            flushed_text += seg.text;
                        }
                    }
                    if (!flushed_text.empty()) {
                        if (is_parakeet_tdt) {
                            const std::string normalized_flush = parakeet_strip_recent_committed_prefix(
                                final_confirmed, flushed_text);
                            const std::string effective_flush_delta =
                                parakeet_emit_delta(final_confirmed, normalized_flush);
                            if (!effective_flush_delta.empty()) {
                                if (!final_confirmed.empty()) final_confirmed += " ";
                                final_confirmed += effective_flush_delta;
                            }
                        } else {
                            final_confirmed = flushed_text;
                        }
                    }
                } else {
                    CACTUS_LOG_WARN(
                        "stream_transcribe_stop",
                        "Final flush transcription failed; returning previously confirmed segments.");
                }
            }
        }

        if (!handle->custom_vocabulary.empty()) {
            apply_vocabulary_spelling_correction(final_confirmed, handle->custom_vocabulary);
        }

        std::string json_response = "{\"success\":true,\"confirmed\":\"" +
            escape_json(final_confirmed) + "\"}";

        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_start).count();
        double period_tps = (elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_total_tokens) * 1000.0) / elapsed_ms : 0.0;

        double cumulative_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - handle->stream_session_start).count();
        double cumulative_tps = (cumulative_elapsed_ms > 0.0) ? (static_cast<double>(handle->stream_cumulative_tokens) * 1000.0) / cumulative_elapsed_ms : 0.0;

        cactus::telemetry::recordStreamTranscription(
            handle->model_handle->model_name.c_str(),
            true,
            handle->stream_first_token_ms,
            period_tps,
            elapsed_ms,
            handle->stream_total_tokens,
            handle->stream_session_first_token_ms,
            cumulative_tps,
            cumulative_elapsed_ms,
            handle->stream_cumulative_tokens,
            ""
        );

        if (json_response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, last_error_message.c_str());
            clear_stream_vocab_bias();
            delete handle;
            return -1;
        }

        std::strcpy(response_buffer, json_response.c_str());
        clear_stream_vocab_bias();
        delete handle;
        return static_cast<int>(json_response.length());
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_stop: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, e.what());
        clear_stream_vocab_bias();
        delete handle;
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription stop";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response("Unknown error during stream stop", response_buffer, buffer_size);
        cactus::telemetry::recordStreamTranscription(handle->model_handle ? handle->model_handle->model_name.c_str() : nullptr, false, handle->stream_first_token_ms, 0.0, 0.0, handle->stream_total_tokens, handle->stream_session_first_token_ms, 0.0, 0.0, handle->stream_cumulative_tokens, "Unknown error during stream stop");
        clear_stream_vocab_bias();
        delete handle;
        return -1;
    }
}

}
