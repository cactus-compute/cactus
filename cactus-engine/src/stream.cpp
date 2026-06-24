#include "../cactus_engine.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace cactus::ffi;

namespace {

constexpr size_t kScratchSize = 1u << 16;
constexpr size_t kSampleRate = 16000;
constexpr float kSampleRateF = 16000.0f;
constexpr size_t kLeftContextSamples = 8 * kSampleRate;   // left context re-fed each window
constexpr size_t kRightContextSamples = 1 * kSampleRate;  // look-ahead before confirming
constexpr size_t kChunkSamples = 1 * kSampleRate;         // warm decode step
constexpr size_t kColdStartSamples = 6 * kSampleRate;     // first decode from empty state
constexpr size_t kResumeColdSamples = 2 * kSampleRate;    // first decode after silence
constexpr size_t kSilenceResetSamples = 3 * kSampleRate;  // silence run that triggers a cold restart
constexpr size_t kMaxDecodeSamples = 10 * kSampleRate;

struct StreamStats {
    size_t decode_tokens = 0;
    double total_time_ms = 0.0;
    double decode_tps = 0.0;
    double time_to_first_token_ms = 0.0;

    void finalize() {
        if (total_time_ms > 0.0) decode_tps = decode_tokens * 1000.0 / total_time_ms;
    }
};

struct StreamTranscribe {
    CactusModelHandle* model = nullptr;
    std::string options_json;
    bool is_parakeet = false;

    std::vector<float> samples;
    size_t samples_decoded_up_to = 0;
    size_t silence_run = 0;
    bool cold_restart = false;

    cactus::engine::Model::ParakeetTdtStreamState pstate;
    std::vector<uint32_t> committed_tokens;
    std::string emitted_text;
    std::string previous_pending;

    std::vector<TranscriptSegment> whisper_prev_segments;
    size_t whisper_last_len = 0;
};

double json_num(const std::string& json, const std::string& key) {
    float v = 0.0f;
    return try_parse_json_float(json, key, v) ? static_cast<double>(v) : 0.0;
}

std::vector<int16_t> to_pcm16(const std::vector<float>& samples) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float x = std::max(-32768.0f, std::min(32767.0f, samples[i] * 32768.0f));
        pcm[i] = static_cast<int16_t>(x);
    }
    return pcm;
}

std::string with_timestamps_option(std::string opts) {
    if (opts.find("\"timestamps\"") != std::string::npos) return opts;
    if (opts.empty() || opts == "{}") return "{\"timestamps\":true}";
    return opts.substr(0, opts.find_last_of('}')) + ",\"timestamps\":true}";
}

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(' '), e = s.find_last_not_of(' ');
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

std::string strip_annotations(const std::string& text) {
    static const std::regex pattern(R"(\([^)]*\)|\[[^\]]*\]|\.\.\.)");
    std::string out = std::regex_replace(text, pattern, "");
    out = std::regex_replace(out, std::regex(R"(\s+)"), " ");
    return trim(out);
}

std::vector<TranscriptSegment> parse_segments(const std::string& json) {
    std::vector<TranscriptSegment> segs;
    for (const std::string& obj : split_json_array(json_array_field(json, "segments"))) {
        float start = 0.0f, end = 0.0f;
        try_parse_json_float(obj, "start", start);
        try_parse_json_float(obj, "end", end);
        segs.push_back({start, end, strip_annotations(json_string_field(obj, "text"))});
    }
    return segs;
}

std::vector<TranscriptSegment> whisper_transcribe(StreamTranscribe* s, const std::vector<float>& samples,
                                                  StreamStats& stats) {
    std::vector<int16_t> pcm = to_pcm16(samples);
    std::string scratch(kScratchSize, '\0');
    const std::string opts = with_timestamps_option(s->options_json);
    const bool prev_should_stop = s->model->should_stop.load();
    const int rc = cactus_transcribe(
        static_cast<cactus_model_t>(s->model), nullptr, nullptr,
        scratch.data(), scratch.size(), opts.c_str(),
        nullptr, nullptr,
        reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size() * sizeof(int16_t));
    if (prev_should_stop) s->model->should_stop.store(true);
    if (rc <= 0) return {};
    const std::string json(scratch.c_str());
    stats.decode_tps = json_num(json, "decode_tps");
    stats.total_time_ms = json_num(json, "total_time_ms");
    stats.time_to_first_token_ms = json_num(json, "time_to_first_token_ms");
    stats.decode_tokens = static_cast<size_t>(json_num(json, "decode_tokens"));
    return parse_segments(json);
}

std::vector<float> window_features(std::vector<float> window, size_t mel_bins) {
    if (window.empty()) return {};
    auto cfg = cactus::audio::get_parakeet_spectrogram_config();
    const size_t waveform_samples = window.size();
    cactus::audio::apply_preemphasis(window, 0.97f);
    std::vector<float> features = cactus::audio::compute_spectrogram_graph(
        window, cfg, mel_bins, 0.0f, 8000.0f, cactus::audio::WHISPER_SAMPLE_RATE, 0, 0);
    cactus::audio::normalize_parakeet_log_mel(features, mel_bins);
    size_t valid_frames = waveform_samples / cfg.hop_length;
    if (valid_frames == 0) valid_frames = 1;
    cactus::audio::trim_mel_frames(features, mel_bins, valid_frames);
    return features;
}

std::string parakeet_decode_window(StreamTranscribe* s, size_t window_start, size_t window_end,
                                   size_t decode_start_frame, size_t decode_end_frame,
                                   bool is_final, std::string* pending_text, StreamStats& stats) {
    if (pending_text) pending_text->clear();
    auto* model = s->model->model.get();
    const size_t mel_bins = std::max<size_t>(1, static_cast<size_t>(model->get_config().num_mel_bins));
    std::vector<float> features = window_features(
        std::vector<float>(s->samples.begin() + window_start, s->samples.begin() + window_end), mel_bins);
    if (features.empty()) return "";

    s->pstate.time_index = decode_start_frame;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> tokens = model->transcribe_parakeet_tdt(
        features, &s->pstate, is_final, is_final ? 0 : decode_end_frame);
    stats.total_time_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.decode_tokens += s->pstate.decoded_tokens;

    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) return "";

    s->committed_tokens.insert(s->committed_tokens.end(), tokens.begin(), tokens.end());
    std::string full = tokenizer->decode(s->committed_tokens);
    std::string delta = full.size() > s->emitted_text.size() ? full.substr(s->emitted_text.size()) : std::string();
    s->emitted_text = full;

    if (pending_text && !s->pstate.pending.empty()) {
        std::vector<uint32_t> combined = s->committed_tokens;
        combined.insert(combined.end(), s->pstate.pending.begin(), s->pstate.pending.end());
        std::string with_pending = tokenizer->decode(combined);
        if (with_pending.size() > full.size()) *pending_text = with_pending.substr(full.size());
    }

    if (!is_final && !tokens.empty() && s->pstate.confirmed_sec > 0.0f) {
        s->samples_decoded_up_to = window_start + static_cast<size_t>(s->pstate.confirmed_sec * kSampleRateF);
    }
    return delta;
}

size_t parakeet_spf(StreamTranscribe* s) {
    const uint32_t subsampling = std::max<uint32_t>(1, s->model->model->get_config().subsampling_factor);
    return cactus::audio::get_parakeet_spectrogram_config().hop_length * subsampling;
}

std::string parakeet_process(StreamTranscribe* s, std::string& pending, StreamStats& stats) {
    std::lock_guard<std::mutex> lock(s->model->model_mutex);
    const size_t spf = parakeet_spf(s);

    std::string confirmed;
    for (;;) {
        const size_t total = s->samples.size();
        const size_t decodable = total > kRightContextSamples ? total - kRightContextSamples : 0;
        const bool cold = s->samples_decoded_up_to == 0 || s->cold_restart;
        const size_t cold_min = s->cold_restart ? kResumeColdSamples : kColdStartSamples;
        const size_t min_chunk = cold ? cold_min : kChunkSamples;
        if (decodable <= s->samples_decoded_up_to ||
            decodable - s->samples_decoded_up_to < min_chunk) {
            break;
        }
        const size_t decode_to = std::min(decodable, s->samples_decoded_up_to + kMaxDecodeSamples);
        const size_t window_start = cold ? s->samples_decoded_up_to
            : (s->samples_decoded_up_to > kLeftContextSamples ? s->samples_decoded_up_to - kLeftContextSamples : 0);
        const size_t window_end = decode_to + kRightContextSamples;
        const size_t decode_start_frame = (s->samples_decoded_up_to - window_start) / spf;
        const size_t decode_end_frame = decode_start_frame + (decode_to - s->samples_decoded_up_to) / spf;
        if (cold) s->pstate = {};

        std::string pend;
        const size_t prev_cursor = s->samples_decoded_up_to;
        const size_t tokens_before = stats.decode_tokens;
        confirmed += parakeet_decode_window(s, window_start, window_end,
                                            decode_start_frame, decode_end_frame, false, &pend, stats);
        pending = pend;
        if (s->pstate.decoded_tokens > 0) { s->cold_restart = false; s->silence_run = 0; }
        if (s->samples_decoded_up_to > prev_cursor) continue;
        if (s->pstate.decoded_tokens == 0) {
            s->silence_run += decode_to - s->samples_decoded_up_to;
            s->samples_decoded_up_to = decode_to;
            if (s->silence_run >= kSilenceResetSamples) s->cold_restart = true;
            continue;
        }
        if (decode_to - s->samples_decoded_up_to < cold_min) break;
        stats.decode_tokens = tokens_before;
        confirmed += parakeet_decode_window(s, window_start, window_end,
                                            decode_start_frame, decode_end_frame, true, &pend, stats);
        pending = pend;
        s->samples_decoded_up_to = decode_to;
    }

    const size_t keep_from = s->samples_decoded_up_to > kLeftContextSamples
        ? s->samples_decoded_up_to - kLeftContextSamples : 0;
    if (keep_from > kLeftContextSamples) {
        s->samples.erase(s->samples.begin(), s->samples.begin() + keep_from);
        s->samples_decoded_up_to -= keep_from;
    }

    stats.finalize();
    if (confirmed.empty() && pending.empty()) pending = s->previous_pending;
    else s->previous_pending = pending;
    return confirmed;
}

std::string parakeet_flush(StreamTranscribe* s, StreamStats& stats) {
    std::lock_guard<std::mutex> lock(s->model->model_mutex);
    const size_t total = s->samples.size();
    if (total <= s->samples_decoded_up_to) return "";
    const size_t spf = parakeet_spf(s);
    const bool cold = s->samples_decoded_up_to == 0 || s->cold_restart;
    if (cold) s->pstate = {};
    const size_t window_start = cold
        ? s->samples_decoded_up_to
        : (s->samples_decoded_up_to > kLeftContextSamples ? s->samples_decoded_up_to - kLeftContextSamples : 0);
    const size_t decode_start_frame = (s->samples_decoded_up_to - window_start) / spf;
    std::string confirmed = parakeet_decode_window(s, window_start, total, decode_start_frame, 0, true, nullptr, stats);
    stats.finalize();
    return confirmed;
}

std::string segments_text(const std::vector<TranscriptSegment>& segs, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to && i < segs.size(); ++i) {
        if (segs[i].text.empty()) continue;
        if (!out.empty()) out += ' ';
        out += segs[i].text;
    }
    return out;
}

std::string whisper_process(StreamTranscribe* s, std::string& pending, StreamStats& stats) {
    if (s->samples.size() < s->whisper_last_len + kChunkSamples) {
        pending = s->previous_pending;
        return "";
    }
    s->whisper_last_len = s->samples.size();
    std::vector<TranscriptSegment> segs = whisper_transcribe(s, s->samples, stats);

    size_t agree = 0;
    float confirmed_end_sec = 0.0f;
    while (agree < segs.size() && agree < s->whisper_prev_segments.size() &&
           segs[agree].text == s->whisper_prev_segments[agree].text) {
        confirmed_end_sec = std::min(segs[agree].end, s->whisper_prev_segments[agree].end);
        ++agree;
    }

    std::string confirmed;
    pending = segments_text(segs, agree, segs.size());
    if (agree > 0 && confirmed_end_sec > 0.0f) {
        confirmed = segments_text(segs, 0, agree);
        const size_t cut = std::min(static_cast<size_t>(confirmed_end_sec * kSampleRateF), s->samples.size());
        s->samples.erase(s->samples.begin(), s->samples.begin() + cut);
        s->whisper_last_len -= cut;
        std::vector<TranscriptSegment> tail;
        for (size_t i = agree; i < segs.size(); ++i)
            tail.push_back({segs[i].start - confirmed_end_sec, segs[i].end - confirmed_end_sec, segs[i].text});
        s->whisper_prev_segments = std::move(tail);
    } else {
        s->whisper_prev_segments = segs;
    }

    s->previous_pending = pending;
    return confirmed;
}

std::string whisper_flush(StreamTranscribe* s, StreamStats& stats) {
    std::vector<TranscriptSegment> segs = whisper_transcribe(s, s->samples, stats);
    return segments_text(segs, 0, segs.size());
}

int write_result(char* buffer, size_t size, const std::string& confirmed,
                 const std::string& pending, const StreamStats& stats) {
    std::ostringstream os;
    os << "{\"success\":true,\"confirmed\":\"" << escape_json_string(confirmed)
       << "\",\"pending\":\"" << escape_json_string(pending)
       << "\",\"decode_tps\":" << stats.decode_tps
       << ",\"total_time_ms\":" << stats.total_time_ms
       << ",\"time_to_first_token_ms\":" << stats.time_to_first_token_ms
       << ",\"decode_tokens\":" << stats.decode_tokens << "}";
    const std::string json = os.str();
    if (!buffer || size == 0) return 0;
    if (json.size() >= size) {
        handle_error_response("Stream response buffer too small", buffer, size);
        return -1;
    }
    std::memcpy(buffer, json.c_str(), json.size() + 1);
    return static_cast<int>(json.size());
}

} // namespace

extern "C" {

cactus_stream_transcribe_t cactus_stream_transcribe_start(cactus_model_t model, const char* options_json) {
    if (!model) {
        last_error_message = "stream_transcribe_start: model is null";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }
    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        const auto type = handle->model->get_config().model_type;
        const bool is_whisper = type == cactus::engine::Config::ModelType::WHISPER;
        const bool is_parakeet = type == cactus::engine::Config::ModelType::PARAKEET_TDT;
        if (!is_whisper && !is_parakeet) {
            last_error_message = "stream_transcribe_start: only Whisper and Parakeet models support streaming";
            CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
            return nullptr;
        }
        auto s = std::make_unique<StreamTranscribe>();
        s->model = handle;
        s->is_parakeet = is_parakeet;
        if (options_json && options_json[0] != '\0') s->options_json = options_json;
        CACTUS_LOG_INFO("stream_transcribe_start", "streaming session opened");
        return static_cast<cactus_stream_transcribe_t>(s.release());
    } catch (const std::exception& e) {
        last_error_message = std::string("stream_transcribe_start: ") + e.what();
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }
}

int cactus_stream_transcribe_process(cactus_stream_transcribe_t stream,
                                     const uint8_t* pcm_buffer, size_t pcm_buffer_size,
                                     char* response_buffer, size_t buffer_size) {
    if (!stream) {
        last_error_message = "stream_transcribe_process: stream is null";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
    auto* s = static_cast<StreamTranscribe*>(stream);
    try {
        if (pcm_buffer && pcm_buffer_size >= sizeof(int16_t)) {
            std::vector<float> news = cactus::audio::pcm_buffer_to_float_samples(pcm_buffer, pcm_buffer_size);
            s->samples.insert(s->samples.end(), news.begin(), news.end());
        }
        StreamStats stats;
        std::string pending;
        std::string confirmed = s->is_parakeet ? parakeet_process(s, pending, stats)
                                                : whisper_process(s, pending, stats);
        return write_result(response_buffer, buffer_size, confirmed, pending, stats);
    } catch (const std::exception& e) {
        last_error_message = std::string("stream_transcribe_process: ") + e.what();
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
}

int cactus_stream_transcribe_stop(cactus_stream_transcribe_t stream,
                                  char* response_buffer, size_t buffer_size) {
    if (!stream) {
        last_error_message = "stream_transcribe_stop: stream is null";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
    auto* s = static_cast<StreamTranscribe*>(stream);
    int result = 0;
    try {
        StreamStats stats;
        std::string confirmed = s->is_parakeet ? parakeet_flush(s, stats) : whisper_flush(s, stats);
        result = write_result(response_buffer, buffer_size, confirmed, "", stats);
    } catch (const std::exception& e) {
        last_error_message = std::string("stream_transcribe_stop: ") + e.what();
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        result = -1;
    }
    delete s;
    return result;
}

} // extern "C"
