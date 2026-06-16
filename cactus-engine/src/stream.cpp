#include "../cactus_engine.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace cactus::ffi;

namespace {

constexpr size_t kScratchSize = 1u << 16;
constexpr size_t kLeftContextSamples = 8 * 16000;
constexpr size_t kRightContextSamples = 16000;
constexpr size_t kChunkSamples = 16000;
constexpr size_t kColdStartSamples = 6 * 16000;
constexpr size_t kSilenceResetSamples = 3 * 16000;
constexpr size_t kWhisperResetWindow = 8 * 16000;
constexpr size_t kWhisperMaxWindow = 28 * 16000;
constexpr size_t kWhisperStablePolls = 4;

struct StreamStats {
    size_t decode_tokens = 0;
    double total_time_ms = 0.0;
    double raw_decode_ms = 0.0;
    double decode_tps = 0.0;
    double raw_decoder_tps = 0.0;
    double time_to_first_token_ms = 0.0;

    void finalize() {
        if (total_time_ms > 0.0) decode_tps = decode_tokens * 1000.0 / total_time_ms;
        if (raw_decode_ms > 0.0) raw_decoder_tps = decode_tokens * 1000.0 / raw_decode_ms;
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

    std::vector<std::string> whisper_prev_words;
    size_t whisper_committed = 0;
    size_t whisper_last_len = 0;
    size_t whisper_window_start = 0;
    size_t whisper_stable = 0;
};

double json_num(const std::string& json, const std::string& key) {
    float v = 0.0f;
    return try_parse_json_float(json, key, v) ? static_cast<double>(v) : 0.0;
}

std::vector<std::string> split_ws(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) words.push_back(w);
    return words;
}

std::string join_ws(const std::vector<std::string>& words, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to && i < words.size(); ++i) {
        if (!out.empty()) out += ' ';
        out += words[i];
    }
    return out;
}

std::vector<int16_t> to_pcm16(const std::vector<float>& samples) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float x = std::max(-32768.0f, std::min(32767.0f, samples[i] * 32768.0f));
        pcm[i] = static_cast<int16_t>(x);
    }
    return pcm;
}

std::string transcribe_samples(StreamTranscribe* s, const std::vector<float>& samples, StreamStats& stats) {
    std::vector<int16_t> pcm = to_pcm16(samples);
    std::string scratch(kScratchSize, '\0');
    const int rc = cactus_transcribe(
        static_cast<cactus_model_t>(s->model), nullptr, nullptr,
        scratch.data(), scratch.size(),
        s->options_json.empty() ? nullptr : s->options_json.c_str(),
        nullptr, nullptr,
        reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size() * sizeof(int16_t));
    if (rc <= 0) return "";
    const std::string json(scratch.c_str());
    stats.decode_tps = json_num(json, "decode_tps");
    stats.raw_decoder_tps = json_num(json, "raw_decoder_tps");
    stats.total_time_ms = json_num(json, "total_time_ms");
    stats.time_to_first_token_ms = json_num(json, "time_to_first_token_ms");
    stats.decode_tokens = static_cast<size_t>(json_num(json, "decode_tokens"));
    return json_string_field(json, "response");
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
    stats.raw_decode_ms += s->pstate.raw_decode_ms;
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
        s->samples_decoded_up_to = window_start + static_cast<size_t>(s->pstate.confirmed_sec * 16000.0f);
    }
    return delta;
}

size_t parakeet_spf(StreamTranscribe* s) {
    const uint32_t subsampling = std::max<uint32_t>(1, s->model->model->get_config().subsampling_factor);
    return cactus::audio::get_parakeet_spectrogram_config().hop_length * subsampling;
}

std::string parakeet_process(StreamTranscribe* s, std::string* pending_text, StreamStats& stats) {
    std::lock_guard<std::mutex> lock(s->model->model_mutex);
    const size_t spf = parakeet_spf(s);

    std::string confirmed;
    for (;;) {
        const size_t total = s->samples.size();
        const size_t decodable = total > kRightContextSamples ? total - kRightContextSamples : 0;
        const bool cold = s->samples_decoded_up_to == 0 || s->cold_restart;
        const size_t min_chunk = cold ? kColdStartSamples : kChunkSamples;
        if (decodable <= s->samples_decoded_up_to ||
            decodable - s->samples_decoded_up_to < min_chunk) {
            break;
        }
        const size_t window_start = cold ? s->samples_decoded_up_to
            : (s->samples_decoded_up_to > kLeftContextSamples ? s->samples_decoded_up_to - kLeftContextSamples : 0);
        const size_t window_end = std::min(total, decodable + kRightContextSamples);
        const size_t decode_start_frame = (s->samples_decoded_up_to - window_start) / spf;
        const size_t decode_end_frame = decode_start_frame + (decodable - s->samples_decoded_up_to) / spf;
        if (decode_end_frame <= decode_start_frame) break;
        if (cold) s->pstate = {};

        std::string pend;
        const size_t prev_cursor = s->samples_decoded_up_to;
        confirmed += parakeet_decode_window(s, window_start, window_end,
                                            decode_start_frame, decode_end_frame, false, &pend, stats);
        if (pending_text) *pending_text = pend;
        if (s->pstate.decoded_tokens > 0) { s->cold_restart = false; s->silence_run = 0; }
        if (s->samples_decoded_up_to > prev_cursor) continue;
        if (s->pstate.decoded_tokens == 0) {
            s->silence_run += decodable - s->samples_decoded_up_to;
            s->samples_decoded_up_to = decodable;
            if (s->silence_run >= kSilenceResetSamples) s->cold_restart = true;
            continue;
        }
        if (decodable - s->samples_decoded_up_to < kColdStartSamples) break;
        confirmed += parakeet_decode_window(s, window_start, window_end,
                                            decode_start_frame, decode_end_frame, true, &pend, stats);
        if (pending_text) *pending_text = pend;
        s->samples_decoded_up_to = decodable;
    }
    stats.finalize();
    if (confirmed.empty() && pending_text && pending_text->empty()) *pending_text = s->previous_pending;
    else if (pending_text) s->previous_pending = *pending_text;
    return confirmed;
}

std::string parakeet_flush(StreamTranscribe* s, StreamStats& stats) {
    std::lock_guard<std::mutex> lock(s->model->model_mutex);
    const size_t total = s->samples.size();
    if (total <= s->samples_decoded_up_to) return "";
    const size_t spf = parakeet_spf(s);
    const size_t window_start = s->samples_decoded_up_to > kLeftContextSamples
        ? s->samples_decoded_up_to - kLeftContextSamples : 0;
    const size_t decode_start_frame = (s->samples_decoded_up_to - window_start) / spf;
    std::string confirmed = parakeet_decode_window(s, window_start, total, decode_start_frame, 0, true, nullptr, stats);
    stats.finalize();
    return confirmed;
}

std::string whisper_commit(StreamTranscribe* s, const std::vector<std::string>& words, size_t up_to) {
    std::string confirmed = join_ws(words, s->whisper_committed, up_to);
    if (!confirmed.empty()) {
        if (!s->emitted_text.empty()) { s->emitted_text += ' '; confirmed = " " + confirmed; }
        s->emitted_text += join_ws(words, s->whisper_committed, up_to);
        s->whisper_committed = up_to;
    }
    return confirmed;
}

std::vector<float> whisper_window(StreamTranscribe* s) {
    return std::vector<float>(s->samples.begin() + s->whisper_window_start, s->samples.end());
}

std::string whisper_process(StreamTranscribe* s, std::string* pending_text, StreamStats& stats) {
    if (s->samples.size() < s->whisper_last_len + kChunkSamples) {
        if (pending_text) *pending_text = s->previous_pending;
        return "";
    }
    s->whisper_last_len = s->samples.size();
    std::vector<std::string> words = split_ws(transcribe_samples(s, whisper_window(s), stats));

    size_t agree = 0;
    while (agree < words.size() && agree < s->whisper_prev_words.size() &&
           words[agree] == s->whisper_prev_words[agree]) ++agree;
    s->whisper_prev_words = words;
    if (agree < s->whisper_committed) agree = s->whisper_committed;

    std::string confirmed = whisper_commit(s, words, agree);
    std::string pending = join_ws(words, agree, words.size());

    const size_t window_len = s->samples.size() - s->whisper_window_start;
    s->whisper_stable = s->whisper_committed >= words.size() ? s->whisper_stable + 1 : 0;
    if ((window_len >= kWhisperResetWindow && s->whisper_stable >= kWhisperStablePolls) ||
        window_len >= kWhisperMaxWindow) {
        if (window_len >= kWhisperMaxWindow) confirmed += whisper_commit(s, words, words.size());
        s->whisper_window_start = s->samples.size();
        s->whisper_committed = 0;
        s->whisper_stable = 0;
        s->whisper_prev_words.clear();
        pending.clear();
    }

    if (pending_text) { *pending_text = pending; s->previous_pending = pending; }
    return confirmed;
}

std::string whisper_flush(StreamTranscribe* s, StreamStats& stats) {
    std::vector<std::string> words = split_ws(transcribe_samples(s, whisper_window(s), stats));
    if (words.size() < s->whisper_committed) return "";
    return whisper_commit(s, words, words.size());
}

int write_result(char* buffer, size_t size, const std::string& confirmed,
                 const std::string& pending, const StreamStats& stats) {
    std::ostringstream os;
    os << "{\"success\":true,\"confirmed\":\"" << escape_json_string(confirmed)
       << "\",\"pending\":\"" << escape_json_string(pending)
       << "\",\"decode_tps\":" << stats.decode_tps
       << ",\"raw_decoder_tps\":" << stats.raw_decoder_tps
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
        auto* s = new StreamTranscribe();
        s->model = handle;
        s->is_parakeet = is_parakeet;
        if (options_json && options_json[0] != '\0') s->options_json = options_json;
        CACTUS_LOG_INFO("stream_transcribe_start", "streaming session opened");
        return static_cast<cactus_stream_transcribe_t>(s);
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
        std::string confirmed = s->is_parakeet ? parakeet_process(s, &pending, stats)
                                                : whisper_process(s, &pending, stats);
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
