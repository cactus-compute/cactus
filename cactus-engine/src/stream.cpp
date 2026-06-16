#include "../cactus_engine.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace cactus::ffi;

namespace {

constexpr int kSampleRate = cactus::audio::WHISPER_SAMPLE_RATE;
constexpr size_t kVadWindow = kSampleRate / 50;
constexpr float kLeadInSec = 0.3f;
constexpr size_t kScratchSize = 1u << 16;

struct StreamConfig {
    float min_chunk_sec = 1.0f;
    float silence_sec = 0.7f;
    float max_segment_sec = 24.0f;
    float vad_threshold = 0.0025f;
    float commit_holdback = 4.0f;
};

struct StreamTranscribe {
    CactusModelHandle* model = nullptr;
    std::string options_json;
    StreamConfig cfg;

    std::vector<int16_t> pcm;
    size_t transcribed_samples = 0;
    size_t trailing_silence = 0;
    bool has_speech = false;

    std::vector<std::string> committed;
    std::vector<std::string> previous;
};

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) words.push_back(word);
    return words;
}

std::string join_words(const std::vector<std::string>& words, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to && i < words.size(); ++i) {
        if (!out.empty()) out += ' ';
        out += words[i];
    }
    return out;
}

std::string strip_nonspeech(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    int depth = 0;
    for (char c : text) {
        if (c == '[' || c == '(') { ++depth; continue; }
        if (c == ']' || c == ')') { if (depth > 0) --depth; continue; }
        if (depth == 0) out += c;
    }
    return out;
}

float window_rms(const int16_t* samples, size_t count) {
    if (count == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(samples[i]) / 32768.0;
        acc += v * v;
    }
    return static_cast<float>(std::sqrt(acc / static_cast<double>(count)));
}

void update_vad(StreamTranscribe* s, size_t from) {
    for (size_t i = from; i < s->pcm.size(); i += kVadWindow) {
        const size_t count = std::min(kVadWindow, s->pcm.size() - i);
        if (window_rms(&s->pcm[i], count) < s->cfg.vad_threshold) {
            s->trailing_silence += count;
        } else {
            s->trailing_silence = 0;
            s->has_speech = true;
        }
    }
}

std::string transcribe_segment(StreamTranscribe* s) {
    if (s->pcm.empty()) return "";
    std::string scratch(kScratchSize, '\0');
    const int rc = cactus_transcribe(
        static_cast<cactus_model_t>(s->model),
        nullptr, nullptr,
        scratch.data(), scratch.size(),
        s->options_json.empty() ? nullptr : s->options_json.c_str(),
        nullptr, nullptr,
        reinterpret_cast<const uint8_t*>(s->pcm.data()),
        s->pcm.size() * sizeof(int16_t));
    if (rc <= 0) {
        throw std::runtime_error(scratch.c_str()[0] ? scratch.c_str() : "transcribe failed");
    }
    return strip_nonspeech(json_string_field(std::string(scratch.c_str()), "response"));
}

int write_result(char* buffer, size_t size, const std::string& confirmed, const std::string& pending) {
    std::ostringstream os;
    os << "{\"success\":true,\"confirmed\":\"" << escape_json_string(confirmed)
       << "\",\"pending\":\"" << escape_json_string(pending) << "\"}";
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
        const bool supported = type == cactus::engine::Config::ModelType::WHISPER ||
                               type == cactus::engine::Config::ModelType::PARAKEET_TDT;
        if (!supported) {
            last_error_message = "stream_transcribe_start: only Whisper and Parakeet models support streaming";
            CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
            return nullptr;
        }

        auto* s = new StreamTranscribe();
        s->model = handle;
        if (options_json && options_json[0] != '\0') {
            s->options_json = options_json;
            const std::string json = options_json;
            try_parse_json_float(json, "min_chunk_sec", s->cfg.min_chunk_sec);
            try_parse_json_float(json, "silence_sec", s->cfg.silence_sec);
            try_parse_json_float(json, "max_segment_sec", s->cfg.max_segment_sec);
            try_parse_json_float(json, "vad_threshold", s->cfg.vad_threshold);
            try_parse_json_float(json, "commit_holdback", s->cfg.commit_holdback);
            s->cfg.commit_holdback = std::clamp(s->cfg.commit_holdback, 0.0f, 32.0f);
            s->cfg.min_chunk_sec = std::clamp(s->cfg.min_chunk_sec, 0.1f, 30.0f);
            s->cfg.silence_sec = std::clamp(s->cfg.silence_sec, 0.1f, 10.0f);
            s->cfg.max_segment_sec = std::clamp(s->cfg.max_segment_sec, 2.0f, 29.0f);
            s->cfg.vad_threshold = std::clamp(s->cfg.vad_threshold, 0.0f, 1.0f);
        }
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
            const size_t from = s->pcm.size();
            const size_t count = pcm_buffer_size / sizeof(int16_t);
            const auto* in = reinterpret_cast<const int16_t*>(pcm_buffer);
            s->pcm.insert(s->pcm.end(), in, in + count);
            update_vad(s, from);
        }

        if (!s->has_speech) {
            const size_t lead = static_cast<size_t>(kSampleRate * kLeadInSec);
            if (s->pcm.size() > lead) {
                s->pcm.erase(s->pcm.begin(), s->pcm.end() - lead);
                s->trailing_silence = std::min(s->trailing_silence, s->pcm.size());
            }
            s->transcribed_samples = 0;
            return write_result(response_buffer, buffer_size, "", "");
        }

        const float segment_sec = static_cast<float>(s->pcm.size()) / kSampleRate;
        const float new_sec = static_cast<float>(s->pcm.size() - s->transcribed_samples) / kSampleRate;
        const float silence_sec = static_cast<float>(s->trailing_silence) / kSampleRate;
        const bool finalize = silence_sec >= s->cfg.silence_sec || segment_sec >= s->cfg.max_segment_sec;
        const bool refresh = new_sec >= s->cfg.min_chunk_sec;

        if (!finalize && !refresh) {
            return write_result(response_buffer, buffer_size, "",
                                join_words(s->previous, s->committed.size(), s->previous.size()));
        }

        std::vector<std::string> hyp = split_words(transcribe_segment(s));
        s->transcribed_samples = s->pcm.size();

        if (finalize) {
            const std::string confirmed = hyp.size() > s->committed.size()
                ? join_words(hyp, s->committed.size(), hyp.size()) : std::string();
            s->pcm.clear();
            s->transcribed_samples = 0;
            s->trailing_silence = 0;
            s->has_speech = false;
            s->committed.clear();
            s->previous.clear();
            return write_result(response_buffer, buffer_size, confirmed, "");
        }

        if (hyp.size() < s->committed.size()) {
            return write_result(response_buffer, buffer_size, "", "");
        }

        const size_t hold = static_cast<size_t>(s->cfg.commit_holdback);
        const size_t commit_cap = hyp.size() > hold ? hyp.size() - hold : 0;
        size_t agreed = s->committed.size();
        const size_t limit = std::min({s->previous.size(), hyp.size(), commit_cap});
        while (agreed < limit && s->previous[agreed] == hyp[agreed]) ++agreed;
        const std::string confirmed = agreed > s->committed.size()
            ? join_words(hyp, s->committed.size(), agreed) : std::string();
        s->committed.assign(hyp.begin(), hyp.begin() + agreed);
        const std::string pending = join_words(hyp, agreed, hyp.size());
        s->previous = std::move(hyp);
        return write_result(response_buffer, buffer_size, confirmed, pending);
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
        std::string confirmed;
        if (s->has_speech && !s->pcm.empty()) {
            std::vector<std::string> hyp = split_words(transcribe_segment(s));
            if (hyp.size() > s->committed.size()) {
                confirmed = join_words(hyp, s->committed.size(), hyp.size());
            }
        }
        result = write_result(response_buffer, buffer_size, confirmed, "");
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
