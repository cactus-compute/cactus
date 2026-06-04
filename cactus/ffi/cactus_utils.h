#ifndef CACTUS_UTILS_H
#define CACTUS_UTILS_H

#include "../engine/engine.h"
#include "../models/model.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>

#ifdef __APPLE__
#include <uuid/uuid.h>
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

inline size_t get_memory_footprint_bytes() {
#ifdef __APPLE__
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count) == KERN_SUCCESS)
        return vm_info.phys_footprint;

#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return pmc.PrivateUsage;
        
#elif defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size, resident;
        statm >> size >> resident;
        return resident * sysconf(_SC_PAGESIZE);
    }
#endif
    return 0;
}

inline double get_ram_usage_mb() {
    return get_memory_footprint_bytes() / (1024.0 * 1024.0);
}

struct CactusModelHandle {
    std::unique_ptr<cactus::engine::Model> model;
    std::unique_ptr<cactus::engine::Model> vad_model;
    std::atomic<bool> should_stop;
    std::vector<uint32_t> processed_tokens;
    struct ProcessedImage {
        std::string path;
        long long last_modified_timestamp = 0;

        bool operator==(const ProcessedImage& other) const {
            return path == other.path && last_modified_timestamp == other.last_modified_timestamp;
        }
    };

    std::vector<std::vector<ProcessedImage>> processed_images;
    std::vector<size_t> user_audio_counts;
    std::mutex model_mutex;
    std::string model_name;
    std::unique_ptr<cactus::engine::index::Index> corpus_index;
    std::string corpus_dir;
    size_t corpus_embedding_dim = 0;
    std::vector<std::vector<float>> tool_embeddings;
    std::vector<std::string> tool_texts;

    CactusModelHandle() : should_stop(false) {}
};

extern std::string last_error_message;

bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                           const std::vector<std::vector<uint32_t>>& stop_sequences);

std::string retrieve_rag_context(CactusModelHandle* handle, const std::string& query);

namespace cactus {
namespace audio {

static constexpr size_t WHISPER_TARGET_FRAMES = 3000;
static constexpr int WHISPER_SAMPLE_RATE = 16000;

inline cactus::engine::AudioProcessor::SpectrogramConfig get_whisper_spectrogram_config() {
    cactus::engine::AudioProcessor::SpectrogramConfig cfg{};
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

inline cactus::engine::AudioProcessor::SpectrogramConfig get_parakeet_spectrogram_config() {
    cactus::engine::AudioProcessor::SpectrogramConfig cfg{};
    cfg.n_fft        = 512;
    cfg.frame_length = 400;
    cfg.hop_length   = 160;
    cfg.power        = 2.0f;
    cfg.center       = true;
    cfg.pad_mode     = "constant";
    cfg.onesided     = true;
    cfg.dither       = 0.0f;
    cfg.mel_floor    = 5.960464477539063e-08f; // 2^-24 guard value used by HF Parakeet.
    cfg.log_mel      = "log";
    cfg.reference    = 1.0f;
    cfg.min_value    = 1e-10f;
    cfg.remove_dc_offset = false;
    cfg.hann_periodic = false;
    return cfg;
}

inline cactus::engine::AudioProcessor::SpectrogramConfig get_htk_spectrogram_config() {
    cactus::engine::AudioProcessor::SpectrogramConfig cfg{};
    cfg.n_fft        = 321;
    cfg.frame_length = 320;
    cfg.fft_override = 1024;
    cfg.hop_length   = 160;
    cfg.power        = 1.0f;
    cfg.center       = false;
    cfg.pad_mode     = "constant";
    cfg.onesided     = true;
    cfg.dither       = 0.0f;
    cfg.mel_floor    = 0.001f;
    cfg.log_mel      = "log";
    cfg.reference    = 1.0f;
    cfg.min_value    = 0.001f;
    cfg.remove_dc_offset = false;
    cfg.hann_periodic = true;
    return cfg;
}

inline cactus::engine::AudioProcessor::SpectrogramConfig get_gemma4_audio_spectrogram_config(
    const cactus::engine::Config& model_config) {
    auto cfg = get_htk_spectrogram_config();
    cfg.fft_override = model_config.audio_fft_length;
    cfg.mel_floor_additive = true;
    return cfg;
}

inline cactus::engine::AudioProcessor::SpectrogramConfig get_wespeaker_spectrogram_config() {
    cactus::engine::AudioProcessor::SpectrogramConfig cfg{};
    cfg.n_fft            = 512;
    cfg.frame_length     = 400;
    cfg.hop_length       = 160;
    cfg.power            = 2.0f;
    cfg.center           = false;
    cfg.pad_mode         = "constant";
    cfg.onesided         = true;
    cfg.dither           = 0.0f;
    cfg.mel_floor        = 1.1754944e-38f;
    cfg.log_mel          = "log";
    cfg.reference        = 1.0f;
    cfg.min_value        = 1.1754944e-38f;
    cfg.remove_dc_offset = true;
    cfg.preemphasis      = 0.97f;
    cfg.hann_periodic    = false;
    cfg.window_a0        = 0.54f;
    return cfg;
}

// Whisper v1/v2: 80 mel bins, HTK. Whisper v3: 128 mel bins, Slaney, 512-FFT, no DC removal.
inline void init_whisper_mel_filters(cactus::engine::AudioProcessor& ap,
                                     cactus::engine::AudioProcessor::SpectrogramConfig& cfg,
                                     size_t mel_bins) {
    const size_t num_mel_filters = std::max<size_t>(1, mel_bins);
    const bool is_v3 = mel_bins > 80;
    if (is_v3) {
        cfg.fft_override = 512;
        cfg.remove_dc_offset = false;
    }
    const size_t fft_len = cfg.fft_override > 0 ? cfg.fft_override : cfg.n_fft;
    const size_t num_frequency_bins = fft_len / 2 + 1;
    if (is_v3) {
        ap.init_mel_filters(num_frequency_bins, num_mel_filters, 0.0f, 8000.0f,
                            WHISPER_SAMPLE_RATE, "slaney", "slaney");
    } else {
        ap.init_mel_filters(num_frequency_bins, num_mel_filters, 0.0f, 8000.0f,
                            WHISPER_SAMPLE_RATE);
    }
}

// use_mel_floor_padding=true pads short audio with the normalized mel floor (required for v3).
inline std::vector<float> normalize_whisper_mel(std::vector<float>& mel, size_t n_mels,
                                                bool use_mel_floor_padding = false) {
    if (mel.empty() || n_mels == 0) return mel;
    size_t n_frames = mel.size() / n_mels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : mel) if (v > max_val) max_val = v;

    float min_allowed = max_val - 8.0f;
    for (float& v : mel) {
        if (v < min_allowed) v = min_allowed;
        v = (v + 4.0f) * 0.25f;
    }

    if (n_frames != WHISPER_TARGET_FRAMES) {
        float pad_val = use_mel_floor_padding ? (min_allowed + 4.0f) * 0.25f : 0.0f;
        std::vector<float> fixed(n_mels * WHISPER_TARGET_FRAMES, pad_val);
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

inline std::vector<float> transpose_mel_to_frame_major(const std::vector<float>& mel,
                                                        size_t num_mels, size_t num_frames) {
    std::vector<float> transposed(num_frames * num_mels);
    for (size_t m = 0; m < num_mels; m++) {
        for (size_t t = 0; t < num_frames; t++) {
            transposed[t * num_mels + m] = mel[m * num_frames + t];
        }
    }
    return transposed;
}

inline void apply_preemphasis(std::vector<float>& waveform, float coefficient = 0.97f) {
    if (waveform.size() < 2 || coefficient == 0.0f) {
        return;
    }
    for (size_t i = waveform.size() - 1; i > 0; --i) {
        waveform[i] -= coefficient * waveform[i - 1];
    }
}

inline void normalize_parakeet_log_mel(std::vector<float>& mel, size_t num_mels, float epsilon = 1e-5f) {
    if (mel.empty() || num_mels == 0 || (mel.size() % num_mels) != 0) {
        return;
    }
    const size_t num_frames = mel.size() / num_mels;
    if (num_frames == 0) {
        return;
    }

    for (size_t m = 0; m < num_mels; ++m) {
        const size_t base = m * num_frames;
        float mean = 0.0f;
        for (size_t t = 0; t < num_frames; ++t) {
            mean += mel[base + t];
        }
        mean /= static_cast<float>(num_frames);

        float variance = 0.0f;
        for (size_t t = 0; t < num_frames; ++t) {
            const float d = mel[base + t] - mean;
            variance += d * d;
        }
        const float denom = static_cast<float>(std::max<size_t>(1, num_frames - 1));
        const float inv_std = 1.0f / std::sqrt((variance / denom) + epsilon);
        for (size_t t = 0; t < num_frames; ++t) {
            mel[base + t] = (mel[base + t] - mean) * inv_std;
        }
    }
}

inline void trim_mel_frames(std::vector<float>& mel, size_t num_mels, size_t valid_frames) {
    if (mel.empty() || num_mels == 0 || (mel.size() % num_mels) != 0) {
        return;
    }
    size_t total_frames = mel.size() / num_mels;
    if (valid_frames == 0 || valid_frames >= total_frames) {
        return;
    }
    std::vector<float> trimmed(num_mels * valid_frames);
    for (size_t m = 0; m < num_mels; ++m) {
        const float* src = &mel[m * total_frames];
        float* dst = &trimmed[m * valid_frames];
        std::copy(src, src + valid_frames, dst);
    }
    mel.swap(trimmed);
}

struct AudioPreprocessResult {
    std::vector<float> features;
    size_t num_frames = 0;
    size_t num_soft_tokens = 0;
};

inline AudioPreprocessResult preprocess_audio_for_gemma4(
    std::vector<float> audio_samples,
    const cactus::engine::Config& model_config
) {
    AudioPreprocessResult result;
    if (audio_samples.empty()) return result;

    size_t pad_amt = 320 - (audio_samples.size() % 320);
    if (pad_amt < 320)
        audio_samples.resize(audio_samples.size() + pad_amt, 0.0f);

    size_t mel_bins = model_config.audio_input_feat_size;
    auto cfg = get_gemma4_audio_spectrogram_config(model_config);

    size_t semicausal_pad = cfg.frame_length / 2;
    audio_samples.insert(audio_samples.begin(), semicausal_pad, 0.0f);

    cactus::engine::AudioProcessor ap;
    size_t fft_for_mel = cfg.fft_override > 0 ? cfg.fft_override : cfg.n_fft;
    ap.init_mel_filters(fft_for_mel / 2 + 1, mel_bins, 0.0f, 8000.0f, 16000,
                        nullptr, "htk");
    std::vector<float> mel = ap.compute_spectrogram(audio_samples, cfg);

    result.num_frames = mel.size() / mel_bins;
    result.features = transpose_mel_to_frame_major(mel, mel_bins, result.num_frames);

    size_t after_stage1 = (result.num_frames + 1) / 2;
    result.num_soft_tokens = (after_stage1 + 1) / 2;

    return result;
}

inline std::vector<float> pcm_buffer_to_float_samples(
    const uint8_t* pcm_buffer, size_t pcm_buffer_size
) {
    const int16_t* pcm_samples = reinterpret_cast<const int16_t*>(pcm_buffer);
    size_t num_samples = pcm_buffer_size / 2;
    std::vector<float> waveform_fp32(num_samples);
    constexpr float inv_32768 = 1.0f / 32768.0f;
    for (size_t i = 0; i < num_samples; i++)
        waveform_fp32[i] = static_cast<float>(pcm_samples[i]) * inv_32768;
    return waveform_fp32;
}

} // namespace audio
} // namespace cactus

namespace cactus {
namespace ffi {

inline bool env_flag_enabled(const char* key) {
    const char* value = std::getenv(key);
    return value && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

inline std::string generateUUID() {
#ifdef __APPLE__
    uuid_t uuid;
    uuid_generate_random(uuid);
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    return std::string(uuid_str);
#else
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4";
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);
    return ss.str();
#endif
}

struct ToolFunction {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> parameters;
};

struct InferenceOptions {
    float temperature = 0.0f;
    float top_p = 0.0f;
    float min_p = 0.15f;
    float repetition_penalty = 1.1f;
    float confidence_threshold = -1.0f;
    size_t top_k = 0;
    size_t max_tokens = 100;
    size_t tool_rag_top_k = 2;
    size_t cloud_timeout_ms = 15000;
    std::vector<std::string> stop_sequences;
    bool force_tools = false;
    bool include_stop_sequences = false;
    bool use_vad = true;
    bool telemetry_enabled = true;
    bool auto_handoff = true;
    bool handoff_with_images = true;
    bool enable_thinking_if_supported = false;
};

} // namespace ffi
} // namespace cactus

std::vector<cactus::ffi::ToolFunction> select_relevant_tools(
    CactusModelHandle* handle,
    const std::string& query,
    const std::vector<cactus::ffi::ToolFunction>& all_tools,
    size_t top_k);

#include "gemma_tools.h"

namespace cactus {
namespace ffi {

inline std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c == '\\') o << "\\\\";
        else o << c;
    }
    return o.str();
}


inline std::string trim_string(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

inline size_t find_matching_delimiter(const std::string& s, size_t pos, char open, char close) {
    int depth = 1;
    pos++;
    while (pos < s.length() && depth > 0) {
        if (s[pos] == open) depth++;
        else if (s[pos] == close) depth--;
        else if (s[pos] == '"') {
            pos++;
            while (pos < s.length() && s[pos] != '"') {
                if (s[pos] == '\\') pos++;
                pos++;
            }
        }
        pos++;
    }
    return pos;
}

inline std::string env_or_default(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    if (v && v[0] != '\0') return std::string(v);
    return std::string(fallback);
}

inline std::string json_string_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};

    size_t i = pos + pattern.size();
    while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) i++;
    if (i >= json.size() || json[i] != '"') return {};
    ++i;

    std::string out;
    out.reserve(128);
    while (i < json.size()) {
        char c = json[i++];
        if (c == '"') return out;
        if (c == '\\' && i < json.size()) {
            char e = json[i++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                default:   out.push_back(e);    break;
            }
            continue;
        }
        out.push_back(c);
    }
    return {};
}

inline std::string json_array_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "[]";
    size_t start = pos + pattern.size();
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) ++start;
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

inline std::vector<std::string> split_json_array(const std::string& array_json) {
    std::vector<std::string> out;
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') return out;

    size_t i = 1;
    while (i + 1 < array_json.size()) {
        while (i + 1 < array_json.size() &&
               (std::isspace(static_cast<unsigned char>(array_json[i])) || array_json[i] == ',')) i++;
        if (i + 1 >= array_json.size() || array_json[i] != '{') break;

        size_t start = i;
        int depth = 0;
        bool in_str = false;
        bool esc = false;
        for (; i < array_json.size(); ++i) {
            char c = array_json[i];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') { in_str = true; continue; }
            if (c == '{') depth++;
            if (c == '}') {
                depth--;
                if (depth == 0) {
                    out.push_back(array_json.substr(start, i - start + 1));
                    i++;
                    break;
                }
            }
        }
    }
    return out;
}

inline std::string serialize_tools_json(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "{\"type\":\"function\",\"function\":{";
        oss << "\"name\":\"" << escape_json_string(tools[i].name) << "\",";
        oss << "\"description\":\"" << escape_json_string(tools[i].description) << "\"";
        auto it = tools[i].parameters.find("schema");
        if (it != tools[i].parameters.end()) {
            oss << ",\"parameters\":" << it->second;
        }
        oss << "}}";
    }
    oss << "]";
    return oss.str();
}

namespace json_sorted {

inline void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
}

inline std::string parse_string(const std::string& s, size_t& p) {
    std::string r = "\"";
    p++;
    while (p < s.size()) {
        if (s[p] == '\\') {
            r += s[p++];
            if (p < s.size()) r += s[p++];
        } else if (s[p] == '"') {
            r += '"';
            p++;
            return r;
        } else {
            r += s[p++];
        }
    }
    return r;
}

inline std::string parse_value(const std::string& s, size_t& p);

inline std::string parse_object(const std::string& s, size_t& p) {
    p++;
    std::map<std::string, std::string> entries;
    skip_ws(s, p);
    while (p < s.size() && s[p] != '}') {
        if (s[p] == ',') { p++; skip_ws(s, p); continue; }
        std::string key = parse_string(s, p);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ':') p++;
        skip_ws(s, p);
        std::string val = parse_value(s, p);
        entries[key] = val;
        skip_ws(s, p);
    }
    if (p < s.size()) p++;
    std::string r = "{";
    bool first = true;
    for (const auto& kv : entries) {
        if (!first) r += ", ";
        r += kv.first + ": " + kv.second;
        first = false;
    }
    r += "}";
    return r;
}

inline std::string parse_array(const std::string& s, size_t& p) {
    p++;
    std::vector<std::string> items;
    skip_ws(s, p);
    while (p < s.size() && s[p] != ']') {
        if (s[p] == ',') { p++; skip_ws(s, p); continue; }
        items.push_back(parse_value(s, p));
        skip_ws(s, p);
    }
    if (p < s.size()) p++;
    std::string r = "[";
    for (size_t i = 0; i < items.size(); i++) {
        if (i > 0) r += ", ";
        r += items[i];
    }
    r += "]";
    return r;
}

inline std::string parse_value(const std::string& s, size_t& p) {
    skip_ws(s, p);
    if (p >= s.size()) return "";
    if (s[p] == '"') return parse_string(s, p);
    if (s[p] == '{') return parse_object(s, p);
    if (s[p] == '[') return parse_array(s, p);
    size_t start = p;
    while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']' && !std::isspace(static_cast<unsigned char>(s[p]))) p++;
    return s.substr(start, p - start);
}

inline std::string reformat(const std::string& json) {
    size_t p = 0;
    return parse_value(json, p);
}

} // namespace json_sorted

inline std::string serialize_tools_for_template(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::string result;
    for (const auto& tool : tools) {
        std::map<std::string, std::string> func_fields;
        func_fields["\"description\""] = "\"" + escape_json_string(tool.description) + "\"";
        func_fields["\"name\""] = "\"" + escape_json_string(tool.name) + "\"";
        auto it = tool.parameters.find("schema");
        if (it != tool.parameters.end()) {
            func_fields["\"parameters\""] = json_sorted::reformat(it->second);
        }
        std::string func_json = "{";
        bool first = true;
        for (const auto& kv : func_fields) {
            if (!first) func_json += ", ";
            func_json += kv.first + ": " + kv.second;
            first = false;
        }
        func_json += "}";
        result += "\n{\"function\": " + func_json + ", \"type\": \"function\"}";
    }
    return result;
}

inline void handle_error_response(const std::string& error_message, char* response_buffer, size_t buffer_size) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":false,";
    json << "\"error\":\"" << escape_json_string(error_message) << "\",";
    json << "\"cloud_handoff\":false,";
    json << "\"response\":null,";
    json << "\"function_calls\":[],";
    json << "\"confidence\":0.0,";
    json << "\"time_to_first_token_ms\":0.0,";
    json << "\"total_time_ms\":0.0,";
    json << "\"prefill_tps\":0.0,";
    json << "\"decode_tps\":0.0,";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb() << ",";
    json << "\"prefill_tokens\":0,";
    json << "\"decode_tokens\":0,";
    json << "\"total_tokens\":0";
    json << "}";
    std::string error_json = json.str();
    if (response_buffer && error_json.length() < buffer_size) {
        std::strcpy(response_buffer, error_json.c_str());
    }
}

inline std::vector<cactus::engine::ChatMessage> parse_messages_json(const std::string& json,
                                                                   std::vector<std::string>& out_image_paths,
                                                                   std::vector<std::string>* out_audio_paths = nullptr) {
    std::vector<cactus::engine::ChatMessage> messages;
    out_image_paths.clear();
    if (out_audio_paths) out_audio_paths->clear();
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid JSON: expected array");
    }
    
    pos = json.find('{', pos);
    while (pos != std::string::npos) {
        cactus::engine::ChatMessage msg;
        
        size_t obj_start = pos;
        int brace_count = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < json.length() && brace_count > 0) {
            if (json[obj_end] == '{') brace_count++;
            else if (json[obj_end] == '}') brace_count--;
            obj_end++;
        }

        size_t role_pos = json.find("\"role\"", pos);
        if (role_pos == std::string::npos || role_pos >= obj_end) break;
        
        size_t role_start = json.find('"', role_pos + 6) + 1;
        size_t role_end = json.find('"', role_start);
        msg.role = json.substr(role_start, role_end - role_start);
        
        size_t content_pos = json.find("\"content\"", role_end);
        if (content_pos != std::string::npos && content_pos < obj_end) {
            size_t content_start = json.find('"', content_pos + 9) + 1;
            size_t content_end = content_start;
            
            while (content_end < json.length()) {
                content_end = json.find('"', content_end);
                if (content_end == std::string::npos) break;
                if (json[content_end - 1] != '\\') break;
                content_end++;
            }
            
            msg.content = json.substr(content_start, content_end - content_start);
            
            size_t escape_pos = 0;
            while ((escape_pos = msg.content.find("\\n", escape_pos)) != std::string::npos) {
                msg.content.replace(escape_pos, 2, "\n");
                escape_pos += 1;
            }
            escape_pos = 0;
            while ((escape_pos = msg.content.find("\\\"", escape_pos)) != std::string::npos) {
                msg.content.replace(escape_pos, 2, "\"");
                escape_pos += 1;
            }
        }
        
        auto parse_path_array = [&](const char* key, std::vector<std::string>& dest,
                                    std::vector<std::string>* out_paths) {
            size_t key_pos = json.find(key, pos);
            if (key_pos == std::string::npos || key_pos >= obj_end) return;
            size_t array_start = json.find('[', key_pos);
            if (array_start == std::string::npos || array_start >= obj_end) return;
            size_t array_end = json.find(']', array_start);
            if (array_end == std::string::npos || array_end >= obj_end) return;
            size_t cur = array_start;
            while (true) {
                cur = json.find('"', cur + 1);
                if (cur == std::string::npos || cur >= array_end) break;
                size_t str_start = cur + 1;
                size_t str_end = json.find('"', str_start);
                if (str_end == std::string::npos || str_end > array_end) break;
                std::string path = std::filesystem::absolute(
                    std::filesystem::path(json.substr(str_start, str_end - str_start))).string();
                dest.push_back(path);
                if (out_paths) out_paths->push_back(path);
                cur = str_end;
            }
        };

        parse_path_array("\"images\"", msg.images, &out_image_paths);
        parse_path_array("\"audio\"", msg.audio, out_audio_paths);

        if (msg.role == "tool") {
            size_t name_pos = json.find("\"name\"", obj_start);
            if (name_pos != std::string::npos && name_pos < obj_end) {
                size_t name_quote = json.find('"', name_pos + 6);
                if (name_quote != std::string::npos && name_quote < obj_end) {
                    size_t name_start = name_quote + 1;
                    size_t name_end = json.find('"', name_start);
                    if (name_end != std::string::npos && name_end < obj_end) {
                        msg.name = json.substr(name_start, name_end - name_start);
                    }
                }
            }
        }

        size_t tool_calls_pos = json.find("\"tool_calls\"", obj_start);
        if (tool_calls_pos != std::string::npos && tool_calls_pos < obj_end) {
            size_t tool_calls_arr_start = json.find('[', tool_calls_pos);
            if (tool_calls_arr_start != std::string::npos && tool_calls_arr_start < obj_end) {
                size_t tool_calls_arr_end = find_matching_delimiter(json, tool_calls_arr_start, '[', ']');

                size_t search_pos = tool_calls_arr_start;
                while (true) {
                    size_t func_pos = json.find("\"function\"", search_pos);
                    if (func_pos == std::string::npos || func_pos >= tool_calls_arr_end) break;

                    size_t func_obj_start = json.find('{', func_pos + 10);
                    if (func_obj_start == std::string::npos || func_obj_start >= tool_calls_arr_end) break;

                    size_t func_obj_end = find_matching_delimiter(json, func_obj_start, '{', '}');

                    cactus::engine::ToolCallInfo tool_call;

                    size_t fn_name_pos = json.find("\"name\"", func_obj_start);
                    if (fn_name_pos != std::string::npos && fn_name_pos < func_obj_end) {
                        size_t fn_name_quote = json.find('"', fn_name_pos + 6);
                        if (fn_name_quote != std::string::npos && fn_name_quote < func_obj_end) {
                            size_t fn_name_start = fn_name_quote + 1;
                            size_t fn_name_end = json.find('"', fn_name_start);
                            if (fn_name_end != std::string::npos && fn_name_end < func_obj_end) {
                                tool_call.name = json.substr(fn_name_start, fn_name_end - fn_name_start);
                            }
                        }
                    }

                    size_t args_pos = json.find("\"arguments\"", func_obj_start);
                    if (args_pos != std::string::npos && args_pos < func_obj_end) {
                        size_t colon_pos = json.find(':', args_pos + 11);
                        if (colon_pos != std::string::npos && colon_pos < func_obj_end) {
                            size_t args_start = colon_pos + 1;
                            while (args_start < json.length() && std::isspace(static_cast<unsigned char>(json[args_start]))) args_start++;

                            if (args_start < func_obj_end && json[args_start] == '{') {
                                size_t args_end = find_matching_delimiter(json, args_start, '{', '}');
                                tool_call.arguments = json.substr(args_start, args_end - args_start);
                            } else if (args_start < func_obj_end && json[args_start] == '"') {
                                size_t str_start = args_start + 1;
                                size_t str_end = str_start;
                                while (str_end < json.length() && json[str_end] != '"') {
                                    if (json[str_end] == '\\') str_end++;
                                    str_end++;
                                }
                                tool_call.arguments = json.substr(str_start, str_end - str_start);
                            }
                        }
                    }

                    if (!tool_call.name.empty()) {
                        msg.tool_calls.push_back(tool_call);
                    }
                    search_pos = func_obj_end;
                }
            }
        }

        messages.push_back(msg);

        pos = json.find('{', obj_end);
    }

    return messages;
}

inline std::vector<ToolFunction> parse_tools_json(const std::string& json) {
    std::vector<ToolFunction> tools;
    
    if (json.empty()) return tools;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) return tools;
    
    pos = json.find("\"function\"", pos);
    while (pos != std::string::npos) {
        ToolFunction tool;
        size_t next_search = pos + 1;
        
        size_t name_pos = json.find("\"name\"", pos);
        if (name_pos != std::string::npos) {
            size_t name_start = json.find('"', name_pos + 6) + 1;
            size_t name_end = json.find('"', name_start);
            tool.name = json.substr(name_start, name_end - name_start);
        }
        
        size_t desc_pos = json.find("\"description\"", pos);
        if (desc_pos != std::string::npos) {
            size_t desc_start = json.find('"', desc_pos + 13) + 1;
            size_t desc_end = json.find('"', desc_start);
            tool.description = json.substr(desc_start, desc_end - desc_start);
        }
        
        size_t params_pos = json.find("\"parameters\"", pos);
        if (params_pos != std::string::npos) {
            size_t params_start = json.find('{', params_pos);
            if (params_start != std::string::npos) {
                int brace_count = 1;
                size_t params_end = params_start + 1;
                while (params_end < json.length() && brace_count > 0) {
                    if (json[params_end] == '{') brace_count++;
                    else if (json[params_end] == '}') brace_count--;
                    params_end++;
                }
                tool.parameters["schema"] = json.substr(params_start, params_end - params_start);
                next_search = params_end;
            }
        }

        if (!tool.name.empty()) {
            tools.push_back(tool);
        }

        pos = json.find("\"function\"", next_search);
    }

    return tools;
}

inline bool try_parse_json_float(const std::string& json, const std::string& key, float& out_value) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;

    size_t start = pos + pattern.size();
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) ++start;

    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;

    try {
        out_value = std::stof(json.substr(start, end - start));
        return true;
    } catch (...) {
        return false;
    }
}

inline std::vector<std::string> parse_json_string_array_field(const std::string& json, const std::string& key) {
    std::vector<std::string> out;
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return out;

    size_t start = pos + pattern.size();
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) ++start;
    if (start >= json.size() || json[start] != '[') return out;

    int depth = 1;
    bool in_string = false;
    bool escaped = false;
    size_t end = start + 1;

    while (end < json.size() && depth > 0) {
        char c = json[end];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
            else if (c == '[') depth++;
            else if (c == ']') depth--;
        }
        ++end;
    }

    if (depth != 0) return out;
    const std::string array_json = json.substr(start, end - start);
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') return out;

    size_t i = 1;
    while (i + 1 < array_json.size()) {
        while (i + 1 < array_json.size() &&
               (std::isspace(static_cast<unsigned char>(array_json[i])) || array_json[i] == ',')) {
            ++i;
        }
        if (i + 1 >= array_json.size() || array_json[i] == ']') break;
        if (array_json[i] != '"') break;

        ++i;
        std::string value;
        bool escaped = false;
        while (i < array_json.size()) {
            char c = array_json[i++];
            if (escaped) {
                switch (c) {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case '/': value.push_back('/'); break;
                    case 'b': value.push_back('\b'); break;
                    case 'f': value.push_back('\f'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default: value.push_back(c); break;
                }
                escaped = false;
                continue;
            }
            if (c == '\\') {
                escaped = true;
                continue;
            }
            if (c == '"') {
                out.push_back(value);
                break;
            }
            value.push_back(c);
        }
    }

    return out;
}

inline void parse_custom_vocabulary_options(const std::string& json,
                                            std::vector<std::string>& custom_vocabulary,
                                            float& vocabulary_boost) {
    custom_vocabulary.clear();
    vocabulary_boost = 5.0f;
    if (json.empty()) return;

    float parsed_boost = vocabulary_boost;
    if (try_parse_json_float(json, "vocabulary_boost", parsed_boost)) {
        vocabulary_boost = std::clamp(parsed_boost, 0.0f, 20.0f);
    }

    custom_vocabulary = parse_json_string_array_field(json, "custom_vocabulary");
}

inline std::unordered_map<uint32_t, float> build_token_bias_map(const std::vector<std::vector<uint32_t>>& tokenized_entries,
                                                                float vocabulary_boost) {
    std::unordered_map<uint32_t, float> vocab_bias;
    const float clamped_boost = std::clamp(vocabulary_boost, 0.0f, 20.0f);
    if (clamped_boost == 0.0f) return vocab_bias;

    for (const auto& token_ids : tokenized_entries) {
        for (uint32_t token_id : token_ids) {
            float& entry = vocab_bias[token_id];
            if (entry < clamped_boost) {
                entry = clamped_boost;
            }
        }
    }

    return vocab_bias;
}

inline std::unordered_map<uint32_t, float> build_custom_vocabulary_bias(cactus::engine::Tokenizer* tokenizer,
                                                                        const std::vector<std::string>& custom_vocabulary,
                                                                        float vocabulary_boost) {
    if (!tokenizer || custom_vocabulary.empty()) return {};
    std::vector<std::vector<uint32_t>> tokenized_entries;
    tokenized_entries.reserve(custom_vocabulary.size());

    for (const auto& word : custom_vocabulary) {
        if (word.empty()) continue;
        tokenized_entries.push_back(tokenizer->encode(word));
    }

    return build_token_bias_map(tokenized_entries, vocabulary_boost);
}

inline void apply_custom_vocabulary_options(cactus::engine::Model* model, const std::string& json) {
    if (!model) return;

    std::vector<std::string> custom_vocabulary;
    float vocabulary_boost = 5.0f;
    parse_custom_vocabulary_options(json, custom_vocabulary, vocabulary_boost);
    model->set_vocab_bias(build_custom_vocabulary_bias(model->get_tokenizer(), custom_vocabulary, vocabulary_boost));
}

inline size_t levenshtein_ci(const std::string& a, const std::string& b) {
    const size_t m = a.size(), n = b.size();
    std::vector<size_t> prev(n + 1), curr(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = j;
    for (size_t i = 1; i <= m; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= n; ++j) {
            const bool match = std::tolower(static_cast<unsigned char>(a[i - 1])) ==
                               std::tolower(static_cast<unsigned char>(b[j - 1]));
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (match ? 0 : 1)});
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

inline std::string collapse_spaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != ' ') out += c;
    }
    return out;
}

inline void apply_vocabulary_spelling_correction(
    std::string& text,
    const std::vector<std::string>& custom_vocabulary)
{
    if (custom_vocabulary.empty() || text.empty()) return;

    struct VocabEntry {
        const std::string* original;
        std::string collapsed;
    };
    std::vector<VocabEntry> vocab_entries;
    vocab_entries.reserve(custom_vocabulary.size());
    for (const auto& v : custom_vocabulary) {
        vocab_entries.push_back({&v, collapse_spaces(v)});
    }

    struct Token { std::string text; bool is_word; };
    std::vector<Token> tokens;
    size_t pos = 0;
    while (pos < text.size()) {
        if (std::isalnum(static_cast<unsigned char>(text[pos])) ||
            text[pos] == '\'' || text[pos] == '-') {
            size_t start = pos;
            while (pos < text.size() && (std::isalnum(static_cast<unsigned char>(text[pos])) ||
                                          text[pos] == '\'' || text[pos] == '-')) {
                ++pos;
            }
            tokens.push_back({text.substr(start, pos - start), true});
        } else {
            size_t start = pos;
            while (pos < text.size() && !std::isalnum(static_cast<unsigned char>(text[pos])) &&
                   text[pos] != '\'' && text[pos] != '-') {
                ++pos;
            }
            tokens.push_back({text.substr(start, pos - start), false});
        }
    }

    std::vector<size_t> word_indices;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].is_word) word_indices.push_back(i);
    }

    std::vector<bool> consumed(tokens.size(), false);

    auto strip_suffix = [](const std::string& word) -> std::pair<std::string, std::string> {
        if (word.size() >= 3 && word.substr(word.size() - 2) == "'s") {
            return {word.substr(0, word.size() - 2), "'s"};
        }
        if (word.size() >= 3 && word.substr(word.size() - 2) == "'t") {
            return {word.substr(0, word.size() - 2), "'t"};
        }
        if (word.size() >= 4 && word.back() == 's' &&
            word[word.size() - 2] != 's' && // avoid stripping from "boss", "class"
            std::isalpha(static_cast<unsigned char>(word[word.size() - 2]))) {
            return {word.substr(0, word.size() - 1), "s"};
        }
        return {word, ""};
    };

    size_t wi = 0;
    while (wi < word_indices.size()) {
        size_t best_dist = std::numeric_limits<size_t>::max();
        const std::string* best_match = nullptr;
        size_t best_window = 0;
        size_t best_first_token = 0;
        size_t best_last_token = 0;
        std::string best_suffix;

        for (size_t window = std::min<size_t>(3, word_indices.size() - wi); window >= 1; --window) {
            std::string window_collapsed;
            const size_t first_tok = word_indices[wi];
            const size_t last_tok = word_indices[wi + window - 1];
            for (size_t w = 0; w < window; ++w) {
                window_collapsed += tokens[word_indices[wi + w]].text;
            }

            if (window == 1 && window_collapsed.size() < 3) break;

            auto [stem, suffix] = strip_suffix(window_collapsed);
            const std::string* candidates[] = {&window_collapsed, &stem};
            const std::string suffixes[] = {"", suffix};
            const size_t num_candidates = suffix.empty() ? 1 : 2;

            for (size_t ci = 0; ci < num_candidates; ++ci) {
                const std::string& candidate = *candidates[ci];
                if (candidate.empty()) continue;

                for (const auto& entry : vocab_entries) {
                    const size_t wlen = candidate.size();
                    const size_t vlen = entry.collapsed.size();

                    const size_t len_diff = wlen > vlen ? wlen - vlen : vlen - wlen;
                    const size_t max_dist = std::max<size_t>(1, std::min(wlen, vlen) / 3);
                    if (len_diff > max_dist) continue;

                    const size_t dist = levenshtein_ci(candidate, entry.collapsed);

                    // For single-edit corrections, require first char match to prevent
                    // false positives like "vortex" → "Cortex".
                    if (dist == 1 && window == 1) {
                        const bool first_char_match =
                            std::tolower(static_cast<unsigned char>(candidate[0])) ==
                            std::tolower(static_cast<unsigned char>(entry.collapsed[0]));
                        if (!first_char_match) continue;
                    }

                    if (dist <= max_dist && dist < best_dist) {
                        best_dist = dist;
                        best_match = entry.original;
                        best_window = window;
                        best_first_token = first_tok;
                        best_last_token = last_tok;
                        best_suffix = suffixes[ci];
                    }
                }
            }

            if (best_dist == 0) break;
        }

        // Allow dist==0 for multi-word merges where word boundaries changed.
        const bool should_replace = best_match &&
            best_dist != std::numeric_limits<size_t>::max() &&
            (best_dist > 0 || best_window > 1);

        if (should_replace) {
            tokens[best_first_token].text = *best_match + best_suffix;
            for (size_t t = best_first_token + 1; t <= best_last_token; ++t) {
                consumed[t] = true;
            }
            for (size_t t = best_first_token + 1; t <= best_last_token; ++t) {
                if (t > 0) consumed[t - 1] = consumed[t - 1] || !tokens[t - 1].is_word;
            }
            wi += best_window;
        } else {
            ++wi;
        }
    }

    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!consumed[i]) {
            result += tokens[i].text;
        }
    }

    text = std::move(result);
}

inline InferenceOptions parse_inference_options_json(const std::string& json) {
    InferenceOptions options;

    if (json.empty()) return options;

    size_t pos = json.find("\"temperature\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.temperature = std::stof(json.substr(pos));
    }

    pos = json.find("\"top_p\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.top_p = std::stof(json.substr(pos));
    }

    float parsed_min_p = options.min_p;
    if (try_parse_json_float(json, "min_p", parsed_min_p)) {
        options.min_p = std::clamp(parsed_min_p, 0.0f, 1.0f);
    }

    float parsed_rep_penalty = options.repetition_penalty;
    if (try_parse_json_float(json, "repetition_penalty", parsed_rep_penalty)) {
        if (std::isfinite(parsed_rep_penalty) && parsed_rep_penalty > 0.0f) {
            options.repetition_penalty = parsed_rep_penalty;
        }
    }

    pos = json.find("\"top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.top_k = std::stoul(json.substr(pos));
    }

    pos = json.find("\"max_tokens\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.max_tokens = std::stoul(json.substr(pos));
    }

    pos = json.find("\"force_tools\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.force_tools = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"tool_rag_top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.tool_rag_top_k = std::stoul(json.substr(pos));
    }

    pos = json.find("\"confidence_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.confidence_threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"include_stop_sequences\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.include_stop_sequences = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"use_vad\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.use_vad = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"telemetry_enabled\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.telemetry_enabled = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"auto_handoff\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.auto_handoff = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"cloud_timeout_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        options.cloud_timeout_ms = std::stoul(json.substr(pos));
    }

    pos = json.find("\"handoff_with_images\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.handoff_with_images = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"enable_thinking_if_supported\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        options.enable_thinking_if_supported = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"stop_sequences\"");
    if (pos != std::string::npos) {
        pos = json.find('[', pos);
        if (pos != std::string::npos) {
            size_t end_pos = json.find(']', pos);
            size_t seq_pos = json.find('"', pos);

            while (seq_pos != std::string::npos && seq_pos < end_pos) {
                size_t seq_start = seq_pos + 1;
                size_t seq_end = json.find('"', seq_start);
                if (seq_end != std::string::npos) {
                    options.stop_sequences.push_back(json.substr(seq_start, seq_end - seq_start));
                }
                seq_pos = json.find('"', seq_end + 1);
            }
        }
    }

    return options;
}

static inline std::string trim_lfm2_slice(const std::string& value, size_t begin, size_t end) {
    return trim_string(value.substr(begin, end - begin));
}

static inline void append_lfm2_call(const std::string& entry,
                                   std::vector<std::string>& function_calls) {
    if (entry.empty()) return;

    std::string trimmed_entry = trim_lfm2_slice(entry, 0, entry.size());
    if (trimmed_entry.empty()) return;

    size_t paren_pos = trimmed_entry.find('(');
    if (paren_pos == std::string::npos) return;

    std::string func_name = trim_lfm2_slice(trimmed_entry, 0, paren_pos);
    std::string args_str = trim_lfm2_slice(trimmed_entry, paren_pos + 1, trimmed_entry.size());

    while (!args_str.empty() && (args_str.back() == ')' || args_str.back() == ']')) {
        args_str.pop_back();
    }
    args_str = trim_lfm2_slice(args_str, 0, args_str.size());

    std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":{";

    size_t arg_pos = 0;
    bool first_arg = true;
    while (arg_pos < args_str.length()) {
        while (arg_pos < args_str.length() && std::isspace(static_cast<unsigned char>(args_str[arg_pos]))) {
            arg_pos++;
        }

        size_t eq_pos = args_str.find('=', arg_pos);
        if (eq_pos == std::string::npos) break;

        std::string arg_name = args_str.substr(arg_pos, eq_pos - arg_pos);

        size_t val_start = eq_pos + 1;
        size_t val_end = val_start;
        bool quoted = false;

        if (val_start < args_str.length() && args_str[val_start] == '"') {
            quoted = true;
            val_start++;
            val_end = args_str.find('"', val_start);
            if (val_end == std::string::npos) break;
        } else {
            int depth = 0;
            for (val_end = val_start; val_end < args_str.length(); val_end++) {
                char c = args_str[val_end];
                if (c == '[' || c == '{') depth++;
                else if (c == ']' || c == '}') depth--;
                else if (c == ',' && depth == 0) break;
            }
        }

        std::string arg_value = args_str.substr(val_start, val_end - val_start);

        if (!quoted) {
            if (arg_value == "True") arg_value = "true";
            else if (arg_value == "False") arg_value = "false";
            else if (arg_value == "None") arg_value = "null";
        }

        if (!first_arg) json_call += ",";
        json_call += "\"" + arg_name + "\":";
        json_call += quoted ? ("\"" + arg_value + "\"") : arg_value;
        first_arg = false;

        arg_pos = args_str.find(',', val_end);
        if (arg_pos != std::string::npos) {
            arg_pos++;
        } else {
            break;
        }
    }

    json_call += "}}";
    function_calls.push_back(json_call);
}

inline void parse_function_calls_from_response(const std::string& response_text,
                                               std::string& regular_response,
                                               std::vector<std::string>& function_calls) {
    regular_response = response_text;
    function_calls.clear();

    gemma::parse_function_calls(regular_response, function_calls);

    const std::string QWEN_TOOL_START = "<tool_call>";
    const std::string QWEN_TOOL_END = "</tool_call>";
    size_t qwen_start_pos = 0;

    while ((qwen_start_pos = regular_response.find(QWEN_TOOL_START, qwen_start_pos)) != std::string::npos) {
        size_t content_start = qwen_start_pos + QWEN_TOOL_START.length();
        size_t qwen_end_pos = regular_response.find(QWEN_TOOL_END, content_start);

        size_t erase_end;
        std::string json_content;

        if (qwen_end_pos != std::string::npos) {
            json_content = regular_response.substr(content_start, qwen_end_pos - content_start);
            erase_end = qwen_end_pos + QWEN_TOOL_END.length();
        } else {
            json_content = regular_response.substr(content_start);
            erase_end = regular_response.length();
        }

        size_t first = json_content.find_first_not_of(" \t\n\r");
        size_t last = json_content.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos) {
            json_content = json_content.substr(first, last - first + 1);
        }

        if (json_content.size() > 2 && json_content.find("\"name\"") != std::string::npos) {
            // Unwrap array wrapper if present: [{"name":...}] -> {"name":...}
            if (json_content[0] == '[') {
                size_t obj_start = json_content.find('{');
                size_t obj_end = json_content.rfind('}');
                if (obj_start != std::string::npos && obj_end != std::string::npos && obj_end > obj_start) {
                    json_content = json_content.substr(obj_start, obj_end - obj_start + 1);
                }
            }
            if (json_content[0] == '{') {
                size_t depth = 0;
                bool in_string = false;
                bool escaped = false;
                size_t end_pos = 0;
                for (size_t c = 0; c < json_content.size(); c++) {
                    char ch = json_content[c];
                    if (escaped) { escaped = false; continue; }
                    if (ch == '\\' && in_string) { escaped = true; continue; }
                    if (ch == '"') { in_string = !in_string; continue; }
                    if (!in_string) {
                        if (ch == '{') depth++;
                        else if (ch == '}' && --depth == 0) { end_pos = c + 1; break; }
                    }
                }
                if (end_pos > 0) {
                    function_calls.push_back(json_content.substr(0, end_pos));
                }
            }
        }

        regular_response.erase(qwen_start_pos, erase_end - qwen_start_pos);
    }

    const std::string TOOL_CALL_START = "<|tool_call_start|>";
    const std::string TOOL_CALL_END = "<|tool_call_end|>";
    size_t lfm2_start_pos = 0;

    while ((lfm2_start_pos = regular_response.find(TOOL_CALL_START, lfm2_start_pos)) != std::string::npos) {
        size_t content_start = lfm2_start_pos + TOOL_CALL_START.length();
        size_t tool_end_pos = regular_response.find(TOOL_CALL_END, content_start);

        if (tool_end_pos != std::string::npos) {
            std::string tool_content = regular_response.substr(content_start, tool_end_pos - content_start);
            std::string content = tool_content;
            size_t trim_start = 0;
            while (trim_start < content.size() && std::isspace(static_cast<unsigned char>(content[trim_start]))) {
                trim_start++;
            }

            if (trim_start < content.size()) {
                size_t trim_end = content.size() - 1;
                while (trim_end > trim_start && std::isspace(static_cast<unsigned char>(content[trim_end]))) {
                    trim_end--;
                }
                content = content.substr(trim_start, trim_end - trim_start + 1);
            } else {
                content.clear();
            }

            if (!content.empty() && content.front() == '[' && content.back() == ']') {
                std::string inner = content.substr(1, content.size() - 2);

                size_t inner_first = inner.find_first_not_of(" \t\n\r");
                if (inner_first != std::string::npos && inner[inner_first] == '{') {
                    size_t pos = inner_first;
                    while (pos < inner.size()) {
                        if (inner[pos] == '{') {
                            int brace_depth = 1;
                            size_t obj_start = pos;
                            pos++;
                            while (pos < inner.size() && brace_depth > 0) {
                                if (inner[pos] == '{') brace_depth++;
                                else if (inner[pos] == '}') brace_depth--;
                                pos++;
                            }
                            if (brace_depth == 0) {
                                std::string json_obj = inner.substr(obj_start, pos - obj_start);
                                if (json_obj.find("\"name\"") != std::string::npos) {
                                    function_calls.push_back(json_obj);
                                }
                            }
                        } else {
                            pos++;
                        }
                    }
                } else {
                    size_t start = 0;
                    int paren_depth = 0;

                    for (size_t i = 0; i < inner.size(); ++i) {
                        char c = inner[i];
                        if (c == '(') {
                            paren_depth++;
                        } else if (c == ')' && paren_depth > 0) {
                            paren_depth--;
                        } else if (c == ',' && paren_depth == 0) {
                            append_lfm2_call(inner.substr(start, i - start), function_calls);
                            start = i + 1;
                        }
                    }

                    if (start < inner.size()) {
                        append_lfm2_call(inner.substr(start), function_calls);
                    }
                }
            } else if (!content.empty()) {
                append_lfm2_call(content, function_calls);
            }

            regular_response.erase(lfm2_start_pos, tool_end_pos + TOOL_CALL_END.length() - lfm2_start_pos);
        } else {
            break;
        }
    }

    const char* FUNCTION_CALL_MARKER = "\"function_call\"";
    size_t search_pos = 0;
    const size_t text_len = regular_response.length();

    while (search_pos < text_len) {
        size_t marker_pos = regular_response.find(FUNCTION_CALL_MARKER, search_pos);
        if (marker_pos == std::string::npos) break;

        size_t json_start = regular_response.find('{', marker_pos);
        if (json_start == std::string::npos) break;

        int brace_count = 1;
        size_t json_end = json_start + 1;
        while (json_end < text_len && brace_count > 0) {
            char c = regular_response[json_end];
            brace_count += (c == '{') - (c == '}');
            json_end++;
        }

        if (brace_count == 0) {
            function_calls.push_back(regular_response.substr(json_start, json_end - json_start));
            regular_response = regular_response.substr(0, marker_pos);
            size_t last_bracket = regular_response.rfind('{');
            if(last_bracket != std::string::npos) {
                regular_response = regular_response.substr(0, last_bracket);
            }
        }
        search_pos = json_end;
    }
}

inline std::vector<std::pair<size_t, size_t>> find_channel_token_ranges(
    const std::vector<uint32_t>& tokens, size_t offset,
    uint32_t channel_open_id, uint32_t channel_close_id) {
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t pos = 0;
    while (pos < tokens.size()) {
        if (tokens[pos] != channel_open_id) {
            pos++;
            continue;
        }

        size_t block_start = pos;
        pos++;
        while (pos < tokens.size() && tokens[pos] != channel_close_id) {
            pos++;
        }
        if (pos < tokens.size()) {
            pos++;
        }
        ranges.push_back({offset + block_start, pos - block_start});
    }
    return ranges;
}

inline void strip_tag_blocks(std::string& text, std::string& extracted,
                             const std::string& open_tag, const std::string& close_tag) {
    std::string result;
    size_t pos = 0;

    size_t first_close = text.find(close_tag);
    size_t first_open = text.find(open_tag);
    if (first_close != std::string::npos &&
        (first_open == std::string::npos || first_close < first_open)) {
        extracted += text.substr(0, first_close);
        pos = first_close + close_tag.size();
    }

    while (pos < text.size()) {
        size_t open_pos = text.find(open_tag, pos);
        if (open_pos == std::string::npos) {
            result += text.substr(pos);
            break;
        }
        result += text.substr(pos, open_pos - pos);
        size_t content_start = open_pos + open_tag.size();
        size_t close_pos = text.find(close_tag, content_start);
        if (close_pos == std::string::npos) {
            if (!extracted.empty()) extracted += "\n";
            extracted += text.substr(content_start);
            break;
        }
        if (!extracted.empty()) extracted += "\n";
        extracted += text.substr(content_start, close_pos - content_start);
        pos = close_pos + close_tag.size();
    }
    text = result;
}

inline void strip_thinking_block(const std::string& input, std::string& thinking, std::string& content) {
    thinking.clear();
    content = input;

    auto trim = [](std::string& s) {
        size_t first = s.find_first_not_of(" \t\n\r");
        size_t last = s.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos)
            s = s.substr(first, last - first + 1);
        else
            s.clear();
    };

    if (content.find("<|channel>") != std::string::npos || content.find("<channel|>") != std::string::npos) {
        strip_tag_blocks(content, thinking, "<|channel>", "<channel|>");
    } else if (content.find("<think>") != std::string::npos || content.find("</think>") != std::string::npos) {
        strip_tag_blocks(content, thinking, "<think>", "</think>");
    } else {
        return;
    }

    trim(thinking);
    trim(content);
}

struct TranscriptSegment {
    float start;
    float end;
    std::string text;
};

inline std::string construct_response_json(const std::string& regular_response,
                                           const std::vector<std::string>& function_calls,
                                           double time_to_first_token,
                                           double total_time_ms,
                                           double prefill_tps,
                                           double decode_tps,
                                           size_t prompt_tokens,
                                           size_t completion_tokens,
                                           float confidence = 0.0f,
                                           bool cloud_handoff = false,
                                           const std::string& thinking = "",
                                           const std::vector<TranscriptSegment>& segments = {}) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":true,";
    json << "\"error\":null,";
    json << "\"cloud_handoff\":" << (cloud_handoff ? "true" : "false") << ",";
    json << "\"response\":\"" << escape_json_string(regular_response) << "\",";
    if (!thinking.empty()) {
        json << "\"thinking\":\"" << escape_json_string(thinking) << "\",";
    }
    json << "\"function_calls\":[";
    for (size_t i = 0; i < function_calls.size(); ++i) {
        if (i > 0) json << ",";
        json << function_calls[i];
    }
    json << "],";
    json << "\"segments\":[";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) json << ",";
        json << "{\"start\":" << std::fixed << std::setprecision(3) << segments[i].start
             << ",\"end\":" << std::fixed << std::setprecision(3) << segments[i].end
             << ",\"text\":\"" << escape_json_string(segments[i].text) << "\"}";
    }
    json << "],";
    json << "\"confidence\":" << std::fixed << std::setprecision(4) << confidence << ",";
    json << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ",";
    json << "\"decode_tps\":" << std::fixed << std::setprecision(2) << decode_tps << ",";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb() << ",";
    json << "\"prefill_tokens\":" << prompt_tokens << ",";
    json << "\"decode_tokens\":" << completion_tokens << ",";
    json << "\"total_tokens\":" << (prompt_tokens + completion_tokens);
    json << "}";
    return json.str();
}

inline std::string serialize_function_calls(const std::vector<std::string>& calls) {
    if (calls.empty()) return "[]";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i > 0) oss << ",";
        oss << calls[i];
    }
    oss << "]";
    return oss.str();
}

inline int validate_audio_params(
    const char* component,
    void* model,
    char* response_buffer, size_t buffer_size,
    const char* audio_file_path,
    const uint8_t* pcm_buffer, size_t pcm_buffer_size) {
    if (!model) {
        std::string err = last_error_message.empty() ? "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR(component, err);
        handle_error_response(err, response_buffer, buffer_size);
        return -1;
    }
    if (!response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR(component, "Invalid parameters: response_buffer or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }
    if (!audio_file_path && (!pcm_buffer || pcm_buffer_size == 0)) {
        CACTUS_LOG_ERROR(component, "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }
    if (audio_file_path && pcm_buffer && pcm_buffer_size > 0) {
        CACTUS_LOG_ERROR(component, "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        return -1;
    }
    if (pcm_buffer && pcm_buffer_size > 0 && (pcm_buffer_size < 2 || pcm_buffer_size % 2 != 0)) {
        CACTUS_LOG_ERROR(component, "Invalid pcm_buffer_size");
        handle_error_response("pcm_buffer_size must be even and at least 2 bytes", response_buffer, buffer_size);
        return -1;
    }
    return 0;
}

inline std::vector<float> pcm_to_float(const uint8_t* pcm_buffer, size_t pcm_buffer_size) {
    const int16_t* samples = reinterpret_cast<const int16_t*>(pcm_buffer);
    size_t n = pcm_buffer_size / 2;
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>(samples[i]) / 32768.0f;
    return out;
}

} // namespace ffi
} // namespace cactus

#ifdef __cplusplus
extern "C" {
#endif

const char* cactus_get_last_error();

#ifdef __cplusplus
}
#endif

#endif // CACTUS_UTILS_H
