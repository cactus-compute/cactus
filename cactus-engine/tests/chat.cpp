#include "../cactus_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <SDL_audio.h>
#endif

namespace {

constexpr int kMaxTokens = 1024;
constexpr size_t kResponseBufferSize = kMaxTokens * 128;

#ifdef HAVE_SDL2
constexpr int kRecordSampleRate = 16000;

struct RecordState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
    int actual_sample_rate = kRecordSampleRate;
    SDL_AudioFormat actual_format = AUDIO_S16LSB;
    int actual_channels = 1;
};

RecordState g_record;

void record_callback(void*, Uint8* stream, int len) {
    if (!g_record.recording) return;
    std::lock_guard<std::mutex> lock(g_record.mutex);
    g_record.buffer.insert(g_record.buffer.end(), stream, stream + len);
}

std::vector<float> decode_sdl_audio_to_mono_f32(const std::vector<uint8_t>& input,
                                                SDL_AudioFormat format,
                                                int channels) {
    if (input.empty() || channels <= 0) return {};

    size_t bytes_per_sample = SDL_AUDIO_BITSIZE(format) / 8;
    if (bytes_per_sample == 0) return {};
    size_t frame_count = input.size() / (bytes_per_sample * static_cast<size_t>(channels));
    std::vector<float> mono(frame_count);

    auto sample_at = [&](size_t sample_index) -> float {
        const uint8_t* p = input.data() + sample_index * bytes_per_sample;
        switch (format) {
            case AUDIO_S16LSB: {
                int16_t v;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v) / 32768.0f;
            }
            case AUDIO_U16LSB: {
                uint16_t v;
                std::memcpy(&v, p, sizeof(v));
                return (static_cast<float>(v) - 32768.0f) / 32768.0f;
            }
            case AUDIO_S16MSB: {
                int16_t v = static_cast<int16_t>((p[0] << 8) | p[1]);
                return static_cast<float>(v) / 32768.0f;
            }
            case AUDIO_U16MSB: {
                uint16_t v = static_cast<uint16_t>((p[0] << 8) | p[1]);
                return (static_cast<float>(v) - 32768.0f) / 32768.0f;
            }
            case AUDIO_S8:
                return static_cast<float>(*reinterpret_cast<const int8_t*>(p)) / 128.0f;
            case AUDIO_U8:
                return (static_cast<float>(*p) - 128.0f) / 128.0f;
            case AUDIO_F32LSB: {
                float v;
                std::memcpy(&v, p, sizeof(v));
                return std::clamp(v, -1.0f, 1.0f);
            }
            default:
                return 0.0f;
        }
    };

    for (size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            sum += sample_at(frame * static_cast<size_t>(channels) + static_cast<size_t>(ch));
        }
        mono[frame] = sum / static_cast<float>(channels);
    }
    return mono;
}

std::vector<uint8_t> resample_f32_to_s16_pcm(const std::vector<float>& input, int source_rate, int target_rate) {
    if (input.empty()) return {};
    double ratio = static_cast<double>(target_rate) / static_cast<double>(source_rate);
    size_t out_count = static_cast<size_t>(static_cast<double>(input.size()) * ratio);
    if (out_count == 0) return {};

    std::vector<int16_t> out(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        double src_pos = static_cast<double>(i) / ratio;
        size_t i0 = static_cast<size_t>(src_pos);
        size_t i1 = std::min(i0 + 1, input.size() - 1);
        double frac = src_pos - static_cast<double>(i0);
        double sample = static_cast<double>(input[i0]) * (1.0 - frac) + static_cast<double>(input[i1]) * frac;
        sample = std::clamp(sample, -1.0, 1.0);
        out[i] = static_cast<int16_t>(std::lrint(sample * 32767.0));
    }

    std::vector<uint8_t> result(out.size() * sizeof(int16_t));
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

bool record_audio(std::vector<uint8_t>& pcm_out) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << "Failed to init SDL audio: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_AudioSpec want;
    SDL_AudioSpec have;
    SDL_zero(want);
    want.freq = kRecordSampleRate;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = static_cast<Uint16>((kRecordSampleRate * 100) / 1000);
    want.callback = record_callback;

    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have,
                                                   SDL_AUDIO_ALLOW_FREQUENCY_CHANGE |
                                                   SDL_AUDIO_ALLOW_FORMAT_CHANGE |
                                                   SDL_AUDIO_ALLOW_CHANNELS_CHANGE);
    if (device == 0) {
        std::cerr << "Failed to open microphone: " << SDL_GetError() << "\n";
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_record.mutex);
        g_record.buffer.clear();
    }
    g_record.actual_sample_rate = have.freq;
    g_record.actual_format = have.format;
    g_record.actual_channels = have.channels;
    g_record.recording = true;
    SDL_PauseAudioDevice(device, 0);

    std::cout << "Recording... press Enter to stop.\n" << std::flush;
    std::string line;
    std::getline(std::cin, line);

    g_record.recording = false;
    SDL_PauseAudioDevice(device, 1);

    {
        std::lock_guard<std::mutex> lock(g_record.mutex);
        auto mono = decode_sdl_audio_to_mono_f32(g_record.buffer,
                                                 g_record.actual_format,
                                                 g_record.actual_channels);
        pcm_out = resample_f32_to_s16_pcm(mono, g_record.actual_sample_rate, kRecordSampleRate);
    }

    SDL_CloseAudioDevice(device);
    SDL_QuitSubSystem(SDL_INIT_AUDIO);

    double seconds = static_cast<double>(pcm_out.size() / sizeof(int16_t)) / kRecordSampleRate;
    std::cout << "Recorded " << std::fixed << std::setprecision(1) << seconds << "s of audio.\n";
    return !pcm_out.empty();
}
#endif

struct TokenPrinter {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point first;
    bool saw_first = false;
    int count = 0;

    void reset() {
        start = std::chrono::steady_clock::now();
        saw_first = false;
        count = 0;
    }

    void on_token(const char* text) {
        if (!saw_first) {
            first = std::chrono::steady_clock::now();
            saw_first = true;
        }
        std::cout << (text ? text : "") << std::flush;
        ++count;
    }

    void print_stats(double ram_mb) const {
        auto end = std::chrono::steady_clock::now();
        double total_s = std::chrono::duration<double>(end - start).count();
        double ttft_s = saw_first ? std::chrono::duration<double>(first - start).count() : 0.0;
        double decode_s = saw_first ? std::chrono::duration<double>(end - first).count() : total_s;
        double tps = (count > 1 && decode_s > 0.0) ? (count - 1) / decode_s : (total_s > 0.0 ? count / total_s : 0.0);
        std::cout << "\n[" << count << " tokens | latency: "
                  << std::fixed << std::setprecision(3) << ttft_s
                  << "s | total: " << total_s
                  << "s | " << std::setprecision(1) << tps << " tok/s";
        if (ram_mb > 0.0) {
            std::cout << " | RAM: " << ram_mb << " MB";
        }
        std::cout << "]\n";
    }
};

TokenPrinter* g_printer = nullptr;

void token_callback(const char* text, uint32_t, void*) {
    if (g_printer) {
        g_printer->on_token(text);
    }
}

std::string escape_json(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

std::string unescape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        char n = s[++i];
        switch (n) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default: out.push_back(n); break;
        }
    }
    return out;
}

std::string expand_tilde(const std::string& path) {
    if (path.size() < 2 || path[0] != '~' || path[1] != '/') return path;
    const char* home = std::getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string json_string_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    size_t start = json.find(needle);
    if (start == std::string::npos) return {};
    start += needle.size();
    size_t end = start;
    while ((end = json.find('"', end)) != std::string::npos) {
        size_t slashes = 0;
        for (size_t i = end; i > start && json[i - 1] == '\\'; --i) ++slashes;
        if ((slashes % 2) == 0) break;
        ++end;
    }
    if (end == std::string::npos) return {};
    return unescape_json(json.substr(start, end - start));
}

double json_number_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t start = json.find(needle);
    if (start == std::string::npos) return 0.0;
    start += needle.size();
    char* end = nullptr;
    return std::strtod(json.c_str() + start, &end);
}

std::string build_messages(const std::string& system_prompt,
                           const std::vector<std::pair<std::string, std::string>>& history,
                           const std::string& image,
                           const std::string& audio,
                           bool attach_media) {
    std::ostringstream msg;
    msg << "[";
    bool need_comma = false;
    if (!system_prompt.empty()) {
        msg << "{\"role\":\"system\",\"content\":\"" << escape_json(system_prompt) << "\"}";
        need_comma = true;
    }
    for (size_t i = 0; i < history.size(); ++i) {
        if (need_comma) msg << ",";
        need_comma = true;
        msg << "{\"role\":\"" << history[i].first << "\",\"content\":\""
            << escape_json(history[i].second) << "\"";
        if (attach_media && i + 1 == history.size() && history[i].first == "user") {
            if (!image.empty()) msg << ",\"images\":[\"" << escape_json(image) << "\"]";
            if (!audio.empty()) msg << ",\"audio\":[\"" << escape_json(audio) << "\"]";
        }
        msg << "}";
    }
    msg << "]";
    return msg.str();
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <model_path> [--system <prompt>] [--image <path>] [--audio <path>]"
              << " [--prompt <text>] [--thinking]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string system_prompt;
    std::string current_image;
    std::string current_audio;
    std::string initial_prompt;
    bool thinking = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            current_image = expand_tilde(argv[++i]);
        } else if (arg == "--audio" && i + 1 < argc) {
            current_audio = expand_tilde(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            initial_prompt = argv[++i];
        } else if (arg == "--thinking") {
            thinking = true;
        }
    }

    if (!current_image.empty() && !file_exists(current_image)) {
        std::cerr << "Image not found: " << current_image << "\n";
        return 1;
    }
    if (!current_audio.empty() && !file_exists(current_audio)) {
        std::cerr << "Audio file not found: " << current_audio << "\n";
        return 1;
    }

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path.c_str(), nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }

    std::cout << "Model loaded.\n";
    std::cout << "Commands: /image <path> [prompt], /audio <path> [prompt], ";
#ifdef HAVE_SDL2
    std::cout << "/record [prompt], ";
#endif
    std::cout << "/clear, reset, exit\n\n";

    std::vector<std::pair<std::string, std::string>> history;
    std::vector<uint8_t> current_pcm;
    TokenPrinter printer;
    g_printer = &printer;
    bool auto_send = !initial_prompt.empty() || !current_audio.empty() || !current_image.empty();

    while (true) {
        std::string input;
        if (auto_send) {
            auto_send = false;
            input = initial_prompt.empty() ? "Describe the attached input." : initial_prompt;
            std::cout << "You: " << input << "\n";
        } else {
            std::cout << "You: " << std::flush;
            if (!std::getline(std::cin, input)) break;
        }

        while (!input.empty() && (input.back() == ' ' || input.back() == '\t')) input.pop_back();
        if (input.empty()) continue;
        if (input == "exit" || input == "quit") break;
        if (input == "reset") {
            history.clear();
            current_image.clear();
            current_audio.clear();
            current_pcm.clear();
            cactus_reset(model);
            std::cout << "Conversation reset.\n";
            continue;
        }
        if (input == "/clear") {
            current_image.clear();
            current_audio.clear();
            current_pcm.clear();
            std::cout << "Attachments cleared.\n";
            continue;
        }

        auto parse_attachment = [&](const std::string& prefix, std::string& target) -> bool {
            if (input.rfind(prefix, 0) != 0) return false;
            std::string rest = input.substr(prefix.size());
            size_t split = rest.find(' ');
            std::string path = expand_tilde(split == std::string::npos ? rest : rest.substr(0, split));
            if (!file_exists(path)) {
                std::cerr << "File not found: " << path << "\n";
                input.clear();
                return true;
            }
            target = path;
            input = split == std::string::npos ? "" : rest.substr(split + 1);
            return true;
        };

        if (parse_attachment("/image ", current_image) && input.empty()) {
            std::cout << "Image attached: " << current_image << "\n";
            continue;
        }
        if (parse_attachment("/audio ", current_audio) && input.empty()) {
            std::cout << "Audio attached: " << current_audio << "\n";
            continue;
        }

        if (input == "/record" || input.rfind("/record ", 0) == 0) {
#ifdef HAVE_SDL2
            std::string record_prompt;
            if (input.size() > 8) {
                record_prompt = input.substr(8);
                while (!record_prompt.empty() && (record_prompt.front() == ' ' || record_prompt.front() == '\t')) {
                    record_prompt.erase(record_prompt.begin());
                }
            }
            current_pcm.clear();
            current_audio.clear();
            if (!record_audio(current_pcm)) {
                std::cerr << "Recording failed.\n";
                continue;
            }
            input = record_prompt.empty() ? "Transcribe or respond to this audio." : record_prompt;
#else
            std::cerr << "Recording requires SDL2, but this chat binary was built without SDL2.\n";
            continue;
#endif
        }
        if (input.empty()) continue;

        bool attach_media = !current_image.empty() || !current_audio.empty() || !current_pcm.empty();
        if (attach_media) {
            cactus_reset(model);
        }
        history.push_back({"user", input});
        std::string messages = build_messages(system_prompt, history, current_image, current_audio, attach_media);
        std::string options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":"
            + std::to_string(kMaxTokens)
            + ",\"enable_thinking_if_supported\":" + (thinking ? "true" : "false")
            + ",\"auto_handoff\":false,\"confidence_threshold\":0.0"
            + ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

        if (!current_image.empty()) std::cout << "[image: " << current_image << "]\n";
        if (!current_audio.empty()) std::cout << "[audio: " << current_audio << "]\n";
        if (!current_pcm.empty()) {
            double seconds = static_cast<double>(current_pcm.size() / sizeof(int16_t)) / 16000.0;
            std::cout << "[recorded audio: " << std::fixed << std::setprecision(1) << seconds << "s]\n";
        }
        std::cout << "Assistant: " << std::flush;

        std::vector<char> response(kResponseBufferSize, 0);
        printer.reset();
        int rc = cactus_complete(model,
                                 messages.c_str(),
                                 response.data(),
                                 response.size(),
                                 options.c_str(),
                                 nullptr,
                                 token_callback,
                                 nullptr,
                                 current_pcm.empty() ? nullptr : current_pcm.data(),
                                 current_pcm.size());

        std::string response_json(response.data());
        double ram_mb = json_number_value(response_json, "ram_usage_mb");
        printer.print_stats(ram_mb);
        std::cout << "\n";

        if (rc < 0) {
            std::cerr << "Error: " << response.data() << "\n";
            history.pop_back();
            continue;
        }

        std::string assistant = json_string_value(response_json, "response");
        history.push_back({"assistant", assistant});
        current_image.clear();
        current_audio.clear();
        current_pcm.clear();
    }

    cactus_destroy(model);
    std::cout << "Goodbye.\n";
    return 0;
}
