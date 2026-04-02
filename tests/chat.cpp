#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/telemetry/telemetry.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <SDL_audio.h>

namespace {

constexpr int RECORD_SAMPLE_RATE = 16000;

struct RecordState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
    int actual_sample_rate{RECORD_SAMPLE_RATE};
};

RecordState g_record;

void record_callback(void* /*userdata*/, Uint8* stream, int len) {
    if (!g_record.recording) return;
    std::lock_guard<std::mutex> lock(g_record.mutex);
    g_record.buffer.insert(g_record.buffer.end(), stream, stream + len);
}

std::vector<uint8_t> resample_s16(const std::vector<uint8_t>& input, int source_rate, int target_rate) {
    if (source_rate == target_rate || input.empty()) return input;
    size_t num_in = input.size() / 2;
    if (num_in == 0) return input;
    const int16_t* in = reinterpret_cast<const int16_t*>(input.data());
    double ratio = static_cast<double>(target_rate) / source_rate;
    size_t num_out = static_cast<size_t>(num_in * ratio);
    if (num_out == 0) return {};
    std::vector<int16_t> out(num_out);
    for (size_t i = 0; i < num_out; i++) {
        double src_idx = i / ratio;
        size_t i0 = static_cast<size_t>(src_idx);
        size_t i1 = std::min(i0 + 1, num_in - 1);
        double frac = src_idx - i0;
        double sample = in[i0] * (1.0 - frac) + in[i1] * frac;
        if (sample > 32767.0) sample = 32767.0;
        if (sample < -32768.0) sample = -32768.0;
        out[i] = static_cast<int16_t>(sample);
    }
    std::vector<uint8_t> result(num_out * 2);
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

bool record_audio(std::vector<uint8_t>& pcm_out) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
        return false;
    }

    int num_devices = SDL_GetNumAudioDevices(1);
    if (num_devices == 0) {
        std::cerr << "No audio capture devices found\n";
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = RECORD_SAMPLE_RATE;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (RECORD_SAMPLE_RATE * 100) / 1000;
    want.callback = record_callback;

    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (device == 0) {
        std::cerr << "Failed to open mic: " << SDL_GetError() << "\n";
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    g_record.actual_sample_rate = have.freq;
    g_record.buffer.clear();
    g_record.recording = true;
    SDL_PauseAudioDevice(device, 0);

    std::cout << "Recording... press Enter to stop.\n" << std::flush;

    std::atomic<bool> stop{false};
    std::thread input_thread([&stop]() {
        std::string line;
        std::getline(std::cin, line);
        stop = true;
    });

    while (!stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    g_record.recording = false;
    SDL_PauseAudioDevice(device, 1);

    {
        std::lock_guard<std::mutex> lock(g_record.mutex);
        pcm_out = resample_s16(g_record.buffer, g_record.actual_sample_rate, RECORD_SAMPLE_RATE);
    }

    double duration = (pcm_out.size() / 2) / static_cast<double>(RECORD_SAMPLE_RATE);
    std::cout << "Recorded " << std::fixed << std::setprecision(1) << duration << "s of audio.\n";

    input_thread.join();
    SDL_CloseAudioDevice(device);
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    return !pcm_out.empty();
}

} // anonymous namespace
#endif // HAVE_SDL2

constexpr int MAX_TOKENS = 1024;
constexpr size_t MAX_BYTES_PER_TOKEN = 64;
constexpr size_t RESPONSE_BUFFER_SIZE = MAX_TOKENS * MAX_BYTES_PER_TOKEN;

namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    const std::string CYAN    = "\033[36m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string RED     = "\033[31m";
    const std::string GRAY    = "\033[90m";
}

bool supports_color() {
#ifdef _WIN32
    return false;
#else
    const char* term = std::getenv("TERM");
    return term && std::string(term) != "dumb";
#endif
}

bool use_colors = supports_color();

std::string colored(const std::string& text, const std::string& color) {
    if (!use_colors) return text;
    return color + text + Color::RESET;
}

void print_separator(char ch = '-', int width = 60) {
    std::cout << colored(std::string(width, ch), Color::DIM) << "\n";
}

void print_header(const std::string& sys_prompt, const std::string& image, bool has_vision = true) {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("           🌵 CACTUS CHAT INTERFACE 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("  Commands: ", Color::YELLOW);
    if (has_vision) {
        std::cout << colored("/image <path>", Color::CYAN) << colored(" | ", Color::DIM);
    }
    std::cout << colored("/audio <path>", Color::CYAN) << colored(" | ", Color::DIM)
#ifdef HAVE_SDL2
              << colored("/record", Color::CYAN) << colored(" | ", Color::DIM)
#endif
              << colored("/clear", Color::CYAN) << colored(" | ", Color::DIM)
              << colored("reset", Color::CYAN) << colored(" | ", Color::DIM)
              << colored("exit", Color::CYAN) << "\n";
    if (!sys_prompt.empty()) {
        std::cout << colored("  System prompt active", Color::MAGENTA) << "\n";
    }
    if (!image.empty()) {
        std::cout << colored("  Image: ", Color::MAGENTA) << colored(image, Color::CYAN) << "\n";
    }
    print_separator();
    std::cout << "\n";
}

struct TokenPrinter {
    bool first_token = true;
    int token_count = 0;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point first_token_time;
    double time_to_first_token = 0.0;

    void reset() {
        first_token = true;
        token_count = 0;
        time_to_first_token = 0.0;
        start_time = std::chrono::steady_clock::now();
    }

    void print(const char* token) {
        if (first_token) {
            first_token = false;
            first_token_time = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
            time_to_first_token = latency.count() / 1000.0;
        }
        std::cout << token << std::flush;
        token_count++;
    }

    void print_stats(double ram_mb = 0.0) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double total_seconds = duration.count() / 1000.0;
        double tokens_per_second = token_count / total_seconds;

        std::ostringstream stats;
        stats << std::fixed << std::setprecision(3);
        stats << "[" << token_count << " tokens | ";
        stats << "latency: " << time_to_first_token << "s | ";
        stats << "total: " << total_seconds << "s | ";
        stats << std::setprecision(0) << static_cast<int>(tokens_per_second) << " tok/s";
        if (ram_mb > 0.0) {
            stats << std::fixed << std::setprecision(1) << " | RAM: " << ram_mb << " MB";
        }
        stats << "]";

        std::cout << "\n" << colored(stats.str(), Color::GRAY) << "\n";
    }
};

TokenPrinter* g_printer = nullptr;

void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    if (g_printer) {
        g_printer->print(token);
    }
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
                break;
        }
    }
    return o.str();
}

std::string unescape_json(const std::string& s) {
    std::string result;
    result.reserve(s.length());

    for (size_t i = 0; i < s.length(); i++) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            switch (s[i + 1]) {
                case '"':  result += '"'; i++; break;
                case '\\': result += '\\'; i++; break;
                case 'b':  result += '\b'; i++; break;
                case 'f':  result += '\f'; i++; break;
                case 'n':  result += '\n'; i++; break;
                case 'r':  result += '\r'; i++; break;
                case 't':  result += '\t'; i++; break;
                case 'u':
                    if (i + 5 < s.length()) {
                        std::string hex = s.substr(i + 2, 4);
                        char* end;
                        int codepoint = std::strtol(hex.c_str(), &end, 16);
                        if (end == hex.c_str() + 4) {
                            result += static_cast<char>(codepoint);
                            i += 5;
                        } else {
                            result += s[i];
                        }
                    } else {
                        result += s[i];
                    }
                    break;
                default:   result += s[i]; break;
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

std::string expand_tilde(const std::string& path) {
    if (path.size() < 2 || path[0] != '~' || path[1] != '/') return path;
    const char* home = std::getenv("HOME");
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Missing model path\n";
        std::cerr << "Usage: " << argv[0] << " <model_path> [--system <prompt>] [--image <path>] [--audio <path>] [--prompt <text>] [--no-thinking]\n";
        return 1;
    }

    const char* model_path = argv[1];
    std::string system_prompt;
    std::string current_image;
    std::string current_audio;
    std::string initial_prompt;
    bool enable_thinking = true;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (std::string(argv[i]) == "--image" && i + 1 < argc) {
            current_image = expand_tilde(argv[++i]);
        } else if (std::string(argv[i]) == "--audio" && i + 1 < argc) {
            current_audio = expand_tilde(argv[++i]);
        } else if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
            initial_prompt = argv[++i];
        } else if (std::string(argv[i]) == "--no-thinking") {
            enable_thinking = false;
        }
    }

    if (!current_image.empty()) {
        std::ifstream f(current_image);
        if (!f.good()) {
            std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                      << "Image not found: " << current_image << "\n";
            return 1;
        }
    }

    if (!current_audio.empty()) {
        std::ifstream f(current_audio);
        if (!f.good()) {
            std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                      << "Audio file not found: " << current_audio << "\n";
            return 1;
        }
    }

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);

    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        return 1;
    }

    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    // Check if model supports vision/audio by reading config.txt
    bool has_vision = false;
    bool has_audio_cap = false;
    {
        std::ifstream cfg(std::string(model_path) + "/config.txt");
        std::string line;
        while (std::getline(cfg, line)) {
            if (line.substr(0, 19) == "vision_hidden_size=" && std::stoi(line.substr(19)) > 0) {
                has_vision = true;
            }
            if (line.substr(0, 17) == "audio_hidden_dim=" && std::stoi(line.substr(17)) > 0) {
                has_audio_cap = true;
            }
        }
    }

    if (!current_image.empty() && !has_vision) {
        std::cerr << colored("Warning: ", Color::YELLOW + Color::BOLD)
                  << "This model does not support vision — image will be ignored.\n";
        current_image.clear();
    }

    if (!current_audio.empty() && !has_audio_cap) {
        std::cerr << colored("Warning: ", Color::YELLOW + Color::BOLD)
                  << "This model does not support audio — audio will be ignored.\n";
        current_audio.clear();
    }

    print_header(system_prompt, current_image, has_vision);

    std::vector<std::string> history;
    std::vector<std::string> history_images;
    std::vector<std::string> history_audio;
    std::vector<uint8_t> current_pcm;
    TokenPrinter printer;
    g_printer = &printer;

    bool auto_send = !current_audio.empty() || !initial_prompt.empty();

    while (true) {
        bool has_media = !current_image.empty() || !current_audio.empty();
        std::string input;

        if (auto_send) {
            auto_send = false;
            input = initial_prompt;
            initial_prompt.clear();
            if (has_media && input.empty()) {
                std::cout << colored("You \xf0\x9f\x8e\xa4: ", Color::BLUE + Color::BOLD)
                          << colored("[audio input]", Color::DIM) << "\n";
            } else {
                std::string prompt_label = has_media ? "You \xf0\x9f\x93\x8e: " : "You: ";
                std::cout << colored(prompt_label, Color::BLUE + Color::BOLD) << input << "\n";
            }
        } else {
            std::string prompt = has_media ? "You \xf0\x9f\x93\x8e: " : "You: ";
            std::cout << colored(prompt, Color::BLUE + Color::BOLD);
            if (!std::getline(std::cin, input)) break;

            while (!input.empty() && (input.back() == ' ' || input.back() == '\t')) input.pop_back();
            if (input.empty()) continue;
            if (input == "exit" || input == "quit") break;
        }

        if (input == "reset") {
            history.clear();
            history_images.clear();
            history_audio.clear();
            current_image.clear();
            current_audio.clear();
            cactus_reset(model);
            std::cout << colored("Conversation reset.\n", Color::YELLOW);
            print_header(system_prompt, current_image, has_vision);
            continue;
        }

        if (input == "/clear") {
            current_image.clear();
            current_audio.clear();
            std::cout << colored("Image/audio cleared.\n", Color::YELLOW);
            continue;
        }

        // Parse /image or /audio commands: extract file path and optional trailing message
        auto parse_file_cmd = [](const std::string& in, size_t prefix_len, std::string& out_path, std::string& out_msg) -> bool {
            std::string rest = in.substr(prefix_len);
            size_t space = rest.find(' ');
            out_path = (space != std::string::npos) ? rest.substr(0, space) : rest;
            while (!out_path.empty() && (out_path.back() == ' ' || out_path.back() == '\t')) out_path.pop_back();
            out_path = expand_tilde(out_path);
            std::ifstream f(out_path);
            if (!f.good()) return false;
            out_msg.clear();
            if (space != std::string::npos) {
                out_msg = rest.substr(space + 1);
                while (!out_msg.empty() && (out_msg.front() == ' ' || out_msg.front() == '\t')) out_msg.erase(out_msg.begin());
            }
            return true;
        };

        if (input.substr(0, 7) == "/image ") {
            if (!has_vision) {
                std::cerr << colored("  This model does not support vision.\n", Color::RED);
                continue;
            }
            std::string path, msg;
            if (!parse_file_cmd(input, 7, path, msg)) {
                std::cerr << colored("  File not found: ", Color::RED) << path << "\n";
                continue;
            }
            current_image = path;
            if (msg.empty()) continue;
            input = msg;
        }

        if (input.substr(0, 7) == "/audio ") {
            std::string path, msg;
            if (!parse_file_cmd(input, 7, path, msg)) {
                std::cerr << colored("  File not found: ", Color::RED) << path << "\n";
                continue;
            }
            current_audio = path;
            input = msg;
        }

        if (input == "/record") {
#ifdef HAVE_SDL2
            if (!has_audio_cap) {
                std::cerr << colored("  This model does not support audio.\n", Color::RED);
                continue;
            }
            current_pcm.clear();
            if (!record_audio(current_pcm)) {
                std::cerr << colored("  Recording failed.\n", Color::RED);
                continue;
            }
            input = "";
#else
            std::cerr << colored("  Recording requires SDL2 (not available in this build).\n", Color::RED);
            continue;
#endif
        }

        history.push_back(input);
        history_images.push_back(current_image);
        history_audio.push_back(current_audio);

        // Build messages JSON
        std::ostringstream messages_json;
        messages_json << "[";
        if (!system_prompt.empty()) {
            messages_json << "{\"role\":\"system\",\"content\":\""
                         << escape_json(system_prompt) << "\"},";
        }
        for (size_t i = 0; i < history.size(); i++) {
            if (i > 0) messages_json << ",";
            if (i % 2 == 0) {
                messages_json << "{\"role\":\"user\",\"content\":\""
                             << escape_json(history[i]) << "\"";
                if (!history_images[i].empty()) {
                    messages_json << ",\"images\":[\"" << escape_json(history_images[i]) << "\"]";
                }
                if (!history_audio[i].empty()) {
                    messages_json << ",\"audio\":[\"" << escape_json(history_audio[i]) << "\"]";
                }
                messages_json << "}";
            } else {
                messages_json << "{\"role\":\"assistant\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            }
        }
        messages_json << "]";

        std::string options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":"
                    + std::to_string(MAX_TOKENS)
                    + ",\"enable_thinking_if_supported\":" + (enable_thinking ? "true" : "false")
                    + ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

        std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

        if (!current_audio.empty()) {
            std::cout << colored("  [audio: " + current_audio + "]\n", Color::MAGENTA);
            current_audio.clear();
        }
        if (!current_pcm.empty()) {
            double dur = static_cast<double>(current_pcm.size() / 2) / 16000.0;
            std::cout << colored("  [mic recording: ", Color::MAGENTA)
                      << std::fixed << std::setprecision(1) << dur << "s"
                      << colored("]\n", Color::MAGENTA);
        }
        if (!current_image.empty()) {
            std::cout << colored("  [" + current_image + "]\n", Color::MAGENTA);
        }
        std::cout << colored("Assistant: ", Color::GREEN + Color::BOLD);

        const uint8_t* pcm_ptr = current_pcm.empty() ? nullptr : current_pcm.data();
        size_t pcm_size = current_pcm.size();

        printer.reset();
        int result = cactus_complete(
            model,
            messages_json.str().c_str(),
            response_buffer.data(),
            response_buffer.size(),
            options.c_str(),
            nullptr,
            print_token,
            nullptr,
            pcm_ptr,
            pcm_size
        );

        current_pcm.clear();

        std::string json_str(response_buffer.data(), response_buffer.size());

        double ram_mb = 0.0;
        {
            const std::string ram_key = "\"ram_usage_mb\":";
            size_t ram_pos = json_str.find(ram_key);
            if (ram_pos != std::string::npos) {
                ram_mb = std::stod(json_str.substr(ram_pos + ram_key.length()));
            }
        }

        if (result >= 0) {
            printer.print_stats(ram_mb);
        }

        std::cout << "\n";
        print_separator();
        std::cout << "\n";

        if (result < 0) {
            std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                      << response_buffer.data() << "\n\n";
            history.pop_back();
            history_images.pop_back();
            history_audio.pop_back();
            continue;
        }

        const std::string search_str = "\"response\":\"";
        size_t response_start = json_str.find(search_str);
        if (response_start != std::string::npos) {
            response_start += search_str.length();
            size_t response_end = json_str.find("\"", response_start);
            while (response_end != std::string::npos) {
                size_t prior_backslashes = 0;
                for (size_t i = response_end; i > response_start && json_str[i - 1] == '\\'; i--) {
                    prior_backslashes++;
                }
                if (prior_backslashes % 2 == 0) break;
                response_end = json_str.find("\"", response_end + 1);
            }
            if (response_end != std::string::npos) {
                std::string response = json_str.substr(response_start,
                                                       response_end - response_start);
                history.push_back(unescape_json(response));
                history_images.push_back("");
                history_audio.push_back("");
            }
        }
    }

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return 0;
}
