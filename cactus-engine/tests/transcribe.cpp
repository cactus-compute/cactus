#include "../cactus_engine.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <cctype>
#include <algorithm>
#include <unistd.h>
#include <sys/ioctl.h>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <atomic>
#include <mutex>
#endif

constexpr size_t RESPONSE_BUFFER_SIZE = 65536;

namespace ansi {
constexpr const char* reset  = "\033[0m";
constexpr const char* bold   = "\033[1m";
constexpr const char* dim    = "\033[2m";
constexpr const char* green  = "\033[32m";
constexpr const char* yellow = "\033[33m";
constexpr const char* cyan   = "\033[36m";
}

static bool stdout_is_terminal() {
    return isatty(STDOUT_FILENO) != 0 && std::getenv("NO_COLOR") == nullptr;
}
static const bool g_color = stdout_is_terminal();

static std::string colored(const std::string& text, const char* color) {
    return g_color ? color + text + ansi::reset : text;
}

static int terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1 || w.ws_col == 0) return 80;
    return w.ws_col;
}

static void print_rule() {
    std::string rule(terminal_width(), '-');
    std::cout << colored(rule, ansi::dim) << "\n";
}

static void print_banner() {
    if (!g_color) {
        std::cout << "Cactus Transcription\n\n";
        return;
    }
    std::cout << ansi::bold << ansi::green
              << " ██████╗ █████╗  ██████╗████████╗██╗   ██╗███████╗\n"
              << "██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔════╝\n"
              << "██║     ███████║██║        ██║   ██║   ██║███████╗\n"
              << "██║     ██╔══██║██║        ██║   ██║   ██║╚════██║\n"
              << "╚██████╗██║  ██║╚██████╗   ██║   ╚██████╔╝███████║\n"
              << " ╚═════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚══════╝\n"
              << ansi::reset << ansi::dim << "              Transcription" << ansi::reset << "\n\n";
}

static std::string json_string_value(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":\"";
    size_t start = json.find(pattern);
    if (start == std::string::npos) return "";
    start += pattern.length();
    size_t end = start;
    while (end < json.length() && json[end] != '"') {
        if (json[end] == '\\' && end + 1 < json.length()) end++;
        end++;
    }
    return json.substr(start, end - start);
}

static double json_number_value(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t start = json.find(pattern);
    if (start == std::string::npos) return 0.0;
    start += pattern.length();
    while (start < json.length() && std::isspace((unsigned char)json[start])) start++;
    return std::strtod(json.c_str() + start, nullptr);
}

static std::string whisper_prompt(const std::string& model_path, const std::string& language) {
    std::string p = model_path;
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (p.find("whisper") != std::string::npos)
        return "<|startoftranscript|><|" + language + "|><|transcribe|>";
    return "";
}

static void print_token(const char* token, uint32_t, void*) {
    std::cout << token << std::flush;
}

static int transcribe_file(cactus_model_t model, const std::string& audio_path,
                           const std::string& model_path, const std::string& language) {
    std::string prompt = whisper_prompt(model_path, language);
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);
    std::string options = "{\"max_tokens\":500,\"auto_handoff\":false,\"language\":\"" + language + "\"}";

    auto start = std::chrono::steady_clock::now();
    int rc = cactus_transcribe(
        model, audio_path.c_str(), prompt.empty() ? nullptr : prompt.c_str(),
        response_buffer.data(), response_buffer.size(),
        options.c_str(), print_token, nullptr, nullptr, 0);
    double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (rc < 0) {
        std::cerr << "\nTranscription failed: " << cactus_get_last_error() << "\n";
        return -1;
    }

    std::string json(response_buffer.data());
    double model_ms = json_number_value(json, "total_time_ms");
    std::ostringstream stats;
    stats << std::fixed << std::setprecision(2) << "[processed: " << total_s << "s";
    if (model_ms > 0.0) stats << " | model: " << model_ms / 1000.0 << "s";
    stats << "]";
    std::cout << "\n\n" << colored(stats.str(), ansi::dim) << "\n";
    return 0;
}

#ifdef HAVE_SDL2

constexpr int TARGET_SAMPLE_RATE = 16000;
constexpr int AUDIO_BUFFER_MS = 100;
constexpr int PROCESS_INTERVAL_MS = 250;

struct AudioState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
    int actual_sample_rate{TARGET_SAMPLE_RATE};
};
static AudioState g_audio;

static std::vector<uint8_t> resample_to_16k(const std::vector<uint8_t>& input, int source_rate) {
    if (source_rate == TARGET_SAMPLE_RATE) return input;
    size_t num_input = input.size() / 2;
    const int16_t* in = reinterpret_cast<const int16_t*>(input.data());
    double ratio = static_cast<double>(TARGET_SAMPLE_RATE) / source_rate;
    size_t num_output = static_cast<size_t>(num_input * ratio);
    std::vector<int16_t> out(num_output);
    for (size_t i = 0; i < num_output; i++) {
        double src = i / ratio;
        size_t i0 = static_cast<size_t>(src);
        size_t i1 = std::min(i0 + 1, num_input - 1);
        double frac = src - i0;
        out[i] = static_cast<int16_t>(in[i0] * (1.0 - frac) + in[i1] * frac);
    }
    std::vector<uint8_t> result(num_output * 2);
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

static void audio_callback(void*, Uint8* stream, int len) {
    if (!g_audio.recording) return;
    std::lock_guard<std::mutex> lock(g_audio.mutex);
    g_audio.buffer.insert(g_audio.buffer.end(), stream, stream + len);
}

static size_t wrap_index(const std::string& s, size_t limit) {
    size_t len = 0;
    bool in_esc = false;
    for (size_t i = 0; i < s.length(); ++i) {
        char c = s[i];
        if (c == '\033') { in_esc = true; continue; }
        if (in_esc) { if (c == 'm') in_esc = false; continue; }
        if ((c & 0xC0) != 0x80) len++;
        if (len >= limit && c == ' ') return i;
    }
    return std::string::npos;
}

static int run_live_transcription(cactus_model_t model, const std::string& language) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << "Failed to init SDL audio: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = TARGET_SAMPLE_RATE;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (TARGET_SAMPLE_RATE * AUDIO_BUFFER_MS) / 1000;
    want.callback = audio_callback;
    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (device == 0) {
        std::cerr << "Failed to open microphone: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }
    g_audio.actual_sample_rate = have.freq;

    std::string options = "{\"language\":\"" + language + "\"}";
    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, options.c_str());
    if (!stream) {
        std::cerr << "Failed to start streaming (live mode needs a Whisper or Parakeet TDT model): "
                  << cactus_get_last_error() << "\n";
        SDL_CloseAudioDevice(device);
        SDL_Quit();
        return 1;
    }

    std::cout << colored("Listening…", ansi::yellow) << " press " << colored("Enter", ansi::cyan) << " to stop\n";
    print_rule();
    std::cout << "\n";

    g_audio.buffer.clear();
    g_audio.recording = true;
    SDL_PauseAudioDevice(device, 0);

    std::atomic<bool> should_stop{false};
    std::thread input_thread([&should_stop]() {
        std::string line;
        std::getline(std::cin, line);
        should_stop = true;
    });

    std::string confirmed_text;
    std::string current_line;
    int pending_lines = 0;
    std::string stats;
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);
    auto last_process = std::chrono::steady_clock::now();

    auto step = [&](std::vector<uint8_t> chunk, bool render) {
        if (chunk.empty()) return;
        std::vector<uint8_t> pcm = resample_to_16k(chunk, g_audio.actual_sample_rate);
        response_buffer[0] = '\0';
        auto t0 = std::chrono::steady_clock::now();
        if (cactus_stream_transcribe_process(stream, pcm.data(), pcm.size(),
                                             response_buffer.data(), response_buffer.size()) < 0) return;

        std::string json(response_buffer.data());
        std::string confirmed = json_string_value(json, "confirmed");
        std::string pending = json_string_value(json, "pending");
        confirmed_text += confirmed;
        if (!render) return;

        double latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!confirmed.empty() || !pending.empty()) {
            std::ostringstream st;
            st << "[latency: " << (int)latency_ms << "ms]";
            stats = colored(st.str(), ansi::dim);
        }

        int limit = (int)(terminal_width() * 0.7);

        if (pending_lines > 0) {
            std::cout << "\r\033[2K";
            for (int i = 0; i < pending_lines; ++i) std::cout << "\033[1A\033[2K";
        } else {
            std::cout << "\r";
        }

        if (!confirmed.empty()) current_line += colored(confirmed, ansi::green);

        while (true) {
            size_t idx = wrap_index(current_line, (size_t)limit);
            if (idx == std::string::npos) break;
            std::cout << "\r\033[K" << current_line.substr(0, idx) << ansi::reset << "\n";
            current_line = std::string(ansi::green) + current_line.substr(idx + 1);
        }
        std::cout << "\r\033[K" << current_line;

        std::string ghost = stats;
        if (!pending.empty()) {
            if (!ghost.empty()) ghost += "\n";
            ghost += colored("[pending] " + pending, ansi::yellow);
        }
        pending_lines = 0;
        if (!ghost.empty()) {
            std::cout << "\n";
            std::stringstream ss(ghost);
            std::string line;
            bool first = true;
            while (std::getline(ss, line)) {
                while (true) {
                    size_t idx = wrap_index(line, (size_t)limit);
                    if (idx == std::string::npos) break;
                    if (!first) std::cout << "\n";
                    std::cout << line.substr(0, idx);
                    line = line.substr(idx + 1);
                    pending_lines++;
                    first = false;
                }
                if (!first) std::cout << "\n";
                std::cout << line;
                pending_lines++;
                first = false;
            }
        }
        std::cout << std::flush;
    };

    while (!should_stop) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_process >= std::chrono::milliseconds(PROCESS_INTERVAL_MS)) {
            last_process = now;
            std::vector<uint8_t> chunk;
            { std::lock_guard<std::mutex> lock(g_audio.mutex); chunk.swap(g_audio.buffer); }
            step(std::move(chunk), true);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    g_audio.recording = false;
    SDL_PauseAudioDevice(device, 1);
    std::vector<uint8_t> tail;
    { std::lock_guard<std::mutex> lock(g_audio.mutex); tail.swap(g_audio.buffer); }
    step(std::move(tail), false);

    response_buffer[0] = '\0';
    cactus_stream_transcribe_stop(stream, response_buffer.data(), response_buffer.size());
    confirmed_text += json_string_value(std::string(response_buffer.data()), "confirmed");

    std::cout << "\n\n";
    print_rule();
    std::cout << colored("Final transcript:", ansi::bold) << "\n" << confirmed_text << "\n";
    print_rule();

    if (input_thread.joinable()) input_thread.detach();
    SDL_CloseAudioDevice(device);
    SDL_Quit();
    return 0;
}

#else

static int run_live_transcription(cactus_model_t, const std::string&) {
    std::cerr << "Live microphone transcription is unavailable in this build.\n"
                 "Transcribe an audio file instead:  cactus transcribe <model> --file audio.wav\n";
    return 1;
}

#endif

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <model_path> [audio_file] [--language <code>]\n"
              << "  With an audio file: one-shot transcription. Without: live from the microphone.\n";
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;

    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string model_path = argv[1];
    std::string audio_file;
    std::string language = "en";
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--language") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --language requires a value\n";
                return 1;
            }
            language = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            print_usage(argv[0]);
            return 1;
        } else {
            audio_file = arg;
        }
    }

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path.c_str(), nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model: " << cactus_get_last_error() << "\n";
        return 1;
    }

    if (g_color) std::cout << "\033[2J\033[3J\033[H";
    print_banner();

    int result;
    if (!audio_file.empty()) {
        std::cout << colored("Transcribing ", ansi::bold) << audio_file << "\n";
        print_rule();
        std::cout << "\n";
        result = transcribe_file(model, audio_file, model_path, language);
    } else {
        result = run_live_transcription(model, language);
    }

    cactus_destroy(model);
    std::cout << "\nGoodbye.\n";
    return result >= 0 ? 0 : 1;
}
