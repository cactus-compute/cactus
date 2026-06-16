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
#include <deque>
#include <cctype>
#include <algorithm>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <atomic>
#include <mutex>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

constexpr size_t RESPONSE_BUFFER_SIZE = 65536;

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

static bool supports_color() {
    const char* term = std::getenv("TERM");
    return term && std::string(term) != "dumb";
}
static bool use_colors = supports_color();

static std::string colored(const std::string& text, const std::string& color) {
    return use_colors ? color + text + Color::RESET : text;
}

static void print_separator(char ch = '-', int width = 60) {
    std::cout << colored(std::string(width, ch), Color::DIM) << "\n";
}

static void print_header_live_mode() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("     🌵 CACTUS LIVE TRANSCRIPTION 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Listening...", Color::YELLOW) << " Press " << colored("Enter", Color::CYAN) << " to stop\n";
    print_separator();
    std::cout << "\n";
}

static std::string extract_json_value(const std::string& json, const std::string& key) {
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

static std::string extract_json_number(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t start = json.find(pattern);
    if (start == std::string::npos) return "";
    start += pattern.length();
    while (start < json.length() && std::isspace((unsigned char)json[start])) start++;
    const char* begin = json.c_str() + start;
    char* end_ptr = nullptr;
    std::strtod(begin, &end_ptr);
    if (end_ptr == begin) return "";
    return std::string(begin, static_cast<size_t>(end_ptr - begin));
}

static std::string get_transcribe_prompt(const std::string& model_path, const std::string& language) {
    std::string p = model_path;
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (p.find("whisper") != std::string::npos)
        return "<|startoftranscript|><|" + language + "|><|transcribe|>";
    return "";
}

static void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    std::cout << token << std::flush;
}

static std::string transcribe_options_json(const std::string& language) {
    std::ostringstream os;
    os << "{\"max_tokens\":500,\"language\":\"" << language << "\"}";
    return os.str();
}

static int transcribe_file(cactus_model_t model, const std::string& audio_path,
                           const std::string& model_path, const std::string& language) {
    std::string prompt = get_transcribe_prompt(model_path, language);
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);
    const std::string options_json = transcribe_options_json(language);

    auto start_time = std::chrono::steady_clock::now();
    int result = cactus_transcribe(
        model, audio_path.c_str(), prompt.empty() ? nullptr : prompt.c_str(),
        response_buffer.data(), response_buffer.size(),
        options_json.c_str(), print_token, nullptr, nullptr, 0);
    auto end_time = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    if (result < 0) {
        std::cerr << "\n" << colored("Error: ", Color::RED + Color::BOLD) << "Transcription failed\n";
        const char* err = cactus_get_last_error();
        if (err && *err) std::cerr << colored("Details: ", Color::RED) << err << "\n";
        return -1;
    }

    std::string json_str(response_buffer.data());
    std::string time_str = extract_json_number(json_str, "total_time_ms");
    std::ostringstream stats;
    stats << std::fixed << std::setprecision(2) << "\n\n" << colored("[", Color::GRAY)
          << colored("processed in: ", Color::GRAY) << total_seconds << "s";
    if (!time_str.empty())
        stats << colored(" | model time: ", Color::GRAY) << std::stod(time_str) / 1000.0 << "s";
    stats << colored("]", Color::GRAY);
    std::cout << stats.str() << "\n";
    return 0;
}

#ifdef HAVE_SDL2

constexpr int TARGET_SAMPLE_RATE = 16000;
constexpr int AUDIO_BUFFER_MS = 100;

struct AudioState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
    int actual_sample_rate{TARGET_SAMPLE_RATE};
};
static AudioState g_audio_state;

static std::vector<uint8_t> resample_audio(const std::vector<uint8_t>& input, int source_rate, int target_rate) {
    if (source_rate == target_rate || input.empty()) return input;
    size_t num_input = input.size() / 2;
    if (num_input == 0) return input;
    const int16_t* in = reinterpret_cast<const int16_t*>(input.data());
    double ratio = static_cast<double>(target_rate) / source_rate;
    size_t num_output = static_cast<size_t>(num_input * ratio);
    if (num_output == 0) return {};
    std::vector<int16_t> out(num_output);
    for (size_t i = 0; i < num_output; i++) {
        double src = i / ratio;
        size_t i0 = static_cast<size_t>(src);
        size_t i1 = std::min(i0 + 1, num_input - 1);
        double frac = src - i0;
        double s = in[i0] * (1.0 - frac) + in[i1] * frac;
        out[i] = static_cast<int16_t>(std::clamp(s, -32768.0, 32767.0));
    }
    std::vector<uint8_t> result(num_output * 2);
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

static void audio_callback(void* /*userdata*/, Uint8* stream, int len) {
    if (!g_audio_state.recording) return;
    std::lock_guard<std::mutex> lock(g_audio_state.mutex);
    g_audio_state.buffer.insert(g_audio_state.buffer.end(), stream, stream + len);
}

static int get_terminal_width() {
    struct winsize w;
    return (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) ? 80 : w.ws_col;
}

static size_t find_safe_split_index(const std::string& s, size_t limit) {
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

static int run_live_transcription(cactus_model_t model, const std::string& model_path, const std::string& language) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Failed to init SDL: " << SDL_GetError() << "\n";
        return 1;
    }
    if (SDL_GetNumAudioDevices(1) == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "No audio capture devices found\n";
        SDL_Quit();
        return 1;
    }
    std::cout << colored("Available microphones:", Color::YELLOW) << "\n";
    for (int i = 0; i < SDL_GetNumAudioDevices(1); i++)
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, 1) << "\n";
    std::cout << "\n";

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = TARGET_SAMPLE_RATE;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (TARGET_SAMPLE_RATE * AUDIO_BUFFER_MS) / 1000;
    want.callback = audio_callback;
    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (device == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Failed to open microphone: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }
    g_audio_state.actual_sample_rate = have.freq;
    if (have.freq != TARGET_SAMPLE_RATE)
        std::cout << colored("Note: ", Color::YELLOW) << "device runs at " << have.freq
                  << "Hz, resampling to " << TARGET_SAMPLE_RATE << "Hz\n";

    const std::string options = "{\"language\":\"" + language + "\"}";
    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, options.c_str());
    if (!stream) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to start streaming transcription (live mode needs a Whisper or Parakeet TDT model).\n";
        const char* err = cactus_get_last_error();
        if (err && *err) std::cerr << "  " << err << "\n";
        SDL_CloseAudioDevice(device);
        SDL_Quit();
        return 1;
    }

    print_header_live_mode();
    g_audio_state.buffer.clear();
    g_audio_state.recording = true;
    SDL_PauseAudioDevice(device, 0);

    std::atomic<bool> should_stop{false};
    std::thread input_thread([&should_stop]() {
        std::string line;
        std::getline(std::cin, line);
        should_stop = true;
    });

    std::string confirmed_text;          // plain final transcript
    std::string current_line_confirmed;  // colored, current wrapped line
    int last_pending_line_count = 0;
    std::string last_stats;
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

    auto last_process = std::chrono::steady_clock::now();
    const auto interval = std::chrono::milliseconds(250);

    auto step = [&](std::vector<uint8_t> chunk) {
        if (chunk.empty()) return;
        std::vector<uint8_t> pcm = resample_audio(chunk, g_audio_state.actual_sample_rate, TARGET_SAMPLE_RATE);
        response_buffer[0] = '\0';
        auto t0 = std::chrono::high_resolution_clock::now();
        int rc = cactus_stream_transcribe_process(stream, pcm.data(), pcm.size(),
                                                  response_buffer.data(), response_buffer.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rc < 0) return;
        double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

        std::string json_str(response_buffer.data());
        std::string confirmed = extract_json_value(json_str, "confirmed");
        std::string pending = extract_json_value(json_str, "pending");
        std::string decode_tps = extract_json_number(json_str, "decode_tps");
        std::string raw_decoder_tps = extract_json_number(json_str, "raw_decoder_tps");
        if (!confirmed.empty() || !pending.empty()) {
            last_stats = colored("[Latency:" + std::to_string((int)latency_ms) + "ms", Color::GRAY);
            if (!decode_tps.empty())
                last_stats += colored(" Decode speed:" + decode_tps + " tokens/sec", Color::GRAY);
            try {
                if (!raw_decoder_tps.empty() && std::stod(raw_decoder_tps) > 0.0)
                    last_stats += colored(" Raw decoder:" + raw_decoder_tps + " tokens/sec", Color::GRAY);
            } catch (...) {}
            last_stats += colored("]", Color::GRAY);
        }

        int width = get_terminal_width();
        int limit = (int)((width < 20 ? 80 : width) * 0.7);

        if (last_pending_line_count > 0) {
            std::cout << "\r\033[2K";
            for (int i = 0; i < last_pending_line_count; ++i) std::cout << "\033[1A\033[2K";
        } else {
            std::cout << "\r";
        }

        if (!confirmed.empty()) {
            current_line_confirmed += colored(confirmed, Color::GREEN);
            confirmed_text += confirmed;
        }

        while (true) {
            size_t idx = find_safe_split_index(current_line_confirmed, (size_t)limit);
            if (idx == std::string::npos) break;
            std::cout << "\r\033[K" << current_line_confirmed.substr(0, idx) << Color::RESET << "\n";
            current_line_confirmed = Color::GREEN + current_line_confirmed.substr(idx + 1);
        }
        std::cout << "\r\033[K" << current_line_confirmed;

        std::string ghost = last_stats;
        if (!pending.empty()) {
            if (!ghost.empty()) ghost += "\n";
            ghost += colored("[pending] ", Color::YELLOW) + colored(pending, Color::YELLOW);
        }
        last_pending_line_count = 0;
        if (!ghost.empty()) {
            std::cout << "\n";
            std::stringstream ss(ghost);
            std::string line;
            bool first = true;
            while (std::getline(ss, line)) {
                while (true) {
                    size_t idx = find_safe_split_index(line, (size_t)limit);
                    if (idx == std::string::npos) break;
                    if (!first) std::cout << "\n";
                    std::cout << line.substr(0, idx);
                    line = line.substr(idx + 1);
                    last_pending_line_count++;
                    first = false;
                }
                if (!first) std::cout << "\n";
                std::cout << line;
                last_pending_line_count++;
                first = false;
            }
        }
        std::cout << std::flush;
    };

    while (!should_stop) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_process >= interval) {
            last_process = now;
            std::vector<uint8_t> chunk;
            { std::lock_guard<std::mutex> lock(g_audio_state.mutex); chunk.swap(g_audio_state.buffer); }
            step(std::move(chunk));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    g_audio_state.recording = false;
    SDL_PauseAudioDevice(device, 1);
    std::vector<uint8_t> tail;
    { std::lock_guard<std::mutex> lock(g_audio_state.mutex); tail.swap(g_audio_state.buffer); }
    step(std::move(tail));

    response_buffer[0] = '\0';
    cactus_stream_transcribe_stop(stream, response_buffer.data(), response_buffer.size());
    confirmed_text += extract_json_value(std::string(response_buffer.data()), "confirmed");

    std::cout << "\n\n";
    print_separator();
    std::cout << colored("Final transcript:", Color::GREEN + Color::BOLD) << "\n";
    std::cout << confirmed_text << "\n";
    print_separator();

    if (input_thread.joinable()) input_thread.detach();
    SDL_CloseAudioDevice(device);
    SDL_Quit();
    return 0;
}

#else  // HAVE_SDL2

static int run_live_transcription(cactus_model_t, const std::string&, const std::string&) {
    std::cerr << colored("Error: ", Color::RED + Color::BOLD)
              << "Live microphone transcription requires SDL2.\n"
              << "Install SDL2 and rebuild, or pass an audio file:\n"
              << "  macOS:  brew install sdl2\n"
              << "  Linux:  sudo apt-get install libsdl2-dev\n";
    return 1;
}

#endif  // HAVE_SDL2

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Missing model path\n";
        std::cerr << "Usage: " << argv[0] << " <model_path> [audio_file] [--language <code>]\n"
                  << "  With an audio file: one-shot transcription. Without: live from the microphone.\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* audio_file = nullptr;
    std::string language = "en";
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--language" && i + 1 < argc) language = argv[++i];
        else if (argv[i][0] != '-') audio_file = argv[i];
    }

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";
    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        const char* err = cactus_get_last_error();
        if (err && *err) std::cerr << colored("Error: ", Color::RED) << err << "\n";
        return 1;
    }
    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    int result;
    if (audio_file) {
        std::cout << "\n" << colored("Transcribing: ", Color::BLUE + Color::BOLD) << audio_file << "\n\n";
        result = transcribe_file(model, audio_file, model_path, language);
    } else {
        result = run_live_transcription(model, model_path, language);
    }

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return result >= 0 ? 0 : 1;
}
