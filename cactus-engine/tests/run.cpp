#include "../cactus_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/ioctl.h>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <SDL_audio.h>
#endif

namespace {

constexpr int kMaxTokens = 1024;
constexpr size_t kResponseBufferSize = kMaxTokens * 128;
constexpr int kAudioSampleRate = 16000;

#ifdef HAVE_SDL2
constexpr int kRecordSampleRate = kAudioSampleRate;

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

namespace ansi {
constexpr const char* reset     = "\033[0m";
constexpr const char* bold      = "\033[1m";
constexpr const char* dim       = "\033[2m";
constexpr const char* italic    = "\033[3m";
constexpr const char* underline = "\033[4m";
constexpr const char* cyan      = "\033[36m";
constexpr const char* green     = "\033[32m";
constexpr const char* yellow    = "\033[33m";
constexpr const char* blue      = "\033[34m";
}  // namespace ansi

bool stdout_is_terminal() {
    return isatty(STDOUT_FILENO) != 0 && std::getenv("NO_COLOR") == nullptr;
}

int terminal_width() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    if (const char* cols = std::getenv("COLUMNS")) {
        int c = std::atoi(cols);
        if (c > 0) return c;
    }
    return 80;
}

void print_turn_separator() {
    std::cout << "\n";
    if (stdout_is_terminal()) {
        std::string rule;
        for (int i = 0, w = terminal_width(); i < w; ++i) rule += "─";
        std::cout << ansi::dim << rule << ansi::reset << "\n";
    }
}

std::atomic<bool> g_cloud_active{false};

void print_banner() {
    if (!stdout_is_terminal()) {
        std::cout << "Cactus Hybrid Chat\n\n";
        return;
    }
    std::cout << ansi::bold << ansi::green
              << " ██████╗ █████╗  ██████╗████████╗██╗   ██╗███████╗\n"
              << "██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔════╝\n"
              << "██║     ███████║██║        ██║   ██║   ██║███████╗\n"
              << "██║     ██╔══██║██║        ██║   ██║   ██║╚════██║\n"
              << "╚██████╗██║  ██║╚██████╗   ██║   ╚██████╔╝███████║\n"
              << " ╚═════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚══════╝\n"
              << ansi::reset
              << ansi::dim << "              Hybrid Chat" << ansi::reset << "\n\n";
}

struct TokenPrinter {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point first;
    bool saw_first = false;
    int count = 0;
    std::string pending;
    bool in_code_fence = false;
    bool color = false;
    bool suppress_thinking_stream = false;
    bool show_thinking = false;
    std::string stream_carry;
    bool label_printed = false;
    std::thread spinner_thread;
    std::atomic<bool> spinner_running{false};

    ~TokenPrinter() { stop_thinking(); }

    void reset() {
        start = std::chrono::steady_clock::now();
        saw_first = false;
        count = 0;
        pending.clear();
        in_code_fence = false;
        label_printed = false;
        color = stdout_is_terminal();
        suppress_thinking_stream = false;
        stream_carry.clear();
    }

    void start_thinking() {
        if (!color || spinner_running.load()) return;
        spinner_running = true;
        spinner_thread = std::thread([this]() {
            const char* frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            for (int idx = 0; spinner_running.load(); ++idx) {
                std::cout << "\r" << ansi::dim << frames[idx % 10] << " thinking…" << ansi::reset << std::flush;
                for (int k = 0; k < 5 && spinner_running.load(); ++k) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(16));
                }
            }
        });
    }

    void stop_thinking() {
        if (spinner_running.exchange(false)) {
            if (spinner_thread.joinable()) spinner_thread.join();
            std::cout << "\r\033[K" << std::flush;  
        }
    }

    void print_label() {
        if (label_printed) return;
        label_printed = true;
        stop_thinking();
        const char* origin = g_cloud_active.load() ? "cloud" : "local";
        if (color) {
            const char* origin_color = g_cloud_active.load() ? ansi::yellow : ansi::cyan;
            std::cout << ansi::bold << ansi::green << "Assistant" << ansi::reset
                      << " " << origin_color << "(" << origin << ")" << ansi::reset
                      << ansi::bold << ansi::green << ":" << ansi::reset << " ";
        } else {
            std::cout << "Assistant (" << origin << "): ";
        }
        std::cout << std::flush;
    }

    // Render the markdown inline spans (**bold**, *italic*, `code`) within one line.
    void render_inline(const std::string& s) const {
        if (!color) { std::cout << s; return; }
        size_t i = 0, n = s.size();
        while (i < n) {
            char c = s[i];
            if (c == '*' && i + 1 < n && s[i + 1] == '*') {
                size_t end = s.find("**", i + 2);
                if (end != std::string::npos && end > i + 2) {
                    std::cout << ansi::bold << s.substr(i + 2, end - (i + 2)) << ansi::reset;
                    i = end + 2;
                    continue;
                }
            }
            if (c == '`') {
                size_t end = s.find('`', i + 1);
                if (end != std::string::npos) {
                    std::cout << ansi::cyan << s.substr(i + 1, end - (i + 1)) << ansi::reset;
                    i = end + 1;
                    continue;
                }
            }
            
            if (c == '[') {
                size_t rb = s.find(']', i + 1);
                if (rb != std::string::npos && rb + 1 < n && s[rb + 1] == '(') {
                    size_t rp = s.find(')', rb + 2);
                    if (rp != std::string::npos) {
                        std::cout << ansi::underline << ansi::cyan
                                  << s.substr(i + 1, rb - (i + 1)) << ansi::reset;
                        i = rp + 1;
                        continue;
                    }
                }
            }
            if (c == '*' && i + 1 < n && s[i + 1] != '*' && s[i + 1] != ' ') {
                size_t end = s.find('*', i + 1);
                if (end != std::string::npos && s[end - 1] != ' ') {
                    std::cout << ansi::italic << s.substr(i + 1, end - (i + 1)) << ansi::reset;
                    i = end + 1;
                    continue;
                }
            }
            std::cout << c;
            ++i;
        }
    }

    // Render a single complete line, handling block-level markdown (headings, lists, code fences).
    void render_line(const std::string& raw) {
        if (!color) { std::cout << raw; return; }

        size_t s0 = raw.find_first_not_of(" \t");
        std::string lead = (s0 == std::string::npos) ? raw : raw.substr(0, s0);
        std::string body = (s0 == std::string::npos) ? "" : raw.substr(s0);

        if (body.rfind("```", 0) == 0) {
            if (!in_code_fence) {
                in_code_fence = true;
                std::string lang = body.substr(3);
                size_t a = lang.find_first_not_of(" \t");
                size_t b = lang.find_last_not_of(" \t");
                lang = (a == std::string::npos) ? "" : lang.substr(a, b - a + 1);
                std::cout << ansi::dim << "  ┌── " << (lang.empty() ? "code" : lang) << ansi::reset;
            } else {
                in_code_fence = false;
                std::cout << ansi::dim << "  └──" << ansi::reset;
            }
            return;
        }
        if (in_code_fence) {
            std::cout << ansi::dim << "  │ " << ansi::reset << ansi::green << raw << ansi::reset;
            return;
        }

        if (!body.empty() && body[0] == '#') {
            size_t h = 0;
            while (h < body.size() && body[h] == '#') ++h;
            if (h >= 1 && h <= 6 && h < body.size() && body[h] == ' ') {
                std::cout << lead << ansi::bold << ansi::cyan << body.substr(h + 1) << ansi::reset;
                return;
            }
        }

        if (body[0] == '-' || body[0] == '*' || body[0] == '_') {
            char hc = body[0];
            size_t marks = 0;
            bool only = true;
            for (char ch : body) {
                if (ch == hc) ++marks;
                else if (ch != ' ') { only = false; break; }
            }
            if (only && marks >= 3) {
                std::cout << lead << ansi::dim
                          << "────────────────────────────"
                          << ansi::reset;
                return;
            }
        }

        if (body[0] == '>') {
            std::string rest = body.substr(1);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            std::cout << lead << ansi::dim << "▏ " << ansi::reset;
            render_inline(rest);
            return;
        }

        // Bullet list: -, *, + then space(s)
        if (body.size() >= 2 && (body[0] == '-' || body[0] == '*' || body[0] == '+') && body[1] == ' ') {
            std::string rest = body.substr(2);
            size_t r0 = rest.find_first_not_of(" \t");
            rest = (r0 == std::string::npos) ? "" : rest.substr(r0);
            std::cout << lead << ansi::yellow << "• " << ansi::reset;
            render_inline(rest);
            return;
        }

        size_t d = 0;
        while (d < body.size() && std::isdigit(static_cast<unsigned char>(body[d]))) ++d;
        if (d > 0 && d + 1 < body.size() && body[d] == '.' && body[d + 1] == ' ') {
            std::string rest = body.substr(d + 1);
            size_t r0 = rest.find_first_not_of(" \t");
            rest = (r0 == std::string::npos) ? "" : rest.substr(r0);
            std::cout << lead << ansi::yellow << body.substr(0, d + 1) << ansi::reset << " ";
            render_inline(rest);
            return;
        }

        std::cout << lead;
        render_inline(body);
    }

    void on_token(const char* text) {
        if (!saw_first) {
            first = std::chrono::steady_clock::now();
            saw_first = true;
        }
        ++count;
        if (!text) return;
        feed(text);
        std::cout << std::flush;
    }

    void feed(const std::string& chunk) {
        static const std::vector<std::string> markers = {
            "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>", "<think|>",
        };
        stream_carry += chunk;
        size_t i = 0;
        while (i < stream_carry.size()) {
            const std::string* matched = nullptr;
            bool maybe_partial = false;
            for (const auto& m : markers) {
                if (stream_carry.compare(i, m.size(), m) == 0) { matched = &m; break; }
                size_t avail = stream_carry.size() - i;
                if (avail < m.size() && stream_carry.compare(i, avail, m, 0, avail) == 0) maybe_partial = true;
            }
            if (matched) {
                if (*matched == "<|channel>") { suppress_thinking_stream = true; if (!show_thinking) start_thinking(); }
                else if (*matched == "<channel|>") { suppress_thinking_stream = false; stop_thinking(); }
                i += matched->size();
                continue;
            }
            if (maybe_partial) break;
            if (!suppress_thinking_stream || show_thinking) emit_visible(stream_carry[i]);
            ++i;
        }
        stream_carry.erase(0, i);
    }

    void emit_visible(char c) {
        print_label();
        pending += c;
        size_t nl;
        while ((nl = pending.find('\n')) != std::string::npos) {
            render_line(pending.substr(0, nl));
            std::cout << "\n";
            pending.erase(0, nl + 1);
        }
    }

    // Flush the trailing partial line once generation completes (no trailing newline).
    void finish() {
        stop_thinking();
        if (!stream_carry.empty() && (!suppress_thinking_stream || show_thinking)) {
            for (char c : stream_carry) emit_visible(c);
        }
        stream_carry.clear();
        print_label();
        if (!pending.empty()) {
            render_line(pending);
            pending.clear();
        }
        if (color && in_code_fence) std::cout << ansi::reset;
        in_code_fence = false;
        std::cout << std::flush;
    }

    void print_stats(double ram_mb, double confidence, bool cloud_handoff, double threshold,
                     int reported_tokens = -1, double reported_decode_tps = -1.0,
                     double reported_ttft_ms = -1.0, double reported_total_ms = -1.0,
                     const std::string& cloud_reason = "") const {
        auto end = std::chrono::steady_clock::now();
        double total_s = std::chrono::duration<double>(end - start).count();
        double ttft_s = saw_first ? std::chrono::duration<double>(first - start).count() : 0.0;
        double decode_s = saw_first ? std::chrono::duration<double>(end - first).count() : total_s;
        int display_tokens = reported_tokens >= 0 ? reported_tokens : count;
        double tps = (count > 1 && decode_s > 0.0) ? (count - 1) / decode_s : (total_s > 0.0 ? count / total_s : 0.0);
        if (reported_decode_tps >= 0.0) tps = reported_decode_tps;
        if (reported_ttft_ms >= 0.0) ttft_s = reported_ttft_ms / 1000.0;
        if (reported_total_ms >= 0.0) total_s = reported_total_ms / 1000.0;
        std::cout << "\n";
        if (color) std::cout << ansi::dim;
        std::cout << "[" << display_tokens << " tokens | latency: "
                  << std::fixed << std::setprecision(3) << ttft_s
                  << "s | total: " << total_s
                  << "s | " << std::setprecision(1) << tps << " tok/s";
        if (confidence >= 0.0) {
            std::cout << " | confidence: " << std::max(0.0, std::min(100.0, confidence * 100.0)) << "%";
        } else {
            std::cout << " | confidence: n/a";
        }
        if (threshold >= 0.0) {
            std::cout << " | threshold: " << std::max(0.0, std::min(100.0, threshold * 100.0)) << "%";
        }
        std::cout << " | cloud: " << (cloud_handoff ? "yes" : "no");
        if (!cloud_reason.empty()) {
            std::cout << " (" << cloud_reason << ")";
        }
        if (ram_mb > 0.0) {
            std::cout << " | RAM: " << ram_mb << " MB";
        }
        std::cout << "]";
        if (color) std::cout << ansi::reset;
        std::cout << "\n";
    }
};

TokenPrinter* g_printer = nullptr;

void token_callback(const char* text, uint32_t, void*) {
    if (g_printer) {
        g_printer->on_token(text);
    }
}

void log_callback(int level, const char* component, const char* message, void*) {
    const std::string comp = component ? component : "";
    const std::string msg = message ? message : "";
    if (comp == "cloud_handoff") {
        if (msg.find("triggered") != std::string::npos) {
            g_cloud_active = true;
        } else if (msg.find("falling back") != std::string::npos ||
                   msg.find("failed") != std::string::npos ||
                   msg.find("disabling") != std::string::npos) {
            g_cloud_active = false;
        }
        return;
    }
    const bool color = isatty(STDERR_FILENO) != 0 && std::getenv("NO_COLOR") == nullptr;
    const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    const char* name = (level >= 0 && level <= 3) ? names[level] : "LOG";
    std::cerr << "\n";
    if (color) std::cerr << ansi::dim;
    std::cerr << "[" << name << "] " << comp << ": " << msg;
    if (color) std::cerr << ansi::reset;
    std::cerr << "\n";
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

std::vector<uint32_t> parse_token_ids(const std::string& text) {
    std::vector<uint32_t> tokens;
    std::string normalized = text;
    for (char& ch : normalized) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') ch = ',';
    }
    std::stringstream ss(normalized);
    std::string part;
    while (std::getline(ss, part, ',')) {
        size_t start = part.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = part.find_last_not_of(" \t\r\n");
        std::string value = part.substr(start, end - start + 1);
        char* parsed_end = nullptr;
        unsigned long parsed = std::strtoul(value.c_str(), &parsed_end, 10);
        if (!parsed_end || *parsed_end != '\0') {
            throw std::runtime_error("Invalid token id: " + value);
        }
        tokens.push_back(static_cast<uint32_t>(parsed));
    }
    return tokens;
}

bool read_text_file(const std::string& path, std::string& text) {
    std::ifstream in(path);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (!in.good() && !in.eof()) return false;
    text = buffer.str();
    return true;
}

bool write_text_file(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    if (!out) return false;
    out << text;
    return out.good();
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

double json_number_value(const std::string& json, const std::string& key, double fallback = 0.0) {
    std::string needle = "\"" + key + "\":";
    size_t start = json.find(needle);
    if (start == std::string::npos) return fallback;
    start += needle.size();
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) {
        ++start;
    }
    if (json.compare(start, 4, "null") == 0) return fallback;
    char* end = nullptr;
    return std::strtod(json.c_str() + start, &end);
}

bool json_bool_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t start = json.find(needle);
    if (start == std::string::npos) return false;
    start += needle.size();
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) {
        ++start;
    }
    return json.compare(start, 4, "true") == 0;
}

struct ChatTurn {
    std::string role;
    std::string content;
    std::string image;
    std::string audio;
};

std::string build_messages(const std::string& system_prompt,
                           const std::vector<ChatTurn>& history) {
    std::ostringstream msg;
    msg << "[";
    bool need_comma = false;
    if (!system_prompt.empty()) {
        msg << "{\"role\":\"system\",\"content\":\"" << escape_json(system_prompt) << "\"}";
        need_comma = true;
    }
    for (const auto& turn : history) {
        if (need_comma) msg << ",";
        need_comma = true;
        msg << "{\"role\":\"" << turn.role << "\",\"content\":\""
            << escape_json(turn.content) << "\"";
        if (!turn.image.empty()) msg << ",\"images\":[\"" << escape_json(turn.image) << "\"]";
        if (!turn.audio.empty()) msg << ",\"audio\":[\"" << escape_json(turn.audio) << "\"]";
        msg << "}";
    }
    msg << "]";
    return msg.str();
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <model_path> [--backend cpu|metal] [--system <prompt>] [--image <path>] [--audio <path>]"
              << " [--prompt <text>] [--input-ids <ids>] [--input-ids-file <path>] [--max-new-tokens <n>]"
              << " [--result-json <path>] [--thinking] [--no-cloud-handoff]"
              << " [--confidence-threshold <value>] [--cloud-timeout-ms <ms>] [-h|--help]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string system_prompt;
    std::string current_image;
    std::string current_audio;
    std::string initial_prompt;
    std::string input_ids;
    std::string input_ids_file;
    std::string result_json;
    int max_new_tokens = kMaxTokens;
    bool thinking = false;
    bool auto_handoff = true;
    double confidence_threshold = -1.0;
    int cloud_timeout_ms = 15000;
    std::string backend;

    int i = 2;
    auto need_value = [&](const char* flag) -> bool {
        if (i + 1 >= argc) {
            std::cerr << "Error: " << flag << " requires a value\n";
            print_usage(argv[0]);
            return false;
        }
        return true;
    };
    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--system") {
            if (!need_value("--system")) return 1;
            system_prompt = argv[++i];
        } else if (arg == "--image") {
            if (!need_value("--image")) return 1;
            current_image = expand_tilde(argv[++i]);
        } else if (arg == "--audio") {
            if (!need_value("--audio")) return 1;
            current_audio = expand_tilde(argv[++i]);
        } else if (arg == "--prompt") {
            if (!need_value("--prompt")) return 1;
            initial_prompt = argv[++i];
        } else if (arg == "--input-ids") {
            if (!need_value("--input-ids")) return 1;
            input_ids = argv[++i];
        } else if (arg == "--input-ids-file") {
            if (!need_value("--input-ids-file")) return 1;
            input_ids_file = expand_tilde(argv[++i]);
        } else if (arg == "--max-new-tokens") {
            if (!need_value("--max-new-tokens")) return 1;
            max_new_tokens = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--result-json") {
            if (!need_value("--result-json")) return 1;
            result_json = argv[++i];
        } else if (arg == "--thinking") {
            thinking = true;
        } else if (arg == "--no-cloud-handoff") {
            auto_handoff = false;
        } else if (arg == "--confidence-threshold") {
            if (!need_value("--confidence-threshold")) return 1;
            confidence_threshold = std::atof(argv[++i]);
        } else if (arg == "--cloud-timeout-ms") {
            if (!need_value("--cloud-timeout-ms")) return 1;
            cloud_timeout_ms = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--backend") {
            if (!need_value("--backend")) return 1;
            backend = argv[++i];
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            print_usage(argv[0]);
            return 1;
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
    if (!input_ids.empty() && !input_ids_file.empty()) {
        std::cerr << "Use either --input-ids or --input-ids-file, not both\n";
        return 1;
    }
    if (!input_ids_file.empty()) {
        if (!file_exists(input_ids_file)) {
            std::cerr << "Input ids file not found: " << input_ids_file << "\n";
            return 1;
        }
        if (!read_text_file(input_ids_file, input_ids)) {
            std::cerr << "Failed to read input ids file: " << input_ids_file << "\n";
            return 1;
        }
    }

    if (backend == "cpu" || backend == "metal") {
        if (cactus_set_backend(backend.c_str()) == 0)
            std::cout << "Backend: " << (backend == "metal" ? "Metal GPU" : "CPU") << "\n";
        else
            std::cout << "Metal not available; using CPU\n";
    } else if (!backend.empty() && backend != "auto") {
        std::cerr << "Error: unknown backend '" << backend << "' (expected 'cpu', 'metal', or 'auto')\n";
        return 1;
    }

    cactus_log_set_callback(log_callback, nullptr);

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path.c_str(), nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }

    if (!input_ids.empty()) {
        std::vector<uint32_t> tokens;
        try {
            tokens = parse_token_ids(input_ids);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            cactus_destroy(model);
            return 1;
        }
        if (tokens.empty()) {
            std::cerr << "--input-ids did not contain any token ids\n";
            cactus_destroy(model);
            return 1;
        }
        std::vector<char> response(kResponseBufferSize, 0);
        int rc = cactus_benchmark_tokens(
            model,
            tokens.data(),
            tokens.size(),
            static_cast<size_t>(max_new_tokens),
            response.data(),
            response.size());
        std::string response_json(response.data());
        if (!result_json.empty() && !write_text_file(result_json, response_json)) {
            std::cerr << "Failed to write result JSON: " << result_json << "\n";
            cactus_destroy(model);
            return 1;
        }
        std::cout << response_json << "\n";
        cactus_destroy(model);
        return rc < 0 ? 1 : 0;
    }

    if (stdout_is_terminal()) std::cout << "\033[2J\033[3J\033[H";
    print_banner();

    {
        const bool tty = stdout_is_terminal();
        const char* d = tty ? ansi::dim : "";
        const char* r = tty ? ansi::reset : "";
        if (!auto_handoff) {
            std::cout << d << "Hybrid handoff off (--no-cloud-handoff): answering fully on-device." << r << "\n\n";
        } else {
            std::cout << d
                      << "Hybrid mode: answers on-device, handing off to the cloud whenever the local\n"
                      << "model's confidence drops below its default threshold. Set your own with\n"
                      << "--confidence-threshold <0-1>, or pass --no-cloud-handoff to stay fully local.\n"
                      << "Each reply's confidence and the active threshold are shown in its stats line." << r << "\n\n";
        }
    }

    std::cout << (stdout_is_terminal() ? "\033[1mCommands:\033[0m\n" : "Commands:\n");
    auto print_command = [](const char* command, const char* description) {
        if (stdout_is_terminal()) {
            std::cout << "  " << ansi::cyan << std::left << std::setw(24) << command << ansi::reset
                      << ansi::dim << description << ansi::reset << "\n";
        } else {
            std::cout << "  " << std::left << std::setw(24) << command << description << "\n";
        }
    };
    print_command("/image <path> [prompt]", "attach an image and chat about it");
    print_command("/audio <path> [prompt]", "attach an audio file as a spoken prompt");
#ifdef HAVE_SDL2
    print_command("/record [prompt]", "record from the mic as a spoken prompt");
#endif
    print_command("/clear", "drop the attached image/audio");
    print_command("reset", "start a new conversation");
    print_command("exit", "quit");
    std::cout << std::right << "\n";

    std::vector<ChatTurn> history;
    std::vector<uint8_t> current_pcm;
    TokenPrinter printer;
    g_printer = &printer;
    printer.show_thinking = thinking;
    bool auto_send = !initial_prompt.empty() || !current_audio.empty() || !current_image.empty();

    while (true) {
        std::string input;
        if (auto_send) {
            auto_send = false;
            if (!initial_prompt.empty()) input = initial_prompt;
            else if (!current_image.empty()) input = "Describe this image.";
            else input = "Respond to the spoken request in this audio.";
            const char* you = stdout_is_terminal() ? "\033[1;34mYou:\033[0m " : "You: ";
            std::cout << you << input << "\n";
        } else {
            const char* you = stdout_is_terminal() ? "\033[1;34mYou:\033[0m " : "You: ";
            std::cout << you << std::flush;
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

        static const std::string kRecordCmd = "/record";
        if (input == kRecordCmd || input.rfind(kRecordCmd + " ", 0) == 0) {
#ifdef HAVE_SDL2
            std::string record_prompt;
            if (input.size() > kRecordCmd.size() + 1) {
                record_prompt = input.substr(kRecordCmd.size() + 1);
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
            input = record_prompt.empty() ? "Respond to the spoken request in this audio." : record_prompt;
#else
            std::cerr << "Recording requires SDL2, but this binary was built without SDL2.\n";
            continue;
#endif
        }
        if (input.empty()) continue;

        bool attach_media = !current_image.empty() || !current_audio.empty() || !current_pcm.empty();
        if (attach_media) {
            cactus_reset(model);
        }
        history.push_back({"user", input, current_image, current_audio});
        std::string messages = build_messages(system_prompt, history);
        std::string options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":"
            + std::to_string(max_new_tokens)
            + ",\"enable_thinking_if_supported\":" + (thinking ? "true" : "false")
            + ",\"auto_handoff\":" + (auto_handoff ? "true" : "false")
            + ",\"confidence_threshold\":" + std::to_string(confidence_threshold)
            + ",\"cloud_timeout_ms\":" + std::to_string(cloud_timeout_ms)
            + ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\",\"<turn|>\"]}";

        if (!current_image.empty()) std::cout << "[image: " << current_image << "]\n";
        if (!current_audio.empty()) std::cout << "[audio: " << current_audio << "]\n";
        if (!current_pcm.empty()) {
            double seconds = static_cast<double>(current_pcm.size() / sizeof(int16_t)) / static_cast<double>(kAudioSampleRate);
            std::cout << "[recorded audio: " << std::fixed << std::setprecision(1) << seconds << "s]\n";
        }
        std::vector<char> response(kResponseBufferSize, 0);
        g_cloud_active = false;
        printer.reset();
        printer.start_thinking();
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
        printer.finish();

        std::string response_json(response.data());
        if (!result_json.empty() && !write_text_file(result_json, response_json)) {
            std::cerr << "Failed to write result JSON: " << result_json << "\n";
        }
        bool cloud_handoff = json_bool_value(response_json, "cloud_handoff");
        double confidence = json_number_value(response_json, "confidence", -1.0);
        double threshold = json_number_value(response_json, "confidence_threshold", -1.0);
        double ram_mb = json_number_value(response_json, "ram_usage_mb");
        int decode_tokens = static_cast<int>(json_number_value(response_json, "decode_tokens", -1.0));
        double decode_tps = json_number_value(response_json, "decode_tps", -1.0);
        double ttft_ms = json_number_value(response_json, "time_to_first_token_ms", -1.0);
        double total_ms = json_number_value(response_json, "total_time_ms", -1.0);
        std::string cloud_reason = json_string_value(response_json, "cloud_handoff_reason");
        printer.print_stats(ram_mb, confidence, cloud_handoff, threshold, decode_tokens, decode_tps, ttft_ms, total_ms, cloud_reason);

        if (rc < 0) {
            std::cout << "Error: " << response.data() << "\n";
            history.pop_back();
            continue;
        }

        std::string assistant = json_string_value(response_json, "context_response");
        if (assistant.empty()) assistant = json_string_value(response_json, "response");
        history.push_back({"assistant", assistant, "", ""});
        current_image.clear();
        current_audio.clear();
        current_pcm.clear();
        print_turn_separator();
    }

    cactus_destroy(model);
    std::cout << "Goodbye.\n";
    return 0;
}
