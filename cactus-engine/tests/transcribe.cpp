#include "../cactus_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_SDL2
#include <SDL.h>
#include <SDL_audio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

static constexpr size_t kResponseBufferSize = 1 << 16;
static constexpr int kSampleRate = 16000;

namespace Color {
const std::string RESET = "\033[0m";
const std::string BOLD = "\033[1m";
const std::string DIM = "\033[2m";
const std::string CYAN = "\033[36m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string MAGENTA = "\033[35m";
const std::string RED = "\033[31m";
}

static const bool g_use_colors = []() {
    const char* term = std::getenv("TERM");
    return term && std::string(term) != "dumb";
}();

static std::string colored(const std::string& text, const std::string& color) {
    return g_use_colors ? color + text + Color::RESET : text;
}

static void print_separator(char ch = '-', int width = 60) {
    std::cout << colored(std::string(width, ch), Color::DIM) << "\n";
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_path> [audio_file.wav] [--language <code>]\n"
              << "  With an audio file: one-shot transcription. Without: live from the microphone.\n";
}

static std::string json_str_field(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":\"";
    size_t p = json.find(pattern);
    if (p == std::string::npos) return "";
    p += pattern.size();
    std::string out;
    while (p < json.size()) {
        char c = json[p++];
        if (c == '\\' && p < json.size()) {
            switch (json[p++]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default:  out += json[p - 1]; break;
            }
        } else if (c == '"') {
            break;
        } else {
            out += c;
        }
    }
    return out;
}

static std::string build_options(const std::string& language) {
    std::ostringstream os;
    os << "{\"language\":\"" << language << "\",\"telemetry_enabled\":false,\"auto_handoff\":false}";
    return os.str();
}

static std::vector<float> resample_to_16k(const std::vector<float>& in, int sr_in) {
    if (sr_in <= 0 || sr_in == kSampleRate || in.empty()) return in;
    const double ratio = (double)kSampleRate / sr_in;
    const size_t in_len = in.size();
    const size_t out_len = (size_t)(in_len * ratio);
    std::vector<float> out(out_len);

    if (sr_in < kSampleRate) {
        for (size_t i = 0; i < out_len; ++i) {
            double pos = i / ratio;
            size_t i0 = (size_t)pos;
            double frac = pos - i0;
            out[i] = (i0 + 1 < in_len) ? (float)((1.0 - frac) * in[i0] + frac * in[i0 + 1]) : in.back();
        }
        return out;
    }

    constexpr int kHalfWindow = 16;
    const double pi = 3.14159265358979323846;
    const double cutoff = ratio;
    for (size_t i = 0; i < out_len; ++i) {
        double center = i / ratio;
        int left = (int)std::ceil(center) - kHalfWindow;
        int right = (int)std::floor(center) + kHalfWindow;
        double sum = 0.0, wsum = 0.0;
        for (int j = left; j <= right; ++j) {
            if (j < 0 || j >= (int)in_len) continue;
            double x = center - j;
            double sinc = (std::fabs(x) < 1e-9) ? cutoff : cutoff * std::sin(pi * x * cutoff) / (pi * x * cutoff);
            double w = sinc * 0.5 * (1.0 - std::cos(2.0 * pi * (x + kHalfWindow) / (2.0 * kHalfWindow)));
            sum += in[j] * w;
            wsum += w;
        }
        out[i] = (wsum > 1e-9) ? (float)(sum / wsum) : 0.0f;
    }
    return out;
}

static std::vector<int16_t> to_pcm16(const std::vector<float>& f) {
    std::vector<int16_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i)
        out[i] = (int16_t)std::lround(std::max(-1.0, std::min(1.0, (double)f[i])) * 32767.0);
    return out;
}

static bool read_wav_16k_mono(const std::string& path, std::vector<int16_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    auto rd_u32 = [&](uint32_t& v) { return fread(&v, 4, 1, f) == 1; };
    auto rd_u16 = [&](uint16_t& v) { return fread(&v, 2, 1, f) == 1; };

    char tag[4];
    uint32_t riff_size;
    if (fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "RIFF", 4) != 0 || !rd_u32(riff_size) ||
        fread(tag, 1, 4, f) != 4 || std::strncmp(tag, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }

    uint16_t format = 0, channels = 1, bits = 16;
    uint32_t rate = kSampleRate;
    bool have_fmt = false;
    std::vector<int16_t> mono;

    while (fread(tag, 1, 4, f) == 4) {
        uint32_t size;
        if (!rd_u32(size)) break;
        if (std::strncmp(tag, "fmt ", 4) == 0) {
            uint16_t align = 0, bps = 16, ch = 1;
            uint32_t sr = kSampleRate, br = 0;
            rd_u16(format); rd_u16(ch); rd_u32(sr); rd_u32(br); rd_u16(align); rd_u16(bps);
            channels = ch ? ch : 1;
            rate = sr;
            bits = bps;
            have_fmt = true;
            if (size > 16) fseek(f, (long)size - 16, SEEK_CUR);
        } else if (std::strncmp(tag, "data", 4) == 0) {
            if (!have_fmt || format != 1 || bits != 16 || channels > 8) { fclose(f); return false; }
            size_t frames = size / (2u * channels);
            std::vector<int16_t> raw(size / 2);
            if (fread(raw.data(), 2, raw.size(), f) != raw.size()) { fclose(f); return false; }
            mono.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                int acc = 0;
                for (uint16_t c = 0; c < channels; ++c) acc += raw[i * channels + c];
                mono[i] = (int16_t)(acc / channels);
            }
            break;
        } else {
            fseek(f, (long)(size + (size & 1)), SEEK_CUR);
        }
    }
    fclose(f);
    if (mono.empty()) return false;

    if (rate == (uint32_t)kSampleRate) {
        out = std::move(mono);
        return true;
    }
    std::vector<float> samples(mono.size());
    for (size_t i = 0; i < mono.size(); ++i) samples[i] = mono[i] / 32768.0f;
    out = to_pcm16(resample_to_16k(samples, (int)rate));
    return true;
}

#ifdef HAVE_SDL2
namespace {

constexpr int kBufferMs = 100;

struct AudioState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
    int sample_rate = kSampleRate;
};
AudioState g_audio;

void audio_callback(void*, Uint8* stream, int len) {
    std::lock_guard<std::mutex> lock(g_audio.mutex);
    if (!g_audio.recording) return;
    g_audio.buffer.insert(g_audio.buffer.end(), stream, stream + len);
}

std::vector<uint8_t> resample_pcm16(const std::vector<uint8_t>& input, int src_rate) {
    if (input.size() < sizeof(int16_t)) return {};
    size_t n = input.size() / sizeof(int16_t);
    const int16_t* in = reinterpret_cast<const int16_t*>(input.data());
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = in[i] / 32768.0f;
    std::vector<int16_t> out = to_pcm16(resample_to_16k(f, src_rate));
    std::vector<uint8_t> result(out.size() * sizeof(int16_t));
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

size_t terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1 || w.ws_col < 20) return 80;
    return w.ws_col;
}

size_t utf8_safe_len(const std::string& s, size_t n) {
    if (n >= s.size()) return s.size();
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
    return n;
}

void print_header() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("     CACTUS LIVE TRANSCRIPTION", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Listening...", Color::YELLOW) << " Press " << colored("Enter", Color::CYAN) << " to stop\n";
    print_separator();
    std::cout << "\n";
}

} // namespace

static int run_live(cactus_model_t model, const std::string& language) {
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
    for (int i = 0; i < SDL_GetNumAudioDevices(1); ++i)
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, 1) << "\n";
    std::cout << "\n";

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = kSampleRate;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (kSampleRate * kBufferMs) / 1000;
    want.callback = audio_callback;

    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (device == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Failed to open microphone: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }
    g_audio.sample_rate = have.freq;
    if (have.freq != kSampleRate)
        std::cout << colored("Note: ", Color::YELLOW) << "device runs at " << have.freq
                  << "Hz, resampling to " << kSampleRate << "Hz\n";

    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(model, build_options(language).c_str());
    if (!stream) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Failed to start streaming transcription\n";
        SDL_CloseAudioDevice(device);
        SDL_Quit();
        return 1;
    }

    print_header();

    { std::lock_guard<std::mutex> lock(g_audio.mutex); g_audio.buffer.clear(); }
    g_audio.recording = true;
    SDL_PauseAudioDevice(device, 0);

    std::atomic<bool> should_stop{false};
    std::thread input_thread([&should_stop]() {
        std::string ignored;
        std::getline(std::cin, ignored);
        should_stop = true;
    });

    std::string transcript;
    std::string display_line;
    std::vector<char> response(kResponseBufferSize, 0);

    auto render = [&](const std::string& confirmed, const std::string& pending) {
        if (!confirmed.empty()) {
            transcript += confirmed + " ";
            display_line += confirmed + " ";
        }
        size_t width = terminal_width();
        std::cout << "\r\033[2K";
        while (display_line.size() >= width) {
            size_t cut = display_line.rfind(' ', width);
            if (cut == std::string::npos || cut == 0) cut = utf8_safe_len(display_line, width);
            if (cut == 0) cut = width;
            std::cout << colored(display_line.substr(0, cut), Color::GREEN) << "\n";
            display_line.erase(0, (cut < display_line.size() && display_line[cut] == ' ') ? cut + 1 : cut);
        }
        std::cout << colored(display_line, Color::GREEN);
        if (!pending.empty() && display_line.size() + 1 < width)
            std::cout << colored(pending.substr(0, utf8_safe_len(pending, width - display_line.size() - 1)), Color::DIM);
        std::cout << std::flush;
    };

    auto process = [&](std::vector<uint8_t> chunk) {
        if (chunk.empty()) return;
        std::vector<uint8_t> pcm = resample_pcm16(chunk, g_audio.sample_rate);
        response[0] = '\0';
        if (cactus_stream_transcribe_process(stream, pcm.data(), pcm.size(), response.data(), response.size()) >= 0)
            render(json_str_field(response.data(), "confirmed"), json_str_field(response.data(), "pending"));
    };

    auto last_process = std::chrono::steady_clock::now();
    const auto interval = std::chrono::milliseconds(500);
    while (!should_stop) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_process >= interval) {
            last_process = now;
            std::vector<uint8_t> chunk;
            { std::lock_guard<std::mutex> lock(g_audio.mutex); chunk.swap(g_audio.buffer); }
            process(std::move(chunk));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    g_audio.recording = false;
    SDL_PauseAudioDevice(device, 1);

    std::vector<uint8_t> tail;
    { std::lock_guard<std::mutex> lock(g_audio.mutex); tail.swap(g_audio.buffer); }
    process(std::move(tail));

    response[0] = '\0';
    cactus_stream_transcribe_stop(stream, response.data(), response.size());
    transcript += json_str_field(response.data(), "confirmed");

    std::cout << "\n\n";
    print_separator();
    std::cout << colored("Final transcript:", Color::GREEN + Color::BOLD) << "\n";
    std::cout << transcript << "\n";
    print_separator();

    if (input_thread.joinable()) input_thread.detach();
    SDL_CloseAudioDevice(device);
    SDL_Quit();
    return 0;
}
#else
static int run_live(cactus_model_t, const std::string&) {
    std::cerr << colored("Error: ", Color::RED + Color::BOLD)
              << "Live microphone transcription requires SDL2.\n"
              << "Install SDL2 and rebuild, or pass an audio file:\n"
              << "  macOS:  brew install sdl2\n"
              << "  Linux:  sudo apt-get install libsdl2-dev\n";
    return 1;
}
#endif

static size_t quietest_split(const std::vector<int16_t>& s, size_t lo, size_t hi) {
    constexpr size_t win = kSampleRate / 10;
    size_t best = hi;
    double best_energy = -1.0;
    for (size_t i = lo; i + win <= hi; i += win / 2) {
        double acc = 0.0;
        for (size_t j = i; j < i + win; ++j) { double v = s[j] / 32768.0; acc += v * v; }
        if (best_energy < 0.0 || acc < best_energy) { best_energy = acc; best = i + win / 2; }
    }
    return best;
}

static std::string transcribe_buffer(cactus_model_t model, const int16_t* pcm, size_t n, const std::string& opts) {
    std::vector<char> response(kResponseBufferSize, 0);
    int rc = cactus_transcribe(model, nullptr, nullptr, response.data(), response.size(), opts.c_str(),
                               nullptr, nullptr, reinterpret_cast<const uint8_t*>(pcm), n * sizeof(int16_t));
    return rc > 0 ? json_str_field(response.data(), "response") : "";
}

static int run_file(cactus_model_t model, const std::string& audio_path, const std::string& language) {
    const std::string opts = build_options(language);

    std::vector<int16_t> pcm;
    std::string transcript;

    if (read_wav_16k_mono(audio_path, pcm) && !pcm.empty()) {
        constexpr size_t kWindow = 28 * kSampleRate;
        constexpr size_t kSearch = 4 * kSampleRate;
        for (size_t pos = 0; pos < pcm.size();) {
            size_t end = std::min(pos + kWindow, pcm.size());
            if (end < pcm.size()) {
                size_t lo = (end > pos + (kWindow - kSearch)) ? end - kSearch : pos + 1;
                end = quietest_split(pcm, lo, end);
            }
            std::string text = transcribe_buffer(model, pcm.data() + pos, end - pos, opts);
            if (!text.empty()) {
                if (!transcript.empty()) transcript += ' ';
                transcript += text;
            }
            pos = end;
        }
    } else {
        std::vector<char> response(kResponseBufferSize, 0);
        int rc = cactus_transcribe(model, audio_path.c_str(), nullptr, response.data(), response.size(),
                                   opts.c_str(), nullptr, nullptr, nullptr, 0);
        if (rc <= 0) {
            std::string err = json_str_field(response.data(), "error");
            std::cerr << "Transcription failed: " << (err.empty() ? response.data() : err.c_str()) << "\n";
            return 1;
        }
        transcript = json_str_field(response.data(), "response");
    }

    if (transcript.empty()) {
        std::cerr << "Transcription failed\n";
        return 1;
    }
    std::cout << transcript << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    std::string model_path = argv[1];
    std::string audio_path;
    std::string language = "en";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--language" && i + 1 < argc) {
            language = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (audio_path.empty() && arg.rfind("-", 0) != 0) {
            audio_path = arg;
        }
    }

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";
    cactus_model_t model = cactus_init(model_path.c_str(), nullptr, false);
    if (!model) {
        std::cerr << colored("Failed to initialize model from ", Color::RED + Color::BOLD) << model_path << "\n";
        const char* err = cactus_get_last_error();
        if (err && *err) std::cerr << "  " << err << "\n";
        return 1;
    }
    std::cout << colored("Model loaded.", Color::GREEN + Color::BOLD) << "\n";

    int rc = audio_path.empty() ? run_live(model, language) : run_file(model, audio_path, language);

    std::cout << colored("\nGoodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return rc;
}
