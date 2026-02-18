#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/telemetry/telemetry.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <thread>
#include <vector>
#include <deque>
#include <cctype>
#include <algorithm>
#include <sys/ioctl.h>
#include <unistd.h>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#endif

static std::string get_cloud_api_key() {
    const char* key = std::getenv("CACTUS_CLOUD_API_KEY");
    return key ? std::string(key) : "";
}

static std::string get_transcribe_options_json() {
    const char* threshold = std::getenv("CACTUS_CLOUD_HANDOFF_THRESHOLD");
    std::ostringstream oss;
    oss << "{\"max_tokens\":500,\"telemetry_enabled\":true";
    if (threshold && threshold[0] != '\0') {
        oss << ",\"cloud_handoff_threshold\":" << threshold;
    }
    oss << "}";
    return oss.str();
}

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

void print_header_live_mode() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("     🌵 CACTUS LIVE TRANSCRIPTION 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Listening...", Color::YELLOW) << " Press " << colored("Enter", Color::CYAN) << " to stop\n";
    print_separator();
    std::cout << "\n";
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string extract_json_value(const std::string& json, const std::string& key) {
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

std::string extract_json_number(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t start = json.find(pattern);
    if (start == std::string::npos) return "";
    start += pattern.length();
    while (start < json.length() && std::isspace(json[start])) start++;
    size_t end = start;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-')) {
        end++;
    }
    return json.substr(start, end - start);
}

size_t visible_length(const std::string& s) {
    size_t len = 0;
    bool in_esc = false;
    for (char c : s) {
        if (c == '\033') { in_esc = true; continue; }
        if (in_esc) { if (c == 'm') in_esc = false; continue; }
        if ((c & 0xC0) != 0x80) len++;
    }
    return len;
}

int get_terminal_width() {
    struct winsize w;
    return (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) ? 80 : w.ws_col;
}

std::string truncate_visible(const std::string& s, size_t limit) {
    std::string out;
    size_t len = 0;
    bool in_esc = false;
    for (char c : s) {
        if (c == '\033') { in_esc = true; out += c; continue; }
        if (in_esc) { out += c; if (c == 'm') in_esc = false; continue; }
        if (len >= limit && (c & 0xC0) != 0x80) break;
        out += c;
        if ((c & 0xC0) != 0x80) len++;
    }
    return out + "\033[0m";
}

size_t find_safe_split_index(const std::string& s, size_t limit) {
    size_t len = 0;
    bool in_esc = false;
    for (size_t i = 0; i < s.length(); ++i) {
        char c = s[i];
        if (c == '\033') { in_esc = true; continue; }
        if (in_esc) { if (c == 'm') in_esc = false; continue; }
        if ((c & 0xC0) != 0x80) len++;
        if (len >= limit && c == ' ' && !in_esc) return i;
    }
    return std::string::npos;
}

void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    std::cout << token << std::flush;
}

std::string get_transcribe_prompt(const std::string& model_path) {
    std::string path_lower = model_path;
    std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (path_lower.find("whisper") != std::string::npos) {
        return "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";
    }
    return "";  // Moonshine uses empty prompt
}

int transcribe_file(cactus_model_t model, const std::string& audio_path, const std::string& model_path) {
    if (!file_exists(audio_path)) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "File not found: " << audio_path << "\n";
        return -1;
    }

    std::string prompt = get_transcribe_prompt(model_path);
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

    auto start_time = std::chrono::steady_clock::now();

    const std::string options_json = get_transcribe_options_json();

    int result = cactus_transcribe(
        model,
        audio_path.c_str(),
        prompt.c_str(),
        response_buffer.data(),
        response_buffer.size(),
        options_json.c_str(),
        print_token,
        nullptr,
        nullptr,
        0
    );

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_seconds = duration.count() / 1000.0;

    if (result < 0) {
        std::cerr << "\n" << colored("Error: ", Color::RED + Color::BOLD)
                  << "Transcription failed\n";
        const char* err = cactus_get_last_error();
        if (err) {
            std::cerr << colored("Details: ", Color::RED) << err << "\n";
        }
        return -1;
    }

    std::string json_str(response_buffer.data());
    bool cloud_handoff = json_str.find("\"cloud_handoff\":true") != std::string::npos;

    std::string time_str;
    size_t time_pos = json_str.find("\"total_time_ms\":");
    if (time_pos != std::string::npos) {
        size_t start = time_pos + 16;
        size_t end = json_str.find_first_of(",}", start);
        time_str = json_str.substr(start, end - start);
    }

    std::ostringstream stats;
    stats << std::fixed << std::setprecision(2);
    stats << "\n\n" << colored("[", Color::GRAY);
    stats << colored("processed in: ", Color::GRAY) << total_seconds << "s";
    if (!time_str.empty()) {
        double model_time = std::stod(time_str) / 1000.0;
        stats << colored(" | model time: ", Color::GRAY) << model_time << "s";
    }
    stats << colored("]", Color::GRAY);
    stats << "\n" << colored("[cloud_handoff: ", Color::GRAY)
          << (cloud_handoff ? colored("true", Color::YELLOW) : colored("false", Color::GREEN))
          << colored("]", Color::GRAY);

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

std::vector<uint8_t> resample_audio(const std::vector<uint8_t>& input, int source_rate, int target_rate) {
    if (source_rate == target_rate || input.empty()) {
        return input;
    }

    size_t num_input_samples = input.size() / 2;
    if (num_input_samples == 0) return input;

    const int16_t* input_samples = reinterpret_cast<const int16_t*>(input.data());

    double ratio = static_cast<double>(target_rate) / source_rate;
    size_t num_output_samples = static_cast<size_t>(num_input_samples * ratio);
    if (num_output_samples == 0) return {};

    std::vector<int16_t> output_samples(num_output_samples);

    for (size_t i = 0; i < num_output_samples; i++) {
        double src_idx = i / ratio;
        size_t idx0 = static_cast<size_t>(src_idx);
        size_t idx1 = std::min(idx0 + 1, num_input_samples - 1);
        double frac = src_idx - idx0;

        double sample = input_samples[idx0] * (1.0 - frac) + input_samples[idx1] * frac;
        output_samples[i] = static_cast<int16_t>(std::clamp(sample, -32768.0, 32767.0));
    }

    std::vector<uint8_t> result(num_output_samples * 2);
    std::memcpy(result.data(), output_samples.data(), result.size());
    return result;
}

struct Segment {
    std::string text;
    bool pending_cloud = false;
    std::chrono::steady_clock::time_point cloud_start_time;
    int64_t cloud_job_id = -1;
};

AudioState g_audio_state;

void audio_callback(void* /*userdata*/, Uint8* stream, int len) {
    if (!g_audio_state.recording) return;

    std::lock_guard<std::mutex> lock(g_audio_state.mutex);
    g_audio_state.buffer.insert(g_audio_state.buffer.end(), stream, stream + len);
}

int run_live_transcription(cactus_model_t model) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to initialize SDL: " << SDL_GetError() << "\n";
        return 1;
    }

    int num_devices = SDL_GetNumAudioDevices(1); 
    if (num_devices == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "No audio capture devices found\n";
        SDL_Quit();
        return 1;
    }

    std::cout << colored("Available microphones:", Color::YELLOW) << "\n";
    for (int i = 0; i < num_devices; i++) {
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, 1) << "\n";
    }
    std::cout << "\n";

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = TARGET_SAMPLE_RATE;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (TARGET_SAMPLE_RATE * AUDIO_BUFFER_MS) / 1000;
    want.callback = audio_callback;
    want.userdata = nullptr;

    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (device == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to open audio device: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    g_audio_state.actual_sample_rate = have.freq;
    if (have.freq != TARGET_SAMPLE_RATE) {
        std::cout << colored("Note: ", Color::YELLOW) << "Audio device uses " << have.freq
                  << "Hz, will resample to " << TARGET_SAMPLE_RATE << "Hz\n";
    }

    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(
        model, R"({"confirmation_threshold": 0.99, "min_chunk_size": 16000, "telemetry_enabled": true})"
    );

    if (!stream) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to initialize streaming transcription\n";
        SDL_CloseAudioDevice(device);
        SDL_Quit();
        return 1;
    }

    std::string api_key = get_cloud_api_key();
    if (api_key.empty()) {
        std::cout << colored("Warning: ", Color::YELLOW + Color::BOLD)
                  << "CACTUS_CLOUD_API_KEY environment variable not set.\n";
        std::cout << colored("         Cloud handoff will be disabled (fallback to local transcription).\n", Color::YELLOW);
        std::cout << "\n";
    }

    print_header_live_mode();

    g_audio_state.recording = true;
    g_audio_state.buffer.clear();
    SDL_PauseAudioDevice(device, 0);

    std::atomic<bool> should_stop{false};
    std::thread input_thread([&should_stop]() {
        std::string line;
        std::getline(std::cin, line);
        should_stop = true;
    });

    std::deque<Segment> segments;
    std::string confirmed_text;
    std::string current_line_confirmed;
    
    int last_pending_line_count = 0;
    std::string last_stats;

    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

    auto last_process_time = std::chrono::steady_clock::now();
    const auto process_interval = std::chrono::milliseconds(1000);

    while (!should_stop) {
        auto now = std::chrono::steady_clock::now();

        if (now - last_process_time >= process_interval) {
            last_process_time = now;

            std::vector<uint8_t> audio_chunk;
            {
                std::lock_guard<std::mutex> lock(g_audio_state.mutex);
                if (!g_audio_state.buffer.empty()) {
                    audio_chunk = std::move(g_audio_state.buffer);
                    g_audio_state.buffer.clear();
                }
            }

            if (!audio_chunk.empty()) {
                std::vector<uint8_t> resampled = resample_audio(
                    audio_chunk, g_audio_state.actual_sample_rate, TARGET_SAMPLE_RATE
                );

                auto t_start = std::chrono::high_resolution_clock::now();
                int process_result = cactus_stream_transcribe_process(
                    stream,
                    resampled.data(),
                    resampled.size(),
                    response_buffer.data(),
                    response_buffer.size()
                );
                auto t_end = std::chrono::high_resolution_clock::now();
                double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;

                if (process_result >= 0) {
                    std::string json_str(response_buffer.data());
                    std::string confirmed = extract_json_value(json_str, "confirmed");
                    std::string pending = extract_json_value(json_str, "pending");
                    std::string cloud_result = extract_json_value(json_str, "cloud_result");
                    std::string cloud_job_id = extract_json_number(json_str, "cloud_job_id");
                    std::string cloud_result_job_id = extract_json_number(json_str, "cloud_result_job_id");
                    std::string ttft = extract_json_number(json_str, "time_to_first_token_ms");
                    std::string decode_tps = extract_json_number(json_str, "decode_tps");

                    if (!confirmed.empty()) {
                        Segment seg;
                        seg.text = confirmed;

                        bool is_cloud = json_str.find("\"cloud_handoff\":true") != std::string::npos;
                        int64_t parsed_cloud_job_id = cloud_job_id.empty() ? 0 : std::stoll(cloud_job_id);
                        if (is_cloud && parsed_cloud_job_id > 0) {
                            seg.pending_cloud = true;
                            seg.cloud_start_time = std::chrono::steady_clock::now();
                            seg.cloud_job_id = parsed_cloud_job_id;
                        }
                        segments.push_back(seg);
                    }

                    if (!cloud_result.empty()) {
                        int64_t result_job_id = cloud_result_job_id.empty() ? 0 : std::stoll(cloud_result_job_id);
                        if (result_job_id > 0) {
                            for (auto& seg : segments) {
                                if (seg.pending_cloud && seg.cloud_job_id == result_job_id) {
                                    seg.text = cloud_result;
                                    seg.pending_cloud = false;
                                    break;
                                }
                            }
                        }
                    }

                    for (auto& seg : segments) {
                        if (seg.pending_cloud &&
                            std::chrono::steady_clock::now() - seg.cloud_start_time > std::chrono::seconds(10)) {
                            seg.pending_cloud = false;
                        }
                    }

                    if (!confirmed.empty() || !pending.empty()) {
                        last_stats = colored("[Latency:" + std::to_string(int(latency_ms)) + "ms Decode speed:" + decode_tps + " tokens/sec] ", Color::GRAY);
                    }

                    int width = get_terminal_width();
                    int limit = (width < 20 ? 80 : width) * 0.7;

                    if (last_pending_line_count > 0) {
                        std::cout << "\r\033[2K";
                        for (int i = 0; i < last_pending_line_count; ++i)
                            std::cout << "\033[1A\033[2K";
                    } else {
                        std::cout << "\r";
                    }

                    while (!segments.empty() && !segments.front().pending_cloud) {
                        current_line_confirmed += colored(segments.front().text, Color::GREEN) + " ";
                        confirmed_text += segments.front().text + " ";
                        segments.pop_front();
                    }

                    while (true) {
                        size_t idx = find_safe_split_index(current_line_confirmed, limit);
                        if (idx == std::string::npos) break;
                        std::string part = current_line_confirmed.substr(0, idx);
                        std::string rem = current_line_confirmed.substr(idx + 1);
                        std::cout << "\r\033[K" << part + Color::RESET << "\n";
                        current_line_confirmed = Color::GREEN + rem;
                    }

                    std::cout << "\r\033[K" << current_line_confirmed;

                    std::string ghost = last_stats;
                    if (!segments.empty()) {
                        if (!ghost.empty()) ghost += "\n";
                        ghost += colored("[Awaiting Cloud] ", Color::RED);
                        for (const auto& seg : segments)
                            ghost += colored(seg.text, seg.pending_cloud ? Color::RED : Color::GREEN) + " ";
                    }
                    if (!pending.empty()) {
                        if (!ghost.empty()) ghost += "\n";
                        ghost += colored("[Uncommitted] ", Color::YELLOW) + colored(pending, Color::YELLOW);
                    }

                    last_pending_line_count = 0;
                    if (!ghost.empty()) {
                        std::cout << "\n";
                        std::stringstream ss(ghost);
                        std::string line;
                        bool first = true;
                        while (std::getline(ss, line)) {
                            while (true) {
                                size_t idx = find_safe_split_index(line, limit);
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
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    g_audio_state.recording = false;
    SDL_PauseAudioDevice(device, 1);

    int stopping_result = cactus_stream_transcribe_stop(
        stream,
        response_buffer.data(),
        response_buffer.size()
    );

    std::cout << "\n\n";
    print_separator();

    if (stopping_result >= 0) {
        for (const auto& seg : segments) {
            confirmed_text += seg.text + " ";
        }
    
        std::string json_str(response_buffer.data());
        std::string final_text = extract_json_value(json_str, "confirmed");
        std::string full_transcript = confirmed_text + final_text;

        std::cout << colored("Final transcript:", Color::GREEN + Color::BOLD) << "\n";
        std::cout << full_transcript << "\n";
    } else {
        if (!confirmed_text.empty()) {
            std::cout << colored("Partial transcript:", Color::YELLOW + Color::BOLD) << "\n";
            std::cout << confirmed_text << "\n";
        }
    }

    print_separator();

    if (input_thread.joinable()) {
        input_thread.detach();
    }
    SDL_CloseAudioDevice(device);
    SDL_Quit();

    return 0;
}

#endif // HAVE_SDL2

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Missing model path\n";
        std::cerr << "Usage: " << argv[0] << " <model_path> [audio_file]\n";
        std::cerr << "\nModes:\n";
        std::cerr << "  " << argv[0] << " weights/whisper-small              # Live microphone transcription\n";
        std::cerr << "  " << argv[0] << " weights/whisper-small audio.wav    # Transcribe single file\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* audio_file = argc > 2 ? argv[2] : nullptr;

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);

    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        const char* err = cactus_get_last_error();
        if (err) {
            std::cerr << colored("Error: ", Color::RED) << err << "\n";
        }
        return 1;
    }

    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    int result = 0;

    if (audio_file) {
        std::cout << "\n" << colored("Transcribing: ", Color::BLUE + Color::BOLD)
                  << audio_file << "\n\n";
        result = transcribe_file(model, audio_file, model_path);
    } else {
#ifdef HAVE_SDL2
        result = run_live_transcription(model);
#else
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Live transcription requires SDL2.\n";
        std::cerr << "Please install SDL2 and rebuild:\n";
        std::cerr << "  macOS:  brew install sdl2\n";
        std::cerr << "  Linux:  sudo apt-get install libsdl2-dev\n";
        result = 1;
#endif
    }

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return result >= 0 ? 0 : 1;
}
