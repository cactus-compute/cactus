#include "test_utils.h"
#include <cstdlib>
#include <random>
#include <sstream>

namespace TestUtils {

static std::mt19937 gen(42);

size_t random_graph_input(CactusGraph& graph, const std::vector<size_t>& shape, Precision precision) {
    size_t node_id = graph.input(shape, precision);
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;

    if (precision == Precision::INT8) {
        std::uniform_int_distribution<int> dist(-50, 50);
        std::vector<int8_t> data(total_elements);
        for (size_t i = 0; i < total_elements; ++i) data[i] = static_cast<int8_t>(dist(gen));
        graph.set_input(node_id, data.data(), precision);
    } else {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<float> data(total_elements);
        for (size_t i = 0; i < total_elements; ++i) data[i] = dist(gen);
        graph.set_input(node_id, data.data(), precision);
    }
    return node_id;
}

bool verify_graph_outputs(CactusGraph& graph, size_t node_a, size_t node_b, float tolerance) {
    graph.execute();
    const auto& buffer_a = graph.get_output_buffer(node_a);
    const auto& buffer_b = graph.get_output_buffer(node_b);

    if (buffer_a.shape != buffer_b.shape || buffer_a.precision != buffer_b.precision) return false;

    void* data_a = graph.get_output(node_a);
    void* data_b = graph.get_output(node_b);
    size_t total_elements = 1;
    for (size_t dim : buffer_a.shape) total_elements *= dim;

    if (buffer_a.precision == Precision::INT8) {
        const int8_t* ptr_a = static_cast<const int8_t*>(data_a);
        const int8_t* ptr_b = static_cast<const int8_t*>(data_b);
        for (size_t i = 0; i < total_elements; ++i)
            if (std::abs(ptr_a[i] - ptr_b[i]) > tolerance) return false;
    } else {
        const float* ptr_a = static_cast<const float*>(data_a);
        const float* ptr_b = static_cast<const float*>(data_b);
        for (size_t i = 0; i < total_elements; ++i)
            if (std::abs(ptr_a[i] - ptr_b[i]) > tolerance) return false;
    }

    graph.hard_reset();
    return true;
}

bool verify_graph_against_data(CactusGraph& graph, size_t node_id, const void* expected_data, size_t byte_size, float tolerance) {
    graph.execute();
    void* actual_data = graph.get_output(node_id);
    const auto& buffer = graph.get_output_buffer(node_id);

    if (buffer.precision == Precision::INT8) {
        const int8_t* actual = static_cast<const int8_t*>(actual_data);
        const int8_t* expected = static_cast<const int8_t*>(expected_data);
        for (size_t i = 0; i < byte_size; ++i)
            if (std::abs(actual[i] - expected[i]) > tolerance) return false;
    } else {
        const float* actual = static_cast<const float*>(actual_data);
        const float* expected = static_cast<const float*>(expected_data);
        size_t count = byte_size / sizeof(float);
        for (size_t i = 0; i < count; ++i)
            if (std::abs(actual[i] - expected[i]) > tolerance) return false;
    }

    graph.hard_reset();
    return true;
}

void fill_random_int8(std::vector<int8_t>& data) {
    std::uniform_int_distribution<int> dist(-50, 50);
    for (auto& val : data) val = static_cast<int8_t>(dist(gen));
}

void fill_random_float(std::vector<float>& data) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& val : data) val = dist(gen);
}

void fill_random_fp16(std::vector<__fp16>& data) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& val : data) val = static_cast<__fp16>(dist(gen));
}

TestRunner::TestRunner(const std::string& suite_name)
    : suite_name_(suite_name), passed_count_(0), total_count_(0) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║ Running " << std::left << std::setw(76) << suite_name_ << " ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";
}

void TestRunner::run_test(const std::string& test_name, bool result) {
    total_count_++;
    if (result) {
        passed_count_++;
        std::cout << "✓ PASS │ " << std::left << std::setw(25) << test_name << "\n";
    } else {
        std::cout << "✗ FAIL │ " << std::left << std::setw(25) << test_name << "\n";
    }
}

void TestRunner::log_performance(const std::string& test_name, const std::string& details) {
    std::cout << "⚡PERF │ " << std::left << std::setw(38) << test_name << " │ " << details << "\n";
}

void TestRunner::log_skip(const std::string& test_name, const std::string& reason) {
    std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << test_name << " │ " << reason << "\n";
}

void TestRunner::print_summary() {
    std::cout << "────────────────────────────────────────────────────────────────────────────────────────\n";
    if (all_passed())
        std::cout << "✓ All " << total_count_ << " tests passed!\n";
    else
        std::cout << "✗ " << (total_count_ - passed_count_) << " of " << total_count_ << " tests failed!\n";
    std::cout << "\n";
}

bool TestRunner::all_passed() const {
    return passed_count_ == total_count_;
}

bool test_basic_operation(const std::string& op_name,
                          std::function<size_t(CactusGraph&, size_t, size_t)> op_func,
                          const std::vector<__fp16>& data_a,
                          const std::vector<__fp16>& data_b,
                          const std::vector<__fp16>& expected,
                          const std::vector<size_t>& shape) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::FP16);
    size_t input_b = graph.input(shape, Precision::FP16);
    size_t result_id = op_func(graph, input_a, input_b);

    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), Precision::FP16);
    graph.set_input(input_b, const_cast<void*>(static_cast<const void*>(data_b.data())), Precision::FP16);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-2f) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

bool test_scalar_operation(const std::string& op_name,
                           std::function<size_t(CactusGraph&, size_t, float)> op_func,
                           const std::vector<__fp16>& data,
                           float scalar,
                           const std::vector<__fp16>& expected,
                           const std::vector<size_t>& shape) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::FP16);
    size_t result_id = op_func(graph, input_a, scalar);

    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data.data())), Precision::FP16);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-2f) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

}

namespace EngineTestUtils {

Timer::Timer() : start(std::chrono::high_resolution_clock::now()) {}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

double json_number(const std::string& json, const std::string& key, double def) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return def;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;
    try { return std::stod(json.substr(start, end - start)); }
    catch (...) { return def; }
}

std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t start = pos + pattern.size();

    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start >= json.size() || json[start] != '"') return {};
    ++start;

    std::string out;
    out.reserve(128);
    bool escaped = false;
    for (size_t i = start; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(c); break;
            }
            escaped = false;
            continue;
        }

        if (c == '\\') {
            escaped = true;
            continue;
        }

        if (c == '"') {
            return out;
        }

        out.push_back(c);
    }
    return {};
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (auto c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            default:   o << c;      break;
        }
    }
    return o.str();
}

void stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token ? token : "");
    data->token_ids.push_back(token_id);
    data->token_count++;

    std::string out = token ? token : "";
    for (char& c : out) if (c == '\n') c = ' ';
    std::cout << out << std::flush;

    if (data->stop_at > 0 && data->token_count >= data->stop_at) {
        std::cout << " [-> stopped]" << std::flush;
        cactus_stop(data->model);
    }
}

bool json_bool(const std::string& json, const std::string& key, bool def = false) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return def;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    if (start + 4 <= json.size() && json.substr(start, 4) == "true") return true;
    if (start + 5 <= json.size() && json.substr(start, 5) == "false") return false;
    return def;
}

std::string json_array(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "[]";
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
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

void Metrics::parse(const std::string& json) {
    success = json_bool(json, "success", false);
    error = json_string(json, "error");
    cloud_handoff = json_bool(json, "cloud_handoff", false);
    response = json_string(json, "response");
    function_calls = json_array(json, "function_calls");
    confidence = json_number(json, "confidence", -1.0);
    ttft = json_number(json, "time_to_first_token_ms");
    total_ms = json_number(json, "total_time_ms");
    prefill_tps = json_number(json, "prefill_tps");
    decode_tps = json_number(json, "decode_tps");
    ram_mb = json_number(json, "ram_usage_mb");
    prefill_tokens = json_number(json, "prefill_tokens");
    completion_tokens = json_number(json, "decode_tokens");
    total_tokens = json_number(json, "total_tokens");
}

void Metrics::print_json() const {
    std::cout << "  \"success\": " << (success ? "true" : "false") << ",\n"
              << "  \"error\": " << (error.empty() ? "null" : "\"" + error + "\"") << ",\n"
              << "  \"cloud_handoff\": " << (cloud_handoff ? "true" : "false") << ",\n"
              << "  \"response\": \"" << response << "\",\n"
              << "  \"function_calls\": " << function_calls << ",\n"
              << "  \"confidence\": " << std::fixed << std::setprecision(4) << confidence << ",\n"
              << "  \"time_to_first_token_ms\": " << std::setprecision(2) << ttft << ",\n"
              << "  \"total_time_ms\": " << total_ms << ",\n"
              << "  \"prefill_tps\": " << prefill_tps << ",\n"
              << "  \"decode_tps\": " << decode_tps << ",\n"
              << "  \"ram_usage_mb\": " << ram_mb << ",\n"
              << "  \"prefill_tokens\": " << std::setprecision(0) << prefill_tokens << ",\n"
              << "  \"decode_tokens\": " << completion_tokens << ",\n"
              << "  \"total_tokens\": " << total_tokens << std::endl;
}

}

#ifdef HAVE_SDL2

AudioCapture::AudioCapture(int len_ms)
    : m_len_ms(len_ms)
    , m_running(false)
    , m_dev_id_in(0)
    , m_sdl_initialized(false)
    , m_audio_pos(0)
    , m_audio_len(0)
    , m_total_samples_received(0) {}

AudioCapture::~AudioCapture() {
    if (m_dev_id_in) SDL_CloseAudioDevice(m_dev_id_in);
    if (m_sdl_initialized) SDL_Quit();
}

bool AudioCapture::init(int capture_id, int sample_rate) {
    static bool sdl_globally_initialized = false;

    if (!sdl_globally_initialized) {
        if (SDL_Init(SDL_INIT_AUDIO) < 0) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
            return false;
        }
        sdl_globally_initialized = true;
        m_sdl_initialized = true;
    }

    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);
    m_audio.resize((m_len_ms * sample_rate) / 1000);

    int num_devices = SDL_GetNumAudioDevices(SDL_TRUE);
    std::cout << "\nAvailable audio capture devices:\n";
    for (int i = 0; i < num_devices; i++)
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, SDL_TRUE) << "\n";

    if (capture_id >= num_devices) {
        std::cerr << "Invalid capture device ID: " << capture_id << std::endl;
        return false;
    }

    std::cout << "Selected device: [" << capture_id << "] "
              << SDL_GetAudioDeviceName(capture_id, SDL_TRUE) << "\n\n";

    SDL_AudioSpec capture_spec_requested;
    SDL_zero(capture_spec_requested);
    capture_spec_requested.freq = sample_rate;
    capture_spec_requested.format = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples = 1024;
    capture_spec_requested.callback = [](void* userdata, uint8_t* stream, int len) {
        static_cast<AudioCapture*>(userdata)->callback(stream, len);
    };
    capture_spec_requested.userdata = this;

    SDL_AudioSpec capture_spec_obtained;
    m_dev_id_in = SDL_OpenAudioDevice(
        SDL_GetAudioDeviceName(capture_id, SDL_TRUE),
        SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);

    if (!m_dev_id_in) {
        std::cerr << "SDL_OpenAudioDevice failed: " << SDL_GetError() << std::endl;
        return false;
    }

    std::cout << "Audio capture initialized:\n"
              << "  Sample rate: " << capture_spec_obtained.freq << " Hz\n"
              << "  Channels: " << (int)capture_spec_obtained.channels << "\n"
              << "  Samples: " << capture_spec_obtained.samples << "\n"
              << "  Buffer length: " << m_len_ms << " ms\n";

    return true;
}

void AudioCapture::resume() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_running && m_dev_id_in) {
        SDL_PauseAudioDevice(m_dev_id_in, 0);
        m_running = true;
    }
}

void AudioCapture::pause() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_running && m_dev_id_in) {
        SDL_PauseAudioDevice(m_dev_id_in, 1);
        m_running = false;
    }
}

void AudioCapture::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_audio_pos = 0;
    m_audio_len = 0;
}

size_t AudioCapture::get(int duration_ms, std::vector<float>& result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t n_samples = (duration_ms * m_audio.size()) / m_len_ms;
    if (n_samples > m_audio_len) return 0;

    result.resize(n_samples);
    size_t start_pos = (m_audio_pos + m_audio.size() - m_audio_len) % m_audio.size();
    for (size_t i = 0; i < n_samples; i++)
        result[i] = m_audio[(start_pos + i) % m_audio.size()];

    m_audio_len = (m_audio_len > n_samples) ? (m_audio_len - n_samples) : 0;
    return n_samples;
}

size_t AudioCapture::get_all(std::vector<float>& result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_audio_len == 0) return 0;

    result.resize(m_audio_len);
    size_t start_pos = (m_audio_pos + m_audio.size() - m_audio_len) % m_audio.size();
    for (size_t i = 0; i < m_audio_len; i++)
        result[i] = m_audio[(start_pos + i) % m_audio.size()];

    size_t n_samples = m_audio_len;
    m_audio_len = 0;
    return n_samples;
}

size_t AudioCapture::get_buffer_length() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_audio_len;
}

void AudioCapture::callback(uint8_t* stream, int len) {
    const size_t n_samples = len / sizeof(float);
    const float* samples = reinterpret_cast<const float*>(stream);
    if (!m_running) return;

    std::lock_guard<std::mutex> lock(m_mutex);
    for (size_t i = 0; i < n_samples; i++) {
        m_audio[m_audio_pos] = samples[i];
        m_audio_pos = (m_audio_pos + 1) % m_audio.size();
        if (m_audio_len < m_audio.size()) m_audio_len++;
    }
    m_total_samples_received += n_samples;
}

#endif
