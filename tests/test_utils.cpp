#include "test_utils.h"
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
    std::cout << "⚡PERF │ " << std::left << std::setw(25) << test_name << " │ " << details << "\n";
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
                          const std::vector<int8_t>& data_a,
                          const std::vector<int8_t>& data_b,
                          const std::vector<int8_t>& expected,
                          const std::vector<size_t>& shape) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::INT8);
    size_t input_b = graph.input(shape, Precision::INT8);
    size_t result_id = op_func(graph, input_a, input_b);

    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), Precision::INT8);
    graph.set_input(input_b, const_cast<void*>(static_cast<const void*>(data_b.data())), Precision::INT8);
    graph.execute();

    int8_t* output = static_cast<int8_t*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (output[i] != expected[i]) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

bool test_scalar_operation(const std::string& op_name,
                           std::function<size_t(CactusGraph&, size_t, float)> op_func,
                           const std::vector<int8_t>& data,
                           float scalar,
                           const std::vector<int8_t>& expected,
                           const std::vector<size_t>& shape) {
    (void)op_name;
    CactusGraph graph;
    size_t input_a = graph.input(shape, Precision::INT8);
    size_t result_id = op_func(graph, input_a, scalar);

    graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data.data())), Precision::INT8);
    graph.execute();

    int8_t* output = static_cast<int8_t*>(graph.get_output(result_id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (output[i] != expected[i]) {
            graph.hard_reset();
            return false;
        }
    }
    graph.hard_reset();
    return true;
}

}

namespace EngineTestUtils {

static size_t baseline_memory_ = 0;
static size_t peak_memory_ = 0;

size_t get_memory_footprint_bytes() {
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

void capture_memory_baseline() {
    baseline_memory_ = get_memory_footprint_bytes();
    peak_memory_ = baseline_memory_;
}

double get_memory_usage_mb() {
    size_t current = get_memory_footprint_bytes();
    if (current > peak_memory_) peak_memory_ = current;
    size_t model_mem = (current > baseline_memory_) ? (current - baseline_memory_) : 0;
    return model_mem / (1024.0 * 1024.0);
}

double get_peak_model_memory_mb() {
    size_t model_peak = (peak_memory_ > baseline_memory_) ? (peak_memory_ - baseline_memory_) : 0;
    return model_peak / (1024.0 * 1024.0);
}

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
    size_t q1 = json.find('"', pos + pattern.size());
    if (q1 == std::string::npos) return {};
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
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

void Metrics::parse(const std::string& response) {
    ttft = json_number(response, "time_to_first_token_ms");
    tps = json_number(response, "tokens_per_second");
    total_ms = json_number(response, "total_time_ms");
    prompt_tokens = json_number(response, "prompt_tokens", json_number(response, "prefill_tokens"));
    completion_tokens = json_number(response, "completion_tokens", json_number(response, "decode_tokens"));
}

void Metrics::print() const {
    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << ttft << " ms\n"
              << "├─ Tokens per second: " << tps << std::endl;
}

void Metrics::print_full() const {
    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << ttft << " ms\n"
              << "├─ Tokens per second:  " << tps << "\n"
              << "├─ Total time:         " << total_ms << " ms\n"
              << "├─ Prompt tokens:      " << prompt_tokens << "\n"
              << "├─ Completion tokens:  " << completion_tokens << std::endl;
}

void Metrics::print_perf(double ram_mb) const {
    double prefill_tps = (prompt_tokens > 0 && ttft > 0) ? (prompt_tokens * 1000.0 / ttft) : 0.0;
    double ttft_sec = ttft / 1000.0;
    std::cout << "├─ TTFT: " << std::fixed << std::setprecision(2) << ttft_sec << " sec\n"
              << "├─ Prefill: " << std::setprecision(1) << prefill_tps << " toks/sec\n"
              << "├─ Decode: " << tps << " toks/sec\n"
              << "└─ RAM: " << std::setprecision(1) << ram_mb << " MB" << std::endl;
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
