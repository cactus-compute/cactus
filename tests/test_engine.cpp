#include "test_utils.h"
#include "../cactus/ffi/cactus_telemetry.h"
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <chrono>

using namespace EngineTestUtils;

const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");
const char* g_whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";

const char* g_options = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
    })";

template<typename TestFunc>
bool run_test(const char* title, const char* messages, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {
    return EngineTestUtils::run_test(title, g_model_path, messages, g_options, test_logic, tools, stop_at);
}

bool test_streaming() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      STREAMING & FOLLOW-UP TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* messages1 = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"}
    ])";

    StreamingData data1;
    data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: My name is Henry Ndubuaku, how are you?\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1, response1, sizeof(response1),
                                 g_options, nullptr, stream_callback, &data1);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_perf(get_memory_usage_mb());

    bool success1 = result1 > 0 && data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for(const auto& token : data1.tokens) {
        assistant_response += token;
    }

    std::string messages2_str = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"},
        {"role": "assistant", "content": ")" + escape_json(assistant_response) + R"("},
        {"role": "user", "content": "What is my name?"}
    ])";

    StreamingData data2;
    data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: What is my name?\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2_str.c_str(), response2, sizeof(response2),
                                 g_options, nullptr, stream_callback, &data2);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_perf(get_memory_usage_mb());

    bool success2 = result2 > 0 && data2.token_count > 0;

    cactus_destroy(model);
    return success1 && success2;
}

bool test_tool_call() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TOOL CALL TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("function_call") != std::string::npos;
            bool has_tool = response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_perf(get_memory_usage_mb());
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in San Francisco?");
}

bool test_image_input() {
    std::string model_path_str(g_model_path ? g_model_path : "");
    std::string config_path = model_path_str + "/config.txt";
    std::ifstream config_file(config_path);
    if (config_file.good()) {
        std::string line;
        while (std::getline(config_file, line)) {
            if (line.find("model_variant=vlm") != std::string::npos ||
                line.find("model_variant=VLM") != std::string::npos) {
                break;
            }
        }
        config_file.close();
    }

    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "Skipping image input test: vision weights not found." << std::endl;
        return true;
    }
    vf.close();

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          IMAGE INPUT TEST                ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "Failed to initialize model for image test" << std::endl;
        return false;
    }

    std::string img_path = std::string(g_assets_path) + "/test_monkey.png";
    std::string messages_json = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}]";

    StreamingData stream_data;
    stream_data.model = model;

    char response[4096];

    std::cout << "Response: ";
    int result = cactus_complete(model, messages_json.c_str(), response, sizeof(response),
                                 g_options, nullptr, stream_callback, &stream_data);

    std::cout << "\n\n[Results]\n";
    Metrics metrics;
    metrics.parse(response);
    metrics.print_perf(get_memory_usage_mb());

    bool success = result > 0 && stream_data.token_count > 0;
    cactus_destroy(model);
    return success;
}

bool test_tool_call_with_two_tools() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Set an alarm for 10:00 AM."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                    "minute": {"type": "integer", "description": "Minute to set the alarm for"}
                },
                "required": ["hour", "minute"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("DOUBLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("function_call") != std::string::npos ||
                               response.find("set_alarm") != std::string::npos;
            bool has_tool = response.find("set_alarm") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_perf(get_memory_usage_mb());
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Set an alarm for 10:00 AM.");
}

bool test_tool_call_with_three_tools() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to John saying hello."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                    "minute": {"type": "integer", "description": "Minute to set the alarm for"}
                },
                "required": ["hour", "minute"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                    "message": {"type": "string", "description": "The message content to send"}
                },
                "required": ["recipient", "message"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TRIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("function_call") != std::string::npos ||
                               response.find("send_message") != std::string::npos;
            bool has_tool = response.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_perf(get_memory_usage_mb());
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Send a message to John saying hello.");
}

bool test_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          EMBEDDINGS TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, 2048, nullptr);
    if (!model) return false;

    const char* texts[] = {"My name is Henry Ndubuaku", "Your name is Henry Ndubuaku"};
    std::vector<float> emb1(2048), emb2(2048);
    size_t dim1, dim2;

    Timer t1;
    cactus_embed(model, texts[0], emb1.data(), emb1.size() * sizeof(float), &dim1, true);
    double time1 = t1.elapsed_ms();

    Timer t2;
    cactus_embed(model, texts[1], emb2.data(), emb2.size() * sizeof(float), &dim2, true);
    double time2 = t2.elapsed_ms();

    float similarity = 0;
    for (size_t i = 0; i < dim1; ++i) {
        similarity += emb1[i] * emb2[i];
    }

    std::cout << "\n[Results]\n"
              << "├─ Embedding dim: " << dim1 << "\n"
              << "├─ Time (text1): " << std::fixed << std::setprecision(2) << time1 << "ms\n"
              << "├─ Time (text2): " << time2 << "ms\n"
              << "├─ Similarity: " << std::setprecision(4) << similarity << "\n"
              << "└─ RAM: " << std::setprecision(1) << get_memory_usage_mb() << "MB" << std::endl;

    cactus_destroy(model);
    return true;
}

bool test_100_context() {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 10; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 10; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";

    return run_test("100 CONTEXT TEST", msg.c_str(),
        [](int result, const StreamingData&, const std::string&, const Metrics& m) {
            m.print_perf(get_memory_usage_mb());
            return result > 0;
        }, nullptr, 100);
}

bool test_1k_context() {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 50; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 50; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";

    return run_test("1K CONTEXT TEST", msg.c_str(),
        [](int result, const StreamingData&, const std::string&, const Metrics& m) {
            m.print_perf(get_memory_usage_mb());
            return result > 0;
        }, nullptr, 100);
}

bool test_4k_context() {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 230; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 230; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";

    return run_test("4K CONTEXT TEST", msg.c_str(),
        [](int result, const StreamingData&, const std::string&, const Metrics& m) {
            m.print_perf(get_memory_usage_mb());
            return result > 0;
        }, nullptr, 100);
}

bool test_rag() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What has Justin been doing at Cactus Candy?"}
    ])";

    std::string modelPathStr(g_model_path ? g_model_path : "");

    bool is_rag = false;
    if (!modelPathStr.empty()) {
        std::string config_path = modelPathStr + "/config.txt";
        FILE* cfg = std::fopen(config_path.c_str(), "r");
        if (cfg) {
            char buf[4096];
            while (std::fgets(buf, sizeof(buf), cfg)) {
                std::string line(buf);
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
                if (line.find("model_variant=") != std::string::npos) {
                    auto pos = line.find('=');
                    if (pos != std::string::npos) {
                        std::string val = line.substr(pos + 1);
                        if (val.find("rag") != std::string::npos) {
                            is_rag = true;
                            break;
                        }
                    }
                }
            }
            std::fclose(cfg);
        } else {
            if (modelPathStr.find("rag") != std::string::npos) is_rag = true;
        }
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         RAG PREPROCESSING TEST           ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!is_rag) {
        std::cout << "⊘ SKIP │ model variant is not RAG\n";
        return true;
    }

    std::string corpus_dir = std::string(g_assets_path) + "/rag_corpus";

    cactus_model_t model = cactus_init(g_model_path, 2048, corpus_dir.c_str());
    if (!model) {
        std::cerr << "[✗] Failed to initialize RAG model with corpus dir\n";
        return false;
    }

    StreamingData data;
    data.model = model;

    char response[4096];
    std::cout << "Response: ";

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data);

    std::cout << "\n\n[Results]\n";
    Metrics metrics;
    metrics.parse(response);
    metrics.print_perf(get_memory_usage_mb());

    bool success = (result > 0) && (data.token_count > 0);
    cactus_destroy(model);
    return success;
}

bool test_audio_processor() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO PROCESSOR TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";
    using namespace cactus::engine;

    Timer t;

    const size_t n_fft = 400;
    const size_t hop_length = 160;
    const size_t sampling_rate = 16000;
    const size_t feature_size = 80;
    const size_t num_frequency_bins = 1 + n_fft / 2;

    AudioProcessor audio_proc;
    audio_proc.init_mel_filters(num_frequency_bins, feature_size, 0.0f, 8000.0f, sampling_rate);

    const size_t n_samples = sampling_rate;
    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * M_PI * 440.0f * i / sampling_rate);
    }

    AudioProcessor::SpectrogramConfig config;
    config.n_fft = n_fft;
    config.hop_length = hop_length;
    config.frame_length = n_fft;
    config.power = 2.0f;
    config.center = true;
    config.log_mel = "log10";

    auto log_mel_spec = audio_proc.compute_spectrogram(waveform, config);

    double elapsed = t.elapsed_ms();

    const float expected[] = {0.535175f, 0.548542f, 0.590673f, 0.633320f, 0.711979f};
    const float tolerance = 2e-6f;

    const size_t pad_length = n_fft / 2;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - n_fft) / hop_length;

    bool passed = true;
    for (size_t i = 0; i < 5; i++) {
        if (std::abs(log_mel_spec[i * num_frames] - expected[i]) > tolerance) {
            passed = false;
            break;
        }
    }

    std::cout << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n"
              << "└─ RAM: " << std::setprecision(1) << get_memory_usage_mb() << "MB" << std::endl;

    return passed;
}

template<typename Predicate>
bool run_whisper_test(const char* title, const char* options_json, Predicate check) {
    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << title
                  << " │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_transcribe_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    char response[1 << 15] = {0};
    StreamingData stream;
    stream.model = model;

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    std::cout << "Transcript: ";
    int rc = cactus_transcribe(model, audio_path.c_str(), g_whisper_prompt,
                               response, sizeof(response), options_json,
                               stream_callback, &stream, nullptr, 0);

    std::cout << "\n\n[Results]\n";
    if (rc <= 0) {
        std::cerr << "failed\n";
        cactus_destroy(model);
        return false;
    }

    Metrics m;
    m.parse(response);
    m.print_perf(get_memory_usage_mb());

    bool ok = check(rc, m);
    cactus_destroy(model);
    return ok;
}

static bool test_transcription() {
    return run_whisper_test("TRANSCRIPTION", R"({"max_tokens": 100})",
        [](int rc, const Metrics& m) { return rc > 0 && m.completion_tokens >= 8; });
}

static bool test_image_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         IMAGE EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }

    std::string image_path = std::string(g_assets_path) + "/test_monkey.png";
    const size_t buffer_size = 1024 * 1024 * 4;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_model_path, 2048, nullptr);
    if (!model) {
        std::cout << "⊘ SKIP │ Model doesn't support image embeddings\n";
        return true;
    }

    Timer t;
    int result = cactus_image_embed(model, image_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);
    double elapsed = t.elapsed_ms();

    cactus_destroy(model);

    if (result == -1) {
        std::cout << "⊘ SKIP │ Model doesn't support image embeddings\n";
        return true;
    }

    std::cout << "├─ Embedding dim: " << embedding_dim << "\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n"
              << "└─ RAM: " << std::setprecision(1) << get_memory_usage_mb() << "MB" << std::endl;

    return result > 0 && embedding_dim > 0;
}

static bool test_audio_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO EMBEDDING TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    const size_t buffer_size = 1024 * 1024;
    std::vector<float> embeddings(buffer_size / sizeof(float));
    size_t embedding_dim = 0;

    cactus_model_t model = cactus_init(g_transcribe_model_path, 2048, nullptr);
    if (!model) {
        std::cout << "⊘ SKIP │ Failed to init Whisper model\n";
        return true;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    Timer t;
    int result = cactus_audio_embed(model, audio_path.c_str(), embeddings.data(), buffer_size, &embedding_dim);
    double elapsed = t.elapsed_ms();

    cactus_destroy(model);

    if (result == -1) {
        std::cout << "⊘ SKIP │ Model doesn't support audio embeddings\n";
        return true;
    }

    std::cout << "├─ Embedding dim: " << embedding_dim << "\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n"
              << "└─ RAM: " << std::setprecision(1) << get_memory_usage_mb() << "MB" << std::endl;

    return result > 0 && embedding_dim > 0;
}

static bool test_pcm_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       PCM BUFFER TRANSCRIPTION           ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_transcribe_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_transcribe_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    const size_t sample_rate = 16000;
    bool use_microphone = false;
    bool test_passed = false;

#ifdef HAVE_SDL2
    {
        std::cout << "Using microphone input (SDL2)...\n";

        AudioCapture audio_capture(10000);
        if (audio_capture.init(0, sample_rate)) {
            std::cout << "\n🎤 Recording for 10 seconds... Speak now!\n\n";

            audio_capture.resume();
            use_microphone = true;

            std::this_thread::sleep_for(std::chrono::seconds(10));

            audio_capture.pause();

            std::vector<float> audio_float;
            size_t num_samples = audio_capture.get_all(audio_float);

            if (num_samples == 0) {
                std::cerr << "[!] No audio captured\n";
                use_microphone = false;
            } else {
                std::cout << "Captured " << (num_samples / sample_rate)
                          << " seconds of audio, transcribing...\n";

                std::vector<int16_t> pcm_samples(num_samples);
                for (size_t i = 0; i < num_samples; i++) {
                    float clamped = std::max(-1.0f, std::min(1.0f, audio_float[i]));
                    pcm_samples[i] = static_cast<int16_t>(clamped * 32767.0f);
                }

                // Transcribe
                char response[1 << 15] = {0};
                StreamingData stream;
                stream.model = model;

                std::cout << "Transcript: ";
                int rc = cactus_transcribe(
                    model,
                    nullptr,
                    g_whisper_prompt,
                    response,
                    sizeof(response),
                    R"({"max_tokens": 100})",
                    stream_callback,
                    &stream,
                    reinterpret_cast<const uint8_t*>(pcm_samples.data()),
                    pcm_samples.size() * sizeof(int16_t)
                );

                std::cout << "\n\n[Results]\n";
                if (rc > 0) {
                    Metrics m;
                    m.parse(response);
                    m.print_perf(get_memory_usage_mb());
                    test_passed = (rc > 0 && m.completion_tokens >= 1);
                } else {
                    std::cerr << "Transcription failed\n";
                }
            }
        } else {
            std::cerr << "[!] Failed to initialize audio capture, falling back to synthetic audio\n";
        }
    }
#endif
    if (!use_microphone) {
        std::cout << "Using synthetic audio (440Hz sine wave)...\n";
        const size_t duration_seconds = 3;
        const size_t num_samples = sample_rate * duration_seconds;
        std::vector<int16_t> pcm_samples(num_samples);

        for (size_t i = 0; i < num_samples; i++) {
            float t = static_cast<float>(i) / sample_rate;
            float amplitude = 0.3f;
            float value = amplitude * std::sin(2.0f * M_PI * 440.0f * t);
            pcm_samples[i] = static_cast<int16_t>(value * 32767.0f);
        }

        char response[1 << 15] = {0};
        StreamingData stream;
        stream.model = model;

        std::cout << "Transcript: ";
        int rc = cactus_transcribe(
            model,
            nullptr,
            g_whisper_prompt,
            response,
            sizeof(response),
            R"({"max_tokens": 100})",
            stream_callback,
            &stream,
            reinterpret_cast<const uint8_t*>(pcm_samples.data()),
            pcm_samples.size() * sizeof(int16_t)
        );

        std::cout << "\n\n[Results]\n";
        if (rc <= 0) {
            std::cerr << "failed\n";
            cactus_destroy(model);
            return false;
        }

        Metrics m;
        m.parse(response);
        m.print_perf(get_memory_usage_mb());

        std::cout << "├─ PCM samples: " << pcm_samples.size() << "\n"
                  << "├─ Duration: " << duration_seconds << "s\n"
                  << "└─ Sample rate: " << sample_rate << "Hz\n";

        test_passed = (rc > 0 && m.completion_tokens >= 1);
    }

    cactus_destroy(model);
    return test_passed;
}

int main() {
#ifdef __APPLE__
    cactus_set_telemetry_enabled(1);
    cactus_set_telemetry_token("973e4aaa-5ee4-4947-a128-757bb66be75b");
#endif

    capture_memory_baseline();
    TestUtils::TestRunner runner("Engine Tests");
    runner.run_test("streaming", test_streaming());
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("tool_calls_with_two_tools", test_tool_call_with_two_tools());
    runner.run_test("tool_calls_with_three_tools", test_tool_call_with_three_tools());
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("image_embeddings", test_image_embeddings());
    runner.run_test("audio_embeddings", test_audio_embeddings());
    runner.run_test("image_input", test_image_input());
    runner.run_test("audio_processor", test_audio_processor());
    runner.run_test("transcription", test_transcription());
    runner.run_test("pcm_transcription", test_pcm_transcription());
    runner.run_test("rag_preprocessing", test_rag());
    runner.run_test("100_context", test_100_context());
    runner.run_test("1k_context", test_1k_context());
    runner.run_test("4k_context", test_4k_context());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}