#include "test_utils.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>

const char* g_model_path = "../../weights/lfm2-1.2B";
const char* g_options = R"({"max_tokens": 256, "stop_sequences": ["<|im_end|>", "<end_of_turn>"]})";

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

struct StreamingData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count = 0;
    cactus_model_t model = nullptr;
    int stop_at = -1;
};

void stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token);
    data->token_ids.push_back(token_id);
    data->token_count++;
    std::cout << token << std::flush;

    if (data->stop_at > 0 && data->token_count >= data->stop_at) {
        std::cout << "\n\n[→ Stopping at token #" << data->stop_at << "]" << std::endl;
        cactus_stop(data->model);
    }
}

struct Metrics {
    double ttft = 0.0;
    double tps = 0.0;

    void parse(const std::string& response) {
        size_t pos = response.find("\"time_to_first_token_ms\":");
        if (pos != std::string::npos) ttft = std::stod(response.substr(pos + 25));

        pos = response.find("\"tokens_per_second\":");
        if (pos != std::string::npos) tps = std::stod(response.substr(pos + 20));
    }

    void print() const {
        std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2)
                  << ttft << " ms\n"
                  << "├─ Tokens per second: " << tps << std::endl;
    }
};

template<typename TestFunc>
bool run_test(const char* title, const char* messages, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    StreamingData data;
    data.model = model;
    data.stop_at = stop_at;

    char response[4096];
    std::cout << "Response: ";

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 g_options, tools, stream_callback, &data);

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);

    bool success = test_logic(result, data, response, metrics);

    std::cout << "└─ Status: " << (success ? "PASSED ✓" : "FAILED ✗") << std::endl;

    cactus_destroy(model);
    return success;
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
    std::cout << "├─ Total tokens: " << data1.token_count << std::endl;
    metrics1.print();

    bool success1 = result1 > 0 && data1.token_count > 0;
    std::cout << "└─ Status: " << (success1 ? "PASSED ✓" : "FAILED ✗") << std::endl;

    if (!success1) {
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
    std::cout << "├─ Total tokens: " << data2.token_count << std::endl;
    metrics2.print();

    bool success2 = result2 > 0 && data2.token_count > 0;
    std::cout << "└─ Status: " << (success2 ? "PASSED ✓" : "FAILED ✗") << std::endl;

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

    return run_test("TOOL CALL TEST", messages,
        [](int result, const StreamingData& data, const std::string& response, const Metrics& m) {
            bool has_function = response.find("function_call") != std::string::npos;
            bool has_tool = response.find("get_weather") != std::string::npos;

            std::cout << "├─ Function call: " << (has_function ? "YES ✓" : "NO ✗") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES ✓" : "NO ✗") << "\n"
                      << "├─ Total tokens: " << data.token_count << std::endl;
            m.print();

            return result > 0 && has_function && has_tool;
        }, tools);
}

bool test_tool_call_with_multiple_tools() {
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

    return run_test("MULTIPLE TOOLS TEST", messages,
        [](int result, const StreamingData& data, const std::string& response, const Metrics& m) {
            bool has_function = response.find("function_call") != std::string::npos;
            bool has_tool = response.find("set_alarm") != std::string::npos;

            std::cout << "├─ Function call: " << (has_function ? "YES ✓" : "NO ✗") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES ✓" : "NO ✗") << "\n"
                      << "├─ Total tokens: " << data.token_count << std::endl;
            m.print();

            return result > 0 && has_function && has_tool;
        }, tools);
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
    cactus_embed(model, texts[0], emb1.data(), emb1.size() * sizeof(float), &dim1);
    double time1 = t1.elapsed_ms();

    Timer t2;
    cactus_embed(model, texts[1], emb2.data(), emb2.size() * sizeof(float), &dim2);
    double time2 = t2.elapsed_ms();

    float dot = 0, norm1 = 0, norm2 = 0;
    for (size_t i = 0; i < dim1; ++i) {
        dot += emb1[i] * emb2[i];
        norm1 += emb1[i] * emb1[i];
        norm2 += emb2[i] * emb2[i];
    }
    float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));

    std::cout << "\n[Results]\n"
              << "├─ Embedding dim: " << dim1 << "\n"
              << "├─ Time (text1): " << std::fixed << std::setprecision(2) << time1 << " ms\n"
              << "├─ Time (text2): " << time2 << " ms\n"
              << "├─ Similarity: " << std::setprecision(4) << similarity << "\n"
              << "└─ Status: PASSED ✓" << std::endl;

    cactus_destroy(model);
    return true;
}

bool test_huge_context() {
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
        [](int result, const StreamingData& data, const std::string&, const Metrics& m) {
            std::cout << "├─ Tokens generated: " << data.token_count << std::endl;
            m.print();
            std::cout << "├─ Early stop: " << (data.token_count == 100 ? "SUCCESS ✓" : "N/A") << std::endl;
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
        std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << "rag_preprocessing"
                  << " │ " << "model variant is not RAG (skipping)" << "\n";
        return true;
    }

    const char* corpus_dir = "../../tests/assets/rag_corpus";

    cactus_model_t model = cactus_init(g_model_path, 2048, corpus_dir);
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

    std::cout << "RAG PREPROCESSING: total tokens=" << data.token_count << " result=" << result << "\n";
    metrics.print();

    bool success = (result > 0) && (data.token_count > 0);

    cactus_destroy(model);
    return success;
}

bool test_audio_processor() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         Audio Processor Test             ║\n"
              << "╚══════════════════════════════════════════╝\n";
    using namespace cactus::engine;

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

    return passed;
}

int main() {
    TestUtils::TestRunner runner("Engine Tests");
    runner.run_test("streaming", test_streaming());
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("tool_calls_with_multiple_tools", test_tool_call_with_multiple_tools());
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("audio_processor", test_audio_processor());
    runner.run_test("rag_preprocessing", test_rag());
    runner.run_test("huge_context", test_huge_context());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}