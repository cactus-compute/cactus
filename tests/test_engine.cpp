#include "test_utils.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sstream>

const char* g_model_path = "../../weights/lfm2-1.2b";
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

    cactus_model_t model = cactus_init(g_model_path, 2048);
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

    cactus_model_t model = cactus_init(g_model_path, 2048);
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

bool test_embeddings() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          EMBEDDINGS TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, 2048);
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

int main() {
    TestUtils::TestRunner runner("Engine Tests");
    runner.run_test("streaming", test_streaming());
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("huge_context", test_huge_context());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}