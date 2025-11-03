#include "test_utils.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>

const char* g_model_path = "../../weights/lfm2-350m";

const char* g_options = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
    })";

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;
    }
};

struct StreamingTestData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count;
};

void streaming_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingTestData*>(user_data);
    data->tokens.push_back(std::string(token));
    data->token_ids.push_back(token_id);
    data->token_count++;
    std::cout << token << std::flush;
}

bool test_streaming() {
    cactus_model_t model = cactus_init(g_model_path, 2048);

    const char* messages = R"([
        {"role": "system", "content": "/no_think You are a helpful assistant. Be concise and friendly in your responses."},
        {"role": "user", "content": "What is your name?"}
    ])";

    StreamingTestData stream_data;
    stream_data.token_count = 0;

    char response[2048];

    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║          STREAMING TEST                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;
    std::cout << "Response: ";
    int result = cactus_complete(model, messages, response, sizeof(response), g_options, nullptr,
                                streaming_callback, &stream_data);

    std::cout << "\n\n[✓] Stream completed" << std::endl;

    std::string response_str(response);
    double time_to_first_token = 0.0;
    double tokens_per_second = 0.0;

    size_t ttft_pos = response_str.find("\"time_to_first_token_ms\":");
    if (ttft_pos != std::string::npos) {
        ttft_pos += 25;
        time_to_first_token = std::stod(response_str.substr(ttft_pos));
    }

    size_t tps_pos = response_str.find("\"tokens_per_second\":");
    if (tps_pos != std::string::npos) {
        tps_pos += 20;
        tokens_per_second = std::stod(response_str.substr(tps_pos));
    }

    std::cout << "├─ Total tokens: " << stream_data.token_count << std::endl;
    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << time_to_first_token << " ms" << std::endl;
    std::cout << "├─ Tokens per second: " << std::fixed << std::setprecision(2) << tokens_per_second << std::endl;
    std::cout << "└─ Status: " << (result > 0 ? "SUCCESS" : "FAILED") << std::endl;
    
    cactus_destroy(model);

    return result > 0 && stream_data.token_count > 0;
}

bool test_embeddings() {
    cactus_model_t model = cactus_init(g_model_path, 2048);

    if (!model) {
        std::cerr << "[✗] Failed to initialize model" << std::endl;
        return false;
    }

    const char* text1 = "My name is Henry Ndubuaku";
    const char* text2 = "Your name is Henry Ndubuaku";

    std::vector<float> embeddings1(2048);
    std::vector<float> embeddings2(2048);
    size_t embedding_dim1 = 0;
    size_t embedding_dim2 = 0;

    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║          EMBEDDINGS TEST                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    Timer timer1;
    int result1 = cactus_embed(model, text1, embeddings1.data(),
                               embeddings1.size() * sizeof(float), &embedding_dim1);
    (void)result1;
    double time1 = timer1.elapsed_ms();

    Timer timer2;
    int result2 = cactus_embed(model, text2, embeddings2.data(),
                               embeddings2.size() * sizeof(float), &embedding_dim2);
    (void)result2;
    double time2 = timer2.elapsed_ms();

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < embedding_dim1; ++i) {
        dot_product += embeddings1[i] * embeddings2[i];
        norm1 += embeddings1[i] * embeddings1[i];
        norm2 += embeddings2[i] * embeddings2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    float cosine_similarity = dot_product / (norm1 * norm2);

    std::cout << "\n[Input Texts]" << std::endl;
    std::cout << "├─ Text 1: \"" << text1 << "\"" << std::endl;
    std::cout << "└─ Text 2: \"" << text2 << "\"" << std::endl;

    std::cout << "\n[Results]" << std::endl;
    std::cout << "├─ Embedding dimension: " << embedding_dim1 << std::endl;
    std::cout << "├─ Processing time (text1): " << std::fixed << std::setprecision(2) << time1 << " ms" << std::endl;
    std::cout << "├─ Processing time (text2): " << std::fixed << std::setprecision(2) << time2 << " ms" << std::endl;
    std::cout << "└─ Cosine similarity: " << std::fixed << std::setprecision(4) << cosine_similarity << std::endl;
    
    cactus_destroy(model);
    return true;
}

bool test_huge_context() {
    cactus_model_t model = cactus_init(g_model_path, 2048);

    if (!model) {
        std::cerr << "[✗] Failed to initialize model for control test" << std::endl;
        return false;
    }

    std::string large_messages = "[{\"role\": \"system\", \"content\": \"/no_think You are a helpful assistant with extensive knowledge. ";

    for (int i = 0; i < 100; i++) {
        large_messages += "Context item " + std::to_string(i) +
            ": This represents background knowledge about topic " + std::to_string(i) +
            " including facts, data, and relevant information. ";
    }
    large_messages += "\"}, {\"role\": \"user\", \"content\": \"";

    for (int i = 0; i < 100; i++) {
        large_messages += "Data point " + std::to_string(i) + " has value " +
            std::to_string(i * 3.14159) + " with properties [active, validated]. ";
    }

    large_messages += "Given all the above context, please count to 10.\"}]";

    char response[8192];  

    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║       HUGE CONTEXT CONTROL TEST            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    struct ControlData {
        cactus_model_t model;
        int token_count = 0;
    } control_data;

    control_data.model = model;

    std::cout << "\n[Test: Early stopping at 5 tokens]" << std::endl;
    std::cout << "Response: ";

    auto control_callback = [](const char* token, uint32_t token_id, void* user_data) {
        (void)token_id;
        auto* data = static_cast<ControlData*>(user_data);
        std::cout << token << std::flush;
        data->token_count++;

        if (data->token_count >= 5) {
            std::cout << "\n\n[→ Stopping generation at token #5]" << std::endl;
            cactus_stop(data->model);
        }
    };

    int result = cactus_complete(model, large_messages.c_str(), response, sizeof(response),
                                g_options, nullptr, control_callback, &control_data);

    std::cout << "\n[Results]" << std::endl;

    std::string response_str(response);
    double time_to_first_token = 0.0;
    double tokens_per_second = 0.0;

    size_t ttft_pos = response_str.find("\"time_to_first_token_ms\":");
    if (ttft_pos != std::string::npos) {
        ttft_pos += 25;
        time_to_first_token = std::stod(response_str.substr(ttft_pos));
    }

    size_t tps_pos = response_str.find("\"tokens_per_second\":");
    if (tps_pos != std::string::npos) {
        tps_pos += 20;
        tokens_per_second = std::stod(response_str.substr(tps_pos));
    }

    std::cout << "├─ Tokens generated: " << control_data.token_count << std::endl;
    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << time_to_first_token << " ms" << std::endl;
    std::cout << "├─ Tokens per second: " << std::fixed << std::setprecision(2) << tokens_per_second << std::endl;
    std::cout << "├─ Early stop: " << (control_data.token_count == 5 ? "SUCCESS ✓" : "FAILED ✗") << std::endl;
    std::cout << "└─ Status: " << (result > 0 ? "PASSED ✓" : "FAILED ✗") << std::endl;

    std::cout << "\n[Full Model Response]" << std::endl;
    std::cout << response << std::endl;  
    
    cactus_destroy(model);
    return result > 0;
}

bool test_tool_call() {
    cactus_model_t model = cactus_init(g_model_path, 2048);

    if (!model) {
        std::cerr << "[✗] Failed to initialize model for tools test" << std::endl;
        return false;
    }

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ])";

    const char* tools = R"([
        {
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for, in the format \"City, State, Country\"."
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ])";

    StreamingTestData stream_data;
    stream_data.token_count = 0;

    char response[4096];

    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║          TOOL CALL TEST                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    std::cout << "\n[Conversation]" << std::endl;
    std::cout << "├─ User: What's the weather in San Francisco?" << std::endl;
    std::cout << "└─ Assistant: ";

    int result = cactus_complete(model, messages, response, sizeof(response), g_options, tools,
                                streaming_callback, &stream_data);

    std::cout << "\n\n[Tool Call Analysis]" << std::endl;

    std::string response_str(response);
    bool has_function_call = response_str.find("function_call") != std::string::npos;
    bool has_tool_name = response_str.find("get_weather") != std::string::npos;

    double time_to_first_token = 0.0;
    double tokens_per_second = 0.0;

    size_t ttft_pos = response_str.find("\"time_to_first_token_ms\":");
    if (ttft_pos != std::string::npos) {
        ttft_pos += 25;
        time_to_first_token = std::stod(response_str.substr(ttft_pos));
    }

    size_t tps_pos = response_str.find("\"tokens_per_second\":");
    if (tps_pos != std::string::npos) {
        tps_pos += 20;
        tokens_per_second = std::stod(response_str.substr(tps_pos));
    }

    std::cout << "├─ Function call detected: " << (has_function_call ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << "├─ Correct tool selected: " << (has_tool_name ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << "├─ Total tokens: " << stream_data.token_count << std::endl;
    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << time_to_first_token << " ms" << std::endl;
    std::cout << "├─ Tokens per second: " << std::fixed << std::setprecision(2) << tokens_per_second << std::endl;
    std::cout << "└─ Overall status: " << (has_function_call && has_tool_name ? "PASSED ✓" : "FAILED ✗") << std::endl;

    std::cout << "\n[Full Model Response]" << std::endl;
    std::cout << response << std::endl;    
    
    cactus_destroy(model);
    return result > 0 && stream_data.token_count > 0;
}


int main() {
    TestUtils::TestRunner runner("Engine Tests");
    runner.run_test("streaming", test_streaming());  
    runner.run_test("embeddings", test_embeddings());
    // runner.run_test("huge_context", test_huge_context());
    runner.run_test("tool_calls", test_tool_call());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}