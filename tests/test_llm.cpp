#include "test_utils.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define CACTUS_ENGINE_TEST_HAS_CURL 1
#else
#define CACTUS_ENGINE_TEST_HAS_CURL 0
#endif

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false,
    "confidence_threshold": 0.0
    })";

namespace {

bool is_needle_model() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }

    if (!g_model_path || *g_model_path == '\0') {
        cached = 0;
        return false;
    }

    const std::string model_path_str = g_model_path;
    if (model_path_str.find("needle") != std::string::npos) {
        cached = 1;
        return true;
    }

    std::ifstream config(std::filesystem::path(g_model_path) / "config.txt");
    if (!config.is_open()) {
        cached = 0;
        return false;
    }

    std::string line;
    while (std::getline(config, line)) {
        if (line == "model_type=needle") {
            cached = 1;
            return true;
        }
    }

    cached = 0;
    return false;
}

std::string make_chat_messages(const std::string& user_content, const char* system_content = nullptr) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;

    if (system_content && *system_content) {
        oss << "{\"role\":\"system\",\"content\":\"" << escape_json(system_content) << "\"}";
        first = false;
    }

    if (!first) {
        oss << ",";
    }
    oss << "{\"role\":\"user\",\"content\":\"" << escape_json(user_content) << "\"}";
    oss << "]";
    return oss.str();
}

std::string make_tool_test_messages(const std::string& user_content) {
    if (is_needle_model()) {
        return make_chat_messages(user_content);
    }
    return make_chat_messages(user_content, "You are a helpful assistant that can use tools.");
}

} // namespace

template<typename TestFunc>
bool run_test(const char* title, const char* messages, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {
    return EngineTestUtils::run_test(title, g_model_path, messages, g_options, test_logic, tools, stop_at);
}

bool test_streaming() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      STREAMING & FOLLOW-UP TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
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
    metrics1.print_json();

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
    metrics2.print_json();

    bool success2 = result2 > 0 && data2.token_count > 0;

    cactus_destroy(model);
    return success1 && success2;
}

bool test_prefill_idempotent_reuse() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "     PREFILL IDEMPOTENT REUSE TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Write one short sentence about brainrot."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Summarize a topic in one short sentence",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to summarize"}
                },
                "required": ["topic"]
            }
        }
    }])";

    char prefill_response1[2048] = {0};
    int prefill_result1 = cactus_prefill(model, messages, prefill_response1, sizeof(prefill_response1), nullptr, tools);

    PrefillMetrics prefill_metrics1;
    prefill_metrics1.parse(prefill_response1);

    char prefill_response2[2048] = {0};
    int prefill_result2 = cactus_prefill(model, messages, prefill_response2, sizeof(prefill_response2), nullptr, tools);

    PrefillMetrics prefill_metrics2;
    prefill_metrics2.parse(prefill_response2);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill#1 benchmark: ";
    prefill_metrics1.print_line();
    std::cout << "\n"
              << "├─ Prefill#2 benchmark: ";
    prefill_metrics2.print_line();
    std::cout << "\n";

    bool prefill_success = prefill_result1 > 0 && prefill_result2 > 0
        && prefill_metrics1.success && prefill_metrics2.success;
    bool skipped_recompute = prefill_metrics2.prefill_tokens == 0;

    std::cout << "├─ Prefill calls success: " << (prefill_success ? "YES" : "NO") << "\n"
              << "└─ Second prefill skipped recompute: " << (skipped_recompute ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return prefill_success && skipped_recompute;
}

bool test_prefill_prefix_extension_reuse() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "   PREFILL PREFIX EXTENSION TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* messages_base = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Write one short sentence about brainrot."}
    ])";

    const char* messages_extended = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Write one short sentence about brainrot."},
        {"role": "assistant", "content": "Brainrot is internet slang for obsessive, meme-heavy online fixation."},
        {"role": "user", "content": "Now rewrite that in six words."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Summarize a topic in one short sentence",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to summarize"}
                },
                "required": ["topic"]
            }
        }
    }])";

    char prefill_response1[2048] = {0};
    int prefill_result1 = cactus_prefill(model, messages_base, prefill_response1, sizeof(prefill_response1), nullptr, tools);
    PrefillMetrics prefill_metrics1;
    prefill_metrics1.parse(prefill_response1);

    char prefill_response2[2048] = {0};
    int prefill_result2 = cactus_prefill(model, messages_extended, prefill_response2, sizeof(prefill_response2), nullptr, tools);
    PrefillMetrics prefill_metrics2;
    prefill_metrics2.parse(prefill_response2);

    cactus_reset(model);

    char prefill_response3[2048] = {0};
    int prefill_result3 = cactus_prefill(model, messages_extended, prefill_response3, sizeof(prefill_response3), nullptr, tools);
    PrefillMetrics prefill_metrics3;
    prefill_metrics3.parse(prefill_response3);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill#1 (base): ";
    prefill_metrics1.print_line();
    std::cout << "\n"
              << "├─ Prefill#2 (extended, warm): ";
    prefill_metrics2.print_line();
    std::cout << "\n"
              << "├─ Prefill#3 (extended, cold): ";
    prefill_metrics3.print_line();
    std::cout << "\n";

    bool prefill_success = prefill_result1 > 0 && prefill_result2 > 0 && prefill_result3 > 0
        && prefill_metrics1.success && prefill_metrics2.success && prefill_metrics3.success;
    bool second_call_prefilled = prefill_metrics2.prefill_tokens > 0;
    bool warm_reused_prefix = prefill_metrics2.prefill_tokens < prefill_metrics3.prefill_tokens;

    std::cout << "├─ Prefill calls success: " << (prefill_success ? "YES" : "NO") << "\n"
              << "├─ Warm extension prefilled tokens: " << (second_call_prefilled ? "YES" : "NO") << "\n"
              << "└─ Warm extension < cold extension: " << (warm_reused_prefix ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return prefill_success && second_call_prefilled && warm_reused_prefix;
}

bool test_prefill_invalidated_on_message_change() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << " PREFILL INVALIDATION (LLM) TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* prefill_messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Summarize the phrase 'brainrot' in one sentence."}
    ])";

    const char* complete_messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Give one sentence about the power of the 'brainrot'."}
    ])";

    const char* options = R"({
        "max_tokens": 128,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "confidence_threshold": 0.0,
        "telemetry_enabled": false
    })";

    char prefill_response[2048] = {0};
    int prefill_result = cactus_prefill(model, prefill_messages, prefill_response, sizeof(prefill_response), nullptr, nullptr);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages, complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages, complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr);
    Metrics cold_metrics;
    cold_metrics.parse(complete_response_cold);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill success: " << ((prefill_result > 0 && prefill_metrics.success) ? "YES" : "NO") << "\n"
              << "├─ Complete(warm mismatched) prefill_tokens: " << warm_metrics.prefill_tokens << "\n"
              << "├─ Complete(cold) prefill_tokens: " << cold_metrics.prefill_tokens << "\n";

    bool all_success = prefill_result > 0 && prefill_metrics.success
        && complete_result_warm > 0 && warm_metrics.success
        && complete_result_cold > 0 && cold_metrics.success;
    bool invalidated = warm_metrics.prefill_tokens == cold_metrics.prefill_tokens;

    std::cout << "├─ Calls successful: " << (all_success ? "YES" : "NO") << "\n"
              << "└─ Mismatch invalidated cache: " << (invalidated ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return all_success && invalidated;
}

bool test_prefill() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "          PREFILL API TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* prefill_messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Explain what brainrot means in one short sentence."},
        {"role": "assistant", "content": "Brainrot is internet slang for obsessive, meme-heavy online fixation."}
    ])";

    const char* complete_messages = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Explain what brainrot means in one short sentence."},
        {"role": "assistant", "content": "Brainrot is internet slang for obsessive, meme-heavy online fixation."},
        {"role": "user", "content": "Now rewrite that in six words."}
    ])";

    const char* options = R"({
        "max_tokens": 128,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "confidence_threshold": 0.0,
        "telemetry_enabled": false
    })";

    char prefill_response[2048] = {0};
    int prefill_result = cactus_prefill(model, prefill_messages, prefill_response, sizeof(prefill_response), nullptr, nullptr);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages, complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages, complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr);
    Metrics cold_metrics;
    cold_metrics.parse(complete_response_cold);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill success: " << ((prefill_result > 0 && prefill_metrics.success) ? "YES" : "NO") << "\n"
              << "├─ Prefill metrics: ";
    prefill_metrics.print_line();
    std::cout << "\n";
    std::cout << "├─ Complete warm metrics:\n";
    warm_metrics.print_json();
    std::cout << "├─ Complete cold metrics:\n";
    cold_metrics.print_json();

    bool all_success = prefill_result > 0 && prefill_metrics.success
        && complete_result_warm > 0 && warm_metrics.success
        && complete_result_cold > 0 && cold_metrics.success;
    bool warm_prefilled_less = warm_metrics.prefill_tokens < cold_metrics.prefill_tokens;

    std::cout << "├─ Calls successful: " << (all_success ? "YES" : "NO") << "\n"
              << "└─ Warm prefilled less than cold: " << (warm_prefilled_less ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return all_success && warm_prefilled_less;
}

bool test_tool_call() {
    std::string messages = make_tool_test_messages("What's the weather in San Francisco?");

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
        "force_tools": true,
        "confidence_threshold": 0.0
    })";

    return EngineTestUtils::run_test("TOOL CALL TEST", g_model_path, messages.c_str(), options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in San Francisco?");
}

bool test_multiple_tool_call_invocations() {
    std::string messages = make_tool_test_messages("Send a message to Bob and get the weather for San Francisco.");

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
        "force_tools": true,
        "confidence_threshold": 0.0
    })";

    return EngineTestUtils::run_test("MULTIPLE TOOLS TEST", g_model_path, messages.c_str(), options_with_force_tools,
        [](int result, const StreamingData& data, const std::string& response, const Metrics& m) {
            std::string raw;
            for (const auto& t : data.tokens) raw += t;
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_weather_tool = raw.find("get_weather") != std::string::npos;
            bool has_message_tool = raw.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_weather_tool && has_message_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_weather_tool && has_message_tool;
        }, tools, -1, "Send a message to Bob and get the weather for San Francisco.");
}

bool test_tool_call_with_three_tools() {
    std::string messages = make_tool_test_messages("Send a message to John saying hello.");

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
        "force_tools": true,
        "confidence_threshold": 0.0
    })";

    return EngineTestUtils::run_test("TRIPLE TOOLS TEST", g_model_path, messages.c_str(), options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Send a message to John saying hello.");
}

bool test_tool_no_params() {
    std::string messages = make_tool_test_messages("What time is it right now?");

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }])";

    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL NO PARAMS TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_current_time") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What time is it right now?");
}

bool test_tool_optional_params_only() {
    std::string messages = make_tool_test_messages("Search for news.");

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for news articles",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results to return"}
                }
            }
        }
    }])";

    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL OPTIONAL PARAMS TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("search_news") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Search for news.");
}

bool test_tool_integer_and_enum_params() {
    std::string messages = make_tool_test_messages("Set an alarm for 7:30 AM.");

    const char* tools = R"json([{
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a specific time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour 0-23"},
                    "minute": {"type": "integer", "description": "Minute 0-59"},
                    "period": {"type": "string", "enum": ["AM", "PM"], "description": "AM or PM"}
                },
                "required": ["hour", "minute"]
            }
        }
    }])json";

    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL INTEGER+ENUM PARAMS TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("set_alarm") != std::string::npos;
            bool has_hour = has_tool && response.find("hour") != std::string::npos;
            bool has_minute = has_tool && response.find("minute") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'hour': " << (has_hour ? "YES" : "NO") << "\n"
                      << "├─ Required 'minute': " << (has_minute ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_hour && has_minute;
        }, tools, -1, "Set an alarm for 7:30 AM.");
}

bool test_tool_many_required_params() {
    std::string messages = make_tool_test_messages("Book a flight from New York to London for John Smith, departing 2025-06-01.");

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city or airport code"},
                    "destination": {"type": "string", "description": "Arrival city or airport code"},
                    "passenger_name": {"type": "string", "description": "Full name of the passenger"},
                    "departure_date": {"type": "string", "description": "Date of departure in YYYY-MM-DD format"},
                    "seat_class": {"type": "string", "description": "Class of seat: economy, business, or first"}
                },
                "required": ["origin", "destination", "passenger_name", "departure_date"]
            }
        }
    }])";

    const char* options = R"({"max_tokens": 256, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL MANY REQUIRED PARAMS TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("book_flight") != std::string::npos;
            bool has_origin = has_tool && response.find("origin") != std::string::npos;
            bool has_dest = has_tool && response.find("destination") != std::string::npos;
            bool has_name = has_tool && response.find("passenger_name") != std::string::npos;
            bool has_date = has_tool && response.find("departure_date") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'origin': " << (has_origin ? "YES" : "NO") << "\n"
                      << "├─ Required 'destination': " << (has_dest ? "YES" : "NO") << "\n"
                      << "├─ Required 'passenger_name': " << (has_name ? "YES" : "NO") << "\n"
                      << "├─ Required 'departure_date': " << (has_date ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_origin && has_dest && has_name && has_date;
        }, tools, -1, "Book a flight from New York to London for John Smith, departing 2025-06-01.");
}

bool test_tool_nested_object_params() {
    std::string messages = make_tool_test_messages("Create a calendar event called Team Sync tomorrow at 3pm.");

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "time": {
                        "type": "object",
                        "description": "Event time",
                        "properties": {
                            "hour": {"type": "integer"},
                            "minute": {"type": "integer"}
                        }
                    },
                    "duration_minutes": {"type": "integer", "description": "Duration in minutes"}
                },
                "required": ["title"]
            }
        }
    }])";

    const char* options = R"({"max_tokens": 256, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL NESTED OBJECT TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("create_calendar_event") != std::string::npos;
            bool has_title = has_tool && response.find("title") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'title': " << (has_title ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_title;
        }, tools, -1, "Create a calendar event called Team Sync tomorrow at 3pm.");
}

bool test_tool_pick_right_tool() {
    std::string messages = make_tool_test_messages("Translate 'good morning' to French.");

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text from one language to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to translate"},
                    "target_language": {"type": "string", "description": "Language to translate into"}
                },
                "required": ["text", "target_language"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }])";

    const char* options = R"({"max_tokens": 256, "force_tools": true, "confidence_threshold": 0.0})";

    return EngineTestUtils::run_test("TOOL PICK RIGHT TOOL TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_right_tool = has_function && response.find("translate_text") != std::string::npos;
            bool has_text = has_right_tool && response.find("\"text\"") != std::string::npos;
            bool has_lang = has_right_tool && response.find("target_language") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (translate_text): " << (has_right_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'text': " << (has_text ? "YES" : "NO") << "\n"
                      << "├─ Required 'target_language': " << (has_lang ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_right_tool && has_text && has_lang;
        }, tools, -1, "Translate 'good morning' to French.");
}

// --- broader constraint tests ---

bool test_tool_5_tools_select() {
    std::string messages = make_tool_test_messages("What's the weather in Berlin?");
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}},
        {"type":"function","function":{"name":"send_message","description":"Send a message to someone","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"message":{"type":"string"}},"required":["recipient","message"]}}},
        {"type":"function","function":{"name":"set_alarm","description":"Set an alarm","parameters":{"type":"object","properties":{"hour":{"type":"integer"},"minute":{"type":"integer"}},"required":["hour","minute"]}}},
        {"type":"function","function":{"name":"translate_text","description":"Translate text to another language","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"]}}},
        {"type":"function","function":{"name":"lookup_contact","description":"Find a contact by name","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("5 TOOLS SELECT TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            bool has_location = has_tool && response.find("location") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_weather): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'location': " << (has_location ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_location;
        }, tools, -1, "What's the weather in Berlin?");
}

bool test_tool_10_tools_haystack() {
    std::string messages = make_tool_test_messages("What is the current stock price for AAPL?");
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
        {"type":"function","function":{"name":"send_message","description":"Send a message","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"message":{"type":"string"}},"required":["recipient","message"]}}},
        {"type":"function","function":{"name":"set_alarm","description":"Set an alarm","parameters":{"type":"object","properties":{"hour":{"type":"integer"},"minute":{"type":"integer"}},"required":["hour","minute"]}}},
        {"type":"function","function":{"name":"translate_text","description":"Translate text","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"]}}},
        {"type":"function","function":{"name":"get_stock_price","description":"Get the current stock price for a ticker symbol","parameters":{"type":"object","properties":{"ticker":{"type":"string","description":"Stock ticker symbol"}},"required":["ticker"]}}},
        {"type":"function","function":{"name":"create_reminder","description":"Create a reminder","parameters":{"type":"object","properties":{"title":{"type":"string"},"time":{"type":"string"}},"required":["title"]}}},
        {"type":"function","function":{"name":"search_news","description":"Search news articles","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
        {"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
        {"type":"function","function":{"name":"play_music","description":"Play music","parameters":{"type":"object","properties":{"artist":{"type":"string"},"song":{"type":"string"}},"required":[]}}},
        {"type":"function","function":{"name":"lookup_contact","description":"Find a contact","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("10 TOOLS HAYSTACK TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_stock_price") != std::string::npos;
            bool has_ticker = has_tool && response.find("ticker") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_stock_price): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'ticker': " << (has_ticker ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_ticker;
        }, tools, -1, "What is the current stock price for AAPL?");
}

bool test_tool_buried_key_info() {
    std::string messages = make_tool_test_messages(
        "My name is Sarah and I love traveling. I visited many cities last year. "
        "Right now I am planning my next trip and I will be going to Amsterdam. "
        "I enjoy cycling and Dutch food. I want to pack the right clothes. "
        "What is the weather like where I am going?"
    );
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}},
        {"type":"function","function":{"name":"send_message","description":"Send a message","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"message":{"type":"string"}},"required":["recipient","message"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("BURIED KEY INFO TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            bool has_location = has_tool && response.find("location") != std::string::npos;
            bool has_amsterdam = has_location && response.find("Amsterdam") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_weather): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'location': " << (has_location ? "YES" : "NO") << "\n"
                      << "├─ Extracted Amsterdam: " << (has_amsterdam ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_location;
        }, tools, -1, "...(buried Amsterdam)...");
}

bool test_tool_long_irrelevant_preamble() {
    std::string messages = make_tool_test_messages(
        "I am a software engineer working on a distributed systems project. "
        "We use microservices and deploy to Kubernetes. Our team is spread across "
        "three time zones. We have daily standups and use Jira for project tracking. "
        "The backend is written in Go and the frontend uses React with TypeScript. "
        "We recently migrated from a monolith and have been dealing with service mesh issues. "
        "Anyway, none of that is relevant right now. Get the weather in Sydney."
    );
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
        {"type":"function","function":{"name":"search_news","description":"Search news articles","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("LONG IRRELEVANT PREAMBLE TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            bool has_location = has_tool && response.find("location") != std::string::npos;
            bool has_sydney = has_location && response.find("Sydney") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_weather): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'location': " << (has_location ? "YES" : "NO") << "\n"
                      << "├─ Extracted Sydney: " << (has_sydney ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_location;
        }, tools, -1, "...(long preamble, get weather Sydney at end)...");
}

bool test_tool_boolean_param() {
    std::string messages = make_tool_test_messages("Send an urgent alert to the on-call engineer.");
    const char* tools = R"([
        {"type":"function","function":{"name":"send_alert","description":"Send an alert notification","parameters":{"type":"object","properties":{"recipient":{"type":"string","description":"Who to alert"},"urgent":{"type":"boolean","description":"Whether the alert is urgent"},"message":{"type":"string","description":"Alert message"}},"required":["recipient","urgent"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("BOOLEAN PARAM TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("send_alert") != std::string::npos;
            bool has_recipient = has_tool && response.find("recipient") != std::string::npos;
            bool has_urgent = has_tool && response.find("urgent") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (send_alert): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'recipient': " << (has_recipient ? "YES" : "NO") << "\n"
                      << "├─ Required 'urgent': " << (has_urgent ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_recipient && has_urgent;
        }, tools, -1, "Send an urgent alert to the on-call engineer.");
}

bool test_tool_similar_names_disambiguation() {
    std::string messages = make_tool_test_messages("What is the weather like right now in Oslo?");
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get the current weather conditions for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
        {"type":"function","function":{"name":"get_weather_forecast","description":"Get a multi-day weather forecast for a location","parameters":{"type":"object","properties":{"location":{"type":"string"},"days":{"type":"integer"}},"required":["location"]}}},
        {"type":"function","function":{"name":"get_weather_history","description":"Get historical weather data for a location and date","parameters":{"type":"object","properties":{"location":{"type":"string"},"date":{"type":"string"}},"required":["location","date"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("SIMILAR NAMES DISAMBIGUATION TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("\"get_weather\"") != std::string::npos;
            bool wrong_tool = response.find("get_weather_forecast") != std::string::npos
                           || response.find("get_weather_history") != std::string::npos;
            bool has_location = has_tool && response.find("location") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_weather, not forecast/history): " << (has_tool && !wrong_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'location': " << (has_location ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && !wrong_tool && has_location;
        }, tools, -1, "What is the weather like right now in Oslo?");
}

bool test_tool_three_required_numeric() {
    std::string messages = make_tool_test_messages("Convert 500 dollars to Japanese yen.");
    const char* tools = R"([
        {"type":"function","function":{"name":"convert_currency","description":"Convert an amount from one currency to another","parameters":{"type":"object","properties":{"amount":{"type":"number","description":"The amount to convert"},"from_currency":{"type":"string","description":"Source currency code"},"to_currency":{"type":"string","description":"Target currency code"}},"required":["amount","from_currency","to_currency"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("THREE REQUIRED NUMERIC TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("convert_currency") != std::string::npos;
            bool has_amount = has_tool && response.find("amount") != std::string::npos;
            bool has_from = has_tool && response.find("from_currency") != std::string::npos;
            bool has_to = has_tool && response.find("to_currency") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (convert_currency): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'amount': " << (has_amount ? "YES" : "NO") << "\n"
                      << "├─ Required 'from_currency': " << (has_from ? "YES" : "NO") << "\n"
                      << "├─ Required 'to_currency': " << (has_to ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_amount && has_from && has_to;
        }, tools, -1, "Convert 500 dollars to Japanese yen.");
}

bool test_tool_non_obvious_param_names() {
    std::string messages = make_tool_test_messages("Log that user alice just logged in to the system.");
    const char* tools = R"([
        {"type":"function","function":{"name":"log_activity","description":"Record a user activity event in the audit log","parameters":{"type":"object","properties":{"user_identifier":{"type":"string","description":"The username or ID of the user"},"activity_code":{"type":"string","description":"Short code for the activity type, e.g. LOGIN, LOGOUT"}},"required":["user_identifier","activity_code"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("NON-OBVIOUS PARAM NAMES TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("log_activity") != std::string::npos;
            bool has_uid = has_tool && response.find("user_identifier") != std::string::npos;
            bool has_code = has_tool && response.find("activity_code") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (log_activity): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'user_identifier': " << (has_uid ? "YES" : "NO") << "\n"
                      << "├─ Required 'activity_code': " << (has_code ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_uid && has_code;
        }, tools, -1, "Log that user alice just logged in to the system.");
}

bool test_tool_1k_context_with_tool() {
    std::string long_context =
        "I am a software engineer. I work on backend systems. "
        "We use Go, Postgres, and Redis. Our team has 8 engineers. "
        "We deploy to AWS using ECS and Terraform. We have CI/CD with GitHub Actions. "
        "Our services handle about 10 million requests per day. We use Datadog for monitoring. "
        "We recently added a new feature for real-time notifications. We also improved our "
        "database indexing strategy which cut p99 latency by 40%. I enjoy the work. "
        "My manager is supportive and the team culture is good. We do code reviews for all PRs. "
        "Anyway, I need to check on something completely different. Get the weather in Cape Town.";
    std::string messages = make_tool_test_messages(long_context);
    const char* tools = R"([
        {"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
        {"type":"function","function":{"name":"get_stock_price","description":"Get stock price","parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}}
    ])";
    const char* options = R"({"max_tokens": 128, "force_tools": true, "confidence_threshold": 0.0})";
    return EngineTestUtils::run_test("1K CONTEXT WITH TOOL TEST", g_model_path, messages.c_str(), options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            bool has_location = has_tool && response.find("location") != std::string::npos;
            bool has_cape_town = has_location && (response.find("Cape Town") != std::string::npos
                                               || response.find("Cape") != std::string::npos);
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool (get_weather): " << (has_tool ? "YES" : "NO") << "\n"
                      << "├─ Required 'location': " << (has_location ? "YES" : "NO") << "\n"
                      << "├─ Extracted Cape Town: " << (has_cape_town ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool && has_location;
        }, tools, -1, "...(long context, get weather Cape Town at end)...");
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
            m.print_json();
            return result > 0;
        }, nullptr, 100);
}

int main() {
    TestUtils::TestRunner runner("LLM Tests");
    if (is_needle_model()) {
        runner.log_skip("1k_context", "generic QA prompt is off-format for Needle");
        runner.log_skip("streaming", "Needle test path is single-turn query + tools");
        runner.log_skip("prefill", "generic chat-format prefill test is not Needle-native");
        runner.log_skip("prefill_idempotent_reuse", "generic chat-format prefill test is not Needle-native");
        runner.log_skip("prefill_prefix_extension_reuse", "assistant-turn prefix test is not Needle-native");
        runner.log_skip("prefill_invalidated_on_message_change", "generic chat-format prefill test is not Needle-native");
    } else {
        runner.run_test("1k_context", test_1k_context());
        runner.run_test("streaming", test_streaming());
        runner.run_test("prefill", test_prefill());
        runner.run_test("prefill_idempotent_reuse", test_prefill_idempotent_reuse());
        runner.run_test("prefill_prefix_extension_reuse", test_prefill_prefix_extension_reuse());
        runner.run_test("prefill_invalidated_on_message_change", test_prefill_invalidated_on_message_change());
    }
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("tool_multiple_tool_call_invocations", test_multiple_tool_call_invocations());
    runner.run_test("tool_calls_with_three_tools", test_tool_call_with_three_tools());
    runner.run_test("tool_no_params", test_tool_no_params());
    runner.run_test("tool_optional_params_only", test_tool_optional_params_only());
    runner.run_test("tool_integer_and_enum_params", test_tool_integer_and_enum_params());
    runner.run_test("tool_many_required_params", test_tool_many_required_params());
    runner.run_test("tool_nested_object_params", test_tool_nested_object_params());
    runner.run_test("tool_pick_right_tool", test_tool_pick_right_tool());
    runner.run_test("tool_5_tools_select", test_tool_5_tools_select());
    runner.run_test("tool_10_tools_haystack", test_tool_10_tools_haystack());
    runner.run_test("tool_buried_key_info", test_tool_buried_key_info());
    runner.run_test("tool_long_irrelevant_preamble", test_tool_long_irrelevant_preamble());
    runner.run_test("tool_boolean_param", test_tool_boolean_param());
    runner.run_test("tool_similar_names_disambiguation", test_tool_similar_names_disambiguation());
    runner.run_test("tool_three_required_numeric", test_tool_three_required_numeric());
    runner.run_test("tool_non_obvious_param_names", test_tool_non_obvious_param_names());
    runner.run_test("tool_1k_context_with_tool", test_tool_1k_context_with_tool());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
