#include "test_utils.h"
#include "../src/utils.h"
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>

#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define CACTUS_ENGINE_TEST_HAS_CURL 1
#else
#define CACTUS_ENGINE_TEST_HAS_CURL 0
#endif

using namespace EngineTestUtils;
using namespace cactus::engine;
using cactus::ffi::partition_thinking_response;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static bool check_partition(const std::string& input,
                            const std::string& expected_thinking,
                            const std::string& expected_content) {
    std::string thinking, content;
    partition_thinking_response(input, thinking, content);
    if (thinking != expected_thinking) {
        std::cerr << "  thinking: '" << thinking << "' != '" << expected_thinking << "'\n";
        return false;
    }
    if (content != expected_content) {
        std::cerr << "  content: '" << content << "' != '" << expected_content << "'\n";
        return false;
    }
    return true;
}

static cactus_model_t load_gemma4_or_skip() {
    if (!g_model_path) { std::cout << "  [WARN] CACTUS_TEST_MODEL not set; skipping\n"; return nullptr; }
    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cout << "  [WARN] Could not load model; skipping\n"; return nullptr; }
    if (static_cast<CactusModelHandle*>(model)->model->get_config().model_type != Config::ModelType::GEMMA4) {
        std::cout << "  [WARN] chosen model is not Gemma4; skipping\n";
        cactus_destroy(model);
        return nullptr;
    }
    return model;
}

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
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
                                 g_options, nullptr, stream_callback, &data1, nullptr, 0);

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
                                 g_options, nullptr, stream_callback, &data2, nullptr, 0);

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
    int prefill_result1 = cactus_prefill(model, messages, prefill_response1, sizeof(prefill_response1), nullptr, tools, nullptr, 0);

    PrefillMetrics prefill_metrics1;
    prefill_metrics1.parse(prefill_response1);

    char prefill_response2[2048] = {0};
    int prefill_result2 = cactus_prefill(model, messages, prefill_response2, sizeof(prefill_response2), nullptr, tools, nullptr, 0);

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
    int prefill_result1 = cactus_prefill(model, messages_base, prefill_response1, sizeof(prefill_response1), nullptr, tools, nullptr, 0);
    PrefillMetrics prefill_metrics1;
    prefill_metrics1.parse(prefill_response1);

    char prefill_response2[2048] = {0};
    int prefill_result2 = cactus_prefill(model, messages_extended, prefill_response2, sizeof(prefill_response2), nullptr, tools, nullptr, 0);
    PrefillMetrics prefill_metrics2;
    prefill_metrics2.parse(prefill_response2);

    cactus_reset(model);

    char prefill_response3[2048] = {0};
    int prefill_result3 = cactus_prefill(model, messages_extended, prefill_response3, sizeof(prefill_response3), nullptr, tools, nullptr, 0);
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
    int prefill_result = cactus_prefill(model, prefill_messages, prefill_response, sizeof(prefill_response), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages, complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr, nullptr, 0);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages, complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr, nullptr, 0);
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
    int prefill_result = cactus_prefill(model, prefill_messages, prefill_response, sizeof(prefill_response), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages, complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr, nullptr, 0);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages, complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr, nullptr, 0);
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
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in San Francisco?");
}

bool test_multiple_tool_call_invocations() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to Bob and get the weather for San Francisco."}
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

    return EngineTestUtils::run_test("MULTIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_weather_tool = has_function
                && (response.find("\"name\":\"get_weather\"") != std::string::npos
                    || response.find("\"name\": \"get_weather\"") != std::string::npos);
            bool has_message_tool = has_function
                && (response.find("\"name\":\"send_message\"") != std::string::npos
                    || response.find("\"name\": \"send_message\"") != std::string::npos);
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_weather_tool && has_message_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_weather_tool && has_message_tool;
        }, tools, -1, "Send a message to Bob and get the weather for San Francisco.");
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
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Send a message to John saying hello.");
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

bool test_partition_thinking_response() {
    return check_partition("<|channel>reason<channel|>answer", "reason", "answer")
        && check_partition("<|channel>\n  reason\n<channel|>\n\nanswer", "reason", "answer")
        && check_partition("no tags here", "", "no tags here")
        && check_partition("<think>reason</think>answer", "reason", "answer")
        && check_partition("<|channel>thought1<channel|>text1<|channel>thought2<channel|>text2",
                            "thought1\nthought2", "text1text2");
}

bool test_prompt_gemma4_retains_thinking() {
    cactus_model_t model = load_gemma4_or_skip();
    if (!model) return true;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tok = handle->model->get_tokenizer();

    std::vector<ChatMessage> msgs = {
        {"user", "hello", "", {}, {}, 0, {}},
        {"assistant", "<|channel>internal reasoning<channel|>visible response", "", {}, {}, 0, {}},
        {"user", "followup", "", {}, {}, 0, {}}
    };

    std::string prompt = tok->format_chat_prompt(msgs, true, "", true);
    cactus_destroy(model);

    bool has_visible = prompt.find("visible response") != std::string::npos;
    bool has_reasoning = prompt.find("internal reasoning") != std::string::npos;
    bool has_channel_tags = prompt.find("<|channel>") != std::string::npos
                         && prompt.find("<channel|>") != std::string::npos;

    if (!has_visible) std::cerr << "  missing visible response in prompt\n";
    if (!has_reasoning) std::cerr << "  thinking content not retained in assistant turn\n";
    if (!has_channel_tags) std::cerr << "  channel tags not retained in prompt\n";

    return has_visible && has_reasoning && has_channel_tags;
}

bool test_complete_gemma4_thinking_api_clean() {
    cactus_model_t model = load_gemma4_or_skip();
    if (!model) return true;

    const char* msgs = R"([{"role": "user", "content": "What is 2+2?"}])";
    char buf[8192];

    int r = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":128,"enable_thinking_if_supported":true,"telemetry_enabled":false})",
        nullptr, nullptr, nullptr, nullptr, 0);
    std::string resp(buf);
    cactus_destroy(model);

    std::string response = EngineTestUtils::json_string(resp, "response");
    bool ok = r > 0
           && resp.find("\"success\":true") != std::string::npos
           && response.find("<|channel>") == std::string::npos
           && response.find("<channel|>") == std::string::npos;
    if (!ok) std::cerr << "  thinking-enabled completion api not clean: " << resp << "\n";
    return ok;
}

bool test_multiturn_thinking_persist() {
    cactus_model_t model = load_gemma4_or_skip();
    if (!model) return true;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tokenizer = handle->model->get_tokenizer();
    const char* options = R"({"max_tokens":128,"temperature":0,"top_k":1,"enable_thinking_if_supported":true,"telemetry_enabled":false,"auto_handoff":false})";
    const char* turn1_msgs = R"([{"role": "user", "content": "My name is Alice. Please remember this."}])";
    char buf[16384];

    int r1 = cactus_complete(model, turn1_msgs, buf, sizeof(buf), options, nullptr, nullptr, nullptr, nullptr, 0);
    if (r1 <= 0) { std::cerr << "  Turn 1 failed\n"; cactus_destroy(model); return false; }

    std::vector<uint32_t> processed_after_t1 = handle->processed_tokens;
    std::string turn1_json(buf);
    std::string context_response = EngineTestUtils::json_string(turn1_json, "context_response");
    if (context_response.empty()) {
        std::cerr << "  context_response missing from turn 1 result\n";
        cactus_destroy(model);
        return false;
    }

    std::vector<ChatMessage> t2_chat = {
        {"user", "My name is Alice. Please remember this.", "", {}, {}, 0, {}},
        {"assistant", context_response, "", {}, {}, 0, {}},
        {"user", "What is my name?", "", {}, {}, 0, {}}
    };
    std::vector<uint32_t> t2_prompt_tokens = tokenizer->encode(tokenizer->format_chat_prompt(t2_chat, true, "", true));

    bool prefix_ok = (t2_prompt_tokens.size() >= processed_after_t1.size()) &&
                     std::equal(processed_after_t1.begin(), processed_after_t1.end(), t2_prompt_tokens.begin());
    std::cout << "  Prefix match (cache reuse): " << (prefix_ok ? "YES" : "NO") << "\n";

    std::string escaped = EngineTestUtils::escape_json(context_response);
    std::string turn2_json = R"([{"role": "user", "content": "My name is Alice. Please remember this."},{"role": "assistant", "content": ")"
        + escaped + R"("},{"role": "user", "content": "What is my name?"}])";

    int r2 = cactus_complete(model, turn2_json.c_str(), buf, sizeof(buf), options, nullptr, nullptr, nullptr, nullptr, 0);
    if (r2 <= 0) { std::cerr << "  Turn 2 failed\n"; cactus_destroy(model); return false; }

    std::string turn2_result(buf);
    std::string turn2_response = EngineTestUtils::json_string(turn2_result, "response");
    bool mentions_alice = turn2_response.find("Alice") != std::string::npos
                       || turn2_response.find("alice") != std::string::npos;
    std::cout << "  Turn 2 mentions Alice: " << (mentions_alice ? "YES" : "NO") << "\n";

    cactus_destroy(model);

    if (!prefix_ok) std::cerr << "  FAIL: re-rendered history did not prefix-match cache\n";
    if (!mentions_alice) std::cerr << "  FAIL: turn 2 did not recall the name\n";
    return prefix_ok && mentions_alice;
}

int main() {
    TestUtils::TestRunner runner("LLM Tests");
    runner.run_test("1k_context", test_1k_context());
    runner.run_test("streaming", test_streaming());
    runner.run_test("prefill", test_prefill());
    runner.run_test("prefill_idempotent_reuse", test_prefill_idempotent_reuse());
    runner.run_test("prefill_prefix_extension_reuse", test_prefill_prefix_extension_reuse());
    runner.run_test("prefill_invalidated_on_message_change", test_prefill_invalidated_on_message_change());
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("tool_multiple_tool_call_invocations", test_multiple_tool_call_invocations());
    runner.run_test("tool_calls_with_three_tools", test_tool_call_with_three_tools());
    runner.run_test("partition_thinking_response", test_partition_thinking_response());
    runner.run_test("prompt_retains_thinking", test_prompt_gemma4_retains_thinking());
    runner.run_test("complete_thinking_api_clean", test_complete_gemma4_thinking_api_clean());
    runner.run_test("multiturn_thinking_persist", test_multiturn_thinking_persist());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
