#include "test_utils.h"
#include "../src/utils.h"
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>

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

static cactus_model_t load_dynamic_batch_model_or_skip() {
    if (!g_model_path) { std::cout << "  [WARN] CACTUS_TEST_MODEL not set; skipping\n"; return nullptr; }
    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cout << "  [WARN] Could not load model; skipping\n"; return nullptr; }
    if (!static_cast<CactusModelHandle*>(model)->model->supports_dynamic_batch()) {
        std::cout << "  [WARN] model is not dynamic-batch capable (reconvert with `cactus convert`); skipping\n";
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
    bool recalled_name = metrics2.response.find("Henry") != std::string::npos
                      || metrics2.response.find("henry") != std::string::npos;
    std::cout << "├─ Turn 2 recalls name: " << (recalled_name ? "YES" : "NO") << "\n";

    cactus_destroy(model);
    return success1 && success2 && recalled_name;
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

struct LengthTokenizer : public Tokenizer {
    std::vector<uint32_t> encode(const std::string& text) const override {
        return {static_cast<uint32_t>(text.size())};
    }
    std::string decode(const std::vector<uint32_t>&) const override { return std::string(); }
    uint32_t get_vocab_size() const override { return 0; }
    uint32_t get_unk_token() const override { return 0; }
    uint32_t get_bos_token() const override { return 0; }
    uint32_t get_eos_token() const override { return 0; }
    bool load_vocabulary_with_config(const std::string&, const std::string&, const std::string&) override { return true; }
};

bool test_tool_constraint_clear_releases_bias() {
    LengthTokenizer tok;
    ToolCallConstrainer constrainer;
    constrainer.init(Config::ModelType::GEMMA4, {{"get_weather", {"location"}, {}, {"location"}}}, &tok);
    if (constrainer.get_bias().empty()) {
        std::cerr << "  expected bias after activating init\n";
        return false;
    }
    constrainer.reset();
    constrainer.init(Config::ModelType::GEMMA4, {}, &tok);
    if (!constrainer.get_bias().empty()) {
        std::cerr << "  stale bias survived deactivating init\n";
        return false;
    }
    return true;
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

bool test_partition_thinking_response() {
    return check_partition("<|channel>reason<channel|>answer", "reason", "answer")
        && check_partition("<|channel>\n  reason\n<channel|>\n\nanswer", "reason", "answer")
        && check_partition("no tags here", "", "no tags here")
        && check_partition("<think>reason</think>answer", "reason", "answer")
        && check_partition("<|channel>thought1<channel|>text1<|channel>thought2<channel|>text2",
                            "thought1\nthought2", "text1text2");
}

bool test_prompt_gemma4_agentic_turn_closure() {
    cactus_model_t model = load_gemma4_or_skip();
    if (!model) return true;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tok = handle->model->get_tokenizer();
    bool ok = true;

    std::vector<ChatMessage> consecutive = {
        {"user", "start", "", {}, {}, 0, {}},
        {"assistant", "first", "", {}, {}, 0, {}},
        {"assistant", "second", "", {}, {}, 0, {}}
    };
    std::string consecutive_prompt = tok->format_chat_prompt(consecutive, true, "", false);
    std::string consecutive_expected = "<|turn>model\nfirstsecond<turn|>\n<|turn>model\n";
    if (consecutive_prompt.find(consecutive_expected) == std::string::npos) {
        std::cerr << "  consecutive assistant messages did not share a model turn\n";
        ok = false;
    }

    std::vector<ChatMessage> tool_loop = {
        {"user", "weather", "", {}, {}, 0, {}},
        {"assistant", "", "", {}, {}, 0, {{"get_weather", R"({"city":"Paris"})"}}},
        {"tool", "Sunny", "get_weather", {}, {}, 0, {}}
    };
    std::string terminal_plain = tok->format_chat_prompt(tool_loop, true, "", false);
    std::string terminal_response =
        "<|tool_response>response:get_weather{value:<|\"|>Sunny<|\"|>}<tool_response|>";
    if (terminal_plain.size() < terminal_response.size()
        || terminal_plain.compare(terminal_plain.size() - terminal_response.size(),
                                  terminal_response.size(), terminal_response) != 0) {
        std::cerr << "  terminal tool response was closed before generation\n";
        ok = false;
    }

    std::string terminal_thinking = tok->format_chat_prompt(tool_loop, true, "", true);
    std::string thought_primer = terminal_response + "<|channel>thought\n";
    if (terminal_thinking.size() < thought_primer.size()
        || terminal_thinking.compare(terminal_thinking.size() - thought_primer.size(),
                                     thought_primer.size(), thought_primer) != 0) {
        std::cerr << "  thinking primer missing after terminal tool response\n";
        ok = false;
    }

    tool_loop.push_back({"assistant", "first", "", {}, {}, 0, {}});
    tool_loop.push_back({"assistant", "second", "", {}, {}, 0, {}});
    tool_loop.push_back({"user", "done", "", {}, {}, 0, {}});
    std::string continued = tok->format_chat_prompt(tool_loop, true, "", false);
    std::string continued_expected = terminal_response
        + "firstsecond<turn|>\n<|turn>user\ndone<turn|>\n<|turn>model\n";
    if (continued.find(continued_expected) == std::string::npos) {
        std::cerr << "  tool response did not continue into the following assistant turn\n";
        ok = false;
    }

    cactus_destroy(model);
    return ok;
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

static std::string benchmark_tokens_json(cactus_model_t model, const std::vector<uint32_t>& ids, size_t max_new) {
    std::vector<char> response(1 << 16, 0);
    int rc = cactus_benchmark_tokens(model, ids.data(), ids.size(), max_new, response.data(), response.size());
    return rc < 0 ? std::string() : std::string(response.data());
}

static std::string completion_ids_field(const std::string& json) {
    size_t start = json.find("\"completion_token_ids\":[");
    if (start == std::string::npos) return std::string();
    size_t end = json.find(']', start);
    return end == std::string::npos ? std::string() : json.substr(start, end - start + 1);
}

static std::string first_completion_id(const std::string& json) {
    std::string ids = completion_ids_field(json);
    size_t open = ids.find('[');
    if (open == std::string::npos) return std::string();
    size_t end = ids.find_first_of(",]", open + 1);
    return end == std::string::npos ? std::string() : ids.substr(open + 1, end - open - 1);
}

bool test_chunked_prefill_padding() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      CHUNKED PREFILL PADDING TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_model_path) { std::cout << "  [WARN] CACTUS_TEST_MODEL not set; skipping\n"; return true; }

    std::vector<uint32_t> sentence;
    {
        cactus_model_t tok_model = cactus_init(g_model_path, nullptr, false);
        if (!tok_model) { std::cerr << "  [✗] model init failed\n"; return false; }
        auto* tok = static_cast<CactusModelHandle*>(tok_model)->model->get_tokenizer();
        sentence = tok->encode(
            "The quick brown fox jumps over the lazy dog. Paris is the capital of France. "
            "Water is composed of hydrogen and oxygen. The Earth orbits the Sun once every year. "
            "Photosynthesis converts sunlight into chemical energy in plants.");
        cactus_destroy(tok_model);
        if (sentence.empty()) { std::cerr << "  [✗] tokenizer produced no ids\n"; return false; }
    }

    for (size_t prompt_len : {size_t(95), size_t(600)}) {
        std::vector<uint32_t> ids(prompt_len);
        for (size_t i = 0; i < ids.size(); i++) ids[i] = sentence[i % sentence.size()];

        cactus_model_t model = cactus_init(g_model_path, nullptr, false);
        if (!model) { std::cerr << "  [✗] model init failed\n"; return false; }
        std::string padded = benchmark_tokens_json(model, ids, 4);
        std::string padded_again = benchmark_tokens_json(model, ids, 4);
        cactus_destroy(model);
        if (padded.empty() || padded.find("\"success\":true") == std::string::npos
                || padded_again.empty() || padded_again.find("\"success\":true") == std::string::npos) {
            std::cerr << "  [✗] repeated chunked benchmark failed at len " << prompt_len << "\n";
            return false;
        }
        long tail_chunk = static_cast<long>(EngineTestUtils::json_number(padded, "prefill_tail_chunk_tokens", -1));
        long tail_pads = static_cast<long>(EngineTestUtils::json_number(padded, "prefill_tail_padding_tokens", -1));
        long scalar = static_cast<long>(EngineTestUtils::json_number(padded, "prefill_scalar_tail_tokens", -1));
        std::cout << "  len " << prompt_len << ": tail_chunk=" << tail_chunk
                  << " pads=" << tail_pads << " scalar=" << scalar << "\n";
        if (tail_chunk <= 0) {
            std::cout << "  [WARN] padded tail did not engage (no sliding caches?); skipping\n";
            continue;
        }
        if (tail_pads <= 0 || scalar > 1) {
            std::cerr << "  [✗] unexpected padding telemetry\n";
            return false;
        }
        if (completion_ids_field(padded).empty()
                || completion_ids_field(padded) != completion_ids_field(padded_again)) {
            std::cerr << "  [✗] padded prefill is not deterministic\n";
            return false;
        }

        setenv("CACTUS_DISABLE_PREFILL_TAIL_PAD", "1", 1);
        model = cactus_init(g_model_path, nullptr, false);
        std::string scalar_run = model ? benchmark_tokens_json(model, ids, 4) : std::string();
        if (model) cactus_destroy(model);
        unsetenv("CACTUS_DISABLE_PREFILL_TAIL_PAD");
        if (scalar_run.empty() || scalar_run.find("\"success\":true") == std::string::npos) {
            std::cerr << "  [✗] kill-switch benchmark failed\n";
            return false;
        }
        long off_tail = static_cast<long>(EngineTestUtils::json_number(scalar_run, "prefill_tail_chunk_tokens", -1));
        long off_scalar = static_cast<long>(EngineTestUtils::json_number(scalar_run, "prefill_scalar_tail_tokens", -1));
        const size_t chunk_size = static_cast<size_t>(tail_chunk + tail_pads + 1);
        if (off_tail != 0 || off_scalar != static_cast<long>(prompt_len % chunk_size)) {
            std::cerr << "  [✗] kill switch did not restore the scalar tail (tail=" << off_tail
                      << " scalar=" << off_scalar << ")\n";
            return false;
        }
        if (first_completion_id(padded).empty()
                || first_completion_id(padded) != first_completion_id(scalar_run)) {
            std::cerr << "  [✗] padded prefill diverged from the scalar tail\n"
                      << "    padded: " << completion_ids_field(padded) << "\n"
                      << "    scalar: " << completion_ids_field(scalar_run) << "\n";
            return false;
        }
    }
    return true;
}

bool test_batch_distinct4_matches_single() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "     BATCH DISTINCT-4 VS SINGLE TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";
    cactus_model_t model = load_dynamic_batch_model_or_skip();
    if (!model) return true;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tok = handle->model->get_tokenizer();
    if (!handle->model->supports_dynamic_batch()) {
        std::cout << "  [WARN] bundle not dynamic-batch capable (reconvert with `cactus convert`); skipping\n";
        cactus_destroy(model);
        return true;
    }

    const std::vector<std::string> questions = {
        "In one short sentence, what is Paris famous for?",
        "List three primary colors, comma separated.",
        "Count from one to five.",
        "Name two planets in our solar system."
    };
    std::vector<std::vector<uint32_t>> prompts;
    for (const auto& q : questions) {
        ChatMessage msg; msg.role = "user"; msg.content = q;
        prompts.push_back(tok->encode(tok->format_chat_prompt({msg}, true, "", false)));
    }
    const size_t M = 24;
    const size_t K = prompts.size();

    handle->model->set_decode_slots(K);
    auto batched = handle->model->generate_batch(prompts, M, true);
    if (batched.size() != K) { std::cerr << "  batched returned " << batched.size() << " streams\n"; cactus_destroy(model); return false; }

    bool ok = true;
    for (size_t i = 0; i < K; ++i) {
        auto ref = handle->model->generate_batch({prompts[i]}, M, true);
        bool match = !ref.empty() && ref[0] == batched[i];
        std::cout << "  row " << i << " (\"" << questions[i] << "\") " << (match ? "MATCH" : "DIVERGED")
                  << ": " << tok->decode(batched[i]) << "\n";
        if (!match) ok = false;
    }
    cactus_destroy(model);
    return ok;
}

int main() {
    TestUtils::TestRunner runner("LLM Tests");
    runner.run_test("streaming", test_streaming());
    runner.run_test("prefill", test_prefill());
    runner.run_test("prefill_invalidated_on_message_change", test_prefill_invalidated_on_message_change());
    runner.run_test("chunked_prefill_padding", test_chunked_prefill_padding());
    runner.run_test("tool_calls", test_tool_call());
    runner.run_test("tool_multiple_tool_call_invocations", test_multiple_tool_call_invocations());
    runner.run_test("tool_constraint_clear_releases_bias", test_tool_constraint_clear_releases_bias());
    runner.run_test("partition_thinking_response", test_partition_thinking_response());
    runner.run_test("prompt_gemma4_agentic_turn_closure", test_prompt_gemma4_agentic_turn_closure());
    runner.run_test("prompt_retains_thinking", test_prompt_gemma4_retains_thinking());
    runner.run_test("complete_thinking_api_clean", test_complete_gemma4_thinking_api_clean());
    runner.run_test("multiturn_thinking_persist", test_multiturn_thinking_persist());
    runner.run_test("batch_distinct4_matches_single", test_batch_distinct4_matches_single());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
