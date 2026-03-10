#include "test_utils.h"
#include "../cactus/ffi/cactus_utils.h"
#include <iostream>
#include <string>
#include <vector>

using namespace cactus::ffi;
using namespace cactus::engine;

static bool parse_thinking_option(const std::string& json) {
    float temp, top_p, conf;
    size_t top_k, max_tokens, tool_rag_top_k;
    std::vector<std::string> stop;
    bool force, inc_stop, vad, tele, result = false;
    parse_options_json(json, temp, top_p, top_k, max_tokens, stop,
                       force, tool_rag_top_k, conf, inc_stop, vad, tele,
                       nullptr, nullptr, nullptr, &result);
    return result;
}

static bool check_strip(const std::string& input,
                         const std::string& expected_thinking,
                         const std::string& expected_content) {
    std::string thinking, content;
    strip_thinking_block(input, thinking, content);
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

bool test_parse_options_thinking() {
    return parse_thinking_option("{}") == true
        && parse_thinking_option(R"({"enable_thinking_if_supported": false})") == false
        && parse_thinking_option(R"({"enable_thinking_if_supported": true})") == true;
}

bool test_parse_options_thinking_nullptr_safe() {
    float temp, top_p, conf;
    size_t top_k, max_tokens, tool_rag_top_k;
    std::vector<std::string> stop;
    bool force, inc_stop, vad, tele;
    parse_options_json(R"({"enable_thinking_if_supported": false})",
                       temp, top_p, top_k, max_tokens, stop,
                       force, tool_rag_top_k, conf, inc_stop, vad, tele);
    return true;
}

bool test_strip_thinking() {
    return check_strip("<think>reason</think>answer", "reason", "answer")
        && check_strip("<think>\n  reason\n</think>\n\nanswer", "reason", "answer")
        && check_strip("no tags here", "", "no tags here")
        && check_strip("<think></think>answer", "", "answer")
        && check_strip("reason\n</think>\n\nanswer", "reason", "answer")
        && check_strip("<think>unclosed", "unclosed", "");
}

bool test_response_json_thinking_field() {
    std::vector<std::string> calls;
    std::string with = construct_response_json("hi", calls, 0, 0, 0, 0, 0, 0, 0, false, "reason");
    std::string without = construct_response_json("hi", calls, 0, 0, 0, 0, 0, 0, 0, false, "");
    return with.find("\"thinking\":\"reason\"") != std::string::npos
        && without.find("\"thinking\"") == std::string::npos;
}

bool test_prompt_thinking_injection() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { std::cout << "  [SKIP] Could not load model\n"; return true; }

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto mtype = handle->model->get_config().model_type;
    if (mtype != Config::ModelType::QWEN && mtype != Config::ModelType::QWEN3P5) {
        std::cout << "  [SKIP] Not a Qwen model\n";
        cactus_destroy(model);
        return true;
    }

    auto* tok = handle->model->get_tokenizer();
    std::vector<ChatMessage> msgs = {{"user", "hello", "", {}}};

    std::string enabled = tok->format_chat_prompt(msgs, true, "", true);
    std::string disabled = tok->format_chat_prompt(msgs, true, "", false);
    cactus_destroy(model);

    bool ok = enabled.find("<think>\n") != std::string::npos
           && enabled.find("<think>\n\n</think>") == std::string::npos
           && disabled.find("<think>\n\n</think>") != std::string::npos;
    if (!ok) std::cerr << "  thinking injection mismatch in prompt\n";
    return ok;
}

bool test_complete_thinking_toggle() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) return false;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto mtype = handle->model->get_config().model_type;
    if (mtype != Config::ModelType::QWEN && mtype != Config::ModelType::QWEN3P5) {
        std::cout << "  [SKIP] Not a Qwen model\n";
        cactus_destroy(model);
        return true;
    }

    const char* msgs = R"([{"role": "user", "content": "What is 2+2?"}])";
    char buf[8192];

    int r1 = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":128,"enable_thinking_if_supported":true,"telemetry_enabled":false})",
        nullptr, nullptr, nullptr);
    std::string resp1(buf);
    bool ok1 = r1 > 0 && resp1.find("\"success\":true") != std::string::npos;

    handle->model->reset_cache();
    handle->processed_tokens.clear();

    int r2 = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":128,"enable_thinking_if_supported":false,"telemetry_enabled":false})",
        nullptr, nullptr, nullptr);
    std::string resp2(buf);
    bool ok2 = r2 > 0 && resp2.find("\"thinking\"") == std::string::npos;

    cactus_destroy(model);
    return ok1 && ok2;
}

int main() {
    TestUtils::TestRunner runner("Qwen Thinking Tests");
    runner.run_test("parse_options_thinking", test_parse_options_thinking());
    runner.run_test("parse_options_nullptr_safe", test_parse_options_thinking_nullptr_safe());
    runner.run_test("strip_thinking", test_strip_thinking());
    runner.run_test("response_json_thinking_field", test_response_json_thinking_field());
    runner.run_test("prompt_thinking_injection", test_prompt_thinking_injection());
    runner.run_test("complete_thinking_toggle", test_complete_thinking_toggle());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
