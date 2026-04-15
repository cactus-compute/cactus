#include "test_utils.h"
#include "../cactus/ffi/cactus_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace cactus::ffi;
using namespace cactus::engine;

static const char* get_gemma4_model_path() {
    const char* path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (path) return path;
    return std::getenv("CACTUS_TEST_MODEL");
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

static std::string escape_for_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

bool test_strip_thinking_channel() {
    return check_strip("<|channel>reason<channel|>answer", "reason", "answer")
        && check_strip("<|channel>\n  reason\n<channel|>\n\nanswer", "reason", "answer")
        && check_strip("no tags here", "", "no tags here")
        && check_strip("<|channel>unclosed", "unclosed", "")
        && check_strip("reason\n<channel|>\n\nanswer", "reason", "answer")
        && check_strip("<|channel>thought1<channel|>text1<|channel>thought2<channel|>text2",
                        "thought1\nthought2", "text1text2");
}

bool test_find_channel_token_ranges() {
    constexpr uint32_t OPEN = 100;
    constexpr uint32_t CLOSE = 101;

    {
        std::vector<uint32_t> tokens = {OPEN, 50, 51, 52, CLOSE, 60, 61};
        auto ranges = find_channel_token_ranges(tokens, 0, OPEN, CLOSE);
        if (ranges.size() != 1 || ranges[0].first != 0 || ranges[0].second != 5) {
            std::cerr << "  Single block failed\n";
            return false;
        }
    }
    {
        std::vector<uint32_t> tokens = {10, OPEN, 50, CLOSE, 20, OPEN, 51, 52, CLOSE, 30};
        auto ranges = find_channel_token_ranges(tokens, 0, OPEN, CLOSE);
        if (ranges.size() != 2 || ranges[0].first != 1 || ranges[0].second != 3
                               || ranges[1].first != 5 || ranges[1].second != 4) {
            std::cerr << "  Two blocks failed\n";
            return false;
        }
    }
    {
        std::vector<uint32_t> tokens = {10, 20, 30};
        if (!find_channel_token_ranges(tokens, 0, OPEN, CLOSE).empty()) {
            std::cerr << "  No blocks failed\n";
            return false;
        }
    }
    {
        std::vector<uint32_t> tokens = {OPEN, 50, 51};
        auto ranges = find_channel_token_ranges(tokens, 0, OPEN, CLOSE);
        if (ranges.size() != 1 || ranges[0].first != 0 || ranges[0].second != 3) {
            std::cerr << "  Unclosed block failed\n";
            return false;
        }
    }
    {
        std::vector<uint32_t> tokens = {OPEN, 50, CLOSE, 60};
        auto ranges = find_channel_token_ranges(tokens, 100, OPEN, CLOSE);
        if (ranges.size() != 1 || ranges[0].first != 100 || ranges[0].second != 3) {
            std::cerr << "  Offset failed\n";
            return false;
        }
    }

    return true;
}

bool test_prompt_gemma4_thinking_injection() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { std::cout << "  [SKIP] Could not load model\n"; return true; }

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tok = handle->model->get_tokenizer();
    std::vector<ChatMessage> msgs = {{"user", "hello", "", {}, {}, 0, {}}};

    std::string enabled = tok->format_chat_prompt(msgs, true, "", true);
    std::string disabled = tok->format_chat_prompt(msgs, true, "", false);
    cactus_destroy(model);

    bool ok = enabled.find("<|think|>") != std::string::npos
           && disabled.find("<|think|>") == std::string::npos
           && disabled.find("<|channel>") == std::string::npos
           && disabled.find("<channel|>") == std::string::npos;
    if (!ok) {
        std::cerr << "  thinking injection mismatch in prompt\n";
        std::cerr << "  enabled prompt: " << enabled << "\n";
        std::cerr << "  disabled prompt: " << disabled << "\n";
    }
    return ok;
}

bool test_prompt_gemma4_assistant_stripping() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { std::cout << "  [SKIP] Could not load model\n"; return true; }

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
    bool no_reasoning = prompt.find("internal reasoning") == std::string::npos;
    bool no_channel_tags = prompt.find("<|channel>") == std::string::npos
                        && prompt.find("<channel|>") == std::string::npos;

    if (!has_visible) std::cerr << "  missing visible response in prompt\n";
    if (!no_reasoning) std::cerr << "  thinking content not stripped from assistant turn\n";
    if (!no_channel_tags) std::cerr << "  channel tags not stripped from prompt\n";

    return has_visible && no_reasoning && no_channel_tags;
}

bool test_complete_gemma4_thinking_toggle() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) return false;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto mtype = handle->model->get_config().model_type;
    if (mtype != Config::ModelType::GEMMA4) {
        std::cout << "  [SKIP] Not a Gemma4 model\n";
        cactus_destroy(model);
        return true;
    }

    const char* msgs = R"([{"role": "user", "content": "What is 2+2?"}])";
    char buf[8192];

    int r1 = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":128,"enable_thinking_if_supported":true,"telemetry_enabled":false})",
        nullptr, nullptr, nullptr, nullptr, 0);
    std::string resp1(buf);
    bool ok1 = r1 > 0 && resp1.find("\"success\":true") != std::string::npos;

    handle->model->reset_cache();
    handle->processed_tokens.clear();

    int r2 = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":128,"enable_thinking_if_supported":false,"telemetry_enabled":false})",
        nullptr, nullptr, nullptr, nullptr, 0);
    std::string resp2(buf);
    bool ok2 = r2 > 0
            && resp2.find("\"thinking\"") == std::string::npos
            && resp2.find("<|channel>") == std::string::npos
            && resp2.find("<channel|>") == std::string::npos;

    cactus_destroy(model);
    if (!ok1) std::cerr << "  thinking-enabled completion failed\n";
    if (!ok2) std::cerr << "  thinking-disabled completion should not have thinking field\n";
    return ok1 && ok2;
}

bool test_complete_gemma4_tool_call() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) return false;

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto mtype = handle->model->get_config().model_type;
    if (mtype != Config::ModelType::GEMMA4) {
        std::cout << "  [SKIP] Not a Gemma4 model\n";
        cactus_destroy(model);
        return true;
    }

    const char* msgs = R"([{"role": "user", "content": "What is the weather in San Francisco?"}])";
    const char* tools = R"([{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}}])";
    char buf[8192];

    int r = cactus_complete(model, msgs, buf, sizeof(buf),
        R"({"max_tokens":256,"force_tool_call":true,"enable_thinking_if_supported":true,"telemetry_enabled":false})",
        tools, nullptr, nullptr, nullptr, 0);
    std::string resp(buf);

    cactus_destroy(model);

    bool success = r > 0 && resp.find("\"success\":true") != std::string::npos;
    bool has_calls = resp.find("\"function_calls\"") != std::string::npos
                  && resp.find("get_weather") != std::string::npos;

    if (!success) std::cerr << "  tool call completion failed: " << resp << "\n";
    if (!has_calls) std::cerr << "  expected get_weather tool call in response: " << resp << "\n";
    return success && has_calls;
}

bool test_multiturn_cache_reuse() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { std::cerr << "  Failed to load model\n"; return false; }

    auto* handle = static_cast<CactusModelHandle*>(model);
    if (handle->model->get_config().model_type != Config::ModelType::GEMMA4) {
        std::cout << "  [SKIP] Not a Gemma4 model\n";
        cactus_destroy(model);
        return true;
    }

    auto* tokenizer = handle->model->get_tokenizer();
    const char* options = R"({"max_tokens":128,"temperature":0,"top_k":1,"enable_thinking_if_supported":true,"telemetry_enabled":false,"auto_handoff":false})";
    const char* turn1_msgs = R"([{"role": "user", "content": "My name is Alice. Please remember this."}])";
    char buf[16384];

    int r1 = cactus_complete(model, turn1_msgs, buf, sizeof(buf), options, nullptr, nullptr, nullptr, nullptr, 0);
    if (r1 <= 0) { std::cerr << "  Turn 1 failed\n"; cactus_destroy(model); return false; }

    std::vector<uint32_t> processed_after_t1 = handle->processed_tokens;

    std::vector<ChatMessage> t1_chat = {{"user", "My name is Alice. Please remember this.", "", {}, {}, 0, {}}};
    std::vector<uint32_t> t1_prompt_tokens = tokenizer->encode(tokenizer->format_chat_prompt(t1_chat, true, "", true));
    std::vector<uint32_t> gen_tokens(processed_after_t1.begin() + t1_prompt_tokens.size(), processed_after_t1.end());
    std::string assistant_text = tokenizer->decode(gen_tokens);

    std::vector<ChatMessage> t2_chat = {
        {"user", "My name is Alice. Please remember this.", "", {}, {}, 0, {}},
        {"assistant", assistant_text, "", {}, {}, 0, {}},
        {"user", "What is my name?", "", {}, {}, 0, {}}
    };
    std::vector<uint32_t> t2_prompt_tokens = tokenizer->encode(tokenizer->format_chat_prompt(t2_chat, true, "", true));

    bool prefix_ok = (t2_prompt_tokens.size() >= processed_after_t1.size()) &&
                     std::equal(processed_after_t1.begin(), processed_after_t1.end(), t2_prompt_tokens.begin());
    std::cout << "  Prefix match (cache reuse): " << (prefix_ok ? "YES" : "NO") << "\n";

    size_t new_tokens_for_t2 = t2_prompt_tokens.size() - processed_after_t1.size();
    std::cout << "  Turn 2 new tokens: " << new_tokens_for_t2 << " (full prompt: " << t2_prompt_tokens.size() << ")\n";

    std::string escaped = escape_for_json(assistant_text);
    std::string turn2_json = R"([{"role": "user", "content": "My name is Alice. Please remember this."},{"role": "assistant", "content": ")"
        + escaped + R"("},{"role": "user", "content": "What is my name?"}])";

    int r2 = cactus_complete(model, turn2_json.c_str(), buf, sizeof(buf), options, nullptr, nullptr, nullptr, nullptr, 0);
    if (r2 <= 0) { std::cerr << "  Turn 2 failed\n"; cactus_destroy(model); return false; }

    std::string turn2_response(buf);
    bool mentions_alice = turn2_response.find("Alice") != std::string::npos
                       || turn2_response.find("alice") != std::string::npos;
    std::cout << "  Turn 2 mentions Alice: " << (mentions_alice ? "YES" : "NO") << "\n";

    cactus_destroy(model);

    if (!prefix_ok) std::cerr << "  FAIL: cache surgery did not produce valid prefix\n";
    return prefix_ok;
}

int main() {
    TestUtils::TestRunner runner("Gemma4 Thinking Tests");
    runner.run_test("strip_thinking_channel", test_strip_thinking_channel());
    runner.run_test("find_channel_token_ranges", test_find_channel_token_ranges());
    runner.run_test("prompt_thinking_injection", test_prompt_gemma4_thinking_injection());
    runner.run_test("prompt_assistant_stripping", test_prompt_gemma4_assistant_stripping());
    runner.run_test("complete_thinking_toggle", test_complete_gemma4_thinking_toggle());
    runner.run_test("complete_tool_call", test_complete_gemma4_tool_call());
    runner.run_test("multiturn_cache_reuse", test_multiturn_cache_reuse());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
