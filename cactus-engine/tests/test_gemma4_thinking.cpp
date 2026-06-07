#include "test_utils.h"
#include "../src/engine.h"
#include "../src/utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace cactus::engine;
using cactus::ffi::partition_thinking_response;

static const char* get_gemma4_model_path() {
    const char* path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (path) return path;
    return std::getenv("CACTUS_TEST_MODEL");
}

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

// partition_thinking_response is display-only: it splits generated text into (thinking, content)
// for the API, without touching the cache.
bool test_partition_thinking_response() {
    return check_partition("<|channel>reason<channel|>answer", "reason", "answer")
        && check_partition("<|channel>\n  reason\n<channel|>\n\nanswer", "reason", "answer")
        && check_partition("no tags here", "", "no tags here")
        && check_partition("<think>reason</think>answer", "reason", "answer")
        && check_partition("<|channel>thought1<channel|>text1<|channel>thought2<channel|>text2",
                            "thought1\nthought2", "text1text2");
}

// New contract: the Gemma4 formatter RETAINS prior-turn assistant channel/thinking content in the
// rendered prompt (no strip), matching LiteRT so thinking persists across turns.
bool test_prompt_gemma4_retains_thinking() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) { std::cout << "  [SKIP] Could not load model\n"; return true; }

    auto* handle = static_cast<CactusModelHandle*>(model);
    if (handle->model->get_config().model_type != Config::ModelType::GEMMA4) {
        std::cout << "  [SKIP] Not a Gemma4 model\n";
        cactus_destroy(model);
        return true;
    }
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

// API separation is unchanged: response excludes thinking, thinking is populated, and no raw
// channel tags leak into the user-facing response.
bool test_complete_gemma4_thinking_api_clean() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) { std::cout << "  [SKIP] CACTUS_TEST_GEMMA4_MODEL not set\n"; return true; }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) return false;

    auto* handle = static_cast<CactusModelHandle*>(model);
    if (handle->model->get_config().model_type != Config::ModelType::GEMMA4) {
        std::cout << "  [SKIP] Not a Gemma4 model\n";
        cactus_destroy(model);
        return true;
    }

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

// Multi-turn: store the raw context_response (thinking retained) for history; the re-rendered
// turn-2 prompt must prefix-match the turn-1 processed tokens (cache reuse) and recall the fact.
bool test_multiturn_thinking_persist() {
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
    TestUtils::TestRunner runner("Gemma4 Thinking Tests");
    runner.run_test("partition_thinking_response", test_partition_thinking_response());
    runner.run_test("prompt_retains_thinking", test_prompt_gemma4_retains_thinking());
    runner.run_test("complete_thinking_api_clean", test_complete_gemma4_thinking_api_clean());
    runner.run_test("multiturn_thinking_persist", test_multiturn_thinking_persist());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
