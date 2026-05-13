// BFCL-style tool-calling accuracy eval across TQ KV configs.
//
// Each config is one row in the result table. For each config we re-init the
// model with the right env vars before cactus_init() is called. Test cases
// cover three categories matching the user's reference table:
//   - Simple Python: one obvious tool, expect it to be called
//   - Multiple: multiple available tools, expect the right one
//   - Irrelevance: question shouldn't trigger any tool call
//
// Usage:
//   CACTUS_TEST_MODEL=weights/<model> ./test_bfcl_eval
//
// Optional:
//   CACTUS_BFCL_CASES=path/to/cases.json   custom test case file

#include "test_utils.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace EngineTestUtils;

struct TestCase {
    const char* category;
    const char* user_query;
    const char* tools_json;
    const char* expected_function;  // empty string = no function should be called
};

static const std::vector<TestCase> CASES = {
    // --- Simple Python: one tool, should call it ---
    {
        "simple", "What's the weather in San Francisco?",
        R"([{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}])",
        "get_weather"
    },
    {
        "simple", "Calculate the factorial of 7",
        R"([{"type":"function","function":{"name":"factorial","description":"Compute factorial","parameters":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}}])",
        "factorial"
    },
    {
        "simple", "Send an email to alice@example.com saying hello",
        R"([{"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}}])",
        "send_email"
    },
    {
        "simple", "Search the web for transformers paper",
        R"([{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}])",
        "web_search"
    },

    // --- Multiple: pick the right one from several ---
    {
        "multiple", "Translate hello to French",
        R"([{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}},{"type":"function","function":{"name":"translate","description":"Translate text to a target language","parameters":{"type":"object","properties":{"text":{"type":"string"},"target":{"type":"string"}},"required":["text","target"]}}},{"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"}}}}}])",
        "translate"
    },
    {
        "multiple", "Convert 100 USD to EUR",
        R"([{"type":"function","function":{"name":"factorial","description":"Compute factorial","parameters":{"type":"object","properties":{"n":{"type":"integer"}}}}},{"type":"function","function":{"name":"currency_convert","description":"Convert between currencies","parameters":{"type":"object","properties":{"amount":{"type":"number"},"from":{"type":"string"},"to":{"type":"string"}}}},{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}])",
        "currency_convert"
    },
    {
        "multiple", "What's 25 percent of 200?",
        R"([{"type":"function","function":{"name":"calculate","description":"Evaluate an arithmetic expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}}}}},{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}])",
        "calculate"
    },

    // --- Irrelevance: shouldn't call any tool ---
    {
        "irrelevance", "What is your name?",
        R"([{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}])",
        ""
    },
    {
        "irrelevance", "Tell me a joke",
        R"([{"type":"function","function":{"name":"calculate","description":"Evaluate arithmetic","parameters":{"type":"object","properties":{"expression":{"type":"string"}}}}}])",
        ""
    },
    {
        "irrelevance", "I love pizza",
        R"([{"type":"function","function":{"name":"send_email","description":"Send email","parameters":{"type":"object","properties":{"to":{"type":"string"}}}}}])",
        ""
    },
};

struct ConfigResult {
    std::string name;
    int simple_pass = 0, simple_total = 0;
    int multiple_pass = 0, multiple_total = 0;
    int irrelevance_pass = 0, irrelevance_total = 0;
};

static bool eval_one_case(cactus_model_t model, const TestCase& tc) {
    std::string messages = std::string("[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"") + tc.user_query + "\"}]";

    // For simple/multiple, force tools so the model attempts a call.
    // For irrelevance, don't force — model should choose to answer in natural text.
    bool force_tools = std::string(tc.category) != "irrelevance";
    std::string options;
    if (force_tools) {
        options = R"({"max_tokens":256,"stop_sequences":["<|im_end|>","<end_of_turn>"],"force_tools":true,"telemetry_enabled":false})";
    } else {
        options = R"({"max_tokens":256,"stop_sequences":["<|im_end|>","<end_of_turn>"],"telemetry_enabled":false})";
    }

    char response[8192] = {0};
    cactus_complete(model, messages.c_str(), response, sizeof(response),
                    options.c_str(), tc.tools_json, nullptr, nullptr, nullptr, 0);

    std::string resp(response);
    bool has_call = resp.find("\"function_calls\":[") != std::string::npos
                  && resp.find("\"function_calls\":[]") == std::string::npos;
    if (std::string(tc.expected_function).empty()) {
        // Irrelevance: pass if no tool call
        return !has_call;
    }
    return has_call && resp.find(tc.expected_function) != std::string::npos;
}

static void run_config(const char* name,
                        const std::vector<std::pair<const char*, const char*>>& env_pairs,
                        ConfigResult& out, const char* model_path) {
    // Set env vars for this config
    for (auto& [k, v] : env_pairs) {
        if (v == nullptr) unsetenv(k);
        else setenv(k, v, 1);
    }

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to init model for config " << name << std::endl;
        return;
    }

    out.name = name;
    for (const auto& tc : CASES) {
        bool pass = eval_one_case(model, tc);
        std::string cat(tc.category);
        if (cat == "simple") {
            out.simple_total++;
            if (pass) out.simple_pass++;
        } else if (cat == "multiple") {
            out.multiple_total++;
            if (pass) out.multiple_pass++;
        } else if (cat == "irrelevance") {
            out.irrelevance_total++;
            if (pass) out.irrelevance_pass++;
        }
        std::cout << "    [" << (pass ? "PASS" : "FAIL") << "] " << cat << " :: " << tc.user_query << std::endl;
    }
    cactus_destroy(model);
}

bool test_bfcl_eval() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::cout << "\n  Model: " << model_path << std::endl;
    std::cout << "  Cases: " << CASES.size() << " (3 categories)\n" << std::endl;

    std::vector<ConfigResult> results;

    struct ConfigSpec {
        const char* name;
        std::vector<std::pair<const char*, const char*>> env;
    };
    std::vector<ConfigSpec> configs = {
        {"INT8 baseline", {{"CACTUS_KV_TQ_BITS", nullptr}, {"CACTUS_KV_TQ_K_BITS", nullptr}, {"CACTUS_KV_TQ_V_BITS", nullptr}}},
        {"K=4 seed=42",   {{"CACTUS_KV_TQ_BITS", "4"}, {"CACTUS_KV_TQ_K_BITS", nullptr}, {"CACTUS_KV_TQ_V_BITS", nullptr}, {"CACTUS_KV_TQ_SEED", "42"}}},
        {"K=6 seed=42",   {{"CACTUS_KV_TQ_BITS", "6"}, {"CACTUS_KV_TQ_K_BITS", nullptr}, {"CACTUS_KV_TQ_V_BITS", nullptr}, {"CACTUS_KV_TQ_SEED", "42"}}},
        {"K6V2 seed=1337",{{"CACTUS_KV_TQ_BITS", nullptr}, {"CACTUS_KV_TQ_K_BITS", "6"}, {"CACTUS_KV_TQ_V_BITS", "2"}, {"CACTUS_KV_TQ_SEED", "1337"}}},
    };

    for (const auto& cfg : configs) {
        std::cout << "  === " << cfg.name << " ===" << std::endl;
        ConfigResult r;
        run_config(cfg.name, cfg.env, r, model_path);
        results.push_back(r);
        std::cout << std::endl;
    }

    // Summary table
    std::cout << "\n  ==== Accuracy Summary ====\n" << std::endl;
    std::cout << "  " << std::setw(20) << std::left << "Config"
              << std::setw(14) << "Simple"
              << std::setw(14) << "Multiple"
              << std::setw(14) << "Irrelevance" << std::endl;
    std::cout << "  " << std::string(64, '-') << std::endl;
    for (const auto& r : results) {
        std::string s = std::to_string(r.simple_pass) + "/" + std::to_string(std::max(1, r.simple_total));
        std::string m = std::to_string(r.multiple_pass) + "/" + std::to_string(std::max(1, r.multiple_total));
        std::string i = std::to_string(r.irrelevance_pass) + "/" + std::to_string(std::max(1, r.irrelevance_total));
        std::cout << "  " << std::setw(20) << std::left << r.name
                  << std::setw(14) << s
                  << std::setw(14) << m
                  << std::setw(14) << i << std::endl;
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("BFCL-style tool eval");
    runner.run_test("TQ config × tool accuracy", test_bfcl_eval());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
