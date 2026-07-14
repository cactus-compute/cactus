#include "test_utils.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static bool model_is_needle(const std::string& bundle_path) {
    std::ifstream f(bundle_path + "/config.txt");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.rfind("model_type=", 0) == 0) {
            return line.substr(11) == "needle";
        }
    }
    return false;
}

bool test_needle_tool_call() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║          NEEDLE TOOL CALL TEST           ║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!g_model_path) {
        std::cout << "  [WARN] CACTUS_TEST_MODEL not set; skipping\n";
        return true;
    }
    if (!model_is_needle(g_model_path)) {
        std::cout << "  [SKIP] model under test is not needle; skipping needle tool-call test\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "  [✗] Failed to initialize needle model at " << g_model_path << "\n";
        return false;
    }

    const char* messages = R"([
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options = R"({"max_tokens": 64, "force_tools": true, "telemetry_enabled": false, "auto_handoff": false})";

    char response[1 << 15] = {0};
    int rc = cactus_complete(model, messages, response, sizeof(response), options, tools,
                             nullptr, nullptr, nullptr, 0);
    std::string r(response);

    std::cout << "├─ rc: " << rc << "\n";
    std::cout << "├─ Response: " << r << "\n";

    const bool has_function_calls =
        r.find("\"function_calls\":[{") != std::string::npos;
    const bool calls_tool = r.find("get_weather") != std::string::npos;
    const bool has_arg = r.find("San Francisco") != std::string::npos;
    std::cout << "├─ function_calls present: " << (has_function_calls ? "YES" : "NO") << "\n";
    std::cout << "├─ Calls get_weather: " << (calls_tool ? "YES" : "NO") << "\n";
    std::cout << "├─ Location arg present: " << (has_arg ? "YES" : "NO") << "\n";

    const bool ok = rc > 0 && has_function_calls && calls_tool && has_arg;
    std::cout << "└─ Status: " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";

    cactus_destroy(model);
    return ok;
}

int main() {
    TestUtils::apply_backend();
    bool ok = true;
    ok &= test_needle_tool_call();
    std::cout << "\n" << (ok ? "✓ needle tool-call test passed" : "✗ needle tool-call test failed") << "\n";
    return ok ? 0 : 1;
}
