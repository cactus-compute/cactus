#include "test_utils.h"
#include "../src/chat_tools.h"
#include "picojson.h"
#include <iostream>
#include <string>
#include <vector>

using namespace TestUtils;

static bool test_py_literal_to_json_single_quote() {
    std::string s = "'hello \\'world\\''";
    size_t p = 0;
    std::string json_val = chat_tools::py_literal_to_json(s, p);

    // Expected JSON: "hello 'world'"
    std::string expected = "\"hello 'world'\"";
    if (json_val != expected) {
        std::cerr << "FAIL: expected " << expected << ", got " << json_val << std::endl;
        return false;
    }

    // Ensure it parses as valid JSON
    picojson::value v;
    std::string err = picojson::parse(v, json_val);
    if (!err.empty()) {
        std::cerr << "FAIL: picojson parse error: " << err << std::endl;
        return false;
    }
    if (!v.is<std::string>() || v.get<std::string>() != "hello 'world'") {
        std::cerr << "FAIL: parsed string value mismatch: " << v.to_str() << std::endl;
        return false;
    }

    return true;
}

static bool test_extract_lfm2_tool_calls_single_quote() {
    std::string text = "<|tool_call_start|>my_tool(param='hello \\'world\\'', num=42)<|tool_call_end|>";
    std::vector<std::string> calls;
    chat_tools::extract_lfm2_tool_calls(text, calls);

    if (calls.size() != 1) {
        std::cerr << "FAIL: expected 1 call, got " << calls.size() << std::endl;
        return false;
    }

    std::string call_json = calls[0];
    // Expected: {"name": "my_tool", "arguments": {"param": "hello 'world'", "num": 42}}
    picojson::value v;
    std::string err = picojson::parse(v, call_json);
    if (!err.empty()) {
        std::cerr << "FAIL: extracted call is invalid JSON: " << err << " (JSON: " << call_json << ")" << std::endl;
        return false;
    }

    if (!v.is<picojson::object>()) {
        std::cerr << "FAIL: expected JSON object" << std::endl;
        return false;
    }

    const auto& obj = v.get<picojson::object>();
    if (obj.at("name").get<std::string>() != "my_tool") {
        return false;
    }

    const auto& args = obj.at("arguments").get<picojson::object>();
    if (args.at("param").get<std::string>() != "hello 'world'") {
        return false;
    }
    if (args.at("num").get<double>() != 42.0) {
        return false;
    }

    return true;
}

int main() {
    TestRunner runner("Chat Tools Tests");
    runner.run_test("py_literal_to_json_single_quote", test_py_literal_to_json_single_quote());
    runner.run_test("extract_lfm2_tool_calls_single_quote", test_extract_lfm2_tool_calls_single_quote());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
