#include "../src/chat_tools.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    struct Case {
        const char* input;
        const char* expected;
    };

    const std::vector<Case> cases = {
        {R"([get_weather(city="Paris", options={"units": "metric"})])",
         R"({"name": "get_weather", "arguments": {"city": "Paris", "options": {"units": "metric"}}})"},

        {R"([get_weather(city="Paris", options={"units": "metric")])",
         R"({"name": "get_weather", "arguments": {"city": "Paris", "options": {"units": "metric"}}})"},

        {R"([set_alarm(times=["07:00", "07:30"})])",
         R"({"name": "set_alarm", "arguments": {"times": ["07:00", "07:30"]}})"},

        {R"([get_weather(city="Paris"})])",
         R"({"name": "get_weather", "arguments": {"city": "Paris"}})"}
    };

    for (size_t i = 0; i < cases.size(); ++i) {
        std::string text =
            std::string("<|tool_call_start|>") +
            cases[i].input + "<|tool_call_end|>";

        std::vector<std::string> calls;
        chat_tools::extract_lfm2_tool_calls(text, calls);

        if (calls.size() != 1 ||
            calls[0] != cases[i].expected ||
            !text.empty()) {
            std::cerr << "FAIL case " << i + 1 << "\n";
            return 1;
        }

        std::cout << "PASS case " << i + 1 << "\n";
    }

    return 0;
}
