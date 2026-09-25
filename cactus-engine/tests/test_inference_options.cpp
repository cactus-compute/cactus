#include "test_utils.h"
#include "../src/utils.h"

using cactus::ffi::parse_inference_options_json;

bool test_stop_sequences_decode_json_escapes() {
    auto options = parse_inference_options_json(
        R"({"stop_sequences": ["\n\n", "C:\\", "caf\u00e9", "\"\""], "max_tokens": 64})");
    return options.stop_sequences == std::vector<std::string>{"\n\n", "C:\\", "caf\xc3\xa9", "\"\""};
}

bool test_stop_sequence_with_quote_as_last_option() {
    auto options = parse_inference_options_json(R"({"max_tokens": 64, "stop_sequences": ["\""]})");
    return options.stop_sequences == std::vector<std::string>{"\""};
}

int main() {
    TestUtils::TestRunner runner("Inference Options Tests");
    runner.run_test("stop_sequences_decode_json_escapes", test_stop_sequences_decode_json_escapes());
    runner.run_test("stop_sequence_with_quote_as_last_option", test_stop_sequence_with_quote_as_last_option());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
