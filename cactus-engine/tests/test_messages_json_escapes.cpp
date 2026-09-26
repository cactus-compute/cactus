#include "test_utils.h"
#include "../src/utils.h"

using cactus::ffi::parse_messages_json;

bool test_content_decodes_json_escapes() {
    std::vector<std::string> images;
    auto messages = parse_messages_json(
        R"([{"role":"user","content":"func main() {\n\tprint(\"C:\\temp\\new\")\n} caf\u00e9"}])", images);
    return messages.size() == 1
        && messages[0].content == "func main() {\n\tprint(\"C:\\temp\\new\")\n} caf\xc3\xa9";
}

bool test_content_ending_in_backslash() {
    std::vector<std::string> images;
    auto messages = parse_messages_json(
        R"([{"role":"user","content":"the path is C:\\"},{"role":"assistant","content":"ok"}])", images);
    return messages.size() == 2
        && messages[0].content == "the path is C:\\"
        && messages[1].role == "assistant" && messages[1].content == "ok";
}

int main() {
    TestUtils::TestRunner runner("Message Content Escape Tests");
    runner.run_test("content_decodes_json_escapes", test_content_decodes_json_escapes());
    runner.run_test("content_ending_in_backslash", test_content_ending_in_backslash());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
