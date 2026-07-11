#include "../src/utils.h"
#include "test_utils.h"

bool test_role_first() {
    std::vector<std::string> images;
    auto messages = cactus::ffi::parse_messages_json(
        R"([{"role":"user","content":"hello"}])", images);
    return messages.size() == 1 && messages[0].role == "user" && messages[0].content == "hello";
}

bool test_content_first() {
    std::vector<std::string> images;
    auto messages = cactus::ffi::parse_messages_json(
        R"([{"content":"hello","role":"user"}])", images);
    return messages.size() == 1 && messages[0].role == "user" && messages[0].content == "hello";
}

bool test_content_first_with_escapes() {
    std::vector<std::string> images;
    auto messages = cactus::ffi::parse_messages_json(
        R"([{"content":"line one\nline \"two\"","role":"user"}])", images);
    return messages.size() == 1 && messages[0].content == "line one\nline \"two\"";
}

bool test_mixed_order_multi_message() {
    std::vector<std::string> images;
    auto messages = cactus::ffi::parse_messages_json(
        R"([{"role":"system","content":"be terse"},{"content":"hi there","role":"user"}])", images);
    return messages.size() == 2
        && messages[0].role == "system" && messages[0].content == "be terse"
        && messages[1].role == "user" && messages[1].content == "hi there";
}

bool test_content_first_with_image_paths() {
    std::vector<std::string> images;
    std::vector<std::string> audio;
    auto messages = cactus::ffi::parse_messages_json(
        R"([{"content":"describe this","role":"user","images":["a.png"],"audio":["b.wav"]}])",
        images, &audio);
    return messages.size() == 1 && messages[0].content == "describe this"
        && messages[0].images.size() == 1 && images.size() == 1
        && messages[0].audio.size() == 1 && audio.size() == 1;
}

int main() {
    TestUtils::TestRunner runner("Parse Messages Tests");

    runner.run_test("role_first", test_role_first());
    runner.run_test("content_first", test_content_first());
    runner.run_test("content_first_with_escapes", test_content_first_with_escapes());
    runner.run_test("mixed_order_multi_message", test_mixed_order_multi_message());
    runner.run_test("content_first_with_image_paths", test_content_first_with_image_paths());

    runner.print_summary();

    return runner.all_passed() ? 0 : 1;
}
