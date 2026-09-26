#include "test_utils.h"
#include "../src/utils.h"

using cactus::ffi::parse_messages_json;

bool test_unbalanced_brace_in_content_keeps_later_messages() {
    std::vector<std::string> images;
    auto messages = parse_messages_json(
        R"([{"role":"user","content":"why does `if (x) {` not compile?"},)"
        R"({"role":"assistant","content":"it is missing its closing brace"},)"
        R"({"role":"user","content":"thanks"}])", images);
    return messages.size() == 3
        && messages[1].role == "assistant" && messages[1].content == "it is missing its closing brace"
        && messages[2].role == "user" && messages[2].content == "thanks";
}

bool test_images_stay_with_their_message() {
    std::vector<std::string> images;
    auto messages = parse_messages_json(
        R"([{"role":"user","content":"what is {"},)"
        R"({"role":"user","content":"describe this","images":["a.png"]}])", images);
    return messages.size() == 2
        && messages[0].images.empty()
        && messages[1].images.size() == 1 && images.size() == 1;
}

int main() {
    TestUtils::TestRunner runner("Message Boundary Tests");
    runner.run_test("unbalanced_brace_in_content_keeps_later_messages", test_unbalanced_brace_in_content_keeps_later_messages());
    runner.run_test("images_stay_with_their_message", test_images_stay_with_their_message());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
