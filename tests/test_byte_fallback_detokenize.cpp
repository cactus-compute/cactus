#include "test_utils.h"
#include "../cactus/engine/engine.h"
#include <iostream>
#include <string>

using cactus::engine::reassemble_byte_fallback;
using cactus::engine::parse_byte_fallback_piece;

static bool eq(const std::string& got, const std::string& want, const char* label) {
    if (got == want) return true;
    std::cerr << "[mismatch] " << label << "\n  got : ";
    for (unsigned char c : got) {
        if (c >= 0x20 && c < 0x7F) std::cerr << static_cast<char>(c);
        else std::cerr << "\\x" << std::hex << static_cast<int>(c) << std::dec;
    }
    std::cerr << "\n  want: ";
    for (unsigned char c : want) {
        if (c >= 0x20 && c < 0x7F) std::cerr << static_cast<char>(c);
        else std::cerr << "\\x" << std::hex << static_cast<int>(c) << std::dec;
    }
    std::cerr << "\n";
    return false;
}

static bool test_parse_piece_valid() {
    uint8_t out = 0;
    if (!parse_byte_fallback_piece("<0xEA>", &out) || out != 0xEA) return false;
    if (!parse_byte_fallback_piece("<0x00>", &out) || out != 0x00) return false;
    if (!parse_byte_fallback_piece("<0xff>", &out) || out != 0xFF) return false;
    return true;
}

static bool test_parse_piece_rejects() {
    uint8_t out = 0;
    if (parse_byte_fallback_piece("<0xEA", &out)) return false;
    if (parse_byte_fallback_piece("<0xGG>", &out)) return false;
    if (parse_byte_fallback_piece("hello", &out)) return false;
    if (parse_byte_fallback_piece("<0xEA>x", &out)) return false;
    return true;
}

static bool test_korean_three_byte_reassembly() {
    std::string in = std::string("\xEC\x9E\xA0") + " <0xEA><0xB9><0xB0> " + std::string("\xEC\x88\x98");
    std::string want = std::string("\xEC\x9E\xA0") + " " + std::string("\xEA\xB9\xB0") + " " + std::string("\xEC\x88\x98");
    return eq(reassemble_byte_fallback(in), want, "korean_three_byte_reassembly");
}

static bool test_polish_two_byte_reassembly() {
    std::string in = "<0xC4><0x85><0xC4><0x99>";
    std::string want = std::string("\xC4\x85") + std::string("\xC4\x99");
    return eq(reassemble_byte_fallback(in), want, "polish_two_byte_reassembly");
}

static bool test_ascii_byte_fallback_space() {
    return eq(reassemble_byte_fallback("A<0x20>B"), "A B", "ascii_byte_fallback_space");
}

static bool test_invalid_partial_preserves_literal() {
    return eq(reassemble_byte_fallback("hi<0xEA>x"), "hi<0xEA>x", "invalid_partial_preserves_literal");
}

static bool test_trailing_incomplete_preserved() {
    return eq(reassemble_byte_fallback("hi<0xEA><0xB9>"), "hi<0xEA><0xB9>", "trailing_incomplete_preserved");
}

static bool test_empty_passthrough() {
    return eq(reassemble_byte_fallback(""), "", "empty_passthrough") &&
           eq(reassemble_byte_fallback("hello world"), "hello world", "no_byte_tokens_passthrough");
}

static bool test_two_runs_separated_by_text() {
    std::string in = "<0xEA><0xB9><0xB0>X<0xEC><0x88><0x98>";
    std::string want = std::string("\xEA\xB9\xB0") + "X" + std::string("\xEC\x88\x98");
    return eq(reassemble_byte_fallback(in), want, "two_runs_separated_by_text");
}

static bool test_lowercase_hex_accepted() {
    return eq(reassemble_byte_fallback("<0xea><0xb9><0xb0>"), std::string("\xEA\xB9\xB0"), "lowercase_hex_accepted");
}

int main() {
    TestUtils::TestRunner runner("Byte-fallback detokenize");
    runner.run_test("parse_piece_valid", test_parse_piece_valid());
    runner.run_test("parse_piece_rejects", test_parse_piece_rejects());
    runner.run_test("korean_three_byte_reassembly", test_korean_three_byte_reassembly());
    runner.run_test("polish_two_byte_reassembly", test_polish_two_byte_reassembly());
    runner.run_test("ascii_byte_fallback_space", test_ascii_byte_fallback_space());
    runner.run_test("invalid_partial_preserves_literal", test_invalid_partial_preserves_literal());
    runner.run_test("trailing_incomplete_preserved", test_trailing_incomplete_preserved());
    runner.run_test("empty_and_passthrough", test_empty_passthrough());
    runner.run_test("two_runs_separated_by_text", test_two_runs_separated_by_text());
    runner.run_test("lowercase_hex_accepted", test_lowercase_hex_accepted());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
