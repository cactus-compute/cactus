#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

struct TokenizeResult {
    std::string input;
    std::vector<uint32_t> ids;
};

TokenizeResult tokenize(cactus_model_t model, const std::string& text) {
    std::vector<uint32_t> buf(512);
    size_t n = 0;
    cactus_tokenize(model, text.c_str(), buf.data(), buf.size(), &n);
    buf.resize(n);
    return {text, buf};
}

void print_result(const TokenizeResult& r) {
    std::cout << "  encode(" << std::quoted(r.input) << ") -> [";
    for (size_t i = 0; i < r.ids.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << r.ids[i];
    }
    std::cout << "]\n";
}

int main() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set\n";
        return 1;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cerr << "init failed\n"; return 1; }

    std::cout << "\n=== Special token IDs (expected) ===\n";
    std::cout << "  <|tool_call>  should be [48]\n";
    std::cout << "  <tool_call|>  should be [49]\n";
    std::cout << "  <|tool_response> should be [50]\n";
    std::cout << "  call:  should be [6639, 236787]\n";
    std::cout << "  get_weather  should be [828, 236779, 19323]\n";
    std::cout << "  send_message  should be [5738, 236779, 4375]\n";

    std::cout << "\n=== C++ tokenizer results ===\n";
    std::vector<std::string> tests = {
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "call:",
        "<|tool_call>call:",
        "get_weather",
        "send_message",
        "<|tool_call>call:get_weather{location:San Francisco}<tool_call|>",
    };

    bool all_ok = true;
    for (const auto& s : tests) {
        auto r = tokenize(model, s);
        print_result(r);

        // Spot-check key expectations
        if (s == "<|tool_call>" && (r.ids.size() != 1 || r.ids[0] != 48)) {
            std::cout << "  !! FAIL: expected [48]\n"; all_ok = false;
        }
        if (s == "<tool_call|>" && (r.ids.size() != 1 || r.ids[0] != 49)) {
            std::cout << "  !! FAIL: expected [49]\n"; all_ok = false;
        }
        if (s == "<|tool_response>" && (r.ids.size() != 1 || r.ids[0] != 50)) {
            std::cout << "  !! FAIL: expected [50]\n"; all_ok = false;
        }
        if (s == "call:" && r.ids != std::vector<uint32_t>{6639, 236787}) {
            std::cout << "  !! FAIL: expected [6639, 236787]\n"; all_ok = false;
        }
        if (s == "get_weather" && r.ids != std::vector<uint32_t>{828, 236779, 19323}) {
            std::cout << "  !! FAIL: expected [828, 236779, 19323]\n"; all_ok = false;
        }
        if (s == "send_message" && r.ids != std::vector<uint32_t>{5738, 236779, 4375}) {
            std::cout << "  !! FAIL: expected [5738, 236779, 4375]\n"; all_ok = false;
        }
    }

    // Also check the decode direction: does decoding special token IDs give back the right string?
    std::cout << "\n=== Decode direction (special tokens 48/49/50) ===\n";
    // Tokenize a round-trip string and check
    auto rt = tokenize(model, "<|tool_call>call:get_weather{}<tool_call|>");
    std::cout << "  Full round-trip ids: [";
    for (size_t i = 0; i < rt.ids.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << rt.ids[i];
    }
    std::cout << "]\n";
    bool has_48 = std::find(rt.ids.begin(), rt.ids.end(), 48u) != rt.ids.end();
    bool has_49 = std::find(rt.ids.begin(), rt.ids.end(), 49u) != rt.ids.end();
    std::cout << "  Contains 48 (<|tool_call>): " << (has_48 ? "YES" : "NO") << "\n";
    std::cout << "  Contains 49 (<tool_call|>): " << (has_49 ? "YES" : "NO") << "\n";
    if (!has_48) { std::cout << "  !! FAIL: <|tool_call> not tokenized as single token 48\n"; all_ok = false; }
    if (!has_49) { std::cout << "  !! FAIL: <tool_call|> not tokenized as single token 49\n"; all_ok = false; }

    cactus_destroy(model);
    std::cout << "\n" << (all_ok ? "ALL OK" : "FAILURES DETECTED") << "\n";
    return all_ok ? 0 : 1;
}
