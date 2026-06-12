#include "cactus_engine.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> load_tokens(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open token input: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> tokens;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        tokens.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    if (tokens.empty()) {
        throw std::runtime_error("token input is empty: " + path);
    }
    return tokens;
}

std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

std::string complete_options(size_t max_tokens) {
    return R"({"max_tokens": )" + std::to_string(max_tokens) +
           R"(, "temperature": 0.0, "top_p": 0.0, "top_k": 1, "telemetry_enabled": false,)"
           R"( "stop_sequences": ["<|im_end|>", "<end_of_turn>"]})";
}

int usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_dir> --tokens <ids.csv> --decode <n>\n"
              << "       " << prog << " <model_dir> --prompt <text> --max-tokens <n> [--warmup]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage(argv[0]);
    std::string model_dir = argv[1];
    std::string tokens_path;
    std::string prompt;
    size_t decode_tokens = 32;
    size_t max_tokens = 32;
    bool warmup = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) tokens_path = argv[++i];
        else if (arg == "--decode" && i + 1 < argc) decode_tokens = std::stoul(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoul(argv[++i]);
        else if (arg == "--warmup") warmup = true;
        else return usage(argv[0]);
    }
    if (tokens_path.empty() == prompt.empty()) return usage(argv[0]);

    try {
        cactus_log_set_level(2);
        cactus_model_t model = cactus_init(model_dir.c_str(), nullptr, false);
        if (!model) {
            const char* error = cactus_get_last_error();
            std::cerr << (error ? error : "cactus_init failed") << "\n";
            return 1;
        }

        std::vector<char> buffer(65536);
        bool ok;
        if (!tokens_path.empty()) {
            auto tokens = load_tokens(tokens_path);
            int rc = cactus_benchmark_tokens(model, tokens.data(), tokens.size(),
                                             decode_tokens, buffer.data(), buffer.size());
            ok = rc >= 0;
        } else {
            std::string messages = R"([{"role": "user", "content": ")" + escape_json(prompt) + R"("}])";
            if (warmup) {
                cactus_complete(model, messages.c_str(), buffer.data(), buffer.size(),
                                complete_options(16).c_str(), nullptr, nullptr, nullptr, nullptr, 0);
                cactus_reset(model);
            }
            int rc = cactus_complete(model, messages.c_str(), buffer.data(), buffer.size(),
                                     complete_options(max_tokens).c_str(), nullptr, nullptr, nullptr, nullptr, 0);
            ok = rc > 0;
        }
        cactus_destroy(model);
        if (!ok) {
            const char* error = cactus_get_last_error();
            std::cerr << (error ? error : buffer.data()) << "\n";
            return 1;
        }
        std::cout << buffer.data() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
