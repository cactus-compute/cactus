#include "../cactus/ffi/cactus_ffi.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <iomanip>

constexpr int MAX_TOKENS = 512;
constexpr size_t MAX_BYTES_PER_TOKEN = 64;
constexpr size_t RESPONSE_BUFFER_SIZE = MAX_TOKENS * MAX_BYTES_PER_TOKEN;

void print_token(const char* token, uint32_t token_id, void* user_data) {
    std::cout << token << std::flush;
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {  // Use unsigned char to properly handle 0x00-0xFF
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {  // Control characters (0x00-0x1F)
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
                break;
        }
    }
    return o.str();
}

std::string unescape_json(const std::string& s) {
    std::string result;
    result.reserve(s.length());

    for (size_t i = 0; i < s.length(); i++) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            switch (s[i + 1]) {
                case '"':  result += '"'; i++; break;
                case '\\': result += '\\'; i++; break;
                case 'b':  result += '\b'; i++; break;
                case 'f':  result += '\f'; i++; break;
                case 'n':  result += '\n'; i++; break;
                case 'r':  result += '\r'; i++; break;
                case 't':  result += '\t'; i++; break;
                case 'u':  // Handle \uXXXX
                    if (i + 5 < s.length()) {
                        std::string hex = s.substr(i + 2, 4);
                        char* end;
                        int codepoint = std::strtol(hex.c_str(), &end, 16);
                        if (end == hex.c_str() + 4) {  // Valid hex
                            result += static_cast<char>(codepoint);
                            i += 5;
                        } else {
                            result += s[i];  // Invalid, keep backslash
                        }
                    } else {
                        result += s[i];
                    }
                    break;
                default:   result += s[i]; break;  // Unknown escape, keep backslash
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        std::cerr << "Example: " << argv[0] << " weights/gemma3-270m\n";
        return 1;
    }

    const char* model_path = argv[1];

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path, 4096, nullptr);

    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }

    std::cout << "Model loaded successfully!\n\n";

    std::vector<std::string> history;

    while (true) {
        std::cout << "You: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) continue;
        if (user_input == "quit" || user_input == "exit") break;
        if (user_input == "reset") {
            history.clear();
            cactus_reset(model);
            std::cout << "Conversation reset.\n\n";
            continue;
        }

        history.push_back(user_input);

        std::ostringstream messages_json;
        messages_json << "[";
        for (size_t i = 0; i < history.size(); i++) {
            if (i > 0) messages_json << ",";
            if (i % 2 == 0) {
                messages_json << "{\"role\":\"user\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            } else {
                messages_json << "{\"role\":\"assistant\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            }
        }
        messages_json << "]";

        std::string options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":"
                            + std::to_string(MAX_TOKENS)
                            + ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

        std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

        std::cout << "Assistant: ";
        int result = cactus_complete(
            model,
            messages_json.str().c_str(),
            response_buffer.data(),
            response_buffer.size(),
            options.c_str(),
            nullptr,
            print_token,
            nullptr
        );
        std::cout << "\n\n";

        if (result < 0) {
            std::cerr << "Error: " << response_buffer.data() << "\n\n";
            history.pop_back();
            continue;
        }

        std::string json_str(response_buffer.data(), response_buffer.size());
        const std::string search_str = "\"response\":\"";
        size_t response_start = json_str.find(search_str);
        if (response_start != std::string::npos) {
            response_start += search_str.length();
            size_t response_end = json_str.find("\"", response_start);
            while (response_end != std::string::npos) {
                size_t prior_backslashes = 0;
                for (size_t i = response_end; i > response_start && json_str[i - 1] == '\\'; i--) {
                    prior_backslashes++;
                }
                if (prior_backslashes % 2 == 0) {
                    break;
                }
                response_end = json_str.find("\"", response_end + 1);
            }
            if (response_end != std::string::npos) {
                std::string response = json_str.substr(response_start,
                                                       response_end - response_start);
                history.push_back(unescape_json(response));
            }
        }
    }

    std::cout << "Goodbye!\n";
    cactus_destroy(model);
    return 0;
}
