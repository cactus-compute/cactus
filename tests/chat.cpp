#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/telemetry/telemetry.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <thread>

constexpr int MAX_TOKENS = 512;
constexpr size_t MAX_BYTES_PER_TOKEN = 64;
constexpr size_t RESPONSE_BUFFER_SIZE = MAX_TOKENS * MAX_BYTES_PER_TOKEN;

namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    const std::string CYAN    = "\033[36m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string RED     = "\033[31m";
    const std::string GRAY    = "\033[90m";
}

bool supports_color() {
#ifdef _WIN32
    return false; 
#else
    const char* term = std::getenv("TERM");
    return term && std::string(term) != "dumb";
#endif
}

bool use_colors = supports_color();

std::string colored(const std::string& text, const std::string& color) {
    if (!use_colors) return text;
    return color + text + Color::RESET;
}

void print_separator(char ch = '-', int width = 60) {
    std::cout << colored(std::string(width, ch), Color::DIM) << "\n";
}

void print_header() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("           🌵 CACTUS CHAT INTERFACE 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Commands:", Color::YELLOW) << "\n";
    std::cout << "  • " << colored("reset", Color::CYAN) << " - Clear conversation history\n";
    std::cout << "  • " << colored("exit", Color::CYAN) << " or " << colored("quit", Color::CYAN) << " - Exit the program\n";
    print_separator();
    std::cout << "\n";
}

struct TokenPrinter {
    bool first_token = true;
    int token_count = 0;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point first_token_time;
    double time_to_first_token = 0.0;

    void reset() {
        first_token = true;
        token_count = 0;
        time_to_first_token = 0.0;
        start_time = std::chrono::steady_clock::now();
    }

    void print(const char* token) {
        if (first_token) {
            first_token = false;
            first_token_time = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
            time_to_first_token = latency.count() / 1000.0;
        }
        std::cout << token << std::flush;
        token_count++;
    }

    void print_stats() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double total_seconds = duration.count() / 1000.0;
        double tokens_per_second = token_count / total_seconds;

        // Format the stats with fixed decimal places
        std::ostringstream stats;
        stats << std::fixed << std::setprecision(3);
        stats << "[" << token_count << " tokens | ";
        stats << "latency: " << time_to_first_token << "s | ";
        stats << "total: " << total_seconds << "s | ";
        stats << std::setprecision(0) << static_cast<int>(tokens_per_second) << " tok/s]";

        std::cout << "\n" << colored(stats.str(), Color::GRAY) << "\n";
    }
};

TokenPrinter* g_printer = nullptr;

void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    if (g_printer) {
        g_printer->print(token);
    }
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {  
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {  
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
                case 'u':  
                    if (i + 5 < s.length()) {
                        std::string hex = s.substr(i + 2, 4);
                        char* end;
                        int codepoint = std::strtol(hex.c_str(), &end, 16);
                        if (end == hex.c_str() + 4) {  
                            result += static_cast<char>(codepoint);
                            i += 5;
                        } else {
                            result += s[i];  
                        }
                    } else {
                        result += s[i];
                    }
                    break;
                default:   result += s[i]; break; 
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Missing model path\n";
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        std::cerr << "Example: " << argv[0] << " weights/lfm2-1.2B\n";
        return 1;
    }

    const char* model_path = argv[1];

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);

    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        return 1;
    }

    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    print_header();

    std::vector<std::string> history;
    TokenPrinter printer;
    g_printer = &printer;

    while (true) {
        std::cout << colored("You: ", Color::BLUE + Color::BOLD);
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) continue;

        if (user_input == "quit" || user_input == "exit") {
            break;
        }

        if (user_input == "reset") {
            history.clear();
            cactus_reset(model);
            std::cout << colored("🔄 Conversation reset.\n", Color::YELLOW);
            print_separator();
            std::cout << "\n";
            continue;
        }

        history.push_back(user_input);

        // Build the messages JSON
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

        std::cout << colored("Assistant: ", Color::GREEN + Color::BOLD);

        printer.reset();
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

        if (result >= 0) {
            printer.print_stats();
        }

        std::cout << "\n";
        print_separator();
        std::cout << "\n";

        if (result < 0) {
            std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                      << response_buffer.data() << "\n\n";
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

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return 0;
}
