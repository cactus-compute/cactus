#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace needle {

inline std::string to_snake_case(const std::string& name) {
    std::string s;
    s.reserve(name.size() * 2);
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            s += c;
        } else {
            if (s.empty() || s.back() != '_') s += '_';
        }
    }

    std::string s2;
    s2.reserve(s.size() * 2);
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (i > 0 && std::isupper(static_cast<unsigned char>(c))) {
            char prev = s[i - 1];
            if (std::islower(static_cast<unsigned char>(prev)) || std::isdigit(static_cast<unsigned char>(prev))) {
                s2 += '_';
            }
        }
        s2 += c;
    }

    std::string s3;
    s3.reserve(s2.size() * 2);
    for (size_t i = 0; i < s2.size(); i++) {
        s3 += s2[i];
        if (i + 1 < s2.size() &&
            std::isupper(static_cast<unsigned char>(s2[i])) &&
            std::isupper(static_cast<unsigned char>(s2[i + 1]))) {
            if (i + 2 < s2.size() && std::islower(static_cast<unsigned char>(s2[i + 2]))) {
                s3 += '_';
            }
        }
    }

    std::string result;
    result.reserve(s3.size());
    bool prev_underscore = false;
    for (char c : s3) {
        if (c == '_') {
            if (!prev_underscore) result += '_';
            prev_underscore = true;
        } else {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            prev_underscore = false;
        }
    }

    size_t start = result.find_first_not_of('_');
    if (start == std::string::npos) return result;
    size_t end = result.find_last_not_of('_');
    return result.substr(start, end - start + 1);
}

inline void restore_tool_names(std::vector<std::string>& function_calls,
                               const std::unordered_map<std::string, std::string>& name_map) {
    if (name_map.empty()) return;
    for (auto& call : function_calls) {
        for (const auto& [snake, orig] : name_map) {
            std::string from = "\"name\":\"" + snake + "\"";
            size_t pos = call.find(from);
            if (pos == std::string::npos) {
                from = "\"name\": \"" + snake + "\"";
                pos = call.find(from);
            }
            if (pos != std::string::npos) {
                std::string to = from.substr(0, from.size() - snake.size() - 1) + orig + "\"";
                call.replace(pos, from.size(), to);
                break;
            }
        }
    }
}

} // namespace needle
