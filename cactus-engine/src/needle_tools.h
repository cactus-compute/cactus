#pragma once

#include "engine.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

namespace cactus {
namespace engine {
namespace needle {

inline std::string to_snake_case(const std::string& name) {
    std::string normalized;
    normalized.reserve(name.size() * 2);
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            normalized += c;
        } else if (normalized.empty() || normalized.back() != '_') {
            normalized += '_';
        }
    }

    std::string with_word_boundaries;
    with_word_boundaries.reserve(normalized.size() * 2);
    for (size_t i = 0; i < normalized.size(); i++) {
        char c = normalized[i];
        if (i > 0 && std::isupper(static_cast<unsigned char>(c))) {
            char prev = normalized[i - 1];
            if (std::islower(static_cast<unsigned char>(prev)) || std::isdigit(static_cast<unsigned char>(prev))) {
                with_word_boundaries += '_';
            }
        }
        with_word_boundaries += c;
    }

    std::string with_acronym_boundaries;
    with_acronym_boundaries.reserve(with_word_boundaries.size() * 2);
    for (size_t i = 0; i < with_word_boundaries.size(); i++) {
        with_acronym_boundaries += with_word_boundaries[i];
        if (i + 1 < with_word_boundaries.size() &&
            std::isupper(static_cast<unsigned char>(with_word_boundaries[i])) &&
            std::isupper(static_cast<unsigned char>(with_word_boundaries[i + 1])) &&
            i + 2 < with_word_boundaries.size() &&
            std::islower(static_cast<unsigned char>(with_word_boundaries[i + 2]))) {
            with_acronym_boundaries += '_';
        }
    }

    std::string result;
    result.reserve(with_acronym_boundaries.size());
    bool prev_underscore = false;
    for (char c : with_acronym_boundaries) {
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
        for (const auto& [snake, original] : name_map) {
            std::string from = "\"name\":\"" + snake + "\"";
            size_t pos = call.find(from);
            if (pos == std::string::npos) {
                from = "\"name\": \"" + snake + "\"";
                pos = call.find(from);
            }
            if (pos != std::string::npos) {
                std::string to = from.substr(0, from.size() - snake.size() - 1) + original + "\"";
                call.replace(pos, from.size(), to);
                break;
            }
        }
    }
}

inline std::string format_query_text(const std::vector<ChatMessage>& messages) {
    std::string system_text;
    std::string user_query;

    for (const auto& msg : messages) {
        if (msg.role == "system" || msg.role == "developer") {
            if (!system_text.empty()) system_text += "\n";
            system_text += msg.content;
        } else if (msg.role == "user") {
            user_query = msg.content;
        }
    }

    if (user_query.empty() && !messages.empty()) {
        user_query = messages.back().content;
    }
    if (system_text.empty()) return user_query;
    if (user_query.empty()) return system_text;
    return system_text + "\n\n" + user_query;
}

} // namespace needle
} // namespace engine
} // namespace cactus
