#ifndef CACTUS_JSON_ESCAPE_H
#define CACTUS_JSON_ESCAPE_H

#include <string>
#include <cstdio>

inline std::string escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    result += buf;
                } else {
                    result += static_cast<char>(c);
                }
                break;
        }
    }
    return result;
}

inline std::string escape_json_string(const char* str) {
    return str ? escape_json_string(std::string(str)) : std::string();
}

#endif
