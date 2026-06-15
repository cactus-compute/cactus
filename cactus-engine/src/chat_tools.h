#pragma once

#include <cctype>
#include <string>
#include <vector>

namespace chat_tools {

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += c;
                }
        }
    }
    return out;
}

inline std::string respace_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + s.size() / 8);
    bool in_str = false, esc = false;
    for (char c : s) {
        if (in_str) {
            out += c;
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; out += c; continue; }
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        out += c;
        if (c == ',' || c == ':') out += ' ';
    }
    return out;
}

template <typename ToolFunction>
inline std::string tool_object_json(const ToolFunction& tool) {
    std::string schema;
    auto it = tool.parameters.find("schema");
    if (it != tool.parameters.end()) schema = it->second;
    std::string obj = "{\"type\":\"function\",\"function\":{\"name\":\"" + json_escape(tool.name) +
                      "\",\"description\":\"" + json_escape(tool.description) + "\"";
    if (!schema.empty()) obj += ",\"parameters\":" + schema;
    obj += "}}";
    return respace_json(obj);
}

template <typename ToolFunction>
inline std::string serialize_tools_qwen(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::string r =
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for (const auto& t : tools) r += "\n" + tool_object_json(t);
    r += "\n</tools>\n\nFor each function call, return a json object with function name and "
         "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
         "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    return r;
}

template <typename ToolFunction>
inline std::string serialize_tools_lfm2(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::string r = "List of tools: <|tool_list_start|>[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i) r += ", ";
        r += tool_object_json(tools[i]);
    }
    r += "]<|tool_list_end|>";
    return r;
}

inline void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
}

inline std::string py_value(const std::string& s, size_t& p);

inline std::string py_string(const std::string& s, size_t& p) {
    std::string out = "\"";
    ++p;
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) { out += s[p]; out += s[p + 1]; p += 2; }
        else { out += s[p++]; }
    }
    if (p < s.size()) ++p;
    out += "\"";
    return out;
}

inline std::string py_value(const std::string& s, size_t& p) {
    skip_ws(s, p);
    if (p >= s.size()) return "";
    char c = s[p];
    if (c == '"') return py_string(s, p);
    if (c == '{') {
        std::string out = "{";
        ++p; bool first = true;
        while (true) {
            skip_ws(s, p);
            if (p >= s.size() || s[p] == '}') { if (p < s.size()) ++p; break; }
            if (s[p] == ',') { ++p; continue; }
            std::string key = py_string(s, p);
            skip_ws(s, p);
            if (p < s.size() && s[p] == ':') ++p;
            std::string val = py_value(s, p);
            if (!first) out += ", ";
            first = false;
            out += key + ": " + val;
        }
        return out + "}";
    }
    if (c == '[') {
        std::string out = "[";
        ++p; bool first = true;
        while (true) {
            skip_ws(s, p);
            if (p >= s.size() || s[p] == ']') { if (p < s.size()) ++p; break; }
            if (s[p] == ',') { ++p; continue; }
            std::string val = py_value(s, p);
            if (!first) out += ", ";
            first = false;
            out += val;
        }
        return out + "]";
    }
    if (s.compare(p, 4, "true") == 0) { p += 4; return "True"; }
    if (s.compare(p, 5, "false") == 0) { p += 5; return "False"; }
    if (s.compare(p, 4, "null") == 0) { p += 4; return "None"; }
    size_t start = p;
    while (p < s.size() && (std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '.' ||
           s[p] == '-' || s[p] == '+' || s[p] == 'e' || s[p] == 'E')) ++p;
    return s.substr(start, p - start);
}

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline void extract_qwen_tool_calls(std::string& text, std::vector<std::string>& calls) {
    const std::string OPEN = "<tool_call>", CLOSE = "</tool_call>";
    size_t pos;
    while ((pos = text.find(OPEN)) != std::string::npos) {
        size_t cs = pos + OPEN.size();
        size_t ce = text.find(CLOSE, cs);
        std::string inner = trim(text.substr(cs, (ce == std::string::npos ? text.size() : ce) - cs));
        if (!inner.empty() && inner.front() == '{') calls.push_back(inner);
        text.erase(pos, (ce == std::string::npos ? text.size() : ce + CLOSE.size()) - pos);
    }
}

inline std::string py_literal_to_json(const std::string& s, size_t& p) {
    skip_ws(s, p);
    if (p >= s.size()) return "null";
    char c = s[p];
    if (c == '"' || c == '\'') {
        char q = c; ++p;
        std::string out = "\"";
        while (p < s.size() && s[p] != q) {
            if (s[p] == '\\' && p + 1 < s.size()) { out += s[p]; out += s[p + 1]; p += 2; continue; }
            if (s[p] == '"') out += '\\';
            out += s[p++];
        }
        if (p < s.size()) ++p;
        return out + "\"";
    }
    if (c == '[' || c == '{') {
        char close = (c == '[') ? ']' : '}';
        std::string out(1, c == '[' ? '[' : '{');
        ++p; bool first = true;
        while (true) {
            skip_ws(s, p);
            if (p >= s.size() || s[p] == close) { if (p < s.size()) ++p; break; }
            if (s[p] == ',') { ++p; continue; }
            if (!first) out += ", ";
            first = false;
            std::string key_or_val = py_literal_to_json(s, p);
            skip_ws(s, p);
            if (close == '}' && p < s.size() && s[p] == ':') {
                ++p;
                out += key_or_val + ": " + py_literal_to_json(s, p);
            } else {
                out += key_or_val;
            }
        }
        return out + close;
    }
    if (s.compare(p, 4, "True") == 0) { p += 4; return "true"; }
    if (s.compare(p, 5, "False") == 0) { p += 5; return "false"; }
    if (s.compare(p, 4, "None") == 0) { p += 4; return "null"; }
    size_t start = p;
    while (p < s.size() && s[p] != ',' && s[p] != ')' && s[p] != ']' && s[p] != '}') ++p;
    return trim(s.substr(start, p - start));
}

inline void extract_lfm2_tool_calls(std::string& text, std::vector<std::string>& calls) {
    const std::string OPEN = "<|tool_call_start|>", CLOSE = "<|tool_call_end|>";
    size_t pos;
    while ((pos = text.find(OPEN)) != std::string::npos) {
        size_t cs = pos + OPEN.size();
        size_t ce = text.find(CLOSE, cs);
        std::string inner = text.substr(cs, (ce == std::string::npos ? text.size() : ce) - cs);
        size_t p = 0;
        skip_ws(inner, p);
        if (p < inner.size() && inner[p] == '[') ++p;
        while (p < inner.size()) {
            skip_ws(inner, p);
            if (p >= inner.size() || inner[p] == ']') break;
            if (inner[p] == ',') { ++p; continue; }
            size_t name_start = p;
            while (p < inner.size() && (std::isalnum(static_cast<unsigned char>(inner[p])) ||
                   inner[p] == '_' || inner[p] == '.')) ++p;
            std::string name = inner.substr(name_start, p - name_start);
            skip_ws(inner, p);
            if (name.empty() || p >= inner.size() || inner[p] != '(') { ++p; continue; }
            ++p;  // (
            std::string args = "{";
            bool first = true;
            while (true) {
                skip_ws(inner, p);
                if (p >= inner.size() || inner[p] == ')') { if (p < inner.size()) ++p; break; }
                if (inner[p] == ',') { ++p; continue; }
                size_t ks = p;
                while (p < inner.size() && (std::isalnum(static_cast<unsigned char>(inner[p])) || inner[p] == '_')) ++p;
                std::string key = inner.substr(ks, p - ks);
                skip_ws(inner, p);
                if (p < inner.size() && inner[p] == '=') ++p;
                std::string val = py_literal_to_json(inner, p);
                if (!first) args += ", ";
                first = false;
                args += "\"" + key + "\": " + val;
            }
            args += "}";
            calls.push_back("{\"name\": \"" + name + "\", \"arguments\": " + args + "}");
        }
        text.erase(pos, (ce == std::string::npos ? text.size() : ce + CLOSE.size()) - pos);
    }
}

inline std::string pythonic_call(const std::string& name, const std::string& args_json) {
    std::string out = name + "(";
    size_t p = 0;
    skip_ws(args_json, p);
    bool first = true;
    if (p < args_json.size() && args_json[p] == '{') {
        ++p;
        while (true) {
            skip_ws(args_json, p);
            if (p >= args_json.size() || args_json[p] == '}') break;
            if (args_json[p] == ',') { ++p; continue; }
            if (args_json[p] != '"') break;
            std::string key;
            { std::string q = py_string(args_json, p); key = q.substr(1, q.size() - 2); }
            skip_ws(args_json, p);
            if (p < args_json.size() && args_json[p] == ':') ++p;
            std::string val = py_value(args_json, p);
            if (!first) out += ", ";
            first = false;
            out += key + "=" + val;
        }
    }
    return out + ")";
}

}  // namespace chat_tools
