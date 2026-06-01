#pragma once

#include <string>
#include <vector>

// MiniCPM-style XML tool calling.
// Tools are declared in the system prompt inside <tools>...</tools> and the
// model emits calls as <function name="..."><param name="...">value</param></function>.
namespace minicpm {

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

// A param value is plain text in the XML; emit it as a JSON value, inferring
// number/bool/null so strict consumers get typed arguments, else a JSON string.
inline std::string value_to_json(const std::string& raw) {
    size_t b = raw.find_first_not_of(" \t\r\n");
    size_t e = raw.find_last_not_of(" \t\r\n");
    std::string v = (b == std::string::npos) ? "" : raw.substr(b, e - b + 1);
    if (v == "true" || v == "false" || v == "null") return v;
    if (!v.empty()) {
        bool numeric = true, seen_digit = false;
        for (size_t i = 0; i < v.size(); ++i) {
            char c = v[i];
            if (c >= '0' && c <= '9') { seen_digit = true; continue; }
            if (c == '-' && i == 0) continue;
            if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') continue;
            numeric = false; break;
        }
        if (numeric && seen_digit) return v;
    }
    return "\"" + json_escape(v) + "\"";
}

template<typename ToolFunction>
inline std::string format_tools(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::string result =
        "# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for (const auto& tool : tools) {
        std::string schema;
        auto it = tool.parameters.find("schema");
        if (it != tool.parameters.end()) schema = it->second;
        if (schema.empty()) schema = "{\"type\": \"object\", \"properties\": {}}";
        result += "\n{\"type\": \"function\", \"function\": {\"name\": \"" + json_escape(tool.name) +
                  "\", \"description\": \"" + json_escape(tool.description) +
                  "\", \"parameters\": " + schema + "}}";
    }
    result +=
        "\n</tools>\n\nTool usage guidelines:\n"
        "- You may call zero or more functions. If no function call is needed, answer normally.\n"
        "- When calling a function, return an XML object using: "
        "<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>";
    return result;
}

// Render a prior assistant tool call (name + JSON-object arguments) back into
// MiniCPM XML for prompt replay: <function name="N"><param name="k">v</param>...</function>.
inline std::string render_tool_call(const std::string& name, const std::string& args_json) {
    std::string out = "<function name=\"" + name + "\">";
    size_t i = args_json.find('{');
    if (i != std::string::npos) {
        i++;
        while (i < args_json.size()) {
            while (i < args_json.size() && (args_json[i] == ' ' || args_json[i] == '\t' ||
                   args_json[i] == '\n' || args_json[i] == '\r' || args_json[i] == ',')) i++;
            if (i >= args_json.size() || args_json[i] == '}') break;
            if (args_json[i] != '"') break;
            i++;
            std::string key;
            while (i < args_json.size() && args_json[i] != '"') {
                if (args_json[i] == '\\' && i + 1 < args_json.size()) i++;
                key += args_json[i++];
            }
            i++;  // closing quote
            while (i < args_json.size() && args_json[i] != ':') i++;
            i++;  // colon
            while (i < args_json.size() && (args_json[i] == ' ' || args_json[i] == '\t')) i++;
            std::string val;
            if (i < args_json.size() && args_json[i] == '"') {
                i++;
                while (i < args_json.size() && args_json[i] != '"') {
                    if (args_json[i] == '\\' && i + 1 < args_json.size()) {
                        char n = args_json[i + 1];
                        val += (n == 'n') ? '\n' : (n == 't') ? '\t' : (n == 'r') ? '\r' : n;
                        i += 2;
                        continue;
                    }
                    val += args_json[i++];
                }
                i++;  // closing quote
            } else {
                int depth = 0;
                while (i < args_json.size()) {
                    char c = args_json[i];
                    if (c == '{' || c == '[') depth++;
                    else if (c == '}' || c == ']') { if (depth == 0) break; depth--; }
                    else if (c == ',' && depth == 0) break;
                    val += c;
                    i++;
                }
            }
            bool needs_cdata = val.find('<') != std::string::npos ||
                               val.find('&') != std::string::npos ||
                               val.find('\n') != std::string::npos;
            out += "<param name=\"" + key + "\">";
            out += needs_cdata ? ("<![CDATA[" + val + "]]>") : val;
            out += "</param>";
        }
    }
    out += "</function>";
    return out;
}

// Extract the value of an attribute like name="..." starting at/after `from`.
inline std::string extract_attr(const std::string& text, const std::string& attr, size_t from, size_t limit) {
    size_t p = text.find(attr, from);
    if (p == std::string::npos || p >= limit) return "";
    p += attr.size();
    size_t end = text.find('"', p);
    if (end == std::string::npos || end > limit) return "";
    return text.substr(p, end - p);
}

// Parse <function name="..."><param name="...">value</param>...</function> blocks
// out of `text`, appending {"name":...,"arguments":{...}} JSON strings to
// `function_calls` and stripping the matched spans from `text`.
inline void parse_function_calls(std::string& text, std::vector<std::string>& function_calls) {
    const std::string fn_open = "<function name=\"";
    const std::string fn_close = "</function>";
    std::string stripped;
    size_t cursor = 0;
    while (true) {
        size_t start = text.find("<function", cursor);
        if (start == std::string::npos) break;
        size_t name_pos = text.find(fn_open, start);
        size_t close = text.find(fn_close, start);
        if (name_pos == std::string::npos || close == std::string::npos) break;

        stripped += text.substr(cursor, start - cursor);

        size_t name_start = name_pos + fn_open.size();
        size_t name_end = text.find('"', name_start);
        std::string name = (name_end == std::string::npos) ? "" : text.substr(name_start, name_end - name_start);

        std::string args = "{";
        bool first = true;
        size_t pp = name_end;
        const std::string param_open = "<param name=\"";
        const std::string param_close = "</param>";
        while (true) {
            size_t param = text.find(param_open, pp);
            if (param == std::string::npos || param > close) break;
            size_t key_start = param + param_open.size();
            size_t key_end = text.find('"', key_start);
            if (key_end == std::string::npos) break;
            std::string key = text.substr(key_start, key_end - key_start);
            size_t val_start = text.find('>', key_end);
            if (val_start == std::string::npos) break;
            val_start += 1;
            size_t val_end = text.find(param_close, val_start);
            if (val_end == std::string::npos) break;
            std::string value = text.substr(val_start, val_end - val_start);
            const std::string cdata_open = "<![CDATA[";
            const std::string cdata_close = "]]>";
            size_t cd = value.find(cdata_open);
            if (cd != std::string::npos) {
                size_t cd_end = value.find(cdata_close, cd + cdata_open.size());
                if (cd_end != std::string::npos) {
                    value = value.substr(cd + cdata_open.size(), cd_end - (cd + cdata_open.size()));
                }
            }
            if (!first) args += ", ";
            first = false;
            args += "\"" + json_escape(key) + "\": " + value_to_json(value);
            pp = val_end + param_close.size();
        }
        args += "}";

        function_calls.push_back("{\"name\": \"" + json_escape(name) + "\", \"arguments\": " + args + "}");
        cursor = close + fn_close.size();
    }
    stripped += text.substr(cursor);
    text = stripped;
}

}  // namespace minicpm
