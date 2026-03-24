#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>

namespace gemma {

inline std::string to_upper(const std::string& s) {
    std::string result = s;
    for (auto& c : result) c = std::toupper(c);
    return result;
}

inline std::string escape(const std::string& s) {
    return "<escape>" + s + "<escape>";
}

inline void skip_whitespace(const std::string& json, size_t& pos) {
    while (pos < json.length() && std::isspace(json[pos])) pos++;
}

inline std::string extract_json_string(const std::string& json, size_t& pos) {
    std::string value;
    while (pos < json.length() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.length()) {
            pos++;
            if (json[pos] == 'n') value += '\n';
            else if (json[pos] == 't') value += '\t';
            else if (json[pos] == 'r') value += '\r';
            else if (json[pos] == '"') value += '"';
            else if (json[pos] == '\\') value += '\\';
            else value += json[pos];
        } else {
            value += json[pos];
        }
        pos++;
    }
    if (pos < json.length()) pos++; 
    return value;
}

std::string format_argument(const std::string& json, size_t& pos, bool escape_keys);
std::string format_parameters(const std::string& properties_json, const std::string& /*required_json*/);

inline std::string format_argument(const std::string& json, size_t& pos, bool escape_keys = true) {
    skip_whitespace(json, pos);
    if (pos >= json.length()) return "";

    char c = json[pos];

    if (c == '"') {
        std::string value = extract_json_string(json, pos);
        return escape(value);
    } else if (c == '{') {
        std::string result = "{";
        pos++; 
        bool first = true;

        while (pos < json.length()) {
            skip_whitespace(json, pos);
            if (pos >= json.length() || json[pos] == '}') { pos++; break; }
            if (json[pos] == ',') { pos++; continue; }

            if (json[pos] != '"') break;
            pos++;
            std::string key = extract_json_string(json, pos);

            skip_whitespace(json, pos);
            if (pos < json.length() && json[pos] == ':') pos++;

            std::string value = format_argument(json, pos, escape_keys);

            if (!first) result += ",";
            first = false;
            if (escape_keys) {
                result += escape(key) + ":" + value;
            } else {
                result += key + ":" + value;
            }
        }
        result += "}";
        return result;
    } else if (c == '[') {
        std::string result = "[";
        pos++; 
        bool first = true;

        while (pos < json.length()) {
            skip_whitespace(json, pos);
            if (pos >= json.length() || json[pos] == ']') { pos++; break; }
            if (json[pos] == ',') { pos++; continue; }

            std::string value = format_argument(json, pos, escape_keys);

            if (!first) result += ",";
            first = false;
            result += value;
        }
        result += "]";
        return result;
    } else if (json.compare(pos, 4, "true") == 0) {
        pos += 4;
        return "true";
    } else if (json.compare(pos, 5, "false") == 0) {
        pos += 5;
        return "false";
    } else if (json.compare(pos, 4, "null") == 0) {
        pos += 4;
        return "null";
    } else {
        size_t start = pos;
        while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '.' ||
               json[pos] == '-' || json[pos] == '+' || json[pos] == 'e' || json[pos] == 'E')) {
            pos++;
        }
        return json.substr(start, pos - start);
    }
}

inline std::map<std::string, std::string> parse_json_object_raw(const std::string& json, size_t& pos) {
    std::map<std::string, std::string> result;
    skip_whitespace(json, pos);
    if (pos >= json.length() || json[pos] != '{') return result;
    pos++; 

    while (pos < json.length()) {
        skip_whitespace(json, pos);
        if (pos >= json.length() || json[pos] == '}') { pos++; break; }
        if (json[pos] == ',') { pos++; continue; }

        if (json[pos] != '"') break;
        pos++;
        std::string key = extract_json_string(json, pos);

        skip_whitespace(json, pos);
        if (pos < json.length() && json[pos] == ':') pos++;
        skip_whitespace(json, pos);

        size_t value_start = pos;
        if (json[pos] == '"') {
            pos++;
            while (pos < json.length() && json[pos] != '"') {
                if (json[pos] == '\\') pos++;
                pos++;
            }
            pos++; 
        } else if (json[pos] == '{') {
            int depth = 1;
            pos++;
            while (pos < json.length() && depth > 0) {
                if (json[pos] == '{') depth++;
                else if (json[pos] == '}') depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < json.length() && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
        } else if (json[pos] == '[') {
            int depth = 1;
            pos++;
            while (pos < json.length() && depth > 0) {
                if (json[pos] == '[') depth++;
                else if (json[pos] == ']') depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < json.length() && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
        } else {
            while (pos < json.length() && json[pos] != ',' && json[pos] != '}') pos++;
        }
        result[key] = json.substr(value_start, pos - value_start);
    }
    return result;
}

inline std::string get_json_string_value(const std::string& json, size_t pos) {
    skip_whitespace(json, pos);
    if (pos < json.length() && json[pos] == '"') {
        pos++;
        return extract_json_string(json, pos);
    }
    return "";
}

inline std::string format_parameters(const std::string& properties_json, const std::string& /*required_json*/) {
    static const std::set<std::string> standard_keys = {"description", "type", "properties", "required", "nullable"};

    size_t pos = 0;
    auto properties = parse_json_object_raw(properties_json, pos);

    std::string result;
    bool first = true;

    for (const auto& [key, value_json] : properties) {
        if (standard_keys.count(key)) continue;

        if (!first) result += ",";
        first = false;

        size_t prop_pos = 0;
        auto prop_obj = parse_json_object_raw(value_json, prop_pos);

        result += key + ":{";

        if (prop_obj.count("description")) {
            std::string desc = get_json_string_value(prop_obj["description"], 0);
            result += "description:" + escape(desc);
        }

        std::string type_val;
        if (prop_obj.count("type")) {
            type_val = get_json_string_value(prop_obj["type"], 0);
        }

        if (to_upper(type_val) == "STRING") {
            if (prop_obj.count("enum")) {
                size_t enum_pos = 0;
                std::string enum_formatted = format_argument(prop_obj["enum"], enum_pos, true);
                result += ",enum:" + enum_formatted;
            }
        } else if (to_upper(type_val) == "OBJECT") {
            if (prop_obj.count("properties")) {
                std::string nested_required;
                if (prop_obj.count("required")) {
                    nested_required = prop_obj["required"];
                }
                result += ",properties:{" + format_parameters(prop_obj["properties"], nested_required) + "}";
            }
            if (prop_obj.count("required")) {
                result += ",required:[";
                size_t req_pos = 0;
                skip_whitespace(prop_obj["required"], req_pos);
                if (req_pos < prop_obj["required"].length() && prop_obj["required"][req_pos] == '[') {
                    req_pos++;
                    bool req_first = true;
                    while (req_pos < prop_obj["required"].length()) {
                        skip_whitespace(prop_obj["required"], req_pos);
                        if (prop_obj["required"][req_pos] == ']') break;
                        if (prop_obj["required"][req_pos] == ',') { req_pos++; continue; }
                        if (prop_obj["required"][req_pos] == '"') {
                            req_pos++;
                            std::string req_item = extract_json_string(prop_obj["required"], req_pos);
                            if (!req_first) result += ",";
                            req_first = false;
                            result += escape(req_item);
                        }
                    }
                }
                result += "]";
            }
        } else if (to_upper(type_val) == "ARRAY") {
            if (prop_obj.count("items")) {
                result += ",items:{";
                size_t items_pos = 0;
                auto items_obj = parse_json_object_raw(prop_obj["items"], items_pos);
                bool items_first = true;

                for (const auto& [item_key, item_value] : items_obj) {
                    if (!items_first) result += ",";
                    items_first = false;

                    if (item_key == "properties") {
                        std::string items_required;
                        if (items_obj.count("required")) {
                            items_required = items_obj["required"];
                        }
                        result += "properties:{" + format_parameters(item_value, items_required) + "}";
                    } else if (item_key == "required") {
                        result += "required:[";
                        size_t req_pos = 0;
                        skip_whitespace(item_value, req_pos);
                        if (req_pos < item_value.length() && item_value[req_pos] == '[') {
                            req_pos++;
                            bool req_first = true;
                            while (req_pos < item_value.length()) {
                                skip_whitespace(item_value, req_pos);
                                if (item_value[req_pos] == ']') break;
                                if (item_value[req_pos] == ',') { req_pos++; continue; }
                                if (item_value[req_pos] == '"') {
                                    req_pos++;
                                    std::string req_item = extract_json_string(item_value, req_pos);
                                    if (!req_first) result += ",";
                                    req_first = false;
                                    result += escape(req_item);
                                }
                            }
                        }
                        result += "]";
                    } else if (item_key == "type") {
                        std::string item_type = get_json_string_value(item_value, 0);
                        result += "type:" + escape(to_upper(item_type));
                    } else {
                        size_t val_pos = 0;
                        result += item_key + ":" + format_argument(item_value, val_pos, true);
                    }
                }
                result += "}";
            }
        }

        if (!type_val.empty()) {
            result += ",type:" + escape(to_upper(type_val));
        }

        result += "}";
    }

    return result;
}

inline std::string format_function_declaration(const std::string& name,
                                                const std::string& description,
                                                const std::string& params_json) {
    std::string result = "declaration:" + name + "{";
    result += "description:" + escape(description);

    if (!params_json.empty()) {
        result += ",parameters:{";

        size_t pos = 0;
        auto params = parse_json_object_raw(params_json, pos);

        if (params.count("properties")) {
            std::string required_json;
            if (params.count("required")) {
                required_json = params["required"];
            }
            result += "properties:{" + format_parameters(params["properties"], required_json) + "}";
        }

        if (params.count("required")) {
            result += ",required:[";
            size_t req_pos = 0;
            skip_whitespace(params["required"], req_pos);
            if (req_pos < params["required"].length() && params["required"][req_pos] == '[') {
                req_pos++;
                bool first = true;
                while (req_pos < params["required"].length()) {
                    skip_whitespace(params["required"], req_pos);
                    if (params["required"][req_pos] == ']') break;
                    if (params["required"][req_pos] == ',') { req_pos++; continue; }
                    if (params["required"][req_pos] == '"') {
                        req_pos++;
                        std::string item = extract_json_string(params["required"], req_pos);
                        if (!first) result += ",";
                        first = false;
                        result += escape(item);
                    }
                }
            }
            result += "]";
        }

        if (params.count("type")) {
            std::string type_val = get_json_string_value(params["type"], 0);
            result += ",type:" + escape(to_upper(type_val));
        }

        result += "}";
    }

    result += "}";
    return result;
}

template<typename ToolFunction>
inline std::string format_tools(const std::vector<ToolFunction>& tools, bool use_pipe_tags = false) {
    if (tools.empty()) return "";

    const char* decl_start = use_pipe_tags ? "<|tool>" : "<start_function_declaration>";
    const char* decl_end   = use_pipe_tags ? "<tool|>" : "<end_function_declaration>";

    std::string result;
    for (const auto& tool : tools) {
        result += decl_start;
        std::string params_json;
        auto it = tool.parameters.find("schema");
        if (it != tool.parameters.end()) {
            params_json = it->second;
        }

        result += format_function_declaration(tool.name, tool.description, params_json);
        result += decl_end;
    }
    return result;
}


inline size_t match_quote_tag(const std::string& s, size_t pos) {
    if (s.compare(pos, 8, "<escape>") == 0) return 8;
    if (s.compare(pos, 5, "<|\"|>") == 0) return 5;
    return 0;
}

inline size_t find_quote_tag(const std::string& s, size_t pos) {
    size_t e = s.find("<escape>", pos);
    size_t t = s.find("<|\"|>", pos);
    if (e == std::string::npos) return t;
    if (t == std::string::npos) return e;
    return std::min(e, t);
}

inline std::string unescape(const std::string& s) {
    const std::string ESCAPE_TAG = "<escape>";
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(ESCAPE_TAG, pos)) != std::string::npos) {
        result.erase(pos, ESCAPE_TAG.length());
    }
    return result;
}

inline std::string args_to_json(const std::string& args_content) {
    std::string result = "{";
    size_t pos = 0;
    bool first = true;

    if (!args_content.empty() && args_content[0] == '{') pos = 1;

    while (pos < args_content.length()) {
        while (pos < args_content.length() && std::isspace(args_content[pos])) pos++;
        if (pos >= args_content.length() || args_content[pos] == '}') break;
        if (args_content[pos] == ',') { pos++; continue; }

        size_t key_start = pos;
        while (pos < args_content.length() && args_content[pos] != ':') pos++;
        std::string key = args_content.substr(key_start, pos - key_start);
        if (pos < args_content.length()) pos++; 

        std::string value;
        while (pos < args_content.length() && std::isspace(args_content[pos])) pos++;

        if (pos < args_content.length()) {
            size_t qtag_len = match_quote_tag(args_content, pos);
            if (qtag_len > 0) {
                pos += qtag_len;
                size_t val_end = find_quote_tag(args_content, pos);
                if (val_end != std::string::npos) {
                    value = "\"" + args_content.substr(pos, val_end - pos) + "\"";
                    pos = val_end + match_quote_tag(args_content, val_end);
                }
            } else if (args_content[pos] == '{') {
                int depth = 1;
                size_t start = pos;
                pos++;
                while (pos < args_content.length() && depth > 0) {
                    if (args_content[pos] == '{') depth++;
                    else if (args_content[pos] == '}') depth--;
                    pos++;
                }
                value = args_to_json(args_content.substr(start, pos - start));
            } else if (args_content[pos] == '[') {
                int depth = 1;
                size_t start = pos;
                pos++;
                while (pos < args_content.length() && depth > 0) {
                    if (args_content[pos] == '[') depth++;
                    else if (args_content[pos] == ']') depth--;
                    pos++;
                }
                std::string arr_content = args_content.substr(start + 1, pos - start - 2);
                value = "[";
                size_t arr_pos = 0;
                bool first_item = true;
                while (arr_pos < arr_content.length()) {
                    while (arr_pos < arr_content.length() && (std::isspace(arr_content[arr_pos]) || arr_content[arr_pos] == ',')) arr_pos++;
                    if (arr_pos >= arr_content.length()) break;

                    if (!first_item) value += ",";
                    first_item = false;

                    size_t aq_len = match_quote_tag(arr_content, arr_pos);
                    if (aq_len > 0) {
                        arr_pos += aq_len;
                        size_t end = find_quote_tag(arr_content, arr_pos);
                        if (end != std::string::npos) {
                            value += "\"" + arr_content.substr(arr_pos, end - arr_pos) + "\"";
                            arr_pos = end + match_quote_tag(arr_content, end);
                        }
                    } else {
                        size_t end = arr_content.find_first_of(",]", arr_pos);
                        if (end == std::string::npos) end = arr_content.length();
                        value += arr_content.substr(arr_pos, end - arr_pos);
                        arr_pos = end;
                    }
                }
                value += "]";
            } else {
                size_t val_start = pos;
                while (pos < args_content.length() && args_content[pos] != ',' && args_content[pos] != '}') {
                    pos++;
                }
                value = args_content.substr(val_start, pos - val_start);
                while (!value.empty() && std::isspace(value.back())) value.pop_back();
            }
        }

        if (!first) result += ",";
        first = false;
        result += "\"" + key + "\":" + value;
    }

    result += "}";
    return result;
}

inline void parse_function_calls(std::string& response, std::vector<std::string>& function_calls) {

    const std::string CALL_START = (response.find("<|tool_call>") != std::string::npos)
        ? "<|tool_call>" : "<start_function_call>";
    const std::string CALL_END = (CALL_START == "<|tool_call>")
        ? "<tool_call|>" : "<end_function_call>";
    size_t pos = 0;

    while ((pos = response.find(CALL_START, pos)) != std::string::npos) {
        size_t content_start = pos + CALL_START.length();
        size_t call_end_pos = response.find(CALL_END, content_start);

        size_t content_end = (call_end_pos != std::string::npos) ? call_end_pos : response.length();
        std::string call_content = response.substr(content_start, content_end - content_start);

        if (call_content.compare(0, 5, "call:") == 0) {
            size_t brace_pos = call_content.find('{');

            if (brace_pos == std::string::npos) {
                size_t sep_pos = call_content.find_first_of(", ", 5);
                if (sep_pos != std::string::npos) {
                    std::string func_name = call_content.substr(5, sep_pos - 5);
                    size_t args_start = sep_pos + 1;
                    while (args_start < call_content.length() &&
                           (call_content[args_start] == ' ' || call_content[args_start] == ',')) {
                        args_start++;
                    }
                    std::string args_content = "{" + call_content.substr(args_start);
                    if (args_content.back() != '}') args_content += "}";

                    std::string args_json = args_to_json(args_content);
                    std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":" + args_json + "}";
                    function_calls.push_back(json_call);
                }
            } else {
                std::string func_name = call_content.substr(5, brace_pos - 5);
                std::string args_content = call_content.substr(brace_pos);
                if (args_content.back() != '}') args_content += "}";

                std::string args_json = args_to_json(args_content);
                std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":" + args_json + "}";
                function_calls.push_back(json_call);
            }
        }

        size_t erase_end = (call_end_pos != std::string::npos) ?
                           call_end_pos + CALL_END.length() : response.length();
        response.erase(pos, erase_end - pos);
    }
}

} // namespace gemma