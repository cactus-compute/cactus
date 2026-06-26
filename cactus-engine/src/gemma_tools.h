#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <set>

#include "json_escape.h"
#include "picojson.h"

namespace gemma {

inline std::string to_upper(const std::string& s) {
    std::string result = s;
    for (auto& c : result) c = std::toupper(c);
    return result;
}

inline std::string escape(const std::string& s) {
    return "<|\"|>" + s + "<|\"|>";
}

inline std::string use_escape_tags(std::string s) {
    size_t pos = 0;
    while ((pos = s.find("<|\"|>", pos)) != std::string::npos) {
        s.replace(pos, 5, "<escape>");
        pos += 8;
    }
    return s;
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
            else if (json[pos] == 'u' && pos + 4 < json.length()) {
                // Decode \uXXXX (BMP) -> UTF-8. Without this, JSON-escaped
                // chars like < leak literally as "u003c" in tool args.
                unsigned int cp = 0;
                bool ok = true;
                for (int k = 1; k <= 4; k++) {
                    char h = json[pos + k];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                    else { ok = false; break; }
                }
                if (ok) {
                    pos += 4;
                    if (cp < 0x80) {
                        value += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        value += static_cast<char>(0xC0 | (cp >> 6));
                        value += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        value += static_cast<char>(0xE0 | (cp >> 12));
                        value += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        value += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                } else {
                    value += json[pos];
                }
            }
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
        pos++;
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
        if (pos == start && pos < json.length()) pos++;
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
                std::string req_items;
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
                            if (!req_first) req_items += ",";
                            req_first = false;
                            req_items += escape(req_item);
                        }
                    }
                }
                if (!req_items.empty()) {
                    result += ",required:[" + req_items + "]";
                }
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
            std::string req_items;
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
                        if (!first) req_items += ",";
                        first = false;
                        req_items += escape(item);
                    }
                }
            }
            if (!req_items.empty()) {
                result += ",required:[" + req_items + "]";
            }
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

        std::string declaration = format_function_declaration(tool.name, tool.description, params_json);
        result += use_pipe_tags ? declaration : use_escape_tags(declaration);
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

inline std::string to_json_value(const std::string& v) {
    if (v == "true" || v == "false" || v == "null") return v;
    if (!v.empty()) {
        char* end = nullptr;
        std::strtod(v.c_str(), &end);
        if (end == v.c_str() + v.size()) return v;
    }
    return "\"" + escape_json_string(v) + "\"";
}

inline void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) { out += static_cast<char>(cp); }
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
}

inline std::string read_quoted_json_string(const std::string& s, size_t& pos, char quote = '"') {
    std::string out;
    if (pos < s.size() && s[pos] == quote) pos++;
    auto hex4 = [&](size_t p, unsigned& v) -> bool {
        if (p + 4 > s.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            char h = s[p + k]; v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
            else return false;
        }
        return true;
    };
    while (pos < s.size()) {
        char c = s[pos];
        if (c == quote) { pos++; break; }
        if (c == '\\' && pos + 1 < s.size()) {
            char n = s[pos + 1];
            switch (n) {
                case '\'': out += '\''; pos += 2; break;
                case 'n': out += '\n'; pos += 2; break;
                case 't': out += '\t'; pos += 2; break;
                case 'r': out += '\r'; pos += 2; break;
                case 'b': out += '\b'; pos += 2; break;
                case 'f': out += '\f'; pos += 2; break;
                case '"': out += '"'; pos += 2; break;
                case '\\': out += '\\'; pos += 2; break;
                case '/': out += '/'; pos += 2; break;
                case 'u': {
                    unsigned cp;
                    if (hex4(pos + 2, cp)) {
                        size_t adv = pos + 6;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            unsigned lo;
                            if (adv + 6 <= s.size() && s[adv] == '\\' && s[adv + 1] == 'u' &&
                                hex4(adv + 2, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); adv += 6;
                            } else { cp = 0xFFFD; }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) { cp = 0xFFFD; }
                        append_utf8(out, cp); pos = adv;
                    } else { out += '\\'; out += 'u'; pos += 2; }
                    break;
                }
                default: out += n; pos += 2; break;
            }
        } else {
            out += c; pos++;
        }
    }
    return out;
}

inline std::string repair_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    bool in_str = false, esc = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (esc) { out += c; esc = false; continue; }
            if (c == '\\') { out += c; esc = true; continue; }
            if (c == '"') { out += c; in_str = false; continue; }
            unsigned char uc = static_cast<unsigned char>(c);
            if (uc == '\n') out += "\\n";
            else if (uc == '\t') out += "\\t";
            else if (uc == '\r') out += "\\r";
            else if (uc < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", uc); out += b; }
            else out += c;
        } else {
            if (c == '"') { in_str = true; out += c; }
            else if (c == ',') {
                size_t j = i + 1;
                while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) j++;
                bool trailing = (j < s.size() && (s[j] == '}' || s[j] == ']'));
                if (!trailing) out += c;
            } else out += c;
        }
    }
    return out;
}

inline size_t find_raw_value_end(const std::string& s, size_t pos,
                                 const std::vector<std::string>& known_keys) {
    int brace = 0, brack = 0;
    for (size_t i = pos; i < s.size(); ++i) {
        char c = s[i];
        if (c == '{') brace++;
        else if (c == '}') { if (brace == 0) return i; brace--; }
        else if (c == '[') brack++;
        else if (c == ']') { if (brack > 0) brack--; }
        else if (c == ',' && brace == 0 && brack == 0) {
            size_t j = i + 1;
            while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) j++;
            bool quoted = (j < s.size() && s[j] == '"');
            size_t ks = quoted ? j + 1 : j;
            for (const auto& k : known_keys) {
                if (k.empty() || s.compare(ks, k.size(), k) != 0) continue;
                size_t m = ks + k.size();
                if (quoted) { if (m < s.size() && s[m] == '"') m++; else continue; }
                while (m < s.size() && std::isspace(static_cast<unsigned char>(s[m]))) m++;
                if (m < s.size() && s[m] == ':') return i;
            }
        }
    }
    return s.size();
}

inline std::string args_to_json(const std::string& args_content,
                                const std::vector<std::string>& known_keys = {},
                                const std::set<std::string>& string_keys = {});
inline std::string parse_array_items(const std::string& inner);

inline std::string try_json_object(const std::string& s) {
    auto attempt = [](const std::string& c) -> std::string {
        picojson::value v;
        const char* b = c.data();
        const char* e = b + c.size();
        std::string err;
        const char* end = picojson::parse(v, b, e, &err);
        while (end < e && std::isspace(static_cast<unsigned char>(*end))) ++end;
        return (err.empty() && end == e && v.is<picojson::object>()) ? v.serialize() : std::string();
    };
    std::string r = attempt(s);
    if (!r.empty()) return r;
    std::string repaired = repair_json(s);
    return (repaired != s) ? attempt(repaired) : std::string();
}

inline bool skip_delimited_string(const std::string& s, size_t& i) {
    if (size_t qtag = match_quote_tag(s, i)) {
        size_t end = find_quote_tag(s, i + qtag);
        i = (end == std::string::npos) ? s.size() : end + match_quote_tag(s, end);
        return true;
    }
    char q = (i < s.size()) ? s[i] : '\0';
    if (q == '"' || q == '\'') {
        for (i++; i < s.size(); ++i) {
            if (s[i] == '\\') { i++; continue; }
            if (s[i] == q) { i++; break; }
        }
        return true;
    }
    return false;
}

inline std::string read_balanced(const std::string& s, size_t& pos, char open, char close) {
    size_t start = pos;
    int depth = 0;
    while (pos < s.size()) {
        if (skip_delimited_string(s, pos)) continue;
        char c = s[pos++];
        if (c == open) depth++;
        else if (c == close && --depth == 0) break;
    }
    return s.substr(start, pos - start);
}

inline bool read_delimited_string(const std::string& s, size_t& pos, std::string& out) {
    if (size_t qtag = match_quote_tag(s, pos)) {
        pos += qtag;
        size_t end = find_quote_tag(s, pos);
        size_t stop = (end == std::string::npos) ? s.size() : end;
        out = "\"" + escape_json_string(s.substr(pos, stop - pos)) + "\"";
        pos = (end == std::string::npos) ? s.size() : end + match_quote_tag(s, end);
        return true;
    }
    if (pos < s.size() && (s[pos] == '"' || s[pos] == '\'')) {
        out = "\"" + escape_json_string(read_quoted_json_string(s, pos, s[pos])) + "\"";
        return true;
    }
    return false;
}

inline std::string read_raw_value(const std::string& s, size_t& pos,
                                  const std::vector<std::string>& known_keys, bool want_string) {
    size_t start = pos;
    size_t end = known_keys.empty() ? s.find_first_of(",}", pos)
                                    : find_raw_value_end(s, pos, known_keys);
    if (end == std::string::npos) end = s.size();
    std::string raw = s.substr(start, end - start);
    pos = end;

    std::string trimmed = raw;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.pop_back();
    bool scalar = !want_string && (trimmed == "true" || trimmed == "false" || trimmed == "null");
    if (!want_string && !scalar && !trimmed.empty()) {
        char* ep = nullptr;
        std::strtod(trimmed.c_str(), &ep);
        scalar = (ep == trimmed.c_str() + trimmed.size());
    }
    return scalar ? trimmed : ("\"" + escape_json_string(raw) + "\"");
}

inline std::string parse_array_items(const std::string& inner) {
    std::string out = "[";
    size_t pos = 0;
    bool first = true;
    while (pos < inner.size()) {
        while (pos < inner.size() && (std::isspace(static_cast<unsigned char>(inner[pos])) || inner[pos] == ',')) pos++;
        if (pos >= inner.size()) break;
        if (!first) out += ",";
        first = false;

        std::string item;
        if (read_delimited_string(inner, pos, item)) {
            out += item;
        } else if (inner[pos] == '{') {
            out += args_to_json(read_balanced(inner, pos, '{', '}'));
        } else if (inner[pos] == '[') {
            std::string nested = read_balanced(inner, pos, '[', ']');
            out += parse_array_items(nested.substr(1, nested.size() - 2));
        } else {
            out += read_raw_value(inner, pos, {}, false);
        }
    }
    out += "]";
    return out;
}

inline std::string args_to_json(const std::string& args_content,
                                const std::vector<std::string>& known_keys,
                                const std::set<std::string>& string_keys) {
    if (std::string json = try_json_object(args_content); !json.empty()) return json;

    std::string result = "{";
    size_t pos = (!args_content.empty() && args_content[0] == '{') ? 1 : 0;
    bool first = true;

    while (pos < args_content.size()) {
        while (pos < args_content.size() && std::isspace(static_cast<unsigned char>(args_content[pos]))) pos++;
        if (pos >= args_content.size() || args_content[pos] == '}') break;
        if (args_content[pos] == ',') { pos++; continue; }

        std::string key;
        if (args_content[pos] == '"') {
            key = read_quoted_json_string(args_content, pos);
        } else {
            size_t ks = pos;
            while (pos < args_content.size() && args_content[pos] != ':') pos++;
            key = args_content.substr(ks, pos - ks);
            while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) key.pop_back();
        }
        while (pos < args_content.size() && args_content[pos] != ':') pos++;
        if (pos < args_content.size()) pos++;
        while (pos < args_content.size() && std::isspace(static_cast<unsigned char>(args_content[pos]))) pos++;

        const bool want_string = string_keys.count(key) > 0;
        std::string value;
        if (pos < args_content.size()) {
            if (read_delimited_string(args_content, pos, value)) {
            } else if (!want_string && args_content[pos] == '{') {
                value = args_to_json(read_balanced(args_content, pos, '{', '}'));
            } else if (!want_string && args_content[pos] == '[') {
                std::string arr = read_balanced(args_content, pos, '[', ']');
                value = parse_array_items(arr.substr(1, arr.size() - 2));
            } else {
                value = read_raw_value(args_content, pos, known_keys, want_string);
            }
        }

        if (!first) result += ",";
        first = false;
        result += "\"" + escape_json_string(key) + "\":" + (value.empty() ? "\"\"" : value);
    }

    result += "}";
    return result;
}

inline void parse_function_calls_core(std::string& response, std::vector<std::string>& function_calls,
                                      const std::map<std::string, std::vector<std::string>>& fn_keys,
                                      const std::map<std::string, std::set<std::string>>& fn_strkeys = {}) {
    const std::string CALL_START = (response.find("<|tool_call>") != std::string::npos)
        ? "<|tool_call>" : "<start_function_call>";
    const std::string CALL_END = (CALL_START == "<|tool_call>")
        ? "<tool_call|>" : "<end_function_call>";
    size_t pos = 0;

    auto emit_call = [&](const std::string& fn, std::string args_content) {
        if (args_content.empty() || args_content.back() != '}') args_content += "}";
        std::vector<std::string> keys;
        std::set<std::string> strkeys;
        if (auto it = fn_keys.find(fn); it != fn_keys.end()) keys = it->second;
        if (auto it = fn_strkeys.find(fn); it != fn_strkeys.end()) strkeys = it->second;
        std::string args_json = args_to_json(args_content, keys, strkeys);
        function_calls.push_back("{\"name\":\"" + escape_json_string(fn) + "\",\"arguments\":" + args_json + "}");
    };

    while ((pos = response.find(CALL_START, pos)) != std::string::npos) {
        size_t content_start = pos + CALL_START.length();
        while (content_start < response.length() &&
               std::isspace(static_cast<unsigned char>(response[content_start]))) {
            content_start++;
        }

        const bool is_call = response.compare(content_start, 5, "call:") == 0;
        size_t scan_from = content_start;
        if (is_call) {
            size_t brace_pos = response.find('{', content_start + 5);
            if (brace_pos != std::string::npos) {
                size_t bp = brace_pos;
                read_balanced(response, bp, '{', '}');
                scan_from = bp;
            }
        }
        size_t call_end_pos = response.find(CALL_END, scan_from);

        size_t content_end = (call_end_pos != std::string::npos) ? call_end_pos : response.length();
        std::string call_content = response.substr(content_start, content_end - content_start);

        if (is_call) {
            size_t brace_pos = call_content.find('{');
            if (brace_pos != std::string::npos) {
                emit_call(call_content.substr(5, brace_pos - 5), call_content.substr(brace_pos));
            } else {
                size_t sep_pos = call_content.find_first_of(", ", 5);
                if (sep_pos != std::string::npos) {
                    size_t args_start = call_content.find_first_not_of(", ", sep_pos);
                    emit_call(call_content.substr(5, sep_pos - 5),
                              "{" + (args_start == std::string::npos ? std::string() : call_content.substr(args_start)));
                }
            }
        }

        size_t erase_end = (call_end_pos != std::string::npos) ?
                           call_end_pos + CALL_END.length() : response.length();
        response.erase(pos, erase_end - pos);
    }
}

inline void parse_function_calls(std::string& response, std::vector<std::string>& function_calls) {
    parse_function_calls_core(response, function_calls, {});
}

template<typename ToolFunction>
inline void parse_function_calls(std::string& response, std::vector<std::string>& function_calls,
                                 const std::vector<ToolFunction>& tools) {
    std::map<std::string, std::vector<std::string>> fn_keys;
    std::map<std::string, std::set<std::string>> fn_strkeys;
    for (const auto& t : tools) {
        auto it = t.parameters.find("schema");
        if (it == t.parameters.end()) continue;
        picojson::value v;
        if (!picojson::parse(v, it->second).empty() || !v.is<picojson::object>()) continue;
        const auto& obj = v.get<picojson::object>();
        auto pit = obj.find("properties");
        if (pit == obj.end() || !pit->second.is<picojson::object>()) continue;
        std::vector<std::string> names;
        std::set<std::string> strkeys;
        for (const auto& kv : pit->second.get<picojson::object>()) {
            names.push_back(kv.first);
            if (kv.second.is<picojson::object>()) {
                const auto& prop = kv.second.get<picojson::object>();
                auto tit = prop.find("type");
                if (tit != prop.end() && tit->second.is<std::string>() &&
                    tit->second.get<std::string>() == "string") {
                    strkeys.insert(kv.first);
                }
            }
        }
        fn_keys[t.name] = std::move(names);
        fn_strkeys[t.name] = std::move(strkeys);
    }
    parse_function_calls_core(response, function_calls, fn_keys, fn_strkeys);
}

} // namespace gemma