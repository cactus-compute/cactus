#ifndef CACTUS_FFI_UTILS_H
#define CACTUS_FFI_UTILS_H

#include "../engine/engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace cactus {
namespace ffi {

struct ToolFunction {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> parameters;
};

inline void handle_error_response(const std::string& error_message, char* response_buffer, size_t buffer_size) {
    std::string sanitized_msg = error_message;
    for (auto& c : sanitized_msg) {
        if (c == '"') c = '\'';
        if (c == '\n') c = ' ';
    }
    std::string error_json = "{\"success\":false,\"error\":\"" + sanitized_msg + "\"}";
    if (response_buffer && error_json.length() < buffer_size) {
        std::strcpy(response_buffer, error_json.c_str());
    }
}

inline std::vector<cactus::engine::ChatMessage> parse_messages_json(const std::string& json) {
    std::vector<cactus::engine::ChatMessage> messages;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid JSON: expected array");
    }
    
    pos = json.find('{', pos);
    while (pos != std::string::npos) {
        cactus::engine::ChatMessage msg;
        
        size_t role_pos = json.find("\"role\"", pos);
        if (role_pos == std::string::npos) break;
        
        size_t role_start = json.find('"', role_pos + 6) + 1;
        size_t role_end = json.find('"', role_start);
        msg.role = json.substr(role_start, role_end - role_start);
        
        size_t content_pos = json.find("\"content\"", role_end);
        if (content_pos == std::string::npos) break;
        
        size_t content_start = json.find('"', content_pos + 9) + 1;
        size_t content_end = content_start;
        
        while (content_end < json.length()) {
            content_end = json.find('"', content_end);
            if (content_end == std::string::npos) break;
            if (json[content_end - 1] != '\\') break;
            content_end++;
        }
        
        msg.content = json.substr(content_start, content_end - content_start);
        
        size_t escape_pos = 0;
        while ((escape_pos = msg.content.find("\\n", escape_pos)) != std::string::npos) {
            msg.content.replace(escape_pos, 2, "\n");
            escape_pos += 1;
        }
        escape_pos = 0;
        while ((escape_pos = msg.content.find("\\\"", escape_pos)) != std::string::npos) {
            msg.content.replace(escape_pos, 2, "\"");
            escape_pos += 1;
        }
        
        messages.push_back(msg);
        
        pos = json.find('{', content_end);
    }
    
    return messages;
}

inline std::vector<ToolFunction> parse_tools_json(const std::string& json) {
    std::vector<ToolFunction> tools;
    
    if (json.empty()) return tools;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) return tools;
    
    pos = json.find("\"function\"", pos);
    while (pos != std::string::npos) {
        ToolFunction tool;
        
        size_t name_pos = json.find("\"name\"", pos);
        if (name_pos != std::string::npos) {
            size_t name_start = json.find('"', name_pos + 6) + 1;
            size_t name_end = json.find('"', name_start);
            tool.name = json.substr(name_start, name_end - name_start);
        }
        
        size_t desc_pos = json.find("\"description\"", pos);
        if (desc_pos != std::string::npos) {
            size_t desc_start = json.find('"', desc_pos + 13) + 1;
            size_t desc_end = json.find('"', desc_start);
            tool.description = json.substr(desc_start, desc_end - desc_start);
        }
        
        size_t params_pos = json.find("\"parameters\"", pos);
        if (params_pos != std::string::npos) {
            size_t params_start = json.find('{', params_pos);
            if (params_start != std::string::npos) {
                int brace_count = 1;
                size_t params_end = params_start + 1;
                while (params_end < json.length() && brace_count > 0) {
                    if (json[params_end] == '{') brace_count++;
                    else if (json[params_end] == '}') brace_count--;
                    params_end++;
                }
                tool.parameters["schema"] = json.substr(params_start, params_end - params_start);
            }
        }
        
        tools.push_back(tool);
        
        pos = json.find("\"function\"", name_pos);
    }
    
    return tools;
}

inline void parse_options_json(const std::string& json, 
                               float& temperature, float& top_p, 
                               size_t& top_k, size_t& max_tokens,
                               std::vector<std::string>& stop_sequences) {
    temperature = -1.0f; // Use model default
    top_p = -1.0f;       // Use model default
    top_k = 0;           // Use model default
    max_tokens = 100;    // FFI-level default
    stop_sequences.clear();
    
    if (json.empty()) return;
    
    size_t pos = json.find("\"temperature\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        temperature = std::stof(json.substr(pos));
    }
    
    pos = json.find("\"top_p\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        top_p = std::stof(json.substr(pos));
    }
    
    pos = json.find("\"top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        top_k = std::stoul(json.substr(pos));
    }
    
    pos = json.find("\"max_tokens\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        max_tokens = std::stoul(json.substr(pos));
    }
    
    pos = json.find("\"stop_sequences\"");
    if (pos != std::string::npos) {
        pos = json.find('[', pos);
        if (pos != std::string::npos) {
            size_t end_pos = json.find(']', pos);
            size_t seq_pos = json.find('"', pos);
            
            while (seq_pos != std::string::npos && seq_pos < end_pos) {
                size_t seq_start = seq_pos + 1;
                size_t seq_end = json.find('"', seq_start);
                if (seq_end != std::string::npos) {
                    stop_sequences.push_back(json.substr(seq_start, seq_end - seq_start));
                }
                seq_pos = json.find('"', seq_end + 1);
            }
        }
    }
}

inline std::string format_tools_for_prompt(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "";
    std::string formatted_tools_json;
    for (size_t i = 0; i < tools.size(); i++) {
        if (i > 0) formatted_tools_json += ",\n";
        formatted_tools_json += "  {\n";
        formatted_tools_json += "    \"type\": \"function\",\n";
        formatted_tools_json += "    \"function\": {\n";
        formatted_tools_json += "      \"name\": \"" + tools[i].name + "\",\n";
        formatted_tools_json += "      \"description\": \"" + tools[i].description + "\"";
        if (tools[i].parameters.find("schema") != tools[i].parameters.end()) {
            formatted_tools_json += ",\n      \"parameters\": " + tools[i].parameters.at("schema");
        }
        formatted_tools_json += "\n    }\n  }";
    }
    return formatted_tools_json;
}

inline void parse_function_calls_from_response(const std::string& response_text, 
                                               std::string& regular_response, 
                                               std::vector<std::string>& function_calls) {
    regular_response = response_text;
    function_calls.clear();

    const char* FUNCTION_CALL_MARKER = "\"function_call\"";
    size_t search_pos = 0;
    const size_t text_len = response_text.length();

    while (search_pos < text_len) {
        size_t marker_pos = response_text.find(FUNCTION_CALL_MARKER, search_pos);
        if (marker_pos == std::string::npos) break;

        size_t json_start = response_text.find('{', marker_pos);
        if (json_start == std::string::npos) break;

        int brace_count = 1;
        size_t json_end = json_start + 1;
        while (json_end < text_len && brace_count > 0) {
            char c = response_text[json_end];
            brace_count += (c == '{') - (c == '}');
            json_end++;
        }

        if (brace_count == 0) {
            function_calls.push_back(response_text.substr(json_start, json_end - json_start));
            regular_response = response_text.substr(0, marker_pos); 
            size_t last_bracket = regular_response.rfind('{');
            if(last_bracket != std::string::npos) {
                regular_response = regular_response.substr(0, last_bracket);
            }
        }
        search_pos = json_end;
    }
}

inline std::string construct_response_json(const std::string& regular_response,
                                           const std::vector<std::string>& function_calls,
                                           double time_to_first_token,
                                           double total_time_ms,
                                           double tokens_per_second,
                                           size_t prompt_tokens,
                                           size_t completion_tokens) {
    std::ostringstream json_response;
    json_response << "{";
    json_response << "\"success\":true,";
    json_response << "\"response\":\"";
    for (char c : regular_response) {
        if (c == '"') json_response << "\\\"";
        else if (c == '\n') json_response << "\\n";
        else if (c == '\r') json_response << "\\r";
        else if (c == '\t') json_response << "\\t";
        else if (c == '\\') json_response << "\\\\";
        else json_response << c;
    }
    json_response << "\",";
    if (!function_calls.empty()) {
        json_response << "\"function_calls\":[";
        for (size_t i = 0; i < function_calls.size(); ++i) {
            if (i > 0) json_response << ",";
            json_response << function_calls[i];
        }
        json_response << "],";
    }
    json_response << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json_response << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
    json_response << "\"tokens_per_second\":" << std::fixed << std::setprecision(2) << tokens_per_second << ",";
    json_response << "\"prefill_tokens\":" << prompt_tokens << ",";
    json_response << "\"decode_tokens\":" << completion_tokens << ",";
    json_response << "\"total_tokens\":" << (prompt_tokens + completion_tokens);
    json_response << "}";
    return json_response.str();
}

} // namespace ffi
} // namespace cactus

#endif // CACTUS_FFI_UTILS_H
