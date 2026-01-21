#ifndef CACTUS_UTILS_H
#define CACTUS_UTILS_H

#include "../engine/engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cctype>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>

#ifdef __APPLE__
#include <uuid/uuid.h>
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

inline size_t get_memory_footprint_bytes() {
#ifdef __APPLE__
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count) == KERN_SUCCESS)
        return vm_info.phys_footprint;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return pmc.PrivateUsage;
#elif defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size, resident;
        statm >> size >> resident;
        return resident * sysconf(_SC_PAGESIZE);
    }
#endif
    return 0;
}

inline double get_ram_usage_mb() {
    return get_memory_footprint_bytes() / (1024.0 * 1024.0);
}

struct CactusModelHandle {
    std::unique_ptr<cactus::engine::Model> model;
    std::atomic<bool> should_stop;
    std::vector<uint32_t> processed_tokens;
    std::mutex model_mutex;
    std::string model_name;
    std::unique_ptr<cactus::engine::index::Index> corpus_index;
    std::string corpus_dir;
    size_t corpus_embedding_dim = 0;
    std::vector<std::vector<float>> tool_embeddings;
    std::vector<std::string> tool_texts;  

    CactusModelHandle() : should_stop(false) {}
};

extern std::string last_error_message;

bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                           const std::vector<std::vector<uint32_t>>& stop_sequences);

std::string retrieve_rag_context(CactusModelHandle* handle, const std::string& query);

namespace cactus {
namespace ffi {

#ifndef CACTUS_VERSION
#define CACTUS_VERSION "unknown"
#endif

inline const char* getVersion() {
    return CACTUS_VERSION;
}

inline std::string generateUUID() {
#ifdef __APPLE__
    uuid_t uuid;
    uuid_generate_random(uuid);
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    return std::string(uuid_str);
#endif
}

struct ToolFunction {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> parameters;
};

} // namespace ffi
} // namespace cactus

std::vector<cactus::ffi::ToolFunction> select_relevant_tools(
    CactusModelHandle* handle,
    const std::string& query,
    const std::vector<cactus::ffi::ToolFunction>& all_tools,
    size_t top_k);

#include "gemma_tools.h"

namespace cactus {
namespace ffi {

inline std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c == '\\') o << "\\\\";
        else o << c;
    }
    return o.str();
}

inline void handle_error_response(const std::string& error_message, char* response_buffer, size_t buffer_size) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":false,";
    json << "\"error\":\"" << escape_json_string(error_message) << "\",";
    json << "\"cloud_handoff\":false,";
    json << "\"response\":null,";
    json << "\"function_calls\":[],";
    json << "\"confidence\":0.0,";
    json << "\"time_to_first_token_ms\":0.0,";
    json << "\"total_time_ms\":0.0,";
    json << "\"prefill_tps\":0.0,";
    json << "\"decode_tps\":0.0,";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb() << ",";
    json << "\"prefill_tokens\":0,";
    json << "\"decode_tokens\":0,";
    json << "\"total_tokens\":0";
    json << "}";
    std::string error_json = json.str();
    if (response_buffer && error_json.length() < buffer_size) {
        std::strcpy(response_buffer, error_json.c_str());
    }
}

inline std::vector<cactus::engine::ChatMessage> parse_messages_json(const std::string& json, 
                                                                   std::vector<std::string>& out_image_paths) {
    std::vector<cactus::engine::ChatMessage> messages;
    out_image_paths.clear();
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid JSON: expected array");
    }
    
    pos = json.find('{', pos);
    while (pos != std::string::npos) {
        cactus::engine::ChatMessage msg;
        
        size_t obj_start = pos;
        int brace_count = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < json.length() && brace_count > 0) {
            if (json[obj_end] == '{') brace_count++;
            else if (json[obj_end] == '}') brace_count--;
            obj_end++;
        }

        size_t role_pos = json.find("\"role\"", pos);
        if (role_pos == std::string::npos || role_pos >= obj_end) break;
        
        size_t role_start = json.find('"', role_pos + 6) + 1;
        size_t role_end = json.find('"', role_start);
        msg.role = json.substr(role_start, role_end - role_start);
        
        size_t content_pos = json.find("\"content\"", role_end);
        if (content_pos != std::string::npos && content_pos < obj_end) {
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
        }
        
        size_t images_pos = json.find("\"images\"", pos);
        if (images_pos != std::string::npos && images_pos < obj_end) {
            size_t array_start = json.find('[', images_pos);
            if (array_start != std::string::npos && array_start < obj_end) {
                size_t array_end = json.find(']', array_start);
                if (array_end != std::string::npos && array_end < obj_end) {
                    size_t img_pos = array_start;
                    while (true) {
                        img_pos = json.find('"', img_pos + 1);
                        if (img_pos == std::string::npos || img_pos >= array_end) break;
                        
                        size_t img_start = img_pos + 1;
                        size_t img_end = json.find('"', img_start);
                        if (img_end == std::string::npos || img_end > array_end) break;
                        
                        std::string img_path = json.substr(img_start, img_end - img_start);
                        
                        std::filesystem::path p(img_path);
                        img_path = std::filesystem::absolute(p).string();
                        
                        msg.images.push_back(img_path);
                        out_image_paths.push_back(img_path);
                        img_pos = img_end;
                    }
                }
            }
        }
        
        messages.push_back(msg);
        
        pos = json.find('{', obj_end);
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
                               std::vector<std::string>& stop_sequences,
                               bool& force_tools,
                               size_t& tool_rag_top_k,
                               float& confidence_threshold,
                               bool& include_stop_sequences) {
    temperature = 0.0f;
    top_p = 0.0f;
    top_k = 0;
    max_tokens = 100;
    force_tools = false;
    tool_rag_top_k = 2;  // 0 = disabled, N = select top N relevant tools
    confidence_threshold = 0.7f;  // trigger cloud handoff when confidence < this value
    include_stop_sequences = false;
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

    pos = json.find("\"force_tools\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        force_tools = (json.substr(pos, 4) == "true");
    }

    pos = json.find("\"tool_rag_top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        tool_rag_top_k = std::stoul(json.substr(pos));
    }

    pos = json.find("\"confidence_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        confidence_threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"include_stop_sequences\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        include_stop_sequences = (json.substr(pos, 4) == "true");
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
        if (i > 0) formatted_tools_json += "\n";
        formatted_tools_json += "{\"type\":\"function\",\"function\":{\"name\":\""
                              + tools[i].name
                              + "\",\"description\":\""
                              + tools[i].description + "\"";
        if (tools[i].parameters.find("schema") != tools[i].parameters.end()) {
            formatted_tools_json += ",\"parameters\":" + tools[i].parameters.at("schema");
        }
        formatted_tools_json += "}}";
    }
    return formatted_tools_json;
}

inline void parse_function_calls_from_response(const std::string& response_text,
                                               std::string& regular_response,
                                               std::vector<std::string>& function_calls) {
    regular_response = response_text;
    function_calls.clear();

    gemma::parse_function_calls(regular_response, function_calls);

    // Parse Qwen-style function calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    const std::string QWEN_TOOL_START = "<tool_call>";
    const std::string QWEN_TOOL_END = "</tool_call>";
    size_t qwen_start_pos = 0;

    while ((qwen_start_pos = regular_response.find(QWEN_TOOL_START, qwen_start_pos)) != std::string::npos) {
        size_t content_start = qwen_start_pos + QWEN_TOOL_START.length();
        size_t qwen_end_pos = regular_response.find(QWEN_TOOL_END, content_start);

        if (qwen_end_pos != std::string::npos) {
            std::string json_content = regular_response.substr(content_start, qwen_end_pos - content_start);

            size_t first = json_content.find_first_not_of(" \t\n\r");
            size_t last = json_content.find_last_not_of(" \t\n\r");
            if (first != std::string::npos && last != std::string::npos) {
                json_content = json_content.substr(first, last - first + 1);
            }

            if (json_content.size() > 2 && json_content[0] == '{' &&
                json_content.find("\"name\"") != std::string::npos) {
                function_calls.push_back(json_content);
            }

            regular_response.erase(qwen_start_pos, qwen_end_pos + QWEN_TOOL_END.length() - qwen_start_pos);
        } else {
            break;
        }
    }

    // Parse LFM2-style function calls: <|tool_call_start|>[name(args)]<|tool_call_end|>
    const std::string TOOL_CALL_START = "<|tool_call_start|>";
    const std::string TOOL_CALL_END = "<|tool_call_end|>";
    size_t tool_start_pos = 0;

    while ((tool_start_pos = regular_response.find(TOOL_CALL_START, tool_start_pos)) != std::string::npos) {
        size_t content_start = tool_start_pos + TOOL_CALL_START.length();
        size_t tool_end_pos = response_text.find(TOOL_CALL_END, content_start);

        if (tool_end_pos != std::string::npos) {
            std::string tool_content = response_text.substr(content_start, tool_end_pos - content_start);

            if (tool_content.size() > 2 && tool_content[0] == '[' && tool_content[tool_content.size()-1] == ']') {
                tool_content = tool_content.substr(1, tool_content.size() - 2); 

                size_t paren_pos = tool_content.find('(');
                if (paren_pos != std::string::npos) {
                    std::string func_name = tool_content.substr(0, paren_pos);
                    std::string args_str = tool_content.substr(paren_pos + 1);

                    if (!args_str.empty() && args_str.back() == ')') {
                        args_str.pop_back();
                    }

                    std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":{";

                    size_t arg_pos = 0;
                    bool first_arg = true;
                    while (arg_pos < args_str.length()) {
                        while (arg_pos < args_str.length() && std::isspace(args_str[arg_pos])) arg_pos++;

                        size_t eq_pos = args_str.find('=', arg_pos);
                        if (eq_pos == std::string::npos) break;

                        std::string arg_name = args_str.substr(arg_pos, eq_pos - arg_pos);

                        size_t val_start = eq_pos + 1;
                        size_t val_end = val_start;

                        if (val_start < args_str.length() && args_str[val_start] == '"') {
                            val_start++;
                            val_end = args_str.find('"', val_start);
                            if (val_end == std::string::npos) break;
                        } else {
                            val_end = args_str.find(',', val_start);
                            if (val_end == std::string::npos) val_end = args_str.length();
                        }

                        std::string arg_value = args_str.substr(val_start, val_end - val_start);

                        if (!first_arg) json_call += ",";
                        json_call += "\"" + arg_name + "\":\"" + arg_value + "\"";
                        first_arg = false;

                        arg_pos = args_str.find(',', val_end);
                        if (arg_pos != std::string::npos) {
                            arg_pos++;
                        } else {
                            break;
                        }
                    }

                    json_call += "}}";
                    function_calls.push_back(json_call);
                }
            }

            regular_response.erase(tool_start_pos, tool_end_pos + TOOL_CALL_END.length() - tool_start_pos);
            tool_start_pos = tool_end_pos + TOOL_CALL_END.length();
        } else {
            break;
        }
    }

    const char* FUNCTION_CALL_MARKER = "\"function_call\"";
    size_t search_pos = 0;
    const size_t text_len = regular_response.length();

    while (search_pos < text_len) {
        size_t marker_pos = regular_response.find(FUNCTION_CALL_MARKER, search_pos);
        if (marker_pos == std::string::npos) break;

        size_t json_start = regular_response.find('{', marker_pos);
        if (json_start == std::string::npos) break;

        int brace_count = 1;
        size_t json_end = json_start + 1;
        while (json_end < text_len && brace_count > 0) {
            char c = regular_response[json_end];
            brace_count += (c == '{') - (c == '}');
            json_end++;
        }

        if (brace_count == 0) {
            function_calls.push_back(regular_response.substr(json_start, json_end - json_start));
            regular_response = regular_response.substr(0, marker_pos);
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
                                           double prefill_tps,
                                           double decode_tps,
                                           size_t prompt_tokens,
                                           size_t completion_tokens,
                                           float confidence = 0.0f,
                                           bool cloud_handoff = false) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":" << (cloud_handoff ? "false" : "true") << ",";
    json << "\"error\":null,";
    json << "\"cloud_handoff\":" << (cloud_handoff ? "true" : "false") << ",";
    json << "\"response\":\"" << escape_json_string(regular_response) << "\",";
    json << "\"function_calls\":[";
    for (size_t i = 0; i < function_calls.size(); ++i) {
        if (i > 0) json << ",";
        json << function_calls[i];
    }
    json << "],";
    json << "\"confidence\":" << std::fixed << std::setprecision(4) << confidence << ",";
    json << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ",";
    json << "\"decode_tps\":" << std::fixed << std::setprecision(2) << decode_tps << ",";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb() << ",";
    json << "\"prefill_tokens\":" << prompt_tokens << ",";
    json << "\"decode_tokens\":" << completion_tokens << ",";
    json << "\"total_tokens\":" << (prompt_tokens + completion_tokens);
    json << "}";
    return json.str();
}

inline std::string construct_cloud_handoff_json(float confidence,
                                                 double time_to_first_token,
                                                 double prefill_tps,
                                                 size_t prompt_tokens) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":false,";
    json << "\"error\":null,";
    json << "\"cloud_handoff\":true,";
    json << "\"response\":null,";
    json << "\"function_calls\":[],";
    json << "\"confidence\":" << std::fixed << std::setprecision(4) << confidence << ",";
    json << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ",";
    json << "\"decode_tps\":0.0,";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb() << ",";
    json << "\"prefill_tokens\":" << prompt_tokens << ",";
    json << "\"decode_tokens\":0,";
    json << "\"total_tokens\":" << prompt_tokens;
    json << "}";
    return json.str();
}

} // namespace ffi
} // namespace cactus

#ifdef __cplusplus
extern "C" {
#endif

const char* cactus_get_last_error();

__attribute__((weak))
const char* register_app(const char* encrypted_data);

__attribute__((weak))
const char* get_device_id(const char* current_token);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {

__attribute__((weak))
inline const char* register_app(const char* encrypted_data) {
    (void)encrypted_data;
    static thread_local std::string uuid_storage;
    uuid_storage = cactus::ffi::generateUUID();
    return uuid_storage.c_str();
}

__attribute__((weak))
inline const char* get_device_id(const char* current_token) {
    (void)current_token;
    static thread_local std::string uuid_storage;
    uuid_storage = cactus::ffi::generateUUID();
    return uuid_storage.c_str();
}
}
#endif

#endif // CACTUS_UTILS_H
