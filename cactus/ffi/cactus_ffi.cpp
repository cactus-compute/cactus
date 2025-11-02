#include "cactus_ffi.h"
#include "../engine/engine.h"
#include <memory>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <cstring>

using namespace cactus::engine;


struct CactusModel {
    std::unique_ptr<Model> model;
    std::string model_path;
    std::unordered_map<std::string, std::vector<uint32_t>> stop_sequence_cache;
    
    std::atomic<bool> should_stop;
    
    CactusModel() : should_stop(false) {}
};

static std::vector<ChatMessage> parse_messages_json(const std::string& json) {
    std::vector<ChatMessage> messages;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid JSON: expected array");
    }
    
    pos = json.find('{', pos);
    while (pos != std::string::npos) {
        ChatMessage msg;
        
        size_t role_pos = json.find("\"role\"", pos);
        if (role_pos == std::string::npos) break;
        
        size_t role_start = json.find('\"', role_pos + 6) + 1;
        size_t role_end = json.find('\"', role_start);
        msg.role = json.substr(role_start, role_end - role_start);
        
        size_t content_pos = json.find("\"content\"", role_end);
        if (content_pos == std::string::npos) break;
        
        size_t content_start = json.find('\"', content_pos + 9) + 1;
        size_t content_end = content_start;
        
        while (content_end < json.length()) {
            content_end = json.find('\"', content_end);
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

struct ToolFunction {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> parameters;
};

static std::vector<ToolFunction> parse_tools_json(const std::string& json) {
    std::vector<ToolFunction> tools;
    
    if (json.empty()) return tools;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) return tools;
    
    pos = json.find("\"function\"", pos);
    while (pos != std::string::npos) {
        ToolFunction tool;
        
        size_t name_pos = json.find("\"name\"", pos);
        if (name_pos != std::string::npos) {
            size_t name_start = json.find('\"', name_pos + 6) + 1;
            size_t name_end = json.find('\"', name_start);
            tool.name = json.substr(name_start, name_end - name_start);
        }
        
        size_t desc_pos = json.find("\"description\"", pos);
        if (desc_pos != std::string::npos) {
            size_t desc_start = json.find('\"', desc_pos + 13) + 1;
            size_t desc_end = json.find('\"', desc_start);
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

static void parse_options_json(const std::string& json, 
                               float& temperature, float& top_p, 
                               size_t& top_k, size_t& max_tokens,
                               std::vector<std::string>& stop_sequences) {
    temperature = 0.7f;
    top_p = 0.95f;
    top_k = 20;
    max_tokens = 100;
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
            size_t seq_pos = json.find('\"', pos);
            
            while (seq_pos != std::string::npos && seq_pos < end_pos) {
                size_t seq_start = seq_pos + 1;
                size_t seq_end = json.find('\"', seq_start);
                if (seq_end != std::string::npos) {
                    stop_sequences.push_back(json.substr(seq_start, seq_end - seq_start));
                }
                seq_pos = json.find('\"', seq_end + 1);
            }
        }
    }
}


static std::unordered_set<uint32_t> get_stop_tokens(CactusModel* wrapper, const std::vector<std::string>& stop_sequences) {
    std::unordered_set<uint32_t> stop_tokens;
    auto* tokenizer = wrapper->model->get_tokenizer();
    
    stop_tokens.insert(tokenizer->get_eos_token());
    
    for (const auto& stop_seq : stop_sequences) {
        auto cache_it = wrapper->stop_sequence_cache.find(stop_seq);
        if (cache_it != wrapper->stop_sequence_cache.end()) {
            
            for (uint32_t token : cache_it->second) {
                stop_tokens.insert(token);
            }
        } else {
            
            auto tokens = tokenizer->encode(stop_seq);
            wrapper->stop_sequence_cache[stop_seq] = tokens;
            for (uint32_t token : tokens) {
                stop_tokens.insert(token);
            }
        }
    }
    
    return stop_tokens;
}

extern "C" {

static std::string last_error_message;

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, size_t context_size) {
    try {
        auto* wrapper = new CactusModel();
        wrapper->model = create_model(model_path);
        wrapper->model_path = model_path;

        if (!wrapper->model) {
            last_error_message = "Failed to create model from: " + std::string(model_path);
            delete wrapper;
            return nullptr;
        }

        if (!wrapper->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model from: " + std::string(model_path);
            delete wrapper;
            return nullptr;
        }
        
        auto* tokenizer = wrapper->model->get_tokenizer();
        std::vector<std::string> common_stops = {
            "\n\n", "###", "Human:", "Assistant:", "<|end|>", "<|endoftext|>", 
            "\n---", "User:", "AI:", "</s>", "<s>", "\n\nHuman:", "\n\nAssistant:"
        };
        for (const auto& stop_seq : common_stops) {
            wrapper->stop_sequence_cache[stop_seq] = tokenizer->encode(stop_seq);
        }
        
        return wrapper;
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error during model initialization";
        return nullptr;
    }
}

int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? 
            "Model not initialized. Check model path and files." : last_error_message;
        
        for (auto& c : error_msg) {
            if (c == '"') c = '\'';
            if (c == '\n') c = ' ';
        }
        std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
    
    if (!messages_json || !response_buffer || buffer_size == 0) {
        std::string error_json = "{\"success\":false,\"error\":\"Invalid parameters\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto* wrapper = static_cast<CactusModel*>(model);
        auto* tokenizer = wrapper->model->get_tokenizer();
        wrapper->should_stop = false;
        
        wrapper->model->reset_cache();
        
        
        auto messages = parse_messages_json(messages_json);
        if (messages.empty()) {
            std::string error_json = "{\"success\":false,\"error\":\"No messages provided\"}";
            std::strcpy(response_buffer, error_json.c_str());
            return -1;
        }
        
        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        parse_options_json(options_json ? options_json : "", 
                          temperature, top_p, top_k, max_tokens, stop_sequences);
        
        
        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0) {
            tools = parse_tools_json(tools_json);
        }
        
        
        std::string full_prompt;
        if (!tools.empty()) {
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
            
            full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools_json);
        } else {
            full_prompt = tokenizer->format_chat_prompt(messages, true);
        }

        if (full_prompt.find("ERROR:") == 0) {
            std::string error_msg = full_prompt.substr(6);
            std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
            std::strcpy(response_buffer, error_json.c_str());
            return -1;
        }

        std::vector<uint32_t> tokens_to_process = tokenizer->encode(full_prompt);
        size_t prompt_tokens = tokens_to_process.size();
        
        
        std::unordered_set<uint32_t> stop_tokens = get_stop_tokens(wrapper, stop_sequences);
        
        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        std::string decoded_so_far;  


        uint32_t next_token;
        if (tokens_to_process.empty()) {
            next_token = wrapper->model->generate({}, temperature, top_p, top_k);
        } else {
            next_token = wrapper->model->generate(tokens_to_process, temperature, top_p, top_k);
        }

        auto token_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time);
        time_to_first_token = duration.count() / 1000.0;

        if (stop_tokens.count(next_token)) {
            generated_tokens.push_back(next_token);
        } else {
            generated_tokens.push_back(next_token);

            if (callback) {
                std::string full_decoded = tokenizer->decode(generated_tokens);
                std::string new_text = full_decoded.substr(decoded_so_far.length());
                decoded_so_far = full_decoded;
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (wrapper->should_stop) {
                    break;
                }

                std::vector<uint32_t> single_token = {next_token};
                next_token = wrapper->model->generate(single_token, temperature, top_p, top_k);

                if (stop_tokens.count(next_token)) {
                    break;
                }

                generated_tokens.push_back(next_token);

                if (callback) {
                    std::string full_decoded = tokenizer->decode(generated_tokens);
                    std::string new_text = full_decoded.substr(decoded_so_far.length());
                    decoded_so_far = full_decoded;
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time_ms = total_duration.count() / 1000.0;
        
        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double tokens_per_second = completion_tokens > 1 ? 
            ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;
        
        std::string response_text = tokenizer->decode(generated_tokens);

    std::string regular_response = response_text;
    std::vector<std::string> function_calls;
    function_calls.reserve(4); 

    const char* FUNCTION_CALL_MARKER = "\"function_call\"";
    const size_t MARKER_LEN = 15;
    size_t search_pos = 0;

    const size_t text_len = response_text.length();

    while (search_pos < text_len) {
        size_t marker_pos = response_text.find(FUNCTION_CALL_MARKER, search_pos);
        if (marker_pos == std::string::npos) break;

        size_t colon_pos = marker_pos + MARKER_LEN;
        while (colon_pos < text_len && response_text[colon_pos] != ':') colon_pos++;
        if (colon_pos >= text_len) break;

        size_t json_start = colon_pos + 1;
        while (json_start < text_len && response_text[json_start] != '{') json_start++;
        if (json_start >= text_len) break;

        int brace_count = 1;
        size_t json_end = json_start + 1;
        while (json_end < text_len && brace_count > 0) {
            char c = response_text[json_end];
            brace_count += (c == '{') - (c == '}');
            json_end++;
        }

        if (brace_count == 0) {
            function_calls.push_back(response_text.substr(json_start, json_end - json_start));

            size_t outer_start = marker_pos;
            while (outer_start > 0 && response_text[outer_start] != '{') outer_start--;

            if (outer_start > 0 || response_text[0] == '{') {
                int outer_brace_count = 1;
                size_t outer_end = outer_start + 1;
                while (outer_end < text_len && outer_brace_count > 0) {
                    char c = response_text[outer_end];
                    outer_brace_count += (c == '{') - (c == '}');
                    outer_end++;
                }
                if (outer_brace_count == 0) {
                    regular_response = response_text.substr(0, outer_start);
                    
                    size_t last = regular_response.find_last_not_of(" \n\r\t");
                    if (last != std::string::npos) {
                        regular_response.erase(last + 1);
                    }
                }
            }
        }

        search_pos = json_end;
    }

        size_t call_pos = 0;
        const std::string CALL_START = "<|tool_call_start|>";
        const std::string CALL_END = "<|tool_call_end|>";
        while ((call_pos = response_text.find(CALL_START, call_pos)) != std::string::npos) {
            size_t call_end = response_text.find(CALL_END, call_pos + CALL_START.size());
            if (call_end == std::string::npos) break;
            size_t payload_start = call_pos + CALL_START.size();
            std::string payload = response_text.substr(payload_start, call_end - payload_start);

            size_t json_start = payload.find('{');
            if (json_start != std::string::npos) {
                int brace_count = 1;
                size_t json_end = json_start + 1;
                while (json_end < payload.size() && brace_count > 0) {
                    if (payload[json_end] == '{') brace_count++;
                    else if (payload[json_end] == '}') brace_count--;
                    json_end++;
                }
                if (brace_count == 0) {
                    std::string function_call = payload.substr(json_start, json_end - json_start);
                    function_calls.push_back(function_call);

                    regular_response = response_text.substr(0, call_pos);
                    if (call_end + CALL_END.size() < response_text.size()) {
                        regular_response += response_text.substr(call_end + CALL_END.size());
                    }
                    response_text = regular_response;
                    call_pos = 0;
                    continue;
                }
            }

            auto trim = [](std::string s) {
                size_t b = s.find_first_not_of(" \t\n\r");
                if (b == std::string::npos) return std::string();
                size_t e = s.find_last_not_of(" \t\n\r");
                return s.substr(b, e - b + 1);
            };
            std::string core = trim(payload);
            if (!core.empty()) {
                if (core.front() == '[' && core.back() == ']') {
                    core = core.substr(1, core.size() - 2);
                    core = trim(core);
                }
                size_t lp = core.find('(');
                size_t rp = core.rfind(')');
                if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
                    std::string fname = trim(core.substr(0, lp));
                    std::string args_str = core.substr(lp + 1, rp - lp - 1);
                    std::vector<std::string> parts;
                    std::string cur;
                    bool in_quotes = false;
                    for (size_t i = 0; i < args_str.size(); ++i) {
                        char c = args_str[i];
                        if (c == '"') {
                            in_quotes = !in_quotes;
                            cur += c;
                        } else if (c == ',' && !in_quotes) {
                            parts.push_back(trim(cur));
                            cur.clear();
                        } else {
                            cur += c;
                        }
                    }
                    if (!cur.empty()) parts.push_back(trim(cur));

                    auto json_escape = [](const std::string& s) {
                        std::ostringstream o;
                        for (char ch : s) {
                            switch (ch) {
                                case '"': o << "\\\""; break;
                                case '\\': o << "\\\\"; break;
                                case '\n': o << "\\n"; break;
                                case '\r': o << "\\r"; break;
                                case '\t': o << "\\t"; break;
                                default: o << ch; break;
                            }
                        }
                        return o.str();
                    };

                    std::ostringstream args_json;
                    args_json << "{";
                    bool first = true;
                    for (const auto& p : parts) {
                        if (p.empty()) continue;
                        size_t eq = p.find('=');
                        if (eq == std::string::npos) continue;
                        std::string key = trim(p.substr(0, eq));
                        std::string val = trim(p.substr(eq + 1));
                        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                            val = val.substr(1, val.size() - 2);
                        }
                        if (!first) args_json << ",";
                        first = false;
                        args_json << "\"" << json_escape(key) << "\":\"" << json_escape(val) << "\"";
                    }
                    args_json << "}";

                    std::ostringstream fc;
                    fc << "{\"name\":\"" << json_escape(fname) << "\",\"arguments\":" << args_json.str() << "}";
                    function_calls.push_back(fc.str());

                    regular_response = response_text.substr(0, call_pos);
                    if (call_end + CALL_END.size() < response_text.size()) {
                        regular_response += response_text.substr(call_end + CALL_END.size());
                    }
                    response_text = regular_response;
                    call_pos = 0;
                    continue;
                }
            }
            call_pos = call_end + CALL_END.size();
        }
        
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
        
        std::string result = json_response.str();
        if (result.length() >= buffer_size) {
            std::string error_json = "{\"success\":false,\"error\":\"Response buffer too small\"}";
            std::strcpy(response_buffer, error_json.c_str());
            return -1;
        }
        
        std::strcpy(response_buffer, result.c_str());
        
        return static_cast<int>(result.length());
        
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        for (auto& c : error_msg) {
            if (c == '"') c = '\'';
            if (c == '\n') c = ' ';
        }
        std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) {
        delete static_cast<CactusModel*>(model);
    }
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    
    auto* wrapper = static_cast<CactusModel*>(model);
    wrapper->model->reset_cache();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* wrapper = static_cast<CactusModel*>(model);
    wrapper->should_stop = true;
}

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) {
        return -1;
    }
    
    if (!text || !embeddings_buffer || buffer_size == 0) {
        return -1;
    }
    
    try {
        auto* wrapper = static_cast<CactusModel*>(model);
        auto* tokenizer = wrapper->model->get_tokenizer();
        
        std::vector<uint32_t> tokens = tokenizer->encode(text);
        
        if (tokens.empty()) {
            return -1;
        }
        
        std::vector<float> embeddings = wrapper->model->get_embeddings(tokens, true);
        
        if (embeddings.size() * sizeof(float) > buffer_size) {
            return -2; 
        }
        
        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        
        if (embedding_dim) {
            *embedding_dim = embeddings.size();
        }
        
        return static_cast<int>(embeddings.size());
        
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding generation";
        return -1;
    }
}

}