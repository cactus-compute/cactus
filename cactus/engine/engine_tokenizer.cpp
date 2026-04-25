#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>

extern "C" {
    #include "../../libs/stb/stb_image.h"
}

namespace cactus {
namespace engine {

namespace {

std::string trim_copy(const std::string& value) {
    size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

TokenizerRuntimeConfig::TokenizerType parse_tokenizer_type(const std::string& value) {
    if (value == "bpe") return TokenizerRuntimeConfig::TokenizerType::BPE;
    if (value == "sentencepiece") return TokenizerRuntimeConfig::TokenizerType::SENTENCEPIECE;
    return TokenizerRuntimeConfig::TokenizerType::UNKNOWN;
}

TokenizerRuntimeConfig::VocabFormat parse_vocab_format(const std::string& value) {
    if (value == "id_tab_token") return TokenizerRuntimeConfig::VocabFormat::ID_TAB_TOKEN;
    if (value == "line_token") return TokenizerRuntimeConfig::VocabFormat::LINE_TOKEN;
    return TokenizerRuntimeConfig::VocabFormat::UNKNOWN;
}

TokenizerRuntimeConfig::Normalizer parse_normalizer(const std::string& value) {
    if (value == "metaspace") return TokenizerRuntimeConfig::Normalizer::METASPACE;
    if (value == "byte_level") return TokenizerRuntimeConfig::Normalizer::BYTE_LEVEL;
    return TokenizerRuntimeConfig::Normalizer::NONE;
}

TokenizerRuntimeConfig::Decoder parse_decoder(const std::string& value) {
    if (value == "replace_metaspace") return TokenizerRuntimeConfig::Decoder::REPLACE_METASPACE;
    if (value == "byte_level") return TokenizerRuntimeConfig::Decoder::BYTE_LEVEL;
    return TokenizerRuntimeConfig::Decoder::NONE;
}

void skip_json_whitespace(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
}

bool extract_added_token_object(const std::string& json, size_t& pos, std::string& out_object) {
    skip_json_whitespace(json, pos);
    if (pos >= json.size() || json[pos] != '{') {
        return false;
    }

    size_t start = pos;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;

    while (pos < json.size()) {
        char c = json[pos++];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            if (depth == 0) {
                return false;
            }
            --depth;
            if (depth == 0) {
                out_object = json.substr(start, pos - start);
                return true;
            }
        }
    }

    return false;
}

bool parse_added_token_entry(const std::string& object, std::string& token_content, uint32_t& token_id,
                             bool& is_special) {
    token_content.clear();
    token_id = 0;
    is_special = false;

    size_t id_key = object.find("\"id\"");
    if (id_key == std::string::npos) {
        return false;
    }
    size_t id_colon = object.find(':', id_key);
    if (id_colon == std::string::npos) {
        return false;
    }
    size_t id_pos = id_colon + 1;
    skip_json_whitespace(object, id_pos);
    size_t id_end = id_pos;
    while (id_end < object.size() && std::isdigit(static_cast<unsigned char>(object[id_end]))) {
        ++id_end;
    }
    if (id_end == id_pos) {
        return false;
    }
    token_id = static_cast<uint32_t>(std::stoul(object.substr(id_pos, id_end - id_pos)));

    size_t content_key = object.find("\"content\"");
    if (content_key == std::string::npos) {
        return false;
    }
    size_t content_colon = object.find(':', content_key);
    if (content_colon == std::string::npos) {
        return false;
    }
    size_t content_pos = object.find('"', content_colon + 1);
    if (content_pos == std::string::npos) {
        return false;
    }
    ++content_pos;
    token_content = extract_json_string(object, content_pos);

    size_t special_key = object.find("\"special\"");
    if (special_key != std::string::npos) {
        size_t special_colon = object.find(':', special_key);
        if (special_colon != std::string::npos) {
            size_t special_pos = special_colon + 1;
            skip_json_whitespace(object, special_pos);
            is_special = object.compare(special_pos, 4, "true") == 0;
        }
    }

    return true;
}

void load_tokenizer_json_added_special_tokens(
    const std::string& tokenizer_json_path,
    std::unordered_map<std::string, uint32_t>& special_tokens) {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t pos = content.find("\"added_tokens\"");
    if (pos == std::string::npos) {
        return;
    }

    pos = content.find('[', pos);
    if (pos == std::string::npos) {
        return;
    }
    ++pos;

    while (pos < content.size()) {
        skip_json_whitespace(content, pos);
        if (pos >= content.size() || content[pos] == ']') {
            break;
        }

        std::string object;
        if (!extract_added_token_object(content, pos, object)) {
            ++pos;
            continue;
        }

        std::string token_content;
        uint32_t token_id = 0;
        bool is_special = false;
        if (parse_added_token_entry(object, token_content, token_id, is_special) && is_special) {
            special_tokens[token_content] = token_id;
        }

        skip_json_whitespace(content, pos);
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
        }
    }
}

}  // namespace

TokenizerRuntimeConfig load_tokenizer_runtime_config(const std::string& config_file) {
    TokenizerRuntimeConfig config;

    std::ifstream file(config_file);
    if (!file.is_open()) {
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        const std::string key = trim_copy(line.substr(0, eq_pos));
        const std::string value = trim_copy(line.substr(eq_pos + 1));

        if (key == "tokenizer_type") {
            config.tokenizer_type = parse_tokenizer_type(value);
        } else if (key == "vocab_format") {
            config.vocab_format = parse_vocab_format(value);
        } else if (key == "normalizer") {
            config.normalizer = parse_normalizer(value);
        } else if (key == "decoder") {
            config.decoder = parse_decoder(value);
        } else if (key == "byte_fallback") {
            config.byte_fallback = (value == "true" || value == "1");
        } else if (key == "has_chat_template") {
            config.has_chat_template = (value == "true" || value == "1");
        }
    }

    return config;
}

void load_special_tokens_map(const std::string& config_file, std::unordered_map<std::string, uint32_t>& special_tokens) {
    special_tokens.clear();

    std::ifstream file(config_file);
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        size_t pos = content.find("\"special_tokens\"");
        if (pos != std::string::npos) {
            pos = content.find("{", pos);
            if (pos != std::string::npos) {
                size_t end_pos = content.find("}", pos);
                if (end_pos != std::string::npos) {
                    std::string special_tokens_section = content.substr(pos + 1, end_pos - pos - 1);
                    std::istringstream iss(special_tokens_section);
                    std::string line;

                    while (std::getline(iss, line)) {
                        size_t colon_pos = line.find(":");
                        if (colon_pos == std::string::npos) continue;

                        std::string id_part = line.substr(0, colon_pos);
                        std::string token_part = line.substr(colon_pos + 1);

                        size_t id_start = id_part.find("\"");
                        size_t id_end = id_part.find("\"", id_start + 1);
                        if (id_start == std::string::npos || id_end == std::string::npos) continue;

                        uint32_t token_id =
                            static_cast<uint32_t>(std::stoul(id_part.substr(id_start + 1, id_end - id_start - 1)));

                        size_t token_start = token_part.find("\"");
                        if (token_start == std::string::npos) continue;
                        size_t value_pos = token_start + 1;
                        std::string token_content = extract_json_string(token_part, value_pos);
                        special_tokens[token_content] = token_id;
                    }
                }
            }
        }
    }

    size_t slash_pos = config_file.find_last_of("/\\");
    std::string dir = (slash_pos == std::string::npos) ? "." : config_file.substr(0, slash_pos);
    load_tokenizer_json_added_special_tokens(dir + "/tokenizer.json", special_tokens);
}

std::vector<std::string> split_with_special_tokens(const std::string& text,
                                                    const std::unordered_map<std::string, uint32_t>& special_tokens) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < text.size()) {
        size_t best_match_pos = text.size();
        size_t best_match_len = 0;
        std::string best_special_token;

        for (const auto& [special_token, token_id] : special_tokens) {
            size_t pos = text.find(special_token, start);
            if (pos != std::string::npos &&
                (pos < best_match_pos || (pos == best_match_pos && special_token.length() > best_match_len))) {
                best_match_pos = pos;
                best_match_len = special_token.length();
                best_special_token = special_token;
            }
        }

        if (best_match_pos < text.size()) {
            if (best_match_pos > start)
                result.push_back(text.substr(start, best_match_pos - start));
            result.push_back(best_special_token);
            start = best_match_pos + best_match_len;
        } else {
            if (start < text.size())
                result.push_back(text.substr(start));
            break;
        }
    }
    return result;
}

void Tokenizer::load_chat_template(const std::string& template_file) {
    std::ifstream file(template_file);
    if (!file.is_open()) {
        has_chat_template_ = false;
        return;
    }
    chat_template_ = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    has_chat_template_ = !chat_template_.empty();
}

void Tokenizer::detect_model_type(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        model_type_ = ModelType::UNKNOWN;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("model_type");
        if (pos != std::string::npos) {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);

            if (line.find("qwen3_5") != std::string::npos) {
                model_type_ = ModelType::QWEN3P5;
                break;
            } else if (line.find("qwen") != std::string::npos) {
                model_type_ = ModelType::QWEN;
                break;
            } else if (line.find("gemma4") != std::string::npos || line.find("tinyllama") != std::string::npos) {
                model_type_ = ModelType::GEMMA4;
                break;
            } else if (line.find("gemma") != std::string::npos) {
                model_type_ = ModelType::GEMMA;
                break;
            } else if(line.find("lfm2") != std::string::npos) {
                model_type_ = ModelType::LFM2;
            } else if (line.find("bert") != std::string::npos) {
                model_type_ = ModelType::BERT;
                break;
            } else if (line.find("whisper") != std::string::npos) {
                model_type_ = ModelType::WHISPER;
                break;
            } else if (line.find("parakeet") != std::string::npos) {
                model_type_ = ModelType::PARAKEET;
                break;
            } else if (line.find("needle") != std::string::npos) {
                model_type_ = ModelType::NEEDLE;
                break;
            } else if (line.find("youtu") != std::string::npos) {
                model_type_ = ModelType::YOUTU;
                break;
            } else {
                model_type_ = ModelType::UNKNOWN;
            }
        }
    }
    file.clear();
    file.seekg(0);

    while (std::getline(file, line)) {
        size_t pos2 = line.find("model_variant");
        if (pos2 != std::string::npos) {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);

            if (line.find("vlm") != std::string::npos) {
                model_variant_ = ModelVariant::VLM;
                break;
            } else if (line.find("extract") != std::string::npos) {
                model_variant_ = ModelVariant::EXTRACT;
                break;
            } else if (line.find("rag") != std::string::npos) {
                model_variant_ = ModelVariant::RAG;
                break;
            } else {
                model_variant_ = ModelVariant::DEFAULT;
            }
        }
    }

    file.clear();
    file.seekg(0);
    while (std::getline(file, line)) {
        auto parse_uint = [&](const std::string& key, uint32_t& out) {
            size_t p = line.find(key + "=");
            if (p != std::string::npos) {
                out = static_cast<uint32_t>(std::stoul(line.substr(p + key.size() + 1)));
            }
        };
        parse_uint("vision_patch_size", vision_patch_size_);
        parse_uint("vision_pooling_kernel_size", vision_pooling_kernel_size_);
        parse_uint("vision_default_output_length", vision_default_output_length_);
        parse_uint("vision_image_size", vision_image_size_);
    }

    file.close();
}

std::string Tokenizer::get_default_stop_sequence() const {
    switch (model_type_) {
        case ModelType::GEMMA:
            return "<end_of_turn>";
        case ModelType::GEMMA4:
            return "<turn|>";
        case ModelType::QWEN:
        case ModelType::QWEN3P5:
        case ModelType::LFM2:
            return "<|im_end|>";
        case ModelType::NEEDLE:
            return "";
        default:
            return "<|im_end|>";
    }
}

std::vector<uint32_t> Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    std::string formatted_prompt = format_chat_prompt(messages, add_generation_prompt);
    return encode(formatted_prompt);
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                          const std::string& tools_json, bool enable_thinking_if_supported) const {
    bool has_images = false;
    for (const auto& msg : messages) {
        if (!msg.images.empty()) {
            has_images = true;
            break;
        }
    }
    if (model_type_ == ModelType::LFM2 && has_images) {
        return format_lfm2_vl_style(messages, add_generation_prompt, tools_json);
    }
    
    switch (model_type_) {
        case ModelType::QWEN:
        case ModelType::QWEN3P5:
            return format_qwen_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
        case ModelType::GEMMA:
            return format_gemma_style(messages, add_generation_prompt, tools_json);
        case ModelType::GEMMA4:
            return format_gemma4_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
        case ModelType::LFM2:
            return format_lfm2_style(messages, add_generation_prompt, tools_json);
        case ModelType::NEEDLE:
            return format_needle_style(messages, add_generation_prompt, tools_json);
        case ModelType::YOUTU:
            return format_youtu_style(messages, add_generation_prompt, tools_json);
        default:
            return format_qwen_style(messages, add_generation_prompt, tools_json);
    }
}

std::string Tokenizer::format_needle_style(const std::vector<ChatMessage>& messages,
                                           bool /*add_generation_prompt*/,
                                           const std::string& tools_json) const {
    std::string serialized_tools = tools_json.empty() ? "[]" : tools_json;
    return format_needle_query_text(messages) + "<tools>" + serialized_tools + "</s>";
}

std::string Tokenizer::format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported) const {
    std::string result;
    size_t start_idx = 0;

    if (!tools_json.empty()) {
        result += "<|im_start|>system\n";
        if (!messages.empty() && messages[0].role == "system") {
            result += messages[0].content + "\n\n";
            start_idx = 1;
        }
        result += "# Tools\n\n";
        result += "You may call one or more functions to assist with the user query.\n\n";
        result += "You are provided with function signatures within <tools></tools> XML tags:\n";
        result += "<tools>";
        result += tools_json;
        result += "\n</tools>\n\n";
        result += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n";
        result += "<tool_call>\n";
        result += "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";
        result += "</tool_call>";
        result += "<|im_end|>\n";
    } else {
        if (!messages.empty() && messages[0].role == "system") {
            result += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n";
            start_idx = 1;
        }
    }

    size_t last_query_index = messages.empty() ? 0 : messages.size() - 1;
    for (size_t idx = messages.size(); idx > 0; idx--) {
        const auto& m = messages[idx - 1];
        if (m.role == "user") {
            bool is_tool_response = m.content.size() >= 31 &&
                m.content.substr(0, 15) == "<tool_response>" &&
                m.content.substr(m.content.size() - 16) == "</tool_response>";
            if (!is_tool_response) {
                last_query_index = idx - 1;
                break;
            }
        }
    }

    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];
        if (msg.role == "user" || (msg.role == "system" && i > 0)) {
            result += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            if (i > last_query_index && i == messages.size() - 1) {
                result += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
                std::string content = msg.content;
                size_t lstrip = content.find_first_not_of('\n');
                if (lstrip != std::string::npos) {
                    result += content.substr(lstrip);
                }
            } else {
                result += "<|im_start|>assistant\n" + msg.content;
            }
            if (!msg.tool_calls.empty()) {
                for (size_t j = 0; j < msg.tool_calls.size(); j++) {
                    if ((j == 0 && !msg.content.empty()) || j > 0) {
                        result += "\n";
                    }
                    result += "<tool_call>\n{\"name\": \"" + msg.tool_calls[j].name
                        + "\", \"arguments\": " + msg.tool_calls[j].arguments
                        + "}\n</tool_call>";
                }
            }
            result += "<|im_end|>\n";
        } else if (msg.role == "tool") {
            bool first_tool = (i == 0 || messages[i - 1].role != "tool");
            bool last_tool = (i == messages.size() - 1 || messages[i + 1].role != "tool");
            if (first_tool) {
                result += "<|im_start|>user";
            }
            result += "\n<tool_response>\n" + msg.content + "\n</tool_response>";
            if (last_tool) {
                result += "<|im_end|>\n";
            }
        }
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
        if (!enable_thinking_if_supported) {
            result += "<think>\n\n</think>\n\n";
        }
    }

    return result;
}

std::string Tokenizer::format_lfm2_style(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt,
                                         const std::string& tools_json) const
{
    std::string result = "<|startoftext|>";

    std::string sys_content;
    bool has_system_msg = false;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            sys_content = msg.content;
            has_system_msg = true;
            break;
        }
    }

    if (!tools_json.empty()) {
        if (!sys_content.empty()) {
            sys_content += "\n";
        }
        sys_content += "List of tools: <|tool_list_start|>[";
        if (!tools_json.empty()) {
            sys_content += "\n";
            sys_content += tools_json;
            sys_content += "\n";
        }
        sys_content += "]<|tool_list_end|>";
        sys_content += "\n\nWhen you need to call a tool, use this exact format:\n";
        sys_content += "<|tool_call_start|>[function_name(arg1=\"value1\", arg2=\"value2\")]<|tool_call_end|>\n";
        sys_content += "You can call multiple tools by using multiple tool call blocks.";
    }

    if (!sys_content.empty()) {
        result += "<|im_start|>system\n";
        result += sys_content;
        result += "<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        if (msg.role == "system" && has_system_msg) {
            has_system_msg = false;
            continue;
        }
        result += "<|im_start|>" + msg.role + "\n";
        if (msg.role == "tool") {
            result += "<|tool_response_start|>";
            result += msg.content;
            result += "<|tool_response_end|>";
        } else {
            result += msg.content;
            if (msg.role == "assistant" && !msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    result += "<|tool_call_start|>[{\"name\": \"" + tc.name
                        + "\", \"arguments\": " + tc.arguments + "}]<|tool_call_end|>";
                }
            }
        }
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}

static std::string format_tool_call_for_prompt(const std::string& name, const std::string& args_json, bool use_pipe_tags) {
    const char* tag_start = use_pipe_tags ? "<|tool_call>" : "<start_function_call>";
    const char* tag_end   = use_pipe_tags ? "<tool_call|>" : "<end_function_call>";

    std::string formatted_args = "{";
    if (!args_json.empty()) {
        std::map<std::string, std::string> args;
        size_t pos = 0;
        while (pos < args_json.length() && args_json[pos] != '{') pos++;
        if (pos < args_json.length()) pos++;

        while (pos < args_json.length()) {
            while (pos < args_json.length() && std::isspace(static_cast<unsigned char>(args_json[pos]))) pos++;
            if (pos >= args_json.length() || args_json[pos] == '}') break;
            if (args_json[pos] == ',') { pos++; continue; }
            if (args_json[pos] != '"') break;
            pos++;

            std::string key;
            while (pos < args_json.length() && args_json[pos] != '"') {
                if (args_json[pos] == '\\' && pos + 1 < args_json.length()) { pos++; }
                key += args_json[pos++];
            }
            if (pos < args_json.length()) pos++;

            while (pos < args_json.length() && (std::isspace(static_cast<unsigned char>(args_json[pos])) || args_json[pos] == ':')) pos++;

            std::string value;
            if (pos < args_json.length() && args_json[pos] == '"') {
                pos++;
                std::string str_val;
                while (pos < args_json.length() && args_json[pos] != '"') {
                    if (args_json[pos] == '\\' && pos + 1 < args_json.length()) {
                        pos++;
                        if (args_json[pos] == 'n') str_val += '\n';
                        else if (args_json[pos] == 't') str_val += '\t';
                        else str_val += args_json[pos];
                    } else {
                        str_val += args_json[pos];
                    }
                    pos++;
                }
                if (pos < args_json.length()) pos++;
                value = "<escape>" + str_val + "<escape>";
            } else {
                size_t val_start = pos;
                while (pos < args_json.length() && args_json[pos] != ',' && args_json[pos] != '}') pos++;
                value = args_json.substr(val_start, pos - val_start);
                while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
            }

            if (!key.empty()) args[key] = value;
        }

        bool first = true;
        for (const auto& [k, v] : args) {
            if (!first) formatted_args += ",";
            first = false;
            formatted_args += k + ":" + v;
        }
    }
    formatted_args += "}";

    return std::string(tag_start) + "call:" + name + formatted_args + tag_end;
}

std::string Tokenizer::format_gemma4_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                               const std::string& tools_json, bool enable_thinking_if_supported) const {
    std::string result = "<bos>";

    std::string sys_content;
    size_t first_msg = 0;
    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
        sys_content = messages[0].content;
        first_msg = 1;
    }

    if (enable_thinking_if_supported || !sys_content.empty() || !tools_json.empty()) {
        result += "<|turn>system\n";
        if (enable_thinking_if_supported) {
            result += "<|think|>";
        }
        result += sys_content;
        result += tools_json;
        result += "<turn|>\n";
    }

    auto strip_channel = [](const std::string& text) -> std::string {
        const std::string open_tag = "<|channel>";
        const std::string close_tag = "<channel|>";
        std::string out;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t open_pos = text.find(open_tag, pos);
            if (open_pos == std::string::npos) {
                out += text.substr(pos);
                break;
            }
            out += text.substr(pos, open_pos - pos);
            size_t close_pos = text.find(close_tag, open_pos + open_tag.size());
            if (close_pos == std::string::npos) {
                break;
            }
            pos = close_pos + close_tag.size();
        }
        return out;
    };

    auto compute_soft_tokens = [&](const std::string& image_path) -> size_t {
        int w = 0, h = 0, c = 0;
        unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) return 0;
        stbi_image_free(data);

        uint32_t p = vision_patch_size_;
        uint32_t k = vision_pooling_kernel_size_;
        uint32_t side = k * p;
        uint32_t max_patches = vision_default_output_length_ * k * k;
        float factor = std::sqrt(static_cast<float>(max_patches) * p * p /
                                 (static_cast<float>(h) * w));
        int th = static_cast<int>(std::floor(factor * h / side)) * side;
        int tw = static_cast<int>(std::floor(factor * w / side)) * side;
        if (th == 0) th = side;
        if (tw == 0) tw = side;
        return static_cast<size_t>((th / p / k) * (tw / p / k));
    };

    for (size_t i = first_msg; i < messages.size(); i++) {
        const auto& msg = messages[i];
        std::string role = (msg.role == "assistant") ? "model" : msg.role;
        result += "<|turn>" + role + "\n";
        if (role == "model") {
            result += strip_channel(msg.content);
            if (!msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    result += format_tool_call_for_prompt(tc.name, tc.arguments, true);
                }
            }
        } else {
            for (const auto& image_path : msg.images) {
                size_t n = compute_soft_tokens(image_path);
                if (n > 0) {
                    result += "\n\n<|image>";
                    for (size_t j = 0; j < n; j++)
                        result += "<|image|>";
                    result += "<image|>\n\n";
                }
            }
            result += msg.content;
            if (msg.audio_soft_token_count > 0) {
                result += "<|audio>";
                for (size_t j = 0; j < msg.audio_soft_token_count; j++)
                    result += "<|audio|>";
                result += "<audio|>";
            }
        }
        result += "<turn|>\n";
    }

    if (add_generation_prompt) {
        result += "<|turn>model\n";
    }

    return result;
}

std::string Tokenizer::format_lfm2_vl_style(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    const std::string& tools_json) const
{
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are not supported for LFM2-VL models";
    }

    std::string result = "<|startoftext|>";
    
    for (const auto& msg : messages) {
        result += "<|im_start|>" + msg.role + "\n";
        for (const auto& image_path : msg.images) {
            int width = 0, height = 0, channels = 0;
            unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
            
            if (img_data) {
                Siglip2Preprocessor preprocessor;
                auto shape_result = preprocessor.compute_spatial_shapes(height, width);
                int downsample_factor = 2;
                bool use_thumbnail = true;
                int grid_rows = shape_result.grid_rows;
                int grid_cols = shape_result.grid_cols;
                int num_tiles = grid_rows * grid_cols;
                result += "<|image_start|>";
                
                if (num_tiles > 1) {
                    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                        int row = tile_idx / grid_cols;
                        int col = tile_idx % grid_cols;
                        
                        result += "<|img_row_" + std::to_string(row + 1) + "_col_" + std::to_string(col + 1) + "|>";
                        auto [tile_height, tile_width] = shape_result.shapes[tile_idx];
                        int tile_tokens = (tile_height * tile_width) / (downsample_factor * downsample_factor);
                        
                        for (int t = 0; t < tile_tokens; ++t) {
                            result += "<image>";
                        }
                    }
                    if (use_thumbnail && static_cast<size_t>(num_tiles) < shape_result.shapes.size()) {
                        result += "<|img_thumbnail|>";
                        
                        auto [thumb_height, thumb_width] = shape_result.shapes[num_tiles];
                        int thumbnail_tokens = (thumb_height * thumb_width) / (downsample_factor * downsample_factor);
                        
                        for (int t = 0; t < thumbnail_tokens; ++t) {
                            result += "<image>";
                        }
                    }
                } else if (num_tiles == 1) {
                    auto [thumb_height, thumb_width] = shape_result.shapes[0];
                    int thumbnail_tokens = (thumb_height * thumb_width) / (downsample_factor * downsample_factor);
                    
                    for (int t = 0; t < thumbnail_tokens; ++t) {
                        result += "<image>";
                    }
                }
                
                result += "<|image_end|>";
                
                stbi_image_free(img_data);
            } else {
                result += "<image>";
            }
        }
        result += msg.content;

        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}


std::string Tokenizer::format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    auto trim = [](const std::string& s) -> std::string {
        size_t start = 0;
        while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        size_t end = s.size();
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(start, end - start);
    };

    std::string result = "<bos>";

    std::string system_content;
    size_t start_idx = 0;

    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
        system_content = messages[0].content;
        start_idx = 1;
    }

    if (!tools_json.empty() || !system_content.empty()) {
        result += "<start_of_turn>developer\n";
        if (!system_content.empty()) {
            result += trim(system_content);
        }
        if (!tools_json.empty()) {
            result += tools_json;
        }
        result += "<end_of_turn>\n";
    }

    std::string prev_message_type;

    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];

        if (msg.role == "tool") {
            std::string func_name = msg.name.empty() ? "tool" : msg.name;
            result += "<start_function_response>response:" + func_name + "{" + msg.content + "}<end_function_response>";
            prev_message_type = "tool_response";
            
        } else if (msg.role == "user") {
            if (prev_message_type != "tool_response") {
                result += "<start_of_turn>user\n";
            }
            result += trim(msg.content);
            result += "<end_of_turn>\n";
            prev_message_type = "content";
            
        } else if (msg.role == "assistant" || msg.role == "model") {
            if (prev_message_type != "tool_response") {
                result += "<start_of_turn>model\n";
            }
            if (!msg.content.empty()) {
                result += trim(msg.content);
            }
            if (!msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    result += format_tool_call_for_prompt(tc.name, tc.arguments, false);
                }
                if (i == messages.size() - 1) {
                    result += "<start_function_response>";
                }
                prev_message_type = "tool_call";
            } else {
                result += "<end_of_turn>\n";
                prev_message_type = "content";
            }
        }
    }

    if (add_generation_prompt) {
        if (prev_message_type != "tool_response" && prev_message_type != "tool_call") {
            result += "<start_of_turn>model\n";
        }
    }

    return result;
}

std::string Tokenizer::format_youtu_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    std::string result = "<|begin_of_text|>";

    std::string system_block;
    for (const auto& msg : messages) {
        if (msg.role == "system") { system_block = msg.content; break; }
    }

    if (!tools_json.empty()) {
        std::string tool_desc =
            "<|begin_of_tool_description|>Tool calling capabilities.\n"
            "You may call one or more functions to assist with the user query. "
            "You have the following functions available:\n"
            "```json\n" + tools_json + "\n```\n"
            "For tool call returns, you MUST use the following format:\n"
            "<tool_call>{\"name\": \"function-name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}</tool_call>\n"
            "<|end_of_tool_description|>";
        system_block = system_block.empty() ? tool_desc : system_block + "\n\n" + tool_desc;
    }

    result += system_block;

    bool is_last_user = false;
    bool is_tool = false;
    bool is_output_first = true;

    for (const auto& msg : messages) {
        if (msg.role == "system") {
            continue;
        } else if (msg.role == "user") {
            is_last_user = true; is_tool = false;
            result += "<|User|>" + msg.content;
        } else if (msg.role == "tool") {
            is_tool = true; is_last_user = false;
            if (is_output_first) {
                result += "<|User|><tool_response>" + msg.content + "</tool_response>";
                is_output_first = false;
            } else {
                result += "\n<tool_response>" + msg.content + "</tool_response>";
            }
        } else if (msg.role == "assistant" || msg.role == "model") {
            is_last_user = false; is_tool = false; is_output_first = true;
            result += "<|Assistant|>" + msg.content;
            if (!msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    result += "<tool_call>{\"name\": \"" + tc.name
                        + "\", \"arguments\": " + tc.arguments + "}</tool_call>";
                }
            }
            result += "<|end_of_text|>";
        }
    }

    if (add_generation_prompt && (is_last_user || is_tool)) {
        result += "<|Assistant|>";
    }

    return result;
}

} // namespace engine
} // namespace cactus
