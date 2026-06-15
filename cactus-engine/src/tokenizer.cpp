#include "engine.h"
#include "cactus_kernels.h"
#include "gemma_tools.h"
#include "chat_tools.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>

namespace cactus {
namespace engine {

namespace {

std::string format_needle_query_text(const std::vector<ChatMessage>& messages) {
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

    if (user_query.empty() && !messages.empty()) user_query = messages.back().content;
    if (system_text.empty()) return user_query;
    if (user_query.empty()) return system_text;
    return system_text + "\n\n" + user_query;
}

std::string format_tool_call_for_prompt(const std::string& name, const std::string& arguments, bool gemma4) {
    std::string args = arguments.empty() ? "{}" : arguments;
    size_t pos = 0;
    std::string dsl = gemma::format_argument(args, pos, false);
    if (dsl.empty() || dsl.front() != '{') dsl = "{" + dsl + "}";
    if (gemma4) {
        return "<|tool_call>call:" + name + dsl + "<tool_call|>";
    }
    return "<start_function_call>call:" + name + gemma::use_escape_tags(dsl) + "<end_function_call>";
}

std::string format_tool_response_for_prompt(const std::string& name, const std::string& content, bool gemma4) {
    if (gemma4) {
        return "<|tool_response>response:" + name + "{value:<|\"|>" + content + "<|\"|>}<tool_response|>";
    }
    return "<start_function_response>response:" + name + "{value:<escape>" + content + "<escape>}<end_function_response>";
}

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
            if (special_token.empty()) continue;
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
    model_type_ = ModelType::GEMMA4;

    std::ifstream file(config_path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

        if (lower_line.find("model_type") != std::string::npos) {
            if (lower_line.find("needle") != std::string::npos) {
                model_type_ = ModelType::NEEDLE;
            } else if (lower_line.find("qwen") != std::string::npos) {
                model_type_ = ModelType::QWEN;
            } else if (lower_line.find("lfm2") != std::string::npos) {
                model_type_ = ModelType::LFM2;
            } else if (lower_line.find("gemma4") != std::string::npos) {
                model_type_ = ModelType::GEMMA4;
            } else if (lower_line.find("gemma") != std::string::npos) {
                model_type_ = ModelType::GEMMA;
            }
        } else if (lower_line.find("model_variant") != std::string::npos) {
            if (lower_line.find("vlm") != std::string::npos) { model_variant_ = ModelVariant::VLM; }
            else if (lower_line.find("extract") != std::string::npos) { model_variant_ = ModelVariant::EXTRACT; }
            else if (lower_line.find("rag") != std::string::npos) { model_variant_ = ModelVariant::RAG; }
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
}

std::string Tokenizer::get_default_stop_sequence() const {
    if (model_type_ == ModelType::NEEDLE) {
        return "";
    }
    if (model_type_ == ModelType::QWEN || model_type_ == ModelType::LFM2) {
        return "<|im_end|>";
    }
    if (model_type_ == ModelType::GEMMA) {
        return "<end_of_turn>";
    }
    return "<turn|>";
}

std::vector<uint32_t> Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    return encode(format_chat_prompt(messages, add_generation_prompt));
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                          const std::string& tools_json, bool enable_thinking_if_supported) const {
    if (model_type_ == ModelType::QWEN) {
        return format_qwen_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
    }
    if (model_type_ == ModelType::LFM2) {
        return format_lfm2_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
    }
    if (model_type_ == ModelType::NEEDLE) {
        return format_needle_style(messages, add_generation_prompt, tools_json);
    }
    if (model_type_ == ModelType::GEMMA) {
        return format_gemma_style(messages, add_generation_prompt, tools_json);
    }
    return format_gemma4_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
}

std::string Tokenizer::format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                          const std::string& tools_json) const {
    std::string result = "<bos>";
    if (tools_json.empty() && !has_function_call_tokens()) {
        std::string pending_context;
        for (const auto& msg : messages) {
            if (msg.role == "system" || msg.role == "developer") {
                pending_context += msg.content + "\n\n";
                continue;
            }
            std::string role = msg.role == "assistant" ? "model" : "user";
            std::string content = msg.content;
            for (const auto& tc : msg.tool_calls) {
                content += "\ncall:" + tc.name + "(" + tc.arguments + ")";
            }
            if (role == "user" && !pending_context.empty()) {
                content = pending_context + content;
                pending_context.clear();
            }
            result += "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
        }
        if (add_generation_prompt) {
            result += "<start_of_turn>model\n";
        }
        return result;
    }

    size_t first = 0;
    std::string sys;
    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
        sys = messages[0].content;
        first = 1;
    }
    if (first > 0 || !tools_json.empty()) {
        result += "<start_of_turn>developer\n" + sys + tools_json + "<end_of_turn>\n";
    }
    bool after_tool_response = false;
    for (size_t i = first; i < messages.size(); i++) {
        const auto& msg = messages[i];
        if (msg.role == "tool") {
            result += format_tool_response_for_prompt(msg.name, msg.content, false);
            after_tool_response = true;
            continue;
        }
        std::string role = msg.role == "assistant" ? "model" : msg.role;
        if (!after_tool_response) {
            result += "<start_of_turn>" + role + "\n";
        }
        after_tool_response = false;
        result += msg.content;
        for (const auto& tc : msg.tool_calls) {
            result += format_tool_call_for_prompt(tc.name, tc.arguments, false);
        }
        if (msg.tool_calls.empty()) {
            result += "<end_of_turn>\n";
        } else if (i + 1 == messages.size()) {
            result += "<start_function_response>";
        }
    }
    if (add_generation_prompt && !after_tool_response) {
        result += "<start_of_turn>model\n";
    }
    return result;
}

namespace {
std::string strip_newlines(const std::string& s) {
    size_t a = s.find_first_not_of('\n');
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of('\n');
    return s.substr(a, b - a + 1);
}
std::string lstrip_newlines(const std::string& s) {
    size_t a = s.find_first_not_of('\n');
    return a == std::string::npos ? "" : s.substr(a);
}
}  // namespace

std::string Tokenizer::format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                         const std::string& tools_json, bool enable_thinking_if_supported) const {
    std::string result;
    const size_t n = messages.size();
    const bool template_has_thinking = !has_chat_template_ || chat_template_.find("<think>") != std::string::npos;

    size_t first = 0;
    std::string sys;
    const bool has_sys = n > 0 && (messages[0].role == "system" || messages[0].role == "developer");
    if (has_sys) { sys = messages[0].content; first = 1; }
    if (!tools_json.empty()) {
        result += "<|im_start|>system\n";
        if (has_sys) result += sys + "\n\n";
        result += tools_json;
        result += "<|im_end|>\n";
    } else if (has_sys) {
        result += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }

    long last_query_index = static_cast<long>(n) - 1;
    for (long i = static_cast<long>(n) - 1; i >= 0; --i) {
        if (messages[i].role == "user") { last_query_index = i; break; }
    }

    for (size_t i = first; i < n; i++) {
        const auto& msg = messages[i];
        std::string role = msg.role;
        if (role == "developer") role = "system";
        else if (role != "system" && role != "assistant" && role != "tool") role = "user";

        if (role == "user" || role == "system") {
            result += "<|im_start|>" + role + "\n";
            if (role == "user") {
                const size_t soft_n = image_soft_token_count_ > 0 ? image_soft_token_count_ : 1;
                for (const auto& image_path : msg.images) {
                    (void)image_path;
                    result += "<|vision_start|>";
                    for (size_t k = 0; k < soft_n; ++k) result += "<|image_pad|>";
                    result += "<|vision_end|>";
                }
            }
            result += msg.content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string content = msg.content;
            std::string reasoning;
            size_t tpos = content.find("</think>");
            if (tpos != std::string::npos) {
                std::string head = content.substr(0, tpos);
                size_t ts = head.rfind("<think>");
                reasoning = strip_newlines(ts != std::string::npos ? head.substr(ts + 7) : head);
                content = lstrip_newlines(content.substr(tpos + 8));
            }
            result += "<|im_start|>assistant\n";
            if (template_has_thinking && static_cast<long>(i) > last_query_index && (i == n - 1 || !reasoning.empty())) {
                result += "<think>\n" + reasoning + "\n</think>\n\n" + lstrip_newlines(content);
            } else {
                result += content;
            }
            for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
                const auto& tc = msg.tool_calls[k];
                if ((k == 0 && !content.empty()) || k > 0) result += "\n";
                result += "<tool_call>\n{\"name\": \"" + tc.name + "\", \"arguments\": " +
                          chat_tools::respace_json(tc.arguments.empty() ? "{}" : tc.arguments) + "}\n</tool_call>";
            }
            result += "<|im_end|>\n";
        } else {  // tool
            if (i == 0 || messages[i - 1].role != "tool") result += "<|im_start|>user";
            result += "\n<tool_response>\n" + msg.content + "\n</tool_response>";
            if (i + 1 >= n || messages[i + 1].role != "tool") result += "<|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
        if (!enable_thinking_if_supported && template_has_thinking) result += "<think>\n\n</think>\n\n";
    }
    return result;
}

std::string Tokenizer::format_lfm2_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                         const std::string& tools_json, bool /*enable_thinking_if_supported*/) const {
    std::string result = "<|startoftext|>";
    const size_t n = messages.size();

    size_t first = 0;
    std::string sys;
    const bool has_sys = n > 0 && (messages[0].role == "system" || messages[0].role == "developer");
    if (has_sys) { sys = messages[0].content; first = 1; }
    if (!tools_json.empty() || has_sys) {
        result += "<|im_start|>system\n";
        if (has_sys) result += sys;
        if (!tools_json.empty()) { if (has_sys) result += "\n"; result += tools_json; }
        result += "<|im_end|>\n";
    }

    for (size_t i = first; i < n; i++) {
        const auto& msg = messages[i];
        std::string role = msg.role;
        if (role == "developer") role = "system";
        else if (role != "system" && role != "assistant" && role != "tool") role = "user";

        result += "<|im_start|>" + role + "\n";
        if (role == "user") {
            for (const auto& image_path : msg.images) {
                int iw = 0, ih = 0, ic = 0;
                Lfm2VlTokenLayout layout;
                if (has_lfm2_vision_config_ && cactus_image_info(image_path.c_str(), &iw, &ih, &ic)) {
                    layout = lfm2_vl_token_layout(ih, iw, lfm2_vision_config_);
                } else {
                    layout.grid_rows = 1;
                    layout.grid_cols = 1;
                    layout.tokens_per_tile = image_soft_token_count_ > 0 ? static_cast<int>(image_soft_token_count_) : 1;
                }
                result += "<|image_start|>";
                const bool multi_tile = layout.grid_rows > 1 || layout.grid_cols > 1;
                if (multi_tile) {
                    for (int r = 0; r < layout.grid_rows; ++r) {
                        for (int c = 0; c < layout.grid_cols; ++c) {
                            result += "<|img_row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + "|>";
                            for (int k = 0; k < layout.tokens_per_tile; ++k) result += "<image>";
                        }
                    }
                    if (layout.has_thumbnail) {
                        result += "<|img_thumbnail|>";
                        for (int k = 0; k < layout.thumbnail_tokens; ++k) result += "<image>";
                    }
                } else {
                    for (int k = 0; k < layout.tokens_per_tile; ++k) result += "<image>";
                }
                result += "<|image_end|>\n";
            }
        }
        if (role == "tool") {
            result += "<|tool_response_start|>" + msg.content + "<|tool_response_end|>";
        } else {
            result += msg.content;
        }
        if (role == "assistant" && !msg.tool_calls.empty()) {
            result += "<|tool_call_start|>[";
            for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
                if (k) result += ", ";
                result += chat_tools::pythonic_call(msg.tool_calls[k].name, msg.tool_calls[k].arguments);
            }
            result += "]<|tool_call_end|>";
        }
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) result += "<|im_start|>assistant\n";
    return result;
}

std::string Tokenizer::format_needle_style(const std::vector<ChatMessage>& messages, bool /*add_generation_prompt*/,
                                           const std::string& tools_json) const {
    std::string serialized_tools = tools_json.empty() ? "[]" : tools_json;
    return format_needle_query_text(messages) + "<tools>" + serialized_tools + "</s>";
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
            result += "<|think|>\n";
        }
        result += sys_content;
        result += tools_json;
        result += "<turn|>\n";
    }

    auto compute_soft_tokens = [&](const std::string& image_path) -> size_t {
        if (image_soft_token_count_ > 0) return image_soft_token_count_;

        int w = 0, h = 0, c = 0;
        if (!cactus_image_info(image_path.c_str(), &w, &h, &c)) return 0;

        uint32_t p = vision_patch_size_ ? vision_patch_size_ : 16;
        uint32_t k = vision_pooling_kernel_size_ ? vision_pooling_kernel_size_ : 3;
        uint32_t out_len = vision_default_output_length_ ? vision_default_output_length_ : 280;
        uint32_t side = k * p;
        uint32_t max_patches = out_len * k * k;
        float factor = std::sqrt(static_cast<float>(max_patches) * p * p /
                                 (static_cast<float>(h) * w));
        int th = static_cast<int>(std::floor(factor * h / side)) * side;
        int tw = static_cast<int>(std::floor(factor * w / side)) * side;
        if (th == 0) th = side;
        if (tw == 0) tw = side;
        return static_cast<size_t>((th / p / k) * (tw / p / k));
    };
    bool in_model_turn = false;
    std::vector<std::string> pending_call_names;
    auto close_model_turn = [&]() {
        if (in_model_turn) { result += "<turn|>\n"; in_model_turn = false; }
    };

    for (size_t i = first_msg; i < messages.size(); i++) {
        const auto& msg = messages[i];
        const std::string role = (msg.role == "assistant") ? "model"
                               : (msg.role == "developer") ? "system" : msg.role;

        if (role == "model") {
            if (!in_model_turn) { result += "<|turn>model\n"; in_model_turn = true; }
            result += msg.content;
            for (const auto& tc : msg.tool_calls) {
                result += format_tool_call_for_prompt(tc.name, tc.arguments, true);
                pending_call_names.push_back(tc.name);
            }
            if (msg.tool_calls.empty()) close_model_turn();
        } else if (role == "tool") {
            if (!in_model_turn) { result += "<|turn>model\n"; in_model_turn = true; }
            std::string fn = !msg.name.empty() ? msg.name
                           : (!pending_call_names.empty() ? pending_call_names.front()
                                                          : std::string("unknown"));
            if (!pending_call_names.empty()) pending_call_names.erase(pending_call_names.begin());
            result += format_tool_response_for_prompt(fn, msg.content, true);
        } else {
            close_model_turn();
            result += "<|turn>" + role + "\n";
            for (const auto& image_path : msg.images) {
                size_t n = compute_soft_tokens(image_path);
                if (n > 0) {
                    result += "<|image>";
                    for (size_t j = 0; j < n; j++)
                        result += "<|image|>";
                    result += "<image|>";
                }
            }
            result += msg.content;
            if (msg.audio_soft_token_count > 0) {
                result += "<|audio>";
                for (size_t j = 0; j < msg.audio_soft_token_count; j++)
                    result += "<|audio|>";
                result += "<audio|>";
            }
            result += "<turn|>\n";
        }
    }

    if (add_generation_prompt) {
        if (!in_model_turn) result += "<|turn>model\n";
    } else {
        close_model_turn();
    }

    return result;
}

} // namespace engine
} // namespace cactus
