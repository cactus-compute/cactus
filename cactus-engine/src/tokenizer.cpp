#include "engine.h"
#include "cactus_kernels.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>

namespace cactus {
namespace engine {

namespace {

std::string format_tool_call_for_prompt(const std::string& name, const std::string& arguments, bool gemma4) {
    if (gemma4) {
        return "\n<|tool_call>\ncall:" + name + "(" + arguments + ")\n<tool_call|>\n";
    }
    return "\ncall:" + name + "(" + arguments + ")\n";
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
        size_t pos = line.find("model_variant");
        if (pos != std::string::npos) {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            if (line.find("vlm") != std::string::npos) { model_variant_ = ModelVariant::VLM; break; }
            else if (line.find("extract") != std::string::npos) { model_variant_ = ModelVariant::EXTRACT; break; }
            else if (line.find("rag") != std::string::npos) { model_variant_ = ModelVariant::RAG; break; }
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
    return "<turn|>";
}

std::vector<uint32_t> Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    return encode(format_chat_prompt(messages, add_generation_prompt));
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                          const std::string& tools_json, bool enable_thinking_if_supported) const {
    return format_gemma4_style(messages, add_generation_prompt, tools_json, enable_thinking_if_supported);
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
        unsigned char* data = cactus_image_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) return 0;
        cactus_image_free(data);

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

} // namespace engine
} // namespace cactus
