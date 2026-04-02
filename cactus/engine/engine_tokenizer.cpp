#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

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
    if (!file.is_open()) {
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    size_t pos = content.find("\"special_tokens\"");
    if (pos == std::string::npos) return;

    pos = content.find("{", pos);
    if (pos == std::string::npos) return;

    size_t end_pos = content.find("}", pos);
    if (end_pos == std::string::npos) return;

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

        uint32_t token_id = static_cast<uint32_t>(std::stoul(id_part.substr(id_start + 1, id_end - id_start - 1)));

        size_t token_start = token_part.find("\"");
        size_t token_end = token_part.rfind("\"");
        if (token_start == std::string::npos || token_end == std::string::npos || token_start >= token_end) continue;

        std::string token_content = token_part.substr(token_start + 1, token_end - token_start - 1);
        special_tokens[token_content] = token_id;
    }
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
        case ModelType::YOUTU:
            return format_youtu_style(messages, add_generation_prompt, tools_json);
        default:
            return format_qwen_style(messages, add_generation_prompt, tools_json);
    }
}

std::string Tokenizer::format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported) const {
    std::string result;

    if (!tools_json.empty()) {
        result += "<|im_start|>system\n";

        bool has_system_msg = false;
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                result += msg.content;
                result += "\n\n";
                has_system_msg = true;
                break;
            }
        }

        result += "# Tools\n\n";
        result += "You may call one or more functions to assist with the user query.\n\n";
        result += "You are provided with function signatures within <tools></tools> XML tags:\n";
        result += "<tools>\n";
        result += tools_json;
        result += "\n</tools>\n\n";
        result += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n";
        result += "<tool_call>\n";
        result += "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";
        result += "</tool_call>";
        result += "<|im_end|>\n";

        for (const auto& msg : messages) {
            if (msg.role == "system" && has_system_msg) {
                continue;
            } else if (msg.role == "user") {
                result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
    } else {
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                result += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "user") {
                result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
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
        }
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
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
        result += msg.content;
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
        
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}


std::string Tokenizer::format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
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
            result += system_content;
            if (!tools_json.empty()) {
                result += "\n"; 
            }
        }
        if (!tools_json.empty()) {
            result += "You are a model that can do function calling with the following functions";
            result += tools_json;
        }
        result += "<end_of_turn>\n";
    }

    std::string prev_message_type;

    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];

        if (msg.role == "tool") {
            if (prev_message_type != "tool_response") {
                result += "<start_of_turn>developer\n";
            }
            std::string func_name = msg.name.empty() ? "tool" : msg.name;
            result += "<start_function_response>response:" + func_name + "{" + msg.content + "}<end_function_response>";
            prev_message_type = "tool_response";
            
        } else if (msg.role == "user") {
            if (prev_message_type == "tool_response") {
                result += "<end_of_turn>\n";
            }
            result += "<start_of_turn>user\n";
            result += msg.content;
            result += "<end_of_turn>\n";
            prev_message_type = "content";
            
        } else if (msg.role == "assistant" || msg.role == "model") {
            if (prev_message_type == "tool_response") {
                result += "<end_of_turn>\n";
            }
            result += "<start_of_turn>model\n";
            result += msg.content;
            result += "<end_of_turn>\n";
            prev_message_type = "content";
        }
    }

    if (prev_message_type == "tool_response") {
        result += "<end_of_turn>\n";
    }

    if (add_generation_prompt) {
        if (prev_message_type != "tool_response") {
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
            result += "<|Assistant|>" + msg.content + "<|end_of_text|>";
        }
    }

    if (add_generation_prompt && (is_last_user || is_tool)) {
        result += "<|Assistant|>";
    }

    return result;
}

} // namespace engine
} // namespace cactus
