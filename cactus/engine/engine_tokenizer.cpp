#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace cactus {
namespace engine {

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

            if (line.find("qwen") != std::string::npos) {
                model_type_ = ModelType::QWEN;
                break;
            } else if (line.find("gemma") != std::string::npos) {
                model_type_ = ModelType::GEMMA;
                break;
            } else if(line.find("lfm2") != std::string::npos) {
                model_type_ = ModelType::LFM2;
            } else if (line.find("smol") != std::string::npos) {
                model_type_ = ModelType::SMOL;
                break;
            } else if (line.find("bert") != std::string::npos) {
                model_type_ = ModelType::BERT;
                break;
            } else if (line.find("whisper") != std::string::npos) {
                model_type_ = ModelType::WHISPER;
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
    file.close();
}

std::string Tokenizer::get_default_stop_sequence() const {
    switch (model_type_) {
        case ModelType::GEMMA:
            return "<end_of_turn>";
        case ModelType::QWEN:
        case ModelType::LFM2:
        case ModelType::SMOL:
            return "<|im_end|>";
        default:
            return "<|im_end|>";
    }
}

std::vector<uint32_t> Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    std::string formatted_prompt = format_chat_prompt(messages, add_generation_prompt);
    return encode(formatted_prompt);
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
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
            return format_qwen_style(messages, add_generation_prompt, tools_json);
        case ModelType::GEMMA:
            return format_gemma_style(messages, add_generation_prompt, tools_json);
        case ModelType::LFM2:
            return format_lfm2_style(messages, add_generation_prompt, tools_json);
        case ModelType::SMOL:
            return format_smol_style(messages, add_generation_prompt, tools_json);
        default:
            return format_qwen_style(messages, add_generation_prompt, tools_json);
    }
}

std::string Tokenizer::format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
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
        if (!tools_json.empty()) {
            result += "<|im_start|>assistant\n<think>\n</think>\n\n";
        } else {
            result += "<|im_start|>assistant\n";
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
            result += "You are a model that can do function calling with the following functions.";
            result += tools_json;
            result += "\n\nWhen you decide to call a function, output it in this exact format:\n";
            result += "<start_function_call>call:function_name{arg1:<escape>value1<escape>,arg2:<escape>value2<escape>}<end_function_call>";
        }
        result += "<end_of_turn>\n";
    }

    std::string prev_message_type;

    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];

        if (msg.role == "tool") {
            std::string func_name = msg.name.empty() ? "tool" : msg.name;
            result += "<start_function_response>response:" + func_name + "{value:<escape>" + msg.content + "<escape>}<end_function_response>";
            prev_message_type = "tool_response";
        } else if (msg.role == "user") {
            if (prev_message_type != "tool_response") {
                result += "<start_of_turn>user\n";
            }
            result += msg.content;
            result += "<end_of_turn>\n";
            prev_message_type = "content";
        } else if (msg.role == "assistant" || msg.role == "model") {
            if (prev_message_type != "tool_response") {
                result += "<start_of_turn>model\n";
            }
            result += msg.content;
            result += "<end_of_turn>\n";
            prev_message_type = "content";
        }
    }

    if (add_generation_prompt) {
        if (prev_message_type != "tool_response") {
            result += "<start_of_turn>model\n";
        }
    }

    return result;
}

std::string Tokenizer::format_smol_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are currently not supported for Smol models";
    }

    std::string result;

    if (!messages.empty() && messages.front().role != "system") {
        result += "<|im_start|>system\n";
        result += "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
        result += "<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        result += "<|im_start|>";
        result += msg.role;
        result += "\n";
        result += msg.content;
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}


} // namespace engine
} // namespace cactus
