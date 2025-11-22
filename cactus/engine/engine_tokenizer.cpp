#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>

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

        result += "You have access to the following tools:\n";
        result += "[\n";
        result += tools_json;
        result += "\n]\n\n";
        result += "When you need to call a tool, respond with a JSON object in this exact format:\n";
        result += "{\"function_call\": {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}}";
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
            result += "<|im_start|>assistant\n</think>\n\n";
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

    if (model_variant_ == ModelVariant::RAG) {
        const std::string docs_path = !corpus_dir_.empty() ? corpus_dir_ : std::string("../../tests/assets/rag_corpus");
        DIR* dir = opendir(docs_path.c_str());
        if (dir) {
            struct dirent* entry;
            std::vector<std::string> files;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name.size() > 4) {
                    std::string lower = name;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".txt") {
                        files.push_back(name);
                    }
                }
            }
            closedir(dir);

            std::sort(files.begin(), files.end());

            int doc_idx = 1;
            for (const auto& fname : files) {
                std::string full = docs_path + "/" + fname;
                std::ifstream infile(full);
                if (!infile.is_open()) continue;
                std::stringstream ss; ss << infile.rdbuf();
                std::string file_text = ss.str();
                for (size_t i = 0; i < file_text.size(); ++i) {
                    if (file_text[i] == '\0') file_text[i] = ' ';
                }
                if (!sys_content.empty()) {
                    sys_content += "\n";
                }
                sys_content += "The following documents may provide you additional information to answer questions: ";
                sys_content += "<document" + std::to_string(doc_idx) + ">";
                sys_content += file_text;
                sys_content += "</document" + std::to_string(doc_idx) + ">";
                doc_idx++;
            }
        }
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

    if (!tools_json.empty()) {
        return "ERROR: Tool calls are not supported for Gemma models";
    }

    std::string result;

    result = "<bos>";

    std::string first_user_prefix = "";
    size_t start_idx = 0;

    if (!messages.empty() && messages[0].role == "system") {
        first_user_prefix = messages[0].content + "\n\n";
        start_idx = 1;
    }

    bool first_user = true;
    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];

        if (msg.role == "user") {
            result += "<start_of_turn>user";
            result += "\n";
            if (first_user && !first_user_prefix.empty()) {
                result += first_user_prefix;
                first_user = false;
            }
            result += msg.content;
            result += "<end_of_turn>";
            result += "\n";
        } else if (msg.role == "assistant") {
            result += "<start_of_turn>model";
            result += "\n";
            result += msg.content;
            result += "<end_of_turn>";
            result += "\n";
        }
    }

    if (add_generation_prompt) {
        result += "<start_of_turn>model";
        result += "\n";
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