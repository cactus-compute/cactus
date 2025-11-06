#include "engine.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

namespace cactus {
namespace engine {

SPTokenizer::SPTokenizer()
    : trie_root_(std::make_unique<TrieNode>()),
      vocab_size_(0),
      unk_token_id_(3),
      bos_token_id_(2),
      eos_token_id_(1),
      pad_token_id_(0),
      vocab_mmap_ptr_(nullptr),
      vocab_mmap_size_(0) {
    has_chat_template_ = false;
}

SPTokenizer::~SPTokenizer() {
    cleanup_mmap();
}

void SPTokenizer::cleanup_mmap() {
    if (vocab_mmap_ptr_ && vocab_mmap_ptr_ != MAP_FAILED) {
        munmap(vocab_mmap_ptr_, vocab_mmap_size_);
        vocab_mmap_ptr_ = nullptr;
    }
}

bool SPTokenizer::load_vocabulary_with_config(const std::string& vocab_file, const std::string& /*merges_file*/, const std::string& config_file) {
    std::string config_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/config.txt";
    detect_model_type(config_path);
    
    std::ifstream vocab_stream(vocab_file);
    if (!vocab_stream.is_open()) return false;

    token_to_id_.clear();
    id_to_token_.clear();
    token_scores_.clear();

    std::string first_line;
    std::getline(vocab_stream, first_line);
    vocab_stream.seekg(0);  

    bool is_id_token_format = false;
    if (!first_line.empty()) {
        is_id_token_format = (std::isdigit(first_line[0]) &&
                              first_line.find('\t') != std::string::npos);
    }

    if (is_id_token_format) {
        std::string line = "";
        while (std::getline(vocab_stream, line)) {
            std::string token = "";
            uint32_t id = UINT32_MAX;

            std::istringstream iss(line);
            if (iss >> id) {
                if (std::getline(iss, token)) {
                    if (!token.empty() && token[0] == '\t') {
                        token = token.substr(1);
                    }
                }
                
                if (token.empty()) {
                    auto last_pos = vocab_stream.tellg();
                    while (std::getline(vocab_stream, line)) {
                        if (!line.empty()) {
                            break;
                        }
                        token += '\n';
                        last_pos = vocab_stream.tellg();
                    }
                    vocab_stream.seekg(last_pos);
                }
            }
            
            if (!token.empty() && id != UINT32_MAX) {
                token_to_id_[token] = id;
                if (id >= id_to_token_.size()) {
                    id_to_token_.resize(id + 1);
                    token_scores_.resize(id + 1, 0.0f);
                }
                id_to_token_[id] = token;
                token_scores_[id] = 0.0f;
            }
        }
        vocab_size_ = id_to_token_.size();
    } else {
        std::string line;
        uint32_t id = 0;

        vocab_stream.seekg(0); 
        while (std::getline(vocab_stream, line)) {
            token_to_id_[line] = id;
            id_to_token_.push_back(line);
            token_scores_.push_back(0.0f);
            id++;
        }
        vocab_size_ = id;
    }

    vocab_stream.close();
    
    build_trie();
    
    std::ifstream config_stream(config_file);
    if (config_stream.is_open()) {
        std::string config_line;
        while (std::getline(config_stream, config_line)) {
            if (config_line.empty() || config_line[0] == '#') continue;
            
            size_t eq_pos = config_line.find('=');
            if (eq_pos == std::string::npos) continue;
            
            std::string key = config_line.substr(0, eq_pos);
            std::string value = config_line.substr(eq_pos + 1);
            
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "eos_token_id") {
                eos_token_id_ = std::stoul(value);
            } else if (key == "pad_token_id") {
                pad_token_id_ = std::stoul(value);
            } else if (key == "unk_token_id") {
                unk_token_id_ = std::stoul(value);
            } else if (key == "bos_token_id") {
                bos_token_id_ = std::stoul(value);
            }
        }
    }
    
    std::string special_tokens_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/special_tokens.json";
    load_special_tokens(special_tokens_path);

    std::string template_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/chat_template.jinja2";
    load_chat_template(template_path);

    return true;
}

void SPTokenizer::build_trie() {
    for (uint32_t id = 0; id < id_to_token_.size(); ++id) {
        const std::string& token = id_to_token_[id];
        if (token.empty()) continue;
        
        std::u32string u32_token;
        size_t pos = 0;
        while (pos < token.length()) {
            char32_t codepoint = 0;
            unsigned char byte = token[pos];

            if (byte < 0x80) {
                codepoint = byte;
                pos++;
            } else if ((byte & 0xE0) == 0xC0) {
                if (pos + 1 < token.length()) {
                    codepoint = ((byte & 0x1F) << 6) | (token[pos + 1] & 0x3F);
                    pos += 2;
                } else break;
            } else if ((byte & 0xF0) == 0xE0) {
                if (pos + 2 < token.length()) {
                    codepoint = ((byte & 0x0F) << 12) |
                               ((token[pos + 1] & 0x3F) << 6) |
                               (token[pos + 2] & 0x3F);
                    pos += 3;
                } else break;
            } else if ((byte & 0xF8) == 0xF0) {
                if (pos + 3 < token.length()) {
                    codepoint = ((byte & 0x07) << 18) |
                               ((token[pos + 1] & 0x3F) << 12) |
                               ((token[pos + 2] & 0x3F) << 6) |
                               (token[pos + 3] & 0x3F);
                    pos += 4;
                } else break;
            } else {
                pos++;
                continue;
            }

            u32_token.push_back(codepoint);
        }

        if (u32_token.empty()) continue;
        
        TrieNode* current = trie_root_.get();
        for (char32_t ch : u32_token) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = std::make_unique<TrieNode>();
            }
            current = current->children[ch].get();
        }
        current->token_id = static_cast<int32_t>(id);
        current->score = token_scores_[id];
    }
}

std::string SPTokenizer::preprocess_text(const std::string& text) const {
    if (text.empty()) return text;

    std::string processed = "";
    if (model_type_ == ModelType::BERT) {
        processed = "▁";
    }

    for (size_t i = text.find_first_not_of(" "); i < text.length(); i++) {
        char c = text[i];
        if (c == ' ') {
            processed += "▁";
        } else {
            processed += c;
        }
    }

    return processed;
}

std::string SPTokenizer::postprocess_text(const std::string& text) const {
    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        if (i + 2 < text.length() && 
            static_cast<unsigned char>(text[i]) == 0xE2 &&
            static_cast<unsigned char>(text[i+1]) == 0x96 &&
            static_cast<unsigned char>(text[i+2]) == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += text[i];
            i++;
        }
    }
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    return result;
}

std::vector<std::pair<std::string, uint32_t>> SPTokenizer::tokenize_with_trie(const std::string& text) const {
    std::vector<std::pair<std::string, uint32_t>> result;
    
    std::u32string u32_text;
    size_t pos = 0;
    while (pos < text.length()) {
        char32_t codepoint = 0;
        unsigned char byte = text[pos];

        if (byte < 0x80) {
            codepoint = byte;
            pos++;
        } else if ((byte & 0xE0) == 0xC0) {
            if (pos + 1 < text.length()) {
                codepoint = ((byte & 0x1F) << 6) | (text[pos + 1] & 0x3F);
                pos += 2;
            } else break;
        } else if ((byte & 0xF0) == 0xE0) {
            if (pos + 2 < text.length()) {
                codepoint = ((byte & 0x0F) << 12) |
                           ((text[pos + 1] & 0x3F) << 6) |
                           (text[pos + 2] & 0x3F);
                pos += 3;
            } else break;
        } else if ((byte & 0xF8) == 0xF0) {
            if (pos + 3 < text.length()) {
                codepoint = ((byte & 0x07) << 18) |
                           ((text[pos + 1] & 0x3F) << 12) |
                           ((text[pos + 2] & 0x3F) << 6) |
                           (text[pos + 3] & 0x3F);
                pos += 4;
            } else break;
        } else {
            pos++;
            continue;
        }

        u32_text.push_back(codepoint);
    }

    if (u32_text.empty()) {
        result.push_back({text, unk_token_id_});
        return result;
    }

    pos = 0;
    while (pos < u32_text.length()) {
        TrieNode* current = trie_root_.get();
        size_t best_match_len = 0;
        int32_t best_token_id = -1;
        
        for (size_t len = 0; pos + len < u32_text.length(); ++len) {
            char32_t ch = u32_text[pos + len];
            if (current->children.find(ch) == current->children.end()) {
                break;
            }
            current = current->children[ch].get();
            if (current->token_id >= 0) {
                best_match_len = len + 1;
                best_token_id = current->token_id;
            }
        }
        
        if (best_match_len > 0) {
            std::u32string u32_token = u32_text.substr(pos, best_match_len);
            
            std::string token;
            for (char32_t cp : u32_token) {
                if (cp < 0x80) {
                    token.push_back(static_cast<char>(cp));
                } else if (cp < 0x800) {
                    token.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    token.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    token.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    token.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    token.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    token.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                    token.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                    token.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    token.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            }
            result.push_back({token, static_cast<uint32_t>(best_token_id)});
            pos += best_match_len;
        } else {
            char32_t cp = u32_text[pos];
            std::string char_str;
            if (cp < 0x80) {
                char_str.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                char_str.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                char_str.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                char_str.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                char_str.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                char_str.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                char_str.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                char_str.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                char_str.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                char_str.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            result.push_back({char_str, unk_token_id_});
            pos++;
        }
    }
    
    return result;
}

std::vector<std::string> SPTokenizer::split_with_special_tokens(const std::string& text) const {
    std::vector<std::string> result;
    
    size_t start = 0;
    while (start < text.size()) {
        size_t best_match_pos = text.size();
        size_t best_match_len = 0;
        std::string best_special_token;
        
        for (const auto& [special_token, token_id] : special_tokens_) {
            size_t pos = text.find(special_token, start);
            if (pos != std::string::npos && pos < best_match_pos) {
                best_match_pos = pos;
                best_match_len = special_token.length();
                best_special_token = special_token;
            }
        }
        
        if (best_match_pos < text.size()) {
            if (best_match_pos > start) {
                std::string before = text.substr(start, best_match_pos - start);
                result.push_back(before);
            }
            result.push_back(best_special_token);
            start = best_match_pos + best_match_len;
        } else {
            if (start < text.size()) {
                result.push_back(text.substr(start));
            }
            break;
        }
    }
    
    return result;
}

std::vector<uint32_t> SPTokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    auto text_segments = split_with_special_tokens(text);
    std::vector<uint32_t> token_ids;

    for (const auto& segment : text_segments) {
        auto special_it = special_tokens_.find(segment);
        if (special_it != special_tokens_.end()) {
            token_ids.push_back(special_it->second);
        } else {
            std::string processed = preprocess_text(segment);

            auto token_pairs = tokenize_with_trie(processed);

            for (const auto& [token, id] : token_pairs) {
                token_ids.push_back(id);
            }
        }
    }

    return token_ids;
}

std::string SPTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;

    if (tokens.size() == 1) {
        uint32_t token_id = tokens[0];
        if (token_id < id_to_token_.size()) {
            std::string token = id_to_token_[token_id];

            size_t pos = 0;
            while (pos < token.length()) {
                if (pos + 2 < token.length() &&
                    static_cast<unsigned char>(token[pos]) == 0xE2 &&
                    static_cast<unsigned char>(token[pos+1]) == 0x96 &&
                    static_cast<unsigned char>(token[pos+2]) == 0x81) {
                    result += ' ';
                    pos += 3;
                } else {
                    result += token[pos];
                    pos++;
                }
            }
        }
        return result;
    }

    for (size_t i = 0; i < tokens.size(); i++) {
        uint32_t token_id = tokens[i];
        if (token_id < id_to_token_.size()) {
            std::string token = id_to_token_[token_id];

            size_t pos = 0;
            while (pos < token.length()) {
                if (pos + 2 < token.length() &&
                    static_cast<unsigned char>(token[pos]) == 0xE2 &&
                    static_cast<unsigned char>(token[pos+1]) == 0x96 &&
                    static_cast<unsigned char>(token[pos+2]) == 0x81) {
                    if (!result.empty()) {
                        result += ' ';
                    }
                    pos += 3;
                } else {
                    result += token[pos];
                    pos++;
                }
            }
        }
    }

    return result;
}

void SPTokenizer::load_special_tokens(const std::string& config_file) {
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
        
        std::string id_str = id_part.substr(id_start + 1, id_end - id_start - 1);
        uint32_t token_id = std::stoul(id_str);
        
        size_t token_start = token_part.find("\"");
        size_t token_end = token_part.rfind("\"");
        if (token_start == std::string::npos || token_end == std::string::npos || token_start >= token_end) continue;
        
        std::string token_content = token_part.substr(token_start + 1, token_end - token_start - 1);
        
        special_tokens_[token_content] = token_id;
    }
}

void SPTokenizer::load_chat_template(const std::string& template_file) {
    std::ifstream file(template_file);
    if (!file.is_open()) {
        has_chat_template_ = false;
        return;
    }
    
    chat_template_ = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    has_chat_template_ = !chat_template_.empty();
}


} // namespace engine
} // namespace cactus