#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

namespace cactus {
namespace engine {

BPETokenizer::BPETokenizer()
    : vocab_size_(0), unk_token_id_(0), bos_token_id_(1), eos_token_id_(2),
      vocab_mmap_ptr_(nullptr), vocab_mmap_size_(0),
      merges_mmap_ptr_(nullptr), merges_mmap_size_(0) {
    has_chat_template_ = false;
}

BPETokenizer::~BPETokenizer() {
    cleanup_mmap();
}

void BPETokenizer::cleanup_mmap() {
    if (vocab_mmap_ptr_ && vocab_mmap_ptr_ != MAP_FAILED) {
        munmap(vocab_mmap_ptr_, vocab_mmap_size_);
        vocab_mmap_ptr_ = nullptr;
    }
    if (merges_mmap_ptr_ && merges_mmap_ptr_ != MAP_FAILED) {
        munmap(merges_mmap_ptr_, merges_mmap_size_);
        merges_mmap_ptr_ = nullptr;
    }
}

bool BPETokenizer::load_vocabulary_mmap(const std::string& vocab_file, const std::string& merges_file) {
    int vocab_fd = open(vocab_file.c_str(), O_RDONLY);
    if (vocab_fd == -1) return false;

    struct stat vocab_stat;
    if (fstat(vocab_fd, &vocab_stat) == -1) {
        close(vocab_fd);
        return false;
    }

    vocab_mmap_size_ = vocab_stat.st_size;
    vocab_mmap_ptr_ = mmap(nullptr, vocab_mmap_size_, PROT_READ, MAP_PRIVATE, vocab_fd, 0);
    close(vocab_fd);

    if (vocab_mmap_ptr_ == MAP_FAILED) return false;

    std::string vocab_content(static_cast<char*>(vocab_mmap_ptr_), vocab_mmap_size_);
    std::istringstream vocab_stream(vocab_content);

    auto rtrim_cr = [](std::string& s) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
    };

    std::string line;
    uint32_t id = 0;
    token_to_id_.clear();
    id_to_token_.clear();
    special_tokens_.clear();

    while (std::getline(vocab_stream, line)) {
        rtrim_cr(line);
        if (line.empty()) continue;
        token_to_id_[line] = id;
        id_to_token_.push_back(line);

        if (!line.empty() && line.front() == '<' && line.back() == '>') {
            special_tokens_[line] = id;
        }
        id++;
    }
    vocab_size_ = id;

    int merges_fd = open(merges_file.c_str(), O_RDONLY);
    if (merges_fd == -1) return false;

    struct stat merges_stat;
    if (fstat(merges_fd, &merges_stat) == -1) {
        close(merges_fd);
        return false;
    }

    merges_mmap_size_ = merges_stat.st_size;
    merges_mmap_ptr_ = mmap(nullptr, merges_mmap_size_, PROT_READ, MAP_PRIVATE, merges_fd, 0);
    close(merges_fd);

    if (merges_mmap_ptr_ == MAP_FAILED) return false;

    std::string merges_content(static_cast<char*>(merges_mmap_ptr_), merges_mmap_size_);
    std::istringstream merges_stream(merges_content);

    merge_rules_.clear();
    uint32_t priority = 0;

    while (std::getline(merges_stream, line)) {
        rtrim_cr(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            rtrim_cr(first);
            rtrim_cr(second);

            std::string merged = first + second;
            merge_rules_.emplace_back(first, second, merged, priority);

            std::string key = first + "\x00" + second;
            auto it = merge_map_.find(key);
            if (it == merge_map_.end() || priority < it->second) {
                merge_map_[key] = priority;
            }
            priority++;
        }
    }

    std::sort(merge_rules_.begin(), merge_rules_.end(),
              [](const MergeRule& a, const MergeRule& b) {
                  return a.priority < b.priority;
              });

    return true;
}

bool BPETokenizer::load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) {
    if (!load_vocabulary_mmap(vocab_file, merges_file)) {
        return false;
    }

    std::ifstream config_stream(config_file);
    if (!config_stream.is_open()) {
        return true;
    }

    std::string line;
    while (std::getline(config_stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "eos_token_id") {
            eos_token_id_ = std::stoul(value);
        } else if (key == "pad_token_id") {
            if (unk_token_id_ == 0) {
                unk_token_id_ = std::stoul(value);
            }
        } else if (key == "unk_token_id" && value != "null") {
            unk_token_id_ = std::stoul(value);
        } else if (key == "bos_token_id" && value != "null") {
            bos_token_id_ = std::stoul(value);
        } else if (key == "vocab_size") {
            if (std::stoul(value) != vocab_size_) {
            }
        }
    }

    std::string dir = config_file.substr(0, config_file.find_last_of("/\\"));
    std::string special_tokens_path = dir + "/special_tokens.json";
    load_special_tokens(special_tokens_path);

    try {
        std::ifstream tok_json(dir + "/tokenizer.json");
        if (tok_json.is_open()) {
            std::string content((std::istreambuf_iterator<char>(tok_json)), std::istreambuf_iterator<char>());
            size_t pos = 0;
            while (true) {
                size_t id_key = content.find("\"id\"", pos);
                if (id_key == std::string::npos) break;
                size_t id_colon = content.find(':', id_key);
                if (id_colon == std::string::npos) break;
                size_t id_end = content.find_first_of(",}\n", id_colon + 1);
                if (id_end == std::string::npos) break;
                std::string id_str = content.substr(id_colon + 1, id_end - id_colon - 1);
                id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
                id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);

                size_t content_key = content.find("\"content\"", id_end);
                if (content_key == std::string::npos) { pos = id_end; continue; }
                size_t cont_quote1 = content.find('"', content_key + 9);
                if (cont_quote1 == std::string::npos) { pos = id_end; continue; }
                size_t cont_quote2 = content.find('"', cont_quote1 + 1);
                if (cont_quote2 == std::string::npos) { pos = id_end; continue; }
                std::string token_content = content.substr(cont_quote1 + 1, cont_quote2 - cont_quote1 - 1);

                size_t special_key = content.find("\"special\"", cont_quote2);
                if (special_key == std::string::npos) { pos = cont_quote2; continue; }
                size_t special_colon = content.find(':', special_key);
                if (special_colon == std::string::npos) { pos = cont_quote2; continue; }
                size_t special_val_start = content.find_first_not_of(" \t\n\r", special_colon + 1);
                if (special_val_start == std::string::npos) { pos = cont_quote2; continue; }
                bool is_special = false;
                if (content.compare(special_val_start, 4, "true") == 0) {
                    is_special = true;
                }
                if (is_special) {
                    try {
                        uint32_t token_id = static_cast<uint32_t>(std::stoul(id_str));
                        special_tokens_[token_content] = token_id;
                    } catch (...) {
                        std::cerr << "Warning: Failed to parse token ID '" << id_str << "' for special token '" << token_content << "'" << std::endl;
                    }
                }
                pos = cont_quote2 + 1;
            }
        }
    } catch (...) {
        std::cerr << "Warning: Failed to parse tokenizer.json for special tokens" << std::endl;
    }

    std::string template_path = dir + "/chat_template.jinja2";
    load_chat_template(template_path);

    std::string config_path = dir + "/config.txt";
    detect_model_type(config_path);

    return true;
}

void BPETokenizer::load_special_tokens(const std::string& config_file) {
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

std::vector<std::string> BPETokenizer::split_with_special_tokens(const std::string& text) const {
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

void BPETokenizer::init_byte_mappings() const {
    if (!byte_to_unicode_.empty()) return;

    std::vector<int> bytes;

    for (int i = 33; i <= 126; ++i) {
        bytes.push_back(i);
    }


    for (int i = 161; i <= 255; ++i) {
        bytes.push_back(i);
    }

    std::vector<int> remaining_bytes;
    for (int i = 0; i <= 32; ++i) remaining_bytes.push_back(i);
    remaining_bytes.push_back(127);
    for (int i = 128; i <= 160; ++i) remaining_bytes.push_back(i);

    int unicode_start = 256;
    for (int byte : remaining_bytes) {
        bytes.push_back(byte);
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(bytes[i]);

        if (byte >= 33 && byte <= 126) {
            std::string unicode_char(1, static_cast<char>(byte));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else if (byte >= 161 && byte <= 255) {
            std::string unicode_char;
            unicode_char += static_cast<char>(0xC0 | (byte >> 6));
            unicode_char += static_cast<char>(0x80 | (byte & 0x3F));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else {
            int unicode_point = unicode_start++;
            std::string unicode_char;
            if (unicode_point < 0x800) {
                unicode_char += static_cast<char>(0xC0 | (unicode_point >> 6));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            } else {
                unicode_char += static_cast<char>(0xE0 | (unicode_point >> 12));
                unicode_char += static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            }
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        }
    }
}

std::string BPETokenizer::bytes_to_unicode(const std::string& text) const {
    init_byte_mappings();

    std::string result;
    for (uint8_t byte : text) {
        result += byte_to_unicode_.at(byte);
    }
    return result;
}

std::string BPETokenizer::unicode_to_bytes(const std::string& text) const {
    init_byte_mappings();

    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        std::string unicode_char;

        if ((text[i] & 0x80) == 0) {
            unicode_char = text.substr(i, 1);
            i += 1;
        } else if ((text[i] & 0xE0) == 0xC0) {
            unicode_char = text.substr(i, 2);
            i += 2;
        } else if ((text[i] & 0xF0) == 0xE0) {
            unicode_char = text.substr(i, 3);
            i += 3;
        } else {
            unicode_char = text.substr(i, 1);
            i += 1;
        }

        auto it = unicode_to_byte_.find(unicode_char);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        } else if (unicode_char.size() == 1 && (unsigned char)unicode_char[0] < 128) {
            result += unicode_char[0];
        } else {
            result += '?';
        }
    }
    return result;
}

std::vector<std::string> BPETokenizer::byte_level_split(const std::string& text) const {
    std::string unicode_text = bytes_to_unicode(text);

    std::vector<std::string> chars;
    size_t i = 0;
    while (i < unicode_text.length()) {
        size_t char_len = 1;

        if ((unicode_text[i] & 0x80) == 0) {
            char_len = 1;
        } else if ((unicode_text[i] & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((unicode_text[i] & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((unicode_text[i] & 0xF8) == 0xF0) {
            char_len = 4;
        }

        if (i + char_len <= unicode_text.length()) {
            chars.push_back(unicode_text.substr(i, char_len));
        }
        i += char_len;
    }

    return chars;
}


std::pair<int, uint32_t> BPETokenizer::find_best_merge_fast(const std::vector<std::string>& tokens) const {
    int best_pos = -1;
    uint32_t best_priority = UINT32_MAX;

    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        std::string key = tokens[i] + "\x00" + tokens[i + 1];
        auto it = merge_map_.find(key);
        if (it != merge_map_.end()) {
            if (it->second < best_priority) {
                best_priority = it->second;
                best_pos = static_cast<int>(i);
            }
        }
    }

    return {best_pos, best_priority};
}

std::vector<std::string> BPETokenizer::apply_bpe(const std::vector<std::string>& tokens) const {
    if (tokens.size() <= 1) return tokens;

    std::vector<std::string> current_tokens = tokens;


    while (true) {
        auto [merge_pos, priority] = find_best_merge_fast(current_tokens);
        if (merge_pos == -1) break;


        std::vector<std::string> new_tokens;
        new_tokens.reserve(current_tokens.size() - 1);

        for (int i = 0; i < static_cast<int>(current_tokens.size()); ++i) {
            if (i == merge_pos) {
                std::string merged = current_tokens[i] + current_tokens[i + 1];
                new_tokens.push_back(merged);
                i++;
            } else {
                new_tokens.push_back(current_tokens[i]);
            }
        }
        current_tokens = std::move(new_tokens);
    }

    return current_tokens;
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};


    auto text_segments = split_with_special_tokens(text);


    std::vector<uint32_t> token_ids;

    for (const auto& segment : text_segments) {
        auto special_it = special_tokens_.find(segment);
        if (special_it != special_tokens_.end()) {
            token_ids.push_back(special_it->second);
        } else {
            auto chars = byte_level_split(segment);
            auto bpe_tokens = apply_bpe(chars);


            for (const auto& token : bpe_tokens) {
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    token_ids.push_back(unk_token_id_);
                }
            }
        }
    }

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string unicode_result;
    unicode_result.reserve(tokens.size() * 4);

    for (uint32_t token_id : tokens) {
        if (token_id >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[token_id];

        size_t pos = 0;
        while (pos < tok.size()) {
            if (pos + 3 <= tok.size() &&
                (unsigned char)tok[pos]   == 0xE2 &&
                (unsigned char)tok[pos+1] == 0x96 &&
                (unsigned char)tok[pos+2] == 0x81) {
                unicode_result.push_back(' ');
                pos += 3;
            } else {
                unicode_result.push_back(tok[pos++]);
            }
        }
    }

    return unicode_to_bytes(unicode_result);
}


void BPETokenizer::load_chat_template(const std::string& template_file) {
    std::ifstream file(template_file);
    if (!file.is_open()) {
        has_chat_template_ = false;
        return;
    }

    chat_template_ = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    has_chat_template_ = !chat_template_.empty();
}

}
}