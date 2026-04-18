#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cactus {
namespace engine {

namespace {
constexpr const char* kMetaspace = "\xE2\x96\x81";
}  // namespace

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
    token_to_id_.clear();
    id_to_token_.clear();
    special_tokens_.clear();
    const bool use_id_tab_vocab =
        runtime_config_.vocab_format == TokenizerRuntimeConfig::VocabFormat::ID_TAB_TOKEN;

    if (use_id_tab_vocab) {
        while (std::getline(vocab_stream, line)) {
            rtrim_cr(line);

            std::string token;
            uint32_t id = UINT32_MAX;

            std::istringstream iss(line);
            if (iss >> id) {
                if (std::getline(iss, token) && !token.empty() && token[0] == '\t') {
                    token = token.substr(1);
                }

                if (token.empty()) {
                    auto last_pos = vocab_stream.tellg();
                    while (std::getline(vocab_stream, line)) {
                        rtrim_cr(line);
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
                }
                id_to_token_[id] = token;
            }
        }
        vocab_size_ = static_cast<uint32_t>(id_to_token_.size());
    } else {
        uint32_t id = 0;
        while (std::getline(vocab_stream, line)) {
            rtrim_cr(line);
            token_to_id_[line] = id;
            id_to_token_.push_back(line);
            ++id;
        }
        vocab_size_ = id;
    }

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
    
    auto is_whitespace_only = [](const std::string& s) {
        if (s.empty()) return false;
        for (char c : s) {
            if (c != ' ' && c != '\n' && c != '\t' && c != '\r') return false;
        }
        return true;
    };
    std::vector<std::string> ws_tokens;
    for (const auto& tok : id_to_token_) {
        if (tok.size() >= 2 && is_whitespace_only(tok)) ws_tokens.push_back(tok);
    }
    std::sort(ws_tokens.begin(), ws_tokens.end(),
              [](const std::string& a, const std::string& b) { return a.size() < b.size(); });
    for (const auto& tok : ws_tokens) {
        for (size_t i = 1; i < tok.size(); i++) {
            std::string first = tok.substr(0, i);
            std::string second = tok.substr(i);
            if (!token_to_id_.count(first) || !token_to_id_.count(second)) continue;
            std::string key = first + "\x00" + second;
            if (merge_map_.find(key) == merge_map_.end()) {
                merge_rules_.emplace_back(first, second, tok, priority);
                merge_map_[key] = priority;
                priority++;
            }
            break;
        }
    }

    std::sort(merge_rules_.begin(), merge_rules_.end(),
              [](const MergeRule& a, const MergeRule& b) {
                  return a.priority < b.priority;
              });

    return true;
}

bool BPETokenizer::load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) {
    runtime_config_ = load_tokenizer_runtime_config(config_file);
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

    std::string template_path = dir + "/chat_template.jinja2";
    load_chat_template(template_path);

    std::string config_path = dir + "/config.txt";
    detect_model_type(config_path);

    return true;
}

void BPETokenizer::load_special_tokens(const std::string& config_file) {
    load_special_tokens_map(config_file, special_tokens_);
}

std::vector<std::string> BPETokenizer::split_with_special_tokens(const std::string& text) const {
    return cactus::engine::split_with_special_tokens(text, special_tokens_);
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

std::vector<std::string> BPETokenizer::utf8_split(const std::string& text) const {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.length()) {
        size_t char_len = 1;
        const unsigned char byte = static_cast<unsigned char>(text[i]);

        if ((byte & 0x80) == 0) {
            char_len = 1;
        } else if ((byte & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            char_len = 4;
        }

        if (i + char_len <= text.length()) {
            chars.push_back(text.substr(i, char_len));
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
            std::string normalized_segment = segment;
            std::vector<std::string> chars;
            if (runtime_config_.normalizer == TokenizerRuntimeConfig::Normalizer::METASPACE) {
                size_t pos = 0;
                while ((pos = normalized_segment.find(' ', pos)) != std::string::npos) {
                    normalized_segment.replace(pos, 1, kMetaspace);
                    pos += 3;
                }
                chars = utf8_split(normalized_segment);
            } else {
                chars = byte_level_split(segment);
            }
            auto bpe_tokens = apply_bpe(chars);

            for (const auto& token : bpe_tokens) {
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    if (!runtime_config_.byte_fallback) {
                        token_ids.push_back(unk_token_id_);
                        continue;
                    }

                    std::vector<uint32_t> fallback_ids;
                    fallback_ids.reserve(token.size());
                    for (unsigned char byte : token) {
                        char fallback_token[7];
                        std::snprintf(fallback_token, sizeof(fallback_token), "<0x%02X>", byte);
                        auto fallback_it = token_to_id_.find(fallback_token);
                        if (fallback_it == token_to_id_.end()) {
                            fallback_ids.clear();
                            break;
                        }
                        fallback_ids.push_back(fallback_it->second);
                    }

                    if (fallback_ids.empty()) {
                        token_ids.push_back(unk_token_id_);
                    } else {
                        token_ids.insert(token_ids.end(), fallback_ids.begin(), fallback_ids.end());
                    }
                }
            }
        }
    }

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    if (runtime_config_.decoder == TokenizerRuntimeConfig::Decoder::REPLACE_METASPACE) {
        std::string result;
        result.reserve(tokens.size() * 4);

        for (uint32_t token_id : tokens) {
            if (token_id >= id_to_token_.size()) continue;
            result += id_to_token_[token_id];
        }

        size_t pos = 0;
        while ((pos = result.find(kMetaspace, pos)) != std::string::npos) {
            result.replace(pos, 3, " ");
            pos += 1;
        }
        return result;
    }

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


}
}
