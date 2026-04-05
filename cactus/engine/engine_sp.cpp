#include "engine.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>

namespace {

constexpr const char* kSentencePieceSpace = "\xE2\x96\x81";

std::string trim_config_value(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

bool parse_config_bool(const std::string& value) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    return lowered == "1" || lowered == "true" || lowered == "yes";
}

bool decode_utf8_codepoint(const std::string& text, size_t& pos, char32_t& codepoint) {
    if (pos >= text.size()) {
        return false;
    }

    unsigned char byte = static_cast<unsigned char>(text[pos]);
    if (byte < 0x80) {
        codepoint = byte;
        ++pos;
        return true;
    }
    if ((byte & 0xE0) == 0xC0 && pos + 1 < text.size()) {
        codepoint = ((byte & 0x1F) << 6) | (text[pos + 1] & 0x3F);
        pos += 2;
        return true;
    }
    if ((byte & 0xF0) == 0xE0 && pos + 2 < text.size()) {
        codepoint = ((byte & 0x0F) << 12) |
                    ((text[pos + 1] & 0x3F) << 6) |
                    (text[pos + 2] & 0x3F);
        pos += 3;
        return true;
    }
    if ((byte & 0xF8) == 0xF0 && pos + 3 < text.size()) {
        codepoint = ((byte & 0x07) << 18) |
                    ((text[pos + 1] & 0x3F) << 12) |
                    ((text[pos + 2] & 0x3F) << 6) |
                    (text[pos + 3] & 0x3F);
        pos += 4;
        return true;
    }

    ++pos;
    return false;
}

void append_utf8_codepoint(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool parse_byte_piece(const std::string& piece, uint8_t& value) {
    unsigned int v;
    if (piece.size() != 6 || sscanf(piece.c_str(), "<0x%02X>", &v) != 1) return false;
    value = static_cast<uint8_t>(v);
    return true;
}

} // namespace

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
    runtime_config_ = load_tokenizer_runtime_config(config_file);
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
            if (line.empty()) {
                continue;
            }

            size_t first_tab = line.find('\t');
            if (first_tab == std::string::npos) {
                continue;
            }

            uint32_t id = static_cast<uint32_t>(std::stoul(line.substr(0, first_tab)));
            size_t second_tab = line.find('\t', first_tab + 1);
            std::string token = (second_tab == std::string::npos)
                ? line.substr(first_tab + 1)
                : line.substr(first_tab + 1, second_tab - first_tab - 1);

            float score = -static_cast<float>(id);
            if (second_tab != std::string::npos) {
                std::string score_str = trim_config_value(line.substr(second_tab + 1));
                if (!score_str.empty()) {
                    score = std::stof(score_str);
                }
            }

            token_to_id_[token] = id;
            if (id >= id_to_token_.size()) {
                id_to_token_.resize(id + 1);
                token_scores_.resize(id + 1, 0.0f);
            }
            id_to_token_[id] = token;
            token_scores_[id] = score;
        }
        vocab_size_ = id_to_token_.size();
    } else {
        std::string line;
        uint32_t id = 0;

        vocab_stream.seekg(0); 
        while (std::getline(vocab_stream, line)) {
            token_to_id_[line] = id;
            id_to_token_.push_back(line);
            token_scores_.push_back(-static_cast<float>(id));
            id++;
        }
        vocab_size_ = id;
    }

    vocab_stream.close();
    
    build_trie();

    load_tokenizer_config(config_file);
    
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
            if (decode_utf8_codepoint(token, pos, codepoint)) {
                u32_token.push_back(codepoint);
            }
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

    if (!use_sentencepiece_bpe()) {
        std::string processed = "";
        if (model_type_ == ModelType::BERT) {
            processed = kSentencePieceSpace;
        }

        for (size_t i = text.find_first_not_of(" "); i < text.length(); i++) {
            char c = text[i];
            if (c == ' ') {
                processed += kSentencePieceSpace;
            } else {
                processed += c;
            }
        }

        return processed;
    }

    std::string normalized;
    normalized.reserve(text.size());
    bool prev_space = sp_remove_extra_whitespaces_;
    for (char c : text) {
        if (c != ' ') {
            normalized += c;
            prev_space = false;
        } else if (!prev_space) {
            normalized += c;
            if (sp_remove_extra_whitespaces_) prev_space = true;
        }
    }
    if (sp_remove_extra_whitespaces_) {
        while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
    }
    if (normalized.empty()) return "";

    std::string processed;
    if (sp_add_dummy_prefix_) processed += kSentencePieceSpace;
    for (char c : normalized) {
        if (c == ' ' && sp_escape_whitespaces_) processed += kSentencePieceSpace;
        else processed += c;
    }
    return processed;
}

std::string SPTokenizer::postprocess_text(const std::string& text, bool strip_leading_space) const {
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
    if (strip_leading_space && !result.empty() && result[0] == ' ') {
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
        if (decode_utf8_codepoint(text, pos, codepoint)) {
            u32_text.push_back(codepoint);
        }
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
                append_utf8_codepoint(token, cp);
            }
            result.push_back({token, static_cast<uint32_t>(best_token_id)});
            pos += best_match_len;
        } else {
            char32_t cp = u32_text[pos];
            std::string char_str;
            append_utf8_codepoint(char_str, cp);
            result.push_back({char_str, unk_token_id_});
            pos++;
        }
    }
    
    return result;
}

std::vector<uint32_t> SPTokenizer::tokenize_with_bpe(const std::string& text) const {
    std::vector<std::string> symbols;
    for (size_t pos = 0, start = 0; pos < text.size(); ) {
        start = pos;
        char32_t cp;
        decode_utf8_codepoint(text, pos, cp);
        symbols.push_back(text.substr(start, pos - start));
    }
    std::vector<uint32_t> result;
    if (symbols.empty()) return result;

    while (symbols.size() > 1) {
        bool found_merge = false;
        size_t best_index = 0;
        float best_score = -std::numeric_limits<float>::infinity();

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string merged = symbols[i] + symbols[i + 1];
            auto token_it = token_to_id_.find(merged);
            if (token_it == token_to_id_.end()) {
                continue;
            }

            uint32_t token_id = token_it->second;
            float score = token_id < token_scores_.size()
                ? token_scores_[token_id]
                : -static_cast<float>(token_id);

            if (!found_merge || score > best_score || (score == best_score && i < best_index)) {
                found_merge = true;
                best_index = i;
                best_score = score;
            }
        }

        if (!found_merge) {
            break;
        }

        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    for (const auto& symbol : symbols) {
        auto token_it = token_to_id_.find(symbol);
        if (token_it != token_to_id_.end()) {
            result.push_back(token_it->second);
            continue;
        }

        if (sp_byte_fallback_) {
            for (unsigned char byte : symbol) {
                char buffer[7];
                std::snprintf(buffer, sizeof(buffer), "<0x%02X>", byte);
                auto byte_it = token_to_id_.find(buffer);
                result.push_back(byte_it != token_to_id_.end() ? byte_it->second : unk_token_id_);
            }
            continue;
        }

        result.push_back(unk_token_id_);
    }

    return result;
}

std::vector<std::string> SPTokenizer::split_with_special_tokens(const std::string& text) const {
    return cactus::engine::split_with_special_tokens(text, special_tokens_);
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
            if (processed.empty()) {
                continue;
            }

            if (use_sentencepiece_bpe()) {
                auto segment_ids = tokenize_with_bpe(processed);
                token_ids.insert(token_ids.end(), segment_ids.begin(), segment_ids.end());
            } else {
                auto token_pairs = tokenize_with_trie(processed);
                for (const auto& [token, id] : token_pairs) {
                    token_ids.push_back(id);
                }
            }
        }
    }

    return token_ids;
}

std::string SPTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string raw_text;

    for (uint32_t token_id : tokens) {
        if (token_id >= id_to_token_.size()) {
            continue;
        }

        const std::string& token = id_to_token_[token_id];
        uint8_t byte = 0;
        if (parse_byte_piece(token, byte)) {
            raw_text.push_back(static_cast<char>(byte));
        } else {
            raw_text += token;
        }
    }

    return postprocess_text(raw_text, tokens.size() != 1);
}

void SPTokenizer::load_special_tokens(const std::string& config_file) {
    load_special_tokens_map(config_file, special_tokens_);
}

void SPTokenizer::load_tokenizer_config(const std::string& config_file) {
    std::ifstream config_stream(config_file);
    if (!config_stream.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(config_stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::string key = trim_config_value(line.substr(0, eq_pos));
        std::string value = trim_config_value(line.substr(eq_pos + 1));

        if (key == "eos_token_id") {
            eos_token_id_ = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "pad_token_id") {
            pad_token_id_ = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "unk_token_id") {
            unk_token_id_ = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "bos_token_id") {
            bos_token_id_ = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "sp_model_type") {
            sp_model_type_ = (value == "bpe" || value == "BPE")
                ? SentencePieceModelType::BPE
                : SentencePieceModelType::LEGACY;
        } else if (key == "sp_add_dummy_prefix") {
            sp_add_dummy_prefix_ = parse_config_bool(value);
        } else if (key == "sp_remove_extra_whitespaces") {
            sp_remove_extra_whitespaces_ = parse_config_bool(value);
        } else if (key == "sp_escape_whitespaces") {
            sp_escape_whitespaces_ = parse_config_bool(value);
        } else if (key == "sp_byte_fallback") {
            sp_byte_fallback_ = parse_config_bool(value);
        }
    }
}

bool SPTokenizer::use_sentencepiece_bpe() const {
    return sp_model_type_ == SentencePieceModelType::BPE;
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
