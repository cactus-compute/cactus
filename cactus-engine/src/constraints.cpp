#include "engine.h"

namespace cactus {
namespace engine {

constexpr float FORCE_BIAS = 500.0f;
constexpr float BLOCK_BIAS = -500.0f;
constexpr float NEEDLE_BLOCK_BIAS = -1e9f;

static bool contains_odd_quote_tags(const std::string& text) {
    static constexpr const char* kQuoteTag = "<|\"|>";
    static constexpr size_t kQuoteTagLen = 5;
    bool odd = false;
    size_t pos = 0;
    while ((pos = text.find(kQuoteTag, pos)) != std::string::npos) {
        odd = !odd;
        pos += kQuoteTagLen;
    }
    return odd;
}

void ToolCallConstrainer::add_tokens_for_string(const std::string& str, std::unordered_set<uint32_t>& token_set) {
    if (!tokenizer_) return;
    auto tokens = tokenizer_->encode(str);
    for (uint32_t t : tokens) {
        token_set.insert(t);
    }
}

void ToolCallConstrainer::add_tokens_for_prefix_string(const std::string& prefix, std::unordered_set<uint32_t>& token_set) {
    if (!tokenizer_) return;
    const uint32_t vocab = tokenizer_->get_vocab_size();
    for (uint32_t t = 0; t < vocab; t++) {
        if (tokenizer_->decode({t}).rfind(prefix, 0) == 0) {
            token_set.insert(t);
        }
    }
}

void ToolCallConstrainer::tokenize_function_names(bool quote_names) {
    all_func_name_tokens_.clear();
    func_name_sequences_.clear();

    for (const auto& name : function_names_) {
        std::string name_to_encode = quote_names ? ("\"" + name + "\"") : name;
        auto tokens = tokenizer_->encode(name_to_encode);
        func_name_sequences_[name] = tokens;
        for (uint32_t t : tokens) {
            all_func_name_tokens_.insert(t);
        }
        if (quote_names) {
            auto unquoted_tokens = tokenizer_->encode(name);
            for (uint32_t t : unquoted_tokens) {
                all_func_name_tokens_.insert(t);
            }
        }
    }
}

void ToolCallConstrainer::init_common_tokens() {
    backtick_tokens_.clear();
    add_tokens_for_string("`", backtick_tokens_);
    add_tokens_for_string("``", backtick_tokens_);
    add_tokens_for_string("```", backtick_tokens_);
    add_tokens_for_string("````", backtick_tokens_);
    add_tokens_for_string("```json", backtick_tokens_);
    add_tokens_for_string("```JSON", backtick_tokens_);
    add_tokens_for_string("``` json", backtick_tokens_);
    add_tokens_for_string("```\n", backtick_tokens_);
    add_tokens_for_string("` ", backtick_tokens_);
}

void ToolCallConstrainer::needle_insert_word(NeedleTrieNode* root, const std::string& word) {
    if (!root) return;
    NeedleTrieNode* node = root;
    for (char ch : word) {
        auto& child = node->children[ch];
        if (!child) child = std::make_unique<NeedleTrieNode>();
        node = child.get();
    }
    node->is_terminal = true;
}

const ToolCallConstrainer::NeedleTrieNode* ToolCallConstrainer::needle_get_trie_node(
    const NeedleTrieNode* root,
    const std::string& prefix
) const {
    const NeedleTrieNode* node = root;
    for (char ch : prefix) {
        if (!node) return nullptr;
        auto it = node->children.find(ch);
        if (it == node->children.end()) return nullptr;
        node = it->second.get();
    }
    return node;
}

bool ToolCallConstrainer::needle_check_token_valid(const std::string& token_text,
                                                   const NeedleTrieNode* trie_node) const {
    const NeedleTrieNode* node = trie_node;
    if (!node) return false;
    for (char ch : token_text) {
        if (ch == '"') return node->is_terminal;
        auto it = node->children.find(ch);
        if (it == node->children.end()) return false;
        node = it->second.get();
    }
    return true;
}

void ToolCallConstrainer::reset_needle_constraints() {
    needle_json_state_ = NeedleJsonState::FREE;
    needle_buffer_.clear();
    needle_constrained_buf_.clear();
    needle_current_function_.clear();
    needle_in_arguments_ = false;
    needle_arguments_depth_ = 0;
    needle_nesting_depth_ = 0;
    needle_in_string_value_ = false;
    needle_prev_char_escape_ = false;
}

bool ToolCallConstrainer::needle_at_arg_key_start() const {
    if (needle_buffer_.size() < 2) return false;
    return needle_buffer_.compare(needle_buffer_.size() - 2, 2, "{\"") == 0 ||
           needle_buffer_.compare(needle_buffer_.size() - 2, 2, ",\"") == 0;
}

bool ToolCallConstrainer::needle_is_value_string_start() const {
    if (needle_buffer_.empty()) return false;
    for (size_t i = needle_buffer_.size() - 1; i-- > 0;) {
        char ch = needle_buffer_[i];
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') continue;
        return ch == ':';
    }
    return false;
}

void ToolCallConstrainer::feed_needle_char(char ch) {
    if (needle_json_state_ == NeedleJsonState::IN_NAME ||
        needle_json_state_ == NeedleJsonState::IN_ARG_KEY) {
        if (ch == '"') {
            if (needle_json_state_ == NeedleJsonState::IN_NAME) {
                needle_current_function_ = needle_constrained_buf_;
            }
            needle_constrained_buf_.clear();
            needle_json_state_ = NeedleJsonState::FREE;
        } else {
            needle_constrained_buf_.push_back(ch);
        }
        needle_buffer_.push_back(ch);
        return;
    }

    needle_buffer_.push_back(ch);

    if (needle_in_string_value_) {
        if (needle_prev_char_escape_) {
            needle_prev_char_escape_ = false;
            return;
        }
        if (ch == '\\') {
            needle_prev_char_escape_ = true;
            return;
        }
        if (ch == '"') needle_in_string_value_ = false;
        return;
    }

    if (ch == '{' || ch == '[') {
        needle_nesting_depth_++;
    } else if (ch == '}' || ch == ']') {
        needle_nesting_depth_ = std::max(0, needle_nesting_depth_ - 1);
        if (ch == '}' && needle_in_arguments_ && needle_nesting_depth_ < needle_arguments_depth_) {
            needle_in_arguments_ = false;
        }
    }

    if (needle_buffer_.size() >= 8 && !needle_in_arguments_ &&
        needle_buffer_.compare(needle_buffer_.size() - 8, 8, "\"name\":\"") == 0) {
        needle_json_state_ = NeedleJsonState::IN_NAME;
        needle_constrained_buf_.clear();
        return;
    }

    if (needle_buffer_.size() >= 13 &&
        needle_buffer_.compare(needle_buffer_.size() - 13, 13, "\"arguments\":{") == 0) {
        needle_in_arguments_ = true;
        needle_arguments_depth_ = needle_nesting_depth_;
        return;
    }

    if (needle_in_arguments_ &&
        needle_nesting_depth_ == needle_arguments_depth_ &&
        needle_at_arg_key_start()) {
        needle_json_state_ = NeedleJsonState::IN_ARG_KEY;
        needle_constrained_buf_.clear();
        return;
    }

    if (ch == '"' && needle_is_value_string_start()) {
        needle_in_string_value_ = true;
    }
}

void ToolCallConstrainer::feed_needle_text(const std::string& text) {
    for (char ch : text) feed_needle_char(ch);
}

void ToolCallConstrainer::init_needle_constraints() {
    reset_needle_constraints();

    needle_name_trie_ = std::make_unique<NeedleTrieNode>();
    needle_param_tries_.clear();
    for (const auto& tool : tool_specs_) {
        if (!tool.name.empty()) needle_insert_word(needle_name_trie_.get(), tool.name);

        auto param_root = std::make_unique<NeedleTrieNode>();
        for (const auto& param_name : tool.parameter_names) {
            if (!param_name.empty()) needle_insert_word(param_root.get(), param_name);
        }
        needle_param_tries_[tool.name] = std::move(param_root);
    }

    const uint32_t vocab_size = tokenizer_ ? tokenizer_->get_vocab_size() : 0;
    if (needle_token_strings_.size() != vocab_size) {
        needle_token_strings_.assign(vocab_size, "");
        needle_token_index_.clear();

        const uint32_t eos = tokenizer_->get_eos_token();
        const uint32_t bos = tokenizer_->get_bos_token();
        const uint32_t unk = tokenizer_->get_unk_token();
        constexpr uint32_t pad = 0;

        for (uint32_t token_id = 0; token_id < vocab_size; ++token_id) {
            if (token_id == pad || token_id == eos || token_id == bos || token_id == unk) continue;
            std::string decoded = tokenizer_->decode({token_id});
            if (decoded.size() == 1) {
                unsigned char b = static_cast<unsigned char>(decoded[0]);
                if (b >= 0x80) {
                    decoded.assign({static_cast<char>(0xC0 | (b >> 6)),
                                    static_cast<char>(0x80 | (b & 0x3F))});
                }
            }
            needle_token_strings_[token_id] = decoded;
            if (!decoded.empty()) needle_token_index_[decoded.front()].push_back(token_id);
        }
    }
}

void ToolCallConstrainer::tokenize_grammar_elements() {
    if (!tokenizer_) return;

    open_brace_tokens_.clear();
    close_brace_tokens_.clear();
    colon_tokens_.clear();
    comma_tokens_.clear();

    init_common_tokens();

    gemma_call_start_tokens_.clear();
    gemma_call_end_tokens_.clear();
    gemma_call_prefix_tokens_.clear();
    escape_tokens_.clear();
    gemma_response_start_tokens_.clear();

    call_start_sequence_ = tokenizer_->encode(call_start_tag_);
    call_end_sequence_ = tokenizer_->encode(call_end_tag_);
    std::vector<uint32_t> response_sequence = tokenizer_->encode(response_start_tag_);
    if (call_start_sequence_.size() == 1) gemma_call_start_tokens_.insert(call_start_sequence_[0]);
    if (call_end_sequence_.size() == 1) gemma_call_end_tokens_.insert(call_end_sequence_[0]);
    if (response_sequence.size() == 1) gemma_response_start_tokens_.insert(response_sequence[0]);
    add_tokens_for_string("call:", gemma_call_prefix_tokens_);

    add_tokens_for_string("{", open_brace_tokens_);
    add_tokens_for_string("}", close_brace_tokens_);
    add_tokens_for_string(":", colon_tokens_);
    add_tokens_for_string(",", comma_tokens_);

    tokenize_function_names(false);
}

void ToolCallConstrainer::init(Config::ModelType model_type,
                               const std::vector<ToolConstraintSpec>& tools,
                               Tokenizer* tokenizer) {
    model_type_ = model_type;
    tool_specs_ = tools;
    function_names_.clear();
    function_names_.reserve(tool_specs_.size());
    for (const auto& tool : tool_specs_) {
        function_names_.push_back(tool.name);
    }
    tokenizer_ = tokenizer;
    generated_text_.clear();
    current_bias_.clear();
    brace_depth_ = 0;
    in_argument_string_ = false;
    forced_tag_progress_ = 0;
    const bool model_supported = is_needle() || Config::is_gemma_family(model_type_);
    active_ = model_supported && !function_names_.empty() && tokenizer != nullptr;

    reset_needle_constraints();
    state_ = is_needle() ? State::NEEDLE_START : State::GEMMA_START;
    if (Config::is_gemma3_family(model_type_)) {
        call_start_tag_ = "<start_function_call>";
        call_end_tag_ = "<end_function_call>";
        response_start_tag_ = "<start_function_response>";
    } else {
        call_start_tag_ = "<|tool_call>";
        call_end_tag_ = "<tool_call|>";
        response_start_tag_ = "<|tool_response>";
    }

    if (!active_) {
        return;
    }

    tokenize_grammar_elements();
    if (is_needle()) {
        init_needle_constraints();
    }
    compute_bias();
}

void ToolCallConstrainer::advance_forced_tag(const std::vector<uint32_t>& sequence, uint32_t token_id) {
    if (forced_tag_progress_ < sequence.size() && token_id == sequence[forced_tag_progress_]) {
        forced_tag_progress_++;
    } else {
        forced_tag_progress_ = (!sequence.empty() && token_id == sequence[0]) ? 1 : 0;
    }
}

void ToolCallConstrainer::update(uint32_t token_id, const std::string& decoded_text) {
    if (!active_) return;

    generated_text_ += decoded_text;

    if (is_needle()) {
        feed_needle_text(decoded_text);
        generated_text_.clear();
        compute_bias();
        return;
    }

    switch (state_) {
        case State::GEMMA_START:
            advance_forced_tag(call_start_sequence_, token_id);
            if (generated_text_.find(call_start_tag_) != std::string::npos) {
                state_ = State::GEMMA_EXPECT_CALL;
                generated_text_.clear();
                forced_tag_progress_ = 0;
            }
            break;

        case State::GEMMA_EXPECT_CALL:
            if (generated_text_.find("call:") != std::string::npos) {
                state_ = State::GEMMA_IN_FUNC_NAME;
                generated_text_.clear();
            }
            break;

        case State::GEMMA_IN_FUNC_NAME:
            for (const auto& name : function_names_) {
                if (generated_text_.find(name) != std::string::npos) {
                    state_ = State::GEMMA_EXPECT_BRACE;
                    generated_text_.clear();
                    break;
                }
            }
            break;

        case State::GEMMA_EXPECT_BRACE:
            if (generated_text_.find("{") != std::string::npos) {
                state_ = State::GEMMA_IN_ARGUMENTS;
                brace_depth_ = 1;
                in_argument_string_ = false;
                generated_text_.clear();
            }
            break;

        case State::GEMMA_IN_ARGUMENTS:
            generated_text_.clear();
            if (contains_odd_quote_tags(decoded_text)) {
                in_argument_string_ = !in_argument_string_;
            }
            if (!in_argument_string_) {
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        brace_depth_--;
                        if (brace_depth_ == 0) {
                            state_ = State::GEMMA_EXPECT_END;
                            in_argument_string_ = false;
                            forced_tag_progress_ = 0;
                            break;
                        }
                    }
                }
            }
            break;

        case State::GEMMA_EXPECT_END:
            advance_forced_tag(call_end_sequence_, token_id);
            if (generated_text_.find(call_end_tag_) != std::string::npos) {
                state_ = State::DONE;
                generated_text_.clear();
            }
            break;

        default:
            break;
    }

    compute_bias();
}

void ToolCallConstrainer::compute_bias() {
    current_bias_.clear();

    if (!active_) return;

    if (!is_needle()) {
        for (uint32_t t : backtick_tokens_) {
            current_bias_[t] = BLOCK_BIAS;
        }

        for (uint32_t t : gemma_response_start_tokens_) {
            current_bias_[t] = BLOCK_BIAS;
        }
    }

    if (is_needle()) {
        if (needle_json_state_ == NeedleJsonState::FREE) return;

        const NeedleTrieNode* trie_node = nullptr;
        if (needle_json_state_ == NeedleJsonState::IN_NAME) {
            trie_node = needle_get_trie_node(needle_name_trie_.get(), needle_constrained_buf_);
        } else if (needle_json_state_ == NeedleJsonState::IN_ARG_KEY) {
            auto it = needle_param_tries_.find(needle_current_function_);
            if (it != needle_param_tries_.end()) {
                trie_node = needle_get_trie_node(it->second.get(), needle_constrained_buf_);
            }
        }
        if (!trie_node) return;

        std::vector<bool> valid_tokens(needle_token_strings_.size(), false);
        bool has_valid = false;
        auto mark_valid_tokens = [&](char first_char) {
            auto idx_it = needle_token_index_.find(first_char);
            if (idx_it == needle_token_index_.end()) return;
            for (uint32_t token_id : idx_it->second) {
                if (!valid_tokens[token_id] &&
                    needle_check_token_valid(needle_token_strings_[token_id], trie_node)) {
                    valid_tokens[token_id] = true;
                    has_valid = true;
                }
            }
        };

        for (const auto& [first_char, _] : trie_node->children) {
            mark_valid_tokens(first_char);
        }
        if (trie_node->is_terminal) {
            mark_valid_tokens('"');
        }
        if (!has_valid) return;

        for (size_t token_id = 0; token_id < valid_tokens.size(); ++token_id) {
            if (!valid_tokens[token_id]) {
                current_bias_[static_cast<uint32_t>(token_id)] = NEEDLE_BLOCK_BIAS;
            }
        }
        return;
    }

    switch (state_) {
        case State::GEMMA_START:
            if (forced_tag_progress_ < call_start_sequence_.size()) {
                current_bias_[call_start_sequence_[forced_tag_progress_]] = FORCE_BIAS;
            }
            for (uint32_t t : open_brace_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            for (uint32_t t : close_brace_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        case State::GEMMA_EXPECT_CALL:
            for (uint32_t t : gemma_call_prefix_tokens_) {
                current_bias_[t] = FORCE_BIAS;
            }
            for (uint32_t t : open_brace_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            for (uint32_t t : gemma_call_end_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        case State::GEMMA_IN_FUNC_NAME:
            for (uint32_t t : all_func_name_tokens_) {
                current_bias_[t] = FORCE_BIAS;
            }
            for (uint32_t t : close_brace_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            for (uint32_t t : gemma_call_end_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        case State::GEMMA_EXPECT_BRACE:
            for (uint32_t t : open_brace_tokens_) {
                current_bias_[t] = FORCE_BIAS;
            }
            for (uint32_t t : gemma_call_end_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        case State::GEMMA_IN_ARGUMENTS:
            if (!in_argument_string_) {
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = 10.0f;
                }
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = 8.0f;
                }
                for (uint32_t t : escape_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = 3.0f;
                }
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = 3.0f;
                }
            }
            for (uint32_t t : gemma_call_end_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            for (uint32_t t : gemma_call_start_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        case State::GEMMA_EXPECT_END:
            if (forced_tag_progress_ < call_end_sequence_.size()) {
                current_bias_[call_end_sequence_[forced_tag_progress_]] = FORCE_BIAS;
            }
            for (uint32_t t : open_brace_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            for (uint32_t t : gemma_call_start_tokens_) {
                current_bias_[t] = BLOCK_BIAS;
            }
            break;

        default:
            break;
    }
}

void ToolCallConstrainer::reset() {
    generated_text_.clear();
    current_bias_.clear();
    brace_depth_ = 0;
    in_argument_string_ = false;
    forced_tag_progress_ = 0;
    reset_needle_constraints();

    state_ = is_needle() ? State::NEEDLE_START : State::GEMMA_START;

    if (active_) {
        compute_bias();
    }
}


void Model::set_tool_constraints(const std::vector<ToolConstraintSpec>& tools) {
    tool_constrainer_.init(config_.model_type, tools, tokenizer_.get());
}

void Model::clear_tool_constraints() {
    tool_constrainer_.init(config_.model_type, {}, tokenizer_.get());
}

void Model::update_tool_constraints(uint32_t token_id) {
    if (tool_constrainer_.is_active() && tokenizer_) {
        std::string decoded = tokenizer_->decode({token_id});
        tool_constrainer_.update(token_id, decoded);
    }
}

} // namespace engine
} // namespace cactus
