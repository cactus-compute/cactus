#include "engine.h"

namespace cactus {
namespace engine {

constexpr float FORCE_BIAS = 500.0f;
constexpr float BLOCK_BIAS = -500.0f;    

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
        if (!child) {
            child = std::make_unique<NeedleTrieNode>();
        }
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
        if (it == node->children.end()) {
            return nullptr;
        }
        node = it->second.get();
    }
    return node;
}

bool ToolCallConstrainer::needle_has_unseen_completion(const NeedleTrieNode* node, std::string& partial) const {
    if (!node) return false;
    if (node->is_terminal && !needle_seen_arg_keys_.count(partial)) return true;
    for (const auto& [ch, child] : node->children) {
        partial.push_back(ch);
        if (needle_has_unseen_completion(child.get(), partial)) {
            partial.pop_back();
            return true;
        }
        partial.pop_back();
    }
    return false;
}

bool ToolCallConstrainer::needle_check_token_valid(const std::string& token_text,
                                                   const NeedleTrieNode* trie_node) const {
    const NeedleTrieNode* node = trie_node;
    if (!node) return false;

    for (char ch : token_text) {
        if (ch == '"') {
            return node->is_terminal;
        }
        auto it = node->children.find(ch);
        if (it == node->children.end()) {
            return false;
        }
        node = it->second.get();
    }
    return true;
}

void ToolCallConstrainer::reset_needle_constraints() {
    needle_json_state_ = NeedleJsonState::FREE;
    needle_buffer_.clear();
    needle_constrained_buf_.clear();
    needle_current_function_.clear();
    needle_seen_arg_keys_.clear();
    needle_in_arguments_ = false;
    needle_arguments_depth_ = 0;
    needle_nesting_depth_ = 0;
    needle_in_string_value_ = false;
    needle_between_pairs_ = false;
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
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            continue;
        }
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
                needle_seen_arg_keys_.clear();
            } else {
                needle_seen_arg_keys_.insert(needle_constrained_buf_);
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
        if (ch == '"') {
            needle_in_string_value_ = false;
            if (needle_in_arguments_ && needle_nesting_depth_ == needle_arguments_depth_) {
                needle_between_pairs_ = true;
            }
        }
        return;
    }

    if (ch == '{' || ch == '[') {
        needle_nesting_depth_++;
    } else if (ch == '}' || ch == ']') {
        needle_nesting_depth_ = std::max(0, needle_nesting_depth_ - 1);
        if (ch == '}' && needle_in_arguments_ && needle_nesting_depth_ < needle_arguments_depth_) {
            needle_in_arguments_ = false;
        }
    } else if (ch == ',' && needle_in_arguments_ && needle_nesting_depth_ == needle_arguments_depth_) {
        needle_between_pairs_ = true;
    }

    if (needle_buffer_.size() >= 8 &&
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

    if (needle_in_arguments_ && needle_at_arg_key_start()) {
        needle_json_state_ = NeedleJsonState::IN_ARG_KEY;
        needle_constrained_buf_.clear();
        needle_between_pairs_ = false;
        return;
    }

    if (ch == '"' && needle_is_value_string_start()) {
        needle_in_string_value_ = true;
    }
}

void ToolCallConstrainer::feed_needle_text(const std::string& text) {
    for (char ch : text) {
        feed_needle_char(ch);
    }
}

void ToolCallConstrainer::init_needle_constraints() {
    reset_needle_constraints();

    needle_name_trie_ = std::make_unique<NeedleTrieNode>();
    needle_param_tries_.clear();
    needle_required_params_.clear();
    for (const auto& tool : tool_specs_) {
        if (!tool.name.empty()) {
            needle_insert_word(needle_name_trie_.get(), tool.name);
        }

        auto param_root = std::make_unique<NeedleTrieNode>();
        for (const auto& param_name : tool.parameter_names) {
            if (!param_name.empty()) {
                needle_insert_word(param_root.get(), param_name);
            }
        }
        needle_param_tries_[tool.name] = std::move(param_root);
        needle_required_params_[tool.name] = tool.required_parameter_names;
    }

    const uint32_t vocab_size = tokenizer_ ? tokenizer_->get_vocab_size() : 0;
    if (needle_token_strings_.size() != vocab_size) {
        needle_token_strings_.assign(vocab_size, "");
        needle_token_index_.clear();
        needle_arg_close_tokens_.clear();

        auto would_close_args = [](const std::string& tok, bool start_in_str) -> bool {
            bool in_str = start_in_str;
            bool esc = false;
            int depth = 0;
            for (char ch : tok) {
                if (in_str) {
                    if (esc) { esc = false; }
                    else if (ch == '\\') { esc = true; }
                    else if (ch == '"') { in_str = false; }
                    continue;
                }
                if (ch == '"') { in_str = true; continue; }
                if (ch == '{' || ch == '[') { depth++; continue; }
                if (ch == '}') { if (depth == 0) return true; depth--; }
                else if (ch == ']' && depth > 0) depth--;
            }
            return false;
        };

        for (uint32_t token_id = 0; token_id < vocab_size; ++token_id) {
            std::string decoded = tokenizer_->decode({token_id});
            needle_token_strings_[token_id] = decoded;
            if (decoded.empty()) continue;
            needle_token_index_[decoded.front()].push_back(token_id);
            if (would_close_args(decoded, false) || would_close_args(decoded, true)) {
                needle_arg_close_tokens_.insert(token_id);
            }
        }
    }
}

void ToolCallConstrainer::tokenize_grammar_elements() {
    if (!tokenizer_) return;

    // Clear all token sets
    open_brace_tokens_.clear();
    close_brace_tokens_.clear();
    colon_tokens_.clear();
    comma_tokens_.clear();
    name_key_tokens_.clear();
    args_key_tokens_.clear();
    quote_tokens_.clear();
    tool_start_tokens_.clear();
    tool_end_tokens_.clear();
    bracket_open_tokens_.clear();
    bracket_close_tokens_.clear();
    paren_open_tokens_.clear();
    paren_close_tokens_.clear();
    equals_tokens_.clear();

    init_common_tokens();

    if (model_type_ == Config::ModelType::LFM2) {
        add_tokens_for_string("<|tool_call_start|>", tool_start_tokens_);
        add_tokens_for_string("<|tool_call_end|>", tool_end_tokens_);
        add_tokens_for_string("[", bracket_open_tokens_);
        add_tokens_for_string("]", bracket_close_tokens_);
        add_tokens_for_string("(", paren_open_tokens_);
        add_tokens_for_string(")", paren_close_tokens_);
        add_tokens_for_string("=", equals_tokens_);
        add_tokens_for_string(",", comma_tokens_);
        add_tokens_for_string("\"", quote_tokens_);

        tokenize_function_names(false);
        add_tokens_for_prefix_string("(", paren_open_tokens_);
    } else if (is_gemma_family()) {
        gemma_call_start_tokens_.clear();
        gemma_call_end_tokens_.clear();
        gemma_call_prefix_tokens_.clear();
        escape_tokens_.clear();
        gemma_response_start_tokens_.clear();

        add_tokens_for_string(call_start_tag_, gemma_call_start_tokens_);
        add_tokens_for_string(call_end_tag_, gemma_call_end_tokens_);
        if (model_type_ == Config::ModelType::GEMMA4) {
            add_tokens_for_string("<|tool_response>", gemma_response_start_tokens_);
        } else {
            add_tokens_for_string("<start_function_response>", gemma_response_start_tokens_);
            add_tokens_for_string("<escape>", escape_tokens_);
        }
        add_tokens_for_string("call:", gemma_call_prefix_tokens_);

        add_tokens_for_string("{", open_brace_tokens_);
        add_tokens_for_string("}", close_brace_tokens_);
        add_tokens_for_string(":", colon_tokens_);
        add_tokens_for_string(",", comma_tokens_);

        tokenize_function_names(false);  
    } else if (is_needle()) {
        qwen_tool_call_start_tokens_.clear();
        add_tokens_for_string("<tool_call>", qwen_tool_call_start_tokens_);
    } else {
        qwen_tool_call_start_tokens_.clear();
        qwen_tool_call_end_tokens_.clear();

        add_tokens_for_string("<tool_call>", qwen_tool_call_start_tokens_);
        add_tokens_for_string("</tool_call>", qwen_tool_call_end_tokens_);

        add_tokens_for_string("{", open_brace_tokens_);
        add_tokens_for_string("}", close_brace_tokens_);
        add_tokens_for_string(":", colon_tokens_);
        add_tokens_for_string(",", comma_tokens_);
        add_tokens_for_string("\"", quote_tokens_);

        add_tokens_for_string("name", name_key_tokens_);

        add_tokens_for_string("arguments", args_key_tokens_);

        tokenize_function_names(true);
    }
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
    brace_depth_ = 0;
    active_ = !function_names_.empty() && tokenizer != nullptr;

    if (model_type_ == Config::ModelType::LFM2) {
        state_ = State::LFM_START;
        lfm_current_function_.clear();
        lfm_args_buffer_.clear();
        lfm_seen_arg_keys_.clear();
        lfm_required_params_.clear();
        lfm_all_params_.clear();
        for (const auto& tool : tool_specs_) {
            lfm_required_params_[tool.name] = tool.required_parameter_names;
            lfm_all_params_[tool.name] = tool.parameter_names;
        }
    } else if (is_gemma_family()) {
        state_ = State::GEMMA_START;
        if (model_type_ == Config::ModelType::GEMMA4) {
            call_start_tag_ = "<|tool_call>";
            call_end_tag_ = "<tool_call|>";
        } else {
            call_start_tag_ = "<start_function_call>";
            call_end_tag_ = "<end_function_call>";
        }
    } else if (is_needle()) {
        state_ = State::NEEDLE_START;
    } else {
        state_ = State::QWEN_START;
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

void ToolCallConstrainer::update(uint32_t /*token_id*/, const std::string& decoded_text) {
    if (!active_) return;

    generated_text_ += decoded_text;

    if (model_type_ == Config::ModelType::LFM2) {
        switch (state_) {
            case State::LFM_START:
                if (generated_text_.find("<|tool_call_start|>") != std::string::npos) {
                    state_ = State::LFM_EXPECT_BRACKET;
                    generated_text_.clear();
                }
                break;

            case State::LFM_EXPECT_BRACKET:
                if (generated_text_.find("[") != std::string::npos) {
                    state_ = State::LFM_IN_FUNC_NAME;
                    generated_text_.clear();
                }
                break;

            case State::LFM_IN_FUNC_NAME:
                for (const auto& name : function_names_) {
                    if (generated_text_.find(name) != std::string::npos) {
                        lfm_current_function_ = name;
                        state_ = State::LFM_EXPECT_PAREN;
                        generated_text_.clear();
                        break;
                    }
                }
                break;

            case State::LFM_EXPECT_PAREN:
                if (generated_text_.find("(") != std::string::npos) {
                    state_ = State::LFM_IN_ARGUMENTS;
                    lfm_args_buffer_.clear();
                    lfm_seen_arg_keys_.clear();
                    generated_text_.clear();
                }
                break;

            case State::LFM_IN_ARGUMENTS: {
                lfm_args_buffer_ += decoded_text;
                size_t eq_pos = 0;
                while ((eq_pos = lfm_args_buffer_.find('=', eq_pos)) != std::string::npos) {
                    size_t key_end = eq_pos;
                    size_t key_start = key_end;
                    while (key_start > 0 && !std::isspace(static_cast<unsigned char>(lfm_args_buffer_[key_start - 1])) &&
                           lfm_args_buffer_[key_start - 1] != ',' && lfm_args_buffer_[key_start - 1] != '(') {
                        key_start--;
                    }
                    if (key_start < key_end) {
                        std::string key = lfm_args_buffer_.substr(key_start, key_end - key_start);
                        lfm_seen_arg_keys_.insert(key);
                    }
                    eq_pos++;
                }
                if (decoded_text.find(")") != std::string::npos) {
                    state_ = State::LFM_EXPECT_BRACKET_CLOSE;
                    generated_text_.clear();
                } else {
                    generated_text_.clear();
                }
                break;
            }

            case State::LFM_EXPECT_BRACKET_CLOSE:
                if (generated_text_.find("]") != std::string::npos) {
                    state_ = State::LFM_EXPECT_END;
                    generated_text_.clear();
                }
                break;

            case State::LFM_EXPECT_END:
                if (generated_text_.find("<|tool_call_end|>") != std::string::npos) {
                    state_ = State::DONE;
                    generated_text_.clear();
                }
                break;

            default:
                break;
        }
    } else if (is_gemma_family()) {
        switch (state_) {
            case State::GEMMA_START:
                if (generated_text_.find(call_start_tag_) != std::string::npos) {
                    state_ = State::GEMMA_EXPECT_CALL;
                    generated_text_.clear();
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
                    generated_text_.clear();
                }
                break;

            case State::GEMMA_IN_ARGUMENTS:
                generated_text_.clear();
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        brace_depth_--;
                        if (brace_depth_ == 0) {
                            state_ = State::GEMMA_EXPECT_END;
                            break;
                        }
                    }
                }
                break;

            case State::GEMMA_EXPECT_END:
                if (generated_text_.find(call_end_tag_) != std::string::npos) {
                    state_ = State::DONE;
                    generated_text_.clear();
                }
                break;

            default:
                break;
        }
    } else if (is_needle()) {
        switch (state_) {
            case State::NEEDLE_START:
                if (generated_text_.find("<tool_call>") != std::string::npos) {
                    state_ = State::DONE;
                    generated_text_.clear();
                }
                break;

            case State::DONE:
                feed_needle_text(decoded_text);
                generated_text_.clear();
                break;

            default:
                break;
        }
    } else {
        switch (state_) {
            case State::QWEN_START:
                if (generated_text_.find("<tool_call>") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_OPEN_BRACE;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_OPEN_BRACE: {
                size_t pos = generated_text_.find("{");
                if (pos != std::string::npos) {
                    state_ = State::QWEN_EXPECT_NAME_KEY;
                    generated_text_ = generated_text_.substr(pos + 1);
                }
                break;
            }

            case State::QWEN_EXPECT_NAME_KEY:
                if (generated_text_.find("name\"") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_NAME_COLON;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_NAME_COLON:
                if (generated_text_.find(":") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_NAME_VALUE;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_NAME_VALUE:
                for (const auto& name : function_names_) {
                    if (generated_text_.find(name + "\"") != std::string::npos) {
                        state_ = State::QWEN_EXPECT_COMMA;
                        generated_text_.clear();
                        break;
                    }
                }
                break;

            case State::QWEN_EXPECT_COMMA:
                if (generated_text_.find(",") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_ARGS_KEY;
                    generated_text_.clear();
                } else if (generated_text_.find("}") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_END;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_ARGS_KEY:
                if (generated_text_.find("arguments\"") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_ARGS_COLON;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_ARGS_COLON:
                if (generated_text_.find(":") != std::string::npos) {
                    state_ = State::QWEN_IN_ARGUMENTS;
                    brace_depth_ = 0;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_IN_ARGUMENTS:
                generated_text_.clear();
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        if (brace_depth_ > 0) {
                            brace_depth_--;
                        } else {
                            state_ = State::QWEN_EXPECT_CLOSE_BRACE;
                            generated_text_.clear();
                            break;
                        }
                    }
                }
                break;

            case State::QWEN_EXPECT_CLOSE_BRACE:
                if (generated_text_.find("}") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_END;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_END:
                if (generated_text_.find("</tool_call>") != std::string::npos) {
                    state_ = State::DONE;
                    generated_text_.clear();
                }
                break;

            case State::DONE:
                if (generated_text_.find("<tool_call>") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_OPEN_BRACE;
                    generated_text_.clear();
                }
                break;

            default:
                break;
        }
    }

    compute_bias();
}

void ToolCallConstrainer::compute_bias() {
    current_bias_.clear();

    if (!active_) return;

    for (uint32_t t : backtick_tokens_) {
        current_bias_[t] = BLOCK_BIAS;
    }

    if (model_type_ == Config::ModelType::LFM2) {
        switch (state_) {
            case State::LFM_START:
                for (uint32_t t : tool_start_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : bracket_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::LFM_EXPECT_BRACKET:
                for (uint32_t t : bracket_open_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::LFM_IN_FUNC_NAME:
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : paren_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : equals_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::LFM_EXPECT_PAREN:
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : equals_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::LFM_IN_ARGUMENTS: {
                for (uint32_t t : equals_tokens_) {
                    current_bias_[t] = 10.0f;
                }
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = 8.0f;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                for (uint32_t t : paren_close_tokens_) {
                    current_bias_[t] = 3.0f;
                }
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : tool_end_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                auto req_it = lfm_required_params_.find(lfm_current_function_);
                if (req_it != lfm_required_params_.end()) {
                    bool has_all_required = true;
                    for (const auto& req : req_it->second) {
                        if (!lfm_seen_arg_keys_.count(req)) {
                            has_all_required = false;
                            break;
                        }
                    }
                    if (!has_all_required) {
                        for (uint32_t t : paren_close_tokens_) {
                            current_bias_[t] = BLOCK_BIAS;
                        }
                    }
                }
                break;
            }

            case State::LFM_EXPECT_BRACKET_CLOSE:
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : equals_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::LFM_EXPECT_END:
                for (uint32_t t : tool_end_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : bracket_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : paren_open_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            default:
                break;
        }
    } else if (is_gemma_family()) {
        for (uint32_t t : gemma_response_start_tokens_) {
            current_bias_[t] = BLOCK_BIAS;
        }

        switch (state_) {
            case State::GEMMA_START:
                for (uint32_t t : gemma_call_start_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
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
                for (uint32_t t : gemma_call_end_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::GEMMA_EXPECT_END:
                for (uint32_t t : gemma_call_end_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
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
    } else if (is_needle()) {
        switch (state_) {
            case State::NEEDLE_START:
                for (uint32_t t : qwen_tool_call_start_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                if (tokenizer_) {
                    current_bias_[tokenizer_->get_eos_token()] = FORCE_BIAS;
                }
                break;

            default:
                if (needle_json_state_ == NeedleJsonState::FREE) {
                    if (needle_in_arguments_ && needle_nesting_depth_ == needle_arguments_depth_) {
                        auto param_it = needle_param_tries_.find(needle_current_function_);
                        if (param_it != needle_param_tries_.end()) {
                            std::string partial;
                            bool all_seen = !needle_has_unseen_completion(param_it->second.get(), partial);
                            if (all_seen && needle_between_pairs_) {
                                for (uint32_t t : needle_arg_close_tokens_) {
                                    current_bias_[t] = FORCE_BIAS;
                                }
                            } else if (!all_seen && (needle_between_pairs_ || needle_in_string_value_)) {
                                auto req_it = needle_required_params_.find(needle_current_function_);
                                if (req_it != needle_required_params_.end()) {
                                    for (const auto& req : req_it->second) {
                                        if (!needle_seen_arg_keys_.count(req)) {
                                            for (uint32_t t : needle_arg_close_tokens_) {
                                                current_bias_[t] = BLOCK_BIAS;
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }

                const NeedleTrieNode* trie_node = nullptr;
                if (needle_json_state_ == NeedleJsonState::IN_NAME) {
                    trie_node = needle_get_trie_node(needle_name_trie_.get(), needle_constrained_buf_);
                } else if (needle_json_state_ == NeedleJsonState::IN_ARG_KEY) {
                    auto it = needle_param_tries_.find(needle_current_function_);
                    if (it != needle_param_tries_.end()) {
                        trie_node = needle_get_trie_node(it->second.get(), needle_constrained_buf_);
                    }
                }

                if (!trie_node) {
                    break;
                }

                std::vector<bool> valid_tokens(needle_token_strings_.size(), false);
                bool has_valid = false;
                auto mark_valid_tokens = [&](char first_char) {
                    auto idx_it = needle_token_index_.find(first_char);
                    if (idx_it == needle_token_index_.end()) {
                        return;
                    }
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

                if (!has_valid) {
                    break;
                }

                if (needle_json_state_ == NeedleJsonState::IN_ARG_KEY && !needle_seen_arg_keys_.empty()) {
                    for (size_t token_id = 0; token_id < valid_tokens.size(); ++token_id) {
                        if (!valid_tokens[token_id]) continue;
                        const std::string& tok = needle_token_strings_[token_id];
                        size_t quote_pos = tok.find('"');
                        if (quote_pos != std::string::npos) {
                            if (needle_seen_arg_keys_.count(needle_constrained_buf_ + tok.substr(0, quote_pos))) {
                                valid_tokens[token_id] = false;
                            }
                        } else {
                            std::string partial = needle_constrained_buf_ + tok;
                            const NeedleTrieNode* new_node = needle_get_trie_node(trie_node, tok);
                            if (!needle_has_unseen_completion(new_node, partial)) {
                                valid_tokens[token_id] = false;
                            }
                        }
                    }
                }

                for (size_t token_id = 0; token_id < valid_tokens.size(); ++token_id) {
                    if (!valid_tokens[token_id]) {
                        current_bias_[static_cast<uint32_t>(token_id)] = BLOCK_BIAS;
                    }
                }
                break;
        }
    } else {
        switch (state_) {
            case State::QWEN_START:
                for (uint32_t t : qwen_tool_call_start_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::QWEN_EXPECT_OPEN_BRACE:
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : qwen_tool_call_end_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::QWEN_EXPECT_NAME_KEY: {
                bool has_name = generated_text_.find("name") != std::string::npos;
                bool has_quote = generated_text_.find("\"") != std::string::npos;
                if (has_name) {
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                } else if (has_quote) {
                    for (uint32_t t : name_key_tokens_) { current_bias_[t] = FORCE_BIAS; }
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = BLOCK_BIAS; }
                } else {
                    for (uint32_t t : all_func_name_tokens_) { current_bias_[t] = BLOCK_BIAS; }
                    for (uint32_t t : args_key_tokens_) { current_bias_[t] = BLOCK_BIAS; }
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                }
                break;
            }

            case State::QWEN_EXPECT_NAME_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::QWEN_EXPECT_NAME_VALUE: {
                bool name_complete = false;
                for (const auto& name : function_names_) {
                    if (generated_text_.find(name) != std::string::npos) { name_complete = true; break; }
                }
                bool has_open_quote = generated_text_.find("\"") != std::string::npos;
                if (name_complete) {
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                } else if (has_open_quote) {
                    for (uint32_t t : all_func_name_tokens_) { current_bias_[t] = FORCE_BIAS; }
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = 5.0f; }
                } else {
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                }
                break;
            }

            case State::QWEN_EXPECT_COMMA:
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::QWEN_EXPECT_ARGS_KEY: {
                bool has_args = generated_text_.find("arguments") != std::string::npos;
                bool has_quote = generated_text_.find("\"") != std::string::npos;
                if (has_args) {
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                } else if (has_quote) {
                    for (uint32_t t : args_key_tokens_) { current_bias_[t] = FORCE_BIAS; }
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = BLOCK_BIAS; }
                } else {
                    for (uint32_t t : quote_tokens_) { current_bias_[t] = FORCE_BIAS; }
                }
                break;
            }

            case State::QWEN_EXPECT_ARGS_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::QWEN_IN_ARGUMENTS:
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = 3.0f;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = 3.0f;
                }
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = 2.0f;
                }
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = 2.0f;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 2.0f;
                }
                for (uint32_t t : qwen_tool_call_end_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::QWEN_EXPECT_CLOSE_BRACE:
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::QWEN_EXPECT_END:
                for (uint32_t t : qwen_tool_call_end_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : qwen_tool_call_start_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            default:
                break;
        }
    }
}

void ToolCallConstrainer::reset() {
    generated_text_.clear();
    current_bias_.clear();
    brace_depth_ = 0;
    reset_needle_constraints();

    if (model_type_ == Config::ModelType::LFM2) {
        state_ = State::LFM_START;
        lfm_current_function_.clear();
        lfm_args_buffer_.clear();
        lfm_seen_arg_keys_.clear();
    } else if (is_gemma_family()) {
        state_ = State::GEMMA_START;
    } else if (is_needle()) {
        state_ = State::NEEDLE_START;
    } else {
        state_ = State::QWEN_START;
    }

    if (active_) {
        compute_bias();
    }
}


void Model::set_tool_constraints(const std::vector<ToolConstraintSpec>& tools) {
    tool_constrainer_.init(config_.model_type, tools, tokenizer_.get());
}

void Model::clear_tool_constraints() {
    tool_constrainer_.reset();
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
