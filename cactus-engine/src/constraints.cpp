#include "engine.h"

namespace cactus {
namespace engine {

constexpr float FORCE_BIAS = 500.0f;
constexpr float BLOCK_BIAS = -500.0f;
constexpr float TRIE_BLOCK_BIAS = -1e9f;

static constexpr const char* kEscapeTag = "<|\"|>";
static constexpr size_t kEscapeTagLen = 5;

static bool ends_with_escape_tag(const std::string& text) {
    return text.size() >= kEscapeTagLen &&
           text.compare(text.size() - kEscapeTagLen, kEscapeTagLen, kEscapeTag) == 0;
}

void ToolCallConstrainer::add_tokens_for_string(const std::string& str, std::unordered_set<uint32_t>& token_set) {
    if (!tokenizer_) return;
    auto tokens = tokenizer_->encode(str);
    for (uint32_t t : tokens) {
        token_set.insert(t);
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

void ToolCallConstrainer::trie_insert(TrieNode* root, const std::string& word) {
    if (!root) return;
    TrieNode* node = root;
    for (char ch : word) {
        auto& child = node->children[ch];
        if (!child) child = std::make_unique<TrieNode>();
        node = child.get();
    }
    node->is_terminal = true;
}

const ToolCallConstrainer::TrieNode* ToolCallConstrainer::trie_seek(
    const TrieNode* root,
    const std::string& prefix
) const {
    const TrieNode* node = root;
    for (char ch : prefix) {
        if (!node) return nullptr;
        auto it = node->children.find(ch);
        if (it == node->children.end()) return nullptr;
        node = it->second.get();
    }
    return node;
}

bool ToolCallConstrainer::trie_token_valid(const std::string& token_text,
                                           const TrieNode* node,
                                           char terminator) const {
    if (!node) return false;
    for (char ch : token_text) {
        if (ch == terminator) return node->is_terminal;
        auto it = node->children.find(ch);
        if (it == node->children.end()) return false;
        node = it->second.get();
    }
    return true;
}

const ToolCallConstrainer::TrieNode* ToolCallConstrainer::current_param_trie() const {
    auto it = param_tries_.find(current_function_);
    return it != param_tries_.end() ? it->second.get() : nullptr;
}

void ToolCallConstrainer::mark_trie_bias(const TrieNode* node, char terminator,
                                         const std::vector<char>& extra_allowed,
                                         int32_t terminal_token, bool exact_terminator) {
    if (!node) return;

    std::vector<bool> valid(token_strings_.size(), false);
    bool has_valid = false;
    auto mark = [&](char first_char, bool unconditional) {
        auto idx_it = token_index_.find(first_char);
        if (idx_it == token_index_.end()) return;
        for (uint32_t token_id : idx_it->second) {
            if (valid[token_id]) continue;
            if (unconditional || trie_token_valid(token_strings_[token_id], node, terminator)) {
                valid[token_id] = true;
                has_valid = true;
            }
        }
    };

    for (const auto& [first_char, _] : node->children) mark(first_char, false);
    if (node->is_terminal) {
        if (exact_terminator) {
            auto idx_it = token_index_.find(terminator);
            if (idx_it != token_index_.end()) {
                for (uint32_t token_id : idx_it->second) {
                    if (!valid[token_id] && token_strings_[token_id].size() == 1) {
                        valid[token_id] = true;
                        has_valid = true;
                    }
                }
            }
        } else {
            mark(terminator, false);
        }
    }
    for (char c : extra_allowed) mark(c, true);
    if (node->is_terminal && terminal_token >= 0) {
        uint32_t tid = static_cast<uint32_t>(terminal_token);
        if (tid < valid.size() && !valid[tid]) {
            valid[tid] = true;
            has_valid = true;
        }
    }
    if (!has_valid) return;

    dense_bias_.assign(valid.size(), TRIE_BLOCK_BIAS);
    for (size_t token_id = 0; token_id < valid.size(); ++token_id) {
        if (valid[token_id]) dense_bias_[token_id] = 0.0f;
    }
    dense_ready_ = true;
}

void ToolCallConstrainer::reset_constraint_state() {
    region_ = Region::FREE;
    constraint_buffer_.clear();
    constrained_buf_.clear();
    current_function_.clear();
    in_arguments_ = false;
    arguments_depth_ = 0;
    nesting_depth_ = 0;
    in_string_value_ = false;
    prev_char_escape_ = false;
    at_key_start_ = false;
    await_enum_open_ = false;
    active_enum_trie_ = nullptr;
    emitted_keys_.clear();
    remaining_key_trie_.reset();
}

void ToolCallConstrainer::rebuild_remaining_keys() {
    remaining_key_trie_ = std::make_unique<TrieNode>();
    for (const auto& tool : tool_specs_) {
        if (tool.name != current_function_) continue;
        for (const auto& param_name : tool.parameter_names) {
            if (!param_name.empty() && !emitted_keys_.count(param_name)) {
                trie_insert(remaining_key_trie_.get(), param_name);
            }
        }
        break;
    }
}

bool ToolCallConstrainer::required_satisfied() const {
    for (const auto& tool : tool_specs_) {
        if (tool.name != current_function_) continue;
        for (const auto& req : tool.required_parameter_names) {
            if (!emitted_keys_.count(req)) return false;
        }
        return true;
    }
    return true;
}

void ToolCallConstrainer::build_constraint_tables() {
    name_trie_ = std::make_unique<TrieNode>();
    param_tries_.clear();
    enum_tries_.clear();
    for (const auto& tool : tool_specs_) {
        if (!tool.name.empty()) trie_insert(name_trie_.get(), tool.name);

        if (is_needle()) {
            auto param_root = std::make_unique<TrieNode>();
            for (const auto& param_name : tool.parameter_names) {
                if (!param_name.empty()) trie_insert(param_root.get(), param_name);
            }
            param_tries_[tool.name] = std::move(param_root);
        }

        for (size_t i = 0; i < tool.parameter_names.size() && i < tool.parameter_enums.size(); ++i) {
            if (tool.parameter_enums[i].empty()) continue;
            auto enum_root = std::make_unique<TrieNode>();
            for (const auto& value : tool.parameter_enums[i]) {
                if (!value.empty()) trie_insert(enum_root.get(), value);
            }
            enum_tries_[tool.name][tool.parameter_names[i]] = std::move(enum_root);
        }
    }

    const uint32_t vocab_size = tokenizer_ ? tokenizer_->get_vocab_size() : 0;
    if (token_strings_.size() == vocab_size) return;

    token_strings_.assign(vocab_size, "");
    token_index_.clear();

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
        token_strings_[token_id] = decoded;
        if (!decoded.empty()) token_index_[decoded.front()].push_back(token_id);
    }
}

bool ToolCallConstrainer::needle_at_arg_key_start() const {
    if (constraint_buffer_.size() < 2) return false;
    return constraint_buffer_.compare(constraint_buffer_.size() - 2, 2, "{\"") == 0 ||
           constraint_buffer_.compare(constraint_buffer_.size() - 2, 2, ",\"") == 0;
}

bool ToolCallConstrainer::needle_is_value_string_start() const {
    if (constraint_buffer_.empty()) return false;
    for (size_t i = constraint_buffer_.size() - 1; i-- > 0;) {
        char ch = constraint_buffer_[i];
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') continue;
        return ch == ':';
    }
    return false;
}

void ToolCallConstrainer::feed_needle_char(char ch) {
    if (region_ == Region::IN_NAME || region_ == Region::IN_ARG_KEY) {
        if (ch == '"') {
            if (region_ == Region::IN_NAME) current_function_ = constrained_buf_;
            constrained_buf_.clear();
            region_ = Region::FREE;
        } else {
            constrained_buf_.push_back(ch);
        }
        constraint_buffer_.push_back(ch);
        return;
    }

    constraint_buffer_.push_back(ch);

    if (in_string_value_) {
        if (prev_char_escape_) {
            prev_char_escape_ = false;
            return;
        }
        if (ch == '\\') {
            prev_char_escape_ = true;
            return;
        }
        if (ch == '"') in_string_value_ = false;
        return;
    }

    if (ch == '{' || ch == '[') {
        nesting_depth_++;
    } else if (ch == '}' || ch == ']') {
        nesting_depth_ = std::max(0, nesting_depth_ - 1);
        if (ch == '}' && in_arguments_ && nesting_depth_ < arguments_depth_) {
            in_arguments_ = false;
        }
    }

    if (constraint_buffer_.size() >= 8 && !in_arguments_ &&
        constraint_buffer_.compare(constraint_buffer_.size() - 8, 8, "\"name\":\"") == 0) {
        region_ = Region::IN_NAME;
        constrained_buf_.clear();
        return;
    }

    if (constraint_buffer_.size() >= 13 &&
        constraint_buffer_.compare(constraint_buffer_.size() - 13, 13, "\"arguments\":{") == 0) {
        in_arguments_ = true;
        arguments_depth_ = nesting_depth_;
        return;
    }

    if (in_arguments_ &&
        nesting_depth_ == arguments_depth_ &&
        needle_at_arg_key_start()) {
        region_ = Region::IN_ARG_KEY;
        constrained_buf_.clear();
        return;
    }

    if (ch == '"' && needle_is_value_string_start()) {
        in_string_value_ = true;
    }
}

void ToolCallConstrainer::feed_needle_text(const std::string& text) {
    for (char ch : text) feed_needle_char(ch);
}

void ToolCallConstrainer::feed_gemma_char(char ch) {
    if (state_ != State::GEMMA_NAME && state_ != State::GEMMA_ARGS) return;

    constraint_buffer_.push_back(ch);

    if (region_ == Region::IN_NAME) {
        if (ch == '{') {
            current_function_ = constrained_buf_;
            constrained_buf_.clear();
            region_ = Region::FREE;
            state_ = State::GEMMA_ARGS;
            in_arguments_ = true;
            nesting_depth_ = 1;
            arguments_depth_ = nesting_depth_;
            at_key_start_ = true;
            rebuild_remaining_keys();
        } else {
            constrained_buf_.push_back(ch);
        }
        return;
    }

    if (region_ == Region::IN_ARG_KEY) {
        if (ch == ':') {
            emitted_keys_.insert(constrained_buf_);
            active_enum_trie_ = nullptr;
            if (escape_tag_valid_) {
                auto fit = enum_tries_.find(current_function_);
                if (fit != enum_tries_.end()) {
                    auto kit = fit->second.find(constrained_buf_);
                    if (kit != fit->second.end()) active_enum_trie_ = kit->second.get();
                }
            }
            await_enum_open_ = active_enum_trie_ != nullptr;
            rebuild_remaining_keys();
            constrained_buf_.clear();
            region_ = Region::FREE;
        } else {
            constrained_buf_.push_back(ch);
        }
        return;
    }

    if (in_string_value_) {
        if (ends_with_escape_tag(constraint_buffer_)) {
            in_string_value_ = false;
            active_enum_trie_ = nullptr;
            constrained_buf_.clear();
        } else if (active_enum_trie_) {
            constrained_buf_.push_back(ch);
        }
        return;
    }
    if (ends_with_escape_tag(constraint_buffer_)) {
        in_string_value_ = true;
        at_key_start_ = false;
        await_enum_open_ = false;
        constrained_buf_.clear();
        return;
    }
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') return;

    if (at_key_start_ && ch != '}' && ch != ',') {
        region_ = Region::IN_ARG_KEY;
        constrained_buf_.clear();
        constrained_buf_.push_back(ch);
        at_key_start_ = false;
        return;
    }

    if (ch == '{' || ch == '[') {
        nesting_depth_++;
    } else if (ch == '}' || ch == ']') {
        nesting_depth_--;
        if (ch == '}' && nesting_depth_ < arguments_depth_) {
            in_arguments_ = false;
            state_ = State::GEMMA_EXPECT_END;
            forced_tag_progress_ = 0;
        }
    } else if (ch == ',' && nesting_depth_ == arguments_depth_) {
        at_key_start_ = true;
    }
}

void ToolCallConstrainer::feed_gemma_text(const std::string& text) {
    for (char ch : text) feed_gemma_char(ch);
}

void ToolCallConstrainer::tokenize_grammar_elements() {
    if (!tokenizer_) return;

    open_brace_tokens_.clear();
    close_brace_tokens_.clear();

    init_common_tokens();

    gemma_call_start_tokens_.clear();
    gemma_call_end_tokens_.clear();
    gemma_call_prefix_tokens_.clear();
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

    std::vector<uint32_t> escape_sequence = tokenizer_->encode(kEscapeTag);
    escape_tag_valid_ = escape_sequence.size() == 1;
    escape_tag_token_ = escape_tag_valid_ ? escape_sequence[0] : 0;
}

void ToolCallConstrainer::init(Config::ModelType model_type,
                               const std::vector<ToolConstraintSpec>& tools,
                               Tokenizer* tokenizer) {
    model_type_ = model_type;
    tool_specs_ = tools;
    tokenizer_ = tokenizer;
    generated_text_.clear();
    current_bias_.clear();
    dense_ready_ = false;
    forced_tag_progress_ = 0;
    const bool model_supported = is_needle() || Config::is_gemma_family(model_type_);
    active_ = model_supported && !tool_specs_.empty() && tokenizer != nullptr;

    reset_constraint_state();
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
    build_constraint_tables();
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
                state_ = State::GEMMA_NAME;
                generated_text_.clear();
                region_ = Region::IN_NAME;
                constrained_buf_.clear();
                constraint_buffer_.clear();
            }
            break;

        case State::GEMMA_NAME:
        case State::GEMMA_ARGS:
            feed_gemma_text(decoded_text);
            generated_text_.clear();
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
    dense_ready_ = false;

    if (!active_) return;

    if (!is_needle()) {
        for (uint32_t t : backtick_tokens_) {
            current_bias_[t] = BLOCK_BIAS;
        }
        for (uint32_t t : gemma_response_start_tokens_) {
            current_bias_[t] = BLOCK_BIAS;
        }
    }

    if (region_ != Region::FREE) {
        const TrieNode* node = nullptr;
        char terminator = '"';
        if (region_ == Region::IN_NAME) {
            node = trie_seek(name_trie_.get(), constrained_buf_);
            if (!is_needle()) terminator = '{';
        } else {
            const TrieNode* root = is_needle() ? current_param_trie() : remaining_key_trie_.get();
            node = trie_seek(root, constrained_buf_);
            if (!is_needle()) terminator = ':';
        }
        mark_trie_bias(node, terminator, {}, -1, !is_needle());
        return;
    }

    if (is_needle()) return;

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

        case State::GEMMA_ARGS:
            if (await_enum_open_) {
                current_bias_[escape_tag_token_] = FORCE_BIAS;
                for (uint32_t t : gemma_call_end_tokens_) current_bias_[t] = BLOCK_BIAS;
                for (uint32_t t : gemma_call_start_tokens_) current_bias_[t] = BLOCK_BIAS;
            } else if (active_enum_trie_ && in_string_value_) {
                mark_trie_bias(trie_seek(active_enum_trie_, constrained_buf_), '\0', {},
                               static_cast<int32_t>(escape_tag_token_));
            } else if (at_key_start_) {
                std::vector<char> extra;
                if (required_satisfied()) extra.push_back('}');
                mark_trie_bias(remaining_key_trie_.get(), ':', extra);
            } else {
                for (uint32_t t : gemma_call_end_tokens_) current_bias_[t] = BLOCK_BIAS;
                for (uint32_t t : gemma_call_start_tokens_) current_bias_[t] = BLOCK_BIAS;
                if (nesting_depth_ == arguments_depth_ && !in_string_value_ && !required_satisfied()) {
                    for (uint32_t t : close_brace_tokens_) current_bias_[t] = BLOCK_BIAS;
                }
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
    dense_ready_ = false;
    forced_tag_progress_ = 0;
    reset_constraint_state();

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
