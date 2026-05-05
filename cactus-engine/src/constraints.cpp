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

    add_tokens_for_string(call_start_tag_, gemma_call_start_tokens_);
    add_tokens_for_string(call_end_tag_, gemma_call_end_tokens_);
    add_tokens_for_string("<|tool_response>", gemma_response_start_tokens_);
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
    brace_depth_ = 0;
    active_ = !function_names_.empty() && tokenizer != nullptr;

    state_ = State::GEMMA_START;
    call_start_tag_ = "<|tool_call>";
    call_end_tag_ = "<tool_call|>";

    if (!active_) {
        return;
    }

    tokenize_grammar_elements();
    compute_bias();
}

void ToolCallConstrainer::update(uint32_t /*token_id*/, const std::string& decoded_text) {
    if (!active_) return;

    generated_text_ += decoded_text;

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

    compute_bias();
}

void ToolCallConstrainer::compute_bias() {
    current_bias_.clear();

    if (!active_) return;

    for (uint32_t t : backtick_tokens_) {
        current_bias_[t] = BLOCK_BIAS;
    }

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
}

void ToolCallConstrainer::reset() {
    generated_text_.clear();
    current_bias_.clear();
    brace_depth_ = 0;

    state_ = State::GEMMA_START;

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
