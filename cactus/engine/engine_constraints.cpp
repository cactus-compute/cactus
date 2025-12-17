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

void ToolCallConstrainer::tokenize_grammar_elements() {
    if (!tokenizer_) return;

    open_brace_tokens_.clear();
    close_brace_tokens_.clear();
    colon_tokens_.clear();
    comma_tokens_.clear();
    fc_key_tokens_.clear();
    name_key_tokens_.clear();
    args_key_tokens_.clear();
    quote_tokens_.clear();
    backtick_tokens_.clear();
    all_func_name_tokens_.clear();
    func_name_sequences_.clear();
    tool_start_tokens_.clear();
    tool_end_tokens_.clear();
    bracket_open_tokens_.clear();
    bracket_close_tokens_.clear();
    paren_open_tokens_.clear();
    paren_close_tokens_.clear();
    equals_tokens_.clear();

    add_tokens_for_string("`", backtick_tokens_);
    add_tokens_for_string("``", backtick_tokens_);
    add_tokens_for_string("```", backtick_tokens_);
    add_tokens_for_string("````", backtick_tokens_);
    add_tokens_for_string("```json", backtick_tokens_);
    add_tokens_for_string("```JSON", backtick_tokens_);
    add_tokens_for_string("``` json", backtick_tokens_);
    add_tokens_for_string("```\n", backtick_tokens_);
    add_tokens_for_string("` ", backtick_tokens_);

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

        for (const auto& name : function_names_) {
            auto tokens = tokenizer_->encode(name);
            func_name_sequences_[name] = tokens;
            for (uint32_t t : tokens) {
                all_func_name_tokens_.insert(t);
            }
        }
    } else {
        add_tokens_for_string("{", open_brace_tokens_);
        add_tokens_for_string("}", close_brace_tokens_);
        add_tokens_for_string(":", colon_tokens_);
        add_tokens_for_string(",", comma_tokens_);
        add_tokens_for_string("\"", quote_tokens_);

        add_tokens_for_string("\"function_call\"", fc_key_tokens_);
        add_tokens_for_string("function_call", fc_key_tokens_);

        add_tokens_for_string("\"name\"", name_key_tokens_);
        add_tokens_for_string("name", name_key_tokens_);

        add_tokens_for_string("\"arguments\"", args_key_tokens_);
        add_tokens_for_string("arguments", args_key_tokens_);

        for (const auto& name : function_names_) {
            std::string quoted_name = "\"" + name + "\"";
            auto tokens = tokenizer_->encode(quoted_name);
            func_name_sequences_[name] = tokens;
            for (uint32_t t : tokens) {
                all_func_name_tokens_.insert(t);
            }
            auto unquoted_tokens = tokenizer_->encode(name);
            for (uint32_t t : unquoted_tokens) {
                all_func_name_tokens_.insert(t);
            }
        }
    }
}

void ToolCallConstrainer::init(Config::ModelType model_type,
                                const std::vector<std::string>& function_names,
                                Tokenizer* tokenizer) {
    model_type_ = model_type;
    function_names_ = function_names;
    tokenizer_ = tokenizer;
    generated_text_.clear();
    brace_depth_ = 0;
    active_ = !function_names.empty() && tokenizer != nullptr;

    if (model_type_ == Config::ModelType::LFM2) {
        state_ = State::LFM_START;
    } else {
        state_ = State::START;
    }

    if (!active_) {
        return;
    }

    tokenize_grammar_elements();
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
                        state_ = State::LFM_EXPECT_PAREN;
                        generated_text_.clear();
                        break;
                    }
                }
                break;

            case State::LFM_EXPECT_PAREN:
                if (generated_text_.find("(") != std::string::npos) {
                    state_ = State::LFM_IN_ARGUMENTS;
                    generated_text_.clear();
                }
                break;

            case State::LFM_IN_ARGUMENTS:
                if (generated_text_.find(")") != std::string::npos) {
                    state_ = State::LFM_EXPECT_BRACKET_CLOSE;
                }
                break;

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
    } else {
        switch (state_) {
            case State::START:
                if (generated_text_.find("{") != std::string::npos) {
                    state_ = State::EXPECT_FC_KEY;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_FC_KEY:
                if (generated_text_.find("function_call") != std::string::npos) {
                    state_ = State::EXPECT_FC_COLON;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_FC_COLON:
                if (generated_text_.find(":") != std::string::npos) {
                    state_ = State::EXPECT_FC_OPEN_BRACE;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_FC_OPEN_BRACE:
                if (generated_text_.find("{") != std::string::npos) {
                    state_ = State::EXPECT_NAME_KEY;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_NAME_KEY:
                if (generated_text_.find("name") != std::string::npos) {
                    state_ = State::EXPECT_NAME_COLON;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_NAME_COLON:
                if (generated_text_.find(":") != std::string::npos) {
                    state_ = State::EXPECT_NAME_VALUE;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_NAME_VALUE:
                for (const auto& name : function_names_) {
                    if (generated_text_.find(name) != std::string::npos) {
                        state_ = State::EXPECT_COMMA;
                        generated_text_.clear();
                        break;
                    }
                }
                break;

            case State::EXPECT_COMMA:
                if (generated_text_.find(",") != std::string::npos) {
                    state_ = State::EXPECT_ARGS_KEY;
                    generated_text_.clear();
                } else if (generated_text_.find("}") != std::string::npos) {
                    state_ = State::EXPECT_OUTER_CLOSE;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_ARGS_KEY:
                if (generated_text_.find("arguments") != std::string::npos) {
                    state_ = State::EXPECT_ARGS_COLON;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_ARGS_COLON:
                if (generated_text_.find(":") != std::string::npos) {
                    state_ = State::IN_ARGUMENTS;
                    brace_depth_ = 0;
                    generated_text_.clear();
                }
                break;

            case State::IN_ARGUMENTS:
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        if (brace_depth_ > 0) {
                            brace_depth_--;
                        } else {
                            state_ = State::EXPECT_OUTER_CLOSE;
                            break;
                        }
                    }
                }
                break;

            case State::EXPECT_INNER_CLOSE:
                if (generated_text_.find("}") != std::string::npos) {
                    state_ = State::EXPECT_OUTER_CLOSE;
                    generated_text_.clear();
                }
                break;

            case State::EXPECT_OUTER_CLOSE:
                if (generated_text_.find("}") != std::string::npos) {
                    state_ = State::DONE;
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
                // Force [, block everything else structural
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
                // Force function name tokens, block structural tokens
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : bracket_close_tokens_) {
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
                // Force (, block other structural tokens
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

            case State::LFM_IN_ARGUMENTS:
                for (uint32_t t : paren_close_tokens_) {
                    current_bias_[t] = 15.0f; 
                }
                for (uint32_t t : equals_tokens_) {
                    current_bias_[t] = 10.0f; 
                }
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = 8.0f;   
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;   
                }
                for (uint32_t t : bracket_close_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : tool_end_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

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
    } else {
        switch (state_) {
            case State::START:
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::EXPECT_FC_KEY:
                for (uint32_t t : fc_key_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::EXPECT_FC_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::EXPECT_FC_OPEN_BRACE:
                for (uint32_t t : open_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::EXPECT_NAME_KEY:
                for (uint32_t t : name_key_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                for (uint32_t t : args_key_tokens_) {
                    current_bias_[t] = BLOCK_BIAS;
                }
                break;

            case State::EXPECT_NAME_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::EXPECT_NAME_VALUE:
                // Strongly bias towards function names
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::EXPECT_COMMA:
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::EXPECT_ARGS_KEY:
                for (uint32_t t : args_key_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::EXPECT_ARGS_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::IN_ARGUMENTS:
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
                break;

            case State::EXPECT_INNER_CLOSE:
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::EXPECT_OUTER_CLOSE:
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
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

    if (model_type_ == Config::ModelType::LFM2) {
        state_ = State::LFM_START;
    } else {
        state_ = State::START;
    }

    if (active_) {
        compute_bias();
    }
}


void Model::set_tool_constraints(const std::vector<std::string>& function_names) {
    tool_constrainer_.init(config_.model_type, function_names, tokenizer_.get());
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