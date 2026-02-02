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

void ToolCallConstrainer::tokenize_function_names(bool quote_names) {
    func_name_sequences_.clear();
    all_func_name_tokens_.clear();

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
    } else if (model_type_ == Config::ModelType::GEMMA) {
        gemma_call_start_tokens_.clear();
        gemma_call_end_tokens_.clear();
        gemma_call_prefix_tokens_.clear();
        escape_tokens_.clear();
        gemma_response_start_tokens_.clear();

        add_tokens_for_string("<start_function_call>", gemma_call_start_tokens_);
        add_tokens_for_string("<end_function_call>", gemma_call_end_tokens_);
        add_tokens_for_string("<start_function_response>", gemma_response_start_tokens_);
        add_tokens_for_string("call:", gemma_call_prefix_tokens_);
        add_tokens_for_string("<escape>", escape_tokens_);

        add_tokens_for_string("{", open_brace_tokens_);
        add_tokens_for_string("}", close_brace_tokens_);
        add_tokens_for_string(":", colon_tokens_);
        add_tokens_for_string(",", comma_tokens_);

        tokenize_function_names(false);  
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

        add_tokens_for_string("\"name\"", name_key_tokens_);
        add_tokens_for_string("name", name_key_tokens_);

        add_tokens_for_string("\"arguments\"", args_key_tokens_);
        add_tokens_for_string("arguments", args_key_tokens_);

        tokenize_function_names(true); 
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
    } else if (model_type_ == Config::ModelType::GEMMA) {
        state_ = State::GEMMA_START;
    } else {
        state_ = State::QWEN_START;
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
    } else if (model_type_ == Config::ModelType::GEMMA) {
        switch (state_) {
            case State::GEMMA_START:
                if (generated_text_.find("<start_function_call>") != std::string::npos) {
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
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        brace_depth_--;
                        if (brace_depth_ == 0) {
                            state_ = State::GEMMA_EXPECT_END;
                            generated_text_.clear();
                            break;
                        }
                    }
                }
                break;

            case State::GEMMA_EXPECT_END:
                if (generated_text_.find("<end_function_call>") != std::string::npos) {
                    state_ = State::DONE;
                    generated_text_.clear();
                }
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

            case State::QWEN_EXPECT_OPEN_BRACE:
                if (generated_text_.find("{") != std::string::npos) {
                    state_ = State::QWEN_EXPECT_NAME_KEY;
                    generated_text_.clear();
                }
                break;

            case State::QWEN_EXPECT_NAME_KEY:
                if (generated_text_.find("name") != std::string::npos) {
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
                    if (generated_text_.find(name) != std::string::npos) {
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
                if (generated_text_.find("arguments") != std::string::npos) {
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
                for (char c : decoded_text) {
                    if (c == '{') brace_depth_++;
                    else if (c == '}') {
                        if (brace_depth_ > 0) {
                            brace_depth_--;
                        } else {
                            state_ = State::QWEN_EXPECT_CLOSE_BRACE;
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
    } else if (model_type_ == Config::ModelType::GEMMA) {
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

            case State::QWEN_EXPECT_NAME_KEY:
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

            case State::QWEN_EXPECT_NAME_COLON:
                for (uint32_t t : colon_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                break;

            case State::QWEN_EXPECT_NAME_VALUE:
                for (uint32_t t : all_func_name_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::QWEN_EXPECT_COMMA:
                for (uint32_t t : comma_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : close_brace_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

            case State::QWEN_EXPECT_ARGS_KEY:
                for (uint32_t t : args_key_tokens_) {
                    current_bias_[t] = FORCE_BIAS;
                }
                for (uint32_t t : quote_tokens_) {
                    current_bias_[t] = 5.0f;
                }
                break;

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

    if (model_type_ == Config::ModelType::LFM2) {
        state_ = State::LFM_START;
    } else if (model_type_ == Config::ModelType::GEMMA) {
        state_ = State::GEMMA_START;
    } else {
        state_ = State::QWEN_START;
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