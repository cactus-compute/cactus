#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_telemetry.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t ROLLING_ENTROPY_WINDOW = 10;

extern "C" {

int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ?
            "Model not initialized. Check model path and files." : last_error_message;
        CACTUS_LOG_ERROR("complete", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!messages_json || !response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("complete", "Invalid parameters: messages_json, response_buffer, or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();
        handle->should_stop = false;

        std::vector<std::string> image_paths;
        auto messages = parse_messages_json(messages_json, image_paths);

        if (messages.empty()) {
            CACTUS_LOG_ERROR("complete", "No messages provided in request");
            handle_error_response("No messages provided", response_buffer, buffer_size);
            return -1;
        }

        if (handle->corpus_index) {
            std::string query;
            for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
                if (it->role == "user") {
                    query = it->content;
                    break;
                }
            }

            if (!query.empty()) {
                std::string rag_context = retrieve_rag_context(handle, query);
                if (!rag_context.empty()) {
                    if (!messages.empty() && messages[0].role == "system") {
                        messages[0].content = rag_context + messages[0].content;
                    } else {
                        ChatMessage system_msg;
                        system_msg.role = "system";
                        system_msg.content = rag_context + "Answer the user's question using ONLY the context above. Do not use any prior knowledge. If the answer cannot be found in the context, respond with \"I don't have enough information to answer that.\"";
                        messages.insert(messages.begin(), system_msg);
                    }
                }
            }
        }

        float temperature, top_p, confidence_threshold;
        size_t top_k, max_tokens, tool_rag_top_k;
        std::vector<std::string> stop_sequences;
        bool force_tools, include_stop_sequences;
        parse_options_json(options_json ? options_json : "",
                          temperature, top_p, top_k, max_tokens, stop_sequences, force_tools, tool_rag_top_k, confidence_threshold, include_stop_sequences);

        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0)
            tools = parse_tools_json(tools_json);

        if (tool_rag_top_k > 0 && tools.size() > tool_rag_top_k) {
            std::string query;
            for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
                if (it->role == "user") {
                    query = it->content;
                    break;
                }
            }
            if (!query.empty()) {
                tools = select_relevant_tools(handle, query, tools, tool_rag_top_k);
            }
        }

        if (force_tools && !tools.empty()) {
            std::vector<std::string> function_names;
            function_names.reserve(tools.size());
            for (const auto& tool : tools) {
                function_names.push_back(tool.name);
            }
            handle->model->set_tool_constraints(function_names);

            if (temperature == 0.0f) {
                temperature = 0.01f;
            }
        }

        Config::ModelType model_type = handle->model->get_config().model_type;
        std::string formatted_tools;
        if (model_type == Config::ModelType::GEMMA) {
            formatted_tools = gemma::format_tools(tools);
        } else {
            formatted_tools = format_tools_for_prompt(tools);
        }
        std::string full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools);

        if (full_prompt.find("ERROR:") == 0) {
            CACTUS_LOG_ERROR("complete", "Prompt formatting failed: " << full_prompt.substr(6));
            handle_error_response(full_prompt.substr(6), response_buffer, buffer_size);
            return -1;
        }

        std::vector<uint32_t> current_prompt_tokens = tokenizer->encode(full_prompt);

        CACTUS_LOG_DEBUG("complete", "Prompt tokens: " << current_prompt_tokens.size() << ", max_tokens: " << max_tokens);

        std::vector<uint32_t> tokens_to_process;

        bool has_images = !image_paths.empty();
        bool is_prefix = !has_images &&
                         (current_prompt_tokens.size() >= handle->processed_tokens.size()) &&
                         std::equal(handle->processed_tokens.begin(), handle->processed_tokens.end(), current_prompt_tokens.begin());

        if (handle->processed_tokens.empty() || !is_prefix) {
            if (!has_images) {
                handle->model->reset_cache();
            }
            tokens_to_process = current_prompt_tokens;
        } else {
            tokens_to_process.assign(current_prompt_tokens.begin() + handle->processed_tokens.size(), current_prompt_tokens.end());
        }

        size_t prompt_tokens = tokens_to_process.size();

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({tokenizer->get_eos_token()});
        if (stop_sequences.empty()) {
            std::string default_stop = tokenizer->get_default_stop_sequence();
            if (!default_stop.empty()) {
                stop_sequences.push_back(default_stop);
            }
        }
        for (const auto& stop_seq : stop_sequences)
            stop_token_sequences.push_back(tokenizer->encode(stop_seq));

        if (model_type == Config::ModelType::GEMMA && !tools.empty()) {
            stop_token_sequences.push_back(tokenizer->encode("<end_function_call>"));
            stop_token_sequences.push_back(tokenizer->encode("<start_function_response>"));
        }

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        uint32_t next_token;
        float first_token_entropy = 0.0f;

        auto trim_stop_suffix = [&]() {
            if (include_stop_sequences) return;
            for (const auto& stop_seq : stop_token_sequences) {
                if (stop_seq.empty()) continue;
                if (generated_tokens.size() >= stop_seq.size() &&
                    std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin())) {
                    generated_tokens.resize(generated_tokens.size() - stop_seq.size());
                    break;
                }
            }
        };

        if (tokens_to_process.empty()) {
            if (handle->processed_tokens.empty()) {
                handle_error_response("Cannot generate from empty prompt", response_buffer, buffer_size);
                return -1;
            }
            std::vector<uint32_t> last_token_vec = { handle->processed_tokens.back() };
            next_token = handle->model->decode(last_token_vec, temperature, top_p, top_k, "", &first_token_entropy);
        } else {
            if (!image_paths.empty()) {
                next_token = handle->model->decode_with_images(tokens_to_process, image_paths, temperature, top_p, top_k, "", &first_token_entropy);
            } else {
                size_t prefill_chunk_size = handle->model->get_prefill_chunk_size();

                if (tokens_to_process.size() > 1) {
                    std::vector<uint32_t> prefill_tokens(tokens_to_process.begin(),
                                                         tokens_to_process.end() - 1);
                    handle->model->prefill(prefill_tokens, prefill_chunk_size);

                    std::vector<uint32_t> last_token = {tokens_to_process.back()};
                    next_token = handle->model->decode(last_token, temperature, top_p, top_k, "", &first_token_entropy);
                } else {
                    next_token = handle->model->decode(tokens_to_process, temperature, top_p, top_k, "", &first_token_entropy);
                }
            }
        }

        handle->processed_tokens = current_prompt_tokens;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        float confidence = 1.0f - first_token_entropy;

        if (confidence < confidence_threshold) {
            double prefill_tps = time_to_first_token > 0 ? (prompt_tokens * 1000.0) / time_to_first_token : 0.0;
            std::string result = construct_cloud_handoff_json(confidence, time_to_first_token, prefill_tps, prompt_tokens);
            if (result.length() >= buffer_size) {
                handle_error_response("Response buffer too small", response_buffer, buffer_size);
                return -1;
            }
            std::strcpy(response_buffer, result.c_str());

            CactusTelemetry::getInstance().recordCompletion(
                handle->model_name,
                true,
                time_to_first_token,
                0.0,
                time_to_first_token,
                static_cast<int>(prompt_tokens),
                ""
            );

            return static_cast<int>(result.length());
        }

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (force_tools && !tools.empty()) {
            handle->model->update_tool_constraints(next_token);
        }

        std::vector<float> entropy_window;
        entropy_window.push_back(first_token_entropy);
        float entropy_sum = first_token_entropy;
        float total_entropy_sum = first_token_entropy;
        size_t total_entropy_count = 1;
        bool entropy_spike_handoff = false;

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (callback) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (handle->should_stop) break;

                float token_entropy = 0.0f;
                next_token = handle->model->decode({next_token}, temperature, top_p, top_k, "", &token_entropy);
                generated_tokens.push_back(next_token);
                handle->processed_tokens.push_back(next_token);

                total_entropy_sum += token_entropy;
                total_entropy_count++;

                entropy_window.push_back(token_entropy);
                entropy_sum += token_entropy;
                if (entropy_window.size() > ROLLING_ENTROPY_WINDOW) {
                    entropy_sum -= entropy_window.front();
                    entropy_window.erase(entropy_window.begin());
                }

                float rolling_mean_entropy = entropy_sum / entropy_window.size();
                float rolling_confidence = 1.0f - rolling_mean_entropy;
                if (rolling_confidence < confidence_threshold) {
                    entropy_spike_handoff = true;
                    break;
                }

                if (force_tools && !tools.empty()) {
                    handle->model->update_tool_constraints(next_token);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                    trim_stop_suffix();
                    break;
                }

                if (callback) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        } else {
            trim_stop_suffix();
        }

        float mean_entropy = total_entropy_sum / static_cast<float>(total_entropy_count);
        confidence = 1.0f - mean_entropy;

        if (force_tools && !tools.empty()) {
            handle->model->clear_tool_constraints();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double prefill_tps = time_to_first_token > 0 ? (prompt_tokens * 1000.0) / time_to_first_token : 0.0;
        double decode_tps = (completion_tokens > 1 && decode_time_ms > 0) ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;

        std::string response_text = tokenizer->decode(generated_tokens);

        std::string regular_response;
        std::vector<std::string> function_calls;
        parse_function_calls_from_response(response_text, regular_response, function_calls);

        std::string result = construct_response_json(regular_response, function_calls, time_to_first_token,
                                                     total_time_ms, prefill_tps, decode_tps, prompt_tokens,
                                                     completion_tokens, confidence, entropy_spike_handoff);

        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());

        CactusTelemetry::getInstance().recordCompletion(
            handle->model_name,
            true,
            time_to_first_token,
            decode_tps,
            total_time_ms,
            static_cast<int>(prompt_tokens + completion_tokens),
            ""
        );

        return static_cast<int>(result.length());

    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("complete", "Exception: " << e.what());

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordCompletion(
            handle ? handle->model_name : "unknown",
            false,
            0.0, 0.0, 0.0, 0,
            std::string(e.what())
        );

        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        CACTUS_LOG_ERROR("complete", "Unknown exception during completion");

        auto* handle = static_cast<CactusModelHandle*>(model);
        CactusTelemetry::getInstance().recordCompletion(
            handle ? handle->model_name : "unknown",
            false,
            0.0, 0.0, 0.0, 0,
            "Unknown exception"
        );

        handle_error_response("Unknown error during completion", response_buffer, buffer_size);
        return -1;
    }
}

int cactus_tokenize(
    cactus_model_t model,
    const char* text,
    uint32_t* token_buffer,
    size_t token_buffer_len,
    size_t* out_token_len
) {
    if (!model || !text || !out_token_len) return -1;

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();

        std::vector<uint32_t> toks = tokenizer->encode(std::string(text));
        *out_token_len = toks.size();

        if (!token_buffer || token_buffer_len == 0) return 0;
        if (token_buffer_len < toks.size()) return -2;

        std::memcpy(token_buffer, toks.data(), toks.size() * sizeof(uint32_t));
        return 0;
    } catch (...) {
        return -1;
    }
}

int cactus_score_window(
    cactus_model_t model,
    const uint32_t* tokens,
    size_t token_len,
    size_t start,
    size_t end,
    size_t context,
    char* response_buffer,
    size_t buffer_size
) {
    if (!model || !tokens || token_len == 0 || !response_buffer || buffer_size == 0) {
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);

        std::vector<uint32_t> vec(tokens, tokens + token_len);

        size_t scored = 0;
        double logprob = handle->model->score_tokens_window_logprob(vec, start, end, context, &scored);

        std::ostringstream oss;
        oss << "{"
            << "\"success\":true,"
            << "\"logprob\":" << std::setprecision(10) << logprob << ","
            << "\"tokens\":" << scored
            << "}";

        std::string result = oss.str();
        if (result.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());
        return (int)result.size();

    } catch (const std::exception& e) {
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    }
}

}
