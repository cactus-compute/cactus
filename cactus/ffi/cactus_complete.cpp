#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_telemetry.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;

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

        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        bool force_tools = false;
        parse_options_json(options_json ? options_json : "",
                          temperature, top_p, top_k, max_tokens, stop_sequences, force_tools);

        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0)
            tools = parse_tools_json(tools_json);

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

        std::string formatted_tools = format_tools_for_prompt(tools);
        std::string full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools);

        if (full_prompt.find("ERROR:") == 0) {
            CACTUS_LOG_ERROR("complete", "Prompt formatting failed: " << full_prompt.substr(6));
            handle_error_response(full_prompt.substr(6), response_buffer, buffer_size);
            return -1;
        }

        std::vector<uint32_t> current_prompt_tokens = tokenizer->encode(full_prompt);

        CACTUS_LOG_DEBUG("complete", "Prompt tokens: " << current_prompt_tokens.size() << ", max_tokens: " << max_tokens);

        std::vector<uint32_t> tokens_to_process;
        bool is_prefix = (current_prompt_tokens.size() >= handle->processed_tokens.size()) &&
                         std::equal(handle->processed_tokens.begin(), handle->processed_tokens.end(), current_prompt_tokens.begin());

        if (handle->processed_tokens.empty() || !is_prefix) {
            handle->model->reset_cache();
            tokens_to_process = current_prompt_tokens;
        } else {
            tokens_to_process.assign(current_prompt_tokens.begin() + handle->processed_tokens.size(), current_prompt_tokens.end());
        }

        size_t prompt_tokens = tokens_to_process.size();

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({tokenizer->get_eos_token()});
        for (const auto& stop_seq : stop_sequences)
            stop_token_sequences.push_back(tokenizer->encode(stop_seq));

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        uint32_t next_token;

        if (tokens_to_process.empty()) {
            if (handle->processed_tokens.empty()) {
                handle_error_response("Cannot generate from empty prompt", response_buffer, buffer_size);
                return -1;
            }
            std::vector<uint32_t> last_token_vec = { handle->processed_tokens.back() };
            next_token = handle->model->decode(last_token_vec, temperature, top_p, top_k);
        } else {
            if (!image_paths.empty()) {
                next_token = handle->model->decode_with_images(tokens_to_process, image_paths, temperature, top_p, top_k);
            } else {
                size_t prefill_chunk_size = handle->model->get_prefill_chunk_size();

                if (tokens_to_process.size() > 1) {
                    std::vector<uint32_t> prefill_tokens(tokens_to_process.begin(),
                                                         tokens_to_process.end() - 1);
                    handle->model->prefill(prefill_tokens, prefill_chunk_size);

                    std::vector<uint32_t> last_token = {tokens_to_process.back()};
                    next_token = handle->model->decode(last_token, temperature, top_p, top_k);
                } else {
                    next_token = handle->model->decode(tokens_to_process, temperature, top_p, top_k, "profile.txt");
                }
            }
        }

        handle->processed_tokens = current_prompt_tokens;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (force_tools && !tools.empty()) {
            handle->model->update_tool_constraints(next_token);
        }

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (callback) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (handle->should_stop) break;

                next_token = handle->model->decode({next_token}, temperature, top_p, top_k);
                generated_tokens.push_back(next_token);
                handle->processed_tokens.push_back(next_token);

                if (force_tools && !tools.empty()) {
                    handle->model->update_tool_constraints(next_token);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;

                if (callback) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        }

        if (force_tools && !tools.empty()) {
            handle->model->clear_tool_constraints();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double tokens_per_second = completion_tokens > 1 ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;

        std::string response_text = tokenizer->decode(generated_tokens);

        std::string regular_response;
        std::vector<std::string> function_calls;
        parse_function_calls_from_response(response_text, regular_response, function_calls);

        std::string result = construct_response_json(regular_response, function_calls, time_to_first_token,
                                                     total_time_ms, tokens_per_second, prompt_tokens,
                                                     completion_tokens);

        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());

        CactusTelemetry::getInstance().recordCompletion(
            handle->model_name,
            true,
            time_to_first_token,
            tokens_per_second,
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

}
