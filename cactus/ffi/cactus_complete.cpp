#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "telemetry/telemetry.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t ROLLING_ENTROPY_WINDOW = 10;

namespace {

std::string extract_last_user_query(const std::vector<ChatMessage>& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == "user") {
            return it->content;
        }
    }
    return {};
}

void inject_rag_context(CactusModelHandle* handle, std::vector<ChatMessage>& messages) {
    if (!handle->corpus_index) return;

    std::string query = extract_last_user_query(messages);
    if (query.empty()) return;

    std::string rag_context = retrieve_rag_context(handle, query);
    if (rag_context.empty()) return;

    if (!messages.empty() && messages[0].role == "system") {
        messages[0].content = rag_context + messages[0].content;
    } else {
        ChatMessage system_msg;
        system_msg.role = "system";
        system_msg.content = rag_context + "Answer the user's question using ONLY the context above. Do not use any prior knowledge. If the answer cannot be found in the context, respond with \"I don't have enough information to answer that.\"";
        messages.insert(messages.begin(), system_msg);
    }
}

void setup_tool_constraints(CactusModelHandle* handle, const std::vector<ToolFunction>& tools,
                           bool force_tools, float& temperature) {
    if (!force_tools || tools.empty()) return;

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

std::vector<std::vector<uint32_t>> build_stop_sequences(
    Tokenizer* tokenizer,
    const std::vector<std::string>& stop_sequences,
    Config::ModelType model_type,
    bool has_tools
) {
    std::vector<std::vector<uint32_t>> stop_token_sequences;
    stop_token_sequences.push_back({tokenizer->get_eos_token()});

    std::vector<std::string> sequences = stop_sequences;
    if (sequences.empty()) {
        std::string default_stop = tokenizer->get_default_stop_sequence();
        if (!default_stop.empty()) {
            sequences.push_back(default_stop);
        }
    }
    for (const auto& stop_seq : sequences) {
        stop_token_sequences.push_back(tokenizer->encode(stop_seq));
    }

    if (model_type == Config::ModelType::GEMMA && has_tools) {
        stop_token_sequences.push_back(tokenizer->encode("<end_function_call>"));
        stop_token_sequences.push_back(tokenizer->encode("<start_function_response>"));
    }

    return stop_token_sequences;
}

void trim_stop_suffix(std::vector<uint32_t>& generated_tokens,
                     const std::vector<std::vector<uint32_t>>& stop_token_sequences,
                     bool include_stop_sequences) {
    if (include_stop_sequences) return;
    for (const auto& stop_seq : stop_token_sequences) {
        if (stop_seq.empty()) continue;
        if (generated_tokens.size() >= stop_seq.size() &&
            std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin())) {
            generated_tokens.resize(generated_tokens.size() - stop_seq.size());
            break;
        }
    }
}

struct EntropyState {
    std::vector<float> window;
    float window_sum = 0.0f;
    float total_sum = 0.0f;
    size_t total_count = 0;
    bool spike_handoff = false;

    void add(float entropy) {
        window.push_back(entropy);
        window_sum += entropy;
        total_sum += entropy;
        total_count++;

        if (window.size() > ROLLING_ENTROPY_WINDOW) {
            window_sum -= window.front();
            window.erase(window.begin());
        }
    }

    float rolling_confidence() const {
        return 1.0f - (window_sum / window.size());
    }

    float mean_confidence() const {
        return 1.0f - (total_sum / static_cast<float>(total_count));
    }
};

uint32_t generate_first_token(
    CactusModelHandle* handle,
    const std::vector<uint32_t>& tokens_to_process,
    const std::vector<std::string>& image_paths,
    float temperature, float top_p, size_t top_k,
    float* first_token_entropy
) {
    if (tokens_to_process.empty()) {
        if (handle->processed_tokens.empty()) {
            throw std::runtime_error("Cannot generate from empty prompt");
        }
        std::vector<uint32_t> last_token_vec = { handle->processed_tokens.back() };
        return handle->model->decode(last_token_vec, temperature, top_p, top_k, "", first_token_entropy);
    }

    if (!image_paths.empty()) {
        return handle->model->decode_with_images(tokens_to_process, image_paths, temperature, top_p, top_k, "", first_token_entropy);
    }

    size_t prefill_chunk_size = handle->model->get_prefill_chunk_size();
    if (tokens_to_process.size() > 1) {
        std::vector<uint32_t> prefill_tokens(tokens_to_process.begin(), tokens_to_process.end() - 1);
        handle->model->prefill(prefill_tokens, prefill_chunk_size);

        std::vector<uint32_t> last_token = {tokens_to_process.back()};
        return handle->model->decode(last_token, temperature, top_p, top_k, "", first_token_entropy);
    }
    return handle->model->decode(tokens_to_process, temperature, top_p, top_k, "", first_token_entropy);
}

} // anonymous namespace

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

        inject_rag_context(handle, messages);

        float temperature, top_p, confidence_threshold;
        size_t top_k, max_tokens, tool_rag_top_k;
        std::vector<std::string> stop_sequences;
        bool force_tools, include_stop_sequences, use_vad, telemetry_enabled;
        parse_options_json(
            options_json ? options_json : "", temperature,
            top_p, top_k, max_tokens, stop_sequences,
            force_tools, tool_rag_top_k, confidence_threshold,
            include_stop_sequences, use_vad, telemetry_enabled
        );

        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0)
            tools = parse_tools_json(tools_json);

        if (tool_rag_top_k > 0 && tools.size() > tool_rag_top_k) {
            std::string query = extract_last_user_query(messages);
            if (!query.empty()) {
                tools = select_relevant_tools(handle, query, tools, tool_rag_top_k);
            }
        }

        setup_tool_constraints(handle, tools, force_tools, temperature);

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

        auto stop_token_sequences = build_stop_sequences(tokenizer, stop_sequences, model_type, !tools.empty());

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        float first_token_entropy = 0.0f;

        uint32_t next_token = generate_first_token(handle, tokens_to_process, image_paths,
                                                    temperature, top_p, top_k, &first_token_entropy);

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

            cactus::telemetry::CompletionMetrics metrics{};
            metrics.success = false;
            metrics.cloud_handoff = true;
            metrics.ttft_ms = time_to_first_token;
            metrics.prefill_tps = prefill_tps;
            metrics.decode_tps = 0.0;
            metrics.response_time_ms = time_to_first_token;
            metrics.confidence = confidence;
            metrics.ram_usage_mb = get_ram_usage_mb();
            metrics.prefill_tokens = prompt_tokens;
            metrics.decode_tokens = 0;
            metrics.error_message = nullptr;
            metrics.function_calls_json = "[]";
            cactus::telemetry::recordCompletion(handle->model_name.c_str(), metrics);

            return static_cast<int>(result.length());
        }

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (force_tools && !tools.empty()) {
            handle->model->update_tool_constraints(next_token);
        }

        EntropyState entropy;
        entropy.add(first_token_entropy);

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

                entropy.add(token_entropy);

                if (entropy.rolling_confidence() < confidence_threshold) {
                    entropy.spike_handoff = true;
                    break;
                }

                if (force_tools && !tools.empty()) {
                    handle->model->update_tool_constraints(next_token);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                    trim_stop_suffix(generated_tokens, stop_token_sequences, include_stop_sequences);
                    break;
                }

                if (callback) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        } else {
            trim_stop_suffix(generated_tokens, stop_token_sequences, include_stop_sequences);
        }

        confidence = entropy.mean_confidence();

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
                                                     completion_tokens, confidence, entropy.spike_handoff);

        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());

        std::string function_calls_json = serialize_function_calls(function_calls);
        cactus::telemetry::CompletionMetrics metrics{};
        metrics.success = !entropy.spike_handoff;
        metrics.cloud_handoff = entropy.spike_handoff;
        metrics.ttft_ms = time_to_first_token;
        metrics.prefill_tps = prefill_tps;
        metrics.decode_tps = decode_tps;
        metrics.response_time_ms = total_time_ms;
        metrics.confidence = confidence;
        metrics.ram_usage_mb = get_ram_usage_mb();
        metrics.prefill_tokens = prompt_tokens;
        metrics.decode_tokens = completion_tokens;
        metrics.error_message = nullptr;
        metrics.function_calls_json = function_calls_json.c_str();
        cactus::telemetry::recordCompletion(handle->model_name.c_str(), metrics);

        return static_cast<int>(result.length());

    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("complete", "Exception: " << e.what());
        handle_error_response(e.what(), response_buffer, buffer_size);

        cactus::telemetry::CompletionMetrics metrics{};
        metrics.success = false;
        metrics.cloud_handoff = false;
        metrics.ttft_ms = 0.0;
        metrics.prefill_tps = 0.0;
        metrics.decode_tps = 0.0;
        metrics.response_time_ms = 0.0;
        metrics.confidence = 0.0;
        metrics.ram_usage_mb = get_ram_usage_mb();
        metrics.prefill_tokens = 0;
        metrics.decode_tokens = 0;
        metrics.error_message = e.what();
        metrics.function_calls_json = "[]";
        auto* h = static_cast<CactusModelHandle*>(model);
        cactus::telemetry::recordCompletion(h ? h->model_name.c_str() : "unknown", metrics);

        return -1;
    } catch (...) {
        CACTUS_LOG_ERROR("complete", "Unknown exception during completion");
        handle_error_response("Unknown error during completion", response_buffer, buffer_size);

        cactus::telemetry::CompletionMetrics metrics{};
        metrics.success = false;
        metrics.cloud_handoff = false;
        metrics.ttft_ms = 0.0;
        metrics.prefill_tps = 0.0;
        metrics.decode_tps = 0.0;
        metrics.response_time_ms = 0.0;
        metrics.confidence = 0.0;
        metrics.ram_usage_mb = get_ram_usage_mb();
        metrics.prefill_tokens = 0;
        metrics.decode_tokens = 0;
        metrics.error_message = "Unknown error during completion";
        metrics.function_calls_json = "[]";
        auto* h = static_cast<CactusModelHandle*>(model);
        cactus::telemetry::recordCompletion(h ? h->model_name.c_str() : "unknown", metrics);

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
