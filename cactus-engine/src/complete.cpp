#include "../cactus_engine.h"
#include "cloud.h"
#include "utils.h"
#include "chat_tools.h"
#include "telemetry.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include "wav.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace cactus::engine;
using namespace cactus::ffi;

namespace {

std::vector<std::pair<std::string, std::string>> extract_schema_property_types(const std::string& schema);
std::vector<std::string> extract_schema_required(const std::string& schema);
std::vector<std::string> extract_string_array(const std::string& json, const std::string& key);
std::string extract_json_object_field(const std::string& json, const std::string& key);

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

std::vector<ToolConstraintSpec> build_tool_constraint_specs(const std::vector<ToolFunction>& tools) {
    std::vector<ToolConstraintSpec> specs;
    specs.reserve(tools.size());

    for (const auto& tool : tools) {
        ToolConstraintSpec spec;
        spec.name = tool.name;

        auto schema_it = tool.parameters.find("schema");
        if (schema_it != tool.parameters.end()) {
            auto properties = extract_schema_property_types(schema_it->second);
            std::string properties_object = extract_json_object_field(schema_it->second, "properties");
            spec.parameter_names.reserve(properties.size());
            spec.parameter_enums.reserve(properties.size());
            for (const auto& [name, _] : properties) {
                spec.parameter_names.push_back(name);
                spec.parameter_enums.push_back(
                    extract_string_array(extract_json_object_field(properties_object, name), "enum"));
            }
            spec.required_parameter_names = extract_schema_required(schema_it->second);
        }

        specs.push_back(std::move(spec));
    }

    return specs;
}

void setup_tool_constraints(CactusModelHandle* handle, const std::vector<ToolFunction>& tools,
                           bool force_tools, float& temperature) {
    if (!force_tools || tools.empty()) return;

    handle->model->set_tool_constraints(build_tool_constraint_specs(tools));

    if (temperature == 0.0f) {
        temperature = 0.01f;
    }
}

size_t find_json_block_end(const std::string& json, size_t start) {
    if (start >= json.size() || json[start] != '{') {
        return std::string::npos;
    }

    int depth = 1;
    bool in_string = false;
    bool escaped = false;
    size_t pos = start + 1;
    while (pos < json.size() && depth > 0) {
        char c = json[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else {
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
            }
        }
        ++pos;
    }

    return depth == 0 ? pos : std::string::npos;
}

std::string extract_json_object_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return {};
    }

    size_t object_start = json.find('{', key_pos + pattern.size());
    if (object_start == std::string::npos) {
        return {};
    }

    size_t object_end = find_json_block_end(json, object_start);
    if (object_end == std::string::npos) {
        return {};
    }

    return json.substr(object_start, object_end - object_start);
}

std::vector<std::pair<std::string, std::string>> extract_schema_property_types(const std::string& schema) {
    std::vector<std::pair<std::string, std::string>> properties;
    std::string properties_object = extract_json_object_field(schema, "properties");
    if (properties_object.empty() || properties_object.size() < 2) {
        return properties;
    }

    size_t pos = 1;
    while (pos + 1 < properties_object.size()) {
        size_t key_start = properties_object.find('"', pos);
        if (key_start == std::string::npos || key_start + 1 >= properties_object.size()) {
            break;
        }
        size_t key_end = properties_object.find('"', key_start + 1);
        if (key_end == std::string::npos) {
            break;
        }

        std::string name = properties_object.substr(key_start + 1, key_end - key_start - 1);
        size_t value_start = properties_object.find('{', key_end);
        if (value_start == std::string::npos) {
            break;
        }
        size_t value_end = find_json_block_end(properties_object, value_start);
        if (value_end == std::string::npos) {
            break;
        }

        std::string value = properties_object.substr(value_start, value_end - value_start);
        std::string type = "string";
        std::string type_pattern = "\"type\":\"";
        size_t type_pos = value.find(type_pattern);
        if (type_pos != std::string::npos) {
            size_t type_start = type_pos + type_pattern.size();
            size_t type_end = value.find('"', type_start);
            if (type_end != std::string::npos) {
                type = value.substr(type_start, type_end - type_start);
            }
        } else if (value.find("\"enum\"") != std::string::npos) {
            type = "string";
        } else if (value.find("\"properties\"") != std::string::npos) {
            type = "object";
        }

        properties.emplace_back(std::move(name), std::move(type));
        pos = value_end;
    }

    return properties;
}

std::vector<std::string> extract_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> values;
    if (json.empty()) return values;
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern);
    if (key_pos == std::string::npos) return values;
    size_t arr_start = json.find('[', key_pos + pattern.size());
    if (arr_start == std::string::npos) return values;
    size_t arr_end = json.find(']', arr_start);
    if (arr_end == std::string::npos) return values;
    size_t pos = arr_start + 1;
    while (pos < arr_end) {
        size_t qs = json.find('"', pos);
        if (qs == std::string::npos || qs >= arr_end) break;
        size_t qe = json.find('"', qs + 1);
        if (qe == std::string::npos || qe > arr_end) break;
        values.push_back(json.substr(qs + 1, qe - qs - 1));
        pos = qe + 1;
    }
    return values;
}

std::vector<std::string> extract_schema_required(const std::string& schema) {
    return extract_string_array(schema, "required");
}

static bool json_object_has_key(const std::string& obj, const std::string& key) {
    const std::string target = "\"" + key + "\"";
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = 0; i < obj.size(); ++i) {
        char c = obj[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') {
            if (depth == 1 && obj.compare(i, target.size(), target) == 0) {
                size_t j = i + target.size();
                while (j < obj.size() && std::isspace(static_cast<unsigned char>(obj[j]))) ++j;
                if (j < obj.size() && obj[j] == ':') return true;
            }
            in_str = true;
            continue;
        }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
    }
    return false;
}

static bool function_calls_missing_required(const std::vector<std::string>& calls,
                                            const std::vector<ToolFunction>& tools) {
    if (calls.empty() || tools.empty()) return false;
    for (const auto& call : calls) {
        std::string name = json_string_field(call, "name");
        const ToolFunction* tool = nullptr;
        for (const auto& t : tools) {
            if (t.name == name) { tool = &t; break; }
        }
        if (!tool) continue;
        auto schema_it = tool->parameters.find("schema");
        if (schema_it == tool->parameters.end()) continue;
        std::vector<std::string> required = extract_schema_required(schema_it->second);
        if (required.empty()) continue;
        std::string args = extract_json_object_field(call, "arguments");
        for (const auto& req : required) {
            if (!json_object_has_key(args, req)) return true;
        }
    }
    return false;
}

std::string serialize_needle_tools(const std::vector<ToolFunction>& tools) {
    if (tools.empty()) return "[]";

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i > 0) oss << ",";

        oss << "{\"name\":\"" << escape_json_string(tools[i].name) << "\"";
        if (!tools[i].description.empty()) {
            oss << ",\"description\":\"" << escape_json_string(tools[i].description) << "\"";
        }

        oss << ",\"parameters\":";
        auto schema_it = tools[i].parameters.find("schema");
        if (schema_it == tools[i].parameters.end()) {
            oss << "{}";
        } else {
            auto properties = extract_schema_property_types(schema_it->second);
            auto required = extract_schema_required(schema_it->second);
            std::unordered_set<std::string> required_set(required.begin(), required.end());

            if (properties.empty()) {
                oss << "{}";
            } else {
                std::string properties_object = extract_json_object_field(schema_it->second, "properties");
                oss << "{";
                for (size_t p = 0; p < properties.size(); ++p) {
                    if (p > 0) oss << ",";
                    const std::string& param_name = properties[p].first;
                    const std::string& param_type = properties[p].second;
                    bool is_required = required_set.count(param_name) > 0;

                    std::string description;
                    std::string prop_obj = extract_json_object_field(properties_object, param_name);
                    if (!prop_obj.empty()) {
                        std::string desc_pattern = "\"description\":\"";
                        size_t desc_pos = prop_obj.find(desc_pattern);
                        if (desc_pos != std::string::npos) {
                            size_t desc_start = desc_pos + desc_pattern.size();
                            size_t desc_end = desc_start;
                            bool escaped = false;
                            while (desc_end < prop_obj.size()) {
                                if (escaped) {
                                    escaped = false;
                                    desc_end++;
                                    continue;
                                }
                                if (prop_obj[desc_end] == '\\') {
                                    escaped = true;
                                    desc_end++;
                                    continue;
                                }
                                if (prop_obj[desc_end] == '"') break;
                                desc_end++;
                            }
                            description = prop_obj.substr(desc_start, desc_end - desc_start);
                        }
                    }

                    oss << "\"" << escape_json_string(param_name) << "\":{";
                    oss << "\"type\":\"" << escape_json_string(param_type) << "\"";
                    if (!description.empty()) {
                        oss << ",\"description\":\"" << description << "\"";
                    }
                    oss << ",\"required\":" << (is_required ? "true" : "false") << "}";
                }
                oss << "}";
            }
        }
        oss << "}";
    }
    oss << "]";
    return oss.str();
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

    if (model_type == Config::ModelType::GEMMA4) {
        stop_token_sequences.push_back(tokenizer->encode("<turn|>"));
        if (has_tools) {
            stop_token_sequences.push_back(tokenizer->encode("<|tool_response>"));
        }
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

void reset_cache(CactusModelHandle* handle) {
    handle->model->reset_cache();
    handle->processed_tokens.clear();
    handle->processed_images.clear();
    handle->user_audio_counts.clear();
}

struct PrefillResult {
    std::vector<uint32_t> remaining_tokens;
    size_t prefilled_count = 0;
    bool was_prefix = false;
    bool was_exact_match = false;
};

struct EntropyState {
    float total_sum = 0.0f;
    size_t total_count = 0;

    void add(float entropy) {
        total_sum += entropy;
        total_count++;
    }

    float mean_confidence() const {
        return 1.0f - (total_sum / static_cast<float>(total_count));
    }
};

struct PreparedPrompt {
    InferenceOptions options;
    Config::ModelType model_type = Config::ModelType::GEMMA4;
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    std::vector<ChatMessage> messages;
    std::vector<ToolFunction> tools;
    std::string rendered;
    std::vector<uint32_t> tokens;
    size_t context_token_count = 0;
    std::vector<std::vector<CactusModelHandle::ProcessedImage>> images;

    std::vector<std::vector<float>> audio_features;
    size_t audio_num_frames = 0;

    bool has_images() const {
        return std::any_of(images.begin(), images.end(),
            [](const auto& msg_imgs) { return !msg_imgs.empty(); });
    }

    bool has_audio() const {
        return std::any_of(audio_features.begin(), audio_features.end(),
            [](const auto& mel) { return !mel.empty(); });
    }
};

CactusModelHandle::ProcessedImage image_signature(const std::string& image_path) {
    std::filesystem::path normalized_path(image_path);
    std::error_code ec;

    auto absolute_path = std::filesystem::absolute(normalized_path, ec);
    if (!ec) {
        normalized_path = absolute_path;
    }

    CactusModelHandle::ProcessedImage image;
    image.path = normalized_path.string();

    ec.clear();
    auto status = std::filesystem::status(normalized_path, ec);
    if (!ec && std::filesystem::is_regular_file(status)) {
        std::error_code time_ec;
        auto mtime = std::filesystem::last_write_time(normalized_path, time_ec);
        if (!time_ec) {
            image.last_modified_timestamp = static_cast<long long>(mtime.time_since_epoch().count());
        }
    }

    return image;
}

std::vector<std::vector<CactusModelHandle::ProcessedImage>> images_from_message(const std::vector<ChatMessage>& messages) {
    std::vector<std::vector<CactusModelHandle::ProcessedImage>> message_signatures;
    message_signatures.reserve(messages.size());

    for (const auto& message : messages) {
        std::vector<CactusModelHandle::ProcessedImage> image_signatures;
        image_signatures.reserve(message.images.size());
        for (const auto& image_path : message.images) {
            image_signatures.push_back(image_signature(image_path));
        }
        message_signatures.push_back(std::move(image_signatures));
    }

    return message_signatures;
}

bool image_context_prefix_matches(
    const std::vector<std::vector<CactusModelHandle::ProcessedImage>>& prefix,
    const std::vector<std::vector<CactusModelHandle::ProcessedImage>>& full
) {
    return prefix.size() <= full.size() &&
           std::equal(prefix.begin(), prefix.end(), full.begin());
}

bool prompt_context_matches(
    const CactusModelHandle* handle,
    const PreparedPrompt& prompt
) {
    if (handle->processed_tokens.empty()) {
        return false;
    }
    if (handle->model &&
        handle->model->get_config().model_type == Config::ModelType::NEEDLE &&
        prompt.tokens.size() != handle->processed_tokens.size() + 1) {
        return false;
    }
    if (prompt.context_token_count < handle->processed_tokens.size()) {
        return false;
    }
    if (!std::equal(handle->processed_tokens.begin(), handle->processed_tokens.end(), prompt.tokens.begin())) {
        return false;
    }
    if (prompt.has_images()) {
        return image_context_prefix_matches(handle->processed_images, prompt.images);
    }
    return !prompt.has_images();
}

PreparedPrompt prepare_prompt(
    CactusModelHandle* handle,
    const char* messages_json,
    const char* options_json,
    const char* tools_json,
    bool apply_tool_constraints,
    bool add_generation_prompt,
    const uint8_t* pcm_buffer = nullptr,
    size_t pcm_buffer_size = 0
) {
    if (!handle || !handle->model) {
        throw std::runtime_error("Invalid model handle");
    }

    PreparedPrompt prompt;
    prompt.options = parse_inference_options_json(options_json ? options_json : "");
    prompt.messages = parse_messages_json(messages_json, prompt.image_paths, &prompt.audio_paths);
    if (prompt.messages.empty()) {
        throw std::runtime_error("No messages provided");
    }

    inject_rag_context(handle, prompt.messages);

    if (tools_json && std::strlen(tools_json) > 0) {
        prompt.tools = parse_tools_json(tools_json);
    }

    if (prompt.options.tool_rag_top_k > 0 && prompt.tools.size() > prompt.options.tool_rag_top_k) {
        std::string query = extract_last_user_query(prompt.messages);
        if (!query.empty()) {
            prompt.tools = select_relevant_tools(handle, query, prompt.tools, prompt.options.tool_rag_top_k);
        }
    }

    auto* tokenizer = handle->model->get_tokenizer();
    if (!tokenizer) {
        throw std::runtime_error("Tokenizer unavailable");
    }

    prompt.model_type = handle->model->get_config().model_type;

    if (prompt.options.confidence_threshold < 0.0f) {
        if (handle->model->has_handoff_probe()) {
            // The Gemma4 probe returns p_wrong; confidence is 1 - p_wrong.
            // Route when the probe is less than 50% confident in local output.
            prompt.options.confidence_threshold = 0.50f;
        } else {
            float model_default = handle->model->get_config().default_cloud_handoff_threshold;
            prompt.options.confidence_threshold = (model_default > 0.0f) ? model_default : 0.7f;
        }
    }

    if (prompt.model_type == Config::ModelType::GEMMA4) {
        std::vector<size_t> user_indices;
        for (size_t i = 0; i < prompt.messages.size(); i++) {
            if (prompt.messages[i].role == "user") user_indices.push_back(i);
        }
        auto& counts = handle->user_audio_counts;
        if (counts.size() < user_indices.size()) counts.resize(user_indices.size(), 0);

        if (pcm_buffer != nullptr && pcm_buffer_size > 1 && !user_indices.empty()) {
            auto waveform_fp32 = cactus::audio::pcm_buffer_to_float_samples(pcm_buffer, pcm_buffer_size);
            auto samples_16k = resample_to_16k_fp32(waveform_fp32, 16000);
            if (!samples_16k.empty()) {
                auto audio_prep = cactus::audio::preprocess_audio_for_gemma4(samples_16k, handle->model->get_config());
                prompt.audio_features.push_back(std::move(audio_prep.features));
                size_t u = user_indices.size() - 1;
                prompt.messages[user_indices[u]].audio_soft_token_count = audio_prep.num_soft_tokens;
                counts[u] = audio_prep.num_soft_tokens;
            }
        } else if (!prompt.audio_paths.empty()) {
            for (size_t u = 0; u < user_indices.size(); u++) {
                const auto& msg = prompt.messages[user_indices[u]];
                if (msg.audio.empty()) continue;
                const std::string& audio_path = msg.audio.back();
                AudioFP32 wav = load_wav(audio_path);
                auto samples_16k = resample_to_16k_fp32(wav.samples, wav.sample_rate);
                if (samples_16k.empty()) continue;
                auto audio_prep = cactus::audio::preprocess_audio_for_gemma4(samples_16k, handle->model->get_config());
                prompt.audio_features.push_back(std::move(audio_prep.features));
                prompt.messages[user_indices[u]].audio_soft_token_count = audio_prep.num_soft_tokens;
                counts[u] = audio_prep.num_soft_tokens;
            }
        }
    }

    if (prompt.model_type == Config::ModelType::NEEDLE && !prompt.tools.empty()) {
        prompt.options.force_tools = true;
    }

    // Build the tool-definition block in the format each model family was trained on.
    std::string formatted_tools;
    if (prompt.model_type == Config::ModelType::NEEDLE) {
        formatted_tools = serialize_needle_tools(prompt.tools);
    } else if (tokenizer->is_qwen_family()) {
        formatted_tools = chat_tools::serialize_tools_qwen(prompt.tools);
    } else if (tokenizer->is_lfm2_family()) {
        formatted_tools = chat_tools::serialize_tools_lfm2(prompt.tools);
    } else {
        formatted_tools = gemma::format_tools(prompt.tools, !Config::is_gemma3_family(prompt.model_type));
    }

    if (apply_tool_constraints) {
        setup_tool_constraints(handle, prompt.tools, prompt.options.force_tools, prompt.options.temperature);
    }

    prompt.rendered = tokenizer->format_chat_prompt(
        prompt.messages,
        add_generation_prompt,
        formatted_tools,
        prompt.options.enable_thinking_if_supported
    );
    if (prompt.rendered.find("ERROR:") == 0) {
        throw std::runtime_error(prompt.rendered.substr(6));
    }
    prompt.tokens = tokenizer->encode(prompt.rendered);
    prompt.context_token_count = prompt.tokens.size();
    prompt.images = images_from_message(prompt.messages);
    return prompt;
}

PrefillResult do_prefill(
    CactusModelHandle* handle,
    const PreparedPrompt& prompt,
    const std::vector<uint32_t>& target_tokens
) {
    PrefillResult result = {};
    bool has_images = prompt.has_images();
    bool has_audio = prompt.has_audio();

    result.was_prefix = prompt_context_matches(handle, prompt);
    result.was_exact_match = result.was_prefix &&
        target_tokens.size() == handle->processed_tokens.size();

    if (result.was_exact_match) {
        return result;
    }

    std::vector<uint32_t> tokens_to_process;
    if (!result.was_prefix) {
        reset_cache(handle);
        tokens_to_process = target_tokens;
    } else {
        tokens_to_process.assign(
            target_tokens.begin() + handle->processed_tokens.size(),
            target_tokens.end()
        );
    }

    if (tokens_to_process.size() > 1) {
        std::vector<uint32_t> prefill_tokens(tokens_to_process.begin(), tokens_to_process.end() - 1);
        result.prefilled_count = prefill_tokens.size();

        auto slice_delta_audio = [&]() -> std::vector<std::vector<float>> {
            if (!result.was_prefix) return prompt.audio_features;
            const size_t cached_msg_count = handle->processed_images.size();
            size_t cached_audio_count = 0;
            for (size_t i = 0; i < cached_msg_count && i < prompt.messages.size(); ++i) {
                const auto& msg = prompt.messages[i];
                if (msg.role == "user" && !msg.audio.empty()) cached_audio_count++;
            }
            cached_audio_count = std::min(cached_audio_count, prompt.audio_features.size());
            return std::vector<std::vector<float>>(
                prompt.audio_features.begin() + cached_audio_count,
                prompt.audio_features.end());
        };

        if (has_images && has_audio) {
            std::vector<std::string> delta_image_paths;
            if (result.was_prefix) {
                size_t cached_image_count = 0;
                for (const auto& msg_imgs : handle->processed_images) {
                    cached_image_count += msg_imgs.size();
                }
                delta_image_paths.assign(
                    prompt.image_paths.begin() + cached_image_count,
                    prompt.image_paths.end()
                );
            } else {
                delta_image_paths = prompt.image_paths;
            }
            handle->model->prefill_with_media(prefill_tokens, delta_image_paths, slice_delta_audio());
        } else if (has_images) {
            std::vector<std::string> delta_image_paths;
            if (result.was_prefix) {
                size_t cached_image_count = 0;
                for (const auto& msg_imgs : handle->processed_images) {
                    cached_image_count += msg_imgs.size();
                }
                delta_image_paths.assign(
                    prompt.image_paths.begin() + cached_image_count,
                    prompt.image_paths.end()
                );
            } else {
                delta_image_paths = prompt.image_paths;
            }
            handle->model->prefill_with_images(prefill_tokens, delta_image_paths);
        } else if (has_audio) {
            handle->model->prefill_with_audio(prefill_tokens, slice_delta_audio());
        } else {
            handle->model->prefill(prefill_tokens, handle->model->get_prefill_chunk_size());
        }
        result.remaining_tokens = {tokens_to_process.back()};
    } else {
        result.remaining_tokens = tokens_to_process;
    }

    return result;
}

uint32_t decode(
    std::unique_ptr<Model>& model,
    const std::vector<uint32_t>& tokens,
    const InferenceOptions& options,
    float* out_entropy
) {
    return model->decode(tokens, options.temperature, options.top_p, options.top_k,
                         "", out_entropy, options.min_p, options.repetition_penalty);
}

uint32_t generate_first_token(
    CactusModelHandle* handle,
    const PrefillResult& prefill_result,
    const PreparedPrompt& prompt,
    float* first_token_entropy
) {
    if (prefill_result.was_exact_match || prefill_result.remaining_tokens.empty()) {
        if (handle->processed_tokens.empty()) {
            throw std::runtime_error("Cannot generate from empty prompt");
        }
        return decode(handle->model, {handle->processed_tokens.back()}, prompt.options, first_token_entropy);
    }
    return decode(handle->model, prefill_result.remaining_tokens, prompt.options, first_token_entropy);
}

std::string construct_prefill_response_json(
    bool success,
    const std::string* error,
    size_t prefill_tokens,
    double prefill_tps,
    double total_time_ms
) {
    std::ostringstream json;
    json << "{";
    json << "\"success\":" << (success ? "true" : "false") << ",";
    if (error) {
        json << "\"error\":\"" << escape_json_string(*error) << "\",";
    } else {
        json << "\"error\":null,";
    }
    json << "\"prefill_tokens\":" << prefill_tokens << ",";
    json << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ",";
    json << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
    json << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb();
    json << "}";
    return json.str();
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
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    struct MetalTrimGuard {
        ~MetalTrimGuard() { cactus_metal_trim_prefill_cache(); }
    } metal_trim_guard;

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
        CactusThreading::prepare_current_thread_for_cactus_work();
        auto start_time = std::chrono::high_resolution_clock::now();

        auto* handle = static_cast<CactusModelHandle*>(model);
        handle->should_stop = false;
        if (!handle->model->can_generate()) {
            const std::string err = "Model has no language-model decoder; cannot generate (embedding/encoder-only model)";
            CACTUS_LOG_ERROR("complete", err);
            handle_error_response(err, response_buffer, buffer_size);
            return -1;
        }
        auto* tokenizer = handle->model->get_tokenizer();
        auto prompt = prepare_prompt(handle, messages_json, options_json, tools_json, true, true, pcm_buffer, pcm_buffer_size);
        if (prompt.options.sample_seed != 0) {
            handle->model->set_sample_seed(prompt.options.sample_seed);
        }

        CACTUS_LOG_DEBUG("complete", "Prompt tokens: " << prompt.tokens.size()
            << ", max_tokens: " << prompt.options.max_tokens);

        bool has_images = prompt.has_images();
        bool has_audio = prompt.has_audio();
        const bool cloud_disabled = env_flag_enabled("CACTUS_DISABLE_CLOUD_HANDOFF");
        const bool cloud_eligible = !cloud_disabled && !handle->cloud_handoff_disabled &&
            prompt.options.auto_handoff && (!has_images || prompt.options.handoff_with_images);
        handle->model->reset_handoff_probe_rollout();
        const bool defer_local_stream_until_probe = cloud_eligible && handle->model->has_handoff_probe();
        bool pre_generation_cloud_attempted = false;

        std::string handoff_reason;
        if (!prompt.options.auto_handoff) {
            handoff_reason = "handoff off";
        } else if (cloud_disabled) {
            handoff_reason = "disabled (env)";
        } else if (handle->cloud_handoff_disabled) {
            handoff_reason = "disabled (auth failed earlier)";
        } else if (has_images && !prompt.options.handoff_with_images) {
            handoff_reason = "images kept local";
        }
        auto friendly_cloud_error = [](const std::string& e) -> std::string {
            if (e == "missing_api_key") return "no API key";
            if (e.rfind("http_401", 0) == 0) return "auth failed (401)";
            if (e.rfind("http_403", 0) == 0) return "auth failed (403)";
            if (e == "cloud_handoff_disabled") return "disabled";
            if (e == "curl_init_failed") return "curl init failed";
            if (e.empty()) return "cloud error";
            return e;
        };

        auto make_cloud_request = [&](const std::string& local_output_hint,
                                      const std::vector<std::string>& local_calls_hint) {
            CloudCompletionRequest request;
            request.messages = prompt.messages;
            request.tools = prompt.tools;
            request.local_output = local_output_hint;
            request.local_function_calls = local_calls_hint;
            request.has_images = has_images;
            request.has_audio = has_audio;
            if (has_audio && pcm_buffer != nullptr && pcm_buffer_size > 0) {
                request.audio_pcm.assign(pcm_buffer, pcm_buffer + pcm_buffer_size);
            }
            request.cloud_key = resolve_cloud_api_key(nullptr);
            return request;
        };

        auto disable_handoff_on_auth_failure = [&](const std::string& error) {
            if (error.rfind("http_401", 0) == 0 || error.rfind("http_403", 0) == 0) {
                handle->cloud_handoff_disabled = true;
                CACTUS_LOG_WARN("cloud_handoff", "Cloud auth failed (" << error
                    << "); disabling cloud handoff for this session");
            }
        };

        auto return_cloud_completion = [&](const CloudCompletionResult& cloud_result,
                                           double ttft_ms,
                                           double total_ms,
                                           float confidence,
                                           size_t prompt_token_count,
                                           const std::string& reason) {
            std::string cloud_response = cloud_result.response;
            std::vector<std::string> cloud_calls = cloud_result.function_calls;
            if (callback && !cloud_response.empty()) {
                callback(cloud_response.c_str(), 0, user_data);
            }
            std::string result = construct_response_json(cloud_response, cloud_calls, ttft_ms,
                                                         total_ms, 0.0, 0.0, prompt_token_count,
                                                         0, confidence, true, "", {}, "",
                                                         prompt.options.confidence_threshold, reason);
            if (result.length() >= buffer_size) {
                handle_error_response("Response buffer too small", response_buffer, buffer_size);
                return -1;
            }
            std::strcpy(response_buffer, result.c_str());

            cactus::telemetry::CompletionMetrics metrics{};
            metrics.success = true;
            metrics.cloud_handoff = true;
            metrics.ttft_ms = ttft_ms;
            metrics.prefill_tps = 0.0;
            metrics.decode_tps = 0.0;
            metrics.response_time_ms = total_ms;
            metrics.confidence = confidence;
            metrics.ram_usage_mb = get_ram_usage_mb();
            metrics.prefill_tokens = prompt_token_count;
            metrics.decode_tokens = 0;
            metrics.error_message = nullptr;
            metrics.function_calls_json = nullptr;
            cactus::telemetry::recordCompletion(handle->model_name.c_str(), metrics);
            return static_cast<int>(result.length());
        };

        if (cloud_eligible && prompt.options.confidence_threshold >= 1.0f) {
            pre_generation_cloud_attempted = true;
            CACTUS_LOG_INFO("cloud_handoff", "Cloud handoff triggered before local generation; waiting up to "
                << prompt.options.cloud_timeout_ms << " ms before falling back");
            auto cloud_result = cloud_complete_request(
                make_cloud_request("", {}),
                static_cast<long>(prompt.options.cloud_timeout_ms));
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1000.0;
            if (cloud_result.ok && (!cloud_result.response.empty() || !cloud_result.function_calls.empty())) {
                return return_cloud_completion(cloud_result, elapsed_ms, elapsed_ms, 0.0f, prompt.tokens.size(),
                                               "forced (threshold 100%)");
            }
            std::string cloud_error = cloud_result.error.empty() ? "cloud completion failed" : cloud_result.error;
            CACTUS_LOG_WARN("cloud_handoff", "Cloud completion failed before local generation: " << cloud_error);
            disable_handoff_on_auth_failure(cloud_error);
            handle_error_response(("cloud handoff failed before local generation: " + cloud_error).c_str(),
                                  response_buffer, buffer_size);
            return -1;
        }

        auto stop_token_sequences = build_stop_sequences(tokenizer, prompt.options.stop_sequences, prompt.model_type, !prompt.tools.empty());

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        float first_token_entropy = 0.0f;
        uint32_t next_token;
        size_t prompt_tokens;

        if (has_audio && !handle->processed_tokens.empty()) {
            auto& cache = handle->processed_tokens;
            size_t common = 0;
            size_t limit = std::min(cache.size(), prompt.tokens.size());
            while (common < limit && cache[common] == prompt.tokens[common]) common++;
            if (common < cache.size()) {
                CACTUS_LOG_WARN("complete", "KV cache diverges from new prompt at position " << common
                    << "/" << cache.size() << "; resetting cache for clean audio re-prefill");
                reset_cache(handle);
            }
        }

        bool first_token_from_prefill = false;
        if (!has_images && !has_audio && handle->processed_tokens.empty()) {
            reset_cache(handle);
            first_token_from_prefill = handle->model->prefill_and_sample_first_token(prompt.tokens, next_token, &first_token_entropy);
            if (first_token_from_prefill) {
                prompt_tokens = prompt.tokens.size();
            }
        }
        if (!first_token_from_prefill) {
            auto prefill_result = do_prefill(handle, prompt, prompt.tokens);
            prompt_tokens = prefill_result.prefilled_count + prefill_result.remaining_tokens.size();
            next_token = generate_first_token(handle, prefill_result, prompt, &first_token_entropy);
        }

        handle->processed_tokens = prompt.tokens;
        handle->processed_images = prompt.images;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        float confidence = 1.0f - first_token_entropy;
        std::string cloud_error;

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (prompt.options.force_tools && !prompt.tools.empty()) {
            handle->model->update_tool_constraints(next_token);
        }

        EntropyState entropy;
        entropy.add(first_token_entropy);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (cloud_eligible
                && !defer_local_stream_until_probe
                && !pre_generation_cloud_attempted
                && confidence < prompt.options.confidence_threshold) {
                CACTUS_LOG_INFO("cloud_handoff", "Cloud handoff triggered before local streaming; waiting up to "
                    << prompt.options.cloud_timeout_ms << " ms before falling back");
                CloudCompletionResult cloud_result = cloud_complete_request(
                    make_cloud_request("", {}),
                    static_cast<long>(prompt.options.cloud_timeout_ms));
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1000.0;
                if (cloud_result.ok && (!cloud_result.response.empty() || !cloud_result.function_calls.empty())) {
                    if (prompt.options.force_tools && !prompt.tools.empty()) {
                        handle->model->clear_tool_constraints();
                    }
                    return return_cloud_completion(cloud_result, elapsed_ms, elapsed_ms, confidence, prompt_tokens,
                                                   "low confidence");
                }
                cloud_error = cloud_result.error.empty() ? "cloud completion failed" : cloud_result.error;
                CACTUS_LOG_WARN("cloud_handoff", "Cloud completion failed before local streaming, falling back to local output: " << cloud_error);
                disable_handoff_on_auth_failure(cloud_error);
                handoff_reason = "handoff failed: " + friendly_cloud_error(cloud_error);
            }

            if (callback && !defer_local_stream_until_probe) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < prompt.options.max_tokens; i++) {
                if (handle->should_stop) break;

                float token_entropy = 0.0f;
                if (has_audio) {
                    uint32_t last_token = handle->processed_tokens.empty() ? next_token : handle->processed_tokens.back();
                    next_token = handle->model->decode_with_audio(
                        {last_token}, prompt.audio_features,
                        prompt.options.temperature, prompt.options.top_p, prompt.options.top_k,
                        "", &token_entropy,
                        prompt.options.min_p, prompt.options.repetition_penalty);
                } else {
                    next_token = decode(handle->model, {next_token}, prompt.options, &token_entropy);
                }
                handle->processed_tokens.push_back(next_token);
                generated_tokens.push_back(next_token);

                entropy.add(token_entropy);

                if (prompt.options.force_tools && !prompt.tools.empty()) {
                    handle->model->update_tool_constraints(next_token);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                    trim_stop_suffix(generated_tokens, stop_token_sequences, prompt.options.include_stop_sequences);
                    break;
                }

                if (callback && !defer_local_stream_until_probe) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        } else {
            trim_stop_suffix(generated_tokens, stop_token_sequences, prompt.options.include_stop_sequences);
        }

        if (defer_local_stream_until_probe) {
            confidence = entropy.mean_confidence();
            if (handle->model->has_handoff_probe_rollout()) {
                float wrong_probability = handle->model->handoff_probe_wrong_probability();
                if (std::isfinite(wrong_probability)) {
                    confidence = std::max(0.0f, std::min(1.0f, 1.0f - wrong_probability));
                    CACTUS_LOG_DEBUG("cloud_handoff", "Gemma4 handoff probe p_wrong="
                        << wrong_probability << " confidence=" << confidence);
                }
            }
        }

        if (prompt.options.force_tools && !prompt.tools.empty()) {
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
        parse_function_calls_from_response(response_text, regular_response, function_calls, prompt.tools);

        std::string thinking_text;
        if (prompt.model_type == Config::ModelType::GEMMA4 || prompt.options.enable_thinking_if_supported) {
            std::string stripped_content;
            partition_thinking_response(regular_response, thinking_text, stripped_content);
            regular_response = stripped_content;
            if (!prompt.options.enable_thinking_if_supported) {
                thinking_text.clear();
            }
        }

        std::string local_completion = regular_response;
        const bool parsed_empty_needle_tool_call =
            prompt.model_type == Config::ModelType::NEEDLE &&
            response_text.find("<tool_call>") != std::string::npos;
        if (local_completion.empty() && function_calls.empty() && !parsed_empty_needle_tool_call) {
            local_completion = response_text;
        }
        std::string primary_response = local_completion;
        std::vector<std::string> primary_function_calls = function_calls;

        bool handoff_succeeded = false;
        const bool low_confidence = defer_local_stream_until_probe
            && confidence < prompt.options.confidence_threshold;
        const bool invalid_local_tool_call = cloud_eligible && defer_local_stream_until_probe
            && function_calls_missing_required(function_calls, prompt.tools);
        if (!pre_generation_cloud_attempted && (low_confidence || invalid_local_tool_call)) {
            const char* trigger_reason = invalid_local_tool_call && !low_confidence
                ? "local tool call missing required args" : "low confidence (probe)";
            CACTUS_LOG_INFO("cloud_handoff", "Cloud handoff triggered (" << trigger_reason
                << "): p_wrong=" << (1.0f - confidence) << " confidence=" << confidence
                << "; waiting up to " << prompt.options.cloud_timeout_ms << " ms");
            CloudCompletionResult cloud_result = cloud_complete_request(
                make_cloud_request(local_completion, function_calls),
                static_cast<long>(prompt.options.cloud_timeout_ms));
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1000.0;
            if (cloud_result.ok && (!cloud_result.response.empty() || !cloud_result.function_calls.empty())) {
                if (prompt.options.force_tools && !prompt.tools.empty()) {
                    handle->model->clear_tool_constraints();
                }
                return return_cloud_completion(cloud_result, elapsed_ms, elapsed_ms, confidence, prompt_tokens,
                                               trigger_reason);
            }
            cloud_error = cloud_result.error.empty() ? "cloud completion failed" : cloud_result.error;
            CACTUS_LOG_WARN("cloud_handoff", "Cloud completion failed after probe handoff, falling back to local output: "
                << cloud_error);
            disable_handoff_on_auth_failure(cloud_error);
            handoff_reason = "handoff failed: " + friendly_cloud_error(cloud_error);
        }

        if (callback && defer_local_stream_until_probe && !primary_response.empty()) {
            callback(primary_response.c_str(), 0, user_data);
        }

        // No cloud handoff happened (success paths returned early above). If we never recorded a
        // reason, the request was eligible and the local model cleared the threshold.
        if (handoff_reason.empty()) {
            handoff_reason = (confidence >= prompt.options.confidence_threshold)
                ? "above threshold"
                : "kept local";
        }

        if (prompt.options.force_tools && !prompt.tools.empty() && primary_function_calls.empty() &&
            !parsed_empty_needle_tool_call) {
            CACTUS_LOG_WARN("force_tools", "force_tools was set but no tool call was parsed from the constrained output");
        }

        std::string result = construct_response_json(primary_response, primary_function_calls, time_to_first_token,
                                                     total_time_ms, prefill_tps, decode_tps, prompt_tokens,
                                                     completion_tokens, confidence, handoff_succeeded,
                                                     thinking_text, {}, response_text,
                                                     prompt.options.confidence_threshold, handoff_reason);

        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());

        std::string function_calls_json = serialize_function_calls(primary_function_calls);
        cactus::telemetry::CompletionMetrics metrics{};
        metrics.success = true;
        metrics.cloud_handoff = handoff_succeeded;
        metrics.ttft_ms = time_to_first_token;
        metrics.prefill_tps = prefill_tps;
        metrics.decode_tps = decode_tps;
        metrics.response_time_ms = total_time_ms;
        metrics.confidence = confidence;
        metrics.ram_usage_mb = get_ram_usage_mb();
        metrics.prefill_tokens = prompt_tokens;
        metrics.decode_tokens = completion_tokens;
        metrics.error_message = nullptr;
        metrics.function_calls_json = nullptr;
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
        metrics.function_calls_json = nullptr;
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
        metrics.function_calls_json = nullptr;
        auto* h = static_cast<CactusModelHandle*>(model);
        cactus::telemetry::recordCompletion(h ? h->model_name.c_str() : "unknown", metrics);

        return -1;
    }
}

int cactus_prefill(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!model) {
        std::string error_msg = last_error_message.empty()
            ? "Model not initialized. Check model path and files."
            : last_error_message;
        if (response_buffer && buffer_size > 0) {
            std::string result = construct_prefill_response_json(false, &error_msg, 0, 0.0, 0.0);
            if (result.size() < buffer_size) {
                std::strcpy(response_buffer, result.c_str());
            }
        }
        return -1;
    }

    if (!messages_json || !response_buffer || buffer_size == 0) {
        std::string error_msg = "Invalid parameters";
        if (response_buffer && buffer_size > 0) {
            std::string result = construct_prefill_response_json(false, &error_msg, 0, 0.0, 0.0);
            if (result.size() < buffer_size) {
                std::strcpy(response_buffer, result.c_str());
            }
        }
        return -1;
    }

    try {
        CactusThreading::prepare_current_thread_for_cactus_work();
        auto start_time = std::chrono::high_resolution_clock::now();

        auto* handle = static_cast<CactusModelHandle*>(model);
        if (!handle->model->can_generate()) {
            std::string err = "Model has no language-model decoder; cannot generate (embedding/encoder-only model)";
            CACTUS_LOG_ERROR("prefill", err);
            std::string result = construct_prefill_response_json(false, &err, 0, 0.0, 0.0);
            if (result.size() < buffer_size) std::strcpy(response_buffer, result.c_str());
            return -1;
        }
        auto prompt = prepare_prompt(handle, messages_json, options_json, tools_json, false, false, pcm_buffer, pcm_buffer_size);

        std::vector<uint32_t> context_tokens(prompt.tokens.begin(), prompt.tokens.begin() + prompt.context_token_count);
        auto prefill_result = do_prefill(handle, prompt, context_tokens);

        if (!prefill_result.was_exact_match) {
            handle->processed_tokens = context_tokens;
            if (!handle->processed_tokens.empty()) {
                handle->processed_tokens.pop_back();
            }
        }
        handle->processed_images = prompt.images;

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        double prefill_tps = (prefill_result.prefilled_count > 0 && elapsed_ms > 0.0)
            ? (static_cast<double>(prefill_result.prefilled_count) * 1000.0) / elapsed_ms
            : 0.0;

        std::string result = construct_prefill_response_json(true, nullptr, prefill_result.prefilled_count, prefill_tps, elapsed_ms);
        if (result.size() >= buffer_size) {
            std::string error_msg = "Response buffer too small";
            std::string error_json = construct_prefill_response_json(false, &error_msg, 0, 0.0, 0.0);
            if (error_json.size() < buffer_size) {
                std::strcpy(response_buffer, error_json.c_str());
            }
            return -1;
        }

        std::strcpy(response_buffer, result.c_str());
        return static_cast<int>(result.size());
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        std::string result = construct_prefill_response_json(false, &error_msg, 0, 0.0, 0.0);
        if (result.size() < buffer_size) {
            std::strcpy(response_buffer, result.c_str());
        }
        return -1;
    } catch (...) {
        std::string error_msg = "Unknown error during prefill";
        std::string result = construct_prefill_response_json(false, &error_msg, 0, 0.0, 0.0);
        if (result.size() < buffer_size) {
            std::strcpy(response_buffer, result.c_str());
        }
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

int cactus_render_prompt(
    cactus_model_t model,
    const char* messages_json,
    const char* options_json,
    const char* tools_json,
    char* prompt_buffer,
    size_t buffer_size
) {
    if (!model || !messages_json || !prompt_buffer || buffer_size == 0) return -1;

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto prompt = prepare_prompt(handle, messages_json, options_json, tools_json, false, true);
        if (prompt.rendered.size() >= buffer_size) return -2;
        std::strcpy(prompt_buffer, prompt.rendered.c_str());
        return static_cast<int>(prompt.rendered.size());
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
        CactusThreading::prepare_current_thread_for_cactus_work();
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

int cactus_benchmark_tokens(
    cactus_model_t model,
    const uint32_t* prompt_tokens,
    size_t prompt_token_len,
    size_t decode_token_len,
    char* response_buffer,
    size_t buffer_size
) {
    if (!model || !prompt_tokens || prompt_token_len == 0 || !response_buffer || buffer_size == 0) {
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    try {
        CactusThreading::prepare_current_thread_for_cactus_work();
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::vector<uint32_t> prompt(prompt_tokens, prompt_tokens + prompt_token_len);

        auto start_time = std::chrono::high_resolution_clock::now();
        double peak_ram_usage_mb = get_ram_usage_mb();
        auto sample_peak_ram = [&]() {
            peak_ram_usage_mb = std::max(peak_ram_usage_mb, get_ram_usage_mb());
        };
        handle->model->reset_cache();
        auto prefill_start_time = std::chrono::high_resolution_clock::now();
        size_t cache_prime_tokens = 0;
        uint32_t next_token = 0;
        bool first_token_from_prefill = false;
        if (decode_token_len == 0) {
            handle->model->prefill(prompt, handle->model->get_prefill_chunk_size(), "", false);
            cache_prime_tokens = prompt.size();
        } else {
            first_token_from_prefill = handle->model->prefill_and_sample_first_token(prompt, next_token);
            if (first_token_from_prefill) {
                cache_prime_tokens = prompt.size();
            } else if (prompt.size() > 1) {
                handle->model->prefill(
                    std::vector<uint32_t>(prompt.begin(), prompt.end() - 1),
                    handle->model->get_prefill_chunk_size()
                );
                cache_prime_tokens = prompt.size() - 1;
            }
        }
        auto prefill_end_time = std::chrono::high_resolution_clock::now();
        sample_peak_ram();
        auto first_decode_start_time = std::chrono::high_resolution_clock::now();
        if (decode_token_len > 0 && !first_token_from_prefill) {
            next_token = handle->model->decode({prompt.back()}, 0.0f, 1.0f, 1);
        }
        sample_peak_ram();
        auto first_token_time = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> completion_tokens;
        if (decode_token_len > 0) completion_tokens.push_back(next_token);
        size_t generated = decode_token_len > 0 ? 1 : 0;
        while (generated < decode_token_len) {
            next_token = handle->model->decode({next_token}, 0.0f, 1.0f, 1);
            completion_tokens.push_back(next_token);
            sample_peak_ram();
            ++generated;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double ttft_ms = std::chrono::duration_cast<std::chrono::microseconds>(first_token_time - start_time).count() / 1000.0;
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        double cache_prime_ms = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end_time - prefill_start_time).count() / 1000.0;
        double first_decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(first_token_time - first_decode_start_time).count() / 1000.0;
        double cache_state_copy_ms = handle->model->last_prefill_cache_copy_ms();
        double cache_prime_compute_ms = std::max(0.0, cache_prime_ms - cache_state_copy_ms);
        double decode_ms = std::max(0.0, total_ms - ttft_ms);
        double prefill_prepare_tps = (cache_prime_tokens > 0 && cache_prime_ms > 0.0)
            ? (static_cast<double>(cache_prime_tokens) * 1000.0) / cache_prime_ms
            : 0.0;
        double prefill_tps = (cache_prime_tokens > 0 && cache_prime_compute_ms > 0.0)
            ? (static_cast<double>(cache_prime_tokens) * 1000.0) / cache_prime_compute_ms
            : 0.0;
        double ttft_prompt_tps = ttft_ms > 0.0 ? (static_cast<double>(prompt_token_len) * 1000.0) / ttft_ms : 0.0;
        double decode_tps = (generated > 1 && decode_ms > 0.0) ? (static_cast<double>(generated - 1) * 1000.0) / decode_ms : 0.0;

        std::ostringstream oss;
        oss << "{"
            << "\"success\":true,"
            << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << ttft_ms << ","
            << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_ms << ","
            << "\"prefill_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ","
            << "\"prefill_compute_tps\":" << std::fixed << std::setprecision(2) << prefill_tps << ","
            << "\"prefill_prepare_tps\":" << std::fixed << std::setprecision(2) << prefill_prepare_tps << ","
            << "\"ttft_prompt_tps\":" << std::fixed << std::setprecision(2) << ttft_prompt_tps << ","
            << "\"cache_prime_ms\":" << std::fixed << std::setprecision(2) << cache_prime_ms << ","
            << "\"cache_prime_compute_ms\":" << std::fixed << std::setprecision(2) << cache_prime_compute_ms << ","
            << "\"cache_state_copy_ms\":" << std::fixed << std::setprecision(2) << cache_state_copy_ms << ","
            << "\"cache_prime_tokens\":" << cache_prime_tokens << ","
            << "\"first_decode_ms\":" << std::fixed << std::setprecision(2) << first_decode_ms << ","
            << "\"first_token_from_prefill\":" << (first_token_from_prefill ? "true" : "false") << ","
            << "\"prefill_padding_tokens\":" << handle->model->last_prefill_padding_tokens() << ","
            << "\"prefill_scalar_tail_tokens\":" << handle->model->last_prefill_scalar_tail_tokens() << ","
            << "\"prefill_tail_chunk_tokens\":" << handle->model->last_prefill_tail_chunk_tokens() << ","
            << "\"prefill_tail_padding_tokens\":" << handle->model->last_prefill_tail_padding_tokens() << ","
            << "\"decode_tps\":" << std::fixed << std::setprecision(2) << decode_tps << ","
            << "\"prompt_tokens\":" << prompt_token_len << ","
            << "\"completion_tokens\":" << generated << ","
            << "\"completion_token_ids\":[";
        for (size_t i = 0; i < completion_tokens.size(); ++i) {
            if (i) oss << ",";
            oss << completion_tokens[i];
        }
        oss << "],"
            << "\"peak_ram_usage_mb\":" << std::fixed << std::setprecision(2) << peak_ram_usage_mb << ","
            << "\"ram_usage_mb\":" << std::fixed << std::setprecision(2) << get_ram_usage_mb()
            << "}";

        std::string result = oss.str();
        if (result.size() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }
        std::strcpy(response_buffer, result.c_str());
        return static_cast<int>(result.size());
    } catch (const std::exception& e) {
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        handle_error_response("Unknown error during token benchmark", response_buffer, buffer_size);
        return -1;
    }
}

}
