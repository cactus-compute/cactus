#include "../cactus_engine.h"
#include "cloud.h"
#include "gemma4_mtp_decode.h"
#include "utils.h"
#include "telemetry.h"
#include "cactus_kernels.h"
#include "wav.h"
#include "../models/gemma4/gemma4_mtp_assistant.h"
#include "../models/gemma4/model_gemma4.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t DEFAULT_ROLLING_ENTROPY_WINDOW = 10;

namespace {

std::vector<std::pair<std::string, std::string>> extract_schema_property_types(const std::string& schema);
std::vector<std::string> extract_schema_required(const std::string& schema);

Gemma4Model* gemma4_language_model(Model* model) {
    if (auto* gemma4 = dynamic_cast<Gemma4Model*>(model)) {
        return gemma4;
    }
    if (auto* mm = dynamic_cast<Gemma4MmModel*>(model)) {
        return &mm->language_model();
    }
    return nullptr;
}

std::string default_gemma4_mtp_assistant_path(Model* model) {
    if (!model) {
        return {};
    }
    std::filesystem::path assistant_path = std::filesystem::path(model->get_model_folder_path()) / "assistant";
    return assistant_path.string();
}

std::string token_list_for_trace(const std::vector<uint32_t>& tokens) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) out << ",";
        out << tokens[i];
    }
    out << "]";
    return out.str();
}

std::string env_string(const char* key, const std::string& fallback = "") {
    const char* value = std::getenv(key);
    return value && value[0] ? std::string(value) : fallback;
}

std::string csv_escape_field(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

std::string bool_field(bool value) {
    return value ? "true" : "false";
}

std::string double_field(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

std::string join_ms(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ";";
        out << values[i];
    }
    return out.str();
}

void append_csv_line(const std::filesystem::path& path,
                     const std::string& header,
                     const std::vector<std::string>& fields) {
    bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Unable to open diagnostic trace CSV: " + path.string());
    }
    if (write_header) out << header << "\n";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) out << ",";
        out << csv_escape_field(fields[i]);
    }
    out << "\n";
}

struct DiagnosticTraceContext {
    bool enabled = false;
    std::filesystem::path dir;
    std::string run_id;
    std::string model;
    std::string prompt_name;
    std::string shape;
    std::string rep;

    static DiagnosticTraceContext from_env() {
        DiagnosticTraceContext ctx;
        std::string dir = env_string("CACTUS_GEMMA4_MTP_TRACE_CSV_DIR");
        if (dir.empty()) return ctx;
        ctx.run_id = env_string("CACTUS_MTP_DIAG_RUN_ID");
        if (ctx.run_id.empty()) return ctx;
        ctx.enabled = true;
        ctx.dir = dir;
        std::filesystem::create_directories(ctx.dir);
        ctx.model = env_string("CACTUS_MTP_DIAG_MODEL", "model");
        ctx.prompt_name = env_string("CACTUS_MTP_DIAG_PROMPT", "prompt");
        ctx.shape = env_string("CACTUS_MTP_DIAG_SHAPE", "unknown");
        ctx.rep = env_string("CACTUS_MTP_DIAG_REP", "0");
        return ctx;
    }

    void write_round(size_t round_index,
                     size_t generated_start,
                     size_t generated_end,
                     size_t target_batch_m,
                     size_t assistant_pass_count,
                     size_t drafted_tokens,
                     size_t accepted_drafts,
                     bool rejected,
                     bool alt_branch_accepted,
                     bool emitted_extra_target_token,
                     double target_forward_ms,
                     double assistant_total_ms,
                     const std::vector<double>& assistant_step_ms,
                     double sampling_or_argmax_ms,
                     double kv_transaction_ms,
                     double callback_stream_ms,
                     double loop_overhead_ms,
                     double round_total_ms) const {
        if (!enabled || generated_end <= generated_start) return;
        static const std::string header =
            "run_id,model,prompt_name,shape,rep,round_index,generated_token_start,generated_token_end,"
            "generated_tokens_emitted,target_batch_m,assistant_pass_count,drafted_tokens,accepted_drafts,"
            "rejected,alt_branch_accepted,emitted_extra_target_token,target_forward_ms,assistant_total_ms,"
            "assistant_step_ms,sampling_or_argmax_ms,kv_transaction_ms,callback_stream_ms,loop_overhead_ms,round_total_ms";
        append_csv_line(dir / "round_trace.csv", header, {
            run_id, model, prompt_name, shape, rep,
            std::to_string(round_index),
            std::to_string(generated_start),
            std::to_string(generated_end),
            std::to_string(generated_end - generated_start),
            std::to_string(target_batch_m),
            std::to_string(assistant_pass_count),
            std::to_string(drafted_tokens),
            std::to_string(accepted_drafts),
            bool_field(rejected),
            bool_field(alt_branch_accepted),
            bool_field(emitted_extra_target_token),
            double_field(target_forward_ms),
            double_field(assistant_total_ms),
            join_ms(assistant_step_ms),
            double_field(sampling_or_argmax_ms),
            double_field(kv_transaction_ms),
            double_field(callback_stream_ms),
            double_field(loop_overhead_ms),
            double_field(round_total_ms),
        });
    }

    void write_token(size_t token_position,
                     uint32_t token_id,
                     size_t round_index,
                     size_t token_index_in_round,
                     const std::string& source,
                     size_t round_emitted,
                     double target_forward_ms,
                     double assistant_total_ms,
                     double other_ms,
                     double round_total_ms) const {
        if (!enabled || round_emitted == 0) return;
        static const std::string header =
            "run_id,model,prompt_name,shape,rep,token_position,token_id,round_index,token_index_in_round,"
            "source,round_generated_tokens_emitted,round_target_forward_ms,round_assistant_total_ms,"
            "round_other_ms,round_total_ms,allocated_target_forward_ms,allocated_assistant_ms,"
            "allocated_other_ms,allocated_total_ms";
        double denom = static_cast<double>(round_emitted);
        append_csv_line(dir / "token_trace.csv", header, {
            run_id, model, prompt_name, shape, rep,
            std::to_string(token_position),
            std::to_string(token_id),
            std::to_string(round_index),
            std::to_string(token_index_in_round),
            source,
            std::to_string(round_emitted),
            double_field(target_forward_ms),
            double_field(assistant_total_ms),
            double_field(other_ms),
            double_field(round_total_ms),
            double_field(target_forward_ms / denom),
            double_field(assistant_total_ms / denom),
            double_field(other_ms / denom),
            double_field(round_total_ms / denom),
        });
    }
};

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
            spec.parameter_names.reserve(properties.size());
            for (const auto& [name, _] : properties) {
                spec.parameter_names.push_back(name);
            }
            spec.required_parameter_names = extract_schema_required(schema_it->second);
        }

        specs.push_back(std::move(spec));
    }

    return specs;
}

void strip_thinking_from_cache(CactusModelHandle* handle,
                               const std::vector<uint32_t>& generated_tokens,
                               size_t prompt_len) {
    const auto& cfg = handle->model->get_config();
    uint32_t open_id = cfg.channel_open_token_id;
    uint32_t close_id = cfg.channel_close_token_id;
    auto ranges = find_channel_token_ranges(generated_tokens, prompt_len,
                                            open_id, close_id);
    if (ranges.empty()) return;

    handle->model->remove_thinking_tokens(ranges);
    for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
        auto start = handle->processed_tokens.begin() + it->first;
        handle->processed_tokens.erase(start, start + it->second);
    }
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

std::vector<std::string> extract_schema_required(const std::string& schema) {
    std::vector<std::string> required;
    std::string key = "\"required\"";
    size_t key_pos = schema.find(key);
    if (key_pos == std::string::npos) return required;
    size_t arr_start = schema.find('[', key_pos + key.size());
    if (arr_start == std::string::npos) return required;
    size_t arr_end = schema.find(']', arr_start);
    if (arr_end == std::string::npos) return required;
    size_t pos = arr_start + 1;
    while (pos < arr_end) {
        size_t qs = schema.find('"', pos);
        if (qs == std::string::npos || qs >= arr_end) break;
        size_t qe = schema.find('"', qs + 1);
        if (qe == std::string::npos || qe > arr_end) break;
        required.push_back(schema.substr(qs + 1, qe - qs - 1));
        pos = qe + 1;
    }
    return required;
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
    std::vector<float> window;
    float window_sum = 0.0f;
    float total_sum = 0.0f;
    size_t total_count = 0;
    bool spike_handoff = false;
    size_t window_size = DEFAULT_ROLLING_ENTROPY_WINDOW;

    void add(float entropy) {
        window.push_back(entropy);
        window_sum += entropy;
        total_sum += entropy;
        total_count++;

        if (window.size() > window_size) {
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

struct PreparedPrompt {
    InferenceOptions options;
    Config::ModelType model_type = Config::ModelType::GEMMA4;
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    std::vector<ChatMessage> messages;
    std::vector<ToolFunction> tools;
    std::vector<uint32_t> tokens;
    size_t context_token_count = 0;
    std::vector<std::vector<CactusModelHandle::ProcessedImage>> images;

    std::vector<float> audio_features;
    size_t audio_num_frames = 0;

    bool has_images() const {
        return std::any_of(images.begin(), images.end(),
            [](const auto& msg_imgs) { return !msg_imgs.empty(); });
    }

    bool has_audio() const {
        return !audio_features.empty();
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
    if (prompt.context_token_count < handle->processed_tokens.size()) {
        return false;
    }
    size_t cache_size = handle->model->get_cache_size();
    size_t desired_cache_size = prompt.context_token_count > 0 ? prompt.context_token_count - 1 : 0;
    if (cache_size > handle->processed_tokens.size() || cache_size > desired_cache_size) {
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

    if (apply_tool_constraints) {
        setup_tool_constraints(handle, prompt.tools, prompt.options.force_tools, prompt.options.temperature);
    }

    auto* tokenizer = handle->model->get_tokenizer();
    if (!tokenizer) {
        throw std::runtime_error("Tokenizer unavailable");
    }

    prompt.model_type = handle->model->get_config().model_type;

    if (prompt.options.confidence_threshold < 0.0f) {
        float model_default = handle->model->get_config().default_cloud_handoff_threshold;
        prompt.options.confidence_threshold = (model_default > 0.0f) ? model_default : 0.7f;
    }

    if (prompt.model_type == Config::ModelType::GEMMA4) {
        std::vector<float> audio_samples;
        if (pcm_buffer != nullptr && pcm_buffer_size > 1) {
            auto waveform_fp32 = cactus::audio::pcm_buffer_to_float_samples(pcm_buffer, pcm_buffer_size);
            audio_samples = resample_to_16k_fp32(waveform_fp32, 16000);
        } else if (!prompt.audio_paths.empty()) {
            for (auto it = prompt.messages.rbegin(); it != prompt.messages.rend(); ++it) {
                if (!it->audio.empty()) {
                    const std::string& audio_path = it->audio.back();
                    AudioFP32 wav = load_wav(audio_path);
                    audio_samples = resample_to_16k_fp32(wav.samples, wav.sample_rate);
                    break;
                }
            }
        }
        std::vector<size_t> user_indices;
        for (size_t i = 0; i < prompt.messages.size(); i++) {
            if (prompt.messages[i].role == "user") user_indices.push_back(i);
        }
        auto& counts = handle->user_audio_counts;
        for (size_t u = 0; u + 1 < user_indices.size(); u++) {
            if (u < counts.size() && counts[u] > 0) {
                prompt.messages[user_indices[u]].audio_soft_token_count = counts[u];
            }
        }
        if (!audio_samples.empty() && !user_indices.empty()) {
            auto audio_prep = cactus::audio::preprocess_audio_for_gemma4(audio_samples, handle->model->get_config());
            prompt.audio_features = std::move(audio_prep.features);
            prompt.audio_num_frames = audio_prep.num_frames;
            size_t u = user_indices.size() - 1;
            prompt.messages[user_indices[u]].audio_soft_token_count = audio_prep.num_soft_tokens;
            if (counts.size() <= u) counts.resize(u + 1, 0);
            counts[u] = audio_prep.num_soft_tokens;
        }
    }

    std::string formatted_tools = gemma::format_tools(prompt.tools, true);

    {
        std::string full_prompt = tokenizer->format_chat_prompt(
            prompt.messages,
            add_generation_prompt,
            formatted_tools,
            prompt.options.enable_thinking_if_supported
        );
        if (full_prompt.find("ERROR:") == 0) {
            throw std::runtime_error(full_prompt.substr(6));
        }
        prompt.tokens = tokenizer->encode(full_prompt);
    }
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
    size_t cache_size = handle->model->get_cache_size();
    size_t desired_cache_size = target_tokens.empty() ? 0 : target_tokens.size() - 1;

    result.was_prefix = prompt_context_matches(handle, prompt);
    result.was_exact_match = result.was_prefix &&
        target_tokens.size() == handle->processed_tokens.size() &&
        cache_size == desired_cache_size;

    if (result.was_exact_match) {
        return result;
    }

    std::vector<uint32_t> tokens_to_process;
    if (!result.was_prefix) {
        reset_cache(handle);
        tokens_to_process = target_tokens;
    } else {
        size_t cached_tokens = std::min(cache_size, handle->processed_tokens.size());
        tokens_to_process.assign(
            target_tokens.begin() + cached_tokens,
            target_tokens.end()
        );
    }

    if (tokens_to_process.size() > 1) {
        std::vector<uint32_t> prefill_tokens(tokens_to_process.begin(), tokens_to_process.end() - 1);
        result.prefilled_count = prefill_tokens.size();
        if (has_images) {
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
        handle->should_stop = false;
        auto* tokenizer = handle->model->get_tokenizer();
        auto prompt = prepare_prompt(handle, messages_json, options_json, tools_json, true, true, pcm_buffer, pcm_buffer_size);
        if (prompt.options.mtp && prompt.options.mtp_assistant_path.empty()) {
            prompt.options.mtp_assistant_path = default_gemma4_mtp_assistant_path(handle->model.get());
        }

        bool mtp_assistant_available = false;
        if (prompt.options.mtp) {
            mtp_assistant_available = !prompt.options.mtp_assistant_path.empty()
                && std::filesystem::exists(prompt.options.mtp_assistant_path);
            if (prompt.options.mtp_required && !mtp_assistant_available) {
                throw std::runtime_error("Gemma 4 MTP assistant is unavailable: " + prompt.options.mtp_assistant_path);
            }
        }

        CACTUS_LOG_DEBUG("complete", "Prompt tokens: " << prompt.tokens.size()
            << ", max_tokens: " << prompt.options.max_tokens);

        bool has_images = prompt.has_images();
        bool has_audio = prompt.has_audio();
        Gemma4Model* mtp_target = gemma4_language_model(handle->model.get());
        Gemma4MtpAssistant mtp_assistant;
        bool mtp_enabled = false;
        size_t mtp_drafted_tokens = 0;
        size_t mtp_accepted_tokens = 0;
        size_t mtp_rejected_tokens = 0;
        size_t mtp_rounds = 0;
        double mtp_assistant_draft_ms = 0.0;
        double mtp_target_verify_ms = 0.0;
        double mtp_sampling_or_argmax_ms = 0.0;
        double mtp_kv_transaction_ms = 0.0;
        double mtp_callback_stream_ms = 0.0;
        std::string mtp_fallback_reason;
        if (prompt.options.mtp) {
            const bool text_only = !has_images && !has_audio && prompt.tools.empty();
            const bool short_completion_gate = !prompt.options.mtp_required
                && prompt.options.temperature == 0.0f
                && prompt.options.mtp_max_draft_tokens >= 2
                && prompt.options.max_tokens <= 32;
            if (short_completion_gate) {
                mtp_fallback_reason = "short_completion";
            } else if (!mtp_target) {
                mtp_fallback_reason = "target_not_gemma4_text";
            } else if (!text_only) {
                mtp_fallback_reason = "only_text_supported";
            } else if (!mtp_assistant_available) {
                mtp_fallback_reason = "assistant_unavailable";
            } else {
                auto* graph = static_cast<CactusGraph*>(mtp_target->graph_handle_);
                mtp_enabled = mtp_assistant.init(graph, prompt.options.mtp_assistant_path);
                if (!mtp_enabled) {
                    mtp_fallback_reason = "assistant_load_failed";
                }
            }
            if (prompt.options.mtp_required && !mtp_enabled) {
                throw std::runtime_error("Gemma 4 native MTP unavailable: " + mtp_fallback_reason);
            }
        }
        const bool has_gemma4_mixed_media = prompt.model_type == Config::ModelType::GEMMA4 && has_images && has_audio;
        auto decode_gemma4_mixed_media = [&](const std::vector<uint32_t>& tokens, float* out_entropy) -> uint32_t {
            auto* gemma4_mm = dynamic_cast<Gemma4MmModel*>(handle->model.get());
            if (!gemma4_mm) {
                throw std::runtime_error("Gemma4 mixed-media decode requested on non-Gemma4 multimodal model");
            }
            return gemma4_mm->decode_with_media(
                tokens,
                prompt.image_paths,
                prompt.audio_features,
                prompt.options.temperature, prompt.options.top_p, prompt.options.top_k,
                "", out_entropy,
                prompt.options.min_p, prompt.options.repetition_penalty
            );
        };

        auto stop_token_sequences = build_stop_sequences(tokenizer, prompt.options.stop_sequences, prompt.model_type, !prompt.tools.empty());

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        float first_token_entropy = 0.0f;
        uint32_t next_token;
        std::vector<__fp16> mtp_prev_hidden;
        std::mt19937 mtp_rng(prompt.options.seed == 0 ? std::random_device{}() : static_cast<uint32_t>(prompt.options.seed));
        const bool mtp_sparse_sampled = prompt.options.temperature > 0.0f && prompt.options.top_k > 0;
        const bool mtp_trace = env_flag_enabled("CACTUS_GEMMA4_MTP_TRACE");
        const DiagnosticTraceContext diag_trace = DiagnosticTraceContext::from_env();
        size_t diagnostic_round_index = 0;
        size_t prompt_tokens;

        if ((has_gemma4_mixed_media || has_audio) && !handle->processed_tokens.empty()) {
            auto& cache = handle->processed_tokens;
            size_t common = 0;
            size_t limit = std::min(cache.size(), prompt.tokens.size());
            while (common < limit && cache[common] == prompt.tokens[common]) common++;
            if (common < cache.size()) {
                CACTUS_LOG_WARN("complete", "KV cache diverges from new prompt at position " << common
                    << "/" << cache.size() << "; trimming and re-prefilling the divergent suffix");
                size_t kv_len = handle->model->get_cache_size();
                if (kv_len > common) {
                    handle->model->remove_thinking_tokens({{common, kv_len - common}});
                }
                cache.resize(common);
            }
        }

        if (has_gemma4_mixed_media) {
            prompt_tokens = prompt.tokens.size();
            next_token = decode_gemma4_mixed_media(prompt.tokens, &first_token_entropy);
        } else if (has_audio) {
            prompt_tokens = prompt.tokens.size();
            next_token = handle->model->decode_with_audio(
                prompt.tokens, prompt.audio_features,
                prompt.options.temperature, prompt.options.top_p, prompt.options.top_k,
                "", &first_token_entropy,
                prompt.options.min_p, prompt.options.repetition_penalty);
        } else {
            auto prefill_result = do_prefill(handle, prompt, prompt.tokens);
            prompt_tokens = prefill_result.prefilled_count + prefill_result.remaining_tokens.size();
            if (mtp_enabled && prompt.options.temperature == 0.0f) {
                std::vector<uint32_t> first_input;
                if (prefill_result.was_exact_match || prefill_result.remaining_tokens.empty()) {
                    if (handle->processed_tokens.empty()) {
                        throw std::runtime_error("Cannot generate from empty prompt");
                    }
                    first_input = {handle->processed_tokens.back()};
                } else {
                    first_input = prefill_result.remaining_tokens;
                }
                auto first = mtp_target->decode_greedy_with_hidden(first_input);
                next_token = first.token;
                mtp_prev_hidden = std::move(first.hidden);
            } else if (mtp_enabled) {
                std::vector<uint32_t> first_input;
                if (prefill_result.was_exact_match || prefill_result.remaining_tokens.empty()) {
                    if (handle->processed_tokens.empty()) {
                        throw std::runtime_error("Cannot generate from empty prompt");
                    }
                    first_input = {handle->processed_tokens.back()};
                } else {
                    first_input = prefill_result.remaining_tokens;
                }
                auto first = mtp_sparse_sampled
                    ? mtp_target->decode_tokens_with_hidden_and_sparse_probs(
                        first_input, prompt.options.temperature, prompt.options.top_p,
                        prompt.options.top_k, prompt.options.min_p)
                    : mtp_target->decode_tokens_with_hidden_and_probs(
                        first_input, prompt.options.temperature, prompt.options.top_p,
                        prompt.options.top_k, prompt.options.min_p);
                next_token = mtp_sparse_sampled
                    ? gemma4_mtp_sample_sparse_distribution(first.sparse_probabilities.back(), mtp_rng)
                    : gemma4_mtp_sample_distribution(first.probabilities.back(), mtp_rng);
                size_t hidden_row = first.hidden_dim == 0 ? 0 : first.hidden.size() / first.hidden_dim - 1;
                mtp_prev_hidden.assign(
                    first.hidden.begin() + hidden_row * first.hidden_dim,
                    first.hidden.begin() + (hidden_row + 1) * first.hidden_dim);
                mtp_target->record_sampled_token(next_token);
            } else {
                next_token = generate_first_token(handle, prefill_result, prompt, &first_token_entropy);
            }
        }

        handle->processed_tokens = prompt.tokens;
        handle->processed_images = prompt.images;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        float confidence = 1.0f - first_token_entropy;
        bool cloud_used = false;
        std::string cloud_error;
        std::future<CloudCompletionResult> cloud_future;
        bool cloud_future_started = false;
        const bool cloud_disabled = env_flag_enabled("CACTUS_DISABLE_CLOUD_HANDOFF");
        const bool cloud_eligible = !cloud_disabled &&
            prompt.options.auto_handoff && (!has_images || prompt.options.handoff_with_images);

        auto maybe_start_cloud_handoff = [&](const std::string& local_output_hint,
                                             const std::vector<std::string>& local_calls_hint) {
            if (!cloud_eligible || cloud_future_started) {
                return;
            }
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

            cloud_future_started = true;
            cloud_future = std::async(std::launch::async, [request, &prompt]() {
                return cloud_complete_request(request, static_cast<long>(prompt.options.cloud_timeout_ms));
            });
        };

        if (confidence < prompt.options.confidence_threshold) {
            maybe_start_cloud_handoff("", {});
        }

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (prompt.options.force_tools && !prompt.tools.empty()) {
            handle->model->update_tool_constraints(next_token);
        }

        EntropyState entropy;
        {
            size_t cfg_window = handle->model->get_config().default_rolling_entropy_window;
            if (cfg_window > 0) entropy.window_size = cfg_window;
        }
        entropy.add(first_token_entropy);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (callback) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            auto consume_token = [&](uint32_t token) {
                handle->processed_tokens.push_back(token);
                generated_tokens.push_back(token);
                entropy.add(0.0f);

                if (prompt.options.force_tools && !prompt.tools.empty()) {
                    handle->model->update_tool_constraints(token);
                }

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                    trim_stop_suffix(generated_tokens, stop_token_sequences, prompt.options.include_stop_sequences);
                    return true;
                }

                if (callback) {
                    std::string new_text = tokenizer->decode({token});
                    callback(new_text.c_str(), token, user_data);
                }
                return false;
            };

            auto run_standard_decode = [&](size_t produced) {
                for (size_t i = produced; i < prompt.options.max_tokens; i++) {
                    if (handle->should_stop) break;

                    float token_entropy = 0.0f;
                    size_t generated_start = generated_tokens.size();
                    auto round_start = std::chrono::steady_clock::now();
                    auto target_start = round_start;
                    if (has_gemma4_mixed_media) {
                        next_token = decode_gemma4_mixed_media(handle->processed_tokens, &token_entropy);
                    } else if (has_audio) {
                        next_token = handle->model->decode_with_audio(
                            handle->processed_tokens, prompt.audio_features,
                            prompt.options.temperature, prompt.options.top_p, prompt.options.top_k,
                            "", &token_entropy,
                            prompt.options.min_p, prompt.options.repetition_penalty);
                    } else {
                        next_token = decode(handle->model, {next_token}, prompt.options, &token_entropy);
                    }
                    auto target_end = std::chrono::steady_clock::now();
                    handle->processed_tokens.push_back(next_token);
                    generated_tokens.push_back(next_token);

                    entropy.add(token_entropy);

                    if (entropy.rolling_confidence() < prompt.options.confidence_threshold) {
                        entropy.spike_handoff = true;
                        maybe_start_cloud_handoff("", {});
                    }

                    if (prompt.options.force_tools && !prompt.tools.empty()) {
                        handle->model->update_tool_constraints(next_token);
                    }

                    if (matches_stop_sequence(generated_tokens, stop_token_sequences)) {
                        trim_stop_suffix(generated_tokens, stop_token_sequences, prompt.options.include_stop_sequences);
                        break;
                    }

                    double callback_ms = 0.0;
                    if (callback) {
                        auto callback_start = std::chrono::steady_clock::now();
                        std::string new_text = tokenizer->decode({next_token});
                        callback(new_text.c_str(), next_token, user_data);
                        auto callback_end = std::chrono::steady_clock::now();
                        callback_ms = std::chrono::duration<double, std::milli>(callback_end - callback_start).count();
                    }
                    auto round_end = std::chrono::steady_clock::now();
                    double target_ms = std::chrono::duration<double, std::milli>(target_end - target_start).count();
                    double round_total_ms = std::chrono::duration<double, std::milli>(round_end - round_start).count();
                    double loop_overhead_ms = round_total_ms - target_ms - callback_ms;
                    if (loop_overhead_ms < 0.0) loop_overhead_ms = 0.0;
                    size_t generated_end = generated_tokens.size();
                    size_t round_index = diagnostic_round_index++;
                    diag_trace.write_round(
                        round_index,
                        generated_start,
                        generated_end,
                        1,
                        0,
                        0,
                        0,
                        false,
                        false,
                        true,
                        target_ms,
                        0.0,
                        {},
                        0.0,
                        0.0,
                        callback_ms,
                        loop_overhead_ms,
                        round_total_ms);
                    if (generated_end > generated_start) {
                        double other_ms = callback_ms + loop_overhead_ms;
                        diag_trace.write_token(
                            generated_end - 1,
                            next_token,
                            round_index,
                            0,
                            "baseline_target",
                            generated_end - generated_start,
                            target_ms,
                            0.0,
                            other_ms,
                            round_total_ms);
                    }
                }
            };

            auto rebuild_mtp_cache_for_invariant_check = [&]() {
                if (!mtp_target || prompt.tokens.empty() || generated_tokens.empty()) return;
                mtp_target->reset_cache();
                if (prompt.tokens.size() > 1) {
                    std::vector<uint32_t> prompt_prefix(prompt.tokens.begin(), prompt.tokens.end() - 1);
                    mtp_target->prefill_for_mtp(prompt_prefix, mtp_target->get_prefill_chunk_size());
                }

                std::vector<uint32_t> committed;
                committed.reserve(prompt.tokens.size() + generated_tokens.size());
                committed.insert(committed.end(), prompt.tokens.begin(), prompt.tokens.end());
                committed.insert(committed.end(), generated_tokens.begin(), generated_tokens.end());
                for (size_t i = prompt.tokens.size() - 1; i + 1 < committed.size(); ++i) {
                    mtp_target->prefill_for_mtp({committed[i]}, 1);
                    mtp_target->record_sampled_token(committed[i + 1]);
                }
            };

            auto check_mtp_cache_position_invariant = [&]() {
                if (!mtp_target || prompt.tokens.empty() || generated_tokens.empty()) return;
                size_t expected_position = prompt.tokens.size() + generated_tokens.size() - 1;
                size_t actual_position = mtp_target->cache_position();
                if (actual_position != expected_position) {
                    throw std::runtime_error(
                        "Gemma 4 MTP cache position invariant failed: position "
                        + std::to_string(actual_position)
                        + " != committed prefix "
                        + std::to_string(expected_position));
                }
            };

            auto check_mtp_cache_invariant = [&](uint32_t last_token) {
                if (!prompt.options.mtp_cache_invariant_check
                    || !mtp_target
                    || prompt.options.temperature != 0.0f
                    || prompt.tokens.empty()
                    || generated_tokens.empty()) {
                    return;
                }
                CactusGraph* mtp_graph = static_cast<CactusGraph*>(mtp_target->graph_handle_);
                auto cache_state_nodes = mtp_target->cache_state_nodes_for_mtp();
                if (cache_state_nodes.empty()) return;

                size_t current_position = mtp_target->cache_position();
                auto current_txn = mtp_graph->begin_kv_cache_transaction(cache_state_nodes);
                uint32_t current_next = mtp_target->decode_greedy_with_hidden({last_token}).token;
                current_txn.rollback();
                mtp_graph->apply_pending_kv_cache_sequence_lengths();
                mtp_target->set_cache_position_for_mtp(current_position);

                rebuild_mtp_cache_for_invariant_check();
                size_t replay_position = mtp_target->cache_position();
                auto replay_cache_state_nodes = mtp_target->cache_state_nodes_for_mtp();
                auto replay_txn = mtp_graph->begin_kv_cache_transaction(replay_cache_state_nodes);
                uint32_t replay_next = mtp_target->decode_greedy_with_hidden({last_token}).token;
                replay_txn.rollback();
                mtp_graph->apply_pending_kv_cache_sequence_lengths();
                mtp_target->set_cache_position_for_mtp(replay_position);

                if (current_next != replay_next) {
                    throw std::runtime_error(
                        "Gemma 4 MTP cache invariant failed: live next token "
                        + std::to_string(current_next)
                        + " != replayed next token "
                        + std::to_string(replay_next));
                }
            };

            if (mtp_enabled) {
                size_t produced = 1;
                bool stopped = false;
                size_t mtp_previous_n_matches = 0;
                Gemma4MtpSamplingOptions mtp_sampling_options{
                    .temperature = prompt.options.temperature,
                    .top_p = prompt.options.top_p,
                    .top_k = prompt.options.top_k,
                    .min_p = prompt.options.min_p,
                };
                Gemma4MtpDecodeStrategy mtp_strategy(*mtp_target, mtp_assistant);
                while (produced < prompt.options.max_tokens && !handle->should_stop) {
                    size_t remaining = prompt.options.max_tokens - produced;
                    if (remaining == 0) break;
                    size_t generated_start = generated_tokens.size();
                    auto round_start = std::chrono::steady_clock::now();

                    size_t draft_limit = std::min<size_t>(
                        prompt.options.mtp_max_draft_tokens,
                        remaining > 1 ? remaining - 1 : 1);
                    auto cache_nodes = mtp_target->shared_cache_nodes_for_mtp();
                    const size_t assistant_position = handle->processed_tokens.empty() ? 0 : handle->processed_tokens.size() - 1;

                    auto round = mtp_strategy.decode_round(
                        next_token,
                        mtp_prev_hidden,
                        cache_nodes,
                        assistant_position,
                        draft_limit,
                        remaining,
                        mtp_sampling_options,
                        mtp_sparse_sampled,
                        mtp_rng);
                    mtp_rounds++;
                    auto& draft = round.draft;
                    auto& verified = round.verification;
                    mtp_drafted_tokens += draft.tokens.size();
                    mtp_accepted_tokens += verified.accepted;
                    mtp_assistant_draft_ms += draft.assistant_draft_ms;
                    mtp_target_verify_ms += verified.target_verify_ms;
                    mtp_sampling_or_argmax_ms += draft.sampling_or_argmax_ms + verified.sampling_or_argmax_ms;
                    mtp_kv_transaction_ms += verified.kv_transaction_ms;

                    if (mtp_trace) {
                        std::cerr
                            << "[mtp_trace] round=" << mtp_rounds
                            << " input_len=" << handle->processed_tokens.size()
                            << " previous_n_matches=" << mtp_previous_n_matches
                            << " assistant_position_id=" << assistant_position
                            << " assistant_input_token=" << next_token
                            << " assistant_draft_tokens=" << token_list_for_trace(draft.tokens)
                            << " target_verify_input_tokens=" << token_list_for_trace(verified.verify_tokens)
                            << " target_argmax_tokens=" << token_list_for_trace(verified.target_tokens)
                            << " accepted_count=" << verified.accepted
                            << " valid_committed_tokens=" << token_list_for_trace(verified.output_tokens)
                            << " target_cache_crop_length=" << verified.target_cache_crop_length
                            << " assistant_draft_ms=" << draft.assistant_draft_ms
                            << " target_verify_ms=" << verified.target_verify_ms
                            << " sampling_or_argmax_ms=" << (draft.sampling_or_argmax_ms + verified.sampling_or_argmax_ms)
                            << " kv_transaction_ms=" << verified.kv_transaction_ms
                            << "\n";
                    }
                    mtp_previous_n_matches = verified.accepted;

                    if (!verified.next_hidden.empty()) {
                        mtp_prev_hidden = std::move(verified.next_hidden);
                    }

                    if (verified.rejected) {
                        mtp_rejected_tokens++;
                    }

                    double round_callback_ms = 0.0;
                    for (uint32_t token : verified.output_tokens) {
                        next_token = token;
                        produced++;
                        if (prompt.options.temperature > 0.0f) {
                            mtp_target->record_sampled_token(token);
                        }
                        auto callback_start = std::chrono::steady_clock::now();
                        stopped = consume_token(token);
                        auto callback_end = std::chrono::steady_clock::now();
                        double callback_ms = std::chrono::duration<double, std::milli>(callback_end - callback_start).count();
                        round_callback_ms += callback_ms;
                        mtp_callback_stream_ms += callback_ms;
                        if (stopped || produced >= prompt.options.max_tokens) break;
                    }
                    auto round_end = std::chrono::steady_clock::now();
                    size_t generated_end = generated_tokens.size();
                    size_t emitted_count = generated_end > generated_start ? generated_end - generated_start : 0;
                    double round_total_ms = std::chrono::duration<double, std::milli>(round_end - round_start).count();
                    double round_sampling_ms = draft.sampling_or_argmax_ms + verified.sampling_or_argmax_ms;
                    double named_ms = draft.assistant_draft_ms + verified.target_verify_ms
                        + round_sampling_ms + verified.kv_transaction_ms + round_callback_ms;
                    double loop_overhead_ms = round_total_ms - named_ms;
                    if (loop_overhead_ms < 0.0) loop_overhead_ms = 0.0;
                    bool alt_branch_accepted = draft.has_alt_first_token
                        && !verified.output_tokens.empty()
                        && verified.output_tokens[0] == draft.alt_first_token
                        && (draft.tokens.empty() || draft.tokens[0] != draft.alt_first_token);
                    size_t round_index = diagnostic_round_index++;
                    diag_trace.write_round(
                        round_index,
                        generated_start,
                        generated_end,
                        verified.verify_tokens.size(),
                        draft.assistant_step_ms.size(),
                        draft.tokens.size() + (draft.has_alt_first_token ? 1 : 0),
                        verified.accepted,
                        verified.rejected,
                        alt_branch_accepted,
                        verified.emitted_extra_target_token,
                        verified.target_verify_ms,
                        draft.assistant_draft_ms,
                        draft.assistant_step_ms,
                        round_sampling_ms,
                        verified.kv_transaction_ms,
                        round_callback_ms,
                        loop_overhead_ms,
                        round_total_ms);
                    if (emitted_count > 0) {
                        std::vector<std::string> sources;
                        sources.reserve(emitted_count);
                        if (alt_branch_accepted) {
                            sources.push_back("accepted_alt_draft");
                            while (sources.size() < emitted_count) {
                                sources.push_back(prompt.options.temperature > 0.0f ? "sampled_target" : "free_target");
                            }
                        } else {
                            for (size_t i = 0; i < emitted_count; ++i) {
                                if (i < verified.accepted) {
                                    sources.push_back("accepted_main_draft");
                                } else if (verified.rejected) {
                                    sources.push_back("rejection_target");
                                } else {
                                    sources.push_back(prompt.options.temperature > 0.0f ? "sampled_target" : "free_target");
                                }
                            }
                        }
                        double other_ms = round_sampling_ms + verified.kv_transaction_ms + round_callback_ms + loop_overhead_ms;
                        for (size_t i = 0; i < emitted_count; ++i) {
                            diag_trace.write_token(
                                generated_start + i,
                                verified.output_tokens[i],
                                round_index,
                                i,
                                sources[i],
                                emitted_count,
                                verified.target_verify_ms,
                                draft.assistant_draft_ms,
                                other_ms,
                                round_total_ms);
                        }
                    }

                    if (!stopped && produced < prompt.options.max_tokens && !generated_tokens.empty()) {
                        check_mtp_cache_position_invariant();
                        check_mtp_cache_invariant(next_token);
                    }

                    if (stopped) break;
                }
            } else {
                run_standard_decode(1);
            }
        } else {
            trim_stop_suffix(generated_tokens, stop_token_sequences, prompt.options.include_stop_sequences);
        }

        confidence = entropy.mean_confidence();

        if (prompt.options.force_tools && !prompt.tools.empty()) {
            handle->model->clear_tool_constraints();
        }

        if (prompt.model_type == Config::ModelType::GEMMA4 && prompt.options.enable_thinking_if_supported && !generated_tokens.empty()) {
            strip_thinking_from_cache(handle, generated_tokens, prompt.tokens.size());
        }

        if (prompt.model_type == Config::ModelType::GEMMA4) {
            handle->model->compact_kv_cache();
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

        std::string thinking_text;
        if (prompt.model_type == Config::ModelType::GEMMA4 || prompt.options.enable_thinking_if_supported) {
            std::string stripped_content;
            strip_thinking_block(regular_response, thinking_text, stripped_content);
            regular_response = stripped_content;
            if (!prompt.options.enable_thinking_if_supported) {
                thinking_text.clear();
            }
        }

        if (confidence < prompt.options.confidence_threshold) {
            maybe_start_cloud_handoff(regular_response, function_calls);
        }

        std::string local_completion = regular_response;
        if (local_completion.empty() && function_calls.empty()) {
            local_completion = response_text;
        }
        std::string primary_response = local_completion;
        std::vector<std::string> primary_function_calls = function_calls;

        if (cloud_future_started) {
            auto status = cloud_future.wait_for(std::chrono::milliseconds(prompt.options.cloud_timeout_ms));
            if (status == std::future_status::ready) {
                CloudCompletionResult cloud_result = cloud_future.get();
                if (cloud_result.ok && (!cloud_result.response.empty() || !cloud_result.function_calls.empty())) {
                    cloud_used = true;
                    if (!cloud_result.response.empty()) {
                        primary_response = cloud_result.response;
                    }
                    if (!cloud_result.function_calls.empty()) {
                        primary_function_calls = cloud_result.function_calls;
                    }
                } else {
                    cloud_error = cloud_result.error.empty() ? "cloud completion failed" : cloud_result.error;
                    CACTUS_LOG_WARN("cloud_handoff", "Cloud completion failed, falling back to local output: " << cloud_error);
                }
            } else {
                cloud_error = "timeout";
                CACTUS_LOG_WARN("cloud_handoff", "Cloud completion timed out, falling back to local output: " << cloud_error);
            }
        }

        const bool handoff_succeeded = cloud_used;
        std::string mtp_json;
        if (prompt.options.mtp) {
            std::ostringstream mtp;
            mtp << "{";
            mtp << "\"requested\":true,";
            mtp << "\"enabled\":" << (mtp_enabled ? "true" : "false") << ",";
            mtp << "\"assistant_available\":" << (mtp_assistant_available ? "true" : "false") << ",";
            mtp << "\"max_draft_tokens\":" << prompt.options.mtp_max_draft_tokens << ",";
            mtp << "\"drafted_tokens\":" << mtp_drafted_tokens << ",";
            mtp << "\"accepted_tokens\":" << mtp_accepted_tokens << ",";
            mtp << "\"rejected_tokens\":" << mtp_rejected_tokens << ",";
            mtp << "\"rounds\":" << mtp_rounds << ",";
            mtp << std::fixed << std::setprecision(3);
            mtp << "\"assistant_draft_ms\":" << mtp_assistant_draft_ms << ",";
            mtp << "\"target_verify_ms\":" << mtp_target_verify_ms << ",";
            mtp << "\"sampling_or_argmax_ms\":" << mtp_sampling_or_argmax_ms << ",";
            mtp << "\"kv_transaction_ms\":" << mtp_kv_transaction_ms << ",";
            mtp << "\"callback_stream_ms\":" << mtp_callback_stream_ms << ",";
            mtp << "\"fallback_reason\":\"" << escape_json_string(mtp_enabled ? "" : mtp_fallback_reason) << "\"";
            mtp << "}";
            mtp_json = mtp.str();
        }
        std::string result = construct_response_json(primary_response, primary_function_calls, time_to_first_token,
                                                     total_time_ms, prefill_tps, decode_tps, prompt_tokens,
                                                     completion_tokens, confidence, handoff_succeeded,
                                                     thinking_text, {}, mtp_json);

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
        auto start_time = std::chrono::high_resolution_clock::now();

        auto* handle = static_cast<CactusModelHandle*>(model);
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
