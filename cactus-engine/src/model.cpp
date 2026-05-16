#include "engine.h"
#include "cactus_graph.h"
#include "mtp_sampler.h"
#include "mtp_decode.h"
#include "spec_decode.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace cactus {
namespace engine {

namespace {

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::string read_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("unable to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string find_transpiled_manifest_path(const std::string& model_folder) {
    std::string components_manifest = model_folder + "/components/manifest.json";
    if (file_exists(components_manifest)) {
        return components_manifest;
    }
    std::string root_manifest = model_folder + "/manifest.json";
    if (file_exists(root_manifest)) {
        return root_manifest;
    }
    return "";
}

size_t find_json_string_end(const std::string& json, size_t start) {
    bool escaped = false;
    for (size_t pos = start; pos < json.size(); ++pos) {
        char c = json[pos];
        if (escaped) {
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return pos;
        }
    }
    return std::string::npos;
}

std::string json_string_field(const std::string& json, const std::string& key, size_t start = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t colon = json.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return "";
    }
    size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) {
        return "";
    }
    size_t end = find_json_string_end(json, quote + 1);
    if (end == std::string::npos) {
        return "";
    }
    return json.substr(quote + 1, end - quote - 1);
}

std::vector<size_t> json_integer_array_field(const std::string& json, const std::string& key, size_t start = 0) {
    std::vector<size_t> values;
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return values;
    }
    size_t open = json.find('[', key_pos + pattern.size());
    size_t close = json.find(']', open == std::string::npos ? key_pos : open);
    if (open == std::string::npos || close == std::string::npos) {
        return values;
    }
    size_t pos = open + 1;
    while (pos < close) {
        while (pos < close && !std::isdigit(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        if (pos >= close) {
            break;
        }
        size_t end = pos;
        while (end < close && std::isdigit(static_cast<unsigned char>(json[end]))) {
            ++end;
        }
        values.push_back(static_cast<size_t>(std::stoull(json.substr(pos, end - pos))));
        pos = end;
    }
    return values;
}

int json_integer_field(const std::string& json, const std::string& key, size_t start = 0, int fallback = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    size_t colon = json.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    bool negative = pos < json.size() && json[pos] == '-';
    if (negative) {
        ++pos;
    }
    if (pos >= json.size() || !std::isdigit(static_cast<unsigned char>(json[pos]))) {
        return fallback;
    }
    int value = std::stoi(json.substr(pos));
    return negative ? -value : value;
}

std::vector<std::string> json_string_array_field(const std::string& json, const std::string& key, size_t start = 0) {
    std::vector<std::string> values;
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return values;
    }
    size_t open = json.find('[', key_pos + pattern.size());
    size_t close = json.find(']', open == std::string::npos ? key_pos : open);
    if (open == std::string::npos || close == std::string::npos) {
        return values;
    }
    size_t pos = open + 1;
    while (pos < close) {
        size_t quote = json.find('"', pos);
        if (quote == std::string::npos || quote >= close) {
            break;
        }
        size_t end = find_json_string_end(json, quote + 1);
        if (end == std::string::npos || end > close) {
            break;
        }
        values.push_back(json.substr(quote + 1, end - quote - 1));
        pos = end + 1;
    }
    return values;
}

size_t find_json_matching_brace(const std::string& json, size_t open) {
    if (open >= json.size() || json[open] != '{') {
        return std::string::npos;
    }
    int depth = 1;
    bool in_string = false;
    bool escaped = false;
    for (size_t pos = open + 1; pos < json.size(); ++pos) {
        char c = json[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return pos;
            }
        }
    }
    return std::string::npos;
}

std::vector<std::string> json_object_array_field(const std::string& json, const std::string& key, size_t start = 0) {
    std::vector<std::string> objects;
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return objects;
    }
    size_t open = json.find('[', key_pos + pattern.size());
    size_t close = json.find(']', open == std::string::npos ? key_pos : open);
    if (open == std::string::npos || close == std::string::npos) {
        return objects;
    }
    size_t pos = open + 1;
    while (pos < close) {
        size_t object_start = json.find('{', pos);
        if (object_start == std::string::npos || object_start >= close) {
            break;
        }
        size_t object_end = find_json_matching_brace(json, object_start);
        if (object_end == std::string::npos || object_end > close) {
            break;
        }
        objects.push_back(json.substr(object_start, object_end - object_start + 1));
        pos = object_end + 1;
    }
    return objects;
}

std::string dirname_of(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string join_path(const std::string& base, const std::string& rel) {
    if (rel.empty()) {
        return base;
    }
    if (!rel.empty() && rel[0] == '/') {
        return rel;
    }
    if (base.empty() || base == ".") {
        return rel;
    }
    return base + "/" + rel;
}

std::string bundle_root_for_manifest(const std::string& manifest_path) {
    std::string manifest_dir = dirname_of(manifest_path);
    size_t slash = manifest_dir.find_last_of("/\\");
    std::string dir_name = slash == std::string::npos ? manifest_dir : manifest_dir.substr(slash + 1);
    return dir_name == "components" ? dirname_of(manifest_dir) : manifest_dir;
}

size_t find_component_object(const std::string& manifest, const std::string& component_name) {
    size_t search = 0;
    while (true) {
        size_t component_pos = manifest.find("\"component\"", search);
        if (component_pos == std::string::npos) {
            return std::string::npos;
        }
        if (json_string_field(manifest, "component", component_pos) == component_name) {
            size_t object_start = manifest.rfind('{', component_pos);
            while (object_start != std::string::npos) {
                size_t object_end = find_json_matching_brace(manifest, object_start);
                if (object_end != std::string::npos && object_end > component_pos) {
                    return object_start;
                }
                if (object_start == 0) {
                    break;
                }
                object_start = manifest.rfind('{', object_start - 1);
            }
            return component_pos;
        }
        search = component_pos + 1;
    }
}

size_t find_json_object_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    return json.find('{', key_pos + pattern.size());
}

std::string json_object_field(const std::string& json, const std::string& key, size_t start = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t key_pos = json.find(pattern, start);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t open = json.find('{', key_pos + pattern.size());
    if (open == std::string::npos) {
        return "";
    }
    size_t close = find_json_matching_brace(json, open);
    if (close == std::string::npos) {
        return "";
    }
    return json.substr(open, close - open + 1);
}

bool has_string_value(const std::vector<std::string>& values, const std::string& expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

bool has_all_string_values(const std::vector<std::string>& values, std::initializer_list<const char*> expected) {
    for (const char* value : expected) {
        if (!has_string_value(values, value)) {
            return false;
        }
    }
    return true;
}

SpecDecodeManifest parse_spec_decode_manifest_from_json(const std::string& manifest) {
    size_t spec_pos = find_json_object_field(manifest, "spec_decode");
    if (spec_pos == std::string::npos) {
        return {};
    }
    const std::string spec_json = json_object_field(manifest, "spec_decode");
    const std::string target_json = json_object_field(spec_json, "target");
    const std::string target_shared_kv = json_object_field(target_json, "shared_kv");
    const std::string target_full_attention = json_object_field(target_shared_kv, "full_attention");
    const std::string target_sliding_attention = json_object_field(target_shared_kv, "sliding_attention");
    const std::string assistant_json = json_object_field(spec_json, "assistant");
    const std::string assistant_shared_kv = json_object_field(assistant_json, "shared_kv");
    const std::string assistant_full_attention = json_object_field(assistant_shared_kv, "full_attention");
    const std::string assistant_sliding_attention = json_object_field(assistant_shared_kv, "sliding_attention");
    SpecDecodeManifest spec;
    size_t version_key = spec_json.find("\"version\"");
    if (version_key != std::string::npos) {
        size_t colon = spec_json.find(':', version_key);
        if (colon != std::string::npos) {
            size_t pos = colon + 1;
            while (pos < spec_json.size() && std::isspace(static_cast<unsigned char>(spec_json[pos]))) {
                ++pos;
            }
            if (pos < spec_json.size() && std::isdigit(static_cast<unsigned char>(spec_json[pos]))) {
                spec.version = std::stoi(spec_json.substr(pos));
            }
        }
    }
    spec.method = json_string_field(spec_json, "method");
    spec.target.verifier_logits = json_string_field(target_json, "verifier_logits");
    spec.target.target_hidden_state = json_string_field(target_json, "target_hidden_state");
    spec.target.target_token_embedding = json_string_field(target_json, "target_token_embedding");
    spec.target.shared_kv_full_key = json_string_field(target_full_attention, "key");
    spec.target.shared_kv_full_value = json_string_field(target_full_attention, "value");
    spec.target.shared_kv_sliding_key = json_string_field(target_sliding_attention, "key");
    spec.target.shared_kv_sliding_value = json_string_field(target_sliding_attention, "value");
    spec.assistant.current_token_embedding = json_string_field(assistant_json, "current_token_embedding");
    spec.assistant.previous_target_hidden = json_string_field(assistant_json, "previous_target_hidden");
    spec.assistant.position = json_string_field(assistant_json, "position");
    spec.assistant.shared_kv_full_key = json_string_field(assistant_full_attention, "key");
    spec.assistant.shared_kv_full_value = json_string_field(assistant_full_attention, "value");
    spec.assistant.shared_kv_sliding_key = json_string_field(assistant_sliding_attention, "key");
    spec.assistant.shared_kv_sliding_value = json_string_field(assistant_sliding_attention, "value");
    spec.assistant.logits_output = json_string_field(assistant_json, "logits_output");
    spec.assistant.next_hidden_output = json_string_field(assistant_json, "next_hidden_output");
    validate_spec_decode_manifest(spec);
    return spec;
}

std::vector<float> logits_row_to_fp32(CactusGraph* gb, size_t logits_node, size_t row) {
    const auto& logits_buf = gb->get_output_buffer(logits_node);
    if (logits_buf.shape.empty()) {
        throw std::runtime_error("transpiled decoder logits output is empty");
    }
    const size_t vocab_size = logits_buf.shape.back();
    size_t row_count = 1;
    if (logits_buf.shape.size() >= 2) {
        row_count = logits_buf.shape[logits_buf.shape.size() - 2];
    }
    if (row >= row_count) {
        row = row_count - 1;
    }
    const size_t row_offset = row * vocab_size;
    void* logits_ptr = gb->get_output(logits_node);
    std::vector<float> logits(vocab_size);
    if (logits_buf.precision == Precision::FP32) {
        const float* src = static_cast<const float*>(logits_ptr) + row_offset;
        std::copy(src, src + vocab_size, logits.begin());
    } else if (logits_buf.precision == Precision::FP16) {
        const __fp16* src = static_cast<const __fp16*>(logits_ptr) + row_offset;
        Quantization::fp16_to_fp32(const_cast<__fp16*>(src), logits.data(), vocab_size);
    } else {
        const int8_t* src = static_cast<const int8_t*>(logits_ptr) + row_offset;
        Quantization::int8_to_fp32(const_cast<int8_t*>(src), logits.data(), vocab_size, 1.0f);
    }
    return logits;
}

struct LoadedNpyArray {
    std::vector<size_t> shape;
    Precision precision = Precision::FP32;
    std::vector<uint8_t> bytes;
};

LoadedNpyArray load_npy_array(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("unable to open npy constant: " + path);
    }
    char magic[6];
    file.read(magic, 6);
    if (file.gcount() != 6 || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
        throw std::runtime_error("invalid npy constant header: " + path);
    }
    char version[2];
    file.read(version, 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t header16 = 0;
        file.read(reinterpret_cast<char*>(&header16), sizeof(header16));
        header_len = header16;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    std::string header(header_len, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (file.gcount() != static_cast<std::streamsize>(header.size())) {
        throw std::runtime_error("truncated npy constant header: " + path);
    }

    LoadedNpyArray result;
    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": True") != std::string::npos) {
        throw std::runtime_error("Fortran-order npy constants are not supported: " + path);
    }
    if (header.find("<f2") != std::string::npos || header.find("|f2") != std::string::npos) {
        result.precision = Precision::FP16;
    } else if (header.find("<f4") != std::string::npos || header.find("|f4") != std::string::npos) {
        result.precision = Precision::FP32;
    } else if (header.find("|i1") != std::string::npos || header.find("<i1") != std::string::npos) {
        result.precision = Precision::INT8;
    } else {
        throw std::runtime_error("unsupported npy constant dtype: " + path);
    }

    size_t shape_key = header.find("shape");
    size_t open = header.find('(', shape_key);
    size_t close = header.find(')', open);
    if (shape_key == std::string::npos || open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("npy constant is missing shape: " + path);
    }
    size_t pos = open + 1;
    while (pos < close) {
        while (pos < close && !std::isdigit(static_cast<unsigned char>(header[pos]))) {
            ++pos;
        }
        if (pos >= close) {
            break;
        }
        size_t end = pos;
        while (end < close && std::isdigit(static_cast<unsigned char>(header[end]))) {
            ++end;
        }
        result.shape.push_back(static_cast<size_t>(std::stoull(header.substr(pos, end - pos))));
        pos = end;
    }
    if (result.shape.empty()) {
        throw std::runtime_error("npy constant has empty shape: " + path);
    }

    size_t element_count = 1;
    for (size_t dim : result.shape) {
        element_count *= dim;
    }
    size_t bytes = element_count * PrecisionTraits::size_of(result.precision);
    result.bytes.resize(bytes);
    file.read(reinterpret_cast<char*>(result.bytes.data()), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("truncated npy constant data: " + path);
    }
    return result;
}

Precision precision_from_manifest_value(int value, const std::string& path) {
    switch (value) {
        case 0: return Precision::INT8;
        case 1: return Precision::FP16;
        case 2: return Precision::FP32;
        default:
            throw std::runtime_error("unsupported bound constant precision " + std::to_string(value) + ": " + path);
    }
}

std::string resolve_manifest_path(const std::string& bundle_root,
                                  const std::string& manifest_dir,
                                  const std::string& rel) {
    std::string path = join_path(bundle_root, rel);
    if (file_exists(path)) {
        return path;
    }
    path = join_path(manifest_dir, rel);
    if (file_exists(path)) {
        return path;
    }
    return join_path(bundle_root, rel);
}

class TranspiledCausalLmModel final : public Model {
public:
    explicit TranspiledCausalLmModel(std::string manifest_path)
        : manifest_path_(std::move(manifest_path)) {}

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override {
        (void)do_warmup;
        return Model::init(model_folder, context_size, system_prompt, false);
    }

    uint32_t decode(const std::vector<uint32_t>& tokens, float temperature = -1.0f, float top_p = -1.0f,
                    size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                    float min_p = 0.15f, float repetition_penalty = 1.1f) override {
        (void)profile_file;
        (void)repetition_penalty;
        if (tokens.empty()) {
            throw std::runtime_error("transpiled causal LM decode requires at least one token");
        }
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        size_t logits_node = run_decoder(gb, context_tokens_);
        std::vector<float> logits = logits_row_to_fp32(gb, logits_node, context_tokens_.empty() ? 0 : context_tokens_.size() - 1);

        MtpSamplingOptions options;
        options.temperature = temperature < 0.0f ? config_.default_temperature : temperature;
        options.top_p = top_p < 0.0f ? config_.default_top_p : top_p;
        options.top_k = top_k == 0 ? config_.default_top_k : top_k;
        options.min_p = min_p;
        if (options.temperature < 0.0f) {
            options.temperature = 0.0f;
        }
        MtpDistribution dist = mtp_distribution_from_logits(std::move(logits), options);
        uint32_t next_token = mtp_argmax(dist);
        if (out_entropy) {
            double entropy = 0.0;
            for (float p : dist.probabilities) {
                if (p > 0.0f) {
                    entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
                }
            }
            *out_entropy = static_cast<float>(entropy);
        }
        record_sampled_token(next_token);
        return next_token;
    }

    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256, const std::string& profile_file = "") override {
        (void)chunk_size;
        (void)profile_file;
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());
    }

    void reset_cache() override {
        context_tokens_.clear();
    }

    std::string speculative_decode_status() const override {
        return speculative_status_;
    }

    SpeculativeDecodeResult speculative_decode(
        const std::vector<uint32_t>& tokens,
        size_t max_tokens,
        size_t draft_tokens,
        float temperature = -1.0f,
        float top_p = -1.0f,
        size_t top_k = 0,
        float min_p = 0.15f,
        float repetition_penalty = 1.1f) override {
        (void)repetition_penalty;
        if (!speculative_status_.empty()) {
            throw std::runtime_error("MTP requested but unavailable: " + speculative_status_);
        }
        if (tokens.empty() || max_tokens == 0) {
            return {};
        }
        context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.end());

        MtpSamplingOptions options;
        options.temperature = temperature < 0.0f ? config_.default_temperature : temperature;
        options.top_p = top_p < 0.0f ? config_.default_top_p : top_p;
        options.top_k = top_k == 0 ? config_.default_top_k : top_k;
        options.min_p = min_p;

        SpeculativeDecodeResult result;
        const size_t limit = std::max<size_t>(1, draft_tokens);
        while (result.tokens.size() < max_tokens) {
            std::vector<uint32_t> base_context = context_tokens_;
            CactusGraph* target_graph = static_cast<CactusGraph*>(graph_handle_);
            run_decoder(target_graph, base_context);
            std::vector<float> current_embedding = run_target_embedding(base_context.back());
            std::vector<float> previous_hidden = output_row_to_fp32(
                target_graph,
                output_node_for_role(decoder_logical_outputs_, output_node_ids_, "target_hidden_state"),
                base_context.empty() ? 0 : base_context.size() - 1);
            std::vector<float> full_key = output_tensor_to_fp32(
                target_graph,
                output_node_for_role(decoder_logical_outputs_, output_node_ids_, "shared_kv.full_attention.key"));
            std::vector<float> full_value = output_tensor_to_fp32(
                target_graph,
                output_node_for_role(decoder_logical_outputs_, output_node_ids_, "shared_kv.full_attention.value"));
            std::vector<float> sliding_key = output_tensor_to_fp32(
                target_graph,
                output_node_for_role(decoder_logical_outputs_, output_node_ids_, "shared_kv.sliding_attention.key"));
            std::vector<float> sliding_value = output_tensor_to_fp32(
                target_graph,
                output_node_for_role(decoder_logical_outputs_, output_node_ids_, "shared_kv.sliding_attention.value"));
            MtpDraftBatch draft;
            draft.tokens.reserve(limit);
            draft.probabilities.reserve(limit);

            for (size_t i = 0; i < limit && result.tokens.size() + draft.tokens.size() < max_tokens; ++i) {
                size_t logits_node = run_assistant_graph(
                    current_embedding,
                    previous_hidden,
                    full_key,
                    full_value,
                    sliding_key,
                    sliding_value,
                    base_context.size() + i);
                std::vector<float> logits = logits_row_to_fp32(
                    assistant_graph_.get(),
                    logits_node,
                    0);
                MtpDistribution dist = mtp_distribution_from_logits(std::move(logits), options);
                uint32_t token = mtp_argmax(dist);
                draft.tokens.push_back(token);
                draft.probabilities.push_back(std::move(dist));
                previous_hidden = output_tensor_to_fp32(
                    assistant_graph_.get(),
                    output_node_for_role(assistant_logical_outputs_, assistant_output_node_ids_, "next_hidden_output"));
                current_embedding = run_target_embedding(token);
            }
            if (draft.tokens.empty()) {
                break;
            }

            std::vector<uint32_t> verify_context = base_context;
            verify_context.insert(verify_context.end(), draft.tokens.begin(), draft.tokens.end());
            size_t target_logits_node = run_decoder(static_cast<CactusGraph*>(graph_handle_), verify_context);
            std::vector<MtpDistribution> target_distributions;
            target_distributions.reserve(draft.tokens.size() + 1);
            size_t first_row = base_context.empty() ? 0 : base_context.size() - 1;
            for (size_t i = 0; i <= draft.tokens.size(); ++i) {
                auto logits = logits_row_to_fp32(static_cast<CactusGraph*>(graph_handle_), target_logits_node, first_row + i);
                target_distributions.push_back(mtp_distribution_from_logits(std::move(logits), options));
            }

            MtpVerificationResult verified = verify_greedy_mtp_draft(draft, target_distributions);
            result.drafted_tokens += draft.tokens.size();
            result.accepted_draft_tokens += verified.accepted_draft_tokens;
            result.rejected_tokens += verified.rejected ? 1 : 0;

            for (uint32_t token : verified.output_tokens) {
                if (result.tokens.size() >= max_tokens) {
                    break;
                }
                result.tokens.push_back(token);
                context_tokens_.push_back(token);
                record_sampled_token(token);
            }
        }
        return result;
    }

protected:
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override {
        (void)use_cache;
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        return run_decoder(gb, tokens);
    }

    void load_weights_to_graph(CactusGraph* gb) override {
        const std::string manifest = read_text_file(manifest_path_);
        if (json_string_field(manifest, "task") != "causal_lm_logits") {
            throw std::runtime_error("transpiled C++ model only supports causal_lm_logits bundles");
        }
        const size_t decoder_pos = find_component_object(manifest, "decoder");
        if (decoder_pos == std::string::npos) {
            throw std::runtime_error("transpiled causal LM bundle missing decoder component");
        }
        const std::string decoder_json = manifest.substr(decoder_pos, find_json_matching_brace(manifest, decoder_pos) - decoder_pos + 1);
        const std::string graph_rel = json_string_field(decoder_json, "graph");
        if (graph_rel.empty()) {
            throw std::runtime_error("transpiled causal LM bundle missing decoder graph");
        }
        runtime_input_node_ids_ = json_integer_array_field(decoder_json, "runtime_input_node_ids");
        output_node_ids_ = json_integer_array_field(decoder_json, "output_node_ids");
        if (runtime_input_node_ids_.empty()) {
            throw std::runtime_error("transpiled causal LM bundle decoder graph has no runtime input node ids");
        }
        if (output_node_ids_.empty()) {
            throw std::runtime_error("transpiled causal LM bundle decoder graph has no output node ids");
        }

        const std::string bundle_root = bundle_root_for_manifest(manifest_path_);
        const std::string manifest_dir = dirname_of(manifest_path_);
        std::string graph_path = join_path(bundle_root, graph_rel);
        if (!file_exists(graph_path)) {
            graph_path = join_path(manifest_dir, graph_rel);
        }
        if (!file_exists(graph_path)) {
            throw std::runtime_error("transpiled causal LM bundle decoder graph not found: " + graph_rel);
        }
        *gb = CactusGraph::load(graph_path);
        bind_saved_constants(decoder_json,
                             gb,
                             bound_constant_storage_,
                             bundle_root,
                             manifest_dir);

        if (manifest.find("\"spec_decode\"") == std::string::npos) {
            speculative_status_ = "unsupported_target";
            return;
        }
        spec_decode_manifest_ = parse_spec_decode_manifest_from_json(manifest);
        decoder_logical_outputs_ = json_string_array_field(decoder_json, "logical_outputs");
        if (!has_all_string_values(decoder_logical_outputs_, {
                "verifier_logits",
                "target_hidden_state",
                "shared_kv.full_attention.key",
                "shared_kv.full_attention.value",
                "shared_kv.sliding_attention.key",
                "shared_kv.sliding_attention.value",
            })) {
            speculative_status_ = "target_roles_unavailable";
            return;
        }
        if (output_node_ids_.size() < decoder_logical_outputs_.size()) {
            speculative_status_ = "target_roles_unavailable";
            return;
        }
        const size_t target_embedding_pos = find_component_object(manifest, "target_embedding");
        if (target_embedding_pos == std::string::npos) {
            speculative_status_ = "target_embedding_unavailable";
            return;
        }
        const std::string target_embedding_json = manifest.substr(target_embedding_pos, find_json_matching_brace(manifest, target_embedding_pos) - target_embedding_pos + 1);
        const std::string target_embedding_graph_rel = json_string_field(target_embedding_json, "graph");
        target_embedding_runtime_input_node_ids_ = json_integer_array_field(target_embedding_json, "runtime_input_node_ids");
        target_embedding_output_node_ids_ = json_integer_array_field(target_embedding_json, "output_node_ids");
        if (target_embedding_graph_rel.empty() ||
            target_embedding_runtime_input_node_ids_.empty() ||
            target_embedding_output_node_ids_.empty()) {
            speculative_status_ = "target_embedding_unavailable";
            return;
        }
        target_embedding_logical_inputs_ = json_string_array_field(target_embedding_json, "logical_inputs");
        target_embedding_logical_outputs_ = json_string_array_field(target_embedding_json, "logical_outputs");
        if (!has_string_value(target_embedding_logical_inputs_, "current_token_ids") ||
            !has_string_value(target_embedding_logical_outputs_, "target_token_embedding")) {
            speculative_status_ = "target_embedding_unavailable";
            return;
        }
        if (target_embedding_runtime_input_node_ids_.size() < target_embedding_logical_inputs_.size() ||
            target_embedding_output_node_ids_.size() < target_embedding_logical_outputs_.size()) {
            speculative_status_ = "target_embedding_unavailable";
            return;
        }
        std::string target_embedding_graph_path = join_path(bundle_root, target_embedding_graph_rel);
        if (!file_exists(target_embedding_graph_path)) {
            target_embedding_graph_path = join_path(manifest_dir, target_embedding_graph_rel);
        }
        if (!file_exists(target_embedding_graph_path)) {
            speculative_status_ = "target_embedding_unavailable";
            return;
        }
        target_embedding_graph_ = std::make_unique<CactusGraph>(CactusGraph::load(target_embedding_graph_path));
        bind_saved_constants(target_embedding_json,
                             target_embedding_graph_.get(),
                             target_embedding_bound_constant_storage_,
                             bundle_root,
                             manifest_dir);

        const size_t assistant_pos = find_component_object(manifest, "assistant");
        if (assistant_pos == std::string::npos) {
            speculative_status_ = "assistant_unavailable";
            return;
        }
        const std::string assistant_json = manifest.substr(assistant_pos, find_json_matching_brace(manifest, assistant_pos) - assistant_pos + 1);
        const std::string assistant_graph_rel = json_string_field(assistant_json, "graph");
        assistant_runtime_input_node_ids_ = json_integer_array_field(assistant_json, "runtime_input_node_ids");
        assistant_output_node_ids_ = json_integer_array_field(assistant_json, "output_node_ids");
        if (assistant_graph_rel.empty() || assistant_runtime_input_node_ids_.empty() || assistant_output_node_ids_.empty()) {
            speculative_status_ = "assistant_unavailable";
            return;
        }
        assistant_logical_inputs_ = json_string_array_field(assistant_json, "logical_inputs");
        assistant_logical_outputs_ = json_string_array_field(assistant_json, "logical_outputs");
        if (!has_all_string_values(assistant_logical_inputs_, {
                "current_token_embedding",
                "previous_target_hidden",
                "position",
                "shared_kv.full_attention.key",
                "shared_kv.full_attention.value",
                "shared_kv.sliding_attention.key",
                "shared_kv.sliding_attention.value",
            }) ||
            !has_all_string_values(assistant_logical_outputs_, {"logits_output", "next_hidden_output"})) {
            speculative_status_ = "assistant_roles_unavailable";
            return;
        }
        if (assistant_runtime_input_node_ids_.size() < assistant_logical_inputs_.size() ||
            assistant_output_node_ids_.size() < assistant_logical_outputs_.size()) {
            speculative_status_ = "assistant_roles_unavailable";
            return;
        }
        std::string assistant_graph_path = join_path(bundle_root, assistant_graph_rel);
        if (!file_exists(assistant_graph_path)) {
            assistant_graph_path = join_path(manifest_dir, assistant_graph_rel);
        }
        if (!file_exists(assistant_graph_path)) {
            speculative_status_ = "assistant_unavailable";
            return;
        }
        assistant_graph_ = std::make_unique<CactusGraph>(CactusGraph::load(assistant_graph_path));
        bind_saved_constants(assistant_json,
                             assistant_graph_.get(),
                             assistant_bound_constant_storage_,
                             bundle_root,
                             manifest_dir);
        speculative_status_.clear();
    }

    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool = false, size_t = 0) override {
        throw std::runtime_error("transpiled causal LM model does not expose native attention construction");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("transpiled causal LM model does not expose native MLP construction");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool = false, size_t = 0) override {
        throw std::runtime_error("transpiled causal LM model does not expose native transformer block construction");
    }

private:
    void bind_saved_constants(const std::string& component_json,
                              CactusGraph* graph,
                              std::vector<std::vector<uint8_t>>& storage,
                              const std::string& bundle_root,
                              const std::string& manifest_dir) {
        for (const std::string& binding : json_object_array_field(component_json, "bound_constant_bindings")) {
            std::string kind = json_string_field(binding, "kind");
            std::string format = json_string_field(binding, "format");
            std::string rel_path = json_string_field(binding, "path");
            int node_id = json_integer_field(binding, "node_id", 0, -1);
            if (node_id < 0 || rel_path.empty()) {
                throw std::runtime_error("transpiled bound constant binding is missing node_id or path");
            }
            std::string path = resolve_manifest_path(bundle_root, manifest_dir, rel_path);
            if (kind == "weight" || kind == "embedding") {
                if (std::getenv("CACTUS_TRACE_BINDINGS")) {
                    std::cerr << "[binding] " << kind << " node=" << node_id << " path=" << path << "\n";
                }
                graph->bind_mmap_weights(static_cast<size_t>(node_id), path);
                continue;
            }
            if (!kind.empty() && kind != "saved_constant") {
                continue;
            }
            if (!format.empty() && format != "npy") {
                throw std::runtime_error("unsupported transpiled bound constant format: " + format);
            }

            LoadedNpyArray array = load_npy_array(path);
            int manifest_precision = json_integer_field(binding, "precision", 0, static_cast<int>(array.precision));
            Precision expected_precision = precision_from_manifest_value(manifest_precision, path);
            if (expected_precision != array.precision) {
                throw std::runtime_error("bound constant precision does not match npy dtype: " + path);
            }

            const auto& buffer = graph->get_output_buffer(static_cast<size_t>(node_id));
            if (buffer.precision != array.precision) {
                throw std::runtime_error("bound constant graph input precision does not match manifest: " + path);
            }
            if (buffer.total_size * PrecisionTraits::size_of(buffer.precision) != array.bytes.size()) {
                throw std::runtime_error("bound constant byte size does not match graph input: " + path);
            }
            if (!array.shape.empty() && buffer.shape != array.shape) {
                throw std::runtime_error("bound constant shape does not match graph input: " + path);
            }

            storage.push_back(std::move(array.bytes));
            graph->set_external_input(static_cast<size_t>(node_id), storage.back().data(), expected_precision);
        }
    }

    size_t output_node_for_role(const std::vector<std::string>& roles,
                                const std::vector<size_t>& node_ids,
                                const std::string& role) const {
        auto it = std::find(roles.begin(), roles.end(), role);
        if (it == roles.end()) {
            throw std::runtime_error("transpiled MTP component missing role: " + role);
        }
        size_t index = static_cast<size_t>(std::distance(roles.begin(), it));
        if (index >= node_ids.size()) {
            throw std::runtime_error("transpiled MTP component role has no node id: " + role);
        }
        return node_ids[index];
    }

    size_t input_node_for_role(const std::vector<std::string>& roles,
                               const std::vector<size_t>& node_ids,
                               const std::string& role) const {
        return output_node_for_role(roles, node_ids, role);
    }

    std::vector<float> output_tensor_to_fp32(CactusGraph* gb, size_t node_id) {
        const auto& buffer = gb->get_output_buffer(node_id);
        void* output = gb->get_output(node_id);
        std::vector<float> values(buffer.total_size, 0.0f);
        if (buffer.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(output);
            std::copy(src, src + buffer.total_size, values.begin());
        } else if (buffer.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(output);
            for (size_t i = 0; i < buffer.total_size; ++i) {
                values[i] = static_cast<float>(src[i]);
            }
        } else if (buffer.precision == Precision::INT8) {
            const int8_t* src = static_cast<const int8_t*>(output);
            for (size_t i = 0; i < buffer.total_size; ++i) {
                values[i] = static_cast<float>(src[i]);
            }
        } else {
            throw std::runtime_error("transpiled MTP role output uses unsupported precision");
        }
        return values;
    }

    std::vector<float> output_row_to_fp32(CactusGraph* gb, size_t node_id, size_t row) {
        const auto& buffer = gb->get_output_buffer(node_id);
        std::vector<float> tensor = output_tensor_to_fp32(gb, node_id);
        if (buffer.shape.empty()) {
            return tensor;
        }
        size_t width = buffer.shape.back();
        if (width == 0 || tensor.size() <= width) {
            return tensor;
        }
        size_t row_count = tensor.size() / width;
        if (row >= row_count) {
            row = row_count - 1;
        }
        std::vector<float> row_values(width);
        std::copy(tensor.begin() + static_cast<std::ptrdiff_t>(row * width),
                  tensor.begin() + static_cast<std::ptrdiff_t>((row + 1) * width),
                  row_values.begin());
        return row_values;
    }

    void set_fp32_input(CactusGraph* gb, size_t node_id, const std::vector<float>& values) {
        const auto& input_buf = gb->get_output_buffer(node_id);
        size_t capacity = input_buf.total_size;
        if (capacity == 0) {
            throw std::runtime_error("transpiled MTP input has empty shape");
        }
        if (input_buf.precision == Precision::FP32) {
            std::vector<float> data(capacity, 0.0f);
            std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(std::min(values.size(), capacity)), data.begin());
            gb->set_input(node_id, data.data(), Precision::FP32);
        } else if (input_buf.precision == Precision::FP16) {
            std::vector<__fp16> data(capacity, __fp16(0));
            for (size_t i = 0; i < std::min(values.size(), capacity); ++i) {
                data[i] = __fp16(values[i]);
            }
            gb->set_input(node_id, data.data(), Precision::FP16);
        } else if (input_buf.precision == Precision::INT8) {
            std::vector<int8_t> data(capacity, 0);
            for (size_t i = 0; i < std::min(values.size(), capacity); ++i) {
                data[i] = static_cast<int8_t>(values[i]);
            }
            gb->set_input(node_id, data.data(), Precision::INT8);
        } else {
            throw std::runtime_error("transpiled MTP input uses unsupported precision");
        }
    }

    std::vector<float> run_target_embedding(uint32_t token) {
        size_t output_node = run_component_graph(
            target_embedding_graph_.get(),
            target_embedding_runtime_input_node_ids_,
            target_embedding_output_node_ids_,
            std::vector<uint32_t>{token});
        (void)output_node;
        return output_tensor_to_fp32(
            target_embedding_graph_.get(),
            output_node_for_role(target_embedding_logical_outputs_, target_embedding_output_node_ids_, "target_token_embedding"));
    }

    size_t run_assistant_graph(const std::vector<float>& current_embedding,
                               const std::vector<float>& previous_hidden,
                               const std::vector<float>& full_key,
                               const std::vector<float>& full_value,
                               const std::vector<float>& sliding_key,
                               const std::vector<float>& sliding_value,
                               size_t position) {
        CactusGraph* gb = assistant_graph_.get();
        if (!gb) {
            throw std::runtime_error("transpiled assistant graph is not initialized");
        }
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "current_token_embedding"), current_embedding);
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "previous_target_hidden"), previous_hidden);
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "position"), {static_cast<float>(position)});
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "shared_kv.full_attention.key"), full_key);
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "shared_kv.full_attention.value"), full_value);
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "shared_kv.sliding_attention.key"), sliding_key);
        set_fp32_input(gb, input_node_for_role(assistant_logical_inputs_, assistant_runtime_input_node_ids_, "shared_kv.sliding_attention.value"), sliding_value);
        gb->execute();
        return output_node_for_role(assistant_logical_outputs_, assistant_output_node_ids_, "logits_output");
    }

    size_t run_decoder(CactusGraph* gb, const std::vector<uint32_t>& tokens) {
        return run_component_graph(gb, runtime_input_node_ids_, output_node_ids_, tokens);
    }

    size_t run_component_graph(CactusGraph* gb,
                               const std::vector<size_t>& runtime_input_node_ids,
                               const std::vector<size_t>& output_node_ids,
                               const std::vector<uint32_t>& tokens) {
        if (!gb) {
            throw std::runtime_error("transpiled causal LM graph is not initialized");
        }
        if (runtime_input_node_ids.empty() || output_node_ids.empty()) {
            throw std::runtime_error("transpiled causal LM decoder manifest is incomplete");
        }
        if (tokens.empty()) {
            throw std::runtime_error("transpiled causal LM decoder requires non-empty input tokens");
        }

        size_t input_node = runtime_input_node_ids[0];
        const auto& input_buf = gb->get_output_buffer(input_node);
        size_t capacity = input_buf.total_size;
        if (capacity == 0) {
            for (size_t dim : input_buf.shape) {
                capacity = capacity == 0 ? dim : capacity * dim;
            }
        }
        if (capacity == 0) {
            throw std::runtime_error("transpiled causal LM decoder input has empty shape");
        }
        if (tokens.size() > capacity) {
            throw std::runtime_error("transpiled causal LM prompt exceeds decoder graph context");
        }

        if (input_buf.precision == Precision::FP32) {
            std::vector<float> data(capacity, 0.0f);
            for (size_t i = 0; i < tokens.size(); ++i) {
                data[i] = static_cast<float>(tokens[i]);
            }
            gb->set_input(input_node, data.data(), Precision::FP32);
        } else if (input_buf.precision == Precision::FP16) {
            std::vector<__fp16> data(capacity, __fp16(0));
            for (size_t i = 0; i < tokens.size(); ++i) {
                data[i] = __fp16(static_cast<float>(tokens[i]));
            }
            gb->set_input(input_node, data.data(), Precision::FP16);
        } else {
            std::vector<int8_t> data(capacity, 0);
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (tokens[i] > 127) {
                    throw std::runtime_error("transpiled causal LM INT8 token input cannot represent token id " + std::to_string(tokens[i]));
                }
                data[i] = static_cast<int8_t>(tokens[i]);
            }
            gb->set_input(input_node, data.data(), Precision::INT8);
        }
        gb->execute();
        return output_node_ids[0];
    }

    std::string manifest_path_;
    SpecDecodeManifest spec_decode_manifest_;
    std::string speculative_status_ = "unsupported_target";
    std::vector<size_t> runtime_input_node_ids_;
    std::vector<size_t> output_node_ids_;
    std::vector<std::string> decoder_logical_outputs_;
    std::unique_ptr<CactusGraph> target_embedding_graph_;
    std::vector<size_t> target_embedding_runtime_input_node_ids_;
    std::vector<size_t> target_embedding_output_node_ids_;
    std::vector<std::string> target_embedding_logical_inputs_;
    std::vector<std::string> target_embedding_logical_outputs_;
    std::unique_ptr<CactusGraph> assistant_graph_;
    std::vector<size_t> assistant_runtime_input_node_ids_;
    std::vector<size_t> assistant_output_node_ids_;
    std::vector<std::string> assistant_logical_inputs_;
    std::vector<std::string> assistant_logical_outputs_;
    std::vector<std::vector<uint8_t>> bound_constant_storage_;
    std::vector<std::vector<uint8_t>> target_embedding_bound_constant_storage_;
    std::vector<std::vector<uint8_t>> assistant_bound_constant_storage_;
    std::vector<uint32_t> context_tokens_;
};

bool manifest_is_transpiled_causal_lm(const std::string& manifest_path) {
    if (manifest_path.empty()) {
        return false;
    }
    std::string manifest = read_text_file(manifest_path);
    return json_string_field(manifest, "task") == "causal_lm_logits";
}

} // namespace

void ConvCache::init(size_t layers, size_t hidden_dim, size_t window_len, Precision model_precision) {
    num_layers = layers;
    hidden_size = hidden_dim;
    window_size = window_len;
    precision = model_precision;
    element_size = PrecisionTraits::size_of(precision);

    size_t state_bytes = window_size * hidden_size * element_size;
    layer_states.resize(num_layers);
    for (auto& state : layer_states) {
        state.data.resize(state_bytes);
        std::memset(state.data.data(), 0, state_bytes);
        state.head = 0;
        state.count = 0;
    }
}

ConvCache::CircularView ConvCache::get_window(size_t layer) const {
    CircularView view{};
    if (layer >= num_layers) {
        return view;
    }

    const auto& state = layer_states[layer];
    if (state.count == 0) {
        return view;
    }

    size_t stride = hidden_size * element_size;
    if (state.count < window_size) {
        view.ptr1 = state.data.data();
        view.len1 = state.count;
        view.total_len = state.count;
        return view;
    }

    view.ptr1 = state.data.data();
    view.len1 = state.head;
    view.ptr2 = state.data.data() + state.head * stride;
    view.len2 = window_size - state.head;
    view.total_len = window_size;
    return view;
}

void ConvCache::update(CactusGraph* gb, size_t layer, const size_t bx_node) {
    if (layer >= num_layers || !bx_node || window_size == 0 || hidden_size == 0) {
        return;
    }

    auto& state = layer_states[layer];
    const void* output_ptr = gb->get_output(bx_node);
    if (!output_ptr) {
        return;
    }

    const auto& buffer = gb->get_output_buffer(bx_node);
    const size_t stride_bytes = hidden_size * element_size;

    size_t rows = 1;
    if (!buffer.shape.empty()) {
        rows = buffer.shape.size() == 1 ? 1 : buffer.shape[0];
    }

    if (buffer.total_size > 0 && hidden_size > 0) {
        size_t inferred = buffer.total_size / hidden_size;
        if (inferred > 0) {
            rows = inferred;
        }
    }

    if (rows == 0) {
        return;
    }

    size_t copy_rows = std::min(rows, window_size);
    size_t start_row = rows > window_size ? rows - window_size : 0;
    const auto* src = static_cast<const uint8_t*>(output_ptr) + start_row * stride_bytes;

    for (size_t i = 0; i < copy_rows; ++i) {
        std::memcpy(state.data.data() + state.head * stride_bytes, src + i * stride_bytes, stride_bytes);
        state.head = (state.head + 1) % window_size;
        if (state.count < window_size) {
            ++state.count;
        }
    }
}

void ConvCache::reset() {
    for (auto& state : layer_states) {
        std::fill(state.data.begin(), state.data.end(), 0);
        state.head = 0;
        state.count = 0;
    }
}


Model::Model()
        : graph_handle_(nullptr),
            config_(),
            tokenizer_(nullptr),
            initialized_(false),
            attention_scale_(0.0f),
            output_weight_node_id_(0),
            owns_graph_(false) {
}

Model::Model(const Config& config)
    : graph_handle_(nullptr),
      config_(config),
      tokenizer_(nullptr),
      initialized_(false),
      attention_scale_(0.0f),
      output_weight_node_id_(0),
      owns_graph_(false) {
}

Model::~Model() {
    if (graph_handle_ && owns_graph_) {
        delete static_cast<CactusGraph*>(graph_handle_);
    }
}

bool Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt, bool do_warmup) {
    if (initialized_) {
        return true;
    }   
    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    owns_graph_ = true;
    embedding_file_path_ = model_folder + "/token_embeddings.weights";
    return init_internal(gb, model_folder, context_size, system_prompt, do_warmup);
}

bool Model::init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
                 const std::string& system_prompt, bool do_warmup) {
    if (!external_graph) {
        throw std::invalid_argument("External graph pointer must not be null");
    }
    if (initialized_) {
        graph_handle_ = external_graph;
        owns_graph_ = false;
        return true;
    }

    owns_graph_ = false;
    graph_handle_ = external_graph;
    return init_internal(external_graph, model_folder, context_size, system_prompt, do_warmup);
}

bool Model::init_internal(CactusGraph* gb, const std::string& model_folder, size_t context_size,
                          const std::string& system_prompt, bool do_warmup) {
    (void)system_prompt;
    CACTUS_LOG_DEBUG("model", "Initializing model from: " << model_folder);
    model_folder_path_ = model_folder;
    std::string config_path = model_folder + "/config.txt";

    if (!config_.from_json(config_path)) {
        CACTUS_LOG_ERROR("model", "Model initialization failed - config not loaded from: " << model_folder);
        return false;
    }

    std::string vocab_file = model_folder + "/vocab.txt";
    std::string merges_file = model_folder + "/merges.txt";
    std::string tokenizer_config_file = model_folder + "/tokenizer_config.txt";
    TokenizerRuntimeConfig tokenizer_runtime_config = load_tokenizer_runtime_config(tokenizer_config_file);

    std::ifstream merges_check(merges_file);
    bool has_merges = false;
    if (merges_check.is_open()) {
        std::string line;
        int line_count = 0;
        while (std::getline(merges_check, line) && line_count < 10) {
            if (!line.empty() && line[0] != '#') {
                has_merges = true;
                break;
            }
            line_count++;
        }
        merges_check.close();
    }

    if (tokenizer_runtime_config.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::BPE ||
        (tokenizer_runtime_config.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::UNKNOWN && has_merges)) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    } else {
        tokenizer_ = std::make_unique<SPTokenizer>();
    }

    if (!tokenizer_->load_vocabulary_with_config(vocab_file, merges_file, tokenizer_config_file)) {
        return false;
    }

    graph_handle_ = gb;

    embedding_file_path_ = model_folder + "/token_embeddings.weights";

    load_weights_to_graph(gb);

    if (config_.model_type == Config::ModelType::GEMMA3N || config_.model_type == Config::ModelType::GEMMA4) {
        attention_scale_ = 1.0f;
    } else if (config_.model_type == Config::ModelType::GEMMA) {
        attention_scale_ = 1.0f / std::sqrt(256.0f);
    } else {
        attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));
    }

    cache_max_seq_len_ = context_size;
    cache_window_size_ = std::min(context_size, size_t(512));
    cache_sink_size_ = 4;
    const char* env_window = std::getenv("CACTUS_KV_WINDOW_SIZE");
    const char* env_sink = std::getenv("CACTUS_KV_SINK_SIZE");
    if (env_window) {
        cache_window_size_ = std::stoul(env_window);
    }
    if (env_sink) {
        cache_sink_size_ = std::stoul(env_sink);
    }

    post_init();

    initialized_ = true;

    if (do_warmup &&
        config_.model_type != Config::ModelType::WHISPER &&
        config_.model_type != Config::ModelType::PARAKEET_TDT) {
        std::vector<uint32_t> warmup_tokens = {2};
        forward(warmup_tokens);
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        gb->execute();
    }

    reset_cache();
    return true;
}

size_t Model::forward(const std::vector<float>& /*mel_bins*/, const std::vector<uint32_t>& tokens, bool use_cache){
    return forward(tokens, use_cache);
}

void Model::prefill(const std::vector<uint32_t>& tokens, size_t chunk_size, const std::string& profile_file) {
    if (tokens.empty()) {
        return;
    }

    if (has_npu_prefill()) {
        size_t npu_chunk_size = static_cast<size_t>(npu_prefill_->get_chunk_size());
        if (tokens.size() > npu_chunk_size) {
            prefill_npu(tokens);
            return;
        }
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);

    auto process_chunk = [&](const std::vector<uint32_t>& chunk) {
        forward(chunk, true);
        gb->execute(profile_file);
        post_execute_updates(gb, chunk.size());
        cache_total_seq_len_ += chunk.size();
    };

    if (tokens.size() <= chunk_size) {
        process_chunk(tokens);
        return;
    }

    size_t num_full_chunks = (tokens.size() - 1) / chunk_size;

    for (size_t chunk_idx = 0; chunk_idx < num_full_chunks; ++chunk_idx) {
        size_t start = chunk_idx * chunk_size;
        size_t end = start + chunk_size;
        std::vector<uint32_t> chunk(tokens.begin() + start, tokens.begin() + end);
        if (chunk_idx == 1) {
            gb->set_prefill_mode(true);
        }
        process_chunk(chunk);
    }

    gb->set_prefill_mode(false);
    size_t final_start = num_full_chunks * chunk_size;
    std::vector<uint32_t> final_chunk(tokens.begin() + final_start, tokens.end());
    process_chunk(final_chunk);
}

void Model::prefill_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                const std::string& profile_file) {
    (void)image_paths;
    prefill(tokens, get_prefill_chunk_size(), profile_file);
}

uint32_t Model::decode(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                        size_t top_k, const std::string& profile_file, float* out_entropy,
                        float min_p, float repetition_penalty) {

    if (temperature < 0) {
        temperature = config_.default_temperature;
    }
    if (top_p < 0) {
        top_p = config_.default_top_p;
    }
    if (top_k == 0) {
        top_k = config_.default_top_k;
    }
    auto final_hidden = forward(tokens, true);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto last_hidden = gb->index(final_hidden, tokens.size() - 1, 0);
    const auto& last_hidden_buf = gb->get_output_buffer(last_hidden);
    size_t hidden_dim = last_hidden_buf.shape[0];
    last_hidden = gb->reshape(last_hidden, {1, hidden_dim});

    auto logits_node_id = gb->matmul(last_hidden, output_weight_node_id_, true, backend);

    if (config_.final_logit_softcapping > 0.0f) {
        float inv_cap = 1.0f / config_.final_logit_softcapping;
        logits_node_id = gb->scalar_multiply(logits_node_id, inv_cap);
        logits_node_id = gb->tanh(logits_node_id);
        logits_node_id = gb->scalar_multiply(logits_node_id, config_.final_logit_softcapping);
    }
    auto sampled_token_id = sample_token(gb, logits_node_id, temperature, top_p, top_k, min_p, repetition_penalty);

    gb->execute(profile_file);

    compute_entropy(gb, logits_node_id, out_entropy);

    post_execute_updates(gb, tokens.size());
    cache_total_seq_len_ += tokens.size();

    auto* output_ptr = gb->get_output(sampled_token_id);
    uint32_t result_token = *static_cast<uint32_t*>(output_ptr);
    record_sampled_token(result_token);
    return result_token;
}

Model::SpeculativeDecodeResult Model::speculative_decode(
    const std::vector<uint32_t>&,
    size_t,
    size_t,
    float,
    float,
    size_t,
    float,
    float) {
    throw std::runtime_error("speculative decoding is not available for this model");
}

size_t Model::sample_token(CactusGraph* gb, size_t logits_node_id, float temperature, float top_p, size_t top_k,
                           float min_p, float repetition_penalty,
                           const std::unordered_map<uint32_t, float>* extra_bias) const {
    auto combined_bias = tool_constrainer_.get_bias();
    for (const auto& [token_id, boost] : vocab_bias_) {
        combined_bias[token_id] += boost;
    }
    if (extra_bias) {
        for (const auto& [token_id, boost] : *extra_bias) {
            combined_bias[token_id] += boost;
        }
    }
    if (!token_history_.empty() && repetition_penalty > 1.0f && std::isfinite(repetition_penalty)) {
        float log_penalty = std::log(repetition_penalty);
        for (uint32_t tok : token_history_) {
            combined_bias[tok] -= log_penalty;
        }
    }
    return gb->sample_with_options(logits_node_id, temperature, top_p, min_p, 1.0f, top_k, combined_bias);
}

void Model::compute_entropy(CactusGraph* gb, size_t logits_node_id, float* out_entropy) {
    if (!out_entropy) return;

    const auto& logits_buf = gb->get_output_buffer(logits_node_id);
    void* logits_ptr = gb->get_output(logits_node_id);
    size_t vocab_size = logits_buf.shape.back();
    size_t seq_len = 1;
    if (logits_buf.shape.size() >= 2)
        seq_len = logits_buf.shape[logits_buf.shape.size() - 2];
    size_t row_offset = (seq_len > 0 ? (seq_len - 1) * vocab_size : 0);

    std::vector<float> logits(vocab_size);
    if (logits_buf.precision == Precision::FP32) {
        float* src = static_cast<float*>(logits_ptr) + row_offset;
        std::copy(src, src + vocab_size, logits.begin());
    } else if (logits_buf.precision == Precision::FP16) {
        __fp16* src = static_cast<__fp16*>(logits_ptr) + row_offset;
        Quantization::fp16_to_fp32(src, logits.data(), vocab_size);
    } else {
        int8_t* src = static_cast<int8_t*>(logits_ptr) + row_offset;
        Quantization::int8_to_fp32(src, logits.data(), vocab_size, 1.0f);
    }

    float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab_size; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
    double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

    double entropy = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        double log_prob = static_cast<double>(logits[i]) - log_sum_exp;
        double prob = std::exp(log_prob);
        if (prob > 1e-10)
            entropy -= prob * log_prob;
    }

    double max_entropy = std::log(static_cast<double>(vocab_size));
    *out_entropy = static_cast<float>(entropy / max_entropy);
}

uint32_t Model::decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& /*mel_bins*/, float temperature, float top_p, size_t top_k, const std::string& profile_file, float* out_entropy,
                                 float min_p, float repetition_penalty,
                                 float* /*out_token_time_start*/, float* /*out_token_time_end*/){
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

uint32_t Model::decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                     float temperature, float top_p, size_t top_k, const std::string& profile_file, float* out_entropy,
                                     float min_p, float repetition_penalty) {
    (void)image_paths;
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
}

std::vector<float> Model::get_image_embeddings(const std::string& /*image_path*/) {
    throw std::runtime_error("Image embeddings not supported for this model type");
}

std::vector<float> Model::get_audio_embeddings(const std::vector<float>& /*mel_bins*/) {
    throw std::runtime_error("Audio embeddings not supported for this model type");
}

void Model::init_graph_cache(CactusGraph* gb) {
    auto layer_dims = get_kv_layer_dims();
    auto layer_heads = get_kv_layer_heads();
    auto layer_windows = get_kv_layer_windows();
    size_t n = config_.num_layers;

    graph_cache_k_nodes_.resize(n, 0);
    graph_cache_v_nodes_.resize(n, 0);

    for (size_t i = 0; i < n; i++) {
        if (layer_dims[i] == 0) continue;  // shared layers have dim=0
        size_t window = (i < layer_windows.size()) ? layer_windows[i] : cache_window_size_;
        size_t max_seq = (window > 0) ? window : cache_max_seq_len_;
        graph_cache_k_nodes_[i] = gb->kv_cache_state(max_seq, layer_heads[i], layer_dims[i], window, cache_sink_size_);
        graph_cache_v_nodes_[i] = gb->kv_cache_state(max_seq, layer_heads[i], layer_dims[i], window, cache_sink_size_);
    }
    cache_total_seq_len_ = 0;
}

void Model::invalidate_graph_cache(CactusGraph* gb) {
    for (size_t i = 0; i < graph_cache_k_nodes_.size(); i++) {
        if (graph_cache_k_nodes_[i] != 0) gb->invalidate_persistent(graph_cache_k_nodes_[i]);
        if (graph_cache_v_nodes_[i] != 0) gb->invalidate_persistent(graph_cache_v_nodes_[i]);
    }
    graph_cache_k_nodes_.clear();
    graph_cache_v_nodes_.clear();
    cache_total_seq_len_ = 0;
}

void Model::reset_cache() {
    if (graph_handle_) {
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        invalidate_graph_cache(gb);
        init_graph_cache(gb);
    }
    token_history_.clear();
}

void Model::set_cache_window(size_t window_size, size_t sink_size) {
    cache_window_size_ = window_size;
    cache_sink_size_ = sink_size;
    if (graph_handle_) {
        auto* gb = static_cast<CactusGraph*>(graph_handle_);
        invalidate_graph_cache(gb);
        init_graph_cache(gb);
    }
}

size_t Model::get_cache_size() const {
    return cache_total_seq_len_;
}

void Model::remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges) {
    if (!ranges.empty()) {
        size_t total_removed = 0;
        for (const auto& r : ranges) total_removed += r.second;
        if (cache_total_seq_len_ >= total_removed)
            cache_total_seq_len_ -= total_removed;
        else
            cache_total_seq_len_ = 0;
    }
}

std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& tokens, bool pooled, bool normalize, const std::string& profile_file) {
    std::vector<float> embeddings;
    auto final_hidden = forward(tokens);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto* output_ptr = gb->get_output(final_hidden);
    const auto& output_buffer = gb->get_output_buffer(final_hidden);

    if (pooled) {
        auto pooled_hidden = gb->mean(final_hidden, 0);

        if (!profile_file.empty()) {
            gb->execute(profile_file);
        } else {
            gb->execute();
        }
        post_execute_updates(gb, tokens.size());
        auto* pooled_ptr = gb->get_output(pooled_hidden);
        const auto& pooled_buffer = gb->get_output_buffer(pooled_hidden);

        size_t hidden_dim = pooled_buffer.total_size;
        embeddings.resize(hidden_dim);

        if (pooled_buffer.precision == Precision::FP32) {
            float* pooled_data = static_cast<float*>(pooled_ptr);
            std::copy(pooled_data, pooled_data + hidden_dim, embeddings.begin());
        } else if (pooled_buffer.precision == Precision::FP16) {
            __fp16* pooled_data = static_cast<__fp16*>(pooled_ptr);
            Quantization::fp16_to_fp32(pooled_data, embeddings.data(), hidden_dim);
        } else if (pooled_buffer.precision == Precision::INT8) {
            int8_t* pooled_data = static_cast<int8_t*>(pooled_ptr);
            Quantization::int8_to_fp32(pooled_data, embeddings.data(), hidden_dim, 1.0f);
        }
    } else {
        if (!profile_file.empty()) {
            gb->execute(profile_file);
        } else {
            gb->execute();
        }
        post_execute_updates(gb, tokens.size());

        size_t total_size = output_buffer.total_size;
        embeddings.resize(total_size);

        if (output_buffer.precision == Precision::FP32) {
            float* hidden_states = static_cast<float*>(output_ptr);
            std::copy(hidden_states, hidden_states + total_size, embeddings.begin());
        } else if (output_buffer.precision == Precision::FP16) {
            __fp16* hidden_states = static_cast<__fp16*>(output_ptr);
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = static_cast<float>(hidden_states[i]);
            }
        } else if (output_buffer.precision == Precision::INT8) {
            int8_t* hidden_states = static_cast<int8_t*>(output_ptr);
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = static_cast<float>(hidden_states[i]);
            }
        }
    }

    if (normalize && !embeddings.empty()) {
        float norm_sq = 0.0f;
        for (float v : embeddings) {
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        if (norm > 1e-12f) {
            float inv_norm = 1.0f / norm;
            for (float& v : embeddings) {
                v *= inv_norm;
            }
        }
    }

    reset_cache();

    return embeddings;
}

bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        CACTUS_LOG_ERROR("config", "Failed to open config file: " << config_path);
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        if (key == "vocab_size") vocab_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "bos_token_id") bos_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "eos_token_id") eos_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_layers") num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "hidden_dim") hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "ffn_intermediate_dim") ffn_intermediate_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_heads") attention_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_kv_heads") attention_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_head_dim") attention_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "layer_norm_eps") layer_norm_eps = std::stof(value);
        else if (key == "rope_theta") rope_theta = std::stof(value);
        else if (key == "num_experts") num_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_shared_experts") num_shared_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_top_experts") num_top_experts = static_cast<uint32_t>(std::stoul(value));
        else if (key == "moe_every_n_layers") moe_every_n_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "moe_intermediate_dim" || key == "moe_intermediate_size") moe_intermediate_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_dense_layers") num_dense_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_experts_per_tok") num_experts_per_tok = static_cast<uint32_t>(std::stoul(value));
        else if (key == "norm_topk_prob") norm_topk_prob = (value == "true" || value == "1");
        else if (key == "use_expert_bias") use_expert_bias = (value == "true" || value == "1");
        else if (key == "routed_scaling_factor") routed_scaling_factor = std::stof(value);
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "vision_hidden_dim" || key == "vision_hidden_size") vision_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_num_layers") vision_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_attention_heads") vision_attention_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_image_size") vision_image_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_patch_size") vision_patch_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_num_channels") vision_num_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_embed_dim") vision_embed_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "visual_tokens_per_img") visual_tokens_per_img = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_pixel_shuffle") use_pixel_shuffle = (value == "true" || value == "1");
        else if (key == "pixel_shuffle_factor") pixel_shuffle_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_image_tokens") use_image_tokens = (value == "true" || value == "1");
        else if (key == "image_token_id") image_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_layout_tags") use_layout_tags = (value == "true" || value == "1");
        else if (key == "image_seq_len") image_seq_len = static_cast<uint32_t>(std::stoul(value));
        else if (key == "global_image_size") global_image_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_tile_size") max_tile_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rescale_factor") rescale_factor = std::stof(value);
        else if (key == "image_mean") image_mean = std::stof(value);
        else if (key == "image_std") image_std = std::stof(value);
        else if (key == "downsample_factor") downsample_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "min_tiles") min_tiles = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_tiles") max_tiles = static_cast<uint32_t>(std::stoul(value));
        else if (key == "use_thumbnail") use_thumbnail = (value == "true" || value == "1");
        else if (key == "min_image_tokens") min_image_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_image_tokens") max_image_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tile_size") tile_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "max_pixels_tolerance") max_pixels_tolerance = std::stof(value);
        else if (key == "do_image_splitting") do_image_splitting = (value == "true" || value == "1");
        else if (key == "precision") {
            if (value == "INT8") precision = Precision::INT8;
            else if (value == "FP16") precision = Precision::FP16;
            else precision = Precision::FP32;
        }
        else if (key == "model_type") {
            std::string mt = value;
            std::transform(mt.begin(), mt.end(), mt.begin(), ::tolower);
            if (mt == "qwen") model_type = ModelType::QWEN;
            else if (mt == "qwen3p5" || mt == "qwen3_5") model_type = ModelType::QWEN3P5;
            else if (mt == "gemma") model_type = ModelType::GEMMA;
            else if (mt == "gemma3n") model_type = ModelType::GEMMA3N;
            else if (mt == "lfm2") model_type = ModelType::LFM2;
            else if (mt == "whisper") model_type = ModelType::WHISPER;
            else if (mt == "parakeet_tdt" || mt == "parakeet-tdt") model_type = ModelType::PARAKEET_TDT;
            else if (mt == "youtu") model_type = ModelType::YOUTU;
            else if (mt == "needle") model_type = ModelType::NEEDLE;
            else model_type = ModelType::GEMMA4;
        }
        else if (key == "model_variant") {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            if (v == "vlm") model_variant = ModelVariant::VLM;
            else if (v == "extract") model_variant = ModelVariant::EXTRACT;
            else if (v == "rag") model_variant = ModelVariant::RAG;
            else model_variant = ModelVariant::DEFAULT;
        }
        else if (key == "conv_L_cache") conv_L_cache = static_cast<size_t>(std::stoul(value));
        else if (key == "layer_types") {
            layer_types.clear();
            std::string sanitized;
            sanitized.reserve(value.size());
            for (char c : value) {
                if (c == '[' || c == ']' || c == '\'' || c == '"') {
                    continue;
                }
                sanitized.push_back(c);
            }
            std::stringstream ss(sanitized);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty()) layer_types.push_back(item);
                }
            }
        }
        else if (key == "enc_hidden_act") encoder_act_gelu = (value == "gelu");
        else if (key == "dec_hidden_act") decoder_act_gelu = (value == "gelu");
        else if (key == "num_encoder_layers") num_encoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_decoder_layers") num_decoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "partial_rotary_factor") partial_rotary_factor = std::stof(value);
        else if (key == "pad_token_id") pad_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "conv_kernel_size") conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_kernel_size") subsampling_conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_stride") subsampling_conv_stride = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_conv_channels") subsampling_conv_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "subsampling_factor") subsampling_factor = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_mel_bins") num_mel_bins = static_cast<uint32_t>(std::stoul(value));
        else if (key == "encoder_hidden_act") encoder_hidden_act = value;
        else if (key == "linear_num_key_heads") linear_num_key_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_key_head_dim") linear_key_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_num_value_heads") linear_num_value_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_value_head_dim") linear_value_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_q_proj_dim") linear_q_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "kv_lora_rank") kv_lora_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "q_lora_rank") q_lora_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_head_dim") qk_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_nope_head_dim") qk_nope_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "qk_rope_head_dim") qk_rope_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "v_head_dim") v_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rope_interleave") rope_interleave = (value == "true" || value == "1");
        else if (key == "attention_bias") attention_bias = (value == "true" || value == "1");
        else if (key == "rope_scaling_factor") rope_scaling_factor = std::stof(value);
        else if (key == "rope_mscale_all_dim") rope_mscale_all_dim = std::stof(value);
        else if (key == "linear_k_proj_dim") linear_k_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "linear_v_proj_dim") linear_v_proj_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "predictor_hidden_dim") predictor_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "predictor_num_layers") predictor_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_joint_dim") tdt_joint_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_num_durations") tdt_num_durations = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_blank_id") tdt_blank_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "tdt_durations") {
            tdt_durations.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t first = item.find_first_not_of(" \t");
                if (first == std::string::npos) continue;
                size_t last = item.find_last_not_of(" \t");
                item = item.substr(first, last - first + 1);
                tdt_durations.push_back(static_cast<uint32_t>(std::stoul(item)));
            }
        }
        else if (key == "altup_num_inputs") altup_num_inputs = static_cast<uint32_t>(std::stoul(value));
        else if (key == "laurel_rank") laurel_rank = static_cast<uint32_t>(std::stoul(value));
        else if (key == "hidden_size_per_layer_input") hidden_size_per_layer_input = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_kv_shared_layers") num_kv_shared_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "sliding_window") sliding_window = static_cast<uint32_t>(std::stoul(value));
        else if (key == "rope_local_base_freq") rope_local_base_freq = std::stof(value);
        else if (key == "final_logit_softcapping") final_logit_softcapping = std::stof(value);
        else if (key == "global_partial_rotary_factor") global_partial_rotary_factor = std::stof(value);
        else if (key == "expert_intermediate_size") expert_intermediate_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "global_head_dim") global_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_global_kv_heads" || key == "num_global_key_value_heads") num_global_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "attention_k_eq_v") attention_k_eq_v = (value == "true" || value == "1");
        else if (key == "enable_moe_block") enable_moe_block = (value == "true" || value == "1");
        else if (key == "vision_head_dim") vision_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_kv_heads") vision_kv_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_intermediate_size") vision_intermediate_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_position_embedding_size") vision_position_embedding_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_pooling_kernel_size") vision_pooling_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_default_output_length") vision_default_output_length = static_cast<uint32_t>(std::stoul(value));
        else if (key == "vision_rope_theta") vision_rope_theta = std::stof(value);
        else if (key == "audio_hidden_dim") audio_hidden_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_num_layers") audio_num_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_num_heads") audio_num_heads = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_head_dim") audio_head_dim = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_input_feat_size") audio_input_feat_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_conf_conv_kernel_size") audio_conf_conv_kernel_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_chunk_size") audio_chunk_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_context_left") audio_context_left = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_context_right") audio_context_right = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_logit_cap") audio_logit_cap = std::stof(value);
        else if (key == "audio_residual_weight") audio_residual_weight = std::stof(value);
        else if (key == "audio_output_proj_dims") audio_output_proj_dims = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_vocab_size") audio_vocab_size = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_vocab_offset") audio_vocab_offset = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_soft_tokens") audio_soft_tokens = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv0_channels") audio_sscp_conv0_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv1_channels") audio_sscp_conv1_channels = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_sscp_conv_eps") audio_sscp_conv_eps = std::stof(value);
        else if (key == "audio_rms_norm_eps") audio_rms_norm_eps = std::stof(value);
        else if (key == "audio_fft_length") audio_fft_length = static_cast<uint32_t>(std::stoul(value));
        else if (key == "audio_fft_overdrive") {
            audio_fft_overdrive = (value == "true" || value == "1");
            audio_fft_length = audio_fft_overdrive ? 1024u : 512u;
        }
        else if (key == "audio_token_id") audio_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "channel_open_token_id") channel_open_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "channel_close_token_id") channel_close_token_id = static_cast<uint32_t>(std::stoul(value));
        else if (key == "activation_sparsity_ppf") {
            activation_sparsity_ppf.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t first = item.find_first_not_of(" \t");
                if (first == std::string::npos) continue;
                size_t last = item.find_last_not_of(" \t");
                item = item.substr(first, last - first + 1);
                activation_sparsity_ppf.push_back(std::stof(item));
            }
        }
    }

    if (is_gemma_family(model_type)) {
        default_temperature = 1.0f;
        default_top_p = 0.95f;
        default_top_k = 64;
        if (model_type == ModelType::GEMMA4) {
            default_cloud_handoff_threshold = 0.92f;
            default_rolling_entropy_window = 16;
        }
    } else if (model_type == ModelType::LFM2) {
        default_temperature = 0.3f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN) {
        default_temperature = 0.6f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN3P5) {
        default_temperature = 0.7f;
        default_top_p = 0.8f;
        default_top_k = 20;
    }

    if (model_type == ModelType::GEMMA4) {
        auto missing_u32 = [](uint32_t v) { return v == UNSET_U32; };
        auto missing_f32 = [](float v) { return v == UNSET_F32; };
        std::string missing;
        if (missing_u32(hidden_size_per_layer_input)) missing += " hidden_size_per_layer_input";
        if (missing_u32(num_kv_shared_layers)) missing += " num_kv_shared_layers";
        if (missing_u32(sliding_window)) missing += " sliding_window";
        if (missing_u32(global_head_dim)) missing += " global_head_dim";
        if (missing_f32(rope_local_base_freq)) missing += " rope_local_base_freq";
        if (missing_f32(final_logit_softcapping)) missing += " final_logit_softcapping";
        if (missing_f32(global_partial_rotary_factor)) missing += " global_partial_rotary_factor";
        if (layer_types.empty()) missing += " layer_types";
        if (!missing.empty()) {
            CACTUS_LOG_ERROR("config", "Gemma4 config missing required fields:" << missing);
            return false;
        }
    }

    return true;
}

std::string Config::to_json() const {
    return "{}";
}

std::unique_ptr<Model> create_model(const std::string& model_folder) {
    CACTUS_LOG_DEBUG("model", "Creating model from: " << model_folder);
    const std::string transpiled_manifest = find_transpiled_manifest_path(model_folder);
    if (!transpiled_manifest.empty() && manifest_is_transpiled_causal_lm(transpiled_manifest)) {
        return std::make_unique<TranspiledCausalLmModel>(transpiled_manifest);
    }

    Config config;
    std::string config_path = model_folder + "/config.txt";

    if (!config.from_json(config_path)) {
        CACTUS_LOG_ERROR("model", "Failed to create model - cannot load config from: " << model_folder);
        return nullptr;
    }

    CACTUS_LOG_ERROR("model",
        "Native model subclasses are not present in this build. "
        "Use cactus run-transpiled for transpiled graph bundles.");
    return nullptr;
}

void Model::capture_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id) const {
    auto* graph = static_cast<CactusGraph*>(graph_handle_);
    if (!graph) {
        return;
    }
    graph->capture_debug_node(layer_idx, name, node_id);
}

void Model::clear_debug_nodes() {
    auto* graph = static_cast<CactusGraph*>(graph_handle_);
    if (!graph) {
        return;
    }
    graph->clear_debug_nodes();
}

const std::vector<Model::DebugNode>& Model::get_debug_nodes() const {
    auto* graph = static_cast<CactusGraph*>(graph_handle_);
    debug_nodes_.clear();
    if (!graph) {
        return debug_nodes_;
    }

    const auto& entries = graph->get_debug_nodes();
    debug_nodes_.reserve(entries.size());
    for (const auto& entry : entries) {
        debug_nodes_.push_back({entry.layer_idx, entry.name, entry.node_id});
    }
    return debug_nodes_;
}

bool Model::load_npu_prefill(const std::string& model_path) {
    CACTUS_LOG_DEBUG("npu", "Attempting to load NPU prefill from: " << model_path);

    npu_prefill_ = npu::create_prefill();
    if (!npu_prefill_) {
        CACTUS_LOG_DEBUG("npu", "NPU prefill creation failed (not supported on this device)");
        return false;
    }

    bool loaded = npu_prefill_->load(model_path);
    if (loaded) {
        CACTUS_LOG_INFO("npu", "NPU prefill loaded successfully from: " << model_path);
    } else {
        CACTUS_LOG_DEBUG("npu", "NPU prefill model not found at: " << model_path);
    }
    return loaded;
}

bool Model::has_npu_prefill() const {
    return npu_prefill_ && npu_prefill_->is_available();
}

size_t Model::get_prefill_chunk_size() const {
    if (has_npu_prefill()) {
        return static_cast<size_t>(npu_prefill_->get_chunk_size());
    }
    return 256;  // default chunk size
}

std::vector<__fp16> Model::get_token_embeddings(const std::vector<uint32_t>& tokens) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb || tokens.empty()) {
        return {};
    }

    gb->soft_reset();

    size_t tok_input = gb->input({tokens.size()}, Precision::FP32);
    std::vector<float> tok_f(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        tok_f[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    size_t embedding_node = gb->embedding(embedding_node_id_, tok_input);

    gb->execute();

    const auto& emb_buf = gb->get_output_buffer(embedding_node);
    void* emb_ptr = gb->get_output(embedding_node);

    size_t num_tokens = tokens.size();
    size_t hidden_dim = config_.hidden_dim;
    std::vector<__fp16> embeddings(num_tokens * hidden_dim);

    if (emb_buf.precision == Precision::FP16) {
        __fp16* src = static_cast<__fp16*>(emb_ptr);
        std::copy(src, src + num_tokens * hidden_dim, embeddings.begin());
    } else if (emb_buf.precision == Precision::FP32) {
        float* src = static_cast<float*>(emb_ptr);
        for (size_t i = 0; i < num_tokens * hidden_dim; i++) {
            embeddings[i] = static_cast<__fp16>(src[i]);
        }
    } else if (emb_buf.precision == Precision::INT8) {
        int8_t* src = static_cast<int8_t*>(emb_ptr);
        for (size_t i = 0; i < num_tokens * hidden_dim; i++) {
            embeddings[i] = static_cast<__fp16>(src[i]);
        }
    }

    return embeddings;
}

void Model::prefill_npu(const std::vector<uint32_t>& tokens) {
    if (!npu_prefill_ || !npu_prefill_->is_available()) {
        throw std::runtime_error("NPU prefill not available");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const int chunk_size = npu_prefill_->get_chunk_size();
    const int hidden_dim = npu_prefill_->get_hidden_dim();
    const int num_layers = npu_prefill_->get_num_layers();
    const int fallback_num_kv_heads = npu_prefill_->get_num_kv_heads();
    const int fallback_head_dim = npu_prefill_->get_head_dim();

    const std::vector<size_t> layer_dims = get_kv_layer_dims();
    const std::vector<size_t> layer_heads = get_kv_layer_heads();
    const int layers_to_update = std::min<int>(num_layers, static_cast<int>(config_.num_layers));

    std::vector<__fp16> all_embeddings = get_token_embeddings(tokens);
    if (all_embeddings.empty()) {
        throw std::runtime_error("Failed to get token embeddings for NPU prefill");
    }

    if (Config::is_gemma_family(config_.model_type)) {
        float scale = std::sqrt(static_cast<float>(hidden_dim));
        for (size_t i = 0; i < all_embeddings.size(); i++) {
            all_embeddings[i] = __fp16(static_cast<float>(all_embeddings[i]) * scale);
        }
    }

    size_t num_tokens = tokens.size();
    size_t num_chunks = (num_tokens + chunk_size - 1) / chunk_size;

    for (size_t c = 0; c < num_chunks; c++) {
        size_t start = c * chunk_size;
        size_t actual_tokens = std::min(static_cast<size_t>(chunk_size), num_tokens - start);

        std::vector<__fp16> chunk_embeddings(chunk_size * hidden_dim, __fp16(0));
        std::copy(all_embeddings.begin() + start * hidden_dim,
                  all_embeddings.begin() + (start + actual_tokens) * hidden_dim,
                  chunk_embeddings.begin());

        int position_offset = static_cast<int>(start);

        npu::NPUPrefillDirectResult direct_result = npu_prefill_->prefill_chunk_direct(chunk_embeddings, position_offset);

        if (direct_result.valid) {
            gb->soft_reset_keep_pool();
            for (int layer_idx = 0; layer_idx < layers_to_update; layer_idx++) {
                const auto& k_ref = direct_result.k_caches[layer_idx];
                const auto& v_ref = direct_result.v_caches[layer_idx];

                if (k_ref.data && v_ref.data && graph_cache_k_nodes_[layer_idx] != 0) {
                    size_t layer_kv_heads = layer_idx < static_cast<int>(layer_heads.size())
                        ? layer_heads[layer_idx]
                        : static_cast<size_t>(fallback_num_kv_heads);
                    size_t layer_head_dim = layer_idx < static_cast<int>(layer_dims.size())
                        ? layer_dims[layer_idx]
                        : static_cast<size_t>(fallback_head_dim);

                    size_t expected = static_cast<size_t>(chunk_size) * layer_kv_heads * layer_head_dim;
                    if (expected > 0 && (k_ref.count < expected || v_ref.count < expected)) {
                        CACTUS_LOG_WARN(
                            "npu",
                            "NPU prefill cache output too small for layer " << layer_idx
                            << " (expected>=" << expected
                            << ", got k=" << k_ref.count << ", v=" << v_ref.count << "); skipping layer");
                        continue;
                    }

                    size_t kv_elements = actual_tokens * layer_kv_heads * layer_head_dim;
                    size_t k_input = gb->input({kv_elements}, Precision::FP16);
                    gb->set_external_input(k_input, const_cast<__fp16*>(k_ref.data), Precision::FP16);
                    size_t v_input = gb->input({kv_elements}, Precision::FP16);
                    gb->set_external_input(v_input, const_cast<__fp16*>(v_ref.data), Precision::FP16);

                    size_t layer_window = get_kv_layer_windows()[layer_idx];
                    gb->kv_cache_append(k_input, graph_cache_k_nodes_[layer_idx], layer_window, cache_sink_size_);
                    gb->kv_cache_append(v_input, graph_cache_v_nodes_[layer_idx], layer_window, cache_sink_size_);
                }
            }
            gb->execute();
            cache_total_seq_len_ += actual_tokens;
        }
    }
}

double Model::score_tokens_window_logprob(
    const std::vector<uint32_t>& tokens,
    size_t start,
    size_t end,
    size_t context,
    size_t* tokens_scored
) {
    if (tokens_scored)
        *tokens_scored = 0;

    if (tokens.empty()) 
        return 0.0;

    if (end > tokens.size()) 
        end = tokens.size();

    if (start >= end) 
        return 0.0;

    if (start == 0) 
        start = 1;

    if (start >= end) 
        return 0.0;

    const size_t target_len = end - start;
    const size_t ctx_begin = (start > context) ? (start - context) : 0;

    if (end < 2) return 0.0;
    const size_t input_end = end - 1;

    if (input_end <= ctx_begin) 
        return 0.0;

    std::vector<uint32_t> input_tokens(tokens.begin() + ctx_begin,tokens.begin() + input_end);

    if (tokens_scored) 
        *tokens_scored = target_len;

    reset_cache();

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const auto backend = (config_.default_backend == Config::Backend::CPU) ? ComputeBackend::CPU : ComputeBackend::NPU;

    const size_t hidden_node = forward(input_tokens, /*use_cache=*/false);
    const auto& hidden_buf = gb->get_output_buffer(hidden_node);

    if (hidden_buf.shape.size() != 2) {
        throw std::runtime_error("Expected hidden to be rank-2 [L, hidden_dim]");
    }


    const size_t first_pos = start - ctx_begin - 1;
    const size_t hidden_slice = gb->slice(hidden_node, /*axis=*/0, first_pos, target_len);
    bool transpose_w = true;
    const size_t logits_node = gb->matmul(hidden_slice, output_weight_node_id_, transpose_w, backend);
    gb->execute();

    const auto& logits_buf = gb->get_output_buffer(logits_node);
    if (logits_buf.shape.size() != 2) 
        throw std::runtime_error("Expected logits to be rank-2 [T, vocab]");

    const size_t T = logits_buf.shape[0];
    const size_t vocab_size = logits_buf.shape[1];

    if (T != target_len)
        throw std::runtime_error("Logits T dimension does not match target_len");

    void* logits_ptr = gb->get_output(logits_node);
    std::vector<float> row(vocab_size);
    double total_logprob = 0.0;

    for (size_t i = 0; i < target_len; ++i) {
        const uint32_t y = tokens[start + i];
        if (y >= vocab_size) 
            throw std::runtime_error("Target token out of vocab range");

        if (logits_buf.precision == Precision::FP32) {
            const float* src = static_cast<const float*>(logits_ptr) + i * vocab_size;
            std::memcpy(row.data(), src, vocab_size * sizeof(float));
        } 
        else if (logits_buf.precision == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(logits_ptr) + i * vocab_size;
            Quantization::fp16_to_fp32(const_cast<__fp16*>(src), row.data(), vocab_size);
        } 
        else {
            const int8_t* src = static_cast<const int8_t*>(logits_ptr) + i * vocab_size;
            Quantization::int8_to_fp32(const_cast<int8_t*>(src), row.data(), vocab_size, 1.0f);
        }

        float max_logit = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        
        for (size_t j = 0; j < vocab_size; ++j)
            sum += std::exp(double(row[j] - max_logit));

        const double lse = double(max_logit) + std::log(sum);
        total_logprob += double(row[y]) - lse;
    }

    return total_logprob;
}
}
}
