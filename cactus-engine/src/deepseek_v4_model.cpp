#include "deepseek_v4_model.h"

#include "cactus_graph.h"
#include "cactus_kernels.h"

#define PICOJSON_USE_INT64
#include "picojson.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

namespace cactus {
namespace engine {
namespace fs = std::filesystem;

namespace {

bool dsv4_capture_enabled() {
    const char* v = std::getenv("CACTUS_DSV4_CAPTURE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool dsv4_capture_layer(uint32_t layer_idx) {
    const char* v = std::getenv("CACTUS_DSV4_CAPTURE_LAYERS");
    if (v == nullptr || v[0] == '\0') return layer_idx < 3;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        if (std::stoul(item) == layer_idx) return true;
    }
    return false;
}

void dsv4_capture(CactusGraph& gb, uint32_t layer_idx, const std::string& name, size_t node_id) {
    if (dsv4_capture_enabled() && dsv4_capture_layer(layer_idx)) {
        gb.capture_debug_node(layer_idx, name, node_id);
    }
}

std::vector<size_t> read_size_list(const picojson::object& cfg, const char* key) {
    std::vector<size_t> out;
    auto it = cfg.find(key);
    if (it == cfg.end() || !it->second.is<picojson::array>()) return out;
    for (const auto& v : it->second.get<picojson::array>()) {
        if (v.is<double>()) out.push_back(static_cast<size_t>(v.get<double>()));
    }
    return out;
}

std::vector<std::string> read_string_list(const picojson::object& cfg, const char* key) {
    std::vector<std::string> out;
    auto it = cfg.find(key);
    if (it == cfg.end() || !it->second.is<picojson::array>()) return out;
    for (const auto& v : it->second.get<picojson::array>()) {
        if (v.is<std::string>()) out.push_back(v.get<std::string>());
    }
    return out;
}

std::map<std::string, std::string> read_key_value_config(const fs::path& path) {
    std::map<std::string, std::string> out;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        auto trim = [](std::string& s) {
            size_t first = s.find_first_not_of(" \t");
            if (first == std::string::npos) {
                s.clear();
                return;
            }
            size_t last = s.find_last_not_of(" \t");
            s = s.substr(first, last - first + 1);
        };
        trim(key);
        trim(value);
        out[key] = value;
    }
    return out;
}

std::vector<size_t> parse_size_list(std::string value) {
    for (char& c : value) {
        if (c == '[' || c == ']' || c == '"' || c == '\'') c = ' ';
    }
    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t first = item.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        size_t last = item.find_last_not_of(" \t");
        out.push_back(static_cast<size_t>(std::stoull(item.substr(first, last - first + 1))));
    }
    return out;
}

std::vector<std::string> parse_string_list(std::string value) {
    for (char& c : value) {
        if (c == '[' || c == ']' || c == '"' || c == '\'') c = ' ';
    }
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t first = item.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        size_t last = item.find_last_not_of(" \t");
        out.push_back(item.substr(first, last - first + 1));
    }
    return out;
}

} // namespace

DeepSeekV4Model::DeepSeekV4Model() : Model(), config_copy_() {}

DeepSeekV4Model::DeepSeekV4Model(const Config& config) : Model(config), config_copy_(config) {}

bool DeepSeekV4Model::init(const std::string& model_dir, size_t context_size,
                           const std::string&, bool) {
    model_dir_ = model_dir;
    context_size_ = std::max(context_size, static_cast<size_t>(config_copy_.max_position_embeddings));
    if (!load_config(model_dir)) {
        CACTUS_LOG_ERROR("deepseek_v4", "Failed to read config.json or config.txt from " << model_dir);
        return false;
    }
    validate_architecture();
    setup_tokenizer(model_dir);
    if (!load_weight_manifest(model_dir)) {
        CACTUS_LOG_ERROR("deepseek_v4", "Failed to load DeepSeek weight manifest from " << model_dir);
        return false;
    }
    validate_required_weights();
    graph_ = std::make_unique<CactusGraph>();
    load_weights_to_graph();
    prefix_tokens_.clear();
    return true;
}

uint32_t DeepSeekV4Model::decode(const std::vector<uint32_t>& tokens,
                                 float temperature, float top_p, size_t top_k,
                                 const std::string& profile_file, float* out_entropy,
                                 float min_p, float repetition_penalty) {
    if (!graph_) throw std::runtime_error("DeepSeek graph is not initialized");
    if (tokens.empty()) throw std::runtime_error("DeepSeek decode requires at least one token");
    if (prefix_tokens_.size() + tokens.size() > context_size_) {
        throw std::runtime_error("DeepSeek decode input exceeds configured context");
    }
    if (out_entropy) *out_entropy = 0.0f;

    prefix_tokens_.insert(prefix_tokens_.end(), tokens.begin(), tokens.end());
    graph_->soft_reset_keep_pool();
    size_t hidden = build_forward(*graph_, prefix_tokens_);
    size_t last_hidden = graph_->slice(hidden, 0, prefix_tokens_.size() - 1, 1);
    size_t normed = graph_->rms_norm(last_hidden, output_norm_node_, ds_.eps);
    size_t logits = graph_->matmul(normed, output_weight_node_, true, ComputeBackend::CPU);
    if (temperature < 0.0f) temperature = 0.0f;
    if (top_p < 0.0f) top_p = 1.0f;
    size_t sample = graph_->sample_with_options(logits, temperature, top_p, min_p, repetition_penalty, top_k);
    graph_->execute(profile_file);
    const auto* output = static_cast<const uint32_t*>(graph_->get_output(sample));
    uint32_t token = output[0];
    record_sampled_token(token);
    return token;
}

bool DeepSeekV4Model::prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token) {
    if (tokens.empty()) return false;
    out_token = decode(tokens, 0.0f, 1.0f, 1);
    return true;
}

void DeepSeekV4Model::prefill(const std::vector<uint32_t>& tokens, size_t, const std::string& profile_file, bool) {
    if (tokens.empty()) return;
    if (prefix_tokens_.size() + tokens.size() > context_size_) {
        throw std::runtime_error("DeepSeek prefill exceeds configured context");
    }
    prefix_tokens_.insert(prefix_tokens_.end(), tokens.begin(), tokens.end());
    graph_->soft_reset_keep_pool();
    size_t hidden = build_forward(*graph_, prefix_tokens_);
    (void)hidden;
    graph_->execute(profile_file);
}

void DeepSeekV4Model::reset_cache() {
    prefix_tokens_.clear();
    if (graph_) {
        graph_ = std::make_unique<CactusGraph>();
        load_weights_to_graph();
    }
}

void DeepSeekV4Model::prefetch_moe_expert_pages() {
    if (!graph_) return;
    for (const auto& layer : layers_) {
        for (const auto& expert : layer.experts) {
            for (size_t node : {expert.gate, expert.up, expert.down}) {
                if (node) (void)graph_->get_output_buffer(node).byte_size;
            }
        }
    }
}

bool DeepSeekV4Model::setup_tokenizer(const std::string& model_dir) {
    std::string vocab = (fs::path(model_dir) / "vocab.txt").string();
    std::string merges = (fs::path(model_dir) / "merges.txt").string();
    std::string cfg = (fs::path(model_dir) / "tokenizer_config.txt").string();
    if (!fs::exists(cfg)) cfg = (fs::path(model_dir) / "tokenizer_config.json").string();
    if (!fs::exists(vocab)) return true;
    auto rt = load_tokenizer_runtime_config(cfg);
    bool use_bpe = rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::BPE ||
                   (rt.tokenizer_type == TokenizerRuntimeConfig::TokenizerType::UNKNOWN && fs::exists(merges));
    if (use_bpe) tokenizer_ = std::make_unique<BPETokenizer>();
    else tokenizer_ = std::make_unique<SPTokenizer>();
    return tokenizer_->load_vocabulary_with_config(vocab, merges, cfg);
}

bool DeepSeekV4Model::load_config(const std::string& model_dir) {
    const fs::path config_txt = fs::path(model_dir) / "config.txt";
    if (fs::exists(config_txt) && config_copy_.from_json(config_txt.string())) {
        config_copy_.model_type = Config::ModelType::DEEPSEEK_V4;
        const auto kv = read_key_value_config(config_txt);
        auto get_u = [&](const char* key, size_t fallback) {
            auto it = kv.find(key);
            return it == kv.end() ? fallback : static_cast<size_t>(std::stoull(it->second));
        };
        auto get_f = [&](const char* key, float fallback) {
            auto it = kv.find(key);
            return it == kv.end() ? fallback : std::stof(it->second);
        };
        ds_.vocab_size = config_copy_.vocab_size;
        ds_.num_layers = config_copy_.num_layers;
        ds_.hidden_dim = config_copy_.hidden_dim;
        ds_.hc_mult = get_u("hc_mult", 4);
        ds_.attention_heads = config_copy_.attention_heads;
        ds_.head_dim = get_u("head_dim", config_copy_.attention_head_dim ? config_copy_.attention_head_dim : config_copy_.v_head_dim);
        ds_.rope_dim = get_u("qk_rope_head_dim", config_copy_.qk_rope_head_dim ? config_copy_.qk_rope_head_dim : std::max<size_t>(2, ds_.head_dim / 2));
        ds_.q_lora_rank = config_copy_.q_lora_rank;
        ds_.o_groups = get_u("o_groups", 1);
        ds_.o_lora_rank = get_u("o_lora_rank", ds_.hidden_dim / ds_.o_groups);
        ds_.num_experts = config_copy_.num_experts;
        ds_.num_experts_per_tok = config_copy_.num_experts_per_tok;
        ds_.moe_intermediate_dim = config_copy_.moe_intermediate_dim;
        ds_.num_shared_experts = config_copy_.num_shared_experts;
        ds_.index_heads = get_u("index_n_heads", 0);
        ds_.index_head_dim = get_u("index_head_dim", 0);
        ds_.index_topk = get_u("index_topk", 0);
        ds_.sliding_window = config_copy_.sliding_window == Config::UNSET_U32 ? 0 : config_copy_.sliding_window;
        ds_.eps = config_copy_.layer_norm_eps;
        ds_.route_scale = config_copy_.routed_scaling_factor;
        ds_.swiglu_limit = get_f("swiglu_limit", 10.0f);
        ds_.rope_theta = config_copy_.rope_theta;
        ds_.compress_rope_theta = get_f("compress_rope_theta", 160000.0f);
        ds_.yarn_factor = get_f("rope_scaling_factor", get_f("yarn_factor", 16.0f));
        ds_.yarn_original_max = get_u("rope_original_max_position_embeddings", 65536);
        ds_.yarn_beta_fast = get_f("rope_yarn_beta_fast", 32.0f);
        ds_.yarn_beta_slow = get_f("rope_yarn_beta_slow", 1.0f);
        ds_.attention_compress_rates.assign(ds_.num_layers, 0);
        auto ac_it = kv.find("attention_compress_rates");
        if (ac_it == kv.end()) ac_it = kv.find("compress_ratios");
        if (ac_it != kv.end()) {
            auto rates = parse_size_list(ac_it->second);
            for (size_t i = 0; i < ds_.num_layers && i < rates.size(); ++i) ds_.attention_compress_rates[i] = rates[i];
        } else {
            auto lt_it = kv.find("layer_types");
            std::vector<std::string> types = lt_it == kv.end() ? config_copy_.layer_types : parse_string_list(lt_it->second);
            for (size_t i = 0; i < ds_.num_layers && i < types.size(); ++i) {
                if (types[i] == "compressed_sparse_attention") ds_.attention_compress_rates[i] = 4;
                else if (types[i] == "heavily_compressed_attention") ds_.attention_compress_rates[i] = 128;
            }
        }
        ds_.hash_moe_layers.assign(ds_.num_layers, false);
        auto mt_it = kv.find("mlp_layer_types");
        if (mt_it != kv.end()) {
            auto types = parse_string_list(mt_it->second);
            for (size_t i = 0; i < ds_.num_layers && i < types.size(); ++i) ds_.hash_moe_layers[i] = types[i] == "hash_moe";
        } else {
            size_t num_hash_layers = get_u("num_hash_layers", 0);
            for (size_t i = 0; i < ds_.num_layers && i < num_hash_layers; ++i) ds_.hash_moe_layers[i] = true;
        }
        return true;
    }

    const fs::path config_json = fs::path(model_dir) / "config.json";
    if (!fs::exists(config_json)) return false;
    std::ifstream in(config_json);
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty() || !root.is<picojson::object>()) {
        throw std::runtime_error("DeepSeek config.json parse failed: " + err);
    }
    const picojson::object* cfg = &root.get<picojson::object>();
    if (auto it = cfg->find("text_config"); it != cfg->end() && it->second.is<picojson::object>()) {
        cfg = &it->second.get<picojson::object>();
    }
    auto number = [&](const char* key, double fallback) {
        auto it = cfg->find(key);
        return (it != cfg->end() && it->second.is<double>()) ? it->second.get<double>() : fallback;
    };
    auto boolean = [&](const char* key, bool fallback) {
        auto it = cfg->find(key);
        return (it != cfg->end() && it->second.is<bool>()) ? it->second.get<bool>() : fallback;
    };
    auto u32 = [&](const char* key, uint32_t fallback) {
        return static_cast<uint32_t>(number(key, static_cast<double>(fallback)));
    };

    config_copy_.model_type = Config::ModelType::DEEPSEEK_V4;
    config_copy_.vocab_size = u32("vocab_size", config_copy_.vocab_size);
    config_copy_.num_layers = u32("num_hidden_layers", config_copy_.num_layers);
    config_copy_.hidden_dim = u32("hidden_size", config_copy_.hidden_dim);
    config_copy_.attention_heads = u32("num_attention_heads", config_copy_.attention_heads);
    config_copy_.attention_head_dim = u32("head_dim", u32("v_head_dim", config_copy_.attention_head_dim));
    config_copy_.q_lora_rank = u32("q_lora_rank", config_copy_.q_lora_rank);
    config_copy_.num_experts = u32("n_routed_experts", config_copy_.num_experts);
    config_copy_.num_shared_experts = u32("n_shared_experts", 1);
    config_copy_.num_experts_per_tok = u32("num_experts_per_tok", config_copy_.num_experts_per_tok);
    config_copy_.moe_intermediate_dim = u32("moe_intermediate_size", config_copy_.moe_intermediate_dim);
    config_copy_.layer_norm_eps = static_cast<float>(number("rms_norm_eps", config_copy_.layer_norm_eps));
    config_copy_.rope_theta = static_cast<float>(number("rope_theta", 10000.0));
    config_copy_.sliding_window = u32("sliding_window", 0);
    config_copy_.tie_word_embeddings = boolean("tie_word_embeddings", config_copy_.tie_word_embeddings);
    config_copy_.routed_scaling_factor = static_cast<float>(number("routed_scaling_factor", 1.0));
    config_copy_.bos_token_id = u32("bos_token_id", config_copy_.bos_token_id);
    config_copy_.eos_token_id = u32("eos_token_id", config_copy_.eos_token_id);

    ds_.vocab_size = config_copy_.vocab_size;
    ds_.num_layers = config_copy_.num_layers;
    ds_.hidden_dim = config_copy_.hidden_dim;
    ds_.attention_heads = config_copy_.attention_heads;
    ds_.head_dim = config_copy_.attention_head_dim;
    ds_.q_lora_rank = config_copy_.q_lora_rank;
    ds_.num_experts = config_copy_.num_experts;
    ds_.num_experts_per_tok = config_copy_.num_experts_per_tok;
    ds_.moe_intermediate_dim = config_copy_.moe_intermediate_dim;
    ds_.num_shared_experts = config_copy_.num_shared_experts;
    ds_.eps = config_copy_.layer_norm_eps;
    ds_.route_scale = config_copy_.routed_scaling_factor;
    ds_.rope_theta = config_copy_.rope_theta;
    ds_.sliding_window = config_copy_.sliding_window == Config::UNSET_U32 ? 0 : config_copy_.sliding_window;
    ds_.hc_mult = static_cast<size_t>(number("hc_mult", 4.0));
    ds_.rope_dim = u32("qk_rope_head_dim", static_cast<uint32_t>(std::max<size_t>(2, ds_.head_dim / 2)));
    ds_.o_groups = u32("o_groups", 1);
    ds_.o_lora_rank = u32("o_lora_rank", static_cast<uint32_t>(ds_.hidden_dim / ds_.o_groups));
    ds_.index_heads = u32("index_n_heads", 0);
    ds_.index_head_dim = u32("index_head_dim", 0);
    ds_.index_topk = u32("index_topk", 0);
    ds_.swiglu_limit = static_cast<float>(number("swiglu_limit", 10.0));
    ds_.compress_rope_theta = static_cast<float>(number("compress_rope_theta", 160000.0));

    if (auto it = cfg->find("rope_scaling"); it != cfg->end() && it->second.is<picojson::object>()) {
        const auto& rope = it->second.get<picojson::object>();
        auto rn = [&](const char* key, double fallback) {
            auto jt = rope.find(key);
            return (jt != rope.end() && jt->second.is<double>()) ? jt->second.get<double>() : fallback;
        };
        ds_.yarn_factor = static_cast<float>(rn("factor", ds_.yarn_factor));
        ds_.yarn_original_max = static_cast<size_t>(rn("original_max_position_embeddings", ds_.yarn_original_max));
        ds_.yarn_beta_fast = static_cast<float>(rn("beta_fast", ds_.yarn_beta_fast));
        ds_.yarn_beta_slow = static_cast<float>(rn("beta_slow", ds_.yarn_beta_slow));
    }

    auto compress_rates = read_size_list(*cfg, "attention_compress_rates");
    if (compress_rates.empty()) compress_rates = read_size_list(*cfg, "layer_compress_rates");
    if (compress_rates.empty()) compress_rates = read_size_list(*cfg, "compress_ratios");
    auto layer_types = read_string_list(*cfg, "layer_types");
    ds_.attention_compress_rates.assign(ds_.num_layers, 0);
    for (size_t i = 0; i < ds_.num_layers; ++i) {
        if (i < compress_rates.size()) ds_.attention_compress_rates[i] = compress_rates[i];
        else if (i < layer_types.size()) {
            if (layer_types[i] == "compressed_sparse_attention") ds_.attention_compress_rates[i] = 4;
            else if (layer_types[i] == "heavily_compressed_attention") ds_.attention_compress_rates[i] = 128;
        }
    }

    auto mlp_types = read_string_list(*cfg, "mlp_layer_types");
    ds_.hash_moe_layers.assign(ds_.num_layers, false);
    for (size_t i = 0; i < ds_.num_layers && i < mlp_types.size(); ++i) {
        ds_.hash_moe_layers[i] = mlp_types[i] == "hash_moe";
    }
    if (mlp_types.empty()) {
        const size_t num_hash_layers = static_cast<size_t>(number("num_hash_layers", 0.0));
        for (size_t i = 0; i < ds_.num_layers && i < num_hash_layers; ++i) ds_.hash_moe_layers[i] = true;
    }
    return true;
}

bool DeepSeekV4Model::load_weight_manifest(const std::string& model_dir) {
    const std::vector<fs::path> candidates = {
        fs::path(model_dir) / "conversion_manifest.json",
        fs::path(model_dir) / "weights_manifest.json",
        fs::path(model_dir) / "manifest.json",
    };
    auto normalize = [](std::string name) {
        const std::vector<std::string> prefixes = {"language_model.", "model."};
        for (const auto& p : prefixes) {
            if (name.rfind(p, 0) == 0) return name.substr(p.size());
        }
        return name;
    };
    auto add = [&](const picojson::object& entry) {
        auto name_it = entry.find("hf_name");
        if (name_it == entry.end()) name_it = entry.find("adapter_name");
        auto path_it = entry.find("path");
        if (path_it == entry.end()) path_it = entry.find("output_name");
        if (path_it == entry.end()) path_it = entry.find("output_file");
        if (name_it == entry.end() || path_it == entry.end() ||
            !name_it->second.is<std::string>() || !path_it->second.is<std::string>()) return;
        std::string name = normalize(name_it->second.get<std::string>());
        weight_manifest_[name] = path_it->second.get<std::string>();
        weight_manifest_["model." + name] = path_it->second.get<std::string>();
    };
    auto read_map = [&](const picojson::object& values) {
        for (const auto& [name, value] : values) {
            if (value.is<std::string>()) {
                std::string n = normalize(name);
                weight_manifest_[n] = value.get<std::string>();
                weight_manifest_["model." + n] = value.get<std::string>();
            } else if (value.is<picojson::object>()) {
                const auto& entry = value.get<picojson::object>();
                auto it = entry.find("path");
                if (it != entry.end() && it->second.is<std::string>()) {
                    std::string n = normalize(name);
                    weight_manifest_[n] = it->second.get<std::string>();
                    weight_manifest_["model." + n] = it->second.get<std::string>();
                }
            }
        }
    };
    for (const auto& manifest_path : candidates) {
        if (!fs::exists(manifest_path)) continue;
        std::ifstream in(manifest_path);
        picojson::value root;
        std::string err = picojson::parse(root, in);
        if (!err.empty()) continue;
        if (root.is<picojson::array>()) {
            for (const auto& v : root.get<picojson::array>()) if (v.is<picojson::object>()) add(v.get<picojson::object>());
        } else if (root.is<picojson::object>()) {
            const auto& obj = root.get<picojson::object>();
            if (auto it = obj.find("weights"); it != obj.end() && it->second.is<picojson::array>()) {
                for (const auto& v : it->second.get<picojson::array>()) if (v.is<picojson::object>()) add(v.get<picojson::object>());
            } else if (auto it = obj.find("weights"); it != obj.end() && it->second.is<picojson::object>()) {
                read_map(it->second.get<picojson::object>());
            } else if (auto it = obj.find("tensors"); it != obj.end() && it->second.is<picojson::object>()) {
                read_map(it->second.get<picojson::object>());
            } else {
                read_map(obj);
            }
        }
        if (!weight_manifest_.empty()) return true;
    }
    return false;
}

void DeepSeekV4Model::validate_architecture() const {
    if (ds_.hc_mult != 4) throw std::runtime_error("DeepSeek V4 runner requires hc_mult=4");
    if (ds_.hidden_dim == 0 || ds_.head_dim == 0 || ds_.attention_heads == 0 || ds_.q_lora_rank == 0) {
        throw std::runtime_error("DeepSeek V4 config is missing hidden/head/q_lora dimensions");
    }
    if (ds_.num_experts == 0 || ds_.num_experts_per_tok == 0 || ds_.moe_intermediate_dim == 0) {
        throw std::runtime_error("DeepSeek V4 config is missing MoE dimensions");
    }
    if (ds_.o_groups == 0 || ds_.o_lora_rank == 0 || (ds_.attention_heads * ds_.head_dim) % ds_.o_groups != 0) {
        throw std::runtime_error("DeepSeek V4 config has invalid grouped output projection dimensions");
    }
}

bool DeepSeekV4Model::has_weight(const std::string& logical_name) const {
    return weight_manifest_.find(logical_name) != weight_manifest_.end();
}

void DeepSeekV4Model::validate_required_weights() const {
    std::vector<std::vector<std::string>> required = {
        {"embed_tokens.weight", "embed.weight", "model.embed_tokens.weight"},
        {"norm.weight"},
        {"lm_head.weight", "output.weight", "head.weight"},
        {"hc_head.hc_fn", "hc_head_fn"},
        {"hc_head.hc_base", "hc_head_base"},
        {"hc_head.hc_scale", "hc_head_scale"},
    };
    auto any = [&](std::initializer_list<std::string> names) {
        required.emplace_back(names);
    };
    for (size_t i = 0; i < ds_.num_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";
        any({p + "attn_hc.fn", p + "hc_attn_fn"});
        any({p + "attn_hc.base", p + "hc_attn_base"});
        any({p + "attn_hc.scale", p + "hc_attn_scale"});
        any({p + "ffn_hc.fn", p + "hc_ffn_fn"});
        any({p + "ffn_hc.base", p + "hc_ffn_base"});
        any({p + "ffn_hc.scale", p + "hc_ffn_scale"});
        any({p + "input_layernorm.weight", p + "attn_norm.weight"});
        any({p + "post_attention_layernorm.weight", p + "ffn_norm.weight"});
        any({p + "self_attn.q_a_proj.weight", p + "attn.wq_a.weight"});
        any({p + "self_attn.q_a_norm.weight", p + "self_attn.q_a_layernorm.weight", p + "attn.q_norm.weight"});
        any({p + "self_attn.q_b_proj.weight", p + "attn.wq_b.weight"});
        any({p + "self_attn.kv_proj.weight", p + "attn.wkv.weight"});
        any({p + "self_attn.kv_norm.weight", p + "self_attn.kv_layernorm.weight", p + "attn.kv_norm.weight"});
        any({p + "self_attn.o_a_proj.weight", p + "attn.wo_a.weight"});
        any({p + "self_attn.o_b_proj.weight", p + "attn.wo_b.weight"});
        any({p + "self_attn.sinks", p + "self_attn.attn_sinks", p + "attn.attn_sink"});
        any({p + "mlp.gate.weight", p + "ffn.gate.weight"});
        for (size_t expert = 0; expert < ds_.num_experts; ++expert) {
            any({p + "ffn.experts." + std::to_string(expert) + ".w1.weight", p + "mlp.experts.gate_up_proj"});
            any({p + "ffn.experts." + std::to_string(expert) + ".w2.weight", p + "mlp.experts.down_proj"});
            any({p + "ffn.experts." + std::to_string(expert) + ".w3.weight", p + "mlp.experts.gate_up_proj"});
        }
        any({p + "mlp.shared_experts.gate_proj.weight", p + "ffn.shared_experts.w1.weight"});
        any({p + "mlp.shared_experts.up_proj.weight", p + "ffn.shared_experts.w3.weight"});
        any({p + "mlp.shared_experts.down_proj.weight", p + "ffn.shared_experts.w2.weight"});
        if (ds_.attention_compress_rates[i] == 128) {
            any({p + "self_attn.compressor.kv_proj.weight", p + "attn.compressor.wkv.weight"});
            any({p + "self_attn.compressor.gate_proj.weight", p + "attn.compressor.wgate.weight"});
            any({p + "self_attn.compressor.norm.weight", p + "attn.compressor.norm.weight"});
            any({p + "self_attn.compressor.position_bias", p + "self_attn.compressor.position_embeddings", p + "attn.compressor.ape"});
        } else if (ds_.attention_compress_rates[i] == 4) {
            any({p + "self_attn.compressor.kv_proj.weight", p + "attn.compressor.wkv.weight"});
            any({p + "self_attn.compressor.gate_proj.weight", p + "attn.compressor.wgate.weight"});
            any({p + "self_attn.compressor.norm.weight", p + "attn.compressor.norm.weight"});
            any({p + "self_attn.compressor.position_bias", p + "self_attn.compressor.position_embeddings", p + "attn.compressor.ape"});
            any({p + "self_attn.compressor.indexer.kv_proj.weight", p + "attn.indexer.compressor.wkv.weight"});
            any({p + "self_attn.compressor.indexer.gate_proj.weight", p + "attn.indexer.compressor.wgate.weight"});
            any({p + "self_attn.compressor.indexer.norm.weight", p + "attn.indexer.compressor.norm.weight"});
            any({p + "self_attn.compressor.indexer.q_b_proj.weight", p + "attn.indexer.wq_b.weight"});
            any({p + "self_attn.compressor.indexer.weights_proj.weight", p + "attn.indexer.weights_proj.weight"});
            any({p + "self_attn.compressor.indexer.position_bias", p + "self_attn.compressor.indexer.position_embeddings", p + "attn.indexer.compressor.ape"});
        }
        if (ds_.hash_moe_layers[i]) any({p + "mlp.gate.tid2eid", p + "ffn.gate.tid2eid"});
        else any({p + "mlp.gate.e_score_correction_bias", p + "ffn.gate.bias"});
    }
    std::vector<std::string> missing;
    for (const auto& group : required) {
        bool found = false;
        for (const auto& name : group) {
            if (has_weight(name) || has_weight("model." + name)) { found = true; break; }
        }
        if (!found && !group.empty()) missing.push_back(group.front());
    }
    if (missing.empty()) return;
    std::ostringstream os;
    os << "DeepSeek weight bundle is incomplete: missing manifest entries [";
    for (size_t i = 0; i < std::min<size_t>(missing.size(), 24); ++i) {
        if (i) os << ", ";
        os << missing[i];
    }
    if (missing.size() > 24) os << ", ... +" << (missing.size() - 24) << " more";
    os << "]";
    throw std::runtime_error(os.str());
}

std::string DeepSeekV4Model::weight_path(const std::string& logical_name) const {
    auto it = weight_manifest_.find(logical_name);
    if (it == weight_manifest_.end()) it = weight_manifest_.find("model." + logical_name);
    if (it == weight_manifest_.end()) throw std::runtime_error("Missing DeepSeek tensor in manifest: " + logical_name);
    fs::path p(it->second);
    if (p.is_absolute()) return p.string();
    return (fs::path(model_dir_) / p).string();
}

size_t DeepSeekV4Model::mmap_weight(CactusGraph& gb, const std::string& logical_name) {
    return gb.mmap_weights(weight_path(logical_name));
}

size_t DeepSeekV4Model::mmap_weight_any(CactusGraph& gb, const std::vector<std::string>& logical_names) {
    for (const auto& name : logical_names) {
        if (has_weight(name) || has_weight("model." + name)) return mmap_weight(gb, name);
    }
    throw std::runtime_error("Missing DeepSeek tensor in manifest: " + logical_names.front());
}

void DeepSeekV4Model::load_weights_to_graph() {
    auto& gb = *graph_;
    embedding_node_ = mmap_weight_any(gb, {"embed_tokens.weight", "model.embed_tokens.weight", "embed.weight"});
    head_hc_.fn = mmap_weight_any(gb, {"hc_head.hc_fn", "hc_head_fn"});
    head_hc_.base = mmap_weight_any(gb, {"hc_head.hc_base", "hc_head_base"});
    head_hc_.scale = mmap_weight_any(gb, {"hc_head.hc_scale", "hc_head_scale"});
    output_norm_node_ = mmap_weight(gb, "norm.weight");
    output_weight_node_ = mmap_weight_any(gb, {"lm_head.weight", "output.weight", "head.weight"});

    layers_.clear();
    layers_.resize(ds_.num_layers);
    for (size_t i = 0; i < ds_.num_layers; ++i) {
        auto& l = layers_[i];
        std::string p = "layers." + std::to_string(i) + ".";
        l.attn_hc.fn = mmap_weight_any(gb, {p + "attn_hc.fn", p + "hc_attn_fn"});
        l.attn_hc.base = mmap_weight_any(gb, {p + "attn_hc.base", p + "hc_attn_base"});
        l.attn_hc.scale = mmap_weight_any(gb, {p + "attn_hc.scale", p + "hc_attn_scale"});
        l.ffn_hc.fn = mmap_weight_any(gb, {p + "ffn_hc.fn", p + "hc_ffn_fn"});
        l.ffn_hc.base = mmap_weight_any(gb, {p + "ffn_hc.base", p + "hc_ffn_base"});
        l.ffn_hc.scale = mmap_weight_any(gb, {p + "ffn_hc.scale", p + "hc_ffn_scale"});
        l.input_norm = mmap_weight_any(gb, {p + "input_layernorm.weight", p + "attn_norm.weight"});
        l.post_norm = mmap_weight_any(gb, {p + "post_attention_layernorm.weight", p + "ffn_norm.weight"});
        l.q_a = mmap_weight_any(gb, {p + "self_attn.q_a_proj.weight", p + "attn.wq_a.weight"});
        l.q_a_norm = mmap_weight_any(gb, {p + "self_attn.q_a_norm.weight", p + "self_attn.q_a_layernorm.weight", p + "attn.q_norm.weight"});
        l.q_b = mmap_weight_any(gb, {p + "self_attn.q_b_proj.weight", p + "attn.wq_b.weight"});
        l.kv = mmap_weight_any(gb, {p + "self_attn.kv_proj.weight", p + "attn.wkv.weight"});
        l.kv_norm = mmap_weight_any(gb, {p + "self_attn.kv_norm.weight", p + "self_attn.kv_layernorm.weight", p + "attn.kv_norm.weight"});
        l.o_a = mmap_weight_any(gb, {p + "self_attn.o_a_proj.weight", p + "attn.wo_a.weight"});
        l.o_b = mmap_weight_any(gb, {p + "self_attn.o_b_proj.weight", p + "attn.wo_b.weight"});
        l.attn_sink = mmap_weight_any(gb, {p + "self_attn.sinks", p + "self_attn.attn_sinks", p + "attn.attn_sink"});
        if (ds_.attention_compress_rates[i] == 128 || ds_.attention_compress_rates[i] == 4) {
            l.hca_kv = l.csa_kv = mmap_weight_any(gb, {p + "self_attn.compressor.kv_proj.weight", p + "attn.compressor.wkv.weight"});
            l.hca_gate = l.csa_gate = mmap_weight_any(gb, {p + "self_attn.compressor.gate_proj.weight", p + "attn.compressor.wgate.weight"});
            l.hca_norm = l.csa_norm = mmap_weight_any(gb, {p + "self_attn.compressor.norm.weight", p + "attn.compressor.norm.weight"});
            l.hca_pos = l.csa_pos = mmap_weight_any(gb, {p + "self_attn.compressor.position_bias",
                                                          p + "self_attn.compressor.position_embeddings",
                                                          p + "attn.compressor.ape"});
        }
        if (ds_.attention_compress_rates[i] == 4) {
            l.idx_kv = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.kv_proj.weight", p + "attn.indexer.compressor.wkv.weight"});
            l.idx_gate = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.gate_proj.weight", p + "attn.indexer.compressor.wgate.weight"});
            l.idx_norm = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.norm.weight", p + "attn.indexer.compressor.norm.weight"});
            l.idx_q_b = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.q_b_proj.weight", p + "attn.indexer.wq_b.weight"});
            l.idx_weights = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.weights_proj.weight", p + "attn.indexer.weights_proj.weight"});
            l.idx_pos = mmap_weight_any(gb, {p + "self_attn.compressor.indexer.position_bias",
                                             p + "self_attn.compressor.indexer.position_embeddings",
                                             p + "attn.indexer.compressor.ape"});
        }
        l.router = mmap_weight_any(gb, {p + "mlp.gate.weight", p + "ffn.gate.weight"});
        if (ds_.hash_moe_layers[i]) l.tid2eid = mmap_weight_any(gb, {p + "mlp.gate.tid2eid", p + "ffn.gate.tid2eid"});
        else l.router_bias = mmap_weight_any(gb, {p + "mlp.gate.e_score_correction_bias", p + "ffn.gate.bias"});

        l.experts.clear();
        l.experts.resize(ds_.num_experts);
        const bool has_packed_experts = has_weight(p + "mlp.experts.gate_up_proj") || has_weight("model." + p + "mlp.experts.gate_up_proj");
        if (has_packed_experts) {
            l.experts_gate_up = mmap_weight(gb, p + "mlp.experts.gate_up_proj");
            l.experts_down = mmap_weight(gb, p + "mlp.experts.down_proj");
        } else {
            for (size_t expert = 0; expert < ds_.num_experts; ++expert) {
                const std::string ep = p + "ffn.experts." + std::to_string(expert) + ".";
                l.experts[expert].gate = mmap_weight(gb, ep + "w1.weight");
                l.experts[expert].down = mmap_weight(gb, ep + "w2.weight");
                l.experts[expert].up = mmap_weight(gb, ep + "w3.weight");
            }
        }
        l.shared.gate = mmap_weight_any(gb, {p + "mlp.shared_experts.gate_proj.weight", p + "ffn.shared_experts.w1.weight"});
        l.shared.up = mmap_weight_any(gb, {p + "mlp.shared_experts.up_proj.weight", p + "ffn.shared_experts.w3.weight"});
        l.shared.down = mmap_weight_any(gb, {p + "mlp.shared_experts.down_proj.weight", p + "ffn.shared_experts.w2.weight"});
    }
}

size_t DeepSeekV4Model::build_forward(CactusGraph& gb, const std::vector<uint32_t>& tokens) {
    const size_t seq_len = tokens.size();
    std::vector<float> token_ids(tokens.begin(), tokens.end());
    std::vector<float> pos(seq_len);
    for (size_t i = 0; i < seq_len; ++i) pos[i] = static_cast<float>(i);
    size_t token_input = gb.input({seq_len}, Precision::FP32);
    size_t position_input = gb.input({1, seq_len}, Precision::FP32);
    gb.set_input(token_input, token_ids.data(), Precision::FP32);
    gb.set_input(position_input, pos.data(), Precision::FP32);

    size_t hidden = gb.embedding(embedding_node_, token_input);
    if (dsv4_capture_enabled()) gb.capture_debug_node(0, "embed", hidden);
    size_t one_stream = gb.reshape(hidden, {seq_len, 1, ds_.hidden_dim});
    size_t streams = gb.cat({one_stream, one_stream, one_stream, one_stream}, 1);
    if (dsv4_capture_enabled()) gb.capture_debug_node(0, "streams_initial", streams);
    for (uint32_t layer_idx = 0; layer_idx < ds_.num_layers; ++layer_idx) {
        streams = build_layer(gb, streams, token_input, position_input, layer_idx, seq_len);
    }
    size_t collapsed = gb.dsv4_hc_head(streams, head_hc_.fn, head_hc_.base, head_hc_.scale, ds_.eps);
    if (dsv4_capture_enabled()) gb.capture_debug_node(static_cast<uint32_t>(ds_.num_layers), "hc_head", collapsed);
    return collapsed;
}

size_t DeepSeekV4Model::build_layer(CactusGraph& gb, size_t streams, size_t token_input,
                                    size_t position_input, uint32_t layer_idx, size_t seq_len) {
    const auto& l = layers_[layer_idx];
    dsv4_capture(gb, layer_idx, "streams_in", streams);
    size_t mix_a = gb.dsv4_hc_mix(streams, l.attn_hc.fn, l.attn_hc.base, l.attn_hc.scale, ds_.eps, 20);
    dsv4_capture(gb, layer_idx, "attn_hc_mix", mix_a);
    size_t attn_collapsed = gb.dsv4_hc_collapse(streams, mix_a);
    dsv4_capture(gb, layer_idx, "attn_collapsed", attn_collapsed);
    size_t attn_in = gb.rms_norm(attn_collapsed, l.input_norm, ds_.eps);
    dsv4_capture(gb, layer_idx, "attn_normed", attn_in);
    size_t q_residual = gb.rms_norm(gb.matmul(attn_in, l.q_a, true, ComputeBackend::CPU), l.q_a_norm, ds_.eps);
    dsv4_capture(gb, layer_idx, "q_residual", q_residual);
    size_t attn_out = build_attention(gb, attn_in, q_residual, position_input, layer_idx, seq_len);
    dsv4_capture(gb, layer_idx, "attn_out", attn_out);
    streams = gb.dsv4_hc_post(attn_out, streams, mix_a);
    dsv4_capture(gb, layer_idx, "streams_after_attn", streams);

    size_t mix_f = gb.dsv4_hc_mix(streams, l.ffn_hc.fn, l.ffn_hc.base, l.ffn_hc.scale, ds_.eps, 20);
    dsv4_capture(gb, layer_idx, "ffn_hc_mix", mix_f);
    size_t ffn_collapsed = gb.dsv4_hc_collapse(streams, mix_f);
    dsv4_capture(gb, layer_idx, "ffn_collapsed", ffn_collapsed);
    size_t ffn_in = gb.rms_norm(ffn_collapsed, l.post_norm, ds_.eps);
    dsv4_capture(gb, layer_idx, "ffn_normed", ffn_in);
    size_t ffn_out = build_moe(gb, ffn_in, token_input, layer_idx);
    dsv4_capture(gb, layer_idx, "ffn_out", ffn_out);
    size_t out = gb.dsv4_hc_post(ffn_out, streams, mix_f);
    dsv4_capture(gb, layer_idx, "streams_after_ffn", out);
    return out;
}

size_t DeepSeekV4Model::grouped_o_a(CactusGraph& gb, size_t attn_flat, size_t o_a_proj) const {
    return gb.dsv4_grouped_linear(attn_flat, o_a_proj, ds_.o_groups);
}

size_t DeepSeekV4Model::build_static_indices(CactusGraph& gb, size_t seq_len, size_t width, size_t compression_ratio) const {
    std::vector<float> idx(seq_len * width, -1.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        size_t first = 0;
        if (ds_.sliding_window > 0 && t + 1 > ds_.sliding_window) first = t + 1 - ds_.sliding_window;
        for (size_t j = first; j <= t; ++j) idx[t * width + j] = static_cast<float>(j);
        if (compression_ratio > 1) {
            size_t extra = width > seq_len ? width - seq_len : 0;
            for (size_t e = 0; e < extra; ++e) {
                if (e < (t + 1) / compression_ratio) idx[t * width + seq_len + e] = static_cast<float>(seq_len + e);
            }
        }
    }
    size_t node = gb.input({1, seq_len, width}, Precision::FP32);
    gb.set_input(node, idx.data(), Precision::FP32);
    return node;
}

size_t DeepSeekV4Model::build_attention(CactusGraph& gb, size_t attn_in, size_t q_residual,
                                        size_t position_input, uint32_t layer_idx, size_t seq_len) {
    const auto& l = layers_[layer_idx];
    const size_t ratio = ds_.attention_compress_rates[layer_idx];
    const bool compressed = ratio == 4 || ratio == 128;
    size_t q = gb.matmul(q_residual, l.q_b, true, ComputeBackend::CPU);
    dsv4_capture(gb, layer_idx, "attn_q_b", q);
    q = gb.reshape(q, {1, seq_len, ds_.attention_heads, ds_.head_dim});
    q = gb.dsv4_rms_norm(q, ds_.eps);
    dsv4_capture(gb, layer_idx, "attn_q_normed", q);
    q = gb.dsv4_rope(q, ds_.rope_dim, compressed ? ds_.compress_rope_theta : ds_.rope_theta,
                     0, compressed, compressed ? ds_.yarn_factor : 1.0f,
                     ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false);
    dsv4_capture(gb, layer_idx, "attn_q_rope", q);

    size_t kv = gb.rms_norm(gb.matmul(attn_in, l.kv, true, ComputeBackend::CPU), l.kv_norm, ds_.eps);
    dsv4_capture(gb, layer_idx, "attn_kv_normed", kv);
    kv = gb.reshape(kv, {1, seq_len, 1, ds_.head_dim});
    kv = gb.dsv4_rope(kv, ds_.rope_dim, compressed ? ds_.compress_rope_theta : ds_.rope_theta,
                      0, compressed, compressed ? ds_.yarn_factor : 1.0f,
                      ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false);
    dsv4_capture(gb, layer_idx, "attn_kv_rope", kv);
    kv = gb.reshape(kv, {1, seq_len, ds_.head_dim});

    size_t attn_idx = build_static_indices(gb, seq_len, seq_len, 1);
    if (ratio == 128) {
        const size_t comp_len = seq_len / 128;
        if (comp_len > 0) {
            size_t ck = gb.matmul(attn_in, l.hca_kv, true, ComputeBackend::CPU);
            size_t cg = gb.matmul(attn_in, l.hca_gate, true, ComputeBackend::CPU);
            ck = gb.reshape(ck, {1, seq_len, ds_.head_dim});
            cg = gb.reshape(cg, {1, seq_len, ds_.head_dim});
            size_t comp = gb.dsv4_compress_hca(ck, cg, l.hca_pos, l.hca_norm, ds_.eps, 128);
            dsv4_capture(gb, layer_idx, "hca_comp", comp);
            comp = gb.reshape(comp, {1, comp_len, 1, ds_.head_dim});
            comp = gb.dsv4_rope(comp, ds_.rope_dim, ds_.compress_rope_theta, 0, true, ds_.yarn_factor,
                                ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false, 128);
            comp = gb.reshape(comp, {1, comp_len, ds_.head_dim});
            kv = gb.cat({kv, comp}, 1);
            attn_idx = build_static_indices(gb, seq_len, seq_len + comp_len, 128);
        }
    } else if (ratio == 4) {
        const size_t comp_len = seq_len / 4;
        if (comp_len > 0) {
            size_t ck = gb.matmul(attn_in, l.csa_kv, true, ComputeBackend::CPU);
            size_t cg = gb.matmul(attn_in, l.csa_gate, true, ComputeBackend::CPU);
            ck = gb.reshape(ck, {1, seq_len, 2 * ds_.head_dim});
            cg = gb.reshape(cg, {1, seq_len, 2 * ds_.head_dim});
            size_t comp = gb.dsv4_compress_csa(ck, cg, l.csa_pos, l.csa_norm, ds_.eps, 4);
            dsv4_capture(gb, layer_idx, "csa_comp", comp);
            comp = gb.reshape(comp, {1, comp_len, 1, ds_.head_dim});
            comp = gb.dsv4_rope(comp, ds_.rope_dim, ds_.compress_rope_theta, 0, true, ds_.yarn_factor,
                                ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false, 4);
            comp = gb.reshape(comp, {1, comp_len, ds_.head_dim});
            kv = gb.cat({kv, comp}, 1);

            size_t ik = gb.matmul(attn_in, l.idx_kv, true, ComputeBackend::CPU);
            size_t ig = gb.matmul(attn_in, l.idx_gate, true, ComputeBackend::CPU);
            ik = gb.reshape(ik, {1, seq_len, 2 * ds_.index_head_dim});
            ig = gb.reshape(ig, {1, seq_len, 2 * ds_.index_head_dim});
            size_t icomp = gb.dsv4_compress_csa(ik, ig, l.idx_pos, l.idx_norm, ds_.eps, 4);
            dsv4_capture(gb, layer_idx, "csa_index_comp", icomp);
            icomp = gb.reshape(icomp, {1, comp_len, 1, ds_.index_head_dim});
            icomp = gb.dsv4_rope(icomp, ds_.rope_dim, ds_.compress_rope_theta, 0, true, ds_.yarn_factor,
                                 ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false, 4);
            icomp = gb.reshape(icomp, {1, comp_len, ds_.index_head_dim});
            size_t iq = gb.matmul(q_residual, l.idx_q_b, true, ComputeBackend::CPU);
            iq = gb.reshape(iq, {1, seq_len, ds_.index_heads, ds_.index_head_dim});
            iq = gb.dsv4_rope(iq, ds_.rope_dim, ds_.compress_rope_theta, 0, true, ds_.yarn_factor,
                              ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, false);
            size_t weights = gb.matmul(attn_in, l.idx_weights, true, ComputeBackend::CPU);
            weights = gb.reshape(weights, {1, seq_len, ds_.index_heads});
            size_t dyn_idx = gb.dsv4_indexer_topk(iq, icomp, weights, position_input,
                                                  std::min(ds_.index_topk, comp_len), 4, seq_len,
                                                  1.0f / std::sqrt(static_cast<float>(ds_.index_heads * ds_.index_head_dim)));
            dsv4_capture(gb, layer_idx, "csa_dyn_idx", dyn_idx);
            attn_idx = gb.cat({attn_idx, dyn_idx}, 2);
        }
    }

    dsv4_capture(gb, layer_idx, "attn_idx", attn_idx);
    dsv4_capture(gb, layer_idx, "attn_kv_final", kv);
    size_t attn = gb.dsv4_sparse_attention(q, kv, l.attn_sink, attn_idx, attention_softmax_scale());
    dsv4_capture(gb, layer_idx, "attn_sparse", attn);
    attn = gb.dsv4_rope(attn, ds_.rope_dim, compressed ? ds_.compress_rope_theta : ds_.rope_theta,
                        0, compressed, compressed ? ds_.yarn_factor : 1.0f,
                        ds_.yarn_original_max, ds_.yarn_beta_fast, ds_.yarn_beta_slow, true);
    dsv4_capture(gb, layer_idx, "attn_unrope", attn);
    attn = gb.reshape(attn, {seq_len, ds_.attention_heads * ds_.head_dim});
    dsv4_capture(gb, layer_idx, "attn_flat", attn);
    size_t o_a = grouped_o_a(gb, attn, l.o_a);
    dsv4_capture(gb, layer_idx, "attn_o_a", o_a);
    size_t out = gb.matmul(o_a, l.o_b, true, ComputeBackend::CPU);
    dsv4_capture(gb, layer_idx, "attn_o_b", out);
    return out;
}

size_t DeepSeekV4Model::build_moe(CactusGraph& gb, size_t normalized_input, size_t token_input, uint32_t layer_idx) {
    const auto& l = layers_[layer_idx];
    size_t route = ds_.hash_moe_layers[layer_idx]
        ? gb.dsv4_hash_router(normalized_input, token_input, l.router, l.tid2eid,
                              ds_.num_experts, ds_.num_experts_per_tok, ds_.route_scale, 1e-20f)
        : gb.dsv4_router_topk(normalized_input, l.router, l.router_bias,
                              ds_.num_experts, ds_.num_experts_per_tok, ds_.route_scale, 1e-20f);
    std::vector<size_t> eg, eu, ed;
    eg.reserve(ds_.num_experts);
    eu.reserve(ds_.num_experts);
    ed.reserve(ds_.num_experts);
    if (l.experts_gate_up && l.experts_down) {
        for (size_t expert = 0; expert < ds_.num_experts; ++expert) {
            size_t gu = gb.reshape(gb.slice(l.experts_gate_up, 0, expert, 1),
                                   {2 * ds_.moe_intermediate_dim, ds_.hidden_dim});
            eg.push_back(gb.slice(gu, 0, 0, ds_.moe_intermediate_dim));
            eu.push_back(gb.slice(gu, 0, ds_.moe_intermediate_dim, ds_.moe_intermediate_dim));
            ed.push_back(gb.reshape(gb.slice(l.experts_down, 0, expert, 1),
                                    {ds_.hidden_dim, ds_.moe_intermediate_dim}));
        }
    } else {
        for (size_t expert = 0; expert < ds_.num_experts; ++expert) {
            eg.push_back(l.experts[expert].gate);
            eu.push_back(l.experts[expert].up);
            ed.push_back(l.experts[expert].down);
        }
    }
    size_t routed = gb.dsv4_moe_layer(normalized_input, route, eg, eu, ed,
                                      ds_.num_experts, ds_.num_experts_per_tok, ds_.swiglu_limit);
    size_t shared = gb.dsv4_shared_expert(normalized_input, l.shared.gate, l.shared.up, l.shared.down,
                                          ds_.swiglu_limit);
    return gb.add(routed, shared);
}

float DeepSeekV4Model::attention_softmax_scale() const {
    return 1.0f / std::sqrt(static_cast<float>(ds_.head_dim));
}

} // namespace engine
} // namespace cactus
