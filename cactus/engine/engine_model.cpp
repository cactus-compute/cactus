#include "engine.h"
#include "../models/model.h"
#include "../graph/graph.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <set>
#include <sstream>

namespace cactus {
namespace engine {


Model::Model()
    : tokenizer_(nullptr),
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f),
      output_weight_node_id_(0) {
}

Model::Model(const Config& config)
    : config_(config),
      tokenizer_(nullptr),
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f),
      output_weight_node_id_(0) {
}

Model::~Model() {
    if (graph_handle_) {
        delete static_cast<CactusGraph*>(graph_handle_);
    }
}

bool Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt) {
    if (initialized_) {
        return true;
    }
    
    model_folder_path_ = model_folder;
    std::string config_path = model_folder + "/config.txt";
    
    if (!config_.from_json(config_path)) {
        return false;
    }
    
    std::string vocab_file = model_folder + "/vocab.txt";
    std::string merges_file = model_folder + "/merges.txt";
    std::string tokenizer_config_file = model_folder + "/tokenizer_config.txt";
    
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
    
    if (has_merges) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    } else {
        tokenizer_ = std::make_unique<SPTokenizer>();
    }
    
    if (!tokenizer_->load_vocabulary_with_config(vocab_file, merges_file, tokenizer_config_file)) {
        return false;
    }

    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    
    embedding_file_path_ = model_folder + "/token_embeddings.weights";

    load_weights_to_graph(gb);
    
    if (config_.model_type == Config::ModelType::GEMMA) {
        attention_scale_ = 1.0f / std::sqrt(256.0f); 
    } else {
        attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));
    }
    
    Precision cache_precision;
    std::string precision_name;
    switch (config_.precision) {
        case Config::Precision::INT8:
            cache_precision = Precision::INT8;
            precision_name = "INT8";
            break;
        case Config::Precision::FP16:
            cache_precision = Precision::FP16;
            precision_name = "FP16";
            break;
        case Config::Precision::FP32:
            cache_precision = Precision::FP32;
            precision_name = "FP32";
            break;
    }
    kv_cache_.init(config_.num_layers, context_size, config_.attention_kv_heads, config_.attention_head_dim, cache_precision);
    
    size_t window_size = std::min(context_size, size_t(1024));
    size_t sink_size = 4;
    const char* env_window = std::getenv("CACTUS_KV_WINDOW_SIZE");
    const char* env_sink = std::getenv("CACTUS_KV_SINK_SIZE");
    if (env_window) {
        window_size = std::stoul(env_window);
    }
    if (env_sink) {
        sink_size = std::stoul(env_sink);
    }
    kv_cache_.set_window_size(window_size, sink_size);
    cache_k_output_nodes_.resize(config_.num_layers);
    cache_v_output_nodes_.resize(config_.num_layers);
    
    post_init();
    
    initialized_ = true;
    
    std::string warmup_text = system_prompt.empty() ? "Henry" : system_prompt;
    auto warmup_tokens = tokenizer_->encode(warmup_text);
    forward(warmup_tokens);
    kv_cache_.reset();
    return true;
}


uint32_t Model::generate(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                        size_t top_k, const std::string& profile_file) {
                            
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

    auto logits_node_id = gb->matmul(final_hidden, output_weight_node_id_, true, backend);
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k);
    
    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }
    post_execute_updates(gb, tokens.size());
    update_kv_cache(gb, tokens.size());
    
    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

void Model::update_kv_cache(CactusGraph* gb, size_t seq_len) {
    kv_cache_.update_from_graph(gb, cache_k_output_nodes_, cache_v_output_nodes_, 
                               seq_len, config_.num_layers, config_.attention_kv_heads, 
                               config_.attention_head_dim);
}


std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& tokens, bool pooled, const std::string& profile_file) {
    auto final_hidden = forward(tokens);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto* output_ptr = gb->get_output(final_hidden);
    const auto& output_buffer = gb->get_output_buffer(final_hidden);

    std::vector<float> embeddings;

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
            float scale = pooled_buffer.quantization_scale;
            Quantization::int8_to_fp32(pooled_data, embeddings.data(), hidden_dim, scale);
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
            float scale = output_buffer.quantization_scale;
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = hidden_states[i] * scale;
            }
        }
    }

    kv_cache_.reset();

    return embeddings;
}

bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
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
        
        if (key == "vocab_size") vocab_size = std::stoul(value);
        else if (key == "bos_token_id") bos_token_id = std::stoul(value);
        else if (key == "eos_token_id") eos_token_id = std::stoul(value);
        else if (key == "num_layers") num_layers = std::stoul(value);
        else if (key == "hidden_dim") hidden_dim = std::stoul(value);
        else if (key == "ffn_intermediate_dim") ffn_intermediate_dim = std::stoul(value);
        else if (key == "attention_heads") attention_heads = std::stoul(value);
        else if (key == "attention_kv_heads") attention_kv_heads = std::stoul(value);
        else if (key == "attention_head_dim") attention_head_dim = std::stoul(value);
        else if (key == "layer_norm_eps") layer_norm_eps = std::stof(value);
        else if (key == "rope_theta") rope_theta = std::stof(value);
        else if (key == "num_experts") num_experts = std::stoul(value);
        else if (key == "num_shared_experts") num_shared_experts = std::stoul(value);
        else if (key == "num_top_experts") num_top_experts = std::stoul(value);
        else if (key == "moe_every_n_layers") moe_every_n_layers = std::stoul(value);
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "precision") {
            if (value == "INT8") precision = Precision::INT8;
            else if (value == "FP16") precision = Precision::FP16;
            else precision = Precision::FP32;
        }
        else if (key == "model_type") {
            if (value == "gemma" || value == "GEMMA") model_type = ModelType::GEMMA;
            else if (value == "lfm2" || value == "LFM2") model_type = ModelType::LFM2;
            else if (value == "smol" || value == "SMOL" || value == "Smol") model_type = ModelType::SMOL;
            else if (value == "bert" || value == "BERT") model_type = ModelType::NOMIC;
            else model_type = ModelType::QWEN;
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
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty()) layer_types.push_back(item);
                }
            }
        }
    }

    if (model_type == ModelType::GEMMA) {
        default_temperature = 1.0f;
        default_top_p = 0.95f;
        default_top_k = 64;
    } else if (model_type == ModelType::SMOL) {
        default_temperature = 0.2f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::LFM2) {
        default_temperature = 0.3f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN) {
        default_temperature = 0.7f;
        default_top_p = 0.8f;
        default_top_k = 20;
    }

    return true;
}

std::string Config::to_json() const {
    return "{}";
}

std::unique_ptr<Model> create_model(const std::string& model_folder) {
    Config config;
    std::string config_path = model_folder + "/config.txt";

    if (!config.from_json(config_path)) {
        return nullptr;
    }

    switch (config.model_type) {
        case Config::ModelType::QWEN:
            return std::make_unique<QwenModel>(config);
        case Config::ModelType::GEMMA:
            return std::make_unique<GemmaModel>(config);
        case Config::ModelType::LFM2:
            return std::make_unique<LFM2Model>(config);
        case Config::ModelType::SMOL:
            return std::make_unique<SmolModel>(config);
        case Config::ModelType::NOMIC:
            return std::make_unique<NomicModel>(config);
        default:
            return std::make_unique<QwenModel>(config);
    }
}

}
}