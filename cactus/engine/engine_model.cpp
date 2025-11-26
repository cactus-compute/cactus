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
#include <stdexcept>
#include <iostream>

namespace cactus {
namespace engine {


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

    graph_handle_ = gb;

    if(config_.model_type == Config::ModelType::WHISPER){
        embedding_file_path_ = model_folder+"/decoder_token_embeddings.weights";
    }
    else{
        embedding_file_path_ = model_folder + "/token_embeddings.weights";
    }

    load_weights_to_graph(gb);

    if (config_.model_type == Config::ModelType::GEMMA) {
        attention_scale_ = 1.0f / std::sqrt(256.0f);
    } else {
        attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));
    }

    Precision cache_precision;
    switch (config_.precision) {
        case Config::Precision::INT8:
            cache_precision = Precision::INT8;
            break;
        case Config::Precision::FP16:
            cache_precision = Precision::FP16;
            break;
        case Config::Precision::FP32:
            cache_precision = Precision::FP32;
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

    if (do_warmup && config_.model_type != Config::ModelType::WHISPER) {
        std::string warmup_text = system_prompt.empty() ? "Hello" : system_prompt;
        auto warmup_tokens = tokenizer_->encode(warmup_text);
        forward(warmup_tokens);
    }

    reset_cache();
    return true;
}

size_t Model::forward(const std::vector<float>& /*mel_bins*/, const std::vector<uint32_t>& tokens, bool use_cache){
    return forward(tokens, use_cache);
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

uint32_t Model::generate_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& /*mel_bins*/, float temperature, float top_p, size_t top_k, const std::string& profile_file){
    return generate(tokens, temperature, top_p, top_k, profile_file);
}

uint32_t Model::generate_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                     float temperature, float top_p, size_t top_k, const std::string& profile_file) {
    (void)image_paths;
    return generate(tokens, temperature, top_p, top_k, profile_file);
}

std::vector<float> Model::get_image_embeddings(const std::string& /*image_path*/) {
    throw std::runtime_error("Image embeddings not supported for this model type");
}

std::vector<float> Model::get_audio_embeddings(const std::vector<float>& /*mel_bins*/) {
    throw std::runtime_error("Audio embeddings not supported for this model type");
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
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "vision_hidden_dim") vision_hidden_dim = static_cast<uint32_t>(std::stoul(value));
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
            if (value == "gemma" || value == "GEMMA") model_type = ModelType::GEMMA;
            else if (value == "lfm2" || value == "LFM2") model_type = ModelType::LFM2;
            else if (value == "smol" || value == "SMOL" || value == "Smol") model_type = ModelType::SMOL;
            else if (value == "bert" || value == "BERT") model_type = ModelType::NOMIC;
            else if (value == "whisper" || value == "WHISPER") model_type = ModelType::WHISPER;
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
        default_temperature = 0.6f;
        default_top_p = 0.95f;
        default_top_k = 20;
    } else if (model_type == ModelType::QWEN) {
        default_temperature = 0.7f;
        default_top_p = 0.8f;
        default_top_k = 20;
    } else if (model_type == ModelType::WHISPER) {
        default_temperature = 0.0f;
        default_top_p = 0.0f;
        default_top_k = 0;
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

    const bool has_vision_support =
    config.use_image_tokens ||
    config.vision_num_layers > 0 ||
    config.vision_embed_dim > 0 ||
    config.vision_hidden_dim > 0 ||
    config.visual_tokens_per_img > 0;

    if (config.model_type == Config::ModelType::LFM2 && has_vision_support) {
        return std::make_unique<Lfm2VlModel>(config);
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
        case Config::ModelType::WHISPER:
            return std::make_unique<WhisperModel>(config);
        default:
            return std::make_unique<QwenModel>(config);
    }
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

}
}