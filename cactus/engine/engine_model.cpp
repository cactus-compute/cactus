#include "engine.h"
#include "../models/model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
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

    Precision cache_precision = (config_.model_type == Config::ModelType::WHISPER || config_.model_type == Config::ModelType::MOONSHINE)
                               ? Precision::FP16
                               : Precision::INT8;
    kv_cache_.init(config_.num_layers, context_size, config_.attention_kv_heads, config_.attention_head_dim, cache_precision);

    size_t window_size = std::min(context_size, size_t(512));
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

    if (do_warmup && config_.model_type != Config::ModelType::WHISPER && config_.model_type != Config::ModelType::MOONSHINE) {
        std::string warmup_text = system_prompt.empty() ? "Hello" : system_prompt;
        auto warmup_tokens = tokenizer_->encode(warmup_text);
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
        update_kv_cache(gb, chunk.size());
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

uint32_t Model::decode(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                        size_t top_k, const std::string& profile_file, float* out_entropy) {

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
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k, tool_constrainer_.get_bias());

    gb->execute(profile_file);

    if (out_entropy) {
        const auto& logits_buf = gb->get_output_buffer(logits_node_id);
        void* logits_ptr = gb->get_output(logits_node_id);
        size_t vocab_size = logits_buf.shape.back();
        size_t seq_len = 1;
        if (logits_buf.shape.size() >= 2) {
            seq_len = logits_buf.shape[logits_buf.shape.size() - 2];
        }
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
        for (size_t i = 0; i < vocab_size; ++i) {
            sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
        }
        double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

        double entropy = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            double log_prob = static_cast<double>(logits[i]) - log_sum_exp;
            double prob = std::exp(log_prob);
            if (prob > 1e-10) {
                entropy -= prob * log_prob;
            }
        }

        double max_entropy = std::log(static_cast<double>(vocab_size));
        *out_entropy = static_cast<float>(entropy / max_entropy);
    }

    post_execute_updates(gb, tokens.size());
    update_kv_cache(gb, tokens.size());

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

uint32_t Model::decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& /*mel_bins*/, float temperature, float top_p, size_t top_k, const std::string& profile_file, float* out_entropy){
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy);
}

uint32_t Model::decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                     float temperature, float top_p, size_t top_k, const std::string& profile_file, float* out_entropy) {
    (void)image_paths;
    return decode(tokens, temperature, top_p, top_k, profile_file, out_entropy);
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

    kv_cache_.reset();

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
            else if (value == "bert" || value == "BERT") model_type = ModelType::NOMIC;
            else if (value == "whisper" || value == "WHISPER") model_type = ModelType::WHISPER;
            else if (value == "moonshine" || value == "MOONSHINE") model_type = ModelType::MOONSHINE;
            else if (value == "silero_vad" || value == "SILERO_VAD") model_type = ModelType::SILERO_VAD;
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
        else if (key == "enc_hidden_act") encoder_act_gelu = (value == "gelu");
        else if (key == "dec_hidden_act") decoder_act_gelu = (value == "gelu");
        else if (key == "num_encoder_layers") num_encoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "num_decoder_layers") num_decoder_layers = static_cast<uint32_t>(std::stoul(value));
        else if (key == "partial_rotary_factor") partial_rotary_factor = std::stof(value);
    }

    if (model_type == ModelType::GEMMA) {
        default_temperature = 1.0f;
        default_top_p = 0.95f;
        default_top_k = 64;
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
        default_cloud_handoff_threshold = 0.4f;
    } else if (model_type == ModelType::MOONSHINE) {
        default_temperature = 0.0f;
        default_top_p = 0.0f;
        default_top_k = 0;
        default_max_tps = 6.5f;
        default_cloud_handoff_threshold = 0.35f;
    }

    return true;
}

std::string Config::to_json() const {
    return "{}";
}

std::unique_ptr<Model> create_model(const std::string& model_folder) {
    CACTUS_LOG_DEBUG("model", "Creating model from: " << model_folder);
    Config config;
    std::string config_path = model_folder + "/config.txt";

    if (!config.from_json(config_path)) {
        CACTUS_LOG_ERROR("model", "Failed to create model - cannot load config from: " << model_folder);
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
        case Config::ModelType::NOMIC:
            return std::make_unique<NomicModel>(config);
        case Config::ModelType::WHISPER:
            return std::make_unique<WhisperModel>(config);
        case Config::ModelType::MOONSHINE:
            return std::make_unique<MoonshineModel>(config);
        case Config::ModelType::SILERO_VAD:
            return std::make_unique<SileroVADModel>(config);
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

    const int chunk_size = npu_prefill_->get_chunk_size();
    const int hidden_dim = npu_prefill_->get_hidden_dim();
    const int num_layers = npu_prefill_->get_num_layers();
    const int num_kv_heads = npu_prefill_->get_num_kv_heads();
    const int head_dim = npu_prefill_->get_head_dim();

    std::vector<__fp16> all_embeddings = get_token_embeddings(tokens);
    if (all_embeddings.empty()) {
        throw std::runtime_error("Failed to get token embeddings for NPU prefill");
    }

    if (config_.model_type == Config::ModelType::GEMMA) {
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
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                const auto& k_ref = direct_result.k_caches[layer_idx];
                const auto& v_ref = direct_result.v_caches[layer_idx];

                if (k_ref.data && v_ref.data) {
                    kv_cache_.update_from_npu(layer_idx, k_ref.data, v_ref.data,
                                               actual_tokens, num_kv_heads, head_dim);
                }
            }
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