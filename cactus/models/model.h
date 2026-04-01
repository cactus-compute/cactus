#pragma once

#include "../engine/engine.h"
#include "../npu/npu.h"

namespace cactus {
namespace engine {



class QwenModel : public Model {
public:
    QwenModel();
    explicit QwenModel(const Config& config);
    ~QwenModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t pre_feedforward_layernorm_weight;
            size_t post_feedforward_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class Qwen3p5Model : public Model {
public:
    Qwen3p5Model();
    explicit Qwen3p5Model(const Config& config);
    ~Qwen3p5Model() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;
    void reset_cache() override;

private:
    size_t build_gated_deltanet(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                ComputeBackend backend, bool use_cache, size_t position_offset);

    struct WeightNodeIDs {
        size_t output_weight = 0;
        size_t output_norm_weight = 0;

        enum class LayerType : uint8_t { ATTENTION, DELTANET };

        struct LayerWeights {
            size_t attn_q_weight = 0;
            size_t attn_k_weight = 0;
            size_t attn_v_weight = 0;
            size_t attn_output_weight = 0;
            size_t input_layernorm_weight = 0;
            size_t attn_q_norm_weight = 0;
            size_t attn_k_norm_weight = 0;
            size_t deltanet_qkv_weight = 0;
            size_t deltanet_gate_weight = 0;
            size_t deltanet_beta_weight = 0;
            size_t deltanet_gate_bias = 0;
            size_t deltanet_beta_bias = 0;
            size_t deltanet_z_weight = 0;
            size_t deltanet_conv_weight = 0;
            size_t ffn_gate_weight = 0;
            size_t ffn_up_weight = 0;
            size_t ffn_down_weight = 0;
            size_t post_attention_layernorm_weight = 0;
        };

        struct LayerEntry {
            LayerType type = LayerType::ATTENTION;
            LayerWeights weights;
        };

        std::vector<LayerEntry> layers;
    } weight_nodes_;

    ConvCache conv_cache_;
    std::vector<size_t> conv_cache_state_nodes_;
    bool last_forward_used_cache_ = false;
    size_t deltanet_total_seq_len_ = 0;
    size_t deltanet_heads_ = 0;
    size_t deltanet_key_dim_ = 0;
    size_t deltanet_value_dim_ = 0;
    size_t deltanet_state_flat_dim_ = 0;
    size_t deltanet_mixed_dim_ = 0;
    size_t deltanet_conv_history_len_ = 0;
    size_t deltanet_conv_flat_dim_ = 0;
    size_t deltanet_cache_row_dim_ = 0;
};



class GemmaModel : public Model {
public:
    GemmaModel();
    explicit GemmaModel(const Config& config);
    ~GemmaModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    void post_init() override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t pre_feedforward_layernorm_weight;
            size_t post_feedforward_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class GemmaModel3n : public Model {
public:
    GemmaModel3n();
    explicit GemmaModel3n(const Config& config);
    ~GemmaModel3n() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256, const std::string& profile_file = "") override;
    void load_weights_to_graph(CactusGraph* gb) override;
    void post_init() override;
    std::vector<size_t> get_kv_layer_dims() const override;

private:
    size_t forward_split(const std::vector<uint32_t>& tokens, bool use_cache);

    size_t build_preamble(CactusGraph* gb, size_t seq_len, ComputeBackend backend,
                          size_t& token_input, size_t& pli_input, size_t* streams);
    void build_layer(CactusGraph* gb, uint32_t layer_idx, ComputeBackend backend,
                     bool use_cache, size_t pos_offset, size_t pli, size_t* streams);
    size_t build_output_head(CactusGraph* gb, size_t* streams, ComputeBackend backend);
    void set_token_inputs(CactusGraph* gb, size_t token_input, size_t pli_input,
                          const std::vector<uint32_t>& tokens);

    size_t build_laurel(CactusGraph* gb, size_t normed_input, uint32_t layer_idx, ComputeBackend backend) const;
    size_t build_gaussian_topk(CactusGraph* gb, size_t input, float ppf) const;
    size_t build_rms_norm_no_weight(CactusGraph* gb, size_t input, size_t num_rows, size_t row_dim) const;
    size_t build_magnitude_normalize(CactusGraph* gb, size_t reference, size_t target) const;

    size_t build_altup_router_modalities(CactusGraph* gb, size_t stream0, uint32_t layer_idx, ComputeBackend backend) const;
    void build_altup_predict(CactusGraph* gb, size_t modalities, uint32_t layer_idx,
                             const size_t* streams, size_t* predictions) const;
    void build_altup_correct(CactusGraph* gb, size_t activated, size_t modalities, uint32_t layer_idx,
                             ComputeBackend backend, const size_t* predictions, size_t* corrected) const;
    void build_per_layer_input(CactusGraph* gb, size_t pli_combined, uint32_t layer_idx,
                               ComputeBackend backend, size_t* streams) const;

    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        size_t altup_proj_weights[3];
        size_t altup_unembed_proj_weights[3];
        size_t embed_tokens_per_layer;
        size_t per_layer_model_proj;
        size_t per_layer_proj_norm;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t pre_feedforward_layernorm_weight;
            size_t post_feedforward_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;

            size_t altup_router_norm;
            size_t altup_prediction_coefs;
            size_t altup_correction_coefs;
            size_t altup_correct_output_scale;
            size_t altup_modality_router;
            size_t laurel_left;
            size_t laurel_right;
            size_t laurel_norm;
            size_t per_layer_gate;
            size_t per_layer_proj;
            size_t post_per_layer_norm;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    std::vector<int> kv_share_map_;
    std::vector<size_t> shared_k_nodes_;
    std::vector<size_t> shared_v_nodes_;

    std::vector<__fp16> v_norm_ones_weight_;
    size_t v_norm_ones_node_ = 0;
};

class Siglip2VisionModel : public Model {
    friend class Lfm2VlModel;  
    
public:
    struct VisionEmbeddingResult {
        size_t combined_embeddings;
        std::vector<size_t> tile_embeddings;
    };

    Siglip2VisionModel();
    explicit Siglip2VisionModel(const Config& cfg);
    ~Siglip2VisionModel() override = default;
    virtual size_t forward_vision(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image);
    virtual size_t forward_vision(CactusGraph* gb, 
                         const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
                         ComputeBackend backend);
    std::vector<float> get_image_features(const std::string& image_path);
    std::vector<float> get_image_features(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image);
    std::vector<float> get_image_embedding(const std::string& image_path);
    size_t get_image_features_node(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image);
    Siglip2Preprocessor& get_preprocessor() { return preprocessor_; }
    const Siglip2Preprocessor& get_preprocessor() const { return preprocessor_; }

protected:
    VisionEmbeddingResult build_vision_embeddings(CactusGraph* gb,
                                                  const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
                                                  ComputeBackend backend);
    
    size_t build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                         ComputeBackend backend);
    
    size_t build_vision_attention(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                  ComputeBackend backend);
    
    size_t build_vision_mlp(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                           ComputeBackend backend);

    void load_weights_to_graph(CactusGraph* gb) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

protected:
    struct VisionWeightNodeIDs {
        size_t patch_embedding_weight;
        size_t patch_embedding_bias;
        size_t position_embedding;
        size_t post_layernorm_weight;
        size_t post_layernorm_bias;

        struct VisionLayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t attn_q_bias;
            size_t attn_k_bias;
            size_t attn_v_bias;
            size_t attn_output_bias;
            size_t layer_norm1_weight;
            size_t layer_norm1_bias;
            size_t layer_norm2_weight;
            size_t layer_norm2_bias;
            size_t mlp_fc1_weight;
            size_t mlp_fc1_bias;
            size_t mlp_fc2_weight;
            size_t mlp_fc2_bias;
        };

        std::vector<VisionLayerWeights> vision_layers;
    } vision_weight_nodes_;

    Siglip2Preprocessor preprocessor_;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
};


class LFM2Model : public Model {
    friend class Lfm2VlModel;

public:
    LFM2Model();
    explicit LFM2Model(const Config& config);
    ~LFM2Model() override = default;

    bool is_cache_empty() const;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override;
    bool init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;

protected:
    using Model::forward;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                    ComputeBackend backend, bool use_cache);

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    size_t forward(CactusGraph* gb, const std::vector<uint32_t>& tokens, ComputeBackend backend, bool use_cache = false);
    size_t forward(CactusGraph* gb, size_t input_embeddings, size_t seq_len, ComputeBackend backend, bool use_cache = false);
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;
    void reset_cache() override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:

    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
        size_t attn_q_weight;
        size_t attn_k_weight;
        size_t attn_v_weight;
        size_t attn_output_weight;
        size_t attn_q_norm_weight;   
        size_t attn_k_norm_weight;

        size_t conv_depthwise_weight;
        size_t conv_in_proj_weight;
        size_t conv_out_proj_weight;

        size_t input_layernorm_weight;
        size_t post_attention_layernorm_weight;
        size_t ffn_gate_weight;
        size_t ffn_up_weight;
        size_t ffn_down_weight;
        };

        enum class LayerType : uint8_t { ATTENTION, CONV };

        struct LayerEntry {
            LayerType type;
            LayerWeights weights;
        };

        std::vector<LayerEntry> layers;
    } weight_nodes_;

    ConvCache conv_cache_;
    std::vector<size_t> conv_cache_bx_nodes_;
    bool last_forward_used_cache_ = false;
};


class LFM2MoEModel : public Model {
public:
    LFM2MoEModel();
    explicit LFM2MoEModel(const Config& config);
    ~LFM2MoEModel() override = default;

    bool is_cache_empty() const;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override;
    bool init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;

protected:
    using Model::forward;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                        ComputeBackend backend, bool use_cache);

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    size_t forward(CactusGraph* gb, const std::vector<uint32_t>& tokens, ComputeBackend backend, bool use_cache = false);
    size_t forward(CactusGraph* gb, size_t input_embeddings, size_t seq_len, ComputeBackend backend, bool use_cache = false);
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;
    void reset_cache() override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    bool is_dense_layer(uint32_t layer_idx) const;

    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct ExpertWeights {
            size_t w1_weight = 0;
            size_t w3_weight = 0;
            size_t w2_weight = 0;
        };

        struct LayerWeights {
            size_t attn_q_weight = 0;
            size_t attn_k_weight = 0;
            size_t attn_v_weight = 0;
            size_t attn_output_weight = 0;
            size_t attn_q_norm_weight = 0;
            size_t attn_k_norm_weight = 0;

            size_t conv_depthwise_weight = 0;
            size_t conv_in_proj_weight = 0;
            size_t conv_out_proj_weight = 0;

            size_t input_layernorm_weight = 0;
            size_t post_attention_layernorm_weight = 0;
            size_t ffn_gate_weight = 0;
            size_t ffn_up_weight = 0;
            size_t ffn_down_weight = 0;

            size_t moe_router_weight = 0;
            size_t moe_expert_bias = 0;
            std::vector<ExpertWeights> moe_experts;
        };

        enum class LayerType : uint8_t { ATTENTION, CONV };

        struct LayerEntry {
            LayerType type;
            LayerWeights weights;
        };

        std::vector<LayerEntry> layers;
    } weight_nodes_;

    ConvCache conv_cache_;
    std::vector<size_t> conv_cache_bx_nodes_;
    bool last_forward_used_cache_ = false;
};


class NomicModel : public Model {
public:
    NomicModel();
    explicit NomicModel(const Config& config);
    ~NomicModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                            ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                    ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    size_t build_standard_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                ComputeBackend backend) const;

    size_t build_moe_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                        ComputeBackend backend) const;

    struct WeightNodeIDs {
        size_t embedding_layernorm_weight;
        size_t embedding_layernorm_bias;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_q_bias;
            size_t attn_k_bias;
            size_t attn_v_bias;
            size_t attn_output_weight;
            size_t attn_output_bias;
            size_t ffn_up_weight;
            size_t ffn_up_bias;
            size_t ffn_norm_1_weight;
            size_t ffn_norm_1_bias;
            size_t ffn_down_weight;
            size_t ffn_down_bias;
            size_t ffn_norm_2_weight;
            size_t ffn_norm_2_bias;
            size_t mlp_router_layer_weight;
            size_t mlp_experts_bias;
            std::vector<size_t> mlp_experts_mlp1_weight;
            std::vector<size_t> mlp_experts_mlp2_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class WhisperModel : public Model {
public:
    WhisperModel();
    explicit WhisperModel(const Config& config);
    ~WhisperModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t,ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Whisper: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>& /*tokens*/, bool /*use_cache*/ = false) override {
        throw std::runtime_error("Whisper requires mel+token forward().");
    }

    size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    void run_encoder(const std::vector<float>& audio_features);
    void reset_graph_side_cache_nodes();

    size_t run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    void load_weights_to_graph(CactusGraph* gb) override;

    size_t build_encoder_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend);
    
    size_t build_decoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const;
    
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_conv1d(CactusGraph* gb, size_t input);

    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
                                    float temperature = 0.0f, float top_p = 0.0f, size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                                    float* out_token_time_start = nullptr, float* out_token_time_end = nullptr) override;

    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    
    void reset_cache() override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        size_t decoder_norm_weight;
        size_t decoder_norm_bias;
        size_t decoder_position_embeddings_weight;

        size_t encoder_position_embeddings;
        size_t encoder_conv1_weight;
        size_t encoder_conv1_bias;
        size_t encoder_conv2_weight;
        size_t encoder_conv2_bias;
        size_t encoder_conv3_weight;
        size_t encoder_conv3_bias;
        size_t encoder_norm_weight;
        size_t encoder_norm_bias;

        size_t encoder_output;

        struct LayerWeights {
            size_t decoder_output_norm_bias;
            size_t decoder_output_norm_weight;
            size_t decoder_position_embeddings_weight;
            size_t decoder_token_embeddings_weight;

            size_t decoder_encoder_attn_q_weight;
            size_t decoder_encoder_attn_k_weight;
            size_t decoder_encoder_attn_v_weight;
            size_t decoder_encoder_attn_q_bias;
            size_t decoder_encoder_attn_v_bias;
            size_t decoder_encoder_attn_output_weight;
            size_t decoder_encoder_attn_output_bias;

            size_t decoder_post_encoder_layernorm_weight;
            size_t decoder_post_encoder_layernorm_bias;

            size_t decoder_ffn1_weight;
            size_t decoder_ffn1_bias;
            size_t decoder_ffn2_weight;
            size_t decoder_ffn2_bias;

            size_t decoder_post_ffn_layernorm_weight;
            size_t decoder_post_ffn_layernorm_bias;
            
            size_t decoder_self_attn_q_weight;
            size_t decoder_self_attn_k_weight;
            size_t decoder_self_attn_v_weight;
            size_t decoder_self_attn_q_bias;
            size_t decoder_self_attn_v_bias;
            size_t decoder_self_attn_output_weight;
            size_t decoder_self_attn_output_bias;

            size_t decoder_post_attn_layernorm_weight;
            size_t decoder_post_attn_layernorm_bias;

            //Encoder layers
            size_t encoder_ffn1_weight;
            size_t encoder_ffn1_bias;
            size_t encoder_ffn2_weight;
            size_t encoder_ffn2_bias;

            size_t encoder_post_ffn_layernorm_weight;
            size_t encoder_post_ffn_layernorm_bias;
            
            size_t encoder_self_attn_q_weight;
            size_t encoder_self_attn_k_weight;
            size_t encoder_self_attn_v_weight;
            size_t encoder_self_attn_q_bias;
            size_t encoder_self_attn_v_bias;
            size_t encoder_self_attn_output_weight;
            size_t encoder_self_attn_output_bias;

            size_t encoder_post_attn_layernorm_weight;
            size_t encoder_post_attn_layernorm_bias;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool encoder_ready_ = false;
    size_t last_new_tokens_;
    size_t last_conv1_node_ = 0;
    size_t last_conv2_node_ = 0;
    size_t last_encoder_post_norm_node_ = 0;
    size_t last_enc_plus_pos_node_ = 0;
    size_t encoder_transformer_block_0 = 0;
    size_t encoder_pre_gelu = 0;
    size_t encoder_post_gelu = 0;

    std::vector<size_t> encoder_block_out_nodes_;

    size_t encoder_output_persistent_ = 0;

    std::vector<size_t> suppress_tokens_ = {
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50358,
    50359,
    50360,
    50361,
    50362
    };

    std::vector<size_t> begin_suppress_tokens_ = {
    220,
    50257
    };

    bool first_decode_step_ = true;

    std::unordered_map<uint32_t, float> suppress_bias_;
    std::unordered_map<uint32_t, float> suppress_bias_first_step_;

    std::vector<size_t> encoder_k_persistent_;
    std::vector<size_t> encoder_v_persistent_;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;

};


class MoonshineModel : public Model {
public:
    MoonshineModel();
    explicit MoonshineModel(const Config& config);
    ~MoonshineModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t,ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Moonshine: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Moonshine: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Moonshine: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>& /*tokens*/, bool /*use_cache*/ = false) override {
        throw std::runtime_error("Moonshine requires mel+token forward().");
    }

    size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    size_t build_encoder(CactusGraph* gb, const std::vector<float>& audio_features);
    
    void reset_graph_side_cache_nodes();

    size_t build_decoder(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    void load_weights_to_graph(CactusGraph* gb) override;

    size_t build_decoder_cross_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend);
    
    size_t build_decoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const;
    
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_conv1d(CactusGraph* gb, size_t input);
    size_t build_audio_preprocessor(CactusGraph* gb, size_t input);

    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
                                    float temperature = 0.0f, float top_p = 0.0f, size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                                    float* out_token_time_start = nullptr, float* out_token_time_end = nullptr) override;

    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    
    void reset_cache() override;

private:
    struct WeightNodeIDs {
        size_t output_weight;

        size_t decoder_norm_weight;

        size_t encoder_conv1_weight;
        size_t encoder_conv2_weight;
        size_t encoder_conv2_bias;
        size_t encoder_conv3_weight;
        size_t encoder_conv3_bias;
        size_t encoder_norm_weight;
        size_t encoder_norm_bias;
        size_t encoder_layer_norm_weight;


        struct LayerWeights {
            //Decoder layers

            size_t decoder_encoder_attn_q_weight;
            size_t decoder_encoder_attn_k_weight;
            size_t decoder_encoder_attn_v_weight;
            size_t decoder_encoder_attn_output_weight;

            size_t decoder_post_encoder_layernorm_weight;

            size_t decoder_ffn1_weight;
            size_t decoder_ffn1_bias;
            size_t decoder_ffn_gate_weight;
            size_t decoder_ffn_gate_bias;
            size_t decoder_ffn_up_weight;
            size_t decoder_ffn_up_bias;
            size_t decoder_ffn2_weight;
            size_t decoder_ffn2_bias;

            size_t decoder_post_ffn_layernorm_weight;
            
            size_t decoder_self_attn_q_weight;
            size_t decoder_self_attn_k_weight;
            size_t decoder_self_attn_v_weight;
            size_t decoder_self_attn_output_weight;

            size_t decoder_post_attn_layernorm_weight;

            //Encoder layers
            size_t encoder_ffn1_weight;
            size_t encoder_ffn1_bias;
            size_t encoder_ffn_gate_weight;
            size_t encoder_ffn_gate_bias;
            size_t encoder_ffn_up_weight;
            size_t encoder_ffn_up_bias;
            size_t encoder_ffn2_weight;
            size_t encoder_ffn2_bias;

            size_t encoder_post_ffn_layernorm_weight;
            
            size_t encoder_self_attn_q_weight;
            size_t encoder_self_attn_k_weight;
            size_t encoder_self_attn_v_weight;
            size_t encoder_self_attn_output_weight;

            size_t encoder_post_attn_layernorm_weight;
        };

        std::vector<LayerWeights> encoder_layers;
        std::vector<LayerWeights> decoder_layers;
    } weight_nodes_;

    bool encoder_ready_ = false;
    size_t last_new_tokens_;
    std::vector<float> encoder_output_host_;
    std::vector<size_t> encoder_output_shape_;
    size_t last_conv1_node_ = 0;
    size_t last_conv2_node_ = 0;
    size_t last_encoder_post_norm_node_ = 0;
    size_t last_enc_plus_pos_node_ = 0;
    size_t encoder_transformer_block_0 = 0;
    size_t encoder_pre_gelu = 0;
    size_t encoder_post_gelu = 0;

    std::vector<size_t> encoder_block_out_nodes_;
    std::vector<uint8_t> encoder_output_bytes_;
    Precision encoder_output_precision_ = Precision::FP32;

    bool first_decode_step_ = true;

    std::vector<size_t> encoder_k_persistent_;
    std::vector<size_t> encoder_v_persistent_;

    Precision encoder_kv_precision_ = Precision::FP32;
    bool encoder_kv_ready_ = false;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;

};

class ParakeetModel : public Model {
public:
    ParakeetModel();
    explicit ParakeetModel(const Config& config);
    ~ParakeetModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Parakeet: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Parakeet: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Parakeet: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>& /*tokens*/, bool /*use_cache*/ = false) override {
        throw std::runtime_error("Parakeet requires audio feature forward().");
    }

    size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
                               float temperature = 0.0f, float top_p = 0.0f, size_t top_k = 0,
                               const std::string& profile_file = "", float* out_entropy = nullptr,
                               float* out_token_time_start = nullptr, float* out_token_time_end = nullptr) override;
    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    void reset_cache() override;

private:
    size_t build_encoder(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_ctc_logits(CactusGraph* gb, size_t hidden_states);
    size_t build_subsampling(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_relative_position_embeddings(CactusGraph* gb, size_t seq_len);
    size_t build_self_attention(CactusGraph* gb, size_t hidden, size_t position_embeddings, uint32_t layer_idx, ComputeBackend backend);
    size_t build_feed_forward(CactusGraph* gb, size_t hidden, uint32_t layer_idx, bool second_ff, ComputeBackend backend);
    size_t build_convolution_module(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend);
    size_t build_encoder_block(CactusGraph* gb, size_t hidden, size_t position_embeddings, uint32_t layer_idx, ComputeBackend backend);
    std::vector<uint32_t> greedy_decode_tokens(CactusGraph* gb, size_t logits_node) const;

    struct WeightNodeIDs {
        size_t ctc_head_weight = 0;
        size_t ctc_head_bias = 0;

        size_t subsampling_conv0_weight = 0;
        size_t subsampling_conv0_bias = 0;
        size_t subsampling_depthwise1_weight = 0;
        size_t subsampling_depthwise1_bias = 0;
        size_t subsampling_pointwise1_weight = 0;
        size_t subsampling_pointwise1_bias = 0;
        size_t subsampling_depthwise2_weight = 0;
        size_t subsampling_depthwise2_bias = 0;
        size_t subsampling_pointwise2_weight = 0;
        size_t subsampling_pointwise2_bias = 0;
        size_t subsampling_linear_weight = 0;
        size_t subsampling_linear_bias = 0;

        struct LayerWeights {
            size_t ff1_linear1_weight = 0;
            size_t ff1_linear1_bias = 0;
            size_t ff1_linear2_weight = 0;
            size_t ff1_linear2_bias = 0;

            size_t ff2_linear1_weight = 0;
            size_t ff2_linear1_bias = 0;
            size_t ff2_linear2_weight = 0;
            size_t ff2_linear2_bias = 0;

            size_t self_attn_q_weight = 0;
            size_t self_attn_q_bias = 0;
            size_t self_attn_k_weight = 0;
            size_t self_attn_k_bias = 0;
            size_t self_attn_v_weight = 0;
            size_t self_attn_v_bias = 0;
            size_t self_attn_output_weight = 0;
            size_t self_attn_output_bias = 0;
            size_t self_attn_relative_k_weight = 0;
            size_t self_attn_bias_u = 0;
            size_t self_attn_bias_v = 0;

            size_t norm_ff1_weight = 0;
            size_t norm_ff1_bias = 0;
            size_t norm_self_attn_weight = 0;
            size_t norm_self_attn_bias = 0;
            size_t norm_conv_weight = 0;
            size_t norm_conv_bias = 0;
            size_t norm_ff2_weight = 0;
            size_t norm_ff2_bias = 0;
            size_t norm_out_weight = 0;
            size_t norm_out_bias = 0;

            size_t conv_pointwise1_weight = 0;
            size_t conv_pointwise1_bias = 0;
            size_t conv_depthwise_weight = 0;
            size_t conv_depthwise_bias = 0;
            size_t conv_pointwise2_weight = 0;
            size_t conv_pointwise2_bias = 0;
            size_t conv_batchnorm_weight = 0;
            size_t conv_batchnorm_bias = 0;
            size_t conv_batchnorm_running_mean = 0;
            size_t conv_batchnorm_running_var = 0;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool ctc_tokens_ready_ = false;
    size_t ctc_emit_index_ = 0;
    std::vector<uint32_t> ctc_tokens_;
    size_t last_input_token_count_ = 0;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool has_cpu_encoder_weights_ = false;
};

class ParakeetTDTModel : public Model {
public:
    ParakeetTDTModel();
    explicit ParakeetTDTModel(const Config& config);
    ~ParakeetTDTModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("ParakeetTDT: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("ParakeetTDT: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("ParakeetTDT: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>& /*tokens*/, bool /*use_cache*/ = false) override {
        throw std::runtime_error("ParakeetTDT requires audio feature forward().");
    }

    size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
                               float temperature = 0.0f, float top_p = 0.0f, size_t top_k = 0,
                               const std::string& profile_file = "", float* out_entropy = nullptr,
                               float* out_token_time_start = nullptr, float* out_token_time_end = nullptr) override;
    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    void reset_cache() override;

private:
    size_t build_encoder(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_subsampling(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_relative_position_embeddings(CactusGraph* gb, size_t seq_len);
    size_t build_self_attention(CactusGraph* gb, size_t hidden, size_t position_embeddings, uint32_t layer_idx, ComputeBackend backend);
    size_t build_feed_forward(CactusGraph* gb, size_t hidden, uint32_t layer_idx, bool second_ff, ComputeBackend backend);
    size_t build_convolution_module(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend);
    size_t build_encoder_block(CactusGraph* gb, size_t hidden, size_t position_embeddings, uint32_t layer_idx, ComputeBackend backend);
    struct TDTToken { uint32_t id; float time_start; float time_end; };
    std::vector<TDTToken> greedy_decode_tdt_tokens(CactusGraph* gb, size_t encoder_hidden_node) const;

    struct WeightNodeIDs {
        size_t subsampling_conv0_weight = 0;
        size_t subsampling_conv0_bias = 0;
        size_t subsampling_depthwise1_weight = 0;
        size_t subsampling_depthwise1_bias = 0;
        size_t subsampling_pointwise1_weight = 0;
        size_t subsampling_pointwise1_bias = 0;
        size_t subsampling_depthwise2_weight = 0;
        size_t subsampling_depthwise2_bias = 0;
        size_t subsampling_pointwise2_weight = 0;
        size_t subsampling_pointwise2_bias = 0;
        size_t subsampling_linear_weight = 0;
        size_t subsampling_linear_bias = 0;

        struct LayerWeights {
            size_t ff1_linear1_weight = 0;
            size_t ff1_linear1_bias = 0;
            size_t ff1_linear2_weight = 0;
            size_t ff1_linear2_bias = 0;

            size_t ff2_linear1_weight = 0;
            size_t ff2_linear1_bias = 0;
            size_t ff2_linear2_weight = 0;
            size_t ff2_linear2_bias = 0;

            size_t self_attn_q_weight = 0;
            size_t self_attn_q_bias = 0;
            size_t self_attn_k_weight = 0;
            size_t self_attn_k_bias = 0;
            size_t self_attn_v_weight = 0;
            size_t self_attn_v_bias = 0;
            size_t self_attn_output_weight = 0;
            size_t self_attn_output_bias = 0;
            size_t self_attn_relative_k_weight = 0;
            size_t self_attn_bias_u = 0;
            size_t self_attn_bias_v = 0;

            size_t norm_ff1_weight = 0;
            size_t norm_ff1_bias = 0;
            size_t norm_self_attn_weight = 0;
            size_t norm_self_attn_bias = 0;
            size_t norm_conv_weight = 0;
            size_t norm_conv_bias = 0;
            size_t norm_ff2_weight = 0;
            size_t norm_ff2_bias = 0;
            size_t norm_out_weight = 0;
            size_t norm_out_bias = 0;

            size_t conv_pointwise1_weight = 0;
            size_t conv_pointwise1_bias = 0;
            size_t conv_depthwise_weight = 0;
            size_t conv_depthwise_bias = 0;
            size_t conv_pointwise2_weight = 0;
            size_t conv_pointwise2_bias = 0;
            size_t conv_batchnorm_weight = 0;
            size_t conv_batchnorm_bias = 0;
            size_t conv_batchnorm_running_mean = 0;
            size_t conv_batchnorm_running_var = 0;
        };

        struct PredictorLayerWeights {
            size_t weight_ih = 0;
            size_t weight_hh = 0;
            size_t bias = 0;
        };

        size_t predictor_embed = 0;
        std::vector<PredictorLayerWeights> predictor_layers;
        size_t joint_enc_weight = 0;
        size_t joint_enc_bias = 0;
        size_t joint_pred_weight = 0;
        size_t joint_pred_bias = 0;
        size_t joint_out_weight = 0;
        size_t joint_out_bias = 0;

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool tdt_tokens_ready_ = false;
    size_t tdt_emit_index_ = 0;
    std::vector<TDTToken> tdt_tokens_;
    size_t last_input_token_count_ = 0;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool has_cpu_encoder_weights_ = false;
};


class Lfm2VlModel : public Model {
public:
    Lfm2VlModel();
    explicit Lfm2VlModel(const Config& config);
    ~Lfm2VlModel() override = default;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    uint32_t decode(const std::vector<uint32_t>& tokens,
                      float temperature = -1.0f,
                      float top_p = -1.0f,
                      size_t top_k = 0,
                      const std::string& profile_file = "",
                      float* out_entropy = nullptr) override;

    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256, const std::string& profile_file = "") override;

    void prefill_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                             const std::string& profile_file = "") override;

    uint32_t decode_with_images(
        const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        float temperature = -1.0f,
        float top_p = -1.0f,
        size_t top_k = 0,
        const std::string& profile_file = "",
        float* out_entropy = nullptr) override;

    void reset_cache() override;
    std::vector<float> get_image_embeddings(const std::string& image_path) override;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override;
    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct ProjectedTileFeature {
        size_t node_id;
        size_t token_count;
    };

    struct TextEmbeddingInput {
        size_t input_node;
        std::vector<uint32_t> tokens;
    };

    struct MergedEmbeddingResult {
        size_t node_id;
        size_t seq_len;
    };

    struct ForwardImageResult {
        size_t final_hidden_node;
        size_t seq_len;
    };
    std::vector<ProjectedTileFeature> get_image_features(
        CactusGraph* gb,
        const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
        ComputeBackend backend);

    ForwardImageResult forward_images(
        CactusGraph* gb,
        const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        ComputeBackend backend,
        bool use_cache);
    size_t build_multimodal_projector(
        CactusGraph* gb,
        size_t image_features,
        size_t tile_h,
        size_t tile_w,
        ComputeBackend backend);
    size_t pixel_unshuffle(CactusGraph* gb, size_t hidden_states, size_t height, size_t width, size_t channels);
    MergedEmbeddingResult merge_image_text_embeddings(
        CactusGraph* gb,
        const std::vector<uint32_t>& tokens,
        const std::vector<std::vector<ProjectedTileFeature>>& image_embedding_nodes,
        std::vector<TextEmbeddingInput>& text_embedding_inputs);
    Siglip2VisionModel vision_tower_;
    LFM2Model language_model_;
    Siglip2Preprocessor preprocessor_;
    struct ProjectorWeights {
        size_t layer_norm_weight;
        size_t layer_norm_bias;
        size_t linear_1_weight;
        size_t linear_1_bias;
        size_t linear_2_weight;
        size_t linear_2_bias;
    } projector_weights_;
    
    bool vision_weights_loaded_ = false;
    bool language_weights_loaded_ = false;

    bool image_prefill_completed_ = false;
    size_t last_token_count_ = 0;
};

class SileroVADModel : public Model {
public:
    static constexpr size_t CONTEXT_SIZE = 64;
    static constexpr size_t CHUNK_SIZE = 512;
    static constexpr size_t REFLECT_PAD_SIZE = 64;
    static constexpr size_t HIDDEN_SIZE = 128;
    static constexpr size_t GATE_SIZE = 512;

    SileroVADModel();
    explicit SileroVADModel(const Config& config);
    ~SileroVADModel() override;

    bool init(const std::string& model_folder, size_t context_size = 0,
              const std::string& system_prompt = "", bool do_warmup = false) override;

    struct SpeechTimestamp {
        size_t start;
        size_t end;
    };

    struct SpeechTimestampsOptions {
        float threshold = 0.5f;
        float neg_threshold = 0.0f;
        int min_speech_duration_ms = 250;
        float max_speech_duration_s = std::numeric_limits<float>::infinity();
        int min_silence_duration_ms = 100;
        int speech_pad_ms = 30;
        int window_size_samples = 512;
        int min_silence_at_max_speech = 98;
        bool use_max_poss_sil_at_max_speech = true;
        int sampling_rate = 16000;
    };

    float process_chunk(const std::vector<float>& audio_chunk);
    void reset_states();
    std::vector<SpeechTimestamp> get_speech_timestamps(const std::vector<float>& audio, const SpeechTimestampsOptions& options);

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("SileroVAD: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("SileroVAD: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("SileroVAD: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>&, bool = false) override {
        throw std::runtime_error("SileroVAD: use process_chunk() instead");
    }

    size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    void build_graph();

    struct VADGraphNodes {
        size_t input;
        size_t h_prev;
        size_t c_prev;
        size_t h_new;
        size_t c_new;
        size_t output;
        size_t lstm_output;
        size_t encoder_output;
    } graph_nodes_;

    struct VADWeightNodes {
        size_t stft_basis;
        struct EncoderBlock {
            size_t conv_weight;
            size_t conv_bias;
        };
        std::vector<EncoderBlock> encoder_blocks;
        size_t lstm_weight_ih;
        size_t lstm_weight_hh;
        size_t lstm_bias_ih;
        size_t lstm_bias_hh;
        size_t output_conv_weight;
        size_t output_conv_bias;
    } weight_nodes_;

    struct VADState {
        std::vector<__fp16> h;
        std::vector<__fp16> c;
        std::vector<float> context;
        std::vector<float> input_buf;
        std::vector<__fp16> input_fp16;
    } state_;

    CactusGraph graph_;
    std::string weights_path_;
};

class YoutuModel : public Model {
public:
    YoutuModel();
    explicit YoutuModel(const Config& config);
    ~YoutuModel() override = default;

    void reset_cache() override;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    std::vector<size_t> get_kv_layer_dims() const override;
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;

private:
    KVCache v_cache_;
    std::vector<size_t> cache_v_nodes_;

    struct WeightNodeIDs {
        size_t output_weight = 0;
        size_t output_norm_weight = 0;

        struct LayerWeights {
            size_t attn_q_weight = 0;
            size_t attn_q_a_weight = 0;
            size_t attn_q_a_norm_weight = 0;
            size_t attn_q_b_weight = 0;
            size_t attn_kv_a_weight = 0;
            size_t attn_kv_a_norm_weight = 0;
            size_t attn_kv_b_weight = 0;
            size_t attn_output_weight = 0;
            size_t input_layernorm_weight = 0;
            size_t post_attention_layernorm_weight = 0;
            size_t ffn_gate_weight = 0;
            size_t ffn_up_weight = 0;
            size_t ffn_down_weight = 0;
            size_t attn_q_a_bias = 0;
            size_t attn_kv_a_bias = 0;
            size_t attn_output_bias = 0;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class PyAnnoteModel : public Model {
public:
    PyAnnoteModel();
    explicit PyAnnoteModel(const Config& config);
    ~PyAnnoteModel() override = default;

    bool init(const std::string& model_folder, size_t context_size = 0,
              const std::string& system_prompt = "", bool do_warmup = false) override;

    std::vector<float> diarize(const float* pcm_f32, size_t num_samples, size_t step_samples = 16000);

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("PyAnnote: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("PyAnnote: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("PyAnnote: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>&, bool = false) override {
        throw std::runtime_error("PyAnnote: use diarize() instead");
    }

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    void build_graph();

    struct LSTMLayerWeights {
        size_t w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd;
        size_t w_ih_bwd, w_hh_bwd, b_ih_bwd, b_hh_bwd;
    };

    struct WeightNodeIDs {
        size_t sinc_filters;
        size_t wav_norm_weight, wav_norm_bias;
        size_t norm0_weight, norm0_bias;
        size_t conv1_weight, conv1_bias;
        size_t norm1_weight, norm1_bias;
        size_t conv2_weight, conv2_bias;
        size_t norm2_weight, norm2_bias;
        LSTMLayerWeights lstm_layers[4];
        size_t linear0_weight, linear0_bias;
        size_t linear1_weight, linear1_bias;
        size_t classifier_weight, classifier_bias;
    } weight_nodes_;

    CactusGraph graph_;
    size_t audio_input_ = 0;
    size_t output_node_ = 0;
    std::vector<float> hamming_;
    std::vector<__fp16> chunk_buf_;
};

class WeSpeakerModel : public Model {
public:
    WeSpeakerModel();
    explicit WeSpeakerModel(const Config& config);
    ~WeSpeakerModel() override = default;

    bool init(const std::string& model_folder, size_t context_size = 0,
              const std::string& system_prompt = "", bool do_warmup = false) override;

    std::vector<float> embed(const float* fbank_features, size_t num_features);

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("WeSpeaker: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("WeSpeaker: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("WeSpeaker: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>&, bool = false) override {
        throw std::runtime_error("WeSpeaker: use embed() instead");
    }

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    void build_graph(size_t num_frames);

    struct ResBlockWeights {
        size_t conv1_w, conv2_w;
        size_t bn1_w, bn1_b, bn1_mean, bn1_var;
        size_t bn2_w, bn2_b, bn2_mean, bn2_var;
        size_t shortcut_conv_w;
        size_t shortcut_bn_w, shortcut_bn_b, shortcut_bn_mean, shortcut_bn_var;
        bool has_shortcut = false;
    };

    static ResBlockWeights load_resblock(CactusGraph* gb, const std::string& prefix, bool has_shortcut);
    static size_t build_resblock(CactusGraph* gb, size_t x, const ResBlockWeights& rb, bool stride2);

    struct WeightNodeIDs {
        size_t conv1_w;
        size_t bn1_w, bn1_b, bn1_mean, bn1_var;
        std::vector<ResBlockWeights> layer1, layer2, layer3, layer4;
        size_t seg1_w, seg1_b;
    } weight_nodes_;

    CactusGraph graph_;
    size_t audio_input_ = 0;
    size_t output_node_ = 0;
    size_t current_num_frames_ = 0;
    std::vector<__fp16> input_buf_;
};

}
}

#include "gemma4/model_gemma4.h"
