#pragma once

#include "../engine/engine.h"

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
class SmolModel : public Model{
public:
    SmolModel();
    explicit SmolModel(const Config& config);
    ~SmolModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    
protected:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
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


class Lfm2VlModel : public Model {
public:
    Lfm2VlModel();
    explicit Lfm2VlModel(const Config& config);
    ~Lfm2VlModel() override = default;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    uint32_t generate(const std::vector<uint32_t>& tokens,
                      float temperature = -1.0f,
                      float top_p = -1.0f,
                      size_t top_k = 0,
                      const std::string& profile_file = "") override;
    uint32_t generate_with_images(
        const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        float temperature = -1.0f,
        float top_p = -1.0f,
        size_t top_k = 0,
        const std::string& profile_file = "") override;

    void reset_cache() override;

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

}
}
