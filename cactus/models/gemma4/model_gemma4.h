#pragma once

#include "../model.h"

bool test_gemma4_vision(bool expect_npu);
bool test_gemma4_audio(bool expect_npu);

namespace cactus {
namespace engine {

class Gemma4Model : public Model {
    friend class Gemma4MmModel;
public:
    Gemma4Model();
    explicit Gemma4Model(const Config& config);
    ~Gemma4Model() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_moe(CactusGraph* gb, size_t input, uint32_t layer_idx,
                    ComputeBackend backend) const;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256, const std::string& profile_file = "") override;
    void load_weights_to_graph(CactusGraph* gb) override;
    void post_init() override;
    std::vector<size_t> get_kv_layer_dims() const override;
    std::vector<size_t> get_kv_layer_heads() const override;
    void compact_kv_cache() override;

    size_t forward_from_embeddings(CactusGraph* gb, size_t hidden, const std::vector<uint32_t>& pli_tokens,
                                   size_t seq_len, ComputeBackend backend, bool use_cache);
    size_t forward_from_embeddings(CactusGraph* gb, size_t hidden, size_t pli_hidden_source,
                                   const std::vector<uint32_t>& pli_tokens, size_t seq_len,
                                   ComputeBackend backend, bool use_cache);
    size_t build_pli_combined_from_tokens(CactusGraph* gb, size_t hidden,
                                          const std::vector<uint32_t>& pli_tokens,
                                          size_t seq_len, ComputeBackend backend);

private:
    size_t forward_split(const std::vector<uint32_t>& tokens, bool use_cache);

    std::pair<size_t, size_t> build_preamble_and_embed(CactusGraph* gb, size_t seq_len, ComputeBackend backend,
                                                       size_t& token_input, size_t& pli_input);

    void set_token_inputs(CactusGraph* gb, size_t token_input, size_t pli_input,
                          const std::vector<uint32_t>& tokens);

    size_t build_pli_combined(CactusGraph* gb, size_t hidden, size_t pli_embed,
                              size_t seq_len, ComputeBackend backend);

    size_t build_per_layer_input(CactusGraph* gb, size_t hidden, size_t pli_combined, uint32_t layer_idx,
                               ComputeBackend backend) const;

    bool is_global_layer(uint32_t idx) const;
    size_t apply_partial_rope(CactusGraph* gb, size_t tensor, size_t head_dim, size_t rot_dim,
                              float rope_freq, size_t position_offset);
    size_t apply_transformer_layer(CactusGraph* gb, size_t hidden, size_t pli, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache, size_t pos_offset);

    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

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
            size_t per_layer_gate;
            size_t per_layer_proj;
            size_t post_per_layer_norm;
            size_t layer_scalar;

            std::vector<size_t> moe_w1_experts;
            std::vector<size_t> moe_w3_experts;
            std::vector<size_t> moe_w2_experts;
            size_t moe_per_expert_scale = 0;
            size_t router_proj = 0;
            size_t router_scale = 0;
            size_t post_ffn_norm_1 = 0;
            size_t pre_ffn_norm_2 = 0;
            size_t post_ffn_norm_2 = 0;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    uint32_t first_shared_layer_ = 0;
    std::vector<int> kv_share_map_;
    std::vector<size_t> shared_k_nodes_;
    std::vector<size_t> shared_v_nodes_;

    std::vector<__fp16> v_norm_ones_weight_;
    size_t v_norm_ones_node_ = 0;
    size_t v_norm_ones_global_node_ = 0;
};

class Gemma4VisionModel : public Model {
    friend class Gemma4MmModel;
public:
    struct PreprocessedImage {
        std::vector<float> pixel_values;
        size_t height;
        size_t width;
        size_t patch_height;
        size_t patch_width;
        size_t num_patches;
    };

    Gemma4VisionModel();
    explicit Gemma4VisionModel(const Config& config);
    ~Gemma4VisionModel() override = default;

    PreprocessedImage preprocess_image(const std::string& image_path);
    size_t forward_vision(CactusGraph* gb, const PreprocessedImage& img, ComputeBackend backend);
    size_t build_vision_projector(CactusGraph* gb, size_t vision_features, ComputeBackend backend);

protected:
    size_t forward(const std::vector<uint32_t>&, bool) override {
        throw std::runtime_error("Gemma4VisionModel: use forward_vision() instead");
    }
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Gemma4VisionModel: build_attention unused");
    }
    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Gemma4VisionModel: build_mlp unused");
    }
    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Gemma4VisionModel: build_transformer_block unused");
    }
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    size_t build_vision_patch_embedding(CactusGraph* gb, const PreprocessedImage& img, ComputeBackend backend);
    size_t build_vision_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                   size_t cos_node, size_t sin_node,
                                   size_t attn_mask_node, ComputeBackend backend);
    size_t build_vision_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend);
    size_t build_vision_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                          size_t cos_node, size_t sin_node,
                                          size_t attn_mask_node, ComputeBackend backend);
    size_t build_vision_pooler(CactusGraph* gb, size_t hidden, const PreprocessedImage& img, ComputeBackend backend);
    std::pair<size_t, size_t> build_2d_rope_nodes(CactusGraph* gb, const PreprocessedImage& img, size_t max_patches);
    size_t build_padding_mask(CactusGraph* gb, size_t num_real, size_t max_patches);

    struct VisionWeightNodes {
        size_t patch_input_proj = 0;
        size_t position_table = 0;

        struct LayerWeights {
            size_t attn_q_weight = 0;
            size_t attn_k_weight = 0;
            size_t attn_v_weight = 0;
            size_t attn_output_weight = 0;
            size_t attn_q_norm = 0;
            size_t attn_k_norm = 0;
            size_t input_layernorm = 0;
            size_t post_attention_layernorm = 0;
            size_t pre_feedforward_layernorm = 0;
            size_t post_feedforward_layernorm = 0;
            size_t mlp_gate_proj = 0;
            size_t mlp_up_proj = 0;
            size_t mlp_down_proj = 0;
            size_t layer_scalar = 0;
        };

        std::vector<LayerWeights> layers;
        size_t embed_vision_proj = 0;
        size_t post_proj_norm = 0;
    } vision_weights_;

    std::vector<__fp16> vision_v_norm_ones_;
    size_t vision_v_norm_ones_node_ = 0;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool disable_npu_ = false;

    friend bool ::test_gemma4_vision(bool);
};

class Gemma4AudioModel : public Model {
    friend class Gemma4MmModel;
public:
    Gemma4AudioModel();
    explicit Gemma4AudioModel(const Config& config);
    ~Gemma4AudioModel() override = default;

    struct ConformerContext {
        size_t timing_fp16 = 0;
        size_t front_pad = 0;
        size_t back_pad = 0;
        size_t seq_len = 0;
    };

    size_t forward_audio(CactusGraph* gb, const std::vector<float>& mel_features,
                         size_t num_frames, ComputeBackend backend);

    size_t build_audio_projector(CactusGraph* gb, size_t audio_features, ComputeBackend backend);

    size_t build_sscp(CactusGraph* gb, const std::vector<float>& mel_features,
                      size_t num_frames, ComputeBackend backend);

    ConformerContext build_conformer_context(CactusGraph* gb, size_t sscp_output);

    size_t build_conformer_ffw(CactusGraph* gb, size_t input, uint32_t layer_idx,
                               bool is_end, ComputeBackend backend);
    size_t build_conformer_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                     const ConformerContext& ctx, ComputeBackend backend);
    size_t build_conformer_lconv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                    ComputeBackend backend);
    size_t build_conformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  const ConformerContext& ctx, ComputeBackend backend);

    struct AudioWeightNodes {
        size_t sscp_conv0_weight = 0;
        size_t sscp_conv0_norm = 0;
        size_t sscp_conv1_weight = 0;
        size_t sscp_conv1_norm = 0;
        size_t sscp_input_proj = 0;

        struct ClipBounds {
            float in_min = -1e10f, in_max = 1e10f;
            float out_min = -1e10f, out_max = 1e10f;
        };

        struct ConformerLayerWeights {
            size_t ffw_start_1 = 0, ffw_start_2 = 0;
            ClipBounds ffw_start_1_clip, ffw_start_2_clip;
            size_t ffw_start_pre_norm = 0, ffw_start_post_norm = 0;
            size_t attn_q = 0, attn_k = 0, attn_v = 0;
            ClipBounds attn_q_clip, attn_k_clip, attn_v_clip;
            size_t attn_per_dim_scale = 0;
            size_t attn_rel_pos_proj = 0;
            size_t attn_post = 0;
            ClipBounds attn_post_clip;
            size_t attn_pre_norm = 0, attn_post_norm = 0;
            size_t lconv_start = 0, lconv_depthwise = 0, lconv_end = 0;
            ClipBounds lconv_start_clip, lconv_end_clip;
            size_t lconv_pre_norm = 0, lconv_conv_norm = 0;
            size_t ffw_end_1 = 0, ffw_end_2 = 0;
            ClipBounds ffw_end_1_clip, ffw_end_2_clip;
            size_t ffw_end_pre_norm = 0, ffw_end_post_norm = 0;
            size_t block_norm = 0;
        };
        std::vector<ConformerLayerWeights> layers;

        size_t output_proj = 0;
        size_t output_proj_bias = 0;

        size_t embed_audio_proj = 0;
    } audio_weights_;

    std::vector<__fp16> audio_proj_norm_ones_;
    size_t audio_proj_norm_ones_node_ = 0;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool disable_npu_ = false;
    std::vector<__fp16> npu_audio_input_scratch_;
    std::vector<__fp16> npu_audio_output_scratch_;
    std::vector<__fp16> npu_audio_reorder_scratch_;

    friend bool ::test_gemma4_audio(bool);

protected:
    size_t forward(const std::vector<uint32_t>&, bool) override {
        throw std::runtime_error("Gemma4AudioModel: use forward_audio() instead");
    }
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Gemma4AudioModel: build_attention unused");
    }
    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Gemma4AudioModel: build_mlp unused");
    }
    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Gemma4AudioModel: build_transformer_block unused");
    }
    void load_weights_to_graph(CactusGraph* gb) override;
};

class Gemma4MmModel : public Model {
public:
    Gemma4MmModel();
    explicit Gemma4MmModel(const Config& config);
    ~Gemma4MmModel() override = default;

    bool init(const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    uint32_t decode(const std::vector<uint32_t>& tokens,
                    float temperature = -1.0f, float top_p = -1.0f, size_t top_k = 0,
                    const std::string& profile_file = "", float* out_entropy = nullptr,
                    float min_p = 0.15f, float repetition_penalty = 1.1f) override;

    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256,
                 const std::string& profile_file = "") override;

    void prefill_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                             const std::string& profile_file = "") override;

    uint32_t decode_with_images(
        const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        float temperature = -1.0f, float top_p = -1.0f, size_t top_k = 0,
        const std::string& profile_file = "", float* out_entropy = nullptr,
        float min_p = 0.15f, float repetition_penalty = 1.1f) override;

    uint32_t decode_with_audio(
        const std::vector<uint32_t>& tokens,
        const std::vector<float>& audio_features,
        float temperature = 0.0f, float top_p = 0.0f, size_t top_k = 0,
        const std::string& profile_file = "", float* out_entropy = nullptr,
        float min_p = 0.15f, float repetition_penalty = 1.1f,
        float* out_token_time_start = nullptr, float* out_token_time_end = nullptr) override;

    uint32_t decode_with_media(
        const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        const std::vector<float>& audio_features,
        float temperature = -1.0f, float top_p = -1.0f, size_t top_k = 0,
        const std::string& profile_file = "", float* out_entropy = nullptr,
        float min_p = 0.15f, float repetition_penalty = 1.1f);

    void reset_cache() override;
    std::vector<float> get_image_embeddings(const std::string& image_path) override;
    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    void compact_kv_cache() override;
    void remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges) override;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override;
    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct ForwardResult {
        size_t final_hidden_node;
        size_t seq_len;
    };

    ForwardResult forward_multimodal(CactusGraph* gb, const std::vector<uint32_t>& tokens,
                                     const std::vector<std::string>& image_paths,
                                     const std::vector<float>* audio_features,
                                     size_t audio_num_frames,
                                     ComputeBackend backend, bool use_cache);

    uint32_t decode_multimodal(const std::vector<uint32_t>& tokens,
                               const std::vector<std::string>& image_paths,
                               const std::vector<float>* audio_features,
                               size_t audio_num_frames,
                               float temperature, float top_p, size_t top_k,
                               const std::string& profile_file, float* out_entropy,
                               float min_p, float repetition_penalty);

public:
    struct MultimodalInputs {
        size_t hidden_node = 0;
        size_t pli_hidden_source_node = 0;
        std::vector<uint32_t> pli_tokens;
        size_t seq_len = 0;
    };

    const Gemma4VisionModel& vision_encoder() const { return vision_encoder_; }
    Gemma4VisionModel& vision_encoder() { return vision_encoder_; }
    const Gemma4AudioModel& audio_encoder() const { return audio_encoder_; }
    Gemma4AudioModel& audio_encoder() { return audio_encoder_; }
    const Gemma4Model& language_model() const { return language_model_; }
    Gemma4Model& language_model() { return language_model_; }
    MultimodalInputs build_multimodal_inputs(
        CactusGraph* gb, const std::vector<uint32_t>& tokens,
        const std::vector<std::string>& image_paths,
        const std::vector<float>* audio_features,
        size_t audio_num_frames,
        ComputeBackend backend);

private:
    Gemma4VisionModel vision_encoder_;
    Gemma4AudioModel audio_encoder_;
    Gemma4Model language_model_;

    bool prefill_completed_ = false;
    size_t last_token_count_ = 0;
};

}
}
