#pragma once

#include "../engine/engine.h"
#include "../npu/npu.h"

namespace cactus {
namespace engine {
class TrOCRModel : public Model {
public:
    TrOCRModel();
    explicit TrOCRModel(const Config& config);
    ~TrOCRModel() override = default;
    uint32_t decode_with_image(
        const std::vector<uint32_t>& tokens,
        const std::vector<float>& image_pixels,
        size_t image_height,
        size_t image_width,
        float temperature = 0.0f,
        float top_p = 0.0f,
        size_t top_k = 0,
        const std::string& profile_file = "",
        float* out_entropy = nullptr);
    std::vector<float> get_image_embeddings(const std::string& image_path) override;

    void reset_cache() override;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("TrOCR: use specialized encoder/decoder attention methods");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("TrOCR: use specialized encoder/decoder MLP methods");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("TrOCR: use specialized encoder/decoder transformer blocks");
    }

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override {
        throw std::runtime_error("TrOCR requires image+token forward(). Use decode_with_image().");
    }

    void load_weights_to_graph(CactusGraph* gb) override;
    size_t build_patch_embedding(CactusGraph* gb, size_t image_input, size_t height, size_t width);
    size_t build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend);
    size_t build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend);
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend);
    void run_encoder(const std::vector<float>& image_pixels, size_t height, size_t width);
    size_t build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                        ComputeBackend backend, bool use_cache, size_t position_offset);
    size_t build_decoder_cross_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache);
    size_t build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const;
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                           ComputeBackend backend, bool use_cache, size_t position_offset);
    size_t run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    void reset_graph_side_cache_nodes();

private:
    struct WeightNodeIDs {
        size_t encoder_patch_embedding_weight;
        size_t encoder_patch_embedding_bias;
        size_t encoder_position_embedding;
        size_t encoder_cls_token;
        size_t encoder_layernorm_weight;
        size_t encoder_layernorm_bias;
        size_t encoder_output;

        size_t decoder_embed_tokens;
        size_t decoder_position_embedding;
        size_t decoder_embed_layernorm_weight;
        size_t decoder_embed_layernorm_bias;
        size_t output_weight;

        struct LayerWeights {
            size_t encoder_self_attn_q_weight;
            size_t encoder_self_attn_k_weight;
            size_t encoder_self_attn_v_weight;
            size_t encoder_self_attn_output_weight;
            size_t encoder_self_attn_q_bias;
            size_t encoder_self_attn_k_bias;
            size_t encoder_self_attn_v_bias;
            size_t encoder_self_attn_output_bias;
            size_t encoder_layernorm1_weight;
            size_t encoder_layernorm1_bias;
            size_t encoder_layernorm2_weight;
            size_t encoder_layernorm2_bias;
            size_t encoder_mlp_fc1_weight;
            size_t encoder_mlp_fc1_bias;
            size_t encoder_mlp_fc2_weight;
            size_t encoder_mlp_fc2_bias;

            size_t decoder_self_attn_q_weight;
            size_t decoder_self_attn_k_weight;
            size_t decoder_self_attn_v_weight;
            size_t decoder_self_attn_output_weight;
            size_t decoder_self_attn_q_bias;
            size_t decoder_self_attn_k_bias;
            size_t decoder_self_attn_v_bias;
            size_t decoder_self_attn_output_bias;
            size_t decoder_self_attn_layernorm_weight;
            size_t decoder_self_attn_layernorm_bias;

            size_t decoder_cross_attn_q_weight;
            size_t decoder_cross_attn_k_weight;
            size_t decoder_cross_attn_v_weight;
            size_t decoder_cross_attn_output_weight;
            size_t decoder_cross_attn_q_bias;
            size_t decoder_cross_attn_k_bias;
            size_t decoder_cross_attn_v_bias;
            size_t decoder_cross_attn_output_bias;
            size_t decoder_cross_attn_layernorm_weight;
            size_t decoder_cross_attn_layernorm_bias;

            size_t decoder_mlp_fc1_weight;
            size_t decoder_mlp_fc1_bias;
            size_t decoder_mlp_fc2_weight;
            size_t decoder_mlp_fc2_bias;
            size_t decoder_final_layernorm_weight;
            size_t decoder_final_layernorm_bias;
        };

        std::vector<LayerWeights> encoder_layers;
        std::vector<LayerWeights> decoder_layers;
    } weight_nodes_;

    bool encoder_ready_ = false;
    size_t encoder_output_persistent_ = 0;
    size_t last_encoder_post_norm_node_ = 0;
    std::vector<size_t> encoder_k_persistent_;
    std::vector<size_t> encoder_v_persistent_;

    size_t last_new_tokens_ = 0;
    bool first_decode_step_ = true;

    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
};

} // namespace engine
} // namespace cactus
