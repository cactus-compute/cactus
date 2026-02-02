#pragma once

#include "../engine/engine.h"
#include "../npu/npu.h"

namespace cactus {
namespace engine {

/**
 * TrOCRModel - Transformer-based Optical Character Recognition
 *
 * Architecture:
 * - Vision Encoder: ViT (Vision Transformer) for processing image patches
 * - Text Decoder: Autoregressive transformer decoder for generating text
 *
 * This is particularly useful for VIN (Vehicle Identification Number) recognition
 * and other OCR tasks on mobile devices using Cactus Graph.
 *
 * Based on Microsoft TrOCR: https://arxiv.org/abs/2109.10282
 */
class TrOCRModel : public Model {
public:
    TrOCRModel();
    explicit TrOCRModel(const Config& config);
    ~TrOCRModel() override = default;

    /**
     * Decode with image input - main entry point for OCR inference
     * Similar to Whisper's decode_with_audio but for images
     */
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

    /**
     * Get image embeddings from the vision encoder (for embedding extraction)
     */
    std::vector<float> get_image_embeddings(const std::string& image_path) override;

    void reset_cache() override;

protected:
    // Base class overrides - these throw since TrOCR uses specialized methods
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

    // ============ Vision Encoder (ViT) Methods ============

    /**
     * Build patch embedding layer
     * Splits image into patches and projects them to hidden dimension
     */
    size_t build_patch_embedding(CactusGraph* gb, size_t image_input, size_t height, size_t width);

    /**
     * Build encoder self-attention layer
     */
    size_t build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend);

    /**
     * Build encoder MLP (feed-forward network)
     */
    size_t build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend);

    /**
     * Build complete encoder transformer block
     */
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend);

    /**
     * Run the full vision encoder on image input
     */
    void run_encoder(const std::vector<float>& image_pixels, size_t height, size_t width);

    // ============ Text Decoder Methods ============

    /**
     * Build decoder self-attention (causal attention on generated tokens)
     */
    size_t build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                        ComputeBackend backend, bool use_cache, size_t position_offset);

    /**
     * Build decoder cross-attention (attend to encoder output)
     */
    size_t build_decoder_cross_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache);

    /**
     * Build decoder MLP
     */
    size_t build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const;

    /**
     * Build complete decoder transformer block
     */
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                           ComputeBackend backend, bool use_cache, size_t position_offset);

    /**
     * Run one decoder step (for autoregressive generation)
     */
    size_t run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    void reset_graph_side_cache_nodes();

private:
    struct WeightNodeIDs {
        // Vision Encoder global weights
        size_t encoder_patch_embedding_weight;
        size_t encoder_patch_embedding_bias;
        size_t encoder_position_embedding;
        size_t encoder_cls_token;
        size_t encoder_layernorm_weight;
        size_t encoder_layernorm_bias;
        size_t encoder_output;  // Stores encoder output for cross-attention

        // Decoder global weights
        size_t decoder_embed_tokens;
        size_t decoder_position_embedding;
        size_t decoder_embed_layernorm_weight;
        size_t decoder_embed_layernorm_bias;
        size_t output_weight;

        struct LayerWeights {
            // Encoder self-attention
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

            // Decoder self-attention
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

            // Decoder cross-attention (encoder-decoder attention)
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

            // Decoder MLP
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

    // Encoder state
    bool encoder_ready_ = false;
    std::vector<uint8_t> encoder_output_host_;
    std::vector<size_t> encoder_output_shape_;
    Precision encoder_output_precision_ = Precision::FP32;

    // Cross-attention KV cache (encoder keys/values are fixed after encoding)
    bool encoder_kv_ready_ = false;
    std::vector<size_t> encoder_k_nodes_;
    std::vector<size_t> encoder_v_nodes_;
    std::vector<std::vector<uint8_t>> encoder_k_host_;
    std::vector<std::vector<uint8_t>> encoder_v_host_;
    std::vector<std::vector<size_t>> encoder_k_shape_;
    std::vector<std::vector<size_t>> encoder_v_shape_;
    Precision encoder_kv_precision_ = Precision::FP32;

    // Decoder state
    size_t last_new_tokens_ = 0;
    bool first_decode_step_ = true;

    // NPU encoder support (optional)
    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
};

} // namespace engine
} // namespace cactus
