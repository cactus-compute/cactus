#pragma once

#include "model_gemma4.h"
#include "cactus_graph.h"

#include <string>
#include <utility>
#include <vector>

namespace cactus {
namespace engine {

class Gemma4MtpAssistant {
public:
    struct StepResult {
        uint32_t token = 0;
        uint32_t second_token = 0;
        bool has_second_token = false;
        std::vector<__fp16> hidden;
        std::vector<float> probabilities;
        std::vector<std::pair<uint32_t, float>> sparse_probabilities;
    };

    bool init(CactusGraph* graph, const std::string& assistant_path);
    bool initialized() const { return initialized_; }

    StepResult draft_one(uint32_t target_token,
                         size_t target_embedding_node,
                         const std::vector<__fp16>& target_hidden,
                         const Gemma4Model::SharedCacheNodes& cache_nodes,
                         size_t position,
                         float hidden_scale = 1.0f / 16.0f,
                         float temperature = 0.0f,
                         float top_p = 0.0f,
                         size_t top_k = 0,
                         float min_p = 0.0f,
                         bool return_hidden = true,
                         bool return_second_token = false);

private:
    struct LayerWeights {
        size_t input_norm = 0;
        size_t attn_q = 0;
        size_t attn_q_norm = 0;
        size_t attn_output = 0;
        size_t post_attn_norm = 0;
        size_t pre_ffn_norm = 0;
        size_t ffn_gate = 0;
        size_t ffn_up = 0;
        size_t ffn_down = 0;
        size_t post_ffn_norm = 0;
        size_t layer_scalar = 0;
    };

    size_t apply_rope(CactusGraph* gb, size_t tensor, size_t head_dim, size_t rot_dim,
                      float rope_freq, size_t position);
    size_t build_layer(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                       const Gemma4Model::SharedCacheNodes& cache_nodes, size_t position);
    std::vector<uint32_t> candidate_tokens_from_centroids(CactusGraph* gb, size_t centroid_logits_node) const;
    uint32_t argmax_masked_logits(CactusGraph* gb, size_t logits_node, size_t centroid_logits_node) const;
    uint32_t argmax_masked_logits_from_hidden(CactusGraph* gb, size_t hidden_node, size_t centroid_logits_node) const;
    std::pair<uint32_t, uint32_t> top2_masked_logits_from_hidden(CactusGraph* gb, size_t hidden_node, size_t centroid_logits_node) const;

    CactusGraph* graph_ = nullptr;
    std::string path_;
    bool initialized_ = false;

    size_t token_embeddings_ = 0;
    size_t masked_centroids_ = 0;
    size_t pre_projection_ = 0;
    size_t post_projection_ = 0;
    size_t output_norm_ = 0;
    std::vector<LayerWeights> layers_;
    std::vector<uint32_t> token_ordering_;
};

}
}
