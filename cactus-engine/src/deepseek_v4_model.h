#pragma once

#include "engine.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class CactusGraph;

namespace cactus {
namespace engine {

class DeepSeekV4Model : public Model {
public:
    DeepSeekV4Model();
    explicit DeepSeekV4Model(const Config& config);
    ~DeepSeekV4Model() override = default;

    const Config& get_config() const override { return config_copy_; }
    Tokenizer* get_tokenizer() const override { return tokenizer_.get(); }

    bool init(const std::string& model_dir, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;
    uint32_t decode(const std::vector<uint32_t>& tokens, float temperature = -1.0f, float top_p = -1.0f,
                    size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                    float min_p = 0.15f, float repetition_penalty = 1.1f) override;
    bool prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token) override;
    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 128, const std::string& profile_file = "",
                 bool prepare_decode = true) override;
    void reset_cache() override;
    void prefetch_moe_expert_pages();

private:
    struct DeepSeekConfig {
        size_t vocab_size = 0;
        size_t num_layers = 0;
        size_t hidden_dim = 0;
        size_t hc_mult = 4;
        size_t attention_heads = 0;
        size_t head_dim = 0;
        size_t rope_dim = 0;
        size_t q_lora_rank = 0;
        size_t o_groups = 1;
        size_t o_lora_rank = 0;
        size_t num_experts = 0;
        size_t num_experts_per_tok = 0;
        size_t moe_intermediate_dim = 0;
        size_t num_shared_experts = 1;
        size_t index_heads = 0;
        size_t index_head_dim = 0;
        size_t index_topk = 0;
        size_t sliding_window = 0;
        float eps = 1e-6f;
        float route_scale = 1.0f;
        float swiglu_limit = 10.0f;
        float rope_theta = 10000.0f;
        float compress_rope_theta = 160000.0f;
        float yarn_factor = 16.0f;
        size_t yarn_original_max = 65536;
        float yarn_beta_fast = 32.0f;
        float yarn_beta_slow = 1.0f;
        std::vector<size_t> attention_compress_rates;
        std::vector<bool> hash_moe_layers;
    };

    struct HcWeights {
        size_t fn = 0;
        size_t base = 0;
        size_t scale = 0;
    };
    struct ExpertWeights {
        size_t gate = 0;
        size_t up = 0;
        size_t down = 0;
    };
    struct LayerWeights {
        HcWeights attn_hc;
        HcWeights ffn_hc;
        size_t input_norm = 0;
        size_t post_norm = 0;
        size_t q_a = 0;
        size_t q_a_norm = 0;
        size_t q_b = 0;
        size_t kv = 0;
        size_t kv_norm = 0;
        size_t o_a = 0;
        size_t o_b = 0;
        size_t attn_sink = 0;
        size_t hca_kv = 0;
        size_t hca_gate = 0;
        size_t hca_norm = 0;
        size_t hca_pos = 0;
        size_t csa_kv = 0;
        size_t csa_gate = 0;
        size_t csa_norm = 0;
        size_t csa_pos = 0;
        size_t idx_kv = 0;
        size_t idx_gate = 0;
        size_t idx_norm = 0;
        size_t idx_q_b = 0;
        size_t idx_weights = 0;
        size_t idx_pos = 0;
        size_t router = 0;
        size_t router_bias = 0;
        size_t tid2eid = 0;
        size_t experts_gate_up = 0;
        size_t experts_down = 0;
        std::vector<ExpertWeights> experts;
        ExpertWeights shared;
    };

    bool setup_tokenizer(const std::string& model_dir);
    bool load_config(const std::string& model_dir);
    bool load_weight_manifest(const std::string& model_dir);
    void load_weights_to_graph();
    void validate_architecture() const;
    void validate_required_weights() const;
    std::string weight_path(const std::string& logical_name) const;
    bool has_weight(const std::string& logical_name) const;
    size_t mmap_weight(CactusGraph& gb, const std::string& logical_name);
    size_t mmap_weight_any(CactusGraph& gb, const std::vector<std::string>& logical_names);

    size_t build_forward(CactusGraph& gb, const std::vector<uint32_t>& tokens);
    size_t build_layer(CactusGraph& gb, size_t streams, size_t token_input, size_t position_input,
                       uint32_t layer_idx, size_t seq_len);
    size_t build_attention(CactusGraph& gb, size_t normalized_input, size_t q_residual,
                           size_t position_input, uint32_t layer_idx, size_t seq_len);
    size_t build_moe(CactusGraph& gb, size_t normalized_input, size_t token_input, uint32_t layer_idx);
    size_t grouped_o_a(CactusGraph& gb, size_t attn_flat, size_t o_a_proj) const;
    size_t build_static_indices(CactusGraph& gb, size_t seq_len, size_t width, size_t compression_ratio) const;
    float attention_softmax_scale() const;

    Config config_copy_;
    DeepSeekConfig ds_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<CactusGraph> graph_;
    std::string model_dir_;
    size_t context_size_ = 0;
    std::vector<uint32_t> prefix_tokens_;
    std::map<std::string, std::string> weight_manifest_;

    size_t embedding_node_ = 0;
    HcWeights head_hc_;
    size_t output_norm_node_ = 0;
    size_t output_weight_node_ = 0;
    std::vector<LayerWeights> layers_;
};

} // namespace engine
} // namespace cactus
