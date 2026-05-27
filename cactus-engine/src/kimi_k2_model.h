#pragma once

#include "engine.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class CactusGraph;

namespace cactus {
namespace engine {

class KimiK2Model : public Model {
public:
    KimiK2Model();
    explicit KimiK2Model(const Config& config);
    ~KimiK2Model() override = default;

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

    const std::map<std::string, std::string>& weight_manifest() const { return weight_manifest_; }

private:
    bool setup_tokenizer(const std::string& model_dir);
    bool load_config(const std::string& model_dir, Config& out_config) const;
    bool load_weight_manifest(const std::string& model_dir);
    void load_weights_to_graph();
    void initialize_cache_states();
    void validate_architecture() const;
    void validate_required_weights() const;
    std::string weight_path(const std::string& logical_name) const;
    size_t build_forward(CactusGraph& gb, const std::vector<uint32_t>& tokens, bool use_cache, size_t position_offset);
    size_t build_attention(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx, size_t seq_len,
                           bool use_cache, size_t position_offset);
    size_t build_dense_mlp(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx);
    size_t build_moe(CactusGraph& gb, size_t normalized_input, uint32_t layer_idx);
    float attention_softmax_scale() const;
    size_t mmap_weight(CactusGraph& gb, const std::string& logical_name);

    Config config_copy_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<CactusGraph> graph_;
    std::string model_dir_;
    size_t context_size_ = 0;
    size_t kimi_cache_seq_len_ = 0;
    std::map<std::string, std::string> weight_manifest_;

    size_t embedding_node_ = 0;
    size_t output_norm_node_ = 0;
    size_t output_weight_node_ = 0;
    struct LayerCacheNodes {
        size_t key = 0;
        size_t value = 0;
    };
    std::vector<LayerCacheNodes> cache_nodes_;
    struct DenseWeights {
        size_t gate = 0;
        size_t up = 0;
        size_t down = 0;
    };
    struct MoeWeights {
        size_t router = 0;
        size_t router_bias = 0;
        std::vector<DenseWeights> experts;
        DenseWeights shared;
    };
    struct LayerWeights {
        size_t input_norm = 0;
        size_t q_a = 0;
        size_t q_a_norm = 0;
        size_t q_b = 0;
        size_t kv_a = 0;
        size_t kv_a_norm = 0;
        size_t kv_b = 0;
        size_t o = 0;
        size_t post_attn_norm = 0;
        DenseWeights dense;
        MoeWeights moe;
    };
    std::vector<LayerWeights> layers_;
};

} // namespace engine
} // namespace cactus
