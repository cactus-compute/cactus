#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

struct WrongnessProbeScore {
    float logit = 0.0f;
    float probability_wrong = 0.0f;
    float confidence = 1.0f;
};

class WrongnessProbe {
public:
    static constexpr size_t FEAT_DIM = 1536;
    static constexpr size_t PROJ_DIM = 32;
    static constexpr size_t HIDDEN0 = 128;
    static constexpr size_t HIDDEN1 = 64;
    static constexpr size_t MAX_TOKENS = 1024;

    bool load(const std::string& path);
    bool is_loaded() const { return loaded_; }
    WrongnessProbeScore score(const std::vector<float>& hidden_states, size_t token_count) const;

private:
    bool loaded_ = false;
    std::vector<float> norm_weight_;
    std::vector<float> norm_bias_;
    std::vector<float> proj_weight_;
    std::vector<float> proj_bias_;
    std::vector<float> attn_query_;
    std::vector<float> head0_weight_;
    std::vector<float> head0_bias_;
    std::vector<float> head1_weight_;
    std::vector<float> head1_bias_;
    std::vector<float> head2_weight_;
    std::vector<float> head2_bias_;
};

} // namespace engine
} // namespace cactus
