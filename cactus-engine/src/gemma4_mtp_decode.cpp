#include "gemma4_mtp_decode.h"

#include "mtp_sampler.h"
#include "cactus_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace cactus {
namespace engine {

namespace {

std::vector<float> rejection_adjusted_distribution(const std::vector<float>& target,
                                                   const std::vector<float>& assistant) {
    std::vector<float> adjusted(target.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < target.size() && i < assistant.size(); ++i) {
        float value = std::max(target[i] - assistant[i], 0.0f);
        adjusted[i] = value;
        sum += value;
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        adjusted = target;
        sum = std::accumulate(adjusted.begin(), adjusted.end(), 0.0f);
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        std::fill(adjusted.begin(), adjusted.end(), adjusted.empty() ? 0.0f : 1.0f / static_cast<float>(adjusted.size()));
        return adjusted;
    }
    for (float& value : adjusted) value /= sum;
    return adjusted;
}

std::vector<__fp16> hidden_row(const Gemma4Model::GreedyBatchWithHidden& batch, size_t row) {
    if (batch.hidden_dim == 0 || row >= batch.hidden.size() / batch.hidden_dim) {
        return {};
    }
    return std::vector<__fp16>(
        batch.hidden.begin() + row * batch.hidden_dim,
        batch.hidden.begin() + (row + 1) * batch.hidden_dim);
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<__fp16> tree_mask(size_t token_count) {
    std::vector<__fp16> mask(token_count * token_count, static_cast<__fp16>(0));
    for (size_t row = 0; row < token_count; ++row) {
        mask[row * token_count] = static_cast<__fp16>(1);
        if (row == 1 || row >= 3) mask[row * token_count + 1] = static_cast<__fp16>(1);
        if (row == 2) mask[row * token_count + 2] = static_cast<__fp16>(1);
        for (size_t col = 3; col <= row; ++col) {
            mask[row * token_count + col] = static_cast<__fp16>(1);
        }
    }
    return mask;
}

size_t tree_main_tokens_for_draft_limit(size_t draft_limit) {
    const char* env = std::getenv("CACTUS_GEMMA4_MTP_TREE_MAIN_TOKENS");
    if (env && env[0] != '\0') {
        char* end = nullptr;
        unsigned long value = std::strtoul(env, &end, 10);
        if (end != env && *end == '\0' && value >= 1 && value <= 3 && draft_limit == value + 1) {
            return static_cast<size_t>(value);
        }
        return 0;
    }
    return 0;
}

}

uint32_t gemma4_mtp_sample_distribution(const std::vector<float>& probabilities, std::mt19937& rng) {
    if (probabilities.empty()) return 0;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += std::max(0.0f, probabilities[i]);
        if (sample <= cumulative) return static_cast<uint32_t>(i);
    }
    return static_cast<uint32_t>(probabilities.size() - 1);
}

uint32_t gemma4_mtp_sample_sparse_distribution(
    const std::vector<std::pair<uint32_t, float>>& probabilities,
    std::mt19937& rng) {
    if (probabilities.empty()) return 0;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return mtp_sample_sparse(probabilities, dist(rng));
}

Gemma4AssistantDraftProvider::Gemma4AssistantDraftProvider(Gemma4Model& target,
                                                           Gemma4MtpAssistant& assistant)
    : target_(target), assistant_(assistant) {}

Gemma4MtpDraftBatch Gemma4AssistantDraftProvider::draft(
    uint32_t input_token,
    const std::vector<__fp16>& previous_hidden,
    const Gemma4Model::SharedCacheNodes& cache_nodes,
    size_t assistant_position,
    size_t draft_limit,
    const Gemma4MtpSamplingOptions& options,
    bool sparse_sampled,
    std::mt19937& rng) {
    Gemma4MtpDraftBatch batch;
    batch.tokens.reserve(draft_limit);
    if (options.temperature > 0.0f) {
        if (sparse_sampled) {
            batch.sparse_probabilities.reserve(draft_limit);
        } else {
            batch.probabilities.reserve(draft_limit);
        }
    }

    std::vector<__fp16> assistant_hidden = previous_hidden;
    uint32_t assistant_input_token = input_token;
    const size_t tree_main_tokens = tree_main_tokens_for_draft_limit(draft_limit);
    const bool use_tree = options.temperature == 0.0f
        && tree_main_tokens > 0
        && target_.can_use_mtp_tree_attention(tree_main_tokens + 2);
    const size_t main_draft_limit = use_tree ? tree_main_tokens : draft_limit;
    batch.assistant_step_ms.reserve(main_draft_limit);
    for (size_t i = 0; i < main_draft_limit; ++i) {
        auto draft_start = std::chrono::steady_clock::now();
        auto step = assistant_.draft_one(
            assistant_input_token,
            target_.token_embedding_node_for_mtp(),
            assistant_hidden,
            cache_nodes,
            assistant_position,
            i == 0 ? (1.0f / 16.0f) : 1.0f,
            options.temperature,
            options.top_p,
            options.top_k,
            options.min_p,
            i + 1 < main_draft_limit,
            use_tree && i == 0);
        auto draft_end = std::chrono::steady_clock::now();
        double step_ms = elapsed_ms(draft_start, draft_end);
        batch.assistant_draft_ms += step_ms;
        batch.assistant_step_ms.push_back(step_ms);

        uint32_t draft_token = step.token;
        if (use_tree && i == 0 && step.has_second_token) {
            batch.alt_first_token = step.second_token;
            batch.has_alt_first_token = true;
        }
        if (options.temperature > 0.0f) {
            auto sample_start = std::chrono::steady_clock::now();
            if (sparse_sampled) {
                draft_token = gemma4_mtp_sample_sparse_distribution(step.sparse_probabilities, rng);
                batch.sparse_probabilities.push_back(std::move(step.sparse_probabilities));
            } else {
                draft_token = gemma4_mtp_sample_distribution(step.probabilities, rng);
                batch.probabilities.push_back(std::move(step.probabilities));
            }
            auto sample_end = std::chrono::steady_clock::now();
            batch.sampling_or_argmax_ms += elapsed_ms(sample_start, sample_end);
        }

        batch.tokens.push_back(draft_token);
        if (i + 1 < main_draft_limit) {
            assistant_hidden = std::move(step.hidden);
        }
        assistant_input_token = draft_token;
    }
    return batch;
}

Gemma4SpeculativeVerifier::Gemma4SpeculativeVerifier(Gemma4Model& target) : target_(target) {}

Gemma4MtpVerifyResult Gemma4SpeculativeVerifier::verify_and_commit(
    uint32_t input_token,
    const Gemma4MtpDraftBatch& draft,
    size_t remaining_tokens,
    const Gemma4MtpSamplingOptions& options,
    bool sparse_sampled,
    std::mt19937& rng) {
    Gemma4MtpVerifyResult result;
    result.verify_tokens.reserve(draft.tokens.size() + 1);
    result.verify_tokens.push_back(input_token);
    const bool use_tree = options.temperature == 0.0f
        && draft.has_alt_first_token
        && !draft.tokens.empty()
        && draft.tokens.size() <= 3
        && target_.can_use_mtp_tree_attention(draft.tokens.size() + 2);
    if (use_tree) {
        result.verify_tokens.push_back(draft.tokens[0]);
        result.verify_tokens.push_back(draft.alt_first_token);
        result.verify_tokens.insert(result.verify_tokens.end(), draft.tokens.begin() + 1, draft.tokens.end());
    } else {
        result.verify_tokens.insert(result.verify_tokens.end(), draft.tokens.begin(), draft.tokens.end());
    }

    auto* graph = static_cast<CactusGraph*>(target_.graph_handle_);
    size_t verifier_cache_start = target_.cache_position();
    auto kv_start = std::chrono::steady_clock::now();
    auto cache_txn = graph->begin_kv_cache_transaction(target_.cache_state_nodes_for_mtp());
    auto kv_end = std::chrono::steady_clock::now();
    result.kv_transaction_ms += elapsed_ms(kv_start, kv_end);

    Gemma4Model::GreedyBatchWithHidden target_batch;
    auto verify_start = std::chrono::steady_clock::now();
    if (use_tree) {
        target_batch = target_.decode_greedy_tokens_with_hidden_tree(
            result.verify_tokens, tree_mask(result.verify_tokens.size()));
    } else if (options.temperature == 0.0f) {
        target_batch = target_.decode_greedy_tokens_with_hidden_early_stop(
            result.verify_tokens, draft.tokens);
    } else if (sparse_sampled) {
        target_batch = target_.decode_tokens_with_hidden_and_sparse_probs(
            result.verify_tokens, options.temperature, options.top_p, options.top_k, options.min_p);
    } else {
        target_batch = target_.decode_tokens_with_hidden_and_probs(
            result.verify_tokens, options.temperature, options.top_p, options.top_k, options.min_p);
    }
    auto verify_end = std::chrono::steady_clock::now();
    result.target_verify_ms += elapsed_ms(verify_start, verify_end);
    result.target_tokens = target_batch.tokens;

    auto sample_start = std::chrono::steady_clock::now();
    if (options.temperature == 0.0f) {
        if (use_tree) {
            if (target_batch.tokens.size() != result.verify_tokens.size()) {
                throw std::runtime_error("Gemma 4 tree verifier target row count mismatch");
            }
            auto main_row = [](size_t draft_index) {
                return draft_index == 0 ? size_t{1} : draft_index + 2;
            };
            auto prediction_row = [&](size_t draft_index) {
                return draft_index == 0 ? size_t{0} : main_row(draft_index - 1);
            };
            result.committed_verify_indices.push_back(0);
            result.next_hidden_row = 0;

            for (size_t i = 0; i < draft.tokens.size(); ++i) {
                const size_t pred_row = prediction_row(i);
                if (target_batch.tokens[pred_row] != draft.tokens[i]) {
                    if (i == 0 && target_batch.tokens[0] == draft.alt_first_token) {
                        result.output_tokens.push_back(draft.alt_first_token);
                        result.accepted = 1;
                        result.next_hidden_row = 2;
                        if (result.output_tokens.size() < remaining_tokens) {
                            result.output_tokens.push_back(target_batch.tokens[2]);
                            result.emitted_extra_target_token = true;
                        }
                        result.committed_verify_indices = {0, 2};
                    } else {
                        result.output_tokens.push_back(target_batch.tokens[pred_row]);
                        result.rejected = true;
                    }
                    break;
                }
                result.output_tokens.push_back(draft.tokens[i]);
                result.accepted = i + 1;
                result.next_hidden_row = main_row(i);
                result.committed_verify_indices.push_back(result.next_hidden_row);
            }

            if (!result.rejected
                && !result.emitted_extra_target_token
                && result.accepted == draft.tokens.size()
                && result.output_tokens.size() < remaining_tokens) {
                result.output_tokens.push_back(target_batch.tokens[result.next_hidden_row]);
                result.emitted_extra_target_token = true;
            }
        } else {
            const size_t produced = target_batch.tokens.size();
            const size_t draft_size = draft.tokens.size();
            for (; result.accepted < draft_size && result.accepted < produced; ++result.accepted) {
                if (draft.tokens[result.accepted] != target_batch.tokens[result.accepted]) {
                    result.output_tokens.push_back(target_batch.tokens[result.accepted]);
                    result.rejected = true;
                    break;
                }
                result.output_tokens.push_back(draft.tokens[result.accepted]);
            }
            if (!result.rejected && result.output_tokens.size() < remaining_tokens
                && produced > draft_size) {
                result.output_tokens.push_back(target_batch.tokens[draft_size]);
                result.emitted_extra_target_token = true;
            }
        }
    } else {
        for (; result.accepted < draft.tokens.size(); ++result.accepted) {
            uint32_t y = draft.tokens[result.accepted];
            float p_y = 0.0f;
            float q_y = 0.0f;
            if (sparse_sampled) {
                p_y = mtp_sparse_probability_at(target_batch.sparse_probabilities[result.accepted], y);
                q_y = mtp_sparse_probability_at(draft.sparse_probabilities[result.accepted], y);
            } else {
                const auto& p = target_batch.probabilities[result.accepted];
                const auto& q = draft.probabilities[result.accepted];
                p_y = y < p.size() ? p[y] : 0.0f;
                q_y = y < q.size() ? q[y] : 0.0f;
            }

            float accept_probability = q_y <= 0.0f ? 1.0f : std::min(1.0f, p_y / q_y);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(rng) <= accept_probability) {
                result.output_tokens.push_back(y);
                continue;
            }

            if (sparse_sampled) {
                auto adjusted = mtp_rejection_adjusted_sparse_distribution(
                    target_batch.sparse_probabilities[result.accepted],
                    draft.sparse_probabilities[result.accepted]);
                result.output_tokens.push_back(gemma4_mtp_sample_sparse_distribution(adjusted, rng));
            } else {
                auto adjusted = rejection_adjusted_distribution(
                    target_batch.probabilities[result.accepted],
                    draft.probabilities[result.accepted]);
                result.output_tokens.push_back(gemma4_mtp_sample_distribution(adjusted, rng));
            }
            result.rejected = true;
            break;
        }
        if (!result.rejected && result.output_tokens.size() < remaining_tokens) {
            if (sparse_sampled) {
                result.output_tokens.push_back(gemma4_mtp_sample_sparse_distribution(
                    target_batch.sparse_probabilities[draft.tokens.size()], rng));
            } else {
                result.output_tokens.push_back(gemma4_mtp_sample_distribution(
                    target_batch.probabilities[draft.tokens.size()], rng));
            }
            result.emitted_extra_target_token = true;
        }
    }
    auto sample_end = std::chrono::steady_clock::now();
    result.sampling_or_argmax_ms += elapsed_ms(sample_start, sample_end);

    if (use_tree) {
        for (uint32_t token : result.output_tokens) {
            target_.record_sampled_token(token);
        }
    }

    if (!result.output_tokens.empty()) {
        size_t row = use_tree
            ? result.next_hidden_row
            : (options.temperature == 0.0f ? result.accepted : result.output_tokens.size() - 1);
        result.next_hidden = hidden_row(target_batch, row);
    }

    size_t committed_cache_tokens = result.output_tokens.size();
    result.target_cache_crop_length = verifier_cache_start + committed_cache_tokens;
    kv_start = std::chrono::steady_clock::now();
    if (use_tree) {
        cache_txn.commit_selected(result.committed_verify_indices);
        graph->apply_pending_kv_cache_sequence_lengths();
        target_.set_cache_position_for_mtp(result.target_cache_crop_length);
    } else if (committed_cache_tokens < result.verify_tokens.size()) {
        cache_txn.commit_prefix(committed_cache_tokens);
        graph->apply_pending_kv_cache_sequence_lengths();
        target_.set_cache_position_for_mtp(result.target_cache_crop_length);
    } else {
        cache_txn.commit_all();
        graph->apply_pending_kv_cache_sequence_lengths();
    }
    kv_end = std::chrono::steady_clock::now();
    result.kv_transaction_ms += elapsed_ms(kv_start, kv_end);

    return result;
}

Gemma4MtpDecodeStrategy::Gemma4MtpDecodeStrategy(Gemma4Model& target,
                                                 Gemma4MtpAssistant& assistant)
    : draft_provider_(target, assistant), verifier_(target) {}

Gemma4MtpRoundResult Gemma4MtpDecodeStrategy::decode_round(
    uint32_t input_token,
    const std::vector<__fp16>& previous_hidden,
    const Gemma4Model::SharedCacheNodes& cache_nodes,
    size_t assistant_position,
    size_t draft_limit,
    size_t remaining_tokens,
    const Gemma4MtpSamplingOptions& options,
    bool sparse_sampled,
    std::mt19937& rng) {
    Gemma4MtpRoundResult round;
    round.draft = draft_provider_.draft(
        input_token,
        previous_hidden,
        cache_nodes,
        assistant_position,
        draft_limit,
        options,
        sparse_sampled,
        rng);
    round.verification = verifier_.verify_and_commit(
        input_token,
        round.draft,
        remaining_tokens,
        options,
        sparse_sampled,
        rng);
    return round;
}

}
}
