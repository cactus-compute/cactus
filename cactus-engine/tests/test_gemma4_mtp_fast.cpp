#include "test_utils.h"
#include "models/gemma4/gemma4_mtp_assistant.h"
#include "models/gemma4/model_gemma4.h"
#include "src/mtp_sampler.h"
#include "src/utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TestUtils;
using namespace EngineTestUtils;
using cactus::engine::Gemma4Model;
using cactus::engine::Gemma4MtpAssistant;

static const char* target_path() {
    return std::getenv("CACTUS_TEST_GEMMA4_TARGET");
}

static const char* assistant_path() {
    return std::getenv("CACTUS_TEST_GEMMA4_ASSISTANT");
}

static bool has_real_model_env() {
    return target_path() && std::strlen(target_path()) > 0
        && assistant_path() && std::strlen(assistant_path()) > 0;
}

static int max_tokens() {
    const char* value = std::getenv("CACTUS_TEST_GEMMA4_MAX_TOKENS");
    return value && std::strlen(value) > 0 ? std::atoi(value) : 24;
}

static bool run_complete_collect_ids(cactus_model_t model,
                                     const char* messages,
                                     const char* options,
                                     std::vector<uint32_t>& token_ids,
                                     std::string& json_response) {
    StreamingData data;
    data.model = model;
    char response[8192] = {0};

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 options, nullptr, stream_callback, &data, nullptr, 0);
    json_response = response;
    token_ids = data.token_ids;

    return result > 0 && !token_ids.empty();
}

static uint32_t sample_sparse_for_test(const std::vector<std::pair<uint32_t, float>>& probabilities,
                                       std::mt19937& rng) {
    if (probabilities.empty()) return 0;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return mtp_sample_sparse(probabilities, dist(rng));
}

static std::vector<uint32_t> prompt_tokens_for_messages(cactus_model_t model, const char* messages) {
    auto* handle = static_cast<CactusModelHandle*>(model);
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    auto parsed = cactus::ffi::parse_messages_json(messages, image_paths, &audio_paths);
    auto* tokenizer = handle->model->get_tokenizer();
    std::string prompt = tokenizer->format_chat_prompt(parsed, true, "", false);
    if (prompt.find("ERROR:") == 0) {
        throw std::runtime_error(prompt.substr(6));
    }
    return tokenizer->encode(prompt);
}

static std::vector<__fp16> hidden_row_for_test(const Gemma4Model::GreedyBatchWithHidden& batch,
                                               size_t row) {
    if (batch.hidden_dim == 0 || row >= batch.hidden.size() / batch.hidden_dim) {
        throw std::runtime_error("Gemma 4 MTP test hidden row is unavailable");
    }
    return std::vector<__fp16>(
        batch.hidden.begin() + row * batch.hidden_dim,
        batch.hidden.begin() + (row + 1) * batch.hidden_dim);
}

static std::vector<uint32_t> run_manual_sampled_mtp_reference(cactus_model_t model,
                                                              const char* messages,
                                                              const std::string& assistant_dir,
                                                              size_t max_tokens,
                                                              size_t max_draft,
                                                              uint32_t seed) {
    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* target = dynamic_cast<Gemma4Model*>(handle->model.get());
    if (!target) {
        if (auto* mm = dynamic_cast<cactus::engine::Gemma4MmModel*>(handle->model.get())) {
            target = &mm->language_model();
        }
    }
    if (!target) {
        throw std::runtime_error("Gemma 4 MTP sampled reference requires a Gemma4Model target");
    }

    auto* graph = static_cast<CactusGraph*>(target->graph_handle_);
    Gemma4MtpAssistant assistant;
    if (!assistant.init(graph, assistant_dir)) {
        throw std::runtime_error("Gemma 4 MTP sampled reference assistant load failed");
    }

    const float temperature = 0.7f;
    const float top_p = 0.95f;
    const size_t top_k = 64;
    const float min_p = 0.0f;

    target->reset_cache();
    std::vector<uint32_t> prompt_tokens = prompt_tokens_for_messages(model, messages);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("Gemma 4 MTP sampled reference prompt is empty");
    }
    if (prompt_tokens.size() > 1) {
        std::vector<uint32_t> prefix(prompt_tokens.begin(), prompt_tokens.end() - 1);
        target->prefill_for_mtp(prefix, target->get_prefill_chunk_size());
    }

    std::mt19937 rng(seed);
    auto first = target->decode_tokens_with_hidden_and_sparse_probs(
        {prompt_tokens.back()}, temperature, top_p, top_k, min_p);
    uint32_t next_token = sample_sparse_for_test(first.sparse_probabilities.back(), rng);
    target->record_sampled_token(next_token);

    std::vector<uint32_t> generated = {next_token};
    std::vector<__fp16> mtp_prev_hidden = hidden_row_for_test(
        first, first.hidden.size() / first.hidden_dim - 1);
    size_t produced = 1;

    while (produced < max_tokens) {
        size_t remaining = max_tokens - produced;
        size_t draft_limit = std::min<size_t>(max_draft, remaining > 1 ? remaining - 1 : 1);
        auto cache_nodes = target->shared_cache_nodes_for_mtp();
        size_t assistant_position = prompt_tokens.size() + generated.size() - 1;

        std::vector<uint32_t> draft_tokens;
        std::vector<std::vector<std::pair<uint32_t, float>>> draft_probabilities;
        draft_tokens.reserve(draft_limit);
        draft_probabilities.reserve(draft_limit);

        std::vector<__fp16> assistant_hidden = mtp_prev_hidden;
        uint32_t assistant_input_token = next_token;
        for (size_t i = 0; i < draft_limit; ++i) {
            auto draft = assistant.draft_one(
                assistant_input_token,
                target->token_embedding_node_for_mtp(),
                assistant_hidden,
                cache_nodes,
                assistant_position,
                i == 0 ? (1.0f / 16.0f) : 1.0f,
                temperature,
                top_p,
                top_k,
                min_p);
            uint32_t draft_token = sample_sparse_for_test(draft.sparse_probabilities, rng);
            draft_tokens.push_back(draft_token);
            draft_probabilities.push_back(std::move(draft.sparse_probabilities));
            assistant_hidden = std::move(draft.hidden);
            assistant_input_token = draft_token;
        }

        std::vector<uint32_t> verify_tokens;
        verify_tokens.reserve(draft_tokens.size() + 1);
        verify_tokens.push_back(next_token);
        verify_tokens.insert(verify_tokens.end(), draft_tokens.begin(), draft_tokens.end());

        size_t verifier_cache_start = target->cache_position();
        auto cache_txn = graph->begin_kv_cache_transaction(target->cache_state_nodes_for_mtp());
        auto verifier = target->decode_tokens_with_hidden_and_sparse_probs(
            verify_tokens, temperature, top_p, top_k, min_p);

        bool rejected = false;
        size_t accepted = 0;
        std::vector<uint32_t> output_batch;
        for (; accepted < draft_tokens.size(); ++accepted) {
            uint32_t y = draft_tokens[accepted];
            float p_y = mtp_sparse_probability_at(verifier.sparse_probabilities[accepted], y);
            float q_y = mtp_sparse_probability_at(draft_probabilities[accepted], y);
            float accept_probability = q_y <= 0.0f ? 1.0f : std::min(1.0f, p_y / q_y);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(rng) <= accept_probability) {
                output_batch.push_back(y);
                continue;
            }
            auto adjusted = mtp_rejection_adjusted_sparse_distribution(
                verifier.sparse_probabilities[accepted],
                draft_probabilities[accepted]);
            output_batch.push_back(sample_sparse_for_test(adjusted, rng));
            rejected = true;
            break;
        }
        if (!rejected && output_batch.size() < remaining) {
            output_batch.push_back(sample_sparse_for_test(
                verifier.sparse_probabilities[draft_tokens.size()], rng));
        }

        if (!output_batch.empty()) {
            mtp_prev_hidden = hidden_row_for_test(verifier, output_batch.size() - 1);
        }

        size_t committed_cache_tokens = output_batch.size();
        if (committed_cache_tokens < verify_tokens.size()) {
            cache_txn.commit_prefix(committed_cache_tokens);
            graph->apply_pending_kv_cache_sequence_lengths();
            target->set_cache_position_for_mtp(verifier_cache_start + committed_cache_tokens);
        } else {
            cache_txn.commit_all();
            graph->apply_pending_kv_cache_sequence_lengths();
        }

        for (uint32_t token : output_batch) {
            next_token = token;
            produced++;
            target->record_sampled_token(token);
            generated.push_back(token);
            if (produced >= max_tokens) break;
        }
    }

    return generated;
}

bool test_gemma4_mtp_greedy_matches_standard_decode() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    std::string normal_options = "{"
        "\"temperature\":0,"
        "\"top_p\":0,"
        "\"max_tokens\":" + std::to_string(max_tokens()) + ","
        "\"telemetry_enabled\":false"
    "}";

    const std::vector<std::string> prompts = {
        "who are you",
        "Write one short sentence about desert rain.",
        "Count from 1 to 100, separated by commas.",
        "Write a short Python function that adds two numbers.",
        "Return a JSON object with keys name, role, and status."
    };

    bool ok = true;
    for (const auto& prompt : prompts) {
        std::string messages = "[{\"role\":\"user\",\"content\":\"" + prompt + "\"}]";
        std::vector<uint32_t> normal_ids;
        std::string normal_json;
        bool normal_ok = run_complete_collect_ids(model, messages.c_str(), normal_options.c_str(), normal_ids, normal_json);
        ok = ok && normal_ok;
        for (int draft : {1, 2, 3, 4}) {
            cactus_reset(model);
            std::string mtp_options = "{"
                "\"temperature\":0,"
                "\"top_p\":0,"
                "\"max_tokens\":" + std::to_string(max_tokens()) + ","
                "\"telemetry_enabled\":false,"
                "\"mtp\":true,"
                "\"mtp_required\":true,"
                "\"mtp_max_draft_tokens\":" + std::to_string(draft) + ","
                "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
            "}";
            std::vector<uint32_t> mtp_ids;
            std::string mtp_json;
            bool mtp_ok = run_complete_collect_ids(model, messages.c_str(), mtp_options.c_str(), mtp_ids, mtp_json);
            ok = ok
                && mtp_ok
                && normal_ids == mtp_ids
                && json_number(mtp_json, "drafted_tokens") > 0.0
                && mtp_json.find("\"enabled\":true") != std::string::npos
                && mtp_json.find("low_acceptance") == std::string::npos
                && mtp_json.find("mtp") != std::string::npos;
        }
        cactus_reset(model);
    }
    cactus_destroy(model);

    return ok;
}

bool test_gemma4_mtp_short_chat_accepts_drafts() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "user", "content": "who are you"}
    ])";

    std::string normal_options = "{"
        "\"temperature\":0,"
        "\"top_k\":1,"
        "\"max_tokens\":48,"
        "\"telemetry_enabled\":false"
    "}";

    std::string mtp_options = "{"
        "\"temperature\":0,"
        "\"top_k\":1,"
        "\"max_tokens\":48,"
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_max_draft_tokens\":2,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> normal_ids;
    std::vector<uint32_t> mtp_ids;
    std::string normal_json;
    std::string mtp_json;

    bool normal_ok = run_complete_collect_ids(model, messages, normal_options.c_str(), normal_ids, normal_json);
    cactus_reset(model);
    bool mtp_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), mtp_ids, mtp_json);

    cactus_destroy(model);

    double drafted = json_number(mtp_json, "drafted_tokens");
    double accepted = json_number(mtp_json, "accepted_tokens");
    double rejected = json_number(mtp_json, "rejected_tokens");

    return normal_ok
        && mtp_ok
        && normal_ids == mtp_ids
        && drafted >= 12.0
        && accepted >= 0.6 * drafted
        && rejected <= std::max(3.0, 0.25 * drafted)
        && mtp_json.find("\"enabled\":true") != std::string::npos
        && mtp_json.find("low_acceptance") == std::string::npos;
}

bool test_gemma4_mtp_cache_invariant() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "user", "content": "Write one short sentence about desert rain."}
    ])";

    std::string normal_options = "{"
        "\"temperature\":0,"
        "\"top_p\":0,"
        "\"max_tokens\":" + std::to_string(max_tokens()) + ","
        "\"telemetry_enabled\":false"
    "}";

    std::string mtp_options = "{"
        "\"temperature\":0,"
        "\"top_p\":0,"
        "\"max_tokens\":" + std::to_string(max_tokens()) + ","
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_cache_invariant_check\":true,"
        "\"mtp_max_draft_tokens\":2,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> normal_ids;
    std::vector<uint32_t> mtp_ids;
    std::string normal_json;
    std::string mtp_json;

    bool normal_ok = run_complete_collect_ids(model, messages, normal_options.c_str(), normal_ids, normal_json);
    cactus_reset(model);
    bool mtp_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), mtp_ids, mtp_json);

    cactus_destroy(model);

    return normal_ok
        && mtp_ok
        && normal_ids == mtp_ids
        && json_number(mtp_json, "rounds") > 0.0
        && mtp_json.find("mtp") != std::string::npos;
}

bool test_gemma4_mtp_multiturn_prefix_cache_invariant() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* first_messages = R"([
        {"role": "user", "content": "what is your favorite ice cream flavor, be honest. I know even you have a preference"}
    ])";

    std::string mtp_options = "{"
        "\"temperature\":0,"
        "\"top_p\":0,"
        "\"max_tokens\":24,"
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_cache_invariant_check\":true,"
        "\"mtp_max_draft_tokens\":2,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> first_ids;
    std::string first_json;
    bool first_ok = run_complete_collect_ids(model, first_messages, mtp_options.c_str(), first_ids, first_json);
    std::string first_response = json_string(first_json, "response");

    std::string second_messages = "["
        "{\"role\":\"user\",\"content\":\"what is your favorite ice cream flavor, be honest. I know even you have a preference\"},"
        "{\"role\":\"assistant\",\"content\":\"" + escape_json(first_response) + "\"},"
        "{\"role\":\"user\",\"content\":\"oooo I like that, and what of your favorite color\"}"
    "]";

    std::vector<uint32_t> second_ids;
    std::string second_json;
    bool second_ok = first_ok && !first_response.empty()
        && run_complete_collect_ids(model, second_messages.c_str(), mtp_options.c_str(), second_ids, second_json);

    cactus_destroy(model);

    return first_ok
        && !first_response.empty()
        && second_ok
        && json_number(second_json, "rounds") > 0.0
        && second_json.find("\"enabled\":true") != std::string::npos;
}

bool test_gemma4_mtp_low_acceptance_stays_enabled() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "user", "content": "Return a JSON object with keys name, role, and status."}
    ])";

    const char* normal_options = R"({
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 64,
        "telemetry_enabled": false
    })";

    std::string mtp_options = "{"
        "\"temperature\":0,"
        "\"top_p\":0,"
        "\"max_tokens\":64,"
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_max_draft_tokens\":2,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> normal_ids;
    std::vector<uint32_t> mtp_ids;
    std::string normal_json;
    std::string mtp_json;

    bool normal_ok = run_complete_collect_ids(model, messages, normal_options, normal_ids, normal_json);
    cactus_reset(model);
    bool mtp_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), mtp_ids, mtp_json);

    cactus_destroy(model);

    return normal_ok
        && mtp_ok
        && normal_ids == mtp_ids
        && json_number(mtp_json, "drafted_tokens") > 0.0
        && mtp_json.find("\"enabled\":true") != std::string::npos
        && mtp_json.find("low_acceptance") == std::string::npos;
}

bool test_gemma4_mtp_seeded_sampling_smoke() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write one creative sentence about desert rain."}
    ])";

    std::string mtp_options = "{"
        "\"temperature\":0.7,"
        "\"top_p\":0.95,"
        "\"top_k\":64,"
        "\"min_p\":0,"
        "\"max_tokens\":16,"
        "\"seed\":1234,"
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_max_draft_tokens\":2,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> first_ids;
    std::vector<uint32_t> second_ids;
    std::string first_json;
    std::string second_json;

    bool first_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), first_ids, first_json);
    cactus_reset(model);
    bool second_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), second_ids, second_json);

    cactus_destroy(model);

    return first_ok
        && second_ok
        && first_ids == second_ids
        && first_json.find("mtp") != std::string::npos;
}

bool test_gemma4_mtp_sampled_matches_manual_reference() {
    if (!has_real_model_env()) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET and CACTUS_TEST_GEMMA4_ASSISTANT are not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target_path(), nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write one creative sentence about desert rain."}
    ])";

    std::string mtp_options = "{"
        "\"temperature\":0.7,"
        "\"top_p\":0.95,"
        "\"top_k\":64,"
        "\"min_p\":0,"
        "\"max_tokens\":16,"
        "\"seed\":1234,"
        "\"telemetry_enabled\":false,"
        "\"mtp\":true,"
        "\"mtp_required\":true,"
        "\"mtp_max_draft_tokens\":3,"
        "\"mtp_assistant_path\":\"" + std::string(assistant_path()) + "\""
    "}";

    std::vector<uint32_t> mtp_ids;
    std::string mtp_json;
    bool mtp_ok = run_complete_collect_ids(model, messages, mtp_options.c_str(), mtp_ids, mtp_json);

    cactus_reset(model);
    std::vector<uint32_t> manual_ids = run_manual_sampled_mtp_reference(
        model, messages, assistant_path(), 16, 3, 1234);

    cactus_destroy(model);

    return mtp_ok
        && mtp_ids == manual_ids
        && json_number(mtp_json, "drafted_tokens") > 0.0
        && mtp_json.find("\"enabled\":true") != std::string::npos;
}

bool test_gemma4_mtp_missing_required_assistant_errors() {
    const char* target = target_path();
    if (!target || std::strlen(target) == 0) {
        std::cout << "Skipping: CACTUS_TEST_GEMMA4_TARGET is not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(target, nullptr, false);
    if (!model) return false;

    const char* messages = R"([
        {"role": "user", "content": "Say hello in three words."}
    ])";

    const char* options = R"({
        "temperature": 0,
        "max_tokens": 8,
        "telemetry_enabled": false,
        "mtp": true,
        "mtp_required": true,
        "mtp_max_draft_tokens": 2,
        "mtp_assistant_path": "/definitely/missing/gemma4-assistant"
    })";

    std::vector<uint32_t> ids;
    std::string json;
    bool ok = run_complete_collect_ids(model, messages, options, ids, json);

    cactus_destroy(model);

    return !ok
        || json.find("assistant") != std::string::npos
        || json.find("mtp") != std::string::npos;
}

int main() {
    TestRunner runner("Gemma 4 MTP Fast Smoke");

    runner.run_test("greedy_matches_standard", test_gemma4_mtp_greedy_matches_standard_decode());
    runner.run_test("short_chat_accepts_drafts", test_gemma4_mtp_short_chat_accepts_drafts());
    runner.run_test("cache_invariant", test_gemma4_mtp_cache_invariant());
    runner.run_test("multiturn_prefix_cache_invariant", test_gemma4_mtp_multiturn_prefix_cache_invariant());
    runner.run_test("low_acceptance_stays_enabled", test_gemma4_mtp_low_acceptance_stays_enabled());
    runner.run_test("seeded_sampling_smoke", test_gemma4_mtp_seeded_sampling_smoke());
    runner.run_test("sampled_manual_reference", test_gemma4_mtp_sampled_matches_manual_reference());
    runner.run_test("missing_required_assistant", test_gemma4_mtp_missing_required_assistant_errors());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
