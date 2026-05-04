#include "test_utils.h"

#include "../cactus/ffi/cactus_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cactus::engine;
using namespace cactus::ffi;

namespace {

using GrammarHandle = std::unique_ptr<void, decltype(&cactus_grammar_destroy)>;
using GrammarVocabularyHandle = std::unique_ptr<void, decltype(&cactus_grammar_vocabulary_destroy)>;
using GrammarEngineHandle = std::unique_ptr<void, decltype(&cactus_grammar_engine_destroy)>;
using GrammarMatcherHandle = std::unique_ptr<void, decltype(&cactus_grammar_matcher_destroy)>;

static std::unique_ptr<cactus::engine::Tokenizer> create_test_tokenizer() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        throw std::runtime_error("CACTUS_TEST_MODEL is not set");
    }

    auto tokenizer = create_tokenizer_from_model_dir(model_path);
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer from test model files");
    }
    return tokenizer;
}

static cactus_grammar_t make_grammar_handle(cactus_grammar_t grammar) {
    if (!grammar) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return grammar;
}

static cactus_grammar_t make_grammar_ebnf(const std::string& ebnf, const char* start_symbol = "root") {
    return make_grammar_handle(cactus_grammar_init_ebnf(ebnf.c_str(), start_symbol));
}

static cactus_grammar_t make_grammar_empty() {
    return make_grammar_handle(cactus_grammar_init_empty());
}

static cactus_grammar_t make_grammar_universal() {
    return make_grammar_handle(cactus_grammar_init_universal());
}

static cactus_grammar_t make_grammar_json_schema(const std::string& json_schema) {
    return make_grammar_handle(
        cactus_grammar_init_json_schema(
            json_schema.c_str(),
            cactus_grammar_json_schema_default_options()
        )
    );
}

static cactus_grammar_t make_grammar_regex(const std::string& regex) {
    return make_grammar_handle(cactus_grammar_init_regex(regex.c_str()));
}

static cactus_grammar_t make_grammar_structural_tag(
    const std::string& structural_tag_json,
    cactus_grammar_vocabulary_t vocabulary = nullptr
) {
    return make_grammar_handle(
        cactus_grammar_init_structural_tag(structural_tag_json.c_str(), vocabulary)
    );
}

static cactus_grammar_vocabulary_t make_vocab_handle_from_tokenizer(const Tokenizer& tokenizer) {
    return make_grammar_vocabulary_from_tokenizer(tokenizer);
}

static cactus_grammar_matcher_t make_matcher(cactus_grammar_t grammar, cactus_grammar_engine_t engine) {
    auto matcher = cactus_grammar_engine_compile_matcher(engine, grammar);
    if (!matcher) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return matcher;
}

struct GrammarFixture {
    std::unique_ptr<cactus::engine::Tokenizer> tokenizer;
    GrammarVocabularyHandle vocab;
    GrammarEngineHandle engine;
    size_t vocab_size = 0;
    std::vector<uint32_t> stop_token_ids;

    GrammarFixture()
        : tokenizer(create_test_tokenizer()),
          vocab(make_vocab_handle_from_tokenizer(*tokenizer), &cactus_grammar_vocabulary_destroy),
          engine(cactus_grammar_engine_init(vocab.get()), &cactus_grammar_engine_destroy) {
        if (!engine) {
            throw std::runtime_error(cactus_get_last_error());
        }
        vocab_size = cactus_grammar_vocabulary_get_size(vocab.get());
        if (vocab_size == 0) {
            throw std::runtime_error(cactus_get_last_error());
        }

        stop_token_ids = tokenizer->get_grammar_vocabulary().stop_token_ids();
        std::vector<uint32_t> actual_stop_token_ids(stop_token_ids.size());
        size_t out_token_count = 0;
        if (cactus_grammar_vocabulary_get_stop_token_ids(
                vocab.get(),
                actual_stop_token_ids.data(),
                actual_stop_token_ids.size(),
                &out_token_count
            ) != 0) {
            throw std::runtime_error(cactus_get_last_error());
        }
        actual_stop_token_ids.resize(out_token_count);
        stop_token_ids = std::move(actual_stop_token_ids);
    }
};

static bool accept_text(cactus_grammar_matcher_t matcher, const GrammarFixture& fixture, const std::string& text) {
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode(text);
    if (tokens.empty() && !text.empty()) return false;
    for (uint32_t token : tokens) {
        if (!cactus_grammar_matcher_accept(matcher, token)) return false;
    }
    return true;
}

static bool accepts_complete_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = GrammarMatcherHandle(make_matcher(grammar, fixture.engine.get()), &cactus_grammar_matcher_destroy);
    if (!accept_text(matcher.get(), fixture, text)) return false;
    return cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool rejects_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = GrammarMatcherHandle(make_matcher(grammar, fixture.engine.get()), &cactus_grammar_matcher_destroy);
    return !accept_text(matcher.get(), fixture, text);
}

static bool rejects_eos_after_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = GrammarMatcherHandle(make_matcher(grammar, fixture.engine.get()), &cactus_grammar_matcher_destroy);
    if (!accept_text(matcher.get(), fixture, text)) return true;
    return !cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool bitmask_allows_token(const std::vector<int32_t>& bitmask, uint32_t token_id) {
    const size_t word_index = token_id / 32;
    const uint32_t bit_index = token_id % 32;
    if (word_index >= bitmask.size()) return false;
    return ((bitmask[word_index]) & (uint32_t{1} << bit_index)) != 0;
}

static size_t bitmask_size(size_t logits_buffer_size) {
    return (logits_buffer_size + 31) / 32;
}

static std::string grammar_ebnf(cactus_grammar_t grammar) {
    std::vector<char> buffer(4096, '\0');
    if (cactus_grammar_get_ebnf(grammar, buffer.data(), buffer.size()) != 0) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return std::string(buffer.data());
}

static std::string tool_call_structural_tag_json() {
    return R"({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<｜tool▁calls▁begin｜>"],
            "tags": [
                {
                    "begin": "<｜tool▁calls▁begin｜>",
                    "end": "<｜tool▁calls▁end｜>",
                    "content": {
                        "type": "tags_with_separator",
                        "separator": "\n",
                        "tags": [
                            {
                                "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n",
                                "content": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "city": {"type": "string"}
                                        },
                                        "required": ["city"],
                                        "additionalProperties": false
                                    }
                                },
                                "end": "\n```<｜tool▁call▁end｜>"
                            },
                            {
                                "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n",
                                "content": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "type": "object",
                                        "properties": {
                                            "count": {"type": "integer"}
                                        },
                                        "required": ["count"],
                                        "additionalProperties": false
                                    }
                                },
                                "end": "\n```<｜tool▁call▁end｜>"
                            }
                        ]
                    }
                }
            ],
            "stop_after_first": true
        }
    })";
}

static bool test_vocab_accessors(const GrammarFixture& fixture) {
    return fixture.vocab_size == fixture.tokenizer->get_vocab_size()
        && !fixture.stop_token_ids.empty()
        && fixture.stop_token_ids.front() == fixture.tokenizer->get_eos_token();
}

static bool test_empty_grammar_properties() {
    auto empty = GrammarHandle(make_grammar_empty(), &cactus_grammar_destroy);
    auto empty2 = GrammarHandle(make_grammar_empty(), &cactus_grammar_destroy);
    auto simple = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);

    if (!cactus_grammar_is_empty(empty.get()) || !cactus_grammar_is_empty(empty2.get()) || cactus_grammar_is_empty(simple.get())) {
        return false;
    }

    if (cactus_grammar_is_universal(empty.get()) || cactus_grammar_is_universal(empty2.get()) || cactus_grammar_is_universal(simple.get())) {
        return false;
    }

    cactus_grammar_t empty_union_inputs[] = {empty.get(), empty2.get()};
    auto empty_union = GrammarHandle(make_grammar_handle(cactus_grammar_union(empty_union_inputs, 2)), &cactus_grammar_destroy);
    auto empty_concat = GrammarHandle(make_grammar_handle(cactus_grammar_concatenate(empty_union_inputs, 2)), &cactus_grammar_destroy);

    cactus_grammar_t union_with_simple_inputs[] = {empty.get(), simple.get()};
    auto union_with_simple = GrammarHandle(make_grammar_handle(cactus_grammar_union(union_with_simple_inputs, 2)), &cactus_grammar_destroy);
    auto concat_with_simple = GrammarHandle(make_grammar_handle(cactus_grammar_concatenate(union_with_simple_inputs, 2)), &cactus_grammar_destroy);

    return cactus_grammar_is_empty(empty_union.get())
        && cactus_grammar_is_empty(empty_concat.get())
        && !cactus_grammar_is_empty(union_with_simple.get())
        && !cactus_grammar_is_empty(concat_with_simple.get());
}

static bool test_ebnf_string_export_matches_parenthesized_input_ebnf() {
    const std::string source = "root ::= ((\"hello\") | (\"hi\"))\n";
    auto grammar = GrammarHandle(make_grammar_ebnf(source), &cactus_grammar_destroy);
    return grammar_ebnf(grammar.get()) == source;
}

static bool test_concat_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto right = GrammarHandle(make_grammar_ebnf("root ::= \" world\""), &cactus_grammar_destroy);
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = GrammarHandle(make_grammar_handle(cactus_grammar_concatenate(handles, 2)), &cactus_grammar_destroy);

    return accepts_complete_text(combined.get(), fixture, "hello world")
        && rejects_text(combined.get(), fixture, "world hello")
        && rejects_eos_after_text(combined.get(), fixture, "hello");
}

static bool test_union_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto right = GrammarHandle(make_grammar_ebnf("root ::= \"goodbye\""), &cactus_grammar_destroy);
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = GrammarHandle(make_grammar_handle(cactus_grammar_union(handles, 2)), &cactus_grammar_destroy);

    return accepts_complete_text(combined.get(), fixture, "hello")
        && accepts_complete_text(combined.get(), fixture, "goodbye")
        && rejects_text(combined.get(), fixture, "hello goodbye");
}

static bool test_three_way_concat(const GrammarFixture& fixture) {
    auto alpha = GrammarHandle(make_grammar_ebnf("root ::= \"alpha\""), &cactus_grammar_destroy);
    auto dash = GrammarHandle(make_grammar_ebnf("root ::= \"-\""), &cactus_grammar_destroy);
    auto beta = GrammarHandle(make_grammar_ebnf("root ::= \"beta\""), &cactus_grammar_destroy);
    cactus_grammar_t handles[] = {alpha.get(), dash.get(), beta.get()};
    auto combined = GrammarHandle(make_grammar_handle(cactus_grammar_concatenate(handles, 3)), &cactus_grammar_destroy);

    return accepts_complete_text(combined.get(), fixture, "alpha-beta")
        && rejects_text(combined.get(), fixture, "alphabeta")
        && rejects_text(combined.get(), fixture, "beta-alpha");
}

static bool test_three_way_union(const GrammarFixture& fixture) {
    auto red = GrammarHandle(make_grammar_ebnf("root ::= \"red\""), &cactus_grammar_destroy);
    auto green = GrammarHandle(make_grammar_ebnf("root ::= \"green\""), &cactus_grammar_destroy);
    auto blue = GrammarHandle(make_grammar_ebnf("root ::= \"blue\""), &cactus_grammar_destroy);
    cactus_grammar_t handles[] = {red.get(), green.get(), blue.get()};
    auto combined = GrammarHandle(make_grammar_handle(cactus_grammar_union(handles, 3)), &cactus_grammar_destroy);

    return accepts_complete_text(combined.get(), fixture, "red")
        && accepts_complete_text(combined.get(), fixture, "green")
        && accepts_complete_text(combined.get(), fixture, "blue")
        && rejects_text(combined.get(), fixture, "orange");
}

static bool test_unordered_choice(const GrammarFixture& fixture) {
    auto combined = GrammarHandle(make_grammar_ebnf("root ::= \"a\" | \"ab\""), &cactus_grammar_destroy);

    return accepts_complete_text(combined.get(), fixture, "a")
        && accepts_complete_text(combined.get(), fixture, "ab")
        && rejects_text(combined.get(), fixture, "ac");
}

static bool test_regex_and_json_schema_construction() {
    auto regex = GrammarHandle(make_grammar_regex("(cat|dog)"), &cactus_grammar_destroy);
    auto json_schema = GrammarHandle(make_grammar_json_schema(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"), &cactus_grammar_destroy);
    return !cactus_grammar_is_empty(regex.get()) && !cactus_grammar_is_empty(json_schema.get());
}

static bool test_universal_grammar_accepts_anything(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_universal(), &cactus_grammar_destroy);
    return cactus_grammar_is_universal(grammar.get())
        && accepts_complete_text(grammar.get(), fixture, "")
        && accepts_complete_text(grammar.get(), fixture, "blob says hello from cactus")
        && accepts_complete_text(grammar.get(), fixture, "line one\nline two\nline three");
}

static bool test_union_with_universal_returns_universal() {
    auto universal = GrammarHandle(make_grammar_universal(), &cactus_grammar_destroy);
    auto specific = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    cactus_grammar_t handles[] = {specific.get(), universal.get()};
    auto combined = GrammarHandle(make_grammar_handle(cactus_grammar_union(handles, 2)), &cactus_grammar_destroy);
    return cactus_grammar_is_universal(combined.get());
}

static bool test_structural_tag_accepts_and_rejects_expected_text(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_structural_tag(tool_call_structural_tag_json(), fixture.vocab.get()), &cactus_grammar_destroy);

    return accepts_complete_text(
        grammar.get(),
        fixture,
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
        && accepts_complete_text(
            grammar.get(),
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar.get(),
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar.get(),
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_3\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_eos_after_text(
            grammar.get(),
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":1}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar.get(),
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        );
}

static bool test_regex_accepts_expected_text(const GrammarFixture& fixture) {
    auto regex = GrammarHandle(make_grammar_regex("(cat|dog)"), &cactus_grammar_destroy);

    return accepts_complete_text(regex.get(), fixture, "cat")
        && accepts_complete_text(regex.get(), fixture, "dog")
        && rejects_text(regex.get(), fixture, "cow");
}

static bool test_json_schema_accepts_expected_text(const GrammarFixture& fixture) {
    auto json_schema = GrammarHandle(make_grammar_json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    ), &cactus_grammar_destroy);

    return accepts_complete_text(json_schema.get(), fixture, R"({"name":"cactus"})")
        && rejects_text(json_schema.get(), fixture, R"({"age":1})");
}

static bool test_grammar_matcher_reset_restores_initial_state(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);

    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token())) return false;

    cactus_grammar_matcher_reset(matcher.get());
    return rejects_text(grammar.get(), fixture, "goodbye")
        && accept_text(matcher.get(), fixture, "hello")
        && cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool test_grammar_matcher_rollback_restores_previous_state(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello\" | \"hi\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);
    const std::vector<uint32_t> hello_tokens = fixture.tokenizer->encode("hello");
    const std::vector<uint32_t> hi_tokens = fixture.tokenizer->encode("hi");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    for (uint32_t token : hello_tokens) {
        if (!cactus_grammar_matcher_accept(matcher.get(), token)) return false;
    }
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;

    cactus_grammar_matcher_rollback(matcher.get(), static_cast<int>(hello_tokens.size()) + 1);

    for (uint32_t token : hi_tokens) {
        if (!cactus_grammar_matcher_accept(matcher.get(), token)) return false;
    }
    return cactus_grammar_matcher_accept(matcher.get(), eos_token);
}

static bool test_grammar_matcher_completion_state(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;
    return cactus_grammar_matcher_is_completed(matcher.get()) && cactus_grammar_matcher_is_terminated(matcher.get());
}

static bool test_grammar_matcher_fork_preserves_accept_state(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello world\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode("hello world");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (tokens.size() < 2) return false;

    const size_t fork_point = tokens.size() / 2;

    for (size_t i = 0; i < fork_point; ++i) {
        if (!cactus_grammar_matcher_accept(matcher.get(), tokens[i])) return false;
    }

    auto forked = GrammarMatcherHandle(cactus_grammar_matcher_fork(matcher.get()), &cactus_grammar_matcher_destroy);
    if (!forked) return false;

    for (size_t i = fork_point; i < tokens.size(); ++i) {
        if (!cactus_grammar_matcher_accept(matcher.get(), tokens[i])) return false;
    }
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;

    for (size_t i = fork_point; i < tokens.size(); ++i) {
        if (!cactus_grammar_matcher_accept(forked.get(), tokens[i])) return false;
    }
    return cactus_grammar_matcher_accept(forked.get(), eos_token);
}

static bool test_grammar_matcher_next_bitmask_tracks_simple_grammar(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);
    const std::vector<uint32_t> hello_tokens = fixture.tokenizer->encode("hello");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (hello_tokens.empty()) return false;

    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size));
    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
    if (!bitmask_allows_token(bitmask, hello_tokens.front())) return false;
    if (bitmask_allows_token(bitmask, eos_token)) return false;

    for (size_t i = 0; i < hello_tokens.size(); ++i) {
        if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
        if (!bitmask_allows_token(bitmask, hello_tokens[i])) return false;
        if (!cactus_grammar_matcher_accept(matcher.get(), hello_tokens[i])) return false;
    }

    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
    if (!bitmask_allows_token(bitmask, eos_token)) return false;
    return !bitmask_allows_token(bitmask, hello_tokens.front());
}

static bool test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(const GrammarFixture& fixture) {
    auto grammar = GrammarHandle(make_grammar_ebnf("root ::= \"hello\""), &cactus_grammar_destroy);
    auto matcher = GrammarMatcherHandle(make_matcher(grammar.get(), fixture.engine.get()), &cactus_grammar_matcher_destroy);
    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size + 1), -1);

    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size + 1) != 1) return false;
    return (bitmask.back() & 0xFF000000u) == 0;
}

} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Grammar Tests");

    try {
        GrammarFixture fixture;
        runner.run_test("vocab_accessors", test_vocab_accessors(fixture));
        runner.run_test("empty_properties", test_empty_grammar_properties());
        runner.run_test("ebnf_export_matches", test_ebnf_string_export_matches_parenthesized_input_ebnf());
        runner.run_test("regex_json_schema_init", test_regex_and_json_schema_construction());
        runner.run_test("concat_language", test_concat_accepts_expected_language(fixture));
        runner.run_test("union_language", test_union_accepts_expected_language(fixture));
        runner.run_test("three_way_concat", test_three_way_concat(fixture));
        runner.run_test("three_way_union", test_three_way_union(fixture));
        runner.run_test("unordred_choice", test_unordered_choice(fixture));
        runner.run_test("regex_language", test_regex_accepts_expected_text(fixture));
        runner.run_test("json_schema_language", test_json_schema_accepts_expected_text(fixture));
        runner.run_test("universal", test_universal_grammar_accepts_anything(fixture));
        runner.run_test("union_with_universal_returns_universal", test_union_with_universal_returns_universal());
        runner.run_test("structural_tag_language", test_structural_tag_accepts_and_rejects_expected_text(fixture));
        runner.run_test("grammar_matcher_reset", test_grammar_matcher_reset_restores_initial_state(fixture));
        runner.run_test("grammar_matcher_rollback", test_grammar_matcher_rollback_restores_previous_state(fixture));
        runner.run_test("grammar_matcher_completion_state", test_grammar_matcher_completion_state(fixture));
        runner.run_test("grammar_matcher_fork", test_grammar_matcher_fork_preserves_accept_state(fixture));
        runner.run_test("grammar_matcher_next_bitmask", test_grammar_matcher_next_bitmask_tracks_simple_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask_overallocated_tail", test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
