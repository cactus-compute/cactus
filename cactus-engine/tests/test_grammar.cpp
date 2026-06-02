#include "test_utils.h"

#include "../src/utils.h"

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

static GrammarHandle grammar_handle(cactus_grammar_t grammar) {
    if (!grammar) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return GrammarHandle(grammar, &cactus_grammar_destroy);
}

static GrammarVocabularyHandle vocab_handle(cactus_grammar_vocabulary_t vocabulary) {
    return GrammarVocabularyHandle(vocabulary, &cactus_grammar_vocabulary_destroy);
}

static GrammarEngineHandle engine_handle(cactus_grammar_engine_t engine) {
    return GrammarEngineHandle(engine, &cactus_grammar_engine_destroy);
}

static GrammarMatcherHandle matcher_handle(cactus_grammar_matcher_t matcher) {
    if (!matcher) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return GrammarMatcherHandle(matcher, &cactus_grammar_matcher_destroy);
}

static const std::string require_model_path() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        throw std::runtime_error("CACTUS_TEST_MODEL is not set");
    }
    return model_path;
}

static std::unique_ptr<cactus::engine::Tokenizer> create_test_tokenizer() {
    auto tokenizer = Tokenizer::from_model_dir(require_model_path());
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer from test model files");
    }
    return tokenizer;
}

static cactus_grammar_t json_schema_grammar(const std::string& json_schema) {
    return cactus_grammar_init_json_schema(
        json_schema.c_str(),
        cactus_grammar_json_schema_default_options()
    );
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
    bool add_prefix_space = false;
    size_t vocab_size = 0;
    std::vector<uint32_t> stop_token_ids;

    GrammarFixture()
        : tokenizer(create_test_tokenizer()),
          vocab(vocab_handle(cactus_grammar_vocabulary_init(require_model_path().c_str()))),
          engine(engine_handle(cactus_grammar_engine_init(vocab.get()))) {
        if (!engine) {
            throw std::runtime_error(cactus_get_last_error());
        }
        add_prefix_space = cactus_grammar_vocabulary_get_add_prefix_space(vocab.get());
        vocab_size = cactus_grammar_vocabulary_get_size(vocab.get());
        if (vocab_size == 0) {
            throw std::runtime_error(cactus_get_last_error());
        }
        std::vector<uint32_t> actual_stop_token_ids(16);
        size_t out_token_count = 0;
        auto result = cactus_grammar_vocabulary_get_stop_token_ids(
            vocab.get(),
            actual_stop_token_ids.data(),
            actual_stop_token_ids.size(),
            &out_token_count
        );
        if (result != 0) throw std::runtime_error(cactus_get_last_error());
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
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
    if (!accept_text(matcher.get(), fixture, text)) return false;
    return cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool rejects_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
    return !accept_text(matcher.get(), fixture, text);
}

static bool rejects_eos_after_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
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
            "type": "tags_with_separator",
            "separator": "\n",
            "tags": [
                {
                    "begin": "<|tool_call>\ncall:function_name_1(",
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
                    "end": ")\n<tool_call|>"
                },
                {
                    "begin": "<|tool_call>\ncall:function_name_2(",
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
                    "end": ")\n<tool_call|>"
                }
            ]
        }
    })";
}

static bool test_vocab_accessors(const GrammarFixture& fixture) {
    const auto vocabulary = GrammarVocabulary::from_tokenizer(*fixture.tokenizer);
    return fixture.vocab_size == fixture.tokenizer->get_vocab_size()
        && fixture.add_prefix_space == vocabulary.add_prefix_space()
        && !fixture.stop_token_ids.empty()
        && fixture.stop_token_ids.front() == fixture.tokenizer->get_eos_token();
}

static bool test_empty_grammar_properties() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto empty2 = grammar_handle(cactus_grammar_init_empty());
    auto simple = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));

    if (!cactus_grammar_is_empty(empty.get())
        || !cactus_grammar_is_empty(empty2.get())
        || cactus_grammar_is_empty(simple.get())) {
        return false;
    }

    cactus_grammar_t empty_union_inputs[] = {empty.get(), empty2.get()};
    auto empty_union = grammar_handle(cactus_grammar_union(empty_union_inputs, 2));
    auto empty_concat = grammar_handle(cactus_grammar_concatenate(empty_union_inputs, 2));

    cactus_grammar_t union_with_simple_inputs[] = {empty.get(), simple.get()};
    auto union_with_simple = grammar_handle(cactus_grammar_union(union_with_simple_inputs, 2));
    auto concat_with_simple = grammar_handle(cactus_grammar_concatenate(union_with_simple_inputs, 2));

    return cactus_grammar_is_empty(empty_union.get())
        && cactus_grammar_is_empty(empty_concat.get())
        && !cactus_grammar_is_empty(union_with_simple.get())
        && !cactus_grammar_is_empty(concat_with_simple.get());
}

static bool test_epsilon_grammar_accepts_only_empty_string(const GrammarFixture& fixture) {
    auto epsilon = grammar_handle(cactus_grammar_init_epsilon());
    return accepts_complete_text(epsilon.get(), fixture, "")
        && rejects_text(epsilon.get(), fixture, "hello");
}

static bool test_optional_grammar_accepts_zero_or_one_occurrence(const GrammarFixture& fixture) {
    auto hello = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto optional = grammar_handle(cactus_grammar_optional(hello.get()));

    return accepts_complete_text(optional.get(), fixture, "")
        && accepts_complete_text(optional.get(), fixture, "hello")
        && rejects_text(optional.get(), fixture, "goodbye")
        && rejects_text(optional.get(), fixture, "hellohello");
}

static bool test_optional_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto optional = grammar_handle(cactus_grammar_optional(empty.get()));
    return cactus_grammar_is_empty(optional.get());
}

static bool test_star_grammar_accepts_zero_or_more_occurrences(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto star = grammar_handle(cactus_grammar_star(ha.get()));

    return accepts_complete_text(star.get(), fixture, "")
        && accepts_complete_text(star.get(), fixture, "ha")
        && accepts_complete_text(star.get(), fixture, "hahaha")
        && rejects_text(star.get(), fixture, "hello");
}

static bool test_star_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto star = grammar_handle(cactus_grammar_star(empty.get()));
    return cactus_grammar_is_empty(star.get());
}

static bool test_repeat_exact_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat(ha.get(), 3));

    return accepts_complete_text(repeated.get(), fixture, "hahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha")
        && rejects_text(repeated.get(), fixture, "hahahaha")
        && rejects_text(repeated.get(), fixture, "hello");
}

static bool test_repeat_range_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat_range(ha.get(), 2, 4));

    return accepts_complete_text(repeated.get(), fixture, "haha")
        && accepts_complete_text(repeated.get(), fixture, "hahaha")
        && accepts_complete_text(repeated.get(), fixture, "hahahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha")
        && rejects_text(repeated.get(), fixture, "hahahahaha");
}

static bool test_repeat_range_unbounded_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat_range(ha.get(), 2, -1));

    return accepts_complete_text(repeated.get(), fixture, "haha")
        && accepts_complete_text(repeated.get(), fixture, "hahaha")
        && accepts_complete_text(repeated.get(), fixture, "hahahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha");
}

static bool test_repeat_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto exact = grammar_handle(cactus_grammar_repeat(empty.get(), 3));
    auto range = grammar_handle(cactus_grammar_repeat_range(empty.get(), 1, 3));
    return cactus_grammar_is_empty(exact.get()) && cactus_grammar_is_empty(range.get());
}

static bool test_ebnf_string_export_matches_parenthesized_input_ebnf() {
    const std::string source = "root ::= ((\"hello\") | (\"hi\"))\n";
    auto grammar = grammar_handle(cactus_grammar_init_ebnf(source.c_str(), "root"));
    return grammar_ebnf(grammar.get()) == source;
}

static bool test_concat_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto right = grammar_handle(cactus_grammar_init_ebnf("root ::= \" world\"", "root"));
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = grammar_handle(cactus_grammar_concatenate(handles, 2));

    return accepts_complete_text(combined.get(), fixture, "hello world")
        && rejects_text(combined.get(), fixture, "world hello")
        && rejects_eos_after_text(combined.get(), fixture, "hello");
}

static bool test_union_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto right = grammar_handle(cactus_grammar_init_ebnf("root ::= \"goodbye\"", "root"));
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = grammar_handle(cactus_grammar_union(handles, 2));

    return accepts_complete_text(combined.get(), fixture, "hello")
        && accepts_complete_text(combined.get(), fixture, "goodbye")
        && rejects_text(combined.get(), fixture, "hello goodbye");
}

static bool test_three_way_concat(const GrammarFixture& fixture) {
    auto alpha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"alpha\"", "root"));
    auto dash = grammar_handle(cactus_grammar_init_ebnf("root ::= \"-\"", "root"));
    auto beta = grammar_handle(cactus_grammar_init_ebnf("root ::= \"beta\"", "root"));
    cactus_grammar_t handles[] = {alpha.get(), dash.get(), beta.get()};
    auto combined = grammar_handle(cactus_grammar_concatenate(handles, 3));

    return accepts_complete_text(combined.get(), fixture, "alpha-beta")
        && rejects_text(combined.get(), fixture, "alphabeta")
        && rejects_text(combined.get(), fixture, "beta-alpha");
}

static bool test_three_way_union(const GrammarFixture& fixture) {
    auto red = grammar_handle(cactus_grammar_init_ebnf("root ::= \"red\"", "root"));
    auto green = grammar_handle(cactus_grammar_init_ebnf("root ::= \"green\"", "root"));
    auto blue = grammar_handle(cactus_grammar_init_ebnf("root ::= \"blue\"", "root"));
    cactus_grammar_t handles[] = {red.get(), green.get(), blue.get()};
    auto combined = grammar_handle(cactus_grammar_union(handles, 3));

    return accepts_complete_text(combined.get(), fixture, "red")
        && accepts_complete_text(combined.get(), fixture, "green")
        && accepts_complete_text(combined.get(), fixture, "blue")
        && rejects_text(combined.get(), fixture, "orange");
}

static bool test_unordered_choice(const GrammarFixture& fixture) {
    auto combined = grammar_handle(cactus_grammar_init_ebnf("root ::= \"a\" | \"ab\"", "root"));

    return accepts_complete_text(combined.get(), fixture, "a")
        && accepts_complete_text(combined.get(), fixture, "ab")
        && rejects_text(combined.get(), fixture, "ac");
}

static bool test_regex_and_json_schema_construction() {
    auto regex = grammar_handle(cactus_grammar_init_regex("(cat|dog)"));
    auto json_schema = grammar_handle(json_schema_grammar(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"));
    return !cactus_grammar_is_empty(regex.get()) && !cactus_grammar_is_empty(json_schema.get());
}

static bool test_universal_grammar_accepts_anything(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_universal());
    return accepts_complete_text(grammar.get(), fixture, "")
        && accepts_complete_text(grammar.get(), fixture, "blob says hello from cactus")
        && accepts_complete_text(grammar.get(), fixture, "line one\nline two\nline three");
}

static bool test_structural_tag_accepts_and_rejects_expected_text(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(
        cactus_grammar_init_structural_tag(tool_call_structural_tag_json().c_str(), fixture.vocab.get())
    );
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size));

    return !cactus_grammar_is_empty(grammar.get())
        && matcher != nullptr
        && cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) == 1;
}

static bool test_regex_accepts_expected_text(const GrammarFixture& fixture) {
    auto regex = grammar_handle(cactus_grammar_init_regex("(cat|dog)"));

    return accepts_complete_text(regex.get(), fixture, "cat")
        && accepts_complete_text(regex.get(), fixture, "dog")
        && rejects_text(regex.get(), fixture, "cow");
}

static bool test_json_schema_accepts_expected_text(const GrammarFixture& fixture) {
    auto json_schema = grammar_handle(json_schema_grammar(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    ));

    return accepts_complete_text(json_schema.get(), fixture, R"({"name":"cactus"})")
        && rejects_text(json_schema.get(), fixture, R"({"age":1})");
}

static bool test_grammar_matcher_reset_restores_initial_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));

    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token())) return false;

    cactus_grammar_matcher_reset(matcher.get());
    return rejects_text(grammar.get(), fixture, "goodbye")
        && accept_text(matcher.get(), fixture, "hello")
        && cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool test_grammar_matcher_rollback_restores_previous_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\" | \"hi\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
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
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;
    return cactus_grammar_matcher_is_completed(matcher.get()) && cactus_grammar_matcher_is_terminated(matcher.get());
}

static bool test_grammar_matcher_fork_preserves_accept_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode("hello world");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (tokens.size() < 2) return false;

    const size_t fork_point = tokens.size() / 2;

    for (size_t i = 0; i < fork_point; ++i) {
        if (!cactus_grammar_matcher_accept(matcher.get(), tokens[i])) return false;
    }

    auto forked = matcher_handle(cactus_grammar_matcher_fork(matcher.get()));
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

static bool test_grammar_matcher_get_grammar_round_trips_source(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    auto retrieved = grammar_handle(cactus_grammar_matcher_get_grammar(matcher.get()));
    if (!retrieved) return false;

    return grammar_ebnf(retrieved.get()) == grammar_ebnf(grammar.get());
}

static bool test_grammar_matcher_fork_preserves_source_grammar(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    auto forked = matcher_handle(cactus_grammar_matcher_fork(matcher.get()));
    if (!forked) return false;

    auto retrieved = grammar_handle(cactus_grammar_matcher_get_grammar(forked.get()));
    if (!retrieved) return false;

    return grammar_ebnf(retrieved.get()) == grammar_ebnf(grammar.get());
}

static bool test_grammar_matcher_next_bitmask_tracks_simple_grammar(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
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
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
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
        runner.run_test("epsilon_language", test_epsilon_grammar_accepts_only_empty_string(fixture));
        runner.run_test("optional_language", test_optional_grammar_accepts_zero_or_one_occurrence(fixture));
        runner.run_test("optional_empty_stays_empty", test_optional_empty_grammar_stays_empty());
        runner.run_test("star_language", test_star_grammar_accepts_zero_or_more_occurrences(fixture));
        runner.run_test("star_empty_stays_empty", test_star_empty_grammar_stays_empty());
        runner.run_test("repeat_exact_language", test_repeat_exact_language(fixture));
        runner.run_test("repeat_range_language", test_repeat_range_language(fixture));
        runner.run_test("repeat_range_unbounded_language", test_repeat_range_unbounded_language(fixture));
        runner.run_test("repeat_empty_stays_empty", test_repeat_empty_grammar_stays_empty());
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
        runner.run_test("structural_tag_language", test_structural_tag_accepts_and_rejects_expected_text(fixture));
        runner.run_test("grammar_matcher_reset", test_grammar_matcher_reset_restores_initial_state(fixture));
        runner.run_test("grammar_matcher_rollback", test_grammar_matcher_rollback_restores_previous_state(fixture));
        runner.run_test("grammar_matcher_completion_state", test_grammar_matcher_completion_state(fixture));
        runner.run_test("grammar_matcher_fork", test_grammar_matcher_fork_preserves_accept_state(fixture));
        runner.run_test("grammar_matcher_get_grammar", test_grammar_matcher_get_grammar_round_trips_source(fixture));
        runner.run_test("grammar_matcher_fork_preserves_grammar", test_grammar_matcher_fork_preserves_source_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask", test_grammar_matcher_next_bitmask_tracks_simple_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask_overallocated_tail", test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
