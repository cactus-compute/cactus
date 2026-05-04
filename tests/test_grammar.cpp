#include "test_utils.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

using namespace cactus::engine;

namespace {

struct GrammarFixture {
    std::unique_ptr<cactus::engine::Tokenizer> tokenizer;
    GrammarVocabulary vocab;

    GrammarFixture() : vocab() {
        const char* model_path = std::getenv("CACTUS_TEST_MODEL");
        if (!model_path) {
            throw std::runtime_error("CACTUS_TEST_MODEL is not set");
        }

        tokenizer = create_tokenizer_from_model_dir(model_path);
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer from test model files");
        }
        vocab = tokenizer->get_grammar_vocabulary();
    }
};

static bool accept_text(GrammarMatcher& matcher, const GrammarFixture& fixture, const std::string& text) {
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode(text);
    if (tokens.empty() && !text.empty()) {
        return false;
    }
    for (uint32_t token : tokens) {
        if (!matcher.accept(token, false)) {
            return false;
        }
    }
    return true;
}

static bool accepts_complete_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.vocab);
    if (!accept_text(matcher, fixture, text)) {
        return false;
    }
    return matcher.accept(fixture.tokenizer->get_eos_token(), false);
}

static bool rejects_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.vocab);
    return !accept_text(matcher, fixture, text);
}

static bool rejects_eos_after_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.vocab);
    if (!accept_text(matcher, fixture, text)) {
        return true;
    }
    return !matcher.accept(fixture.tokenizer->get_eos_token(), false);
}

static bool bitmask_allows_token(const std::vector<int32_t>& bitmask, uint32_t token_id) {
    const size_t word_index = token_id / 32;
    const uint32_t bit_index = token_id % 32;
    if (word_index >= bitmask.size()) return false;
    return ((bitmask[word_index]) & (uint32_t{1} << bit_index)) != 0;
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

static bool test_empty_grammar_properties() {
    Grammar empty;
    Grammar empty2;
    Grammar simple = Grammar::gbnf("root ::= \"hello\"");

    if (!empty.is_empty() || !empty2.is_empty() || simple.is_empty()) {
        return false;
    }

    if (empty.is_universal() || empty2.is_universal() || simple.is_universal()) {
        return false;
    }

    Grammar empty_union = Grammar::unite({empty, empty2});
    Grammar empty_concat = Grammar::concatenate({empty, empty2});
    Grammar union_with_simple = Grammar::unite({empty, simple});
    Grammar concat_with_simple = Grammar::concatenate({empty, simple});

    return empty_union.is_empty()
        && empty_concat.is_empty()
        && !union_with_simple.is_empty()
        && !concat_with_simple.is_empty();
}

static bool test_concat_accepts_expected_language(const GrammarFixture& fixture) {
    Grammar left = Grammar::gbnf("root ::= \"hello\"");
    Grammar right = Grammar::gbnf("root ::= \" world\"");
    Grammar combined = Grammar::concatenate({left, right});

    return accepts_complete_text(combined, fixture, "hello world")
        && rejects_text(combined, fixture, "world hello")
        && rejects_eos_after_text(combined, fixture, "hello");
}

static bool test_union_accepts_expected_language(const GrammarFixture& fixture) {
    Grammar left = Grammar::gbnf("root ::= \"hello\"");
    Grammar right = Grammar::gbnf("root ::= \"goodbye\"");
    Grammar combined = Grammar::unite({left, right});

    return accepts_complete_text(combined, fixture, "hello")
        && accepts_complete_text(combined, fixture, "goodbye")
        && rejects_text(combined, fixture, "hello goodbye");
}

static bool test_three_way_concat(const GrammarFixture& fixture) {
    Grammar combined = Grammar::concatenate({
        Grammar::gbnf("root ::= \"alpha\""),
        Grammar::gbnf("root ::= \"-\""),
        Grammar::gbnf("root ::= \"beta\"")
    });

    return accepts_complete_text(combined, fixture, "alpha-beta")
        && rejects_text(combined, fixture, "alphabeta")
        && rejects_text(combined, fixture, "beta-alpha");
}

static bool test_three_way_union(const GrammarFixture& fixture) {
    Grammar combined = Grammar::unite({
        Grammar::gbnf("root ::= \"red\""),
        Grammar::gbnf("root ::= \"green\""),
        Grammar::gbnf("root ::= \"blue\"")
    });

    return accepts_complete_text(combined, fixture, "red")
        && accepts_complete_text(combined, fixture, "green")
        && accepts_complete_text(combined, fixture, "blue")
        && rejects_text(combined, fixture, "orange");
}

static bool test_unordered_choice(const GrammarFixture& fixture) {
    Grammar combined = Grammar::gbnf("root ::= \"a\" | \"ab\"");

    return accepts_complete_text(combined, fixture, "a")
        && accepts_complete_text(combined, fixture, "ab")
        && rejects_text(combined, fixture, "ac");
}

static bool test_regex_and_json_schema_construction() {
    Grammar regex = Grammar::regex("(cat|dog)");
    Grammar json_schema = Grammar::json_schema(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})");
    return !regex.is_empty() && !json_schema.is_empty();
}

static bool test_universal_grammar_accepts_anything(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::universal();
    return grammar.is_universal()
        && accepts_complete_text(grammar, fixture, "")
        && accepts_complete_text(grammar, fixture, "blob says hello from cactus")
        && accepts_complete_text(grammar, fixture, "line one\nline two\nline three");
}

static bool test_union_with_universal_returns_universal() {
    Grammar universal = Grammar::universal();
    Grammar specific = Grammar::gbnf("root ::= \"hello\"");
    return Grammar::unite({specific, universal}).is_universal();
}

static bool test_structural_tag_accepts_and_rejects_expected_text(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::structural_tag(tool_call_structural_tag_json());

    return accepts_complete_text(
        grammar,
        fixture,
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
        && accepts_complete_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_3\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_eos_after_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":1}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        );
}

static bool test_regex_accepts_expected_text(const GrammarFixture& fixture) {
    Grammar regex = Grammar::regex("(cat|dog)");

    return accepts_complete_text(regex, fixture, "cat")
        && accepts_complete_text(regex, fixture, "dog")
        && rejects_text(regex, fixture, "cow");
}

static bool test_json_schema_accepts_expected_text(const GrammarFixture& fixture) {
    Grammar json_schema = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );

    return accepts_complete_text(json_schema, fixture, R"({"name":"cactus"})")
        && rejects_text(json_schema, fixture, R"({"age":1})");
}

static bool test_grammar_matcher_reset_restores_initial_state(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::gbnf("root ::= \"hello\"");
    GrammarMatcher matcher(&grammar, fixture.vocab);

    if (!accept_text(matcher, fixture, "hello")) {
        return false;
    }
    if (!matcher.accept(fixture.tokenizer->get_eos_token(), false)) {
        return false;
    }

    matcher.reset();
    return rejects_text(grammar, fixture, "goodbye")
        && accept_text(matcher, fixture, "hello")
        && matcher.accept(fixture.tokenizer->get_eos_token(), false);
}

static bool test_grammar_matcher_next_bitmask_tracks_simple_grammar(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::gbnf("root ::= \"hello\"");
    GrammarMatcher matcher(&grammar, fixture.vocab);
    const std::vector<uint32_t> hello_tokens = fixture.tokenizer->encode("hello");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (hello_tokens.empty()) return false;

    std::vector<int32_t> bitmask;
    if (!matcher.next_bitmask(bitmask, fixture.vocab.vocab_size)) return false;
    if (!bitmask_allows_token(bitmask, hello_tokens.front())) return false;
    if (bitmask_allows_token(bitmask, eos_token)) return false;

    for (size_t i = 0; i < hello_tokens.size(); ++i) {
        if (!matcher.next_bitmask(bitmask, fixture.vocab.vocab_size)) return false;
        if (!bitmask_allows_token(bitmask, hello_tokens[i])) return false;
        if (!matcher.accept(hello_tokens[i], false)) return false;
    }

    if (!matcher.next_bitmask(bitmask, fixture.vocab.vocab_size)) return false;
    if (!bitmask_allows_token(bitmask, eos_token)) return false;
    return !bitmask_allows_token(bitmask, hello_tokens.front());
}

static bool test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::gbnf("root ::= \"hello\"");
    GrammarMatcher matcher(&grammar, fixture.vocab);
    std::vector<int32_t> bitmask;

    if (!matcher.next_bitmask(bitmask, fixture.vocab.vocab_size + 1)) return false;
    return (bitmask.back() & 0xFF000000u) == 0;
}


} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Grammar Tests");

    runner.run_test("empty_properties", test_empty_grammar_properties());
    runner.run_test("regex_json_schema_init", test_regex_and_json_schema_construction());

    try {
        GrammarFixture fixture;
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
        runner.run_test("grammar_matcher_next_bitmask", test_grammar_matcher_next_bitmask_tracks_simple_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask_overallocated_tail", test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
