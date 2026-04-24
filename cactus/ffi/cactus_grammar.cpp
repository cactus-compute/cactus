#include "cactus_ffi.h"
#include "cactus_utils.h"

#include <vector>
#include <optional>

using namespace cactus::ffi;
using namespace cactus::engine;

namespace {

static std::vector<Grammar> collect_grammars(cactus_grammar_t* grammars, size_t num_grammars) {
    std::vector<Grammar> collected;
    collected.reserve(num_grammars);
    for (size_t i = 0; i < num_grammars; ++i) {
        if (!grammars || !grammars[i]) {
            continue;
        }

        auto* handle = static_cast<CactusGrammarHandle*>(grammars[i]);
        collected.push_back(*handle->grammar);
    }
    return collected;
}

static cactus_grammar_t handle_exception(const char* operation, const std::string& error) {
    last_error_message = std::string(operation) + ": " + error;
    CACTUS_LOG_ERROR("grammar", last_error_message);
    return nullptr;
}

static bool handle_bool_exception(const char* operation, const std::string& error) {
    last_error_message = std::string(operation) + ": " + error;
    CACTUS_LOG_ERROR("grammar", last_error_message);
    return false;
}

} // anonymous namespace

extern "C" {

cactus_grammar_json_schema_options_t cactus_grammar_json_schema_default_options() {
    return { true, 2, {",", ":"}, true, 1, };
}

cactus_grammar_t cactus_grammar_init_gbnf(const char* gbnf, const char* start_symbol) {
    if (!gbnf) {
        return handle_exception("cactus_grammar_init_gbnf", "gbnf is null");
    }

    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(Grammar::gbnf(gbnf, start_symbol ? start_symbol : "root"))
        };
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_init_gbnf", e.what());
    }
}

cactus_grammar_t cactus_grammar_init_json() {
    return new CactusGrammarHandle{std::make_unique<Grammar>(Grammar::json())};
}

cactus_grammar_t cactus_grammar_init_empty() {
    return new CactusGrammarHandle{std::make_unique<Grammar>()};
}

cactus_grammar_t cactus_grammar_init_universal() {
    return new CactusGrammarHandle{std::make_unique<Grammar>(Grammar::universal())};
}

cactus_grammar_t cactus_grammar_init_json_schema(
    const char* json_schema,
    cactus_grammar_json_schema_options_t options
) {
    if (!json_schema) {
        return handle_exception("cactus_grammar_init_json_schema", "json_schema is null");
    }

    try {
        if ((!options.any_whitespace) && (!options.separators[0] || !options.separators[1])) {
            return handle_exception(
                "cactus_grammar_init_json_schema",
                "json schema separators must have 2 strings"
            );
        }
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(Grammar::json_schema(
                json_schema,
                options.any_whitespace,
                options.indent,
                {
                    options.separators[0] ? options.separators[0] : ",",
                    options.separators[1] ? options.separators[1] : ":"
                },
                options.strict_mode,
                options.max_whitespace_count
            ))
        };
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_init_json_schema", e.what());
    }
}

cactus_grammar_t cactus_grammar_init_regex(const char* regex) {
    if (!regex) {
        return handle_exception("cactus_grammar_init_regex", "regex is null");
    }

    try {
        return new CactusGrammarHandle{std::make_unique<Grammar>(Grammar::regex(regex))};
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_init_regex", e.what());
    }
}

cactus_grammar_t cactus_grammar_init_structural_tag(const char* structural_tag_json) {
    if (!structural_tag_json) {
        return handle_exception("cactus_grammar_init_structural_tag", "structural_tag_json is null");
    }

    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(Grammar::structural_tag(structural_tag_json))
        };
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_init_structural_tag", e.what());
    }
}

cactus_grammar_t cactus_grammar_union(cactus_grammar_t* grammars, size_t num_grammars) {
    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(Grammar::unite(collect_grammars(grammars, num_grammars)))
        };
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_union", e.what());
    }
}

cactus_grammar_t cactus_grammar_concatenate(cactus_grammar_t* grammars, size_t num_grammars) {
    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(Grammar::concatenate(collect_grammars(grammars, num_grammars)))
        };
    } catch (const std::exception& e) {
        return handle_exception("cactus_grammar_concatenate", e.what());
    }
}

bool cactus_grammar_is_empty(cactus_grammar_t grammar) {
    if (!grammar) {
        return handle_bool_exception("cactus_grammar_is_empty", "grammar is null");
    }

    return static_cast<CactusGrammarHandle*>(grammar)->grammar->is_empty();
}

bool cactus_grammar_is_universal(cactus_grammar_t grammar) {
    if (!grammar) {
        return handle_bool_exception("cactus_grammar_is_universal", "grammar is null");
    }

    return static_cast<CactusGrammarHandle*>(grammar)->grammar->is_universal();
}

void cactus_grammar_destroy(cactus_grammar_t grammar) {
    if (grammar) delete static_cast<CactusGrammarHandle*>(grammar);
}

}
