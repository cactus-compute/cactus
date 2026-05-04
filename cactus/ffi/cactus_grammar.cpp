#include "cactus_ffi.h"
#include "cactus_utils.h"

#include <cstring>
#include <utility>
#include <vector>

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

static int handle_int_exception(const char* operation, const std::string& error) {
    last_error_message = std::string(operation) + ": " + error;
    CACTUS_LOG_ERROR("grammar", last_error_message);
    return -1;
}

template <typename Factory>
static cactus_grammar_t make_grammar(const char* operation, Factory&& factory) {
    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(std::forward<Factory>(factory)())
        };
    } catch (const std::exception& e) {
        return handle_exception(operation, e.what());
    }
}

} // anonymous namespace

extern "C" {

cactus_grammar_json_schema_options_t cactus_grammar_json_schema_default_options() {
    return { true, 2, {",", ":"}, true, 1, };
}

cactus_grammar_t cactus_grammar_init_ebnf(const char* ebnf, const char* start_symbol) {
    if (!ebnf) {
        return handle_exception(__func__, "ebnf is null");
    }

    return make_grammar(__func__, [&] {
        return Grammar::ebnf(ebnf, start_symbol ? start_symbol : "root");
    });
}

cactus_grammar_t cactus_grammar_init_json() {
    return make_grammar(__func__, [] {
        return Grammar::json();
    });
}

cactus_grammar_t cactus_grammar_init_empty() {
    return make_grammar(__func__, [] {
        return Grammar();
    });
}

cactus_grammar_t cactus_grammar_init_universal() {
    return make_grammar(__func__, [] {
        return Grammar::universal();
    });
}

cactus_grammar_t cactus_grammar_init_json_schema(
    const char* json_schema,
    cactus_grammar_json_schema_options_t options
) {
    if (!json_schema) {
        return handle_exception(__func__, "json_schema is null");
    }

    if ((!options.any_whitespace) && (!options.separators[0] || !options.separators[1])) {
        return handle_exception(
            __func__,
            "json schema separators must have 2 strings"
        );
    }

    return make_grammar(__func__, [&] {
        return Grammar::json_schema(
            json_schema,
            options.any_whitespace,
            options.indent,
            {
                options.separators[0] ? options.separators[0] : ",",
                options.separators[1] ? options.separators[1] : ":"
            },
            options.strict_mode,
            options.max_whitespace_count
        );
    });
}

cactus_grammar_t cactus_grammar_init_regex(const char* regex) {
    if (!regex) {
        return handle_exception(__func__, "regex is null");
    }

    return make_grammar(__func__, [&] {
        return Grammar::regex(regex);
    });
}

cactus_grammar_t cactus_grammar_init_structural_tag(const char* structural_tag_json) {
    if (!structural_tag_json) {
        return handle_exception(__func__, "structural_tag_json is null");
    }

    return make_grammar(__func__, [&] {
        return Grammar::structural_tag(structural_tag_json);
    });
}

cactus_grammar_t cactus_grammar_union(cactus_grammar_t* grammars, size_t num_grammars) {
    return make_grammar(__func__, [&] {
        return Grammar::unite(collect_grammars(grammars, num_grammars));
    });
}

cactus_grammar_t cactus_grammar_concatenate(cactus_grammar_t* grammars, size_t num_grammars) {
    return make_grammar(__func__, [&] {
        return Grammar::concatenate(collect_grammars(grammars, num_grammars));
    });
}

int cactus_grammar_get_ebnf(cactus_grammar_t grammar, char* buffer, size_t buffer_size) {
    if (!grammar) {
        return handle_int_exception(__func__, "grammar is null");
    }
    if (!buffer || buffer_size == 0) {
        return handle_int_exception(__func__, "buffer is null or buffer_size is 0");
    }

    try {
        std::string ebnf = static_cast<CactusGrammarHandle*>(grammar)->grammar->ebnf();
        if (ebnf.size() >= buffer_size) {
            return handle_int_exception(__func__, "buffer too small");
        }
        std::strcpy(buffer, ebnf.c_str());
        return 0;
    } catch (const std::exception& e) {
        return handle_int_exception(__func__, e.what());
    }
}

bool cactus_grammar_is_empty(cactus_grammar_t grammar) {
    if (!grammar) {
        return handle_bool_exception(__func__, "grammar is null");
    }

    return static_cast<CactusGrammarHandle*>(grammar)->grammar->is_empty();
}

bool cactus_grammar_is_universal(cactus_grammar_t grammar) {
    if (!grammar) {
        return handle_bool_exception(__func__, "grammar is null");
    }

    return static_cast<CactusGrammarHandle*>(grammar)->grammar->is_universal();
}

void cactus_grammar_destroy(cactus_grammar_t grammar) {
    if (grammar) delete static_cast<CactusGrammarHandle*>(grammar);
}

}
