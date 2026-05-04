#include "cactus_ffi.h"
#include "cactus_utils.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

using namespace cactus::ffi;
using namespace cactus::engine;

namespace cactus {
namespace ffi {

struct CactusGrammarHandle {
    std::unique_ptr<cactus::engine::Grammar> grammar;
};

struct CactusGrammarVocabularyHandle {
    std::unique_ptr<cactus::engine::GrammarVocabulary> vocabulary;
};

struct CactusGrammarEngineHandle {
    std::unique_ptr<cactus::engine::GrammarEngine> engine;
};

struct CactusGrammarMatcherHandle {
    std::unique_ptr<cactus::engine::GrammarMatcher> matcher;
};

cactus_grammar_vocabulary_t make_grammar_vocabulary_from_tokenizer(const cactus::engine::Tokenizer& tokenizer) {
    return new CactusGrammarVocabularyHandle{
        std::make_unique<GrammarVocabulary>(tokenizer.get_grammar_vocabulary())
    };
}

} // namespace ffi
} // namespace cactus

namespace {

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

template <typename Handle, typename RawHandle>
static Handle* require_handle(const char* operation, RawHandle handle, const char* label) {
    if (!handle) {
        handle_exception(operation, std::string(label) + " is null");
        return nullptr;
    }
    return static_cast<Handle*>(handle);
}

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

static CactusGrammarVocabularyHandle* require_vocabulary_handle(
    const char* operation,
    cactus_grammar_vocabulary_t vocabulary
) {
    return require_handle<CactusGrammarVocabularyHandle>(operation, vocabulary, "grammar vocabulary");
}

static CactusGrammarEngineHandle* require_engine_handle(
    const char* operation,
    cactus_grammar_engine_t engine
) {
    return require_handle<CactusGrammarEngineHandle>(operation, engine, "grammar engine");
}

static CactusGrammarMatcherHandle* require_matcher_handle(
    const char* operation,
    cactus_grammar_matcher_t matcher
) {
    return require_handle<CactusGrammarMatcherHandle>(operation, matcher, "grammar matcher");
}

static CactusGrammarHandle* require_grammar_handle(const char* operation, cactus_grammar_t grammar) {
    return require_handle<CactusGrammarHandle>(operation, grammar, "grammar");
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

cactus_grammar_vocabulary_t cactus_grammar_vocabulary_init(cactus_model_t model) {
    if (!model) {
        return handle_exception(__func__, "model is null");
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model ? handle->model->get_tokenizer() : nullptr;
        if (!tokenizer) {
            return handle_exception(__func__, "model tokenizer is null");
        }
        return new CactusGrammarVocabularyHandle{
            std::make_unique<GrammarVocabulary>(tokenizer->get_grammar_vocabulary())
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

size_t cactus_grammar_vocabulary_get_size(cactus_grammar_vocabulary_t vocabulary) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return 0;

    return handle->vocabulary->vocab_size();
}

int cactus_grammar_vocabulary_get_stop_token_ids(
    cactus_grammar_vocabulary_t vocabulary,
    uint32_t* buffer,
    size_t buffer_size,
    size_t* out_token_count
) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return -1;
    if (!out_token_count) {
        return handle_int_exception(__func__, "out_token_count is null");
    }

    const auto& stop_token_ids = handle->vocabulary->stop_token_ids();
    *out_token_count = stop_token_ids.size();
    if (!buffer) {
        return handle_int_exception(__func__, "buffer is null");
    }
    if (buffer_size < stop_token_ids.size()) {
        return handle_int_exception(__func__, "buffer too small");
    }

    std::copy(stop_token_ids.begin(), stop_token_ids.end(), buffer);
    return 0;
}

void cactus_grammar_vocabulary_destroy(cactus_grammar_vocabulary_t vocabulary) {
    if (vocabulary) delete static_cast<CactusGrammarVocabularyHandle*>(vocabulary);
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

cactus_grammar_t cactus_grammar_init_structural_tag(
    const char* structural_tag_json,
    cactus_grammar_vocabulary_t vocabulary
) {
    if (!structural_tag_json) {
        return handle_exception(__func__, "structural_tag_json is null");
    }

    return make_grammar(__func__, [&] {
        auto* handle = static_cast<CactusGrammarVocabularyHandle*>(vocabulary);
        return Grammar::structural_tag(
            structural_tag_json,
            handle ? handle->vocabulary.get() : nullptr
        );
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

cactus_grammar_engine_t cactus_grammar_engine_init(cactus_grammar_vocabulary_t vocabulary) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return nullptr;

    try {
        return new CactusGrammarEngineHandle{
            std::make_unique<GrammarEngine>(*handle->vocabulary)
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

void cactus_grammar_engine_destroy(cactus_grammar_engine_t engine) {
    if (engine) delete static_cast<CactusGrammarEngineHandle*>(engine);
}

cactus_grammar_matcher_t cactus_grammar_engine_compile_matcher(
    cactus_grammar_engine_t engine,
    cactus_grammar_t grammar
) {
    auto* engine_handle = require_engine_handle(__func__, engine);
    if (!engine_handle) return nullptr;
    auto* grammar_handle = require_grammar_handle(__func__, grammar);
    if (!grammar_handle) return nullptr;

    try {
        return new CactusGrammarMatcherHandle{
            std::make_unique<GrammarMatcher>(
                engine_handle->engine->compile_matcher(*grammar_handle->grammar)
            )
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

void cactus_grammar_matcher_destroy(cactus_grammar_matcher_t matcher) {
    if (matcher) delete static_cast<CactusGrammarMatcherHandle*>(matcher);
}

void cactus_grammar_matcher_reset(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return;
    handle->matcher->reset();
}

void cactus_grammar_matcher_rollback(cactus_grammar_matcher_t matcher, int tokens) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return;
    handle->matcher->rollback(tokens);
}

cactus_grammar_matcher_t cactus_grammar_matcher_fork(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return nullptr;

    try {
        return new CactusGrammarMatcherHandle{
            std::make_unique<GrammarMatcher>(handle->matcher->fork())
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

bool cactus_grammar_matcher_is_completed(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? handle->matcher->is_completed() : false;
}

bool cactus_grammar_matcher_is_terminated(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? handle->matcher->is_terminated() : false;
}

bool cactus_grammar_matcher_accept(cactus_grammar_matcher_t matcher, uint32_t token_id) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? handle->matcher->accept(token_id) : false;
}

int cactus_grammar_matcher_next_bitmask(
    cactus_grammar_matcher_t matcher,
    int32_t* bitmask,
    size_t logits_buffer_size
) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return -1;
    if (!bitmask) {
        return handle_int_exception(__func__, "bitmask is null");
    }

    try {
        std::vector<int32_t> next_bitmask;
        const bool should_apply = handle->matcher->next_bitmask(next_bitmask, logits_buffer_size);
        std::copy(next_bitmask.begin(), next_bitmask.end(), bitmask);
        return should_apply ? 1 : 0;
    } catch (const std::exception& e) {
        return handle_int_exception(__func__, e.what());
    }
}

}
